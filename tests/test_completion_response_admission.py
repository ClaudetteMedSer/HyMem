"""Received non-stop replies are failures, not retryable HTTP attempts.

All provider responses are synthetic; these tests make no network requests.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from benchmarks import longmemeval_adapter as lme
from benchmarks.lme_protocol import _usage
from benchmarks.strictness import usage_snapshot
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.extraction.llm import LLMRequest


USAGE = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def response(*, finish="stop", content="ok", usage=USAGE):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": deepcopy(usage)}


def sdk_response(value):
    if not isinstance(value, dict):
        return value
    result = SimpleNamespace(**value)
    if "usage" in value and isinstance(value["usage"], dict):
        result.usage = SimpleNamespace(**value["usage"])
    if isinstance(value.get("choices"), list):
        result.choices = []
        for choice in value["choices"]:
            if isinstance(choice, dict):
                converted = SimpleNamespace(**choice)
                if isinstance(choice.get("message"), dict):
                    converted.message = SimpleNamespace(**choice["message"])
                result.choices.append(converted)
            else:
                result.choices.append(choice)
    return result


@pytest.fixture(params=["sdk", "raw"])
def client_factory(request, monkeypatch):
    clients = []
    backend = request.param
    monkeypatch.setattr("hymem.extraction.retry.time.sleep", lambda _delay: None)
    monkeypatch.setattr(lme.time, "sleep", lambda _delay: None)

    def build(responses, **kwargs):
        calls = []
        def reply(wire):
            calls.append(wire)
            value = responses(wire) if callable(responses) else responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        if backend == "sdk":
            import openai
            class Resource:
                def create(self, **wire):
                    return sdk_response(reply(wire))
            class SDK:
                def __init__(self, **options):
                    self._client = options.get("http_client")
                    self.chat = SimpleNamespace(completions=Resource())
                def close(self):
                    self._client.close()
            monkeypatch.setattr(openai, "OpenAI", SDK)
            client = OpenAICompatibleClient(api_key="test-key", model="deepseek-flash")
        else:
            class HTTPResponse:
                def __init__(self, value):
                    self.value = value
                def raise_for_status(self):
                    pass
                def json(self):
                    return self.value
            def post(_url, **options):
                return HTTPResponse(reply(options["json"]))
            monkeypatch.setattr(lme.http, "post", post)
            client = lme.LLMClient("deepseek-flash", "test-key", **kwargs)
        clients.append(client)
        return backend, client, calls

    yield build
    for client in clients:
        client.close()


def invoke(backend, client, *, rejected=False):
    if backend == "sdk":
        if rejected:
            with pytest.raises(RuntimeError):
                client.complete(LLMRequest("system", "user"))
            return
        return client.complete(LLMRequest("system", "user"))
    result = client.chat([{"role": "user", "content": "user"}])
    if rejected:
        assert result.startswith("[LLM_ERROR:")
    else:
        return result


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", "function_call", None, "STOP", "unknown"])
@pytest.mark.parametrize("content", ["yes", '{"accepted":true}', ""])
def test_nonstop_string_rejected_without_retry_and_paid_usage_retained(client_factory, finish, content):
    backend, client, calls = client_factory([response(finish=finish, content=content)])
    invoke(backend, client, rejected=True)
    assert len(calls) == client.request_attempts == 1
    assert client.call_count == client.successful_responses == 0
    assert (client.prompt_tokens, client.completion_tokens, client.total_tokens) == (3, 5, 8)
    snapshot = usage_snapshot(client)
    assert snapshot["token_usage_available"] is True and snapshot["total_tokens"] == 8
    normalized = _usage(snapshot, label="rejected fixture")
    assert (normalized["calls"], normalized["attempts"], normalized["successes"],
            normalized["total_tokens"]) == (0, 1, 0, 8)


@pytest.mark.parametrize("mutation", ["missing_finish", "no_choices", "empty_choices", "bad_choices", "bad_choice", "no_message", "null_content", "number_content"])
def test_received_schema_failure_does_not_retry_or_lose_usage(client_factory, mutation):
    value = response()
    if mutation == "missing_finish":
        del value["choices"][0]["finish_reason"]
    elif mutation == "no_choices":
        del value["choices"]
    elif mutation == "empty_choices":
        value["choices"] = []
    elif mutation == "bad_choices":
        value["choices"] = "not a list"
    elif mutation == "bad_choice":
        value["choices"] = [None]
    elif mutation == "no_message":
        del value["choices"][0]["message"]
    else:
        value["choices"][0]["message"]["content"] = None if mutation == "null_content" else 3
    backend, client, calls = client_factory([value])
    invoke(backend, client, rejected=True)
    assert len(calls) == client.request_attempts == 1
    assert client.call_count == client.successful_responses == 0
    assert usage_snapshot(client)["total_tokens"] == 8


@pytest.mark.parametrize("content", ["", "ok", "not JSON"])
def test_exact_stop_admits_strings_without_new_content_or_json_policy(client_factory, content):
    backend, client, calls = client_factory([response(content=content)])
    assert invoke(backend, client) == content
    assert len(calls) == client.call_count == client.successful_responses == 1
    assert usage_snapshot(client)["total_tokens"] == 8


@pytest.mark.parametrize("invalid", [None, {}, {**USAGE, "total_tokens": True},
    {**USAGE, "total_tokens": float("inf")}, {**USAGE, "total_tokens": float("nan")},
    {**USAGE, "total_tokens": -1}, {**USAGE, "total_tokens": 10**400},
    {**USAGE, "total_tokens": 9}, {**USAGE, "prompt_tokens": 3.5, "total_tokens": 8.5}])
def test_unknown_usage_stays_unknown_after_later_good_reply(client_factory, invalid):
    backend, client, calls = client_factory([response(finish="length", usage=invalid), response()])
    invoke(backend, client, rejected=True)
    assert invoke(backend, client) == "ok"
    assert len(calls) == 2 and client.successful_responses == 1
    assert client.total_tokens == 8  # known subtotal, never invented missing cost
    assert usage_snapshot(client)["token_usage_available"] is False
    assert _usage(usage_snapshot(client), label="unknown fixture")["total_tokens"] is None


def test_transport_retry_preserves_unknown_cost_even_after_known_rejected_reply(client_factory):
    backend, client, calls = client_factory([TimeoutError("synthetic"), response(finish="length")])
    invoke(backend, client, rejected=True)
    assert len(calls) == client.request_attempts == 2
    assert client.successful_responses == 0 and client.total_tokens == 8
    assert usage_snapshot(client)["token_usage_available"] is False


def test_accounting_cannot_be_complete_while_another_attempt_is_in_flight(client_factory):
    started, release = threading.Event(), threading.Event()
    serial_lock = threading.Lock()
    serial = 0
    def reply(_wire):
        nonlocal serial
        with serial_lock:
            serial += 1
            first = serial == 1
        if first:
            started.set()
            assert release.wait(5)
        return response()
    backend, client, calls = client_factory(reply)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(invoke, backend, client)
        assert started.wait(5)
        try:
            assert invoke(backend, client) == "ok"
            assert client.successful_responses == 1 and client.request_attempts == 2
            assert usage_snapshot(client)["token_usage_available"] is False
        finally:
            release.set()
        assert pending.result(5) == "ok"
    assert client.total_tokens == 16 and client.successful_responses == 2
    assert usage_snapshot(client)["token_usage_available"] is True


def test_raw_requested_multiple_choices_preserves_first_choice_policy(client_factory):
    value = response(content="first")
    value["choices"].append(response(finish="length", content="second")["choices"][0])
    backend, client, calls = client_factory([value], n=2)
    if backend == "sdk":
        invoke(backend, client, rejected=True)  # SDK did not request n>1.
    else:
        assert invoke(backend, client) == "first"
        assert calls[0]["n"] == 2
    assert len(calls) == 1 and client.total_tokens == 8


def test_multiple_choices_cannot_rescue_nonstop_first_choice(client_factory):
    value = response(finish="length", content="bad first")
    value["choices"].append(response(content="good second")["choices"][0])
    backend, client, calls = client_factory([value], n=2)
    invoke(backend, client, rejected=True)
    assert len(calls) == 1 and client.successful_responses == 0 and client.total_tokens == 8


def test_received_late_sdk_reply_retains_usage_without_admission_or_retry(client_factory):
    from hymem.deadline import DeadlineExceeded, MonotonicDeadline, use_deadline
    clock = [0.0]
    def late(_wire):
        clock[0] = 2.0
        return response()
    backend, client, calls = client_factory(late)
    if backend != "sdk":
        pytest.skip("raw benchmark deadline ownership is outside this client")
    with use_deadline(MonotonicDeadline(1.0, clock=lambda: clock[0])):
        with pytest.raises(DeadlineExceeded):
            invoke(backend, client)
    assert len(calls) == 1 and client.call_count == client.successful_responses == 0
    assert calls[0]["timeout"] == 1.0
    assert _usage(usage_snapshot(client), label="late fixture")["total_tokens"] == 8


def test_post_response_sdk_integrity_failure_keeps_known_usage(client_factory):
    owner = []
    def drift(_wire):
        owner[0]._client = object()
        return response()
    backend, client, calls = client_factory(drift)
    if backend != "sdk":
        pytest.skip("maintained SDK transport seal only")
    owner.append(client)
    invoke(backend, client, rejected=True)
    assert len(calls) == 1 and client.successful_responses == 0
    assert _usage(usage_snapshot(client), label="integrity fixture")["total_tokens"] == 8


@pytest.mark.parametrize("replacement", ["cleared", "extended"])
def test_late_sdk_reply_cannot_replace_owning_deadline(client_factory, replacement):
    from hymem import deadline as deadline_module
    clock = [0.0]
    def replace_context(_wire):
        clock[0] = 2.0
        deadline_module._CURRENT_DEADLINE.set(
            None if replacement == "cleared" else deadline_module.MonotonicDeadline(
                100.0, clock=lambda: clock[0]
            )
        )
        return response()
    backend, client, calls = client_factory(replace_context)
    if backend != "sdk":
        pytest.skip("raw benchmark deadline ownership is outside this client")
    owner = deadline_module.MonotonicDeadline(1.0, clock=lambda: clock[0])
    with deadline_module.use_deadline(owner):
        with pytest.raises(deadline_module.DeadlineExceeded):
            invoke(backend, client)
    assert len(calls) == 1 and calls[0]["timeout"] == 1.0
    assert client.call_count == client.successful_responses == 0
    assert _usage(usage_snapshot(client), label="replaced deadline")["total_tokens"] == 8


def test_raw_invalid_json_response_is_nonretryable_and_usage_unknown(monkeypatch):
    calls = []
    class InvalidJSON:
        def raise_for_status(self):
            pass
        def json(self):
            raise ValueError("private response details")
    def post(*_args, **_kwargs):
        calls.append(1)
        return InvalidJSON()
    monkeypatch.setattr(lme.http, "post", post)
    monkeypatch.setattr(lme.time, "sleep", lambda _: pytest.fail("received reply retried"))
    client = lme.LLMClient("deepseek-flash", "test-key")
    try:
        assert client.chat([]) == "[LLM_ERROR:LLMResponseError]"
        assert len(calls) == client.request_attempts == 1
        assert client.call_count == 0
        assert _usage(usage_snapshot(client), label="invalid JSON")["total_tokens"] is None
    finally:
        client.close()
