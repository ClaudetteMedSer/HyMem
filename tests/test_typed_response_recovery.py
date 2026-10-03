"""Real maintained SDK dispatch into extraction; all provider replies synthetic."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from benchmarks import extraction_canary as canary
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.deadline import DeadlineExceeded, MonotonicDeadline, use_deadline
from hymem.dreaming import digest, summary_recovery
from hymem.extraction import chunk
from hymem.extraction.llm import LLMRequest, LLMOutputTruncatedError, LLMResponseError
from tests.test_completion_response_admission import response, sdk_response
from tests.test_digest_bounded_summary_repair import source, payload, extract, SequenceLLM
from tests.test_summary_recovery_v63 import conn, _seed, _public, _items, _no_lease
from tests.test_benchmark_extraction_canary import _representative_response


@pytest.fixture
def sdk_factory(monkeypatch):
    import openai
    clients = []
    monkeypatch.delenv("HYMEM_LLM_EXTRA_BODY", raising=False)
    monkeypatch.setattr("hymem.extraction.retry.time.sleep", lambda _: None)

    def build(values):
        calls = []
        class Resource:
            def create(self, **wire):
                calls.append(wire)
                value = values(wire) if callable(values) else values.pop(0)
                if isinstance(value, BaseException):
                    raise value
                return sdk_response(value)
        class SDK:
            def __init__(self, **options):
                self._client = options["http_client"]
                self.chat = SimpleNamespace(completions=Resource())
            def close(self):
                self._client.close()
        monkeypatch.setattr(openai, "OpenAI", SDK)
        client = OpenAICompatibleClient(api_key="synthetic-not-a-key", model="deepseek-flash")
        clients.append(client)
        return client, calls

    yield build
    for client in clients:
        client.close()


def request(wire):
    return LLMRequest(system=wire["messages"][0]["content"],
                      user=wire["messages"][1]["content"])


def test_sdk_length_has_only_typed_bounded_evidence_and_paid_usage(sdk_factory):
    client, calls = sdk_factory([response(finish="length", content="PRIVATE_PARTIAL")])
    with client.track_provider_attempts() as tracker:
        with pytest.raises(LLMOutputTruncatedError) as caught:
            client.complete(LLMRequest("system", "user"))
    assert caught.value.args == ("LLM response exceeded its output budget",)
    assert not vars(caught.value)
    assert len(calls) == tracker.attempts == client.request_attempts == 1
    assert client.call_count == client.successful_responses == 0
    assert client.token_usage_available and client.total_tokens == 8


@pytest.mark.parametrize("mutation", ["two_choices", "no_message", "no_content", "bad_content"])
def test_malformed_length_reply_never_gets_recoverable_authority(sdk_factory, mutation):
    value = response(finish="length")
    if mutation == "two_choices":
        value["choices"].append(deepcopy(value["choices"][0]))
    elif mutation == "no_message":
        del value["choices"][0]["message"]
    elif mutation == "no_content":
        del value["choices"][0]["message"]["content"]
    else:
        value["choices"][0]["message"]["content"] = 3
    client, calls = sdk_factory([value])
    with pytest.raises(LLMResponseError) as caught:
        client.complete(LLMRequest("system", "user"))
    assert not isinstance(caught.value, LLMOutputTruncatedError)
    assert len(calls) == 1 and client.total_tokens == 8


@pytest.mark.parametrize("when", ["primary", "omission"])
def test_real_sdk_length_recovers_real_chunk_with_omission_and_exact_accounting(sdk_factory, when):
    rejected = []
    def reply(wire):
        req = request(wire)
        if not rejected and ((when == "primary") or "OMISSION VERIFICATION PASS" in req.system):
            rejected.append(req)
            return response(finish="length", content="PRIVATE_PARTIAL")
        return response(content=_representative_response(req))
    client, calls = sdk_factory(reply)
    result = chunk.extract_chunk(client, canary._CANARY_CONTENT,
                                 source_records=canary._source_records(), completion_call_limit=24)
    assert rejected and not result.failed
    assert len(result.triples) == 2 and not result.markers
    assert result.completion_calls == result.provider_attempts == len(calls) <= 24
    assert client.call_count == client.successful_responses == len(calls) - 1
    assert client.total_tokens == 8 * len(calls) and client.token_usage_available
    assert any("OMISSION VERIFICATION PASS" in request(wire).system for wire in calls)
    # The failed source unit is actually replaced by smaller source requests.
    assert any(len(request(wire).user) < len(rejected[0].user) for wire in calls[1:])


@pytest.mark.parametrize("finish", ["content_filter", "tool_calls", None, "unknown"])
def test_other_sdk_rejections_do_not_split_or_masquerade_as_length(sdk_factory, finish):
    client, calls = sdk_factory([response(finish=finish, content="PRIVATE_PARTIAL")])
    result = chunk.extract_chunk(client, canary._CANARY_CONTENT,
                                 source_records=canary._source_records(), completion_call_limit=24)
    assert result.failed and not result.triples
    assert len(calls) == result.completion_calls == result.provider_attempts == 1


@pytest.mark.parametrize("separated", [False, True])
def test_real_sdk_truncated_digest_repair_preserves_only_proven_items(source, sdk_factory, separated):
    client, calls = sdk_factory([
        response(content=json.dumps(payload(source, "x" * 644))),
        response(finish="length", content='{"summary":"PRIVATE_PARTIAL'),
    ])
    before = source[0].conn.total_changes
    result = extract(source, client, separate_summary=separated)
    assert len(calls) == 2 and client.request_attempts == 2
    assert client.call_count == 1 and client.total_tokens == 16 and client.token_usage_available
    assert result.summary is None and source[0].conn.total_changes == before
    if separated:
        reference = extract(source, SequenceLLM(payload(source)))
        assert not result.parse_failed and result.summary_failure_reason == "output_truncated"
        assert result.episodes == reference.episodes and result.procedures == reference.procedures
        assert result.source_sha256 == reference.source_sha256
        assert result.covered_message_id == reference.covered_message_id
    else:
        assert result.parse_failed and result.failure_reason == "output_truncated"
        assert result.failure_stage == "summary_compaction"
        assert result.source_sha256 is None and result.covered_message_id is None
        assert not result.episodes.items and not result.procedures.items


def test_real_sdk_truncated_primary_digest_never_salvages_partial_items(source, sdk_factory):
    client, calls = sdk_factory([response(finish="length", content=json.dumps(payload(source)))])
    with pytest.raises(digest.DigestCompletionError) as caught:
        extract(source, client, separate_summary=True)
    assert caught.value.failure_stage == "primary"
    assert isinstance(caught.value.__cause__, LLMOutputTruncatedError)
    assert len(calls) == 1 and client.call_count == 0 and client.total_tokens == 8


@pytest.mark.parametrize("finish", ["content_filter", "unknown"])
def test_real_sdk_nonlength_summary_repair_is_fatal(source, sdk_factory, finish):
    client, calls = sdk_factory([response(content=json.dumps(payload(source, "x" * 644))),
                                 response(finish=finish)])
    with pytest.raises(digest.DigestCompletionError) as caught:
        extract(source, client, separate_summary=True)
    assert caught.value.failure_stage == "summary_compaction"
    assert type(caught.value.__cause__) is LLMResponseError and len(calls) == 2


def test_owning_deadline_precedes_length_downgrade_in_summary_repair(source, sdk_factory):
    clock, seen = [0.0], []
    def reply(wire):
        seen.append(wire)
        if len(seen) == 1:
            return response(content=json.dumps(payload(source, "x" * 644)))
        clock[0] = 2.0
        return response(finish="length")
    client, calls = sdk_factory(reply)
    with use_deadline(MonotonicDeadline(1.0, clock=lambda: clock[0])):
        with pytest.raises(DeadlineExceeded):
            extract(source, client, separate_summary=True)
    assert len(calls) == 2 and client.total_tokens == 16


def test_real_sdk_length_holds_private_summary_job_without_advancing(conn, sdk_factory):
    _seed(conn, long=True)
    old = _public(conn), _items(conn)
    client, calls = sdk_factory([response(finish="length", content="PRIVATE_PARTIAL")])
    report = summary_recovery.run_summary_recovery(conn, client, max_calls=4)
    assert report["calls"] == report["provider_attempts"] == report["held"] == 1
    assert report["provider_attempts_exact"] is True
    assert report["advanced"] == report["published"] == 0
    assert (_public(conn), _items(conn)) == old
    job = conn.execute("SELECT * FROM summary_recovery").fetchone()
    assert (job["cursor_message_id"], job["cursor_partial_message_id"], job["cursor_offset"], job["draft"]) == (None, None, 0, "")
    assert job["failure_reason"] == "output_truncated" and job["attempts"] == 1
    assert len(calls) == 1 and client.total_tokens == 8
    _no_lease(conn)


def test_real_configured_canary_accounts_recovered_sdk_length_and_closes(sdk_factory):
    seen = []
    def reply(wire):
        seen.append(wire)
        if len(seen) == 1:
            return response(finish="length", content="PRIVATE_PARTIAL")
        return response(content=_representative_response(request(wire)))
    # Installing the synthetic SDK leaves the configured canary's real
    # constructor, response admission, extraction and close path intact.
    unused, _ = sdk_factory(reply)
    unused.close()
    report = canary.run_configured_extraction_canary(api_key="synthetic-not-a-key",
        base_url="https://api.deepseek.com", model="deepseek-flash", thinking="disabled")
    assert report["status"] == "passed" and report["client_closed"] is True
    assert report["version"] == "hymem-phase1-extraction-canary-v20"
    assert report["execution_path"]["provider_output_truncations"] == 1
    assert report["usage"]["calls"] + 1 == report["completion_calls"] == len(seen)
    assert report["provider_attempts"] == len(seen) <= 24
    assert report["usage"]["total_tokens"] == len(seen) * 8


def test_operator_extra_body_remains_an_unmodified_extension(sdk_factory, monkeypatch):
    client, calls = sdk_factory([response()])
    monkeypatch.setenv("HYMEM_LLM_EXTRA_BODY", '{"diagnostic_extension":true}')
    assert client.complete(LLMRequest("system", "user")) == "ok"
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}, "diagnostic_extension": True}


@pytest.mark.parametrize("failure", ["content_filter", "unknown", "deadline"])
def test_private_summary_worker_does_not_downgrade_other_failures(conn, sdk_factory, failure):
    _seed(conn, long=True)
    old = _public(conn), _items(conn)
    reply = DeadlineExceeded("synthetic deadline") if failure == "deadline" else response(finish=failure)
    client, calls = sdk_factory([reply])
    expected = DeadlineExceeded if failure == "deadline" else LLMResponseError
    with pytest.raises(expected):
        summary_recovery.run_summary_recovery(conn, client, max_calls=4)
    assert (_public(conn), _items(conn)) == old and len(calls) == 1
    job = conn.execute("SELECT * FROM summary_recovery").fetchone()
    assert job["attempts"] == 1 and job["failure_reason"] is None
    assert job["cursor_message_id"] is None and job["draft"] == ""
    _no_lease(conn)


def test_persistent_sdk_length_cannot_escape_chunk_resource_cap(sdk_factory):
    client, calls = sdk_factory(lambda _: response(finish="length", content="PRIVATE_PARTIAL"))
    result = chunk.extract_chunk(client, canary._CANARY_CONTENT,
                                 source_records=canary._source_records(), completion_call_limit=24)
    assert result.failed and not result.triples and not result.markers
    assert result.completion_calls == result.provider_attempts == len(calls) <= 24
    assert client.call_count == 0 and client.total_tokens == len(calls) * 8


def test_sdk_response_error_alias_is_part_of_live_integrity_guard(sdk_factory, monkeypatch):
    from hymem.contrib import openai_client as sdk
    client, calls = sdk_factory([response()])
    class AlteredLengthError(LLMOutputTruncatedError):
        pass
    monkeypatch.setattr(sdk, "LLMOutputTruncatedError", AlteredLengthError)
    with pytest.raises(RuntimeError, match="identity changed"):
        client.complete(LLMRequest("system", "user"))
    assert not calls
