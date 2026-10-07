"""Offline public SIWC Responses transport controls; no credential or network I/O."""
from __future__ import annotations

import json
import multiprocessing
import time

import pytest

from benchmarks import chatgpt_plan_responses_v1 as transport


SECRET = "PRIVATE_TOKEN_OR_BODY_7A9"


def envelope():
    return {
        "status": "completed", "model": transport.MODEL,
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
                  "input_tokens_details": {"cached_tokens": 3},
                  "output_tokens_details": {"reasoning_tokens": 2}},
        "output": [{"type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": " exact \n"}]}],
    }


def sse(*events):
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)


class FakeResponse:
    def __init__(self, body=b"", *, status=200, content_type="text/event-stream"):
        self.body = body
        self.position = 0
        self.status = status
        self.content_type = content_type

    def getheader(self, name, default=None):
        return self.content_type if name == "Content-Type" else default

    def read(self, limit):
        value = self.body[self.position:self.position + limit]
        self.position += len(value)
        return value

    def readline(self, limit):
        end = self.body.find(b"\n", self.position)
        end = len(self.body) if end < 0 else end + 1
        end = min(end, self.position + limit)
        value = self.body[self.position:end]
        self.position = end
        return value


def invoke(monkeypatch, response, *, schema=None):
    seen = {"requests": []}

    class FakeConnection:
        def __init__(self, host, port, *, timeout, context):
            seen.update(host=host, port=port, timeout=timeout, context=context)

        def request(self, method, path, body, headers):
            seen["requests"].append((method, path, body, headers))

        def getresponse(self):
            return response

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(transport.http.client, "HTTPSConnection", FakeConnection)
    monkeypatch.setattr(transport.ssl, "create_default_context", lambda: "verified")
    request = transport.build_request("system", "user", schema)
    return seen, lambda: transport._request_once(transport.Credentials(SECRET), request, 5)


def assert_failure(call, code, status=None, shape=None):
    with pytest.raises(transport.TransportError) as exc:
        call()
    assert (exc.value.code, exc.value.http_status, exc.value.body_shape) == (code, status, shape)
    assert SECRET not in repr(exc.value)
    assert SECRET not in str(exc.value)


def test_ordinary_request_is_exact_public_contract(monkeypatch):
    response = FakeResponse(sse({"type": "response.output_text.delta", "delta": SECRET},
                                {"type": "response.completed", "response": envelope()}))
    seen, call = invoke(monkeypatch, response)
    assert call() == transport.Completed(" exact \n", 11, 7, 18, 3, 2)
    assert seen["host"] == "api.openai.com" and seen["port"] == 443
    assert seen["context"] == "verified" and seen["closed"]
    assert len(seen["requests"]) == 1
    method, path, body, headers = seen["requests"][0]
    assert (method, path) == ("POST", "/v1/responses")
    assert headers == {"Authorization": "Bearer " + SECRET,
                       "Content-Type": "application/json", "Accept": "text/event-stream"}
    assert json.loads(body) == {
        "model": "gpt-5.6-luna", "reasoning": {"effort": "low"},
        "store": False, "stream": True, "instructions": "system",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "user"}]}],
    }
    assert not ({"temperature", "max_output_tokens", "service_tier", "prompt", "tools"}
                & json.loads(body).keys())


def test_structured_request_is_copy_and_exact_format(monkeypatch):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    request = transport.build_request("s", "u", schema)
    assert request["text"] == {"format": {"type": "json_schema", "name": "response",
                                          "strict": True, "schema": schema}}
    schema["properties"]["answer"]["type"] = "integer"
    assert request["text"]["format"]["schema"]["properties"]["answer"]["type"] == "string"
    seen, call = invoke(monkeypatch, FakeResponse(sse(
        {"type": "response.completed", "response": envelope()})), schema=schema)
    assert call().text == " exact \n"
    assert "text" in json.loads(seen["requests"][0][2])


@pytest.mark.parametrize("credentials", ["x", None])
def test_complete_requires_exact_credentials_without_dispatch(monkeypatch, credentials):
    monkeypatch.setattr(transport, "_run_child", lambda *args: pytest.fail("dispatched"))
    assert_failure(lambda: transport.complete(credentials, "s", "u"), "invalid_credentials")


def test_private_reprs_and_bad_header():
    credentials = transport.Credentials(SECRET)
    assert SECRET not in repr(credentials)
    assert SECRET not in repr(transport.Completed(SECRET, 1, 1, 2, 0, 0))
    assert_failure(lambda: transport.Credentials("bad\r\nheader"), "invalid_credentials")


@pytest.mark.parametrize("system,user,schema,code", [
    (None, "u", None, "invalid_request"),
    ("s", None, None, "invalid_request"),
    ("s", "u", {"x": float("nan")}, "invalid_request"),
    ("s", "u", ["not a schema object"], "invalid_request"),
    ("s", "x" * transport.MAX_REQUEST_BYTES, None, "request_limit"),
])
def test_invalid_request_fails_before_dispatch(system, user, schema, code):
    assert_failure(lambda: transport.build_request(system, user, schema), code)


@pytest.mark.parametrize("change,code", [
    (lambda value: value.update(model="gpt-6-luna"), "model_mismatch"),
    (lambda value: value.update(status="incomplete"), "incomplete_response"),
    (lambda value: value.pop("usage"), "missing_usage"),
    (lambda value: value["usage"].update(output_tokens=0, total_tokens=11), "invalid_usage"),
    (lambda value: value["usage"].update(output_tokens=True), "invalid_usage"),
    (lambda value: value["output"].append({"type": "function_call", "name": SECRET}), "unsupported_output"),
    (lambda value: value["output"][0]["content"].append({"type": "refusal", "refusal": SECRET}), "unsupported_output"),
    (lambda value: value["output"][0].update(status="in_progress"), "unsupported_output"),
    (lambda value: value["output"][0]["content"][0].update(text="x" * (transport.MAX_OUTPUT_CHARS + 1)), "output_limit"),
])
def test_terminal_validation(change, code):
    value = envelope()
    change(value)
    assert_failure(lambda: transport.parse_stream_events([
        {"type": "response.completed", "response": value}]), code)


def test_errors_after_deltas_are_terminal_and_private(monkeypatch):
    events = [{"type": "response.output_text.delta", "delta": SECRET},
              {"type": "response.failed", "response": {"error": {
                  "code": "subscription_sharing_usage_limit_exceeded", "message": SECRET}}}]
    seen, call = invoke(monkeypatch, FakeResponse(sse(*events)))
    assert_failure(call, "subscription_sharing_usage_limit_exceeded", 200, "sse_event")
    assert len(seen["requests"]) == 1 and seen["closed"]
    assert_failure(lambda: transport.parse_stream_events(events[:1]), "missing_completion")


@pytest.mark.parametrize("event,code", [
    ({"type": "response.incomplete", "response": {"status": "incomplete"}}, "incomplete_response"),
    ({"type": "error", "error": {"code": "chatpass_v2_scope_not_authorized", "message": SECRET}},
     "chatpass_v2_scope_not_authorized"),
    ({"type": "error", "code": "chatpass_v2_invalid_authorization_context", "message": SECRET},
     "chatpass_v2_invalid_authorization_context"),
    ({"type": "response.failed", "response": {"error": {"code": SECRET}}}, "response_failure"),
    ({"type": "response.output_item.added", "item": {"type": "function_call"}}, "unsupported_output"),
    ({"type": "new_unknown_event", "message": SECRET}, "unsupported_output"),
])
def test_failure_and_unsupported_events(event, code):
    assert_failure(lambda: transport.parse_stream_events([event]), code)


def test_post_completion_and_event_limit():
    terminal = {"type": "response.completed", "response": envelope()}
    assert_failure(lambda: transport.parse_stream_events([terminal,
        {"type": "response.output_text.delta", "delta": SECRET}]), "event_after_completion")
    events = [{"type": "response.in_progress"}] * (transport.MAX_MEANINGFUL_EVENTS + 1)
    assert_failure(lambda: transport.parse_stream_events(events), "event_limit")


@pytest.mark.parametrize("body,code", [
    (b"data: {\"type\":\"response.completed\"\n\n", "invalid_event"),
    (b"data: {\"type\":\"response.completed\",\"type\":\"response.completed\"}\n\n", "invalid_event"),
    (b"data: {\"type\":NaN}\n\n", "invalid_event"),
    (b"data: {\"type\":1e999}\n\n", "invalid_event"),
    (b"data: {\"type\":\"response.created\"}", "truncated_stream"),
    (b"data: [DONE]\n\n", "missing_completion"),
    (b"data: " + b"x" * transport.MAX_LINE_BYTES + b"\n", "wire_limit"),
])
def test_bounded_sse_parser(monkeypatch, body, code):
    _, call = invoke(monkeypatch, FakeResponse(body))
    assert_failure(call, code)


@pytest.mark.parametrize("body,status,code,shape", [
    (b"", 401, "auth_failure", "empty"),
    (json.dumps({"detail": SECRET}).encode(), 403, "access_failure", "detail"),
    (json.dumps({"error": {"code": "subscription_sharing_unsupported_capability",
                           "message": SECRET, "param": SECRET}}).encode(),
     400, "subscription_sharing_unsupported_capability", "error_object"),
    (json.dumps({"error": {"code": "chatpass_v2_invalid_authorization_context"}}).encode(),
     403, "chatpass_v2_invalid_authorization_context", "error_object"),
    (json.dumps({"error": {"code": SECRET}}).encode(), 429, "quota_failure", "error_object"),
    (b"<html>" + SECRET.encode(), 503, "http_failure", "non_json"),
    (b"[]", 302, "http_failure", "other_json"),
    (b"x" * (transport.MAX_WIRE_BYTES + 1), 503, "http_failure", "oversized"),
])
def test_non_200_finite_attribution(monkeypatch, body, status, code, shape):
    seen, call = invoke(monkeypatch, FakeResponse(body, status=status))
    assert_failure(call, code, status, shape)
    assert len(seen["requests"]) == 1 and seen["closed"]


def test_200_requires_sse(monkeypatch):
    _, call = invoke(monkeypatch, FakeResponse(SECRET.encode(), content_type="application/json"))
    assert_failure(call, "invalid_content_type", 200)


def hanging_child(send, timeout):
    time.sleep(timeout + 3)


def successful_child(send, timeout):
    send.send(("ok", transport.Completed("ok", 1, 2, 3, 0, 0)))
    send.close()


def failure_child(send, code, status, shape, timeout):
    send.send(("error", code, status, shape))
    send.close()


def test_spawned_hung_child_is_killed_and_reaped():
    before = {process.pid for process in multiprocessing.active_children()}
    start = time.monotonic()
    assert_failure(lambda: transport._run_child(hanging_child, (), 0.2), "timeout")
    assert time.monotonic() - start < 3
    assert {process.pid for process in multiprocessing.active_children()} == before
    assert transport._run_child(successful_child, (), 5).text == "ok"
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_parent_child_error_boundary_and_timeouts():
    assert_failure(lambda: transport._run_child(failure_child,
        ("subscription_sharing_user_not_eligible", 403, "error_object"), 5),
        "subscription_sharing_user_not_eligible", 403, "error_object")
    assert_failure(lambda: transport._run_child(failure_child,
        (SECRET, 999, SECRET), 5), "transport_failure")
    assert_failure(lambda: transport._run_child(failure_child,
        ("http_failure", 503, [SECRET]), 5), "http_failure", 503)
    for invalid in (0, -1, 121, float("nan"), float("inf"), "5"):
        assert_failure(lambda: transport._run_child(successful_child, (), invalid),
                       "invalid_timeout")
