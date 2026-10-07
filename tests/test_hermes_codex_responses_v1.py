"""Offline contract tests for the direct Hermes OAuth transport."""
from __future__ import annotations

import http.client
import ssl
import time

import pytest

from benchmarks import hermes_codex_responses_v1 as transport


def terminal(*, text=" exact \n", usage=None, output=None):
    return {"type": "response.completed", "response": {
        "status": "completed", "model": transport.MODEL,
        "usage": usage if usage is not None else {
            "input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
        "output": output if output is not None else [{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }],
    }}


def hanging_child(send, timeout):
    time.sleep(timeout + 3)


def successful_child(send, timeout):
    send.send(("ok", transport.Completed("ok", 1, 2, 3, 0, 0)))
    send.close()


def test_request_exact_model_and_schema():
    request = transport.build_request("system", "user")
    assert request == {
        "model": "gpt-6-luna", "reasoning": {"effort": "low"},
        "store": False, "stream": True, "instructions": "system",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "user"}]}],
    }
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    assert transport.build_request("s", "u", schema)["text"] == {"format": {
        "type": "json_schema", "name": "response", "strict": True, "schema": schema}}
    built = transport.build_request("s", "u", schema)
    schema["properties"]["ok"]["type"] = "string"
    assert built["text"]["format"]["schema"]["properties"]["ok"]["type"] == "boolean"
    with pytest.raises(transport.TransportError, match="request_limit"):
        transport.build_request("x" * transport.MAX_REQUEST_BYTES, "u")


def test_credentials_redacted_and_header_safe():
    credentials = transport.Credentials("secret-token", "account")
    assert "secret" not in repr(credentials)
    assert "account" not in repr(credentials)
    for value in ("", "a\nb", "a\rb", "é", "x" * 8193):
        with pytest.raises(transport.TransportError, match="invalid_credentials"):
            transport.Credentials(value, "account")


def test_parse_terminal_exact_usage_and_text_without_delta_buffer():
    events = ({"type": "response.output_text.delta", "delta": "discard"} for _ in range(5000))
    result = transport.parse_stream_events((*events, terminal()))
    assert result == transport.Completed(" exact \n", 11, 7, 18, 3, 2)
    assert "exact" not in repr(result)


def test_commentary_is_not_returned_as_answer():
    messages = [
        {"type": "message", "role": "assistant", "channel": "commentary", "status": "completed",
         "content": [{"type": "output_text", "text": "private planning"}]},
        {"type": "message", "role": "assistant", "channel": "final_answer", "status": "completed",
         "content": [{"type": "output_text", "text": "answer"}]},
    ]
    assert transport.parse_stream_events([terminal(output=messages)]).text == "answer"
    with pytest.raises(transport.TransportError, match="invalid_output"):
        transport.parse_stream_events([terminal(output=messages[:1])])


@pytest.mark.parametrize("events,code", [
    ([], "missing_completion"),
    ([{"type": "response.failed"}], "response_failure"),
    ([{"type": "response.incomplete"}], "response_failure"),
    ([{"type": "response.refusal.delta"}], "unsupported_output"),
    ([terminal(usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 4})], "invalid_usage"),
    ([terminal(output=[{"type": "function_call", "name": "x"}])], "unsupported_output"),
    ([terminal(output=[{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}])], "unsupported_output"),
    ([terminal(), {"type": "response.output_text.delta", "delta": "late"}], "event_after_completion"),
])
def test_parser_fail_closed(events, code):
    with pytest.raises(transport.TransportError, match=code):
        transport.parse_stream_events(events)


def test_missing_and_invalid_usage():
    event = terminal()
    del event["response"]["usage"]
    with pytest.raises(transport.TransportError, match="missing_usage"):
        transport.parse_stream_events([event])
    event = terminal()
    event["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] = 8
    with pytest.raises(transport.TransportError, match="invalid_usage"):
        transport.parse_stream_events([event])
    event = terminal()
    event["response"]["usage"] = {"input_tokens": 0, "output_tokens": 0,
                                    "total_tokens": 0}
    with pytest.raises(transport.TransportError, match="invalid_usage"):
        transport.parse_stream_events([event])


def test_incomplete_status_and_output_bound():
    event = terminal()
    event["response"]["status"] = "incomplete"
    with pytest.raises(transport.TransportError, match="incomplete_response"):
        transport.parse_stream_events([event])
    with pytest.raises(transport.TransportError, match="output_limit"):
        transport.parse_stream_events([terminal(text="x" * (transport.MAX_OUTPUT_CHARS + 1))])
    event = terminal()
    event["response"]["model"] = "other-model"
    with pytest.raises(transport.TransportError, match="model_mismatch"):
        transport.parse_stream_events([event])
    event = terminal()
    del event["response"]["model"]
    with pytest.raises(transport.TransportError, match="model_mismatch"):
        transport.parse_stream_events([event])


def test_malformed_details_and_nested_tool_or_refusal_rejected():
    for details in (None, [], False, 0):
        event = terminal()
        event["response"]["usage"]["input_tokens_details"] = details
        with pytest.raises(transport.TransportError, match="invalid_usage"):
            transport.parse_stream_events([event])
    for item in ({"type": "function_call", "name": "bad"},
                 {"type": "message", "content": [{"type": "refusal"}]}):
        with pytest.raises(transport.TransportError, match="unsupported_output"):
            transport.parse_stream_events([{"type": "response.output_item.added", "item": item}, terminal()])
    with pytest.raises(transport.TransportError, match="unsupported_output"):
        transport.parse_stream_events([{"type": "response.content_part.added", "part": {"type": "refusal"}}, terminal()])
    with pytest.raises(transport.TransportError, match="unsupported_output"):
        transport.parse_stream_events([{"type": "response.some_future_event"}, terminal()])


def test_meaningful_event_limit_excludes_deltas():
    events = [{"type": "response.created"}] * 4096 + [terminal()]
    with pytest.raises(transport.TransportError, match="event_limit"):
        transport.parse_stream_events(events)


class FakeResponse:
    status = 200

    def __init__(self, lines, content_type="text/event-stream"):
        self.lines = iter(lines)
        self.content_type = content_type

    def getheader(self, name, default=None):
        return self.content_type if name == "Content-Type" else default

    def readline(self, limit):
        return next(self.lines, b"")


def test_https_fixed_origin_no_redirect_or_proxy(monkeypatch):
    seen = {}
    class FakeConnection:
        def __init__(self, host, port, *, timeout, context):
            seen.update(host=host, port=port, timeout=timeout, context=context)

        def request(self, method, path, body, headers):
            seen.update(method=method, path=path, body=body, headers=headers)

        def getresponse(self):
            return FakeResponse([b"data: " + __import__("json").dumps(terminal()).encode() + b"\n", b"\n"])

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    monkeypatch.setattr(ssl, "create_default_context", lambda: "verified-context")
    monkeypatch.setenv("HTTPS_PROXY", "http://evil.invalid")
    result = transport._request_once(transport.Credentials("secret", "acct"),
                                     transport.build_request("s", "u"), 5)
    assert result.text == " exact \n"
    assert seen["host"] == "chatgpt.com" and seen["port"] == 443
    assert seen["path"] == "/backend-api/codex/responses"
    assert seen["context"] == "verified-context"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["headers"]["ChatGPT-Account-Id"] == "acct"
    assert seen["closed"]


def test_redirect_is_rejected_without_followup(monkeypatch):
    class Redirect:
        def __init__(self, *args, **kwargs):
            self.requests = 0

        def request(self, *args, **kwargs):
            self.requests += 1

        def getresponse(self):
            response = FakeResponse([])
            response.status = 302
            return response

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", Redirect)
    with pytest.raises(transport.TransportError, match="http_failure"):
        transport._request_once(transport.Credentials("secret", "acct"),
                                transport.build_request("s", "u"), 5)


@pytest.mark.parametrize("status,code", [(401, "auth_failure"), (403, "access_failure"),
                                          (429, "quota_failure")])
def test_http_admission_failures_have_fixed_codes(monkeypatch, status, code):
    class Denied:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            response = FakeResponse([])
            response.status = status
            return response

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", Denied)
    with pytest.raises(transport.TransportError, match=code):
        transport._request_once(transport.Credentials("secret", "acct"),
                                transport.build_request("s", "u"), 5)


def test_timeout_kills_and_reaps_child():
    before = {process.pid for process in __import__("multiprocessing").active_children()}
    start = time.monotonic()
    with pytest.raises(transport.TransportError, match="timeout"):
        transport._run_child(hanging_child, (), 0.2)
    assert time.monotonic() - start < 3
    assert {process.pid for process in __import__("multiprocessing").active_children()} == before


def test_child_success_is_reaped():
    before = {process.pid for process in __import__("multiprocessing").active_children()}
    assert transport._run_child(successful_child, (), 5).text == "ok"
    assert {process.pid for process in __import__("multiprocessing").active_children()} == before
