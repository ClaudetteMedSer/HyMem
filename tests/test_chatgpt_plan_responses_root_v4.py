"""Root-owned compatibility controls; synthetic HTTP/SSE and invented text."""
import json

import pytest

from benchmarks import chatgpt_plan_responses_v4 as wire
from tests.test_chatgpt_plan_responses_root_v3 import response
from tests.test_chatgpt_plan_responses_root_v1 import terminal
from tests.test_chatgpt_plan_responses_v1 import sse


def call(monkeypatch, body, headers=b""):
    reply = response(headers, body)
    class Connection:
        def __init__(self, host, port, **kwargs):
            assert (host, port) == ("api.openai.com", 443)
        def request(self, method, path, body, headers):
            assert (method, path) == ("POST", "/v1/responses")
            assert json.loads(body) == wire.build_request("s", "u")
        def getresponse(self): return reply
        def close(self): pass
    monkeypatch.setattr(wire.http.client, "HTTPSConnection", Connection)
    return wire._request_once(wire.Credentials("invented-token"), wire.build_request("s", "u"), 1)


@pytest.mark.parametrize("headers", [b"", b"X-Example: invented\r\n", b"Content-Type: text/event-stream\r\n"])
def test_valid_stream_preserves_required_semantics_without_mime(monkeypatch, headers):
    result = call(monkeypatch, sse(terminal()), headers)
    assert result.total_tokens == 13


@pytest.mark.parametrize("body", [
    b"<html>PRIVATE_SENTINEL</html>\n", b'{"data":"PRIVATE_SENTINEL"}\n',
    b"garbage\n", b"retry: nope\n", b"event: mismatched\n",
])
def test_arbitrary_prefix_cannot_smuggle_valid_completion(monkeypatch, body):
    with pytest.raises(wire.TransportError) as caught:
        call(monkeypatch, body + sse(terminal()))
    assert "PRIVATE_SENTINEL" not in str(caught.value) + repr(caught.value)


@pytest.mark.parametrize("headers", [
    b"Content-Type: text/html\r\n", b"Content-Type: application/json\r\n",
    b"Content-Type : text/event-stream\r\n",
    b"BadHeader\r\nContent-Type: text/event-stream\r\n",
    b"Content-Encoding: gzip\r\n",
])
def test_only_evidenced_missing_header_case_has_compatibility_path(monkeypatch, headers):
    with pytest.raises(wire.TransportError):
        call(monkeypatch, sse(terminal()), headers)


def test_documented_reasoning_done_does_not_export_reasoning(monkeypatch):
    done = {"type": "response.reasoning_text.done", "text": "PRIVATE_SENTINEL"}
    result = call(monkeypatch, sse(done) + sse(terminal()))
    assert result.total_tokens == 13 and "PRIVATE_SENTINEL" not in repr(result)
    with pytest.raises(wire.TransportError) as caught:
        call(monkeypatch, sse(terminal()) + sse(done))
    assert caught.value.code == "event_after_completion"
    assert "PRIVATE_SENTINEL" not in repr(caught.value)


@pytest.mark.parametrize("case,code", [("model", "model_mismatch"), ("usage", "invalid_usage"),
                                      ("incomplete", "incomplete_response"), ("denial", "subscription_sharing_usage_unavailable")])
def test_original_denial_model_usage_and_completion_gates(monkeypatch, case, code):
    event = terminal()
    if case == "model": event["response"]["model"] = "wrong-invented-model"
    elif case == "usage": event["response"]["usage"]["total_tokens"] = 0
    elif case == "incomplete": event["response"]["status"] = "incomplete"
    else: event = {"type": "response.failed", "response": {"error": {"code": code}}}
    with pytest.raises(wire.TransportError) as caught:
        call(monkeypatch, sse(event))
    assert caught.value.code == code
    assert "wrong-invented-model" not in repr(caught.value)


def test_standard_bom_comment_id_retry_and_matching_event(monkeypatch):
    body = (b"\xef\xbb\xbf: invented\n\nid: opaque-invented\nretry: 123\n"
            b"event: response.completed\n" + sse(terminal()))
    assert call(monkeypatch, body).total_tokens == 13


def test_unfinished_frame_not_accepted(monkeypatch):
    with pytest.raises(wire.TransportError):
        call(monkeypatch, sse(terminal()).rstrip(b"\n"))


@pytest.mark.parametrize("field", ["type", "channel", "content"])
@pytest.mark.parametrize("value", [{"PRIVATE_SENTINEL": []}, ["PRIVATE_SENTINEL"]])
def test_malformed_terminal_observation_preserves_finite_failure(monkeypatch, field, value):
    event = terminal()
    item = event["response"]["output"][0]
    if field == "content":
        item["content"][0]["type"] = value
    else:
        item[field] = value
    with pytest.raises(wire.TransportError) as caught:
        call(monkeypatch, sse(event))
    assert caught.value.code in {"unsupported_output", "missing_final_answer"}
    assert caught.value.stream_observation is not None
    assert "PRIVATE_SENTINEL" not in repr(caught.value)


def test_surrogate_event_label_is_finite_failure(monkeypatch):
    body = b'event: response.completed\ndata: {"type":"\\ud800"}\n\n'
    with pytest.raises(wire.TransportError) as caught:
        call(monkeypatch, body)
    assert caught.value.code == "invalid_event"
    assert caught.value.stream_observation["event_type"] == "unknown"


def test_wire_and_event_limits_still_enforced(monkeypatch):
    monkeypatch.setattr(wire, "MAX_WIRE_BYTES", 10)
    with pytest.raises(wire.TransportError) as caught:
        call(monkeypatch, sse(terminal()))
    assert caught.value.code == "wire_limit"
    monkeypatch.setattr(wire, "MAX_MEANINGFUL_EVENTS", 1)
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([{"type": "response.reasoning_text.done"}, terminal()])
    assert caught.value.code == "event_limit"
