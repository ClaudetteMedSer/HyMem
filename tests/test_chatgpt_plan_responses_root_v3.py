"""Independent synthetic-wire controls; no live endpoint or private state."""
import io
import http.client
import json

import pytest

from benchmarks import chatgpt_plan_responses_v3 as wire
from tests.test_chatgpt_plan_responses_root_v1 import terminal
from tests.test_chatgpt_plan_responses_v1 import sse


def response(headers, body):
    class Socket:
        def makefile(self, *args):
            return io.BytesIO(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n" + body)
    result = http.client.HTTPResponse(Socket())
    result.begin()
    return result


def request(monkeypatch, reply):
    class Connection:
        def __init__(self, host, port, **kwargs):
            assert host == "api.openai.com" and port == 443
        def request(self, method, path, body, headers):
            assert method == "POST" and path == "/v1/responses"
            assert headers["Authorization"] == "Bearer invented-token"
            assert json.loads(body) == wire.build_request("invented", "invented")
        def getresponse(self):
            return reply
        def close(self):
            pass
    monkeypatch.setattr(wire.http.client, "HTTPSConnection", Connection)
    return wire._request_once(wire.Credentials("invented-token"),
                              wire.build_request("invented", "invented"), 1)


@pytest.mark.parametrize("headers,defect,count", [
    (b"X-Example: invented\r\n", "none", 1),
    (b"Content-Type : text/event-stream\r\n", "missing_separator", 0),
    (b"BadHeader\r\nContent-Type: text/event-stream\r\n", "missing_separator", 0),
])
def test_missing_vs_unparsed_headers_distinguished(monkeypatch, headers, defect, count):
    body = sse(terminal())
    with pytest.raises(wire.TransportError) as caught:
        request(monkeypatch, response(headers, body))
    error = caught.value
    assert error.code == "invalid_content_type"
    assert error.media_type_class == "missing"
    assert error.body_shape == "non_json"
    obs = error.wire_observation
    assert obs["header_defect"] == defect and obs["parsed_header_count"] == count
    assert obs["body_prefix"] == "sse_prefix"
    assert obs["body_bytes"] == len(body) and obs["body_truncated"] is False
    assert obs["sse_validation"] == "validated_completion"


@pytest.mark.parametrize("body,expected", [
    (b"event: response.created\n", "sse_prefix"),
    (b"\xef\xbb\xbf\n\ndata: {}\n\n", "sse_prefix"),
    (b": invented private comment\n", "sse_prefix"),
    (b"<!DOCTYPE html><html>PRIVATE_SENTINEL</html>", "html_prefix"),
    (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n", "http_prefix"),
    (b'{"PRIVATE_SENTINEL":1}', "json_prefix"),
    (b"PRIVATE_SENTINEL", "text"),
    (b"\x1f\x8bPRIVATE_SENTINEL", "binary"),
    (b"", "empty"),
])
def test_body_prefix_is_metadata_only_not_acceptance(monkeypatch, body, expected):
    with pytest.raises(wire.TransportError) as caught:
        request(monkeypatch, response(b"", body))
    error = caught.value
    assert error.wire_observation["body_prefix"] == expected
    assert error.code == "invalid_content_type"
    assert "PRIVATE_SENTINEL" not in repr(error) + str(error) + json.dumps(error.wire_observation)


def test_normal_sse_and_usage_validation_preserved(monkeypatch):
    result = request(monkeypatch, response(b"Content-Type: text/event-stream\r\n", sse(terminal())))
    assert result.total_tokens == 13
    event = terminal()
    event["response"]["usage"]["total_tokens"] = 14
    with pytest.raises(wire.TransportError) as caught:
        request(monkeypatch, response(b"Content-Type: text/event-stream\r\n", sse(event)))
    assert caught.value.code == "invalid_usage"


def test_bounded_diagnostic_body_size(monkeypatch):
    with pytest.raises(wire.TransportError) as caught:
        request(monkeypatch, response(b"", b"x" * 70000))
    obs = caught.value.wire_observation
    assert obs["body_bytes"] == 65537 and obs["body_truncated"] is True
    assert caught.value.body_shape == "oversized"


def test_duplicate_mime_headers_cannot_create_success(monkeypatch):
    headers = b"Content-Type: text/event-stream\r\nContent-Type: text/html\r\n"
    with pytest.raises(wire.TransportError):
        request(monkeypatch, response(headers, sse(terminal())))


def test_json_completion_remains_failure(monkeypatch):
    with pytest.raises(wire.TransportError) as caught:
        request(monkeypatch, response(b"Content-Type: application/json\r\n",
                                     json.dumps(terminal()["response"]).encode()))
    assert caught.value.code == "invalid_content_type"
    assert caught.value.wire_observation["body_prefix"] == "json_prefix"


@pytest.mark.parametrize("mutation", ["usage", "model", "error"])
def test_sse_observer_preserves_actual_terminal_checks(monkeypatch, mutation):
    event = terminal()
    if mutation == "usage":
        event["response"]["usage"]["total_tokens"] = 99
    elif mutation == "model":
        event["response"]["model"] = "different-invented-model"
    else:
        event["response"]["error"] = {"code": "invented"}
    with pytest.raises(wire.TransportError) as caught:
        request(monkeypatch, response(b"", sse(event)))
    assert caught.value.code == "invalid_content_type"
    assert caught.value.wire_observation["sse_validation"] == "invalid"


@pytest.mark.parametrize("change", [
    {"parsed_header_count": True}, {"parsed_header_count": 101},
    {"header_defect": ["PRIVATE_SENTINEL"]}, {"body_bytes": -1},
    {"body_truncated": True}, {"body_prefix": "empty"},
    {"sse_validation": "validated_completion"}, {"extra": "PRIVATE_SENTINEL"},
])
def test_sanitizer_rejects_inconsistent_or_private_metadata(change):
    observation = {"header_defect": "none", "parsed_header_count": 0,
        "transfer_encoding": "missing", "content_encoding": "missing",
        "body_prefix": "text", "body_bytes": 3, "body_truncated": False,
        "sse_validation": "not_checked"}
    observation.update(change)
    error = wire.TransportError("invalid_content_type", wire_observation=observation)
    assert error.wire_observation is None
    assert "PRIVATE_SENTINEL" not in repr(error)
