"""Offline synthetic-wire and spawned-child controls for transport v3."""
from __future__ import annotations

import http.client
import io
import json
import multiprocessing
import time

import pytest

from benchmarks import chatgpt_plan_responses_v3 as transport
from tests.test_chatgpt_plan_responses_root_v1 import terminal
from tests.test_chatgpt_plan_responses_v1 import sse


SECRET = "PRIVATE_SENTINEL_v3"


def _response(status: int, headers: bytes, body: bytes) -> http.client.HTTPResponse:
    class Socket:
        def makefile(self, *_args):
            return io.BytesIO(f"HTTP/1.1 {status} Reply\r\n".encode() + headers + b"\r\n" + body)

    response = http.client.HTTPResponse(Socket())
    response.begin()
    return response


def _request(monkeypatch, response):
    class Connection:
        def __init__(self, host, port, **_kwargs):
            assert (host, port) == ("api.openai.com", 443)

        def request(self, method, path, body, headers):
            assert (method, path) == ("POST", "/v1/responses")
            assert json.loads(body) == transport.build_request("system", "user")
            assert headers["Authorization"] == "Bearer invented-token"

        def getresponse(self):
            return response

        def close(self):
            pass

    monkeypatch.setattr(transport.http.client, "HTTPSConnection", Connection)
    return transport._request_once(transport.Credentials("invented-token"),
                                   transport.build_request("system", "user"), 1)


@pytest.mark.parametrize("headers,defect,count", [
    (b"", "none", 0),
    (b"Content-Type : text/event-stream\r\n", "missing_separator", 0),
    (b"BadHeader\r\nContent-Type: text/event-stream\r\n", "missing_separator", 0),
])
def test_http_response_header_faults_are_finite(monkeypatch, headers, defect, count):
    body = sse(terminal())
    with pytest.raises(transport.TransportError) as caught:
        _request(monkeypatch, _response(200, headers, body))
    exc = caught.value
    assert (exc.code, exc.http_status, exc.body_shape, exc.media_type_class) == (
        "invalid_content_type", 200, "non_json", "missing")
    assert exc.wire_observation == {
        "header_defect": defect, "parsed_header_count": count,
        "transfer_encoding": "missing", "content_encoding": "missing",
        "body_prefix": "sse_prefix", "body_bytes": len(body), "body_truncated": False,
        "sse_validation": "validated_completion"}


def test_framing_classes_and_valid_sse(monkeypatch):
    headers = (b"Transfer-Encoding: chunked\r\nContent-Encoding: gzip\r\n"
               b"Content-Type: application/json\r\n")
    # HTTPResponse de-chunks this body; the observed bytes are the payload.
    body = b"3\r\nabc\r\n0\r\n\r\n"
    with pytest.raises(transport.TransportError) as caught:
        _request(monkeypatch, _response(200, headers, body))
    obs = caught.value.wire_observation
    assert obs["transfer_encoding"] == "chunked"
    assert obs["content_encoding"] == "gzip"
    assert obs["body_bytes"] == 3
    assert obs["body_prefix"] == "text"

    result = _request(monkeypatch, _response(200, b"Content-Type: text/event-stream\r\n",
                                           sse(terminal())))
    assert result.total_tokens == 13


def test_non_200_denial_code_unchanged(monkeypatch):
    body = json.dumps({"error": {"code": "subscription_sharing_user_not_eligible",
                                 "message": SECRET}}).encode()
    with pytest.raises(transport.TransportError) as caught:
        _request(monkeypatch, _response(403, b"Content-Type: application/json\r\n", body))
    exc = caught.value
    assert (exc.code, exc.http_status, exc.body_shape, exc.media_type_class) == (
        "subscription_sharing_user_not_eligible", 403, "error_object", "json")
    assert exc.wire_observation is None
    assert SECRET not in repr(exc)


def test_private_and_mutated_observation_is_rejected():
    valid = {"header_defect": "none", "parsed_header_count": 1,
             "transfer_encoding": "missing", "content_encoding": "missing",
             "body_prefix": "text", "body_bytes": 1, "body_truncated": False,
             "sse_validation": "not_checked"}
    exc = transport.TransportError("invalid_content_type", 200, "non_json", "missing", valid)
    valid["body_prefix"] = SECRET
    assert exc.wire_observation["body_prefix"] == "text"
    exc.wire_observation["body_prefix"] = SECRET
    assert SECRET not in repr(exc)
    for bad in (list(valid), {**valid, "raw_body": SECRET},
                {**valid, "header_defect": SECRET},
                {**valid, "body_bytes": 65538},
                {**valid, "body_truncated": 1},
                {**valid, "body_truncated": True},
                {**valid, "body_prefix": "empty"},
                {**valid, "sse_validation": "validated_completion"},
                {**valid, "body_prefix": "sse_prefix", "sse_validation": "not_checked"},
                {**valid, "body_prefix": "sse_prefix", "sse_validation": "truncated"}):
        assert transport.TransportError("invalid_content_type", wire_observation=bad).wire_observation is None


def _ipc_child(send, observation, timeout):
    send.send(("error", "invalid_content_type", 200, "non_json", "missing", observation))
    send.close()


def _hanging_child(send, timeout):
    time.sleep(timeout + 2)


def test_spawn_ipc_sanitization_and_cleanup():
    before = {child.pid for child in multiprocessing.active_children()}
    valid = {"header_defect": "none", "parsed_header_count": 0,
             "transfer_encoding": "missing", "content_encoding": "missing",
             "body_prefix": "text", "body_bytes": 2, "body_truncated": False,
             "sse_validation": "not_checked"}
    with pytest.raises(transport.TransportError) as caught:
        transport._run_child(_ipc_child, (valid,), 5)
    assert caught.value.wire_observation == valid
    with pytest.raises(transport.TransportError) as caught:
        transport._run_child(_ipc_child, ({**valid, "raw": SECRET},), 5)
    assert caught.value.wire_observation is None
    with pytest.raises(transport.TransportError) as caught:
        transport._run_child(_hanging_child, (), .2)
    assert caught.value.code == "timeout"
    assert {child.pid for child in multiprocessing.active_children()} == before
