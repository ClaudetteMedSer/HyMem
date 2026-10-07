"""Root-owned finite non-SSE diagnostics. Invented data, no network or account use."""
import json
import multiprocessing
import time

import pytest

from benchmarks import chatgpt_plan_responses_v2 as wire
from tests.test_chatgpt_plan_responses_root_v1 import terminal
from tests.test_chatgpt_plan_responses_root_v1 import partial_ipc
from tests.test_chatgpt_plan_responses_v1 import FakeResponse, sse


def call_with(monkeypatch, response):
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, *args, **kwargs): pass
        def getresponse(self): return response
        def close(self): pass
    monkeypatch.setattr(wire.http.client, "HTTPSConnection", Connection)
    return lambda: wire._request_once(wire.Credentials("invented-token"),
                                      wire.build_request("s", "u"), 1)


@pytest.mark.parametrize("header,media,body,shape", [
    ("application/json", "json", b'{"detail":"PRIVATE_DIAGNOSTIC"}', "detail"),
    ("text/html; charset=utf-8", "html", b"<html>PRIVATE_DIAGNOSTIC</html>", "non_json"),
    ("text/plain", "text", b"PRIVATE_DIAGNOSTIC", "non_json"),
    (None, "missing", b"", "empty"),
    ("application/x-PRIVATE_DIAGNOSTIC", "other", b"[]", "other_json"),
    ("x" * 9000, "invalid", b"{}", "other_json"),
])
def test_finite_shape_never_exports_body_or_header(monkeypatch, header, media, body, shape):
    with pytest.raises(wire.TransportError) as caught:
        call_with(monkeypatch, FakeResponse(body, content_type=header))()
    error = caught.value
    assert error.code == "invalid_content_type" and error.http_status == 200
    assert error.media_type_class == media and error.body_shape == shape
    assert "PRIVATE_DIAGNOSTIC" not in str(error) + repr(error)


def test_valid_json_completed_response_still_fails(monkeypatch):
    response = FakeResponse(json.dumps(terminal()["response"]).encode(), content_type="application/json")
    with pytest.raises(wire.TransportError) as caught:
        call_with(monkeypatch, response)()
    assert caught.value.code == "invalid_content_type"
    assert caught.value.body_shape == "other_json"


def test_known_provider_denial_is_preserved_not_message(monkeypatch):
    body = json.dumps({"error": {"code": "subscription_sharing_usage_limit_exceeded",
                                 "message": "PRIVATE_DIAGNOSTIC"}}).encode()
    with pytest.raises(wire.TransportError) as caught:
        call_with(monkeypatch, FakeResponse(body, content_type="application/json"))()
    assert caught.value.code == "subscription_sharing_usage_limit_exceeded"
    assert caught.value.http_status == 200
    assert caught.value.body_shape == "error_object"
    assert "PRIVATE_DIAGNOSTIC" not in repr(caught.value)


def test_non_sse_body_read_is_small_and_bounded(monkeypatch):
    class Response(FakeResponse):
        def read(self, limit):
            assert limit <= 65537
            return super().read(limit)
    with pytest.raises(wire.TransportError) as caught:
        call_with(monkeypatch, Response(b"x" * 65538, content_type="application/json"))()
    assert caught.value.body_shape == "oversized"


def test_sse_success_and_model_usage_gates_unchanged(monkeypatch):
    result = call_with(monkeypatch, FakeResponse(sse(terminal())))()
    assert result.total_tokens == 13
    event = terminal()
    event["response"]["usage"]["total_tokens"] = 12
    with pytest.raises(wire.TransportError) as caught:
        call_with(monkeypatch, FakeResponse(sse(event)))()
    assert caught.value.code == "invalid_usage"


def test_unhashable_or_arbitrary_media_is_sanitized():
    for media in (["PRIVATE_DIAGNOSTIC"], {"x": 1}, "PRIVATE_DIAGNOSTIC", True):
        error = wire.TransportError("invalid_content_type", 200, "other_json", media)
        assert error.media_type_class is None
        assert "PRIVATE_DIAGNOSTIC" not in repr(error)


def metadata_child(send, media, timeout):
    send.send(("error", "invalid_content_type", 200, "other_json", media))
    send.close()


def test_actual_spawn_media_ipc_and_partial_pipe_cleanup():
    before = {p.pid for p in multiprocessing.active_children()}
    for media, expected in [("json", "json"), (["PRIVATE_DIAGNOSTIC"], None)]:
        with pytest.raises(wire.TransportError) as caught:
            wire._run_child(metadata_child, (media,), 3)
        assert caught.value.media_type_class == expected
        assert "PRIVATE_DIAGNOSTIC" not in repr(caught.value)
    start = time.monotonic()
    with pytest.raises(wire.TransportError):
        wire._run_child(partial_ipc, (), 0.7)
    assert time.monotonic() - start < 3
    assert {p.pid for p in multiprocessing.active_children()} == before
