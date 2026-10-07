"""Offline, invented-response controls for SIWC failure attribution v2."""
from __future__ import annotations

import json
import multiprocessing
import time

import pytest

from benchmarks import chatgpt_plan_responses_v2 as transport


SECRET = "PRIVATE_TOKEN_BODY_HEADER_9f37"


class FakeResponse:
    def __init__(self, body=b"", *, status=200, content_type="text/event-stream"):
        self.body = body
        self.position = 0
        self.status = status
        self.content_type = content_type

    def getheader(self, name, default=None):
        return self.content_type if name == "Content-Type" else default

    def read(self, limit):
        result = self.body[self.position:self.position + limit]
        self.position += len(result)
        return result

    def readline(self, limit):
        end = self.body.find(b"\n", self.position)
        end = len(self.body) if end < 0 else end + 1
        end = min(end, self.position + limit)
        result = self.body[self.position:end]
        self.position = end
        return result


def invoke(monkeypatch, response):
    seen = {"count": 0, "closed": False}

    class FakeConnection:
        def __init__(self, host, port, *, timeout, context):
            assert (host, port) == ("api.openai.com", 443)
            assert timeout == 5

        def request(self, method, path, body, headers):
            seen["count"] += 1
            assert (method, path) == ("POST", "/v1/responses")
            assert headers == {"Authorization": "Bearer " + SECRET,
                               "Content-Type": "application/json", "Accept": "text/event-stream"}
            assert json.loads(body) == transport.build_request("s", "u")

        def getresponse(self):
            return response

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(transport.http.client, "HTTPSConnection", FakeConnection)
    monkeypatch.setattr(transport.ssl, "create_default_context", lambda: object())
    call = lambda: transport._request_once(
        transport.Credentials(SECRET), transport.build_request("s", "u"), 5)
    return seen, call


def failure(call, code, status=None, shape=None, media=None):
    with pytest.raises(transport.TransportError) as caught:
        call()
    exc = caught.value
    assert (exc.code, exc.http_status, exc.body_shape, exc.media_type_class) == (
        code, status, shape, media)
    for value in (str(exc), repr(exc)):
        assert SECRET not in value
    return exc


@pytest.mark.parametrize("header,body,shape,media", [
    (None, b"", "empty", "missing"),
    ("", b"[]", "other_json", "missing"),
    ("application/json", b"{\"detail\":\"" + SECRET.encode() + b"\"}", "detail", "json"),
    ("application/problem+json; charset=utf-8", b"{}", "other_json", "json"),
    ("text/html", b"<html>" + SECRET.encode(), "non_json", "html"),
    ("text/plain", b"bad", "non_json", "text"),
    ("image/png", b"bad", "non_json", "other"),
    ("application/json", b"{not json}", "non_json", "json"),
    ("application/json", b"{\"x\":1,\"x\":2}", "non_json", "json"),
    ("application/json", b"{\"x\":NaN}", "non_json", "json"),
    ("application/json", b"x" * (transport.MAX_ATTRIBUTION_BYTES + 1), "oversized", "json"),
    (SECRET + "\r\nInjected: secret", b"{}", "other_json", "invalid"),
    ("x" * 8193, b"{}", "other_json", "invalid"),
    ([SECRET], b"{}", "other_json", "invalid"),
])
def test_http_200_non_sse_is_bounded_failure(monkeypatch, header, body, shape, media):
    response = FakeResponse(body, content_type=header)
    seen, call = invoke(monkeypatch, response)
    failure(call, "invalid_content_type", 200, shape, media)
    assert seen == {"count": 1, "closed": True}
    assert response.position == min(len(body), transport.MAX_ATTRIBUTION_BYTES + 1)


@pytest.mark.parametrize("code,expected", [
    ("subscription_sharing_usage_limit_exceeded", "subscription_sharing_usage_limit_exceeded"),
    ("chatpass_v2_scope_not_authorized", "chatpass_v2_scope_not_authorized"),
    (SECRET, "invalid_content_type"),
])
def test_http_200_error_object_preserves_only_allowlisted_denial(monkeypatch, code, expected):
    body = json.dumps({"error": {"code": code, "message": SECRET,
                                 "param": SECRET, "request_id": SECRET}}).encode()
    _, call = invoke(monkeypatch, FakeResponse(body, content_type="application/json"))
    failure(call, expected, 200, "error_object", "json")


@pytest.mark.parametrize("body,shape", [
    ({"code": "subscription_sharing_usage_unavailable", "message": SECRET}, "other_json"),
    ({"detail": {"code": "subscription_sharing_usage_unavailable", "message": SECRET}}, "detail"),
])
def test_known_denial_in_other_json_shapes_is_terminal(monkeypatch, body, shape):
    _, call = invoke(monkeypatch, FakeResponse(json.dumps(body).encode(),
                                                content_type="application/json"))
    failure(call, "subscription_sharing_usage_unavailable", 200, shape, "json")


def test_non_200_retains_status_body_shape_and_media(monkeypatch):
    body = json.dumps({"error": {"code": "subscription_sharing_user_not_eligible",
                                 "message": SECRET}}).encode()
    _, call = invoke(monkeypatch, FakeResponse(body, status=403, content_type="application/json"))
    failure(call, "subscription_sharing_user_not_eligible", 403, "error_object", "json")


def test_sse_terminal_completion_and_denial(monkeypatch):
    completed = {"status": "completed", "model": transport.MODEL,
                 "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                 "output": [{"type": "message", "role": "assistant", "status": "completed",
                             "content": [{"type": "output_text", "text": "done"}]}]}
    body = b"data: " + json.dumps({"type": "response.completed", "response": completed}).encode() + b"\n\n"
    _, call = invoke(monkeypatch, FakeResponse(body, content_type="Text/Event-Stream; charset=utf-8"))
    assert call() == transport.Completed("done", 2, 3, 5, 0, 0)
    denial = {"type": "response.failed", "response": {"error": {
        "code": "subscription_sharing_usage_limit_exceeded", "message": SECRET}}}
    body = b"data: " + json.dumps(denial).encode() + b"\n\n"
    _, call = invoke(monkeypatch, FakeResponse(body))
    failure(call, "subscription_sharing_usage_limit_exceeded", 200, "sse_event", "sse")


def test_malicious_metadata_never_escapes():
    for media in ([SECRET], {SECRET: SECRET}, SECRET, object()):
        exc = transport.TransportError(SECRET, 999, [SECRET], media)
        assert (exc.code, exc.http_status, exc.body_shape, exc.media_type_class) == (
            "transport_failure", None, None, None)
        assert SECRET not in repr(exc)
    assert transport._media_type_class([SECRET]) == "invalid"
    failure(lambda: transport.Credentials("bad\r\n" + SECRET), "invalid_credentials")

    class HostileHeader:
        def __eq__(self, other):
            raise RuntimeError(SECRET)

    assert transport._media_type_class(HostileHeader()) == "invalid"
    altered = transport.TransportError("invalid_content_type", 200, "other_json", "json")
    altered.code = SECRET
    altered.body_shape = [SECRET]
    altered.media_type_class = {SECRET: SECRET}
    assert SECRET not in str(altered) + repr(altered)


def ipc_error_child(send, code, status, shape, media, timeout):
    send.send(("error", code, status, shape, media))
    send.close()


def ipc_success_child(send, timeout):
    send.send(("ok", transport.Completed("ok", 1, 1, 2, 0, 0)))
    send.close()


def hanging_child(send, timeout):
    time.sleep(timeout + 2)


def test_spawn_ipc_error_propagation_and_cleanup():
    before = {p.pid for p in multiprocessing.active_children()}
    failure(lambda: transport._run_child(ipc_error_child,
        ("subscription_sharing_usage_limit_exceeded", 200, "error_object", "json"), 5),
        "subscription_sharing_usage_limit_exceeded", 200, "error_object", "json")
    failure(lambda: transport._run_child(ipc_error_child,
        (SECRET, 999, [SECRET], [SECRET]), 5), "transport_failure")
    failure(lambda: transport._run_child(ipc_error_child,
        ("invalid_content_type", 200, "other_json", {SECRET: SECRET}), 5),
        "invalid_content_type", 200, "other_json")
    assert transport._run_child(ipc_success_child, (), 5).text == "ok"
    failure(lambda: transport._run_child(hanging_child, (), 0.2), "timeout")
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_child_serializes_finite_error_tuple(monkeypatch):
    class Sender:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

        def close(self):
            pass

    sender = Sender()
    def reject(*_args):
        raise transport.TransportError("invalid_content_type", 200, "non_json", "html")
    monkeypatch.setattr(transport, "_request_once", reject)
    transport._child(sender, transport.Credentials(SECRET), {}, 5)
    assert sender.messages == [("error", "invalid_content_type", 200, "non_json", "html")]
    def altered_error(*_args):
        exc = transport.TransportError("invalid_content_type", 200, "non_json", "html")
        exc.code = SECRET
        exc.media_type_class = [SECRET]
        raise exc
    monkeypatch.setattr(transport, "_request_once", altered_error)
    transport._child(sender, transport.Credentials(SECRET), {}, 5)
    assert sender.messages[-1] == ("error", "transport_failure", 200, "non_json", None)
