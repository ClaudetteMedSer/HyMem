"""Offline contract tests for the versioned native media-type repair."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import runpy
import time

import pytest

from benchmarks import hermes_codex_responses_v1 as v1
from benchmarks import hermes_codex_responses_v2 as transport


def envelope():
    return {
        "status": "completed", "model": transport.MODEL,
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
                  "input_tokens_details": {"cached_tokens": 3},
                  "output_tokens_details": {"reasoning_tokens": 2}},
        "output": [{"type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": " exact \n"}]}],
    }


class FakeResponse:
    def __init__(self, body=b"", *, status=200, content_type="application/json", mitigation=""):
        self.body = body
        self.position = 0
        self.status = status
        self.content_type = content_type
        self.mitigation = mitigation

    def getheader(self, name, default=None):
        return {"Content-Type": self.content_type,
                "cf-mitigated": self.mitigation}.get(name, default)

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


def invoke(monkeypatch, response, *, module=transport):
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

    monkeypatch.setattr(module.http.client, "HTTPSConnection", FakeConnection)
    monkeypatch.setattr(module.ssl, "create_default_context", lambda: "verified-context")
    request = module.build_request("system", "user")
    def call():
        return module._request_once(module.Credentials("secret", "acct"), request, 5)
    return seen, call


def assert_failure(call, code):
    with pytest.raises(transport.TransportError) as exc:
        call()
    assert str(exc.value) == code


def test_sse_completion_and_request_wire_unchanged(monkeypatch):
    event = {"type": "response.completed", "response": envelope()}
    body = b"data: " + json.dumps(event).encode() + b"\n\n"
    v2_seen, v2_call = invoke(monkeypatch, FakeResponse(body, content_type="Text/Event-Stream; charset=utf-8"))
    assert v2_call() == transport.Completed(" exact \n", 11, 7, 18, 3, 2)
    v1_seen, v1_call = invoke(monkeypatch, FakeResponse(body, content_type="text/event-stream"), module=v1)
    assert v1_call() == transport.Completed(" exact \n", 11, 7, 18, 3, 2)
    assert v2_seen["requests"] == v1_seen["requests"]
    assert v2_seen["requests"] == [(
        "POST", "/backend-api/codex/responses",
        json.dumps(transport.build_request("system", "user"), ensure_ascii=False,
                   allow_nan=False).encode("utf-8"),
        {"Authorization": "Bearer secret", "ChatGPT-Account-Id": "acct",
         "Content-Type": "application/json", "Accept": "text/event-stream"},
    )]
    assert v2_seen["host"] == v1_seen["host"] == "chatgpt.com"
    assert v2_seen["port"] == v1_seen["port"] == 443
    assert v2_seen["closed"] and v1_seen["closed"]


def test_json_complete_uses_same_terminal_validator(monkeypatch):
    response = FakeResponse(json.dumps(envelope()).encode(), content_type="Application/JSON; charset=utf-8")
    _, call = invoke(monkeypatch, response)
    assert call() == transport.Completed(" exact \n", 11, 7, 18, 3, 2)


@pytest.mark.parametrize("change,code", [
    (lambda value: value["usage"].update(output_tokens=0, total_tokens=11), "invalid_usage"),
    (lambda value: value.update(model="other"), "model_mismatch"),
    (lambda value: value["output"].append({"type": "function_call", "name": "x"}), "unsupported_output"),
    (lambda value: value.update(status="incomplete"), "incomplete_response"),
])
def test_json_rejects_invalid_completion(monkeypatch, change, code):
    value = envelope()
    change(value)
    _, call = invoke(monkeypatch, FakeResponse(json.dumps(value).encode()))
    assert_failure(call, code)


@pytest.mark.parametrize("body,code", [
    (b'{"status":"completed",', "invalid_json"),
    (b'{"status":"completed","status":"completed"}', "invalid_json"),
    (b'{"usage":{"input_tokens":NaN}}', "invalid_json"),
    (b'{"usage":{"input_tokens":1e999}}', "invalid_json"),
    (b'not-json-secret-message', "invalid_json"),
])
def test_json_syntax_duplicate_and_nonfinite_rejected(monkeypatch, body, code):
    _, call = invoke(monkeypatch, FakeResponse(body))
    assert_failure(call, code)


def test_json_wire_limit(monkeypatch):
    _, call = invoke(monkeypatch, FakeResponse(b"x" * (transport.MAX_WIRE_BYTES + 1)))
    assert_failure(call, "wire_limit")


@pytest.mark.parametrize("content_type,mitigation,code", [
    ("text/html; charset=UTF-8", "", "html_response"),
    ("application/xhtml+xml", "", "html_response"),
    ("application/json", "challenge", "access_challenge"),
    ("text/html", " Challenge ", "access_challenge"),
    ("application/problem+json", "", "unsupported_media_type"),
    ("text/event-stream-garbage", "", "unsupported_media_type"),
])
def test_media_type_and_challenge_terminal(monkeypatch, content_type, mitigation, code):
    seen, call = invoke(monkeypatch, FakeResponse(b"SECRET BODY", content_type=content_type,
                                                 mitigation=mitigation))
    assert_failure(call, code)
    assert len(seen["requests"]) == 1
    assert seen["closed"]


@pytest.mark.parametrize("status,code", [(401, "auth_failure"), (403, "access_failure"),
                                          (429, "quota_failure"), (302, "http_failure")])
def test_http_status_precedes_media_and_never_retries(monkeypatch, status, code):
    seen, call = invoke(monkeypatch, FakeResponse(b"SECRET BODY", status=status,
                                                 content_type="text/html", mitigation="challenge"))
    assert_failure(call, code)
    assert len(seen["requests"]) == 1


@pytest.mark.parametrize("provider_code,expected", [
    ("invalid_api_key", "auth_failure"),
    ("permission_denied", "access_failure"),
    ("insufficient_quota", "quota_failure"),
    ("new_unknown_code", "response_failure"),
    (None, "response_failure"),
])
def test_json_error_envelope_exports_only_fixed_code(monkeypatch, provider_code, expected):
    body = json.dumps({"error": {"code": provider_code,
                                 "message": "private provider message SECRET"}}).encode()
    seen, call = invoke(monkeypatch, FakeResponse(body))
    assert_failure(call, expected)
    assert len(seen["requests"]) == 1


def hanging_child(send, timeout):
    time.sleep(timeout + 3)


def successful_child(send, timeout):
    send.send(("ok", transport.Completed("ok", 1, 2, 3, 0, 0)))
    send.close()


def failure_child(send, code, timeout):
    send.send(("error", code))
    send.close()


def test_v1_watchdog_contract_preserved():
    before = {process.pid for process in multiprocessing.active_children()}
    start = time.monotonic()
    assert_failure(lambda: transport._run_child(hanging_child, (), 0.2), "timeout")
    assert time.monotonic() - start < 3
    assert {process.pid for process in multiprocessing.active_children()} == before
    assert transport._run_child(successful_child, (), 5).text == "ok"
    assert {process.pid for process in multiprocessing.active_children()} == before


@pytest.mark.parametrize("child_code,parent_code", [
    ("html_response", "html_response"),
    ("access_challenge", "access_challenge"),
    ("unsupported_media_type", "unsupported_media_type"),
    ("invalid_json", "invalid_json"),
    ("SECRET provider message", "transport_failure"),
])
def test_child_parent_only_preserves_whitelisted_codes(child_code, parent_code):
    assert_failure(lambda: transport._run_child(failure_child, (child_code,), 5), parent_code)


def test_v1_source_tamper_fails_before_import_or_admission(monkeypatch):
    original_read_bytes = Path.read_bytes
    v1_path = Path(transport.__file__).with_name("hermes_codex_responses_v1.py")

    def tampered_read_bytes(path):
        if path == v1_path:
            return b"tampered v1 source"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tampered_read_bytes)
    with pytest.raises(RuntimeError, match="^v1_source_mismatch$"):
        runpy.run_path(str(Path(transport.__file__)))
    monkeypatch.setattr(transport, "_run_child", lambda *args: pytest.fail("admission attempted"))
    with pytest.raises(RuntimeError, match="^v1_source_mismatch$"):
        transport.complete(transport.Credentials("secret", "acct"), "s", "u")
