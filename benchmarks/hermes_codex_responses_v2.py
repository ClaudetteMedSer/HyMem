"""Versioned media-type compatibility for the isolated Hermes OAuth transport.

The v1 request, SSE parser, completion checks, and process isolation are reused.
Only response media-type classification and bounded JSON decoding differ.
"""
from __future__ import annotations

import http.client
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import ssl
import threading
import time
from typing import Any

_V1_SHA256 = "fa88c3c2aa9cf042cb0110e5d9f3eea4ffd951270daf57d27dbe972bd2105e94"


def _verify_v1_source() -> None:
    try:
        source = Path(__file__).with_name("hermes_codex_responses_v1.py").read_bytes()
    except OSError:
        raise RuntimeError("v1_source_mismatch") from None
    if hashlib.sha256(source).hexdigest() != _V1_SHA256:
        raise RuntimeError("v1_source_mismatch")


_verify_v1_source()
from benchmarks import hermes_codex_responses_v1 as v1


HOST = v1.HOST
PATH = v1.PATH
MODEL = v1.MODEL
MAX_WALL_SECONDS = v1.MAX_WALL_SECONDS
MAX_WIRE_BYTES = v1.MAX_WIRE_BYTES
MAX_LINE_BYTES = v1.MAX_LINE_BYTES
MAX_MEANINGFUL_EVENTS = v1.MAX_MEANINGFUL_EVENTS
MAX_OUTPUT_CHARS = v1.MAX_OUTPUT_CHARS
MAX_REQUEST_BYTES = v1.MAX_REQUEST_BYTES
TransportError = v1.TransportError
Credentials = v1.Credentials
Completed = v1.Completed
build_request = v1.build_request
parse_stream_events = v1.parse_stream_events
_completed = v1._completed
_sse_events = v1._sse_events
_fail = v1._fail

# Only these exact provider error codes may become one of our fixed failure codes.
# Never export the error message, raw code, response body, or header value.
_JSON_ERROR_CODES = {
    "invalid_api_key": "auth_failure",
    "invalid_token": "auth_failure",
    "token_expired": "auth_failure",
    "unauthorized": "auth_failure",
    "permission_denied": "access_failure",
    "access_denied": "access_failure",
    "insufficient_quota": "quota_failure",
    "rate_limit_exceeded": "quota_failure",
}

_CHILD_FAILURES = frozenset({
    "invalid_request", "invalid_event", "invalid_usage", "missing_usage",
    "incomplete_response", "invalid_output", "unsupported_output",
    "output_limit", "event_limit", "event_after_completion",
    "response_failure", "missing_completion", "wire_limit",
    "truncated_stream", "http_failure", "invalid_content_type",
    "auth_failure", "access_failure", "quota_failure", "model_mismatch",
    "html_response", "access_challenge", "unsupported_media_type", "invalid_json",
})


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_json")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    _fail("invalid_json")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _fail("invalid_json")
    return number


def _json_response(response: http.client.HTTPResponse) -> Completed:
    body = response.read(MAX_WIRE_BYTES + 1)
    if len(body) > MAX_WIRE_BYTES:
        _fail("wire_limit")
    try:
        value = json.loads(body, object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant, parse_float=_finite_float)
    except (UnicodeError, ValueError, RecursionError):
        _fail("invalid_json")
    if not isinstance(value, dict):
        _fail("invalid_json")
    if "error" in value:
        error = value["error"]
        code = error.get("code") if isinstance(error, dict) else None
        _fail(_JSON_ERROR_CODES.get(code, "response_failure")
              if type(code) is str else "response_failure")
    return _completed(value)


def _request_once(credentials: Credentials, request: dict[str, Any], timeout: float) -> Completed:
    # Identical v1 origin, request bytes and headers; http.client does not redirect.
    connection = http.client.HTTPSConnection(HOST, 443, timeout=timeout,
                                            context=ssl.create_default_context())
    try:
        body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        connection.request("POST", PATH, body=body, headers={
            "Authorization": "Bearer " + credentials.access_token,
            "ChatGPT-Account-Id": credentials.account_id,
            "Content-Type": "application/json", "Accept": "text/event-stream",
        })
        response = connection.getresponse()
        if response.status != 200:
            if response.status == 401:
                _fail("auth_failure")
            if response.status == 403:
                _fail("access_failure")
            if response.status == 429:
                _fail("quota_failure")
            _fail("http_failure")
        mitigation = response.getheader("cf-mitigated", "")
        if (type(mitigation) is str and len(mitigation) <= 8192 and
                mitigation.strip().lower() == "challenge"):
            _fail("access_challenge")
        content_type = response.getheader("Content-Type", "")
        if type(content_type) is not str or len(content_type) > 8192:
            _fail("unsupported_media_type")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type == "text/event-stream":
            return parse_stream_events(_sse_events(response))
        if media_type == "application/json":
            return _json_response(response)
        if media_type in ("text/html", "application/xhtml+xml"):
            _fail("html_response")
        _fail("unsupported_media_type")
    finally:
        connection.close()


def _child(send, credentials: Credentials, request: dict[str, Any], timeout: float) -> None:
    try:
        result = _request_once(credentials, request, timeout)
        send.send(("ok", result))
    except BaseException as exc:
        code = str(exc) if type(exc) is TransportError else "transport_failure"
        try:
            send.send(("error", code))
        except (OSError, EOFError):
            pass
    finally:
        send.close()


def _run_child(target, args: tuple[Any, ...], timeout: float) -> Any:
    if type(timeout) not in (int, float) or not 0 < timeout <= MAX_WALL_SECONDS:
        _fail("invalid_timeout")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(send, *args, timeout), daemon=True)
    deadline = time.monotonic() + timeout
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        try:
            if process.pid is not None and process.is_alive():
                process.kill()
        except (OSError, ValueError):
            pass

    watchdog = threading.Timer(timeout, expire)
    watchdog.daemon = True
    try:
        watchdog.start()
        process.start()
        send.close()
        if not receive.poll(max(0, deadline - time.monotonic())):
            _fail("timeout")
        try:
            status, value = receive.recv()
        except (EOFError, OSError, ValueError):
            if expired.is_set() or time.monotonic() >= deadline:
                _fail("timeout")
            _fail("transport_failure")
        if expired.is_set() or time.monotonic() >= deadline:
            _fail("timeout")
        if status != "ok":
            _fail(value if type(value) is str and value in _CHILD_FAILURES
                  else "transport_failure")
        if type(value) is not Completed:
            _fail("transport_failure")
        return value
    except TransportError:
        raise
    except (OSError, ValueError, RuntimeError):
        _fail("transport_failure")
    finally:
        watchdog.cancel()
        receive.close()
        send.close()
        if process.pid is not None:
            process.join(timeout=0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            if process.is_alive():
                _fail("cleanup_failure")


def complete(credentials: Credentials, system: str, user: str,
             json_schema: dict[str, Any] | None = None, *, timeout: float = MAX_WALL_SECONDS) -> Completed:
    """One admitted response with v1 credentials and request, without retry."""
    _verify_v1_source()
    if type(credentials) is not Credentials:
        _fail("invalid_credentials")
    request = build_request(system, user, json_schema)
    return _run_child(_child, (credentials, request), timeout)
