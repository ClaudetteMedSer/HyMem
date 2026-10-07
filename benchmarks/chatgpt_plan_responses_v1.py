"""Single-turn, public Responses transport for an already admitted SIWC grant.

The caller owns credential storage, refresh, account/model admission, and quota
policy. This module performs one bounded request and never reads credentials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import json
import math
import multiprocessing
import ssl
import threading
import time
from typing import Any, Iterable


HOST = "api.openai.com"
PATH = "/v1/responses"
MODEL = "gpt-5.6-luna"
MAX_WALL_SECONDS = 120.0
MAX_WIRE_BYTES = 16_000_000
MAX_LINE_BYTES = 2_000_000
MAX_MEANINGFUL_EVENTS = 4096
MAX_OUTPUT_CHARS = 1_000_000
MAX_REQUEST_BYTES = 2_000_000

# Only documented machine codes may cross the transport boundary. Unknown
# provider messages, IDs, params and response bodies never do.
PROVIDER_CODES = frozenset({
    "subscription_sharing_user_not_eligible",
    "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_unsupported_capability",
    "subscription_sharing_route_not_supported",
    "subscription_sharing_invalid_user",
    "subscription_sharing_user_unavailable",
    "chatpass_v2_scope_not_authorized",
    "chatpass_v2_invalid_authorization_context",
})
_BODY_SHAPES = frozenset({"empty", "error_object", "detail", "other_json", "non_json", "oversized",
                          "sse_event"})
_LOCAL_CODES = frozenset({
    "invalid_credentials", "invalid_request", "request_limit", "invalid_timeout",
    "invalid_event", "invalid_usage", "missing_usage", "incomplete_response",
    "invalid_output", "unsupported_output", "output_limit", "event_limit",
    "event_after_completion", "response_failure", "missing_completion",
    "wire_limit", "truncated_stream", "http_failure", "invalid_content_type",
    "auth_failure", "access_failure", "quota_failure", "model_mismatch",
    "transport_failure", "timeout", "cleanup_failure",
})
_CHILD_CODES = _LOCAL_CODES | PROVIDER_CODES


class TransportError(RuntimeError):
    """Finite failure metadata; no provider text, identifiers, or secrets."""

    def __init__(self, code: str, http_status: int | None = None,
                 body_shape: str | None = None):
        safe_code = code if type(code) is str and code in _CHILD_CODES else "transport_failure"
        self.code = safe_code
        self.http_status = (http_status if type(http_status) is int and 100 <= http_status <= 599
                            else None)
        self.body_shape = (body_shape if type(body_shape) is str and body_shape in _BODY_SHAPES
                           else None)
        super().__init__(safe_code)

    def __repr__(self) -> str:
        return (f"TransportError(code={self.code!r}, http_status={self.http_status!r}, "
                f"body_shape={self.body_shape!r})")


def _fail(code: str, http_status: int | None = None,
          body_shape: str | None = None) -> None:
    raise TransportError(code, http_status, body_shape) from None


def _header_value(value: str) -> bool:
    return (type(value) is str and 1 <= len(value) <= 8192
            and all(33 <= ord(char) <= 126 for char in value))


@dataclass(frozen=True)
class Credentials:
    access_token: str = field(repr=False)

    def __post_init__(self) -> None:
        if not _header_value(self.access_token):
            _fail("invalid_credentials")


@dataclass(frozen=True)
class Completed:
    text: str = field(repr=False)
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    reasoning_output_tokens: int


def build_request(system: str, user: str,
                  json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    if type(system) is not str or type(user) is not str:
        _fail("invalid_request")
    request: dict[str, Any] = {
        "model": MODEL, "reasoning": {"effort": "low"}, "store": False,
        "stream": True, "instructions": system,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
    }
    if json_schema is not None:
        if type(json_schema) is not dict:
            _fail("invalid_request")
        try:
            schema_copy = json.loads(json.dumps(json_schema, allow_nan=False))
        except (TypeError, ValueError, OverflowError, RecursionError):
            _fail("invalid_request")
        request["text"] = {"format": {"type": "json_schema", "name": "response",
                                      "strict": True, "schema": schema_copy}}
    try:
        size = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (UnicodeError, TypeError, ValueError, OverflowError, RecursionError):
        _fail("invalid_request")
    if size > MAX_REQUEST_BYTES:
        _fail("request_limit")
    return request


def _count(value: Any) -> int:
    if type(value) is not int or value < 0:
        _fail("invalid_usage")
    return value


def _completed(response: Any) -> Completed:
    if not isinstance(response, dict) or response.get("status") != "completed":
        _fail("incomplete_response")
    if response.get("model") != MODEL:
        _fail("model_mismatch")
    if response.get("error") is not None or response.get("incomplete_details") is not None:
        _fail("response_failure")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        _fail("missing_usage")
    input_tokens = _count(usage.get("input_tokens"))
    output_tokens = _count(usage.get("output_tokens"))
    total_tokens = _count(usage.get("total_tokens"))
    input_details = usage.get("input_tokens_details", {})
    output_details = usage.get("output_tokens_details", {})
    if not isinstance(input_details, dict) or not isinstance(output_details, dict):
        _fail("invalid_usage")
    cached = _count(input_details.get("cached_tokens", 0))
    reasoning = _count(output_details.get("reasoning_tokens", 0))
    if (input_tokens == 0 or output_tokens == 0 or
            total_tokens != input_tokens + output_tokens or
            cached > input_tokens or reasoning > output_tokens):
        _fail("invalid_usage")
    output = response.get("output")
    if not isinstance(output, list) or not output:
        _fail("invalid_output")
    parts: list[str] = []
    length = 0
    for item in output:
        if not isinstance(item, dict):
            _fail("invalid_output")
        if item.get("type") == "reasoning":
            if item.get("status", "completed") != "completed":
                _fail("incomplete_response")
            continue
        if (item.get("type") != "message" or item.get("role") != "assistant" or
                item.get("status", "completed") != "completed"):
            _fail("unsupported_output")
        content = item.get("content")
        if not isinstance(content, list) or not content:
            _fail("invalid_output")
        if item.get("channel") not in (None, "final_answer"):
            _fail("unsupported_output")
        for part in content:
            if (not isinstance(part, dict) or part.get("type") != "output_text" or
                    type(part.get("text")) is not str):
                _fail("unsupported_output")
            value = part["text"]
            length += len(value)
            if length > MAX_OUTPUT_CHARS:
                _fail("output_limit")
            parts.append(value)
    if not parts or not any(parts):
        _fail("invalid_output")
    return Completed("".join(parts), input_tokens, output_tokens, total_tokens,
                     cached, reasoning)


def _provider_code(value: Any) -> str:
    code = value.get("code") if isinstance(value, dict) else None
    return code if type(code) is str and code in PROVIDER_CODES else "response_failure"


def parse_stream_events(events: Iterable[dict[str, Any]]) -> Completed:
    """Require a terminal completion; never accumulate streamed text deltas."""
    meaningful = 0
    completed: Completed | None = None
    allowed = frozenset({
        "response.created", "response.in_progress", "response.output_item.added",
        "response.output_item.done", "response.content_part.added",
        "response.content_part.done", "response.output_text.done",
        "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.done", "response.completed",
    })
    deltas = frozenset({"response.output_text.delta", "response.reasoning_text.delta",
                        "response.reasoning_summary_text.delta"})
    for event in events:
        if not isinstance(event, dict) or type(event.get("type")) is not str:
            _fail("invalid_event")
        kind = event["type"]
        if completed is not None:
            _fail("event_after_completion")
        if kind in deltas:
            continue
        meaningful += 1
        if meaningful > MAX_MEANINGFUL_EVENTS:
            _fail("event_limit")
        if kind == "response.completed":
            completed = _completed(event.get("response"))
        elif kind == "response.failed":
            response = event.get("response")
            _fail(_provider_code(response.get("error") if isinstance(response, dict) else None))
        elif kind == "error":
            _fail(_provider_code(event.get("error") if isinstance(event.get("error"), dict)
                                 else event))
        elif kind == "response.incomplete":
            _fail("incomplete_response")
        elif kind not in allowed:
            _fail("unsupported_output")
        elif kind in ("response.output_item.added", "response.output_item.done"):
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") not in ("message", "reasoning"):
                _fail("unsupported_output")
            if item.get("type") == "message" and "content" in item:
                content = item["content"]
                if (not isinstance(content, list) or any(
                        not isinstance(part, dict) or part.get("type") != "output_text"
                        for part in content)):
                    _fail("unsupported_output")
        elif kind in ("response.content_part.added", "response.content_part.done"):
            part = event.get("part")
            if not isinstance(part, dict) or part.get("type") not in ("output_text", "reasoning_text"):
                _fail("unsupported_output")
    if completed is None:
        _fail("missing_completion")
    return completed


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_event")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    _fail("invalid_event")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _fail("invalid_event")
    return number


def _decode_json(body: bytes) -> Any:
    return json.loads(body, object_pairs_hook=_unique_pairs,
                      parse_constant=_reject_constant, parse_float=_finite_float)


def _sse_events(response: http.client.HTTPResponse) -> Iterable[dict[str, Any]]:
    wire_bytes = 0
    data: list[bytes] = []
    seen_completion = False
    while True:
        line = response.readline(MAX_LINE_BYTES + 1)
        if not line:
            break
        wire_bytes += len(line)
        if wire_bytes > MAX_WIRE_BYTES or len(line) > MAX_LINE_BYTES:
            _fail("wire_limit")
        if line in (b"\n", b"\r\n"):
            if data:
                try:
                    obj = _decode_json(b"\n".join(data))
                except (UnicodeError, ValueError, RecursionError):
                    _fail("invalid_event")
                if not isinstance(obj, dict):
                    _fail("invalid_event")
                if obj.get("type") == "response.completed":
                    seen_completion = True
                yield obj
                data.clear()
            continue
        if line.startswith(b"data:"):
            value = line[5:].strip()
            if value == b"[DONE]":
                if data:
                    _fail("invalid_event")
                if not seen_completion:
                    _fail("missing_completion")
                continue
            data.append(value)
            if sum(len(part) for part in data) > MAX_LINE_BYTES:
                _fail("wire_limit")
    if data:
        _fail("truncated_stream")


def _status_code(status: Any) -> str:
    if status == 401:
        return "auth_failure"
    if status == 403:
        return "access_failure"
    if status == 429:
        return "quota_failure"
    return "http_failure"


def _http_error(response: http.client.HTTPResponse) -> None:
    status = response.status if type(response.status) is int else None
    body = response.read(MAX_WIRE_BYTES + 1)
    if len(body) > MAX_WIRE_BYTES:
        _fail(_status_code(status), status, "oversized")
    if not body:
        _fail(_status_code(status), status, "empty")
    try:
        value = _decode_json(body)
    except (TransportError, UnicodeError, ValueError, RecursionError):
        _fail(_status_code(status), status, "non_json")
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        _fail(_provider_code(value["error"]) if _provider_code(value["error"]) != "response_failure"
              else _status_code(status), status, "error_object")
    if isinstance(value, dict) and "detail" in value:
        _fail(_status_code(status), status, "detail")
    _fail(_status_code(status), status, "other_json")


def _request_once(credentials: Credentials, request: dict[str, Any], timeout: float) -> Completed:
    # http.client ignores environment proxies and does not follow redirects.
    connection = http.client.HTTPSConnection(HOST, 443, timeout=timeout,
                                            context=ssl.create_default_context())
    try:
        body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(body) > MAX_REQUEST_BYTES:
            _fail("request_limit")
        connection.request("POST", PATH, body=body, headers={
            "Authorization": "Bearer " + credentials.access_token,
            "Content-Type": "application/json", "Accept": "text/event-stream",
        })
        response = connection.getresponse()
        if response.status != 200:
            _http_error(response)
        content_type = response.getheader("Content-Type", "")
        if (type(content_type) is not str or len(content_type) > 8192 or
                content_type.split(";", 1)[0].strip().lower() != "text/event-stream"):
            _fail("invalid_content_type", 200)
        try:
            return parse_stream_events(_sse_events(response))
        except TransportError as exc:
            if exc.code in PROVIDER_CODES or exc.code in ("response_failure", "incomplete_response"):
                _fail(exc.code, 200, "sse_event")
            raise
    finally:
        connection.close()


def _child(send, credentials: Credentials, request: dict[str, Any], timeout: float) -> None:
    try:
        send.send(("ok", _request_once(credentials, request, timeout)))
    except BaseException as exc:
        safe = exc if type(exc) is TransportError else TransportError("transport_failure")
        try:
            send.send(("error", safe.code, safe.http_status, safe.body_shape))
        except (OSError, EOFError):
            pass
    finally:
        send.close()


def _run_child(target, args: tuple[Any, ...], timeout: float) -> Completed:
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or
            not 0 < timeout <= MAX_WALL_SECONDS):
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
            message = receive.recv()
        except (EOFError, OSError, ValueError):
            _fail("timeout" if expired.is_set() or time.monotonic() >= deadline
                  else "transport_failure")
        if expired.is_set() or time.monotonic() >= deadline:
            _fail("timeout")
        if type(message) is not tuple:
            _fail("transport_failure")
        if len(message) == 2 and message[0] == "ok" and type(message[1]) is Completed:
            return message[1]
        if len(message) == 4 and message[0] == "error":
            code, status, shape = message[1:]
            _fail(code if type(code) is str and code in _CHILD_CODES else "transport_failure",
                  status, shape)
        _fail("transport_failure")
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
             json_schema: dict[str, Any] | None = None,
             *, timeout: float = MAX_WALL_SECONDS) -> Completed:
    """One admitted SIWC request, with no credential read, retry or fallback."""
    if type(credentials) is not Credentials:
        _fail("invalid_credentials")
    request = build_request(system, user, json_schema)
    return _run_child(_child, (credentials, request), timeout)
