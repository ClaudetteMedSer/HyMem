"""Isolated, single-turn Hermes Codex OAuth Responses transport.

The caller owns account admission, credential acquisition, and quota policy.
This module never reads credentials, refreshes tokens, or selects another model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import json
import multiprocessing
import ssl
import threading
import time
from typing import Any, Iterable


HOST = "chatgpt.com"
PATH = "/backend-api/codex/responses"
MODEL = "gpt-6-luna"
MAX_WALL_SECONDS = 120.0
MAX_WIRE_BYTES = 16_000_000
MAX_LINE_BYTES = 2_000_000
MAX_MEANINGFUL_EVENTS = 4096
MAX_OUTPUT_CHARS = 1_000_000
MAX_REQUEST_BYTES = 2_000_000


class TransportError(RuntimeError):
    """Only fixed, safe failure codes are exposed to the caller."""


def _fail(code: str) -> None:
    raise TransportError(code) from None


def _header_value(value: str) -> bool:
    return (type(value) is str and 1 <= len(value) <= 8192
            and all(33 <= ord(char) <= 126 for char in value))


@dataclass(frozen=True)
class Credentials:
    access_token: str = field(repr=False)
    account_id: str = field(repr=False)

    def __post_init__(self) -> None:
        if not _header_value(self.access_token) or not _header_value(self.account_id):
            _fail("invalid_credentials")


@dataclass(frozen=True)
class Completed:
    text: str = field(repr=False)
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    reasoning_output_tokens: int


def build_request(system: str, user: str, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    if type(system) is not str or type(user) is not str:
        _fail("invalid_request")
    request: dict[str, Any] = {
        "model": MODEL, "reasoning": {"effort": "low"}, "store": False,
        "stream": True, "instructions": system,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
    }
    if json_schema is not None:
        if not isinstance(json_schema, dict):
            _fail("invalid_request")
        try:
            schema_copy = json.loads(json.dumps(json_schema, allow_nan=False))
        except (TypeError, ValueError, OverflowError, RecursionError):
            _fail("invalid_request")
        request["text"] = {"format": {"type": "json_schema", "name": "response",
                                      "strict": True, "schema": schema_copy}}
    try:
        size = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (UnicodeError, ValueError, OverflowError, RecursionError):
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
    # The native Responses terminal envelope carries a model name; bind the
    # result to the one model admitted by this transport.
    if response.get("model") != MODEL:
        _fail("model_mismatch")
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
        kind = item.get("type")
        if kind == "reasoning":
            # Hidden reasoning is accounted for by output_tokens, never returned.
            continue
        if kind != "message" or item.get("role") != "assistant" or item.get("status", "completed") != "completed":
            _fail("unsupported_output")
        content = item.get("content")
        if not isinstance(content, list) or not content:
            _fail("invalid_output")
        channel = item.get("channel")
        if channel not in (None, "commentary", "final_answer"):
            _fail("unsupported_output")
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "output_text" or type(part.get("text")) is not str:
                _fail("unsupported_output")
            if channel == "commentary":
                continue
            value = part["text"]
            length += len(value)
            if length > MAX_OUTPUT_CHARS:
                _fail("output_limit")
            parts.append(value)
    if not parts:
        _fail("invalid_output")
    return Completed("".join(parts), input_tokens, output_tokens, total_tokens, cached, reasoning)


def parse_stream_events(events: Iterable[dict[str, Any]]) -> Completed:
    """Validate terminal events; deltas are intentionally not accumulated."""
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
        if kind in deltas:
            if completed is not None:
                _fail("event_after_completion")
            continue
        meaningful += 1
        if meaningful > MAX_MEANINGFUL_EVENTS:
            _fail("event_limit")
        if completed is not None:
            _fail("event_after_completion")
        if kind == "response.completed":
            completed = _completed(event.get("response"))
        elif kind in ("response.failed", "response.incomplete", "error"):
            _fail("response_failure")
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


def _sse_events(response: http.client.HTTPResponse) -> Iterable[dict[str, Any]]:
    wire_bytes = 0
    data: list[bytes] = []
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
                    obj = json.loads(b"\n".join(data))
                except (UnicodeError, ValueError):
                    _fail("invalid_event")
                if not isinstance(obj, dict):
                    _fail("invalid_event")
                yield obj
                data.clear()
            continue
        if line.startswith(b"data:"):
            value = line[5:].strip()
            if value == b"[DONE]":
                continue
            data.append(value)
    if data:
        _fail("truncated_stream")


def _request_once(credentials: Credentials, request: dict[str, Any], timeout: float) -> Completed:
    # http.client uses no environment proxy and follows no redirects.
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
        if "text/event-stream" not in response.getheader("Content-Type", "").lower():
            _fail("invalid_content_type")
        return parse_stream_events(_sse_events(response))
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
            _fail(value if type(value) is str and value in {
                "invalid_request", "invalid_event", "invalid_usage", "missing_usage",
                "incomplete_response", "invalid_output", "unsupported_output",
                "output_limit", "event_limit", "event_after_completion",
                "response_failure", "missing_completion", "wire_limit",
                "truncated_stream", "http_failure", "invalid_content_type",
                "auth_failure", "access_failure", "quota_failure", "model_mismatch",
            } else "transport_failure")
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
    """One admitted response; no retry, fallback, credential loading or refresh."""
    if type(credentials) is not Credentials:
        _fail("invalid_credentials")
    request = build_request(system, user, json_schema)
    return _run_child(_child, (credentials, request), timeout)
