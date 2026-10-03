"""Versioned SIWC Responses transport with bounded failure attribution.

Request construction and SSE validation are inherited from the immutable v1
transport. This version never treats a non-SSE HTTP 200 body as a completion.
The caller owns credential storage, refresh, and admission.
"""
from __future__ import annotations

import http.client
from dataclasses import dataclass, field
import json
import math
import multiprocessing
import re
import ssl
import threading
import time
from typing import Any, Iterable

from benchmarks import chatgpt_plan_responses_v1 as _v1


HOST = _v1.HOST
PATH = _v1.PATH
MODEL = _v1.MODEL
MAX_WALL_SECONDS = _v1.MAX_WALL_SECONDS
MAX_WIRE_BYTES = _v1.MAX_WIRE_BYTES
MAX_LINE_BYTES = _v1.MAX_LINE_BYTES
MAX_MEANINGFUL_EVENTS = _v1.MAX_MEANINGFUL_EVENTS
MAX_OUTPUT_CHARS = _v1.MAX_OUTPUT_CHARS
MAX_REQUEST_BYTES = _v1.MAX_REQUEST_BYTES
MAX_ATTRIBUTION_BYTES = 64 * 1024
PROVIDER_CODES = _v1.PROVIDER_CODES
Completed = _v1.Completed

_BODY_SHAPES = frozenset({"empty", "error_object", "detail", "other_json",
                          "non_json", "oversized", "sse_event"})
_MEDIA_CLASSES = frozenset({"missing", "sse", "json", "html", "text", "other", "invalid"})
_CHILD_CODES = _v1._CHILD_CODES
_MIME_TOKEN = re.compile(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+\Z")


class TransportError(RuntimeError):
    """Only finite metadata may cross the transport or child boundary."""

    def __init__(self, code: str, http_status: int | None = None,
                 body_shape: str | None = None, media_type_class: str | None = None):
        self.code = code if type(code) is str and code in _CHILD_CODES else "transport_failure"
        self.http_status = (http_status if type(http_status) is int and 100 <= http_status <= 599
                            else None)
        self.body_shape = (body_shape if type(body_shape) is str and body_shape in _BODY_SHAPES
                           else None)
        self.media_type_class = (media_type_class if type(media_type_class) is str and
                                 media_type_class in _MEDIA_CLASSES else None)
        super().__init__(self.code)

    def __repr__(self) -> str:
        safe = TransportError(self.code, self.http_status, self.body_shape,
                              self.media_type_class)
        return (f"TransportError(code={safe.code!r}, http_status={safe.http_status!r}, "
                f"body_shape={safe.body_shape!r}, media_type_class={safe.media_type_class!r})")


def _fail(code: str, http_status: int | None = None,
          body_shape: str | None = None, media_type_class: str | None = None) -> None:
    raise TransportError(code, http_status, body_shape, media_type_class) from None


def _translate(exc: _v1.TransportError, *, status: int | None = None,
               shape: str | None = None, media: str | None = None) -> None:
    _fail(exc.code, status if status is not None else exc.http_status,
          shape if shape is not None else exc.body_shape, media)


@dataclass(frozen=True)
class Credentials:
    access_token: str = field(repr=False)

    def __post_init__(self) -> None:
        if not _v1._header_value(self.access_token):
            _fail("invalid_credentials")


def _media_type_class(value: Any) -> str:
    """Classify a bounded header without retaining or exporting its contents."""
    if value is None:
        return "missing"
    if type(value) is not str:
        return "invalid"
    if value == "":
        return "missing"
    if len(value) > 8192 or not value.isascii():
        return "invalid"
    if any(ord(char) < 32 or ord(char) > 126 for char in value):
        return "invalid"
    media_type = value.split(";", 1)[0].strip().lower()
    if _MIME_TOKEN.fullmatch(media_type) is None:
        return "invalid"
    if media_type == "text/event-stream":
        return "sse"
    if media_type == "application/json" or media_type.endswith("+json"):
        return "json"
    if media_type == "text/html":
        return "html"
    if media_type == "text/plain":
        return "text"
    return "other"


def _response_media(response: http.client.HTTPResponse) -> str:
    try:
        return _media_type_class(response.getheader("Content-Type", None))
    except BaseException:
        return "invalid"


def build_request(system: str, user: str,
                  json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return _v1.build_request(system, user, json_schema)
    except _v1.TransportError as exc:
        _translate(exc)


def parse_stream_events(events: Iterable[dict[str, Any]]) -> Completed:
    try:
        return _v1.parse_stream_events(events)
    except _v1.TransportError as exc:
        _translate(exc)


def _non_sse_200(response: http.client.HTTPResponse, media: str) -> None:
    body = response.read(MAX_ATTRIBUTION_BYTES + 1)
    if type(body) is not bytes:
        _fail("transport_failure", 200, media_type_class=media)
    if len(body) > MAX_ATTRIBUTION_BYTES:
        _fail("invalid_content_type", 200, "oversized", media)
    if not body:
        _fail("invalid_content_type", 200, "empty", media)
    try:
        value = _v1._decode_json(body)
    except (_v1.TransportError, UnicodeError, ValueError, RecursionError):
        _fail("invalid_content_type", 200, "non_json", media)
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        code = _v1._provider_code(value["error"])
        _fail(code if code in PROVIDER_CODES else "invalid_content_type",
              200, "error_object", media)
    if isinstance(value, dict) and "detail" in value:
        detail = value["detail"]
        code = _v1._provider_code(detail)
        _fail(code if code in PROVIDER_CODES else "invalid_content_type",
              200, "detail", media)
    code = _v1._provider_code(value)
    _fail(code if code in PROVIDER_CODES else "invalid_content_type",
          200, "other_json", media)


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
        media = _response_media(response)
        if response.status != 200:
            try:
                _v1._http_error(response)
            except _v1.TransportError as exc:
                _translate(exc, media=media)
        if media != "sse":
            _non_sse_200(response, media)
        try:
            return _v1.parse_stream_events(_v1._sse_events(response))
        except _v1.TransportError as exc:
            _translate(exc, status=200, shape="sse_event", media=media)
    finally:
        connection.close()


def _child(send, credentials: Credentials, request: dict[str, Any], timeout: float) -> None:
    try:
        send.send(("ok", _request_once(credentials, request, timeout)))
    except BaseException as exc:
        safe = (TransportError(exc.code, exc.http_status, exc.body_shape,
                               exc.media_type_class) if type(exc) is TransportError
                else TransportError("transport_failure"))
        try:
            send.send(("error", safe.code, safe.http_status, safe.body_shape,
                       safe.media_type_class))
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
        if len(message) == 5 and message[0] == "error":
            code, status, shape, media = message[1:]
            _fail(code, status, shape, media)
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
