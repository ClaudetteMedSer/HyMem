"""Versioned SIWC Responses transport with finite HTTP wire attribution.

Only a failed HTTP 200 non-SSE response gains diagnostic metadata. It remains a
failure even when its body looks like SSE. Credentials and provider text never
cross the child IPC error boundary.
"""
from __future__ import annotations

import email.errors
import http.client
import io
import json
import math
import multiprocessing
import ssl
import threading
import time
from typing import Any, Iterable

from benchmarks import chatgpt_plan_responses_v1 as _v1
from benchmarks import chatgpt_plan_responses_v2 as _v2


HOST = _v2.HOST
PATH = _v2.PATH
MODEL = _v2.MODEL
MAX_WALL_SECONDS = _v2.MAX_WALL_SECONDS
MAX_REQUEST_BYTES = _v2.MAX_REQUEST_BYTES
MAX_ATTRIBUTION_BYTES = _v2.MAX_ATTRIBUTION_BYTES
PROVIDER_CODES = _v2.PROVIDER_CODES
Completed = _v2.Completed
Credentials = _v2.Credentials

_CHILD_CODES = _v1._CHILD_CODES
_BODY_SHAPES = _v2._BODY_SHAPES
_MEDIA_CLASSES = _v2._MEDIA_CLASSES
_OBS_KEYS = frozenset({"header_defect", "parsed_header_count", "transfer_encoding",
                       "content_encoding", "body_prefix", "body_bytes", "body_truncated",
                       "sse_validation"})
_HEADER_DEFECTS = frozenset({"none", "missing_separator", "first_continuation",
                             "other", "unknown"})
_TRANSFER_ENCODINGS = frozenset({"missing", "chunked", "identity", "other", "invalid"})
_CONTENT_ENCODINGS = frozenset({"missing", "identity", "gzip", "deflate", "br",
                                "other", "invalid"})
_BODY_PREFIXES = frozenset({"empty", "sse_prefix", "html_prefix", "http_prefix",
                            "json_prefix", "text", "binary", "unknown"})
_SSE_VALIDATIONS = frozenset({"not_checked", "validated_completion", "invalid", "truncated"})


def _sanitize_observation(value: Any) -> dict[str, Any] | None:
    """Accept only the complete finite schema, copying it at every boundary."""
    if type(value) is not dict or value.keys() != _OBS_KEYS:
        return None
    try:
        defect = value["header_defect"]
        count = value["parsed_header_count"]
        transfer = value["transfer_encoding"]
        content = value["content_encoding"]
        prefix = value["body_prefix"]
        size = value["body_bytes"]
        truncated = value["body_truncated"]
        validation = value["sse_validation"]
        if not (type(defect) is str and defect in _HEADER_DEFECTS and
                (count is None or type(count) is int and 0 <= count <= 100) and
                type(transfer) is str and transfer in _TRANSFER_ENCODINGS and
                type(content) is str and content in _CONTENT_ENCODINGS and
                type(prefix) is str and prefix in _BODY_PREFIXES and
                type(size) is int and 0 <= size <= MAX_ATTRIBUTION_BYTES + 1 and
                type(truncated) is bool and type(validation) is str and
                validation in _SSE_VALIDATIONS):
            return None
        if (truncated != (size > MAX_ATTRIBUTION_BYTES) or
                (prefix == "empty") != (size == 0)):
            return None
        if prefix != "sse_prefix" and validation != "not_checked":
            return None
        if prefix == "sse_prefix" and validation not in (
                {"truncated"} if truncated else {"validated_completion", "invalid"}):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return {"header_defect": defect, "parsed_header_count": count,
            "transfer_encoding": transfer, "content_encoding": content,
            "body_prefix": prefix, "body_bytes": size, "body_truncated": truncated,
            "sse_validation": validation}


class TransportError(RuntimeError):
    """Finite metadata only; malformed observations are discarded wholesale."""

    def __init__(self, code: str, http_status: int | None = None,
                 body_shape: str | None = None, media_type_class: str | None = None,
                 wire_observation: dict[str, Any] | None = None):
        self.code = code if type(code) is str and code in _CHILD_CODES else "transport_failure"
        self.http_status = (http_status if type(http_status) is int and 100 <= http_status <= 599
                            else None)
        self.body_shape = (body_shape if type(body_shape) is str and body_shape in _BODY_SHAPES
                           else None)
        self.media_type_class = (media_type_class if type(media_type_class) is str and
                                 media_type_class in _MEDIA_CLASSES else None)
        self.wire_observation = _sanitize_observation(wire_observation)
        super().__init__(self.code)

    def __repr__(self) -> str:
        safe = TransportError(self.code, self.http_status, self.body_shape,
                              self.media_type_class, self.wire_observation)
        return (f"TransportError(code={safe.code!r}, http_status={safe.http_status!r}, "
                f"body_shape={safe.body_shape!r}, media_type_class={safe.media_type_class!r}, "
                f"wire_observation={safe.wire_observation!r})")


def _fail(code: str, http_status: int | None = None, body_shape: str | None = None,
          media_type_class: str | None = None,
          wire_observation: dict[str, Any] | None = None) -> None:
    raise TransportError(code, http_status, body_shape, media_type_class,
                         wire_observation) from None


def _translate(exc: _v1.TransportError, *, status: int | None = None,
               shape: str | None = None, media: str | None = None) -> None:
    _fail(exc.code, status if status is not None else exc.http_status,
          shape if shape is not None else exc.body_shape, media)


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


def _header_defect(headers: Any) -> str:
    try:
        defects = headers.defects
        if type(defects) is not list:
            return "unknown"
        if any(isinstance(item, email.errors.MissingHeaderBodySeparatorDefect)
               for item in defects):
            return "missing_separator"
        if any(isinstance(item, email.errors.FirstHeaderLineIsContinuationDefect)
               for item in defects):
            return "first_continuation"
        return "other" if defects else "none"
    except BaseException:
        return "unknown"


def _encoding_class(value: Any, allowed: frozenset[str]) -> str:
    if value is None:
        return "missing"
    if type(value) is not str or len(value) > 8192 or not value.isascii():
        return "invalid"
    if value == "":
        return "missing"
    token = value.strip().lower()
    if not token or any(not (char.isalnum() or char in "!#$%&'*+-.^_`|~")
                        for char in token):
        return "invalid"
    return token if token in allowed else "other"


def _response_observation(response: http.client.HTTPResponse, body: bytes) -> dict[str, Any]:
    try:
        headers = response.headers
        count = min(len(headers), 100) if headers is not None else None
        if type(count) is not int or count < 0:
            count = None
    except BaseException:
        headers, count = None, None
    try:
        transfer = response.getheader("Transfer-Encoding", None)
    except BaseException:
        transfer = object()
    try:
        content = response.getheader("Content-Encoding", None)
    except BaseException:
        content = object()
    prefix = _body_prefix(body)
    return {"header_defect": _header_defect(headers), "parsed_header_count": count,
            "transfer_encoding": _encoding_class(transfer, frozenset({"chunked", "identity"})),
            "content_encoding": _encoding_class(content, frozenset({"identity", "gzip", "deflate", "br"})),
            "body_prefix": prefix, "body_bytes": len(body),
            "body_truncated": len(body) > MAX_ATTRIBUTION_BYTES,
            "sse_validation": _sse_validation(body) if prefix == "sse_prefix" else "not_checked"}


def _sse_validation(body: bytes) -> str:
    if len(body) > MAX_ATTRIBUTION_BYTES:
        return "truncated"
    try:
        _v1.parse_stream_events(_v1._sse_events(io.BytesIO(body)))
    except (_v1.TransportError, UnicodeError, ValueError, RecursionError):
        return "invalid"
    return "validated_completion"


def _body_prefix(body: bytes) -> str:
    if not body:
        return "empty"
    # Prefix analysis is lexical only: it never admits a response as SSE.
    prefix = body[:256]
    if prefix.startswith(b"\xef\xbb\xbf"):
        prefix = prefix[3:]
    prefix = prefix.lstrip(b" \t\r\n")
    lower = prefix.lower()
    if prefix.startswith((b"data:", b"event:", b":")):
        return "sse_prefix"
    if lower.startswith((b"<!doctype html", b"<html")):
        return "html_prefix"
    if prefix.startswith(b"HTTP/"):
        return "http_prefix"
    if prefix.startswith((b"{", b"[")):
        return "json_prefix"
    try:
        sample = prefix.decode("utf-8")
    except UnicodeError:
        return "binary"
    return "text" if all(char.isprintable() or char in "\r\n\t" for char in sample) else "binary"


def _non_sse_200(response: http.client.HTTPResponse, media: str) -> None:
    body = response.read(MAX_ATTRIBUTION_BYTES + 1)
    if type(body) is not bytes:
        _fail("transport_failure", 200, media_type_class=media)
    observation = _response_observation(response, body)
    code, shape = _non_sse_classification(body)
    del body
    _fail(code, 200, shape, media, observation)


def _non_sse_classification(body: bytes) -> tuple[str, str]:
    """Return only finite labels, letting raw body and parsed text fall away."""
    if len(body) > MAX_ATTRIBUTION_BYTES:
        return "invalid_content_type", "oversized"
    if not body:
        return "invalid_content_type", "empty"
    try:
        value = _v1._decode_json(body)
    except (_v1.TransportError, UnicodeError, ValueError, RecursionError):
        return "invalid_content_type", "non_json"
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        code = _v1._provider_code(value["error"])
        return code if code in PROVIDER_CODES else "invalid_content_type", "error_object"
    if isinstance(value, dict) and "detail" in value:
        code = _v1._provider_code(value["detail"])
        return code if code in PROVIDER_CODES else "invalid_content_type", "detail"
    code = _v1._provider_code(value)
    return code if code in PROVIDER_CODES else "invalid_content_type", "other_json"


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
        media = _v2._response_media(response)
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
                               exc.media_type_class, exc.wire_observation)
                if type(exc) is TransportError else TransportError("transport_failure"))
        try:
            send.send(("error", safe.code, safe.http_status, safe.body_shape,
                       safe.media_type_class, _sanitize_observation(safe.wire_observation)))
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
        if len(message) == 6 and message[0] == "error":
            code, status, shape, media, observation = message[1:]
            _fail(code, status, shape, media, _sanitize_observation(observation))
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
