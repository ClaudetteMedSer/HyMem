"""Bounded SIWC Responses stream with empty terminal output reconstruction.

The caller owns admission, credentials, and quota policy. No provider text or
identifiers are returned in errors. Earlier transport versions stay immutable.
"""
from __future__ import annotations

import http.client
import json
import math
import multiprocessing
import ssl
import threading
import time
from typing import Any, Iterable

from benchmarks import chatgpt_plan_responses_v1 as _v1
from benchmarks import chatgpt_plan_responses_v2 as _v2
from benchmarks import chatgpt_plan_responses_v3 as _v3


HOST, PATH, MODEL = _v1.HOST, _v1.PATH, _v1.MODEL
MAX_WALL_SECONDS = _v1.MAX_WALL_SECONDS
MAX_WIRE_BYTES = _v1.MAX_WIRE_BYTES
MAX_LINE_BYTES = _v1.MAX_LINE_BYTES
MAX_MEANINGFUL_EVENTS = _v1.MAX_MEANINGFUL_EVENTS
MAX_OUTPUT_CHARS = _v1.MAX_OUTPUT_CHARS
MAX_REQUEST_BYTES = _v1.MAX_REQUEST_BYTES
MAX_ATTRIBUTION_BYTES = _v3.MAX_ATTRIBUTION_BYTES
PROVIDER_CODES = _v1.PROVIDER_CODES
Completed, Credentials = _v1.Completed, _v2.Credentials

_EVENT_TYPES = frozenset({
    "response.created", "response.in_progress", "response.queued",
    "response.output_item.added", "response.output_item.done",
    "response.content_part.added", "response.content_part.done",
    "response.output_text.delta", "response.output_text.done",
    "response.reasoning_text.delta", "response.reasoning_text.done",
    "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
    "response.refusal.delta", "response.refusal.done",
    "response.function_call_arguments.delta", "response.function_call_arguments.done",
    "response.output_text.annotation.added", "response.completed",
    "response.failed", "response.incomplete", "error",
})
_OBS_KEYS = frozenset({"event_type", "terminal_status", "terminal_model_matches",
                       "terminal_output_kind", "terminal_channel", "terminal_content_kind",
                       "terminal_output_state", "finalized_item_count", "output_reconstructed"})
_STATUSES = frozenset({"missing", "completed", "incomplete", "failed", "other"})
_OUTPUT_KINDS = frozenset({"missing", "message", "reasoning", "mixed", "other"})
_CHANNELS = frozenset({"missing", "final", "final_answer", "analysis",
                       "commentary", "mixed", "other"})
_CONTENT_KINDS = frozenset({"missing", "output_text", "refusal", "reasoning_text",
                            "mixed", "other"})
_OUTPUT_STATES = frozenset({"unseen", "missing", "null", "empty", "list", "invalid"})


def _sanitize_stream_observation(value: Any) -> dict[str, Any] | None:
    if type(value) is not dict or value.keys() != _OBS_KEYS:
        return None
    event = value["event_type"]
    status = value["terminal_status"]
    model = value["terminal_model_matches"]
    output = value["terminal_output_kind"]
    channel = value["terminal_channel"]
    content = value["terminal_content_kind"]
    output_state = value["terminal_output_state"]
    finalized_count = value["finalized_item_count"]
    reconstructed = value["output_reconstructed"]
    if not (type(event) is str and event in _EVENT_TYPES | {"unknown"} and
            type(status) is str and status in _STATUSES and
            (model is None or type(model) is bool) and
            type(output) is str and output in _OUTPUT_KINDS and
            type(channel) is str and channel in _CHANNELS and
            type(content) is str and content in _CONTENT_KINDS and
            type(output_state) is str and output_state in _OUTPUT_STATES and
            type(finalized_count) is int and 0 <= finalized_count <= MAX_MEANINGFUL_EVENTS and
            type(reconstructed) is bool and
            (not reconstructed or (output_state in {"missing", "null", "empty"} and finalized_count > 0))):
        return None
    return {"event_type": event, "terminal_status": status,
            "terminal_model_matches": model, "terminal_output_kind": output,
            "terminal_channel": channel, "terminal_content_kind": content,
            "terminal_output_state": output_state,
            "finalized_item_count": finalized_count,
            "output_reconstructed": reconstructed}


class TransportError(RuntimeError):
    def __init__(self, code: str, http_status: int | None = None,
                 body_shape: str | None = None, media_type_class: str | None = None,
                 wire_observation: dict[str, Any] | None = None,
                 stream_observation: dict[str, Any] | None = None):
        self.code = code if type(code) is str and code in _v1._CHILD_CODES else "transport_failure"
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        self.body_shape = body_shape if type(body_shape) is str and body_shape in _v2._BODY_SHAPES else None
        self.media_type_class = (media_type_class if type(media_type_class) is str and
                                 media_type_class in _v2._MEDIA_CLASSES else None)
        self.wire_observation = _v3._sanitize_observation(wire_observation)
        self.stream_observation = _sanitize_stream_observation(stream_observation)
        super().__init__(self.code)

    def __repr__(self) -> str:
        safe = TransportError(self.code, self.http_status, self.body_shape,
                              self.media_type_class, self.wire_observation,
                              self.stream_observation)
        return (f"TransportError(code={safe.code!r}, http_status={safe.http_status!r}, "
                f"body_shape={safe.body_shape!r}, media_type_class={safe.media_type_class!r}, "
                f"wire_observation={safe.wire_observation!r}, "
                f"stream_observation={safe.stream_observation!r})")


def _fail(code: str, http_status: int | None = None, body_shape: str | None = None,
          media_type_class: str | None = None,
          wire_observation: dict[str, Any] | None = None,
          stream_observation: dict[str, Any] | None = None) -> None:
    raise TransportError(code, http_status, body_shape, media_type_class,
                         wire_observation, stream_observation) from None


def _translate(exc: _v1.TransportError | _v2.TransportError | _v3.TransportError,
               *, status: int | None = None, shape: str | None = None,
               media: str | None = None,
               stream: dict[str, Any] | None = None) -> None:
    _fail(exc.code, status if status is not None else exc.http_status,
          shape if shape is not None else exc.body_shape,
          media if media is not None else getattr(exc, "media_type_class", None),
          getattr(exc, "wire_observation", None), stream)


def build_request(system: str, user: str,
                  json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return _v1.build_request(system, user, json_schema)
    except _v1.TransportError as exc:
        _translate(exc)


def _category(values: list[str], allowed: frozenset[str]) -> str:
    if not values:
        return "missing"
    kinds = {value if type(value) is str and value in allowed else "other" for value in values}
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _terminal_observation(event: dict[str, Any]) -> dict[str, Any]:
    response = event.get("response")
    status = response.get("status")
    status_kind = status if type(status) is str and status in _STATUSES - {"missing", "other"} else (
        "missing" if status is None else "other")
    model = response.get("model")
    output = response.get("output")
    output_state = ("missing" if "output" not in response else
                    "null" if output is None else
                    "empty" if type(output) is list and not output else
                    "list" if type(output) is list else "invalid")
    items = output if type(output) is list else []
    kinds = [item.get("type") if type(item) is dict else "other" for item in items]
    channels = [item.get("channel") for item in items if type(item) is dict and item.get("type") == "message"]
    contents = [part.get("type") if type(part) is dict else "other"
                for item in items if type(item) is dict and type(item.get("content")) is list
                for part in item["content"]]
    return {"terminal_status": status_kind,
            "terminal_model_matches": None if model is None else type(model) is str and model == MODEL,
            "terminal_output_kind": (_category(kinds, frozenset({"message", "reasoning"}))
                                     if output is None or type(output) is list else "other"),
            "terminal_channel": _category(["missing" if value is None else value for value in channels],
                                           _CHANNELS - {"mixed", "other"}),
            "terminal_content_kind": _category(contents, _CONTENT_KINDS - {"mixed", "other"}),
            "terminal_output_state": output_state}


class _Observation:
    def __init__(self) -> None:
        self.value: dict[str, Any] | None = None

    def see(self, event: dict[str, Any]) -> None:
        raw = event.get("type")
        kind = raw if type(raw) is str and raw in _EVENT_TYPES else "unknown"
        previous = self.value or {}
        terminal = {key: previous.get(key) for key in _OBS_KEYS - {"event_type"}}
        if not previous:
            terminal = {"terminal_status": "missing", "terminal_model_matches": None,
                        "terminal_output_kind": "missing", "terminal_channel": "missing",
                        "terminal_content_kind": "missing", "terminal_output_state": "unseen",
                        "finalized_item_count": 0, "output_reconstructed": False}
        if kind in {"response.completed", "response.failed", "response.incomplete"} and type(event.get("response")) is dict:
            terminal.update(_terminal_observation(event))
            terminal["output_reconstructed"] = False
        self.value = {"event_type": kind, **terminal}

    def finalized(self, count: int) -> None:
        if self.value is not None:
            self.value["finalized_item_count"] = count

    def reconstructed(self) -> None:
        if self.value is not None:
            self.value["output_reconstructed"] = True


_ALLOWED = frozenset({
    "response.created", "response.in_progress", "response.output_item.added",
    "response.output_item.done", "response.content_part.added",
    "response.content_part.done", "response.output_text.done",
    "response.reasoning_text.done", "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done", "response.reasoning_summary_text.done",
    "response.completed",
})
_DELTAS = frozenset({"response.output_text.delta", "response.reasoning_text.delta",
                     "response.reasoning_summary_text.delta"})


def _identity_size(identity: Any) -> int:
    if identity is None:
        return 0
    if type(identity) is not str:
        _v1._fail("invalid_event")
    try:
        return len(identity.encode("utf-8")) + 64
    except UnicodeError:
        _v1._fail("invalid_event")


def _finalized_item(item: Any) -> tuple[dict[str, Any], str | None, int]:
    """Keep only fields needed by the terminal validator, never a whole event."""
    if type(item) is not dict:
        _v1._fail("unsupported_output")
    kind = item.get("type")
    if kind not in ("message", "reasoning"):
        _v1._fail("unsupported_output")
    if item.get("status", "completed") != "completed":
        _v1._fail("incomplete_response")
    identity = item.get("id")
    identity_bytes = _identity_size(identity)
    if kind == "reasoning":
        return {"type": "reasoning", "status": "completed"}, identity, 128 + identity_bytes
    if item.get("role") != "assistant" or item.get("channel") not in (None, "final_answer"):
        _v1._fail("unsupported_output")
    content = item.get("content")
    if type(content) is not list or not content:
        _v1._fail("invalid_output")
    parts: list[dict[str, str]] = []
    chars = 0
    bytes_used = 128 + identity_bytes
    for part in content:
        if type(part) is not dict or part.get("type") != "output_text" or type(part.get("text")) is not str:
            _v1._fail("unsupported_output")
        value = part["text"]
        chars += len(value)
        if chars > MAX_OUTPUT_CHARS:
            _v1._fail("output_limit")
        try:
            bytes_used += len(value.encode("utf-8")) + 64
        except UnicodeError:
            _v1._fail("invalid_event")
        if bytes_used > MAX_WIRE_BYTES:
            _v1._fail("wire_limit")
        parts.append({"type": "output_text", "text": value})
    snapshot = {"type": "message", "role": "assistant", "status": "completed",
                "content": parts}
    if item.get("channel") is not None:
        snapshot["channel"] = "final_answer"
    return snapshot, identity, bytes_used


def parse_stream_events(events: Iterable[dict[str, Any]],
                        observation: _Observation | None = None) -> Completed:
    tracker = observation or _Observation()
    meaningful = 0
    completed: Completed | None = None
    finalized: dict[int, dict[str, Any]] = {}
    finalized_ids: set[str] = set()
    added: dict[int, tuple[str | None, str]] = {}
    unindexed_added = False
    retained_bytes = 0
    retained_chars = 0
    try:
        for event in events:
            if type(event) is dict:
                tracker.see(event)
            if not isinstance(event, dict) or type(event.get("type")) is not str:
                _v1._fail("invalid_event")
            kind = event["type"]
            if completed is not None:
                _v1._fail("event_after_completion")
            if kind in _DELTAS:
                continue
            meaningful += 1
            if meaningful > MAX_MEANINGFUL_EVENTS:
                _v1._fail("event_limit")
            if kind == "response.completed":
                response = event.get("response")
                terminal_output = response.get("output") if type(response) is dict else None
                if (type(response) is dict and finalized and
                        (terminal_output is None or
                         (type(terminal_output) is list and not terminal_output))):
                    # Reuse validated finalized items for absent, null, or empty
                    # terminal output. Never use unfinished deltas.
                    if unindexed_added or any(index not in finalized for index in added):
                        _v1._fail("incomplete_response")
                    tracker.reconstructed()
                    response = {**response, "output": [finalized[i] for i in sorted(finalized)]}
                completed = _v1._completed(response)
            elif kind == "response.failed":
                response = event.get("response")
                _v1._fail(_v1._provider_code(response.get("error") if isinstance(response, dict) else None))
            elif kind == "error":
                _v1._fail(_v1._provider_code(event.get("error") if isinstance(event.get("error"), dict) else event))
            elif kind == "response.incomplete":
                _v1._fail("incomplete_response")
            elif kind not in _ALLOWED:
                _v1._fail("unsupported_output")
            elif kind in ("response.output_item.added", "response.output_item.done"):
                item = event.get("item")
                if not isinstance(item, dict) or item.get("type") not in ("message", "reasoning"):
                    _v1._fail("unsupported_output")
                if item.get("type") == "message" and "content" in item:
                    content = item["content"]
                    if not isinstance(content, list) or any(
                            not isinstance(part, dict) or part.get("type") != "output_text"
                            for part in content):
                        _v1._fail("unsupported_output")
                index = event.get("output_index")
                if kind == "response.output_item.done":
                    if type(index) is not int or index < 0:
                        _v1._fail("invalid_event")
                    if index in finalized:
                        _v1._fail("invalid_event")
                    snapshot, identity, item_bytes = _finalized_item(item)
                    if identity is not None and identity in finalized_ids:
                        _v1._fail("invalid_event")
                    if index in added:
                        added_id, added_kind = added[index]
                        if added_kind != item["type"] or (added_id is not None and
                                added_id != identity):
                            _v1._fail("invalid_event")
                    retained_chars += sum(len(part["text"]) for part in snapshot.get("content", []))
                    retained_bytes += item_bytes
                    if retained_chars > MAX_OUTPUT_CHARS:
                        _v1._fail("output_limit")
                    if retained_bytes > MAX_WIRE_BYTES:
                        _v1._fail("wire_limit")
                    finalized[index] = snapshot
                    if identity is not None:
                        finalized_ids.add(identity)
                    tracker.finalized(len(finalized))
                elif index is not None:
                    if type(index) is not int or index < 0:
                        _v1._fail("invalid_event")
                    identity = item.get("id")
                    if identity is not None and type(identity) is not str:
                        _v1._fail("invalid_event")
                    if index in added:
                        _v1._fail("invalid_event")
                    if index in finalized:
                        _v1._fail("invalid_event")
                    retained_bytes += 128 + _identity_size(identity)
                    if retained_bytes > MAX_WIRE_BYTES:
                        _v1._fail("wire_limit")
                    added[index] = (identity, item["type"])
                else:
                    unindexed_added = True
            elif kind in ("response.content_part.added", "response.content_part.done"):
                part = event.get("part")
                if not isinstance(part, dict) or part.get("type") not in ("output_text", "reasoning_text"):
                    _v1._fail("unsupported_output")
        if completed is None:
            _v1._fail("missing_completion")
        return completed
    except _v1.TransportError as exc:
        _translate(exc, stream=tracker.value)


def _sse_events(response: http.client.HTTPResponse,
                observation: _Observation | None = None) -> Iterable[dict[str, Any]]:
    wire_bytes = 0
    data: list[bytes] = []
    data_bytes = 0
    label: bytes | None = None
    seen_completion = False
    seen_done = False
    first = True
    while True:
        line = response.readline(MAX_LINE_BYTES + 1)
        if not line:
            break
        if type(line) is not bytes:
            _v1._fail("invalid_event")
        wire_bytes += len(line)
        if wire_bytes > MAX_WIRE_BYTES or len(line) > MAX_LINE_BYTES:
            _v1._fail("wire_limit")
        if first and line.startswith(b"\xef\xbb\xbf"):
            line = line[3:]
        first = False
        if not line.endswith(b"\n"):
            _v1._fail("truncated_stream")
        line = line[:-1]
        if line.endswith(b"\r"):
            line = line[:-1]
        try:
            line.decode("utf-8")
        except UnicodeError:
            _v1._fail("invalid_event")
        if not line:
            if data:
                if seen_done:
                    _v1._fail("event_after_completion")
                try:
                    obj = _v1._decode_json(b"\n".join(data))
                except (UnicodeError, ValueError, RecursionError):
                    _v1._fail("invalid_event")
                if type(obj) is not dict:
                    _v1._fail("invalid_event")
                if observation is not None:
                    observation.see(obj)
                if label is not None and (type(obj.get("type")) is not str or
                                          obj["type"] != label.decode("ascii")):
                    _v1._fail("invalid_event")
                if obj.get("type") == "response.completed":
                    seen_completion = True
                yield obj
            elif label is not None:
                _v1._fail("invalid_event")
            data, data_bytes, label = [], 0, None
            continue
        if line.startswith(b":"):
            continue
        field, separator, value = line.partition(b":")
        if not field or any(byte not in b"abcdefghijklmnopqrstuvwxyz" for byte in field):
            _v1._fail("invalid_event")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            if value == b"[DONE]":
                if data or label is not None or not seen_completion or seen_done:
                    _v1._fail("missing_completion" if not seen_completion else "invalid_event")
                seen_done = True
            else:
                if seen_done:
                    _v1._fail("event_after_completion")
                data.append(value)
                data_bytes += len(value)
                if data_bytes > MAX_LINE_BYTES:
                    _v1._fail("wire_limit")
        elif field == b"event":
            if label is not None or not value or len(value) > 128 or not value.isascii():
                _v1._fail("invalid_event")
            label = value
        elif field == b"id":
            if b"\x00" in value:
                _v1._fail("invalid_event")
        elif field == b"retry":
            if not value or len(value) > 20 or not value.isdigit():
                _v1._fail("invalid_event")
        else:
            _v1._fail("invalid_event")
    if data or label is not None:
        _v1._fail("truncated_stream")


def _request_once(credentials: Credentials, request: dict[str, Any], timeout: float) -> Completed:
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
        if media not in ("sse", "missing"):
            try:
                _v3._non_sse_200(response, media)
            except _v3.TransportError as exc:
                _translate(exc)
        try:
            defect = _v3._header_defect(response.headers)
            encoding = _v3._encoding_class(response.getheader("Content-Encoding", None),
                                           frozenset({"identity", "gzip", "deflate", "br"}))
        except BaseException:
            defect, encoding = "unknown", "invalid"
        if defect != "none" or encoding not in ("missing", "identity"):
            try:
                _v3._non_sse_200(response, media)
            except _v3.TransportError as exc:
                _translate(exc)
        tracker = _Observation()
        try:
            return parse_stream_events(_sse_events(response, tracker), tracker)
        except TransportError as exc:
            _fail(exc.code, 200, "sse_event", media,
                  stream_observation=exc.stream_observation)
    finally:
        connection.close()


def _child(send, credentials: Credentials, request: dict[str, Any], timeout: float) -> None:
    try:
        send.send(("ok", _request_once(credentials, request, timeout)))
    except BaseException as exc:
        safe = (TransportError(exc.code, exc.http_status, exc.body_shape,
                               exc.media_type_class, exc.wire_observation,
                               exc.stream_observation)
                if type(exc) is TransportError else TransportError("transport_failure"))
        try:
            send.send(("error", safe.code, safe.http_status, safe.body_shape,
                       safe.media_type_class, safe.wire_observation,
                       safe.stream_observation))
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
        if len(message) == 7 and message[0] == "error":
            _fail(*message[1:])
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
    if type(credentials) is not Credentials:
        _fail("invalid_credentials")
    return _run_child(_child, (credentials, build_request(system, user, json_schema)), timeout)
