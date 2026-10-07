"""300-second diagnostic Responses transport with bounded owned-child reaping.

The observation is the last completed local snapshot, never a causal diagnosis.
Wire, parser, usage, and request behavior delegate directly to immutable v6.
"""
from __future__ import annotations

import http.client
import json
import math
import multiprocessing
from multiprocessing.connection import wait as wait_connections
from multiprocessing import popen_spawn_posix
import os
import ssl
import threading
import time
from typing import Any

from benchmarks import chatgpt_plan_responses_v1 as _v1
from benchmarks import chatgpt_plan_responses_v2 as _v2
from benchmarks import chatgpt_plan_responses_v3 as _v3
from benchmarks import chatgpt_plan_responses_v6 as _v6


HOST, PATH, MODEL = _v6.HOST, _v6.PATH, _v6.MODEL
MAX_WALL_SECONDS = 300.0
MAX_WIRE_BYTES = _v6.MAX_WIRE_BYTES
MAX_LINE_BYTES = _v6.MAX_LINE_BYTES
MAX_MEANINGFUL_EVENTS = _v6.MAX_MEANINGFUL_EVENTS
MAX_OUTPUT_CHARS = _v6.MAX_OUTPUT_CHARS
MAX_REQUEST_BYTES = _v6.MAX_REQUEST_BYTES
PROVIDER_CODES = _v6.PROVIDER_CODES
Completed, Credentials = _v6.Completed, _v6.Credentials
_sanitize_stream_observation = _v6._sanitize_stream_observation

_PHASES = ("unknown", "child_entry", "request_send", "headers_wait",
           "response_check", "stream_read", "parse", "response_close", "result_ipc")
_SITES = frozenset({"none", "result_wait", "result_recv", "deadline_after_recv"})
_KEYS = frozenset({"child_phase", "parent_timeout_site", "last_progress_elapsed_ms",
                   "parent_elapsed_ms", "elapsed_saturated", "snapshot_valid",
                   "timeout_allowance_ms", "wire_bytes", "event_count",
                   "completion_seen", "result_ready", "result_ipc_started",
                   "child_alive_when_sampled"})
_MAX_MS = int(MAX_WALL_SECONDS * 1000) + 1000
# Every yielded SSE dict consumes at least ``data:{}\n\n`` (9 wire bytes).
# Delta events are excluded from v6's meaningful-event limit, so that limit
# cannot bound the total number of yielded events.
_MAX_EVENTS = MAX_WIRE_BYTES // 9 + 1
# Sequence, phase, elapsed milliseconds, wire bytes, event count, completed,
# result ready, IPC started. Only the child writes. No lock or progress pipe.
_SLOTS = 8
_STATUS_LOCK_SECONDS = .05


def sanitize_timeout_observation(value: Any) -> dict[str, Any] | None:
    """Copy an exact, finite, content-free timeout observation for bridge use."""
    if type(value) is not dict or value.keys() != _KEYS:
        return None
    phase, site = value["child_phase"], value["parent_timeout_site"]
    if type(phase) is not str or phase not in _PHASES or type(site) is not str or site not in _SITES:
        return None
    for key, maximum in (("last_progress_elapsed_ms", _MAX_MS),
                         ("parent_elapsed_ms", _MAX_MS),
                         ("timeout_allowance_ms", int(MAX_WALL_SECONDS * 1000)),
                         ("wire_bytes", MAX_WIRE_BYTES),
                         ("event_count", _MAX_EVENTS)):
        number = value[key]
        if number is None:
            if key in ("parent_elapsed_ms", "timeout_allowance_ms"):
                return None
        elif type(number) is not int or not 0 <= number <= maximum:
            return None
    for key in ("elapsed_saturated", "snapshot_valid"):
        if type(value[key]) is not bool:
            return None
    for key in ("completion_seen", "result_ready", "result_ipc_started",
                "child_alive_when_sampled"):
        if value[key] is not None and type(value[key]) is not bool:
            return None
    child_keys = ("last_progress_elapsed_ms", "wire_bytes", "event_count",
                  "completion_seen", "result_ready", "result_ipc_started")
    if value["snapshot_valid"]:
        if phase == "unknown" or any(value[key] is None for key in child_keys):
            return None
        if (value["completion_seen"] and value["event_count"] == 0 or
                value["result_ipc_started"] and not value["result_ready"] or
                phase == "result_ipc" and not (value["result_ready"] and
                                                value["result_ipc_started"])):
            return None
    elif phase != "unknown" or any(value[key] is not None for key in child_keys):
        return None
    if site == "none" and value["child_alive_when_sampled"] is not None:
        return None
    return dict(value)


class TransportError(_v6.TransportError):
    def __init__(self, code: str, http_status: int | None = None,
                 body_shape: str | None = None, media_type_class: str | None = None,
                 wire_observation: dict[str, Any] | None = None,
                 stream_observation: dict[str, Any] | None = None,
                 timeout_observation: dict[str, Any] | None = None):
        super().__init__(code, http_status, body_shape, media_type_class,
                         wire_observation, stream_observation)
        self.timeout_observation = (sanitize_timeout_observation(timeout_observation)
                                    if self.code == "timeout" else None)

    def __repr__(self) -> str:
        safe = TransportError(self.code, self.http_status, self.body_shape,
                              self.media_type_class, self.wire_observation,
                              self.stream_observation, self.timeout_observation)
        return (f"TransportError(code={safe.code!r}, http_status={safe.http_status!r}, "
                f"body_shape={safe.body_shape!r}, media_type_class={safe.media_type_class!r}, "
                f"wire_observation={safe.wire_observation!r}, "
                f"stream_observation={safe.stream_observation!r}, "
                f"timeout_observation={safe.timeout_observation!r})")


def _fail(code: str, http_status: int | None = None,
          body_shape: str | None = None, media_type_class: str | None = None,
          wire_observation: dict[str, Any] | None = None,
          stream_observation: dict[str, Any] | None = None,
          timeout_observation: dict[str, Any] | None = None) -> None:
    raise TransportError(code, http_status, body_shape, media_type_class,
                         wire_observation, stream_observation, timeout_observation) from None


def build_request(system: str, user: str,
                  json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return _v6.build_request(system, user, json_schema)
    except _v6.TransportError as exc:
        _fail(exc.code, exc.http_status, exc.body_shape, exc.media_type_class,
              exc.wire_observation, exc.stream_observation)


def parse_stream_events(events, observation=None) -> Completed:
    try:
        return _v6.parse_stream_events(events, observation)
    except _v6.TransportError as exc:
        _fail(exc.code, exc.http_status, exc.body_shape, exc.media_type_class,
              exc.wire_observation, exc.stream_observation)


class _Progress:
    def __init__(self, slots, allowance: float):
        self.slots = slots
        self.started = time.monotonic()
        self.allowance_ms = min(_MAX_MS, max(0, int(allowance * 1000)))
        self.phase = 0
        self.wire = 0
        self.events = 0
        self.completed = 0
        self.ready = 0
        self.ipc = 0

    def mark(self, phase: str | None = None, *, wire: int = 0,
             event: bool = False, completed: bool = False,
             ready: bool = False, ipc: bool = False) -> None:
        if phase is not None:
            self.phase = _PHASES.index(phase)
        self.wire = min(MAX_WIRE_BYTES, self.wire + max(0, wire))
        self.events = min(_MAX_EVENTS, self.events + int(event))
        self.completed |= int(completed)
        self.ready |= int(ready)
        self.ipc |= int(ipc)
        values = (self.phase, min(_MAX_MS, max(0, int((time.monotonic() - self.started) * 1000))),
                  self.wire, self.events, self.completed, self.ready, self.ipc)
        self.slots[0] += 1  # odd: snapshot in progress
        for index, value in enumerate(values, 1):
            self.slots[index] = value
        self.slots[0] += 1  # even: complete snapshot


def _snapshot(slots, allowance: float, site: str, parent_started: float,
              alive: bool | None = None) -> dict[str, Any]:
    sequence = slots[0]
    values = tuple(slots[index] for index in range(1, _SLOTS))
    valid = sequence > 0 and sequence % 2 == 0 and slots[0] == sequence
    if valid:
        phase, elapsed, wire, events, completed, ready, ipc = values
        valid = (type(phase) is int and 0 < phase < len(_PHASES) and
                 0 <= elapsed <= _MAX_MS and 0 <= wire <= MAX_WIRE_BYTES and
                 0 <= events <= _MAX_EVENTS and
                 all(number in (0, 1) for number in (completed, ready, ipc)))
    raw_parent_elapsed = max(0, int((time.monotonic() - parent_started) * 1000))
    observation = {
        "child_phase": _PHASES[phase] if valid else "unknown",
        "parent_timeout_site": site,
        "last_progress_elapsed_ms": elapsed if valid else None,
        "parent_elapsed_ms": min(_MAX_MS, raw_parent_elapsed),
        # Parent elapsed begins before spawn. Child elapsed begins at child
        # entry; saturation is true if either known elapsed hits the clamp.
        "elapsed_saturated": raw_parent_elapsed >= _MAX_MS or
                             (valid and elapsed >= _MAX_MS),
        "snapshot_valid": valid,
        "timeout_allowance_ms": min(int(MAX_WALL_SECONDS * 1000),
                                    max(0, int(allowance * 1000))),
        "wire_bytes": wire if valid else None,
        "event_count": events if valid else None,
        "completion_seen": bool(completed) if valid else None,
        "result_ready": bool(ready) if valid else None,
        "result_ipc_started": bool(ipc) if valid else None,
        # Sampled after the deadline/watchdog may have killed the child:
        # false does not establish an earlier child exit.
        "child_alive_when_sampled": alive if site != "none" else None,
    }
    return sanitize_timeout_observation(observation) or {
        **observation, "child_phase": "unknown", "last_progress_elapsed_ms": None,
        "snapshot_valid": False,
        "wire_bytes": None, "event_count": None, "completion_seen": None,
        "result_ready": None, "result_ipc_started": None}


class _ReadProgress:
    def __init__(self, response, progress: _Progress):
        self.response, self.progress = response, progress

    def readline(self, size):
        self.progress.mark("stream_read")
        line = self.response.readline(size)
        if type(line) is bytes:
            self.progress.mark("parse", wire=len(line))
        else:
            self.progress.mark("parse")
        return line


def _events(response, tracker, progress: _Progress):
    for event in _v6._sse_events(_ReadProgress(response, progress), tracker):
        # This notes an event type before the v6 parser validates the event.
        progress.mark("parse", event=True,
                      completed=type(event) is dict and event.get("type") == "response.completed")
        yield event


def _request_once(credentials: Credentials, request: dict[str, Any], timeout: float,
                  progress: _Progress | None = None) -> Completed:
    # The request and response policy is intentionally identical to v6.
    # request_send covers local connection/TLS and request writes; it does not
    # establish that any bytes reached the provider.
    if progress is not None:
        progress.mark("request_send")
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
        if progress is not None:
            progress.mark("headers_wait")
        response = connection.getresponse()
        # Includes header/media checks and bounded non-SSE/error-body reads.
        # SSE parsing has not started at this phase.
        if progress is not None:
            progress.mark("response_check")
        media = _v2._response_media(response)
        if response.status != 200:
            try:
                _v1._http_error(response)
            except _v1.TransportError as exc:
                _fail(exc.code, exc.http_status, exc.body_shape, media,
                      getattr(exc, "wire_observation", None))
        if media not in ("sse", "missing"):
            try:
                _v3._non_sse_200(response, media)
            except _v3.TransportError as exc:
                _fail(exc.code, exc.http_status, exc.body_shape,
                      exc.media_type_class, exc.wire_observation)
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
                _fail(exc.code, exc.http_status, exc.body_shape,
                      exc.media_type_class, exc.wire_observation)
        tracker = _v6._Observation()
        try:
            events = (_events(response, tracker, progress) if progress is not None
                      else _v6._sse_events(response, tracker))
            return parse_stream_events(events, tracker)
        except _v6.TransportError as exc:
            _fail(exc.code, 200, "sse_event", media,
                  stream_observation=exc.stream_observation)
    finally:
        if progress is not None:
            progress.mark("response_close")
        connection.close()


def _child(send, slots, credentials: Credentials, request: dict[str, Any], timeout: float) -> None:
    progress = _Progress(slots, timeout)
    progress.mark("child_entry")
    try:
        result = _request_once(credentials, request, timeout, progress)
        # A local result is ready for IPC; this does not attest model validity.
        progress.mark(ready=True)
        progress.mark("result_ipc", ipc=True)
        send.send(("ok", result))
    except BaseException as exc:
        safe = (TransportError(exc.code, exc.http_status, exc.body_shape,
                               exc.media_type_class, exc.wire_observation,
                               exc.stream_observation, exc.timeout_observation)
                if type(exc) is TransportError else TransportError("transport_failure"))
        try:
            # Error results are also locally ready for IPC.
            progress.mark(ready=True)
            progress.mark("result_ipc", ipc=True)
            send.send(("error", safe.code, safe.http_status, safe.body_shape,
                       safe.media_type_class, safe.wire_observation,
                       safe.stream_observation, safe.timeout_observation))
        except (OSError, EOFError):
            pass
    finally:
        send.close()


class _OwnedPopen(popen_spawn_posix.Popen):
    """Only the starting request thread may reap this child's exit status."""

    def __init__(self, process_obj):
        self._status_owner = threading.current_thread()
        self._status_lock = threading.Lock()
        super().__init__(process_obj)

    def poll(self, flag=None):
        if threading.current_thread() is not self._status_owner:
            return self.returncode
        # Never hold the lock across blocking waitpid. A stalled signal holder
        # fails closed instead of delaying request cleanup indefinitely.
        if not self._status_lock.acquire(timeout=_STATUS_LOCK_SECONDS):
            _fail("cleanup_failure")
        try:
            return super().poll(os.WNOHANG)
        finally:
            self._status_lock.release()

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        if threading.current_thread() is not self._status_owner:
            return self.returncode
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        sentinel_ready = False
        while True:
            status = self.poll()
            if status is not None:
                return status
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            if sentinel_ready:
                # Sentinel EOF can precede a waitable exit status.
                time.sleep(.01 if remaining is None else min(.01, remaining))
            else:
                sentinel_ready = bool(wait_connections([self.sentinel], remaining))

    def _send_signal(self, sig):
        # Coordinate with status publication to avoid signaling a recycled PID.
        # A foreign watchdog may defer to mandatory owner cleanup; the owner
        # fails closed if the status lock cannot be acquired within 50 ms.
        owner = threading.current_thread() is self._status_owner
        if not self._status_lock.acquire(timeout=_STATUS_LOCK_SECONDS):
            if owner:
                _fail("cleanup_failure")
            return
        try:
            if self.returncode is None:
                try:
                    os.kill(self.pid, sig)
                except ProcessLookupError:
                    pass
        finally:
            self._status_lock.release()


class _OwnedSpawnProcess(multiprocessing.get_context("spawn").Process):
    @staticmethod
    def _Popen(process_obj):
        return _OwnedPopen(process_obj)


def _run_child(target, args: tuple[Any, ...], timeout: float) -> Completed:
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or
            not 0 < timeout <= MAX_WALL_SECONDS):
        _fail("invalid_timeout")
    context = multiprocessing.get_context("spawn")
    slots = context.RawArray("q", _SLOTS)
    receive, send = context.Pipe(duplex=False)
    process = _OwnedSpawnProcess(target=target, args=(send, slots, *args, timeout), daemon=True)
    parent_started = time.monotonic()
    deadline = parent_started + timeout
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        try:
            if process.pid is not None and process.is_alive():
                process.kill()
        except (OSError, ValueError):
            pass

    def timeout_error(site: str) -> None:
        try:
            alive = process.is_alive() if process.pid is not None else False
        except (OSError, ValueError):
            alive = None
        _fail("timeout", timeout_observation=_snapshot(slots, timeout, site,
                                                        parent_started, alive))

    watchdog = threading.Timer(timeout, expire)
    watchdog.daemon = True
    try:
        watchdog.start()
        process.start()
        send.close()
        if not receive.poll(max(0, deadline - time.monotonic())):
            timeout_error("result_wait")
        try:
            message = receive.recv()
        except (EOFError, OSError, ValueError):
            if expired.is_set() or time.monotonic() >= deadline:
                timeout_error("result_recv")
            _fail("transport_failure")
        if expired.is_set() or time.monotonic() >= deadline:
            timeout_error("deadline_after_recv")
        if type(message) is not tuple:
            _fail("transport_failure")
        if len(message) == 2 and message[0] == "ok" and type(message[1]) is Completed:
            return message[1]
        if len(message) == 8 and message[0] == "error":
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
    try:
        request = build_request(system, user, json_schema)
    except _v6.TransportError as exc:
        _fail(exc.code, exc.http_status, exc.body_shape, exc.media_type_class,
              exc.wire_observation, exc.stream_observation)
    return _run_child(_child, (credentials, request), timeout)
