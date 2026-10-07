"""Finite timeout observations around the SHA-pinned warm-v8 transport.

Use ``TimeoutSubscriptionClient`` exactly as ``warm_v8.WarmSubscriptionClient``.
``diagnostic_records()`` returns detached public records (at most 16). These
are observations, never parser acceptance or settled usage. Queue timestamps
are local reader enqueue/dequeue times, not provider send times. A successful
warm call leaves its process open; its cleanup duration is therefore null.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path
import queue
import sys
import time
import types
from typing import Any


PINNED_WARM_V8_SHA256 = "0a55d44053349eb90511a597dae20f295c19bb53c734a78b5db3c41197343c21"
MAX_RECORDS = 16
_path = Path(__file__).resolve().with_name("codex_subscription_warm_v8.py")
_source = _path.read_bytes()
if hashlib.sha256(_source).hexdigest() != PINNED_WARM_V8_SHA256:
    raise RuntimeError("pinned_warm_v8_source_mismatch")
warm = types.ModuleType("pinned_codex_subscription_timeout_v1_base")
warm.__file__ = str(_path)
sys.modules[warm.__name__] = warm
exec(compile(_source, str(_path), "exec"), warm.__dict__)

base = warm.base
BudgetLimits = warm.BudgetLimits
SharedBudget = warm.SharedBudget
ConcurrentStop = warm.ConcurrentStop

_PHASES = ("startup", "preflight", "turn_start", "events", "unsubscribe", "cleanup")
_READER_STATES = frozenset({"not_started", "running", "stopped", "fault"})
_USAGE_STATES = frozenset({"absent", "zero", "positive"})


def _seconds(value: Any, *, absolute: bool = False) -> float | None:
    limit = 1_000_000_000_000 if absolute else 1_000_000
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > limit:
        return None
    return round(float(value), 6)


class _TimedQueue(queue.Queue[Any]):
    """The base constructor's queue shape, with local timestamps under its mutex."""

    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)
        self._stamps: deque[float] = deque()
        self.enqueued = 0
        self.dequeued = 0
        self.last_enqueued_at: float | None = None
        self.last_dequeued_at: float | None = None
        self.last_queue_delay: float | None = None
        self.fault = False

    def _put(self, item: Any) -> None:
        super()._put(item)
        try:
            stamp = time.monotonic()
            self._stamps.append(stamp)
            self.last_enqueued_at = stamp
            self.enqueued += 1
        except BaseException:
            self.fault = True
            raise

    def _get(self) -> Any:
        item = super()._get()
        try:
            stamp = self._stamps.popleft()
            now = time.monotonic()
            self.last_dequeued_at = now
            self.last_queue_delay = max(0.0, now - stamp)
            self.dequeued += 1
        except BaseException:
            self.fault = True
            raise
        return item

    def diagnostic_snapshot(self) -> dict[str, Any]:
        # Do not hold this mutex while consulting the session or budget.
        with self.mutex:
            return {"queue_depth": len(self.queue), "reader_enqueued": self.enqueued,
                    "reader_dequeued": self.dequeued,
                    "oldest_enqueue_monotonic": self._stamps[0] if self._stamps else None,
                    "last_enqueue_monotonic": self.last_enqueued_at,
                    "last_dequeue_monotonic": self.last_dequeued_at,
                    "last_queue_delay_seconds": self.last_queue_delay,
                    "queue_observer_fault": self.fault}


class TimeoutSession(warm.WarmSession):
    """Warm-v8 session with no changes to events, RPCs, or deadlines."""

    def __init__(self, binary: str, cwd: str, timeout: float = 120):
        self._reader_state = "not_started"
        self._observer_fault = False
        self._phase_started: dict[str, float] = {}
        self._phase_seconds: dict[str, float | None] = {name: None for name in _PHASES}
        self._deadline_original: float | None = None
        self._remaining_at_turn_start: float | None = None
        self._last_event_consumed_at: float | None = None
        self._event_consumed_count = 0
        self._invocation_started: float | None = None
        self._enqueued_baseline = 0
        self._dequeued_baseline = 0
        super().__init__(binary, cwd, timeout)

    @property
    def events(self) -> _TimedQueue:
        return self._events

    @events.setter
    def events(self, value: queue.Queue[Any]) -> None:
        # Called by pinned StdioSession before its reader thread starts.
        if type(value) is not queue.Queue or value.maxsize != 4096:
            raise RuntimeError("pinned_reader_queue_shape_mismatch")
        self._events = _TimedQueue(value.maxsize)

    def _read(self) -> None:
        self._reader_state = "running"
        try:
            super()._read()
        except BaseException:
            self._reader_state = "fault"
            self._observer_fault = True
            raise
        else:
            self._reader_state = "stopped"

    def _check_observer(self) -> None:
        if self._observer_fault or self.events.fault:
            base._fail("transport_failure")

    def set_deadline(self, deadline: float) -> None:
        super().set_deadline(deadline)
        self._invocation_started = time.monotonic()
        q = self.events.diagnostic_snapshot()
        self._enqueued_baseline = q["reader_enqueued"]
        self._dequeued_baseline = q["reader_dequeued"]
        self._deadline_original = deadline
        self._phase_started["preflight"] = time.monotonic()
        self._phase_seconds = {name: None for name in _PHASES}
        self._remaining_at_turn_start = None
        self._last_event_consumed_at = None
        self._event_consumed_count = 0

    def receive(self) -> dict[str, Any]:
        self._check_observer()
        event = super().receive()
        self._check_observer()
        return event

    def rpc(self, method: str, params: dict[str, Any], *, preserve_notifications: bool = False) -> dict[str, Any]:
        if method == "turn/start":
            now = time.monotonic()
            began = self._phase_started.pop("preflight", None)
            if began is not None:
                self._phase_seconds["preflight"] = now - began
            self._remaining_at_turn_start = max(0.0, self.deadline - now)
            self._phase_started["turn_start"] = now
        succeeded = False
        try:
            result = super().rpc(method, params, preserve_notifications=preserve_notifications)
            succeeded = True
            return result
        finally:
            if method == "turn/start":
                now = time.monotonic()
                began = self._phase_started.pop("turn_start", None)
                if began is not None:
                    self._phase_seconds["turn_start"] = now - began
                if succeeded:
                    self._phase_started["events"] = now

    def next_event(self) -> dict[str, Any]:
        self._check_observer()
        event = super().next_event()
        self._last_event_consumed_at = time.monotonic()
        self._event_consumed_count = min(base.MAX_EVENTS, self._event_consumed_count + 1)
        self._check_observer()
        return event

    def unsubscribe(self, thread_id: str) -> None:
        now = time.monotonic()
        began = self._phase_started.pop("events", None)
        if began is not None:
            self._phase_seconds["events"] = now - began
        self._phase_started["unsubscribe"] = now
        try:
            return super().unsubscribe(thread_id)
        finally:
            began = self._phase_started.pop("unsubscribe", None)
            if began is not None:
                self._phase_seconds["unsubscribe"] = time.monotonic() - began

    def timeout_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        durations = dict(self._phase_seconds)
        for name, began in self._phase_started.items():
            if name in durations and durations[name] is None:
                durations[name] = now - began
        observed = warm.v7.v6.v5.v4._project_observation(getattr(self, "turn_observation", None))
        queue_state = self.events.diagnostic_snapshot()
        origin = self._invocation_started
        for name in ("last_enqueue_monotonic", "last_dequeue_monotonic", "oldest_enqueue_monotonic"):
            if origin is not None and queue_state[name] is not None and queue_state[name] < origin:
                queue_state[name] = None
        enqueued = max(0, queue_state["reader_enqueued"] - self._enqueued_baseline)
        dequeued = max(0, queue_state["reader_dequeued"] - self._dequeued_baseline)
        queue_state["reader_enqueued"] = min(4096, enqueued)
        queue_state["reader_dequeued"] = min(4096, dequeued)
        queue_state["reader_enqueued_saturated"] = enqueued > 4096
        queue_state["reader_dequeued_saturated"] = dequeued > 4096
        process = getattr(self, "process", None)
        try:
            process_alive = process.poll() is None if process is not None else None
        except Exception:
            process_alive = None
        return {"deadline_monotonic": self._deadline_original,
                "remaining_at_turn_start_seconds": self._remaining_at_turn_start,
                "phase_seconds": durations,
                "reader_state": self._reader_state,
                "last_event_consumed_monotonic": self._last_event_consumed_at,
                "events_returned": self._event_consumed_count,
                "observed": observed, "process_alive": process_alive, **queue_state}


def _project_record(value: Any) -> dict[str, Any] | None:
    """Reject malformed or injected private fields at the public boundary."""
    keys = {"schema", "status", "failure_code", "known_usage", "total_seconds",
            "invocation_start_monotonic", "deadline_monotonic", "remaining_at_turn_start_seconds", "phase_seconds",
            "reader_state", "queue_depth", "reader_enqueued", "reader_dequeued",
            "reader_enqueued_saturated", "reader_dequeued_saturated",
            "last_enqueue_monotonic", "last_dequeue_monotonic", "oldest_enqueue_monotonic",
            "last_queue_delay_seconds", "last_event_consumed_monotonic", "events_returned",
            "observed", "process_alive", "precleanup"}
    if type(value) is not dict or set(value) != keys or value["schema"] != "warm_timeout_observation_v1":
        return None
    if type(value["status"]) is not str or value["status"] not in {"success", "failure"}:
        return None
    if type(value["known_usage"]) is not bool and value["known_usage"] is not None:
        return None
    code = value["failure_code"]
    if code is not None and (type(code) is not str or
            warm.serialize_failure({"code": code, "phase": "run", "rpc": "turn/events"})["code"] != code):
        return None
    if value["status"] == "success" and (code is not None or value["known_usage"] is not True):
        return None
    if _seconds(value["total_seconds"]) is None:
        return None
    absolute_names = {"invocation_start_monotonic", "deadline_monotonic", "last_enqueue_monotonic",
                      "last_dequeue_monotonic", "oldest_enqueue_monotonic", "last_event_consumed_monotonic"}
    for name in ("invocation_start_monotonic", "deadline_monotonic", "remaining_at_turn_start_seconds",
                 "last_enqueue_monotonic", "last_dequeue_monotonic", "oldest_enqueue_monotonic",
                 "last_queue_delay_seconds", "last_event_consumed_monotonic"):
        if value[name] is not None and _seconds(value[name], absolute=name in absolute_names) is None:
            return None
    phases = value["phase_seconds"]
    if type(phases) is not dict or set(phases) != set(_PHASES):
        return None
    if any(v is not None and _seconds(v) is None for v in phases.values()):
        return None
    if value["reader_state"] is not None and (type(value["reader_state"]) is not str
            or value["reader_state"] not in _READER_STATES):
        return None
    if value["process_alive"] is not None and type(value["process_alive"]) is not bool:
        return None
    for name in ("queue_depth", "reader_enqueued", "reader_dequeued", "events_returned"):
        v = value[name]
        if v is not None and (type(v) is not int or not 0 <= v <= 4096):
            return None
    if (type(value["reader_enqueued_saturated"]) is not bool
            or type(value["reader_dequeued_saturated"]) is not bool):
        return None
    observed = value["observed"]
    if observed is not None and warm.v7.v6.v5.v4._project_observation(observed) != observed:
        return None
    pre = value["precleanup"]
    if pre is not None:
        if type(pre) is not dict or set(pre) != {"queue_depth", "reader_state", "last_enqueue_monotonic",
                "last_dequeue_monotonic", "oldest_enqueue_monotonic", "events_returned", "process_alive"}:
            return None
        if type(pre["reader_state"]) is not str or pre["reader_state"] not in _READER_STATES:
            return None
        if any(type(pre[k]) is not int or not 0 <= pre[k] <= 4096 for k in ("queue_depth", "events_returned")):
            return None
        if pre["process_alive"] is not None and type(pre["process_alive"]) is not bool:
            return None
        if any(pre[k] is not None and _seconds(pre[k], absolute=True) is None for k in
               ("last_enqueue_monotonic", "last_dequeue_monotonic", "oldest_enqueue_monotonic")):
            return None
    return {"schema": value["schema"], "status": value["status"], "failure_code": code,
            "known_usage": value["known_usage"], "total_seconds": _seconds(value["total_seconds"]),
            **{k: _seconds(value[k], absolute=k in absolute_names) if value[k] is not None else None for k in
               ("invocation_start_monotonic", "deadline_monotonic", "remaining_at_turn_start_seconds",
                "last_enqueue_monotonic", "last_dequeue_monotonic", "oldest_enqueue_monotonic",
                "last_queue_delay_seconds", "last_event_consumed_monotonic")},
            "phase_seconds": {k: _seconds(phases[k]) if phases[k] is not None else None for k in _PHASES},
            "reader_state": value["reader_state"],
            "process_alive": value["process_alive"],
            **{k: value[k] for k in ("queue_depth", "reader_enqueued", "reader_dequeued",
                                    "reader_enqueued_saturated", "reader_dequeued_saturated", "events_returned")},
            "observed": dict(observed) if observed is not None else None,
            "precleanup": dict(pre) if pre is not None else None}


class TimeoutSubscriptionClient(warm.WarmSubscriptionClient):
    """Opt-in client; ``diagnostic_records()`` is the only public trace API."""

    def __init__(self, *args: Any, session_factory: Any = TimeoutSession, **kwargs: Any):
        self._diagnostics: list[dict[str, Any]] = []
        self._active_diagnostic: dict[str, Any] | None = None
        self._session_factory_original = session_factory
        super().__init__(*args, session_factory=self._new_session, **kwargs)

    def _new_session(self, *args: Any, **kwargs: Any) -> Any:
        began = time.monotonic()
        try:
            return self._session_factory_original(*args, **kwargs)
        finally:
            if self._active_diagnostic is not None:
                self._active_diagnostic["phase_seconds"]["startup"] = time.monotonic() - began

    def _snapshot_session(self, record: dict[str, Any]) -> None:
        session = self.session
        if not isinstance(session, TimeoutSession):
            return
        snap = session.timeout_snapshot()
        for key in ("deadline_monotonic", "remaining_at_turn_start_seconds", "reader_state",
                    "queue_depth", "reader_enqueued", "reader_dequeued",
                    "reader_enqueued_saturated", "reader_dequeued_saturated", "last_enqueue_monotonic",
                    "last_dequeue_monotonic", "oldest_enqueue_monotonic", "last_queue_delay_seconds",
                    "last_event_consumed_monotonic", "events_returned", "observed", "process_alive"):
            record[key] = snap[key]
        for key, value in snap["phase_seconds"].items():
            if value is not None:
                record["phase_seconds"][key] = value

    def _before_cleanup(self) -> None:
        record = self._active_diagnostic
        if record is None or record["precleanup"] is not None:
            return
        if not isinstance(self.session, TimeoutSession):
            return
        self._snapshot_session(record)
        record["precleanup"] = {key: record[key] for key in
            ("queue_depth", "reader_state", "last_enqueue_monotonic",
             "last_dequeue_monotonic", "oldest_enqueue_monotonic", "events_returned", "process_alive")}

    def _record_failure(self, code: str, phase: str, turn_admitted: bool, known_usage: bool) -> None:
        record = self._active_diagnostic
        if record is not None:
            if record["failure_code"] is None:
                record["known_usage"] = known_usage if turn_admitted else None
                projected = warm.serialize_failure({"code": code, "phase": phase, "rpc": "turn/events"})
                record["failure_code"] = projected["code"] if projected is not None else "fixed_other"
            self._before_cleanup()
        return super()._record_failure(code, phase, turn_admitted, known_usage)

    def _close_process(self) -> None:
        self._before_cleanup()
        began = time.monotonic()
        try:
            return super()._close_process()
        finally:
            record = self._active_diagnostic
            if record is not None:
                record["phase_seconds"]["cleanup"] = time.monotonic() - began

    def _complete_locked(self, request: Any) -> str:
        if len(self._diagnostics) >= MAX_RECORDS:
            self.budget.halt("transport_failure")
            raise ConcurrentStop("transport_failure")
        began = time.monotonic()
        record: dict[str, Any] = {"schema": "warm_timeout_observation_v1",
            "status": "failure", "failure_code": None, "known_usage": None,
            "total_seconds": 0.0, "invocation_start_monotonic": began,
            "deadline_monotonic": None,
            "remaining_at_turn_start_seconds": None,
            "phase_seconds": {name: None for name in _PHASES}, "reader_state": None,
            "queue_depth": None, "reader_enqueued": None, "reader_dequeued": None,
            "reader_enqueued_saturated": False, "reader_dequeued_saturated": False,
            "last_enqueue_monotonic": None, "last_dequeue_monotonic": None,
            "oldest_enqueue_monotonic": None,
            "last_queue_delay_seconds": None, "last_event_consumed_monotonic": None,
            "events_returned": None, "observed": None, "process_alive": None,
            "precleanup": None}
        self._active_diagnostic = record
        try:
            answer = super()._complete_locked(request)
            if isinstance(self.session, TimeoutSession):
                self.session._check_observer()
            record["status"] = "success"
            record["known_usage"] = True
            return answer
        finally:
            try:
                if record["precleanup"] is None:
                    self._snapshot_session(record)
                record["total_seconds"] = time.monotonic() - began
                if record["status"] == "failure" and record["failure_code"] is None:
                    record["failure_code"] = "fixed_other"
                if _project_record(record) is None:
                    self.budget.halt("transport_failure")
                    raise ConcurrentStop("transport_failure")
                self._diagnostics.append(record)
            finally:
                self._active_diagnostic = None

    def diagnostic_records(self) -> tuple[dict[str, Any], ...]:
        """Detached, strictly validated finite records; malformed state fails closed."""
        projected = tuple(_project_record(record) for record in self._diagnostics)
        if any(record is None for record in projected):
            raise ValueError("diagnostic_record_invalid")
        return projected
