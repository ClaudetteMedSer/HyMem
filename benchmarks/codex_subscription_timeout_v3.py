"""Finite timeout observations around the SHA-pinned warm-v9 transport.

Use ``TimeoutSubscriptionClient`` exactly as ``warm_v9.WarmSubscriptionClient``.
``diagnostic_records()`` returns the detached last 16 records, while
``diagnostic_summary()`` retains a per-client first failure and finite totals.
These are observations, never parser acceptance or settled usage. Queue
timestamps are local reader enqueue/dequeue times, not provider send times.
A successful warm call leaves its process open; its cleanup duration is null.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import math
from pathlib import Path
import queue
import sys
import time
import types
from typing import Any


PINNED_WARM_V9_SHA256 = "f9081dda08a6e1975a3e951190ae6580161d89817303a1b75d0ba21982fcd470"
MAX_RECORDS = 16
MAX_SUMMARY_COUNT = 1_000_000
MAX_SUMMARY_SECONDS = 1_000_000.0
_path = Path(__file__).resolve().with_name("codex_subscription_warm_v9.py")
_source = _path.read_bytes()
if hashlib.sha256(_source).hexdigest() != PINNED_WARM_V9_SHA256:
    raise RuntimeError("pinned_warm_v9_source_mismatch")
warm = types.ModuleType("pinned_codex_subscription_timeout_v3_base")
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
    """Warm-v9 session with no changes to events, RPCs, or deadlines."""

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
        observed = warm.v8.v7.v6.v5.v4._project_observation(getattr(self, "turn_observation", None))
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
    if observed is not None and warm.v8.v7.v6.v5.v4._project_observation(observed) != observed:
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


def _project_summary(value: Any) -> dict[str, Any] | None:
    """Accept only the fixed finite public shape; copy nested evidence."""
    keys = {"schema", "calls", "successes", "failures", "counts_saturated",
            "first_failure", "last_record", "timing_seconds", "timing_saturated"}
    if type(value) is not dict or set(value) != keys or value["schema"] != "warm_timeout_summary_v1":
        return None
    counts = (value["calls"], value["successes"], value["failures"])
    if (any(type(n) is not int or not 0 <= n <= MAX_SUMMARY_COUNT for n in counts)
            or type(value["counts_saturated"]) is not bool
            or type(value["timing_saturated"]) is not bool):
        return None
    calls, successes, failures = counts
    if not value["counts_saturated"] and calls != successes + failures:
        return None
    if value["counts_saturated"] and (calls != MAX_SUMMARY_COUNT or successes + failures < calls):
        return None
    if calls == 0 and (value["last_record"] is not None or value["first_failure"] is not None):
        return None
    last = None if value["last_record"] is None else _project_record(value["last_record"])
    if calls > 0 and last is None:
        return None
    if last is not None and (last["status"] == "success" and successes == 0
                             or last["status"] == "failure" and failures == 0):
        return None
    first = value["first_failure"]
    projected_first = None
    if first is not None:
        if (type(first) is not dict or set(first) != {"call_index", "call_index_saturated", "record"}
                or type(first["call_index"]) is not int
                or not 1 <= first["call_index"] <= MAX_SUMMARY_COUNT
                or type(first["call_index_saturated"]) is not bool
                or (not first["call_index_saturated"] and first["call_index"] > calls)
                or (not value["counts_saturated"] and first["call_index"] - 1 > successes)
                or (first["call_index_saturated"] and
                    (not value["counts_saturated"] or first["call_index"] != MAX_SUMMARY_COUNT))):
            return None
        record = _project_record(first["record"])
        if record is None or record["status"] != "failure":
            return None
        if successes == 0 and (first["call_index"] != 1 or first["call_index_saturated"]):
            return None
        if (not value["counts_saturated"] and first["call_index"] == calls
                and (failures != 1 or record != last)):
            return None
        projected_first = {"call_index": first["call_index"],
                           "call_index_saturated": first["call_index_saturated"], "record": record}
    if (failures == 0) != (projected_first is None):
        return None
    timing = value["timing_seconds"]
    if (type(timing) is not dict or set(timing) != {"total", "phases"}
            or type(timing["phases"]) is not dict or set(timing["phases"]) != set(_PHASES)):
        return None
    times = (timing["total"], *(timing["phases"][name] for name in _PHASES))
    if any(_seconds(number) is None for number in times):
        return None
    if calls == 0 and any(number != 0 for number in times):
        return None
    if value["timing_saturated"] and not any(number == MAX_SUMMARY_SECONDS for number in times):
        return None
    return {"schema": value["schema"], "calls": calls, "successes": successes,
            "failures": failures, "counts_saturated": value["counts_saturated"],
            "first_failure": projected_first, "last_record": last,
            "timing_seconds": {"total": _seconds(timing["total"]),
                               "phases": {name: _seconds(timing["phases"][name]) for name in _PHASES}},
            "timing_saturated": value["timing_saturated"]}


class TimeoutSubscriptionClient(warm.WarmSubscriptionClient):
    """Opt-in client with bounded records and finite detached summary."""

    def __init__(self, *args: Any, session_factory: Any = TimeoutSession, **kwargs: Any):
        self._diagnostics: deque[dict[str, Any]] = deque(maxlen=MAX_RECORDS)
        self._active_diagnostic: dict[str, Any] | None = None
        self._calls = self._successes = self._failures = 0
        self._counts_saturated = False
        self._first_failure: dict[str, Any] | None = None
        self._timing_seconds = {"total": 0.0, "phases": {name: 0.0 for name in _PHASES}}
        self._timing_saturated = False
        self._session_factory_original = session_factory
        super().__init__(*args, session_factory=self._new_session, **kwargs)

    def _observed_session(self) -> TimeoutSession | None:
        """Override only to unwrap a trusted wrapper to this graph's exact session."""
        session = self.session
        return session if type(session) is TimeoutSession else None

    def _add_count(self, name: str) -> None:
        current = getattr(self, name)
        if current == MAX_SUMMARY_COUNT:
            self._counts_saturated = True
        else:
            setattr(self, name, current + 1)

    def _add_seconds(self, name: str, seconds: float, *, phase: bool = False) -> None:
        target = self._timing_seconds["phases"] if phase else self._timing_seconds
        current = target[name]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            self._timing_saturated = True
            return
        if seconds > MAX_SUMMARY_SECONDS - current:
            target[name] = MAX_SUMMARY_SECONDS
            self._timing_saturated = True
        else:
            target[name] = current + seconds

    def _retain(self, record: dict[str, Any]) -> None:
        projected = _project_record(record)
        if projected is None:
            self.budget.halt("transport_failure")
            raise ConcurrentStop("transport_failure")
        self._add_count("_calls")
        self._add_count("_successes" if projected["status"] == "success" else "_failures")
        if projected["status"] == "failure" and self._first_failure is None:
            self._first_failure = {"call_index": self._calls,
                                   "call_index_saturated": self._counts_saturated,
                                   "record": copy.deepcopy(projected)}
        self._add_seconds("total", projected["total_seconds"])
        for name in _PHASES:
            seconds = projected["phase_seconds"][name]
            if seconds is not None:
                self._add_seconds(name, seconds, phase=True)
        self._diagnostics.append(projected)

    def _new_session(self, *args: Any, **kwargs: Any) -> Any:
        began = time.monotonic()
        try:
            return self._session_factory_original(*args, **kwargs)
        finally:
            if self._active_diagnostic is not None:
                self._active_diagnostic["phase_seconds"]["startup"] = time.monotonic() - began

    def _snapshot_session(self, record: dict[str, Any]) -> bool:
        session = self._observed_session()
        # Rotation closes the preceding session inside the next invocation.
        # A rejected reservation can also leave that session attached. Neither
        # may be presented as evidence for the current call.
        if (session is None or type(session._invocation_started) not in (int, float)
                or session._invocation_started < record["invocation_start_monotonic"]):
            return False
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
        return True

    def _before_cleanup(self) -> None:
        record = self._active_diagnostic
        if record is None or record["precleanup"] is not None:
            return
        if not self._snapshot_session(record):
            return
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
            observed_session = self._observed_session()
            if observed_session is not None:
                observed_session._check_observer()
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
                self._retain(record)
            finally:
                self._active_diagnostic = None

    def diagnostic_records(self) -> tuple[dict[str, Any], ...]:
        """Detached, strictly validated tail; malformed state fails closed."""
        projected = tuple(_project_record(record) for record in self._diagnostics)
        if any(record is None for record in projected):
            raise ValueError("diagnostic_record_invalid")
        return projected

    def diagnostic_summary(self) -> dict[str, Any]:
        """Detached finite per-client summary; no global first-fault claim."""
        value = {"schema": "warm_timeout_summary_v1", "calls": self._calls,
                 "successes": self._successes, "failures": self._failures,
                 "counts_saturated": self._counts_saturated,
                 "first_failure": self._first_failure,
                 "last_record": self._diagnostics[-1] if self._diagnostics else None,
                 "timing_seconds": self._timing_seconds,
                 "timing_saturated": self._timing_saturated}
        projected = _project_summary(value)
        if projected is None:
            raise ValueError("diagnostic_summary_invalid")
        return projected
