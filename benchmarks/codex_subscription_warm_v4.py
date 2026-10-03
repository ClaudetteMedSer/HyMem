"""Bounded, text-free observation of incomplete strict warm turns.

The frozen parser is still the sole authority for acceptance and usage. This
module observes events it consumes; an observed shape is not an attestation that
the parser accepted that event or that usage was settled.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import time
import types
from typing import Any

_v3_path = Path(__file__).resolve().with_name("codex_subscription_warm_v3.py")
_v3_source = _v3_path.read_bytes()
if hashlib.sha256(_v3_source).hexdigest() != "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d":
    raise RuntimeError("pinned_warm_v3_source_mismatch")
v3 = types.ModuleType("pinned_codex_subscription_warm_v4_base")
v3.__file__ = str(_v3_path)
sys.modules[v3.__name__] = v3
exec(compile(_v3_source, str(_v3_path), "exec"), v3.__dict__)

base = v3.base
concurrent = v3.concurrent
BudgetLimits = v3.BudgetLimits
ConcurrentStop = v3.ConcurrentStop
SharedBudget = v3.SharedBudget

_STATES = frozenset({"absent", "zero", "positive"})
_KNOWN_EVENTS = v3._EVENTS | v3._TURN_ACCEPTED
_FAMILIES = {name.replace("/", "_") for name in _KNOWN_EVENTS} | {"unknown", "invalid_method"}


def _event_family(method: Any) -> str:
    if type(method) is not str:
        return "invalid_method"
    return method.replace("/", "_") if method in _KNOWN_EVENTS else "unknown"


def _project_observation(value: Any) -> dict[str, Any] | None:
    if type(value) is not dict or value.get("basis") != "consumed_observed_shape":
        return None
    count = value.get("events_consumed")
    final_count = value.get("final_count")
    usage_count = value.get("usage_update_count")
    completed = value.get("completed_seen")
    final = value.get("final_seen")
    usage = value.get("usage_state")
    family = value.get("last_event_family")
    if (type(count) is not int or not 0 <= count <= base.MAX_EVENTS
            or type(final_count) is not int or not 0 <= final_count <= count
            or type(usage_count) is not int or not 0 <= usage_count <= count
            or type(completed) is not bool or type(final) is not bool
            or final != (final_count > 0)
            or type(usage) is not str or usage not in _STATES
            or (usage == "absent") != (usage_count == 0)
            or (family is not None and (type(family) is not str or family not in _FAMILIES))
            or (count == 0) != (family is None)):
        return None
    return {"basis": "consumed_observed_shape", "events_consumed": count,
            "completed_seen": completed, "final_seen": final,
            "final_count": final_count, "usage_update_count": usage_count,
            "usage_state": usage, "last_event_family": family}


def serialize_failure(first_failure: Any) -> dict[str, Any] | None:
    """Project only finite labels and counts, including the new turn snapshot."""
    out = v3.serialize_failure(first_failure)
    if (out is not None and type(first_failure) is dict
            and out.get("code") == "incomplete_turn_or_usage"
            and out.get("phase") == "run" and out.get("rpc") == "turn/events"):
        observation = _project_observation(first_failure.get("turn_observation"))
        if observation is not None:
            out["turn_observation"] = observation
    return out


class WarmSession(v3.WarmSession):
    def __init__(self, binary: str, cwd: str, timeout: float = 120):
        super().__init__(binary, cwd, timeout)
        self.turn_observation: dict[str, Any] | None = None
        self._observation_thread: str | None = None
        self._observation_turn: str | None = None

    def reset_turn_observation(self) -> None:
        self.turn_observation = None
        self._observation_thread = None
        self._observation_turn = None

    def rpc(self, method: str, params: dict[str, Any], *, preserve_notifications: bool = False) -> dict[str, Any]:
        if method == "turn/start":
            self.reset_turn_observation()
        result = super().rpc(method, params, preserve_notifications=preserve_notifications)
        if method == "turn/start":
            thread = params.get("threadId") if type(params) is dict else None
            turn = result.get("turn") if type(result) is dict else None
            turn_id = turn.get("id") if type(turn) is dict else None
            if (type(thread) is str and thread and thread == getattr(self, "active_thread", None)
                    and type(turn_id) is str and turn_id):
                self._observation_thread = thread
                self._observation_turn = turn_id
                self.turn_observation = {
                    "basis": "consumed_observed_shape", "events_consumed": 0,
                    "completed_seen": False, "final_seen": False, "final_count": 0,
                    "usage_update_count": 0, "usage_state": "absent",
                    "last_event_family": None,
                }
        return result

    def next_event(self) -> dict[str, Any]:
        event = super().next_event()
        state = getattr(self, "turn_observation", None)
        if type(state) is not dict:
            return event
        count = state["events_consumed"]
        if count >= base.MAX_EVENTS:
            return event
        state["events_consumed"] = count + 1
        method = event.get("method")
        state["last_event_family"] = _event_family(method)
        data = event.get("params")
        if type(data) is not dict:
            return event
        thread = getattr(self, "_observation_thread", None)
        turn = getattr(self, "_observation_turn", None)
        if method == "thread/tokenUsage/updated" and data.get("threadId") == thread and data.get("turnId") == turn:
            total = data.get("tokenUsage")
            total = total.get("total") if type(total) is dict else None
            total = total.get("totalTokens") if type(total) is dict else None
            if type(total) is int and total >= 0:
                state["usage_update_count"] += 1
                state["usage_state"] = "positive" if total > 0 else "zero"
        elif method == "item/completed" and data.get("threadId") == thread and data.get("turnId") == turn:
            item = data.get("item")
            if (type(item) is dict and item.get("type") == "agentMessage"
                    and item.get("phase") == "final_answer" and type(item.get("text")) is str):
                state["final_count"] += 1
                state["final_seen"] = True
        elif method == "turn/completed" and data.get("threadId") == thread:
            observed_turn = data.get("turn")
            if (type(observed_turn) is dict and observed_turn.get("id") == turn
                    and observed_turn.get("status") == "completed"):
                state["completed_seen"] = True
        return event


class WarmSubscriptionClient(v3.WarmSubscriptionClient):
    def __init__(self, *args: Any, session_factory: Any = WarmSession, **kwargs: Any):
        super().__init__(*args, session_factory=session_factory, **kwargs)

    def _complete_locked(self, request: Any) -> str:
        session = self.session
        if session is not None:
            reset = getattr(session, "reset_turn_observation", None)
            if callable(reset):
                reset()
        return super()._complete_locked(request)

    def _record_failure(self, code: str, phase: str, turn_admitted: bool,
                        known_usage: bool) -> None:
        session = self.session
        event = getattr(session, "last_event", None)
        rpc = getattr(session, "stage", None) if session is not None else None
        age = (time.monotonic() - session.created_at
               if session is not None and type(getattr(session, "created_at", None)) in (int, float)
               else None)
        if age is not None and (not math.isfinite(age) or age < 0):
            age = None
        retired = getattr(session, "retired_threads", ()) if session is not None else ()
        pending = getattr(session, "pending", ()) if session is not None else ()
        starting = getattr(session, "starting_events", ()) if session is not None else ()
        state = self.budget.snapshot()
        metadata = {
            "phase": phase, "rpc": rpc if type(rpc) is str and rpc in v3.v2._RPC_METHODS else None,
            "process_age_seconds": round(min(age, 1_000_000.0), 3) if age is not None else None,
            "process_index": self.processes_started, "request_index": self.requests_on_process + 1,
            "retired_count": min(len(retired), base.MAX_EVENTS),
            "queue_count": min(len(pending) + len(starting), base.MAX_EVENTS),
            "turn_admitted": turn_admitted, "known_usage": known_usage,
            "known_tokens": state["known_tokens"], "usage_complete": state["usage_complete"],
        }
        if type(event) is dict:
            projected = serialize_failure({"code": code, **event}) or {}
            if "event_family" in projected:
                metadata["event_family"] = projected["event_family"]
                if code == "fixed_other":
                    metadata["failure_family"] = "unexpected_notification"
            if "app_server_error" in projected:
                metadata["app_server_error"] = projected["app_server_error"]
        rpc_error = getattr(session, "rpc_error", None)
        if type(rpc_error) is dict and rpc != "turn/events":
            projected = serialize_failure({"code": code, "rpc_error": rpc_error}) or {}
            if "rpc_error" in projected:
                metadata["rpc_error"] = projected["rpc_error"]
        if code == "incomplete_turn_or_usage" and phase == "run" and rpc == "turn/events" and turn_admitted:
            observation = _project_observation(getattr(session, "turn_observation", None))
            if observation is not None:
                metadata["turn_observation"] = observation
        self.budget.record_first_failure(code, metadata)
