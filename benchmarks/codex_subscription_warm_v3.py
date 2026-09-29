"""Offline, source-bound diagnostic adapter for the strict warm transport.

No notification is accepted here. The pinned turn parser remains the authority;
this module only describes its first failure with finite public vocabulary.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import time
import types
from typing import Any

_v2_path = Path(__file__).resolve().with_name("codex_subscription_warm_v2.py")
_v2_source = _v2_path.read_bytes()
if hashlib.sha256(_v2_source).hexdigest() != "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593":
    raise RuntimeError("pinned_warm_v2_source_mismatch")
v2 = types.ModuleType("pinned_codex_subscription_warm_v3_base")
v2.__file__ = str(_v2_path)
sys.modules[v2.__name__] = v2
exec(compile(_v2_source, str(_v2_path), "exec"), v2.__dict__)

base = v2.base
concurrent = v2.concurrent
BudgetLimits = v2.BudgetLimits
ConcurrentStop = v2.ConcurrentStop
SharedBudget = v2.SharedBudget
_FIXED_CODES = v2._FIXED_CODES
_OWN_BUDGET_CODES = v2._OWN_BUDGET_CODES
_RPC_METHODS = v2._RPC_METHODS
_EVENT_METHODS = v2._EVENT_METHODS

_EVENTS = frozenset(v2._EVENT_METHODS | {
    "error", "model/rerouted", "model/verification", "model/safetyBuffering/updated",
    "turn/plan/updated", "turn/diff/updated",
})
_STRING_ERRORS = frozenset({
    "contextWindowExceeded", "sessionBudgetExceeded", "usageLimitExceeded",
    "rateLimitExceeded", "flexUnavailable", "serverOverloaded", "cyberPolicy",
    "misalignmentPolicyViolation", "internalServerError", "unauthorized",
    "badRequest", "threadRollbackFailed", "sandboxError", "other",
})
_OBJECT_ERRORS = frozenset({
    "httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected", "responseTooManyFailedAttempts",
})
_RPC_ERROR_CODES = {-32700: "parse_error", -32600: "invalid_request",
                    -32601: "method_not_found", -32602: "invalid_params",
                    -32603: "internal_error"}
_TURN_ACCEPTED = frozenset({
    "thread/started", "remoteControl/status/changed", "warning",
    "account/updated", "account/rateLimits/updated", "thread/status/changed",
    "turn/started", "thread/tokenUsage/updated", "item/agentMessage/delta",
    "item/reasoning/summaryPartAdded", "item/reasoning/summaryTextDelta",
    "item/reasoning/textDelta", "item/started", "item/completed", "turn/completed",
})


def _event_family(method: Any) -> str:
    if type(method) is not str:
        return "invalid_method"
    if method in _EVENTS:
        return method.replace("/", "_")
    return "unknown"


def _app_error(event: dict[str, Any], thread_id: str | None,
               turn_id: str | None) -> dict[str, Any]:
    """Return only schema-bounded fields; identifiers and prose never escape."""
    params = event.get("params")
    if type(params) is not dict:
        return {"identity": "invalid"}
    if (type(params.get("threadId")) is not str or not params["threadId"]
            or type(params.get("turnId")) is not str or not params["turnId"]):
        return {"identity": "invalid"}
    if thread_id is None or turn_id is None:
        return {"identity": "unbound"}
    if params["threadId"] != thread_id or params["turnId"] != turn_id:
        return {"identity": "mismatch"}
    result: dict[str, Any] = {"identity": "matched"}
    retry = params.get("willRetry")
    result["will_retry"] = retry if type(retry) is bool else "invalid"
    error = params.get("error")
    if type(error) is not dict or type(error.get("message")) is not str:
        result["error_class"] = "invalid"
        return result
    info = error.get("codexErrorInfo")
    if info is None:
        result["error_class"] = "unspecified"
    elif type(info) is str:
        result["error_class"] = info if info in _STRING_ERRORS else "invalid"
    elif type(info) is dict and len(info) == 1 and "activeTurnNotSteerable" in info:
        detail = info["activeTurnNotSteerable"]
        result["error_class"] = ("activeTurnNotSteerable"
                                 if type(detail) is dict and type(detail.get("turnKind")) is str
                                 and detail["turnKind"] in {"review", "compact"}
                                 else "invalid")
    elif type(info) is dict and len(info) == 1:
        key, detail = next(iter(info.items()))
        if key not in _OBJECT_ERRORS or type(detail) is not dict:
            result["error_class"] = "invalid"
        else:
            status = detail.get("httpStatusCode")
            result["error_class"] = key
            result["http_status_code"] = (status if type(status) is int and 0 <= status <= 65535
                                          else None if status is None else "invalid")
    else:
        result["error_class"] = "invalid"
    return result


def _rpc_error(event: dict[str, Any]) -> dict[str, Any]:
    error = event.get("error")
    if type(error) is not dict or type(error.get("code")) is not int:
        return {"category": "invalid"}
    code = error["code"]
    if not -2147483648 <= code <= 2147483647:
        return {"category": "invalid"}
    return {"category": _RPC_ERROR_CODES.get(code, "other_numeric"), "code": code}


def serialize_failure(first_failure: Any) -> dict[str, Any] | None:
    """Finite public projection for a future runner/reader boundary."""
    if type(first_failure) is not dict:
        return None
    code = first_failure.get("code")
    safe_codes = {"unexpected_notification_unknown"} | {
        "unexpected_notification:" + _event_family(name) for name in _EVENTS}
    out: dict[str, Any] = {"code": code if type(code) is str and
                           (code in safe_codes or v2._safe_code(base.SubscriptionTransportError(code)) == code)
                           else "fixed_other"}
    phase = first_failure.get("phase")
    out["phase"] = phase if type(phase) is str and phase in {"startup", "preflight", "run", "unsubscribe",
                                     "rotation_cleanup", "cleanup"} else "unknown"
    rpc = first_failure.get("rpc")
    out["rpc"] = rpc if type(rpc) is str and rpc in v2._RPC_METHODS else None
    family = first_failure.get("event_family")
    families = {_event_family(name) for name in _EVENTS} | {"unknown", "invalid_method"}
    if type(family) is str and family in families:
        out["event_family"] = family
    if first_failure.get("failure_family") == "unexpected_notification":
        out["failure_family"] = "unexpected_notification"
    for key in ("process_index", "request_index", "retired_count", "queue_count", "known_tokens"):
        value = first_failure.get(key)
        if type(value) is int and 0 <= value <= 1_000_000_000:
            out[key] = value
    age = first_failure.get("process_age_seconds")
    if type(age) in (int, float) and math.isfinite(age) and 0 <= age <= 1_000_000:
        out["process_age_seconds"] = age
    for key in ("turn_admitted", "known_usage", "usage_complete"):
        value = first_failure.get(key)
        if type(value) is bool:
            out[key] = value
    value = first_failure.get("app_server_error")
    if type(value) is dict:
        identity = value.get("identity")
        if type(identity) is str and identity in {"invalid", "unbound", "mismatch", "matched"}:
            detail: dict[str, Any] = {"identity": identity}
            if identity == "matched":
                retry = value.get("will_retry")
                if type(retry) is bool or retry == "invalid":
                    detail["will_retry"] = retry
                error_class = value.get("error_class")
                if type(error_class) is str and error_class in (_STRING_ERRORS | _OBJECT_ERRORS | {"activeTurnNotSteerable", "invalid", "unspecified"}):
                    detail["error_class"] = error_class
                status = value.get("http_status_code")
                if (type(status) is int and 0 <= status <= 65535) or status is None or status == "invalid":
                    if "http_status_code" in value:
                        detail["http_status_code"] = status
            out["app_server_error"] = detail
    value = first_failure.get("rpc_error")
    if type(value) is dict:
        category = value.get("category")
        if type(category) is str and category in set(_RPC_ERROR_CODES.values()) | {"other_numeric", "invalid"}:
            detail = {"category": category}
            number = value.get("code")
            if type(number) is int and -2147483648 <= number <= 2147483647:
                detail["code"] = number
            out["rpc_error"] = detail
    return out


class WarmSession(v2.WarmSession):
    """Observe exact events and RPC errors without changing parser decisions."""

    def __init__(self, binary: str, cwd: str, timeout: float = 120):
        super().__init__(binary, cwd, timeout)
        self.active_turn: str | None = None
        self.last_event: dict[str, Any] | None = None
        self.rpc_error: dict[str, Any] | None = None
        self.request_id: int | None = None

    def send(self, method: str, params: dict[str, Any], *, notification: bool = False) -> int:
        if method == "thread/unsubscribe":
            self.active_turn = None
        request_id = super().send(method, params, notification=notification)
        if not notification:
            self.request_id = request_id
            self.rpc_error = None
        return request_id

    def receive(self) -> dict[str, Any]:
        event = super().receive()
        if (type(event.get("id")) is int and event["id"] == getattr(self, "request_id", None)
                and "error" in event):
            self.rpc_error = _rpc_error(event)
        return event

    def rpc(self, method: str, params: dict[str, Any], *, preserve_notifications: bool = False) -> dict[str, Any]:
        try:
            result = super().rpc(method, params, preserve_notifications=preserve_notifications)
        finally:
            self.request_id = None
        if method == "turn/start":
            turn = result.get("turn")
            self.active_turn = turn.get("id") if type(turn) is dict and type(turn.get("id")) is str else None
        elif method == "thread/unsubscribe":
            self.active_turn = None
        return result

    def _observe(self, event: dict[str, Any]) -> None:
        family = _event_family(event.get("method"))
        self.last_event = {"event_family": family}
        if event.get("method") == "error":
            self.last_event["app_server_error"] = _app_error(
                event, getattr(self, "active_thread", None), getattr(self, "active_turn", None))

    def _route(self, event: dict[str, Any], method: str, preserve: bool) -> None:
        try:
            return super()._route(event, method, preserve)
        except base.SubscriptionTransportError as exc:
            if (len(exc.args) == 1 and type(exc.args[0]) is str
                    and exc.args[0].startswith("unexpected_notification:")):
                self._observe(event)
            raise

    def next_event(self) -> dict[str, Any]:
        self.stage = "turn/events"
        event = super().next_event()
        if event.get("method") not in _TURN_ACCEPTED:
            self._observe(event)
        else:
            self.last_event = None
        return event


class WarmSubscriptionClient(v2.WarmSubscriptionClient):
    def __init__(self, *args: Any, session_factory: Any = WarmSession, **kwargs: Any):
        super().__init__(*args, session_factory=session_factory, **kwargs)

    def _complete_locked(self, request: Any) -> str:
        if self.session is not None:
            self.session.last_event = None
            self.session.rpc_error = None
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
            "phase": phase, "rpc": rpc if type(rpc) is str and rpc in v2._RPC_METHODS else None,
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
        self.budget.record_first_failure(code, metadata)
