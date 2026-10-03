"""Versioned warm App Server transport with fixed-code first-failure attribution.

The process is reused only between completed calls. Any uncertainty stops the
shared campaign; this module never reconnects or retries an invocation.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
from typing import Any

from hymem.extraction.llm import LLMRequest

PINNED_CONCURRENT_SHA256 = "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0"
_concurrent_path = Path(__file__).resolve().with_name("codex_subscription_concurrent_v2.py")
_concurrent_source = _concurrent_path.read_bytes()
if hashlib.sha256(_concurrent_source).hexdigest() != PINNED_CONCURRENT_SHA256:
    raise RuntimeError("pinned_concurrent_source_mismatch")
concurrent = types.ModuleType("pinned_codex_subscription_warm_concurrent")
concurrent.__file__ = str(_concurrent_path)
sys.modules[concurrent.__name__] = concurrent
exec(compile(_concurrent_source, str(_concurrent_path), "exec"), concurrent.__dict__)

base = concurrent.base
BudgetLimits = concurrent.BudgetLimits
ConcurrentStop = concurrent.ConcurrentStop

# Only literals raised by the pinned transport and this router may cross the
# diagnostic boundary. In particular, a syntactically safe server method is
# still not a permitted suffix unless it is one of these known methods.
_FIXED_CODES = frozenset({
    "account_changed", "account_notification_invalid", "active_thread_closed",
    "binary_version_mismatch", "binary_version_unverified",
    "capability_isolation_unverified", "credit_balance_present", "duplicate_item",
    "event_limit", "exact_model_unavailable", "final_message_invalid",
    "incomplete_model_catalog", "incomplete_turn_or_usage", "instruction_isolation_unverified",
    "invalid_quota", "invalid_request", "item_id_invalid", "item_identity_mismatch", "item_lifecycle_invalid",
    "mcp_isolation_unverified", "message_delta_invalid", "model_or_search_unverified",
    "notification_invalid", "output_limit", "pilot_budget_exhausted", "pilot_stopped",
    "process_not_reaped", "protocol_failure", "provider_endpoint_override", "quota_exhausted",
    "quota_floor", "reasoning_delta_invalid", "remote_control_enabled",
    "retired_thread_closed_invalid", "retired_thread_status_invalid",
    "routing_unverified", "runtime_isolation_not_accepted", "service_tier_unverified", "startup_failed",
    "subscription_auth_required", "subscription_plan_unverified", "thread_id_missing",
    "thread_identity_mismatch", "thread_isolation_unverified", "thread_start_status_invalid",
    "timeout", "tool_or_extra_item", "turn_failed", "turn_identity_mismatch",
    "turn_start_invalid", "unexpected_response", "unexpected_server_request",
    "unknown_quota", "unsubscribe_thread_mismatch", "unsubscribe_unverified",
    "unsupported_chat_control", "unsupported_chat_history", "usage_identity_mismatch",
    "usage_invalid", "usage_regressed", "warning_limit", "transport_failure",
    "warning_thread_unverified", "warning_unapproved", "concurrent_completion_rejected",
})
_RPC_METHODS = frozenset({
    "initialize", "config/read", "model/list", "account/read", "account/rateLimits/read",
    "thread/start", "turn/start", "thread/unsubscribe", "startup", "turn/events",
})
_EVENT_METHODS = frozenset({
    "thread/started", "thread/status/changed", "thread/closed", "turn/started",
    "turn/completed", "turn/failed", "item/started", "item/completed", "item/agentMessage/delta",
    "item/reasoning/textDelta", "thread/tokenUsage/updated", "account/updated", "account/rateLimits/updated",
    "remoteControl/status/changed", "warning", "invalid_method",
})
_OWN_BUDGET_CODES = frozenset({
    "ledger_protocol_violation", "wall_limit", "campaign_wall_limit",
    "campaign_budget_exhausted", "question_budget_exhausted", "budget_exhausted_before_turn",
    "admission_rejected", "quota_unverified", "cleanup_failure",
})


def _safe_code(exc: BaseException) -> str:
    value = exc.args[0] if len(exc.args) == 1 and type(exc.args[0]) is str else None
    if value in _FIXED_CODES:
        return value
    if type(value) is str and ":" in value:
        family, suffix = value.split(":", 1)
        if family in {"process_exit", "protocol_failure", "rpc_failure"} and suffix in _RPC_METHODS:
            return value
        if family == "unexpected_notification" and suffix in _EVENT_METHODS:
            return value
    return "fixed_other"


class SharedBudget(concurrent.SharedBudget):
    """Pinned ledger with one immutable, sanitized first-fault record."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.first_failure: dict[str, Any] | None = None

    def record_first_failure(self, code: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            if self.first_failure is None and (self.stop_code is None or self.stop_code == code):
                self.first_failure = {"code": code, **metadata}
            self.halt(code)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = super().snapshot()
            state["first_failure"] = (dict(self.first_failure)
                                      if self.first_failure is not None else None)
            return state


# The runner constructs its budget through warm.concurrent, so keep that
# factory on the attributed type while retaining the pinned stop class.
concurrent.SharedBudget = SharedBudget


class WarmSession(base.StdioSession):
    """Strict single flight notification router around the pinned turn parser."""

    def __init__(self, binary: str, cwd: str, timeout: float = 120):
        super().__init__(binary, cwd, timeout)
        self.created_at = time.monotonic()
        self.initialized_result: dict[str, Any] | None = None
        self.initialized_sent = False
        self.active_thread: str | None = None
        self.starting_threads: list[str] = []
        self.starting_events: list[dict[str, Any]] = []
        self.started_notifications: set[str] = set()
        self.seen_threads: set[str] = set()
        self.retired_threads: set[str] = set()
        self.closed = False

    def set_deadline(self, deadline: float) -> None:
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            base._fail("timeout")
        self.deadline = deadline

    def send(self, method: str, params: dict[str, Any], *, notification: bool = False) -> int:
        if time.monotonic() >= self.deadline:
            base._fail("timeout")
        if method == "initialized" and notification:
            if self.initialized_sent:
                return self.next_id
            self.initialized_sent = True
        return super().send(method, params, notification=notification)

    def _route(self, event: dict[str, Any], method: str, preserve: bool) -> None:
        name = event.get("method")
        data = event.get("params")
        if not isinstance(data, dict):
            base._fail("notification_invalid")
        if name in {"thread/status/changed", "thread/closed"}:
            tid = data.get("threadId")
            if method == "thread/start" and self.active_thread is None and tid not in self.retired_threads:
                if len(self.starting_events) >= base.MAX_EVENTS:
                    base._fail("event_limit")
                self.starting_events.append(event)
                return
            if tid in self.retired_threads or (method == "thread/unsubscribe" and tid == self.active_thread):
                if name == "thread/status/changed":
                    status = data.get("status")
                    if not isinstance(status, dict) or status.get("type") not in {"idle", "notLoaded"}:
                        base._fail("retired_thread_status_invalid")
                elif set(data) != {"threadId"}:
                    base._fail("retired_thread_closed_invalid")
                return
            if tid != self.active_thread:
                base._fail("thread_identity_mismatch")
            if method == "thread/start" and name == "thread/status/changed":
                if data.get("status") != {"type": "idle"}:
                    base._fail("thread_start_status_invalid")
                return
        if name == "thread/started":
            tid = data.get("thread", {}).get("id") if isinstance(data.get("thread"), dict) else None
            if (not isinstance(tid, str) or not tid or tid in self.retired_threads
                    or tid in self.started_notifications):
                base._fail("thread_identity_mismatch")
            if method == "thread/start" and self.active_thread is None:
                self.starting_threads.append(tid)
            elif tid != self.active_thread:
                base._fail("thread_identity_mismatch")
            self.started_notifications.add(tid)
            return
        if name == "remoteControl/status/changed":
            if data.get("status") != "disabled":
                base._fail("remote_control_enabled")
            return
        if name in {"account/updated", "account/rateLimits/updated"}:
            base._validate_account_notification(event)
            return
        if name == "warning":
            target = base._validate_warning(event, self.active_thread)
            if method != "thread/start" and self.active_thread is None:
                base._fail("warning_unapproved")
            if len(self.warning_targets) >= base.MAX_BENIGN_WARNINGS:
                base._fail("warning_limit")
            self.warning_targets.append(target)
            return
        if name == "thread/closed":
            base._fail("active_thread_closed")
        # An in-flight turn owns its notifications; its pinned parser verifies
        # turn and item identities. Other phases may see only lifecycle notices.
        if preserve and method == "turn/start":
            self.pending.append(event)
            return
        if name == "thread/started" and method == "thread/start":
            return
        base._fail("unexpected_notification:" + base._safe_method(name))

    def rpc(self, method: str, params: dict[str, Any], *, preserve_notifications: bool = False) -> dict[str, Any]:
        if method == "initialize" and self.initialized_result is not None:
            return self.initialized_result
        self.stage = method
        request_id = self.send(method, params)
        for _ in range(4096):
            event = self.receive()
            if event.get("id") == request_id:
                if "error" in event or not isinstance(event.get("result"), dict):
                    base._fail("rpc_failure:" + method)
                result = event["result"]
                if method == "initialize":
                    self.initialized_result = result
                elif method == "thread/start":
                    tid = result.get("thread", {}).get("id")
                    if (not isinstance(tid, str) or not tid or tid in self.seen_threads
                            or tid in self.retired_threads or
                            any(other != tid for other in self.starting_threads)):
                        base._fail("thread_identity_mismatch")
                    self.seen_threads.add(tid)
                    self.starting_threads.clear()
                    self.active_thread = tid
                    for pending in self.starting_events:
                        self._route(pending, "thread/start", False)
                    self.starting_events.clear()
                elif method == "thread/unsubscribe":
                    if result.get("status") != "unsubscribed" or self.active_thread != params.get("threadId"):
                        base._fail("unsubscribe_unverified")
                    self.retired_threads.add(self.active_thread)
                    self.active_thread = None
                    self.bound_thread_id = None
                    self.warning_targets.clear()
                return result
            if "id" in event:
                base._fail("unexpected_response")
            self._route(event, method, preserve_notifications)
        base._fail("event_limit")

    def next_event(self) -> dict[str, Any]:
        for _ in range(base.MAX_EVENTS):
            event = self.pending.popleft() if self.pending else self.receive()
            if event.get("method") == "thread/started":
                self._route(event, "turn/events", False)
                continue
            if event.get("method") in {"thread/status/changed", "thread/closed"}:
                data = event.get("params", {})
                if isinstance(data, dict) and data.get("threadId") in self.retired_threads:
                    self._route(event, "turn/events", False)
                    continue
            return event
        base._fail("event_limit")

    def retire_pending(self) -> None:
        for _ in range(base.MAX_EVENTS):
            if not self.pending:
                return
            event = self.pending.popleft()
            if "id" in event:
                base._fail("unexpected_response")
            self._route(event, "thread/unsubscribe", False)
        base._fail("event_limit")

    def bind_thread_id(self, thread_id: str) -> None:
        if (thread_id != self.active_thread or
                any(target != thread_id for target in self.warning_targets)):
            base._fail("warning_thread_unverified")
        self.bound_thread_id = thread_id

    def unsubscribe(self, thread_id: str) -> None:
        if thread_id != self.active_thread:
            base._fail("unsubscribe_thread_mismatch")
        self.retire_pending()
        self.rpc("thread/unsubscribe", {"threadId": thread_id})

    def close(self) -> None:
        if self.closed:
            return
        errors = []
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            errors.append(exc)
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        except BaseException as exc:
            errors.append(exc)
        # The parent may have exited while descendants in its group survived.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            errors.append(exc)
        try:
            self.process.wait(timeout=2)
        except BaseException as exc:
            errors.append(exc)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as exc:
                    errors.append(exc)
        if self.process.poll() is None:
            base._fail("process_not_reaped")
        if errors:
            base._fail("cleanup_failure")
        self.closed = True


class WarmSubscriptionClient:
    """One question worker; at most one App Server and one active thread."""

    def __init__(self, binary: str, budget: SharedBudget, question_id: str,
                 question_limits: BudgetLimits, *, session_factory: Any = WarmSession,
                 max_requests: int = 16, max_age_seconds: float = 300):
        if (type(max_requests) is not int or not 1 <= max_requests <= 16 or
                isinstance(max_age_seconds, bool) or
                not isinstance(max_age_seconds, (int, float)) or
                not math.isfinite(max_age_seconds) or not 1 <= max_age_seconds <= 300):
            raise ValueError("warm_lifetime_invalid")
        budget.register(question_id, question_limits)
        self.binary, self.budget, self.question_id = binary, budget, question_id
        self.session_factory = session_factory
        self.max_requests, self.max_age_seconds = max_requests, float(max_age_seconds)
        self.session: WarmSession | None = None
        self.directory: tempfile.TemporaryDirectory | None = None
        self.requests_on_process = 0
        self.processes_started = 0
        self.rotations = 0
        self.cold_calls = 0
        self.warm_calls = 0
        self.startup_seconds = 0.0
        self.unsubscribe_seconds = 0.0
        self.rotation_cleanup_seconds = 0.0
        self.final_cleanup_seconds = 0.0
        self.closed = False
        self._flight = threading.Lock()
        self.internal_http_attempts = None
        self.requested_controls: list[dict[str, Any]] = []

    @property
    def observed_turns(self) -> int:
        return self.budget.snapshot()["questions"][self.question_id]["turns"]

    @property
    def observed_tokens(self) -> int:
        return self.budget.snapshot()["questions"][self.question_id]["known_tokens"]

    @property
    def usage_complete(self) -> bool:
        return self.budget.snapshot()["questions"][self.question_id]["usage_complete"]

    def _record_failure(self, code: str, phase: str, turn_admitted: bool,
                        known_usage: bool) -> None:
        session = self.session
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
            "phase": phase,
            "rpc": rpc if type(rpc) is str and rpc in _RPC_METHODS else None,
            "process_age_seconds": round(min(age, 1_000_000.0), 3) if age is not None else None,
            "process_index": self.processes_started,
            "request_index": self.requests_on_process + 1,
            "retired_count": min(len(retired), base.MAX_EVENTS),
            "queue_count": min(len(pending) + len(starting), base.MAX_EVENTS),
            "turn_admitted": turn_admitted,
            "known_usage": known_usage,
            "known_tokens": state["known_tokens"],
            "usage_complete": state["usage_complete"],
        }
        self.budget.record_first_failure(code, metadata)

    def _close_process(self) -> None:
        try:
            if self.session is not None:
                self.session.close()
                self.session = None
            if self.directory is not None:
                self.directory.cleanup()
                self.directory = None
        except BaseException:
            self.budget.halt("cleanup_failure")
            concurrent._stop("cleanup_failure")

    def close(self) -> None:
        if not self._flight.acquire(blocking=False):
            self.budget.halt("close_during_invocation")
            concurrent._stop("close_during_invocation")
        try:
            close_start = time.monotonic()
            try:
                self._close_process()
            except concurrent.ConcurrentStop as exc:
                if len(exc.args) == 1 and exc.args[0] == "cleanup_failure":
                    self._record_failure("cleanup_failure", "cleanup", False, True)
                raise
            self.final_cleanup_seconds += time.monotonic() - close_start
            self.closed = True
        finally:
            self._flight.release()

    def complete(self, request: LLMRequest) -> str:
        if not self._flight.acquire(blocking=False):
            self.budget.halt("concurrent_completion_rejected")
            concurrent._stop("concurrent_completion_rejected")
        try:
            return self._complete_locked(request)
        finally:
            self._flight.release()

    def _complete_locked(self, request: LLMRequest) -> str:
        if self.closed or not isinstance(request.system, str) or not isinstance(request.user, str):
            self.budget.halt("invalid_request_or_closed")
            concurrent._stop("invalid_request_or_closed")
        self.requested_controls.append({"temperature_requested": request.temperature,
            "temperature_effective": None, "max_tokens_requested": request.max_tokens,
            "max_tokens_effective": None, "response_format_requested": request.response_format,
            "response_format_effective": None})
        invocation_start = time.monotonic()
        timeout = self.budget.reserve(self.question_id)
        invocation_deadline = invocation_start + timeout
        start = time.monotonic()
        preflight_seconds = model_seconds = cleanup_seconds = 0.0
        turn_started = False
        used: int | None = None
        failure: str | None = None
        aborted = False
        answer: str | None = None
        phase = "startup"
        try:
            if self.session is not None and (self.requests_on_process >= self.max_requests or
                    time.monotonic() - self.session.created_at >= self.max_age_seconds):
                phase = "rotation_cleanup"
                rotation_start = time.monotonic()
                self._close_process()
                self.rotation_cleanup_seconds += time.monotonic() - rotation_start
                self.rotations += 1
            if self.session is None:
                phase = "startup"
                startup_start = time.monotonic()
                self.directory = tempfile.TemporaryDirectory(prefix="hymem-luna-warm-empty-")
                self.session = self.session_factory(self.binary, self.directory.name,
                    timeout=max(0.001, invocation_deadline - time.monotonic()))
                self.startup_seconds += time.monotonic() - startup_start
                self.processes_started += 1
                self.requests_on_process = 0
                self.cold_calls += 1
            else:
                self.warm_calls += 1
            # Age is checked between calls. The current call retains its full
            # invocation budget; owning question cleanup closes idle servers.
            self.session.set_deadline(min(invocation_deadline,
                self.session.created_at + self.max_age_seconds + 120))
            phase = "preflight"
            admission = base.inspect_preflight(self.session, base_instructions=request.system)
            preflight_seconds = time.monotonic() - start
            thread_id = admission.get("_thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                base._fail("thread_id_missing")
            self.budget.before_turn(self.question_id, admission)
            turn_started = True
            phase = "run"
            model_start = time.monotonic()
            try:
                answer, used = base._run_turn(self.session, thread_id, request.user)
            finally:
                model_seconds = time.monotonic() - model_start
            unsubscribe_start = time.monotonic()
            phase = "unsubscribe"
            try:
                self.session.unsubscribe(thread_id)
            finally:
                self.unsubscribe_seconds += time.monotonic() - unsubscribe_start
            self.requests_on_process += 1
        except concurrent.ConcurrentStop as exc:
            aborted = True
            state = self.budget.snapshot()
            own_code = exc.args[0] if len(exc.args) == 1 and type(exc.args[0]) is str else None
            if (own_code in _OWN_BUDGET_CODES and
                    (not state["stopped"] or state["stop_code"] == own_code)):
                self._record_failure(own_code, phase, turn_started, used is not None or not turn_started)
            if not state["stopped"] and not state["questions"][self.question_id]["stopped"]:
                failure = "transport_or_admission_failure"
                self.budget.halt(failure)
            raise
        except base.SubscriptionTransportError as exc:
            aborted = True
            prior_stop = self.budget.snapshot()["stopped"]
            code = _safe_code(exc)
            self._record_failure(code, phase, turn_started, used is not None or not turn_started)
            failure = "campaign_stopped" if prior_stop else code
        except Exception:
            aborted = True
            prior_stop = self.budget.snapshot()["stopped"]
            self._record_failure("transport_failure", phase, turn_started,
                                 used is not None or not turn_started)
            failure = "campaign_stopped" if prior_stop else "transport_failure"
        except BaseException:
            aborted = True
            failure = "interrupted_invocation"
            self.budget.halt(failure)
            raise
        finally:
            if preflight_seconds == 0.0:
                preflight_seconds = time.monotonic() - start
            if aborted or self.budget.snapshot()["stopped"]:
                phase = "cleanup"
                cleanup_start = time.monotonic()
                try:
                    self._close_process()
                except BaseException:
                    failure = "cleanup_failure"
                    self._record_failure(failure, phase, turn_started, used is not None or not turn_started)
                cleanup_seconds = time.monotonic() - cleanup_start
            self.budget.settle(self.question_id, used=used, turn_started=turn_started,
                failure=failure, preflight_seconds=preflight_seconds,
                model_seconds=model_seconds, cleanup_seconds=cleanup_seconds)
        if failure:
            concurrent._stop(failure)
        if answer is None:
            concurrent._stop("incomplete_answer")
        return answer

    def chat(self, messages: list[dict[str, str]], **controls: Any) -> str:
        if (len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]
                or set(controls) - {"max_tokens", "temperature", "model", "response_format"}
                or controls.get("model", base.MODEL) != base.MODEL
                or controls.get("response_format", "text") not in {"text", "json"}):
            self.budget.halt("unsupported_chat_control")
            concurrent._stop("unsupported_chat_control")
        return self.complete(LLMRequest(system=messages[0]["content"],
            user=messages[1]["content"], response_format=controls.get("response_format", "text"),
            max_tokens=controls.get("max_tokens", 1024),
            temperature=controls.get("temperature", 0.0)))
