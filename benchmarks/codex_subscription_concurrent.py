"""Versioned, explicit shared budgets for parallel experimental Luna turns.

Observed tokens are a stop-before-next-call limit, not an output-token ceiling.
The pinned single-flight transport remains unchanged. No API-key route exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import tempfile
import threading
import time
import types
from typing import Any, Callable

from hymem.extraction.llm import LLMRequest

PINNED_TRANSPORT_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
_base_path = Path(__file__).resolve().with_name("codex_subscription.py")
_base_source = _base_path.read_bytes()
if hashlib.sha256(_base_source).hexdigest() != PINNED_TRANSPORT_SHA256:
    raise RuntimeError("pinned_transport_source_mismatch")
base = types.ModuleType("pinned_codex_subscription_concurrent_base")
base.__file__ = str(_base_path)
exec(compile(_base_source, str(_base_path), "exec"), base.__dict__)


class ConcurrentStop(BaseException):
    """A fixed diagnostic code; never provider text or prompt content."""


def _stop(code: str) -> None:
    raise ConcurrentStop(code)


@dataclass(frozen=True)
class BudgetLimits:
    turns: int
    known_tokens: int
    seconds: float

    def __post_init__(self) -> None:
        if (type(self.turns) is not int or self.turns <= 0
                or type(self.known_tokens) is not int or self.known_tokens <= 0
                or isinstance(self.seconds, bool)
                or not isinstance(self.seconds, (int, float))
                or not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("limits_must_be_finite_positive")


@dataclass
class _Question:
    limits: BudgetLimits
    started_at: float
    turns: int = 0
    reserved: int = 0
    known_tokens: int = 0
    in_flight: int = 0
    usage_complete: bool = True
    stopped: bool = False


class SharedBudget:
    """One thread-safe campaign ledger shared by all registered question clients."""

    def __init__(self, limits: BudgetLimits, *, max_in_flight: int = 2,
                 clock: Callable[[], float] = time.monotonic):
        if type(max_in_flight) is not int or not 1 <= max_in_flight <= 2:
            raise ValueError("in_flight_limit_invalid")
        self.limits = limits
        self.max_in_flight = max_in_flight
        self.clock = clock
        self.started_at = clock()
        self._lock = threading.RLock()
        self._questions: dict[str, _Question] = {}
        self.turns = 0
        self.reserved = 0
        self.known_tokens = 0
        self.in_flight = 0
        self.usage_complete = True
        self.stopped = False
        self.stop_code: str | None = None
        self.timings = {"preflight_seconds": 0.0, "model_seconds": 0.0,
                        "cleanup_seconds": 0.0}

    def register(self, question_id: str, limits: BudgetLimits) -> None:
        if not isinstance(question_id, str) or not question_id or len(question_id) > 128:
            raise ValueError("question_id_invalid")
        with self._lock:
            if question_id in self._questions:
                raise ValueError("question_already_registered")
            if self.stopped:
                _stop("campaign_stopped")
            self._questions[question_id] = _Question(limits, self.clock())

    def _remaining(self, question: _Question) -> float:
        now = self.clock()
        return min(self.limits.seconds - (now - self.started_at),
                   question.limits.seconds - (now - question.started_at))

    def reserve(self, question_id: str) -> float:
        with self._lock:
            q = self._questions[question_id]
            if self.stopped:
                _stop("campaign_stopped")
            if q.stopped:
                _stop("question_stopped")
            if q.in_flight:
                _stop("question_concurrent_invocation")
            if self._remaining(q) <= 0:
                if self.clock() - self.started_at >= self.limits.seconds:
                    self.halt("campaign_wall_limit")
                else:
                    q.stopped = True
                _stop("wall_limit")
            if (self.turns + self.reserved >= self.limits.turns
                    or self.known_tokens >= self.limits.known_tokens):
                self.halt("campaign_budget_exhausted")
                _stop("campaign_budget_exhausted")
            if (q.turns + q.reserved >= q.limits.turns
                    or q.known_tokens >= q.limits.known_tokens):
                q.stopped = True
                _stop("question_budget_exhausted")
            if self.in_flight >= self.max_in_flight:
                _stop("concurrency_limit")
            self.in_flight += 1
            q.in_flight += 1
            self.reserved += 1
            q.reserved += 1
            return min(120.0, self._remaining(q))

    def before_turn(self, question_id: str, admission: dict[str, Any]) -> float:
        """Recheck the shared stop and quota immediately before `_run_turn`."""
        with self._lock:
            q = self._questions[question_id]
            if q.in_flight != 1 or q.reserved != 1 or self.reserved < 1:
                self.halt("ledger_protocol_violation")
                _stop("ledger_protocol_violation")
            if self.stopped or q.stopped:
                _stop("budget_stopped_before_turn")
            if self._remaining(q) <= 0:
                if self.clock() - self.started_at >= self.limits.seconds:
                    self.halt("campaign_wall_limit")
                else:
                    q.stopped = True
                _stop("wall_limit")
            if (self.known_tokens >= self.limits.known_tokens
                    or q.known_tokens >= q.limits.known_tokens):
                if self.known_tokens >= self.limits.known_tokens:
                    self.halt("campaign_budget_exhausted")
                else:
                    q.stopped = True
                _stop("budget_exhausted_before_turn")
            if (admission.get("auth") != "chatgpt" or admission.get("model") != base.MODEL
                    or admission.get("config_isolation_admitted") is not True
                    or admission.get("inference_enabled") is not False):
                self.halt("admission_rejected")
                _stop("admission_rejected")
            windows = admission.get("quota_windows")
            if (not isinstance(windows, list) or not windows
                    or any(not isinstance(w, dict) or isinstance(w.get("remaining_percent"), bool)
                           or not isinstance(w.get("remaining_percent"), (int, float))
                           or not math.isfinite(w["remaining_percent"])
                           or not 25 <= w["remaining_percent"] <= 100 for w in windows)):
                self.halt("quota_unverified")
                _stop("quota_unverified")
            self.reserved -= 1
            q.reserved -= 1
            self.turns += 1
            q.turns += 1
            return min(120.0, self._remaining(q))

    def settle(self, question_id: str, *, used: int | None,
               turn_started: bool, failure: str | None = None,
               preflight_seconds: float = 0.0, model_seconds: float = 0.0,
               cleanup_seconds: float = 0.0) -> None:
        with self._lock:
            q = self._questions[question_id]
            if (q.in_flight != 1 or self.in_flight < 1
                    or (turn_started and q.reserved != 0)
                    or (not turn_started and q.reserved != 1)):
                self.halt("ledger_protocol_violation")
                _stop("ledger_protocol_violation")
            if not turn_started:
                self.reserved -= 1
                q.reserved -= 1
            self.in_flight -= 1
            q.in_flight -= 1
            if turn_started:
                if type(used) is int and used > 0:
                    self.known_tokens += used
                    q.known_tokens += used
                else:
                    self.usage_complete = False
                    q.usage_complete = False
                    failure = failure or "usage_unknown"
            for key, amount in (("preflight_seconds", preflight_seconds),
                                ("model_seconds", model_seconds),
                                ("cleanup_seconds", cleanup_seconds)):
                if math.isfinite(amount) and amount >= 0:
                    self.timings[key] += amount
            if failure:
                self.halt(failure)

    def halt(self, code: str) -> None:
        with self._lock:
            self.stopped = True
            if self.stop_code is None:
                self.stop_code = code

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"turns": self.turns, "reserved": self.reserved,
                    "known_tokens": self.known_tokens,
                    "usage_complete": self.usage_complete and self.in_flight == 0,
                    "in_flight": self.in_flight, "stopped": self.stopped,
                    "stop_code": self.stop_code, "timings": dict(self.timings),
                    "known_tokens_scope": "completed_turns_only_failed_turn_usage_unknown",
                    "token_cap_kind": "stop_before_next_observed_usage",
                    "questions": {key: {"turns": q.turns, "known_tokens": q.known_tokens,
                        "in_flight": q.in_flight,
                        "usage_complete": q.usage_complete and q.in_flight == 0,
                        "stopped": q.stopped}
                        for key, q in self._questions.items()}}


class ConcurrentSubscriptionClient:
    """One registered question; fresh isolated App Server for every completion."""

    def __init__(self, binary: str, budget: SharedBudget, question_id: str,
                 question_limits: BudgetLimits, *, session_factory: Any = base.StdioSession):
        budget.register(question_id, question_limits)
        self.binary = binary
        self.budget = budget
        self.question_id = question_id
        self.session_factory = session_factory
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
        state = self.budget.snapshot()
        return state["questions"][self.question_id]["usage_complete"]

    def complete(self, request: LLMRequest) -> str:
        if not isinstance(request.system, str) or not isinstance(request.user, str):
            self.budget.halt("invalid_request")
            _stop("invalid_request")
        self.requested_controls.append({"temperature_requested": request.temperature,
            "temperature_effective": None, "max_tokens_requested": request.max_tokens,
            "max_tokens_effective": None, "response_format_requested": request.response_format,
            "response_format_effective": None})
        timeout = self.budget.reserve(self.question_id)
        preflight_start = time.monotonic()
        preflight_seconds = model_seconds = cleanup_seconds = 0.0
        turn_started = False
        used: int | None = None
        failure: str | None = None
        session = None
        directory = None
        try:
            directory = tempfile.TemporaryDirectory(prefix="hymem-luna-concurrent-empty-")
            session = self.session_factory(self.binary, directory.name, timeout=timeout)
            admission = base.inspect_preflight(session, base_instructions=request.system)
            preflight_seconds = time.monotonic() - preflight_start
            thread_id = admission.get("_thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                _stop("thread_id_missing")
            self.budget.before_turn(self.question_id, admission)
            turn_started = True  # conservative if turn/start fails
            model_start = time.monotonic()
            try:
                answer, used = base._run_turn(session, thread_id, request.user)
            finally:
                model_seconds = time.monotonic() - model_start
            return answer
        except base.SubscriptionTransportError:
            failure = "transport_or_admission_failure"
            self.budget.halt(failure)
            _stop("transport_or_admission_failure")
        except ConcurrentStop:
            state = self.budget.snapshot()
            if not state["stopped"] and not state["questions"][self.question_id]["stopped"]:
                failure = "transport_or_admission_failure"
                self.budget.halt(failure)
            raise
        except Exception:
            failure = "transport_failure"
            self.budget.halt(failure)
            _stop("transport_failure")
        except BaseException:
            failure = "interrupted_invocation"
            self.budget.halt(failure)
            raise
        finally:
            if preflight_seconds == 0.0:
                preflight_seconds = time.monotonic() - preflight_start
            cleanup_error: BaseException | None = None
            if session is not None:
                cleanup_start = time.monotonic()
                try:
                    session.close()
                except BaseException as exc:
                    cleanup_error = exc
                    failure = "cleanup_failure"
                    self.budget.halt(failure)
                cleanup_seconds = time.monotonic() - cleanup_start
            if directory is not None:
                cleanup_start = time.monotonic()
                try:
                    directory.cleanup()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                    failure = "cleanup_failure"
                    self.budget.halt(failure)
                cleanup_seconds += time.monotonic() - cleanup_start
            self.budget.settle(self.question_id, used=used, turn_started=turn_started,
                failure=failure, preflight_seconds=preflight_seconds,
                model_seconds=model_seconds, cleanup_seconds=cleanup_seconds)
            if cleanup_error is not None:
                if isinstance(cleanup_error, Exception):
                    _stop("cleanup_failure")
                raise cleanup_error

    def chat(self, messages: list[dict[str, str]], **controls: Any) -> str:
        if (len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]
                or set(controls) - {"max_tokens", "temperature", "model", "response_format"}
                or controls.get("model", base.MODEL) != base.MODEL
                or controls.get("response_format", "text") not in {"text", "json"}):
            self.budget.halt("unsupported_chat_control")
            _stop("unsupported_chat_control")
        return self.complete(LLMRequest(system=messages[0]["content"],
            user=messages[1]["content"], response_format=controls.get("response_format", "text"),
            max_tokens=controls.get("max_tokens", 1024),
            temperature=controls.get("temperature", 0.0)))
