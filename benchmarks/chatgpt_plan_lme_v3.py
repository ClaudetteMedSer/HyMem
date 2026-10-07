"""Bounded SIWC LME bridge. Importing this module performs no network work.

The VM credential owner authorizes each turn. The public Responses transport
owns the HTTP request; this adapter owns the shared question ledger and finite
diagnostic projection. No Codex quota window is inferred for SIWC.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable

from benchmarks import chatgpt_plan_responses_v8 as transport
from benchmarks import chatgpt_plan_responses_v7 as transport_v7
from benchmarks import chatgpt_plan_responses_v6 as transport_v6
from benchmarks import codex_subscription_staged_v6 as staged_v6
from tools.diagnostics import lme_chatgpt_plan_owner_v1 as owner
from hymem.extraction.llm import LLMRequest

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_PINS = ((transport, "benchmarks/chatgpt_plan_responses_v8.py",
          "f560f283852202ad15dead1d7fedb94ae5c361c03019973798b92f482964699b", "complete"),
         (transport_v7, "benchmarks/chatgpt_plan_responses_v7.py",
          "00ccb579bd38a8eb9b21eb05d4d83667de4a699ac2b63dba610cc94550ebea64", "complete"),
         (transport_v6, "benchmarks/chatgpt_plan_responses_v6.py",
          "811bff13ebc4b24ebd22cad16542c3b58597dc04538a1d7deb5085df7190a28f", "parse_stream_events"),
         (owner, "tools/diagnostics/lme_chatgpt_plan_owner_v1.py",
          "a893541f2a8a0c8f2ece0c00f4a4da4fbb763f9876c51632555535389aadb982", "CredentialBroker"))
for _module, _relative, _digest, _symbol in _PINS:
    _path = _SOURCE_ROOT / _relative
    _object = getattr(_module, _symbol, None)
    _code = _object.acquire if _symbol == "CredentialBroker" and _object is not None else _object
    if (Path(getattr(_module, "__file__", "")).resolve() != _path.resolve()
            or hashlib.sha256(_path.read_bytes()).hexdigest() != _digest
            or Path(getattr(getattr(_code, "__code__", None), "co_filename", "")).resolve() != _path.resolve()):
        raise RuntimeError("pinned_siwc_source_mismatch")
if transport._v7 is not transport_v7 or transport_v7._v6 is not transport_v6:
    raise RuntimeError("pinned_siwc_source_mismatch")

warm = staged_v6.warm
MAX_INVOCATION = 120.0
POLICY = owner.POLICY
_CODES = frozenset({"timeout", "admission_rejected", "invalid_request", "invalid_request_or_closed",
    "concurrent_completion_rejected", "client_busy", "transport_failure", "bridge_failure",
    "campaign_stopped", "question_stopped", "question_concurrent_invocation", "wall_limit",
    "campaign_wall_limit", "campaign_budget_exhausted", "question_budget_exhausted",
    "concurrency_limit", "budget_stopped_before_turn", "budget_exhausted_before_turn",
    "ledger_protocol_violation", "cleanup_failure", "budget_failure",
    "resource_observer_unverified", "resource_task_denial"}) | owner.ERRORS | transport_v6.PROVIDER_CODES | transport_v6._v1._CHILD_CODES


class BridgeError(RuntimeError):
    """Only a fixed code crosses the bridge boundary."""


def _fail(code: str) -> None:
    raise BridgeError(code if code in _CODES else "bridge_failure") from None


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if not math.isfinite(left) or left <= 0:
        _fail("timeout")
    return left


class _AdmissionProof:
    __slots__ = ("secret", "broker", "lease", "question_id", "deadline", "identity", "used")

    def __init__(self, secret: object, broker: owner.CredentialBroker,
                 lease: owner.CredentialLease, question_id: str, deadline: float):
        self.secret, self.broker, self.lease = secret, broker, lease
        self.question_id, self.deadline = question_id, deadline
        self.identity = broker.identity_digest
        self.used = False


class SharedBudget(warm.SharedBudget):
    """The pinned v9 ledger with SIWC lease admission in place of quota windows."""

    def __init__(self, limits: warm.BudgetLimits, *, max_in_flight: int = 4,
                 clock: Callable[[], float] = time.monotonic):
        if type(max_in_flight) is not int or not 1 <= max_in_flight <= 4:
            raise ValueError("in_flight_limit_invalid")
        super().__init__(limits, max_in_flight=min(max_in_flight, 2), clock=clock)
        self.max_in_flight = max_in_flight
        self._siwc_proof_secret = object()
        self._siwc_owner: owner.CredentialBroker | None = None
        self._siwc_identity: str | None = None

    def bind_owner(self, broker: owner.CredentialBroker) -> None:
        identity = broker.identity_digest
        if (type(broker) is not owner.CredentialBroker or type(identity) is not str
                or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity)):
            raise ValueError("siwc_identity_invalid")
        with self._lock:
            if self._siwc_owner is None:
                self._siwc_owner = broker
                self._siwc_identity = identity
            elif self._siwc_owner is not broker or self._siwc_identity != identity:
                raise ValueError("siwc_identity_changed")

    def before_turn(self, question_id: str, proof: _AdmissionProof) -> float:
        with self._lock:
            q = self._questions[question_id]
            if q.in_flight != 1 or q.reserved != 1 or self.reserved < 1:
                self.halt("ledger_protocol_violation")
                warm.concurrent._stop("ledger_protocol_violation")
            if self.stopped or q.stopped:
                warm.concurrent._stop("budget_stopped_before_turn")
            if self._remaining(q) <= 0:
                if self.clock() - self.started_at >= self.limits.seconds:
                    self.halt("campaign_wall_limit")
                else:
                    q.stopped = True
                warm.concurrent._stop("wall_limit")
            if self.known_tokens >= self.limits.known_tokens or q.known_tokens >= q.limits.known_tokens:
                if self.known_tokens >= self.limits.known_tokens:
                    self.halt("campaign_budget_exhausted")
                else:
                    q.stopped = True
                warm.concurrent._stop("budget_exhausted_before_turn")
            valid = (type(proof) is _AdmissionProof and proof.secret is self._siwc_proof_secret
                     and not proof.used and proof.question_id == question_id
                     and type(proof.broker) is owner.CredentialBroker
                     and proof.broker is self._siwc_owner
                     and type(proof.lease) is owner.CredentialLease
                     and proof.lease.policy == POLICY
                     and type(proof.identity) is str and len(proof.identity) == 64
                     and all(c in "0123456789abcdef" for c in proof.identity)
                     and proof.identity == self._siwc_identity
                     and proof.identity == proof.broker.identity_digest
                     and type(proof.lease.expires_at) is int
                     and proof.lease.expires_at > time.time() + max(0, proof.deadline - time.monotonic()) + 30
                     and time.monotonic() < proof.deadline)
            if not valid:
                self.halt("admission_rejected")
                warm.concurrent._stop("admission_rejected")
            proof.used = True
            self.reserved -= 1
            q.reserved -= 1
            self.turns += 1
            q.turns += 1
            return min(MAX_INVOCATION, self._remaining(q), proof.deadline - time.monotonic())


def _safe_observation(exc: transport.TransportError, failure: str) -> dict[str, Any]:
    """Revalidate mutable exception fields at the ledger boundary."""
    out: dict[str, Any] = {}
    status = getattr(exc, "http_status", None)
    if type(status) is int and 100 <= status <= 599:
        out["http_status"] = status
    for key, choices in (("body_shape", transport_v6._v2._BODY_SHAPES),
                         ("media_type_class", transport_v6._v2._MEDIA_CLASSES)):
        value = getattr(exc, key, None)
        if type(value) is str and value in choices:
            out[key] = value
    wire = transport_v6._v3._sanitize_observation(getattr(exc, "wire_observation", None))
    if wire is not None:
        out["wire_observation"] = wire
    stream = transport_v6._sanitize_stream_observation(getattr(exc, "stream_observation", None))
    if stream is not None:
        out["stream_observation"] = stream
    if failure == "timeout":
        timeout = transport.sanitize_timeout_observation(getattr(exc, "timeout_observation", None))
        if timeout is not None:
            out["timeout_observation"] = timeout
    return out


class SIWCLMEClient:
    """One registered question view; use RegistrationAlias for its second view."""

    def __init__(self, broker: owner.CredentialBroker, budget: SharedBudget, question_id: str,
                 question_limits: warm.BudgetLimits,
                 *, response_call: Callable[..., transport.Completed] = transport.complete):
        # The runner's RegistrationAlias delegates to one already registered
        # SharedBudget, so ordinary and structured views use one question slot.
        ledger = budget if isinstance(budget, SharedBudget) else getattr(budget, "_budget", None)
        if type(broker) is not owner.CredentialBroker or not isinstance(ledger, SharedBudget):
            raise ValueError("siwc_owner_or_budget_invalid")
        ledger.bind_owner(broker)
        budget.register(question_id, question_limits)
        self.broker, self.budget, self.question_id = broker, budget, question_id
        self.response_call = response_call
        self.requested_controls: list[dict[str, Any]] = []
        self._flight = threading.Lock()
        self._closed = False
        self.calls = self.successes = self.failures = self.internal_http_attempts = 0
        self.first_failure: dict[str, Any] | None = None
        self.last_failure_code: str | None = None
        self._total = self._admission = self._http = 0.0
        self._timing_saturated = False

    @property
    def observed_turns(self) -> int:
        return self.budget.snapshot()["questions"][self.question_id]["turns"]

    @property
    def observed_tokens(self) -> int:
        return self.budget.snapshot()["questions"][self.question_id]["known_tokens"]

    @property
    def usage_complete(self) -> bool:
        return self.budget.snapshot()["questions"][self.question_id]["usage_complete"]

    def close(self) -> None:
        if not self._flight.acquire(blocking=False):
            _fail("client_busy")
        try:
            self._closed = True
        finally:
            self._flight.release()

    def _timing(self, attr: str, seconds: float) -> None:
        current = getattr(self, attr)
        if not math.isfinite(seconds) or seconds < 0 or current + seconds > 1_000_000:
            self._timing_saturated = True
            setattr(self, attr, 1_000_000.0)
        else:
            setattr(self, attr, current + seconds)

    def diagnostic_summary(self) -> dict[str, Any]:
        q = self.budget.snapshot()["questions"][self.question_id]
        return {"schema": "siwc_lme_summary_v2", "calls": self.calls,
                "successes": self.successes, "failures": self.failures,
                "internal_http_attempts": self.internal_http_attempts,
                "provider_internal_retries_known": False,
                "admitted_turns": q["turns"], "known_tokens": q["known_tokens"],
                "usage_complete": q["usage_complete"],
                "timing_seconds": {"total": round(self._total, 6),
                                   "admission": round(self._admission, 6), "http": round(self._http, 6)},
                "timing_saturated": self._timing_saturated,
                "first_failure": copy.deepcopy(self.first_failure) if self.first_failure else None,
                "last_failure_code": self.last_failure_code}

    def complete(self, request: LLMRequest) -> str:
        return self._complete(request, None)

    def complete_stage(self, request: LLMRequest, batch: Any, stage: str, recheck: bool) -> str:
        if type(stage) is not str or type(recheck) is not bool:
            raise staged_v6.staged.GroundingContractError("stage:invalid")
        if stage == "original":
            staged_v6.staged.validate_original_request(request, batch)
            schema = staged_v6.staged.build_original_output_schema(batch)
        elif stage == "alternatives" and not recheck:
            staged_v6.staged.validate_alternatives_request(request, batch)
            schema = staged_v6.staged.build_alternatives_output_schema(batch)
        else:
            raise staged_v6.staged.GroundingContractError("stage:invalid")
        return self._complete(request, schema)

    def _pre_admission_failure(self, code: str) -> None:
        """An invocation failed before reservation; there is no turn to settle."""
        if code not in _CODES:
            code = "bridge_failure"
        self.calls += 1
        self.failures += 1
        self.last_failure_code = code
        if self.first_failure is None:
            self.first_failure = {"code": code, "phase": "admission",
                                  "turn_admitted": False, "unknown_usage": False}
        self.budget.record_first_failure(code, {key: value for key, value in
            self.first_failure.items() if key != "code"})
        _fail(code)

    def _complete(self, request: LLMRequest, schema: dict[str, Any] | None) -> str:
        if not self._flight.acquire(blocking=False):
            self.budget.halt("concurrent_completion_rejected")
            _fail("concurrent_completion_rejected")
        try:
            if self._closed or type(request) is not LLMRequest:
                self._pre_admission_failure("invalid_request_or_closed")
            controls = {"temperature_requested": request.temperature, "temperature_effective": None,
                        "max_tokens_requested": request.max_tokens, "max_tokens_effective": None,
                        "response_format_requested": request.response_format, "response_format_effective": None,
                        "output_schema_sent": False, "output_schema_acknowledged": False}
            self.requested_controls.append(controls)
            try:
                transport.build_request(request.system, request.user, schema)
            except transport.TransportError as exc:
                self._pre_admission_failure(exc.code)
            except Exception:
                self._pre_admission_failure("bridge_failure")
            started = time.monotonic()
            limit = self.budget.reserve(self.question_id)
            self.calls += 1
            deadline = started + min(MAX_INVOCATION, limit)
            admitted = False
            used: int | None = None
            failure: str | None = None
            admission_seconds = http_seconds = 0.0
            try:
                lease = self.broker.acquire(caller_deadline=deadline)
                admission_seconds = time.monotonic() - started
                credentials = transport.Credentials(lease.access_token)
                proof = _AdmissionProof(self.budget._siwc_proof_secret, self.broker, lease,
                                        self.question_id, deadline)
                self.budget.before_turn(self.question_id, proof)
                admitted = True
                http_started = time.monotonic()
                timeout = _remaining(deadline)
                self.internal_http_attempts += 1
                try:
                    completed = self.response_call(credentials, request.system, request.user,
                                                   schema, timeout=timeout)
                finally:
                    http_seconds = time.monotonic() - http_started
                if (type(completed) is not transport.Completed or type(completed.total_tokens) is not int
                        or completed.total_tokens <= 0 or type(completed.text) is not str or not completed.text):
                    _fail("transport_failure")
                used = completed.total_tokens
                controls["output_schema_sent"] = schema is not None
                controls["output_schema_acknowledged"] = schema is not None
                self.successes += 1
                return completed.text
            except BaseException as exc:
                self.failures += 1
                if isinstance(exc, owner.OwnerError):
                    failure = exc.code
                elif isinstance(exc, (transport.TransportError, transport._v1.TransportError,
                                      transport._v2.TransportError, transport._v3.TransportError)):
                    failure = exc.code
                elif isinstance(exc, warm.ConcurrentStop):
                    code = exc.args[0] if len(exc.args) == 1 else None
                    failure = code if type(code) is str and code in _CODES else "budget_failure"
                elif isinstance(exc, BridgeError):
                    failure = str(exc)
                else:
                    failure = "bridge_failure"
                if failure not in _CODES:
                    failure = "bridge_failure"
                self.last_failure_code = failure
                if self.first_failure is None:
                    self.first_failure = {"code": failure, "phase": "http" if admitted else "admission",
                                          "turn_admitted": admitted, "unknown_usage": admitted and used is None}
                    if type(exc) is transport.TransportError:
                        self.first_failure.update(_safe_observation(exc, failure))
                self.budget.record_first_failure(failure, copy.deepcopy({key: value for key, value in
                    self.first_failure.items() if key != "code"}))
                _fail(failure)
            finally:
                if admission_seconds == 0:
                    admission_seconds = time.monotonic() - started
                self._timing("_total", time.monotonic() - started)
                self._timing("_admission", admission_seconds)
                self._timing("_http", http_seconds)
                self.budget.settle(self.question_id, used=used, turn_started=admitted,
                                   failure=failure, preflight_seconds=admission_seconds,
                                   model_seconds=http_seconds)
        finally:
            self._flight.release()


_SUMMARY_KEYS = frozenset({"schema", "calls", "successes", "failures", "internal_http_attempts",
    "provider_internal_retries_known", "admitted_turns", "known_tokens", "usage_complete",
    "timing_seconds", "timing_saturated", "first_failure", "last_failure_code"})


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_typed_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def project_first_failure(value: Any) -> dict[str, Any] | None:
    """Sanitize a client or resource-observed ledger fault for a future reader."""
    if value is None:
        return None
    required = {"code", "phase", "turn_admitted", "unknown_usage"}
    optional = {"http_status", "body_shape", "media_type_class", "wire_observation",
                "stream_observation", "timeout_observation", "resource_observation", "underlying_code"}
    if (type(value) is not dict or not required <= value.keys() or value.keys() - required - optional
            or type(value["code"]) is not str or value["code"] not in _CODES
            or type(value["phase"]) is not str or value["phase"] not in {"admission", "http"}
            or type(value["turn_admitted"]) is not bool or type(value["unknown_usage"]) is not bool
            or value["turn_admitted"] != (value["phase"] == "http")
            or value["unknown_usage"] != value["turn_admitted"]):
        raise ValueError("siwc_first_failure_invalid")
    if "http_status" in value and (type(value["http_status"]) is not int or not 100 <= value["http_status"] <= 599):
        raise ValueError("siwc_first_failure_invalid")
    for key, choices in (("body_shape", transport_v6._v2._BODY_SHAPES),
                         ("media_type_class", transport_v6._v2._MEDIA_CLASSES)):
        if key in value and (type(value[key]) is not str or value[key] not in choices):
            raise ValueError("siwc_first_failure_invalid")
    if "wire_observation" in value and (type(value["wire_observation"]) is not dict or not _typed_equal(
            transport_v6._v3._sanitize_observation(value["wire_observation"]), value["wire_observation"])):
        raise ValueError("siwc_first_failure_invalid")
    if "stream_observation" in value and (type(value["stream_observation"]) is not dict or not _typed_equal(
            transport_v6._sanitize_stream_observation(value["stream_observation"]), value["stream_observation"])):
        raise ValueError("siwc_first_failure_invalid")
    if "timeout_observation" in value:
        timeout = value["timeout_observation"]
        if (value["phase"] != "http" or not value["turn_admitted"] or not value["unknown_usage"]
                or (value["code"] != "timeout" and value.get("underlying_code") != "timeout")
                or type(timeout) is not dict or not _typed_equal(
                    transport.sanitize_timeout_observation(timeout), timeout)):
            raise ValueError("siwc_first_failure_invalid")
    if "resource_observation" in value:
        sample = value["resource_observation"]
        if sample is not None and (type(sample) is not dict
                or sample.keys() != {"current", "peak", "limit", "denials"}
                or any(type(sample[key]) is not int or not 0 <= sample[key] <= 1_000_000 for key in sample)
                or sample["limit"] != 256 or sample["current"] > sample["peak"]
                or sample["peak"] > sample["limit"]):
            raise ValueError("siwc_first_failure_invalid")
    if "underlying_code" in value and (type(value["underlying_code"]) is not str
                                       or value["underlying_code"] not in _CODES):
        raise ValueError("siwc_first_failure_invalid")
    return copy.deepcopy(value)


def validate_summary_projection(value: Any) -> dict[str, Any]:
    """Strict finite reader boundary for one public client projection."""
    if type(value) is not dict or value.keys() != _SUMMARY_KEYS or value["schema"] != "siwc_lme_summary_v2":
        raise ValueError("siwc_summary_invalid")
    for key in ("calls", "successes", "failures", "internal_http_attempts", "admitted_turns", "known_tokens"):
        ceiling = 1_000_000_000_000 if key == "known_tokens" else 1_000_000
        if type(value[key]) is not int or not 0 <= value[key] <= ceiling:
            raise ValueError("siwc_summary_invalid")
    if (value["calls"] != value["successes"] + value["failures"]
            or value["internal_http_attempts"] > value["admitted_turns"]
            or value["successes"] > value["internal_http_attempts"]
            or value["provider_internal_retries_known"] is not False
            or type(value["usage_complete"]) is not bool
            or type(value["timing_saturated"]) is not bool):
        raise ValueError("siwc_summary_invalid")
    timing = value["timing_seconds"]
    if type(timing) is not dict or timing.keys() != {"total", "admission", "http"}:
        raise ValueError("siwc_summary_invalid")
    if any(type(timing[key]) not in (int, float) or not math.isfinite(timing[key])
           or not 0 <= timing[key] <= 1_000_000 for key in timing):
        raise ValueError("siwc_summary_invalid")
    first = value["first_failure"]
    if first is not None:
        project_first_failure(first)
    if (value["last_failure_code"] is not None and
            (type(value["last_failure_code"]) is not str or value["last_failure_code"] not in _CODES)
            or (value["failures"] == 0) != (first is None)
            or (value["failures"] == 0) != (value["last_failure_code"] is None)):
        raise ValueError("siwc_summary_invalid")
    return value


def validate_pilot_projection(value: Any) -> dict[str, Any]:
    """Check one canary pair and four question pairs without double-counting views."""
    if (type(value) is not dict or value.keys() != {"schema", "canary", "questions", "aggregate"}
            or value["schema"] != "siwc_lme_pilot_projection_v2"
            or type(value["canary"]) is not dict
            or type(value["questions"]) is not list or len(value["questions"]) != 4):
        raise ValueError("siwc_pilot_projection_invalid")
    rows = [value["canary"], *value["questions"]]
    ids: set[str] = set()
    calls = successes = failures = attempts = turns = tokens = 0
    for row in rows:
        if type(row) is not dict or row.keys() != {"question_id", "ordinary", "structured", "ledger"}:
            raise ValueError("siwc_pilot_projection_invalid")
        qid = row["question_id"]
        if type(qid) is not str or not qid or len(qid) > 128 or qid in ids:
            raise ValueError("siwc_pilot_projection_invalid")
        ids.add(qid)
        a = validate_summary_projection(row["ordinary"])
        b = validate_summary_projection(row["structured"])
        ledger = row["ledger"]
        if (type(ledger) is not dict or ledger.keys() != {"admitted_turns", "known_tokens", "usage_complete"}
                or type(ledger["admitted_turns"]) is not int or ledger["admitted_turns"] < 0
                or type(ledger["known_tokens"]) is not int or ledger["known_tokens"] < 0
                or type(ledger["usage_complete"]) is not bool
                or any(view[key] != ledger[key] for view in (a, b)
                       for key in ("admitted_turns", "known_tokens", "usage_complete"))
                or a["internal_http_attempts"] + b["internal_http_attempts"] > ledger["admitted_turns"]):
            raise ValueError("siwc_pilot_projection_invalid")
        calls += a["calls"] + b["calls"]
        successes += a["successes"] + b["successes"]
        failures += a["failures"] + b["failures"]
        attempts += a["internal_http_attempts"] + b["internal_http_attempts"]
        turns += ledger["admitted_turns"]
        tokens += ledger["known_tokens"]
    aggregate = value["aggregate"]
    if (type(aggregate) is not dict or aggregate.keys() != {"calls", "successes", "failures",
            "internal_http_attempts", "admitted_turns", "known_tokens"}
            or any(type(number) is not int or not 0 <= number <= (
                1_000_000_000_000 if key == "known_tokens" else 1_000_000)
                   for key, number in aggregate.items())
            or aggregate != {"calls": calls, "successes": successes, "failures": failures,
                             "internal_http_attempts": attempts, "admitted_turns": turns,
                             "known_tokens": tokens}):
        raise ValueError("siwc_pilot_projection_invalid")
    return value
