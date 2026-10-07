"""Offline-testable admission bridge for the legacy Hermes Codex Responses path.

The managed Codex process may refresh its own ChatGPT credential. It never
performs inference here; the isolated native Responses transport does that.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import tempfile
import threading
import time
from typing import Any, Callable

from hymem.extraction.llm import LLMRequest
from benchmarks import codex_subscription_staged_v6 as staged_v6
from benchmarks import hermes_codex_responses_v2 as native
warm = staged_v6.warm


PINNED_NATIVE_SHA256 = "6845e6a395b40d22cad204bb1e6f4e62001aaa0f9a17609e15051fde620df7f4"
if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != PINNED_NATIVE_SHA256:
    raise RuntimeError("pinned_native_source_mismatch")

MAX_AUTH_BYTES = 64_000
MAX_INVOCATION = 120.0
_BRIDGE_CODES = frozenset({"timeout", "invalid_auth_path", "auth_file_untrusted",
    "invalid_credentials", "account_mismatch", "account_unverified", "model_unverified",
    "auth_expired", "quota_unverified", "broker_closed", "admission_failure",
    "cleanup_failure", "broker_busy", "client_busy", "invalid_request_or_closed",
    "concurrent_completion_rejected", "transport_failure", "bridge_failure"})
_NATIVE_CODES = frozenset({"invalid_request", "request_limit", "invalid_credentials",
    "invalid_timeout", "timeout", "transport_failure", "cleanup_failure",
    "invalid_event", "invalid_usage", "missing_usage", "incomplete_response",
    "invalid_output", "unsupported_output", "output_limit", "event_limit",
    "event_after_completion", "response_failure", "missing_completion", "wire_limit",
    "truncated_stream", "http_failure", "invalid_content_type", "auth_failure",
    "access_failure", "quota_failure", "model_mismatch", "html_response",
    "access_challenge", "unsupported_media_type", "invalid_json"})
_BUDGET_CODES = frozenset({"campaign_stopped", "question_stopped", "question_concurrent_invocation",
    "wall_limit", "campaign_wall_limit", "campaign_budget_exhausted", "question_budget_exhausted",
    "concurrency_limit", "budget_stopped_before_turn", "budget_exhausted_before_turn",
    "admission_rejected", "quota_unverified", "ledger_protocol_violation"})


class BridgeError(RuntimeError):
    """A fixed diagnostic code, never credential or provider material."""


def _fail(code: str) -> None:
    raise BridgeError(code) from None


def _remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if not math.isfinite(seconds) or seconds <= 0:
        _fail("timeout")
    return seconds


def _claims(token: str) -> dict[str, Any]:
    try:
        segment = token.split(".")[1]
        if not segment or len(segment) > 16_000:
            _fail("invalid_credentials")
        claims = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
        if type(claims) is not dict:
            _fail("invalid_credentials")
        return claims
    except (IndexError, ValueError, UnicodeError, TypeError):
        _fail("invalid_credentials")


def _read_auth(path: str) -> tuple[native.Credentials, float, str]:
    if type(path) is not str or not os.path.isabs(path):
        _fail("invalid_auth_path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or not 0 < info.st_size <= MAX_AUTH_BYTES):
                _fail("auth_file_untrusted")
            data = os.read(fd, MAX_AUTH_BYTES + 1)
            if len(data) != info.st_size:
                _fail("auth_file_untrusted")
        finally:
            os.close(fd)
        auth = json.loads(data)
    except (OSError, ValueError, UnicodeError):
        _fail("auth_file_untrusted")
    if (type(auth) is not dict or auth.get("auth_mode") != "chatgpt"
            or auth.get("OPENAI_API_KEY") not in (None, "")
            or type(auth.get("tokens")) is not dict):
        _fail("invalid_credentials")
    tokens = auth["tokens"]
    try:
        credentials = native.Credentials(tokens.get("access_token"), tokens.get("account_id"))
    except native.TransportError:
        _fail("invalid_credentials")
    claims = _claims(credentials.access_token)
    exp = claims.get("exp")
    if type(exp) not in (int, float) or not math.isfinite(exp):
        _fail("invalid_credentials")
    binding = claims.get("https://api.openai.com/auth")
    if type(binding) is not dict or binding.get("chatgpt_account_id") != credentials.account_id:
        _fail("account_mismatch")
    profile = claims.get("https://api.openai.com/profile")
    email_candidates = [claims.get("email")]
    if type(profile) is dict:
        email_candidates.append(profile.get("email"))
    id_token = tokens.get("id_token")
    if id_token is not None:
        if type(id_token) is not str:
            _fail("invalid_credentials")
        id_claims = _claims(id_token)
        id_binding = id_claims.get("https://api.openai.com/auth")
        if type(id_binding) is not dict or id_binding.get("chatgpt_account_id") != credentials.account_id:
            _fail("account_mismatch")
        email_candidates.append(id_claims.get("email"))
    present = [value for value in email_candidates if value is not None]
    if (not present or any(type(value) is not str or not value for value in present)
            or len(set(present)) != 1):
        _fail("account_unverified")
    email = present[0]
    return credentials, float(exp), email


def _check_account(account: Any, credentials: native.Credentials, email: str) -> None:
    if (type(account) is not dict or account.get("type") != "chatgpt"
            or account.get("planType") not in warm.base.SUBSCRIPTION_PLANS):
        _fail("account_unverified")
    if account.get("id") is not None and account["id"] != credentials.account_id:
        _fail("account_mismatch")
    if type(account.get("email")) is not str or account["email"] != email:
        _fail("account_mismatch")


def _check_catalog(catalog: Any) -> None:
    if (type(catalog) is not dict or catalog.get("nextCursor") is not None
            or type(catalog.get("data")) is not list):
        _fail("model_unverified")
    rows = [row for row in catalog["data"] if type(row) is dict and row.get("model") == native.MODEL]
    if (len(rows) != 1 or not any(type(e) is dict and e.get("reasoningEffort") == "low"
                                  for e in rows[0].get("supportedReasoningEfforts", []))):
        _fail("model_unverified")


class AdmissionBroker:
    """One serialized, bounded managed session shared by question workers."""

    def __init__(self, binary: str, auth_path: str, *, session_factory: Callable[..., Any] = warm.WarmSession,
                 max_requests: int = 16, max_age_seconds: float = 300):
        if (type(max_requests) is not int or not 1 <= max_requests <= 16
                or type(max_age_seconds) not in (int, float) or not 1 <= max_age_seconds <= 300):
            raise ValueError("broker_lifetime_invalid")
        if session_factory is warm.WarmSession:
            managed_home = os.environ.get("CODEX_HOME") or (
                os.path.join(os.environ["HOME"], ".codex") if os.environ.get("HOME") else "")
            if (not managed_home or not os.path.isabs(managed_home)
                    or type(auth_path) is not str or not os.path.isabs(auth_path)
                    or os.path.normpath(auth_path) != os.path.normpath(os.path.join(managed_home, "auth.json"))):
                raise ValueError("managed_auth_path_mismatch")
        self.binary, self.auth_path, self.session_factory = binary, auth_path, session_factory
        self.max_requests, self.max_age_seconds = max_requests, float(max_age_seconds)
        self._lock = threading.Lock()
        self._session: Any = None
        self._directory: tempfile.TemporaryDirectory | None = None
        self._requests = 0
        self._closed = False
        self._account_id: str | None = None

    def _watchdog(self, session: Any, deadline: float) -> tuple[threading.Timer | None, threading.Event]:
        """Bound a managed metadata pipe; capture only this session's process group."""
        expired = threading.Event()
        if self.session_factory is not warm.WarmSession:
            return None, expired
        process = session.process
        pid = process.pid
        if type(pid) is not int or pid <= 0:
            _fail("admission_failure")
        def expire() -> None:
            expired.set()
            if self._session is session and process.poll() is None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    pass
        timer = threading.Timer(_remaining(deadline), expire)
        timer.daemon = True
        timer.start()
        return timer, expired

    def _retire(self) -> None:
        session, directory = self._session, self._directory
        self._session = self._directory = None
        self._requests = 0
        try:
            if session is not None:
                session.close()
        finally:
            if directory is not None:
                directory.cleanup()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            _fail("broker_busy")
        try:
            self._closed = True
            self._retire()
        finally:
            self._lock.release()

    def _rpc(self, method: str, params: dict[str, Any], deadline: float) -> dict[str, Any]:
        self._session.set_deadline(deadline)
        result = self._session.rpc(method, params)
        _remaining(deadline)
        if type(result) is not dict:
            _fail("admission_failure")
        return result

    def admit(self, deadline: float) -> tuple[native.Credentials, dict[str, Any]]:
        if not self._lock.acquire(timeout=_remaining(deadline)):
            _fail("timeout")
        timer: threading.Timer | None = None
        expired = threading.Event()
        try:
            if self._closed:
                _fail("broker_closed")
            if self._session is not None and (self._requests >= self.max_requests or
                    time.monotonic() - self._session.created_at >= self.max_age_seconds):
                self._retire()
            if self._session is None:
                self._directory = tempfile.TemporaryDirectory(prefix="hymem-native-admission-")
                self._session = self.session_factory(self.binary, self._directory.name,
                                                     timeout=_remaining(deadline))
                timer, expired = self._watchdog(self._session, deadline)
                self._session.set_deadline(deadline)
                self._rpc("initialize", {"clientInfo": {"name": "hymem_native_admission", "version": "0.1"},
                                         "capabilities": {"experimentalApi": True}}, deadline)
                self._session.send("initialized", {}, notification=True)
                _remaining(deadline)
            else:
                timer, expired = self._watchdog(self._session, deadline)
            before, before_expiry, _ = _read_auth(self.auth_path)
            needs_refresh = before_expiry <= time.time() + _remaining(deadline)
            account = self._rpc("account/read", {"refreshToken": needs_refresh}, deadline).get("account")
            credentials, expiry, email = _read_auth(self.auth_path)
            if credentials.account_id != before.account_id:
                _fail("account_mismatch")
            if expiry <= time.time() + _remaining(deadline):
                _fail("auth_expired")
            if self._account_id is not None and credentials.account_id != self._account_id:
                _fail("account_mismatch")
            _check_account(account, credentials, email)
            _check_catalog(self._rpc("model/list", {"includeHidden": True}, deadline))
            try:
                windows = warm.quota_metadata(self._rpc("account/rateLimits/read", {}, deadline))
            except warm.base.SubscriptionTransportError:
                _fail("quota_unverified")
            _remaining(deadline)
            if expired.is_set():
                _fail("timeout")
            self._account_id = credentials.account_id
            self._requests += 1
            return credentials, {"auth": "chatgpt", "model": native.MODEL,
                                 "quota_windows": windows, "config_isolation_admitted": True,
                                 "inference_enabled": False,
                                 "isolation_basis": "direct_fixed_https_no_agent_thread"}
        except BaseException as exc:
            try:
                self._retire()
            except BaseException:
                _fail("cleanup_failure")
            if expired.is_set():
                _fail("timeout")
            if isinstance(exc, BridgeError):
                raise
            _fail("admission_failure")
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            self._lock.release()


class NativeLMEClient:
    """A registered question client; ordinary and staged calls share its ledger."""

    def __init__(self, broker: AdmissionBroker, budget: warm.SharedBudget, question_id: str,
                 question_limits: warm.BudgetLimits, *, transport: Callable[..., native.Completed] = native.complete):
        budget.register(question_id, question_limits)
        self.broker, self.budget, self.question_id = broker, budget, question_id
        self.transport = transport
        self.requested_controls: list[dict[str, Any]] = []
        self.internal_http_attempts = None
        self._flight = threading.Lock()
        self._closed = False
        self.successes = 0
        self.failures = 0
        self.last_failure_code: str | None = None
        self.first_failure: dict[str, Any] | None = None
        self._calls = 0
        self._total_seconds = 0.0
        self._preflight_seconds = 0.0
        self._model_seconds = 0.0
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

    def diagnostic_summary(self) -> dict[str, Any]:
        """Finite native-only observations; never includes prompts or identities.

        Turns and known_tokens are for the shared question ledger. When a
        runner displays ordinary and staged views, it must not sum those fields.
        """
        question = self.budget.snapshot()["questions"][self.question_id]
        first = dict(self.first_failure) if self.first_failure is not None else None
        return {"schema": "native_oauth_summary_v1", "calls": self._calls,
                "successes": self.successes, "failures": self.failures,
                "turns": question["turns"], "known_tokens": question["known_tokens"],
                "usage_complete": question["usage_complete"],
                "timing_seconds": {"total": round(self._total_seconds, 6),
                                   "admission": round(self._preflight_seconds, 6),
                                   "http": round(self._model_seconds, 6)},
                "timing_saturated": self._timing_saturated,
                "first_failure": first, "last_failure_code": self.last_failure_code}

    def _add_timing(self, name: str, seconds: float) -> None:
        old = getattr(self, name)
        if not math.isfinite(seconds) or seconds < 0 or old + seconds > 1_000_000.0:
            self._timing_saturated = True
            setattr(self, name, 1_000_000.0)
        else:
            setattr(self, name, old + seconds)

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

    def _complete(self, request: LLMRequest, schema: dict[str, Any] | None) -> str:
        if not self._flight.acquire(blocking=False):
            self.budget.halt("concurrent_completion_rejected")
            _fail("concurrent_completion_rejected")
        try:
            if self._closed or type(request) is not LLMRequest:
                _fail("invalid_request_or_closed")
            controls = {"temperature_requested": request.temperature, "temperature_effective": None,
                        "max_tokens_requested": request.max_tokens, "max_tokens_effective": None,
                        "response_format_requested": request.response_format, "response_format_effective": None,
                        "output_schema_sent": False, "output_schema_acknowledged": False}
            self.requested_controls.append(controls)
            # Build/validate before reservation; the native transport builds it again
            # in its isolated child, preserving that source's size and schema checks.
            native.build_request(request.system, request.user, schema)
            started_at = time.monotonic()
            limit = self.budget.reserve(self.question_id)
            self._calls += 1
            deadline = started_at + min(MAX_INVOCATION, limit)
            admitted = False
            used = None
            failure = None
            preflight = model = 0.0
            try:
                credentials, admission = self.broker.admit(deadline)
                preflight = time.monotonic() - started_at
                self.budget.before_turn(self.question_id, admission)
                admitted = True
                model_start = time.monotonic()
                try:
                    completed = self.transport(credentials, request.system, request.user, schema,
                                               timeout=min(MAX_INVOCATION, _remaining(deadline)))
                finally:
                    model = time.monotonic() - model_start
                if (type(completed) is not native.Completed or type(completed.total_tokens) is not int
                        or completed.total_tokens <= 0 or type(completed.text) is not str
                        or not completed.text):
                    _fail("transport_failure")
                used = completed.total_tokens
                controls["output_schema_sent"] = schema is not None
                controls["output_schema_acknowledged"] = schema is not None
                self.successes += 1
                return completed.text
            except BaseException as exc:
                self.failures += 1
                if isinstance(exc, BridgeError):
                    candidate = str(exc)
                    failure = candidate if candidate in _BRIDGE_CODES else "bridge_failure"
                elif isinstance(exc, native.TransportError):
                    candidate = str(exc)
                    failure = candidate if candidate in _NATIVE_CODES else "transport_failure"
                elif isinstance(exc, warm.ConcurrentStop):
                    candidate = exc.args[0] if len(exc.args) == 1 and type(exc.args[0]) is str else None
                    failure = candidate if candidate in _BUDGET_CODES else "budget_failure"
                else:
                    failure = "bridge_failure"
                self.last_failure_code = failure
                if self.first_failure is None:
                    self.first_failure = {"code": failure, "phase": "http" if admitted else "admission",
                                          "turn_admitted": admitted, "known_usage": used is not None}
                self.budget.record_first_failure(failure, {key: value for key, value in
                    self.first_failure.items() if key != "code"})
                if isinstance(exc, warm.ConcurrentStop) and failure != "budget_failure":
                    raise
                _fail(failure)
            finally:
                if preflight == 0:
                    preflight = time.monotonic() - started_at
                self._add_timing("_total_seconds", time.monotonic() - started_at)
                self._add_timing("_preflight_seconds", preflight)
                self._add_timing("_model_seconds", model)
                self.budget.settle(self.question_id, used=used, turn_started=admitted,
                                   failure=failure, preflight_seconds=preflight, model_seconds=model)
        finally:
            self._flight.release()
