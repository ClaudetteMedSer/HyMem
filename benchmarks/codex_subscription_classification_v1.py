"""Inactive, source-bound output-schema transport for classification v1.

This adapter changes only the matching turn/start parameters. The pinned warm
transport still owns admission, parsing, accounting, deadlines and cleanup.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
import sys
import types
from typing import Any

_warm_path = Path(__file__).resolve().with_name("codex_subscription_warm_v3.py")
_warm_source = _warm_path.read_bytes()
if hashlib.sha256(_warm_source).hexdigest() != "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d":
    raise RuntimeError("pinned_warm_v3_source_mismatch")
warm = types.ModuleType("pinned_codex_subscription_classification_warm_v3")
warm.__file__ = str(_warm_path)
sys.modules[warm.__name__] = warm
exec(compile(_warm_source, str(_warm_path), "exec"), warm.__dict__)

_contract_path = Path(__file__).resolve().parents[1] / "hymem/extraction/grounding_classification_v1.py"
if hashlib.sha256(_contract_path.read_bytes()).hexdigest() != "7c62c6e58305a5b7be256825a52a8cc4b9119888011bf18fe267b18dccab2c06":
    raise RuntimeError("pinned_classification_source_mismatch")
classification = importlib.import_module("hymem.extraction.grounding_classification_v1")
if (Path(getattr(classification, "__file__", "")).resolve() != _contract_path.resolve()
        or getattr(classification, "EXTRACTION_IMPLEMENTATION_SHA256", None)
        != "sha256:cdca0550bc258934d09302246ac9f8e8bdd6c94c41c1e1db693c4d65e8163773"):
    raise RuntimeError("pinned_classification_import_mismatch")


class _BoundSession:
    """Per-session guard around the unchanged warm protocol implementation."""

    def __init__(self, raw: Any, observe: Any):
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_observe", observe)
        object.__setattr__(self, "_binding", None)
        object.__setattr__(self, "_thread_id", None)
        object.__setattr__(self, "_thread_starts", 0)
        object.__setattr__(self, "_turn_starts", 0)
        object.__setattr__(self, "output_schema_sent", False)
        object.__setattr__(self, "output_schema_acknowledged", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_raw", "_observe", "_binding", "_thread_id", "_thread_starts", "_turn_starts",
                    "output_schema_sent", "output_schema_acknowledged"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._raw, name, value)

    def bind(self, system: str, user: str, schema: dict[str, Any]) -> None:
        if self._binding is not None:
            warm.base._fail("invalid_request")
        self._binding = (system, user, copy.deepcopy(schema))
        self._thread_id = None
        self._thread_starts = 0
        self._turn_starts = 0
        self.output_schema_sent = False
        self.output_schema_acknowledged = False

    def clear_binding(self) -> None:
        self._binding = None
        self._thread_id = None
        self._thread_starts = 0
        self._turn_starts = 0

    def rpc(self, method: str, params: dict[str, Any], *, preserve_notifications: bool = False) -> dict[str, Any]:
        binding = self._binding
        if method == "thread/start":
            if (binding is None or self._thread_starts != 0 or self._turn_starts != 0
                    or type(params) is not dict or params.get("baseInstructions") != binding[0]):
                warm.base._fail("invalid_request")
            self._thread_starts = 1
            result = self._raw.rpc(method, params, preserve_notifications=preserve_notifications)
            thread = result.get("thread")
            self._thread_id = thread.get("id") if type(thread) is dict else None
            return result
        if method == "turn/start":
            if (binding is None or self._thread_starts != 1 or self._turn_starts != 0
                    or type(self._thread_id) is not str or not self._thread_id
                    or type(params) is not dict or "outputSchema" in params
                    or params.get("threadId") != self._thread_id
                    or params.get("input") != [{"type": "text", "text": binding[1]}]):
                warm.base._fail("invalid_request")
            self._turn_starts = 1
            wire = dict(params)
            wire["outputSchema"] = copy.deepcopy(binding[2])
            self.output_schema_sent = True
            self._observe(True, False)
            result = self._raw.rpc(method, wire, preserve_notifications=preserve_notifications)
            self.output_schema_acknowledged = True
            self._observe(True, True)
            return result
        return self._raw.rpc(method, params, preserve_notifications=preserve_notifications)

    def close(self) -> None:
        self.clear_binding()
        self._raw.close()


class ClassificationSubscriptionClient(warm.WarmSubscriptionClient):
    """One in-flight classification request; returns raw model text for validation."""

    def __init__(self, binary: str, budget: warm.SharedBudget, question_id: str,
                 question_limits: warm.BudgetLimits, *, session_factory: Any = warm.WarmSession,
                 max_requests: int = 16, max_age_seconds: float = 300):
        def bound_factory(binary: str, cwd: str, timeout: float = 120) -> _BoundSession:
            if self._active_binding is None:
                warm.base._fail("invalid_request")
            raw = session_factory(binary, cwd, timeout=timeout)
            self._pending_raw = raw
            try:
                wrapped = _BoundSession(raw, self._observe_schema)
                wrapped.bind(*self._active_binding)
                self._pending_raw = None
                return wrapped
            except BaseException:
                try:
                    raw.close()
                except BaseException:
                    self.budget.halt("cleanup_failure")
                    warm.concurrent._stop("cleanup_failure")
                self._pending_raw = None
                raise

        self._active_binding: tuple[str, str, dict[str, Any]] | None = None
        self._pending_raw: Any | None = None
        self._schema_sent = False
        self._schema_acknowledged = False
        super().__init__(binary, budget, question_id, question_limits,
                         session_factory=bound_factory, max_requests=max_requests,
                         max_age_seconds=max_age_seconds)

    def _close_process(self) -> None:
        # A factory-created raw session is not yet in self.session. Keep it
        # reachable across a failed close so the owning client can retry.
        if self._pending_raw is not None:
            try:
                self._pending_raw.close()
            except BaseException:
                self.budget.halt("cleanup_failure")
                warm.concurrent._stop("cleanup_failure")
            self._pending_raw = None
        super()._close_process()

    def complete(self, request: Any) -> str:
        raise ValueError("classification_only")

    def chat(self, messages: Any, **controls: Any) -> str:
        raise ValueError("classification_only")

    def _observe_schema(self, sent: bool, acknowledged: bool) -> None:
        self._schema_sent = sent
        self._schema_acknowledged = acknowledged

    def complete_grounding(self, request: Any, batch: Any) -> str:
        if not self._flight.acquire(blocking=False):
            self.budget.halt("concurrent_completion_rejected")
            warm.concurrent._stop("concurrent_completion_rejected")
        try:
            control_count = len(self.requested_controls)
            try:
                classification.validate_request(request, batch)
                schema = classification.build_output_schema(batch)
                self._schema_sent = False
                self._schema_acknowledged = False
                self._active_binding = (request.system, request.user, schema)
                if self.session is not None:
                    self.session.bind(*self._active_binding)
            except BaseException as exc:
                self._active_binding = None
                if self.session is not None:
                    self._close_process()
                if isinstance(exc, classification.GroundingContractError):
                    raise
                if isinstance(exc, Exception):
                    warm.base._fail("invalid_request")
                raise
            try:
                return super()._complete_locked(request)
            finally:
                # This metadata describes only attempted dispatch and RPC
                # acknowledgment; neither proves semantic validity.
                if len(self.requested_controls) > control_count:
                    self.requested_controls[-1]["output_schema_sent"] = self._schema_sent
                    self.requested_controls[-1]["output_schema_acknowledged"] = self._schema_acknowledged
                if self.session is not None:
                    self.session.clear_binding()
                self._active_binding = None
        finally:
            self._flight.release()
