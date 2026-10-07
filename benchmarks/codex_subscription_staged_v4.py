"""Inactive, source-bound subscription transport for staged grounding v4.

Each stage consumes one warm-v7 completion. The gate owns the stage
sequence and semantic validation; this adapter binds the exact request and
output schema to that single wire turn.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
import sys
import types
from typing import Any


_CODE_ROOT = Path(__file__).resolve().parents[1]
_ROOT = (_CODE_ROOT.parent / "candidate"
         if _CODE_ROOT.name == "code" else _CODE_ROOT)
_PINS = {
    "hymem.extraction.grounding_v2": (
        "hymem/extraction/grounding_v2.py",
        "377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec"),
    "hymem.extraction.grounding": (
        "hymem/extraction/grounding.py",
        "dd1a49b56abf569b4476a0b735e67e72a86723bf88e3a4739998aceeabe19c18"),
    "hymem.extraction.grounding_gate": (
        "hymem/extraction/grounding_gate.py",
        "bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8"),
    "hymem.extraction.grounding_classification_v4": (
        "hymem/extraction/grounding_classification_v4.py",
        "37ab836c45cb306d5e67d17066aed107578a812d37ffc4f06fd47ecf5fd667d2"),
    "hymem.extraction.grounding_staged_v1": (
        "hymem/extraction/grounding_staged_v1.py",
        "4862e4aedba5be756ea877d65a91b515a142b2fb46e2efb6f91c800e5096b3c9"),
    "hymem.extraction.grounding_staged_gate_v1": (
        "hymem/extraction/grounding_staged_gate_v1.py",
        "e843a3112a2ed0e7900f97b19a944d0a74fa458682c88a9504379c08c07deaa8"),
}
_IDENTITIES = {
    "hymem.extraction.grounding_v2": "sha256:b1f1d191579166b6ed118a116aee353c8cbfe00cc528bf6694e87744b6814927",
    "hymem.extraction.grounding": "sha256:3e3e72439221a54059fcd0b3c339132e570a947a831407b7dfa3471b552b5bfa",
    "hymem.extraction.grounding_gate": "sha256:836bc87e80068e2b6a9ce129b96d21d438472824a1bb2e73d773d415f3a6107c",
    "hymem.extraction.grounding_classification_v4": "sha256:ac7307f810d480d3c3037cfe1847b270a5ed1d0af9d83b4adf15340c3ff40fda",
    "hymem.extraction.grounding_staged_v1": "sha256:1d31e61b120ac871826a251c20e473eb36350fafb5c555c3ca47f4031c38153c",
    "hymem.extraction.grounding_staged_gate_v1": "sha256:5253f0994f42d510d07413bd868c7ddd00b6e72e773aad1c78120e0b65288a0d",
}


def _pinned_import(name: str) -> types.ModuleType:
    relative, digest = _PINS[name]
    path = _ROOT / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("pinned_contract_source_mismatch")
    module = importlib.import_module(name)
    if (sys.modules.get(name) is not module
            or Path(getattr(module, "__file__", "")).resolve() != path.resolve()
            or getattr(module, "EXTRACTION_IMPLEMENTATION_SHA256", None) != _IDENTITIES[name]):
        raise RuntimeError("pinned_contract_import_mismatch")
    return module


v2 = _pinned_import("hymem.extraction.grounding_v2")
grounding = _pinned_import("hymem.extraction.grounding")
source_gate = _pinned_import("hymem.extraction.grounding_gate")
classification = _pinned_import("hymem.extraction.grounding_classification_v4")
staged = _pinned_import("hymem.extraction.grounding_staged_v1")
gate = _pinned_import("hymem.extraction.grounding_staged_gate_v1")
if (staged.v4 is not classification or gate._staged is not staged
        or gate.ClassificationBatch is not classification.ClassificationBatch
        or gate._source_gate is not source_gate
        or classification._build_v2 is not v2.build_grounding_request
        or classification._parse_v2 is not v2.parse_grounding_response
        or classification.GroundingSource is not v2.GroundingSource
        or classification.GroundingContext is not v2.GroundingContext
        or classification.GroundingContractError is not v2.GroundingContractError
        or staged._build_v2 is not v2.build_grounding_request
        or staged._parse_v2 is not v2.parse_grounding_response
        or staged.GroundingContractError is not v2.GroundingContractError
        or gate.GroundingSource is not v2.GroundingSource
        or gate.GroundingContext is not v2.GroundingContext
        or gate.GroundingContractError is not v2.GroundingContractError
        or source_gate.GroundingSource is not grounding.GroundingSource
        or gate.GroundingGateError is not source_gate.GroundingGateError
        or not gate.grounding_gate_support_integrity()):
    raise RuntimeError("pinned_contract_dependency_mismatch")

_warm_path = Path(__file__).resolve().with_name("codex_subscription_warm_v7.py")
_warm_source = _warm_path.read_bytes()
if hashlib.sha256(_warm_source).hexdigest() != "94234eca1daeb542a8d8f92a5b7178ba8b91060f33415f76343bd13e6d5953d9":
    raise RuntimeError("pinned_warm_v7_source_mismatch")
warm = types.ModuleType("pinned_codex_subscription_staged_v4_warm_v7")
warm.__file__ = str(_warm_path)
sys.modules[warm.__name__] = warm
exec(compile(_warm_source, str(_warm_path), "exec"), warm.__dict__)


class _BoundSession:
    """Bind one trusted system, user and schema to the next fresh thread turn."""

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


class StagedSubscriptionClient(warm.WarmSubscriptionClient):
    """Gate callback adapter: complete_stage(request, batch, stage, recheck)."""

    def __init__(self, binary: str, budget: warm.SharedBudget, question_id: str,
                 question_limits: warm.BudgetLimits, *, session_factory: Any = warm.WarmSession,
                 max_requests: int = 16, max_age_seconds: float = 300,
                 private_failure_sink: warm.PrivateFailureSink | None = None):
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
                         max_age_seconds=max_age_seconds,
                         private_failure_sink=private_failure_sink)

    def _close_process(self) -> None:
        if self._pending_raw is not None:
            try:
                self._pending_raw.close()
            except BaseException:
                self.budget.halt("cleanup_failure")
                warm.concurrent._stop("cleanup_failure")
            self._pending_raw = None
        super()._close_process()

    def complete(self, request: Any) -> str:
        raise ValueError("staged_only")

    def chat(self, messages: Any, **controls: Any) -> str:
        raise ValueError("staged_only")

    def complete_grounding(self, request: Any, batch: Any) -> str:
        raise ValueError("staged_only")

    def _observe_schema(self, sent: bool, acknowledged: bool) -> None:
        self._schema_sent = sent
        self._schema_acknowledged = acknowledged

    def complete_stage(self, request: Any, batch: Any, stage: str, recheck: bool) -> str:
        if not self._flight.acquire(blocking=False):
            self.budget.halt("concurrent_completion_rejected")
            warm.concurrent._stop("concurrent_completion_rejected")
        try:
            control_count = len(self.requested_controls)
            try:
                if type(stage) is not str or type(recheck) is not bool:
                    raise staged.GroundingContractError("stage:invalid")
                if stage == "original":
                    staged.validate_original_request(request, batch)
                    schema = staged.build_original_output_schema(batch)
                elif stage == "alternatives" and not recheck:
                    staged.validate_alternatives_request(request, batch)
                    schema = staged.build_alternatives_output_schema(batch)
                else:
                    raise staged.GroundingContractError("stage:invalid")
                self._schema_sent = False
                self._schema_acknowledged = False
                self._active_binding = (request.system, request.user, schema)
                if self.session is not None:
                    self.session.bind(*self._active_binding)
            except BaseException as exc:
                self._active_binding = None
                if self.session is not None:
                    self._close_process()
                if isinstance(exc, staged.GroundingContractError):
                    raise
                if isinstance(exc, Exception):
                    warm.base._fail("invalid_request")
                raise
            try:
                return super()._complete_locked(request)
            finally:
                if len(self.requested_controls) > control_count:
                    self.requested_controls[-1]["output_schema_sent"] = self._schema_sent
                    self.requested_controls[-1]["output_schema_acknowledged"] = self._schema_acknowledged
                if self.session is not None:
                    self.session.clear_binding()
                self._active_binding = None
        finally:
            self._flight.release()
