"""Finite, source-bound first application fault observation for the Luna probe.

This module stores classifications only. It never stores an exception, traceback,
frame, message, request, response, source identifier, or arbitrary string.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sqlite3
import sys
import threading
from types import CodeType, ModuleType
from typing import Any

SCHEMA = "luna_application_fault_v1"
_CHUNK_SHA = "c644513152e2ddefbe0d0d18dce7d597d18b4ba78ce275dd04ad4b2133ee7b92"
_GATE_SHA = "bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8"
_STAGED_GATE_SHA = "e843a3112a2ed0e7900f97b19a944d0a74fa458682c88a9504379c08c07deaa8"
_LLM_SHA = "97f7bf187ff17fbe0cfd059ace79b437af0945587d3082e604fecb342cba6784"
_DREAM_RUNNER_SHA = "25387efe4ef6cf5ca6d6178f96a0c7872748cdb3758bbf68836c222027691f5a"
_CODE_SHA = {
    "benchmarks/codex_subscription_staged_v6.py": "558a6cdee2c5562ff1bd4106b3abc2dadad77a6b3fd980e69420678248af2900",
    "benchmarks/codex_subscription_timeout_v3.py": "2d59fb8c59c5e304e557b05dbd346998ed46a9c2ba14a726924dc0517d352778",
}
_PHASES = frozenset({"heartbeat_boundary", "attempt_measurement", "ordinary_completion",
                     "structured_completion", "grounding_gate", "accounting_boundary",
                     "adapter_open", "client_setup", "ingest", "dream", "result_validation",
                     "question_path", "cleanup", "unknown"})
_EXCEPTIONS = frozenset({"value_error", "type_error", "runtime_error", "key_error",
                         "attribute_error", "index_error", "assertion_error", "os_error",
                         "timeout_error", "grounding_gate_error", "llm_response_error",
                         "llm_output_truncated_error", "dream_lease_lost",
                         "sqlite_database_error", "sqlite_operational_error",
                         "sqlite_programming_error", "sqlite_integrity_error",
                         "sqlite_interface_error", "other"})
_GATE_FAMILIES = frozenset({"none", "source", "contract", "support", "verdict",
                            "correction", "other"})
_KINDS = frozenset({"call_failure", "top_level", "cleanup"})
_TOP_PHASES = frozenset({"adapter_open", "client_setup", "ingest", "dream",
                         "result_validation", "question_path", "unknown"})
_CALL_PHASES = frozenset({"heartbeat_boundary", "attempt_measurement",
                          "ordinary_completion", "structured_completion",
                          "grounding_gate", "accounting_boundary", "unknown"})
_lock = threading.RLock()
_active: set[int] = set()


class FirstApplicationFaultStop(BaseException):
    """Escapes the candidate's `except Exception` after the first captured fault."""


def _regular_pinned(path: Path, digest: str) -> bool:
    try:
        return (path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest)
    except OSError:
        return False


def _module_at(module: object, path: Path, digest: str) -> bool:
    return (type(module) is ModuleType and type(getattr(module, "__file__", None)) is str
            and Path(module.__file__).resolve() == path.resolve()
            and _regular_pinned(path, digest))


def _pinned_function_code(path: Path, name: str) -> CodeType:
    """Compare live functions with compiled pinned source without executing it."""
    module_code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    matches = tuple(item for item in module_code.co_consts
                    if type(item) is CodeType and item.co_name == name)
    if len(matches) != 1:
        raise ValueError("capture_pinned_function_invalid")
    return matches[0]


def _exception_label(exc: BaseException, gate_type: type[BaseException],
                     extra: dict[type[BaseException], str]) -> str:
    if type(exc) is gate_type:
        return "grounding_gate_error"
    if type(exc) in extra:
        return extra[type(exc)]
    for cls, label in ((ValueError, "value_error"), (TypeError, "type_error"),
                       (RuntimeError, "runtime_error"), (KeyError, "key_error"),
                       (AttributeError, "attribute_error"), (IndexError, "index_error"),
                       (AssertionError, "assertion_error"), (TimeoutError, "timeout_error"),
                       (OSError, "os_error")):
        if type(exc) is cls:
            return label
    return "other"


def _gate_family(exc: BaseException, gate_type: type[BaseException]) -> str:
    if type(exc) is not gate_type:
        return "none"
    code = getattr(exc, "code", None)
    if type(code) is not str:
        return "other"
    family = code.partition(":")[0]
    return family if family in _GATE_FAMILIES - {"none", "other"} else "other"


def _record(kind: str, phase: str, exc: BaseException,
            gate_type: type[BaseException],
            extra: dict[type[BaseException], str]) -> dict[str, str]:
    return {"kind": kind, "phase": phase,
            "exception": _exception_label(exc, gate_type, extra),
            "gate_family": _gate_family(exc, gate_type)}


def _valid_record(value: object) -> bool:
    return (type(value) is dict and set(value) == {"kind", "phase", "exception", "gate_family"}
            and type(value["kind"]) is str and value["kind"] in _KINDS
            and type(value["phase"]) is str and value["phase"] in _PHASES
            and type(value["exception"]) is str and value["exception"] in _EXCEPTIONS
            and type(value["gate_family"]) is str and value["gate_family"] in _GATE_FAMILIES
            and (value["kind"] != "cleanup" or value["phase"] == "cleanup")
            and (value["kind"] != "top_level" or value["phase"] in _TOP_PHASES)
            and (value["kind"] != "call_failure" or value["phase"] in _CALL_PHASES)
            and ((value["exception"] == "grounding_gate_error") == (value["gate_family"] != "none")))


def validate_snapshot(value: object) -> dict[str, Any] | None:
    """Return a detached, exact-schema projection; invalid input fails closed."""
    if type(value) is not dict or set(value) != {"schema", "first", "cleanup"}:
        return None
    if type(value["schema"]) is not str or value["schema"] != SCHEMA:
        return None
    first, cleanup = value["first"], value["cleanup"]
    if first is not None and (not _valid_record(first) or first["kind"] == "cleanup"):
        return None
    if cleanup is not None and (not _valid_record(cleanup) or cleanup["kind"] != "cleanup"
                                or cleanup["phase"] != "cleanup"):
        return None
    return {"schema": SCHEMA, "first": dict(first) if first is not None else None,
            "cleanup": dict(cleanup) if cleanup is not None else None}


class FirstApplicationFaultCapture:
    """Install only around one verified candidate's question path.

    Required loaded keys: ``candidate`` (Path) and ``chunk`` (module).
    ``intentional_types`` must contain exact BaseException subclasses supplied
    by the caller for known budget/transport/cancellation exits. They are never
    inferred from messages or names.
    """

    def __init__(self, loaded: dict[str, Any], budget: Any,
                 *, intentional_types: tuple[type[BaseException], ...] = (),
                 on_first: Any = None) -> None:
        if type(loaded) is not dict or type(intentional_types) is not tuple or any(
            type(cls) is not type or not issubclass(cls, BaseException) for cls in intentional_types
        ) or (on_first is not None and not callable(on_first)):
            raise ValueError("capture_contract_invalid")
        candidate = loaded.get("candidate")
        chunk = loaded.get("chunk")
        if (not isinstance(candidate, Path) or not candidate.is_absolute()
                or not _module_at(chunk, candidate / "hymem/extraction/chunk.py", _CHUNK_SHA)
                or getattr(chunk, "_failure", None) is None
                or not inspect.isfunction(chunk._failure)
                or chunk._failure.__module__ != chunk.__name__
                or chunk._failure.__code__.co_filename != chunk.__file__
                or chunk._failure.__code__.co_name != "_failure"):
            raise ValueError("capture_source_invalid")
        chunk_path = candidate / "hymem/extraction/chunk.py"
        if (chunk._failure.__code__ != _pinned_function_code(chunk_path, "_failure")
                or chunk.extract_chunk.__code__ != _pinned_function_code(chunk_path, "extract_chunk")):
            raise ValueError("capture_live_function_drift")
        gate_type = getattr(chunk, "GroundingGateError", None)
        gate_module = sys.modules.get(getattr(gate_type, "__module__", ""))
        if (type(gate_type) is not type or not issubclass(gate_type, BaseException)
                or not _module_at(gate_module, candidate / "hymem/extraction/grounding_gate.py", _GATE_SHA)
                or getattr(gate_module, "GroundingGateError", None) is not gate_type):
            raise ValueError("capture_gate_source_invalid")
        staged_module = sys.modules.get("hymem.extraction.grounding_staged_gate_v1")
        if (not _module_at(staged_module, candidate / "hymem/extraction/grounding_staged_gate_v1.py",
                           _STAGED_GATE_SHA) or getattr(staged_module, "GroundingGateError", None) is not gate_type):
            raise ValueError("capture_staged_gate_source_invalid")
        llm_module = sys.modules.get("hymem.extraction.llm")
        if not _module_at(llm_module, candidate / "hymem/extraction/llm.py", _LLM_SHA):
            raise ValueError("capture_llm_source_invalid")
        if not callable(getattr(budget, "halt", None)):
            raise ValueError("capture_budget_invalid")
        self._candidate = candidate
        self._chunk = chunk
        self._budget = budget
        self._gate_type = gate_type
        self._extra_types: dict[type[BaseException], str] = {
            llm_module.LLMResponseError: "llm_response_error",
            llm_module.LLMOutputTruncatedError: "llm_output_truncated_error",
            sqlite3.DatabaseError: "sqlite_database_error",
            sqlite3.OperationalError: "sqlite_operational_error",
            sqlite3.ProgrammingError: "sqlite_programming_error",
            sqlite3.IntegrityError: "sqlite_integrity_error",
            sqlite3.InterfaceError: "sqlite_interface_error",
        }
        self._intentional_types = intentional_types
        self._on_first = on_first
        self._original = chunk._failure
        extract = getattr(chunk, "extract_chunk", None)
        single_codes = (tuple(const for const in extract.__code__.co_consts
                              if type(const) is CodeType and const.co_name == "single_attempt")
                        if inspect.isfunction(extract) and extract.__module__ == chunk.__name__ else ())
        if len(single_codes) != 1:
            raise ValueError("capture_single_attempt_invalid")
        self._single_code = single_codes[0]
        verify_codes = tuple(const for const in extract.__code__.co_consts
                             if type(const) is CodeType and const.co_name == "verify_nonempty")
        if len(verify_codes) != 1:
            raise ValueError("capture_verify_nonempty_invalid")
        self._verify_code = verify_codes[0]
        grounding_codes = tuple(const for const in extract.__code__.co_consts
                                if type(const) is CodeType and const.co_name == "grounding_call")
        if len(grounding_codes) != 1:
            raise ValueError("capture_grounding_call_invalid")
        self._grounding_code = grounding_codes[0]
        self._installed = False
        self._first: dict[str, str] | None = None
        self._cleanup: dict[str, str] | None = None
        self._origin_phases: dict[CodeType, str] = {}
        self._origin_phases[llm_module.measure_provider_attempts.__wrapped__.__code__] = "attempt_measurement"
        self._origin_phases[staged_module.ground_triples.__code__] = "grounding_gate"
        self._origin_phases[staged_module._contract.__code__] = "grounding_gate"
        dream_module = sys.modules.get("hymem.dreaming.runner")
        if _module_at(dream_module, candidate / "hymem/dreaming/runner.py", _DREAM_RUNNER_SHA):
            self._heartbeat_code = dream_module._HeartbeatLLMClient.complete.__code__
            self._extra_types[dream_module.DreamLeaseLost] = "dream_lease_lost"
        else:
            self._heartbeat_code = None
        code = loaded.get("code")
        if isinstance(code, Path) and code.is_absolute():
            for relative, digest in _CODE_SHA.items():
                path = code / relative
                if _regular_pinned(path, digest):
                    if "staged_v6" in relative:
                        staged = loaded.get("staged")
                        if _module_at(staged, path, digest):
                            self._origin_phases[staged.StagedSubscriptionClient.complete_stage.__code__] = "structured_completion"
                            self._origin_phases[staged.StagedSubscriptionClient.complete_grounding.__code__] = "structured_completion"
                    else:
                        observer = loaded.get("observer")
                        if _module_at(observer, path, digest):
                            self._origin_phases[observer.TimeoutSubscriptionClient._complete_locked.__code__] = "ordinary_completion"
            runner = loaded.get("runner")
            runner_path = code / "tools/diagnostics/luna_lme_diagnostic_v8.py"
            if _module_at(runner, runner_path, "7f96f2ac53039805d8324055edcc0902d7210195e075300eca1b0fb961764f82"):
                self._origin_phases[runner.AccountedClient._call.__code__] = "accounting_boundary"

    def _is_intentional(self, exc: BaseException) -> bool:
        return type(exc) in self._intentional_types or type(exc) is FirstApplicationFaultStop

    def _phase(self, exc: BaseException) -> str:
        tb = exc.__traceback__
        selected = "unknown"
        for _ in range(32):
            if tb is None:
                break
            code = tb.tb_frame.f_code
            if code is self._heartbeat_code and tb.tb_lineno in (417, 423):
                phase = "heartbeat_boundary"
            else:
                phase = self._origin_phases.get(code)
            if phase is not None:
                selected = phase
            tb = tb.tb_next
        return selected if tb is None else "unknown"

    def _hook(self, reason: str, *details: str) -> Any:
        if reason != "call_failure":
            return self._original(reason, *details)
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if (not self._installed or caller is None
                or caller.f_code not in (self._single_code, self._verify_code)):
            raise ValueError("capture_call_origin_invalid")
        exc = sys.exception()
        if (caller.f_code is self._verify_code and type(exc) is self._gate_type
                and getattr(exc, "code", None) == "provider:call_failed"
                and exc.__cause__ is not None
                and any(tb.tb_frame.f_code is self._grounding_code
                        for tb in self._bounded_trace(exc.__traceback__))):
            cause = exc.__cause__
            if isinstance(cause, Exception) and not self._is_intentional(cause):
                exc = cause
        if type(exc) is not FirstApplicationFaultStop and isinstance(exc, Exception) and not self._is_intentional(exc):
            if self._first is None:
                self._first = _record("call_failure", self._phase(exc), exc,
                                      self._gate_type, self._extra_types)
                self._notify_first()
            try:
                self._budget.halt("application_fault")
            finally:
                raise FirstApplicationFaultStop() from None
        # No active application exception is evidence of a broken call seam.
        try:
            self._budget.halt("capture_origin_invalid")
        finally:
            raise FirstApplicationFaultStop() from None

    def _notify_first(self) -> None:
        if self._on_first is None:
            return
        try:
            self._on_first(self.snapshot())
        except BaseException:
            try:
                self._budget.halt("capture_checkpoint_failure")
            finally:
                raise FirstApplicationFaultStop() from None

    @staticmethod
    def _bounded_trace(tb: Any) -> tuple[Any, ...]:
        entries = []
        for _ in range(32):
            if tb is None:
                break
            entries.append(tb)
            tb = tb.tb_next
        return tuple(entries) if tb is None else ()

    def __enter__(self) -> FirstApplicationFaultCapture:
        with _lock:
            identity = id(self._chunk)
            if (self._installed or identity in _active
                    or self._chunk._failure is not self._original):
                raise ValueError("capture_install_conflict")
            _active.add(identity)
            try:
                self._chunk._failure = self._hook
                self._installed = True
            except BaseException:
                _active.remove(identity)
                raise
        return self

    def __exit__(self, _type: object, _value: object, _tb: object) -> bool:
        with _lock:
            if self._installed:
                self._chunk._failure = self._original
                self._installed = False
                _active.discard(id(self._chunk))
        return False

    def record_top_level(self, exc: BaseException, *, phase: str) -> bool:
        """Record an explicit question phase outside chunk; no phase inference."""
        if (type(phase) is not str or phase not in _TOP_PHASES
                or not isinstance(exc, Exception) or self._is_intentional(exc)):
            return False
        if self._first is not None:
            return False
        self._first = _record("top_level", phase, exc, self._gate_type, self._extra_types)
        self._notify_first()
        try:
            self._budget.halt("application_fault")
        except BaseException:
            raise FirstApplicationFaultStop() from None
        return True

    def record_cleanup(self, exc: BaseException, *, phase: str = "cleanup") -> bool:
        if (phase != "cleanup" or not isinstance(exc, Exception)
                or self._is_intentional(exc) or self._cleanup is not None):
            return False
        self._cleanup = _record("cleanup", "cleanup", exc, self._gate_type,
                                self._extra_types)
        self._budget.halt("cleanup_failure")
        return True

    def snapshot(self) -> dict[str, Any]:
        value = {"schema": SCHEMA, "first": self._first, "cleanup": self._cleanup}
        projected = validate_snapshot(value)
        if projected is None:
            raise ValueError("capture_snapshot_invalid")
        return projected
