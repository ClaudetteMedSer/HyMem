"""Inactive, source-bound transport for the two-arm claim-task diagnostic.

Only ``complete_arm`` can dispatch a turn. The pinned classification adapter
and warm protocol retain ownership of admission, accounting, and cleanup.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types
from typing import Any


def _pinned_source(path: Path, digest: str, error: str) -> bytes:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise RuntimeError(error)
    return source


_root = Path(__file__).resolve().parents[1]
_adapter_path = _root / "benchmarks/codex_subscription_classification_v3.py"
_adapter_source = _pinned_source(
    _adapter_path, "d3c955922219d99b359aa6619b821e7da1662e36ecea54d09b1267dce38d2c32",
    "pinned_classification_adapter_source_mismatch")
adapter = types.ModuleType("pinned_codex_subscription_claim_task_v1_adapter")
adapter.__file__ = str(_adapter_path)
sys.modules[adapter.__name__] = adapter
exec(compile(_adapter_source, str(_adapter_path), "exec"), adapter.__dict__)

_contract_path = _root / "tools/diagnostics/luna_claim_task_contract_v1.py"
_contract_source = _pinned_source(
    _contract_path, "6e8df574632533437d6c10db7391cc419da792f180f7794e96950bc318307f89",
    "pinned_claim_task_contract_source_mismatch")
contract = types.ModuleType("pinned_codex_subscription_claim_task_v1_contract")
contract.__file__ = str(_contract_path)
sys.modules[contract.__name__] = contract
exec(compile(_contract_source, str(_contract_path), "exec"), contract.__dict__)

# The helper is executed from the captured source above, rather than resolved
# through a possibly stale or foreign sys.modules entry. Its v3 dependency
# must be the same import object the pinned adapter authenticated.
if contract.v3 is not adapter.classification:
    raise RuntimeError("pinned_claim_task_v3_import_mismatch")

warm = adapter.warm


class ClaimTaskSubscriptionClient(adapter.ClassificationSubscriptionClient):
    """One in-flight diagnostic turn with an explicit arm and exact schema."""

    def complete_grounding(self, request: Any, batch: Any) -> str:
        raise ValueError("claim_task_only")

    def complete_arm(self, arm: str, request: Any, batch: Any) -> str:
        if not self._flight.acquire(blocking=False):
            self.budget.halt("concurrent_completion_rejected")
            warm.concurrent._stop("concurrent_completion_rejected")
        try:
            control_count = len(self.requested_controls)
            try:
                contract.validate_arm_request(arm, request, batch)
                schema = contract.build_arm_output_schema(arm, batch)
                self._schema_sent = False
                self._schema_acknowledged = False
                self._active_binding = (request.system, request.user, schema)
                if self.session is not None:
                    self.session.bind(*self._active_binding)
            except BaseException as exc:
                self._active_binding = None
                if self.session is not None:
                    self._close_process()
                if isinstance(exc, adapter.classification.GroundingContractError):
                    raise
                if isinstance(exc, Exception):
                    warm.base._fail("invalid_request")
                raise
            try:
                return super(adapter.ClassificationSubscriptionClient, self)._complete_locked(request)
            finally:
                if len(self.requested_controls) > control_count:
                    self.requested_controls[-1]["output_schema_sent"] = self._schema_sent
                    self.requested_controls[-1]["output_schema_acknowledged"] = self._schema_acknowledged
                if self.session is not None:
                    self.session.clear_binding()
                self._active_binding = None
        finally:
            self._flight.release()
