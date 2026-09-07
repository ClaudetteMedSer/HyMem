"""Explicit current-contract receipts for synthetic registry fixtures.

These are fixture construction helpers, not proof of provider execution. Runner
round-trip tests separately exercise the actual checkpoint and receipt writers.
"""
from benchmarks.archive_evidence import checkpoint_attestation
from benchmarks.strictness import CHECKPOINT_VERSION


def bind_checkpoint(artifact):
    rows = artifact["per_question"]
    old = artifact["execution"].get("checkpoint", {}).get("state", {})
    old_attempts = {entry["question_id"]: entry["attempts"] for entry in old.get("entries", [])}
    snapshot = {
        "schema": CHECKPOINT_VERSION, "status": "complete",
        "run_id": artifact["manifest"]["run_id"],
        "manifest": artifact["manifest"],
        "scored": artifact["manifest"]["scored_run"],
        "verdict_key": "result_valid" if artifact["benchmark"] == "BEAM" else "correct",
        "expected_ids": [row["question_id"] for row in rows],
        "execution_segments": artifact["execution"]["segments"],
        "counts": artifact["execution"]["counts"],
        "failure_ids": [row["question_id"] for row in rows if row.get("benchmark_failure")],
        "entries": {row["question_id"]: {
            "status": "failed" if row.get("benchmark_failure") else "completed",
            "attempts": old_attempts.get(row["question_id"], 1),
        } for row in rows if row.get("benchmark_failure") != "missing_prediction"},
    }
    artifact["execution"]["checkpoint"] = checkpoint_attestation(snapshot, rows)


def skipped_indexing(scope_id, *, reason="simulation", max_cycles=100, timeout_s=3600.0):
    from benchmarks.msc_adapter import MSCAdapter, STORE_BUILD_RECEIPT_VERSION, STORE_BUILD_RECEIPT_NAME
    adapter = object.__new__(MSCAdapter)
    adapter.hy = None
    adapter.indexing_runs = []
    adapter.mark_indexing_skipped(reason, max_cycles=max_cycles, timeout_s=timeout_s)
    result = adapter.indexing_provenance(scope_id=scope_id)
    result["store_build_receipt"] = {
        "version": STORE_BUILD_RECEIPT_VERSION, "file": STORE_BUILD_RECEIPT_NAME,
        "status": "not_published_non_comparable", "identity_sha256": None,
        "indexing_sha256": None, "material_state_sha256": None,
    }
    return result


def healthy_convergence(config):
    from tests.test_lme_protocol_hardening import _current_indexing_status, _current_indexing_report
    return {
        "cycles": 1, "max_cycles": config["indexing_max_cycles"],
        "timeout_s": config["indexing_timeout_s"], "elapsed_s": 0.01,
        "complete": True, "healthy": True, "failure_reason": None,
        "reports": [_current_indexing_report()],
        "final_status": _current_indexing_status(), "quarantined": {},
    }


def scoped_indexing(scope_id, args):
    if args.sim or args.no_dream:
        return skipped_indexing(scope_id, reason="simulation" if args.sim else "no_dream",
                                max_cycles=args.indexing_max_cycles, timeout_s=args.indexing_timeout_s)
    from copy import deepcopy
    from tests.test_msc_locomo_convergence import _healthy_indexing
    result = deepcopy(_healthy_indexing(scope_id))
    result["settings"].update(max_cycles_per_convergence=args.indexing_max_cycles,
                              timeout_s_per_convergence=args.indexing_timeout_s)
    from benchmarks.msc_adapter import (
        STORE_BUILD_RECEIPT_VERSION, STORE_BUILD_RECEIPT_NAME, _canonical_indexing_attestation,
    )
    from benchmarks.strictness import content_hash
    result["store_build_receipt"] = {
        "version": STORE_BUILD_RECEIPT_VERSION, "file": STORE_BUILD_RECEIPT_NAME,
        "status": "published", "identity_sha256": content_hash({"fixture_source": scope_id}),
        "indexing_sha256": content_hash(_canonical_indexing_attestation(result, item={"id": scope_id.split(":", 1)[1]})),
        "material_state_sha256": content_hash({"fixture_material": scope_id}),
    }
    return result
