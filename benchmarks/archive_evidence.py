"""Portable benchmark evidence checks, not signatures or filesystem attestations.

The checkpoint receipt exposes a small finalized-ledger projection. Its digest
can be recomputed from the archive without pretending an opaque checksum proves
the existence, authenticity, or contents of an unavailable checkpoint file.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

try:
    from .strictness import BenchmarkIntegrityError, CHECKPOINT_VERSION, content_hash
except ImportError:
    from strictness import BenchmarkIntegrityError, CHECKPOINT_VERSION, content_hash

CHECKPOINT_ATTESTATION_VERSION = "hymem-benchmark-checkpoint-attestation-v1"
_MAX_COUNT = 2_147_483_647
# This is converge_indexing's raw wire contract, not LME's normalized failure
# vocabulary (which drops bounded exception types and maps some raw reasons).
_RAW_CONVERGENCE_FAILURE_CODES = frozenset({
    "timeout_before_cycle", "timeout_during_cycle", "timeout_after_cycle",
    "malformed_status_shape", "phase1_producer_unavailable",
    "malformed_pending_backlog", "malformed_quarantine_state",
    "malformed_terminal_loss_state", "malformed_coverage_integrity_state",
    "malformed_aggregation_failure_report", "malformed_cycle_failure_report",
    "coverage_integrity_failure", "malformed_durable_state",
    "terminal_extraction_source_loss", "quarantined_extraction",
    "max_cycles_exhausted",
})


def _fail(message):
    raise BenchmarkIntegrityError("archive evidence: " + message)


def checkpoint_attestation(snapshot, rows):
    """Project actual validated finalized checkpoint state without source payloads."""
    row_by_id = {row["question_id"]: row for row in rows}
    state = {
        "checkpoint_schema": snapshot["schema"],
        "status": snapshot["status"],
        "run_id": snapshot["run_id"],
        "scored": snapshot["scored"],
        "verdict_key": snapshot["verdict_key"],
        "expected_ids": snapshot["expected_ids"],
        "manifest_sha256": content_hash(snapshot["manifest"]),
        "segments_sha256": content_hash(snapshot["execution_segments"]),
        "rows_sha256": content_hash(rows),
        "counts": snapshot["counts"],
        "failure_ids": snapshot["failure_ids"],
        "entries": [{
            "question_id": item_id,
            "status": snapshot["entries"][item_id]["status"],
            "attempts": snapshot["entries"][item_id]["attempts"],
            "row_sha256": content_hash(row_by_id[item_id]),
        } for item_id in snapshot["expected_ids"] if item_id in snapshot["entries"]],
    }
    return {"schema": CHECKPOINT_ATTESTATION_VERSION,
            "state": state, "state_sha256": content_hash(state)}


def validate_checkpoint_attestation(artifact):
    execution = artifact.get("execution")
    if not isinstance(execution, Mapping):
        _fail("checkpoint execution is absent")
    receipt = execution.get("checkpoint")
    if not isinstance(receipt, dict) or set(receipt) != {"schema", "state", "state_sha256"}:
        _fail("checkpoint attestation is missing or obsolete")
    state = receipt["state"]
    if receipt["schema"] != CHECKPOINT_ATTESTATION_VERSION or not isinstance(state, dict):
        _fail("checkpoint attestation schema is invalid")
    if set(state) != {
        "checkpoint_schema", "status", "run_id", "scored", "verdict_key",
        "expected_ids", "manifest_sha256", "segments_sha256", "rows_sha256",
        "counts", "failure_ids", "entries",
    } or receipt["state_sha256"] != content_hash(state):
        _fail("checkpoint digest or projection is invalid")
    rows = artifact.get("per_question")
    manifest = artifact.get("manifest")
    segments = execution.get("segments")
    if not isinstance(rows, list) or not isinstance(manifest, Mapping) or not isinstance(segments, list):
        _fail("checkpoint-bound archive fields are malformed")
    ids = [row.get("question_id") for row in rows if isinstance(row, Mapping)]
    if (len(ids) != len(rows) or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)):
        _fail("checkpoint row identifiers are malformed")
    expected_verdict = "result_valid" if manifest.get("benchmark") == "BEAM" else "correct"
    if (
        state["checkpoint_schema"] != CHECKPOINT_VERSION
        or state["status"] != "complete"
        or state["run_id"] != manifest.get("run_id")
        or type(state["scored"]) is not bool
        or state["scored"] is not manifest.get("scored_run")
        or state["verdict_key"] != expected_verdict
        or state["expected_ids"] != ids
        or manifest.get("expected_ids_hash") != content_hash(ids)
        or state["manifest_sha256"] != content_hash(manifest)
        or state["segments_sha256"] != content_hash(segments)
        or state["rows_sha256"] != content_hash(rows)
        or state["counts"] != execution.get("counts")
    ):
        _fail("checkpoint projection differs from archived execution")
    if not isinstance(state["counts"], dict) or any(type(value) is not int or not 0 <= value <= _MAX_COUNT for value in state["counts"].values()):
        _fail("checkpoint counts are malformed")
    id_set = set(ids)
    entries = state["entries"]
    if not isinstance(entries, list) or len(entries) > len(rows):
        _fail("checkpoint entries are malformed")
    indexed = {}
    attempts = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"question_id", "status", "attempts", "row_sha256"}:
            _fail("checkpoint entry is malformed")
        item_id = entry["question_id"]
        if not isinstance(item_id, str) or item_id not in id_set or item_id in indexed:
            _fail("checkpoint entry ownership is invalid")
        if type(entry["attempts"]) is not int or not 1 <= entry["attempts"] <= _MAX_COUNT:
            _fail("checkpoint entry attempts are invalid")
        indexed[item_id] = entry
        attempts += entry["attempts"]
        if attempts > _MAX_COUNT:
            _fail("checkpoint total attempts are too large")
    if [entry["question_id"] for entry in entries] != [item for item in ids if item in indexed]:
        _fail("checkpoint entry order is invalid")
    completed = missing = 0
    failure_ids = []
    for row in rows:
        item_id = row["question_id"]
        entry = indexed.get(item_id)
        failure = row.get("benchmark_failure")
        if failure is not None and not isinstance(failure, str):
            _fail("checkpoint row failure is malformed")
        if entry is None:
            if failure != "missing_prediction":
                _fail("checkpoint missing entry is not a missing prediction")
            missing += 1
            failure_ids.append(item_id)
            continue
        failed = bool(failure)
        if entry["status"] != ("failed" if failed else "completed"):
            _fail("checkpoint row/entry outcome differs")
        if entry["row_sha256"] != content_hash(row):
            _fail("checkpoint entry row digest differs")
        if failed:
            failure_ids.append(item_id)
        else:
            completed += 1
    segment_attempts = 0
    for segment in segments:
        if (not isinstance(segment, Mapping)
                or type(segment.get("attempted_attempts")) is not int
                or segment["attempted_attempts"] < 0):
            _fail("checkpoint segment attempts are malformed")
        segment_attempts += segment["attempted_attempts"]
    expected_counts = {
        "expected": len(rows), "attempted": len(entries), "unique_attempted": len(entries),
        "total_attempts": max(attempts, segment_attempts), "completed": completed,
        "failed": len(failure_ids), "missing": missing,
    }
    if (state["counts"] != expected_counts
            or any(type(value) is not int for value in state["counts"].values())
            or state["failure_ids"] != failure_ids):
        _fail("checkpoint finalized counts do not reconcile")


def validate_scoped_indexing(summary, *, scope_id, config, failed=False):
    """Validate the receipt actually returned by ``prepare_indexing``.

    The store pointer is not the store receipt itself: only a freshly published
    pointer's indexing digest can be rebound to evidence in this archive. A
    reused pointer refers to the original build, not the new validation wave.
    Neither form authenticates an unavailable filesystem object.
    """
    if not isinstance(summary, dict):
        _fail("indexing receipt is not an object")
    if failed:
        _validate_scoped_failure(summary, scope_id=scope_id, config=config)
        return False
    base = {key: value for key, value in summary.items() if key != "store_build_receipt"}
    comparable = _validate_scoped_indexing_content(base, scope_id=scope_id, config=config)
    _validate_store_pointer(summary.get("store_build_receipt"), base, comparable=comparable)
    return comparable


def _validate_store_pointer(pointer, indexing, *, comparable):
    try:
        from .msc_adapter import (STORE_BUILD_RECEIPT_VERSION, STORE_BUILD_RECEIPT_NAME,
                                  _canonical_indexing_attestation)
    except ImportError:
        from msc_adapter import (STORE_BUILD_RECEIPT_VERSION, STORE_BUILD_RECEIPT_NAME,
                                 _canonical_indexing_attestation)
    digests = {"identity_sha256", "indexing_sha256", "material_state_sha256"}
    if (not isinstance(pointer, dict)
            or set(pointer) != digests | {"version", "status", "file"}
            or pointer["version"] != STORE_BUILD_RECEIPT_VERSION
            or pointer["file"] != STORE_BUILD_RECEIPT_NAME):
        _fail("store-build receipt pointer is missing or malformed")
    if not comparable:
        if (pointer["status"] != "not_published_non_comparable"
                or any(pointer[key] is not None for key in digests)):
            _fail("skipped indexing claims a published store")
        return
    reused = any(run["trigger"] == "reused_store_validation" for run in indexing["runs"])
    if pointer["status"] != ("validated" if reused else "published"):
        _fail("store-build receipt pointer contradicts indexing lifecycle")
    for key in digests:
        _require_digest(pointer[key], key)
    if not reused:
        attestation = _canonical_indexing_attestation(indexing, item={
            "id": indexing["scope_id"].split(":", 1)[1],
        })
        if pointer["indexing_sha256"] != content_hash(attestation):
            _fail("published store pointer differs from archived indexing")


def _require_digest(value, label, *, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        _fail(label + " is not a digest")


def _validate_scoped_indexing_content(summary, *, scope_id, config):
    """Validate the base receipt also embedded in publication-failure evidence."""
    try:
        from .msc_adapter import (INDEXING_PROVENANCE_VERSION, _validate_indexing_provenance,
                                  _validate_indexing_usage)
    except ImportError:
        from msc_adapter import (INDEXING_PROVENANCE_VERSION, _validate_indexing_provenance,
                                 _validate_indexing_usage)
    if not isinstance(summary, dict):
        _fail("indexing receipt is not an object")
    simulation = config.get("sim") is True
    skip_reason = "simulation" if simulation else "no_dream" if config.get("no_dream") is True else None
    if skip_reason is None:
        _validate_indexing_provenance(summary, item={"id": scope_id.split(":", 1)[1]})
        if summary.get("scope_id") != scope_id:
            _fail("indexing source scope differs")
        attempts = summary["dream_report_totals"]["chunk_extraction_provider_attempts"]
        usage = summary["pipeline_usage"]
        if usage.get("request_attempts_available") is True and usage["request_attempts"] < attempts:
            _fail("indexing extraction attempts exceed pipeline usage")
    else:
        if set(summary) != {"protocol", "scope_id", "mode", "skip_reason", "comparable", "complete", "healthy", "convergence_count", "cycles", "settings", "settings_applied", "observed_status", "pipeline_usage"}:
            _fail("skipped indexing receipt is incomplete")
        if (summary["protocol"] != INDEXING_PROVENANCE_VERSION
                or summary["scope_id"] != scope_id
                or summary["mode"] != "skipped_non_comparable"
                or summary["skip_reason"] != skip_reason
                or any(summary[k] is not False for k in ("comparable", "complete", "healthy", "settings_applied"))
                or any(type(summary[k]) is not int or summary[k] != 0 for k in ("convergence_count", "cycles"))):
            _fail("skipped indexing contradicts run mode")
        _validate_indexing_usage(summary["pipeline_usage"])
        for field in ("calls", "request_attempts", "successful_responses"):
            if summary["pipeline_usage"].get(field) != 0 or summary["pipeline_usage"].get(field + "_available") is not True:
                _fail("skipped indexing claims pipeline work")
        observed = summary["observed_status"]
        # Skipping makes no completion or producer-authority assertion. The
        # diagnostic snapshot can truthfully describe an unbuilt store.
        if observed is not None and not isinstance(observed, dict):
            _fail("skipped indexing observed status is not an object")
    settings = summary.get("settings")
    if (not isinstance(settings, dict)
            or type(settings.get("max_cycles_per_convergence")) is not int
            or isinstance(settings.get("timeout_s_per_convergence"), bool)
            or settings.get("require_healthy") is not True):
        _fail("indexing settings are malformed")
    if summary.get("settings") != {
        "max_cycles_per_convergence": config.get("indexing_max_cycles"),
        "timeout_s_per_convergence": config.get("indexing_timeout_s"),
        "require_healthy": True,
    }:
        _fail("indexing settings differ from manifested policy")
    return skip_reason is None


def _validate_scoped_failure(summary, *, scope_id, config):
    """Admit bounded failure envelopes without turning them into success."""
    try:
        from .msc_adapter import INDEXING_PROVENANCE_VERSION, _validate_indexing_usage
    except ImportError:
        from msc_adapter import INDEXING_PROVENANCE_VERSION, _validate_indexing_usage
    if summary.get("status") != "failed_before_scoring":
        validate_convergence_summary(summary, config=config, allow_failure=True)
        if summary["healthy"] is not False:
            _fail("indexing failure carries healthy success")
        return
    if "scope_id" not in summary:
        allowed = {"status", "failure_reason", "complete", "healthy", "comparable"}
        if (set(summary) - allowed
                or any(summary.get(k, False) is not False for k in ("complete", "healthy", "comparable"))):
            _fail("pre-scoring failure contradicts completion")
        reason = summary.get("failure_reason")
        if (not isinstance(reason, str) or (reason != "unspecified_failure"
                and re.fullmatch(r"(?:pre_scoring|conversation)_failure:[A-Za-z_][A-Za-z0-9_.]{0,127}", reason) is None)):
            _fail("pre-scoring failure reason is malformed")
        return
    if (summary["scope_id"] != scope_id
            or summary.get("protocol") != INDEXING_PROVENANCE_VERSION):
        _fail("indexing failure scope or protocol differs")
    if "cycles" in summary:
        _validate_indexing_usage(summary.get("pipeline_usage"))
        raw = {key: value for key, value in summary.items()
               if key not in {"scope_id", "status", "pipeline_usage"}}
        validate_convergence_summary(raw, config=config, allow_failure=True)
        if raw["healthy"] is not False:
            _fail("indexing failure carries healthy success")
        return
    _validate_store_failure(summary, scope_id=scope_id, config=config)


def _validate_store_failure(summary, *, scope_id, config):
    """The closed, source-free failure union emitted around store attestation."""
    try:
        from .msc_adapter import (STORE_BUILD_RECEIPT_VERSION, STORE_BUILD_RECEIPT_NAME,
                                  _validate_indexing_usage)
    except ImportError:
        from msc_adapter import (STORE_BUILD_RECEIPT_VERSION, STORE_BUILD_RECEIPT_NAME,
                                 _validate_indexing_usage)
    reason = summary.get("failure_reason")
    envelope = {"protocol", "scope_id", "status", "failure_reason", "remediation"}
    if reason == "store_build_receipt_publication_failed":
        if (set(summary) != envelope | {"indexing", "exception_type"}
                or summary["remediation"] != "rerun this store with --fresh"
                or not isinstance(summary["exception_type"], str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", summary["exception_type"]) is None):
            _fail("store publication failure is malformed")
        if not _validate_scoped_indexing_content(summary["indexing"], scope_id=scope_id, config=config):
            _fail("store publication failure has no converged indexing")
        return
    envelope.add("pipeline_usage")
    _validate_indexing_usage(summary.get("pipeline_usage"))
    if reason in ("skipped_indexing_reused_store", "store_build_receipt_changed_during_reuse"):
        expected_remediation = ("use --fresh or omit --db-dir" if reason == "skipped_indexing_reused_store"
                                else "rebuild this conversation store with --fresh")
        if set(summary) != envelope or summary["remediation"] != expected_remediation:
            _fail("store reuse failure is malformed")
        if reason == "skipped_indexing_reused_store" and not (config.get("sim") is True or config.get("no_dream") is True):
            _fail("skipped store reuse failure contradicts mode")
        return
    envelope |= {"receipt_file", "expected_identity_sha256"}
    if (not envelope.issubset(summary)
            or summary["receipt_file"] != STORE_BUILD_RECEIPT_NAME
            or summary["remediation"] != "rebuild this conversation store with --fresh"):
        _fail("store attestation failure is malformed")
    _require_digest(summary["expected_identity_sha256"], "expected store identity",
                    nullable=reason == "store_build_identity_unavailable")
    details = set(summary) - envelope
    empty_reasons = {
        "store_build_identity_unavailable", "missing_store_build_receipt",
        "oversized_store_build_receipt", "malformed_store_build_receipt",
        "store_embedding_attestation_failed",
    }
    if isinstance(reason, str) and reason in empty_reasons:
        expected = set()
        if reason == "store_build_identity_unavailable" and summary["expected_identity_sha256"] is not None:
            _fail("unavailable store identity carries a digest")
    elif reason == "incompatible_store_build_receipt_version":
        expected = {"expected_receipt_version", "recorded_receipt_version"}
        recorded = summary.get("recorded_receipt_version")
        if (summary.get("expected_receipt_version") != STORE_BUILD_RECEIPT_VERSION
                or recorded == STORE_BUILD_RECEIPT_VERSION
                or (recorded is not None and (not isinstance(recorded, str)
                    or re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", recorded) is None))):
            _fail("store receipt version failure is inconsistent")
    elif reason == "corrupt_store_build_receipt":
        expected = next(({"recorded_" + name, "actual_" + name}
                         for name in ("indexing_sha256", "embedding_state_sha256", "identity_sha256")
                         if details == {"recorded_" + name, "actual_" + name}), None)
        if expected is None:
            _fail("corrupt store receipt has no bounded digest pair")
    elif reason == "store_build_identity_mismatch":
        expected = {"actual_identity_sha256", "mismatch_fields", "mismatch_fields_truncated"}
    elif reason == "store_material_state_mismatch":
        expected = {"recorded_material_state_sha256", "current_material_state_sha256",
                    "mismatch_tables", "mismatch_tables_truncated"}
    elif reason == "store_embedding_state_mismatch":
        expected = {"recorded_embedding_state_sha256", "current_embedding_state_sha256"}
    elif reason == "store_material_attestation_failed":
        expected = ({"attestation_failure_reason", "attestation_tables", "attestation_tables_truncated"}
                    if details else set())
    else:
        _fail("unknown store attestation failure")
    if details != expected:
        _fail("store attestation failure details are incomplete")
    for key in details:
        value = summary[key]
        if key.endswith("_sha256"):
            _require_digest(value, key, nullable=reason == "corrupt_store_build_receipt" and key.startswith("recorded_"))
        elif key.endswith("_truncated"):
            if type(value) is not bool:
                _fail("store failure truncation flag is malformed")
        elif key in {"mismatch_fields", "mismatch_tables", "attestation_tables"}:
            pattern = r"identity(?:\.[A-Za-z0-9_?.-]{1,96})*" if key == "mismatch_fields" else r"(?:[A-Za-z0-9_?.-]{1,96}|<state>|<tables>)"
            if (not isinstance(value, list) or len(value) > 50
                    or any(not isinstance(item, str) or len(item) > 4096 or re.fullmatch(pattern, item) is None for item in value)):
                _fail("store failure structural identifiers are malformed")
        elif key == "attestation_failure_reason":
            if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,127}", value) is None:
                _fail("store attestation failure reason is malformed")


def validate_scoped_row(row, *, scope_id, receipts):
    """Bind optional duplicated row evidence to a receipt actually archived."""
    if "indexing_scope_id" in row and row["indexing_scope_id"] != scope_id:
        _fail("row indexing scope differs from its source")
    if "indexing_ref" in row and row["indexing_ref"] != scope_id:
        _fail("row indexing reference differs from its source")
    candidates = receipts.get(scope_id, [])
    if "indexing" in row:
        if not isinstance(row["indexing"], Mapping) or row["indexing"] not in candidates:
            _fail("row indexing receipt is not owned by its source execution")
        candidates = [row["indexing"]]
    flags = {key: row[key] for key in ("indexing_complete", "indexing_healthy", "indexing_comparable") if key in row}
    if flags and (any(type(value) is not bool for value in flags.values()) or not any(
        all(receipt.get(key.removeprefix("indexing_")) is value for key, value in flags.items())
        for receipt in candidates
    )):
        _fail("row indexing health contradicts its execution receipt")


def validate_scoped_pipeline_usage(summaries, usage):
    """Receipts precede retrieval; their known cumulative calls are a lower bound."""
    for field in ("calls", "request_attempts", "successful_responses"):
        known = sum(summary["pipeline_usage"][field] for summary in summaries
                    if summary["pipeline_usage"].get(field + "_available") is True)
        if usage.get(field + "_available") is True and usage[field] < known:
            _fail("scope indexing calls exceed execution pipeline usage")


def validate_convergence_summary(summary, *, config, allow_failure=False):
    """Validate the raw convergence receipt emitted by BEAM and failure paths."""
    try:
        from . import lme_protocol as protocol
    except ImportError:
        import lme_protocol as protocol
    _canonical_final_indexing_status = protocol._canonical_final_indexing_status
    if not isinstance(summary, Mapping):
        _fail("convergence summary is absent")
    core = {"cycles", "max_cycles", "timeout_s", "elapsed_s", "complete", "healthy", "failure_reason", "reports", "final_status", "quarantined"}
    if not core.issubset(summary) or set(summary) - core - {"protocol", "trigger", "initial_status", "cleanup_errors"}:
        _fail("convergence summary fields are malformed")
    cycles = summary["cycles"]
    maximum = summary["max_cycles"]
    if type(cycles) is not int or type(maximum) is not int or not 0 <= cycles <= maximum or maximum <= 0:
        _fail("convergence cycle count is invalid")
    if maximum != config.get("indexing_max_cycles") or summary["timeout_s"] != config.get("indexing_timeout_s"):
        _fail("convergence bounds differ from policy")
    for name in ("elapsed_s", "timeout_s"):
        value = summary[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            _fail("convergence timing is malformed")
    if summary["timeout_s"] <= 0 or type(summary["complete"]) is not bool or type(summary["healthy"]) is not bool:
        _fail("convergence outcome is malformed")
    reports = summary["reports"]
    if not isinstance(reports, list) or len(reports) != cycles:
        _fail("convergence report count differs")
    if summary["complete"] and summary["healthy"]:
        normalized = [protocol._canonical_indexing_report(report, require_current_failures=True) for report in reports]
        if (not cycles or summary["failure_reason"] is not None
                or summary["elapsed_s"] > summary["timeout_s"]
                or summary["quarantined"] != {} or summary.get("cleanup_errors", []) != []):
            _fail("successful convergence contradicts failure evidence")
        final = _canonical_final_indexing_status(summary["final_status"])
        if (any(final["pending"].values()) or any(final["malformed"].values())
                or any(final["quarantined"].values()) or final["terminal_loss"]["chunks"]
                or final["coverage_integrity"]["failures"] or final["in_progress"]
                or any(normalized[-1][key] for key in (*protocol._INDEXING_REPORT_BOOLEAN_FIELDS, *protocol._INDEXING_CYCLE_FAILURE_FIELDS))):
            _fail("successful convergence is not healthy")
        return True
    reason = summary["failure_reason"]
    known_failure = isinstance(reason, str) and (reason in _RAW_CONVERGENCE_FAILURE_CODES
        or re.fullmatch(r"cycle_exception:[A-Za-z_][A-Za-z0-9_.]{0,127}", reason) is not None)
    if not allow_failure or summary["healthy"] is not False or not known_failure:
        _fail("convergence did not finish healthy")
    if summary["complete"] and reason not in protocol._MECHANICALLY_COMPLETE_FAILURE_CODES:
        _fail("failed convergence contradicts mechanical completion")
    # Failed receipts can truthfully describe a malformed provider/status
    # payload. They never establish healthy coverage for a completed row.
    if not isinstance(summary["final_status"], Mapping) or not isinstance(summary["quarantined"], Mapping):
        _fail("failed convergence diagnostics are malformed")
    if any(not isinstance(report, Mapping) for report in reports):
        _fail("failed convergence report framing is malformed")
    if reason == "max_cycles_exhausted" and cycles != maximum:
        _fail("cycle-limit failure precedes its limit")
    if reason.startswith("timeout_") and summary["elapsed_s"] < summary["timeout_s"]:
        _fail("timeout failure precedes its deadline")
    return False
