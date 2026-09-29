#!/usr/bin/env python3
"""Counts-only postflight for the closed, private schema-64 dream clone."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import stat
import sys
import traceback

sys.path.insert(0, "/candidate")

AUDIT_FILE = Path("/diag/claim_conflict_store_audit.py")
AUDIT_SHA256 = "e2efe365c5aedbbe88d86d521dd37b821759afc5fb21c21a6662e6a8ce567f42"
TARGET_CHUNK = "chk_c14182771bd8e2a7e583f7d693cee29a5fb159fe"
TARGET_GENERATION = (
    "hymem-phase1-generation-v1:"
    "6075085e12e32e1b790e49b99b8c3bb50718b18be0f28762d1582c58ee8e35eb"
)
PHASE1_SHA256 = "bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac"
WORK = Path("/work")


def load_audit(path: Path = AUDIT_FILE):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("audit_helper_invalid")
    if hashlib.sha256(path.read_bytes()).hexdigest() != AUDIT_SHA256:
        raise RuntimeError("audit_helper_pin_drift")
    spec = importlib.util.spec_from_file_location("private_dream_pinned_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("audit_helper_unloadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quarantine_ids(conn) -> dict[str, set[str]]:
    """Keep identifiers in memory; only cardinalities leave the worker."""
    return {
        "retry_limit": {str(row[0]) for row in conn.execute(
            "SELECT DISTINCT chunk_id FROM chunk_extraction_attempts WHERE attempts>=3")},
        "terminal_loss": {str(row[0]) for row in conn.execute(
            "SELECT chunk_id FROM chunk_extraction_terminal_losses")},
    }


def publication_counts(conn, *, has_proof_column: bool,
                       target: str = TARGET_CHUNK,
                       generation: str = TARGET_GENERATION) -> dict[str, int]:
    current = int(conn.execute(
        "SELECT COUNT(*) FROM current_phase1_publications WHERE chunk_id=?",
        (target,),
    ).fetchone()[0])
    pinned = int(conn.execute(
        "SELECT COUNT(*) FROM current_phase1_publications "
        "WHERE chunk_id=? AND phase1_generation_key=?",
        (target, generation),
    ).fetchone()[0])
    with_proof = int(conn.execute(
        "SELECT COUNT(*) FROM current_phase1_publications publication "
        "JOIN kg_claim_extraction_outcomes claim "
        "ON claim.chunk_id=publication.chunk_id "
        "AND claim.phase1_generation_key=publication.phase1_generation_key "
        "AND claim.prompt_version=publication.prompt_version "
        "WHERE publication.chunk_id=? AND publication.phase1_generation_key=? "
        "AND claim.local_replay_proof IS NOT NULL",
        (target, generation),
    ).fetchone()[0]) if has_proof_column else 0
    return {"target_current": current, "target_pinned": pinned,
            "target_pinned_with_proof": with_proof}


def validate_worker_report(report: object) -> dict:
    from hymem.dreaming.runner import (
        DREAM_REPORT_COUNT_FIELDS, DREAM_REPORT_ERROR_FIELDS,
        DREAM_REPORT_NULLABLE_COUNT_FIELDS, DREAM_REPORT_TEXT_FIELDS,
        DREAM_REPORT_BOOLEAN_GATE_FIELDS,
    )

    expected = set(DREAM_REPORT_COUNT_FIELDS + DREAM_REPORT_ERROR_FIELDS
                   + DREAM_REPORT_NULLABLE_COUNT_FIELDS
                   + DREAM_REPORT_BOOLEAN_GATE_FIELDS)
    if set(DREAM_REPORT_TEXT_FIELDS) != {"aggregation_blocking"}:
        raise RuntimeError("candidate_dream_report_partition_changed")
    if not isinstance(report, dict) or set(report) != expected:
        raise ValueError("worker_report_fields_invalid")
    for name in DREAM_REPORT_COUNT_FIELDS + DREAM_REPORT_ERROR_FIELDS:
        if type(report[name]) is not int or report[name] < 0:
            raise ValueError("worker_report_value_invalid")
    for name in DREAM_REPORT_NULLABLE_COUNT_FIELDS:
        value = report[name]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("worker_report_value_invalid")
    for name in DREAM_REPORT_BOOLEAN_GATE_FIELDS:
        if type(report[name]) is not bool:
            raise ValueError("worker_report_value_invalid")
    return report


def worker_evidence(path: Path, audit, dream, summary: dict) -> tuple[dict, bool, bool]:
    audit.checked_file(path, mode=0o600)
    worker = json.loads(path.read_bytes())
    if not isinstance(worker, dict) or (
        worker.get("status") != "completed"
        or worker.get("source_sha256") != audit.REFERENCE_SHA
        or worker.get("phase1_sha256") != PHASE1_SHA256
        or worker.get("target_chunk_id") != TARGET_CHUNK
        or worker.get("generation_key") != TARGET_GENERATION
        or worker.get("runtime_generation_verified") is not True
    ):
        raise ValueError("worker_result_identity_invalid")
    report = validate_worker_report(worker.get("report"))
    if type(worker.get("chunks_processed")) is not int or (
        worker["chunks_processed"] != report["chunks_processed"]
    ):
        raise ValueError("worker_result_count_invalid")
    row = dream.execute("SELECT * FROM dream_runs ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        raise ValueError("worker_dream_run_missing")
    persisted = set(row.keys()) & set(report)
    consistent = int(row["id"]) == summary["dream"]["latest_dream"]["id"]
    for name in persisted:
        value = report[name]
        if type(value) is bool:
            value = int(value)
        if row[name] != value:
            consistent = False
    campaign_processed = summary["campaign"].get("chunks_processed")
    if campaign_processed is not None and campaign_processed != report["chunks_processed"]:
        consistent = False
    blocking = row["aggregation_blocking"]
    if not isinstance(blocking, str):
        raise ValueError("dream_aggregation_blocking_invalid")
    return report, consistent, bool(blocking)


def assess(summary: dict, baseline_version: int, dream_version: int,
           before_publications: dict[str, int], after_publications: dict[str, int],
           before_quarantines: dict[str, set[str]],
           after_quarantines: dict[str, set[str]],
           degraded_counters: tuple[str, ...], worker_report: dict,
           stored_counter_consistent: bool,
           aggregation_blocking_present: bool) -> dict:
    from hymem.dreaming.runner import (
        DREAM_REPORT_ERROR_FIELDS, DREAM_REPORT_BOOLEAN_GATE_FIELDS,
    )

    worker_report = validate_worker_report(worker_report)
    required = {"target_current", "target_pinned", "target_pinned_with_proof"}
    if set(before_publications) != required or set(after_publications) != required:
        raise ValueError("publication_fields_invalid")
    if set(before_quarantines) != {"retry_limit", "terminal_loss"} or set(after_quarantines) != {"retry_limit", "terminal_loss"}:
        raise ValueError("quarantine_fields_invalid")
    schema_ok = baseline_version == 63 and dream_version == 64
    publication_ok = (before_publications["target_current"] == 0
                      and after_publications == {"target_current": 1,
                                                 "target_pinned": 1,
                                                 "target_pinned_with_proof": 1})
    new_quarantines = {
        kind: len(after_quarantines[kind] - before_quarantines[kind])
        for kind in ("retry_limit", "terminal_loss")
    }
    quarantine_counts = {
        kind: {"baseline": len(before_quarantines[kind]),
               "dream": len(after_quarantines[kind]),
               "introduced": new_quarantines[kind]}
        for kind in new_quarantines
    }
    counters = summary["dream"]["latest_dream"]["counters"]
    run_degraded = any(counters.get(name, 0) > 0 for name in degraded_counters)
    worker_clean = (all(worker_report[name] == 0 for name in DREAM_REPORT_ERROR_FIELDS)
                    and all(worker_report[name] is False for name in DREAM_REPORT_BOOLEAN_GATE_FIELDS))
    no_new_degradation = (not run_degraded and worker_clean
                          and not aggregation_blocking_present
                          and all(n == 0 for n in new_quarantines.values()))
    passed = (schema_ok and publication_ok and summary["audit_clean"] is True
              and summary["source_unchanged"] is True
              and summary["dream_completed"] is True
              and summary["campaign"]["worker_status"] == "completed"
              and worker_report["chunks_processed"] > 0
              and stored_counter_consistent
              and no_new_degradation)
    return {
        "status": "pass" if passed else "fail",
        "schema": {"baseline": baseline_version, "dream": dream_version,
                   "expected_transition": schema_ok},
        "publication": {"baseline": before_publications,
                        "dream": after_publications, "valid": publication_ok},
        "quarantines": quarantine_counts,
        "baseline_quarantines_present": any(before_quarantines.values()),
        "new_degradation_detected": not no_new_degradation,
        "worker_report_clean": worker_clean,
        "worker_store_campaign_consistent": stored_counter_consistent,
        "aggregation_blocking_present": aggregation_blocking_present,
        "audit_clean": summary["audit_clean"],
        "source_unchanged": summary["source_unchanged"],
        "baseline_sha256": summary["baseline_sha256"],
        "source_sha256": summary["source_sha256"],
        "latest_dream_advanced": summary["latest_dream_advanced"],
        "dream_completed": summary["dream_completed"],
        "campaign": summary["campaign"],
        "baseline_counts": summary["baseline"]["counts"],
        "dream_counts": summary["dream"]["counts"],
        "dream_run_counters": counters,
        "baseline_integrity": summary["baseline"]["integrity"],
        "dream_integrity": summary["dream"]["integrity"],
    }


def postflight(audit, worker_result: Path) -> dict:
    from hymem.core.db import schema_version

    summary = audit.audit()
    baseline = audit.inspect_copy(WORK / "baseline.sqlite")
    dream = audit.inspect_copy(WORK / "dream.sqlite")
    try:
        before_version, after_version = schema_version(baseline), schema_version(dream)
        proof_columns = {str(row["name"]) for row in dream.execute(
            "PRAGMA table_info(kg_claim_extraction_outcomes)")}
        if after_version == 64 and "local_replay_proof" not in proof_columns:
            raise ValueError("dream_proof_column_missing")
        worker_report, stored_match, aggregation_blocking = worker_evidence(
            worker_result, audit, dream, summary)
        report = assess(summary, before_version, after_version,
                        publication_counts(baseline, has_proof_column=False),
                        publication_counts(dream, has_proof_column="local_replay_proof" in proof_columns),
                        quarantine_ids(baseline), quarantine_ids(dream),
                        audit.DEGRADED_COUNTERS, worker_report, stored_match,
                        aggregation_blocking)
    finally:
        baseline.close()
        dream.close()
    # The second inspection must not invalidate the source immutability result.
    still_unchanged = (audit.sha(audit.BASELINE) == report["baseline_sha256"]
                       and audit.sha(audit.SOURCE) == report["source_sha256"])
    report["source_unchanged"] = report["source_unchanged"] and still_unchanged
    if not report["source_unchanged"]:
        report["status"] = "fail"
    return report


def main() -> int:
    global WORK
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-helper", type=Path, default=AUDIT_FILE)
    parser.add_argument("--source", type=Path, default=Path("/private-dream/hymem.sqlite"))
    parser.add_argument("--baseline", type=Path, default=Path("/reference/source.sqlite"))
    parser.add_argument("--result", type=Path, default=Path("/campaign/result.json"))
    parser.add_argument("--worker-result", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    WORK = args.work
    audit = load_audit(args.audit_helper)
    audit.SOURCE, audit.BASELINE, audit.RESULT, audit.WORK = (
        args.source, args.baseline, args.result, WORK)
    report = postflight(audit, args.worker_result)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink() and stat.S_IMODE(WORK.stat().st_mode) == 0o700:
                # Details remain in private evidence; stdout reveals only a type.
                raw = json.dumps({
                    "type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(exc)),
                }, sort_keys=True, ensure_ascii=True).encode()
                fd = os.open(WORK / "postflight-failure.json",
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                captured = True
        except BaseException:
            pass
        safe = type(exc).__name__ if type(exc).__name__ in {
            "ValueError", "RuntimeError", "TypeError", "OperationalError",
            "DatabaseError", "OSError", "KeyError"} else "Exception"
        print(json.dumps({"status": "error", "error_type": safe,
                          "failure_captured": captured}, sort_keys=True))
        code = 1
    raise SystemExit(code)
