#!/usr/bin/env python3
"""Read-only post-dream health audit of a closed private clone.

Source and baseline are mounted read-only. SQLite backups into fresh 0700
/work provide consistent inspection copies. No provider or runtime env loads.
Stdout is bounded metadata; exceptions remain private 0600 evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import traceback

sys.path.insert(0, "/candidate")

SOURCE = Path("/private-dream/hymem.sqlite")
BASELINE = Path("/reference/source.sqlite")
RESULT = Path("/campaign/result.json")
WORK = Path("/work")
REFERENCE_SHA = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
COUNTER_TABLES = (
    "chunks", "processed_chunks", "current_phase1_publications",
    "chunk_extraction_attempts", "chunk_extraction_terminal_losses",
    "phase1_auxiliary_outcomes", "kg_claim_extraction_outcomes",
    "kg_claim_observations", "kg_evidence", "knowledge_graph",
    "message_retention_coverage", "chunk_message_sources",
    "chunk_embeddings", "message_embeddings", "edge_embeddings",
    "episode_embeddings", "fact_embeddings", "embedding_cache",
    "vec_chunks", "vec_messages", "vec_edges", "vec_episodes", "vec_facts",
)
DREAM_COUNTERS = (
    "sessions_processed", "chunks_seen", "chunks_processed",
    "chunk_extraction_completion_calls", "chunk_extraction_provider_attempts",
    "extraction_provider_attempt_budget_exhausted",
    "coverage_integrity_failures", "chunks_embedded", "edges_embedded",
    "triples_extracted", "markers_extracted", "aggregation_fusion_failures",
    "aggregation_build_exceptions", "digest_failures", "digest_quarantined",
    "fact_failures", "profile_failures", "skipped_locked",
)
DEGRADED_COUNTERS = (
    "extraction_provider_attempt_budget_exhausted", "coverage_integrity_failures",
    "aggregation_fusion_failures", "aggregation_build_exceptions",
    "digest_failures", "digest_quarantined", "fact_failures",
    "profile_failures", "skipped_locked",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def private_json(path: Path, value: object) -> None:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True,
                     allow_nan=False, separators=(",", ":")).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def checked_file(path: Path, *, mode: int | None = None) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError("audit_input_not_regular")
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        raise ValueError("audit_input_mode_invalid")


def backup_readonly(source: Path, target: Path) -> None:
    checked_file(source)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    origin = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        copy = sqlite3.connect(target)
        try:
            origin.backup(copy)
        finally:
            copy.close()
    finally:
        origin.close()


def inspect_copy(path: Path) -> sqlite3.Connection:
    from hymem.core.db import connect, _load_vec_extension
    conn = connect(path)
    try:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%USING vec0%'"
        ).fetchone() and not _load_vec_extension(conn):
            raise RuntimeError("vector_extension_unavailable")
        conn.execute("PRAGMA query_only=ON")
        return conn
    except BaseException:
        conn.close()
        raise


def integrity(conn: sqlite3.Connection) -> dict:
    from hymem.dreaming.canonicalize import find_canonical_drift
    from hymem.dreaming.evidence import count_mismatches

    rows = conn.execute("PRAGMA integrity_check").fetchall()
    foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
    conflicts = conn.execute(
        "SELECT COUNT(*) FROM ("
        "SELECT edge_id,source_session_id,source_message_id,evidence_kind,"
        "prompt_generation,phase1_generation_key FROM kg_claim_observations "
        "GROUP BY edge_id,source_session_id,source_message_id,evidence_kind,"
        "prompt_generation,phase1_generation_key "
        "HAVING MIN(polarity)<>MAX(polarity) OR "
        "MIN(interpretation_key)<>MAX(interpretation_key))"
    ).fetchone()[0]
    return {"integrity_ok": len(rows) == 1 and rows[0][0] == "ok",
            "foreign_key_findings": len(foreign),
            "canonical_drift_findings": len(find_canonical_drift(conn)),
            "ledger_count_mismatches": len(count_mismatches(conn)),
            "same_generation_disagreeing_groups": int(conflicts)}


def is_clean(audit: dict) -> bool:
    return (audit["integrity_ok"] is True
            and all(audit[name] == 0 for name in (
                "foreign_key_findings", "canonical_drift_findings",
                "ledger_count_mismatches", "same_generation_disagreeing_groups")))


def counts(conn: sqlite3.Connection) -> dict:
    available = {row["name"] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    values = {}
    for table in COUNTER_TABLES:
        if table in available:
            quoted = '"' + table.replace('"', '""') + '"'
            values[table] = int(conn.execute(
                f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
        else:
            values[table] = None
    values["extraction_chunks"] = int(conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_kind='extraction'").fetchone()[0])
    values["extraction_attempts_positive"] = int(conn.execute(
        "SELECT COUNT(*) FROM chunk_extraction_attempts WHERE attempts>0"
    ).fetchone()[0])
    values["extraction_attempts_ge_default_3"] = int(conn.execute(
        "SELECT COUNT(*) FROM chunk_extraction_attempts WHERE attempts>=3"
    ).fetchone()[0])
    values["chunk_embeddings_missing"] = int(conn.execute(
        "SELECT COUNT(*) FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id "
        "WHERE c.chunk_kind='extraction' AND e.chunk_id IS NULL"
    ).fetchone()[0])
    values["chunk_embedding_model_spaces"] = int(conn.execute(
        "SELECT COUNT(*) FROM (SELECT model,dim FROM chunk_embeddings GROUP BY model,dim)"
    ).fetchone()[0])
    values["message_embedding_model_spaces"] = int(conn.execute(
        "SELECT COUNT(*) FROM (SELECT model,dim FROM message_embeddings GROUP BY model,dim)"
    ).fetchone()[0])
    return values


def latest_dream(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT id,started_at,ended_at,error," + ",".join(DREAM_COUNTERS)
        + " FROM dream_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {"id": 0, "record_state": "none", "counters": {}}
    counters = {name: int(row[name]) for name in DREAM_COUNTERS}
    state = ("unfinished" if row["ended_at"] is None
             else "recorded_error" if row["error"] is not None
             else "skipped_locked" if counters["skipped_locked"] > 0
             else "recorded_completed")
    return {"id": int(row["id"]), "record_state": state, "counters": counters}


def campaign_result(path: Path) -> dict:
    checked_file(path, mode=0o600)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("campaign_result_invalid")
    metadata = value.get("stages", {}).get("live", {}).get("metadata", {})
    status = metadata.get("status")
    if status not in ("completed", "captured_failure", "budget_stopped", "error"):
        raise ValueError("campaign_live_status_invalid")
    projected = {"worker_status": status}
    for name in ("completion_calls", "http_attempts", "llm_http_attempts",
                 "embedding_http_attempts", "chunks_processed"):
        number = metadata.get(name)
        if type(number) is int and 0 <= number <= 100000000:
            projected[name] = number
    return projected


def audit() -> dict:
    if not WORK.is_dir() or WORK.is_symlink() or stat.S_IMODE(WORK.stat().st_mode) != 0o700:
        raise ValueError("private_work_directory_invalid")
    checked_file(BASELINE, mode=0o400)
    checked_file(SOURCE)
    baseline_sha = sha(BASELINE)
    source_sha = sha(SOURCE)
    if baseline_sha != REFERENCE_SHA:
        raise ValueError("baseline_reference_pin_mismatch")
    campaign = campaign_result(RESULT)
    backup_readonly(BASELINE, WORK / "baseline.sqlite")
    backup_readonly(SOURCE, WORK / "dream.sqlite")
    baseline = inspect_copy(WORK / "baseline.sqlite")
    dream = inspect_copy(WORK / "dream.sqlite")
    try:
        before = {"integrity": integrity(baseline), "counts": counts(baseline),
                  "latest_dream": latest_dream(baseline)}
        after = {"integrity": integrity(dream), "counts": counts(dream),
                 "latest_dream": latest_dream(dream)}
    finally:
        baseline.close()
        dream.close()
    unchanged = sha(BASELINE) == baseline_sha and sha(SOURCE) == source_sha
    latest_advanced = after["latest_dream"]["id"] > before["latest_dream"]["id"]
    degradation = (any(after["latest_dream"]["counters"].get(name, 0) > 0
                       for name in DEGRADED_COUNTERS)
                   or after["counts"]["extraction_attempts_ge_default_3"] > 0)
    audit_clean = is_clean(before["integrity"]) and is_clean(after["integrity"])
    dream_completed = (campaign["worker_status"] == "completed"
                       and latest_advanced
                       and after["latest_dream"]["record_state"] == "recorded_completed")
    return {"status": "integrity_failed" if not audit_clean else "audited",
            "audit_clean": audit_clean,
            "source_unchanged": unchanged,
            "baseline_sha256": baseline_sha, "source_sha256": source_sha,
            "baseline": before, "dream": after, "campaign": campaign,
            "latest_dream_advanced": latest_advanced,
            "dream_completed": dream_completed,
            "degraded_counters_present": degradation,
            "interpretation": ("not_clean" if not audit_clean
                               else "completed_with_degradation" if dream_completed and degradation
                               else "completed_no_detected_degradation" if dream_completed
                               else "not_completed")}


def safe_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if name in {"ValueError", "RuntimeError", "TypeError",
                            "OperationalError", "DatabaseError", "OSError"} else "Exception"


def main() -> int:
    global SOURCE, BASELINE, RESULT
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    SOURCE, BASELINE, RESULT = args.source, args.baseline, args.result
    report = audit()
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["audit_clean"] and report["source_unchanged"] else 1


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink():
                private_json(WORK / "audit-failure.json", {
                    "type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(
                        type(exc), exc, exc.__traceback__)),
                })
                captured = True
        except BaseException:
            pass
        print(json.dumps({"status": "error", "error_type": safe_type(exc),
                          "failure_captured": captured}, sort_keys=True))
        rc = 1
    raise SystemExit(rc)
