#!/usr/bin/env python3
"""Fresh offline R7 v63-to-v64 replay with a stable physical-main audit.

The retained audit proved that SQLite lazily reclassifies vec0 storage tables
from table to shadow after ALTER TABLE. Both are physical data and remain in
this digest. Only the diagnostic hash changes; application and old diagnostic
files remain immutable and the publication/reopen/repeat gates are unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import traceback

WORK = Path("/work")
V2_WORKER = Path("/diag/claim_conflict_proof_replay_v2.py")
V2_SHA = "8de6f54cc98abaaf5ac7f328ce2b4e4c5547128e580f3f0ecd82f8d6d353ec82"
DRIFT_WORKER = Path("/diag/claim_conflict_proof_drift.py")
DRIFT_SHA = "52ed6cf0543933249fabc41cbbd99770ba10d3e4e669dc4e97fc121b85aaf08f"
REQUIRED_TABLES = frozenset({
    "sessions", "messages", "chunks", "chunk_message_sources", "message_retention_coverage",
    "entity_aliases", "knowledge_graph", "kg_evidence", "kg_claim_observations",
    "kg_claim_extraction_outcomes", "processed_chunks", "phase1_generations",
    "phase1_auxiliary_outcomes",
})


def load_pinned(path, expected, name):
    if (not path.is_file() or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise RuntimeError("v3_diagnostic_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_semantic_digest(conn, audit):
    rows = audit.physical_rows(conn)
    if not REQUIRED_TABLES.issubset(rows):
        raise RuntimeError("semantic_audit_table_missing")
    # column_names_sha256 deliberately includes the newly added proof field;
    # selected_columns_sha256 excludes only that one approved outcome field.
    # No vector, FTS, other shadow, or ordinary application rows are omitted.
    fields = ("count", "selected_columns_sha256", "ordered_sha256", "unordered_sha256")
    payload = {table: {field: item[field] for field in fields} for table, item in rows.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def configured_v2():
    v2 = load_pinned(V2_WORKER, V2_SHA, "v3_pinned_v2_worker")
    audit = load_pinned(DRIFT_WORKER, DRIFT_SHA, "v3_pinned_physical_census")
    original_load = v2.load_v1
    def load_v1():
        v1 = original_load()
        if frozenset(v1.SEMANTIC_TABLES) != REQUIRED_TABLES:
            raise RuntimeError("v3_required_audit_tables_drift")
        v1.semantic_digest = lambda conn: stable_semantic_digest(conn, audit)
        return v1
    v2.load_v1 = load_v1
    return v2, audit


def main():
    v2, _ = configured_v2()
    return v2.main()


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:
        captured = False
        try:
            audit = load_pinned(DRIFT_WORKER, DRIFT_SHA, "v3_failure_writer")
            audit.private_json(WORK / "proof-v3-failure.json", {
                "type": type(exc).__name__,
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            })
            captured = True
        except BaseException:
            pass
        print(json.dumps({"status": "error", "reason_code": "proof_replay_failed",
                          "error_type": (type(exc).__name__ if type(exc).__name__ in {
                              "RuntimeError", "ValueError", "OSError", "OperationalError",
                              "IntegrityError", "TypeError"} else "Exception"),
                          "failure_captured": captured}, sort_keys=True))
        code = 1
    raise SystemExit(code)
