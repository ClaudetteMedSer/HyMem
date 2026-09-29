"""Offline same-connection diagnosis of the retained v63-to-v64 digest gate.

The exact arm preserves the failed worker's call order before adding any row
scans. The instrumented arm adds physical-main-table snapshots. Neither arm
publishes claims or contacts a provider. Full metadata and tracebacks stay in
private /work; stdout contains only bounded codes, counts, hashes and approved
schema identifiers. This diagnoses the existing audit; it does not replace it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import stat
import sys
import traceback

sys.path.insert(0, "/candidate")
WORK = Path("/work")
SNAPSHOT = Path("/capture/source.sqlite")
V1_WORKER = Path("/diag/claim_conflict_proof_replay_v1.py")
V1_SHA = "014e045de6a5591eb87aabf0760247a69cd4d5a75f5bb7462e2d404caff7a9de"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PUBLIC_TABLES = frozenset({
    "sessions", "messages", "chunks", "chunk_message_sources", "message_retention_coverage",
    "entity_aliases", "knowledge_graph", "kg_evidence", "kg_claim_observations",
    "kg_claim_extraction_outcomes", "processed_chunks", "phase1_generations", "phase1_auxiliary_outcomes",
    "schema_meta",
} | {"vec_" + domain + suffix
     for domain in ("chunks", "messages", "edges", "episodes", "facts")
     for suffix in ("", "_chunks", "_rowids", "_info", "_vector_chunks00")})


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def private_json(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    need(len(raw) <= 4194304, "metadata_size_exceeded")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def load_v1():
    need(V1_WORKER.is_file() and not V1_WORKER.is_symlink() and sha(V1_WORKER) == V1_SHA,
         "audit_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location("drift_pinned_v1_audit", V1_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table_inventory(conn):
    rows = [tuple(row) for row in conn.execute("PRAGMA table_list")]
    need(len(rows) <= 2048 and all(
        len(row) == 6 and row[0] in ("main", "temp") and isinstance(row[1], str)
        and len(row[1]) <= 256 and row[2] in ("table", "shadow", "virtual", "view")
        and all(type(value) is int for value in row[3:]) for row in rows), "table_inventory_invalid")
    return sorted(rows)


def table_id(name):
    return name if name in PUBLIC_TABLES else "sha256:" + hashlib.sha256(name.encode()).hexdigest()


def inventory_changes(before, after):
    first = {(row[0], row[1]): row[2:] for row in before}
    second = {(row[0], row[1]): row[2:] for row in after}
    result = [{"schema": schema, "table_id": table_id(name),
               "before": first.get((schema, name)), "after": second.get((schema, name))}
              for schema, name in sorted(first.keys() | second.keys())
              if first.get((schema, name)) != second.get((schema, name))]
    need(len(result) <= 128, "inventory_change_limit")
    return result


def quoted(name):
    # Names are resolved from this connection's exact main schema inventory;
    # quoting is still mandatory. Schema qualification never comes from data.
    return '"' + name.replace('"', '""') + '"'


def physical_rows(conn):
    """Metadata-only main physical-table census, insensitive to shadow flags."""
    result = {}
    inventory = table_inventory(conn)
    authorized = {str(row[0]) for row in conn.execute("SELECT name FROM main.sqlite_master WHERE type='table'")}
    for schema, name, kind, _, without_rowid, _ in inventory:
        if (schema != "main" or kind not in ("table", "shadow")
                or name.startswith("sqlite_") or name == "schema_meta"):
            continue
        need(name in authorized, "table_not_in_exact_main_schema")
        columns = [str(row[1]) for row in conn.execute("PRAGMA main.table_info(" + quoted(name) + ")")]
        need(columns, "physical_columns_missing")
        selected = [column for column in columns
                    if not (name == "kg_claim_extraction_outcomes" and column == "local_replay_proof")]
        fields = ",".join(quoted(column) for column in selected)
        order = fields if without_rowid else "rowid"
        ordered, rows = hashlib.sha256(), []
        for row in conn.execute("SELECT " + fields + " FROM main." + quoted(name) + " ORDER BY " + order):
            item = hashlib.sha256(repr(tuple(row)).encode("utf-8", "backslashreplace")).digest()
            ordered.update(item)
            rows.append(item)
        unordered = hashlib.sha256(b"".join(sorted(rows))).hexdigest()
        result[name] = {"count": len(rows), "ordered_sha256": ordered.hexdigest(),
                        "unordered_sha256": unordered, "column_names_sha256": value_sha(columns),
                        "selected_columns_sha256": value_sha(selected)}
    return result


def compare_rows(before, after):
    names = before.keys() | after.keys()
    ordered_equal = before.keys() == after.keys() and all(
        before[name]["count"] == after[name]["count"]
        and before[name]["selected_columns_sha256"] == after[name]["selected_columns_sha256"]
        and before[name]["ordered_sha256"] == after[name]["ordered_sha256"] for name in names)
    unordered_equal = before.keys() == after.keys() and all(
        before[name]["count"] == after[name]["count"]
        and before[name]["selected_columns_sha256"] == after[name]["selected_columns_sha256"]
        and before[name]["unordered_sha256"] == after[name]["unordered_sha256"] for name in names)
    changed = [table_id(name) for name in sorted(names)
               if name not in before or name not in after or any(
                   before[name][key] != after[name][key]
                   for key in ("count", "selected_columns_sha256", "ordered_sha256", "unordered_sha256"))]
    need(len(changed) <= 128, "row_change_limit")
    return {"tables": len(names), "changed": len(changed), "ordered_equal": ordered_equal,
            "unordered_equal": unordered_equal, "changed_table_ids": changed}


def one_arm(v1, replay, *, instrumented, metadata):
    from hymem.core.db import initialize, schema_version
    label = "instrumented" if instrumented else "exact"
    target = WORK / (label + ".sqlite")
    replay.clone(SNAPSHOT, target)
    conn = replay.open_clone(target)
    record = {"inventories": {}}
    metadata[label] = record
    def inventory(label):
        rows = table_inventory(conn)
        record["inventories"][label] = rows
        return rows
    try:
        schema_before = schema_version(conn)
        need(schema_before == 63, "source_schema_not_v63")
        need("local_replay_proof" not in {row[1] for row in conn.execute(
            "PRAGMA table_info(kg_claim_extraction_outcomes)")}, "source_proof_already_present")
        inventory("opened")
        record["integrity_before"] = replay.integrity(conn)
        v1.require_clean(record["integrity_before"], "source_integrity_failure")
        before_inventory = inventory("before_first_digest")
        before = v1.semantic_digest(conn)
        inventory("after_first_digest")
        if instrumented:
            record["physical_before"] = physical_rows(conn)
            inventory("after_before_row_census")
            record["digest_before_after_census"] = v1.semantic_digest(conn)
            inventory("after_repeated_before_digest")
        initialize(conn)
        schema_after = schema_version(conn)
        need(schema_after == 64, "target_schema_not_v64")
        proof_nonnull = conn.execute("SELECT COUNT(*) FROM kg_claim_extraction_outcomes "
                                     "WHERE local_replay_proof IS NOT NULL").fetchone()[0]
        need(proof_nonnull == 0, "historical_proof_invented")
        inventory("after_initialize")
        record["integrity_after"] = replay.integrity(conn)
        v1.require_clean(record["integrity_after"], "target_integrity_failure")
        inventory("before_after_digest")
        after = v1.semantic_digest(conn)
        after_inventory = inventory("after_after_digest")
        again = v1.semantic_digest(conn)
        inventory("after_repeat_digest")
        record["physical_after"] = physical_rows(conn)
        inventory("after_after_row_census")
        result = {"schema_before": schema_before, "schema_after": schema_after,
                  "proof_nonnull": proof_nonnull, "digest_before": before, "digest_after": after,
                  "digest_after_repeat": again, "inventory_before_sha256": value_sha(before_inventory),
                  "inventory_after_sha256": value_sha(after_inventory),
                  "table_changes": inventory_changes(before_inventory, after_inventory)}
        if instrumented:
            result["row_comparison"] = compare_rows(record["physical_before"], record["physical_after"])
        record["summary"] = result
        return result
    finally:
        conn.close()


def main():
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--phase1-sha256", required=True)
    args = parser.parse_args()
    need(HEX64.fullmatch(args.snapshot_sha256) and HEX64.fullmatch(args.phase1_sha256), "input_hash_invalid")
    need(WORK.is_dir() and not WORK.is_symlink() and stat.S_IMODE(WORK.stat().st_mode) == 0o700,
         "work_not_private")
    need(SNAPSHOT.is_file() and not SNAPSHOT.is_symlink() and sha(SNAPSHOT) == args.snapshot_sha256,
         "snapshot_pin_drift")
    need(sha(Path("/candidate/hymem/dreaming/phase1.py")) == args.phase1_sha256, "phase1_pin_drift")
    v1 = load_v1()
    replay = v1.load_replay()
    metadata = {}
    try:
        exact = one_arm(v1, replay, instrumented=False, metadata=metadata)
        instrumented = one_arm(v1, replay, instrumented=True, metadata=metadata)
    finally:
        private_json(WORK / "drift-metadata.json", metadata)
    need(sha(SNAPSHOT) == args.snapshot_sha256, "source_changed")
    print(json.dumps({"status": "completed", "snapshot_sha256": args.snapshot_sha256,
                      "phase1_sha256": args.phase1_sha256, "source_unchanged": True,
                      "metadata_sha256": sha(WORK / "drift-metadata.json"),
                      "exact": exact, "instrumented": instrumented}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:
        captured = False
        try:
            private_json(WORK / "drift-failure.json", {
                "type": type(exc).__name__,
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            })
            captured = True
        except BaseException:
            pass
        print(json.dumps({"status": "error", "reason_code": "drift_audit_failed",
                          "failure_captured": captured}, sort_keys=True))
        code = 1
    raise SystemExit(code)
