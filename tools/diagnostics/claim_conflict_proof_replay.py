#!/usr/bin/env python3
"""Network-free v61→v62 migration and exact retained extraction proof replay.

Inputs are read-only /capture/prepersist-001.{json,sqlite}, /reference/source.sqlite,
and pinned /diag/claim_conflict_instrumented_replay.py. All writes stay in
fresh private /work. Stdout contains only finite codes, counts, and hashes.
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
import sqlite3
import stat
import sys
import traceback

sys.path.insert(0, "/candidate")

WORK = Path("/work")
CAPTURE = Path("/capture")
REFERENCE = Path("/reference/source.sqlite")
OLD_REPLAY = Path("/diag/claim_conflict_instrumented_replay.py")
OLD_REPLAY_SHA = "7b94a69f2a152f1a818ab7b5833b8d619734b3f7023de2dca89b4190f9013085"
REFERENCE_SHA = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
PROOF = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SEMANTIC_TABLES = (
    "sessions", "messages", "chunks", "chunk_message_sources",
    "message_retention_coverage", "entity_aliases", "knowledge_graph",
    "kg_evidence", "kg_claim_observations", "kg_claim_extraction_outcomes",
    "processed_chunks", "phase1_generations", "phase1_auxiliary_outcomes",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def private_input(path: Path) -> None:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or stat.S_IMODE(info.st_mode) not in (0o400, 0o600)):
        raise ValueError("private_input_mode_invalid")


def load_replay():
    if not OLD_REPLAY.is_file() or OLD_REPLAY.is_symlink() or sha(OLD_REPLAY) != OLD_REPLAY_SHA:
        raise RuntimeError("replay_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location("proof_replay_pinned_exact", OLD_REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CAPTURE = CAPTURE
    module.WORK = WORK
    return module


def semantic_digest(conn: sqlite3.Connection) -> str:
    """Hash all ordinary application rows, excluding only schema version/proof."""
    digest = hashlib.sha256()
    ordinary = {
        str(row[1]): bool(row[4])
        for row in conn.execute("PRAGMA table_list")
        if row[2] == "table" and not str(row[1]).startswith("sqlite_")
        and row[1] != "schema_meta"
    }
    if not set(SEMANTIC_TABLES).issubset(ordinary):
        raise RuntimeError("semantic_audit_table_missing")
    for table in sorted(ordinary):
        quoted_table = '"' + table.replace('"', '""') + '"'
        columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({quoted_table})")]
        if not columns:
            raise RuntimeError("semantic_audit_table_missing")
        columns = [name for name in columns if name != "local_replay_proof"]
        quoted = ",".join('"' + name.replace('"', '""') + '"' for name in columns)
        digest.update(table.encode("ascii") + b"\0")
        order = quoted if ordinary[table] else "rowid"
        for row in conn.execute(f"SELECT {quoted} FROM {quoted_table} ORDER BY {order}"):
            digest.update(repr(tuple(row)).encode("utf-8", "backslashreplace"))
            digest.update(b"\n")
    return digest.hexdigest()


def require_clean(audit: dict, code: str) -> None:
    if not (audit["integrity_ok"] is True and all(audit[field] == 0 for field in (
            "foreign_key_findings", "canonical_drift_findings",
            "ledger_count_mismatches", "same_generation_disagreeing_groups"))):
        raise RuntimeError(code)


def proof_row(conn: sqlite3.Connection, chunk_id: str,
              cache_key: str, generation: str) -> str | None:
    row = conn.execute(
        "SELECT local_replay_proof FROM kg_claim_extraction_outcomes "
        "WHERE chunk_id=? AND prompt_version=? AND phase1_generation_key=?",
        (chunk_id, cache_key, generation),
    ).fetchone()
    return None if row is None else row[0]


def one_arm(replay, raw: dict, source: Path, *, dedup: bool, label: str) -> dict:
    from hymem.core.db import initialize, schema_version

    cache_key = replay.captured_cache_key(raw)
    generation = raw["extraction"]["phase1_generation"]["generation_key"]
    if generation != replay.GENERATION or raw["extraction"]["phase1_generation"].get(
            "extraction_cache_key") != cache_key:
        raise ValueError("capture_generation_binding_mismatch")
    target = WORK / (label + ".sqlite")
    replay.clone(source, target)
    conn = replay.open_clone(target)
    try:
        if schema_version(conn) != 61:
            raise RuntimeError("snapshot_schema_not_v61")
        before = replay.integrity(conn)
        require_clean(before, "preupgrade_integrity_failure")
        before_semantic = semantic_digest(conn)
        initialize(conn)
        if schema_version(conn) != 62:
            raise RuntimeError("schema_upgrade_incomplete")
        historical_count = int(conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes"
        ).fetchone()[0])
        historical_proofs = int(conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes "
            "WHERE local_replay_proof IS NOT NULL"
        ).fetchone()[0])
        after_upgrade = replay.integrity(conn)
        require_clean(after_upgrade, "postupgrade_integrity_failure")
        after_upgrade_semantic = semantic_digest(conn)
        if historical_proofs or after_upgrade_semantic != before_semantic:
            raise RuntimeError("upgrade_semantic_or_proof_drift")
        once_digest = replay.logical_digest(conn)
        initialize(conn)
        twice_digest = replay.logical_digest(conn)
        if once_digest != twice_digest or semantic_digest(conn) != before_semantic:
            raise RuntimeError("second_initialize_changed_state")
        chunk, extraction, _, _, _ = replay.reconstruct(raw, dedup_enabled=dedup)
        if extraction.phase1_generation["generation_key"] != generation:
            raise RuntimeError("reconstructed_generation_drift")
        registered = conn.execute(
            "SELECT extraction_cache_key FROM phase1_generations WHERE generation_key=?",
            (generation,),
        ).fetchone()
        if registered is None or registered[0] != cache_key:
            raise RuntimeError("registered_generation_namespace_drift")
        published_before = replay.publication_count(conn, chunk.id, cache_key, generation)
        if published_before != 0:
            raise RuntimeError("target_already_published")
        first = replay.attempt(conn, raw, dedup_enabled=dedup, label=label + "-first")
        published_after_first = replay.publication_count(conn, chunk.id, cache_key, generation)
        first_proof = proof_row(conn, chunk.id, cache_key, generation)
        after_first = replay.integrity(conn)
        require_clean(after_first, "first_publication_integrity_failure")
        if (first["status"] != "persisted" or published_after_first != 1
                or not isinstance(first_proof, str) or not PROOF.fullmatch(first_proof)
                or first["logical_digest_before"] == first["logical_digest_after"]):
            raise RuntimeError("first_publication_or_proof_missing")
        first_digest = replay.logical_digest(conn)
        if first_digest != first["logical_digest_after"]:
            raise RuntimeError("first_publication_digest_drift")
        conn.close()
        conn = replay.open_clone(target)
        initialize(conn)
        reopened_proof = proof_row(conn, chunk.id, cache_key, generation)
        reopened_digest = replay.logical_digest(conn)
        after_reopen = replay.integrity(conn)
        require_clean(after_reopen, "reopen_integrity_failure")
        if reopened_proof != first_proof or reopened_digest != first_digest:
            raise RuntimeError("proof_or_state_changed_on_reopen")
        repeat = replay.attempt(conn, raw, dedup_enabled=dedup, label=label + "-repeat")
        published_after_repeat = replay.publication_count(conn, chunk.id, cache_key, generation)
        repeat_proof = proof_row(conn, chunk.id, cache_key, generation)
        after_repeat = replay.integrity(conn)
        require_clean(after_repeat, "repeat_integrity_failure")
        if (repeat["status"] != "persisted" or published_after_repeat != 1
                or repeat_proof != first_proof
                or repeat["logical_digest_before"] != reopened_digest
                or repeat["logical_digest_after"] != reopened_digest):
            raise RuntimeError("exact_repeat_changed_state")
        report = {"status": "completed", "dedup_enabled": dedup,
                "historical_outcomes": historical_count,
                "historical_proofs_after_upgrade": historical_proofs,
                "semantic_digest_before_upgrade": before_semantic,
                "semantic_digest_after_upgrade": after_upgrade_semantic,
                "second_initialize_unchanged": once_digest == twice_digest,
                "published_before": published_before,
                "published_after_first": published_after_first,
                "published_after_repeat": published_after_repeat,
                "proof_sha256": hashlib.sha256(first_proof.encode()).hexdigest(),
                "proof_reopen_unchanged": reopened_proof == first_proof,
                "proof_repeat_unchanged": repeat_proof == first_proof,
                "exact_repeat_unchanged": True,
                "integrity_before": before, "integrity_after_upgrade": after_upgrade,
                "integrity_after_first": after_first,
                "integrity_after_reopen": after_reopen,
                "integrity_after_repeat": after_repeat,
                }
    finally:
        conn.close()
    report["database_sha256"] = sha(target)
    return report


def main() -> int:
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-sha256", required=True)
    args = parser.parse_args()
    if not HEX64.fullmatch(args.phase1_sha256):
        raise ValueError("candidate_phase1_pin_invalid")
    if (not WORK.is_dir() or WORK.is_symlink()
            or stat.S_IMODE(WORK.stat().st_mode) != 0o700):
        raise ValueError("private_work_directory_invalid")
    replay = load_replay()
    replay.PHASE1_SHAS = frozenset({args.phase1_sha256})
    private_input(REFERENCE)
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("reference_source_pin_mismatch")
    raw, source, pins = replay.load_input(1)
    arms = {name: one_arm(replay, raw, source, dedup=enabled, label=name)
            for name, enabled in (("dedup_on", True), ("dedup_off", False))}
    if sha(source) != pins["snapshot_sha256"] or sha(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError("read_only_inputs_changed")
    print(json.dumps({"status": "completed", **pins,
                      "reference_sha256": REFERENCE_SHA, **arms},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink():
                replay = load_replay()
                replay.private_json(WORK / "proof-replay-failure.json", {
                    "type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(
                        type(exc), exc, exc.__traceback__)),
                })
                captured = True
        except BaseException:
            pass
        print(json.dumps({"status": "error", "reason_code": "proof_replay_failed",
                          "error_type": (type(exc).__name__ if type(exc).__name__ in {
                              "RuntimeError", "ValueError", "OSError", "OperationalError",
                              "IntegrityError", "TypeError"} else "Exception"),
                          "failure_captured": captured}, sort_keys=True))
        exit_code = 1
    raise SystemExit(exit_code)
