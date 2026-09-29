#!/usr/bin/env python3
"""Network-free exact private replay after the true R7 schema 63→64 upgrade.

This new stage imports only pinned, generic v1 audit/reconstruction helpers;
it never resumes or rewrites the failed proof-replay-v1 stage. Input is the
retained private prepublication capture, with a fresh writable /work clone.
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
REFERENCE = Path("/reference/source.sqlite")
V1_WORKER = Path("/diag/claim_conflict_proof_replay_v1.py")
V1_WORKER_SHA = "014e045de6a5591eb87aabf0760247a69cd4d5a75f5bb7462e2d404caff7a9de"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def load_v1():
    if (not V1_WORKER.is_file() or V1_WORKER.is_symlink()
            or hashlib.sha256(V1_WORKER.read_bytes()).hexdigest() != V1_WORKER_SHA):
        raise RuntimeError("v1_audit_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location("proof_v2_pinned_v1_audit", V1_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_arm(v1, replay, raw, source, *, dedup, label):
    from hymem.core.db import initialize, schema_version

    cache_key = replay.captured_cache_key(raw)
    generation = raw["extraction"]["phase1_generation"]["generation_key"]
    if (generation != replay.GENERATION
            or raw["extraction"]["phase1_generation"].get("extraction_cache_key")
            != cache_key):
        raise ValueError("capture_generation_binding_mismatch")
    target = WORK / (label + ".sqlite")
    replay.clone(source, target)
    conn = replay.open_clone(target)
    try:
        if schema_version(conn) != 63:
            raise RuntimeError("snapshot_schema_not_v63")
        old_columns = {str(row[1]) for row in conn.execute(
            "PRAGMA table_info(kg_claim_extraction_outcomes)")}
        if "local_replay_proof" in old_columns:
            raise RuntimeError("preupgrade_proof_column_present")
        before = replay.integrity(conn)
        v1.require_clean(before, "preupgrade_integrity_failure")
        before_semantic = v1.semantic_digest(conn)
        initialize(conn)
        if schema_version(conn) != 64:
            raise RuntimeError("schema_upgrade_incomplete")
        new_columns = {str(row[1]) for row in conn.execute(
            "PRAGMA table_info(kg_claim_extraction_outcomes)")}
        if "local_replay_proof" not in new_columns:
            raise RuntimeError("postupgrade_proof_column_missing")
        historical_count = int(conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes"
        ).fetchone()[0])
        historical_proofs = int(conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes "
            "WHERE local_replay_proof IS NOT NULL"
        ).fetchone()[0])
        after_upgrade = replay.integrity(conn)
        v1.require_clean(after_upgrade, "postupgrade_integrity_failure")
        after_upgrade_semantic = v1.semantic_digest(conn)
        if historical_proofs or after_upgrade_semantic != before_semantic:
            raise RuntimeError("upgrade_semantic_or_proof_drift")
        once_digest = replay.logical_digest(conn)
        initialize(conn)
        twice_digest = replay.logical_digest(conn)
        if once_digest != twice_digest or v1.semantic_digest(conn) != before_semantic:
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
        first_proof = v1.proof_row(conn, chunk.id, cache_key, generation)
        after_first = replay.integrity(conn)
        v1.require_clean(after_first, "first_publication_integrity_failure")
        if (first["status"] != "persisted" or published_after_first != 1
                or not isinstance(first_proof, str) or not v1.PROOF.fullmatch(first_proof)
                or first["logical_digest_before"] == first["logical_digest_after"]):
            raise RuntimeError("first_publication_or_proof_missing")
        first_digest = replay.logical_digest(conn)
        if first_digest != first["logical_digest_after"]:
            raise RuntimeError("first_publication_digest_drift")
        conn.close()
        conn = replay.open_clone(target)
        initialize(conn)
        reopened_proof = v1.proof_row(conn, chunk.id, cache_key, generation)
        reopened_digest = replay.logical_digest(conn)
        after_reopen = replay.integrity(conn)
        v1.require_clean(after_reopen, "reopen_integrity_failure")
        if reopened_proof != first_proof or reopened_digest != first_digest:
            raise RuntimeError("proof_or_state_changed_on_reopen")
        repeat = replay.attempt(conn, raw, dedup_enabled=dedup, label=label + "-repeat")
        published_after_repeat = replay.publication_count(conn, chunk.id, cache_key, generation)
        repeat_proof = v1.proof_row(conn, chunk.id, cache_key, generation)
        after_repeat = replay.integrity(conn)
        v1.require_clean(after_repeat, "repeat_integrity_failure")
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
                  "integrity_after_repeat": after_repeat}
    finally:
        conn.close()
    report["database_sha256"] = v1.sha(target)
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
    v1 = load_v1()
    replay = v1.load_replay()
    replay.PHASE1_SHAS = frozenset({args.phase1_sha256})
    v1.private_input(REFERENCE)
    if v1.sha(REFERENCE) != v1.REFERENCE_SHA:
        raise ValueError("reference_source_pin_mismatch")
    raw, source, pins = replay.load_input(1)
    arms = {name: one_arm(v1, replay, raw, source, dedup=enabled, label=name)
            for name, enabled in (("dedup_on", True), ("dedup_off", False))}
    if (v1.sha(source) != pins["snapshot_sha256"]
            or v1.sha(REFERENCE) != v1.REFERENCE_SHA):
        raise RuntimeError("read_only_inputs_changed")
    print(json.dumps({"status": "completed", **pins,
                      "reference_sha256": v1.REFERENCE_SHA, **arms},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink():
                v1 = load_v1()
                v1.load_replay().private_json(WORK / "proof-v2-failure.json", {
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
