#!/usr/bin/env python3
"""Exact, network-free replay of one private prepublication capture.

Run in a pinned container with /candidate, /capture (read only), and a fresh
0700 /work. Stdout contains counts and hashes only; full errors stay private.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
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

CAPTURE = Path("/capture")
WORK = Path("/work")
SOURCE_SHA256 = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
GENERATION = "hymem-phase1-generation-v1:6075085e12e32e1b790e49b99b8c3bb50718b18be0f28762d1582c58ee8e35eb"
PHASE1_SHAS = frozenset({
    "31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136",
    "540d257c596850b44adaacab922587b99445def7a7b6b6d7696e0739bbc2e27d",
    "c7946257957d2dd6b6e00728ce2c12a0b69031376a286c722748472575339fbb",
})
FRAME_RE = re.compile(r"hymem(?:/[A-Za-z_][A-Za-z_0-9]*)*/[A-Za-z_][A-Za-z_0-9]*\.py\Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("private_capture_file_invalid")


def private_json(path: Path, value: object) -> None:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True,
                     allow_nan=False, separators=(",", ":")).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def logical_digest(conn: sqlite3.Connection) -> str:
    h = hashlib.sha256()
    for statement in conn.iterdump():
        h.update(statement.encode())
        h.update(b"\n")
    return h.hexdigest()


def safe_frames(exc: BaseException) -> list[dict]:
    frames = []
    for index, (frame, line) in enumerate(traceback.walk_tb(exc.__traceback__)):
        if index >= 96:
            break
        filename = frame.f_code.co_filename
        if not filename.startswith("/candidate/hymem/"):
            continue
        relative = filename[len("/candidate/"):]
        name = frame.f_code.co_name
        if FRAME_RE.fullmatch(relative) and name.isidentifier() and len(name) <= 96 and 1 <= line <= 100000:
            frames.append({"path": relative, "function": name, "line": line})
    return frames[-12:]


def safe_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if name in {"ValueError", "RuntimeError", "TypeError", "KeyError",
                            "IntegrityError", "OperationalError", "DatabaseError",
                            "OSError", "AssertionError", "MemoryError"} else "Exception"


def reason(exc: BaseException) -> str:
    fixed = {
        "same-generation claim observations disagree": "same_generation_observation_disagreement",
        "duplicate claim citation in validated extraction": "duplicate_validated_claim_citation",
        "same prompt generation claim extraction outcomes disagree": "same_generation_extraction_outcome_disagreement",
        "canonical evidence source provenance collides": "canonical_evidence_provenance_collision",
        "canonical edge merge found conflicting evidence audit": "canonical_evidence_audit_collision",
        "canonical edge merge found conflicting claim authority": "canonical_claim_authority_collision",
        "claim extraction outcome requires a published source manifest": "missing_published_source_manifest",
    }
    if isinstance(exc, sqlite3.IntegrityError):
        return "sqlite_integrity_rejection"
    message = str(exc)
    if (type(exc) is ValueError and len(message) <= 1024
            and message.startswith("canonical identity ")
            and message.endswith(" already owns state; use merge_canonical/merge")
            and any(frame.f_code.co_name == "register_alias"
                    and frame.f_code.co_filename.endswith("/hymem/dreaming/canonicalize.py")
                    for frame, _ in traceback.walk_tb(exc.__traceback__))):
        return "alias_owned_state_guard"
    return fixed.get(message, "unclassified_exception")


def input_paths(index: int) -> tuple[Path, Path]:
    if not 1 <= index <= 32:
        raise ValueError("capture_index_out_of_range")
    stem = f"prepersist-{index:03d}"
    return CAPTURE / (stem + ".json"), CAPTURE / (stem + ".sqlite")


def load_input(index: int) -> tuple[dict, Path, dict]:
    manifest_path, source = input_paths(index)
    private_file(manifest_path)
    private_file(source)
    raw = json.loads(manifest_path.read_bytes())
    if not isinstance(raw, dict) or raw.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("capture_source_identity_mismatch")
    source_hash = sha(source)
    if raw.get("database_sha256") != source_hash:
        raise ValueError("prepublication_snapshot_pin_mismatch")
    extraction = raw.get("extraction")
    generation = extraction.get("phase1_generation") if isinstance(extraction, dict) else None
    if not isinstance(generation, dict) or generation.get("generation_key") != GENERATION:
        raise ValueError("capture_generation_mismatch")
    cache_key = captured_cache_key(raw)
    if generation.get("extraction_cache_key") != cache_key:
        raise ValueError("capture_cache_key_mismatch")
    candidate_sha = sha(Path("/candidate/hymem/dreaming/phase1.py"))
    if candidate_sha not in PHASE1_SHAS:
        raise ValueError("candidate_phase1_pin_mismatch")
    return raw, source, {"capture_sha256": sha(manifest_path),
                         "snapshot_sha256": source_hash,
                         "phase1_sha256": candidate_sha}


def captured_cache_key(raw: dict) -> str:
    from hymem.extraction.contract import extraction_cache_key

    label = raw.get("prompt_version")
    if not isinstance(label, str) or not label:
        raise ValueError("capture_prompt_version_invalid")
    return extraction_cache_key(label)


def reconstruct(raw: dict, *, dedup_enabled: bool):
    from hymem.config import HyMemConfig
    from hymem.dreaming.chunks import Chunk
    from hymem.dreaming.lossless import CoveredMessage
    from hymem.dreaming.phase1 import ChunkExtraction, _InCycleEdge, _PreparedDedupVectors
    from hymem.extraction.markers import Marker
    from hymem.extraction.triples import Triple

    chunk_data = dict(raw["chunk"])
    chunk_data["source_message_ids"] = tuple(chunk_data["source_message_ids"])
    chunk = Chunk(**chunk_data)
    extraction_data = dict(raw["extraction"])
    extraction_data["triples"] = [Triple(**item) for item in extraction_data["triples"]]
    extraction_data["markers"] = [Marker(**item) for item in extraction_data["markers"]]
    extraction_data["failure_details"] = tuple(extraction_data["failure_details"])
    extraction_data["claim_sources"] = {
        int(key): CoveredMessage(**value)
        for key, value in extraction_data["claim_sources"].items()
    }
    extraction = ChunkExtraction(**extraction_data)
    config_data = dict(raw["cfg"])
    config_fields = fields(HyMemConfig)
    permitted = {field.name for field in config_fields}
    init_fields = {field.name for field in config_fields if field.init}
    if set(config_data) != permitted:
        raise ValueError("captured_config_shape_changed")
    captured_derived = {name: config_data.pop(name) for name in permitted - init_fields}
    config_data["root"] = WORK
    baseline_cfg = HyMemConfig(**config_data)
    if any(json.loads(json.dumps(asdict(baseline_cfg)[name], default=str)) != value
           for name, value in captured_derived.items()):
        raise ValueError("captured_config_derived_identity_changed")
    cfg = replace(baseline_cfg, triple_dedup_enabled=dedup_enabled)
    saved_vectors = raw["dedup_vectors"]
    if saved_vectors is None:
        vectors = None
    elif raw.get("dedup_model") is None and raw.get("dedup_dim") is None:
        if saved_vectors:
            raise ValueError("dedup_vector_identity_missing")
        vectors = {}
    else:
        vectors = _PreparedDedupVectors(model=raw["dedup_model"], dim=raw["dedup_dim"])
        vectors.update(saved_vectors)
    saved_pool = raw["in_cycle_edges"]
    pool = None if saved_pool is None else [_InCycleEdge(**item) for item in saved_pool]
    return chunk, extraction, cfg, vectors, pool


def clone(source: Path, target: Path) -> None:
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    origin = sqlite3.connect(f"file:{source}?mode=ro&immutable=1", uri=True)
    try:
        dest = sqlite3.connect(target)
        try:
            origin.backup(dest)
        finally:
            dest.close()
    finally:
        origin.close()


def open_clone(path: Path):
    from hymem.core.db import connect, _load_vec_extension
    conn = connect(path)
    try:
        has_vec = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%USING vec0%'"
        ).fetchone()
        if has_vec and not _load_vec_extension(conn):
            raise RuntimeError("snapshot_vector_extension_unavailable")
        return conn
    except BaseException:
        conn.close()
        raise


def publication_count(conn: sqlite3.Connection, chunk_id: str,
                      cache_key: str, generation_key: str) -> int:
    return int(conn.execute(
        "SELECT COUNT(*) FROM current_phase1_publications "
        "WHERE chunk_id=? AND prompt_version=? AND phase1_generation_key=?",
        (chunk_id, cache_key, generation_key),
    ).fetchone()[0])


def integrity(conn) -> dict:
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
            "same_generation_disagreeing_groups": conflicts}


def attempt(conn, raw: dict, *, dedup_enabled: bool, label: str) -> dict:
    from hymem.core.db import transaction
    from hymem.dreaming.phase1 import persist_chunk_results

    chunk, extraction, cfg, vectors, pool = reconstruct(raw, dedup_enabled=dedup_enabled)
    before = logical_digest(conn)
    pool_count_before = None if pool is None else len(pool)
    try:
        with transaction(conn):
            staged = persist_chunk_results(
                conn, chunk, extraction, prompt_version=raw["prompt_version"],
                cfg=cfg, embedding_client=None, dedup_vectors=vectors,
                in_cycle_edges=pool,
            )
        after = logical_digest(conn)
        return {"status": "persisted", "reason_code": None,
                "logical_digest_before": before, "logical_digest_after": after,
                "pool_count_before": pool_count_before,
                "pool_count_after": None if staged is None else len(staged),
                "rollback_preserved": None}
    except Exception as exc:
        diagnostic_error_type = None
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            after = logical_digest(conn)
        except BaseException as diagnostic_exc:
            after = None
            diagnostic_error_type = safe_type(diagnostic_exc)
        rollback = before == after if after is not None else None
        code = reason(exc)
        try:
            private_json(WORK / (label + "-failure.json"), {
                "type": type(exc).__name__,
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            })
            failure_captured = True
        except BaseException:
            failure_captured = False
        status = ("rejected" if code != "unclassified_exception" and rollback and failure_captured
                  else "execution_failure")
        return {"status": status, "reason_code": code,
                "error_type": safe_type(exc), "candidate_frames": safe_frames(exc),
                "failure_captured": failure_captured,
                "diagnostic_error_type": diagnostic_error_type,
                "logical_digest_before": before, "logical_digest_after": after,
                "rollback_preserved": rollback,
                "pool_count_before": pool_count_before}


def run_arm(raw: dict, source: Path, *, dedup_enabled: bool, label: str) -> dict:
    cache_key = captured_cache_key(raw)
    generation = raw.get("extraction", {}).get("phase1_generation")
    if not isinstance(generation, dict) or generation.get("generation_key") != GENERATION \
            or generation.get("extraction_cache_key") != cache_key:
        raise ValueError("capture_generation_binding_mismatch")
    target = WORK / (label + ".sqlite")
    clone(source, target)
    conn = open_clone(target)
    try:
        chunk, extraction, _, _, _ = reconstruct(raw, dedup_enabled=dedup_enabled)
        if extraction.phase1_generation["generation_key"] != GENERATION:
            raise ValueError("capture_generation_mismatch")
        registered = conn.execute(
            "SELECT extraction_cache_key FROM phase1_generations WHERE generation_key=?",
            (GENERATION,),
        ).fetchone()
        if registered is None or registered[0] != cache_key:
            raise ValueError("generation_cache_key_not_registered_in_snapshot")
        saved = conn.execute("SELECT 1 FROM chunks WHERE id=? AND session_id=?",
                             (chunk.id, chunk.session_id)).fetchone()
        if saved is None:
            raise ValueError("captured_chunk_absent_from_snapshot")
        def current_publications() -> int:
            return publication_count(conn, chunk.id, cache_key, GENERATION)
        published_before = current_publications()
        before_integrity = integrity(conn)
        first = attempt(conn, raw, dedup_enabled=dedup_enabled, label=label + "-first")
        published_after_first = current_publications()
        second = (attempt(conn, raw, dedup_enabled=dedup_enabled, label=label + "-exact-repeat")
                  if first["status"] == "persisted" else None)
        published_after_repeat = current_publications() if second is not None else None
        repeat_unchanged = (second["status"] == "persisted"
                            and second["logical_digest_before"] == second["logical_digest_after"]
                            if second is not None else None)
        after_integrity = integrity(conn)
        clean = all(integrity_result["integrity_ok"] is True and all(
            integrity_result[key] == 0 for key in (
                "foreign_key_findings", "canonical_drift_findings",
                "ledger_count_mismatches", "same_generation_disagreeing_groups"))
            for integrity_result in (before_integrity, after_integrity))
        result = {"status": "completed" if clean
                and first["status"] in ("persisted", "rejected")
                and (first["status"] != "rejected" or first["rollback_preserved"] is True)
                and (second is None or repeat_unchanged is True)
                else "inconclusive",
                "dedup_enabled": dedup_enabled,
                "first": first, "exact_repeat": second,
                "published_before": published_before,
                "published_after_first": published_after_first,
                "published_after_repeat": published_after_repeat,
                "exact_repeat_unchanged": repeat_unchanged,
                "integrity_before": before_integrity,
                "integrity_after": after_integrity}
    finally:
        conn.close()
    result["database_sha256"] = sha(target)
    return result


def main() -> int:
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-index", required=True, type=int)
    args = parser.parse_args()
    if not WORK.is_dir() or WORK.is_symlink() or stat.S_IMODE(WORK.stat().st_mode) != 0o700:
        raise ValueError("private_work_directory_invalid")
    raw, source, pins = load_input(args.capture_index)
    result = {"status": "replayed", "capture_index": args.capture_index,
              **pins,
              "dedup_on": run_arm(raw, source, dedup_enabled=True, label="dedup-on"),
              "dedup_off": run_arm(raw, source, dedup_enabled=False, label="dedup-off")}
    if sha(source) != pins["snapshot_sha256"]:
        raise RuntimeError("capture_snapshot_changed_during_replay")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if all(result[name]["status"] == "completed"
                    for name in ("dedup_on", "dedup_off")) else 1


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink():
                private_json(WORK / "setup-failure.json", {
                    "type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                })
                captured = True
        except BaseException:
            pass
        print(json.dumps({"status": "error", "reason_code": "replay_setup_failed",
                          "error_type": safe_type(exc),
                          "candidate_frames": safe_frames(exc),
                          "failure_captured": captured}, sort_keys=True))
        rc = 1
    raise SystemExit(rc)
