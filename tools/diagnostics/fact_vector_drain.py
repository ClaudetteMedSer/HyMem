"""Lease-fenced local fact-vector drain for a stopped partial replay.

Preview is read-only. Apply requires exact fact-state, vector-space, physical
mirror, and gap-count guards; it makes no LLM request. Only the configured
in-container embedding service may receive proof-valid current fact text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import struct

from hymem.core import db as core_db
from hymem.core.vectors import decode_vector
from hymem.dreaming import facts, runner
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.embeddings import (
    _finite_embedding_vector, fetch_fact_embeddings, persist_fact_embeddings,
)
from hymem.extraction.embeddings import embedding_text_hash

if __package__:
    from . import fact_only_replay as replay
else:
    import fact_only_replay as replay


MAX_BATCHES = 32
MAX_TEXTS = 2048
BATCH_SIZE = 64
REVIEWED_CAP32_TARGET = (
    "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=3072|items=32|"
    "semantic=sha256:545261b726153f79c65448281e0f426d8320f37c814f0fa9f5c39fa49600ab9c"
)
BACKUP_PATH = Path(
    "/home/node/.hermes/repair-backups/"
    "hymem-pre-partial-fact-vector-drain-20261017.sqlite"
)


def _load_running_embedding_env(proc_root: Path = Path("/proc")) -> None:
    """Bind only the local embedding settings of matching running producers."""
    candidates: dict[str, list[dict[str, str]]] = {
        "hymem-honcho": [], "hymem-server": [],
    }
    for process in sorted(proc_root.iterdir()):
        if not process.name.isdecimal():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
            tokens = {argument.rsplit(b"/", 1)[-1] for argument in argv}
            role = (
                "hymem-honcho" if tokens & {b"hymem-honcho", b"hymem.honcho.app"}
                else "hymem-server" if tokens & {b"hymem-server", b"hymem.server"}
                else None
            )
            if role is None:
                continue
            variables = dict(
                item.split(b"=", 1)
                for item in (process / "environ").read_bytes().split(b"\0")
                if b"=" in item
            )
            if variables.get(b"HYMEM_EMBEDDING_BASE_URL") != b"http://embedding-server:8766/v1":
                continue
            selected = {
                key.decode("ascii"): value.decode("utf-8")
                for key, value in variables.items()
                if key.startswith(b"HYMEM_EMBEDDING_")
            }
        except (OSError, UnicodeError, ValueError):
            continue
        candidates[role].append(selected)
    if (any(len(values) != 1 for values in candidates.values())
            or candidates["hymem-honcho"][0] != candidates["hymem-server"][0]):
        raise replay.RepairStopped("matching running embedding environments unavailable")
    for key in tuple(os.environ):
        if key.startswith("HYMEM_EMBEDDING_"):
            del os.environ[key]
    os.environ.update(candidates["hymem-honcho"][0])


def _target32(conn: sqlite3.Connection) -> str:
    versions = [row[0] for row in conn.execute(
        "SELECT DISTINCT prompt_version FROM fact_extraction_outcomes "
        "WHERE prompt_version LIKE '%|items=32|%'"
    )]
    if versions != [REVIEWED_CAP32_TARGET]:
        raise replay.RepairStopped("reviewed cap-32 fact generation is required")
    return REVIEWED_CAP32_TARGET


def _mirror_digest(conn: sqlite3.Connection) -> tuple[int, str]:
    """Require every JSON fact vector to have the identical physical row."""
    logical = conn.execute(
        "SELECT fact_id,vector_json FROM narrative_fact_embeddings "
        "ORDER BY fact_id"
    ).fetchall()
    physical = conn.execute(
        "SELECT rowid,embedding FROM vec_facts ORDER BY rowid"
    ).fetchall()
    if len(logical) != len(physical):
        raise replay.RepairStopped("fact vector mirror count differs")
    digest = hashlib.sha256()
    for left, right in zip(logical, physical):
        try:
            packed = core_db._pack_vector(decode_vector(left["vector_json"]))
            mirrored = bytes(right["embedding"])
        except (TypeError, ValueError, OverflowError):
            raise replay.RepairStopped("fact vector mirror contains invalid bytes") from None
        if int(left["fact_id"]) != int(right["rowid"]) or packed != mirrored:
            raise replay.RepairStopped("fact vector mirror differs")
        digest.update(struct.pack("<qI", int(left["fact_id"]), len(packed)))
        digest.update(packed)
    return len(logical), digest.hexdigest()


def _inventory(conn: sqlite3.Connection, embedder) -> dict:
    model, dim = embedding_storage_identity(embedder)
    stored = dict(conn.execute(
        "SELECT key,value FROM schema_meta WHERE key IN ('vec_model','vec_dim')"
    ).fetchall())
    if stored.get("vec_model") != model or stored.get("vec_dim") != str(dim):
        raise replay.RepairStopped("embedding identity differs from stored vector space")
    if not core_db._load_vec_extension(conn) or not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='vec_facts'"
    ).fetchone():
        raise replay.RepairStopped("physical fact vector mirror is unavailable")
    rows = conn.execute(
        "SELECT f.id,f.text,e.model,e.dim,e.text_hash,e.vector_json "
        "FROM narrative_facts f LEFT JOIN narrative_fact_embeddings e "
        "ON e.fact_id=f.id WHERE f.source_outcome_key IS NOT NULL "
        "AND f.lifecycle_status='active' AND f.invalid_at IS NULL "
        "ORDER BY f.id"
    ).fetchall()
    ids = [int(row["id"]) for row in rows]
    if set(facts.load_fact_source_manifests(conn, ids)) != set(ids):
        raise replay.RepairStopped("an active fact lacks complete source authority")
    gaps = 0
    for row in rows:
        valid = False
        if (row["model"] == model and row["dim"] == dim
                and row["text_hash"] == embedding_text_hash(row["text"])):
            try:
                vec = decode_vector(row["vector_json"])
                valid = _finite_embedding_vector(vec, expected_dim=dim) is not None
            except (TypeError, ValueError, OverflowError):
                pass
        if not valid:
            gaps += 1
    mirrored, mirror_sha = _mirror_digest(conn)
    return {
        "active_proof_valid": len(ids), "gaps": gaps,
        "mirror_rows": mirrored, "mirror_sha256": mirror_sha,
        "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
        "dim": dim,
    }


def _current_state(conn: sqlite3.Connection, target32: str) -> str:
    return replay._state(conn, target32)["state_sha256"]


def _reviewed_partial_baseline(state: dict) -> bool:
    return (
        state["outcomes"] == 128 and state["sessions"] == 49
        and state["stale"] == 77 and state["pending_fresh"] == 1
        and state["quarantined"] == 0
    )


def apply(conn: sqlite3.Connection, db_path: Path, embedder, *,
          target32: str, expected_state: str, expected_inventory: dict,
          backup_path: Path, max_batches: int, max_texts: int) -> dict:
    """Drain at most 32 local batches under one exact dreaming lease."""
    if (max_batches < 1 or max_batches > MAX_BATCHES
            or max_texts < BATCH_SIZE or max_texts > MAX_TEXTS):
        raise replay.RepairStopped("local vector ceilings exceed reviewed bounds")
    if (backup_path != BACKUP_PATH or backup_path.exists()
            or backup_path.is_symlink()):
        raise replay.RepairStopped("reviewed private backup path is unavailable")
    if db_path != replay.LIVE_CONTAINER_DB:
        raise replay.RepairStopped("apply requires the reviewed live container store")
    if core_db.schema_version(conn) != 64 or _target32(conn) != target32:
        raise replay.RepairStopped("fact schema or producer generation changed")
    if conn.execute("SELECT 1 FROM run_lock WHERE name='dreaming'").fetchone():
        raise replay.RepairStopped("dream lock is already held")
    state = replay._state(conn, target32)
    if state["state_sha256"] != expected_state:
        raise replay.RepairStopped("fact state differs from approved preview")
    if not _reviewed_partial_baseline(state):
        raise replay.RepairStopped("partial fact replay baseline changed")
    before = _inventory(conn, embedder)
    if (before != expected_inventory or before["gaps"] < 1
            or before["gaps"] > max_texts):
        raise replay.RepairStopped("fact vector inventory differs from approved preview")
    replay._check_authority(conn, replay._state(conn, target32)["session_ids"])
    embedder_identity_model, embedder_identity_dim = embedding_storage_identity(embedder)
    holder = runner._new_lease_token()
    if not runner._acquire_lock(conn, holder):
        raise replay.RepairStopped("dream lock acquisition lost")
    heartbeat = None
    fence = None
    try:
        heartbeat = runner._DreamLeaseHeartbeat(
            conn, holder, interval_seconds=runner._LOCK_REFRESH_INTERVAL_SECONDS,
        )
        fence = core_db.activate_transaction_lease_fence(
            conn, name="dreaming", holder=holder,
        )
        heartbeat.start()
        if (_current_state(conn, target32) != expected_state
                or _inventory(conn, embedder) != expected_inventory):
            raise replay.RepairStopped("fact vector state changed before backup")
        replay._backup_under_lease(conn, backup_path, expected_state, target32)
        with core_db.read_snapshot(backup_path) as snapshot:
            if _inventory(snapshot, embedder) != expected_inventory:
                raise replay.RepairStopped("online backup differs from reviewed vector state")
        batches = texts = cache_hits = 0
        while True:
            heartbeat.check()
            inventory = _inventory(conn, embedder)
            if inventory["gaps"] == 0:
                break
            if batches >= max_batches or texts + BATCH_SIZE > max_texts:
                raise replay.RepairStopped("local fact embedding ceiling reached")
            before_state = _current_state(conn, target32)
            pending = fetch_fact_embeddings(conn, embedder, batch_size=BATCH_SIZE)
            heartbeat.check()
            if pending is None or len(pending.fact_ids) > BATCH_SIZE:
                raise replay.RepairStopped("fact embedding scan made no bounded progress")
            if pending.model != embedder_identity_model or pending.dim != embedder_identity_dim:
                raise replay.RepairStopped("embedding producer changed during batch")
            with core_db.transaction(conn):
                if (_current_state(conn, target32) != before_state
                        or _target32(conn) != target32):
                    raise replay.RepairStopped("fact authority changed during local embedding")
                if set(facts.load_fact_source_manifests(conn, pending.fact_ids)) != set(pending.fact_ids):
                    raise replay.RepairStopped("fact source proof changed during local embedding")
                for fact_id, expected_hash in zip(pending.fact_ids, pending.text_hashes):
                    row = conn.execute(
                        "SELECT text FROM narrative_facts WHERE id=? "
                        "AND lifecycle_status='active' AND invalid_at IS NULL",
                        (fact_id,),
                    ).fetchone()
                    if row is None or embedding_text_hash(row["text"]) != expected_hash:
                        raise replay.RepairStopped("fact text changed during local embedding")
                if pending.scan_state_key is not None:
                    scan = conn.execute(
                        "SELECT value FROM schema_meta WHERE key=?",
                        (pending.scan_state_key,),
                    ).fetchone()
                    if (str(scan["value"]) if scan is not None else None) != pending.scan_start_value:
                        raise replay.RepairStopped("fact scan cursor changed during local embedding")
                written = persist_fact_embeddings(conn, pending)
                if written != len(pending.fact_ids):
                    raise replay.RepairStopped("fact vector publication was incomplete")
            batches += 1
            texts += len(pending.fact_ids)
            cache_hits += pending.cache_hits
        final_state = replay._state(conn, target32)
        if not _reviewed_partial_baseline(final_state):
            raise replay.RepairStopped("partial fact replay baseline changed during drain")
        replay._check_authority(conn, final_state["session_ids"])
        after = _inventory(conn, embedder)
        if after["gaps"] != 0 or after["active_proof_valid"] != before["active_proof_valid"]:
            raise replay.RepairStopped("fact vector drain postflight is incomplete")
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise replay.RepairStopped("SQLite quick_check failed after vector drain")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise replay.RepairStopped("foreign-key check failed after vector drain")
        heartbeat.check()
        return {"batches": batches, "texts": texts, "cache_hits": cache_hits,
                "before_gaps": before["gaps"], "after_gaps": after["gaps"],
                "mirror_rows": after["mirror_rows"],
                "mirror_sha256": after["mirror_sha256"],
                "backup": str(backup_path)}
    finally:
        try:
            if heartbeat is not None:
                heartbeat.stop()
        finally:
            try:
                if fence is not None:
                    core_db.deactivate_transaction_lease_fence(fence)
            finally:
                runner._release_lock(conn, holder)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--server-process-env", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-local-embedding-calls", action="store_true")
    parser.add_argument("--expect-state-sha256")
    parser.add_argument("--expect-target32")
    parser.add_argument("--expect-model-sha256")
    parser.add_argument("--expect-dim", type=int)
    parser.add_argument("--expect-mirror-sha256")
    parser.add_argument("--expect-mirror-rows", type=int)
    parser.add_argument("--expect-active-proof-valid", type=int)
    parser.add_argument("--expect-gaps", type=int)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-texts", type=int)
    args = parser.parse_args()
    if args.apply != args.allow_local_embedding_calls:
        parser.error("apply requires --allow-local-embedding-calls")
    if args.apply and any(value is None for value in (
        args.expect_state_sha256, args.expect_target32,
        args.expect_model_sha256, args.expect_dim,
        args.expect_mirror_sha256, args.expect_mirror_rows,
        args.expect_active_proof_valid, args.expect_gaps,
        args.backup, args.max_batches, args.max_texts,
    )):
        parser.error("apply requires exact state, vector and budget guards")
    db_path = args.db.resolve(strict=True)
    if replay._is_live_store(db_path) and db_path != replay.LIVE_CONTAINER_DB:
        parser.error("live operator must run inside Hermes1")
    if replay._is_live_store(db_path) and not args.server_process_env:
        parser.error("live operator requires running producer environment")
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    embedder = None
    try:
        if args.server_process_env:
            _load_running_embedding_env()
        embedder = replay._embedder_from_env()
        if not args.apply:
            with core_db.read_snapshot(db_path) as conn:
                target = _target32(conn)
                state = replay._state(conn, target)
                if state["schema"] != 64:
                    raise replay.RepairStopped("unexpected HyMem schema")
                inventory = _inventory(conn, embedder)
                output = {"state_sha256": state["state_sha256"],
                          "target32": target,
                          "outcomes": state["outcomes"],
                          "sessions": state["sessions"],
                          "stale": state["stale"],
                          "pending_fresh": state["pending_fresh"],
                          "quarantined": state["quarantined"],
                          **inventory,
                          "dream_lock_held": conn.execute(
                              "SELECT 1 FROM run_lock WHERE name='dreaming'"
                          ).fetchone() is not None}
        else:
            expected = {
                "model_sha256": args.expect_model_sha256,
                "dim": args.expect_dim,
                "mirror_sha256": args.expect_mirror_sha256,
                "mirror_rows": args.expect_mirror_rows,
                "active_proof_valid": args.expect_active_proof_valid,
                "gaps": args.expect_gaps,
            }
            conn = core_db.connect(db_path)
            try:
                output = apply(
                    conn, db_path, embedder,
                    target32=args.expect_target32,
                    expected_state=args.expect_state_sha256,
                    expected_inventory=expected,
                    backup_path=args.backup,
                    max_batches=args.max_batches, max_texts=args.max_texts,
                )
            finally:
                conn.close()
        print(json.dumps(output, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "stopped": type(exc).__name__,
            "reason": str(exc) if isinstance(exc, replay.RepairStopped)
            else "unexpected failure; no source or provider text emitted",
        }, sort_keys=True))
        return 2
    finally:
        if embedder is not None:
            try:
                embedder.close()
            except Exception:
                pass
        logging.disable(previous_logging_disable)


if __name__ == "__main__":
    raise SystemExit(main())
