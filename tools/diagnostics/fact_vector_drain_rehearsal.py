"""Disposable, network-free rehearsal of the guarded partial fact-vector drain.

The source is opened read-only and copied into a temporary 0700 directory.
Only the clone is retagged to a maintained deterministic embedding stub. Its
vectors therefore test transaction and mirror mechanics, not search quality.
The clone and both rehearsal backups are removed on exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import stat
import tempfile

from hymem.core import db as core_db
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.extraction.embeddings import MappedStubEmbeddingClient

if __package__:
    from . import fact_vector_drain as drain
else:
    import fact_vector_drain as drain


class InjectedFailure(RuntimeError):
    """Stop the second clone publication after its write, before commit."""


def _private_parent(path: Path) -> None:
    details = path.stat() if path.exists() else None
    if (not path.is_absolute() or not path.is_dir() or path.is_symlink()
            or path.resolve(strict=True) != path
            or details is None or stat.S_IMODE(details.st_mode) != 0o700
            or details.st_uid != os.geteuid() or details.st_gid != os.getegid()):
        raise drain.replay.RepairStopped("rehearsal parent must be owned by the runner and private 0700")


def _clone_read_only(source_path: Path, clone_path: Path) -> None:
    descriptor = os.open(clone_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)
    with core_db.read_snapshot(source_path) as source:
        clone = sqlite3.connect(str(clone_path), isolation_level=None)
        try:
            source.backup(clone)
        finally:
            clone.close()


def rehearse(source_path: Path, private_parent: Path, *,
             expected_state: str, expected_model_sha256: str,
             expected_dim: int, expected_mirror_rows: int,
             expected_mirror_sha256: str) -> dict:
    """Require a pinned source; fail one clone batch, resume, then check no-op."""
    if source_path.resolve(strict=True) != drain.replay.LIVE_CONTAINER_DB:
        raise drain.replay.RepairStopped("rehearsal source must be the reviewed container store")
    _private_parent(private_parent)
    target = drain.REVIEWED_CAP32_TARGET
    with core_db.read_snapshot(source_path) as source:
        if (drain._target32(source) != target
                or drain.replay._state(source, target)["state_sha256"] != expected_state):
            raise drain.replay.RepairStopped("rehearsal source state differs from preview")
        metadata = dict(source.execute(
            "SELECT key,value FROM schema_meta WHERE key IN ('vec_model','vec_dim')"
        ).fetchall())
        model = metadata.get("vec_model")
        dim = int(metadata.get("vec_dim", "0"))
        if (not isinstance(model, str)
                or hashlib.sha256(model.encode()).hexdigest() != expected_model_sha256
                or dim != expected_dim):
            raise drain.replay.RepairStopped("rehearsal source vector identity differs")
        if not core_db._load_vec_extension(source):
            raise drain.replay.RepairStopped("rehearsal source vector mirror is unavailable")
        if drain._mirror_digest(source) != (expected_mirror_rows, expected_mirror_sha256):
            raise drain.replay.RepairStopped("rehearsal source vector mirror differs")
    with tempfile.TemporaryDirectory(
        prefix="fact-vector-drain-rehearsal-", dir=private_parent,
    ) as private_dir:
        clone_path = Path(private_dir) / "clone.sqlite"
        _clone_read_only(source_path, clone_path)
        with core_db.read_snapshot(source_path) as source:
            if drain.replay._state(source, target)["state_sha256"] != expected_state:
                raise drain.replay.RepairStopped("source changed while rehearsal clone was made")
        conn = core_db.connect(clone_path)
        original_live = drain.replay.LIVE_CONTAINER_DB
        original_backup = drain.BACKUP_PATH
        original_persist = drain.persist_fact_embeddings
        try:
            if not core_db._load_vec_extension(conn):
                raise drain.replay.RepairStopped("rehearsal clone vector mirror is unavailable")
            if (drain._target32(conn) != target
                    or drain.replay._state(conn, target)["state_sha256"] != expected_state
                    or drain._mirror_digest(conn) != (expected_mirror_rows, expected_mirror_sha256)):
                raise drain.replay.RepairStopped("rehearsal clone differs from pinned source")
            stub = MappedStubEmbeddingClient(
                model="fact-vector-drain-rehearsal", dim=expected_dim,
                default=[1.0] + [0.0] * (expected_dim - 1), conn=conn,
            )
            stub_model, stub_dim = embedding_storage_identity(stub)
            with core_db.embedding_mutation(conn), core_db.transaction(conn):
                conn.execute(
                    "UPDATE schema_meta SET value=? WHERE key='vec_model'",
                    (stub_model,),
                )
                conn.execute(
                    "UPDATE narrative_fact_embeddings SET model=?",
                    (stub_model,),
                )
            before = drain._inventory(conn, stub)
            if before["gaps"] != 902 or before["mirror_rows"] != expected_mirror_rows:
                raise drain.replay.RepairStopped("rehearsal clone lacks reviewed 902-gap baseline")
            if stub_dim != expected_dim or not drain._reviewed_partial_baseline(
                    drain.replay._state(conn, target)):
                raise drain.replay.RepairStopped("rehearsal clone identity or baseline changed")

            drain.replay.LIVE_CONTAINER_DB = clone_path
            first_backup = Path(private_dir) / "before-injected-failure.sqlite"
            drain.BACKUP_PATH = first_backup
            publication_calls = 0
            nonempty_writes = 0
            first_batch_facts = 0
            pre_second_mirror = None
            pre_second_scan = None

            def fail_second_publication(batch_conn, pending):
                nonlocal publication_calls, nonempty_writes
                nonlocal first_batch_facts, pre_second_mirror, pre_second_scan
                publication_calls += 1
                if pending.fact_ids:
                    nonempty_writes += 1
                    if nonempty_writes == 1:
                        first_batch_facts = len(pending.fact_ids)
                    if nonempty_writes == 2:
                        pre_second_mirror = drain._mirror_digest(batch_conn)
                        pre_second_scan = (
                            pending.scan_state_key, pending.scan_start_value,
                        )
                persisted = original_persist(batch_conn, pending)
                if nonempty_writes == 2:
                    raise InjectedFailure("injected clone publication failure")
                return persisted

            drain.persist_fact_embeddings = fail_second_publication
            try:
                drain.apply(
                    conn, clone_path, stub, target32=target,
                    expected_state=drain._current_state(conn, target),
                    expected_inventory=before, backup_path=first_backup,
                    max_batches=32, max_texts=2048,
                )
            except InjectedFailure:
                pass
            else:
                raise drain.replay.RepairStopped("injected clone failure was not reached")
            finally:
                drain.persist_fact_embeddings = original_persist
            interrupted = drain._inventory(conn, stub)
            scan_key, scan_value = pre_second_scan or (None, None)
            actual_scan = conn.execute(
                "SELECT value FROM schema_meta WHERE key=?", (scan_key,),
            ).fetchone() if scan_key is not None else None
            rollback_checks = {
                "publication_calls": publication_calls,
                "nonempty_publications": nonempty_writes,
                "scan_only_publications": publication_calls - nonempty_writes,
                "backup_exists": first_backup.exists(),
                "first_batch_facts": first_batch_facts,
                "before_gaps": before["gaps"],
                "after_gaps": interrupted["gaps"],
                "gap_delta_matches_first_batch": (
                    interrupted["gaps"] == before["gaps"] - first_batch_facts
                ),
                "mirror_matches_pre_second": (
                    drain._mirror_digest(conn) == pre_second_mirror
                ),
                "scan_cursor_matches_pre_second": (
                    scan_key is not None
                    and (str(actual_scan["value"]) if actual_scan is not None else None)
                    == scan_value
                ),
            }
            if (nonempty_writes != 2 or not rollback_checks["backup_exists"]
                    or first_batch_facts < 1 or first_batch_facts > drain.BATCH_SIZE
                    or not rollback_checks["gap_delta_matches_first_batch"]
                    or not rollback_checks["mirror_matches_pre_second"]
                    or not rollback_checks["scan_cursor_matches_pre_second"]):
                raise drain.replay.RepairStopped(
                    "injected clone rollback checks failed "
                    + json.dumps(rollback_checks, sort_keys=True)
                )
            if conn.execute("SELECT 1 FROM run_lock WHERE name='dreaming'").fetchone():
                raise drain.replay.RepairStopped("interrupted clone retained the dream lock")

            second_backup = Path(private_dir) / "before-resume.sqlite"
            drain.BACKUP_PATH = second_backup
            result = drain.apply(
                conn, clone_path, stub, target32=target,
                expected_state=drain._current_state(conn, target),
                expected_inventory=interrupted, backup_path=second_backup,
                max_batches=32, max_texts=2048,
            )
            complete = drain._inventory(conn, stub)
            calls_before_noop = len(stub.calls)
            noop_backup = Path(private_dir) / "unused-noop.sqlite"
            drain.BACKUP_PATH = noop_backup
            try:
                drain.apply(
                    conn, clone_path, stub, target32=target,
                    expected_state=drain._current_state(conn, target),
                    expected_inventory=complete,
                    backup_path=noop_backup,
                    max_batches=32, max_texts=2048,
                )
            except drain.replay.RepairStopped as exc:
                if str(exc) != "fact vector inventory differs from approved preview":
                    raise
            else:
                raise drain.replay.RepairStopped("completed clone unexpectedly admitted another drain")
            if (complete["gaps"] != 0 or len(stub.calls) != calls_before_noop
                    or any(stub.transaction_states)):
                raise drain.replay.RepairStopped("clone drain was incomplete or embedded under a write transaction")
            return {
                "source_state_sha256": expected_state,
                "initial_gaps": before["gaps"],
                "gaps_after_injection": interrupted["gaps"],
                "pre_injection_scan_only_batches": publication_calls - nonempty_writes,
                "pre_injection_publication_calls": publication_calls,
                "final_gaps": complete["gaps"],
                "resume_batches": result["batches"],
                "resume_texts": result["texts"],
                "stub_calls": len(stub.calls),
                "mirror_rows": complete["mirror_rows"],
                "mirror_sha256": complete["mirror_sha256"],
                "private_clone_removed_on_exit": True,
            }
        finally:
            drain.persist_fact_embeddings = original_persist
            drain.BACKUP_PATH = original_backup
            drain.replay.LIVE_CONTAINER_DB = original_live
            conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private-parent", type=Path, required=True)
    parser.add_argument("--expect-state-sha256", required=True)
    parser.add_argument("--expect-model-sha256", required=True)
    parser.add_argument("--expect-dim", type=int, required=True)
    parser.add_argument("--expect-mirror-rows", type=int, required=True)
    parser.add_argument("--expect-mirror-sha256", required=True)
    parser.add_argument("--rehearse-disposable-clone", action="store_true")
    args = parser.parse_args()
    if not args.rehearse_disposable_clone:
        parser.error("explicit disposable clone rehearsal flag is required")
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        result = rehearse(
            args.source, args.private_parent,
            expected_state=args.expect_state_sha256,
            expected_model_sha256=args.expect_model_sha256,
            expected_dim=args.expect_dim,
            expected_mirror_rows=args.expect_mirror_rows,
            expected_mirror_sha256=args.expect_mirror_sha256,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "stopped": type(exc).__name__,
            "reason": str(exc) if isinstance(exc, drain.replay.RepairStopped)
            else "unexpected clone rehearsal failure; no source text emitted",
        }, sort_keys=True))
        return 2
    finally:
        logging.disable(previous_logging_disable)


if __name__ == "__main__":
    raise SystemExit(main())
