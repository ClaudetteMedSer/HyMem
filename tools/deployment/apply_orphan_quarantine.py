#!/usr/bin/env python3
"""Guarded in-place application of the separately reviewed rowid-1058 repair.

Requires explicit --apply, exact whole-row fingerprint and reviewed schema59.
The source must already exist in WAL mode. The supplied artifact directory
must be a NEW direct child of its existing sibling `backups` directory.
No initialize/migration, providers, fabricated source rows or database swap.
Private baseline, native archive and verified full restore are retained on
success AND failure. After commit, never automatically restore a stale backup.
Limits and dependency/delta semantics come from the unchanged clone rehearsal.
"""
from __future__ import annotations

import json
import importlib.util
import logging
import math
import os
from pathlib import Path
import sqlite3
import stat
import sys

from hymem.core import db
from hymem.deadline import DeadlineExceeded, MonotonicDeadline, current_deadline, use_deadline

# Deployment stages this reviewed file beside the application helper. Never
# resolve it through an ambient `tools` package from an older installation.
_reviewed_spec = importlib.util.spec_from_file_location(
    "_hymem_reviewed_orphan_quarantine",
    Path(__file__).resolve().with_name("rehearse_orphan_quarantine.py"),
)
if _reviewed_spec is None or _reviewed_spec.loader is None:
    raise RuntimeError("reviewed_helper_unavailable")
reviewed = importlib.util.module_from_spec(_reviewed_spec)
_reviewed_spec.loader.exec_module(reviewed)

Refused = reviewed.Refused
REVIEWED_SCHEMA = 59
_DENIED_MUTATION_FUNCTIONS = (
    "hymem_evidence_mutation_authorized", "hymem_evidence_history_authorized",
    "hymem_evidence_destructive_authorized", "hymem_embedding_mutation_authorized",
    "hymem_phase1_generation_prune_authorized",
)


class CommitOutcomeUnknown(RuntimeError):
    """A failure crossed the commit boundary; inspect, never auto-restore."""


def _clean_context(conn=None):
    if current_deadline() is not None or any(variable.get() for variable in (
        db._EVIDENCE_MUTATION_KEYS, db._EVIDENCE_HISTORY_KEYS,
        db._EVIDENCE_DESTRUCTIVE_KEYS, db._EMBEDDING_MUTATION_KEYS,
        db._PHASE1_GENERATION_PRUNE_KEYS, db._TRANSACTION_LEASE_FENCES,
    )):
        raise Refused("preexisting_execution_context")
    if conn is not None and conn.in_transaction:
        raise Refused("preexisting_transaction")


def _file_identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise Refused("existing_regular_source_required")
    return info.st_dev, info.st_ino


def _sync_file(path):
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path):
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _open_existing(path, timeout):
    """Limited, non-creating maintenance connection, NOT full core connect.

    Real core read predicates keep stock trigger/proof checks meaningful.
    Mutation predicates are deny-only even inside a core authority context;
    the reviewed orphan DELETE needs no evidence/history/embedding grant.
    An unreviewed trigger requiring another UDF fails closed. Never initialize,
    migrate, or change journal mode on this source connection.
    """
    conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True,
                           isolation_level=None, timeout=timeout)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={int(1000 * timeout)}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA secure_delete=ON")
        conn.execute("PRAGMA synchronous=FULL")
        if conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            raise Refused("existing_wal_source_required")
        db.register_read_authority_functions(conn)
        for name in _DENIED_MUTATION_FUNCTIONS:
            conn.create_function(name, 0, lambda: 0, deterministic=False)
        return conn
    except BaseException:
        conn.close()
        raise


def _guard_state(conn):
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise Refused("foreign_key_enforcement_required")
    if conn.execute("PRAGMA secure_delete").fetchone()[0] != 1:
        raise Refused("secure_delete_required")
    if conn.execute("PRAGMA synchronous").fetchone()[0] != 2:
        raise Refused("full_sync_required")
    if any(conn.execute(f"SELECT {name}()").fetchone()[0] != 0
           for name in _DENIED_MUTATION_FUNCTIONS):
        raise Refused("unexpected_mutation_authority")


def _verify_recovery_copy(path, schema, snapshot, reference, deadline):
    # Core UDF/FK wiring on an OWNED copy, never initialize or repair it.
    conn = db.connect(path)
    try:
        reviewed._load_vectors(conn)
        conn.set_progress_handler(lambda: int(deadline.expired), 10000)
        reviewed._preflight(conn, reference, deadline)
        if reviewed._schema(conn) != schema or reviewed._snapshot(conn, deadline) != snapshot:
            raise Refused("baseline_restore_failed")
        reviewed._integrity(conn)
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()
    _sync_file(path)


def _postcommit(conn, expected_schema, expected_snapshot, data_version, deadline):
    """A concurrent writer is not evidence that our atomic delta was wrong."""
    try:
        conn.execute("BEGIN")
        schema = reviewed._schema(conn)  # establish one committed read snapshot
        if conn.execute("PRAGMA data_version").fetchone()[0] != data_version:
            return "applied_concurrent_write", False
        if schema != expected_schema or reviewed._snapshot(conn, deadline) != expected_snapshot:
            return "applied_verification_failed", False
        return "verified_applied", True
    except (Exception, KeyboardInterrupt, DeadlineExceeded):
        return "applied_verification_unavailable", False
    finally:
        if conn.in_transaction:
            conn.rollback()  # read transaction only; never restore the source


def apply_quarantine(source_path, artifact_dir, reference_sha256, *, apply=False,
                     expected_schema=None, timeout_seconds=120, lock_timeout_seconds=5):
    _clean_context()
    if apply is not True:
        raise Refused("explicit_apply_required")
    if type(expected_schema) is not int or expected_schema != REVIEWED_SCHEMA or db.EXPECTED_SCHEMA_VERSION != REVIEWED_SCHEMA:
        raise Refused("reviewed_schema_required")
    if not isinstance(reference_sha256, str) or reviewed._REFERENCE.fullmatch(reference_sha256) is None:
        raise Refused("invalid_reference")
    for value, ceiling in ((timeout_seconds, 3600), (lock_timeout_seconds, 30)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
            raise Refused("invalid_timeout")
    deadline = MonotonicDeadline.after(timeout_seconds)
    source_path, destination = Path(source_path), Path(artifact_dir)
    if not source_path.is_absolute() or not destination.is_absolute():
        raise Refused("absolute_paths_required")
    identity = _file_identity(source_path)
    source_path = source_path.resolve(strict=True)
    backup_root = source_path.parent / "backups"
    if backup_root.is_symlink() or not backup_root.is_dir():
        raise Refused("existing_backup_root_required")
    if destination.parent.resolve(strict=True) != backup_root.resolve(strict=True):
        raise Refused("private_backup_child_required")
    probe = reviewed._readonly(source_path)
    try:
        probe.execute(f"PRAGMA busy_timeout={int(1000 * min(lock_timeout_seconds, deadline.remaining()))}")
        if probe.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            raise Refused("existing_wal_source_required")
        if db.schema_version(probe) != expected_schema:
            raise Refused("reviewed_schema_required")
    finally:
        probe.close()
    os.mkdir(destination, 0o700)
    _sync_directory(backup_root)
    baseline, archive, restored = (destination / name for name in (
        "baseline.sqlite", "quarantine.sqlite", "baseline-restored.sqlite",
    ))
    conn = _open_existing(source_path, min(lock_timeout_seconds, deadline.remaining()))
    try:
        _clean_context(conn)
    except BaseException:
        # Do not rollback/close a caller-owned transaction returned by an
        # unexpected connection factory; the operation has not taken ownership.
        if not conn.in_transaction:
            conn.close()
        raise
    commit_attempted = False
    try:
        conn.execute(f"PRAGMA busy_timeout={int(1000 * min(lock_timeout_seconds, deadline.remaining()))}")
        reviewed._load_vectors(conn)
        conn.set_progress_handler(lambda: int(deadline.expired), 10000)
        with use_deadline(deadline):
            try:
                with db.transaction(conn):
                    _guard_state(conn)
                    if _file_identity(source_path) != identity:
                        raise Refused("source_file_changed")
                    if conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute("PRAGMA page_size").fetchone()[0] > reviewed.MAX_DATABASE_BYTES:
                        raise Refused("database_size_limit")
                    target = reviewed._preflight(conn, reference_sha256, deadline)
                    schema = reviewed._schema(conn)
                    before = reviewed._snapshot(conn, deadline)
                    expected = reviewed._snapshot(conn, deadline, omit_chunk=target["id"])
                    version = conn.execute("PRAGMA data_version").fetchone()[0]
                    captured = reviewed._capture(conn, target)
                    # NEVER backup from the writing connection: SQLite would
                    # wait on its own uncommitted transaction. This separate
                    # read-only connection sees the same committed state while
                    # BEGIN IMMEDIATE prevents any other writer from drifting.
                    reader = reviewed._readonly(source_path)
                    try:
                        reader.execute("BEGIN")
                        reviewed._backup(reader, baseline, deadline)
                    finally:
                        reader.close()
                    _verify_recovery_copy(baseline, schema, before, reference_sha256, deadline)
                    reader = reviewed._readonly(baseline)
                    try:
                        reviewed._backup(reader, restored, deadline)
                    finally:
                        reader.close()
                    _verify_recovery_copy(restored, schema, before, reference_sha256, deadline)
                    reviewed._archive(archive, captured)  # native readback + fsync
                    _sync_directory(destination)
                    deadline.check()
                    _guard_state(conn)
                    reviewed._preflight(conn, reference_sha256, deadline)
                    if (reviewed._schema(conn) != schema
                            or reviewed._snapshot(conn, deadline) != before
                            or reviewed._capture(conn, target) != captured
                            or _file_identity(source_path) != identity):
                        raise Refused("source_changed_before_delete")
                    if "vec_chunks" in captured:
                        conn.execute("DELETE FROM vec_chunks WHERE rowid=?", (reviewed.TARGET_ROWID,))
                    deleted = conn.execute("DELETE FROM chunks WHERE rowid=? AND id=?", (reviewed.TARGET_ROWID, target["id"]))
                    if deleted.rowcount != 1:
                        raise Refused("target_delete_count_changed")
                    if conn.execute("PRAGMA foreign_key_check").fetchall():
                        raise Refused("remaining_foreign_key_faults")
                    reviewed._integrity(conn)
                    if reviewed._schema(conn) != schema or reviewed._snapshot(conn, deadline) != expected:
                        raise Refused("unexpected_logical_delta")
                    deadline.check()
                    commit_attempted = True
            except BaseException as exc:
                if commit_attempted:
                    raise CommitOutcomeUnknown("commit_outcome_unknown") from None
                raise
            try:
                status, postcommit_verified = _postcommit(conn, schema, expected, version, deadline)
            except (Exception, KeyboardInterrupt, DeadlineExceeded):
                status, postcommit_verified = "applied_verification_unavailable", False
            return {
                "status": status, "committed": True, "source_writes": 1,
                "quarantined_chunks": 1, "quarantined_durable_vectors": 1,
                "quarantined_entity_mentions": len(captured["entity_mentions"][1]),
                "quarantined_shadow_vectors": len(captured.get("vec_chunks", ([], []))[1]),
                "archive_native_roundtrip_verified": True,
                "full_baseline_restore_verified": True,
                "archive_only_runtime_reinsert_tested": False,
                "locked_exact_delta_verified": True,
                "postcommit_snapshot_verified": postcommit_verified,
                "remaining_foreign_key_faults_at_commit": 0,
            }
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def main(argv=None):
    logging.disable(logging.CRITICAL)
    parser = reviewed._SafeParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--expected-schema", required=True, type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--lock-timeout-seconds", type=float, default=5)
    try:
        args = parser.parse_args(argv)
        result = apply_quarantine(args.source, args.artifact_dir, args.reference_sha256,
                                  apply=args.apply, expected_schema=args.expected_schema,
                                  timeout_seconds=args.timeout_seconds,
                                  lock_timeout_seconds=args.lock_timeout_seconds)
    except CommitOutcomeUnknown:
        result = {"status": "commit_outcome_unknown", "committed": None, "source_writes": None}
    except Refused:
        result = {"status": "refused", "reason": "guard_refused", "committed": False, "source_writes": 0}
    except (KeyboardInterrupt, DeadlineExceeded):
        result = {"status": "cancelled", "reason": "cancelled_or_deadline", "committed": False, "source_writes": 0}
    except Exception:
        result = {"status": "application_outcome_unknown", "reason": "application_failed", "committed": None, "source_writes": None}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "verified_applied" else 1


if __name__ == "__main__":
    sys.exit(main())
