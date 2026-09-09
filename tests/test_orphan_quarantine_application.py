"""Synthetic-only guarded production application; never a deployed database."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import pytest

from hymem.core import db
from hymem.deadline import DeadlineExceeded, MonotonicDeadline, use_deadline
from tests.test_orphan_quarantine_rehearsal import source, _logical, _reference
from tools.deployment import apply_orphan_quarantine as application


@pytest.fixture
def operation(source):
    path, _ = source
    backups = path.parent / "backups"
    backups.mkdir(mode=0o700)
    reference = _reference(application.reviewed, path)
    return path, backups / "application", reference


def _apply(operation, **kwargs):
    return application.apply_quarantine(*operation, apply=True, expected_schema=59, **kwargs)


def _snapshot(path, *, omit=False):
    helper = application.reviewed
    conn = helper._readonly(path)
    try:
        helper._load_vectors(conn)
        return helper._schema(conn), helper._snapshot(
            conn, MonotonicDeadline.after(30), omit_chunk="orphan-id" if omit else None,
        )
    finally:
        conn.close()


def test_success_changes_live_store_in_place_and_keeps_exact_durable_recovery(operation):
    path, destination, reference = operation
    before, expected = _snapshot(path), _snapshot(path, omit=True)
    inode = path.stat().st_ino
    result = _apply(operation)
    assert result["status"] == "verified_applied"
    assert result["committed"] and result["source_writes"] == 1
    assert result["locked_exact_delta_verified"] and result["postcommit_snapshot_verified"]
    assert result["quarantined_entity_mentions"] == 4
    assert result["archive_only_runtime_reinsert_tested"] is False
    assert path.stat().st_ino == inode  # Never replace source with an older clone.
    assert _snapshot(path) == expected
    for name in ("baseline.sqlite", "baseline-restored.sqlite"):
        copy = destination / name
        assert _logical(application.reviewed, copy) == before
        assert _reference(application.reviewed, copy) == reference
        conn = application.reviewed._readonly(copy)
        assert len(conn.execute("PRAGMA foreign_key_check").fetchall()) == 1
        conn.close()
    assert destination.stat().st_mode & 0o777 == 0o700
    for child in destination.glob("*.sqlite"):
        assert child.stat().st_mode & 0o777 == 0o600
    archive = application.reviewed._readonly(destination / "quarantine.sqlite")
    try:
        assert archive.execute("SELECT rowid FROM chunks").fetchone()[0] == 1058
        assert tuple(archive.execute("SELECT rowid,typeof(vector_json) FROM chunk_embeddings").fetchone()) == (1059, "blob")
        assert archive.execute("SELECT count(*) FROM entity_mentions").fetchone()[0] == 4
    finally:
        archive.close()


@pytest.mark.parametrize("fault", ["archive", "restore", "precommit", "cancel"])
def test_failures_rollback_source_without_unrelated_loss(operation, monkeypatch, fault):
    path, destination, _ = operation
    before = _snapshot(path)
    if fault == "archive":
        def fail(*args):
            raise application.Refused("archive_roundtrip_failed")
        monkeypatch.setattr(application.reviewed, "_archive", fail)
    elif fault == "restore":
        original = application._verify_recovery_copy
        def fail(*args):
            original(*args)
            if args[0].name == "baseline-restored.sqlite":
                raise application.Refused("baseline_restore_failed")
        monkeypatch.setattr(application, "_verify_recovery_copy", fail)
    else:
        original = application.reviewed._integrity
        def fail(conn):
            original(conn)
            if conn.in_transaction:
                if fault == "cancel":
                    raise KeyboardInterrupt()
                raise application.Refused("precommit_failed")
        monkeypatch.setattr(application.reviewed, "_integrity", fail)
    with pytest.raises((application.Refused, KeyboardInterrupt)):
        _apply(operation)
    assert _snapshot(path) == before
    assert destination.exists()


def test_backup_uses_separate_readonly_connection_and_verifies_it_against_locked_source(operation, monkeypatch):
    path, destination, _ = operation
    original = application.reviewed._backup
    calls = []
    def backup(reader, target, deadline):
        calls.append(target.name)
        if target.name == "baseline.sqlite":
            with pytest.raises(sqlite3.OperationalError):
                reader.execute("UPDATE sessions SET id=id")
            contender = db.connect(path)
            try:
                contender.execute("PRAGMA busy_timeout=20")
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    contender.execute("BEGIN IMMEDIATE")
            finally:
                contender.close()
        original(reader, target, deadline)
    monkeypatch.setattr(application.reviewed, "_backup", backup)
    assert _apply(operation)["status"] == "verified_applied"
    assert calls == ["baseline.sqlite", "baseline-restored.sqlite"]


def test_corrupt_baseline_is_never_authority_to_delete(operation, monkeypatch):
    path, _, _ = operation
    before = _snapshot(path)
    original = application.reviewed._backup
    def backup(reader, target, deadline):
        original(reader, target, deadline)
        if target.name == "baseline.sqlite":
            conn = db.connect(target)
            conn.execute("UPDATE chunks SET text='changed control' WHERE rowid=7")
            conn.close()
    monkeypatch.setattr(application.reviewed, "_backup", backup)
    with pytest.raises(application.Refused, match="baseline_restore_failed"):
        _apply(operation)
    assert _snapshot(path) == before


@pytest.mark.parametrize("drift", ["dependency", "fingerprint", "trigger", "missing-fts"])
def test_changed_or_unreviewed_source_refuses_without_deletion(operation, drift):
    path, _, _ = operation
    conn = db.connect(path)
    if drift == "dependency":
        conn.execute("INSERT INTO processed_chunks(chunk_id,prompt_version) VALUES ('orphan-id','new-dependency')")
    elif drift == "fingerprint":
        conn.execute("UPDATE chunks SET text='changed orphan' WHERE rowid=1058")
    elif drift == "trigger":
        conn.execute("CREATE TRIGGER unexpected_delta AFTER DELETE ON chunks BEGIN DELETE FROM chunk_embeddings WHERE chunk_id='control'; END")
    else:
        conn.execute("DROP TRIGGER chunks_fts_delete")
    conn.close()
    before = _snapshot(path)
    with pytest.raises(application.Refused):
        _apply(operation)
    assert _snapshot(path) == before


def test_normal_guard_and_udf_authority_remain_active(operation, monkeypatch):
    path, _, _ = operation
    conn = db.connect(path)
    conn.execute("CREATE TRIGGER active_guard BEFORE DELETE ON chunks BEGIN SELECT RAISE(ABORT,'guard-active'); END")
    conn.close()
    checks = []
    original = application._guard_state
    def guard(conn):
        original(conn)
        checks.append(True)
    monkeypatch.setattr(application, "_guard_state", guard)
    before = _snapshot(path)
    with pytest.raises(sqlite3.IntegrityError, match="guard-active"):
        _apply(operation)
    assert checks == [True, True]
    assert _snapshot(path) == before


def test_busy_writer_has_bounded_wait_and_no_source_delta(operation):
    path, _, _ = operation
    holder = db.connect(path)
    before = _snapshot(path)
    holder.execute("BEGIN IMMEDIATE")
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            _apply(operation, lock_timeout_seconds=0.03, timeout_seconds=1)
        assert time.monotonic() - started < 1
        assert holder.in_transaction
    finally:
        holder.rollback()
        holder.close()
    assert _snapshot(path) == before


def test_intervening_postcommit_writer_is_preserved_and_reported(operation, monkeypatch):
    path, _, _ = operation
    original = application._postcommit
    def verify(conn, *args):
        writer = db.connect(path)
        writer.execute("INSERT INTO sessions(id) VALUES ('concurrent-after-commit')")
        writer.close()
        return original(conn, *args)
    monkeypatch.setattr(application, "_postcommit", verify)
    result = _apply(operation)
    assert result["status"] == "applied_concurrent_write"
    assert result["committed"] and result["locked_exact_delta_verified"]
    assert result["postcommit_snapshot_verified"] is False
    conn = application.reviewed._readonly(path)
    try:
        assert conn.execute("SELECT 1 FROM sessions WHERE id='concurrent-after-commit'").fetchone()
        assert not conn.execute("SELECT 1 FROM chunks WHERE rowid=1058").fetchone()
    finally:
        conn.close()


def test_postcommit_verification_failure_never_claims_rollback(operation, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic-private-error")
    monkeypatch.setattr(application, "_postcommit", fail)
    result = _apply(operation)
    assert result["status"] == "applied_verification_unavailable"
    assert result["committed"] is True and result["source_writes"] == 1


def test_existing_directory_and_wrong_authority_are_refused(operation):
    path, destination, reference = operation
    before = _snapshot(path)
    with pytest.raises(application.Refused, match="explicit_apply_required"):
        application.apply_quarantine(path, destination, reference, expected_schema=59)
    with pytest.raises(application.Refused, match="reviewed_schema_required"):
        application.apply_quarantine(path, destination, reference, apply=True, expected_schema=58)
    destination.mkdir()
    with pytest.raises(FileExistsError):
        _apply(operation)
    assert _snapshot(path) == before


def test_preexisting_context_and_transaction_are_not_consumed(operation, monkeypatch):
    path, destination, _ = operation
    holder = db.connect(path)
    try:
        with db.embedding_mutation(holder):
            with pytest.raises(application.Refused, match="preexisting_execution_context"):
                _apply(operation)
        with use_deadline(MonotonicDeadline.after(30)):
            with pytest.raises(application.Refused, match="preexisting_execution_context"):
                _apply(operation)
        assert not destination.exists()
        holder.execute("BEGIN IMMEDIATE")
        monkeypatch.setattr(application, "_open_existing", lambda *args: holder)
        with pytest.raises(application.Refused, match="preexisting_transaction"):
            _apply(operation)
        assert holder.in_transaction
    finally:
        holder.rollback()
        holder.close()


def test_limited_connection_never_creates_missing_source(tmp_path):
    missing = tmp_path / "not-a-database.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        application._open_existing(missing, 0.03)
    assert not missing.exists()


def test_disappearing_source_is_not_recreated(operation, monkeypatch):
    path, _, _ = operation
    original = application._open_existing
    moved = path.with_name("retained-source.sqlite")
    before = _snapshot(path)
    def open_existing(*args):
        path.rename(moved)
        return original(*args)
    monkeypatch.setattr(application, "_open_existing", open_existing)
    with pytest.raises(sqlite3.OperationalError):
        _apply(operation)
    assert not path.exists()
    assert _snapshot(moved) == before


def test_limited_connection_denies_every_mutation_grant(operation):
    path, _, _ = operation
    conn = application._open_existing(path, 0.03)
    try:
        application._guard_state(conn)
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30
        # Even a caller setting every core authority context cannot turn these
        # deny-only SQL functions into grants on the maintenance connection.
        variables = (db._EVIDENCE_MUTATION_KEYS, db._EVIDENCE_HISTORY_KEYS,
                     db._EVIDENCE_DESTRUCTIVE_KEYS, db._EMBEDDING_MUTATION_KEYS,
                     db._PHASE1_GENERATION_PRUNE_KEYS)
        for name, variable in zip(application._DENIED_MUTATION_FUNCTIONS, variables):
            token = variable.set(frozenset({id(conn)}))
            try:
                assert conn.execute(f"SELECT {name}()").fetchone()[0] == 0
            finally:
                variable.reset(token)
    finally:
        conn.close()


def test_unreviewed_trigger_udf_fails_closed(operation):
    path, _, _ = operation
    conn = db.connect(path)
    conn.execute("CREATE TRIGGER unknown_requirement BEFORE DELETE ON chunks BEGIN SELECT unreviewed_authority(); END")
    conn.close()
    before = _snapshot(path)
    with pytest.raises(sqlite3.OperationalError, match="no such function"):
        _apply(operation)
    assert _snapshot(path) == before


@pytest.mark.parametrize("failed_file", ["baseline.sqlite", "baseline-restored.sqlite", "quarantine.sqlite", "application"])
def test_fsync_failure_prevents_deletion_and_closes_source(operation, monkeypatch, failed_file):
    path, destination, _ = operation
    before = _snapshot(path)
    real_open, real_fsync = application._open_existing, os.fsync
    opened = []
    def open_existing(*args):
        conn = real_open(*args)
        opened.append(conn)
        return conn
    def sync(fd):
        target = destination if failed_file == "application" else destination / failed_file
        if target.exists() and os.fstat(fd).st_ino == target.stat().st_ino:
            raise OSError("synthetic durability failure")
        return real_fsync(fd)
    monkeypatch.setattr(application, "_open_existing", open_existing)
    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(OSError, match="durability failure"):
        _apply(operation)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    assert _snapshot(path) == before


def test_deadline_failure_before_delete_rolls_back_and_closes_source(operation, monkeypatch):
    path, _, _ = operation
    before = _snapshot(path)
    real_open = application._open_existing
    opened = []
    def open_existing(*args):
        conn = real_open(*args)
        opened.append(conn)
        return conn
    def expire(*args):
        raise DeadlineExceeded("synthetic deadline")
    monkeypatch.setattr(application, "_open_existing", open_existing)
    monkeypatch.setattr(application.reviewed, "_archive", expire)
    with pytest.raises(DeadlineExceeded):
        _apply(operation)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    assert _snapshot(path) == before


def test_cli_requires_apply_and_keeps_private_arguments_out_of_output(operation):
    path, destination, reference = operation
    script = Path(application.__file__)
    args = [sys.executable, str(script), "--source", str(path), "--artifact-dir", str(destination),
            "--reference-sha256", reference, "--expected-schema", "59"]
    refused = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert refused.returncode == 1
    assert json.loads(refused.stdout)["committed"] is False
    abbreviated = subprocess.run([*args, "--ap"], capture_output=True, text=True, timeout=10)
    assert abbreviated.returncode == 1
    assert json.loads(abbreviated.stdout)["committed"] is False
    success = subprocess.run([*args, "--apply"], capture_output=True, text=True, timeout=30)
    assert success.returncode == 0
    assert json.loads(success.stdout)["status"] == "verified_applied"
    for output in (refused, abbreviated, success):
        assert output.stderr == ""
        assert not any(value in output.stdout for value in (str(path), reference, "orphan-id", "missing-parent", "private orphan payload"))


def test_standalone_cli_loads_staged_sibling_despite_older_tools_package(operation, tmp_path):
    path, destination, reference = operation
    stage, shadow = tmp_path / "stage", tmp_path / "older-installation"
    stage.mkdir()
    package = shadow / "tools" / "deployment"
    package.mkdir(parents=True)
    (shadow / "tools" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "redact.py").write_text("# Older published tools package, no rehearsal helper.\n")
    script = stage / Path(application.__file__).name
    shutil.copyfile(application.__file__, script)
    shutil.copyfile(application.reviewed.__file__, stage / "rehearse_orphan_quarantine.py")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(shadow), str(Path(application.__file__).resolve().parents[2])))
    result = subprocess.run(
        [sys.executable, str(script), "--source", str(path), "--artifact-dir", str(destination),
         "--reference-sha256", reference, "--expected-schema", "59", "--apply"],
        cwd=stage, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "verified_applied"
