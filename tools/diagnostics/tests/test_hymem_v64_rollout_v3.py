"""Local SQLite WAL checkpoint regression; no production or Docker actions."""
import hashlib
import sqlite3

import pytest

from tools.diagnostics import hymem_v64_rollout_v3 as adapter
from tools.diagnostics.tests.test_hymem_v64_rollout_v2 import fixture, execute


def wal_database(tmp_path):
    path = tmp_path / "wal.sqlite"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE durable(value TEXT)")
    writer.execute("INSERT INTO durable VALUES('first')")
    writer.commit()
    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM durable").fetchall()
    writer.execute("INSERT INTO durable VALUES('second')")
    writer.commit()
    writer.close()
    return path, reader


def test_busy_checkpoint_refuses_and_preserves_rows(tmp_path):
    path, reader = wal_database(tmp_path)
    with pytest.raises(RuntimeError, match="checkpoint_busy_or_unverified"):
        adapter.checkpoint(path)
    assert path.with_name(path.name + "-wal").stat().st_size > 0
    reader.rollback()
    assert path.with_name(path.name + "-wal").stat().st_size > 0
    adapter.checkpoint(path)
    assert path.with_name(path.name + "-wal").stat().st_size == 0
    reader.close()
    c = sqlite3.connect(path)
    assert c.execute("SELECT * FROM durable").fetchall() == [('first',), ('second',)]
    c.close()
    adapter.checkpoint(path)
    assert not path.with_name(path.name + "-wal").exists() or path.with_name(path.name + "-wal").stat().st_size == 0


def test_snapshot_includes_schema_vec_and_shadow_without_migration(tmp_path, monkeypatch, capsys):
    path, tables, calls = fixture(tmp_path, monkeypatch)
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    first = execute(path, adapter.SNAPSHOT)
    second = execute(path, adapter.SNAPSHOT)
    assert first['rows'] == second['rows']
    assert set(first['rows']) == tables | {'schema_meta', 'sqlite_sequence'}
    assert first['rows']['schema_meta']['count'] == 1
    assert len(calls) == 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    capsys.readouterr()


def test_original_guards_and_stop_sequence_remain():
    assert adapter.original.Rollout.stopped_guard.__globals__['MIGRATION'] == adapter.MIGRATION
    assert 'stopped_wal_requires_explicit_checkpoint_review' in adapter.original.Rollout.stopped_guard.__code__.co_consts
    assert adapter.stop_source.index('checked = self.offline') < adapter.stop_source.index('verified_checkpoint(self, backup)')
    assert adapter.stop_source.index('verify_files(source_backup, self.old)') < adapter.stop_source.index('verified_checkpoint(self, backup)')
    assert adapter.stop_source.index('verified_checkpoint(self, backup)') < adapter.stop_source.index('"production_db_sha256": sha(DB)')


def test_pins_and_symlink_refusal(tmp_path):
    drift = tmp_path / 'v2.py'
    drift.write_bytes(b'drift')
    with pytest.raises(RuntimeError, match='v2_helper_pin_mismatch'):
        adapter.load_v2(drift)
    path, reader = wal_database(tmp_path)
    alias = tmp_path / 'alias.sqlite'
    alias.symlink_to(path)
    with pytest.raises(RuntimeError):
        adapter.checkpoint(alias)
    reader.close()
