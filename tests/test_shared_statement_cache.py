"""Shared SQLite reads must not reuse Python's unsafe cached statements."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from hymem.core import db


@pytest.mark.parametrize(
    ("version", "cache_size"),
    [((3, 11, 2), 128), ((3, 11, 99), 128), ((3, 12, 0), 0),
     ((3, 13, 5), 0), ((3, 14, 0), 0), ((3, 15, 0), 0), ((4, 0, 0), 0)],
)
def test_shared_constructor_selects_cache_by_python_version(
    tmp_path, monkeypatch, version, cache_size,
):
    original_connect = sqlite3.connect
    calls = []

    def connect(*args, **kwargs):
        calls.append((args, kwargs))
        return original_connect(*args, **kwargs)

    # Replace this module's sys binding, not the interpreter's version_info:
    # dependency imports and other tests must still observe the real runtime.
    monkeypatch.setattr(db, "sys", SimpleNamespace(version_info=version))
    monkeypatch.setattr(db.sqlite3, "connect", connect)
    path = tmp_path / "shared.sqlite"
    conn = db.connect(path)
    try:
        assert calls == [((str(path),), {
            "isolation_level": None,
            "check_same_thread": False,
            "cached_statements": cache_size,
        })]
        assert conn.row_factory is sqlite3.Row
        for pragma, expected in (
            ("foreign_keys", 1), ("busy_timeout", 10000),
            ("journal_mode", "wal"), ("synchronous", 1), ("secure_delete", 1),
        ):
            assert conn.execute(f"PRAGMA {pragma}").fetchone()[0] == expected
        # The workaround must not weaken connection-local mutation authority.
        assert conn.execute(
            "SELECT hymem_evidence_mutation_authorized(), "
            "hymem_evidence_history_authorized(), "
            "hymem_evidence_destructive_authorized(), "
            "hymem_embedding_mutation_authorized()"
        ).fetchone()[:] == (0, 0, 0, 0)
    finally:
        conn.close()


def test_private_snapshot_keeps_default_cache_and_closes(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.sqlite"
    original_connect = sqlite3.connect
    seed = original_connect(path)
    seed.execute("CREATE TABLE sample (value INTEGER)")
    seed.execute("INSERT INTO sample VALUES (42)")
    seed.commit()
    seed.close()
    calls = []

    def connect(*args, **kwargs):
        calls.append((args, kwargs))
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(db.sqlite3, "connect", connect)
    with db.read_snapshot(path) as conn:
        assert len(calls) == 1
        assert "cached_statements" not in calls[0][1]
        assert calls[0][1]["uri"] is True
        assert calls[0][0][0].endswith("?mode=ro")
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        assert conn.execute("SELECT value FROM sample").fetchone()[0] == 42
        assert conn.in_transaction
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO sample VALUES (43)")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


@pytest.mark.skipif(sqlite3.threadsafety != 3, reason="requires serialized SQLite")
def test_concurrent_shared_reads_keep_complete_results_and_bound_parameters(tmp_path):
    # Exercise the real constructor, not a test-only cache override. Exact
    # constructor-policy assertions above remain deterministic even when an
    # affected runtime happens not to reproduce the race on a particular run.
    conn = db.connect(tmp_path / "concurrent.sqlite")
    conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
    expected = [(i, f"value-{i}") for i in range(12)]
    conn.executemany("INSERT INTO sample VALUES (?, ?)", expected)
    conn.execute("PRAGMA query_only = ON")
    barrier = threading.Barrier(16)

    def read(worker_id):
        failures = []
        barrier.wait(timeout=15)
        for iteration in range(100):
            pragma = conn.execute("PRAGMA query_only").fetchone()
            if pragma is None or tuple(pragma) != (1,):
                failures.append(("pragma", worker_id, iteration))
            rows = conn.execute("SELECT id, value FROM sample ORDER BY id").fetchall()
            if [tuple(row) for row in rows] != expected:
                failures.append(("all_rows", worker_id, iteration))
            wanted = (worker_id + iteration) % len(expected)
            row = conn.execute(
                "SELECT ?, id, value FROM sample WHERE id = ?",
                (worker_id, wanted),
            ).fetchone()
            if row is None or tuple(row) != (worker_id, *expected[wanted]):
                failures.append(("bound_row", worker_id, iteration))
        return failures

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(read, range(16), timeout=30))
        assert not any(results), results
    finally:
        conn.close()
