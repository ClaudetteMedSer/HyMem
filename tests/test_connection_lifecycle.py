"""Connection setup is private until schema, scope and read-only guards exist."""
from __future__ import annotations

import sqlite3
import threading

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem import api
from hymem.core import db
from hymem.extraction.llm import StubLLMClient


_TIMEOUT = 15.0  # Deadlock protection only; assertions use explicit handshakes.


class _ObservedRLock:
    """Report actual lock contention, not a guessed thread-start delay."""

    def __init__(self):
        self.lock = threading.RLock()
        self.contended = threading.Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.contended.set()
            assert self.lock.acquire(timeout=_TIMEOUT), "lifecycle lock deadlocked"
        return self

    def __exit__(self, *_):
        self.lock.release()


def _start(fn, results, errors):
    def run():
        try:
            results.append(fn())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker


@pytest.fixture
def instance(tmp_path):
    hy = HyMem(
        HyMemConfig(root=tmp_path, aggregation_nodes_enabled=False),
        llm=StubLLMClient(default="first producer"),
    )
    yield hy
    hy.close()


def _assert_scoped(conn, hy):
    generation = hy._phase1_generation
    assert generation is not None
    assert conn.execute(
        "SELECT hymem_phase1_generation_is_current(?, 1)",
        (generation["generation_key"],),
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT hymem_phase1_generation_is_current('unrelated', 1)"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT hymem_rule_routing_is_current('unrelated')"
    ).fetchone()[0] == 0


@pytest.mark.parametrize("attribute", ["conn", "read_conn"])
def test_concurrent_callers_share_only_fully_initialized_connection(
    instance, monkeypatch, attribute,
):
    if attribute == "read_conn":
        instance.conn
    lock = _ObservedRLock()
    monkeypatch.setattr(instance, "_connection_lock", lock)
    entered, release = threading.Event(), threading.Event()
    original_scope = instance._scope_phase1_reads
    original_connect = db.connect
    opened = []

    def connect(path):
        conn = original_connect(path)
        opened.append(conn)
        return conn

    def scope(conn):
        entered.set()
        assert release.wait(_TIMEOUT), "test did not release connection setup"
        original_scope(conn)

    monkeypatch.setattr(db, "connect", connect)
    monkeypatch.setattr(instance, "_scope_phase1_reads", scope)
    results, errors, workers = [], [], []
    try:
        workers.append(_start(lambda: getattr(instance, attribute), results, errors))
        assert entered.wait(_TIMEOUT), errors
        assert getattr(instance, "_" + attribute) is None
        if attribute == "conn":
            assert instance._initialized is False
        workers.append(_start(lambda: getattr(instance, attribute), results, errors))
        assert lock.contended.wait(_TIMEOUT), errors
        assert results == []
        assert len(opened) == 1
    finally:
        release.set()
        for worker in workers:
            worker.join(_TIMEOUT)
    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    assert len(results) == 2
    assert results[0] is results[1] is opened[0]
    _assert_scoped(results[0], instance)
    assert results[0].execute("PRAGMA query_only").fetchone()[0] == (
        attribute == "read_conn"
    )
    if attribute == "read_conn":
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            results[0].execute("CREATE TABLE forbidden_write (id INTEGER)")


@pytest.mark.parametrize("stage", ["schema", "redaction", "scope"])
def test_failed_primary_setup_closes_unpublished_connection_and_retries(
    instance, monkeypatch, stage,
):
    original_connect = db.connect
    opened = []

    def connect(path):
        conn = original_connect(path)
        opened.append(conn)
        return conn

    def fail(*_):
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(db, "connect", connect)
    with monkeypatch.context() as patch:
        if stage == "schema":
            patch.setattr(db, "initialize", fail)
        elif stage == "redaction":
            patch.setattr(api, "enforce_profile_redaction_policy", fail)
        else:
            patch.setattr(instance, "_scope_phase1_reads", fail)
        with pytest.raises(RuntimeError, match="injected setup failure"):
            instance.conn
    assert instance._conn is None
    assert instance._initialized is False
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    recovered = instance.conn
    assert recovered is opened[1]
    assert instance._initialized is True
    _assert_scoped(recovered, instance)
    assert db.schema_version(recovered) == db.EXPECTED_SCHEMA_VERSION


@pytest.mark.parametrize("stage", ["scope", "vec", "query_only"])
def test_failed_reader_setup_closes_unpublished_connection_and_retries(
    instance, monkeypatch, stage,
):
    primary = instance.conn
    original_connect = db.connect
    opened = []

    def connect(path):
        conn = original_connect(path)
        opened.append(conn)
        if stage == "query_only" and len(opened) == 1:
            conn.set_authorizer(lambda action, name, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_PRAGMA and name == "query_only"
                else sqlite3.SQLITE_OK
            ))
        return conn

    def fail(*_):
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(db, "connect", connect)
    with monkeypatch.context() as patch:
        if stage == "scope":
            patch.setattr(instance, "_scope_phase1_reads", fail)
        elif stage == "vec":
            patch.setattr(db, "_load_vec_extension", fail)
        error = sqlite3.DatabaseError if stage == "query_only" else RuntimeError
        with pytest.raises(error):
            instance.read_conn
    assert instance._read_conn is None
    assert instance.conn is primary
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    recovered = instance.read_conn
    assert recovered is opened[1]
    assert recovered.execute("PRAGMA query_only").fetchone()[0] == 1
    _assert_scoped(recovered, instance)


@pytest.mark.parametrize("attribute", ["conn", "read_conn"])
def test_cleanup_failure_preserves_primary_setup_error(instance, monkeypatch, attribute):
    if attribute == "read_conn":
        instance.conn
    primary_error = ValueError("injected setup failure")

    class FailedConnection:
        close_calls = 0

        def close(self):
            self.close_calls += 1
            raise RuntimeError("private cleanup detail must not enter error notes")

    conn = FailedConnection()

    def fail(*_):
        raise primary_error

    with monkeypatch.context() as patch:
        patch.setattr(db, "connect", lambda _: conn)
        if attribute == "conn":
            patch.setattr(db, "initialize", fail)
        else:
            patch.setattr(instance, "_scope_phase1_reads", fail)
        with pytest.raises(ValueError, match="injected setup failure") as caught:
            getattr(instance, attribute)
    assert caught.value is primary_error
    assert caught.value.__notes__ == ["SQLite connection cleanup failed: RuntimeError"]
    assert conn.close_calls == 1
    assert getattr(instance, "_" + attribute) is None
    _assert_scoped(getattr(instance, attribute), instance)


@pytest.mark.parametrize("operation", ["close", "set_llm"])
def test_lifecycle_change_waits_for_reader_setup(instance, monkeypatch, operation):
    primary = instance.conn
    old_generation = instance._phase1_generation["generation_key"]
    lock = _ObservedRLock()
    monkeypatch.setattr(instance, "_connection_lock", lock)
    entered, release = threading.Event(), threading.Event()
    original_scope = instance._scope_phase1_reads

    def scope(conn):
        if not entered.is_set():
            entered.set()
            assert release.wait(_TIMEOUT), "test did not release connection setup"
        original_scope(conn)

    monkeypatch.setattr(instance, "_scope_phase1_reads", scope)
    results, changes, errors, workers = [], [], [], []
    replacement = StubLLMClient(default="different producer")
    change = instance.close if operation == "close" else lambda: instance.set_llm(replacement)
    try:
        workers.append(_start(lambda: instance.read_conn, results, errors))
        assert entered.wait(_TIMEOUT), errors
        workers.append(_start(change, changes, errors))
        assert lock.contended.wait(_TIMEOUT), errors
        assert changes == []
    finally:
        release.set()
        for worker in workers:
            worker.join(_TIMEOUT)
    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    assert len(results) == len(changes) == 1
    if operation == "close":
        assert instance._conn is instance._read_conn is None
        assert instance._initialized is False
        for conn in (primary, results[0]):
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                conn.execute("SELECT 1")
        assert instance.read_conn is not results[0]
        _assert_scoped(instance.read_conn, instance)
    else:
        assert instance._llm is replacement
        assert instance._phase1_generation["generation_key"] != old_generation
        for conn in (primary, results[0]):
            _assert_scoped(conn, instance)
            assert conn.execute(
                "SELECT hymem_phase1_generation_is_current(?, 1)",
                (old_generation,),
            ).fetchone()[0] == 0


def test_fork_and_reopen_keep_independent_connections_and_locks(instance):
    primary, reader = instance.conn, instance.read_conn
    embed = StubEmbeddingClient()
    instance.set_embedding_client(embed)
    fork = instance.fork()
    try:
        assert fork._connection_lock is not instance._connection_lock
        assert fork._llm is instance._llm
        assert fork._embed is embed
        assert fork.conn is not primary
        assert fork.read_conn is not reader
        instance._token_overlap_index = {"stale": ["cached"]}
        instance.close()
        instance.close()  # idempotent; closing the parent does not close a fork.
        assert instance._token_overlap_index is None
        assert fork.read_conn.execute("SELECT 1").fetchone()[0] == 1
        assert instance.conn is not primary
        assert instance.read_conn is not reader
        _assert_scoped(instance.read_conn, instance)
    finally:
        fork.close()
