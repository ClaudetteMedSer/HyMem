from __future__ import annotations

import sqlite3

import pytest

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.deadline import DeadlineExceeded, MonotonicDeadline, use_deadline
from hymem.dreaming import retention
from hymem.dreaming.message_coverage import release_message_coverage
from tests.test_retention import _cover, _message, _session


@pytest.fixture
def covered_store(tmp_path):
    path = tmp_path / "retention.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    observer = core_db.connect(path)
    observer.execute("PRAGMA busy_timeout = 1")
    _session(conn, "ended", days_ago=200, summary=None, ended=True)
    messages = []
    for text in ("first lossless source", "second lossless source"):
        message_id = _message(conn, "ended", text, days_ago=150)
        messages.append((message_id, _cover(conn, "ended", message_id, text)))
    try:
        yield conn, observer, HyMemConfig(root=tmp_path, message_retention_days=1), messages
    finally:
        conn.close()
        observer.close()


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_pruning_cannot_race_a_legal_coverage_release(
    covered_store, monkeypatch, caller_transaction,
):
    conn, observer, cfg, messages = covered_store
    message_id, chunk_id = messages[0]
    if caller_transaction:
        # A caller's deferred transaction is not necessarily a writer yet.
        conn.execute("BEGIN")
    normal = retention.chunk_contains_message_record
    attempts = []

    def release_between_validation_and_delete(**kwargs):
        valid = normal(**kwargs)
        if valid and not attempts:
            attempts.append(True)
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                with core_db.transaction(observer):
                    release_message_coverage(
                        observer, message_id=message_id, chunk_id=chunk_id,
                        coverage_version="test-lossless-v1",
                    )
                    observer.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        return valid

    monkeypatch.setattr(
        retention, "chunk_contains_message_record", release_between_validation_and_delete,
    )
    assert retention.prune_messages(conn, cfg) == 2
    assert attempts == [True]
    assert conn.in_transaction is caller_transaction
    if caller_transaction:
        conn.execute("COMMIT")
    assert observer.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert observer.execute("SELECT COUNT(*) FROM message_retention_coverage").fetchone()[0] == 2
    assert observer.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    # Once raw pruning commits, the surviving proof is no longer releasable.
    with pytest.raises(RuntimeError, match="raw source is absent"):
        release_message_coverage(
            observer, message_id=message_id, chunk_id=chunk_id,
            coverage_version="test-lossless-v1",
        )


def test_pruning_inside_caller_transaction_does_not_commit_it(covered_store):
    conn, observer, cfg, _ = covered_store
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE sessions SET summary = 'caller work' WHERE id = 'ended'")
    assert retention.prune_messages(conn, cfg) == 2
    assert conn.in_transaction
    assert observer.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert observer.execute("SELECT summary FROM sessions").fetchone()[0] is None
    conn.execute("ROLLBACK")
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert conn.execute("SELECT summary FROM sessions").fetchone()[0] is None


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_pruning_failure_rolls_back_only_its_own_work(
    covered_store, monkeypatch, caller_transaction,
):
    conn, observer, cfg, _ = covered_store
    if caller_transaction:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE sessions SET summary = 'caller work' WHERE id = 'ended'")
    normal = retention.chunk_contains_message_record
    failure = KeyboardInterrupt("synthetic cancellation")
    calls = []

    def interrupt_second_record(**kwargs):
        calls.append(True)
        if len(calls) == 2:
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
            raise failure
        return normal(**kwargs)

    monkeypatch.setattr(retention, "chunk_contains_message_record", interrupt_second_record)
    with pytest.raises(KeyboardInterrupt) as caught:
        retention.prune_messages(conn, cfg)
    assert caught.value is failure
    assert conn.in_transaction is caller_transaction
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'lossless'"
    ).fetchone()[0] == 2
    if caller_transaction:
        assert conn.execute("SELECT summary FROM sessions").fetchone()[0] == "caller work"
        conn.execute("COMMIT")
        assert observer.execute("SELECT summary FROM sessions").fetchone()[0] == "caller work"
    assert observer.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_pruning_rechecks_deadline_before_publication(
    covered_store, monkeypatch, caller_transaction,
):
    conn, _, cfg, _ = covered_store
    if caller_transaction:
        conn.execute("BEGIN")
    now = [0.0]
    normal = retention.chunk_contains_message_record

    def cross_deadline(**kwargs):
        valid = normal(**kwargs)
        now[0] = 2.0
        return valid

    monkeypatch.setattr(retention, "chunk_contains_message_record", cross_deadline)
    with use_deadline(MonotonicDeadline(1.0, clock=lambda: now[0])):
        with pytest.raises(DeadlineExceeded):
            retention.prune_messages(conn, cfg)
    assert conn.in_transaction is caller_transaction
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


@pytest.mark.parametrize("caller_transaction", [False, True])
@pytest.mark.parametrize("lose_during_validation", [False, True])
def test_pruning_obeys_lease_fences(
    covered_store, monkeypatch, caller_transaction, lose_during_validation,
):
    conn, _, cfg, _ = covered_store
    conn.execute(
        "INSERT INTO run_lock(name, acquired_at, holder) VALUES "
        "('dreaming', CURRENT_TIMESTAMP, ?)",
        ("owner" if lose_during_validation else "successor",),
    )
    if caller_transaction:
        conn.execute("BEGIN")
    normal = retention.chunk_contains_message_record

    def lose_lease(**kwargs):
        conn.execute("UPDATE run_lock SET holder = 'successor' WHERE name = 'dreaming'")
        return normal(**kwargs)

    if lose_during_validation:
        monkeypatch.setattr(retention, "chunk_contains_message_record", lose_lease)
    token = core_db.activate_transaction_lease_fence(conn, name="dreaming", holder="owner")
    try:
        with pytest.raises(core_db.LeaseOwnershipLost):
            retention.prune_messages(conn, cfg)
    finally:
        core_db.deactivate_transaction_lease_fence(token)
    assert conn.in_transaction is caller_transaction
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert conn.execute("SELECT holder FROM run_lock").fetchone()[0] == (
        "owner" if lose_during_validation else "successor"
    )


def test_pruning_rechecks_exact_validated_source_timestamp(covered_store, monkeypatch):
    conn, _, cfg, messages = covered_store
    message_id, _ = messages[0]
    normal = retention.chunk_contains_message_record

    def change_timestamp_after_validation(**kwargs):
        valid = normal(**kwargs)
        # Custom coverage permits this raw metadata change; both dates satisfy
        # retention age, but the saved proof authenticates only the old date.
        conn.execute(
            "UPDATE messages SET created_at = '2000-01-01 00:00:00' WHERE id = ?",
            (message_id,),
        )
        return valid

    monkeypatch.setattr(retention, "chunk_contains_message_record", change_timestamp_after_validation)
    assert retention.prune_messages(conn, cfg) == 1
    assert [row[0] for row in conn.execute("SELECT id FROM messages")] == [message_id]


def test_stale_caller_snapshot_fails_before_pruning_without_ending_caller_transaction(
    covered_store,
):
    conn, observer, cfg, messages = covered_store
    message_id, chunk_id = messages[0]
    conn.execute("BEGIN")
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    with core_db.transaction(observer):
        release_message_coverage(
            observer, message_id=message_id, chunk_id=chunk_id,
            coverage_version="test-lossless-v1",
        )
        observer.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        retention.prune_messages(conn, cfg)
    assert conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    conn.execute("ROLLBACK")
    # With a fresh snapshot, only the other, still-covered message is pruned.
    assert retention.prune_messages(conn, cfg) == 1
    assert [row[0] for row in observer.execute("SELECT id FROM messages")] == [message_id]


def test_savepoint_publication_failure_rolls_back_pruning_not_caller_work(covered_store):
    conn, observer, cfg, _ = covered_store
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE sessions SET summary = 'caller work' WHERE id = 'ended'")
    failure = sqlite3.OperationalError("synthetic savepoint release failure")

    class ReleaseFailureProxy:
        failed = False

        @property
        def in_transaction(self):
            return conn.in_transaction

        def execute(self, sql, parameters=()):
            if sql == "RELEASE hymem_prune_messages" and not self.failed:
                self.failed = True
                raise failure
            return conn.execute(sql, parameters)

    with pytest.raises(sqlite3.OperationalError) as caught:
        retention.prune_messages(ReleaseFailureProxy(), cfg)
    assert caught.value is failure
    assert conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert conn.execute("SELECT summary FROM sessions").fetchone()[0] == "caller work"
    conn.execute("COMMIT")
    assert observer.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert observer.execute("SELECT summary FROM sessions").fetchone()[0] == "caller work"


def test_disabled_pruning_does_not_acquire_a_writer_lock(covered_store):
    conn, observer, cfg, _ = covered_store
    conn.execute("PRAGMA busy_timeout = 1")
    with core_db.transaction(observer):
        assert retention.prune_messages(conn, HyMemConfig(root=cfg.root)) == 0
    assert not conn.in_transaction
