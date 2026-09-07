"""Concurrency hardening: dreaming, ingestion, and reads must coexist.

Mirrors production: the Honcho server handles ingestion on its main HyMem
instance while the DreamScheduler runs on a *separate* instance (separate
SQLite connection). WAL must let two writers + a reader run against the same
database file without `database is locked` errors.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.extraction.prompts import (
    SESSION_DIGEST_SYSTEM,
    build_chunk_empty_verification_system,
    build_chunk_extraction_system,
    build_chunk_omission_verification_system,
)

_ITERATIONS = 25
_EMPTY_EXTRACTION = '{"triples":[],"markers":[],"complete":true}'


def _seed(hy: HyMem, session_id: str, n: int) -> None:
    turns = [
        (("user" if i % 2 == 0 else "assistant"),
         f"This is conversational turn number {i} with enough characters "
         f"to clear the salience threshold for chunk extraction.")
        for i in range(n)
    ]
    hy.log_messages(session_id, turns)


def test_dreaming_ingestion_and_reads_coexist(tmp_path: Path) -> None:
    cfg = HyMemConfig(
        root=tmp_path,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
        episode_granularity_enabled=False,
    )

    # Two independent instances on the same DB file == two SQLite connections,
    # exactly like the Honcho server + its background dream worker.
    ingest = HyMem(cfg, llm=StubLLMClient(default=_EMPTY_EXTRACTION),
                   embedding_client=StubEmbeddingClient())
    dreamer = HyMem(cfg, llm=StubLLMClient(default=_EMPTY_EXTRACTION),
                    embedding_client=StubEmbeddingClient())

    _seed(ingest, "sess-seed", 12)
    # Force initialization of every connection on the main thread so the test
    # exercises read/write concurrency, not init races.
    ingest.conn
    ingest.read_conn
    dreamer.conn

    errors: list[BaseException] = []

    def guard(fn):
        def wrapped():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - test wants every failure
                errors.append(exc)
        return wrapped

    def ingest_loop():
        for i in range(_ITERATIONS):
            ingest.log_messages(
                f"sess-{i % 3}",
                [("user", f"live ingestion message {i} long enough to persist "
                          f"as a chunk during the dreaming cycle")],
            )

    def dream_loop():
        for _ in range(_ITERATIONS):
            dreamer.dream()

    def read_loop():
        for i in range(_ITERATIONS):
            ingest.augment(f"conversational turn {i}")

    threads = [
        threading.Thread(target=guard(ingest_loop)),
        threading.Thread(target=guard(dream_loop)),
        threading.Thread(target=guard(read_loop)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ingest.close()
    dreamer.close()

    assert not errors, f"concurrent access raised: {errors!r}"


_COORDINATION_TIMEOUT = 15.0  # Hang protection, not a latency/performance SLO.
_LIVE_TURNS = [
    ("user", "parkedbatchtoken first complete source message"),
    ("assistant", "parkedbatchtoken second complete source message"),
]


@dataclass
class _ParkedLLM:
    """Park one real provider invocation until the independent writer finishes."""

    scope: str
    hold_writer_lock: bool = False
    conn: sqlite3.Connection | None = None
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    exited: threading.Event = field(default_factory=threading.Event)
    entry_transactions: list[bool] = field(default_factory=list)
    parked_request: LLMRequest | None = None
    parked_in_transaction: bool | None = None
    timed_out: bool = False

    def complete(self, request: LLMRequest) -> str:
        assert self.conn is not None
        self.entry_transactions.append(self.conn.in_transaction)
        if request.system == SESSION_DIGEST_SYSTEM:
            scope = "digest"
            result = (
                '{"episodes":[],"summary":"Conversational turns were recorded.",'
                '"procedures":[]}'
            )
        else:
            assert request.system in {
                build_chunk_extraction_system(),
                build_chunk_empty_verification_system(),
                build_chunk_omission_verification_system(),
            }, "unexpected provider scope"
            scope = "phase1"
            result = _EMPTY_EXTRACTION
        if scope != self.scope or self.entered.is_set():
            return result

        # The negative control deliberately recreates the original regression
        # on the actual dream connection, without patching production fences.
        if self.hold_writer_lock:
            self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.parked_request = request
            self.parked_in_transaction = self.conn.in_transaction
            self.entered.set()
            if not self.release.wait(_COORDINATION_TIMEOUT):
                self.timed_out = True
                raise AssertionError("test did not release the parked provider")
        finally:
            if self.hold_writer_lock:
                self.conn.rollback()
            self.exited.set()
        return result


@contextmanager
def _in_flight_dream(tmp_path: Path, scope: str, *, hold_writer_lock=False):
    cfg = HyMemConfig(
        root=tmp_path,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
        episode_granularity_enabled=False,
        dream_extraction_provider_attempt_budget=4,
    )
    provider = _ParkedLLM(scope, hold_writer_lock=hold_writer_lock)
    ingest = HyMem(cfg, llm=StubLLMClient(default=_EMPTY_EXTRACTION),
                   embedding_client=StubEmbeddingClient())
    dreamer = HyMem(cfg, llm=provider, embedding_client=StubEmbeddingClient())
    observer = None
    worker = None
    errors: list[BaseException] = []

    def dream_runner() -> None:
        try:
            dreamer.dream(session_ids=["sess-seed"])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        # Complete schema/connection initialization before synchronization.
        _seed(ingest, "sess-seed", 4)
        provider.conn = dreamer.conn
        assert ingest.conn is not dreamer.conn
        ingest.conn.execute("PRAGMA busy_timeout=0")
        assert ingest.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 0
        observer = sqlite3.connect(cfg.db_path, isolation_level=None)
        observer.row_factory = sqlite3.Row
        worker = threading.Thread(target=dream_runner, daemon=True)
        worker.start()
        assert provider.entered.wait(_COORDINATION_TIMEOUT), (
            f"dream never reached the real {scope} provider: {errors!r}"
        )
        assert provider.parked_request is not None
        assert "conversational turn number" in provider.parked_request.user
        if scope == "phase1":
            assert '"source_message_id":' in provider.parked_request.user
        else:
            assert "[chunk msgcov_" in provider.parked_request.user
        assert not any(provider.entry_transactions), (
            "production entered a provider while holding a transaction"
        )
        assert provider.parked_in_transaction is hold_writer_lock
        assert not provider.exited.is_set()
        yield ingest, observer, provider, worker
    finally:
        # Also runs when any assertion or fail-fast writer raises. Never leave
        # a parked provider behind or close its SQLite handle while it is used.
        provider.release.set()
        if worker is not None:
            worker.join(_COORDINATION_TIMEOUT)
        if observer is not None:
            observer.close()
        ingest.close()
        if worker is None or not worker.is_alive():
            dreamer.close()
        assert worker is None or not worker.is_alive(), "dream worker did not stop"
        assert not provider.timed_out, "provider safety deadline expired"
        assert not errors, f"concurrent access raised: {errors!r}"
        assert not any(provider.entry_transactions)


def _assert_committed_batch(ingest, observer, message_ids):
    """A third connection must see the entire batch, not uncommitted SQL."""
    assert len(message_ids) == len(_LIVE_TURNS)
    assert not ingest.conn.in_transaction
    rows = observer.execute(
        "SELECT id, role, content FROM messages WHERE session_id='sess-live' ORDER BY id"
    ).fetchall()
    assert [row["id"] for row in rows] == message_ids
    assert [(row["role"], row["content"]) for row in rows] == _LIVE_TURNS
    session = observer.execute(
        "SELECT ended_at, coverage_message_id FROM sessions WHERE id='sess-live'"
    ).fetchone()
    assert session["ended_at"] is not None
    assert session["coverage_message_id"] == message_ids[-1]
    coverage = observer.execute(
        "SELECT mc.message_id, json_extract(c.text, '$.content') AS content "
        "FROM message_retention_coverage mc JOIN chunks c ON c.id=mc.chunk_id "
        "WHERE c.session_id='sess-live' ORDER BY mc.message_id"
    ).fetchall()
    assert [(row["message_id"], row["content"]) for row in coverage] == list(
        zip(message_ids, [content for _, content in _LIVE_TURNS])
    )
    assert [row[0] for row in observer.execute(
        "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'parkedbatchtoken' "
        "ORDER BY rowid"
    )] == message_ids
    assert [row[0] for row in observer.execute(
        "SELECT mc.message_id FROM message_coverage_fts f "
        "JOIN chunks c ON c.rowid=f.rowid "
        "JOIN message_retention_coverage mc ON mc.chunk_id=c.id "
        "WHERE message_coverage_fts MATCH 'parkedbatchtoken' ORDER BY mc.message_id"
    )] == message_ids
    assert [row[0] for row in observer.execute(
        "SELECT message_id FROM message_embeddings "
        "WHERE message_id IN (SELECT id FROM messages WHERE session_id='sess-live') "
        "ORDER BY message_id"
    )] == message_ids


@pytest.mark.parametrize("scope", ["phase1", "digest"])
def test_ingestion_not_blocked_by_in_flight_dream(tmp_path: Path, scope: str) -> None:
    """A provider must not retain the WAL writer lock for its whole latency.

    An end-to-end log_messages p95 conflates ordinary transaction contention,
    SQL/Python work, scheduling and post-commit embedding. It cannot establish
    that a provider holds a transaction, nor is this test a throughput SLO.
    Instead, hold the actual provider in flight and disable SQLite lock waits:
    the complete batch and both FTS indexes must commit before its release.
    """
    with _in_flight_dream(tmp_path, scope) as (ingest, observer, provider, _):
        ids = ingest.log_messages("sess-live", _LIVE_TURNS, close_session=True)
        _assert_committed_batch(ingest, observer, ids)
        assert not provider.release.is_set()
        assert not provider.exited.is_set()
        assert not provider.conn.in_transaction


@pytest.mark.parametrize("scope", ["phase1", "digest"])
def test_in_flight_lock_probe_detects_provider_held_write_transaction(
    tmp_path: Path, scope: str,
) -> None:
    """Negative control: the same writer detects the original lock regression."""
    with _in_flight_dream(tmp_path, scope, hold_writer_lock=True) as state:
        ingest, observer, provider, worker = state
        for _ in range(2):
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                ingest.log_messages("sess-live", _LIVE_TURNS, close_session=True)
            assert not ingest.conn.in_transaction
            assert observer.execute(
                "SELECT count(*) FROM sessions WHERE id='sess-live'"
            ).fetchone()[0] == 0
            assert observer.execute(
                "SELECT count(*) FROM messages WHERE session_id='sess-live'"
            ).fetchone()[0] == 0
            assert not provider.exited.is_set()

        provider.release.set()
        worker.join(_COORDINATION_TIMEOUT)
        assert not worker.is_alive()
        assert provider.exited.is_set()
        assert not provider.conn.in_transaction
        ids = ingest.log_messages("sess-live", _LIVE_TURNS, close_session=True)
        _assert_committed_batch(ingest, observer, ids)


def test_in_flight_probe_releases_provider_after_assertion_failure(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="forced assertion failure"):
        with _in_flight_dream(tmp_path, "phase1") as (_, _, provider, worker):
            raise AssertionError("forced assertion failure")
    assert provider.release.is_set()
    assert provider.exited.is_set()
    assert not worker.is_alive()
