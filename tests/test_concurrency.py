"""Concurrency hardening: dreaming, ingestion, and reads must coexist.

Mirrors production: the Honcho server handles ingestion on its main HyMem
instance while `_background_dream` runs on a *separate* instance (separate
SQLite connection). WAL must let two writers + a reader run against the same
database file without `database is locked` errors.
"""
from __future__ import annotations

import threading
from pathlib import Path

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.extraction.llm import StubLLMClient

_ITERATIONS = 25


def _seed(hy: HyMem, session_id: str, n: int) -> None:
    turns = [
        (("user" if i % 2 == 0 else "assistant"),
         f"This is conversational turn number {i} with enough characters "
         f"to clear the salience threshold for chunk extraction.")
        for i in range(n)
    ]
    hy.log_messages(session_id, turns)


def test_dreaming_ingestion_and_reads_coexist(tmp_path: Path) -> None:
    cfg = HyMemConfig(root=tmp_path)

    # Two independent instances on the same DB file == two SQLite connections,
    # exactly like the Honcho server + its background dream worker.
    ingest = HyMem(cfg, llm=StubLLMClient(default="[]"),
                   embedding_client=StubEmbeddingClient())
    dreamer = HyMem(cfg, llm=StubLLMClient(default="[]"),
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
