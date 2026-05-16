"""Concurrency hardening: dreaming, ingestion, and reads must coexist.

Mirrors production: the Honcho server handles ingestion on its main HyMem
instance while the DreamScheduler runs on a *separate* instance (separate
SQLite connection). WAL must let two writers + a reader run against the same
database file without `database is locked` errors.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.extraction.llm import LLMClient, LLMRequest, StubLLMClient

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


@dataclass
class _SlowLLM:
    """LLM stub that sleeps on every call — mimics real provider latency."""
    delay_seconds: float = 0.2
    default: str = "[]"
    call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def complete(self, request: LLMRequest) -> str:
        with self._lock:
            self.call_count += 1
        time.sleep(self.delay_seconds)
        return self.default


def test_ingestion_not_blocked_by_in_flight_dream(tmp_path: Path) -> None:
    """Regression for the proc=0 stall.

    With LLM calls inside BEGIN IMMEDIATE (pre-fix), an in-flight dream holds
    the SQLite WAL writer lock for the duration of each LLM call (~200 ms in
    this test, multi-seconds in prod). Concurrent log_messages writes block
    on the file-level lock and pile up.

    Post-fix, LLM calls run outside transactions, so the dream only holds the
    writer lock for the brief persist step. Ingestion writes should complete
    in milliseconds even while the dream is mid-cycle.
    """
    cfg = HyMemConfig(root=tmp_path)
    slow = _SlowLLM(delay_seconds=0.2)

    ingest = HyMem(cfg, llm=StubLLMClient(default="[]"),
                   embedding_client=StubEmbeddingClient())
    dreamer: HyMem = HyMem(cfg, llm=slow, embedding_client=StubEmbeddingClient())

    # Seed enough chunks so the dream actually has phase1 work to do.
    _seed(ingest, "sess-seed", 20)
    ingest.conn
    dreamer.conn

    dream_done = threading.Event()
    ingest_latencies: list[float] = []
    errors: list[BaseException] = []

    def dream_runner() -> None:
        try:
            dreamer.dream()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            dream_done.set()

    def ingest_runner() -> None:
        # Give the dream a head start so it's mid-cycle when we write.
        time.sleep(0.05)
        for i in range(20):
            if dream_done.is_set():
                break
            start = time.monotonic()
            try:
                ingest.log_messages(
                    "sess-live",
                    [("user", f"live message {i} long enough to be a chunk in the dream cycle")],
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return
            ingest_latencies.append(time.monotonic() - start)
            time.sleep(0.02)

    t_dream = threading.Thread(target=dream_runner)
    t_ingest = threading.Thread(target=ingest_runner)
    t_dream.start()
    t_ingest.start()
    t_dream.join(timeout=30)
    t_ingest.join(timeout=30)

    ingest.close()
    dreamer.close()

    assert not errors, f"concurrent access raised: {errors!r}"
    assert slow.call_count > 0, "dream should have made LLM calls during the test"
    assert ingest_latencies, "ingest writer should have run during the dream"
    # Pre-fix: latencies would cluster at ~200 ms (full LLM-call hold time)
    # or hit busy_timeout. Post-fix: tens of milliseconds at most.
    p95 = sorted(ingest_latencies)[int(0.95 * (len(ingest_latencies) - 1))]
    assert p95 < 0.1, (
        f"ingestion p95 latency {p95*1000:.0f} ms — dream still blocks writers"
    )
