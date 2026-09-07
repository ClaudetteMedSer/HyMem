from __future__ import annotations

import json
import math
import threading

import pytest

from hymem import HyMem, StubEmbeddingClient
from hymem.extraction.embeddings import (
    CachedEmbeddingClient,
    embedding_text_hash,
    normalize_text,
)
from hymem.extraction.llm import StubLLMClient
from hymem.core import db as core_db
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.query.augment import _vector_search


class _ExactTestEmbeddingDeclaration:
    """Explicit durable authority for behavior-changing test subclasses."""

    def embedding_producer_declaration(self):
        return {
            "schema": "hymem-custom-embedding-producer-declaration-v2",
            "implementation": "tests.embedding-control",
            "implementation_revision": "v1",
            "deployment_revision": "fixture-v1",
            "deployment_tenant": "tests",
            "model_revision": self.model,
            "request_policy": "deterministic-test-v1",
            "dimension": self.dim,
            "network_free": True,
        }


def test_cached_embedding_close_delegates_exactly_once_and_rejects_late_use():
    class ClosableEmbedding:
        model = "close-test"
        dim = 2

        def __init__(self):
            self.close_calls = 0

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def close(self):
            self.close_calls += 1

    inner = ClosableEmbedding()
    cached = CachedEmbeddingClient(inner)

    assert cached.embed(["before close"]) == [[1.0, 0.0]]
    cached.close()
    cached.close()

    assert inner.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        cached.embed(["after close"])


def test_cached_embedding_close_waits_for_inflight_provider_call():
    entered = threading.Event()
    release = threading.Event()
    events: list[str] = []

    class BlockingEmbedding:
        model = "blocking"
        dim = 2

        def embed(self, texts):
            entered.set()
            assert release.wait(timeout=5.0)
            events.append("embed:return")
            return [[1.0, 0.0] for _ in texts]

        def close(self):
            events.append("close")

    cached = CachedEmbeddingClient(BlockingEmbedding())
    worker = threading.Thread(target=lambda: cached.embed(["x"]))
    worker.start()
    assert entered.wait(timeout=2.0)
    closer = threading.Thread(target=cached.close)
    closer.start()
    assert closer.is_alive()
    release.set()
    worker.join(timeout=2.0)
    closer.join(timeout=2.0)

    assert not worker.is_alive() and not closer.is_alive()
    assert events == ["embed:return", "close"]


def test_cached_embedding_does_not_retry_a_failed_transport_close():
    primary = RuntimeError("provider close failed")

    class FailingCloseEmbedding:
        model = "close-failure"
        dim = 2

        def __init__(self):
            self.close_calls = 0

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def close(self):
            self.close_calls += 1
            raise primary

    inner = FailingCloseEmbedding()
    cached = CachedEmbeddingClient(inner)

    with pytest.raises(RuntimeError) as caught:
        cached.close()
    assert caught.value is primary
    cached.close()
    assert inner.close_calls == 1


def test_stub_embedding_client_shape_and_determinism():
    e = StubEmbeddingClient()
    v1 = e.embed(["hello world"])
    v2 = e.embed(["hello world"])
    assert v1 == v2
    assert len(v1) == 1
    assert len(v1[0]) == e.dim == 16

    norm = math.sqrt(sum(x * x for x in v1[0]))
    assert math.isclose(norm, 1.0, rel_tol=1e-9)

    different = e.embed(["something completely different"])[0]
    assert different != v1[0]


def test_stub_embedding_identical_text_cosine_one():
    e = StubEmbeddingClient()
    [a, b] = e.embed(["same text", "same text"])
    cos = sum(x * y for x, y in zip(a, b))
    assert math.isclose(cos, 1.0, rel_tol=1e-9)


def test_dreaming_populates_chunk_embeddings(hy_with_embed):
    sid = "s1"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(
        sid, "assistant", "I'll set up Docker for the local dev environment."
    )
    hy_with_embed.log_message(
        sid,
        "user",
        "No, actually we don't use Docker for local dev anymore. We use uv.",
    )
    hy_with_embed.close_session(sid)

    report = hy_with_embed.dream()
    assert report.chunks_embedded >= 1

    rows = hy_with_embed.conn.execute(
        "SELECT chunk_id, model, dim FROM chunk_embeddings"
    ).fetchall()
    assert len(rows) >= 1
    model, _dim = embedding_storage_identity(hy_with_embed._embed)
    assert all(r["model"] == model for r in rows)
    assert all(r["dim"] == 16 for r in rows)


def test_persist_chunk_embeddings_reembed_same_rowid(hy_with_embed):
    """Re-embedding a chunk whose rowid already exists in vec_chunks must not
    crash. vec0 rejects INSERT OR REPLACE with a UNIQUE-constraint error, so the
    chunk path delete-then-inserts (matching the episode path)."""
    from hymem.dreaming.embeddings import (
        PendingChunkEmbeddings,
        persist_chunk_embeddings,
    )

    conn = hy_with_embed.conn
    conn.execute("INSERT INTO sessions(id) VALUES ('s')")
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('c1', 's', 1, 1, 'r', 'txt')"
    )
    rowid = conn.execute("SELECT rowid FROM chunks WHERE id = 'c1'").fetchone()["rowid"]
    producer = StubEmbeddingClient(model_name="stub", dim_value=3)
    model, _dim = embedding_storage_identity(producer)

    def pending(vec: list[float]) -> PendingChunkEmbeddings:
        return PendingChunkEmbeddings(
            ids=["c1"], chunk_rowids=[rowid], vectors=[vec], dim=len(vec),
                model=model, text_hashes=[embedding_text_hash("txt")],
                from_cache=[False],
        )

    with core_db.transaction(conn):
        persist_chunk_embeddings(conn, pending([1.0, 0.0, 0.0]))
    # Second write for the same rowid — used to raise OperationalError.
    with core_db.transaction(conn):
        persist_chunk_embeddings(conn, pending([0.0, 1.0, 0.0]))

    if core_db.has_vec_table(conn, table="vec_chunks"):
        cnt = conn.execute(
            "SELECT COUNT(*) AS c FROM vec_chunks WHERE rowid = ?", (rowid,)
        ).fetchone()["c"]
        assert cnt == 1  # replaced, not duplicated


def test_augment_without_embedding_client_uses_fts_only(hy):
    sid = "s1"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "anything")
    hy.log_message(
        sid,
        "user",
        "Let's use postgres for the production database, it scales well for our needs.",
    )
    hy.close_session(sid)
    hy.dream()

    ctx = hy.augment("postgres")
    assert any("postgres" in h.text.lower() for h in ctx.fts_hits)


def test_fts_only_hits_have_bm25_score_kind(hy):
    sid = "s1"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "anything")
    hy.log_message(
        sid,
        "user",
        "Let's use postgres for the production database, it scales well for our needs.",
    )
    hy.close_session(sid)
    hy.dream()

    ctx = hy.augment("postgres")
    assert ctx.fts_hits, "expected at least one hit"
    assert all(h.score_kind == "bm25" for h in ctx.fts_hits)


def test_rrf_merged_hits_have_rrf_score_kind(hy_with_embed):
    sid = "s1"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(sid, "assistant", "anything")
    hy_with_embed.log_message(
        sid,
        "user",
        "Let's use postgres for the production database, it scales well for our needs.",
    )
    hy_with_embed.close_session(sid)
    hy_with_embed.dream()

    ctx = hy_with_embed.augment("postgres")
    assert ctx.fts_hits, "expected at least one hit"
    assert all(h.score_kind == "rrf" for h in ctx.fts_hits)


def test_augment_with_embedding_client_ranks_semantic_match_higher(cfg):
    """A query identical to one chunk's text should rank that chunk above
    another chunk that shares an FTS keyword but is otherwise unrelated."""
    embed = StubEmbeddingClient()
    llm = StubLLMClient(default="[]")
    hy = HyMem(cfg, llm=llm, embedding_client=embed)
    try:
        sid = "s1"
        hy.open_session(sid)
        # Chunk A: target — semantically identical to the query we'll issue.
        hy.log_message(sid, "assistant", "anything one")
        hy.log_message(
            sid,
            "user",
            "I prefer fastapi for my web services because it is async and modern.",
        )
        # Chunk B: different topic but shares the keyword "fastapi".
        hy.log_message(sid, "assistant", "anything two")
        hy.log_message(
            sid,
            "user",
            "I prefer postgres over mysql, and unrelated to that the keyword fastapi appears here too.",
        )
        hy.close_session(sid)
        hy.dream()

        # Query is *exactly* the user-text of chunk A — stub embedding -> cosine 1.0.
        # Chunk B shares the literal token "fastapi" but is semantically different.
        target_text = (
            "I prefer fastapi for my web services because it is async and modern."
        )
        ctx = hy.augment(target_text)
        assert ctx.fts_hits, "expected at least one hit"

        # Find the chunk whose text includes the target — it must be ranked first.
        ranks = {h.chunk_id: i for i, h in enumerate(ctx.fts_hits)}
        target_hit = next(
            h for h in ctx.fts_hits if "async and modern" in h.text
        )
        other_hit = next(
            (h for h in ctx.fts_hits if "postgres over mysql" in h.text), None
        )
        if other_hit is not None:
            assert ranks[target_hit.chunk_id] < ranks[other_hit.chunk_id]
    finally:
        hy.close()


def test_cached_embedding_client_skips_repeat_calls():
    """Second embed of the same text hits the cache — inner client not called."""
    inner = StubEmbeddingClient()
    cached = CachedEmbeddingClient(inner)

    v1 = cached.embed(["hello world"])
    v2 = cached.embed(["hello world"])
    assert v1 == v2
    # Inner stub saw the request once; the second call was served from cache.
    assert len(inner.calls) == 1
    assert cached.hits == 1
    assert cached.misses == 1


def test_cached_embedding_client_batch_splits_hits_and_misses():
    """A mixed batch forwards only uncached texts and re-stitches in order."""
    inner = StubEmbeddingClient()
    cached = CachedEmbeddingClient(inner)
    cached.embed(["alpha", "beta"])  # warm cache
    inner.calls.clear()

    out = cached.embed(["alpha", "gamma", "beta", "delta"])
    # Inner only sees the misses (gamma, delta) in their input order.
    assert inner.calls == [["gamma", "delta"]]
    assert len(out) == 4
    # Each output aligns with the input text — verify by recomputing on a
    # fresh stub (deterministic hash → same vector).
    fresh = StubEmbeddingClient()
    expected = fresh.embed(["alpha", "gamma", "beta", "delta"])
    assert out == expected


def test_cached_embedding_client_lru_evicts_oldest():
    inner = StubEmbeddingClient()
    cached = CachedEmbeddingClient(inner, max_size=2)
    cached.embed(["a"])      # cache order: [a]
    cached.embed(["b"])      # cache order: [a, b]
    cached.embed(["a"])      # HIT → cache order: [b, a]
    cached.embed(["c"])      # MISS, evicts b (oldest) → cache: [a, c]
    inner.calls.clear()

    cached.embed(["a"])      # HIT — survives because it was just used.
    assert inner.calls == []
    cached.embed(["b"])      # MISS — b was the one evicted.
    assert inner.calls == [["b"]]


def test_cached_embedding_client_preserves_model_and_dim():
    inner = StubEmbeddingClient()
    cached = CachedEmbeddingClient(inner)
    assert cached.model == inner.model
    assert cached.dim == inner.dim


def test_cached_embedding_client_empty_batch_short_circuits():
    inner = StubEmbeddingClient()
    cached = CachedEmbeddingClient(inner)
    assert cached.embed([]) == []
    assert inner.calls == []


def test_normalize_text_strips_collapses_lowercases():
    assert normalize_text("  Hello   World  ") == "hello world"
    assert normalize_text("\tHELLO\nworld\n") == "hello world"
    assert normalize_text("same") == normalize_text(" SAME ")


def test_embedding_cache_skips_repeat_chunk_text_across_dreams(cfg):
    """Two chunks with identical text in separate dream runs embed once: the
    second run reads the vector from embedding_cache."""
    embed = StubEmbeddingClient()
    llm = StubLLMClient(default="[]")
    hy = HyMem(cfg, llm=llm, embedding_client=embed)
    try:
        text = (
            "I prefer fastapi for my web services because it is async and modern."
        )
        hy.open_session("s1")
        hy.log_message("s1", "assistant", "anything")
        hy.log_message("s1", "user", text)
        hy.close_session("s1")
        report1 = hy.dream()
        assert report1.chunks_embedded >= 1
        assert report1.chunks_embedded_from_cache == 0

        embed.calls.clear()

        hy.open_session("s2")
        hy.log_message("s2", "assistant", "anything")
        hy.log_message("s2", "user", text)
        hy.close_session("s2")
        report2 = hy.dream()
        assert report2.chunks_embedded >= 1
        assert report2.chunks_embedded_from_cache >= 1
        # The duplicate text must not appear in any embedder batch on run 2.
        assert all(text not in batch for batch in embed.calls)

        cache_rows = hy.conn.execute(
            "SELECT COUNT(*) AS c FROM embedding_cache"
        ).fetchone()["c"]
        assert cache_rows >= 1
    finally:
        hy.close()


def test_chunk_embedding_runs_in_parallel_with_phase1(cfg, monkeypatch):
    """The first chunk-embedding request and Phase 1 make progress together.

    The event handshake asserts the actual happens-before relationship instead
    of inferring it from a wall-clock threshold.  The embedding worker waits
    until the first Phase-1 provider call has entered; that provider call only
    releases it after observing the embed request still in flight.  A serial
    implementation therefore times out the handshake and fails
    deterministically.

    Prompt v20 performs a primary extraction and one omission-verification pass
    for every non-empty terminal result, hence exactly two Phase-1 calls for
    each of the five chunks in this fixture.
    """
    import threading
    from dataclasses import replace as _dc_replace

    from hymem.extraction.llm import LLMRequest

    sync_timeout = 2.0
    embed_started = threading.Event()
    phase1_entered_while_embedding = threading.Event()
    embed_observed_phase1 = threading.Event()
    phase1_observed_embed_inflight = threading.Event()
    first_embed_finished = threading.Event()

    from concurrent.futures import ThreadPoolExecutor as _RealThreadPoolExecutor
    from hymem.dreaming import runner as runner_mod

    class CoordinatedExecutor:
        """Instrument the worker boundary without altering producer identity."""

        def __init__(self, *args, **kwargs):
            self._inner = _RealThreadPoolExecutor(*args, **kwargs)

        def submit(self, function):
            def coordinated():
                embed_started.set()
                if phase1_entered_while_embedding.wait(sync_timeout):
                    embed_observed_phase1.set()
                try:
                    return function()
                finally:
                    first_embed_finished.set()

            return self._inner.submit(coordinated)

        def shutdown(self, *args, **kwargs):
            return self._inner.shutdown(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "ThreadPoolExecutor", CoordinatedExecutor)

    class CoordinatedLLM(StubLLMClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.phase1_calls = 0

        def complete(self, request: LLMRequest) -> str:
            if "source_message_id (integer)" in request.system:
                self.phase1_calls += 1
                if self.phase1_calls == 1 and embed_started.wait(sync_timeout):
                    if not first_embed_finished.is_set():
                        phase1_observed_embed_inflight.set()
                        phase1_entered_while_embedding.set()
            return super().complete(request)

    embed = StubEmbeddingClient()
    tight_cfg = _dc_replace(cfg, dream_budget=5, dream_baseline_budget=0)

    llm = CoordinatedLLM(
        fixtures={
            "single pass": json.dumps({
                "triples": [],
                "markers": [{
                    "kind": "preference",
                    "statement": "user explicitly prefers the named choice",
                }],
                "complete": True,
            }),
        },
        default="[]",
    )
    hy = HyMem(tight_cfg, llm=llm, embedding_client=embed)
    try:
        hy.open_session("s1")
        for i in range(5):
            hy.log_message("s1", "assistant", "anything")
            hy.log_message(
                "s1", "user",
                f"I prefer choice_{i} for the local dev environment because it is fast.",
            )
        hy.close_session("s1")

        report = hy.dream()

        assert report.chunks_embedded >= 5
        assert llm.phase1_calls == 5 * 2
        assert phase1_observed_embed_inflight.is_set(), (
            "Phase 1 did not observe the chunk embedding request in flight"
        )
        assert embed_observed_phase1.is_set(), (
            "the chunk embedding request finished before Phase 1 entered"
        )
    finally:
        hy.close()


def test_background_embed_failure_falls_back_to_post_loop_fetch(cfg, monkeypatch):
    """If the background embed task raises, the dream cycle must continue
    and the post-loop fetch_chunk_embeddings call must still embed the
    affected chunks."""
    from concurrent.futures import ThreadPoolExecutor as _RealThreadPoolExecutor
    from hymem.dreaming import runner as runner_mod

    class FailingFirstExecutor:
        def __init__(self, *args, **kwargs):
            self._inner = _RealThreadPoolExecutor(*args, **kwargs)
            self._failed = False

        def submit(self, function):
            if not self._failed:
                self._failed = True

                def fail():
                    raise RuntimeError("simulated background failure")

                return self._inner.submit(fail)
            return self._inner.submit(function)

        def shutdown(self, *args, **kwargs):
            return self._inner.shutdown(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "ThreadPoolExecutor", FailingFirstExecutor)
    embed = StubEmbeddingClient()
    llm = StubLLMClient(default="[]")
    hy = HyMem(cfg, llm=llm, embedding_client=embed)
    try:
        hy.open_session("s1")
        hy.log_message("s1", "assistant", "anything")
        hy.log_message(
            "s1", "user",
            "I prefer fastapi for my web services because it is async and modern.",
        )
        hy.close_session("s1")

        # Should NOT raise — background failure is logged, fallback embeds.
        report = hy.dream()
        # Background call failed, fallback fetch_chunk_embeddings ran the
        # second embedder call and persisted the chunk.
        assert report.chunks_embedded >= 1
        assert embed.calls

        rows = hy.conn.execute(
            "SELECT COUNT(*) AS c FROM chunk_embeddings"
        ).fetchone()["c"]
        assert rows >= 1
    finally:
        hy.close()


def test_vector_search_respects_embedding_max_scan(cfg):
    embed = StubEmbeddingClient()
    conn = core_db.connect(cfg.db_path)
    try:
        core_db.initialize(conn)
        with core_db.transaction(conn):
            conn.execute("INSERT INTO sessions(id) VALUES (?)", ("s1",))
            for i in range(5):
                cid = f"c{i}"
                ts = f"2026-01-0{i + 1} 00:00:00"
                conn.execute(
                    "INSERT INTO chunks(id, session_id, start_message_id, "
                    "end_message_id, salience_reason, text, created_at) "
                    "VALUES (?, ?, 0, 0, 'test', ?, ?)",
                    (cid, "s1", f"chunk text {i}", ts),
                )
                vec = embed.embed([f"chunk text {i}"])[0]
                model, dim = embedding_storage_identity(embed)
                with core_db.embedding_mutation(conn):
                    conn.execute(
                        "INSERT INTO chunk_embeddings("
                        "chunk_id,vector_json,model,dim,text_hash) "
                        "VALUES (?,?,?,?,?)",
                        (
                            cid, json.dumps(vec), model, dim,
                            embedding_text_hash(f"chunk text {i}"),
                        ),
                    )

        hits = _vector_search(conn, embed, "anything", top_k=5, max_scan=2)
        assert len(hits) <= 2
        # Only the two most-recent chunks (c3, c4) may appear.
        assert all(h.chunk_id in {"c3", "c4"} for h in hits)
    finally:
        conn.close()
