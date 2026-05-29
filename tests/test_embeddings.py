from __future__ import annotations

import json
import math

from hymem import HyMem, StubEmbeddingClient
from hymem.extraction.embeddings import CachedEmbeddingClient, normalize_text
from hymem.extraction.llm import StubLLMClient
from hymem.core import db as core_db
from hymem.query.augment import _vector_search


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
    assert all(r["model"] == "stub" for r in rows)
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

    def pending(vec: list[float]) -> PendingChunkEmbeddings:
        return PendingChunkEmbeddings(
            ids=["c1"], chunk_rowids=[rowid], vectors=[vec], dim=len(vec),
            model="stub", text_hashes=["h"], from_cache=[False],
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


def test_chunk_embedding_runs_in_parallel_with_phase1(cfg):
    """A slow embedder + a non-trivial Phase 1 LLM stream should finish in
    roughly max(LLM*N, EMBED) wall-time rather than their sum, because chunk
    embedding is kicked off on a background thread after each persist_chunks
    and joined after the per-session loop.

    Tunings: with 5 chunks → 5 Phase-1 LLM calls (one combined triples+markers
    call per chunk) + 1 batched digest tail call = 6 LLM calls. LLM_DELAY is
    sized so the Phase-1 stream (5*LLM_DELAY) is comparable to EMBED_DELAY, so
    overlapping the two saves close to a full EMBED_DELAY. Serial:
    6*LLM_DELAY + EMBED_DELAY. Parallel: ~max(5*LLM_DELAY, EMBED_DELAY) + the
    digest call.
    """
    import time
    from dataclasses import replace as _dc_replace

    from hymem.extraction.llm import LLMRequest

    LLM_DELAY = 0.04
    EMBED_DELAY = 0.20

    class SlowEmbed(StubEmbeddingClient):
        def embed(self, texts):
            time.sleep(EMBED_DELAY)
            return super().embed(texts)

    class SlowLLM(StubLLMClient):
        def complete(self, request: LLMRequest) -> str:
            time.sleep(LLM_DELAY)
            return super().complete(request)

    embed = SlowEmbed()
    tight_cfg = _dc_replace(cfg, dream_budget=5, dream_baseline_budget=0)

    llm = SlowLLM(default="[]")
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

        t0 = time.monotonic()
        report = hy.dream()
        elapsed = time.monotonic() - t0

        assert report.chunks_embedded >= 5
        # We saved roughly EMBED_DELAY by running it parallel to Phase 1.
        # The serial floor is the sum of Phase 1 LLM + tail LLM + 1 embed.
        # Require at least 30% of EMBED_DELAY shaved off vs serial.
        n_llm_calls = 5 * 1 + 1  # 1 combined call per chunk + 1 batched digest tail
        serial_floor = n_llm_calls * LLM_DELAY + EMBED_DELAY
        savings_target = EMBED_DELAY * 0.30
        assert elapsed < serial_floor - savings_target, (
            f"expected ≥{savings_target:.3f}s saved by parallelism; "
            f"elapsed={elapsed:.3f}s serial_floor={serial_floor:.3f}s"
        )
    finally:
        hy.close()


def test_background_embed_failure_falls_back_to_post_loop_fetch(cfg):
    """If the background embed task raises, the dream cycle must continue
    and the post-loop fetch_chunk_embeddings call must still embed the
    affected chunks."""
    from hymem.extraction.embeddings import StubEmbeddingClient as _Stub

    class FlakyEmbed(_Stub):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def embed(self, texts):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("simulated background failure")
            return super().embed(texts)

    embed = FlakyEmbed()
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
        assert embed.call_count >= 2

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
                conn.execute(
                    "INSERT INTO chunk_embeddings(chunk_id, vector_json, model, dim) "
                    "VALUES (?, ?, ?, ?)",
                    (cid, json.dumps(vec), embed.model, embed.dim),
                )

        hits = _vector_search(conn, embed, "anything", top_k=5, max_scan=2)
        assert len(hits) <= 2
        # Only the two most-recent chunks (c3, c4) may appear.
        assert all(h.chunk_id in {"c3", "c4"} for h in hits)
    finally:
        conn.close()
