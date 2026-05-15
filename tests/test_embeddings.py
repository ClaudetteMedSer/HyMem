from __future__ import annotations

import json
import math

from hymem import HyMem, StubEmbeddingClient
from hymem.extraction.embeddings import CachedEmbeddingClient
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
