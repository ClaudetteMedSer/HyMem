from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hymem import StubEmbeddingClient
from hymem.config import HyMemConfig
from hymem.core import db
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.extraction.triples import Triple


def _source(conn):
    conn.execute("INSERT INTO sessions(id) VALUES ('s')")
    mid = int(conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) "
        "VALUES ('s','user','Service uses Redis', '2026-06-01T12:00:00Z')"
    ).lastrowid)
    with db.transaction(conn):
        materialize_message_coverage(conn, "s")
    return mid


def _chunk(conn, mid, name):
    chunk = Chunk(
        name, "s", mid, mid, "test", "user: Service uses Redis",
        source_message_ids=(mid,),
    )
    with db.transaction(conn):
        persist_chunks(conn, [chunk])
    return chunk


def _vectors():
    model = embedding_storage_identity(StubEmbeddingClient(
        model_name="fixture-v1", dim_value=2,
    ))[0]
    vectors = phase1._PreparedDedupVectors(model=model, dim=2)
    vectors["service uses redis"] = [1.0, 0.0]
    vectors["service uses redis_cache"] = [1.0, 0.0]
    return vectors


def _persist(conn, chunk, triples, cfg, *, vectors=None, pool=None):
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    extraction = ChunkExtraction(
        triples=triples, markers=[],
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
    )
    with db.transaction(conn):
        staged = phase1.persist_chunk_results(
            conn, chunk, extraction, prompt_version="v14", cfg=cfg,
            dedup_vectors=vectors, in_cycle_edges=pool,
        )
    return staged


@pytest.mark.parametrize("difference", ["temporal", "typed", "polarity", "same"])
@pytest.mark.parametrize("route", ["samewave", "cached"])
def test_semantic_dedup_respects_source_interpretation(
    tmp_path, monkeypatch, route, difference,
):
    conn = db.connect(tmp_path / "claim.sqlite")
    db.initialize(conn)
    try:
        mid = _source(conn)
        cfg = replace(HyMemConfig(root=tmp_path),
                      triple_dedup_enabled=True,
                      triple_dedup_cosine_threshold=0.9)
        kwargs = {"source_message_id": mid, "temporal_scope": "2025"}
        if difference == "typed":
            kwargs["value_numeric"] = 65.0
            kwargs["value_unit"] = "percent"
        first = Triple("service", "uses", "redis", 1, **kwargs)
        second_kwargs = dict(kwargs)
        second_polarity = 1
        if difference == "temporal":
            second_kwargs["temporal_scope"] = "2026"
        elif difference == "typed":
            second_kwargs["value_numeric"] = 66.0
        elif difference == "polarity":
            second_polarity = -1
        second = Triple("service", "uses", "redis_cache", second_polarity,
                        **second_kwargs)
        screening = phase1._semantic_dedup_conflicts_with_claim_observation
        screened = []

        def observed_screen(*args, **kwargs):
            result = screening(*args, **kwargs)
            screened.append(result)
            return result

        monkeypatch.setattr(
            phase1, "_semantic_dedup_conflicts_with_claim_observation",
            observed_screen,
        )
        pool = phase1.new_in_cycle_pool()
        if route == "samewave":
            _persist(conn, _chunk(conn, mid, "both"), [first, second], cfg,
                     vectors=_vectors(), pool=pool)
        else:
            _persist(conn, _chunk(conn, mid, "first"), [first], cfg)
            with db.embedding_mutation(conn):
                conn.execute(
                    "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) "
                    "VALUES (?,?,?,?)",
                    ("service uses redis", json.dumps([1.0, 0.0]),
                     _vectors().model, 2),
                )
            _persist(conn, _chunk(conn, mid, "second"), [second], cfg,
                     vectors=_vectors(), pool=pool)

        edge_count = conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0]
        observation_count = conn.execute(
            "SELECT COUNT(*) FROM kg_claim_observations"
        ).fetchone()[0]
        assert screened and screened[-1] is (difference != "same")
        if difference == "same":
            assert edge_count == 1
            assert observation_count == (1 if route == "samewave" else 2)
        else:
            assert edge_count == 2
            assert observation_count == 2
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_exact_canonical_conflict_remains_atomic(tmp_path):
    conn = db.connect(tmp_path / "claim.sqlite")
    db.initialize(conn)
    try:
        mid = _source(conn)
        cfg = HyMemConfig(root=tmp_path)
        first = Triple("service", "uses", "redis", 1,
                       source_message_id=mid, temporal_scope="2025")
        other = Triple("service", "uses", "redis", 1,
                       source_message_id=mid, temporal_scope="2026")
        _persist(conn, _chunk(conn, mid, "first"), [first], cfg)
        before = [tuple(row) for row in conn.execute(
            "SELECT id,is_current,revision,interpretation_key FROM kg_evidence"
        )]
        pool = phase1.new_in_cycle_pool()
        vectors = _vectors()
        vectors["service uses memcache"] = [0.0, 1.0]
        staged = Triple("service", "uses", "memcache", 1,
                        source_message_id=mid)
        with pytest.raises(ValueError, match="same-generation"):
            _persist(conn, _chunk(conn, mid, "second"), [staged, other], cfg,
                     vectors=vectors, pool=pool)
        assert [tuple(row) for row in conn.execute(
            "SELECT id,is_current,revision,interpretation_key FROM kg_evidence"
        )] == before
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_claim_observations"
        ).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == 1
        assert pool == []
    finally:
        conn.close()


# Parent-added controls: exercise the runner's post-commit pool adoption,
# different-source independence, and exact replay after the new routing path.
@pytest.mark.parametrize("same_source", [True, False])
def test_cross_chunk_same_wave_and_replay(tmp_path, same_source):
    conn = db.connect(tmp_path / "parent.sqlite")
    db.initialize(conn)
    try:
        mid = _source(conn)
        other_mid = mid
        if not same_source:
            other_mid = int(conn.execute(
                "INSERT INTO messages(session_id,role,content,created_at) "
                "VALUES ('s','user','Service now uses Redis cache', "
                "'2026-06-02T12:00:00Z')"
            ).lastrowid)
            with db.transaction(conn):
                materialize_message_coverage(conn, "s")
        cfg = replace(HyMemConfig(root=tmp_path),
                      triple_dedup_cosine_threshold=0.9)
        vectors = _vectors()
        pool = phase1.new_in_cycle_pool()
        first = Triple("service", "uses", "redis", 1,
                       source_message_id=mid, temporal_scope="2025")
        second = Triple("service", "uses", "redis_cache", 1,
                        source_message_id=other_mid, temporal_scope="2026")
        pool[:] = _persist(conn, _chunk(conn, mid, "one"), [first], cfg,
                           vectors=vectors, pool=pool)
        assert len(pool) == 1
        chunk = _chunk(conn, other_mid, "two")
        pool[:] = _persist(conn, chunk, [second], cfg, vectors=vectors, pool=pool)
        assert conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == (
            2 if same_source else 1
        )
        assert conn.execute("SELECT COUNT(*) FROM kg_claim_observations").fetchone()[0] == 2
        before = [tuple(row) for row in conn.execute("SELECT * FROM kg_evidence ORDER BY id")]
        pool[:] = _persist(conn, chunk, [second], cfg, vectors=vectors, pool=pool)
        assert [tuple(row) for row in conn.execute("SELECT * FROM kg_evidence ORDER BY id")] == before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
