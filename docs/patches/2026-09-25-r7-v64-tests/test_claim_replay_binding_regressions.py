"""Exact claim replay must stay bound to its published routing decision."""

from __future__ import annotations

import json
from dataclasses import replace

from hymem import HyMemConfig
from hymem.core import db
from hymem.dreaming import evidence, phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.triples import Triple
from tests.test_claim_semantic_dedup_guard import _chunk, _persist, _source, _vectors


def _second_source(conn) -> int:
    message_id = int(conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) "
        "VALUES ('s','user','Service uses Redis cache', "
        "'2026-06-02T12:00:00Z')"
    ).lastrowid)
    with db.transaction(conn):
        materialize_message_coverage(conn, "s")
    return message_id


def _config(tmp_path) -> HyMemConfig:
    return replace(
        HyMemConfig(root=tmp_path),
        triple_dedup_enabled=True,
        triple_dedup_cosine_threshold=0.9,
    )


def _claim_state(conn, chunk_id: str) -> tuple[list[str], list[tuple]]:
    return (
        list(conn.iterdump()),
        [tuple(row) for row in conn.execute(
            "SELECT edge_id,source_message_id,polarity,interpretation_key "
            "FROM kg_claim_observations WHERE chunk_id=? "
            "ORDER BY source_message_id,edge_id", (chunk_id,),
        )],
    )


def _assert_published(conn, chunk_id: str, expected_observations: int) -> None:
    row = conn.execute(
        "SELECT outcome.result_hash,outcome.phase1_generation_key,"
        "chunk.source_manifest_count,"
        "hymem_phase1_generation_is_authorized("
        "generation.generation_key,generation.identity_exact) AS authorized "
        "FROM kg_claim_extraction_outcomes outcome "
        "JOIN chunks chunk ON chunk.id=outcome.chunk_id "
        "JOIN phase1_generations generation "
        "ON generation.generation_key=outcome.phase1_generation_key "
        "WHERE outcome.chunk_id=?", (chunk_id,),
    ).fetchone()
    assert row is not None and row["authorized"] == 1
    assert row["source_manifest_count"] > 0
    assert row["result_hash"] == evidence.claim_observation_result_hash(
        conn, chunk_id,
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM kg_claim_observations WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()[0] == expected_observations


def test_cached_alias_replay_survives_vector_cache_change(tmp_path):
    conn = db.connect(tmp_path / "cached-alias.sqlite")
    db.initialize(conn)
    try:
        first_source = _source(conn)
        second_source = _second_source(conn)
        cfg = _config(tmp_path)
        vectors = _vectors()
        exact = Triple(
            "service", "uses", "redis", 1, source_message_id=first_source,
        )
        alias = Triple(
            "service", "uses", "redis_cache", 1,
            source_message_id=second_source,
        )
        _persist(conn, _chunk(conn, first_source, "seed"), [exact], cfg)
        with db.embedding_mutation(conn):
            conn.execute(
                "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) "
                "VALUES (?,?,?,?)",
                ("service uses redis", json.dumps([1.0, 0.0]), vectors.model, 2),
            )

        chunk = _chunk(conn, second_source, "cached-alias")
        _persist(conn, chunk, [alias], cfg, vectors=vectors)
        assert conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == 1
        _assert_published(conn, chunk.id, 1)
        first = _claim_state(conn, chunk.id)
        _persist(conn, chunk, [alias], cfg, vectors=vectors)
        assert _claim_state(conn, chunk.id) == first

        # The published claim is the same even when the optional vector cache
        # no longer offers its earlier semantic route.
        with db.embedding_mutation(conn):
            conn.execute("DELETE FROM edge_embeddings WHERE edge_text=?", (
                "service uses redis",
            ))
        before = _claim_state(conn, chunk.id)
        _persist(conn, chunk, [alias], cfg, vectors=vectors)
        assert _claim_state(conn, chunk.id) == before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_same_wave_alias_then_two_exact_claims_replay_with_empty_pool(tmp_path):
    conn = db.connect(tmp_path / "batch-order.sqlite")
    db.initialize(conn)
    try:
        first_source = _source(conn)
        second_source = _second_source(conn)
        chunk = Chunk(
            "ordered-batch", "s", first_source, second_source, "test",
            "user: Service uses Redis and Redis cache",
            source_message_ids=(first_source, second_source),
        )
        with db.transaction(conn):
            persist_chunks(conn, [chunk])
        cfg = _config(tmp_path)
        vectors = _vectors()
        triples = [
            Triple(
                "service", "uses", "redis_cache", 1,
                source_message_id=second_source, value_numeric=66.0,
                value_unit="percent",
            ),
            Triple(
                "service", "uses", "redis", 1,
                source_message_id=first_source, value_numeric=65.0,
                value_unit="percent",
            ),
            Triple(
                "service", "uses", "redis", 1,
                source_message_id=second_source, value_numeric=65.0,
                value_unit="percent",
            ),
        ]
        pool = phase1.new_in_cycle_pool()
        pool[:] = _persist(conn, chunk, triples, cfg, vectors=vectors, pool=pool)
        rows = conn.execute(
            "SELECT kg.object_canonical,observation.source_message_id "
            "FROM kg_claim_observations observation "
            "JOIN knowledge_graph kg ON kg.id=observation.edge_id "
            "WHERE observation.chunk_id=? "
            "ORDER BY observation.source_message_id,kg.object_canonical",
            (chunk.id,),
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("redis_cache", first_source),
            ("redis", second_source),
            ("redis_cache", second_source),
        ]
        _assert_published(conn, chunk.id, 3)

        # A later dream starts with no in-cycle pool. It must still recognize
        # the exact same published input and preserve the earlier route.
        before = _claim_state(conn, chunk.id)
        _persist(
            conn, chunk, triples, cfg, vectors=vectors,
            pool=phase1.new_in_cycle_pool(),
        )
        assert _claim_state(conn, chunk.id) == before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
