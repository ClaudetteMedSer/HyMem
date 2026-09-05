"""Regression tests for the v36 knowledge-graph evidence ledger."""

from __future__ import annotations

import dataclasses
import sqlite3

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import canonicalize, evidence, phase1, phase3
from hymem.dreaming.behavioral_dedup import (
    DuplicateMember,
    ProposedMerge,
    apply_behavioral_merges,
)
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.markers import Marker
from hymem.extraction.triples import Triple


def _chunk(hy: HyMem, chunk_id: str, text: str = "text", role: str = "user") -> Chunk:
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s_ledger')")
    mid = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s_ledger', ?, ?)",
        (role, text),
    ).lastrowid
    chunk = Chunk(
        chunk_id, "s_ledger", mid, mid, "test", f"{role}: {text}",
        source_message_ids=(int(mid),),
    )
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "s_ledger")
        persist_chunks(conn, [chunk])
    return chunk


def _persist(
    hy: HyMem,
    chunk: Chunk,
    triple: Triple,
    *,
    prompt_version: str,
    failed: bool = False,
) -> None:
    triple = dataclasses.replace(
        triple,
        source_message_id=(triple.source_message_id or chunk.start_message_id),
    )
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn,
            chunk,
            ChunkExtraction(
                triples=[triple],
                markers=[Marker("preference", "partial marker")] if failed else [],
                failed=failed,
                claim_sources={source.message_id: source for source in sources},
                source_validated=True,
            ),
            prompt_version=prompt_version,
            cfg=hy.config,
        )


def _edge_counts(hy: HyMem, obj: str) -> tuple[int, int]:
    row = hy.conn.execute(
        "SELECT pos_evidence, neg_evidence FROM knowledge_graph WHERE object_canonical = ?",
        (obj,),
    ).fetchone()
    return int(row["pos_evidence"]), int(row["neg_evidence"])


def test_prompt_version_replay_does_not_duplicate_weighted_evidence(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _chunk(hy, "c_prompt")
        triple = Triple("app", "uses", "postgres", 1)
        _persist(hy, chunk, triple, prompt_version="triples.v1")
        _persist(hy, chunk, triple, prompt_version="triples.v2")

        assert _edge_counts(hy, "postgres") == (2, 0)
        row = hy.conn.execute(
            """SELECT COUNT(*) AS n, evidence_weight, weight_source,
                      extraction_prompt_version, source_role
               FROM kg_evidence"""
        ).fetchone()
        assert row["n"] == 1
        assert row["evidence_weight"] == 2
        assert row["weight_source"] == "configured_role:user"
        assert row["extraction_prompt_version"] == "triples.v1"
        assert row["source_role"] == "user"
        assert evidence.count_mismatches(hy.conn) == []
    finally:
        hy.close()


def test_failed_partial_attempts_are_atomic_and_retry_cleanly(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _chunk(hy, "c_partial")
        triple = Triple("app", "uses", "redis", 1)
        _persist(hy, chunk, triple, prompt_version="triples.v1", failed=True)
        _persist(hy, chunk, triple, prompt_version="triples.v1", failed=True)

        assert hy.conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM behavioral_markers").fetchone()[0] == 0

        _persist(hy, chunk, triple, prompt_version="triples.v1")
        assert _edge_counts(hy, "redis") == (2, 0)
        assert hy.conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 1
        assert evidence.count_mismatches(hy.conn) == []
    finally:
        hy.close()


def test_latest_successful_polarity_replaces_same_source_assertion(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _chunk(hy, "c_flip")
        _persist(hy, chunk, Triple("app", "uses", "sqlite", 1), prompt_version="v1")
        _persist(hy, chunk, Triple("app", "uses", "sqlite", -1), prompt_version="v2")
        assert _edge_counts(hy, "sqlite") == (0, 2)

        row = hy.conn.execute(
            "SELECT polarity, evidence_weight, extraction_prompt_version FROM kg_evidence "
            "WHERE is_current=1"
        ).fetchone()
        assert (row["polarity"], row["evidence_weight"], row["extraction_prompt_version"]) == (
            -1,
            2,
            "v2",
        )

        _persist(hy, chunk, Triple("app", "uses", "sqlite", 1), prompt_version="v3")
        assert _edge_counts(hy, "sqlite") == (2, 0)
        assert hy.conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 3
        assert evidence.count_mismatches(hy.conn) == []
    finally:
        hy.close()


def test_phase3_reinforcement_and_decay_are_idempotent_per_chunk(cfg):
    hy = HyMem(cfg)
    try:
        source = _chunk(hy, "c_source", "app uses postgres")
        _persist(hy, source, Triple("app", "uses", "postgres", 1), prompt_version="v1")
        edge_id = hy.conn.execute(
            "SELECT id FROM knowledge_graph WHERE object_canonical = 'postgres'"
        ).fetchone()["id"]

        reinforcement = _chunk(hy, "c_reinforce", "app and postgres remain paired")
        older_reinforcement = _chunk(
            hy, "c_reinforce_a", "app and postgres were also paired here"
        )
        hy.conn.executemany(
            "INSERT INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, ?)",
            [
                (reinforcement.id, "app"),
                (reinforcement.id, "postgres"),
                (older_reinforcement.id, "app"),
                (older_reinforcement.id, "postgres"),
            ],
        )
        phase3.reinforce(hy.conn, cfg)
        phase3.reinforce(hy.conn, cfg)
        assert _edge_counts(hy, "postgres") == (3, 0)

        hy.conn.execute(
            "UPDATE knowledge_graph SET last_reinforced = datetime('now', '-90 days') "
            "WHERE id = ?",
            (edge_id,),
        )
        decay_chunk = _chunk(hy, "c_decay", "the app changed substantially")
        older_decay_chunk = _chunk(hy, "c_decay_a", "the app had changed before")
        hy.conn.execute(
            "INSERT INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, 'app')",
            (decay_chunk.id,),
        )
        hy.conn.execute(
            "INSERT INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, 'app')",
            (older_decay_chunk.id,),
        )
        phase3.decay(hy.conn, cfg)
        phase3.decay(hy.conn, cfg)
        assert _edge_counts(hy, "postgres") == (3, 1)

        kinds = hy.conn.execute(
            "SELECT evidence_kind, COUNT(*) AS n FROM kg_evidence "
            "WHERE edge_id = ? GROUP BY evidence_kind ORDER BY evidence_kind",
            (edge_id,),
        ).fetchall()
        assert [(r["evidence_kind"], r["n"]) for r in kinds] == [
            ("decay", 1),
            ("extraction", 1),
            ("reinforcement", 1),
        ]
        assert evidence.count_mismatches(hy.conn) == []
    finally:
        hy.close()


def test_manual_retraction_has_chunkless_audit_signal_and_is_idempotent(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _chunk(hy, "c_manual")
        _persist(hy, chunk, Triple("app", "uses", "mysql", 1), prompt_version="v1")

        assert hy.retract_edge("app", "uses", "mysql") is True
        assert hy.retract_edge("app", "uses", "mysql") is False
        assert _edge_counts(hy, "mysql") == (2, 1)
        signal = hy.conn.execute(
            "SELECT signal_kind, polarity, evidence_weight FROM kg_evidence_signals"
        ).fetchone()
        assert (signal["signal_kind"], signal["polarity"], signal["evidence_weight"]) == (
            "manual_retraction",
            -1,
            1,
        )
        assert evidence.count_mismatches(hy.conn) == []
    finally:
        hy.close()


def test_evidence_history_prevents_chunk_cascade_deletion(cfg):
    hy = HyMem(cfg)
    try:
        first = _chunk(hy, "c_delete_1")
        second = _chunk(hy, "c_delete_2")
        triple = Triple("app", "uses", "duckdb", 1)
        _persist(hy, first, triple, prompt_version="v1")
        _persist(hy, second, triple, prompt_version="v1")
        assert _edge_counts(hy, "duckdb") == (4, 0)

        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute("DELETE FROM chunks WHERE id = ?", (first.id,))

        assert _edge_counts(hy, "duckdb") == (4, 0)
        assert hy.conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 2
        assert evidence.count_mismatches(hy.conn) == []
    finally:
        hy.close()


def _two_alias_edges_with_one_source(hy: HyMem) -> tuple[int, int, Chunk]:
    chunk = _chunk(hy, "c_merge")
    survivor = hy.conn.execute(
        """INSERT INTO knowledge_graph(
               subject_canonical, predicate, object_canonical, pos_evidence, neg_evidence
           ) VALUES ('app', 'uses', 'docker', 0, 0)"""
    ).lastrowid
    member = hy.conn.execute(
        """INSERT INTO knowledge_graph(
               subject_canonical, predicate, object_canonical, pos_evidence, neg_evidence
           ) VALUES ('app', 'uses', 'docker_old', 0, 0)"""
    ).lastrowid
    for edge_id in (survivor, member):
        evidence.record_chunk_evidence(
            hy.conn,
            edge_id=edge_id,
            chunk_id=chunk.id,
            evidence_kind="extraction",
            polarity=1,
            evidence_weight=1,
            weight_source="test",
        )
    return survivor, member, chunk


def test_canonical_merge_deduplicates_overlapping_source_provenance(cfg):
    hy = HyMem(cfg)
    try:
        survivor, _member, _chunk_row = _two_alias_edges_with_one_source(hy)
        with core_db.transaction(hy.conn):
            canonicalize.merge(hy.conn, keep="docker", drop="docker_old")

        rows = hy.conn.execute(
            "SELECT id, pos_evidence FROM knowledge_graph WHERE object_canonical = 'docker'"
        ).fetchall()
        assert [(row["id"], row["pos_evidence"]) for row in rows] == [(survivor, 1)]
        assert hy.conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 1
        assert evidence.count_mismatches(hy.conn) == []
    finally:
        hy.close()


def test_behavioral_merge_deduplicates_overlapping_source_provenance(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _chunk(hy, "c_behavior_merge")
        ids = []
        for obj in ("concise", "concise_mode"):
            edge_id = hy.conn.execute(
                """INSERT INTO knowledge_graph(
                       subject_canonical, predicate, object_canonical,
                       pos_evidence, neg_evidence
                   ) VALUES ('user', 'prefers', ?, 0, 0)""",
                (obj,),
            ).lastrowid
            ids.append(edge_id)
            evidence.record_chunk_evidence(
                hy.conn,
                edge_id=edge_id,
                chunk_id=chunk.id,
                evidence_kind="extraction",
                polarity=1,
                evidence_weight=1,
                weight_source="test",
            )
        proposal = ProposedMerge(
            subject="user",
            predicate="prefers",
            survivor_id=ids[0],
            survivor_object="concise",
            survivor_pos=1,
            survivor_neg=0,
            members=[DuplicateMember(ids[1], "concise_mode", 1, 0, 0.99)],
        )
        with core_db.transaction(hy.conn):
            apply_behavioral_merges(hy.conn, [proposal])

        assert hy.conn.execute(
            "SELECT pos_evidence FROM knowledge_graph WHERE id = ?", (ids[0],)
        ).fetchone()["pos_evidence"] == 1
        assert hy.conn.execute(
            "SELECT 1 FROM knowledge_graph WHERE id = ?", (ids[1],)
        ).fetchone() is None
        assert hy.conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 1
        assert evidence.count_mismatches(hy.conn) == []
    finally:
        hy.close()


def test_integrity_check_ignores_computed_derived_edges(cfg):
    hy = HyMem(cfg)
    try:
        derived_id = hy.conn.execute(
            """INSERT INTO knowledge_graph(
                   subject_canonical, predicate, object_canonical,
                   pos_evidence, neg_evidence, derived
               ) VALUES ('a', 'depends_on', 'b', 1, 0, 1)"""
        ).lastrowid
        direct_id = hy.conn.execute(
            """INSERT INTO knowledge_graph(
                   subject_canonical, predicate, object_canonical,
                   pos_evidence, neg_evidence, derived
               ) VALUES ('x', 'uses', 'y', 1, 0, 0)"""
        ).lastrowid

        mismatches = evidence.count_mismatches(hy.conn)
        assert [item["edge_id"] for item in mismatches] == [direct_id]
        assert derived_id not in {item["edge_id"] for item in mismatches}
    finally:
        hy.close()
