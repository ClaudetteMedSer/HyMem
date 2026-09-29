"""Cold imported semantic claims must replay from durable citations without vectors."""

from __future__ import annotations

from dataclasses import replace
import json
import re
import sqlite3

import pytest

from hymem import HyMemConfig, portability
from hymem.core import db
from hymem.dreaming import evidence, phase1
from hymem.dreaming.chunks import load_pending_persisted_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.contract import extraction_cache_key
from hymem.extraction.producer import phase1_generation_binding
from tests.test_claim_semantic_dedup_guard import _chunk, _vectors
from tests.test_phase1_producer_identity import _DeclaredLLM


_CLAIM_TABLES = (
    "knowledge_graph", "kg_evidence", "kg_claim_observations",
    "kg_claim_extraction_outcomes", "kg_edge_lifecycle",
)


def _claim_snapshot(conn):
    return {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
        for table in _CLAIM_TABLES
    }


def _replay_kwargs(state):
    return dict(
        prompt_version=extraction_cache_key("v14"),
        phase1_generation_key=state["generation"]["generation_key"],
        cfg=state["cfg"], dedup_vectors=None, in_cycle_edges=None,
    )


def _historical_match(state, extraction=None, *, cfg=None):
    """Exercise only the new NULL-proof matcher; legacy fallback is separate."""
    return phase1._matches_historical_citations(
        state["target"], state["alias_chunk"],
        extraction if extraction is not None else state["replay"],
        cfg=cfg if cfg is not None else state["cfg"],
        result_hash=state["outcome_hash"],
    )


def _clone_row(conn, table, row, **changes):
    """Clone one fixture row with explicit replacements under DB mutation guards."""
    keys = [key for key in row.keys() if key != "id"]
    values = [changes.get(key, row[key]) for key in keys]
    return conn.execute(
        f"INSERT INTO {table}({','.join(keys)}) "
        f"VALUES ({','.join('?' for _ in keys)})", values,
    ).lastrowid


@pytest.fixture
def cold_alias(tmp_path):
    origin = db.connect(tmp_path / "origin.sqlite")
    target = db.connect(tmp_path / "imported.sqlite")
    for conn in (origin, target):
        db.initialize(conn)
    try:
        origin.execute("INSERT INTO sessions(id) VALUES ('s')")
        mids = [int(origin.execute(
            "INSERT INTO messages(session_id,role,content,created_at) "
            "VALUES ('s','user',?,?)", (content, created_at),
        ).lastrowid) for content, created_at in (
            ("App uses Redis", "2026-06-01T12:00:00Z"),
            ("App uses Redis cache", "2026-06-02T12:00:00Z"),
        )]
        with db.transaction(origin):
            materialize_message_coverage(origin, "s")
        chunks = [_chunk(origin, mid, f"cold-normal-{mid}") for mid in mids]
        cfg = replace(
            HyMemConfig(root=tmp_path), triple_dedup_enabled=True,
            triple_dedup_cosine_threshold=0.9,
        )
        client = _DeclaredLLM("cold-replay-model", "redis")
        generation = phase1_generation_binding("v14", client)
        vectors = _vectors()
        vectors["app uses redis"] = [1.0, 0.0]
        vectors["app uses redis_cache"] = [1.0, 0.0]
        for index, chunk in enumerate(chunks):
            client.object_name = "redis" if index == 0 else "redis_cache"
            extraction = phase1.extract_chunk_results(
                origin, chunk, client, prompt_version="v14",
                phase1_generation=generation,
            )
            assert extraction is not None and extraction.source_validated
            with db.transaction(origin):
                phase1.persist_chunk_results(
                    origin, chunk, extraction, prompt_version="v14",
                    cfg=cfg, dedup_vectors=vectors,
                )
            if index == 0:
                with db.embedding_mutation(origin):
                    origin.execute(
                        "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) "
                        "VALUES (?,?,?,?)",
                        ("app uses redis", json.dumps([1.0, 0.0]), vectors.model, 2),
                    )
        assert origin.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == 1
        assert origin.execute(
            "SELECT COUNT(*) FROM kg_claim_observations"
        ).fetchone()[0] == 2
        wire = tmp_path / "claims.jsonl"
        portability.export_jsonl(origin, wire)
        assert "local_replay_proof" not in wire.read_text()
        portability.import_jsonl(target, wire)
        assert target.execute("SELECT COUNT(*) FROM edge_embeddings").fetchone()[0] == 0
        assert target.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes "
            "WHERE local_replay_proof IS NOT NULL"
        ).fetchone()[0] == 0
        assert target.execute("SELECT COUNT(*) FROM current_phase1_publications").fetchone()[0] == 0
        pending = load_pending_persisted_chunks(
            target, "s", prompt_version="v14", limit=10,
            phase1_generation_key=generation["generation_key"],
        )
        assert {chunk.id for chunk in pending} == {chunk.id for chunk in chunks}
        client.object_name = "redis_cache"
        replay = phase1.extract_chunk_results(
            target, chunks[1], client, prompt_version="v14",
            phase1_generation=generation,
        )
        assert replay is not None and replay.source_validated
        assert replay.claim_sources == {
            source.message_id: source
            for source in phase1._claim_sources_for_chunk(target, chunks[1])
        }
        outcome_hash = target.execute(
            "SELECT result_hash FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (chunks[1].id,),
        ).fetchone()[0]
        yield dict(
            origin=origin, target=target, chunks=chunks,
            alias_chunk=chunks[1], replay=replay, cfg=cfg,
            generation=generation, outcome_hash=outcome_hash, wire=wire,
        )
    finally:
        origin.close()
        target.close()


def test_normal_extractor_replays_cold_imported_alias_without_claim_churn(cold_alias):
    state = cold_alias
    conn, chunk = state["target"], state["alias_chunk"]
    before = _claim_snapshot(conn)
    statements = []
    claim_writes = []

    def authorize(action, table, column, database, trigger):
        if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
                      sqlite3.SQLITE_DELETE) and table in _CLAIM_TABLES:
            claim_writes.append((action, table, column, trigger))
        return sqlite3.SQLITE_OK

    conn.set_trace_callback(statements.append)
    conn.set_authorizer(authorize)
    try:
        with db.transaction(conn):
            phase1.persist_chunk_results(
                conn, chunk, state["replay"], prompt_version="v14",
                cfg=state["cfg"], dedup_vectors=None,
                in_cycle_edges=phase1.new_in_cycle_pool(),
            )
    finally:
        conn.set_authorizer(None)
        conn.set_trace_callback(None)
    assert _claim_snapshot(conn) == before
    assert claim_writes == []
    assert not [statement for statement in statements if re.search(
        r"^\s*(?:--[^\n]*\n\s*)*(?:INSERT|UPDATE|DELETE|REPLACE)\b"
        r"(?:\s+OR\s+\w+)?\s+(?:INTO|FROM)?\s*"
        r"(?:knowledge_graph|kg_evidence|kg_claim_observations|"
        r"kg_claim_extraction_outcomes|kg_edge_lifecycle)\b",
        statement, re.IGNORECASE,
    )]
    assert conn.execute(
        "SELECT local_replay_proof FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()[0] is None
    assert conn.execute(
        "SELECT phase1_generation_key FROM processed_chunks WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()[0] == state["generation"]["generation_key"]
    assert conn.execute("SELECT COUNT(*) FROM current_phase1_publications").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert phase1.extract_chunk_results(
        conn, chunk, _DeclaredLLM("cold-replay-model", "redis_cache"),
        prompt_version="v14", phase1_generation=state["generation"],
    ) is None
    before_repeat = list(conn.iterdump())
    with db.transaction(conn):
        phase1.persist_chunk_results(
            conn, chunk, state["replay"], prompt_version="v14",
            cfg=state["cfg"], dedup_vectors=None,
        )
    assert list(conn.iterdump()) == before_repeat


@pytest.mark.parametrize("field,value", (
    ("content", "different message bytes"),
    ("role", "assistant"),
    ("source_created_at", "2025-01-01T00:00:00Z"),
    ("source_peer_id", "other-peer"),
    ("source_workspace_id", "other-workspace"),
    ("chunk_id", "other-coverage-artifact"),
))
def test_historical_matcher_refuses_changed_source_even_if_legacy_fallback_does_not(
    cold_alias, field, value,
):
    state = cold_alias
    source_id = state["replay"].triples[0].source_message_id
    source = state["replay"].claim_sources[source_id]
    altered = replace(state["replay"], claim_sources={
        **state["replay"].claim_sources,
        source_id: replace(source, **{field: value}),
    })
    assert not _historical_match(state, altered)
    # A forged source_validated object may still satisfy the pre-existing
    # semantic fallback. This focused check makes no whole-function claim.


@pytest.mark.parametrize("field,value", (
    ("subject", "other-app"), ("predicate", "contains"),
    ("object", "other-cache"), ("polarity", -1),
    ("value_text", "changed"), ("value_numeric", 4.0),
    ("value_unit", "ms"), ("temporal_scope", "2025"),
))
def test_historical_matcher_refuses_changed_claim(cold_alias, field, value):
    state = cold_alias
    altered = replace(state["replay"], triples=[
        replace(state["replay"].triples[0], **{field: value}),
    ])
    assert not _historical_match(state, altered)


def test_historical_matcher_requires_exact_source_and_observation_multisets(cold_alias):
    state = cold_alias
    extraction = state["replay"]
    assert _historical_match(state)
    cited = extraction.triples[0].source_message_id
    assert len(extraction.claim_sources) == 1
    assert not _historical_match(state, replace(extraction, claim_sources={}))
    assert not _historical_match(state, replace(extraction, claim_sources={
        **extraction.claim_sources, 999999: extraction.claim_sources[cited],
    }))
    assert not _historical_match(state, replace(extraction, triples=[]))
    assert not _historical_match(state, replace(extraction, triples=extraction.triples * 2))
    assert not _historical_match(state, replace(extraction, triples=[
        replace(extraction.triples[0], source_message_id=999999),
    ]))


def test_historical_matcher_refuses_changed_weight_and_chunk_metadata(cold_alias):
    state = cold_alias
    cfg = replace(state["cfg"], evidence_role_weights={"user": 7})
    assert not _historical_match(state, cfg=cfg)
    altered = dict(state, alias_chunk=replace(state["alias_chunk"], text="different chunk"))
    assert not _historical_match(altered)


def test_generation_boundary_still_precedes_historical_match(cold_alias):
    state = cold_alias
    args = _replay_kwargs(state)
    args["phase1_generation_key"] = "sha256:" + "0" * 64
    assert not phase1._is_exact_published_replay(
        state["target"], state["alias_chunk"], state["replay"], **args,
    )


@pytest.mark.parametrize("surface", ("redis_cache", "redis_alt"))
def test_historical_matcher_refuses_ambiguous_or_extra_durable_observation(
    cold_alias, surface,
):
    """Two healthy observations cannot be covered by one incoming claim.

    The matching surface gives one input two possible historical edges. A
    different surface leaves an extra historical observation. This scoped
    synthetic history is rolled back; no schema guards are disabled.
    """
    state = cold_alias
    conn, chunk = state["target"], state["alias_chunk"]
    before = list(conn.iterdump())
    conn.execute("SAVEPOINT extra_historical_observation")
    try:
        observed = conn.execute(
            "SELECT * FROM kg_claim_observations WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()
        cited = conn.execute(
            "SELECT * FROM kg_evidence WHERE id=?", (observed["evidence_id"],),
        ).fetchone()
        lifecycle = conn.execute(
            "SELECT * FROM kg_edge_lifecycle WHERE source_evidence_id=? "
            "AND event_kind='claim_assertion'", (cited["id"],),
        ).fetchone()
        assert lifecycle is not None
        edge_id = conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical," 
            "pos_evidence,neg_evidence) VALUES ('app','uses','redis_alt',0,0)"
        ).lastrowid
        interpretation = evidence._interpretation_key(
            polarity=cited["polarity"],
            evidence_weight=cited["evidence_weight"],
            weight_source=cited["weight_source"],
            source_role=cited["source_role"],
            surface_subject=cited["surface_subject"],
            surface_object=surface,
            value_text=cited["value_text"],
            value_numeric=cited["value_numeric"],
            value_unit=cited["value_unit"],
            temporal_scope=cited["temporal_scope"],
        )
        with db.evidence_history_mutation(conn):
            evidence_id = _clone_row(
                conn, "kg_evidence", cited, edge_id=edge_id,
                surface_object=surface, interpretation_key=interpretation,
            )
            _clone_row(
                conn, "kg_edge_lifecycle", lifecycle, edge_id=edge_id,
                source_evidence_id=evidence_id,
            )
            _clone_row(
                conn, "kg_claim_observations", observed, edge_id=edge_id,
                evidence_id=evidence_id, interpretation_key=interpretation,
            )
            result_hash = evidence.claim_observation_result_hash(conn, chunk.id)
            conn.execute(
                "UPDATE kg_claim_extraction_outcomes SET result_hash=? WHERE chunk_id=?",
                (result_hash, chunk.id),
            )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        doubled = dict(state, outcome_hash=result_hash)
        assert not _historical_match(doubled)
        assert not phase1._is_exact_published_replay(
            conn, chunk, state["replay"], **_replay_kwargs(state),
        )
    finally:
        conn.execute("ROLLBACK TO extra_historical_observation")
        conn.execute("RELEASE extra_historical_observation")
    assert list(conn.iterdump()) == before


def test_missing_assertion_lifecycle_never_qualifies_as_cold_replay(cold_alias):
    state = cold_alias
    conn = state["target"]
    evidence_id = conn.execute(
        "SELECT evidence_id FROM kg_claim_observations WHERE chunk_id=?",
        (state["alias_chunk"].id,),
    ).fetchone()[0]
    with db.evidence_history_mutation(conn):
        conn.execute(
            "DELETE FROM kg_edge_lifecycle WHERE source_evidence_id=?",
            (evidence_id,),
        )
    assert not phase1._is_exact_published_replay(
        conn, state["alias_chunk"], state["replay"], **_replay_kwargs(state),
    )


@pytest.mark.parametrize("field,value", (
    ("evidence_weight", 7),
    ("weight_source", "changed-weight-policy"),
    ("surface_object", "other-cache"),
))
def test_historical_matcher_refuses_damaged_cited_evidence_metadata(
    cold_alias, field, value,
):
    """A scoped evidence fault cannot be treated as an exact citation replay."""
    state = cold_alias
    conn, chunk = state["target"], state["alias_chunk"]
    evidence_id = conn.execute(
        "SELECT evidence_id FROM kg_claim_observations WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()[0]
    before = list(conn.iterdump())
    conn.execute("SAVEPOINT damaged_citation")
    try:
        with db.evidence_history_mutation(conn):
            conn.execute(
                f"UPDATE kg_evidence SET {field}=? WHERE id=?",
                (value, evidence_id),
            )
        assert not _historical_match(state)
        assert not phase1._is_exact_published_replay(
            conn, chunk, state["replay"], **_replay_kwargs(state),
        )
    finally:
        conn.execute("ROLLBACK TO damaged_citation")
        conn.execute("RELEASE damaged_citation")
    assert list(conn.iterdump()) == before


@pytest.mark.parametrize("field,value", (
    ("source_role", "assistant"),
    ("source_created_at", "2025-01-01T00:00:00Z"),
    ("source_peer_id", "other-peer"),
    ("source_workspace_id", "other-workspace"),
))
def test_durable_source_metadata_guards_reject_citation_damage(cold_alias, field, value):
    """Published source identity is protected by DB guards before replay."""
    state = cold_alias
    conn, chunk = state["target"], state["alias_chunk"]
    evidence_id = conn.execute(
        "SELECT evidence_id FROM kg_claim_observations WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()[0]
    before = list(conn.iterdump())
    with pytest.raises(sqlite3.IntegrityError):
        with db.evidence_history_mutation(conn):
            conn.execute(
                f"UPDATE kg_evidence SET {field}=? WHERE id=?",
                (value, evidence_id),
            )
    assert list(conn.iterdump()) == before


def test_damaged_cited_evidence_clock_fails_outer_authority_gate(cold_alias):
    """Clock health is checked before the NULL-proof matcher is consulted."""
    state = cold_alias
    conn, chunk = state["target"], state["alias_chunk"]
    before = list(conn.iterdump())
    conn.execute("SAVEPOINT damaged_citation_clock")
    try:
        with db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_claim_observations SET observed_at=? WHERE chunk_id=?",
                ("2099-01-01T00:00:00.000Z", chunk.id),
            )
        assert not phase1._is_exact_published_replay(
            conn, chunk, state["replay"], **_replay_kwargs(state),
        )
    finally:
        conn.execute("ROLLBACK TO damaged_citation_clock")
        conn.execute("RELEASE damaged_citation_clock")
    assert list(conn.iterdump()) == before


def test_auxiliary_repair_is_independent_of_claim_history(cold_alias):
    state = cold_alias
    conn = state["target"]
    before = _claim_snapshot(conn)
    enriched = replace(state["replay"], entity_type_hints={"app": "service"})
    with db.transaction(conn):
        phase1.persist_chunk_results(
            conn, state["alias_chunk"], enriched,
            prompt_version="v14", cfg=state["cfg"], dedup_vectors=None,
        )
    assert _claim_snapshot(conn) == before
    assert conn.execute(
        "SELECT type FROM entity_types WHERE entity_canonical='app'"
    ).fetchone()[0] == "service"
    assert conn.execute(
        "SELECT local_replay_proof FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (state["alias_chunk"].id,),
    ).fetchone()[0] is None


class _EmptyDeclaredLLM(_DeclaredLLM):
    def complete(self, request):
        self.calls.append(request)
        return '{"triples":[],"markers":[],"complete":true}'


def test_cold_imported_empty_outcome_replays_without_inventing_proof(tmp_path):
    origin = db.connect(tmp_path / "empty-origin.sqlite")
    target = db.connect(tmp_path / "empty-target.sqlite")
    for conn in (origin, target):
        db.initialize(conn)
    try:
        origin.execute("INSERT INTO sessions(id) VALUES ('s')")
        mid = int(origin.execute(
            "INSERT INTO messages(session_id,role,content,created_at) "
            "VALUES ('s','user','No durable claim','2026-06-01T12:00:00Z')"
        ).lastrowid)
        with db.transaction(origin):
            materialize_message_coverage(origin, "s")
        chunk = _chunk(origin, mid, "empty-cold")
        cfg = HyMemConfig(root=tmp_path)
        client = _EmptyDeclaredLLM("empty-cold-model", "unused")
        generation = phase1_generation_binding("v14", client)
        extraction = phase1.extract_chunk_results(
            origin, chunk, client, prompt_version="v14",
            phase1_generation=generation,
        )
        assert extraction is not None and extraction.source_validated
        assert extraction.triples == []
        with db.transaction(origin):
            phase1.persist_chunk_results(
                origin, chunk, extraction, prompt_version="v14", cfg=cfg,
            )
        wire = tmp_path / "empty.jsonl"
        portability.export_jsonl(origin, wire)
        portability.import_jsonl(target, wire)
        assert target.execute(
            "SELECT local_replay_proof FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] is None
        replay = phase1.extract_chunk_results(
            target, chunk, client, prompt_version="v14",
            phase1_generation=generation,
        )
        assert replay is not None and replay.triples == []
        before = _claim_snapshot(target)
        with db.transaction(target):
            phase1.persist_chunk_results(
                target, chunk, replay, prompt_version="v14", cfg=cfg,
            )
        assert _claim_snapshot(target) == before
        assert target.execute(
            "SELECT phase1_generation_key FROM processed_chunks WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == generation["generation_key"]
        assert target.execute(
            "SELECT local_replay_proof FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] is None
    finally:
        origin.close()
        target.close()
