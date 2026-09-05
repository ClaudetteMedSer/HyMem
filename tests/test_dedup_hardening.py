from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.triples import Triple


def _chunk(hy, chunk_id: str) -> Chunk:
    hy.conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('dedup-hard')")
    message_id = hy.conn.execute(
        "INSERT INTO messages(session_id,role,content) "
        "VALUES ('dedup-hard','assistant',?)",
        (chunk_id,),
    ).lastrowid
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text) VALUES (?,'dedup-hard',?,?,?,?)",
        (chunk_id, message_id, message_id, "test", f"assistant: {chunk_id}"),
    )
    return Chunk(
        chunk_id, "dedup-hard", message_id, message_id, "test",
        f"assistant: {chunk_id}",
    )


def _publish_claim(
    hy,
    *,
    chunk_id: str,
    polarity: int,
    object_: str = "redis",
    weight: int = 1,
) -> int:
    session_id = f"session-{chunk_id}"
    hy.open_session(session_id, source_workspace_id="w")
    message_id = hy.log_message(
        session_id, "user", f"service uses {object_}",
        source_peer_id="p", source_workspace_id="w",
    )
    chunk = Chunk(
        chunk_id, session_id, message_id, message_id, "test",
        f"user: service uses {object_}", source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [chunk])
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    cfg = replace(hy.config, evidence_role_weights={"user": weight})
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn,
            chunk,
            ChunkExtraction(
                triples=[Triple(
                    "service", "uses", object_, polarity,
                    source_message_id=message_id,
                )],
                markers=[],
                claim_sources={source.message_id: source for source in sources},
                source_validated=True,
            ),
            prompt_version="dedup-hard-v1",
            cfg=cfg,
        )
    return int(hy.conn.execute(
        "SELECT id FROM knowledge_graph WHERE subject_canonical='service' "
        "AND predicate='uses' AND object_canonical=?", (object_,),
    ).fetchone()[0])


def _embedding(hy, edge_id: int, *, model: str = "fake", vector=None) -> None:
    vector = [1.0, 0.0] if vector is None else vector
    row = hy.conn.execute(
        "SELECT subject_canonical,predicate,object_canonical "
        "FROM knowledge_graph WHERE id=?", (edge_id,),
    ).fetchone()
    hy.conn.execute(
        "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) "
        "VALUES (?,?,?,?)",
        (
            f"{row['subject_canonical']} {row['predicate']} {row['object_canonical']}",
            json.dumps(vector), model, len(vector),
        ),
    )


def _eligible(hy):
    return phase1._eligible_dedup_edges(
        hy.conn, hy.config, "service", "uses", "redis_cache",
        model="fake", dim=2,
    )


def test_authoritative_projection_ignores_drifted_cached_counters(hy):
    edge_id = _publish_claim(hy, chunk_id="positive", polarity=1)
    _embedding(hy, edge_id)
    with core_db.evidence_mutation(hy.conn):
        hy.conn.execute(
            "UPDATE knowledge_graph SET pos_evidence=0,neg_evidence=999 WHERE id=?",
            (edge_id,),
        )
    assert [row["edge_id"] for row in _eligible(hy)] == [edge_id]


def test_authoritative_tie_is_not_a_dedup_target_even_if_lifecycle_open(hy):
    edge_id = _publish_claim(hy, chunk_id="tie-pos", polarity=1)
    _publish_claim(hy, chunk_id="tie-neg", polarity=-1)
    _embedding(hy, edge_id)
    assert _eligible(hy) == []


@pytest.mark.parametrize("ledger", ["signal", "lifecycle"])
def test_zero_counter_seed_with_any_durable_history_is_not_pristine(hy, ledger):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
        "pos_evidence,neg_evidence) VALUES ('service','uses','redis',0,0)"
    )
    edge_id = int(hy.conn.execute(
        "SELECT id FROM knowledge_graph WHERE object_canonical='redis'"
    ).fetchone()[0])
    _embedding(hy, edge_id)
    with core_db.evidence_mutation(hy.conn):
        if ledger == "signal":
            hy.conn.execute(
                "INSERT INTO kg_evidence_signals(edge_id,signal_key,signal_kind,"
                "polarity,evidence_weight,counts_toward_confidence) "
                "VALUES (?,'history','legacy_unattributed',-1,1,0)",
                (edge_id,),
            )
        else:
            hy.conn.execute(
                "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,"
                "direction,event_at,dependency_count) "
                "VALUES (?,'old-close','legacy_state',-1,"
                "'2020-01-01T00:00:00.000Z',0)",
                (edge_id,),
            )
    assert _eligible(hy) == []


def test_exact_embedding_model_and_strict_stored_coordinates_required(hy):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
        "pos_evidence,neg_evidence) VALUES ('service','uses','redis',0,0)"
    )
    edge_id = int(hy.conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
    _embedding(hy, edge_id, model="other", vector=[1.0, 0.0])
    assert _eligible(hy) == []
    hy.conn.execute("DELETE FROM edge_embeddings")
    _embedding(hy, edge_id, vector=[True, 0.0])
    assert phase1._find_near_duplicate_edge(
        hy.conn, hy.config, [1.0, 0.0], "service", "uses", "redis_cache",
        model="fake", dim=2,
    ) is None


class _Provider:
    def __init__(self, result, *, model="fake", dim=2, mutate_identity=False):
        self.model = model
        self.dim = dim
        self.result = result
        self.mutate_identity = mutate_identity

    def embed(self, texts):
        if self.mutate_identity:
            self.model = "changed"
        return self.result


@pytest.mark.parametrize("result", [
    [],
    [[1.0]],
    [[float("nan"), 0.0]],
    [[float("inf"), 0.0]],
    [[0.0, 0.0]],
    [[True, 0.0]],
    [["1.0", 0.0]],
])
def test_prepare_rejects_wrong_cardinality_shape_and_invalid_vectors(hy, result):
    extraction = ChunkExtraction(
        triples=[Triple("service", "uses", "redis_cache", 1)], markers=[]
    )
    assert phase1.prepare_dedup_vectors(
        hy.conn, extraction, hy.config, _Provider(result)
    ) == {}


def test_prepare_rejects_provider_identity_change(hy):
    extraction = ChunkExtraction(
        triples=[Triple("service", "uses", "redis_cache", 1)], markers=[]
    )
    assert phase1.prepare_dedup_vectors(
        hy.conn, extraction, hy.config,
        _Provider([[1.0, 0.0]], mutate_identity=True),
    ) == {}


@pytest.mark.parametrize("candidate", [
    ["1", 0.0], [True, 0.0], [float("nan"), 0.0],
    [float("inf"), 0.0], [0.0, 0.0], [1.0], [10**10000, 0.0],
])
def test_dedup_helpers_fail_closed_without_raising_on_malformed_candidate(
    hy, candidate
):
    assert phase1._find_near_duplicate_edge(
        hy.conn, hy.config, candidate, "service", "uses", "redis_cache",
        model="fake", dim=2,
    ) is None
    pool = [phase1._InCycleEdge(
        "service", "uses", "redis", [1.0, 0.0], 1, "fake", 2,
    )]
    assert phase1._find_near_duplicate_in_cycle(
        hy.config, candidate, "service", "uses", "redis_cache", pool,
        model="fake", dim=2,
    ) is None


def _prepared(text: str, *, model: str = "fake", vector=None):
    prepared = phase1._PreparedDedupVectors(model=model, dim=2)
    prepared[text] = [1.0, 0.0] if vector is None else vector
    return prepared


def _direct_upsert(
    hy, chunk: Chunk, object_: str, polarity: int, pool, *, weight=1, model="fake"
):
    text = f"service uses {object_}"
    with core_db.transaction(hy.conn):
        return phase1._upsert_triple(
            hy.conn, chunk.id, Triple("service", "uses", object_, polarity),
            evidence_weight=weight, cfg=hy.config,
            dedup_vectors=_prepared(text, model=model), in_cycle_edges=pool,
        )


def test_samewave_vector_space_identity_cannot_change_between_chunks(hy):
    pool = phase1.new_in_cycle_pool()
    _direct_upsert(hy, _chunk(hy, "model-a"), "redis", 1, pool, model="a")
    _direct_upsert(
        hy, _chunk(hy, "model-b"), "redis_cache", 1, pool, model="b"
    )
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM knowledge_graph WHERE predicate='uses'"
    ).fetchone()[0] == 2


def test_samewave_negative_first_cannot_absorb_later_positive(hy):
    pool = phase1.new_in_cycle_pool()
    _direct_upsert(hy, _chunk(hy, "negative-first"), "redis", -1, pool)
    assert len(pool) == 1 and not pool[0].authoritative
    _direct_upsert(
        hy, _chunk(hy, "positive-second"), "redis_cache", 1, pool
    )
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM knowledge_graph WHERE predicate='uses'"
    ).fetchone()[0] == 2


def test_samewave_positive_then_tying_negative_retires_target(hy):
    pool = phase1.new_in_cycle_pool()
    first_edge = _direct_upsert(
        hy, _chunk(hy, "positive-first"), "redis", 1, pool
    )[2]
    negative_edge = _direct_upsert(
        hy, _chunk(hy, "negative-second"), "redis_cache", -1, pool
    )[2]
    assert negative_edge == first_edge
    assert len(pool) == 1 and not pool[0].authoritative
    third_edge = _direct_upsert(
        hy, _chunk(hy, "positive-third"), "redis_mode", 1, pool
    )[2]
    assert third_edge != first_edge


def test_samewave_weighted_positive_survives_weaker_negative(hy):
    pool = phase1.new_in_cycle_pool()
    first_edge = _direct_upsert(
        hy, _chunk(hy, "weighted-positive"), "redis", 1, pool, weight=2
    )[2]
    assert _direct_upsert(
        hy, _chunk(hy, "weak-negative"), "redis_cache", -1, pool
    )[2] == first_edge
    assert [entry.edge_id for entry in pool if entry.authoritative] == [first_edge]
    assert _direct_upsert(
        hy, _chunk(hy, "positive-third-weighted"), "redis_mode", 1, pool
    )[2] == first_edge


def test_samewave_negative_first_exact_weighted_positive_reactivates_vector(hy):
    pool = phase1.new_in_cycle_pool()
    first_edge = _direct_upsert(
        hy, _chunk(hy, "negative-exact-first"), "redis", -1, pool
    )[2]
    assert len(pool) == 1 and not pool[0].authoritative
    # prepare_dedup_vectors intentionally skips exact DB edges, so reactivation
    # must use the dormant same-wave vector rather than embedding under lock.
    with core_db.transaction(hy.conn):
        exact_edge = phase1._upsert_triple(
            hy.conn,
            _chunk(hy, "positive-exact-second").id,
            Triple("service", "uses", "redis", 1),
            evidence_weight=2,
            cfg=hy.config,
            dedup_vectors={},
            in_cycle_edges=pool,
        )[2]
    assert exact_edge == first_edge
    assert pool[0].authoritative
    sibling = _direct_upsert(
        hy, _chunk(hy, "positive-sibling-third"), "redis_cache", 1, pool
    )[2]
    assert sibling == first_edge


def test_rollback_does_not_publish_stale_samewave_registry_entry(hy, monkeypatch):
    vectors = _prepared("service uses redis")
    shared = phase1.new_in_cycle_pool()

    def fail_outcome(*args, **kwargs):
        raise RuntimeError("forced publication failure")

    # Use a validated extraction so the forced failure occurs after _upsert has
    # staged the minted edge. A malformed claim source is unnecessary here:
    # monkeypatch the final publication call and mark the extraction validated
    # only after installing an exact source below.
    source_session = "rollback-source"
    hy.open_session(source_session, source_workspace_id="w")
    message_id = hy.log_message(
        source_session, "user", "service uses redis",
        source_peer_id="p", source_workspace_id="w",
    )
    source_chunk = Chunk(
        "rollback-source-chunk", source_session, message_id, message_id, "test",
        "user: service uses redis", source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [source_chunk])
    sources = phase1._claim_sources_for_chunk(hy.conn, source_chunk)
    extraction = ChunkExtraction(
        triples=[Triple(
            "service", "uses", "redis", 1, source_message_id=message_id,
        )],
        markers=[],
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
    )
    monkeypatch.setattr(
        phase1.evidence, "record_claim_extraction_outcome", fail_outcome
    )
    with pytest.raises(RuntimeError, match="forced publication failure"):
        with core_db.transaction(hy.conn):
            phase1.persist_chunk_results(
                hy.conn, source_chunk, extraction,
                prompt_version="rollback-v1", cfg=hy.config,
                dedup_vectors=vectors, in_cycle_edges=shared,
            )
    assert shared == []
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM knowledge_graph WHERE predicate='uses'"
    ).fetchone()[0] == 0
    # The next chunk can reuse the same registry without resolving a stale id or
    # violating the evidence FK; it publishes the one real new edge instead.
    next_edge = _direct_upsert(
        hy, _chunk(hy, "after-rollback"), "redis_cache", 1, shared
    )[2]
    assert [entry.edge_id for entry in shared if entry.authoritative] == [next_edge]
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM knowledge_graph WHERE predicate='uses'"
    ).fetchone()[0] == 1


def test_published_staged_entry_is_revalidated_after_finalization(hy):
    session_id = "published-stage"
    hy.open_session(session_id, source_workspace_id="w")
    message_id = hy.log_message(
        session_id, "user", "service uses redis",
        source_peer_id="p", source_workspace_id="w",
    )
    chunk = Chunk(
        "published-stage-chunk", session_id, message_id, message_id, "test",
        "user: service uses redis", source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [chunk])
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    extraction = ChunkExtraction(
        triples=[Triple(
            "service", "uses", "redis", 1, source_message_id=message_id,
        )],
        markers=[],
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
    )
    shared = phase1.new_in_cycle_pool()
    staged = None
    with core_db.transaction(hy.conn):
        staged = phase1.persist_chunk_results(
            hy.conn, chunk, extraction,
            prompt_version="published-stage-v1", cfg=hy.config,
            dedup_vectors=_prepared("service uses redis"),
            in_cycle_edges=shared,
        )
        assert shared == []
        assert staged is not None and staged[0].authoritative
    shared[:] = staged or []
    assert len(shared) == 1 and shared[0].authoritative
    assert phase1._dedup_edge_has_authoritative_positive_majority(
        hy.conn, shared[0].edge_id
    )


def test_unpublished_commit_does_not_activate_staged_samewave_entry(
    hy, monkeypatch
):
    """The post-finalization gate must reject a commit with no publication.

    This is deliberately not a rollback: the edge and its draft evidence reach
    SQLite, but the missing canonical extraction outcome keeps that evidence
    unpublished and therefore ineligible for same-wave write routing.
    """
    session_id = "unpublished-stage"
    hy.open_session(session_id, source_workspace_id="w")
    message_id = hy.log_message(
        session_id, "user", "service uses redis",
        source_peer_id="p", source_workspace_id="w",
    )
    chunk = Chunk(
        "unpublished-stage-chunk", session_id, message_id, message_id, "test",
        "user: service uses redis", source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [chunk])
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    extraction = ChunkExtraction(
        triples=[Triple(
            "service", "uses", "redis", 1, source_message_id=message_id,
        )],
        markers=[],
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
    )

    monkeypatch.setattr(
        phase1.evidence, "record_claim_extraction_outcome",
        lambda *args, **kwargs: None,
    )
    shared = phase1.new_in_cycle_pool()
    with core_db.transaction(hy.conn):
        staged = phase1.persist_chunk_results(
            hy.conn, chunk, extraction,
            prompt_version="unpublished-stage-v1", cfg=hy.config,
            dedup_vectors=_prepared("service uses redis"),
            in_cycle_edges=shared,
        )
        assert shared == []

    assert staged is not None and len(staged) == 1
    assert not staged[0].authoritative
    assert hy.conn.execute(
        "SELECT published_at FROM kg_evidence WHERE edge_id=?",
        (staged[0].edge_id,),
    ).fetchone()[0] is None
    shared[:] = staged
    assert phase1._find_near_duplicate_in_cycle(
        hy.config,
        [1.0, 0.0],
        "service",
        "uses",
        "redis_cache",
        shared,
        model="fake",
        dim=2,
    ) is None


def test_high_degree_decoys_do_not_hide_late_valid_sibling_or_trigger_proofs(
    hy, monkeypatch
):
    for index in range(600):
        obj = f"unrelated_{index}"
        hy.conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
            "pos_evidence,neg_evidence) VALUES ('service','uses',?,0,0)",
            (obj,),
        )
        edge_id = int(hy.conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        _embedding(hy, edge_id)
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
        "pos_evidence,neg_evidence) VALUES ('service','uses','redis',0,0)"
    )
    valid_id = int(hy.conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    _embedding(hy, valid_id)

    import hymem.query.graph_state as graph_state

    calls = 0
    original = graph_state.current_positive_state

    def recording(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_state, "current_positive_state", recording)
    assert [row["edge_id"] for row in _eligible(hy)] == [valid_id]
    assert calls == 0  # all ledger-empty rows resolve at the cheap lexical gate
