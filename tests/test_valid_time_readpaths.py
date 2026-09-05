from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from hymem import HyMem
from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import evidence as evidence_ledger
from hymem.dreaming import phase1
from hymem.dreaming import phase3
from hymem.dreaming.bitemporal import record_lifecycle_event
from hymem.dreaming.canonicalize import merge
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.embeddings import fetch_edge_embeddings
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.retention import prune_retracted_edges
from hymem.dreaming.runner import run_dreaming
from hymem.dreaming.value_supersession import supersede_competing_values
from hymem.extraction.embeddings import StubEmbeddingClient
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.triples import Triple
from hymem.query.augment import _graph_lookup, _python_cosine_edge_search
from hymem.query.augment import _temporal_graph_events
from hymem.query.entities import count_relations, match_known_entities, timeline
from hymem.query.graph_state import facts_at
from hymem.query.state_anchor import select_anchor_edges
from hymem.core.time import normalize_iso_timestamp, validate_event_clock


def _open(tmp_path: Path):
    conn = core_db.connect(tmp_path / "valid-time.sqlite")
    core_db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('sources')")
    return conn


def _messages(conn, rows: list[tuple[str, str, str]]) -> list[int]:
    ids: list[int] = []
    for role, content, created_at in rows:
        cur = conn.execute(
            "INSERT INTO messages(session_id,role,content,created_at) "
            "VALUES ('sources',?,?,?)",
            (role, content, created_at),
        )
        ids.append(int(cur.lastrowid))
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "sources")
    return ids


def _chunk(conn, chunk_id: str, message_id: int) -> Chunk:
    row = conn.execute(
        "SELECT role,content FROM messages WHERE id=?", (message_id,)
    ).fetchone()
    chunk = Chunk(
        id=chunk_id,
        session_id="sources",
        start_message_id=message_id,
        end_message_id=message_id,
        salience_reason="valid-time-test",
        text=f"{row['role']}: {row['content']}",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(conn):
        persist_chunks(conn, [chunk])
    return chunk


def _persist(
    conn,
    cfg: HyMemConfig,
    chunk: Chunk,
    triple: Triple,
    *,
    prompt_version: str = "v13",
) -> None:
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    extraction = phase1.ChunkExtraction(
        triples=[triple],
        markers=[],
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
    )
    with core_db.transaction(conn):
        phase1.persist_chunk_results(
            conn,
            chunk,
            extraction,
            prompt_version=prompt_version,
            cfg=cfg,
        )


def _manual_close(
    conn,
    edge_id: int,
    *,
    signal_key: str,
    event_at: str,
    recorded_at: str | None = None,
) -> None:
    with core_db.evidence_mutation(conn):
        conn.execute(
            "INSERT INTO kg_evidence_signals(edge_id,signal_key,signal_kind,"
            "polarity,evidence_weight,counts_toward_confidence,created_at) "
            "VALUES (?,?,'manual_retraction',-1,1,1,?)",
            (edge_id, signal_key, event_at),
        )
    evidence_ledger.reconcile_edge_counts(conn, [edge_id])
    event_key = evidence_ledger.manual_retraction_event_key(signal_key)
    record_lifecycle_event(
        conn,
        edge_id=edge_id,
        event_key=event_key,
        event_kind="manual_retraction",
        direction=-1,
        event_at=event_at,
    )
    if recorded_at is not None:
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_edge_lifecycle SET created_at=? "
                "WHERE edge_id=? AND event_key=?",
                (recorded_at, edge_id, event_key),
            )


def _objects(rows) -> set[str]:
    return {row.object for row in rows}


def test_facts_at_assert_retract_reassert_boundaries_and_open_citations(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        first_id, second_id, negative_id = _messages(
            conn,
            [
                ("user", "App uses Redis", "2024-01-01T00:00:00Z"),
                ("assistant", "App again uses Redis", "2024-04-01T00:00:00Z"),
                ("user", "App does not use Redis", "2024-05-01T00:00:00Z"),
            ],
        )
        first = _chunk(conn, "first", first_id)
        second = _chunk(conn, "second", second_id)
        negative = _chunk(conn, "negative", negative_id)
        _persist(
            conn, cfg, first,
            Triple("app", "uses", "redis", 1, source_message_id=first_id),
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        _manual_close(
            conn, edge_id, signal_key="close-first",
            event_at="2024-02-01T00:00:00Z",
        )
        _persist(
            conn, cfg, second,
            Triple("app", "uses", "redis", 1, source_message_id=second_id),
        )

        assert facts_at(conn, "2023-12-31T23:59:59Z") == []
        jan = facts_at(conn, "2024-01-01T00:00:00Z")
        assert _objects(jan) == {"redis"}
        assert jan[0].valid_at == "2024-01-01T00:00:00.000Z"
        assert jan[0].invalid_at == "2024-02-01T00:00:00.000Z"
        assert jan[0].citations[0].source_message_id == first_id
        assert facts_at(conn, "2024-02-01T00:00:00Z") == []
        assert facts_at(conn, "2024-03-01T00:00:00Z") == []

        apr = facts_at(conn, "2024-04-01T00:00:00Z")
        assert _objects(apr) == {"redis"}
        assert apr[0].valid_at == "2024-04-01T00:00:00.000Z"
        assert [c.source_message_id for c in apr[0].citations] == [second_id]

        # A current negative assertion never leaks into citations. Lifecycle
        # history does not invent a close from a pointwise confidence tie; the
        # ordinary current graph read applies the stricter evidence-majority
        # gate until a real lifecycle close is persisted.
        _persist(
            conn, cfg, negative,
            Triple("app", "uses", "redis", -1, source_message_id=negative_id),
        )
        before_negative = facts_at(conn, "2024-04-30T23:59:59Z")
        assert [c.source_message_id for c in before_negative[0].citations] == [second_id]
        at_negative = facts_at(conn, "2024-05-01T00:00:00Z")
        assert at_negative[0].valid_at == "2024-04-01T00:00:00.000Z"
        assert at_negative[0].invalid_at is None
        assert [c.source_message_id for c in at_negative[0].citations] == [second_id]

        graph = _graph_lookup(
            conn, cfg, "app uses", ["app"], {}, frozenset({"uses"})
        )
        assert graph == []
    finally:
        conn.close()


def test_facts_at_equal_time_causal_order_is_deterministic(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        first_id, tied_id, reopened_id = _messages(
            conn,
            [
                ("user", "A uses B", "2024-01-01T00:00:00Z"),
                ("assistant", "A still uses B", "2024-01-01T00:00:00Z"),
                ("user", "A uses B again", "2024-01-01T00:00:00.001Z"),
            ],
        )
        for name, message_id in (("first", first_id), ("tied", tied_id)):
            _persist(
                conn, cfg, _chunk(conn, name, message_id),
                Triple("a", "uses", "b", 1, source_message_id=message_id),
            )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        _manual_close(
            conn, edge_id, signal_key="tie-close",
            event_at="2024-01-01T00:00:00Z",
        )
        # Assertions sort before an explicit manual decision at the same valid
        # instant, so the boundary is deterministically closed.
        assert facts_at(conn, "2024-01-01T00:00:00Z") == []

        _persist(
            conn, cfg, _chunk(conn, "reopened", reopened_id),
            Triple("a", "uses", "b", 1, source_message_id=reopened_id),
        )
        reopened = facts_at(conn, "2024-01-01T00:00:00.001Z")
        assert len(reopened) == 1
        assert [c.source_message_id for c in reopened[0].citations] == [reopened_id]
    finally:
        conn.close()


def test_recorded_at_selects_revision_then_and_keeps_clock_metadata_distinct(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2020-06-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "revision", message_id)
        old = Triple(
            "app", "uses", "redis", 1,
            temporal_scope="during the pilot",
            source_message_id=message_id,
        )
        new = Triple(
            "app", "uses", "redis", 1,
            temporal_scope="after the pilot",
            source_message_id=message_id,
        )
        _persist(conn, cfg, chunk, old, prompt_version="v13")
        _persist(conn, cfg, chunk, new, prompt_version="v14")

        evidence = conn.execute(
            "SELECT id,is_current FROM kg_evidence ORDER BY revision"
        ).fetchall()
        old_id, new_id = (int(row["id"]) for row in evidence)
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_evidence SET extracted_at='2024-01-01T00:00:00Z', "
                "published_at='2024-01-01T00:00:00.000Z', "
                "superseded_at='2024-03-01T00:00:00Z' WHERE id=?",
                (old_id,),
            )
            conn.execute(
                "UPDATE kg_evidence SET extracted_at='2024-03-01T00:00:00Z', "
                "published_at='2024-03-01T00:00:00.000Z' "
                "WHERE id=?",
                (new_id,),
            )
            conn.execute(
                "UPDATE kg_edge_lifecycle SET created_at="
                "CASE source_evidence_id WHEN ? THEN '2024-01-01T00:00:00Z' "
                "ELSE '2024-03-01T00:00:00Z' END "
                "WHERE source_evidence_id IN (?,?)",
                (old_id, old_id, new_id),
            )

        then = facts_at(
            conn,
            "2020-07-01T00:00:00Z",
            recorded_at="2024-02-01T00:00:00Z",
        )
        assert [c.evidence_id for c in then[0].citations] == [old_id]
        citation = then[0].citations[0]
        assert citation.currently_authoritative is False
        assert citation.authoritative_at_recorded_time is True
        assert citation.source_event_at == "2020-06-01T00:00:00.000Z"
        assert citation.recorded_at == "2024-01-01T00:00:00.000Z"

        later = facts_at(
            conn,
            "2020-07-01T00:00:00Z",
            recorded_at="2024-04-01T00:00:00Z",
        )
        assert [c.evidence_id for c in later[0].citations] == [new_id]
        current = facts_at(conn, "2020-07-01T00:00:00Z")
        assert [c.evidence_id for c in current[0].citations] == [new_id]

        before = asdict(then[0])
        _manual_close(
            conn, int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0]),
            signal_key="learned-later",
            event_at="2021-01-01T00:00:00Z",
            recorded_at="2024-05-01T00:00:00Z",
        )
        assert asdict(facts_at(
            conn,
            "2020-07-01T00:00:00Z",
            recorded_at="2024-02-01T00:00:00Z",
        )[0]) == before
    finally:
        conn.close()


@pytest.mark.parametrize("bad_created_at", [None, "not-a-timestamp"])
def test_recorded_at_fails_closed_for_unknown_lifecycle_transaction_time(
    tmp_path, bad_created_at
):
    conn = _open(tmp_path)
    try:
        cur = conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
            "pos_evidence,valid_at,status,derived) "
            "VALUES ('legacy','uses','sqlite',1,'2024-01-01','active',0)"
        )
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,"
                "direction,event_at,details,created_at) "
                "VALUES (?,'legacy-state','legacy_state',1,"
                "'2024-01-01T00:00:00.000Z','legacy',?)",
                (int(cur.lastrowid), bad_created_at),
            )
        assert facts_at(
            conn, "2025-01-01", recorded_at="2100-01-01"
        ) == []
    finally:
        conn.close()


@pytest.mark.parametrize("bad_extracted_at", [None, "not-a-timestamp"])
def test_recorded_at_fails_closed_for_unknown_evidence_transaction_time(
    tmp_path, bad_extracted_at
):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2020-01-01T00:00:00Z")]
        )
        _persist(
            conn, cfg, _chunk(conn, "bad-evidence-time", message_id),
            Triple("app", "uses", "redis", 1, source_message_id=message_id),
        )
        evidence_id = int(conn.execute("SELECT id FROM kg_evidence").fetchone()[0])
        conn.execute("DROP TRIGGER kg_evidence_published_at_update_guard")
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_evidence SET extracted_at=? WHERE id=?",
                (bad_extracted_at, evidence_id),
            )
            conn.execute(
                "UPDATE kg_edge_lifecycle SET created_at='2024-01-01T00:00:00Z' "
                "WHERE source_evidence_id=?",
                (evidence_id,),
            )
        assert facts_at(
            conn, "2020-02-01", recorded_at="2024-02-01"
        ) == []
        # A malformed extraction coordinate cannot prove current publication
        # either; current and historical authority both fail closed.
        assert facts_at(conn, "2020-02-01") == []
    finally:
        conn.close()


def test_default_retention_preserves_retracted_valid_time_history(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2020-01-01T00:00:00Z")]
        )
        _persist(
            conn, cfg, _chunk(conn, "retained-history", message_id),
            Triple("app", "uses", "redis", 1, source_message_id=message_id),
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        _manual_close(
            conn, edge_id, signal_key="old-close",
            event_at="2020-02-01T00:00:00Z",
        )
        conn.execute(
            "UPDATE knowledge_graph SET last_seen='2000-01-01' WHERE id=?",
            (edge_id,),
        )
        before = facts_at(conn, "2020-01-15T00:00:00Z")
        assert len(before) == 1
        assert cfg.tombstone_retention_days == 0
        assert prune_retracted_edges(conn, cfg) == 0
        assert facts_at(conn, "2020-01-15T00:00:00Z") == before

        enabled = replace(cfg, tombstone_retention_days=1)
        assert prune_retracted_edges(conn, enabled) == 1
        assert facts_at(conn, "2020-01-15T00:00:00Z") == []
    finally:
        conn.close()


def test_default_dream_runner_retains_retracted_lifecycle_history(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        edge_id = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,last_seen) "
            "VALUES ('history','uses','sqlite',1,'retracted','2000-01-01')"
        ).lastrowid)
        details = "portable test history"
        record_lifecycle_event(
            conn,
            edge_id=edge_id,
            event_key="portable-v6-legacy-0-open",
            event_kind="legacy_state",
            direction=1,
            event_at="2020-01-01T00:00:00Z",
            details=details,
        )
        record_lifecycle_event(
            conn,
            edge_id=edge_id,
            event_key="portable-v6-legacy-1-close",
            event_kind="legacy_state",
            direction=-1,
            event_at="2020-02-01T00:00:00Z",
            details=details,
        )
        before = facts_at(conn, "2020-01-15T00:00:00Z")
        assert len(before) == 1

        run_dreaming(conn, cfg, StubLLMClient(default="[]"))

        assert conn.execute(
            "SELECT 1 FROM knowledge_graph WHERE id=?", (edge_id,)
        ).fetchone() is not None
        assert facts_at(conn, "2020-01-15T00:00:00Z") == before
    finally:
        conn.close()


def test_live_reader_parity_rejects_invalid_derived_and_negative_dominant(tmp_path):
    conn = _open(tmp_path)
    try:
        rows = [
            ("live", "uses", "sqlite", 2, 0, None, 0),
            ("invalid", "uses", "mysql", 2, 0, "2024-02-01", 0),
            ("inferred", "uses", "redis", 2, 0, None, 1),
            ("losing", "uses", "mongo", 1, 2, None, 0),
            ("future", "uses", "oracle", 2, 0, None, 0),
            ("julian", "uses", "mariadb", 2, 0, None, 0),
            ("calendar", "uses", "cockroach", 2, 0, None, 0),
            ("trailing", "uses", "firebird", 2, 0, None, 0),
            ("hour24", "uses", "db2", 2, 0, None, 0),
            ("yearzero", "uses", "dynamodb", 2, 0, None, 0),
            ("overflow", "uses", "cosmos", 2, 0, None, 0),
        ]
        for subject, predicate, obj, pos, neg, invalid_at, derived in rows:
            conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical,predicate,"
                "object_canonical,pos_evidence,neg_evidence,valid_at,invalid_at,"
                "status,derived,last_seen,last_reinforced) "
                "VALUES (?,?,?,?,?,'2024-01-01',?,'active',?,CURRENT_TIMESTAMP,"
                "CURRENT_TIMESTAMP)",
                (subject, predicate, obj, pos, neg, invalid_at, derived),
            )
        conn.execute(
            "UPDATE knowledge_graph SET valid_at='2100-01-01T00:00:00Z' "
            "WHERE subject_canonical='future'"
        )
        conn.execute(
            "UPDATE knowledge_graph SET valid_at='2459999' "
            "WHERE subject_canonical='julian'"
        )
        for subject, valid_at in (
            ("calendar", "2024-02-30"),
            ("trailing", "2024-01-01T"),
            ("hour24", "2024-01-01T24:00:00Z"),
            ("yearzero", "0000-01-01"),
            ("overflow", "0001-01-01T00:00:00+00:01"),
        ):
            conn.execute(
                "UPDATE knowledge_graph SET valid_at=? "
                "WHERE subject_canonical=?",
                (valid_at, subject),
            )

        assert match_known_entities(
            conn,
            "live invalid inferred losing future julian calendar trailing "
            "hour24 yearzero overflow",
        ) == ["live"]
        assert _objects(timeline(conn, "live")) == {"sqlite"}
        assert timeline(conn, "invalid") == []
        assert count_relations(conn, count="subject").entities == ["live"]
        assert count_relations(
            conn, count="subject", include_derived=True
        ).entities == ["inferred", "live"]
        assert [row["subject_canonical"] for row in select_anchor_edges(conn)] == ["live"]
        facts = _graph_lookup(
            conn, HyMemConfig(root=tmp_path), "uses", ["live", "invalid", "inferred", "losing"],
            {}, frozenset({"uses"}),
        )
        assert [(fact.subject, fact.object) for fact in facts] == [("live", "sqlite")]

        embedder = StubEmbeddingClient()
        pending = fetch_edge_embeddings(conn, embedder)
        assert pending is not None
        assert set(pending.edge_text_by_id.values()) == {"live uses sqlite"}
        edge_id = next(iter(pending.edge_text_by_id))
        vector = pending.new_text_vectors["live uses sqlite"]
        conn.execute(
            "INSERT INTO edge_embeddings(edge_text,model,dim,vector_json) "
            "VALUES (?,?,?,?)",
            ("live uses sqlite", embedder.model, embedder.dim, json.dumps(vector)),
        )
        hits = _python_cosine_edge_search(
            conn, embedder, "live", top_k=10, max_scan=100
        )
        assert [item[0] for item in hits] == [edge_id]
    finally:
        conn.close()


def test_entity_matching_requires_live_alias_and_live_object_shape(tmp_path):
    conn = _open(tmp_path)
    try:
        def edge(
            subject,
            object_,
            *,
            valid_at="2024-01-01T00:00:00Z",
            invalid_at=None,
            derived=0,
            pos=2,
            neg=0,
        ):
            conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical,predicate,"
                "object_canonical,pos_evidence,neg_evidence,valid_at,invalid_at,"
                "status,derived) VALUES (?, 'uses', ?, ?, ?, ?, ?, 'active', ?)",
                (subject, object_, pos, neg, valid_at, invalid_at, derived),
            )

        alias_cases = (
            ("alias_live", "canon_live", {}),
            ("alias_future", "canon_future", {
                "valid_at": "2100-01-01T00:00:00Z",
            }),
            ("alias_closed", "canon_closed", {
                "invalid_at": "2024-02-01T00:00:00Z",
            }),
            ("alias_derived", "canon_derived", {"derived": 1}),
            ("alias_losing", "canon_losing", {"pos": 1, "neg": 2}),
        )
        for alias, canonical, options in alias_cases:
            conn.execute(
                "INSERT INTO entity_aliases(alias,canonical) VALUES (?,?)",
                (alias, canonical),
            )
            edge(canonical, f"{canonical}_value", **options)

        # The outer object row itself must be live, even if a type would
        # otherwise prove that the object is entity-shaped.
        edge("owner_future", "outer_future", valid_at="2100-01-01T00:00:00Z")
        conn.execute(
            "INSERT INTO entity_types(entity_canonical,type,confidence) "
            "VALUES ('outer_future','database',1.0)"
        )

        # Graph-shape support is live-filtered too: stale second uses do not
        # turn a one-off object into a known entity.
        edge("owner_stale_shape", "multi_stale")
        edge(
            "other_stale_shape", "multi_stale",
            invalid_at="2024-02-01T00:00:00Z",
        )
        edge("owner_live_shape", "multi_live")
        edge("other_live_shape", "multi_live")

        # Entity typing is an explicit non-graph shape proof, but the outer
        # fact still passes through the live predicate.
        edge("owner_typed", "typed_object")
        conn.execute(
            "INSERT INTO entity_types(entity_canonical,type,confidence) "
            "VALUES ('typed_object','database',1.0)"
        )

        matched = set(match_known_entities(
            conn,
            "alias_live alias_future alias_closed alias_derived alias_losing "
            "outer_future multi_stale multi_live typed_object",
        ))
        assert matched == {"canon_live", "multi_live", "typed_object"}
    finally:
        conn.close()


@pytest.mark.parametrize(
    "poison",
    ["2459999", "2024-02-30", "2100-01-01T00:00:00Z"],
)
def test_legacy_lifecycle_seed_and_history_fail_closed_on_poison(tmp_path, poison):
    conn = _open(tmp_path)
    try:
        poisoned_id = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,derived,valid_at) "
            "VALUES ('poisoned','uses','sqlite',2,'active',0,?)",
            (poison,),
        ).lastrowid)
        unknown_id = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,derived,valid_at,first_seen) "
            "VALUES ('unknown','uses','sqlite',2,'active',0,NULL,NULL)"
        ).lastrowid)
        core_db._seed_v40_legacy_lifecycle(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE edge_id=?",
            (poisoned_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT event_at FROM kg_edge_lifecycle WHERE edge_id=?",
            (unknown_id,),
        ).fetchone()[0] == "0001-01-01T00:00:00.000Z"
        assert [fact.subject for fact in facts_at(
            conn, "2025-01-01T00:00:00Z"
        )] == ["unknown"]

        # Read-path defense remains independent of migration and trigger
        # integrity for a database damaged by an external SQL writer.
        conn.execute("DROP TRIGGER kg_edge_lifecycle_insert_guard")
        with core_db.evidence_mutation(conn):
            conn.execute(
                "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,"
                "direction,event_at) VALUES (?,'corrupt','legacy_state',1,?)",
                (poisoned_id, poison),
            )
        assert [fact.subject for fact in facts_at(
            conn, "9999-12-31T23:59:59.999Z"
        )] == ["unknown"]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "valid_at",
    [
        "2024-01-01",
        "2024-01-01T12:30:45Z",
        "2024-01-01 12:30:45+01:00",
        "2024-01-01T14:00:00+14:00",
        "2024-01-01T12:30:45.9999Z",
        "2024-01-01T12:30:45.123456Z",
    ],
)
def test_live_reader_accepts_supported_iso_timestamp_forms(tmp_path, valid_at):
    conn = _open(tmp_path)
    try:
        conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,neg_evidence,valid_at,status,derived) "
            "VALUES ('supported','uses','sqlite',2,0,?,'active',0)",
            (valid_at,),
        )
        assert match_known_entities(conn, "supported") == ["supported"]
        assert _objects(timeline(conn, "supported")) == {"sqlite"}
    finally:
        conn.close()


def test_timeline_and_graph_fact_order_offset_instants_and_return_utc(tmp_path):
    conn = _open(tmp_path)
    try:
        earlier = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,derived,valid_at) "
            "VALUES ('service','uses','earlier_utc',2,'active',0,"
            "'2024-01-02T00:30:00+02:00')"
        ).lastrowid)
        conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,derived,valid_at) "
            "VALUES ('service','uses','later_utc',2,'active',0,"
            "'2024-01-01T23:00:00Z')"
        )
        item = timeline(conn, "service")[0]
        assert item.edge_id == earlier
        assert item.valid_at == "2024-01-01T22:30:00.000Z"
        facts = _graph_lookup(
            conn, HyMemConfig(root=tmp_path), "service uses", ["service"],
            {}, frozenset({"uses"}),
        )
        by_object = {fact.object: fact for fact in facts}
        assert by_object["earlier_utc"].valid_at == "2024-01-01T22:30:00.000Z"
        assert by_object["later_utc"].valid_at == "2024-01-01T23:00:00.000Z"
    finally:
        conn.close()


def test_temporal_graph_event_uses_utc_date_across_offset_boundary(tmp_path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(conn, [
            ("user", "Service uses SQLite", "2024-01-01T22:30:00Z")
        ])
        chunk = _chunk(conn, "utc-date-boundary", message_id)
        _persist(
            conn, HyMemConfig(root=tmp_path), chunk,
            Triple("service", "uses", "sqlite", 1,
                   source_message_id=message_id),
        )
        conn.execute(
            "UPDATE knowledge_graph SET valid_at='2024-01-02T00:30:00+02:00'"
        )
        events = _temporal_graph_events(conn, ["service"], top_k=5)
        assert len(events) == 1
        assert events[0].date == "2024-01-01"
    finally:
        conn.close()


def test_lifecycle_clock_guard_rejects_forged_future_manual_close_atomically(
    tmp_path,
):
    conn = _open(tmp_path)
    try:
        edge_id = int(
            conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical,predicate,"
                "object_canonical,pos_evidence,neg_evidence,valid_at,status,derived) "
                "VALUES ('service','uses','sqlite',2,0,'2024-01-01',"
                "'active',0)"
            ).lastrowid
        )
        record_lifecycle_event(
            conn,
            edge_id=edge_id,
            event_key="legacy-state",
            event_kind="legacy_state",
            direction=1,
            event_at="2024-01-01T00:00:00Z",
            recorded_at="2024-01-01T00:00:00Z",
        )
        with core_db.evidence_mutation(conn):
            conn.execute(
                "INSERT INTO kg_evidence_signals(edge_id,signal_key,signal_kind,"
                "polarity,evidence_weight,counts_toward_confidence,details,"
                "created_at) VALUES (?, 'forged-future', 'manual_retraction',"
                "-1,1,1,'operator','2100-01-01T00:00:00Z')",
                (edge_id,),
            )
        before = tuple(
            conn.execute(
                "SELECT status,valid_at,invalid_at FROM knowledge_graph WHERE id=?",
                (edge_id,),
            ).fetchone()
        )
        with pytest.raises(ValueError, match="lifecycle event valid time"):
            record_lifecycle_event(
                conn,
                edge_id=edge_id,
                event_key=evidence_ledger.manual_retraction_event_key(
                    "forged-future"
                ),
                event_kind="manual_retraction",
                direction=-1,
                event_at="2100-01-01T00:00:00Z",
                details="operator",
            )
        assert tuple(
            conn.execute(
                "SELECT status,valid_at,invalid_at FROM knowledge_graph WHERE id=?",
                (edge_id,),
            ).fetchone()
        ) == before
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE edge_id=?",
            (edge_id,),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_lifecycle_clock_guard_allows_300_seconds_but_rejects_301(tmp_path):
    conn = _open(tmp_path)
    try:
        accepted = int(
            conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical,predicate,"
                "object_canonical,pos_evidence,status,derived) "
                "VALUES ('accepted','uses','sqlite',1,'active',0)"
            ).lastrowid
        )
        rejected = int(
            conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical,predicate,"
                "object_canonical,pos_evidence,status,derived) "
                "VALUES ('rejected','uses','sqlite',1,'active',0)"
            ).lastrowid
        )
        assert record_lifecycle_event(
            conn,
            edge_id=accepted,
            event_key="legacy-state",
            event_kind="legacy_state",
            direction=1,
            event_at="2024-01-01T00:05:00Z",
            recorded_at="2024-01-01T00:00:00Z",
        )
        with pytest.raises(ValueError, match="lifecycle event valid time"):
            record_lifecycle_event(
                conn,
                edge_id=rejected,
                event_key="legacy-state",
                event_kind="legacy_state",
                direction=1,
                event_at="2024-01-01T00:05:01Z",
                recorded_at="2024-01-01T00:00:00Z",
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE edge_id=?", (rejected,)
        ).fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    "malformed",
    ["not-a-time", "2459999", "2024-02-30", "2024-01-01T24:00:00Z"],
)
def test_lifecycle_only_maps_explicit_legacy_none_to_ancient(tmp_path, malformed):
    conn = _open(tmp_path)
    try:
        unknown_edge = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,status,derived) "
            "VALUES ('unknown','uses','sqlite','active',0)"
        ).lastrowid)
        assert record_lifecycle_event(
            conn,
            edge_id=unknown_edge,
            event_key="legacy-state",
            event_kind="legacy_state",
            direction=1,
            event_at=None,
            recorded_at="2024-01-01T00:00:00Z",
        )
        assert conn.execute(
            "SELECT event_at FROM kg_edge_lifecycle WHERE edge_id=?",
            (unknown_edge,),
        ).fetchone()[0] == "0001-01-01T00:00:00.000Z"

        malformed_edge = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,status,derived) "
            "VALUES ('malformed','uses','sqlite','active',0)"
        ).lastrowid)
        with pytest.raises(ValueError, match="event_at timestamps"):
            record_lifecycle_event(
                conn,
                edge_id=malformed_edge,
                event_key="legacy-state",
                event_kind="legacy_state",
                direction=1,
                event_at=malformed,
                recorded_at="2024-01-01T00:00:00Z",
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE edge_id=?",
            (malformed_edge,),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_value_supersession_cannot_promote_hidden_tied_candidate(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        old_id, new_id, dispute_id = _messages(
            conn,
            [
                ("user", "Target was 65 percent", "2024-01-01T00:00:00Z"),
                ("user", "Target is 90 percent", "2025-01-01T00:00:00Z"),
                ("user", "Target is not 90 percent", "2025-02-01T00:00:00Z"),
            ],
        )
        for chunk_id, message_id, value, polarity in (
            ("old-value", old_id, "65_percent", 1),
            ("new-value", new_id, "90_percent", 1),
            ("new-dispute", dispute_id, "90_percent", -1),
        ):
            _persist(
                conn,
                cfg,
                _chunk(conn, chunk_id, message_id),
                Triple(
                    "service",
                    "has_attribute",
                    value,
                    polarity,
                    source_message_id=message_id,
                ),
            )

        assert [
            fact.object
            for fact in _graph_lookup(
                conn,
                cfg,
                "service target",
                ["service"],
                {},
                frozenset({"has_attribute"}),
            )
        ] == ["65_percent"]
        assert supersede_competing_values(conn, cfg) == 0
        assert conn.execute(
            "SELECT status FROM knowledge_graph "
            "WHERE object_canonical='65_percent'"
        ).fetchone()[0] == "active"
    finally:
        conn.close()


def test_value_supersession_orders_legacy_offset_instants_in_utc(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        message_ids = _messages(conn, [
            ("user", "Target 65", "2024-01-01T00:00:00Z"),
            ("user", "Target 78", "2024-01-01T00:00:01Z"),
        ])
        coordinates = (
            "2024-01-02T00:30:00+02:00",  # 2024-01-01 22:30Z
            "2024-01-01T23:00:00Z",
        )
        edges: list[int] = []
        for index, value in enumerate(("65_percent", "78_percent")):
            edge_id = int(conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical,predicate,"
                "object_canonical,status,derived) "
                "VALUES ('service','has_attribute',?,'active',0)",
                (value,),
            ).lastrowid)
            chunk = _chunk(conn, f"offset-value-{index}", message_ids[index])
            mutation = evidence_ledger.record_chunk_evidence(
                conn, edge_id=edge_id, chunk_id=chunk.id,
                evidence_kind="extraction", polarity=1, evidence_weight=1,
                weight_source="legacy-offset-test",
            )
            with core_db.evidence_mutation(conn):
                conn.execute(
                    "UPDATE kg_evidence SET extracted_at=? WHERE id=?",
                    (coordinates[index], mutation.evidence_id),
                )
            record_lifecycle_event(
                conn, edge_id=edge_id, event_key="legacy-state",
                event_kind="legacy_state", direction=1,
                event_at=coordinates[index], recorded_at="2025-01-01",
            )
            edges.append(edge_id)
        assert supersede_competing_values(conn, cfg) == 1
        assert conn.execute(
            "SELECT status FROM knowledge_graph WHERE id=?", (edges[0],)
        ).fetchone()[0] == "retracted"
        assert conn.execute(
            "SELECT status FROM knowledge_graph WHERE id=?", (edges[1],)
        ).fetchone()[0] == "active"
        assert conn.execute(
            "SELECT invalid_at FROM knowledge_graph WHERE id=?", (edges[0],)
        ).fetchone()[0] == "2024-01-01T23:00:00.000Z"
    finally:
        conn.close()


def test_phase3_does_not_mutate_future_edges_and_legacy_malformed_cause_is_safe(
    tmp_path,
):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App and SQLite changed", "2024-01-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "phase3-clock", message_id)
        conn.executemany(
            "INSERT INTO entity_mentions(chunk_id,entity_canonical) VALUES (?,?)",
            [(chunk.id, "app"), (chunk.id, "sqlite")],
        )
        future_live = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,derived,valid_at,last_reinforced) "
            "VALUES ('app','uses','sqlite',2,'active',0,"
            "'2100-01-01T00:00:00Z','2000-01-01')"
        ).lastrowid)
        future_zombie = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,status,derived,valid_at) "
            "VALUES ('future_app','uses','future_db','active',0,"
            "'2100-01-01T00:00:00Z')"
        ).lastrowid)
        evidence_ledger.record_chunk_evidence(
            conn, edge_id=future_zombie, chunk_id=chunk.id,
            evidence_kind="decay", polarity=-1, evidence_weight=3,
            weight_source="clock-test",
        )
        before_evidence = conn.execute(
            "SELECT COUNT(*) FROM kg_evidence"
        ).fetchone()[0]
        phase3.reinforce(conn, cfg)
        phase3.decay(conn, cfg)
        assert conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == (
            before_evidence
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE edge_id IN (?,?)",
            (future_live, future_zombie),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE id IN (?,?) "
            "AND status='active'",
            (future_live, future_zombie),
        ).fetchone()[0] == 2

        malformed_edge = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,status,derived,valid_at) "
            "VALUES ('legacy_bad','uses','db','active',0,'2024-01-01')"
        ).lastrowid)
        malformed = evidence_ledger.record_chunk_evidence(
            conn, edge_id=malformed_edge, chunk_id=chunk.id,
            evidence_kind="decay", polarity=-1, evidence_weight=3,
            weight_source="malformed-clock-test",
        )
        with core_db.evidence_mutation(conn):
            conn.execute(
                "UPDATE kg_evidence SET extracted_at='2459999' WHERE id=?",
                (malformed.evidence_id,),
            )
        phase3.decay(conn, cfg)
        assert conn.execute(
            "SELECT status FROM knowledge_graph WHERE id=?", (malformed_edge,)
        ).fetchone()[0] == "active"
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE edge_id=?",
            (malformed_edge,),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_recorded_at_uses_current_canonical_topology(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses Redis DB", "2020-01-01T00:00:00Z")]
        )
        _persist(
            conn,
            cfg,
            _chunk(conn, "before-merge", message_id),
            Triple(
                "app", "uses", "redis_db", 1, source_message_id=message_id
            ),
        )
        evidence_id = int(conn.execute("SELECT id FROM kg_evidence").fetchone()[0])
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_evidence SET extracted_at='2021-01-01T00:00:00Z', "
                "published_at='2021-01-01T00:00:00.000Z' "
                "WHERE id=?",
                (evidence_id,),
            )
            conn.execute(
                "UPDATE kg_edge_lifecycle SET created_at='2021-01-01T00:00:00Z' "
                "WHERE source_evidence_id=?",
                (evidence_id,),
            )

        # Merges are destructive topology maintenance, not versioned events.
        # A pre-merge authority cutoff is therefore projected onto today's
        # endpoint and remains discoverable through the old alias.
        with core_db.transaction(conn):
            merge(conn, keep="redis", drop="redis_db")
        rows = facts_at(
            conn,
            "2020-02-01T00:00:00Z",
            recorded_at="2021-02-01T00:00:00Z",
            entity="redis_db",
        )
        assert [(row.subject, row.object) for row in rows] == [("app", "redis")]
    finally:
        conn.close()


def test_canonical_edge_collision_preserves_earliest_publication_history(
    tmp_path,
):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses a database", "2023-01-01T00:00:00Z")]
        )
        early_chunk = _chunk(conn, "merge-history-early", message_id)
        late_chunk = _chunk(conn, "merge-history-late", message_id)
        _persist(
            conn, cfg, early_chunk,
            Triple("app", "uses", "db", 1, source_message_id=message_id),
        )
        _persist(
            conn, cfg, late_chunk,
            Triple("app", "uses", "redis", 1, source_message_id=message_id),
        )
        revisions = {
            row["chunk_id"]: int(row["id"])
            for row in conn.execute(
                "SELECT id,chunk_id FROM kg_evidence WHERE is_current=1"
            ).fetchall()
        }
        with core_db.evidence_history_mutation(conn):
            for chunk, prefix in (
                (early_chunk, "2024-01-01"),
                (late_chunk, "2025-01-01"),
            ):
                evidence_id = revisions[chunk.id]
                conn.execute(
                    "UPDATE kg_evidence SET extracted_at=?,published_at=? "
                    "WHERE id=?",
                    (
                        f"{prefix}T00:00:00.000Z",
                        f"{prefix}T00:03:00.000Z",
                        evidence_id,
                    ),
                )
                conn.execute(
                    "UPDATE kg_edge_lifecycle SET created_at=? "
                    "WHERE source_evidence_id=?",
                    (f"{prefix}T00:01:00.000Z", evidence_id),
                )
                conn.execute(
                    "UPDATE kg_claim_observations SET observed_at=? "
                    "WHERE evidence_id=?",
                    (f"{prefix}T00:02:00.000Z", evidence_id),
                )
                conn.execute(
                    "UPDATE kg_claim_extraction_outcomes SET succeeded_at=? "
                    "WHERE chunk_id=?",
                    (f"{prefix}T00:03:00.000Z", chunk.id),
                )
        conn.execute(
            "UPDATE sessions SET started_at='2023-01-01T00:00:00.000Z' "
            "WHERE id='sources'"
        )
        conn.execute(
            "UPDATE chunks SET created_at='2023-01-01T00:01:00.000Z' "
            "WHERE chunk_kind='coverage'"
        )
        conn.execute(
            "UPDATE chunks SET created_at='2024-01-01T00:00:00.000Z' "
            "WHERE id=?", (early_chunk.id,),
        )
        conn.execute(
            "UPDATE chunks SET created_at='2025-01-01T00:00:00.000Z' "
            "WHERE id=?", (late_chunk.id,),
        )
        conn.execute("DROP TRIGGER message_lossless_stream_update_guard")
        conn.execute(
            "UPDATE message_retention_coverage "
            "SET created_at='2023-01-01T00:01:00.000Z'"
        )

        with core_db.transaction(conn):
            merge(conn, keep="redis", drop="db")

        retained = conn.execute(
            "SELECT extracted_at,published_at FROM kg_evidence"
        ).fetchone()
        assert tuple(retained) == (
            "2024-01-01T00:00:00.000Z",
            "2024-01-01T00:03:00.000Z",
        )
        assert conn.execute(
            "SELECT created_at FROM kg_edge_lifecycle "
            "WHERE event_kind='claim_assertion'"
        ).fetchone()[0] == "2024-01-01T00:01:00.000Z"
        historical = facts_at(
            conn, "2023-06-01", recorded_at="2024-06-01", entity="db"
        )
        assert [(fact.subject, fact.object) for fact in historical] == [
            ("app", "redis")
        ]

        from hymem import portability

        wire = tmp_path / "runtime-edge-merge.jsonl"
        portability.export_jsonl(conn, wire)
        imported = HyMem(HyMemConfig(root=tmp_path / "runtime-edge-merge-import"))
        try:
            imported.import_(wire)
            replay = imported.import_(wire)
            assert sum(replay.values()) == 0
            assert [(fact.subject, fact.object) for fact in facts_at(
                imported.conn, "2023-06-01", recorded_at="2024-06-01"
            )] == [("app", "redis")]
        finally:
            imported.close()
    finally:
        conn.close()


@pytest.mark.parametrize(("keep", "drop"), [("redis", "db"), ("db", "redis")])
def test_canonical_edge_merge_uses_outcome_boundary_and_keeps_winning_copy(
    tmp_path, keep, drop,
):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses a database", "2023-01-01T00:00:00Z")]
        )
        chunks = {
            name: _chunk(conn, f"merge-authority-{name}", message_id)
            for name in ("old", "middle", "winner")
        }
        _persist(
            conn, cfg, chunks["old"],
            Triple(
                "app", "uses", "redis", 1, temporal_scope="old",
                source_message_id=message_id,
            ),
            prompt_version="v13",
        )
        _persist(
            conn, cfg, chunks["middle"],
            Triple(
                "app", "uses", "redis", 1, temporal_scope="middle",
                source_message_id=message_id,
            ),
            prompt_version="v14",
        )
        _persist(
            conn, cfg, chunks["winner"],
            Triple(
                "app", "uses", "db", 1, temporal_scope="old",
                source_message_id=message_id,
            ),
            prompt_version="v15",
        )
        evidence = {
            row["chunk_id"]: int(row["id"])
            for row in conn.execute(
                "SELECT id,chunk_id FROM kg_evidence ORDER BY id"
            ).fetchall()
        }
        clocks = {
            chunks["old"].id: (
                "2024-01-01T00:00:00.000Z",
                "2024-01-01T00:01:00.000Z",
                "2024-01-01T00:02:00.000Z",
                "2024-01-01T00:03:00.000Z",
                "2024-06-01T00:03:00.000Z",
            ),
            chunks["middle"].id: (
                "2024-06-01T00:00:00.000Z",
                "2024-06-01T00:01:00.000Z",
                "2024-06-01T00:02:00.000Z",
                "2024-06-01T00:03:00.000Z",
                None,
            ),
            chunks["winner"].id: (
                "2025-01-01T00:00:00.000Z",
                "2025-01-01T00:01:00.000Z",
                "2025-01-01T00:02:00.000Z",
                "2025-01-01T00:03:00.000Z",
                None,
            ),
        }
        with core_db.evidence_history_mutation(conn):
            for chunk_id, (
                extracted_at, lifecycle_at, observed_at, published_at,
                superseded_at,
            ) in clocks.items():
                evidence_id = evidence[chunk_id]
                conn.execute(
                    "UPDATE kg_evidence SET extracted_at=?,published_at=?,"
                    "superseded_at=COALESCE(?,superseded_at) WHERE id=?",
                    (
                        extracted_at, published_at, superseded_at,
                        evidence_id,
                    ),
                )
                conn.execute(
                    "UPDATE kg_edge_lifecycle SET created_at=? "
                    "WHERE source_evidence_id=?",
                    (lifecycle_at, evidence_id),
                )
                conn.execute(
                    "UPDATE kg_claim_observations SET observed_at=? "
                    "WHERE evidence_id=?",
                    (observed_at, evidence_id),
                )
                conn.execute(
                    "UPDATE kg_claim_extraction_outcomes SET succeeded_at=? "
                    "WHERE chunk_id=?",
                    (published_at, chunk_id),
                )

        with core_db.transaction(conn):
            merge(conn, keep=keep, drop=drop)

        # The v14 interpretation remains authoritative until the v15 whole
        # chunk publication. Closing it at observed_at would create a 60-second
        # transaction-time gap.
        before_publication = facts_at(
            conn,
            "2023-06-01T00:00:00Z",
            recorded_at="2025-01-01T00:02:30Z",
        )
        after_publication = facts_at(
            conn,
            "2023-06-01T00:00:00Z",
            recorded_at="2025-01-01T00:04:00Z",
        )
        assert [(fact.subject, fact.object) for fact in before_publication] == [
            ("app", keep)
        ]
        assert [(fact.subject, fact.object) for fact in after_publication] == [
            ("app", keep)
        ]
        current = conn.execute(
            "SELECT revision,temporal_scope,extracted_at,published_at "
            "FROM kg_evidence WHERE is_current=1"
        ).fetchone()
        assert tuple(current) == (
            3,
            "old",
            "2025-01-01T00:00:00.000Z",
            "2025-01-01T00:03:00.000Z",
        )
        assert [tuple(row) for row in conn.execute(
            "SELECT revision,temporal_scope,is_current FROM kg_evidence "
            "ORDER BY revision"
        ).fetchall()] == [
            (1, "old", 0),
            (2, "middle", 0),
            (3, "old", 1),
        ]
        assert [row[0] for row in conn.execute(
            "SELECT event_key FROM kg_edge_lifecycle "
            "WHERE event_kind='claim_assertion' ORDER BY event_key"
        ).fetchall()] == [
            "claim-assertion:sources:1:extraction:r1",
            "claim-assertion:sources:1:extraction:r2",
            "claim-assertion:sources:1:extraction:r3",
        ]
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence "
            "WHERE extracted_at >= '2026-01-01'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_temporal_graph_uses_valid_time_and_current_scope_only(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App started using Redis", "2020-06-01T12:00:00Z")]
        )
        chunk = _chunk(conn, "temporal-revision", message_id)
        _persist(
            conn,
            cfg,
            chunk,
            Triple(
                "app", "uses", "redis", 1,
                temporal_scope="retired scope must not leak",
                source_message_id=message_id,
            ),
            prompt_version="v13",
        )
        _persist(
            conn,
            cfg,
            chunk,
            Triple("app", "uses", "redis", 1, source_message_id=message_id),
            prompt_version="v14",
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        conn.execute(
            "UPDATE knowledge_graph SET first_seen='2035-01-01T00:00:00Z' "
            "WHERE id=?",
            (edge_id,),
        )
        # These rows exercise every default exclusion before citation lookup.
        conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
            "pos_evidence,valid_at,status,derived) VALUES "
            "('app','uses','uncited','1','2010-01-01','active',0),"
            "('app','uses','inferred','1','2011-01-01','active',1),"
            "('app','uses','invalid','1','2012-01-01','active',0)"
        )
        conn.execute(
            "UPDATE knowledge_graph SET invalid_at='2013-01-01' "
            "WHERE object_canonical='invalid'"
        )

        events = _temporal_graph_events(conn, ["app"], top_k=20)
        assert [(event.date, event.text) for event in events] == [
            ("2020-06-01", "app uses redis")
        ]
        assert "retired scope" not in events[0].text
        assert events[0].why_retrieved == ["edge_valid_at"]
    finally:
        conn.close()


def test_graph_fact_citations_are_bounded_current_positive_and_exact(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = replace(HyMemConfig(root=tmp_path), graph_top_k=10)
        message_rows = [
            (
                "user" if index % 2 == 0 else "assistant",
                f"App uses Redis confirmation {index}",
                f"2024-01-{index + 1:02d}T0{index}:00:00Z",
            )
            for index in range(7)
        ]
        message_ids = _messages(conn, message_rows)
        chunks: list[Chunk] = []
        for index, message_id in enumerate(message_ids):
            chunk = _chunk(conn, f"citation-{index}", message_id)
            chunks.append(chunk)
            _persist(
                conn,
                cfg,
                chunk,
                Triple("app", "uses", "redis", 1, source_message_id=message_id),
            )

        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        retired_id = int(conn.execute(
            "SELECT id FROM kg_evidence WHERE source_message_id=? AND is_current=1",
            (message_ids[-1],),
        ).fetchone()[0])
        _persist(
            conn,
            cfg,
            chunks[-1],
            Triple(
                "app", "uses", "redis", 1,
                temporal_scope="current interpretation",
                source_message_id=message_ids[-1],
            ),
            prompt_version="v14",
        )
        before_replay = int(conn.execute(
            "SELECT COUNT(*) FROM kg_evidence"
        ).fetchone()[0])
        _persist(
            conn,
            cfg,
            chunks[-1],
            Triple(
                "app", "uses", "redis", 1,
                temporal_scope="current interpretation",
                source_message_id=message_ids[-1],
            ),
            prompt_version="v14",
        )
        assert int(conn.execute(
            "SELECT COUNT(*) FROM kg_evidence"
        ).fetchone()[0]) == before_replay
        current_revision_id = int(conn.execute(
            "SELECT id FROM kg_evidence WHERE source_message_id=? AND is_current=1",
            (message_ids[-1],),
        ).fetchone()[0])
        [negative_id] = _messages(
            conn,
            [("user", "App does not use Redis", "2024-02-01T00:00:00Z")],
        )
        negative_chunk = _chunk(conn, "citation-negative", negative_id)
        _persist(
            conn,
            cfg,
            negative_chunk,
            Triple("app", "uses", "redis", -1, source_message_id=negative_id),
        )

        facts = _graph_lookup(
            conn, cfg, "app uses redis", ["app"], {}, frozenset({"uses"})
        )
        assert len(facts) == 1
        fact = facts[0]
        assert fact.edge_id == edge_id
        assert fact.valid_at == "2024-01-01T00:00:00.000Z"
        assert fact.invalid_at is None
        assert fact.derived is False
        assert len(fact.citations) == 5
        assert [citation.source_message_id for citation in fact.citations] == list(
            reversed(message_ids[-5:])
        )
        assert fact.citations[0].evidence_id == current_revision_id
        assert retired_id not in {item.evidence_id for item in fact.citations}
        assert negative_id not in {
            item.source_message_id for item in fact.citations
        }
        for citation in fact.citations:
            index = message_ids.index(citation.source_message_id)
            assert citation.source_role == message_rows[index][0]
            assert citation.source_session_id == "sources"
            assert citation.source_event_at == message_rows[index][2].replace(
                "Z", ".000Z"
            )
            assert citation.source_created_at == message_rows[index][2]
            assert citation.coverage_chunk_id
            assert citation.coverage_version
            assert citation.extraction_chunk_id == chunks[index].id
            assert citation.currently_authoritative is True
            assert citation.authoritative_at_recorded_time is True
            assert citation.provenance_status == "canonical"
    finally:
        conn.close()


@pytest.mark.parametrize(
    "invalid_time",
    [
        "2100-01-01T00:00:00Z",
        "2459999",
        "",
        False,
        0,
        "20240101",
        "2024-W01-1",
        "2024-01-01T00:00:00,123Z",
        "2024-01-01T12:34:56+15:00",
        "2024-01-01T12:34:56+23:59",
        "0001-01-01T00:00:00+00:01",
        "2024-01-01T",
        "2024-01-01T24:00:00Z",
        "2024-02-30",
        "0000-01-01",
    ],
)
def test_public_ingress_rejects_future_or_non_iso_time_atomically(
    tmp_path, invalid_time
):
    hy = HyMem(HyMemConfig(root=tmp_path), llm=StubLLMClient(default="[]"))
    try:
        with pytest.raises(ValueError, match="message source|created_at"):
            hy.log_message("single", "user", "poison", created_at=invalid_time)
        assert hy.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_retention_coverage"
        ).fetchone()[0] == 0

        with pytest.raises(ValueError, match="message source|created_at"):
            hy.log_messages(
                "batch",
                [
                    ("user", "valid first", "2024-01-01T00:00:00Z"),
                    ("assistant", "poison second", invalid_time),
                ],
            )
        assert hy.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_retention_coverage"
        ).fetchone()[0] == 0

        hy.log_message(
            "healthy", "user", "normal past message", created_at="2024-01-01"
        )
        report = hy.dream(session_ids=["healthy"])
        assert report.skipped_locked is False
    finally:
        hy.close()


def test_shared_clock_canonicalizer_handles_precision_and_datetime_limits(tmp_path):
    conn = _open(tmp_path)
    try:
        assert normalize_iso_timestamp(
            "2024-01-01T00:00:00.9999Z", context="test"
        ) == "2024-01-01T00:00:00.999Z"
        assert normalize_iso_timestamp(
            "2024-01-01T00:00:00.000999Z", context="test"
        ) == "2024-01-01T00:00:00.000Z"
        maximum = "9999-12-31T23:59:59.999999Z"
        assert normalize_iso_timestamp(maximum, context="test") == (
            "9999-12-31T23:59:59.999Z"
        )
        # The skew check must not add onto datetime.max and leak OverflowError.
        validate_event_clock(conn, maximum, maximum, context="maximum clock")
        for malformed in (
            "2459999", "2024-02-30", "2024-01-01T",
            "2024-01-01T24:00:00Z", "0000-01-01", "20240101",
            "2024-W01-1", "2024-01-01T00:00:00,123Z",
            "2024-01-01T00:00:00+23:59",
            "0001-01-01T00:00:00+00:01",
        ):
            with pytest.raises(ValueError):
                normalize_iso_timestamp(malformed, context="test")
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("created_at", "canonical"),
    [
        ("2024-01-01T01:00:00+01:00", "2024-01-01T00:00:00.000Z"),
        ("2024-01-01T00:00:00.0009Z", "2024-01-01T00:00:00.000Z"),
        ("2024-01-01T00:00:00.9995Z", "2024-01-01T00:00:00.999Z"),
    ],
)
def test_supported_public_clock_survives_claim_materialization_and_export(
    tmp_path, created_at, canonical
):
    cfg = HyMemConfig(root=tmp_path)
    hy = HyMem(cfg)
    try:
        message_id = hy.log_message(
            "clock-source",
            "user",
            "Clock app uses SQLite",
            created_at=created_at,
        )
        chunk = Chunk(
            id="supported-clock",
            session_id="clock-source",
            start_message_id=message_id,
            end_message_id=message_id,
            salience_reason="valid-time-test",
            text="user: Clock app uses SQLite",
            source_message_ids=(message_id,),
        )
        with core_db.transaction(hy.conn):
            persist_chunks(hy.conn, [chunk])
        _persist(
            hy.conn,
            cfg,
            chunk,
            Triple(
                "clock_app",
                "uses",
                "sqlite",
                1,
                source_message_id=message_id,
            ),
        )
        assert tuple(hy.conn.execute(
            "SELECT source_created_at,source_event_at FROM kg_evidence"
        ).fetchone()) == (canonical, canonical)
        assert hy.conn.execute(
            "SELECT created_at FROM messages WHERE id=?", (message_id,)
        ).fetchone()[0] == canonical
        assert _objects(
            facts_at(hy.conn, canonical)
        ) == {"sqlite"}
        export_path = tmp_path / "supported-clock.jsonl"
        hy.export(export_path)
        assert export_path.is_file()
    finally:
        hy.close()


def test_ledger_defense_rejects_future_assertion_and_negative(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        future_id, past_id = _messages(
            conn,
            [
                ("user", "Future claim", "2100-01-01T00:00:00Z"),
                ("user", "Current claim", "2024-01-01T00:00:00Z"),
            ],
        )
        with pytest.raises(ValueError, match="canonical evidence valid time"):
            _persist(
                conn,
                cfg,
                _chunk(conn, "future-positive", future_id),
                Triple("app", "uses", "future_db", 1, source_message_id=future_id),
            )
        assert conn.execute(
            "SELECT 1 FROM knowledge_graph WHERE object_canonical='future_db'"
        ).fetchone() is None

        _persist(
            conn,
            cfg,
            _chunk(conn, "past-positive", past_id),
            Triple("app", "uses", "redis", 1, source_message_id=past_id),
        )
        [future_negative_id] = _messages(
            conn,
            [("user", "Future negative", "2100-02-01T00:00:00Z")],
        )
        with pytest.raises(ValueError, match="canonical evidence valid time"):
            _persist(
                conn,
                cfg,
                _chunk(conn, "future-negative", future_negative_id),
                Triple(
                    "app", "uses", "redis", -1,
                    source_message_id=future_negative_id,
                ),
            )

        history = facts_at(conn, "2026-09-04T00:00:00Z")
        assert _objects(history) == {"redis"}
        current = _graph_lookup(
            conn, cfg, "app uses redis", ["app"], {}, frozenset({"uses"})
        )
        assert [(fact.subject, fact.object) for fact in current] == [
            ("app", "redis")
        ]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "legacy_source_time",
    [
        "2100-01-01T00:00:00Z",
        "2459999",
        "2024-02-30",
    ],
)
def test_pre_v40_unsafe_source_stays_legacy_instead_of_bricking_heal(
    tmp_path, legacy_source_time
):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "Unsafe legacy source", legacy_source_time)]
        )
        chunk = _chunk(conn, "legacy-future", message_id)
        edge_id = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,derived) "
            "VALUES ('legacy','uses','future_db',1,'active',0)"
        ).lastrowid)
        with core_db.evidence_mutation(conn):
            conn.execute(
                "INSERT INTO kg_evidence(edge_id,chunk_id,evidence_kind,polarity,"
                "evidence_weight,weight_source,extracted_at) "
                "VALUES (?,?,'extraction',1,1,'legacy','2024-01-01T00:00:00Z')",
                (edge_id, chunk.id),
            )

        core_db._backfill_v40_chunk_manifests(conn)

        evidence = conn.execute(
            "SELECT provenance_status,source_message_id FROM kg_evidence"
        ).fetchone()
        assert tuple(evidence) == ("legacy_unattributed", None)
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle "
            "WHERE event_kind='claim_assertion'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_pre_v40_high_precision_source_uses_shared_canonical_clock(tmp_path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn,
            [("user", "Precise legacy source", "2024-01-01T00:00:00.9995Z")],
        )
        chunk = _chunk(conn, "legacy-precise", message_id)
        edge_id = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,derived) "
            "VALUES ('legacy','uses','precise_db',1,'active',0)"
        ).lastrowid)
        with core_db.evidence_mutation(conn):
            conn.execute(
                "INSERT INTO kg_evidence(edge_id,chunk_id,evidence_kind,polarity,"
                "evidence_weight,weight_source,extracted_at) "
                "VALUES (?,?,'extraction',1,1,'legacy','2024-01-01T00:00:01Z')",
                (edge_id, chunk.id),
            )

        core_db._backfill_v40_chunk_manifests(conn)

        row = conn.execute(
            "SELECT provenance_status,source_message_id,source_event_at "
            "FROM kg_evidence"
        ).fetchone()
        assert tuple(row) == (
            "canonical", message_id, "2024-01-01T00:00:00.999Z"
        )
    finally:
        conn.close()


@pytest.mark.parametrize("legacy_created_at", [None, "not-a-time"])
def test_direct_legacy_message_with_unknown_time_uses_ancient_source(tmp_path, legacy_created_at):
    conn = _open(tmp_path)
    try:
        message_id = int(conn.execute(
            "INSERT INTO messages(session_id,role,content,created_at) "
            "VALUES ('sources','user','Legacy app uses SQLite',?)",
            (legacy_created_at,),
        ).lastrowid)
        with core_db.transaction(conn):
            materialize_message_coverage(conn, "sources")
        chunk = _chunk(conn, "legacy-unknown-source", message_id)
        _persist(
            conn,
            HyMemConfig(root=tmp_path),
            chunk,
            Triple("legacy_app", "uses", "sqlite", 1,
                   source_message_id=message_id),
        )
        row = conn.execute(
            "SELECT source_created_at,source_event_at FROM kg_evidence"
        ).fetchone()
        assert row["source_created_at"] == legacy_created_at
        assert row["source_event_at"] == "0001-01-01T00:00:00.000Z"
    finally:
        conn.close()


def test_first_publication_survives_unchanged_and_changed_reextraction(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2020-01-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "publication-history", message_id)
        original = Triple(
            "app", "uses", "redis", 1, temporal_scope="initial",
            source_message_id=message_id,
        )
        _persist(conn, cfg, chunk, original, prompt_version="v13")
        first_id = int(conn.execute("SELECT id FROM kg_evidence").fetchone()[0])
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_evidence SET extracted_at=?,published_at=? WHERE id=?",
                (
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:01.000Z",
                    first_id,
                ),
            )
            conn.execute(
                "UPDATE kg_edge_lifecycle SET created_at=? "
                "WHERE source_evidence_id=?",
                ("2024-01-01T00:00:00Z", first_id),
            )
            conn.execute(
                "UPDATE kg_claim_observations SET observed_at=? WHERE evidence_id=?",
                ("2024-01-01T00:00:00Z", first_id),
            )
            conn.execute(
                "UPDATE kg_claim_extraction_outcomes SET succeeded_at=? "
                "WHERE chunk_id=?",
                ("2024-01-01T00:00:01Z", chunk.id),
            )

        cutoff = "2024-06-01T00:00:00Z"
        assert facts_at(conn, "2020-02-01", recorded_at=cutoff)[0].citations[
            0
        ].evidence_id == first_id

        # Same semantics update the mutable latest observation/outcome only.
        _persist(conn, cfg, chunk, original, prompt_version="v14")
        assert conn.execute(
            "SELECT published_at FROM kg_evidence WHERE id=?", (first_id,)
        ).fetchone()[0] == "2024-01-01T00:00:01.000Z"
        assert facts_at(conn, "2020-02-01", recorded_at=cutoff)[0].citations[
            0
        ].evidence_id == first_id

        # A changed interpretation gets a new publication, while the old
        # revision remains visible at a cutoff before its retirement.
        changed = replace(original, temporal_scope="reinterpreted")
        _persist(conn, cfg, chunk, changed, prompt_version="v15")
        old = facts_at(conn, "2020-02-01", recorded_at=cutoff)
        assert [citation.evidence_id for citation in old[0].citations] == [first_id]
        current = facts_at(conn, "2020-02-01")
        assert current[0].citations[0].evidence_id != first_id
    finally:
        conn.close()


def test_whole_chunk_publication_clock_gates_history_and_canonical_citation(
    tmp_path,
):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2023-06-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "publication-gap", message_id)
        _persist(
            conn, cfg, chunk,
            Triple("app", "uses", "redis", 1, source_message_id=message_id),
        )
        evidence_id = int(conn.execute("SELECT id FROM kg_evidence").fetchone()[0])
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_evidence SET extracted_at=?,published_at=? WHERE id=?",
                (
                    "2024-01-01T01:00:00.123456+01:00",
                    "2024-01-01T00:04:00.000Z",
                    evidence_id,
                ),
            )
            conn.execute(
                "UPDATE kg_edge_lifecycle SET created_at=? "
                "WHERE source_evidence_id=?",
                ("2024-01-01T00:00:00.124Z", evidence_id),
            )
            conn.execute(
                "UPDATE kg_claim_observations SET observed_at=? WHERE evidence_id=?",
                ("2024-01-01T00:00:00.124Z", evidence_id),
            )
            conn.execute(
                "UPDATE kg_claim_extraction_outcomes SET succeeded_at=? "
                "WHERE chunk_id=?",
                ("2024-01-01T00:04:00Z", chunk.id),
            )

        assert facts_at(
            conn, "2023-07-01", recorded_at="2024-01-01T00:02:00Z"
        ) == []
        published = facts_at(
            conn, "2023-07-01", recorded_at="2024-01-01T00:04:00Z"
        )
        assert published[0].citations[0].recorded_at == (
            "2024-01-01T00:04:00.000Z"
        )
        assert _objects(facts_at(conn, "2023-07-01")) == {"redis"}
    finally:
        conn.close()


def test_future_lifecycle_transaction_hides_every_current_projection(tmp_path):
    conn = _open(tmp_path)
    try:
        cfg = HyMemConfig(root=tmp_path)
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2024-01-01T00:00:00Z")]
        )
        _persist(
            conn, cfg, _chunk(conn, "future-lifecycle-record", message_id),
            Triple("app", "uses", "redis", 1, source_message_id=message_id),
        )
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_edge_lifecycle SET created_at="
                "'2100-01-01T00:00:00.000Z'"
            )

        assert _graph_lookup(
            conn, cfg, "app uses redis", ["app"], {}, frozenset({"uses"})
        ) == []
        assert timeline(conn, "app") == []
        assert count_relations(conn, count="subject").entities == []
        assert facts_at(conn, "2024-02-01") == []
        assert facts_at(
            conn, "2024-02-01", recorded_at="2026-01-01"
        ) == []
    finally:
        conn.close()


@pytest.mark.parametrize("value", ["", "not-a-date", "2024-02-30"])
def test_facts_at_rejects_malformed_public_timestamps(tmp_path, value):
    conn = _open(tmp_path)
    try:
        with pytest.raises(ValueError, match="valid_time"):
            facts_at(conn, value)
        with pytest.raises(ValueError, match="recorded_at"):
            facts_at(conn, "2024-01-01", recorded_at=value)
    finally:
        conn.close()
