from __future__ import annotations

import math

from hymem.core import db as core_db
from hymem.query.conflicts import find_conflicts
from hymem.query.predicate_routing import route_predicates
from tests.conftest import make_routed_llm, seed_edge


# --- schema migration v6 ----------------------------------------------------


def test_migration_v6_creates_edge_embeddings(hy):
    conn = hy.conn
    assert core_db.schema_version(conn) == 6
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='edge_embeddings'"
    ).fetchone()
    assert row is not None
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(dream_runs)").fetchall()}
    assert "edges_embedded" in cols


# --- embed_pending_edges ----------------------------------------------------


def _dream_with_edges(hy_with_embed):
    sid = "s1"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(sid, "user", "We use fast_api for the backend service.")
    hy_with_embed.log_message(sid, "assistant", "Got it, fast_api it is.")
    hy_with_embed.close_session(sid)
    triples = [
        {"subject": "backend", "predicate": "uses", "object": "fast_api", "polarity": 1},
        {"subject": "backend", "predicate": "depends_on", "object": "postgres", "polarity": 1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    return hy_with_embed.dream()


def test_embed_pending_edges_populates_tables(hy_with_embed):
    report = _dream_with_edges(hy_with_embed)
    assert report.edges_embedded >= 2

    conn = hy_with_embed.conn
    texts = {r["edge_text"] for r in conn.execute("SELECT edge_text FROM edge_embeddings")}
    assert "backend uses fast_api" in texts
    assert "backend depends_on postgres" in texts

    vec_count = conn.execute("SELECT COUNT(*) AS c FROM vec_edges").fetchone()["c"]
    assert vec_count >= 2


def test_embed_pending_edges_idempotent(hy_with_embed, embed_stub):
    _dream_with_edges(hy_with_embed)
    calls_after_first = len(embed_stub.calls)
    report2 = hy_with_embed.dream()
    # No new triples -> no new edge texts -> no new embedding API calls.
    assert report2.edges_embedded == 0
    assert len(embed_stub.calls) == calls_after_first


# --- hybrid ranker & why_retrieved -----------------------------------------


def test_graph_facts_carry_why_retrieved(hy_with_embed):
    _dream_with_edges(hy_with_embed)
    ctx = hy_with_embed.augment("what technologies does the backend use")
    assert ctx.graph_facts
    for fact in ctx.graph_facts:
        assert fact.why_retrieved, f"{fact} has no reason codes"
    # Predicate routing on "use" should tag the uses edge.
    uses_fact = next(
        (f for f in ctx.graph_facts if f.predicate == "uses"), None
    )
    assert uses_fact is not None
    assert any(r == "predicate:uses" for r in uses_fact.why_retrieved)
    assert any(r.startswith("semantic_") for r in uses_fact.why_retrieved)


def test_no_embedder_path_uses_entity_match(hy):
    sid = "s1"
    hy.open_session(sid)
    hy.log_message(sid, "user", "We use fast_api for the backend.")
    hy.close_session(sid)
    triples = [
        {"subject": "backend", "predicate": "uses", "object": "fast_api", "polarity": 1},
    ]
    hy.set_llm(make_routed_llm(triples, []))
    hy.dream()

    ctx = hy.augment("tell me about fast_api")
    assert ctx.graph_facts
    fact = ctx.graph_facts[0]
    assert "entity_match" in fact.why_retrieved
    # No embedding client -> no semantic reason code.
    assert not any(r.startswith("semantic_") for r in fact.why_retrieved)


def test_recency_reason_code(hy):
    conn = hy.conn
    seed_edge(conn, "alpha", "uses", "recent_lib", days_ago=2)
    seed_edge(conn, "alpha", "uses", "old_lib", days_ago=400)

    ctx = hy.augment("tell me about alpha")
    by_obj = {f.object: f for f in ctx.graph_facts}
    assert "recency_2d" in by_obj["recent_lib"].why_retrieved
    assert not any(
        r.startswith("recency_") for r in by_obj["old_lib"].why_retrieved
    )
    # Recent edge outranks the stale one.
    assert by_obj["recent_lib"].score > by_obj["old_lib"].score


# --- predicate routing (pure) ----------------------------------------------


def test_route_predicates():
    assert route_predicates("what technologies does it use") >= {"uses", "runs_on"}
    assert route_predicates("what does alpha depend on") == frozenset({"depends_on"})
    assert route_predicates("just a normal sentence") == frozenset()


# --- conflicts --------------------------------------------------------------


def test_conflicts_competing_objects(hy):
    conn = hy.conn
    seed_edge(conn, "atta", "prefers", "english")
    seed_edge(conn, "atta", "prefers", "dutch")
    conflicts = hy.conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].kind == "competing_object"
    assert conflicts[0].subject == "atta"


def test_conflicts_opposing_predicates(hy):
    conn = hy.conn
    seed_edge(conn, "team", "prefers", "docker")
    seed_edge(conn, "team", "rejects", "docker")
    conflicts = hy.conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].kind == "opposing_predicate"


def test_conflicts_consistent_graph_is_empty(hy):
    conn = hy.conn
    seed_edge(conn, "team", "uses", "docker")
    seed_edge(conn, "team", "depends_on", "postgres")
    assert hy.conflicts() == []


def test_conflicts_ignores_retracted_and_derived(hy):
    conn = hy.conn
    seed_edge(conn, "atta", "prefers", "english")
    seed_edge(conn, "atta", "prefers", "dutch", status="retracted")
    seed_edge(conn, "atta", "prefers", "german", derived=1)
    assert hy.conflicts() == []


# --- backwards-compat: sqlite-vec absent -----------------------------------


def test_semantic_fallback_without_sqlite_vec(hy_with_embed, monkeypatch):
    _dream_with_edges(hy_with_embed)
    # Force the Python-cosine edge path.
    monkeypatch.setattr(core_db, "_load_vec_extension", lambda conn: False)
    ctx = hy_with_embed.augment("what does the backend use")
    assert ctx.graph_facts
    assert any(
        r.startswith("semantic_")
        for f in ctx.graph_facts
        for r in f.why_retrieved
    )


def test_recency_weight_math():
    half_life = 30.0
    assert math.isclose(math.exp(-0.0 / half_life), 1.0)
    assert math.isclose(math.exp(-half_life / half_life), math.exp(-1.0))
