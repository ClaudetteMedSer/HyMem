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


def test_no_predicate_entity_match_uses_entity_anchored_branch(hy_with_embed):
    """When entities match in the no-predicate fallback, candidates score by
    confidence × recency and carry `fallback:entity_anchored`. Source 2 is
    skipped, so no `fallback:semantic` or `semantic_*` codes leak through."""
    sid = "s1"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(sid, "user", "The backend is part of the platform.")
    hy_with_embed.close_session(sid)
    triples = [
        {"subject": "backend", "predicate": "part_of", "object": "platform", "polarity": 1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    hy_with_embed.dream()

    assert route_predicates("tell me about the backend") == frozenset()
    ctx = hy_with_embed.augment("tell me about the backend")
    assert ctx.graph_facts
    anchored = next(
        (f for f in ctx.graph_facts if f.subject == "backend"), None
    )
    assert anchored is not None
    assert "entity_match" in anchored.why_retrieved
    assert "fallback:entity_anchored" in anchored.why_retrieved
    for fact in ctx.graph_facts:
        assert "fallback:semantic" not in fact.why_retrieved
        assert not any(r.startswith("semantic_") for r in fact.why_retrieved)


def test_no_predicate_no_entities_uses_semantic_branch(hy_with_embed, monkeypatch):
    """When no entity matches in the no-predicate fallback, Source 2 still
    runs and ranks edges by pure similarity with `fallback:semantic`."""
    sid = "s1"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(sid, "user", "The backend is part of the platform.")
    hy_with_embed.close_session(sid)
    triples = [
        {"subject": "backend", "predicate": "part_of", "object": "platform", "polarity": 1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    hy_with_embed.dream()

    # Force Source 2 to produce a high-similarity candidate deterministically:
    # return the only seeded edge's id with sim=1.0. The query is generic and
    # matches no entity, so entity-anchored branch can't fire.
    edge_id = hy_with_embed.conn.execute(
        "SELECT id FROM knowledge_graph WHERE subject_canonical = 'backend'"
    ).fetchone()["id"]
    from hymem.query import augment as augment_mod
    monkeypatch.setattr(
        augment_mod, "_semantic_edge_hits",
        lambda conn, cfg, embedder, query: [(edge_id, 1.0)],
    )

    query = "hello world generic phrase"
    assert route_predicates(query) == frozenset()
    ctx = hy_with_embed.augment(query)
    assert ctx.matched_entities == []
    assert ctx.graph_facts
    for fact in ctx.graph_facts:
        assert "fallback:semantic" in fact.why_retrieved
    assert any(r.startswith("semantic_") for f in ctx.graph_facts for r in f.why_retrieved)


def test_fallback_skips_semantic_knn_when_entities_match(hy_with_embed, monkeypatch):
    """Regression for the YantrikDB failure: when entities match in the
    fallback path, noisy edge-level KNN must not run at all. Otherwise a
    surface-similarity edge like `deepseek_embedding rejects working` can
    out-score the genuinely entity-anchored `medflow part_of atta_projects`.
    """
    sid = "s1"
    hy_with_embed.open_session(sid)
    hy_with_embed.log_message(
        sid, "user", "Atta is part of the medflow project team."
    )
    hy_with_embed.close_session(sid)
    triples = [
        {"subject": "atta", "predicate": "part_of", "object": "medflow", "polarity": 1},
    ]
    hy_with_embed.set_llm(make_routed_llm(triples, []))
    hy_with_embed.dream()

    from hymem.query import augment as augment_mod

    def _explode(*args, **kwargs):
        raise AssertionError(
            "_semantic_edge_hits must not run when entities match in fallback"
        )

    monkeypatch.setattr(augment_mod, "_semantic_edge_hits", _explode)

    query = "tell me about atta"
    assert route_predicates(query) == frozenset()
    ctx = hy_with_embed.augment(query)
    assert ctx.matched_entities == ["atta"]
    assert ctx.graph_facts
    for fact in ctx.graph_facts:
        assert "fallback:entity_anchored" in fact.why_retrieved


def test_no_predicate_no_embeddings_falls_back_to_recency(hy):
    """With no embedder and no entity match, the fallback still returns
    recent edges ranked by confidence × recency, tagged fallback:recency."""
    conn = hy.conn
    seed_edge(conn, "library_recent", "part_of", "ecosystem", days_ago=1)
    seed_edge(conn, "library_stale", "part_of", "ecosystem", days_ago=400)

    # No predicate cues, no entity tokens that overlap the seeded subjects.
    assert route_predicates("hello world this is a generic query") == frozenset()
    ctx = hy.augment("hello world this is a generic query")
    assert ctx.matched_entities == []
    assert ctx.graph_facts
    for fact in ctx.graph_facts:
        assert "fallback:recency" in fact.why_retrieved
    by_subj = {f.subject: f for f in ctx.graph_facts}
    assert "library_recent" in by_subj and "library_stale" in by_subj
    assert by_subj["library_recent"].score > by_subj["library_stale"].score


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

    # Spy on the python-cosine search to verify it was actually exercised.
    # A routed query keeps Source 2 active (it isn't skipped in the routed
    # branch, only in the no-predicate entity-anchored fallback).
    from hymem.query import augment as augment_mod
    calls: list[bool] = []
    orig = augment_mod._python_cosine_edge_search

    def _spy(*args, **kwargs):
        calls.append(True)
        return orig(*args, **kwargs)

    monkeypatch.setattr(augment_mod, "_python_cosine_edge_search", _spy)

    ctx = hy_with_embed.augment("what technologies does the backend use")
    assert ctx.graph_facts
    assert calls, "python-cosine edge search should run when vec extension is off"


def test_recency_weight_math():
    half_life = 30.0
    assert math.isclose(math.exp(-0.0 / half_life), 1.0)
    assert math.isclose(math.exp(-half_life / half_life), math.exp(-1.0))
