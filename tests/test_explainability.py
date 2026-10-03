"""Tests for retrieval explainability chips (improv item H).

FtsHit / EpisodeHit / ProcedureHit now carry `why_retrieved` reason chips,
mirroring GraphFact, so a consumer can quote why a hit surfaced.
"""

from __future__ import annotations

from hymem.query.augment import FtsHit, _rrf_merge


def test_fts_hit_carries_match_chip(hy):
    conn = hy.conn
    conn.execute("INSERT INTO sessions(id) VALUES ('s1')")
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) "
        "VALUES ('c1', 's1', 1, 1, 'long_user_turn', 'we deploy postgres to production')"
    )
    ctx = hy.augment("postgres deploy")
    assert ctx.fts_hits
    chips = ctx.fts_hits[0].why_retrieved
    assert any(c.startswith("fts_match(") for c in chips), chips


def test_procedure_hit_carries_chip(hy):
    conn = hy.conn
    conn.execute("INSERT INTO sessions(id) VALUES ('s1')")
    conn.execute(
        "INSERT INTO procedures(id, session_id, name, description, steps) "
        "VALUES ('p1', 's1', 'Deploy to staging', 'build and push image', '[]')"
    )
    ctx = hy.augment("deploy staging")
    assert ctx.procedures
    assert any(
        c.startswith("procedure_fts(") for c in ctx.procedures[0].why_retrieved
    ), ctx.procedures[0].why_retrieved


def test_episode_hit_carries_chip(hy):
    conn = hy.conn
    conn.execute("INSERT INTO sessions(id) VALUES ('s1')")
    conn.execute(
        "INSERT INTO episodes(id, session_id, title, summary) "
        "VALUES ('e1', 's1', 'Postgres setup', 'We configured the postgres connection pool')"
    )
    ctx = hy.augment("postgres pool")
    assert ctx.episodes
    assert any(
        c.startswith("episode_fts(") for c in ctx.episodes[0].why_retrieved
    ), ctx.episodes[0].why_retrieved


def test_rrf_merge_composes_source_and_score_chips():
    """A chunk present in both ranked lists gets its source chip plus a fused
    rrf(...) chip naming both contributors."""
    fts = [FtsHit("c1", "s", "t", 1.0, why_retrieved=['fts_match("postgres")'])]
    vec = [FtsHit("c1", "s", "t", 0.9, score_kind="vec",
                  why_retrieved=["vec_topk(sim=0.900)"])]
    merged = _rrf_merge(fts, vec, top_k=5)
    chips = merged[0].why_retrieved
    assert merged[0].score_kind == "rrf"
    assert any(c.startswith("fts_match(") for c in chips), chips
    assert any(c.startswith("rrf(fts+vec,") for c in chips), chips


def test_rrf_chip_marks_single_source():
    """A chunk only in the FTS list is tagged rrf(fts, ...)."""
    fts = [FtsHit("c1", "s", "t", 1.0, why_retrieved=['fts_match("x")'])]
    merged = _rrf_merge(fts, [], top_k=5)
    assert any(c.startswith("rrf(fts,") for c in merged[0].why_retrieved), \
        merged[0].why_retrieved
