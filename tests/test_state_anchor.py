"""Plan D — state-anchor selection and seed terms (`hymem/query/state_anchor.py`).

A state anchor seeds a secondary expansion from what is CURRENTLY TRUE, so that
evidence rows sharing no lexical or vector overlap with the query — but overlap
with the answer state — become reachable. The selection is the digest anchor's
predicate (`aggregate.py:826-834`) with ONE deliberate deviation, banked before
the shadow probe ran: separate caps for profile and edges instead of the shared
one, and no early return.

`test_edges_are_not_starved_by_a_large_profile` is the test that carries that
deviation: `_anchor_facts` returns profile-only once profile rows fill the shared
cap, which on the production box means 0 of 8754 active edges. Copying that into
a SEED source would make the tier inert and close Plan D for the wrong reason.
`test_the_predicate_matches_the_digest_anchor_row_for_row` is its control — the
deviation must be the cap and nothing else, or the tier stops anchoring on the
state the digest considers true.
"""
from __future__ import annotations

import json

import pytest

from hymem import HyMem, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.dreaming.aggregate import _anchor_facts
from hymem.extraction.llm import StubLLMClient
from hymem.query.state_anchor import (
    expand_over_chunks,
    expand_over_messages,
    seed_terms_from_edges,
    select_anchor_edges,
    select_anchor_profile,
)


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    yield hy.conn
    hy.close()


def _edge(conn, subject, predicate, obj, *, pos=3, neg=0, status="active",
          derived=0, invalid_at=None, last_seen="2024-06-01 00:00:00") -> int:
    cur = conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, first_seen, last_seen, last_reinforced, "
        "invalid_at, status, derived) "
        "VALUES (?, ?, ?, ?, ?, '2024-01-01 00:00:00', ?, ?, ?, ?, ?)",
        (subject, predicate, obj, pos, neg, last_seen, last_seen,
         invalid_at, status, derived),
    )
    return cur.lastrowid


def _profile(conn, slot, value, slot_key=None) -> None:
    conn.execute(
        "INSERT INTO user_profile(slot, slot_key, value, confidence) VALUES (?,?,?,1.0)",
        (slot, slot_key, value),
    )


# ── selection: the digest predicate, verbatim ───────────────────────────────

def test_only_active_non_derived_valid_margin_positive_edges_are_selected(conn):
    with core_db.transaction(conn):
        _edge(conn, "app", "uses", "postgres")                       # keep
        _edge(conn, "app", "uses", "mysql", status="retracted")      # dropped
        _edge(conn, "app", "uses", "redis", derived=1)               # dropped
        _edge(conn, "app", "uses", "mongo", pos=1, neg=4)            # dropped: margin
        _edge(conn, "app", "uses", "kafka",
              invalid_at="2024-03-01 00:00:00")                      # dropped: superseded

    edges = select_anchor_edges(conn, edge_cap=20)

    assert [e["o"] for e in edges] == ["postgres"]


def test_edges_are_ordered_by_evidence_margin_then_recency(conn):
    """The digest anchor's ORDER BY, verbatim. Ordering is load-bearing: the cap
    turns it into a selection, so a different order seeds a different tier."""
    with core_db.transaction(conn):
        _edge(conn, "a", "uses", "weak", pos=2, neg=1)
        _edge(conn, "a", "uses", "strong", pos=9, neg=0)
        _edge(conn, "a", "uses", "recent", pos=2, neg=1,
              last_seen="2025-01-01 00:00:00")

    edges = select_anchor_edges(conn, edge_cap=20)

    assert [e["o"] for e in edges] == ["strong", "recent", "weak"]


def test_the_predicate_matches_the_digest_anchor_row_for_row(conn):
    """The control for the cap deviation. With a cap large enough that
    `_anchor_facts` never truncates, the selector must return exactly the edges
    the digest considers true — same filter, same order. If this drifts, the
    tier is anchoring on a different state than the digest does."""
    with core_db.transaction(conn):
        _edge(conn, "app", "uses", "postgres", pos=9)
        _edge(conn, "app", "deploys_to", "fly", pos=4)
        _edge(conn, "app", "uses", "mysql", status="retracted")
        _edge(conn, "app", "uses", "mongo", pos=1, neg=4)

    rendered = [f"{e['s']} {e['p']} {e['o']}" for e in select_anchor_edges(conn, edge_cap=50)]

    assert rendered == _anchor_facts(conn, 50)


def test_edges_are_not_starved_by_a_large_profile(conn):
    """THE deviation. `_anchor_facts` gives profile rows the whole cap and
    returns early (`aggregate.py:823-824`) — on the box, 22 profile rows against
    cap=20 leaves 0 of 8754 edges. That is right for a prompt block and fatal for
    a seed source, so the selector budgets the two independently."""
    with core_db.transaction(conn):
        for i in range(25):
            _profile(conn, "relationship", f"person{i}", slot_key=f"k{i}")
        _edge(conn, "app", "uses", "postgres")

    assert _anchor_facts(conn, 20) and "postgres" not in " ".join(_anchor_facts(conn, 20))
    assert [e["o"] for e in select_anchor_edges(conn, edge_cap=20)] == ["postgres"]


def test_each_cap_is_respected_independently(conn):
    with core_db.transaction(conn):
        for i in range(5):
            _profile(conn, "relationship", f"person{i}", slot_key=f"k{i}")
        for i in range(5):
            _edge(conn, f"app{i}", "uses", "postgres")

    assert len(select_anchor_edges(conn, edge_cap=2)) == 2
    assert len(select_anchor_profile(conn, profile_cap=3)) == 3


def test_profile_is_a_separate_seed_source_with_its_own_budget(conn):
    """Profile rows carry the identity facts the 22-predicate graph vocabulary
    can never mint (the P4 Stage-0 finding), so they are seeded too — but as
    their OWN source, so the probe can attribute a hit to edges vs profile
    instead of reporting one undifferentiated number."""
    with core_db.transaction(conn):
        _profile(conn, "employer", "acme_health")
        _edge(conn, "app", "uses", "postgres")

    terms = seed_terms_from_edges(select_anchor_edges(conn, edge_cap=20),
                                  select_anchor_profile(conn, profile_cap=20))

    assert "acme_health" in terms
    assert "postgres" in terms


def test_a_zero_cap_disables_the_source(conn):
    """House convention: 0 disables, so the tier can be turned off per-source
    without a flag."""
    with core_db.transaction(conn):
        _edge(conn, "app", "uses", "postgres")

    assert select_anchor_edges(conn, edge_cap=0) == []


def test_a_pre_migration_store_degrades_to_empty(conn):
    """Same contract as every other retrieval helper: a missing table returns []
    rather than raising, so an old store degrades instead of breaking augment."""
    with core_db.transaction(conn):
        conn.execute("DROP TABLE knowledge_graph")

    assert select_anchor_edges(conn, edge_cap=20) == []


# ── seed terms ──────────────────────────────────────────────────────────────

def test_seed_terms_carry_subject_predicate_and_object(conn):
    with core_db.transaction(conn):
        _edge(conn, "hymem_api", "deploys_to", "fly_io")

    terms = seed_terms_from_edges(select_anchor_edges(conn, edge_cap=20))

    assert set(terms) == {"hymem_api", "deploys_to", "fly_io"}


def test_seed_terms_are_deduped_across_edges(conn):
    """Anchor sets are subject-heavy — the same subject recurs across predicates.
    Without dedup the FTS query degenerates into one term repeated N times."""
    with core_db.transaction(conn):
        _edge(conn, "app", "uses", "postgres")
        _edge(conn, "app", "uses", "redis")

    terms = seed_terms_from_edges(select_anchor_edges(conn, edge_cap=20))

    assert sorted(terms) == ["app", "postgres", "redis", "uses"]


def test_no_edges_means_no_terms_and_therefore_no_cost(conn):
    """C3 is a cost gate: an empty anchor set must short-circuit before any
    search runs, not issue an empty FTS query."""
    assert seed_terms_from_edges([]) == []


# ── expansion core ──────────────────────────────────────────────────────────
# Two corpora, probed separately and shipped at most one (the D5 rule): raw
# messages are BM25-only — `_VEC_TABLES` (core/db.py:156) has no `vec_messages`
# — while chunks carry both arms. Which one pays is a measurement, not a guess.

def _session_with_turns(hy, sid, turns):
    hy.open_session(sid)
    for role, text in turns:
        hy.log_message(sid, role, text)
    hy.close_session(sid)


def _chunk(hy, chunk_id, text, sid="s0"):
    """A chunk row plus its FTS shadow — what persist_chunks writes."""
    hy.open_session(sid)
    mid = hy.log_message(sid, "user", text)
    hy.close_session(sid)
    with core_db.transaction(hy.conn):
        hy.conn.execute(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text) VALUES (?, ?, ?, ?, 'test', ?)",
            (chunk_id, sid, mid, mid, text),
        )
        hy.conn.execute(
            "INSERT INTO chunks_fts(rowid, text) "
            "SELECT rowid, text FROM chunks WHERE id = ?", (chunk_id,)
        )


def test_chunk_expansion_reaches_a_row_the_query_never_could(hy):
    """The mechanism, stated as a test: the query says "what do we run on", the
    evidence says "pinned 15.4 after the vacuum incident", and the only bridge
    is the anchor term `postgres`. If this fails the tier has no mechanism."""
    _chunk(hy, "c1", "we pinned postgres 15.4 after the vacuum incident")

    hits = expand_over_chunks(hy.conn, ["postgres"], top_k=5)

    assert [h.chunk_id for h in hits] == ["c1"]


def test_message_expansion_uses_the_raw_turn_corpus(hy):
    _session_with_turns(hy, "s1", [
        ("user", "we pinned postgres 15.4 after the vacuum incident"),
    ])

    hits = expand_over_messages(hy.conn, ["postgres"], top_k=5)

    assert len(hits) == 1
    assert "vacuum incident" in hits[0].text


def test_no_seed_terms_costs_zero_queries(hy):
    """C3 is a cost gate, so this counts STATEMENTS, not results.

    Asserting on the return value alone would measure nothing: both helpers give
    [] on an empty seed set whether or not they short-circuit, because an empty
    FTS query matches nothing. This is a CONTRACT test, not a control for one
    line — the zero-SQL guarantee is currently provided twice (here, and again
    by the `_fts_search` / `_message_fts_search` token guards), and it must
    survive either one being refactored away."""
    executed: list[str] = []
    hy.conn.set_trace_callback(executed.append)
    try:
        assert expand_over_chunks(hy.conn, [], top_k=5) == []
        assert expand_over_messages(hy.conn, [], top_k=5) == []
    finally:
        hy.conn.set_trace_callback(None)

    assert executed == []


def test_expansion_respects_top_k(hy):
    _session_with_turns(hy, "s1", [
        ("user", f"postgres note number {i}") for i in range(8)
    ])

    assert len(expand_over_messages(hy.conn, ["postgres"], top_k=3)) == 3


def test_chunk_expansion_costs_exactly_one_embedding_call(hy):
    """C3: <=1 vector call per query. Seeds are joined into ONE query string, so
    the whole anchor set embeds once however many edges it holds.

    The assertion is `== 1`, not `<= 1`, deliberately. `_python_cosine_search`
    returns before embedding when no chunk_embeddings rows exist, so a store
    without persisted vectors makes this test pass with ZERO calls while
    exercising nothing — the vacuous-pass trap. Persisting one vector first is
    what makes the count a measurement."""
    embedder = StubEmbeddingClient()
    _chunk(hy, "c1", "postgres notes")
    with core_db.transaction(hy.conn):
        hy.conn.execute(
            "INSERT INTO chunk_embeddings(chunk_id, vector_json, model, dim) "
            "VALUES ('c1', ?, ?, ?)",
            (json.dumps(embedder.embed(["postgres notes"])[0]),
             embedder.model, embedder.dim),
        )
    embedder.calls.clear()

    hits = expand_over_chunks(hy.conn, ["postgres", "uses", "app", "redis", "kafka"],
                              top_k=5, embedding_client=embedder)

    assert len(embedder.calls) == 1, "the vector arm did not run — test is vacuous"
    assert [h.chunk_id for h in hits] == ["c1"]


def test_expansion_degrades_on_a_pre_migration_store(hy):
    with core_db.transaction(hy.conn):
        hy.conn.execute("DROP TABLE messages_fts")

    assert expand_over_messages(hy.conn, ["postgres"], top_k=5) == []
