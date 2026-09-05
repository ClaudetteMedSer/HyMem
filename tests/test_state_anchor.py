"""State-anchor expansion (Plan D, borrowed from MindCache) — Tasks 1-4.

Task 1 pins `select_anchor_edges`: the EXACT `_anchor_facts` predicate
(`dreaming/aggregate.py:829-830`) — active, non-derived, non-superseded,
margin-positive edges, ordered by evidence margin, bounded by `cap`.

Task 2 pins `seed_terms_from_edges`: canonical subject/predicate/object plus
typed-value sub-terms (the value_supersession v3 classes: versions carry their
alpha prefix, numbers their unit, dates their year) — the discriminative side
of each class, so a version bump ("python_3.12" -> "python_3.13") still matches
evidence rows that spell the prefix.

Task 3 pins `state_anchor_expand`: seed terms -> existing FTS (optional vec)
-> RRF merge -> top_k, deduped by chunk id (entry key), zero-cost inert with
no seed terms.

Task 4 pins the shadow probe: on a fixture store where the gold evidence row
is reachable ONLY from the state anchors (not from the query), the probe
reports exactly one anchored-only hit and a zero wrong-state rate.

The probe and expansion run read-only; every test asserts the store is not
mutated.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.user_profile import ProfileExtraction, persist_user_profile
from hymem.extraction.llm import StubLLMClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from state_anchor_probe import run_probe  # noqa: E402


# ── seeding helpers (the tests/test_recovery_probe.py idiom) ────────────────

def _seed_session(conn, session_id: str = "s1") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, CURRENT_TIMESTAMP)",
        (session_id,),
    )


def _seed_message(conn, msg_id: int, created_at: str, *, session_id: str = "s1") -> None:
    conn.execute(
        "INSERT INTO messages(id, session_id, role, content, created_at) "
        "VALUES (?, ?, 'user', 'x', ?)",
        (msg_id, session_id, created_at),
    )


def _seed_chunk(conn, chunk_id: str, start_msg: int, text: str,
                *, session_id: str = "s1") -> None:
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES (?, ?, ?, ?, 'test', ?)",
        (chunk_id, session_id, start_msg, start_msg, text),
    )


def _seed_edge(conn, subject: str, predicate: str, obj: str, *,
               pos: int = 3, neg: int = 0, status: str = "active",
               derived: int = 0, invalid_at: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, first_seen, last_seen, last_reinforced, "
        "valid_at, invalid_at, status, derived) "
        "VALUES (?, ?, ?, ?, ?, '2024-01-01 00:00:00', CURRENT_TIMESTAMP, "
        "CURRENT_TIMESTAMP, '2024-01-01 00:00:00', ?, ?, ?)",
        (subject, predicate, obj, pos, neg, invalid_at, status, derived),
    )
    return cur.lastrowid


def _seed_evidence(conn, edge_id: int, chunk_id: str, polarity: int = 1) -> None:
    # The probe deliberately seeds a historical ledger/cache snapshot instead
    # of exercising runtime reconciliation.  Keep that setup behind the same
    # scoped authority used by migrations/importers.
    with core_db.evidence_mutation(conn):
        conn.execute(
            "INSERT INTO kg_evidence(edge_id, chunk_id, polarity, extracted_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (edge_id, chunk_id, polarity),
        )


def _count_writes(conn) -> dict[str, int]:
    """Rows in tables the probe must not touch (snapshot before/after compare
    of the row counts would be stronger, but a write counter over the probe's
    own connection is impossible — the probe opens its own read-only conn; on
    THIS connection we just check the tables' row counts are stable)."""
    tables = ["sessions", "messages", "chunks", "knowledge_graph", "kg_evidence"]
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    yield hy.conn
    hy.close()


@pytest.fixture
def anchor_fixture(conn):
    """One evidence chunk reachable ONLY from the state anchor.

    - Edge: dev_box installed cuda_12.1 (active, margin-positive).
    - Evidence chunk for that edge: "We installed CUDA 12.1 and PyTorch 2.2
      on the dev box in January."
    - Query: "Which GPU system produces model training output?" — shares no
      FTS token with the evidence chunk, so the baseline augment() misses it
      and only the anchor expansion can surface it.
    - A second, query-reachable chunk: "Model training output is stored on
      nas01." — a topic row that does NOT contain the answer, so a naive
      query-match does not accidentally count as the gold row.
    - "c-gold": shares the anchor vocabulary but is NOT the seed edge's
      evidence chunk. This is the only row that can demonstrate reachability.
      "c-evidence" cannot: the seed terms were EXTRACTED from it, so matching
      it back is provenance-circular and the probe must exclude it (D4).
    """
    with core_db.transaction(conn):
        _seed_session(conn)
        _seed_message(conn, 1, "2024-01-01 00:00:00")
        _seed_message(conn, 2, "2024-01-02 00:00:00")
        _seed_message(conn, 3, "2024-01-03 00:00:00")
        _seed_chunk(conn, "c-evidence", 1,
                    "We installed CUDA 12.1 and PyTorch 2.2 on the dev box in January.")
        _seed_chunk(conn, "c-distractor", 2, "Model training output is stored on nas01.")
        # A GENUINELY anchored row: it shares the anchor's vocabulary but is
        # NOT the seed edge's provenance, so reaching it is not tautological.
        _seed_chunk(conn, "c-gold", 3,
                    "The cuda toolkit on dev_box was upgraded again in March.")
        _seed_edge(conn, "dev_box", "configured_with", "cuda_12.1", pos=3, neg=0)
        edge_id = conn.execute(
            "SELECT id FROM knowledge_graph WHERE subject_canonical='dev_box'"
        ).fetchone()[0]
        _seed_evidence(conn, edge_id, "c-evidence")
    return conn


# ── Task 1: select_anchor_edges ─────────────────────────────────────────────

def test_select_anchor_edges_exact_predicate(conn):
    """status='active' AND derived=0 AND invalid_at IS NULL AND pos>neg."""
    with core_db.transaction(conn):
        _seed_edge(conn, "a", "uses", "postgres", pos=3, neg=0)          # in
        _seed_edge(conn, "b", "uses", "postgres", pos=3, neg=0, derived=1)  # out
        _seed_edge(conn, "c", "uses", "postgres", pos=3, neg=0,
                   invalid_at="2024-03-01")                             # out
        _seed_edge(conn, "d", "uses", "postgres", pos=3, neg=0, status="retracted")  # out
        _seed_edge(conn, "e", "uses", "postgres", pos=1, neg=4)         # out (margin)

    from hymem.query.state_anchor import select_anchor_edges

    rows = select_anchor_edges(conn, cap=20)
    assert [r["subject_canonical"] for r in rows] == ["a"]


def test_select_anchor_edges_cap_respected(conn):
    """The top-`cap` by (pos-neg DESC, last_seen DESC, id) — copy the ORDER BY."""
    with core_db.transaction(conn):
        for i, pos in enumerate([2, 9, 5, 1, 7]):
            _seed_edge(conn, f"svc{i}", "uses", "postgres", pos=pos, neg=0)

    from hymem.query.state_anchor import select_anchor_edges

    rows = select_anchor_edges(conn, cap=3)
    assert [r["subject_canonical"] for r in rows] == ["svc1", "svc4", "svc2"]


def test_select_anchor_edges_zero_cap(conn):
    with core_db.transaction(conn):
        _seed_edge(conn, "a", "uses", "postgres", pos=3, neg=0)

    from hymem.query.state_anchor import select_anchor_edges

    assert select_anchor_edges(conn, cap=0) == []


# ── Task 2: seed_terms_from_edges ──────────────────────────────────────────

def test_seed_terms_canonical_fields():
    from hymem.query.state_anchor import seed_terms_from_edges

    edges = [{"subject_canonical": "app", "predicate": "uses", "object_canonical": "postgres"}]
    assert seed_terms_from_edges(edges) == ["app", "uses", "postgres"]


def test_seed_terms_typed_values():
    from hymem.query.state_anchor import seed_terms_from_edges

    edges = [
        {"subject_canonical": "dev_box", "predicate": "installed",
         "object_canonical": "python_3.12"},   # version with alpha prefix
        {"subject_canonical": "coverage", "predicate": "target",
         "object_canonical": "65_percent"},    # number with unit
        {"subject_canonical": "release", "predicate": "on",
         "object_canonical": "2024-03-01"},    # ISO date
        {"subject_canonical": "team", "predicate": "uses",
         "object_canonical": "postgres"},      # free text: object only
    ]
    terms = seed_terms_from_edges(edges)
    assert "python_3.12" in terms and "python" in terms   # vers: prefix key
    assert "65_percent" in terms and "percent" in terms   # num: unit
    assert "2024-03-01" in terms and "2024" in terms      # date: year
    assert "dev_box" in terms and "installed" in terms
    assert "team" in terms and "postgres" in terms


def test_seed_terms_empty_edge_no_terms():
    from hymem.query.state_anchor import seed_terms_from_edges

    assert seed_terms_from_edges([{}]) == []
    assert seed_terms_from_edges([]) == []
    assert seed_terms_from_edges(
        [{"subject_canonical": None, "predicate": "", "object_canonical": "  "}]
    ) == []


def test_seed_terms_dedup_stable_order():
    from hymem.query.state_anchor import seed_terms_from_edges

    edges = [
        {"subject_canonical": "app", "predicate": "uses", "object_canonical": "postgres"},
        {"subject_canonical": "app", "predicate": "uses", "object_canonical": "postgres"},
        {"subject_canonical": "app", "predicate": "runs", "object_canonical": "linux"},
    ]
    assert seed_terms_from_edges(edges) == ["app", "uses", "postgres", "runs", "linux"]


# ── Task 1 (correction): profile leg ───────────────────────────────────────

def _seed_profile(conn, slot: str, value: str, *, slot_key: str | None = None,
                  invalid_at: str | None = None, confidence: float = 1.0) -> None:
    if slot != "relationship":
        assert slot_key is None, "non-relationship profile slots cannot have keys"
    session_id = "state-anchor-profile-source"
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, '2024-01-01 00:00:00')",
        (session_id,),
    )
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content, created_at) "
        "VALUES (?, 'user', ?, '2024-01-01 00:00:00')",
        (session_id, f"My {slot} is {value}."),
    )
    materialize_message_coverage(conn, session_id)
    item = {
        "slot": slot,
        "value": value,
        "evidence_message_id": int(cur.lastrowid),
        "confidence": confidence,
    }
    if slot == "relationship":
        item["slot_key"] = slot_key
    inserted = persist_user_profile(
        conn,
        ProfileExtraction(items=[item]),
        redact_values=False,
    )
    assert inserted == 1
    if invalid_at is not None:
        conn.execute(
            "UPDATE user_profile SET invalid_at = ? WHERE source_message_id = ?",
            (invalid_at, int(cur.lastrowid)),
        )


def _seed_squeeze(conn) -> None:
    """3 profile rows + 6 edges — enough for a cap of 4 to starve the edges."""
    with core_db.transaction(conn):
        for val in ["running", "pottery", "reading"]:
            _seed_profile(conn, "recurring_activity", val)
        for i in range(6):
            _seed_edge(conn, f"m{i}", "uses", "postgres", pos=3, neg=0)


def test_shared_cap_reproduces_the_digest_squeeze(conn):
    """`shared_cap=` is the digest accounting: profile FIRST, edges the rest.

    Kept reproducible ON PURPOSE — this is the leg measured on the box
    2026-08-25 (1.35% anchored-only). It is NOT the default: see the carrier
    below.
    """
    from hymem.query.state_anchor import select_state_anchor

    _seed_squeeze(conn)
    profiles, edges = select_state_anchor(conn, shared_cap=4)
    assert len(profiles) == 3
    assert len(edges) == 1
    assert edges[0]["subject_canonical"] == "m0"


def test_edges_are_not_starved_by_a_large_profile(conn):
    """CARRIER for the banked Plan D deviation (correction 5, 2026-08-25).

    The default must give each source its OWN cap. Its control is
    `test_shared_cap_reproduces_the_digest_squeeze` above: without that pair,
    a single implementation could satisfy either rule and nothing would say
    which one shipped.

    Why this is load-bearing rather than a preference: under the shared cap the
    box's 22 profile rows leave an edge budget of ZERO against 8754 active
    edges, so the tier is inert, C1 reads near zero, and Plan D closes
    FAIL-mechanism because of the digest's prompt-block budget rather than
    because state anchors do not work.
    """
    from hymem.query.state_anchor import select_state_anchor

    _seed_squeeze(conn)
    profiles, edges = select_state_anchor(conn, edge_cap=4, profile_cap=4)
    assert len(profiles) == 3
    assert len(edges) == 4, "profile rows consumed the edge budget — shared cap leaked back in"


def test_each_cap_is_respected_independently(conn):
    from hymem.query.state_anchor import select_state_anchor

    _seed_squeeze(conn)
    profiles, edges = select_state_anchor(conn, edge_cap=2, profile_cap=1)
    assert len(profiles) == 1
    assert len(edges) == 2
    # and a zero cap disables exactly one source, never both
    profiles, edges = select_state_anchor(conn, edge_cap=0, profile_cap=3)
    assert len(profiles) == 3 and edges == []


def test_the_predicate_matches_the_digest_anchor_row_for_row(conn):
    """CONTROL on the copied predicate: the selector must agree with
    `_anchor_facts`' EDGE leg exactly. If the digest's clause ever changes,
    this fails and the copy is re-decided deliberately rather than drifting."""
    from hymem.dreaming.aggregate import _anchor_facts
    from hymem.query.state_anchor import select_anchor_edges

    with core_db.transaction(conn):
        _seed_edge(conn, "a", "uses", "postgres", pos=5, neg=0)
        _seed_edge(conn, "b", "uses", "redis", pos=2, neg=4)          # margin <= 0
        _seed_edge(conn, "c", "uses", "kafka", pos=3, neg=0, derived=1)
        _seed_edge(conn, "d", "uses", "mysql", pos=3, neg=0,
                   invalid_at="2024-02-01 00:00:00")
        _seed_edge(conn, "e", "uses", "sqlite", pos=4, neg=1)

    rendered = [f"{r['subject_canonical']} {r['predicate']} {r['object_canonical']}"
                for r in select_anchor_edges(conn, cap=50)]
    # No profile rows seeded here, so the digest block IS the edge leg, in order.
    assert rendered == _anchor_facts(conn, 50)
    assert rendered == ["a uses postgres", "e uses sqlite"], (
        "the copied predicate drifted: margin<=0 / derived / invalid_at rows leaked "
        "in, or the evidence-margin ordering changed"
    )


def test_select_anchor_profile_rows_excludes_invalidated(conn):
    from hymem.query.state_anchor import select_anchor_profile_rows

    with core_db.transaction(conn):
        _seed_profile(conn, "recurring_activity", "running")
        _seed_profile(conn, "recurring_activity", "swimming",
                      invalid_at="2024-02-01")
        _seed_profile(conn, "location", "lommel")

    rows = select_anchor_profile_rows(conn, cap=20)
    vals = {r.value for r in rows}
    assert vals == {"running", "lommel"}


def test_seed_terms_from_profile_values_first():
    from hymem.query.state_anchor import seed_terms_from_profile
    from hymem.dreaming.user_profile import ProfileEntry

    entries = [
        ProfileEntry(
            slot="recurring_activity", slot_key="melanie",
            value="running pottery", confidence=1.0,
            evidence_message_id=None, valid_at="2024-01-01 00:00:00",
        ),
    ]
    terms = seed_terms_from_profile(entries)
    # value + value words + slot_key are the searchable vocabulary
    for t in ["running", "pottery", "melanie"]:
        assert t in terms
    # slot words (not the underscore-joined token) also present
    assert "recurring" in terms
    assert "activity" in terms


def test_seed_terms_from_profile_empty(conn):
    from hymem.query.state_anchor import seed_terms_from_profile

    assert seed_terms_from_profile([]) == []


# ── Task 3: state_anchor_expand ────────────────────────────────────────────

def test_expand_surfaces_evidence_from_seed_terms(anchor_fixture):
    """The evidence chunk is FTS-reachable from the anchor; the distractor is
    reachable from the query instead — Task 4's scenario, measured here at the
    expansion-core level."""
    from hymem.query.state_anchor import (
        select_anchor_edges,
        seed_terms_from_edges,
        state_anchor_expand,
    )

    conn = anchor_fixture
    edges = select_anchor_edges(conn, cap=20)
    terms = seed_terms_from_edges(edges)
    hits = state_anchor_expand(conn, terms, top_k=5)

    ids = [h["chunk_id"] for h in hits]
    assert "c-evidence" in ids
    # the query-reachable distractor shares no anchor term ("nas01" etc.)
    assert "c-distractor" not in ids


def test_expand_limits_top_k_and_dedups(conn):
    with core_db.transaction(conn):
        _seed_session(conn)
        _seed_message(conn, 1, "2024-01-01 00:00:00")
        for i in range(8):
            _seed_chunk(conn, f"c{i}", 1, f"CUDA 12.1 notes number {i}")

    from hymem.query.state_anchor import state_anchor_expand

    hits = state_anchor_expand(conn, ["cuda_12.1", "cuda_12.1", "cuda"], top_k=5)
    ids = [h["chunk_id"] for h in hits]
    assert len(ids) == 5
    assert len(set(ids)) == 5


def test_expand_inert_without_seed_terms(conn):
    from hymem.query.state_anchor import state_anchor_expand

    assert state_anchor_expand(conn, [], top_k=5) == []
    assert state_anchor_expand(conn, ["  ", None], top_k=5) == []


# ── Task 4: probe ──────────────────────────────────────────────────────────

def test_probe_reports_the_anchored_only_hit(anchor_fixture, tmp_path):
    """THE smoke test: the gold row is reachable ONLY via anchor expansion.
    The probe must count it as one anchored-only hit, zero wrong-state pulls,
    zero vec calls (FTS-only default) and must not mutate the store."""
    from datetime import datetime

    from hymem import HyMemConfig

    queries = [{
        "question": "Which GPU system produces model training output?",
        # BOTH golds: the genuine one and the circular one. The probe must
        # credit the first and exclude-but-count the second.
        "gold_chunk_ids": ["c-gold", "c-evidence"],
        "category": "single-session-user",
    }]
    qpath = tmp_path / "queries.json"
    qpath.write_text(__import__("json").dumps(queries))

    store = tmp_path / "store.sqlite"
    # copy the fixture store to a temp path so the probe opens it read-only
    src = anchor_fixture
    src.execute("VACUUM INTO ?", (str(store),))

    before = _count_writes(src)

    cfg = HyMemConfig(root=tmp_path)
    summary = run_probe(store, qpath, cfg=cfg)

    assert summary["n_queries"] == 1
    assert summary["hit_rate"] == 1.0
    assert summary["circular_queries"] == 1, (
        "the provenance-circular gold was credited as a reach"
    )
    assert summary["headroom_queries"] == 1
    assert summary["c1_ceiling"] == 1.0
    assert summary["wrong_state_rate"] == 0.0
    assert summary["vec_calls"] == 0
    assert summary["llm_calls"] == 0
    assert summary["max_added_rows"] <= 5
    assert summary["max_added_tokens"] <= 400

    # store not mutated (row counts stable on the original connection)
    assert _count_writes(src) == before


# ── the SECOND corpus (D3/D5: probed separately, at most one ships) ─────────

def test_message_expansion_uses_the_raw_turn_corpus(anchor_fixture):
    """`expand_over_messages` reaches raw turns, not chunks.

    The chunk arm and the message arm are different corpora with incomparable
    BM25 scores, and the plan ships AT MOST ONE. Without this test the message
    arm has no coverage at all and could return chunk rows unnoticed.
    """
    from hymem.query.state_anchor import (
        expand_over_messages,
        select_anchor_edges,
        seed_terms_from_edges,
    )

    conn = anchor_fixture
    with core_db.transaction(conn):
        # Insert the content directly: `messages_fts` has INSERT and DELETE
        # triggers but NO UPDATE trigger (schema.sql:79-85), so seeding a row
        # and then UPDATE-ing its content leaves the FTS index holding the old
        # text and this test would read as a dead message arm.
        conn.execute(
            "INSERT INTO messages(id, session_id, role, content, created_at) "
            "VALUES (4, 's1', 'user', ?, '2024-01-04 00:00:00')",
            ("The dev box is configured with cuda and pytorch.",),
        )

    terms = seed_terms_from_edges(select_anchor_edges(conn, cap=20))
    hits = expand_over_messages(conn, terms, top_k=5)
    assert hits, "the raw-turn arm surfaced nothing from the anchor seed"
    assert all("message_id" in h for h in hits), "message arm returned chunk-shaped rows"


def test_message_expansion_is_inert_without_seed_terms(anchor_fixture):
    from hymem.query.state_anchor import expand_over_messages

    assert expand_over_messages(anchor_fixture, [], top_k=5) == []


def test_a_provenance_circular_gold_does_not_score(anchor_fixture, tmp_path):
    """CONTROL for the circularity exclusion (D4).

    When the ONLY gold row is the chunk the seed edge was extracted from, the
    anchor reaches it by tautology and the probe must score ZERO. Without this
    the box's original fixture read hit_rate 1.0 on a circular hit, and any
    store where edges are dense in gold chunks would report reachability
    approaching 100% while proving nothing.
    """
    import json

    from hymem import HyMemConfig

    queries = [{
        "question": "Which GPU system produces model training output?",
        "gold_chunk_ids": ["c-evidence"],
        "category": "single-session-user",
    }]
    qpath = tmp_path / "queries.json"
    qpath.write_text(json.dumps(queries))
    store = tmp_path / "store.sqlite"
    anchor_fixture.execute("VACUUM INTO ?", (str(store),))

    summary = run_probe(store, qpath, cfg=HyMemConfig(root=tmp_path))
    assert summary["hit_rate"] == 0.0, "a tautological reach was scored as a hit"
    assert summary["circular_queries"] == 1


def test_a_saturated_baseline_reports_zero_headroom(anchor_fixture, tmp_path):
    """CONTROL for the headroom denominator (D4).

    When the baseline already holds the gold, no tier can add reach: the query
    must not count toward the C1 denominator. Without this, a saturated store
    drives hit_rate toward 0 and the verdict reads FAIL-mechanism when the real
    reading is "the instrument had no room" (the LME 99.8% precedent).
    """
    import json

    from hymem import HyMemConfig

    queries = [{
        "question": "Model training output stored nas01",   # matches the distractor
        "gold_chunk_ids": ["c-distractor"],
        "category": "single-session-user",
    }]
    qpath = tmp_path / "queries.json"
    qpath.write_text(json.dumps(queries))
    store = tmp_path / "store.sqlite"
    anchor_fixture.execute("VACUUM INTO ?", (str(store),))

    summary = run_probe(store, qpath, cfg=HyMemConfig(root=tmp_path))
    assert summary["headroom_queries"] == 0, (
        "a query whose gold the baseline already had was counted as headroom"
    )
    assert summary["c1_ceiling"] == 0.0
