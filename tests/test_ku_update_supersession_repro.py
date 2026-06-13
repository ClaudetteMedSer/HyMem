"""Characterization / repro test for the knowledge-graph bi-temporal
supersession gap (schema v15+).

The hypothesized gap: a single authoritative knowledge UPDATE
("the deadline changed from April 1 to April 5") does NOT supersede the old
value. Unlike `user_profile` (which closes the old row and inserts the new one
immediately), the knowledge graph only *retracts* an edge through evidence
accumulation in phase3 (confidence decay below `retract_threshold`, or
`neg_evidence >= 2*pos_evidence + zombie_neg_threshold`). A plain restatement of
the new value emits POSITIVE evidence for the new object and NOTHING against the
old one, so the April-1 edge stays `status='active'` with `invalid_at IS NULL`
and both values coexist in retrieval.

This test pins the CURRENT behavior. The assertions marked `# GAP:` encode the
behavior we intend to flip once supersession is implemented; the test will then
start failing on exactly those lines, which is the signal the fix landed.

It uses the supported deterministic-extraction path: a `StubLLMClient` keyed on
a substring unique to each user turn's chunk text (see `tests/conftest.py` and
`tests/test_dreaming.py` for the canned-triple convention). No production code is
touched and no real LLM is involved.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hymem import HyMem
from hymem.extraction.llm import StubLLMClient

# Stable subject/predicate; only the object (the dated value) changes between the
# original assertion and the update. Canonicalization lowercases, so we use
# already-canonical-shaped tokens to keep the assertions readable.
# `configured_with` is an ALLOWED single-valued predicate (a deadline is
# "configured with" a date). `is` is not in ALLOWED_PREDICATES, so the validator
# would silently drop it.
SUBJECT = "project_deadline"
PREDICATE = "configured_with"
OLD_OBJECT = "april_1_2024"
NEW_OBJECT = "april_5_2024"


def _chunk_extraction_response(triples: list[dict]) -> str:
    """The combined phase-1 chunk-extraction call expects a JSON object with
    `triples` and `markers` keys (see hymem/extraction/chunk.py)."""
    return json.dumps({"triples": triples, "markers": []})


def _update_routed_llm() -> StubLLMClient:
    """Route extraction per-chunk on a substring unique to each user turn.

    Phase 1 issues ONE call per chunk whose user prompt embeds the chunk text
    (`CHUNK_EXTRACTION_USER_TEMPLATE.format(text=...)`), and `StubLLMClient`
    matches fixtures against system+user. The first turn's chunk text contains
    "April 1" and the second's contains "April 5", so each routes to its own
    canned triple. Both triples are POSITIVE (+1): the user is simply stating a
    value, never explicitly negating the old one — which is the crux of the gap.
    """
    first = _chunk_extraction_response(
        [{"subject": SUBJECT, "predicate": PREDICATE, "object": OLD_OBJECT, "polarity": 1}]
    )
    second = _chunk_extraction_response(
        [{"subject": SUBJECT, "predicate": PREDICATE, "object": NEW_OBJECT, "polarity": 1}]
    )
    return StubLLMClient(
        fixtures={
            "April 1": first,
            "April 5": second,
        },
        default="[]",
    )


def test_ku_update_does_not_supersede_old_edge_REPRO(hy):
    # 1. Ingest the original assertion and the later authoritative update as two
    #    user turns with distinct WORLD dates (created_at). Each is long/explicit
    #    enough to become its own high-salience chunk.
    sid = "s_deadline"
    hy.open_session(sid)
    hy.log_message(
        sid,
        "user",
        "My project deadline is April 1, 2024. Please remember that date for the launch.",
        created_at="2024-03-01 09:00:00",
    )
    hy.log_message(
        sid,
        "user",
        "Update: the project deadline changed to April 5, 2024. Use the new date from now on.",
        created_at="2024-03-20 09:00:00",
    )
    hy.close_session(sid)

    # 2. Drive the full dreaming pipeline so both edges are minted and the
    #    bi-temporal stamping (stamp_validity / any stamp_invalidation) runs.
    hy.set_llm(_update_routed_llm())
    report = hy.dream()
    assert report.chunks_processed >= 2, (
        "both turns must reach extraction for the repro to be meaningful"
    )

    conn = hy.conn

    # 3. Inspect the knowledge_graph table directly.
    rows = conn.execute(
        "SELECT object_canonical, status, pos_evidence, neg_evidence, "
        "       valid_at, invalid_at "
        "FROM knowledge_graph "
        "WHERE subject_canonical = ? AND predicate = ? "
        "ORDER BY object_canonical",
        (SUBJECT, PREDICATE),
    ).fetchall()
    by_obj = {r["object_canonical"]: r for r in rows}

    # Sanity: both the old and new value edges exist.
    assert OLD_OBJECT in by_obj, "old-value edge should have been minted"
    assert NEW_OBJECT in by_obj, "new-value edge should have been minted"

    old = by_obj[OLD_OBJECT]
    new = by_obj[NEW_OBJECT]

    active_count = conn.execute(
        "SELECT COUNT(*) AS c FROM knowledge_graph "
        "WHERE subject_canonical = ? AND predicate = ? AND status = 'active'",
        (SUBJECT, PREDICATE),
    ).fetchone()["c"]

    # GAP: the update did NOT supersede the old value. BOTH single-valued edges
    # remain active, so the stale April-1 fact still competes with April-5.
    # When supersession lands, exactly one of these should remain active and this
    # assertion flips to `== 1`.
    assert active_count == 2, (
        f"expected the gap (both values active) but found {active_count} active "
        "edges for the subject+predicate"
    )

    # GAP: the old-value edge is still active with an OPEN validity interval —
    # it was never closed by the update.
    assert old["status"] == "active"
    assert old["invalid_at"] is None, (
        "GAP: old-value edge has invalid_at IS NULL — never invalidated by the update"
    )

    # The new-value edge is (as expected) active with an open interval.
    assert new["status"] == "active"
    assert new["invalid_at"] is None

    # 4. Did extraction emit any NEGATIVE-polarity evidence against the old
    #    value? This is the routing question: extraction vs reconciliation.
    old_neg_evidence = old["neg_evidence"]
    old_evidence_polarities = [
        r["polarity"]
        for r in conn.execute(
            "SELECT ev.polarity AS polarity "
            "FROM kg_evidence ev "
            "JOIN knowledge_graph kg ON kg.id = ev.edge_id "
            "WHERE kg.subject_canonical = ? AND kg.predicate = ? "
            "      AND kg.object_canonical = ?",
            (SUBJECT, PREDICATE, OLD_OBJECT),
        ).fetchall()
    ]

    # GAP (routing): NO negative evidence was ever recorded against the old
    # value. neg_evidence stays 0 and kg_evidence holds only the original
    # positive row — so phase3's retraction rule can never fire. This localizes
    # the gap to EXTRACTION (no contradiction emitted), not RECONCILIATION.
    assert old_neg_evidence == 0, (
        "GAP: no negative evidence accrued against the old value"
    )
    assert old_evidence_polarities == [1], (
        "GAP: the old-value edge has only its single positive evidence row"
    )

    # 5. What does retrieval surface for the deadline question? With both edges
    #    active, the stale value can come back alongside (or instead of) the new
    #    one. We assert the OLD value is still retrievable — the user-visible
    #    symptom of the gap.
    ctx = hy.augment("When is the project deadline?")
    retrieved_objects = {f.object for f in ctx.graph_facts if f.subject == SUBJECT}

    # GAP: the stale April-1 value is still retrievable. When supersession lands
    # the old object should drop out of active retrieval and this assertion flips.
    assert OLD_OBJECT in retrieved_objects, (
        "GAP: stale old-value edge still surfaces in retrieval — "
        f"graph_facts objects for {SUBJECT}: {sorted(retrieved_objects)}"
    )


# ── Step 1: value-supersession fix (cfg.value_supersession_enabled) ──────────
# A NUMERIC value update (test coverage 65% -> 78%). Both objects carry a
# value_numeric + value_unit, so they are typed values the supersession step can
# recognise, and they share a unit so they compete. This is the LME-KU value
# class (counts / percentages), modelled with the exact kg_evidence metadata the
# fix keys on — unlike the date REPRO above, whose triples carry no value fields.
NUM_SUBJECT = "test_coverage"
NUM_PREDICATE = "configured_with"
NUM_OLD = "65_percent"
NUM_NEW = "78_percent"


def _numeric_update_llm() -> StubLLMClient:
    first = _chunk_extraction_response(
        [{"subject": NUM_SUBJECT, "predicate": NUM_PREDICATE, "object": NUM_OLD,
          "polarity": 1, "value_numeric": 65, "value_unit": "percent"}]
    )
    second = _chunk_extraction_response(
        [{"subject": NUM_SUBJECT, "predicate": NUM_PREDICATE, "object": NUM_NEW,
          "polarity": 1, "value_numeric": 78, "value_unit": "percent"}]
    )
    return StubLLMClient(
        fixtures={"65 percent": first, "78 percent": second}, default="[]"
    )


@pytest.mark.parametrize("flag_on", [False, True])
def test_value_supersession_flag_flips_behavior(cfg, flag_on):
    """OFF reproduces the gap (both values active); ON supersedes the older
    value — exactly one active edge, the old one retracted with a closed
    interval, and the stale value gone from retrieval."""
    hy = HyMem(
        replace(cfg, value_supersession_enabled=flag_on),
        llm=StubLLMClient(default="[]"),
    )
    try:
        sid = "s_cov"
        hy.open_session(sid)
        hy.log_message(
            sid, "user",
            "Our test coverage reached 65 percent this sprint.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user",
            "Update: test coverage is now 78 percent after the new tests.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_numeric_update_llm())
        hy.dream()

        conn = hy.conn
        minted = {
            r["object_canonical"]
            for r in conn.execute(
                "SELECT object_canonical FROM knowledge_graph "
                "WHERE subject_canonical = ? AND predicate = ?",
                (NUM_SUBJECT, NUM_PREDICATE),
            ).fetchall()
        }
        # Both value edges must be minted regardless of the flag, else the repro
        # (per-turn routing / separate chunks) is not exercising what we think.
        assert {NUM_OLD, NUM_NEW} <= minted, f"both edges should mint, got {minted}"

        active = {
            r["object_canonical"]
            for r in conn.execute(
                "SELECT object_canonical FROM knowledge_graph "
                "WHERE subject_canonical = ? AND predicate = ? AND status = 'active'",
                (NUM_SUBJECT, NUM_PREDICATE),
            ).fetchall()
        }

        if not flag_on:
            # Baseline gap unchanged: both values remain active.
            assert NUM_OLD in active and NUM_NEW in active
            return

        # FIX: only the new value stays active; the old edge is retracted with a
        # closed validity interval set to the new value's world date.
        assert NUM_NEW in active
        assert NUM_OLD not in active, "old value should have been superseded"
        old = conn.execute(
            "SELECT status, valid_at, invalid_at FROM knowledge_graph "
            "WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?",
            (NUM_SUBJECT, NUM_PREDICATE, NUM_OLD),
        ).fetchone()
        assert old["status"] == "retracted"
        assert old["invalid_at"] is not None
        # invalid_at closes at the newer value's valid_at (when it took over).
        new_valid = conn.execute(
            "SELECT valid_at FROM knowledge_graph "
            "WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?",
            (NUM_SUBJECT, NUM_PREDICATE, NUM_NEW),
        ).fetchone()["valid_at"]
        assert old["invalid_at"] == new_valid

        # The stale value no longer surfaces in retrieval.
        ctx = hy.augment("What is the test coverage?")
        objs = {f.object for f in ctx.graph_facts if f.subject == NUM_SUBJECT}
        assert NUM_OLD not in objs, f"stale value still retrievable: {sorted(objs)}"
    finally:
        hy.close()
