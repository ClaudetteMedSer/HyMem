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

This test pins the FLAG-OFF baseline. It was written when supersession did not
exist ("the assertions marked `# GAP:` will start failing when the fix lands" —
which happened: the consumer landed and its default flipped ON 2026-07-02 after
the LME guard cleared). The repro now runs with
`value_supersession_enabled=False` and keeps two things pinned: the raw gap the
consumer closes, and the extraction-side routing fact (a plain update emits no
negative evidence), which is flag-independent.

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
from hymem.dreaming.value_supersession import _classify_object
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


@pytest.fixture
def hy_supersession_off(cfg):
    """Flag-off HyMem for the Step-0 repro: the default flipped ON 2026-07-02,
    so the raw gap (both values active) only reproduces with the consumer
    disabled."""
    instance = HyMem(
        replace(cfg, value_supersession_enabled=False),
        llm=StubLLMClient(default="[]"),
    )
    yield instance
    instance.close()


def test_ku_update_does_not_supersede_old_edge_REPRO(hy_supersession_off):
    hy = hy_supersession_off
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


# ── Step 1 (v2): parse-based discriminator ───────────────────────────────────
# The real extractor leaves `value_numeric` NULL but captures the value in the
# object string (a box run measured 1 of 207 evidence rows tagged). v2 classifies
# the object_canonical itself, so a DATE update and an UNTAGGED numeric update
# both supersede — the classes that dominate the LME knowledge-update floor.


def test_classify_object():
    """The discriminator: numbers and dates are typed; free text is not."""
    assert _classify_object("165") == ("num", None)
    assert _classify_object("65_percent") == ("num", "percent")
    assert _classify_object("78%") == ("num", "%")
    assert _classify_object("$120") == ("num", "$")
    assert _classify_object("-3.5_kg") == ("num", "kg")
    assert _classify_object("april_5_2024") == ("date", None)
    assert _classify_object("2024-04-05") == ("date", None)
    # Free-text objects must never be treated as a single-valued quantity, so a
    # multi-valued relation (preferences, tools) is never collapsed.
    assert _classify_object("adidas_black_sneakers") is None
    assert _classify_object("postgres") is None
    assert _classify_object("") is None
    assert _classify_object(None) is None


def _date_update_llm() -> StubLLMClient:
    """A deadline date update — POSITIVE evidence only, NO value_numeric. This is
    the case v1 could not reach (the date lives only in the object string)."""
    first = _chunk_extraction_response(
        [{"subject": SUBJECT, "predicate": PREDICATE, "object": OLD_OBJECT, "polarity": 1}]
    )
    second = _chunk_extraction_response(
        [{"subject": SUBJECT, "predicate": PREDICATE, "object": NEW_OBJECT, "polarity": 1}]
    )
    return StubLLMClient(fixtures={"April 1": first, "April 5": second}, default="[]")


@pytest.mark.parametrize("flag_on", [False, True])
def test_value_supersession_parses_date_objects(cfg, flag_on):
    """A date update (april_1 -> april_5) carries no value_numeric, yet v2
    supersedes the older date when the flag is on and leaves both active off."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        sid = "s_deadline_parse"
        hy.open_session(sid)
        hy.log_message(
            sid, "user",
            "My project deadline is April 1, 2024. Please remember that date.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user",
            "Update: the project deadline changed to April 5, 2024.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_date_update_llm())
        hy.dream()

        conn = hy.conn
        active = {
            r["object_canonical"]
            for r in conn.execute(
                "SELECT object_canonical FROM knowledge_graph "
                "WHERE subject_canonical = ? AND predicate = ? AND status = 'active'",
                (SUBJECT, PREDICATE),
            ).fetchall()
        }
        if not flag_on:
            assert active == {OLD_OBJECT, NEW_OBJECT}
        else:
            assert active == {NEW_OBJECT}, f"date supersession failed: {active}"
    finally:
        hy.close()


def _untagged_numeric_llm() -> StubLLMClient:
    """A count update (5 -> 7 team members) with NO value_numeric metadata — the
    number lives only in the canonical object string, as the real extractor emits."""
    first = _chunk_extraction_response(
        [{"subject": "team_size", "predicate": "configured_with", "object": "5", "polarity": 1}]
    )
    second = _chunk_extraction_response(
        [{"subject": "team_size", "predicate": "configured_with", "object": "7", "polarity": 1}]
    )
    return StubLLMClient(fixtures={"five engineers": first, "seven engineers": second}, default="[]")


@pytest.mark.parametrize("flag_on", [False, True])
def test_value_supersession_parses_untagged_numeric(cfg, flag_on):
    """A bare count update with no value_numeric still supersedes under v2."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        sid = "s_team"
        hy.open_session(sid)
        hy.log_message(
            sid, "user", "We have five engineers on the team right now.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user", "Update: we now have seven engineers after two hires.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_untagged_numeric_llm())
        hy.dream()

        conn = hy.conn
        active = {
            r["object_canonical"]
            for r in conn.execute(
                "SELECT object_canonical FROM knowledge_graph "
                "WHERE subject_canonical = 'team_size' AND predicate = 'configured_with' "
                "AND status = 'active'",
            ).fetchall()
        }
        if not flag_on:
            assert active == {"5", "7"}
        else:
            assert active == {"7"}, f"untagged-numeric supersession failed: {active}"
    finally:
        hy.close()


# ── Step 1 (v2): cross-SESSION mechanism validation ──────────────────────────
# The box found LME-S embeds value updates WITHIN one session (both values share a
# single valid_at), so the `valid_at` tie-breaker correctly skips them and the
# lever is a no-op on that dataset. This test supplies the shape LME-S lacks and
# the lever actually targets: the SAME attribute asserted in two SEPARATE sessions
# with DISTINCT world dates. It proves the full path — per-session valid_at
# stamping -> grouping -> the older session's value superseded by the newer — end
# to end, so we know the mechanism is sound for a dataset (e.g. BEAM) that does
# exercise cross-session drift, independently of the null LME-S signal.
XS_SUBJECT = "headcount"
XS_PRED = "configured_with"
XS_OLD = "20"
XS_NEW = "35"


def _xsession_llm() -> StubLLMClient:
    first = _chunk_extraction_response(
        [{"subject": XS_SUBJECT, "predicate": XS_PRED, "object": XS_OLD, "polarity": 1}]
    )
    second = _chunk_extraction_response(
        [{"subject": XS_SUBJECT, "predicate": XS_PRED, "object": XS_NEW, "polarity": 1}]
    )
    return StubLLMClient(
        fixtures={"twenty people": first, "thirty-five people": second}, default="[]"
    )


@pytest.mark.parametrize("flag_on", [False, True])
def test_value_supersession_across_sessions(cfg, flag_on):
    """Two SEPARATE sessions, distinct world dates: the January value (20) is
    superseded by the March value (35) when the flag is on. This is the
    cross-session drift the lever is for and that LME-S does not contain."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        # Session 1 — January.
        hy.open_session("s_jan")
        hy.log_message(
            "s_jan", "user", "Right now we employ twenty people across the company.",
            created_at="2024-01-10 09:00:00",
        )
        hy.close_session("s_jan")
        # Session 2 — March, a later world date in its own session.
        hy.open_session("s_mar")
        hy.log_message(
            "s_mar", "user", "After the spring hiring round we employ thirty-five people now.",
            created_at="2024-03-15 09:00:00",
        )
        hy.close_session("s_mar")

        hy.set_llm(_xsession_llm())
        hy.dream()

        conn = hy.conn
        rows = {
            r["object_canonical"]: r
            for r in conn.execute(
                "SELECT object_canonical, status, valid_at, invalid_at "
                "FROM knowledge_graph WHERE subject_canonical = ? AND predicate = ?",
                (XS_SUBJECT, XS_PRED),
            ).fetchall()
        }
        assert {XS_OLD, XS_NEW} <= set(rows), f"both edges should mint: {set(rows)}"
        # Sanity: the two sessions produced DISTINCT valid_at (the LME-S gap).
        assert rows[XS_OLD]["valid_at"] != rows[XS_NEW]["valid_at"], (
            "cross-session edges must carry different world dates for this test"
        )
        assert rows[XS_OLD]["valid_at"] < rows[XS_NEW]["valid_at"]

        if not flag_on:
            assert rows[XS_OLD]["status"] == "active" and rows[XS_NEW]["status"] == "active"
        else:
            assert rows[XS_NEW]["status"] == "active"
            assert rows[XS_OLD]["status"] == "retracted", "older session value not superseded"
            # Interval closed at the newer session's world date.
            assert rows[XS_OLD]["invalid_at"] == rows[XS_NEW]["valid_at"]
            ctx = hy.augment("How many people does the company employ?")
            objs = {f.object for f in ctx.graph_facts if f.subject == XS_SUBJECT}
            assert XS_OLD not in objs, f"stale cross-session value still retrievable: {sorted(objs)}"
    finally:
        hy.close()


# ── Step 1 (v3): VERSION-typed values ─────────────────────────────────────────
# A version update (requires_version 2.3.1 -> 2.4.0, uses python_3.12 ->
# python_3.13) is the remaining single-valued class v2 left as free text. The
# alpha prefix is the compatibility key, so python_* can never compete with
# node_* — and undotted single-number names (sprint_3, node_20, endpoint_v2)
# stay free text so distinct coexisting entities are never collapsed. NOTE:
# canonicalization flattens dots to underscores, so "2.3.1" MINTS as "2_3_1";
# the classifier accepts both shapes and the integration tests below assert
# against the flattened canonical form.


def test_classify_object_versions():
    """v3 discriminator: a dotted numeric core is a VERSION keyed on its alpha
    prefix; undotted single-number names stay free text; nothing previously
    typed (numbers, dates) changes class."""
    # Bare dotted core with >=3 components.
    assert _classify_object("2.3.1") == ("ver", None)
    assert _classify_object("2.4.0") == ("ver", None)
    # Alpha-prefixed dotted core with >=2 components — prefix = compatibility key.
    assert _classify_object("python_3.12") == ("ver", "python")
    assert _classify_object("python_3.13") == ("ver", "python")
    assert _classify_object("api_v2.3") == ("ver", "api")
    # A leading `v` on the core is allowed and stripped.
    assert _classify_object("v2.3") == ("ver", None)
    # The underscore-flattened shapes that canonicalization actually mints
    # classify identically.
    assert _classify_object("2_3_1") == ("ver", None)
    assert _classify_object("python_3_12") == ("ver", "python")
    assert _classify_object("api_v2_3") == ("ver", "api")
    assert _classify_object("v2_3") == ("ver", None)
    # NOT versions: single-number suffixed names with no dotted core are
    # typically distinct coexisting entities — collapsing them would destroy
    # multi-valued facts.
    assert _classify_object("sprint_3") is None
    assert _classify_object("sprint_4") is None
    assert _classify_object("endpoint_v2") is None
    assert _classify_object("node_20") is None
    assert _classify_object("v2") is None
    # Unchanged classifications: the version check runs AFTER date/number, so
    # a bare two-part decimal stays a number and dates stay dates.
    assert _classify_object("3.12") == ("num", None)
    assert _classify_object("165") == ("num", None)
    assert _classify_object("65_percent") == ("num", "percent")
    assert _classify_object("$120") == ("num", "$")
    assert _classify_object("-3.5_kg") == ("num", "kg")
    assert _classify_object("april_5_2024") == ("date", None)
    assert _classify_object("2024-04-05") == ("date", None)


VER_SUBJECT = "billing_service"
VER_PRED = "requires_version"
VER_OLD = "2.3.1"
VER_NEW = "2.4.0"
VER_OLD_CANON = "2_3_1"  # what normalize() mints for "2.3.1"
VER_NEW_CANON = "2_4_0"


def _version_update_llm() -> StubLLMClient:
    """A bare version bump — POSITIVE evidence only, no value_numeric; the
    version lives only in the object string, as the real extractor emits."""
    first = _chunk_extraction_response(
        [{"subject": VER_SUBJECT, "predicate": VER_PRED, "object": VER_OLD, "polarity": 1}]
    )
    second = _chunk_extraction_response(
        [{"subject": VER_SUBJECT, "predicate": VER_PRED, "object": VER_NEW, "polarity": 1}]
    )
    return StubLLMClient(fixtures={"2.3.1": first, "2.4.0": second}, default="[]")


@pytest.mark.parametrize("flag_on", [False, True])
def test_value_supersession_version_update(cfg, flag_on):
    """A requires_version bump (2.3.1 -> 2.4.0): with the flag on the older
    version is retracted with its interval closed at the newer valid_at."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        sid = "s_ver"
        hy.open_session(sid)
        hy.log_message(
            sid, "user",
            "The billing service requires framework version 2.3.1 right now.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user",
            "Update: the billing service now requires framework version 2.4.0.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_version_update_llm())
        hy.dream()

        conn = hy.conn
        rows = {
            r["object_canonical"]: r
            for r in conn.execute(
                "SELECT object_canonical, status, valid_at, invalid_at "
                "FROM knowledge_graph WHERE subject_canonical = ? AND predicate = ?",
                (VER_SUBJECT, VER_PRED),
            ).fetchall()
        }
        assert {VER_OLD_CANON, VER_NEW_CANON} <= set(rows), (
            f"both version edges should mint (flattened canonicals), got {set(rows)}"
        )
        if not flag_on:
            assert rows[VER_OLD_CANON]["status"] == "active"
            assert rows[VER_NEW_CANON]["status"] == "active"
            return
        assert rows[VER_NEW_CANON]["status"] == "active"
        assert rows[VER_OLD_CANON]["status"] == "retracted", "older version not superseded"
        # Interval closed at the world date the new version took over.
        assert rows[VER_OLD_CANON]["invalid_at"] == rows[VER_NEW_CANON]["valid_at"]
    finally:
        hy.close()


def _python_bump_llm() -> StubLLMClient:
    first = _chunk_extraction_response(
        [{"subject": "backend", "predicate": "uses", "object": "python_3.12", "polarity": 1}]
    )
    second = _chunk_extraction_response(
        [{"subject": "backend", "predicate": "uses", "object": "python_3.13", "polarity": 1}]
    )
    return StubLLMClient(fixtures={"Python 3.12": first, "Python 3.13": second}, default="[]")


@pytest.mark.parametrize("flag_on", [False, True])
def test_value_supersession_prefixed_version_update(cfg, flag_on):
    """`uses python_3.12` -> `python_3.13`: same alpha prefix, so the versions
    compete and the older one is superseded when the flag is on."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        sid = "s_py"
        hy.open_session(sid)
        hy.log_message(
            sid, "user",
            "Our backend uses Python 3.12 across every service today.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user",
            "Update: the backend now uses Python 3.13 after the migration.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_python_bump_llm())
        hy.dream()

        conn = hy.conn
        rows = {
            r["object_canonical"]: r
            for r in conn.execute(
                "SELECT object_canonical, status, valid_at, invalid_at "
                "FROM knowledge_graph "
                "WHERE subject_canonical = 'backend' AND predicate = 'uses'",
            ).fetchall()
        }
        assert {"python_3_12", "python_3_13"} <= set(rows), f"both edges should mint: {set(rows)}"
        if not flag_on:
            assert rows["python_3_12"]["status"] == "active"
            assert rows["python_3_13"]["status"] == "active"
            return
        assert rows["python_3_13"]["status"] == "active"
        assert rows["python_3_12"]["status"] == "retracted", "older python version not superseded"
        assert rows["python_3_12"]["invalid_at"] == rows["python_3_13"]["valid_at"]
    finally:
        hy.close()


def _python_vs_node_llm() -> StubLLMClient:
    """Two DIFFERENT technologies on one subject+predicate. Both evidences carry
    value_numeric (3.12 / 20) to pin the class-first grouping: the object-string
    parse is authoritative, so python_3.12 keeps its ("ver", "python") key and
    node_20 (free text) never enters any pool — under the old has_numeric
    fast-path both routed as bare numbers and node_20 wrongly superseded
    python_3.12."""
    first = _chunk_extraction_response(
        [{"subject": "backend", "predicate": "uses", "object": "python_3.12",
          "polarity": 1, "value_numeric": 3.12}]
    )
    second = _chunk_extraction_response(
        [{"subject": "backend", "predicate": "uses", "object": "node_20",
          "polarity": 1, "value_numeric": 20}]
    )
    return StubLLMClient(fixtures={"Python 3.12": first, "Node 20": second}, default="[]")


@pytest.mark.parametrize("flag_on", [False, True])
def test_no_supersession_across_version_prefixes(cfg, flag_on):
    """python_3.12 vs node_20 on the same subject+predicate: different
    compatibility keys (and node_20 is not a version at all), so BOTH stay
    active whatever the flag — a multi-valued tech stack is never collapsed."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        sid = "s_stack"
        hy.open_session(sid)
        hy.log_message(
            sid, "user",
            "Our backend uses Python 3.12 for the API layer services.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user",
            "The backend also uses Node 20 for the build tooling.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_python_vs_node_llm())
        hy.dream()

        conn = hy.conn
        rows = {
            r["object_canonical"]: r
            for r in conn.execute(
                "SELECT object_canonical, status, invalid_at FROM knowledge_graph "
                "WHERE subject_canonical = 'backend' AND predicate = 'uses'",
            ).fetchall()
        }
        assert {"python_3_12", "node_20"} <= set(rows), f"both edges should mint: {set(rows)}"
        assert rows["python_3_12"]["status"] == "active", (
            "python_3_12 must never be superseded by node_20"
        )
        assert rows["node_20"]["status"] == "active"
        assert rows["python_3_12"]["invalid_at"] is None
        assert rows["node_20"]["invalid_at"] is None
    finally:
        hy.close()


def _sprints_llm() -> StubLLMClient:
    first = _chunk_extraction_response(
        [{"subject": "project_atlas", "predicate": "contains", "object": "sprint_3", "polarity": 1}]
    )
    second = _chunk_extraction_response(
        [{"subject": "project_atlas", "predicate": "contains", "object": "sprint_4", "polarity": 1}]
    )
    return StubLLMClient(fixtures={"sprint 3": first, "sprint 4": second}, default="[]")


@pytest.mark.parametrize("flag_on", [False, True])
def test_no_supersession_undotted_numbered_names(cfg, flag_on):
    """sprint_3 vs sprint_4: single-number suffixed names have no dotted core,
    classify as free text, and NEVER compete — both stay active whatever the
    flag. Distinct coexisting entities must not be collapsed."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        sid = "s_sprints"
        hy.open_session(sid)
        hy.log_message(
            sid, "user",
            "Project Atlas contains sprint 3 with the payments milestone.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user",
            "Project Atlas also contains sprint 4 with the reporting milestone.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_sprints_llm())
        hy.dream()

        conn = hy.conn
        rows = {
            r["object_canonical"]: r
            for r in conn.execute(
                "SELECT object_canonical, status FROM knowledge_graph "
                "WHERE subject_canonical = 'project_atlas' AND predicate = 'contains'",
            ).fetchall()
        }
        assert {"sprint_3", "sprint_4"} <= set(rows), f"both edges should mint: {set(rows)}"
        assert rows["sprint_3"]["status"] == "active", "sprint_3 must not be superseded"
        assert rows["sprint_4"]["status"] == "active"
    finally:
        hy.close()


# ── v3.1: string parse is authoritative over has_numeric ─────────────────────

def _possessions_llm() -> StubLLMClient:
    """Two possessions whose evidence carries value_numeric (years the extractor
    lifted from the phrasing) on FREE-TEXT objects. Pins the v3.1 bug: the old
    has_numeric fast-path routed both into the ("num", None) pool although
    _classify_object correctly returned None, and the older possession was
    retracted as a "superseded value" — a collapsed multi-valued fact."""
    first = _chunk_extraction_response(
        [{"subject": "atta", "predicate": "owns",
          "object": "vintage_omega_seamaster_watch",
          "polarity": 1, "value_numeric": 1960}]
    )
    second = _chunk_extraction_response(
        [{"subject": "atta", "predicate": "owns", "object": "leica_m6_camera",
          "polarity": 1, "value_numeric": 1984}]
    )
    return StubLLMClient(
        fixtures={"Omega Seamaster": first, "Leica M6": second}, default="[]"
    )


@pytest.mark.parametrize("flag_on", [False, True])
def test_no_supersession_free_text_with_value_numeric(cfg, flag_on):
    """value_numeric tagged on a free-text object must NOT route it into the
    numeric pool: the string parse (None) is authoritative, so two possessions
    both stay active whatever the flag."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        sid = "s_owns"
        hy.open_session(sid)
        hy.log_message(
            sid, "user",
            "I still own my vintage Omega Seamaster watch from the 1960s.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user",
            "I also own a Leica M6 camera built in 1984.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_possessions_llm())
        hy.dream()

        conn = hy.conn
        rows = {
            r["object_canonical"]: r
            for r in conn.execute(
                "SELECT object_canonical, status, invalid_at FROM knowledge_graph "
                "WHERE subject_canonical = 'atta' AND predicate = 'owns'",
            ).fetchall()
        }
        assert {"vintage_omega_seamaster_watch", "leica_m6_camera"} <= set(rows), (
            f"both edges should mint: {set(rows)}"
        )
        assert rows["vintage_omega_seamaster_watch"]["status"] == "active", (
            "free-text possession must never be superseded on tagged value_numeric"
        )
        assert rows["leica_m6_camera"]["status"] == "active"
        assert rows["vintage_omega_seamaster_watch"]["invalid_at"] is None
        assert rows["leica_m6_camera"]["invalid_at"] is None
    finally:
        hy.close()


def _bare_number_unit_refinement_llm() -> StubLLMClient:
    """A bare-number object ("165", parses to unit None) whose evidence carries
    value_unit, updated by a unit-suffixed object ("90_minutes"). Pins the
    fill-only refinement: ev_unit fills the missing unit so the bare number
    joins the ("num", "minutes") pool and the update supersedes it."""
    first = _chunk_extraction_response(
        [{"subject": "api_error_budget", "predicate": "configured_with",
          "object": "165", "polarity": 1, "value_numeric": 165,
          "value_unit": "minutes"}]
    )
    second = _chunk_extraction_response(
        [{"subject": "api_error_budget", "predicate": "configured_with",
          "object": "90_minutes", "polarity": 1, "value_numeric": 90,
          "value_unit": "minutes"}]
    )
    return StubLLMClient(
        fixtures={"165 minutes": first, "90 minutes": second}, default="[]"
    )


@pytest.mark.parametrize("flag_on", [False, True])
def test_bare_number_unit_refined_from_evidence_supersedes(cfg, flag_on):
    """has_numeric survives as a fill-only unit refinement: "165" + evidence
    unit "minutes" competes with "90_minutes", and the older value is
    superseded when the flag is on."""
    hy = HyMem(replace(cfg, value_supersession_enabled=flag_on), llm=StubLLMClient(default="[]"))
    try:
        sid = "s_budget"
        hy.open_session(sid)
        hy.log_message(
            sid, "user",
            "The API error budget is 165 minutes per quarter.",
            created_at="2024-03-01 09:00:00",
        )
        hy.log_message(
            sid, "user",
            "We tightened the API error budget to 90 minutes per quarter.",
            created_at="2024-03-20 09:00:00",
        )
        hy.close_session(sid)

        hy.set_llm(_bare_number_unit_refinement_llm())
        hy.dream()

        conn = hy.conn
        rows = {
            r["object_canonical"]: r
            for r in conn.execute(
                "SELECT object_canonical, status, valid_at, invalid_at "
                "FROM knowledge_graph "
                "WHERE subject_canonical = 'api_error_budget' "
                "AND predicate = 'configured_with'",
            ).fetchall()
        }
        assert {"165", "90_minutes"} <= set(rows), f"both edges should mint: {set(rows)}"
        if not flag_on:
            assert rows["165"]["status"] == "active"
            assert rows["90_minutes"]["status"] == "active"
            return
        assert rows["90_minutes"]["status"] == "active"
        assert rows["165"]["status"] == "retracted", (
            "bare number with evidence-refined unit should be superseded"
        )
        assert rows["165"]["invalid_at"] == rows["90_minutes"]["valid_at"]
    finally:
        hy.close()
