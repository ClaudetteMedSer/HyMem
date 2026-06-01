from __future__ import annotations

import pytest

from hymem.query.intent import detect_ability

# --- Unit: detect_ability classification -----------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "how many project cards did I add?",
        "How much did I spend on groceries?",
        "how often do I deploy to prod?",
        "what is the number of open tickets?",
        "give me the count of failed builds",
        "what's the total number of widgets I logged?",
    ],
)
def test_mr_phrases_detected(query: str) -> None:
    assert detect_ability(query) == "MR"


@pytest.mark.parametrize(
    "query",
    [
        "hoeveel kaarten heb ik toegevoegd?",
        "hoe vaak deploy ik naar productie?",
        "wat is het aantal openstaande tickets?",
    ],
)
def test_mr_dutch_phrases_detected(query: str) -> None:
    assert detect_ability(query) == "MR"


@pytest.mark.parametrize(
    "query",
    [
        "how long between the first and second deploy?",
        "how long after the migration did the bug appear?",
        "what happened first, the outage or the fix?",
        "which came first, the PR or the issue?",
        "in what order did I add these items?",
        "how long ago did I start using postgres?",
    ],
)
def test_tr_phrases_detected(query: str) -> None:
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        "hoe lang tussen de eerste en tweede deploy?",
        "wat gebeurde eerst, de storing of de fix?",
        "in welke volgorde heb ik deze toegevoegd?",
        "hoe lang geleden ben ik postgres gaan gebruiken?",
    ],
)
def test_tr_dutch_phrases_detected(query: str) -> None:
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        # The crucial overlap: a counting opener + temporal unit + anchor is TR.
        "how many days between the order and the delivery?",
        "how many weeks after launch did we hit the bug?",
        "how many months before the deadline did I finish?",
        "hoeveel dagen tussen de bestelling en de levering?",
        "hoeveel weken na de lancering kwam de fout?",
    ],
)
def test_mr_tr_overlap_resolves_to_tr(query: str) -> None:
    # "how many days between X and Y" is a duration question -> TR, not MR.
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        "what build tools do we use?",
        "tell me about the postgres migration",
        "how are you doing today?",
        "how about we deploy tomorrow",
        "remind me what the deploy steps were",
        "",
    ],
)
def test_non_matching_queries_return_none(query: str) -> None:
    assert detect_ability(query) is None


# --- Integration: auto-detection wired into augment() ----------------------


def test_counting_question_auto_triggers_aggregate_without_explicit_ability(hy_agg):
    # The whole point: a "how many" question must hit the MR aggregate path
    # (total_message_matches populated) WITHOUT the caller passing ability="MR".
    sid = "auto-mr"
    hy_agg.open_session(sid)
    for i in range(7):
        hy_agg.log_message(sid, "user", f"I added project card number {i} to my gallery")

    ctx = hy_agg.augment("how many project cards did I add to my gallery?")

    assert ctx.detected_ability == "MR"
    assert ctx.total_message_matches == 7


def test_explicit_ability_overrides_auto_detection(hy_agg):
    # A temporal-looking query would auto-detect TR, but an explicit ability="MR"
    # must win and run the MR aggregate path; detected_ability stays None so the
    # host-supplied hint is distinguishable from an inference.
    sid = "override"
    hy_agg.open_session(sid)
    for i in range(4):
        hy_agg.log_message(sid, "user", f"deploy {i} happened first before the others")

    ctx = hy_agg.augment(
        "what happened first across my deploys?", ability="MR"
    )

    assert ctx.detected_ability is None  # host hint -> no inference recorded
    assert ctx.total_message_matches == 4  # MR aggregate ran, not TR


def test_non_matching_query_stays_on_default_path(hy):
    # No counting/temporal phrasing -> no inference, default path, total stays 0.
    sid = "auto-none"
    hy.open_session(sid)
    for i in range(9):
        hy.log_message(sid, "user", f"card {i} added to gallery")

    ctx = hy.augment("gallery cards")

    assert ctx.detected_ability is None
    assert ctx.total_message_matches == 0


def test_dutch_counting_question_auto_triggers_aggregate(hy_agg):
    sid = "auto-nl"
    hy_agg.open_session(sid)
    for i in range(5):
        hy_agg.log_message(sid, "user", f"ik heb kaart {i} aan de galerij toegevoegd")

    ctx = hy_agg.augment("hoeveel kaarten heb ik toegevoegd aan de galerij?")

    assert ctx.detected_ability == "MR"
    assert ctx.total_message_matches == 5


def test_temporal_question_auto_detected_records_tr(hy):
    # A temporal question with no host hint records detected_ability="TR" (the
    # TR shaping/timeline runs; here we assert the routing decision is observable).
    sid = "auto-tr"
    hy.open_session(sid)
    hy.log_message(sid, "user", "I deployed on 2026-01-10 and again on 2026-02-15")

    ctx = hy.augment("how long between my deploys?")

    assert ctx.detected_ability == "TR"
    # TR path runs instead of MR aggregation, so the counting total stays 0.
    assert ctx.total_message_matches == 0
