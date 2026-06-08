from __future__ import annotations

import unicodedata

import pytest

from hymem.query.intent import (
    _MAX_SCAN_CHARS,
    AbilitySignal,
    detect_ability,
    detect_ability_signal,
)

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
        # Recency / "last time" — none of the legacy span/ordering forms caught
        # these, yet they are core BEAM temporal-reasoning questions.
        "when did I last go to the dentist?",
        "when was the last time I saw my brother?",
        "when's the most recent time I changed my password?",
        "do you remember the last time I traveled abroad?",
        # First occurrence / start — distinct from "what happened first".
        "when did I first try meditation?",
        "when did I start using postgres?",
        "when did the project begin?",
        # Count + "ago" used to fall through to MR; now stays TR.
        "how many months ago did I start the new job?",
        "how many weeks ago did we launch?",
        # Duration-to-now: "how long have I been …" had no anchor before.
        "how long have I been learning the guitar?",
        "how long has it been since my last checkup?",
    ],
)
def test_tr_recency_and_duration_to_now_detected(query: str) -> None:
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        "wanneer was de laatste keer dat ik de tandarts bezocht?",
        "wanneer ben ik voor het eerst gaan mediteren?",
        "wanneer ben ik voor het laatst op reis geweest?",
        "wanneer begon ik met postgres?",
        "hoeveel maanden geleden ben ik begonnen?",
        "hoe lang ben ik al aan het leren?",
    ],
)
def test_tr_recency_dutch_detected(query: str) -> None:
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        # The adjacency-bug class: a noun sits between the WH-word and the verb,
        # so the legacy `which\s+(?:happened|was)` never matched these.
        "which event happened first?",
        "which event did I attend first?",
        "who graduated first, second, third?",
        "which seeds were started first?",
        "which device did I get first?",
        "which came first, the outage or the fix?",
        "which trip did I take earliest?",
        "which of my projects did I finish most recently?",
    ],
)
def test_tr_ordering_with_intervening_noun_detected(query: str) -> None:
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        "welke gebeurtenis gebeurde eerst?",
        "wie studeerde als eerste af?",
        "welk apparaat kreeg ik het eerst?",
    ],
)
def test_tr_ordering_dutch_with_intervening_noun_detected(query: str) -> None:
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        # Bare distance/deictic anchors with NO count opener -> temporal.
        "what did I buy a week ago?",
        "did I see her two weeks ago?",
        "what was I working on a month ago?",
        "what happened four weeks ago?",
        "what did I do last Saturday?",
        "where did I go last month?",
        "wat heb ik vorige week gedaan?",
        "wat deed ik afgelopen maandag?",
    ],
)
def test_tr_distance_anchors_detected(query: str) -> None:
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        # Duration-to-now via a count opener + unit + perfect auxiliary (no anchor
        # or "ago") — used to fall through to MR.
        "how many weeks have I been exercising?",
        "how many months have I had this subscription?",
        "how many years has she been on the team?",
    ],
)
def test_tr_count_duration_to_now_detected(query: str) -> None:
    assert detect_ability(query) == "TR"


@pytest.mark.parametrize(
    "query",
    [
        # A timeframe on a genuine COUNT question must stay MR — the count is the
        # subject, the timeframe is incidental.
        "how many times did I go to the gym last week?",
        "how many emails did I send a week ago?",
    ],
)
def test_count_with_timeframe_stays_mr(query: str) -> None:
    assert detect_ability(query) == "MR"


@pytest.mark.parametrize(
    "query",
    [
        "what build tools do we use?",
        "tell me about the postgres migration",
        "how are you doing today?",
        "how about we deploy tomorrow",
        "remind me what the deploy steps were",
        "",
        # Precision guards for the broadened TR patterns: a "first" that is an
        # ADJECTIVE on a thing, not a recency frame, must NOT become TR.
        "what's the first thing I should do?",
        "what was my last order total?",
        # "how long" as a degree/length question (no temporal continuation).
        "how long is the rope I bought?",
        "how long should the README be?",
    ],
)
def test_non_matching_queries_return_none(query: str) -> None:
    assert detect_ability(query) is None


@pytest.mark.parametrize(
    "query",
    [
        # A genuine item count whose noun happens to carry "first"/"last" as an
        # adjective stays MR — the broadened recency patterns must not steal it.
        "how many first-edition books do I own?",
        "how many times was the last build retried?",
    ],
)
def test_count_with_first_last_adjective_stays_mr(query: str) -> None:
    assert detect_ability(query) == "MR"


# --- Hardening: pathological / malformed production input ------------------


@pytest.mark.parametrize("bad", [None, 123, 3.14, b"how many cards?", [], {}, object()])
def test_non_string_input_abstains_without_raising(bad) -> None:
    # detect_ability sits on the hot path of every augment() call; a host bug that
    # passes a non-string must abstain, never raise.
    assert detect_ability(bad) is None
    sig = detect_ability_signal(bad)
    assert sig == AbilitySignal(None, "non_str")


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", " "])
def test_blank_input_abstains(blank) -> None:
    assert detect_ability(blank) is None
    assert detect_ability_signal(blank).rule == "empty"


def test_oversized_input_is_bounded_and_still_classifies_leading_intent() -> None:
    # The intent opener lives at the START; a megabyte of trailing text must not
    # change the verdict (signal at the front) nor blow up (bounded scan).
    q = "how many project cards did I add? " + ("lorem ipsum dolor " * 100_000)
    assert len(q) > _MAX_SCAN_CHARS
    assert detect_ability(q) == "MR"


def test_oversized_near_miss_does_not_hang() -> None:
    # An opener+unit with NO following anchor is the backtracking-prone shape for
    # the lazy [\s\S]*? TR bridges; on unbounded input it would rescan to EOS at
    # every start. The prefix cap must bound it — assert it returns promptly.
    import time

    q = "how many days " + ("x " * 500_000)  # unit present, no duration-end anchor
    start = time.monotonic()
    result = detect_ability(q)
    assert time.monotonic() - start < 1.0  # bounded, not O(n^2) over the full paste
    assert result == "MR"  # "how many" opener with no TR anchor stays a count


def test_nfc_normalization_makes_decomposed_diacritics_match() -> None:
    # Dutch with a DECOMPOSED diacritic (combining mark) must classify identically
    # to its composed form — _prepare NFC-normalises before matching.
    composed = "hoeveel café-bezoeken heb ik gelogd?"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed  # genuinely different code-point sequences
    assert detect_ability(decomposed) == detect_ability(composed) == "MR"


# --- Observability: detect_ability_signal names the firing rule -------------


@pytest.mark.parametrize(
    "query,ability,rule",
    [
        ("how many days between the order and delivery?", "TR", "tr_duration"),
        ("how long after the migration did the bug appear?", "TR", "tr_howlong"),
        ("which event happened first?", "TR", "tr_order"),
        ("when did I last go to the dentist?", "TR", "tr_recency"),
        ("how many project cards did I add?", "MR", "mr_count"),
        ("what did I buy a week ago?", "TR", "tr_distance"),
        ("what build tools do we use?", None, "none"),
    ],
)
def test_signal_reports_firing_rule(query, ability, rule) -> None:
    sig = detect_ability_signal(query)
    assert sig == AbilitySignal(ability, rule)
    # The wrapper must agree with the signal's ability on every branch.
    assert detect_ability(query) == sig.ability


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
    assert ctx.detected_rule == "mr_count"  # observability: WHY it routed MR
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
    assert ctx.detected_rule is None  # no inference ran, so no rule recorded
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
