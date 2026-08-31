"""Offline tests for the BEAM adapter's episode-tier answer-bearing pre-check.

This instrument exists because the LME granularity guard tied, and the keep-db
probe showed why: both arms saturate the episode cap, so the lever changes
episode CONTENT only -- and on LME the tier never carried the answer, giving the
lever no path to the score. BEAM's EO/SUM abilities are where that could differ.

The tests below are mostly about the instrument's ability to be WRONG. The same
readout that motivated it also contained a structural zero read as a result
("recall_tier: episode 0", where `episode` was never an emittable value), so
what is asserted here is:

  * it CAN fire (a measure that cannot return a hit measures nothing);
  * it distinguishes "tier absent" from "tier present and empty" -- None vs 0.0;
  * its positive control (the message tier) is computed the same way, so a dead
    measure is visible as both columns collapsing rather than as a finding;
  * the tier names it counts are the names the adapter actually emits, so a
    rename cannot silently zero a column.

Skipped where `requests` is absent (the adapter imports it at module scope).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
from beam_adapter import (  # noqa: E402
    ABILITY_MAP,
    GOLD_FIELDS,
    PROBE_CONTROL_SHARE,
    PROBE_MIN_ROWS,
    PROBE_NULL_MARGIN,
    _content_tokens,
    PROBE_GOLD_KINDS,
    _decoy_answer,
    _gold_tokens,
    _probe_gold,
    _resolve_gold,
    print_gold_audit,
    _probe_verdict,
    _tier_coverage,
    episode_probe,
    print_episode_probe,
)

QUESTION = "What order did the user complete the migration milestones in?"
IDEAL = ("The user first drafted the schema migration, then benchmarked the "
         "aggregation rebuild, and finally flipped the granularity default.")
DECOY = "The user reviewed timings and discussed configuration options at length."


def _mem(tier: str, content: str) -> dict:
    return {"type": tier, "content": content, "confidence": 0.8}


_ANSWER_BEARING = _mem(
    "episode",
    "Migration work: drafted schema, benchmarked aggregation rebuild, "
    "then flipped the granularity default.")
_NARRATIVE = _mem(
    "episode", "Session summary: the user and assistant had a long discussion.")


def test_the_measure_can_actually_fire():
    """Positive control for the instrument itself.

    Without this, every other assertion is satisfied by a measure that returns
    nothing on every input -- which is precisely the failure being guarded.
    """
    probe = episode_probe([_ANSWER_BEARING, _mem("message_hit", "unrelated")],
                          QUESTION, IDEAL)
    assert probe["cov_episodes"] is not None and probe["cov_episodes"] > 0.5
    assert probe["cov_messages"] == 0.0


def test_narrative_episodes_read_low_against_a_healthy_control():
    """The LME shape: episodes carry the session's gist, messages carry the answer."""
    probe = episode_probe(
        [_NARRATIVE,
         _mem("message_hit", "I drafted the schema, benchmarked the aggregation "
                             "rebuild, then flipped the granularity default.")],
        QUESTION, IDEAL)
    assert probe["cov_episodes"] < 0.3
    assert probe["cov_messages"] > 0.5


def test_absent_tier_is_none_not_zero():
    """"Not retrieved" and "retrieved and carried nothing" must not share an
    encoding -- conflating them is how a structural zero passes for a finding."""
    probe = episode_probe([_mem("message_hit", "x y z")], QUESTION, IDEAL)
    assert probe["n_episodes"] == 0
    assert probe["cov_episodes"] is None
    assert probe["gold_in_episodes"] is None
    # ... while a present-but-empty tier reports a real 0.0.
    present = episode_probe([_NARRATIVE], QUESTION, IDEAL)
    assert present["n_episodes"] == 1 and present["cov_episodes"] == 0.0


def test_no_distinctive_gold_tokens_is_unmeasurable_not_zero():
    probe = episode_probe([_mem("episode", "anything at all")],
                          "What did they do?", "They did it.")
    assert probe["n_gold_tokens"] == 0
    assert probe["cov_episodes"] is None


def test_question_terms_are_subtracted_from_the_gold_set():
    """Retrieval selected these memories BY the question, so question terms are
    covered by every tier for free; leaving them in measures the retriever."""
    raw, net = _content_tokens(IDEAL), _gold_tokens(IDEAL, QUESTION)
    assert net < raw
    assert "migration" in raw and "migration" not in net


def test_coverage_is_none_on_an_empty_tier_and_a_bare_denominator():
    assert _tier_coverage(set(), ["text"]) is None
    assert _tier_coverage({"alpha"}, []) is None
    assert _tier_coverage({"alpha"}, ["   ", ""]) is None
    assert _tier_coverage({"alpha", "beta"}, ["alpha only"]) == 0.5


def test_probe_does_not_mutate_the_reader_input():
    """It is a recording instrument; the run must be identical with it present."""
    memories = [_ANSWER_BEARING, _mem("message_hit", "unrelated")]
    before = [dict(m) for m in memories]
    episode_probe(memories, QUESTION, IDEAL)
    assert memories == before


def test_counted_tier_names_are_the_names_the_adapter_emits():
    """A tier rename would silently zero a column and read as 'tier absent'.

    Asserted against the adapter source rather than a hand-kept list, so the
    test fails on the rename instead of on the next analysis built over it.
    """
    src = (_BENCH / "beam_adapter.py").read_text()
    emitted = set(re.findall(r'"type":\s*"([a-z_]+)"', src))
    for tier in ("episode", "message_hit", "procedure", "fts_hit", "graph_fact",
                 "recent"):
        assert tier in emitted, f"probe counts a tier the adapter never emits: {tier}"


def test_every_probe_key_is_present_on_every_question():
    """A missing key would make the per-ability readout skip rows unevenly,
    turning a coverage question into a survivorship one."""
    expected = {
        "n_memories", "n_episodes", "n_messages", "n_procedures", "n_fts",
        "n_graph", "n_recent", "n_gold_tokens", "cov_episodes", "cov_messages",
        "cov_episodes_null", "cov_messages_null",
        "gold_in_episodes", "gold_in_messages",
    }
    for memories in ([], [_NARRATIVE], [_ANSWER_BEARING, _mem("recent", "hi")]):
        assert set(episode_probe(memories, QUESTION, IDEAL)) == expected


def test_both_decision_levers_are_pinned_explicitly(tmp_path, monkeypatch):
    """The pins must be PASSED, not inherited.

    `episode_granularity_enabled` currently agrees with the library default, so
    asserting the resulting value would pass just as well with no pin at all --
    and would keep passing right up until the default flipped and moved BEAM off
    its baseline silently. What is asserted instead is that the adapter names
    both levers itself, which is the property that survives a default change.
    """
    import hymem

    import beam_adapter

    seen: dict = {}
    real = hymem.HyMemConfig

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(hymem, "HyMemConfig", spy)
    adapter = beam_adapter.HyMemAdapter(tmp_path / "hymem.sqlite", api_key="unused")
    adapter.open()
    try:
        assert seen["aggregation_nodes_enabled"] is False
        assert seen["episode_granularity_enabled"] is False
    finally:
        adapter.close()


# ── the chance floor ───────────────────────────────────────────────
# Every BEAM question draws on one shared corpus, so a tier covers some answer
# tokens by vocabulary alone. Without a floor, any non-zero cov_ep reads as
# signal -- the same shape as a measure that returns a confident constant.

def test_null_is_none_without_a_decoy_rather_than_zero():
    probe = episode_probe([_ANSWER_BEARING], QUESTION, IDEAL)
    assert probe["cov_episodes_null"] is None
    assert probe["cov_messages_null"] is None


def test_the_null_fires_on_shared_vocabulary_and_stays_below_the_signal():
    """A tier that genuinely carries THIS answer must score above what it scores
    against another answer built from the same conversation's vocabulary."""
    decoy = ("The user reviewed the aggregation rebuild timings and discussed "
             "the schema at length.")
    probe = episode_probe([_ANSWER_BEARING], QUESTION, IDEAL, decoy_answer=decoy)
    assert probe["cov_episodes_null"] is not None
    assert probe["cov_episodes"] > probe["cov_episodes_null"]


def test_a_decoy_is_drawn_from_the_same_conversation_preferring_the_ability():
    questions = [
        {"ability_short": "EO", "gold_text": "first A then B", "gold_kind": "response"},
        {"ability_short": "IE", "gold_text": "the answer is C", "gold_kind": "response"},
        {"ability_short": "EO", "gold_text": "first D then E", "gold_kind": "response"},
    ]
    assert _decoy_answer(questions, 0) == "first D then E"   # same ability
    assert _decoy_answer(questions, 1) in {"first A then B", "first D then E"}
    assert _decoy_answer([questions[0]], 0) == ""            # nothing to draw on


# ── the pre-registered rule ────────────────────────────────────────

def test_verdict_requires_all_three_criteria():
    n = PROBE_MIN_ROWS
    # clears the floor and the control share
    assert _probe_verdict(n, 0.40, 0.60, 0.10, 0.10) == "YES"
    # fails the chance floor: coverage is shared vocabulary
    assert _probe_verdict(n, 0.19, 0.60, 0.10, 0.10) == "no"
    # clears the floor but is far below the control
    assert _probe_verdict(n, 0.20, 0.90, 0.05, 0.10) == "no"


def test_verdict_is_not_a_result_below_the_row_minimum():
    assert _probe_verdict(PROBE_MIN_ROWS - 1, 0.9, 0.1, 0.0, 0.0) == "n-a"


def test_a_control_that_cannot_beat_its_own_floor_invalidates_the_row():
    """If the message tier scores no better against the real answer than against
    an unrelated one, the measure failed there -- neither column is evidence."""
    assert _probe_verdict(PROBE_MIN_ROWS, 0.40, 0.10, 0.01, 0.10) == "INVALID"


def test_the_thresholds_are_the_pre_registered_ones():
    """Guards the pre-registration itself: a threshold edited after a run is a
    new pre-registration, and this test is where that has to be acknowledged."""
    assert (PROBE_MIN_ROWS, PROBE_NULL_MARGIN, PROBE_CONTROL_SHARE) == (12, 2.0, 0.5)


# ── the control has to be a SAME-ROW control ───────────────────────

def test_verdict_rows_are_fully_measured(capsys):
    """Every number the verdict reads must come from ONE row set.

    Averaging two independently-filtered columns compares different question
    sets and calls it a ratio -- and the same applies to a column against its
    own chance floor, which sits inside the decision rule rather than beside it.
    """
    both = episode_probe(
        [_NARRATIVE, _mem("message_hit", "drafted schema, benchmarked "
                          "aggregation rebuild, flipped granularity default")],
        QUESTION, IDEAL, decoy_answer=DECOY)
    msg_only = episode_probe(
        [_mem("message_hit", "totally unrelated chatter")], QUESTION, IDEAL,
        decoy_answer=DECOY)
    no_null = episode_probe(          # single-question conversation: no decoy
        [_NARRATIVE, _mem("message_hit", "drafted schema")], QUESTION, IDEAL)

    assert msg_only["cov_episodes"] is None
    assert no_null["cov_episodes_null"] is None

    print_episode_probe([{"questions": [
        {"ability": "EO", "probe": both},
        {"ability": "EO", "probe": msg_only},   # dropped: no episode coverage
        {"ability": "EO", "probe": no_null},    # dropped: no chance floor
    ]}])
    out = capsys.readouterr().out
    decision = out.split("ability     n")[0]
    eo = next(ln for ln in decision.splitlines() if ln.strip().startswith("EO"))
    assert eo.split()[1] == "1"
    # ... while the secondary table still reports all three raw rows, so the
    # gap between "questions asked" and "questions the verdict rests on" is
    # visible rather than silently absorbed.
    secondary = out.split("ability     n")[1]
    eo2 = next(ln for ln in secondary.splitlines() if ln.strip().startswith("EO"))
    assert eo2.split()[1] == "3"


def test_the_pooled_row_emits_no_verdict(capsys):
    """Pooling abilities can clear every criterion while its components
    disagree, so ALL reports numbers and withholds a decision."""
    ep_good = _mem("episode", "drafted schema, benchmarked aggregation "
                              "rebuild, flipped granularity default")
    msg = _mem("message_hit", "drafted schema, benchmarked aggregation "
                              "rebuild, flipped granularity default")
    questions = (
        [{"ability": "EO", "probe": episode_probe([ep_good, msg], QUESTION,
                                                  IDEAL, DECOY)}
         for _ in range(PROBE_MIN_ROWS + 2)]
        + [{"ability": "SUM", "probe": episode_probe([_NARRATIVE, msg], QUESTION,
                                                     IDEAL, DECOY)}
           for _ in range(PROBE_MIN_ROWS + 1)])
    print_episode_probe([{"questions": questions}])
    decision = capsys.readouterr().out.split("ability     n")[0].splitlines()
    verdicts = {ln.split()[0]: ln.split()[-1] for ln in decision
                if ln.strip().startswith(("EO", "SUM", "ALL"))}
    assert verdicts["EO"] == "YES" and verdicts["SUM"] == "no"
    assert verdicts["ALL"] == "—"


# ── the gold-field map ─────────────────────────────────────────────
# BEAM keys the gold answer differently per ability. The old parse
# (`q.get("ideal_response", q.get("ideal_answer", ""))`) resolved for 2 of 10
# and returned "" for the rest, so the coverage probe had no denominator on
# the abilities it exists to measure and the judge received an empty IDEAL
# ANSWER field. The failure was silent because a lookup that could not find
# its value returned a value anyway, and nothing asked whether it was real.

def test_every_ability_has_a_gold_field():
    """The map must be exhaustive over the abilities the adapter can emit --
    a missing entry is how eight of them went three months without gold."""
    covered = {ABILITY_MAP[a] for a in GOLD_FIELDS if a in ABILITY_MAP}
    assert covered == set(ABILITY_MAP.values()), (
        f"abilities with no GOLD_FIELDS entry: "
        f"{set(ABILITY_MAP.values()) - covered}")


@pytest.mark.parametrize("ability,field,kind", [
    ("abstention", "ideal_response", "response"),
    ("event_ordering", "answer", "response"),
    ("summarization", "ideal_summary", "summary"),
    ("instruction_following", "expected_compliance", "compliance_spec"),
])
def test_gold_resolves_from_the_ability_s_own_field(ability, field, kind):
    text, got_kind = _resolve_gold({field: "the gold text"}, ability)
    assert (text, got_kind) == ("the gold text", kind)


def test_missing_gold_is_kind_none_not_an_empty_answer():
    """"" as a gold answer is indistinguishable from a real empty one. The
    kind carries the difference so downstream can check it."""
    assert _resolve_gold({"question": "q"}, "event_ordering") == ("", "none")


def test_a_map_miss_recovers_loudly_rather_than_silently(capsys):
    text, kind = _resolve_gold({"ideal_summary": "recovered"}, "event_ordering")
    assert text == "recovered"
    out = capsys.readouterr().out
    assert "gold-field map miss" in out and "event_ordering" in out


def test_compliance_specs_are_never_probe_gold():
    """A spec describes what an answer must DO. Scoring tier coverage against
    it measures the spec's vocabulary, which is a different quantity."""
    assert "compliance_spec" not in PROBE_GOLD_KINDS
    spec = {"gold_text": "must mention the deadline", "gold_kind": "compliance_spec"}
    real = {"gold_text": "the deadline is April 5", "gold_kind": "response"}
    assert _probe_gold(spec) == ""
    assert _probe_gold(real) == "the deadline is April 5"


def test_the_probe_has_a_denominator_on_an_event_ordering_question():
    """The regression test for the whole defect: before the map, an EO question
    parsed to "" and every coverage number on it was None for lack of gold."""
    q = {"ability": "event_ordering",
         "answer": "First the schema migration, then the aggregation rebuild."}
    text, kind = _resolve_gold(q, "event_ordering")
    probe = episode_probe([_ANSWER_BEARING], QUESTION,
                          _probe_gold({"gold_text": text, "gold_kind": kind}))
    assert probe["n_gold_tokens"] > 0
    assert probe["cov_episodes"] is not None


def test_decoys_are_only_drawn_from_questions_with_usable_gold():
    questions = [
        {"ability_short": "EO", "gold_text": "a", "gold_kind": "response"},
        {"ability_short": "IF", "gold_text": "spec", "gold_kind": "compliance_spec"},
        {"ability_short": "EO", "gold_text": "real gold", "gold_kind": "response"},
    ]
    assert _decoy_answer(questions, 0) == "real gold"
    # The compliance-spec row is not a usable decoy for anyone.
    assert _decoy_answer(questions, 1) in {"a", "real gold"}


def test_the_gold_audit_names_abilities_with_no_gold(capsys):
    conversations = {"100K": [{"questions": [
        {"ability_short": "EO", "gold_text": "", "gold_kind": "none"},
        {"ability_short": "ABS", "gold_text": "yes", "gold_kind": "response"},
    ]}]}
    print_gold_audit(conversations)
    out = capsys.readouterr().out
    assert "MISSING" in out and "WARNING" in out
    assert "EO" in out


def test_the_judge_s_field_is_unchanged_by_the_probe_fix():
    """The parse fix must not silently repoint the judge.

    Feeding the judge real gold changes what every BEAM score MEANS -- post-fix
    runs stop being comparable to v13-v16 -- so it is gated behind --judge-gold
    and pre-registered separately. Here the two fields must DIFFER on an EO
    question: `gold_text` resolves, `ideal_answer` stays empty exactly as it
    was for the runs already in the record.
    """
    from beam_adapter import _parse_sample

    sample = {
        "conversation_id": "c1",
        "chat": [[{"role": "user", "content": "hi", "time_anchor": None}]],
        "probing_questions": {
            "event_ordering": [
                {"question": "in what order?", "answer": "first A then B",
                 "rubric": ["mentions A before B"]},
            ],
        },
    }
    q = _parse_sample(sample, "100K", 0)["questions"][0]
    assert q["gold_text"] == "first A then B"
    assert q["gold_kind"] == "response"
    assert q["ideal_answer"] == ""      # the judge's input, deliberately untouched
