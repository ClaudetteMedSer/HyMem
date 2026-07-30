"""Tests for `benchmarks/reldate_probe.py` — the E4 front-run gate.

The probe decides whether E4 gets built, so what is tested here is not "does the
resolver parse dates" but the three ways a gate instrument can lie:

  1. resolving something that is not relative (a FALSE FIRE inflates the fire
     rate, the criterion the build hangs on),
  2. anchoring to the wrong date (which reports the ANCHOR's error as the
     RESOLVER's imprecision — the exact class of defect that made the G-F1 date
     reading unusable), and
  3. mis-attributing the miss direction (the after/before split is what
     separates a bi-temporal axis mismatch from a resolver accuracy problem, and
     the two have opposite remedies).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from reldate_probe import (  # noqa: E402
    effective_anchor,
    has_temporal_language,
    load_lme,
    load_locomo,
    measure,
    normalize_date,
    parse_locomo_date,
    resolve_range,
    summarize,
    vague_markers,
)

ANCHOR = date(2024, 3, 15)  # a Friday


def _r(text, anchor=ANCHOR):
    return resolve_range(text, anchor)


# ── resolvable forms ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,rule,start,end", [
    ("what did we decide yesterday?", "day_word", "2024-03-14", "2024-03-14"),
    ("wat hebben we gisteren besloten?", "day_word", "2024-03-14", "2024-03-14"),
    ("what happened last night?", "day_word", "2024-03-14", "2024-03-14"),
    # calendar-aligned, not rolling: last week is that Mon-Sun block
    ("what did we ship last week?", "calendar_last", "2024-03-04", "2024-03-10"),
    ("wat hebben we vorige week gedaan?", "calendar_last",
     "2024-03-04", "2024-03-10"),
    ("the plan from last month", "calendar_last", "2024-02-01", "2024-02-29"),
    ("projects from last year", "calendar_last", "2023-01-01", "2023-12-31"),
    ("between 2023-01-05 and 2023-02-01", "between", "2023-01-05", "2023-02-01"),
])
def test_resolvable_forms(text, rule, start, end):
    hit = _r(text)
    assert hit is not None, text
    assert (hit.rule, hit.start, hit.end) == (rule, start, end)


@pytest.mark.parametrize("text,expect_start,expect_end", [
    # 2 weeks ago = 2024-03-01 ± 3 days
    ("we talked about it two weeks ago", "2024-02-27", "2024-03-04"),
    ("dat was twee weken geleden", "2024-02-27", "2024-03-04"),
    ("3 days ago", "2024-03-11", "2024-03-13"),
    ("a couple of months back", "2024-01-08", "2024-01-22"),
])
def test_n_units_ago_window(text, expect_start, expect_end):
    hit = _r(text)
    assert hit is not None and hit.rule == "n_units_ago"
    assert (hit.start, hit.end) == (expect_start, expect_end)


def test_within_last_n_is_bounded_by_the_anchor():
    hit = _r("what did I log in the last 10 days?")
    assert hit.rule == "within_last_n"
    assert (hit.start, hit.end) == ("2024-03-05", "2024-03-15")


def test_dutch_afgelopen_resolves():
    hit = _r("wat heb ik de afgelopen twee weken gedaan?")
    assert hit is not None
    assert hit.start == "2024-03-01" and hit.end == "2024-03-15"


# ── the three ways this instrument could lie ────────────────────────────────────
@pytest.mark.parametrize("text", [
    "what did Caroline say about her job?",
    "which book did he recommend?",
    "wat is het adres van de kliniek?",
    "how many people were at the wedding?",
])
def test_no_temporal_language_never_fires(text):
    """Control population. A fire here is a false fire by construction, and the
    no-harm criterion is zero — the same shape as the E5 gate."""
    assert _r(text) is None
    assert not has_temporal_language(text)


@pytest.mark.parametrize("text", [
    "where was Dave in the last week of August 2023?",
    "which country during the last week of August 2023?",
    "what happened in the first week of maart 2022?",
])
def test_absolute_of_month_is_not_a_relative_expression(text):
    """'the last week OF August 2023' is anchored to a named period, not to now.
    Unguarded it resolves to a window that can be a year off — a false fire
    wearing a low-precision hit's clothes, which inflates the fire rate AND
    depresses precision at the same time."""
    assert _r(text) is None


def test_vague_markers_are_never_resolved():
    """Vague temporal intent is real intent with no arithmetic. Inventing a
    window for it is the failure mode the probe exists to avoid."""
    for text in ("what did we discuss recently?", "a while back we agreed",
                 "wat hebben we onlangs besproken?", "the other day you said"):
        assert _r(text) is None
        assert vague_markers(text)
        assert has_temporal_language(text)  # counted, just not as a fire


def test_effective_anchor_prefers_a_date_stated_in_the_text():
    """LoCoMo writes '…, as mentioned on November 6, 2023'; production passes
    `question_date`. Ignoring it measures the anchor's error as the resolver's."""
    default = date(2024, 1, 1)
    for text, want in [
        ("what was the issue last week, as mentioned on November 6, 2023?",
         date(2023, 11, 6)),
        ("what did we finish last week before 23 January, 2023?",
         date(2023, 1, 23)),
        ("what changed since 2022-07-04?", date(2022, 7, 4)),
    ]:
        got, overridden = effective_anchor(text, default)
        assert (got, overridden) == (want, True), text


def test_effective_anchor_falls_back_when_no_date_is_stated():
    got, overridden = effective_anchor("what did we do last week?", ANCHOR)
    assert got == ANCHOR and overridden is False


def test_miss_direction_separates_axis_mismatch_from_resolver_error():
    """The load-bearing diagnostic. Gold consistently AFTER the range means the
    range is about EVENT time while the stored date is SPEECH time; gold on both
    sides would mean the resolver is simply inaccurate. Opposite remedies."""
    rows = [
        # "4 years ago" resolves correctly to 2020; the turn SAYING it is 2023.
        {"id": "a", "text": "where did she move from 4 years ago?",
         "anchor": "2024-01-04", "gold_dates": ["2023-06-09"]},
        {"id": "b", "text": "what did he start two years ago?",
         "anchor": "2024-01-11", "gold_dates": ["2023-10-25"]},
    ]
    res = measure(rows, name="t", gating=True)
    assert res["fired"] == 2 and res["precision"] == 0.0
    assert res["miss_sides"] == {"after": 2}


def test_a_range_can_be_perfectly_precise_and_perfectly_useless():
    """The hole G-E4a had. Precision asks whether the gold is inside the window;
    selectivity asks what ELSE is. 'this year' over a corpus that is entirely one
    year scores 100% on the first and boosts every item in the store."""
    corpus = [f"2024-{m:02d}-10" for m in range(1, 13)]
    rows = [{"id": "a", "text": "what did I do this year?", "anchor": "2024-05-30",
             "gold_dates": ["2024-03-10"], "corpus_dates": corpus}]
    res = measure(rows, name="t", gating=True)
    assert res["precision"] == 100.0
    assert res["rows_fired"][0]["selectivity"] == 1.0
    assert res["selective_fired"] == 0 and res["wide_ranges"] == 1


def test_an_empty_window_is_not_the_good_end_of_the_selectivity_scale():
    """0% selectivity reads as 'narrow' in a median and is really 'boosts
    nothing'. Both tails are dead and they must not be pooled."""
    corpus = [(date(2024, 1, 1) + timedelta(days=4 * i)).isoformat()
              for i in range(40)]
    rows = [
        # resolves to 2022 — before anything in the corpus exists.
        {"id": "a", "text": "what happened two years ago?",
         "anchor": "2024-05-30", "gold_dates": [], "corpus_dates": corpus},
        # resolves inside the corpus and covers 5% of it.
        {"id": "b", "text": "what happened four weeks ago?",
         "anchor": "2024-05-30", "gold_dates": [], "corpus_dates": corpus},
    ]
    res = measure(rows, name="t", gating=True)
    assert res["empty_ranges"] == 1
    assert res["selective_fired"] == 1
    assert res["selective_fire_rate"] == 50.0


def test_selectivity_counts_repeated_corpus_dates_separately():
    """Two sessions on one date are two things a boost would lift. De-duplicating
    the corpus flatters a wide window."""
    rows = [{"id": "a", "text": "what did I do last week?", "anchor": "2024-05-30",
             "gold_dates": [],
             "corpus_dates": ["2024-05-22", "2024-05-22", "2024-05-22",
                              "2024-01-01"]}]
    res = measure(rows, name="t", gating=True)
    assert res["rows_fired"][0]["selectivity"] == 0.75


def test_selectivity_is_absent_when_the_corpus_is_unknown():
    rows = [{"id": "a", "text": "what did I do last week?",
             "anchor": "2024-05-30", "gold_dates": []}]
    res = measure(rows, name="t", gating=True)
    assert "selectivity" not in res["rows_fired"][0]
    assert res["selectivity"] is None and res["selective_fire_rate"] is None


def test_per_rule_precision_exposes_an_average_over_a_broken_rule():
    """An aggregate can clear 90% while the single most common construction is
    wrong, with rarely-firing rules carrying the mean. A boost that misses on the
    expression people actually use is worse than no boost, so the split is
    reported rather than left inside the average."""
    rows = [
        # "last month" — arithmetically right, gold sits after the window.
        {"id": "a", "text": "what did I finish last month?",
         "anchor": "2024-05-30", "gold_dates": ["2024-05-20"]},
        {"id": "b", "text": "how many hours last month?",
         "anchor": "2024-05-30", "gold_dates": ["2024-05-22"]},
        # a different rule, correct.
        {"id": "c", "text": "what did I buy three weeks ago?",
         "anchor": "2024-05-30", "gold_dates": ["2024-05-09"]},
    ]
    by_rule = measure(rows, name="t", gating=True)["by_rule"]
    assert by_rule["calendar_last"]["scored"] == 2
    assert by_rule["calendar_last"]["hits"] == 0
    assert by_rule["n_units_ago"]["hits"] == 1


def test_per_rule_precision_skips_rules_with_no_scorable_gold():
    rows = [{"id": "a", "text": "what did I finish last month?",
             "anchor": "2024-05-30", "gold_dates": []}]
    by_rule = measure(rows, name="t", gating=True)["by_rule"]
    assert by_rule["calendar_last"]["fired"] == 1
    assert by_rule["calendar_last"]["scored"] == 0


def test_future_windows_are_counted_but_never_fired():
    """Superseded by revision 1: a forward-facing window used to be a fire (and
    therefore always a precision miss). It is now its own category."""
    rows = [{"id": "f", "text": "what is planned for next month?",
             "anchor": "2024-03-15", "gold_dates": ["2023-09-21"]}]
    res = measure(rows, name="t", gating=True)
    assert res["fired"] == 0 and res["prospective"] == 1


# ── gate arithmetic ─────────────────────────────────────────────────────────────
def _pop(**kw):
    base = {"name": "p", "gating": True, "n": 100, "fired": 10,
            "fire_rate": 10.0, "vague_only": 0, "vague_rate": 0.0,
            "any_temporal": 10, "n_control": 90, "control_fires": 0,
            "reanchored": 0, "prospective": 0, "miss_sides": {},
            "precision_n": 10, "precision": 100.0, "rules": {},
            "rows_fired": [], "rows_vague": [], "rows_control_fires": []}
    return {**base, **kw}


def test_gate_passes_only_when_all_three_criteria_hold():
    assert summarize([_pop()])["pass"] is True


@pytest.mark.parametrize("override,failing", [
    ({"fired": 2, "fire_rate": 2.0}, "fire_rate_ok"),
    ({"precision": 50.0}, "precision_ok"),
    ({"control_fires": 1}, "no_harm_ok"),
])
def test_each_criterion_can_fail_alone(override, failing):
    s = summarize([_pop(**override)])
    assert s["pass"] is False and s["gate"][failing] is False


def test_precision_below_the_discrimination_floor_is_unread_not_failed():
    """A probe that fails a build on 3 scored questions is worse than one that
    says it cannot tell — the LME variance-band lesson applied to an n, not a
    delta."""
    s = summarize([_pop(precision_n=3, precision=33.0)])
    assert s["precision_read"] is False
    # Not FAIL — the number is noise at n=3 and a probe must not kill a build on
    # noise. Not PASS either: unread is unmeasured.
    assert s["verdict"] == "INCOMPLETE"


def test_non_gating_populations_never_enter_the_verdict():
    """The content-side turn population is context, not evidence: a corpus whose
    TURNS are full of relative dates says nothing about whether a QUERY-side
    boost fires."""
    s = summarize([_pop(fired=2, fire_rate=2.0),
                   _pop(name="turns", gating=False, fired=90, fire_rate=90.0)])
    assert s["pass"] is False and s["fire_rate"] == pytest.approx(2.0)


# ── LoCoMo loading ──────────────────────────────────────────────────────────────
def test_parse_locomo_date_reads_the_benchmarks_own_stamp_format():
    """'1:56 pm on 8 May, 2023' — `dreaming/dates.py` drops the year on this
    form, so the adapter owns it rather than a production parser being loosened
    for one benchmark."""
    assert parse_locomo_date("1:56 pm on 8 May, 2023") == date(2023, 5, 8)
    assert parse_locomo_date("no date here") is None


def test_load_locomo_anchors_questions_to_the_last_session(tmp_path):
    sample = [{
        "sample_id": "s1",
        "conversation": {
            "session_1_date_time": "1:00 pm on 8 May, 2023",
            "session_1": [{"text": "I moved here last week"}],
            "session_2_date_time": "2:00 pm on 3 July, 2023",
            "session_2": [{"text": "nothing temporal"}],
        },
        "qa": [{"question": "when did she move?", "evidence": ["D1:0"]}],
    }]
    path = tmp_path / "locomo.json"
    path.write_text(__import__("json").dumps(sample))
    questions, turns = load_locomo(path)
    # Questions are posed after the conversation ends → last session date.
    assert questions[0]["anchor"] == "2023-07-03"
    assert questions[0]["gold_dates"] == ["2023-05-08"]
    # Turns are anchored to their OWN session — a turn's "last week" is relative
    # to when it was said, not to the end of the conversation.
    assert [t["anchor"] for t in turns] == ["2023-05-08", "2023-07-03"]


# ── verdict states: an unmeasured criterion is not a satisfied one ──────────────
def test_unmeasured_precision_is_incomplete_not_pass():
    """The defect this guards: with no dated gold, `precision is None` used to
    satisfy criterion 2, so a 2-of-3 run printed PASS. `fact_probe.py` reports
    INCOMPLETE in the same situation and this probe now agrees."""
    s = summarize([_pop(precision_n=0, precision=None)])
    assert s["verdict"] == "INCOMPLETE"
    assert s["pass"] is False
    assert s["gate"]["precision_ok"] is False


def test_a_failing_measured_criterion_outranks_incomplete():
    """FAIL beats INCOMPLETE: a run that misses the fire-rate floor is decided,
    whether or not precision was measurable."""
    s = summarize([_pop(fired=1, fire_rate=1.0, precision_n=0, precision=None)])
    assert s["verdict"] == "FAIL"


def test_all_three_measured_and_holding_is_a_real_pass():
    assert summarize([_pop()])["verdict"] == "PASS"


def test_rows_with_gold_separates_a_loader_failure_from_a_bare_corpus():
    """Zero rows carrying gold looks identical to 'no fired question had gold'
    downstream, but one is a bug and the other is a corpus property."""
    rows = [{"id": "a", "text": "what did we do last week?",
             "anchor": "2024-03-15", "gold_dates": []}]
    assert measure(rows, name="t", gating=True)["rows_with_gold"] == 0
    rows[0]["gold_dates"] = ["2024-03-06"]
    assert measure(rows, name="t", gating=True)["rows_with_gold"] == 1


# ── LME loading: the format that silently voided criterion 2 ────────────────────
@pytest.mark.parametrize("raw,want", [
    ("2023/05/20 (Sat) 02:21", "2023-05-20"),   # LongMemEval's actual stamp
    ("2023-05-20", "2023-05-20"),
    ("2023/5/6", "2023-05-06"),
    ("6 November, 2023", "2023-11-06"),
    ("", ""),
    ("not a date", ""),
    ("2023/13/45", ""),
])
def test_normalize_date_accepts_the_slash_format(raw, want):
    """`str(raw)[:10]` + an ISO match drops EVERY LongMemEval date, which
    reports criterion 2 as a confident 'n/a' instead of failing loudly."""
    assert normalize_date(raw) == want


def test_load_lme_attaches_gold_dates_from_slash_stamps(tmp_path):
    data = [{
        "question_id": "q1", "question": "what did we decide last week?",
        "question_date": "2023/05/20 (Sat) 02:21",
        "question_type": "temporal-reasoning",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/05/10 (Wed) 09:00", "2023/05/18 (Thu) 11:00"],
        "answer_session_ids": ["s2"],
    }]
    path = tmp_path / "lme.json"
    path.write_text(__import__("json").dumps(data))
    rows = load_lme(path)
    assert rows[0]["anchor"] == "2023-05-20"
    assert rows[0]["gold_dates"] == ["2023-05-18"]  # gold session only


def test_by_category_tracks_fires_per_category():
    """A fire rate carried by one annotator-designed category is a property of
    the benchmark, not of how people ask — a different claim from 'queries carry
    relative dates'."""
    rows = [
        {"id": "1", "text": "what did we do last week?", "anchor": "2024-03-15",
         "gold_dates": [], "category": "temporal-reasoning"},
        {"id": "2", "text": "what is her address?", "anchor": "2024-03-15",
         "gold_dates": [], "category": "single-session-user"},
    ]
    cats = measure(rows, name="t", gating=True)["by_category"]
    assert cats["temporal-reasoning"] == {"n": 1, "fired": 1}
    assert cats["single-session-user"] == {"n": 1, "fired": 0}


def test_balanced_misses_are_not_the_axis_mismatch():
    """The complement of `test_miss_direction_separates_axis_mismatch_...`.
    A roughly even after/before split means the resolver is inaccurate on
    particular expressions, NOT that the dates are on the wrong axis — the
    2026-07-30 LME run (5 after / 4 before at 80.9%) was read as the LoCoMo
    mechanism (15/4 at 20.8%) purely because the report said nothing when the
    split was balanced."""
    rows = [
        # window lands before the gold
        {"id": "a", "text": "what did we decide last week?",
         "anchor": "2024-03-15", "gold_dates": ["2024-03-20"]},
        # window lands after the gold
        {"id": "b", "text": "what did we decide last week?",
         "anchor": "2024-03-15", "gold_dates": ["2024-02-01"]},
    ]
    res = measure(rows, name="t", gating=True)
    assert res["miss_sides"] == {"after": 1, "before": 1}
    after, before = res["miss_sides"]["after"], res["miss_sides"]["before"]
    assert after < 3 * max(before, 1)  # the axis warning must NOT fire


def test_misses_are_never_hidden_by_the_row_cap(capsys):
    """A capped diagnostic table that still reports the full miss COUNT hides
    evidence: the 2026-07-30 LME run showed 8 of 9 misses because `fired` (47)
    overran a flat [:40] cap, and the gap was read as a counting bug rather than
    a truncated table. Misses drive the revision decision, so they are exempt."""
    from reldate_probe import report

    rows = [{"id": f"h{i}", "text": "what did we do last week?",
             "anchor": "2024-03-15", "gold_dates": ["2024-03-06"]}
            for i in range(12)]
    rows.append({"id": "miss", "text": "what did we do last week?",
                 "anchor": "2024-03-15", "gold_dates": ["2024-01-01"]})
    pop = measure(rows, name="t", gating=True)
    report([pop], summarize([pop]), verbose=True, limit=3)
    out = capsys.readouterr().out
    assert "miss" in out            # the single miss survives a limit of 3
    assert "rows hidden" in out     # and the truncation is stated, not silent


# ── revision 1 (pre-registered): directional qualifiers + prospective windows ───
@pytest.mark.parametrize("text,rule,start,end", [
    # "before today" denotes everything up to now, not the single day.
    ("what is the order of airlines I flew with before today?",
     "before_day_word", "0001-01-01", "2024-03-15"),
    ("which projects did I finish before last week?",
     "before_calendar_last", "0001-01-01", "2024-03-10"),
    # "since X" opens forward to the anchor instead of sitting on X.
    ("how much have I written since I started again three weeks ago?",
     "since_n_units_ago", "2024-02-20", "2024-03-15"),
    ("what changed since last month?",
     "since_calendar_last", "2024-02-01", "2024-03-15"),
])
def test_directional_qualifiers_open_the_window(text, rule, start, end):
    """The defect class the 2026-07-30 LME misses exposed: these expressions
    denote HALF-OPEN intervals and the resolver emitted a point window."""
    hit = _r(text)
    assert hit is not None, text
    assert (hit.rule, hit.start, hit.end) == (rule, start, end)
    assert hit.prospective is False


@pytest.mark.parametrize("text", [
    "can you recommend a show for me to watch tonight?",
    "I've got free time tonight, any documentary recommendations?",
    "I'm planning my meal prep next week, any suggestions?",
    "wat zal ik vanavond kijken?",
])
def test_prospective_windows_resolve_but_never_fire(text):
    """A forward-facing window can never contain a stored past item, so boosting
    on it is cost with no upside. It still RESOLVES — suppressing it silently
    would sort a temporal question into the marker-free control."""
    hit = _r(text)
    assert hit is not None and hit.prospective is True
    assert has_temporal_language(text)


def test_prospective_questions_are_neither_fires_nor_controls():
    """The bookkeeping that matters: the 2026-07-30 LME control read 0/453 as
    clean partly because two prospective questions sat in the FIRED bucket."""
    rows = [
        {"id": "p", "text": "any recommendations for tonight?",
         "anchor": "2024-03-15", "gold_dates": ["2024-03-01"]},
        {"id": "f", "text": "what did we decide last week?",
         "anchor": "2024-03-15", "gold_dates": ["2024-03-06"]},
    ]
    res = measure(rows, name="t", gating=True)
    assert res["prospective"] == 1
    assert res["fired"] == 1           # only the retrospective question
    assert res["n_control"] == 0       # the prospective one is NOT a control row
    assert res["precision"] == 100.0   # and never counts as a precision miss


def test_a_qualifier_flips_a_prospective_base_back_to_retrospective():
    """Order of operations: the qualifier is applied BEFORE the prospective
    check, because 'before tomorrow' asks about the past."""
    hit = _r("what did I log before tomorrow?")
    assert hit.prospective is False and hit.start == "0001-01-01"


def test_a_distant_qualifier_does_not_capture_the_expression():
    """The cue window is bounded so a `since` belonging to another clause cannot
    reach across the sentence and reshape an unrelated range."""
    text = ("since the very beginning of this long and rambling account of "
            "everything that happened, what did we decide last week?")
    hit = _r(text)
    assert hit.rule == "calendar_last"  # not since_calendar_last
