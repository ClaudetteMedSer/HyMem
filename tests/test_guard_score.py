"""Gate 4's scorer: the refusal has to be structural, not a habit.

The property under test is that `report()` computes NO accuracy when the pair
cannot evidence its own contrast. Not "prints a warning first" -- does not
compute it. The previous guard pair's 71.0 vs 71.0 was quoted in three places
before anyone asked which arm was which, and the reason that was possible is
that reading the numbers was one step and checking provenance was another.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import guard_score as gs  # noqa: E402


def artifact(*, lever=None, overall=71.0, ms=53.4, n=6, episodes=None,
             correct_from=0, correct=None, qtype="multi-session",
             qtypes=None, cfg_extra=None):
    """One arm. `lever=None` omits the key entirely (the pre-6543ee6 shape).

    `correct` gives the per-question outcomes explicitly. The gate is paired
    now, so it reads those and not the headline `overall` -- an arm whose
    summary says 60.0 while every question matches the other arm has not
    regressed by anything this test can see, and that is the point."""
    cfg = {"scale": "S", "sample": 0, "workers": 8}
    if lever is not None:
        cfg[gs.LEVER] = lever
    cfg.update(cfg_extra or {})
    if correct is not None:
        n = len(correct)
    pq = []
    for i in range(n):
        row = {"question_id": f"q{i}",
               "correct": correct[i] if correct is not None else i >= correct_from,
               "question_type": qtypes[i] if qtypes is not None else qtype}
        if episodes is not None:
            row["n_episodes"] = episodes[i]
        pq.append(row)
    return {
        "config": cfg,
        "scores": {"OVERALL": {"accuracy": overall, "count": n},
                   gs.MS_KEY: {"accuracy": ms, "count": n}},
        "per_question": pq,
    }


def run(a, b):
    lines: list[str] = []
    verdict, detail = gs.report(a, b, out=lines.append)
    return verdict, detail, "\n".join(lines)


# ------------------------------------------------------- the structural refusal

def test_an_unevidenced_pair_is_INCOMPLETE_and_prints_no_score():
    v, detail, text = run(artifact(lever=None), artifact(lever=None))
    assert v == "INCOMPLETE"
    assert "checks" not in detail, "no bar was evaluated"
    assert "71.0" not in text and "53.4" not in text
    assert "OVERALL" not in text


def test_a_same_arm_pair_is_also_INCOMPLETE():
    """Both arms recording False is not an A/B, however the files are named."""
    v, detail, text = run(artifact(lever=False), artifact(lever=False))
    assert v == "INCOMPLETE"
    assert "71.0" not in text


def test_a_pair_with_one_silent_arm_is_INCOMPLETE():
    v, _, text = run(artifact(lever=False), artifact(lever=None))
    assert v == "INCOMPLETE"
    assert "71.0" not in text


# ------------------------------------------------------------------- the bar

def test_an_evidenced_pair_that_clears_the_bar_passes():
    v, _, text = run(artifact(lever=False), artifact(lever=True))
    assert v == "PASS"
    assert "OVERALL" in text


# 20 questions ON lost and none it gained: p = 2^-19, unmistakably a
# regression. 20 the other way is the mirror image and must NOT be one.
_OFF_ALL_RIGHT = [True] * 20 + [True] * 10
_ON_LOST_20 = [False] * 20 + [True] * 10
_OFF_LOST_20 = [False] * 20 + [True] * 10
_ON_ALL_RIGHT = [True] * 20 + [True] * 10


def test_a_significant_paired_loss_is_a_REGRESSION():
    v, d, text = run(artifact(lever=False, correct=_OFF_ALL_RIGHT),
                     artifact(lever=True, correct=_ON_LOST_20))
    assert v == "FAIL"
    assert "REGRESSION" in text
    assert d["paired"]["OVERALL"]["regressed"] == 20


def test_a_headline_gap_with_no_paired_loss_is_not_a_regression():
    """The defect the old bars had, inverted into a test. Two arms that agree
    on every single question cannot have regressed, whatever their recorded
    summaries say -- and under `OVERALL >= 70.0` the second one FAILED."""
    v, _, _ = run(artifact(lever=False, overall=75.0, correct=_OFF_ALL_RIGHT),
                  artifact(lever=True, overall=60.0, correct=_OFF_ALL_RIGHT))
    assert v == "PASS"


def test_an_improvement_is_not_read_as_a_regression():
    """Direction is half the verdict: McNemar rejects on either tail."""
    v, _, text = run(artifact(lever=False, correct=_OFF_LOST_20),
                     artifact(lever=True, correct=_ON_ALL_RIGHT))
    assert v == "PASS"
    assert "NO REGRESSION DETECTED" in text


def test_the_arms_are_oriented_by_the_config_block_not_by_argv_order():
    """Orientation matters MORE under a paired test than under a bar: swap
    the arms and a regression reads as an improvement, which passes."""
    v1, d1, _ = run(artifact(lever=False, correct=_OFF_ALL_RIGHT),
                    artifact(lever=True, correct=_ON_LOST_20))
    v2, d2, _ = run(artifact(lever=True, correct=_ON_LOST_20),
                    artifact(lever=False, correct=_OFF_ALL_RIGHT))
    assert v1 == v2 == "FAIL"
    assert d1["paired"]["OVERALL"]["regressed"] == \
        d2["paired"]["OVERALL"]["regressed"] == 20


# --------------------------------------------------------------- fired subset

def test_a_missing_episode_count_reads_UNAVAILABLE_not_no_effect():
    """An arm predating 6543ee6 records no n_episodes. That is a gap in the
    instrument; reporting it as a null would be the locomo_flip defect."""
    v, detail, text = run(artifact(lever=False), artifact(lever=True))
    assert v == "PASS"
    assert detail["fired"]["available"] is False
    assert "UNAVAILABLE" in text
    assert "gap in the instrument" in text


def test_the_fired_subset_is_the_questions_whose_episode_count_moved():
    a = artifact(lever=False, episodes=[1, 1, 1, 1, 1, 1], correct_from=3)
    b = artifact(lever=True, episodes=[1, 1, 1, 4, 4, 4], correct_from=0)
    _, detail, _ = run(a, b)
    fs = detail["fired"]
    assert fs["available"] and fs["shared"] == 6 and fs["fired"] == 3
    # q3,q4,q5 moved: OFF has them correct, ON has them correct -> 100/100.
    assert fs["a_fired"] == 100.0 and fs["b_fired"] == 100.0
    # q0,q1,q2 unmoved: OFF wrong, ON correct.
    assert fs["a_same"] == 0.0 and fs["b_same"] == 100.0


def test_identical_episode_counts_make_the_fired_subset_empty():
    a = artifact(lever=False, episodes=[2] * 6)
    b = artifact(lever=True, episodes=[2] * 6)
    _, detail, text = run(a, b)
    assert detail["fired"]["fired"] == 0
    assert "LOWER BOUND" in text, (
        "an empty subset is exactly when the caveat matters most: the lever "
        "may have re-cut every episode without changing any count")


def test_the_lower_bound_caveat_is_always_printed_when_available():
    a = artifact(lever=False, episodes=[1, 2, 3, 4, 5, 6])
    b = artifact(lever=True, episodes=[6, 5, 4, 3, 2, 1])
    _, _, text = run(a, b)
    assert "LOWER BOUND" in text


def test_episode_totals_are_reported_for_both_arms():
    a = artifact(lever=False, episodes=[1] * 6)
    b = artifact(lever=True, episodes=[3] * 6)
    _, detail, _ = run(a, b)
    assert detail["fired"]["a_episodes_total"] == 6
    assert detail["fired"]["b_episodes_total"] == 18


# ------------------------------------------------------------------ accuracy

@pytest.mark.parametrize("rows,expected", [
    ([], None),
    ([{"correct": True}], 100.0),
    ([{"correct": True}, {"correct": False}], 50.0),
])
def test_accuracy_handles_the_empty_subset(rows, expected):
    assert gs.accuracy(rows) == expected


# ------------------------------------------------- the paired design's own rules

def test_a_negative_verdict_never_says_bare_no_regression():
    """The load-bearing sentence. "No regression" is a claim the instrument
    cannot support; "no regression larger than Xpp" is the one it can. The
    old gate said the first, which is how a null at 2.5pp resolution got read
    as evidence the lever was harmless."""
    _, _, text = run(artifact(lever=False, correct=[True] * 10 + [False] * 10),
                     artifact(lever=True, correct=[True] * 9 + [False] * 11))
    assert "NO REGRESSION DETECTED" in text
    assert "NO REGRESSION LARGER THAN" in text
    assert "Not 'no regression'." in text


def test_a_PASS_states_the_resolution_it_passed_at():
    _, _, text = run(artifact(lever=False, correct=[True] * 10 + [False] * 10),
                     artifact(lever=True, correct=[True] * 9 + [False] * 11))
    assert "VERDICT: PASS" in text
    assert "no regression larger than" in text
    assert "a PASS is not evidence the lever is" in text


def test_two_arms_that_never_disagree_are_not_reported_as_a_clean_null():
    """Zero discordant questions has no MDE to quote, and it is also exactly
    what two runs of ONE arm look like. It must not print as the strongest
    possible pass."""
    same = [True] * 15 + [False] * 5
    v, _, text = run(artifact(lever=False, correct=same),
                     artifact(lever=True, correct=same))
    assert v == "PASS"
    assert "agreed on every scored question" in text
    assert "two runs of ONE arm" in text


def test_a_moved_answer_model_is_INCOMPLETE_and_prints_no_score():
    """A paired test assumes the arms differ in the lever only. Two answer
    models is not a footnote — it is a different experiment, and the pairing
    that makes the gate readable does not survive it."""
    v, d, text = run(
        artifact(lever=False, correct=[True] * 20,
                 cfg_extra={"answer_model": "deepseek-v4-flash"}),
        artifact(lever=True, correct=[False] * 20,
                 cfg_extra={"answer_model": "deepseek-chat"}))
    assert v == "INCOMPLETE"
    assert d["fatal"] == ["answer_model"]
    assert "REGRESSION" not in text, "no verdict may be computed"


def test_an_ordinary_confound_is_noted_but_does_not_stop_the_read():
    """Only the confounds that break PAIRING are fatal. Refusing on any
    difference at all would make the gate unrunnable, and a gate that never
    runs is not a stricter gate."""
    v, _, text = run(artifact(lever=False, correct=[True] * 20,
                              cfg_extra={"workers": 4}),
                     artifact(lever=True, correct=[True] * 20,
                              cfg_extra={"workers": 8}))
    assert v == "PASS"
    assert "workers" in text


def test_unscored_rows_are_excluded_from_the_pairing():
    """D3. A judge that never answered has no outcome to pair, and counting
    it as a miss on both arms pads the concordant cell — the one that carries
    no information but does divide the net."""
    off = artifact(lever=False, correct=[True] * 4)
    on = artifact(lever=True, correct=[True] * 4)
    off["per_question"][0]["correct"] = None
    on["per_question"][1]["correct"] = None
    _, d, _ = run(off, on)
    assert d["paired"]["OVERALL"]["n"] == 2


def test_the_MS_subset_is_only_the_multi_session_questions():
    types = ["multi-session"] * 10 + ["single-session-user"] * 10
    off = artifact(lever=False, correct=[True] * 20, qtypes=types)
    on = artifact(lever=True, correct=[True] * 20, qtypes=types)
    _, d, _ = run(off, on)
    assert d["paired"]["OVERALL"]["n"] == 20
    assert d["paired"][gs.MS_KEY]["n"] == 10


@pytest.mark.parametrize("drifts", ["off", "on"])
def test_a_question_labelled_MS_on_only_one_arm_leaves_the_subset(drifts):
    """Reading one arm's label would let the denominator move between arms,
    which is a difference in WHICH questions were scored dressed up as a
    difference in how they scored.

    Both directions, because either arm alone catches only the drift in the
    other one -- a single-direction fixture passes against a scorer that
    reads exactly one label, which is the bug."""
    mixed = ["multi-session"] * 3 + ["temporal-reasoning"]
    pure = ["multi-session"] * 4
    off = artifact(lever=False, correct=[True] * 4,
                   qtypes=mixed if drifts == "off" else pure)
    on = artifact(lever=True, correct=[True] * 4,
                  qtypes=mixed if drifts == "on" else pure)
    _, d, _ = run(off, on)
    assert d["paired"][gs.MS_KEY]["n"] == 3


def test_the_MS_subset_reports_its_coarser_resolution():
    """A null on a tenth of the questions is a much weaker claim, and the old
    absolute MS floor said nothing about that at all."""
    types = ["multi-session"] * 6 + ["single-session-user"] * 40
    off = artifact(lever=False, correct=[True] * 3 + [False] * 3 + [True] * 20 + [False] * 20,
                   qtypes=types)
    on = artifact(lever=True, correct=[False] * 3 + [True] * 3 + [False] * 20 + [True] * 20,
                  qtypes=types)
    _, _, text = run(off, on)
    assert "resolves only" in text
    assert "smaller sample of the same" in text


def test_a_question_only_one_arm_answered_is_not_paired():
    off = artifact(lever=False, correct=[True] * 5)
    on = artifact(lever=True, correct=[True] * 5)
    on["per_question"] = on["per_question"][:3]
    _, d, _ = run(off, on)
    assert d["paired"]["OVERALL"]["n"] == 3
