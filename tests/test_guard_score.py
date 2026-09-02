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
             correct_from=0):
    """One arm. `lever=None` omits the key entirely (the pre-6543ee6 shape)."""
    cfg = {"scale": "S", "sample": 0, "workers": 8}
    if lever is not None:
        cfg[gs.LEVER] = lever
    pq = []
    for i in range(n):
        row = {"question_id": f"q{i}", "correct": i >= correct_from,
               "question_type": "multi-session"}
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


def test_an_overall_below_the_canonical_fails():
    v, _, _ = run(artifact(lever=False), artifact(lever=True, overall=69.0))
    assert v == "FAIL"


def test_an_ms_below_the_floor_fails():
    v, _, _ = run(artifact(lever=False), artifact(lever=True, ms=50.0))
    assert v == "FAIL"


def test_the_bar_is_read_on_the_ON_arm_not_the_OFF_arm():
    """A regression on ON must fail even when OFF is healthy."""
    v, _, _ = run(artifact(lever=False, overall=75.0),
                  artifact(lever=True, overall=60.0))
    assert v == "FAIL"


def test_the_arms_are_oriented_by_the_config_block_not_by_argv_order():
    """Passing ON first must not silently score the OFF arm against the bar."""
    v1, _, _ = run(artifact(lever=False, overall=75.0),
                   artifact(lever=True, overall=60.0))
    v2, _, _ = run(artifact(lever=True, overall=60.0),
                   artifact(lever=False, overall=75.0))
    assert v1 == v2 == "FAIL"


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
