"""The resolution of the LME harness, and whether a bar respects it.

Gate 4 was pre-registered as `OVERALL >= 70.0`, and 70.0 came from a single
2026-06-10 run on deepseek-chat -- a model hard-deprecated 2026-07-24. This
module estimates what the harness can actually resolve and how a bar behaves
against a change that does nothing. The tests below exist mostly to stop the
arithmetic drifting, and one of them to keep the warning honest: a bar at the
centre of its own distribution must be reported as mis-calibrated, because
that is the case that reads PASS/FAIL at random.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import lme_noise_model as nm  # noqa: E402


def run(correct_by_id):
    return {"per_question": [{"question_id": q, "correct": c}
                             for q, c in correct_by_id.items()]}


def test_two_identical_runs_have_no_discordance():
    a = run({"q0": True, "q1": False, "q2": True})
    d = nm.paired_discordance(a, a)
    assert d["discordant"] == 0 and d["net_pp"] == 0.0
    assert d["mde_pp"] == 0.0, "with no discordance there is nothing to resolve"


def test_the_discordant_cells_are_counted_separately():
    a = run({"q0": True, "q1": False, "q2": True, "q3": False})
    b = run({"q0": False, "q1": True, "q2": True, "q3": False})
    d = nm.paired_discordance(a, b)
    assert (d["a_only"], d["b_only"], d["discordant"]) == (1, 1, 2)
    assert d["net_pp"] == 0.0


def test_the_net_is_signed_towards_the_second_run():
    a = run({"q0": True, "q1": True, "q2": False, "q3": False})
    b = run({"q0": True, "q1": True, "q2": True, "q3": True})
    d = nm.paired_discordance(a, b)
    assert d["b_only"] == 2 and d["a_only"] == 0
    assert d["net_pp"] == pytest.approx(50.0)


def test_a_run_pairs_only_on_shared_questions():
    a = run({"q0": True, "q1": True})
    b = run({"q1": False, "q2": True})
    d = nm.paired_discordance(a, b)
    assert d["n"] == 1 and d["a_only"] == 1


def test_no_shared_questions_raises_rather_than_returning_zero():
    """A zero discordance on an empty intersection would read as a perfect
    null -- the vacuous-agreement shape."""
    with pytest.raises(ValueError):
        nm.paired_discordance(run({"q0": True}), run({"q9": True}))


def test_the_single_run_sd_is_the_pair_sd_over_root_two():
    a = run({f"q{i}": i % 2 == 0 for i in range(100)})
    b = run({f"q{i}": i % 3 == 0 for i in range(100)})
    d = nm.paired_discordance(a, b)
    assert d["sd_run_pp"] == pytest.approx(d["sd_diff_pp"] / math.sqrt(2))


def test_the_mde_scales_with_the_root_of_the_discordant_count():
    """Resolution is set by the discordant cells, not by n -- doubling the
    sample while doubling the churn buys nothing."""
    a = run({f"q{i}": True for i in range(100)})
    b = run({f"q{i}": i >= 10 for i in range(100)})
    d = nm.paired_discordance(a, b)
    assert d["discordant"] == 10
    assert d["mde_pp"] == pytest.approx(100 * nm.Z95 * math.sqrt(10) / 100)


# ------------------------------------------------------------------ the bar

def test_a_bar_at_the_mean_fails_an_inert_arm_half_the_time():
    assert nm.bar_risk(70.0, 70.0, 1.0) == pytest.approx(0.5)


def test_a_bar_well_below_the_mean_is_a_real_floor():
    assert nm.bar_risk(66.0, 70.0, 1.0) < 0.01


def test_a_bar_above_the_mean_fails_more_often_than_not():
    assert nm.bar_risk(71.0, 70.0, 1.0) > 0.5


def test_a_central_bar_is_reported_as_MISCALIBRATED():
    lines = []
    nm.report(None, [69.0, 70.0, 71.0, 70.0, 70.0], 70.0, out=lines.append)
    text = "\n".join(lines)
    assert "MIS-CALIBRATED" in text


def test_a_genuine_floor_is_not_reported_as_miscalibrated():
    lines = []
    nm.report(None, [69.0, 70.0, 71.0, 70.0, 70.0], 64.0, out=lines.append)
    assert "MIS-CALIBRATED" not in "\n".join(lines)


def test_a_single_era_run_reports_no_spread_rather_than_zero():
    """One run has no SD. Reporting 0 would make every bar look perfect."""
    lines = []
    res = nm.report(None, [70.0], 70.0, out=lines.append)
    assert res["era"]["sd"] is None
    assert "no spread" in "\n".join(lines)
    assert "MIS-CALIBRATED" not in "\n".join(lines)


# ------------------------------------------------------------ McNemar, exact

def test_a_perfectly_split_discordance_is_the_null():
    assert nm.mcnemar_exact_p(21, 21) == pytest.approx(1.0)


def test_no_discordance_at_all_is_p_one_not_p_zero():
    """Two arms that never disagree are not significantly different. Reading
    0/0 as a rejection would make an identical pair the strongest result the
    test can produce."""
    assert nm.mcnemar_exact_p(0, 0) == 1.0


def test_a_lopsided_discordance_rejects():
    assert nm.mcnemar_exact_p(30, 12) == pytest.approx(0.00785, abs=1e-4)
    assert nm.mcnemar_exact_p(35, 7) < 0.001


def test_the_test_is_two_sided():
    """Either tail rejects. The DIRECTION is guard_score's to check; a
    one-sided p here would silently halve the alpha of every other caller."""
    assert nm.mcnemar_exact_p(30, 12) == nm.mcnemar_exact_p(12, 30)


def test_a_single_discordant_question_cannot_reject():
    """b=1, c=0 is one coin flip. If this ever returned < .05 the gate would
    fire on any arm that moved exactly one question."""
    assert nm.mcnemar_exact_p(1, 0) == 1.0
    assert nm.mcnemar_exact_p(4, 0) > 0.05


def test_the_smallest_rejecting_shutout_is_five_questions():
    """2 * 0.5^5 = 0.0625 > .05; 2 * 0.5^6 = 0.03125. So five is not enough
    and six is -- pinned because it is the floor on what the gate can see
    when every discordant question moves the same way."""
    assert nm.mcnemar_exact_p(5, 0) > 0.05
    assert nm.mcnemar_exact_p(6, 0) < 0.05


def test_p_never_exceeds_one():
    """2 * tail overshoots at small n if it is not clamped."""
    for b in range(6):
        for c in range(6):
            assert 0.0 <= nm.mcnemar_exact_p(b, c) <= 1.0


def test_the_concordant_count_does_not_enter_the_p_value():
    """The signature cannot even accept it, which is the point: n=500 buys no
    resolution the discordant cells do not already contain."""
    import inspect
    assert list(inspect.signature(nm.mcnemar_exact_p).parameters) == ["b", "c"]
