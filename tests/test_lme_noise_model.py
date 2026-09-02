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
