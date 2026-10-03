"""Measuring f without paying for the reader — and without over-reading it.

`f` decides whether gate 4 is worth buying: at 25% a subset-scored gate
resolves twice as finely, near 100% it resolves no better and the gate should
be retired rather than re-run. So the number has to be right, and it has to
stay a statement about RETRIEVAL. The two ways it could quietly stop being
one, both tested here:

  * measured off a pair that cannot say which arm is which, where it becomes
    the fraction retrieval CHURN moves -- a different number with the same
    shape;
  * measured off the count fields when `context_sha` is missing, which
    saturate at the retrieval cap on ~84% of questions and would read f as
    near zero for a lever that moved everything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import fired_fraction as ff  # noqa: E402


def artifact(shas, *, lever=False, retrieval_only=True, cost=None, extra=None):
    cfg = {ff.LEVER: lever, "retrieval_only": retrieval_only}
    cfg.update(extra or {})
    art = {
        "config": cfg,
        "per_question": [
            {"question_id": f"q{i}", "correct": None,
             "retrieval_only": retrieval_only,
             **({"context_sha": s} if s is not None else {})}
            for i, s in enumerate(shas)],
    }
    if cost is not None:
        art["retrieval_cost"] = cost
    return art


def run(a, b):
    lines: list[str] = []
    res = ff.report(a, b, out=lines.append)
    return res, "\n".join(lines)


# ------------------------------------------------------------- the refusals

def test_a_full_run_is_REFUSED():
    """Mixing a cheap run with an expensive one invites quoting the cheap
    number as though the cheap run produced it."""
    a = artifact(["x", "y"], lever=False)
    b = artifact(["x", "z"], lever=True, retrieval_only=False)
    res, text = run(a, b)
    assert res["refused"] and res["arm"] == "B"
    assert "not a --retrieval-only run" in text
    assert "%" not in text, "no fraction may be printed"


def test_a_pair_that_cannot_evidence_its_arms_is_REFUSED():
    """Otherwise f is the fraction retrieval churn moves, which on the
    2026-09-02 same-arm pair was 34% — a plausible-looking answer to a
    question nobody asked."""
    a = artifact(["x", "y"], lever=False)
    b = artifact(["x", "z"], lever=False)
    res, text = run(a, b)
    assert res["refused"]
    assert "retrieval churn" in text


# --------------------------------------------------------------- the number

def test_f_is_the_share_of_prompts_that_moved():
    a = artifact(["a", "b", "c", "d"], lever=False)
    b = artifact(["a", "B", "C", "d"], lever=True)
    res, _ = run(a, b)
    assert res["fired"]["moved"] == 2
    assert res["fired"]["f"] == pytest.approx(0.5)


def test_f_carries_an_interval():
    """The decision turns on which side of ~25% f falls, and 2 of 4 is not
    the same evidence as 250 of 500."""
    small, _ = run(artifact(["a", "b", "c", "d"], lever=False),
                   artifact(["a", "B", "C", "d"], lever=True))
    big, _ = run(artifact([f"x{i}" for i in range(400)], lever=False),
                 artifact([f"x{i}" if i % 2 else f"y{i}" for i in range(400)],
                          lever=True))
    s_lo, s_hi = small["fired"]["ci95"]
    b_lo, b_hi = big["fired"]["ci95"]
    assert (s_hi - s_lo) > (b_hi - b_lo)


def test_a_run_without_context_sha_is_UNAVAILABLE_not_fallen_back():
    """The count fields saturate at the retrieval cap on ~84% of questions,
    so falling back to them would read f as near zero for a lever that moved
    every prompt."""
    a = artifact([None, None], lever=False)
    b = artifact([None, None], lever=True)
    res, text = run(a, b)
    assert res["fired"]["available"] is False
    assert "UNAVAILABLE" in text
    assert "not a fallback" in text


def test_no_shared_questions_raises():
    a = artifact(["a"], lever=False)
    b = artifact(["b"], lever=True)
    b["per_question"][0]["question_id"] = "other"
    with pytest.raises(ValueError):
        ff.fired_fraction(a, b)


# ------------------------------------------------------- reading it honestly

def test_a_narrow_lever_is_reported_as_worth_concentrating():
    a = artifact([f"x{i}" for i in range(100)], lever=False)
    b = artifact([f"y{i}" if i < 10 else f"x{i}" for i in range(100)],
                 lever=True)
    _, text = run(a, b)
    assert "NARROW" in text
    assert "3.16x" in text


def test_a_broad_lever_is_reported_as_an_argument_for_RETIRING_gate_4():
    """The result this whole line of work was set up to be able to reach."""
    a = artifact([f"x{i}" for i in range(100)], lever=False)
    b = artifact([f"y{i}" for i in range(100)], lever=True)
    _, text = run(a, b)
    assert "BROAD" in text
    assert "retiring it, not for re-running it" in text


def test_a_lever_that_moves_nothing_is_not_reported_as_infinite_gain():
    """f=0 makes 1/sqrt(f) undefined, and the honest reading is that the two
    arms are the same experiment — not that the gate became perfect."""
    a = artifact([f"x{i}" for i in range(10)], lever=False)
    b = artifact([f"x{i}" for i in range(10)], lever=True)
    res, text = run(a, b)
    assert res["fired"]["f"] == 0.0
    assert "the arms are the same experiment" in text
    assert "gain" not in text.split("=== what that means")[1].split("\n")[1]


def test_f_is_never_reported_as_evidence_of_an_effect():
    """A moved prompt is not a moved answer, still less a moved verdict."""
    a = artifact([f"x{i}" for i in range(10)], lever=False)
    b = artifact([f"y{i}" if i < 5 else f"x{i}" for i in range(10)], lever=True)
    _, text = run(a, b)
    assert "not evidence" in text and "changes any answer" in text


# --------------------------------------------------------------- the cost

def test_the_measured_cost_is_reported_from_the_artifacts():
    c = {"llm_calls": 12, "distill_calls": 12, "answer_calls": 0,
         "judge_calls": 0}
    a = artifact(["a", "b"], lever=False, cost=c)
    b = artifact(["a", "c"], lever=True, cost=c)
    res, text = run(a, b)
    assert res["cost"]["llm_calls"] == 24
    assert res["cost"]["answer_calls"] == 0
    assert "judge calls: 0" in text


def test_an_artifact_without_a_cost_block_says_so_rather_than_reporting_zero():
    """"0 calls" from a run that never counted them is the vacuity this repo
    keeps finding: an instrument that never met the surface it certifies."""
    a = artifact(["a", "b"], lever=False)
    b = artifact(["a", "c"], lever=True)
    _, text = run(a, b)
    assert "NOT RECORDED" in text


def test_the_interval_is_the_wilson_score_interval():
    """Pinned against the published value for 50/100 at 95%. A width that
    merely LOOKS narrower for bigger n can still be wrong in the n it uses;
    the earlier test compared two sizes and passed against a formula with the
    sample size hardcoded."""
    lo, hi = ff._wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=5e-4)
    assert hi == pytest.approx(0.5962, abs=5e-4)


def test_the_interval_narrows_as_the_root_of_n():
    """Ten times the questions, about a third the width."""
    w100 = ff._wilson(50, 100)
    w1000 = ff._wilson(500, 1000)
    assert (w1000[1] - w1000[0]) == pytest.approx(
        (w100[1] - w100[0]) / (10 ** 0.5), rel=0.05)


def test_the_report_says_the_mode_does_not_skip_the_dream():
    """The correction to the pitch that motivated this module. Retrieval-only
    saves the reader and judge, ~7% of a run; dreaming is the other ~93% and
    a dream-time lever cannot skip it. Anyone reading a cheap-looking number
    here has to see that on the same page."""
    a = artifact(["a", "b"], lever=False, cost={"llm_calls": 0,
                                                "distill_calls": 0,
                                                "answer_calls": 0,
                                                "judge_calls": 0})
    b = artifact(["a", "c"], lever=True, cost={"llm_calls": 0,
                                               "distill_calls": 0,
                                               "answer_calls": 0,
                                               "judge_calls": 0})
    _, text = run(a, b)
    assert "does not skip the DREAM" in text
    assert "93%" in text
