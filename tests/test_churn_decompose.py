"""A split that reads "the judge is fine" without ever being able to say so.

`churn_decompose` attributes each flipped question to the judge or the reader
by asking whether the two runs produced the same answer text. That inference
is only as good as the variation in its key: if no two runs ever repeat an
answer string, the module reports 0% judge-side from a comparison that could
never have returned anything else.

These tests exist mostly to keep that failure mode impossible -- the
UNAVAILABLE branch, the concordant control rate, and the parser-consistency
check are all load-bearing, and each has a test that fails if it is removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import churn_decompose as cd  # noqa: E402


def row(qid, correct, hyp, raw="yes", **kw):
    return {"question_id": qid, "correct": correct, "hypothesis": hyp,
            "judge_raw": raw, **kw}


def art(*rows):
    return {"per_question": list(rows)}


def test_same_answer_different_verdict_is_judge_side():
    d = cd.decompose(art(row("q1", True, "Paris")),
                     art(row("q1", False, "Paris", raw="no")))
    assert d["judge_side"] == 1 and d["answer_side"] == 0


def test_different_answer_different_verdict_is_answer_side():
    d = cd.decompose(art(row("q1", True, "Paris")),
                     art(row("q1", False, "Lyon", raw="no")))
    assert d["answer_side"] == 1 and d["judge_side"] == 0


def test_a_whitespace_only_difference_is_not_a_different_answer():
    """Otherwise re-wrapping inflates the expensive side of the split."""
    d = cd.decompose(art(row("q1", True, "Paris, France")),
                     art(row("q1", False, "Paris,\n  France", raw="no")))
    assert d["judge_side"] == 1
    assert d["exact_same_hypothesis"] == 0, "exact count stays exact"


def test_unscored_rows_are_excluded_not_counted_concordant():
    """D3: a judge that never answered has no verdict to flip. Counting it as
    agreement would dilute the churn rate with rows nobody judged."""
    a = art(row("q1", True, "Paris"), row("q2", None, ""),
            row("q3", True, "x", judge_error=True))
    b = art(row("q1", True, "Paris"), row("q2", None, ""),
            row("q3", False, "x", judge_error=True))
    d = cd.decompose(a, b)
    assert d["shared"] == 3 and d["unscored"] == 2 and d["n"] == 1
    assert d["discordant"] == 0


def test_no_repeated_answer_anywhere_is_UNAVAILABLE_not_zero_judge_churn():
    """The whole point of the module. Every answer differs, so the key is
    constant -- and a constant cannot explain a flip. Reporting "0%
    judge-side" here would be a fact about string equality, not the judge."""
    a = art(row("q1", True, "aaa"), row("q2", True, "bbb"))
    b = art(row("q1", False, "ccc", raw="no"), row("q2", True, "ddd"))
    d = cd.decompose(a, b)
    assert d["available"] is False
    assert d["judge_side"] == 0, "the count is zero, which is why it must "\
        "not be read as a finding"


def test_one_repeated_answer_is_enough_to_make_the_split_available():
    a = art(row("q1", True, "aaa"), row("q2", True, "same"))
    b = art(row("q1", False, "ccc", raw="no"), row("q2", True, "same"))
    d = cd.decompose(a, b)
    assert d["available"] is True
    assert d["concordant_same_hyp"] == 1


def test_the_concordant_control_rate_is_reported():
    """Without it there is nothing to compare the discordant rate against,
    and the attribution has no power statement at all."""
    a = art(row("q1", True, "same"), row("q2", True, "x"),
            row("q3", True, "keep"))
    b = art(row("q1", False, "same", raw="no"), row("q2", True, "y"),
            row("q3", True, "keep"))
    d = cd.decompose(a, b)
    assert d["identical_rate_discordant"] == 1.0
    assert d["identical_rate_concordant"] == pytest.approx(0.5)


def test_same_answer_and_same_raw_but_a_different_verdict_is_flagged():
    """`parse_judge_verdict` is pure string logic. If this ever fires the
    artifact is inconsistent and the split must not be read."""
    d = cd.decompose(art(row("q1", True, "Paris", raw="yes")),
                     art(row("q1", False, "Paris", raw="yes")))
    assert d["parser_impossible"] == ["q1"]


def test_an_honest_judge_flip_is_not_flagged_as_impossible():
    d = cd.decompose(art(row("q1", True, "Paris", raw="yes")),
                     art(row("q1", False, "Paris", raw="no")))
    assert d["parser_impossible"] == [] and d["judge_side"] == 1


def test_no_shared_questions_raises_rather_than_reporting_nothing():
    with pytest.raises(ValueError):
        cd.decompose(art(row("q1", True, "a")), art(row("q2", True, "a")))


def test_removing_judge_churn_lowers_the_mde():
    a = art(*[row(f"q{i}", True, "same") for i in range(10)])
    rows_b = [row("q0", False, "same", raw="no"),      # judge-side
              row("q1", False, "different", raw="no")]  # answer-side
    b = art(*rows_b, *[row(f"q{i}", True, "same") for i in range(2, 10)])
    d = cd.decompose(a, b)
    assert d["judge_side"] == 1 and d["answer_side"] == 1
    assert d["judge_share"] == pytest.approx(0.5)
    assert d["mde_pp_judge_free"] < d["mde_pp"]


def test_all_churn_judge_side_means_a_perfect_judge_leaves_none():
    a = art(row("q1", True, "same"), row("q2", True, "same2"))
    b = art(row("q1", False, "same", raw="no"), row("q2", True, "same2"))
    d = cd.decompose(a, b)
    assert d["mde_pp_judge_free"] is None, "no answer-side churn remains"


def test_report_on_an_unavailable_split_prints_no_attribution():
    """A reader who sees "judge-side 0" cannot un-see it. The UNAVAILABLE
    branch must RETURN, not merely warn -- the same rule guard_score follows
    for an unevidenced arm."""
    a = art(row("q1", True, "aaa"))
    b = art(row("q1", False, "ccc", raw="no"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "UNAVAILABLE" in text
    assert "judge-side" not in text.lower().replace("judge-side (", "")


def test_report_prints_the_power_check():
    a = art(row("q1", True, "same"), row("q2", True, "k"))
    b = art(row("q1", False, "same", raw="no"), row("q2", True, "k"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    text = "\n".join(lines)
    assert "power check" in text
    assert "among CONCORDANT" in text


def test_report_warns_when_the_parser_looks_non_deterministic():
    a = art(row("q1", True, "same", raw="yes"), row("q2", True, "k"))
    b = art(row("q1", False, "same", raw="yes"), row("q2", True, "k"))
    lines: list[str] = []
    cd.report(cd.decompose(a, b), out=lines.append)
    assert "not readable" in "\n".join(lines)
