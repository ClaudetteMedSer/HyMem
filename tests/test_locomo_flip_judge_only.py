"""The judge-only detector decides whether the dilution signature is printed.

`locomo_flip.py` exists for one signature: questions CORRECT in the baseline
going WRONG at the wider setting, with the evidence reaching the reader in both
arms. That is dilution. The same count, when B is a `--rejudge` of A, is judge
nondeterminism instead -- the reader never moved -- and the script relabels the
line accordingly.

So `judge_only` is not cosmetic: it is the difference between reporting
dilution and reporting noise, over the identical rows. A detector that cannot
come out FALSE therefore does not merely fail silently; it prints "JUDGE churn"
across the whole dilution signal.

The old form was `a.get("ai_answer") == b.get("ai_answer")`. On a pair that
records the answer under another name -- `facts_ab.py` documents that LME uses
`hypothesis`, and keeps a four-name list because of it -- every row compares
None to None, `all()` returns True, and a genuine reader-side A/B is announced
as a re-judge of itself. Absence is not agreement. It is now a third outcome.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
_FLIP = _BENCH / "locomo_flip.py"


def _row(i, *, correct, answer=None, key="ai_answer", category=1):
    r = {"id": f"q{i}", "category": category, "correct": correct,
         "question": f"question {i}", "answer": f"gold {i}",
         # Evidence reached the reader in both arms, so a regression here is
         # the reader-side one -- the row the label is argued over.
         "gold_in_pool": True, "gold_in_topk": True, "gold_in_context": True}
    if answer is not None:
        r[key] = answer
    return r


def _run(tmp_path, a_rows, b_rows, *args):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(a_rows), encoding="utf-8")
    b.write_text(json.dumps(b_rows), encoding="utf-8")
    return subprocess.run([sys.executable, str(_FLIP), str(a), str(b), *args],
                          capture_output=True, text=True)


def _pair(n=6, *, a_answer, b_answer, key="ai_answer", regress=2):
    """n rows; the first `regress` are CORRECT in A and WRONG in B."""
    a = [_row(i, correct=True, answer=a_answer(i), key=key) for i in range(n)]
    b = [_row(i, correct=i >= regress, answer=b_answer(i), key=key)
         for i in range(n)]
    return a, b


@pytest.mark.parametrize("key", ["ai_answer", "hypothesis", "prediction", "response"])
def test_identical_answers_are_judge_churn_under_every_answer_name(tmp_path, key):
    a, b = _pair(a_answer=lambda i: f"said {i}", b_answer=lambda i: f"said {i}",
                 key=key)
    r = _run(tmp_path, a, b)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[judge-only]" in r.stdout
    assert "JUDGE churn" in r.stdout
    assert "← dilution" not in r.stdout


def test_differing_answers_are_dilution_not_churn(tmp_path):
    a, b = _pair(a_answer=lambda i: f"A said {i}", b_answer=lambda i: f"B said {i}")
    r = _run(tmp_path, a, b)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[judge-only]" not in r.stdout
    assert "← dilution" in r.stdout


def test_a_pair_that_records_no_answer_is_not_called_a_rejudge(tmp_path):
    """THE test this correction exists for.

    Every row lacks a reader answer, so the old `.get("ai_answer")` comparison
    was None == None on all of them and reported a re-judge. Two independently
    read arms would then have their reader-side regressions -- the dilution
    signature, the only thing this script can see that the adapter report
    cannot -- printed as judge nondeterminism."""
    a, b = _pair(a_answer=lambda i: None, b_answer=lambda i: None)
    r = _run(tmp_path, a, b)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[judge-only]" not in r.stdout
    assert "JUDGE churn" not in r.stdout
    assert "[unclassified]" in r.stdout
    assert "6/6 shared row(s) record no reader answer" in r.stdout


def test_one_unanswered_row_is_enough_to_refuse_the_claim(tmp_path):
    """Classifying over only the rows that happen to carry an answer would
    narrow the claim without saying so: 'every shared answer is byte-identical'
    would mean 'every one I could read'."""
    a, b = _pair(a_answer=lambda i: f"said {i}",
                 b_answer=lambda i: None if i == 3 else f"said {i}")
    r = _run(tmp_path, a, b)
    assert "[unclassified]" in r.stdout
    assert "1/6 shared row(s)" in r.stdout
    assert "[judge-only]" not in r.stdout


def test_an_empty_answer_string_does_not_count_as_recorded(tmp_path):
    """A skipped reader writes "" (longmemeval_adapter's outage row does). Two
    empty strings are equal, so the bare comparison called that agreement."""
    a, b = _pair(a_answer=lambda i: "", b_answer=lambda i: "")
    r = _run(tmp_path, a, b)
    assert "[unclassified]" in r.stdout
    assert "[judge-only]" not in r.stdout


def test_unclassified_falls_back_to_the_checkable_label(tmp_path):
    """Between the two labels, DILUTION is the one an analyst can go and test
    against the run; JUDGE churn is the one that says stop looking."""
    a, b = _pair(a_answer=lambda i: None, b_answer=lambda i: None)
    r = _run(tmp_path, a, b)
    assert "← dilution" in r.stdout


def test_the_json_and_the_report_cannot_disagree(tmp_path):
    """`dilution_regressions` names rows the printed report may have just
    relabelled JUDGE churn. A consumer reading only the JSON had no way to
    know that."""
    a, b = _pair(a_answer=lambda i: f"said {i}", b_answer=lambda i: f"said {i}")
    r = _run(tmp_path, a, b, "--json")
    payload = json.loads(r.stdout[r.stdout.index("{"):])
    assert payload["judge_only"] is True
    assert payload["unanswered"] == 0
    assert payload["dilution_regressions"], "rows exist and are labelled churn"

    a, b = _pair(a_answer=lambda i: None, b_answer=lambda i: None)
    r = _run(tmp_path, a, b, "--json")
    payload = json.loads(r.stdout[r.stdout.index("{"):])
    assert payload["judge_only"] is False
    assert payload["unanswered"] == 6


def test_the_listing_reads_the_answer_under_its_own_name(tmp_path):
    """`--list` exists so the analyst can see WHY a question moved, which is
    only visible in the two answers side by side. Keyed on one field name it
    prints a column of `None` on any other adapter's file."""
    a, b = _pair(a_answer=lambda i: f"A said {i}", b_answer=lambda i: f"B said {i}",
                 key="hypothesis")
    r = _run(tmp_path, a, b, "--list")
    assert "B answered: B said 0" in r.stdout
    assert "B answered: None" not in r.stdout


def test_the_bare_field_comparison_does_not_come_back():
    """Asserted from source: the defect is one expression, and it reads as an
    obvious simplification of the guard that replaced it."""
    src = _FLIP.read_text(encoding="utf-8")
    assert 'a_rows[i].get("ai_answer") == b_rows[i].get("ai_answer")' not in src
    assert "answer_text(a_rows[i]) is None" in src
