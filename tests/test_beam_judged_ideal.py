"""An artifact must record the gold the judge actually read.

Step 2's canonical (results_20260901T171245Z.json) stores `ideal_answer` — the
dataset field — while the judge had scored against the resolved `gold_text`.
Under `--judge-gold` those are different strings, and for IF/PF they are
different KINDS of thing: the compliance spec rather than a response.

The rejudge path is worse. It reparses gold fresh, judges against that, and
writes back the row's INHERITED `ideal_answer` from the source artifact. Four
rejudge arms of one anchor therefore agree on `ideal_answer` by construction,
which is why the re-derivation protocol's §4.2 "gold identity" precondition was
vacuous: it compared four copies of one field. That check is retracted in the
protocol's §10, and these tests are what stop it being written again.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import beam_adapter as ba  # noqa: E402


def test_gold_on_reads_the_gold_not_the_ideal_answer():
    assert ba.select_judge_ideal(True, "GOLD", "IDEAL") == "GOLD"


def test_gold_off_reads_the_ideal_answer():
    assert ba.select_judge_ideal(False, "GOLD", "IDEAL") == "IDEAL"


def test_the_two_differ_which_is_the_whole_point():
    """If they were the same string, recording only one would be harmless."""
    assert ba.select_judge_ideal(True, "GOLD", "IDEAL") != \
        ba.select_judge_ideal(False, "GOLD", "IDEAL")


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_gold_under_gold_on_is_empty_not_a_fallback(missing):
    """IF/PF resolve gold from compliance_spec and an ability can have none.

    The judge must then see an empty ideal — NOT silently fall back to
    `ideal_answer`, which would make a gold-on run quietly gold-off on exactly
    the rows where gold is hardest to get, and nothing in the artifact would
    say so."""
    assert ba.select_judge_ideal(True, missing, "IDEAL") == ""


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_ideal_under_gold_off_is_empty(missing):
    assert ba.select_judge_ideal(False, "GOLD", missing) == ""


def test_both_call_sites_go_through_the_helper():
    """The inline expressions are what let the two paths drift apart while
    looking identical. Neither may come back."""
    src = (_BENCH / "beam_adapter.py").read_text()
    assert src.count("select_judge_ideal(") == 4, \
        "helper + three callers: the canary, the main run, the rejudge"
    assert 'q.get("gold_text", "") if judge_gold' not in src
    assert 'gold["gold_text"] if args.judge_gold' not in src


def test_both_paths_record_what_the_judge_read():
    src = (_BENCH / "beam_adapter.py").read_text()
    assert '"judged_ideal": _judge_ideal' in src, "main run"
    assert 'r["judged_ideal"] = ideal' in src, "rejudge"


def test_the_main_run_records_its_gold_setting():
    """The rejudge path has recorded `judge_gold` since the gold-delta phase;
    the main path did not, so Step 2's canonical cannot witness its own gold
    setting from its own metadata."""
    src = (_BENCH / "beam_adapter.py").read_text()
    assert src.count('"judge_gold": bool(args.judge_gold)') == 2
