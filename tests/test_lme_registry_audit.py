"""The date-check `lme_registry`'s docstring promised and did not have.

`aggregation_nodes_enabled = 0` in this ledger is three different claims
wearing one value: RECORDED by the run's own config block, ASSERTED after the
fact with `--set`, or ABSENT. Only the first is a measurement. The other two
say what the operator MEANT, and for four days the code did not follow intent:

    2026-08-26T16:26:57Z  52adfe5  library default False -> True
    2026-08-30T20:50:00Z  2247074  longmemeval_adapter pins the lever both ways

Between them the adapter set only the True leg, so an un-flagged run inherited
the new True. Any row in that window labelled 0 is contradicted by the code
that produced it.

Run against the live ledger the audit reports **zero** rows in the window --
which is the outcome that most resembles a check that isn't wired up. So the
load-bearing tests here are the ones that put a row INSIDE the window and
require it to fire. A clean audit is only worth reading if a dirty one is not.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import benchmarks.lme_registry as reg  # noqa: E402

FLIP = reg.AGGREGATION_DEFAULT_FLIP      # 2026-08-26T16:26:57Z
PIN = reg.AGGREGATION_ADAPTER_PIN        # 2026-08-30T20:50:00Z

IN_WINDOW = "2026-08-28T12:00:00Z"
PRE_FLIP = "2026-08-20T12:00:00Z"
POST_PIN = "2026-08-31T12:00:00Z"

ASSERTED = ("recorded + analyst:aggregation_nodes_enabled=0; "
            "analyst:episode_granularity_enabled=0")


def _audit(run_date, *, elapsed=0, aggr=0, epg=0, prov="recorded"):
    return reg.audit_row(run_date, elapsed, aggr, epg, prov)


def test_the_window_is_the_two_commits_and_is_not_empty():
    """If these ever collapsed to the same instant the audit would pass on
    every possible input, and nothing in its output would say so."""
    assert FLIP < PIN


# --------------------------------------------------------------- fires ----

def test_an_asserted_off_inside_the_window_is_contradicted():
    """THE case this audit exists for: an analyst's `--set ...=0` on a run the
    adapter could not have honoured. The label is not wrong about intent; it
    is wrong about what executed."""
    _, verdict, note = _audit(IN_WINDOW, aggr=0, prov=ASSERTED)
    assert verdict == "CONTRADICTED"
    assert "the layer was ON regardless of the label" in note
    assert "ASSERTED" in note


def test_a_recorded_off_inside_the_window_is_contradicted_too():
    """Provenance does not rescue it. A run whose own config block said 0 was
    still executed by an adapter that set only the True leg -- the config
    block records the REQUEST, and the request is what was ignored."""
    _, verdict, note = _audit(IN_WINDOW, aggr=0, prov="recorded")
    assert verdict == "CONTRADICTED"
    assert "RECORDED" in note


def test_an_absent_label_inside_the_window_is_contradicted_not_unknown():
    """NULL here is not missing information -- the library default supplies it.
    Reporting UNKNOWN would leave a known aggregation-ON run available as an
    OFF baseline on the grounds that nobody wrote it down."""
    _, verdict, note = _audit(IN_WINDOW, aggr=None, prov="recorded")
    assert verdict == "CONTRADICTED"
    assert "inherited the library's True" in note


def test_a_run_that_ended_after_the_pin_but_started_before_it_fires():
    """`run_date` is the END stamp. The two real guard arms clear the pin by
    26 minutes, so a version of this check keyed on the end stamp would give
    the right answer on today's ledger for the wrong reason -- and the wrong
    answer on the first run that straddles the commit."""
    start, verdict, _ = _audit("2026-08-30T21:00:00Z", elapsed=3600, aggr=0)
    assert start == "2026-08-30T20:00:00Z"
    assert verdict == "CONTRADICTED"


def test_the_granularity_lever_cannot_predate_its_own_commit():
    """A `--set episode_granularity_enabled=1` on a pre-2247074 run asserts a
    code path that did not exist."""
    _, verdict, note = _audit(PRE_FLIP, epg=1)
    assert verdict == "CONTRADICTED"
    assert "before the lever existed" in note


# ------------------------------------------------------------ does not ----

def test_pre_flip_is_clean():
    """Before 52adfe5 the library default agreed with the label."""
    _, verdict, note = _audit(PRE_FLIP, aggr=0)
    assert verdict == "OK"
    assert "pre-flip" in note


def test_post_pin_is_clean():
    _, verdict, note = _audit(POST_PIN, aggr=0)
    assert verdict == "OK"
    assert "post-pin" in note


def test_an_in_window_run_labelled_ON_is_not_a_conflict():
    """The window makes the layer ON. A row that says ON is simply right, and
    flagging it would make the audit fire on rows it has nothing against."""
    _, verdict, note = _audit(IN_WINDOW, aggr=1)
    assert verdict == "OK"
    assert "no conflict" in note


def test_a_row_with_no_date_is_unknown_not_ok():
    _, verdict, note = _audit(None, aggr=0)
    assert verdict == "UNKNOWN"
    assert "no window can be decided" in note


# ------------------------------------------------------- label_source ----

def test_label_source_separates_the_three_claims():
    col = "aggregation_nodes_enabled"
    assert reg.label_source("recorded", col, 0) == "RECORDED"
    assert reg.label_source(ASSERTED, col, 0) == "ASSERTED"
    assert reg.label_source("recorded", col, None) == "ABSENT"


def test_an_analyst_set_on_a_DIFFERENT_column_does_not_taint_this_one():
    prov = "recorded + analyst:episode_granularity_enabled=1"
    assert reg.label_source(prov, "aggregation_nodes_enabled", 0) == "RECORDED"


# ------------------------------------------------------------- report ----

def _rows_into(tmp_path, monkeypatch, rows):
    import importlib
    import sqlite3
    db = tmp_path / "runs.db"
    monkeypatch.setenv("LME_REGISTRY_DB", str(db))
    monkeypatch.setenv("LME_BENCH_DIR", str(tmp_path))
    mod = importlib.reload(reg)
    con = mod.connect()
    for i, (archive, rd, es, aggr, epg, prov) in enumerate(rows, start=1):
        con.execute(
            "INSERT INTO runs (archive, kind, run_date, elapsed_s, "
            "aggregation_nodes_enabled, episode_granularity_enabled, "
            "flags_provenance) VALUES (?,?,?,?,?,?,?)",
            (archive, "variant", rd, es, aggr, epg, prov))
    con.commit()
    return mod


def test_an_empty_window_is_reported_as_a_count_not_as_silence(tmp_path, monkeypatch, capsys):
    """The live ledger has zero rows in the window, and that reads exactly like
    an audit that was never wired up. It must say which one it is."""
    mod = _rows_into(tmp_path, monkeypatch,
                     [("a.json", PRE_FLIP, 100, 0, 0, "recorded"),
                      ("b.json", POST_PIN, 100, 0, 0, "recorded")])
    assert mod.cmd_audit() == 0
    out = capsys.readouterr().out
    assert "runs whose execution overlapped that window: 0" in out
    assert "has no victims in this ledger" in out


def test_a_contaminated_row_makes_the_command_exit_nonzero(tmp_path, monkeypatch, capsys):
    mod = _rows_into(tmp_path, monkeypatch,
                     [("clean.json", PRE_FLIP, 100, 0, 0, "recorded"),
                      ("dirty.json", IN_WINDOW, 100, 0, 0, ASSERTED)])
    assert mod.cmd_audit() == 1
    out = capsys.readouterr().out
    assert "runs whose execution overlapped that window: 1" in out
    assert "[CONTRADICTED] id=2 dirty.json" in out
    assert "must NOT be used as an aggregation-OFF baseline" in out
    assert "has no victims" not in out


def test_the_ok_rows_are_hidden_unless_asked_for(tmp_path, monkeypatch, capsys):
    """A 74-row dump of OKs buries the one line that matters."""
    rows = [("clean.json", PRE_FLIP, 100, 0, 0, "recorded")]
    mod = _rows_into(tmp_path, monkeypatch, rows)
    mod.cmd_audit()
    assert "clean.json" not in capsys.readouterr().out
    mod.cmd_audit(strict=True)
    assert "[OK] id=1 clean.json" in capsys.readouterr().out
