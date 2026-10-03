"""The pre-spend argv check, tested against the argv that actually failed.

A 5.5-hour guard run produced two copies of the OFF arm because the runner's
`run_arm on` consumed "on" as a label and expanded "$@" to nothing. The
pre-flight for that run had proved the two INTENDED command lines differed --
it never saw the one the script built. These tests are anchored on that exact
argv so the gap cannot reopen quietly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import assert_arm as aa  # noqa: E402

pytest.importorskip("requests")

# The argv the broken script actually built for the arm it labelled "on".
BROKEN_ON_ARGV = [
    "--scales", "S", "--sample", "0", "--seed", "0", "--workers", "8",
    "--top-k", "15", "--auto-ability", "--permissive-default",
]
FIXED_ON_ARGV = BROKEN_ON_ARGV + ["--episode-granularity"]


def test_the_argv_that_cost_the_run_is_rejected():
    ok, note = aa.check(BROKEN_ON_ARGV, "episode_granularity", True)
    assert not ok
    assert "missing from the command the runner built" in note


def test_the_corrected_argv_is_accepted():
    ok, note = aa.check(FIXED_ON_ARGV, "episode_granularity", True)
    assert ok and "True" in note


def test_the_off_arm_is_accepted_without_the_flag():
    ok, _ = aa.check(BROKEN_ON_ARGV, "episode_granularity", False)
    assert ok


def test_an_unexpected_flag_on_the_off_arm_is_rejected():
    """The mirror case: the OFF arm must not silently carry the lever."""
    ok, note = aa.check(FIXED_ON_ARGV, "episode_granularity", False)
    assert not ok
    assert "labelled False" in note


def test_a_renamed_dest_reads_STALE_not_PASS():
    """If the flag is renamed, this check must fail loudly rather than
    quietly conclude the arm is fine -- that is the vacuous-gate shape."""
    ok, note = aa.check(FIXED_ON_ARGV, "no_such_lever", True)
    assert not ok
    assert "stale" in note


def test_parsing_does_not_leak_the_monkeypatch(monkeypatch):
    """parse_adapter_argv patches argparse globally; a leak would break every
    later test in the session in a way that looks unrelated to this file.

    sys.argv is set to a SENTINEL here rather than captured from the ambient
    process. A first cut captured it, and passed against the mutation that
    removes the restore -- because an earlier test in this file had already
    leaked, so the captured "before" WAS the leaked value and comparing them
    proved nothing. Vacuous, order-dependent, and green: the exact shape this
    module exists to prevent, reproduced inside its own tests."""
    sentinel = ["SENTINEL-argv", "--untouched"]
    monkeypatch.setattr(sys, "argv", list(sentinel))
    before = argparse.ArgumentParser.parse_args
    aa.check(FIXED_ON_ARGV, "episode_granularity", True)
    assert argparse.ArgumentParser.parse_args is before
    assert sys.argv == sentinel


def test_the_patch_is_restored_even_when_the_argv_is_invalid():
    before = argparse.ArgumentParser.parse_args
    with pytest.raises(SystemExit):
        aa.check(["--not-a-real-flag"], "episode_granularity", True)
    assert argparse.ArgumentParser.parse_args is before


@pytest.mark.parametrize("s,want", [
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("no", False), ("off", False),
])
def test_coerce_expected(s, want):
    assert aa.coerce_expected(s) is want


@pytest.mark.parametrize("s", ["", "maybe", "2", "ON_ARM"])
def test_coerce_expected_rejects_garbage(s):
    """An unparseable label must raise, not default to False -- defaulting
    would make `--expect ON` (a plausible typo) assert the OFF arm."""
    with pytest.raises(ValueError):
        aa.coerce_expected(s)
