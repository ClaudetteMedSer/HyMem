"""Tests for benchmarks/run_registry.py stamp helpers (§6, 2026-08-31).

Contract under test:
- stem_source_date: first \\d{8}T\\d{6}Z stamp; None under policy
  'optional'; ValueError under policy 'required' (a stamp-bearing
  benchmark must always yield a stamp — a NULL there is a defect, not a
  domain fact, and would look identical to a legitimately stamp-less
  beam/locomo row).
- rejudge_dates: (source_date, run_date) for rejudge artifacts.
  source_date = first stamp of the recorded source pointer
  (rejudged_from); falls back to the own stem's first stamp only when the
  stem carries >= 2 stamps (source-stem + exec-stamp); 1-stamp stems are
  the exec stamp only -> source_date None.  run_date = last stem stamp
  (the rejudge execution time).
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import benchmarks.run_registry as rr  # noqa: E402


def test_stem_source_date_first_stamp():
    name = "longmemeval-v2-hymem-20260805T054914Z-seed0.json"
    assert rr.stem_source_date(name) == "20260805T054914Z"
    assert rr.stem_stamps(name) == ["20260805T054914Z"]


def test_stem_source_date_optional_none_on_stampless():
    for name in ("beam-v14-preference-fix.json",
                 "locomo_conv26_diag.json",
                 "longmemeval-v2-hymem-additive.json"):
        assert rr.stem_source_date(name, "optional") is None
        assert rr.stem_source_date(name) is None  # default is optional


def test_stem_source_date_required_raises_on_stampless():
    with pytest.raises(ValueError):
        rr.stem_source_date("longmemeval-v2-hymem-additive.json", "required")
    with pytest.raises(ValueError):
        rr.stem_source_date("no-stamp-at-all.json", "required")


def test_stem_source_date_required_ok_with_stamp():
    assert rr.stem_source_date("longmemeval-v2-hymem-20260602T134049Z-seed0.json",
                               "required") == "20260602T134049Z"


def test_rejudge_dates_two_stamps_from_source_pointer():
    own = "longmemeval-v2-hymem-20260610T094858Z-seed0-rejudged-deepseek-v4-flash-20260725T191314Z.json"
    src = "longmemeval-v2-hymem-20260610T094858Z-seed0.json"
    source_date, run_date = rr.rejudge_dates(own, src)
    assert source_date == "20260610T094858Z"   # source run's date
    assert run_date == "20260725T191314Z"      # rejudge exec date


def test_rejudge_dates_beam_style_source_pointer():
    own = "results_20260831T165039Z-rejudged-deepseek-chat-20260831T200531Z.json"
    src = "results_20260831T165039Z.json"
    source_date, run_date = rr.rejudge_dates(own, src, "optional")
    assert source_date == "20260831T165039Z"
    assert run_date == "20260831T200531Z"


def test_rejudge_dates_one_stamp_is_exec_only():
    # beam rejudge of a stamp-less v13-v16 source: the own stem carries ONE
    # stamp (the exec) — source_date is NOT attributable from the stem.
    own = "beam-v16-mr-tr-fix-rejudged-deepseek-v4-flash-20260831T200531Z.json"
    source_date, run_date = rr.rejudge_dates(own, None, "optional")
    assert source_date is None     # must not mislabel the exec stamp as source
    assert run_date == "20260831T200531Z"


def test_rejudge_dates_no_stamps_optional_and_required():
    assert rr.rejudge_dates("locomo_conv26_diag.rejudged.json", None) == (None, None)
    with pytest.raises(ValueError):
        rr.rejudge_dates("locomo_conv26_diag.rejudged.json", None, "required")
    with pytest.raises(ValueError):
        rr.rejudge_dates(
            "longmemeval-v2-hymem-rejudged-deepseek-v4-flash.json", None,
            "required")
