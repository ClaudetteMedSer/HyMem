"""Offline tests for the E5 gate instrument (`benchmarks/coref_eval.py`).

Same discipline as `test_episode_coverage_probe`: the instrument's pure core is
importable and pinned here, so the gate cannot silently rot into a script that
prints PASS for the wrong reason. LLM-free and box-free — the eval set is
hand-built JSON and the resolver is stdlib.

The headline assertion is that the SHIPPED eval set still clears the SHIPPED
thresholds. That makes the E5 gate a standing regression test, not a one-off
measurement: any future loosening of the trigger set that re-introduces a
false fire on a self-contained query fails the suite.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from hymem import HyMemConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from coref_eval import (  # noqa: E402
    _MAX_CONTROL_REWRITES,
    _MIN_RESOLUTION,
    _resolved,
    _run,
    _summary,
)

from hymem.query.coref import QueryRewrite  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "coref_eval_set.json"


@pytest.fixture
def spec() -> dict:
    return json.loads(EVAL_PATH.read_text())


def test_eval_set_shape(spec: dict) -> None:
    """The plan specified a ~30-item set with EN + NL and a control block. Pin
    that shape: a set that quietly shrank would make the rate meaningless."""
    items, controls = spec["items"], spec["controls"]
    assert len(items) >= 30
    assert len(controls) >= 10
    ids = [i["id"] for i in items] + [c["id"] for c in controls]
    assert len(ids) == len(set(ids)), "duplicate ids in the eval set"
    assert any(i["id"].startswith("nl-") for i in items), "no Dutch items"
    assert any(i["id"].startswith("en-") for i in items), "no English items"
    assert {i.get("kind") for i in items} == {"pronoun", "ellipsis", "demonstrative"}
    for it in items:
        assert it.get("expect"), f"{it['id']} has no acceptable referents"
        assert it.get("turns"), f"{it['id']} has no antecedent turns"
    for c in controls:
        assert "expect" not in c, f"{c['id']} is a control — it must not expect a rewrite"


def test_shipped_eval_set_clears_the_shipped_gate(spec: dict, cfg: HyMemConfig, tmp_path: Path) -> None:
    res = _run(spec, dataclasses.replace(cfg, coref_enabled=True), tmp_path)
    s = _summary(res)
    assert s["resolution_rate"] / 100.0 >= _MIN_RESOLUTION, (
        f"resolution {s['resolution_rate']:.1f}% below the pre-registered "
        f"{_MIN_RESOLUTION:.0%}"
    )
    assert s["control_rewrites"] <= _MAX_CONTROL_REWRITES, (
        f"no-harm control broken by: {s['control_offenders']}"
    )
    assert s["pass"]


def test_both_resolution_paths_are_exercised(spec: dict, cfg: HyMemConfig, tmp_path: Path) -> None:
    """A rate carried entirely by one path would be a different (weaker) result
    than the headline number, so both must be populated AND both must clear."""
    res = _run(spec, dataclasses.replace(cfg, coref_enabled=True), tmp_path)
    s = _summary(res)
    for path in ("graph", "salient"):
        sub = s["by_path"][path]
        assert sub["n"] > 0, f"no {path}-path items in the eval set"
        assert sub["resolved"] / sub["n"] >= _MIN_RESOLUTION


def test_disabled_config_yields_no_resolution(spec: dict, cfg: HyMemConfig, tmp_path: Path) -> None:
    """Plumbing check: the instrument reads the config it is handed, so a flag-off
    run must score zero rather than silently reporting the ON numbers."""
    res = _run(spec, dataclasses.replace(cfg, coref_enabled=False), tmp_path)
    s = _summary(res)
    assert s["resolved"] == 0
    assert s["control_rewrites"] == 0
    assert not s["pass"]


def test_resolved_requires_both_a_rewrite_and_a_referent() -> None:
    expect = ["medflow"]
    assert _resolved(QueryRewrite("q (context: medflow)", True, "pronoun"), expect)
    # Fired, but on the wrong referent — not a resolution.
    assert not _resolved(QueryRewrite("q (context: friday)", True, "pronoun"), expect)
    # Referent present but no rewrite fired — the referent was in the query all
    # along, which is exactly the case that must NOT be scored as a success.
    assert not _resolved(QueryRewrite("q about medflow", False, "self_contained"), expect)
