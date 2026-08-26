"""Offline tests for the flip-watch layer-state instrumentation (schema v32,
banked 2026-08-26 with the dead-watch guard, before any post-fix row existed).

Pins three things so the instrument can no longer die silently:

1. `aggregation_effective` is a real column on a fresh store (migration 032).
2. `classify()` distinguishes `layer-off` (the v32 column says the layer was
   disabled at dream start) from plain `no-agg` — so built == 0 no longer
   reads identically for "layer off" and "layer on, nothing to do".
3. `gate()` hard-FAILs on a run of >= MIN_VERDICT_ROWS consecutive
   no-agg/layer-off rows (dead-watch guard) instead of silently excluding
   them — the 118-row dead stretch of 2026-08-09..08-26 must never again
   surface as "extend the watch".

No LLM token is spent. Row dicts are assembled by hand, pre-classification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from hymem import HyMem, StubEmbeddingClient  # noqa: E402
from hymem.extraction.llm import StubLLMClient  # noqa: E402
from flipwatch_classify import (  # noqa: E402
    MIN_VERDICT_ROWS,
    REUSE_BAR,
    classify,
    gate,
)


def _row(rid: int, **kw) -> dict:
    """A classified-ready dream_runs row. Defaults to a healthy append row
    (built=100, reused=95, zero residual), overridable per test."""
    row = {
        "id": rid,
        "started_at": f"2026-08-{10 + rid:02d} 12:00:00",
        "built": 100,
        "reused": 95,
        "failures": 0,
        "input_eps": 1000 + rid,
        "blocking": "",
        "skipped_locked": False,
        "error": "",
        "effective": "enabled",
        "level0_missed": 0,
        "leaf_changed": 0,
        "predicted": 5,
        "residual": 0,
        "facts_rekey": 0,
    }
    row.update(kw)
    return row


def _appends(n: int, base_id: int = 1) -> list[dict]:
    rows = []
    for i in range(n):
        rid = base_id + i
        rows.append(
            _row(rid, input_eps=1000 + rid,
                 built=100, reused=95, residual=0, predicted=5)
        )
    return rows


def _dead(n: int, *, effective: str | None, base_id: int = 1000) -> list[dict]:
    rows = []
    for i in range(n):
        rid = base_id + i
        kw = {"effective": effective} if effective is not None else {"effective": None}
        rows.append(_row(rid, built=0, reused=0, input_eps=1000 + rid, **kw))
    return rows


def test_v32_column_exists_on_fresh_store(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    try:
        cols = {r[1] for r in hy.conn.execute("PRAGMA table_info(dream_runs)")}
        assert "aggregation_effective" in cols
    finally:
        hy.close()


def test_classify_labels_layer_off_when_column_says_disabled():
    rows = [_row(1, built=0, reused=0, effective="disabled")]
    out = classify(rows)
    assert out[0]["label"] == "layer-off"
    assert "disabled" in out[0]["note"]


def test_classify_no_agg_when_layer_enabled_but_idle():
    rows = [_row(1, built=0, reused=0, effective="enabled")]
    out = classify(rows)
    assert out[0]["label"] == "no-agg"
    assert "enabled" in out[0]["note"]


def test_classify_no_agg_for_pre_v32_unrecorded():
    rows = [_row(1, built=0, reused=0, effective=None)]
    out = classify(rows)
    assert out[0]["label"] == "no-agg"
    assert "pre-v32" in out[0]["note"]


def test_gate_hard_fails_on_interior_no_agg_streak():
    rows = _appends(3) + _dead(MIN_VERDICT_ROWS, effective=None, base_id=10) \
        + _appends(2, base_id=100)
    verdict, checks, _advisories = gate(classify(rows))
    assert verdict == "FAIL"
    assert any("dead-watch guard" in c for c in checks)


def test_gate_hard_fails_on_layer_off_streak_too():
    rows = _appends(3) + _dead(MIN_VERDICT_ROWS, effective="disabled", base_id=10) \
        + _appends(2, base_id=100)
    verdict, checks, _advisories = gate(classify(rows))
    assert verdict == "FAIL"
    assert any("dead-watch guard" in c and "disabled" in c for c in checks)


def test_gate_short_streak_is_not_a_dead_fail():
    rows = _appends(5) + _dead(2, effective=None, base_id=10)
    verdict, checks, _advisories = gate(classify(rows))
    assert not any("dead-watch guard" in c and "FAIL" in c for c in checks)


def test_gate_passes_without_streak_and_clean_rows():
    rows = _appends(MIN_VERDICT_ROWS + 1)
    verdict, checks, _advisories = gate(classify(rows))
    assert verdict == "PASS"
