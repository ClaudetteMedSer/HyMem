"""Offline tests for the E3 measurement harness (`benchmarks/rerank_ab.py`).

No model download, no network, no LLM. Pinned here:

  * the STATS — Spearman against hand-computed values, percentiles, and the
    gold-rank rule that a dropped gold turn must not flatter an arm's median;
  * the GATE arithmetic for M1 (parity, not superiority) and M2 (Dutch gain with
    a bounded English regression), including the direction of every comparison —
    a sign error here would adopt a worse reranker;
  * the hand-set itself: every question's gold must actually be a verbatim
    substring of its block's corpus, and it must reach the BM25 pool. A hand-set
    whose gold is unreachable measures nothing, silently.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from rerank_ab import (  # noqa: E402
    _M2_MAX_EN_REGRESSION,
    _gold_rank,
    _pctile,
    arm_stats,
    gate_m1,
    gate_m2,
    handset_pools,
    rank_agreement,
    run_arm,
    sim_rerank,
    spearman,
)

HANDSET = Path(__file__).resolve().parent.parent / "benchmarks" / "rerank_handset.json"


@dataclass
class Hit:
    """Minimal stand-in for MessageHit — the reranker contract is `.text`,
    `.score`, `.score_kind` and nothing else."""
    text: str
    score: float = 0.0
    score_kind: str = "bm25"


# ── Stats ───────────────────────────────────────────────────────────────────

def test_spearman_known_values() -> None:
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Undefined cases return None rather than a misleading 0.0.
    assert spearman([1, 2], [1, 2]) is None
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert spearman([1, 2, 3], [1, 2]) is None


def test_spearman_handles_ties_with_average_ranks() -> None:
    rho = spearman([1, 2, 2, 3], [1, 2, 2, 3])
    assert rho == pytest.approx(1.0)


def test_percentiles() -> None:
    assert _pctile([], 95) == 0.0
    assert _pctile([5.0], 95) == 5.0
    assert _pctile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


def test_gold_rank_is_one_indexed_and_none_when_absent() -> None:
    ranked = [Hit("nothing here"), Hit("the pool was raised to forty"), Hit("more")]
    assert _gold_rank(["raised to forty"], ranked) == 2
    assert _gold_rank(["kangaroo taxonomy"], ranked) is None


def test_dropped_gold_does_not_flatter_the_median() -> None:
    """An arm that returns gold only when it ranks first must NOT post a better
    median than one that returns it at rank 3 every time — the ≤15 share and the
    missing count are what expose it."""
    good = {"arm": "ce", "model": None, "ranks": [3, 3, 3, 3],
            "latencies": [1.0], "orders": [], "tokens": 0}
    cherry = {"arm": "llm", "model": None, "ranks": [1, None, None, None],
              "latencies": [1.0], "orders": [], "tokens": 0}
    g, c = arm_stats(good, top_k=15), arm_stats(cherry, top_k=15)
    assert c["median_rank"] < g["median_rank"]      # the trap: median looks better
    assert c["top15_share"] < g["top15_share"]      # the share exposes it
    assert c["missing"] == 3 and g["missing"] == 0


def test_rank_agreement_reports_how_many_questions_it_scored() -> None:
    a = {"orders": [[0, 1, 2, 3], [0, 1, 2, 3]]}
    b = {"orders": [[0, 1, 2, 3], [3, 2, 1, 0]]}
    rho, n = rank_agreement(a, b)
    assert n == 2
    assert rho == pytest.approx(0.0)  # median of (+1, -1)
    # Non-overlapping outputs are unmeasurable, and that is reported as such.
    rho2, n2 = rank_agreement({"orders": [[0, 1, 2]]}, {"orders": [[7, 8, 9]]})
    assert (rho2, n2) == (None, 0)


# ── Gates ───────────────────────────────────────────────────────────────────

def _stats(median: float, top15: float, p95: float) -> dict:
    return {"median_rank": median, "top15_share": top15, "p95_ms": p95}


def test_m1_gate_is_parity_not_superiority() -> None:
    llm = _stats(3.0, 80.0, 1200.0)
    # CE one position worse, 1pp lower share, 60× faster → adoptable.
    assert gate_m1(_stats(4.0, 79.0, 20.0), llm)["pass"]
    # Two positions worse → not parity.
    assert not gate_m1(_stats(5.0, 80.0, 20.0), llm)["pass"]
    # Share 3pp down → not parity.
    assert not gate_m1(_stats(3.0, 77.0, 20.0), llm)["pass"]
    # Ranks fine but only 5× faster → the CE's whole case is latency, so no.
    assert not gate_m1(_stats(3.0, 80.0, 240.0), llm)["pass"]
    # Better on rank AND fast → obviously adoptable.
    assert gate_m1(_stats(1.0, 95.0, 10.0), llm)["pass"]


def test_m1_speedup_is_llm_over_ce() -> None:
    g = gate_m1(_stats(3.0, 80.0, 10.0), _stats(3.0, 80.0, 1000.0))
    assert g["speedup"] == pytest.approx(100.0)


def test_m1_zero_latency_does_not_crash_or_pass_silently() -> None:
    g = gate_m1(_stats(3.0, 80.0, 0.0), _stats(3.0, 80.0, 1000.0))
    assert g["speedup"] is None
    assert not g["checks"]["latency_10x_better"]


def test_m2_gate_needs_dutch_gain_and_bounded_english_cost() -> None:
    nl_mx, en_mx = _stats(6.0, 60.0, 20.0), _stats(2.0, 90.0, 20.0)
    # Dutch improves, English holds → adopt.
    assert gate_m2(nl_mx, _stats(3.0, 75.0, 20.0), en_mx, _stats(2.0, 90.0, 20.0))["pass"]
    # Dutch improves but English regresses by more than the tolerance → no.
    assert not gate_m2(nl_mx, _stats(3.0, 75.0, 20.0), en_mx,
                       _stats(2.0 + _M2_MAX_EN_REGRESSION + 1, 85.0, 20.0))["pass"]
    # Dutch gets worse → no, however good English is.
    assert not gate_m2(nl_mx, _stats(8.0, 50.0, 20.0), en_mx, _stats(1.0, 95.0, 20.0))["pass"]


# ── The hand-set is a valid instrument ──────────────────────────────────────

@pytest.fixture
def handset() -> dict:
    return json.loads(HANDSET.read_text())


def test_handset_shape(handset: dict) -> None:
    for lang in ("nl", "en"):
        block = handset[lang]
        assert len(block["corpus"]) >= 30, f"{lang} corpus too small to rank in"
        assert len(block["items"]) >= 12, f"{lang} has too few questions"
        ids = [i["id"] for i in block["items"]]
        assert len(ids) == len(set(ids))


def test_every_gold_is_verbatim_in_its_corpus(handset: dict) -> None:
    """A gold string that is not actually in the corpus can never be found, and
    the arm would silently score 0 for a reason that has nothing to do with
    reranking."""
    for lang in ("nl", "en"):
        block = handset[lang]
        corpus = " \n ".join(text for _role, text in block["corpus"]).lower()
        for it in block["items"]:
            assert it["gold"].lower() in corpus, f"{it['id']}: gold not in corpus"


@pytest.mark.parametrize("lang", ["nl", "en"])
def test_every_gold_reaches_the_bm25_pool(handset: dict, lang: str, tmp_path: Path) -> None:
    """The measurement is about ORDERING, so the gold must be inside the pool the
    arms are handed. If it is not, the question measures retrieval, not rerank."""
    rows = handset_pools(handset[lang], pool=40, root=tmp_path / lang)
    unreachable = [r["id"] for r in rows if _gold_rank(r["gold"], r["pool"]) is None]
    assert not unreachable, f"gold outside the pool for: {unreachable}"


# ── Plumbing: the --sim arms ────────────────────────────────────────────────

def test_sim_arms_are_deterministic_and_tagged() -> None:
    pool = [Hit("alpha beta"), Hit("gamma"), Hit("alpha beta gamma delta")]
    a = sim_rerank("alpha beta gamma", pool, top_k=3, arm="ce")
    b = sim_rerank("alpha beta gamma", pool, top_k=3, arm="ce")
    assert [h.text for h in a] == [h.text for h in b]
    assert a[0].text == "alpha beta gamma delta"   # highest overlap first
    assert all(h.score_kind == "reranked" for h in a)
    rev = sim_rerank("x", pool, top_k=3, arm="llm")
    assert [h.text for h in rev] == [h.text for h in reversed(pool)]


def test_run_arm_respects_top_k_and_records_latency() -> None:
    rows = [{"id": "r1", "question": "alpha beta",
             "gold": ["gamma"],
             "pool": [Hit("alpha"), Hit("beta"), Hit("gamma"), Hit("delta")]}]
    res = run_arm("ce", rows, top_k=2, sim=True)
    assert len(res["orders"][0]) == 2
    assert len(res["latencies"]) == 1
    s = arm_stats(res, top_k=2)
    assert s["n"] == 1


def test_run_arm_tolerates_an_empty_pool() -> None:
    rows = [{"id": "r1", "question": "q", "gold": ["g"], "pool": []}]
    res = run_arm("ce", rows, top_k=5, sim=True)
    assert res["ranks"] == [None]
    assert arm_stats(res, top_k=5)["missing"] == 1
