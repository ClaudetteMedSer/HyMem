#!/usr/bin/env python3
"""E3 — the two reranker measurements, offline, decision data only.

Campaign E, Step 3 (`additional_planning.md` §Campaign E). Review constraint 4 is
the whole design of this script: **E3 is two changes, not one**, and conflating
them is how a rebaseline becomes unattributable.

  **M1 — backend.** Cross-encoder (`mixedbread-ai/mxbai-rerank-base-v1`, the
  shipping `rerank_cross_encoder_model`) vs the LLM reranker (`rerank_model="llm"`,
  today's default). Same query, same candidate pool, two orderings. The question
  is PARITY, not superiority: the CE's case is latency and token cost, so it only
  has to rank as well.

  **M2 — model.** mxbai → `bge-reranker-v2-m3`. mxbai is English-first; HyMem's
  scope is Latin-script with Dutch prioritized, so the shipping model is a known
  mismatch for half the target languages. Measured on a Dutch hand-set with an
  English control, because a multilingual swap that quietly costs English rank is
  not an upgrade.

Neither measurement touches a frozen baseline, and NOTHING here flips a default —
adoption is Step 6, one deliberate rebaseline after Step 5's scored runs. This
script only produces the numbers that decision reads.

── Pre-registered gates ────────────────────────────────────────────────────────
  M1 (CE is adoptable): median gold rank within 1 position of the LLM arm
      AND ≤15 share within 2pp AND p95 latency ≥10× better.
  M2 (bge replaces mxbai): bge ≥ mxbai on the Dutch set (median gold rank no
      worse) AND English median-rank regression ≤1 position.

── What the pool is, and why it is honest anyway ───────────────────────────────
The candidate pool is a wide BM25 sweep of the raw-message tier
(`_message_fts_search` at `--pool`), the tier the reranker actually reorders and
the dominant recovery source. It is NOT the full fused pool: with no dream there
are no chunks, and the vector path needs an embedding server. That biases the
ABSOLUTE ranks (the `gold_rank_probe.py` caveat applies verbatim) but not the
comparison, because both arms of every measurement are handed the IDENTICAL pool
object. Read these numbers as "how well does backend X order this pool", never as
"what rank does production give this gold turn".

Latency is measured around the rerank call ONLY — that is the unit the flip
changes; ingest and BM25 are shared by both arms.

── Usage (from benchmarks/) ────────────────────────────────────────────────────
  # plumbing, no LLM, no model download, no network:
  python rerank_ab.py --sim

  # M1 on the English hand-set (CE local, LLM over the API):
  python rerank_ab.py --m1 --api-key $DEEPSEEK_API_KEY
  # M1 on an LME MS slice instead of the hand-set:
  python rerank_ab.py --m1 --dataset lme_s.json --category multi-session --sample 40 \
      --api-key $DEEPSEEK_API_KEY
  # M2 (no LLM needed — both arms are local cross-encoders):
  python rerank_ab.py --m2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longmemeval_adapter import (  # noqa: E402
    _extract_gold_turns,
    _gold_in_pool,
    load_longmemeval_data,
)

from hymem.query.augment import MessageHit, _message_fts_search  # noqa: E402
from hymem.query.rerank import (  # noqa: E402
    _get_cross_encoder,
    cross_encoder_rerank,
    llm_rerank,
)

# Shipping default and the multilingual candidate. Both are read from here rather
# than from cfg so a run pins what it measured even if the default moves under it.
MXBAI = "mixedbread-ai/mxbai-rerank-base-v1"
BGE_M3 = "BAAI/bge-reranker-v2-m3"

# Pre-registered gate thresholds (see module docstring). Constants, not flags.
_M1_MEDIAN_TOLERANCE = 1      # positions
_M1_TOP15_TOLERANCE_PP = 2.0  # percentage points
_M1_MIN_SPEEDUP = 10.0        # p95 latency ratio, LLM / CE
_M2_MAX_EN_REGRESSION = 1     # positions

_DEFAULT_POOL = 40
_DEFAULT_TOP_K = 15


# ── Candidate pools ─────────────────────────────────────────────────────────

def _pool_from_store(conn, question: str, pool: int) -> list[MessageHit]:
    return _message_fts_search(conn, question, top_k=pool)


def handset_pools(block: dict, *, pool: int, root: Path) -> list[dict]:
    """Ingest the block's shared corpus once, then pull one pool per question.

    One store for the whole block is the point: every question competes against
    the same haystack, so a rank is a ranking result and not an artifact of how
    much text happened to be indexed.
    """
    from hymem import HyMem, HyMemConfig
    from hymem.extraction.llm import StubLLMClient

    hy = HyMem(HyMemConfig(root=root), llm=StubLLMClient(default="[]"))
    try:
        hy.log_messages("handset", [(role, text, None)
                                    for role, text in block["corpus"]])
        out = []
        for it in block["items"]:
            cands = _pool_from_store(hy.conn, it["question"], pool)
            out.append({"id": it["id"], "question": it["question"],
                        "gold": [it["gold"]], "pool": cands})
        return out
    finally:
        hy.close()


def lme_pools(questions: list[dict], *, pool: int, root: Path,
              limit: int = 0) -> list[dict]:
    """One temp store per LME question (its haystack is question-specific), BM25
    pool from the raw-message tier. LLM-free: `messages_fts` is built at ingest,
    so no dream and no model is involved in producing the pool."""
    from hymem import HyMem, HyMemConfig
    from hymem.extraction.llm import StubLLMClient

    out = []
    for n, q in enumerate(questions if not limit else questions[:limit], 1):
        gold, _ = _extract_gold_turns(q)
        if not gold:
            continue
        qroot = root / f"q{n}"
        qroot.mkdir(parents=True, exist_ok=True)
        hy = HyMem(HyMemConfig(root=qroot), llm=StubLLMClient(default="[]"))
        try:
            sessions = q.get("haystack_sessions", []) or []
            sids = q.get("haystack_session_ids",
                         [str(i) for i in range(len(sessions))])
            for sid, messages in zip(sids, sessions):
                entries = [(m.get("role", "user"), m.get("content", ""), None)
                           for m in messages
                           if isinstance(m, dict) and (m.get("content") or "").strip()]
                if entries:
                    hy.log_messages(sid, entries)
            cands = _pool_from_store(hy.conn, q["question"], pool)
        finally:
            hy.close()
        out.append({"id": q.get("question_id"), "question": q["question"],
                    "gold": gold, "pool": cands})
        if n % 10 == 0:
            print(f"  ── pooled {n} questions", flush=True)
    return out


# ── Arms ────────────────────────────────────────────────────────────────────

def sim_rerank(query: str, candidates: list, *, top_k: int, arm: str) -> list:
    """Deterministic fake reranker for `--sim`.

    "ce" sorts by query-token overlap (a plausible relevance proxy); "llm"
    reverses the pool. Both are pure and stable, so the plumbing — pool → arm →
    gold rank → gate arithmetic — is exercisable with no model, no network, and no
    download. Its numbers are meaningless as evidence and the report says so."""
    from dataclasses import replace

    if arm == "llm":
        ordered = list(reversed(candidates))
    else:
        q_tokens = set(query.lower().split())
        ordered = sorted(
            candidates,
            key=lambda h: -len(q_tokens & set(h.text.lower().split())),
        )
    return [replace(h, score=float(-i), score_kind="reranked")
            for i, h in enumerate(ordered[:top_k])]


def run_arm(arm: str, rows: list[dict], *, top_k: int, llm=None,
            ce_model: str = MXBAI, sim: bool = False) -> dict:
    """Rerank every row's pool with one arm; return ranks + latency + token cost."""
    ranks: list[int | None] = []
    latencies: list[float] = []
    orders: list[list[int]] = []
    for row in rows:
        pool = row["pool"]
        if not pool:
            ranks.append(None)
            orders.append([])
            continue
        t0 = time.perf_counter()
        if sim:
            ranked = sim_rerank(row["question"], pool, top_k=top_k, arm=arm)
        elif arm == "llm":
            ranked = llm_rerank(row["question"], pool, llm, top_k=top_k)
        else:
            ranked = cross_encoder_rerank(row["question"], pool, top_k=top_k,
                                          model_name=ce_model)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ranks.append(_gold_rank(row["gold"], ranked))
        # Pool positions in the arm's output order, for the rank correlation.
        by_id = {id(h): i for i, h in enumerate(pool)}
        orders.append([by_id.get(id(h), -1) for h in ranked])
    return {"arm": arm, "model": (None if arm == "llm" else ce_model),
            "ranks": ranks, "latencies": latencies, "orders": orders,
            "tokens": (getattr(llm, "total_tokens", 0) if arm == "llm" and llm else 0)}


def _gold_rank(gold: list[str], ranked: list) -> int | None:
    """1-indexed position of the first gold-bearing hit, or None if absent."""
    for i, hit in enumerate(ranked, 1):
        if _gold_in_pool(gold, [hit.text]):
            return i
    return None


# ── Stats ───────────────────────────────────────────────────────────────────

def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _pctile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation, stdlib only. None when undefined (<3 pairs or a
    constant sequence). Ties get average ranks."""
    if len(a) != len(b) or len(a) < 3:
        return None

    def _ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den_a = sum((x - mean_a) ** 2 for x in ra)
    den_b = sum((y - mean_b) ** 2 for y in rb)
    if den_a <= 0 or den_b <= 0:
        return None
    return num / ((den_a * den_b) ** 0.5)


def arm_stats(res: dict, *, top_k: int) -> dict:
    """Gold-rank distribution + latency for one arm.

    A gold turn the arm did not return at all is EXCLUDED from the median (it has
    no rank) but counted in `found`/`missing` and in the ≤15 share's denominator —
    otherwise an arm that drops gold entirely would post the best median."""
    found = [r for r in res["ranks"] if r is not None]
    n = len(res["ranks"]) or 1
    return {
        "arm": res["arm"], "model": res["model"], "n": len(res["ranks"]),
        "found": len(found), "missing": len(res["ranks"]) - len(found),
        "median_rank": _median([float(r) for r in found]),
        "mean_rank": (sum(found) / len(found)) if found else 0.0,
        "top15_share": 100.0 * sum(1 for r in found if r <= 15) / n,
        f"top{top_k}_share": 100.0 * sum(1 for r in found if r <= top_k) / n,
        "p50_ms": _pctile(res["latencies"], 50),
        "p95_ms": _pctile(res["latencies"], 95),
        "tokens": res["tokens"],
    }


def rank_agreement(a: dict, b: dict) -> tuple[float | None, int]:
    """Median per-question Spearman correlation between two arms' orderings of
    the same pool — how much the two backends actually disagree, independent of
    whether either found gold.

    Computed over the INTERSECTION of the two arms' returned candidates (both
    return only `top_k` of the pool, so there is no shared ordering outside it).
    Returns (rho, n_questions_scored); a low n is itself the finding — arms whose
    outputs barely overlap have no measurable correlation, which is a stronger
    disagreement signal than any rho would be."""
    rhos = []
    for oa, ob in zip(a["orders"], b["orders"]):
        common = [p for p in oa if p in set(ob)]
        if len(common) < 3:
            continue
        pos_a = {p: i for i, p in enumerate(oa)}
        pos_b = {p: i for i, p in enumerate(ob)}
        rho = spearman([pos_a[p] for p in common], [pos_b[p] for p in common])
        if rho is not None:
            rhos.append(rho)
    return (_median(rhos) if rhos else None), len(rhos)


def gate_m1(ce: dict, llm: dict) -> dict:
    """CE parity vs the LLM arm — the adoption question is 'as good', not 'better'."""
    speedup = (llm["p95_ms"] / ce["p95_ms"]) if ce["p95_ms"] else None
    checks = {
        "median_within_tolerance":
            ce["median_rank"] <= llm["median_rank"] + _M1_MEDIAN_TOLERANCE,
        "top15_within_tolerance":
            ce["top15_share"] >= llm["top15_share"] - _M1_TOP15_TOLERANCE_PP,
        "latency_10x_better":
            speedup is not None and speedup >= _M1_MIN_SPEEDUP,
    }
    return {"checks": checks, "speedup": speedup, "pass": all(checks.values())}


def gate_m2(nl_mxbai: dict, nl_bge: dict, en_mxbai: dict, en_bge: dict) -> dict:
    checks = {
        "dutch_no_worse": nl_bge["median_rank"] <= nl_mxbai["median_rank"],
        "english_regression_bounded":
            en_bge["median_rank"] <= en_mxbai["median_rank"] + _M2_MAX_EN_REGRESSION,
    }
    return {"checks": checks, "pass": all(checks.values())}


# ── Report ──────────────────────────────────────────────────────────────────

def _print_arm(label: str, s: dict) -> None:
    print(f"  {label:<22}median={s['median_rank']:>5.1f}  "
          f"mean={s['mean_rank']:>5.1f}  ≤15={s['top15_share']:>5.1f}%  "
          f"found={s['found']}/{s['n']}  "
          f"p50={s['p50_ms']:>7.1f}ms  p95={s['p95_ms']:>7.1f}ms"
          + (f"  tokens={s['tokens']}" if s["tokens"] else ""))


def _checkblock(name: str, gate: dict, labels: dict[str, str]) -> None:
    print(f"\n── {name}: {'PASS' if gate['pass'] else 'FAIL'} ──")
    for key, ok in gate["checks"].items():
        print(f"  [{'✓' if ok else '✗'}] {labels.get(key, key)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m1", action="store_true", help="cross-encoder vs LLM backend")
    ap.add_argument("--m2", action="store_true", help="mxbai vs bge-reranker-v2-m3")
    ap.add_argument("--handset", type=Path,
                    default=Path(__file__).resolve().parent / "rerank_handset.json",
                    help="NL/EN hand-set JSON (default: rerank_handset.json)")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="run M1 on an LME slice instead of the EN hand-set")
    ap.add_argument("--category", default="multi-session",
                    help="LME question_type filter for --dataset")
    ap.add_argument("--sample", type=int, default=40,
                    help="max LME questions to pool (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=_DEFAULT_POOL,
                    help=f"BM25 candidates handed to each arm (default {_DEFAULT_POOL})")
    ap.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K,
                    help=f"survivors each arm returns (default {_DEFAULT_TOP_K})")
    ap.add_argument("--ce-model", default=MXBAI, help="M1's cross-encoder model")
    ap.add_argument("--bge-model", default=BGE_M3, help="M2's challenger model")
    ap.add_argument("--api-key", default="", help="LLM arm (M1 only)")
    ap.add_argument("--model", default="deepseek-v4-flash", help="LLM arm model")
    ap.add_argument("--sim", action="store_true",
                    help="fake rerankers — plumbing only, numbers are meaningless")
    ap.add_argument("--out", type=Path, default=None, help="write the summary JSON")
    args = ap.parse_args()

    if not (args.m1 or args.m2 or args.sim):
        ap.error("pick a measurement: --m1, --m2 (or --sim for plumbing)")

    spec = json.loads(args.handset.read_text())
    tmp = tempfile.TemporaryDirectory()
    summary: dict = {"config": {"pool": args.pool, "top_k": args.top_k,
                                "sim": args.sim, "ce_model": args.ce_model,
                                "bge_model": args.bge_model}}
    passed = True
    try:
        root = Path(tmp.name)

        # Fail LOUD when the cross-encoder backend is unavailable: `cross_encoder_rerank`
        # degrades to returning the pool unchanged, which would post a plausible-looking
        # parity result for a reranker that never ran.
        if not args.sim:
            for needed in ([args.ce_model] if args.m1 else []) + (
                    [args.ce_model, args.bge_model] if args.m2 else []):
                if _get_cross_encoder(needed) is None:
                    print(f"ERROR: cross-encoder {needed!r} unavailable "
                          f"(install sentence-transformers / download the model). "
                          f"Refusing to run: the backend degrades to an UNCHANGED "
                          f"pool, which would read as parity.")
                    sys.exit(2)

        if args.m1 or args.sim:
            if args.dataset:
                qs = load_longmemeval_data(str(args.dataset), max_questions=None,
                                           seed=args.seed)
                qs = [q for q in qs
                      if args.category == "all" or q.get("question_type") == args.category]
                rows = lme_pools(qs, pool=args.pool, root=root / "m1",
                                 limit=args.sample)
                source = f"LME {args.category} ({len(rows)} questions)"
            else:
                rows = handset_pools(spec["en"], pool=args.pool, root=root / "m1en")
                source = f"EN hand-set ({len(rows)} questions)"

            llm = None
            if not args.sim:
                if not args.api_key:
                    print("ERROR: --m1 needs --api-key for the LLM arm.")
                    sys.exit(2)
                from hymem.contrib.openai_client import OpenAICompatibleClient
                llm = OpenAICompatibleClient(
                    api_key=args.api_key, base_url="https://api.deepseek.com",
                    model=args.model)

            ce = run_arm("ce", rows, top_k=args.top_k, ce_model=args.ce_model,
                         sim=args.sim)
            ll = run_arm("llm", rows, top_k=args.top_k, llm=llm, sim=args.sim)
            ce_s, ll_s = arm_stats(ce, top_k=args.top_k), arm_stats(ll, top_k=args.top_k)
            gate = gate_m1(ce_s, ll_s)
            agree, agree_n = rank_agreement(ce, ll)

            print(f"\n=== M1 — backend: cross-encoder vs LLM ===\n  source: {source}"
                  f"   pool={args.pool} → top_k={args.top_k}")
            _print_arm("cross-encoder", ce_s)
            _print_arm("llm", ll_s)
            print("  rank agreement (median Spearman over shared candidates): "
                  + (f"{agree:+.3f} over {agree_n}/{len(rows)} questions"
                     if agree is not None else
                     "n/a — the arms returned <3 shared candidates on every "
                     "question, i.e. they disagree almost completely"))
            _checkblock("M1 gate", gate, {
                "median_within_tolerance":
                    f"CE median gold rank within {_M1_MEDIAN_TOLERANCE} of LLM "
                    f"({ce_s['median_rank']:.1f} vs {ll_s['median_rank']:.1f})",
                "top15_within_tolerance":
                    f"CE ≤15 share within {_M1_TOP15_TOLERANCE_PP:.0f}pp of LLM "
                    f"({ce_s['top15_share']:.1f}% vs {ll_s['top15_share']:.1f}%)",
                "latency_10x_better":
                    f"CE p95 ≥{_M1_MIN_SPEEDUP:.0f}× faster "
                    + (f"({gate['speedup']:.1f}×)" if gate["speedup"] else "(n/a)"),
            })
            summary["m1"] = {"source": source, "ce": ce_s, "llm": ll_s,
                             "agreement": agree, "agreement_n": agree_n,
                             "gate": gate}
            passed = passed and gate["pass"]

        if args.m2 or args.sim:
            nl = handset_pools(spec["nl"], pool=args.pool, root=root / "m2nl")
            en = handset_pools(spec["en"], pool=args.pool, root=root / "m2en")
            arms = {}
            for lang, rows in (("nl", nl), ("en", en)):
                for tag, model in (("mxbai", args.ce_model), ("bge", args.bge_model)):
                    arms[(lang, tag)] = arm_stats(
                        run_arm("ce", rows, top_k=args.top_k, ce_model=model,
                                sim=args.sim),
                        top_k=args.top_k)
            gate = gate_m2(arms[("nl", "mxbai")], arms[("nl", "bge")],
                           arms[("en", "mxbai")], arms[("en", "bge")])

            print(f"\n=== M2 — model: mxbai vs bge-reranker-v2-m3 ===\n"
                  f"  NL hand-set: {len(nl)} questions   EN control: {len(en)}"
                  f"   pool={args.pool} → top_k={args.top_k}")
            for (lang, tag), s in arms.items():
                _print_arm(f"{lang.upper()} {tag}", s)
            _checkblock("M2 gate", gate, {
                "dutch_no_worse":
                    f"bge ≥ mxbai on Dutch "
                    f"({arms[('nl','bge')]['median_rank']:.1f} vs "
                    f"{arms[('nl','mxbai')]['median_rank']:.1f} median)",
                "english_regression_bounded":
                    f"English median regression ≤ {_M2_MAX_EN_REGRESSION} "
                    f"({arms[('en','bge')]['median_rank']:.1f} vs "
                    f"{arms[('en','mxbai')]['median_rank']:.1f})",
            })
            summary["m2"] = {"arms": {f"{k[0]}-{k[1]}": v for k, v in arms.items()},
                             "gate": gate}
            passed = passed and gate["pass"]
    finally:
        tmp.cleanup()

    if args.sim:
        print("\n  ⚠ --sim: fake rerankers. This run proves the PLUMBING only — "
              "its\n    ranks, latencies and gate verdicts are not evidence about "
              "any backend.")
    else:
        print("\n  Adoption is Step 6, not this script: a pass here is decision "
              "data.\n  Nothing flips until Step 5's scored runs are in.")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"  summary → {args.out}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
