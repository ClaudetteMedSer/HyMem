#!/usr/bin/env python3
"""Diff the recall-ceiling diagnostics across a sweep of benchmark runs.

The in-run recall diagnostic tells you ranking is the bottleneck; it can't tell
you whether an L2a budget bump (`--rerank-top-k 40/60`) helped MS vs SS-user,
because a "delta" needs a baseline run to subtract. This tool reads the
`recall_diagnostics` + `per_question` blocks already persisted in each run's
result JSON and prints, per category, the RANKING-miss count for every run with
its delta vs the first (baseline) file — so a sweep is read at a glance instead
of by eyeballing N separate banners.

Each run is labelled by the levers in its config block (rerank_top_k / model /
embeddings / auto_ability) so the columns are self-describing.

Usage (first file = baseline; order the rest as the sweep):
  python compare_recall.py \
      ~/.hermes/benchmarks/longmemeval-...-baseline.json \
      ~/.hermes/benchmarks/longmemeval-...-rtk40.json \
      ~/.hermes/benchmarks/longmemeval-...-rtk60.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _label(cfg: dict) -> str:
    """Compact lever signature for a run's column header."""
    parts = [f"rtk={cfg.get('rerank_top_k') or 20}"]
    if cfg.get("rerank_model"):
        parts.append(cfg["rerank_model"])
    if cfg.get("embeddings"):
        parts.append("emb")
    if cfg.get("auto_ability"):
        parts.append("auto")
    return ",".join(parts)


def _overall(run: dict) -> tuple[int, int]:
    """(correct, total) from per_question — robust, run-format-independent."""
    pq = run.get("per_question", [])
    correct = sum(1 for r in pq if r.get("correct"))
    return correct, len(pq)


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+",
                    help="result JSONs; the FIRST is the baseline all deltas subtract.")
    ap.add_argument("--metric", default="miss_ranking",
                    choices=["miss_ranking", "miss_retrieval"],
                    help="which miss bucket to diff (default ranking — the L2a target).")
    args = ap.parse_args()

    runs = [_load(p) for p in args.files]
    labels = [_label(r.get("config", {})) for r in runs]
    diags = [r.get("recall_diagnostics", {}) for r in runs]

    # Disambiguate duplicate labels (same levers run twice) by index suffix.
    seen: dict[str, int] = {}
    for i, lab in enumerate(labels):
        if labels.count(lab) > 1:
            seen[lab] = seen.get(lab, 0) + 1
            labels[i] = f"{lab}#{seen[lab]}"

    cats = sorted({c for d in diags for c in d if not c.startswith("_")})

    print(f"\nRecall-diagnostic sweep — metric: {args.metric}  "
          f"(Δ vs baseline = {labels[0]})")
    print(f"  baseline: {Path(args.files[0]).name}\n")

    # ── overall ──────────────────────────────────────────────────────
    base_c, base_t = _overall(runs[0])
    base_pct = 100 * base_c / base_t if base_t else 0.0
    hdr = f"  {'':<22}" + "".join(f"{lab:>18}" for lab in labels)
    print(hdr)
    print(f"  {'─'*(22+18*len(labels))}")
    row = f"  {'OVERALL acc':<22}"
    for i, r in enumerate(runs):
        c, t = _overall(r)
        pct = 100 * c / t if t else 0.0
        cell = f"{pct:.1f}%" if i == 0 else f"{pct:.1f}% ({pct-base_pct:+.1f})"
        row += f"{cell:>18}"
    print(row + "\n")

    # ── per-category miss counts ─────────────────────────────────────
    print(f"  {args.metric} by category (lower is better):")
    print(f"  {'─'*(22+18*len(labels))}")
    for cat in cats:
        base_v = diags[0].get(cat, {}).get(args.metric)
        row = f"  {cat:<22}"
        for i, d in enumerate(diags):
            v = d.get(cat, {}).get(args.metric)
            if v is None:
                row += f"{'—':>18}"
            elif i == 0 or base_v is None:
                row += f"{v:>18}"
            else:
                delta = v - base_v
                arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "·")
                row += f"{f'{v} ({delta:+d}{arrow})':>18}"
        print(row)

    print(f"\n  Read: a category whose {args.metric} drops (↓) under a wider rtk is\n"
          f"  where the budget bump landed; flat (·) means the gold wasn't in the\n"
          f"  newly-reached band (see gold_rank_probe.py to confirm where it sits).")


if __name__ == "__main__":
    main()
