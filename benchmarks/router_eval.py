#!/usr/bin/env python3
"""Zero-LLM evaluation of HyMem's production ability router (`detect_ability`).

WHY this exists, separate from the full benchmark: the MR/TR retrieval shaping
in `augment()` only fires in real Hermes when `detect_ability` infers the intent
from raw query text — the host supplies no oracle label. The full LongMemEval run
measures this only as a side-effect (and costs an answer + judge LLM call per
question). But `detect_ability` is pure text -> label: it needs no retrieval, no
LLM, no DB. So we can sweep the ENTIRE dataset in seconds and read the exact
router recall/precision against the oracle `question_type`, giving a tight loop
for tuning `intent.py` decoupled from the expensive retrieval+judge benchmark.

It reuses the adapter's `QUESTION_TYPE_TO_ABILITY`, `compute_router_diagnostics`,
and `print_router_diagnostics` so the numbers are identical to the shadow block
the full run prints — this is just the same measurement, run cheaply and over
everything, with the residual misses listed so the next pattern round is targeted.

Usage:
    python benchmarks/router_eval.py --data-dir <dir> [--scale S] [--show 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the sibling adapter importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from longmemeval_adapter import (  # noqa: E402
    QUESTION_TYPE_TO_ABILITY,
    compute_router_diagnostics,
    load_longmemeval_data,
    print_router_diagnostics,
)

from hymem.query.intent import detect_ability  # noqa: E402


def _target(oracle: str | None) -> str:
    return oracle if oracle in ("MR", "TR") else "NONE"


def _det(d: str | None) -> str:
    return d if d in ("MR", "TR") else "NONE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/home/node/longmemeval",
                    help="Directory holding longmemeval_<scale>_cleaned.json")
    ap.add_argument("--scale", default="S", help="S / M / ... (default S)")
    ap.add_argument("--show", type=int, default=40,
                    help="Max residual misses to list per bucket (default 40)")
    args = ap.parse_args()

    scale = args.scale.upper()
    name = "longmemeval_s_cleaned.json" if scale == "S" else f"longmemeval_{scale.lower()}_cleaned.json"
    data_file = Path(args.data_dir) / name
    if not data_file.exists():
        print(f"ERROR: Dataset not found at {data_file}")
        sys.exit(1)

    # Full set, no sampling — the router pass is free, so there is no reason not
    # to measure every question.
    questions = load_longmemeval_data(str(data_file), max_questions=None)

    results = []
    for q in questions:
        qtype = q["question_type"]
        oracle = QUESTION_TYPE_TO_ABILITY.get(qtype, None)
        detected = detect_ability(q.get("question", "") or "")
        results.append({
            "question_id": q.get("question_id", "?"),
            "question": q.get("question", ""),
            "question_type": qtype,
            "oracle_ability": oracle,
            "detected_ability": detected,
        })

    diag = compute_router_diagnostics(results)
    print(f"\n{'='*72}")
    print(f"  ROUTER EVAL — detect_ability vs oracle  (n={len(results)}, zero-LLM)")
    print(f"  Dataset: {data_file}")
    print(f"{'='*72}")
    print_router_diagnostics(diag, auto_ability=False)

    # Residual misses, the actionable output: each intent's questions the router
    # FAILED to catch (recall loss), then the NONE-category questions it WRONGLY
    # shaped (precision loss). These name exactly what the next pattern round must
    # fix — recall misses widen patterns, false positives tighten them.
    for intent in ("TR", "MR"):
        miss = [r for r in results
                if _target(r["oracle_ability"]) == intent and _det(r["detected_ability"]) != intent]
        print(f"\n  {intent} recall misses — oracle {intent}, router said "
              f"'{{detected}}'  ({len(miss)} total):")
        for r in miss[:args.show]:
            got = r["detected_ability"] or "None"
            print(f"    [{r['question_type']}] det={got:<4} | {r['question'][:96]}")
        if len(miss) > args.show:
            print(f"    … +{len(miss) - args.show} more")

    fp = [r for r in results
          if _target(r["oracle_ability"]) == "NONE" and _det(r["detected_ability"]) != "NONE"]
    print(f"\n  False positives — non-MR/TR oracle, router shaped MR/TR  ({len(fp)} total):")
    for r in fp[:args.show]:
        print(f"    [{r['question_type']}] det={r['detected_ability']:<4} | {r['question'][:96]}")
    if len(fp) > args.show:
        print(f"    … +{len(fp) - args.show} more")


if __name__ == "__main__":
    main()
