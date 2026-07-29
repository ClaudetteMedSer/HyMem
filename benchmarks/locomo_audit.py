#!/usr/bin/env python3
"""Adjudication aid for LoCoMo's synthesis bucket.

`gold_in_context` is a lexical τ=0.6 judgement, not proof the reader saw the
gold: it can fire when the evidence turn merely shares content words with an
unrelated retrieved memory. Every such false positive moves a question OUT of
the retrieval bucket and INTO synthesis, which is the one direction that
flatters the retriever and blames the reader — so the synthesis share is an
UPPER bound until this is checked.

This prints each synthesis miss with its evidence turn text beside the rendered
context (requires a run made with --dump-context), and re-scores the match at a
stricter τ so the suspect ones are ranked first. It does not decide anything;
it puts the two strings next to each other so a human can.

Usage:
  python locomo_audit.py RESULTS.json --data data/locomo10.json
  python locomo_audit.py RESULTS.json --data ... --suspect-only --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from locomo_adapter import load_locomo_data, CATEGORY_NAME
from msc_adapter import _lex_match


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="a --out file, ideally made with --dump-context")
    ap.add_argument("--data", required=True, help="locomo10.json")
    ap.add_argument("--user-speaker", choices=["a", "b"], default="a",
                    help="must match the run being audited")
    ap.add_argument("--tau-strict", type=float, default=0.85,
                    help="stricter threshold; misses that pass 0.6 but fail this "
                         "are the τ=0.6 false-positive candidates (default 0.85)")
    ap.add_argument("--suspect-only", action="store_true",
                    help="only show the likely false positives")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chars", type=int, default=700, help="context excerpt length")
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    ev_map: dict[str, dict] = {}
    for conv in load_locomo_data(args.data, user_speaker=args.user_speaker):
        ev_map[conv["id"]] = conv["evidence_map"]

    misses = [r for r in rows
              if not r["correct"] and r["category"] != 5 and r.get("gold_in_context")]
    if not misses:
        sys.exit("No synthesis-bucket misses in this file.")
    has_ctx = sum("context" in r for r in misses)
    if not has_ctx:
        print("[warn] no `context` field — re-run with --dump-context for the "
              "strict re-check; showing evidence text only.\n")

    scored = []
    for r in misses:
        ev = [ev_map.get(r["conv_id"], {}).get(e) for e in (r.get("evidence") or [])]
        texts = [t for hit in ev if hit for _, t in [hit]]
        ctx = r.get("context") or ""
        strict = all(_lex_match(t, ctx, tau=args.tau_strict) for t in texts) if (texts and ctx) else None
        scored.append((r, texts, strict))

    suspect = [s for s in scored if s[2] is False]
    print(f"=== synthesis-bucket audit — {len(misses)} misses "
          f"({has_ctx} with rendered context) ===")
    if has_ctx:
        print(f"  τ=0.6 says gold reached the reader in all {len(misses)}.")
        print(f"  At τ={args.tau_strict}: {len(suspect)} FAIL → likely lexical false "
              f"positives, i.e. retrieval misses miscounted as synthesis "
              f"({len(suspect)/len(misses)*100:.0f}% of the bucket).")
    by_cat = Counter(CATEGORY_NAME[r["category"]] for r, _, _ in scored)
    print("  by category: " + ", ".join(f"{k} {v}" for k, v in sorted(by_cat.items())))

    show = suspect if args.suspect_only else scored
    if args.limit:
        show = show[:args.limit]
    for r, texts, strict in show:
        tag = {True: "gold present", False: "SUSPECT — gold not really present",
               None: "unchecked"}[strict]
        print(f"\n{'─'*72}\n[{r['id']}] {CATEGORY_NAME[r['category']]}  ({tag})")
        print(f"  Q:    {r.get('question','')}")
        print(f"  gold: {str(r.get('answer'))[:300]}")
        print(f"  said: {str(r.get('ai_answer'))[:300]}")
        for t in texts:
            print(f"  EVIDENCE TURN: {t[:300]}")
        if r.get("context"):
            print(f"  CONTEXT[:{args.chars}]: {r['context'][:args.chars]}")


if __name__ == "__main__":
    main()
