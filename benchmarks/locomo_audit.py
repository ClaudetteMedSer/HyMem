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
    ap.add_argument("--topk-dump", default=None, metavar="DIAG.json",
                    help="a --diag-only sidecar carrying `topk_text`; joined by "
                         "question id so each suspect is ALSO re-scored at the "
                         "top_k surface. This is what separates a composition "
                         "loss (strict-passes at top_k, fails in the render) "
                         "from a recall loss (strict-fails at both)")
    ap.add_argument("--show-control", action="store_true",
                    help="dump the CONTROL rows (correct answers failing the strict "
                         "check) instead of the misses — read these to see what a "
                         "false alarm looks like before trusting the miss list")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chars", type=int, default=700, help="context excerpt length")
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    topk: dict[str, str] = {}
    pool: dict[str, str] = {}
    if args.topk_dump:
        dump = json.loads(Path(args.topk_dump).read_text(encoding="utf-8"))
        topk = {d["id"]: d.get("topk_text", "") for d in dump}
        # pool ⊇ topk: _evidence_diagnostics scores gold_in_pool against
        # topk_text + pool_text, so reproduce that same concatenation here.
        pool = {d["id"]: (d.get("topk_text", "") + " " + d.get("pool_text", ""))
                for d in dump if d.get("pool_text") is not None}
        missing = sum(r["id"] not in topk for r in rows)
        if missing:
            print(f"[warn] {missing}/{len(rows)} results have no row in the dump "
                  f"— sample/seed mismatch? Those are scored as unchecked.\n")
    ev_map: dict[str, dict] = {}
    for conv in load_locomo_data(args.data, user_speaker=args.user_speaker):
        ev_map[conv["id"]] = conv["evidence_map"]

    def _pop(want_correct: bool) -> list[dict]:
        # `correct is None` means the JUDGE never scored the row (D3), not that
        # the reader was wrong. Without this an outage would be handed to the
        # synthesis-bucket hand-check as a reader failure — a confident finding
        # produced by a timeout.
        return [r for r in rows if r.get("correct") is not None
                and bool(r["correct"]) is want_correct
                and r["category"] != 5 and r.get("gold_in_context")]

    misses = _pop(False)
    if not misses:
        sys.exit("No synthesis-bucket misses in this file.")
    has_ctx = sum("context" in r for r in misses)
    if not has_ctx:
        print("[warn] no `context` field — re-run with --dump-context for the "
              "strict re-check; showing evidence text only.\n")

    def _score(pop: list[dict]) -> list[tuple]:
        out = []
        for r in pop:
            ev = [ev_map.get(r["conv_id"], {}).get(e) for e in (r.get("evidence") or [])]
            texts = [t for hit in ev if hit for _, t in [hit]]
            ctx = r.get("context") or ""
            strict = (all(_lex_match(t, ctx, tau=args.tau_strict) for t in texts)
                      if (texts and ctx) else None)
            tk = topk.get(r["id"])
            strict_tk = (all(_lex_match(t, tk, tau=args.tau_strict) for t in texts)
                         if (texts and tk) else None)
            pl = pool.get(r["id"])
            strict_pl = (all(_lex_match(t, pl, tau=args.tau_strict) for t in texts)
                         if (texts and pl) else None)
            out.append((r, texts, strict, strict_tk, strict_pl))
        return out

    scored = _score(misses)
    suspect = [s for s in scored if s[2] is False]

    # THE CONTROL. A strict re-check on the MISSES alone cannot distinguish "the
    # gold never reached the reader" from "the check is too strict" — HyMem renders
    # consolidated memories, not verbatim turns, so a delivered fact can be worded
    # nothing like its source turn and fail a lexical test for that reason alone.
    # Questions the reader got RIGHT had the gold delivered by construction, so
    # their failure rate IS the check's false-alarm rate. Only the EXCESS over it
    # is evidence of miscounted retrieval. (Imperfect: correct answers may skew
    # toward easier lexical matches, which makes the control conservative.)
    ctrl = _score(_pop(True))
    ctrl_bad = [s for s in ctrl if s[2] is False]

    print(f"=== synthesis-bucket audit — {len(misses)} misses "
          f"({has_ctx} with rendered context) ===")
    if has_ctx:
        f = len(suspect) / len(misses)
        print(f"  τ=0.6 says gold reached the reader in all {len(misses)}.")
        print(f"  MISSES   fail τ={args.tau_strict}: {len(suspect):>4}/{len(misses)} "
              f"({f*100:.0f}%)")
        if ctrl:
            c = len(ctrl_bad) / len(ctrl)
            print(f"  CONTROL  fail τ={args.tau_strict}: {len(ctrl_bad):>4}/{len(ctrl)} "
                  f"({c*100:.0f}%)  <- reader answered CORRECTLY, so gold WAS "
                  f"delivered; this is the check's false-alarm rate")
            if f > c and c < 1.0:
                est = (f - c) / (1 - c) * len(misses)
                print(f"  EXCESS   {(f-c)*100:+.0f}pp  =>  ~{est:.0f} of {len(misses)} "
                      f"are plausibly genuine lexical FPs (retrieval misses booked "
                      f"as synthesis)")
            else:
                print(f"  EXCESS   {(f-c)*100:+.0f}pp  =>  NO evidence of miscounting: "
                      f"the strict check fails just as often where delivery is "
                      f"certain. The {len(suspect)} suspects are check artifacts.")
        else:
            print("  [warn] no correct+gold_in_context rows — control unavailable, "
                  "so the miss rate above is UNINTERPRETABLE on its own.")
    # THE LEVER TEST. Every suspect strict-fails in the render by construction;
    # the question is whether it strict-PASSES one surface earlier. Passing at
    # top_k means the memory was retrieved and then lost a render slot
    # (composition — the profile tier is the suspect). Failing at both means it
    # was never really retrieved (recall — --name-prefix). The tau=0.6 booleans
    # cannot make this call: render text is a SUBSET of top_k text, so
    # gold_in_context=True forces gold_in_topk=True and 100% is the only
    # possible answer.
    if topk and suspect:
        comp = [s for s in suspect if s[3] is True]
        lost = [s for s in suspect if s[3] is False and s[4] is True]
        gone = [s for s in suspect if s[3] is False and s[4] is False]
        unk = [s for s in suspect if s[3] is None
               or (s[3] is False and s[4] is None)]
        print(f"\n  ── where the {len(suspect)} suspects actually died "
              f"(strict tau={args.tau_strict}, three surfaces) ──")
        print(f"  composition  (in top_k, lost the render):    {len(comp):>3}"
              f"   -> render/tier ordering; budget or profile tier")
        print(f"  ranking      (in POOL, lost the top_k cut):  {len(lost):>3}"
              f"   -> matching quality; --name-prefix / bigger cut")
        print(f"  recall       (not even in the pool):         {len(gone):>3}"
              f"   -> never retrieved; indexing/pool aperture")
        if unk:
            print(f"  unchecked    (dump lacks pool_text or row):  {len(unk):>3}"
                  f"   -> re-run --diag-only to get the pool surface")

    by_cat = Counter(CATEGORY_NAME[r["category"]] for r, *_ in scored)
    print("  by category: " + ", ".join(f"{k} {v}" for k, v in sorted(by_cat.items())))

    show = ctrl_bad if args.show_control else (suspect if args.suspect_only else scored)
    if args.limit:
        show = show[:args.limit]
    for r, texts, strict, strict_tk, strict_pl in show:
        tag = {True: "gold present", False: "SUSPECT — gold not really present",
               None: "unchecked"}[strict]
        if strict is False and strict_tk is not None:
            tag += " | " + ("COMPOSITION (was in top_k)" if strict_tk
                            else "RANKING (in pool, missed the cut)" if strict_pl
                            else "RECALL (not even in the pool)" if strict_pl is False
                            else "top_k-miss (pool surface unchecked)")
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
