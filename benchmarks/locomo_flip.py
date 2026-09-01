#!/usr/bin/env python3
"""Per-question flip comparison between two LoCoMo runs.

Accuracy deltas and bucket totals hide offsetting moves: a run that fixes 7
questions and breaks 7 looks identical to one that touched nothing. Distractor
dilution has exactly one signature — questions that were CORRECT in the baseline
going WRONG at the wider setting — and no aggregate in the adapter report can
show it. This joins two `--out` files on `id` and reports the flips.

The dilution-critical distinction is between a regression where the evidence was
reaching the reader in BOTH runs (a genuine reader-side loss: same evidence, more
distractors, now wrong) and one where the evidence stopped reaching the reader
(a retrieval/budget regression that happens to surface as a wrong answer). Only
the former is dilution; they are counted separately.

Usage:
  python locomo_flip.py BASE.json NEW.json
  python locomo_flip.py BASE.json NEW.json --list          # show flipped questions
  python locomo_flip.py BASE.json NEW.json --category 1    # restrict to one category
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Miss buckets, in pipeline order — mirrors the adapter's miss decomposition.
BUCKETS = ["correct", "retrieval", "ranking", "budget", "synthesis"]
_CATNAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop",
            5: "adversarial"}

# The adapters disagree on what the model's answer is called (`facts_ab.py:51`
# carries the same list, for the same reason). Reading only `ai_answer` does not
# fail loudly on a file that names it something else: absent on both sides, the
# judge-only test below compares None to None on every row and passes.
_ANSWER_KEYS = ("ai_answer", "hypothesis", "prediction", "response")


def answer_text(row: dict) -> str | None:
    """What the reader said, under whichever name this adapter used.

    None means the row records no answer at all -- which is NOT the same as the
    two arms agreeing, and must never be read as agreement."""
    for k in _ANSWER_KEYS:
        v = row.get(k)
        if v:
            return v
    return None


def bucket(r: dict) -> str:
    """Where this question died. `gold_in_topk` is absent in runs made before the
    four-surface diagnostic landed; falling back to `gold_in_context` collapses
    `budget` into `synthesis` exactly as those older runs reported it."""
    if r.get("correct"):
        return "correct"
    if not r.get("gold_in_pool"):
        return "retrieval"
    if not r.get("gold_in_topk", r.get("gold_in_context")):
        return "ranking"
    if not r.get("gold_in_context"):
        return "budget"
    return "synthesis"


def load(path: str) -> tuple[dict[str, dict], bool]:
    """Returns {id: record} plus whether the file carries the four-surface
    diagnostic (pre-fix files can't distinguish budget loss from synthesis)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        sys.exit(f"{path}: expected a list of per-question results")
    staged = any("gold_in_topk" in r for r in rows)
    return {r["id"]: r for r in rows}, staged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="baseline results JSON (the A arm)")
    ap.add_argument("new", help="new results JSON (the B arm)")
    ap.add_argument("--category", type=int, default=None,
                    help="restrict to one LoCoMo category (1-5)")
    ap.add_argument("--list", action="store_true",
                    help="print the flipped questions (id, category, question)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    a_rows, a_staged = load(args.base)
    b_rows, b_staged = load(args.new)
    shared = sorted(set(a_rows) & set(b_rows))
    if not shared:
        sys.exit("No question ids in common — different samples or seeds?")
    if args.category:
        shared = [i for i in shared if a_rows[i].get("category") == args.category]
        if not shared:
            sys.exit(f"No category-{args.category} questions in common.")

    only_a, only_b = len(a_rows) - len(shared), len(b_rows) - len(shared)
    # A row the JUDGE never scored (`correct is None`, D3) is dropped from BOTH
    # arms or from neither. Dropping it from one would make the comparison
    # unpaired on exactly the rows an outage touched — the C4 arm-asymmetry void
    # condition, arriving through the back door. Reported, never silent.
    unscored = [i for i in shared
                if a_rows[i].get("correct") is None
                or b_rows[i].get("correct") is None]
    if unscored:
        shared = [i for i in shared if i not in set(unscored)]
        if not shared:
            sys.exit(f"Every shared question is UNSCORED in one arm or the "
                     f"other ({len(unscored)}) — the judge failed, so there is "
                     f"nothing to compare. Re-judge before reading this gate.")
    a_acc = sum(a_rows[i]["correct"] for i in shared) / len(shared)
    b_acc = sum(b_rows[i]["correct"] for i in shared) / len(shared)

    print(f"\n=== LoCoMo flip comparison — {len(shared)} shared questions ===")
    print(f"  A  {args.base}")
    print(f"  B  {args.new}")
    if only_a or only_b:
        print(f"  [note] {only_a} ids only in A, {only_b} only in B — ignored")
    if unscored:
        print(f"  [note] {len(unscored)} shared id(s) UNSCORED in one arm "
              f"(judge error) — dropped from BOTH arms, not counted wrong")
    if not (a_staged and b_staged):
        stale = args.base if not a_staged else args.new
        print(f"  [note] {stale} predates the four-surface diagnostic: its budget\n"
              f"         losses are still folded into `synthesis` (see bucket table)")
    # A --rejudge pair holds the reader byte-identical, so nothing that moves can
    # be dilution — it is judge nondeterminism by construction. Detect it rather
    # than trusting the caller to remember which pair they are looking at.
    #
    # This decides the label on the reader-side regression count below, so a
    # wrong answer here is not a cosmetic one: it prints "JUDGE churn" over the
    # exact rows that carry the dilution signature. It must therefore be able to
    # come out FALSE. Comparing `.get("ai_answer")` could not: on a pair that
    # records the answer under another name, or records none, every row compares
    # None to None and the run is declared a re-judge of itself. Absence is not
    # agreement, so it is a third outcome, not a pass.
    unanswered = [i for i in shared
                  if answer_text(a_rows[i]) is None or answer_text(b_rows[i]) is None]
    judge_only = (not unanswered and
                  all(answer_text(a_rows[i]) == answer_text(b_rows[i])
                      for i in shared))
    if unanswered:
        print(f"  [unclassified] {len(unanswered)}/{len(shared)} shared row(s) "
              f"record no reader answer under any of {', '.join(_ANSWER_KEYS)},\n"
              f"                 so whether B re-judged A cannot be read off "
              f"these files. The regression\n                 cause below is "
              f"labelled DILUTION, which is the assumption that can be checked.")
    elif judge_only:
        print("  [judge-only] every shared answer is byte-identical — B is a "
              "re-judge of A.\n               All flips below are JUDGE churn; "
              "the reader never moved.")
    print(f"\n  A accuracy: {a_acc*100:.1f}%")
    print(f"  B accuracy: {b_acc*100:.1f}%   ({(b_acc-a_acc)*100:+.1f}pp)")

    gains, regressions = [], []
    for i in shared:
        a, b = a_rows[i], b_rows[i]
        if a["correct"] and not b["correct"]:
            regressions.append(i)
        elif not a["correct"] and b["correct"]:
            gains.append(i)

    print(f"\n  ── flips ──")
    print(f"  fixed   (A wrong → B correct): {len(gains):>3}")
    print(f"  broken  (A correct → B wrong): {len(regressions):>3}")
    print(f"  net                          : {len(gains)-len(regressions):>+3}"
          f"  ({(b_acc-a_acc)*100:+.1f}pp)")
    if not regressions:
        print("  → no regressions: B is a strict superset of A's correct answers.")

    # The dilution test. A regression only implicates the READER if the evidence
    # reached it in both arms; otherwise B simply stopped surfacing the evidence.
    if regressions:
        same_ev = [i for i in regressions
                   if a_rows[i].get("gold_in_context") and b_rows[i].get("gold_in_context")]
        lost_ev = [i for i in regressions if i not in same_ev]
        print(f"\n  ── regression cause ──")
        print(f"  reader-side  (evidence reached reader in BOTH, now wrong): {len(same_ev):>3}"
              f"   {'← JUDGE churn (same answer text)' if judge_only else '← dilution'}")
        print(f"  evidence lost (surfaced in A, not in B):                   {len(lost_ev):>3}")
        by_cat = Counter(_CATNAME.get(a_rows[i].get("category"), "?") for i in same_ev)
        if by_cat:
            print("  dilution by category: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(by_cat.items())))

    # Bucket migration — where the moved questions came from and went to.
    # Answerable cats only: cat-5 "evidence" is the trap source, not a gold
    # location, so its surfacing flags are not miss causes.
    movers = defaultdict(int)
    for i in shared:
        if a_rows[i].get("category") == 5:
            continue
        ba, bb = bucket(a_rows[i]), bucket(b_rows[i])
        if ba != bb:
            movers[(ba, bb)] += 1
    if movers:
        print(f"\n  ── bucket migration (answerable cats; A → B) ──")
        for (ba, bb), n in sorted(movers.items(),
                                  key=lambda kv: (-kv[1], kv[0])):
            arrow = "✓" if bb == "correct" else ("✗" if ba == "correct" else " ")
            print(f"  {arrow} {ba:>10} → {bb:<10} {n:>3}")

    # Per-category flip table.
    cats = sorted({a_rows[i].get("category") for i in shared})
    print(f"\n  {'category':<14} {'A':>7} {'B':>7} {'Δ':>7} {'fixed':>6} {'broken':>7} {'n':>5}")
    for c in cats:
        ids = [i for i in shared if a_rows[i].get("category") == c]
        aa = sum(a_rows[i]["correct"] for i in ids) / len(ids)
        bb = sum(b_rows[i]["correct"] for i in ids) / len(ids)
        f = sum(1 for i in ids if i in set(gains))
        br = sum(1 for i in ids if i in set(regressions))
        print(f"  {_CATNAME.get(c, c):<14} {aa*100:>6.1f}% {bb*100:>6.1f}% "
              f"{(bb-aa)*100:>+6.1f} {f:>6} {br:>7} {len(ids):>5}")

    if args.list:
        for label, ids in (("BROKEN (A correct → B wrong)", regressions),
                           ("FIXED (A wrong → B correct)", gains)):
            if not ids:
                continue
            print(f"\n  ── {label} ──")
            for i in ids:
                a, b = a_rows[i], b_rows[i]
                print(f"  [{i}] {_CATNAME.get(a.get('category'), '?')}"
                      f"  {bucket(a)} → {bucket(b)}")
                print(f"      Q: {a.get('question', '')[:150]}")
                print(f"      gold: {str(a.get('answer'))[:110]}")
                print(f"      B answered: {str(answer_text(b))[:110]}")

    if args.json:
        print(json.dumps({
            "n_shared": len(shared), "a_accuracy": a_acc, "b_accuracy": b_acc,
            "fixed": gains, "broken": regressions,
            # Without this the JSON and the printed report disagree about what
            # the same rows mean: `dilution_regressions` names rows the text
            # above may have just relabelled JUDGE churn.
            "judge_only": judge_only, "unanswered": len(unanswered),
            "dilution_regressions": [i for i in regressions
                                     if a_rows[i].get("gold_in_context")
                                     and b_rows[i].get("gold_in_context")],
            "migration": {f"{k[0]}->{k[1]}": v for k, v in movers.items()},
        }, indent=2))


if __name__ == "__main__":
    main()
