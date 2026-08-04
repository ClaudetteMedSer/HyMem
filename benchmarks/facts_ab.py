#!/usr/bin/env python3
"""Paired A/B re-analysis for the E1 narrative-facts tier (read-only, no model calls).

Why this exists: a whole-benchmark net delta is the WRONG statistic for a tier
that only fires on some questions. On every question where the tier did not
fire, the two arms received byte-identical reader input, so those questions
carry zero expected signal and pure reader churn. Including them dilutes the
effect and inflates the band -- the net delta's z-score is degraded by exactly
sqrt(n_total / n_fired) relative to the fired subset.

Two things this computes that a net score cannot:

1. FIRE RATE (`n_facts` > 0). The pre-registered mechanism read. All zeros means
   the tier never reached the reader and the score is a no-op BY CONSTRUCTION,
   not a null result -- the dead-path class that has bitten this project three
   times (see the diagnostic-controls memory).

2. A MEASURED noise floor instead of a remembered one. The n_facts == 0 subset
   is a built-in negative control: identical inputs in both arms, so its McNemar
   split IS this run pair's churn, measured on this run pair. If that subset
   shows a real delta, the arms differ by something other than the facts flag
   (different store, different config) and the whole comparison is void.

Test is McNemar on the discordant pairs, which is exact and self-calibrating:
it needs no assumed churn rate, unlike a sqrt(p/n) band.

    python facts_ab.py --on run_facts_on.json --off run_facts_off.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_rows(path: str) -> list[dict]:
    """Accept either a flat per-question list (LoCoMo/MSC `--out`) or a
    {config, per_question} envelope (LME)."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    for key in ("per_question", "results", "rows"):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise SystemExit(f"{path}: no per-question list found "
                     f"(keys: {sorted(obj)[:10]})")


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    """Exact-ish McNemar on discordant pairs.

    b = ON correct / OFF wrong, c = ON wrong / OFF correct. Under the null,
    b ~ Binomial(b+c, 0.5), so sd(b-c) = sqrt(b+c) and z = (b-c)/sqrt(b+c).
    Concordant pairs carry no information about the difference and are excluded
    -- that is the whole point of pairing.
    """
    b = sum(1 for on, off in pairs if on and not off)
    c = sum(1 for on, off in pairs if off and not on)
    n_disc = b + c
    z = (b - c) / math.sqrt(n_disc) if n_disc else 0.0
    n = len(pairs) or 1
    return {"n": len(pairs), "b": b, "c": c, "discordant": n_disc,
            "net_q": b - c, "net_pp": 100.0 * (b - c) / n, "z": z,
            # band on THIS subset, derived from the observed discordance rate
            "band_2sig_pp": 200.0 * math.sqrt(n_disc) / n if n_disc else 0.0}


def fmt(label: str, m: dict) -> str:
    verdict = "SIGNAL" if abs(m["z"]) >= 2.0 else "inside noise"
    if m["discordant"] == 0:
        verdict = "no discordant pairs"
    return (f"  {label:<26} n={m['n']:>4}  net={m['net_q']:+3d}q "
            f"({m['net_pp']:+5.2f}pp)  b={m['b']:<3} c={m['c']:<3} "
            f"z={m['z']:+5.2f}  +-{m['band_2sig_pp']:.2f}pp(2sig)  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--on", required=True, help="treatment arm (facts ON)")
    ap.add_argument("--off", required=True, help="control arm (--no-facts)")
    ap.add_argument("--key", default="id", help="join key (default: id)")
    ap.add_argument("--dump-flips", metavar="PATH",
                    help="write the discordant fired-subset questions to PATH "
                         "for hand-reading WHY each one moved")
    ap.add_argument("--by", metavar="FIELD",
                    help="also split the FIRED subset by this record field "
                         "(e.g. question_type, category). LOCALISING ONLY -- "
                         "slicing k ways multiplies the false-alarm rate, so "
                         "read these as where-to-look, never as a second "
                         "significance claim.")
    args = ap.parse_args()

    on_rows = {str(r[args.key]): r for r in load_rows(args.on)}
    off_rows = {str(r[args.key]): r for r in load_rows(args.off)}
    shared = sorted(set(on_rows) & set(off_rows))
    if not shared:
        raise SystemExit("no shared question ids -- are these the same benchmark?")
    print(f"\n  paired on {len(shared)} questions "
          f"(ON {len(on_rows)}, OFF {len(off_rows)})")

    # ---- 1. mechanism, BEFORE any score --------------------------------------
    fired = [q for q in shared if (on_rows[q].get("n_facts") or 0) > 0]
    n_facts_vals = [on_rows[q].get("n_facts") or 0 for q in shared]
    total_facts = sum(n_facts_vals)
    print(f"\n  ── mechanism (read this first) ──")
    print(f"  fired on          {len(fired)}/{len(shared)} questions "
          f"({100.0*len(fired)/len(shared):.1f}%)")
    print(f"  facts rendered    {total_facts} total, "
          f"{total_facts/max(len(fired),1):.1f} per fired question")
    leaked = [q for q in shared if (off_rows[q].get("n_facts") or 0) > 0]
    if leaked:
        print(f"  !! {len(leaked)} OFF-arm questions still rendered facts -- "
              f"the control arm is not clean, stop here")
    if not fired:
        print("\n  !! The tier never reached the reader. Any score delta here is\n"
              "     reader churn, and a flat result is a NO-OP BY CONSTRUCTION,\n"
              "     not a null result. Fix the plumbing before reading a number.")
        return
    if on_rows[shared[0]].get("gold_in_facts") is not None:
        checkable = [q for q in fired if on_rows[q].get("gold_in_facts") is not None]
        hits = sum(1 for q in checkable if on_rows[q].get("gold_in_facts"))
        if checkable:
            print(f"  gold_in_facts     {hits}/{len(checkable)} of fired "
                  f"({100.0*hits/len(checkable):.1f}%) -- the tier's own hit rate")

    # ---- 2. paired scores -----------------------------------------------------
    def pairs_for(qs: list[str]) -> list[tuple[bool, bool]]:
        out = []
        for q in qs:
            a, b = on_rows[q].get("correct"), off_rows[q].get("correct")
            if a is None or b is None:      # abstention/unjudged: no verdict to pair
                continue
            out.append((bool(a), bool(b)))
        return out

    not_fired = [q for q in shared if q not in set(fired)]
    print(f"\n  ── paired McNemar (ON minus OFF) ──")
    print(fmt("ALL questions", mcnemar(pairs_for(shared))))
    print(fmt("FIRED (facts rendered)", mcnemar(pairs_for(fired))))
    ctrl = mcnemar(pairs_for(not_fired))
    print(fmt("NOT FIRED [control]", ctrl))
    print("\n  The NOT FIRED row is the negative control: both arms saw identical\n"
          "  input there, so its spread IS this run pair's churn. It should sit\n"
          "  at z~0. If it does not, the arms differ by more than the facts flag\n"
          "  and the FIRED row cannot be attributed to the tier.")
    if abs(ctrl["z"]) >= 2.0:
        print("  !! control is NOT flat -- treat the comparison as void")

    # ---- 2b. optional localisation -------------------------------------------
    if args.by:
        groups: dict[str, list[str]] = {}
        for q in fired:
            groups.setdefault(str(on_rows[q].get(args.by)), []).append(q)
        print(f"\n  ── FIRED split by {args.by} (localising only) ──")
        for g in sorted(groups, key=lambda k: -len(groups[k])):
            print(fmt(str(g)[:24], mcnemar(pairs_for(groups[g]))))
        k = len(groups)
        print(f"\n  {k} slices at a 2sig threshold => ~{100*(1-0.95**k):.0f}% chance\n"
              f"  of at least one crossing under a pure-noise null. These say WHERE\n"
              f"  to look; only the pre-registered whole-subset test says WHETHER.")

    # ---- 3. optional flip dump ------------------------------------------------
    if args.dump_flips:
        flips = []
        for q in fired:
            a, b = on_rows[q].get("correct"), off_rows[q].get("correct")
            if a is None or b is None or bool(a) == bool(b):
                continue
            r = on_rows[q]
            flips.append({"id": q, "direction": "gained" if a else "lost",
                          "question": r.get("question"), "answer": r.get("answer"),
                          "n_facts": r.get("n_facts"),
                          "gold_in_facts": r.get("gold_in_facts"),
                          "ai_on": r.get("ai_answer"),
                          "ai_off": off_rows[q].get("ai_answer")})
        Path(args.dump_flips).write_text(json.dumps(flips, indent=2), encoding="utf-8")
        print(f"\n  {len(flips)} discordant fired questions → {args.dump_flips}")
        print("  Hand-read these. A real tier effect has a READABLE mechanism in\n"
              "  the gained/lost pairs; churn does not.")


if __name__ == "__main__":
    main()
