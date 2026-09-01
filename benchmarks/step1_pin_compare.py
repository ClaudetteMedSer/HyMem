#!/usr/bin/env python3
"""Step 1 read protocol of docs/plans/2026-09-01-beam-model-pin-pre-reg.md.

Implements §5 exactly and in its stated ORDER: D_self (the pinned judge's own
churn) is computed and printed BEFORE D_pin, because the band comes from the
churn and a band chosen after seeing the effect is not a band.

Written while C1 was still judging, before any score had been read. It computes
the pre-registered statistics; it does not choose them.

    step1_compare.py <B.json> <C1.json> <C2.json>
"""
import json
import statistics
import sys
from collections import defaultdict

POOL = {"EO", "IE", "IF", "KU", "MR", "PF", "SUM", "TR"}
CONTROL = {"ABS", "CR"}


def rows(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for c in d["conversations"]:
        for q in c["questions"]:
            out[(q["ability"], q["question"])] = q
    return d["metadata"], out


def gate(name, rs):
    """4.4: a rate above zero voids the comparison before scores are read."""
    silent0 = [k for k, q in rs.items() if q.get("rubric") and q.get("scores") == []]
    errs = [k for k, q in rs.items()
            if q.get("judge_error") or str(q.get("judge_raw", "")).startswith("[LLM_ERROR")]
    print(f"  {name}: {len(rs)} rows, silent-0 {len(silent0)}/{len(rs)}, "
          f"LLM_ERROR {len(errs)}/{len(rs)}")
    return silent0, errs


def main():
    b_meta, B = rows(sys.argv[1])
    c1_meta, C1 = rows(sys.argv[2])
    c2_meta, C2 = rows(sys.argv[3])

    print("=== provenance ===")
    for tag, m in (("B ", b_meta), ("C1", c1_meta), ("C2", c2_meta)):
        pr = m.get("prereg") or {}
        print(f"  {tag} judge={m.get('judge_model')} gold={m.get('judge_gold')} "
              f"extra={m.get('judge_extra_body')} calls={m.get('judge_calls')} "
              f"elapsed={m.get('elapsed_s', 0):.0f}s")
        print(f"     prereg={str(pr.get('blob'))[:12]} code={str(pr.get('code_commit'))[:8]} "
              f"dataset={m.get('dataset_revisions')}")

    print("\n=== GATE (before any score is compared) ===")
    bad = False
    for name, rs in (("C1", C1), ("C2", C2)):
        s0, er = gate(name, rs)
        bad = bad or bool(s0) or bool(er)
    keys = sorted(set(B) & set(C1) & set(C2))
    print(f"  rows matched across all three arms: {len(keys)}")
    if len(keys) != len(B):
        print(f"  ABORT: B has {len(B)} rows, only {len(keys)} match across arms.")
        return 3
    if bad:
        print("  GATE FAILED - the pin is broken; no score comparison is interpretable.")
        return 2
    print("  GATE PASSED (B's rates were 0/160 on both).")

    self_d = [C1[k]["score"] - C2[k]["score"] for k in keys]
    D_self = sum(1 for d in self_d if d != 0)
    print("\n=== D_self - the PINNED judge's own churn (read FIRST) ===")
    print(f"  D_self = {D_self}/{len(keys)} rows differ between C1 and C2")
    if D_self:
        sd_self = statistics.stdev(self_d)
        band = 2 * sd_self / len(keys) ** 0.5
        print(f"  SD_self = {sd_self:.6f}  ->  band = 2*SD_self/sqrt(n) = "
              f"+/-{band * 100:.4f}pp")
    else:
        sd_self, band = 0.0, 0.0
        print("  SD_self = 0.0000 - the pinned judge is score-deterministic, so ANY")
        print("  C1 != B row is attributable to model identity.")

    pin_d = [C1[k]["score"] - B[k]["score"] for k in keys]
    S = [k for k, d in zip(keys, pin_d) if d != 0]
    D_pin = len(S)
    mean_pin = statistics.fmean(pin_d)
    print("\n=== D_pin - the pin vs the alias ===")
    print(f"  D_pin = {D_pin}/{len(keys)}   delta-bar = {mean_pin * 100:+.4f}pp")

    print("\n=== VERDICT (fixed before counts) ===")
    if D_self == 0 and D_pin == 0:
        print("  PASS - deterministic and score-identical. The migration doc's")
        print("  'byte-path-equivalent' claim holds on this workload; B survives")
        print("  as a comparator across the pin. (Weak positive evidence per S1.)")
    elif D_self > 0 and abs(mean_pin) <= band and D_pin <= D_self:
        print("  PASS - the pin churns, but C1's distance from B is inside the")
        print(f"  pin's own churn ({abs(mean_pin) * 100:.4f}pp <= {band * 100:.4f}pp)")
        print(f"  and D_pin {D_pin} <= D_self {D_self}.")
    else:
        d_only = [C1[k]["score"] - B[k]["score"] for k in S]
        signs = {1 if d > 0 else -1 for d in d_only}
        kind = "FAIL-1 (rescale)" if len(signs) == 1 else "FAIL-2 (different interpreter)"
        print(f"  FAIL - {kind}")
        if len(signs) == 1:
            print("  Every differing row shares one sign: the pin is uniformly")
            print("  harsher/laxer, B's SHAPE is intact, delta-bar is the offset.")
            print("  -> pin adopted; B carries as a shape comparator only.")
        else:
            print("  Signs diverge: the pin is not shifted, it reads the rubric")
            print("  differently. -> pin adopted; B does not carry across at all.")
            print("  The pinned canonical becomes a stand-alone baseline with no")
            print("  ancestor: no 'improved by X pp' claim is available.")
        print(f"  (descriptive, never used to re-classify: spread of the {len(S)} "
              f"deltas min {min(d_only):+.4f} max {max(d_only):+.4f} "
              f"mean {statistics.fmean(d_only):+.4f})")

    print("\n=== per-ability (n=16, DESCRIPTIVE ONLY - one flip is 6.25pp) ===")
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])
    for k in keys:
        a = agg[k[0]]
        a[0] += 1
        a[1] += B[k]["score"]
        a[2] += C1[k]["score"]
        a[3] += C2[k]["score"]
        a[4] += 1 if C1[k]["score"] != B[k]["score"] else 0
    print(f"  {'ab':<5}{'arm':<5}{'n':>4}{'B':>9}{'C1':>9}{'C2':>9}{'d_pp':>9}{'moved':>7}")
    for ab in sorted(agg):
        n, sb, s1, s2, mv = agg[ab]
        arm = "pool" if ab in POOL else ("ctl" if ab in CONTROL else "?")
        print(f"  {ab:<5}{arm:<5}{n:>4}{sb / n:>9.4f}{s1 / n:>9.4f}{s2 / n:>9.4f}"
              f"{(s1 - sb) / n * 100:>+9.2f}{mv:>7}")
    for label, sel in (("POOL (8 abilities)", POOL), ("CONTROL ABS/CR", CONTROL),
                       ("OVERALL", None)):
        ks = [k for k in keys if sel is None or k[0] in sel]
        sb = statistics.fmean(B[k]["score"] for k in ks)
        s1 = statistics.fmean(C1[k]["score"] for k in ks)
        print(f"  {label:<22} n={len(ks):<4} B {sb:.4f}  C1 {s1:.4f}  "
              f"d {(s1 - sb) * 100:+.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
