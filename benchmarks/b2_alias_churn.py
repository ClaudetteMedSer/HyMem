#!/usr/bin/env python3
"""Read protocol of docs/plans/2026-09-01-alias-churn-b2-pre-reg.md.

Answers one question: does the ALIAS judge reproduce itself?

Step 1 measured the PIN's churn at 7/160 and passed. That PASS is compatible
with two opposite worlds and blind between them -- the alias churning too (so
the gold-delta zero-width band was a small-sample draw), or the alias being
genuinely deterministic while the pin is not (so the pin is the noisier
instrument and the migration doc's equivalence claim fails on the sampling
axis). Only a direct measurement of the alias separates them.

Written and committed while B2 was still judging, before any B2-vs-B row was
compared.

    b2_alias_churn.py <B.json> <B2.json>
"""
import json
import math
import statistics
import sys

CONTROL = {"ABS", "CR"}
GOLD_DELTA_EFFECT = -0.00582   # the §8 primary, -0.582pp
PIN_D_SELF = 7                 # Step 1, for the descriptive comparison


def rows(path):
    with open(path) as f:
        d = json.load(f)
    return d["metadata"], {(q["ability"], q["question"]): q
                           for c in d["conversations"] for q in c["questions"]}


def main():
    b_meta, B = rows(sys.argv[1])
    n_meta, B2 = rows(sys.argv[2])

    print("=== provenance ===")
    for tag, m in (("B ", b_meta), ("B2", n_meta)):
        pr = m.get("prereg") or {}
        print(f"  {tag} judge={m.get('judge_model')} gold={m.get('judge_gold')} "
              f"extra={m.get('judge_extra_body')} elapsed={m.get('elapsed_s', 0):.0f}s")
        print(f"     prereg={str(pr.get('blob'))[:12]} code={str(pr.get('code_commit'))[:8]} "
              f"dataset={m.get('dataset_revisions')}")

    print("\n=== GATE (before any comparison) ===")
    s0 = [k for k, q in B2.items() if q.get("rubric") and q.get("scores") == []]
    er = [k for k, q in B2.items()
          if q.get("judge_error") or str(q.get("judge_raw", "")).startswith("[LLM_ERROR")]
    print(f"  B2: {len(B2)} rows, silent-0 {len(s0)}/{len(B2)}, LLM_ERROR {len(er)}/{len(B2)}")
    keys = sorted(set(B) & set(B2))
    print(f"  rows matched: {len(keys)}")
    if s0 or er:
        print("  GATE FAILED - no churn reading is interpretable.")
        return 2
    if len(keys) != len(B):
        print(f"  ABORT: {len(B)} rows in B, {len(keys)} matched.")
        return 3
    print("  GATE PASSED.")

    deltas = [B2[k]["score"] - B[k]["score"] for k in keys]
    moved = [k for k, d in zip(keys, deltas) if d != 0]
    D_alias = len(moved)
    ctl = [k for k in keys if k[0] in CONTROL]
    ctl_d = [B2[k]["score"] - B[k]["score"] for k in ctl]
    D_ctl = sum(1 for d in ctl_d if d != 0)

    print("\n=== D_alias - does the alias reproduce itself? ===")
    print(f"  D_alias = {D_alias}/{len(keys)} rows differ between B and B2")
    print(f"  control arm ABS/CR: {D_ctl}/{len(ctl)}")
    if moved:
        print("  rows that moved:")
        for k in moved:
            print(f"    {k[0]:<4} B={B[k]['score']:.4f} B2={B2[k]['score']:.4f} "
                  f"{'ctl' if k[0] in CONTROL else 'pool'}")
    sd_all = statistics.stdev(deltas)
    sd_ctl = statistics.stdev(ctl_d)
    print(f"  SD_alias(all)={sd_all:.6f}   SD_alias_ctl={sd_ctl:.6f}")

    print("\n=== VERDICT (fixed before counts) ===")
    if D_alias == 0:
        print("  WORLD 2 - the alias reproduces itself exactly at n=160, where the")
        print(f"  pin moves {PIN_D_SELF}/160. At the pin's rate a zero draw here would be")
        print(f"  a {(1 - PIN_D_SELF / 160) ** 160:.5f} event, so this is not a sampling accident.")
        print("  Consequences, per §5:")
        print("   - the migration doc's byte-path-equivalence claim is FALSE on the")
        print("     sampling axis and must be corrected;")
        print("   - §8's zero-width band stands as a real property of the alias;")
        print("   - pin adoption becomes a DELIBERATE TRADE (witnessability against a")
        print("     measured loss of reproducibility), not a free upgrade, and cannot")
        print("     be inferred from Step 1's PASS.")
    else:
        print(f"  WORLD 1 - the alias churns ({D_alias}/160). §8's SD_ctl = 0 was a")
        print("  small-sample draw, not determinism.")
        se = sd_ctl / math.sqrt(128)
        band = 2 * se
        inside = abs(GOLD_DELTA_EFFECT) <= band
        print(f"\n  Gold-delta primary recomputed in §8's OWN formula")
        print(f"  (SE = SD_alias_ctl/sqrt(128), band = 2*SE):")
        print(f"    SD_alias_ctl = {sd_ctl:.6f} -> band = +/-{band * 100:.4f}pp")
        print(f"    effect = {GOLD_DELTA_EFFECT * 100:.3f}pp -> "
              f"{'INSIDE' if inside else 'OUTSIDE'}")
        if inside:
            print("  -> REBASE REQUIRED is NOT supported at the measured variance.")
            print("     It must be re-derived under a fresh pre-registration before")
            print("     Step 2 can be justified on those grounds.")
        else:
            print("  -> the gold-delta verdict survives its own formula at this variance.")

    print("\n=== descriptive only (never used to reclassify) ===")
    print(f"  alias {D_alias}/160 vs pin {PIN_D_SELF}/160")
    if D_alias > PIN_D_SELF:
        print("  The ALIAS is the noisier instrument: the pin's reproducibility cost")
        print("  is negative, which would strengthen pin adoption. Reported, not decided.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
