#!/usr/bin/env python3
"""What difference can this harness actually resolve at n=500?

Written to re-baseline Plan C gate 4 after a run that spent 5.5h to discover
that both its arms were the same arm. That accident produced the thing the
project had never measured: two runs of an IDENTICAL configuration, 500
paired questions, so run-to-run noise can be estimated directly instead of
assumed to be small.

Two independent estimates, and they should agree:

  PAIRED   from the discordant questions of one same-arm pair. Immune to
           drift, because both runs happened within hours of each other.
  ERA      the spread of OVERALL across every comparable full-500 run since
           the answer/judge models last changed. Includes real drift, so it
           should be the LARGER of the two.

The gate bar this replaces was `OVERALL >= 70.0`, a single-run point estimate
from 2026-06-10 -- taken on `deepseek-chat`, for BOTH answer and judge, a
model hard-deprecated 2026-07-24. It cannot be reproduced at all, and it is
being used as a floor for runs on a different model.

The output is the minimum detectable effect: below it, a PASS and a FAIL are
the same measurement. Offline, reads artifacts and the registry only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# 1.96 sigma, two-sided alpha = 0.05.
Z95 = 1.959963985


def load(p) -> dict:
    return json.loads(Path(p).read_text())


def paired_discordance(a: dict, b: dict) -> dict:
    """McNemar counts over the questions both runs answered.

    b_only/c_only are the discordant cells: questions one run got right and
    the other did not. Under a true null they are exchangeable, and their
    SUM sets the resolution of the instrument however large n is."""
    ar = {r["question_id"]: r for r in a.get("per_question", [])}
    br = {r["question_id"]: r for r in b.get("per_question", [])}
    shared = sorted(set(ar) & set(br))
    if not shared:
        raise ValueError("the two runs share no question ids")
    a_only = sum(1 for q in shared
                 if ar[q].get("correct") and not br[q].get("correct"))
    b_only = sum(1 for q in shared
                 if br[q].get("correct") and not ar[q].get("correct"))
    n = len(shared)
    disc = a_only + b_only
    # SD of the DIFFERENCE in accuracy between two runs, in percentage points.
    sd_diff = 100.0 * math.sqrt(disc) / n if n else float("nan")
    return {
        "n": n,
        "a_only": a_only,
        "b_only": b_only,
        "discordant": disc,
        "discordant_pct": 100.0 * disc / n,
        "net_pp": 100.0 * (b_only - a_only) / n,
        "sd_diff_pp": sd_diff,
        # Two independent runs of one arm: Var(diff) = 2 Var(run).
        "sd_run_pp": sd_diff / math.sqrt(2),
        # |b-c| must exceed this for McNemar to reject at alpha=.05.
        "mde_pp": 100.0 * Z95 * math.sqrt(disc) / n if disc else 0.0,
    }


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial p for McNemar's test.

    Exact rather than the chi-square approximation because the discordant
    count is what it is -- 42 in the calibration pair, and far smaller on any
    per-ability subset, where the approximation is worst exactly when the
    claim rests on it most.

    Under the null the discordant questions split 50/50, so this is a sign
    test on b vs c and the CONCORDANT questions carry no information at all.
    That is the whole reason n=500 does not buy the resolution it looks like
    it buys."""
    n = b + c
    if n == 0:
        # Two arms that never disagree on any question. No evidence of a
        # difference, and equally no evidence of none -- p=1 says the first,
        # and the MDE beside it has to say the second.
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def era_spread(values: list[float]) -> dict:
    if len(values) < 2:
        return {"n": len(values), "mean": values[0] if values else None,
                "sd": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bar_risk(bar: float, mean: float, sd: float) -> float:
    """P(a run of an INERT arm falls below `bar`).

    A bar set at the centre of the distribution it is meant to floor fails
    half the time on a change that does nothing at all."""
    if sd is None or sd <= 0:
        return float("nan")
    return normal_cdf((bar - mean) / sd)


def report(pair: tuple[dict, dict] | None, era: list[float], bar: float,
           out=print) -> dict:
    res: dict = {}
    if pair is not None:
        d = paired_discordance(*pair)
        res["paired"] = d
        out("=== PAIRED estimate (two runs of one arm) ===")
        out(f"  shared questions: {d['n']}")
        out(f"  discordant: {d['discordant']} ({d['discordant_pct']:.1f}%)  "
            f"[{d['a_only']} / {d['b_only']}]")
        out(f"  net difference: {d['net_pp']:+.1f}pp")
        out(f"  SD of a run-to-run difference: {d['sd_diff_pp']:.2f}pp")
        out(f"  implied SD of a single run:    {d['sd_run_pp']:.2f}pp")
        out(f"  minimum detectable effect (McNemar, a=.05): "
            f"{d['mde_pp']:.2f}pp")
        out("")

    e = era_spread(era)
    res["era"] = e
    out("=== ERA estimate (comparable full-500 runs, same models) ===")
    if e["sd"] is None:
        out(f"  only {e['n']} run(s); no spread to estimate")
        return res
    out(f"  n={e['n']}  mean {e['mean']:.2f}  SD {e['sd']:.2f}  "
        f"range {e['min']:.1f}-{e['max']:.1f}")
    out("")
    out("=== the bar ===")
    risk = bar_risk(bar, e["mean"], e["sd"])
    res["bar"] = {"bar": bar, "risk_inert_fails": risk}
    out(f"  bar = {bar}")
    out(f"  era mean = {e['mean']:.2f}  ->  the bar sits "
        f"{(e['mean'] - bar) / e['sd']:+.2f} SD from the centre")
    out(f"  P(an INERT arm scores below the bar) = {risk:.0%}")
    if risk > 0.20:
        out("")
        out("  MIS-CALIBRATED: this is a central estimate being used as a")
        out("  floor. A change with no effect whatsoever fails it roughly")
        out(f"  {risk:.0%} of the time, so a FAIL carries almost no")
        out("  information about the lever. Use a PAIRED contemporaneous")
        out("  control and a margin from the discordance above.")
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pair", nargs=2, metavar=("A", "B"),
                   help="two artifacts of the SAME arm (null calibration)")
    p.add_argument("--era", nargs="+", type=float, required=True,
                   help="OVERALL of each comparable run")
    p.add_argument("--bar", type=float, default=70.0)
    args = p.parse_args()
    pair = (load(args.pair[0]), load(args.pair[1])) if args.pair else None
    report(pair, list(args.era), args.bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
