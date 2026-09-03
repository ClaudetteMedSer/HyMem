#!/usr/bin/env python3
"""Would scoring gate 4 on the questions the lever TOUCHED beat scoring all 500?

`lme_noise_model` put the harness's resolution at 2.54pp and `churn_decompose`
showed there is no fix for the churn that produces it: the judge is stable to
within 1.6%, our retrieval is not what flips verdicts, and the reader already
runs at temperature=0.0. LME-S caps n at 500, so more questions are not
available either.

What IS available is spending the questions better. McNemar rejects when the
net questions moved exceeds `Z * sqrt(D)`, where D counts only the DISCORDANT
questions. If the lever's effect is confined to a subset S, then scoring S
alone keeps the whole numerator -- every question the lever moved is in S by
assumption -- while discarding the churn contributed by everything outside it.
The gain is therefore exactly

    MDE(S) / MDE(all)  =  sqrt( D_S / D_all )

expressed in the same overall percentage points, so the two are comparable.
Both terms are measurable, and this module measures them.

IT MUST BE CALIBRATED ON A SAME-ARM PAIR, and it refuses anything else. On two
runs of one arm every discordant question is churn by construction, which is
what makes D_S an estimate of the noise a subset carries. Run it on a real A/B
and D_S contains the signal too, so the projected gain would be inflated by
the very effect it claims to be able to detect.

TWO THINGS IT CANNOT TELL YOU, both stated in the output:

  * LEAKAGE. The gain above assumes every question the lever moves is inside
    S. Effect that lands outside is lost from the numerator, and a fraction
    `lam` outside multiplies the gain by (1 - lam). No same-arm pair can
    measure `lam` -- it is a fact about the lever, not about the noise.
  * CONTAMINATION. An indicator fires on a same-arm pair too, purely from
    retrieval churn. Those questions join S with noise and no signal. The
    null firing rate is the floor on that and is reported.

Offline: reads two artifacts, makes no call.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
from churn_decompose import CONTEXT_FIELDS, is_scored  # noqa: E402
from lme_noise_model import Z95  # noqa: E402
from run_registry import ARM_SAME, arm_evidence  # noqa: E402

LEVER = "episode_granularity_enabled"


def _episode_count(ra, rb):
    return (ra.get("n_episodes") is not None
            and ra.get("n_episodes") != rb.get("n_episodes"))


def _context_sha(ra, rb):
    return bool(ra.get("context_sha")) and bool(rb.get("context_sha")) \
        and ra["context_sha"] != rb["context_sha"]


def _any_context_field(ra, rb):
    return any(ra.get(k) != rb.get(k) for k in CONTEXT_FIELDS)


# Candidate fired-indicators, weakest first. `episode_count` is what
# `guard_score.fired_subset` uses today and is a LOWER BOUND on what the lever
# touched: a re-cut yielding the same NUMBER of different episodes is invisible
# to it. `context_sha` is exact but only exists on runs since 5acedf7.
INDICATORS = {
    "episode_count": _episode_count,
    "any_context_field": _any_context_field,
    "context_sha": _context_sha,
}


def subset_stats(a_rows: dict, b_rows: dict, fires) -> dict:
    """Discordance inside and outside the indicator's subset.

    On a same-arm pair every discordant question is churn, so `d_in` is the
    noise the subset carries -- the denominator a subset-scored gate would
    face."""
    shared = sorted(q for q in set(a_rows) & set(b_rows)
                    if is_scored(a_rows[q]) and is_scored(b_rows[q]))
    inside = [q for q in shared if fires(a_rows[q], b_rows[q])]
    inside_set = set(inside)

    def disc(qs):
        return sum(1 for q in qs
                   if bool(a_rows[q]["correct"]) != bool(b_rows[q]["correct"]))

    d_all = disc(shared)
    d_in = disc(inside)
    n = len(shared)
    n_in = len(inside)
    return {
        "n": n,
        "n_in": n_in,
        "fire_rate": n_in / n if n else None,
        "d_all": d_all,
        "d_in": d_in,
        "d_out": d_all - d_in,
        # Churn per question inside vs outside. If the indicator selects
        # questions that are noisier than average -- and "retrieval moved" is
        # exactly the kind of thing that would -- the gain shrinks, because
        # the subset keeps more of the churn than its size suggests.
        "churn_in": d_in / n_in if n_in else None,
        "churn_out": ((d_all - d_in) / (n - n_in)) if n - n_in else None,
    }


def projection(s: dict, n_total: int) -> dict:
    """MDE of a subset-scored gate, in the SAME overall percentage points.

    Both are `100 * Z * sqrt(D) / N` with the FULL N in the denominator: the
    net questions the lever moves is an absolute count, so expressing the
    subset's MDE against the subset's own n would quote a bigger number for a
    better instrument."""
    d_all, d_in = s["d_all"], s["d_in"]
    mde_all = 100.0 * Z95 * math.sqrt(d_all) / n_total if d_all else None
    if not s["n_in"]:
        return {"available": False,
                "reason": "the indicator fires on no shared question"}
    if d_in == 0:
        # sqrt(0) makes the projected MDE 0, which reads as a gate of
        # unlimited resolution. It is the opposite: a subset with no observed
        # churn has no estimate of its churn, and the true value is somewhere
        # under the rule of three.
        return {"available": False, "mde_all": mde_all,
                "reason": (f"no discordant question among the {s['n_in']} the "
                           "indicator fires on, so the subset's churn is "
                           "unestimated -- not zero")}
    mde_in = 100.0 * Z95 * math.sqrt(d_in) / n_total
    return {
        "available": True,
        "mde_all": mde_all,
        "mde_sub": mde_in,
        "gain": mde_all / mde_in if mde_in else None,
        "retained_discordance": d_in / d_all if d_all else None,
        # Effect leaking outside S costs the numerator directly.
        "breakeven_leakage": 1.0 - (mde_in / mde_all) if mde_all else None,
    }


def report(a: dict, b: dict, lever: str = LEVER, out=print) -> dict:
    verdict_arm, note, _ = arm_evidence(
        a.get("config", {}), b.get("config", {}), lever)
    out("=== calibration pair ===")
    out(f"  [{verdict_arm}] {note}")
    if verdict_arm != ARM_SAME:
        out("")
        out("  REFUSED — this projection is only valid on two runs of ONE")
        out("  arm. On a real A/B the discordance it reads as churn contains")
        out("  the lever's effect as well, so every gain below would be")
        out("  inflated by the very thing it claims to be able to detect.")
        return {"refused": True, "arm": verdict_arm}

    ar = {r["question_id"]: r for r in a.get("per_question", [])}
    br = {r["question_id"]: r for r in b.get("per_question", [])}
    results = {}
    base = subset_stats(ar, br, lambda ra, rb: True)
    out("")
    out(f"  shared, scored questions: {base['n']}   "
        f"discordant: {base['d_all']} "
        f"({100.0 * base['d_all'] / base['n']:.1f}%, all of it churn)")
    out(f"  full-500 MDE: "
        f"{100.0 * Z95 * math.sqrt(base['d_all']) / base['n']:.2f}pp")

    for name, fires in INDICATORS.items():
        s = subset_stats(ar, br, fires)
        p = projection(s, base["n"])
        results[name] = {"stats": s, "projection": p}
        out("")
        out(f"=== indicator: {name} ===")
        if not s["n_in"]:
            out("  fires on nothing here — no projection.")
            continue
        out(f"  fires on {s['n_in']}/{s['n']} questions "
            f"({s['fire_rate']:.0%}) with NO lever set: that is the")
        out("  contamination floor — questions that would join the subset")
        out("  carrying churn and no signal.")
        out(f"  churn inside {s['churn_in']:.1%} vs outside "
            f"{s['churn_out']:.1%}" if s["churn_out"] is not None
            else f"  churn inside {s['churn_in']:.1%}")
        if not p["available"]:
            out(f"  NO PROJECTION — {p['reason']}.")
            continue
        out(f"  discordance retained: {p['retained_discordance']:.0%} "
            f"of {s['d_all']}")
        out(f"  projected MDE if the gate scored this subset: "
            f"{p['mde_sub']:.2f}pp  (vs {p['mde_all']:.2f}pp)")
        out(f"  gain: {p['gain']:.2f}x")
        out(f"  BREAK-EVEN LEAKAGE: {p['breakeven_leakage']:.0%}. If more "
            "than that share of the")
        out("  lever's effect lands OUTSIDE this subset, scoring the subset "
            "is WORSE")
        out("  than scoring all 500. No same-arm pair can measure it — it is "
            "a fact")
        out("  about the lever, not about the noise.")
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--lever", default=LEVER)
    args = p.parse_args()
    res = report(json.loads(Path(args.a).read_text()),
                 json.loads(Path(args.b).read_text()), args.lever)
    return 2 if isinstance(res, dict) and res.get("refused") else 0


if __name__ == "__main__":
    raise SystemExit(main())
