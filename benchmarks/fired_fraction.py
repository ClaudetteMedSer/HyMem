#!/usr/bin/env python3
"""Measure `f` — the fraction of questions a lever actually moves — for free.

`concentration_model` showed that scoring gate 4 on the subset a lever touches
would cut the MDE by sqrt(f), and that everything turns on f: at f=25% the gain
is 2x and gate 4 becomes worth running; near f=1 it is 1x and gate 4 cannot
resolve the lever at any price. It also showed that f is NOT measurable from a
same-arm pair, and that the indicator gate 4 uses today (`n_episodes` differs)
is saturated on 84% of the run and cannot measure it either.

f is a property of RETRIEVAL. It does not depend on what the reader says or on
how the judge scores it. So it can be measured with two `--retrieval-only`
runs, one per arm, which make no answer call and no judge call -- skipping
essentially the whole cost of a benchmark while answering the question that
decides whether the benchmark is worth buying.

IT IS NOT FREE, AND THE SAVING IS NOT WHERE IT LOOKS. `--retrieval-only`
skips the reader and the judge -- but NOT the dream, and episode granularity
is a DREAM-TIME lever: it changes how episodes are cut, so the dream is the
one thing that cannot be skipped when measuring its `f`. Measured across the
archive, no-dream 500-question runs take ~0.2h while dreamed ones take
2.2-2.8h, and the no-dream runs already include all 500 answer and 500 judge
calls. Dreaming is therefore ~93% of the wall clock, and dropping the reader
and judge saves roughly 7% of a run, not 90%.

THE SAVING IS IN `n`, NOT IN THE MODE. `f` is a proportion, and the decision
it feeds is coarse -- concentrate (f <= 25%) or retire (f >= 75%). A
50-question sample resolves that comfortably, which is why this module reports
a Wilson interval rather than a point: run the smallest n whose interval
clears the threshold, and escalate only if it straddles. 50 questions per arm
is ~4% of a full gate-4 pair.

WHAT THIS DOES NOT TELL YOU. That the lever moved the reader's input on a
question is not evidence it moved the ANSWER, still less the verdict. f bounds
what a subset-scored gate could see; it says nothing about whether there is an
effect to see. A high f means concentration buys nothing; a low f means it
buys a lot, and neither is a result about episode granularity.

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
from concentration_model import gain_curve  # noqa: E402
from lme_noise_model import Z95  # noqa: E402
from run_registry import ARM_EVIDENCED, arm_evidence, is_retrieval_only  # noqa: E402

LEVER = "episode_granularity_enabled"


def fired_fraction(a: dict, b: dict) -> dict:
    """Share of shared questions whose reader prompt differs between arms.

    Keyed on `context_sha`, which is the rendered prompt hashed -- not the
    count fields, which saturate. A run predating the hash cannot answer this
    and says so rather than falling back to a fingerprint that is blind on
    five-sixths of the questions."""
    ar = {r["question_id"]: r for r in a.get("per_question", [])}
    br = {r["question_id"]: r for r in b.get("per_question", [])}
    shared = sorted(set(ar) & set(br))
    if not shared:
        raise ValueError("the two runs share no question ids")
    missing = [q for q in shared
               if not ar[q].get("context_sha") or not br[q].get("context_sha")]
    if missing:
        return {"available": False, "shared": len(shared),
                "reason": (f"{len(missing)}/{len(shared)} shared questions "
                           "carry no context_sha (run predates 5acedf7)")}
    moved = [q for q in shared if ar[q]["context_sha"] != br[q]["context_sha"]]
    n = len(shared)
    k = len(moved)
    return {"available": True, "shared": n, "moved": k, "f": k / n,
            # A binomial interval, because f is estimated from n questions and
            # the decision it feeds (run gate 4 or retire it) turns on which
            # side of ~25% it falls.
            "ci95": _wilson(k, n)}


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / d
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cost(a: dict, b: dict) -> dict:
    """What the two retrieval-only runs actually spent, as they recorded it."""
    out = {"llm_calls": 0, "distill_calls": 0, "answer_calls": 0,
           "judge_calls": 0, "recorded": True}
    for art in (a, b):
        rc = art.get("retrieval_cost")
        if not rc:
            out["recorded"] = False
            continue
        for k in ("llm_calls", "distill_calls", "answer_calls", "judge_calls"):
            v = rc.get(k)
            if v is not None:
                out[k] += v
    return out


def report(a: dict, b: dict, lever: str = LEVER, out=print) -> dict:
    for name, art in (("A", a), ("B", b)):
        if not is_retrieval_only(art):
            out(f"  REFUSED — arm {name} is not a --retrieval-only run.")
            out("  f must be measured on runs that made no answer call; a")
            out("  full run would work too but there would be no point, and")
            out("  mixing the two invites quoting a cheap number that a")
            out("  different kind of run produced.")
            return {"refused": True, "arm": name}

    verdict_arm, note, confounds = arm_evidence(
        a.get("config", {}), b.get("config", {}), lever,
        ignore=("elapsed_s", "total_tokens", "answer_calls", "judge_calls"))
    out("=== arm evidence ===")
    out(f"  [{verdict_arm}] {note}")
    if verdict_arm != ARM_EVIDENCED:
        out("")
        out("  REFUSED — f is the fraction of questions THIS LEVER moves. A")
        out("  pair that cannot evidence which arm is which measures the")
        out("  fraction that retrieval churn moves, which is a different")
        out("  number with the same shape.")
        return {"refused": True, "arm": verdict_arm}
    if confounds:
        out(f"  confounds: {', '.join(confounds)}")

    ff = fired_fraction(a, b)
    out("")
    out("=== f — questions whose reader prompt the lever moved ===")
    if not ff["available"]:
        out(f"  UNAVAILABLE — {ff['reason']}.")
        out("  The count fields are not a fallback: they saturate at the")
        out("  retrieval cap on ~84% of questions.")
        return {"fired": ff}
    lo, hi = ff["ci95"]
    out(f"  {ff['moved']}/{ff['shared']} = {ff['f']:.0%}  "
        f"(95% CI {lo:.0%}–{hi:.0%})")
    out("")
    out("=== what that means for a subset-scored gate 4 ===")
    g = gain_curve(ff["f"]) if ff["f"] > 0 else {"available": False}
    if not g["available"]:
        out("  The lever moved NO question's prompt. There is nothing for a")
        out("  fired subset to contain, and nothing for gate 4 to detect:")
        out("  the arms are the same experiment.")
        return {"fired": ff, "gain": g, "cost": cost(a, b)}
    out(f"  gain {g['gain']:.2f}x — MDE 2.54pp would become "
        f"{2.54 / g['gain']:.2f}pp")
    out(f"  break-even leakage {g['breakeven_leakage']:.0%}")
    if ff["f"] <= 0.25:
        out("  NARROW. Concentration is worth having; a subset-scored gate 4")
        out("  resolves materially more than the full 500.")
    elif ff["f"] >= 0.75:
        out("  BROAD. Concentration buys almost nothing here, so gate 4 at")
        out("  2.5pp is the best this harness can do for this lever — which")
        out("  is an argument for retiring it, not for re-running it.")
    else:
        out("  MIDDLING. Some gain, not a transformation. Weigh it against")
        out("  the spend rather than treating it as a fix.")
    out("")
    out("  f bounds what a subset-scored gate could SEE. It is not evidence")
    out("  the lever changes any answer, and must never be reported as one.")

    c = cost(a, b)
    out("")
    out("=== what this measurement cost ===")
    out("  NOTE: --retrieval-only does not skip the DREAM, and episode")
    out("  granularity is a dream-time lever. Dreaming is ~93% of a run's")
    out("  wall clock, so the mode saves the reader and judge only. The")
    out("  saving that matters is a small n, which the interval above is")
    out("  there to let you choose.")
    if not c["recorded"]:
        out("  NOT RECORDED — at least one artifact predates the cost block.")
    else:
        out(f"  answer calls: {c['answer_calls']}   "
            f"judge calls: {c['judge_calls']}")
        out(f"  retrieval-path LLM calls: {c['llm_calls']} "
            f"(of which distillation {c['distill_calls']})")
    return {"fired": ff, "gain": g, "cost": c}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--lever", default=LEVER)
    args = p.parse_args()
    res = report(json.loads(Path(args.a).read_text()),
                 json.loads(Path(args.b).read_text()), args.lever)
    return 2 if res.get("refused") else 0


if __name__ == "__main__":
    raise SystemExit(main())
