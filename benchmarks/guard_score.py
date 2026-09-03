#!/usr/bin/env python3
"""Plan C gate 4 — score the LME full guard PAIRED, in the order it requires.

The 2026-08-30/31 guard pair produced clean numbers (71.0 vs 71.0) that could
not be read, because neither artifact recorded which arm it was. The scores
were correct and worthless: a real null and two runs of one arm look identical.

So this scorer enforces the reading order the re-run was pre-registered with,
rather than leaving it to whoever runs it. `report()` asks `arm_evidence()`
first and RETURNS before computing any accuracy if the pair cannot evidence its
own contrast. That is deliberate and structural: a number you have already seen
cannot be un-seen, and "I checked provenance afterwards" is how the last pair
came to be quoted in three documents before anyone asked which arm was which.

It also reports the FIRED subset, which the previous guard could not. `6543ee6`
records `n_episodes` per question for the reason `n_facts` already was: E1's
all-800 net read NULL while the subset where the tier actually reached the
reader read -2.9pp (p=0.024). An unconditional all-500 net is not evidence of
no effect unless you can also show the tier reached the reader.

READ THE FIRED SUBSET'S LIMIT HONESTLY. `n_episodes` counts the episodes handed
to the reader; the granularity lever changes how episodes are CUT, so two arms
can hand over the same NUMBER of different episodes. The subset is therefore a
LOWER BOUND on the questions the lever touched, not the set of them, and a null
on it is correspondingly weaker evidence than a null on a real fired-indicator.
It is reported, never gated.

THE BAR IS NOT A CONSTANT. It used to be: `OVERALL >= 70.0` with an MS floor
of 51.9, both from ONE 2026-06-10 run taken on `deepseek-chat` for answer AND
judge -- a model hard-deprecated 2026-07-24, so the canonical could not be
reproduced at all. Measured against nine comparable runs, 70.0 sat +0.23 SD
from the centre of its own era: an arm that does nothing whatsoever failed it
41% of the time, and jointly the two bars failed an inert lever about half the
time. That is the same defect as gate 3 and the 71.0-vs-71.0 guard pair in a
third form -- not a bar a no-effect change passes, but one it fails at random,
which is worse only because it invites a re-run.

So the contrast is PAIRED against the OFF arm of the same session, which both
arms already are. McNemar on the shared per-question outcomes, exact rather
than chi-square, and REGRESSION only when ON is worse AND the test rejects.

AND THE NEGATIVE VERDICT CARRIES ITS OWN RESOLUTION. Only the discordant
questions inform McNemar, so 500 questions at 8.4% churn resolve 2.54pp and no
more; the concordant 458 buy nothing. The gate therefore never reports "no
regression" -- it reports "no regression larger than X pp", with X computed
from the run in hand. A gate that cannot state what it would have missed is
how a null at 2.5pp resolution came to be read as evidence a lever was
harmless.

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
from lme_noise_model import Z95, mcnemar_exact_p  # noqa: E402
from run_registry import ARM_EVIDENCED, arm_evidence  # noqa: E402

# ------------------------------------------------------------- pre-registered
# Gate 4 is NON-REGRESSION ONLY. Not a tuning signal: a change with no effect
# clears it too.
#
# It is scored PAIRED, against the OFF arm of the same session -- never
# against a historical constant. The constants it used to carry,
# `OVERALL >= 70.0` and an MS floor of 51.9, both came from ONE run
# (longmemeval-v2-hymem-20260610T094858Z-seed0, 2026-06-10) taken on
# `deepseek-chat` for BOTH answer and judge, a model hard-deprecated
# 2026-07-24. They were not reproducible, and 70.0 sat +0.23 SD from the
# centre of the era it was floored against: an arm that does nothing at all
# failed that bar 41% of the time. See the RE-BASELINING block in
# additional_planning.md and `benchmarks/lme_noise_model.py`.
ALPHA = 0.05
MS_KEY = "multi-session"
LEVER = "episode_granularity_enabled"

# Confounds that break the pairing outright rather than merely muddying it.
# A paired test assumes the two arms differ in the lever and the lever only;
# two different answer models is not a confound to note in passing, it is a
# different experiment.
FATAL_CONFOUNDS = ("answer_model", "judge_model", "scale", "sample", "seed")


def accuracy(rows) -> float | None:
    rows = list(rows)
    if not rows:
        return None
    return 100.0 * sum(1 for r in rows if r.get("correct")) / len(rows)


def by_id(artifact) -> dict:
    return {r["question_id"]: r for r in artifact.get("per_question", [])}


def fired_subset(a_rows: dict, b_rows: dict) -> dict:
    """Questions where the episode pool the reader saw differs in SIZE.

    Returns a dict with `available` False when either arm predates `6543ee6`
    and records no episode count -- which is a gap in the instrument, not a
    finding about the lever, and must never be reported as "no effect"."""
    shared = sorted(set(a_rows) & set(b_rows))
    missing = [q for q in shared
               if "n_episodes" not in a_rows[q] or "n_episodes" not in b_rows[q]]
    if missing:
        return {"available": False, "shared": len(shared),
                "reason": f"{len(missing)}/{len(shared)} shared questions record "
                          f"no n_episodes (arm predates 6543ee6)"}
    fired = [q for q in shared
             if a_rows[q]["n_episodes"] != b_rows[q]["n_episodes"]]
    same = [q for q in shared if q not in set(fired)]
    return {
        "available": True,
        "shared": len(shared),
        "fired": len(fired),
        "a_fired": accuracy(a_rows[q] for q in fired),
        "b_fired": accuracy(b_rows[q] for q in fired),
        "a_same": accuracy(a_rows[q] for q in same),
        "b_same": accuracy(b_rows[q] for q in same),
        "a_episodes_total": sum(a_rows[q]["n_episodes"] for q in shared),
        "b_episodes_total": sum(b_rows[q]["n_episodes"] for q in shared),
    }


def is_scored(r: dict) -> bool:
    """D3: a judge that never answered is UNSCORED, not wrong.

    `accuracy` above reads a None verdict as falsy, which is right for a
    headline rate. It is wrong here: an unscored row has no outcome to pair,
    and treating it as a miss on both arms would pad the CONCORDANT count --
    the one that carries no information but does divide the net."""
    return r.get("correct") is not None


def paired_test(off_rows: dict, on_rows: dict, subset=None) -> dict:
    """McNemar, ON against the contemporaneous OFF arm, on shared questions.

    `regressed` counts questions OFF got right and ON got wrong; `gained` the
    reverse. Only those two cells enter the test -- which is why the MDE
    below is set by the churn and not by n, and why it must be reported
    whatever the verdict."""
    shared = sorted(q for q in set(off_rows) & set(on_rows)
                    if is_scored(off_rows[q]) and is_scored(on_rows[q])
                    and (subset is None or subset(off_rows[q], on_rows[q])))
    regressed = sum(1 for q in shared
                    if off_rows[q]["correct"] and not on_rows[q]["correct"])
    gained = sum(1 for q in shared
                 if on_rows[q]["correct"] and not off_rows[q]["correct"])
    n = len(shared)
    disc = regressed + gained
    return {
        "n": n,
        "regressed": regressed,
        "gained": gained,
        "discordant": disc,
        "net_pp": 100.0 * (gained - regressed) / n if n else None,
        "p": mcnemar_exact_p(regressed, gained),
        # |b-c| must exceed 1.96*sqrt(disc) for the test to reject. Below this
        # a PASS and a FAIL are the same measurement.
        "mde_pp": 100.0 * Z95 * math.sqrt(disc) / n if disc and n else None,
    }


def paired_verdict(t: dict) -> tuple[str, str]:
    """(verdict, sentence). REGRESSION only when ON is worse AND it rejects.

    Requiring the direction as well as the p-value makes this a one-sided
    test at alpha/2, which is conservative in the direction a non-regression
    gate should be conservative in.

    The negative verdict NEVER says "no regression". It says no regression
    LARGER THAN the MDE, because that is the only thing the instrument can
    support -- a gate that cannot state its own resolution is how 71.0 vs
    71.0 came to be read as evidence."""
    if t["n"] == 0:
        return "INCOMPLETE", "no shared, scored questions to pair"
    mde = t["mde_pp"]
    if t["net_pp"] < 0 and t["p"] < ALPHA:
        return "REGRESSION", (
            f"ON is worse by {abs(t['net_pp']):.1f}pp "
            f"(p={t['p']:.4f}, {t['regressed']} lost / {t['gained']} gained)")
    if mde is None:
        return "NO REGRESSION DETECTED", (
            "the two arms agreed on every scored question — which bounds the "
            "effect at 0pp and is also what two runs of ONE arm would look "
            "like; read it with the arm evidence above, not alone")
    return "NO REGRESSION DETECTED", (
        f"net {t['net_pp']:+.1f}pp, p={t['p']:.3f} — i.e. NO REGRESSION "
        f"LARGER THAN {mde:.1f}pp, which is all {t['discordant']} discordant "
        f"questions can support. Not 'no regression'.")


def report(a: dict, b: dict, lever: str = LEVER, out=print) -> tuple[str, dict]:
    """Score the pair. Returns (verdict, detail).

    Verdict INCOMPLETE means the pair cannot evidence its own contrast, and in
    that case NO accuracy is computed or printed -- see the module docstring."""
    verdict_arm, note, confounds = arm_evidence(
        a.get("config", {}), b.get("config", {}), lever)
    out(f"=== arm evidence — {lever} ===")
    out(f"  [{verdict_arm}] {note}")
    if confounds:
        out(f"  confounds (other keys that also moved): {', '.join(confounds)}")
    fatal = [k for k in confounds if k in FATAL_CONFOUNDS]
    if fatal and verdict_arm == ARM_EVIDENCED:
        out("")
        out(f"  INCOMPLETE — the scores are NOT read. {', '.join(fatal)} "
            "also moved.")
        out("  A paired test assumes the arms differ in the lever and the")
        out("  lever only. Two answer models, or two question samples, is")
        out("  not a confound to note in passing — it is a different")
        out("  experiment, and the pairing that makes this gate readable")
        out("  does not hold across it.")
        return "INCOMPLETE", {"arm": verdict_arm, "confounds": confounds,
                              "fatal": fatal}
    if verdict_arm != ARM_EVIDENCED:
        out("")
        out("  INCOMPLETE — the scores are NOT read.")
        out("  A number that cannot say which arm produced it cannot discharge")
        out("  a gate on that lever, and reading it anyway is how the previous")
        out("  guard pair came to be quoted before anyone asked.")
        return "INCOMPLETE", {"arm": verdict_arm, "confounds": confounds}

    a_rows, b_rows = by_id(a), by_id(b)
    # A is the arm recording lever=False; B the arm recording True. Taken from
    # the config blocks, never from the order of the arguments or the stems.
    if a["config"][lever]:
        a, b = b, a
        a_rows, b_rows = b_rows, a_rows
    a_s, b_s = a.get("scores", {}), b.get("scores", {})

    out("")
    out("=== scores (A = lever OFF, B = lever ON, per the config blocks) ===")
    out(f"  {'ability':<28} {'OFF':>7} {'ON':>7} {'delta':>7}  n")
    for k in sorted(set(a_s) | set(b_s), key=lambda x: (x == "OVERALL", x)):
        av = a_s.get(k, {}).get("accuracy")
        bv = b_s.get(k, {}).get("accuracy")
        n = b_s.get(k, {}).get("count", a_s.get(k, {}).get("count", ""))
        d = f"{bv - av:+.1f}" if av is not None and bv is not None else "—"
        out(f"  {k:<28} {av if av is not None else '—':>7} "
            f"{bv if bv is not None else '—':>7} {d:>7}  {n}")

    out("")
    out("=== gate 4 — PAIRED, ON vs the contemporaneous OFF arm ===")
    out("  No historical constant is read. The bars this replaces (OVERALL")
    out("  >= 70.0, MS >= 51.9) came from one 2026-06-10 run on a model")
    out("  deprecated in July, and an inert arm failed them about half the")
    out("  time. See lme_noise_model.py.")
    overall = paired_test(a_rows, b_rows)
    # BOTH arms must call the question multi-session. Reading one arm's
    # label would let a question drift into or out of the subset between the
    # arms, which is a difference in the denominator masquerading as one in
    # the outcome.
    ms = paired_test(a_rows, b_rows,
                     subset=lambda ra, rb: ra.get("question_type") == MS_KEY
                     == rb.get("question_type"))
    results = [("OVERALL", overall), (MS_KEY, ms)]
    checks: list[tuple[str, bool, str]] = []
    for name, t in results:
        v, sentence = paired_verdict(t)
        checks.append((name, v != "REGRESSION", f"{v} — {sentence}"))
        out("")
        out(f"  {name}: n={t['n']}  discordant={t['discordant']} "
            f"({t['regressed']} lost / {t['gained']} gained)")
        out(f"    [{'PASS' if v != 'REGRESSION' else 'FAIL'}] {v}")
        out(f"    {sentence}")
    if ms["discordant"] and overall["discordant"] and \
            ms["mde_pp"] > overall["mde_pp"]:
        out("")
        out(f"  The {MS_KEY} subset resolves only {ms['mde_pp']:.1f}pp, "
            f"against OVERALL's {overall['mde_pp']:.1f}pp:")
        out("  a smaller sample of the same churn. A null there is")
        out("  correspondingly weaker, and it is NOT the absolute MS floor it")
        out("  replaces — that floor never said what it would have missed.")

    fs = fired_subset(a_rows, b_rows)
    out("")
    out("=== fired subset (reported, never gated) ===")
    if not fs["available"]:
        out(f"  UNAVAILABLE — {fs['reason']}.")
        out("  That is a gap in the instrument, not a finding about the lever.")
    else:
        out(f"  episodes handed to the reader: OFF {fs['a_episodes_total']}, "
            f"ON {fs['b_episodes_total']}")
        out(f"  questions where the episode COUNT differs: "
            f"{fs['fired']}/{fs['shared']}")
        if fs["fired"]:
            out(f"    on those: OFF {fs['a_fired']:.1f} → ON {fs['b_fired']:.1f} "
                f"({fs['b_fired'] - fs['a_fired']:+.1f}pp)")
        if fs["shared"] - fs["fired"]:
            out(f"    on the rest: OFF {fs['a_same']:.1f} → ON {fs['b_same']:.1f} "
                f"({fs['b_same'] - fs['a_same']:+.1f}pp)")
        out("  LOWER BOUND, AND A WEAK ONE: a re-cut that yields the same")
        out("  COUNT of different episodes is invisible here, and the count")
        out("  saturates at the retrieval cap on ~84% of questions, where it")
        out("  can never move at all. This is not a null on the fired set.")
        out("  Runs carrying context_sha do not have the problem.")

    verdict = "PASS" if all(ok for _, ok, _ in checks) else "FAIL"
    out("")
    out(f"  VERDICT: {verdict}")
    if verdict == "PASS" and overall["mde_pp"] is not None:
        out(f"  Read as: no regression larger than {overall['mde_pp']:.1f}pp. "
            "The gate cannot")
        out("  see a smaller one, so a PASS is not evidence the lever is "
            "inert.")
    return verdict, {"arm": verdict_arm, "checks": checks, "fired": fs,
                     "paired": {"OVERALL": overall, MS_KEY: ms}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--lever", default=LEVER)
    args = p.parse_args()
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    verdict, _ = report(a, b, args.lever)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
