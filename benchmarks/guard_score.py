#!/usr/bin/env python3
"""Plan C gate 4 — score the LME full guard, in the order gate 4 requires.

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

Offline: reads two artifacts, makes no call.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
from run_registry import ARM_EVIDENCED, arm_evidence  # noqa: E402

# ------------------------------------------------------------- pre-registered
# Gate 4 is NON-REGRESSION ONLY, against the canonical full-dream baseline.
# Not a tuning signal: a change with no effect clears this bar too.
CANONICAL_OVERALL = 70.0
MS_FLOOR = 51.9
MS_KEY = "multi-session"
LEVER = "episode_granularity_enabled"


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


def bar_check(on_scores: dict) -> list[tuple[str, bool, str]]:
    out = []
    ov = on_scores.get("OVERALL", {}).get("accuracy")
    out.append((f"OVERALL >= {CANONICAL_OVERALL} (canonical full-dream)",
                ov is not None and ov >= CANONICAL_OVERALL,
                f"{ov}"))
    ms = on_scores.get(MS_KEY, {}).get("accuracy")
    out.append((f"{MS_KEY} >= {MS_FLOOR} (MS floor)",
                ms is not None and ms >= MS_FLOOR, f"{ms}"))
    return out


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

    checks = bar_check(b_s)
    out("")
    out("=== gate 4 bar (NON-REGRESSION ONLY, on the ON arm) ===")
    for name, ok, detail in checks:
        out(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

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
        out("  LOWER BOUND: a re-cut that yields the same COUNT of different")
        out("  episodes is invisible here, so this undercounts what the lever")
        out("  touched. A null on it is weaker evidence than a null on a true")
        out("  fired-indicator.")

    verdict = "PASS" if all(ok for _, ok, _ in checks) else "FAIL"
    out("")
    out(f"  VERDICT: {verdict}")
    return verdict, {"arm": verdict_arm, "checks": checks, "fired": fs}


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
