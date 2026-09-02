#!/usr/bin/env python3
"""Plan C gate 5 — the dream cost watch, run on a SNAPSHOT of a live store.

Gate 5 asks what flipping `episode_granularity_enabled` costs the dream: a
one-time re-digest of every session, plus whatever the steady state becomes
afterwards. The plan banked the shape ("one-time re-digest cost plus
steady-state delta, measurable before/after") and no numbers, so the criteria
are pre-registered in `score()` below and committed before the run.

WHY A SNAPSHOT. The production store dreams on a schedule (rows 1394-1397 all
landed on 2026-09-01). Flipping the lever on it would rewrite live episode rows
and let the scheduled dreamer interleave with the measurement -- mutating the
user's memory to take a reading, and taking a bad reading. The watch therefore
copies the store through sqlite's backup API (consistent under a concurrent
writer, which `cp` is not) and runs every leg against the copy.

WHY FOUR LEGS. `settle` exists because the snapshot is not in steady state: 9
of 110 sessions carry `digested_prompt_version` NULL and would be digested by
the FIRST blob cycle whatever the lever says. Reading that cycle as the blob
baseline would charge the granular arm for work the blob arm also does. So:

    settle  (OFF)  bring the store to zero-tail-call steady state
    before  (OFF)  the baseline: what a blob dream costs on a settled store
    migrate (ON)   the one-time re-digest -- the cost the flip actually incurs
    after   (ON)   the granular steady state, which is the thing under test

The gated comparison is `after` vs `before`. `migrate` is reported, not gated:
a one-off is a price, not a regression.

Calls are attributed by matching the LLM request's system prompt against the
two digest prompts by identity, so a leg cannot claim a granularity it did not
send. That is the gate-4 lesson carried over -- the 2026-08-30 guard pair could
not evidence which arm it was, and a cost watch that cannot show which prompt
it dreamt under would have exactly the same defect.

No LLM call is made by `snapshot` or `score`; only `run` spends.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

LEG_SETTLE, LEG_BEFORE, LEG_MIGRATE, LEG_AFTER = "settle", "before", "migrate", "after"
LEG_ORDER = (LEG_SETTLE, LEG_BEFORE, LEG_MIGRATE, LEG_AFTER)
GRANULAR_LEGS = (LEG_MIGRATE, LEG_AFTER)

# ------------------------------------------------------------- pre-registered
# Committed before the run; read after it. Bounds, not point predictions --
# wall clock crosses a network and a same-numbers criterion would fail on
# API jitter alone.
STEADY_EXTRA_CALLS_MAX = 5            # `after` may not out-call `before` by more
STEADY_WALL_FACTOR = 2.0              # ...nor take longer than this x `before`
STEADY_WALL_FLOOR_S = 60.0            # ...with a floor, so a 3s baseline is fair
STAMP_COVERAGE_MIN = 0.90             # stamps landed / granular calls sent

# The denominator for the stamp criteria is the number of granular digest
# CALLS the leg actually made -- not the session count. A dress rehearsal
# against the production snapshot found 70 of its 110 sessions are empty
# stubs: `run_dreaming` mints no fallback chunk for a session with no
# user/assistant content and `continue`s before the digest, so those sessions
# can never carry a stamp however the lever is set. A criterion keyed on all
# 110 would have read FAIL at 36% coverage and charged the flip for the
# store's own shape.
#
# Calls-to-stamps is the ratio that means something, and it is bounded on
# BOTH sides because the two sides are different defects:
#   stamps / calls too LOW  -- calls that landed no stamp: either a dropped
#       UPSERT (the store re-digests forever) or a loop over one session.
#   stamps / calls too HIGH -- stamps with no call behind them, i.e. a
#       session marked migrated that was never re-read.
# An earlier cut expressed the second as `calls <= 1.2 x stamped`, which is
# the SAME inequality as the first with a looser constant -- so no test could
# distinguish them and dropping either one changed nothing. It survived
# mutation-checking, which is how it was found.


class CountingLLM:
    """Wraps an LLMClient and counts what was sent, by prompt identity."""

    def __init__(self, inner, blob_system: str, granular_system: str):
        self._inner = inner
        self._blob = blob_system
        self._granular = granular_system
        self.calls = 0
        self.digest_blob_calls = 0
        self.digest_granular_calls = 0
        self.other_calls = 0
        self.prompt_chars = 0

    def complete(self, request):
        self.calls += 1
        self.prompt_chars += len(request.system or "") + len(request.user or "")
        if request.system == self._granular:
            self.digest_granular_calls += 1
        elif request.system == self._blob:
            self.digest_blob_calls += 1
        else:
            self.other_calls += 1
        return self._inner.complete(request)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def counts(self) -> dict:
        return {
            "calls": self.calls,
            "digest_blob_calls": self.digest_blob_calls,
            "digest_granular_calls": self.digest_granular_calls,
            "digest_calls": self.digest_blob_calls + self.digest_granular_calls,
            "other_calls": self.other_calls,
            "prompt_chars": self.prompt_chars,
        }


def stamp_census(db_path: Path) -> dict:
    """Per-session episode-prompt stamps + episode row count, read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT episodes_prompt_version AS v, COUNT(*) AS n "
            "FROM sessions GROUP BY 1").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        # Reported, never gated: 61 sessions on the production snapshot carry
        # a digested_prompt_version but are no longer digestible (their
        # content no longer yields a chunk or a fallback), so this is not a
        # denominator for anything -- it is context for reading the others.
        digested = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE digested_prompt_version IS NOT NULL").fetchone()[0]
    finally:
        conn.close()
    return {
        "sessions": total,
        "digested": digested,
        "episodes": episodes,
        "stamps": {(v if v is not None else "NULL"): n for v, n in rows},
    }


def cmd_snapshot(args) -> int:
    src, root = Path(args.src), Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    dst = root / "hymem.sqlite"
    if dst.exists() and not args.force:
        print(f"refusing to overwrite {dst} (pass --force)", file=sys.stderr)
        return 1
    # backup() rather than cp: the source has a scheduled writer, and a
    # half-copied WAL would be a corrupt baseline that still opens.
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    print(f"snapshot: {src} -> {dst}")
    print(f"  {json.dumps(stamp_census(dst))}")
    return 0


def cmd_run(args) -> int:
    from hymem.api import HyMem
    from hymem.config import HyMemConfig
    from hymem.bootstrap import resolve_env
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.extraction.prompts import (
        SESSION_DIGEST_GRANULAR_SYSTEM, SESSION_DIGEST_SYSTEM)

    root = Path(args.root)
    db_path = root / "hymem.sqlite"
    if not db_path.exists():
        print(f"no store at {db_path}; run `snapshot` first", file=sys.stderr)
        return 1

    granular = args.leg in GRANULAR_LEGS
    env = resolve_env()
    if not env.has_llm_key:
        print("no LLM API key in the environment", file=sys.stderr)
        return 1

    cfg = HyMemConfig(root=root, episode_granularity_enabled=granular)
    llm = CountingLLM(
        OpenAICompatibleClient(api_key=env.llm_api_key,
                               base_url=env.llm_base_url,
                               model=env.llm_model),
        SESSION_DIGEST_SYSTEM, SESSION_DIGEST_GRANULAR_SYSTEM)

    before_census = stamp_census(db_path)
    # No embedding client: this watch measures DIGEST cost, and an embedder
    # would add network time that belongs to a different tier. Both arms are
    # measured without one, so the contrast is unaffected.
    mem = HyMem(cfg, llm=llm)
    t0 = time.time()
    try:
        report = mem.dream()
    finally:
        elapsed = time.time() - t0
        mem.close()
    after_census = stamp_census(db_path)

    rep = {k: v for k, v in vars(report).items() if not k.startswith("_")}
    out = {
        "leg": args.leg,
        "granularity": granular,
        "prompt_version": "episodes.granular.v1" if granular else None,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": elapsed,
        "llm": llm.counts(),
        "report": rep,
        "census_before": before_census,
        "census_after": after_census,
        "model": env.llm_model,
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"[{args.leg}] granular={granular} elapsed={elapsed:.1f}s "
          f"calls={llm.calls} (digest blob={llm.digest_blob_calls} "
          f"granular={llm.digest_granular_calls}) -> {dest}")
    return 0


def evaluate(legs: dict) -> tuple[str, list[tuple[str, bool, str]]]:
    """Score the four legs against the criteria pre-registered above.

    Returns (verdict, checks). Verdict is PASS, FAIL, or INCOMPLETE -- the last
    when a leg is missing, which is not a failure of the feature."""
    missing = [n for n in LEG_ORDER if n not in legs]
    if missing:
        return "INCOMPLETE", [("legs present", False,
                               f"missing: {', '.join(missing)}")]

    before, migrate, after = legs[LEG_BEFORE], legs[LEG_MIGRATE], legs[LEG_AFTER]
    checks: list[tuple[str, bool, str]] = []

    # 1. The lever fired, evidenced by the prompt the calls actually carried.
    #    Without this the watch is a measurement of nothing -- the gate-4 defect.
    g = migrate["llm"]["digest_granular_calls"]
    b = migrate["llm"]["digest_blob_calls"]
    checks.append((
        "migrate sent the GRANULAR digest prompt", g > 0 and b == 0,
        f"granular={g} blob={b} (blob must be 0: a mixed leg cannot be priced)"))

    def _stamps(census):
        return census["stamps"].get("episodes.granular.v1", 0)
    delta = _stamps(migrate["census_after"]) - _stamps(migrate["census_before"])
    frac = delta / g if g else 0.0
    checks.append((
        f"stamps landed for >= {STAMP_COVERAGE_MIN:.0%} of granular calls",
        g > 0 and frac >= STAMP_COVERAGE_MIN,
        f"{delta} new stamps / {g} calls = {frac:.1%} "
        f"(low = a dropped stamp, or a loop over one session)"))

    # 2. ...and no stamp without a call behind it.
    checks.append((
        "no session stamped that was not re-read",
        delta <= g, f"{delta} new stamps <= {g} calls"))

    # 3. The steady state comes back -- the architectural claim under test.
    bc, ac = before["llm"]["digest_calls"], after["llm"]["digest_calls"]
    checks.append((
        f"after digest calls <= before + {STEADY_EXTRA_CALLS_MAX}",
        ac <= bc + STEADY_EXTRA_CALLS_MAX, f"after={ac} before={bc}"))

    bw, aw = before["elapsed_s"], after["elapsed_s"]
    wall_cap = max(STEADY_WALL_FACTOR * bw, bw + STEADY_WALL_FLOOR_S)
    checks.append((
        "after wall clock within the steady-state bound",
        aw <= wall_cap, f"after={aw:.1f}s cap={wall_cap:.1f}s (before={bw:.1f}s)"))

    # 4. No new failure modes on either granular leg.
    for name in GRANULAR_LEGS:
        r = legs[name]["report"]
        df = r.get("digest_failures", 0) or 0
        ff = r.get("aggregation_fusion_failures", 0) or 0
        checks.append((f"{name}: no digest/fusion failures", df == 0 and ff == 0,
                       f"digest_failures={df} fusion_failures={ff}"))

    verdict = "PASS" if all(ok for _, ok, _ in checks) else "FAIL"
    return verdict, checks


def cmd_score(args) -> int:
    legs = {}
    for p in args.legs:
        d = json.loads(Path(p).read_text())
        legs[d["leg"]] = d
    verdict, checks = evaluate(legs)

    print("Plan C gate 5 — dream cost watch")
    print(f"  legs: {', '.join(sorted(legs))}\n")
    if len(legs) == len(LEG_ORDER):
        print("  leg      granular  elapsed_s  calls  digest(blob/gran)  episodes")
        for n in LEG_ORDER:
            d = legs[n]
            c = d["llm"]
            print(f"  {n:<8} {str(d['granularity']):<9} {d['elapsed_s']:>8.1f}  "
                  f"{c['calls']:>5}  {c['digest_blob_calls']:>6}/"
                  f"{c['digest_granular_calls']:<6}     "
                  f"{d['census_after']['episodes']:>6}")
        mig = legs[LEG_MIGRATE]
        print(f"\n  one-time migration cost (reported, not gated): "
              f"{mig['llm']['digest_calls']} digest calls, "
              f"{mig['elapsed_s']:.0f}s, "
              f"{mig['llm']['prompt_chars']:,} prompt chars")
        d0 = legs[LEG_BEFORE]["census_after"]["episodes"]
        d1 = mig["census_after"]["episodes"]
        print(f"  episode rows {d0} -> {d1} "
              f"({d1 - d0:+d}; the shape change the flip is for)")
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n  VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="consistent copy of a live store")
    s.add_argument("--src", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_snapshot)

    r = sub.add_parser("run", help="one dream cycle against the snapshot (SPENDS)")
    r.add_argument("--root", required=True)
    r.add_argument("--leg", required=True, choices=LEG_ORDER)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("score", help="score the legs (offline)")
    c.add_argument("legs", nargs="+")
    c.set_defaults(func=cmd_score)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
