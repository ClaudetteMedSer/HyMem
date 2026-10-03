"""Idea B — rules-tier per-call overhead (the "cost watch").

Rules inject into EVERY augment()/context call — the hot path — so the tier has
to be near-free or it taxes every request. This probe measures the cost with
three arms on the same store:

  OFF       rules_enabled=False           (baseline — tier not loaded)
  ON/empty  rules_enabled=True, 0 rules    (the default in production: ON but no
                                            rules added yet — MUST be ~free)
  ON/full   rules_enabled=True, cap rules  (a saturated rulebook — the worst case)

It reports, per arm: augment() p50/p95 latency and the rendered-context char
delta (a token-cost proxy). The gates encode the two claims the flip rests on:

  1. ON/empty adds ZERO rendered chars vs OFF — "inert until add_rule()".
  2. ON/full stays within a bounded budget (`--max-overhead-ms`, chars ≤ cap·line).

LLM-free, deterministic, runs anywhere. This is data for the competitive
scorecard: always-injected + precisely-extracted + provably cheap.

Usage:
  python benchmarks/rules_overhead.py [--reps 400] [--max-overhead-ms 1.0] [--json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

from hymem import HyMem, HyMemConfig
from hymem.extraction.llm import StubLLMClient
from hymem.query.ask import render_context

_QUERY = "what should I keep in mind when deploying the payments service?"


def _pctile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def _time_augment(hy, reps: int) -> tuple[float, float, int]:
    """Return (p50_ms, p95_ms, rendered_chars) for augment()+render over `reps`."""
    lat: list[float] = []
    chars = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        ctx = hy.augment(_QUERY)
        lat.append((time.perf_counter() - t0) * 1000.0)
        chars = len(render_context(ctx, max_chars=0))
    return _pctile(lat, 50), _pctile(lat, 95), chars


def _arm(rules_enabled: bool, n_rules: int, reps: int) -> dict:
    cfg = HyMemConfig(root=Path(tempfile.mkdtemp()))
    hy = HyMem(dataclasses.replace(cfg, rules_enabled=rules_enabled),
               llm=StubLLMClient(default="[]"))
    for i in range(n_rules):
        hy.add_rule(f"Standing rule number {i}: always keep step {i} in mind before acting.")
    p50, p95, chars = _time_augment(hy, reps)
    hy.close()
    return {"p50_ms": p50, "p95_ms": p95, "rendered_chars": chars, "rules": n_rules}


def main() -> None:
    ap = argparse.ArgumentParser(description="Idea B rules-tier overhead probe.")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--cap", type=int, default=16, help="rules for the ON/full arm")
    ap.add_argument("--max-overhead-ms", type=float, default=1.0,
                    help="max allowed ON/full p95 latency overhead vs OFF")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    off = _arm(False, 0, args.reps)
    on_empty = _arm(True, 0, args.reps)
    on_full = _arm(True, args.cap, args.reps)

    empty_char_overhead = on_empty["rendered_chars"] - off["rendered_chars"]
    full_char_overhead = on_full["rendered_chars"] - off["rendered_chars"]
    full_ms_overhead = on_full["p95_ms"] - off["p95_ms"]

    gate_inert = empty_char_overhead == 0
    gate_latency = full_ms_overhead <= args.max_overhead_ms
    passed = gate_inert and gate_latency

    summary = {
        "off": off, "on_empty": on_empty, "on_full": on_full,
        "empty_char_overhead": empty_char_overhead,
        "full_char_overhead": full_char_overhead,
        "full_p95_ms_overhead": full_ms_overhead,
        "max_overhead_ms": args.max_overhead_ms,
        "gate_inert": gate_inert, "gate_latency": gate_latency, "pass": passed,
    }
    if args.json:
        print(json.dumps(summary))
        sys.exit(0 if passed else 1)

    print(f"\n=== Idea B — rules-tier per-call overhead (reps={args.reps}) ===\n")
    print(f"{'arm':<12}{'rules':>6}{'p50 ms':>10}{'p95 ms':>10}{'ctx chars':>12}")
    for name, a in (("OFF", off), ("ON/empty", on_empty), ("ON/full", on_full)):
        print(f"{name:<12}{a['rules']:>6}{a['p50_ms']:>10.3f}{a['p95_ms']:>10.3f}"
              f"{a['rendered_chars']:>12}")
    print(f"\nON/empty char overhead vs OFF: {empty_char_overhead}  (want 0 — inert)")
    print(f"ON/full  char overhead vs OFF: {full_char_overhead}  "
          f"({args.cap} rules)")
    print(f"ON/full  p95 latency overhead: {full_ms_overhead:.3f} ms")
    print(f"\n── rules overhead gate: {'PASS' if passed else 'FAIL'} ──")
    print(f"  [{'✓' if gate_inert else '✗'}] ON-but-empty adds 0 rendered chars "
          f"(default is free until add_rule())")
    print(f"  [{'✓' if gate_latency else '✗'}] ON/full p95 overhead "
          f"{full_ms_overhead:.3f}ms ≤ {args.max_overhead_ms}ms")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
