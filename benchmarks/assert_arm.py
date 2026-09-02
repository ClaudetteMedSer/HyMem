#!/usr/bin/env python3
"""Assert that an adapter argv actually selects the arm it is labelled with.

Written after a 5.5-hour, ~2,000-call guard run produced two copies of the OFF
arm. The pre-flight for that run drove the real parser with both command lines
and proved they differed in exactly `episode_granularity` -- it validated the
command lines the operator INTENDED. The runner script built them with

    run_arm () { label="$1"; shift; ... "$@"; }
    run_arm on

where `on` is consumed as the label and `"$@"` expands to nothing, so the flag
was never passed. Nothing checked the argv the script actually constructed, and
the gap between "the command line I designed" and "the command line the harness
builds" is precisely where the cost went.

So this takes the SAME argv array the runner is about to execute, parses it
with the adapter's own parser, and fails if the lever does not come out as the
label claims. Nothing is spent before it passes. It is the argv equivalent of
`arm_evidence`, moved from after the run to before it: one reads the config
block a run wrote, the other reads the argv a run is about to use.

Exits 0 on match, 1 on mismatch, 2 if the argv will not parse at all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}


class _Caught(Exception):
    def __init__(self, ns):
        self.ns = ns


def parse_adapter_argv(argv: list[str]):
    """Run the adapter's real parser over `argv`, stopping at parse_args.

    Nothing after parse_args executes, so this makes no call and touches no
    store. Using the adapter's OWN parser is the point: a reimplementation
    here would drift from it and re-open the gap it exists to close."""
    import longmemeval_adapter as lme

    real = argparse.ArgumentParser.parse_args

    def stop(self, args=None, namespace=None):
        raise _Caught(real(self, args, namespace))

    argparse.ArgumentParser.parse_args = stop
    old_argv = sys.argv
    try:
        sys.argv = ["longmemeval_adapter.py"] + list(argv)
        lme.main()
    except _Caught as c:
        return vars(c.ns)
    finally:
        argparse.ArgumentParser.parse_args = real
        sys.argv = old_argv
    raise RuntimeError("parse_args was never reached; the check is invalid")


def coerce_expected(value: str) -> bool:
    v = value.strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ValueError(f"expected true/false, got {value!r}")


def check(argv: list[str], dest: str, expected: bool) -> tuple[bool, str]:
    ns = parse_adapter_argv(argv)
    if dest not in ns:
        return False, (f"the parser produced no {dest!r} at all -- the flag "
                       f"was renamed or removed, and this check is stale")
    actual = bool(ns[dest])
    if actual != expected:
        return False, (f"argv selects {dest}={actual!r} but the arm is "
                       f"labelled {expected!r}; the flag is missing from the "
                       f"command the runner built")
    return True, f"{dest}={actual!r}, as labelled"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dest", default="episode_granularity",
                   help="argparse dest of the lever (NOT the config key)")
    p.add_argument("--expect", required=True)
    p.add_argument("argv", nargs=argparse.REMAINDER,
                   help="-- followed by the exact adapter argv")
    args = p.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        print("assert_arm: no argv given", file=sys.stderr)
        return 2
    try:
        expected = coerce_expected(args.expect)
    except ValueError as e:
        print(f"assert_arm: {e}", file=sys.stderr)
        return 2
    try:
        ok, note = check(argv, args.dest, expected)
    except SystemExit:
        print("assert_arm: the argv does not parse", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"assert_arm: {e}", file=sys.stderr)
        return 2
    print(f"assert_arm: {'OK' if ok else 'MISMATCH'} — {note}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
