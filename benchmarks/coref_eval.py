#!/usr/bin/env python3
"""E5 gate — anaphora/ellipsis resolution rate + the no-harm control.

The gate for Campaign E, Step 2 (`additional_planning.md` §Campaign E). Two
numbers, both pre-registered, both LLM-free by default so this runs in seconds on
the box:

  1. **resolution rate** on the 31 hand-built follow-up items — must be ≥ 80%.
     An item RESOLVES when `rewrite_query` fired AND at least one acceptable
     referent (`expect`) appears in the rewritten query.
  2. **no-harm control**: rewrites on the self-contained items — must be ZERO.
     This is the half that matters. E5 is additive by construction (the rewrite
     only appends), so a resolution miss costs nothing; a rewrite on an ordinary
     lookup is the only way this feature can hurt, and the control is what proves
     it doesn't.

Deliberately NOT an LME run: LME questions are self-contained single-shot lookups
with no conversational antecedent, so its score cannot move either way. E5 is
production value that the benchmark triad is blind to.

── Eval set (`benchmarks/coref_eval_set.json`) ─────────────────────────────────
Items with an `edges` block are seeded into a throwaway store, so the CANONICAL
graph-entity path is exercised (what production hits once a session has been
dreamed). Items without one exercise the salient-token fallback — a brand-new
session with no edges yet, which is exactly when a follow-up is most likely. The
report breaks the rate down by path, because a heuristic that only works with a
warm graph would be a different (weaker) result than the headline number.

── Usage (from benchmarks/) ─────────────────────────────────────────────────────
  python coref_eval.py
  python coref_eval.py --eval coref_eval_set.json --verbose
  python coref_eval.py --llm-fallback   # Stage 2 arm; needs a real LLM client
  python coref_eval.py --json           # one machine-readable summary line
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hymem.query.coref import QueryRewrite, rewrite_query  # noqa: E402
from hymem.session import Message  # noqa: E402

_INSERT_EDGE = """
    INSERT INTO knowledge_graph
        (subject_canonical, predicate, object_canonical, pos_evidence,
         neg_evidence, last_seen, last_reinforced, status, derived)
    VALUES (?, ?, ?, 2, 0, datetime('now'), datetime('now'), 'active', 0)
"""

# Pre-registered gate thresholds (see module docstring). Changing these changes
# the gate, so they live here as named constants, not as CLI defaults.
_MIN_RESOLUTION = 0.80
_MAX_CONTROL_REWRITES = 0


def _turns(pairs: list[list[str]]) -> list[Message]:
    return [
        Message(id=i, session_id="eval", role=role, content=text)
        for i, (role, text) in enumerate(pairs, 1)
    ]


def _seeded_conn(root: Path, edges: list[list[str]]):
    """A throwaway HyMem store with `edges` seeded — returns (conn, closer)."""
    from hymem import HyMem, HyMemConfig
    from hymem.extraction.llm import StubLLMClient

    hy = HyMem(HyMemConfig(root=root), llm=StubLLMClient(default="[]"))
    for subj, pred, obj in edges:
        hy.conn.execute(_INSERT_EDGE, (subj, pred, obj))
    hy.conn.commit()
    return hy.conn, hy.close


def _run_item(item: dict, cfg, tmp: Path, llm=None) -> tuple[QueryRewrite, str]:
    """Rewrite one item's query; returns (rewrite, path) where path is
    "graph" (edges seeded) or "salient" (no store)."""
    edges = item.get("edges") or []
    if not edges:
        return rewrite_query(item["query"], _turns(item["turns"]), cfg=cfg,
                             llm=llm), "salient"
    root = tmp / item["id"]
    root.mkdir(parents=True, exist_ok=True)
    conn, close = _seeded_conn(root, edges)
    try:
        return rewrite_query(item["query"], _turns(item["turns"]), cfg=cfg,
                             conn=conn, llm=llm), "graph"
    finally:
        close()


def _resolved(rw: QueryRewrite, expect: list[str]) -> bool:
    """An item resolves when the rewrite fired AND carries an acceptable referent.
    Matching is substring-on-lowercase: referents are canonical names or word
    tokens, so a containment test is the right granularity (and mirrors the
    adapters' `_gold_in_pool` posture — be generous about surface form, strict
    about presence)."""
    if not rw.changed:
        return False
    low = rw.rewritten.lower()
    return any(e.lower() in low for e in expect)


def _run(spec: dict, cfg, tmp: Path, llm=None) -> dict:
    items, controls = spec["items"], spec.get("controls", [])
    rows: list[dict] = []
    for it in items:
        rw, path = _run_item(it, cfg, tmp, llm=llm)
        rows.append({
            "id": it["id"], "kind": it.get("kind", "?"), "path": path,
            "query": it["query"], "rewritten": rw.rewritten,
            "rule": rw.rule, "changed": rw.changed,
            "resolved": _resolved(rw, it.get("expect", [])),
        })
    ctl_rows: list[dict] = []
    for c in controls:
        rw, path = _run_item(c, cfg, tmp, llm=llm)
        ctl_rows.append({
            "id": c["id"], "path": path, "query": c["query"],
            "rewritten": rw.rewritten, "rule": rw.rule, "changed": rw.changed,
        })
    return {"items": rows, "controls": ctl_rows}


def _summary(res: dict) -> dict:
    rows, ctl = res["items"], res["controls"]
    n = len(rows) or 1
    resolved = sum(1 for r in rows if r["resolved"])
    fired = sum(1 for r in rows if r["changed"])
    rewrites = [c for c in ctl if c["changed"]]

    def _slice(key: str, value: str) -> dict:
        sub = [r for r in rows if r[key] == value]
        return {"n": len(sub), "resolved": sum(1 for r in sub if r["resolved"])}

    gate = {
        "resolution_ok": (resolved / n) >= _MIN_RESOLUTION,
        "no_harm_ok": len(rewrites) <= _MAX_CONTROL_REWRITES,
    }
    return {
        "n_items": len(rows), "fired": fired, "resolved": resolved,
        "resolution_rate": 100.0 * resolved / n,
        "by_path": {p: _slice("path", p) for p in ("graph", "salient")},
        "by_kind": {k: _slice("kind", k)
                    for k in ("pronoun", "ellipsis", "demonstrative")},
        "n_controls": len(ctl), "control_rewrites": len(rewrites),
        "control_offenders": [c["id"] for c in rewrites],
        "gate": gate, "pass": all(gate.values()),
    }


def _report(res: dict, verbose: bool) -> bool:
    s = _summary(res)
    print("\n=== E5 — anaphora/ellipsis resolution ===\n")
    print(f"follow-up items: {s['n_items']}   fired: {s['fired']}   "
          f"resolved: {s['resolved']}  ({s['resolution_rate']:.1f}%)")
    print("\nby resolution path (graph = canonical entity, salient = token fallback):")
    for path, r in s["by_path"].items():
        if r["n"]:
            print(f"  {path:<9}{r['resolved']:>3}/{r['n']:<3}"
                  f"{100.0 * r['resolved'] / r['n']:>7.1f}%")
    print("\nby trigger:")
    for kind, r in s["by_kind"].items():
        if r["n"]:
            print(f"  {kind:<15}{r['resolved']:>3}/{r['n']:<3}"
                  f"{100.0 * r['resolved'] / r['n']:>7.1f}%")
    print(f"\nno-harm control: {s['control_rewrites']}/{s['n_controls']} "
          f"self-contained queries rewritten"
          + (f"  ← {', '.join(s['control_offenders'])}"
             if s["control_offenders"] else ""))

    if verbose:
        print("\nper-item:")
        for r in res["items"]:
            mark = "✓" if r["resolved"] else ("~" if r["changed"] else "✗")
            print(f"  [{mark}] {r['id']:<12}{r['kind']:<15}{r['path']:<9}"
                  f"rule={r['rule']}")
            print(f"        {r['rewritten']}")
        print("\ncontrols:")
        for c in res["controls"]:
            mark = "✗ REWRITTEN" if c["changed"] else "✓ untouched"
            print(f"  [{mark}] {c['id']:<12}rule={c['rule']}")
            if c["changed"]:
                print(f"        {c['rewritten']}")

    checks = [
        (s["gate"]["resolution_ok"],
         f"resolution ≥ {_MIN_RESOLUTION:.0%} ({s['resolution_rate']:.1f}%)"),
        (s["gate"]["no_harm_ok"],
         f"control rewrites ≤ {_MAX_CONTROL_REWRITES} "
         f"({s['control_rewrites']})"),
    ]
    print(f"\n── E5 gate: {'PASS' if s['pass'] else 'FAIL'} ──")
    for ok, label in checks:
        print(f"  [{'✓' if ok else '✗'}] {label}")
    print("  (the no-harm control is the load-bearing half: the rewrite is "
          "append-only,\n   so a miss costs nothing and a false fire is the only "
          "way E5 can hurt.)")
    return s["pass"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval", type=Path,
                    default=Path(__file__).resolve().parent / "coref_eval_set.json",
                    help="eval-set JSON (default: coref_eval_set.json)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override cfg.coref_max_turns")
    ap.add_argument("--llm-fallback", action="store_true",
                    help="enable the Stage-2 LLM arm (needs --api-key; the "
                         "default arm is heuristic-only and LLM-free)")
    ap.add_argument("--api-key", default="", help="for --llm-fallback")
    ap.add_argument("--verbose", action="store_true", help="per-item table")
    ap.add_argument("--json", action="store_true",
                    help="one machine-readable summary line instead of the report")
    args = ap.parse_args()

    spec = json.loads(args.eval.read_text())
    if not spec.get("items"):
        ap.error("eval JSON has no items")

    from hymem import HyMemConfig

    tmp = tempfile.TemporaryDirectory()
    try:
        base = HyMemConfig(root=Path(tmp.name) / "_cfg")
        over = {"coref_enabled": True}
        if args.max_turns is not None:
            over["coref_max_turns"] = args.max_turns
        llm = None
        if args.llm_fallback:
            from hymem.contrib.openai_client import OpenAICompatibleClient

            over["coref_llm_enabled"] = True
            llm = OpenAICompatibleClient(
                api_key=args.api_key, base_url="https://api.deepseek.com",
                model="deepseek-v4-flash",
            )
        cfg = dataclasses.replace(base, **over)
        print(f"[cfg] max_turns={cfg.coref_max_turns} "
              f"llm_fallback={cfg.coref_llm_enabled}   "
              f"items={len(spec['items'])} controls={len(spec.get('controls', []))}",
              file=sys.stderr)

        res = _run(spec, cfg, Path(tmp.name), llm=llm)
        if args.json:
            print(json.dumps(_summary(res)))
            passed = _summary(res)["pass"]
        else:
            passed = _report(res, args.verbose)
    finally:
        tmp.cleanup()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
