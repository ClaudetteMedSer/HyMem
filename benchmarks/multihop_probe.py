#!/usr/bin/env python3
"""Track A — bridging-edge recall@k probe for query-time multi-hop traversal.

The recall probe for Idea A (`additional_planning.md` §Idea A /
`benchmarks/readside_synthesis_plan.md` Track A). The synthetic half lives in
`tests/test_multihop.py` as pytest ground truth; THIS is the harness for the
mined LME/BEAM slice — the second probe source — plus any hand-built chains too
large for a unit test. It runs the G-A1 A/B (same graph, `graph_multihop_enabled`
off → on) and reports the three numbers the gate reads:

  1. bridging-edge recall@k on the MULTIHOP set   — must RISE off → on
  2. bridging-edge recall@k on the 1-hop CONTROL   — must NOT drop (additive
     invariant as a metric: multi-hop adds, never displaces direct hits)
  3. p50/p95 `_graph_lookup` latency off vs on     — cost gate (budget p95 on
     < 1.5× off). Multi-hop lives entirely inside `_graph_lookup`, so its
     latency delta IS the augment() delta — timing it is the honest unit.

It is deliberately LLM-free and embedding-free (the `gold_rank_probe.py`
pattern), so it runs in seconds on the box and decides G-A1 before any LME
compute is spent. The final PASS/FAIL is the reader's call (mirroring the
plan's box-read discipline); the banner is a convenience, not an oracle.

── Labeled probe JSON (self-describing) ─────────────────────────────────────
Two graph sources:
  • fresh-seed (default): the JSON carries an `edges` block; the probe builds an
    isolated temp store and seeds it. Best for hand-built / mined chains.
  • `--store <built.sqlite>`: run against an existing built graph (opened
    read-only); `edges` is then ignored. Best for probing a real LME/BEAM store.

  {
    "description": "...",
    "edges": [                                     # fresh-seed mode only
      {"subject": "atta", "predicate": "part_of", "object": "medflow"},
      {"subject": "medflow", "predicate": "deploys_to", "object": "fly.io",
       "pos": 3, "neg": 0, "days_ago": 0}
    ],
    "items": [
      {"id": "syn-1", "set": "multihop", "question": "where is atta deployed?",
       "seeds": ["atta"], "bridge": ["medflow", "deploys_to", "fly.io"],
       "route": false},
      {"id": "ctl-1", "set": "control", "question": "what does atta own?",
       "seeds": ["atta"], "bridge": ["atta", "owns", "laptop"], "route": false}
    ]
  }

Per item: `set` ∈ {"multihop","control"}; `bridge` is the target (s,p,o) whose
presence in graph_facts[:cut] counts as a recall hit; `seeds` are the direct
entity anchors; `route` (default true) mirrors the real pipeline by routing
predicates from the question (set false to force the entity-anchored fallback
path where multi-hop earns its keep). The predicate vocabulary is CHECK-
constrained by the schema — mined labels must use canonical predicates.

── Usage (from benchmarks/) ─────────────────────────────────────────────────
  python multihop_probe.py --probe multihop_probe_example.json
  python multihop_probe.py --probe mined_lme.json --store ~/.hermes/hymem.sqlite
  # sweep a single point:
  python multihop_probe.py --probe mined_lme.json --max-hops 2 --decay 0.4 --min-score 0.02
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

# Sibling import kept minimal on purpose: pull only the query primitives, not
# the adapter's heavy LLM/embedding stack. The probe never calls a model.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hymem.query.augment import _graph_lookup  # noqa: E402
from hymem.query.predicate_routing import route_predicates  # noqa: E402

_INSERT_EDGE = """
    INSERT INTO knowledge_graph
        (subject_canonical, predicate, object_canonical, pos_evidence,
         neg_evidence, last_seen, last_reinforced, status, derived)
    VALUES (?, ?, ?, ?, ?, datetime('now', ?), datetime('now', ?), 'active', 0)
"""


# Below this p95 (ms), the off→on latency ratio is timing jitter, not workload —
# a sub-millisecond lookup cannot blow any budget. The ratio is only a meaningful
# gate on a real store (thousands of edges, ms-scale lookups); on a tiny seeded
# graph it is reported but not gated.
_LATENCY_FLOOR_MS = 1.0


def _pctile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100). Empty → 0.0."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _seed_store(root: Path, edges: list[dict]) -> sqlite3.Connection:
    """Build an isolated HyMem store (runs migrations) and seed `edges`."""
    from hymem import HyMem, HyMemConfig
    from hymem.extraction.llm import StubLLMClient

    hy = HyMem(HyMemConfig(root=root), llm=StubLLMClient(default="[]"))
    conn = hy.conn
    for e in edges:
        conn.execute(
            _INSERT_EDGE,
            (
                e["subject"], e["predicate"], e["object"],
                int(e.get("pos", 1)), int(e.get("neg", 0)),
                f"-{int(e.get('days_ago', 0))} days",
                f"-{int(e.get('days_ago', 0))} days",
            ),
        )
    conn.commit()
    return conn


def _open_store_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _bridge_in_facts(facts, bridge: tuple[str, str, str], cut: int) -> bool:
    target = tuple(bridge)
    for f in facts[:cut]:
        if (f.subject, f.predicate, f.object) == target:
            return True
    return False


def _lookup(conn, cfg, item):
    routed = (
        route_predicates(item["question"])
        if item.get("route", True)
        else frozenset()
    )
    return _graph_lookup(
        conn, cfg, item["question"], list(item["seeds"]), {}, routed,
        overlap_info={}, embedding_client=None,
    )


def _run(conn, off_cfg, on_cfg, items: list[dict], cut: int,
         latency_reps: int) -> dict:
    """Score bridging-edge recall@cut off vs on, per set, plus latency."""
    by_set: dict[str, dict] = {}
    lat_off: list[float] = []
    lat_on: list[float] = []
    per_item: list[dict] = []

    for it in items:
        s = it.get("set", "multihop")
        rec = by_set.setdefault(s, {"n": 0, "off": 0, "on": 0})
        rec["n"] += 1

        off_facts = _lookup(conn, off_cfg, it)
        on_facts = _lookup(conn, on_cfg, it)
        hit_off = _bridge_in_facts(off_facts, it["bridge"], cut)
        hit_on = _bridge_in_facts(on_facts, it["bridge"], cut)
        rec["off"] += int(hit_off)
        rec["on"] += int(hit_on)

        # Latency: repeat the same call so the timing isn't a single-sample fluke.
        for cfg, bucket in ((off_cfg, lat_off), (on_cfg, lat_on)):
            for _ in range(latency_reps):
                t0 = time.perf_counter()
                _lookup(conn, cfg, it)
                bucket.append((time.perf_counter() - t0) * 1000.0)  # ms

        per_item.append({
            "id": it.get("id", "?"), "set": s, "bridge": tuple(it["bridge"]),
            "hit_off": hit_off, "hit_on": hit_on,
        })

    return {"by_set": by_set, "lat_off": lat_off, "lat_on": lat_on,
            "per_item": per_item}


def _summary(res: dict, cut: int) -> dict:
    """Machine-readable roll-up: per-set recall off/on, latency, gate booleans."""
    by_set = res["by_set"]

    def _set(name: str) -> dict:
        r = by_set.get(name, {"n": 0, "off": 0, "on": 0})
        n = r["n"] or 1
        return {"n": r["n"], "off": r["off"], "on": r["on"],
                "recall_off": 100.0 * r["off"] / n, "recall_on": 100.0 * r["on"] / n}

    p95_off, p95_on = _pctile(res["lat_off"], 95), _pctile(res["lat_on"], 95)
    mh, ct = _set("multihop"), _set("control")
    lat_meaningful = p95_off >= _LATENCY_FLOOR_MS
    gate = {
        "multihop_rose": mh["on"] > mh["off"],
        "control_held": ct["on"] >= ct["off"],
        "latency_ok": (not lat_meaningful) or (p95_on < 1.5 * p95_off),
    }
    return {
        "cut": cut, "multihop": mh, "control": ct,
        "latency_ms": {
            "off_p50": _pctile(res["lat_off"], 50), "off_p95": p95_off,
            "on_p50": _pctile(res["lat_on"], 50), "on_p95": p95_on,
            "p95_ratio": (p95_on / p95_off) if p95_off else None,
            "gated": lat_meaningful,
        },
        "gate": gate, "pass": all(gate.values()),
    }


def _report(res: dict, cut: int, verbose: bool) -> bool:
    by_set = res["by_set"]
    p50_off, p95_off = _pctile(res["lat_off"], 50), _pctile(res["lat_off"], 95)
    p50_on, p95_on = _pctile(res["lat_on"], 50), _pctile(res["lat_on"], 95)

    print(f"\n=== Track A — bridging-edge recall@{cut} (off → on) ===\n")
    print(f"{'set':<10}{'n':>4}{'recall_off':>12}{'recall_on':>12}{'Δ':>8}")
    for s in ("multihop", "control"):
        if s not in by_set:
            continue
        r = by_set[s]
        n = r["n"] or 1
        off_pct = 100.0 * r["off"] / n
        on_pct = 100.0 * r["on"] / n
        print(f"{s:<10}{r['n']:>4}{off_pct:>11.1f}%{on_pct:>11.1f}%"
              f"{on_pct - off_pct:>+7.1f}")

    print(f"\nlatency (_graph_lookup, ms):  "
          f"off p50={p50_off:.2f} p95={p95_off:.2f}   "
          f"on p50={p50_on:.2f} p95={p95_on:.2f}   "
          f"p95 ratio={p95_on / p95_off:.2f}×" if p95_off else "latency: n/a")

    if verbose:
        print("\nper-item:")
        for pi in res["per_item"]:
            flip = "  ← FLIP" if (pi["hit_on"] and not pi["hit_off"]) else (
                "  ← LOST" if (pi["hit_off"] and not pi["hit_on"]) else "")
            print(f"  {pi['id']:<12}{pi['set']:<10}"
                  f"off={int(pi['hit_off'])} on={int(pi['hit_on'])}"
                  f"  {pi['bridge']}{flip}")

    # G-A1 gate (advisory banner; the reader takes the final call).
    mh = by_set.get("multihop", {"n": 0, "off": 0, "on": 0})
    ct = by_set.get("control", {"n": 0, "off": 0, "on": 0})
    mh_rise = mh["on"] > mh["off"]
    ct_hold = ct["on"] >= ct["off"]
    lat_meaningful = p95_off >= _LATENCY_FLOOR_MS
    lat_ok = (not lat_meaningful) or (p95_on < 1.5 * p95_off)
    lat_label = (
        f"p95 latency on < 1.5× off ({p95_on:.2f} vs {p95_off:.2f} ms)"
        if lat_meaningful else
        f"latency below {_LATENCY_FLOOR_MS:.0f}ms floor — ratio not gated "
        f"(off p95={p95_off:.2f}ms)"
    )
    checks = [
        (mh_rise, f"multihop recall rose ({mh['off']}→{mh['on']} / {mh['n']})"),
        (ct_hold, f"control recall held ({ct['off']}→{ct['on']} / {ct['n']})"),
        (lat_ok, lat_label),
    ]
    passed = all(c for c, _ in checks)
    print(f"\n── G-A1 advisory: {'PASS' if passed else 'FAIL'} ──")
    for ok, label in checks:
        print(f"  [{'✓' if ok else '✗'}] {label}")
    print("  (advisory only — G-A1 is the reader's call on the mined set; "
          "then run the LME guard as non-regression.)")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--probe", required=True, type=Path,
                    help="labeled probe JSON (see module docstring)")
    ap.add_argument("--store", type=Path, default=None,
                    help="existing built store (read-only); ignores JSON `edges`")
    ap.add_argument("--cut", type=int, default=8,
                    help="recall@cut — bridge must appear in graph_facts[:cut]")
    ap.add_argument("--latency-reps", type=int, default=50,
                    help="timed repeats per item per arm (latency distribution)")
    ap.add_argument("--max-hops", type=int, default=None, help="sweep override")
    ap.add_argument("--decay", type=float, default=None, help="sweep override")
    ap.add_argument("--min-score", type=float, default=None, help="sweep override")
    ap.add_argument("--verbose", action="store_true", help="per-item table")
    ap.add_argument("--json", action="store_true",
                    help="emit one machine-readable JSON summary line (for the "
                         "sweep loop) instead of the human report")
    args = ap.parse_args()

    spec = json.loads(args.probe.read_text())
    items = spec["items"]
    if not items:
        ap.error("probe JSON has no items")

    from hymem import HyMemConfig

    tmp = tempfile.TemporaryDirectory()
    try:
        if args.store is not None:
            conn = _open_store_ro(args.store)
            print(f"[store] read-only {args.store} — {len(items)} items", file=sys.stderr)
        else:
            conn = _seed_store(Path(tmp.name), spec.get("edges", []))
            print(f"[seed] {len(spec.get('edges', []))} edges — {len(items)} items",
                  file=sys.stderr)

        # graph_top_k = cut so _graph_lookup returns exactly the top-cut window.
        base = HyMemConfig(root=Path(tmp.name), graph_top_k=args.cut)
        on_over = {}
        if args.max_hops is not None:
            on_over["graph_multihop_max_hops"] = args.max_hops
        if args.decay is not None:
            on_over["graph_multihop_decay"] = args.decay
        if args.min_score is not None:
            on_over["graph_multihop_min_score"] = args.min_score

        off_cfg = dataclasses.replace(base, graph_multihop_enabled=False)
        on_cfg = dataclasses.replace(base, graph_multihop_enabled=True, **on_over)
        print(f"[cfg] on: max_hops={on_cfg.graph_multihop_max_hops} "
              f"decay={on_cfg.graph_multihop_decay} "
              f"min_score={on_cfg.graph_multihop_min_score}", file=sys.stderr)

        res = _run(conn, off_cfg, on_cfg, items, args.cut, args.latency_reps)
        if args.json:
            summary = _summary(res, args.cut)
            summary["config"] = {
                "max_hops": on_cfg.graph_multihop_max_hops,
                "decay": on_cfg.graph_multihop_decay,
                "min_score": on_cfg.graph_multihop_min_score,
            }
            print(json.dumps(summary))
            passed = summary["pass"]
        else:
            passed = _report(res, args.cut, args.verbose)
    finally:
        tmp.cleanup()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
