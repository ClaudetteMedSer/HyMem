#!/usr/bin/env python3
"""Track A — miner: pre-fill a bridging-edge probe set for hand-verification.

Hand-labeling ~60–100 items for the G-A1 recall probe (`multihop_probe.py`) is
the slow step. This miner cuts it to *verification*: given (a) a run's questions
+ gold answers and (b) a dreamed store with the knowledge graph, it proposes the
seed entities and the bridging edge for each question, auto-classifies each into
the `multihop` / `control` sets, and writes a probe-compatible JSON the box only
has to check.

How it proposes — reusing the feature's own machinery, so a proposed bridge is
exactly what the feature would traverse:
  • seeds  = `match_known_entities(store, question)`  (same matcher as augment)
  • bridges = `_multihop_edges(store, cfg, seeds)`     (Source 4's BFS, hop≥2)
  • directs = 1-hop edges incident to a seed           (what Source 1 already has)
It then ranks candidate edges by token overlap with the GOLD ANSWER and picks
the best. Using the gold answer here is legitimate: it LABELS the ground-truth
probe set — it is not read at retrieval time (the probe measures recall of the
labeled bridge with the feature blind to the label). Classification:
  • `multihop` — a hop≥2 bridge overlaps the gold answer AND no direct 1-hop edge
    does → the answer needs the chain (the case Source 4 targets).
  • `control`  — a direct 1-hop edge overlaps the gold answer → answerable without
    bridging; guards the additive invariant (must not drop).
  • dropped    — no edge overlaps the answer, or no seed matched → can't
    auto-label (counted, not emitted; label by hand if you want them).

Each emitted item carries the probe fields (id/set/question/seeds/bridge/route)
plus `_`-prefixed hints (`_hop`, `_answer_overlap`, `_alt_bridges`, `_gold`, …)
for the human; `multihop_probe.py` ignores unknown keys, so verified stubs run
as-is. VERIFY each item: confirm the `bridge` is the answer-bearing edge (swap in
an `_alt_bridges` entry or delete the item otherwise).

Usage (from repo root; LLM-free, seconds):
  python benchmarks/multihop_miner.py \
    --from ~/.hermes/benchmarks/<run>.json \
    --store STORE.sqlite --out SLICE.json
  # then hand-verify SLICE.json, then:
  python benchmarks/multihop_probe.py --probe SLICE.json --store STORE.sqlite --verbose

STORE must contain the edges for these questions. LME haystacks are per-question,
so build one combined store first (ingest the selected questions' sessions into a
single store and dream it), or point at a persistent Hermes store.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hymem.query.augment import _multihop_edges  # noqa: E402
from hymem.query.entities import match_known_entities  # noqa: E402

# question_types worth mining for cross-predicate bridges (MR / TR abilities).
# _abs (abstention) types are excluded — there is no fact to bridge to.
_MINE_TYPES = frozenset({"multi-session", "temporal-reasoning"})

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "is", "are", "was", "were", "and",
    "or", "for", "with", "at", "by", "that", "this", "it", "as", "what", "where",
    "when", "which", "who", "how", "did", "does", "do", "his", "her", "their",
    "my", "your", "our", "about", "from", "has", "have", "had", "you", "i",
}


def _toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower())
            if len(t) >= 3 and t not in _STOP}


def _open_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _direct_edges(conn: sqlite3.Connection, seeds: list[str]) -> list[tuple]:
    """1-hop edges incident to any seed — what Source 1 already retrieves."""
    if not seeds:
        return []
    ph = ",".join("?" * len(seeds))
    rows = conn.execute(
        f"""SELECT subject_canonical AS s, predicate AS p, object_canonical AS o
            FROM knowledge_graph
            WHERE status='active'
              AND (subject_canonical IN ({ph}) OR object_canonical IN ({ph}))""",
        seeds + seeds,
    ).fetchall()
    return [(r["s"], r["p"], r["o"]) for r in rows]


def _far_terms(edge: tuple, seeds: set[str]) -> list[str]:
    """The endpoints that carry the answer signal — the non-seed side of the edge
    (both, if neither endpoint is a seed, e.g. a hop≥2 bridge)."""
    s, _p, o = edge
    far = [x for x in (s, o) if x not in seeds]
    return far or [s, o]


def _rank(edges: list[tuple], seeds: set[str], gold: set[str]) -> list[dict]:
    """Score each edge by gold-answer overlap of its far endpoint(s)."""
    scored = []
    for e in edges:
        ov = len(_toks(" ".join(_far_terms(e, seeds))) & gold)
        scored.append({"edge": list(e), "overlap": ov})
    scored.sort(key=lambda d: d["overlap"], reverse=True)
    return scored


def _load_questions(spec: dict | list) -> list[dict]:
    """Accept a results JSON ({'per_question': [...]}) or a bare question list."""
    if isinstance(spec, dict) and "per_question" in spec:
        rows = spec["per_question"]
    elif isinstance(spec, list):
        rows = spec
    else:
        raise SystemExit("--from must be a results JSON (per_question) or a list")
    out = []
    for i, r in enumerate(rows):
        out.append({
            "id": r.get("question_id") or r.get("id") or f"q{i}",
            "question": r.get("question", ""),
            "gold": str(r.get("answer", "")),
            "type": r.get("question_type", ""),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, type=Path,
                    help="results JSON (uses per_question) or a [{id,question,answer}] list")
    ap.add_argument("--store", required=True, type=Path,
                    help="dreamed store (read-only) holding the edges for these questions")
    ap.add_argument("--out", required=True, type=Path, help="probe JSON to write")
    ap.add_argument("--max-hops", type=int, default=3,
                    help="broad bridge enumeration depth (default 3; probe/ship tunes later)")
    ap.add_argument("--min-score", type=float, default=0.01,
                    help="broad path-score floor for enumeration (default 0.01)")
    ap.add_argument("--max-alt", type=int, default=3,
                    help="alternative bridge suggestions kept per item (for verify)")
    ap.add_argument("--limit", type=int, default=0, help="cap questions scanned (0=all)")
    ap.add_argument("--types", default=",".join(sorted(_MINE_TYPES)),
                    help="comma-separated question_types to mine (default MR+TR)")
    args = ap.parse_args()

    want_types = {t.strip() for t in args.types.split(",") if t.strip()}
    questions = _load_questions(json.loads(args.src.read_text()))
    if args.limit:
        questions = questions[:args.limit]

    conn = _open_ro(args.store)
    tmp = tempfile.TemporaryDirectory()
    from hymem import HyMemConfig
    cfg = dataclasses.replace(
        HyMemConfig(root=Path(tmp.name)),
        graph_multihop_enabled=True,
        graph_multihop_max_hops=args.max_hops,
        graph_multihop_min_score=args.min_score,
    )

    items: list[dict] = []
    stats = {"scanned": 0, "wrong_type": 0, "no_seed": 0,
             "multihop": 0, "control": 0, "dropped": 0}

    for q in questions:
        if want_types and q["type"] not in want_types:
            stats["wrong_type"] += 1
            continue
        stats["scanned"] += 1
        seeds = match_known_entities(conn, q["question"])
        if not seeds:
            stats["no_seed"] += 1
            continue
        seed_set = set(seeds)
        gold = _toks(q["gold"])

        bridge_meta = _multihop_edges(conn, cfg, seeds)   # {(s,p,o): {row,path_score,hop}}
        ranked_bridges = _rank(list(bridge_meta.keys()), seed_set, gold)
        ranked_directs = _rank(_direct_edges(conn, seeds), seed_set, gold)

        best_b = ranked_bridges[0] if ranked_bridges else None
        best_d = ranked_directs[0] if ranked_directs else None

        # multihop: a bridge explains the answer and no direct edge does.
        if best_b and best_b["overlap"] > 0 and (not best_d or best_d["overlap"] == 0):
            meta = bridge_meta[tuple(best_b["edge"])]
            items.append({
                "id": q["id"], "set": "multihop", "route": False,
                "question": q["question"], "seeds": seeds,
                "bridge": best_b["edge"],
                "_hop": meta["hop"], "_path_score": round(meta["path_score"], 4),
                "_answer_overlap": best_b["overlap"], "_gold": q["gold"],
                "_alt_bridges": [b["edge"] for b in ranked_bridges[1:1 + args.max_alt]],
                "_verify": "confirm bridge is the answer-bearing edge; "
                           "else pick from _alt_bridges or drop this item",
            })
            stats["multihop"] += 1
        # control: a direct 1-hop edge explains the answer.
        elif best_d and best_d["overlap"] > 0:
            items.append({
                "id": q["id"], "set": "control", "route": False,
                "question": q["question"], "seeds": seeds,
                "bridge": best_d["edge"],
                "_answer_overlap": best_d["overlap"], "_gold": q["gold"],
                "_verify": "confirm this direct edge answers the question",
            })
            stats["control"] += 1
        else:
            stats["dropped"] += 1

    out = {
        "description": f"Track A mined probe stub from {args.src.name} + {args.store.name}. "
                       "AUTO-PROPOSED — verify every item (_verify / _alt_bridges) before "
                       "the G-A1 read. `_`-prefixed fields are hints; the probe ignores them.",
        "generated": {"source": str(args.src), "store": str(args.store),
                       "max_hops": args.max_hops, "min_score": args.min_score,
                       "stats": stats},
        "items": items,
    }
    args.out.write_text(json.dumps(out, indent=2))
    tmp.cleanup()

    print(f"scanned {stats['scanned']} (skipped {stats['wrong_type']} wrong-type)  "
          f"→ multihop {stats['multihop']}, control {stats['control']}, "
          f"dropped {stats['dropped']} (no-seed {stats['no_seed']})", file=sys.stderr)
    print(f"wrote {len(items)} items → {args.out}", file=sys.stderr)
    if not items:
        print("WARNING: no items emitted — check the store has edges for these "
              "questions and that question_types match --types.", file=sys.stderr)


if __name__ == "__main__":
    main()
