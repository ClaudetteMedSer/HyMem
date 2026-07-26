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

Two modes:

  • STORE MODE (`--store`, LLM-free, seconds) — mine an existing dreamed store.
    Fast, but a combined LME store must be dreamed TO COMPLETION first (one
    `dream()` only drains `cfg.dream_budget`=50 chunks, so a mega-store is ~1%
    dreamed and yields a false-empty graph — loop `dream()` until
    `not report.budget_exhausted`).

    python benchmarks/multihop_miner.py \
      --from ~/.hermes/benchmarks/<run>.json --store STORE.sqlite --out SLICE.json

  • PER-QUESTION MODE (`--lme-data`, LLM-bound) — rebuild+dream each question's
    OWN haystack (small → one/few cycles fully drain it), then mine it. Sidesteps
    the mega-store budget trap entirely and is faithful to how LME retrieves
    (isolated per-question store). One dream per question; `--from` optionally
    restricts to a run's qids. Dreams with `--dream-model` (default
    deepseek-v4-flash — thinking MUST be disabled, the box's patched client);
    `--dream-model stub` is a no-op plumbing test.

    python benchmarks/multihop_miner.py \
      --lme-data <longmemeval.json> --types multi-session --limit 40 \
      --out SLICE.json

Then hand-verify SLICE.json (short — every item carries _verify / _alt_bridges)
and probe:
  # STORE mode — reuse the same store:
  python benchmarks/multihop_probe.py --probe SLICE.json --store STORE.sqlite --verbose
  # PER-QUESTION mode — SLICE.json carries a fresh-seed `edges` block (the union of
  # the emitted items' store edges), so it is self-contained; NO --store:
  python benchmarks/multihop_probe.py --probe SLICE.json --verbose
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


def _mine_question(conn, q: dict, cfg, max_alt: int) -> tuple[dict | None, str]:
    """Propose + classify one question. Returns (probe_item_or_None, category)
    where category ∈ {no_seed, multihop, control, dropped}. Shared by both modes."""
    seeds = match_known_entities(conn, q["question"])
    if not seeds:
        return None, "no_seed"
    seed_set = set(seeds)
    gold = _toks(q["gold"])

    bridge_meta = _multihop_edges(conn, cfg, seeds)   # {(s,p,o): {row,path_score,hop}}
    ranked_bridges = _rank(list(bridge_meta.keys()), seed_set, gold)
    ranked_directs = _rank(_direct_edges(conn, seeds), seed_set, gold)
    best_b = ranked_bridges[0] if ranked_bridges else None
    best_d = ranked_directs[0] if ranked_directs else None

    # multihop: a hop≥2 bridge explains the answer and no direct edge does.
    if best_b and best_b["overlap"] > 0 and (not best_d or best_d["overlap"] == 0):
        meta = bridge_meta[tuple(best_b["edge"])]
        return {
            "id": q["id"], "set": "multihop", "route": False,
            "question": q["question"], "seeds": seeds, "bridge": best_b["edge"],
            "_hop": meta["hop"], "_path_score": round(meta["path_score"], 4),
            "_answer_overlap": best_b["overlap"], "_gold": q["gold"],
            "_alt_bridges": [b["edge"] for b in ranked_bridges[1:1 + max_alt]],
            "_verify": "confirm bridge is the answer-bearing edge; "
                       "else pick from _alt_bridges or drop this item",
        }, "multihop"
    # control: a direct 1-hop edge explains the answer.
    if best_d and best_d["overlap"] > 0:
        return {
            "id": q["id"], "set": "control", "route": False,
            "question": q["question"], "seeds": seeds, "bridge": best_d["edge"],
            "_answer_overlap": best_d["overlap"], "_gold": q["gold"],
            "_verify": "confirm this direct edge answers the question",
        }, "control"
    return None, "dropped"


def _run_store_mode(args, want_types, mine_cfg, items, stats) -> None:
    """Mine against an existing dreamed store (read-only, LLM-free)."""
    if not args.src:
        raise SystemExit("--store mode needs --from (gold answers for classification)")
    questions = _load_questions(json.loads(args.src.read_text()))
    if args.limit:
        questions = questions[:args.limit]
    conn = _open_ro(args.store)
    for q in questions:
        if want_types and q["type"] not in want_types:
            stats["wrong_type"] += 1
            continue
        stats["scanned"] += 1
        item, cat = _mine_question(conn, q, mine_cfg, args.max_alt)
        stats[cat] += 1
        if item:
            items.append(item)


def _build_dream_llm(model: str, base_url: str | None, api_key: str | None):
    """Extraction LLM for per-question dreaming. `stub` = no-op plumbing test."""
    if model == "stub":
        from hymem.extraction.llm import StubLLMClient
        return StubLLMClient(default="[]")
    if "chat" in (model or "") and "v4" not in (model or ""):
        print(f"WARNING: dream model '{model}' looks like the deprecated "
              "deepseek-chat — extraction will fail. Use deepseek-v4-flash "
              "(thinking disabled).", file=sys.stderr)
    from hymem.contrib.openai_client import OpenAICompatibleClient
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)


def _dump_edges(conn, edges_block: dict) -> None:
    """Accumulate a store's active edges into a fresh-seed `edges` block, so a
    per-question item survives its (deleted) temp store and the probe can
    reproduce it without --store. Deduped by (s,p,o) across questions."""
    for r in conn.execute(
        "SELECT subject_canonical s, predicate p, object_canonical o, "
        "pos_evidence pos, neg_evidence neg FROM knowledge_graph WHERE status='active'"
    ):
        edges_block[(r["s"], r["p"], r["o"])] = {
            "subject": r["s"], "predicate": r["p"], "object": r["o"],
            "pos": int(r["pos"]), "neg": int(r["neg"]),
        }


def _store_health(conn) -> tuple[int, int]:
    row = conn.execute(
        "SELECT COUNT(*) e, COUNT(DISTINCT subject_canonical) s "
        "FROM knowledge_graph WHERE status='active'"
    ).fetchone()
    return int(row[0]), int(row[1])


def _ingest_and_dream(hy, qd: dict, normalize_date, max_cycles: int) -> int:
    """Ingest a question's haystack and dream it TO COMPLETION — loop dream()
    until it stops hitting the per-cycle `dream_budget` (the exact cap that
    silently under-dreams a mega-store). Returns the cycle count."""
    sessions = qd.get("haystack_sessions", []) or []
    ids = qd.get("haystack_session_ids") or [str(i) for i in range(len(sessions))]
    dates = qd.get("haystack_dates", []) or []
    for idx, (sid, messages) in enumerate(zip(ids, sessions)):
        date = normalize_date(dates[idx]) if idx < len(dates) else None
        entries = [(m.get("role", "user"), m.get("content", ""), date)
                   for m in messages if (m.get("content") or "").strip()]
        for i in range(0, len(entries), 50):
            hy.log_messages(f"{sid}_{i // 50}", entries[i:i + 50])
    cycles = 0
    while cycles < max_cycles:
        report = hy.dream()
        cycles += 1
        if not report.budget_exhausted:
            break
    return cycles


def _run_perq_mode(args, want_types, mine_cfg, items, stats, health, edges_block) -> None:
    """Rebuild + dream each question's OWN haystack, then mine it. Each store is
    small so one/few dream cycles drain it — sidesteps the mega-store budget cap.
    LLM-bound (one dream per question)."""
    import shutil
    from longmemeval_adapter import load_longmemeval_data, _normalize_date
    from hymem import HyMem, HyMemConfig

    dream_llm = _build_dream_llm(args.dream_model, args.dream_base_url, args.dream_api_key)
    qdatas = load_longmemeval_data(str(args.lme_data))

    keep = None
    if args.src:  # optional qid filter — restrict to a specific run's questions
        src = json.loads(args.src.read_text())
        rows = src.get("per_question", src) if isinstance(src, dict) else src
        keep = {(r.get("question_id") or r.get("id")) for r in rows}

    selected = []
    for qd in qdatas:
        if want_types and qd.get("question_type", "") not in want_types:
            continue
        if keep is not None and qd.get("question_id") not in keep:
            continue
        selected.append(qd)
        if args.limit and len(selected) >= args.limit:
            break

    print(f"[perq] rebuilding+dreaming {len(selected)} questions "
          f"(model={args.dream_model})", file=sys.stderr)
    for n, qd in enumerate(selected, 1):
        stats["scanned"] += 1
        tmpd = tempfile.mkdtemp(prefix="hymem-mine-")
        hy = None
        try:
            hy = HyMem(HyMemConfig(root=Path(tmpd)), llm=dream_llm)
            cycles = _ingest_and_dream(hy, qd, _normalize_date, args.max_dream_cycles)
            edges, subj = _store_health(hy.conn)
            health["stores"] += 1
            health["edges_total"] += edges
            health["subjects_max"] = max(health["subjects_max"], subj)
            q = {"id": qd.get("question_id"), "question": qd.get("question", ""),
                 "gold": str(qd.get("answer", "")), "type": qd.get("question_type", "")}
            item, cat = _mine_question(hy.conn, q, mine_cfg, args.max_alt)
            stats[cat] += 1
            if item:
                items.append(item)
                _dump_edges(hy.conn, edges_block)   # self-contained probe input
            print(f"  [{n}/{len(selected)}] {q['id']}: {edges} edges / {subj} subj, "
                  f"{cycles} dream cycle(s) → {cat}", file=sys.stderr)
        finally:
            if hy:
                hy.close()
            shutil.rmtree(tmpd, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, default=None,
                    help="STORE mode: REQUIRED — results JSON (per_question) or "
                         "[{id,question,answer}] list (gold answers). PER-QUESTION mode: "
                         "OPTIONAL qid filter (restrict to a run's questions).")
    ap.add_argument("--store", type=Path, default=None,
                    help="STORE mode: an existing dreamed store (read-only) with the edges.")
    ap.add_argument("--lme-data", type=Path, default=None,
                    help="PER-QUESTION mode: raw LME dataset — rebuild+dream each question's "
                         "own haystack to completion (sidesteps the dream_budget under-dream).")
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
    # per-question dreaming
    ap.add_argument("--dream-model", default="deepseek-v4-flash",
                    help="per-question mode: extraction LLM. Thinking MUST be disabled for "
                         "v4-flash (the box's patched openai_client). 'stub' = no-op plumbing test.")
    ap.add_argument("--dream-base-url", default=None,
                    help="per-question mode: extraction endpoint (else env HYMEM_LLM_BASE_URL)")
    ap.add_argument("--dream-api-key", default=None,
                    help="per-question mode: extraction API key (else env)")
    ap.add_argument("--max-dream-cycles", type=int, default=1000,
                    help="per-question mode: safety cap on dream() cycles per question")
    args = ap.parse_args()

    if bool(args.store) == bool(args.lme_data):
        ap.error("pass exactly one of --store (existing dreamed store) or "
                 "--lme-data (per-question rebuild)")

    want_types = {t.strip() for t in args.types.split(",") if t.strip()}
    from hymem import HyMemConfig
    cfgtmp = tempfile.TemporaryDirectory()
    mine_cfg = dataclasses.replace(
        HyMemConfig(root=Path(cfgtmp.name)),
        graph_multihop_enabled=True,
        graph_multihop_max_hops=args.max_hops,
        graph_multihop_min_score=args.min_score,
    )

    items: list[dict] = []
    stats = {"scanned": 0, "wrong_type": 0, "no_seed": 0,
             "multihop": 0, "control": 0, "dropped": 0}
    health = {"stores": 0, "edges_total": 0, "subjects_max": 0}
    edges_block: dict = {}

    mode = "store" if args.store else "per-question"
    if args.store:
        _run_store_mode(args, want_types, mine_cfg, items, stats)
        source_desc = f"{args.src.name} + store {args.store.name}"
    else:
        _run_perq_mode(args, want_types, mine_cfg, items, stats, health, edges_block)
        source_desc = f"per-question rebuild from {args.lme_data.name}"
    cfgtmp.cleanup()

    gen = {"mode": mode, "max_hops": args.max_hops,
           "min_score": args.min_score, "stats": stats}
    if mode == "per-question":
        gen["store_health"] = health
    out = {
        "description": f"Track A mined probe stub ({source_desc}). AUTO-PROPOSED — verify "
                       "every item (_verify / _alt_bridges) before the G-A1 read. "
                       "`_`-prefixed fields are hints; the probe ignores them.",
        "generated": gen,
        "items": items,
    }
    # Per-question mode: emit a fresh-seed `edges` block so SLICE.json is
    # self-contained (temp stores are gone) — probe with NO --store.
    if mode == "per-question" and edges_block:
        out["edges"] = list(edges_block.values())
    args.out.write_text(json.dumps(out, indent=2))

    print(f"[{mode}] scanned {stats['scanned']} → multihop {stats['multihop']}, "
          f"control {stats['control']}, dropped {stats['dropped']} "
          f"(no-seed {stats['no_seed']}, wrong-type {stats['wrong_type']})", file=sys.stderr)
    if mode == "per-question" and health["stores"]:
        avg = health["edges_total"] / health["stores"]
        print(f"[per-question] dreamed {health['stores']} stores — "
              f"{health['edges_total']} edges total (avg {avg:.1f}/store, "
              f"max {health['subjects_max']} subjects). If avg edges is tiny, the "
              "dreams under-ran — check --dream-model / thinking-disable.", file=sys.stderr)
    print(f"wrote {len(items)} items → {args.out}", file=sys.stderr)
    if not items:
        print("WARNING: no items emitted — store under-dreamed, entities not matched, "
              "or types don't match --types.", file=sys.stderr)


if __name__ == "__main__":
    main()
