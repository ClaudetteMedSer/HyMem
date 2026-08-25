#!/usr/bin/env python3
"""State-anchor shadow probe — Plan D Task 4 (LLM-free, read-only).

Measures the state-anchor expansion against the SHIPPING augment() pipeline
on a fixed store snapshot, without touching production: for every query it
runs the current `augment()` to collect the baseline entry keys, then runs
select_anchor_edges -> seed_terms_from_edges -> state_anchor_expand and
collects the anchor entry keys. The comparison is by entry key (chunk id).

Pre-registered gate (docs/plans/2026-08-14-state-anchor-expansion.md):
  C1  hit rate:    >=5% of sampled queries have >=1 gold-evidence row
                   reachable ONLY via anchor expansion (missed by the
                   current pipeline)
  C2  harm:        0 wrong-state pulls (anchors may not surface evidence of
                   invalidated/superseded edges)
  C3  cost:        0 LLM calls; <=1 vector call per query; added context
                   <=5 rows / <=400 tokens; latency +100ms
  C4  (Task 6 A/B) no reordering/suppression; per-category deltas within 2σ
Verdict: PASS -> flip `state_anchor_enabled`; FAIL-mechanism (C1 fails) ->
close, no score-chasing; UNMEASURED -> extend sample once or keep shadow.
Band arithmetic applies (sd≈20/√n, 2σ; per-category <±5pp = noise).

Usage:
  python benchmarks/state_anchor_probe.py --store <store.sqlite> \\
      --queries queries.json [--top-k 5] [--cap 20] [--out summary.json]
      [--embeddings]  # optional: ONE vector call per query (C3 allowance)

queries.json: a JSON array of {"question": str, "gold_chunk_ids": [str],
"category": str}. gold_chunk_ids are entry keys in the fixed store (the
rows that contain the gold evidence for that question).

The probe opens the store READ-ONLY (mode=ro) and never writes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from hymem import HyMemConfig  # noqa: E402

LOCAL_EMBED_BASE_URL = "http://localhost:8766/v1"
LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBED_DIM = 384
LOCAL_EMBED_API_KEY = "local"

_BUDGET_MAX_ROWS = 5
_BUDGET_MAX_TOKENS = 400


def _open_readonly(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row  # augment/_fts_search index columns by name
    return conn


def _wrong_state_chunk_ids(conn: sqlite3.Connection) -> set[str]:
    """Entry keys of chunks that serve as evidence for a NON-active or
    superseded edge — anchor pulls from this set are wrong-state (C2)."""
    rows = conn.execute(
        """
        SELECT DISTINCT e.chunk_id
        FROM kg_evidence e
        JOIN knowledge_graph k ON k.id = e.edge_id
        WHERE k.status != 'active' OR k.invalid_at IS NOT NULL
        """
    ).fetchall()
    return {r[0] for r in rows}


def _gold_absent(gold: set[str], store_keys: set[str]) -> set[str]:
    return gold - store_keys


def _provenance_chunk_ids(conn: sqlite3.Connection, edges) -> set[str]:
    """Chunks the SEED EDGES were themselves extracted from.

    Retrieving one of these is tautological: the anchor term came out of that
    chunk, so matching it back proves nothing about reachability. The plan
    requires these excluded and COUNTED separately — without the exclusion,
    reachability can approach 100% and mean nothing (D4).
    """
    ids = [e["id"] for e in edges if _has_id(e)]
    if not ids:
        return set()
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT DISTINCT chunk_id FROM kg_evidence WHERE edge_id IN ({marks})",
        ids,
    ).fetchall()
    return {r[0] for r in rows if r[0]}


def _has_id(edge) -> bool:
    try:
        return edge["id"] is not None
    except (KeyError, IndexError, TypeError):
        return False


def run_probe(
    store_path: str,
    queries_path: str,
    *,
    cfg: HyMemConfig | None = None,
    embedding_client: Any = None,
    cap: int = 20,
    cap_mode: str = "separate",
    top_k: int = 5,
) -> dict:
    """Run the shadow probe over a fixed store + query set. Read-only.

    Returns the summary dict (the pre-registered numbers). No verdict is
    printed inside this function — the caller (or the CLI) reads the numbers
    against the gate, so the verdict can never be re-derived silently.
    """
    from hymem.query.augment import augment
    from hymem.query.state_anchor import (
        seed_terms_from_edges,
        seed_terms_from_profile,
        select_state_anchor,
        state_anchor_expand,
    )

    queries = json.loads(Path(queries_path).read_text())
    if cfg is None:
        import tempfile

        cfg = HyMemConfig(root=Path(tempfile.mkdtemp(prefix="state-anchor-probe-")))

    conn = _open_readonly(store_path)

    # Everything the store can answer with, in one pass: all chunk ids (for
    # the gold-absent diagnostic) and the wrong-state chunk ids (C2).
    store_keys = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
    wrong_state = _wrong_state_chunk_ids(conn)

    per_query: list[dict] = []
    llm_calls = 0
    vec_calls = 0
    hit_queries = 0
    wrong_queries = 0
    headroom_queries = 0
    headroom_hits = 0
    circular_queries = 0

    for q in queries:
        question = q["question"]
        gold = set(q.get("gold_chunk_ids") or [])
        category = q.get("category", "unknown")

        # Baseline: the shipping pipeline, verbatim (no llm, no rerank).
        ctx = augment(conn, cfg, question, embedding_client=embedding_client)
        baseline_keys = {h.chunk_id for h in ctx.fts_hits}

        # Expansion: the anchor selection -> seed terms -> search, per source
        # with its own counter (Plan D correction 1). cap_mode='separate' (the
        # default and the banked config, correction 5) gives each source its
        # own cap; 'shared' reproduces the digest's profile-first squeeze, which
        # starves the edge seed and is a gate-INVALID reading of C1 — kept only
        # so the leg measured on the box 2026-08-25 stays reproducible.
        # FTS legs per source are free; the
        # optional VECTOR call runs ONCE over the combined seed text (C3:
        # <=1 vector call per query).
        t0 = time.perf_counter()
        if cap_mode == "shared":
            profiles, edges = select_state_anchor(conn, shared_cap=cap)
        else:
            profiles, edges = select_state_anchor(
                conn, edge_cap=cap, profile_cap=cap
            )
        edge_terms = seed_terms_from_edges(edges)
        profile_terms = seed_terms_from_profile(profiles)
        edge_hits = state_anchor_expand(conn, edge_terms, top_k=top_k)
        profile_hits = state_anchor_expand(conn, profile_terms, top_k=top_k)
        vec_hits: list[dict] = []
        if embedding_client is not None:
            combined = edge_terms + profile_terms
            vec_hits = state_anchor_expand(
                conn, combined, top_k=top_k, embedding_client=embedding_client
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Merge for the additive budget: union (edge + profile + vec), capped
        # at top_k — the Task 6 merge would cap before appending anyway.
        seen_keys: set[str] = set()
        anchor_hits_merged: list[dict] = []
        for h in edge_hits + profile_hits + vec_hits:
            if h["chunk_id"] not in seen_keys:
                seen_keys.add(h["chunk_id"])
                anchor_hits_merged.append(h)
                if len(anchor_hits_merged) >= top_k:
                    break

        anchor_keys = seen_keys
        fired = bool(anchor_keys)

        # HEADROOM (pre-registered, D4): the gold this query's baseline already
        # has. Without it a low hit rate is unreadable — saturation and a dead
        # mechanism produce the same number (the LME 99.8% precedent). When
        # `gold_missing` is empty there is nothing left for ANY tier to reach,
        # so the query cannot contribute to C1 and is not evidence against it.
        gold_covered_baseline = sorted(gold & baseline_keys)
        gold_missing = sorted(gold - baseline_keys)

        # CIRCULARITY (pre-registered, D4): drop gold that is merely the seed
        # edge's own provenance.
        circular = set(_provenance_chunk_ids(conn, edges))
        circular_keys = sorted((gold & anchor_keys - baseline_keys) & circular)
        anchored_only = sorted(gold & anchor_keys - baseline_keys - circular)
        anchored_only_edge = sorted(
            gold & {h["chunk_id"] for h in edge_hits} - baseline_keys - circular
        )
        anchored_only_profile = sorted(
            gold & {h["chunk_id"] for h in profile_hits} - baseline_keys - circular
        )
        wrong_state_keys = sorted(anchor_keys & wrong_state)
        added = [h for h in anchor_hits_merged if h["chunk_id"] not in baseline_keys]
        added_tokens = sum(len(h["text"].split()) for h in added)

        if anchored_only:
            hit_queries += 1
        if gold_missing:
            headroom_queries += 1
            if anchored_only:
                headroom_hits += 1
        if circular_keys:
            circular_queries += 1
        if wrong_state_keys:
            wrong_queries += 1
        llm_calls += 0
        vec_calls += 1 if embedding_client is not None else 0

        per_query.append({
            "question": question,
            "category": category,
            "fired": fired,
            "anchored_only_keys": anchored_only,
            "anchored_only_edge_keys": anchored_only_edge,
            "anchored_only_profile_keys": anchored_only_profile,
            "gold_covered_baseline_keys": gold_covered_baseline,
            "gold_missing_baseline_keys": gold_missing,
            "circular_keys": circular_keys,
            "wrong_state_keys": wrong_state_keys,
            "added_rows": len(added),
            "added_tokens": added_tokens,
            "anchor_rows": len(anchor_hits_merged),
            "baseline_rows": len(baseline_keys),
            "vec_calls": 1 if embedding_client is not None else 0,
            "latency_ms": round(latency_ms, 2),
        })

    n = len(queries)
    lat = sorted(r["latency_ms"] for r in per_query)

    def _pct(p: float) -> float:
        if not lat:
            return 0.0
        idx = min(len(lat) - 1, int(p * len(lat)))
        return lat[idx]

    by_cat: dict[str, dict] = {}
    edge_hits_q = profile_hits_q = 0
    for r in per_query:
        c = by_cat.setdefault(r["category"], {"n": 0, "hits": 0})
        c["n"] += 1
        if r["anchored_only_keys"]:
            c["hits"] += 1
        if r["anchored_only_edge_keys"]:
            edge_hits_q += 1
        if r["anchored_only_profile_keys"]:
            profile_hits_q += 1

    summary = {
        "n_queries": n,
        "hit_rate": (hit_queries / n) if n else 0.0,
        # --- headroom: read these BEFORE hit_rate ---
        "headroom_queries": headroom_queries,
        "headroom_rate": (headroom_queries / n) if n else 0.0,
        "hit_rate_within_headroom": (
            (headroom_hits / headroom_queries) if headroom_queries else 0.0
        ),
        "c1_ceiling": (headroom_queries / n) if n else 0.0,
        "circular_queries": circular_queries,
        "hit_rate_edge_source": (edge_hits_q / n) if n else 0.0,
        "hit_rate_profile_source": (profile_hits_q / n) if n else 0.0,
        "wrong_state_rate": (wrong_queries / n) if n else 0.0,
        "llm_calls": llm_calls,
        "vec_calls": vec_calls,
        "max_added_rows": max((r["added_rows"] for r in per_query), default=0),
        "max_added_tokens": max((r["added_tokens"] for r in per_query), default=0),
        "latency_ms_p50": round(_pct(0.5), 2),
        "latency_ms_p95": round(_pct(0.95), 2),
        "per_category": by_cat,
        "budget_rows": _BUDGET_MAX_ROWS,
        "budget_tokens": _BUDGET_MAX_TOKENS,
    }
    # gold-absent diagnostic: ids referenced by queries but missing from the store
    absent = sorted(
        set().union(*[set(q.get("gold_chunk_ids") or []) for q in queries]) - store_keys
    )
    summary["gold_absent_ids"] = absent
    conn.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, type=Path,
                    help="fixed store snapshot (sqlite), opened READ-ONLY")
    ap.add_argument("--queries", required=True, type=Path,
                    help='JSON array of {"question", "gold_chunk_ids", "category"}')
    ap.add_argument("--top-k", type=int, default=5,
                    help="anchor rows per query (C3 budget; default 5)")
    ap.add_argument("--cap", type=int, default=20,
                    help="anchor cap (matches aggregation_digest_anchor_facts)")
    ap.add_argument("--cap-mode", choices=("separate", "shared"), default="separate",
                    help="'separate' (default, the banked Plan D config): profile "
                         "and edges each get --cap. 'shared': profile consumes "
                         "--cap first and edges fill the remainder, as the digest "
                         "does — starves the edge seed, C1 unreadable.")
    ap.add_argument("--embeddings", action="store_true",
                    help="enable the single optional vector call per query (C3 allowance)")
    ap.add_argument("--embedding-base-url", default=None,
                    help="override embedding endpoint (default LOCAL_EMBED_BASE_URL)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write summary JSON (default: stdout)")
    args = ap.parse_args()

    embedding_client = None
    if args.embeddings:
        import os

        from hymem.contrib.openai_embedding_client import (
            OpenAICompatibleEmbeddingClient,
        )
        env = os.environ.get
        embedding_client = OpenAICompatibleEmbeddingClient(
            api_key=env("HYMEM_EMBEDDING_API_KEY") or LOCAL_EMBED_API_KEY,
            base_url=args.embedding_base_url
            or env("HYMEM_EMBEDDING_BASE_URL") or LOCAL_EMBED_BASE_URL,
            model=env("HYMEM_EMBEDDING_MODEL") or LOCAL_EMBED_MODEL,
            dim=int(env("HYMEM_EMBEDDING_DIM") or LOCAL_EMBED_DIM),
        )

    summary = run_probe(
        str(args.store), str(args.queries),
        cap=args.cap, cap_mode=args.cap_mode, top_k=args.top_k,
        embedding_client=embedding_client,
    )

    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"\n[summary saved: {args.out}]")


if __name__ == "__main__":
    main()
