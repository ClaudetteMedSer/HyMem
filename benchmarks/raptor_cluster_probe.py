#!/usr/bin/env python3
"""Front-run gate for Phase 2 (RAPTOR cross-session aggregation nodes).

The RAPTOR hypothesis: the MS *synthesis* residual (gold turns ARE in context,
model still fails to fuse 45 raw slots — the L3 "fully-covered" bucket, D4
out-of-scope at the reader) is fixable UPSTREAM by pre-computing hierarchical
aggregation nodes, so the answer model fuses ~3 cluster summaries instead of 45
raw slots. Dreaming already makes per-session episodes; the gap is *cross-session*
aggregation.

RAPTOR can only help if clustering CO-LOCATES the synthesis inputs: the episodes
carrying a question's gold turns must fall into a SINGLE cluster, so one
aggregation-node summary captures everything the model must fuse. If the gold
episodes scatter across many clusters, the node summaries don't co-locate them
and the model is no better off than with raw slots → RAPTOR is dead on arrival.

This probe measures exactly that, BEFORE any aggregation_nodes table / migration
v16 / dreaming wiring is built. It mirrors the L3 gate that killed diversity-pack
(see gold_rank_probe.py): a cheap offline diagnostic that decides the phase.

UNLIKE gold_rank_probe (LLM-free — messages_fts is built at ingest), this probe's
unit is the EPISODE, which only exists after a dream pass. So the real run needs
the dream pipeline (real LLM + embeddings) and belongs on the Hermes box, same as
a full LME run minus the answer+judge spend. The clustering itself
(`cluster_episodes`) is a pure function, unit-tested offline on the Mac
(test_raptor_cluster_probe.py) — that is the part Phase-2 build logic will reuse.

What it does, per MS question (optionally only the run's synthesis misses):
  1. ingest the haystack → dream → episodes (+ key_entities, episode_embeddings)
  2. map each gold turn → the episode whose message range contains it
  3. cluster ALL episodes across sessions: union-find, two episodes linked when
     cosine(embedding) ≥ --emb-threshold  OR  jaccard(key_entities) ≥ --ent-threshold
     ("reuse episode_embeddings / entity overlap", per the roadmap)
  4. record whether the gold-bearing episodes land in ONE cluster (co-located)

Read the verdict like this (over the probed MS misses):
  CO-LOCATED high (gold episodes share one cluster in most misses)
        → an aggregation node WOULD bundle the synthesis inputs → RAPTOR has a
          shot → proceed to build (migration v16 + dreaming/aggregate.py).
  CO-LOCATED low (gold episodes scatter across clusters)
        → no single node fuses the answer; RAPTOR summaries won't help the model
          synthesize → BANK like L3: keep this writeup, do NOT build.
  UNMAPPED high (gold turns produce no episode at all)
        → a dream-COVERAGE gap, not a clustering gap. RAPTOR aggregates episodes;
          if the synthesis turns never become episodes, aggregation can't see them.
          This is a separate, prior problem (episode extraction recall) and also
          argues against building RAPTOR before it is fixed.

CAVEATS (same spirit as gold_rank_probe's):
  - Co-location is threshold-dependent: a single emb/ent threshold over-merges (one
    giant cluster → trivially "co-located", useless) or under-merges (singletons →
    never co-located). Sweep --grid and read the band where mean cluster count is
    sane (≈ a few per question), not the degenerate ends. The build's clustering
    will use whatever threshold this sweep shows actually separates the gold.
  - The probe clusters the episodes a real dream produced, but a different episode
    granularity (Phase-2 may re-segment) would shift boundaries. Treat the number
    as directional, not a guarantee — exactly the gold_rank_probe contract.
  - Gold→episode mapping is normalized-substring (the _gold_in_pool contract); a
    gold turn split across two episodes is attributed to the first range that
    contains its message id.

Usage (run on the Hermes box, from benchmarks/):
  python raptor_cluster_probe.py --category multi-session --sample 0 --seed 0
  # focus on the synthesis misses of a real run (the RAPTOR target band):
  python raptor_cluster_probe.py --category multi-session --sample 0 \
      --join-run ~/.hermes/benchmarks/longmemeval-...-baseline.json
  # threshold sweep:
  python raptor_cluster_probe.py --category multi-session \
      --grid 0.55:0.50,0.65:0.50,0.75:0.40
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Sibling import: reuse the adapter's data loader + gold helpers verbatim so the
# probe and the real run agree on what "gold" and "in pool" mean.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from longmemeval_adapter import (  # noqa: E402
    HyMemAdapter,
    _extract_gold_turns,
    _gold_in_pool,
    _norm_text,
    load_longmemeval_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure clustering core — canonical home is hymem.dreaming.aggregate (the Phase-2
# build), re-exported here so probe, unit tests (test_raptor_cluster_probe), and
# production all run the SAME clusterer and can never silently drift. The DB-side
# probe helpers below stay local (they're benchmark-only).
# ─────────────────────────────────────────────────────────────────────────────
from hymem.dreaming.aggregate import (  # noqa: E402
    _cosine,
    _jaccard,
    _linked,
    cluster_episodes,
)


# ─────────────────────────────────────────────────────────────────────────────
# DB-side helpers (Hermes box only — operate on a dreamed temp DB).
# ─────────────────────────────────────────────────────────────────────────────

def _load_episodes(conn) -> list[dict]:
    """All episodes in the dreamed DB with their vector + entity set."""
    from hymem.core.vectors import decode_vector

    rows = conn.execute(
        """
        SELECT e.id, e.session_id, e.start_message_id, e.end_message_id,
               e.key_entities, em.vector_json
        FROM episodes e
        LEFT JOIN episode_embeddings em ON em.episode_id = e.id
        """
    ).fetchall()
    episodes: list[dict] = []
    for r in rows:
        try:
            entities = set(json.loads(r["key_entities"] or "[]"))
        except (ValueError, TypeError):
            entities = set()
        vec = decode_vector(r["vector_json"]) if r["vector_json"] else None
        episodes.append({
            "id": r["id"],
            "session_id": r["session_id"],
            "start": r["start_message_id"],
            "end": r["end_message_id"],
            "entities": {_norm_text(x) for x in entities if x},
            "vector": vec,
        })
    return episodes


def _gold_episode_ids(conn, gold_turns: list[str], episodes: list[dict]) -> tuple[set[str], int]:
    """Map gold turns → the episode ids whose message range contains them.

    Returns (episode_ids, n_unmapped) where n_unmapped counts gold turns that
    matched a message but no episode range covered it (a dream-coverage gap, kept
    distinct from a clustering failure). A gold turn whose message we can't even
    find is also counted unmapped.
    """
    msgs = conn.execute("SELECT id, session_id, content FROM messages").fetchall()
    # Pre-normalize once.
    msg_rows = [(m["id"], m["session_id"], _norm_text(m["content"])) for m in msgs]

    ep_ids: set[str] = set()
    unmapped = 0
    for g in gold_turns:
        gn = _norm_text(g)
        if not gn:
            continue
        hit = None
        for mid, sid, mn in msg_rows:
            if gn in mn or mn in gn or (len(gn) >= 40 and gn[:40] in mn):
                hit = (mid, sid)
                break
        if hit is None:
            unmapped += 1
            continue
        mid, sid = hit
        covering = [
            e["id"] for e in episodes
            if e["session_id"] == sid
            and e["start"] is not None and e["end"] is not None
            and e["start"] <= mid <= e["end"]
        ]
        if covering:
            ep_ids.update(covering)
        else:
            unmapped += 1
    return ep_ids, unmapped


def _probe_question(q: dict, emb_threshold: float, ent_threshold: float) -> dict | None:
    """Ingest → dream → cluster one question. Returns a record or None on no-gold."""
    gold_turns, _mode = _extract_gold_turns(q)
    if not gold_turns:
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-raptorprobe-"))
    hy = None
    try:
        hy = HyMemAdapter(tmp_dir / "hymem.sqlite", api_key="probe", embeddings=True)
        hy.open()
        sessions = q.get("haystack_sessions", [])
        session_ids = q.get("haystack_session_ids",
                            [str(i) for i in range(len(sessions))])
        hy.ingest_sessions(sessions, session_ids, q.get("haystack_dates", []))
        hy.dream_and_wait()                      # ← the expensive, box-only step
        episodes = _load_episodes(hy.hy.conn)
        gold_eps, unmapped = _gold_episode_ids(hy.hy.conn, gold_turns, episodes)
        labels = cluster_episodes(episodes, emb_threshold, ent_threshold)
    finally:
        if hy:
            hy.close()

    gold_clusters = {labels[e] for e in gold_eps if e in labels}
    n_clusters = len(set(labels.values())) if labels else 0
    return {
        "question_id": q.get("question_id"),
        "question_type": q.get("question_type"),
        "n_gold_turns": len(gold_turns),
        "n_gold_episodes": len(gold_eps),
        "n_unmapped_gold": unmapped,
        "n_episodes": len(episodes),
        "n_clusters": n_clusters,
        "n_gold_clusters": len(gold_clusters),
        # Co-located iff every gold episode shares one cluster (and there is ≥1
        # gold episode). An all-unmapped question is NOT co-located — there is
        # nothing for an aggregation node to bundle.
        "co_located": len(gold_eps) > 0 and len(gold_clusters) == 1,
        "has_gold_episodes": len(gold_eps) > 0,
        # Audit trail so a borderline verdict is inspectable without a re-dream:
        # which episodes carried gold, and which cluster label each fell into.
        "gold_episode_ids": sorted(gold_eps),
        "gold_episode_clusters": {e: labels[e] for e in sorted(gold_eps) if e in labels},
    }


def _summarize(records: list[dict], emb_t: float, ent_t: float) -> None:
    n = len(records)
    if not n:
        print("  (no probed questions)")
        return
    with_eps = [r for r in records if r["has_gold_episodes"]]
    co = [r for r in with_eps if r["co_located"]]
    all_unmapped = [r for r in records if not r["has_gold_episodes"]]
    mean_clusters = sum(r["n_clusters"] for r in records) / n
    mean_gold_clusters = (
        sum(r["n_gold_clusters"] for r in with_eps) / len(with_eps) if with_eps else 0.0
    )
    co_rate = (100 * len(co) / len(with_eps)) if with_eps else 0.0
    print(f"  emb≥{emb_t:.2f} OR ent≥{ent_t:.2f}:")
    print(f"    questions={n}   with-gold-episodes={len(with_eps)}   "
          f"all-gold-unmapped={len(all_unmapped)}")
    print(f"    mean clusters/question={mean_clusters:.1f}   "
          f"mean gold-clusters (of with-eps)={mean_gold_clusters:.2f}")
    print(f"    CO-LOCATED (gold episodes in ONE cluster): "
          f"{len(co)}/{len(with_eps)} ({co_rate:.0f}% of with-eps)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="LongMemEval json (defaults to the adapter's resolver)")
    ap.add_argument("--category", default="multi-session",
                    help="question_type filter, or 'all'")
    ap.add_argument("--sample", type=int, default=0,
                    help="max questions (0 = all in category)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-threshold", type=float, default=0.65)
    ap.add_argument("--ent-threshold", type=float, default=0.50)
    ap.add_argument("--grid", default=None,
                    help="sweep 'emb:ent,emb:ent,...' (overrides single thresholds)")
    ap.add_argument("--join-run", default=None,
                    help="run json; restrict to its FULLY-COVERED synthesis misses "
                         "(recall_ceiling=True ranking misses) — the RAPTOR target band")
    ap.add_argument("--dump", default=None,
                    help="write per-question records (one block per grid point) to this "
                         "json path, so a borderline verdict is auditable without a re-dream")
    args = ap.parse_args()

    questions = load_longmemeval_data(args.dataset, max_questions=None, seed=args.seed)
    if args.category != "all":
        questions = [q for q in questions if q.get("question_type") == args.category]

    # Optional join: keep only the ranking misses of a real run (gold was retrieved
    # but the answer was wrong → the synthesis band RAPTOR targets, not retrieval).
    if args.join_run:
        with open(args.join_run) as f:
            run = json.load(f)
        pq = run.get("per_question", [])
        synth_ids = {
            r.get("question_id") for r in pq
            if not r.get("correct") and r.get("recall_ceiling") is True
            and (args.category == "all" or r.get("question_type") == args.category)
        }
        questions = [q for q in questions if q.get("question_id") in synth_ids]
        print(f"Joined {Path(args.join_run).name}: "
              f"{len(questions)} synthesis (fully-covered) misses in {args.category}")

    if args.sample:
        questions = questions[:args.sample]

    grid = (
        [tuple(float(x) for x in pair.split(":")) for pair in args.grid.split(",")]
        if args.grid else [(args.emb_threshold, args.ent_threshold)]
    )

    print(f"\nRAPTOR co-location probe (Phase-2 front-run gate)")
    print(f"  Category: {args.category}   Questions: {len(questions)}   Seed: {args.seed}")
    print(f"  (per question: ingest → dream → episode cluster; box-only, no answer/judge)\n",
          flush=True)

    # Dream once per question per threshold-set would be wasteful; dream once, then
    # re-cluster the cached episodes for each grid point. We do that by probing at the
    # FIRST grid point (which dreams) and re-clustering in-memory is not possible here
    # because episodes live in a per-question temp DB torn down after _probe_question.
    # So for a grid we re-ingest per point — acceptable for a one-shot gate; pass a
    # single threshold for the cheap path and only --grid when locating the band.
    dump_blocks: list[dict] = []
    for emb_t, ent_t in grid:
        records: list[dict] = []
        for qi, q in enumerate(questions):
            try:
                rec = _probe_question(q, emb_t, ent_t)
            except Exception as e:                # noqa: BLE001 — one bad question shouldn't abort the gate
                print(f"  [{qi+1}] {q.get('question_id','?')} ERROR: {e}", flush=True)
                continue
            if rec is not None:
                records.append(rec)
            if (qi + 1) % 10 == 0:
                print(f"  …{qi+1}/{len(questions)} probed (emb={emb_t}, ent={ent_t})",
                      flush=True)
        print("=" * 64)
        _summarize(records, emb_t, ent_t)
        print()
        if args.dump:
            dump_blocks.append({
                "emb_threshold": emb_t,
                "ent_threshold": ent_t,
                "records": records,
            })

    if args.dump:
        payload = {
            "config": {
                "category": args.category,
                "seed": args.seed,
                "sample": args.sample,
                "join_run": args.join_run,
                "n_questions": len(questions),
            },
            "grid": dump_blocks,
        }
        Path(args.dump).write_text(json.dumps(payload, indent=2))
        print(f"Per-question records written to {args.dump}\n")

    print("VERDICT GUIDE:")
    print("  high CO-LOCATED %% in a sane-cluster band (mean clusters ≈ a few)")
    print("      → an aggregation node bundles the synthesis inputs → BUILD Phase 2.")
    print("  low CO-LOCATED %% (gold scatters across clusters)")
    print("      → RAPTOR summaries won't co-locate the answer → BANK like L3.")
    print("  high all-gold-unmapped → episode-extraction coverage gap, a prior problem.")


if __name__ == "__main__":
    main()
