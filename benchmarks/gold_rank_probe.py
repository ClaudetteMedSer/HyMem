#!/usr/bin/env python3
"""Front-run diagnostic for lever L2a (rerank candidate budget).

The L2a hypothesis: the ~190 ranking misses (MS dominant) are gold turns that
sit BELOW the reranker's candidate window. Today the message tier pulls
`max(message_fts_top_k=15, rerank_top_k=20)=20` BM25 candidates and reranks to
15, so a gold turn at BM25 rank 25 can never be lifted — it is never a candidate.
Widening `rerank_top_k` only helps if the gold actually lives in the reachable
band. This script measures WHERE the gold sits in the raw BM25 ranking, per
category, WITHOUT spending a full retrieve+answer+judge run.

It is deliberately LLM-free and embedding-free (router_eval.py pattern):
  - ingest is pure persistence (messages_fts is built at ingest, no dream, no LLM);
  - retrieval is a single direct `_message_fts_search` over raw messages (pure BM25).
So it runs in minutes on the box and decides L2a before any compute is spent.

CAVEAT — this is a LOWER BOUND on combined-pool rank, not the real cut.
The probe measures rank in the message FTS tier ONLY (the tier L2a widens). The
real augment pipeline FUSES message hits + chunk hits + graph hits + MR
aggregation before the top_k cut, so a gold turn at message-BM25 rank 35 may
surface at combined rank 3 (e.g. dreamed into a high-salience chunk) and already
be reachable at rerank_top_k=20. Cross-tier recovery only rescues ~2 turns per
the L1 analysis, so the bias is small in practice — but it makes the probe
PESSIMISTIC: it may recommend L2a when the gold is already reachable. For a gate
that is the safe direction (an extra experiment beats a skipped one); just don't
read the histogram as ground-truth combined rank.

Read the histogram like this:
  rank ≤15           already inside the default cut — not a budget problem
  rank 16–20         in today's pool (20) but trimmed to 15 — a reranker can save it now
  rank 21–40         L2a --rerank-top-k 40 brings it into the pool  ← the L2a target band
  rank 41–60         needs --rerank-top-k 60
  rank 61+           a bigger hammer (pool) — diminishing returns
  NOT in BM25 top-N  gold is NOT in the raw MESSAGE-FTS top-N at any budget.
                     This bucket FLAGS the redundancy question "does a wider BM25
                     pool subsume what embeddings were adding?" — if it is ~0,
                     BM25@40/60 covers everything message-tier vec did and the
                     vector path is a candidate to drop for LME.
                     It does NOT ANSWER it. Same cross-tier caveat as above: a
                     turn missing from message-BM25 top-N may still be recovered
                     by CHUNK embeddings, so this bucket is conservative on the
                     "keep embeddings" side. A non-zero count here is NOT proof
                     embeddings are essential — confirm that with a real
                     embeddings-on run (the L1 results showed no vec-only bucket),
                     not with this probe alone.

Usage (run on the Hermes box, from benchmarks/):
  python gold_rank_probe.py --category multi-session --sample 0 --seed 0
  python gold_rank_probe.py --category all --sample 0 --seed 0   # every category

--coverage mode (L3 gate — multi-gold window coverage, NOT single-best-turn rank):
  The rank histogram above reports the MIN gold rank (the best turn), which is why MS
  read so clean (median 2). But MS answers need SEVERAL gold turns across sessions, and
  the recall diagnostic's "ranking miss" label fires on ANY gold turn in the pool — so it
  silently bundles coverage-short (gold squeezed out of the 15-slot message window by a
  verbose session) with fully-covered-but-model-fails (synthesis). --coverage measures how
  many of a question's N gold turns fit the cut + the window's session concentration, and
  --join-run splits a run's ranking misses into coverage-short (L3-fixable) vs fully-covered
  (synthesis, out of scope) — the gate that decides whether L3 packing is worth building:
  python gold_rank_probe.py --coverage --category multi-session --sample 0 \
      --join-run ~/.hermes/benchmarks/longmemeval-...-baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# Sibling import: reuse the adapter's data loader + gold helpers verbatim so the
# probe and the real run agree on what "gold" and "in pool" mean.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from longmemeval_adapter import (  # noqa: E402
    HyMemAdapter,
    _extract_gold_turns,
    _gold_in_pool,
    load_longmemeval_data,
)

# Buckets are (label, lower_inclusive, upper_inclusive); upper None = open-ended.
# Boundaries chosen to map directly onto the L2a sweep points (20 / 40 / 60).
_BUCKETS = [
    ("≤15  (in default cut)", 1, 15),
    ("16-20 (in pool, trimmed→15)", 16, 20),
    ("21-40 (L2a top_k=40 reaches)", 21, 40),
    ("41-60 (L2a top_k=60 reaches)", 41, 60),
    ("61+   (bigger pool needed)", 61, None),
]


def _bucket_label(rank: int | None) -> str:
    if rank is None:
        return "NOT in BM25 top-N (vec-only / unrecoverable)"
    for label, lo, hi in _BUCKETS:
        if rank >= lo and (hi is None or rank <= hi):
            return label
    return "61+   (bigger pool needed)"


def _gold_bm25_rank(hy: HyMemAdapter, question: str, gold_turns: list[str],
                    pool_depth: int) -> int | None:
    """Min BM25 rank (1-based) at which any gold turn appears in the raw message
    FTS ranking, or None if no gold turn is in the top `pool_depth`."""
    from hymem.query.augment import _message_fts_search

    hits = _message_fts_search(hy.hy.conn, question, top_k=pool_depth)
    for i, h in enumerate(hits, start=1):
        # Match one turn at a time so the returned index is the gold's true rank,
        # not just "some gold is somewhere in the pool".
        if _gold_in_pool(gold_turns, [h.text]):
            return i
    return None


def _coverage_record(hits, gold_turns: list[str], cut: int) -> dict:
    """Multi-gold coverage of the message tier's `cut`-slot window (L3 probe).

    The recall diagnostic calls a question a "ranking miss" if ANY gold turn is
    in the pool — fine for single-session questions where one gold turn = answer,
    but MS answers need SEVERAL gold turns across sessions. This measures how many
    of a question's N gold turns actually fit in the `cut`-slot message window, and
    how concentrated that window is across sessions (the L3 monopoly signal).

    Per-gold rank is raw BM25 (no rerank, no chunks — the probe doesn't dream), so
    `n_in_cut` is a LOWER bound on the real reranked coverage → coverage-short is
    (slightly) over-counted, biasing toward "L3 has teeth" (the extra-experiment
    direction). Magnitude ≈ the few rank-(cut..rerank_top_k) turns the reranker lifts.
    """
    # Rank of each individual gold turn (min position it appears at), independently.
    gold_ranks: list[int | None] = []
    gold_hit_sessions: list[str] = []
    for gt in gold_turns:
        rank = None
        sess = None
        for i, h in enumerate(hits, start=1):
            if _gold_in_pool([gt], [h.text]):
                rank, sess = i, h.session_id
                break
        gold_ranks.append(rank)
        if sess is not None:
            gold_hit_sessions.append(sess)

    n_gold = len(gold_turns)
    n_in_cut = sum(1 for r in gold_ranks if r is not None and r <= cut)

    # Session concentration of the `cut`-slot window the model actually sees.
    window = hits[:cut]
    win_sessions = [h.session_id for h in window]
    sess_counts = Counter(win_sessions)
    distinct_sessions = len(sess_counts)
    max_share = (max(sess_counts.values()) / len(win_sessions)) if win_sessions else 0.0

    return {
        "n_gold": n_gold,
        "n_in_cut": n_in_cut,
        "coverage": (n_in_cut / n_gold) if n_gold else None,
        "gold_ranks": gold_ranks,
        "gold_session_span": len(set(gold_hit_sessions)),  # how multi-session the gold is
        "window_distinct_sessions": distinct_sessions,
        "window_max_session_share": max_share,
        "coverage_short": n_in_cut < n_gold,
        # Full ranked session sequence (1-based position i → session of hits[i-1]),
        # so a diversity-pack policy can be simulated as a pure post-process without
        # re-ingesting. Bounded by pool_depth; trivial memory at MS scale.
        "ranked_sessions": [h.session_id for h in hits],
    }


def _pack_select(ranked_sessions: list[str], cut: int, cap: int, pool: int) -> set[int]:
    """Simulate diversity-pack selection: pick `cut` of the top-`pool` BM25 hits with
    at most `cap` turns per session (1-based positions). Pass 1 enforces the cap so
    under-represented sessions get slots; pass 2 backfills by raw rank if the cap left
    the window under-full. The VISIBLE window stays `cut` — only its composition changes,
    so there is no answer-side (45-slot) crowding, unlike widening message_fts_top_k."""
    candidates = range(1, min(pool, len(ranked_sessions)) + 1)
    selected: list[int] = []
    per_session: Counter = Counter()
    for pos in candidates:                       # pass 1 — capped, diversity-first
        if len(selected) >= cut:
            break
        s = ranked_sessions[pos - 1]
        if per_session[s] < cap:
            selected.append(pos)
            per_session[s] += 1
    if len(selected) < cut:                       # pass 2 — backfill by rank, cap lifted
        for pos in candidates:
            if len(selected) >= cut:
                break
            if pos not in selected:
                selected.append(pos)
    return set(selected)


def _pack_coverage_short(rec: dict, cut: int, cap: int, pool: int) -> bool:
    """Is the question still coverage-short after a per-session-cap pack of the window?"""
    sel = _pack_select(rec["ranked_sessions"], cut, cap, pool)
    n_in = sum(1 for r in rec["gold_ranks"] if r is not None and r in sel)
    return n_in < rec["n_gold"]


def _run_coverage(questions: list[dict], args) -> None:
    """L3 front-run: per-question multi-gold coverage of the `--cut`-slot message
    window, optionally JOINED to a run's ranking-miss labels to split the misses
    into coverage-short (L3-fixable) vs fully-covered (synthesis, out of scope)."""
    from hymem.query.augment import _message_fts_search

    cut = args.cut
    print(f"\nGold COVERAGE probe  (L3 — multi-gold window coverage, LLM-free)")
    print(f"  Category: {args.category}   Questions: {len(questions)}   "
          f"cut={cut} (message_fts_top_k)   pool_depth={args.pool_depth}   "
          f"Seed: {args.seed}\n", flush=True)

    records: dict[str, dict] = {}   # question_id -> coverage record
    no_gold = 0
    for qi, q in enumerate(questions):
        gold_turns, _mode = _extract_gold_turns(q)
        if not gold_turns:
            no_gold += 1
            continue
        tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-covprobe-"))
        hy = None
        try:
            hy = HyMemAdapter(tmp_dir / "hymem.sqlite", api_key="probe")
            hy.open()
            sessions = q.get("haystack_sessions", [])
            session_ids = q.get("haystack_session_ids",
                                [str(i) for i in range(len(sessions))])
            hy.ingest_sessions(sessions, session_ids, q.get("haystack_dates", []))
            hits = _message_fts_search(hy.hy.conn, q["question"], top_k=args.pool_depth)
            rec = _coverage_record(hits, gold_turns, cut)
        except Exception as e:
            print(f"  [{qi+1}] {q.get('question_id','?')} ERROR: {e}", flush=True)
            continue
        finally:
            if hy:
                hy.close()
        records[q.get("question_id", f"idx{qi}")] = rec
        if (qi + 1) % 25 == 0:
            print(f"  …{qi+1}/{len(questions)} probed", flush=True)

    n = len(records)
    if not n:
        print("No questions with gold turns to report.")
        return

    def _summarize(recs: list[dict], label: str) -> None:
        if not recs:
            print(f"  {label}: (none)")
            return
        short = [r for r in recs if r["coverage_short"]]
        multi = [r for r in recs if r["n_gold"] >= 2]
        covs = [r["coverage"] for r in recs if r["coverage"] is not None]
        med_cov = sorted(covs)[len(covs) // 2] if covs else 0.0
        spans = sorted(r["window_distinct_sessions"] for r in recs)
        med_span = spans[len(spans) // 2] if spans else 0
        shares = sorted(r["window_max_session_share"] for r in recs)
        med_share = shares[len(shares) // 2] if shares else 0.0
        print(f"  {label}: n={len(recs)}  multi-gold={len(multi)}  "
              f"coverage-short={len(short)} ({100*len(short)/len(recs):.0f}%)  "
              f"median coverage={med_cov:.2f}")
        print(f"    window sessions (median distinct in {cut} slots): {med_span}   "
              f"median max-session-share: {med_share:.2f}")

    print("=" * 64)
    print("MARGINAL coverage (all probed questions in category):")
    _summarize(list(records.values()), "all")
    _summarize([r for r in records.values() if r["n_gold"] >= 2], "multi-gold only")

    if not args.join_run:
        print("\nRead: 'coverage-short' = at least one needed gold turn falls outside\n"
              f"the {cut}-slot message window. High max-session-share + low distinct\n"
              "sessions = one verbose session monopolizing the window (the L3 signal).\n"
              "Pass --join-run <baseline.json> to split THIS category's ranking misses\n"
              "into coverage-short (L3-fixable) vs fully-covered (synthesis, out of scope).")
        return

    # ── JOIN: split the run's ranking misses into coverage-short vs fully-covered ──
    with open(args.join_run) as f:
        run = json.load(f)
    pq = run.get("per_question", [])
    in_cat = [r for r in pq
              if args.category == "all" or r.get("question_type") == args.category]
    rank_miss = [r for r in in_cat
                 if not r.get("correct") and r.get("recall_ceiling") is True]
    retr_miss = [r for r in in_cat
                 if not r.get("correct") and r.get("recall_ceiling") is False]

    matched = [(r, records[r["question_id"]]) for r in rank_miss
               if r.get("question_id") in records]
    unmatched = len(rank_miss) - len(matched)
    short = [rec for _, rec in matched if rec["coverage_short"]]
    full = [rec for _, rec in matched if not rec["coverage_short"]]

    print("\n" + "=" * 64)
    print(f"JOIN — {args.category} misses from {Path(args.join_run).name}")
    print(f"  ranking misses: {len(rank_miss)}   retrieval misses: {len(retr_miss)}"
          + (f"   ({unmatched} ranking-miss qids had no probe record)" if unmatched else ""))
    print(f"  {'─'*58}")
    tot = len(matched)
    if tot:
        sc_short = sorted(r["window_max_session_share"] for r in short)
        sc_full = sorted(r["window_max_session_share"] for r in full)
        med = lambda xs: (sorted(xs)[len(xs)//2] if xs else 0.0)
        print(f"  COVERAGE-SHORT (L3-fixable):   {len(short):>3}  "
              f"({100*len(short)/tot:.0f}%)   median max-session-share {med(sc_short):.2f}")
        print(f"  FULLY-COVERED (synthesis):     {len(full):>3}  "
              f"({100*len(full)/tot:.0f}%)   median max-session-share {med(sc_full):.2f}")
        print(f"  {'─'*58}")
        verdict = ("L3 HAS TEETH — most ranking misses are coverage-short; the gold"
                   if len(short) > len(full) else
                   "L3 WON'T HELP — most ranking misses are fully covered; the gold")
        tail = ("doesn't fit the window. Widen message_fts_top_k or diversity-pack."
                if len(short) > len(full) else
                "is in context and the model still fails → synthesis, answer-side.")
        print(f"  VERDICT: {verdict} {tail}")

        # ── Cut-sweep projection: front-run the message_fts_top_k benchmark sweep ──
        # gold_ranks is cut-independent, so coverage-short at a WIDER cut is computable
        # from THIS probe run — no re-ingest, no LLM. This tells you the cut where
        # coverage-short bottoms out (the recoverable ceiling) BEFORE spending an
        # answer+judge sweep, and exposes the "gold is at rank 40+ / not in pool"
        # tail that widening message_fts_top_k can NEVER reach.
        if args.cut_sweep:
            sweep = sorted({int(c) for c in args.cut_sweep.split(",")})
            short_recs = [rec for rec in (r for _, r in matched) if rec["coverage_short"]]
            base_short = len(short_recs)
            print(f"  {'─'*58}")
            print(f"  CUT-SWEEP projection on the {base_short} coverage-short misses")
            print(f"  (how many stay short as message_fts_top_k widens — raw-BM25 floor):")
            for c in sweep:
                still_short = sum(
                    1 for rec in short_recs
                    if sum(1 for r in rec["gold_ranks"] if r is not None and r <= c)
                    < rec["n_gold"])
                recovered = base_short - still_short
                print(f"    cut={c:>3}:  still coverage-short {still_short:>3}   "
                      f"recovered {recovered:>3} ({100*recovered/base_short:.0f}% of short)")
            # The residual that NO cut reaches: at least one gold turn never appears
            # in the BM25 pool at all (rank is None) → widening message FTS can't help.
            unreachable = sum(
                1 for rec in short_recs
                if any(r is None for r in rec["gold_ranks"]))
            print(f"  → floor: {unreachable} short miss(es) have ≥1 gold turn NOT in the "
                  f"BM25 pool at any depth — widening message_fts_top_k can't reach those.")

        # ── Pack-sim: diversity-pack the FIXED-size window over a deep candidate pool ──
        # Head-to-head vs the widen sweep above, at the SAME visible cut — so any gain
        # here is free of answer-side crowding. Recovers the deep-in-pool turns whose
        # session is under-represented; cannot touch the floor (gold not in pool at all).
        if args.pack_sim:
            short_recs = [rec for rec in (r for _, r in matched) if rec["coverage_short"]]
            base_short = len(short_recs)
            # Independent of --cut-sweep: a short miss with a None gold rank is floor.
            floor_n = sum(1 for rec in short_recs
                          if any(r is None for r in rec["gold_ranks"]))
            print(f"  {'─'*58}")
            print(f"  PACK-SIM on the {base_short} coverage-short misses  "
                  f"(visible cut={cut} held fixed, pool={args.pack_pool}):")
            for cap in sorted({int(c) for c in args.pack_sim.split(",")}):
                still = sum(1 for rec in short_recs
                            if _pack_coverage_short(rec, cut, cap, args.pack_pool))
                recovered = base_short - still
                print(f"    per-session cap={cap:>2}:  still short {still:>3}   "
                      f"recovered {recovered:>3} ({100*recovered/base_short:.0f}% of short)")
            print(f"  → pack reorders a {cut}-slot window (no answer-side crowding); its "
                  f"ceiling is the {base_short - floor_n} non-floor misses.")

        # ── Floor audit: reconcile the probe's message-only floor with the run's ──
        # FUSED-pool per-turn membership (needs an instrumented run carrying
        # gold_turn_tiers). Splits the message-FTS floor into PHANTOM (chunks/embeddings
        # rescued the turn → gone) vs REAL recall gap (still "none" with both tiers run →
        # the next retrieval lever, narrow enough to target directly).
        has_tiers = any("gold_turn_tiers" in r for r, _ in matched)
        if has_tiers:
            probe_floor = [(r, rec) for r, rec in matched
                           if any(gr is None for gr in rec["gold_ranks"])]
            phantom, real = [], []
            for r, _rec in probe_floor:
                tiers = r.get("gold_turn_tiers") or []
                (real if any(t == "none" for t in tiers) else phantom).append(r)
            print(f"  {'─'*58}")
            print(f"  FLOOR AUDIT — {len(probe_floor)} probe-floor misses vs the run's "
                  f"fused pool:")
            print(f"    PHANTOM (chunks/embeddings rescued, all turns pooled): "
                  f"{len(phantom):>3}")
            print(f"    REAL recall gap (≥1 gold turn 'none' in fused pool):  "
                  f"{len(real):>3}")
            if real:
                ids = ", ".join(r.get("question_id", "?") for r in real[:8])
                print(f"    → real-gap qids (target for chunk/embedding recall): {ids}"
                      + (" …" if len(real) > 8 else ""))
        elif args.floor_audit:
            print(f"  {'─'*58}")
            print("  FLOOR AUDIT requested but the run JSON has no 'gold_turn_tiers' —\n"
                  "  re-run the baseline with the instrumented adapter first.")
    print("\n  Caveat: probe coverage is raw-BM25 message-only (no rerank, no chunks),\n"
          "  a LOWER bound on real coverage → coverage-short is mildly over-counted.\n"
          "  If the split is borderline, confirm with a reranked-coverage run.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="multi-session",
                    help="question_type to probe (e.g. multi-session, "
                         "temporal-reasoning), or 'all' for every category.")
    ap.add_argument("--scales", default="S")
    ap.add_argument("--sample", type=int, default=0,
                    help="questions to load; 0 = full set (no sampling variance).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-depth", type=int, default=200,
                    help="how deep to scan the BM25 ranking before calling a gold "
                         "turn unreachable (the 'NOT in top-N' bucket).")
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).resolve().parent.parent.parent
                                / "hymem_beam" / "data"))
    ap.add_argument("--coverage", action="store_true",
                    help="L3 mode: measure MULTI-GOLD coverage of the message window "
                         "(how many of a question's N gold turns fit the cut) + session "
                         "concentration, instead of the single-best-turn rank histogram.")
    ap.add_argument("--cut", type=int, default=15,
                    help="message-tier slot count (message_fts_top_k, default 15) — the "
                         "window multi-gold coverage is measured against in --coverage mode.")
    ap.add_argument("--join-run", default=None,
                    help="(--coverage) a benchmark result JSON; splits THIS category's "
                         "ranking misses (correct=False & recall_ceiling=True) into "
                         "coverage-short (L3-fixable) vs fully-covered (synthesis).")
    ap.add_argument("--cut-sweep", default=None,
                    help="(--coverage --join-run) comma-list of candidate "
                         "message_fts_top_k cuts (e.g. '20,25,30'); projects how many "
                         "coverage-short misses each cut recovers from the SAME probe "
                         "run (gold_ranks is cut-independent) — front-runs the benchmark "
                         "sweep and exposes the rank-40+/not-in-pool tail widening can't reach.")
    ap.add_argument("--pack-sim", default=None,
                    help="(--coverage --join-run) comma-list of per-session caps (e.g. "
                         "'2,3,4'); simulates diversity-pack selection of the FIXED cut-slot "
                         "window from the top --pack-pool BM25 hits — head-to-head vs "
                         "--cut-sweep, but with no answer-side crowding (visible window size "
                         "unchanged). Front-runs the diversity-pack feature.")
    ap.add_argument("--pack-pool", type=int, default=60,
                    help="(--pack-sim) candidate-pool depth to diversity-pack the window "
                         "from (default 60). Deeper pool reaches deeper gold turns.")
    ap.add_argument("--floor-audit", action="store_true",
                    help="(--coverage --join-run) split the probe's message-FTS floor into "
                         "PHANTOM (the run's fused pool rescued it via chunks/embeddings) vs "
                         "REAL recall gap. Auto-fires when the run JSON carries gold_turn_tiers; "
                         "this flag only warns if the run wasn't instrumented.")
    args = ap.parse_args()

    scale = args.scales.upper()
    data_file = (Path(args.data_dir) / "longmemeval_s_cleaned.json" if scale == "S"
                 else Path(args.data_dir) / f"longmemeval_{scale.lower()}_cleaned.json")
    if not data_file.exists():
        print(f"ERROR: dataset not found at {data_file}")
        sys.exit(1)

    questions = load_longmemeval_data(str(data_file),
                                      max_questions=(args.sample or None),
                                      seed=args.seed)
    if args.category != "all":
        questions = [q for q in questions if q.get("question_type") == args.category]
    if not questions:
        print(f"ERROR: no questions for category '{args.category}'")
        sys.exit(1)

    if args.coverage:
        _run_coverage(questions, args)
        return

    print(f"\nGold BM25-rank probe  (LLM-free, embedding-free)")
    print(f"  Category: {args.category}   Questions: {len(questions)}   "
          f"Pool depth: {args.pool_depth}   Seed: {args.seed}\n", flush=True)

    # Per-category histograms: {category: {bucket_label: count}}.
    hist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    no_gold: dict[str, int] = defaultdict(int)
    ranks_seen: dict[str, list[int]] = defaultdict(list)

    for qi, q in enumerate(questions):
        cat = q.get("question_type", "?")
        gold_turns, mode = _extract_gold_turns(q)
        if not gold_turns:
            no_gold[cat] += 1
            continue

        tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-rankprobe-"))
        hy = None
        try:
            # api_key is never used — no LLM call is made; ingest is pure persistence.
            hy = HyMemAdapter(tmp_dir / "hymem.sqlite", api_key="probe")
            hy.open()
            sessions = q.get("haystack_sessions", [])
            session_ids = q.get("haystack_session_ids",
                                [str(i) for i in range(len(sessions))])
            hy.ingest_sessions(sessions, session_ids, q.get("haystack_dates", []))
            rank = _gold_bm25_rank(hy, q["question"], gold_turns, args.pool_depth)
        except Exception as e:  # one bad question must not abort the sweep
            print(f"  [{qi+1}] {q.get('question_id','?')} ERROR: {e}", flush=True)
            continue
        finally:
            if hy:
                hy.close()

        hist[cat][_bucket_label(rank)] += 1
        if rank is not None:
            ranks_seen[cat].append(rank)
        if (qi + 1) % 25 == 0:
            print(f"  …{qi+1}/{len(questions)} probed", flush=True)

    # ── Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    bucket_order = [lbl for lbl, _, _ in _BUCKETS] + [
        "NOT in BM25 top-N (vec-only / unrecoverable)"]
    for cat in sorted(hist):
        rows = hist[cat]
        n = sum(rows.values())
        ranks = sorted(ranks_seen[cat])
        med = ranks[len(ranks) // 2] if ranks else None
        print(f"\n{cat}   (n={n} with gold"
              + (f", {no_gold[cat]} excluded: no gold label" if no_gold[cat] else "")
              + ")")
        print(f"  median gold BM25 rank: {med if med is not None else '∅'}")
        # Cumulative reach: how many gold turns a given top_k budget would cover.
        reachable = {20: 0, 40: 0, 60: 0}
        for r in ranks:
            for k in reachable:
                if r <= k:
                    reachable[k] += 1
        for label in bucket_order:
            c = rows.get(label, 0)
            if c:
                bar = "█" * round(40 * c / n) if n else ""
                print(f"    {label:<46} {c:>4}  {100*c/n:4.0f}%  {bar}")
        if ranks:
            print(f"  → reachable by top_k=20: {reachable[20]}/{n} ({100*reachable[20]/n:.0f}%)"
                  f" | =40: {reachable[40]}/{n} ({100*reachable[40]/n:.0f}%)"
                  f" | =60: {reachable[60]}/{n} ({100*reachable[60]/n:.0f}%)")
            gain_40 = reachable[40] - reachable[20]
            print(f"  → L2a verdict: top_k 20→40 newly reaches {gain_40} gold turn(s) "
                  f"({100*gain_40/n:.0f}% of this category)")

    print("\n" + "=" * 64)
    print("Read: the '21-40' band is what --rerank-top-k 40 newly reaches; '41-60'\n"
          "needs 60. A large 'NOT in BM25 top-N' bucket means widening the MESSAGE\n"
          "pool won't reach those turns — flagging them as the vector path's job\n"
          "(confirm with a real embeddings-on run; chunk-tier recovery may still\n"
          "get some). Ranks here are message-FTS only: a LOWER BOUND on the\n"
          "combined-pool rank, so this gate is pessimistic by design.")


if __name__ == "__main__":
    main()
