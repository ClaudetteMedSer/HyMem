#!/usr/bin/env python3
"""Read-only KU retrieval probe — does the *new* value surface, and is it dated?

Why this exists: the proposed knowledge-update (KU) fix is to stamp each
`MessageHit.created_at` into the answer context + add a "most recent value wins"
clause, so the model can resolve an old-vs-new conflict by recency. That fix only
pays off if TWO preconditions hold per KU item, and neither is checkable
statically:

  (1) RECALL  — the *updated* (new) value actually surfaces in `message_hits`
                (BM25 alone is recency-blind; if the new turn never enters the
                candidate pool, no amount of dating helps).
  (2) DATING  — the surfaced statements carry distinguishable `created_at` dates,
                and the new value (== the gold answer) sits on the *latest* one.

This script answers both by reusing the benchmark adapter's OWN load/ingest path
and calling `augment()` directly to read the raw `MessageHit` objects (the only
tier carrying `created_at` — the merged `search()` dicts drop it). It runs no
full eval, makes no judge/answer LLM calls, and writes nothing to the repo.

Default is no-dream + raw-BM25 message tier (offline, free): message_hits read
`messages_fts`, populated at ingest, so dreaming isn't needed to see the recall
question. Pass --dream to additionally populate + print conflicting graph edges
(the secondary undated surface).

Usage (on the Hermes box, from benchmarks/):
    python ku_probe.py                 # first 8 KU items, S-scale
    python ku_probe.py --n 12 --wide 40
    python ku_probe.py --dream         # also show graph-fact conflicts
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from longmemeval_adapter import (  # noqa: E402
    HyMemAdapter,
    _detect_ability_safe,
    _extract_gold_turns,
    _gold_in_pool,
    load_longmemeval_data,
)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _answer_in(text: str, answer: str) -> bool:
    """Loose presence test: full answer substring, or any content token >3 chars.

    KU answers are short (a number, a brand, a size). Substring on the whole
    answer catches exact restatements; the per-token fallback catches a value
    embedded in a longer turn ("...so I upgraded it to 500 Mbps last week")."""
    t = _norm(text)
    a = _norm(answer)
    if a and a in t:
        return True
    return any(tok in t for tok in a.split() if len(tok) > 3)


def _date10(created_at: str) -> str:
    """Leading YYYY-MM-DD of an ISO created_at, or '' if absent/short."""
    return created_at[:10] if created_at and len(created_at) >= 10 else ""


def probe_question(adapter: HyMemAdapter, q: dict, top_k: int) -> dict:
    question = q["question"]
    answer = str(q.get("answer", ""))
    sessions = q.get("haystack_sessions", [])
    session_ids = q.get("haystack_session_ids", [str(i) for i in range(len(sessions))])
    session_dates = q.get("haystack_dates", [])
    while len(session_ids) < len(sessions):
        session_ids.append(f"extra_{len(session_ids)}")

    adapter.ingest_sessions(sessions, session_ids, session_dates)

    # KU's production path: detect_ability returns None -> message-first ordering.
    ability = _detect_ability_safe(question)
    ctx = adapter.hy.augment(question, ability=ability)
    hits = list(getattr(ctx, "message_hits", None) or [])

    # Sort chronologically so the eyeball reads old -> new.
    hits.sort(key=lambda h: (getattr(h, "created_at", "") or "", getattr(h, "message_id", 0)))

    hit_texts = [getattr(h, "text", "") for h in hits]
    gold_turns, gold_mode = _extract_gold_turns(q)
    gold_in_msg = _gold_in_pool(gold_turns, hit_texts) if gold_turns else None

    dates = [_date10(getattr(h, "created_at", "")) for h in hits]
    distinct_dates = sorted({d for d in dates if d})
    answer_dates = sorted({
        _date10(getattr(h, "created_at", ""))
        for h in hits if _answer_in(getattr(h, "text", ""), answer)
    } - {""})
    answer_present = bool(answer_dates) or any(_answer_in(t, answer) for t in hit_texts)
    latest_date = distinct_dates[-1] if distinct_dates else ""
    answer_on_latest = bool(answer_dates) and answer_dates[-1] == latest_date

    return {
        "q": q,
        "ability": ability,
        "hits": hits,
        "dates": dates,
        "distinct_dates": distinct_dates,
        "gold_turns": gold_turns,
        "gold_mode": gold_mode,
        "gold_in_msg": gold_in_msg,
        "answer_present": answer_present,
        "answer_dates": answer_dates,
        "answer_on_latest": answer_on_latest,
        "graph_facts": list(getattr(ctx, "graph_facts", None) or []),
    }


def print_report(r: dict, top_k: int, show_graph: bool) -> None:
    q = r["q"]
    print("=" * 88)
    print(f"  id        : {q.get('question_id')}")
    print(f"  question  : {q.get('question')}")
    print(f"  answer*   : {q.get('answer')}   (* = the NEW value the judge requires)")
    print(f"  q_date    : {q.get('question_date', '') or '(none)'}   ability={r['ability']!r}")
    print(f"  gold turns: {len(r['gold_turns'])} (mode={r['gold_mode']})")
    print(f"  message_hits: {len(r['hits'])}   distinct dates: {r['distinct_dates'] or '(none)'}")
    g = r["gold_in_msg"]
    print(f"  gold turn in message_hits : "
          f"{'YES' if g else ('NO' if g is False else 'n/a (no gold marks)')}")
    print(f"  new value (answer) present: {'YES' if r['answer_present'] else 'NO'}"
          f"  on dates {r['answer_dates'] or '(undated/absent)'}"
          f"  -> on latest date: {'YES' if r['answer_on_latest'] else 'no'}")
    print(f"  --- message_hits (oldest -> newest), capped display at {top_k} ---")
    for h, d in list(zip(r["hits"], r["dates"]))[:top_k]:
        role = getattr(h, "role", "?")
        score = getattr(h, "score", 0.0)
        text = _norm(getattr(h, "text", ""))[:140]
        mark = " «ANSWER»" if _answer_in(getattr(h, "text", ""), str(q.get("answer", ""))) else ""
        print(f"    [{d or '----------'}] ({role:>9} {score:5.1f}) {text}{mark}")
    if show_graph and r["graph_facts"]:
        print(f"  --- graph_facts (UNDATED tier — conflict check) ---")
        for f in r["graph_facts"][:12]:
            print(f"    [FACT conf={getattr(f, 'confidence', 0):.2f}] "
                  f"{getattr(f, 'subject', '')} {getattr(f, 'predicate', '')} {getattr(f, 'object', '')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only KU retrieval/dating probe")
    repo_root = Path(__file__).resolve().parent.parent
    ap.add_argument("--data-dir", default=str(repo_root.parent / "hymem_beam" / "data"))
    ap.add_argument("--scale", default="S")
    ap.add_argument("--n", type=int, default=8, help="number of KU questions to probe")
    ap.add_argument("--top-k", type=int, default=15, help="message_hits shown per question")
    ap.add_argument("--wide", type=int, default=None,
                    help="override message_fts_top_k to widen the recall pool (diagnostic)")
    ap.add_argument("--dream", action="store_true",
                    help="run dream to populate + print conflicting graph edges")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()

    scale = args.scale.upper()
    fname = "longmemeval_s_cleaned.json" if scale == "S" else f"longmemeval_{scale.lower()}_cleaned.json"
    data_file = Path(args.data_dir) / fname
    if not data_file.exists():
        print(f"ERROR: dataset not found at {data_file}")
        sys.exit(1)

    print(f"Loading {data_file} ...", flush=True)
    all_q = load_longmemeval_data(str(data_file), max_questions=None, seed=args.seed)
    ku = [q for q in all_q if q.get("question_type", "").replace("_abs", "") == "knowledge-update"
          and not q.get("question_type", "").endswith("_abs")]
    if not ku:
        print("No knowledge-update questions found in this dataset.")
        sys.exit(1)
    ku = ku[: args.n]
    print(f"Probing {len(ku)} knowledge-update questions "
          f"(dream={'ON' if args.dream else 'OFF'}, "
          f"message tier=raw BM25, wide={args.wide or 'default'})\n", flush=True)

    reports = []
    for q in ku:
        tmp = Path(tempfile.mkdtemp(prefix="ku_probe_"))
        adapter = HyMemAdapter(
            db_path=tmp / "hymem.db",
            api_key=args.api_key,
            embeddings=False,
            rerank_message_hits=False,  # raw BM25 — offline, and recall (membership) is order-independent
        )
        try:
            adapter.open()
            if args.wide:
                # Widen the candidate pool to test whether the new value is
                # retrievable at all (vs lost below the default cap).
                try:
                    adapter.hy.cfg.message_fts_top_k = args.wide
                except Exception:
                    pass
            r = probe_question(adapter, q, args.top_k)
            if args.dream:
                adapter.dream_and_wait()
                ctx = adapter.hy.augment(q["question"], ability=r["ability"])
                r["graph_facts"] = list(getattr(ctx, "graph_facts", None) or [])
            reports.append(r)
            print_report(r, args.top_k, args.dream)
        except Exception as e:
            print(f"  [ERROR on {q.get('question_id')}]: {type(e).__name__}: {e}")
        finally:
            try:
                adapter.close()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    # ── Summary: the two preconditions, tallied ──────────────────────────
    n = len(reports)
    if not n:
        return
    gold_known = [r for r in reports if r["gold_in_msg"] is not None]
    gold_hit = sum(1 for r in gold_known if r["gold_in_msg"])
    ans_present = sum(1 for r in reports if r["answer_present"])
    ans_latest = sum(1 for r in reports if r["answer_on_latest"])
    multi_date = sum(1 for r in reports if len(r["distinct_dates"]) >= 2)
    print("=" * 88)
    print("SUMMARY")
    print(f"  questions probed                  : {n}")
    if gold_known:
        print(f"  gold turn in message_hits         : {gold_hit}/{len(gold_known)}  "
              f"(recall precondition #1)")
    print(f"  new value (answer) present in hits: {ans_present}/{n}  (recall precondition #1)")
    print(f"  >=2 distinct dates among hits     : {multi_date}/{n}  "
          f"(is a recency conflict even visible?)")
    print(f"  new value on the LATEST date      : {ans_latest}/{n}  "
          f"(dating precondition #2 — clean win for the fix)")
    print()
    print("  READ: if 'new value present' is HIGH and 'on latest date' is HIGH -> the")
    print("  dating+clause fix is well-targeted; a full strict run is worth buying.")
    print("  If 'new value present' is LOW -> the lever is KU RECALL, not answer-shaping;")
    print("  redirect before spending a run. If present-but-not-latest dominates -> the")
    print("  conflict isn't recency-separable and the clause won't disambiguate it.")


if __name__ == "__main__":
    main()
