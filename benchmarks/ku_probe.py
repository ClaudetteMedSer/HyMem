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
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from longmemeval_adapter import (  # noqa: E402
    MAX_CONTEXT_CHARS,
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


def _gold_value_tokens(answer: str) -> list[str]:
    """Discriminating tokens of the gold value: any token carrying a digit (a
    number/size/version) or an alpha word >3 chars (a brand/name). Short
    stopwords are dropped so the edge match isn't swamped."""
    out = []
    for t in re.split(r"[^a-z0-9]+", (answer or "").lower()):
        if t and (any(c.isdigit() for c in t) or len(t) > 3):
            out.append(t)
    return out


def _gold_edges(conn, answer: str) -> list[tuple]:
    """COVERAGE PROBE: does the gold value exist as a knowledge_graph EDGE at
    all — minted by extraction, regardless of whether retrieval surfaced it?

    This is the extraction-vs-ranking fork for the [MEM]-consumption lever. The
    `graph_facts` tier shows only RETRIEVED edges (top-k, relevance-filtered); a
    direct table scan tells us instead whether the value ever entered the graph.
    A numeric gold token matches a whole object/subject token exactly (so "35"
    doesn't hit "350"); an alpha token >3 chars matches as a substring (a value
    embedded in a longer canonical). Returns (row, matched_tokens) per edge."""
    toks = _gold_value_tokens(answer)
    if not toks:
        return []
    rows = conn.execute(
        "SELECT subject_canonical AS s, predicate AS p, object_canonical AS o, "
        "       status AS st, valid_at AS v, invalid_at AS iv "
        "FROM knowledge_graph"
    ).fetchall()
    out = []
    for row in rows:
        hay = f"{row['s']} {row['o']}".lower()
        hay_toks = set(re.split(r"[^a-z0-9]+", hay))
        matched = [t for t in toks if t in hay_toks or (len(t) > 3 and t in hay)]
        if matched:
            out.append((row, matched))
    return out


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

    # SPOILERS: hits dated strictly AFTER the newest answer-bearing turn — these are
    # the turns that "beat" the new value on recency and would mislead a naive
    # latest-date-wins clause. Eyeball each: does it RESTATE a (stale) value for the
    # same attribute [(a) genuine conflict — unfixable from raw turns], or does it
    # merely mention the topic with no value [(b) tangential — a value-aware clause
    # can skip it]? The (a):(b) split is the real ceiling of the dating fix.
    ans_latest_date = answer_dates[-1] if answer_dates else ""
    spoilers = [
        h for h in hits
        if ans_latest_date and _date10(getattr(h, "created_at", "")) > ans_latest_date
    ]

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
        "ans_latest_date": ans_latest_date,
        "spoilers": spoilers,
        "graph_facts": list(getattr(ctx, "graph_facts", None) or []),
    }


# ── Ranking diagnosis ────────────────────────────────────────────────────────
# "Ranking miss" in the recall-ceiling split = gold was in the pool but the answer
# was still wrong. That bundles two fates with OPPOSITE fixes:
#   (B) TRUNCATION — gold turn never survived into the final answer context
#       (cut by the 45-candidate top_k or, far more often, by the 8000-char budget).
#       Recoverable by ranking / budget / rerank-off.
#   (C) SYNTHESIS  — gold turn WAS in the context the model saw, and it still missed.
#       Ranking can't help; the value-aware clause already applies here.
# We reconstruct the exact final context OFFLINE (no answer LLM) by mirroring
# answer_question's char-budget loop, so the B:C split is free to compute. Run with
# --rerank to match the headline run's pipeline (LLM reranker ON); run without it
# (raw BM25) to test the "reranker demotes gold it already sees" hypothesis — if a
# B-bucket question flips to C under raw BM25, rerank_message_hits=False recovers it.


def _reconstruct_context_memories(memories: list[dict], context_limit: int) -> list[dict]:
    """The memories that actually enter the answer context — mirrors the
    char-budget loop in answer_question (KU has no MR/TR preamble, so the full
    budget goes to memories). Tag length is counted exactly as the adapter does,
    including the new [MEM <date>] stamp."""
    out: list[dict] = []
    total = 0
    for m in memories:
        content = m["content"]
        if total + len(content) > context_limit:
            break
        if m["type"] == "graph_fact":
            tag = "[FACT]"
        else:
            d = (m.get("created_at") or "")[:10]
            tag = f"[MEM {d}]" if (m["type"] == "message_hit" and d) else "[MEM]"
        out.append(m)
        total += len(content) + len(tag) + 2
    return out


def _context_limit_for(ability, base: int) -> int:
    """answer_question doubles the char budget for MR/TR (they span sessions)."""
    return base * 2 if ability in ("MR", "TR") else base


def _gold_rank(memories: list[dict], gold_turns: list[str]):
    for i, m in enumerate(memories):
        if _gold_in_pool(gold_turns, [m["content"]]):
            return i
    return None


def _classify_gold(memories: list[dict], pool: dict, gold_turns: list[str],
                   context_limit: int) -> dict:
    """Bucket the gold turn's fate: A=retrieval miss, B=truncation, C=in-context."""
    in_pool = (
        _gold_in_pool(gold_turns, pool["message"]) or _gold_in_pool(gold_turns, pool["fts"])
    ) if gold_turns else None
    rank = _gold_rank(memories, gold_turns) if gold_turns else None
    ctx = _reconstruct_context_memories(memories, context_limit)
    in_ctx = any(_gold_in_pool(gold_turns, [m["content"]]) for m in ctx) if gold_turns else None
    if not gold_turns:
        bucket, why = "n/a", "no gold marks in dataset"
    elif not in_pool:
        bucket, why = "A_retrieval", "gold not retrieved into any pool"
    elif in_ctx:
        bucket, why = "C_synthesis", f"gold@rank{rank} IS in context ({len(ctx)} mems fit) — model saw it"
    elif rank is not None:
        bucket, why = "B_truncation", f"gold@rank{rank} but only {len(ctx)} mems fit the char budget — char-budget cut"
    else:
        bucket, why = "B_truncation", "gold in pool but below the 45-candidate top_k cut"
    return {"in_pool": in_pool, "gold_rank": rank, "gold_in_ctx": in_ctx,
            "ctx_size": len(ctx), "bucket": bucket, "why": why}


def _ingest_and_search(adapter: HyMemAdapter, q: dict, run_top_k: int):
    """Reproduce the run's ingest + retrieval. ability=detect (auto-ability path);
    search caps at run_top_k*3 candidates (the adapter passes top_k*3 to search)."""
    sessions = q.get("haystack_sessions", [])
    session_ids = q.get("haystack_session_ids", [str(i) for i in range(len(sessions))])
    session_dates = q.get("haystack_dates", [])
    while len(session_ids) < len(sessions):
        session_ids.append(f"extra_{len(session_ids)}")
    adapter.ingest_sessions(sessions, session_ids, session_dates)
    ability = _detect_ability_safe(q["question"])
    memories, _tm, _gc, _tev, pool = adapter.search(
        q["question"], ability=ability, top_k=run_top_k * 3)
    return ability, memories, pool


def diagnose_ranking(adapter: HyMemAdapter, q: dict, run_top_k: int,
                     base_context_limit: int) -> dict:
    ability, memories, pool = _ingest_and_search(adapter, q, run_top_k)
    gold_turns, gold_mode = _extract_gold_turns(q)
    cl = _context_limit_for(ability, base_context_limit)
    cls = _classify_gold(memories, pool, gold_turns, cl)
    return {"q": q, "ability": ability, "n_memories": len(memories),
            "ctx_size": cls["ctx_size"], "gold_rank": cls["gold_rank"],
            "in_pool": cls["in_pool"], "gold_in_ctx": cls["gold_in_ctx"],
            "bucket": cls["bucket"], "why": cls["why"], "gold_mode": gold_mode}


def print_ranking_report(r: dict) -> None:
    q = r["q"]
    print("=" * 88)
    print(f"  id        : {q.get('question_id')}")
    print(f"  question  : {q.get('question')}")
    print(f"  answer*   : {q.get('answer')}")
    print(f"  candidates: {r['n_memories']}   in-context (char-budget): {r['ctx_size']}")
    rank = r["gold_rank"]
    print(f"  gold turn : rank={'—' if rank is None else rank}  "
          f"in_pool={r['in_pool']}  in_context={r['gold_in_ctx']}  (mode={r['gold_mode']})")
    print(f"  BUCKET    : {r['bucket']}  — {r['why']}")


# ── MS diagnosis ─────────────────────────────────────────────────────────────
# Multi-session is heterogeneous under --auto-ability: each question detects as MR
# (counting), TR, or None, and those take different answer paths. On top of the
# A/B/C localization, MR-detected questions hit a LATENT BUG: answer_question's
# "filter to user-only turns" runs AFTER the context is already built (it reassigns
# `memories` at L713, but `context` was frozen at L696), so the filter is dead and
# the model sees the full UNFILTERED context — assistant turns included, despite
# the MR preamble claiming "assistant echoes excluded". The what-if below
# reconstructs context BOTH ways (current=unfiltered vs intended=user-only) to test
# whether applying the filter before the loop would recover misses or DROP gold.


def _user_only(memories: list[dict]) -> list[dict]:
    """The adapter's intended MR filter (L713) — user message turns only."""
    return [m for m in memories
            if m["type"] == "message_hit" and "[user]" in m.get("content", "")]


def diagnose_ms(adapter: HyMemAdapter, q: dict, run_top_k: int,
                base_context_limit: int) -> dict:
    ability, memories, pool = _ingest_and_search(adapter, q, run_top_k)
    gold_turns, gold_mode = _extract_gold_turns(q)
    cl = _context_limit_for(ability, base_context_limit)
    cls = _classify_gold(memories, pool, gold_turns, cl)

    r = {"q": q, "ability": ability, "gold_mode": gold_mode, "mr_whatif": None}
    r.update(cls)

    if ability == "MR" and gold_turns:
        cur_ctx = _reconstruct_context_memories(memories, cl)          # REAL (buggy) behavior
        cur_in = any(_gold_in_pool(gold_turns, [m["content"]]) for m in cur_ctx)
        filt = _user_only(memories)                                   # the comment's intent
        int_ctx = _reconstruct_context_memories(filt, cl)
        int_in = any(_gold_in_pool(gold_turns, [m["content"]]) for m in int_ctx)
        gi = _gold_rank(memories, gold_turns)
        gold_role = "—"
        if gi is not None:
            c = memories[gi]["content"]
            gold_role = ("user" if c.startswith("[user]")
                         else "assistant" if c.startswith("[assistant]")
                         else memories[gi]["type"])
        mh = [m for m in cur_ctx if m["type"] == "message_hit"]
        asst = [m for m in mh if "[assistant]" in m["content"] and not m["content"].startswith("[user]")]
        noise = (len(asst) / len(mh)) if mh else 0.0
        if not cur_in:
            verdict = "n/a  (gold not in current context — A/B issue, not the filter)"
        elif int_in:
            verdict = "SAFE  (gold survives user-only; removing assistant noise is pure upside)"
        else:
            verdict = "RISK  (user-only filter DROPS gold — gold is assistant or pushed out)"
        r["mr_whatif"] = {"gold_role": gold_role, "cur_in": cur_in, "int_in": int_in,
                          "noise": noise, "n_user_only": len(filt), "verdict": verdict}
    return r


def print_ms_report(r: dict) -> None:
    q = r["q"]
    print("=" * 88)
    print(f"  id      : {q.get('question_id')}")
    print(f"  question: {q.get('question')}")
    print(f"  answer  : {q.get('answer')}")
    print(f"  ability : {r['ability']!r}   bucket: {r['bucket']}  — {r['why']}")
    w = r["mr_whatif"]
    if w:
        print(f"  MR-filter what-if: gold_role={w['gold_role']}  "
              f"current_ctx={w['cur_in']}  user_only_ctx={w['int_in']}  "
              f"assistant_noise={w['noise'] * 100:.0f}%  (user-only kept {w['n_user_only']} mems)")
        print(f"                  -> {w['verdict']}")


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
    # The decisive eyeball: turns that out-date the new value. Classify each
    # (a) STALE-VALUE re-assertion (unfixable) vs (b) TANGENTIAL mention (fixable).
    if r["answer_present"] and not r["answer_on_latest"] and r["spoilers"]:
        print(f"  --- SPOILERS: turns dated AFTER the new value ({r['ans_latest_date']}) "
              f"— classify (a) stale-value vs (b) tangential ---")
        for h in r["spoilers"]:
            d = _date10(getattr(h, "created_at", ""))
            role = getattr(h, "role", "?")
            text = _norm(getattr(h, "text", ""))[:160]
            print(f"    [{d}] ({role:>9}) {text}")
    if show_graph and r["graph_facts"]:
        print(f"  --- graph_facts (UNDATED tier — conflict check) ---")
        for f in r["graph_facts"][:12]:
            print(f"    [FACT conf={getattr(f, 'confidence', 0):.2f}] "
                  f"{getattr(f, 'subject', '')} {getattr(f, 'predicate', '')} {getattr(f, 'object', '')}")
    if show_graph:
        ge = r.get("gold_edges") or []
        print(f"  GOLD VALUE as graph edge   : {'YES' if ge else 'NO'}  "
              f"({len(ge)} matching edge(s); table scan, not retrieval) "
              f"[coverage: was the value extracted at all?]")
        for row, matched in ge[:8]:
            print(f"    [{row['st']:>9} v={(row['v'] or '')[:10] or '----------'}] "
                  f"{row['s']} {row['p']} {row['o']}  «match {','.join(matched)}»")


def run_ranking(ku: list[dict], args) -> None:
    rerank_flag = True if args.rerank else False  # raw BM25 by default
    print(f"RANKING DIAGNOSIS — {len(ku)} KU questions  "
          f"(message reranker={'ON (matches headline run)' if args.rerank else 'OFF (raw BM25)'}, "
          f"run_top_k={args.run_top_k}, char_budget={MAX_CONTEXT_CHARS})\n", flush=True)
    reports = []
    for q in ku:
        tmp = Path(tempfile.mkdtemp(prefix="ku_rank_"))
        adapter = HyMemAdapter(
            db_path=tmp / "hymem.db",
            api_key=args.api_key,
            embeddings=False,
            rerank_message_hits=rerank_flag,
        )
        try:
            adapter.open()
            r = diagnose_ranking(adapter, q, args.run_top_k, MAX_CONTEXT_CHARS)
            reports.append(r)
            print_ranking_report(r)
        except Exception as e:
            print(f"  [ERROR on {q.get('question_id')}]: {type(e).__name__}: {e}")
        finally:
            try:
                adapter.close()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    n = len(reports)
    if not n:
        return
    scored = [r for r in reports if r["bucket"] != "n/a"]
    a = sum(1 for r in scored if r["bucket"] == "A_retrieval")
    b = sum(1 for r in scored if r["bucket"] == "B_truncation")
    c = sum(1 for r in scored if r["bucket"] == "C_synthesis")
    print("=" * 88)
    print("RANKING SUMMARY")
    print(f"  questions diagnosed         : {len(scored)}/{n}")
    print(f"  A  retrieval miss           : {a}  (gold never retrieved — needs RECALL, not ranking)")
    print(f"  B  truncation/ranking miss  : {b}  (gold retrieved but cut from context — "
          f"RECOVERABLE by ranking/budget/rerank-off)")
    print(f"  C  in-context (synthesis)   : {c}  (model saw gold, still missed — ranking can't help)")
    print()
    print("  READ: B is the addressable bucket. If B is large -> a ranking/budget change is")
    print("  worth a run. Re-run WITHOUT --rerank vs WITH --rerank: if B-items flip to C under")
    print("  raw BM25, the LLM reranker is demoting gold and rerank_message_hits=False is a")
    print("  one-flag fix. If C dominates -> the residual is synthesis, not ranking; stop here.")


def run_ms(ms: list[dict], args) -> None:
    from collections import Counter
    rerank_flag = True if args.rerank else False
    print(f"MS DIAGNOSIS — {len(ms)} multi-session questions  "
          f"(reranker={'ON (matches run)' if args.rerank else 'OFF (raw BM25)'}, "
          f"run_top_k={args.run_top_k}, base char_budget={MAX_CONTEXT_CHARS})\n", flush=True)
    reports = []
    for q in ms:
        tmp = Path(tempfile.mkdtemp(prefix="ms_probe_"))
        adapter = HyMemAdapter(
            db_path=tmp / "hymem.db",
            api_key=args.api_key,
            embeddings=False,
            rerank_message_hits=rerank_flag,
        )
        try:
            adapter.open()
            r = diagnose_ms(adapter, q, args.run_top_k, MAX_CONTEXT_CHARS)
            reports.append(r)
            print_ms_report(r)
        except Exception as e:
            print(f"  [ERROR on {q.get('question_id')}]: {type(e).__name__}: {e}")
        finally:
            try:
                adapter.close()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    n = len(reports)
    if not n:
        return
    abil = Counter(r["ability"] or "None" for r in reports)
    scored = [r for r in reports if r["bucket"] != "n/a"]
    a = sum(1 for r in scored if r["bucket"] == "A_retrieval")
    b = sum(1 for r in scored if r["bucket"] == "B_truncation")
    c = sum(1 for r in scored if r["bucket"] == "C_synthesis")
    mr = [r for r in reports if r["mr_whatif"]]
    safe = sum(1 for r in mr if r["mr_whatif"]["verdict"].startswith("SAFE"))
    risk = sum(1 for r in mr if r["mr_whatif"]["verdict"].startswith("RISK"))
    na = sum(1 for r in mr if r["mr_whatif"]["verdict"].startswith("n/a"))
    avg_noise = (sum(r["mr_whatif"]["noise"] for r in mr) / len(mr)) if mr else 0.0
    print("=" * 88)
    print("MS SUMMARY")
    print(f"  questions diagnosed   : {len(scored)}/{n}")
    print(f"  ability distribution  : " + ", ".join(f"{k}={v}" for k, v in abil.most_common()))
    print(f"  A retrieval miss      : {a}  (gold never retrieved — RECALL lever)")
    print(f"  B truncation          : {b}  (gold cut from context — ranking/budget lever)")
    print(f"  C in-context (synth)  : {c}  (model saw gold, still wrong — synthesis)")
    print()
    print(f"  MR-detected dead-filter what-if: {len(mr)} questions, avg assistant-noise {avg_noise * 100:.0f}%")
    print(f"    SAFE (filter keeps gold, drops noise) : {safe}")
    print(f"    RISK (filter would drop gold)         : {risk}")
    print(f"    n/a  (gold not in context anyway)     : {na}")
    print()
    print("  READ: if MR-detected is a big share + SAFE dominates + assistant-noise is nonzero")
    print("  -> applying the dead user-only filter BEFORE building context is upside; worth a run.")
    print("  If RISK > 0 -> filtering drops gold for those (gold is assistant); a blanket filter")
    print("  would regress them — filter only assistant ECHOES, or skip. If MS misses are mostly")
    print("  C on ability=None -> it's default-path synthesis, a different lever than the filter.")


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
    ap.add_argument("--ranking", action="store_true",
                    help="RANKING MODE: split each item into B (truncation) vs C (synthesis) "
                         "by reconstructing the final answer context offline")
    ap.add_argument("--ms", action="store_true",
                    help="MS MODE: multi-session localization — ability distribution, A/B/C split, "
                         "and the MR dead-filter what-if (current vs user-only context)")
    ap.add_argument("--rerank", action="store_true",
                    help="ranking mode: match the headline run's LLM message reranker (ON). "
                         "Default is raw BM25 (rerank off) — compare the two to test demotion.")
    ap.add_argument("--run-top-k", type=int, default=15,
                    help="the --top-k the headline run used (search sees top_k*3 candidates)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()

    scale = args.scale.upper()
    fname = "longmemeval_s_cleaned.json" if scale == "S" else f"longmemeval_{scale.lower()}_cleaned.json"
    data_file = Path(args.data_dir) / fname
    if not data_file.exists():
        print(f"ERROR: dataset not found at {data_file}")
        sys.exit(1)

    target = "multi-session" if args.ms else "knowledge-update"
    print(f"Loading {data_file} ...", flush=True)
    all_q = load_longmemeval_data(str(data_file), max_questions=None, seed=args.seed)
    ku = [q for q in all_q if q.get("question_type", "").replace("_abs", "") == target
          and not q.get("question_type", "").endswith("_abs")]
    if not ku:
        print(f"No {target} questions found in this dataset.")
        sys.exit(1)
    ku = ku[: args.n]

    if args.ms:
        run_ms(ku, args)
        return

    if args.ranking:
        run_ranking(ku, args)
        return

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
                # COVERAGE GATE: did the gold value mint an edge at all (vs only
                # living in raw turns)? Direct table scan, not retrieval-filtered.
                r["gold_edges"] = _gold_edges(adapter.hy.conn, str(q.get("answer", "")))
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
    present_not_latest = [r for r in reports if r["answer_present"] and not r["answer_on_latest"]]
    spoiler_turns = sum(len(r["spoilers"]) for r in present_not_latest)
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
          f"(naive latest-date-wins — pessimistic floor)")
    print(f"  present-but-NOT-latest (spoiled)  : {len(present_not_latest)}/{n}  "
          f"across {spoiler_turns} spoiler turns to classify above")
    if args.dream:
        edge_present = sum(1 for r in reports if r.get("gold_edges"))
        print()
        print(f"  >>> COVERAGE GATE — gold value present as graph edge : {edge_present}/{n}")
        print(f"      (direct knowledge_graph scan; the [MEM]-consumption lever's prerequisite)")
        print(f"      LOW  -> bottleneck is EXTRACTION coverage, not [MEM] ranking; L1-L3 have a")
        print(f"              low ceiling until extraction captures the value. Redirect upstream.")
        print(f"      HIGH -> value IS in the graph; L1 ([FACT] dating) + L2 (stale-[MEM] annot.)")
        print(f"              can consume it. CROSS-REF the per-question YES/NO against the KU zeros.")
    print()
    print("  CEILING of a VALUE-AWARE clause = (on-latest + tangential-spoiled)/n.")
    print("  Read the SPOILER blocks above and count, per spoiled question:")
    print("    (a) STALE-VALUE re-assertion  -> unfixable from raw turns (recency is genuinely wrong)")
    print("    (b) TANGENTIAL topic mention  -> a value-aware clause skips it (recoverable)")
    print("  If (b) dominates -> ceiling is high, a value-aware clause + dating is worth one run.")
    print("  If (a) shows up often -> raw-turn recency can't separate it; consider the graph")
    print("  (value-bearing edges, --dream) or stop — KU's big lift is already banked under permissive.")


if __name__ == "__main__":
    main()
