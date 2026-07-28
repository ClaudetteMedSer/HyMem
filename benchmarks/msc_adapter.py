#!/usr/bin/env python3
"""
HyMem Multi-Session Chat (MSC) Benchmark Adapter
================================================
Runs the MSC benchmark ("Beyond Goldfish Memory", Xu et al., ACL 2022) against
HyMem's Python SDK — the cross-session complement to `longmemeval_adapter.py`.

WHY MSC (and not just LME). LongMemEval is single-shot / star-topology: within a
question's haystack a fact appears once and never recurs across sessions. MSC is
genuinely multi-session — the SAME two speakers reconvene over up to 5 sessions
(hours-to-days apart) and their personas accumulate and get restated. That is
exactly the structure LME can't provide, and it's what the Idea-B repetition
signal, the `suggest_rules()` `session_count`, and Track-A multi-hop all need.

DATA (verified 2026-07-28 against the MemGPT/MSC-Self-Instruct HF dataset — the
concrete, downloadable QA derivative with clean labels):

    example = {
      "previous_dialogs": [ {"dialog": [{"text": str}, ...],   # the multi-session
                             "personas": [[str], [str]],       #   history to ingest
                             "time_num": int, "time_unit": str}, ... ],
      "self_instruct": {"B": <question>, "A": <gold answer>},  # the recall probe
      "personas": [[str], [str]], "init_personas": ..., "personas_update1/2": ...,
      "metadata": {"initial_data_id": str, "session_id": int},
    }

Two probe modes:

  --probe-mode recall   (the headline, LME-comparable number)
      Ingest `previous_dialogs` as sessions, dream, then ask `self_instruct.B`
      and judge against `self_instruct.A`. Measures cross-session fact recall,
      with an E1 accuracy-by-session-distance breakdown LME structurally can't
      produce. Reuses the LME answer/judge machinery verbatim (frozen posture).

  --probe-mode recurrence   (produces the E3 input for the existing engine)
      Ingest + dream, then DUMP the extracted behavioral markers with their
      HyMem session_id and an is_rule label derived from MSC's own persona
      annotations (a marker is "durable" iff it matches an annotated persona
      fact — MSC's ground truth, NOT the session_count we're validating, so no
      circularity). Feed the dump to `rule_extraction_experiment.py --labels ...
      --policy-from-canonical`: the honest retry of the corpus-artifact result,
      now on REAL cross-session recurrence. Note: MSC content is preference/fact
      shaped, so many markers are `preference`-kind (profile-tier, not rules) —
      the dump makes that empirically visible rather than assuming it.

Usage:
  python msc_adapter.py --data msc.json --probe-mode recall --sample 50
  python msc_adapter.py --data msc.json --probe-mode recurrence --out markers_msc.json
  python msc_adapter.py --sim            # offline: loader + labeling mechanics, no API
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling benchmark imports

# Reuse the LME machinery unchanged — same answer/judge clients and scoring keep
# MSC and LME numbers in ONE comparability frame (frozen posture). Imported
# lazily inside functions to keep --sim import-light and API-free.
_ANSWER_MODEL = "deepseek-v4-flash"
_JUDGE_MODEL = "deepseek-v4-flash"
_HYMEM_MODEL = "deepseek-v4-flash"   # NOT the deprecated deepseek-chat (2026-07-24)
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# ── lexical helpers (persona-fact matching, gold-turn location) ─────────────
# MSC recurrence ground truth and the E1 session-distance breakdown both need to
# ask "does this short fact appear in this turn/session?" — a content-word
# Jaccard, no embeddings required (an --embeddings cosine is a future upgrade).

_STOP = frozenset("a an the of to in on at for and or but is are was were be been "
                  "i you he she it we they my your his her our their me him them "
                  "this that these those with as by from about into over".split())


def _content_tokens(s: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    return {t for t in toks if len(t) > 2 and t not in _STOP}


def _lex_match(fact: str, text: str, tau: float = 0.5) -> bool:
    """True if `fact`'s content words are largely present in `text` (asymmetric
    Jaccard: |fact ∩ text| / |fact|). Asymmetric because a fact is 'stated' in a
    turn when the turn CONTAINS it, even if the turn says much more."""
    f = _content_tokens(fact)
    if not f:
        return False
    return len(f & _content_tokens(text)) / len(f) >= tau


# ── dataset loader ──────────────────────────────────────────────────────────

_TIME_UNIT_DAYS = {"hour": 1 / 24, "hours": 1 / 24, "day": 1.0, "days": 1.0,
                   "week": 7.0, "weeks": 7.0, "month": 30.0, "months": 30.0,
                   "year": 365.0, "years": 365.0}


def _gap_days(pd: dict, default: float) -> float:
    """Parse a previous_dialog's inter-session gap (time_num/time_unit) into days,
    falling back to `default` when the fields are missing/unparseable."""
    try:
        n = float(pd.get("time_num"))
        unit = str(pd.get("time_unit", "")).lower().strip()
        return max(n * _TIME_UNIT_DAYS.get(unit, 1.0), 0.01)
    except (TypeError, ValueError):
        return default


def _session_turns(pd: dict, start_role: str) -> list[dict]:
    """A previous_dialog's `dialog` (list of {text}) as alternating role turns.
    Speakers alternate; `start_role` says whether turn 0 is the user or the
    assistant. (Attribution matters little for retrieval — the answer-bearing
    turn is found by text either way — but keeping it stable lets HyMem's
    user-centric extraction treat one side consistently as 'the user'.)"""
    other = "assistant" if start_role == "user" else "user"
    turns = []
    for i, t in enumerate(pd.get("dialog") or []):
        content = (t.get("text") if isinstance(t, dict) else str(t)) or ""
        if content.strip():
            turns.append({"role": start_role if i % 2 == 0 else other,
                          "content": content.strip()})
    return turns


def _persona_facts(ex: dict) -> list[str]:
    """Every annotated persona line for the example, de-duplicated — MSC's
    'important personal points', used as the durability ground truth."""
    facts: list[str] = []
    def _add(x):
        if isinstance(x, str) and x.strip():
            facts.append(x.strip())
        elif isinstance(x, list):
            for y in x:
                _add(y)
    for key in ("personas", "init_personas", "personas_update1", "personas_update2"):
        _add(ex.get(key))
    for pd in ex.get("previous_dialogs") or []:
        _add(pd.get("personas"))
    seen, out = set(), []
    for f in facts:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def load_msc_data(path: str | None, sample: int, seed: int, *,
                  start_role: str = "user", gap_days: float = 1.0,
                  base_date: str = "2023-01-01") -> list[dict]:
    """Load + normalize MSC examples. Accepts a JSON array or JSONL of the
    MemGPT/MSC-Self-Instruct shape. Returns normalized examples:
        {id, sessions: [[{role,content}]], session_dates: [str], question, answer,
         persona_facts: [str], n_sessions}
    Session dates are SYNTHESIZED (MSC carries relative gaps, not timestamps):
    monotonic from `base_date`, spaced by each session's parsed gap — real
    temporal separation for the recency/supersession mechanics, not ground-truth
    dates. `path=None` returns the built-in synthetic fixture (for --sim)."""
    if not path:
        raw = _SIM_FIXTURE
    else:
        text = Path(path).read_text(encoding="utf-8")
        raw = ([json.loads(l) for l in text.splitlines() if l.strip()]
               if path.endswith(".jsonl") else json.loads(text))
        if not isinstance(raw, list):
            raise ValueError("MSC data must be a JSON array (or JSONL) of examples")

    out = []
    base = datetime.strptime(base_date, "%Y-%m-%d")
    for i, ex in enumerate(raw):
        prev = ex.get("previous_dialogs") or []
        sessions, dates, cursor = [], [], base
        for pd in prev:
            turns = _session_turns(pd, start_role)
            if not turns:
                continue
            sessions.append(turns)
            dates.append(cursor.strftime("%Y-%m-%d %H:%M"))
            cursor += timedelta(days=_gap_days(pd, gap_days))
        if not sessions:
            continue
        si = ex.get("self_instruct") or {}
        out.append({
            "id": str((ex.get("metadata") or {}).get("initial_data_id") or f"msc_{i}"),
            "sessions": sessions,
            "session_dates": dates,
            "question": (si.get("B") or "").strip() or None,
            "answer": (si.get("A") or "").strip() or None,
            "persona_facts": _persona_facts(ex),
            "n_sessions": len(sessions),
        })
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:sample] if sample else out


# ── the HyMem driver (self-contained; configurable model) ───────────────────

class MSCAdapter:
    """A HyMem instance over one example's isolated temp DB. Mirrors
    `HyMemAdapter` but is self-contained so it controls the dream model (the LME
    adapter hardcodes the deprecated deepseek-chat) and carries only the levers
    MSC needs."""

    def __init__(self, db_path: Path, *, api_key: str = "", sim: bool = False,
                 hymem_model: str = _HYMEM_MODEL, hymem_base_url: str = _DEEPSEEK_BASE_URL,
                 embeddings: bool = False, rules_extraction: bool | None = None,
                 graph_multihop: bool = False):
        self.db_path = db_path
        self.api_key = api_key
        self.sim = sim
        self.hymem_model = hymem_model
        self.hymem_base_url = hymem_base_url
        self.embeddings = embeddings
        self.rules_extraction = rules_extraction
        self.graph_multihop = graph_multihop
        self.hy = None

    def open(self):
        from hymem import HyMem, HyMemConfig
        overrides: dict = {}
        if self.rules_extraction is not None:
            overrides["rules_extraction_enabled"] = self.rules_extraction
        if self.graph_multihop:
            overrides["graph_multihop_enabled"] = True
        cfg = HyMemConfig(root=self.db_path.parent, message_fts_top_k=15,
                          fts_top_k=10, graph_top_k=10, **overrides)
        embedding_client = None
        if self.sim:
            from hymem.extraction.llm import StubLLMClient
            llm = StubLLMClient(default="[]")
        else:
            from hymem.contrib.openai_client import OpenAICompatibleClient
            llm = OpenAICompatibleClient(
                api_key=self.api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
                base_url=self.hymem_base_url, model=self.hymem_model)
            if self.embeddings:
                from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
                env = os.environ.get
                embedding_client = OpenAICompatibleEmbeddingClient(
                    api_key=env("HYMEM_EMBEDDING_API_KEY") or env("HYMEM_LLM_API_KEY") or "sk-local",
                    base_url=env("HYMEM_EMBEDDING_BASE_URL") or "http://localhost:8000/v1",
                    model=env("HYMEM_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5",
                    dim=int(env("HYMEM_EMBEDDING_DIM") or 384))
        self.hy = HyMem(cfg, llm=llm, embedding_client=embedding_client)
        return self

    def close(self):
        if self.hy:
            self.hy.close()
            self.hy = None

    def ingest(self, ex: dict) -> None:
        """One HyMem session per MSC session — NEVER merged (session_count is the
        whole point of the corpus), each stamped with its synthesized date."""
        for i, (turns, date) in enumerate(zip(ex["sessions"], ex["session_dates"])):
            self.hy.log_messages(
                f'{ex["id"]}_s{i}',
                [(t["role"], t["content"], date) for t in turns])

    def dream(self) -> None:
        dh = self.hy.fork()
        try:
            dh.dream()
        finally:
            dh.close()

    def search(self, query: str, top_k: int = 10) -> tuple[list[dict], int]:
        """Minimal message-first retrieval shaped for the reused `answer_question`
        (raw turns lead; MSC recall is a retrieval-over-turns task)."""
        result = self.hy.augment(query)
        memories: list[dict] = []
        for hit in getattr(result, "message_hits", None) or []:
            text = (getattr(hit, "text", "") or "")[:600]
            if text.strip():
                memories.append({"content": f'[{getattr(hit, "role", "?")}] {text}',
                                 "type": "message_hit", "confidence": 0.7,
                                 "created_at": getattr(hit, "created_at", "") or ""})
        for hit in getattr(result, "fts_hits", None) or []:
            text = (getattr(hit, "text", "") or "")[:600]
            if text.strip():
                memories.append({"content": text, "type": "fts_hit", "confidence": 0.6})
        return memories[:top_k], getattr(result, "total_message_matches", 0)

    def dump_markers(self, ex: dict) -> list[dict]:
        """Extracted behavioral markers with HyMem session_id + an is_rule label
        from MSC's persona annotations. Read post-dream (rows persist even after
        consolidation); the JOIN gives the per-MSC-session provenance recurrence
        needs."""
        rows = self.hy.conn.execute(
            "SELECT bm.kind AS kind, bm.statement AS statement, c.session_id AS session_id "
            "FROM behavioral_markers bm JOIN chunks c ON bm.chunk_id = c.id "
            "ORDER BY bm.id").fetchall()
        facts = ex["persona_facts"]
        out = []
        for r in rows:
            stmt = r["statement"]
            is_rule = any(_lex_match(f, stmt) or _lex_match(stmt, f) for f in facts)
            out.append({"kind": r["kind"], "statement": stmt,
                        "session_id": r["session_id"], "is_rule": bool(is_rule)})
        return out


# ── probes ──────────────────────────────────────────────────────────────────

def _gold_session_index(ex: dict) -> int:
    """Which session the gold answer was stated in (best lexical match), or -1.
    Drives the E1 accuracy-by-session-distance breakdown."""
    ans = ex.get("answer") or ""
    best = -1
    for i, turns in enumerate(ex["sessions"]):
        if any(_lex_match(ans, t["content"], tau=0.6) for t in turns):
            best = i
    return best


def run_recall(ex: dict, args, answer_llm, judge_llm) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="msc_"))
    adapter = MSCAdapter(tmp / "m.sqlite", api_key=args.api_key, sim=args.sim,
                         hymem_model=args.hymem_model, hymem_base_url=args.hymem_base_url,
                         embeddings=args.embeddings, rules_extraction=args.rules_extraction,
                         graph_multihop=args.graph_multihop).open()
    try:
        adapter.ingest(ex)
        if not args.no_dream:
            adapter.dream()
        memories, total = adapter.search(ex["question"], top_k=args.top_k)
        if args.sim:
            # Offline: no answer/judge LLM. Test the thing --sim CAN test — did
            # retrieval surface the gold answer? — via a lexical check.
            joined = " ".join(m["content"] for m in memories)
            ai = memories[0]["content"] if memories else ""
            correct = _lex_match(ex["answer"], joined, tau=0.6)
        else:
            from longmemeval_adapter import answer_question, judge_answer
            question_date = ex["session_dates"][-1] if ex["session_dates"] else ""
            ai = answer_question(answer_llm, memories, ex["question"],
                                 total_matches=total, question_date=question_date)
            correct = judge_answer(judge_llm, "single-session-user",
                                   ex["question"], ex["answer"], ai)
        gi = _gold_session_index(ex)
        return {"id": ex["id"], "question_type": "recall", "correct": bool(correct),
                "question": ex["question"], "answer": ex["answer"], "ai_answer": ai,
                "n_sessions": ex["n_sessions"], "gold_session": gi,
                "gold_distance": (ex["n_sessions"] - gi) if gi >= 0 else -1,
                "n_memories": len(memories)}
    finally:
        adapter.close()
        if not args.keep_db:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


def run_recurrence_dump(ex: dict, args) -> list[dict]:
    tmp = Path(tempfile.mkdtemp(prefix="msc_"))
    adapter = MSCAdapter(tmp / "m.sqlite", api_key=args.api_key, sim=args.sim,
                         hymem_model=args.hymem_model, hymem_base_url=args.hymem_base_url,
                         rules_extraction=args.rules_extraction).open()
    try:
        adapter.ingest(ex)
        if not args.no_dream:
            adapter.dream()
        return adapter.dump_markers(ex)
    finally:
        adapter.close()
        if not args.keep_db:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ── reporting ───────────────────────────────────────────────────────────────

def _print_recall_report(results: list[dict]) -> None:
    from longmemeval_adapter import compute_scores
    scores = compute_scores(results)
    ov = scores["OVERALL"]
    print(f"\n=== MSC recall — n={ov['count']} ===")
    print(f"  overall accuracy: {ov['accuracy']*100:.1f}%\n")
    # E1: accuracy by how many sessions back the gold fact was stated.
    from collections import defaultdict
    by_dist: dict[int, list[bool]] = defaultdict(list)
    for r in results:
        by_dist[r.get("gold_distance", -1)].append(r["correct"])
    print("  ── accuracy by session distance (E1: recall vs how far back) ──")
    print(f"  {'distance':>9} {'acc':>7} {'n':>5}")
    for d in sorted(by_dist):
        c = by_dist[d]
        label = "unknown" if d < 0 else f"{d} back"
        print(f"  {label:>9} {sum(c)/len(c)*100:>6.1f}% {len(c):>5}")


def _print_recurrence_summary(markers: list[dict], out_path: str | None) -> None:
    from collections import Counter
    from hymem.rules import is_rule_eligible_kind
    kinds = Counter(m["kind"] for m in markers)
    eligible = sum(is_rule_eligible_kind(m["kind"]) for m in markers)
    rules = sum(m["is_rule"] for m in markers)
    sessions = {m["session_id"] for m in markers}
    print(f"\n=== MSC recurrence dump — {len(markers)} markers ===")
    print(f"  kinds: {dict(kinds)}")
    print(f"  rule-eligible (rejection/style/correction): {eligible}  "
          f"[preference→profile, not rules: {len(markers)-eligible}]")
    print(f"  is_rule (matches an annotated persona fact): {rules}")
    print(f"  distinct sessions represented: {len(sessions)}")
    if out_path:
        Path(out_path).write_text(json.dumps(markers, indent=2), encoding="utf-8")
        print(f"\n  markers → {out_path}")
        print("  next: python benchmarks/rule_extraction_experiment.py \\")
        print(f"           --labels {out_path} --answer-model <tagger> --policy-from-canonical")
    else:
        print("\n  (pass --out markers_msc.json to feed rule_extraction_experiment.py)")
    # Honest read: if rule-eligible ≈ 0, MSC exercises the PROFILE tier, not rules
    # — which is itself the finding (MSC content is preference/fact shaped).
    if markers and eligible == 0:
        print("\n  NOTE: 0 rule-eligible markers — MSC content is preference/fact shaped,")
        print("        so it drives the profile tier, not the rules recurrence signal.")


# ── main ────────────────────────────────────────────────────────────────────

def _build_llm(model, base_url, api_key, extra_body):
    from longmemeval_adapter import LLMClient
    return LLMClient(model=model, api_key=api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
                     base_url=base_url, extra_body=extra_body)


def main() -> None:
    ap = argparse.ArgumentParser(description="HyMem MSC benchmark adapter.")
    ap.add_argument("--data", default=None, help="MSC JSON/JSONL (MemGPT/MSC-Self-Instruct shape)")
    ap.add_argument("--probe-mode", choices=["recall", "recurrence"], default="recall")
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--start-role", choices=["user", "assistant"], default="user")
    ap.add_argument("--session-gap-days", type=float, default=1.0,
                    help="fallback inter-session spacing when MSC gaps are missing")
    ap.add_argument("--answer-model", default=_ANSWER_MODEL)
    ap.add_argument("--answer-base-url", default=_DEEPSEEK_BASE_URL)
    ap.add_argument("--answer-api-key", default=None)
    ap.add_argument("--answer-extra-body", default=None, metavar="JSON",
                    help='e.g. \'{"thinking":{"type":"disabled"}}\' for v4-flash')
    ap.add_argument("--judge-model", default=_JUDGE_MODEL)
    ap.add_argument("--judge-extra-body", default=None, metavar="JSON")
    ap.add_argument("--hymem-model", default=_HYMEM_MODEL, help="HyMem's dream LLM")
    ap.add_argument("--hymem-base-url", default=_DEEPSEEK_BASE_URL)
    ap.add_argument("--api-key", default="", help="HyMem dream LLM key")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--rules-extraction", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--graph-multihop", action="store_true")
    ap.add_argument("--no-dream", action="store_true")
    ap.add_argument("--keep-db", action="store_true")
    ap.add_argument("--out", default=None, help="recurrence: write the marker dump here")
    ap.add_argument("--sim", action="store_true", help="offline: StubLLM, no API")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    examples = load_msc_data(args.data, args.sample, args.seed,
                             start_role=args.start_role, gap_days=args.session_gap_days)
    if not examples:
        print("No MSC examples loaded.")
        sys.exit(1)
    print(f"Loaded {len(examples)} MSC examples "
          f"(sessions/ex: {sum(e['n_sessions'] for e in examples)/len(examples):.1f} avg)"
          f"{'  [SIM]' if args.sim else ''}", flush=True)

    if args.probe_mode == "recall":
        recall_ex = [e for e in examples if e["question"] and e["answer"]]
        if not recall_ex:
            print("No examples carry a self_instruct QA pair — recall mode needs them.")
            sys.exit(1)
        answer_llm = judge_llm = None
        if not args.sim:
            ab = json.loads(args.answer_extra_body) if args.answer_extra_body else None
            jb = json.loads(args.judge_extra_body) if args.judge_extra_body else None
            answer_llm = _build_llm(args.answer_model, args.answer_base_url, args.answer_api_key, ab)
            judge_llm = _build_llm(args.judge_model, _DEEPSEEK_BASE_URL, args.answer_api_key, jb)
        results = []
        t0 = time.time()
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(run_recall, e, args, answer_llm, judge_llm): e
                        for e in recall_ex}
                for k, fut in enumerate(as_completed(futs), 1):
                    results.append(fut.result())
                    print(f"  [{k}/{len(recall_ex)}]", end="\r", flush=True)
        else:
            for k, e in enumerate(recall_ex, 1):
                results.append(run_recall(e, args, answer_llm, judge_llm))
                print(f"  [{k}/{len(recall_ex)}]", end="\r", flush=True)
        print(f"  done in {time.time()-t0:.0f}s{' '*20}")
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            _print_recall_report(results)

    else:  # recurrence
        all_markers: list[dict] = []
        for k, e in enumerate(examples, 1):
            all_markers.extend(run_recurrence_dump(e, args))
            print(f"  [{k}/{len(examples)}]", end="\r", flush=True)
        print(" " * 30, end="\r")
        _print_recurrence_summary(all_markers, args.out)


# A tiny in-schema fixture so --sim exercises the loader + labeling with no API.
_SIM_FIXTURE = [{
    "metadata": {"initial_data_id": "sim_0", "session_id": 3},
    "previous_dialogs": [
        {"time_num": 2, "time_unit": "days",
         "personas": [["I have two dogs"], ["I love hiking"]],
         "dialog": [{"text": "I just adopted two dogs, a lab and a beagle."},
                    {"text": "Nice! I love hiking with my dog on weekends."}]},
        {"time_num": 5, "time_unit": "hours",
         "personas": [["I have two dogs"], ["I work as a nurse"]],
         "dialog": [{"text": "The dogs kept me busy, but work as a nurse is hectic too."},
                    {"text": "Hiking clears my head after long shifts."}]},
    ],
    "personas": [["I have two dogs"], ["I love hiking", "I work as a nurse"]],
    "init_personas": [["I have two dogs"], ["I love hiking"]],
    "self_instruct": {"B": "How many dogs did you say you have?", "A": "Two dogs."},
}]


if __name__ == "__main__":
    main()
