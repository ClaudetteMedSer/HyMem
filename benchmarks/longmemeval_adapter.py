#!/usr/bin/env python3
"""
HyMem LongMemEval Benchmark Adapter
====================================
Runs the LongMemEval benchmark (ICLR 2025) against HyMem's Python SDK.

LongMemEval tests 5 core long-term memory abilities:
  - Information Extraction (single-session-user, single-session-assistant)
  - Multi-session Reasoning
  - Temporal Reasoning
  - Knowledge Update
  - Abstention

Usage:
  python longmemeval_adapter.py --sample 50 --scales S
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests as http

# Add HyMem to path
# Ensure the HyMem package is importable (repo root is two levels up from benchmarks/)
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

# ── Config ──────────────────────────────────────────────────────────

def _normalize_date(raw: str | None) -> str | None:
    """Convert a LongMemEval haystack_date like '2023/05/20 (Sat) 02:21'
    to ISO-8601 '2023-05-20T02:21:00'. Returns None for empty/None input."""
    if not raw or not raw.strip():
        return None
    # Strip day-of-week parenthetical: '2023/05/20 (Sat) 02:21' -> '2023/05/20 02:21'
    import re
    cleaned = re.sub(r'\s*\([^)]*\)', '', raw).strip()
    # Try common formats
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    # Return cleaned if we can't parse — better than wall-clock
    return cleaned if cleaned else None


# ── Recall-ceiling instrumentation ──────────────────────────────────
# A category's miss has two opposite root causes that need opposite fixes:
#   - retrieval loss: the gold turn never entered the candidate pool at all
#     (fix = embeddings / chunking / cross-session fan-out)
#   - ranking/synthesis loss: the gold turn WAS retrieved but lost the cut or
#     the model couldn't assemble it (fix = rerank / wider budget / packing)
# These helpers answer, per question, "did the answer-bearing turn appear
# ANYWHERE in the pre-truncation retrieval pool?" — splitting the two so a
# fix targets the right stage instead of being a coin flip.

def _norm_text(s: str) -> str:
    """Whitespace-collapse + lowercase for robust substring matching."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# Minimum normalized answer length for the containment check below to carry
# any signal. LME answers are often a bare value ("40", "ruff"), and a 2-char
# string appears inside some longer text by chance — which would report as a
# hit. Mirrors `_MIN_ANSWER_CHARS` in benchmarks/fact_probe.py, where the same
# trap was found and named.
_MIN_ANSWER_CHARS = 4


def _answer_in_texts(answer: str, texts: list[str]) -> bool | None:
    """Does the gold ANSWER string appear in any of `texts`?

    Deliberately NOT `_gold_in_pool`: that one matches gold TURNS (long
    strings) and accepts a match in EITHER direction, which is right for turns
    and wrong for a short answer — "40" is inside a thousand sentences.
    Containment is one-directional here, and an answer too short to be
    distinctive returns None ("unmeasurable") rather than a fabricated
    True/False, so a short-answer category can't silently report signal.
    """
    a = _norm_text(answer)
    if len(a) < _MIN_ANSWER_CHARS:
        return None
    return any(a in _norm_text(t) for t in texts if t and t.strip())


def _extract_gold_turns(q_data: dict) -> tuple[list[str], str]:
    """Return (gold_turn_contents, mode).

    Prefers LongMemEval's turn-level `has_answer: true` flags (mode="turn",
    the precise signal). Falls back to every turn of the `answer_session_ids`
    sessions (mode="session", coarser — any turn from an answer session counts).
    Returns ([], "none") when the dataset carries neither, so the question is
    excluded from the ceiling rate rather than scored against a fabricated gold.
    """
    sessions = q_data.get("haystack_sessions", []) or []
    session_ids = q_data.get("haystack_session_ids",
                             [str(i) for i in range(len(sessions))])

    gold: list[str] = []
    for sess in sessions:
        for m in sess:
            if isinstance(m, dict) and m.get("has_answer"):
                c = m.get("content", "")
                if c.strip():
                    gold.append(c)
    if gold:
        return gold, "turn"

    ans_ids = set(q_data.get("answer_session_ids", []) or [])
    if ans_ids:
        for sid, sess in zip(session_ids, sessions):
            if sid in ans_ids:
                for m in sess:
                    c = m.get("content", "") if isinstance(m, dict) else ""
                    if c.strip():
                        gold.append(c)
        if gold:
            return gold, "session"

    return [], "none"


def _gold_in_pool(gold_turns: list[str], pool_texts: list[str]) -> bool:
    """True if any gold turn is present in any pooled hit text.

    message_hits expose the raw turn (truncated to 600 chars); fts chunks are a
    slice of one. So a match is: one string contains the other, or they share a
    distinctive 40-char prefix (covers the 600-char cap and chunk slicing)."""
    pool_n = [_norm_text(p) for p in pool_texts if p and p.strip()]
    for g in gold_turns:
        gn = _norm_text(g)
        if not gn:
            continue
        for pn in pool_n:
            if not pn:
                continue
            if gn in pn or pn in gn or (len(gn) >= 40 and gn[:40] in pn):
                return True
    return False


def _gold_turn_tiers(gold_turns: list[str], pool: dict) -> list[str]:
    """Per-gold-turn membership in the FUSED pool: which tier (if any) carries each
    of a question's N gold turns. Unlike recall_ceiling (an any-match bool — "is SOME
    gold turn in the pool"), this de-conflates multi-gold MS questions: a "none" entry
    is a gold turn the whole pipeline (message + chunk/fts) failed to retrieve. The
    L3 floor audit reads off this directly — a "floor" question (gold ∉ raw message
    FTS, per the probe) is a PHANTOM if every turn here is non-"none" (chunks/embeddings
    rescued it), or a REAL recall gap if any turn is still "none" with both tiers run."""
    msg, fts = pool.get("message", []), pool.get("fts", [])
    tiers: list[str] = []
    for g in gold_turns:
        in_m = _gold_in_pool([g], msg)
        in_f = _gold_in_pool([g], fts)
        tiers.append("both" if in_m and in_f
                     else "message" if in_m
                     else "fts" if in_f else "none")
    return tiers


# ── Ability-router instrumentation ──────────────────────────────────
# The harness shapes retrieval from the ORACLE question_type label, but real
# Hermes has no such label — augment() must infer the ability itself via
# detect_ability(). So any MR/TR gain banked under the oracle label can be
# illusory in production if the router misses. We record the router's verdict
# on every run (free) and can optionally DRIVE shaping from it (--auto-ability)
# to measure the true production score.

def _detect_ability_safe(question: str) -> str | None:
    """HyMem's production ability inference, or None if unavailable.

    detect_ability emits only "MR"/"TR"/None by design — those are the only
    wired-in shaping paths; every other oracle ability (IE/KU/PF/ABS) correctly
    maps to a None inference (no shaping), so a None here on those categories is
    a correct abstain, not a miss."""
    try:
        from hymem.query.intent import detect_ability
        return detect_ability(question or "")
    except Exception:
        return None


DEFAULT_SCALE = "S"
DEFAULT_SAMPLE = 50  # questions to evaluate (500 total)
DEFAULT_TOP_K = 15
MAX_CONTEXT_CHARS = 8000

# DeepSeek API
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ANSWER_MODEL = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"

# Local embedding server (lever L1) — the FastEmbed ONNX server Hermes runs in
# production. These are the OUT-OF-THE-BOX defaults for --embeddings so the flag
# works with no env setup; every field is still overridable via HYMEM_EMBEDDING_*.
# DeepSeek has no embeddings API, so the client's own deepseek defaults are a dead
# end — point at the local server. api_key="local" because the server ignores it.
LOCAL_EMBED_BASE_URL = "http://localhost:8766/v1"
LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBED_DIM = 384
LOCAL_EMBED_API_KEY = "local"

# Recency-conflict resolution (KU lever). message_hits are stamped with their date
# in the answer context (see the context builder in answer_question), so when a
# fact was UPDATED over time the model can prefer the newest *value-bearing*
# statement. The KU probe (benchmarks/ku_probe.py) showed every strict KU miss is
# "present-but-not-latest": the new value IS retrieved, but a later turn merely
# RE-MENTIONS the topic without restating the value (37/37 spoiler turns were
# tangential, zero stale re-assertions), so naive latest-date-wins would pick the
# wrong turn. This clause makes recency VALUE-AWARE — a later mention carrying no
# value does not override an earlier turn that states one. Label-free (reads no
# question_type), always-on, and inert for single-value categories (never fires
# without a genuine multi-date conflict). Appended to BOTH default prompts because
# the headline config runs the permissive default.
RECENCY_CONFLICT_CLAUSE = (
    "\nSome memories are stamped with their date, e.g. [MEM 2023-11-30]. When the same fact "
    "appears with different values at different dates, use the value from the MOST RECENT memory "
    "that actually states that value — a later memory that only mentions the topic without giving "
    "the value does NOT override an earlier one that does."
)

ANSWERING_SYSTEM_PROMPT = ("""You are an AI assistant answering questions based on retrieved memories from past conversations.
Answer the question concisely using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Do not make up information. Do not use outside knowledge.""" + RECENCY_CONFLICT_CLAUSE)

ANSWERING_PREFERENCE_PROMPT = """You are an AI assistant answering questions based on retrieved memories from past conversations.
The context contains personal information about the user (preferences, possessions, habits, experiences).
Use this personal information to generate a personalized response to the question.
You may draw on general knowledge to fill in details, but tailor your answer to respect what you know about the user.
If the context contains NO relevant personal information about the user, say "I don't have enough information to answer this question." """

# Permissive DEFAULT prompt (lever D4 — the SS-P auto-ability crater fix).
# The strict ANSWERING_SYSTEM_PROMPT ("ONLY provided context, no outside
# knowledge") is the right posture for factual lookups but craters preference/
# recommendation questions: those need the model to bridge the user's stored
# preference ("uses Premiere Pro") to general knowledge ("here are editing
# resources"). The oracle path routes SS-P → ANSWERING_PREFERENCE_PROMPT, but the
# production router (detect_ability) can only emit MR/TR/None, so a label-free
# SS-P question falls to the default `else` branch and gets the strict prompt →
# refusal. This permissive default mirrors the preference posture for the
# unknown-ability case so the fix carries WITHOUT reading the oracle label.
# KEPT the abstention guard (last two sentences) — it must still say "I don't
# know" when the context lacks the asked information, so the `*_abs` slice is not
# silently traded away. Whether it IS traded away is what the broken-out
# abstention report measures.
ANSWERING_PERMISSIVE_PROMPT = ("""You are an AI assistant answering questions based on retrieved memories from past conversations.
The context contains personal information about the user (preferences, possessions, habits, experiences, history).
Use this personal information to give a helpful, personalized answer to the question.
For recommendations, suggestions, or advice you MAY draw on general knowledge — but ground the answer in what the context actually tells you about the user.
If the context contains NO information relevant to what the question asks, say "I don't have enough information to answer this question."
Do not invent specific facts about the user (names, dates, numbers, events) that the context does not support.""" + RECENCY_CONFLICT_CLAUSE)

ANSWERING_MR_PROMPT = """You are an AI assistant answering questions based on retrieved memories from multiple conversation sessions.
The question requires counting or aggregating information across sessions.
Carefully scan ALL the context for every relevant mention. Count distinct items — do not double-count restatements.
If the question asks "how many", return just the number (or a short answer with the number).
If you cannot find enough evidence in the context, say "I don't have enough information to answer."
Do not make up information."""

ANSWERING_TR_PROMPT = """You are an AI assistant answering questions based on retrieved memories from past conversations.
The question requires reasoning about when events happened — dates, timelines, or the order of events.
Carefully scan ALL the context for relevant dates, times, and event mentions.
Calculate the answer from the evidence provided. For dated events, compute durations precisely.
If you cannot determine the answer from the context, say "I don't have enough information."
Do not make up dates or events."""

# ── LongMemEval question type → HyMem ability mapping ────────────────

QUESTION_TYPE_TO_ABILITY = {
    "single-session-user": "IE",
    "single-session-assistant": "IE",
    "multi-session": "MR",
    "temporal-reasoning": "TR",
    "knowledge-update": "KU",
    "single-session-preference": "PF",
    "single-session-user_abs": "ABS",
    "single-session-assistant_abs": "ABS",
    "multi-session_abs": "ABS",
    "temporal-reasoning_abs": "ABS",
    "knowledge-update_abs": "ABS",
    "single-session-preference_abs": "ABS",
}

# ── LLM outage sentinel ─────────────────────────────────────────────
#
# `LLMClient.chat` returns this marker string instead of raising once retries are
# exhausted, so an outage travels through the pipeline as DATA. Five sites spelt
# the prefix out by hand; they now share one predicate, because the D3 fix turns
# it from a cosmetic detail into a decision about whether a row is scored.
LLM_ERROR_PREFIX = "[LLM_ERROR"


def is_llm_error(text: str | None) -> bool:
    return bool(text) and str(text).startswith(LLM_ERROR_PREFIX)


# ── LLM Client ──────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str, api_key: str, base_url: str = DEEPSEEK_BASE_URL,
                 extra_body: dict | None = None):
        self.model = model
        self.api_key = api_key
        # Default keeps every existing caller byte-path-identical; only the ANSWER
        # client is ever pointed elsewhere (via --answer-base-url), so the judge
        # posture stays the frozen comparability contract with the canonical run.
        self.base_url = base_url.rstrip("/")
        # Extra top-level request-body fields merged into every call — the raw-HTTP
        # equivalent of the OpenAI SDK's `extra_body`. Needed post-2026-07-24: the
        # deepseek-chat deprecation moved reader/judge to deepseek-v4-flash, a
        # REASONING model that prepends thinking tokens (corrupting the yes/no judge
        # parse) unless sent {"thinking":{"type":"disabled"}}. Empty = unchanged.
        self.extra_body = dict(extra_body) if extra_body else {}
        self.call_count = 0
        self.total_tokens = 0
        self.last_error: str | None = None
        # Guards the two counters so they aggregate correctly when many worker
        # threads share this client (--workers > 1).
        self._lock = threading.Lock()

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        last_error = None
        for attempt in range(3):
            try:
                content, usage = self._call(messages, temperature, max_tokens)
                with self._lock:
                    self.total_tokens += usage.get("total_tokens", 0)
                return content
            except Exception as e:
                last_error = str(e)
                if "429" in last_error or "rate" in last_error.lower():
                    time.sleep(15 * (attempt + 1))
                elif "null content" in last_error and "finish=length" in last_error:
                    # Deterministic truncation: the reasoning model burned the
                    # output budget and emitted no answer. Retrying with the SAME
                    # cap re-spends the call for the same result — fail fast and
                    # let the caller's parse-failure ceiling catch the aggregate.
                    break
                elif attempt < 2:
                    time.sleep(3)
                else:
                    break
        self.last_error = last_error
        return f"[LLM_ERROR: {last_error[:100]}]"

    def _call(self, messages: list, temperature: float, max_tokens: int) -> tuple[str, dict]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Merge last so a caller can force provider-specific fields (e.g. disable
        # v4-flash thinking). Collisions with the four keys above are the caller's.
        body.update(self.extra_body)
        resp = http.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        with self._lock:
            self.call_count += 1
        content = data["choices"][0]["message"].get("content")
        if content is None:
            # A 200 with content=null. Transient provider behavior should be
            # retried like a 429; a null from finish_reason=length is
            # deterministic truncation (chat() fails that one fast). Either
            # way, returning None used to crash callers (`None.startswith`) —
            # raising makes the failure explicit and countable.
            raise RuntimeError(
                f"null content (finish={data['choices'][0].get('finish_reason')})")
        return (
            content,
            data.get("usage", {}),
        )


# ── Dataset Loader (streaming) ──────────────────────────────────────

def load_longmemeval_data(dataset_path: str, max_questions: int = None, seed: int = 0) -> list[dict]:
    """Stream-load LongMemEval questions using ijson, with stratified sampling.

    `seed` makes the stratified sample + shuffle deterministic so two runs
    (e.g. old code vs new code) evaluate the IDENTICAL question set — without
    it every run draws a fresh sample and per-category deltas are dominated by
    which questions happened to be drawn, not by the code change. Pass
    `max_questions=None` (CLI `--sample 0`) to evaluate the full set and remove
    sampling variance entirely.
    """
    import ijson, random

    rng = random.Random(seed)

    # First pass: collect all questions grouped by type
    by_type = defaultdict(list)
    with open(dataset_path, "rb") as f:
        for item in ijson.items(f, "item"):
            # Official LongMemEval flags ABSTENTION questions via the question_id
            # suffix `_abs`, NOT question_type — but the judge (get_judge_prompt)
            # and the answerable-vs-abstention report (compute_abstention_scores)
            # both key on `_abs` in question_type. Normalize here so an abstention-
            # bearing dataset measures correctly and the --permissive-default guard
            # rail actually fires. The `_cleaned` S file dropped abstention entirely
            # (all-answerable), so this is a no-op there.
            qtype = item["question_type"]
            if str(item.get("question_id", "")).endswith("_abs") and not qtype.endswith("_abs"):
                qtype = f"{qtype}_abs"
                item["question_type"] = qtype
            by_type[qtype].append(item)

    total_available = sum(len(v) for v in by_type.values())
    num_types = len(by_type)
    n_abs = sum(len(v) for k, v in by_type.items() if k.endswith("_abs"))
    if n_abs:
        print(f"  Abstention questions present: {n_abs} (guard-rail measurable)", flush=True)
    else:
        print(f"  ⚠ No abstention (_abs) questions in this dataset — the "
              f"answerable-vs-abstention guard rail cannot fire (all-answerable set)", flush=True)

    if max_questions is None or max_questions >= total_available:
        # Return all questions
        questions = [q for group in by_type.values() for q in group]
        print(f"  Loaded all {total_available} questions ({num_types} types)", flush=True)
        return questions

    # Stratified sampling: distribute max_questions across types
    per_type = max(1, max_questions // num_types)
    remaining = max_questions - per_type * num_types

    questions = []
    for qtype, items in sorted(by_type.items()):
        n = min(per_type + (1 if remaining > 0 else 0), len(items))
        sampled = rng.sample(items, n) if n < len(items) else items
        questions.extend(sampled)
        if remaining > 0:
            remaining -= 1
        print(f"    {qtype}: {n}/{len(items)} sampled", flush=True)

    rng.shuffle(questions)
    print(f"  Loaded {len(questions)} questions ({num_types} types, stratified, seed={seed})", flush=True)
    return questions


def load_longmemeval_oracle(oracle_path: str) -> dict[str, dict]:
    """Load oracle file for answer references."""
    with open(oracle_path) as f:
        oracle_data = json.load(f)
    return {q["question_id"]: q for q in oracle_data}


# ── HyMem Integration ───────────────────────────────────────────────

class HyMemAdapter:
    """Direct HyMem Python API adapter with isolated temp DB."""

    def __init__(self, db_path: Path, api_key: str = "", embeddings: bool = False,
                 rerank_top_k: int | None = None, rerank_model: str | None = None,
                 rerank_message_hits: bool | None = None,
                 aggregation_nodes: bool = False, aggregation_broad: bool = False,
                 episode_granularity: bool = False,
                 value_supersession: bool = True,
                 graph_multihop: bool = False,
                 graph_multihop_max_hops: int | None = None,
                 graph_multihop_decay: float | None = None,
                 graph_multihop_min_score: float | None = None,
                 rules_enabled: bool | None = None,
                 rules_extraction: bool | None = None,
                 facts_enabled: bool | None = None,
                 facts_extraction: bool | None = None):
        self.db_path = db_path
        self.api_key = api_key
        self.embeddings = embeddings
        self.rerank_top_k = rerank_top_k
        self.rerank_model = rerank_model
        self.rerank_message_hits = rerank_message_hits
        self.aggregation_nodes = aggregation_nodes
        self.aggregation_broad = aggregation_broad
        self.episode_granularity = episode_granularity
        self.value_supersession = value_supersession
        self.graph_multihop = graph_multihop
        self.graph_multihop_max_hops = graph_multihop_max_hops
        self.graph_multihop_decay = graph_multihop_decay
        self.graph_multihop_min_score = graph_multihop_min_score
        self.rules_enabled = rules_enabled
        self.rules_extraction = rules_extraction
        self.facts_enabled = facts_enabled
        self.facts_extraction = facts_extraction
        self.hy = None

    def open(self):
        from hymem import HyMem, HyMemConfig
        from hymem.contrib.openai_client import OpenAICompatibleClient

        # L2 ranking levers (None = keep the config default). rerank_top_k widens the
        # candidate pool the message/chunk reranker sees — a gold turn below this BM25
        # rank can't be lifted because it's never a candidate. rerank_model swaps the
        # LLM reranker for a local cross-encoder. rerank_message_hits=False restores raw
        # BM25 order on the dominant message tier (L2c): the gold-rank probe showed 92%
        # of MS gold already sits at BM25 rank ≤15, so the LLM reranker is demoting gold
        # it already sees — this toggle measures whether turning it OFF beats it.
        overrides = {}
        if self.rerank_top_k is not None:
            overrides["rerank_top_k"] = self.rerank_top_k
        if self.rerank_model is not None:
            overrides["rerank_model"] = self.rerank_model
        if self.rerank_message_hits is not None:
            overrides["rerank_message_hits"] = self.rerank_message_hits
        # RAPTOR A/B levers. --aggregation-nodes enables the layer (dream builds
        # nodes, the TR-gated tier fires per cfg.aggregation_inject_abilities);
        # --aggregation-broad additionally clears the ability allowlist, which
        # reproduces the broad-injection G4 run that lost 69.0 vs 70.0.
        #
        # PINNED BOTH WAYS, and that is the fix, not the style. This adapter set
        # only the True leg until 2026-08-31, so when the config default flipped
        # False -> True on 2026-08-26 (G-FLIP PASS) every run here silently
        # gained the aggregation layer + digest while still being compared to a
        # 68.4 baseline that ran without it. beam_adapter and msc_adapter were
        # pinned that day and this one was missed; the record read "the three
        # benchmark adapters now pin it False", so the gap was invisible to the
        # docs as well as to the code. A conditional override inherits whatever
        # the library decides later, which is exactly what a benchmark must not
        # do -- moving onto a new shipped default is a pre-registered scored
        # decision, never a side effect.
        overrides["aggregation_nodes_enabled"] = self.aggregation_nodes
        if self.aggregation_broad:
            overrides["aggregation_inject_abilities"] = ()
        # Plan C lever: episode granularity (decision-level episodes instead of
        # one blob segment per session). WRITE-side only -- it changes what the
        # dream extracts, so it does nothing at all against an existing store:
        # the guard arm needs a --fresh rebuild, and a run that reuses a store
        # built under the other prompt measures the store, not the lever. Pinned
        # both ways for the same reason as the line above: the flip is under
        # consideration, and the day it lands this adapter must not move with
        # it silently.
        overrides["episode_granularity_enabled"] = self.episode_granularity
        # Bi-temporal KU lever: dream-cycle single-assertion value supersession.
        # Pinned explicitly BOTH ways so a run is reproducible whatever the
        # library default: ON since 2026-07-02 (guard cleared — score-neutral,
        # zero false positives); --no-value-supersession restores the historical
        # flag-off control arm (the pre-flip canonical baselines, e.g.
        # full-dream 70.0, ran off).
        overrides["value_supersession_enabled"] = self.value_supersession
        # Track A / Idea A: query-time multi-hop graph traversal (Source 4 of
        # _graph_lookup). Default OFF; --graph-multihop enables it for the G-A2
        # non-regression guard. The three knob overrides are the swept Pareto
        # point from the recall probe (benchmarks/multihop_probe.py); when None
        # the config defaults (max_hops=2, decay=0.5, min_score=0.05) win.
        if self.graph_multihop:
            overrides["graph_multihop_enabled"] = True
            if self.graph_multihop_max_hops is not None:
                overrides["graph_multihop_max_hops"] = self.graph_multihop_max_hops
            if self.graph_multihop_decay is not None:
                overrides["graph_multihop_decay"] = self.graph_multihop_decay
            if self.graph_multihop_min_score is not None:
                overrides["graph_multihop_min_score"] = self.graph_multihop_min_score
        # Idea B rules tier. READ side (rules_enabled) is inert on LME — the
        # harness never calls add_rule() and there are no rule-obedience
        # questions — so --no-rules vs default is a flat non-regression control.
        # WRITE side (--rules-extraction) is the only lever that changes the LME
        # answer path: it routes dream markers into agent_inferred rules that then
        # inject into every ask(). None = keep the config default (rules on,
        # extraction off).
        if self.rules_enabled is not None:
            overrides["rules_enabled"] = self.rules_enabled
        if self.rules_extraction is not None:
            overrides["rules_extraction_enabled"] = self.rules_extraction
        # Campaign E / E1 narrative facts (schema v26). BOTH sides ship ON, so
        # the control arm is the explicit OFF: `--no-facts` clears the READ side
        # (the tier renders nothing) and `--no-facts-extraction` additionally
        # stops the dream from spending a call per session tail. Read-side-off
        # against the SAME store is the paired A/B — the write side needs a
        # `--fresh` rebuild to change, so don't mix the two in one comparison.
        if self.facts_enabled is not None:
            overrides["facts_enabled"] = self.facts_enabled
        if self.facts_extraction is not None:
            overrides["facts_extraction_enabled"] = self.facts_extraction
        cfg = HyMemConfig(
            root=self.db_path.parent,
            message_fts_top_k=15,
            fts_top_k=10,
            graph_top_k=10,
            **overrides,
        )
        llm = OpenAICompatibleClient(
            api_key=self.api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )
        # Optional semantic-recall A/B (lever L1). Drives the SAME local FastEmbed
        # server Hermes uses in production. Pass the local defaults explicitly so
        # --embeddings works with ZERO env setup (the client's own defaults point at
        # DeepSeek, which has no embeddings API — a dead end); HYMEM_EMBEDDING_* still
        # overrides every field for anyone pointing at a different server. Off by
        # default: the headline baseline is lexical-only (a paired comparison).
        embedding_client = None
        if self.embeddings:
            from hymem.contrib.openai_embedding_client import (
                OpenAICompatibleEmbeddingClient,
            )

            env = os.environ.get
            embedding_client = OpenAICompatibleEmbeddingClient(
                api_key=(env("HYMEM_EMBEDDING_API_KEY")
                         or env("HYMEM_LLM_API_KEY") or LOCAL_EMBED_API_KEY),
                base_url=env("HYMEM_EMBEDDING_BASE_URL") or LOCAL_EMBED_BASE_URL,
                model=env("HYMEM_EMBEDDING_MODEL") or LOCAL_EMBED_MODEL,
                dim=int(env("HYMEM_EMBEDDING_DIM") or LOCAL_EMBED_DIM),
            )
        self.hy = HyMem(cfg, llm=llm, embedding_client=embedding_client)
        return self

    def close(self):
        if self.hy:
            self.hy.close()
            self.hy = None

    def ingest_sessions(self, sessions: list[list[dict]], session_ids: list[str],
                         session_dates: list[str] | None = None) -> dict:
        """Ingest all sessions for a question. Each session is a list of messages.
        
        If session_dates is provided (one ISO-8601 date per session), each message
        gets that session's date as its created_at, giving HyMem real event times
        instead of wall-clock clustering."""
        total_msgs = 0
        total_chars = 0
        dates = session_dates or []
        for idx, (sess_id, messages) in enumerate(zip(session_ids, sessions)):
            session_date = _normalize_date(dates[idx]) if idx < len(dates) else None
            entries = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content.strip():
                    entries.append((role, content, session_date))
                    total_msgs += 1
                    total_chars += len(content)
            if entries:
                chunk_size = 50
                for i in range(0, len(entries), chunk_size):
                    chunk = entries[i : i + chunk_size]
                    self.hy.log_messages(f"{sess_id}_{i//chunk_size}", chunk)
        return {"sessions": len(sessions), "messages": total_msgs, "chars": total_chars}

    def dream_and_wait(self, timeout: int = 300):
        """Run dream cycle and wait for completion."""
        start = time.time()
        dream_hy = self.hy.fork()
        try:
            dream_hy.dream()
        finally:
            dream_hy.close()
        elapsed = time.time() - start
        print(f"      Dream completed in {elapsed:.0f}s", flush=True)

    def search(self, query: str, ability: str = None, top_k: int = 10,
               graph_facts_first: bool = False):
        """Search HyMem for the given query.

        Returns (memories, total_matches, graph_count, temporal_events,
        aggregation_nodes, narrative_facts, pool) where `pool` is the FULL
        pre-truncation candidate text by tier ({"message": [...], "fts": [...]})
        — used for recall-ceiling analysis, so a category's misses can be split
        into retrieval loss vs ranking loss. `aggregation_nodes` (RAPTOR tier,
        TR-gated by default) and `narrative_facts` (E1 tier, schema v26) are
        returned SEPARATELY from `memories` on purpose: the G4 A/B showed that
        letting a summary tier compete for memories[:top_k] slots crowds gold
        message hits out of the answer pool (KU −9.0pp). Both render as their
        own bracketed context block instead.
        """
        try:
            result = self.hy.augment(query, ability=ability)
        except Exception as e:
            print(f"    [DEBUG] augment error: {e}", flush=True)
            return [], 0, None, [], [], [], {"message": [], "fts": []}

        # Collect all sources
        graph_facts = []
        for fact in (getattr(result, "graph_facts", None) or []):
            graph_facts.append({
                "content": f"{fact.subject} {fact.predicate} {fact.object}",
                "type": "graph_fact",
                "confidence": getattr(fact, "confidence", 0.5),
            })

        message_hits = []
        for hit in (getattr(result, "message_hits", None) or []):
            text = getattr(hit, "text", "")[:600]
            role = getattr(hit, "role", "unknown")
            if text.strip():
                message_hits.append({
                    "content": f"[{role}] {text}",
                    "type": "message_hit",
                    "confidence": 0.7,
                    # created_at carried through so the answer context can date-stamp
                    # each turn — the signal the value-aware recency clause relies on.
                    "created_at": getattr(hit, "created_at", "") or "",
                })

        fts_hits = []
        for hit in (getattr(result, "fts_hits", None) or []):
            text = getattr(hit, "text", "")[:600]
            if text.strip():
                fts_hits.append({
                    "content": text,
                    "type": "fts_hit",
                    "confidence": 0.6,
                })

        procedure_hits = []
        for proc in (getattr(result, "procedures", None) or []):
            name = getattr(proc, "name", "")
            desc = getattr(proc, "description", "")[:400]
            content = f"Procedure: {name}: {desc}" if name else desc
            if content.strip():
                procedure_hits.append({
                    "content": content[:600],
                    "type": "procedure",
                    "confidence": 0.75,
                })

        # RAPTOR aggregation nodes (empty unless the layer is enabled AND the
        # ability passed the inject gate — TR-only by default). Kept OUT of the
        # `memories` pool: they go to answer_question as a separate block.
        aggregation_nodes = []
        for node in (getattr(result, "aggregation_nodes", None) or []):
            title = getattr(node, "title", "")
            summary = getattr(node, "summary", "")[:600]
            content = f"{title}: {summary}" if title else summary
            if content.strip():
                aggregation_nodes.append(content)

        # E1 narrative facts (schema v26). Like aggregation_nodes: collected
        # here but kept OUT of the `memories` pool so the tier can never take a
        # memories[:top_k] slot or spend context budget ahead of the raw turns.
        # Dates ride along — a dated fact is the whole point of the tier for
        # the recency-sensitive abilities.
        narrative_facts = []
        for nf in (getattr(result, "facts", None) or []):
            text = getattr(nf, "text", "")[:600]
            if text.strip():
                date = getattr(nf, "fact_date", None) or ""
                narrative_facts.append(f"[{date}] {text}" if date else text)

        episode_hits = []
        for ep in (getattr(result, "episodes", None) or []):
            title = getattr(ep, "title", "")
            summary = getattr(ep, "summary", "")[:500]
            content = f"{title}: {summary}" if title else summary
            if content.strip():
                episode_hits.append({
                    "content": content,
                    "type": "episode",
                    "confidence": 0.8,
                })

        # ── Ordering: task-recall vs knowledge ───────────────────────
        # MR, TR, EO, SUM, IF: need raw messages + procedures first.
        # Graph facts and episodes are mostly cross-session noise for
        # these abilities — the answer is in the message text.
        TASK_RECALL = {"IF", "MR", "EO", "SUM", "TR"}

        # DEFAULT = message-first for EVERY ability: raw answer-bearing turns lead,
        # dream-derived graph_facts demoted to a confidence-ranked tail. This is the
        # production-realistic shape — detect_ability returns None for IE/KU/PF/SS-user,
        # and routing None to graph-facts-first is exactly what caused the −14.3pp
        # SS-user regression. The full-dream "harm" was 100% this ordering artifact:
        # --message-first WITH full dream tied no-dream at 65.0% and recovered SS-user
        # +11.5pp (see project_beam_retrieval memory). No category is proven to prefer
        # graph-facts-first (the apparent multi-session win was a phantom — MS→MR is
        # already TASK_RECALL, so it never took the graph-facts-first branch).
        # --graph-facts-first restores the legacy ordering for the non-task-recall
        # (IE/KU/PF) lookups, for A/B comparison only.
        if graph_facts_first and ability not in TASK_RECALL:
            # Legacy: graph facts first, then message hits (knowledge/preference).
            graph_facts.sort(key=lambda m: -m.get("confidence", 0))
            memories = graph_facts + message_hits + episode_hits + fts_hits + procedure_hits
            memories.sort(key=lambda m: (
                m["type"] != "graph_fact",
                0 if m["type"] == "message_hit" else 1,
                -m.get("confidence", 0),
            ))
        else:
            memories = message_hits + procedure_hits
            rest = episode_hits + fts_hits + graph_facts
            rest.sort(key=lambda m: -m.get("confidence", 0))
            memories += rest

        # The recall-ceiling pool is the FULL retrieved set per tier, captured
        # before the memories[:top_k] cut, so we measure whether the gold turn
        # was retrievable at all — independent of the final ordering/truncation.
        pool = {
            "message": [m["content"] for m in message_hits],
            "fts": [m["content"] for m in fts_hits],
        }
        return (memories[:top_k], getattr(result, 'total_message_matches', 0),
                getattr(result, 'graph_count', None),
                getattr(result, 'temporal_events', []), aggregation_nodes,
                narrative_facts, pool)


# ── Answer & Judge ──────────────────────────────────────────────────

# ── P1 read-side synthesis: question-conditioned fact distillation ──
# A bounded single-step approximation of Hindsight's ≤10-iteration "reflect"
# loop: before the final answer call, map a small extraction call over each
# retrieved hit — "extract statements relevant to {question}, else NONE" — then
# answer over the distilled list PLUS the raw hits. ADDITIVE by contract: the
# distilled facts JOIN the raw memories, never replace them (the MR-filter lesson
# is an invariant). Question-conditioned + transient sidesteps the over-extraction
# risk that shelved write-time incidental extraction. Targets three banked
# buckets: the 14-floor sparse-signal misses (each turn read individually), the
# ~20 MS synthesis misses (fuse ~15 one-line facts, not 45 raw slots), and D2's
# can't-tally (tallying a short extracted list is easier).

# Versioned so a prompt change is visible in diffs/tests and an A/B can key on
# the constant, mirroring ASK_PROMPT_V1 / the fusion salts.
DISTILL_PROMPT_V1 = (
    "From the memory excerpt below, extract every statement relevant to this "
    "question, quoting concrete values, names, and dates verbatim. One line per "
    "statement. If nothing is relevant, reply exactly NONE.\n"
    "Question: {question}"
)

# V2 (G-P1a iteration): V1's "extract EVERY statement RELEVANT to..." over-extracts
# — the dry-run banked 6 flips but 6 control regressions (net-zero), 3.3 lines/Q
# kept, the distilled block crowding raw turns with on-topic-but-not-answer-bearing
# noise (the RAPTOR KU −9pp lesson). V2 tightens the RELEVANCE bar, not the line
# count: "directly answer" + "omit merely on-topic". It deliberately keeps
# "at most one line PER DISTINCT answer-bearing fact" (NOT one line total) so a
# multi-value tally turn — "drove 3h Monday, 2h Tuesday" — still yields both items;
# those multi-value turns are among the flips, and a single-value cap would
# undercount them back to wrong.
DISTILL_PROMPT_V2 = (
    "From the memory excerpt below, extract ONLY facts that directly answer the "
    "question — the specific value, name, date, or count it asks about, quoted "
    "verbatim. Omit anything merely on-topic but not answer-bearing. At most one "
    "line per distinct answer-bearing fact. If the excerpt contains no fact that "
    "directly answers the question, reply exactly NONE.\n"
    "Question: {question}"
)

DISTILL_PROMPTS = {"v1": DISTILL_PROMPT_V1, "v2": DISTILL_PROMPT_V2}
# The active default. Bumped to v2 after the V1 G-P1a FAIL (net-zero on
# regressions); v1 stays selectable via --distill-prompt-version for repro.
DEFAULT_DISTILL_PROMPT_VERSION = "v2"

# Distillation reads raw turn / chunk / episode text; graph_facts are already
# atomic subject-predicate-object triples, so distilling them is redundant.
DISTILLABLE_TYPES = frozenset({"message_hit", "fts_hit", "episode"})

# Abilities that fire distillation unconditionally (count/synthesis-heavy). The
# gate is a COST control, not a quality filter — additive either way.
DISTILL_ABILITIES = frozenset({"MR", "TR"})

# Hard cap on the map fan-out per question so a wide retrieval can't explode the
# distill call budget. Mirrors ask_distill_max_calls in the productization path.
DISTILL_MAX_CALLS = 24


def _distill_hit(llm: LLMClient, question: str, excerpt: str,
                 *, prompt_version: str = DEFAULT_DISTILL_PROMPT_VERSION) -> list[str]:
    """One extraction call over a single rendered hit. Returns kept statement
    lines; an explicit NONE (or an LLM error) yields []. Label-free by
    construction: reads only the question + hit text, never a question_type or
    gold mark."""
    template = DISTILL_PROMPTS[prompt_version]
    resp = llm.chat(
        [{"role": "system", "content": template.format(question=question)},
         {"role": "user", "content": f"Memory excerpt:\n{excerpt}"}],
        temperature=0.0, max_tokens=256,
    )
    stripped = (resp or "").strip()
    if not stripped or stripped.upper() == "NONE" or is_llm_error(stripped):
        return []
    lines = []
    for ln in stripped.splitlines():
        s = ln.strip().lstrip("-•*").strip()
        if s and s.upper() != "NONE":
            lines.append(s)
    return lines


def distill_memories(llm: LLMClient, question: str, memories: list[dict],
                     *, max_calls: int = DISTILL_MAX_CALLS,
                     prompt_version: str = DEFAULT_DISTILL_PROMPT_VERSION) -> tuple[list[str], int]:
    """Map DISTILL_PROMPT over the distillable hits in render order, capped at
    `max_calls`. Returns (kept_lines, calls_made). The caller renders these ABOVE
    the raw memories, never in place of them."""
    kept: list[str] = []
    calls = 0
    for m in memories:
        if calls >= max_calls:
            break
        if m.get("type") not in DISTILLABLE_TYPES:
            continue
        calls += 1
        kept.extend(_distill_hit(llm, question, m["content"], prompt_version=prompt_version))
    return kept, calls


def distill_should_fire(ability: str | None, memories: list[dict]) -> bool:
    """COST gate (label-free): fire on the count/synthesis-heavy abilities, or
    when the retrieval is wide enough (≥12 hits) that fusing a short extracted
    list beats reading many raw slots. Otherwise the question runs untouched."""
    return ability in DISTILL_ABILITIES or len(memories) >= 12


def _render_answer_context(memories: list[dict], ability: str | None,
                           total_matches: int, graph_count,
                           temporal_events: list | None,
                           aggregation_nodes: list | None,
                           distilled: list[str] | None = None,
                           narrative_facts: list[str] | None = None) -> str:
    """Build the CONTEXT block the answerer sees from the retrieved tiers.

    Extracted from answer_question so (a) the distillation dry-run can re-render
    the IDENTICAL context (same char caps) to decide the deep-lexical split, and
    (b) the additive [DISTILLED EVIDENCE] block has one canonical insertion point
    — directly ABOVE the raw memories, mirroring how aggregation_nodes render as
    a separate non-competing block (never consuming a raw-turn slot or budget
    ahead of the turns; the crowding that cost KU −9.0pp once already)."""
    # MR and TR questions span many sessions — expand context window
    context_limit = MAX_CONTEXT_CHARS * 2 if ability in ("MR", "TR") else MAX_CONTEXT_CHARS

    # MR counting: prefer graph_count (EXACT graph-native count) over
    # total_matches (keyword candidate). When graph_count is present it is
    # the dedup-correct answer — trust it.
    parts: list[str] = []
    if ability == "MR" and graph_count is not None:
        count = graph_count.count
        counted = getattr(graph_count, 'counted', 'items')
        parts = [f"[HyMem graph-native count: {count} distinct {counted} "
                  f"(exact COUNT(DISTINCT) over knowledge graph edges). "
                  f"Use this as the answer. Verify against evidence below.]\n"]
    elif ability == "MR" and total_matches > 0:
        parts = [f"[HyMem counted {total_matches} distinct user messages "
                  f"matching your question (assistant echoes excluded, "
                  f"restatements deduped). Verify this count against the "
                  f"evidence below and return the final number.]\n"]

    # TR: inject temporal events as a date-ordered chronology
    if ability == "TR" and temporal_events:
        parts.append("[TEMPORAL CHRONOLOGY — events in date order. A 'discussed' "
                     "line is the date the turn was logged (when-discussed), not "
                     "necessarily when the event happened:]\n")
        for ev in temporal_events:
            date = getattr(ev, 'date', '')
            desc = getattr(ev, 'text', str(ev))
            marker = " (discussed)" if getattr(ev, 'source', '') == "session-date" else ""
            parts.append(f"  {date}{marker}: {desc}\n")
        parts.append("[END CHRONOLOGY]\n\n")

    # RAPTOR aggregation nodes render as their own block, NOT as memories: they
    # never consume a memories[:top_k] slot or context-budget chars ahead of raw
    # turns — the crowding that cost KU −9.0pp in the broad-injection A/B.
    # Empty unless the layer is enabled and the ability passed the inject gate
    # (TR-only by default).
    if aggregation_nodes:
        parts.append("[CROSS-SESSION SUMMARIES — each fuses related episodes "
                     "from multiple sessions; verify details against the "
                     "memories below:]\n")
        for node in aggregation_nodes:
            parts.append(f"  {node}\n")
        parts.append("[END SUMMARIES]\n\n")

    # E1 narrative facts: dream-extracted self-contained statements, rendered as
    # their own block for the same reason as the summaries above — never a
    # memories[:top_k] slot, never budget ahead of the raw turns. The
    # verify-against-source framing is deliberate and matches the tier's own
    # contract in ask(): facts LEAD the evidence but the raw turns stay below as
    # the check (the Acme lesson — a summary is never the only copy).
    if narrative_facts:
        parts.append("[NARRATIVE FACTS — self-contained statements extracted "
                     "from past sessions; a leading date is when the fact "
                     "happened. Verify details against the memories below:]\n")
        for nf in narrative_facts:
            parts.append(f"  • {nf}\n")
        parts.append("[END NARRATIVE FACTS]\n\n")

    # Distilled evidence (P1): question-conditioned facts extracted per-turn,
    # rendered ABOVE the raw memories as their own non-competing block. Additive
    # — the raw turns stay below, unchanged. Verify-against-source framing so the
    # answerer treats it as a lens, not a replacement.
    if distilled:
        parts.append("[DISTILLED EVIDENCE — extracted per-turn, verify against "
                     "the memories below:]\n")
        for line in distilled:
            parts.append(f"  • {line}\n")
        parts.append("[END DISTILLED EVIDENCE]\n\n")

    total_chars = 0
    for m in memories:
        content = m["content"]
        if total_chars + len(content) > context_limit:
            break
        # Date-stamp raw turns (the only tier carrying created_at) so the recency-
        # conflict clause can prefer the newest value-bearing statement. FACT/fts/
        # episode tiers stay undated (graph dating deferred — see KU analysis).
        if m["type"] == "graph_fact":
            tag = "[FACT]"
        else:
            date10 = (m.get("created_at") or "")[:10]
            tag = f"[MEM {date10}]" if (m["type"] == "message_hit" and date10) else "[MEM]"
        parts.append(f"{tag} {content}")
        total_chars += len(content) + len(tag) + 2

    return "\n".join(parts) if parts else "No relevant memories found."


def context_sha(messages: list[dict]) -> str:
    """A fingerprint of exactly what the reader was handed.

    `churn_decompose` had to attribute run-to-run answer churn using the COUNT
    fields the artifact records (n_episodes, n_facts, ability_used, ...), which
    cannot tell "the same 15 episodes" from "15 different episodes". That made
    its retrieval/decoder split a pair of bounds rather than a partition, and
    the same lower-bound caveat sits on `guard_score.fired_subset`.

    Hashing the rendered prompt closes the gap for every future run at zero
    cost: two runs whose context_sha AGREE handed the reader byte-identical
    input, so any difference in the answer is the provider's decoder and
    nothing of ours. It is 64 hex chars per question and no extra call.

    The system prompt is included deliberately -- the MR branch swaps both the
    prompt and the memory list, and a fingerprint that missed that would call
    two different reader configurations identical.

    Fields are LENGTH-PREFIXED, not separated by a delimiter. A delimiter is
    only injective while no field contains it, and retrieved turns are
    arbitrary text: under `role + NUL + content + NUL`, the two-message list
    [("a","b"), ("c","d")] and the one-message list [("a\x00b","c\x00d")]
    hash identically. Two different reader inputs that collide is precisely
    the failure this fingerprint exists to make impossible."""
    h = hashlib.sha256()
    for m in messages:
        for field in (m.get("role", ""), m.get("content", "")):
            raw = field.encode()
            h.update(str(len(raw)).encode())
            h.update(b":")
            h.update(raw)
    return h.hexdigest()


def answer_question(*args, **kwargs) -> str:
    """`answer_question_raw`, dropping the context fingerprint.

    Byte-identical in behaviour and signature to the function three other
    adapters (beam, locomo, msc) already import, which is why the split is
    shaped this way rather than by changing every caller -- the same pattern
    `judge_answer` uses over `judge_answer_raw`, for the same reason."""
    return answer_question_raw(*args, **kwargs)[0]


class CountingLLM:
    """Counts what retrieval actually spends, rather than asserting it is free.

    `--retrieval-only` skips the reader and the judge, which is where the
    tokens are. It does NOT skip distillation, which fires inside the
    retrieval path on MR/TR and wide retrievals, and it cannot vouch for what
    reranking does inside the store. So the mode measures its own cost and
    puts it in the artifact: a pre-flight estimate nobody checked is how a
    5.5-hour run came to be described as cheap."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.prompt_chars = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        self.prompt_chars += sum(len(m.get("content", "")) for m in messages)
        return self._inner.chat(messages, temperature=temperature,
                                max_tokens=max_tokens)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class PoisonLLM:
    """Raises if the reader or judge is reached under `--retrieval-only`.

    A flag that merely skips a branch can be defeated by a later refactor
    routing round it, and the failure would be silent and expensive. This
    makes the guarantee structural: retrieval-only cannot answer a question,
    because there is nothing there to answer with."""

    # The counters the run summary reads off whatever client it was handed.
    # Present and ZERO, because that is the truth: this client made no call.
    #
    # They are attributes rather than something the summary guards with
    # getattr, because the 2026-09-03 f probe died on exactly that -- one call
    # site was guarded, a second was not, and the crash landed AFTER 50
    # questions had been dreamed and BEFORE the artifact was written. A
    # stand-in that is not a drop-in just relocates the failure.
    call_count = 0
    total_tokens = 0

    def __init__(self, which: str):
        self._which = which

    def chat(self, *a, **kw):
        raise AssertionError(
            f"--retrieval-only reached the {self._which} path; the mode "
            f"exists precisely so that call is never made")


def build_answer_messages(memories: list[dict], question: str,
                          ability: str = None, total_matches: int = 0,
                          graph_count=None, temporal_events: list | None = None,
                          aggregation_nodes: list | None = None,
                          question_date: str = "",
                          permissive_default: bool = False,
                          distilled: list[str] | None = None,
                          extra_system: str | None = None,
                          narrative_facts: list[str] | None = None) -> list[dict]:
    """The reader's prompt, built and not sent.

    Split out of `answer_question_raw` so `--retrieval-only` can fingerprint
    what the reader WOULD have been handed without paying for the answer.
    That mode exists to measure `f` -- the fraction of questions a lever
    actually moves -- which sets what a subset-scored gate 4 could resolve
    (`benchmarks/concentration_model.py`), and `f` is a property of retrieval
    alone.

    The split is load-bearing in one specific way: a retrieval-only run and a
    full run must produce the SAME `context_sha` for the same retrieval, or
    the cheap measurement does not describe the expensive one. Re-deriving the
    prompt in a second place is exactly how that drifts, so there is only one
    place."""
    return _answer_messages(
        memories, question, ability, total_matches, graph_count,
        temporal_events, aggregation_nodes, question_date, permissive_default,
        distilled, extra_system, narrative_facts)


def answer_question_raw(llm: LLMClient, memories: list[dict], question: str,
                        *args, **kwargs) -> tuple[str, str]:
    """`answer_question`, plus the context fingerprint it would otherwise drop.

    Prompt construction lives in `build_answer_messages` and happens in ONE
    place, so a `--retrieval-only` run fingerprints exactly the prompt a full
    run would have sent."""
    messages = build_answer_messages(memories, question, *args, **kwargs)
    return llm.chat(messages, temperature=0.0, max_tokens=1024), \
        context_sha(messages)


def _answer_messages(memories: list[dict], question: str,
                     ability: str = None, total_matches: int = 0,
                     graph_count=None, temporal_events: list | None = None,
                     aggregation_nodes: list | None = None,
                     question_date: str = "", permissive_default: bool = False,
                     distilled: list[str] | None = None,
                     extra_system: str | None = None,
                     narrative_facts: list[str] | None = None) -> list[dict]:
    """Render the reader's prompt. The only place it is built.

    Uses ability-aware prompts and expanded context for multi-session
    and temporal reasoning questions that need more cross-session data.
    For MR questions, prefers graph_count (exact graph-native count) over
    total_matches (keyword candidate). For TR questions, injects
    temporal_events as a date-ordered chronology. `aggregation_nodes` (RAPTOR
    cross-session summaries, TR-gated upstream) render as a separate block so
    they never compete with raw turns for top_k slots or context budget.

    `question_date` is the "now" the question is asked at — the reference point
    relative-date questions ("how many days ago?", "a month ago") subtract from.
    Without it the chronology gives event dates but the model has no anchor to
    compute an interval against, and answers "current date not provided".
    """
    # Render the CONTEXT block from the retrieved tiers (+ the additive distilled
    # block, above the raw turns). Shared with the distillation dry-run so both
    # see identical char caps and insertion order.
    context = _render_answer_context(
        memories, ability, total_matches, graph_count,
        temporal_events, aggregation_nodes, distilled=distilled,
        narrative_facts=narrative_facts)

    # The default (unknown-ability) prompt. When --permissive-default is set this
    # is the permissive preference-style prompt (D4 fix) instead of the strict
    # "only provided context" one — so a label-free SS-P question (router emits
    # None → this branch) can bridge to general knowledge instead of refusing.
    default_prompt = ANSWERING_PERMISSIVE_PROMPT if permissive_default else ANSWERING_SYSTEM_PROMPT

    # Preference questions need generation + personalization, not fact extraction
    # MR/TR questions need counting + temporal reasoning prompts + more context
    if ability == "PF":
        system_prompt = ANSWERING_PREFERENCE_PROMPT
    elif ability == "MR":
        system_prompt = ANSWERING_MR_PROMPT
        # Filter to user-only messages for MR counting questions.
        # Assistant responses are mostly noise for "how many" questions —
        # the answer-bearing information is always in user messages.
        memories = [m for m in memories if m["type"] == "message_hit" and "[user]" in m.get("content", "")]
        if not memories:
            system_prompt = default_prompt  # fallback if no user messages
    elif ability == "TR":
        system_prompt = ANSWERING_TR_PROMPT
    else:
        system_prompt = default_prompt

    # Benchmark-specific system-prompt suffix (e.g. the MSC perspective clause).
    # Additive and None by default, so every LME posture is byte-identical.
    if extra_system:
        system_prompt = system_prompt + extra_system

    # The reference "now" for relative-date math. Stated explicitly so the model
    # can subtract event dates from it ("how many days ago", "a month ago")
    # instead of complaining the current date is unknown.
    today_line = f"Today's date is {question_date}.\n\n" if question_date else ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{today_line}CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"},
    ]

    return messages


def get_judge_prompt(question_type: str, question: str, answer: str, response: str) -> str:
    """Get the appropriate judge prompt per question type (from original LongMemEval evaluate_qa.py)."""
    is_abstention = "_abs" in question_type
    base_type = question_type.replace("_abs", "")

    if is_abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            f"Question: {question}\n\n"
            f"Explanation: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )

    if base_type in ("single-session-user", "single-session-assistant", "multi-session"):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. "
            "If the response only contains a subset of the information required by the answer, answer no.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if base_type == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. "
            "If the response only contains a subset of the information required by the answer, answer no. "
            "In addition, do not penalize off-by-one errors for the number of days. "
            "If the question asks for the number of days/weeks/months, etc., and the model makes "
            "off-by-one errors (e.g., predicting 19 days when the answer is 18), "
            "the model's response is still correct.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if base_type == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the required answer.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if base_type == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, "
            "and a response from a model. Please answer yes if the response satisfies the desired response. "
            "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
            f"Question: {question}\n\n"
            f"Rubric: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    raise NotImplementedError(f"Unknown question type: {question_type}")


# ── Judge verdict parsing ───────────────────────────────────────────
# Landed 2026-08-25 to replace `"yes" in raw.lower()`, an UNANCHORED substring
# test that scored "yesterday" and "eyes" as CORRECT, never consulted the "no"
# half of a reply, and inverted a negated affirmative.
#
# WHY THE CHANGE IS SAFE TO LAND WITHOUT RE-BASELINING. `benchmarks/judge_audit.py`
# recorded all 500 raw judge replies of the 2026-08-25 LME run (the first run to
# record `raw` at all) and measured C2 non-compliance = 0.00%: every reply was a
# bare yes/no. On that evidence the rules below are a PROVEN no-op over the only
# corpus of real judge replies in existence — verify it with
# `judge_audit.py --verify-parse <spend.json>`, which must print 0 flips.
#
# That is the entire argument for landing it NOW. deepseek-chat's deprecation
# already forced one judge migration (1.6pp of judge harshness); the next verbose
# judge would otherwise change the decision rule and the data in the same step,
# with no way to separate the two. Anchoring while it is provably inert buys the
# insurance at zero cost, and the cost is only zero once.
_YES_WORD = re.compile(r"\byes\b")
_NO_WORD = re.compile(r"\bno\b")
# A NEGATED affirmative ("not yes", "never a yes"), which the substring rule
# scored CORRECT because it reads the token and never the polarity.
#
# DELIBERATELY TIGHTER than `judge_audit._NEGATED_YES`, and the asymmetry is the
# point: the audit's regex is a COUNTER, where over-matching inflates a bucket
# that is already reported as a lower bound and is therefore conservative. This
# is a DECISION rule, where over-matching silently marks a correct answer wrong.
# The audit's `[^.]{0,20}?` window would fire on "it is not incorrect, yes" —
# a judge saying yes — so the negation here must sit adjacent to the token.
_NEGATED_YES = re.compile(r"\b(?:not|never|isn'?t|wasn'?t|aren'?t|ain'?t)\s+"
                          r"(?:really\s+|quite\s+|exactly\s+|an?\s+)?yes\b")


def parse_judge_verdict(raw: str) -> bool:
    """Score one raw judge reply. First word-boundary verdict token wins.

    Extracted from `judge_answer` as a pure function so it can be tested without
    a client and diffed against the frozen legacy rule over stored replies.

    Three rules, in order:
      1. An `[LLM_ERROR: ...]` sentinel is never a verdict. It already scored
         `False` by luck (no "yes" substring); it now scores `False` by
         construction, so an outage message that happens to contain the word
         cannot be read as the judge saying the answer was correct. Visibility
         is no longer this function's problem: `judge_scored` (D3, 2026-08-26)
         reports the sentinel as `correct=None` — UNSCORED, not wrong — and
         every call site across the three adapters goes through it. This rule
         stays because the two are independent: a sentinel must fail closed
         even where a caller ignores the channel.
      2. A negated affirmative scores `False`.
      3. Otherwise the FIRST of `\byes\b` / `\bno\b` wins; no verdict token at
         all scores `False`, the same fail-closed direction as the empty reply.

    Known and deliberately unfixed, both pinned by tests so neither can drift:

      * A truncated non-verdict that contains the bare word ("The question is
        whether a yes would be") still scores `True`. Separating that needs the
        reply's structure, not its tokens, and `max_tokens=10` is itself part of
        the frozen comparability contract.
      * "yes and no" still scores `True` — first token wins, exactly as the
        legacy rule did. Resolving a hedging judge to `False` would be a
        defensible reading of "Answer yes or no only", but it decides what a
        non-committal judge MEANT, which is a criterion question (D1, still open
        at WATCH) and not a parse question. This function fixes the parse only;
        mixing an unmeasured criterion change into a provably-inert parse change
        is precisely the coupling the inert window exists to avoid.

        And the pin protects more than that choice. `reference_verdict`, banked
        pre-run, scores "yes and no" True on first-token-wins. Flipping it here
        would break rule 3's IDENTITY with that reference — and that identity is
        the entire warrant for the paragraph below. C1 = 0.00% would stop
        certifying this function and start certifying something merely like it.

    Rule 3 IS `judge_audit.reference_verdict`, banked before the 2026-08-25 run,
    which is why that run's C1 = 0.00% certifies this function rather than
    something merely like it. Rules 1 and 2 are additive to it and were measured
    separately on the same run: 0 judge-side sentinels, and C1b = 0 negated-yes
    replies. All three are therefore inert on the recorded corpus, each on its
    own evidence.
    """
    text = raw or ""
    # Shares the CONSTANT but deliberately does not call `is_llm_error`: that
    # predicate adds a truthiness test and a str() coercion, and this function is
    # certified byte-for-byte by the 2026-08-25 run (C1 = 0.00%). Widening a
    # certified decision rule, even harmlessly, costs the "identical to what was
    # measured" claim. Every other site does route through the predicate.
    if text.startswith(LLM_ERROR_PREFIX):
        return False
    low = text.lower()
    if _NEGATED_YES.search(low):
        return False
    y, n = _YES_WORD.search(low), _NO_WORD.search(low)
    if y and n:
        return y.start() < n.start()
    return bool(y)


def judge_answer_raw(llm: LLMClient, question_type: str, question: str,
                     answer: str, ai_answer: str) -> tuple[bool, str]:
    """`judge_answer`, plus the reply it used to throw away. D3.

    The bare-bool return is why an outage was invisible: a judge that never
    answered and a judge that said "no" are the same value at the call site, so
    an outage streak silently DEFLATES the score of the arm it hits. It is also
    why `benchmarks/judge_audit.py` had to re-judge 500 rows to measure a rate
    that was already produced once and discarded.

    `judge_answer` now delegates here and drops the raw, so it is byte-identical
    in behaviour and signature. That is deliberate and load-bearing: rule 3 of
    `parse_judge_verdict` is IDENTICAL to `judge_audit.reference_verdict`, banked
    before the 2026-08-25 run, and that identity is the whole warrant for
    C1 = 0.00% certifying this function rather than something merely like it.
    Neither function's logic is touched here — only the channel around them."""
    prompt = get_judge_prompt(question_type, question, answer, ai_answer)
    messages = [{"role": "user", "content": prompt}]
    raw = llm.chat(messages, temperature=0.0, max_tokens=10)
    return parse_judge_verdict(raw), raw


def judge_answer(llm: LLMClient, question_type: str, question: str, answer: str, ai_answer: str) -> bool:
    """Judge whether the answer is correct (binary yes/no per LongMemEval protocol)."""
    return judge_answer_raw(llm, question_type, question, answer, ai_answer)[0]


def judge_scored(llm: LLMClient, question_type: str, question: str,
                 answer: str, ai_answer: str) -> tuple[bool | None, str]:
    """(correct, raw) where `correct` is None iff the JUDGE itself errored.

    The one place the D3 policy is written down, so six call sites across three
    adapters cannot each decide it differently:

      * a judge sentinel yields `correct = None` — the row is UNSCORED, not
        wrong. It drops out of the accuracy denominator (`accuracy` below) and
        is counted separately.
      * `correct = False` keeps meaning the judge read the answer and rejected
        it, which is what every canonical number was scored under.

    PROVABLY INERT WHERE IT MATTERS: the 2026-08-25 audit recorded 0 judge-side
    sentinels over 500 replies, so on a clean run this returns exactly what
    `judge_answer` returned and no canonical number moves. That is the same
    inert-window argument that licensed landing the parse fix rather than
    deferring it to the next judge migration — and the insurance is only free
    once, because at the next verbose judge the decision rule and the data would
    change in the same step with no way to separate them."""
    verdict, raw = judge_answer_raw(llm, question_type, question, answer, ai_answer)
    return (None if is_llm_error(raw) else verdict), raw


# ── Scoring over rows that may be UNSCORED ──────────────────────────

def is_scored(row: dict) -> bool:
    """A row carries a verdict. `correct is None` means no verdict exists —
    either the judge errored (`judge_error`) or no reader ran at all
    (`--diag-only`). Both are excluded from accuracy; only the first is a
    defect, and `judge_error` is what tells them apart."""
    return row.get("correct") is not None


def is_retrieval_only(artifact: dict) -> bool:
    """See `run_registry.is_retrieval_only` -- one definition, re-exported so
    the adapter and the scorers cannot drift on what counts as a run with no
    verdicts."""
    from run_registry import is_retrieval_only as _impl
    return _impl(artifact)


def scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if is_scored(r)]


def accuracy(rows: list[dict]) -> float:
    """Accuracy over SCORED rows only.

    Exists because `sum(r["correct"] for r in rows) / len(rows)` — the shape
    repeated at roughly fifteen sites across the three adapters — TypeErrors the
    moment one row is unscored, and because the fix for that must be ONE
    decision rather than fifteen slightly different guards. Returns 0.0 for an
    empty set; callers that must distinguish "scored zero" from "measured
    nothing" check `len(scored(rows))` and say so, exactly as the `--diag-only`
    branch already refuses to print 0.0% for a run that measured nothing."""
    s = scored(rows)
    return sum(1 for r in s if r["correct"]) / len(s) if s else 0.0


def judge_error_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("judge_error")]


def judge_error_note(rows: list[dict]) -> str:
    """One line for a run summary, with the vacuity split built in.

    "0 judge errors" over a run that made no judge calls is not reassurance,
    it is an instrument that never met the surface it certifies — the same
    reason `--verify-parse` reports `rows_that_could_flip`. So the denominator
    is stated whenever the count is zero."""
    errs, judged = judge_error_rows(rows), scored(rows)
    if errs:
        return (f"⚠ {len(errs)} row(s) UNSCORED — the judge returned an "
                f"[LLM_ERROR] sentinel. They are excluded from the "
                f"{len(judged)}-row accuracy denominator, NOT counted wrong. "
                f"ids: {[r.get('id') or r.get('question_id') for r in errs][:10]}")
    return f"judge errors: 0 of {len(rows)} row(s) that could have errored"

# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_question(
    llm: LLMClient,
    judge_llm: LLMClient,
    hy: HyMemAdapter,
    q_data: dict,
    top_k: int,
    auto_ability: bool = False,
    no_dream: bool = False,
    graph_facts_first: bool = False,
    permissive_default: bool = False,
    distill: bool = False,
    distill_prompt_version: str = DEFAULT_DISTILL_PROMPT_VERSION,
    retrieval_only: bool = False,
    distill_llm: LLMClient | None = None,
) -> dict:
    """Evaluate a single LongMemEval question.

    Under `retrieval_only` the reader and judge are never called: the row
    carries the context fingerprint and the retrieval counts, and `correct`
    is None. Such a row has no verdict and every scorer must refuse it -- see
    `is_retrieval_only` and the guards in guard_score / churn_decompose /
    concentration_model."""
    question_id = q_data["question_id"]
    question_type = q_data["question_type"]
    question = q_data["question"]
    answer = q_data["answer"]
    sessions = q_data.get("haystack_sessions", [])
    session_ids = q_data.get("haystack_session_ids", [str(i) for i in range(len(sessions))])
    session_dates = q_data.get("haystack_dates", [])

    # Reference "now" for relative-date math. LongMemEval ships `question_date`;
    # if the cleaned dataset stripped it, fall back to the latest session date
    # (questions are asked after all sessions, so the most recent haystack date
    # is the best available proxy for "today").
    question_date = q_data.get("question_date", "") or (
        max(session_dates) if session_dates else ""
    )

    # Ensure session_ids length matches sessions
    while len(session_ids) < len(sessions):
        session_ids.append(f"extra_{len(session_ids)}")

    # Map question type to HyMem ability. We ALWAYS record what the production
    # router (detect_ability) would infer from the raw question, so the router's
    # accuracy is measurable on every run. Only when --auto-ability is set do we
    # actually DRIVE shaping from the inferred label (the true production path);
    # otherwise the oracle question_type label drives it as before.
    oracle_ability = QUESTION_TYPE_TO_ABILITY.get(question_type, None)
    detected_ability = _detect_ability_safe(question)
    ability = detected_ability if auto_ability else oracle_ability

    # Ingest
    stats = hy.ingest_sessions(sessions, session_ids, session_dates)
    print(f"    Ingested {stats['sessions']} sessions ({stats['messages']} msgs, {stats['chars']} chars)", flush=True)

    # Dream — skipped in --no-dream fast mode. The message/rerank/MR paths under
    # test read messages_fts (populated at ingest, no dream needed); the dreamed
    # chunk tier uniquely recovers ~2/500 on LME and graph_count is None on every
    # consumer-domain question, so skipping it barely moves the score while
    # deleting the dominant cost. NOT a faithful full-system run — for relative
    # A/B iteration only; do one full-dream pass for the headline number.
    if no_dream:
        print(f"    Skipping dream (--no-dream)", flush=True)
    else:
        print(f"    Running dream cycle...", flush=True)
        hy.dream_and_wait()

    # Search
    (memories, total_matches, graph_count, temporal_events, aggregation_nodes,
     narrative_facts, pool) = hy.search(
        question, ability=ability, top_k=top_k * 3, graph_facts_first=graph_facts_first)
    src = "question_date" if q_data.get("question_date") else ("haystack_max" if session_dates else "none")

    # Recall ceiling: was the answer-bearing turn anywhere in the pre-truncation
    # pool? Splits a miss into retrieval loss (ceiling=False) vs ranking/
    # synthesis loss (ceiling=True). None when the dataset carries no gold marks.
    gold_turns, gold_mode = _extract_gold_turns(q_data)
    if gold_turns:
        in_msg = _gold_in_pool(gold_turns, pool["message"])
        in_fts = _gold_in_pool(gold_turns, pool["fts"])
        recall_ceiling = in_msg or in_fts
        recall_tier = ("both" if in_msg and in_fts
                       else "message" if in_msg
                       else "fts" if in_fts else "none")
        # Per-gold-turn fused-pool membership — de-conflates the any-match ceiling so
        # the L3 floor audit can tell phantom (chunks rescue) from real recall gap.
        gold_turn_tiers = _gold_turn_tiers(gold_turns, pool)
        gold_turns_in_pool = sum(1 for t in gold_turn_tiers if t != "none")
    else:
        recall_ceiling, recall_tier = None, "unknown"
        gold_turn_tiers, gold_turns_in_pool = [], None

    ceiling_str = ("∅" if recall_ceiling is None
                   else f"{recall_ceiling}[{recall_tier}]")
    used_marker = "←used" if auto_ability else ""
    router_str = f"{oracle_ability or '∅'}/det={detected_ability or '∅'}{used_marker}"
    print(f"    Retrieved {len(memories)} memories (total_matches={total_matches}, graph_count={graph_count is not None}, temporal_events={len(temporal_events)}, agg_nodes={len(aggregation_nodes)}, facts={len(narrative_facts)}, now={question_date or '∅'}[{src}], ceiling={ceiling_str}, ability={router_str})", flush=True)

    # P1 distillation (additive, cost-gated, label-free): map an extraction call
    # over the distillable hits, then answer over the kept lines PLUS the raw
    # turns. The gate is a COST control (fires on MR/TR or a wide retrieval), not
    # a quality filter — the raw memories are always passed through untouched.
    distilled_lines, distill_calls, distill_fired = None, 0, False
    if distill and distill_should_fire(ability, memories):
        distill_fired = True
        # Distillation is part of RETRIEVAL and still runs under
        # --retrieval-only, so it needs its own client: `llm` is a PoisonLLM
        # there, which is what makes "the reader is never called" structural
        # rather than a property of the branch below.
        distilled_lines, distill_calls = distill_memories(
            distill_llm or llm, question, memories,
            prompt_version=distill_prompt_version)
        print(f"    Distill[{distill_prompt_version}]: {distill_calls} calls → "
              f"{len(distilled_lines)} lines kept", flush=True)

    _answer_kw = dict(
        ability=ability, total_matches=total_matches,
        graph_count=graph_count, temporal_events=temporal_events,
        aggregation_nodes=aggregation_nodes,
        question_date=question_date, permissive_default=permissive_default,
        distilled=distilled_lines,
        narrative_facts=narrative_facts)
    if retrieval_only:
        # Build the prompt, fingerprint it, send nothing. `f` -- the fraction
        # of questions a lever moves -- is a property of retrieval, so it can
        # be measured without paying for 500 reader calls and 500 judge calls.
        ctx_sha = context_sha(build_answer_messages(memories, question,
                                                   **_answer_kw))
        ai_answer = ""
    else:
        ai_answer, ctx_sha = answer_question_raw(
            llm, memories, question, **_answer_kw)

    _ep_texts = [m["content"] for m in memories
                 if isinstance(m, dict) and m.get("type") == "episode"]

    # Judge (binary yes/no, or None when the JUDGE itself errored — D3)
    if retrieval_only:
        correct, judge_raw = None, ""
        print(f"    retrieval-only: ctx={ctx_sha[:12]} "
              f"episodes={len(_ep_texts)} distill_calls={distill_calls}",
              flush=True)
    else:
        correct, judge_raw = judge_scored(judge_llm, question_type, question,
                                          answer, ai_answer)
        print(f"    Correct: {correct if correct is not None else 'UNSCORED (judge error)'}"
              f" | Answer: {ai_answer[:120]}...", flush=True)

    return {
        "question_id": question_id,
        "question_type": question_type,
        # Full text (previously q[:200]/a[:200]/hyp[:500]) — the judge saw the full
        # strings, so storing them un-clipped makes a later --rejudge byte-faithful
        # (needed once a judge model is deprecated and the baseline must be re-paired).
        "question": question,
        "answer": str(answer),
        "hypothesis": ai_answer,
        "correct": correct,
        # The judge's own reply, kept for the same reason the strings above are
        # un-clipped: a discarded reply is why judge_audit had to re-judge 500
        # rows to measure a rate that had already been produced once. ~10 tokens.
        "judge_raw": judge_raw,
        # A judge that was never CALLED did not error. Under --retrieval-only
        # every row has correct=None by design, and flagging 50 "judge errors"
        # on a run that made zero judge calls is the same vacuity as "0 judge
        # errors" over a run that made none: a count whose denominator is not
        # what the reader assumes.
        "judge_error": (correct is None) and not retrieval_only,
        # What the reader was HANDED, hashed. Two runs that agree here gave
        # the reader byte-identical input, so a differing answer is the
        # provider's decoder and nothing of ours -- which is the distinction
        # `churn_decompose` can currently only bound. See `context_sha`.
        "context_sha": ctx_sha,
        # No verdict was produced. Scorers must refuse this row rather than
        # read `correct: None` as a miss -- see `is_retrieval_only`.
        "retrieval_only": bool(retrieval_only),
        "num_sessions": stats["sessions"],
        "num_messages": stats["messages"],
        "num_memories": len(memories),
        # Fired-indicators for the two tiers a lever can switch on, recorded
        # for the reason `n_facts` below already is: E1's all-800 net read NULL
        # while the FIRED subset read -2.9pp (p=0.024), so an unconditional net
        # is not evidence of no effect unless you can also show the tier reached
        # the reader. The 2026-08-31 episode-granularity guard could only produce
        # the all-500 net because nothing here counted episodes -- the E1 lesson
        # had been instrumented for E1 and not generalised. Same basis as
        # `num_memories`: the pool handed to the reader.
        "n_episodes": sum(1 for m in memories
                         if isinstance(m, dict) and m.get("type") == "episode"),
        "n_agg_nodes": len(aggregation_nodes) if aggregation_nodes else 0,
        # `n_episodes > 0` is NOT the fired-subset variable, and pre-registering
        # it as one would repeat E1's LME death exactly: episodes reach the
        # reader on nearly every question (cap 10 via the fts_top_k=10 pin at
        # :598-604, sorted to the head of `rest` at confidence 0.8), so the
        # "fired" subset is ~the whole set and its band equals the all-500 band,
        # with a not-fired control of n≈0. E1's fire rate was 99.8% and its
        # control was n=1. `gold_in_episodes` is the analogue of the instrument
        # that DID work there -- the tier's own hit rate, independent of whether
        # the answer was right. None = answer too short to test.
        "gold_in_episodes": (
            _answer_in_texts(str(answer), _ep_texts)
            if _ep_texts else False
        ),
        # Procedures share the episode budget (`proc_top_k = fts_top_k` for
        # every ability but IF, where procedure_top_k_if is also 10), so the
        # retrieved pool can reach 15+10+10+10+10 = 55 and the memories[:45] cut
        # is NOT inert by construction -- it is inert for EPISODES, which land in
        # slots 16-25 (no procedures) to 26-35 (max), never near the cut.
        # Recorded because nothing else distinguishes "45 = every tier full"
        # from "45 = truncated from 55", and a 70% pile at exactly 45 reads the
        # same either way.
        "n_procedures": sum(1 for m in memories
                            if isinstance(m, dict) and m.get("type") == "procedure"),
        "recall_ceiling": recall_ceiling,
        "recall_tier": recall_tier,
        "gold_mode": gold_mode,
        "gold_turns": len(gold_turns),
        # Floor audit: how many of N gold turns the FUSED pool actually carried, and
        # the per-turn tier ("none" entries = the unrecovered floor turns).
        "gold_turns_in_pool": gold_turns_in_pool,
        "gold_turn_tiers": gold_turn_tiers,
        "oracle_ability": oracle_ability,
        "detected_ability": detected_ability,
        "ability_used": ability,
        # P1 distillation instrumentation (fired/not per question, map fan-out,
        # lines kept) — so the A/B can tell whether the mechanism hit its named
        # targets (mechanism > score) and read distill cost per question.
        "distill_fired": distill_fired,
        "distill_calls": distill_calls,
        "distill_kept": len(distilled_lines) if distilled_lines else 0,
        # E1 mechanism instrumentation, for the pre-registered read that must
        # happen BEFORE the score: `n_facts` is whether the tier fired at all
        # (a run of zeros means the lever never reached the reader, so the
        # score is a no-op by construction, not a null result), and
        # `gold_in_facts` is whether the answer string is actually IN the
        # facts block — the tier's own hit rate, independent of the answer.
        # None = the answer is too short to test (see _answer_in_texts).
        "n_facts": len(narrative_facts),
        "gold_in_facts": (
            _answer_in_texts(str(answer), narrative_facts)
            if narrative_facts else False
        ),
    }


def compute_scores(results: list[dict]) -> dict:
    """Compute accuracy by question type, over SCORED rows only."""
    # UNSCORED rows (correct is None — the judge errored, D3) are dropped HERE,
    # once, rather than guarded at each summation below. They are reported by
    # `judge_error_note`, never counted wrong.
    results = scored(results)
    by_type = defaultdict(list)
    for r in results:
        qtype = r["question_type"]
        # Group by base type (strip _abs suffix for reporting)
        base_type = qtype.replace("_abs", "")
        by_type[base_type].append(r["correct"])

    scores = {}
    all_correct = []
    for qtype, corrects in sorted(by_type.items()):
        acc = sum(corrects) / len(corrects) if corrects else 0.0
        scores[qtype] = {"accuracy": acc, "count": len(corrects)}
        all_correct.extend(corrects)

    scores["OVERALL"] = {
        "accuracy": sum(all_correct) / len(all_correct) if all_correct else 0.0,
        "count": len(all_correct),
    }
    return scores


def compute_abstention_scores(results: list[dict]) -> dict:
    """Split accuracy into ANSWERABLE vs ABSTENTION questions — the guard rail for
    the --permissive-default (D4) trade.

    A permissive default prompt buys back SS-P recommendation questions by letting
    the model bridge to general knowledge; the SAME license can turn a correct "I
    don't know" into a hallucinated answer on the `_abs` questions (whose gold
    answer IS abstention). compute_scores strips the `_abs` suffix and folds those
    into the base category, hiding exactly this regression. Here we keep the split:
      - ANSWERABLE: question_type without `_abs`
      - ABSTENTION: question_type ending `_abs`
    reported overall AND per base category, so a permissive run can be A/B'd
    against strict with the abstention cost made explicit. If overall goes up while
    ABSTENTION drops, the gain is partly a hallucination trade, not a clean win.
    """
    # UNSCORED rows (correct is None — the judge errored, D3) are dropped HERE,
    # once, rather than guarded at each summation below. They are reported by
    # `judge_error_note`, never counted wrong.
    results = scored(results)
    answerable: list[bool] = []
    abstention: list[bool] = []
    by_cat: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"answerable": [], "abstention": []})
    for r in results:
        qtype = r.get("question_type", "")
        is_abs = qtype.endswith("_abs")
        base = qtype[:-4] if is_abs else qtype
        bucket = "abstention" if is_abs else "answerable"
        (abstention if is_abs else answerable).append(bool(r.get("correct")))
        by_cat[base][bucket].append(bool(r.get("correct")))

    def _acc(xs: list[bool]) -> dict:
        return {"accuracy": (sum(xs) / len(xs)) if xs else None, "count": len(xs)}

    return {
        "answerable": _acc(answerable),
        "abstention": _acc(abstention),
        "by_category": {
            base: {"answerable": _acc(b["answerable"]),
                   "abstention": _acc(b["abstention"])}
            for base, b in sorted(by_cat.items())
        },
    }


def print_abstention_scores(diag: dict):
    """Render the answerable-vs-abstention split so a permissive-prompt run shows
    its abstention cost next to its answerable gain."""
    def _fmt(d: dict) -> str:
        a = d["accuracy"]
        return "  n/a " if a is None else f"{a*100:>5.1f}% ({d['count']})"

    print(f"\n  Answerable vs Abstention  (the --permissive-default trade)")
    print(f"    {'category':<28} {'answerable':>14}  {'abstention':>14}")
    print(f"    {'─'*60}")
    for base, b in diag["by_category"].items():
        print(f"    {base:<28} {_fmt(b['answerable']):>14}  {_fmt(b['abstention']):>14}")
    print(f"    {'─'*60}")
    print(f"    {'ALL':<28} {_fmt(diag['answerable']):>14}  {_fmt(diag['abstention']):>14}")
    print(f"\n    Read: a permissive default should LIFT answerable (esp. "
          f"single-session-preference)\n          without sinking abstention — if "
          f"abstention drops, it's trading refusals for hallucinations.")


def compute_recall_diagnostics(results: list[dict]) -> dict:
    """Per-category recall-ceiling stats: split misses into retrieval vs ranking.

    For each base type, over questions whose gold turns are known:
      - ceiling_rate: fraction whose answer turn entered the pre-truncation pool
      - among the INCORRECT ones, how many were retrieval losses (gold never
        retrieved) vs ranking/synthesis losses (gold retrieved but answer wrong)
    The miss split is the actionable signal: retrieval-dominant → embeddings/
    chunking; ranking-dominant → rerank/budget/packing.
    """
    # UNSCORED rows (correct is None — the judge errored, D3) are dropped HERE,
    # once, rather than guarded at each summation below. They are reported by
    # `judge_error_note`, never counted wrong.
    results = scored(results)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"].replace("_abs", "")].append(r)

    tiers = Counter()
    modes = Counter()
    diag: dict[str, dict] = {}
    for qtype, rows in sorted(by_type.items()):
        known = [r for r in rows if r.get("recall_ceiling") is not None]
        hit = [r for r in known if r["recall_ceiling"]]
        misses = [r for r in rows if not r["correct"]]
        miss_retrieval = sum(1 for r in misses if r.get("recall_ceiling") is False)
        miss_ranking = sum(1 for r in misses if r.get("recall_ceiling") is True)
        miss_unknown = sum(1 for r in misses if r.get("recall_ceiling") is None)
        for r in known:
            tiers[r.get("recall_tier", "none")] += 1
        for r in rows:
            modes[r.get("gold_mode", "none")] += 1
        diag[qtype] = {
            "known": len(known),
            "unknown": len(rows) - len(known),
            "ceiling_rate": (len(hit) / len(known)) if known else None,
            "misses": len(misses),
            "miss_retrieval": miss_retrieval,
            "miss_ranking": miss_ranking,
            "miss_unknown": miss_unknown,
        }
    diag["_tiers"] = dict(tiers)
    diag["_gold_mode"] = dict(modes)
    return diag


def print_recall_diagnostics(diag: dict):
    """Render the retrieval-vs-ranking split so the next fix targets the right
    stage instead of guessing."""
    modes = diag.get("_gold_mode", {})
    mode_str = ", ".join(f"{k}={v}" for k, v in sorted(modes.items()))
    print(f"\n  Recall-Ceiling Diagnostics  (gold marks: {mode_str})")
    print(f"    {'category':<28} {'ceiling':>8}  {'known':>6}   misses → retrieval / ranking / unknown")
    print(f"    {'─'*82}")
    for qtype, d in sorted(diag.items()):
        if qtype.startswith("_"):
            continue
        rate = d["ceiling_rate"]
        rate_s = "  n/a " if rate is None else f"{rate*100:>5.0f}%"
        print(f"    {qtype:<28} {rate_s:>8}  {d['known']:>6}   "
              f"{d['misses']:>3}  →  {d['miss_retrieval']:>3}  /  "
              f"{d['miss_ranking']:>3}  /  {d['miss_unknown']:>3}")
    tiers = diag.get("_tiers", {})
    if tiers:
        tier_str = ", ".join(f"{k}={v}" for k, v in sorted(tiers.items()))
        print(f"    {'─'*82}")
        print(f"    recovered-by tier (known questions): {tier_str}")
    print(f"\n    Read: high retrieval-loss → recall problem (embeddings/chunking/"
          f"fan-out).\n          high ranking-loss → the turn was retrieved but "
          f"lost the cut (rerank/budget).")


def compute_router_diagnostics(results: list[dict]) -> dict:
    """How well HyMem's production detect_ability matches the oracle label.

    The oracle reduces to a shaping target of MR / TR / NONE (every non-MR/TR
    oracle ability is a category with no wired shaping, so the correct router
    verdict there is None — an abstain, not a miss). We build the MR/TR/NONE
    confusion of detected-vs-target and report per-intent recall/precision plus
    the abstain accuracy on NONE categories. This is what tells you whether an
    oracle-label MR/TR gain actually survives in label-free production.
    """
    def target(oracle: str | None) -> str:
        return oracle if oracle in ("MR", "TR") else "NONE"

    def det(d: str | None) -> str:
        return d if d in ("MR", "TR") else "NONE"

    labels = ("MR", "TR", "NONE")
    confusion = {t: Counter() for t in labels}  # confusion[target][detected]
    for r in results:
        confusion[target(r.get("oracle_ability"))][det(r.get("detected_ability"))] += 1

    per_intent = {}
    for intent in ("MR", "TR"):
        tp = confusion[intent][intent]
        actual = sum(confusion[intent].values())                       # oracle==intent
        predicted = sum(confusion[t][intent] for t in labels)          # detected==intent
        per_intent[intent] = {
            "recall": (tp / actual) if actual else None,               # caught of true
            "precision": (tp / predicted) if predicted else None,      # right of fired
            "actual": actual,
            "predicted": predicted,
            "tp": tp,
        }
    none_total = sum(confusion["NONE"].values())
    abstain_ok = confusion["NONE"]["NONE"]
    return {
        "confusion": {t: dict(confusion[t]) for t in labels},
        "per_intent": per_intent,
        "abstain_accuracy": (abstain_ok / none_total) if none_total else None,
        "false_positives": none_total - abstain_ok,  # normal Qs mis-shaped to MR/TR
        "none_total": none_total,
    }


def print_router_diagnostics(diag: dict, auto_ability: bool):
    """Render the detect_ability-vs-oracle confusion. With --auto-ability the
    inferred label DROVE retrieval; otherwise this is a free shadow measurement
    of what production would have shaped."""
    mode = "DROVE shaping (production path)" if auto_ability else "shadow (oracle drove shaping)"
    print(f"\n  Ability-Router Diagnostics  (detect_ability, {mode})")
    labels = ("MR", "TR", "NONE")
    conf = diag["confusion"]
    print(f"    confusion — rows=oracle target, cols=detected")
    print(f"    {'':>10}" + "".join(f"{c:>8}" for c in labels))
    for t in labels:
        print(f"    {t:>10}" + "".join(f"{conf[t].get(c, 0):>8}" for c in labels))
    print(f"    {'─'*42}")
    for intent in ("MR", "TR"):
        p = diag["per_intent"][intent]
        rec = "n/a" if p["recall"] is None else f"{p['recall']*100:.0f}%"
        pre = "n/a" if p["precision"] is None else f"{p['precision']*100:.0f}%"
        print(f"    {intent}: recall {rec:>4} ({p['tp']}/{p['actual']})   "
              f"precision {pre:>4} ({p['tp']}/{p['predicted']})")
    aa = diag["abstain_accuracy"]
    aa_s = "n/a" if aa is None else f"{aa*100:.0f}%"
    print(f"    NONE categories: abstain {aa_s} "
          f"({diag['none_total'] - diag['false_positives']}/{diag['none_total']}), "
          f"{diag['false_positives']} mis-shaped to MR/TR")
    print(f"\n    Read: low MR/TR recall → the production router misses these "
          f"questions, so\n          their oracle-label gain is partly illusory "
          f"in real Hermes (build detection).")


def print_report(scores: dict, metadata: dict):
    """Print LongMemEval results."""
    print(f"\n{'='*80}")
    print(f"  HYMEM LONGMEMEVAL RESULTS")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Answer LLM: {metadata.get('answer_model')} / Judge: {metadata.get('judge_model')}")
    print(f"  Questions evaluated: {metadata.get('num_questions')}")
    print(f"  Top-K: {metadata.get('top_k', DEFAULT_TOP_K)}")
    print(f"  Scale: {metadata.get('scale', DEFAULT_SCALE)}")
    print(f"{'='*80}")

    print(f"\n  Per-Ability Scores:")
    for qtype, data in sorted(scores.items()):
        if qtype == "OVERALL":
            continue
        print(f"    {qtype:<30} {data['accuracy']*100:>5.1f}%  (n={data['count']})")
    print(f"    {'─'*45}")
    overall = scores.get("OVERALL", {})
    print(f"    {'OVERALL':<30} {overall.get('accuracy', 0)*100:>5.1f}%  (n={overall.get('count', 0)})")

    print(f"\n  Published SOTA (LongMemEval-S, GPT-4o judge):")
    print(f"    Hindsight: 89.4%")
    print(f"    Honcho:    63.8%")
    print(f"    LIGHT:     51.7%")
    print(f"    RAG:       48.5%")


# ── Per-question worker ─────────────────────────────────────────────

def _evaluate_one_question(qi, total, q_data, args, answer_llm, judge_llm,
                           api_key, distill_llm=None):
    """Full lifecycle for one question: fresh temp DB → open → evaluate → cleanup.

    Self-contained so it can run in a worker thread. Each question gets its own
    SQLite file + HyMem instance (created and used entirely within this call, so
    no connection crosses threads); the only shared state is the two LLMClients,
    whose counters are lock-guarded. Returns the result dict (never raises — an
    error is captured as an incorrect result so one bad question can't abort a
    parallel run)."""
    print(f"[{qi+1}/{total}] Q: {q_data['question_id']} ({q_data['question_type']})", flush=True)

    # Fresh temp DB per question (sessions are question-specific)
    tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-lme-"))
    db_path = tmp_dir / "hymem.sqlite"
    hy = None
    try:
        hy = HyMemAdapter(db_path, api_key=api_key, embeddings=args.embeddings,
                          rerank_top_k=args.rerank_top_k, rerank_model=args.rerank_model,
                          rerank_message_hits=args.rerank_message_hits,
                          aggregation_nodes=args.aggregation_nodes,
                          aggregation_broad=args.aggregation_broad,
                          episode_granularity=args.episode_granularity,
                          value_supersession=args.value_supersession,
                          graph_multihop=args.graph_multihop,
                          graph_multihop_max_hops=args.graph_multihop_max_hops,
                          graph_multihop_decay=args.graph_multihop_decay,
                          graph_multihop_min_score=args.graph_multihop_min_score,
                          rules_enabled=args.rules,
                          rules_extraction=args.rules_extraction,
                          facts_enabled=args.facts,
                          facts_extraction=args.facts_extraction)
        hy.open()
        result = evaluate_question(
            answer_llm, judge_llm, hy, q_data, args.top_k,
            auto_ability=args.auto_ability, no_dream=args.no_dream,
            graph_facts_first=args.graph_facts_first,
            permissive_default=args.permissive_default,
            distill=args.distill,
            distill_prompt_version=args.distill_prompt_version,
            retrieval_only=getattr(args, "retrieval_only", False),
            distill_llm=distill_llm,
        )
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        result = {
            "question_id": q_data.get("question_id", "unknown"),
            "question_type": q_data.get("question_type", "unknown"),
            "correct": False,
            "error": str(e),
        }
    finally:
        if hy:
            hy.close()
        if not args.keep_db:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # Force GC to prevent file handle leaks
        gc.collect()
    return result


# ── Floor inspector (characterize WHY the floor turns evade every tier) ──
# The floor audit proved 14 MS gold turns reach NO retrieval tier. This turns
# "unrecoverable" from a count into a named failure mode per question, so we can
# decide whether a NEW retrieval path is worth building (and would carry to real
# Hermes) or whether the residual is genuinely synthesis. LLM-free in the analysis;
# runs the real ingest/dream/search so the tiers shown are the production tiers.

_INSPECT_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with from by as is are was were be been "
    "being do does did have has had i you he she it we they my your his her our their "
    "this that these those what when where who which why how me him us them not no yes "
    "can could would should will shall may might must about into over under then than "
    "there here so if out up down off all any some most more very just like get got "
    "what's i'm i've it's don't didn't isn't there's how's".split()
)


def _salient_tokens(text: str) -> set[str]:
    """Lowercased content tokens (≥3 chars, non-stopword) for overlap diagnosis."""
    import re
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in _INSPECT_STOPWORDS}


def _find_gold_location(q_data: dict, gold_text: str) -> str:
    """Where a gold turn sits in the haystack — sanity that the floor is a retrieval
    gap, not a gold-extraction artifact. Returns 'session sid, turn k/n' or 'NOT FOUND'."""
    gnorm = _norm_text(gold_text)
    sessions = q_data.get("haystack_sessions", []) or []
    sids = q_data.get("haystack_session_ids", [str(i) for i in range(len(sessions))])
    for sid, sess in zip(sids, sessions):
        for k, m in enumerate(sess):
            if isinstance(m, dict) and _norm_text(m.get("content", "")) == gnorm:
                return f"session {sid}, turn {k+1}/{len(sess)} (role={m.get('role','?')})"
    return "NOT FOUND in haystack (gold-extraction artifact?)"


def _inspect_floor_questions(questions: list[dict], args, api_key: str) -> None:
    """For each floor qid (ranking miss with ≥1 gold turn in NO tier), dump the
    question, the unrecovered gold turn(s), where they sit, their raw message-FTS
    rank, the question↔gold token overlap, and what the retriever surfaced instead —
    so the failure mode is legible, not a black box."""
    from hymem.query.augment import _message_fts_search

    # Select the floor set from the instrumented run JSON (exactly the audited qids).
    with open(args.inspect_floor) as f:
        run = json.load(f)
    pq = run.get("per_question", [])
    if not any("gold_turn_tiers" in r for r in pq):
        print(f"\n⚠ {Path(args.inspect_floor).name} has no 'gold_turn_tiers' — re-run the "
              "baseline with the instrumented adapter (any recent run) before inspecting.")
        return
    floor_ids = [r["question_id"] for r in pq
                 if (args.category == "all" or r.get("question_type") == args.category)
                 and not r.get("correct") and r.get("recall_ceiling") is True
                 and any(t == "none" for t in (r.get("gold_turn_tiers") or []))]
    by_id = {q.get("question_id"): q for q in questions}
    missing = [qid for qid in floor_ids if qid not in by_id]
    floor_ids = [qid for qid in floor_ids if qid in by_id]
    if missing:
        print(f"\n⚠ {len(missing)} floor qid(s) not in the loaded dataset sample — "
              f"re-run with --sample 0 to inspect all: {missing[:5]}")

    tier_mode = ("message+chunk+vector" if args.embeddings and not args.no_dream
                 else "message+chunk (no embeddings)" if not args.no_dream
                 else "message-only (--no-dream)")
    print(f"\n{'='*72}\nFLOOR INSPECTOR — {len(floor_ids)} {args.category} floor questions "
          f"from {Path(args.inspect_floor).name}")
    print(f"  tiers exercised: {tier_mode}   "
          f"(run with --embeddings and full dream to reproduce the audited floor)\n{'='*72}")

    mode_tally: Counter = Counter()
    for n, qid in enumerate(floor_ids, 1):
        q_data = by_id[qid]
        question = q_data["question"]
        q_tokens = _salient_tokens(question)
        tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-inspect-"))
        hy = None
        try:
            hy = HyMemAdapter(tmp_dir / "hymem.sqlite", api_key=api_key,
                              embeddings=args.embeddings,
                              rerank_top_k=args.rerank_top_k, rerank_model=args.rerank_model,
                              rerank_message_hits=args.rerank_message_hits)
            hy.open()
            sessions = q_data.get("haystack_sessions", [])
            sids = q_data.get("haystack_session_ids",
                              [str(i) for i in range(len(sessions))])
            hy.ingest_sessions(sessions, sids, q_data.get("haystack_dates", []))
            if not args.no_dream:
                hy.dream_and_wait()
            gold_turns, _ = _extract_gold_turns(q_data)
            # Search with the SAME oracle ability the audited run used (MR for
            # multi-session), so the live tiers reproduce the audited floor.
            oracle_ability = QUESTION_TYPE_TO_ABILITY.get(q_data.get("question_type"), None)
            # `pool` is the LAST element of search()'s tuple — indexed from the
            # end so adding a tier (narrative_facts made this 7-wide) can't
            # silently hand the floor audit a different tier's list.
            pool = hy.search(question, ability=oracle_ability, top_k=args.top_k * 3)[-1]
            tiers = _gold_turn_tiers(gold_turns, pool)
            floor_turns = [g for g, t in zip(gold_turns, tiers) if t == "none"]
            # Deep raw-message-FTS scan to confirm the floor + show what DID rank.
            hits = _message_fts_search(hy.hy.conn, question, top_k=60)
        except Exception as e:
            print(f"\n[{n}] {qid} ERROR: {e}")
            continue
        finally:
            if hy:
                hy.close()
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            gc.collect()

        print(f"\n[{n}/{len(floor_ids)}] {qid}  ({q_data.get('question_type')})")
        print(f"  Q: {question}")
        print(f"  A: {str(q_data.get('answer',''))[:160]}")
        print(f"  question salient tokens: {sorted(q_tokens)}")
        print(f"  ── floor gold turn(s) [{len(floor_turns)} of {len(gold_turns)} gold "
              f"reach NO tier] ──")
        for g in floor_turns:
            gtok = _salient_tokens(g)
            shared = sorted(q_tokens & gtok)
            # Raw message-FTS rank of THIS gold turn (None = not even in 60-deep BM25).
            rank = next((i for i, h in enumerate(hits, 1)
                         if _gold_in_pool([g], [h.text])), None)
            loc = _find_gold_location(q_data, g)
            print(f"    • {g[:240]}")
            print(f"      at: {loc}")
            print(f"      raw msg-FTS rank: {rank if rank else 'NOT in top-60'}   "
                  f"shared salient tokens w/ Q: {len(shared)} {shared}")
            print(f"      gold-only tokens (what Q never says): "
                  f"{sorted(gtok - q_tokens)[:12]}")
            # Heuristic failure-mode tag (advisory — read the text to confirm).
            if not shared:
                mode = "VOCAB GAP (zero shared content tokens — paraphrase/synonym)"
            elif len(shared) <= 2:
                mode = "WEAK OVERLAP (1-2 shared tokens — buried under verbose siblings)"
            else:
                mode = "IMPLICIT (tokens overlap but turn is contextually indirect)"
            mode_tally[mode.split(" (")[0]] += 1
            print(f"      FAILURE MODE: {mode}")
        print(f"  ── what the retriever surfaced instead (top 3 raw msg-FTS) ──")
        for i, h in enumerate(hits[:3], 1):
            htok = _salient_tokens(h.text)
            print(f"    {i}. [{getattr(h,'role','?')}] {h.text[:140]}")
            print(f"       shares w/ Q: {sorted(q_tokens & htok)[:8]}")

    print(f"\n{'='*72}\nFAILURE-MODE TALLY (advisory): "
          + "  ".join(f"{k}={v}" for k, v in mode_tally.most_common()))
    print("  Read: VOCAB GAP dominant → a paraphrase/semantic path (better embeddings,\n"
          "  query expansion, or HyDE) is the lever — but L1 showed vector adds no recall\n"
          "  on LME, so confirm the gap is true paraphrase, not just sparse signal.\n"
          "  IMPLICIT/multi-hop dominant → the answer needs turn-linking (graph/dream\n"
          "  bridging), not a flat retriever. Weigh CARRY-OVER: an LME-only fix that\n"
          "  doesn't help real Hermes is out of scope.")


# ── Distillation dry-run (G-P1a front-run gate for --distill) ────────
# Offline test of the P1 bounded-reflect mechanism on the banked MS synthesis
# misses BEFORE spending a full A/B. Mirrors the --inspect-floor pattern: rebuild
# a per-question temp DB from the source run's dataset, ingest/dream/search with
# the SAME config, then run the distillation arm and judge. The deep-lexical
# split (2.1) is verified LIVE — a candidate only counts as a synthesis miss if
# its gold turn actually survived into the char-capped context the answerer sees.

def _distill_run_one(q_data: dict, args, answer_llm: LLMClient, judge_llm: LLMClient,
                     api_key: str, *, check_gold_in_context: bool) -> dict:
    """Rebuild one question's store, retrieve, distill, answer over distilled+raw,
    and judge. Returns a result dict; never raises (an error → incorrect). When
    `check_gold_in_context`, also renders the RAW (no-distill) context and reports
    whether a gold turn survived into it — the live deep-lexical split."""
    question = q_data["question"]
    question_type = q_data["question_type"]
    answer = q_data["answer"]
    sessions = q_data.get("haystack_sessions", [])
    sids = q_data.get("haystack_session_ids", [str(i) for i in range(len(sessions))])
    dates = q_data.get("haystack_dates", [])
    question_date = q_data.get("question_date", "") or (max(dates) if dates else "")
    oracle_ability = QUESTION_TYPE_TO_ABILITY.get(question_type, None)
    ability = _detect_ability_safe(question) if args.auto_ability else oracle_ability

    out = {"question_id": q_data.get("question_id"), "question": question,
           "gold_answer": str(answer), "ability": ability, "gold_in_context": None,
           "distill_calls": 0, "distill_kept": 0, "distilled": [], "ai_answer": "",
           "correct": False, "error": None,
           # Defaults so the record carries the D3 channel even when the judge
           # branch below is never reached (an exception before judging leaves
           # `correct` False guarded by `error`, which is reader-side and out of
           # D3's scope — D3 is about the JUDGE failing, not the reader).
           "judge_raw": "", "judge_error": False}

    tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-distill-"))
    hy = None
    try:
        hy = HyMemAdapter(tmp_dir / "hymem.sqlite", api_key=api_key,
                          embeddings=args.embeddings,
                          rerank_top_k=args.rerank_top_k, rerank_model=args.rerank_model,
                          rerank_message_hits=args.rerank_message_hits,
                          aggregation_nodes=args.aggregation_nodes,
                          aggregation_broad=args.aggregation_broad,
                          episode_granularity=getattr(
                              args, "episode_granularity", False),
                          value_supersession=args.value_supersession,
                          facts_enabled=getattr(args, "facts", None),
                          facts_extraction=getattr(args, "facts_extraction", None))
        hy.open()
        hy.ingest_sessions(sessions, sids, dates)
        if not args.no_dream:
            hy.dream_and_wait()
        (memories, total_matches, graph_count, temporal_events, aggregation_nodes,
         narrative_facts, pool) = hy.search(
            question, ability=ability, top_k=args.top_k * 3,
            graph_facts_first=args.graph_facts_first)

        if check_gold_in_context:
            gold_turns, _ = _extract_gold_turns(q_data)
            # Render the RAW context exactly as the answerer would see it (same
            # char caps) and check the gold turn survived the cut. Gold below the
            # cut = deep-lexical (retrieval-ranking loss, not synthesis) → excluded.
            raw_ctx = _render_answer_context(memories, ability, total_matches, graph_count,
                                             temporal_events, aggregation_nodes, distilled=None,
                                             narrative_facts=narrative_facts)
            out["gold_in_context"] = bool(gold_turns) and _gold_in_pool(gold_turns, [raw_ctx])

        distilled, calls = distill_memories(answer_llm, question, memories,
                                            prompt_version=args.distill_prompt_version)
        out["distill_calls"], out["distill_kept"], out["distilled"] = calls, len(distilled), distilled
        ai_answer = answer_question(answer_llm, memories, question, ability=ability,
                                    total_matches=total_matches, graph_count=graph_count,
                                    temporal_events=temporal_events,
                                    aggregation_nodes=aggregation_nodes,
                                    question_date=question_date,
                                    permissive_default=args.permissive_default,
                                    distilled=distilled,
                                    narrative_facts=narrative_facts)
        out["ai_answer"] = ai_answer
        _correct, _judge_raw = judge_scored(judge_llm, question_type, question,
                                            answer, ai_answer)
        out["correct"] = _correct
        out["judge_raw"] = _judge_raw
        out["judge_error"] = _correct is None
    except Exception as e:
        out["error"] = str(e)
    finally:
        if hy:
            hy.close()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
    return out


def _distill_dryrun_questions(questions: list[dict], args, api_key: str) -> None:
    """G-P1a front-run gate. Reads the instrumented source run, recovers the MS
    synthesis misses (with a live deep-lexical split), runs the distillation arm
    on each + an equal-sized MS-hit control, and reports the flip rate + control
    regressions against G-P1a — plus every flipped answer for the hand-read."""
    with open(args.distill_dryrun) as f:
        run = json.load(f)
    pq = run.get("per_question", [])
    if not any("gold_turn_tiers" in r for r in pq):
        print(f"\n⚠ {Path(args.distill_dryrun).name} has no 'gold_turn_tiers' — the "
              "synthesis-miss selection needs an instrumented run. Re-run the baseline "
              "with this adapter first.")
        return
    by_id = {q.get("question_id"): q for q in questions}

    def _is_ms(r):  # multi-session, answerable (non-_abs)
        return r.get("question_type") == "multi-session"

    # 2.1 selection (pre deep-lexical split — that is verified live below):
    # MS, wrong, recall_ceiling=true (gold was in the pool), NOT floor (no "none"
    # gold-turn tier).
    candidates = [r["question_id"] for r in pq
                  if _is_ms(r) and not r.get("correct")
                  and r.get("recall_ceiling") is True
                  and not any(t == "none" for t in (r.get("gold_turn_tiers") or []))
                  and r.get("question_id") in by_id]
    # Control: MS hits (correct=true), equal-sized random sample (seeded → paired).
    hit_ids = [r["question_id"] for r in pq
               if _is_ms(r) and r.get("correct") and r.get("question_id") in by_id]
    import random
    rng = random.Random(args.seed)
    n_ctrl = min(len(candidates), len(hit_ids))
    control = rng.sample(hit_ids, n_ctrl) if n_ctrl < len(hit_ids) else list(hit_ids)

    missing = [r["question_id"] for r in pq
               if _is_ms(r) and not r.get("correct") and r.get("recall_ceiling") is True
               and not any(t == "none" for t in (r.get("gold_turn_tiers") or []))
               and r.get("question_id") not in by_id]

    print(f"\n{'='*72}\nDISTILL DRY-RUN (G-P1a) — source: {Path(args.distill_dryrun).name}")
    print(f"  candidates (MS synthesis-miss, pre split): {len(candidates)}   "
          f"control (MS hits): {len(control)}")
    if missing:
        print(f"  ⚠ {len(missing)} candidate qid(s) not in the loaded sample — "
              f"re-run with --sample 0: {missing[:5]}")
    if args.answer_base_url != DEEPSEEK_BASE_URL:
        print(f"  answer reader: {args.answer_model} @ {args.answer_base_url}  "
              f"(judge frozen: {args.judge_model} @ {DEEPSEEK_BASE_URL})")
    print(f"  distill prompt: {args.distill_prompt_version.upper()}   "
          f"max calls/q: {DISTILL_MAX_CALLS}\n{'='*72}", flush=True)

    answer_llm = LLMClient(args.answer_model, args.answer_api_key or api_key,
                           base_url=args.answer_base_url,
                           extra_body=getattr(args, "answer_extra_body_obj", None))
    judge_llm = LLMClient(args.judge_model, api_key,
                          extra_body=getattr(args, "judge_extra_body_obj", None))

    tasks = [("cand", qid) for qid in candidates] + [("ctrl", qid) for qid in control]

    def _run(kind: str, qid: str) -> dict:
        res = _distill_run_one(by_id[qid], args, answer_llm, judge_llm, api_key,
                               check_gold_in_context=(kind == "cand"))
        res["_kind"] = kind
        return res

    results: list[dict] = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_run, k, q) for k, q in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 5 == 0:
                    print(f"  ── {i}/{len(tasks)} done", flush=True)
    else:
        for i, (k, q) in enumerate(tasks, 1):
            results.append(_run(k, q))
            if i % 5 == 0:
                print(f"  ── {i}/{len(tasks)} done", flush=True)

    cand_res = [r for r in results if r["_kind"] == "cand"]
    ctrl_res = [r for r in results if r["_kind"] == "ctrl"]
    # Live deep-lexical split: only gold-in-context candidates are true synthesis
    # misses; gold-below-cut rows are deep-lexical (retrieval/ranking loss).
    synthesis = [r for r in cand_res if r.get("gold_in_context") and not r.get("error")]
    deeplexical = [r for r in cand_res if not r.get("gold_in_context") and not r.get("error")]
    cand_errors = [r for r in cand_res if r.get("error")]
    flips = [r for r in synthesis if r["correct"]]
    regressions = [r for r in ctrl_res if not r["correct"] and not r.get("error")]
    ctrl_errors = [r for r in ctrl_res if r.get("error")]

    n_syn = len(synthesis)
    flip_rate = (len(flips) / n_syn) if n_syn else 0.0

    print(f"\n{'='*72}\nRESULT — G-P1a")
    print(f"  candidates run: {len(cand_res)}  "
          f"(synthesis: {n_syn}, deep-lexical excluded: {len(deeplexical)}, "
          f"errors: {len(cand_errors)})")
    if n_syn != 20:
        print(f"  ⚠ recovered synthesis set = {n_syn}, not the banked 20 — reconcile "
              f"against the decomposition; the gate scales as a FRACTION (≥25%), not ≥5.")
    print(f"  FLIPS to correct: {len(flips)}/{n_syn}  ({flip_rate*100:.0f}%)")
    print(f"  control regressions (MS hit → wrong under distill): "
          f"{len(regressions)}/{len(ctrl_res)}"
          + (f"  (+{len(ctrl_errors)} errors)" if ctrl_errors else ""))
    avg_calls = (sum(r["distill_calls"] for r in synthesis) / n_syn) if n_syn else 0
    avg_kept = (sum(r["distill_kept"] for r in synthesis) / n_syn) if n_syn else 0
    print(f"  distill cost (synthesis rows): avg {avg_calls:.1f} calls, "
          f"{avg_kept:.1f} lines kept per question")

    pass_flip = flip_rate >= 0.25
    pass_ctrl = len(regressions) <= 1
    verdict = "PASS (pending hand-read)" if (pass_flip and pass_ctrl) else "FAIL"
    print(f"\n  GATE: flip≥25% {'✓' if pass_flip else '✗'}  "
          f"regressions≤1 {'✓' if pass_ctrl else '✗'}  →  {verdict}")
    print("  Hand-read every FLIPPED answer below for INVENTED facts before "
          "accepting the pass (the judge can be charitable — this is the honesty check).")

    print(f"\n{'─'*72}\nFLIPPED ANSWERS (hand-read for invention):")
    for r in flips:
        print(f"\n  [{r['question_id']}]  Q: {r['question'][:140]}")
        print(f"    gold: {r['gold_answer'][:160]}")
        print(f"    answer: {r['ai_answer'][:240]}")
        print(f"    distilled ({r['distill_kept']} lines from {r['distill_calls']} calls):")
        for line in r["distilled"][:12]:
            print(f"      • {line[:160]}")
    if regressions:
        print(f"\n{'─'*72}\nCONTROL REGRESSIONS (were correct, now wrong under distill) —"
              f"\n  read the distilled lines: over-extraction (on-topic noise) vs a lossy"
              f"\n  line the model trusted over the raw turn tells which lever failed:")
        for r in regressions:
            print(f"\n  [{r['question_id']}]  Q: {r['question'][:140]}")
            print(f"    gold: {r['gold_answer'][:160]}")
            print(f"    answer: {r['ai_answer'][:240]}")
            print(f"    distilled ({r['distill_kept']} lines from {r['distill_calls']} calls):")
            for line in r["distilled"][:12]:
                print(f"      • {line[:160]}")
    print(f"\n{'='*72}\nBank this block + the verdict in longmemeval_roadmap.md under P1.\n")


# ── Re-judge (re-pair a banked baseline under a new judge) ───────────
# Built for the 2026-07-24 deepseek-chat hard-deprecation: the canonical 70.0
# baseline was answered AND judged by deepseek-chat, so it can't be reproduced.
# A parity run judged by the replacement (deepseek-v4-flash) is only comparable
# to a baseline judged by the SAME judge — but re-answering the baseline is
# wasteful. This re-runs ONLY the judge over the stored hypotheses, no ingest /
# no answer, and reports the per-category judge drift.

def _rejudge_run(args, api_key: str) -> None:
    """Re-judge a stored results JSON under the current --judge-model
    (+ --judge-extra-body). Writes a re-judged copy and prints original-vs-new
    per-category drift. Rows with no hypothesis (or an [LLM_ERROR] answer) keep
    their prior verdict, uncounted as re-judged."""
    src = Path(args.rejudge)
    with open(src) as f:
        run = json.load(f)
    pq = run.get("per_question", [])
    if not pq:
        print(f"ERROR: {src.name} has no per_question rows to re-judge.")
        return
    orig_judge = (run.get("config", {}) or {}).get("judge_model", "unknown")

    judge_llm = LLMClient(args.judge_model, api_key,
                          extra_body=getattr(args, "judge_extra_body_obj", None))

    # Flag the pre-untruncation artifact (q[:200]/a[:200]/hyp[:500]) so the
    # approximation caveat is raised only when it actually applies.
    clipped = sum(1 for r in pq
                  if len(str(r.get("hypothesis", ""))) == 500
                  or len(str(r.get("question", ""))) == 200
                  or len(str(r.get("answer", ""))) == 200)

    print(f"\n{'='*72}\nRE-JUDGE — {src.name}")
    print(f"  rows: {len(pq)}   original judge: {orig_judge}   new judge: {args.judge_model}"
          + (f"  +extra_body={args.judge_extra_body_obj}" if args.judge_extra_body_obj else ""))
    if not args.judge_extra_body_obj and "v4-flash" in args.judge_model:
        print("  ⚠ v4-flash judge WITHOUT --judge-extra-body "
              "'{\"thinking\":{\"type\":\"disabled\"}}' — reasoning tokens may corrupt the yes/no parse.")
    if clipped:
        print(f"  ⚠ ~{clipped} rows look field-clipped (pre-untruncation run) — "
              "re-judge is a close approximation for those, not byte-faithful.")
    print(f"{'='*72}", flush=True)

    def _rj(r: dict) -> dict:
        hyp = str(r.get("hypothesis", ""))
        judge_raw = ""
        if not hyp or is_llm_error(hyp) or r.get("error"):
            new, judged = bool(r.get("correct")), False   # nothing judgeable → keep prior
        else:
            verdict, judge_raw = judge_scored(
                judge_llm, r.get("question_type", ""), r.get("question", ""),
                str(r.get("answer", "")), hyp)
            # A JUDGE error keeps the prior verdict and leaves `_rejudged` False,
            # reusing the answer-side rule one line up rather than inventing a
            # second one. Overwriting with None would drop the row out of the
            # flip denominator AND destroy the baseline it is being compared to.
            new, judged = (bool(r.get("correct")), False) if verdict is None \
                else (verdict, True)
        out = dict(r)
        out["judge_raw"] = judge_raw
        out["judge_error"] = bool(judge_raw) and is_llm_error(judge_raw)
        out["correct_original"] = r.get("correct")
        out["correct"] = new
        out["_rejudged"] = judged
        return out

    new_rows: list[dict] = [None] * len(pq)
    t0 = time.time()
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_rj, r): i for i, r in enumerate(pq)}
            done = 0
            for fut in as_completed(futs):
                new_rows[futs[fut]] = fut.result()
                done += 1
                if done % 25 == 0:
                    print(f"  ── re-judged {done}/{len(pq)}", flush=True)
    else:
        for i, r in enumerate(pq):
            new_rows[i] = _rj(r)

    orig_scores = compute_scores(
        [{**r, "correct": r.get("correct_original")} for r in new_rows])
    new_scores = compute_scores(new_rows)
    elapsed = time.time() - t0

    print(f"\n{'─'*72}\nJUDGE DRIFT  ({orig_judge} → {args.judge_model})")
    print(f"  {'category':<26} {'orig':>7} {'rejudged':>9} {'Δpp':>7} {'n':>5}")
    for qtype in sorted(new_scores.keys()):
        o = orig_scores.get(qtype, {}).get("accuracy", 0) * 100
        n = new_scores[qtype]["accuracy"] * 100
        print(f"  {qtype:<26} {o:>6.1f} {n:>8.1f} {n - o:>+7.1f} {new_scores[qtype]['count']:>5}")
    to_wrong = [r for r in new_rows if r["_rejudged"] and r.get("correct_original") and not r["correct"]]
    to_right = [r for r in new_rows if r["_rejudged"] and not r.get("correct_original") and r["correct"]]
    n_judged = sum(1 for r in new_rows if r["_rejudged"])
    print(f"\n  flips: correct→wrong {len(to_wrong)}, wrong→correct {len(to_right)}, "
          f"net {len(to_right) - len(to_wrong):+d}   (re-judged {n_judged}/{len(new_rows)})")

    out = dict(run)
    # §6.3 (2026-08-31): the rejudge artifact must carry its OWN date and
    # stats — previously `out = dict(run)` inherited the source run's date,
    # total_tokens and elapsed_s (registry row id=41: byte-identical to the
    # source row id=42), so a reader could not tell the rejudge's cost from
    # the run's cost.  The registry NULLs these columns for rejudge rows
    # regardless; the artifact itself simply must not lie.
    out["date"] = datetime.now(timezone.utc).isoformat()
    out["per_question"] = new_rows
    out["scores"] = {qtype: {"accuracy": round(d["accuracy"] * 100, 1), "count": d["count"]}
                     for qtype, d in new_scores.items()}
    cfg = dict(out.get("config", {}) or {})
    cfg.update({"rejudged_from": src.name, "rejudge_original_judge": orig_judge,
                "judge_model": args.judge_model,
                "judge_extra_body": getattr(args, "judge_extra_body_obj", None),
                "answer_calls": 0,
                "judge_calls": judge_llm.call_count,
                "total_tokens": judge_llm.total_tokens,
                "elapsed_s": elapsed})
    out["config"] = cfg
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = src.with_name(f"{src.stem}-rejudged-{args.judge_model.replace('/', '_')}-{stamp}.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Archived re-judged results → {dest.name}")
    print(f"{'='*72}\n  Use the re-judged OVERALL as the paired baseline for a parity run "
          f"judged by {args.judge_model}.\n")


# ── Main ────────────────────────────────────────────────────────────

def main():
    global DEEPSEEK_API_KEY

    parser = argparse.ArgumentParser(description="HyMem LongMemEval Benchmark")
    parser.add_argument("--scales", default=DEFAULT_SCALE)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="questions to evaluate; 0 = full set (no sampling variance)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Retrieve and fingerprint the reader's prompt, but "
                             "make NO answer or judge call. Produces rows with "
                             "correct=None and a context_sha, for measuring how "
                             "many questions a lever actually moves (`f`) "
                             "without paying for the reader. Every scorer "
                             "REFUSES the resulting artifact.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for stratified sampling; fixed so runs are paired/comparable")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--answer-model", default=ANSWER_MODEL)
    parser.add_argument("--answer-base-url", default=DEEPSEEK_BASE_URL,
                        help="P0 parity lever: OpenAI-compatible endpoint for the "
                             "ANSWER client only (default DeepSeek). Point at a "
                             "gpt-oss-120b-class reader to measure how much of the "
                             "gap to Hindsight's 91.4 is reader vs architecture. "
                             "The JUDGE stays deepseek-chat @ DeepSeek regardless — "
                             "judge posture is the frozen comparability contract, "
                             "never varied. Recorded in the output metadata so a "
                             "run is self-describing.")
    parser.add_argument("--answer-api-key", default=None,
                        help="API key for the --answer-base-url endpoint. Defaults "
                             "to the resolved --api-key/env/config DeepSeek key, so "
                             "the default path is unchanged; set this only when the "
                             "answer endpoint needs a different credential.")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--answer-extra-body", default=None, metavar="JSON",
                        help="JSON object merged into the ANSWER request body (raw-HTTP "
                             "`extra_body`). Use for a reasoning reader that needs "
                             "thinking off, e.g. deepseek-v4-flash: "
                             "'{\"thinking\":{\"type\":\"disabled\"}}'. gpt-oss-120b "
                             "does not need it.")
    parser.add_argument("--judge-extra-body", default=None, metavar="JSON",
                        help="JSON object merged into the JUDGE request body. REQUIRED "
                             "when judging with deepseek-v4-flash (deepseek-chat was "
                             "hard-deprecated 2026-07-24): "
                             "'{\"thinking\":{\"type\":\"disabled\"}}' — else the "
                             "reasoning preamble corrupts the yes/no parse.")
    parser.add_argument("--rejudge", default=None, metavar="RUN.json",
                        help="Re-judge a stored results JSON under the current "
                             "--judge-model (+ --judge-extra-body) — NO ingest, NO answer "
                             "calls, just the judge pass over each stored hypothesis. "
                             "Built for the deepseek-chat deprecation: re-pair the banked "
                             "70.0 baseline under deepseek-v4-flash so a parity run judged "
                             "by the same new judge is comparable. Reports per-category "
                             "judge drift (original vs re-judged) and archives a new JSON. "
                             "Skips the benchmark. NOTE: runs produced before the "
                             "field-truncation was lifted carry clipped q/a/hypothesis — "
                             "the re-judge is then a close approximation, not byte-faithful.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--data-dir", default=str(_repo_root.parent / "hymem_beam" / "data"))
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of questions to evaluate concurrently. "
                             "Questions are independent (own temp DB), so this "
                             "scales near-linearly on the I/O-bound LLM calls. "
                             "Results are byte-identical to --workers 1; only "
                             "throughput changes. Try 8 to start.")
    parser.add_argument("--no-dream", action="store_true",
                        help="Skip the per-question dream cycle (the dominant "
                             "cost). The message/rerank/MR retrieval paths read "
                             "messages_fts (built at ingest) and don't need it. "
                             "FAST-MODE FOR RELATIVE A/B ONLY — degrades the "
                             "dreamed-chunk/KG/temporal tiers, so it is NOT a "
                             "faithful headline run. Do one full-dream pass for "
                             "the published number.")
    parser.add_argument("--graph-facts-first", action="store_true",
                        help="A/B OPT-OUT: restore the legacy graph-facts-first "
                             "ordering for IE/KU/PF lookups (dream-derived "
                             "graph_facts ranked above raw message_hits). The "
                             "DEFAULT is now message-first for every ability — the "
                             "production-realistic shape, since detect_ability "
                             "returns None for those lookups and graph-facts-first "
                             "caused the −14.3pp SS-user regression. Use this flag "
                             "only to reproduce the old ordering for comparison.")
    parser.add_argument("--auto-ability", action="store_true",
                        help="Drive retrieval shaping from HyMem's production "
                             "detect_ability() instead of the oracle question_type "
                             "label — measures the true label-free production score. "
                             "(The router confusion is reported either way.)")
    parser.add_argument("--permissive-default", action="store_true",
                        help="LEVER D4: use a permissive preference-style DEFAULT "
                             "answer prompt for the unknown-ability case instead of "
                             "the strict 'only provided context' one. Targets the "
                             "SS-P auto-ability crater (11.7 to ~73 acc): a label-free "
                             "preference question (router emits None) currently gets "
                             "the strict prompt and refuses recommendation questions. "
                             "Adapter-side prompt change only — not a HyMem change. "
                             "ALWAYS read the broken-out Answerable-vs-Abstention "
                             "report after: permissiveness can trade correct '_abs' "
                             "refusals for hallucinations. A/B against the strict "
                             "default on a fixed seed.")
    parser.add_argument("--rerank-top-k", type=int, default=None,
                        help="LEVER L2a (ranking): candidate-pool width the message/chunk "
                             "reranker sees (config default 20). The message tier pulls "
                             "max(message_fts_top_k=15, rerank_top_k) BM25 candidates and "
                             "reranks down to 15 — so a gold turn below this BM25 rank "
                             "NEVER enters the rerank window and no reranker can lift it. "
                             "Widen (40, 60) to give deeper-BM25 gold a rerank shot; this "
                             "directly targets the ranking misses (recall is already "
                             "ruled out). Adds reranker cost per query.")
    parser.add_argument("--rerank-model", default=None, choices=["llm", "cross-encoder"],
                        help="LEVER L2b (ranking): reranker backend (config default 'llm', "
                             "reuses the deepseek host client). 'cross-encoder' uses a local "
                             "sentence-transformers model (mxbai-rerank-base — English-only, "
                             "fine for LME; production multilingual needs bge-reranker-v2-m3). "
                             "A/B against the RAW-BM25 baseline (--no-rerank-message-hits), "
                             "not just against 'llm': the gold-rank probe showed the LLM "
                             "reranker demotes gold it already sees, so a replacement must "
                             "beat OFF, not merely beat the incumbent.")
    parser.add_argument("--rerank-message-hits", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="LEVER L2c (ranking): toggle the dominant MESSAGE-tier reranker "
                             "(config default ON). --no-rerank-message-hits restores raw BM25 "
                             "order on the message tier. The gold-rank probe found 92%% of MS "
                             "gold already at BM25 rank ≤15 yet 65 MS ranking misses remain — "
                             "i.e. the reranker is dropping gold it already sees. This is the "
                             "DIAGNOSTIC GATE: run it FIRST. If raw BM25 beats the LLM "
                             "reranker on MS, the fix is removing the reranker, not replacing "
                             "it (skip L2b). If it doesn't recover MS, the loss is downstream "
                             "(packing/budget, → L3), not the reranker. Default None = config.")
    parser.add_argument("--embeddings", action="store_true",
                        help="LEVER L1: enable semantic vector recall. Works with NO env "
                             f"setup — defaults to the local FastEmbed server "
                             f"({LOCAL_EMBED_MODEL} @ {LOCAL_EMBED_BASE_URL}, dim "
                             f"{LOCAL_EMBED_DIM}, api_key='local') that Hermes runs in "
                             "production. Override any field with HYMEM_EMBEDDING_API_KEY/"
                             "_BASE_URL/_MODEL/_DIM to point at a different server. DEFAULT "
                             "OFF (lexical-only baseline); run paired on --seed to measure "
                             "the recall the FTS-only path leaves behind.")
    parser.add_argument("--aggregation-nodes", action="store_true",
                        help="RAPTOR G4 lever: enable the Phase-2 cross-session aggregation "
                             "layer (dream builds cluster-summary nodes; the retrieval tier "
                             "fires only for abilities in cfg.aggregation_inject_abilities — "
                             "TR-only by default — and renders as a separate "
                             "[CROSS-SESSION SUMMARIES] block that never competes with raw "
                             "turns for top_k slots). DEFAULT OFF. Requires a dream pass "
                             "(do not combine with --no-dream) and --embeddings for the "
                             "node vector arm; with --auto-ability the gate uses the "
                             "router's TR detection, mirroring production.")
    parser.add_argument("--aggregation-broad", action="store_true",
                        help="(with --aggregation-nodes) clear the ability allowlist so the "
                             "aggregation tier fires on EVERY question — reproduces the "
                             "broad-injection G4 A/B that lost 69.0 vs 70.0 (KU −9.0pp from "
                             "nodes crowding gold turns out of the answer pool). For "
                             "comparison runs only.")
    parser.add_argument("--episode-granularity", action="store_true",
                        help="Plan C lever (cfg.episode_granularity_enabled): extract "
                             "DECISION-level episodes -- several per session, each with "
                             "its own outcome -- instead of one blob segment per session. "
                             "DEFAULT OFF, and OFF is the frozen-baseline arm. This is a "
                             "WRITE-side lever: it changes what the dream extracts, so it "
                             "is inert against an existing store and the arm MUST be built "
                             "with --fresh (and without --no-dream). A run that reuses a "
                             "store dreamt under the other prompt measures the store, not "
                             "the lever, and will read as a clean null. Front-run gate "
                             "G-EP1 PASSED 2026-08-31 (benchmarks/episode_probe.py); this "
                             "flag exists to run the LME non-regression guard the flip "
                             "still owes.")
    parser.add_argument("--graph-multihop", action="store_true",
                        help="Track A / Idea A lever (cfg.graph_multihop_enabled): enable "
                             "query-time multi-hop graph traversal (Source 4 of _graph_lookup) "
                             "— a read-only BFS from directly-anchored entities that bridges "
                             "edges 1-hop retrieval misses (e.g. atta —part_of→ medflow "
                             "—deploys_to→ fly.io). Additive (never displaces direct hits). "
                             "DEFAULT OFF. This is the G-A2 non-regression guard arm; the recall "
                             "gate G-A1 runs separately via benchmarks/multihop_probe.py. "
                             "Requires a dream pass (edges come from dreaming).")
    parser.add_argument("--graph-multihop-max-hops", type=int, default=None,
                        help="(with --graph-multihop) override cfg.graph_multihop_max_hops — "
                             "the swept Pareto point (default 2). None = config default.")
    parser.add_argument("--graph-multihop-decay", type=float, default=None,
                        help="(with --graph-multihop) override cfg.graph_multihop_decay "
                             "(default 0.5). None = config default.")
    parser.add_argument("--graph-multihop-min-score", type=float, default=None,
                        help="(with --graph-multihop) override cfg.graph_multihop_min_score "
                             "(default 0.05). None = config default.")
    parser.add_argument("--rules", action=argparse.BooleanOptionalAction, default=None,
                        help="Idea B READ side (cfg.rules_enabled). None = config default "
                             "(ON). Pass --no-rules for the pre-Idea-B control arm. NOTE: "
                             "INERT on LME — the harness never calls add_rule() and LME has "
                             "no rule-obedience questions, so --rules vs --no-rules is a flat "
                             "non-regression check, NOT a needle-mover. Rule adherence is "
                             "gated by benchmarks/rules_compliance.py, not here.")
    parser.add_argument("--rules-extraction", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Idea B WRITE side (cfg.rules_extraction_enabled). None = config "
                             "default (OFF). The ONLY rules lever that changes the LME answer "
                             "path: it routes dream markers into agent_inferred rules that then "
                             "inject into every ask(). This is the non-regression guard for "
                             "flipping the write-side default — expected FLAT on LME (factual "
                             "recall, not behavior); a regression means auto-rules pollute "
                             "answers, so DON'T flip. Requires a dream pass.")
    parser.add_argument("--facts", action=argparse.BooleanOptionalAction, default=None,
                        help="Campaign E / E1 narrative-facts READ side "
                             "(cfg.facts_enabled). None = config default (ON). "
                             "Pass --no-facts for the paired control arm: same store, "
                             "tier rendered or not. UNLIKE --rules this is NOT inert — "
                             "the tier renders its own [NARRATIVE FACTS] block above the "
                             "raw turns, so --facts vs --no-facts on one store is the "
                             "clean A/B for what the tier is worth. Read `n_facts` in the "
                             "per-question rows FIRST: all zeros means the tier never "
                             "fired and the score is a no-op by construction.")
    parser.add_argument("--facts-extraction", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="E1 WRITE side (cfg.facts_extraction_enabled). None = config "
                             "default (ON). Costs one extra dream call per session tail. "
                             "Turning this off changes what is STORED, so it only takes "
                             "effect on a --fresh rebuild — do not mix it into a "
                             "read-side A/B against an existing store.")
    parser.add_argument("--value-supersession", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Bi-temporal KU lever (cfg.value_supersession_enabled): the "
                             "dream cycle retracts the OLDER of two competing typed-value "
                             "edges (number/date/version) sharing subject+predicate, "
                             "closing its invalid_at at the newer value's valid_at — so an "
                             "updated fact supersedes the stale one instead of both staying "
                             "active. Default ON, matching the library default since the "
                             "2026-07-02 guard run (score-neutral, zero false positives); "
                             "--no-value-supersession restores the historical flag-off "
                             "control arm. Requires a dream pass (do NOT combine "
                             "with --no-dream). Confirm firing via the dream-log line "
                             "'bitemporal.value_superseded count=' before reading results.")
    parser.add_argument("--distill", action="store_true",
                        help="P1 read-side lever: before the final answer call, map a "
                             "question-conditioned extraction call over the retrieved "
                             "hits (message/fts/episode) and answer over the distilled "
                             "one-line facts PLUS the raw turns — a bounded single-step "
                             "'reflect'. ADDITIVE (distilled facts join, never replace "
                             "the raw memories); the distill block renders ABOVE the "
                             "turns as a non-competing tier. Cost-gated: fires only on "
                             "MR/TR or a ≥12-hit retrieval (label-free). Adds up to "
                             f"{DISTILL_MAX_CALLS} small LLM calls per fired question. "
                             "Run the free --distill-dryrun front-run gate FIRST; A/B "
                             "against the paired baseline on a fixed seed.")
    parser.add_argument("--distill-prompt-version", default=DEFAULT_DISTILL_PROMPT_VERSION,
                        choices=sorted(DISTILL_PROMPTS.keys()),
                        help="Which DISTILL_PROMPT to map (default v2). v2 tightens the "
                             "relevance bar to answer-bearing facts only after v1's G-P1a "
                             "FAIL (6 flips / 6 control regressions — over-extraction "
                             "crowding). v1 stays selectable to reproduce that run.")
    parser.add_argument("--distill-dryrun", default=None, metavar="RUN.json",
                        help="FRONT-RUN GATE (G-P1a) for --distill: reads an instrumented "
                             "run JSON, recovers the banked MS synthesis misses (MS, "
                             "wrong, recall_ceiling, no floor turn, gold survived into "
                             "the sent context — the deep-lexical split is verified live), "
                             "runs the distillation arm on each, and reports how many flip "
                             "to correct. Also runs an equal-sized random control of MS "
                             "HITS to catch regressions. LLM cost is ~40 small questions, "
                             "not a full run. Skips the benchmark. Bank the verdict before "
                             "spending a full --distill A/B.")
    parser.add_argument("--inspect-floor", default=None, metavar="RUN.json",
                        help="DIAGNOSTIC: characterize WHY the floor questions (ranking "
                             "misses whose gold reaches NO tier) are unrecoverable. Reads an "
                             "instrumented run JSON (needs gold_turn_tiers), then for each "
                             "floor qid dumps the question, the unrecovered gold turn(s), "
                             "their haystack location + raw msg-FTS rank, the question↔gold "
                             "token overlap, and what ranked instead. Run WITH --embeddings "
                             "(+full dream) to reproduce the audited floor. Skips the benchmark.")
    parser.add_argument("--category", default="multi-session",
                        help="(--inspect-floor) question_type to inspect, or 'all'.")
    args = parser.parse_args()

    # Resolve API key
    DEEPSEEK_API_KEY = args.api_key or os.environ.get("HYMEM_LLM_API_KEY", "")
    if not DEEPSEEK_API_KEY:
        config_path = Path("/home/node/.hermes/config.yaml")
        if config_path.exists():
            for line in config_path.read_text().split("\n"):
                s = line.strip()
                if s.startswith("HYMEM_LLM_API_KEY:"):
                    DEEPSEEK_API_KEY = s.split(":", 1)[1].strip().strip('"').strip("'")
                    break
    if not DEEPSEEK_API_KEY:
        print("ERROR: No API key. Set --api-key, HYMEM_LLM_API_KEY env var, or config.yaml key.")
        sys.exit(1)

    print(f"API key: ...{DEEPSEEK_API_KEY[-4:]}", flush=True)

    # Parse the optional per-client extra_body JSON (fail fast on bad JSON).
    def _parse_extra_body(raw: str | None, which: str) -> dict | None:
        if not raw:
            return None
        try:
            val = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"ERROR: --{which}-extra-body is not valid JSON: {e}")
            sys.exit(1)
        if not isinstance(val, dict):
            print(f"ERROR: --{which}-extra-body must be a JSON object, got {type(val).__name__}")
            sys.exit(1)
        return val
    args.answer_extra_body_obj = _parse_extra_body(args.answer_extra_body, "answer")
    args.judge_extra_body_obj = _parse_extra_body(args.judge_extra_body, "judge")

    # Re-judge a stored results JSON under the current judge — no dataset needed,
    # so dispatch before the (large) dataset load. Built for the deepseek-chat
    # deprecation: re-pair the banked baseline under deepseek-v4-flash.
    if args.rejudge:
        _rejudge_run(args, DEEPSEEK_API_KEY)
        return

    # Determine dataset path
    scale = args.scales.upper()
    if scale == "S":
        data_file = Path(args.data_dir) / "longmemeval_s_cleaned.json"
    else:
        data_file = Path(args.data_dir) / f"longmemeval_{scale.lower()}_cleaned.json"

    if not data_file.exists():
        print(f"ERROR: Dataset not found at {data_file}")
        print("Download: curl -L -o <path> https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json")
        sys.exit(1)

    print(f"\nHyMem LongMemEval Benchmark")
    print(f"  Dataset: {data_file} ({data_file.stat().st_size / 1024 / 1024:.0f} MB)")
    print(f"  Scale: {scale}")
    print(f"  Max questions: {args.sample if args.sample else 'ALL (no sampling)'}")
    print(f"  Seed: {args.seed}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Answer model: {args.answer_model}")
    if args.answer_base_url != DEEPSEEK_BASE_URL:
        print(f"  ⚠ Answer endpoint: {args.answer_base_url} (PARITY READER — "
              f"judge stays deepseek-chat @ {DEEPSEEK_BASE_URL})")
    print(f"  Judge model: {args.judge_model}")
    print(f"  Workers: {args.workers}")
    if args.embeddings:
        emb_url = os.environ.get("HYMEM_EMBEDDING_BASE_URL") or LOCAL_EMBED_BASE_URL
        emb_model = os.environ.get("HYMEM_EMBEDDING_MODEL") or LOCAL_EMBED_MODEL
        src = "env" if os.environ.get("HYMEM_EMBEDDING_BASE_URL") else "local default"
        print(f"  Embeddings: ON (semantic recall) — {emb_model} @ {emb_url} [{src}]")
    else:
        print(f"  Embeddings: OFF (lexical/FTS-only — baseline)")
    if (args.rerank_top_k is not None or args.rerank_model is not None
            or args.rerank_message_hits is not None):
        rk = args.rerank_top_k if args.rerank_top_k is not None else "default(20)"
        rm = args.rerank_model if args.rerank_model is not None else "default(llm)"
        if args.rerank_message_hits is False:
            mt = "OFF (raw BM25 — L2c diagnostic)"
        elif args.rerank_message_hits is True:
            mt = "ON (forced)"
        else:
            mt = "default(ON)"
        print(f"  Rerank: top_k={rk}, model={rm}, message_tier={mt}  (L2 ranking lever)")
    if args.no_dream:
        print(f"  ⚠ --no-dream: FAST MODE (relative A/B only, NOT a headline run "
              f"— dream/KG/temporal tiers degraded)")
    print(f"  Default answer prompt: "
          + ("PERMISSIVE (D4 — preference-style, abstention-guarded)"
             if args.permissive_default else "STRICT (only provided context)"))

    # Load data
    print("\nLoading dataset...", flush=True)
    questions = load_longmemeval_data(
        str(data_file),
        max_questions=args.sample or None,  # --sample 0 -> full set
        seed=args.seed,
    )
    total_sessions = sum(len(q.get("haystack_sessions", [])) for q in questions)
    total_msgs = sum(sum(len(s) for s in q.get("haystack_sessions", [])) for q in questions)
    print(f"  Total: {len(questions)} questions, ~{total_sessions} sessions, ~{total_msgs} messages\n")

    # Floor inspector: a diagnostic, not a benchmark run — dump and exit.
    if args.inspect_floor:
        _inspect_floor_questions(questions, args, DEEPSEEK_API_KEY)
        return

    # Distillation dry-run (G-P1a front-run gate): offline test on the banked
    # synthesis misses — dump the verdict and exit, no full benchmark.
    if args.distill_dryrun:
        _distill_dryrun_questions(questions, args, DEEPSEEK_API_KEY)
        return

    # LLM clients. The ANSWER client can be pointed at a non-DeepSeek reader
    # (P0 parity lever); its key falls back to the resolved DeepSeek key so the
    # default path is unchanged. The JUDGE stays on DEEPSEEK_BASE_URL; post the
    # deepseek-chat deprecation it must be deepseek-v4-flash + --judge-extra-body
    # '{"thinking":{"type":"disabled"}}' to keep the yes/no parse clean.
    answer_api_key = args.answer_api_key or DEEPSEEK_API_KEY
    # Under --retrieval-only the reader and judge are never reached, and the
    # clients enforce that rather than trusting the branch: PoisonLLM raises,
    # CountingLLM records what distillation and the store actually spend.
    answer_llm = LLMClient(args.answer_model, answer_api_key, base_url=args.answer_base_url,
                           extra_body=args.answer_extra_body_obj)
    judge_llm = LLMClient(args.judge_model, DEEPSEEK_API_KEY,
                          extra_body=args.judge_extra_body_obj)
    retrieval_counter = None
    distill_llm = None
    if args.retrieval_only:
        # The reader and the judge become unreachable OBJECTS, not skipped
        # branches. Distillation keeps a real (counted) client because it is
        # part of retrieval and genuinely fires.
        retrieval_counter = CountingLLM(answer_llm)
        distill_llm = retrieval_counter
        answer_llm = PoisonLLM("reader")
        judge_llm = PoisonLLM("judge")

    # Evaluate each question. Questions are fully independent (own temp DB +
    # HyMem instance), so --workers > 1 fans them across a thread pool — the work
    # is ~entirely LLM network I/O, so the GIL is released and threads scale
    # near-linearly while sharing the LLMClient token counters.
    start_time = time.time()
    total = len(questions)

    def _progress(done: int):
        elapsed = time.time() - start_time
        suffix_w = f" (×{args.workers} workers)" if args.workers > 1 else ""
        if args.retrieval_only:
            # No verdicts exist. "Acc: 0.0%" here is not a low score, it is a
            # measurement that was never taken.
            print(f"  ── Progress: {done}/{total} | retrieval-only (no "
                  f"verdicts) | Elapsed: {elapsed:.0f}s | "
                  f"Avg: {elapsed/max(1, done):.0f}s/q{suffix_w}", flush=True)
            return
        acc = accuracy(all_results)
        n_err = len(judge_error_rows(all_results))
        suffix = f" (×{args.workers} workers)" if args.workers > 1 else ""
        # An outage should be visible WHILE the run burns reader calls, not only
        # in the post-mortem — that is the whole point of D3. Silent when zero:
        # the run summary states the denominator instead.
        err = f" | ⚠ UNSCORED (judge error): {n_err}" if n_err else ""
        print(f"  ── Progress: {done}/{total} | Acc: {acc*100:.1f}% | "
              f"Elapsed: {elapsed:.0f}s | Avg: {elapsed/max(1, done):.0f}s/q"
              f"{suffix}{err}", flush=True)

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Collect by original index so per_question stays input-ordered (stable
        # / comparable across runs) even though completion order is arbitrary.
        results_by_idx: dict[int, dict] = {}
        all_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_evaluate_one_question, qi, total, q_data, args,
                            answer_llm, judge_llm, DEEPSEEK_API_KEY, distill_llm): qi
                for qi, q_data in enumerate(questions)
            }
            for fut in as_completed(futures):
                qi = futures[fut]
                results_by_idx[qi] = fut.result()
                all_results = list(results_by_idx.values())
                if len(results_by_idx) % 10 == 0:
                    _progress(len(results_by_idx))
        all_results = [results_by_idx[i] for i in range(total)]
    else:
        all_results = []
        for qi, q_data in enumerate(questions):
            all_results.append(
                _evaluate_one_question(qi, total, q_data, args,
                                       answer_llm, judge_llm, DEEPSEEK_API_KEY,
                                       distill_llm)
            )
            if (qi + 1) % 10 == 0:
                _progress(qi + 1)

    elapsed = time.time() - start_time

    # ---- SAFETY DUMP -----------------------------------------------------
    # Written BEFORE any reporting touches these rows, because everything
    # expensive has already happened and everything after this line is
    # presentation. On 2026-09-03 the f probe dreamed 50 questions in 876s and
    # then died in the summary `print` on an attribute the stand-in client did
    # not carry: the run was complete, and the result was destroyed by a
    # formatting statement. Five diagnostic passes sit between here and the
    # real artifact; any of them can raise, and none of them is worth the
    # compute above.
    #
    # Deliberately minimal and deliberately NOT named like a real artifact, so
    # nothing globbing `longmemeval-v2-hymem-*` picks it up by accident.
    try:
        _safety = Path("/home/node/.hermes/benchmarks") / (
            f"partial-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            f"-seed{args.seed}.json")
        _safety.write_text(json.dumps({
            "partial": True,
            "note": "written before reporting; carries the rows, not the "
                    "summary",
            "config": {
                "retrieval_only": bool(args.retrieval_only),
                "episode_granularity_enabled": bool(
                    getattr(args, "episode_granularity", False)),
                "seed": args.seed,
                "sample": args.sample,
                "scale": args.scales,
                "answer_model": args.answer_model,
                "judge_model": args.judge_model,
                "elapsed_s": elapsed,
            },
            "per_question": all_results,
        }))
        print(f"  safety dump: {_safety}", flush=True)
    except Exception as _e:                      # never let the guard be the
        print(f"  ⚠ safety dump failed: {_e}", flush=True)   # thing that fails

    total_calls = answer_llm.call_count + judge_llm.call_count
    print(f"\nEvaluation complete in {elapsed:.0f}s")
    print(f"  Answer calls: {answer_llm.call_count}, Judge calls: {judge_llm.call_count}")
    print(f"  Total tokens: {answer_llm.total_tokens + judge_llm.total_tokens}")
    if args.distill:
        n_fired = sum(1 for r in all_results if r.get("distill_fired"))
        n_calls = sum(r.get("distill_calls", 0) for r in all_results)
        print(f"  Distill: fired on {n_fired}/{len(all_results)} questions, "
              f"{n_calls} extraction calls (included in Answer calls above)")
    print(f"  Avg time/question: {elapsed/len(questions):.0f}s")

    # Report. The judge-error line comes FIRST and is unconditional: a score
    # printed without it cannot be read, because the reader cannot tell an
    # accuracy over n rows from an accuracy over n-k. When k is 0 the line
    # states the denominator instead, so "0 errors" is never a claim made by an
    # instrument that had nothing to measure.
    print(f"\n  {judge_error_note(all_results)}")
    scores = compute_scores(all_results)
    print_report(scores, {
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "num_questions": len(questions),
        "top_k": args.top_k,
        "scale": scale,
    })
    abstention_diag = compute_abstention_scores(all_results)
    print_abstention_scores(abstention_diag)
    recall_diag = compute_recall_diagnostics(all_results)
    print_recall_diagnostics(recall_diag)
    router_diag = compute_router_diagnostics(all_results)
    print_router_diagnostics(router_diag, args.auto_ability)

    # Save
    results_dir = Path("/home/node/.hermes/benchmarks")
    results_dir.mkdir(exist_ok=True, parents=True)

    output = {
        "benchmark": "LongMemEval",
        "version": "v2-hymem-tr-mr-wired",
        "date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "scale": scale,
            "sample": args.sample,
            "seed": args.seed,
            "retrieval_only": bool(args.retrieval_only),
            "top_k": args.top_k,
            "auto_ability": args.auto_ability,
            "workers": args.workers,
            "no_dream": args.no_dream,
            # The config LEVERS this adapter pins, recorded because a run
            # that varies one of them has to be able to EVIDENCE which arm it
            # was. The 2026-08-31 granularity guard diffed the two arms' config
            # blocks and found them "byte-identical except elapsed_s" -- which
            # was true, and vacuous: neither lever was in the block, so the diff
            # could not see the only thing the run varied, and would have read
            # the same had --episode-granularity been forgotten on the ON arm.
            # A drift control that cannot observe the variable is not a control
            # over it (the E3 shape: a check that reads PASS when broken).
            # `aggregation_nodes` matters for a different reason: the library
            # default flipped False -> True on 2026-08-26, so it is what makes a
            # score comparable (or not) to the pre-flip 70.0 / 68.4 baselines.
            # Key names match hymem.config field names exactly, because
            # benchmarks/lme_registry.py reads them by that name -- it lists
            # all three under KNOWN_ABSENT ("filled from `overrides` in the
            # adapter, which never made it into the config block"). Writing
            # them under any other spelling would leave those columns NULL
            # forever while looking recorded here.
            "aggregation_nodes_enabled": args.aggregation_nodes,
            "episode_granularity_enabled": getattr(
                args, "episode_granularity", False),
            "value_supersession_enabled": args.value_supersession,
            "graph_facts_first": args.graph_facts_first,
            "permissive_default": args.permissive_default,
            "embeddings": args.embeddings,
            "rerank_top_k": args.rerank_top_k,
            "rerank_model": args.rerank_model,
            "rerank_message_hits": args.rerank_message_hits,
            "answer_model": args.answer_model,
            "answer_base_url": args.answer_base_url,
            "answer_extra_body": args.answer_extra_body_obj,
            "judge_model": args.judge_model,
            "judge_extra_body": args.judge_extra_body_obj,
            "graph_multihop": args.graph_multihop,
            "graph_multihop_knobs": (
                {"max_hops": args.graph_multihop_max_hops,
                 "decay": args.graph_multihop_decay,
                 "min_score": args.graph_multihop_min_score}
                if args.graph_multihop else None
            ),
            "distill": args.distill,
            "distill_prompt_version": args.distill_prompt_version.upper() if args.distill else None,
            "distill_fired_count": sum(1 for r in all_results if r.get("distill_fired")),
            "distill_total_calls": sum(r.get("distill_calls", 0) for r in all_results),
            "hy_mem": "beam-optimisation branch (53d490d + adapter wiring)",
            "features": "created_at from haystack_dates, graph_count trusted, temporal_events injected (hits-based anchors), question_date as reference-now, str(answer) fix, recall-ceiling instrumentation (retrieval-vs-ranking miss split), ability-router shadow/auto measurement (detect_ability vs oracle)",
            "elapsed_s": elapsed,
            "answer_calls": answer_llm.call_count,
            "judge_calls": judge_llm.call_count,
            "total_tokens": answer_llm.total_tokens + judge_llm.total_tokens,
        },
        "scores": {qtype: {
            "accuracy": round(data["accuracy"] * 100, 1),
            "count": data["count"],
        } for qtype, data in scores.items()},
        "abstention_diagnostics": abstention_diag,
        "recall_diagnostics": recall_diag,
        "router_diagnostics": router_diag,
        "per_question": all_results,
    }
    if args.retrieval_only:
        # What the mode ACTUALLY spent, measured rather than assumed. It skips
        # the reader and the judge; it does not skip distillation, and it
        # cannot vouch for what reranking does inside the store.
        output["retrieval_cost"] = {
            "llm_calls": retrieval_counter.calls if retrieval_counter else None,
            "prompt_chars": (retrieval_counter.prompt_chars
                             if retrieval_counter else None),
            "answer_calls": 0,
            "judge_calls": 0,
            "distill_calls": sum(r.get("distill_calls") or 0
                                 for r in all_results),
        }

    # Canonical "latest" pointer (stable filename other tools read)...
    results_path = results_dir / "longmemeval-v2-hymem.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    # ...plus an immutable, uniquely-named archive copy so a later run never
    # clobbers a prior result (a seeded run is reproducible, but per-question
    # detail and one-off numbers are worth keeping). Key by timestamp+seed.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = results_dir / f"longmemeval-v2-hymem-{stamp}-seed{args.seed}.json"
    with open(archive_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Archived: {archive_path.name}", flush=True)

    # Update manifest
    manifest_path = results_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest["LongMemEval-v2"] = {
        "latest": "longmemeval-v2-hymem.json",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "overall_score": round(scores.get("OVERALL", {}).get("accuracy", 0) * 100, 1),
        "scale": scale,
        "sample": args.sample,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
