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

ANSWERING_SYSTEM_PROMPT = """You are an AI assistant answering questions based on retrieved memories from past conversations.
Answer the question concisely using ONLY the provided context. 
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Do not make up information. Do not use outside knowledge."""

ANSWERING_PREFERENCE_PROMPT = """You are an AI assistant answering questions based on retrieved memories from past conversations.
The context contains personal information about the user (preferences, possessions, habits, experiences).
Use this personal information to generate a personalized response to the question.
You may draw on general knowledge to fill in details, but tailor your answer to respect what you know about the user.
If the context contains NO relevant personal information about the user, say "I don't have enough information to answer this question." """

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

# ── LLM Client ──────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.base_url = DEEPSEEK_BASE_URL.rstrip("/")
        self.call_count = 0
        self.total_tokens = 0
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
                elif attempt < 2:
                    time.sleep(3)
                else:
                    break
        return f"[LLM_ERROR: {last_error[:100]}]"

    def _call(self, messages: list, temperature: float, max_tokens: int) -> tuple[str, dict]:
        resp = http.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        with self._lock:
            self.call_count += 1
        return (
            data["choices"][0]["message"].get("content", ""),
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
            qtype = item["question_type"]
            by_type[qtype].append(item)

    total_available = sum(len(v) for v in by_type.values())
    num_types = len(by_type)

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

    def __init__(self, db_path: Path, api_key: str = ""):
        self.db_path = db_path
        self.api_key = api_key
        self.hy = None

    def open(self):
        from hymem import HyMem, HyMemConfig
        from hymem.contrib.openai_client import OpenAICompatibleClient

        cfg = HyMemConfig(
            root=self.db_path.parent,
            message_fts_top_k=15,
            fts_top_k=10,
            graph_top_k=10,
        )
        llm = OpenAICompatibleClient(
            api_key=self.api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )
        self.hy = HyMem(cfg, llm=llm)
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
               message_first: bool = False):
        """Search HyMem for the given query.

        Returns (memories, total_matches, graph_count, temporal_events, pool)
        where `pool` is the FULL pre-truncation candidate text by tier
        ({"message": [...], "fts": [...]}) — used for recall-ceiling analysis,
        so a category's misses can be split into retrieval loss vs ranking loss.
        """
        try:
            result = self.hy.augment(query, ability=ability)
        except Exception as e:
            print(f"    [DEBUG] augment error: {e}", flush=True)
            return [], 0, None, [], {"message": [], "fts": []}

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

        if ability in TASK_RECALL or message_first:
            # message_first extends the task-recall ordering (raw turns lead,
            # dream-derived graph_facts demoted to a confidence-ranked tail) to
            # the IE/KU/PF lookups too. Tests whether full-dream's SS-user drop is
            # caused by graph_facts crowding out the answer turn, rather than by
            # dreaming itself — see the dream-vs-no-dream matrix.
            memories = message_hits + procedure_hits
            rest = episode_hits + fts_hits + graph_facts
            rest.sort(key=lambda m: -m.get("confidence", 0))
            memories += rest
        else:
            # Knowledge/preference: graph facts first, then message hits
            graph_facts.sort(key=lambda m: -m.get("confidence", 0))
            memories = graph_facts + message_hits + episode_hits + fts_hits + procedure_hits
            memories.sort(key=lambda m: (
                m["type"] != "graph_fact",
                0 if m["type"] == "message_hit" else 1,
                -m.get("confidence", 0),
            ))

        # The recall-ceiling pool is the FULL retrieved set per tier, captured
        # before the memories[:top_k] cut, so we measure whether the gold turn
        # was retrievable at all — independent of the final ordering/truncation.
        pool = {
            "message": [m["content"] for m in message_hits],
            "fts": [m["content"] for m in fts_hits],
        }
        return (memories[:top_k], getattr(result, 'total_message_matches', 0),
                getattr(result, 'graph_count', None),
                getattr(result, 'temporal_events', []), pool)


# ── Answer & Judge ──────────────────────────────────────────────────

def answer_question(llm: LLMClient, memories: list[dict], question: str, ability: str = None,
                    total_matches: int = 0, graph_count=None, temporal_events: list | None = None,
                    question_date: str = "") -> str:
    """Ask LLM to answer based on retrieved memories.

    Uses ability-aware prompts and expanded context for multi-session
    and temporal reasoning questions that need more cross-session data.
    For MR questions, prefers graph_count (exact graph-native count) over
    total_matches (keyword candidate). For TR questions, injects
    temporal_events as a date-ordered chronology.

    `question_date` is the "now" the question is asked at — the reference point
    relative-date questions ("how many days ago?", "a month ago") subtract from.
    Without it the chronology gives event dates but the model has no anchor to
    compute an interval against, and answers "current date not provided".
    """
    # MR and TR questions span many sessions — expand context window
    context_limit = MAX_CONTEXT_CHARS * 2 if ability in ("MR", "TR") else MAX_CONTEXT_CHARS

    # MR counting: prefer graph_count (EXACT graph-native count) over
    # total_matches (keyword candidate). When graph_count is present it is
    # the dedup-correct answer — trust it.
    parts = []
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
    total_chars = 0
    for m in memories:
        content = m["content"]
        if total_chars + len(content) > context_limit:
            break
        tag = "[FACT]" if m["type"] == "graph_fact" else "[MEM]"
        parts.append(f"{tag} {content}")
        total_chars += len(content) + len(tag) + 2

    context = "\n".join(parts) if parts else "No relevant memories found."

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
            system_prompt = ANSWERING_SYSTEM_PROMPT  # fallback if no user messages
    elif ability == "TR":
        system_prompt = ANSWERING_TR_PROMPT
    else:
        system_prompt = ANSWERING_SYSTEM_PROMPT

    # The reference "now" for relative-date math. Stated explicitly so the model
    # can subtract event dates from it ("how many days ago", "a month ago")
    # instead of complaining the current date is unknown.
    today_line = f"Today's date is {question_date}.\n\n" if question_date else ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{today_line}CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"},
    ]

    return llm.chat(messages, temperature=0.0, max_tokens=1024)


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


def judge_answer(llm: LLMClient, question_type: str, question: str, answer: str, ai_answer: str) -> bool:
    """Judge whether the answer is correct (binary yes/no per LongMemEval protocol)."""
    prompt = get_judge_prompt(question_type, question, answer, ai_answer)
    messages = [{"role": "user", "content": prompt}]
    raw = llm.chat(messages, temperature=0.0, max_tokens=10)
    return "yes" in raw.lower()


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_question(
    llm: LLMClient,
    judge_llm: LLMClient,
    hy: HyMemAdapter,
    q_data: dict,
    top_k: int,
    auto_ability: bool = False,
    no_dream: bool = False,
    message_first: bool = False,
) -> dict:
    """Evaluate a single LongMemEval question."""
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
    memories, total_matches, graph_count, temporal_events, pool = hy.search(
        question, ability=ability, top_k=top_k * 3, message_first=message_first)
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
    else:
        recall_ceiling, recall_tier = None, "unknown"

    ceiling_str = ("∅" if recall_ceiling is None
                   else f"{recall_ceiling}[{recall_tier}]")
    used_marker = "←used" if auto_ability else ""
    router_str = f"{oracle_ability or '∅'}/det={detected_ability or '∅'}{used_marker}"
    print(f"    Retrieved {len(memories)} memories (total_matches={total_matches}, graph_count={graph_count is not None}, temporal_events={len(temporal_events)}, now={question_date or '∅'}[{src}], ceiling={ceiling_str}, ability={router_str})", flush=True)

    # Answer
    ai_answer = answer_question(llm, memories, question, ability=ability, total_matches=total_matches,
                                 graph_count=graph_count, temporal_events=temporal_events,
                                 question_date=question_date)

    # Judge (binary yes/no)
    correct = judge_answer(judge_llm, question_type, question, answer, ai_answer)
    print(f"    Correct: {correct} | Answer: {ai_answer[:120]}...", flush=True)

    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question[:200],
        "answer": str(answer)[:200],
        "hypothesis": ai_answer[:500],
        "correct": correct,
        "num_sessions": stats["sessions"],
        "num_messages": stats["messages"],
        "num_memories": len(memories),
        "recall_ceiling": recall_ceiling,
        "recall_tier": recall_tier,
        "gold_mode": gold_mode,
        "gold_turns": len(gold_turns),
        "oracle_ability": oracle_ability,
        "detected_ability": detected_ability,
        "ability_used": ability,
    }


def compute_scores(results: list[dict]) -> dict:
    """Compute accuracy by question type."""
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


def compute_recall_diagnostics(results: list[dict]) -> dict:
    """Per-category recall-ceiling stats: split misses into retrieval vs ranking.

    For each base type, over questions whose gold turns are known:
      - ceiling_rate: fraction whose answer turn entered the pre-truncation pool
      - among the INCORRECT ones, how many were retrieval losses (gold never
        retrieved) vs ranking/synthesis losses (gold retrieved but answer wrong)
    The miss split is the actionable signal: retrieval-dominant → embeddings/
    chunking; ranking-dominant → rerank/budget/packing.
    """
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

def _evaluate_one_question(qi, total, q_data, args, answer_llm, judge_llm, api_key):
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
        hy = HyMemAdapter(db_path, api_key=api_key)
        hy.open()
        result = evaluate_question(
            answer_llm, judge_llm, hy, q_data, args.top_k,
            auto_ability=args.auto_ability, no_dream=args.no_dream,
            message_first=args.message_first,
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


# ── Main ────────────────────────────────────────────────────────────

def main():
    global DEEPSEEK_API_KEY

    parser = argparse.ArgumentParser(description="HyMem LongMemEval Benchmark")
    parser.add_argument("--scales", default=DEFAULT_SCALE)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="questions to evaluate; 0 = full set (no sampling variance)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for stratified sampling; fixed so runs are paired/comparable")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--answer-model", default=ANSWER_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
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
    parser.add_argument("--message-first", action="store_true",
                        help="Rank raw message_hits ahead of dream-derived "
                             "graph_facts for IE/KU/PF lookups (graph_facts "
                             "demoted to a tail supplement), instead of "
                             "graph-facts-first. Isolates whether full-dream's "
                             "SS-user regression is a graph_facts-crowding "
                             "ordering artifact vs dreaming itself. Run with full "
                             "dream (i.e. WITHOUT --no-dream) to measure.")
    parser.add_argument("--auto-ability", action="store_true",
                        help="Drive retrieval shaping from HyMem's production "
                             "detect_ability() instead of the oracle question_type "
                             "label — measures the true label-free production score. "
                             "(The router confusion is reported either way.)")
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
    print(f"  Judge model: {args.judge_model}")
    print(f"  Workers: {args.workers}")
    if args.no_dream:
        print(f"  ⚠ --no-dream: FAST MODE (relative A/B only, NOT a headline run "
              f"— dream/KG/temporal tiers degraded)")

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

    # LLM clients
    answer_llm = LLMClient(args.answer_model, DEEPSEEK_API_KEY)
    judge_llm = LLMClient(args.judge_model, DEEPSEEK_API_KEY)

    # Evaluate each question. Questions are fully independent (own temp DB +
    # HyMem instance), so --workers > 1 fans them across a thread pool — the work
    # is ~entirely LLM network I/O, so the GIL is released and threads scale
    # near-linearly while sharing the LLMClient token counters.
    start_time = time.time()
    total = len(questions)

    def _progress(done: int):
        elapsed = time.time() - start_time
        acc = sum(1 for r in all_results if r.get("correct")) / max(1, len(all_results))
        suffix = f" (×{args.workers} workers)" if args.workers > 1 else ""
        print(f"  ── Progress: {done}/{total} | Acc: {acc*100:.1f}% | "
              f"Elapsed: {elapsed:.0f}s | Avg: {elapsed/max(1, done):.0f}s/q{suffix}",
              flush=True)

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Collect by original index so per_question stays input-ordered (stable
        # / comparable across runs) even though completion order is arbitrary.
        results_by_idx: dict[int, dict] = {}
        all_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_evaluate_one_question, qi, total, q_data, args,
                            answer_llm, judge_llm, DEEPSEEK_API_KEY): qi
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
                                       answer_llm, judge_llm, DEEPSEEK_API_KEY)
            )
            if (qi + 1) % 10 == 0:
                _progress(qi + 1)

    elapsed = time.time() - start_time
    total_calls = answer_llm.call_count + judge_llm.call_count
    print(f"\nEvaluation complete in {elapsed:.0f}s")
    print(f"  Answer calls: {answer_llm.call_count}, Judge calls: {judge_llm.call_count}")
    print(f"  Total tokens: {answer_llm.total_tokens + judge_llm.total_tokens}")
    print(f"  Avg time/question: {elapsed/len(questions):.0f}s")

    # Report
    scores = compute_scores(all_results)
    print_report(scores, {
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "num_questions": len(questions),
        "top_k": args.top_k,
        "scale": scale,
    })
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
            "top_k": args.top_k,
            "auto_ability": args.auto_ability,
            "workers": args.workers,
            "no_dream": args.no_dream,
            "message_first": args.message_first,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
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
        "recall_diagnostics": recall_diag,
        "router_diagnostics": router_diag,
        "per_question": all_results,
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
