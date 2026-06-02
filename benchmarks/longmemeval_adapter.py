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

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        last_error = None
        for attempt in range(3):
            try:
                content, usage = self._call(messages, temperature, max_tokens)
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

    def search(self, query: str, ability: str = None, top_k: int = 10):
        """Search HyMem for the given query. Returns (memories, total_matches)."""
        try:
            result = self.hy.augment(query, ability=ability)
        except Exception as e:
            print(f"    [DEBUG] augment error: {e}", flush=True)
            return [], 0, None, []

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

        if ability in TASK_RECALL:
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

        return memories[:top_k], getattr(result, 'total_message_matches', 0), getattr(result, 'graph_count', None), getattr(result, 'temporal_events', [])


# ── Answer & Judge ──────────────────────────────────────────────────

def answer_question(llm: LLMClient, memories: list[dict], question: str, ability: str = None,
                    total_matches: int = 0, graph_count=None, temporal_events: list | None = None) -> str:
    """Ask LLM to answer based on retrieved memories.

    Uses ability-aware prompts and expanded context for multi-session
    and temporal reasoning questions that need more cross-session data.
    For MR questions, prefers graph_count (exact graph-native count) over
    total_matches (keyword candidate). For TR questions, injects
    temporal_events as a date-ordered chronology.
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"},
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
) -> dict:
    """Evaluate a single LongMemEval question."""
    question_id = q_data["question_id"]
    question_type = q_data["question_type"]
    question = q_data["question"]
    answer = q_data["answer"]
    sessions = q_data.get("haystack_sessions", [])
    session_ids = q_data.get("haystack_session_ids", [str(i) for i in range(len(sessions))])
    session_dates = q_data.get("haystack_dates", [])

    # Ensure session_ids length matches sessions
    while len(session_ids) < len(sessions):
        session_ids.append(f"extra_{len(session_ids)}")

    # Map question type to HyMem ability
    ability = QUESTION_TYPE_TO_ABILITY.get(question_type, None)

    # Ingest
    stats = hy.ingest_sessions(sessions, session_ids, session_dates)
    print(f"    Ingested {stats['sessions']} sessions ({stats['messages']} msgs, {stats['chars']} chars)", flush=True)

    # Dream
    print(f"    Running dream cycle...", flush=True)
    hy.dream_and_wait()

    # Search
    memories, total_matches, graph_count, temporal_events = hy.search(question, ability=ability, top_k=top_k * 3)
    print(f"    Retrieved {len(memories)} memories (total_matches={total_matches}, graph_count={graph_count is not None}, temporal_events={len(temporal_events)})", flush=True)

    # Answer
    ai_answer = answer_question(llm, memories, question, ability=ability, total_matches=total_matches,
                                 graph_count=graph_count, temporal_events=temporal_events)

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

    # Evaluate each question
    all_results = []
    start_time = time.time()
    hy = None

    for qi, q_data in enumerate(questions):
        print(f"[{qi+1}/{len(questions)}] Q: {q_data['question_id']} ({q_data['question_type']})", flush=True)

        # Fresh temp DB per question (sessions are question-specific)
        tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-lme-"))
        db_path = tmp_dir / "hymem.sqlite"

        try:
            hy = HyMemAdapter(db_path, api_key=DEEPSEEK_API_KEY)
            hy.open()

            result = evaluate_question(answer_llm, judge_llm, hy, q_data, args.top_k)
            all_results.append(result)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            all_results.append({
                "question_id": q_data.get("question_id", "unknown"),
                "question_type": q_data.get("question_type", "unknown"),
                "correct": False,
                "error": str(e),
            })
        finally:
            if hy:
                hy.close()
                hy = None

            if not args.keep_db:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # Force GC to prevent file handle leaks
            gc.collect()

        # Progress report every 10 questions
        if (qi + 1) % 10 == 0:
            elapsed = time.time() - start_time
            acc_so_far = sum(1 for r in all_results if r.get("correct")) / len(all_results)
            print(f"  ── Progress: {qi+1}/{len(questions)} | Acc: {acc_so_far*100:.1f}% | Elapsed: {elapsed:.0f}s | Avg: {elapsed/(qi+1):.0f}s/q", flush=True)

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
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "hy_mem": "beam-optimisation branch (53d490d + adapter wiring)",
            "features": "created_at from haystack_dates, graph_count trusted, temporal_events injected, str(answer) fix",
            "elapsed_s": elapsed,
            "answer_calls": answer_llm.call_count,
            "judge_calls": judge_llm.call_count,
            "total_tokens": answer_llm.total_tokens + judge_llm.total_tokens,
        },
        "scores": {qtype: {
            "accuracy": round(data["accuracy"] * 100, 1),
            "count": data["count"],
        } for qtype, data in scores.items()},
        "per_question": all_results,
    }

    results_path = results_dir / "longmemeval-v2-hymem.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

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
