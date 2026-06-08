#!/usr/bin/env python3
"""
HyMem BEAM Benchmark Adapter (v2 — direct Python API)
======================================================
Runs the BEAM benchmark directly against the HyMem Python SDK.
No HTTP layer, no subprocess — same process, same env.

Usage:
  python beam_adapter.py --sample 5 --scales 100K
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add HyMem to path
# Ensure the HyMem package is importable (repo root is two levels up from benchmarks/)
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

import requests as http

# ── Config ──────────────────────────────────────────────────────────

DEFAULT_SCALE = "100K"
DEFAULT_SAMPLE = 3
DEFAULT_TOP_K = 10
MAX_CONTEXT_CHARS = 8000

# DeepSeek API
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ANSWER_MODEL = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"

ANSWERING_SYSTEM_PROMPT = """You are an AI assistant answering questions based on retrieved memories.
Answer the question concisely using ONLY the provided context. 
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Do not make up information. Do not use outside knowledge."""

ANSWERING_PREFERENCE_PROMPT = """You are an AI assistant answering questions based on retrieved memories from past conversations.
The context contains personal information about the user (preferences, possessions, habits, experiences).
Use this personal information to generate a personalized response to the question.
You may draw on general knowledge to fill in details, but tailor your answer to respect what you know about the user.
If the context contains NO relevant personal information about the user, say "I don't have enough information to answer this question." """

ANSWERING_MR_PROMPT = """You are an AI assistant answering questions based on retrieved memories from multiple conversations.
The question requires counting or aggregating information across conversations.
Carefully scan ALL the context for every relevant mention. Count distinct items — do not double-count restatements.
If the question asks "how many", return just the number (or a short answer with the number).
If you cannot find enough evidence in the context, say "I don't have enough information."
Do not make up information."""

ANSWERING_TR_PROMPT = """You are an AI assistant answering questions based on retrieved memories from past conversations.
The question requires reasoning about when events happened — dates, timelines, or event ordering.
Carefully scan ALL the context for relevant dates, times, and event mentions.
Calculate the answer from the evidence provided. For dated events, compute durations precisely.
If you cannot determine the answer from the context, say "I don't have enough information."
Do not make up dates or events."""

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating AI responses against rubrics.

Score 0 or 1 for each rubric item:
- 1: The answer fully satisfies this criterion
- 0: The answer does not satisfy this criterion

Return ONLY valid JSON:
{"scores": [1, 0, ...], "total_score": 0.X, "explanation": "brief"}
where total_score = sum(scores) / len(scores)"""

ABILITY_MAP = {
    "information_extraction": "IE",
    "multi_session_reasoning": "MR", "multi_session": "MR",
    "knowledge_update": "KU", "knowledge": "KU",
    "temporal_reasoning": "TR", "temporal": "TR",
    "abstention": "ABS",
    "contradiction_resolution": "CR",
    "event_ordering": "EO",
    "instruction_following": "IF",
    "preference_following": "PF",
    "summarization": "SUM",
}

PUBLISHED_SOTA = {
    "100K": {"Hindsight": 73.4, "Honcho": 63.0, "Mnemosyne v3": 65.2, "LIGHT": 35.8, "RAG": 32.3},
    "500K": {"Hindsight": 71.1, "Honcho": 64.9, "LIGHT": 35.9, "RAG": 33.0},
    "1M":   {"Hindsight": 73.9, "Honcho": 63.1, "LIGHT": 33.6, "RAG": 30.7},
    "10M":  {"Hindsight": 64.1, "Honcho": 40.6, "LIGHT": 26.6, "RAG": 24.9},
}


# ── LLM Client ──────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.base_url = DEEPSEEK_BASE_URL.rstrip("/")
        self.call_count = 0

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        last_error = None
        for attempt in range(3):
            try:
                return self._call(messages, temperature, max_tokens)
            except Exception as e:
                last_error = str(e)
                if "429" in last_error or "rate" in last_error.lower():
                    time.sleep(15 * (attempt + 1))
                elif attempt < 2:
                    time.sleep(3)
                else:
                    break
        return f"[LLM_ERROR: {last_error[:100]}]"

    def _call(self, messages: list, temperature: float, max_tokens: int) -> str:
        resp = http.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        self.call_count += 1
        return data["choices"][0]["message"].get("content", "")


# ── BEAM Dataset Loader ─────────────────────────────────────────────

def load_beam_conversations(scales: list[str], max_conv: int = None) -> dict:
    from datasets import load_dataset

    data = {}
    for scale in scales:
        print(f"  Loading BEAM {scale}...", flush=True)
        if scale == "10M":
            ds = load_dataset("Mohammadta/BEAM-10M", streaming=True)
            split_name = list(ds.keys())[0]
            conversations = []
            for i, sample in enumerate(ds[split_name]):
                if max_conv and i >= max_conv:
                    break
                conversations.append(_parse_sample(sample, scale, i))
            data[scale] = conversations
        else:
            ds = load_dataset("Mohammadta/BEAM", streaming=False)
            if scale not in ds:
                continue
            conversations = []
            for i, sample in enumerate(ds[scale]):
                if max_conv and i >= max_conv:
                    break
                conversations.append(_parse_sample(sample, scale, i))
            data[scale] = conversations
        print(f"    Loaded {len(conversations)} conversations", flush=True)
    return data


def _parse_sample(sample: dict, scale: str, idx: int) -> dict:
    all_messages = []
    chat = sample.get("chat", [])
    for block in chat:
        if isinstance(block, list):
            for msg in block:
                if isinstance(msg, dict):
                    all_messages.append({
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                    })

    pq_raw = sample.get("probing_questions", "{}")
    if isinstance(pq_raw, str):
        try:
            probing = ast.literal_eval(pq_raw)
        except Exception:
            probing = {}
    else:
        probing = pq_raw

    all_questions = []
    for ability, questions in probing.items():
        if isinstance(questions, list):
            for q in questions:
                if isinstance(q, dict):
                    all_questions.append({
                        "ability": ability,
                        "ability_short": ABILITY_MAP.get(ability, ability[:3].upper()),
                        "question": q.get("question", ""),
                        "ideal_answer": q.get("ideal_response", q.get("ideal_answer", "")),
                        "rubric": q.get("rubric", []),
                    })

    return {
        "id": sample.get("conversation_id", str(idx)),
        "messages": all_messages,
        "questions": all_questions,
        "scale": scale,
    }


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
            message_fts_top_k=15,  # raw message keyword hits — critical for BEAM
            fts_top_k=10,
            graph_top_k=10,
        )
        llm = OpenAICompatibleClient(
            api_key=self.api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )
        self.hy = HyMem(cfg, llm=llm)

    def close(self):
        if self.hy:
            self.hy.close()
            self.hy = None

    def ingest(self, session_id: str, messages: list[dict]):
        """Ingest messages directly into HyMem."""
        # Log all messages in one batch
        log_entries = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content.strip():
                continue
            log_entries.append((role, content))

        if log_entries:
            self.hy.log_messages(session_id, log_entries)

        return {"total_msgs": len(messages), "total_chars": sum(len(m.get("content", "")) for m in messages)}

    def dream_and_wait(self, timeout: int = 180):
        """Run dream cycle and wait for completion."""
        start = time.time()
        dream_hy = self.hy.fork()
        try:
            dream_hy.dream()
        finally:
            dream_hy.close()
        elapsed = time.time() - start
        print(f"      Dream completed in {elapsed:.0f}s", flush=True)

    def search(self, session_id: str, query: str, ability: str = None, top_k: int = 10) -> tuple[list[dict], int]:
        """Search HyMem for the given query. Returns (memories, total_message_matches)."""
        TASK_RECALL = {"IF", "MR", "EO", "SUM", "TR"}

        try:
            result = self.hy.augment(query, session_id=session_id, ability=ability)
        except Exception as e:
            print(f"    [DEBUG] augment error: {e}", flush=True)
            return [], 0

        total_matches = getattr(result, "total_message_matches", 0)

        # Collect all sources into separate lists
        graph_facts = []
        for fact in (getattr(result, "graph_facts", None) or []):
            graph_facts.append({
                "content": f"{fact.subject} {fact.predicate} {fact.object}",
                "type": "graph_fact",
                "confidence": getattr(fact, "confidence", 0.5),
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

        message_hits = []
        for hit in (getattr(result, "message_hits", None) or []):
            text = getattr(hit, "text", "")[:600]
            role = getattr(hit, "role", "unknown")
            score = getattr(hit, "score", 0.0)
            if text.strip():
                message_hits.append({
                    "content": f"[{role}] {text}",
                    "type": "message_hit",
                    "confidence": 0.7,
                    "_score": score,
                })

        procedure_hits = []
        for proc in (getattr(result, "procedures", None) or []):
            name = getattr(proc, "name", "")
            desc = getattr(proc, "description", "")[:400]
            steps = getattr(proc, "steps", [])
            step_text = "; ".join(s.get("description", "")[:80] for s in (steps or [])[:5])
            content = f"Procedure: {name}: {desc}"
            if step_text:
                content += f" [Steps: {step_text}]"
            if content.strip():
                procedure_hits.append({
                    "content": content[:600],
                    "type": "procedure",
                    "confidence": 0.75,
                })

        episode_hits = []
        for ep in (getattr(result, "episodes", None) or []):
            title = getattr(ep, "title", "")
            summary = getattr(ep, "summary", "")
            content = f"{title}: {summary}" if title else summary
            if content.strip():
                episode_hits.append({
                    "content": content[:500],
                    "type": "episode",
                    "confidence": 0.8,
                })

        recent = []
        for msg in (getattr(result, "recent_turns", None) or []):
            content = getattr(msg, "content", str(msg))
            if content.strip():
                recent.append({
                    "content": content[:500],
                    "type": "recent",
                    "confidence": 0.4,
                })

        # ── Order by ability ──────────────────────────────────────────
        # Task-recall abilities (IF/MR/EO/SUM): the question is "what did
        # the user do / what steps / in what order / summarize" — raw
        # message hits and procedures are the most relevant sources.
        # Graph facts (cross-session tool preferences) actively hurt here
        # because they float above the answer-carrying messages.
        #
        # MR/TR: also task-recall — need messages first for counting and
        # temporal reasoning. Graph facts are cross-session noise.
        #
        # Knowledge/preference abilities (IE/KU/CR/PF/ABS): the
        # question is about facts, preferences, or contradictions — graph
        # facts are the most relevant, so they come first, but we
        # interleave by confidence so message_hits aren't pushed out of
        # the context window entirely.
        if ability in TASK_RECALL:
            # MR aggregation path (opt-in, cap>0): counting mode still works
            if ability == "MR" and total_matches > 0:
                message_hits.sort(key=lambda h: h["_score"])  # lower BM25 = better
                memories = [{
                    "content": f"[HyMem counted {total_matches} distinct user messages matching this question]",
                    "type": "system",
                    "confidence": 1.0,
                }]
                memories += message_hits + procedure_hits + episode_hits + fts_hits + graph_facts
                return [m for m in memories if m.pop("_score", None) or True][:min(top_k * 6, 120)], total_matches

            # Task-recall: messages > procedures > then interleave rest by confidence
            memories = message_hits + procedure_hits
            rest = episode_hits + fts_hits + graph_facts + recent
            rest.sort(key=lambda m: -m.get("confidence", 0))
            memories += rest
        else:
            # Knowledge/preference: graph facts first, then message hits
            # (keyword-relevant raw turns — critical for IE/PF), then
            # everything else interleaved by confidence.
            # Procedures come last here — they're mostly noise for
            # knowledge questions and only relevant for IF.
            graph_facts.sort(key=lambda m: -m.get("confidence", 0))
            memories = graph_facts + message_hits + episode_hits + fts_hits + procedure_hits + recent
            # Sort: graph_facts stay first, message_hits next, rest by confidence
            memories.sort(key=lambda m: (
                m["type"] != "graph_fact",
                0 if m["type"] == "message_hit" else 1,
                -m.get("confidence", 0),
            ))

        memories = memories[:top_k]

        # Strip internal _score field before returning
        for m in memories:
            m.pop("_score", None)

        return memories, total_matches


def answer_question(llm: LLMClient, memories: list[dict], question: str, ability: str, total_matches: int = 0) -> str:
    """Ask LLM to answer based on retrieved memories."""
    # Build context from memories
    parts = []
    total_chars = 0

    # MR counting: HyMem already counted distinct user turns + deduped.
    # total_message_matches is the candidate answer — LLM just verifies it.
    if ability == "MR" and total_matches > 0:
        parts.append(f"[HyMem counted {total_matches} distinct user messages "
                      f"matching your question (assistant echoes excluded, "
                      f"restatements deduped). Verify this count against the "
                      f"evidence below and return the final number.]\n")

    # MR and TR: need more context — cross-session data is fractured
    context_limit = MAX_CONTEXT_CHARS * 2 if ability in ("MR", "TR") else MAX_CONTEXT_CHARS

    for m in memories:
        content = m["content"]
        if total_chars + len(content) > context_limit:
            break
        tag = "[FACT]" if m["type"] == "graph_fact" else "[MEM]"
        parts.append(f"{tag} {content}")
        total_chars += len(content) + len(tag) + 2

    context = "\n".join(parts) if parts else "No relevant memories found."

    # Ability-aware answering prompts
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


def judge_answer(llm: LLMClient, question: str, ideal: str, rubric: list, ai_answer: str) -> dict:
    if not rubric:
        return {"score": 0.0, "scores": []}

    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric))
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"QUESTION: {question}\n\n"
            f"IDEAL ANSWER: {ideal}\n\n"
            f"AI ANSWER: {ai_answer}\n\n"
            f"RUBRIC (score 0-1 each):\n{rubric_text}\n\n"
            f"Return JSON with 'scores' list and 'total_score'."
        )},
    ]

    raw = llm.chat(messages, temperature=0.0, max_tokens=512)
    try:
        json_match = re.search(r'\{[^}]+\}', raw.replace('\n', ' '))
        if json_match:
            result = json.loads(json_match.group())
            scores = result.get("scores", [])
            total = sum(scores) / len(scores) if scores else 0.0
            return {"score": total, "scores": scores}
    except Exception:
        pass
    return {"score": 0.0, "scores": []}


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_conversation(llm: LLMClient, judge_llm: LLMClient, hy: HyMemAdapter,
                          conv: dict, top_k: int) -> dict:
    conv_id = conv["id"]
    scale = conv["scale"]
    session_id = f"beam-{scale}-{conv_id}"

    # Ingest
    print(f"  Ingesting conv {conv_id} ({len(conv['messages'])} msgs)...", flush=True)
    stats = hy.ingest(session_id, conv["messages"])

    # Dream
    print(f"    Running dream cycle...", flush=True)
    hy.dream_and_wait()

    # Evaluate questions
    results = []
    for qi, q in enumerate(conv["questions"]):
        print(f"    [{qi+1}/{len(conv['questions'])}] {q['ability_short']}: {q['question'][:80]}...", flush=True)

        memories, total_matches = hy.search(session_id, q["question"], ability=q["ability_short"], top_k=top_k * 3)
        ai_answer = answer_question(llm, memories, q["question"], q["ability_short"], total_matches)
        judgment = judge_answer(judge_llm, q["question"], q["ideal_answer"], q["rubric"], ai_answer)

        results.append({
            "ability": q["ability_short"],
            "question": q["question"][:200],
            "score": judgment["score"],
        })
        metrics = f"{len(memories)} memories"
        if total_matches > 0:
            metrics += f" (total matches: {total_matches})"
        print(f"      Score: {judgment['score']:.2f} ({metrics})", flush=True)

    return {"scale": scale, "conversation_id": conv_id, "stats": stats, "results": results}


def compute_scores(all_results: list[dict]) -> dict:
    by_scale_ability = defaultdict(lambda: defaultdict(list))
    for conv_result in all_results:
        scale = conv_result["scale"]
        for r in conv_result["results"]:
            by_scale_ability[scale][r["ability"]].append(r["score"])

    summary = {}
    for scale, abilities in by_scale_ability.items():
        scale_scores = {}
        all_sc = []
        for ab, scores in abilities.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            scale_scores[ab] = {"avg": avg, "count": len(scores)}
            all_sc.extend(scores)
        overall = sum(all_sc) / len(all_sc) if all_sc else 0.0
        scale_scores["OVERALL"] = {"avg": overall, "count": len(all_sc)}
        summary[scale] = scale_scores
    return summary


def print_report(ability_summary: dict, metadata: dict):
    ABILITIES = ["IE", "MR", "KU", "TR", "ABS", "CR", "EO", "IF", "PF", "SUM"]

    print(f"\n{'='*80}")
    print(f"  HYMEM BEAM END-TO-END RESULTS")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  LLM: {metadata.get('answer_model')} / Judge: {metadata.get('judge_model')}")
    print(f"  Conversations: {metadata.get('sample_size')}")
    print(f"  Top-K: {metadata.get('top_k', DEFAULT_TOP_K)}")
    print(f"{'='*80}")

    print(f"\n  Per-Ability Scores:")
    header = f"  {'Scale':<8} {'OVERALL':>8}"
    for ab in ABILITIES:
        header += f" {ab:>6}"
    print(header)
    print(f"  {'-'*len(header)}")

    for scale in sorted(ability_summary.keys()):
        scores = ability_summary[scale]
        overall = scores.get("OVERALL", {}).get("avg", 0.0)
        line = f"  {scale:<8} {overall*100:>7.1f}%"
        for ab in ABILITIES:
            s = scores.get(ab, {}).get("avg", 0.0)
            line += f" {s*100:>5.1f}%"
        print(line)

    print(f"\n  SOTA Comparison:")
    for scale in sorted(ability_summary.keys()):
        hy_overall = ability_summary[scale].get("OVERALL", {}).get("avg", 0.0)
        sota = PUBLISHED_SOTA.get(scale, {})
        print(f"  {scale}:  HyMem {hy_overall*100:.1f}%", end="")
        for name, score in sota.items():
            print(f"  |  {name} {score:.1f}%", end="")
        print()


# ── Main ────────────────────────────────────────────────────────────

def main():
    global DEEPSEEK_API_KEY

    parser = argparse.ArgumentParser(description="HyMem BEAM Benchmark (direct API)")
    parser.add_argument("--scales", default=DEFAULT_SCALE)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--answer-model", default=ANSWER_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()

    # Resolve API key
    DEEPSEEK_API_KEY = args.api_key or os.environ.get("HYMEM_LLM_API_KEY", "")
    if not DEEPSEEK_API_KEY:
        # Try config.yaml
        config_path = Path("/home/node/.hermes/config.yaml")
        if config_path.exists():
            for line in config_path.read_text().split("\n"):
                s = line.strip()
                if s.startswith("HYMEM_LLM_API_KEY:"):
                    DEEPSEEK_API_KEY = s.split(":", 1)[1].strip().strip('"').strip("'")
                    break
    if not DEEPSEEK_API_KEY:
        print("ERROR: No API key. Set --api-key, HYMEM_LLM_API_KEY env var, or ensure config.yaml has it.")
        sys.exit(1)

    print(f"API key: ...{DEEPSEEK_API_KEY[-4:]}", flush=True)  # confirm suffix

    scales = [s.strip() for s in args.scales.split(",")]
    max_conv = args.sample if args.sample > 0 else None
    top_k = args.top_k

    print(f"\nHyMem BEAM Benchmark (direct API)")
    print(f"  Scales: {scales}")
    print(f"  Max conversations: {max_conv or 'all'}")
    print(f"  Top-K: {top_k}")
    print(f"  Answer model: {args.answer_model}")
    print(f"  Judge model: {args.judge_model}")

    # Temp DB
    tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-beam-"))
    db_path = tmp_dir / "hymem.sqlite"
    print(f"\nTemp DB: {db_path}\n")

    # Initialize HyMem
    hy = HyMemAdapter(db_path, api_key=DEEPSEEK_API_KEY)
    hy.open()

    # LLM clients
    answer_llm = LLMClient(args.answer_model, DEEPSEEK_API_KEY)
    judge_llm = LLMClient(args.judge_model, DEEPSEEK_API_KEY)

    # Load data
    print("Loading BEAM dataset...", flush=True)
    conversations = load_beam_conversations(scales, max_conv)
    total_convs = sum(len(v) for v in conversations.values())
    total_questions = sum(len(c["questions"]) for v in conversations.values() for c in v)
    print(f"  Total: {total_convs} conversations, {total_questions} questions\n")

    # Evaluate
    all_results = []
    start_time = time.time()

    for scale in scales:
        if scale not in conversations:
            continue
        print(f"Evaluating {scale} ({len(conversations[scale])} conversations)...", flush=True)
        for ci, conv in enumerate(conversations[scale]):
            print(f"  [{ci+1}/{len(conversations[scale])}] Conv {conv['id']}", flush=True)
            result = evaluate_conversation(answer_llm, judge_llm, hy, conv, top_k)
            all_results.append(result)
            print()

    elapsed = time.time() - start_time
    print(f"Evaluation complete in {elapsed:.0f}s ({answer_llm.call_count} answer calls, {judge_llm.call_count} judge calls)")

    # Report
    summary = compute_scores(all_results)
    print_report(summary, {
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "sample_size": max_conv,
        "top_k": top_k,
    })

    # Save
    results_path = _repo_root.parent / "hymem_beam" / "results.json"
    results_path.parent.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "scales": scales,
            "sample": max_conv,
            "top_k": top_k,
            "elapsed_s": elapsed,
            "answer_calls": answer_llm.call_count,
            "judge_calls": judge_llm.call_count,
        },
        "summary": {scale: {ab: data["avg"] for ab, data in abilities.items()}
                     for scale, abilities in summary.items()},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    hy.close()

    if not args.keep_db:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Temp DB cleaned up.")


if __name__ == "__main__":
    main()
