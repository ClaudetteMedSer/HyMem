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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Add HyMem to path
# Ensure the HyMem package is importable (repo root is two levels up from benchmarks/)
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

import requests as http

# ── Config ─────────────────────────────────────────────────────────

DEFAULT_SCALE = "100K"
DEFAULT_SAMPLE = 3
DEFAULT_TOP_K = 10
MAX_CONTEXT_CHARS = 8000

# DeepSeek API
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ANSWER_MODEL = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"

# Recency-conflict resolution — imported from the LME adapter's dating fix.
# When a fact was UPDATED over time, the model can prefer the newest *value-bearing*
# statement. A later turn that only mentions the topic without restating the value
# does NOT override an earlier one that does.
RECENCY_CONFLICT_CLAUSE = (
    "\nSome memories are stamped with their date, e.g. [MEM 2023-11-30]. When the same fact "
    "appears with different values at different dates, use the value from the MOST RECENT memory "
    "that actually states that value — a later memory that only mentions the topic without giving "
    "the value does NOT override an earlier one that does."
)

ANSWERING_SYSTEM_PROMPT = ("You are an AI assistant answering questions based on retrieved memories.\n"
"Answer the question concisely using ONLY the provided context.\n"
"If the context doesn't contain the answer, say \"I don't have enough information to answer this question.\"\n"
"Do not make up information. Do not use outside knowledge." + RECENCY_CONFLICT_CLAUSE)

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
Memories tagged [MEM YYYY-MM-DD] are listed in chronological order, earliest first.
Carefully scan ALL the context for relevant dates, times, and event mentions.
Calculate the answer from the evidence provided. For dated events, compute durations precisely.
If you cannot determine the answer from the context, say "I don't have enough information."
Do not make up dates or events."""

# CR rubrics grade on SURFACING the contradiction, the opposite of the default
# prompt's recency clause (which coaches silently preferring the newest value —
# right for KU, wrong here). CR therefore gets its own prompt, additively; the
# default prompt and its clause are unchanged for KU/IE/ABS.
ANSWERING_CR_PROMPT = """You are an AI assistant answering questions based on retrieved memories.
The user's statements in the context may CONTRADICT each other. Scan the context for conflicting values or claims about the question topic. When you find a conflict, do NOT silently pick one side: state both versions explicitly (with their [MEM] dates when available), point out that they contradict each other, and then say which one you consider current and why — usually the most recent value-bearing statement.
If there is no contradiction, answer normally using ONLY the context.
If the context contains nothing relevant, say "I don't have enough information to answer this question."
Do not make up information."""

ANSWERING_EO_PROMPT = """You are an AI assistant answering questions about the order in which events happened.
Memories tagged [MEM YYYY-MM-DD] are listed in chronological order, earliest first; use those dates and that ordering as evidence. Watch for events RECOUNTED later than they happened — order events by when they occurred, not by when they were mentioned.
Identify every event the question asks about, then answer with the ordering — as a numbered list when a sequence is requested.
Base the ordering ONLY on the provided context. If some relevant event is missing from the context, say so, but still order the events that ARE present rather than refusing to answer."""

ANSWERING_IF_PROMPT = """You are an AI assistant answering questions based on retrieved memories.
The user has previously given standing instructions about HOW answers must be presented (for example: use code blocks with syntax highlighting, include version numbers, include numeric codes, follow a specific structure). Scan the context for any such instructions and FOLLOW them exactly in your answer.
Reproduce technical specifics — commands, codes, versions, names — verbatim from the context. Completeness and required formatting matter more than brevity.
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Do not make up information."""

ANSWERING_SUM_PROMPT = """You are an AI assistant summarizing past conversations from retrieved memories.
Write a comprehensive, specific summary of everything in the context that is relevant to the question: cover every distinct topic, decision, and outcome, and include concrete details — names, numbers, dates, versions — rather than generalities.
Summarize whatever relevant material is present even if it looks incomplete; only say "I don't have enough information to answer this question" if the context contains nothing relevant at all.
Do not add information that is not in the context."""

# NB (2026-06-14): a dedicated procedural KU prompt was A/B'd and REGRESSED KU hard
# (45%→20%, IE/ABS/OVERALL flat — additive design held, the regression was all KU).
# DeepSeek executes the simple shared RECENCY_CONFLICT_CLAUSE ("use the latest value")
# BETTER than an explicit "decide what's asked → find direct statements → check
# recency → ignore different-kind mentions" procedure: the extra reasoning steps make
# it abstain/mispick. So KU intentionally falls through to ANSWERING_SYSTEM_PROMPT —
# do NOT re-add a procedural KU prompt. The spoiler-split headroom is real but it's
# retrieval/selection-side, not promptable on this model.

# Abilities not listed here fall through to ANSWERING_SYSTEM_PROMPT.
ANSWERING_PROMPTS = {
    "PF": ANSWERING_PREFERENCE_PROMPT,
    "MR": ANSWERING_MR_PROMPT,
    "TR": ANSWERING_TR_PROMPT,
    "CR": ANSWERING_CR_PROMPT,
    "EO": ANSWERING_EO_PROMPT,
    "IF": ANSWERING_IF_PROMPT,
    "SUM": ANSWERING_SUM_PROMPT,
}

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


# ── LLM Client ────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str, api_key: str, base_url: str = DEEPSEEK_BASE_URL):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
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


# ── Answer-model provider registry ────────────────────────────────────────
# The BEAM ANSWERER is swappable to isolate the answer-side ceiling (KU/CR/EO
# all fail on context that is already present, 2026-06) from extraction/assembly.
# ONLY the answerer changes here: extraction + dream stay on DeepSeek
# (HyMemAdapter, untouched), and the JUDGE stays DeepSeek always — swapping the
# grader would move scores without moving capability, breaking the A/B. Each
# provider exposes an OpenAI-compatible /chat/completions endpoint, so the same
# LLMClient payload works. Spec form is "provider:model"
# (e.g. "gemini:gemini-2.5-flash"); a bare model name ("deepseek-chat") stays on
# DeepSeek for back-compat with the old --answer-model.
ANSWER_PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", ("HYMEM_LLM_API_KEY", "DEEPSEEK_API_KEY")),
    "gemini":   ("https://generativelanguage.googleapis.com/v1beta/openai", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "openai":   ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
}


def resolve_answer_provider(spec: str, deepseek_key: str):
    """Map an answer-model spec to (model, base_url, api_key, provider).

    'provider:model' selects a provider and pulls its key from the first set
    env var in that provider's tuple; a bare model name stays on DeepSeek and
    reuses the already-resolved DeepSeek key. Exits if a non-DeepSeek provider
    is selected without its key set."""
    if ":" in spec and spec.split(":", 1)[0] in ANSWER_PROVIDERS:
        provider, model = spec.split(":", 1)
    else:
        provider, model = "deepseek", spec
    base_url, key_envs = ANSWER_PROVIDERS[provider]
    if provider == "deepseek":
        api_key = deepseek_key
    else:
        api_key = next((os.environ[e] for e in key_envs if os.environ.get(e)), "")
        if not api_key:
            print(f"ERROR: answer provider '{provider}' needs one of {key_envs} set.", flush=True)
            sys.exit(1)
    return model, base_url, api_key, provider


# ── BEAM Dataset Loader ───────────────────────────────────────────────

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


def _parse_time_anchor(raw: str | None) -> str | None:
    """BEAM stamps each session block's opening message with a time anchor
    like 'March-15-2024' — the in-world date of that session. Returns an
    ISO date, or None when absent/unparseable (→ ingestion-time fallback)."""
    if not raw:
        return None
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_sample(sample: dict, scale: str, idx: int) -> dict:
    all_messages = []
    chat = sample.get("chat", [])
    for block in chat:
        if isinstance(block, list):
            # One anchor per session block (carried by its first message);
            # it dates every turn of that session.
            block_date = None
            for msg in block:
                if isinstance(msg, dict) and msg.get("time_anchor"):
                    block_date = _parse_time_anchor(msg["time_anchor"])
                    if block_date:
                        break
            for msg in block:
                if isinstance(msg, dict):
                    all_messages.append({
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "date": block_date,
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
                        "question_id": q.get("question_id") or q.get("id") or "",
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


# ── HyMem Integration ───────────────────────────────────────────────────────

class HyMemAdapter:
    """Direct HyMem Python API adapter with isolated temp DB."""

    def __init__(self, db_path: Path, api_key: str = "",
                 facts_enabled: bool | None = None,
                 facts_extraction: bool | None = None):
        self.db_path = db_path
        self.api_key = api_key
        self.facts_enabled = facts_enabled
        self.facts_extraction = facts_extraction
        self.hy = None

    def open(self):
        from hymem import HyMem, HyMemConfig
        from hymem.contrib.openai_client import OpenAICompatibleClient

        # E1 narrative facts (schema v26). None = config default (both ON);
        # --no-facts is the read-side control arm, --no-facts-extraction stops
        # the dream spending a call per session tail (needs a fresh store).
        overrides = {}
        if self.facts_enabled is not None:
            overrides["facts_enabled"] = self.facts_enabled
        if self.facts_extraction is not None:
            overrides["facts_extraction_enabled"] = self.facts_extraction
        cfg = HyMemConfig(
            root=self.db_path.parent,
            message_fts_top_k=15,  # raw message keyword hits — critical for BEAM
            fts_top_k=10,
            graph_top_k=10,
            **overrides,
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
        # Log all messages in one batch, dated with the BEAM session time
        # anchor as the event time — this is what [MEM YYYY-MM-DD] tags and
        # the EO/TR chronological ordering run on. The anchor is date-only,
        # so a per-turn second offset keeps within-session turn order; turns
        # without an anchor fall back to ingestion time.
        log_entries = []
        for i, m in enumerate(messages):
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content.strip():
                continue
            date = m.get("date")
            created_at = None
            if date:
                base = datetime.fromisoformat(f"{date}T12:00:00")
                created_at = (base + timedelta(seconds=i)).isoformat()
            log_entries.append((role, content, created_at))

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

    def search(self, session_id: str, query: str, ability: str = None,
               top_k: int = 10) -> tuple[list[dict], int, list[str]]:
        """Search HyMem for the given query.

        Returns (memories, total_message_matches, narrative_facts). The E1 facts
        tier (schema v26) is returned SEPARATELY from `memories` on purpose: it
        renders as its own block, so it never takes a memories[:top_k] slot, and
        — the BEAM-specific reason — it stays out of the EO/TR re-sorts in
        answer_question, which partition `memories` by type and would otherwise
        file undated facts among the "other" tail they were tuned to demote."""
        TASK_RECALL = {"IF", "MR", "EO", "SUM", "TR"}

        try:
            result = self.hy.augment(query, session_id=session_id, ability=ability)
        except Exception as e:
            print(f"    [DEBUG] augment error: {e}", flush=True)
            return [], 0, []

        total_matches = getattr(result, "total_message_matches", 0)

        # E1 narrative facts (schema v26) — kept OUT of `memories` (see the
        # docstring). Dates ride along: BEAM's KU/EO/TR prompts all reason off
        # [MEM YYYY-MM-DD] stamps, and a fact_date is an EVENT date rather than
        # the session time_anchor those stamps carry.
        narrative_facts = []
        for nf in (getattr(result, "facts", None) or []):
            text = (getattr(nf, "text", "") or "")[:600]
            if text.strip():
                date = getattr(nf, "fact_date", None) or ""
                narrative_facts.append(f"[{date}] {text}" if date else text)

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
            if text.strip():
                message_hits.append({
                    "content": f"[{role}] {text}",
                    "type": "message_hit",
                    "confidence": 0.7,
                    # created_at carried through so the answer context can date-stamp
                    # each turn — the signal the value-aware recency clause relies on.
                    "created_at": getattr(hit, "created_at", "") or "",
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

        # ── Order by ability ──────────────────────────
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
            # MR aggregation path (opt-in, cap>0): counting mode still works.
            # message_hits arrive best-first from augment() in both backends
            # (raw BM25 ascending, reranked combined-score descending) — do not
            # re-sort on the raw score here, the two directions are opposite.
            if ability == "MR" and total_matches > 0:
                memories = [{
                    "content": f"[HyMem counted {total_matches} distinct user messages matching this question]",
                    "type": "system",
                    "confidence": 1.0,
                }]
                memories += message_hits + procedure_hits + episode_hits + fts_hits + graph_facts
                return memories[:min(top_k * 6, 120)], total_matches, narrative_facts

            # Coverage abilities (EO/SUM): the question is "order / summarize
            # EVERYTHING that happened", not "find the turns most similar to the
            # question". Relevance-ranked raw turns systematically surface the
            # planning/meta turns (which lexically match ordering/summary
            # language) over the specific events the rubric grades on — the model
            # then orders/summarizes the wrong events (observed: every EO answer
            # listed planning items, not implementation milestones). Episode
            # summaries are session-level "what happened" coverage at high
            # density-per-char, so lead with them; otherwise message_hits fill
            # every top_k slot and episodes are sliced off entirely. The raw-turn
            # timeline follows as the dating/ordering evidence.
            if ability in ("EO", "SUM"):
                overview = episode_hits[:8]
                rest = fts_hits + graph_facts + recent
                rest.sort(key=lambda m: -m.get("confidence", 0))
                memories = overview + message_hits + procedure_hits + rest
                # Reserve the overview on top of the normal message budget so
                # the slice can't eat it.
                return memories[:len(overview) + top_k], total_matches, narrative_facts

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
            # NOTE: a turn-level recency blend was tried here and REVERTED — it was
            # net −3 (fixed 1 KU zero, broke 4 working cases). The gold turn ASSERTS
            # the update ("updated to April 5") and is usually OLDER than a later turn
            # that merely REFERENCES the stale value ("per the April 1 deadline"), so
            # mention-recency systematically pulls the wrong turn up and slices the
            # assertion. The answerer's recency clause already resolves updates
            # correctly WHEN both values reach it — the real failure is fact-VALIDITY
            # recency (valid_at), not mention order. That's the schema-v15 bi-temporal
            # path (core), not an adapter reorder. See beam_investigation_notes.md.

        memories = memories[:top_k]

        return memories, total_matches, narrative_facts


# Opt-in transcript logging for localizing a category's floor (retrieval vs
# answer-side). Set BEAM_CONTEXT_LOG=/path/to/file.txt to append, per question,
# the EXACT assembled context the model saw (presented order, [MEM date]/[FACT]
# tags intact) + the prediction. Zero cost when the env var is unset; wrapped so
# a logging error can never break a benchmark run. Optionally narrow with
# BEAM_CONTEXT_LOG_ABILITIES=CR,EO to only dump those abilities.
def _log_context(question_id: str, ability: str, system_prompt: str,
                 question: str, context: str, prediction: str) -> None:
    path = os.environ.get("BEAM_CONTEXT_LOG")
    if not path:
        return
    only = os.environ.get("BEAM_CONTEXT_LOG_ABILITIES", "")
    if only and ability not in {a.strip() for a in only.split(",") if a.strip()}:
        return
    try:
        sys_line = (system_prompt or "").strip().splitlines()[0] if system_prompt else ""
        block = (
            "\n================ CONTEXT LOG ================\n"
            f"QID: {question_id or '(none)'}\n"
            f"ABILITY: {ability}\n"
            f"SYSTEM_PROMPT[0]: {sys_line}\n"
            f"QUESTION: {question}\n"
            f"---- CONTEXT ({len(context)} chars, presented order) ----\n"
            f"{context}\n"
            "---- PREDICTION ----\n"
            f"{prediction}\n"
            "============================================\n"
        )
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(block)
    except Exception as e:
        print(f"      [context-log skipped: {type(e).__name__}: {e}]", flush=True)


def answer_question(llm: LLMClient, memories: list[dict], question: str, ability: str,
                    total_matches: int = 0, question_id: str = "",
                    narrative_facts: list[str] | None = None) -> str:
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

    # E1 narrative facts: their own block above the raw turns, never a
    # memories[:top_k] slot and never part of the EO/TR re-sort below. Facts
    # lead, the raw turns stay beneath as the check (the tier's contract in
    # ask(), and the Acme lesson — a summary is never the only copy).
    if narrative_facts:
        parts.append("[NARRATIVE FACTS — self-contained statements extracted "
                     "from past sessions; a leading date is when the fact "
                     "happened. Verify details against the memories below:]\n")
        for nf in narrative_facts:
            parts.append(f"  • {nf}\n")
        parts.append("[END NARRATIVE FACTS]\n")

    # MR/TR: cross-session data is fractured; EO: full timeline must fit;
    # SUM: coverage-graded — all four get the doubled context budget.
    context_limit = MAX_CONTEXT_CHARS * 2 if ability in ("MR", "TR", "EO", "SUM") else MAX_CONTEXT_CHARS

    # EO/TR: dated turns read as a timeline — chronological, earliest first.
    # search() already picked the survivors by relevance; display order is the
    # only ordering signal the answer model gets, and it cannot sort shuffled
    # snippets itself (every EO question failed that way).
    if ability == "EO":
        # The answer model orders events by PRESENTATION ORDER, not by reading the
        # [MEM] dates (confirmed: 14/20 EO failures followed context order). So
        # presentation order MUST be chronological. Earlier this block led with the
        # UNDATED RAPTOR episode/aggregation nodes (for coverage) — but leading with
        # undated summaries put non-timeline entries at the front and the model
        # ordered by them, sabotaging the very signal we sorted. Lead with the
        # date-sorted raw-turn timeline so following presentation order IS correct;
        # demote the undated episodes BELOW it (kept for coverage, not truncated out
        # by EO's doubled budget) and any other undated tiers last.
        episodes = [m for m in memories if m["type"] == "episode"]
        dated = sorted(
            (m for m in memories if m["type"] != "episode" and m.get("created_at")),
            key=lambda m: m["created_at"],
        )
        other = [m for m in memories
                 if m["type"] != "episode" and not m.get("created_at")]
        memories = dated + episodes + other
    elif ability == "TR":
        dated = sorted(
            (m for m in memories if m.get("created_at")),
            key=lambda m: m["created_at"],
        )
        memories = dated + [m for m in memories if not m.get("created_at")]

    for m in memories:
        content = m["content"]
        if total_chars + len(content) > context_limit:
            break
        # Date-stamp raw turns (the only tier carrying created_at) so the recency-
        # conflict clause can prefer the newest value-bearing statement. FACT/fts/
        # episode tiers stay undated (graph dating deferred).
        if m["type"] == "graph_fact":
            tag = "[FACT]"
        else:
            date10 = (m.get("created_at") or "")[:10]
            tag = f"[MEM {date10}]" if (m["type"] == "message_hit" and date10) else "[MEM]"
        parts.append(f"{tag} {content}")
        total_chars += len(content) + len(tag) + 2

    context = "\n".join(parts) if parts else "No relevant memories found."

    # Ability-aware answering prompts
    system_prompt = ANSWERING_PROMPTS.get(ability, ANSWERING_SYSTEM_PROMPT)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"},
    ]

    prediction = llm.chat(messages, temperature=0.0, max_tokens=1024)
    _log_context(question_id, ability, system_prompt, question, context, prediction)
    return prediction


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


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_conversation(llm: LLMClient, judge_llm: LLMClient, hy: HyMemAdapter,
                          conv: dict, top_k: int) -> dict:
    conv_id = conv["id"]
    scale = conv["scale"]
    session_id = f"beam-{scale}-{conv_id}"

    # Ingest
    print(f"  Ingesting conv {conv_id} ({len(conv['messages'])} msgs)...", flush=True)
    stats = hy.ingest(session_id, conv["messages"])
    print(f"    Ingested: {stats['total_msgs']} msgs, {stats['total_chars']} chars", flush=True)

    # Dream
    print(f"  Dreaming...", flush=True)
    hy.dream_and_wait()

    # Evaluate each question
    results = []
    for qi, q in enumerate(conv["questions"]):
        ability = q["ability_short"]
        question = q["question"]
        print(f"    [{qi+1}/{len(conv['questions'])}] {ability}: {question[:100]}...", flush=True)

        # Search. The ×3 widens the answer context to three times the per-run
        # --top-k (30 memories at the default 10) — the v16 baseline width.
        # dea8d94's rewrite of this loop silently dropped the multiplier; the
        # 10-memory runs that followed scored ~11pp below the 30-memory ones
        # (KU fell from 70-83% to 0-17%), so this width is load-bearing.
        memories, total_matches, narrative_facts = hy.search(
            session_id, question, ability=ability, top_k=top_k * 3)
        print(f"      {len(memories)} memories", end="")
        if total_matches > 0:
            print(f" (total matches: {total_matches})", end="")
        if narrative_facts:
            print(f" (facts: {len(narrative_facts)})", end="")
        print()

        # Answer
        answer = answer_question(llm, memories, question, ability, total_matches,
                                 question_id=q.get("question_id", ""),
                                 narrative_facts=narrative_facts)

        # Judge
        judge_result = judge_answer(judge_llm, question, q["ideal_answer"], q["rubric"], answer)
        print(f"      Score: {judge_result['score']:.2f}")

        results.append({
            "ability": ability,
            "question": question,
            "answer": answer,
            "ideal_answer": q["ideal_answer"],
            "rubric": q["rubric"],
            "score": judge_result["score"],
            "scores": judge_result["scores"],
        })

    return {
        "conv_id": conv_id,
        "scale": scale,
        "questions": results,
    }


def compute_scores(all_results: list[dict]) -> dict:
    per_scale = defaultdict(lambda: defaultdict(list))
    for conv in all_results:
        scale = conv["scale"]
        for q in conv["questions"]:
            per_scale[scale][q["ability"]].append(q["score"])
            per_scale[scale]["OVERALL"].append(q["score"])

    summary = {}
    for scale, abilities in per_scale.items():
        summary[scale] = {}
        for ab, scores in abilities.items():
            summary[scale][ab] = {
                "avg": sum(scores) / len(scores),
                "count": len(scores),
            }
    return summary


def print_report(summary: dict, config: dict):
    print()
    print("=" * 80)
    print("  HYMEM BEAM END-TO-END RESULTS")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  LLM: {config['answer_model']} / Judge: {config['judge_model']}")
    print(f"  Conversations: {config['sample_size']}")
    print(f"  Top-K: {config['top_k']}")
    print("=" * 80)

    for scale, abilities in summary.items():
        print(f"\n  Scale {scale}:")
        headers = ["IE", "MR", "KU", "TR", "ABS", "CR", "EO", "IF", "PF", "SUM"]
        print("  " + " ".join(f"{h:>7}" for h in headers) + "  OVERALL")
        print("  " + "-" * (7 * len(headers) + 9))
        scores = []
        for h in headers:
            s = abilities.get(h, {}).get("avg", None)
            scores.append(f"{s*100:>6.1f}%" if s is not None else "     —")
        ov = abilities["OVERALL"]["avg"]
        print("  " + " ".join(scores) + f"  {ov*100:>6.1f}%")

    print(f"\n  SOTA Comparison:")
    for scale in summary:
        if scale in PUBLISHED_SOTA:
            ours = summary[scale]["OVERALL"]["avg"] * 100
            print(f"  {scale}:  HyMem {ours:.1f}%  |  ", end="")
            sota_parts = [f"{name} {val}%" for name, val in PUBLISHED_SOTA[scale].items()]
            print("  |  ".join(sota_parts))


def main():
    global DEEPSEEK_API_KEY

    parser = argparse.ArgumentParser(description="HyMem BEAM Benchmark (direct API)")
    parser.add_argument("--scales", default=DEFAULT_SCALE)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--answer-model", default=os.environ.get("BEAM_ANSWER_MODEL", ANSWER_MODEL),
                        help="Answerer spec 'provider:model' (e.g. gemini:gemini-2.5-flash) "
                             "or a bare DeepSeek model. Env: BEAM_ANSWER_MODEL.")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--facts", action=argparse.BooleanOptionalAction, default=None,
                        help="E1 narrative-facts READ side (cfg.facts_enabled). None = "
                             "config default (ON); --no-facts is the paired control arm. "
                             "The tier renders its own [NARRATIVE FACTS] block above the "
                             "raw turns and never takes a memory slot.")
    parser.add_argument("--facts-extraction", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="E1 WRITE side (cfg.facts_extraction_enabled). None = config "
                             "default (ON). Changes what the dream STORES, so it only "
                             "differs on a fresh store — not a read-side A/B knob.")
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
    ans_model, ans_base, ans_key, ans_provider = resolve_answer_provider(args.answer_model, DEEPSEEK_API_KEY)
    print(f"  Answer model: {ans_model} (provider={ans_provider}, base={ans_base})")
    print(f"  Judge model: {args.judge_model} (provider=deepseek)")

    # Temp DB
    tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-beam-"))
    db_path = tmp_dir / "hymem.sqlite"
    print(f"\nTemp DB: {db_path}\n")

    # Initialize HyMem
    hy = HyMemAdapter(db_path, api_key=DEEPSEEK_API_KEY,
                      facts_enabled=args.facts,
                      facts_extraction=args.facts_extraction)
    hy.open()

    # LLM clients
    answer_llm = LLMClient(ans_model, ans_key, base_url=ans_base)
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

    # Save — one file per run, so cross-run comparisons keep their metadata
    # (sample size, top_k, models) instead of each run clobbering the last.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = _repo_root.parent / "hymem_beam" / f"results_{stamp}.json"
    results_path.parent.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "scales": scales,
            "sample": max_conv,
            "top_k": top_k,
            # Effective answer-context width (top_k × 3). Saved explicitly:
            # the June 2026 score step came down to this number changing
            # without any record of it in run output.
            "context_memories": top_k * 3,
            "elapsed_s": elapsed,
            "answer_calls": answer_llm.call_count,
            "judge_calls": judge_llm.call_count,
        },
        "summary": {scale: {ab: data["avg"] for ab, data in abilities.items()}
                     for scale, abilities in summary.items()},
        # Full per-question records (answer, ideal, rubric, judge scores) so
        # runs can be re-judged and diffed post-hoc. An empty "scores" list on
        # a question with a non-empty rubric means the judge reply failed to
        # parse and the 0.0 is an artifact, not a graded zero.
        "conversations": all_results,
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
