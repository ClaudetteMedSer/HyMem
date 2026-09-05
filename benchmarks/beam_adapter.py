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
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

# Add HyMem to path
# Ensure the HyMem package is importable (repo root is two levels up from benchmarks/)
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from benchmarks.strictness import (
    AtomicCheckpoint,
    BenchmarkIntegrityError,
    add_strict_run_arguments,
    aggregate_embedding_usage_snapshots,
    aggregate_usage_snapshots,
    build_manifest,
    code_hash,
    converge_indexing,
    content_hash,
    dataclass_identity,
    embedding_usage_snapshot,
    freeze_calibration,
    load_calibration,
    resolve_checkpoint_path,
    select_protocol_ids,
    usage_snapshot,
    validate_ids,
    write_immutable_artifact,
    write_latest_pointer,
)
from longmemeval_adapter import (
    _detect_ability,
    _detect_ability_safe,
    _render_answer_context,
)

import requests as http

# ── Config ─────────────────────────────────────────────────────────

DEFAULT_SCALE = "100K"
DEFAULT_SAMPLE = 3
DEFAULT_TOP_K = 10
MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_INPUT_TOKENS = 16000
DEFAULT_EMBEDDING_BACKEND = "local-hash"
DEFAULT_LOCAL_EMBEDDING_MODEL = "hymem-local-feature-hash-v1"
DEFAULT_LOCAL_EMBEDDING_DIM = 384
DEFAULT_REMOTE_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_REMOTE_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_REMOTE_EMBEDDING_DIM = 1536
VALID_BEAM_SCALES = ("100K", "500K", "1M", "10M")
VALID_BEAM_MESSAGE_ROLES = frozenset({"user", "assistant"})
RESERVED_CHAT_BODY_KEYS = frozenset({
    "model", "messages", "temperature", "max_tokens",
})

# DeepSeek API
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ANSWER_MODEL = "deepseek-v4-flash"
JUDGE_MODEL = "deepseek-v4-flash"
OFFICIAL_JUDGE_SPEC = "openai:gpt-4.1-mini"
BEAM_UPSTREAM_COMMIT = "b2da22eac88bb0874c64665f13457eb99835774a"
BEAM_OFFICIAL_EVALUATOR_URL = (
    "https://github.com/mohammadtavakoli78/BEAM/blob/"
    f"{BEAM_UPSTREAM_COMMIT}/src/evaluation/compute_metrics.py#L339-L636"
)
BEAM_OFFICIAL_PROMPT_URL = (
    "https://github.com/mohammadtavakoli78/BEAM/blob/"
    f"{BEAM_UPSTREAM_COMMIT}/src/prompts.py#L11547-L11616"
)

# Vendored byte-for-byte from upstream ``unified_llm_judge_base_prompt`` at
# BEAM_UPSTREAM_COMMIT. Upstream invokes this prompt once per rubric criterion
# as a user message, with temperature 0 and gpt-4.1-mini. The probing question
# argument is not substituted by the official evaluator.
OFFICIAL_JUDGE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT
A compliant response must be **on-topic** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""
BEAM_OFFICIAL_JUDGE_PROMPT_HASH = (
    "sha256:" + hashlib.sha256(OFFICIAL_JUDGE_PROMPT.encode("utf-8")).hexdigest()
)

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

# BEAM stores the gold answer under a DIFFERENT key per ability, and behind
# those keys sit three different KINDS of gold. `_parse_sample` used to read
# `q.get("ideal_response", q.get("ideal_answer", ""))`, which resolved for
# abstention and contradiction_resolution and silently produced "" for the
# other EIGHT -- including EO and SUM. Empty gold has been reaching the judge
# as `IDEAL ANSWER: ` (an empty LABELLED field, which asserts the ideal answer
# is blank rather than omitting it) since 145eff8, 2026-06-02.
#
# The field names were the symptom. The defect was the `""` default: a lookup
# that cannot find its value returned a value anyway, and nothing downstream
# asked whether it was real. So the map is explicit, exhaustively tested
# against ABILITY_MAP, and misses are LOUD.
#
# The kinds are tracked because they are not interchangeable:
#   response       -- an answer to the question. Usable as probe gold.
#   summary        -- a gold summary. Usable as probe gold.
#   compliance_spec-- a description of what a correct answer must DO, not an
#                     answer. Scoring a tier's coverage against it measures
#                     whether the tier carries the SPEC's vocabulary, which is
#                     a different quantity; it is recorded and excluded rather
#                     than quietly pooled with the other two.
GOLD_FIELDS = {
    "abstention":               ("ideal_response", "response"),
    "contradiction_resolution": ("ideal_answer", "response"),
    "event_ordering":           ("answer", "response"),
    "information_extraction":   ("answer", "response"),
    "knowledge_update":         ("answer", "response"),
    "multi_session_reasoning":  ("answer", "response"),
    "temporal_reasoning":       ("answer", "response"),
    "summarization":            ("ideal_summary", "summary"),
    "instruction_following":    ("expected_compliance", "compliance_spec"),
    "preference_following":     ("expected_compliance", "compliance_spec"),
}
# Kinds the coverage probe may use as gold. compliance_spec is deliberately out.
PROBE_GOLD_KINDS = frozenset({"response", "summary"})

_ALL_GOLD_KEYS = ("ideal_response", "ideal_answer", "answer", "ideal_summary",
                  "expected_compliance", "gold_answer", "response")
_gold_warnings: set = set()


def _resolve_gold_with_metadata(
    q: dict, ability: str,
) -> tuple[str, str, str | None, str]:
    """Resolve gold and report whether the canonical field matched exactly."""

    field, kind = GOLD_FIELDS.get(ability, (None, None))
    if field:
        text = q.get(field)
        if isinstance(text, (list, tuple)):
            text = " ".join(str(t) for t in text)
        if isinstance(text, str) and text.strip():
            return text, kind, field, "exact"
    present = [
        key for key in _ALL_GOLD_KEYS
        if isinstance(q.get(key), str) and q[key].strip()
    ]
    if present:
        recovered_field = present[0]
        _gold_warnings.add((ability, field, recovered_field))
        print(
            f"  WARN gold-field map miss: ability={ability!r} expected "
            f"{field!r}, recovered from {recovered_field!r}. Update "
            "GOLD_FIELDS -- a recovered field's KIND is a guess.",
            flush=True,
        )
        return str(q[recovered_field]), (kind or "unknown"), recovered_field, "recovered"
    return "", "none", None, "missing"


def _resolve_gold(q: dict, ability: str) -> tuple[str, str]:
    """(text, kind) for one question, loudly rather than silently.

    A miss falls back to scanning every known key -- not to paper over the map,
    but so the WARN names the keys the row actually has. A row that resolves to
    nothing returns kind "none", which is a value the probe and the gold audit
    both check for; it is never an empty string standing in for an answer.
    """
    text, kind, _field, _resolution = _resolve_gold_with_metadata(q, ability)
    return text, kind


# ── LLM Client ────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str, api_key: str, base_url: str = DEEPSEEK_BASE_URL,
                 extra_body: dict | None = None, token_counter=None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.token_usage_available = False
        self._usage_complete = True
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable or None")
        self._token_counter = token_counter
        # Provider-specific request fields (DeepSeek's `thinking` switch is the
        # only one in use). EMPTY BY DEFAULT and only ever set from an explicit
        # flag: an unflagged run must send the same bytes it sent before this
        # plumbing existed, or every prior artifact silently stops being a
        # comparator. That is why this is not the `auto` host-substring gate
        # hymem/contrib/openai_client.py:81-86 uses for the library client --
        # a benchmark cannot afford a request-body change it did not ask for.
        self.extra_body = validate_request_extra_body(extra_body or {})
        # Why the response ENDED, kept from the last call. B2 (2026-09-01) hit
        # a silent-0 whose cause -- the judge scoring 1.0 and then running out
        # of tokens mid-explanation -- was only recoverable by eyeballing the
        # raw text. finish_reason == "length" says the same thing structurally,
        # which is what a gate needs if it is to separate "the plumbing broke"
        # from "the judge ran long" without a human reading prose. Purely
        # additive: it records why a call ended and changes no score and no
        # request byte. Single-threaded client, so plain attribute state.
        self.last_finish_reason = None

    def count_tokens(self, text: str) -> int:
        if self._token_counter is None:
            raise RuntimeError("no trusted tokenizer configured for this model")
        return self._token_counter(text)

    def chat(
        self, messages: list, temperature: float = 0.1,
        max_tokens: int | None = 1024,
    ) -> str:
        # Cleared per call, so a stale value from an earlier row can never be
        # read as this row's -- absence must look like absence.
        self.last_finish_reason = None
        last_error = None
        for attempt in range(3):
            try:
                return self._call(messages, temperature, max_tokens)
            except Exception as e:
                last_error = str(e)
                self._usage_complete = False
                self.token_usage_available = False
                if "429" in last_error or "rate" in last_error.lower():
                    time.sleep(15 * (attempt + 1))
                elif attempt < 2:
                    time.sleep(3)
                else:
                    break
        return f"[LLM_ERROR: {last_error[:100]}]"

    def _call(
        self, messages: list, temperature: float, max_tokens: int | None,
    ) -> str:
        body = {
            "model": self.model, "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        # Only provider-specific extensions reach this merge. Core request
        # identity is immutable and was rejected at construction if supplied.
        body.update(self.extra_body)
        self.request_attempts += 1
        started = time.monotonic()
        try:
            resp = http.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        finally:
            self.total_latency_s += time.monotonic() - started
        usage = data.get("usage") if isinstance(data, dict) else None
        required = ("prompt_tokens", "completion_tokens", "total_tokens")
        valid_usage = isinstance(usage, dict) and all(
            isinstance(usage.get(key), (int, float))
            and not isinstance(usage.get(key), bool)
            and math.isfinite(float(usage[key]))
            and usage[key] >= 0
            for key in required
        )
        if valid_usage:
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]
            self.total_tokens += usage["total_tokens"]
        else:
            self._usage_complete = False
        choice = data["choices"][0]
        # Set BEFORE the empty-content check: the trap's whole signature is
        # content == "" together with finish_reason == "length", so the field
        # has to survive the raise that the empty content triggers.
        self.last_finish_reason = choice.get("finish_reason")
        content = choice["message"].get("content")
        if not (content or "").strip():
            # LME (:384) raises on content IS NONE. Beam raises on EMPTY too,
            # because empty is the shape the trap actually takes: a reasoning
            # model spends max_tokens in reasoning_content and returns
            # content="" with finish_reason="length" -- an HTTP 200, not an
            # error. Returned as-is it reached judge_answer, missed the regex,
            # took the except path, and scored 0.0 indistinguishably from a
            # real 0.0 (lme_runs.db id=53: 0.6% without extra_body vs id=54:
            # 69.8% same day with it). Raising makes it explicit and countable:
            # chat() retries, then surfaces "[LLM_ERROR: empty content ...]".
            raise RuntimeError(
                f"empty content (finish={choice.get('finish_reason')}, "
                f"reasoning={len(choice['message'].get('reasoning_content') or '')} chars)")
        self.call_count += 1
        self.successful_responses += 1
        self.token_usage_available = (
            self.successful_responses > 0 and self._usage_complete
        )
        return content


# ── Answer-model provider registry ────────────────────────────────────────
# The BEAM ANSWERER is swappable to isolate the answer-side ceiling (KU/CR/EO
# all fail on context that is already present, 2026-06) from extraction/assembly.
# ONLY the answerer changes here: extraction + dream use the separately
# manifested HyMem pipeline, and the judge uses the selected protocol/provider.
# Each
# provider exposes an OpenAI-compatible /chat/completions endpoint, so the same
# LLMClient payload works. Spec form is "provider:model"
# (e.g. "gemini:gemini-2.5-flash"); a bare model name ("deepseek-chat") stays on
# DeepSeek for back-compat with the old --answer-model.
ANSWER_PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", ("HYMEM_LLM_API_KEY", "DEEPSEEK_API_KEY")),
    "gemini":   ("https://generativelanguage.googleapis.com/v1beta/openai", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "openai":   ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
}


def validate_request_extra_body(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject fields that could make the wire request disagree with manifest."""

    if not isinstance(value, Mapping):
        raise BenchmarkIntegrityError("request extra_body must be an object")
    collisions = RESERVED_CHAT_BODY_KEYS & set(value)
    if collisions:
        raise BenchmarkIntegrityError(
            "request extra_body cannot override core field(s): "
            f"{sorted(collisions)}"
        )
    return dict(value)


def parse_provider_spec(spec: str) -> tuple[str, str, str]:
    """Resolve provider/model/base without reading credentials."""

    if ":" in spec and spec.split(":", 1)[0] in ANSWER_PROVIDERS:
        provider, model = spec.split(":", 1)
    else:
        provider, model = "deepseek", spec
    if not model.strip():
        raise BenchmarkIntegrityError("model spec must contain a model id")
    return provider, model, ANSWER_PROVIDERS[provider][0]


def is_official_judge_configuration(
    *, protocol: str, provider: str, model: str, base_url: str,
    extra_body: Mapping[str, Any],
) -> bool:
    return bool(
        protocol == "official"
        and provider == "openai"
        and model == "gpt-4.1-mini"
        and base_url.rstrip("/") == "https://api.openai.com/v1"
        and not extra_body
    )


def resolve_answer_provider(
    spec: str, deepseek_key: str, *, role: str = "answer",
):
    """Map an answer-model spec to (model, base_url, api_key, provider).

    'provider:model' selects a provider and pulls its key from the first set
    env var in that provider's tuple; a bare model name stays on DeepSeek and
    reuses the already-resolved DeepSeek key. Exits if a non-DeepSeek provider
    is selected without its key set."""
    provider, model, base_url = parse_provider_spec(spec)
    key_envs = ANSWER_PROVIDERS[provider][1]
    if provider == "deepseek":
        api_key = deepseek_key
    else:
        api_key = next((os.environ[e] for e in key_envs if os.environ.get(e)), "")
        if not api_key:
            print(
                f"ERROR: {role} provider '{provider}' needs one of "
                f"{key_envs} set.", flush=True,
            )
            sys.exit(1)
    return model, base_url, api_key, provider


THINKING_DISABLED = {"thinking": {"type": "disabled"}}


def select_judge_ideal(judge_gold: bool, gold_text: str | None,
                       ideal_answer: str | None) -> str:
    """The text actually sent to the judge as the ideal answer.

    Extracted because both call sites had it inline and NEITHER recorded the
    result. Step 2's artifact stored `ideal_answer` — the dataset field — while
    the judge had scored against `gold_text`, and under `--judge-gold` those
    differ (IF/PF resolve gold from `compliance_spec`). An artifact that records
    a field the judge did not read, under a name that says it did, is the exact
    defect this series retired the June record for.

    The rejudge path is worse: it reparses gold fresh, judges against that, and
    writes back the row's INHERITED `ideal_answer` from the source artifact. So
    comparing `ideal_answer` across rejudge arms compares four copies of one
    field and can never disagree — which is why the re-derivation protocol's
    §4.2 check was vacuous. See that protocol's §10."""
    return ((gold_text or "") if judge_gold else (ideal_answer or ""))


def apply_thinking_default(role: str, model: str, provider: str,
                           absent: bool, obj: dict) -> tuple[dict, bool]:
    """Model-pin pre-reg §6: default v4-flash to thinking-disabled when the
    operator passed no flag at all.

    The match term is `"v4-flash" in model`, NOT `"deepseek" in model`. The
    latter is the library client's gate (`hymem/contrib/openai_client.py:81-86`)
    and it would also fire on `deepseek-chat`, silently adding the flag to the
    alias path, changing A/B byte-identity and retiring the comparator without
    anyone deciding to. The v4-flash term cannot fire on the alias: it fires
    only where the alternative is a run `check_model_pin` refuses, so no
    artifact worth comparing to was ever produced by the path it changes.

    ABSENT is not EMPTY. `--{role}-extra-body ''` or `'{}'` is the operator
    explicitly asking for no extra body, and a convenience must never override
    an explicit statement -- so those keep `{}` and the guard then refuses the
    run, which is the correct outcome for a request that cannot work."""
    if not absent or provider != "deepseek" or "v4-flash" not in model:
        return obj, False
    # Deep, not `dict(...)`: a shallow copy shares the nested dict, so one
    # run mutating its own extra_body would edit every later run's default.
    return copy.deepcopy(THINKING_DISABLED), True


def check_model_pin(role: str, model: str, provider: str, extra_body: dict) -> None:
    """Refuse the two ways a model pin turns into silent empty completions.

    (1) A v4-flash DeepSeek model WITHOUT thinking disabled answers in
        `reasoning_content` and leaves `content` empty. `_rejudge_run` has
        aborted on this since the gold-delta pre-registration, but the normal
        answer/judge path had no guard at all -- a bare
        `--answer-model deepseek-v4-flash` ran straight into it and the run
        looked like a capability result.
    (2) DeepSeek's `thinking` key sent to OpenAI/Gemini is a 400. The ANSWERER
        is provider-swappable (ANSWER_PROVIDERS), so this is reachable by flag
        combination; the judge is DeepSeek-only and cannot hit it.
    """
    thinking = extra_body.get("thinking")
    if provider != "deepseek" and thinking is not None:
        print(f"ERROR: {role} provider {provider!r} rejects DeepSeek's `thinking` key "
              f"(HTTP 400). Drop it from --{role}-extra-body.")
        sys.exit(2)
    if provider == "deepseek" and "v4-flash" in model and \
            (thinking or {}).get("type") != "disabled":
        print(f"ERROR: {role} model {model!r} requires "
              f"--{role}-extra-body '{{\"thinking\": {{\"type\": \"disabled\"}}}}'. "
              "Without it the model writes to reasoning_content, this client reads "
              "content, and every empty read scores 0.")
        sys.exit(2)


def _git(*args: str) -> str | None:
    """Run git inside the repo; stripped stdout, or None if it failed."""
    try:
        r = subprocess.run(("git", "-C", str(_repo_root)) + args,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def resolve_prereg(path: str | None) -> dict | None:
    """Pin an artifact to the spec that authorised it, or refuse to run.

    "A verdict whose spec-hash post-dates its artifact is void by
    construction." That only bites if the check happens BEFORE the spend: a
    spec edited afterwards to fit the numbers no longer matches the hash its
    artifact recorded, and the mismatch is visible to anyone who looks. A spec
    merely *cited* by filename gives none of that, because the file is
    mutable and the citation is not.

    So the spec must be committed and clean at run start. `path=None` is the
    explicit --no-prereg escape for exploratory runs; it returns None, which
    the artifact records as `prereg: null` -- an exploratory run should be
    identifiable as one, not silently indistinguishable from a canonical.

    The three hashes are not interchangeable, and writing comparison logic on
    the wrong one gives a wrong answer:

    - `blob` is the SPEC'S CONTENT identity and the only equality key. Two runs
      under an unedited spec share it however far the repo has moved; any edit
      changes it. This is the field that makes "spec edited to fit the numbers"
      a visible mismatch.
    - `commit` is provenance: the last commit that touched the SPEC PATH, not
      HEAD. Checking it out does not reproduce the run.
    - `code_commit` is HEAD -- the tree to check out to reproduce. It is only
      meaningful because a dirty tree is refused outright (below); recording
      HEAD beside an uncommitted diff would name a tree that never produced
      the artifact, which is worse than recording nothing.
    """
    if path is None:
        return None
    if _git("rev-parse", "--git-dir") is None:
        print("ERROR: --prereg needs a git repo; run with --no-prereg to opt out.")
        sys.exit(2)
    abs_path = Path(path) if os.path.isabs(path) else (_repo_root / path)
    if not abs_path.exists():
        print(f"ERROR: pre-registration {path!r} does not exist.")
        sys.exit(2)
    try:
        rel = str(abs_path.resolve().relative_to(_repo_root))
    except ValueError:
        print(f"ERROR: pre-registration {path!r} is outside the repo, so it cannot be "
              "committed with the code it authorises.")
        sys.exit(2)
    if _git("status", "--porcelain", "--", rel):
        print(f"ERROR: pre-registration {rel!r} is uncommitted or modified. Commit it "
              "BEFORE the run -- a spec that can still change is not a pre-registration.")
        sys.exit(2)
    commit = _git("log", "-1", "--format=%H", "--", rel)
    if not commit:
        print(f"ERROR: {rel!r} has no commit touching it; git cannot date the spec.")
        sys.exit(2)
    # No escape hatch, deliberately. An --allow-dirty flag would let the run
    # record code_commit = HEAD while the tree was HEAD plus an uncommitted
    # diff that nothing captures: not an incomplete field but a WRONG one, and
    # a reader who checks that commit out and re-runs gets a different result.
    # A dirty tree already has an honest path -- --no-prereg, which records
    # prereg: null. If a run is worth banking as canonical, its code is worth
    # committing; if it is not, it is exploratory and should say so.
    if _git("status", "--porcelain", "--untracked-files=no"):
        print("ERROR: tracked files are modified, so no commit names the code that "
              "would produce this artifact. Commit or stash them, or run with "
              "--no-prereg if this is exploratory (recorded as prereg: null). "
              "Untracked files are fine -- only modified tracked files are the "
              "problem, because only they change behaviour without changing HEAD.")
        sys.exit(2)
    return {
        "path": rel,
        "commit": commit,
        "blob": _git("rev-parse", f"HEAD:{rel}"),
        "committed_at": _git("log", "-1", "--format=%cI", "--", rel),
        "code_commit": _git("rev-parse", "HEAD"),
    }


# Preference order for the canary's question. The trap only reproduces when
# the model burns its whole budget reasoning before writing content, so an
# easy question is a canary that passes for the wrong reason. These four
# abilities are the ones that demand multi-step reasoning; the canary picks
# the first question it finds in this order and only falls back to "whatever
# is first" if a sample somehow contains none of them.
CANARY_ABILITIES = ("TR", "MR", "EO", "SUM")


def pick_canary_question(conversations: dict, scales: list) -> dict:
    """The hardest-reasoning question available, not an arbitrary one."""
    questions = [q for sc in scales for conv in conversations.get(sc, [])
                 for q in conv["questions"]]
    if not questions:
        print("ERROR: no questions loaded; nothing to canary.")
        sys.exit(2)
    by_ability = {}
    for q in questions:
        by_ability.setdefault(q["ability_short"], q)
    for ability in CANARY_ABILITIES:
        if ability in by_ability:
            return by_ability[ability]
    return questions[0]


def run_canary(
    role: str, llm: LLMClient, messages: list, max_tokens: int | None,
) -> str:
    """One real-shaped call per client, before the run spends anything.

    A trivial 'say OK' prompt cannot reproduce the trap: the model has to burn
    its whole budget on reasoning before it would have written content. So the
    canary sends a REAL prompt at the path's own max_tokens ceiling, which is
    the same reasoning the rejudge canary was built on -- generalised here from
    the judge to both clients, because both are pinnable.
    """
    raw = llm.chat(messages, temperature=0.0, max_tokens=max_tokens)
    if raw.startswith("[LLM_ERROR"):
        print(f"  CANARY FAILED ({role}): {raw[:200]}")
        sys.exit(1)
    if not raw.strip():
        # Unreachable while _call raises on empty, kept so the canary stays
        # correct on its own terms if that ever loosens.
        print(f"  CANARY FAILED ({role}): empty content. Raw repr: {raw!r}")
        sys.exit(1)
    print(f"  CANARY OK ({role}): {len(raw)} chars on the real {role} path.")
    return raw


def _run_canary_with_checkpoint(
    ledger: AtomicCheckpoint,
    segment_id: str,
    segment_snapshot,
    role: str,
    llm: LLMClient,
    messages: list,
    max_tokens: int | None,
) -> str:
    """Run a canary and durably persist its spend even when it aborts."""

    try:
        return run_canary(role, llm, messages, max_tokens)
    finally:
        ledger.update_execution_segment(
            segment_id, segment_snapshot("running")
        )


def build_canary_messages(
    conversations: dict, scales: list, judge_gold: bool,
    judge_protocol: str = "legacy-custom",
) -> dict:
    """Assemble both canary prompts without calling anything.

    Split out of main() so the ASSEMBLY is testable. Stubbing chat() -- which
    is how run_canary's own unit tests work -- exercises the send and the
    verdict but never the construction, so a wrong constant name or a question
    key that does not exist would survive a green suite and only surface when a
    real run reached this line, after ingestion had already been paid for.
    """
    q = pick_canary_question(conversations, scales)
    ideal = select_judge_ideal(judge_gold, q.get("gold_text"), q["ideal_answer"])
    if judge_protocol == "official":
        judge_messages = lambda ai_answer: _official_judge_messages(
            q["rubric"][0], ai_answer
        )
    elif judge_protocol == "legacy-custom":
        judge_messages = lambda ai_answer: _judge_messages(
            q["question"], ideal, q["rubric"], ai_answer
        )
    else:
        raise BenchmarkIntegrityError(
            f"unknown BEAM judge protocol: {judge_protocol!r}"
        )
    return {
        "ability": q["ability_short"],
        "answer": [
            {"role": "system",
             "content": ANSWERING_PROMPTS.get(q["ability_short"], ANSWERING_SYSTEM_PROMPT)},
            # No retrieved memories: retrieval needs ingestion, which is the
            # expensive half. The trap does not depend on context -- a
            # reasoning model burns its budget on the QUESTION -- which is why
            # pick_canary_question insists on a reasoning-heavy ability rather
            # than trusting whichever question happened to be parsed first.
            {"role": "user",
             "content": f"CONTEXT:\n(canary: no retrieved memories)"
                        f"\n\nQUESTION: {q['question']}\n\nANSWER:"},
        ],
        "judge": judge_messages,
    }


# ── Episode-tier answer-bearing probe (Plan C pre-check) ──────────
# The LME guard for `episode_granularity_enabled` came back an exact tie, and the
# 2026-08-31 keep-db probe explained why: both arms saturate the episode cap, so
# the lever can only change episode CONTENT -- and on LME `gold_in_episodes` was
# 0/9, i.e. the tier is narrative, not answer-bearing. A lever whose only channel
# is the content of a tier that never carries the answer has no path to the score,
# which makes LME structurally uninformative rather than merely underpowered.
#
# BEAM is where that could differ: EO and SUM grade ORDERING and SUMMARISATION,
# and the assembly below already leads those two abilities with `episode_hits[:8]`
# on the observed finding that message hits otherwise slice episodes out entirely.
# So this block measures, per question and per tier, how much of the ideal answer
# the tier actually carries -- BEFORE any A/B is scheduled against it.
#
# The instrument is built to be unable to return a confident zero:
#   * every measure returns None, never 0.0, when it cannot be computed (empty
#     tier, no distinctive gold tokens) -- an unretrieved tier must not report as
#     "retrieved and empty", which is exactly how `recall_tier`'s missing
#     "episode" value read as a result;
#   * the MESSAGE tier is measured identically as a positive control. If message
#     coverage is also ~0 the measure is broken, not the tier. No episode number
#     here means anything without its message counterpart on the same question.


def _norm_text(s: str) -> str:
    """Whitespace-collapse + lowercase for robust substring matching."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# Minimum normalized answer length for containment to carry signal; a 2-char
# string is inside a thousand sentences by chance. Mirrors the constant of the
# same name in longmemeval_adapter.py / fact_probe.py.
_MIN_ANSWER_CHARS = 4

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset("""
a an the and or but if then else than that this these those of in on at to for
from by with without about into over under again further is are was were be been
being am do does did doing have has had having it its he she they them him her his
their we us our you your i me my mine not no nor so too very can could will would
shall should may might must just only own same as up out off down here there also
because while during before after between each few more most other some such any
both all what when where which who whom whose how why
""".split())


def _content_tokens(s: str) -> set[str]:
    """Distinctive lowercase tokens of a string.

    Stopwords and 1-2 char words are dropped as collision-prone; 2-digit numbers
    are KEPT because dates and counts are precisely the answer-bearing content on
    TR/MR, while single digits are not distinctive enough to survive.
    """
    out = set()
    for t in _TOKEN_RE.findall(_norm_text(s)):
        if t in _STOPWORDS:
            continue
        if len(t) > 2 or (t.isdigit() and len(t) == 2):
            out.add(t)
    return out


def _gold_tokens(ideal_answer: str, question: str) -> set[str]:
    """Answer tokens that are NOT already in the question.

    Load-bearing subtraction: retrieval selected these memories BY matching the
    question, so every tier trivially "covers" question terms. Leaving them in
    measures the retriever's input, not the tier's answer content -- the same
    precision-without-selectivity error that closed E4's query-side range boost.
    """
    return _content_tokens(ideal_answer) - _content_tokens(question)


def _answer_in_texts(answer: str, texts: list[str]) -> bool | None:
    """One-directional containment of a SHORT gold answer in any of `texts`.

    Returns None when the answer is too short to be distinctive. BEAM ideal
    answers are frequently full sentences, for which containment is near-
    impossible even when the tier carries the content -- so this is the
    secondary measure here and `_tier_coverage` is the primary one.
    """
    a = _norm_text(answer)
    if len(a) < _MIN_ANSWER_CHARS:
        return None
    return any(a in _norm_text(t) for t in texts if t and t.strip())


def _tier_coverage(gold: set[str], texts: list[str]) -> float | None:
    """Share of the answer's distinctive tokens present anywhere in a tier.

    None (not 0.0) when the tier is empty or no gold token survived filtering:
    "the tier was not retrieved" and "the tier was retrieved and carried nothing"
    are different findings and must not share an encoding.
    """
    if not gold or not texts:
        return None
    seen: set[str] = set()
    for t in texts:
        if t and t.strip():
            seen |= set(_TOKEN_RE.findall(_norm_text(t)))
    if not seen:
        return None
    return round(len(gold & seen) / len(gold), 4)


# Pre-registered decision thresholds. Fixed 2026-08-31, BEFORE any BEAM run of
# this instrument. They are constants executed by the readout rather than prose
# left to whoever reads the table, because a threshold chosen after seeing the
# numbers is not a threshold. Changing one is a NEW pre-registration, not a tweak.
PROBE_MIN_ROWS = 12       # per ability; below this the verdict is n/a, not a result
PROBE_NULL_MARGIN = 2.0   # cov_ep must be >= 2x its own shuffled null
PROBE_CONTROL_SHARE = 0.5 # cov_ep must be >= 0.5x the message control


def _probe_gold(q: dict) -> str:
    """Gold text the coverage probe may use, or "" if this ability has none of
    a usable KIND. Gated on PROBE_GOLD_KINDS so a compliance_spec can never
    silently become the denominator of a coverage number."""
    return q.get("gold_text", "") if q.get("gold_kind") in PROBE_GOLD_KINDS else ""


def _decoy_answer(questions: list[dict], qi: int) -> str:
    """Another question's ideal answer from the SAME conversation.

    Every BEAM question draws on one shared corpus, so a tier will cover some
    answer tokens by vocabulary alone -- which makes a bare cov_ep number
    uninterpretable. Scoring the same tier against an answer it cannot possibly
    contain gives the chance floor. Same ability where one exists, so answer
    STYLE (ordering language, summary language) is controlled too, and picked
    deterministically so the readout is reproducible from the run alone.
    """
    mine = questions[qi]
    same = [q for j, q in enumerate(questions)
            if j != qi and q.get("ability_short") == mine.get("ability_short")
            and _probe_gold(q)]
    pool = same or [q for j, q in enumerate(questions)
                    if j != qi and _probe_gold(q)]
    if not pool:
        return ""
    return _probe_gold(pool[qi % len(pool)])


def _probe_verdict(n_pair: int, cov_ep, cov_msg, null_ep, null_msg) -> str:
    """The pre-registered rule, executed. Returns YES / no / INVALID / n-a."""
    # The None guards are belt-and-braces: the readout hands this only
    # fully-measured row sets. They stay because this is also called directly.
    if n_pair < PROBE_MIN_ROWS or cov_ep is None or cov_msg is None:
        return "n-a"
    # Control sanity first: if the message tier cannot beat its own chance
    # floor, the MEASURE failed on these rows and neither column is evidence.
    if null_msg is not None and cov_msg <= null_msg:
        return "INVALID"
    if null_ep is None:
        return "n-a"
    if cov_ep < PROBE_NULL_MARGIN * null_ep:
        return "no"
    if cov_ep < PROBE_CONTROL_SHARE * cov_msg:
        return "no"
    return "YES"


def episode_probe(memories: list[dict], question: str, ideal_answer: str,
                  decoy_answer: str = "") -> dict:
    """Per-question, per-tier answer-bearing record. Purely additive: reads the
    already-assembled `memories` list (so it measures what actually reached the
    reader, post-slice) and changes nothing about the run.

    `decoy_answer` is another question's ideal answer from the same conversation;
    scoring the same tiers against it gives each column its own chance floor.
    """
    by_tier: dict[str, list[str]] = defaultdict(list)
    for m in memories:
        if isinstance(m, dict) and (m.get("content") or "").strip():
            by_tier[m.get("type") or "?"].append(m["content"])

    ep, msg = by_tier.get("episode", []), by_tier.get("message_hit", [])
    gold = _gold_tokens(ideal_answer, question)
    return {
        # Fired-indicators: which tiers reached the reader at all.
        "n_memories": len(memories),
        "n_episodes": len(ep),
        "n_messages": len(msg),
        "n_procedures": len(by_tier.get("procedure", [])),
        "n_fts": len(by_tier.get("fts_hit", [])),
        "n_graph": len(by_tier.get("graph_fact", [])),
        "n_recent": len(by_tier.get("recent", [])),
        # Denominator, recorded so a coverage of None is readable as
        # "no distinctive gold tokens" rather than an unexplained blank.
        "n_gold_tokens": len(gold),
        # PRIMARY measure + its positive control, same question, same tokens.
        "cov_episodes": _tier_coverage(gold, ep),
        "cov_messages": _tier_coverage(gold, msg),
        # NEGATIVE CONTROL: the same tiers scored against an answer they cannot
        # contain. Coverage above this is tier content; coverage at it is the
        # shared vocabulary of one conversation and means nothing.
        "cov_episodes_null": (
            _tier_coverage(_gold_tokens(decoy_answer, question), ep)
            if decoy_answer else None),
        "cov_messages_null": (
            _tier_coverage(_gold_tokens(decoy_answer, question), msg)
            if decoy_answer else None),
        # SECONDARY: LME-comparable containment, mostly None on long answers.
        "gold_in_episodes": _answer_in_texts(ideal_answer, ep) if ep else None,
        "gold_in_messages": _answer_in_texts(ideal_answer, msg) if msg else None,
    }


def print_episode_probe(all_results: list[dict]) -> None:
    """Per-ability readout that EXECUTES the pre-registered rule.

    The verdict column is computed, not narrated, so the decision cannot drift
    to fit the numbers once they are on screen. Three things must hold for an
    ability to read YES, and all three are constants above:
      * enough paired rows to be a result at all (PROBE_MIN_ROWS);
      * cov_ep clears its own chance floor by PROBE_NULL_MARGIN -- otherwise the
        coverage is one conversation's shared vocabulary, not tier content;
      * cov_ep is at least PROBE_CONTROL_SHARE of the message control.
    A control that cannot beat its own floor reads INVALID: on those rows the
    measure failed, and neither column is evidence either way.
    """
    rows = [(q["ability"], q.get("probe") or {})
            for conv in all_results for q in conv["questions"]]
    rows = [(a, p) for a, p in rows if p]
    if not rows:
        return

    def _pct(v, w=9):
        return f"{'—':>{w}}" if v is None else f"{v*100:>{w - 1}.1f}%"

    def _num(v, w=7):
        return f"{'—':>{w}}" if v is None else f"{v:>{w}.2f}"

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    print()
    print("=" * 80)
    print("  EPISODE-TIER ANSWER-BEARING PRE-CHECK (Plan C)")
    print("  cov = share of answer-specific tokens (question terms removed)")
    print("  carried by a tier. null = same tier vs another question's answer")
    print("  (chance floor). msg = positive control, on the SAME rows.")
    print(f"  Rule: YES iff pair>={PROBE_MIN_ROWS} and cov_ep >= "
          f"{PROBE_NULL_MARGIN:g}x null_ep and cov_ep >= "
          f"{PROBE_CONTROL_SHARE:g}x cov_msg.")
    print("=" * 80)
    print(f"  {'ability':<8}{'pair':>5}{'cov_ep':>9}{'null_ep':>9}"
          f"{'cov_msg':>9}{'null_msg':>10}{'ratio':>7}{'verdict':>9}")
    print("  " + "-" * 74)

    for ability in sorted({a for a, _ in rows}) + ["ALL"]:
        sel = rows if ability == "ALL" else [(a, p) for a, p in rows if a == ability]
        # FULLY-MEASURED rows only: all four numbers the verdict reads, on one
        # identical row set. A control is only a control on the rows it shares
        # with the thing it controls, and that applies to a column against its
        # own chance floor exactly as it applies to the ep/msg ratio -- the
        # floor is IN the decision rule, so a floor averaged over a different
        # subset than the coverage it gates is the same defect somewhere worse.
        # Rows short a decoy (single-question conversation) or short distinctive
        # decoy tokens therefore drop out entirely rather than partially.
        keys = ("cov_episodes", "cov_messages", "cov_episodes_null",
                "cov_messages_null")
        pair = [p for _, p in sel if all(p[k] is not None for k in keys)]
        me = _mean([p["cov_episodes"] for p in pair])
        mm = _mean([p["cov_messages"] for p in pair])
        ne = _mean([p["cov_episodes_null"] for p in pair])
        nm = _mean([p["cov_messages_null"] for p in pair])
        ratio = (me / mm) if (me is not None and mm not in (None, 0)) else None
        # "No usable gold" is not "too few rows". IF/PF carry compliance_spec,
        # which is deliberately excluded from PROBE_GOLD_KINDS, so their pair
        # count is 0 by DESIGN and will be 0 at every sample size. Printing that
        # as n-a invites the reading "underpowered, raise --sample", which would
        # be chasing a number that cannot move. Distinguished by the rows
        # themselves: no gold tokens anywhere, as opposed to no episodes.
        no_gold = bool(sel) and all(p["n_gold_tokens"] == 0 for _, p in sel)
        # No verdict on the pooled row. The rule is per-ability by design, and
        # pooling abilities with different answer SHAPES can clear every
        # criterion while its own components disagree -- a synthetic run with
        # EO=YES and SUM=no pooled to YES. A number that can contradict all of
        # its parts is not a summary of them, and it would get quoted.
        verdict = ("—" if ability == "ALL"
                   else "no-gold" if no_gold
                   else _probe_verdict(len(pair), me, mm, ne, nm))
        sep = "  " + "-" * 74 + "\n" if ability == "ALL" else ""
        print(f"{sep}  {ability:<8}{len(pair):>5}{_pct(me)}{_pct(ne)}"
              f"{_pct(mm)}{_pct(nm, 10)}{_num(ratio)}{verdict:>9}")

    # Secondary: what reached the reader, and the LME-comparable containment
    # number. Kept out of the decision table because containment is near-blind
    # on the long ideal answers EO/SUM carry -- it is a cross-benchmark
    # comparison point, not evidence about BEAM.
    print()
    print(f"  {'ability':<8}{'n':>5}{'ep>0':>6}{'ep/q':>7}{'msg/q':>7}"
          f"{'proc/q':>8}{'contain_ep':>12}")
    print("  " + "-" * 74)
    for ability in sorted({a for a, _ in rows}) + ["ALL"]:
        sel = [p for a, p in rows if ability in (a, "ALL")]
        gi = [p["gold_in_episodes"] for p in sel if p["gold_in_episodes"] is not None]
        sep = "  " + "-" * 74 + "\n" if ability == "ALL" else ""
        print(f"{sep}  {ability:<8}{len(sel):>5}"
              f"{sum(1 for p in sel if p['n_episodes'] > 0):>6}"
              f"{_num(_mean([p['n_episodes'] for p in sel]))}"
              f"{_num(_mean([p['n_messages'] for p in sel]))}"
              f"{_num(_mean([p['n_procedures'] for p in sel]), 8)}"
              f"{_pct(_mean(gi) if gi else None, 12)}")

    print()
    print("  YES on EO/SUM  -> episodes are answer-bearing there; Plan C gets an")
    print("                    instrument and an A/B can be pre-registered on it.")
    print("  no  everywhere -> reproduces the LME finding; Plan C closes on score")
    print("                    grounds battery-wide.")
    print("  INVALID        -> the measure failed on those rows; discard the rows,")
    print("                    not the hypothesis.")
    print("  n-a            -> too few paired rows to be a result. Raise --sample;")
    print("                    do not read the numbers next to it.")
    print("  no-gold        -> no usable gold on this ability (IF/PF carry a")
    print("                    compliance spec, excluded by design). Not a power")
    print("                    problem: --sample cannot move it.")


# ── BEAM Dataset Loader ───────────────────────────────────────────────

# The 10M scale lives in its own HF repo; everything else shares one.
BEAM_REPO_10M = "Mohammadta/BEAM-10M"
BEAM_REPO = "Mohammadta/BEAM"


def beam_repo(scale: str) -> str:
    return BEAM_REPO_10M if scale == "10M" else BEAM_REPO


def resolve_dataset_revisions(scales: list[str], pin: str | None = None) -> dict:
    """Record WHICH dataset a run read, not merely its name.

    `load_dataset("Mohammadta/BEAM")` names a moving target in exactly the way
    `deepseek-chat` does: the host can change what the name resolves to, and
    nothing in the artifact would show it. The rejudge path is already covered
    -- its 160/160 reparse guard aborts if the gold moved -- but a full run has
    no stored baseline to diff against, so for the canonical this is the one
    input nothing witnesses. A revision that cannot be resolved is recorded as
    null rather than omitted: "we do not know" is itself a fact about the run.
    """
    repos = tuple(dict.fromkeys(beam_repo(scale) for scale in scales))
    if pin and len(repos) > 1:
        raise BenchmarkIntegrityError(
            "one --dataset-revision cannot identify commits in both BEAM "
            "repositories; run the repositories separately"
        )
    out = {}
    for scale in scales:
        repo = beam_repo(scale)
        if repo in out:
            continue
        try:
            from huggingface_hub import HfApi
            if pin:
                out[repo] = HfApi().dataset_info(repo, revision=pin).sha
            else:
                out[repo] = HfApi().dataset_info(repo).sha
        except Exception as e:
            if pin:
                raise BenchmarkIntegrityError(
                    "could not resolve explicit BEAM dataset revision "
                    f"{pin!r} for {repo}: {type(e).__name__}: {e}"
                ) from e
            out[repo] = None
            print(f"  WARNING: could not resolve {repo} revision "
                  f"({type(e).__name__}: {e}); recorded as null — this artifact "
                  "cannot witness the dataset it was scored on.", flush=True)
    return out


def validate_dataset_revision_binding(
    revisions: Mapping[str, str | None], *, canonical: bool,
) -> tuple[str, ...]:
    """Return unresolved repositories, rejecting them for canonical runs."""

    unresolved = tuple(sorted(
        repo for repo, revision in revisions.items()
        if not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-fA-F]{40,64}", revision.strip()) is None
    ))
    if unresolved and canonical:
        raise BenchmarkIntegrityError(
            "canonical BEAM run requires resolved dataset revisions; unresolved: "
            f"{list(unresolved)}"
        )
    return unresolved


def _label_blind_conversation_sample(
    conversations: list[dict], max_conv: int | None, *, seed: int, scale: str,
) -> list[dict]:
    """Select conversations deterministically without inspecting QA labels."""

    if max_conv is None or max_conv >= len(conversations):
        return list(conversations)
    if isinstance(max_conv, bool) or not isinstance(max_conv, int) or max_conv <= 0:
        raise BenchmarkIntegrityError("BEAM sample size must be positive or None")
    ranked = sorted(
        enumerate(conversations),
        key=lambda pair: content_hash({
            "seed": seed,
            "scale": scale,
            "source_index": pair[0],
            "conversation_id": pair[1].get("id"),
        }),
    )
    chosen_indices = sorted(index for index, _conv in ranked[:max_conv])
    return [conversations[index] for index in chosen_indices]


def load_beam_conversations(scales: list[str], max_conv: int = None,
                            revision: str | None = None,
                            seed: int | None = None,
                            revisions: Mapping[str, str | None] | None = None) -> dict:
    from datasets import load_dataset

    data = {}
    for scale in scales:
        print(f"  Loading BEAM {scale}...", flush=True)
        repo = beam_repo(scale)
        if revisions is not None and repo not in revisions:
            raise BenchmarkIntegrityError(
                f"BEAM revision binding is missing repository {repo}"
            )
        bound_revision = revisions[repo] if revisions is not None else revision
        if scale == "10M":
            ds = load_dataset(
                BEAM_REPO_10M, streaming=True, revision=bound_revision
            )
            if "10M" not in ds:
                raise BenchmarkIntegrityError(
                    "BEAM-10M dataset does not contain the requested 10M split"
                )
            split_name = "10M"
            conversations = []
            for i, sample in enumerate(ds[split_name]):
                if seed is None and max_conv and i >= max_conv:
                    break
                conversations.append(_parse_sample(sample, scale, i))
        else:
            ds = load_dataset(
                BEAM_REPO, streaming=False, revision=bound_revision
            )
            if scale not in ds:
                raise BenchmarkIntegrityError(
                    f"BEAM dataset does not contain requested split {scale!r}"
                )
            conversations = []
            for i, sample in enumerate(ds[scale]):
                if seed is None and max_conv and i >= max_conv:
                    break
                conversations.append(_parse_sample(sample, scale, i))
        if seed is not None:
            conversations = _label_blind_conversation_sample(
                conversations, max_conv, seed=seed, scale=scale
            )
        data[scale] = conversations
        print(f"    Loaded {len(conversations)} conversations", flush=True)
    return data


OFFICIAL_BEAM_DENOMINATORS = {
    "100K": {"conversations": 20, "questions": 400},
    "500K": {"conversations": 35, "questions": 700},
    "1M": {"conversations": 35, "questions": 700},
    "10M": {"conversations": 10, "questions": 200},
}


def validate_official_denominators(
    conversations: Mapping[str, list[dict]], scales: list[str],
) -> None:
    """Reject silently shortened official BEAM scale loads."""

    for scale in scales:
        expected = OFFICIAL_BEAM_DENOMINATORS[scale]
        rows = conversations.get(scale)
        if rows is None:
            raise BenchmarkIntegrityError(f"BEAM scale {scale} was not loaded")
        questions = sum(len(conv.get("questions", ())) for conv in rows)
        actual = {"conversations": len(rows), "questions": questions}
        if actual != expected:
            raise BenchmarkIntegrityError(
                f"BEAM {scale} denominator mismatch: expected {expected}, got {actual}"
            )
        for conv in rows:
            ability_counts = Counter(
                q.get("ability_short") for q in conv.get("questions", ())
            )
            expected_abilities = {
                ability: 2 for ability in ABILITY_MAP.values()
            }
            if ability_counts != expected_abilities:
                raise BenchmarkIntegrityError(
                    f"BEAM {scale}/{conv.get('id')} ability distribution "
                    f"mismatch: expected {expected_abilities}, got "
                    f"{dict(ability_counts)}"
                )


def _parse_time_anchor(raw: str | None) -> str | None:
    """BEAM stamps each session block's opening message with a time anchor
    like 'March-15-2024' — the in-world date of that session. Invalid input
    returns None so the canonical parser can reject it before ingestion;
    scored runs never fall back to wall-clock time."""
    if not raw:
        return None
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def print_gold_audit(conversations: dict) -> None:
    """Per-ability gold coverage, printed BEFORE any question is scored.

    This is the check whose absence let empty gold run for three months: the
    parse produced "" for 8 of 10 abilities and nothing ever asked whether a
    gold answer was actually there. A count of zero is now impossible to miss,
    and it is printed for every run rather than only when something looks wrong
    -- a check that only runs on suspicion is a check nobody runs.
    """
    stats: dict = defaultdict(lambda: [0, 0, set()])
    for convs in conversations.values():
        for conv in convs:
            for q in conv["questions"]:
                row = stats[q["ability_short"]]
                row[0] += 1
                if str(q.get("gold_text") or "").strip():
                    row[1] += 1
                row[2].add(q.get("gold_kind", "none"))
    if not stats:
        return
    print()
    print("  Gold-answer coverage by ability (GOLD_FIELDS resolution):")
    empty = []
    for ability in sorted(stats):
        n, have, kinds = stats[ability]
        kind = "/".join(sorted(kinds))
        flag = "" if have == n else "   <-- MISSING"
        if have < n:
            empty.append(ability)
        print(f"    {ability:<5} {have:>4}/{n:<4} {kind:<16}{flag}")
    if empty:
        print(f"  WARNING: {', '.join(empty)} have questions with NO gold text. "
              f"The coverage probe has no denominator there and reports n-a; "
              f"check GOLD_FIELDS against the dataset before reading any "
              f"verdict.", flush=True)
    print()


def _parse_sample(sample: dict, scale: str, idx: int) -> dict:
    if not isinstance(sample, dict):
        raise BenchmarkIntegrityError(f"BEAM {scale}/{idx} sample must be an object")
    source_conv_id = sample.get("conversation_id")
    if source_conv_id is None:
        source_conv_id = idx
    if isinstance(source_conv_id, bool) or not isinstance(source_conv_id, (str, int)):
        raise BenchmarkIntegrityError(
            f"BEAM {scale}/{idx} conversation_id must be a string or integer"
        )
    conv_id = str(source_conv_id).strip()
    if not conv_id:
        raise BenchmarkIntegrityError(f"BEAM {scale}/{idx} conversation_id is empty")
    all_messages = []
    chat = sample.get("chat")
    if not isinstance(chat, list) or not chat:
        raise BenchmarkIntegrityError(
            f"BEAM {scale}/{conv_id} chat must be a non-empty list of session blocks"
        )
    for block_index, block in enumerate(chat):
        if not isinstance(block, list) or not block:
            raise BenchmarkIntegrityError(
                f"BEAM {scale}/{conv_id} chat block {block_index} must be a non-empty list"
            )
        # One anchor per session block (carried by its first anchored message);
        # it dates every turn of that session. A provided but unreadable anchor
        # must never silently fall back to the wall clock.
        block_date = None
        anchor_count = 0
        for message_index, msg in enumerate(block):
            if not isinstance(msg, dict):
                raise BenchmarkIntegrityError(
                    f"BEAM {scale}/{conv_id} chat {block_index}/{message_index} "
                    "must be an object"
                )
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(role, str) or not role.strip():
                raise BenchmarkIntegrityError(
                    f"BEAM {scale}/{conv_id} chat {block_index}/{message_index} "
                    "has a malformed role"
                )
            normalized_role = role.strip()
            if normalized_role not in VALID_BEAM_MESSAGE_ROLES:
                raise BenchmarkIntegrityError(
                    f"BEAM {scale}/{conv_id} chat {block_index}/{message_index} "
                    f"has unsupported role {role!r}"
                )
            if not isinstance(content, str):
                raise BenchmarkIntegrityError(
                    f"BEAM {scale}/{conv_id} chat {block_index}/{message_index} "
                    "has non-string content"
                )
            if "time_anchor" in msg and msg["time_anchor"] is not None:
                anchor_count += 1
                raw_anchor = msg["time_anchor"]
                if not isinstance(raw_anchor, str) or not raw_anchor.strip():
                    raise BenchmarkIntegrityError(
                        f"BEAM {scale}/{conv_id} chat {block_index}/{message_index} "
                        "has a malformed time_anchor"
                    )
                parsed_anchor = _parse_time_anchor(raw_anchor)
                if parsed_anchor is None:
                    raise BenchmarkIntegrityError(
                        f"BEAM {scale}/{conv_id} chat {block_index}/{message_index} "
                        f"has an unparseable time_anchor: {raw_anchor!r}"
                    )
                if block_date is not None and parsed_anchor != block_date:
                    raise BenchmarkIntegrityError(
                        f"BEAM {scale}/{conv_id} chat block {block_index} has "
                        "conflicting time anchors"
                    )
                block_date = parsed_anchor
        if anchor_count != 1 or block_date is None:
            raise BenchmarkIntegrityError(
                f"BEAM {scale}/{conv_id} chat block {block_index} must carry "
                "exactly one parseable time_anchor"
            )
        for msg in block:
            all_messages.append({
                "role": msg["role"].strip(),
                "content": msg["content"],
                "date": block_date,
            })
    if not any(message["content"].strip() for message in all_messages):
        raise BenchmarkIntegrityError(
            f"BEAM {scale}/{conv_id} chat contains no message content"
        )

    pq_raw = sample.get("probing_questions", "{}")
    if isinstance(pq_raw, str):
        try:
            probing = ast.literal_eval(pq_raw)
        except Exception as exc:
            raise BenchmarkIntegrityError(
                f"BEAM {scale}/{conv_id} probing_questions is malformed: {exc}"
            ) from exc
    else:
        probing = pq_raw
    if not isinstance(probing, dict):
        raise BenchmarkIntegrityError(
            f"BEAM {scale}/{conv_id} probing_questions must be an object"
        )

    all_questions = []
    for ability, questions in probing.items():
        if ability not in GOLD_FIELDS:
            raise BenchmarkIntegrityError(
                f"BEAM {scale}/{conv_id} has unknown ability key {ability!r}"
            )
        if not isinstance(questions, list):
            raise BenchmarkIntegrityError(
                f"BEAM {scale}/{conv_id}/{ability} questions must be a list"
            )
        for ability_index, q in enumerate(questions):
            if not isinstance(q, dict):
                raise BenchmarkIntegrityError(
                    f"BEAM {scale}/{conv_id}/{ability}/{ability_index} is not an object"
                )
            question = q.get("question", "")
            if not isinstance(question, str) or not question.strip():
                raise BenchmarkIntegrityError(
                    f"BEAM {scale}/{conv_id}/{ability}/{ability_index} has no question"
                )
            rubric = q.get("rubric", [])
            if (
                not isinstance(rubric, list)
                or not rubric
                or any(not isinstance(item, str) or not item.strip() for item in rubric)
            ):
                raise BenchmarkIntegrityError(
                    f"BEAM {scale}/{conv_id}/{ability}/{ability_index} has malformed rubric"
                )
            gold_text, gold_kind, gold_field, gold_resolution = \
                _resolve_gold_with_metadata(q, ability)
            native_id = (
                q.get("question_id")
                if "question_id" in q else q.get("id")
            )
            if native_id is not None:
                valid_native = (
                    isinstance(native_id, str) and bool(native_id.strip())
                ) or (
                    isinstance(native_id, (int, float))
                    and not isinstance(native_id, bool)
                    and math.isfinite(float(native_id))
                )
                if not valid_native:
                    raise BenchmarkIntegrityError(
                        f"BEAM {scale}/{conv_id}/{ability}/{ability_index} "
                        "has a malformed native question id"
                    )
                if isinstance(native_id, str):
                    native_id = native_id.strip()
            global_index = len(all_questions)
            # Calibration assignment must not be a function of the oracle
            # ability/category. Dataset-native ids take precedence; otherwise
            # bind the global source ordinal to a hash of the question text.
            stable_tail = (
                f"native:{native_id}" if native_id is not None
                else f"ordinal:{global_index}:{content_hash(question)[:20]}"
            )
            all_questions.append({
                        "ability": ability,
                        "ability_short": ABILITY_MAP[ability],
                        "question_id": (
                            f"beam:{scale}:{conv_id}:{stable_tail}"
                        ),
                        "source_question_id": native_id,
                        "question": question,
                        # `ideal_answer` intentionally keeps the ORIGINAL
                        # (mostly empty) parse, because it is what the judge
                        # reads: repointing the judge changes what every BEAM
                        # score MEANS and is a separate pre-registered decision
                        # (--judge-gold), not a side effect of fixing the probe.
                        "ideal_answer": q.get("ideal_response",
                                              q.get("ideal_answer", "")),
                        "gold_text": gold_text,
                        "gold_kind": gold_kind,
                        "gold_field": gold_field,
                        "gold_resolution": gold_resolution,
                        "rubric": rubric,
            })

    if not all_questions:
        raise BenchmarkIntegrityError(
            f"BEAM {scale}/{conv_id} has no probing questions"
        )
    return {
        "id": conv_id,
        "messages": all_messages,
        "questions": all_questions,
        "scale": scale,
    }


# ── HyMem Integration ───────────────────────────────────────────────────────

def resolve_embedding_config(
    backend: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    dimension: int | None = None,
) -> dict[str, Any]:
    """Resolve an embedding posture without ever including credentials."""

    if backend not in {"local-hash", "openai-compatible", "none"}:
        raise BenchmarkIntegrityError(f"unknown embedding backend: {backend!r}")
    if dimension is not None and (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise BenchmarkIntegrityError("embedding dimension must be positive")
    if backend == "none":
        if model is not None or base_url is not None or dimension is not None:
            raise BenchmarkIntegrityError(
                "embedding overrides cannot be combined with backend none"
            )
        return {
            "configured": False, "backend": "none", "model": None,
            "base_url": None, "dimension": None, "quality": "none",
            "network_free": True, "fallback_policy": "none",
            "fallback_reason": None,
        }
    if backend == "local-hash":
        if base_url not in (None, "local://feature-hash"):
            raise BenchmarkIntegrityError(
                "local-hash accepts only local://feature-hash as an endpoint label"
            )
        resolved_model = model or DEFAULT_LOCAL_EMBEDDING_MODEL
        if not isinstance(resolved_model, str) or not resolved_model.strip():
            raise BenchmarkIntegrityError("embedding model must be non-empty")
        return {
            "configured": True, "backend": "local-hash",
            "model": resolved_model.strip(), "base_url": "local://feature-hash",
            "dimension": dimension or DEFAULT_LOCAL_EMBEDDING_DIM,
            "quality": "lexical-feature-hash", "network_free": True,
            "fallback_policy": "none", "fallback_reason": None,
        }

    from hymem.contrib.openai_embedding_client import (
        safe_embedding_base_url,
        validate_embedding_base_url,
    )
    resolved_model = model or DEFAULT_REMOTE_EMBEDDING_MODEL
    resolved_base = base_url or DEFAULT_REMOTE_EMBEDDING_BASE_URL
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise BenchmarkIntegrityError("embedding model must be non-empty")
    try:
        validate_embedding_base_url(resolved_base)
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(str(exc)) from exc
    parsed_base = urlsplit(resolved_base)
    if parsed_base.username is not None or parsed_base.password is not None:
        raise BenchmarkIntegrityError(
            "embedding endpoint credentials must use --embedding-api-key, not URL userinfo"
        )
    if parsed_base.query:
        raise BenchmarkIntegrityError(
            "benchmark embedding endpoints must not contain query parameters; "
            "put credentials in --embedding-api-key"
        )
    if parsed_base.fragment:
        raise BenchmarkIntegrityError("embedding endpoint must not contain a fragment")
    return {
        "configured": True, "backend": "openai-compatible",
        "model": resolved_model.strip(),
        "base_url": safe_embedding_base_url(resolved_base),
        # Kept only in process and removed before manifest construction.
        "request_base_url": resolved_base,
        "dimension": dimension or DEFAULT_REMOTE_EMBEDDING_DIM,
        "quality": "semantic", "network_free": False,
        "fallback_policy": "fail-closed", "fallback_reason": None,
    }


def public_embedding_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Manifest-safe exact embedding identity."""

    return {
        key: value for key, value in config.items()
        if key != "request_base_url"
    }


class BenchmarkPinnedEmbeddingClient:
    """Fail closed if a provider's observed vector identity drifts."""

    def __init__(self, inner: object, *, expected_dimension: int) -> None:
        self._inner = inner
        self.expected_dimension = int(expected_dimension)
        self._initial_model = getattr(inner, "model")

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    @property
    def model(self):
        return getattr(self._inner, "model")

    @property
    def dim(self):
        return getattr(self._inner, "dim")

    def embed(self, texts):
        if self.model != self._initial_model or self.dim != self.expected_dimension:
            raise BenchmarkIntegrityError(
                "embedding client identity differs from manifested model/dimension"
            )
        vectors = self._inner.embed(texts)
        if (
            self.model != self._initial_model
            or self.dim != self.expected_dimension
            or any(
                not isinstance(vector, (list, tuple))
                or len(vector) != self.expected_dimension
                for vector in vectors
            )
        ):
            raise BenchmarkIntegrityError(
                "embedding provider returned a dimension/identity that differs "
                "from the benchmark manifest"
            )
        return vectors


def build_embedding_client(
    config: Mapping[str, Any], *, api_key: str = "",
):
    """Construct exactly the resolved backend; never silently fall back."""

    backend = config.get("backend")
    if backend == "none":
        return None
    from hymem.extraction.embeddings import (
        CachedEmbeddingClient,
        LocalHashEmbeddingClient,
    )
    if backend == "local-hash":
        return BenchmarkPinnedEmbeddingClient(
            CachedEmbeddingClient(LocalHashEmbeddingClient(
                dim_value=int(config["dimension"]),
                model_name=str(config["model"]),
            )),
            expected_dimension=int(config["dimension"]),
        )
    if backend != "openai-compatible":
        raise BenchmarkIntegrityError(
            f"invalid resolved embedding backend: {backend!r}"
        )
    from hymem.contrib.openai_embedding_client import (
        OpenAICompatibleEmbeddingClient,
    )
    return BenchmarkPinnedEmbeddingClient(
        CachedEmbeddingClient(OpenAICompatibleEmbeddingClient(
            api_key=api_key or None,
            base_url=str(config["request_base_url"]),
            model=str(config["model"]),
            dim=int(config["dimension"]),
        )),
        expected_dimension=int(config["dimension"]),
    )


def embedding_backlog_status(conn, client: object | None) -> dict[str, int]:
    """Count absent/stale vector mirrors entirely inside SQLite.

    This deliberately uses the lossless coverage stream rather than raw
    messages, which may be pruned after their proof-valid mirrors exist. Each
    query returns one integer; no 10M-scale vector corpus is materialized in
    Python on every dream cycle.
    """

    if client is None:
        return {
            "pending_chunk_embeddings": 0,
            "pending_message_embeddings": 0,
            "pending_edge_embeddings": 0,
            "pending_episode_embeddings": 0,
            "pending_fact_embeddings": 0,
        }
    from hymem.core.graph import live_edge_predicate
    from hymem.core.vectors import decode_vector
    from hymem.extraction.embeddings import embedding_text_hash

    model = getattr(client, "model", None)
    dim = getattr(client, "dim", None)
    if (
        not isinstance(model, str) or not model
        or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
    ):
        raise BenchmarkIntegrityError("embedding client identity is invalid")

    conn.create_function(
        "hymem_benchmark_embedding_hash", 1,
        lambda value: embedding_text_hash(str(value)), deterministic=True,
    )
    def valid_vector(value, expected_dim):
        try:
            vector = decode_vector(value)
            return int(
                len(vector) == int(expected_dim)
                and all(math.isfinite(float(item)) for item in vector)
                and math.sqrt(sum(float(item) ** 2 for item in vector)) > 0.0
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return 0

    conn.create_function(
        "hymem_benchmark_vector_valid", 2, valid_vector, deterministic=True,
    )

    def invalid_vector(alias: str) -> str:
        return (
            f"({alias}.vector_json IS NULL OR {alias}.model<>? OR {alias}.dim<>? "
            f"OR hymem_benchmark_vector_valid({alias}.vector_json,?)<>1)"
        )

    def count(sql: str, params: tuple[Any, ...]) -> int:
        row = conn.execute(sql, params).fetchone()
        if row is None:
            raise BenchmarkIntegrityError("embedding backlog query returned no row")
        return int(row[0])

    vector_params = (model, dim, dim)
    pending_chunks = count(
        "SELECT COUNT(*) FROM chunks c "
        "LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id "
        "WHERE c.chunk_kind='extraction' AND (e.text_hash<>"
        "hymem_benchmark_embedding_hash(c.text) OR e.text_hash IS NULL OR "
        + invalid_vector("e") + ")",
        vector_params,
    )
    pending_messages = count(
        "SELECT COUNT(*) FROM message_retention_coverage mc "
        "JOIN sessions s ON s.id=mc.source_session_id "
        "JOIN chunks c ON c.id=mc.chunk_id "
        "LEFT JOIN message_embeddings e ON e.message_id=mc.message_id "
        "WHERE mc.coverage_version='dream-lossless-message-v1' "
        "AND mc.source_role IN ('user','assistant') "
        "AND s.coverage_message_id IS NOT NULL "
        "AND typeof(mc.message_id)='integer' "
        "AND mc.message_id<=s.coverage_message_id "
        "AND c.session_id=mc.source_session_id "
        "AND c.start_message_id=mc.message_id "
        "AND c.end_message_id=mc.message_id AND c.chunk_kind='coverage' "
        "AND hymem_message_record_proof_valid(c.text,mc.message_content_hash,"
        "mc.hash_version,mc.record_version)=1 "
        "AND (e.text_hash<>hymem_benchmark_embedding_hash("
        "json_extract(c.text,'$.content')) OR e.text_hash IS NULL OR "
        + invalid_vector("e") + ")",
        vector_params,
    )
    edge_text = "(k.subject_canonical || ' ' || k.predicate || ' ' || k.object_canonical)"
    pending_edges = count(
        "SELECT COUNT(*) FROM knowledge_graph k "
        f"LEFT JOIN edge_embeddings e ON e.edge_text={edge_text} "
        f"WHERE {live_edge_predicate('k')} AND " + invalid_vector("e"),
        vector_params,
    )
    pending_episodes = count(
        "SELECT COUNT(*) FROM episodes ep JOIN sessions s ON s.id=ep.session_id "
        "LEFT JOIN episode_embeddings e ON e.episode_id=ep.id "
        "WHERE (ep.digest_generation IS NULL OR "
        "ep.digest_generation=s.digest_published_generation) "
        "AND (e.text_hash<>hymem_benchmark_embedding_hash("
        "ep.title || char(10) || ep.summary) OR e.text_hash IS NULL OR "
        + invalid_vector("e") + ")",
        vector_params,
    )
    pending_facts = count(
        "SELECT COUNT(*) FROM narrative_facts f "
        "JOIN fact_extraction_outcomes o ON o.slice_key=f.source_outcome_key "
        "LEFT JOIN narrative_fact_embeddings e ON e.fact_id=f.id "
        "WHERE f.source_outcome_key IS NOT NULL "
        "AND f.lifecycle_status='active' AND f.invalid_at IS NULL "
        "AND o.outcome_status='success' AND o.source_manifest_complete=1 "
        "AND o.source_manifest_version='fact-source-manifest-v1' "
        "AND o.source_manifest_count>0 "
        "AND (e.text_hash<>hymem_benchmark_embedding_hash(f.text) "
        "OR e.text_hash IS NULL OR " + invalid_vector("e") + ")",
        vector_params,
    )

    return {
        "pending_chunk_embeddings": pending_chunks,
        "pending_message_embeddings": pending_messages,
        "pending_edge_embeddings": pending_edges,
        "pending_episode_embeddings": pending_episodes,
        "pending_fact_embeddings": pending_facts,
    }


class HyMemAdapter:
    """Direct HyMem Python API adapter with isolated temp DB."""

    def __init__(self, db_path: Path, api_key: str = "",
                 facts_enabled: bool | None = None,
                 facts_extraction: bool | None = None,
                 pipeline_model: str = "deepseek-chat",
                 pipeline_base_url: str = DEEPSEEK_BASE_URL,
                 pipeline_thinking: str = "off",
                 embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
                 embedding_model: str | None = None,
                 embedding_base_url: str | None = None,
                 embedding_dim: int | None = None,
                 embedding_api_key: str = ""):
        self.db_path = db_path
        self.api_key = api_key
        self.facts_enabled = facts_enabled
        self.facts_extraction = facts_extraction
        self.pipeline_model = pipeline_model
        self.pipeline_base_url = pipeline_base_url
        self.pipeline_thinking = pipeline_thinking
        self.embedding_config = resolve_embedding_config(
            embedding_backend, model=embedding_model,
            base_url=embedding_base_url, dimension=embedding_dim,
        )
        self.embedding_api_key = embedding_api_key
        self.hy = None
        self.pipeline_llm = None
        self.embedding_client = None
        self.last_indexing_summary = None

    def build_config(self):
        from hymem import HyMemConfig

        overrides = {}
        if self.facts_enabled is not None:
            overrides["facts_enabled"] = self.facts_enabled
        if self.facts_extraction is not None:
            overrides["facts_extraction_enabled"] = self.facts_extraction
        overrides["aggregation_nodes_enabled"] = False
        overrides["episode_granularity_enabled"] = False
        return HyMemConfig(
            root=self.db_path.parent,
            message_fts_top_k=15,
            fts_top_k=10,
            graph_top_k=10,
            **overrides,
        )

    def open(self):
        from hymem import HyMem
        from hymem.contrib.openai_client import OpenAICompatibleClient

        # E1 narrative facts (schema v26). None = config default (both ON);
        # --no-facts is the read-side control arm, --no-facts-extraction stops
        # the dream spending a call per session tail (needs a fresh store).
        # RAPTOR aggregation layer: pinned OFF explicitly. The config default
        # flipped False -> True on 2026-08-26 (G-FLIP PASS); this benchmark was
        # a default-config consumer, so without the pin the flip would silently
        # switch the layer + digest ON and the canonical baseline would stop
        # being comparable to every run behind it. Moving a benchmark onto the
        # shipped config is a pre-registered scored decision, not a side effect
        # of a default change.
        # Same reasoning, one lever earlier: `episode_granularity_enabled`
        # is under active decision (Plan C) and this adapter was the last
        # unpinned default-config consumer of it. The pin matches the
        # current default, so it changes nothing today -- that is the point.
        # It means the pre-check below reads a KNOWN arm, and a future
        # default flip cannot silently move BEAM off its baseline the way
        # the aggregation flip nearly did.
        cfg = self.build_config()
        llm = OpenAICompatibleClient(
            api_key=self.api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
            base_url=self.pipeline_base_url,
            model=self.pipeline_model,
            thinking=self.pipeline_thinking,
        )
        self.pipeline_llm = llm
        self.embedding_client = build_embedding_client(
            self.embedding_config, api_key=self.embedding_api_key,
        )
        self.hy = HyMem(
            cfg, llm=llm, embedding_client=self.embedding_client,
        )

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

    def dream_and_wait(
        self, timeout: float = 3600, *, max_cycles: int = 100,
        require_healthy: bool = True,
    ):
        """Run bounded dream cycles until indexing is complete and healthy."""
        start = time.time()
        dream_hy = self.hy.fork()
        try:
            def durable_status():
                current = dict(dream_hy.dream_status())
                current["quarantined_facts"] = int(
                    dream_hy.read_conn.execute(
                        "SELECT COALESCE(SUM(facts_quarantined), 0) FROM sessions"
                    ).fetchone()[0]
                )
                current.update(embedding_backlog_status(
                    dream_hy.read_conn, self.embedding_client,
                ))
                return current

            try:
                self.last_indexing_summary = converge_indexing(
                    dream_hy.dream,
                    status=durable_status,
                    max_cycles=max_cycles,
                    timeout_s=timeout,
                    require_healthy=require_healthy,
                )
            except Exception as exc:
                if hasattr(exc, "summary"):
                    self.last_indexing_summary = dict(exc.summary)
                raise
        finally:
            dream_hy.close()
        self.hy.invalidate_query_caches()
        elapsed = time.time() - start
        print(
            f"      Dream converged in {elapsed:.0f}s across "
            f"{self.last_indexing_summary['cycles']} cycle(s)", flush=True,
        )
        return self.last_indexing_summary

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

        result = self.hy.augment(
            query,
            session_id=session_id,
            source_session_id=session_id,
            ability=ability,
        )

        if self.embedding_config["configured"]:
            semantic = getattr(result, "semantic_status", None)
            expected_model = getattr(self.embedding_client, "model", None)
            expected_dim = getattr(self.embedding_client, "dim", None)
            if (
                expected_dim != self.embedding_config["dimension"]
                or semantic is None
                or getattr(semantic, "configured", None) is not True
                or getattr(semantic, "attempted", None) is not True
                or getattr(semantic, "available", None) is not True
                or getattr(semantic, "model", None) != expected_model
                or getattr(semantic, "dim", None) != expected_dim
            ):
                reason = getattr(semantic, "reason", "missing_status")
                raise BenchmarkIntegrityError(
                    "configured embedding retrieval was unavailable or changed "
                    f"identity (reason={reason})"
                )

        # A shared BEAM store contains many independent examples. The
        # source-session boundary is therefore part of benchmark correctness,
        # not merely a retrieval hint. Fail loudly if any tier violates it.
        for tier in ("message_hits", "count_message_hits", "recent_turns"):
            for hit in (getattr(result, tier, None) or []):
                direct_session = getattr(hit, "session_id", None)
                if not isinstance(direct_session, str) or direct_session != session_id:
                    raise BenchmarkIntegrityError(
                        f"BEAM source isolation violation in {tier}: "
                        f"{direct_session!r} != {session_id!r}"
                    )

        # Summaries/chunks/facts are composites. A convenient direct
        # ``session_id`` is descriptive, not proof that every source behind the
        # rendered text belongs to this independent benchmark conversation.
        for tier in ("fts_hits", "facts", "episodes", "aggregation_nodes"):
            for hit in (getattr(result, tier, None) or []):
                occurrences = getattr(hit, "source_occurrences", None) or ()
                if (
                    getattr(hit, "source_provenance_complete", None) is not True
                    or not occurrences
                ):
                    raise BenchmarkIntegrityError(
                        f"BEAM source isolation violation in {tier}: "
                        "complete source provenance is absent"
                    )
                if any(
                    getattr(occurrence, "session_id", None) != session_id
                    for occurrence in occurrences
                ):
                    raise BenchmarkIntegrityError(
                        f"BEAM source isolation violation in {tier} provenance"
                    )

        for fact in (getattr(result, "graph_facts", None) or []):
            citations = getattr(fact, "citations", None) or ()
            if not citations:
                raise BenchmarkIntegrityError(
                    "BEAM source isolation violation in graph_facts: "
                    "citations are absent"
                )
            for citation in citations:
                citation_session = getattr(citation, "source_session_id", None)
                citation_message = getattr(citation, "source_message_id", None)
                if (
                    citation_session != session_id
                    or isinstance(citation_message, bool)
                    or not isinstance(citation_message, int)
                ):
                    raise BenchmarkIntegrityError(
                        "BEAM source isolation violation in graph citation"
                    )
        for unsupported in ("procedures", "temporal_events", "user_profile"):
            if getattr(result, unsupported, None):
                raise BenchmarkIntegrityError(
                    f"BEAM scoped retrieval returned unscoped tier {unsupported}"
                )

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
                    narrative_facts: list[str] | None = None,
                    max_input_tokens: int | None = DEFAULT_MAX_INPUT_TOKENS) -> str:
    """Ask LLM to answer based on retrieved memories."""
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

    # Ability-aware answering prompts
    system_prompt = ANSWERING_PROMPTS.get(ability, ANSWERING_SYSTEM_PROMPT)
    user_prefix = "CONTEXT:\n"
    user_suffix = f"\n\nQUESTION: {question}\n\nANSWER:"
    counter = getattr(llm, "count_tokens", None)
    context = _render_answer_context(
        memories, ability, total_matches, None, None, None,
        narrative_facts=narrative_facts,
        max_context_chars=context_limit,
        max_input_tokens=max_input_tokens,
        token_counter=counter if callable(counter) else None,
        prompt_prefix=f"system:{system_prompt}\nuser:{user_prefix}",
        prompt_suffix=user_suffix,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prefix}{context}{user_suffix}"},
    ]

    prediction = llm.chat(messages, temperature=0.0, max_tokens=1024)
    _log_context(question_id, ability, system_prompt, question, context, prediction)
    return prediction


def _judge_messages(question: str, ideal: str, rubric: list, ai_answer: str) -> list:
    """Assemble the judge messages. Extracted verbatim from judge_answer so a
    byte-equality test can pin the construction (and the rejudge can compare
    before/after). MUST NOT change message content."""
    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric))
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"QUESTION: {question}\n\n"
            f"IDEAL ANSWER: {ideal}\n\n"
            f"AI ANSWER: {ai_answer}\n\n"
            f"RUBRIC (score 0-1 each):\n{rubric_text}\n\n"
            f"Return JSON with 'scores' list and 'total_score'."
        )},
    ]


def _official_judge_messages(rubric_item: str, ai_answer: str) -> list[dict]:
    """Exact upstream prompt construction: no question and no gold answer."""

    prompt = OFFICIAL_JUDGE_PROMPT.replace(
        "<rubric_item>", rubric_item
    ).replace("<llm_response>", ai_answer)
    return [{"role": "user", "content": prompt}]


def extract_judge_json(raw: str) -> tuple[dict | None, str]:
    r"""The judge's verdict object, and how hard it was to get.

    Returns (obj, how) where how is "ok" | "recovered" | "unreadable".

    `re.search(r'\{[^}]+\}')` stops at the FIRST `}`. When an explanation
    quotes text containing a brace -- code, JSON, a `${template}` literal --
    the match ends mid-string, json.loads raises, and the caller emits a
    sentinel 0.0 that is indistinguishable in the score column from a real 0.0.
    Observed on a complete, valid, finish_reason="stop" reply in which the
    judge had written `scores: [1]`: the answer under grading contained
    `${response.status}`, so a 1.0 was recorded as 0.0.

    So: brace-match, tracking string literals and escapes, and return the first
    complete top-level object. Strictly MORE PERMISSIVE than the regex, never
    different -- anything the regex parsed must parse identically here, which
    is pinned by test rather than by inspection. A reply neither can read still
    returns None, so an unreadable judge stays unreadable.
    """
    if not raw:
        return None, "unreadable"
    flat = raw.replace("\n", " ")
    naive = re.search(r"\{[^}]+\}", flat)
    if naive:
        try:
            return json.loads(naive.group()), "ok"
        except Exception:
            pass
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(flat):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(flat[start:i + 1]), "recovered"
                except Exception:
                    start = None
    return None, "unreadable"


def _finish_reason(llm) -> str | None:
    """Why the last call ended, for any object that plays the LLM's part.

    judge_answer is duck-typed -- tests and callers pass stubs exposing only
    chat() -- so reading the attribute directly would narrow the contract from
    "anything with chat()" to "an LLMClient" and break them. Missing reads as
    None, which is the honest value: nothing told us why the call ended.
    """
    return getattr(llm, "last_finish_reason", None)


def official_judge_answer(
    llm: LLMClient, rubric: list[str], ai_answer: str,
) -> dict:
    """Run BEAM's upstream rubric-by-rubric ternary judge protocol.

    Every criterion is a separate temperature-zero call. Any transport,
    parsing, score-domain, or reason failure invalidates the whole row rather
    than manufacturing a semantic zero from an infrastructure failure.
    """

    if (
        not isinstance(rubric, list)
        or not rubric
        or any(not isinstance(item, str) or not item.strip() for item in rubric)
    ):
        return {
            "score": 0.0, "llm_judge_score": 0.0, "scores": [],
            "criterion_results": [], "judge_parse": "missing_rubric",
        }
    criterion_results: list[dict] = []
    scores: list[float] = []
    for index, item in enumerate(rubric):
        raw = llm.chat(
            _official_judge_messages(item, ai_answer),
            temperature=0.0,
            max_tokens=None,
        )
        finish_reason = _finish_reason(llm)
        parsed, parse_kind = extract_judge_json(raw)
        score = parsed.get("score") if isinstance(parsed, dict) else None
        reason = parsed.get("reason") if isinstance(parsed, dict) else None
        valid_score = bool(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and float(score) in {0.0, 0.5, 1.0}
        )
        valid_reason = isinstance(reason, str) and bool(reason.strip())
        transport_error = isinstance(raw, str) and raw.startswith("[LLM_ERROR")
        criterion = {
            "criterion_index": index,
            "rubric_item": item,
            "raw": raw,
            "finish_reason": finish_reason,
            "parse": parse_kind,
            "score": float(score) if valid_score else None,
            "reason": reason if valid_reason else None,
        }
        criterion_results.append(criterion)
        if transport_error:
            failure = f"criterion_{index}_transport"
        elif parsed is None:
            failure = f"criterion_{index}_unreadable"
        elif not valid_score:
            failure = f"criterion_{index}_invalid_score"
        elif not valid_reason:
            failure = f"criterion_{index}_invalid_reason"
        else:
            failure = None
        if failure is not None:
            return {
                "score": 0.0, "llm_judge_score": 0.0, "scores": [],
                "criterion_results": criterion_results,
                "judge_parse": failure,
            }
        scores.append(float(score))
    mean_score = sum(scores) / len(scores)
    return {
        "score": mean_score,
        "llm_judge_score": mean_score,
        "scores": scores,
        "criterion_results": criterion_results,
        "judge_parse": "ok",
    }


def judge_answer(llm: LLMClient, question: str, ideal: str, rubric: list, ai_answer: str,
                 return_raw: bool = False) -> dict:
    if not rubric:
        out = {"score": 0.0, "scores": [], "judge_parse": "missing_rubric"}
        if return_raw:
            out.update({"judge_raw": "", "judge_finish_reason": None})
        return out

    messages = _judge_messages(question, ideal, rubric, ai_answer)

    raw = llm.chat(messages, temperature=0.0, max_tokens=512)
    result, how = extract_judge_json(raw)
    scores = result.get("scores", []) if isinstance(result, dict) else []
    valid_scores = bool(
        isinstance(scores, list)
        and len(scores) == len(rubric)
        and all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and 0.0 <= float(score) <= 1.0
            for score in scores
        )
    )
    if result is not None and valid_scores:
        total = sum(float(score) for score in scores) / len(scores)
        out = {"score": total, "scores": scores}
    else:
        out = {"score": 0.0, "scores": []}
        how = "malformed" if result is not None else how
    if return_raw:
        out["judge_raw"] = raw
        out["judge_finish_reason"] = _finish_reason(llm)
        out["judge_parse"] = how
    return out


# ── Evaluation ───────────────────────────────────────────────────────

def _evaluate_beam_question(
    judge_gold: bool,
    llm: LLMClient,
    judge_llm: LLMClient,
    hy: HyMemAdapter,
    conv: dict,
    q: dict,
    qi: int,
    top_k: int,
    *,
    oracle_ability: bool,
    judge_protocol: str = "official",
    max_input_tokens: int | None = DEFAULT_MAX_INPUT_TOKENS,
) -> dict:
    oracle = q["ability_short"]
    question = q["question"]
    detected = (
        _detect_ability_safe(question) if oracle_ability
        else _detect_ability(question)
    )
    ability_used = oracle if oracle_ability else detected
    print(f"    [{qi+1}/{len(conv['questions'])}] {oracle}: "
          f"{question[:100]}...", flush=True)

    memories, total_matches, narrative_facts = hy.search(
        f"beam-{conv['scale']}-{conv['id']}", question,
        ability=ability_used, top_k=top_k * 3,
    )
    print(f"      {len(memories)} memories", end="")
    if total_matches > 0:
        print(f" (total matches: {total_matches})", end="")
    if narrative_facts:
        print(f" (facts: {len(narrative_facts)})", end="")
    print()

    answer = answer_question(
        llm, memories, question, ability_used, total_matches,
        question_id=q["question_id"], narrative_facts=narrative_facts,
        max_input_tokens=max_input_tokens,
    )
    base = {
        "question_id": q["question_id"],
        "scale": conv["scale"],
        "conv_id": conv["id"],
        "ability": oracle,
        "oracle_ability": oracle,
        "detected_ability": detected,
        "ability_used": ability_used,
        "question": question,
        "answer": answer,
        "ideal_answer": q["ideal_answer"],
        "rubric": q["rubric"],
        "gold_kind": q.get("gold_kind", "none"),
        "indexing": getattr(hy, "last_indexing_summary", None),
    }
    if (answer or "").startswith("[LLM_ERROR"):
        return {
            **base, "judged_ideal": None, "score": 0.0, "scores": [],
            "llm_judge_score": 0.0, "judge_protocol": judge_protocol,
            "result_valid": False, "correct": False, "judge_parse": "not_called",
            "benchmark_failure": "reader_transport_or_content_failure",
        }

    if judge_protocol == "official":
        _judge_ideal = None
        judged = official_judge_answer(judge_llm, q["rubric"], answer)
        parse_valid = judged.get("judge_parse") == "ok"
    elif judge_protocol == "legacy-custom":
        _judge_ideal = select_judge_ideal(
            judge_gold, q.get("gold_text"), q["ideal_answer"]
        )
        judged = judge_answer(
            judge_llm, question, _judge_ideal, q["rubric"], answer,
            return_raw=True,
        )
        parse_valid = judged.get("judge_parse") in {"ok", "recovered"}
    else:
        raise BenchmarkIntegrityError(
            f"unknown BEAM judge protocol: {judge_protocol!r}"
        )
    failure = None if parse_valid else f"judge_{judged.get('judge_parse', 'unreadable')}"
    row = {
        **base,
        "judged_ideal": _judge_ideal,
        "judge_protocol": judge_protocol,
        "score": judged["score"] if parse_valid else 0.0,
        "llm_judge_score": judged["score"] if parse_valid else 0.0,
        "scores": judged["scores"] if parse_valid else [],
        # Execution validity is separate from semantic quality. The continuous
        # BEAM capability metric remains ``score``; ``correct`` means every
        # rubric item passed, never merely that the judge response parsed.
        "result_valid": bool(parse_valid),
        "correct": bool(parse_valid and judged["score"] == 1.0),
        "judge_raw": judged.get("judge_raw"),
        "judge_finish_reason": judged.get("judge_finish_reason"),
        "judge_parse": judged.get("judge_parse"),
        "judge_criterion_results": judged.get("criterion_results"),
        "benchmark_failure": failure,
    }
    try:
        row["probe"] = episode_probe(
            memories, question, _probe_gold(q),
            _decoy_answer(conv["questions"], qi),
        )
        row["probe_error"] = None
    except Exception as exc:
        # Gold-derived probes are post-answer diagnostics. They may explain a
        # result, but can never replace a valid reader/judge result with zero.
        row["probe"] = None
        row["probe_error"] = f"{type(exc).__name__}: {exc}"
    print(f"      Score: {row['score']:.2f}")
    return row


def evaluate_conversation(
    judge_gold: bool,
    llm: LLMClient,
    judge_llm: LLMClient,
    hy: HyMemAdapter,
    conv: dict,
    top_k: int,
    *,
    oracle_ability: bool = False,
    judge_protocol: str = "official",
    pending_ids: set[str] | None = None,
    on_result=None,
    indexing_max_cycles: int | None = None,
    indexing_timeout_s: float | None = None,
    max_input_tokens: int | None = DEFAULT_MAX_INPUT_TOKENS,
) -> dict:
    conv_id = conv["id"]
    scale = conv["scale"]
    session_id = f"beam-{scale}-{conv_id}"

    # Ingest
    print(f"  Ingesting conv {conv_id} ({len(conv['messages'])} msgs)...", flush=True)
    stats = hy.ingest(session_id, conv["messages"])
    print(f"    Ingested: {stats['total_msgs']} msgs, {stats['total_chars']} chars", flush=True)

    # Dream
    print(f"  Dreaming...", flush=True)
    if indexing_max_cycles is None and indexing_timeout_s is None:
        hy.dream_and_wait()
    else:
        hy.dream_and_wait(
            max_cycles=indexing_max_cycles or 100,
            timeout=indexing_timeout_s or 3600,
            require_healthy=True,
        )

    # Evaluate each question. Resume rebuilds/ingests the conversation but
    # completed question ids are skipped before any reader/judge call.
    results = []
    for qi, q in enumerate(conv["questions"]):
        if pending_ids is not None and q["question_id"] not in pending_ids:
            continue
        try:
            row = _evaluate_beam_question(
                judge_gold, llm, judge_llm, hy, conv, q, qi, top_k,
                oracle_ability=oracle_ability,
                judge_protocol=judge_protocol,
                max_input_tokens=max_input_tokens,
            )
        except Exception as exc:
            row = {
                "question_id": q["question_id"],
                "scale": scale,
                "conv_id": conv_id,
                "ability": q["ability_short"],
                "oracle_ability": q["ability_short"],
                "detected_ability": _detect_ability_safe(q.get("question", "")),
                "ability_used": None,
                "question": q.get("question", ""),
                "score": 0.0,
                "llm_judge_score": 0.0,
                "scores": [],
                "judge_protocol": judge_protocol,
                "result_valid": False,
                "correct": False,
                "benchmark_failure": (
                    f"execution_failure: {type(exc).__name__}: {exc}"
                ),
                "indexing": getattr(hy, "last_indexing_summary", None),
            }
        results.append(row)
        if on_result is not None:
            on_result(row)

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


def _strict_beam_payload(
    ledger: AtomicCheckpoint,
    selected_conversations: dict[str, list[dict]],
    scales: list[str],
    *,
    label_free: bool,
    judge_gold: bool,
    official_judge_match: bool = False,
    lifecycle_errors: list[str] | None = None,
) -> tuple[dict, dict, list[dict]]:
    """Build optional diagnostics without putting durable rows at risk."""

    errors: list[str] = list(lifecycle_errors or ())
    all_results: list[dict] = []
    try:
        reconciled = ledger.reconcile()
        by_id = {row["question_id"]: dict(row) for row in reconciled.rows}
        for scale in scales:
            for conv in selected_conversations.get(scale, ()):
                rows = []
                for q in conv["questions"]:
                    row = {
                        "scale": scale,
                        "conv_id": conv["id"],
                        "ability": q["ability_short"],
                        "score": 0.0,
                        **by_id[q["question_id"]],
                    }
                    if row.get("benchmark_failure"):
                        row["score"] = 0.0
                        row["result_valid"] = False
                        row["correct"] = False
                    rows.append(row)
                all_results.append({
                    "conv_id": conv["id"], "scale": scale, "questions": rows,
                })
    except Exception as exc:
        errors.append(f"result_reconstruction: {type(exc).__name__}: {exc}")
        all_results = []

    try:
        summary = compute_scores(all_results) if all_results else {}
    except Exception as exc:
        errors.append(f"score_summary: {type(exc).__name__}: {exc}")
        summary = {}

    valid_scores = [
        q["score"] for conv in all_results for q in conv.get("questions", ())
        if q.get("result_valid") is True
        and isinstance(q.get("score"), (int, float))
        and not isinstance(q.get("score"), bool)
    ]
    payload = {
        "benchmark": "BEAM",
        "version": "strict-v1",
        "date": datetime.now(timezone.utc).isoformat(),
        "protocol_disclosure": (
            "label-free routing with the pinned upstream BEAM rubric judge; "
            "gold retained only as a post-answer diagnostic"
            if label_free and judge_gold and official_judge_match
            else "EXPLORATORY NON-COMPARABLE configuration"
        ),
        "summary": {
            scale: {ability: data["avg"] for ability, data in abilities.items()}
            for scale, abilities in summary.items()
        },
        "summary_counts": {
            scale: {ability: data["count"] for ability, data in abilities.items()}
            for scale, abilities in summary.items()
        },
        "conditional_valid_only": {
            "mean_score": (
                sum(valid_scores) / len(valid_scores) if valid_scores else None
            ),
            "count": len(valid_scores),
        },
        "diagnostic_errors": errors,
        "conversations": all_results,
    }
    return payload, summary, all_results


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

    print("\n  External vendor figures are intentionally not mixed into this "
          "table; their models, judges, prompts, versions and sample sets are "
          "not controlled here. See README.md for sourced limitations.")


def _rejudge_run(args, api_key: str) -> None:
    """Judge-only rejudge of a stored beam artifact (B in the gold-delta plan).

    Port of longmemeval_adapter._rejudge_run (mechanism, not design) adapted to
    beam's per-row shape and the pre-registered error classes:
      * silent-0 (non-empty rubric AND scores == [])  → ABORT (exit 2)
      * explicit LLM error ([LLM_ERROR / rate failures) → keep prior score,
        _rejudged=False, excluded from the falsifier; >5% → INVALID
    No answer calls. Answer bytes fixed from the artifact. Gold comes from a
    fresh dataset reparse (deterministic; 160/160 guard) matched on
    (ability, question).
    """
    src = Path(args.rejudge)
    with open(src) as f:
        run = json.load(f)
    rows = [q for c in run.get("conversations", []) for q in c.get("questions", [])]
    if not rows:
        print(f"ERROR: {src.name} has no rows to re-judge.")
        sys.exit(2)
    orig_judge = run.get("metadata", {}).get("judge_model", "unknown")
    n_rows = len(rows)
    print(f"\n=== REJUDGE {src.name} ===")
    print(f"  rows: {n_rows}   original judge: {orig_judge}   new judge: {args.judge_model}")

    # Was an unconditional ABORT while beam had no extra_body plumbing (the
    # gold-delta phase explicitly deferred it). The plumbing exists now, so the
    # same trap is a GUARD: v4-flash is allowed here once thinking is disabled,
    # and still refused when it is not. The judge is DeepSeek-only.
    judge_extra = getattr(args, "judge_extra_body_obj", None) or {}
    check_model_pin("judge", args.judge_model, "deepseek", judge_extra)

    judge_llm = LLMClient(args.judge_model, api_key, extra_body=judge_extra)

    # ── gold reparse (before canary: canary uses a real row's gold) ────────
    dataset_revisions = resolve_dataset_revisions(["100K"], args.dataset_revision)
    print(f"  dataset revisions: {dataset_revisions}", flush=True)
    data = load_beam_conversations(
        ["100K"], max_conv=8, revisions=dataset_revisions
    )
    convs = data["100K"]
    gold_map, gold_rows = _rejudge_gold_map(run, convs=convs, rows=rows)
    cover = sum(len(v) for v in gold_rows.values())
    if cover != n_rows:
        print(f"  ABORT: gold reparse covers {cover} rows, artifact has {n_rows}.")
        sys.exit(3)

    # ── CANARY: exact client path, representative judge prompt ──────────────
    # A trivial "say OK" prompt cannot reproduce the trap shape (content == ""
    # with text in reasoning_content, finish=length): the model must burn its
    # whole budget before writing content. Judge-style messages (long, real)
    # reproduce it. Asserts content NON-EMPTY (not just non-null).
    first = rows[0]
    canary_ideal = gold_rows[first["ability"]][first["question"]]["gold_text"]
    canary_msgs = _judge_messages(first["question"], canary_ideal, first["rubric"], first["answer"])
    canary_raw = judge_llm.chat(canary_msgs, temperature=0.0, max_tokens=512)
    if not canary_raw.strip():
        print("  CANARY FAILED: judge returned EMPTY content on the exact client path "
              "(trap shape). Raw repr: " + repr(canary_raw))
        sys.exit(1)
    if canary_raw.startswith("[LLM_ERROR"):
        print(f"  CANARY FAILED: {canary_raw[:120]}")
        sys.exit(1)
    print(f"  CANARY OK: {len(canary_raw)} chars content on full judge prompt.")

    # ── gap / metadata ─────────────────────────────────────────────────────
    a_date = run.get("metadata", {}).get("date", "")
    try:
        a_dt = datetime.fromisoformat(a_date)
        if a_dt.tzinfo is None:
            a_dt = a_dt.replace(tzinfo=timezone.utc)
    except Exception:
        a_dt = None
    b_start = datetime.now(timezone.utc)
    gap_hours = (b_start - a_dt).total_seconds() / 3600.0 if a_dt else None

    # ── judge every row ────────────────────────────────────────────────────
    new_questions = []
    silent0 = []
    truncated = []
    explicit_err = []
    t0 = time.time()
    for i, r in enumerate(rows):
        gold = gold_rows[r["ability"]][r["question"]]
        ideal = select_judge_ideal(args.judge_gold, gold.get("gold_text"),
                                   r.get("ideal_answer"))
        # What the judge actually read, beside the inherited field that says it
        # did. Without this, an arm comparison reads `ideal_answer` and learns
        # only that four rows were copied from the same source.
        r["judged_ideal"] = ideal
        jr = judge_answer(judge_llm, r["question"], ideal, r["rubric"], r["answer"],
                          return_raw=True)
        raw = jr.get("judge_raw", "")
        if not r["rubric"]:
            # nothing to score against; keep prior (mirrors LME no-hyp rule)
            judged = False
            new_score, new_scores = r["score"], r["scores"]
        elif jr["score"] == 0.0 and jr["scores"] == []:
            if raw.startswith("[LLM_ERROR"):
                # explicit, loud, retriable — keep prior, don't count as verdict
                explicit_err.append((r["ability"], r["question"], raw[:80]))
                judged = False
                new_score, new_scores = r["score"], r["scores"]
            elif is_truncation(raw, jr.get("judge_finish_reason")):
                # TRUNCATION (B2 v0.2 §3). The judge scored the row and then ran
                # out of tokens mid-explanation, so the regex found no closing
                # brace. The PLUMBING IS FINE -- this must not void the run --
                # but the 0.0 it produced is FABRICATED, neither the judge's
                # actual score nor a real 0.0, so it must not be counted either.
                # Keep prior, exclude, report the rate: the explicit_err shape.
                truncated.append((r["ability"], r["question"], raw[:120],
                                  jr.get("judge_finish_reason")))
                judged = False
                new_score, new_scores = r["score"], r["scores"]
            else:
                # silent-0 with the plumbing implicated: empty raw, or a parse
                # failure that truncation does not explain. Indistinguishable
                # from a real 0.0 without the raw. VOID.
                silent0.append((r["ability"], r["question"], raw[:120],
                                jr.get("judge_finish_reason")))
                judged = False
                new_score, new_scores = r["score"], r["scores"]
        else:
            judged = True
            new_score, new_scores = jr["score"], jr["scores"]
        out = dict(r)
        out["score"] = new_score
        out["scores"] = new_scores
        out["score_original"] = r["score"]
        out["judge_raw"] = raw
        out["judge_error"] = bool(raw) and raw.startswith("[LLM_ERROR")
        out["judge_finish_reason"] = jr.get("judge_finish_reason")
        out["judge_truncated"] = is_truncation(raw, jr.get("judge_finish_reason"))
        out["judge_parse"] = jr.get("judge_parse")
        out["_rejudged"] = judged
        out["judge_ideal_used"] = ideal
        new_questions.append(out)
        if (i + 1) % 20 == 0:
            print(f"  ── judged {i+1}/{n_rows}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  judge calls: {judge_llm.call_count} (+1 canary)   elapsed: {elapsed:.0f}s")
    # A silent-0 voids the VERDICT. It must not also destroy the EVIDENCE.
    # This branch used to sys.exit(3) here, before the artifact was written --
    # so a run that had already paid for all 160 judge calls threw the rows
    # away, and the only way to look at what the judge actually said was to
    # buy them again. The refusal is right; discarding the data was a second
    # defect wearing the first one's clothes. The artifact is now written,
    # marked void in its metadata and in its FILENAME, and the exit code is
    # unchanged so nothing automated can mistake it for a verdict.
    if truncated:
        print(f"  ⚠ {len(truncated)} TRUNCATED judge replies — excluded from the "
              f"statistic, prior kept (B2 v0.2 §3). Ceiling {TRUNCATION_CEILING}:")
        for ab, qu, raw, fin in truncated[:10]:
            print(f"    {ab} | {qu[:60]} | finish={fin!r} | raw={raw[:80]!r}")

    void = None
    if len(truncated) > TRUNCATION_CEILING:
        # Not a plumbing failure, but too much of the sample is unreadable to
        # say anything about the rest. Report the rate, do not interpret.
        print(f"  VOID: {len(truncated)} truncated replies exceeds the "
              f"{TRUNCATION_CEILING}-row ceiling; the run is invalid for the falsifier.")
        void = {"reason": "truncation rate above ceiling",
                "rule": f"B2 v0.2 §3: truncation must not exceed {TRUNCATION_CEILING}/160",
                "n_truncated": len(truncated),
                "rows": [{"ability": ab, "question": qu, "judge_raw_head": raw,
                          "finish_reason": fin} for ab, qu, raw, fin in truncated]}
    if silent0:
        print(f"  VOID: {len(silent0)} silent-0 parse failures (A had 0/160; "
              f"B rate must be <= A). First rows:")
        for ab, qu, raw, fin in silent0[:10]:
            print(f"    {ab} | {qu[:60]} | finish={fin!r} | raw={raw[:100]!r}")
        void = void_record(silent0)
        print("  Verdict void. Rows ARE written, marked void, for characterisation.")
    if explicit_err:
        print(f"  ⚠ {len(explicit_err)} explicit judge errors kept as prior (excluded from falsifier):")
        for ab, qu, raw in explicit_err[:10]:
            print(f"    {ab} | {qu[:60]} | {raw}")

    # ── write rejudged artifact ────────────────────────────────────────────
    new_convs = []
    qi = 0
    for c in run.get("conversations", []):
        m = len(c.get("questions", []))
        nc = dict(c)
        nc["questions"] = new_questions[qi:qi + m]
        qi += m
        new_convs.append(nc)
    summary = compute_scores(new_convs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = dict(run.get("metadata", {}))
    meta.update({
        "date": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "answer_calls": 0,
        "judge_calls": judge_llm.call_count,
        "judge_gold": bool(args.judge_gold),
        "a_date": a_date,
        "gap_hours": gap_hours,
        "rejudged_from": src.name,
        "rejudge_original_judge": orig_judge,
        "judge_model": args.judge_model,
        "judge_extra_body": judge_extra,
        # Whether the operator or §6 chose that body. An artifact that records
        # only the value cannot answer which, and they are different runs.
        "extra_body_defaulted": list(getattr(args, "extra_body_defaulted", [])),
        # Overwrites any prereg inherited from the source artifact: this run is
        # authorised by its own spec, not by the one that authorised A.
        "prereg": args.prereg_obj,
        "dataset_revisions": dataset_revisions,
        # Present and null on a good run, so "void" is a field a reader can
        # test rather than an absence they have to notice.
        "void": void,
    })
    out = {"metadata": meta, "summary": {sc: {ab: d["avg"] for ab, d in abils.items()}
                                         for sc, abils in summary.items()},
           "conversations": new_convs}
    dest = rejudge_dest(src, args.judge_model, stamp, void)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Archived → {dest.name}")
    print(f"  {len(silent0)} silent-0 / {len(truncated)} truncated / "
          f"{len(explicit_err)} explicit / "
          f"{sum(1 for q in new_questions if q['_rejudged'])} rejudged of {n_rows}")

    if void:
        # No readout: the readout IS the verdict, and the verdict is void.
        print("  No readout printed — the rows are evidence, not a result.")
        sys.exit(3)

    # ── readout (pre-registered §5 — formulas fixed before counts) ─────────
    _rejudge_readout(run, new_questions, meta=meta)


# B2 v0.2 §3: above this many unreadable-by-truncation rows, too much of the
# sample is missing to say anything about the rest. 5% of 160.
TRUNCATION_CEILING = 8


def is_truncation(raw: str, finish_reason: str | None) -> bool:
    """Did the judge run out of tokens mid-answer, rather than fail?

    The separator is a CONJUNCTION, not `finish_reason` alone, because both
    failure shapes carry "length":

      - the v4-flash trap is EMPTY content + length. It never reaches here:
        the falsy-content raise turns it into "[LLM_ERROR: empty content ...",
        which the caller routes to the explicit bucket.
      - truncation is NON-EMPTY content + length, with the parse failing for
        want of a closing brace. Nothing else catches it.

    Gating on "length" alone would merge the two and call a broken pin a long
    explanation. Callers must have established the parse failure already; this
    answers only "was it truncation".
    """
    return finish_reason == "length" and bool(raw) and bool(raw.strip()) \
        and not raw.startswith("[LLM_ERROR")


def void_record(silent0: list) -> dict | None:
    """The void marker for a run whose verdict is refused but whose rows stand.

    None when the run is clean, so `metadata["void"]` is a field a reader can
    TEST rather than an absence they have to notice -- an artifact predating
    this field and a void artifact must not look alike.
    """
    if not silent0:
        return None
    return {"reason": "silent-0 parse failures",
            "rule": "gold-delta pre-reg §4.7(a): B's silent-0 rate must be <= A's",
            "n_silent0": len(silent0),
            "rows": [{"ability": ab, "question": qu, "judge_raw_head": raw,
                      "finish_reason": fin} for ab, qu, raw, fin in silent0]}


def rejudge_dest(src: Path, judge_model: str, stamp: str, void: dict | None) -> Path:
    """Void-ness belongs in the FILENAME, not only the metadata.

    Artifacts get quoted by name in commit messages, pre-registrations and
    chat. A name that reads like every other result is how a voided run gets
    cited as one two weeks later by someone who never opened it.
    """
    tag = "-VOID" if void else ""
    return src.with_name(
        f"{src.stem}-rejudged-{judge_model.replace('/', '_')}-{stamp}{tag}.json")


def _rejudge_gold_map(run: dict, convs: list = None, rows: list = None) -> tuple:
    """Reparse the dataset and return (None, gold_rows); gold_rows is
    gold_rows[ability][question] -> reparse question dict. Guards: every
    artifact row must match a reparse row by (ability, question) and stored
    fields must be identical — dataset drift cannot silently change gold."""
    if convs is None:
        data = load_beam_conversations(["100K"], max_conv=8)
        convs = data["100K"]
    if rows is None:
        rows = [q for c in run.get("conversations", []) for q in c.get("questions", [])]
    fresh = {}
    for conv in convs:
        for q in conv["questions"]:
            fresh[(q["ability_short"], q["question"])] = q
    gold_rows = {}
    diffs = []
    for r in rows:
        key = (r["ability"], r["question"])
        q = fresh.get(key)
        if q is None:
            diffs.append(f"{key}: no reparse match")
            continue
        for field in ("ideal_answer", "rubric", "gold_kind"):
            if r.get(field) != q.get(field):
                diffs.append(f"{key}: {field} differs")
        if not q.get("gold_text"):
            diffs.append(f"{key}: EMPTY gold_text")
        gold_rows.setdefault(r["ability"], {})[r["question"]] = q
    if diffs:
        print(f"  ABORT: reparse does not reproduce the artifact ({len(diffs)} diffs).")
        for d in diffs[:15]:
            print("   ", d)
        sys.exit(3)
    return (None, gold_rows)


def _rejudge_readout(run: dict, new_questions: list, meta: dict) -> None:
    """Pre-registered §5 readout. Formulas fixed before counts: continuous
    primary (SE from control SD), binarized companion (t=0.45, 2√D band),
    OR verdict, ABS/CR gate."""
    import math
    pool_abs = {"EO", "IE", "IF", "KU", "MR", "PF", "SUM", "TR"}
    ctl_abs = {"ABS", "CR"}

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0

    def sd(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    pool = [q for q in new_questions if q["ability"] in pool_abs and q["_rejudged"]]
    ctl = [q for q in new_questions if q["ability"] in ctl_abs and q["_rejudged"]]
    dpool = [q["score"] - q["score_original"] for q in pool]
    dctl = [q["score"] - q["score_original"] for q in ctl]

    print(f"\n{'─'*72}")
    print("  READOUT (pre-registered 2026-08-31) — gold delta B − A")
    print(f"  pool rows rejudged: {len(pool)}   control rows rejudged: {len(ctl)}")
    print(f"  pool mean delta δ̄  = {mean(dpool)*100:+.3f}pp   "
          f"control mean δ̄_ctl = {mean(dctl)*100:+.3f}pp")
    sd_ctl = sd(dctl)
    print(f"  control SD per row = {sd_ctl:.4f}  (n={len(dctl)})")

    # GATE (amendment d4): control beyond its own band → VOID
    if sd_ctl == 0:
        flips_ctl = sum(1 for q in ctl if q["score"] != q["score_original"])
        gate_bad = flips_ctl > 0
        gate_txt = f"flip-gate (SD=0): {flips_ctl} control flips"
    else:
        gate_bad = abs(mean(dctl)) > 2 * sd_ctl / math.sqrt(max(len(dctl), 1))
        gate_txt = "2·SD/√32"
    verdict_gate = "VOID (unattributable)" if gate_bad else "PASS"
    print(f"  CONTROL GATE: |δ̄_ctl| vs {gate_txt} → {verdict_gate}")

    # PRIMARY (continuous): SE from control SD
    se = sd_ctl / math.sqrt(len(pool)) if sd_ctl > 0 else 0.0
    band = 2 * se
    delta_pp = mean(dpool) * 100
    inside = abs(delta_pp) <= band * 100
    print(f"\n  PRIMARY (continuous): δ̄ = {delta_pp:+.3f}pp   band = ±{band*100:.3f}pp "
          f"(2·SD_ctl/√128)  → {'INSIDE (H0 holds)' if inside else 'OUTSIDE (meaning changed)'}")

    # COMPANION (binarized): t=0.45 from A's marginal
    T = 0.45
    pairs = [(q["score_original"], q["score"]) for q in pool]
    D = sum(1 for a, b in pairs if (a >= T) != (b >= T))
    gained = sum(1 for a, b in pairs if a < T <= b)
    lost = sum(1 for a, b in pairs if a >= T > b)
    net_q = gained - lost
    band_c = 2 * math.sqrt(D)
    net_inside = abs(net_q) <= band_c
    print(f"  COMPANION (t=0.45): D={D}  gained={gained}  lost={lost}  net={net_q:+d}  "
          f"band 2√D={band_c:.2f}  → {'INSIDE' if net_inside else 'OUTSIDE'}")

    # VERDICT (OR, 2α conservative — stated reason)
    if verdict_gate.startswith("VOID"):
        verdict = "RUN VOID (control gate)"
    else:
        reject = (not inside) or (not net_inside)
        verdict = ("REBASE REQUIRED (meaning changed)" if reject
                   else "RECORD STANDS (H0 holds; defect cost nothing measurable)")
    print(f"\n  VERDICT: {verdict}")
    print("  (OR reason: two tests → FPR toward 2α, conservative toward declaring")
    print("   a rebase — correct asymmetry: trusting a contaminated record costs more)")

    # per-ability descriptive (n=16, knife-edge caveat)
    print(f"\n  {'ability':<6} {'A':>7} {'B':>7} {'Δpp':>7} {'flips':>5}")
    for ab in sorted(set(q["ability"] for q in new_questions)):
        qs = [q for q in new_questions if q["ability"] == ab and q["_rejudged"]]
        if not qs:
            continue
        a = mean([q["score_original"] for q in qs])
        b = mean([q["score"] for q in qs])
        fl = sum(1 for q in qs if (q["score_original"] >= T) != (q["score"] >= T))
        print(f"  {ab:<6} {a*100:>6.2f} {b*100:>6.2f} {(b-a)*100:>+6.2f} {fl:>5}")


def _main(_owned_ledgers: list[AtomicCheckpoint] | None = None):
    global DEEPSEEK_API_KEY

    parser = argparse.ArgumentParser(description="HyMem BEAM Benchmark (direct API)")
    parser.add_argument("--scales", default=DEFAULT_SCALE)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-input-tokens", type=int,
                        default=DEFAULT_MAX_INPUT_TOKENS,
                        help="Hard ceiling over the complete rendered reader input.")
    parser.add_argument(
        "--indexing-max-cycles", type=int, default=100,
        help="Maximum dream cycles allowed while draining the indexing backlog.",
    )
    parser.add_argument(
        "--indexing-timeout-s", type=float, default=3600.0,
        help="Wall-clock limit for complete per-conversation indexing.",
    )
    parser.add_argument("--results-dir", default=str(_repo_root.parent / "hymem_beam"))
    parser.add_argument("--answer-model", default=os.environ.get("BEAM_ANSWER_MODEL", ANSWER_MODEL),
                        help="Answerer spec 'provider:model' (e.g. gemini:gemini-2.5-flash) "
                             "or a bare DeepSeek model. Env: BEAM_ANSWER_MODEL.")
    parser.add_argument(
        "--judge-protocol", choices=("official", "legacy-custom"),
        default="official",
        help=(
            "official (default) mirrors upstream BEAM's one-call-per-rubric "
            "ternary judge; legacy-custom is exploratory/non-comparable"
        ),
    )
    parser.add_argument(
        "--judge-model", default=None,
        help=(
            "Judge provider:model. Defaults to openai:gpt-4.1-mini for the "
            "official protocol and the historical DeepSeek model for legacy-custom."
        ),
    )
    parser.add_argument("--hymem-model", default="deepseek-chat",
                        help="Model used by HyMem dream/extraction/rerank calls.")
    parser.add_argument("--hymem-base-url", default=DEEPSEEK_BASE_URL)
    parser.add_argument("--hymem-thinking", choices=("auto", "disabled", "off", "enabled"),
                        default="off")
    parser.add_argument(
        "--embedding-backend",
        choices=("local-hash", "openai-compatible", "none"),
        default=DEFAULT_EMBEDDING_BACKEND,
        help=(
            "Embedding posture. local-hash is the shipped dependency-free "
            "lexical vector default; openai-compatible is semantic; none "
            "disables vector retrieval explicitly."
        ),
    )
    parser.add_argument(
        "--embedding-model", default=None,
        help="Embedding model/identity override for the selected backend.",
    )
    parser.add_argument(
        "--embedding-base-url", default=None,
        help="OpenAI-compatible embedding endpoint (or local://feature-hash label).",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=None,
        help="Declared embedding dimension (backend-specific default when omitted).",
    )
    parser.add_argument(
        "--embedding-api-key", default="",
        help="Embedding credential; never written to checkpoints or artifacts.",
    )
    parser.add_argument("--answer-extra-body", default=None,
                        help="JSON merged into every ANSWER request body. Required "
                             "as '{\"thinking\": {\"type\": \"disabled\"}}' when the "
                             "answerer is a DeepSeek v4-flash model; rejected for "
                             "non-DeepSeek providers, which 400 on that key.")
    parser.add_argument("--judge-extra-body", default=None,
                        help="JSON merged into every JUDGE request body. Same "
                             "thinking-disabled requirement for v4-flash. Applies "
                             "to --rejudge too.")
    parser.add_argument("--dataset-revision", default=None,
                        help="Pin the BEAM dataset to a git revision on the Hub. "
                             "Unset resolves and records whatever the name points "
                             "at now; either way the artifact carries the sha.")
    parser.add_argument("--prereg", default=None,
                        help="Path to the pre-registration authorising this run. It "
                             "must be committed and unmodified; its commit and blob "
                             "hashes are recorded in the artifact. Required unless "
                             "--no-prereg.")
    parser.add_argument("--no-prereg", action="store_true",
                        help="Exploratory run with no pre-registration. Recorded as "
                             "prereg: null in the artifact, so such a run stays "
                             "distinguishable from a canonical one.")
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
    parser.add_argument("--judge-gold", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use resolved official per-ability gold (default). "
                             "--no-judge-gold is legacy exploratory/non-comparable.")
    parser.add_argument("--oracle-ability", action="store_true",
                        help="EXPLORATORY NON-COMPARABLE: route from the oracle "
                             "ability label instead of production detection.")
    parser.add_argument("--rejudge", default="",
                        help="Judge-only rejudge of an existing results artifact "
                             "(B in the gold-delta plan). Answer bytes are fixed "
                             "from the stored rows; no answer calls. Gold comes "
                             "from a deterministic dataset reparse guarded "
                             "160/160. Reads output with --judge-gold to feed "
                             "real gold instead of the legacy IDEAL ANSWER.")
    parser.add_argument("--keep-db", action="store_true")
    add_strict_run_arguments(parser)
    args = parser.parse_args()
    if args.sample < 0:
        parser.error("--sample must be non-negative (0 means all conversations)")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be positive")
    if args.indexing_max_cycles <= 0:
        parser.error("--indexing-max-cycles must be positive")
    if not math.isfinite(args.indexing_timeout_s) or args.indexing_timeout_s <= 0:
        parser.error("--indexing-timeout-s must be positive and finite")
    try:
        args.embedding_config = resolve_embedding_config(
            args.embedding_backend, model=args.embedding_model,
            base_url=args.embedding_base_url, dimension=args.embedding_dim,
        )
    except BenchmarkIntegrityError as exc:
        parser.error(str(exc))
    if args.judge_model is None:
        args.judge_model = (
            OFFICIAL_JUDGE_SPEC
            if args.judge_protocol == "official"
            else JUDGE_MODEL
        )
    judge_provider, judge_model, judge_base = parse_provider_spec(args.judge_model)

    # Parse the extra-body flags into the *_obj attrs the guards and clients
    # read. Done before anything spends money: a typo'd JSON should cost a
    # syntax error, not a partial run.
    for role in ("answer", "judge"):
        raw = getattr(args, f"{role}_extra_body")
        try:
            obj = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            print(f"ERROR: --{role}-extra-body is not valid JSON: {e}")
            sys.exit(2)
        if not isinstance(obj, dict):
            print(f"ERROR: --{role}-extra-body must be a JSON object, got {type(obj).__name__}.")
            sys.exit(2)
        try:
            obj = validate_request_extra_body(obj)
        except BenchmarkIntegrityError as exc:
            parser.error(f"--{role}-extra-body: {exc}")
        setattr(args, f"{role}_extra_body_obj", obj)
        setattr(args, f"{role}_extra_body_absent", raw is None)

    # §6, and the ordering is load-bearing: strictly BETWEEN the parse loop and
    # check_model_pin. Applied after the guard, a bare v4-flash run is refused
    # before the default can fire and §6 silently does nothing. The answer role
    # is defaulted further down, where its provider is finally resolved.
    args.extra_body_defaulted = []
    args.judge_extra_body_obj, _judge_defaulted = apply_thinking_default(
        "judge", judge_model, judge_provider,
        args.judge_extra_body_absent, args.judge_extra_body_obj)
    if _judge_defaulted:
        args.extra_body_defaulted.append("judge")
        print(f"judge extra_body DEFAULTED to {args.judge_extra_body_obj} "
              f"(v4-flash, no --judge-extra-body passed)")

    # Pin the spec before the money. Requiring one of the two flags is the
    # point: a default would let a canonical run be produced by forgetting.
    if bool(args.prereg) == bool(args.no_prereg):
        print("ERROR: pass exactly one of --prereg <path> or --no-prereg.")
        sys.exit(2)
    args.prereg_obj = resolve_prereg(None if args.no_prereg else args.prereg)
    if args.prereg_obj:
        print(f"pre-registration: {args.prereg_obj['path']} @ "
              f"{args.prereg_obj['commit'][:8]} (committed {args.prereg_obj['committed_at']})")
    else:
        print("pre-registration: NONE (--no-prereg) — this run is exploratory.")

    def _resolve_deepseek_key() -> str:
        key = args.api_key or os.environ.get("HYMEM_LLM_API_KEY", "")
        if not key:
            config_path = Path("/home/node/.hermes/config.yaml")
            if config_path.exists():
                for line in config_path.read_text().split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("HYMEM_LLM_API_KEY:"):
                        key = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                        break
        return key

    if args.rejudge:
        if args.judge_protocol != "legacy-custom":
            parser.error(
                "historical --rejudge currently requires --judge-protocol "
                "legacy-custom; official strict runs use the resumable main path"
            )
        DEEPSEEK_API_KEY = _resolve_deepseek_key()
        if not DEEPSEEK_API_KEY:
            parser.error("--rejudge requires a DeepSeek API key")
        _rejudge_run(args, DEEPSEEK_API_KEY)
        return

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    if not scales:
        parser.error("--scales must contain at least one scale")
    if len(scales) != len(set(scales)):
        parser.error("--scales must not contain duplicates")
    unknown_scales = [scale for scale in scales if scale not in VALID_BEAM_SCALES]
    if unknown_scales:
        parser.error(
            f"unsupported BEAM scale(s): {unknown_scales}; "
            f"choose from {list(VALID_BEAM_SCALES)}"
        )
    if (args.freeze_calibration or args.protocol_split != "full") and args.sample:
        parser.error(
            "--freeze-calibration and dev/holdout require --sample 0 so the "
            "receipt covers the complete selected scale(s)"
        )
    max_conv = args.sample if args.sample > 0 else None
    top_k = args.top_k

    ans_provider, ans_model, ans_base = parse_provider_spec(args.answer_model)
    args.answer_extra_body_obj, _answer_defaulted = apply_thinking_default(
        "answer", ans_model, ans_provider,
        args.answer_extra_body_absent, args.answer_extra_body_obj)
    if _answer_defaulted:
        args.extra_body_defaulted.append("answer")
        print(f"answer extra_body DEFAULTED to {args.answer_extra_body_obj} "
              f"(v4-flash, no --answer-extra-body passed)")
    check_model_pin("answer", ans_model, ans_provider, args.answer_extra_body_obj)
    check_model_pin(
        "judge", judge_model, judge_provider, args.judge_extra_body_obj
    )
    official_judge_match = is_official_judge_configuration(
        protocol=args.judge_protocol,
        provider=judge_provider,
        model=judge_model,
        base_url=judge_base,
        extra_body=args.judge_extra_body_obj,
    )
    full_scale_run = max_conv is None
    official_protocol_aligned = bool(
        official_judge_match and not args.oracle_ability
        and args.protocol_split == "full"
    )
    canonical_dataset_run = bool(
        full_scale_run and official_protocol_aligned and args.prereg_obj
    )

    print(f"\nHyMem BEAM Benchmark (direct API)")
    print(f"  Scales: {scales}")
    print(f"  Max conversations: {max_conv or 'all'}")
    print(f"  Top-K: {top_k}")
    print(f"  Answer model: {ans_model} (provider={ans_provider}, base={ans_base}, "
          f"extra_body={args.answer_extra_body_obj or '{}'})")
    print(f"  Judge model: {judge_model} (provider={judge_provider}, "
          f"protocol={args.judge_protocol}, "
          f"extra_body={args.judge_extra_body_obj or '{}'})")
    print(
        "  Embeddings: "
        f"{args.embedding_config['backend']} "
        f"(model={args.embedding_config['model']}, "
        f"dim={args.embedding_config['dimension']}, "
        f"quality={args.embedding_config['quality']})"
    )

    # Dataset and frozen run identity are resolved before any provider client or
    # API key. Calibration can therefore never touch the evaluation model.
    print("Loading BEAM dataset...", flush=True)
    dataset_revisions = resolve_dataset_revisions(scales, args.dataset_revision)
    print(f"  dataset revisions: {dataset_revisions}", flush=True)
    unresolved_revisions = validate_dataset_revision_binding(
        dataset_revisions,
        canonical=canonical_dataset_run,
    )
    conversations = load_beam_conversations(
        scales, max_conv, seed=args.seed, revisions=dataset_revisions
    )
    total_convs = sum(len(v) for v in conversations.values())
    total_questions = sum(len(c["questions"]) for v in conversations.values() for c in v)
    print(f"  Total: {total_convs} conversations, {total_questions} questions")
    if full_scale_run and official_protocol_aligned:
        validate_official_denominators(conversations, scales)
    print_gold_audit(conversations)
    flat_questions = [
        q for scale in scales for conv in conversations.get(scale, ())
        for q in conv["questions"]
    ]
    all_ids = validate_ids(
        (q["question_id"] for q in flat_questions), label="BEAM dataset"
    )
    if args.judge_protocol == "legacy-custom" and args.judge_gold:
        missing_gold = [
            q["question_id"] for q in flat_questions
            if not isinstance(q.get("gold_text"), str) or not q["gold_text"].strip()
        ]
        if missing_gold:
            raise BenchmarkIntegrityError(
                "canonical BEAM gold is absent for question ids: "
                f"{missing_gold[:5]}"
            )
        recovered_gold = [
            q["question_id"] for q in flat_questions
            if q.get("gold_resolution") != "exact"
        ]
        if recovered_gold:
            raise BenchmarkIntegrityError(
                "canonical BEAM gold did not resolve from the exact expected "
                f"ability field for question ids: {recovered_gold[:5]}"
            )

    strict_config = {
        "scales": scales,
        "sample": max_conv,
        "sample_strategy": "seeded-label-blind-hash-v1" if max_conv else "all",
        "subset_run": not full_scale_run,
        "top_k": top_k,
        "max_input_tokens": args.max_input_tokens,
        "indexing_max_cycles": args.indexing_max_cycles,
        "indexing_timeout_s": args.indexing_timeout_s,
        "indexing_require_healthy": True,
        "embedding": public_embedding_config(args.embedding_config),
        "facts": args.facts,
        "facts_extraction": args.facts_extraction,
        "judge_gold": bool(args.judge_gold),
        "judge_protocol": args.judge_protocol,
        "official_judge_protocol_match": official_judge_match,
        "official_protocol_aligned": official_protocol_aligned,
        "official_denominator_validated": bool(
            full_scale_run and official_protocol_aligned
        ),
        "official_judge_prompt_hash": BEAM_OFFICIAL_JUDGE_PROMPT_HASH,
        "official_judge_upstream_commit": BEAM_UPSTREAM_COMMIT,
        "official_judge_evaluator_url": BEAM_OFFICIAL_EVALUATOR_URL,
        "official_judge_prompt_url": BEAM_OFFICIAL_PROMPT_URL,
        "oracle_ability": bool(args.oracle_ability),
        "label_free_answer_path": not args.oracle_ability,
        "scored_run": True,
        "exploratory_label_steering": bool(args.oracle_ability),
        "exploratory_non_comparable": bool(
            args.oracle_ability or not args.judge_gold or args.no_prereg
            or unresolved_revisions or not official_judge_match
            or not full_scale_run or args.protocol_split != "full"
        ),
        "answer_extra_body": args.answer_extra_body_obj,
        "judge_extra_body": args.judge_extra_body_obj,
        "extra_body_defaulted": list(args.extra_body_defaulted),
        "prereg": args.prereg_obj,
        "dataset_revisions": dataset_revisions,
        "dataset_revision_provenance_complete": not unresolved_revisions,
    }
    config_probe = HyMemAdapter(
        Path("/benchmark-identity/hymem.sqlite"), facts_enabled=args.facts,
        facts_extraction=args.facts_extraction,
        pipeline_model=args.hymem_model,
        pipeline_base_url=args.hymem_base_url,
        pipeline_thinking=args.hymem_thinking,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_dim=args.embedding_dim,
    ).build_config()
    strict_config["effective_hymem_config"] = dataclass_identity(
        config_probe, exclude={"root"}
    )
    pipeline_sends_thinking = (
        args.hymem_thinking == "disabled"
        or (
            args.hymem_thinking == "auto"
            and (
                "deepseek" in args.hymem_base_url.casefold()
                or "deepseek" in args.hymem_model.casefold()
            )
        )
    )
    strict_models = {
        "reader": {"provider": ans_provider, "model": ans_model,
                   "base_url": ans_base},
        "judge": {
            "provider": judge_provider, "model": judge_model,
            "base_url": judge_base, "temperature": 0.0,
            "max_tokens": None if args.judge_protocol == "official" else 512,
            "extra_body": args.judge_extra_body_obj,
            "protocol": args.judge_protocol,
            "upstream_commit": (
                BEAM_UPSTREAM_COMMIT
                if args.judge_protocol == "official" else None
            ),
            "prompt_hash": (
                BEAM_OFFICIAL_JUDGE_PROMPT_HASH
                if args.judge_protocol == "official"
                else content_hash(JUDGE_SYSTEM_PROMPT)
            ),
        },
        "memory_pipeline": {
            "provider": "openai-compatible", "model": args.hymem_model,
            "base_url": args.hymem_base_url,
            "thinking_mode": args.hymem_thinking,
            "effective_extra_body": (
                {"thinking": {"type": "disabled"}}
                if pipeline_sends_thinking else {}
            ),
        },
        "embedding": public_embedding_config(args.embedding_config),
    }
    dataset_sha = content_hash({
        "revisions": dataset_revisions,
        "conversations": conversations,
    })
    if args.freeze_calibration:
        receipt = freeze_calibration(
            args.freeze_calibration, benchmark="BEAM", dataset_hash=dataset_sha,
            ids=all_ids, config=strict_config, models=strict_models,
            seed=args.seed, dev_fraction=args.dev_fraction,
        )
        print(f"Frozen BEAM calibration: dev={len(receipt['dev_ids'])}, "
              f"holdout={len(receipt['holdout_ids'])}")
        return

    calibration = None
    if args.calibration_receipt:
        calibration = load_calibration(
            args.calibration_receipt, benchmark="BEAM",
            dataset_hash=dataset_sha, config=strict_config,
            models=strict_models, ids=all_ids,
        )
    selected_ids = select_protocol_ids(
        all_ids, split=args.protocol_split, receipt=calibration
    )
    selected_set = set(selected_ids)
    selected_conversations: dict[str, list[dict]] = {}
    for scale in scales:
        selected_conversations[scale] = []
        for original in conversations.get(scale, ()):
            selected_questions = [
                q for q in original["questions"] if q["question_id"] in selected_set
            ]
            if selected_questions:
                selected_conversations[scale].append({
                    **original, "questions": selected_questions,
                })
    selected_order = tuple(
        q["question_id"] for scale in scales
        for conv in selected_conversations.get(scale, ())
        for q in conv["questions"]
    )
    if selected_order != selected_ids:
        raise BenchmarkIntegrityError("selected BEAM id order drifted")

    manifest = build_manifest(
        benchmark="BEAM",
        code_sha256=code_hash(
            [Path(__file__), Path(__file__).with_name("strictness.py"),
             Path(__file__).with_name("longmemeval_adapter.py"),
             _repo_root / "hymem"], root=_repo_root,
        ),
        data_sha256=dataset_sha, config=strict_config, models=strict_models,
        seed=args.seed, expected_ids=selected_ids,
        protocol_split=args.protocol_split, calibration=calibration,
    )
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, is_resume = resolve_checkpoint_path(
        checkpoint=args.checkpoint, resume_from=args.resume_from,
        base_dir=results_dir, benchmark="beam", run_id=manifest["run_id"],
    )
    ledger = AtomicCheckpoint(
        checkpoint_path, manifest=manifest, expected_ids=selected_ids,
        resume=is_resume, retry_failures=args.retry_failures,
        verdict_key="result_valid",
    )
    if _owned_ledgers is not None:
        _owned_ledgers.append(ledger)
    pending = set(ledger.pending_ids)
    print(f"  Strict checkpoint: {checkpoint_path} "
          f"({len(pending)} pending / {len(selected_ids)} expected)")

    start_time = time.time()
    segment_id = (
        f"process-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{os.getpid()}"
    )
    answer_llm = judge_llm = None
    hy = None
    tmp_dir = None
    attempted = 0
    lifecycle_errors: list[str] = []
    pipeline_usage_instances: list[dict[str, Any]] = []
    embedding_usage_instances: list[dict[str, Any]] = []
    indexing_runs: list[dict[str, Any]] = []

    def _segment(status: str) -> dict:
        pipeline_rows = list(pipeline_usage_instances)
        embedding_rows = list(embedding_usage_instances)
        if hy is not None:
            pipeline_rows.append(usage_snapshot(hy.pipeline_llm))
            embedding_rows.append(embedding_usage_snapshot(
                hy.embedding_client,
                configured=bool(args.embedding_config["configured"]),
            ))
        return {
            "segment_id": segment_id, "status": status,
            "elapsed_s": time.time() - start_time,
            "attempted_attempts": attempted,
            "reader_usage": usage_snapshot(answer_llm),
            "judge_usage": usage_snapshot(judge_llm),
            "memory_pipeline_usage": (
                aggregate_usage_snapshots(pipeline_rows)
                if pipeline_rows else usage_snapshot(None)
            ),
            "embedding_usage": (
                aggregate_embedding_usage_snapshots(embedding_rows)
                if embedding_rows else embedding_usage_snapshot(
                    None, configured=bool(args.embedding_config["configured"])
                )
            ),
            "latest_indexing": (
                hy.last_indexing_summary if hy is not None
                else (indexing_runs[-1] if indexing_runs else None)
            ),
            "indexing_runs": [dict(item) for item in indexing_runs],
        }

    def _record(row: dict) -> None:
        nonlocal attempted
        attempted += 1
        ledger.record(
            row["question_id"], row=row,
            execution_segment=_segment("running"),
        )

    try:
        if pending:
            DEEPSEEK_API_KEY = _resolve_deepseek_key()
            if not DEEPSEEK_API_KEY:
                parser.error(
                    "pending BEAM questions require --api-key or HYMEM_LLM_API_KEY"
                )
            ans_model_live, ans_base_live, ans_key, provider_live = \
                resolve_answer_provider(args.answer_model, DEEPSEEK_API_KEY)
            if (ans_model_live, ans_base_live, provider_live) != \
                    (ans_model, ans_base, ans_provider):
                raise BenchmarkIntegrityError("answer-provider identity drifted")
            judge_model_live, judge_base_live, judge_key, judge_provider_live = \
                resolve_answer_provider(
                    args.judge_model, DEEPSEEK_API_KEY, role="judge"
                )
            if (judge_model_live, judge_base_live, judge_provider_live) != \
                    (judge_model, judge_base, judge_provider):
                raise BenchmarkIntegrityError("judge-provider identity drifted")
            answer_llm = LLMClient(
                ans_model, ans_key, base_url=ans_base,
                extra_body=args.answer_extra_body_obj,
            )
            judge_llm = LLMClient(
                judge_model, judge_key, base_url=judge_base,
                extra_body=args.judge_extra_body_obj,
            )
            ledger.update_execution_segment(segment_id, _segment("running"))

            # Canary spend is persisted immediately in the cumulative segment.
            canary_msgs = build_canary_messages(
                selected_conversations, scales, args.judge_gold,
                judge_protocol=args.judge_protocol,
            )
            canary_answer = _run_canary_with_checkpoint(
                ledger, segment_id, _segment,
                "answer", answer_llm, canary_msgs["answer"], 1024,
            )
            _run_canary_with_checkpoint(
                ledger, segment_id, _segment,
                "judge", judge_llm,
                canary_msgs["judge"](canary_answer),
                None if args.judge_protocol == "official" else 512,
            )

            for scale in scales:
                convs = selected_conversations.get(scale, ())
                for ci, conv in enumerate(convs):
                    conv_pending = {
                        q["question_id"] for q in conv["questions"]
                        if q["question_id"] in pending
                    }
                    if not conv_pending:
                        continue
                    print(f"  [{ci+1}/{len(convs)}] Conv {conv['id']}", flush=True)
                    # Every BEAM conversation is an independent memory/user.
                    # A fresh store prevents graph evidence counts, retention
                    # budgets, inference and ranking state from one example
                    # changing another example's write-side representation.
                    tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-beam-conv-"))
                    hy = HyMemAdapter(
                        tmp_dir / "hymem.sqlite", api_key=DEEPSEEK_API_KEY,
                        facts_enabled=args.facts,
                        facts_extraction=args.facts_extraction,
                        pipeline_model=args.hymem_model,
                        pipeline_base_url=args.hymem_base_url,
                        pipeline_thinking=args.hymem_thinking,
                        embedding_backend=args.embedding_backend,
                        embedding_model=args.embedding_model,
                        embedding_base_url=args.embedding_base_url,
                        embedding_dim=args.embedding_dim,
                        embedding_api_key=args.embedding_api_key,
                    )
                    try:
                        hy.open()
                        evaluate_conversation(
                            args.judge_gold, answer_llm, judge_llm, hy, conv,
                            top_k, oracle_ability=args.oracle_ability,
                            judge_protocol=args.judge_protocol,
                            pending_ids=conv_pending, on_result=_record,
                            indexing_max_cycles=args.indexing_max_cycles,
                            indexing_timeout_s=args.indexing_timeout_s,
                            max_input_tokens=args.max_input_tokens,
                        )
                    except Exception as exc:
                        # Ingest/dream failures occur before the per-question
                        # callback; materialize every affected expected id.
                        remaining_now = set(ledger.pending_ids)
                        for q in conv["questions"]:
                            if q["question_id"] not in remaining_now:
                                continue
                            _record({
                                "question_id": q["question_id"],
                                "scale": scale, "conv_id": conv["id"],
                                "ability": q["ability_short"],
                                "oracle_ability": q["ability_short"],
                                "detected_ability": None,
                                "ability_used": None,
                                "question": q["question"],
                                "score": 0.0, "scores": [],
                                "llm_judge_score": 0.0,
                                "judge_protocol": args.judge_protocol,
                                "result_valid": False, "correct": False,
                                "benchmark_failure": (
                                    f"conversation_failure: {type(exc).__name__}: {exc}"
                                ),
                                "indexing": (
                                    hy.last_indexing_summary if hy is not None else None
                                ),
                            })
                    finally:
                        if hy is not None:
                            pipeline_usage_instances.append(
                                usage_snapshot(hy.pipeline_llm)
                            )
                            embedding_usage_instances.append(
                                embedding_usage_snapshot(
                                    hy.embedding_client,
                                    configured=bool(
                                        args.embedding_config["configured"]
                                    ),
                                )
                            )
                            if hy.last_indexing_summary is not None:
                                indexing_runs.append({
                                    "scale": scale,
                                    "conversation_id": conv["id"],
                                    **dict(hy.last_indexing_summary),
                                })
                            try:
                                hy.close()
                            except Exception as exc:
                                lifecycle_errors.append(
                                    "adapter_close "
                                    f"{scale}/{conv['id']}: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                        if tmp_dir is not None and not args.keep_db:
                            try:
                                import shutil
                                shutil.rmtree(tmp_dir, ignore_errors=False)
                            except Exception as exc:
                                lifecycle_errors.append(
                                    "temporary_store_cleanup "
                                    f"{scale}/{conv['id']}: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                        hy = None
                        tmp_dir = None
                        ledger.update_execution_segment(
                            segment_id, _segment("running")
                        )
        if pending:
            ledger.update_execution_segment(segment_id, _segment("complete"))
    finally:
        if hy is not None:
            try:
                hy.close()
            except Exception as exc:
                lifecycle_errors.append(
                    f"adapter_close: {type(exc).__name__}: {exc}"
                )
        if tmp_dir is not None and not args.keep_db:
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=False)
            except Exception as exc:
                lifecycle_errors.append(
                    f"temporary_store_cleanup: {type(exc).__name__}: {exc}"
                )

    try:
        elapsed = time.time() - start_time
        payload, summary, all_results = _strict_beam_payload(
            ledger, selected_conversations, scales,
            label_free=not args.oracle_ability, judge_gold=args.judge_gold,
            official_judge_match=official_judge_match,
            lifecycle_errors=lifecycle_errors,
        )
        payload["elapsed_s"] = elapsed
        archive_now = datetime.now(timezone.utc)
        stamp = archive_now.strftime("%Y%m%dT%H%M%SZ")
        publication_nonce = archive_now.strftime("%f")
        archive_path = results_dir / (
            f"results_{stamp}-{publication_nonce}-strict-"
            f"{manifest['run_id'].removeprefix('sha256:')[:12]}.json"
        )
        from benchmarks.strictness import publish_checkpoint_artifact
        publish_checkpoint_artifact(ledger, archive_path, payload=payload)
        latest_path = results_dir / "results_latest.json"
        write_latest_pointer(latest_path, archive=archive_path,
                             run_id=manifest["run_id"])

        # Optional presentation occurs after immutable publication.
        print(f"Evaluation complete in {elapsed:.0f}s")
        print_report(summary, {
            "answer_model": args.answer_model, "judge_model": args.judge_model,
            "sample_size": max_conv, "top_k": top_k,
        })
        print_episode_probe(all_results)
        print(f"\nResults saved to {archive_path}")
    finally:
        ledger.close()


def main():
    """CLI entry point that owns every checkpoint lease for its full lifetime."""

    owned_ledgers: list[AtomicCheckpoint] = []
    try:
        return _main(owned_ledgers)
    finally:
        # Covers BaseException/SystemExit from client creation, canaries,
        # evaluation, checkpoint I/O, publication and presentation. Inner
        # success-path closes are idempotent.
        for ledger in reversed(owned_ledgers):
            ledger.close()


if __name__ == "__main__":
    main()
