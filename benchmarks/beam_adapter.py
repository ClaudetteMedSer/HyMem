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
import subprocess
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


def _resolve_gold(q: dict, ability: str) -> tuple[str, str]:
    """(text, kind) for one question, loudly rather than silently.

    A miss falls back to scanning every known key -- not to paper over the map,
    but so the WARN names the keys the row actually has. A row that resolves to
    nothing returns kind "none", which is a value the probe and the gold audit
    both check for; it is never an empty string standing in for an answer.
    """
    field, kind = GOLD_FIELDS.get(ability, (None, None))
    if field:
        text = (q.get(field) or "")
        if isinstance(text, (list, tuple)):
            text = " ".join(str(t) for t in text)
        if str(text).strip():
            return str(text), kind
    present = [k for k in _ALL_GOLD_KEYS if str(q.get(k) or "").strip()]
    if present:
        recovered = str(q.get(present[0]))
        warn = (ability, field, present[0])
        if warn not in _gold_warnings:
            _gold_warnings.add(warn)
            print(f"  WARN gold-field map miss: ability={ability!r} expected "
                  f"{field!r}, recovered from {present[0]!r}. Update "
                  f"GOLD_FIELDS -- a recovered field's KIND is a guess.",
                  flush=True)
        return recovered, (kind or "unknown")
    return "", "none"


PUBLISHED_SOTA = {
    "100K": {"Hindsight": 73.4, "Honcho": 63.0, "Mnemosyne v3": 65.2, "LIGHT": 35.8, "RAG": 32.3},
    "500K": {"Hindsight": 71.1, "Honcho": 64.9, "LIGHT": 35.9, "RAG": 33.0},
    "1M":   {"Hindsight": 73.9, "Honcho": 63.1, "LIGHT": 33.6, "RAG": 30.7},
    "10M":  {"Hindsight": 64.1, "Honcho": 40.6, "LIGHT": 26.6, "RAG": 24.9},
}


# ── LLM Client ────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str, api_key: str, base_url: str = DEEPSEEK_BASE_URL,
                 extra_body: dict | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.call_count = 0
        # Provider-specific request fields (DeepSeek's `thinking` switch is the
        # only one in use). EMPTY BY DEFAULT and only ever set from an explicit
        # flag: an unflagged run must send the same bytes it sent before this
        # plumbing existed, or every prior artifact silently stops being a
        # comparator. That is why this is not the `auto` host-substring gate
        # hymem/contrib/openai_client.py:81-86 uses for the library client --
        # a benchmark cannot afford a request-body change it did not ask for.
        self.extra_body = dict(extra_body or {})
        # Why the response ENDED, kept from the last call. B2 (2026-09-01) hit
        # a silent-0 whose cause -- the judge scoring 1.0 and then running out
        # of tokens mid-explanation -- was only recoverable by eyeballing the
        # raw text. finish_reason == "length" says the same thing structurally,
        # which is what a gate needs if it is to separate "the plumbing broke"
        # from "the judge ran long" without a human reading prose. Purely
        # additive: it records why a call ended and changes no score and no
        # request byte. Single-threaded client, so plain attribute state.
        self.last_finish_reason = None

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        # Cleared per call, so a stale value from an earlier row can never be
        # read as this row's -- absence must look like absence.
        self.last_finish_reason = None
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
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        # Merge last so a caller can force provider-specific fields; collisions
        # with the four keys above are the caller's (mirrors LME :373).
        body.update(self.extra_body)
        resp = http.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        self.call_count += 1
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
        return content


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


THINKING_DISABLED = {"thinking": {"type": "disabled"}}


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


def run_canary(role: str, llm: LLMClient, messages: list, max_tokens: int) -> str:
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


def build_canary_messages(conversations: dict, scales: list, judge_gold: bool) -> dict:
    """Assemble both canary prompts without calling anything.

    Split out of main() so the ASSEMBLY is testable. Stubbing chat() -- which
    is how run_canary's own unit tests work -- exercises the send and the
    verdict but never the construction, so a wrong constant name or a question
    key that does not exist would survive a green suite and only surface when a
    real run reached this line, after ingestion had already been paid for.
    """
    q = pick_canary_question(conversations, scales)
    ideal = q.get("gold_text", "") if judge_gold else q["ideal_answer"]
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
        "judge": lambda ai_answer: _judge_messages(q["question"], ideal, q["rubric"], ai_answer),
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
    out = {}
    for scale in scales:
        repo = beam_repo(scale)
        if repo in out:
            continue
        if pin:
            out[repo] = pin
            continue
        try:
            from huggingface_hub import HfApi
            out[repo] = HfApi().dataset_info(repo).sha
        except Exception as e:
            out[repo] = None
            print(f"  WARNING: could not resolve {repo} revision "
                  f"({type(e).__name__}: {e}); recorded as null — this artifact "
                  "cannot witness the dataset it was scored on.", flush=True)
    return out


def load_beam_conversations(scales: list[str], max_conv: int = None,
                            revision: str | None = None) -> dict:
    from datasets import load_dataset

    data = {}
    for scale in scales:
        print(f"  Loading BEAM {scale}...", flush=True)
        if scale == "10M":
            ds = load_dataset(BEAM_REPO_10M, streaming=True, revision=revision)
            split_name = list(ds.keys())[0]
            conversations = []
            for i, sample in enumerate(ds[split_name]):
                if max_conv and i >= max_conv:
                    break
                conversations.append(_parse_sample(sample, scale, i))
            data[scale] = conversations
        else:
            ds = load_dataset(BEAM_REPO, streaming=False, revision=revision)
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
                    _gold = _resolve_gold(q, ability)
                    all_questions.append({
                        "ability": ability,
                        "ability_short": ABILITY_MAP.get(ability, ability[:3].upper()),
                        "question_id": q.get("question_id") or q.get("id") or "",
                        "question": q.get("question", ""),
                        # `ideal_answer` intentionally keeps the ORIGINAL
                        # (mostly empty) parse, because it is what the judge
                        # reads: repointing the judge changes what every BEAM
                        # score MEANS and is a separate pre-registered decision
                        # (--judge-gold), not a side effect of fixing the probe.
                        "ideal_answer": q.get("ideal_response",
                                              q.get("ideal_answer", "")),
                        "gold_text": _gold[0],
                        "gold_kind": _gold[1],
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
        # RAPTOR aggregation layer: pinned OFF explicitly. The config default
        # flipped False -> True on 2026-08-26 (G-FLIP PASS); this benchmark was
        # a default-config consumer, so without the pin the flip would silently
        # switch the layer + digest ON and the canonical baseline would stop
        # being comparable to every run behind it. Moving a benchmark onto the
        # shipped config is a pre-registered scored decision, not a side effect
        # of a default change.
        overrides["aggregation_nodes_enabled"] = False
        # Same reasoning, one lever earlier: `episode_granularity_enabled`
        # is under active decision (Plan C) and this adapter was the last
        # unpinned default-config consumer of it. The pin matches the
        # current default, so it changes nothing today -- that is the point.
        # It means the pre-check below reads a KNOWN arm, and a future
        # default flip cannot silently move BEAM off its baseline the way
        # the aggregation flip nearly did.
        overrides["episode_granularity_enabled"] = False
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


def judge_answer(llm: LLMClient, question: str, ideal: str, rubric: list, ai_answer: str,
                 return_raw: bool = False) -> dict:
    if not rubric:
        return {"score": 0.0, "scores": []}

    messages = _judge_messages(question, ideal, rubric, ai_answer)

    raw = llm.chat(messages, temperature=0.0, max_tokens=512)
    result, how = extract_judge_json(raw)
    if result is not None:
        scores = result.get("scores", [])
        total = sum(scores) / len(scores) if scores else 0.0
        out = {"score": total, "scores": scores}
    else:
        out = {"score": 0.0, "scores": []}
    if return_raw:
        out["judge_raw"] = raw
        out["judge_finish_reason"] = _finish_reason(llm)
        out["judge_parse"] = how
    return out


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_conversation(judge_gold: bool, llm: LLMClient, judge_llm: LLMClient, hy: HyMemAdapter,
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
        # Default keeps the ORIGINAL (mostly empty) gold so this run stays
        # comparable to v13-v16. --judge-gold is the pre-registered switch: it
        # changes what the score MEANS, so post-flip runs need a fresh baseline
        # and the old canonical retires as a comparison point rather than
        # reading as a regression or an improvement.
        _judge_ideal = q.get("gold_text", "") if judge_gold else q["ideal_answer"]
        judge_result = judge_answer(judge_llm, question, _judge_ideal, q["rubric"], answer)
        print(f"      Score: {judge_result['score']:.2f}")

        results.append({
            "ability": ability,
            "question": question,
            "answer": answer,
            "ideal_answer": q["ideal_answer"],
            "rubric": q["rubric"],
            "score": judge_result["score"],
            "scores": judge_result["scores"],
            # Plan C pre-check. Additive record only -- computed from the
            # already-returned `memories`, so it cannot affect the answer or
            # the score, and it costs no extra call.
            "gold_kind": q.get("gold_kind", "none"),
            "probe": episode_probe(memories, question, _probe_gold(q),
                                   _decoy_answer(conv["questions"], qi)),
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
    data = load_beam_conversations(["100K"], max_conv=8,
                                   revision=args.dataset_revision)
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
        ideal = gold["gold_text"] if args.judge_gold else r.get("ideal_answer", "")
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
    parser.add_argument("--answer-extra-body", default="",
                        help="JSON merged into every ANSWER request body. Required "
                             "as '{\"thinking\": {\"type\": \"disabled\"}}' when the "
                             "answerer is a DeepSeek v4-flash model; rejected for "
                             "non-DeepSeek providers, which 400 on that key.")
    parser.add_argument("--judge-extra-body", default="",
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
    parser.add_argument("--judge-gold", action="store_true",
                        help="Feed the judge the REAL per-ability gold answer "
                             "(GOLD_FIELDS) instead of the legacy parse, which "
                             "resolved for 2 of 10 abilities and sent an empty "
                             "IDEAL ANSWER field for the rest from 145eff8 "
                             "onward. This changes what every BEAM score means: "
                             "runs with it on are NOT comparable to v13-v16 and "
                             "need their own baseline.")
    parser.add_argument("--rejudge", default="",
                        help="Judge-only rejudge of an existing results artifact "
                             "(B in the gold-delta plan). Answer bytes are fixed "
                             "from the stored rows; no answer calls. Gold comes "
                             "from a deterministic dataset reparse guarded "
                             "160/160. Reads output with --judge-gold to feed "
                             "real gold instead of the legacy IDEAL ANSWER.")
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()

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
        setattr(args, f"{role}_extra_body_obj", obj)

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

    if args.rejudge:
        _rejudge_run(args, DEEPSEEK_API_KEY)
        return

    scales = [s.strip() for s in args.scales.split(",")]
    max_conv = args.sample if args.sample > 0 else None
    top_k = args.top_k

    print(f"\nHyMem BEAM Benchmark (direct API)")
    print(f"  Scales: {scales}")
    print(f"  Max conversations: {max_conv or 'all'}")
    print(f"  Top-K: {top_k}")
    ans_model, ans_base, ans_key, ans_provider = resolve_answer_provider(args.answer_model, DEEPSEEK_API_KEY)
    check_model_pin("answer", ans_model, ans_provider, args.answer_extra_body_obj)
    check_model_pin("judge", args.judge_model, "deepseek", args.judge_extra_body_obj)
    print(f"  Answer model: {ans_model} (provider={ans_provider}, base={ans_base}, "
          f"extra_body={args.answer_extra_body_obj or '{}'})")
    print(f"  Judge model: {args.judge_model} (provider=deepseek, "
          f"extra_body={args.judge_extra_body_obj or '{}'})")

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
    answer_llm = LLMClient(ans_model, ans_key, base_url=ans_base,
                           extra_body=args.answer_extra_body_obj)
    judge_llm = LLMClient(args.judge_model, DEEPSEEK_API_KEY,
                          extra_body=args.judge_extra_body_obj)

    # Load data
    print("Loading BEAM dataset...", flush=True)
    dataset_revisions = resolve_dataset_revisions(scales, args.dataset_revision)
    print(f"  dataset revisions: {dataset_revisions}", flush=True)
    conversations = load_beam_conversations(scales, max_conv,
                                            revision=args.dataset_revision)
    total_convs = sum(len(v) for v in conversations.values())
    total_questions = sum(len(c["questions"]) for v in conversations.values() for c in v)
    print(f"  Total: {total_convs} conversations, {total_questions} questions")
    print_gold_audit(conversations)

    # ── CANARY: both live clients, real prompts, before the run spends ─────
    # The rejudge path has canaried its judge since the gold-delta phase; the
    # main path -- the one that makes the expensive ANSWER calls -- had no such
    # check, so a pin landing in reasoning_content would have surfaced as a
    # suspiciously low score hours later, if at all.
    canary_msgs = build_canary_messages(conversations, scales, args.judge_gold)
    print(f"  canary question ability: {canary_msgs['ability']}")
    canary_answer = run_canary("answer", answer_llm, canary_msgs["answer"], 1024)
    run_canary("judge", judge_llm,
               canary_msgs["judge"](canary_answer), 512)
    # Counted separately, NOT folded into answer_calls/judge_calls. The rejudge
    # artifact set the opposite precedent (judge_calls=161 for 160 rows), but
    # that is a wart to stop repeating rather than a convention to cement: a
    # column named answer_calls should equal the number of questions scored, so
    # that "calls != rows" stays readable as a defect.
    canary_calls = {"canary_answer_calls": answer_llm.call_count,
                    "canary_judge_calls": judge_llm.call_count}

    # Evaluate
    all_results = []
    start_time = time.time()

    for scale in scales:
        if scale not in conversations:
            continue
        print(f"Evaluating {scale} ({len(conversations[scale])} conversations)...", flush=True)
        for ci, conv in enumerate(conversations[scale]):
            print(f"  [{ci+1}/{len(conversations[scale])}] Conv {conv['id']}", flush=True)
            result = evaluate_conversation(args.judge_gold, answer_llm, judge_llm,
                                           hy, conv, top_k)
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
    print_episode_probe(all_results)

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
            "answer_calls": answer_llm.call_count - canary_calls["canary_answer_calls"],
            "judge_calls": judge_llm.call_count - canary_calls["canary_judge_calls"],
            **canary_calls,
            # WHAT WAS SENT, not what the guard would have permitted. A reader
            # of answer_model="deepseek-v4-flash" cannot otherwise tell whether
            # thinking was disabled -- and that single field is the difference
            # between a capability number and a plumbing failure (lme_runs.db
            # id=53 0.6% vs id=54 69.8%). Inferring it from check_model_pin's
            # rules is an inference about the code that scored the run, which
            # is exactly the kind of thing artifacts exist to stop.
            "answer_extra_body": args.answer_extra_body_obj,
            "judge_extra_body": args.judge_extra_body_obj,
            "prereg": args.prereg_obj,
            "dataset_revisions": dataset_revisions,
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
