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
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests as http

# Add HyMem to path
# Ensure the HyMem package is importable (repo root is two levels up from benchmarks/)
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from hymem.contrib.endpoint_policy import (  # noqa: E402
    EMBEDDING_INTERNAL_HTTP_ENV,
    TRANSPORT_SECURITY_NONE,
    endpoint_transport_security,
    resolve_embedding_api_key,
    resolve_llm_api_key,
    safe_endpoint_label,
    secret_free_endpoint_identity,
    validate_http_endpoint,
)
from hymem.contrib.model_policy import (  # noqa: E402
    DeprecatedModelAliasError,
    require_active_model,
)

from benchmarks.strictness import (
    AtomicCheckpoint,
    BenchmarkCleanupError,
    BenchmarkIntegrityError,
    IndexingConvergenceError,
    OwnedResourceScope,
    PythonSourceSlice,
    aggregate_embedding_usage_snapshots,
    aggregate_usage_snapshots,
    add_strict_run_arguments,
    benchmark_hymem_source_paths,
    build_manifest,
    code_hash,
    content_hash,
    converge_indexing,
    durable_indexing_status,
    file_hash,
    freeze_calibration,
    is_structural_benchmark_error,
    load_calibration,
    prepare_checkpoint_artifact,
    publish_prepared_artifact_after_cleanup,
    python_file_imported_symbols,
    python_slice_imported_symbols,
    resolve_checkpoint_path,
    run_cleanup_actions,
    select_protocol_ids,
    sanitize_for_artifact,
    strict_accuracy,
    usage_snapshot,
    embedding_usage_snapshot,
    effective_hymem_config_identity,
    validate_ids,
    write_immutable_artifact,
    write_latest_pointer,
)
from benchmarks.lme_protocol import (
    LME_ABILITY_BY_TYPE,
    LME_BASE_QUESTION_TYPES,
    LME_EVALUATOR_COMMIT,
    LME_EVALUATOR_SHA256,
    LME_EVALUATOR_URL,
    LME_HISTORICAL_LOCAL_JUDGE_PROMPTS_EXACT_OFFICIAL,
    LME_INDEXING_SUMMARY_VERSION,
    LME_OFFICIAL_JUDGE_BASE_URL,
    LME_OFFICIAL_JUDGE_MAX_TOKENS,
    LME_OFFICIAL_JUDGE_MODEL,
    LME_OFFICIAL_JUDGE_TEMPERATURE,
    LME_OFFICIAL_VERDICT_PARSER,
    LME_LOCAL_RETRY_POLICY,
    LME_UPSTREAM_RETRY_POLICY,
    LME_S_DATASET_REVISION,
    LME_S_DATASET_SHA256,
    LME_S_DATASET_URL,
    LME_S_EXPECTED_COUNT,
    LME_S_QTYPE_COUNTS,
    LME_S_SOURCE_IDS_HASH,
    LME_SUPPORTED_SCALES,
    canonicalize_lme_indexing_summary,
    export_official_predictions,
    is_official_abstention_id,
    normalize_extra_body,
    normalize_lme_date,
    official_judge_match,
    parse_official_verdict,
    validate_lme_dataset,
    validate_safe_endpoint,
)
from benchmarks.extraction_canary import (
    ExtractionCanaryError,
    extraction_canary_client_policy,
    secret_free_extraction_canary_report,
    extraction_canary_policy,
    print_extraction_canary,
    run_configured_extraction_canary,
    skipped_extraction_canary,
    validate_extraction_canary_config_binding,
    validate_extraction_canary_report,
)


def _pipeline_extraction_prompt_version(args: argparse.Namespace) -> str:
    """Resolve the canary/cache prompt label from the effective adapter config."""

    cfg = _adapter_for_args(
        Path("/benchmark-identity/hymem.sqlite"), args, ""
    ).build_config()
    validate_extraction_canary_config_binding(
        extraction_canary_policy(prompt_version=cfg.prompt_version), cfg
    )
    return cfg.prompt_version


def _validate_pipeline_extraction_canary(
    report: object, args: argparse.Namespace, *, mode: str,
) -> dict[str, Any]:
    """Bind a live preflight to the exact memory-pipeline request identity."""

    prompt_version = _pipeline_extraction_prompt_version(args)
    return validate_extraction_canary_report(
        report,
        expected_mode=mode,
        expected_client=(
            extraction_canary_client_policy(
                base_url=args.hymem_base_url,
                model=args.hymem_model,
                thinking=args.hymem_thinking,
            )
            if mode == "required" else None
        ),
        require_client_closed=mode == "required",
        expected_prompt_version=prompt_version,
    )


def _bounded_exception_type(exc: BaseException) -> str:
    """Return only a bounded class identity for durable failure evidence."""

    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", name) else "Exception"

# ── Config ──────────────────────────────────────────────────────────

def _normalize_date(raw: str | None) -> str | None:
    """Convert a LongMemEval haystack_date like '2023/05/20 (Sat) 02:21'
    to ISO-8601 '2023-05-20T02:21:00'. Returns None for empty/None input."""
    if not raw or not raw.strip():
        return None
    return normalize_lme_date(raw, label="LongMemEval date")


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

    Message hits expose the complete raw turn; FTS hits may still be a chunked
    slice of one. So a match is: one string contains the other, or they share a
    distinctive 40-character prefix (covering chunk boundaries)."""
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

def _detect_ability(question: str) -> str | None:
    """HyMem's production ability inference, failing loudly if it is broken.

    ``None`` is a legitimate generic-route decision. Import/runtime failure is
    not: strict scored runs must never turn a broken router into that abstention
    signal and continue scoring.

    detect_ability emits only "MR"/"TR"/None by design — those are the only
    wired-in shaping paths; every other oracle ability (IE/KU/PF/ABS) correctly
    maps to a None inference (no shaping), so a None here on those categories is
    a correct abstain, not a miss."""
    from hymem.query.intent import detect_ability

    detected = detect_ability(question or "")
    if detected not in {"MR", "TR", None}:
        raise BenchmarkIntegrityError(
            f"production ability router returned invalid value: {detected!r}"
        )
    return detected


def _detect_ability_safe(question: str) -> str | None:
    """Best-effort shadow diagnostic; never use this to drive a scored path."""

    try:
        return _detect_ability(question)
    except Exception:
        return None


DEFAULT_SCALE = "S"
DEFAULT_SAMPLE = 50  # questions to evaluate (500 total)
DEFAULT_TOP_K = 15
# Historical compatibility constant only.  The strict adapter no longer uses
# this 8K/16K character bottleneck beneath its explicit token budget.
MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_INPUT_TOKENS = 16000
DEFAULT_MAX_INPUT_BYTES = 60000
# A deliberately conservative floor for the default DeepSeek reader.  The
# byte fallback relies on the mathematical ``tokens <= UTF-8 bytes`` bound, so
# input bytes plus the output reserve must fit inside this manifested ceiling.
DEFAULT_PROVIDER_CONTEXT_TOKENS = 65536
READER_OUTPUT_RESERVE_TOKENS = 1024
READER_TRANSPORT_OVERHEAD_TOKENS = 256
RAW_EVIDENCE_RESERVE_FRACTION = 0.60
MIN_SEMANTIC_EXCERPT_ALNUM = 8
MIN_SEMANTIC_EXCERPT_CHARS = 12
DEFAULT_INDEXING_MAX_CYCLES = 100
DEFAULT_INDEXING_TIMEOUT_S = 3600.0

# DeepSeek API
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PINNED_DEEPSEEK_MODEL = "deepseek-v4-flash"
ANSWER_MODEL = PINNED_DEEPSEEK_MODEL
JUDGE_MODEL = PINNED_DEEPSEEK_MODEL

# Local embedding server (lever L1) — the FastEmbed ONNX server Hermes runs in
# production. These are the OUT-OF-THE-BOX defaults for --embeddings so the flag
# works with no env setup; every field is still overridable via HYMEM_EMBEDDING_*.
# DeepSeek has no embeddings API, so this benchmark deliberately points at its
# own local FastEmbed service. api_key="local" because that service ignores it.
LOCAL_EMBED_BASE_URL = "http://localhost:8766/v1"
LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBED_DIM = 384
LOCAL_EMBED_API_KEY = "local"


def longmemeval_code_hash(
    *,
    adapter_path: Path | None = None,
    strictness_path: Path | None = None,
    archive_evidence_path: Path | None = None,
    protocol_path: Path | None = None,
    run_registry_path: Path | None = None,
    extraction_canary_path: Path | None = None,
    hymem_path: Path | None = None,
    root: Path | None = None,
) -> str:
    """Hash exact executable dependencies that can change LME evidence.

    Arguments are injectable so the identity dependency can be regression
    tested against temporary files without editing the working tree.
    """

    root_path = Path(root or _repo_root).resolve()
    benchmark_dir = Path(__file__).resolve().parent
    adapter = Path(adapter_path or __file__)
    protocol = Path(protocol_path or benchmark_dir / "lme_protocol.py")
    registry = Path(run_registry_path or benchmark_dir / "run_registry.py")
    protocol_symbols = python_file_imported_symbols(
        adapter,
        module_names=("benchmarks.lme_protocol", "lme_protocol"),
    )
    registry_symbols = python_file_imported_symbols(
        adapter,
        module_names=("benchmarks.run_registry", "run_registry"),
    )
    canary = Path(extraction_canary_path or benchmark_dir / "extraction_canary.py")
    canary_symbols = python_file_imported_symbols(
        adapter,
        module_names=("benchmarks.extraction_canary", "extraction_canary"),
    )
    dependency_slices: list[PythonSourceSlice] = []
    if protocol_symbols:
        dependency_slices.append(PythonSourceSlice(protocol, protocol_symbols))
    if registry_symbols:
        dependency_slices.append(PythonSourceSlice(registry, registry_symbols))
    if canary_symbols:
        dependency_slices.append(PythonSourceSlice(canary, canary_symbols))
    strictness = Path(strictness_path or benchmark_dir / "strictness.py")
    strictness_modules = ("benchmarks.strictness", "strictness")
    strictness_symbols = set(python_file_imported_symbols(
        adapter, module_names=strictness_modules
    ))
    for source_slice in dependency_slices:
        strictness_symbols.update(python_slice_imported_symbols(
            source_slice, module_names=strictness_modules
        ))
    if not strictness_symbols:
        raise BenchmarkIntegrityError("LME code identity lacks strictness imports")
    strictness_slice = PythonSourceSlice(
        strictness, tuple(strictness_symbols)
    )
    dependency_slices.append(strictness_slice)
    archive_symbols: set[str] = set()
    for source_slice in dependency_slices:
        archive_symbols.update(python_slice_imported_symbols(
            source_slice, module_names=("benchmarks.archive_evidence", "archive_evidence"),
        ))
    if archive_symbols:
        dependency_slices.append(PythonSourceSlice(
            Path(archive_evidence_path or benchmark_dir / "archive_evidence.py"),
            tuple(archive_symbols),
        ))
    dependency_sources: list[Path | PythonSourceSlice] = [
        adapter, *dependency_slices,
    ]
    inputs: list[Path | PythonSourceSlice] = [
        adapter,
        *dependency_slices,
        *benchmark_hymem_source_paths(
            Path(hymem_path or root_path / "hymem"),
            root=root_path,
            dependency_sources=dependency_sources,
        ),
    ]
    return code_hash(
        inputs,
        root=root_path,
    )


def _load_local_tokenizer_counter(path: Path):
    """Load a Hugging Face ``tokenizer.json`` without network access.

    ``tokenizers`` is intentionally optional. Selecting this strict path when
    the package or local file is unavailable fails before any provider client
    is constructed; the adapter never downloads or guesses a tokenizer.
    """

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BenchmarkIntegrityError(
            "--tokenizer-json requires the local 'tokenizers' package"
        ) from exc
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as exc:
        raise BenchmarkIntegrityError(
            f"cannot load local tokenizer JSON {path}: {exc}"
        ) from exc

    def count(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    return count


def resolve_context_policy(args, parser=None) -> tuple[dict[str, Any], Any]:
    """Resolve the truthful reader budget and optional exact local counter.

    The no-tokenizer policy is explicitly byte-denominated. Since a byte-level
    tokenizer cannot emit more tokens than UTF-8 input bytes, a byte budget plus
    output and chat-framing reserves below the declared provider ceiling is
    conservative without pretending bytes are model tokens. A selected local
    tokenizer is a different, fail-closed policy; it never silently switches
    units during a strict run.
    """

    def fail(message: str):
        if parser is not None:
            parser.error(message)
        raise BenchmarkIntegrityError(message)

    endpoint = validate_safe_endpoint(args.answer_base_url, label="reader")
    ceiling = getattr(args, "provider_context_tokens", None)
    if ceiling is None:
        if endpoint == DEEPSEEK_BASE_URL:
            ceiling = DEFAULT_PROVIDER_CONTEXT_TOKENS
        else:
            fail(
                "--provider-context-tokens is required for a non-default "
                "answer endpoint"
            )
    if (
        isinstance(ceiling, bool) or not isinstance(ceiling, int)
        or ceiling <= 0
    ):
        fail("--provider-context-tokens must be a positive integer")
    byte_budget = getattr(args, "max_input_bytes", DEFAULT_MAX_INPUT_BYTES)
    if (
        isinstance(byte_budget, bool) or not isinstance(byte_budget, int)
        or byte_budget <= 0
    ):
        fail("--max-input-bytes must be a positive integer")
    tokenizer_path_raw = getattr(args, "tokenizer_json", None)
    counter = None
    tokenizer_identity = None
    exact_budget = getattr(args, "max_input_tokens", None)
    if tokenizer_path_raw:
        tokenizer_path = Path(tokenizer_path_raw).expanduser().resolve()
        if not tokenizer_path.is_file():
            fail(f"--tokenizer-json is not a file: {tokenizer_path}")
        if exact_budget is None:
            exact_budget = DEFAULT_MAX_INPUT_TOKENS
        if (
            isinstance(exact_budget, bool) or not isinstance(exact_budget, int)
            or exact_budget <= 0
        ):
            fail("--max-input-tokens must be a positive integer")
        if (
            exact_budget + READER_OUTPUT_RESERVE_TOKENS
            + READER_TRANSPORT_OVERHEAD_TOKENS > ceiling
        ):
            fail(
                "--max-input-tokens plus reader output/framing reserves exceeds "
                "the declared provider context ceiling"
            )
        try:
            counter = _load_local_tokenizer_counter(tokenizer_path)
        except BenchmarkIntegrityError as exc:
            fail(str(exc))
        tokenizer_identity = {
            "configured": True,
            "backend": "huggingface-tokenizers-json",
            "bound_model": args.answer_model,
            "file_sha256": file_hash(tokenizer_path),
            "local_only": True,
        }
        budget_unit = "model_tokens"
        policy_name = "model-bound-tokenizer-query-head-tail-v2"
        effective_byte_budget = None
    else:
        if exact_budget is not None:
            fail("--max-input-tokens requires --tokenizer-json")
        if (
            byte_budget + READER_OUTPUT_RESERVE_TOKENS
            + READER_TRANSPORT_OVERHEAD_TOKENS > ceiling
        ):
            fail(
                "--max-input-bytes plus reader output/framing reserves exceeds "
                "the declared provider context ceiling"
            )
        budget_unit = "utf8_bytes"
        policy_name = "conservative-utf8-byte-query-head-tail-v2"
        effective_byte_budget = byte_budget

    policy = {
        "name": policy_name,
        "budget_unit": budget_unit,
        "max_input_tokens": exact_budget,
        "max_input_bytes": effective_byte_budget,
        "provider_context_window_tokens": ceiling,
        "reserved_output_tokens": READER_OUTPUT_RESERVE_TOKENS,
        "reserved_transport_overhead_tokens": READER_TRANSPORT_OVERHEAD_TOKENS,
        "tokenizer": tokenizer_identity,
        "tokenizer_failure_policy": (
            "fail-closed" if tokenizer_identity else "not-applicable"
        ),
        "source_boundaries": ["head", "query-window", "tail"],
        "raw_evidence_reserve_fraction": RAW_EVIDENCE_RESERVE_FRACTION,
        "min_semantic_excerpt_alnum": MIN_SEMANTIC_EXCERPT_ALNUM,
        "min_semantic_excerpt_chars": MIN_SEMANTIC_EXCERPT_CHARS,
        "gold_access": False,
    }
    return policy, counter

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
RESERVED_CHAT_BODY_KEYS = frozenset({
    "model", "messages", "temperature", "max_tokens", "n",
})
THINKING_DISABLED = {"thinking": {"type": "disabled"}}


def is_llm_error(text: str | None) -> bool:
    return bool(text) and str(text).startswith(LLM_ERROR_PREFIX)


def validate_request_extra_body(value: dict | None) -> dict:
    """Keep provider extensions from overriding manifested core request fields."""

    return normalize_extra_body(value, label="request")


def resolve_model_extra_body(
    model: str,
    base_url: str,
    extra_body: dict | None,
) -> tuple[dict, bool]:
    """Return a safe, effective raw request body extension.

    DeepSeek v4-flash writes its usable answer to the ordinary ``content``
    field only when thinking is disabled.  An omitted body therefore gets the
    required vendor extension automatically on the actual DeepSeek endpoint.
    A caller-provided body remains authoritative, but a DeepSeek v4-flash body
    that would leave thinking enabled is rejected instead of producing a
    capability score from empty completions.

    The endpoint check is deliberately exact-host.  A custom OpenAI-compatible
    gateway never receives a DeepSeek-only key merely because its model name
    resembles ours; operators of such gateways can pass an explicit body when
    their provider supports it.
    """

    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise BenchmarkIntegrityError("LLM model identity must be non-empty")
    normalized_base = validate_safe_endpoint(base_url, label="LLM")
    body = validate_request_extra_body(extra_body)
    was_absent = extra_body is None
    host = (urlsplit(normalized_base).hostname or "").casefold()
    is_deepseek_endpoint = host == "api.deepseek.com"
    is_v4_flash = "v4-flash" in model.casefold()

    if was_absent and is_deepseek_endpoint and is_v4_flash:
        return copy.deepcopy(THINKING_DISABLED), True
    if is_deepseek_endpoint and is_v4_flash:
        thinking = body.get("thinking")
        if not isinstance(thinking, dict) or thinking.get("type") != "disabled":
            raise BenchmarkIntegrityError(
                "DeepSeek v4-flash requires thinking.type='disabled'; omit "
                "the extra-body option to use the safe default"
            )
    return body, False


# ── LLM Client ──────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str, api_key: str, base_url: str = DEEPSEEK_BASE_URL,
                 extra_body: dict | None = None, *, n: int | None = None):
        # The benchmark's raw requests client does not pass through
        # OpenAICompatibleClient.  Enforce the shared active-model policy
        # before endpoint credential resolution or any network-capable state.
        require_active_model(model, role="LongMemEval reader/judge")
        if not isinstance(model, str) or not model.strip() or model != model.strip():
            raise BenchmarkIntegrityError("LLM model identity must be non-empty")
        self.model = model
        # The answer and judge can be pointed at independent endpoints.  The
        # endpoint and credential are jointly resolved before any call.  This
        # keeps a process-wide provider key from becoming a bearer token for a
        # custom host even when a caller bypasses the CLI resolver.
        try:
            endpoint, resolved_key = resolve_llm_api_key(
                base_url, explicit_key=api_key or None
            )
        except (EnvironmentError, ValueError) as exc:
            raise BenchmarkIntegrityError(str(exc)) from exc
        self.api_key = resolved_key
        self.base_url = endpoint.url
        # Extra top-level request-body fields merged into every call — the raw-HTTP
        # equivalent of the OpenAI SDK's `extra_body`. Needed post-2026-07-24:
        # the retired deepseek-chat alias had moved reader/judge to
        # deepseek-v4-flash, whose thinking mode can consume the response budget
        # (and corrupt the yes/no judge parse) unless thinking is disabled.
        # ``None`` means no operator body was supplied and activates the safe
        # DeepSeek-only default; an explicit object is validated as-is.
        self.extra_body, self.extra_body_defaulted = resolve_model_extra_body(
            model, self.base_url, extra_body
        )
        if n is not None and (
            isinstance(n, bool) or not isinstance(n, int) or n <= 0
        ):
            raise BenchmarkIntegrityError("LLM completion count must be positive")
        self.n = n
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.token_usage_available = False
        self._usage_complete = True
        self.last_error: str | None = None
        # Guards the two counters so they aggregate correctly when many worker
        # threads share this client (--workers > 1).
        self._lock = threading.Lock()
        self._closed = False

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        last_error = None
        last_exception_type = "Exception"
        for attempt in range(3):
            try:
                content, usage = self._call(messages, temperature, max_tokens)
                with self._lock:
                    required = ("prompt_tokens", "completion_tokens", "total_tokens")
                    valid = all(
                        isinstance(usage.get(key), (int, float))
                        and not isinstance(usage.get(key), bool)
                        and usage[key] >= 0
                        for key in required
                    )
                    if valid:
                        self.prompt_tokens += usage["prompt_tokens"]
                        self.completion_tokens += usage["completion_tokens"]
                        self.total_tokens += usage["total_tokens"]
                    else:
                        self._usage_complete = False
                    self.token_usage_available = (
                        self.successful_responses > 0 and self._usage_complete
                    )
                return content
            except Exception as e:
                last_error = str(e)
                last_exception_type = _bounded_exception_type(e)
                # A failed request may have reached the provider but did not
                # return an auditable usage block. Even a later successful
                # retry cannot make the call-chain token total complete.
                with self._lock:
                    self._usage_complete = False
                    self.token_usage_available = False
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
        self.last_error = last_exception_type
        return f"[LLM_ERROR:{last_exception_type}]"

    def _call(self, messages: list, temperature: float, max_tokens: int) -> tuple[str, dict]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.n is not None:
            body["n"] = self.n
        # Only provider-specific extensions reach this merge; core request
        # fields cannot disagree with the manifest.
        body.update(self.extra_body)
        started = time.monotonic()
        with self._lock:
            self.request_attempts += 1
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
            with self._lock:
                self.total_latency_s += time.monotonic() - started
        content = data["choices"][0]["message"].get("content")
        if content is None:
            # A 200 with content=null. Transient provider behavior should be
            # retried like a 429; a null from finish_reason=length is
            # deterministic truncation (chat() fails that one fast). Either
            # way, returning None used to crash callers (`None.startswith`) —
            # raising makes the failure explicit and countable.
            raise RuntimeError(
                f"null content (finish={data['choices'][0].get('finish_reason')})")
        if not isinstance(content, str):
            raise RuntimeError(
                "non-string content "
                f"(finish={data['choices'][0].get('finish_reason')})"
            )
        with self._lock:
            self.call_count += 1
            self.successful_responses += 1
        return (
            content,
            data.get("usage", {}),
        )

    def close(self) -> None:
        """End this wrapper's lifecycle exactly once.

        ``requests.post`` owns and closes its short-lived Session per request,
        so there is no persistent transport here.  The explicit close hook
        still gives benchmark ownership scopes one uniform, testable contract.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True


# ── Dataset Loader (streaming) ──────────────────────────────────────

def load_longmemeval_data(
    dataset_path: str,
    max_questions: int = None,
    seed: int = 0,
    *,
    strict_schema: bool = False,
    scale: str = DEFAULT_SCALE,
) -> list[dict]:
    """Stream-load LongMemEval questions with label-blind sampling.

    `seed` makes the sample deterministic so two runs
    (e.g. old code vs new code) evaluate the IDENTICAL question set — without
    it every run draws a fresh sample and per-category deltas are dominated by
    which questions happened to be drawn, not by the code change. Pass
    `max_questions=None` (CLI `--sample 0`) to evaluate the full set and remove
    sampling variance entirely.
    """
    import ijson

    # Labels are retained for official judging and post-answer diagnostics, but
    # they never determine which examples enter a scored run.
    questions: list[dict] = []
    with open(dataset_path, "rb") as f:
        for item in ijson.items(f, "item"):
            # Preserve the upstream row byte-semantics: its question_type stays
            # one of six base categories.  The evaluator selects abstention from
            # ``'_abs' in question_id`` separately, after the answer exists.
            questions.append(item)

    if strict_schema:
        questions = list(validate_lme_dataset(questions, scale=scale))

    total_available = len(questions)
    # Compute display-only diversity after strict validation. A corrupt,
    # unhashable qtype must become BenchmarkIntegrityError, never leak a raw
    # TypeError from set insertion before schema checks run.
    num_types = len({
        item.get("question_type") for item in questions
        if isinstance(item.get("question_type"), str)
    })
    n_abs = sum(1 for item in questions if is_official_abstention_id(item.get("question_id")))
    if n_abs:
        print(f"  Abstention questions present: {n_abs} (guard-rail measurable)", flush=True)
    else:
        print(f"  ⚠ No abstention (_abs) questions in this dataset — the "
              f"answerable-vs-abstention guard rail cannot fire (all-answerable set)", flush=True)

    if max_questions is None or max_questions >= total_available:
        print(f"  Loaded all {total_available} questions ({num_types} types)", flush=True)
        return questions

    sampled = select_label_blind_questions(
        questions, sample=min(max_questions, total_available), seed=seed
    )
    print(f"  Loaded {len(sampled)} questions ({num_types} types, "
          f"label-blind sample, seed={seed})", flush=True)
    return sampled


def select_label_blind_questions(
    questions: list[dict], *, sample: int, seed: int,
) -> list[dict]:
    """Deterministically select examples without reading type, answer, or gold."""

    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise BenchmarkIntegrityError("sample must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BenchmarkIntegrityError("seed must be an integer")
    if sample == 0 or sample >= len(questions):
        return list(questions)
    # Rank source positions by a stable SHA-256 draw, then restore source
    # order.  The draw never reads qid (whose `_abs` marker is a label), qtype,
    # answer, answer_session_ids or has_answer.  Unlike random.sample's output
    # ordering, this makes selected-row order a stable subsequence of the
    # pinned source order across Python versions.
    ranked = sorted(
        range(len(questions)),
        key=lambda index: hashlib.sha256(
            f"longmemeval-label-blind-v1\0{seed}\0{index}".encode("ascii")
        ).digest(),
    )[:sample]
    chosen = set(ranked)
    return [row for index, row in enumerate(questions) if index in chosen]


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
                 facts_extraction: bool | None = None,
                 pipeline_model: str = PINNED_DEEPSEEK_MODEL,
                 pipeline_base_url: str = DEEPSEEK_BASE_URL,
                 pipeline_thinking: str = "auto",
                 embedding_base_url: str | None = None,
                 embedding_model: str | None = None,
                 embedding_dim: int | None = None,
                 embedding_api_key: str | None = None,
                 embedding_deployment_revision: str | None = None,
                 embedding_deployment_tenant: str | None = None):
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
        self.pipeline_model = pipeline_model
        self.pipeline_base_url = validate_safe_endpoint(
            pipeline_base_url, label="memory pipeline"
        )
        if pipeline_thinking not in {"auto", "disabled", "off", "enabled"}:
            raise BenchmarkIntegrityError("invalid memory-pipeline thinking mode")
        self.pipeline_thinking = pipeline_thinking
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_api_key = embedding_api_key
        self.embedding_deployment_revision = embedding_deployment_revision
        self.embedding_deployment_tenant = embedding_deployment_tenant
        self.hy = None
        self.pipeline_llm = None
        self.embedding_client = None
        self.last_indexing_summary = None
        self._owned_resources = OwnedResourceScope(
            "LongMemEval memory-adapter resources"
        )

    def build_config(self):
        """Exact effective HyMem config, usable before provider construction."""
        from hymem import HyMemConfig

        overrides = {}
        if self.rerank_top_k is not None:
            overrides["rerank_top_k"] = self.rerank_top_k
        if self.rerank_model is not None:
            overrides["rerank_model"] = self.rerank_model
        if self.rerank_message_hits is not None:
            overrides["rerank_message_hits"] = self.rerank_message_hits
        overrides["aggregation_nodes_enabled"] = self.aggregation_nodes
        if self.aggregation_broad:
            overrides["aggregation_inject_abilities"] = ()
        overrides["episode_granularity_enabled"] = self.episode_granularity
        overrides["value_supersession_enabled"] = self.value_supersession
        if self.graph_multihop:
            overrides["graph_multihop_enabled"] = True
            if self.graph_multihop_max_hops is not None:
                overrides["graph_multihop_max_hops"] = self.graph_multihop_max_hops
            if self.graph_multihop_decay is not None:
                overrides["graph_multihop_decay"] = self.graph_multihop_decay
            if self.graph_multihop_min_score is not None:
                overrides["graph_multihop_min_score"] = self.graph_multihop_min_score
        if self.rules_enabled is not None:
            overrides["rules_enabled"] = self.rules_enabled
        if self.rules_extraction is not None:
            overrides["rules_extraction_enabled"] = self.rules_extraction
        if self.facts_enabled is not None:
            overrides["facts_enabled"] = self.facts_enabled
        if self.facts_extraction is not None:
            overrides["facts_extraction_enabled"] = self.facts_extraction
        return HyMemConfig(
            root=self.db_path.parent,
            message_fts_top_k=15,
            fts_top_k=10,
            graph_top_k=10,
            **overrides,
        )

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
        cfg = self.build_config()
        try:
            llm = OpenAICompatibleClient(
                api_key=self.api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
                base_url=self.pipeline_base_url,
                model=self.pipeline_model,
                thinking=self.pipeline_thinking,
            )
            self.pipeline_llm = self._owned_resources.own(
                llm, label="memory pipeline client"
            )
            # Optional semantic-recall A/B (lever L1). Drives the SAME local FastEmbed
            # server Hermes uses in production. Pass this benchmark's local FastEmbed
            # defaults explicitly so --embeddings works with ZERO env setup. DeepSeek
            # has no embeddings API; HYMEM_EMBEDDING_* can instead select a real
            # embedding endpoint. Off by default: the headline baseline is lexical-only
            # (a paired comparison).
            embedding_client = None
            if self.embeddings:
                from hymem.contrib.openai_embedding_client import (
                    OpenAICompatibleEmbeddingClient,
                    is_loopback_embedding_url,
                    is_official_openai_embedding_url,
                )

                env = os.environ.get
                embedding_base_url = (
                    self.embedding_base_url
                    or env("HYMEM_EMBEDDING_BASE_URL") or LOCAL_EMBED_BASE_URL
                )
                embedding_api_key = self.embedding_api_key or env("HYMEM_EMBEDDING_API_KEY")
                if not embedding_api_key and is_loopback_embedding_url(embedding_base_url):
                    embedding_api_key = LOCAL_EMBED_API_KEY
                if (
                    not embedding_api_key
                    and is_official_openai_embedding_url(embedding_base_url)
                ):
                    embedding_api_key = env("OPENAI_API_KEY")
                embedding_client = OpenAICompatibleEmbeddingClient(
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                    model=(self.embedding_model or env("HYMEM_EMBEDDING_MODEL")
                           or LOCAL_EMBED_MODEL),
                    dim=(self.embedding_dim if self.embedding_dim is not None else
                         int(env("HYMEM_EMBEDDING_DIM") or LOCAL_EMBED_DIM)),
                    pin_dimension=True,
                    deployment_revision=(
                        self.embedding_deployment_revision
                        or env("HYMEM_EMBEDDING_DEPLOYMENT_REVISION")
                    ),
                    deployment_tenant=(
                        self.embedding_deployment_tenant
                        or env("HYMEM_EMBEDDING_DEPLOYMENT_TENANT")
                    ),
                )
                self._owned_resources.own(
                    embedding_client, label="embedding client"
                )
            self.embedding_client = embedding_client
            self.hy = HyMem(cfg, llm=llm, embedding_client=embedding_client)
            self._owned_resources.own(self.hy, label="memory store")
            return self
        except BaseException as exc:
            self._owned_resources.close(primary_exception=exc)
            raise

    def close(self):
        try:
            self._owned_resources.close()
        finally:
            self.hy = None

    def ingest_sessions(self, sessions: list[list[dict]], session_ids: list[str],
                         session_dates: list[str] | None = None, *,
                         namespace: str = "question") -> dict:
        """Ingest all sessions for a question. Each session is a list of messages.
        
        If session_dates is provided (one ISO-8601 date per session), each message
        gets that session's date as its created_at, giving HyMem real event times
        instead of wall-clock clustering."""
        total_msgs = 0
        total_chars = 0
        empty_msgs = 0
        if not isinstance(session_dates, list):
            raise BenchmarkIntegrityError(
                "LongMemEval ingestion requires explicit session dates"
            )
        dates = session_dates
        if len(session_ids) != len(sessions):
            raise BenchmarkIntegrityError("session IDs and sessions differ in length")
        if len(dates) != len(sessions):
            raise BenchmarkIntegrityError("session dates and sessions differ in length")
        namespace_hash = hashlib.sha256(
            str(namespace).encode("utf-8")
        ).hexdigest()[:12]
        for idx, (sess_id, messages) in enumerate(
            zip(session_ids, sessions, strict=True)
        ):
            if (
                not isinstance(sess_id, str) or not sess_id.strip()
                or sess_id != sess_id.strip()
            ):
                raise BenchmarkIntegrityError("LongMemEval session id is malformed")
            if not isinstance(messages, list) or not messages:
                raise BenchmarkIntegrityError("LongMemEval session messages are malformed")
            session_date = _normalize_date(dates[idx])
            if session_date is None:
                raise BenchmarkIntegrityError("LongMemEval session date is malformed")
            entries = []
            for m in messages:
                if not isinstance(m, dict) or m.get("role") not in {
                    "user", "assistant",
                } or not isinstance(m.get("content"), str):
                    raise BenchmarkIntegrityError("LongMemEval message is malformed")
                role = m["role"]
                content = m["content"]
                if content.strip():
                    entries.append((role, content, session_date))
                    total_msgs += 1
                    total_chars += len(content)
                else:
                    empty_msgs += 1
            if entries:
                chunk_size = 50
                for i in range(0, len(entries), chunk_size):
                    chunk = entries[i : i + chunk_size]
                    # Source session IDs are not unique in the pinned dataset.
                    # Bind the internal key to the ordered occurrence so no
                    # duplicate silently overwrites/merges another session.
                    self.hy.log_messages(
                        f"lme_{namespace_hash}_{idx}_{i//chunk_size}", chunk
                    )
        return {
            "sessions": len(sessions), "messages": total_msgs,
            "chars": total_chars, "empty_messages_skipped": empty_msgs,
        }

    def dream_and_wait(
        self,
        timeout: float = DEFAULT_INDEXING_TIMEOUT_S,
        *,
        max_cycles: int = DEFAULT_INDEXING_MAX_CYCLES,
        require_healthy: bool = True,
    ):
        """Run bounded cycles until the durable extraction backlog is healthy."""
        start = time.time()
        dream_hy = self.hy.fork()
        try:
            try:
                raw_summary = converge_indexing(
                    dream_hy.dream,
                    status=lambda: durable_indexing_status(
                        dream_hy, getattr(self, "embedding_client", None),
                    ),
                    max_cycles=max_cycles,
                    timeout_s=timeout,
                    require_healthy=require_healthy,
                )
                self.last_indexing_summary = canonicalize_lme_indexing_summary(
                    raw_summary
                )
                if self.last_indexing_summary["outcome"] == "failure":
                    raise IndexingConvergenceError(
                        "memory indexing completed without usable health",
                        self.last_indexing_summary,
                    )
            except Exception as exc:
                if isinstance(exc, IndexingConvergenceError):
                    if exc.summary.get("schema") == LME_INDEXING_SUMMARY_VERSION:
                        self.last_indexing_summary = dict(exc.summary)
                    else:
                        self.last_indexing_summary = canonicalize_lme_indexing_summary(
                            exc.summary
                        )
                    # Direct callers and tests inspect the raised summary too;
                    # never leave the raw exception/source-bearing variant on
                    # one surface while persisting the bounded one on another.
                    exc.summary = self.last_indexing_summary
                raise
        finally:
            cleanup_sink = None
            if self.last_indexing_summary is not None:
                cleanup_sink = self.last_indexing_summary["cleanup_errors"]
            run_cleanup_actions(
                [
                    ("dream_fork_close", dream_hy.close),
                    ("query_cache_invalidation", self.hy.invalidate_query_caches),
                ],
                primary_exception=sys.exc_info()[1],
                evidence_sink=cleanup_sink,
            )
        elapsed = time.time() - start
        print(
            f"      Dream converged in {elapsed:.0f}s across "
            f"{self.last_indexing_summary['cycles']} cycle(s)", flush=True,
        )
        return self.last_indexing_summary

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
        result = self.hy.augment(query, ability=ability)
        if self.embedding_client is not None:
            from hymem.dreaming.aggregation_material import (
                embedding_execution_identity,
            )

            _binding, producer_key, dimension = embedding_execution_identity(
                self.embedding_client
            )
            semantic = getattr(result, "semantic_status", None)
            if (
                semantic is None
                or getattr(semantic, "configured", None) is not True
                or getattr(semantic, "attempted", None) is not True
                or getattr(semantic, "available", None) is not True
                or getattr(semantic, "model", None) != producer_key
                or getattr(semantic, "dim", None) != dimension
            ):
                reason = getattr(semantic, "reason", "missing_status")
                raise BenchmarkIntegrityError(
                    "configured embedding retrieval was unavailable or changed "
                    f"identity (reason={reason})"
                )

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
            # Keep the complete DTO payload.  Presentation may excerpt under the
            # final token budget, but retrieval must not permanently discard an
            # answer-bearing tail before the packer sees it.
            text = getattr(hit, "text", "")
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
            text = getattr(hit, "text", "")
            if text.strip():
                fts_hits.append({
                    "content": text,
                    "type": "fts_hit",
                    "confidence": 0.6,
                })

        procedure_hits = []
        for proc in (getattr(result, "procedures", None) or []):
            name = getattr(proc, "name", "")
            desc = getattr(proc, "description", "")
            content = f"Procedure: {name}: {desc}" if name else desc
            if content.strip():
                procedure_hits.append({
                    "content": content,
                    "type": "procedure",
                    "confidence": 0.75,
                })

        # RAPTOR aggregation nodes (empty unless the layer is enabled AND the
        # ability passed the inject gate — TR-only by default). Kept OUT of the
        # `memories` pool: they go to answer_question as a separate block.
        aggregation_nodes = []
        for node in (getattr(result, "aggregation_nodes", None) or []):
            title = getattr(node, "title", "")
            summary = getattr(node, "summary", "")
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
            text = getattr(nf, "text", "")
            if text.strip():
                date = getattr(nf, "fact_date", None) or ""
                narrative_facts.append(f"[{date}] {text}" if date else text)

        episode_hits = []
        for ep in (getattr(result, "episodes", None) or []):
            title = getattr(ep, "title", "")
            summary = getattr(ep, "summary", "")
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


def query_centered_boundary_excerpt(text: str, *, query: str, limit: int) -> str:
    """Deterministic bounded excerpt retaining head, query window, and tail.

    LongMemEval turns can be much longer than a legacy 500/600-character slice.
    A leading-only slice is especially destructive because answers are often a
    final correction or concrete value.  This presentation-only helper leaves
    retrieval payloads intact and, when a single item must shrink, preserves
    both source boundaries plus the most discriminative query-centered window.
    """
    from hymem.query.presentation import query_centered_excerpt

    normalized = " ".join(str(text or "").split())
    if limit <= 0:
        return ""
    if len(normalized) <= limit:
        return normalized
    if limit <= 12:
        return normalized[:limit]
    separator = " … "
    available = limit - len(separator) * 2
    head_n = max(1, available // 5)
    tail_n = max(1, available // 4)
    middle_n = max(1, available - head_n - tail_n)
    middle = query_centered_excerpt(normalized, query=query, limit=middle_n)
    parts = [normalized[:head_n], middle, normalized[-tail_n:]]
    out: list[str] = []
    for part in parts:
        if part and part not in out:
            out.append(part)
    rendered = separator.join(out)
    return rendered[:limit]


def _render_answer_context(memories: list[dict], ability: str | None,
                           total_matches: int, graph_count,
                           temporal_events: list | None,
                           aggregation_nodes: list | None,
                           distilled: list[str] | None = None,
                           narrative_facts: list[str] | None = None,
                           *, max_context_chars: int | None = None,
                           max_input_tokens: int | None = None,
                           max_input_bytes: int | None = None,
                           token_counter=None,
                           fail_on_tokenizer_error: bool = False,
                           prompt_prefix: str = "",
                           prompt_suffix: str = "",
                           query: str = "",
                           _stable_counter: bool = False) -> str:
    """Build a deterministic, source-bounded reader context.

    Auxiliary synthesis is selected separately from raw retrieval evidence. It
    may lead the rendered prompt, but it cannot consume the raw-evidence reserve.
    A configured model tokenizer governs the token path. Strict LME callers fail
    closed if it errors; compatibility callers may explicitly provide a byte
    budget for a deterministic whole-pack retry under that separate unit.
    """
    from hymem.query.fusion import (
        _ConfiguredTokenizerFailure,
        estimate_tokens,
        stable_token_counter,
    )

    if token_counter is not None and not _stable_counter:
        try:
            return _render_answer_context(
                memories, ability, total_matches, graph_count,
                temporal_events, aggregation_nodes, distilled=distilled,
                narrative_facts=narrative_facts,
                max_context_chars=max_context_chars,
                max_input_tokens=max_input_tokens,
                max_input_bytes=max_input_bytes,
                token_counter=stable_token_counter(token_counter),
                fail_on_tokenizer_error=fail_on_tokenizer_error,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                query=query,
                _stable_counter=True,
            )
        except _ConfiguredTokenizerFailure as exc:
            if fail_on_tokenizer_error:
                raise BenchmarkIntegrityError(
                    f"configured reader tokenizer failed: {exc}"
                ) from exc
            return _render_answer_context(
                memories, ability, total_matches, graph_count,
                temporal_events, aggregation_nodes, distilled=distilled,
                narrative_facts=narrative_facts,
                max_context_chars=max_context_chars,
                max_input_tokens=None,
                max_input_bytes=max_input_bytes,
                token_counter=None,
                fail_on_tokenizer_error=False,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                query=query,
                _stable_counter=True,
            )

    if max_input_tokens is not None and token_counter is None:
        if fail_on_tokenizer_error:
            raise BenchmarkIntegrityError(
                "a model-token input budget requires its configured token counter"
            )
        # Legacy/direct callers (not strict LME manifests) historically passed
        # this knob without a tokenizer and received the conservative UTF-8
        # byte path. Preserve that API/BEAM behavior while strict LME calls
        # select and name their byte policy before entering the renderer.
        max_input_bytes = (
            max_input_bytes if max_input_bytes is not None
            else max_input_tokens
        )
        max_input_tokens = None

    # Character limits are compatibility/test-only. Strict runs use either the
    # exact configured model counter or the explicitly byte-denominated policy.
    context_limit = max_context_chars

    # MR counting: prefer graph_count (EXACT graph-native count) over
    # total_matches (keyword candidate). When graph_count is present it is
    # the dedup-correct answer — trust it.
    auxiliary: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    def candidate(kind: str, body: str, *, prefix: str = "") -> dict[str, Any]:
        return {
            "kind": kind,
            "body": str(body),
            "prefix": prefix,
            "ordinal": len(auxiliary) + len(raw),
        }

    if ability == "MR" and graph_count is not None:
        count = graph_count.count
        counted = getattr(graph_count, 'counted', 'items')
        auxiliary.append(candidate("graph-count",
            f"[HyMem graph-native count: {count} distinct {counted} "
            f"(exact COUNT(DISTINCT) over knowledge graph edges). "
            f"Use this as the answer. Verify against evidence below.]"
        ))
    elif ability == "MR" and total_matches > 0:
        auxiliary.append(candidate("retrieval-count",
            f"[HyMem counted {total_matches} distinct user messages "
            f"matching your question (assistant echoes excluded, "
            f"restatements deduped). Verify this count against the "
            f"evidence below and return the final number.]"
        ))

    # If retrieval produced a chronology, it is evidence regardless of which
    # label-free prompt won. Routing may prioritize it; routing must not erase it.
    if temporal_events:
        block = ["[TEMPORAL CHRONOLOGY — events in date order. A 'discussed' "
                 "line is the date the turn was logged (when-discussed), not "
                 "necessarily when the event happened:]"]
        for ev in temporal_events:
            date = getattr(ev, 'date', '')
            desc = getattr(ev, 'text', str(ev))
            marker = " (discussed)" if getattr(ev, 'source', '') == "session-date" else ""
            block.append(f"  {date}{marker}: {desc}")
        block.append("[END CHRONOLOGY]")
        auxiliary.append(candidate("temporal", "\n".join(block)))

    # RAPTOR aggregation nodes render as their own block, NOT as memories: they
    # never consume a memories[:top_k] slot and cannot consume the reserved raw-
    # evidence share of the context budget — the crowding that cost KU −9.0pp
    # in the broad-injection A/B.
    # Empty unless the layer is enabled and the ability passed the inject gate
    # (TR-only by default).
    if aggregation_nodes:
        block = ["[CROSS-SESSION SUMMARIES — each fuses related episodes "
                 "from multiple sessions; verify details against the "
                 "memories below:]"]
        for node in aggregation_nodes:
            block.append(f"  {node}")
        block.append("[END SUMMARIES]")
        auxiliary.append(candidate("aggregation", "\n".join(block)))

    # E1 narrative facts: dream-extracted self-contained statements, rendered as
    # their own block for the same reason as the summaries above — never a
    # memories[:top_k] slot, never budget ahead of the raw turns. The
    # verify-against-source framing is deliberate and matches the tier's own
    # contract in ask(): facts LEAD the evidence but the raw turns stay below as
    # the check (the Acme lesson — a summary is never the only copy).
    if narrative_facts:
        block = ["[NARRATIVE FACTS — self-contained statements extracted "
                 "from past sessions; a leading date is when the fact "
                 "happened. Verify details against the memories below:]"]
        for nf in narrative_facts:
            block.append(f"  • {nf}")
        block.append("[END NARRATIVE FACTS]")
        auxiliary.append(candidate("narrative", "\n".join(block)))

    # Distilled evidence (P1): question-conditioned facts extracted per-turn,
    # rendered ABOVE the raw memories as their own non-competing block. Additive
    # — the raw turns stay below, unchanged. Verify-against-source framing so the
    # answerer treats it as a lens, not a replacement.
    if distilled:
        block = ["[DISTILLED EVIDENCE — extracted per-turn, verify against "
                 "the memories below:]"]
        for line in distilled:
            block.append(f"  • {line}")
        block.append("[END DISTILLED EVIDENCE]")
        auxiliary.append(candidate("distilled", "\n".join(block)))

    for m in memories:
        content = str(m["content"])
        # Date-stamp raw turns (the only tier carrying created_at) so the recency-
        # conflict clause can prefer the newest value-bearing statement. FACT/fts/
        # episode tiers stay undated (graph dating deferred — see KU analysis).
        if m["type"] == "graph_fact":
            tag = "[FACT]"
        else:
            date10 = (m.get("created_at") or "")[:10]
            tag = f"[MEM {date10}]" if (m["type"] == "message_hit" and date10) else "[MEM]"
        raw.append(candidate(m.get("type", "memory"), content, prefix=f"{tag} "))

    def render(item: dict[str, Any]) -> str:
        return f"{item['prefix']}{item['body']}"

    def evidence_payload(body: str, *, kind: str) -> str:
        # Retrieved contents sometimes already carry their own source tag.
        # Strip complete tags before deciding whether anything semantic remains,
        # and reject a cut-off tag such as ``[ME`` outright.
        stripped = body.strip()
        if kind in {"temporal", "aggregation", "narrative", "distilled"}:
            closing = stripped.find("]")
            if not stripped.startswith("[") or closing < 0:
                return ""
            stripped = stripped[closing + 1:].lstrip()
            stripped = re.sub(r"\s*\[END [^\]]+\]\s*$", "", stripped)
        elif kind not in {"graph-count", "retrieval-count", "truncation", "empty"}:
            match = re.match(
                r"^\[(?:MEM(?:\s+[^\]]+)?|FACT|user|assistant)\]\s*",
                stripped,
                flags=re.IGNORECASE,
            )
            if match:
                stripped = stripped[match.end():]
            elif stripped.startswith("[") and "]" not in stripped:
                return ""
        return stripped

    def excerpt_meaningful(item: dict[str, Any]) -> bool:
        payload = evidence_payload(item["body"], kind=item["kind"])
        return (
            len(payload) >= MIN_SEMANTIC_EXCERPT_CHARS
            and sum(character.isalnum() for character in payload)
            >= MIN_SEMANTIC_EXCERPT_ALNUM
        )

    def joined(aux_items: list[dict[str, Any]],
               raw_items: list[dict[str, Any]]) -> str:
        # Selection priority and render priority are intentionally different:
        # raw evidence gets reserved first, while concise aids lead the prompt.
        return "\n".join(render(item) for item in [*aux_items, *raw_items])

    def measure_visible(context: str) -> int:
        visible_input = prompt_prefix + context + prompt_suffix
        if max_input_tokens is not None:
            return estimate_tokens(visible_input, token_counter)
        if max_input_bytes is not None:
            return len(visible_input.encode("utf-8"))
        return len(visible_input)

    def fits(aux_items: list[dict[str, Any]], raw_items: list[dict[str, Any]],
             *, primary_limit: int | None = None) -> bool:
        context = joined(aux_items, raw_items)
        visible_input = prompt_prefix + context + prompt_suffix
        if context_limit is not None and len(visible_input) > context_limit:
            return False
        hard_limit = (
            max_input_tokens if max_input_tokens is not None
            else max_input_bytes
        )
        if primary_limit is not None:
            hard_limit = primary_limit
        return hard_limit is None or measure_visible(context) <= hard_limit

    if not fits([], []):
        raise BenchmarkIntegrityError(
            "reader system/question wrappers alone exceed hard input budget"
        )

    # When everything fits, preserve every source byte-for-byte exactly once.
    if fits(auxiliary, raw):
        return joined(auxiliary, raw)

    hard_limit = (
        max_input_tokens if max_input_tokens is not None else max_input_bytes
    )
    if hard_limit is None:
        hard_limit = context_limit
    wrapper_units = measure_visible("")
    raw_reserve_limit = (
        wrapper_units + int(max(0, hard_limit - wrapper_units)
                            * RAW_EVIDENCE_RESERVE_FRACTION)
        if hard_limit is not None else None
    )

    selected_aux: list[dict[str, Any]] = []
    selected_raw: list[dict[str, Any]] = []
    selected_ordinals: set[int] = set()
    dropped = 0

    def pack_group(group: list[dict[str, Any]], *, placement: str,
                   primary_limit: int | None = None) -> None:
        nonlocal dropped
        available_group = [
            item for item in group if item["ordinal"] not in selected_ordinals
        ]
        for index, item in enumerate(available_group):
            target = selected_aux if placement == "aux" else selected_raw

            def trial_with(extra: list[dict[str, Any]]) -> tuple[
                list[dict[str, Any]], list[dict[str, Any]]
            ]:
                if placement == "aux":
                    return [*selected_aux, *extra], selected_raw
                return selected_aux, [*selected_raw, *extra]

            # Keep room for a later compact source in the same tier. This is
            # the guard against a leading '['/verbose distractor starving a
            # later answer-bearing source.
            reserve = None
            for later in available_group[index + 1:]:
                aux_trial, raw_trial = trial_with([later])
                if fits(
                    aux_trial, raw_trial, primary_limit=primary_limit
                ):
                    if reserve is None or len(render(later)) < len(render(reserve)):
                        reserve = later

            extras = [item, *([reserve] if reserve is not None else [])]
            aux_trial, raw_trial = trial_with(extras)
            if fits(
                aux_trial, raw_trial, primary_limit=primary_limit
            ):
                target.append(item)
                selected_ordinals.add(item["ordinal"])
                continue

            # Excerpt only the evidence body; a syntactic source header can
            # never make a punctuation/one-character stub look meaningful.
            best = None
            low, high = 1, len(item["body"])
            while low <= high:
                middle = (low + high) // 2
                excerpt_body = query_centered_boundary_excerpt(
                    item["body"], query=query, limit=middle
                )
                excerpt = {**item, "body": excerpt_body}
                if not excerpt_meaningful(excerpt):
                    low = middle + 1
                    continue
                excerpt_extras = [excerpt, *(
                    [reserve] if reserve is not None else []
                )]
                aux_trial, raw_trial = trial_with(excerpt_extras)
                if fits(aux_trial, raw_trial, primary_limit=primary_limit):
                    best = excerpt
                    low = middle + 1
                else:
                    high = middle - 1
            if best is not None:
                target.append(best)
                selected_ordinals.add(item["ordinal"])
            dropped += 1

    # Reserve raw evidence first, then select aids, then spend any remainder on
    # additional raw sources. Render order remains aids-before-raw.
    pack_group(raw, placement="raw", primary_limit=raw_reserve_limit)
    pack_group(auxiliary, placement="aux")
    pack_group(raw, placement="raw")

    if not selected_aux and not selected_raw:
        empty = "No relevant memories found."
        empty_item = candidate("empty", empty)
        if fits([empty_item], []):
            return empty
        if fits([], []):
            return ""
        raise BenchmarkIntegrityError(
            "reader system/question wrappers alone exceed hard input budget"
        )
    if dropped:
        marker = candidate("truncation", "[... context truncated]")
        # A disclosure marker is metadata, not evidence. Never evict the last
        # compact/answer-bearing source merely to make the marker fit.
        if fits([*selected_aux, marker], selected_raw):
            selected_aux.append(marker)
        else:
            # Disclosure may replace an auxiliary excerpt, never raw evidence.
            # This also prevents a clipped auxiliary header from being the only
            # signal that the prompt was truncated.
            with_marker = list(selected_aux)
            while with_marker and not fits([*with_marker, marker], selected_raw):
                with_marker.pop()
            if fits([*with_marker, marker], selected_raw):
                selected_aux = [*with_marker, marker]
    rendered = joined(selected_aux, selected_raw)
    if not fits(selected_aux, selected_raw):
        raise BenchmarkIntegrityError("rendered answer context exceeds hard budget")
    return rendered


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
    request_attempts = 0
    successful_responses = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    total_latency_s = 0.0
    cost_usd = 0.0
    token_usage_available = True

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
                          narrative_facts: list[str] | None = None,
                          max_input_tokens: int | None = None,
                          max_input_bytes: int | None = None,
                          token_counter=None,
                          fail_on_tokenizer_error: bool = False) -> list[dict]:
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
        distilled, extra_system, narrative_facts, max_input_tokens,
        max_input_bytes, token_counter, fail_on_tokenizer_error)


def answer_question_raw(llm: LLMClient, memories: list[dict], question: str,
                        *args, **kwargs) -> tuple[str, str]:
    """`answer_question`, plus the context fingerprint it would otherwise drop.

    Prompt construction lives in `build_answer_messages` and happens in ONE
    place, so a `--retrieval-only` run fingerprints exactly the prompt a full
    run would have sent."""
    if "token_counter" not in kwargs:
        counter = getattr(llm, "count_tokens", None)
        kwargs["token_counter"] = counter if callable(counter) else None
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
                     narrative_facts: list[str] | None = None,
                     max_input_tokens: int | None = None,
                     max_input_bytes: int | None = None,
                     token_counter=None,
                     fail_on_tokenizer_error: bool = False) -> list[dict]:
    """Render the reader's prompt. The only place it is built.

    Uses ability-aware prompts and expanded context for multi-session
    and temporal reasoning questions that need more cross-session data.
    For MR questions, prefers graph_count (exact graph-native count) over
    total_matches (keyword candidate). For TR questions, injects
    temporal_events as a date-ordered chronology. `aggregation_nodes` (RAPTOR
    cross-session summaries, TR-gated upstream) render as a separate block so
    they never compete with raw turns for top_k slots and cannot consume the
    explicit raw-evidence budget reserve.

    `question_date` is the "now" the question is asked at — the reference point
    relative-date questions ("how many days ago?", "a month ago") subtract from.
    Without it the chronology gives event dates but the model has no anchor to
    compute an interval against, and answers "current date not provided".
    """
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
    user_prefix = f"{today_line}CONTEXT:\n"
    user_suffix = f"\n\nQUESTION: {question}\n\nANSWER:"

    # The selected policy measures every part of the visible system/user input,
    # including question wrappers and all context headers, in either exact model
    # tokens or explicitly named UTF-8 bytes. Context candidates retain source
    # boundaries; oversized items are excerpted deterministically across
    # head/query/tail while reserving later evidence.
    context = _render_answer_context(
        memories, ability, total_matches, graph_count,
        temporal_events, aggregation_nodes, distilled=distilled,
        narrative_facts=narrative_facts,
        max_input_tokens=max_input_tokens,
        max_input_bytes=max_input_bytes,
        token_counter=token_counter,
        fail_on_tokenizer_error=fail_on_tokenizer_error,
        prompt_prefix=f"system:{system_prompt}\nuser:{user_prefix}",
        prompt_suffix=user_suffix,
        query=question,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prefix}{context}{user_suffix}"},
    ]

    return messages


def get_judge_prompt(
    question_type: str, question: str, answer: str, response: str, *,
    question_id: str | None = None,
) -> str:
    """Historical local prompt; deliberately not claimed byte-exact upstream."""
    is_abstention = (
        is_official_abstention_id(question_id)
        if question_id is not None else "_abs" in question_type
    )
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


def get_official_judge_prompt(
    question_type: str,
    question_id: str,
    question: str,
    answer: str,
    response: str,
) -> str:
    """Pinned upstream ``get_anscheck_prompt`` byte semantics.

    The two ordinary templates retain upstream's space immediately before the
    first newline.  Abstention is selected from the prediction question id,
    exactly as evaluate_qa.py does, never from a mutated question type.
    """
    if is_official_abstention_id(question_id):
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\n"
            "Model Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
    elif question_type in {"single-session-user", "single-session-assistant", "multi-session"}:
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no. \n\nQuestion: {}\n\n"
            "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no. In addition, do not penalize off-by-one "
            "errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
            "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the "
            "model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the required "
            "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized response, and a response from a "
            "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long "
            "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
            "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    else:
        raise NotImplementedError(f"Unknown question type: {question_type}")
    return template.format(question, answer, response)


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
                     answer: str, ai_answer: str, *,
                     question_id: str | None = None,
                     protocol: str = "legacy-custom") -> tuple[bool, str]:
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
    if protocol == "official":
        if not isinstance(question_id, str) or not question_id:
            raise BenchmarkIntegrityError("official LongMemEval judge requires question_id")
        prompt = get_official_judge_prompt(
            question_type, question_id, question, answer, ai_answer
        )
    elif protocol == "legacy-custom":
        prompt = get_judge_prompt(
            question_type, question, answer, ai_answer,
            question_id=question_id,
        )
    else:
        raise BenchmarkIntegrityError(f"unknown LongMemEval judge protocol {protocol!r}")
    messages = [{"role": "user", "content": prompt}]
    raw = llm.chat(
        messages, temperature=LME_OFFICIAL_JUDGE_TEMPERATURE,
        max_tokens=LME_OFFICIAL_JUDGE_MAX_TOKENS,
    )
    verdict = (
        parse_official_verdict(raw) if protocol == "official"
        else parse_judge_verdict(raw)
    )
    return verdict, raw


def judge_answer(llm: LLMClient, question_type: str, question: str, answer: str, ai_answer: str) -> bool:
    """Judge whether the answer is correct (binary yes/no per LongMemEval protocol)."""
    return judge_answer_raw(llm, question_type, question, answer, ai_answer)[0]


def judge_scored(llm: LLMClient, question_type: str, question: str,
                 answer: str, ai_answer: str, *,
                 question_id: str | None = None,
                 protocol: str = "legacy-custom") -> tuple[bool | None, str]:
    """Return a verdict only when the judge transport and reply are parseable.

    ``None`` is retained in the raw row for auditability, but strict
    reconciliation converts it to a wrong answer in the full expected
    denominator.  The separately named judged-only diagnostic may condition on
    valid replies; the headline score never does.
    """
    verdict, raw = judge_answer_raw(
        llm, question_type, question, answer, ai_answer,
        question_id=question_id, protocol=protocol,
    )
    if protocol == "official":
        # Upstream accepts any successful string and applies its literal
        # substring rule. Transport/sentinel failures remain benchmark failures.
        parseable = bool(
            isinstance(raw, str) and raw.strip() and not is_llm_error(raw)
        )
        return (verdict if parseable else None), raw
    low = (raw or "").casefold()
    has_yes = bool(_YES_WORD.search(low))
    has_no = bool(_NO_WORD.search(low))
    parseable = bool(raw and not is_llm_error(raw) and has_yes != has_no)
    return (verdict if parseable else None), raw


# ── Scoring over rows that may be UNSCORED ──────────────────────────

def is_scored(row: dict) -> bool:
    """Whether a row belongs in the explicitly conditional judged-only view."""
    return isinstance(row.get("correct"), bool) and not row.get("benchmark_failure")


def is_retrieval_only(artifact: dict) -> bool:
    """See `run_registry.is_retrieval_only` -- one definition, re-exported so
    the adapter and the scorers cannot drift on what counts as a run with no
    verdicts."""
    try:
        from .run_registry import is_retrieval_only as _impl
    except (ImportError, ValueError):
        from run_registry import is_retrieval_only as _impl
    return _impl(artifact)


def scored(rows: list[dict]) -> list[dict]:
    """Conditional judged-only rows; never use this for a headline score."""
    return [r for r in rows if is_scored(r)]


def accuracy(rows: list[dict]) -> float:
    """Strict accuracy: invalid/missing verdicts are wrong, never omitted."""
    normalized = []
    for index, row in enumerate(rows):
        verdict = row.get("correct")
        if verdict is not None and not isinstance(verdict, bool):
            raise BenchmarkIntegrityError(
                f"malformed verdict at row {index}: {verdict!r}"
            )
        normalized.append({"correct": bool(verdict)})
    return strict_accuracy(normalized)


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
        return (f"⚠ {len(errs)} judge transport/parse failure(s); each counts "
                f"WRONG in the strict {len(rows)}-row denominator. Conditional "
                f"judged-only n={len(judged)}. "
                f"ids: {[r.get('id') or r.get('question_id') for r in errs][:10]}")
    return f"judge errors: 0 of {len(rows)} row(s) that could have errored"

# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_question(
    llm: LLMClient,
    judge_llm: LLMClient,
    hy: HyMemAdapter,
    q_data: dict,
    top_k: int,
    auto_ability: bool = True,
    no_dream: bool = False,
    graph_facts_first: bool = False,
    permissive_default: bool = False,
    distill: bool = False,
    distill_prompt_version: str = DEFAULT_DISTILL_PROMPT_VERSION,
    retrieval_only: bool = False,
    distill_llm: LLMClient | None = None,
    max_input_tokens: int | None = None,
    max_input_bytes: int | None = None,
    token_counter=None,
    judge_protocol: str = "legacy-custom",
    indexing_max_cycles: int = DEFAULT_INDEXING_MAX_CYCLES,
    indexing_timeout_s: float = DEFAULT_INDEXING_TIMEOUT_S,
    indexing_require_healthy: bool = True,
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
    sessions = q_data.get("haystack_sessions")
    session_ids = q_data.get("haystack_session_ids")
    session_dates = q_data.get("haystack_dates")
    question_date = q_data.get("question_date")
    if not all(isinstance(value, list) for value in (
        sessions, session_ids, session_dates,
    )):
        raise BenchmarkIntegrityError(
            "LongMemEval question lacks explicit session/id/date arrays"
        )
    if len(sessions) != len(session_ids) or len(sessions) != len(session_dates):
        raise BenchmarkIntegrityError(
            "LongMemEval session/id/date lengths differ"
        )
    # The question timestamp is score-affecting. Never substitute the latest
    # haystack date (or wall clock) for a malformed official source value.
    question_date = normalize_lme_date(
        question_date, label=f"{question_id} question_date"
    )

    # The default route is structurally label-free: it does not even resolve
    # the source qtype to an oracle ability before the reader has answered.
    # Explicit oracle mode is exploratory and is the sole pre-answer exception.
    oracle_ability = (
        QUESTION_TYPE_TO_ABILITY.get(question_type) if not auto_ability else None
    )
    detected_ability = (
        _detect_ability(question) if auto_ability
        else _detect_ability_safe(question)
    )
    ability = detected_ability if auto_ability else oracle_ability

    # Ingest
    # The official qid may contain `_abs`, which is a gold label.  Keep it out
    # of every retrieval-affecting identifier; the DB itself is question-local,
    # and ordered session/chunk positions provide collision-free keys.
    stats = hy.ingest_sessions(
        sessions, session_ids, session_dates,
        # Bind the isolated question's internal namespace to reader-visible text,
        # never the qid: upstream qids carry the `_abs` gold label.
        namespace=question,
    )
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
        hy.dream_and_wait(
            timeout=indexing_timeout_s,
            max_cycles=indexing_max_cycles,
            require_healthy=indexing_require_healthy,
        )

    # Search
    (memories, total_matches, graph_count, temporal_events, aggregation_nodes,
     narrative_facts, pool) = hy.search(
        question, ability=ability, top_k=top_k * 3, graph_facts_first=graph_facts_first)
    src = "question_date" if q_data.get("question_date") else ("haystack_max" if session_dates else "none")

    used_marker = "←used" if auto_ability else ""
    router_str = f"oracle={'hidden' if auto_ability else (oracle_ability or '∅')}/det={detected_ability or '∅'}{used_marker}"
    print(f"    Retrieved {len(memories)} memories (total_matches={total_matches}, graph_count={graph_count is not None}, temporal_events={len(temporal_events)}, agg_nodes={len(aggregation_nodes)}, facts={len(narrative_facts)}, now={question_date or '∅'}[{src}], ability={router_str})", flush=True)

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
        narrative_facts=narrative_facts,
        max_input_tokens=max_input_tokens,
        max_input_bytes=max_input_bytes,
        token_counter=token_counter,
        fail_on_tokenizer_error=token_counter is not None)
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

    # Labels/gold are first read only after the retrieval and reader route is
    # complete.  These diagnostics are nonblocking: corrupt optional marks can
    # never discard an answer or alter its context.
    oracle_ability = QUESTION_TYPE_TO_ABILITY.get(question_type)
    answer = str(q_data["answer"])
    recall_diagnostic_error = None
    try:
        gold_turns, gold_mode = _extract_gold_turns(q_data)
        if gold_turns:
            in_msg = _gold_in_pool(gold_turns, pool["message"])
            in_fts = _gold_in_pool(gold_turns, pool["fts"])
            recall_ceiling = in_msg or in_fts
            recall_tier = ("both" if in_msg and in_fts
                           else "message" if in_msg
                           else "fts" if in_fts else "none")
            gold_turn_tiers = _gold_turn_tiers(gold_turns, pool)
            gold_turns_in_pool = sum(1 for tier in gold_turn_tiers if tier != "none")
        else:
            recall_ceiling, recall_tier = None, "unknown"
            gold_turn_tiers, gold_turns_in_pool = [], None
    except Exception as exc:
        gold_turns, gold_mode = [], "diagnostic-error"
        recall_ceiling, recall_tier = None, "unknown"
        gold_turn_tiers, gold_turns_in_pool = [], None
        recall_diagnostic_error = (
            f"recall_gold_turns:{_bounded_exception_type(exc)}"
        )
    try:
        gold_in_episodes = (
            _answer_in_texts(answer, _ep_texts) if _ep_texts else False
        )
        gold_in_facts = (
            _answer_in_texts(answer, narrative_facts)
            if narrative_facts else False
        )
    except Exception as exc:
        gold_in_episodes = gold_in_facts = None
        detail = f"answer_containment:{_bounded_exception_type(exc)}"
        recall_diagnostic_error = (
            f"{recall_diagnostic_error}; {detail}"
            if recall_diagnostic_error else detail
        )

    # Judge (binary yes/no, or None when the JUDGE itself errored — D3)
    benchmark_failure = None
    if retrieval_only:
        correct, judge_raw = None, ""
        print(f"    retrieval-only: ctx={ctx_sha[:12]} "
              f"episodes={len(_ep_texts)} distill_calls={distill_calls}",
              flush=True)
    elif is_llm_error(ai_answer) or not str(ai_answer).strip():
        # A reader transport failure is not a candidate answer.  Never pay a
        # judge to turn an outage sentinel into benchmark data.
        correct, judge_raw = False, ""
        benchmark_failure = "reader_transport_or_empty_response"
        print(f"    Reader failure before judge: {ai_answer[:120]}", flush=True)
    else:
        correct, judge_raw = judge_scored(
            judge_llm, question_type, question, answer, ai_answer,
            question_id=question_id, protocol=judge_protocol,
        )
        if correct is None:
            benchmark_failure = "judge_transport_or_parse_failure"
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
        "judge_protocol": judge_protocol,
        # The judge's own reply, kept for the same reason the strings above are
        # un-clipped: a discarded reply is why judge_audit had to re-judge 500
        # rows to measure a rate that had already been produced once. ~10 tokens.
        "judge_raw": judge_raw,
        # A judge that was never CALLED did not error. Under --retrieval-only
        # every row has correct=None by design, and flagging 50 "judge errors"
        # on a run that made zero judge calls is the same vacuity as "0 judge
        # errors" over a run that made none: a count whose denominator is not
        # what the reader assumes.
        "judge_error": bool(benchmark_failure and benchmark_failure.startswith("judge_")),
        "judge_parse_valid": (
            None if retrieval_only or benchmark_failure == "reader_transport_or_empty_response"
            else correct is not None
        ),
        "benchmark_failure": benchmark_failure,
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
        "empty_messages_skipped": stats["empty_messages_skipped"],
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
        "gold_in_episodes": gold_in_episodes,
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
        "recall_diagnostic_error": recall_diagnostic_error,
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
        "gold_in_facts": gold_in_facts,
        "indexing": (None if no_dream else hy.last_indexing_summary),
    }


def compute_scores(results: list[dict]) -> dict:
    """Compute strict accuracy by type over the complete supplied denominator."""
    by_type = defaultdict(list)
    for index, r in enumerate(results):
        verdict = r.get("correct")
        if verdict is not None and not isinstance(verdict, bool):
            raise BenchmarkIntegrityError(
                f"malformed verdict at row {index}: {verdict!r}"
            )
        qtype = r["question_type"]
        # Source question_type remains one of the six base categories. The
        # upstream abstention branch is carried only by question_id.
        by_type[qtype].append(bool(verdict))

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
    answer IS abstention). Upstream keeps question_type as a base category and
    routes abstention from ``'_abs' in question_id``. Here we keep that split:
      - ANSWERABLE: id without `_abs`
      - ABSTENTION: id containing `_abs`
    reported overall AND per base category, so a permissive run can be A/B'd
    against strict with the abstention cost made explicit. If overall goes up while
    ABSTENTION drops, the gain is partly a hallucination trade, not a clean win.
    """
    def _infrastructure_failure(row: dict) -> bool:
        return bool(row.get("judge_error") or row.get("benchmark_failure"))

    answerable: list[tuple[bool, bool]] = []
    abstention: list[tuple[bool, bool]] = []
    by_cat: dict[str, dict[str, list[tuple[bool, bool]]]] = defaultdict(
        lambda: {"answerable": [], "abstention": []})
    for r in results:
        qtype = r.get("question_type", "")
        is_abs = is_official_abstention_id(r.get("question_id"))
        base = qtype
        bucket = "abstention" if is_abs else "answerable"
        value = (bool(r.get("correct")), _infrastructure_failure(r))
        (abstention if is_abs else answerable).append(value)
        by_cat[base][bucket].append(value)

    def _acc(xs: list[tuple[bool, bool]]) -> dict:
        valid = [correct for correct, failed in xs if not failed]
        return {
            "accuracy": (
                sum(correct for correct, _failed in xs) / len(xs)
                if xs else None
            ),
            "count": len(xs),
            "benchmark_failures": sum(1 for _correct, failed in xs if failed),
            "conditional_valid_accuracy": (
                sum(valid) / len(valid) if valid else None
            ),
            "conditional_valid_count": len(valid),
        }

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
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"].replace("_abs", "")].append(r)

    tiers = Counter()
    modes = Counter()
    diag: dict[str, dict] = {}
    for qtype, rows in sorted(by_type.items()):
        infrastructure = [
            r for r in rows
            if r.get("judge_error") or r.get("benchmark_failure")
        ]
        causal_rows = [
            r for r in rows
            if not (r.get("judge_error") or r.get("benchmark_failure"))
        ]
        known = [r for r in causal_rows if r.get("recall_ceiling") is not None]
        hit = [r for r in known if r["recall_ceiling"]]
        misses = [r for r in causal_rows if not r["correct"]]
        miss_retrieval = sum(1 for r in misses if r.get("recall_ceiling") is False)
        miss_ranking = sum(1 for r in misses if r.get("recall_ceiling") is True)
        miss_unknown = sum(1 for r in misses if r.get("recall_ceiling") is None)
        for r in known:
            tiers[r.get("recall_tier", "none")] += 1
        for r in causal_rows:
            modes[r.get("gold_mode", "none")] += 1
        diag[qtype] = {
            "known": len(known),
            "unknown": len(causal_rows) - len(known),
            "ceiling_rate": (len(hit) / len(known)) if known else None,
            "misses": len(misses),
            "miss_retrieval": miss_retrieval,
            "miss_ranking": miss_ranking,
            "miss_unknown": miss_unknown,
            "benchmark_failures_excluded": len(infrastructure),
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

    print("\n  No external leaderboard is printed: vendor-reported results use "
          "different models, judges, prompts, versions and sample sets. See "
          "README.md for protocol-specific source links and limitations.")


# ── Per-question worker ─────────────────────────────────────────────

def _adapter_for_args(db_path: Path, args, api_key: str) -> HyMemAdapter:
    """Single construction path for runtime and pre-run config identity."""

    return HyMemAdapter(
        db_path, api_key=api_key, embeddings=args.embeddings,
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
        rules_enabled=args.rules, rules_extraction=args.rules_extraction,
        facts_enabled=args.facts, facts_extraction=args.facts_extraction,
        pipeline_model=getattr(args, "hymem_model", PINNED_DEEPSEEK_MODEL),
        pipeline_base_url=getattr(args, "hymem_base_url", DEEPSEEK_BASE_URL),
        pipeline_thinking=getattr(args, "hymem_thinking", "auto"),
        embedding_base_url=getattr(args, "embedding_base_url", None),
        embedding_model=getattr(args, "embedding_model", None),
        embedding_dim=getattr(args, "embedding_dim", None),
        embedding_api_key=getattr(args, "embedding_api_key", None),
        embedding_deployment_revision=getattr(
            args, "embedding_deployment_revision", None,
        ),
        embedding_deployment_tenant=getattr(
            args, "embedding_deployment_tenant", None,
        ),
    )


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ("git", "-C", str(_repo_root), *args), capture_output=True,
            text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def resolve_prereg(path: str | None) -> dict | None:
    """Bind a canonical claim to a committed spec before any benchmark spend."""
    if path is None:
        return None
    if _git("rev-parse", "--git-dir") is None:
        raise BenchmarkIntegrityError("pre-registration requires a git repository")
    source = Path(path) if os.path.isabs(path) else _repo_root / path
    if not source.is_file():
        raise BenchmarkIntegrityError(f"pre-registration does not exist: {path!r}")
    try:
        relative = source.resolve().relative_to(_repo_root).as_posix()
    except ValueError as exc:
        raise BenchmarkIntegrityError("pre-registration must be inside this repository") from exc
    if _git("status", "--porcelain", "--", relative):
        raise BenchmarkIntegrityError("pre-registration is uncommitted or modified")
    commit = _git("log", "-1", "--format=%H", "--", relative)
    blob = _git("rev-parse", f"HEAD:{relative}")
    committed_at = _git("log", "-1", "--format=%cI", "--", relative)
    head = _git("rev-parse", "HEAD")
    if not all((commit, blob, committed_at, head)):
        raise BenchmarkIntegrityError("pre-registration has no complete git provenance")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise BenchmarkIntegrityError(
            "tracked code is dirty; canonical pre-registration cannot name it"
        )
    return {
        "path": relative, "commit": commit, "blob": blob,
        "committed_at": committed_at, "code_commit": head,
    }


def _provider_for_url(base_url: str) -> str:
    normalized = validate_safe_endpoint(base_url, label="provider")
    official = validate_http_endpoint(
        normalized, label="provider"
    ).official_provider
    if official is not None:
        return official
    return "openai-compatible"


def resolve_endpoint_key(
    *, role: str, base_url: str, explicit_key: str | None,
    deepseek_key: str,
) -> str:
    """Resolve credentials without leaking one provider's key to another.

    Only the exact DeepSeek origin may inherit the shared historical key.  A
    custom reader/pipeline endpoint always needs its role-specific option.
    The exact official OpenAI judge may use OPENAI_API_KEY, because that is a
    provider-specific judge credential rather than cross-provider fallback.
    """

    normalized = validate_safe_endpoint(base_url, label=role)
    # ``deepseek_key`` may include the legacy CLI/config-file source, but it is
    # promoted to explicit only after exact-origin validation.  The shared
    # resolver independently permits HYMEM_LLM_API_KEY for custom endpoints and
    # binds OPENAI_API_KEY/DEEPSEEK_API_KEY to their official origins.
    authorized_explicit = explicit_key
    endpoint = validate_http_endpoint(normalized, label=role)
    if not authorized_explicit and endpoint.official_provider == "deepseek":
        authorized_explicit = deepseek_key
    try:
        _endpoint, key = resolve_llm_api_key(
            normalized, explicit_key=authorized_explicit
        )
        return key
    except (EnvironmentError, ValueError):
        pass
    option = {
        "reader": "--answer-api-key",
        "judge": "--judge-api-key",
        "memory pipeline": "--hymem-api-key",
    }.get(role, "an explicit role-specific API key")
    raise BenchmarkIntegrityError(
        f"{role} endpoint {safe_endpoint_label(normalized, label=role)!r} "
        f"requires {option}"
    )


def resolve_embedding_identity(args) -> dict[str, Any]:
    from hymem.dreaming.aggregation_material import (
        configured_openai_embedding_producer_binding,
        embedding_producer_binding,
        public_embedding_identity,
    )
    if not args.embeddings:
        return public_embedding_identity(
            embedding_producer_binding(None), None,
            fallback_policy="none", fallback_reason=None,
            transport_security=TRANSPORT_SECURITY_NONE,
        )
    base_url = (
        args.embedding_base_url
        or os.environ.get("HYMEM_EMBEDDING_BASE_URL") or LOCAL_EMBED_BASE_URL
    )
    model = (
        args.embedding_model
        or os.environ.get("HYMEM_EMBEDDING_MODEL") or LOCAL_EMBED_MODEL
    )
    raw_dim = (
        args.embedding_dim if args.embedding_dim is not None
        else os.environ.get("HYMEM_EMBEDDING_DIM", LOCAL_EMBED_DIM)
    )
    try:
        dimension = int(raw_dim)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkIntegrityError("embedding dimension must be a positive integer") from exc
    if isinstance(raw_dim, bool) or dimension <= 0 or str(raw_dim).strip() != str(dimension):
        raise BenchmarkIntegrityError("embedding dimension must be a positive integer")
    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise BenchmarkIntegrityError("embedding model identity is malformed")
    try:
        endpoint = validate_http_endpoint(
            base_url,
            label="embedding",
            allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            "embedding endpoint is unsafe or ambiguous"
        ) from exc
    revision = getattr(args, "embedding_deployment_revision", None) or os.environ.get(
        "HYMEM_EMBEDDING_DEPLOYMENT_REVISION"
    )
    tenant = getattr(args, "embedding_deployment_tenant", None) or os.environ.get(
        "HYMEM_EMBEDDING_DEPLOYMENT_TENANT"
    )
    try:
        binding = configured_openai_embedding_producer_binding(
            base_url=endpoint.url,
            request_model=model,
            dimension=dimension,
            pin_dimension=True,
            deployment_revision=revision,
            deployment_tenant=tenant,
        )
        return public_embedding_identity(
            binding, dimension,
            fallback_policy="fail-closed", fallback_reason=None,
            transport_security=endpoint_transport_security(endpoint),
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            "strict embeddings require explicit deployment revision, tenant, "
            "and a pinned dimension"
        ) from exc


def resolve_embedding_key(args) -> str | None:
    """Resolve only an embedding-specific credential for pending work."""

    if not args.embeddings:
        return None
    base_url = (
        args.embedding_base_url
        or os.environ.get("HYMEM_EMBEDDING_BASE_URL") or LOCAL_EMBED_BASE_URL
    )
    try:
        _endpoint, key = resolve_embedding_api_key(
            base_url, explicit_key=args.embedding_api_key or None
        )
        return key
    except (EnvironmentError, ValueError) as exc:
        raise BenchmarkIntegrityError(str(exc)) from exc


def validate_runtime_arguments(args, parser: argparse.ArgumentParser) -> None:
    def positive_number(name: str, value: object) -> None:
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value <= 0
        ):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")

    def integer(name: str, value: object, *, allow_zero: bool = False) -> None:
        if (
            isinstance(value, bool) or not isinstance(value, int)
            or value < 0 or (not allow_zero and value <= 0)
        ):
            parser.error(
                f"--{name.replace('_', '-')} must be "
                + ("a non-negative integer" if allow_zero else "a positive integer")
            )

    if args.scales.upper() not in LME_SUPPORTED_SCALES:
        parser.error("--scales must be S or M")
    integer("sample", args.sample, allow_zero=True)
    integer("top_k", args.top_k)
    integer("workers", args.workers)
    if args.max_input_tokens is not None:
        integer("max_input_tokens", args.max_input_tokens)
    integer("max_input_bytes", args.max_input_bytes)
    if args.provider_context_tokens is not None:
        integer("provider_context_tokens", args.provider_context_tokens)
    integer("indexing_max_cycles", args.indexing_max_cycles)
    positive_number("indexing_timeout_s", args.indexing_timeout_s)
    if args.rerank_top_k is not None:
        integer("rerank_top_k", args.rerank_top_k)
    if args.graph_multihop_max_hops is not None:
        integer("graph_multihop_max_hops", args.graph_multihop_max_hops)
    if args.graph_multihop_decay is not None and not (
        math.isfinite(args.graph_multihop_decay) and 0 < args.graph_multihop_decay <= 1
    ):
        parser.error("--graph-multihop-decay must be finite in (0, 1]")
    if args.graph_multihop_min_score is not None and not (
        math.isfinite(args.graph_multihop_min_score)
        and 0 <= args.graph_multihop_min_score <= 1
    ):
        parser.error("--graph-multihop-min-score must be finite in [0, 1]")
    if not args.graph_multihop and any(value is not None for value in (
        args.graph_multihop_max_hops,
        args.graph_multihop_decay,
        args.graph_multihop_min_score,
    )):
        parser.error("graph multihop parameters require --graph-multihop")
    if args.aggregation_broad and not args.aggregation_nodes:
        parser.error("--aggregation-broad requires --aggregation-nodes")
    for field in ("answer_model", "judge_model", "hymem_model"):
        value = getattr(args, field)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            parser.error(f"--{field.replace('_', '-')} must be a non-empty model id")
    # Keep provider-free calibration/export and historical artifact transforms
    # readable, while rejecting every role this invocation can actually call.
    # This runs before credentials, dataset IO, checkpoints, or temp stores.
    if args.rejudge:
        active_models = (("LongMemEval judge", args.judge_model),)
    elif args.freeze_calibration:
        active_models = ()
    elif args.inspect_floor:
        active_models = (("LongMemEval memory pipeline", args.hymem_model),)
    elif args.distill_dryrun:
        active_models = (
            ("LongMemEval reader", args.answer_model),
            ("LongMemEval judge", args.judge_model),
            ("LongMemEval memory pipeline", args.hymem_model),
        )
    else:
        active_models = [("LongMemEval memory pipeline", args.hymem_model)]
        if not args.retrieval_only or args.distill:
            active_models.append(("LongMemEval reader", args.answer_model))
        if not args.retrieval_only:
            active_models.append(("LongMemEval judge", args.judge_model))
    try:
        for role, active_model in active_models:
            require_active_model(active_model, role=role)
    except DeprecatedModelAliasError as exc:
        parser.error(str(exc))
    for label in ("answer_base_url", "judge_base_url", "hymem_base_url"):
        try:
            validate_safe_endpoint(getattr(args, label), label=label)
        except BenchmarkIntegrityError as exc:
            parser.error(str(exc))


class _ParallelQuestionStopped(BaseException):
    """Internal cooperative cancellation; never materialize this as a row."""


def _evaluate_one_question(qi, total, q_data, args, answer_llm, judge_llm,
                           api_key, distill_llm=None, *,
                           _parallel_stop: threading.Event | None = None,
                           _on_fatal_abort=None, _on_runtime=None):
    """Full lifecycle for one question: fresh temp DB → open → evaluate → cleanup.

    Self-contained so it can run in a worker thread. Each question gets its own
    SQLite file + HyMem instance (created and used entirely within this call, so
    no connection crosses threads); the only shared state is the two LLMClients,
    whose counters are lock-guarded. Ordinary per-row exceptions are captured as
    incorrect results so one bad question cannot abort a parallel run. Process-
    control ``BaseException`` values still propagate after best-effort cleanup.
    """
    def _raise_if_parallel_stopped() -> None:
        if _parallel_stop is not None and _parallel_stop.is_set():
            raise _ParallelQuestionStopped()

    def _notify_fatal_abort(exc: BaseException) -> None:
        if _on_fatal_abort is not None:
            _on_fatal_abort(exc)

    # A submitted task may have been dequeued just as another worker or the
    # coordinator discovered a fatal persistence fault.  Do not construct a
    # store (or reach a provider-capable path) after that signal.
    _raise_if_parallel_stopped()
    print(f"[{qi+1}/{total}] Q: {q_data['question_id']} ({q_data['question_type']})", flush=True)

    # Fresh temp DB per question (sessions are question-specific)
    tmp_dir: Path | None = None
    hy = None
    result = {
        "question_id": q_data.get("question_id", "unknown"),
        "question_type": q_data.get("question_type", "unknown"),
        "correct": False,
        "benchmark_failure": "execution_did_not_produce_a_row",
        "distill_fired": False,
        "distill_calls": 0,
    }
    lifecycle_errors: list[dict[str, str]] = []
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="hymem-lme-"))
        db_path = tmp_dir / "hymem.sqlite"
        hy = _adapter_for_args(db_path, args, api_key)
        hy.open()
        # Opening is a safe lifecycle boundary.  A task already running while
        # another task failed can still stop before indexing/reader/judge work.
        _raise_if_parallel_stopped()
        result = evaluate_question(
            answer_llm, judge_llm, hy, q_data, args.top_k,
            auto_ability=args.auto_ability, no_dream=args.no_dream,
            graph_facts_first=args.graph_facts_first,
            permissive_default=args.permissive_default,
            distill=args.distill,
            distill_prompt_version=args.distill_prompt_version,
            retrieval_only=getattr(args, "retrieval_only", False),
            distill_llm=distill_llm,
            max_input_tokens=getattr(args, "max_input_tokens", None),
            max_input_bytes=getattr(args, "max_input_bytes", None),
            token_counter=getattr(args, "token_counter", None),
            judge_protocol=getattr(args, "judge_protocol", "legacy-custom"),
            indexing_max_cycles=getattr(
                args, "indexing_max_cycles", DEFAULT_INDEXING_MAX_CYCLES
            ),
            indexing_timeout_s=getattr(
                args, "indexing_timeout_s", DEFAULT_INDEXING_TIMEOUT_S
            ),
            indexing_require_healthy=getattr(
                args, "indexing_require_healthy", True
            ),
        )
    except BenchmarkCleanupError as exc:
        _notify_fatal_abort(exc)
        raise
    except IndexingConvergenceError as e:
        summary = getattr(hy, "last_indexing_summary", None)
        failure = summary.get("failure") if isinstance(summary, dict) else None
        failure_code = failure.get("code") if isinstance(failure, dict) else None
        if not isinstance(failure_code, str):
            failure_code = "malformed_status_shape"
        print(f"    INDEXING FAILED: {failure_code}", flush=True)
        result = {
            "question_id": q_data.get("question_id", "unknown"),
            "question_type": q_data.get("question_type", "unknown"),
            "correct": False,
            "benchmark_failure": f"indexing_failure:{failure_code}",
            "indexing": summary,
            "oracle_ability": QUESTION_TYPE_TO_ABILITY.get(
                q_data.get("question_type")
            ),
            "detected_ability": None,
            "ability_used": (
                None if getattr(args, "auto_ability", True) else
                QUESTION_TYPE_TO_ABILITY.get(q_data.get("question_type"))
            ),
            "retrieval_only": bool(getattr(args, "retrieval_only", False)),
            "distill_fired": False,
            "distill_calls": 0,
        }
    except Exception as e:
        if is_structural_benchmark_error(e):
            _notify_fatal_abort(e)
            raise
        print(f"    ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exception_type = _bounded_exception_type(e)
        result = {
            "question_id": q_data.get("question_id", "unknown"),
            "question_type": q_data.get("question_type", "unknown"),
            "correct": False,
            # The traceback is operator-visible above. Durable benchmark
            # evidence keeps only a bounded class identity, never arbitrary
            # exception/source/provider text.
            "benchmark_failure": f"execution_failure:{exception_type}",
            "oracle_ability": QUESTION_TYPE_TO_ABILITY.get(
                q_data.get("question_type")
            ),
            "detected_ability": None,
            "ability_used": (
                None if getattr(args, "auto_ability", True) else
                QUESTION_TYPE_TO_ABILITY.get(q_data.get("question_type"))
            ),
            "retrieval_only": bool(getattr(args, "retrieval_only", False)),
            "distill_fired": False,
            "distill_calls": 0,
        }
    except _ParallelQuestionStopped:
        raise
    except BaseException as exc:
        # Process-control failures must stop sibling tasks before this worker's
        # cleanup finishes; the original object remains the primary exception.
        _notify_fatal_abort(exc)
        raise
    finally:
        cleanup_actions = []
        if hy is not None:
            def snapshot_pipeline_usage():
                result.setdefault(
                    "memory_pipeline_usage",
                    usage_snapshot(getattr(hy, "pipeline_llm", None)),
                )

            def snapshot_embedding_usage():
                result.setdefault(
                    "embedding_usage",
                    embedding_usage_snapshot(
                        getattr(hy, "embedding_client", None),
                        configured=bool(getattr(args, "embeddings", False)),
                    ),
                )

            def snapshot_indexing_summary():
                if getattr(hy, "last_indexing_summary", None) is not None:
                    result.setdefault("indexing", hy.last_indexing_summary)

            cleanup_actions.extend([
                ("pipeline_usage_snapshot", snapshot_pipeline_usage),
                ("embedding_usage_snapshot", snapshot_embedding_usage),
                ("indexing_summary_snapshot", snapshot_indexing_summary),
                ("adapter_close", hy.close),
            ])
        if tmp_dir is not None and not args.keep_db:
            def remove_temporary_store():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=False)

            cleanup_actions.append(
                ("temporary_store_cleanup", remove_temporary_store)
            )
        cleanup_actions.append(("gc_collect", gc.collect))
        if _on_runtime is not None:
            # The coordinator cannot recover a local per-question pipeline or
            # embedding meter from a Future that is discarded during a fatal
            # checkpoint abort.  Hand the final snapshots off from the worker's
            # cleanup path, including cooperative-stop and structural failures.
            cleanup_actions.append(
                ("runtime_usage_handoff", lambda: _on_runtime(result))
            )
        run_cleanup_actions(
            cleanup_actions,
            primary_exception=sys.exc_info()[1],
            evidence_sink=lifecycle_errors,
        )
        if lifecycle_errors:
            result.setdefault("lifecycle_errors", []).extend(lifecycle_errors)
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

    if floor_ids:
        extraction_prompt_version = _pipeline_extraction_prompt_version(args)
        extraction_canary_report = (
            skipped_extraction_canary(
                "no_dream", prompt_version=extraction_prompt_version
            )
            if args.no_dream else
            run_configured_extraction_canary(
                api_key=api_key,
                base_url=args.hymem_base_url,
                model=args.hymem_model,
                thinking=args.hymem_thinking,
                prompt_version=extraction_prompt_version,
            )
        )
        _validate_pipeline_extraction_canary(
            extraction_canary_report, args,
            mode="no_dream" if args.no_dream else "required",
        )
        print_extraction_canary(extraction_canary_report)

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
                              rerank_message_hits=args.rerank_message_hits,
                              pipeline_model=args.hymem_model,
                              pipeline_base_url=args.hymem_base_url,
                              pipeline_thinking=args.hymem_thinking)
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
            cleanup_actions = []
            if hy:
                cleanup_actions.append(("adapter_close", hy.close))
            import shutil
            cleanup_actions.extend([
                (
                    "temporary_store_cleanup",
                    lambda: shutil.rmtree(tmp_dir, ignore_errors=False),
                ),
                ("gc_collect", gc.collect),
            ])
            run_cleanup_actions(
                cleanup_actions, primary_exception=sys.exc_info()[1],
            )

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
                          facts_extraction=getattr(args, "facts_extraction", None),
                          pipeline_model=args.hymem_model,
                          pipeline_base_url=args.hymem_base_url,
                          pipeline_thinking=args.hymem_thinking)
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
                                            answer, ai_answer,
                                            question_id=q_data.get("question_id"),
                                            protocol=getattr(
                                                args, "judge_protocol",
                                                "legacy-custom",
                                            ))
        out["correct"] = _correct
        out["judge_raw"] = _judge_raw
        out["judge_error"] = _correct is None
    except Exception as e:
        out["error"] = f"execution_failure:{_bounded_exception_type(e)}"
    finally:
        cleanup_actions = []
        if hy:
            cleanup_actions.append(("adapter_close", hy.close))
        import shutil
        cleanup_actions.extend([
            (
                "temporary_store_cleanup",
                lambda: shutil.rmtree(tmp_dir, ignore_errors=False),
            ),
            ("gc_collect", gc.collect),
        ])
        run_cleanup_actions(
            cleanup_actions, primary_exception=sys.exc_info()[1],
        )
    return out


def _distill_dryrun_questions_impl(
    questions: list[dict], args, pipeline_key: str,
    answer_key: str, judge_key: str, owned_clients: OwnedResourceScope,
) -> None:
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
        print(
            f"  answer reader: {args.answer_model} @ "
            f"{safe_endpoint_label(args.answer_base_url, label='reader')}  "
            f"(judge frozen: {args.judge_model} @ "
            f"{safe_endpoint_label(DEEPSEEK_BASE_URL, label='judge')})"
        )
    print(f"  distill prompt: {args.distill_prompt_version.upper()}   "
          f"max calls/q: {DISTILL_MAX_CALLS}\n{'='*72}", flush=True)

    answer_llm = owned_clients.own(
        LLMClient(args.answer_model, answer_key,
                  base_url=args.answer_base_url,
                  extra_body=getattr(args, "answer_extra_body_obj", None)),
        label="distillation reader client",
    )
    judge_llm = owned_clients.own(
        LLMClient(args.judge_model, judge_key,
                  base_url=args.judge_base_url,
                  extra_body=getattr(args, "judge_extra_body_obj", None),
                  n=1 if args.judge_protocol == "official" else None),
        label="distillation judge client",
    )

    tasks = [("cand", qid) for qid in candidates] + [("ctrl", qid) for qid in control]

    if tasks:
        extraction_prompt_version = _pipeline_extraction_prompt_version(args)
        extraction_canary_report = (
            skipped_extraction_canary(
                "no_dream", prompt_version=extraction_prompt_version
            )
            if args.no_dream else
            run_configured_extraction_canary(
                api_key=pipeline_key,
                base_url=args.hymem_base_url,
                model=args.hymem_model,
                thinking=args.hymem_thinking,
                prompt_version=extraction_prompt_version,
            )
        )
        _validate_pipeline_extraction_canary(
            extraction_canary_report, args,
            mode="no_dream" if args.no_dream else "required",
        )
        print_extraction_canary(extraction_canary_report)

    def _run(kind: str, qid: str) -> dict:
        res = _distill_run_one(by_id[qid], args, answer_llm, judge_llm,
                               pipeline_key,
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


def _distill_dryrun_questions(
    questions: list[dict], args, pipeline_key: str,
    answer_key: str, judge_key: str,
) -> None:
    """Run the dry-run while owning its shared clients through final output."""

    with OwnedResourceScope("LongMemEval distillation clients") as owned_clients:
        return _distill_dryrun_questions_impl(
            questions, args, pipeline_key, answer_key, judge_key, owned_clients
        )


# ── Re-judge (re-pair a banked baseline under a new judge) ───────────
# Built for the 2026-07-24 deepseek-chat hard-deprecation: the canonical 70.0
# baseline was answered AND judged by deepseek-chat, so it can't be reproduced.
# A parity run judged by the replacement (deepseek-v4-flash) is only comparable
# to a baseline judged by the SAME judge — but re-answering the baseline is
# wasteful. This re-runs ONLY the judge over the stored hypotheses, no ingest /
# no answer, and reports the per-category judge drift.

def _rejudge_run_impl(
    args, api_key: str, owned_clients: OwnedResourceScope,
) -> None:
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

    judge_llm = owned_clients.own(
        LLMClient(args.judge_model, api_key,
                  base_url=args.judge_base_url,
                  extra_body=getattr(args, "judge_extra_body_obj", None),
                  n=1 if args.judge_protocol == "official" else None),
        label="rejudge client",
    )

    # Flag the pre-untruncation artifact (q[:200]/a[:200]/hyp[:500]) so the
    # approximation caveat is raised only when it actually applies.
    clipped = sum(1 for r in pq
                  if len(str(r.get("hypothesis", ""))) == 500
                  or len(str(r.get("question", ""))) == 200
                  or len(str(r.get("answer", ""))) == 200)

    print(f"\n{'='*72}\nRE-JUDGE — {src.name}")
    print(f"  rows: {len(pq)}   original judge: {orig_judge}   new judge: {args.judge_model}"
          + (f"  +extra_body={args.judge_extra_body_obj}" if args.judge_extra_body_obj else ""))
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
                str(r.get("answer", "")), hyp,
                question_id=r.get("question_id", ""),
                protocol=args.judge_protocol,
            )
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
    owned_clients.close()

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
                "judge_extra_body": copy.deepcopy(getattr(
                    args, "judge_extra_body_obj", None
                )),
                "extra_body_defaulted": [
                    role for role in getattr(args, "extra_body_defaulted", [])
                    if role == "judge"
                ],
                "answer_calls": 0,
                "judge_calls": judge_llm.call_count,
                "total_tokens": judge_llm.total_tokens,
                "elapsed_s": elapsed})
    out["config"] = cfg
    archive_now = datetime.now(timezone.utc)
    stamp = archive_now.strftime("%Y%m%dT%H%M%SZ")
    nonce = archive_now.strftime("%f")
    dest = src.with_name(
        f"{src.stem}-rejudged-{args.judge_model.replace('/', '_')}-"
        f"{stamp}-{nonce}.json"
    )
    write_immutable_artifact(dest, out)
    print(f"\n  Archived re-judged results → {dest.name}")
    print(f"{'='*72}\n  Use the re-judged OVERALL as the paired baseline for a parity run "
          f"judged by {args.judge_model}.\n")


def _rejudge_run(args, api_key: str) -> None:
    """Rejudge with deterministic teardown after usage is archived."""

    with OwnedResourceScope("LongMemEval rejudge clients") as owned_clients:
        return _rejudge_run_impl(args, api_key, owned_clients)


# ── Main ────────────────────────────────────────────────────────────

def _run_main(
    _owned_ledgers: list[AtomicCheckpoint] | None,
    owned_clients: OwnedResourceScope,
):
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
                        help="RNG seed for label-blind sampling/internal splitting")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-input-tokens", type=int,
                        default=None,
                        help="exact model-token ceiling over the complete visible "
                             "reader input; requires --tokenizer-json (default "
                             f"{DEFAULT_MAX_INPUT_TOKENS} when configured)")
    parser.add_argument(
        "--max-input-bytes", type=int, default=DEFAULT_MAX_INPUT_BYTES,
        help="conservative UTF-8 byte ceiling used when no tokenizer is "
             "selected; strict configured-tokenizer failures fail closed; default "
             f"{DEFAULT_MAX_INPUT_BYTES}",
    )
    parser.add_argument(
        "--tokenizer-json", default=None, metavar="LOCAL_TOKENIZER.json",
        help="offline Hugging Face tokenizer.json bound to --answer-model; "
             "never downloaded",
    )
    parser.add_argument(
        "--provider-context-tokens", type=int, default=None,
        help="declared provider context ceiling. Required for non-default "
             "answer endpoints; the conservative byte budget and output "
             "plus chat-framing reserves must fit beneath it",
    )
    parser.add_argument("--answer-model", default=ANSWER_MODEL)
    parser.add_argument("--answer-base-url", default=DEEPSEEK_BASE_URL,
                        help="P0 parity lever: OpenAI-compatible endpoint for the "
                             "ANSWER client only (default DeepSeek). Point at a "
                             "gpt-oss-120b-class reader to measure how much of the "
                             "reader/architecture split. The judge endpoint/model "
                             "is configured and recorded independently; use "
                             "--judge-protocol official for the pinned evaluator.")
    parser.add_argument("--answer-api-key", default=None,
                        help="API key for the --answer-base-url endpoint. Defaults "
                             "to the resolved DeepSeek key ONLY for the exact "
                             "DeepSeek endpoint. Other endpoints require this flag.")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEEPSEEK_BASE_URL)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument(
        "--judge-protocol", choices=("legacy-custom", "official"),
        default="legacy-custom",
        help="legacy-custom preserves historical local scoring; official pins "
             "the upstream gpt-4o-2024-08-06 evaluator request/parser exactly",
    )
    parser.add_argument("--answer-extra-body", default=None, metavar="JSON",
                        help="JSON object merged into the ANSWER request body (raw-HTTP "
                             "`extra_body`). When omitted, the pinned DeepSeek "
                             "v4-flash default gets thinking disabled automatically; "
                             "custom endpoints are never given a vendor body implicitly.")
    parser.add_argument("--judge-extra-body", default=None, metavar="JSON",
                        help="JSON object merged into the JUDGE request body. When "
                             "omitted, the pinned DeepSeek v4-flash default gets "
                             "thinking disabled automatically.")
    parser.add_argument("--rejudge", default=None, metavar="RUN.json",
                        help="Re-judge a stored results JSON under the current "
                             "--judge-model (+ --judge-extra-body) — NO ingest, NO answer "
                             "calls, just the judge pass over each stored hypothesis. "
                             "Built for the deepseek-chat deprecation: re-pair the banked "
                             "historical local baseline under a replacement judge so a paired run "
                             "by the same new judge is comparable. Reports per-category "
                             "judge drift (original vs re-judged) and archives a new JSON. "
                             "Skips the benchmark. NOTE: runs produced before the "
                             "field-truncation was lifted carry clipped q/a/hypothesis — "
                             "the re-judge is then a close approximation, not byte-faithful.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--hymem-model", default=PINNED_DEEPSEEK_MODEL)
    parser.add_argument("--hymem-base-url", default=DEEPSEEK_BASE_URL)
    parser.add_argument(
        "--hymem-thinking", choices=("auto", "disabled", "off", "enabled"),
        default="auto",
    )
    parser.add_argument("--hymem-api-key", default=None)
    parser.add_argument("--data-dir", default=str(_repo_root.parent / "hymem_beam" / "data"))
    parser.add_argument("--results-dir", default="/home/node/.hermes/benchmarks")
    parser.add_argument(
        "--export-official", metavar="STRICT_RUN.json", default=None,
        help="offline: validate a completed exact full-S artifact and export "
             "the upstream question_id/hypothesis JSONL schema",
    )
    parser.add_argument("--official-output", default=None, metavar="FILE.jsonl")
    parser.add_argument("--prereg", default=None, metavar="SPEC.md")
    parser.add_argument(
        "--no-prereg", action="store_true",
        help="explicitly mark this run exploratory/development-only",
    )
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
    parser.add_argument(
        "--indexing-max-cycles", type=int, default=DEFAULT_INDEXING_MAX_CYCLES,
        help="bounded dream-cycle count before fail-close",
    )
    parser.add_argument(
        "--indexing-timeout-s", type=float, default=DEFAULT_INDEXING_TIMEOUT_S,
        help="wall-clock bound for per-question indexing convergence",
    )
    parser.add_argument(
        "--indexing-require-healthy", action=argparse.BooleanOptionalAction,
        default=True,
        help="canonical runs fail on quarantined extraction (must remain true)",
    )
    parser.add_argument("--graph-facts-first", action="store_true",
                        help="A/B OPT-OUT: restore the legacy graph-facts-first "
                             "ordering for IE/KU/PF lookups (dream-derived "
                             "graph_facts ranked above raw message_hits). The "
                             "DEFAULT is now message-first for every ability — the "
                             "production-realistic shape, since detect_ability "
                             "returns None for those lookups and graph-facts-first "
                             "caused the −14.3pp SS-user regression. Use this flag "
                             "only to reproduce the old ordering for comparison.")
    parser.add_argument("--auto-ability", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="label-free production routing (default). "
                             "--no-auto-ability explicitly enables oracle-label "
                             "steering and makes the artifact exploratory/non-comparable")
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
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument(
        "--embedding-deployment-revision",
        default=os.environ.get("HYMEM_EMBEDDING_DEPLOYMENT_REVISION"),
        help="Non-secret immutable embedding deployment revision attestation.",
    )
    parser.add_argument(
        "--embedding-deployment-tenant",
        default=os.environ.get("HYMEM_EMBEDDING_DEPLOYMENT_TENANT"),
        help="Non-secret semantic tenant/routing attestation.",
    )
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
    add_strict_run_arguments(parser)
    args = parser.parse_args()

    if args.export_official:
        if not args.official_output:
            parser.error("--export-official requires --official-output")
        # Provider-free by construction: validation + exclusive JSONL write,
        # dispatched before credential resolution, dataset loading, or clients.
        summary = export_official_predictions(
            args.export_official, args.official_output
        )
        print(
            f"Official LongMemEval predictions: {summary['count']} rows -> "
            f"{summary['path']}"
        )
        return
    if args.official_output:
        parser.error("--official-output is only valid with --export-official")
    validate_runtime_arguments(args, parser)
    args.prereg_obj = None
    if not args.rejudge:
        if bool(args.prereg) == bool(args.no_prereg):
            parser.error("pass exactly one of --prereg SPEC.md or --no-prereg")
        try:
            args.prereg_obj = resolve_prereg(None if args.no_prereg else args.prereg)
        except BenchmarkIntegrityError as exc:
            parser.error(str(exc))

    # Resolve API key
    DEEPSEEK_API_KEY = (
        args.api_key
        or os.environ.get("HYMEM_LLM_API_KEY", "")
        or os.environ.get("DEEPSEEK_API_KEY", "")
    )
    if not DEEPSEEK_API_KEY:
        config_path = Path("/home/node/.hermes/config.yaml")
        if config_path.exists():
            for line in config_path.read_text().split("\n"):
                s = line.strip()
                if s.startswith("HYMEM_LLM_API_KEY:"):
                    DEEPSEEK_API_KEY = s.split(":", 1)[1].strip().strip('"').strip("'")
                    break
    # Parse the optional per-client extra_body JSON (fail fast on bad JSON).
    def _parse_extra_body(raw: str | None, which: str) -> dict | None:
        if raw is None:
            return None
        try:
            val = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"ERROR: --{which}-extra-body is not valid JSON: {e}")
            sys.exit(1)
        if not isinstance(val, dict):
            print(f"ERROR: --{which}-extra-body must be a JSON object, got {type(val).__name__}")
            sys.exit(1)
        try:
            return validate_request_extra_body(val)
        except BenchmarkIntegrityError as exc:
            parser.error(f"--{which}-extra-body: {exc}")
    parsed_answer_body = _parse_extra_body(args.answer_extra_body, "answer")
    parsed_judge_body = _parse_extra_body(args.judge_extra_body, "judge")
    try:
        args.answer_extra_body_obj, answer_body_defaulted = resolve_model_extra_body(
            args.answer_model, args.answer_base_url, parsed_answer_body
        )
        args.judge_extra_body_obj, judge_body_defaulted = resolve_model_extra_body(
            args.judge_model, args.judge_base_url, parsed_judge_body
        )
    except BenchmarkIntegrityError as exc:
        parser.error(str(exc))
    args.extra_body_defaulted = [
        role for role, defaulted in (
            ("answer", answer_body_defaulted),
            ("judge", judge_body_defaulted),
        ) if defaulted and (not args.rejudge or role == "judge")
    ]

    # Re-judge a stored results JSON under the current judge — no dataset needed,
    # so dispatch before the (large) dataset load. Built for the deepseek-chat
    # deprecation: re-pair the banked baseline under deepseek-v4-flash.
    if args.rejudge:
        try:
            rejudge_key = resolve_endpoint_key(
                role="judge", base_url=args.judge_base_url,
                explicit_key=args.judge_api_key,
                deepseek_key=DEEPSEEK_API_KEY,
            )
        except BenchmarkIntegrityError as exc:
            parser.error(str(exc))
        _rejudge_run(args, rejudge_key)
        return

    # Resolve the input-capacity policy before dataset work can reach any
    # provider client. Unknown endpoints must state their context ceiling;
    # configured tokenizers are local-only and hashed into run identity.
    args.context_policy_obj, args.token_counter = resolve_context_policy(
        args, parser
    )
    args.max_input_tokens = args.context_policy_obj["max_input_tokens"]
    args.max_input_bytes = args.context_policy_obj["max_input_bytes"]
    args.provider_context_tokens = args.context_policy_obj[
        "provider_context_window_tokens"
    ]

    # Determine dataset path
    scale = args.scales.upper()
    args.scales = scale
    if scale == "S":
        data_file = Path(args.data_dir) / "longmemeval_s_cleaned.json"
    elif scale == "M":
        data_file = Path(args.data_dir) / f"longmemeval_{scale.lower()}_cleaned.json"
    else:  # validate_runtime_arguments should make this unreachable
        raise BenchmarkIntegrityError(f"unsupported LongMemEval scale {scale!r}")

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
        print(
            "  ⚠ Answer endpoint: "
            f"{safe_endpoint_label(args.answer_base_url, label='reader')} "
            "(PARITY READER — judge remains independently configured at "
            f"{safe_endpoint_label(args.judge_base_url, label='judge')})"
        )
    print(f"  Judge model: {args.judge_model}")
    print(f"  Workers: {args.workers}")
    embedding_identity = resolve_embedding_identity(args)
    if args.embeddings:
        declaration = embedding_identity["producer_binding"]["declaration"]
        origin = declaration.get("endpoint_origin", "local")
        src = "env" if os.environ.get("HYMEM_EMBEDDING_BASE_URL") else "local default"
        print(
            "  Embeddings: ON (semantic recall) — "
            f"{embedding_identity['vector_space_key']} @ {origin} [{src}]"
        )
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
    if (args.freeze_calibration or args.protocol_split != "full") and args.sample:
        parser.error(
            "--freeze-calibration and dev/holdout runs require --sample 0 so "
            "the frozen internal split covers the complete dataset"
        )
    source_questions = load_longmemeval_data(
        str(data_file), max_questions=None, seed=args.seed,
        strict_schema=True, scale=scale,
    )
    if args.sample > len(source_questions):
        parser.error(
            f"--sample {args.sample} exceeds the validated dataset size "
            f"({len(source_questions)}); use --sample 0 for the full set"
        )
    source_ids = validate_ids(
        (q["question_id"] for q in source_questions),
        label="LongMemEval source dataset",
    )
    questions = (
        list(source_questions)
        if args.freeze_calibration or args.protocol_split != "full"
        else select_label_blind_questions(
            source_questions, sample=args.sample, seed=args.seed
        )
    )
    total_sessions = sum(len(q.get("haystack_sessions", [])) for q in questions)
    total_msgs = sum(sum(len(s) for s in q.get("haystack_sessions", [])) for q in questions)
    print(f"  Total: {len(questions)} questions, ~{total_sessions} sessions, ~{total_msgs} messages\n")

    # Freeze every score-affecting input before a holdout can be touched.  The
    # labels remain in rows for official judging/diagnostics, never selection or
    # default routing. Runtime totals and credentials are deliberately absent.
    dataset_sha = file_hash(data_file)
    source_ids_hash = content_hash(list(source_ids))
    observed_qtype_counts = dict(Counter(
        q["question_type"] for q in source_questions
    ))
    pinned_s_source = bool(
        scale == "S" and dataset_sha == LME_S_DATASET_SHA256
        and len(source_questions) == LME_S_EXPECTED_COUNT
        and source_ids_hash == LME_S_SOURCE_IDS_HASH
        and observed_qtype_counts == LME_S_QTYPE_COUNTS
    )
    exact_full_s = bool(
        pinned_s_source and args.sample == 0
        and args.protocol_split == "full"
    )

    excluded_config = {
        "api_key", "answer_api_key", "judge_api_key", "hymem_api_key",
        "embedding_api_key", "data_dir", "results_dir",
        "checkpoint", "resume_from", "retry_failures",
        "calibration_receipt", "freeze_calibration", "dev_fraction",
        "protocol_split",
        "rejudge", "inspect_floor", "distill_dryrun", "export_official",
        "official_output", "prereg", "no_prereg", "prereg_obj",
        "tokenizer_json", "token_counter", "context_policy_obj",
        # Raw JSON strings can hide credential-shaped nested keys from the
        # recursive sanitizer. Only the parsed objects below enter identity.
        "answer_extra_body", "judge_extra_body",
        # The live memory-pipeline route may carry an opaque deployment or
        # tenant identifier.  Persist only its origin and exact digest below.
        "answer_base_url", "judge_base_url", "hymem_base_url",
    }
    strict_config = {
        key: value for key, value in vars(args).items()
        if key not in excluded_config and "api_key" not in key
    }
    subset_run = bool(args.sample > 0 and args.protocol_split == "full")
    exploratory_non_comparable = bool(
        args.prereg_obj is None or subset_run or not args.auto_ability
        or not args.indexing_require_healthy or args.retrieval_only
        or args.no_dream or not pinned_s_source
    )
    config_probe = _adapter_for_args(
        Path("/benchmark-identity/hymem.sqlite"), args, ""
    ).build_config()
    effective_hymem_config = effective_hymem_config_identity(config_probe)
    strict_config.update({
        "label_free_answer_path": bool(args.auto_ability),
        "scored_run": not args.retrieval_only,
        "exploratory_label_steering": not args.auto_ability,
        "exploratory_non_comparable": exploratory_non_comparable,
        "subset_run": subset_run,
        "sample_strategy": (
            "sha256-seed-source-index-preserve-order-v1"
            if subset_run else "all-source-order"
        ),
        "official_denominator_validated": exact_full_s,
        "source_order_validated": pinned_s_source,
        "source_ids_hash": source_ids_hash,
        "source_qtype_counts": observed_qtype_counts,
        "dataset_revision": (
            LME_S_DATASET_REVISION if pinned_s_source
            else f"unverified-local-{scale}"
        ),
        "dataset_sha256": dataset_sha,
        "dataset_expected_count": len(source_questions),
        "dataset_url": LME_S_DATASET_URL if pinned_s_source else None,
        "evaluator_commit": LME_EVALUATOR_COMMIT,
        "evaluator_sha256": LME_EVALUATOR_SHA256,
        "evaluator_url": LME_EVALUATOR_URL,
        "official_judge_model": LME_OFFICIAL_JUDGE_MODEL,
        **{
            "official_judge_" + key: value
            for key, value in secret_free_endpoint_identity(
                LME_OFFICIAL_JUDGE_BASE_URL, label="official judge"
            ).items()
        },
        "official_judge_temperature": LME_OFFICIAL_JUDGE_TEMPERATURE,
        "official_judge_max_tokens": LME_OFFICIAL_JUDGE_MAX_TOKENS,
        "official_verdict_parser": LME_OFFICIAL_VERDICT_PARSER,
        "historical_local_judge_prompts_exact_official": (
            LME_HISTORICAL_LOCAL_JUDGE_PROMPTS_EXACT_OFFICIAL
        ),
        "judge_transport_retry_policy": LME_LOCAL_RETRY_POLICY,
        "official_transport_retry_policy": LME_UPSTREAM_RETRY_POLICY,
        "official_transport_exact": False,
        "retrieval_usage_owner": (
            "separate-retrieval-meter" if args.retrieval_only and args.distill
            else "reader" if args.distill else "none"
        ),
        "prereg": args.prereg_obj,
        "indexing_require_healthy": bool(args.indexing_require_healthy),
        "embedding_runtime": embedding_identity,
        "context_policy": args.context_policy_obj,
        "extraction_canary": extraction_canary_policy(
            prompt_version=config_probe.prompt_version
        ),
        "effective_hymem_config": effective_hymem_config,
    })
    runtime_extraction_binding = validate_extraction_canary_config_binding(
        strict_config["extraction_canary"], effective_hymem_config
    )
    reader_endpoint = secret_free_endpoint_identity(
        args.answer_base_url, label="reader"
    )
    judge_endpoint = secret_free_endpoint_identity(
        args.judge_base_url, label="judge"
    )
    pipeline_base = validate_safe_endpoint(args.hymem_base_url, label="memory pipeline")
    pipeline_endpoint = secret_free_endpoint_identity(
        pipeline_base, label="memory pipeline"
    )
    strict_config.update({
        "answer_endpoint_origin": reader_endpoint["endpoint_origin"],
        "answer_endpoint_sha256": reader_endpoint["endpoint_sha256"],
        "judge_endpoint_origin": judge_endpoint["endpoint_origin"],
        "judge_endpoint_sha256": judge_endpoint["endpoint_sha256"],
        "hymem_endpoint_origin": pipeline_endpoint["endpoint_origin"],
        "hymem_endpoint_sha256": pipeline_endpoint["endpoint_sha256"],
    })
    pipeline_host = (__import__("urllib.parse", fromlist=["urlsplit"])
                     .urlsplit(pipeline_base).hostname or "").casefold()
    pipeline_sends_thinking = args.hymem_thinking == "disabled" or (
        args.hymem_thinking == "auto"
        and ("deepseek" in pipeline_host or "deepseek" in args.hymem_model.casefold())
    )
    try:
        pipeline_transport_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BenchmarkIntegrityError(
            "memory pipeline transport version is unavailable"
        ) from exc
    from hymem.contrib.openai_client import (
        DEFAULT_LLM_TIMEOUT_SECONDS,
        llm_attestation_sha256,
        openai_compatible_producer_declaration,
    )
    from hymem.extraction.producer import producer_binding_from_typed_declaration

    pipeline_revision_sha256 = llm_attestation_sha256(
        os.environ.get("HYMEM_LLM_DEPLOYMENT_REVISION"),
        label="memory pipeline deployment revision",
    )
    pipeline_tenant_sha256 = llm_attestation_sha256(
        os.environ.get("HYMEM_LLM_DEPLOYMENT_TENANT"),
        label="memory pipeline deployment tenant",
    )
    pipeline_aggregation_producer = producer_binding_from_typed_declaration(
        openai_compatible_producer_declaration(
            model=args.hymem_model,
            endpoint=pipeline_base,
            thinking_mode=args.hymem_thinking,
            effective_extra_body=(
                {"thinking": {"type": "disabled"}}
                if pipeline_sends_thinking else {}
            ),
            transport_package_version=pipeline_transport_version,
            request_timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            deployment_revision_sha256=pipeline_revision_sha256,
            deployment_tenant_sha256=pipeline_tenant_sha256,
            require_consistent_thinking=True,
        ),
        declaration_hook="aggregation_producer_declaration",
    )
    strict_models = {
        "reader": {
            "provider": _provider_for_url(args.answer_base_url),
            "model": args.answer_model,
            **reader_endpoint,
            "temperature": 0.0, "max_tokens": 1024,
            "extra_body": args.answer_extra_body_obj,
        },
        "judge": {
            "provider": _provider_for_url(args.judge_base_url),
            "model": args.judge_model,
            **judge_endpoint,
            "temperature": 0.0, "max_tokens": 10,
            "n": 1 if args.judge_protocol == "official" else None,
            "extra_body": args.judge_extra_body_obj,
            "protocol": args.judge_protocol,
            "evaluator_commit": LME_EVALUATOR_COMMIT,
            "evaluator_sha256": LME_EVALUATOR_SHA256,
            "verdict_parser": (
                LME_OFFICIAL_VERDICT_PARSER if args.judge_protocol == "official"
                else "anchored-exclusive-yes-no-local-v1"
            ),
            "prompt_exact_official": args.judge_protocol == "official",
            "retry_policy": LME_LOCAL_RETRY_POLICY,
        },
        "memory_pipeline": {
            "provider": _provider_for_url(args.hymem_base_url),
            "model": args.hymem_model,
            **pipeline_endpoint,
            "thinking_mode": args.hymem_thinking,
            "effective_extra_body": (
                {"thinking": {"type": "disabled"}}
                if pipeline_sends_thinking else {}
            ),
            "aggregation_producer": pipeline_aggregation_producer,
            "deployment_revision_sha256": pipeline_revision_sha256,
            "deployment_tenant_sha256": pipeline_tenant_sha256,
            "transport_package_version": pipeline_transport_version,
            "request_timeout_seconds": DEFAULT_LLM_TIMEOUT_SECONDS,
        },
        "embedding": embedding_identity,
    }
    strict_config["official_judge_match"] = official_judge_match(
        strict_config, strict_models
    )
    if args.judge_protocol == "official" and not strict_config["official_judge_match"]:
        parser.error(
            "--judge-protocol official requires gpt-4o-2024-08-06 at "
            "https://api.openai.com/v1, temperature 0, max_tokens 10, and no extra_body"
        )
    all_ids = validate_ids(
        (q.get("question_id") for q in questions), label="LongMemEval dataset"
    )
    if args.freeze_calibration:
        receipt = freeze_calibration(
            args.freeze_calibration,
            benchmark="LongMemEval",
            dataset_hash=dataset_sha,
            ids=source_ids,
            config=strict_config,
            models=strict_models,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
        )
        print(f"Frozen internal calibration receipt: {args.freeze_calibration}")
        print(f"  dev={len(receipt['dev_ids'])}, holdout={len(receipt['holdout_ids'])}")
        return

    calibration = None
    if args.calibration_receipt:
        calibration = load_calibration(
            args.calibration_receipt,
            benchmark="LongMemEval",
            dataset_hash=dataset_sha,
            config=strict_config,
            models=strict_models,
            ids=source_ids,
        )

    def _role_key(role: str, base_url: str, explicit: str | None) -> str:
        try:
            return resolve_endpoint_key(
                role=role, base_url=base_url, explicit_key=explicit,
                deepseek_key=DEEPSEEK_API_KEY,
            )
        except BenchmarkIntegrityError as exc:
            parser.error(str(exc))

    # Floor inspector: a diagnostic, not a benchmark run — dump and exit.
    if args.inspect_floor:
        pipeline_key = _role_key(
            "memory pipeline", args.hymem_base_url, args.hymem_api_key
        )
        try:
            args.embedding_api_key = resolve_embedding_key(args)
        except BenchmarkIntegrityError as exc:
            parser.error(str(exc))
        _inspect_floor_questions(questions, args, pipeline_key)
        return

    # Distillation dry-run (G-P1a front-run gate): offline test on the banked
    # synthesis misses — dump the verdict and exit, no full benchmark.
    if args.distill_dryrun:
        pipeline_key = _role_key(
            "memory pipeline", args.hymem_base_url, args.hymem_api_key
        )
        answer_key = _role_key(
            "reader", args.answer_base_url, args.answer_api_key
        )
        judge_key = _role_key(
            "judge", args.judge_base_url, args.judge_api_key
        )
        try:
            args.embedding_api_key = resolve_embedding_key(args)
        except BenchmarkIntegrityError as exc:
            parser.error(str(exc))
        _distill_dryrun_questions(
            questions, args, pipeline_key, answer_key, judge_key
        )
        return

    selected_ids = select_protocol_ids(
        all_ids, split=args.protocol_split, receipt=calibration
    )
    selected_set = set(selected_ids)
    questions = [q for q in questions if q["question_id"] in selected_set]
    if tuple(q["question_id"] for q in questions) != selected_ids:
        raise BenchmarkIntegrityError("selected LongMemEval id order drifted")
    manifest = build_manifest(
        benchmark="LongMemEval",
        code_sha256=longmemeval_code_hash(),
        data_sha256=dataset_sha,
        config=strict_config,
        models=strict_models,
        seed=args.seed,
        expected_ids=selected_ids,
        protocol_split=args.protocol_split,
        calibration=calibration,
    )
    # Re-validate the serialized manifest boundary itself.  This is after
    # sanitization/hash construction and before any benchmark provider can be
    # called, so neither a resume artifact nor config/report drift can rely on
    # the independently held in-process prompt label.
    manifest_extraction_binding = validate_extraction_canary_config_binding(
        manifest["config"].get("extraction_canary"),
        manifest["config"].get("effective_hymem_config"),
    )
    if manifest_extraction_binding != runtime_extraction_binding:
        raise BenchmarkIntegrityError(
            "LongMemEval manifest extraction contract differs from runtime config"
        )
    extraction_prompt_version = manifest_extraction_binding["prompt_version"]
    # HyMem's current architecture/prompt campaign was informed by the public S
    # set.  A frozen internal split remains useful for disciplined iteration,
    # but cannot retroactively create clean benchmark evidence.
    manifest["development_only"] = True
    manifest["official_split"] = False
    manifest["official_comparable"] = False
    manifest["run_id"] = content_hash({
        key: value for key, value in manifest.items() if key != "run_id"
    })
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path, is_resume = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        resume_from=args.resume_from,
        base_dir=results_dir,
        benchmark="longmemeval",
        run_id=manifest["run_id"],
    )
    ledger = AtomicCheckpoint(
        checkpoint_path,
        manifest=manifest,
        expected_ids=selected_ids,
        resume=is_resume,
        retry_failures=args.retry_failures,
        scored=not args.retrieval_only,
    )
    if _owned_ledgers is not None:
        _owned_ledgers.append(ledger)
    pending = set(ledger.pending_ids)
    print(f"  Strict checkpoint: {checkpoint_path} "
          f"({len(pending)} pending / {len(selected_ids)} expected)")
    work_questions = [q for q in questions if q["question_id"] in pending]
    work_total = len(work_questions)
    pipeline_key = ""
    if work_total:
        pipeline_key = _role_key(
            "memory pipeline", args.hymem_base_url, args.hymem_api_key
        )
        try:
            args.embedding_api_key = resolve_embedding_key(args)
        except BenchmarkIntegrityError as exc:
            parser.error(str(exc))

    # Terminal checkpoint publication is provider-free. For pending work,
    # construct only clients that the selected mode can actually reach.
    # Under --retrieval-only the reader and judge are never reached, and the
    # clients enforce that rather than trusting the branch: PoisonLLM raises,
    # CountingLLM records what distillation and the store actually spend.
    needs_reader_client = bool(work_total and (
        not args.retrieval_only or args.distill
    ))
    needs_judge_client = bool(work_total and not args.retrieval_only)
    if needs_reader_client:
        answer_api_key = _role_key(
            "reader", args.answer_base_url, args.answer_api_key
        )
        answer_llm = owned_clients.own(
            LLMClient(
                args.answer_model, answer_api_key, base_url=args.answer_base_url,
                extra_body=args.answer_extra_body_obj,
            ),
            label="reader client",
        )
    else:
        answer_llm = PoisonLLM("reader")
    if needs_judge_client:
        judge_api_key = _role_key(
            "judge", args.judge_base_url, args.judge_api_key
        )
        judge_llm = owned_clients.own(
            LLMClient(
                args.judge_model, judge_api_key, base_url=args.judge_base_url,
                extra_body=args.judge_extra_body_obj,
                n=1 if args.judge_protocol == "official" else None,
            ),
            label="judge client",
        )
    else:
        judge_llm = PoisonLLM("judge")
    retrieval_counter = None
    distill_llm = None
    if args.retrieval_only:
        # The reader and the judge become unreachable OBJECTS, not skipped
        # branches. Distillation keeps a real (counted) client because it is
        # part of retrieval and genuinely fires.
        if args.distill:
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
    all_results: list[dict] = list(ledger.reconcile().rows)
    segment_id = (
        f"process-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{os.getpid()}"
    )
    pipeline_usage_instances: list[dict[str, Any]] = []
    embedding_usage_instances: list[dict[str, Any]] = []
    indexing_runs: list[dict[str, Any]] = []
    runtime_instrumentation_errors: list[str] = []
    extraction_canary_report: dict[str, Any] = (
        skipped_extraction_canary(
            "no_pending_work", prompt_version=extraction_prompt_version
        )
        if not work_total else
        skipped_extraction_canary(
            "no_dream", prompt_version=extraction_prompt_version
        )
        if args.no_dream else
        {
            **extraction_canary_policy(
                prompt_version=extraction_prompt_version
            ),
            "status": "pending",
        }
    )
    if work_total and not args.no_dream:
        validate_extraction_canary_report(
            extraction_canary_report, expected_mode="pending",
            expected_prompt_version=extraction_prompt_version,
        )

    def unavailable_llm_usage() -> dict[str, Any]:
        return {
            "calls": None, "calls_available": False,
            "request_attempts": None, "request_attempts_available": False,
            "successful_responses": None,
            "successful_responses_available": False,
            "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": None, "token_usage_available": False,
            "latency_s": None, "latency_available": False,
            "cost_usd": None, "cost_available": False,
        }

    def zero_llm_usage() -> dict[str, Any]:
        return usage_snapshot(PoisonLLM("unused usage role"))

    def zero_embedding_usage() -> dict[str, Any]:
        identity = manifest["models"]["embedding"]
        return {
            "configured": identity["configured"],
            "backend": identity["backend"],
            "quality": identity["quality"],
            "network_free": identity["network_free"],
            "model": identity["vector_space_key"],
            "dimension": identity["dimension"],
            "identity_available": True,
            "identity_exact": identity["identity_exact"],
            "reuse_scope": identity["reuse_scope"],
            "calls": 0, "calls_available": True,
            "request_attempts": 0,
            "request_attempts_available": True,
            "successful_responses": 0,
            "successful_responses_available": True,
            "input_count": 0, "input_count_available": True,
            "input_characters": 0,
            "input_characters_available": True,
            "prompt_tokens": None, "total_tokens": None,
            "provider_token_usage_available": False,
            "latency_s": 0.0, "latency_available": True,
            "cost_usd": None, "cost_available": False,
        }

    def unavailable_embedding_usage() -> dict[str, Any]:
        if not bool(args.embeddings):
            return zero_embedding_usage()
        return {
            "configured": True, "backend": "unavailable",
            "quality": "none", "network_free": None,
            "model": None, "dimension": None,
            "identity_available": False,
            "identity_exact": None, "reuse_scope": None,
            "calls": None, "calls_available": False,
            "request_attempts": None,
            "request_attempts_available": False,
            "successful_responses": None,
            "successful_responses_available": False,
            "input_count": None, "input_count_available": False,
            "input_characters": None,
            "input_characters_available": False,
            "prompt_tokens": None, "total_tokens": None,
            "provider_token_usage_available": False,
            "latency_s": None, "latency_available": False,
            "cost_usd": None, "cost_available": False,
        }

    runtime_capture_lock = threading.RLock()
    captured_runtime_ids: set[str] = set()

    def _capture_runtime(row: dict[str, Any]) -> None:
        # Worker-finally handoff and coordinator result collection intentionally
        # overlap.  Keying by the protocol-unique id makes that overlap safe and
        # prevents cumulative per-question meters from being counted twice.
        runtime_id = str(row.get("question_id", "unknown"))
        with runtime_capture_lock:
            if runtime_id in captured_runtime_ids:
                return
            captured_runtime_ids.add(runtime_id)
            pipeline = row.get("memory_pipeline_usage")
            if not isinstance(pipeline, dict):
                runtime_instrumentation_errors.append(
                    "memory_pipeline_usage:Unavailable"
                )
            pipeline_usage_instances.append(
                dict(pipeline) if isinstance(pipeline, dict) else unavailable_llm_usage()
            )
            embedding = row.get("embedding_usage")
            if bool(args.embeddings) and not isinstance(embedding, dict):
                runtime_instrumentation_errors.append(
                    "embedding_usage:Unavailable"
                )
            embedding_usage_instances.append(
                dict(embedding) if isinstance(embedding, dict) else
                unavailable_embedding_usage()
            )
            indexing = row.get("indexing")
            if isinstance(indexing, dict):
                indexing_runs.append({
                    "question_id": row.get("question_id"),
                    "summary": dict(indexing),
                })

    def _observable_attempts() -> int:
        with runtime_capture_lock:
            return len(captured_runtime_ids)

    def _segment(status: str, attempted: int) -> dict:
        # Worker-finally callbacks can update these collections while an
        # ordinary result is being checkpointed.  Take one coherent image under
        # their callback lock, then aggregate outside it.
        with runtime_capture_lock:
            instrumentation_errors: list[str] = list(
                runtime_instrumentation_errors
            )
            pipeline_snapshots = [
                dict(snapshot) for snapshot in pipeline_usage_instances
            ]
            embedding_snapshots = [
                dict(snapshot) for snapshot in embedding_usage_instances
            ]
            indexing_snapshots = [dict(item) for item in indexing_runs]

        def captured(label: str, fn, fallback):
            try:
                return fn()
            except Exception as exc:
                instrumentation_errors.append(
                    f"{label}:{_bounded_exception_type(exc)}"
                )
                return fallback()

        reader_usage = captured(
            "reader_usage", lambda: usage_snapshot(answer_llm),
            unavailable_llm_usage,
        )
        judge_usage = captured(
            "judge_usage", lambda: usage_snapshot(judge_llm),
            unavailable_llm_usage,
        )
        retrieval_usage = captured(
            "retrieval_usage",
            lambda: (
                usage_snapshot(retrieval_counter)
                if retrieval_counter is not None else zero_llm_usage()
            ),
            unavailable_llm_usage,
        )
        pipeline_usage = captured(
            "memory_pipeline_usage",
            lambda: (
                aggregate_usage_snapshots(pipeline_snapshots)
                if pipeline_snapshots else usage_snapshot(
                    PoisonLLM("unused memory pipeline")
                )
            ),
            unavailable_llm_usage,
        )

        embedding_usage = captured(
            "embedding_usage",
            lambda: (
                aggregate_embedding_usage_snapshots(
                    embedding_snapshots
                ) if embedding_snapshots else zero_embedding_usage()
            ),
            unavailable_embedding_usage,
        )
        return {
            "segment_id": segment_id,
            "status": status,
            "elapsed_s": time.time() - start_time,
            "attempted_attempts": attempted,
            # Use the manifest's recursively sanitized identity. Provider
            # extension bodies can contain credential-shaped fields and the
            # checkpoint writer intentionally preserves execution segments.
            "model_identities": manifest["models"],
            "reader_usage": reader_usage,
            "judge_usage": judge_usage,
            "retrieval_usage": retrieval_usage,
            "memory_pipeline_usage": pipeline_usage,
            "embedding_usage": embedding_usage,
            "latest_indexing": (
                dict(indexing_snapshots[-1]) if indexing_snapshots else None
            ),
            "indexing_runs": indexing_snapshots,
            # This dedicated-client spend is intentionally NOT part of
            # memory_pipeline_usage. It is separately attributable and occurs
            # once here, before any worker can create a benchmark store.
            "extraction_canary": secret_free_extraction_canary_report(
                extraction_canary_report
            ),
            "instrumentation_errors": instrumentation_errors,
        }

    if work_total:
        ledger.update_execution_segment(segment_id, _segment("running", 0))
        if not args.no_dream:
            try:
                extraction_canary_report = run_configured_extraction_canary(
                    api_key=pipeline_key,
                    base_url=args.hymem_base_url,
                    model=args.hymem_model,
                    thinking=args.hymem_thinking,
                    prompt_version=extraction_prompt_version,
                )
                _validate_pipeline_extraction_canary(
                    extraction_canary_report, args, mode="required",
                )
            except ExtractionCanaryError as exc:
                extraction_canary_report = dict(exc.report)
                validate_extraction_canary_report(
                    extraction_canary_report,
                    expected_mode="failed",
                    expected_client=extraction_canary_client_policy(
                        base_url=args.hymem_base_url,
                        model=args.hymem_model,
                        thinking=args.hymem_thinking,
                    ),
                    require_client_closed=True,
                    expected_prompt_version=extraction_prompt_version,
                )
                ledger.update_execution_segment(
                    segment_id, _segment("running", 0)
                )
                raise
            ledger.update_execution_segment(segment_id, _segment("running", 0))
            print_extraction_canary(extraction_canary_report)
        else:
            _validate_pipeline_extraction_canary(
                extraction_canary_report, args, mode="no_dream",
            )
            print_extraction_canary(extraction_canary_report)
    elif is_resume:
        # A crash after its last durable row but before segment finalization is
        # recoverable without constructing a provider client. Close that history
        # with an explicit zero-work segment. Already-finalized checkpoints are
        # terminal and need no mutation.
        try:
            ledger.update_execution_segment(
                segment_id, _segment("complete", 0)
            )
        except BenchmarkIntegrityError as exc:
            if "cannot mutate a finalized checkpoint" not in str(exc):
                raise

    def _progress(done: int):
        elapsed = time.time() - start_time
        suffix_w = f" (×{args.workers} workers)" if args.workers > 1 else ""
        if args.retrieval_only:
            # No verdicts exist. "Acc: 0.0%" here is not a low score, it is a
            # measurement that was never taken.
            print(f"  ── Progress: {done}/{work_total} attempted this segment | "
                  f"retrieval-only (no "
                  f"verdicts) | Elapsed: {elapsed:.0f}s | "
                  f"Avg: {elapsed/max(1, done):.0f}s/q{suffix_w}", flush=True)
            return
        acc = accuracy(all_results)
        n_err = len(judge_error_rows(all_results))
        suffix = f" (×{args.workers} workers)" if args.workers > 1 else ""
        # An outage should be visible WHILE the run burns reader calls, not only
        # in the post-mortem — that is the whole point of D3. Silent when zero:
        # the run summary states the denominator instead.
        err = f" | strict judge failures: {n_err}" if n_err else ""
        print(f"  ── Progress: {done}/{work_total} attempted this segment | "
              f"strict full-denominator Acc: {acc*100:.1f}% | "
              f"Elapsed: {elapsed:.0f}s | Avg: {elapsed/max(1, done):.0f}s/q"
              f"{suffix}{err}", flush=True)

    if args.workers > 1 and work_total:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        parallel_stop = threading.Event()
        parallel_state_lock = threading.RLock()
        parallel_primary: list[BaseException] = []

        def _signal_parallel_abort(exc: BaseException) -> BaseException:
            # This lock also orders fatal worker notification against the
            # coordinator's final pre-record stop check.  Consequently no
            # ledger.record call can begin after a structural worker failure.
            with parallel_state_lock:
                if not parallel_primary:
                    parallel_primary.append(exc)
                parallel_stop.set()
                return parallel_primary[0]

        def _raise_parallel_primary() -> None:
            with parallel_state_lock:
                if parallel_stop.is_set():
                    if parallel_primary:
                        raise parallel_primary[0]
                    raise _ParallelQuestionStopped()

        def _parallel_evaluate(qi: int, q_data: dict) -> dict:
            if parallel_stop.is_set():
                raise _ParallelQuestionStopped()
            try:
                return _evaluate_one_question(
                    qi, work_total, q_data, args, answer_llm, judge_llm,
                    pipeline_key, distill_llm,
                    _parallel_stop=parallel_stop,
                    _on_fatal_abort=_signal_parallel_abort,
                    _on_runtime=_capture_runtime,
                )
            except _ParallelQuestionStopped:
                raise
            except BaseException as exc:
                # Ordinary provider/question exceptions retain the historical
                # failed-row behavior below.  Structural and process-control
                # failures signal siblings before this future becomes ready.
                if (
                    not isinstance(exc, Exception)
                    or is_structural_benchmark_error(exc)
                ):
                    primary = _signal_parallel_abort(exc)
                    if primary is not exc:
                        raise _ParallelQuestionStopped()
                raise

        worker_count = min(args.workers, work_total)
        pool = ThreadPoolExecutor(max_workers=worker_count)
        work_iter = iter(enumerate(work_questions))
        futures: dict[Any, tuple[int, dict]] = {}
        attempted_count = 0
        checkpoint_abort: list[BaseException] = []

        def _fill_parallel_window() -> None:
            while len(futures) < worker_count and not parallel_stop.is_set():
                try:
                    qi, q_data = next(work_iter)
                except StopIteration:
                    return
                future = pool.submit(_parallel_evaluate, qi, q_data)
                futures[future] = (qi, q_data)

        try:
            _fill_parallel_window()
            while futures:
                _raise_parallel_primary()
                ready, _pending_futures = wait(
                    tuple(futures), return_when=FIRST_COMPLETED,
                )
                # A worker signals before publishing a structural exception to
                # its Future.  Check before accepting any simultaneously ready
                # success, and process the ready batch in input order.
                _raise_parallel_primary()
                ordered_ready = sorted(ready, key=lambda fut: futures[fut][0])
                for fut in ordered_ready:
                    _qi, q_data = futures.pop(fut)
                    _raise_parallel_primary()
                    try:
                        result = fut.result()
                        _capture_runtime(result)
                    except _ParallelQuestionStopped:
                        _raise_parallel_primary()
                        raise
                    except BenchmarkCleanupError as exc:
                        primary = _signal_parallel_abort(exc)
                        if primary is exc:
                            raise
                        raise primary
                    except Exception as exc:
                        if is_structural_benchmark_error(exc):
                            primary = _signal_parallel_abort(exc)
                            if primary is exc:
                                raise
                            raise primary
                        result = {
                            "question_id": q_data["question_id"],
                            "question_type": q_data.get("question_type", "unknown"),
                            "correct": False,
                            "benchmark_failure": (
                                f"worker_failure:{_bounded_exception_type(exc)}"
                            ),
                            "oracle_ability": LME_ABILITY_BY_TYPE.get(
                                q_data.get("question_type")
                            ),
                            "detected_ability": None,
                            "ability_used": (
                                None if args.auto_ability else
                                LME_ABILITY_BY_TYPE.get(q_data.get("question_type"))
                            ),
                            "retrieval_only": bool(args.retrieval_only),
                            "distill_fired": False,
                            "distill_calls": 0,
                        }
                        _capture_runtime(result)

                    next_attempted = attempted_count + 1
                    # Persistence and fatal notification share one gate: once
                    # any fatal fault wins it, no later record call can start.
                    with parallel_state_lock:
                        _raise_parallel_primary()
                        try:
                            ledger.record(
                                q_data["question_id"], row=result,
                                execution_segment=_segment(
                                    "running", next_attempted
                                ),
                            )
                        except BaseException as exc:
                            checkpoint_abort.append(exc)
                            primary = _signal_parallel_abort(exc)
                            if primary is exc:
                                raise
                            raise primary
                    attempted_count = next_attempted
                    all_results = list(ledger.reconcile().rows)
                    if (
                        attempted_count % 10 == 0
                        or attempted_count == work_total
                    ):
                        _progress(attempted_count)
                # Refill only after every ready result has been reconciled.  At
                # most worker_count evaluations therefore exist at any time.
                _fill_parallel_window()
        except BaseException as exc:
            if isinstance(exc, _ParallelQuestionStopped):
                with parallel_state_lock:
                    primary = parallel_primary[0] if parallel_primary else exc
                    parallel_stop.set()
            else:
                primary = _signal_parallel_abort(exc)
            for future in futures:
                future.cancel()
            if primary is exc:
                raise
            raise primary
        finally:
            primary_exception = sys.exc_info()[1]
            if primary_exception is not None:
                parallel_stop.set()
                for future in futures:
                    future.cancel()
            # Explicit shutdown is required here.  The executor context
            # manager cannot request cancellation of queued futures and would
            # wait while those tasks continued to spend.
            shutdown_complete = [False]

            def _shutdown_and_capture_runtime() -> None:
                pool.shutdown(wait=True, cancel_futures=True)
                # Futures may have finished after the coordinator's fatal record
                # failure.  Inspect their values only for runtime evidence; they
                # must never reach ledger.record on this abort path.
                for future in tuple(futures):
                    if future.cancelled():
                        continue
                    try:
                        sibling_result = future.result()
                    except BaseException:
                        continue
                    if isinstance(sibling_result, dict):
                        _capture_runtime(sibling_result)
                shutdown_complete[0] = True

            cleanup_actions = [("resource_close", _shutdown_and_capture_runtime)]
            if checkpoint_abort:
                cleanup_actions.append((
                    "execution_segment_snapshot",
                    lambda: ledger.update_execution_segment(
                        segment_id,
                        _segment(
                            "complete" if shutdown_complete[0] else "running",
                            _observable_attempts(),
                        ),
                    ),
                ))
            run_cleanup_actions(
                cleanup_actions,
                primary_exception=primary_exception,
            )
    else:
        checkpoint_abort: list[BaseException] = []
        try:
            for qi, q_data in enumerate(work_questions):
                result = _evaluate_one_question(
                    qi, work_total, q_data, args, answer_llm, judge_llm,
                    pipeline_key, distill_llm,
                    _on_runtime=_capture_runtime,
                )
                _capture_runtime(result)
                try:
                    ledger.record(
                        q_data["question_id"], row=result,
                        execution_segment=_segment(
                            "running", _observable_attempts()
                        ),
                    )
                except BaseException as exc:
                    checkpoint_abort.append(exc)
                    raise
                all_results = list(ledger.reconcile().rows)
                if (qi + 1) % 10 == 0:
                    _progress(qi + 1)
        finally:
            primary_exception = sys.exc_info()[1]
            if checkpoint_abort:
                run_cleanup_actions(
                    [("execution_segment_snapshot", lambda:
                        ledger.update_execution_segment(
                            segment_id,
                            _segment("complete", _observable_attempts()),
                        ))],
                    primary_exception=primary_exception,
                )

    elapsed = time.time() - start_time
    if work_total:
        ledger.update_execution_segment(
            segment_id, _segment("complete", work_total)
        )
    checkpoint_snapshot = ledger.finalize()
    all_results = list(ledger.reconcile().rows)

    # Compute optional diagnostics defensively, then publish the immutable
    # row/denominator evidence BEFORE any presentation function. A formatting
    # bug must never strand an expensive completed checkpoint.
    diagnostic_errors: dict[str, str] = {}

    def _diagnostic(name: str, fn, default):
        try:
            return fn(all_results)
        except Exception as exc:
            diagnostic_errors[name] = _bounded_exception_type(exc)
            return default

    if args.retrieval_only:
        scores: dict[str, dict] = {}
        abstention_diag: dict = {}
        recall_diag: dict = {}
    else:
        scores = _diagnostic("scores", compute_scores, {
            "OVERALL": {
                "accuracy": strict_accuracy(all_results),
                "count": len(all_results),
            }
        })
        abstention_diag = _diagnostic(
            "abstention", compute_abstention_scores, {}
        )
        recall_diag = _diagnostic("recall", compute_recall_diagnostics, {})
    router_diag = _diagnostic("router", compute_router_diagnostics, {})

    conditional_rows = scored(all_results) if not args.retrieval_only else []
    payload = {
        "benchmark": "LongMemEval",
        "version": "strict-v1",
        "date": datetime.now(timezone.utc).isoformat(),
        "protocol_disclosure": (
            "label-free default routing; labels used only for official judging "
            "and post-answer diagnostics" if args.auto_ability else
            "EXPLORATORY NON-COMPARABLE: oracle question-type routing enabled"
        ),
        "scores": {qtype: {
            "accuracy": round(data["accuracy"] * 100, 1),
            "count": data["count"],
        } for qtype, data in scores.items()},
        "conditional_judged_only": {
            "accuracy": accuracy(conditional_rows) if conditional_rows else None,
            "count": len(conditional_rows),
        },
        "abstention_diagnostics": abstention_diag,
        "recall_diagnostics": recall_diag,
        "router_diagnostics": router_diag,
        "diagnostic_errors": diagnostic_errors,
        # Canonical ordered row commitment. The full artifact digest lives
        # outside the archive (pointer/registry) to avoid self-reference.
        "result_digest": content_hash(sanitize_for_artifact(all_results)),
    }
    if args.retrieval_only:
        # What the mode ACTUALLY spent, measured rather than assumed. It skips
        # the reader and the judge; it does not skip distillation, and it
        # cannot vouch for what reranking does inside the store.
        cost_segments = checkpoint_snapshot.get("execution_segments", [])
        exact_retrieval_calls = None
        if cost_segments and all(
            isinstance(segment, dict)
            and segment.get("status") == "complete"
            and isinstance(segment.get("retrieval_usage"), dict)
            and segment["retrieval_usage"].get("calls_available") is True
            and isinstance(segment["retrieval_usage"].get("calls"), int)
            for segment in cost_segments
        ):
            exact_retrieval_calls = sum(
                segment["retrieval_usage"]["calls"] for segment in cost_segments
            )
        payload["retrieval_cost"] = {
            "usage_owner": strict_config["retrieval_usage_owner"],
            "llm_calls": exact_retrieval_calls,
            "answer_calls": 0,
            "judge_calls": 0,
            "distill_calls": sum(r.get("distill_calls") or 0
                                 for r in all_results),
        }

    # The archive is exclusive/immutable. The stable filename is only a small
    # atomic pointer, never a mutable second copy of the benchmark evidence.
    results_path = results_dir / "longmemeval-v2-hymem.json"
    archive_now = datetime.now(timezone.utc)
    stamp = archive_now.strftime("%Y%m%dT%H%M%SZ")
    publication_nonce = archive_now.strftime("%f")
    archive_path = results_dir / (
        f"longmemeval-v2-hymem-{stamp}-{publication_nonce}-seed{args.seed}-"
        f"strict-{manifest['run_id'].removeprefix('sha256:')[:12]}.json"
    )
    artifact = prepare_checkpoint_artifact(ledger, payload=payload)
    complete_segments = [
        segment for segment in checkpoint_snapshot.get("execution_segments", [])
        if isinstance(segment, dict) and segment.get("status") == "complete"
    ]
    final_segment = complete_segments[-1] if complete_segments else {}
    answer_usage = dict(
        final_segment.get("reader_usage") or unavailable_llm_usage()
    )
    judge_usage = dict(
        final_segment.get("judge_usage") or unavailable_llm_usage()
    )
    if _owned_ledgers is not None:
        _owned_ledgers[:] = [
            owned for owned in _owned_ledgers if owned is not ledger
        ]
    # Final usage is frozen in ``artifact`` before transports close.  Neither
    # the immutable archive nor its latest pointer may exist unless provider
    # teardown and checkpoint lease release both succeed.
    output = publish_prepared_artifact_after_cleanup(
        archive_path,
        artifact,
        cleanup_actions=[
            ("resource_close", owned_clients.close),
            ("checkpoint_close", ledger.close),
        ],
    )
    artifact_digest = content_hash(output)
    write_latest_pointer(results_path, archive=archive_path,
                         run_id=manifest["run_id"],
                         artifact_digest=artifact_digest)
    print(f"  Archived: {archive_path.name}", flush=True)

    print(f"\nEvaluation complete in {elapsed:.0f}s")
    print(f"  Answer calls: {answer_usage['calls']}, "
          f"Judge calls: {judge_usage['calls']}")
    totals = (answer_usage["total_tokens"], judge_usage["total_tokens"])
    print("  Total tokens: " + (
        str(sum(totals)) if all(value is not None for value in totals)
        else "unavailable (provider usage missing)"
    ))
    if args.distill:
        n_fired = sum(1 for r in all_results if r.get("distill_fired"))
        n_calls = sum(r.get("distill_calls", 0) for r in all_results)
        owner = (
            "the separately metered retrieval usage"
            if args.retrieval_only else "reader usage"
        )
        print(f"  Distill: fired on {n_fired}/{len(all_results)} questions, "
              f"{n_calls} extraction calls (owned by {owner})")
    print(f"  Avg time/question: {elapsed/len(questions):.0f}s")

    # Presentation occurs only after durable publication.
    if args.retrieval_only:
        print("\n  Retrieval-only diagnostic: no accuracy was measured.")
    else:
        print(f"\n  {judge_error_note(all_results)}")
        print_report(scores, {
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "num_questions": len(questions),
            "top_k": args.top_k,
            "scale": scale,
        })
        if abstention_diag:
            print_abstention_scores(abstention_diag)
        if recall_diag:
            print_recall_diagnostics(recall_diag)
    if router_diag:
        print_router_diagnostics(router_diag, args.auto_ability)

    print(f"\nResults saved to {results_path}")


def _main(_owned_ledgers: list[AtomicCheckpoint] | None = None):
    """Run LongMemEval and close shared clients after their final snapshots."""

    owns_ledgers = _owned_ledgers is None
    ledgers = [] if _owned_ledgers is None else _owned_ledgers
    try:
        with OwnedResourceScope(
            "LongMemEval shared provider clients"
        ) as owned_clients:
            return _run_main(ledgers, owned_clients)
    finally:
        if owns_ledgers:
            run_cleanup_actions(
                [("checkpoint_close", ledger.close)
                 for ledger in reversed(ledgers)],
                primary_exception=sys.exc_info()[1],
            )


def main():
    """CLI entry point; always release a checkpoint lease on BaseException."""

    owned_ledgers: list[AtomicCheckpoint] = []
    try:
        return _main(owned_ledgers)
    finally:
        run_cleanup_actions(
            [("checkpoint_close", ledger.close)
             for ledger in reversed(owned_ledgers)],
            primary_exception=sys.exc_info()[1],
        )


if __name__ == "__main__":
    main()
