#!/usr/bin/env python3
"""
HyMem LoCoMo Benchmark Adapter
==============================
Runs LoCoMo ("Evaluating Very Long-Term Conversational Memory of LLM Agents",
Maharana et al., ACL 2024 — snap-research/locomo, `locomo10.json`) against
HyMem's Python SDK. The third leg of the benchmark triad: LME (single-shot
star topology), MSC (short genuine multi-session), LoCoMo (LONG genuine
multi-session — 19-32 sessions / 369-689 turns per conversation, real
timestamps, and an adversarial-question class neither of the others has).

DATA (verified 2026-07-28 against the actual snap-research locomo10.json):

    conversation = {
      "sample_id": "conv-26",
      "conversation": {
        "speaker_a": str, "speaker_b": str,
        "session_<N>": [ {"speaker": str, "dia_id": "D<N>:<t>", "text": str,
                          # optional image fields on photo-share turns:
                          "img_url": [str], "blip_caption": str, "query": str}, ...],
        "session_<N>_date_time": "1:56 pm on 8 May, 2023", ... },
      "qa": [ {"question": str, "answer": str|int, "evidence": ["D1:3", ...],
               "category": 1|2|3|4},
              {"question": str, "adversarial_answer": str, "evidence": [...],
               "category": 5}, ... ],
      "observation": ..., "session_summary": ..., "event_summary": ...,   # unused
    }

  Empirical quirks the loader must absorb (all present in the real file):
  `category` is sometimes a STRING ('5'); cat-5 `evidence` is a string REPR of a
  list ("['D2:3']"); answers can be int (6 rows) or absent (cat-5); 2 rows carry
  BOTH `answer` and `adversarial_answer`. Categories (paper §3):
  1=multi-hop (282), 2=temporal (321), 3=open-domain inference (96),
  4=single-hop (841), 5=adversarial (446; the same question as a cat-4 row but
  with the speaker/premise swapped — `adversarial_answer` is the TRAP answer,
  and the CORRECT behavior is to say the information isn't there).

CONTRACT DECISIONS (the MSC lesson ×3 — feeding parity, deixis, answerability —
restated up front instead of rediscovered one 15pp regression at a time):

  * Feeding parity: retrieval goes through `MSCAdapter.search` UNCHANGED (the
    LME-parity tier collection: top_k*3 at the pipeline layer, message-first
    ordering, additive profile tier, full pre-truncation pool for diagnostics).
  * Deixis: LoCoMo questions are THIRD-PERSON BY NAME ("What did Caroline
    research?") while memories carry [user]/[assistant] tags. A per-conversation
    perspective clause states the name↔role mapping — without it the reader
    cannot attribute facts, and cat-5 exists precisely to punish attribution
    swaps.
  * Answerability: cats 1-4 are answerable by construction, but cat-5's whole
    point is that abstention is CORRECT — so unlike MSC there is NO blanket
    answerability clause. The LME base prompt's abstention permission is
    load-bearing here (it is what a cat-5 pass looks like), and cat-5 is judged
    with the LME `_abs` abstention judge. `--answerable-clause` exists as an
    opt-in A/B lever but is LABEL-LEAKY (it conditions the prompt on the very
    thing cat-5 tests) and therefore non-canonical; the report brands runs
    that use it.
  * Label-routing only where LME itself does it: cat 2 → ability "TR" (the
    time-anchor stack: TR prompt + temporal_events chronology), cat 3 →
    permissive default prompt (the D4 posture — open-domain questions require
    world-knowledge bridging by construction; abstention guard kept).

Usage:
  python locomo_adapter.py --data data/locomo10.json --sample 200
  python locomo_adapter.py --data data/locomo10.json --db-dir /tmp/locomo_dbs \
         --workers 10                     # persistent per-conversation stores
  python locomo_adapter.py --sim          # offline: loader + mechanics, no API
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling benchmark imports

# MSCAdapter is reused WHOLESALE (open/ingest/dream/search) so LoCoMo inherits
# LME feeding parity from one shared implementation instead of a third copy.
# msc_adapter's module level is stdlib-only, so this import stays --sim-safe;
# longmemeval_adapter pieces are imported lazily inside functions, MSC-style.
from msc_adapter import (
    DEFAULT_INDEXING_MAX_CYCLES,
    DEFAULT_INDEXING_TIMEOUT_S,
    MSCAdapter,
    _indexing_limits,
    _lex_match,
    model_identity_fields as _msc_model_identity_fields,
    parse_extra_body_arg,
    prepare_indexing,
    run_or_record_indexing_failure,
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
    bounded_exception_type,
    build_manifest,
    code_hash,
    content_hash,
    embedding_usage_snapshot,
    effective_hymem_config_identity,
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
    sanitize_for_artifact,
    select_protocol_ids,
    strict_accuracy,
    usage_snapshot,
    validate_ids,
    write_latest_pointer,
)
from benchmarks.extraction_canary import (
    ExtractionCanaryError,
    extraction_canary_client_policy,
    extraction_canary_policy,
    print_extraction_canary,
    run_configured_extraction_canary,
    skipped_extraction_canary,
    validate_extraction_canary_config_binding,
    validate_extraction_canary_report,
)
from hymem.contrib.endpoint_policy import validate_http_endpoint
from hymem.contrib.model_policy import (
    DeprecatedModelAliasError,
    require_active_model,
)

_ANSWER_MODEL = "deepseek-v4-flash"
_JUDGE_MODEL = "deepseek-v4-flash"
_HYMEM_MODEL = "deepseek-v4-flash"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_MAX_CONVERSATION_ID_LENGTH = 128
_MAX_QUESTION_ID_LENGTH = 256
_SAFE_CONVERSATION_BASENAME_RE = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._-]{{0,{_MAX_CONVERSATION_ID_LENGTH - 1}}}\Z",
    re.ASCII,
)
_WINDOWS_RESERVED_BASENAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{n}" for n in range(1, 10)),
    *(f"LPT{n}" for n in range(1, 10)),
}


# ── category contract ───────────────────────────────────────────────────────

# question_type strings for compute_scores (the _abs suffix keeps the LME
# abstention machinery — compute_abstention_scores, judge routing — working
# untouched; compute_scores strips it for the per-category table).
CATEGORY_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                 4: "single-hop", 5: "adversarial_abs"}
# Which LME judge each category maps to. Cat 2 gets the temporal judge — but
# NOTE (verified 2026-07-29 against get_judge_prompt): its off-by-one tolerance
# covers DURATIONS only ("19 days when the answer is 18"), NOT calendar dates.
# LoCoMo cat-2 golds are mostly dates, so a one-day-off date IS scored wrong;
# do not adjudicate those as judge artifacts. Cat 5
# routes to the abstention judge via the _abs suffix inside judge_answer.
CATEGORY_JUDGE = {1: "multi-session", 2: "temporal-reasoning",
                  3: "single-session-user", 4: "single-session-user",
                  5: "single-session-user_abs"}


def locomo_perspective_clause(speaker_a: str, speaker_b: str,
                              user_is_a: bool = True) -> str:
    """The deixis contract, per conversation: questions name the speakers, the
    memories tag roles. Same bug class as MSC's perspective clause (which was
    worth +14pp there) — stated up front this time, not after a miss audit."""
    user, partner = (speaker_a, speaker_b) if user_is_a else (speaker_b, speaker_a)
    return (
        f"\nThe memories are turns from past conversations between two people: "
        f"{user} ([user] turns, and the 'Known user profile' facts) and {partner} "
        f"([assistant] turns). The question refers to them BY NAME. Attribute every "
        f"fact to the speaker whose turn actually said it: what {user} said, did, or "
        f"experienced comes from [user] turns; what {partner} said, did, or "
        f"experienced comes from [assistant] turns. Never transfer one speaker's "
        f"experience, plan, or statement to the other."
    )


# Cat-3 only. Style-routing with LME precedent (oracle SS-P → preference
# prompt); the abstention guard in the permissive prompt stays intact, so this
# does NOT leak answerability the way the opt-in clause below does.
LOCOMO_OPEN_DOMAIN_CLAUSE = (
    "\nThis question may ask for a likely inference or hypothetical rather than a "
    "directly stated fact. Combine what the memories establish about the speakers "
    "with general knowledge, and commit to the single best-supported inference. "
    "Keep the answer short."
)

# OPT-IN and LABEL-LEAKY: applying "this is answerable" only to cats 1-4 uses
# the per-question category label to defeat exactly what cat-5 measures. LME
# never leaks its _abs label into the answer prompt, so neither does the
# canonical LoCoMo posture. Kept as an explicit A/B lever (--answerable-clause)
# to size the abstention-miss cost the honest posture pays on cats 1-4.
LOCOMO_ANSWERABLE_CLAUSE = (
    "\nThis question has an answer stated in the memories — never reply that you "
    "don't have enough information. If no memory states the answer outright, commit "
    "to the single best-supported answer from what the memories do say, directly "
    "and without disclaimers."
)


# ── dataset loader ──────────────────────────────────────────────────────────

_DIA_ID_RE = re.compile(r"D\d+:\d+")
_SESSION_KEY_RE = re.compile(r"session_(\d+)$")
# "1:56 pm on 8 May, 2023" — the only format observed in locomo10.json; the
# alternates are defensive (strptime %p is case-insensitive, %d/%I accept
# unpadded values; %B needs an English-month locale, the C-locale default).
_DT_FORMATS = ("%I:%M %p on %d %B, %Y", "%H:%M on %d %B, %Y", "%d %B, %Y")


def _contains_control_characters(value: str) -> bool:
    # ``isprintable`` also catches Unicode format/control characters (for
    # example bidi overrides), not just the ASCII C0/C1 ranges.
    return any(not char.isprintable() for char in value)


def _validate_conversation_id(raw: object, *, index: int | str) -> str:
    """Validate the exact dataset id used as a persistent-store basename.

    IDs are deliberately not stripped, slugged, or otherwise normalized: two
    dataset records must never alias after sampling or at the filesystem
    boundary.  The portable ASCII policy preserves LoCoMo's existing
    ``conv-26``-style directory names while excluding path syntax and platform
    reserved basenames.
    """
    label = f"LoCoMo conversation id at index {index}"
    if not isinstance(raw, str) or not raw or not raw.strip():
        raise BenchmarkIntegrityError(f"{label} must be a non-empty string")
    if raw != raw.strip():
        raise BenchmarkIntegrityError(f"{label} must use its exact trimmed form")
    if len(raw) > _MAX_CONVERSATION_ID_LENGTH:
        raise BenchmarkIntegrityError(
            f"{label} exceeds {_MAX_CONVERSATION_ID_LENGTH} characters"
        )
    if _contains_control_characters(raw):
        raise BenchmarkIntegrityError(f"{label} contains control characters")
    if raw in {".", ".."} or Path(raw).is_absolute() or "/" in raw or "\\" in raw:
        raise BenchmarkIntegrityError(f"{label} is not a safe basename")
    if not _SAFE_CONVERSATION_BASENAME_RE.fullmatch(raw):
        raise BenchmarkIntegrityError(f"{label} is not a safe basename")
    windows_stem = raw.rstrip(" .").split(".", 1)[0].upper()
    if windows_stem in _WINDOWS_RESERVED_BASENAMES or raw.endswith((".", " ")):
        raise BenchmarkIntegrityError(f"{label} is not a portable safe basename")
    return raw


def _conversation_alias_key(value: str) -> str:
    """Portable filesystem identity (ASCII policy makes casefold sufficient)."""
    return value.casefold()


def _validate_question_id(raw: object, *, location: str) -> str:
    """Validate an exact result/checkpoint identifier without leaking its text."""
    label = f"LoCoMo question id at {location}"
    if not isinstance(raw, str) or not raw or not raw.strip():
        raise BenchmarkIntegrityError(f"{label} must be a non-empty string")
    if raw != raw.strip():
        raise BenchmarkIntegrityError(f"{label} must use its exact trimmed form")
    if len(raw) > _MAX_QUESTION_ID_LENGTH:
        raise BenchmarkIntegrityError(
            f"{label} exceeds {_MAX_QUESTION_ID_LENGTH} characters"
        )
    if _contains_control_characters(raw):
        raise BenchmarkIntegrityError(f"{label} contains control characters")
    return raw


def _raw_question_id(q: dict, sample_id: str, ci: int, qi: int) -> str:
    """Return a validated explicit id, or LoCoMo's stable generated id."""
    keys = [key for key in ("question_id", "qa_id") if key in q]
    location = f"conversation {ci}, question {qi}"
    if not keys:
        return _validate_question_id(f"{sample_id}_q{qi}", location=location)
    values = [
        _validate_question_id(q[key], location=location)
        for key in keys
    ]
    if len(values) == 2 and values[0] != values[1]:
        raise BenchmarkIntegrityError(
            f"LoCoMo question identifiers disagree at {location}"
        )
    return values[0]


def _validate_normalized_conversations(convs: list[dict]) -> None:
    """Fail closed on aliases before sampling, grouping, or store mutation."""
    seen_conversations: dict[str, int] = {}
    seen_questions: dict[str, tuple[int, int]] = {}
    for ci, conv in enumerate(convs):
        if not isinstance(conv, dict):
            raise BenchmarkIntegrityError(
                f"LoCoMo normalized conversation {ci} must be an object"
            )
        sample_id = _validate_conversation_id(conv.get("id"), index=ci)
        alias_key = _conversation_alias_key(sample_id)
        if alias_key in seen_conversations:
            raise BenchmarkIntegrityError(
                "duplicate LoCoMo conversation id or filesystem alias at indices "
                f"{seen_conversations[alias_key]} and {ci}"
            )
        seen_conversations[alias_key] = ci
        qa = conv.get("qa")
        if not isinstance(qa, list):
            raise BenchmarkIntegrityError(
                f"LoCoMo normalized conversation {ci} qa must be a list"
            )
        for qi, question in enumerate(qa):
            if not isinstance(question, dict):
                raise BenchmarkIntegrityError(
                    f"LoCoMo normalized question at conversation {ci}, "
                    f"question {qi} must be an object"
                )
            location = f"conversation {ci}, question {qi}"
            question_id = _validate_question_id(
                question.get("question_id"), location=location
            )
            qa_id = _validate_question_id(question.get("qa_id"), location=location)
            if question_id != qa_id:
                raise BenchmarkIntegrityError(
                    f"LoCoMo question identifiers disagree at {location}"
                )
            if question_id in seen_questions:
                first_ci, first_qi = seen_questions[question_id]
                raise BenchmarkIntegrityError(
                    "duplicate LoCoMo question id at conversation/question indices "
                    f"{first_ci}/{first_qi} and {ci}/{qi}"
                )
            seen_questions[question_id] = (ci, qi)


def resolve_locomo_store_root(db_dir: str | Path, conversation_id: object) -> Path:
    """Resolve a persistent store to one strict, non-symlink direct child.

    The returned path is safe to inspect or remove only because both the exact
    basename policy and resolved-parent equality hold.  In particular, it can
    never equal ``db_dir`` itself.
    """
    sample_id = _validate_conversation_id(conversation_id, index="evaluation")
    raw_base = Path(db_dir)
    raw_candidate = raw_base / sample_id
    try:
        base = raw_base.resolve(strict=False)
        if raw_candidate.is_symlink():
            raise BenchmarkIntegrityError(
                "LoCoMo store root must not be a symbolic link"
            )
        candidate = raw_candidate.resolve(strict=False)
    except BenchmarkIntegrityError:
        raise
    except (OSError, RuntimeError) as exc:
        raise BenchmarkIntegrityError(
            "LoCoMo store root could not be resolved safely"
        ) from exc
    if candidate == base or candidate.parent != base:
        raise BenchmarkIntegrityError(
            "LoCoMo store root is not a strict direct child of --db-dir"
        )
    return candidate


def _validate_store_database_path(root: Path) -> Path:
    """Keep the reusable SQLite artifact inside its already-safe store root."""
    db_path = root / "hymem.sqlite"
    try:
        if db_path.is_symlink():
            raise BenchmarkIntegrityError(
                "LoCoMo store database must not be a symbolic link"
            )
        resolved = db_path.resolve(strict=False)
    except BenchmarkIntegrityError:
        raise
    except (OSError, RuntimeError) as exc:
        raise BenchmarkIntegrityError(
            "LoCoMo store database could not be resolved safely"
        ) from exc
    if resolved.parent != root or resolved == root:
        raise BenchmarkIntegrityError(
            "LoCoMo store database escapes its conversation root"
        )
    return resolved


def _parse_session_dt(raw: str | None) -> datetime | None:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _turn_content(turn: dict) -> str:
    """Turn text, with the BLIP caption of a shared photo appended in-line —
    the standard text-only LoCoMo treatment; image content is otherwise lost."""
    text = (turn.get("text") or "").strip()
    cap = (turn.get("blip_caption") or "").strip()
    if cap:
        text = f"{text} [shared a photo: {cap}]".strip()
    return text


def _coerce_category(q: dict) -> int | None:
    try:
        return int(str(q.get("category")).strip())
    except (TypeError, ValueError):
        return None


def _coerce_evidence(q: dict) -> list[str]:
    """Evidence dia_ids. Regex over str() absorbs both real lists and the cat-5
    string-repr quirk ("['D2:3']")."""
    return _DIA_ID_RE.findall(str(q.get("evidence") or ""))


def _coerce_answer(q: dict) -> str | None:
    a = q.get("answer")
    if a is None:
        return None
    return str(a).strip() or None


def load_locomo_data(path: str | None, *, user_speaker: str = "a",
                     categories: set[int] | None = None,
                     name_prefix: bool = False) -> list[dict]:
    """Load + normalize LoCoMo conversations. Returns:
        {id, speaker_a, speaker_b, sessions: [[{role,content}]],
         session_dates: [str], n_sessions, evidence_map: {dia_id: (sess_idx,
         content)}, qa: [{qa_id, question, answer, adversarial_answer,
         category, qtype, judge_type, evidence}]}
    `sessions`/`session_dates`/`id` intentionally match the MSC normalized
    shape so `MSCAdapter.ingest` runs verbatim. Session dates come from the
    REAL `session_N_date_time` stamps (unlike MSC there is nothing to
    synthesize); an unparseable date falls back to previous + 1 day, ordering
    preserved. `path=None` returns the built-in --sim fixture."""
    raw = _SIM_FIXTURE if not path else json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LoCoMo data must be a JSON array of conversations")

    # Preflight the complete identifier namespace before normalizing any
    # conversation.  Sampling groups by conversation id and strict artifacts
    # key by question id, so accepting aliases here would silently merge rows
    # long before evaluate_conversation sees them.
    sample_ids: list[str] = []
    raw_question_ids: dict[tuple[int, int], str] = {}
    seen_conversations: dict[str, int] = {}
    seen_questions: dict[str, tuple[int, int]] = {}
    for ci, rec in enumerate(raw):
        if not isinstance(rec, dict):
            raise BenchmarkIntegrityError(
                f"LoCoMo conversation {ci} must be an object"
            )
        sample_id = _validate_conversation_id(rec.get("sample_id"), index=ci)
        alias_key = _conversation_alias_key(sample_id)
        if alias_key in seen_conversations:
            raise BenchmarkIntegrityError(
                "duplicate LoCoMo conversation id or filesystem alias at indices "
                f"{seen_conversations[alias_key]} and {ci}"
            )
        seen_conversations[alias_key] = ci
        sample_ids.append(sample_id)
        raw_qa = rec.get("qa") or []
        if not isinstance(raw_qa, list):
            raise BenchmarkIntegrityError(
                f"LoCoMo conversation {ci} qa must be a list"
            )
        for qi, question in enumerate(raw_qa):
            if not isinstance(question, dict):
                raise BenchmarkIntegrityError(
                    f"LoCoMo question at conversation {ci}, question {qi} "
                    "must be an object"
                )
            question_id = _raw_question_id(question, sample_id, ci, qi)
            if question_id in seen_questions:
                first_ci, first_qi = seen_questions[question_id]
                raise BenchmarkIntegrityError(
                    "duplicate LoCoMo question id at conversation/question indices "
                    f"{first_ci}/{first_qi} and {ci}/{qi}"
                )
            seen_questions[question_id] = (ci, qi)
            raw_question_ids[(ci, qi)] = question_id

    out = []
    for ci, rec in enumerate(raw):
        conv = rec.get("conversation") or {}
        if not isinstance(conv, dict):
            raise BenchmarkIntegrityError(
                f"LoCoMo conversation {ci} payload must be an object"
            )
        speaker_a = (conv.get("speaker_a") or "Speaker A").strip()
        speaker_b = (conv.get("speaker_b") or "Speaker B").strip()
        user_name = speaker_a if user_speaker == "a" else speaker_b

        sess_nums = sorted(int(m.group(1)) for k in conv
                           if (m := _SESSION_KEY_RE.fullmatch(k)))
        sessions, dates, evidence_map = [], [], {}
        cursor: datetime | None = None
        for n in sess_nums:
            turns_raw = conv.get(f"session_{n}") or []
            dt = _parse_session_dt(conv.get(f"session_{n}_date_time"))
            if dt is None:
                dt = (cursor + timedelta(days=1)) if cursor else datetime(2023, 5, 1)
            cursor = dt
            turns = []
            for t in turns_raw:
                content = _turn_content(t)
                if not content:
                    continue
                speaker = (t.get("speaker") or "").strip()
                role = "user" if speaker == user_name else "assistant"
                if name_prefix and speaker:
                    content = f"{speaker}: {content}"
                turns.append({"role": role, "content": content})
                dia = t.get("dia_id")
                if dia:
                    evidence_map[dia] = (len(sessions), content)
            if turns:
                sessions.append(turns)
                dates.append(dt.strftime("%Y-%m-%d %H:%M"))
        if not sessions:
            raise BenchmarkIntegrityError(
                f"LoCoMo conversation {ci} contains no usable sessions"
            )

        sample_id = sample_ids[ci]
        qa = []
        raw_qa = rec.get("qa") or []
        if not isinstance(raw_qa, list):
            raise BenchmarkIntegrityError(
                f"LoCoMo {sample_id} qa must be a list"
            )
        for qi, q in enumerate(raw_qa):
            if not isinstance(q, dict):
                raise BenchmarkIntegrityError(
                    f"LoCoMo {sample_id} question {qi} must be an object"
                )
            cat = _coerce_category(q)
            question = (q.get("question") or "").strip()
            if cat not in CATEGORY_NAME:
                raise BenchmarkIntegrityError(
                    f"LoCoMo {sample_id} question {qi} has invalid category"
                )
            if not question:
                raise BenchmarkIntegrityError(
                    f"LoCoMo {sample_id} question {qi} is empty"
                )
            if categories and cat not in categories:
                continue
            answer = _coerce_answer(q)
            adversarial = (q.get("adversarial_answer") or "").strip()
            if cat != 5 and answer is None:
                raise BenchmarkIntegrityError(
                    f"LoCoMo {sample_id} question {qi} has no answer"
                )
            if cat == 5 and not adversarial:
                raise BenchmarkIntegrityError(
                    f"LoCoMo {sample_id} adversarial question {qi} has no trap answer"
                )
            question_id = raw_question_ids[(ci, qi)]
            qa.append({
                "qa_id": question_id,
                "question_id": question_id,
                "question": question,
                "answer": answer,          # None on cat-5
                "adversarial_answer": adversarial,
                "category": cat,
                "qtype": CATEGORY_NAME[cat],
                "judge_type": CATEGORY_JUDGE[cat],
                "evidence": _coerce_evidence(q),
            })
        if not qa and not categories:
            raise BenchmarkIntegrityError(
                f"LoCoMo {sample_id} contains no usable questions"
            )
        out.append({
            "id": sample_id, "speaker_a": speaker_a, "speaker_b": speaker_b,
            "sessions": sessions, "session_dates": dates,
            "n_sessions": len(sessions), "evidence_map": evidence_map,
            "qa": qa,
        })
    _validate_normalized_conversations(out)
    return out


def sample_questions(convs: list[dict], sample: int, seed: int) -> list[dict]:
    """Global seeded QA sampling. Shuffles the (conversation, question) pool,
    keeps `sample` of them (0 = all), and drops conversations left with no
    questions — those are never ingested. Random-at-n≥100 approximates the
    category mix; use --categories for a targeted slice instead."""
    _validate_normalized_conversations(convs)
    rng = random.Random(seed)
    refs = [(c["id"], q) for c in convs for q in c["qa"]]
    rng.shuffle(refs)
    if sample:
        refs = refs[:sample]
    keep: dict[str, list[dict]] = defaultdict(list)
    for cid, q in refs:
        keep[cid].append(q)
    out = []
    for c in convs:
        if c["id"] in keep:
            # Restore stable in-file order inside each conversation so runs at
            # the same seed produce identical per-conversation eval order.
            c = dict(c, qa=sorted(keep[c["id"]], key=lambda q: q["qa_id"]))
            out.append(c)
    return out


def _select_questions_by_id(
    convs: list[dict], selected_ids: tuple[str, ...]
) -> list[dict]:
    """Project conversations onto an exact protocol id sequence.

    The source/within-conversation order is part of the checkpoint identity.
    This deliberately refuses to reorder rows to match a malformed receipt.
    """

    selected = set(selected_ids)
    out: list[dict] = []
    for conv in convs:
        questions = [
            question for question in conv["qa"]
            if question["question_id"] in selected
        ]
        if questions:
            out.append({**conv, "qa": questions})
    actual = tuple(
        question["question_id"]
        for conv in out for question in conv["qa"]
    )
    if actual != selected_ids:
        raise BenchmarkIntegrityError("selected LoCoMo id order drifted")
    return out


def _effective_pipeline_body(args) -> dict[str, Any]:
    """Mirror OpenAICompatibleClient's pure thinking-body decision."""

    from urllib.parse import urlsplit

    mode = str(args.hymem_thinking).strip().lower()
    if mode not in {"auto", "disabled", "off", "enabled"}:
        raise BenchmarkIntegrityError("memory-pipeline thinking mode is invalid")
    host = (urlsplit(args.hymem_base_url).hostname or "").casefold()
    send = mode == "disabled" or (
        mode == "auto"
        and ("deepseek" in host or "deepseek" in args.hymem_model.casefold())
    )
    return {"thinking": {"type": "disabled"}} if send else {}


def _judge_base_url(args, judge_llm=None) -> str:
    """Resolve legacy callers and actual clients to one validated endpoint.

    Older direct callers do not have the optional CLI field. A real supplied
    client's endpoint takes precedence over that historical default, but must
    agree with an explicitly configured endpoint so receipts cannot name a
    different provider. Rejudging validates this before making any calls.
    """
    configured = getattr(args, "judge_base_url", None)
    actual = getattr(judge_llm, "base_url", None)
    configured_url = (
        validate_http_endpoint(configured, label="judge").url
        if configured is not None else None
    )
    actual_url = (
        validate_http_endpoint(actual, label="judge").url
        if actual is not None else None
    )
    if configured_url and actual_url and configured_url != actual_url:
        raise BenchmarkIntegrityError("configured judge endpoint differs from client")
    return actual_url or configured_url or _DEEPSEEK_BASE_URL


def model_identity_fields(args, answer_llm, judge_llm, pipeline_llm) -> dict:
    """Preserve MSC's shared fields while binding LoCoMo's optional judge URL."""
    identity = _msc_model_identity_fields(
        args, answer_llm, judge_llm, pipeline_llm
    )
    if not getattr(args, "sim", False):
        identity["judge_base_url"] = _judge_base_url(args, judge_llm)
    return identity


def _effective_hymem_config(args):
    """Return the adapter's exact config object and public serialized identity."""
    from hymem import HyMemConfig

    overrides: dict[str, Any] = {
        **MSCAdapter.APERTURE,
        **{key: value for key, value in _aperture(args).items()
           if value is not None},
        # MSCAdapter pins aggregation off for the historical LoCoMo baseline.
        "aggregation_nodes_enabled": False,
    }
    if args.rules_extraction is not None:
        overrides["rules_extraction_enabled"] = args.rules_extraction
    if args.graph_multihop:
        overrides["graph_multihop_enabled"] = True
    if args.facts is not None:
        overrides["facts_enabled"] = args.facts
    if args.facts_extraction is not None:
        overrides["facts_extraction_enabled"] = args.facts_extraction
    cfg = HyMemConfig(root=Path("/benchmark-identity"), **overrides)
    effective_cfg = effective_hymem_config_identity(cfg)
    # This is a boolean feature switch, not a credential.  Rename it before
    # strict artifact sanitization so its exact score-affecting value survives
    # the generic secret-key scrubber.
    effective_cfg["content_redaction_enabled"] = effective_cfg.pop(
        "redact_secrets"
    )
    return cfg, effective_cfg


def _strict_identity(args) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return score-affecting config and provider identities without clients.

    This function is intentionally pure with respect to benchmark state: it
    may validate local configuration, but creates no store, provider transport,
    or canary.  Resume mismatches can therefore fail before any paid work.
    """

    from longmemeval_adapter import (
        _provider_for_url,
        resolve_embedding_identity,
    )

    cfg, effective_cfg = _effective_hymem_config(args)

    # LoCoMo inherits MSC's environment-selected embedding transport. Resolve
    # exactly that public vector-space identity, never its credential.  The
    # simulation branch creates no embedding client and is rejected if the
    # operator asks for one, so its manifested identity is always disabled.
    embedding_args = SimpleNamespace(
        embeddings=bool(args.embeddings and not args.sim),
        embedding_base_url=None,
        embedding_model=None,
        embedding_dim=None,
    )
    embedding = resolve_embedding_identity(embedding_args)
    if args.sim:
        models = {
            "reader": {
                "configured": False, "client_class": None,
                "provider": "none", "model": None, "base_url": None,
            },
            "judge": {
                "configured": False, "client_class": None,
                "provider": "none", "model": None, "base_url": None,
            },
            "memory_pipeline": {
                "configured": True,
                "client_class": "hymem.extraction.llm.StubLLMClient",
                "provider": "local_stub", "model": None, "base_url": None,
                "response_policy": "constant-empty-json-list-v1",
            },
            "embedding": embedding,
        }
    else:
        judge_base_url = _judge_base_url(args)
        models = {
            "reader": {
                "client_class": "longmemeval_adapter.LLMClient",
                "provider": _provider_for_url(args.answer_base_url),
                "model": args.answer_model,
                "base_url": args.answer_base_url,
                "temperature": 0.0,
                "max_tokens": 1024,
                "extra_body": copy.deepcopy(args.answer_extra_body_obj),
            },
            "judge": {
                "client_class": "longmemeval_adapter.LLMClient",
                "provider": _provider_for_url(judge_base_url),
                "model": args.judge_model,
                "base_url": judge_base_url,
                "temperature": 0.0,
                "max_tokens": 10,
                "extra_body": copy.deepcopy(args.judge_extra_body_obj),
                "protocol": "longmemeval-local-judge",
            },
            "memory_pipeline": {
                "client_class": (
                    "hymem.contrib.openai_client.OpenAICompatibleClient"
                ),
                "provider": _provider_for_url(args.hymem_base_url),
                "model": args.hymem_model,
                "base_url": args.hymem_base_url,
                "thinking_mode": args.hymem_thinking,
                "effective_extra_body": _effective_pipeline_body(args),
            },
            "embedding": embedding,
        }
    category_steering = bool(getattr(args, "category_steering", False))
    subset_run = bool(args.sample)
    config = {
        "sample": args.sample,
        "sample_strategy": (
            "seeded-shuffle-then-source-conversation-order-v1"
            if subset_run else "all-source-order"
        ),
        "seed": args.seed,
        "categories": (
            sorted(int(item) for item in args.categories.split(","))
            if args.categories else None
        ),
        "conversations": (
            [item.strip() for item in args.convs.split(",")]
            if args.convs else None
        ),
        "workers": args.workers,
        "top_k": args.top_k,
        "max_context_chars": args.max_context_chars,
        "user_speaker": args.user_speaker,
        "name_prefix": bool(args.name_prefix),
        "answerable_clause": bool(args.answerable_clause),
        "category_steering": category_steering,
        "embeddings": bool(args.embeddings),
        "rules_extraction": args.rules_extraction,
        "facts": args.facts,
        "facts_extraction": args.facts_extraction,
        "graph_multihop": bool(args.graph_multihop),
        "no_dream": bool(args.no_dream),
        "dream_per_session": bool(args.dream_per_session),
        "indexing_max_cycles": args.indexing_max_cycles,
        "indexing_timeout_s": float(args.indexing_timeout_s),
        "fresh_store": bool(args.fresh),
        "persistent_store": bool(args.db_dir),
        "dump_context": bool(args.dump_context),
        "dump_topk": bool(args.dump_topk),
        "sim": bool(args.sim),
        "effective_hymem_config": effective_cfg,
        "extraction_canary": extraction_canary_policy(
            prompt_version=cfg.prompt_version
        ),
        "label_free_answer_path": not category_steering,
        "scored_run": not args.sim,
        "exploratory_label_steering": category_steering,
        "exploratory_non_comparable": bool(
            subset_run or args.sim or args.no_dream
            or args.answerable_clause or category_steering
        ),
    }
    validate_extraction_canary_config_binding(
        config["extraction_canary"], effective_cfg
    )
    return config, models


def locomo_code_hash(
    *,
    adapter_path: Path | None = None,
    strictness_path: Path | None = None,
    archive_evidence_path: Path | None = None,
    msc_adapter_path: Path | None = None,
    lme_adapter_path: Path | None = None,
    lme_protocol_path: Path | None = None,
    extraction_canary_path: Path | None = None,
    store_attestation_path: Path | None = None,
    hymem_path: Path | None = None,
    root: Path | None = None,
) -> str:
    """Hash exact direct and transitive executable LoCoMo dependencies."""

    root_path = Path(root or _repo_root).resolve()
    benchmark_dir = Path(__file__).resolve().parent
    adapter = Path(adapter_path or __file__)
    msc_adapter = Path(msc_adapter_path or benchmark_dir / "msc_adapter.py")
    lme_adapter = Path(
        lme_adapter_path or benchmark_dir / "longmemeval_adapter.py"
    )
    msc_symbols = python_file_imported_symbols(
        adapter,
        module_names=("benchmarks.msc_adapter", "msc_adapter"),
    )
    dependency_slices: list[PythonSourceSlice] = []
    lme_symbols = set(python_file_imported_symbols(
        adapter,
        module_names=("benchmarks.longmemeval_adapter", "longmemeval_adapter"),
    ))
    msc_slice: PythonSourceSlice | None = None
    if msc_symbols:
        msc_slice = PythonSourceSlice(msc_adapter, msc_symbols)
        dependency_slices.append(msc_slice)
        lme_symbols.update(python_slice_imported_symbols(
            msc_slice,
            module_names=(
                "benchmarks.longmemeval_adapter", "longmemeval_adapter",
            ),
        ))
    if lme_symbols:
        lme_slice = PythonSourceSlice(lme_adapter, tuple(lme_symbols))
        dependency_slices.append(lme_slice)
        protocol_symbols = python_slice_imported_symbols(
            lme_slice,
            module_names=("benchmarks.lme_protocol", "lme_protocol"),
        )
        if protocol_symbols:
            dependency_slices.append(PythonSourceSlice(
                Path(lme_protocol_path or benchmark_dir / "lme_protocol.py"),
                protocol_symbols,
            ))
    canary = Path(extraction_canary_path or benchmark_dir / "extraction_canary.py")
    canary_symbols = python_file_imported_symbols(
        adapter,
        module_names=("benchmarks.extraction_canary", "extraction_canary"),
    )
    if canary_symbols:
        dependency_slices.append(PythonSourceSlice(canary, canary_symbols))
    store_symbols: set[str] = set(python_file_imported_symbols(
        adapter,
        module_names=("benchmarks.store_attestation", "store_attestation"),
    ))
    if msc_slice is not None:
        store_symbols.update(python_slice_imported_symbols(
            msc_slice,
            module_names=("benchmarks.store_attestation", "store_attestation"),
        ))
    if store_symbols:
        dependency_slices.append(PythonSourceSlice(
            Path(store_attestation_path or benchmark_dir / "store_attestation.py"),
            tuple(store_symbols),
        ))
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
        raise BenchmarkIntegrityError("LoCoMo code identity lacks strictness imports")
    dependency_slices.append(PythonSourceSlice(
        strictness, tuple(strictness_symbols)
    ))
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
    return code_hash(inputs, root=root_path)


# ── per-question evaluation ─────────────────────────────────────────────────

def _evidence_diagnostics(q: dict, conv: dict, context_texts: list[str],
                          pool_texts: list[str], rendered: str | None = None) -> dict:
    """Step-0 diagnostics from LoCoMo's OWN evidence annotations (stronger than
    MSC's answer-text heuristic — the gold turn text is known exactly, the
    lexical τ=0.6 only absorbs the 600-char context truncation).

    FOUR nested surfaces, because there are FOUR places evidence can die and
    conflating the last two mislabels a truncation loss as a reader failure:
      gold_in_pool     — all evidence turns retrievable pre-truncation
      gold_in_topk     — survived the `memories[:top_k]` retrieval cut
      gold_in_render   — survived `_render_answer_context`'s MAX_CONTEXT_CHARS
                         budget, i.e. ACTUALLY REACHED THE READER. The renderer
                         `break`s at the first item that overflows, so at wide
                         apertures the top_k list is much larger than the text
                         the model sees; `gold_in_topk` alone silently credits
                         evidence the reader never got.
      gold_in_context  — alias of gold_in_render when a rendered context is
                         supplied (the honest definition), else gold_in_topk.
    `gold_distance` is sessions back from the last session to the FARTHEST-BACK
    evidence turn (1 = final session; -1 = no locatable evidence).

    On cat-5 the evidence points at the TRAP-SOURCE turn, not a gold answer —
    recorded for completeness, excluded from the miss decomposition."""
    ev = [(conv["evidence_map"][e]) for e in q["evidence"] if e in conv["evidence_map"]]
    if not ev:
        return {"gold_in_context": False, "gold_in_pool": False,
                "gold_in_topk": False, "gold_in_render": False,
                "gold_distance": -1, "evidence_in_context_frac": 0.0,
                "n_evidence": 0}
    joined_topk = " ".join(context_texts)
    joined_pool = joined_topk + " " + " ".join(pool_texts)
    in_topk = [_lex_match(text, joined_topk, tau=0.6) for _, text in ev]
    in_pool = [_lex_match(text, joined_pool, tau=0.6) for _, text in ev]
    if rendered is None:
        in_render = in_topk
    else:
        in_render = [_lex_match(text, rendered, tau=0.6) for _, text in ev]
    return {
        "gold_in_context": all(in_render),
        "gold_in_render": all(in_render),
        "gold_in_topk": all(in_topk),
        "gold_in_pool": all(in_pool),
        "gold_distance": conv["n_sessions"] - min(idx for idx, _ in ev),
        "evidence_in_context_frac": sum(in_render) / len(in_render),
        "n_evidence": len(ev),
    }


# The cat-5 judge takes an EXPLANATION of unanswerability, not a gold answer.
# Factored out of evaluate_qa so --rejudge can rebuild the identical judge input
# from a stored results file — a re-judge that reconstructed this differently
# would measure prompt drift, not judge nondeterminism.
def _gold_for_judge(cat: int, answer, trap) -> str:
    if cat != 5:
        return answer if answer is not None else ""
    s = ("The conversation never establishes this — the question's "
         "premise is false or the asked detail was never mentioned. ")
    if trap:
        s += (f"A tempting but WRONG answer (it belongs to a different "
              f"speaker or event) would be: '{trap}'. "
              f"A response giving that answer is incorrect.")
    return s


def evaluate_qa(q: dict, conv: dict, adapter: MSCAdapter, args,
                answer_llm, judge_llm) -> dict:
    cat = q["category"]
    from longmemeval_adapter import _detect_ability, _detect_ability_safe

    category_steering = bool(getattr(args, "category_steering", False))
    detected_ability = (
        _detect_ability_safe(q["question"])
        if category_steering else _detect_ability(q["question"])
    )
    oracle_ability = "TR" if cat == 2 else None
    ability = oracle_ability if category_steering else detected_ability
    # top_k * 3 at the pipeline layer — the LME driver's multiplier, inherited
    # via MSCAdapter.search (the silently-dropped ×3 was the entire BEAM June
    # regression AND the first 23pp of the MSC arc; never again).
    memories, info = adapter.search(q["question"], top_k=args.top_k * 3)

    # Re-render the EXACT context the answerer will build (same helper, same
    # char caps — cat-2 routes to ability="TR", which doubles the budget) so
    # gold-surface is measured against what the reader actually receives, not
    # against the pre-render top_k list. Pure string work, no LLM call.
    extra = locomo_perspective_clause(conv["speaker_a"], conv["speaker_b"],
                                      user_is_a=(args.user_speaker == "a"))
    if category_steering and cat == 3:
        extra += LOCOMO_OPEN_DOMAIN_CLAUSE
    if args.answerable_clause and cat != 5:
        extra += LOCOMO_ANSWERABLE_CLAUSE

    rendered = None
    messages = None
    if not args.sim:
        from longmemeval_adapter import build_answer_messages
        question_date = conv["session_dates"][-1] if conv["session_dates"] else ""
        messages = build_answer_messages(
            memories, q["question"], ability=ability,
            total_matches=info["total_matches"], graph_count=info["graph_count"],
            temporal_events=info["temporal_events"],
            aggregation_nodes=info["aggregation_nodes"],
            narrative_facts=info["narrative_facts"],
            question_date=question_date,
            permissive_default=(category_steering and cat == 3),
            extra_system=extra,
            max_input_tokens=getattr(args, "max_input_tokens", 16000),
            token_counter=(
                getattr(answer_llm, "count_tokens", None)
                if answer_llm is not None else None
            ),
        )
        user_text = messages[1]["content"]
        rendered = user_text.split("CONTEXT:\n", 1)[-1].rsplit(
            "\n\nQUESTION:", 1
        )[0]

    diag = _evidence_diagnostics(q, conv, [m["content"] for m in memories],
                                 info["pool"], rendered=rendered)

    judge_raw = ""            # only the judge branch below can set this
    benchmark_failure = None
    if args.diag_only:
        # Retrieval + render only — no reader, no judge. `correct` stays None
        # because this pass CANNOT produce accuracy; locomo_audit.py joins the
        # dumped surfaces onto a real run by question id. Retrieval is
        # deterministic given the same store, so the top_k reproduces exactly.
        ai, correct = "", None
    elif args.sim:
        # Offline: no answer/judge LLM. "correct" = retrieval surfaced every
        # evidence turn — a retrieval-surface rate, NOT benchmark accuracy
        # (on cat-5 it reports whether the trap-source turn surfaces).
        ai, correct = (memories[0]["content"] if memories else ""), diag["gold_in_context"]
    else:
        from longmemeval_adapter import judge_scored
        ai = answer_llm.chat(messages, temperature=0.0, max_tokens=1024)
        if (ai or "").startswith("[LLM_ERROR"):
            correct = False
            benchmark_failure = "reader_transport_or_content_failure"
        else:
            gold_for_judge = _gold_for_judge(cat, q["answer"] or "",
                                             q["adversarial_answer"])
            correct, judge_raw = judge_scored(judge_llm, q["judge_type"],
                                              q["question"], gold_for_judge, ai)
            if correct is None:
                benchmark_failure = "judge_transport_or_parse_failure"

    rec = {"id": q["qa_id"], "question_id": q["question_id"],
           "conv_id": conv["id"], "question_type": q["qtype"],
           "category": cat,
           "correct": (None if correct is None else bool(correct)),
           "judge_raw": judge_raw,
           # A judge error, NOT the --diag-only/--sim `correct=None`: those two
           # never call a judge, so their None means "no verdict exists" rather
           # than "the judge failed". Conflating them would report a run that
           # deliberately measured nothing as an outage.
           "judge_error": bool(judge_raw) and correct is None,
           "benchmark_failure": benchmark_failure,
           "oracle_ability": oracle_ability,
           "detected_ability": detected_ability,
           "ability_used": ability,
           "question": q["question"],
           "answer": q["answer"] if cat != 5 else f"[unanswerable; trap: {q['adversarial_answer']}]",
           "ai_answer": ai, "n_sessions": conv["n_sessions"],
           "evidence": q["evidence"], **diag,
           "n_memories": len(memories), "n_profile": info["n_profile"],
           # Rendered lines carry a [MEM …]/[FACT] tag; counting them measures
           # how many retrieved memories survived the char budget.
           "n_rendered": (None if rendered is None
                          else rendered.count("[MEM") + rendered.count("[FACT")),
           # E1 mechanism read, to be taken BEFORE the score: a run of zeros
           # means the tier never reached the reader, so a flat all-800 net is
           # a no-op by construction, not a null result.
           "n_facts": len(info["narrative_facts"])}
    if (args.dump_context or args.diag_only) and rendered is not None:
        rec["context"] = rendered
    if args.dump_topk or args.diag_only:
        # EXACTLY the string _evidence_diagnostics scores gold_in_topk against,
        # so a strict re-check in the audit runs on the identical haystack.
        # Dumping both this and `context` is the whole point: the surfaces are
        # NESTED (render ⊆ top_k), so gold_in_context=True forces
        # gold_in_topk=True and the boolean pair can never separate a
        # composition loss from a recall loss. Only re-scoring both strings at a
        # strict τ can.
        rec["topk_text"] = " ".join(m["content"] for m in memories)
        # THE PRE-CUT POOL. `memories` is already cut to top_k*3, so topk_text
        # alone cannot tell "never retrieved" from "retrieved, then lost the
        # ranking competition" — and those two pick DIFFERENT levers (indexing
        # /aperture vs matching quality). Widening --message-fts-top-k while the
        # cut stays fixed only moves the second kind, so the pool surface is the
        # one that makes a sweep interpretable.
        rec["pool_text"] = " ".join(info["pool"])
    return rec


# ── per-conversation driver ─────────────────────────────────────────────────

# Set from --max-context-chars in main(); read by the report line. A list so
# the worker threads and the report share one cell without a global statement.
_MAX_CTX: list[int | None] = [None]


def _aperture(args) -> dict:
    """Lever-L6 retrieval-aperture overrides; None means 'keep the default'."""
    return {"message_fts_top_k": args.message_fts_top_k,
            "rerank_top_k": args.rerank_top_k,
            "fts_top_k": args.fts_top_k,
            "graph_top_k": args.graph_top_k}


class _ParallelConversationStopped(BaseException):
    """Internal cooperative cancellation; never convert this into QA rows."""


def evaluate_conversation(
    conv: dict, args, answer_llm, judge_llm, *,
    pending_ids: set[str] | None = None, on_result=None, on_checkpoint=None,
    _parallel_stop: threading.Event | None = None, _on_fatal_abort=None,
    _on_attempt=None, _on_runtime_snapshot=None,
) -> list[dict]:
    """Ingest one conversation into its own store, then answer its questions.
    With --db-dir the store persists and is REUSED on later runs (ingest+dream
    over 19-32 sessions is the expensive step; QA/prompt iterations shouldn't
    re-pay it). A reused store is only valid for the same core/schema —
    --fresh rebuilds after core changes."""
    def _raise_if_parallel_stopped() -> None:
        if _parallel_stop is not None and _parallel_stop.is_set():
            raise _ParallelConversationStopped()

    def _notify_fatal_abort(exc: BaseException) -> None:
        if _on_fatal_abort is None or isinstance(
            exc, _ParallelConversationStopped
        ):
            return
        if not isinstance(exc, Exception) or is_structural_benchmark_error(exc):
            _on_fatal_abort(exc)

    # This must precede mkdtemp, exists(), reuse detection, mkdir(), and above
    # all --fresh rmtree().  evaluate_conversation is also a public test/probe
    # seam, so it cannot rely only on load_locomo_data having run in main().
    _raise_if_parallel_stopped()
    _validate_normalized_conversations([conv])
    if args.db_dir:
        root = resolve_locomo_store_root(args.db_dir, conv["id"])
        # Validate a pre-existing database link before --fresh, too: deleting
        # that store may be safe on today's shutil implementation, but a
        # benchmark boundary should not depend on platform rmtree semantics.
        _validate_store_database_path(root)
        if args.fresh and root.exists():
            shutil.rmtree(root)
        # Re-resolve after a deletion and after creation so a swapped symlink
        # cannot silently turn later reuse/open operations into an escape.
        root = resolve_locomo_store_root(args.db_dir, conv["id"])
        root.mkdir(parents=True, exist_ok=True)
        root = resolve_locomo_store_root(args.db_dir, conv["id"])
        db_path = _validate_store_database_path(root)
        reuse = not args.fresh and db_path.exists()
        cleanup = False
    else:
        root = Path(tempfile.mkdtemp(prefix=f"locomo_{conv['id']}_"))
        reuse, cleanup = False, not args.keep_db
        db_path = root / "hymem.sqlite"

    adapter = MSCAdapter(db_path, api_key=args.api_key, sim=args.sim,
                         hymem_model=args.hymem_model, hymem_base_url=args.hymem_base_url,
                         hymem_thinking=args.hymem_thinking,
                         embeddings=args.embeddings, rules_extraction=args.rules_extraction,
                         graph_multihop=args.graph_multihop,
                         facts_enabled=args.facts,
                         facts_extraction=args.facts_extraction,
                         aperture=_aperture(args))
    indexing: dict[str, Any] | None = None

    def _conversation_runtime_snapshot() -> dict[str, Any]:
        return {
            "scope_id": f"locomo:{conv['id']}",
            "indexing": dict(indexing) if isinstance(indexing, dict) else None,
            "memory_pipeline_usage": usage_snapshot(
                getattr(adapter, "pipeline_llm", None)
            ),
            "embedding_usage": embedding_usage_snapshot(
                getattr(adapter, "embedding_client", None),
                configured=bool(args.embeddings),
            ),
        }

    try:
        adapter.open()
        _raise_if_parallel_stopped()
        if reuse:
            print(f"  [{conv['id']}] reusing store at {root}", flush=True)
        indexing = prepare_indexing(
            adapter,
            conv,
            args,
            scope_id=f"locomo:{conv['id']}",
            reuse=reuse,
        )
        results = []
        for k, q in enumerate(conv["qa"], 1):
            # A conversation can contain many provider-backed QA items.  Stop
            # an already-running sibling at the next safe item boundary.
            _raise_if_parallel_stopped()
            if pending_ids is not None and q["question_id"] not in pending_ids:
                continue
            # Count the attempt before provider-backed QA begins.  A sibling
            # checkpoint fault can cancel this conversation before its row is
            # publishable, but the work (and its spend) still belongs to the
            # execution segment that is drained during the abort.
            if _on_attempt is not None:
                _on_attempt(q["question_id"])
            try:
                row = evaluate_qa(q, conv, adapter, args, answer_llm, judge_llm)
            except Exception as exc:
                if is_structural_benchmark_error(exc):
                    _notify_fatal_abort(exc)
                    raise
                row = {
                    "id": q["qa_id"], "question_id": q["question_id"],
                    "conv_id": conv["id"], "question_type": q["qtype"],
                    "category": q["category"], "question": q["question"],
                    "correct": False, "judge_raw": "", "judge_error": False,
                    "benchmark_failure": (
                        f"execution_failure:{bounded_exception_type(exc)}"
                    ),
                }
            if args.sim:
                row.update({
                    "answer_model": None,
                    "answer_base_url": None,
                    "answer_extra_body": None,
                    "judge_model": None,
                    "judge_base_url": None,
                    "judge_extra_body": None,
                    "hymem_model": None,
                    "hymem_base_url": None,
                    "hymem_thinking": None,
                    "hymem_extra_body": {},
                    "hymem_client_class": (
                        "hymem.extraction.llm.StubLLMClient"
                    ),
                })
            else:
                row.update(model_identity_fields(
                    args, answer_llm, judge_llm, adapter.pipeline_llm
                ))
            benchmark_non_comparable = (
                indexing.get("skip_reason")
                or ("diagnostics_only" if args.diag_only else None)
                or (
                    "label_leaky_answerable_clause"
                    if getattr(args, "answerable_clause", False) else None
                )
            )
            row.update({
                # LoCoMo artifacts are bare row lists, not config envelopes.
                # Record the effective bodies (after automatic DeepSeek-v4
                # resolution), never the raw CLI None that did not hit the wire.
                "model_identity_recorded": True,
                "indexing_scope_id": indexing["scope_id"],
                "indexing_complete": bool(indexing["complete"]),
                "indexing_healthy": bool(indexing["healthy"]),
                "indexing_comparable": bool(indexing["comparable"]),
                "benchmark_comparable": bool(
                    indexing["comparable"] and benchmark_non_comparable is None
                ),
                "non_comparable_reason": benchmark_non_comparable,
            })
            # Usage is cumulative per pipeline client, so persist exactly one
            # owning receipt per conversation. Other QA rows carry a stable
            # reference instead of multiplying the same calls/tokens/cost.
            if not results:
                row["indexing"] = indexing
            else:
                row["indexing_ref"] = indexing["scope_id"]
            results.append(row)
            if on_checkpoint is not None:
                # The strict runner persists the row and the cumulative
                # conversation-local provider snapshots in one checkpoint
                # replacement.  A crash immediately after this callback can
                # therefore neither lose the score nor silently lose the spend
                # needed to produce it.  ``on_result`` remains the legacy
                # one-argument seam used by probes and older callers.
                on_checkpoint(row, {
                    **_conversation_runtime_snapshot(),
                })
            elif on_result is not None:
                on_result(row)
            if k % 20 == 0:
                print(f"  [{conv['id']}] {k}/{len(conv['qa'])}", flush=True)
        # --diag-only writes correct=None (no reader ran), so there is no accuracy
        # to report here — summing it would TypeError, and printing 0.0% would be
        # worse: a run that measured nothing would look like a run that scored zero.
        if args.diag_only:
            surf = sum(bool(r["gold_in_render"]) for r in results) / len(results) if results else 0.0
            print(f"  [{conv['id']}] done — {len(results)} q, gold_in_render "
                  f"{surf*100:.1f}% (tau=0.6, no reader)"
                  f" ({conv['n_sessions']} sessions)", flush=True)
        else:
            _scored = [r for r in results if r.get("correct") is not None]
            acc = sum(r["correct"] for r in _scored) / len(_scored) if _scored else 0.0
            _unscored = len(results) - len(_scored)
            print(f"  [{conv['id']}] done — {len(results)} q"
                  + (f" ({_unscored} UNSCORED — judge error)" if _unscored else "")
                  + f", {acc*100:.1f}%"
                  f" ({conv['n_sessions']} sessions)", flush=True)
        return results
    except BaseException as exc:
        # Notify before adapter/temp-store cleanup so sibling workers cannot
        # start another item while this structural/control failure unwinds.
        _notify_fatal_abort(exc)
        raise
    finally:
        cleanup_actions = []
        if _on_runtime_snapshot is not None:
            # This conversation-keyed handoff runs before adapter teardown.
            # The runner overwrites its earlier per-row snapshot for the same
            # scope, so a stopped peer contributes final usage without ever
            # publishing its aborted row.
            cleanup_actions.append((
                "runtime_usage_handoff",
                lambda: _on_runtime_snapshot(
                    _conversation_runtime_snapshot()
                ),
            ))
        cleanup_actions.append(("adapter_close", adapter.close))
        if cleanup:
            cleanup_actions.append((
                "temporary_store_cleanup",
                lambda: shutil.rmtree(root, ignore_errors=False),
            ))
        run_cleanup_actions(
            cleanup_actions, primary_exception=sys.exc_info()[1],
        )


# ── reporting ───────────────────────────────────────────────────────────────

_DIST_BUCKETS = [(1, 1, "1 back"), (2, 3, "2-3 back"), (4, 7, "4-7 back"),
                 (8, 15, "8-15 back"), (16, 10 ** 9, "16+ back")]


def _compute_scores_local(results: list[dict]) -> dict:
    """Fallback mirror of longmemeval_adapter.compute_scores (accuracy by
    question_type, _abs folded into the base type) so --sim runs with zero API
    deps — importing the LME module pulls in `requests` at module level."""
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        verdict = r.get("correct")
        if verdict is not None and not isinstance(verdict, bool):
            raise BenchmarkIntegrityError("LoCoMo result has malformed verdict")
        qtype = str(r.get("question_type") or "unknown").replace("_abs", "")
        by_type[qtype].append(bool(verdict))
    scores = {t: {"accuracy": sum(c) / len(c), "count": len(c)}
              for t, c in by_type.items()}
    all_c = [c for cs in by_type.values() for c in cs]
    scores["OVERALL"] = {"accuracy": sum(all_c) / len(all_c) if all_c else 0.0,
                         "count": len(all_c)}
    return scores


def _print_report(results: list[dict], args) -> None:
    try:
        from longmemeval_adapter import (compute_scores, compute_abstention_scores,
                                         judge_error_note, print_abstention_scores)
    except ImportError:  # offline --sim without the LME module's HTTP deps
        compute_scores = _compute_scores_local
        compute_abstention_scores = print_abstention_scores = None
        judge_error_note = None
    # Strict headline scores retain judge/reader/execution failures as wrong.
    # A separately named conditional count is printed for diagnosis only.
    if judge_error_note:
        print(f"\n  {judge_error_note(results)}")
    scores = compute_scores(results)
    ov = scores.pop("OVERALL")
    print(f"\n=== LoCoMo — n={ov['count']} ===")
    if args.sim:
        print("  [SIM] 'accuracy' below = retrieval-surface rate, not benchmark accuracy")
    if args.answerable_clause:
        print("  [NON-CANONICAL] --answerable-clause is label-leaky "
              "(conditions the prompt on the adversarial label)")
    print(f"  overall accuracy: {ov['accuracy']*100:.1f}%")
    valid_rows = [r for r in results if not r.get("benchmark_failure")]
    print(f"  conditional valid-only: n={len(valid_rows)} "
          f"(strict denominator n={len(results)}, "
          f"failures={len(results) - len(valid_rows)})")
    # Stamp the aperture: a run is only comparable to another at the SAME one,
    # and message_fts_top_k is the hard ceiling on gold-turn surfacing.
    ap_eff = {**MSCAdapter.APERTURE,
              **{k: v for k, v in _aperture(args).items() if v is not None}}
    print(f"  aperture: msg={ap_eff['message_fts_top_k']} "
          f"rerank_pool={ap_eff.get('rerank_top_k', 20)} "
          f"chunk={ap_eff['fts_top_k']} graph={ap_eff['graph_top_k']} "
          f"cut={args.top_k * 3}  embeddings={'on' if args.embeddings else 'off'}\n")
    print(f"  {'category':<16} {'acc':>7} {'n':>6}")
    for name, s in sorted(scores.items()):
        print(f"  {name:<16} {s['accuracy']*100:>6.1f}% {s['count']:>6}")

    if print_abstention_scores and any(r["question_type"].endswith("_abs")
                                       for r in results):
        print_abstention_scores(compute_abstention_scores(results))

    # E1 analogue: accuracy vs sessions-back to the farthest evidence turn,
    # bucketed (LoCoMo distances run 1..32). Answerable cats only — cat-5
    # "evidence" is the trap source, not a gold location.
    answerable = [
        r for r in results
        if r["category"] != 5 and not r.get("benchmark_failure")
    ]
    if answerable:
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in answerable:
            d = r.get("gold_distance", -1)
            label = "unknown" if d < 0 else next(
                lb for lo, hi, lb in _DIST_BUCKETS if lo <= d <= hi)
            by_bucket[label].append(r)
        order = ["unknown"] + [lb for _, _, lb in _DIST_BUCKETS]
        print("\n  ── recall vs session distance (+ gold-surface diagnostics; "
              "evidence-based, lexical τ=0.6) ──")
        print(f"  {'distance':>9} {'acc':>7} {'in-ctx':>7} {'in-pool':>8} {'n':>5}")
        for label in order:
            rs = by_bucket.get(label)
            if not rs:
                continue
            acc = sum(r["correct"] for r in rs) / len(rs)
            ctx = sum(r["gold_in_context"] for r in rs) / len(rs)
            pool = sum(r["gold_in_pool"] for r in rs) / len(rs)
            print(f"  {label:>9} {acc*100:>6.1f}% {ctx*100:>6.1f}% {pool*100:>7.1f}% {len(rs):>5}")

        misses = [r for r in answerable if not r["correct"]]
        if misses:
            # Four buckets, not three: `budget` is evidence that won the
            # retrieval cut and was then dropped by MAX_CONTEXT_CHARS before the
            # reader saw it. It used to be counted as synthesis, which reads as
            # "the reader failed" when in fact the reader was never shown it —
            # and that mislabel grows with the aperture.
            retrieval = sum(not r["gold_in_pool"] for r in misses)
            ranking = sum(r["gold_in_pool"] and not r.get("gold_in_topk", r["gold_in_context"])
                          for r in misses)
            budget = sum(r.get("gold_in_topk", r["gold_in_context"])
                         and not r["gold_in_context"] for r in misses)
            synthesis = sum(r["gold_in_context"] for r in misses)
            n = len(misses)
            print(f"\n  ── miss decomposition ({n} answerable-cat misses) ──")
            print(f"  retrieval loss   (evidence in neither pool nor ctx): {retrieval:>3}  ({retrieval/n*100:.0f}%)")
            print(f"  ranking/cut loss (evidence in pool, not in top_k):   {ranking:>3}  ({ranking/n*100:.0f}%)")
            print(f"  budget loss      (in top_k, cut by context chars):   {budget:>3}  ({budget/n*100:.0f}%)")
            print(f"  synthesis/judge  (evidence REACHED reader, wrong):   {synthesis:>3}  ({synthesis/n*100:.0f}%)")

        # How much of the retrieved list actually reaches the reader. At a wide
        # aperture with an unchanged char budget this is the binding constraint,
        # and it is invisible in every other line of the report.
        rendered_frac = [r["n_rendered"] / r["n_memories"] for r in results
                         if r.get("n_rendered") is not None and r.get("n_memories")]
        if rendered_frac:
            print(f"\n  context budget: {sum(rendered_frac)/len(rendered_frac)*100:.0f}% "
                  f"of retrieved memories survive MAX_CONTEXT_CHARS "
                  f"({_MAX_CTX[0] or 'default'} chars, x2 on cat-2/TR)")

    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_conv[r["conv_id"]].append(r)
    print(f"\n  {'conversation':<12} {'acc':>7} {'n':>6}")
    for cid in sorted(by_conv):
        rs = by_conv[cid]
        print(f"  {cid:<12} {sum(r['correct'] for r in rs)/len(rs)*100:>6.1f}% {len(rs):>6}")

    n_prof = [r.get("n_profile", 0) for r in results]
    if n_prof:
        print(f"\n  profile tier: {sum(n_prof)/len(n_prof):.1f} entries/question avg "
              f"({sum(1 for p in n_prof if p == 0)} questions saw zero)")


# ── main ────────────────────────────────────────────────────────────────────

# ── Re-judge (split the churn floor into reader share vs judge share) ───────
# Identical-config reruns move ~10 of 200 questions even though answer AND judge
# both run at temperature=0.0 (spec §8). Two nondeterministic LLMs sit in that
# loop and the accuracy line cannot separate them. Re-judging ONE stored answer
# file with the SAME judge holds the reader fixed: every flip that survives is
# the judge's share, and the remainder is the reader's. If the judge dominates,
# majority-of-3 judging shrinks the floor for the whole triad at once.
#
# LME has the same facility (`longmemeval_adapter.py:_rejudge_run`) but reads
# `hypothesis` out of a {config, per_question} envelope; LoCoMo `--out` writes a
# bare list keyed on `ai_answer`, hence this shim rather than a shared call.

_TRAP_RE = re.compile(r"^\[unanswerable; trap: (.*)\]$", re.S)


def _rejudge_file(
    args, judge_llm, owned_clients: OwnedResourceScope | None = None,
) -> None:
    """Re-judge a stored `--out` file, writing a flip-compatible copy."""
    from longmemeval_adapter import is_llm_error, judge_scored

    effective_judge_url = _judge_base_url(args, judge_llm)
    rows = json.loads(Path(args.rejudge).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        sys.exit(f"{args.rejudge}: expected a non-empty list of per-question results")

    print(f"\n=== LoCoMo RE-JUDGE — {Path(args.rejudge).name} ===")
    effective_judge_body = copy.deepcopy(judge_llm.extra_body)
    print(f"  rows: {len(rows)}   judge: {args.judge_model}"
          f"  +extra_body={effective_judge_body or '{}'}")
    print("  reader output is held FIXED — every flip below is judge nondeterminism\n",
          flush=True)

    def _rj(r: dict) -> dict:
        cat, ai = r["category"], str(r.get("ai_answer") or "")
        judge_raw = ""
        if not ai or is_llm_error(ai):
            new, judged = bool(r.get("correct")), False   # nothing judgeable
        else:
            gold = r.get("answer")
            if cat == 5:
                m = _TRAP_RE.match(str(gold))
                gold = _gold_for_judge(5, None, m.group(1) if m else "")
            verdict, judge_raw = judge_scored(
                judge_llm, CATEGORY_JUDGE[cat], r["question"],
                gold if gold is not None else "", ai)
            # A JUDGE error keeps the prior verdict and leaves `_rejudged`
            # False, reusing the answer-side rule three lines up rather than
            # inventing a second one. Overwriting would drop the row out of the
            # flip denominator AND destroy the baseline it is compared to.
            new, judged = (bool(r.get("correct")), False) if verdict is None \
                else (verdict, True)
        return {**r, "correct": new, "correct_original": r.get("correct"),
                "judge_raw": judge_raw,
                "judge_error": bool(judge_raw) and is_llm_error(judge_raw),
                "judge_model": args.judge_model,
                "judge_base_url": effective_judge_url,
                "judge_extra_body": copy.deepcopy(effective_judge_body),
                "_rejudged": judged}

    out_rows: list[dict] = [None] * len(rows)
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_rj, r): i for i, r in enumerate(rows)}
            for done, fut in enumerate(as_completed(futs), 1):
                out_rows[futs[fut]] = fut.result()
                if done % 25 == 0:
                    print(f"  ── re-judged {done}/{len(rows)}", flush=True)
    else:
        for i, r in enumerate(rows):
            out_rows[i] = _rj(r)

    if owned_clients is not None:
        owned_clients.close()
    judged = [r for r in out_rows if r["_rejudged"]]
    flipped = [r for r in judged if bool(r["correct"]) != bool(r["correct_original"])]
    t_to_f = [r for r in flipped if r["correct_original"]]
    o = sum(bool(r["correct_original"]) for r in out_rows) / len(out_rows)
    n = sum(bool(r["correct"]) for r in out_rows) / len(out_rows)
    print(f"\n  original: {o*100:.1f}%   re-judged: {n*100:.1f}%   ({(n-o)*100:+.1f}pp)")
    print(f"  judge churn: {len(flipped)}/{len(judged)} judged rows flipped "
          f"({len(flipped)/max(len(judged),1)*100:.1f}%)   "
          f"[{len(t_to_f)} correct→wrong, {len(flipped)-len(t_to_f)} wrong→correct]")
    if len(judged) < len(out_rows):
        print(f"  ({len(out_rows)-len(judged)} rows unjudgeable — kept prior verdict)")
    by_cat = defaultdict(lambda: [0, 0])
    for r in judged:
        c = by_cat[CATEGORY_NAME[r["category"]].replace("_abs", "")]
        c[1] += 1
        c[0] += bool(r["correct"]) != bool(r["correct_original"])
    print(f"\n  {'category':<14} {'flipped':>8} {'n':>5}")
    for name in sorted(by_cat):
        f, tot = by_cat[name]
        print(f"  {name:<14} {f:>8} {tot:>5}")

    # Written in the SAME bare-list shape as --out, so locomo_flip.py compares
    # this against the source file directly: that flip run IS the judge share.
    dest = args.out or str(Path(args.rejudge).with_suffix(".rejudged.json"))
    Path(dest).write_text(
        json.dumps(sanitize_for_artifact(out_rows), indent=2), encoding="utf-8"
    )
    print(f"\n  re-judged results → {dest}")
    print(f"  compare: python locomo_flip.py {args.rejudge} {dest}")


def _build_llm(model, base_url, api_key, extra_body):
    """Build a raw benchmark client with LME's endpoint-safe body defaults."""
    import os
    from longmemeval_adapter import LLMClient
    return LLMClient(model=model, api_key=api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
                     base_url=base_url, extra_body=extra_body)


def _run_main(owned_clients: OwnedResourceScope) -> None:
    ap = argparse.ArgumentParser(description="HyMem LoCoMo benchmark adapter.")
    ap.add_argument("--data", default=None, help="locomo10.json (snap-research/locomo shape)")
    ap.add_argument("--sample", type=int, default=0,
                    help="global QA cap after seeded shuffle; 0 = all 1986")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--categories", default=None, metavar="1,2,4",
                    help="comma-separated LoCoMo categories to keep (default all; "
                         "5 = adversarial)")
    ap.add_argument("--convs", default=None, metavar="conv-26,conv-30",
                    help="restrict to these sample_ids")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel CONVERSATIONS (each owns its store; ≤10 useful)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="base K; the pipeline searches top_k*3 like the LME driver")
    # Lever L6 — retrieval aperture. The MSC-sized defaults (15/10/10, rerank
    # pool 20) surface ~15 of a 369-689-turn LoCoMo history, and `message_hits`
    # is the ONLY tier that can carry a gold *turn* to the reader. Note
    # --rerank-top-k must stay comfortably ABOVE --message-fts-top-k or the
    # reranker has no room to lift a weak-lexical turn into the cut (at the
    # defaults it reranks 20 down to 15 — it can drop 5 items).
    ap.add_argument("--message-fts-top-k", type=int, default=None,
                    help="raw-turn slots surfaced (default 15)")
    ap.add_argument("--rerank-top-k", type=int, default=None,
                    help="BM25 candidate pool fed to the reranker (default 20)")
    ap.add_argument("--fts-top-k", type=int, default=None,
                    help="dreamed-chunk slots (default 10)")
    ap.add_argument("--graph-top-k", type=int, default=None,
                    help="graph-fact slots (default 10)")
    ap.add_argument("--max-context-chars", type=int, default=None,
                    help="reader context budget (LME default 8000; doubled for "
                         "cat-2/TR). Must scale WITH the aperture — a wider "
                         "top_k against an unchanged budget is truncated away "
                         "by _render_answer_context before the reader sees it")
    ap.add_argument("--user-speaker", choices=["a", "b"], default="a",
                    help="which speaker HyMem models as the user (default speaker_a)")
    ap.add_argument("--name-prefix", action="store_true",
                    help="prepend 'Name: ' to each ingested turn (lever L2: lets "
                         "FTS match speaker names in questions; changes extraction "
                         "input, so non-canonical until A/B'd)")
    ap.add_argument("--answerable-clause", action="store_true",
                    help="A/B lever: MSC-style answerability clause on cats 1-4. "
                         "LABEL-LEAKY (see spec) — never canonical")
    ap.add_argument("--answer-model", default=_ANSWER_MODEL)
    ap.add_argument("--answer-base-url", default=_DEEPSEEK_BASE_URL)
    ap.add_argument("--answer-api-key", default=None)
    ap.add_argument("--answer-extra-body", default=None, metavar="JSON",
                    help="optional provider body; omitted DeepSeek v4-flash "
                         "requests disable thinking automatically")
    ap.add_argument("--judge-model", default=_JUDGE_MODEL)
    ap.add_argument(
        "--judge-base-url", default=_DEEPSEEK_BASE_URL,
        help="judge endpoint (also used by --rejudge; default: DeepSeek)",
    )
    ap.add_argument(
        "--judge-api-key", default=None,
        help="judge-specific API key (never inherited from --answer-api-key)",
    )
    ap.add_argument("--judge-extra-body", default=None, metavar="JSON",
                    help="optional provider body; omitted DeepSeek v4-flash "
                         "requests disable thinking automatically")
    ap.add_argument("--hymem-model", default=_HYMEM_MODEL, help="HyMem's dream LLM")
    ap.add_argument("--hymem-base-url", default=_DEEPSEEK_BASE_URL)
    ap.add_argument("--hymem-thinking", choices=("auto", "disabled", "off", "enabled"),
                    default="auto", help="memory-pipeline thinking policy")
    ap.add_argument("--api-key", default="", help="HyMem dream LLM key")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--rules-extraction", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--facts", action=argparse.BooleanOptionalAction, default=None,
                    help="E1 narrative-facts READ side (cfg.facts_enabled). None = "
                         "config default (ON); --no-facts is the paired control arm "
                         "against the same store (pair it with --keep-db/--reuse-db so "
                         "both arms read identical stores). Gate on the all-800 net vs "
                         "the churn band, and read n_facts BEFORE the score.")
    ap.add_argument("--facts-extraction", action=argparse.BooleanOptionalAction, default=None,
                    help="E1 WRITE side (cfg.facts_extraction_enabled). None = config "
                         "default (ON). Changes what is STORED — only differs on a "
                         "rebuild, so never mix it into a read-side A/B.")
    ap.add_argument("--graph-multihop", action="store_true",
                    help="Track-A BFS — cat 1 (multi-hop) is the A/B target")
    ap.add_argument(
        "--no-dream", action="store_true",
        help="skip indexing (explicit non-comparable message-only development "
             "path; reused --db-dir stores are refused unless rebuilt --fresh)",
    )
    ap.add_argument(
        "--indexing-max-cycles", type=int,
        default=DEFAULT_INDEXING_MAX_CYCLES,
        help="per-wave dream-cycle safety cap before failing closed (default 100)",
    )
    ap.add_argument(
        "--indexing-timeout-s", type=float,
        default=DEFAULT_INDEXING_TIMEOUT_S,
        help="per-wave wall-clock convergence bound in seconds (default 3600)",
    )
    ap.add_argument("--dream-per-session", action="store_true",
                    help="fully converge after EACH of the 19-32 sessions "
                         "(live-store posture; expensive — default converges "
                         "the complete history at the end)")
    ap.add_argument("--db-dir", default=None,
                    help="persist per-conversation stores here and REUSE them on "
                         "later runs (skips ingest, validates the immutable build "
                         "receipt, then reconverges). Use --fresh after material "
                         "core/config changes or for legacy receiptless stores")
    ap.add_argument("--fresh", action="store_true",
                    help="with --db-dir: rebuild stores instead of reusing")
    ap.add_argument("--keep-db", action="store_true")
    ap.add_argument(
        "--results-dir", default=None,
        help=("strict immutable archive/checkpoint directory (default: the "
              "--out directory when supplied, otherwise ./locomo_results)"),
    )
    ap.add_argument(
        "--out", default=None,
        help=("legacy bare per-question JSON sidecar; strict scored evidence is "
              "always archived separately and this path remains mutable"),
    )
    ap.add_argument("--dump-context", action="store_true",
                    help="include the exact rendered answer context in each result")
    ap.add_argument("--dump-topk", action="store_true",
                    help="include the joined top_k memory text (the haystack "
                         "gold_in_topk is scored against) in each result")
    ap.add_argument("--diag-only", action="store_true",
                    help="retrieval + render only: NO answering, NO judging, so "
                         "it costs no reader calls. Implies --dump-context and "
                         "--dump-topk and writes correct=null. Join it onto a "
                         "real run by question id (locomo_audit.py --topk-dump) "
                         "to re-score the gold surfaces at a strict tau")
    ap.add_argument("--rejudge", default=None, metavar="RESULTS.json",
                    help="re-judge a stored --out file with the SAME reader output "
                         "(no ingest, no answering) and report judge-only churn; "
                         "writes a flip-compatible copy to --out or *.rejudged.json")
    ap.add_argument("--sim", action="store_true", help="offline: StubLLM, no API")
    ap.add_argument("--json", action="store_true")
    add_strict_run_arguments(ap)
    args = ap.parse_args()

    try:
        _indexing_limits(args)
        if isinstance(args.sample, bool) or args.sample < 0:
            raise BenchmarkIntegrityError("sample must be a non-negative integer")
        if isinstance(args.workers, bool) or args.workers <= 0:
            raise BenchmarkIntegrityError("workers must be a positive integer")
        if isinstance(args.top_k, bool) or args.top_k <= 0:
            raise BenchmarkIntegrityError("top-k must be a positive integer")
        if args.max_context_chars is not None and args.max_context_chars <= 0:
            raise BenchmarkIntegrityError(
                "max-context-chars must be a positive integer"
            )
        args.answer_base_url = validate_http_endpoint(
            args.answer_base_url, label="reader"
        ).url
        args.judge_base_url = validate_http_endpoint(
            args.judge_base_url, label="judge"
        ).url
        args.hymem_base_url = validate_http_endpoint(
            args.hymem_base_url, label="memory pipeline"
        ).url
    except BenchmarkIntegrityError as exc:
        ap.error(str(exc))
    except ValueError as exc:
        ap.error(str(exc))

    strict_controls = bool(
        args.checkpoint or args.resume_from or args.retry_failures
        or args.calibration_receipt or args.freeze_calibration
        or args.protocol_split != "full"
    )
    if args.sim and args.embeddings:
        ap.error("--sim does not construct an embedding client; drop --embeddings")
    if (args.rejudge or args.diag_only) and strict_controls:
        ap.error(
            "checkpoint/calibration/protocol flags are only valid for the "
            "strict benchmark run, not --rejudge or --diag-only"
        )
    if args.retry_failures and not args.resume_from:
        ap.error("--retry-failures requires --resume-from")
    if args.freeze_calibration and (args.checkpoint or args.resume_from):
        ap.error("--freeze-calibration cannot create or resume a checkpoint")
    if (args.freeze_calibration or args.protocol_split != "full") and args.sample:
        ap.error(
            "--freeze-calibration and dev/holdout runs require --sample 0"
        )
    if args.out:
        out_resolved = Path(args.out).resolve(strict=False)
        for label, raw in (
            ("--checkpoint", args.checkpoint),
            ("--resume-from", args.resume_from),
        ):
            if raw and Path(raw).resolve(strict=False) == out_resolved:
                ap.error(f"--out must not overwrite {label}")

    try:
        from longmemeval_adapter import resolve_model_extra_body

        parsed_answer_body = parse_extra_body_arg(
            args.answer_extra_body, "answer"
        )
        parsed_judge_body = parse_extra_body_arg(
            args.judge_extra_body, "judge"
        )
        args.answer_extra_body_obj, _ = resolve_model_extra_body(
            args.answer_model, args.answer_base_url, parsed_answer_body
        )
        args.judge_extra_body_obj, _ = resolve_model_extra_body(
            args.judge_model, args.judge_base_url, parsed_judge_body
        )
    except (BenchmarkIntegrityError, ValueError) as exc:
        ap.error(str(exc))

    if not args.sim:
        if args.rejudge:
            active_models = (("LoCoMo judge", args.judge_model),)
        elif args.diag_only:
            active_models = (("LoCoMo memory pipeline", args.hymem_model),)
        else:
            active_models = (
                ("LoCoMo reader", args.answer_model),
                ("LoCoMo judge", args.judge_model),
                ("LoCoMo memory pipeline", args.hymem_model),
            )
        try:
            for role, active_model in active_models:
                require_active_model(active_model, role=role)
        except DeprecatedModelAliasError as exc:
            ap.error(str(exc))

    if args.diag_only:
        # --sim leaves `rendered` None, which is the one surface this pass exists
        # to capture; a sim diag dump would silently be topk-only.
        if args.sim:
            sys.exit("--diag-only needs the real renderer; drop --sim.")
        if not args.out:
            sys.exit("--diag-only produces a sidecar to join; pass --out FILE.")
    if args.rejudge:
        if args.sim:
            sys.exit("--rejudge needs a real judge; drop --sim.")
        judge_llm = owned_clients.own(
            _build_llm(
                args.judge_model, args.judge_base_url,
                args.judge_api_key, args.judge_extra_body_obj,
            ),
            label="rejudge client",
        )
        _rejudge_file(args, judge_llm, owned_clients)
        return

    try:
        categories = ({int(c) for c in args.categories.split(",")}
                      if args.categories else None)
    except ValueError as exc:
        ap.error(f"--categories must be comma-separated integers: {exc}")
    if categories is not None and not categories <= set(CATEGORY_NAME):
        ap.error("--categories may contain only 1,2,3,4,5")

    source_convs = load_locomo_data(
        args.data, user_speaker=args.user_speaker,
        categories=categories, name_prefix=args.name_prefix,
    )
    if args.convs:
        keep = {item.strip() for item in args.convs.split(",") if item.strip()}
        source_convs = [conv for conv in source_convs if conv["id"] in keep]
    if not any(conv.get("qa") for conv in source_convs):
        print("No LoCoMo questions selected.")
        sys.exit(1)
    source_ids = validate_ids(
        (q["question_id"] for conv in source_convs for q in conv["qa"]),
        label="LoCoMo eligible dataset",
    )
    convs = (
        list(source_convs)
        if args.freeze_calibration or args.protocol_split != "full"
        else sample_questions(source_convs, args.sample, args.seed)
    )
    if args.sample > len(source_ids):
        ap.error(
            f"--sample {args.sample} exceeds the eligible dataset size "
            f"({len(source_ids)})"
        )
    all_ids = validate_ids(
        (q["question_id"] for conv in convs for q in conv["qa"]),
        label="LoCoMo selected dataset",
    )
    data_sha = (
        file_hash(args.data) if args.data
        else content_hash(_SIM_FIXTURE)
    )
    extraction_cfg, extraction_effective_cfg = _effective_hymem_config(args)
    extraction_prompt_version = extraction_cfg.prompt_version
    runtime_extraction_binding = validate_extraction_canary_config_binding(
        extraction_canary_policy(
            prompt_version=extraction_prompt_version
        ),
        extraction_effective_cfg,
    )

    # Retrieval-only diagnostics retain their historical bare-list contract.
    # They do not produce verdicts and are deliberately outside the scored
    # checkpoint/calibration protocol.
    if args.diag_only:
        n_q = len(all_ids)
        print(f"Loaded {len(convs)} conversations, {n_q} questions "
              f"({sum(c['n_sessions'] for c in convs)} sessions total)", flush=True)
        extraction_canary_report = (
            skipped_extraction_canary(
                "no_dream", prompt_version=extraction_prompt_version
            ) if args.no_dream
            else run_configured_extraction_canary(
                api_key=args.api_key, base_url=args.hymem_base_url,
                model=args.hymem_model, thinking=args.hymem_thinking,
                prompt_version=extraction_prompt_version,
            )
        )
        mode = "no_dream" if args.no_dream else "required"
        validate_extraction_canary_report(
            extraction_canary_report, expected_mode=mode,
            expected_client=(
                extraction_canary_client_policy(
                    base_url=args.hymem_base_url, model=args.hymem_model,
                    thinking=args.hymem_thinking,
                ) if mode == "required" else None
            ),
            require_client_closed=mode == "required",
            expected_prompt_version=extraction_prompt_version,
        )
        print_extraction_canary(extraction_canary_report)
        if args.max_context_chars:
            _MAX_CTX[0] = args.max_context_chars
            import longmemeval_adapter as _lme
            _lme.MAX_CONTEXT_CHARS = args.max_context_chars
        results: list[dict] = []
        for conv in convs:
            results.extend(run_or_record_indexing_failure(
                lambda conv=conv: evaluate_conversation(
                    conv, args, None, None
                ),
                benchmark="locomo", out_path=args.out,
                extraction_canary=extraction_canary_report,
            ))
        for row in results:
            row["extraction_canary"] = dict(extraction_canary_report)
        owned_clients.close()
        ans = [r for r in results if r["category"] != 5]
        n = len(ans) or 1
        print(f"\n  ── diagnostics-only pass ({len(ans)} answerable-cat questions, "
              f"no reader, no judge) ──")
        for key in ("gold_in_pool", "gold_in_topk", "gold_in_render"):
            print(f"  {key:<16} {sum(bool(r[key]) for r in ans)/n*100:>5.1f}%  (tau=0.6)")
        Path(args.out).write_text(
            json.dumps(sanitize_for_artifact(results), indent=2),
            encoding="utf-8",
        )
        print(f"\n  diagnostic results → {args.out}")
        return

    strict_config, strict_models = _strict_identity(args)
    if args.freeze_calibration:
        receipt = freeze_calibration(
            args.freeze_calibration,
            benchmark="LoCoMo",
            dataset_hash=data_sha,
            ids=source_ids,
            config=strict_config,
            models=strict_models,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
        )
        print(f"Frozen LoCoMo calibration: dev={len(receipt['dev_ids'])}, "
              f"holdout={len(receipt['holdout_ids'])}")
        return

    calibration = None
    if args.calibration_receipt:
        calibration = load_calibration(
            args.calibration_receipt,
            benchmark="LoCoMo",
            dataset_hash=data_sha,
            config=strict_config,
            models=strict_models,
            ids=source_ids,
        )
    selected_ids = select_protocol_ids(
        all_ids, split=args.protocol_split, receipt=calibration
    )
    convs = _select_questions_by_id(convs, selected_ids)
    manifest = build_manifest(
        benchmark="LoCoMo",
        code_sha256=locomo_code_hash(),
        data_sha256=data_sha,
        config=strict_config,
        models=strict_models,
        seed=args.seed,
        expected_ids=selected_ids,
        protocol_split=args.protocol_split,
        calibration=calibration,
    )
    manifest_extraction_binding = validate_extraction_canary_config_binding(
        manifest["config"].get("extraction_canary"),
        manifest["config"].get("effective_hymem_config"),
    )
    if manifest_extraction_binding != runtime_extraction_binding:
        raise BenchmarkIntegrityError(
            "LoCoMo manifest extraction contract differs from runtime config"
        )
    extraction_prompt_version = manifest_extraction_binding["prompt_version"]

    results_dir = Path(
        args.results_dir
        or (Path(args.out).parent if args.out else Path.cwd() / "locomo_results")
    )
    latest_path = results_dir / "locomo-latest.json"
    if args.out and Path(args.out).resolve(strict=False) == latest_path.resolve(strict=False):
        ap.error("--out must not overwrite the strict latest pointer")
    for label, raw in (
        ("--checkpoint", args.checkpoint),
        ("--resume-from", args.resume_from),
    ):
        if (
            raw
            and Path(raw).resolve(strict=False)
            == latest_path.resolve(strict=False)
        ):
            ap.error(f"{label} must not alias the strict latest pointer")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, is_resume = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        resume_from=args.resume_from,
        base_dir=results_dir,
        benchmark="locomo",
        run_id=manifest["run_id"],
    )

    ledger: AtomicCheckpoint | None = None
    try:
        ledger = AtomicCheckpoint(
            checkpoint_path,
            manifest=manifest,
            expected_ids=selected_ids,
            resume=is_resume,
            retry_failures=args.retry_failures,
            scored=not args.sim,
        )
        pending = set(ledger.pending_ids)
        work_convs = [
            {**conv, "qa": [q for q in conv["qa"]
                              if q["question_id"] in pending]}
            for conv in convs
            if any(q["question_id"] in pending for q in conv["qa"])
        ]
        print(f"Loaded {len(convs)} conversations, {len(selected_ids)} questions "
              f"({sum(c['n_sessions'] for c in convs)} sessions total)"
              f"{'  [SIM]' if args.sim else ''}", flush=True)
        print(f"  Strict checkpoint: {checkpoint_path} "
              f"({len(pending)} pending / {len(selected_ids)} expected)")

        start_time = time.time()
        segment_id = (
            f"process-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{os.getpid()}"
        )
        answer_llm = judge_llm = None
        attempted = 0
        started_ids: set[str] = set()
        attempted_ids: set[str] = set()
        runtime_lock = threading.RLock()
        pipeline_by_scope: dict[str, dict[str, Any]] = {}
        embedding_by_scope: dict[str, dict[str, Any]] = {}
        indexing_by_scope: dict[str, dict[str, Any]] = {}
        indexing_failures: dict[str, dict[str, Any]] = {}
        extraction_canary_report: dict[str, Any] = (
            skipped_extraction_canary(
                "no_pending_work", prompt_version=extraction_prompt_version
            )
            if not pending else
            skipped_extraction_canary(
                "simulation", prompt_version=extraction_prompt_version
            )
            if args.sim else
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

        def _zero_embedding_usage() -> dict[str, Any]:
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

        def _segment(status: str) -> dict[str, Any]:
            with runtime_lock:
                if embedding_by_scope:
                    embedding_usage = aggregate_embedding_usage_snapshots(
                        embedding_by_scope.values()
                    )
                elif pending:
                    embedding_usage = embedding_usage_snapshot(
                        None, configured=bool(args.embeddings)
                    )
                else:
                    embedding_usage = _zero_embedding_usage()
                return {
                    "segment_id": segment_id,
                    "status": status,
                    "elapsed_s": time.time() - start_time,
                    "attempted_attempts": attempted,
                    "model_identities": manifest["models"],
                    "reader_usage": usage_snapshot(answer_llm),
                    "judge_usage": usage_snapshot(judge_llm),
                    "memory_pipeline_usage": aggregate_usage_snapshots(
                        pipeline_by_scope.values()
                    ),
                    "embedding_usage": embedding_usage,
                    "indexing_runs": [
                        {"scope_id": scope, "summary": dict(summary)}
                        for scope, summary in sorted(indexing_by_scope.items())
                    ],
                    "indexing_failures": [
                        {"scope_id": scope, "summary": dict(summary)}
                        for scope, summary in sorted(indexing_failures.items())
                    ],
                    "extraction_canary": dict(extraction_canary_report),
                }

        class _CheckpointPersistenceAbort(BaseException):
            """Never reinterpret structural/checkpoint failure as conversation failure."""

            def __init__(
                self, *, restore_cause_on_exit: bool = False,
            ) -> None:
                super().__init__()
                self.restore_cause_on_exit = restore_cause_on_exit

        parallel_stop = threading.Event()
        parallel_primary: list[BaseException] = []

        def _checkpoint_abort(
            cause: BaseException, *, restore_cause_on_exit: bool = False,
        ) -> BaseException:
            abort = _CheckpointPersistenceAbort(
                restore_cause_on_exit=restore_cause_on_exit,
            )
            abort.__cause__ = cause
            abort.__suppress_context__ = True
            return abort

        def _external_abort_primary(exc: BaseException) -> BaseException:
            if (
                isinstance(exc, _CheckpointPersistenceAbort)
                and exc.restore_cause_on_exit
                and exc.__cause__ is not None
            ):
                return exc.__cause__
            return exc

        def _signal_parallel_abort(exc: BaseException) -> BaseException:
            # ``runtime_lock`` is also the persistence gate.  Setting the stop
            # signal while holding it prevents any concurrent callback from
            # beginning a later ledger record after the first fatal fault.
            with runtime_lock:
                if not parallel_primary:
                    parallel_primary.append(exc)
                parallel_stop.set()
                return parallel_primary[0]

        def _parallel_abort_for(exc: BaseException) -> BaseException | None:
            if isinstance(exc, _ParallelConversationStopped):
                with runtime_lock:
                    return parallel_primary[0] if parallel_primary else None
            if isinstance(exc, _CheckpointPersistenceAbort):
                candidate = exc
            elif isinstance(exc, BenchmarkCleanupError):
                candidate = exc
            elif isinstance(exc, Exception):
                if not is_structural_benchmark_error(exc):
                    return None
                candidate = _checkpoint_abort(exc)
            else:
                candidate = exc
            return _signal_parallel_abort(candidate)

        def _raise_checkpoint_abort(
            cause: BaseException, *, restore_cause_on_exit: bool = False,
        ) -> None:
            abort = _checkpoint_abort(
                cause, restore_cause_on_exit=restore_cause_on_exit,
            )
            primary = _signal_parallel_abort(abort)
            if primary is abort:
                raise abort from cause
            raise _ParallelConversationStopped()

        def _mark_attempt(item_id: str) -> None:
            """Own each started question once, even if its row is cancelled."""

            nonlocal attempted
            with runtime_lock:
                if parallel_stop.is_set():
                    raise _ParallelConversationStopped()
                if item_id in started_ids:
                    _raise_checkpoint_abort(BenchmarkIntegrityError(
                        "LoCoMo started one question more than once in an "
                        "execution segment"
                    ))
                started_ids.add(item_id)
                attempted += 1

        def _capture_runtime(runtime: dict[str, Any]) -> None:
            with runtime_lock:
                scope = str(runtime["scope_id"])
                pipeline_by_scope[scope] = dict(
                    runtime["memory_pipeline_usage"]
                )
                embedding_by_scope[scope] = dict(runtime["embedding_usage"])
                indexing = runtime.get("indexing")
                if isinstance(indexing, dict):
                    indexing_by_scope[scope] = dict(indexing)

        def _persist_aborted_segment(
            primary_exception: BaseException,
        ) -> None:
            """Freeze post-drain spend without replacing the abort primary."""

            def persist_segment_snapshot() -> None:
                ledger.update_execution_segment(
                    segment_id, _segment("complete")
                )

            run_cleanup_actions(
                [("execution_segment_snapshot", persist_segment_snapshot)],
                primary_exception=primary_exception,
            )

        def _persist(row: dict, runtime: dict[str, Any] | None = None) -> None:
            nonlocal attempted
            safe_row = dict(row)
            safe_row["extraction_canary"] = dict(extraction_canary_report)
            with runtime_lock:
                if parallel_stop.is_set():
                    # An in-flight sibling may finish provider work after the
                    # primary record fault.  Retain its cumulative meters, but
                    # never admit its now-aborted row into the checkpoint.
                    if runtime is not None:
                        _capture_runtime(runtime)
                    raise _ParallelConversationStopped()
                item_id = safe_row.get("question_id")
                if item_id in attempted_ids:
                    duplicate = BenchmarkIntegrityError(
                        "LoCoMo emitted one question more than once in an "
                        "execution segment"
                    )
                    _raise_checkpoint_abort(duplicate)
                attempted_ids.add(item_id)
                if item_id not in started_ids:
                    # Backward-compatible evaluator seam: injected/legacy
                    # implementations may only announce work via the result
                    # callback rather than the pre-provider attempt callback.
                    started_ids.add(item_id)
                    attempted += 1
                if runtime is not None:
                    _capture_runtime(runtime)
                try:
                    ledger.record(
                        item_id, row=safe_row,
                        execution_segment=_segment("running"),
                    )
                except BaseException as exc:
                    _raise_checkpoint_abort(
                        exc, restore_cause_on_exit=True,
                    )

        def _record_returned(rows: object) -> None:
            if not isinstance(rows, list):
                _raise_checkpoint_abort(
                    BenchmarkIntegrityError(
                        "LoCoMo conversation returned malformed rows"
                    )
                )
            for row in rows:
                if not isinstance(row, dict):
                    _raise_checkpoint_abort(
                        BenchmarkIntegrityError(
                            "LoCoMo conversation returned a malformed row"
                        )
                    )
                # Actual evaluate_conversation calls the checkpoint callback
                # before returning its rows. This fallback exists for older
                # injected evaluators only and must not interpret a just-failed
                # retried row as pending work a second time.
                if row.get("question_id") not in attempted_ids:
                    _persist(row)

        def _record_conversation_failure(conv: dict, exc: Exception) -> None:
            if isinstance(exc, BenchmarkCleanupError):
                _signal_parallel_abort(exc)
                raise exc
            if (
                isinstance(exc, BenchmarkIntegrityError)
                and not isinstance(exc, IndexingConvergenceError)
            ):
                _raise_checkpoint_abort(exc)
            scope = f"locomo:{conv['id']}"
            summary = (
                dict(exc.summary)
                if isinstance(exc, IndexingConvergenceError) else
                {"status": "failed_before_scoring",
                 "failure_reason": f"conversation_failure:{bounded_exception_type(exc)}"}
            )
            with runtime_lock:
                # A callback may have already observed successful (or
                # explicitly skipped) indexing before a reader/evaluator
                # subsequently failed. Preserve that one source outcome;
                # failures after indexing belong to the affected QA rows.
                if scope not in indexing_by_scope:
                    indexing_failures[scope] = sanitize_for_artifact(
                        summary, _preserve_evidence_text=False
                    )
            remaining = set(ledger.pending_ids)
            for question in conv["qa"]:
                qid = question["question_id"]
                if qid not in remaining or qid in attempted_ids:
                    continue
                _persist({
                    "id": question.get("qa_id", qid),
                    "question_id": qid,
                    "conv_id": conv["id"],
                    "question_type": question.get("qtype", "unknown"),
                    "category": question.get("category"),
                    "question": question.get("question", ""),
                    "correct": False,
                    "judge_raw": "",
                    "judge_error": False,
                    "benchmark_failure": (
                        f"conversation_failure:{bounded_exception_type(exc)}"
                    ),
                })

        if pending:
            ledger.update_execution_segment(segment_id, _segment("running"))
            if args.sim:
                extraction_canary_mode = "simulation"
            elif args.no_dream:
                extraction_canary_mode = "no_dream"
            else:
                extraction_canary_mode = "required"
                try:
                    extraction_canary_report = run_configured_extraction_canary(
                        api_key=args.api_key,
                        base_url=args.hymem_base_url,
                        model=args.hymem_model,
                        thinking=args.hymem_thinking,
                        prompt_version=extraction_prompt_version,
                    )
                except ExtractionCanaryError as exc:
                    extraction_canary_report = dict(exc.report)
                    ledger.update_execution_segment(
                        segment_id, _segment("running")
                    )
                    raise
            validate_extraction_canary_report(
                extraction_canary_report,
                expected_mode=extraction_canary_mode,
                expected_client=(
                    extraction_canary_client_policy(
                        base_url=args.hymem_base_url,
                        model=args.hymem_model,
                        thinking=args.hymem_thinking,
                    ) if extraction_canary_mode == "required" else None
                ),
                require_client_closed=extraction_canary_mode == "required",
                expected_prompt_version=extraction_prompt_version,
            )
            ledger.update_execution_segment(segment_id, _segment("running"))
            print_extraction_canary(extraction_canary_report)

            if args.max_context_chars:
                _MAX_CTX[0] = args.max_context_chars
                if not args.sim:
                    import longmemeval_adapter as _lme
                    _lme.MAX_CONTEXT_CHARS = args.max_context_chars
            if not args.sim:
                answer_llm = owned_clients.own(
                    _build_llm(
                        args.answer_model, args.answer_base_url,
                        args.answer_api_key, args.answer_extra_body_obj,
                    ),
                    label="reader client",
                )
                judge_llm = owned_clients.own(
                    _build_llm(
                        args.judge_model, args.judge_base_url,
                        args.judge_api_key, args.judge_extra_body_obj,
                    ),
                    label="judge client",
                )
                ledger.update_execution_segment(
                    segment_id, _segment("running")
                )

            if args.workers > 1:
                def _raise_parallel_primary() -> None:
                    with runtime_lock:
                        if parallel_stop.is_set():
                            if parallel_primary:
                                raise parallel_primary[0]
                            raise _ParallelConversationStopped()

                def _parallel_evaluate(index: int, conv: dict) -> list[dict]:
                    if parallel_stop.is_set():
                        raise _ParallelConversationStopped()
                    try:
                        return evaluate_conversation(
                            conv, args, answer_llm, judge_llm,
                            pending_ids={
                                q["question_id"] for q in conv["qa"]
                            },
                            on_checkpoint=_persist,
                            _parallel_stop=parallel_stop,
                            _on_fatal_abort=_parallel_abort_for,
                            _on_attempt=_mark_attempt,
                            _on_runtime_snapshot=_capture_runtime,
                        )
                    except _ParallelConversationStopped:
                        raise
                    except BaseException as exc:
                        primary = _parallel_abort_for(exc)
                        if primary is None:
                            # Ordinary provider/conversation failures are still
                            # materialized by the coordinator below.
                            raise
                        if primary is exc:
                            raise
                        raise primary

                worker_count = min(args.workers, len(work_convs))
                pool = ThreadPoolExecutor(max_workers=worker_count)
                work_iter = iter(enumerate(work_convs))
                futures: dict[Any, tuple[int, dict]] = {}

                def _fill_parallel_window() -> None:
                    while (
                        len(futures) < worker_count
                        and not parallel_stop.is_set()
                    ):
                        try:
                            index, conv = next(work_iter)
                        except StopIteration:
                            return
                        future = pool.submit(_parallel_evaluate, index, conv)
                        futures[future] = (index, conv)

                try:
                    _fill_parallel_window()
                    while futures:
                        _raise_parallel_primary()
                        ready, _pending_futures = wait(
                            tuple(futures), return_when=FIRST_COMPLETED,
                        )
                        _raise_parallel_primary()
                        ordered_ready = sorted(
                            ready, key=lambda future: futures[future][0]
                        )
                        for future in ordered_ready:
                            _index, conv = futures.pop(future)
                            _raise_parallel_primary()
                            try:
                                _record_returned(future.result())
                            except _ParallelConversationStopped:
                                _raise_parallel_primary()
                                raise
                            except _CheckpointPersistenceAbort:
                                raise
                            except Exception as exc:
                                _record_conversation_failure(conv, exc)
                        # Refill only after the whole ready batch has been
                        # persisted/reconciled.  There is never an eager queue
                        # of expensive conversations left to drain on abort.
                        _fill_parallel_window()
                except BaseException as exc:
                    primary = _parallel_abort_for(exc)
                    if primary is None:
                        primary = _signal_parallel_abort(exc)
                    for future in futures:
                        future.cancel()
                    external_primary = _external_abort_primary(primary)
                    if external_primary is exc:
                        raise
                    raise external_primary from None
                finally:
                    primary_exception = sys.exc_info()[1]
                    if primary_exception is not None:
                        parallel_stop.set()
                        for future in futures:
                            future.cancel()
                    with runtime_lock:
                        record_abort = bool(
                            parallel_primary
                            and isinstance(
                                parallel_primary[0],
                                _CheckpointPersistenceAbort,
                            )
                            and parallel_primary[0].restore_cause_on_exit
                        )
                    cleanup_actions = [
                        ("resource_close", lambda: pool.shutdown(
                            wait=True, cancel_futures=True,
                        )),
                    ]
                    if record_abort:
                        # Freeze only after every already-running worker has
                        # unwound.  This includes its final shared-client usage
                        # and any runtime snapshot offered by a blocked row.
                        cleanup_actions.append((
                            "execution_segment_snapshot",
                            lambda: ledger.update_execution_segment(
                                segment_id, _segment("complete")
                            ),
                        ))
                    run_cleanup_actions(
                        cleanup_actions,
                        primary_exception=primary_exception,
                    )
            else:
                try:
                    for conv in work_convs:
                        try:
                            rows = evaluate_conversation(
                                conv, args, answer_llm, judge_llm,
                                pending_ids={
                                    q["question_id"] for q in conv["qa"]
                                },
                                on_checkpoint=_persist,
                                _on_attempt=_mark_attempt,
                                _on_runtime_snapshot=_capture_runtime,
                            )
                            _record_returned(rows)
                        except _CheckpointPersistenceAbort:
                            raise
                        except Exception as exc:
                            _record_conversation_failure(conv, exc)
                except _CheckpointPersistenceAbort as exc:
                    # evaluate_conversation has already released its adapter,
                    # so no further spend can race this final segment image.
                    primary = _external_abort_primary(exc)
                    try:
                        if primary is exc:
                            raise
                        raise primary from None
                    except BaseException as active_primary:
                        _persist_aborted_segment(active_primary)
                        raise
            ledger.update_execution_segment(
                segment_id, _segment("complete")
            )
        elif is_resume:
            # Recover a crash after its last row without constructing provider
            # clients or repeating canary/indexing work. A finalized checkpoint
            # is already terminal and intentionally remains byte-stable.
            try:
                ledger.update_execution_segment(
                    segment_id, _segment("complete")
                )
            except BenchmarkIntegrityError as exc:
                if "cannot mutate a finalized checkpoint" not in str(exc):
                    raise

        results = list(ledger.reconcile().rows)
        elapsed = time.time() - start_time
        scores = _compute_scores_local(results)
        payload = {
            "benchmark": "LoCoMo",
            "version": "strict-v1",
            "date": datetime.now(timezone.utc).isoformat(),
            "scores": scores,
            "strict_accuracy": (
                strict_accuracy(results) if not args.sim else None
            ),
            "result_digest": content_hash(sanitize_for_artifact(results)),
            "legacy_bare_out": bool(args.out),
        }
        archive_now = datetime.now(timezone.utc)
        stamp = archive_now.strftime("%Y%m%dT%H%M%SZ")
        nonce = archive_now.strftime("%f")
        archive_path = results_dir / (
            f"locomo-{stamp}-{nonce}-seed{args.seed}-strict-"
            f"{manifest['run_id'].removeprefix('sha256:')[:12]}.json"
        )
        artifact = prepare_checkpoint_artifact(ledger, payload=payload)
        publish_prepared_artifact_after_cleanup(
            archive_path,
            artifact,
            cleanup_actions=[
                ("resource_close", owned_clients.close),
                ("checkpoint_close", ledger.close),
            ],
        )
        write_latest_pointer(
            latest_path,
            archive=archive_path,
            run_id=manifest["run_id"],
            artifact_digest=content_hash(artifact),
        )
        print(f"  done in {elapsed:.0f}s")
        print(f"  strict archive → {archive_path}")

        # Compatibility output for locomo_audit.py / locomo_flip.py. It is
        # deliberately not the authoritative evidence and is written only
        # after provider/checkpoint teardown and immutable publication.
        if args.out:
            Path(args.out).write_text(
                json.dumps(sanitize_for_artifact(results), indent=2),
                encoding="utf-8",
            )
            print(f"  legacy per-question sidecar → {args.out}")
        if args.json:
            print(json.dumps(results, indent=2))
        elif not args.sim:
            _print_report(results, args)
    finally:
        if ledger is not None:
            run_cleanup_actions(
                [("checkpoint_close", ledger.close)],
                primary_exception=sys.exc_info()[1],
            )


def main() -> None:
    """CLI entry point owning shared clients through worker completion."""

    with OwnedResourceScope("LoCoMo shared provider clients") as owned_clients:
        return _run_main(owned_clients)


# A tiny in-schema fixture: 2 speakers, 3 dated sessions, one photo-share turn,
# and one QA per category — including the real file's quirks (string category,
# string-repr evidence, int answer) so --sim exercises every coercion.
_SIM_FIXTURE = [{
    "sample_id": "sim-1",
    "conversation": {
        "speaker_a": "Ada", "speaker_b": "Ben",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Ada", "dia_id": "D1:1",
             "text": "I finally signed up for the pottery class downtown!"},
            {"speaker": "Ben", "dia_id": "D1:2",
             "text": "Nice! I spent the weekend fixing my old motorbike.",
             "img_url": ["http://example.com/bike.jpg"],
             "blip_caption": "a red motorbike in a garage"},
        ],
        "session_2_date_time": "10:04 am on 19 June, 2023",
        "session_2": [
            {"speaker": "Ada", "dia_id": "D2:1",
             "text": "The pottery class is going great, I made three bowls in 2022... "
                     "no wait, I made three bowls already this month."},
            {"speaker": "Ben", "dia_id": "D2:2",
             "text": "I sold the motorbike and bought a bicycle instead."},
        ],
        "session_3_date_time": "9:30 pm on 2 July, 2023",
        "session_3": [
            {"speaker": "Ada", "dia_id": "D3:1",
             "text": "I'm thinking of selling my bowls at the summer market."},
            {"speaker": "Ben", "dia_id": "D3:2",
             "text": "Cycling to work daily now — 20 minutes each way."},
        ],
    },
    "qa": [
        {"question": "What does Ada plan to do with her bowls, and where did she learn to make them?",
         "answer": "Sell them at the summer market; she learned at the pottery class",
         "evidence": ["D3:1", "D1:1"], "category": 1},
        {"question": "When did Ada sign up for the pottery class?",
         "answer": "8 May 2023", "evidence": ["D1:1"], "category": 2},
        {"question": "Would Ben likely enjoy a cycling holiday?",
         "answer": "Yes, he cycles to work daily", "evidence": ["D3:2"], "category": 3},
        {"question": "How many bowls did Ada make?",
         "answer": 3, "evidence": ["D2:1"], "category": 4},
        {"question": "What did Ada do with her motorbike?",
         "adversarial_answer": "sold it and bought a bicycle",
         "evidence": "['D2:2']", "category": "5"},
    ],
}]


if __name__ == "__main__":
    main()
