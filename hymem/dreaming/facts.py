"""Authoritative narrative-fact extraction and lifecycle (schema v46).

The gated ``facts.v2`` prompt still sees ``role: content`` lines. Its input now
comes from the validated lossless-message stream, can resume inside one large
turn, and publishes an exact occurrence manifest. Each bounded source slice is
an authority unit: a later successful replay replaces that unit's current fact
set (including a successful empty set) while immutable lifecycle events retain
the full assertion/retraction/resurrection history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date as calendar_date
from typing import Iterable, Sequence

from hymem.config import HyMemConfig, MAX_FACTS_PER_EXTRACTION_UNIT
from hymem.core import db as core_db
from hymem.core.time import normalize_iso_timestamp
from hymem.dreaming.aggregation_provenance import (
    BoundSourceOccurrence,
    combine_source_occurrences,
    source_manifest_hash,
)
from hymem.dreaming.canonicalize import normalize
from hymem.dreaming.lossless import (
    CoveredMessage,
    covered_messages_after,
    validate_message_coverage_artifact,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.extraction.jsonio import loads_exact_or_fenced
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import FACTS_SYSTEM, FACTS_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.facts")

FACTS_PROMPT_VERSION = "facts.v2"
FACT_SOURCE_MANIFEST_VERSION = "fact-source-manifest-v1"
FACT_RESULT_VERSION = "fact-result-v1"
FACT_SLICE_VERSION = "fact-slice-v1"

# Hard authority-ledger bounds keep proof verification predictable even for a
# directly corrupted SQLite store or checksummed hostile portable snapshot.
# Shipped extraction emits at most eight facts per unit; these limits leave a
# generous operational margin without permitting unbounded history folds.
FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME = MAX_FACTS_PER_EXTRACTION_UNIT
FACT_MAX_REVISIONS_PER_OUTCOME = 256
FACT_MAX_SOURCES_PER_OUTCOME = 4096
FACT_MAX_HISTORY_ITEMS_PER_OUTCOME = (
    FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME * FACT_MAX_REVISIONS_PER_OUTCOME
)
FACT_MAX_LIFECYCLE_EVENTS_PER_OUTCOME = (
    FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME
    * (2 * FACT_MAX_REVISIONS_PER_OUTCOME - 1)
)
FACT_MAX_ENTITIES_PER_ITEM = 64
FACT_MAX_ENTITY_CHARS = 200

_MAX_FACT_CHARS = 600
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FACT_RETRY_RE = re.compile(
    r"^facts-lossless-v1\|prompt=facts\.v\d{1,6}\|chars=\d{1,9}\|"
    r"tokens=\d{1,9}\|items=\d{1,9}"
    r"(?:\|slice=sha256:[0-9a-f]{64})?\|retry-max=\d{1,9}$"
)
_FACT_CONFIG_RE = re.compile(
    r"^facts-lossless-v1\|prompt=facts\.v\d{1,6}\|chars=\d{1,9}\|"
    r"tokens=\d{1,9}\|items=\d{1,9}$"
)
_FACT_PROMPT_RE = re.compile(r"^facts\.v\d{1,6}$")


@dataclass
class FactsExtraction:
    """One exact, resumable extraction unit and its validated result."""

    items: list[dict] = field(default_factory=list)
    start_message_id: int | None = None
    covered_message_id: int | None = None
    parse_failed: bool = False
    cursor_before_message_id: int | None = None
    cursor_before_partial_message_id: int | None = None
    cursor_before_offset: int = 0
    partial_message_id: int | None = None
    next_message_offset: int = 0
    slice_key: str | None = None
    input_hash: str | None = None
    source_occurrences: tuple[BoundSourceOccurrence, ...] = ()
    caught_up: bool = False
    publication_version: str | None = None
    # ``None`` means this was a first-pass extraction and the slice must not
    # already exist (except for an exact idempotent duplicate). Replay binds
    # the generation it read so concurrent workers cannot append a revision to
    # state they never observed.
    expected_generation: int | None = None


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def fact_item_key(item: dict) -> str:
    """Stable immutable identity for one exact fact payload."""
    return _sha256_json({
        "version": FACT_RESULT_VERSION,
        "text": item["text"],
        "date": item.get("date"),
        "entities": list(item.get("entities") or ()),
    })


def canonical_fact_items(items: Iterable[dict]) -> list[dict]:
    """Deduplicate and order exact payloads without trusting model order."""
    by_key: dict[str, dict] = {}
    for item in items:
        value = {
            "text": item["text"],
            "date": item.get("date"),
            "entities": list(item.get("entities") or ()),
        }
        key = fact_item_key(value)
        previous = by_key.get(key)
        if previous is not None and previous != value:
            raise ValueError("fact key collision")
        by_key[key] = value
    return [by_key[key] for key in sorted(by_key)]


def fact_result_hash(items: Iterable[dict]) -> str:
    return _sha256_json({
        "version": FACT_RESULT_VERSION,
        "items": canonical_fact_items(items),
    })


def fact_slice_key(
    session_id: str,
    *,
    cursor_before_message_id: int | None,
    cursor_before_partial_message_id: int | None,
    cursor_before_offset: int,
    cursor_after_message_id: int | None,
    cursor_after_partial_message_id: int | None,
    cursor_after_offset: int,
    occurrences: Iterable[BoundSourceOccurrence],
) -> str:
    """Content-independent identity for one exact cursor interval."""
    return _sha256_json({
        "version": FACT_SLICE_VERSION,
        "session_id": session_id,
        "before": [
            cursor_before_message_id, cursor_before_partial_message_id,
            int(cursor_before_offset),
        ],
        "after": [
            cursor_after_message_id, cursor_after_partial_message_id,
            int(cursor_after_offset),
        ],
        "occurrences": [
            [item.session_id, item.message_id] for item in occurrences
        ],
    })


def fact_input_hash(rendered: str) -> str:
    return _sha256_json({"version": FACT_SLICE_VERSION, "rendered": rendered})


def facts_config_version(cfg: HyMemConfig) -> str:
    return (
        f"facts-lossless-v1|prompt={FACTS_PROMPT_VERSION}|"
        f"chars={int(cfg.dream_digest_max_chars)}|"
        f"tokens={int(cfg.dream_digest_max_tokens)}|"
        f"items={int(cfg.dream_max_facts_per_session)}"
    )


def facts_retry_policy_version(
    cfg: HyMemConfig, *, replay_slice_key: str | None = None
) -> str:
    replay = f"|slice={replay_slice_key}" if replay_slice_key is not None else ""
    return (
        f"{facts_config_version(cfg)}{replay}|"
        f"retry-max={int(cfg.facts_extraction_max_attempts)}"
    )


def fact_cursor_retry_unit_key(
    session_id: str,
    cursor_message_id: int | None,
    partial_message_id: int | None,
    offset: int,
) -> str:
    """Stable, non-sensitive identity for one held tail cursor."""

    return _sha256_json({
        "version": "fact-retry-cursor-v1",
        "session_id": session_id,
        "message_id": cursor_message_id,
        "partial_message_id": partial_message_id,
        "offset": int(offset),
    })


def facts_attempt_max_chars(configured_max: int, retry_count: int) -> int:
    """Shrink one unpublished failed unit without losing source bytes."""
    if configured_max <= 0:
        return configured_max
    # Retrying a model on twelve-character shards preserves bytes but destroys
    # semantic recall. Stop shrinking at a useful context floor (or the
    # configured cap itself when the caller deliberately chose less).
    floor = min(configured_max, max(256, len("assistant: ") + 1))
    return max(floor, configured_max // (2 ** min(max(0, retry_count), 8)))


def _bounded_fact_fragment_end(content: str, offset: int, room: int) -> int:
    """Choose a stable semantic boundary before a necessary hard split."""

    hard_end = min(len(content), offset + room)
    if hard_end >= len(content):
        return hard_end
    fragment = content[offset:hard_end]
    minimum = max(1, len(fragment) // 2)
    sentence_end = max(
        (fragment.rfind(marker) + len(marker) for marker in (". ", "! ", "? ", "\n")),
        default=0,
    )
    if sentence_end >= minimum:
        return offset + sentence_end
    whitespace_end = max(
        (fragment.rfind(marker) + len(marker) for marker in (" ", "\t")),
        default=0,
    )
    if whitespace_end >= minimum:
        return offset + whitespace_end
    return hard_end


def facts_generation_is_recognized(value: object) -> bool:
    return isinstance(value, str) and _FACT_CONFIG_RE.fullmatch(value) is not None


def fact_publication_version_is_recognized(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and (
            _FACT_PROMPT_RE.fullmatch(value)
            or _FACT_CONFIG_RE.fullmatch(value)
        )
    )


def facts_retry_state_is_valid(
    retry_count: object, retry_config_version: object, quarantined: object
) -> bool:
    if (
        not isinstance(retry_count, int) or isinstance(retry_count, bool)
        or retry_count < 0
        or not isinstance(quarantined, int) or isinstance(quarantined, bool)
        or quarantined not in (0, 1)
    ):
        return False
    if retry_count == 0:
        return retry_config_version is None and quarantined == 0
    if not isinstance(retry_config_version, str):
        return False
    if _FACT_RETRY_RE.fullmatch(retry_config_version) is None:
        return False
    try:
        maximum = int(retry_config_version.rsplit("=", 1)[1])
    except (ValueError, OverflowError):
        return False
    return bool(quarantined) == bool(maximum > 0 and retry_count >= maximum)


def record_fact_failure(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    max_attempts: int,
    retry_config_version: str,
) -> bool:
    """Record a held failure and return whether this source cursor quarantined."""
    row = conn.execute(
        "SELECT facts_retry_count,facts_retry_config_version "
        "FROM sessions WHERE id=?", (session_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown session: {session_id}")
    attempts = (
        int(row["facts_retry_count"] or 0) + 1
        if row["facts_retry_config_version"] == retry_config_version else 1
    )
    quarantined = bool(max_attempts > 0 and attempts >= max_attempts)
    conn.execute(
        "UPDATE sessions SET facts_retry_count=?,facts_retry_config_version=?,"
        "facts_quarantined=? WHERE id=?",
        (attempts, retry_config_version, int(quarantined), session_id),
    )
    if quarantined:
        log.warning(
            "facts.extraction_quarantined session_id=%s attempts=%d "
            "cursor_advanced=0", session_id, attempts,
        )
    return quarantined


def record_fact_failure_if_pending(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    max_attempts: int,
    retry_config_version: str,
    expected_cursor_message_id: int | None,
    expected_partial_message_id: int | None,
    expected_offset: int,
    replay_slice_key: str | None = None,
    expected_replay_generation: int | None = None,
    target_publication_version: str | None = None,
) -> bool | None:
    """CAS a held failure onto the source/replay unit that actually failed.

    ``None`` means another worker already advanced or replayed the unit.  Its
    success owns the reset retry state, so a late provider/persistence failure
    must not reintroduce a stale quarantine.
    """

    if not conn.in_transaction:
        raise RuntimeError("fact failure CAS requires a caller-owned transaction")
    current = conn.execute(
        "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
        "facts_cursor_offset FROM sessions WHERE id=?", (session_id,),
    ).fetchone()
    if current is None:
        raise ValueError(f"unknown session: {session_id}")
    if (
        current["facts_cursor_message_id"],
        current["facts_cursor_partial_message_id"],
        int(current["facts_cursor_offset"] or 0),
    ) != (
        expected_cursor_message_id,
        expected_partial_message_id,
        int(expected_offset),
    ):
        return None
    if replay_slice_key is not None:
        if (
            expected_replay_generation is None
            or target_publication_version is None
        ):
            raise ValueError("replay failure CAS requires generation and target")
        outcome = conn.execute(
            "SELECT generation,prompt_version FROM fact_extraction_outcomes "
            "WHERE slice_key=? AND session_id=?",
            (replay_slice_key, session_id),
        ).fetchone()
        if (
            outcome is None
            or int(outcome["generation"]) != expected_replay_generation
            or outcome["prompt_version"] == target_publication_version
        ):
            return None
    return record_fact_failure(
        conn, session_id, max_attempts=max_attempts,
        retry_config_version=retry_config_version,
    )


def facts_tail_message_id(conn: sqlite3.Connection, session_id: str) -> int | None:
    """Newest eligible occurrence inside the producer-published frontier."""
    try:
        row = conn.execute(
            "SELECT MAX(mc.message_id) AS message_id "
            "FROM message_retention_coverage mc JOIN sessions s "
            "ON s.id=mc.source_session_id "
            "WHERE mc.source_session_id=? "
            "AND mc.source_role IN ('user','assistant') "
            "AND mc.coverage_version=? "
            "AND mc.message_id <= COALESCE(s.coverage_message_id,-1)",
            (session_id, LOSSLESS_COVERAGE_VERSION),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row["message_id"]) if row and row["message_id"] is not None else None


def next_fact_outcome_for_replay(
    conn: sqlite3.Connection, session_id: str, publication_version: str
) -> str | None:
    """Return the earliest stale unit without rescanning a current session.

    A completed target-version cursor is the incremental trusted marker written
    only after :func:`fact_session_authority_is_valid` audits the whole chain.
    During a replay walk each stale unit is independently proven here; the
    runner performs one full audit immediately before publishing the new
    marker. Thus N replays cost O(N), while read/export boundaries always audit
    the complete committed chain themselves.
    """

    session = conn.execute(
        "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
        "facts_cursor_offset,facts_cursor_prompt_version FROM sessions WHERE id=?",
        (session_id,),
    ).fetchone()
    if session is None:
        raise RuntimeError("fact outcome session is absent")
    invalid_header = conn.execute(
        "SELECT 1 FROM fact_extraction_outcomes WHERE session_id=? "
        "AND source_manifest_complete=0 LIMIT 1", (session_id,),
    ).fetchone()
    if invalid_header is not None:
        raise RuntimeError("fact outcome source publication is incomplete")
    stale = conn.execute(
        "SELECT slice_key FROM fact_extraction_outcomes WHERE session_id=? "
        "AND source_manifest_complete=1 AND prompt_version<? LIMIT 1",
        (session_id, publication_version),
    ).fetchone()
    if stale is None:
        stale = conn.execute(
            "SELECT slice_key FROM fact_extraction_outcomes WHERE session_id=? "
            "AND source_manifest_complete=1 AND prompt_version>? LIMIT 1",
            (session_id, publication_version),
        ).fetchone()
    if stale is None:
        return None
    slice_key = str(stale["slice_key"])
    if load_fact_outcome_source_manifest(
        conn, slice_key, verify_result=True, _require_committed_chain=False,
    ) is None:
        raise RuntimeError("stale fact outcome contains corrupt authority")
    return slice_key


def fact_session_authority_is_valid(
    conn: sqlite3.Connection, session_id: str
) -> bool:
    """Full read-boundary audit of one session's committed fact ledger."""

    chain_cache: dict[str, bool | None] = {}
    if _committed_fact_slice_keys(conn, session_id) is not True:
        return False
    chain_cache[session_id] = True
    rows = conn.execute(
        "SELECT slice_key FROM fact_extraction_outcomes WHERE session_id=? "
        "ORDER BY COALESCE(cursor_before_partial_message_id,"
        "cursor_before_message_id,-1),"
        "CASE WHEN cursor_before_partial_message_id IS NULL THEN 1 ELSE 0 END,"
        "cursor_before_offset,slice_key",
        (session_id,),
    )
    return all(
        load_fact_outcome_source_manifest(
            conn, row["slice_key"], verify_result=True,
            chain_cache=chain_cache,
        ) is not None
        for row in rows
    )


def _bound_occurrence(
    conn: sqlite3.Connection, message: CoveredMessage
) -> BoundSourceOccurrence:
    row = conn.execute(
        "SELECT message_content_hash FROM message_retention_coverage "
        "WHERE message_id=? AND chunk_id=? AND coverage_version=?",
        (message.message_id, message.chunk_id, LOSSLESS_COVERAGE_VERSION),
    ).fetchone()
    if row is None or not isinstance(row["message_content_hash"], str):
        raise RuntimeError("fact source coverage lacks a content hash")
    return BoundSourceOccurrence(
        message_id=message.message_id,
        session_id=message.session_id,
        role=message.role,
        source_peer_id=message.source_peer_id,
        source_workspace_id=message.source_workspace_id,
        source_created_at=message.source_created_at,
        coverage_chunk_id=message.chunk_id,
        coverage_version=LOSSLESS_COVERAGE_VERSION,
        content_hash=row["message_content_hash"],
    )


def _validate_bound_occurrences(
    conn: sqlite3.Connection,
    session_id: str,
    occurrences: Iterable[BoundSourceOccurrence],
) -> tuple[BoundSourceOccurrence, ...]:
    canonical = combine_source_occurrences((occurrences,))
    if not canonical or any(item.session_id != session_id for item in canonical):
        raise ValueError("fact source manifest crosses a session boundary")
    for occurrence in canonical:
        proof = validate_message_coverage_artifact(
            conn,
            message_id=occurrence.message_id,
            chunk_id=occurrence.coverage_chunk_id,
            coverage_version=occurrence.coverage_version,
        )
        row = conn.execute(
            "SELECT message_content_hash FROM message_retention_coverage "
            "WHERE message_id=? AND chunk_id=? AND coverage_version=?",
            (
                occurrence.message_id, occurrence.coverage_chunk_id,
                occurrence.coverage_version,
            ),
        ).fetchone()
        expected = BoundSourceOccurrence(
            message_id=proof.message_id,
            session_id=proof.session_id,
            role=proof.role,
            source_peer_id=proof.source_peer_id,
            source_workspace_id=proof.source_workspace_id,
            source_created_at=proof.source_created_at,
            coverage_chunk_id=proof.chunk_id,
            coverage_version=occurrence.coverage_version,
            content_hash=row["message_content_hash"] if row else "",
        )
        if occurrence != expected or occurrence.coverage_version != LOSSLESS_COVERAGE_VERSION:
            raise ValueError("fact source occurrence mismatches durable coverage")
    return canonical


def _render_fact_slice(
    proofs: list[CoveredMessage],
    *,
    before_partial_message_id: int | None,
    before_offset: int,
    after_partial_message_id: int | None,
    after_offset: int,
) -> str:
    lines: list[str] = []
    for proof in proofs:
        start = before_offset if proof.message_id == before_partial_message_id else 0
        end = after_offset if proof.message_id == after_partial_message_id else len(proof.content)
        empty_whole = len(proof.content) == 0 and start == 0 and end == 0
        if start < 0 or (end <= start and not empty_whole) or end > len(proof.content):
            raise ValueError("fact outcome has invalid source offsets")
        lines.append(f"{proof.role}: {proof.content[start:end]}")
    return "\n".join(lines)


def _cursor_interval_is_exact(
    conn: sqlite3.Connection,
    outcome: sqlite3.Row,
    proofs: list[CoveredMessage],
) -> bool:
    """Prove the manifest contains every eligible turn crossed by its cursor."""
    before = outcome["cursor_before_message_id"]
    before_partial = outcome["cursor_before_partial_message_id"]
    before_offset = outcome["cursor_before_offset"]
    after = outcome["cursor_after_message_id"]
    after_partial = outcome["cursor_after_partial_message_id"]
    after_offset = outcome["cursor_after_offset"]
    if (
        not isinstance(before_offset, int) or isinstance(before_offset, bool)
        or not isinstance(after_offset, int) or isinstance(after_offset, bool)
        or before_offset < 0 or after_offset < 0
        or (before_partial is None) != (before_offset == 0)
        or (after_partial is None) != (after_offset == 0)
    ):
        return False
    if not proofs:
        return False
    if before_partial is not None:
        if (
            proofs[0].message_id != before_partial
            or not 0 < before_offset < len(proofs[0].content)
        ):
            return False
    end_id = after_partial if after_partial is not None else after
    if not isinstance(end_id, int) or isinstance(end_id, bool):
        return False
    try:
        expected = covered_messages_after(
            conn, outcome["session_id"], before,
            limit=len(proofs) + 1,
            roles=frozenset({"user", "assistant"}),
            through_message_id=end_id,
        )
    except (RuntimeError, TypeError, ValueError):
        return False
    if [item.message_id for item in expected] != [item.message_id for item in proofs]:
        return False
    if after_partial is None:
        return after == proofs[-1].message_id
    if (
        proofs[-1].message_id != after_partial
        or not 0 < after_offset < len(proofs[-1].content)
    ):
        return False
    expected_after = proofs[-2].message_id if len(proofs) > 1 else before
    # A continuation can consume the remainder of prior complete turns only
    # before reaching the final partial source; the full cursor is exactly the
    # last such turn (or remains at its prior value).
    return after == expected_after


def extract_facts(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
    cfg: HyMemConfig,
    *,
    since_message_id: int | None = None,
    partial_message_id: int | None = None,
    start_offset: int = 0,
    max_chars: int | None = None,
) -> FactsExtraction | None:
    """Extract one bounded lossless slice, resuming inside oversized turns."""
    if start_offset < 0 or (partial_message_id is None) != (start_offset == 0):
        raise ValueError("invalid facts partial cursor")
    messages = covered_messages_after(
        conn, session_id, since_message_id, limit=256,
        roles=frozenset({"user", "assistant"}),
    )
    if not messages:
        return None
    if partial_message_id is not None and messages[0].message_id != partial_message_id:
        raise RuntimeError("facts partial cursor does not name the next source")

    # Empty/whitespace turns have exact provenance but no fact-bearing bytes.
    # Publish them as a successful empty unit without paying an LLM call, so a
    # blank eligible turn can never pin the cursor forever.
    if not messages[0].content[start_offset:].strip():
        message = messages[0]
        occurrence = _bound_occurrence(conn, message)
        occurrences = (occurrence,)
        rendered = f"{message.role}: {message.content[start_offset:]}"
        slice_key = fact_slice_key(
            session_id,
            cursor_before_message_id=since_message_id,
            cursor_before_partial_message_id=partial_message_id,
            cursor_before_offset=start_offset,
            cursor_after_message_id=message.message_id,
            cursor_after_partial_message_id=None,
            cursor_after_offset=0,
            occurrences=occurrences,
        )
        return FactsExtraction(
            items=[], start_message_id=message.message_id,
            covered_message_id=message.message_id,
            cursor_before_message_id=since_message_id,
            cursor_before_partial_message_id=partial_message_id,
            cursor_before_offset=start_offset,
            slice_key=slice_key, input_hash=fact_input_hash(rendered),
            source_occurrences=occurrences,
            caught_up=message.message_id == facts_tail_message_id(conn, session_id),
            publication_version=facts_config_version(cfg),
        )

    max_chars = max(
        1,
        int(cfg.dream_digest_max_chars if max_chars is None else max_chars),
    )
    lines: list[str] = []
    used_messages: list[CoveredMessage] = []
    covered = since_message_id
    next_partial = partial_message_id
    next_offset = start_offset
    used = 0
    for message in messages:
        offset = start_offset if message.message_id == partial_message_id else 0
        if offset >= len(message.content):
            raise RuntimeError("facts cursor offset is outside its source turn")
        prefix = f"{message.role}: "
        if not lines and max_chars <= len(prefix):
            raise ValueError("facts max_chars cannot hold source framing")
        separator_cost = 1 if lines else 0
        room = max_chars - used - separator_cost - len(prefix)
        if room <= 0 and lines:
            break
        room = max(1, room)
        remaining = len(message.content) - offset
        # Once at least one complete turn is present, never spend a leftover
        # handful of characters by splitting the next turn. It belongs wholly
        # to the next semantic unit. Only a first/continued oversized turn may
        # be partitioned.
        if lines and remaining > room:
            break
        fragment_end = (
            _bounded_fact_fragment_end(message.content, offset, room)
            if remaining > room else len(message.content)
        )
        fragment = message.content[offset:fragment_end]
        lines.append(prefix + fragment)
        used += separator_cost + len(prefix) + len(fragment)
        used_messages.append(message)
        if offset + len(fragment) < len(message.content):
            next_partial = message.message_id
            next_offset = offset + len(fragment)
            break
        covered = message.message_id
        next_partial = None
        next_offset = 0
        if used >= max_chars:
            break

    combined = "\n".join(lines)
    occurrences = combine_source_occurrences((
        tuple(_bound_occurrence(conn, message) for message in used_messages),
    ))
    slice_key = fact_slice_key(
        session_id,
        cursor_before_message_id=since_message_id,
        cursor_before_partial_message_id=partial_message_id,
        cursor_before_offset=start_offset,
        cursor_after_message_id=covered,
        cursor_after_partial_message_id=next_partial,
        cursor_after_offset=next_offset,
        occurrences=occurrences,
    )
    common = dict(
        start_message_id=used_messages[0].message_id,
        cursor_before_message_id=since_message_id,
        cursor_before_partial_message_id=partial_message_id,
        cursor_before_offset=start_offset,
        slice_key=slice_key,
        input_hash=fact_input_hash(combined),
        source_occurrences=occurrences,
        publication_version=facts_config_version(cfg),
    )
    request = LLMRequest(
        system=FACTS_SYSTEM,
        user=FACTS_USER_TEMPLATE.format(text=combined),
        response_format="json",
        max_tokens=cfg.dream_digest_max_tokens,
    )
    raw = llm.complete(request)
    items = validate_fact_items(raw, max_items=cfg.dream_max_facts_per_session)
    if items is None:
        return FactsExtraction(parse_failed=True, **common)
    tail = facts_tail_message_id(conn, session_id)
    return FactsExtraction(
        items=canonical_fact_items(items),
        covered_message_id=covered,
        partial_message_id=next_partial,
        next_message_offset=next_offset,
        caught_up=next_partial is None and covered == tail,
        **common,
    )


def validate_fact_items(raw: object, *, max_items: int) -> list[dict] | None:
    """Validate a complete reply; valid ``[]`` is distinct from lossy output."""
    if isinstance(raw, str):
        raw = loads_exact_or_fenced(raw)
    if isinstance(raw, dict) and set(raw) == {"facts"}:
        raw = raw["facts"]
    if not isinstance(raw, list) or len(raw) > max_items:
        return None
    out: list[dict] = []
    for item in raw:
        if (
            not isinstance(item, dict)
            or "text" not in item
            or not set(item).issubset({"text", "date", "entities"})
        ):
            return None
        raw_text = item.get("text")
        if not isinstance(raw_text, str):
            return None
        text = raw_text.strip()
        if not text or len(text) > _MAX_FACT_CHARS:
            return None
        raw_date = item.get("date")
        if raw_date is None:
            date = ""
        elif not isinstance(raw_date, str):
            return None
        else:
            date = raw_date.strip()
            if not _ISO_DATE.match(date):
                return None
            try:
                calendar_date.fromisoformat(date)
            except ValueError:
                return None
        raw_entities = item.get("entities", [])
        if (
            not isinstance(raw_entities, list)
            or len(raw_entities) > FACT_MAX_ENTITIES_PER_ITEM
        ):
            return None
        entities: list[str] = []
        for value in raw_entities:
            if (
                not isinstance(value, str) or not value.strip()
                or len(value.strip()) > FACT_MAX_ENTITY_CHARS
            ):
                return None
            canonical = normalize(value)
            if not canonical or len(canonical) > FACT_MAX_ENTITY_CHARS:
                return None
            if canonical not in entities:
                entities.append(canonical)
        out.append({"text": text, "date": date or None, "entities": entities})
    return out


def reextract_fact_outcome(
    conn: sqlite3.Connection,
    slice_key: str,
    llm: LLMClient,
    cfg: HyMemConfig,
    *,
    _require_committed_chain: bool = True,
) -> FactsExtraction:
    """Replay one published unit with its original exact source boundaries."""

    occurrences = load_fact_outcome_source_manifest(
        conn, slice_key, verify_result=True,
        _require_committed_chain=_require_committed_chain,
    )
    outcome = conn.execute(
        "SELECT * FROM fact_extraction_outcomes WHERE slice_key=?", (slice_key,)
    ).fetchone()
    if outcome is None or occurrences is None:
        raise RuntimeError("cannot replay a non-authoritative fact outcome")
    proofs: list[CoveredMessage] = []
    for occurrence in occurrences:
        proofs.append(validate_message_coverage_artifact(
            conn,
            message_id=occurrence.message_id,
            chunk_id=occurrence.coverage_chunk_id,
            coverage_version=occurrence.coverage_version,
        ))
    rendered = _render_fact_slice(
        proofs,
        before_partial_message_id=outcome["cursor_before_partial_message_id"],
        before_offset=int(outcome["cursor_before_offset"]),
        after_partial_message_id=outcome["cursor_after_partial_message_id"],
        after_offset=int(outcome["cursor_after_offset"]),
    )
    if fact_input_hash(rendered) != outcome["input_hash"]:
        raise RuntimeError("fact replay input no longer matches its authority unit")
    source_fragments: list[str] = []
    for proof in proofs:
        start = (
            int(outcome["cursor_before_offset"])
            if proof.message_id == outcome["cursor_before_partial_message_id"]
            else 0
        )
        end = (
            int(outcome["cursor_after_offset"])
            if proof.message_id == outcome["cursor_after_partial_message_id"]
            else len(proof.content)
        )
        source_fragments.append(proof.content[start:end])
    common = dict(
        start_message_id=min(source.message_id for source in occurrences),
        covered_message_id=outcome["cursor_after_message_id"],
        cursor_before_message_id=outcome["cursor_before_message_id"],
        cursor_before_partial_message_id=outcome[
            "cursor_before_partial_message_id"
        ],
        cursor_before_offset=int(outcome["cursor_before_offset"]),
        partial_message_id=outcome["cursor_after_partial_message_id"],
        next_message_offset=int(outcome["cursor_after_offset"]),
        slice_key=slice_key,
        input_hash=outcome["input_hash"],
        source_occurrences=occurrences,
        publication_version=facts_config_version(cfg),
        expected_generation=int(outcome["generation"]),
    )
    if all(not fragment.strip() for fragment in source_fragments):
        return FactsExtraction(items=[], **common)
    raw = llm.complete(LLMRequest(
        system=FACTS_SYSTEM,
        user=FACTS_USER_TEMPLATE.format(text=rendered),
        response_format="json",
        max_tokens=cfg.dream_digest_max_tokens,
    ))
    items = validate_fact_items(raw, max_items=cfg.dream_max_facts_per_session)
    if items is None:
        return FactsExtraction(parse_failed=True, **common)
    return FactsExtraction(items=canonical_fact_items(items), **common)


def _fact_row_item(row: sqlite3.Row) -> dict:
    try:
        entities = json.loads(row["entities"])
    except (TypeError, json.JSONDecodeError):
        raise RuntimeError("authoritative fact has malformed entities") from None
    if not isinstance(entities, list) or any(not isinstance(v, str) for v in entities):
        raise RuntimeError("authoritative fact has malformed entities")
    return {"text": row["text"], "date": row["fact_date"], "entities": entities}


def _committed_fact_slice_keys(
    conn: sqlite3.Connection, session_id: str
) -> bool | None:
    """Stream-validate the unique cursor-committed chain, or fail closed.

    A successful result means every outcome owned by the session occurs once
    on the chain ending at its durable cursor.  Callers already loaded a
    candidate outcome by primary key, so retaining every slice key in memory
    would add no authority and would let a hostile store force an unbounded
    Python allocation.
    """

    session = conn.execute(
        "SELECT facts_message_id,facts_cursor_message_id,"
        "facts_cursor_partial_message_id,facts_cursor_offset,"
        "facts_cursor_prompt_version FROM sessions WHERE id=?",
        (session_id,),
    ).fetchone()
    if session is None:
        return None
    offset = session["facts_cursor_offset"]
    if (
        not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
        or session["facts_message_id"] != session["facts_cursor_message_id"]
        or (session["facts_cursor_partial_message_id"] is None) != (offset == 0)
    ):
        return None
    terminal = (
        session["facts_cursor_message_id"],
        session["facts_cursor_partial_message_id"], int(offset),
    )
    rows = conn.execute(
        "SELECT slice_key,cursor_before_message_id,"
        "cursor_before_partial_message_id,cursor_before_offset,"
        "cursor_after_message_id,cursor_after_partial_message_id,"
        "cursor_after_offset,source_manifest_complete "
        "FROM fact_extraction_outcomes WHERE session_id=? "
        "ORDER BY COALESCE(cursor_before_partial_message_id,"
        "cursor_before_message_id,-1),"
        "CASE WHEN cursor_before_partial_message_id IS NULL THEN 1 ELSE 0 END,"
        "cursor_before_offset,slice_key",
        (session_id,),
    )
    coordinate: tuple[object, object, int] = (None, None, 0)
    row_count = 0
    for row in rows:
        before_offset = row["cursor_before_offset"]
        after_offset = row["cursor_after_offset"]
        if (
            row["source_manifest_complete"] != 1
            or not isinstance(before_offset, int) or isinstance(before_offset, bool)
            or not isinstance(after_offset, int) or isinstance(after_offset, bool)
            or before_offset < 0 or after_offset < 0
            or (row["cursor_before_partial_message_id"] is None)
            != (before_offset == 0)
            or (row["cursor_after_partial_message_id"] is None)
            != (after_offset == 0)
        ):
            return None
        before = (
            row["cursor_before_message_id"],
            row["cursor_before_partial_message_id"], int(before_offset),
        )
        if before != coordinate:
            return None
        slice_key = row["slice_key"]
        if not isinstance(slice_key, str) or not slice_key:
            return None
        coordinate = (
            row["cursor_after_message_id"],
            row["cursor_after_partial_message_id"],
            int(row["cursor_after_offset"]),
        )
        row_count += 1
    if coordinate != terminal:
        return None
    if row_count and not facts_generation_is_recognized(
        session["facts_cursor_prompt_version"]
    ):
        return None
    return True


def load_fact_outcome_source_manifest(
    conn: sqlite3.Connection,
    slice_key: str,
    *,
    verify_result: bool = True,
    chain_cache: dict[str, bool | None] | None = None,
    _require_committed_chain: bool = True,
) -> tuple[BoundSourceOccurrence, ...] | None:
    """Return an outcome's exact sources only when every durable invariant holds."""
    try:
        outcome = conn.execute(
            "SELECT * FROM fact_extraction_outcomes WHERE slice_key=?", (slice_key,)
        ).fetchone()
        if (
            outcome is None
            or outcome["source_manifest_complete"] != 1
            or outcome["source_manifest_version"] != FACT_SOURCE_MANIFEST_VERSION
            or not isinstance(outcome["source_manifest_count"], int)
            or isinstance(outcome["source_manifest_count"], bool)
            or int(outcome["source_manifest_count"]) <= 0
            or int(outcome["source_manifest_count"])
            > FACT_MAX_SOURCES_PER_OUTCOME
        ):
            return None
        session_id = outcome["session_id"]
        if _require_committed_chain:
            committed_cache = chain_cache if chain_cache is not None else {}
            if session_id not in committed_cache:
                committed_cache[session_id] = _committed_fact_slice_keys(
                    conn, session_id
                )
            if committed_cache[session_id] is not True:
                return None
        rows = conn.execute(
            "SELECT * FROM fact_extraction_source_occurrences "
            "WHERE slice_key=? ORDER BY ordinal LIMIT ?",
            (slice_key, FACT_MAX_SOURCES_PER_OUTCOME + 1),
        ).fetchall()
        count = int(outcome["source_manifest_count"])
        if len(rows) != count or any(
            row["ordinal"] != expected for expected, row in enumerate(rows)
        ):
            return None
        occurrences: list[BoundSourceOccurrence] = []
        proofs: list[CoveredMessage] = []
        for row in rows:
            proof = validate_message_coverage_artifact(
                conn,
                message_id=row["source_message_id"],
                chunk_id=row["source_coverage_chunk_id"],
                coverage_version=row["source_coverage_version"],
            )
            occurrence = BoundSourceOccurrence(
                message_id=proof.message_id,
                session_id=proof.session_id,
                role=proof.role,
                source_peer_id=proof.source_peer_id,
                source_workspace_id=proof.source_workspace_id,
                source_created_at=proof.source_created_at,
                coverage_chunk_id=proof.chunk_id,
                coverage_version=row["source_coverage_version"],
                content_hash=row["source_content_hash"],
            )
            stored_tuple = (
                row["source_session_id"], row["source_role"],
                row["source_peer_id"], row["source_workspace_id"],
                row["source_created_at"],
            )
            proof_tuple = (
                proof.session_id, proof.role, proof.source_peer_id,
                proof.source_workspace_id, proof.source_created_at,
            )
            hash_row = conn.execute(
                "SELECT message_content_hash FROM message_retention_coverage "
                "WHERE message_id=? AND chunk_id=? AND coverage_version=?",
                (
                    proof.message_id, proof.chunk_id,
                    row["source_coverage_version"],
                ),
            ).fetchone()
            if (
                stored_tuple != proof_tuple
                or row["source_content_hash"]
                != (hash_row["message_content_hash"] if hash_row else None)
                or row["source_coverage_version"] != LOSSLESS_COVERAGE_VERSION
                or proof.session_id != outcome["session_id"]
            ):
                return None
            occurrences.append(occurrence)
            proofs.append(proof)
        canonical = tuple(occurrences)
        if combine_source_occurrences((canonical,)) != canonical:
            return None
        if outcome["source_manifest_hash"] != source_manifest_hash(
            FACT_SOURCE_MANIFEST_VERSION, canonical
        ):
            return None
        expected_key = fact_slice_key(
            outcome["session_id"],
            cursor_before_message_id=outcome["cursor_before_message_id"],
            cursor_before_partial_message_id=outcome["cursor_before_partial_message_id"],
            cursor_before_offset=outcome["cursor_before_offset"],
            cursor_after_message_id=outcome["cursor_after_message_id"],
            cursor_after_partial_message_id=outcome["cursor_after_partial_message_id"],
            cursor_after_offset=outcome["cursor_after_offset"],
            occurrences=canonical,
        )
        if not _cursor_interval_is_exact(conn, outcome, proofs):
            return None
        rendered = _render_fact_slice(
            proofs,
            before_partial_message_id=outcome["cursor_before_partial_message_id"],
            before_offset=int(outcome["cursor_before_offset"]),
            after_partial_message_id=outcome["cursor_after_partial_message_id"],
            after_offset=int(outcome["cursor_after_offset"]),
        )
        if expected_key != slice_key or outcome["input_hash"] != fact_input_hash(rendered):
            return None
        if verify_result and not _fact_outcome_result_is_valid(
            conn, outcome, occurrences=canonical
        ):
            return None
        return canonical
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError, sqlite3.Error):
        return None


def _fact_outcome_result_is_valid(
    conn: sqlite3.Connection,
    outcome: sqlite3.Row,
    *,
    occurrences: tuple[BoundSourceOccurrence, ...],
) -> bool:
    generation = outcome["generation"]
    if (
        not isinstance(generation, int) or isinstance(generation, bool)
        or generation < 1 or generation > FACT_MAX_REVISIONS_PER_OUTCOME
    ):
        return False
    revisions = conn.execute(
        "SELECT generation,prompt_version,outcome_status,result_hash,succeeded_at "
        "FROM fact_extraction_revisions WHERE slice_key=? ORDER BY generation "
        "LIMIT ?",
        (outcome["slice_key"], FACT_MAX_REVISIONS_PER_OUTCOME + 1),
    ).fetchall()
    if (
        len(revisions) != generation
        or any(
            row["generation"] != expected
            for expected, row in enumerate(revisions, start=1)
        )
    ):
        return False
    previous_succeeded: str | None = None
    first_succeeded: str | None = None
    for revision in revisions:
        try:
            succeeded = normalize_iso_timestamp(
                revision["succeeded_at"], context="fact revision publication"
            )
        except ValueError:
            return False
        if (
            (previous_succeeded is not None and succeeded < previous_succeeded)
            or succeeded != revision["succeeded_at"]
            or revision["outcome_status"] not in ("success", "empty")
            or not fact_publication_version_is_recognized(
                revision["prompt_version"]
            )
            or not isinstance(revision["result_hash"], str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", revision["result_hash"])
            is None
        ):
            return False
        if first_succeeded is None:
            first_succeeded = succeeded
        previous_succeeded = succeeded
    coverage_times = conn.execute(
        "SELECT proof.created_at FROM fact_extraction_source_occurrences source "
        "JOIN message_retention_coverage proof ON "
        "proof.message_id=source.source_message_id AND "
        "proof.chunk_id=source.source_coverage_chunk_id AND "
        "proof.coverage_version=source.source_coverage_version "
        "WHERE source.slice_key=?",
        (outcome["slice_key"],),
    ).fetchall()
    try:
        latest_source_publication = max(
            normalize_iso_timestamp(
                row["created_at"], context="fact source publication"
            )
            for row in coverage_times
        )
    except (ValueError, TypeError):
        return False
    if first_succeeded is None or first_succeeded < latest_source_publication:
        return False
    latest_revision = revisions[-1]
    if any(
        latest_revision[field] != outcome[field]
        for field in (
            "generation", "prompt_version", "outcome_status", "result_hash",
            "succeeded_at",
        )
    ):
        return False
    revision_by_generation = {row["generation"]: row for row in revisions}
    rows = conn.execute(
        "SELECT * FROM narrative_facts WHERE source_outcome_key=? ORDER BY fact_key "
        "LIMIT ?",
        (outcome["slice_key"], FACT_MAX_HISTORY_ITEMS_PER_OUTCOME + 1),
    ).fetchall()
    if len(rows) > FACT_MAX_HISTORY_ITEMS_PER_OUTCOME:
        return False
    lifecycle_rows = conn.execute(
        "SELECT l.fact_id,l.generation,l.direction,l.event_at,l.prompt_version,"
        "l.result_hash,l.recorded_at FROM narrative_fact_lifecycle l "
        "JOIN narrative_facts f ON f.id=l.fact_id "
        "WHERE f.source_outcome_key=? ORDER BY l.fact_id,l.generation LIMIT ?",
        (
            outcome["slice_key"],
            FACT_MAX_LIFECYCLE_EVENTS_PER_OUTCOME + 1,
        ),
    ).fetchall()
    if len(lifecycle_rows) > (
        FACT_MAX_LIFECYCLE_EVENTS_PER_OUTCOME
    ):
        return False
    lifecycle_by_fact: dict[int, list[sqlite3.Row]] = {}
    for event in lifecycle_rows:
        lifecycle_by_fact.setdefault(int(event["fact_id"]), []).append(event)
    item_by_id: dict[int, dict] = {}
    events_by_generation: dict[int, list[tuple[int, int]]] = {}
    projected_active: set[int] = set()
    for row in rows:
        item = _fact_row_item(row)
        validated_item = validate_fact_items([item], max_items=1)
        if (
            validated_item is None
            or canonical_fact_items(validated_item) != [item]
            or row["fact_key"] != fact_item_key(item)
            or row["session_id"] != outcome["session_id"]
            or not isinstance(row["start_message_id"], int)
            or isinstance(row["start_message_id"], bool)
            or not isinstance(row["end_message_id"], int)
            or isinstance(row["end_message_id"], bool)
            or int(row["start_message_id"])
            != min(source.message_id for source in occurrences)
            or int(row["end_message_id"])
            != max(source.message_id for source in occurrences)
        ):
            return False
        fact_id = int(row["id"])
        item_by_id[fact_id] = item
        events = lifecycle_by_fact.get(fact_id, [])
        if (
            not events or events[0]["direction"] != 1
            or any(
                left["generation"] >= right["generation"]
                or left["direction"] == right["direction"]
                for left, right in zip(events, events[1:])
            )
        ):
            return False
        first_revision = revision_by_generation.get(events[0]["generation"])
        if first_revision is None:
            return False
        expected_valid_at = _initial_fact_valid_at(
            item, occurrences, normalize_iso_timestamp(
                first_revision["succeeded_at"],
                context="fact revision publication",
            )
        )
        for event in events:
            revision = revision_by_generation.get(event["generation"])
            try:
                event_time = normalize_iso_timestamp(
                    event["event_at"], context="fact lifecycle valid time"
                )
                recorded_at = normalize_iso_timestamp(
                    event["recorded_at"], context="fact lifecycle transaction time"
                )
            except ValueError:
                return False
            if (
                revision is None
                or event_time != event["event_at"]
                or recorded_at != event["recorded_at"]
                or event["prompt_version"] != revision["prompt_version"]
                or event["result_hash"] != revision["result_hash"]
                or recorded_at != normalize_iso_timestamp(
                    revision["succeeded_at"], context="fact revision publication"
                )
                or event_time != expected_valid_at
            ):
                return False
            events_by_generation.setdefault(int(event["generation"]), []).append(
                (fact_id, int(event["direction"]))
            )
        latest = events[-1]
        if (
            latest["generation"] != row["current_generation"]
            or row["prompt_version"] != events[0]["prompt_version"]
            or row["valid_at"] != expected_valid_at
        ):
            return False
        try:
            created_at = normalize_iso_timestamp(
                row["created_at"], context="fact publication"
            )
            first_recorded_at = normalize_iso_timestamp(
                events[0]["recorded_at"], context="fact lifecycle publication"
            )
        except ValueError:
            return False
        if created_at != row["created_at"] or created_at != first_recorded_at:
            return False
        active = row["lifecycle_status"] == "active" and row["invalid_at"] is None
        retracted = row["lifecycle_status"] == "retracted" and row["invalid_at"] is not None
        if not ((active and latest["direction"] == 1) or (
            retracted and latest["direction"] == -1
        )):
            return False
        if active and row["valid_at"] != latest["event_at"]:
            return False
        if retracted and row["invalid_at"] != latest["event_at"]:
            return False
        if active:
            projected_active.add(fact_id)

    folded_active: set[int] = set()
    for revision in revisions:
        for fact_id, direction in sorted(
            events_by_generation.get(int(revision["generation"]), ())
        ):
            if direction == 1:
                if fact_id in folded_active:
                    return False
                folded_active.add(fact_id)
            else:
                if fact_id not in folded_active:
                    return False
                folded_active.remove(fact_id)
        if len(folded_active) > FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME:
            return False
        generation_items = [item_by_id[fact_id] for fact_id in folded_active]
        expected_status = "success" if generation_items else "empty"
        if (
            revision["outcome_status"] != expected_status
            or revision["result_hash"] != fact_result_hash(generation_items)
        ):
            return False
    return bool(
        folded_active == projected_active
        and outcome["outcome_status"]
        == ("success" if folded_active else "empty")
        and outcome["result_hash"]
        == fact_result_hash(item_by_id[fact_id] for fact_id in folded_active)
    )


def _initial_fact_valid_at(
    item: dict,
    occurrences: tuple[BoundSourceOccurrence, ...],
    accepted_at: str,
) -> str:
    """Initial valid time is explicit fact date, else latest source event."""
    if item.get("date"):
        return normalize_iso_timestamp(
            f"{item['date']}T00:00:00.000Z", context="fact date"
        )
    candidates: list[str] = []
    for occurrence in occurrences:
        if occurrence.source_created_at is None:
            continue
        try:
            candidates.append(normalize_iso_timestamp(
                occurrence.source_created_at, context="fact source event"
            ))
        except ValueError:
            continue
    # Transaction/publication time belongs only in succeeded_at/recorded_at.
    # A genuinely timestamp-less legacy source has an explicit, deterministic
    # unknown valid coordinate rather than borrowing the wall clock.
    _ = accepted_at
    return max(candidates) if candidates else "0001-01-01T00:00:00.000Z"


def _fact_publication_clock(
    conn: sqlite3.Connection,
    occurrences: tuple[BoundSourceOccurrence, ...],
    *,
    prior_succeeded_at: object = None,
) -> str:
    """Monotonic logical transaction time for one fact revision.

    Wall clocks can move backwards and portable source proofs may legitimately
    have been published by a clock slightly ahead of this process.  The fact
    transaction coordinate therefore advances to the maximum of normalized
    SQLite ``now``, every cited proof publication, and the prior revision.
    """

    candidates = [normalize_iso_timestamp(
        conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')").fetchone()[0],
        context="fact publication clock",
    )]
    if prior_succeeded_at is not None:
        candidates.append(normalize_iso_timestamp(
            prior_succeeded_at, context="prior fact publication"
        ))
    for occurrence in occurrences:
        row = conn.execute(
            "SELECT created_at FROM message_retention_coverage "
            "WHERE message_id=? AND chunk_id=? AND coverage_version=?",
            (
                occurrence.message_id, occurrence.coverage_chunk_id,
                occurrence.coverage_version,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("fact source proof disappeared before publication")
        candidates.append(normalize_iso_timestamp(
            row["created_at"], context="fact source publication"
        ))
    return max(candidates)


def load_fact_source_manifest(
    conn: sqlite3.Connection, fact_id: int
) -> tuple[BoundSourceOccurrence, ...] | None:
    """Resolve one currently published fact to its exact source occurrences."""
    try:
        fact = conn.execute(
            "SELECT source_outcome_key,lifecycle_status,invalid_at "
            "FROM narrative_facts WHERE id=?", (int(fact_id),),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if (
        fact is None or fact["source_outcome_key"] is None
        or fact["lifecycle_status"] != "active" or fact["invalid_at"] is not None
    ):
        return None
    return load_fact_outcome_source_manifest(
        conn, fact["source_outcome_key"], verify_result=True
    )


def load_fact_source_manifests(
    conn: sqlite3.Connection,
    fact_ids: Sequence[int],
    *,
    outcome_cache: dict[
        str, tuple[BoundSourceOccurrence, ...] | None
    ] | None = None,
    chain_cache: dict[str, bool | None] | None = None,
) -> dict[int, tuple[BoundSourceOccurrence, ...]]:
    """Bulk-resolve current facts while proving each authority unit once."""

    ids = tuple(dict.fromkeys(
        int(value) for value in fact_ids
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ))
    if not ids:
        return {}
    cache = outcome_cache if outcome_cache is not None else {}
    committed_cache = chain_cache if chain_cache is not None else {}
    resolved: dict[int, tuple[BoundSourceOccurrence, ...]] = {}
    try:
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                "SELECT id,source_outcome_key,lifecycle_status,invalid_at "
                f"FROM narrative_facts WHERE id IN ({placeholders})",
                batch,
            ).fetchall()
            for row in rows:
                key = row["source_outcome_key"]
                if (
                    not isinstance(key, str) or not key
                    or row["lifecycle_status"] != "active"
                    or row["invalid_at"] is not None
                ):
                    continue
                if key not in cache:
                    cache[key] = load_fact_outcome_source_manifest(
                        conn, key, verify_result=True, chain_cache=committed_cache
                    )
                proof = cache[key]
                if proof is not None:
                    resolved[int(row["id"])] = proof
    except sqlite3.OperationalError:
        return {}
    return resolved


def _fact_cursor_accepts_successor(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    publication_version: str,
    before_message_id: int | None,
    before_partial_message_id: int | None,
    before_offset: int,
) -> bool:
    """Cheap write-path proof for the next append-only authority unit.

    Full-chain validation remains mandatory on every read/export boundary.
    Publication itself needs only prove that the durable cursor is unchanged,
    no sibling already starts there, and (except at the initial coordinate)
    the immediate predecessor is a complete, independently valid unit. This
    keeps processing N bounded slices O(N), while direct cursor/history damage
    still makes the resulting chain fail closed to every consumer.
    """

    session = conn.execute(
        "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
        "facts_cursor_offset,facts_cursor_prompt_version "
        "FROM sessions WHERE id=?", (session_id,),
    ).fetchone()
    coordinate = (
        before_message_id, before_partial_message_id, int(before_offset),
    )
    if session is None or (
        session["facts_cursor_message_id"],
        session["facts_cursor_partial_message_id"],
        int(session["facts_cursor_offset"] or 0),
    ) != coordinate:
        return False
    sibling = conn.execute(
        "SELECT 1 FROM fact_extraction_outcomes WHERE session_id=? "
        "AND cursor_before_message_id IS ? "
        "AND cursor_before_partial_message_id IS ? AND cursor_before_offset=? "
        "LIMIT 1",
        (session_id, *coordinate),
    ).fetchone()
    if sibling is not None:
        return False
    if coordinate == (None, None, 0):
        return conn.execute(
            "SELECT 1 FROM fact_extraction_outcomes WHERE session_id=? LIMIT 1",
            (session_id,),
        ).fetchone() is None
    # A prompt/config bump must replay the already-committed immutable units
    # before any new tail unit is appended.  Otherwise a direct publisher could
    # append B after an A chain and make the session-level B marker falsely
    # claim that A was replayed.  The replay index makes these extrema probes
    # O(log N), avoiding a full-chain walk on every append.
    if session["facts_cursor_prompt_version"] != publication_version:
        return False
    first_version = conn.execute(
        "SELECT prompt_version FROM fact_extraction_outcomes "
        "WHERE session_id=? AND source_manifest_complete=1 "
        "ORDER BY prompt_version LIMIT 1",
        (session_id,),
    ).fetchone()
    last_version = conn.execute(
        "SELECT prompt_version FROM fact_extraction_outcomes "
        "WHERE session_id=? AND source_manifest_complete=1 "
        "ORDER BY prompt_version DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if (
        first_version is None or last_version is None
        or first_version["prompt_version"] != publication_version
        or last_version["prompt_version"] != publication_version
    ):
        return False
    predecessors = conn.execute(
        "SELECT slice_key FROM fact_extraction_outcomes WHERE session_id=? "
        "AND cursor_after_message_id IS ? "
        "AND cursor_after_partial_message_id IS ? AND cursor_after_offset=? "
        "AND source_manifest_complete=1 LIMIT 2",
        (session_id, *coordinate),
    ).fetchall()
    return bool(
        len(predecessors) == 1
        and load_fact_outcome_source_manifest(
            conn, predecessors[0]["slice_key"], verify_result=True,
            _require_committed_chain=False,
        ) is not None
    )


def persist_facts(
    conn: sqlite3.Connection,
    session_id: str,
    extraction: FactsExtraction,
    *,
    max_items: int = 8,
    _defer_chain_audit: bool = False,
) -> int:
    """Atomically publish one successful source slice, including empty output."""
    if not conn.in_transaction:
        raise RuntimeError("persist_facts requires a caller-owned transaction")
    if extraction.parse_failed:
        raise ValueError("cannot persist a failed facts extraction")
    if (
        extraction.slice_key is None or extraction.input_hash is None
        or not extraction.source_occurrences or extraction.start_message_id is None
    ):
        raise ValueError("facts extraction has no exact source unit")
    occurrences = _validate_bound_occurrences(
        conn, session_id, extraction.source_occurrences
    )
    proofs = [
        validate_message_coverage_artifact(
            conn,
            message_id=occurrence.message_id,
            chunk_id=occurrence.coverage_chunk_id,
            coverage_version=occurrence.coverage_version,
        )
        for occurrence in occurrences
    ]
    cursor_contract = {
        "session_id": session_id,
        "cursor_before_message_id": extraction.cursor_before_message_id,
        "cursor_before_partial_message_id": (
            extraction.cursor_before_partial_message_id
        ),
        "cursor_before_offset": extraction.cursor_before_offset,
        "cursor_after_message_id": extraction.covered_message_id,
        "cursor_after_partial_message_id": extraction.partial_message_id,
        "cursor_after_offset": extraction.next_message_offset,
    }
    if not _cursor_interval_is_exact(conn, cursor_contract, proofs):
        raise ValueError("facts extraction skips or misframes a source occurrence")
    rendered = _render_fact_slice(
        proofs,
        before_partial_message_id=extraction.cursor_before_partial_message_id,
        before_offset=extraction.cursor_before_offset,
        after_partial_message_id=extraction.partial_message_id,
        after_offset=extraction.next_message_offset,
    )
    if extraction.input_hash != fact_input_hash(rendered):
        raise ValueError("facts extraction input hash mismatches exact source bytes")
    if extraction.start_message_id != occurrences[0].message_id:
        raise ValueError("facts extraction start does not match its first source")
    expected_slice_key = fact_slice_key(
        session_id,
        cursor_before_message_id=extraction.cursor_before_message_id,
        cursor_before_partial_message_id=extraction.cursor_before_partial_message_id,
        cursor_before_offset=extraction.cursor_before_offset,
        cursor_after_message_id=extraction.covered_message_id,
        cursor_after_partial_message_id=extraction.partial_message_id,
        cursor_after_offset=extraction.next_message_offset,
        occurrences=occurrences,
    )
    if expected_slice_key != extraction.slice_key:
        raise ValueError("facts slice identity does not match its sources")
    if len(extraction.items) > FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME:
        raise ValueError("facts extraction exceeds the authority-unit hard limit")
    validated_items = validate_fact_items(
        extraction.items,
        max_items=min(max_items, FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME),
    )
    if validated_items is None:
        raise ValueError("programmatic facts extraction is malformed or lossy")
    publication_version = extraction.publication_version or FACTS_PROMPT_VERSION
    if (
        not fact_publication_version_is_recognized(publication_version)
    ):
        raise ValueError("facts extraction has an invalid publication version")
    items = canonical_fact_items(validated_items)
    result_hash = fact_result_hash(items)
    status = "success" if items else "empty"
    existing = conn.execute(
        "SELECT * FROM fact_extraction_outcomes WHERE slice_key=?",
        (extraction.slice_key,),
    ).fetchone()
    accepted_at = _fact_publication_clock(
        conn, occurrences,
        prior_succeeded_at=(
            existing["succeeded_at"] if existing is not None else None
        ),
    )
    if existing is not None:
        if load_fact_outcome_source_manifest(
            conn, extraction.slice_key, verify_result=True,
            _require_committed_chain=not _defer_chain_audit,
        ) is None:
            raise RuntimeError("existing fact outcome is not authoritative")
        expected = {
            "session_id": session_id,
            "input_hash": extraction.input_hash,
            "cursor_before_message_id": extraction.cursor_before_message_id,
            "cursor_before_partial_message_id": extraction.cursor_before_partial_message_id,
            "cursor_before_offset": extraction.cursor_before_offset,
            "cursor_after_message_id": extraction.covered_message_id,
            "cursor_after_partial_message_id": extraction.partial_message_id,
            "cursor_after_offset": extraction.next_message_offset,
        }
        if any(existing[field] != expected[field] for field in expected):
            raise ValueError("facts replay changes an immutable source unit")
        exact_current_result = bool(
            existing["result_hash"] == result_hash
            and existing["prompt_version"] == publication_version
        )
        current_generation = int(existing["generation"])
        if extraction.expected_generation is None:
            # A concurrent first-pass worker may observe the winner only after
            # its own provider call. Identical output is an idempotent retry;
            # divergent output is not silently promoted to an authoritative
            # replay generation.
            if exact_current_result:
                return 0
            raise RuntimeError("fact source unit was concurrently published")
        if current_generation != extraction.expected_generation:
            # Retrying the exact revision that another worker just committed is
            # harmless. Any other generation/result combination is stale and
            # must be re-read before it can author another revision.
            if (
                current_generation == extraction.expected_generation + 1
                and exact_current_result
            ):
                return 0
            raise RuntimeError("fact replay generation changed concurrently")
        if exact_current_result:
            return 0
        if current_generation >= FACT_MAX_REVISIONS_PER_OUTCOME:
            raise ValueError("fact outcome exceeds the lifecycle history limit")
        generation = current_generation + 1
    else:
        if extraction.expected_generation is not None:
            raise RuntimeError("fact replay source unit disappeared concurrently")
        if not facts_generation_is_recognized(publication_version):
            raise ValueError("new fact units require a complete config generation")
        if not _fact_cursor_accepts_successor(
            conn, session_id,
            publication_version=publication_version,
            before_message_id=extraction.cursor_before_message_id,
            before_partial_message_id=extraction.cursor_before_partial_message_id,
            before_offset=int(extraction.cursor_before_offset),
        ):
            raise ValueError("new fact unit is not the session cursor successor")
        generation = 1

    activated = 0
    manifest_hash = source_manifest_hash(FACT_SOURCE_MANIFEST_VERSION, occurrences)
    with core_db.evidence_mutation(conn):
        if existing is None:
            conn.execute(
                "INSERT INTO fact_extraction_outcomes("
                "slice_key,session_id,prompt_version,input_hash,"
                "cursor_before_message_id,cursor_before_partial_message_id,"
                "cursor_before_offset,cursor_after_message_id,"
                "cursor_after_partial_message_id,cursor_after_offset,generation,"
                "outcome_status,result_hash,succeeded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    extraction.slice_key, session_id, publication_version,
                    extraction.input_hash, extraction.cursor_before_message_id,
                    extraction.cursor_before_partial_message_id,
                    extraction.cursor_before_offset, extraction.covered_message_id,
                    extraction.partial_message_id, extraction.next_message_offset,
                    generation, status, result_hash, accepted_at,
                ),
            )
            for ordinal, occurrence in enumerate(occurrences):
                conn.execute(
                    "INSERT INTO fact_extraction_source_occurrences("
                    "slice_key,ordinal,source_message_id,source_session_id,"
                    "source_role,source_peer_id,source_workspace_id,"
                    "source_created_at,source_coverage_chunk_id,"
                    "source_coverage_version,source_content_hash) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        extraction.slice_key, ordinal, occurrence.message_id,
                        occurrence.session_id, occurrence.role,
                        occurrence.source_peer_id, occurrence.source_workspace_id,
                        occurrence.source_created_at, occurrence.coverage_chunk_id,
                        occurrence.coverage_version, occurrence.content_hash,
                    ),
                )
            conn.execute(
                "UPDATE fact_extraction_outcomes SET source_manifest_version=?,"
                "source_manifest_count=?,source_manifest_hash=?,"
                "source_manifest_complete=1 WHERE slice_key=?",
                (
                    FACT_SOURCE_MANIFEST_VERSION, len(occurrences), manifest_hash,
                    extraction.slice_key,
                ),
            )
            conn.execute(
                "INSERT INTO fact_extraction_revisions("
                "slice_key,generation,prompt_version,outcome_status,result_hash,"
                "succeeded_at) VALUES (?,?,?,?,?,?)",
                (
                    extraction.slice_key, generation, publication_version,
                    status, result_hash, accepted_at,
                ),
            )

        desired = {fact_item_key(item): item for item in items}
        current_rows = conn.execute(
            "SELECT * FROM narrative_facts WHERE source_outcome_key=? ORDER BY id "
            "LIMIT ?",
            (extraction.slice_key, FACT_MAX_HISTORY_ITEMS_PER_OUTCOME + 1),
        ).fetchall()
        if len(current_rows) > FACT_MAX_HISTORY_ITEMS_PER_OUTCOME:
            raise RuntimeError("fact outcome history exceeds its payload limit")
        by_key = {str(row["fact_key"]): row for row in current_rows}
        for key, row in by_key.items():
            if row["lifecycle_status"] == "active" and key not in desired:
                conn.execute(
                    "INSERT INTO narrative_fact_lifecycle("
                    "fact_id,generation,direction,event_at,prompt_version,"
                    "result_hash,recorded_at) VALUES (?,?,-1,?,?,?,?)",
                    (
                        row["id"], generation, row["valid_at"],
                        publication_version, result_hash, accepted_at,
                    ),
                )
                conn.execute(
                    "UPDATE narrative_facts SET lifecycle_status='retracted',"
                    "invalid_at=?,current_generation=? WHERE id=?",
                    (row["valid_at"], generation, row["id"]),
                )
        for key, item in desired.items():
            row = by_key.get(key)
            if row is None:
                initial_valid_at = _initial_fact_valid_at(
                    item, occurrences, accepted_at
                )
                cur = conn.execute(
                    "INSERT INTO narrative_facts("
                    "session_id,start_message_id,end_message_id,text,fact_date,"
                    "entities,prompt_version,valid_at,invalid_at,source_outcome_key,"
                    "fact_key,current_generation,lifecycle_status,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,NULL,?,?,?,'active',?)",
                    (
                        session_id, min(o.message_id for o in occurrences),
                        max(o.message_id for o in occurrences), item["text"],
                        item.get("date"), json.dumps(item.get("entities") or []),
                        publication_version, initial_valid_at, extraction.slice_key,
                        key, generation, accepted_at,
                    ),
                )
                fact_id = int(cur.lastrowid)
                conn.execute(
                    "INSERT INTO narrative_fact_lifecycle("
                    "fact_id,generation,direction,event_at,prompt_version,"
                    "result_hash,recorded_at) VALUES (?, ?, 1, ?, ?, ?, ?)",
                    (
                        fact_id, generation, initial_valid_at,
                        publication_version, result_hash, accepted_at,
                    ),
                )
                activated += 1
            else:
                stored_item = _fact_row_item(row)
                if stored_item != item:
                    raise RuntimeError("fact key collides with different immutable bytes")
                if row["lifecycle_status"] != "active":
                    conn.execute(
                        "INSERT INTO narrative_fact_lifecycle("
                        "fact_id,generation,direction,event_at,prompt_version,"
                        "result_hash,recorded_at) VALUES (?, ?, 1, ?, ?, ?, ?)",
                        (
                            row["id"], generation, row["valid_at"],
                            publication_version, result_hash, accepted_at,
                        ),
                    )
                    conn.execute(
                        "UPDATE narrative_facts SET lifecycle_status='active',"
                        "valid_at=?,invalid_at=NULL,current_generation=? WHERE id=?",
                        (row["valid_at"], generation, row["id"]),
                    )
                    activated += 1
        if existing is not None:
            conn.execute(
                "INSERT INTO fact_extraction_revisions("
                "slice_key,generation,prompt_version,outcome_status,result_hash,"
                "succeeded_at) VALUES (?,?,?,?,?,?)",
                (
                    extraction.slice_key, generation, publication_version,
                    status, result_hash, accepted_at,
                ),
            )
            conn.execute(
                "UPDATE fact_extraction_outcomes SET prompt_version=?,generation=?,"
                "outcome_status=?,result_hash=?,succeeded_at=? WHERE slice_key=?",
                (
                    publication_version, generation, status, result_hash,
                    accepted_at, extraction.slice_key,
                ),
            )
        else:
            advanced = conn.execute(
                "UPDATE sessions SET facts_message_id=?,"
                "facts_cursor_message_id=?,facts_cursor_partial_message_id=?,"
                "facts_cursor_offset=?,facts_cursor_prompt_version=?,"
                "facts_retry_count=0,facts_retry_config_version=NULL,"
                "facts_quarantined=0 WHERE id=? "
                "AND facts_cursor_message_id IS ? "
                "AND facts_cursor_partial_message_id IS ? "
                "AND facts_cursor_offset=?",
                (
                    extraction.covered_message_id,
                    extraction.covered_message_id, extraction.partial_message_id,
                    extraction.next_message_offset, publication_version,
                    session_id, extraction.cursor_before_message_id,
                    extraction.cursor_before_partial_message_id,
                    extraction.cursor_before_offset,
                ),
            )
            if advanced.rowcount != 1:
                raise RuntimeError("fact cursor changed during publication")
    if load_fact_outcome_source_manifest(
        conn, extraction.slice_key, verify_result=True,
        _require_committed_chain=(existing is not None and not _defer_chain_audit),
    ) is None:
        raise RuntimeError("published fact outcome failed validation")
    return activated


def facts_valid_at(
    conn: sqlite3.Connection, valid_time: str, *, session_id: str | None = None
) -> list[dict]:
    """Read authoritative facts whose latest lifecycle event at time was open."""
    target = normalize_iso_timestamp(valid_time, context="fact valid time")
    params: list[object] = [target]
    scope = ""
    if session_id is not None:
        scope = "AND f.session_id=?"
        params.append(session_id)
    rows = conn.execute(
        f"""
        SELECT f.*, l.direction, l.event_at
        FROM narrative_facts f
        JOIN narrative_fact_lifecycle l ON l.fact_id=f.id
        WHERE f.source_outcome_key IS NOT NULL
          AND l.event_at <= ? {scope}
          AND NOT EXISTS (
              SELECT 1 FROM narrative_fact_lifecycle later
              WHERE later.fact_id=l.fact_id AND later.event_at <= ?
                AND (later.event_at > l.event_at OR
                     (later.event_at = l.event_at AND later.generation > l.generation))
          )
        ORDER BY l.event_at,f.id
        """,
        (*params, target),
    ).fetchall()
    result: list[dict] = []
    chain_cache: dict[str, bool | None] = {}
    for row in rows:
        if row["direction"] != 1 or load_fact_outcome_source_manifest(
            conn, row["source_outcome_key"], verify_result=True,
            chain_cache=chain_cache,
        ) is None:
            continue
        result.append(dict(row))
    return result
