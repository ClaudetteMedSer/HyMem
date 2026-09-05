"""Typed user-profile extraction and consumption (Stage 1 / P4, schema v18).

The knowledge graph's 18-predicate vocabulary is tech-domain, so durable
personal facts ("user is a bedrijfsarts in Amsterdam named Atta") never become
edges — the digest stays honest but identity-thin, and incidental personal
facts are unreachable at query time. This module closes that gap with a CLOSED
slot vocabulary extracted from USER turns only: the schema-constrained, safe
version of the shelved open-ended incidental extraction. Unknown slots are
rejected twice — at :func:`validate_profile_items` and by the ``user_profile``
table's CHECK constraint — so the LLM can never invent a slot.

Rows are bi-temporal like the knowledge graph (valid_at / invalid_at, the v15
semantics in :mod:`hymem.dreaming.bitemporal`):

  - ``valid_at``   opens when a fact is asserted — the evidence message's
                   parseable ``created_at`` world date. Unknown legacy dates
                   use one fixed historical sentinel, never replay wall time.
  - ``invalid_at`` closes when a conflicting value supersedes the row — the
                   immediate successor's normalized world date (or the same
                   deterministic legacy fallback).

Supersession policy: single-valued slots (name, role, employer, location,
age_birthday) hold one active value — a new conflicting value closes the old
row. ``relationship`` is multi-valued but keyed per person (slot_key), so a
new value for the SAME person supersedes while different people accumulate.
The remaining multi-valued slots (language, possession, health_condition,
recurring_activity) accumulate. Exclusive temporal slots retain one assertion
per durable source even when two separated assertions have the same value;
this preserves an A→B→A history under arbitrary import order. Re-asserting a
multi-valued value reinforces confidence and is otherwise a no-op.

Redaction is enforced before durable staging and again at the persistence,
import, and strict-store-open boundaries. Both values and relationship keys
are scrubbed, so every consumer — the digest's VERIFIED FACTS anchor,
augment()'s ``ctx.user_profile`` tier, and ``HyMem.profile()`` — inherits safe
text. This matters for sensitive slots like ``health_condition``, whose values
must never resurface embedded secrets/PII the LLM lifted from context.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from hymem import redaction
from hymem.dreaming.lossless import CoveredMessage, covered_messages_after
from hymem.extraction.jsonio import loads_exact_or_fenced
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import USER_PROFILE_SYSTEM, USER_PROFILE_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.user_profile")

# Pinned prompt version, following the salt convention in aggregate.py
# (cluster.v2 / root.v4): bump when USER_PROFILE_SYSTEM / the user template
# change materially so operators can attribute extraction shifts to the prompt.
# profile.v2: aboutness test + per-slot tightening (health_condition, employer,
# possession) after profile.v1 failed the on-box precision gate at ~8%.
# profile.v3: named-object response contract. A bare/scanned ``[]`` could turn
# refusal prose or an unrelated envelope into an authoritative empty.
# profile.v4: resumable windows include bounded preceding context so a fact
# crossing an exact cursor boundary is not converted into two valid empties.
PROFILE_PROMPT_VERSION = "profile.v4"
PROFILE_STREAM_VERSION = "lossless-profile-v2"
_PROFILE_CONFIG_PATTERN = (
    rf"{re.escape(PROFILE_STREAM_VERSION)}\|"
    rf"prompt={re.escape(PROFILE_PROMPT_VERSION)}\|"
    r"chars=[1-9]\d*\|items=(?:0|[1-9]\d*)\|redact=[01]"
)
_PROFILE_GENERATION_RE = re.compile(
    _PROFILE_CONFIG_PATTERN + r"\|walk=[0-9a-f]{32}"
)
_PROFILE_RETRY_RE = re.compile(
    rf"(?P<config>{_PROFILE_CONFIG_PATTERN})\|"
    r"retry-max=(?P<maximum>-?\d+)\|"
    r"(?P<mode>forward|rebuild=.+;stamp=.+)"
)

# The CLOSED slot vocabulary, in the priority order consumers render
# (identity first). Must stay in lockstep with the CHECK constraint in
# migration 018_user_profile.sql / schema.sql.
PROFILE_SLOTS: tuple[str, ...] = (
    "name",
    "role",
    "employer",
    "location",
    "age_birthday",
    "language",
    "relationship",
    "possession",
    "health_condition",
    "recurring_activity",
)
_SLOT_ORDER = {slot: i for i, slot in enumerate(PROFILE_SLOTS)}
_SLOTS_SET = frozenset(PROFILE_SLOTS)

# Slots that hold exactly one active value: a new conflicting value closes the
# old row's validity interval. 'relationship' behaves the same but PER slot_key
# (one active value per person); the remaining multi-valued slots accumulate.
SINGLE_VALUED_SLOTS = frozenset({"name", "role", "employer", "location", "age_birthday"})


@dataclass
class ProfileExtraction:
    """Outcome of one bounded profile slice.

    ``None`` at the call boundary means there was no new USER input and no LLM
    call was made.  ``failed`` is deliberately separate from ``items``: an
    invalid or truncated answer may also contain valid-looking items, but none
    of those are publishable because doing so would make a partial response
    indistinguishable from an authoritative extraction.  A valid ``[]`` is a
    successful empty and advances the cursor.
    """

    items: list[dict] = field(default_factory=list)
    failed: bool = False
    failure_reason: str | None = None
    input_items: int = 0
    rejected_items: int = 0
    covered_message_id: int | None = None
    partial_message_id: int | None = None
    next_message_offset: int = 0
    start_message_id: int | None = None
    start_message_offset: int = 0
    end_message_id: int | None = None
    cursor_before_message_id: int | None = None
    cursor_before_partial_message_id: int | None = None
    cursor_before_offset: int = 0
    caught_up: bool = False
    slice_key: str | None = None


@dataclass(frozen=True)
class ProfileValidationOutcome:
    items: list[dict]
    input_items: int
    rejected_items: int
    capped: bool = False


@dataclass(frozen=True)
class ProfileEntry:
    """One ACTIVE (invalid_at IS NULL) user-profile row, as returned by
    `HyMem.profile()` and the `ctx.user_profile` augment tier. `slot_key` is
    the parameterizing key for keyed slots (relationship → the other person);
    None for unkeyed slots. `valid_at` is the world date the fact became true
    (bi-temporal v15 semantics)."""

    slot: str
    slot_key: str | None
    value: str
    confidence: float
    evidence_message_id: int | None
    valid_at: str


def fetch_user_turns(conn: sqlite3.Connection, session_id: str) -> list[tuple[int, str]]:
    """The session's USER turns as (message_id, content), in conversation
    order. User turns ONLY — the profile must never assert a fact the
    assistant introduced (a confabulated identity would otherwise launder
    itself into ground truth via the anchor)."""
    rows = conn.execute(
        "SELECT id, content FROM messages "
        "WHERE session_id = ? AND role = 'user' ORDER BY id",
        (session_id,),
    ).fetchall()
    return [(int(r["id"]), r["content"]) for r in rows]


def build_profile_user_prompt(turns: list[tuple[int, str]], *, max_chars: int) -> str:
    """Render the EXACT first extraction call for a session's user turns.

    Shared with `benchmarks/profile_prompt_dump.py` so the manual front-run
    precision gate scores the very prompt production runs."""
    if not turns:
        return _wrap_profile_user_prompt("")
    messages = [
        CoveredMessage(
            message_id=int(mid), session_id="profile-prompt-probe", role="user",
            content=text, chunk_id=f"profile-prompt-probe-{mid}",
        )
        for mid, text in turns
    ]
    body, *_ = _build_profile_window(
        messages,
        since_message_id=None,
        since_message_offset=0,
        max_chars=max_chars,
        tail_message_id=int(turns[-1][0]),
    )
    return _wrap_profile_user_prompt(body)


def _wrap_profile_user_prompt(body: str) -> str:
    """Shared production/probe wrapper for the profile user prompt."""
    return USER_PROFILE_USER_TEMPLATE.format(text=body)


def _load_profile_contract(raw: object) -> object | None:
    """Parse exact JSON or a whole-response Markdown fence only.

    Unlike the general dreaming parser this never scans arbitrary prose for a
    JSON-looking substring. This result advances a durable cursor, so salvaging
    an example/denial such as ``I cannot comply. {"items": []}`` would be a
    permanent false empty.
    """
    return loads_exact_or_fenced(raw)


_PROFILE_SEPARATOR = "\n\n"
_USER_ROLE = frozenset({"user"})
_PROFILE_CONTEXT_CHARS = 48
_UNKNOWN_PROFILE_VALID_AT = "0001-01-01T00:00:00.000000+00:00"


def profile_config_version(
    *, max_chars: int, max_items: int, redact_values: bool = True
) -> str:
    """Stable configuration prefix for a resumable profile walk."""
    return (
        f"{PROFILE_STREAM_VERSION}|prompt={PROFILE_PROMPT_VERSION}|"
        f"chars={int(max_chars)}|items={int(max_items)}|"
        f"redact={int(bool(redact_values))}"
    )


def profile_generation_matches_config(generation: object, config: str) -> bool:
    """Whether *generation* is exactly a producer-issued walk for *config*."""
    return bool(
        isinstance(generation, str)
        and re.fullmatch(re.escape(config) + r"\|walk=[0-9a-f]{32}", generation)
    )


def profile_generation_is_recognized(generation: object) -> bool:
    """Whether *generation* has the complete current producer wire shape."""
    return bool(
        isinstance(generation, str)
        and _PROFILE_GENERATION_RE.fullmatch(generation)
    )


def profile_attempt_max_chars(configured_max: int, retry_count: int) -> int:
    """Adaptive exact-input bound for a retry of the same cursor position."""
    if configured_max <= 0:
        return configured_max
    floor = min(configured_max, 256)
    return max(floor, configured_max // (2 ** min(max(0, retry_count), 8)))


def profile_retry_policy_version(
    profile_config: str,
    *,
    max_attempts: int,
    rebuild_from: str | None = None,
    invalidated_stamp: str | None = None,
) -> str:
    """Key retry/quarantine state without changing the source cursor salt.

    Retry policy is deliberately separate from ``profile_config``: changing
    the failure bound (especially to zero) must reopen a quarantined source
    position without forcing an otherwise unnecessary profile replay. A
    same-config explicit rebuild gets its own stable key so stale failures from
    the completed generation cannot instantly quarantine the new attempt.
    """
    mode = (
        f"rebuild={rebuild_from or 'none'};stamp={invalidated_stamp or 'none'}"
        if rebuild_from is not None
        else "forward"
    )
    return f"{profile_config}|retry-max={int(max_attempts)}|{mode}"


def profile_retry_state_is_valid(
    retry_count: object,
    retry_config_version: object,
    quarantined: object,
) -> bool:
    """Validate one durable retry tuple without trusting its boolean flag."""
    if (
        isinstance(retry_count, bool)
        or not isinstance(retry_count, int)
        or retry_count < 0
        or isinstance(quarantined, bool)
        or not isinstance(quarantined, int)
        or quarantined not in (0, 1)
    ):
        return False
    if retry_count == 0:
        return retry_config_version is None and quarantined == 0
    if not isinstance(retry_config_version, str):
        return False
    match = _PROFILE_RETRY_RE.fullmatch(retry_config_version)
    if match is None:
        return False
    maximum = int(match.group("maximum"))
    return bool(quarantined) == bool(maximum > 0 and retry_count >= maximum)


def record_profile_failure(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    max_attempts: int,
    retry_config_version: str,
) -> bool:
    """Record a held failure and return whether the session is quarantined."""
    row = conn.execute(
        "SELECT profile_retry_count, profile_retry_config_version "
        "FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    attempts = (
        int(row["profile_retry_count"] or 0) + 1
        if row["profile_retry_config_version"] == retry_config_version
        else 1
    )
    quarantined = bool(max_attempts > 0 and attempts >= max_attempts)
    conn.execute(
        "UPDATE sessions SET profile_retry_count = ?, "
        "profile_retry_config_version = ?, profile_quarantined = ? "
        "WHERE id = ?",
        (attempts, retry_config_version, int(quarantined), session_id),
    )
    if quarantined:
        log.warning(
            "profile.extraction_quarantined session_id=%s attempts=%d "
            "cursor_advanced=0 partial_published=0",
            session_id,
            attempts,
        )
    return quarantined


def enforce_profile_redaction_policy(
    conn: sqlite3.Connection,
) -> int:
    """Make a False→True privacy transition local and immediate.

    Published active and historical values are scrubbed in place, and
    unpublished slices created under an unredacted generation are discarded
    before any external call. The caller owns the transaction.
    """
    changed = 0
    rows = conn.execute(
        "SELECT id, slot, value, slot_key FROM user_profile"
    ).fetchall()
    key_changes = {
        int(row["id"]): _redact_profile_key(row["slot_key"])
        for row in rows
        if _redact_profile_key(row["slot_key"]) != row["slot_key"]
    }
    if key_changes:
        # The identity-preserving pseudonym can intentionally merge legacy
        # spelling variants of the same secret. Remove the active-key guard
        # only inside the caller's write transaction, rewrite all assertions,
        # derive one winner per resulting key, then restore the guard before
        # returning. This also avoids row-order-dependent UNIQUE failures.
        conn.execute(
            "DROP INDEX IF EXISTS idx_user_profile_one_active_relationship"
        )
    for row in rows:
        safe = redaction.redact(row["value"])
        safe_key = key_changes.get(int(row["id"]), row["slot_key"])
        if safe != row["value"] or safe_key != row["slot_key"]:
            conn.execute(
                "UPDATE user_profile SET value = ?, slot_key = ? WHERE id = ?",
                (safe, safe_key, row["id"]),
            )
            changed += 1
    if key_changes:
        for key_row in conn.execute(
            "SELECT DISTINCT slot_key FROM user_profile "
            "WHERE slot = 'relationship'"
        ).fetchall():
            reconcile_profile_intervals(conn, "relationship", key_row["slot_key"])
        conn.execute(
            "CREATE UNIQUE INDEX idx_user_profile_one_active_relationship "
            "ON user_profile(slot, lower(trim(COALESCE(slot_key, '')))) "
            "WHERE invalid_at IS NULL AND slot = 'relationship'"
        )
    generations = conn.execute(
        "SELECT DISTINCT session_id, generation FROM profile_staging"
    ).fetchall()
    for stage in generations:
        generation = stage["generation"]
        if (
            not profile_generation_is_recognized(generation)
            or "|redact=1|" not in generation
        ):
            conn.execute(
                "DELETE FROM profile_staging "
                "WHERE session_id = ? AND generation = ?",
                (stage["session_id"], generation),
            )
            continue
        stage_rows = conn.execute(
            "SELECT slice_key, items_json FROM profile_staging "
            "WHERE session_id = ? AND generation = ?",
            (stage["session_id"], generation),
        ).fetchall()
        for staged in stage_rows:
            try:
                items = loads_exact_or_fenced(staged["items_json"])
            except (TypeError, ValueError):
                items = None
            if (
                not isinstance(items, list)
                or any(
                    not isinstance(item, dict)
                    or set(item) != (
                        {
                            "slot", "value", "evidence_message_id",
                            "confidence", "source_session_id",
                            "source_created_at",
                        }
                        | ({"slot_key"} if item.get("slot") == "relationship" else set())
                    )
                    or not isinstance(item.get("value"), str)
                    or (
                        item.get("slot_key") is not None
                        and not isinstance(item.get("slot_key"), str)
                    )
                    for item in items
                )
            ):
                conn.execute(
                    "DELETE FROM profile_staging WHERE session_id = ? "
                    "AND generation = ? AND slice_key = ?",
                    (stage["session_id"], generation, staged["slice_key"]),
                )
                continue
            for item in items:
                item["value"] = redaction.redact(item["value"])
                if item.get("slot") == "relationship":
                    item["slot_key"] = _redact_profile_key(item.get("slot_key"))
            conn.execute(
                "UPDATE profile_staging SET items_json = ? "
                "WHERE session_id = ? AND generation = ? AND slice_key = ?",
                (
                    json.dumps(items, ensure_ascii=False, sort_keys=True),
                    stage["session_id"], generation, staged["slice_key"],
                ),
            )
    return changed


def _render_profile_message_part(
    message: CoveredMessage,
    start: int,
    end: int,
    *,
    include_context: bool = True,
) -> str:
    external = (
        f" peer={message.source_peer_id} workspace={message.source_workspace_id}"
        if message.source_peer_id is not None
        else ""
    )
    current = (
        f"[msg {message.message_id}{external}] {message.content[start:end]}"
    )
    if start <= 0 or not include_context:
        return current
    context_start = max(0, start - _PROFILE_CONTEXT_CHARS)
    return (
        f"[previous context for msg {message.message_id} "
        f"range={context_start}:{start}/{len(message.content)}]\n"
        f"{message.content[context_start:start]}\n"
        f"{current}"
    )


def _largest_profile_part_end(
    message: CoveredMessage,
    start: int,
    budget: int,
    *,
    include_context: bool = True,
) -> int | None:
    """Largest exclusive Python-character offset whose framed part fits."""
    low, high = start, len(message.content)
    if len(_render_profile_message_part(
        message, start, start, include_context=include_context
    )) > budget:
        return None
    while low < high:
        mid = (low + high + 1) // 2
        if len(_render_profile_message_part(
            message, start, mid, include_context=include_context
        )) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def _render_profile_leading_context(message: CoveredMessage) -> str:
    start = max(0, len(message.content) - _PROFILE_CONTEXT_CHARS)
    external = (
        f" peer={message.source_peer_id} workspace={message.source_workspace_id}"
        if message.source_peer_id is not None
        else ""
    )
    return (
        f"[previous USER message context msg {message.message_id}{external} "
        f"range={start}:{len(message.content)}/{len(message.content)}]\n"
        f"{message.content[start:]}"
    )


def profile_user_tail_message_id(
    conn: sqlite3.Connection,
    session_id: str,
) -> int | None:
    """Highest USER id in the reviewed ordered coverage stream."""
    frontier = conn.execute(
        "SELECT coverage_message_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if frontier is None or frontier["coverage_message_id"] is None:
        return None
    row = conn.execute(
        """
        SELECT MAX(mc.message_id) AS m
        FROM message_retention_coverage mc
        JOIN chunks c ON c.id = mc.chunk_id
        WHERE mc.source_session_id = ?
          AND mc.source_role = 'user'
          AND mc.coverage_version = 'dream-lossless-message-v1'
          AND mc.message_id <= ?
          AND c.chunk_kind = 'coverage'
        """,
        (session_id, int(frontier["coverage_message_id"])),
    ).fetchone()
    return int(row["m"]) if row and row["m"] is not None else None


def _build_profile_window(
    messages: list[CoveredMessage],
    *,
    since_message_id: int | None,
    since_message_offset: int,
    max_chars: int,
    tail_message_id: int,
    leading_context: CoveredMessage | None = None,
) -> tuple[str, int | None, int | None, int, int | None, int | None, bool]:
    """Build one exact USER-only input slice and its next durable cursor."""
    if max_chars <= 0:
        raise ValueError("profile max_chars must be positive")
    if since_message_offset < 0:
        raise ValueError("profile cursor offset must be non-negative")

    parts: list[str] = []
    covered = since_message_id
    partial_id: int | None = None
    next_offset = 0
    started: int | None = None
    ended: int | None = None
    context = (
        _render_profile_leading_context(leading_context)
        if leading_context is not None else ""
    )
    used_chars = len(context)
    for index, message in enumerate(messages):
        start = since_message_offset if index == 0 else 0
        if start > len(message.content):
            raise RuntimeError(
                f"profile cursor offset {start} exceeds message "
                f"{message.message_id} length {len(message.content)}"
            )
        separator_cost = len(_PROFILE_SEPARATOR) if (parts or context) else 0
        part_budget = max_chars - used_chars - separator_cost
        include_context = bool(
            start > 0
            and len(_render_profile_message_part(
                message,
                start,
                min(len(message.content), start + 1),
                include_context=True,
            )) <= part_budget
        )
        end = _largest_profile_part_end(
            message, start, part_budget, include_context=include_context
        )
        if end is None and context and not parts:
            # Very small compatibility caps may fit framing plus new source
            # but not the optional cross-message context. Never let overlap
            # prevent cursor progress.
            context = ""
            used_chars = 0
            separator_cost = 0
            part_budget = max_chars
            include_context = bool(
                start > 0
                and len(_render_profile_message_part(
                    message,
                    start,
                    min(len(message.content), start + 1),
                    include_context=True,
                )) <= part_budget
            )
            end = _largest_profile_part_end(
                message, start, part_budget, include_context=include_context
            )
        if end is None or (end == start and start < len(message.content)):
            if parts:
                break
            raise ValueError(
                "dream_digest_max_chars is too small for lossless profile framing"
            )
        rendered = _render_profile_message_part(
            message, start, end, include_context=include_context
        )
        parts.append(rendered)
        used_chars += separator_cost + len(rendered)
        started = message.message_id if started is None else started
        ended = message.message_id
        if end == len(message.content):
            covered = message.message_id
            continue
        partial_id = message.message_id
        next_offset = end
        break

    caught_up = bool(
        next_offset == 0
        and covered is not None
        and int(covered) == int(tail_message_id)
    )
    body_parts = ([context] if context else []) + parts
    return (
        _PROFILE_SEPARATOR.join(body_parts),
        covered,
        partial_id,
        next_offset,
        started,
        ended,
        caught_up,
    )


def extract_user_profile(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
    *,
    max_chars: int,
    max_items: int,
    since_message_id: int | None = None,
    partial_message_id: int | None = None,
    since_message_offset: int = 0,
) -> ProfileExtraction | None:
    """Extract one bounded slice from the durable USER-only source stream.

    The raw ``messages`` table is intentionally not read: prompt rewinds must
    work after retention, and assistant/system/tool appends must not trigger a
    profile call.  ``since_message_offset`` resumes an oversized Unicode
    message at the exact Python-character boundary persisted by the runner.
    No write transaction is held while the LLM runs.
    """
    tail_message_id = profile_user_tail_message_id(conn, session_id)
    if tail_message_id is None:
        return None
    messages = covered_messages_after(
        conn,
        session_id,
        since_message_id,
        roles=_USER_ROLE,
        through_message_id=tail_message_id,
    )
    if not messages:
        if (
            since_message_offset == 0
            and partial_message_id is None
            and since_message_id is not None
            and int(since_message_id) >= tail_message_id
        ):
            return None
        raise RuntimeError(
            "profile coverage cursor has no readable USER artifact before its tail"
        )
    if since_message_offset:
        if (
            partial_message_id is None
            or messages[0].message_id != int(partial_message_id)
        ):
            raise RuntimeError("profile partial-message cursor does not match artifact")
    elif partial_message_id is not None:
        raise RuntimeError("partial message id requires a non-zero profile offset")

    leading_context: CoveredMessage | None = None
    if since_message_id is not None and since_message_offset == 0:
        prior = covered_messages_after(
            conn,
            session_id,
            int(since_message_id) - 1,
            limit=1,
            roles=_USER_ROLE,
            through_message_id=int(since_message_id),
        )
        if prior and prior[0].message_id == int(since_message_id):
            leading_context = prior[0]

    (
        body,
        covered,
        next_partial_id,
        next_offset,
        started,
        ended,
        caught_up,
    ) = _build_profile_window(
        messages,
        since_message_id=since_message_id,
        since_message_offset=since_message_offset,
        max_chars=max_chars,
        tail_message_id=tail_message_id,
        leading_context=leading_context,
    )
    input_messages = {
        message.message_id: message
        for message in messages
        if started is not None
        and ended is not None
        and started <= message.message_id <= ended
    }
    if leading_context is not None:
        input_messages[leading_context.message_id] = leading_context
    slice_key = (
        f"after={since_message_id if since_message_id is not None else 'start'};"
        f"partial={partial_message_id if partial_message_id is not None else 'none'};"
        f"offset={since_message_offset};cap={max_chars}"
    )

    request = LLMRequest(
        system=USER_PROFILE_SYSTEM,
        user=_wrap_profile_user_prompt(body),
        response_format="json",
        max_tokens=max(512, min(4096, 512 + max_items * 192)),
    )
    raw = llm.complete(request)

    # The named object is intentional: array-scanning could salvage ``[]`` out
    # of refusal prose or unwrap an unrelated one-list envelope and silently
    # advance the source cursor as a false empty.
    data = _load_profile_contract(raw)
    if data is None:
        log.warning("user_profile.parse_failure session_id=%s raw_len=%d",
                    session_id, len(raw) if isinstance(raw, str) else -1)
        return ProfileExtraction(failed=True, failure_reason="parse_failure")
    if (
        not isinstance(data, dict)
        or set(data) != {"items"}
        or not isinstance(data["items"], list)
    ):
        log.warning("user_profile.shape_failure session_id=%s type=%s",
                    session_id, type(data).__name__)
        return ProfileExtraction(failed=True, failure_reason="shape_failure")

    raw_items = data["items"]
    validation = validate_profile_items_outcome(
        raw_items, set(input_messages), max_items=max_items
    )
    if validation.capped or validation.rejected_items:
        reason = "output_cap" if validation.capped else "validation_failure"
        log.warning(
            "user_profile.%s session_id=%s returned=%d rejected=%d cap=%d",
            reason,
            session_id,
            validation.input_items,
            validation.rejected_items,
            max_items,
        )
        return ProfileExtraction(
            failed=True,
            failure_reason=reason,
            input_items=validation.input_items,
            rejected_items=validation.rejected_items,
        )

    enriched: list[dict] = []
    for item in validation.items:
        message = input_messages[item["evidence_message_id"]]
        clean = dict(item)
        clean["source_session_id"] = message.session_id
        clean["source_created_at"] = message.source_created_at
        enriched.append(clean)
    return ProfileExtraction(
        items=enriched,
        input_items=validation.input_items,
        covered_message_id=covered,
        partial_message_id=next_partial_id,
        next_message_offset=next_offset,
        start_message_id=started,
        start_message_offset=since_message_offset,
        end_message_id=ended,
        cursor_before_message_id=since_message_id,
        cursor_before_partial_message_id=partial_message_id,
        cursor_before_offset=since_message_offset,
        caught_up=caught_up,
        slice_key=slice_key,
    )


def validate_profile_items(
    data: object, valid_message_ids: set[int], *, max_items: int
) -> list[dict]:
    """Strictly validate raw LLM items into clean dicts ready to persist.

    Drops: non-dict items, slots outside the CLOSED vocabulary (the LLM must
    never mint a slot — the DB CHECK is the second line of defense), empty
    values, evidence_message_id missing or not among the input turns
    (hallucinated provenance), and confidence missing or outside [0, 1].
    slot_key is required for 'relationship' (supersession is keyed on it) and
    stripped from every other slot so unkeyed supersession stays deterministic.
    Returns [] for any non-list payload; at most `max_items` items survive."""
    if not isinstance(data, list):
        return []
    items: list[dict] = []
    for item in data:
        if len(items) >= max_items:
            break
        clean = _validate_profile_item(item, valid_message_ids)
        if clean is not None:
            items.append(clean)
    return items


def validate_profile_items_outcome(
    data: object,
    valid_message_ids: set[int],
    *,
    max_items: int,
) -> ProfileValidationOutcome:
    """Validate without silently laundering malformed/capped output.

    The compatibility helper above remains a filtering function for explicit
    callers.  Extraction uses this detailed outcome: if a non-empty item is
    invalid, or the provider returns more items than the configured bound, the
    entire response is retryable and no valid-looking subset is staged.
    """
    if not isinstance(data, list):
        return ProfileValidationOutcome([], 0, 0)
    if len(data) > max_items:
        return ProfileValidationOutcome(
            [], len(data), max(0, len(data) - max_items), capped=True
        )
    items: list[dict] = []
    rejected = 0
    seen_values: dict[tuple[str, str | None, int], str] = {}
    for item in data:
        clean = _validate_profile_item(item, valid_message_ids)
        if clean is None:
            rejected += 1
        else:
            identity = (
                clean["slot"], clean.get("slot_key"),
                clean["evidence_message_id"],
            )
            exclusive = (
                clean["slot"] in SINGLE_VALUED_SLOTS
                or clean["slot"] == "relationship"
            )
            prior = seen_values.get(identity) if exclusive else None
            if prior is not None and not _same_value(prior, clean["value"]):
                # One source cannot authoritatively assert two competing
                # values for the same profile key in one response. Response
                # order is not chronology, so fail the entire slice.
                rejected += 1
            else:
                if exclusive:
                    seen_values[identity] = clean["value"]
            items.append(clean)
    return ProfileValidationOutcome(items, len(data), rejected)


def _validate_profile_item(
    item: object,
    valid_message_ids: set[int],
) -> dict | None:
    if not isinstance(item, dict):
        return None
    slot = item.get("slot")
    if not isinstance(slot, str):
        return None
    slot = slot.strip().lower()
    if slot not in _SLOTS_SET:
        return None
    required = {"slot", "value", "evidence_message_id", "confidence"}
    expected = required | ({"slot_key"} if slot == "relationship" else set())
    if set(item) != expected:
        return None
    value = item.get("value")
    if not isinstance(value, str) or not value.strip():
        return None
    mid = item.get("evidence_message_id")
    if isinstance(mid, bool) or not isinstance(mid, int) or mid not in valid_message_ids:
        return None
    conf = item.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return None
    if not 0.0 <= conf <= 1.0:
        return None
    key = item.get("slot_key")
    if key is not None and not isinstance(key, str):
        return None
    # Casefold the key so "Anna"/"anna" supersede each other rather than
    # accumulating as two people.
    key = key.strip().lower() if key else None
    if slot == "relationship":
        if not key:
            return None
    else:
        key = None
    return {
        "slot": slot,
        "slot_key": key,
        "value": value.strip(),
        "evidence_message_id": mid,
        "confidence": float(conf),
    }


def _same_value(a: str, b: str) -> bool:
    """Conservative re-assertion check: casefold + collapse whitespace, nothing
    else — so any content difference counts as a new (possibly superseding)
    value, mirroring the _dedup_key philosophy in augment.py."""
    return " ".join(a.casefold().split()) == " ".join(b.casefold().split())


def _redact_profile_key(value: str | None) -> str | None:
    """Scrub a relationship identity without collapsing distinct secrets.

    The general redactor intentionally maps every email/key to the same public
    marker. Relationship slot keys are also database identity, however, so two
    different addresses would violate the active-key uniqueness guard during a
    strict-policy transition. A short one-way suffix keeps those identities
    stable and distinct without retaining the original secret.
    """
    if value is None:
        return None
    normalized = value.strip().lower()
    safe = redaction.redact(normalized)
    if safe == normalized:
        return normalized
    fingerprint = hashlib.sha256(
        " ".join(normalized.casefold().split()).encode("utf-8")
    ).hexdigest()[:12]
    return f"{safe.lower()}#{fingerprint}"


def stage_profile_extraction(
    conn: sqlite3.Connection,
    session_id: str,
    generation: str,
    extraction: ProfileExtraction,
    *,
    redact_values: bool = True,
) -> None:
    """Durably stage one successful slice without exposing partial output.

    Redaction happens *before* JSON reaches SQLite. A multi-pass walk may sit
    incomplete for days, so deferring scrubbing until publication would create
    a durable secret-bearing store outside the historical persist chokepoint.
    The caller commits this row and the matching cursor in one transaction.
    """
    if extraction.failed or extraction.slice_key is None:
        raise ValueError("only a successful positioned profile slice can be staged")
    safe_items: list[dict] = []
    for item in extraction.items:
        clean = dict(item)
        if clean.get("slot") != "relationship":
            clean.pop("slot_key", None)
        if redact_values:
            clean["value"] = redaction.redact(clean["value"])
            if clean.get("slot") == "relationship":
                clean["slot_key"] = _redact_profile_key(clean.get("slot_key"))
        safe_items.append(clean)
    conn.execute(
        """
        INSERT INTO profile_staging(
            session_id, generation, slice_key, items_json,
            start_message_id, start_message_offset, end_message_id,
            cursor_before_message_id, cursor_before_partial_message_id,
            cursor_before_offset, cursor_after_message_id,
            cursor_after_partial_message_id, cursor_after_offset
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, generation, slice_key) DO UPDATE SET
            items_json = excluded.items_json,
            start_message_id = excluded.start_message_id,
            start_message_offset = excluded.start_message_offset,
            end_message_id = excluded.end_message_id,
            cursor_before_message_id = excluded.cursor_before_message_id,
            cursor_before_partial_message_id = excluded.cursor_before_partial_message_id,
            cursor_before_offset = excluded.cursor_before_offset,
            cursor_after_message_id = excluded.cursor_after_message_id,
            cursor_after_partial_message_id = excluded.cursor_after_partial_message_id,
            cursor_after_offset = excluded.cursor_after_offset,
            created_at = CURRENT_TIMESTAMP
        """,
        (
            session_id,
            generation,
            extraction.slice_key,
            json.dumps(safe_items, ensure_ascii=False, sort_keys=True),
            extraction.start_message_id,
            extraction.start_message_offset,
            extraction.end_message_id,
            extraction.cursor_before_message_id,
            extraction.cursor_before_partial_message_id,
            extraction.cursor_before_offset,
            extraction.covered_message_id,
            extraction.partial_message_id,
            extraction.next_message_offset,
        ),
    )


def publish_profile_generation(
    conn: sqlite3.Connection,
    session_id: str,
    generation: str,
) -> int:
    """Publish all staged slices for one complete walk atomically.

    Existing claims omitted by a rebuild are deliberately retained. Historical
    ``user_profile`` rows coalesced identical values without recording every
    supporting session, so removing a row here could erase another session's
    still-valid support. This is the documented conservative over-keep policy;
    a future source-evidence layer can make replacement authoritative.
    """
    state = conn.execute(
        """
        SELECT profile_cursor_message_id, profile_cursor_partial_message_id,
               profile_cursor_offset, profile_cursor_prompt_version,
               profile_published_generation
        FROM sessions WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if (
        state is None
        or state["profile_cursor_prompt_version"] != generation
        or not profile_generation_is_recognized(generation)
    ):
        raise RuntimeError("profile publication generation is not the active cursor")
    tail = profile_user_tail_message_id(conn, session_id)
    if (
        tail is None
        or state["profile_cursor_partial_message_id"] is not None
        or int(state["profile_cursor_offset"] or 0) != 0
        or state["profile_cursor_message_id"] is None
        or int(state["profile_cursor_message_id"]) != int(tail)
    ):
        raise RuntimeError("profile publication cursor has not reached the USER tail")

    rows = conn.execute(
        """
        SELECT items_json, cursor_before_message_id,
               cursor_before_partial_message_id, cursor_before_offset,
               cursor_after_message_id, cursor_after_partial_message_id,
               cursor_after_offset
        FROM profile_staging
        WHERE session_id = ? AND generation = ?
        ORDER BY COALESCE(start_message_id, -1),
                 start_message_offset,
                 COALESCE(end_message_id, -1), slice_key
        """,
        (session_id, generation),
    ).fetchall()
    if not rows:
        raise RuntimeError("profile publication has no staged slices")
    first_before = (
        rows[0]["cursor_before_message_id"],
        rows[0]["cursor_before_partial_message_id"],
        int(rows[0]["cursor_before_offset"] or 0),
    )
    if (
        generation != state["profile_published_generation"]
        and first_before != (None, None, 0)
    ):
        raise RuntimeError("profile staging is missing its first slice")
    previous_after = None
    items: list[dict] = []
    seen_values: dict[tuple[str, str | None, int], str] = {}
    for row in rows:
        before = (
            row["cursor_before_message_id"],
            row["cursor_before_partial_message_id"],
            int(row["cursor_before_offset"] or 0),
        )
        after = (
            row["cursor_after_message_id"],
            row["cursor_after_partial_message_id"],
            int(row["cursor_after_offset"] or 0),
        )
        if previous_after is not None and before != previous_after:
            raise RuntimeError("profile staging slice chain is incomplete")
        previous_after = after
        decoded = loads_exact_or_fenced(row["items_json"])
        if not isinstance(decoded, list):
            raise RuntimeError("profile staging payload is not an array")
        for item in decoded:
            if not isinstance(item, dict):
                raise RuntimeError("profile staging contains a non-object item")
            internal_expected = {
                "slot", "value", "evidence_message_id", "confidence",
                "source_session_id", "source_created_at",
            } | ({"slot_key"} if item.get("slot") == "relationship" else set())
            if set(item) != internal_expected:
                raise RuntimeError("profile staging contains unexpected fields")
            mid = item.get("evidence_message_id")
            contract_keys = ["slot", "value", "evidence_message_id", "confidence"]
            if item.get("slot") == "relationship":
                contract_keys.append("slot_key")
            contract_item = {
                key: item[key] for key in contract_keys if key in item
            }
            clean = _validate_profile_item(contract_item, {mid})
            if (
                isinstance(mid, bool)
                or not isinstance(mid, int)
                or clean is None
                or item.get("source_session_id") != session_id
            ):
                raise RuntimeError("profile staging contains an invalid item")
            # Also verifies USER role, canonical artifact/hash, and producer
            # frontier before a staged claim becomes visible.
            source_mid, source_session, source_created, _live_mid = (
                _resolve_profile_source(conn, item)
            )
            identity = (
                clean["slot"], clean.get("slot_key"), source_mid
            )
            exclusive = (
                clean["slot"] in SINGLE_VALUED_SLOTS
                or clean["slot"] == "relationship"
            )
            prior = seen_values.get(identity) if exclusive else None
            if prior is not None and not _same_value(prior, clean["value"]):
                raise RuntimeError(
                    "profile staging contains conflicting source assertions"
                )
            if exclusive:
                seen_values[identity] = clean["value"]
            # Persist the validator's canonical shape, never the original
            # staged dict. In particular, non-relationship slot_key values are
            # stripped so singleton supersession cannot be bypassed.
            items.append({
                **clean,
                "source_message_id": source_mid,
                "source_session_id": source_session,
                "source_created_at": source_created,
            })
    session_after = (
        state["profile_cursor_message_id"],
        state["profile_cursor_partial_message_id"],
        int(state["profile_cursor_offset"] or 0),
    )
    if previous_after != session_after:
        raise RuntimeError("profile staging tail does not match its cursor")
    # Staging was already redacted. Running the redactor again is unnecessary
    # and could make externally supplied redaction functions non-idempotent.
    inserted = persist_user_profile(
        conn,
        ProfileExtraction(items=items),
        redact_values=False,
    )
    conn.execute(
        "DELETE FROM profile_staging WHERE session_id = ? AND generation = ?",
        (session_id, generation),
    )
    return inserted


def _resolve_profile_source(
    conn: sqlite3.Connection,
    item: dict,
) -> tuple[int, str | None, str | None, int | None]:
    """Resolve durable provenance, preferring the exact live source.

    Returns ``(source_message_id, source_session_id, source_created_at,
    live_evidence_message_id)``. The final value is suitable for the legacy FK
    column and is ``None`` after raw retention; the first three remain durable.
    """
    if item.get("source_message_id") is not None:
        raw_mid = item["source_message_id"]
        if isinstance(raw_mid, bool) or not isinstance(raw_mid, int):
            raise ValueError("profile source message id must be an integer")
        mid = raw_mid
    elif item.get("evidence_message_id") is not None:
        raw_mid = item["evidence_message_id"]
        if isinstance(raw_mid, bool) or not isinstance(raw_mid, int):
            raise ValueError("profile evidence message id must be an integer")
        mid = raw_mid
    else:
        raise ValueError("profile item has no evidence message id")
    expected_session = item.get("source_session_id")
    live = conn.execute(
        "SELECT session_id, role, content, created_at FROM messages WHERE id = ?",
        (mid,),
    ).fetchone()
    candidate_sessions: list[str] = []
    if expected_session is not None:
        if not isinstance(expected_session, str) or not expected_session.strip():
            raise ValueError("profile source session id must be a non-empty string")
        candidate_sessions = [expected_session]
    elif live is not None and live["role"] == "user":
        candidate_sessions = [live["session_id"]]
    else:
        candidate_sessions = [
            row["source_session_id"]
            for row in conn.execute(
            """
            SELECT DISTINCT source_session_id
            FROM message_retention_coverage
            WHERE message_id = ?
              AND source_role = 'user'
              AND coverage_version = 'dream-lossless-message-v1'
            ORDER BY source_session_id
            """,
            (mid,),
            ).fetchall()
        ]

    validated: list[CoveredMessage] = []
    for candidate_session in candidate_sessions:
        # Reuse the canonical artifact/hash/role/frontier validator. Merely
        # finding a ledger row or a live USER id is insufficient: neither may
        # launder content that the ordered producer never covered.
        covered = covered_messages_after(
            conn,
            candidate_session,
            mid - 1,
            limit=1,
            roles=_USER_ROLE,
            through_message_id=mid,
        )
        if covered and covered[0].message_id == mid:
            validated.append(covered[0])
    if len(validated) == 1:
        message = validated[0]
        stated_created_at = item.get("source_created_at")
        if (
            stated_created_at is not None
            and stated_created_at != message.source_created_at
        ):
            raise ValueError("profile item source timestamp mismatches coverage")
        live_mid = None
        if live is not None and (
            live["session_id"] == message.session_id
            and live["role"] == "user"
            and live["content"] == message.content
            and live["created_at"] == message.source_created_at
        ):
            live_mid = mid
        return mid, message.session_id, message.source_created_at, live_mid
    raise ValueError(f"profile evidence source {mid} is unavailable")


def _normalized_instant(created_at: str | None) -> str | None:
    if not created_at or not isinstance(created_at, str):
        return None
    candidate = created_at.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="microseconds")


def _chronology_key(
    created_at: str | None,
    session_id: str | None,
    message_id: int | None,
    *,
    fallback_at: str | None = None,
    row_id: int | None = None,
) -> tuple[int, str, str, int, int]:
    """Total source order with unknown time explicitly older than known time."""
    instant = _normalized_instant(created_at)
    fallback = _normalized_instant(fallback_at)
    return (
        1 if instant is not None else 0,
        instant or fallback or "",
        session_id or "",
        int(message_id) if message_id is not None else -1,
        int(row_id) if row_id is not None else -1,
    )


def _interval_timestamp(value: str | None) -> str | None:
    """Coherent validity timestamp while preserving ordinary SQLite dates.

    Offset-aware inputs are converted to UTC so textual storage cannot invert
    an interval (for example ``+10:00`` versus ``Z``). Naive timestamps retain
    their historical representation and are interpreted as UTC by the ordering
    helper.
    """
    if not value or not isinstance(value, str):
        return None
    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return candidate
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def reconcile_profile_intervals(
    conn: sqlite3.Connection,
    slot: str,
    slot_key: str | None,
    *,
    fallback_now: str | None = None,
) -> None:
    """Derive singleton validity intervals from total source chronology.

    Looking only at currently-active rows makes historical close times depend
    on import/call order (1→2→3 differs from 3→2→1). Recompute the complete
    chain after every singleton/keyed-relationship mutation: each assertion
    closes at its immediate successor and only the latest remains active.
    """
    if slot not in SINGLE_VALUED_SLOTS and slot != "relationship":
        return
    fallback = fallback_now or conn.execute(
        "SELECT CURRENT_TIMESTAMP AS t"
    ).fetchone()["t"]
    rows = conn.execute(
        "SELECT id, valid_at, created_at, source_message_id, "
        "source_session_id, source_created_at FROM user_profile "
        "WHERE slot = ? AND slot_key IS ?",
        (slot, slot_key),
    ).fetchall()
    ordered = sorted(
        rows,
        key=lambda row: _chronology_key(
            row["source_created_at"],
            row["source_session_id"],
            row["source_message_id"],
            fallback_at=row["valid_at"] or row["created_at"],
            row_id=row["id"],
        ),
    )
    for index, row in enumerate(ordered):
        close_at: str | None = None
        if index + 1 < len(ordered):
            successor = ordered[index + 1]
            close_at = (
                _interval_timestamp(successor["source_created_at"])
                or _interval_timestamp(successor["valid_at"])
                or _interval_timestamp(successor["created_at"])
                or fallback
            )
            opened = (
                _interval_timestamp(row["source_created_at"])
                or _interval_timestamp(row["valid_at"])
                or _interval_timestamp(row["created_at"])
            )
            if (
                opened is not None
                and _chronology_key(opened, None, None)
                > _chronology_key(close_at, None, None)
            ):
                close_at = opened
        conn.execute(
            "UPDATE user_profile SET invalid_at = ? WHERE id = ?",
            (close_at, row["id"]),
        )


def persist_user_profile(
    conn: sqlite3.Connection,
    extraction: ProfileExtraction,
    *,
    redact_values: bool = True,
) -> int:
    """Persist validated items with bi-temporal supersession. Caller wraps in
    core_db.transaction().

    Per item, exclusive temporal slots retain one assertion per durable source
    and their complete interval chain is re-derived by source chronology.
    Unknown/unparseable source times use a fixed historical sentinel, so
    replay wall time cannot change ordering. Unkeyed multi-valued slots may
    coalesce equivalent values. ``redact_values`` applies the persistence-side
    safety check in addition to the earlier staging/import boundary. Returns
    the number of NEW rows inserted."""
    inserted = 0
    fallback_now = conn.execute("SELECT CURRENT_TIMESTAMP AS t").fetchone()["t"]
    resolved_items: list[
        tuple[
            tuple[int, str, str, int, int],
            dict,
            tuple[int, str | None, str | None, int | None],
        ]
    ] = []
    for item in extraction.items:
        source = _resolve_profile_source(conn, item)
        resolved_items.append(
            (_chronology_key(source[2], source[1], source[0]), item, source)
        )
    # Session iteration order is not chronology. Replays and prompt rewinds
    # must apply older facts first so they can never regress a newer active one.
    resolved_items.sort(key=lambda entry: entry[0])

    for _, item, source in resolved_items:
        slot, key, mid = item["slot"], item.get("slot_key"), item["evidence_message_id"]
        if slot == "relationship" and isinstance(key, str):
            key = key.strip().lower() or None
        conf = item["confidence"]
        value = redaction.redact(item["value"]) if redact_values else item["value"]
        if redact_values:
            key = _redact_profile_key(key)
        source_mid, source_session_id, source_created_at, live_mid = source
        if not redact_values and slot == "relationship":
            # Privacy is monotonic for already-persisted evidence. Replaying
            # one source after True→False must reinforce its pseudonymous key,
            # not create a second raw-key identity. New evidence is still
            # stored according to the now-disabled policy.
            safe_key = _redact_profile_key(key)
            if safe_key != key:
                redacted_prior = conn.execute(
                    "SELECT 1 FROM user_profile WHERE slot = 'relationship' "
                    "AND slot_key IS ? AND source_message_id IS ? "
                    "AND source_session_id IS ? LIMIT 1",
                    (safe_key, source_mid, source_session_id),
                ).fetchone()
                if redacted_prior is not None:
                    key = safe_key
        # Missing/unparseable durable source time is ordered by the remaining
        # provenance tuple. Using wall-clock insertion time here made the full
        # interval chain depend on replay/import order; a fixed historical
        # sentinel is deterministic and cannot supersede known-time evidence.
        world_date = (
            _interval_timestamp(source_created_at)
            or _UNKNOWN_PROFILE_VALID_AT
        )

        # Exact source replay is idempotent even when that row is already
        # historical. The old active-only check duplicated such rows on every
        # full prompt walk.
        prior_rows = conn.execute(
            "SELECT id, value FROM user_profile "
            "WHERE slot = ? AND slot_key IS ? "
            "AND source_message_id IS ? AND source_session_id IS ?",
            (slot, key, source_mid, source_session_id),
        ).fetchall()
        safe_equivalent = redaction.redact(value)
        exact = [
            r for r in prior_rows
            if _same_value(r["value"], value)
            or _same_value(r["value"], safe_equivalent)
        ]
        if exact:
            conn.execute(
                "UPDATE user_profile SET confidence = MAX(confidence, ?) WHERE id = ?",
                (conf, exact[0]["id"]),
            )
            reconcile_profile_intervals(
                conn, slot, key, fallback_now=fallback_now
            )
            continue
        same_source_preferred = False
        if prior_rows and (slot in SINGLE_VALUED_SLOTS or slot == "relationship"):
            # A prompt/import can disagree about one identical source. There
            # is no chronology with which to order those interpretations, so
            # use a stable normalized-value tie-break rather than call/import
            # order. This lets a later prompt heal deterministically while two
            # stores merged in reverse order still converge.
            normalized_value = " ".join(value.casefold().split())
            prior_values = [" ".join(r["value"].casefold().split()) for r in prior_rows]
            if normalized_value >= min(prior_values):
                continue
            keeper = min(prior_rows, key=lambda row: row["id"])
            conn.execute(
                "UPDATE user_profile SET value = ?, "
                "confidence = MAX(confidence, ?), evidence_message_id = ? "
                "WHERE id = ?",
                (value, conf, live_mid, keeper["id"]),
            )
            conn.execute(
                "DELETE FROM user_profile WHERE slot = ? AND slot_key IS ? "
                "AND source_message_id IS ? AND source_session_id IS ? "
                "AND id <> ?",
                (slot, key, source_mid, source_session_id, keeper["id"]),
            )
            reconcile_profile_intervals(
                conn, slot, key, fallback_now=fallback_now
            )
            continue

        # `slot_key IS ?` is SQLite's NULL-safe equality, so unkeyed slots
        # compare on slot alone and keyed slots on the exact (slot, key) pair.
        active = conn.execute(
            "SELECT id, value, valid_at, created_at, source_message_id, "
            "source_session_id, source_created_at FROM user_profile "
            "WHERE slot = ? AND slot_key IS ? AND invalid_at IS NULL",
            (slot, key),
        ).fetchall()

        same = [r for r in active if _same_value(r["value"], value)]
        exclusive_slot = slot in SINGLE_VALUED_SLOTS or slot == "relationship"
        if exclusive_slot:
            # Temporal singleton slots retain one assertion per durable source.
            # Coalescing A@t1 with A@t3 before B@t2 arrives irreversibly erases
            # the A→B→A history and makes merge order observable.
            same = []
        if same:
            existing = same[0]
            newer_support = (
                _chronology_key(
                    source_created_at, source_session_id, source_mid
                )
                > _chronology_key(
                    existing["source_created_at"],
                    existing["source_session_id"],
                    existing["source_message_id"],
                )
            )
            conn.execute(
                "UPDATE user_profile SET confidence = MAX(confidence, ?), "
                "evidence_message_id = CASE WHEN ? THEN ? ELSE evidence_message_id END, "
                "source_message_id = CASE WHEN ? THEN ? ELSE source_message_id END, "
                "source_session_id = CASE WHEN ? THEN ? ELSE source_session_id END, "
                "source_created_at = CASE WHEN ? THEN ? ELSE source_created_at END "
                "WHERE id = ?",
                (
                    conf,
                    newer_support,
                    live_mid,
                    newer_support,
                    source_mid,
                    newer_support,
                    source_session_id,
                    newer_support,
                    source_created_at,
                    existing["id"],
                ),
            )
            reconcile_profile_intervals(
                conn, slot, key, fallback_now=fallback_now
            )
            continue

        historical_invalid_at: str | None = None
        if (slot in SINGLE_VALUED_SLOTS or slot == "relationship") and active:
            newest = max(
                active,
                key=lambda row: _chronology_key(
                    row["source_created_at"],
                    row["source_session_id"],
                    row["source_message_id"],
                ),
            )
            newest_date = (
                _interval_timestamp(newest["source_created_at"])
                or _interval_timestamp(newest["valid_at"])
                or _interval_timestamp(newest["created_at"])
            )
            incoming_key = _chronology_key(
                source_created_at, source_session_id, source_mid
            )
            newest_key = _chronology_key(
                newest["source_created_at"],
                newest["source_session_id"],
                newest["source_message_id"],
            )
            if (
                incoming_key < newest_key
                or (incoming_key == newest_key and not same_source_preferred)
            ):
                # Replaying an older session after a newer one creates a closed
                # historical interval; it never regresses the active profile.
                historical_invalid_at = newest_date or fallback_now
            else:
                for row in active:
                    old_open = (
                        _interval_timestamp(row["source_created_at"])
                        or _interval_timestamp(row["valid_at"])
                        or _interval_timestamp(row["created_at"])
                    )
                    close_at = world_date
                    if (
                        old_open is not None
                        and _chronology_key(old_open, None, None)
                        > _chronology_key(world_date, None, None)
                    ):
                        close_at = old_open
                    conn.execute(
                        "UPDATE user_profile SET invalid_at = ? "
                        "WHERE id = ? AND invalid_at IS NULL",
                        (close_at, row["id"]),
                    )

        conn.execute(
            "INSERT INTO user_profile("
            "    slot, slot_key, value, evidence_message_id, confidence, valid_at, "
            "    invalid_at, source_message_id, source_session_id, source_created_at"
            ") VALUES (?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?, ?)",
            (
                slot,
                key,
                value,
                live_mid,
                conf,
                world_date,
                historical_invalid_at,
                source_mid,
                source_session_id,
                source_created_at,
            ),
        )
        inserted += 1
        reconcile_profile_intervals(
            conn, slot, key, fallback_now=fallback_now
        )

    if inserted:
        log.debug("profile.persisted rows=%d", inserted)
    return inserted


def load_profile(conn: sqlite3.Connection, cap: int | None = None) -> list[ProfileEntry]:
    """All ACTIVE profile rows, identity-first (PROFILE_SLOTS order), then by
    slot_key / confidence for a stable rendering. Returns [] on a pre-v18 DB
    (no table) so every consumer degrades cleanly. Read-only."""
    try:
        rows = conn.execute(
            "SELECT slot, slot_key, value, confidence, "
            "COALESCE(source_message_id, evidence_message_id) AS evidence_message_id, "
            "valid_at "
            "FROM user_profile WHERE invalid_at IS NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    entries = [
        ProfileEntry(
            slot=r["slot"],
            slot_key=r["slot_key"],
            value=r["value"],
            confidence=float(r["confidence"]),
            evidence_message_id=(
                int(r["evidence_message_id"])
                if r["evidence_message_id"] is not None
                else None
            ),
            valid_at=r["valid_at"] or "",
        )
        for r in rows
    ]
    entries.sort(
        key=lambda e: (
            _SLOT_ORDER.get(e.slot, len(PROFILE_SLOTS)),
            e.slot_key or "",
            -e.confidence,
            e.value,
        )
    )
    return entries[:cap] if cap is not None else entries


def render_profile_fact(entry: ProfileEntry) -> str:
    """One-line rendering for the VERIFIED FACTS anchor block, shaped like the
    graph-edge lines it sits above ("user role bedrijfsarts",
    "user relationship(anna) sister")."""
    if entry.slot_key:
        return f"user {entry.slot}({entry.slot_key}) {entry.value}"
    return f"user {entry.slot} {entry.value}"
