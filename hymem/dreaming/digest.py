from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
import json

from hymem.dreaming.episodes import EpisodesExtraction, validate_episode_items
from hymem.dreaming.lossless import CoveredMessage, covered_messages_after
from hymem.dreaming.procedures import ProceduresExtraction, validate_procedure_items
from hymem.dreaming.summary import clean_summary
from hymem.extraction.jsonio import is_ceiling_cut, loads_exact_or_fenced
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import (
    SESSION_DIGEST_GRANULAR_SYSTEM,
    SESSION_DIGEST_GRANULAR_USER_TEMPLATE,
    SESSION_DIGEST_SYSTEM,
    SESSION_DIGEST_USER_TEMPLATE,
)

log = logging.getLogger("hymem.dreaming.digest")

# Pinned prompt version for the Plan C decision-grained episode arm, following
# the PROFILE_PROMPT_VERSION / FACTS_PROMPT_VERSION convention. Unlike the
# facts version this one is NOT forward-only: episodes are UPSERTed by message
# range, so a granularity change that did not invalidate prior extractions
# would leave the old blob episodes sitting in the store beside the new
# decision-grained ones (UPSERT only refreshes a row whose range matches). The
# runner therefore stamps it per session (sessions.episodes_prompt_version,
# schema v35) and a mismatch re-reads the session from the start.
#
# The blob arm has NO version string of its own on purpose: its stamp is NULL,
# which is what every pre-v35 store already reads, so a store that never turns
# granularity on can never see a stamp mismatch and never pays a re-extraction.
# NULL here means "extracted under the shipping digest prompt, unattributed" —
# the store-wide convention, not a counterfeit version.
EPISODE_GRANULAR_PROMPT_VERSION = "episodes.granular.v1"
DIGEST_STREAM_VERSION = "lossless-digest-v2"
_DIGEST_CONTEXT_CHARS = 48
_DIGEST_CONFIG_PATTERN = (
    rf"{re.escape(DIGEST_STREAM_VERSION)}\|"
    r"prompt=[^|\r\n]+\|episodes=[^|\r\n]+\|"
    r"chars=[1-9]\d*\|tokens=[1-9]\d*\|"
    r"episode-cap=(?:blob|0|[1-9]\d*)"
)
_DIGEST_GENERATION_RE = re.compile(
    _DIGEST_CONFIG_PATTERN + r"\|walk=[0-9a-f]{32}"
)
_DIGEST_RETRY_RE = re.compile(
    rf"(?P<config>{_DIGEST_CONFIG_PATTERN})\|retry-max=(?P<maximum>-?\d+)\|"
    r"(?P<mode>forward|rebuild=.+;stamp=.+)"
)


def digest_config_version(
    *, prompt_version: str, episode_prompt_version: str | None, max_chars: int,
    max_tokens: int, max_episodes: int | None,
) -> str:
    """Stable configuration prefix for one resumable digest walk."""
    return (
        f"{DIGEST_STREAM_VERSION}|prompt={prompt_version}|"
        f"episodes={episode_prompt_version or 'blob'}|chars={int(max_chars)}|"
        f"tokens={int(max_tokens)}|episode-cap="
        f"{int(max_episodes) if max_episodes is not None else 'blob'}"
    )


def digest_generation_matches_config(generation: object, config: str) -> bool:
    """Whether *generation* is exactly a producer-issued current walk id."""
    return bool(
        isinstance(generation, str)
        and re.fullmatch(re.escape(config) + r"\|walk=[0-9a-f]{32}", generation)
    )


def digest_generation_is_recognized(generation: object) -> bool:
    """Whether *generation* has the complete current producer wire shape."""
    return bool(
        isinstance(generation, str)
        and _DIGEST_GENERATION_RE.fullmatch(generation)
    )


def digest_attempt_max_chars(configured_max: int, retry_count: int) -> int:
    """Adaptive exact-input bound for a retry of one held cursor position."""
    if configured_max <= 0:
        return configured_max
    floor = min(configured_max, 256)
    return max(floor, configured_max // (2 ** min(max(0, retry_count), 8)))


def digest_retry_policy_version(
    digest_config: str,
    *,
    max_attempts: int,
    rebuild_from: str | None = None,
    invalidated_stamp: str | None = None,
) -> str:
    """Retry-state key separate from the source/config generation."""
    mode = (
        f"rebuild={rebuild_from or 'none'};stamp={invalidated_stamp or 'none'}"
        if rebuild_from is not None
        else "forward"
    )
    return f"{digest_config}|retry-max={int(max_attempts)}|{mode}"


def digest_retry_state_is_valid(
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
    match = _DIGEST_RETRY_RE.fullmatch(retry_config_version)
    if match is None:
        return False
    maximum = int(match.group("maximum"))
    return bool(quarantined) == bool(maximum > 0 and retry_count >= maximum)


def record_digest_failure(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    max_attempts: int,
    retry_config_version: str,
) -> bool:
    """Record a held digest failure and return whether it is quarantined."""
    row = conn.execute(
        "SELECT digest_retry_count, digest_retry_config_version "
        "FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    attempts = (
        int(row["digest_retry_count"] or 0) + 1
        if row["digest_retry_config_version"] == retry_config_version
        else 1
    )
    quarantined = bool(max_attempts > 0 and attempts >= max_attempts)
    conn.execute(
        "UPDATE sessions SET digest_retry_count = ?, "
        "digest_retry_config_version = ?, digest_quarantined = ? "
        "WHERE id = ?",
        (attempts, retry_config_version, int(quarantined), session_id),
    )
    if quarantined:
        log.warning(
            "digest.extraction_quarantined session_id=%s attempts=%d "
            "cursor_advanced=0 partial_published=0",
            session_id,
            attempts,
        )
    return quarantined


def active_episode_prompt_version(granular: bool) -> str | None:
    """The episode-prompt stamp a session should carry right now.

    One function so the runner's skip-guard and the stamp it writes after a
    successful persist cannot drift: both read this, and a guard that compared
    against one string while the stamp wrote another would re-extract every
    session on every dream forever. Returns None for the shipping blob prompt (see
    above), so on a store that has never enabled granularity the comparison is
    ``None == NULL`` — true — and the digest guard behaves exactly as it did
    before v35.
    """
    return EPISODE_GRANULAR_PROMPT_VERSION if granular else None


@dataclass
class SessionDigest:
    """The three per-session tail extractions produced by one LLM call:
    episodes, a one-sentence summary, and procedures.

    ``covered_message_id`` is the highest ``chunks.end_message_id`` that made it
    into the LLM input (None when the chunks carry no message range). The runner
    stores it as ``sessions.digested_message_id`` so the next dream resumes
    above it — see :func:`extract_session_digest`.

    ``start_message_id`` is the low end of that same window (the first chunk's
    ``start_message_id``). It exists for the Plan C granular arm, whose persist
    step supersedes the episodes INSIDE the window it just re-read and must not
    touch anything outside it; the blob arm never reads it."""
    episodes: EpisodesExtraction
    summary: str | None
    procedures: ProceduresExtraction
    covered_message_id: int | None = None
    start_message_id: int | None = None
    # True when the LLM reply could not be parsed as a digest object. The three
    # tiers are then empty for a reason the caller must be able to distinguish
    # from "this slice genuinely held nothing" — it drives dream_runs.
    # digest_failures (v25) and suppresses the watermark advance.
    parse_failed: bool = False
    # Character offset in the first message above ``covered_message_id``.  A
    # non-zero value means one oversized message was only partly consumed and
    # the next successful call must resume at exactly this character.
    next_message_offset: int = 0
    partial_message_id: int | None = None
    # Last message that contributed any characters/framing to this call.  This
    # can be newer than covered_message_id while an oversized turn is partial.
    end_message_id: int | None = None
    # True only when this successful call reached the end of every currently
    # materialized coverage artifact in the session.
    caught_up: bool = False
    # Explicit extraction-outcome attribution. ``parse_failed`` remains the
    # compatibility retry flag; this field tells operators/tests whether the
    # cause was parsing, top-level shape, malformed non-empty items, summary
    # validation, or the configured episode output cap.
    failure_reason: str | None = None
    episode_input_items: int = 0
    episode_rejected_items: int = 0
    procedure_input_items: int = 0
    procedure_rejected_items: int = 0


_DIGEST_SEPARATOR = "\n\n---\n\n"


def _render_message_part(
    message: CoveredMessage,
    start: int,
    end: int,
) -> str:
    external = (
        f" peer={message.source_peer_id} workspace={message.source_workspace_id}"
        if message.source_peer_id is not None
        else ""
    )
    current = (
        f"[chunk {message.chunk_id}] "
        f"[message {message.message_id} role={message.role}{external} "
        f"chars={start}:{end}/{len(message.content)}]\n"
        f"{message.content[start:end]}"
    )
    if start <= 0:
        return current
    context_start = max(0, start - _DIGEST_CONTEXT_CHARS)
    return (
        f"[previous context for message {message.message_id} "
        f"range={context_start}:{start}/{len(message.content)}]\n"
        f"{message.content[context_start:start]}\n"
        f"{current}"
    )


def _largest_part_end(
    message: CoveredMessage,
    start: int,
    budget: int,
) -> int | None:
    """Largest exclusive content offset whose framed part fits ``budget``."""
    low, high = start, len(message.content)
    if len(_render_message_part(message, start, start)) > budget:
        return None
    while low < high:
        mid = (low + high + 1) // 2
        if len(_render_message_part(message, start, mid)) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def _render_digest_leading_context(message: CoveredMessage) -> str:
    start = max(0, len(message.content) - _DIGEST_CONTEXT_CHARS)
    external = (
        f" peer={message.source_peer_id} workspace={message.source_workspace_id}"
        if message.source_peer_id is not None
        else ""
    )
    return (
        f"[previous message context message {message.message_id}{external} "
        f"range={start}:{len(message.content)}/{len(message.content)}]\n"
        f"{message.content[start:]}"
    )


def _build_message_window(
    messages: list[CoveredMessage],
    *,
    since_message_id: int | None,
    since_message_offset: int,
    max_chars: int,
    leading_context: CoveredMessage | None = None,
) -> tuple[
    str, list[str], int | None, int | None, int, int | None, int | None, bool
]:
    """Build one bounded, lossless digest slice and its next cursor state."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if since_message_offset < 0:
        raise ValueError("since_message_offset must be non-negative")
    if not messages:
        return "", [], since_message_id, None, 0, None, None, True

    parts: list[str] = []
    context = (
        _render_digest_leading_context(leading_context)
        if leading_context is not None else ""
    )
    used_chars = len(context)
    valid_ids: list[str] = []
    covered = since_message_id
    next_offset = 0
    partial_message_id: int | None = None
    started: int | None = None
    ended: int | None = None
    caught_up = False

    for index, message in enumerate(messages):
        start = since_message_offset if index == 0 else 0
        if start > len(message.content):
            raise RuntimeError(
                f"digest cursor offset {start} exceeds message "
                f"{message.message_id} length {len(message.content)}"
            )
        separator = _DIGEST_SEPARATOR if (parts or context) else ""
        remaining = max_chars - used_chars - len(separator)
        end = _largest_part_end(message, start, remaining)
        if end is None or (end == start and start < len(message.content)):
            if parts:
                break
            raise ValueError(
                "dream_digest_max_chars is too small for lossless message framing"
            )

        if started is None:
            started = message.message_id
        ended = message.message_id
        rendered = _render_message_part(message, start, end)
        parts.append(rendered)
        used_chars += len(separator) + len(rendered)
        valid_ids.append(message.chunk_id)

        if end == len(message.content):
            covered = message.message_id
            next_offset = 0
            if index == len(messages) - 1:
                caught_up = True
            continue

        # The precise next character is persisted only after the LLM result is
        # parsed and all derived writes commit.  No tail is silently claimed.
        next_offset = end
        partial_message_id = message.message_id
        break

    body_parts = ([context] if context else []) + parts
    return (
        _DIGEST_SEPARATOR.join(body_parts),
        valid_ids,
        covered,
        partial_message_id,
        next_offset,
        started,
        ended,
        caught_up,
    )


def extract_session_digest(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
    *,
    max_tokens: int,
    max_chars: int,
    since_message_id: int | None = None,
    partial_message_id: int | None = None,
    since_message_offset: int = 0,
    prior_summary: str | None = None,
    granular: bool = False,
    max_episodes: int | None = None,
) -> SessionDigest | None:
    """Read the session's durable message stream and run one digest LLM call.
    episodes, summary, and procedures together (the batched replacement for the
    three separate tail calls).

    `granular` (Plan C, `episode_granularity_enabled`, default OFF) swaps the
    prompt pair for the decision-grained variant and bounds the episode list at
    `max_episodes`. Both are inert at their defaults: with `granular=False` this
    function sends the same system/user strings it always sent and validates the
    reply with the same unbounded validator, so the flag-off path is
    byte-identical to the pre-Plan-C one. The cap is deliberately NOT applied to
    the blob arm — a cap that trims a shipping extraction is a default change,
    and this ships default-OFF until `benchmarks/episode_probe.py` scores the
    granular prompt.

    ``since_message_id`` is the last fully consumed message and
    ``since_message_offset`` is the exact character offset already consumed in
    the next message.  Input comes only from v38's protected, canonical message
    artifacts, so assistant/short/system/tool turns are eligible and a prompt
    rewind still works after the raw message table has been pruned.  Oversized
    messages are sliced rather than truncated; their message-level watermark
    advances only after the final character has succeeded.

    Returns None when there is nothing to extract from (including a session
    whose tail is already fully digested). No write transaction held; persist
    via the per-kind persist_* helpers inside one.
    """
    coverage_tail = conn.execute(
        "SELECT coverage_message_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()["coverage_message_id"]
    messages = covered_messages_after(conn, session_id, since_message_id)
    if not messages:
        already_at_tail = (
            since_message_offset == 0
            and partial_message_id is None
            and (
                coverage_tail is None
                or (
                    since_message_id is not None
                    and int(since_message_id) == int(coverage_tail)
                )
            )
        )
        if already_at_tail:
            return None
        raise RuntimeError(
            "digest coverage cursor has no readable artifact before its tail"
        )
    if since_message_offset:
        # An offset belongs to one explicit partial message, validated by the
        # runner before this call.  Never apply it to an arbitrary later row.
        if (
            partial_message_id is None
            or messages[0].message_id != int(partial_message_id)
        ):
            raise RuntimeError("digest partial-message cursor does not match artifact")
    elif partial_message_id is not None:
        raise RuntimeError("partial message id requires a non-zero digest offset")
    leading_context: CoveredMessage | None = None
    if since_message_id is not None and since_message_offset == 0:
        prior = covered_messages_after(
            conn,
            session_id,
            int(since_message_id) - 1,
            limit=1,
            through_message_id=int(since_message_id),
        )
        if prior and prior[0].message_id == int(since_message_id):
            leading_context = prior[0]
    (
        combined,
        valid_chunk_ids,
        covered,
        partial_message_id,
        next_offset,
        started,
        ended,
        caught_up,
    ) = _build_message_window(
        messages,
        since_message_id=since_message_id,
        since_message_offset=since_message_offset,
        max_chars=max_chars,
        leading_context=leading_context,
    )
    # Boundary-only prior-message context is readable to reconstruct a phrase,
    # but never enters the provenance allow-list. Every derived item must cite
    # at least one artifact from the newly consumed slice.
    caught_up = bool(
        next_offset == 0
        and coverage_tail is not None
        and covered is not None
        and int(covered) == int(coverage_tail)
    )

    system, template = (
        (SESSION_DIGEST_GRANULAR_SYSTEM, SESSION_DIGEST_GRANULAR_USER_TEMPLATE)
        if granular
        else (SESSION_DIGEST_SYSTEM, SESSION_DIGEST_USER_TEMPLATE)
    )
    request = LLMRequest(
        system=system,
        user=template.format(text=combined, prior_summary=prior_summary or ""),
        response_format="json",
        max_tokens=max_tokens,
    )
    raw = llm.complete(request)

    # This reply advances a durable cursor, so only exact JSON or one
    # whole-response Markdown fence is accepted.  Scanning prose could turn a
    # refusal/example containing an empty object into false full coverage.
    data = loads_exact_or_fenced(raw)
    if data is None:
        log.warning("digest.parse_failure session_id=%s raw_len=%d",
                    session_id, len(raw) if isinstance(raw, str) else -1)
        reason = (
            "output_truncated"
            if isinstance(raw, str) and is_ceiling_cut(raw)
            else "parse_failure"
        )
        return _empty(reason=reason)

    # A bare array (e.g. a stub LLM's "[]" default) or any non-object payload
    # is a failed response, not an authoritative empty digest.  Return the
    # retry sentinel rather than crashing or advancing coverage.
    if not isinstance(data, dict):
        # Keep the historical stub default quiet to avoid warning noise, but it
        # still holds the watermark. Any OTHER shape is a real reply we dropped;
        # a persistent one re-sends this slice and the log surfaces the stall.
        if data != []:
            log.warning("digest.shape_failure session_id=%s type=%s",
                        session_id, type(data).__name__)
        return _empty(reason="shape_failure")

    required_keys = {"episodes", "summary", "procedures"}
    if (
        set(data) != required_keys
        or not isinstance(data["episodes"], list)
        or not isinstance(data["procedures"], list)
    ):
        log.warning("digest.shape_failure session_id=%s keys=%s", session_id, sorted(data))
        return _empty(reason="shape_failure")
    raw_episodes = data["episodes"]
    if (granular and max_episodes is not None
            and isinstance(raw_episodes, list) and len(raw_episodes) > max_episodes):
        log.warning(
            "digest.episode_cap session_id=%s returned=%d cap=%d "
            "action=held_for_retry",
            session_id, len(raw_episodes), max_episodes,
        )
        return _empty(
            reason="episode_output_cap",
            episode_input_items=len(raw_episodes),
            episode_rejected_items=len(raw_episodes) - max_episodes,
        )
    episode_items, episode_rejected = _validate_digest_episode_items(
        raw_episodes,
        valid_chunk_ids,
    )
    if episode_rejected:
        log.warning(
            "digest.episode_item_failure session_id=%s returned=%d rejected=%d",
            session_id,
            len(raw_episodes),
            episode_rejected,
        )
        return _empty(
            reason="episode_validation_failure",
            episode_input_items=len(raw_episodes),
            episode_rejected_items=episode_rejected,
        )
    episodes = EpisodesExtraction(items=episode_items)
    if "summary" not in data or not isinstance(data["summary"], str):
        log.warning("digest.summary_shape_failure session_id=%s", session_id)
        return _empty(reason="summary_shape_failure")
    raw_summary = data["summary"]
    if len(raw_summary.strip()) > 500:
        log.warning(
            "digest.summary_output_cap session_id=%s returned_chars=%d cap=500",
            session_id,
            len(raw_summary.strip()),
        )
        return _empty(reason="summary_output_cap")
    summary = clean_summary(raw_summary)
    if raw_summary.strip() and summary is None:
        # A present empty string is the prompt's explicit "nothing to add"
        # result and may advance while retaining the prior summary.  A
        # non-empty value rejected by validation is not equivalent: advancing
        # would permanently omit this slice from the rolling summary.
        log.warning("digest.summary_failure session_id=%s", session_id)
        return _empty(reason="summary_validation_failure")
    procedure_items, procedure_rejected = _validate_digest_procedure_items(
        data["procedures"], valid_chunk_ids
    )
    if procedure_rejected:
        log.warning(
            "digest.procedure_item_failure session_id=%s returned=%d rejected=%d",
            session_id,
            len(data["procedures"]),
            procedure_rejected,
        )
        return _empty(
            reason="procedure_validation_failure",
            episode_input_items=len(raw_episodes),
            procedure_input_items=len(data["procedures"]),
            procedure_rejected_items=procedure_rejected,
        )
    procedures = ProceduresExtraction(items=procedure_items)
    return SessionDigest(
        episodes=episodes, summary=summary, procedures=procedures,
        covered_message_id=covered, start_message_id=started,
        next_message_offset=next_offset,
        partial_message_id=partial_message_id,
        end_message_id=ended,
        caught_up=caught_up,
        episode_input_items=len(raw_episodes),
        procedure_input_items=len(data["procedures"]),
    )


def _validate_digest_episode_items(
    data: list,
    valid_chunk_ids: list[str],
) -> tuple[list[dict], int]:
    """Strict per-item digest validation without partial acceptance."""
    items: list[dict] = []
    rejected = 0
    identities: dict[tuple[int, int, str], str] = {}
    chunk_order = {
        chunk_id: index for index, chunk_id in enumerate(valid_chunk_ids)
    }
    for raw_item in data:
        valid = (
            isinstance(raw_item, dict)
            and set(raw_item) == {
                "title", "summary", "outcome", "key_entities", "chunk_ids"
            }
            and isinstance(raw_item.get("title"), str)
            and bool(raw_item.get("title", "").strip())
            and isinstance(raw_item.get("summary"), str)
            and bool(raw_item.get("summary", "").strip())
        )
        if valid:
            chunk_ids = raw_item.get("chunk_ids")
            valid = (
                isinstance(chunk_ids, list)
                and bool(chunk_ids)
                and len(chunk_ids) == len(set(chunk_ids))
                and all(
                    isinstance(chunk_id, str) and chunk_id in valid_chunk_ids
                    for chunk_id in chunk_ids
                )
            )
            if valid:
                positions = [chunk_order[chunk_id] for chunk_id in chunk_ids]
                valid = positions == sorted(positions)
        if valid:
            entities = raw_item["key_entities"]
            valid = isinstance(entities, list) and all(
                isinstance(entity, str) and bool(entity.strip()) for entity in entities
            )
        if valid:
            valid = raw_item["outcome"] in {
                None,
                "resolved",
                "blocked",
                "deferred",
                "informational",
            }
        clean = (
            validate_episode_items([raw_item], valid_chunk_ids)
            if valid else []
        )
        if len(clean) != 1:
            rejected += 1
        else:
            cleaned = clean[0]
            identity = (
                min(chunk_order[c] for c in cleaned["chunk_ids"]),
                max(chunk_order[c] for c in cleaned["chunk_ids"]),
                " ".join(cleaned["title"].casefold().split()),
            )
            semantic = json.dumps(
                cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            prior = identities.get(identity)
            if prior is not None and prior != semantic:
                rejected += 1
            elif prior is None:
                identities[identity] = semantic
                items.append(cleaned)
    return items, rejected


def _validate_digest_procedure_items(
    data: list, valid_chunk_ids: list[str]
) -> tuple[list[dict], int]:
    """Strictly reject a malformed procedure or nested step as one outcome."""
    items: list[dict] = []
    rejected = 0
    identities: dict[str, str] = {}
    chunk_order = {
        chunk_id: index for index, chunk_id in enumerate(valid_chunk_ids)
    }
    for raw_item in data:
        valid = (
            isinstance(raw_item, dict)
            and set(raw_item) == {
                "name", "description", "steps", "triggers",
                "entities_involved", "chunk_ids",
            }
        )
        if valid:
            name = raw_item.get("name")
            description = raw_item.get("description")
            steps = raw_item.get("steps")
            valid = (
                isinstance(name, str)
                and bool(name.strip())
                and isinstance(description, str)
                and bool(description.strip())
                and len(description.strip()) <= 500
                and isinstance(steps, list)
                and bool(steps)
            )
        if valid:
            chunk_ids = raw_item["chunk_ids"]
            valid = (
                isinstance(chunk_ids, list)
                and bool(chunk_ids)
                and len(chunk_ids) == len(set(chunk_ids))
                and all(
                    isinstance(chunk_id, str) and chunk_id in valid_chunk_ids
                    for chunk_id in chunk_ids
                )
            )
            if valid:
                positions = [chunk_order[chunk_id] for chunk_id in chunk_ids]
                valid = positions == sorted(positions)
        if valid:
            seen_orders: set[int] = set()
            for step in steps:
                if not isinstance(step, dict) or set(step) != {
                    "order", "action", "tool"
                }:
                    valid = False
                    break
                step_order = step.get("order")
                action = step.get("action")
                tool = step.get("tool")
                if (
                    isinstance(step_order, bool)
                    or not isinstance(step_order, int)
                    or step_order <= 0
                    or step_order in seen_orders
                    or not isinstance(action, str)
                    or not action.strip()
                    or (tool is not None and not isinstance(tool, str))
                ):
                    valid = False
                    break
                seen_orders.add(step_order)
        if valid:
            for key in ("triggers", "entities_involved"):
                values = raw_item[key]
                if not isinstance(values, list) or not all(
                    isinstance(value, str) and bool(value.strip()) for value in values
                ):
                    valid = False
                    break
        clean = validate_procedure_items([raw_item]) if valid else []
        if len(clean) != 1:
            rejected += 1
        else:
            cleaned = clean[0]
            identity = " ".join(cleaned["name"].casefold().split())
            semantic = json.dumps(
                cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            prior = identities.get(identity)
            if prior is not None and prior != semantic:
                rejected += 1
            elif prior is None:
                identities[identity] = semantic
                items.append(cleaned)
    return items, rejected


def _empty(
    *,
    reason: str = "parse_failure",
    episode_input_items: int = 0,
    episode_rejected_items: int = 0,
    procedure_input_items: int = 0,
    procedure_rejected_items: int = 0,
) -> SessionDigest:
    """A parse failure, NOT coverage: `covered_message_id` stays None so the
    watermark does not advance and the slice is retried on the next dream.
    Advancing here would silently skip the slice forever — the same class of
    silent starvation that migration 024 exists to fix."""
    return SessionDigest(
        episodes=EpisodesExtraction(),
        summary=None,
        procedures=ProceduresExtraction(),
        parse_failed=True,
        failure_reason=reason,
        episode_input_items=episode_input_items,
        episode_rejected_items=episode_rejected_items,
        procedure_input_items=procedure_input_items,
        procedure_rejected_items=procedure_rejected_items,
    )
