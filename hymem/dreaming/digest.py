from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from hymem.dreaming.episodes import EpisodesExtraction, validate_episode_items
from hymem.dreaming.procedures import ProceduresExtraction, validate_procedure_items
from hymem.dreaming.summary import clean_summary
from hymem.extraction.jsonio import loads_lenient
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


def extract_session_digest(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
    *,
    max_tokens: int,
    max_chars: int,
    since_message_id: int | None = None,
    granular: bool = False,
    max_episodes: int | None = None,
) -> SessionDigest | None:
    """Read the session's chunks once and run a single LLM call that returns
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

    `since_message_id` (the session's digest watermark, schema v24) restricts
    the input to chunks that START above it — the undigested tail. Without it
    this read joined every chunk in the session and truncated with
    `combined[:max_chars]`, i.e. kept the OLDEST slice: once a long-lived
    session grew past `max_chars`, its tail could never enter the digest input
    and tail episodes were structurally impossible (2026-07-30: 184 messages,
    zero episodes, six days). Truncation still keeps the oldest part of the
    SLICE, which is what makes progress monotonic — the watermark advances to
    the last message actually covered, so the next dream picks up exactly where
    this one stopped instead of skipping the remainder.

    Returns None when there is nothing to extract from (including a session
    whose tail is already fully digested). No write transaction held; persist
    via the per-kind persist_* helpers inside one.
    """
    if since_message_id is None:
        where, params = "", (session_id,)
    else:
        # A chunk straddling the watermark was already covered by the digest
        # that set it; resume strictly above it.
        where = " AND (start_message_id IS NULL OR start_message_id > ?)"
        params = (session_id, since_message_id)
    rows = conn.execute(
        "SELECT id, text, start_message_id, end_message_id FROM chunks "
        f"WHERE session_id = ?{where} ORDER BY start_message_id",
        params,
    ).fetchall()
    if not rows:
        return None
    valid_chunk_ids = {r["id"] for r in rows}

    # Truncate whole chunks rather than mid-text, so the watermark can name a
    # real message boundary — a half-included chunk would either be re-read
    # forever (watermark below it) or silently dropped (watermark above it).
    combined_parts: list[str] = []
    covered: int | None = None
    started: int | None = None
    used = 0
    for r in rows:
        part = f"[chunk {r['id']}] {r['text']}"
        cost = len(part) + (4 if combined_parts else 0)  # the "\n\n---\n\n" join
        if combined_parts and used + cost > max_chars:
            break
        combined_parts.append(part)
        used += cost
        if r["start_message_id"] is not None and started is None:
            # Chunks come back ordered by start_message_id, so the first one
            # that carries a range is the low end of the window this call read.
            started = r["start_message_id"]
        if r["end_message_id"] is not None:
            covered = r["end_message_id"] if covered is None else max(
                covered, r["end_message_id"]
            )
    combined = "\n\n---\n\n".join(combined_parts)
    if len(combined) > max_chars:
        # ONE chunk longer than the whole cap (the loop always admits the first
        # part, whatever its size — `used` tracks `len(combined)` exactly, so
        # this branch can only mean the head chunk overflowed on its own).
        # Truncate it and still claim coverage: truncation keeps the HEAD, so
        # no later dream could read more of this chunk than this one just did.
        # Holding the watermark instead re-sends the identical slice, and
        # re-spends the call, on EVERY dream forever while storing nothing —
        # the session's episodes then freeze while its messages keep arriving
        # (2026-08-05: watermark pinned at 547 against message 2093 for five
        # days, silently). Losing the tail of one oversized chunk is the lesser
        # evil, and it is logged rather than silent. Same trade, same reasoning
        # as facts.oversized_turn in dreaming/facts.py.
        combined = combined[:max_chars]
        log.warning(
            "digest.oversized_chunk session_id=%s message_id=%s chars=%d cap=%d "
            "(tail not read)",
            session_id, covered, len(combined_parts[0]), max_chars,
        )

    system, template = (
        (SESSION_DIGEST_GRANULAR_SYSTEM, SESSION_DIGEST_GRANULAR_USER_TEMPLATE)
        if granular
        else (SESSION_DIGEST_SYSTEM, SESSION_DIGEST_USER_TEMPLATE)
    )
    request = LLMRequest(
        system=system,
        user=template.format(text=combined),
        response_format="json",
        max_tokens=max_tokens,
    )
    raw = llm.complete(request)

    # SESSION_DIGEST_SYSTEM asks for a top-level JSON object; fences/prose
    # around it are tolerated (dream 1013 — json_object mode is a request, not
    # a contract).
    data = loads_lenient(raw, expect="object")
    if data is None:
        log.warning("digest.parse_failure session_id=%s raw_len=%d",
                    session_id, len(raw) if isinstance(raw, str) else -1)
        return _empty()

    # A bare array (e.g. a stub LLM's "[]" default) or any non-object payload
    # yields an empty digest rather than crashing.
    if not isinstance(data, dict):
        # An empty array is that documented stub default and a routine "nothing
        # here", so it stays quiet. Any OTHER shape is a real reply we dropped;
        # _empty() holds the watermark, so a persistent one re-sends this slice
        # every dream and the log is the only way that surfaces.
        if data != []:
            log.warning("digest.shape_failure session_id=%s type=%s",
                        session_id, type(data).__name__)
        return _empty()

    raw_episodes = data.get("episodes", [])
    episodes = EpisodesExtraction(
        items=validate_episode_items(
            raw_episodes, valid_chunk_ids,
            # None on the blob arm = unbounded, i.e. today's behaviour exactly.
            max_items=max_episodes if granular else None,
        )
    )
    if (granular and max_episodes is not None
            and isinstance(raw_episodes, list) and len(raw_episodes) > max_episodes):
        # Logged, not silent: the cap is a runaway bound, and a session that
        # keeps hitting it is either a genuinely huge session or a prompt that
        # has started enumerating turns — the probe's granularity criterion
        # cannot tell those apart from the stored rows alone.
        log.warning(
            "digest.episode_cap session_id=%s returned=%d cap=%d (tail dropped)",
            session_id, len(raw_episodes), max_episodes,
        )
    raw_summary = data.get("summary")
    summary = clean_summary(raw_summary if isinstance(raw_summary, str) else None)
    procedures = ProceduresExtraction(
        items=validate_procedure_items(data.get("procedures", []))
    )
    return SessionDigest(
        episodes=episodes, summary=summary, procedures=procedures,
        covered_message_id=covered, start_message_id=started,
    )


def _empty() -> SessionDigest:
    """A parse failure, NOT coverage: `covered_message_id` stays None so the
    watermark does not advance and the slice is retried on the next dream.
    Advancing here would silently skip the slice forever — the same class of
    silent starvation that migration 024 exists to fix."""
    return SessionDigest(
        episodes=EpisodesExtraction(),
        summary=None,
        procedures=ProceduresExtraction(),
        parse_failed=True,
    )
