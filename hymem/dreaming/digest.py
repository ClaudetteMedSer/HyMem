from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from hymem.dreaming.episodes import EpisodesExtraction, validate_episode_items
from hymem.dreaming.procedures import ProceduresExtraction, validate_procedure_items
from hymem.dreaming.summary import clean_summary
from hymem.extraction.jsonio import loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import SESSION_DIGEST_SYSTEM, SESSION_DIGEST_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.digest")


@dataclass
class SessionDigest:
    """The three per-session tail extractions produced by one LLM call:
    episodes, a one-sentence summary, and procedures.

    ``covered_message_id`` is the highest ``chunks.end_message_id`` that made it
    into the LLM input (None when the chunks carry no message range). The runner
    stores it as ``sessions.digested_message_id`` so the next dream resumes
    above it — see :func:`extract_session_digest`."""
    episodes: EpisodesExtraction
    summary: str | None
    procedures: ProceduresExtraction
    covered_message_id: int | None = None
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
) -> SessionDigest | None:
    """Read the session's chunks once and run a single LLM call that returns
    episodes, summary, and procedures together (the batched replacement for the
    three separate tail calls).

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
    used = 0
    for r in rows:
        part = f"[chunk {r['id']}] {r['text']}"
        cost = len(part) + (4 if combined_parts else 0)  # the "\n\n---\n\n" join
        if combined_parts and used + cost > max_chars:
            break
        combined_parts.append(part)
        used += cost
        if r["end_message_id"] is not None:
            covered = r["end_message_id"] if covered is None else max(
                covered, r["end_message_id"]
            )
    combined = "\n\n---\n\n".join(combined_parts)
    if len(combined) > max_chars:
        # Single oversized chunk: keep the hard cap, and do not claim coverage
        # of a message range the LLM only partly saw.
        combined = combined[:max_chars]
        covered = None

    request = LLMRequest(
        system=SESSION_DIGEST_SYSTEM,
        user=SESSION_DIGEST_USER_TEMPLATE.format(text=combined),
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

    episodes = EpisodesExtraction(
        items=validate_episode_items(data.get("episodes", []), valid_chunk_ids)
    )
    raw_summary = data.get("summary")
    summary = clean_summary(raw_summary if isinstance(raw_summary, str) else None)
    procedures = ProceduresExtraction(
        items=validate_procedure_items(data.get("procedures", []))
    )
    return SessionDigest(
        episodes=episodes, summary=summary, procedures=procedures,
        covered_message_id=covered,
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
