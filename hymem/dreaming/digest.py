from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

from hymem.dreaming.episodes import EpisodesExtraction, validate_episode_items
from hymem.dreaming.procedures import ProceduresExtraction, validate_procedure_items
from hymem.dreaming.summary import clean_summary
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import SESSION_DIGEST_SYSTEM, SESSION_DIGEST_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.digest")


@dataclass
class SessionDigest:
    """The three per-session tail extractions produced by one LLM call:
    episodes, a one-sentence summary, and procedures."""
    episodes: EpisodesExtraction
    summary: str | None
    procedures: ProceduresExtraction


def extract_session_digest(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
    *,
    max_tokens: int,
    max_chars: int,
) -> SessionDigest | None:
    """Read the session's chunks once and run a single LLM call that returns
    episodes, summary, and procedures together (the batched replacement for the
    three separate tail calls).

    Returns None when there is nothing to extract from. No write transaction
    held; persist via the per-kind persist_* helpers inside one.
    """
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE session_id = ? ORDER BY start_message_id",
        (session_id,),
    ).fetchall()
    if not rows:
        return None
    valid_chunk_ids = {r["id"] for r in rows}

    combined = "\n\n---\n\n".join(f"[chunk {r['id']}] {r['text']}" for r in rows)
    if len(combined) > max_chars:
        combined = combined[:max_chars]

    request = LLMRequest(
        system=SESSION_DIGEST_SYSTEM,
        user=SESSION_DIGEST_USER_TEMPLATE.format(text=combined),
        response_format="json",
        max_tokens=max_tokens,
    )
    raw = llm.complete(request)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _empty()

    # A bare array (e.g. a stub LLM's "[]" default) or any non-object payload
    # yields an empty digest rather than crashing.
    if not isinstance(data, dict):
        return _empty()

    episodes = EpisodesExtraction(
        items=validate_episode_items(data.get("episodes", []), valid_chunk_ids)
    )
    raw_summary = data.get("summary")
    summary = clean_summary(raw_summary if isinstance(raw_summary, str) else None)
    procedures = ProceduresExtraction(
        items=validate_procedure_items(data.get("procedures", []))
    )
    return SessionDigest(episodes=episodes, summary=summary, procedures=procedures)


def _empty() -> SessionDigest:
    return SessionDigest(
        episodes=EpisodesExtraction(),
        summary=None,
        procedures=ProceduresExtraction(),
    )
