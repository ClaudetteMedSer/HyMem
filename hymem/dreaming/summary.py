from __future__ import annotations

import logging
import sqlite3

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import SESSION_SUMMARY_SYSTEM, SESSION_SUMMARY_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.summary")


def extract_session_summary(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
) -> str | None:
    """Run the session-summary LLM call. Returns the new summary string,
    or None when nothing new is needed (already summarized, no content, or
    LLM output rejected). No write transaction held.
    """
    existing = conn.execute(
        "SELECT summary FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if existing and existing["summary"]:
        return None

    chunks = conn.execute(
        "SELECT text FROM chunks WHERE session_id = ? ORDER BY start_message_id",
        (session_id,),
    ).fetchall()

    episodes = conn.execute(
        "SELECT title, summary FROM episodes WHERE session_id = ?",
        (session_id,),
    ).fetchall()

    parts: list[str] = []
    for c in chunks:
        parts.append(c["text"])
    for e in episodes:
        parts.append(f"[Episode: {e['title']}] {e['summary']}")

    if not parts:
        return None

    combined = "\n\n".join(parts)
    if len(combined) > 8000:
        combined = combined[:8000]

    request = LLMRequest(
        system=SESSION_SUMMARY_SYSTEM,
        user=SESSION_SUMMARY_USER_TEMPLATE.format(text=combined),
        response_format="text",
    )
    raw = llm.complete(request)
    summary = raw.strip().strip('"').strip("'")

    if not summary or len(summary) < 10:
        return None

    return summary[:500]


def persist_session_summary(
    conn: sqlite3.Connection,
    session_id: str,
    summary: str,
) -> None:
    """Write a session summary. Caller wraps in core_db.transaction()."""
    conn.execute(
        "UPDATE sessions SET summary = ? WHERE id = ?",
        (summary, session_id),
    )
    log.debug("summary.persisted session_id=%s len=%d", session_id, len(summary))
