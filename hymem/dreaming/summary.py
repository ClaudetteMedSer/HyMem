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
        "SELECT text FROM chunks WHERE session_id = ? "
        "AND chunk_kind = 'extraction' ORDER BY start_message_id",
        (session_id,),
    ).fetchall()

    episodes = conn.execute(
        "SELECT e.title, e.summary FROM episodes e "
        "JOIN sessions s ON s.id = e.session_id "
        "WHERE e.session_id = ? AND (e.digest_generation IS NULL "
        "OR e.digest_generation = s.digest_published_generation)",
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
    return clean_summary(raw)


def clean_summary(raw: str | None) -> str | None:
    """Normalize a raw LLM summary string. Returns None when empty or too short
    to be useful. Shared by the standalone summary call and the session digest."""
    if not raw:
        return None
    summary = raw.strip().strip('"').strip("'")
    if not summary or len(summary) < 10:
        return None
    return summary[:500]


def persist_session_summary(
    conn: sqlite3.Connection,
    session_id: str,
    summary: str,
) -> None:
    """Write an operator-owned session summary.

    Automatic dreaming uses :func:`persist_auto_session_summary`; keeping this
    legacy/public writer operator-owned makes the non-overwrite contract
    explicit instead of relying on a truthy-string heuristic.
    """
    conn.execute(
        "UPDATE sessions SET summary = ?, summary_source = 'operator' WHERE id = ?",
        (summary, session_id),
    )
    log.debug("summary.persisted session_id=%s len=%d", session_id, len(summary))


def persist_auto_session_summary(
    conn: sqlite3.Connection,
    session_id: str,
    summary: str,
    *,
    covered_message_id: int | None,
    partial_message_id: int | None = None,
    covered_message_offset: int = 0,
) -> None:
    """Persist a rolling automatic summary without clobbering curated text."""
    conn.execute(
        """
        UPDATE sessions
        SET auto_summary = ?,
            auto_summary_message_id = ?,
            auto_summary_partial_message_id = ?,
            auto_summary_message_offset = ?,
            summary = CASE
                WHEN summary IS NULL
                  OR (summary_source = 'auto' AND summary IS auto_summary) THEN ?
                ELSE summary
            END,
            summary_source = CASE
                WHEN summary IS NULL
                  OR (summary_source = 'auto' AND summary IS auto_summary) THEN 'auto'
                WHEN summary_source = 'auto' THEN 'operator'
                WHEN summary_source IS NULL THEN 'operator'
                ELSE summary_source
            END
        WHERE id = ?
        """,
        (
            summary,
            covered_message_id,
            partial_message_id,
            int(covered_message_offset),
            summary,
            session_id,
        ),
    )
    log.debug(
        "summary.auto_persisted session_id=%s len=%d "
        "covered_message_id=%s partial_message_id=%s offset=%d",
        session_id,
        len(summary),
        covered_message_id,
        partial_message_id,
        covered_message_offset,
    )


def effective_session_summary(row: sqlite3.Row | None) -> str:
    """Render the operator/legacy and rolling automatic summary coherently."""
    if row is None:
        return ""
    summary = row["summary"] or ""
    auto = row["auto_summary"] or ""
    source = row["summary_source"]
    if source == "auto":
        return auto or summary
    if summary and auto and summary != auto:
        return (
            f"Operator/legacy summary: {summary}\n\n"
            f"Automatic rolling summary: {auto}"
        )
    return summary or auto
