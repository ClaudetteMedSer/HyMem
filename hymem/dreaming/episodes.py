from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import EPISODE_SYSTEM, EPISODE_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.episodes")

_VALID_OUTCOMES = frozenset({"resolved", "blocked", "deferred", "informational"})


@dataclass
class EpisodesExtraction:
    """Validated episode items ready to persist. Empty list = LLM returned
    nothing usable; None at the call boundary = nothing to extract."""
    items: list[dict] = field(default_factory=list)


def extract_episodes_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
) -> EpisodesExtraction | None:
    """Read the session's chunks and run the episode-extraction LLM call.
    Returns None when there is nothing to extract from. No write transaction
    held; persist via persist_episodes inside one.
    """
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE session_id = ? ORDER BY start_message_id",
        (session_id,),
    ).fetchall()
    if not rows:
        return None
    valid_chunk_ids = {r["id"] for r in rows}

    combined = "\n\n---\n\n".join(
        f"[chunk {r['id']}] {r['text']}" for r in rows
    )

    request = LLMRequest(
        system=EPISODE_SYSTEM,
        user=EPISODE_USER_TEMPLATE.format(text=combined),
        response_format="json",
    )
    raw = llm.complete(request)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return EpisodesExtraction()

    if not isinstance(data, list):
        return EpisodesExtraction()

    items: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        summary = item.get("summary", "")
        if not isinstance(title, str) or not isinstance(summary, str):
            continue
        if not title.strip() or not summary.strip():
            continue
        raw_chunk_ids = item.get("chunk_ids", [])
        if not isinstance(raw_chunk_ids, list):
            raw_chunk_ids = []
        # Drop anything the LLM hallucinated that isn't in the input.
        chunk_ids = [c for c in raw_chunk_ids if isinstance(c, str) and c in valid_chunk_ids]
        clean = dict(item)
        clean["title"] = title.strip()
        clean["summary"] = summary.strip()
        clean["chunk_ids"] = chunk_ids
        items.append(clean)
    return EpisodesExtraction(items=items)


def _resolve_message_range(
    conn: sqlite3.Connection, chunk_ids: list[str]
) -> tuple[int | None, int | None]:
    """Return (min start_message_id, max end_message_id) across the named chunks,
    or (None, None) if the list is empty or no chunks resolved."""
    if not chunk_ids:
        return None, None
    placeholders = ",".join("?" * len(chunk_ids))
    row = conn.execute(
        f"""
        SELECT MIN(start_message_id) AS s, MAX(end_message_id) AS e
        FROM chunks WHERE id IN ({placeholders})
        """,
        tuple(chunk_ids),
    ).fetchone()
    if row is None:
        return None, None
    return row["s"], row["e"]


def _resolve_participants(
    conn: sqlite3.Connection,
    session_id: str,
    start_msg: int | None,
    end_msg: int | None,
) -> list[str]:
    """Distinct, sorted roles in the message range (e.g. ['assistant', 'user']).
    Falls back to all session roles if the range is unknown."""
    if start_msg is None or end_msg is None:
        rows = conn.execute(
            "SELECT DISTINCT role FROM messages WHERE session_id = ? ORDER BY role",
            (session_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT role FROM messages
            WHERE session_id = ? AND id BETWEEN ? AND ?
            ORDER BY role
            """,
            (session_id, start_msg, end_msg),
        ).fetchall()
    return [r["role"] for r in rows]


def _episode_id(
    session_id: str,
    start_msg: int | None,
    end_msg: int | None,
    title: str,
) -> str:
    """Stable id for an episode.

    Prefers the message range — same range across re-dreams = same id, so an
    UPSERT updates the title/summary in place. When the LLM didn't provide
    chunk_ids (so range is unknown) we fall back to a content hash so the row
    is still re-findable on the next run for the same content.
    """
    if start_msg is not None and end_msg is not None:
        return f"{session_id}@{start_msg}-{end_msg}"
    digest = hashlib.sha1(
        f"{session_id}|{title.strip().lower()}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{session_id}@h{digest}"


def persist_episodes(
    conn: sqlite3.Connection,
    session_id: str,
    extraction: EpisodesExtraction,
) -> int:
    """Insert/UPSERT validated episodes. Caller wraps in core_db.transaction().

    Each episode gets a stable id derived from the message range (see
    ``_episode_id``). UPSERT-on-conflict refreshes title/summary/outcome
    /key_entities/participants for re-dreams without reshuffling rowids —
    important for FTS and vec_episodes alignment.
    """
    count = 0
    for item in extraction.items:
        title = item.get("title", "").strip()
        summary = item.get("summary", "").strip()
        if not title or not summary:
            continue
        chunk_ids = item.get("chunk_ids", []) or []
        start_msg, end_msg = _resolve_message_range(conn, chunk_ids)
        participants = _resolve_participants(conn, session_id, start_msg, end_msg)
        outcome = item.get("outcome")
        outcome = outcome if outcome in _VALID_OUTCOMES else None
        key_entities = json.dumps(item.get("key_entities", []))
        episode_id = _episode_id(session_id, start_msg, end_msg, title)

        conn.execute(
            """
            INSERT INTO episodes(
                id, session_id, title, summary, participants,
                start_message_id, end_message_id, outcome, key_entities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                participants = excluded.participants,
                start_message_id = excluded.start_message_id,
                end_message_id = excluded.end_message_id,
                outcome = excluded.outcome,
                key_entities = excluded.key_entities
            """,
            (
                episode_id, session_id, title, summary,
                json.dumps(participants),
                start_msg, end_msg, outcome, key_entities,
            ),
        )
        count += 1

    if count:
        log.debug("episodes.persisted session_id=%s count=%d", session_id, count)
    return count
