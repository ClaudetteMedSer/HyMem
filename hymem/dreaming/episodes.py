from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import EPISODE_SYSTEM, EPISODE_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.episodes")


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
        if not title.strip() or not summary.strip():
            continue
        items.append(item)
    return EpisodesExtraction(items=items)


def persist_episodes(
    conn: sqlite3.Connection,
    session_id: str,
    extraction: EpisodesExtraction,
) -> int:
    """Insert validated episodes. Caller wraps in core_db.transaction()."""
    count = 0
    for item in extraction.items:
        title = item.get("title", "").strip()
        summary = item.get("summary", "").strip()
        episode_id = f"{session_id}@{count}"
        outcome = item.get("outcome")
        key_entities = json.dumps(item.get("key_entities", []))

        conn.execute(
            """INSERT OR IGNORE INTO episodes(id, session_id, title, summary,
               outcome, key_entities)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (episode_id, session_id, title, summary,
             outcome if outcome in ("resolved", "blocked", "deferred", "informational") else None,
             key_entities),
        )
        count += 1

    if count:
        log.debug("episodes.persisted session_id=%s count=%d", session_id, count)
    return count
