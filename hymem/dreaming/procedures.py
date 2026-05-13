from __future__ import annotations

import json
import logging
import sqlite3

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import PROCEDURE_SYSTEM, PROCEDURE_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.procedures")


def extract_procedures_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
) -> int:
    """Extract procedures from a session's chunks and episodes. Returns count created."""
    chunks = conn.execute(
        "SELECT id, text FROM chunks WHERE session_id = ? ORDER BY start_message_id",
        (session_id,),
    ).fetchall()

    episodes = conn.execute(
        "SELECT title, summary FROM episodes WHERE session_id = ?",
        (session_id,),
    ).fetchall()

    parts: list[str] = []
    for c in chunks:
        parts.append(f"[chunk] {c['text']}")
    for e in episodes:
        parts.append(f"[episode: {e['title']}] {e['summary']}")

    if not parts:
        return 0

    combined = "\n\n---\n\n".join(parts)
    if len(combined) > 12000:
        combined = combined[:12000]

    request = LLMRequest(
        system=PROCEDURE_SYSTEM,
        user=PROCEDURE_USER_TEMPLATE.format(text=combined),
        response_format="json",
    )
    raw = llm.complete(request)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    if not isinstance(data, list):
        return 0

    count = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        if not name.strip():
            continue

        description = item.get("description", "") or ""
        steps_list = item.get("steps", [])
        if not isinstance(steps_list, list):
            continue

        valid_steps = []
        for s in steps_list:
            if not isinstance(s, dict):
                continue
            order = s.get("order")
            action = s.get("action")
            if isinstance(order, (int, float)) and isinstance(action, str) and action.strip():
                tool = s.get("tool")
                valid_steps.append({
                    "order": int(order),
                    "action": action.strip(),
                    "tool": tool if isinstance(tool, str) and tool.strip() else None,
                })

        if not valid_steps:
            continue

        valid_steps.sort(key=lambda x: x["order"])
        for i, s in enumerate(valid_steps):
            s["order"] = i + 1

        triggers = item.get("triggers", [])
        if isinstance(triggers, list):
            triggers = [t for t in triggers if isinstance(t, str) and t.strip()]
        else:
            triggers = []

        entities = item.get("entities_involved", [])
        if isinstance(entities, list):
            entities = [e for e in entities if isinstance(e, str) and e.strip()]
        else:
            entities = []

        procedure_id = f"{session_id}@proc{count}"

        conn.execute(
            """INSERT OR IGNORE INTO procedures(id, session_id, name, description,
               steps, triggers, entities_involved)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                procedure_id,
                session_id,
                name.strip(),
                description.strip()[:500] if description else None,
                json.dumps(valid_steps),
                json.dumps(triggers),
                json.dumps(entities),
            ),
        )
        count += 1

    if count:
        log.debug("procedures.extracted session_id=%s count=%d", session_id, count)
    return count
