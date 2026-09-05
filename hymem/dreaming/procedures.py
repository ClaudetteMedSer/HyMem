from __future__ import annotations

import json
import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field

from hymem.extraction.jsonio import loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import PROCEDURE_SYSTEM, PROCEDURE_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.procedures")


@dataclass
class ProceduresExtraction:
    """Validated procedure items ready to persist. Each item carries the
    normalized name, description, steps, triggers, and entities lists."""
    items: list[dict] = field(default_factory=list)


def extract_procedures_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    llm: LLMClient,
) -> ProceduresExtraction | None:
    """Read the session's chunks and episodes and run the procedure-extraction
    LLM call. Returns None when there is nothing to extract from. No write
    transaction held; persist via persist_procedures inside one.
    """
    chunks = conn.execute(
        "SELECT id, text FROM chunks WHERE session_id = ? "
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
        parts.append(f"[chunk] {c['text']}")
    for e in episodes:
        parts.append(f"[episode: {e['title']}] {e['summary']}")

    if not parts:
        return None

    combined = "\n\n---\n\n".join(parts)
    if len(combined) > 12000:
        combined = combined[:12000]

    request = LLMRequest(
        system=PROCEDURE_SYSTEM,
        user=PROCEDURE_USER_TEMPLATE.format(text=combined),
        response_format="json",
    )
    raw = llm.complete(request)

    # PROCEDURE_SYSTEM asks for a bare JSON array; fences/prose around it are
    # tolerated (dream 1013 — json_object mode is a request, not a contract).
    data = loads_lenient(raw, expect="array")
    if data is None:
        log.warning("procedures.parse_failure session_id=%s raw_len=%d",
                    session_id, len(raw) if isinstance(raw, str) else -1)
        return ProceduresExtraction()
    if not isinstance(data, list):
        # validate_procedure_items() absorbs a non-list into [], which reads
        # downstream as "no procedures in this session" rather than a drop.
        log.warning("procedures.shape_failure session_id=%s type=%s",
                    session_id, type(data).__name__)
        return ProceduresExtraction()

    return ProceduresExtraction(items=validate_procedure_items(data))


def validate_procedure_items(data: object) -> list[dict]:
    """Validate raw LLM procedure items into clean dicts ready to persist.

    Shared by the standalone procedure call and the batched session digest.
    Drops items missing a name or valid steps, renumbers steps from 1, and
    normalizes triggers/entities. Returns [] for any non-list ``data``.
    """
    if not isinstance(data, list):
        return []
    items: list[dict] = []
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

        valid_steps: list[dict] = []
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

        items.append({
            "name": name.strip(),
            "description": description.strip()[:500] if description else None,
            "steps": valid_steps,
            "triggers": triggers,
            "entities_involved": entities,
        })

    return items


def persist_procedures(
    conn: sqlite3.Connection,
    session_id: str,
    extraction: ProceduresExtraction,
) -> int:
    """Upsert validated procedures by stable session/name identity.

    The old ``@proc0`` ordinal restarted for every digest tail, so the first
    procedure from every later slice collided and was silently ignored.  A
    normalized-name identity is stable across re-dreams and additive across
    unrelated tails; existing legacy rows with the same name are reused.
    """
    count = 0
    for item in extraction.items:
        existing = conn.execute(
            "SELECT id FROM procedures WHERE session_id = ? "
            "AND lower(name) = lower(?) ORDER BY id LIMIT 1",
            (session_id, item["name"]),
        ).fetchone()
        name_hash = hashlib.sha1(
            f"{session_id}\0{item['name'].strip().casefold()}".encode("utf-8")
        ).hexdigest()[:16]
        procedure_id = existing["id"] if existing else f"{session_id}@proc_{name_hash}"
        conn.execute(
            """INSERT INTO procedures(id, session_id, name, description,
               steps, triggers, entities_involved)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   name = excluded.name,
                   description = excluded.description,
                   steps = excluded.steps,
                   triggers = excluded.triggers,
                   entities_involved = excluded.entities_involved,
                   status = 'active'""",
            (
                procedure_id,
                session_id,
                item["name"],
                item["description"],
                json.dumps(item["steps"]),
                json.dumps(item["triggers"]),
                json.dumps(item["entities_involved"]),
            ),
        )
        count += 1

    if count:
        log.debug("procedures.persisted session_id=%s count=%d", session_id, count)
    return count
