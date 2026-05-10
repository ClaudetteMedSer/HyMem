from __future__ import annotations

import re
import sqlite3

from hymem.dreaming.canonicalize import normalize

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]{1,40}")


def _candidates(text: str) -> list[str]:
    """Tokenize and produce normalized lookup keys (matches query.entities)."""
    raw_tokens = {m.group(0) for m in _TOKEN.finditer(text)}
    candidates = {normalize(t) for t in raw_tokens if len(t) >= 2}
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            candidates.add(normalize(phrase))
    return [c for c in candidates if c]


def _resolve_canonicals(conn: sqlite3.Connection, candidates: list[str]) -> set[str]:
    if not candidates:
        return set()
    placeholders = ",".join("?" * len(candidates))
    rows = conn.execute(
        f"""
        SELECT DISTINCT canonical FROM entity_aliases WHERE alias IN ({placeholders})
        UNION
        SELECT DISTINCT subject_canonical FROM knowledge_graph WHERE subject_canonical IN ({placeholders})
        UNION
        SELECT DISTINCT object_canonical FROM knowledge_graph WHERE object_canonical IN ({placeholders})
        """,
        candidates + candidates + candidates,
    ).fetchall()
    return {r[0] for r in rows}


def index_chunk_mentions(
    conn: sqlite3.Connection, chunk_id: str, text: str
) -> int:
    """Scan `text` against known canonical entities and populate entity_mentions.

    Returns the number of mentions inserted (post INSERT OR IGNORE).
    """
    canonicals = _resolve_canonicals(conn, _candidates(text))
    inserted = 0
    for canonical in canonicals:
        cur = conn.execute(
            "INSERT OR IGNORE INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, ?)",
            (chunk_id, canonical),
        )
        inserted += cur.rowcount or 0
    return inserted
