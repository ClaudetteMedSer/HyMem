from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    id: int
    session_id: str
    role: str
    content: str


def open_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id) VALUES (?)",
        (session_id,),
    )


def close_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ? AND ended_at IS NULL",
        (session_id,),
    )


def append_message(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    created_at: str | None = None,
) -> int:
    """Append one turn. `created_at`, when given, is the *event* time supplied by
    the caller (e.g. the real send time of a chat message, or a transcript's
    session date) and is stored verbatim — ISO-8601 strings sort correctly under
    the `ORDER BY created_at` paths that drive chronological/temporal retrieval.
    When omitted, SQLite's `CURRENT_TIMESTAMP` default records ingestion time."""
    if role not in {"user", "assistant", "system", "tool"}:
        raise ValueError(f"unknown role: {role!r}")
    if created_at:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, created_at),
        )
    else:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
    return int(cur.lastrowid)


def messages_for_session(conn: sqlite3.Connection, session_id: str) -> list[Message]:
    rows = conn.execute(
        "SELECT id, session_id, role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [Message(id=r["id"], session_id=r["session_id"], role=r["role"], content=r["content"]) for r in rows]


def recent_messages(
    conn: sqlite3.Connection, session_id: str, limit: int
) -> list[Message]:
    """Return the most-recent `limit` messages for the session, in
    chronological (ascending) order. Returns [] when limit <= 0.

    Selects newest-first (ORDER BY id DESC LIMIT limit) then reverses, so the
    caller gets the working-memory window oldest -> newest.
    """
    if limit <= 0:
        return []
    rows = conn.execute(
        "SELECT id, session_id, role, content FROM messages "
        "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [
        Message(id=r["id"], session_id=r["session_id"], role=r["role"], content=r["content"])
        for r in reversed(rows)
    ]
