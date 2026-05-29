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
) -> int:
    if role not in {"user", "assistant", "system", "tool"}:
        raise ValueError(f"unknown role: {role!r}")
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
