from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from hymem.core.time import normalize_iso_timestamp, validate_event_clock


@dataclass(frozen=True)
class Message:
    id: int
    session_id: str
    role: str
    content: str
    source_peer_id: str | None = None
    source_workspace_id: str | None = None


def _session_is_pristine(conn: sqlite3.Connection, session_id: str) -> bool:
    """Return whether an unbound placeholder carries no durable state.

    Workspace binding is an ownership claim, so a summary, staged digest,
    episode, procedure, or any other session-scoped artifact is just as much
    history as a raw message.  This deliberately mirrors the v43 SQL guard.
    """
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        return True
    ignored = {"id", "source_workspace_id", "started_at"}
    for column in row.keys():
        if column in ignored:
            continue
        if row[column] not in (None, 0):
            return False
    dependencies = (
        ("messages", "session_id"),
        ("chunks", "session_id"),
        ("episodes", "session_id"),
        ("procedures", "session_id"),
        ("profile_staging", "session_id"),
        ("temporal_mentions", "session_id"),
        ("narrative_facts", "session_id"),
        ("fact_extraction_outcomes", "session_id"),
        ("message_retention_coverage", "source_session_id"),
        ("chunk_message_sources", "source_session_id"),
        ("user_profile", "source_session_id"),
        ("kg_evidence", "source_session_id"),
        ("kg_claim_observations", "source_session_id"),
        ("session_peers", "session_id"),
    )
    for table, column in dependencies:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone() is None:
            continue
        if conn.execute(
            f'SELECT 1 FROM "{table}" WHERE "{column}" = ? LIMIT 1',
            (session_id,),
        ).fetchone() is not None:
            return False
    return True


def open_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    source_workspace_id: str | None = None,
) -> None:
    """Open a native or workspace-bound session without guessing ownership.

    A non-empty legacy session is deliberately not claimed by the first
    Honcho request that happens to reuse its id. Empty legacy placeholders can
    be bound safely; a session already bound to another workspace fails.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be non-empty")
    if source_workspace_id is not None and (
        not isinstance(source_workspace_id, str)
        or not source_workspace_id.strip()
    ):
        raise ValueError("source_workspace_id must be non-empty")
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, source_workspace_id) VALUES (?, ?)",
        (session_id, source_workspace_id),
    )
    if source_workspace_id is None:
        return
    row = conn.execute(
        "SELECT source_workspace_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row["source_workspace_id"] == source_workspace_id:
        return
    if row["source_workspace_id"] is not None:
        raise ValueError("session belongs to a different workspace")
    if not _session_is_pristine(conn, session_id):
        raise ValueError("cannot infer workspace ownership for a legacy session")
    conn.execute(
        "UPDATE sessions SET source_workspace_id = ? WHERE id = ? "
        "AND source_workspace_id IS NULL",
        (source_workspace_id, session_id),
    )


def register_session_peer(
    conn: sqlite3.Connection,
    session_id: str,
    workspace_id: str,
    peer_id: str,
    role: str,
    *,
    configuration: dict | None = None,
) -> None:
    """Register one exact workspace peer and its session membership.

    Existing role mappings are authoritative. A caller cannot silently change
    an ambiguous peer from user to assistant (or vice versa) by adding it to a
    later session.
    """
    if role not in {"user", "assistant", "system", "tool"}:
        raise ValueError(f"unknown role: {role!r}")
    if not isinstance(peer_id, str) or not peer_id.strip():
        raise ValueError("peer_id must be non-empty")
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise ValueError("workspace_id must be non-empty")
    open_session(conn, session_id, source_workspace_id=workspace_id)
    existing = conn.execute(
        "SELECT role FROM peers WHERE id = ? AND workspace_id = ?",
        (peer_id, workspace_id),
    ).fetchone()
    if existing is not None and existing["role"] != role:
        raise ValueError("peer role conflicts with its workspace registration")
    conn.execute(
        "INSERT OR IGNORE INTO peers(id, workspace_id, role, metadata) "
        "VALUES (?, ?, ?, '{}')",
        (peer_id, workspace_id, role),
    )
    config_json = json.dumps(
        configuration or {}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )
    if configuration is None:
        conn.execute(
            "INSERT OR IGNORE INTO session_peers("
            "session_id, workspace_id, peer_id, configuration) "
            "VALUES (?, ?, ?, ?)",
            (session_id, workspace_id, peer_id, config_json),
        )
    else:
        conn.execute(
            "INSERT INTO session_peers(session_id, workspace_id, peer_id, configuration) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id, workspace_id, peer_id) DO UPDATE SET "
            "configuration = excluded.configuration",
            (session_id, workspace_id, peer_id, config_json),
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
    *,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> int:
    """Append one turn. `created_at`, when given, is the *event* time supplied by
    the caller (e.g. the real send time of a chat message, or a transcript's
    session date) and is stored in canonical UTC-millisecond form. It must be
    parseable and cannot lead
    the database's acceptance timestamp by more than the shared five-minute
    clock-skew allowance; otherwise later lifecycle materialization could make
    future truth look current. When omitted, SQLite's `CURRENT_TIMESTAMP`
    records ingestion/event time."""
    if role not in {"user", "assistant", "system", "tool"}:
        raise ValueError(f"unknown role: {role!r}")
    if (source_peer_id is None) != (source_workspace_id is None):
        raise ValueError(
            "source_peer_id and source_workspace_id must be provided together"
        )
    if source_peer_id is not None and (
        not isinstance(source_peer_id, str)
        or not source_peer_id.strip()
        or not isinstance(source_workspace_id, str)
        or not source_workspace_id.strip()
    ):
        raise ValueError("external message provenance must be non-empty")
    if created_at is not None:
        canonical_created_at = normalize_iso_timestamp(
            created_at,
            context="message created_at",
        )
        accepted_at = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
        validate_event_clock(
            conn,
            canonical_created_at,
            accepted_at,
            context="message source",
        )
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, source_peer_id, "
            "source_workspace_id, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id, role, source_peer_id, source_workspace_id,
                content, canonical_created_at,
            ),
        )
    else:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, source_peer_id, "
            "source_workspace_id, content) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, source_peer_id, source_workspace_id, content),
        )
    return int(cur.lastrowid)


def messages_for_session(conn: sqlite3.Connection, session_id: str) -> list[Message]:
    rows = conn.execute(
        "SELECT id, session_id, role, content, source_peer_id, "
        "source_workspace_id FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [Message(
        id=r["id"], session_id=r["session_id"], role=r["role"],
        content=r["content"], source_peer_id=r["source_peer_id"],
        source_workspace_id=r["source_workspace_id"],
    ) for r in rows]


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
        "SELECT id, session_id, role, content, source_peer_id, "
        "source_workspace_id FROM messages "
        "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [
        Message(
            id=r["id"], session_id=r["session_id"], role=r["role"],
            content=r["content"], source_peer_id=r["source_peer_id"],
            source_workspace_id=r["source_workspace_id"],
        )
        for r in reversed(rows)
    ]
