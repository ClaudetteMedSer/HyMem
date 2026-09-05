"""Durable, message-complete input stream for session dreaming.

Salience chunks are deliberately selective and may overlap.  They remain the
right unit for graph extraction, but they cannot be the source of truth for
digest coverage.  This module materializes one namespaced, canonical JSONL
artifact per raw message and exposes those artifacts as an ordered stream.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from hymem.core.message_records import (
    canonical_message_record,
)
from hymem.core.time import normalize_iso_timestamp
from hymem.dreaming.message_coverage import (
    LOSSLESS_COVERAGE_VERSION,
    LOSSLESS_READ_VERSIONS,
    _has_columns,
    coverage_chunk_id,
    record_message_coverage,
)


@dataclass(frozen=True)
class CoveredMessage:
    message_id: int
    session_id: str
    role: str
    content: str
    chunk_id: str
    source_created_at: str | None = None
    source_peer_id: str | None = None
    source_workspace_id: str | None = None


def _materialize_one_message_coverage(
    conn: sqlite3.Connection,
    session_id: str,
    row: sqlite3.Row,
) -> None:
    """Write and validate one canonical ordered-stream artifact.

    Kept separate from the frontier walk so the v39 upgrade can backfill
    already-existing raw rows that sit below a legacy cursor.  Callers advance
    the producer frontier only after every intended row has succeeded.
    """
    message_id = int(row["id"])
    chunk_id = coverage_chunk_id(session_id, message_id)
    record, _, _, _ = canonical_message_record(
        message_id=message_id,
        session_id=session_id,
        role=row["role"],
        content=row["content"],
        source_created_at=row["created_at"],
        source_peer_id=row["source_peer_id"],
        source_workspace_id=row["source_workspace_id"],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO chunks(
            id, session_id, start_message_id, end_message_id,
            salience_reason, text, chunk_kind
        ) VALUES (?, ?, ?, ?, 'lossless_message', ?, 'coverage')
        """,
        (chunk_id, session_id, message_id, message_id, record),
    )
    chunk = conn.execute(
        "SELECT session_id, start_message_id, end_message_id, text, chunk_kind "
        "FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    if (
        chunk is None
        or chunk["session_id"] != session_id
        or int(chunk["start_message_id"]) != message_id
        or int(chunk["end_message_id"]) != message_id
        or chunk["text"] != record
        or chunk["chunk_kind"] != "coverage"
    ):
        raise RuntimeError(
            f"coverage artifact identity collision for message {message_id}"
        )
    record_message_coverage(
        conn,
        message_id=message_id,
        chunk_id=chunk_id,
        coverage_version=LOSSLESS_COVERAGE_VERSION,
    )


def materialize_message_coverage(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    limit: int | None = None,
) -> int:
    """Persist exact artifacts for every source message above the cursor.

    The caller should wrap this in a transaction.  The session cursor advances
    only after both the immutable chunk and its v37 proof row exist.  A failed
    write therefore rolls back the cursor with the artifact; a crash between
    autocommit statements merely leaves an idempotent artifact to validate on
    retry and does not skip the message.
    """
    session = conn.execute(
        "SELECT coverage_message_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if session is None:
        raise ValueError(f"unknown session: {session_id}")
    cursor = session["coverage_message_id"]
    peer_select = (
        "source_peer_id, source_workspace_id"
        if _has_columns(
            conn, "messages", "source_peer_id", "source_workspace_id"
        )
        else "NULL AS source_peer_id, NULL AS source_workspace_id"
    )
    sql = (
        f"SELECT id, role, content, created_at, {peer_select} "
        "FROM messages WHERE session_id = ? AND id > ? ORDER BY id"
    )
    params: tuple = (session_id, int(cursor) if cursor is not None else -1)
    if limit is not None:
        if limit <= 0:
            return 0
        sql += " LIMIT ?"
        params = (*params, int(limit))
    rows = conn.execute(sql, params).fetchall()

    created = 0
    for row in rows:
        message_id = int(row["id"])
        _materialize_one_message_coverage(conn, session_id, row)
        conn.execute(
            "UPDATE sessions SET coverage_message_id = "
            "MAX(COALESCE(coverage_message_id, -1), ?) WHERE id = ?",
            (message_id, session_id),
        )
        created += 1
    # Never infer a session frontier from arbitrary v37 proof rows.  That
    # ledger deliberately permits independent/sparse proofs and therefore its
    # MAX(message_id) says nothing about missing turns.  Only this producer's
    # raw-message walk advances the cursor; imported v38 sessions carry their
    # already-proven cursor explicitly, while older/pruned sessions stay
    # conservatively frozen until genuinely new raw input is covered.
    return created


def backfill_all_message_coverage(
    conn: sqlite3.Connection,
    session_id: str,
) -> int:
    """Materialize every surviving raw row for a migration-era session.

    This is intentionally not the hot append path.  Pre-v39 profile rows can
    point at a live USER message that predates the v38 producer cursor; without
    a canonical artifact an immediate export/import would drop that provenance.
    The upgrade covers the complete surviving session first and only then moves
    the frontier, so it never blesses a sparse subset as an ordered stream.
    """
    session = conn.execute(
        "SELECT coverage_message_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if session is None:
        raise ValueError(f"unknown session: {session_id}")
    peer_select = (
        "source_peer_id, source_workspace_id"
        if _has_columns(
            conn, "messages", "source_peer_id", "source_workspace_id"
        )
        else "NULL AS source_peer_id, NULL AS source_workspace_id"
    )
    rows = conn.execute(
        f"SELECT id, role, content, created_at, {peer_select} "
        "FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    for row in rows:
        _materialize_one_message_coverage(conn, session_id, row)
    if rows:
        newest = int(rows[-1]["id"])
        conn.execute(
            "UPDATE sessions SET coverage_message_id = "
            "MAX(COALESCE(coverage_message_id, -1), ?) WHERE id = ?",
            (newest, session_id),
        )
    return len(rows)


COVERAGE_VALIDATION_COLUMNS = """
    mc.message_id, mc.source_session_id, mc.source_role,
    mc.source_peer_id, mc.source_workspace_id, mc.source_created_at,
    mc.message_content_hash, mc.hash_version, mc.record_version,
    mc.coverage_version,
    c.id AS chunk_id, c.session_id AS chunk_session_id,
    c.start_message_id, c.end_message_id, c.text, c.chunk_kind,
    source_session.id AS bound_session_id,
    source_session.source_workspace_id AS bound_workspace_id,
    source_peer.role AS registered_peer_role,
    member.peer_id AS member_peer_id,
    raw.id AS raw_message_id, raw.session_id AS raw_session_id,
    raw.role AS raw_role, raw.content AS raw_content,
    raw.created_at AS raw_created_at,
    raw.source_peer_id AS raw_peer_id,
    raw.source_workspace_id AS raw_workspace_id
"""

COVERAGE_VALIDATION_JOINS = """
    JOIN chunks c ON c.id = mc.chunk_id
    LEFT JOIN sessions source_session
      ON source_session.id = mc.source_session_id
    LEFT JOIN peers source_peer
      ON source_peer.id = mc.source_peer_id
     AND source_peer.workspace_id = mc.source_workspace_id
    LEFT JOIN session_peers member
      ON member.session_id = mc.source_session_id
     AND member.workspace_id = mc.source_workspace_id
     AND member.peer_id = mc.source_peer_id
    LEFT JOIN messages raw
      ON raw.id = mc.message_id
     AND raw.session_id = mc.source_session_id
"""


def validate_message_coverage_row(row: sqlite3.Row) -> CoveredMessage:
    """Validate one fully joined coverage row without issuing SQL."""
    def exact_integer(value: object, field: str) -> int:
        # SQLite's dynamic typing permits damaged INTEGER-affinity columns to
        # contain strings, floats, NaN, or infinity. Never coerce those into a
        # valid proof range: conversion itself can raise OverflowError and a
        # numerically equal float is still not the canonical stored shape.
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError(f"coverage proof has an invalid {field}")
        return value

    try:
        mid = exact_integer(row["message_id"], "message id")
        start_message_id = exact_integer(
            row["start_message_id"], "start message id"
        )
        end_message_id = exact_integer(
            row["end_message_id"], "end message id"
        )
        raw_message_id = (
            None
            if row["raw_message_id"] is None
            else exact_integer(row["raw_message_id"], "raw message id")
        )
    except (IndexError, KeyError):
        raise RuntimeError("coverage proof lacks numeric source fields") from None
    external_source_time_valid = True
    if row["source_peer_id"] is not None:
        try:
            normalize_iso_timestamp(
                row["source_created_at"],
                context="external coverage source",
            )
        except ValueError:
            external_source_time_valid = False
    physical_lines = row["text"].split("\n") if isinstance(row["text"], str) else []
    payload = None
    for line in physical_lines:
        try:
            candidate = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict) and candidate.get("id") == mid:
            payload = candidate
            break
    valid = bool(
        isinstance(row["source_session_id"], str)
        and row["source_session_id"]
        and external_source_time_valid
        and row["source_role"] in {"user", "assistant", "system", "tool"}
        and (
            (row["source_peer_id"] is None
             and row["source_workspace_id"] is None)
            or (
                isinstance(row["source_peer_id"], str)
                and bool(row["source_peer_id"].strip())
                and isinstance(row["source_workspace_id"], str)
                and bool(row["source_workspace_id"].strip())
            )
        )
        and row["chunk_session_id"] == row["source_session_id"]
        and start_message_id <= mid <= end_message_id
        and isinstance(payload, dict)
        and payload.get("id") == mid
        and payload.get("role") == row["source_role"]
        and isinstance(payload.get("content"), str)
    )
    canonical = ""
    if valid:
        try:
            canonical, expected_hash, hash_version, record_version = (
                canonical_message_record(
                    message_id=mid,
                    session_id=row["source_session_id"],
                    role=row["source_role"],
                    content=payload["content"],
                    source_created_at=row["source_created_at"],
                    source_peer_id=row["source_peer_id"],
                    source_workspace_id=row["source_workspace_id"],
                )
            )
        except (TypeError, ValueError):
            valid = False
        else:
            valid = bool(
                canonical in physical_lines
                and row["message_content_hash"] == expected_hash
                and row["hash_version"] == hash_version
                and row["record_version"] == record_version
            )
    if valid and row["coverage_version"] in LOSSLESS_READ_VERSIONS:
        valid = bool(
            row["chunk_id"]
            == coverage_chunk_id(row["source_session_id"], mid)
            and start_message_id == mid
            and end_message_id == mid
            and row["chunk_kind"] == "coverage"
            and row["text"] == canonical
        )
    if valid and row["source_peer_id"] is None:
        valid = bool(
            row["bound_session_id"] == row["source_session_id"]
            and row["bound_workspace_id"] is None
        )
    elif valid:
        valid = bool(
            row["bound_session_id"] == row["source_session_id"]
            and row["bound_workspace_id"] == row["source_workspace_id"]
            and row["registered_peer_role"] == row["source_role"]
            and row["member_peer_id"] == row["source_peer_id"]
        )
    if valid and raw_message_id is not None:
        valid = bool(
            raw_message_id == mid
            and row["raw_session_id"] == row["source_session_id"]
            and row["raw_role"] == row["source_role"]
            and row["raw_content"] == payload["content"]
            and row["raw_created_at"] == row["source_created_at"]
            and row["raw_peer_id"] == row["source_peer_id"]
            and row["raw_workspace_id"] == row["source_workspace_id"]
        )
    if not valid:
        raise RuntimeError(f"coverage proof mismatch for message {mid}")
    return CoveredMessage(
        message_id=mid,
        session_id=row["source_session_id"],
        role=row["source_role"],
        content=payload["content"],
        chunk_id=row["chunk_id"],
        source_created_at=row["source_created_at"],
        source_peer_id=row["source_peer_id"],
        source_workspace_id=row["source_workspace_id"],
    )


def validate_message_coverage_artifact(
    conn: sqlite3.Connection,
    *,
    message_id: int,
    chunk_id: str,
    coverage_version: str,
) -> CoveredMessage:
    """Validate one portable proof without trusting a live raw row/frontier.

    The proof itself is exact storage: canonical JSONL bytes, immutable source
    metadata, and a matching role/content hash.  This helper deliberately does
    not claim ordered-stream completeness; callers such as the digest reader
    must separately enforce a recognized version and producer frontier.
    """
    current_shape = all((
        _has_columns(
            conn, "message_retention_coverage",
            "source_peer_id", "source_workspace_id",
        ),
        _has_columns(conn, "sessions", "source_workspace_id"),
        _has_columns(conn, "messages", "source_peer_id", "source_workspace_id"),
        _has_columns(conn, "session_peers", "workspace_id", "peer_id"),
    ))
    if current_shape:
        projection = COVERAGE_VALIDATION_COLUMNS
        joins = COVERAGE_VALIDATION_JOINS
    else:
        projection = """
            mc.message_id, mc.source_session_id, mc.source_role,
            NULL AS source_peer_id, NULL AS source_workspace_id,
            mc.source_created_at, mc.message_content_hash, mc.hash_version,
            mc.record_version, mc.coverage_version,
            c.id AS chunk_id, c.session_id AS chunk_session_id,
            c.start_message_id, c.end_message_id, c.text, c.chunk_kind,
            source_session.id AS bound_session_id,
            NULL AS bound_workspace_id, NULL AS registered_peer_role,
            NULL AS member_peer_id,
            raw.id AS raw_message_id, raw.session_id AS raw_session_id,
            raw.role AS raw_role, raw.content AS raw_content,
            raw.created_at AS raw_created_at,
            NULL AS raw_peer_id, NULL AS raw_workspace_id
        """
        joins = """
            JOIN chunks c ON c.id = mc.chunk_id
            LEFT JOIN sessions source_session
              ON source_session.id = mc.source_session_id
            LEFT JOIN messages raw
              ON raw.id = mc.message_id
             AND raw.session_id = mc.source_session_id
        """
    row = conn.execute(
        f"SELECT {projection} FROM message_retention_coverage mc {joins} "
        "WHERE mc.message_id=? AND mc.chunk_id=? AND mc.coverage_version=?",
        (int(message_id), chunk_id, coverage_version),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"coverage proof missing for message {message_id}")
    return validate_message_coverage_row(row)


def covered_messages_after(
    conn: sqlite3.Connection,
    session_id: str,
    message_id: int | None,
    *,
    limit: int = 256,
    roles: frozenset[str] | None = None,
    through_message_id: int | None = None,
) -> list[CoveredMessage]:
    """Load and validate the durable stream after a message boundary.

    This intentionally does not join ``messages``: prompt-version rewinds must
    continue to work after opt-in raw-message retention has removed the source
    rows.  Every byte comes from the protected canonical artifact and is
    checked against the immutable proof metadata before it reaches the LLM.
    """
    if limit <= 0:
        return []
    frontier = conn.execute(
        "SELECT coverage_message_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    # A recognized-looking proof row is not by itself an ordered stream. Only
    # the producer-established contiguous frontier authorizes reads; this also
    # excludes sparse/imported rows that were never walked by the producer.
    if frontier is None or frontier["coverage_message_id"] is None:
        return []
    producer_frontier = int(frontier["coverage_message_id"])
    effective_frontier = (
        min(producer_frontier, int(through_message_id))
        if through_message_id is not None
        else producer_frontier
    )
    if roles is not None and not roles:
        return []
    version_placeholders = ",".join("?" * len(LOSSLESS_READ_VERSIONS))
    role_clause = ""
    role_params: tuple[str, ...] = ()
    if roles is not None:
        ordered_roles = tuple(sorted(roles))
        role_clause = f"AND mc.source_role IN ({','.join('?' * len(ordered_roles))})"
        role_params = ordered_roles
    frontier_clause = ""
    frontier_params: tuple[int, ...] = ()
    frontier_clause = "AND mc.message_id <= ?"
    frontier_params = (effective_frontier,)
    coverage_peer_select = (
        "mc.source_peer_id, mc.source_workspace_id"
        if _has_columns(
            conn, "message_retention_coverage",
            "source_peer_id", "source_workspace_id",
        )
        else "NULL AS source_peer_id, NULL AS source_workspace_id"
    )
    rows = conn.execute(
        f"""
        SELECT mc.message_id, mc.source_session_id, mc.source_role,
               {coverage_peer_select},
               mc.source_created_at,
               mc.message_content_hash, mc.hash_version, mc.record_version,
               mc.coverage_version,
               c.id AS chunk_id, c.session_id AS chunk_session_id,
               c.start_message_id, c.end_message_id, c.text, c.chunk_kind
        FROM message_retention_coverage mc
        JOIN chunks c ON c.id = mc.chunk_id
        WHERE mc.source_session_id = ?
          AND mc.message_id > ?
          AND mc.coverage_version IN ({version_placeholders})
          {role_clause}
          {frontier_clause}
          AND c.chunk_kind = 'coverage'
        ORDER BY mc.message_id,
                 (mc.coverage_version = ?) DESC,
                 mc.created_at DESC
        LIMIT ?
        """,
        (
            session_id,
            int(message_id) if message_id is not None else -1,
            *LOSSLESS_READ_VERSIONS,
            *role_params,
            *frontier_params,
            LOSSLESS_COVERAGE_VERSION,
            int(limit),
        ),
    ).fetchall()

    result: list[CoveredMessage] = []
    seen_message_ids: set[int] = set()
    for row in rows:
        mid = int(row["message_id"])
        if mid in seen_message_ids:
            continue
        seen_message_ids.add(mid)
        proof = validate_message_coverage_artifact(
            conn,
            message_id=mid,
            chunk_id=row["chunk_id"],
            coverage_version=row["coverage_version"],
        )
        if proof.session_id != session_id or (
            roles is not None and proof.role not in roles
        ):
            raise RuntimeError(f"coverage proof mismatch for message {mid}")
        result.append(proof)
    return result
