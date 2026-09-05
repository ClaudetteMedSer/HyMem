"""Lossless per-message coverage ledger used by raw-message retention.

Coverage is intentionally explicit. A summary, digest watermark, episode, or
graph assertion is a derived and potentially lossy representation; none of
those may authorize deletion of the source message. The v38 lossless stream
calls :func:`record_message_coverage` after writing one canonical JSONL artifact
per source message; other producers may still record independent exact proofs
under their own version without claiming ordered digest completeness.
"""

from __future__ import annotations

import hashlib
import sqlite3

from hymem.core.message_records import (
    MESSAGE_CONTENT_HASH_VERSION,
    MESSAGE_RECORD_VERSION,
    canonical_message_record,
    chunk_contains_message_record,
    encode_message_record,
    message_content_hash,
)

# v37's ledger accepts independent, caller-defined exact proofs.  Only these
# explicitly reviewed producer versions additionally promise a complete,
# ordered stream and may therefore drive the v38 digest cursor.
LOSSLESS_COVERAGE_VERSION = "dream-lossless-message-v1"
LOSSLESS_READ_VERSIONS = (LOSSLESS_COVERAGE_VERSION,)


def _has_columns(
    conn: sqlite3.Connection,
    table: str,
    *columns: str,
) -> bool:
    """Return whether a migration-era table already has ``columns``.

    The v39 data hook imports this module before v43 has added external peer
    provenance.  Keeping the old read/write shape available here preserves
    historical migration ordering; current databases always take the richer
    branch.
    """
    present = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    return set(columns).issubset(present)


def coverage_chunk_id(session_id: str, message_id: int) -> str:
    """Deterministic namespace reserved for the ordered lossless producer."""
    digest = hashlib.sha256(
        f"{LOSSLESS_COVERAGE_VERSION}\0{session_id}\0{int(message_id)}".encode(
            "utf-8"
        )
    ).hexdigest()
    return f"msgcov_{digest}"


def record_message_coverage(
    conn: sqlite3.Connection,
    *,
    message_id: int,
    chunk_id: str,
    coverage_version: str,
) -> None:
    """Record that ``chunk_id`` losslessly covers ``message_id``.

    The API validates the durable artifact rather than trusting a caller-supplied
    hash. Both records must belong to the same session, the message must fall in
    the chunk's declared interval, and its canonical JSONL record must occur as
    one exact line. Existing prose salience chunks therefore cannot accidentally
    claim retention coverage, nor can text truncated by their character cap.
    """
    if not coverage_version or not coverage_version.strip():
        raise ValueError("coverage_version must be non-empty")

    message_has_peer = _has_columns(
        conn, "messages", "source_peer_id", "source_workspace_id"
    )
    coverage_has_peer = _has_columns(
        conn,
        "message_retention_coverage",
        "source_peer_id",
        "source_workspace_id",
    )
    message_peer_select = (
        "m.source_peer_id, m.source_workspace_id"
        if message_has_peer
        else "NULL AS source_peer_id, NULL AS source_workspace_id"
    )
    row = conn.execute(
        f"""
        SELECT m.id, m.session_id, m.role, m.content, m.created_at,
               {message_peer_select},
               c.session_id AS chunk_session_id,
               c.start_message_id, c.end_message_id, c.text AS chunk_text,
               c.chunk_kind
        FROM messages m
        JOIN chunks c ON c.id = ?
        WHERE m.id = ?
        """,
        (chunk_id, int(message_id)),
    ).fetchone()
    if row is None:
        raise ValueError("message and backing chunk must both exist")
    if row["session_id"] != row["chunk_session_id"]:
        raise ValueError("message and backing chunk must belong to the same session")
    if not (
        int(row["start_message_id"])
        <= int(message_id)
        <= int(row["end_message_id"])
    ):
        raise ValueError("message id is outside the backing chunk's declared range")
    record, content_hash, hash_version, record_version = canonical_message_record(
        message_id=message_id,
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        source_created_at=row["created_at"],
        source_peer_id=row["source_peer_id"],
        source_workspace_id=row["source_workspace_id"],
    )
    if not chunk_contains_message_record(chunk_text=row["chunk_text"], record=record):
        raise ValueError("backing chunk does not contain the canonical message record")

    if coverage_version.strip() in LOSSLESS_READ_VERSIONS:
        if (
            chunk_id != coverage_chunk_id(row["session_id"], int(message_id))
            or int(row["start_message_id"]) != int(message_id)
            or int(row["end_message_id"]) != int(message_id)
            or row["chunk_kind"] != "coverage"
            or row["chunk_text"] != record
        ):
            raise ValueError(
                "reserved ordered coverage requires its exact producer artifact"
            )
        # Ordered-stream records are immutable once published.  Retrying the
        # same producer write is an idempotent no-op, followed by an exact
        # metadata check below; it must not exercise the generic UPDATE path.
        if coverage_has_peer:
            conn.execute(
                """
            INSERT OR IGNORE INTO message_retention_coverage(
                message_id, source_session_id, source_role,
                source_peer_id, source_workspace_id, source_created_at,
                chunk_id, message_content_hash, hash_version, record_version,
                coverage_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(message_id),
                    row["session_id"],
                    row["role"],
                    row["source_peer_id"],
                    row["source_workspace_id"],
                    row["created_at"],
                    chunk_id,
                    content_hash,
                    hash_version,
                    record_version,
                    coverage_version.strip(),
                ),
            )
            stored_sql = """
            SELECT source_session_id, source_role, source_peer_id,
                   source_workspace_id, source_created_at,
                   message_content_hash, hash_version, record_version
            FROM message_retention_coverage
            WHERE message_id = ? AND chunk_id = ? AND coverage_version = ?
            """
            expected = (
                row["session_id"],
                row["role"],
                row["source_peer_id"],
                row["source_workspace_id"],
                row["created_at"],
                content_hash,
                hash_version,
                record_version,
            )
        else:
            conn.execute(
                """
                INSERT OR IGNORE INTO message_retention_coverage(
                    message_id, source_session_id, source_role,
                    source_created_at, chunk_id, message_content_hash,
                    hash_version, record_version, coverage_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(message_id), row["session_id"], row["role"],
                    row["created_at"], chunk_id,
                    content_hash,
                    hash_version, record_version,
                    coverage_version.strip(),
                ),
            )
            stored_sql = """
                SELECT source_session_id, source_role, source_created_at,
                       message_content_hash, hash_version, record_version
                FROM message_retention_coverage
                WHERE message_id = ? AND chunk_id = ? AND coverage_version = ?
            """
            expected = (
                row["session_id"], row["role"], row["created_at"],
                content_hash,
                hash_version, record_version,
            )
        stored = conn.execute(
            stored_sql,
            (int(message_id), chunk_id, coverage_version.strip()),
        ).fetchone()
        if stored is None or tuple(stored) != expected:
            raise RuntimeError("immutable ordered coverage metadata mismatch")
        return

    if coverage_has_peer:
        conn.execute(
            """
        INSERT INTO message_retention_coverage(
            message_id, source_session_id, source_role,
            source_peer_id, source_workspace_id, source_created_at,
            chunk_id, message_content_hash, hash_version, record_version,
            coverage_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id, chunk_id, coverage_version) DO UPDATE SET
            source_session_id = excluded.source_session_id,
            source_role = excluded.source_role,
            source_peer_id = excluded.source_peer_id,
            source_workspace_id = excluded.source_workspace_id,
            source_created_at = excluded.source_created_at,
            message_content_hash = excluded.message_content_hash,
            hash_version = excluded.hash_version,
            record_version = excluded.record_version,
            created_at = CURRENT_TIMESTAMP
            """,
            (
                int(message_id), row["session_id"], row["role"],
                row["source_peer_id"], row["source_workspace_id"],
                row["created_at"], chunk_id,
                content_hash,
                hash_version, record_version,
                coverage_version.strip(),
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO message_retention_coverage(
                message_id, source_session_id, source_role,
                source_created_at, chunk_id, message_content_hash,
                hash_version, record_version, coverage_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, chunk_id, coverage_version) DO UPDATE SET
                source_session_id = excluded.source_session_id,
                source_role = excluded.source_role,
                source_created_at = excluded.source_created_at,
                message_content_hash = excluded.message_content_hash,
                hash_version = excluded.hash_version,
                record_version = excluded.record_version,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                int(message_id), row["session_id"], row["role"],
                row["created_at"], chunk_id,
                content_hash,
                hash_version, record_version,
                coverage_version.strip(),
            ),
        )


def release_message_coverage(
    conn: sqlite3.Connection,
    *,
    message_id: int,
    chunk_id: str,
    coverage_version: str,
) -> None:
    """Release durable coverage only while the matching raw source still exists.

    ``message_retention_coverage.chunk_id`` is ``ON DELETE RESTRICT`` so neither
    automated chunk retention nor a cascading session delete can remove the
    last durable copy. Callers that intentionally replace/rebuild an artifact
    must first restore the exact raw message, then use this explicit lifecycle.
    Recognized ordered-digest records are stronger: they cannot be released at
    all, because doing so behind a persisted frontier would create a hole and
    force every hot-path append to rescan the session's history.
    """
    if coverage_version in LOSSLESS_READ_VERSIONS:
        raise RuntimeError(
            "ordered digest coverage is immutable; retain the artifact or "
            "create a new explicitly versioned stream"
        )

    coverage_has_peer = _has_columns(
        conn, "message_retention_coverage",
        "source_peer_id", "source_workspace_id",
    )
    message_has_peer = _has_columns(
        conn, "messages", "source_peer_id", "source_workspace_id"
    )
    coverage_peer_select = (
        "mc.source_peer_id, mc.source_workspace_id"
        if coverage_has_peer
        else "NULL AS source_peer_id, NULL AS source_workspace_id"
    )
    message_peer_select = (
        "m.source_peer_id AS message_peer_id, "
        "m.source_workspace_id AS message_workspace_id"
        if message_has_peer
        else "NULL AS message_peer_id, NULL AS message_workspace_id"
    )
    row = conn.execute(
        f"""
        SELECT mc.source_session_id, mc.source_role, mc.source_peer_id,
               mc.source_workspace_id, mc.source_created_at,
               mc.message_content_hash, mc.hash_version, mc.record_version,
               m.session_id, m.role, {message_peer_select},
               m.content, m.created_at
        FROM message_retention_coverage mc
        LEFT JOIN messages m ON m.id = mc.message_id
        WHERE mc.message_id = ? AND mc.chunk_id = ? AND mc.coverage_version = ?
        """.replace(
            "mc.source_peer_id,\n               mc.source_workspace_id",
            coverage_peer_select,
        ),
        (int(message_id), chunk_id, coverage_version),
    ).fetchone()
    if row is None:
        return
    source_present = (
        row["content"] is not None
        and row["session_id"] == row["source_session_id"]
        and row["role"] == row["source_role"]
        and row["message_peer_id"] == row["source_peer_id"]
        and row["message_workspace_id"] == row["source_workspace_id"]
        and row["created_at"] == row["source_created_at"]
    )
    if source_present:
        try:
            _, expected_hash, expected_hash_version, expected_record_version = (
                canonical_message_record(
                message_id=int(message_id),
                session_id=row["source_session_id"],
                role=row["source_role"],
                content=row["content"],
                source_created_at=row["source_created_at"],
                source_peer_id=row["source_peer_id"],
                source_workspace_id=row["source_workspace_id"],
                )
            )
        except (TypeError, ValueError):
            source_present = False
        else:
            source_present = bool(
                row["hash_version"] == expected_hash_version
                and row["message_content_hash"] == expected_hash
                and row["record_version"] == expected_record_version
            )
    if not source_present:
        raise RuntimeError("cannot release coverage while its raw source is absent")
    conn.execute(
        """
        DELETE FROM message_retention_coverage
        WHERE message_id = ? AND chunk_id = ? AND coverage_version = ?
        """,
        (int(message_id), chunk_id, coverage_version),
    )
