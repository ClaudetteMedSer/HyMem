"""Build the ``temporal_mentions`` index during the dream cycle.

Why a separate pass (and why per-message, not per-chunk)
--------------------------------------------------------
``mentions.py`` indexes *entity* mentions per chunk; this is its temporal
sibling, but keyed to individual **messages** rather than chunks. The TR
retrieval path needs each dated statement tied to (a) the *event* time of the
turn that made it (``messages.created_at``, now real event time) and (b) the
specific message text as evidence — a chunk can span several turns, so chunk
granularity would blur which turn carried which date. We therefore walk the raw
messages inside a chunk's ``[start_message_id, end_message_id]`` range and parse
each one independently.

Each explicit date written in a message (see ``dates.extract_dates``) becomes a
``temporal_mentions`` row: the normalized ISO date (or NULL when year-less), the
raw matched text, a snippet of surrounding message text for context, and the
message's own ``created_at`` so a year-less mention can still be anchored by the
turn's event time at query time. Inserts are ``INSERT OR IGNORE`` against a
uniqueness constraint so a re-dreamed chunk does not duplicate rows.
"""

from __future__ import annotations

import sqlite3

from hymem.dreaming.dates import extract_dates

# How much message text to keep around a date as context. Enough to show the
# event ("shipped v2 on Feb 15") without storing the whole turn — the raw
# message remains retrievable via message_id if the host wants the full text.
_SURROUNDING_CHARS = 240


def _surrounding(text: str) -> str:
    """A trimmed, single-line snippet of the message for the TR event card."""
    flat = " ".join(text.split())
    if len(flat) <= _SURROUNDING_CHARS:
        return flat
    return flat[:_SURROUNDING_CHARS].rstrip() + "…"


def index_message_temporal_mentions(
    conn: sqlite3.Connection,
    message_id: int,
    session_id: str,
    text: str,
    created_at: str,
) -> int:
    """Parse explicit dates from one message and persist them.

    Returns the number of rows inserted (post ``INSERT OR IGNORE``). Idempotent:
    the ``UNIQUE(message_id, raw_text)`` constraint means re-indexing the same
    message is a no-op rather than a duplicate.
    """
    mentions = extract_dates(text)
    if not mentions:
        return 0
    surrounding = _surrounding(text)
    inserted = 0
    for mention in mentions:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO temporal_mentions(
                message_id, session_id, normalized_date, raw_text,
                surrounding_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                mention.normalized_date,
                mention.raw_text,
                surrounding,
                created_at,
            ),
        )
        inserted += cur.rowcount or 0
    return inserted


def index_chunk_temporal_mentions(
    conn: sqlite3.Connection, chunk_id: str
) -> int:
    """Index every message spanned by ``chunk_id`` for explicit dates.

    Mirrors ``mentions.index_chunk_mentions`` as a dream-cycle entry point, but
    resolves the chunk to its underlying messages first so each date is tied to
    the exact turn (and its event time) that wrote it. Returns the total rows
    inserted. Returns 0 — never raises — if the chunk is unknown or the
    ``temporal_mentions`` table is absent (old DB), so the dream pass degrades
    gracefully rather than aborting.
    """
    row = conn.execute(
        "SELECT session_id, start_message_id, end_message_id "
        "FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return 0

    messages = conn.execute(
        """
        SELECT id, content, created_at
        FROM messages
        WHERE session_id = ? AND id BETWEEN ? AND ? AND role IN ('user', 'assistant')
        ORDER BY id
        """,
        (row["session_id"], row["start_message_id"], row["end_message_id"]),
    ).fetchall()

    inserted = 0
    try:
        for m in messages:
            inserted += index_message_temporal_mentions(
                conn,
                int(m["id"]),
                row["session_id"],
                m["content"],
                m["created_at"] or "",
            )
    except sqlite3.OperationalError:
        # temporal_mentions table absent (pre-v14 DB initialized without the
        # migration having run). Degrade silently — TR simply has no index.
        return 0
    return inserted
