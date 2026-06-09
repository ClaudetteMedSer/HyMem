"""Bi-temporal validity stamping for knowledge_graph edges (schema v15).

Transaction time (first_seen / last_seen) records when HyMem *learned* a fact;
VALID time (valid_at / invalid_at) records when the fact was true *in the
world*. World dates come from the source message's ``created_at`` — the same
host-supplied timestamp the recency-dating retrieval lever stamps onto
message_hits — reached via ``kg_evidence -> chunks -> messages``. Edges with no
message-backed evidence fall back to transaction time so a stamped column is
never left NULL.

Two entry points, both idempotent (they touch only NULL rows, so re-running a
dream cycle leaves existing intervals stable):

  - ``stamp_validity``     opens the interval on newly-minted edges (valid_at).
  - ``stamp_invalidation`` closes it the moment an edge is superseded
                           (invalid_at), called from every status-flip site.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

# World date of an edge's evidence: the source message's created_at, reached
# through the chunk that produced the evidence. ``polarity`` selects positive
# evidence (when the fact became true) vs negative (when it was contradicted).
# The subquery is correlated on knowledge_graph.id, valid inside the UPDATE
# over knowledge_graph below. agg / polarity are fixed internal constants.
_EVIDENCE_DATE = """
    SELECT {agg}(m.created_at)
    FROM kg_evidence ev
    JOIN chunks c ON c.id = ev.chunk_id
    JOIN messages m ON m.id = c.start_message_id
    WHERE ev.edge_id = knowledge_graph.id AND ev.polarity = {polarity}
"""


def stamp_validity(conn: sqlite3.Connection) -> int:
    """Set ``valid_at`` on edges that lack it (minted since the last cycle).

    valid_at = earliest positive-evidence world date, falling back to first_seen
    when no message-backed evidence exists. Write-once: only NULL rows are
    touched, so re-running is a no-op and existing intervals stay stable.
    Returns the number of edges stamped.
    """
    cur = conn.execute(
        f"""
        UPDATE knowledge_graph
        SET valid_at = COALESCE(
            ({_EVIDENCE_DATE.format(agg="MIN", polarity=1)}),
            first_seen)
        WHERE valid_at IS NULL
        """
    )
    return cur.rowcount


def stamp_invalidation(conn: sqlite3.Connection, edge_ids: Iterable[int]) -> None:
    """Close the validity interval for edges being superseded right now.

    invalid_at = newest contradicting (negative) evidence world date — when the
    fact stopped being true — falling back to the flip time when no dated
    negative evidence exists. Idempotent: only edges with a NULL invalid_at are
    stamped, so re-retracting an already-closed edge leaves its date intact.
    """
    ids = list(edge_ids)
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"""
        UPDATE knowledge_graph
        SET invalid_at = COALESCE(
            ({_EVIDENCE_DATE.format(agg="MAX", polarity=-1)}),
            CURRENT_TIMESTAMP)
        WHERE id IN ({placeholders}) AND invalid_at IS NULL
        """,
        ids,
    )
