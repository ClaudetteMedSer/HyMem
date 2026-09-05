"""Deterministic valid-time state for knowledge-graph edges.

Canonical v40 evidence carries a normalized UTC ``source_event_at`` plus an
exact source identity. Edge intervals are recomputed from that full ordered
ledger, so arrival order, retry order, and import order cannot change state.
An explicitly unknown legacy timestamp uses the conservative ancient sentinel
``0001-01-01T00:00:00.000Z`` and then session/message/id tie-breakers. New
non-NULL timestamps are parsed strictly and malformed values are rejected.

Legacy unattributed evidence retains the pre-v40 best-effort path through its
chunk/raw message while that row survives; no exact provenance is fabricated.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from hymem.core.time import (
    normalize_iso_timestamp,
    timestamp_at_or_before,
    validate_event_clock,
    validate_timestamp_order,
)


_LEGACY_EVIDENCE_DATE = """
    SELECT {agg}(COALESCE(
        hymem_normalize_iso_timestamp(m.created_at),
        '0001-01-01T00:00:00.000Z'
    ))
    FROM kg_evidence ev
    JOIN chunks c ON c.id = ev.chunk_id
    JOIN messages m ON m.id = c.start_message_id
    WHERE ev.edge_id = knowledge_graph.id
      AND ev.polarity = {polarity}
      AND ev.provenance_status = 'legacy_unattributed'
      AND ev.is_current = 1
"""


def normalized_event_at(
    conn: sqlite3.Connection,
    value: str | None,
    *,
    allow_legacy_unknown: bool = False,
) -> str:
    """Normalize one valid-time coordinate with the shared strict parser.

    ``None`` maps to the ancient sentinel only at an explicitly marked legacy
    boundary. A non-NULL malformed value is never silently converted into
    history; callers importing/backfilling corrupt legacy metadata must use
    :func:`_legacy_sort_event_at` instead of creating a new lifecycle event.
    """
    del conn
    if value is None and allow_legacy_unknown:
        return "0001-01-01T00:00:00.000Z"
    return normalize_iso_timestamp(value, context="event_at")


def _legacy_sort_event_at(value: object) -> str:
    """Best-effort ordering used only while reading old unattributed state."""
    try:
        return normalize_iso_timestamp(value, context="legacy event")
    except ValueError:
        return "0001-01-01T00:00:00.000Z"


def evidence_event_at(conn: sqlite3.Connection, evidence_id: int) -> str:
    row = conn.execute(
        "SELECT provenance_status,source_event_at,extracted_at "
        "FROM kg_evidence WHERE id=?", (int(evidence_id),),
    ).fetchone()
    if row is None:
        raise ValueError("lifecycle dependency evidence is missing")
    if row["provenance_status"] == "canonical":
        return normalized_event_at(conn, row["source_event_at"])
    return _legacy_sort_event_at(row["extracted_at"])


def validate_evidence_clock(conn: sqlite3.Connection, evidence_id: int) -> None:
    """Validate one canonical evidence source against its extraction clock."""
    row = conn.execute(
        "SELECT provenance_status,source_event_at,extracted_at "
        "FROM kg_evidence WHERE id=?",
        (int(evidence_id),),
    ).fetchone()
    if row is None:
        raise ValueError("lifecycle dependency evidence is missing")
    if row["provenance_status"] == "canonical":
        validate_event_clock(
            conn,
            row["source_event_at"],
            row["extracted_at"],
            context="canonical evidence",
        )


def manual_retraction_event_at(
    conn: sqlite3.Connection, edge_id: int, signal_created_at: str | None
) -> str:
    """Return the one canonical host-close coordinate for a signal."""
    signal_at = normalized_event_at(conn, signal_created_at)
    open_coordinates = [
        row["event_at"]
        for row in conn.execute(
            "SELECT event_at,created_at FROM kg_edge_lifecycle "
            "WHERE edge_id=? AND direction=1", (int(edge_id),),
        ).fetchall()
        if normalized_event_at(conn, row["created_at"]) <= signal_at
    ]
    return max([signal_at, *open_coordinates])


def record_lifecycle_event(
    conn: sqlite3.Connection,
    *,
    edge_id: int,
    event_key: str,
    event_kind: str,
    direction: int,
    event_at: str | None,
    source_evidence_id: int | None = None,
    dependency_evidence_ids: Iterable[int] = (),
    details: str | None = None,
    recorded_at: str | None = None,
) -> bool:
    """Append one idempotent transition and converge the materialized edge."""
    if direction not in (-1, 1):
        raise ValueError("lifecycle direction must be -1 or 1")
    normalized = normalized_event_at(
        conn,
        event_at,
        allow_legacy_unknown=(event_kind == "legacy_state"),
    )
    prior_clock = conn.execute(
        "SELECT created_at FROM kg_edge_lifecycle "
        "WHERE edge_id=? AND event_key=?",
        (edge_id, event_key),
    ).fetchone()
    lifecycle_recorded_at = (
        recorded_at
        or (prior_clock["created_at"] if prior_clock is not None else None)
        or conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
    )
    # Every transition must obey the transaction-time boundary, regardless of
    # its more specific causal checks below.  A forged future-dated cause must
    # not fold a future close/open into today's single cached interval.
    validate_event_clock(
        conn,
        normalized,
        lifecycle_recorded_at,
        context="lifecycle event",
    )
    dependency_ids = list(
        dict.fromkeys(int(evidence_id) for evidence_id in dependency_evidence_ids)
    )
    if dependency_ids:
        placeholders = ",".join("?" for _ in dependency_ids)
        found = int(conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE id IN (" + placeholders + ")",
            dependency_ids,
        ).fetchone()[0])
        if found != len(dependency_ids):
            raise ValueError("lifecycle dependency evidence is missing")
    from hymem.dreaming import evidence as evidence_ledger

    if event_kind == "claim_assertion":
        publication_column = (
            "published_at"
            if any(
                column["name"] == "published_at"
                for column in conn.execute(
                    "PRAGMA table_info(kg_evidence)"
                ).fetchall()
            )
            else "NULL AS published_at"
        )
        source_key = conn.execute(
            "SELECT source_session_id,source_message_id,evidence_kind,revision,"
            f"extracted_at,{publication_column} "
            "FROM kg_evidence WHERE id=? AND edge_id=? "
            "AND provenance_status='canonical' AND polarity=1",
            (source_evidence_id, edge_id),
        ).fetchone()
        if source_key is None or event_key != evidence_ledger.claim_assertion_event_key(
            source_key["source_session_id"], source_key["source_message_id"],
            source_key["evidence_kind"], source_key["revision"],
        ):
            raise ValueError("claim lifecycle key does not match its source")
        if details is not None:
            raise ValueError("claim lifecycle details must be empty")
        validate_event_clock(
            conn,
            normalized,
            source_key["extracted_at"],
            context="claim assertion",
        )
        validate_timestamp_order(
            source_key["extracted_at"],
            lifecycle_recorded_at,
            context="claim assertion transaction",
        )
        if source_key["published_at"] is not None:
            validate_timestamp_order(
                lifecycle_recorded_at,
                source_key["published_at"],
                context="claim assertion publication",
            )
    elif event_kind == "phase3_retraction":
        if event_key != evidence_ledger.phase3_retraction_event_key(
            conn, dependency_ids
        ):
            raise ValueError("phase3 lifecycle key does not match its causes")
        causes = conn.execute(
            "SELECT id,edge_id,polarity FROM kg_evidence WHERE id IN ("
            + ",".join("?" for _ in dependency_ids) + ")",
            dependency_ids,
        ).fetchall() if dependency_ids else []
        if (
            not causes
            or any(
                int(row["edge_id"]) != int(edge_id)
                or int(row["polarity"]) != -1
                for row in causes
            )
            or normalized != max(
                evidence_event_at(conn, evidence_id)
                for evidence_id in dependency_ids
            )
            or details != "confidence_or_negative_dominance"
        ):
            raise ValueError("phase3 lifecycle semantics do not match its causes")
        for evidence_id in dependency_ids:
            validate_evidence_clock(conn, evidence_id)
    elif event_kind == "value_supersession":
        if len(dependency_ids) != 1 or event_key != (
            evidence_ledger.value_supersession_event_key(
                conn,
                loser_edge_id=edge_id,
                winner_evidence_id=dependency_ids[0],
                event_at=normalized,
            )
        ):
            raise ValueError("value lifecycle key does not match its cause")
        if (
            normalized != evidence_event_at(conn, dependency_ids[0])
            or details != "newer typed value superseded this edge"
        ):
            raise ValueError("value lifecycle semantics do not match its cause")
        validate_evidence_clock(conn, dependency_ids[0])
    elif event_kind == "manual_retraction":
        prefix = "manual-retraction:"
        signal_key = event_key[len(prefix):] if event_key.startswith(prefix) else None
        signal = conn.execute(
            "SELECT details,created_at FROM kg_evidence_signals WHERE edge_id=? "
            "AND signal_kind='manual_retraction' AND signal_key=? "
            "AND polarity=-1 AND counts_toward_confidence=1",
            (edge_id, signal_key),
        ).fetchone()
        if (
            signal is None
            or signal["details"] != details
            or normalized != manual_retraction_event_at(
                conn, edge_id, signal["created_at"]
            )
        ):
            raise ValueError("manual lifecycle event has no matching signal")
        validate_event_clock(
            conn,
            normalized,
            signal["created_at"],
            context="manual retraction",
        )
        validate_timestamp_order(
            signal["created_at"],
            lifecycle_recorded_at,
            context="manual lifecycle transaction",
        )
    elif event_kind == "legacy_state":
        if event_key not in {
            "legacy-state", "portable-v6-legacy-state",
            "portable-v6-legacy-0-open", "portable-v6-legacy-1-close",
        }:
            raise ValueError("legacy lifecycle key is not recognized")
        validate_event_clock(
            conn,
            normalized,
            lifecycle_recorded_at,
            context="legacy lifecycle event",
        )
    else:
        raise ValueError("unsupported lifecycle event kind")
    if dependency_ids:
        placeholders = ",".join("?" for _ in dependency_ids)
        causes = conn.execute(
            "SELECT provenance_status,extracted_at,published_at "
            "FROM kg_evidence WHERE id IN (" + placeholders + ")",
            dependency_ids,
        ).fetchall()
        for cause in causes:
            authority_at = (
                cause["published_at"]
                if cause["provenance_status"] == "canonical"
                else cause["extracted_at"]
            )
            if authority_at is None:
                raise ValueError("lifecycle dependency is unpublished")
            validate_timestamp_order(
                authority_at,
                lifecycle_recorded_at,
                context="lifecycle dependency transaction",
            )
    existing = conn.execute(
        "SELECT edge_id, event_kind, direction, event_at, source_evidence_id, "
        "dependency_count, details FROM kg_edge_lifecycle "
        "WHERE edge_id = ? AND event_key = ?",
        (edge_id, event_key),
    ).fetchone()
    expected = (
        edge_id, event_kind, direction, normalized, source_evidence_id,
        len(dependency_ids), details,
    )
    if existing is not None:
        if tuple(existing) != expected:
            raise ValueError("lifecycle event key collides with different state")
        persisted_dependencies = [
            int(row["evidence_id"])
            for row in conn.execute(
                "SELECT evidence_id FROM kg_lifecycle_dependencies "
                "WHERE lifecycle_id = (SELECT id FROM kg_edge_lifecycle "
                "WHERE edge_id = ? AND event_key = ?) ORDER BY evidence_id",
                (edge_id, event_key),
            ).fetchall()
        ]
        if persisted_dependencies != sorted(dependency_ids):
            raise ValueError("lifecycle event dependencies collide")
        recompute_edge_interval(conn, edge_id)
        return False
    if source_evidence_id is not None:
        source = conn.execute(
            "SELECT 1 FROM kg_evidence WHERE id=? AND edge_id=? "
            "AND provenance_status='canonical' AND is_current=1 "
            "AND polarity=? AND source_event_at=?",
            (source_evidence_id, edge_id, direction, normalized),
        ).fetchone()
        if source is None:
            raise ValueError("new lifecycle source must be current canonical evidence")
    if dependency_ids:
        placeholders = ",".join("?" for _ in dependency_ids)
        current_count = conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE id IN (" + placeholders
            + ") AND is_current=1",
            dependency_ids,
        ).fetchone()[0]
        if int(current_count) != len(dependency_ids):
            raise ValueError("new lifecycle dependencies must be current evidence")
    from hymem.core.db import evidence_mutation

    # Connections run in autocommit mode. A savepoint is therefore required to
    # keep the parent, every dependency, and the materialized interval one
    # atomic operation; it also nests cleanly inside a caller transaction.
    conn.execute("SAVEPOINT hymem_record_lifecycle_event")
    try:
        with evidence_mutation(conn):
            cur = conn.execute(
                """
                INSERT INTO kg_edge_lifecycle(
                    edge_id, event_key, event_kind, direction, event_at,
                    source_evidence_id, dependency_count, details, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge_id, event_key, event_kind, direction, normalized,
                    source_evidence_id, len(dependency_ids), details,
                    lifecycle_recorded_at,
                ),
            )
            lifecycle_id = int(cur.lastrowid)
            if dependency_ids:
                conn.executemany(
                    "INSERT INTO kg_lifecycle_dependencies(lifecycle_id, evidence_id) "
                    "VALUES (?, ?)",
                    [(lifecycle_id, evidence_id) for evidence_id in dependency_ids],
                )
        recompute_edge_interval(conn, edge_id)
    except BaseException:
        conn.execute("ROLLBACK TO hymem_record_lifecycle_event")
        conn.execute("RELEASE hymem_record_lifecycle_event")
        raise
    else:
        conn.execute("RELEASE hymem_record_lifecycle_event")
    return True


def _ordered_events(
    conn: sqlite3.Connection, edge_id: int
) -> list[tuple[str, str, int]]:
    """Return current lifecycle ``(event_at, event_key, direction)`` ascending.

    Equal instants are ordered by their canonical causal source rather than a
    local row id or lexical event-key accident.  Canonical message order is
    session/message/kind/revision/interpretation; dependency decisions inherit
    the latest cause's key. Legacy causes sort before canonical sources and
    explicit manual events sort after sourced events at the same instant.
    """
    has_publication_clock = any(
        row["name"] == "published_at"
        for row in conn.execute("PRAGMA table_info(kg_evidence)").fetchall()
    )
    if has_publication_clock:
        # Schema 42's public valid-time reader owns the complete publication,
        # lifecycle-transaction, and dependency-causality policy. Reuse it for
        # cache materialization so a corrupt/bypassed row cannot split current
        # SQL state from ``facts_at``. The legacy SQL below remains only for
        # migration hooks that run before the publication column exists.
        from hymem.query.graph_state import _eligible_lifecycle_events

        return [
            (event.event_at, event.event_key, event.direction)
            for event in _eligible_lifecycle_events(conn, edge_id, cutoff=None)
        ]
    source_publication = (
        "hymem_timestamp_at_or_before(ev.published_at, "
        "strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')) = 1"
        if has_publication_clock else "1 = 1"
    )
    cause_publication_failure = (
        "cause.provenance_status = 'canonical' AND "
        "hymem_timestamp_at_or_before(cause.published_at, "
        "strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')) <> 1"
        if has_publication_clock else "0 = 1"
    )
    rows = conn.execute(
        f"""
        SELECT lifecycle.id, lifecycle.event_at, lifecycle.event_key,
               lifecycle.event_kind, lifecycle.direction,
               lifecycle.source_evidence_id, lifecycle.created_at
        FROM kg_edge_lifecycle lifecycle
        LEFT JOIN kg_evidence ev ON ev.id = lifecycle.source_evidence_id
        WHERE lifecycle.edge_id = ?
          AND (
              lifecycle.source_evidence_id IS NULL
              OR (
                  ev.is_current = 1
                  AND (
                      ev.provenance_status <> 'canonical'
                      OR {source_publication}
                  )
              )
          )
          AND (
              lifecycle.event_kind <> 'legacy_state'
              OR NOT EXISTS (
                  SELECT 1 FROM kg_evidence replacement
                  WHERE replacement.edge_id = lifecycle.edge_id
                    AND replacement.provenance_status = 'canonical'
                    AND replacement.is_current = 1
              )
          )
          AND (
              lifecycle.dependency_count = 0
              OR (
                  (SELECT COUNT(*) FROM kg_lifecycle_dependencies dependency
                   WHERE dependency.lifecycle_id = lifecycle.id)
                    = lifecycle.dependency_count
                  AND NOT EXISTS (
                      SELECT 1
                      FROM kg_lifecycle_dependencies dependency
                      LEFT JOIN kg_evidence cause
                        ON cause.id = dependency.evidence_id
                      WHERE dependency.lifecycle_id = lifecycle.id
                        AND (
                            cause.id IS NULL OR cause.is_current <> 1
                            OR (
                                {cause_publication_failure}
                            )
                        )
                  )
              )
          )
        ORDER BY lifecycle.event_at, lifecycle.event_key
        """,
        (edge_id,),
    ).fetchall()

    def evidence_order(evidence_id: int) -> tuple[object, ...]:
        evidence = conn.execute(
            """
            SELECT provenance_status, source_session_id, source_message_id,
                   evidence_kind, revision, interpretation_key, chunk_id
            FROM kg_evidence WHERE id = ?
            """,
            (evidence_id,),
        ).fetchone()
        if evidence is None:
            return (-2, "missing")
        if evidence["provenance_status"] == "canonical":
            return (
                1, evidence["source_session_id"],
                int(evidence["source_message_id"]), evidence["evidence_kind"],
                int(evidence["revision"]), evidence["interpretation_key"],
            )
        return (
            0, evidence["chunk_id"], evidence["evidence_kind"],
            int(evidence["revision"]), evidence["interpretation_key"],
        )

    sortable: list[
        tuple[str, tuple[object, ...], int, str, int]
    ] = []
    present_cutoff = conn.execute(
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')"
    ).fetchone()[0]
    for row in rows:
        try:
            event_at = normalize_iso_timestamp(
                row["event_at"], context="stored lifecycle event"
            )
        except ValueError:
            continue
        if not timestamp_at_or_before(event_at, present_cutoff):
            continue
        try:
            created_at = normalize_iso_timestamp(
                row["created_at"], context="stored lifecycle transaction"
            )
        except ValueError:
            continue
        if not timestamp_at_or_before(created_at, present_cutoff):
            continue
        if row["source_evidence_id"] is not None:
            causal = evidence_order(int(row["source_evidence_id"]))
        elif row["event_kind"] in {"phase3_retraction", "value_supersession"}:
            dependencies = conn.execute(
                "SELECT evidence_id FROM kg_lifecycle_dependencies "
                "WHERE lifecycle_id = ? ORDER BY evidence_id",
                (row["id"],),
            ).fetchall()
            causal = max(
                (evidence_order(int(item["evidence_id"])) for item in dependencies),
                default=(-2, "missing"),
            )
        elif row["event_kind"] == "manual_retraction":
            causal = (2, row["event_key"])
        else:
            causal = (-1, row["event_key"])
        # An assertion precedes a lifecycle decision tied to the exact same
        # source coordinate.  This makes the explicit decision the terminal
        # transition while retaining source order across distinct messages.
        kind_rank = 0 if row["event_kind"] == "claim_assertion" else 1
        sortable.append(
            (
                event_at, causal, kind_rank, row["event_key"],
                int(row["direction"]),
            )
        )
    sortable.sort(key=lambda item: item[:4])
    return [(item[0], item[3], item[4]) for item in sortable]


def recompute_edge_interval(conn: sqlite3.Connection, edge_id: int) -> bool:
    """Rebuild one edge's current interval from canonical ordered events."""
    events = _ordered_events(conn, edge_id)
    if not events:
        return False
    open_start: str | None = None
    last_closed_start: str | None = None
    state_open = False
    invalid_at: str | None = None
    for event_at, _event_key, direction in events:
        if direction == 1:
            if not state_open:
                open_start = event_at
            state_open = True
            invalid_at = None
        else:
            if state_open:
                last_closed_start = open_start
                invalid_at = event_at
            state_open = False
    if state_open:
        conn.execute(
            "UPDATE knowledge_graph SET status = 'active', valid_at = ?, "
            "invalid_at = NULL WHERE id = ? AND derived = 0",
            (open_start, edge_id),
        )
        return True

    edge = conn.execute(
        "SELECT first_seen FROM knowledge_graph WHERE id = ?", (edge_id,)
    ).fetchone()
    if edge is None:
        return False
    valid_at = _legacy_sort_event_at(
        last_closed_start or open_start or edge["first_seen"]
    )
    invalid_at = invalid_at or valid_at
    invalid_at = max(str(valid_at or invalid_at), str(invalid_at))
    conn.execute(
        "UPDATE knowledge_graph SET status = 'retracted', "
        "valid_at = COALESCE(?, valid_at, first_seen), invalid_at = ? "
        "WHERE id = ? AND derived = 0",
        (valid_at, invalid_at, edge_id),
    )
    return True


def stamp_validity(conn: sqlite3.Connection) -> int:
    """Recompute canonical edges and stamp only still-NULL legacy intervals."""
    changed = 0
    canonical_ids = [
        int(row["edge_id"])
        for row in conn.execute(
            "SELECT DISTINCT edge_id FROM kg_evidence "
            "WHERE provenance_status = 'canonical' AND is_current = 1 "
            "ORDER BY edge_id"
        ).fetchall()
    ]
    for edge_id in canonical_ids:
        before = conn.execute(
            "SELECT status, valid_at, invalid_at FROM knowledge_graph WHERE id = ?",
            (edge_id,),
        ).fetchone()
        recompute_edge_interval(conn, edge_id)
        after = conn.execute(
            "SELECT status, valid_at, invalid_at FROM knowledge_graph WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if before is not None and after is not None and tuple(before) != tuple(after):
            changed += 1
    cur = conn.execute(
        f"""
        UPDATE knowledge_graph
        SET valid_at = COALESCE(
            ({_LEGACY_EVIDENCE_DATE.format(agg='MIN', polarity=1)}), first_seen)
        WHERE valid_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM kg_evidence ev
              WHERE ev.edge_id = knowledge_graph.id
                AND ev.provenance_status = 'canonical'
                AND ev.is_current = 1
          )
        """
    )
    return changed + (cur.rowcount or 0)


def stamp_invalidation(conn: sqlite3.Connection, edge_ids: Iterable[int]) -> None:
    """Close retracted edges without letting older claims invert intervals."""
    for edge_id in dict.fromkeys(int(edge_id) for edge_id in edge_ids):
        edge = conn.execute(
            "SELECT valid_at FROM knowledge_graph WHERE id = ?", (edge_id,)
        ).fetchone()
        if edge is None:
            continue
        events = _ordered_events(conn, edge_id)
        if events and events[-1][2] == -1:
            recompute_edge_interval(conn, edge_id)
            continue
        negative = conn.execute(
            "SELECT MAX(hymem_normalize_iso_timestamp(source_event_at)) AS event_at "
            "FROM kg_evidence "
            "WHERE edge_id = ? AND polarity = -1 "
            "AND provenance_status = 'canonical' AND is_current = 1 "
            "AND hymem_normalize_iso_timestamp(source_event_at) IS NOT NULL",
            (edge_id,),
        ).fetchone()["event_at"]
        fallback = conn.execute(
            "SELECT COALESCE((SELECT MAX(COALESCE("
            "hymem_normalize_iso_timestamp(m.created_at),"
            "'0001-01-01T00:00:00.000Z')) FROM kg_evidence ev "
            "JOIN chunks c ON c.id = ev.chunk_id "
            "JOIN messages m ON m.id = c.start_message_id "
            "WHERE ev.edge_id = ? AND ev.polarity = -1 "
            "AND ev.provenance_status = 'legacy_unattributed' "
            "AND ev.is_current = 1), "
            "hymem_normalize_iso_timestamp(CURRENT_TIMESTAMP))",
            (edge_id,),
        ).fetchone()[0]
        candidate = negative or fallback
        edge_valid_at = (
            _legacy_sort_event_at(edge["valid_at"])
            if edge["valid_at"] is not None else None
        )
        if edge_valid_at is not None and str(candidate) < edge_valid_at:
            candidate = edge_valid_at
        conn.execute(
            "UPDATE knowledge_graph SET invalid_at = COALESCE(invalid_at, ?) "
            "WHERE id = ?",
            (candidate, edge_id),
        )
