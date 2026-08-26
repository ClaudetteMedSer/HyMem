"""Bi-temporal validity interval on knowledge_graph edges (schema v15).

valid_at / invalid_at record VALID time (when a fact was true in the world),
distinct from the transaction-time columns (first_seen / last_seen). World dates
are sourced from the evidence's originating message created_at via
kg_evidence -> chunks -> messages; edges without message-backed evidence fall
back to transaction time. See hymem/dreaming/bitemporal.py.
"""

from __future__ import annotations

from hymem.dreaming import bitemporal


def _seed_session(conn, session_id: str = "s1") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, CURRENT_TIMESTAMP)",
        (session_id,),
    )


def _seed_message(conn, msg_id: int, created_at: str, *, role: str = "user",
                  session_id: str = "s1", content: str = "x") -> None:
    conn.execute(
        "INSERT INTO messages(id, session_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (msg_id, session_id, role, content, created_at),
    )


def _seed_chunk(conn, chunk_id: str, start_msg: int, *, session_id: str = "s1") -> None:
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES (?, ?, ?, ?, 'test', 'text')",
        (chunk_id, session_id, start_msg, start_msg),
    )


def _seed_edge(conn, subject: str, predicate: str, obj: str, *,
               pos: int = 1, neg: int = 0, status: str = "active",
               first_seen: str = "2024-01-01 00:00:00") -> int:
    cur = conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, first_seen, last_seen, last_reinforced, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
        (subject, predicate, obj, pos, neg, first_seen, first_seen, status),
    )
    return cur.lastrowid


def _seed_evidence(conn, edge_id: int, chunk_id: str, polarity: int) -> None:
    conn.execute(
        "INSERT INTO kg_evidence(edge_id, chunk_id, polarity) VALUES (?, ?, ?)",
        (edge_id, chunk_id, polarity),
    )


def _edge(conn, edge_id: int):
    return conn.execute(
        "SELECT valid_at, invalid_at, status FROM knowledge_graph WHERE id = ?",
        (edge_id,),
    ).fetchone()


# --- valid_at: opened from the positive-evidence world date ----------------


def test_stamp_validity_uses_positive_evidence_world_date(hy):
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 10, "2024-03-15 09:00:00")
    _seed_chunk(conn, "c1", 10)
    eid = _seed_edge(conn, "med_flow", "uses", "redis", first_seen="2024-06-01 00:00:00")
    _seed_evidence(conn, eid, "c1", polarity=1)

    n = bitemporal.stamp_validity(conn)

    assert n == 1
    # World date (message created_at), NOT the later transaction-time first_seen.
    assert _edge(conn, eid)["valid_at"] == "2024-03-15 09:00:00"


def test_stamp_validity_takes_earliest_positive_evidence(hy):
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 10, "2024-05-01 00:00:00")
    _seed_message(conn, 11, "2024-02-01 00:00:00")
    _seed_chunk(conn, "c1", 10)
    _seed_chunk(conn, "c2", 11)
    eid = _seed_edge(conn, "med_flow", "uses", "redis")
    _seed_evidence(conn, eid, "c1", polarity=1)
    _seed_evidence(conn, eid, "c2", polarity=1)

    bitemporal.stamp_validity(conn)

    # Earliest of the two positive-evidence dates.
    assert _edge(conn, eid)["valid_at"] == "2024-02-01 00:00:00"


def test_stamp_validity_falls_back_to_first_seen_without_evidence(hy):
    conn = hy.conn
    eid = _seed_edge(conn, "derived_sub", "depends_on", "obj",
                     first_seen="2024-07-04 12:00:00")

    bitemporal.stamp_validity(conn)

    assert _edge(conn, eid)["valid_at"] == "2024-07-04 12:00:00"


def test_stamp_validity_is_write_once_idempotent(hy):
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 10, "2024-03-15 09:00:00")
    _seed_chunk(conn, "c1", 10)
    eid = _seed_edge(conn, "med_flow", "uses", "redis")
    _seed_evidence(conn, eid, "c1", polarity=1)

    assert bitemporal.stamp_validity(conn) == 1
    # Second run touches nothing (only NULL valid_at rows are stamped).
    assert bitemporal.stamp_validity(conn) == 0
    assert _edge(conn, eid)["valid_at"] == "2024-03-15 09:00:00"


# --- invalid_at: closed on supersession ------------------------------------


def test_stamp_invalidation_uses_newest_negative_evidence(hy):
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 20, "2024-08-01 00:00:00")
    _seed_message(conn, 21, "2024-09-15 00:00:00")
    _seed_chunk(conn, "n1", 20)
    _seed_chunk(conn, "n2", 21)
    eid = _seed_edge(conn, "med_flow", "uses", "redis")
    _seed_evidence(conn, eid, "n1", polarity=-1)
    _seed_evidence(conn, eid, "n2", polarity=-1)

    bitemporal.stamp_invalidation(conn, [eid])

    # Newest contradicting-evidence world date = when the fact stopped holding.
    assert _edge(conn, eid)["invalid_at"] == "2024-09-15 00:00:00"


def test_stamp_invalidation_falls_back_to_now_without_dated_evidence(hy):
    conn = hy.conn
    eid = _seed_edge(conn, "med_flow", "uses", "redis")

    bitemporal.stamp_invalidation(conn, [eid])

    assert _edge(conn, eid)["invalid_at"] is not None


def test_stamp_invalidation_is_idempotent(hy):
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 20, "2024-08-01 00:00:00")
    _seed_chunk(conn, "n1", 20)
    eid = _seed_edge(conn, "med_flow", "uses", "redis")
    _seed_evidence(conn, eid, "n1", polarity=-1)

    bitemporal.stamp_invalidation(conn, [eid])
    first = _edge(conn, eid)["invalid_at"]
    # Re-invalidating leaves the original (already-set) date intact.
    bitemporal.stamp_invalidation(conn, [eid])
    assert _edge(conn, eid)["invalid_at"] == first


def test_stamp_invalidation_empty_is_noop(hy):
    bitemporal.stamp_invalidation(hy.conn, [])  # must not raise


# --- invalid_at: re-opened on re-assertion ----------------------------------
#
# A positive polarity mention of a retracted/closed edge flips status back to
# 'active'. The validity interval must reopen too — invalid_at clears,
# mirroring rules.py's re-assert contract (status='active', invalid_at=NULL).
# Without the clear the resurrected edge keeps its close date and stays
# invisible to every reader that filters on `invalid_at IS NULL`
# (state_anchor.py, aggregate.py), in the one state where it should be most
# trusted: asserted, contradicted, then re-asserted.


def test_positive_reassert_clears_invalid_at(hy):
    """Re-asserting a retracted edge reopens the validity interval: status
    flips to active AND the stale invalid_at clears."""
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 10, "2024-03-15 09:00:00")
    _seed_chunk(conn, "c1", 10)
    eid = _seed_edge(conn, "med_flow", "uses", "redis", status="retracted")
    conn.execute(
        "UPDATE knowledge_graph SET invalid_at = '2024-05-01 00:00:00' WHERE id = ?",
        (eid,),
    )

    from hymem.dreaming import phase1
    from hymem.extraction.triples import Triple
    phase1._upsert_triple(conn, "c1", Triple("med_flow", "uses", "redis", 1))

    row = _edge(conn, eid)
    assert row["status"] == "active"
    assert row["invalid_at"] is None


def test_positive_reassert_self_heals_active_with_stale_invalid_at(hy):
    """The already-corrupted state (status='active' BUT invalid_at still set,
    as produced by the never-cleared bug) must heal on the next positive
    mention — the clear must not be gated on status='retracted', or rows that
    died in this state never come back."""
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 10, "2024-03-15 09:00:00")
    _seed_chunk(conn, "c1", 10)
    eid = _seed_edge(conn, "med_flow", "uses", "redis", status="active")
    conn.execute(
        "UPDATE knowledge_graph SET invalid_at = '2024-05-01 00:00:00' WHERE id = ?",
        (eid,),
    )

    from hymem.dreaming import phase1
    from hymem.extraction.triples import Triple
    phase1._upsert_triple(conn, "c1", Triple("med_flow", "uses", "redis", 1))

    row = _edge(conn, eid)
    assert row["status"] == "active"
    assert row["invalid_at"] is None


def test_negative_reassert_does_not_clear_invalid_at(hy):
    """A negative mention on a closed edge must NOT resurrect it or clear the
    close date — only positive re-assertion reopens the interval."""
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 10, "2024-03-15 09:00:00")
    _seed_chunk(conn, "c1", 10)
    eid = _seed_edge(conn, "med_flow", "uses", "redis", status="retracted")
    conn.execute(
        "UPDATE knowledge_graph SET invalid_at = '2024-05-01 00:00:00' WHERE id = ?",
        (eid,),
    )

    from hymem.dreaming import phase1
    from hymem.extraction.triples import Triple
    phase1._upsert_triple(conn, "c1", Triple("med_flow", "uses", "redis", -1))

    row = _edge(conn, eid)
    assert row["status"] == "retracted"
    assert row["invalid_at"] == "2024-05-01 00:00:00"


def test_reinforce_heals_active_with_stale_invalid_at(hy):
    """phase3.reinforce bumps positive evidence on active rows; a row that is
    active-with-stale-invalid_at must be healed there too, not re-bumped into
    guaranteed invisibility. (Box evidence: edge 1872 was re-reinforced at
    18:12:59 while corrupted.)"""
    conn = hy.conn
    _seed_session(conn)
    _seed_message(conn, 10, "2024-03-15 09:00:00")
    _seed_chunk(conn, "c1", 10)
    eid = _seed_edge(conn, "med_flow", "uses", "redis", status="active")
    conn.execute(
        "UPDATE knowledge_graph SET invalid_at = '2024-05-01 00:00:00' WHERE id = ?",
        (eid,),
    )
    conn.execute(
        "INSERT INTO entity_mentions(chunk_id, entity_canonical) VALUES (?, ?), (?, ?)",
        ("c1", "med_flow", "c1", "redis"),
    )

    from hymem.dreaming import phase3
    phase3.reinforce(conn, hy.config)

    row = _edge(conn, eid)
    assert row["invalid_at"] is None


# --- end-to-end through the public retraction API --------------------------


def test_retract_edge_closes_validity_interval(hy):
    conn = hy.conn
    eid = _seed_edge(conn, "med_flow", "depends_on", "redis")

    assert hy.retract_edge("med_flow", "depends_on", "redis") is True

    row = _edge(conn, eid)
    assert row["status"] == "retracted"
    # Explicit host retraction has no dated negative evidence -> flip-time fallback.
    assert row["invalid_at"] is not None


def test_as_of_resolution_distinguishes_superseded_from_active(hy):
    """An interval query returns the fact valid at a past instant even after it
    is superseded — the structural win bi-temporal columns buy over a status
    flip alone."""
    conn = hy.conn
    old = _seed_edge(conn, "med_flow", "runs_on", "python2", status="retracted")
    new = _seed_edge(conn, "med_flow", "runs_on", "python3", status="active")
    conn.execute("UPDATE knowledge_graph SET valid_at = '2023-01-01', "
                 "invalid_at = '2024-06-01' WHERE id = ?", (old,))
    conn.execute("UPDATE knowledge_graph SET valid_at = '2024-06-01' WHERE id = ?", (new,))

    def as_of(date: str) -> set[str]:
        rows = conn.execute(
            "SELECT object_canonical FROM knowledge_graph "
            "WHERE subject_canonical = 'med_flow' AND predicate = 'runs_on' "
            "AND valid_at <= ? AND (invalid_at IS NULL OR invalid_at > ?)",
            (date, date),
        ).fetchall()
        return {r["object_canonical"] for r in rows}

    assert as_of("2023-06-01") == {"python2"}   # before the switch
    assert as_of("2024-09-01") == {"python3"}   # after the switch
