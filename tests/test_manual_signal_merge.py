"""Canonical merges preserve exact signal identities and bound manual events."""

from __future__ import annotations

from collections import Counter
from itertools import permutations
import sqlite3

import pytest

from hymem.core import db
from hymem.dreaming import bitemporal, canonicalize, evidence


SQL_CLOCK = "2024-06-01 12:00:00"
ISO_CLOCK = "2024-06-01T12:00:00.000Z"
OPEN_AT = "2024-01-01T00:00:00.000Z"
SIGNAL_FIELDS = (
    "signal_kind", "polarity", "evidence_weight", "counts_toward_confidence",
    "details", "created_at",
)
EVENT_FIELDS = ("event_at", "details", "direction", "created_at")


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "manual-merge.sqlite")
    db.initialize(connection)
    # Keep the public API and the SQL column defaults untouched, without
    # making byte-exact clock comparisons depend on crossing a wall second.
    connection.create_function("current_timestamp", 0, lambda: SQL_CLOCK)
    try:
        yield connection
    finally:
        connection.close()


def _seed(conn, name, *, key="decision", kind="manual_retraction", details="close"):
    edge_id = conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical,predicate,"
        "object_canonical,pos_evidence,neg_evidence) VALUES (?,'uses','redis',0,0)",
        (name,),
    ).lastrowid
    bitemporal.record_lifecycle_event(
        conn, edge_id=edge_id, event_key="legacy-state", event_kind="legacy_state",
        direction=1, event_at=OPEN_AT,
    )
    evidence.record_signal(
        conn, edge_id=edge_id, signal_key=key, signal_kind=kind,
        polarity=-1, details=details,
    )
    return edge_id


def _payloads(conn):
    result = []
    for signal in conn.execute("SELECT * FROM kg_evidence_signals ORDER BY id"):
        event = conn.execute(
            "SELECT * FROM kg_edge_lifecycle WHERE edge_id=? AND event_key=? "
            "AND event_kind='manual_retraction'",
            (signal["edge_id"], evidence.manual_retraction_event_key(signal["signal_key"])),
        ).fetchone()
        result.append((
            tuple(signal[field] for field in SIGNAL_FIELDS),
            tuple(event[field] for field in EVENT_FIELDS) if event else None,
        ))
    return Counter(result)


def _assert_bound(conn, expected_count):
    assert conn.execute(
        "SELECT COUNT(*) FROM kg_evidence_signals"
    ).fetchone()[0] == expected_count
    assert conn.execute(
        "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE event_kind='manual_retraction'"
    ).fetchone()[0] == expected_count
    assert not conn.execute(
        "SELECT 1 FROM kg_evidence_signals signal WHERE signal_kind='manual_retraction' "
        "AND NOT EXISTS (SELECT 1 FROM kg_edge_lifecycle event "
        "WHERE event.edge_id=signal.edge_id AND event.event_kind='manual_retraction' "
        "AND event.event_key='manual-retraction:'||signal.signal_key)"
    ).fetchall()
    assert not conn.execute(
        "SELECT 1 FROM kg_edge_lifecycle event WHERE event_kind='manual_retraction' "
        "AND NOT EXISTS (SELECT 1 FROM kg_evidence_signals signal "
        "WHERE signal.edge_id=event.edge_id AND signal.signal_kind='manual_retraction' "
        "AND event.event_key='manual-retraction:'||signal.signal_key)"
    ).fetchall()
    assert evidence.count_mismatches(conn) == []
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()


def _merge(conn, keep, drop):
    with db.transaction(conn):
        canonicalize.merge(conn, keep, drop)


@pytest.mark.parametrize("keep,drop", [("alpha", "beta"), ("beta", "alpha")])
def test_default_manual_pairs_coalesce_with_original_clocks(conn, keep, drop):
    for name in ("alpha", "beta"):
        _seed(conn, name)
    before = _payloads(conn)
    assert len(before) == 1 and list(before.values()) == [2]
    assert next(iter(before))[0][-1] == next(iter(before))[1][-1] == SQL_CLOCK

    _merge(conn, keep, drop)

    _assert_bound(conn, 1)
    assert _payloads(conn) == Counter({payload: 1 for payload in before})
    assert tuple(conn.execute(
        "SELECT pos_evidence,neg_evidence,status,valid_at,invalid_at FROM knowledge_graph"
    ).fetchone()) == (0, 1, "retracted", OPEN_AT, ISO_CLOCK)


@pytest.mark.parametrize("keep,drop", [("alpha", "beta"), ("beta", "alpha")])
@pytest.mark.parametrize("table,field,value", [
    ("kg_evidence_signals", "created_at", ISO_CLOCK),
    ("kg_evidence_signals", "details", "another reason"),
    ("kg_evidence_signals", "evidence_weight", 2),
    ("kg_evidence_signals", "counts_toward_confidence", 0),
    ("kg_edge_lifecycle", "created_at", ISO_CLOCK),
    ("kg_edge_lifecycle", "details", "another event reason"),
    ("kg_edge_lifecycle", "event_at", "2024-06-01T11:00:00.000Z"),
    ("kg_edge_lifecycle", "direction", 1),
])
def test_manual_pairs_keep_distinct_raw_payloads(conn, keep, drop, table, field, value):
    _seed(conn, "alpha")
    beta = _seed(conn, "beta")
    # Exercise stored historical variants independently, including states
    # that the public recorder would reject as new manual intent.
    with db.evidence_history_mutation(conn):
        condition = " AND event_kind='manual_retraction'" if table == "kg_edge_lifecycle" else ""
        conn.execute(f"UPDATE {table} SET {field}=? WHERE edge_id=?{condition}", (value, beta))
    evidence.reconcile_edge_counts(conn, [beta])
    before = _payloads(conn)
    assert len(before) == 2

    _merge(conn, keep, drop)

    _assert_bound(conn, 2)
    assert _payloads(conn) == before


@pytest.mark.parametrize("keep,drop", [("alpha", "beta"), ("beta", "alpha")])
def test_manual_same_millisecond_distinct_keys_remain_bound(conn, keep, drop):
    _seed(conn, "alpha", key="first-decision")
    _seed(conn, "beta", key="second-decision")
    before = _payloads(conn)
    assert list(before.values()) == [2]

    _merge(conn, keep, drop)

    _assert_bound(conn, 2)
    assert _payloads(conn) == before
    assert {row[0] for row in conn.execute(
        "SELECT signal_key FROM kg_evidence_signals"
    )} == {"first-decision", "second-decision"}


@pytest.mark.parametrize("keep,drop", [("alpha", "beta"), ("beta", "alpha")])
@pytest.mark.parametrize("field,value", [
    (None, None), ("signal_key", "different-key"), ("signal_kind", "different-kind"),
    ("details", "different-details"), ("created_at", ISO_CLOCK),
    ("polarity", 1), ("evidence_weight", 2), ("counts_toward_confidence", 0),
])
def test_ordinary_signals_use_exact_keys_and_raw_payloads(conn, keep, drop, field, value):
    _seed(conn, "alpha", kind="test_signal")
    beta = _seed(conn, "beta", kind="test_signal")
    if field is not None:
        with db.evidence_history_mutation(conn):
            conn.execute(f"UPDATE kg_evidence_signals SET {field}=? WHERE edge_id=?", (value, beta))
        evidence.reconcile_edge_counts(conn, [beta])
    before = _payloads(conn)

    _merge(conn, keep, drop)

    assert conn.execute("SELECT COUNT(*) FROM kg_evidence_signals").fetchone()[0] == (
        1 if field is None else 2
    )
    assert _payloads(conn) == (Counter({payload: 1 for payload in before}) if field is None else before)
    assert evidence.count_mismatches(conn) == []


@pytest.mark.parametrize("order", list(permutations(("alpha", "beta", "gamma"))))
@pytest.mark.parametrize("distinct", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_three_way_manual_merge_keeps_complete_pairs(conn, order, distinct, batch):
    edges = {
        name: _seed(conn, name, details=name if distinct else "close")
        for name in ("alpha", "beta", "gamma")
    }
    before = _payloads(conn)
    keep, *drops = order
    if batch:
        evidence.move_edge_provenance(conn, edges[keep], [edges[name] for name in drops])
    else:
        for drop in drops:
            _merge(conn, keep, drop)

    _assert_bound(conn, 3 if distinct else 1)
    assert _payloads(conn) == Counter({payload: 1 for payload in before})


@pytest.mark.parametrize("order", list(permutations(("alpha", "beta", "gamma"))))
def test_same_call_three_way_conflict_coalesces_only_exact_original_pairs(conn, order):
    edges = {
        name: _seed(conn, name, details="first" if name == "alpha" else "second")
        for name in ("alpha", "beta", "gamma")
    }
    before = _payloads(conn)
    keep, *drops = order

    evidence.move_edge_provenance(conn, edges[keep], [edges[name] for name in drops])

    _assert_bound(conn, 2)
    assert _payloads(conn) == Counter({payload: 1 for payload in before})


@pytest.mark.parametrize("outer_transaction", [False, True])
def test_ambiguous_suffix_from_prior_merge_fails_without_losing_pairs(conn, outer_transaction):
    alpha = _seed(conn, "alpha", details="first")
    _seed(conn, "beta", details="second")
    _merge(conn, "alpha", "beta")
    gamma = _seed(conn, "gamma", details="second")
    before = list(conn.iterdump())
    # The prior merge's suffix could equally be a caller-supplied key. Its
    # original key is no longer available in this merge's input snapshots.
    with pytest.raises(ValueError, match="collides with different identity"):
        if outer_transaction:
            _merge(conn, "alpha", "gamma")
        else:
            evidence.move_edge_provenance(conn, alpha, [gamma])

    assert list(conn.iterdump()) == before
    assert not conn.in_transaction
    _assert_bound(conn, 3)


@pytest.mark.parametrize("caller_key_on_survivor", [False, True])
def test_caller_key_equal_to_merge_suffix_is_a_distinct_original_identity(
    conn, caller_key_on_survivor,
):
    alpha = _seed(conn, "alpha", details="first")
    beta = _seed(conn, "beta", details="second")
    # Obtain a real generated key, then restore the unmerged input. The
    # fixture need not duplicate the implementation's hash serialization.
    conn.execute("SAVEPOINT discover_merge_key")
    try:
        evidence.move_edge_provenance(conn, alpha, [beta])
        caller_key = conn.execute(
            "SELECT signal_key FROM kg_evidence_signals WHERE details='second'"
        ).fetchone()[0]
    finally:
        conn.execute("ROLLBACK TO discover_merge_key")
        conn.execute("RELEASE discover_merge_key")

    if caller_key_on_survivor:
        evidence.record_signal(
            conn, edge_id=alpha, signal_key=caller_key,
            signal_kind="manual_retraction", polarity=-1, details="second",
        )
        before = list(conn.iterdump())
        with pytest.raises(ValueError, match="collides with different identity"):
            evidence.move_edge_provenance(conn, alpha, [beta])
        assert list(conn.iterdump()) == before
    else:
        gamma = _seed(conn, "gamma", key=caller_key, details="second")
        before = _payloads(conn)
        evidence.move_edge_provenance(conn, alpha, [beta, gamma])
        assert _payloads(conn) == before
    _assert_bound(conn, 3)


@pytest.mark.parametrize("outer_transaction", [False, True])
def test_pair_move_rolls_back_if_event_move_fails(conn, outer_transaction):
    alpha = _seed(conn, "alpha", details="first")
    beta = _seed(conn, "beta", details="second")
    conn.execute(
        "CREATE TEMP TRIGGER reject_manual_move BEFORE UPDATE ON kg_edge_lifecycle "
        "WHEN new.event_kind='manual_retraction' "
        "BEGIN SELECT RAISE(ABORT, 'injected manual event move failure'); END"
    )
    before = list(conn.iterdump())
    with pytest.raises(sqlite3.IntegrityError, match="injected manual event move failure"):
        if outer_transaction:
            _merge(conn, "alpha", "beta")
        else:
            evidence.move_edge_provenance(conn, alpha, [beta])
    assert list(conn.iterdump()) == before
    assert not conn.in_transaction
    _assert_bound(conn, 2)


def test_manual_merge_replay_and_empty_move_are_noops(conn):
    _seed(conn, "alpha")
    _seed(conn, "beta")
    _merge(conn, "alpha", "beta")
    edge_id = conn.execute("SELECT id FROM knowledge_graph").fetchone()[0]
    before = list(conn.iterdump())

    _merge(conn, "alpha", "beta")
    _merge(conn, "alpha", "alpha")
    evidence.move_edge_provenance(conn, edge_id, [edge_id])
    evidence.move_edge_provenance(conn, edge_id, [])

    assert list(conn.iterdump()) == before
    _assert_bound(conn, 1)
