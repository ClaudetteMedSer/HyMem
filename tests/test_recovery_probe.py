"""Stage-0 recovery probe (benchmarks/recovery_probe.py) — Grove E2.

`phase1.py:654-666` resurrects a retracted edge to ``status='active'`` but never
clears ``invalid_at``, and nothing in ``hymem/`` ever clears it on
``knowledge_graph``. So ``status='active' AND invalid_at IS NOT NULL`` is the
population of facts that were retracted and have since been re-asserted.

The naive reading — count those rows — OVERSTATES the impact, because
``_anchor_facts`` (the only query in the codebase that reads ``invalid_at`` on
this table) also requires ``pos_evidence > neg_evidence``, takes only the top
``cap``, and lets profile rows consume part of that cap. Most recovered edges
carry enough negative evidence to fail the margin and would be excluded anyway.
So the probe measures a COUNTERFACTUAL DIFF instead: run the real anchor query
twice, with and without the ``invalid_at IS NULL`` clause, and report how many
facts the clause actually costs the digest.

The test that carries the design is
`test_a_recovered_edge_widens_the_anchor_set`: one edge, active with a stamped
``invalid_at`` and a positive evidence margin, must appear in the without-clause
arm and not in the with-clause arm. `test_a_never_retracted_edge_leaves_the_delta_at_zero`
is its control — an ordinary store must read delta 0 however many active edges
it holds, or the probe is just reporting the size of the graph.
`test_a_recovered_edge_below_the_margin_does_not_widen_it` is the second
control, and the one that keeps the headline number honest: it pins the
difference between "was re-asserted" and "the clause cost the digest a fact".
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.extraction.llm import StubLLMClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from recovery_probe import (  # noqa: E402
    BUCKETS,
    _verdict,
    measure_recovery,
    open_store_readonly,
)


# ── seeding helpers (raw SQL, the tests/test_bitemporal.py idiom) ────────────

def _seed_session(conn, session_id: str = "s1") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, CURRENT_TIMESTAMP)",
        (session_id,),
    )


def _seed_message(conn, msg_id: int, created_at: str, *, session_id: str = "s1") -> None:
    conn.execute(
        "INSERT INTO messages(id, session_id, role, content, created_at) "
        "VALUES (?, ?, 'user', 'x', ?)",
        (msg_id, session_id, created_at),
    )


def _seed_chunk(conn, chunk_id: str, start_msg: int, *, session_id: str = "s1") -> None:
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES (?, ?, ?, ?, 'test', 'text')",
        (chunk_id, session_id, start_msg, start_msg),
    )


def _seed_edge(conn, subject: str, predicate: str, obj: str, *,
               pos: int = 3, neg: int = 0, status: str = "active",
               valid_at: str | None = None, invalid_at: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, first_seen, last_seen, last_reinforced, "
        "valid_at, invalid_at, status) "
        "VALUES (?, ?, ?, ?, ?, '2024-01-01 00:00:00', CURRENT_TIMESTAMP, "
        "CURRENT_TIMESTAMP, ?, ?, ?)",
        (subject, predicate, obj, pos, neg, valid_at, invalid_at, status),
    )
    return cur.lastrowid


def _seed_evidence(conn, edge_id: int, chunk_id: str, polarity: int,
                   extracted_at: str) -> None:
    conn.execute(
        "INSERT INTO kg_evidence(edge_id, chunk_id, polarity, extracted_at) "
        "VALUES (?, ?, ?, ?)",
        (edge_id, chunk_id, polarity, extracted_at),
    )


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    with core_db.transaction(hy.conn):
        _seed_session(hy.conn)
        _seed_message(hy.conn, 1, "2024-01-01 00:00:00")
        _seed_message(hy.conn, 2, "2024-06-01 00:00:00")
        _seed_chunk(hy.conn, "c-neg", 1)
        _seed_chunk(hy.conn, "c-pos", 2)
    yield hy.conn
    hy.close()


def _recovered_edge(conn, subject="app", obj="postgres", *, pos=3, neg=0) -> int:
    """An edge in the defect state: active, margin-positive, invalid_at stamped."""
    return _seed_edge(conn, subject, "uses", obj, pos=pos, neg=neg,
                      status="active", valid_at="2024-01-01 00:00:00",
                      invalid_at="2024-03-01 00:00:00")


# ── the counterfactual diff ─────────────────────────────────────────────────

def test_a_recovered_edge_widens_the_anchor_set(conn):
    """THE detector. One active, margin-positive edge with a stamped invalid_at
    is barred from the digest anchor by a clause that bars nothing else. The
    delta names exactly that, on one store, with no bar to calibrate."""
    with core_db.transaction(conn):
        _recovered_edge(conn)

    report = measure_recovery(conn, cap=20)

    assert report["anchor_delta"] == 1
    assert report["anchor_added"] == ["app uses postgres"]


def test_a_never_retracted_edge_leaves_the_delta_at_zero(conn):
    """The control. Ordinary active edges must not move the delta however many
    there are — otherwise the probe is reporting the size of the graph and would
    read 'defect present' on every store."""
    with core_db.transaction(conn):
        for i in range(5):
            _seed_edge(conn, f"svc{i}", "uses", "postgres", pos=3, neg=0)

    report = measure_recovery(conn, cap=20)

    assert report["anchor_delta"] == 0
    assert report["anchor_added"] == []
    assert report["active_total"] == 5


def test_a_recovered_edge_below_the_margin_does_not_widen_it(conn):
    """The second control, and the reason the headline is a diff and not a count.
    A re-asserted edge that still carries more negative than positive evidence is
    excluded by `pos_evidence > neg_evidence` in BOTH arms, so the invalid_at
    clause costs the digest nothing for it. `recovered` counts it; the delta
    must not."""
    with core_db.transaction(conn):
        _recovered_edge(conn, pos=1, neg=4)

    report = measure_recovery(conn, cap=20)

    assert report["recovered"] == 1
    assert report["anchor_delta"] == 0


def test_the_delta_is_bounded_by_the_remaining_cap(conn):
    """The anchor block is a top-`cap` list, so the clause can only cost as many
    facts as there is room for. A probe that ignored the cap would over-report on
    a large store."""
    with core_db.transaction(conn):
        for i in range(10):
            _recovered_edge(conn, subject=f"app{i}")

    report = measure_recovery(conn, cap=3)

    assert report["recovered"] == 10
    assert report["anchor_delta"] == 3


def test_profile_rows_consume_the_cap_before_edges(conn):
    """`_anchor_facts` renders profile rows FIRST and `cap` bounds the COMBINED
    list, so profile rows shrink the edge budget the diff is taken over. A probe
    that budgeted the full cap to edges would over-report on any store with a
    populated profile — which is every real one."""
    with core_db.transaction(conn):
        for i in range(2):
            conn.execute(
                "INSERT INTO user_profile(slot, value, confidence) VALUES (?, ?, 1.0)",
                ("role" if i == 0 else "employer", f"v{i}"),
            )
        for i in range(4):
            _recovered_edge(conn, subject=f"app{i}")

    report = measure_recovery(conn, cap=3)

    assert report["profile_rows"] == 2
    assert report["edge_budget"] == 1
    assert report["anchor_delta"] == 1


def test_a_retracted_edge_is_not_recovered(conn):
    """`status='retracted'` with invalid_at set is an ordinary tombstone, not a
    recovery. Keying on invalid_at alone would count every tombstone."""
    with core_db.transaction(conn):
        _seed_edge(conn, "app", "uses", "mysql", status="retracted",
                   invalid_at="2024-03-01 00:00:00")

    report = measure_recovery(conn, cap=20)

    assert report["recovered"] == 0
    assert report["retracted_total"] == 1
    assert report["anchor_delta"] == 0


def test_a_derived_edge_is_excluded(conn):
    """Derived edges are wiped and rebuilt every dream (inference.py:32), so they
    can carry neither history nor a recovery."""
    with core_db.transaction(conn):
        conn.execute(
            "UPDATE knowledge_graph SET derived = 1 WHERE id = ?",
            (_recovered_edge(conn),),
        )

    report = measure_recovery(conn, cap=20)

    assert report["recovered"] == 0
    assert report["anchor_delta"] == 0


# ── cause attribution ───────────────────────────────────────────────────────

def test_evidence_backed_requires_negative_then_positive_ordering(conn):
    """The only bucket the verdict may quote as genuine recovery: a polarity=-1
    row, then a polarity=+1 row extracted AFTER the closure. Anything weaker
    cannot distinguish recovery from value oscillation."""
    with core_db.transaction(conn):
        edge_id = _recovered_edge(conn)
        _seed_evidence(conn, edge_id, "c-neg", -1, "2024-03-01 00:00:00")
        _seed_evidence(conn, edge_id, "c-pos", 1, "2024-06-01 00:00:00")

    report = measure_recovery(conn, cap=20)

    assert report["buckets"]["evidence_backed"] == 1
    assert report["buckets"]["unordered"] == 0


def test_positive_evidence_predating_the_closure_is_unordered_not_recovery(conn):
    """Positive evidence older than invalid_at cannot have caused the
    resurrection. Counting it would let a single stale assertion manufacture a
    recovery."""
    with core_db.transaction(conn):
        edge_id = _recovered_edge(conn)
        _seed_evidence(conn, edge_id, "c-neg", -1, "2024-03-01 00:00:00")
        _seed_evidence(conn, edge_id, "c-pos", 1, "2024-02-01 00:00:00")

    report = measure_recovery(conn, cap=20)

    assert report["buckets"]["evidence_backed"] == 0
    assert report["buckets"]["unordered"] == 1


def test_no_negative_evidence_is_bucketed_separately(conn):
    """An edge whose closure has no polarity=-1 row behind it was closed by
    something other than contradicting evidence — a migration-015 backfill, a
    host retract_edge call, or a value_supersession close. It is not evidence of
    the retraction gate being too aggressive, and must not be quoted as one."""
    with core_db.transaction(conn):
        edge_id = _recovered_edge(conn)
        _seed_evidence(conn, edge_id, "c-pos", 1, "2024-06-01 00:00:00")

    report = measure_recovery(conn, cap=20)

    assert report["buckets"]["no_negative_evidence"] == 1
    assert report["buckets"]["evidence_backed"] == 0


def test_value_oscillation_is_attributed_not_counted_as_recovery(conn):
    """`value_supersession` closes the loser at the WINNER's valid_at. When the
    loser later re-wins, phase1 flips it back and it looks identical to a
    recovery. The sibling's valid_at matching this edge's invalid_at is the
    signature that separates them."""
    with core_db.transaction(conn):
        edge_id = _recovered_edge(conn, obj="postgres")
        _seed_evidence(conn, edge_id, "c-neg", -1, "2024-03-01 00:00:00")
        _seed_evidence(conn, edge_id, "c-pos", 1, "2024-06-01 00:00:00")
        # the value that superseded it, opened exactly at the loser's closure
        _seed_edge(conn, "app", "uses", "mysql", valid_at="2024-03-01 00:00:00")

    report = measure_recovery(conn, cap=20)

    assert report["buckets"]["value_oscillation"] == 1
    assert report["buckets"]["evidence_backed"] == 0


def test_unstamped_tombstones_are_reported(conn):
    """`behavioral_dedup.py:260-263` retracts without calling stamp_invalidation,
    so those tombstones carry invalid_at IS NULL and can never appear as a
    recovery. Reporting the count keeps a low headline from reading as safety."""
    with core_db.transaction(conn):
        _seed_edge(conn, "app", "uses", "mysql", status="retracted")

    report = measure_recovery(conn, cap=20)

    assert report["unstamped_tombstones"] == 1


# ── the read-only contract ──────────────────────────────────────────────────

def test_the_probe_writes_nothing(cfg, tmp_path):
    """mode=ro is a sqlite-level guarantee, not a discipline: any stray write in
    the probe raises instead of mutating the production store."""
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    try:
        with core_db.transaction(hy.conn):
            _recovered_edge(hy.conn)
    finally:
        hy.close()

    ro = open_store_readonly(cfg.db_path)
    try:
        report = measure_recovery(ro, cap=20)
        assert report["anchor_delta"] == 1
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("DELETE FROM knowledge_graph")
    finally:
        ro.close()


# ── the pre-registered reading ──────────────────────────────────────────────
# _verdict IS the pre-registration: it is the rule banked before the number
# existed. Pinning it here is what stops the reading drifting once a real store
# produces an inconvenient value.

def _report(**kw) -> dict:
    base = {"anchor_delta": 0, "cap": 20,
            "buckets": dict.fromkeys(BUCKETS, 0)}
    base.update(kw)
    return base


def test_zero_delta_reads_inert_and_closes_stage_1():
    verdict, _ = _verdict(_report(anchor_delta=0))
    assert verdict == "INERT"


def test_a_delta_with_no_evidence_backed_row_is_unattributed_not_sized():
    """The trap this branch exists for: facts are excluded, so the count looks
    like a finding, but nothing shows a closure driven by contradicting
    evidence. Quoting a recovery rate here would report value oscillation and
    migration backfill as 'the retraction gate is too aggressive'."""
    verdict, _ = _verdict(
        _report(anchor_delta=4,
                buckets={**dict.fromkeys(BUCKETS, 0), "value_oscillation": 4}))
    assert verdict == "UNATTRIBUTED"


def test_a_delta_with_evidence_backing_is_sized_and_demands_hand_verification():
    verdict, reason = _verdict(
        _report(anchor_delta=4,
                buckets={**dict.fromkeys(BUCKETS, 0), "evidence_backed": 2}))
    assert verdict == "SIZED"
    assert "hand-verify" in reason
