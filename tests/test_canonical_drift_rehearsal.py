"""Synthetic historical repair rehearsals preserve source/proof ownership."""
from __future__ import annotations

import pytest

from hymem.core import db
from hymem.dreaming import canonicalize
from tests.legacy_canonical import legacy_canonical_rows


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "canonical-rehearsal.sqlite")
    db.initialize(connection)
    try:
        yield connection
    finally:
        connection.close()


def test_repair_refuses_normalized_target_that_is_already_another_identity_alias(conn):
    conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('project_name','established_project')")
    with legacy_canonical_rows(conn):
        conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,pos_evidence) VALUES ('Project_Name','uses','tool',2)")
    before = tuple(conn.iterdump())
    with pytest.raises(ValueError, match="alias"):
        with db.transaction(conn):
            canonicalize.repair_canonical_drift(conn)
    assert tuple(conn.iterdump()) == before


@pytest.mark.parametrize("drop,legacy", [("project_alias", False), ("Project_Name", True)])
def test_direct_merge_target_alias_conflict_is_detected_before_any_mutation(conn, drop, legacy):
    conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('project_name','established_project')")
    with legacy_canonical_rows(conn):
        conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('alternate',?)", (drop,))
        conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,pos_evidence) VALUES (?,'uses','tool',2)", (drop,))
    before = tuple(conn.iterdump())
    with pytest.raises(ValueError, match="alias"):
        canonicalize.merge(conn, keep="project_name", drop=drop, _allow_legacy_drop=legacy)
    assert tuple(conn.iterdump()) == before
    assert not conn.in_transaction
    assert canonicalize.resolve(conn, "project name") == "established_project"


@pytest.mark.parametrize("alias_target", ["project_name", "Project_Name"])
def test_repair_accepts_self_alias_or_alias_to_exact_consumed_identity(conn, alias_target):
    with legacy_canonical_rows(conn):
        conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('project_name',?)", (alias_target,))
        conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,pos_evidence) VALUES ('Project_Name','uses','tool',2)")
    with db.transaction(conn):
        fixes = canonicalize.repair_canonical_drift(conn)
    assert fixes and canonicalize.find_canonical_drift(conn) == []
    assert canonicalize.resolve(conn, "Project Name") == "project_name"
    assert conn.execute("SELECT subject_canonical,pos_evidence FROM knowledge_graph").fetchone()[:] == ("project_name", 2)
    assert conn.execute("SELECT alias,canonical FROM entity_aliases").fetchall()[0][:] == ("project_name", "project_name")
    assert conn.execute("SELECT COUNT(*) FROM kg_evidence_signals").fetchone()[0] == 0
    before = tuple(conn.iterdump())
    with db.transaction(conn):
        assert canonicalize.repair_canonical_drift(conn) == []
    assert tuple(conn.iterdump()) == before


def test_repair_outer_transaction_rolls_back_earlier_success_when_later_identity_conflicts(conn):
    conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('zeta_project','established_project')")
    with legacy_canonical_rows(conn):
        for name in ("Alpha_Project", "Zeta_Project"):
            conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,pos_evidence) VALUES (?,'uses','tool',2)", (name,))
    before = tuple(conn.iterdump())
    seen = []
    conn.set_trace_callback(seen.append)
    try:
        with pytest.raises(ValueError, match="alias"):
            with db.transaction(conn):
                canonicalize.repair_canonical_drift(conn)
    finally:
        conn.set_trace_callback(None)
    # The first identity really was rewritten before the second failed; the
    # caller's whole repair transaction, not one merge, restores every byte.
    assert any("UPDATE knowledge_graph SET subject_canonical = 'alpha_project'" in sql for sql in seen)
    assert tuple(conn.iterdump()) == before
    assert not conn.in_transaction


def test_refused_merge_preserves_caller_owned_transaction_and_prior_work(conn):
    conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('project_name','established_project')")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO sessions(id) VALUES ('caller-work')")
    with pytest.raises(ValueError, match="alias"):
        canonicalize.merge(conn, keep="project_name", drop="other_project")
    assert conn.in_transaction
    assert conn.execute("SELECT id FROM sessions").fetchone()[0] == "caller-work"
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
