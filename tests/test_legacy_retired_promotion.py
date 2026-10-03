"""v47 delayed manifest recovery cannot turn retired history into authority.

These self-contained fixtures retain the current surrounding schema but remove
the v47 terminal-loss tail and stamp v46, like test_terminal_chunk_loss's upgrade
fixture. They reconstruct the old data condition (raw messages, no coverage or
manifest) and run unpatched public initialize; they are not an archival v46 DDL
snapshot. Authentic c215f52-package upgrade is checked separately during review.
"""
from pathlib import Path
import sqlite3

import pytest

from hymem.core import db
from hymem.dreaming import evidence


def _fixture(path: Path):
    conn = db.connect(path)
    db.initialize(conn)
    conn.execute("INSERT INTO sessions(id,started_at) VALUES ('legacy','2024-01-01')")
    return conn


def _source(conn, text="Exact legacy source"):
    return int(conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) VALUES ('legacy','user',?,'2024-01-01')",
        (text,),
    ).lastrowid)


def _chunk(conn, chunk_id, message_id, text="Exact legacy source"):
    conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,salience_reason,text,chunk_kind,created_at) "
        "VALUES (?,'legacy',?,?,'legacy',?,'extraction','2024-01-02')",
        (chunk_id, message_id, message_id, f"user: {text}"),
    )


def _legacy_evidence(conn, chunk_id, *, current, polarity=1, edge_id=None):
    if edge_id is None:
        edge_id = int(conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,first_seen,last_seen,last_reinforced) "
            "VALUES (?,'uses','redis','2024-01-01','2024-01-01','2024-01-01')",
            (chunk_id,),
        ).lastrowid)
    semantic_key = evidence._interpretation_key(
        polarity=polarity, evidence_weight=1, weight_source="legacy",
        source_role=None, surface_subject=None, surface_object=None,
        value_text=None, value_numeric=None, value_unit=None, temporal_scope=None,
    )
    with db.evidence_mutation(conn):
        eid = int(conn.execute(
            "INSERT INTO kg_evidence(edge_id,chunk_id,evidence_kind,polarity,evidence_weight,weight_source,"
            "extracted_at,provenance_status,interpretation_key,is_current,revision,superseded_at,superseded_reason) "
            "VALUES (?,?,'extraction',?,1,'legacy','2024-01-02','legacy_unattributed',?,?,1,?,?)",
            (edge_id, chunk_id, polarity, semantic_key, int(current),
             None if current else "2024-01-03", None if current else "historical revision retired"),
        ).lastrowid)
    return eid, edge_id


def _remove_v47_tail(conn):
    for name, sql in conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'").fetchall():
        if "chunk_extraction_terminal_losses" in (sql or ""):
            conn.execute(f'DROP TRIGGER "{name}"')
    conn.execute("DROP TABLE chunk_extraction_terminal_losses")
    conn.execute("UPDATE schema_meta SET value='46' WHERE key='schema_version'")
    assert conn.execute("SELECT COUNT(*) FROM message_retention_coverage").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunk_message_sources").fetchone()[0] == 0


def _evidence(conn, ids):
    return [tuple(conn.execute("SELECT * FROM kg_evidence WHERE id=?", (eid,)).fetchone()) for eid in ids]


def _provenance_state(conn):
    tables = ("kg_evidence", "kg_claim_observations", "kg_edge_lifecycle",
              "kg_claim_extraction_outcomes", "chunk_message_sources", "message_retention_coverage",
              "chunk_extraction_terminal_losses")
    return {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in tables}


def test_public_upgrade_promotes_current_claims_and_preserves_retired_bytes(tmp_path):
    path = tmp_path / "delayed-v46.sqlite"
    conn = _fixture(path)
    retired, current = [], []
    for label, live, polarity in (("active_positive", True, 1), ("active_negative", True, -1),
                                  ("retired_positive", False, 1), ("retired_negative", False, -1)):
        mid = _source(conn)
        _chunk(conn, label, mid)
        eid, _ = _legacy_evidence(conn, label, current=live, polarity=polarity)
        (current if live else retired).append(eid)
    _chunk(conn, "genuinely_lost", 999)
    lost, _ = _legacy_evidence(conn, "genuinely_lost", current=False)
    retired.append(lost)
    before_retired = _evidence(conn, retired)
    _remove_v47_tail(conn)
    conn.close()

    conn = db.connect(path)
    try:
        db.initialize(conn)
        assert db.schema_version(conn) == db.EXPECTED_SCHEMA_VERSION
        assert _evidence(conn, retired) == before_retired
        for eid in current:
            row = conn.execute("SELECT * FROM kg_evidence WHERE id=?", (eid,)).fetchone()
            assert row["provenance_status"] == "canonical" and row["is_current"] == 1
            assert row["source_message_id"] is not None
            assert conn.execute("SELECT COUNT(*) FROM kg_claim_observations WHERE evidence_id=?", (eid,)).fetchone()[0] == 1
            assertions = conn.execute("SELECT COUNT(*) FROM kg_edge_lifecycle WHERE source_evidence_id=?", (eid,)).fetchone()[0]
            assert assertions == (1 if row["polarity"] == 1 else 0)
        for eid in retired:
            assert conn.execute("SELECT COUNT(*) FROM kg_claim_observations WHERE evidence_id=?", (eid,)).fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM kg_edge_lifecycle WHERE source_evidence_id=?", (eid,)).fetchone()[0] == 0
        assert [tuple(row) for row in conn.execute("SELECT chunk_id,reason FROM chunk_extraction_terminal_losses")] == [
            ("genuinely_lost", "source_manifest_unrecoverable"),
        ]
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE source_manifest_version='claim-source-manifest-v1'").fetchone()[0] == 4
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        before = _provenance_state(conn)
        db.initialize(conn)
        assert _provenance_state(conn) == before
        conn.close()
        conn = db.connect(path)
        db.initialize(conn)
        assert _provenance_state(conn) == before
        # The v47 DDL and hooks may have committed before an old launcher
        # persisted its version marker. Replaying that stamp must stay safe.
        conn.execute("UPDATE schema_meta SET value='46' WHERE key='schema_version'")
        db.initialize(conn)
        assert _provenance_state(conn) == before
        # The fix selects eligible rows; it must not loosen the ordinary
        # migration writer's authorization or resurrection/lifecycle guards.
        with db.evidence_mutation(conn):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("UPDATE kg_evidence SET is_current=1,superseded_at=NULL,superseded_reason=NULL WHERE id=?", (retired[0],))
            with pytest.raises(sqlite3.IntegrityError, match="canonical evidence"):
                conn.execute(
                    "INSERT INTO kg_claim_observations(chunk_id,edge_id,source_session_id,source_message_id,"
                    "evidence_kind,polarity,prompt_version,prompt_generation,evidence_id,interpretation_key) "
                    "SELECT ev.chunk_id,ev.edge_id,cms.source_session_id,cms.source_message_id,ev.evidence_kind,"
                    "ev.polarity,'pre-v40',0,ev.id,ev.interpretation_key FROM kg_evidence ev "
                    "JOIN chunk_message_sources cms ON cms.chunk_id=ev.chunk_id WHERE ev.id=?",
                    (retired[0],),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,direction,event_at,source_evidence_id) "
                    "SELECT edge_id,'forged-retired-assertion','claim_assertion',1,'2024-01-01T00:00:00.000Z',id "
                    "FROM kg_evidence WHERE id=?", (retired[0],),
                )
        assert _provenance_state(conn) == before
    finally:
        conn.close()


@pytest.mark.parametrize("current_count", [0, 1, 2])
def test_retired_rows_do_not_supply_or_obscure_current_source_authority(tmp_path, current_count):
    path = tmp_path / "source-ambiguity.sqlite"
    conn = _fixture(path)
    try:
        mid = _source(conn)
        _chunk(conn, "retired", mid)
        retired, edge_id = _legacy_evidence(conn, "retired", current=False)
        current = []
        for number in range(current_count):
            chunk_id = f"current_{number}"
            _chunk(conn, chunk_id, mid)
            eid, _ = _legacy_evidence(conn, chunk_id, current=True, edge_id=edge_id)
            current.append(eid)
        before_retired = _evidence(conn, [retired])
        before_current = _evidence(conn, current)
        _remove_v47_tail(conn)
        db.initialize(conn)
        assert _evidence(conn, [retired]) == before_retired
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE source_manifest_version='claim-source-manifest-v1'").fetchone()[0] == current_count + 1
        assert conn.execute("SELECT COUNT(*) FROM chunk_extraction_terminal_losses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM kg_claim_observations WHERE evidence_id=?", (retired,)).fetchone()[0] == 0
        if current_count == 1:
            assert conn.execute("SELECT provenance_status FROM kg_evidence WHERE id=?", (current[0],)).fetchone()[0] == "canonical"
            assert conn.execute("SELECT COUNT(*) FROM kg_claim_observations").fetchone()[0] == 1
        else:
            assert _evidence(conn, current) == before_current
            assert conn.execute("SELECT COUNT(*) FROM kg_claim_observations").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM kg_edge_lifecycle").fetchone()[0] == 0
    finally:
        conn.close()


def test_retry_after_backfill_failure_reuses_durable_coverage_without_revising_history(tmp_path):
    path = tmp_path / "retry-v46.sqlite"
    conn = _fixture(path)
    try:
        mid = _source(conn)
        _chunk(conn, "current", mid)
        active, _ = _legacy_evidence(conn, "current", current=True)
        _chunk(conn, "retired", mid)
        retired, _ = _legacy_evidence(conn, "retired", current=False)
        before = _evidence(conn, [active, retired])
        _remove_v47_tail(conn)
        conn.execute("CREATE TEMP TRIGGER injected_promotion_failure BEFORE INSERT ON kg_claim_observations "
                     "BEGIN SELECT RAISE(ABORT,'injected promotion failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match="injected promotion failure"):
            db.initialize(conn)
        assert not conn.in_transaction
        assert db.schema_version(conn) == 46
        # v47's preceding coverage transaction is deliberately independently
        # durable; the failed manifest/promotion transaction must be rolled back.
        assert conn.execute("SELECT COUNT(*) FROM message_retention_coverage").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunk_message_sources").fetchone()[0] == 0
        assert _evidence(conn, [active, retired]) == before
        conn.close()  # removes the temporary fault; model a normal restart
        conn = db.connect(path)
        db.initialize(conn)
        assert db.schema_version(conn) == db.EXPECTED_SCHEMA_VERSION
        assert _evidence(conn, [retired]) == before[1:]
        assert conn.execute("SELECT provenance_status FROM kg_evidence WHERE id=?", (active,)).fetchone()[0] == "canonical"
        assert conn.execute("SELECT COUNT(*) FROM kg_claim_observations WHERE evidence_id=?", (active,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM message_retention_coverage").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
