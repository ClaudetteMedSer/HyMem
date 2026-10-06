"""The local replay proof migration leaves R7 summary recovery state intact."""

from hymem.core import db


def test_v63_summary_recovery_row_survives_v64_upgrade(tmp_path):
    path = tmp_path / "r7-v63.sqlite"
    conn = db.connect(path)
    db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('s')")
    conn.execute(
        "INSERT INTO summary_recovery("
        "session_id,config_version,walk_id,target_generation,"
        "target_message_id,base_sha256,target_source_sha256,draft,"
        "source_sha256,attempt_limit,state_sha256) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "s", "r7-test-v1", "a" * 32, "generation", 1,
            "b" * 64, "c" * 64, "private draft", "d" * 64, 2,
            "e" * 64,
        ),
    )
    before = tuple(conn.execute("SELECT * FROM summary_recovery").fetchone())
    summary_markers = [tuple(row) for row in conn.execute(
        "SELECT key,value FROM schema_meta WHERE key IN "
        "('summary_frontier_schema','summary_recovery_schema') ORDER BY key"
    )]
    conn.execute("DROP TRIGGER kg_claim_extraction_outcomes_insert_guard")
    conn.execute("DROP TRIGGER kg_claim_extraction_outcomes_update_guard")
    conn.execute(
        "ALTER TABLE kg_claim_extraction_outcomes DROP COLUMN local_replay_proof"
    )
    conn.execute("UPDATE schema_meta SET value='63' WHERE key='schema_version'")
    conn.close()

    conn = db.connect(path)
    try:
        db.initialize(conn)
        assert db.schema_version(conn) == 64
        assert tuple(conn.execute("SELECT * FROM summary_recovery").fetchone()) == before
        assert [tuple(row) for row in conn.execute(
            "SELECT key,value FROM schema_meta WHERE key IN "
            "('summary_frontier_schema','summary_recovery_schema') ORDER BY key"
        )] == summary_markers
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
