"""Independent release-lineage controls for the R7 v63 -> v64 replay fix.

Run against the rebased R7 candidate, not the stale pre-R7 workspace.
All model responses below are invented and supplied in process.
"""
import sqlite3
from importlib.resources import files

import pytest

from hymem.core import db
from hymem.dreaming import summary_recovery
from tests.test_summary_recovery_v63 import Replies, _seed


def _summary_rows(conn):
    return {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("sessions", "digest_staging", "summary_recovery")
    }


@pytest.fixture
def source63(tmp_path):
    conn = db.connect(tmp_path / "r7-upgrade.sqlite")
    db.initialize(conn)
    assert db.schema_version(conn) == 64
    _seed(conn, long=True)
    report = summary_recovery.run_summary_recovery(
        conn, Replies(), max_calls=1, max_chars=300,
    )
    assert report["advanced"] == 1 and report["published"] == 0
    assert conn.execute("SELECT COUNT(*) FROM summary_recovery").fetchone()[0] == 1
    # Undo only the new additive migration; keep R7's real summary state and
    # table constraints. The separate retained-store rehearsal covers genuine
    # historical DDL and nonempty historical claim outcomes.
    conn.execute("DROP TRIGGER kg_claim_extraction_outcomes_insert_guard")
    conn.execute("DROP TRIGGER kg_claim_extraction_outcomes_update_guard")
    conn.execute("ALTER TABLE kg_claim_extraction_outcomes DROP COLUMN local_replay_proof")
    conn.execute("UPDATE schema_meta SET value='63' WHERE key='schema_version'")
    try:
        yield conn
    finally:
        conn.close()


def test_migration_numbers_extend_existing_summary_history():
    entries = [entry.name for entry in files("hymem.core.migrations").iterdir()
               if entry.name.endswith(".sql")]
    numbers = [int(name.split("_", 1)[0]) for name in entries]
    assert len(numbers) == len(set(numbers))
    assert "062_independent_summary_frontier.sql" in entries
    assert "063_private_summary_recovery.sql" in entries
    assert "064_local_claim_replay_proof.sql" in entries
    assert "062_local_claim_replay_proof.sql" not in entries
    assert max(numbers) == db.EXPECTED_SCHEMA_VERSION == 64


def test_upgrade_preserves_nonempty_private_summary_walk_and_guards(source63):
    conn = source63
    before = _summary_rows(conn)
    guards = list(conn.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
        "AND name IN ('summary_state_workspace_guard','summary_recovery_workspace_guard') "
        "ORDER BY name"
    ))
    assert len(guards) == 2
    db.initialize(conn)
    assert db.schema_version(conn) == 64
    assert _summary_rows(conn) == before
    assert list(conn.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
        "AND name IN ('summary_state_workspace_guard','summary_recovery_workspace_guard') "
        "ORDER BY name"
    )) == guards
    db.initialize(conn)
    assert _summary_rows(conn) == before
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE sessions SET source_workspace_id='other-workspace' WHERE id='x'")
    assert _summary_rows(conn) == before


def test_rejected_v64_migration_does_not_stamp_or_partially_add_column(source63, monkeypatch):
    conn = source63
    before = _summary_rows(conn)

    def reject_shape(_conn):
        raise RuntimeError("synthetic proof schema rejection")

    monkeypatch.setattr(db, "_validate_v64_local_claim_replay_shape", reject_shape)
    with pytest.raises(RuntimeError, match="synthetic proof schema rejection"):
        db._run_migrations(conn)
    assert db.schema_version(conn) == 63
    assert "local_replay_proof" not in {
        row["name"] for row in conn.execute("PRAGMA table_info(kg_claim_extraction_outcomes)")
    }
    assert _summary_rows(conn) == before
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
