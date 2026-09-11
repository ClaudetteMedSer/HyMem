"""Doctor must not equate a successful schema open with valid source proofs."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from hymem import doctor
from hymem.bootstrap import EnvConfig
from hymem.config import HyMemConfig
from hymem.core import db
from hymem.coverage_health import scan_coverage_health
from hymem.dreaming import lossless
from hymem.dreaming.message_coverage import record_message_coverage


@pytest.fixture
def store(tmp_path):
    path = HyMemConfig(root=tmp_path).db_path
    conn = db.connect(path)
    db.initialize(conn)
    yield conn, path
    conn.close()


def _cfg(path):
    return EnvConfig(
        root=path.parent, llm_api_key=None, llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-v4-flash", embedding_api_key=None,
        embedding_base_url="local://feature-hash", embedding_model="health-test",
        embedding_dim=3, embedding_backend="local_feature_hash",
        embedding_fallback_reason=None, aggregation_nodes_enabled=False,
        aggregation_digest_enabled=False,
    )


def _session(conn, *, external=False, session="private-session"):
    conn.execute("INSERT INTO sessions(id,source_workspace_id) VALUES (?,?)",
                 (session, "private-workspace" if external else None))
    if external:
        _membership(conn, session)
    return session


def _membership(conn, session):
    conn.execute("INSERT OR IGNORE INTO peers(id,workspace_id,role) "
                 "VALUES ('private-peer','private-workspace','user')")
    conn.execute("INSERT INTO session_peers(session_id,workspace_id,peer_id) "
                 "VALUES (?,'private-workspace','private-peer')", (session,))


def _message(conn, session, *, external=False, materialize=True):
    mid = conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at,source_peer_id,source_workspace_id) "
        "VALUES (?,'user','private-source-secret','2026-09-10T10:00:00.000Z',?,?)",
        (session, "private-peer" if external else None, "private-workspace" if external else None),
    ).lastrowid
    if materialize:
        lossless.materialize_message_coverage(conn, session)
    return mid


def _drop_guards(conn, table):
    # Unsupported corruption is deliberate and confined to synthetic fixtures.
    rows = conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?",
                        (table,)).fetchall()
    for row in rows:
        conn.execute('DROP TRIGGER "' + row["name"].replace('"', '""') + '"')
    return [row["sql"] for row in rows]


def _logical(conn):
    return list(conn.iterdump())


def _files(path):
    return {str(item): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in (path, Path(str(path) + "-wal")) if item.exists()}


@pytest.mark.parametrize("external,pruned", [(False, False), (True, False), (False, True), (True, True)])
def test_healthy_native_external_and_pruned_sources_are_read_only(store, external, pruned, monkeypatch):
    conn, path = store
    session = _session(conn, external=external)
    mid = _message(conn, session, external=external)
    if pruned:
        conn.execute("DELETE FROM messages WHERE id=?", (mid,))
    before, files = _logical(conn), _files(path)
    monkeypatch.setattr(db, "initialize", lambda *_: pytest.fail("audit attempted initialization"))
    monkeypatch.setattr(db, "connect", lambda *_: pytest.fail("audit used writable core connector"))
    report = scan_coverage_health(path)
    assert report.status == "valid" and report.complete
    assert report.checked == report.total == 1
    assert report.valid_raw_pruned == int(pruned)
    assert report.valid_raw_present == int(not pruned)
    assert _logical(conn) == before and _files(path) == files
    assert "private" not in json.dumps(asdict(report))


def test_empty_store_is_valid_without_provider_calls(store, monkeypatch):
    conn, path = store
    import socket
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: pytest.fail("audit attempted network"))
    assert scan_coverage_health(path).status == "valid"
    assert doctor._check_lossless_coverage(_cfg(path)).status == doctor.OK


def _rollback_incident(conn):
    session = _session(conn)
    _message(conn, session)
    trigger = conn.execute("SELECT sql FROM sqlite_master WHERE name='session_workspace_binding_guard'").fetchone()[0]
    conn.execute("DROP TRIGGER session_workspace_binding_guard")
    conn.execute("UPDATE sessions SET source_workspace_id='private-workspace' WHERE id=?", (session,))
    _membership(conn, session)
    _message(conn, session, external=True)
    _message(conn, session, external=True)
    conn.execute("UPDATE sessions SET source_workspace_id=NULL WHERE id=?", (session,))
    conn.execute(trigger)


def test_manual_bind_then_rollback_is_not_proved_healthy_by_schema_open(store):
    conn, path = store
    _rollback_incident(conn)
    cfg = _cfg(path)
    # Before this fix doctor relied on these checks: the uncited upload proofs
    # are invalid even though initialization, FKs and empty-vector health pass.
    assert not any(item.status == doctor.FAIL for item in doctor._check_schema_and_dim(cfg, None, None))
    assert list(conn.execute("PRAGMA foreign_key_check")) == []
    before = _logical(conn)
    report = scan_coverage_health(path)
    assert report.complete and report.status == "invalid"
    assert report.total == report.checked == 3 and report.invalid == 2
    assert report.valid_raw_present == 1 and report.recorded_failure_sessions == 0
    result = doctor._check_lossless_coverage(cfg)
    assert result.status == doctor.FAIL and "invalid=2" in result.detail
    assert "private" not in result.render()
    assert _logical(conn) == before


def test_doctor_cli_now_fails_for_uncited_invalid_coverage(store, monkeypatch, capsys):
    conn, path = store
    _rollback_incident(conn)
    cfg = _cfg(path)
    monkeypatch.setattr(doctor, "resolve_env", lambda: cfg)
    for name in ("_check_root", "_check_llm", "_check_canonical_drift"):
        monkeypatch.setattr(doctor, name, lambda *_: doctor._Result(doctor.OK, "offline", "ok"))
    monkeypatch.setattr(doctor, "_check_sqlite_vec", lambda: doctor._Result(doctor.OK, "offline", "ok"))
    monkeypatch.setattr(doctor, "_check_embedding", lambda _: (doctor._Result(doctor.OK, "offline", "ok"), None, None))
    assert doctor.run_doctor() == 1
    output = capsys.readouterr().out
    assert "[FAIL] lossless coverage integrity" in output and "1 failure(s)" in output
    assert "private-source-secret" not in output and "private-session" not in output


@pytest.mark.parametrize("corruption", ["bytes", "hash", "hash_version", "record_version", "role",
                                     "ownership", "member", "peer_role", "raw_session"])
def test_canonical_validator_detects_source_corruption(store, corruption):
    conn, path = store
    session = _session(conn, external=True)
    _message(conn, session, external=True)
    if corruption == "bytes":
        _drop_guards(conn, "chunks")
        conn.execute("UPDATE chunks SET text='private-corrupt-json'")
    elif corruption == "hash":
        _drop_guards(conn, "message_retention_coverage")
        conn.execute("UPDATE message_retention_coverage SET message_content_hash='private-bad-hash'")
    elif corruption in {"hash_version", "record_version"}:
        _drop_guards(conn, "message_retention_coverage")
        conn.execute(f"UPDATE message_retention_coverage SET {corruption}='private-unknown-version'")
    elif corruption == "role":
        _drop_guards(conn, "message_retention_coverage")
        conn.execute("UPDATE message_retention_coverage SET source_role='assistant'")
    elif corruption == "ownership":
        conn.execute("DROP TRIGGER session_workspace_binding_guard")
        conn.execute("UPDATE sessions SET source_workspace_id=NULL")
    elif corruption == "member":
        _drop_guards(conn, "session_peers")
        conn.execute("DELETE FROM session_peers")
    elif corruption == "peer_role":
        _drop_guards(conn, "peers")
        conn.execute("UPDATE peers SET role='assistant'")
    else:
        _session(conn, external=True, session="private-other-session")
        _drop_guards(conn, "messages")
        conn.execute("UPDATE messages SET session_id='private-other-session'")
    report = scan_coverage_health(path)
    assert report.complete and report.invalid == report.total == 1
    assert report.status == "invalid" and report.valid_raw_pruned == 0


@pytest.mark.parametrize("target", ["chunk", "session"])
def test_orphaned_proofs_do_not_disappear_through_inner_join(store, target):
    conn, path = store
    _message(conn, _session(conn))
    conn.execute("PRAGMA foreign_keys=OFF")
    table = "chunks" if target == "chunk" else "sessions"
    _drop_guards(conn, table)
    conn.execute(f"DELETE FROM {table}")
    report = scan_coverage_health(path)
    assert report.complete and report.invalid == report.total == report.checked == 1


def test_missing_proof_under_producer_frontier_fails_but_new_input_is_not_invented_coverage(store):
    conn, path = store
    session = _session(conn)
    _message(conn, session)
    _message(conn, session, materialize=False)
    assert scan_coverage_health(path).status == "valid"
    _drop_guards(conn, "message_retention_coverage")
    conn.execute("DELETE FROM message_retention_coverage")
    report = scan_coverage_health(path)
    assert report.complete and report.missing_proofs == 1 and report.status == "invalid"


@pytest.mark.parametrize("frontier", ["private-malformed-frontier", -1, 1.5])
def test_malformed_producer_frontier_never_hides_failure(store, frontier):
    conn, path = store
    _message(conn, _session(conn))
    conn.execute("UPDATE sessions SET coverage_message_id=?", (frontier,))
    report = scan_coverage_health(path)
    assert report.complete and report.invalid_frontiers == 1 and report.status == "invalid"
    assert "private" not in json.dumps(asdict(report))


def test_generic_exact_proofs_are_valid_but_not_ordered_frontier_authority(store):
    conn, path = store
    session = _session(conn)
    mid = _message(conn, session)
    chunk = lossless.coverage_chunk_id(session, mid)
    record_message_coverage(conn, message_id=mid, chunk_id=chunk, coverage_version="private-custom-version")
    report = scan_coverage_health(path)
    assert report.status == "valid" and report.independent_proofs == 1 and report.checked == 2
    _drop_guards(conn, "message_retention_coverage")
    conn.execute("DELETE FROM message_retention_coverage WHERE coverage_version=?",
                 (lossless.LOSSLESS_COVERAGE_VERSION,))
    report = scan_coverage_health(path)
    assert report.independent_proofs == 1 and report.invalid == 0
    assert report.missing_proofs == 1 and report.status == "invalid"
    assert "private-custom-version" not in json.dumps(asdict(report))
    conn.execute("UPDATE sessions SET coverage_message_id=NULL")
    assert scan_coverage_health(path).status == "valid"


def test_stale_ledger_failure_remains_actionable_without_read_side_clearing(store):
    conn, path = store
    session = _session(conn)
    _message(conn, session)
    lossless.record_coverage_integrity_failure(conn, session, reason="source_stream_invalid")
    before = _logical(conn)
    report = scan_coverage_health(path)
    assert report.complete and report.invalid == 0 and report.recorded_failure_sessions == 1
    assert report.status == "invalid" and _logical(conn) == before
    result = doctor._check_lossless_coverage(_cfg(path))
    assert result.status == doctor.FAIL and "audit does not clear them" in result.detail


@pytest.mark.parametrize("failure", [RuntimeError("private-source-secret"), ValueError("private-source-secret"),
                                     sqlite3.OperationalError("private-source-secret"), AssertionError("private-source-secret")])
def test_validator_exceptions_fail_closed_without_exporting_details(store, monkeypatch, failure):
    conn, path = store
    _message(conn, _session(conn))
    def fail(*_):
        raise failure
    monkeypatch.setattr(lossless, "validate_message_coverage_row", fail)
    report = scan_coverage_health(path)
    assert report.status != "valid"
    assert "private" not in json.dumps(asdict(report))
    assert "private" not in doctor._check_lossless_coverage(_cfg(path)).render()


@pytest.mark.parametrize("bounds,code", [
    ({"max_rows": 1}, "coverage_row_budget_exhausted"),
    ({"max_bytes": 1}, "coverage_byte_budget_exhausted"),
    ({"max_sql_steps": 1}, "diagnostic_budget_exhausted"),
    ({"max_seconds": 1e-12}, "diagnostic_budget_exhausted"),
])
def test_incomplete_audit_is_not_partial_green(store, bounds, code):
    conn, path = store
    session = _session(conn)
    _message(conn, session)
    _message(conn, session)
    before = _logical(conn)
    report = scan_coverage_health(path, **bounds)
    assert not report.complete and report.status == "unavailable" and report.error_code == code
    assert _logical(conn) == before


@pytest.mark.parametrize("bounds", [{"max_rows": True}, {"max_rows": 0}, {"max_bytes": -1},
                                  {"max_seconds": float("nan")}, {"max_sql_steps": 0}])
def test_invalid_bounds_do_not_open_or_create_store(tmp_path, bounds):
    path = tmp_path / "private-missing.sqlite"
    report = scan_coverage_health(path, **bounds)
    assert report.error_code == "invalid_bounds" and report.status == "unavailable"
    assert not path.exists()


def test_missing_file_and_old_schema_are_not_initialized(tmp_path):
    path = tmp_path / "private-missing.sqlite"
    assert scan_coverage_health(path).status == "unavailable" and not path.exists()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT)")
    conn.execute("INSERT INTO schema_meta VALUES ('schema_version','1')")
    conn.commit()
    before = _logical(conn)
    assert scan_coverage_health(path).error_code == "current_schema_required"
    assert _logical(conn) == before
    conn.close()


def test_python_control_flow_is_not_swallowed(store, monkeypatch):
    conn, path = store
    _message(conn, _session(conn))
    def interrupt(*_):
        raise KeyboardInterrupt
    monkeypatch.setattr(lossless, "validate_message_coverage_row", interrupt)
    with pytest.raises(KeyboardInterrupt):
        scan_coverage_health(path)
