"""Synthetic, isolated ownership incident; no live files or provider calls."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from hymem.core import db
from hymem.coverage_health import scan_coverage_health
from hymem.dreaming.lossless import materialize_message_coverage
from tools.deployment import incident_residue_recovery as recovery


@pytest.fixture
def incident(tmp_path):
    path = tmp_path / "incident.sqlite"
    conn = db.connect(path)
    db.initialize(conn)
    session = "private-native-session"
    conn.execute("INSERT INTO sessions(id) VALUES (?)", (session,))
    conn.execute("INSERT INTO sessions(id) VALUES ('unrelated-session')")
    for content in ("old immutable native one", "old immutable native two"):
        conn.execute("INSERT INTO messages(session_id,role,content) VALUES (?,'user',?)", (session, content))
    conn.execute("INSERT INTO messages(session_id,role,content) VALUES ('unrelated-session','user','unrelated exact bytes')")
    materialize_message_coverage(conn, session)
    materialize_message_coverage(conn, "unrelated-session")
    conn.execute("UPDATE sessions SET digest_cursor_message_id=2,digested_message_id=2,profile_cursor_message_id=2,facts_message_id=2 WHERE id=?", (session,))
    # Unsupported direct binding and rollback deliberately reproduce operator
    # corruption. Production admission guards themselves already reject it.
    guard = conn.execute("SELECT sql FROM sqlite_master WHERE name='session_workspace_binding_guard'").fetchone()[0]
    conn.execute("DROP TRIGGER session_workspace_binding_guard")
    conn.execute("UPDATE sessions SET source_workspace_id='private-workspace' WHERE id=?", (session,))
    for peer, role in (("private-user", "user"), ("private-assistant", "assistant")):
        conn.execute("INSERT INTO peers(id,workspace_id,role) VALUES (?,'private-workspace',?)", (peer, role))
        conn.execute("INSERT INTO session_peers(session_id,workspace_id,peer_id) VALUES (?,'private-workspace',?)", (session, peer))
    mids = []
    for content in ("private unique upload cannot be discarded", "old immutable native two"):
        mids.append(conn.execute("INSERT INTO messages(session_id,role,content,created_at,source_peer_id,source_workspace_id) VALUES (?,'user',?,'2026-09-10T12:00:00.000Z','private-user','private-workspace')", (session, content)).lastrowid)
    materialize_message_coverage(conn, session)
    with db.embedding_mutation(conn):
        for mid in mids:
            conn.execute("INSERT INTO message_embeddings(message_id,source_coverage_chunk_id,source_coverage_version,text_hash,vector_json,model,dim) SELECT message_id,chunk_id,coverage_version,?,'[1.0,0.0,0.0]',?,3 FROM message_retention_coverage WHERE message_id=?", ("a" * 64, "hymem-embedding-producer-v1:" + "b" * 64, mid))
    conn.execute("UPDATE sessions SET source_workspace_id=NULL WHERE id=?", (session,))
    conn.execute(guard)
    conn.close()
    return path, tuple(mids), tmp_path / "bundle"


def _snapshot(path):
    conn = recovery._open_ro(path)
    try:
        return recovery._schema_hash(conn), recovery._snapshot(conn, recovery.MonotonicDeadline.after(120))
    finally:
        conn.rollback()
        conn.close()


def test_before_invalid_after_exact_lossless_archive_and_green_reopen(incident):
    path, mids, bundle = incident
    before = _snapshot(path)
    assert scan_coverage_health(path).invalid == 2
    pins = recovery.discover_pins(path, mids)
    assert "private" not in json.dumps(recovery.asdict(pins))
    result = recovery.recover(path, bundle, pins)
    assert result == {"status": "verified_clone", "quarantined_messages": 2, "production_writes": 0}
    assert _snapshot(path) == before == _snapshot(bundle / "original.sqlite")
    assert scan_coverage_health(path).invalid == 2  # Input deliberately untouched.
    assert scan_coverage_health(bundle / "repaired.sqlite").status == "valid"
    archive = json.loads((bundle / "rows.json").read_text())
    originals = archive["rows"]["messages"]
    assert len(originals["rows"]) == 2
    assert ["text", "private unique upload cannot be discarded"] in originals["rows"][0]
    assert len(archive["rows"]["message_embeddings"]["rows"]) == 2
    assert len(archive["rows"]["session_peers"]["rows"]) == 2
    assert json.loads((bundle / "ready.json").read_text())["removed_stale_memberships"] == 2
    conn = db.connect(bundle / "repaired.sqlite")
    try:
        assert conn.execute("SELECT coverage_message_id FROM sessions WHERE id='private-native-session'").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM session_peers").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM peers").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
        # Both ownership and coverage guards remain active after recovery.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE sessions SET source_workspace_id='private-workspace' WHERE id='private-native-session'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM message_retention_coverage WHERE message_id=1")
        assert list(conn.execute("PRAGMA foreign_key_check")) == []
    finally:
        conn.close()
    assert bundle.stat().st_mode & 0o777 == 0o700
    assert all((bundle / name).stat().st_mode & 0o777 == 0o600 for name in ("original.sqlite", "repaired.sqlite", "rows.json", "ready.json"))


def test_repeat_is_verified_noop_not_duplicate_or_missing_target_guess(incident):
    path, mids, bundle = incident
    pins = recovery.discover_pins(path, mids)
    recovery.recover(path, bundle, pins)
    before = _snapshot(bundle / "repaired.sqlite")
    assert recovery.recover(path, bundle, pins)["status"] == "already_verified"
    assert _snapshot(bundle / "repaired.sqlite") == before
    # The same successful archive also proves an already-adopted source.
    assert recovery.recover(bundle / "repaired.sqlite", bundle, pins)["status"] == "already_verified"
    assert _snapshot(bundle / "repaired.sqlite") == before


@pytest.mark.parametrize("stage", ["archived", "guard_suspended", "rows_quarantined", "before_commit"])
def test_injected_failure_retains_original_and_rolls_back_clone_and_guards(incident, stage):
    path, mids, bundle = incident
    pins = recovery.discover_pins(path, mids)
    before = _snapshot(path)

    def fail(current):
        if current == stage:
            raise KeyboardInterrupt("synthetic interruption")

    with pytest.raises(KeyboardInterrupt):
        recovery.recover(path, bundle, pins, _checkpoint=fail)
    assert _snapshot(path) == before == _snapshot(bundle / "original.sqlite")
    assert _snapshot(bundle / "repaired.sqlite") == before
    assert not (bundle / "ready.json").exists()
    assert (bundle / "rows.json").is_file()


@pytest.mark.parametrize("field", ["store_sha256", "schema_sha256", "session_sha256", "membership_sha256", "content_sha256"])
def test_changed_pins_refuse_without_archive_or_mutation(incident, field):
    path, mids, bundle = incident
    pins = recovery.discover_pins(path, mids)
    wrong = ("0" * 64, pins.content_sha256[1]) if field == "content_sha256" else "0" * 64
    before = _snapshot(path)
    with pytest.raises(recovery.Refused):
        recovery.recover(path, bundle, replace(pins, **{field: wrong}))
    assert _snapshot(path) == before and not bundle.exists()


@pytest.mark.parametrize("mutation", ["cursor", "reference", "json_reference", "overlapping_chunk", "other_invalid", "additional_message", "ledger", "missing_proof"])
def test_more_than_uncited_unconsumed_tail_refuses_even_with_fresh_pins(incident, mutation):
    path, mids, bundle = incident
    conn = db.connect(path)
    if mutation == "cursor":
        conn.execute("UPDATE sessions SET facts_message_id=? WHERE id='private-native-session'", (mids[0],))
    elif mutation == "reference":
        conn.execute("CREATE TABLE extra_dependency(source_message_id INTEGER)")
        conn.execute("INSERT INTO extra_dependency VALUES (?)", (mids[0],))
    elif mutation == "json_reference":
        conn.execute("CREATE TABLE extra_dependency(payload TEXT)")
        conn.execute("INSERT INTO extra_dependency VALUES (?)", (json.dumps({"citation": mids[0]}),))
    elif mutation == "overlapping_chunk":
        conn.execute("INSERT INTO chunks(id,session_id,start_message_id,end_message_id,text,salience_reason) VALUES ('derived','private-native-session',1,?,'independently derived','test')", (mids[-1] + 1,))
    elif mutation == "other_invalid":
        conn.execute("DROP TRIGGER session_workspace_binding_guard")
        conn.execute("UPDATE sessions SET source_workspace_id='private-workspace' WHERE id='unrelated-session'")
    elif mutation == "additional_message":
        conn.execute("INSERT INTO messages(session_id,role,content) VALUES ('private-native-session','user','later tail')")
    elif mutation == "ledger":
        from hymem.dreaming.lossless import record_coverage_integrity_failure
        record_coverage_integrity_failure(conn, "private-native-session", reason="source_stream_invalid")
    else:
        conn.execute("DROP TRIGGER message_lossless_stream_delete_guard")
        conn.execute("DELETE FROM message_retention_coverage WHERE message_id=1")
    conn.close()
    pins = recovery.discover_pins(path, mids)
    before = _snapshot(path)
    with pytest.raises(recovery.Refused):
        recovery.recover(path, bundle, pins)
    assert _snapshot(path) == before and not bundle.exists()


def test_target_proof_must_be_invalid_only_due_to_ownership(incident):
    path, mids, bundle = incident
    conn = db.connect(path)
    # Deliberately reproduce an additional unsupported mutation of raw bytes.
    for name, in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='messages'").fetchall():
        conn.execute(f'DROP TRIGGER "{name}"')
    conn.execute("UPDATE messages SET content='different raw bytes' WHERE id=?", (mids[0],))
    conn.close()
    pins = recovery.discover_pins(path, mids)
    with pytest.raises(recovery.Refused, match="ownership_only"):
        recovery.recover(path, bundle, pins)
    assert not bundle.exists()


def test_cli_refusal_never_echoes_private_values(incident, capsys):
    path, _mids, _bundle = incident
    assert recovery.main([str(path), "--discover", "222222", "333333"]) == 1
    output = capsys.readouterr().out
    assert "private" not in output and "incident_targets_not_present" in output


def test_symlink_input_is_refused(incident):
    path, mids, bundle = incident
    pins = recovery.discover_pins(path, mids)
    link = path.with_name("linked.sqlite")
    link.symlink_to(path)
    with pytest.raises(recovery.Refused, match="nonsymlink"):
        recovery.recover(link, bundle, pins)
    assert not bundle.exists()


def test_true_vector_rows_and_unrelated_vector_are_preserved(incident):
    path, mids, bundle = incident
    conn = db.connect(path)
    if not db._load_vec_extension(conn):
        pytest.skip("sqlite-vec unavailable in test interpreter")
    db._ensure_vec_table_named(conn, "vec_messages", 3)
    for mid in (1, *mids):
        conn.execute("INSERT INTO vec_messages(rowid,embedding) VALUES (?,?)", (mid, db._pack_vector([1.0, 0.0, 0.0])))
    conn.close()
    pins = recovery.discover_pins(path, mids)
    before = _snapshot(path)
    recovery.recover(path, bundle, pins)
    conn = db.connect(bundle / "repaired.sqlite")
    try:
        db._load_vec_extension(conn)
        assert [row[0] for row in conn.execute("SELECT rowid FROM vec_messages")] == [1]
    finally:
        conn.close()
    assert _snapshot(path) == before
    receipt = json.loads((bundle / "rows.json").read_text())
    assert len(receipt["rows"]["vec_messages"]["rows"]) == 2
    assert any(cell[0] == "blob" for row in receipt["rows"]["vec_messages"]["rows"] for cell in row)


def test_unexpected_trigger_side_effect_rolls_back_everything(incident):
    path, mids, bundle = incident
    conn = db.connect(path)
    conn.execute("CREATE TRIGGER malicious_side_effect AFTER DELETE ON messages WHEN old.id=" + str(mids[0]) + " BEGIN DELETE FROM messages WHERE id=1; END")
    conn.close()
    pins = recovery.discover_pins(path, mids)
    before = _snapshot(path)
    with pytest.raises(recovery.Refused, match="unexpected_repair_delta"):
        recovery.recover(path, bundle, pins)
    assert _snapshot(path) == before == _snapshot(bundle / "repaired.sqlite")
    assert not (bundle / "ready.json").exists()


def test_tampered_rows_receipt_refuses_idempotent_confirmation(incident):
    path, mids, bundle = incident
    pins = recovery.discover_pins(path, mids)
    recovery.recover(path, bundle, pins)
    (bundle / "rows.json").write_text("{}")
    with pytest.raises(recovery.Refused, match="archive_receipt_changed"):
        recovery.recover(path, bundle, pins)


def test_double_deletion_against_an_already_repaired_source_is_refused(incident):
    path, mids, bundle = incident
    pins = recovery.discover_pins(path, mids)
    recovery.recover(path, bundle, pins)
    repaired = bundle / "repaired.sqlite"
    before = _snapshot(repaired)
    with pytest.raises(recovery.Refused, match="snapshot_pin_mismatch"):
        recovery.recover(repaired, bundle.parent / "double-delete", pins)
    assert _snapshot(repaired) == before
    assert not (bundle.parent / "double-delete").exists()


def test_failure_after_commit_keeps_archive_but_never_advertises_ready(incident):
    path, mids, bundle = incident
    pins = recovery.discover_pins(path, mids)
    before = _snapshot(path)

    def fail(stage):
        if stage == "reopened":
            raise RuntimeError("simulated receipt boundary failure")

    with pytest.raises(RuntimeError):
        recovery.recover(path, bundle, pins, _checkpoint=fail)
    assert _snapshot(path) == before == _snapshot(bundle / "original.sqlite")
    assert scan_coverage_health(bundle / "repaired.sqlite").status == "valid"
    assert not (bundle / "ready.json").exists()


def test_input_never_uses_writable_core_connector_and_no_network(incident, monkeypatch):
    import socket
    path, mids, bundle = incident
    pins = recovery.discover_pins(path, mids)
    original = db.connect

    def connect(candidate):
        assert Path(candidate) != path
        return original(candidate)

    monkeypatch.setattr(db, "connect", connect)
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: pytest.fail("unexpected network"))
    assert recovery.recover(path, bundle, pins)["status"] == "verified_clone"
