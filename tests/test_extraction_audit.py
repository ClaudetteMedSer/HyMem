"""Original occurrences are immutable, non-authoritative and privacy bounded."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from hymem import HyMem, portability
from hymem.config import HyMemConfig
from hymem.core import db, extraction_audit as audit
from hymem.dreaming import canonicalize, evidence, retention
from hymem.extraction.triples import Triple
from tests.test_portability import (
    _persist_portable_claim, _seed_shared_claim_artifact,
)
from tests.test_retention import _cover, _message, _session


@pytest.fixture
def store(tmp_path):
    conn = db.connect(tmp_path / "audit.sqlite")
    db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('s')")
    for chunk in ("c", "unused"):
        conn.execute("INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
                     "salience_reason,text,created_at) VALUES (?, 's',1,1,'test','text','2001-01-01')", (chunk,))
    edge = conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical) "
                        "VALUES ('project','uses','database')").lastrowid
    evidence.record_chunk_evidence(conn, edge_id=edge, chunk_id="c", evidence_kind="extraction",
                                   polarity=1, evidence_weight=1, weight_source="test")
    owner = conn.execute("SELECT id FROM kg_evidence").fetchone()[0]
    try:
        yield conn, owner
    finally:
        conn.close()


def raw_insert(conn, row):
    return conn.execute("INSERT INTO kg_evidence_extraction_audit(" + ",".join(audit.COLUMNS)
                        + ") VALUES (" + ",".join("?" for _ in audit.COLUMNS) + ")",
                        tuple(row[key] for key in audit.COLUMNS))


@pytest.mark.parametrize("field,value", [
    ("polarity", True), ("evidence_weight", 1.0), ("source_message_id", False),
    ("surface_subject", []), ("chunk_id", None), ("value_numeric", float("nan")),
    ("value_numeric", float("inf")), ("extracted_at", 1735689600),
])
def test_invalid_scalar_payloads_are_rejected_by_storage(store, field, value):
    conn, owner = store
    row = audit.carrier_record(conn, owner)
    payload = json.loads(row["payload_json"])
    payload["extraction"][field] = value
    row["payload_json"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    row["occurrence_hash"] = hashlib.sha256(row["payload_json"].encode()).hexdigest()
    with db.evidence_history_mutation(conn), pytest.raises(sqlite3.IntegrityError):
        raw_insert(conn, row)
    assert conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 0


@pytest.mark.parametrize("damage", ["hash", "projection", "duplicate", "unknown", "version", "noncanonical"])
def test_strict_json_and_hash_validation(store, damage):
    conn, owner = store
    row = audit.carrier_record(conn, owner)
    payload = json.loads(row["payload_json"])
    if damage == "hash":
        row["occurrence_hash"] = "0" * 64
    elif damage == "projection":
        row["chunk_id"] = "unused"
    elif damage == "duplicate":
        row["payload_json"] = row["payload_json"].replace('"version":1', '"version":1,"version":1')
    elif damage == "unknown":
        payload["extraction"]["published_at"] = "2000-01-01"
        row["payload_json"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    elif damage == "version":
        payload["version"] = True
        row["payload_json"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        row["payload_json"] = json.dumps(payload)
    if damage not in {"hash", "projection"}:
        row["occurrence_hash"] = hashlib.sha256(row["payload_json"].encode()).hexdigest()
    with db.evidence_history_mutation(conn), pytest.raises(sqlite3.IntegrityError):
        raw_insert(conn, row)


@pytest.mark.parametrize("context", [None, db.evidence_mutation, db.evidence_destructive_mutation])
def test_only_history_authorizes_insertion(store, context):
    import contextlib
    conn, owner = store
    with context(conn) if context else contextlib.nullcontext():
        with pytest.raises(sqlite3.IntegrityError, match="history authority"):
            audit.capture(conn, owner)


@pytest.mark.parametrize("function", ["hymem_evidence_mutation_authorized", "hymem_evidence_history_authorized", "hymem_extraction_audit_valid"])
def test_null_udf_never_admits_insertion(store, function):
    conn, owner = store
    conn.create_function(function, 8 if function.endswith("valid") else 0, lambda *args: None)
    with db.evidence_history_mutation(conn), pytest.raises(sqlite3.IntegrityError):
        audit.capture(conn, owner)


def test_null_authorization_never_admits_deletion(store):
    conn, owner = store
    with db.evidence_history_mutation(conn):
        audit.capture(conn, owner)
    for function in ("hymem_evidence_history_authorized", "hymem_evidence_destructive_authorized"):
        conn.create_function(function, 0, lambda: None)
    with db.evidence_mutation(conn), pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM kg_evidence_extraction_audit")


@pytest.mark.parametrize("clock", [None, "old malformed clock", "2025-01-01T01:00:00+01:00"])
def test_rename_preserves_raw_original_and_repeated_rename_adds_no_snapshot(store, clock):
    conn, owner = store
    with db.evidence_history_mutation(conn):
        conn.execute("UPDATE kg_evidence SET extracted_at=? WHERE id=?", (clock, owner))
    original = audit.carrier_record(conn, owner)
    authority = evidence.count_mismatches(conn)
    with db.transaction(conn):
        canonicalize.merge(conn, keep="renamed", drop="project")
        canonicalize.merge(conn, keep="final", drop="renamed")
    rows = conn.execute("SELECT * FROM kg_evidence_extraction_audit").fetchall()
    assert [dict(row) for row in rows] == [original]
    assert evidence.count_mismatches(conn) == authority == []
    assert conn.execute("SELECT pos_evidence FROM knowledge_graph").fetchone()[0] == 1


def test_audit_source_is_pinned_and_destructive_deletion_cascades(store, tmp_path):
    conn, owner = store
    with db.evidence_history_mutation(conn):
        audit.capture(conn, owner)
        conn.execute("UPDATE kg_evidence SET chunk_id='unused' WHERE id=?", (owner,))
    cfg = HyMemConfig(root=tmp_path, max_chunks=0, retention_days=1)
    assert retention.prune_chunks(conn, cfg) == 0
    with db.evidence_mutation(conn), pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM chunks WHERE id='c'")
    with db.evidence_history_mutation(conn), pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE kg_evidence_extraction_audit SET evidence_id=evidence_id")
    with db.evidence_mutation(conn), pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM kg_evidence_extraction_audit")
    with db.evidence_destructive_mutation(conn):
        conn.execute("DELETE FROM knowledge_graph")
    assert conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 0
    assert retention.prune_chunks(conn, cfg) == 2


def test_maintained_tombstone_pruning_releases_audit_only_donor_chunks(store, tmp_path):
    conn, owner = store
    with db.evidence_history_mutation(conn):
        audit.capture(conn, owner)
        conn.execute("UPDATE kg_evidence SET chunk_id='unused' WHERE id=?", (owner,))
    conn.execute("UPDATE knowledge_graph SET status='retracted',last_seen='2000-01-01',invalid_at='2000-01-01'")
    conn.execute("CREATE TABLE vec_chunks(rowid INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO vec_chunks SELECT rowid,'vector' FROM chunks")
    cfg = HyMemConfig(root=tmp_path, max_chunks=0, retention_days=1, tombstone_retention_days=30)
    # Dream's actual maintenance order first pins both source artifacts, then
    # explicitly destroys the expired tombstone and its audit ownership.
    with db.transaction(conn):
        assert retention.prune_chunks(conn, cfg) == 0
        assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 2
        assert retention.prune_retracted_edges(conn, cfg) == 1
    assert conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 0
    assert retention.prune_chunks(conn, cfg) == 2
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 0
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()


def test_coverage_restricts_release_but_allows_raw_pruning(store, tmp_path):
    conn, owner = store
    _session(conn, "covered", days_ago=200, summary=None, ended=True)
    mid = _message(conn, "covered", "Retain this source", days_ago=150)
    coverage_chunk = _cover(conn, "covered", mid, "Retain this source")
    payload = audit.decode(audit.carrier_record(conn, owner)["payload_json"])
    payload["extraction"].update(source_message_id=mid, source_session_id="covered",
                                 source_coverage_chunk_id=coverage_chunk,
                                 source_coverage_version="test-lossless-v1")
    with db.evidence_history_mutation(conn):
        audit.insert(conn, audit.record(owner, payload))
    with db.evidence_mutation(conn), pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM message_retention_coverage WHERE message_id=?", (mid,))
    assert retention.prune_messages(conn, HyMemConfig(root=tmp_path, message_retention_days=1)) == 1
    assert conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 1
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()


@pytest.mark.parametrize("stamp", [59, 60])
def test_migration_is_empty_and_reopen_heals_only_owned_support(store, stamp):
    conn, owner = store
    conn.execute("DROP TABLE kg_evidence_extraction_audit")
    conn.execute("UPDATE schema_meta SET value=? WHERE key='schema_version'", (str(stamp),))
    original = audit.carrier_record(conn, owner)
    db.initialize(conn)
    assert db.schema_version(conn) == 61
    assert audit.carrier_record(conn, owner) == original
    assert conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 0
    with db.evidence_history_mutation(conn):
        audit.capture(conn, owner)
    conn.execute("DROP TRIGGER extraction_audit_insert_guard")
    conn.execute("CREATE TRIGGER extraction_audit_insert_guard BEFORE INSERT ON kg_evidence_extraction_audit BEGIN SELECT 1; END")
    conn.execute("DROP INDEX idx_extraction_audit_chunk")
    db.initialize(conn)
    before = tuple(conn.iterdump())
    db.initialize(conn)
    assert tuple(conn.iterdump()) == before
    assert [dict(row) for row in conn.execute("SELECT * FROM kg_evidence_extraction_audit")] == [original]


@pytest.mark.parametrize("forged", [False, True])
def test_current_missing_or_forged_storage_fails_before_bootstrap(store, forged):
    conn, _ = store
    conn.execute("DROP TABLE kg_evidence_extraction_audit")
    if forged:
        conn.execute("CREATE TABLE kg_evidence_extraction_audit(evidence_id INTEGER, payload_json TEXT)")
    before = tuple(conn.iterdump())
    with pytest.raises(RuntimeError, match="v61 extraction audit storage"):
        db.initialize(conn)
    assert tuple(conn.iterdump()) == before


def test_migration_failure_rolls_back_storage_and_stamp(store, monkeypatch):
    conn, _ = store
    conn.execute("DROP TABLE kg_evidence_extraction_audit")
    conn.execute("UPDATE schema_meta SET value='60' WHERE key='schema_version'")
    normal = db._install_extraction_audit_guards
    def fail(connection):
        normal(connection)
        raise RuntimeError("injected audit failure")
    monkeypatch.setattr(db, "_install_extraction_audit_guards", fail)
    with pytest.raises(RuntimeError, match="injected audit failure"):
        db._run_migrations(conn)
    assert db.schema_version(conn) == 60
    assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='kg_evidence_extraction_audit'").fetchone()


def test_read_only_integrity_uses_shared_validation_functions(store, tmp_path):
    conn, owner = store
    with db.evidence_history_mutation(conn):
        audit.capture(conn, owner)
    readonly = sqlite3.connect(f"file:{tmp_path / 'audit.sqlite'}?mode=ro", uri=True)
    try:
        db.register_read_authority_functions(readonly)
        assert readonly.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        readonly.close()


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_chunk_pruning_rolls_back_vectors_and_chunks_on_late_failure(store, tmp_path, caller_transaction):
    conn, _ = store
    conn.execute("CREATE TABLE vec_chunks(rowid INTEGER PRIMARY KEY, value TEXT)")
    for index in (1, 2):
        conn.execute("INSERT INTO chunks(id,session_id,start_message_id,end_message_id,salience_reason,text,created_at) "
                     "VALUES (?, 's',1,1,'test','text','2000-01-01')", (f"drop{index}",))
    conn.execute("INSERT INTO vec_chunks SELECT rowid,'vector' FROM chunks")
    conn.execute("CREATE TRIGGER fail_late_chunk_delete BEFORE DELETE ON chunks WHEN old.id='drop2' "
                 "BEGIN SELECT RAISE(ABORT,'injected prune failure'); END")
    if caller_transaction:
        conn.execute("BEGIN")
        conn.execute("UPDATE sessions SET summary='caller work' WHERE id='s'")
    before = tuple(conn.iterdump())
    with pytest.raises(sqlite3.IntegrityError, match="injected prune failure"):
        retention.prune_chunks(conn, HyMemConfig(root=tmp_path, max_chunks=0, retention_days=1))
    assert conn.in_transaction is caller_transaction
    assert tuple(conn.iterdump()) == before
    if caller_transaction:
        conn.execute("ROLLBACK")


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_chunk_pruning_fences_concurrent_audit_admission(store, tmp_path, monkeypatch, caller_transaction):
    conn, owner = store
    observer = db.connect(tmp_path / "audit.sqlite")
    observer.execute("PRAGMA busy_timeout=1")
    payload = audit.decode(audit.carrier_record(conn, owner)["payload_json"])
    payload["extraction"]["chunk_id"] = "unused"
    normal = retention._prune_chunks
    checked = []
    def concurrent_insert(connection, cfg):
        with db.evidence_history_mutation(observer), pytest.raises(sqlite3.OperationalError, match="locked"):
            audit.insert(observer, audit.record(owner, payload))
        checked.append(True)
        return normal(connection, cfg)
    monkeypatch.setattr(retention, "_prune_chunks", concurrent_insert)
    if caller_transaction:
        conn.execute("BEGIN")
    try:
        assert retention.prune_chunks(conn, HyMemConfig(root=tmp_path, max_chunks=0, retention_days=1)) == 1
        assert checked == [True]
        assert conn.in_transaction is caller_transaction
    finally:
        if caller_transaction:
            conn.execute("ROLLBACK")
        observer.close()


def test_export_self_import_does_not_materialize_unchanged_virtual_original(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path / "hy", redact_secrets=False))
    try:
        chunk, mid = _seed_shared_claim_artifact(hy)
        _persist_portable_claim(hy, chunk, [Triple("service", "uses", "database", 1, source_message_id=mid)], prompt_version="v13")
        evidence_before = tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM kg_evidence"))
        before = tuple(hy.conn.iterdump())
        wire = tmp_path / "audit.jsonl"
        hy.export(wire)
        assert tuple(hy.conn.iterdump()) == before
        assert sum(hy.import_(wire).values()) == 0
        assert tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM kg_evidence")) == evidence_before
        assert hy.conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 0
        # The first import may install the existing derived-cache policy marker.
        before = tuple(hy.conn.iterdump())
        assert sum(hy.import_(wire).values()) == 0
        assert tuple(hy.conn.iterdump()) == before
    finally:
        hy.close()


def test_redaction_scrubs_audit_only_text_and_original_identity_before_sql(tmp_path):
    secret = "alice.private@example.com"
    src = HyMem(HyMemConfig(root=tmp_path / "src", redact_secrets=False))
    try:
        chunk, mid = _seed_shared_claim_artifact(src)
        _persist_portable_claim(src, chunk, [Triple("service", "uses", "database", 1, source_message_id=mid)], prompt_version="v13")
        owner = src.conn.execute("SELECT id FROM kg_evidence").fetchone()[0]
        payload = audit.decode(audit.carrier_record(src.conn, owner)["payload_json"])
        payload["original_edge"][0] = canonicalize.normalize(secret)
        payload["extraction"].update(surface_subject=secret, value_text=json.dumps({"nested": [secret]}),
                                     extraction_prompt_version=secret, extracted_at=secret)
        with db.evidence_history_mutation(src.conn):
            audit.insert(src.conn, audit.record(owner, payload))
        wire = tmp_path / "raw.jsonl"
        src.export(wire)
        assert secret in wire.read_text()
    finally:
        src.close()
    dst = HyMem(HyMemConfig(root=tmp_path / "dst", redact_secrets=True))
    traced = []
    dst.conn.set_trace_callback(traced.append)
    try:
        dst.import_(wire)
        before = tuple(dst.conn.iterdump())
        assert sum(dst.import_(wire).values()) == 0
        assert tuple(dst.conn.iterdump()) == before
        exported = tmp_path / "safe.jsonl"
        dst.export(exported)
        text = "\n".join(traced) + "\n".join(before) + exported.read_text()
        assert secret not in text
        assert canonicalize.normalize(secret) not in text
        assert dst.conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 1
    finally:
        dst.close()


def test_audit_only_secret_source_handle_fails_before_destination_sql(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "secret-source", redact_secrets=False))
    secret = "private.workspace@example.com"
    try:
        chunk, mid = _seed_shared_claim_artifact(src)
        _persist_portable_claim(src, chunk, [Triple("service", "uses", "database", 1, source_message_id=mid)], prompt_version="v13")
        owner = src.conn.execute("SELECT id FROM kg_evidence").fetchone()[0]
        payload = audit.decode(audit.carrier_record(src.conn, owner)["payload_json"])
        payload["extraction"]["source_peer_id"] = secret
        with db.evidence_history_mutation(src.conn):
            audit.insert(src.conn, audit.record(owner, payload))
        wire = tmp_path / "secret-handle.jsonl"
        src.export(wire)
    finally:
        src.close()
    dst = HyMem(HyMemConfig(root=tmp_path / "secret-target"))
    try:
        conn = dst.conn
        before = tuple(conn.iterdump())
        traced = []
        conn.set_trace_callback(traced.append)
        with pytest.raises(ValueError, match="source identity requires remapping"):
            dst.import_(wire)
        assert all(secret not in query for query in traced)
        assert not any(query.startswith(("INSERT", "UPDATE", "DELETE", "BEGIN")) for query in traced)
        assert tuple(conn.iterdump()) == before
    finally:
        dst.close()


def test_redaction_deduplicates_only_identical_transformed_payloads(store):
    conn, owner = store
    payload = audit.decode(audit.carrier_record(conn, owner)["payload_json"])
    # These free text variants have no distinct source coordinate: after
    # redaction they intentionally become the same non-authoritative payload.
    originals = []
    for secret in ("alice@example.com", "bob@example.com"):
        item = audit.decode(audit.encode(payload))
        item["extraction"]["value_text"] = secret
        originals.append(audit.record(owner, item))
    grouped = portability._collect_current_records(conn)
    grouped["edge_evidence_extraction_audit"] = originals
    portability._redact_portable_records(grouped)
    assert len(grouped["edge_evidence_extraction_audit"]) == 1
    transformed = grouped["edge_evidence_extraction_audit"][0]
    audit.validate_record(transformed)
    assert transformed["occurrence_hash"] not in {row["occurrence_hash"] for row in originals}


def test_redaction_preserves_benign_original_json_text_spelling(store):
    conn, owner = store
    payload = audit.decode(audit.carrier_record(conn, owner)["payload_json"])
    payload["extraction"]["value_text"] = ' { "unit" : "kg" } \n'
    original = audit.record(owner, payload)
    grouped = portability._collect_current_records(conn)
    grouped["edge_evidence_extraction_audit"] = [original]
    portability._redact_portable_records(grouped)
    assert grouped["edge_evidence_extraction_audit"] == [original]


def test_identical_payload_can_belong_to_distinct_revision_owners(store):
    conn, owner = store
    row = conn.execute("SELECT * FROM kg_evidence WHERE id=?", (owner,)).fetchone()
    columns = [key for key in row.keys() if key != "id"]
    next_revision = {**dict(row), "revision": 2, "is_current": 0,
                     "superseded_at": row["extracted_at"], "superseded_reason": "test_interval"}
    original = audit.carrier_record(conn, owner)
    with db.evidence_history_mutation(conn):
        second = conn.execute("INSERT INTO kg_evidence(" + ",".join(columns) + ") VALUES ("
                              + ",".join("?" for _ in columns) + ")",
                              tuple(next_revision[key] for key in columns)).lastrowid
        audit.insert(conn, original)
        audit.insert(conn, {**original, "evidence_id": second})
        assert audit.insert(conn, original) == 0
    assert conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 2


def _old_format_claim_wire(tmp_path, version):
    src = HyMem(HyMemConfig(root=tmp_path / "old-source", redact_secrets=False))
    wire = tmp_path / "old.jsonl"
    try:
        chunk, mid = _seed_shared_claim_artifact(src)
        _persist_portable_claim(src, chunk, [Triple("service", "uses", "database", 1, source_message_id=mid)], prompt_version="v13")
        src.export(wire)
    finally:
        src.close()
    spec = getattr(portability, f"_V{14 if version == 15 else version}_EXPORT_SPEC")
    columns = {kind: fields for kind, _table, fields in spec}
    objects = [json.loads(line) for line in wire.read_text().splitlines()]
    objects[0]["version"] = version
    body = [objects[0], *[
        {"type": row["type"], "record": {key: row["record"][key] for key in columns[row["type"]]}}
        for row in objects[1:-1] if row["type"] in columns
    ]]
    encoded = "".join(json.dumps(row) + "\n" for row in body)
    end = {"type": "_end", "counts": {kind: sum(row["type"] == kind for row in body) for kind in columns},
           "sha256": hashlib.sha256(encoded.encode()).hexdigest()}
    wire.write_text(encoded + json.dumps(end) + "\n")
    return wire


@pytest.mark.parametrize("version", list(range(7, 17)))
def test_frozen_old_formats_preserve_available_originals(tmp_path, version):
    wire = _old_format_claim_wire(tmp_path, version)
    dst = HyMem(HyMemConfig(root=tmp_path / "old-target", redact_secrets=False))
    try:
        dst.import_(wire)
        assert dst.conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 0
        assert not dst.conn.execute("PRAGMA foreign_key_check").fetchall()
        before = tuple(dst.conn.iterdump())
        assert sum(dst.import_(wire).values()) == 0
        assert tuple(dst.conn.iterdump()) == before
    finally:
        dst.close()


def test_audit_coverage_version_redaction_follows_the_source_proof(store):
    from hymem.dreaming.message_coverage import record_message_coverage
    conn, owner = store
    _session(conn, "covered", days_ago=200, summary=None, ended=True)
    mid = _message(conn, "covered", "Retained source", days_ago=150)
    coverage_chunk = _cover(conn, "covered", mid, "Retained source")
    secret_version = "owner@example.com"
    record_message_coverage(conn, message_id=mid, chunk_id=coverage_chunk, coverage_version=secret_version)
    payload = audit.decode(audit.carrier_record(conn, owner)["payload_json"])
    payload["extraction"].update(source_message_id=mid, source_session_id="covered",
                                 source_coverage_chunk_id=coverage_chunk, source_coverage_version=secret_version)
    grouped = portability._collect_current_records(conn)
    original = audit.record(owner, payload)
    grouped["edge_evidence_extraction_audit"] = [original]
    portability._redact_portable_records(grouped)
    transformed = grouped["edge_evidence_extraction_audit"][0]
    extraction = audit.validate_record(transformed)["extraction"]
    assert secret_version not in transformed["payload_json"]
    assert extraction["source_coverage_version"] == transformed["source_coverage_version"]
    assert transformed["occurrence_hash"] != original["occurrence_hash"]
    assert (mid, coverage_chunk, transformed["source_coverage_version"]) in {
        (proof["message_id"], proof["chunk_id"], proof["coverage_version"])
        for proof in grouped["message_retention_coverage"]
    }
