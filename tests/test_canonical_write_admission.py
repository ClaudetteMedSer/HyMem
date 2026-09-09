"""Canonical writes fail closed without rewriting retained historical rows."""
from __future__ import annotations

import sqlite3

import pytest

from hymem.config import HyMemConfig
from hymem.core import db
from hymem.dreaming import canonicalize, inference, mentions, phase1_auxiliary
from tests.legacy_canonical import legacy_canonical_rows


OWNERS = (
    ("entity_aliases", "alias"),
    ("entity_aliases", "canonical"),
    ("knowledge_graph", "subject_canonical"),
    ("knowledge_graph", "object_canonical"),
    ("entity_mentions", "entity_canonical"),
)


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "admission.sqlite")
    db.initialize(connection)
    connection.execute("INSERT INTO sessions(id) VALUES ('session')")
    connection.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text) VALUES ('chunk','session',1,1,'test','project name')"
    )
    yield connection
    connection.close()


def _insert(conn, table, column, value):
    row = {
        "entity_aliases": {"alias": "surface", "canonical": "target"},
        "knowledge_graph": {
            "subject_canonical": "subject", "predicate": "uses", "object_canonical": "target",
        },
        "entity_mentions": {"chunk_id": "chunk", "entity_canonical": "target"},
        "entity_types": {"entity_canonical": "target", "type": "person"},
        "entity_properties": {"entity_canonical": "target", "key": "mode", "value": "test"},
    }[table]
    row[column] = value
    conn.execute(
        f"INSERT INTO {table}({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
        tuple(row.values()),
    )


@pytest.mark.parametrize("table,column", OWNERS)
@pytest.mark.parametrize("bad", [None, "", "ProjectName", "Cafe\u0301", "東京\x00駅", "x" * 513, b"target", "x'); DROP TABLE sessions; --"])
def test_invalid_insert_is_atomic(conn, table, column, bad):
    before = conn.total_changes
    with pytest.raises(sqlite3.IntegrityError, match="canonical identity"):
        _insert(conn, table, column, bad)
    assert conn.total_changes == before
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


@pytest.mark.parametrize("table,column", OWNERS)
@pytest.mark.parametrize("bad", [None, "", "ProjectName"])
def test_changed_canonical_update_rejected(conn, table, column, bad):
    _insert(conn, table, column, "target")
    with pytest.raises(sqlite3.IntegrityError, match="canonical identity"):
        conn.execute(f"UPDATE {table} SET {column}=?", (bad,))
    assert conn.execute(f"SELECT {column} FROM {table}").fetchone()[0] == "target"


@pytest.mark.parametrize("table,column", OWNERS)
@pytest.mark.parametrize("value", ["東京", "москва", "हिन्दी", "project_name"])
def test_normalized_unicode_and_ascii_admitted(conn, table, column, value):
    assert canonicalize.normalize(value) == value
    _insert(conn, table, column, value)
    conn.execute(f"UPDATE {table} SET {column}=?", (value,))


@pytest.mark.parametrize("table,column", OWNERS)
def test_null_udf_result_does_not_admit_insert_or_update(conn, table, column):
    _insert(conn, table, column, "target")
    conn.create_function("hymem_entity_canonical_is_normalized", 1, lambda _value: None)
    with pytest.raises(sqlite3.IntegrityError, match="canonical identity"):
        _insert(conn, table, column, "new_target")
    with pytest.raises(sqlite3.IntegrityError, match="canonical identity"):
        conn.execute(f"UPDATE {table} SET {column}='new_target'")


def test_missing_udf_fails_closed_for_external_sqlite_writer(conn, tmp_path):
    raw = sqlite3.connect(tmp_path / "admission.sqlite")
    try:
        with pytest.raises(sqlite3.OperationalError, match="hymem_entity_canonical_is_normalized"):
            raw.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('surface','target')")
        assert conn.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0] == 0
    finally:
        raw.close()


def test_legacy_fields_can_be_repaired_independently(conn):
    with legacy_canonical_rows(conn):
        conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical) VALUES ('OldSubject','uses','OldObject')")
        conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('OldAlias','OldTarget')")
    conn.execute("UPDATE knowledge_graph SET pos_evidence=2,subject_canonical=subject_canonical")
    conn.execute("UPDATE knowledge_graph SET subject_canonical='old_subject'")
    assert conn.execute("SELECT object_canonical FROM knowledge_graph").fetchone()[0] == "OldObject"
    conn.execute("UPDATE knowledge_graph SET object_canonical='old_object'")
    conn.execute("UPDATE entity_aliases SET alias='old_alias'")
    conn.execute("UPDATE entity_aliases SET canonical='old_target'")
    assert canonicalize.find_canonical_drift(conn) == []


def test_schema59_upgrade_preserves_legacy_rows_and_installs_guards(conn):
    with legacy_canonical_rows(conn):
        conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('project_name','Project_Name')")
        conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical) VALUES ('Project_Name','uses','ToolName')")
    for table, _column in OWNERS:
        for operation in ("insert", "update"):
            conn.execute(f"DROP TRIGGER IF EXISTS {table}_canonical_{operation}_guard")
    conn.execute("UPDATE schema_meta SET value='59' WHERE key='schema_version'")
    before = canonicalize.find_canonical_drift(conn)
    db.initialize(conn)
    assert db.schema_version(conn) == db.EXPECTED_SCHEMA_VERSION
    assert canonicalize.find_canonical_drift(conn) == before
    with pytest.raises(sqlite3.IntegrityError, match="canonical identity"):
        conn.execute("UPDATE knowledge_graph SET object_canonical='DifferentBadValue'")
    conn.execute("UPDATE knowledge_graph SET last_seen=last_seen")
    snapshot = tuple(conn.iterdump())
    db.initialize(conn)
    assert tuple(conn.iterdump()) == snapshot


def test_reopen_heals_missing_and_weakened_guards(conn):
    conn.execute("DROP TRIGGER knowledge_graph_canonical_insert_guard")
    conn.execute("DROP TRIGGER entity_mentions_canonical_update_guard")
    conn.execute("CREATE TRIGGER entity_mentions_canonical_update_guard BEFORE UPDATE ON entity_mentions BEGIN SELECT 1; END")
    db.initialize(conn)
    with pytest.raises(sqlite3.IntegrityError, match="canonical identity"):
        _insert(conn, "knowledge_graph", "object_canonical", "BadName")
    _insert(conn, "entity_mentions", "entity_canonical", "target")
    with pytest.raises(sqlite3.IntegrityError, match="canonical identity"):
        conn.execute("UPDATE entity_mentions SET entity_canonical='BadName'")


def test_guard_reinstallation_is_ddl_noop_when_already_exact(conn):
    traced = []
    conn.set_trace_callback(traced.append)
    try:
        db._install_canonical_write_guards(conn)
    finally:
        conn.set_trace_callback(None)
    assert not any(sql.startswith(("DROP", "CREATE")) for sql in traced)


@pytest.mark.parametrize("table,column", [
    ("entity_aliases", "canonical"), ("entity_mentions", "entity_canonical"),
])
def test_normal_initialize_refuses_missing_canonical_owner_column(conn, table, column):
    conn.execute(f"ALTER TABLE {table} RENAME COLUMN {column} TO invalid_column")
    with pytest.raises(RuntimeError, match="canonical write admission owner domain"):
        db.initialize(conn)


def test_migration_guard_failure_rolls_back_ddl_and_stamp(conn, monkeypatch):
    conn.execute("DROP TRIGGER entity_aliases_canonical_insert_guard")
    conn.execute("UPDATE schema_meta SET value='59' WHERE key='schema_version'")
    original = db._install_canonical_write_guards
    def fail_after_install(connection):
        original(connection)
        raise RuntimeError("injected admission migration failure")
    monkeypatch.setattr(db, "_install_canonical_write_guards", fail_after_install)
    with pytest.raises(RuntimeError, match="injected"):
        db._run_migrations(conn)
    assert db.schema_version(conn) == 59
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='entity_aliases_canonical_insert_guard'").fetchone() is None


@pytest.mark.parametrize("bad", ["Project_Name", b"project_name", ""])
def test_mentions_skip_legacy_alias_but_keep_valid_mentions_and_read_resolution(conn, bad):
    with legacy_canonical_rows(conn):
        conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('project_name',?)", (bad,))
    canonicalize.register_alias(conn, "database", "database")
    assert canonicalize.resolve(conn, "project name") == bad
    assert mentions.index_chunk_mentions(conn, "chunk", "project name database") == 1
    assert conn.execute("SELECT entity_canonical FROM entity_mentions").fetchone()[0] == "database"
    assert mentions.index_chunk_mentions(conn, "chunk", "project name database") == 0


@pytest.mark.parametrize("predicate", ["depends_on", "uses"])
@pytest.mark.parametrize("position", ["subject", "middle", "object"])
def test_inference_skips_bad_legacy_sources_for_both_rules(conn, tmp_path, predicate, position):
    names = ["legacy_start", "legacy_middle", "legacy_end"]
    names[["subject", "middle", "object"].index(position)] = "LegacyBad"
    with legacy_canonical_rows(conn):
        for s, p, o in (
            (names[0], predicate, names[1]), (names[1], "depends_on", names[2]),
        ):
            conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,pos_evidence) VALUES (?,?,?,10)", (s, p, o))
    for s, o in (("valid_start", "valid_middle"), ("valid_middle", "valid_end")):
        conn.execute("INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,pos_evidence) VALUES (?,'depends_on',?,10)", (s, o))
    assert inference.infer_transitive_edges(conn, HyMemConfig(root=tmp_path)) == 1
    assert conn.execute("SELECT subject_canonical,object_canonical FROM knowledge_graph WHERE derived=1").fetchone()[:] == ("valid_start", "valid_end")
    assert conn.execute("SELECT COUNT(*) FROM knowledge_graph WHERE derived=0").fetchone()[0] == 4


@pytest.mark.parametrize("method,args", [
    (phase1_auxiliary.add_manual_entity_type, ("person",)),
    (phase1_auxiliary.add_manual_entity_property, ("mode", "test")),
])
@pytest.mark.parametrize("bad", ["Project_Name", b"project_name", ""])
def test_manual_entity_writes_request_repair_of_legacy_alias(conn, method, args, bad):
    with legacy_canonical_rows(conn):
        conn.execute("INSERT INTO entity_aliases(alias,canonical) VALUES ('project_name',?)", (bad,))
    with pytest.raises(ValueError, match="canonical repair"):
        method(conn, "project name", *args)
    assert conn.execute("SELECT COUNT(*) FROM entity_types").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM entity_properties").fetchone()[0] == 0
