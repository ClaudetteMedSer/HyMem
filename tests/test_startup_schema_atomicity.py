"""Startup must never publish a temporarily absent authority view or guard."""

from __future__ import annotations

import sqlite3

import pytest

from hymem.core import db


@pytest.fixture
def connections(tmp_path):
    path = tmp_path / "startup.sqlite"
    writer = db.connect(path)
    db.initialize(writer)
    observer = db.connect(path)
    observer.execute("PRAGMA busy_timeout=0")
    try:
        yield writer, observer
    finally:
        writer.close()
        observer.close()


def _schema(conn):
    return [tuple(row) for row in conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE sql IS NOT NULL ORDER BY type,name"
    )]


def _logical_dump(conn):
    # Reinstalled schema objects change sqlite_master insertion order, which
    # iterdump preserves. Compare statements, not that physical row order.
    return sorted(conn.iterdump())


@pytest.mark.parametrize("entrypoint", [
    db.initialize, db._ensure_post_migration_runtime_guards,
    db._install_phase1_auxiliary_guards,
])
def test_repair_never_exposes_partial_schema_to_a_second_connection(
    connections, entrypoint,
):
    writer, observer = connections
    before = _schema(observer)
    observed = []
    failures = []

    def trace(statement):
        # Trace callbacks run immediately before each SQLite statement, not
        # on a guessed timing delay. The second connection therefore samples
        # every inter-statement DROP/CREATE window deterministically.
        if not statement.lstrip().upper().startswith(("CREATE ", "DROP ")):
            return
        try:
            observed.append(_schema(observer))
            observer.execute("SELECT COUNT(*) FROM current_phase1_publications").fetchone()
        except Exception as exc:
            # SQLite suppresses callback exceptions; assert outside it.
            failures.append(exc)

    writer.set_trace_callback(trace)
    try:
        entrypoint(writer)
    finally:
        writer.set_trace_callback(None)
    assert observed
    assert not failures
    assert all(snapshot == before for snapshot in observed)
    assert _schema(observer) == before
    assert not writer.in_transaction


def test_other_writer_cannot_exploit_a_temporarily_dropped_guard(connections):
    writer, observer = connections
    attempted = []

    def invalid_insert():
        observer.execute(
            "INSERT INTO entity_types(entity_canonical,type,confidence,origin) "
            "VALUES ('atomicity-probe','person',-1,'user')"
        )

    def trace(statement):
        if statement.startswith(
            "CREATE TRIGGER IF NOT EXISTS entity_type_authority_insert_guard"
        ):
            try:
                invalid_insert()
            except Exception as exc:
                attempted.append(exc)
            else:
                attempted.append(None)

    writer.set_trace_callback(trace)
    try:
        db._install_phase1_auxiliary_guards(writer)
    finally:
        writer.set_trace_callback(None)
    assert len(attempted) == 1
    assert isinstance(attempted[0], sqlite3.OperationalError)
    assert "locked" in str(attempted[0])
    with pytest.raises(sqlite3.IntegrityError, match="invalid entity type authority"):
        invalid_insert()
    assert observer.execute("SELECT COUNT(*) FROM entity_types").fetchone()[0] == 0


@pytest.mark.parametrize("entrypoint", [
    db._ensure_post_migration_runtime_guards,
    db._install_phase1_auxiliary_guards,
])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_helper_never_commits_or_rolls_back_its_callers_work(
    connections, entrypoint, fail,
):
    writer, observer = connections
    before = _schema(writer)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO schema_meta(key,value) VALUES ('caller-work','keep')")
    if fail:
        writer.set_authorizer(lambda action, name, *_: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_CREATE_TRIGGER
            and name == "entity_type_authority_insert_guard"
            else sqlite3.SQLITE_OK
        ))
    try:
        if fail:
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                entrypoint(writer)
        else:
            entrypoint(writer)
    finally:
        writer.set_authorizer(None)
    assert writer.in_transaction
    assert _schema(writer) == before
    assert writer.execute(
        "SELECT value FROM schema_meta WHERE key='caller-work'"
    ).fetchone()[0] == "keep"
    assert observer.execute(
        "SELECT value FROM schema_meta WHERE key='caller-work'"
    ).fetchone() is None
    writer.rollback()
    assert writer.execute(
        "SELECT value FROM schema_meta WHERE key='caller-work'"
    ).fetchone() is None


def test_initialize_refuses_caller_transaction_before_any_mutation(connections):
    writer, observer = connections
    before = _schema(writer)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO schema_meta(key,value) VALUES ('caller-work','keep')")
    changes = writer.total_changes
    with pytest.raises(RuntimeError, match="requires no active transaction"):
        db.initialize(writer)
    assert writer.in_transaction
    assert writer.total_changes == changes
    assert _schema(writer) == before
    assert observer.execute(
        "SELECT value FROM schema_meta WHERE key='caller-work'"
    ).fetchone() is None
    writer.rollback()


@pytest.mark.parametrize("fail_name", [
    "current_phase1_publications", "entity_type_authority_insert_guard",
])
def test_failed_startup_restores_complete_schema_and_normalization(connections, fail_name):
    writer, observer = connections
    before = _schema(writer)
    before_dump = _logical_dump(writer)
    writer.set_authorizer(lambda action, name, *_: (
        sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_CREATE_VIEW, sqlite3.SQLITE_CREATE_TRIGGER)
        and name == fail_name else sqlite3.SQLITE_OK
    ))
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            db.initialize(writer)
    finally:
        writer.set_authorizer(None)
    assert not writer.in_transaction
    assert _schema(observer) == before
    assert _logical_dump(observer) == before_dump
    observer.execute("SELECT COUNT(*) FROM current_phase1_publications").fetchone()
    db.initialize(writer)
    assert _schema(observer) == before


def test_malformed_owned_view_and_guard_are_repaired_together(connections):
    writer, observer = connections
    canonical = _schema(writer)
    writer.execute("DROP VIEW current_phase1_publications")
    writer.execute("CREATE VIEW current_phase1_publications AS SELECT 1 AS wrong")
    writer.execute("DROP TRIGGER entity_type_authority_insert_guard")
    writer.execute(
        "CREATE TRIGGER entity_type_authority_insert_guard BEFORE INSERT ON "
        "entity_types BEGIN SELECT 1; END"
    )
    corrupted = _schema(observer)
    snapshots = []

    def trace(statement):
        if statement.startswith("CREATE TRIGGER IF NOT EXISTS entity_type_authority_insert_guard"):
            snapshots.append(_schema(observer))

    writer.set_trace_callback(trace)
    try:
        db._install_phase1_auxiliary_guards(writer)
    finally:
        writer.set_trace_callback(None)
    assert snapshots == [corrupted]
    assert _schema(observer) == canonical
    with pytest.raises(sqlite3.IntegrityError, match="invalid entity type authority"):
        observer.execute(
            "INSERT INTO entity_types(entity_canonical,type,confidence) "
            "VALUES ('probe','person',-1)"
        )


def test_late_guard_failure_rolls_back_actual_legacy_row_healing(connections, monkeypatch):
    writer, observer = connections
    for name in ("user_profile_shape_insert_guard", "user_profile_source_insert_guard"):
        writer.execute(f"DROP TRIGGER {name}")
    writer.execute(
        "INSERT INTO user_profile(slot,slot_key,value) "
        "VALUES ('relationship','  Friend  ','Example Person')"
    )
    before = _logical_dump(writer)
    original = db._ensure_profile_active_invariants
    healed_inside = []

    def observed_healing(conn):
        original(conn)
        healed_inside.append(conn.execute("SELECT slot_key FROM user_profile").fetchone()[0])

    monkeypatch.setattr(db, "_ensure_profile_active_invariants", observed_healing)
    writer.set_authorizer(lambda action, name, *_: (
        sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_CREATE_TRIGGER
        and name == "entity_type_authority_insert_guard" else sqlite3.SQLITE_OK
    ))
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            db.initialize(writer)
    finally:
        writer.set_authorizer(None)
    assert healed_inside == ["friend"]
    assert _logical_dump(observer) == before
    assert not writer.in_transaction
    db.initialize(writer)
    assert observer.execute("SELECT slot_key FROM user_profile").fetchone()[0] == "friend"


def test_baseexception_also_rolls_back_our_savepoint_not_caller_work(connections):
    writer, observer = connections

    class Interrupted(BaseException):
        pass

    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO schema_meta(key,value) VALUES ('caller-work','keep')")
    with pytest.raises(Interrupted):
        with db._startup_schema_repair(writer):
            writer.execute("DROP VIEW current_phase1_publications")
            raise Interrupted()
    assert writer.in_transaction
    assert writer.execute("SELECT COUNT(*) FROM current_phase1_publications").fetchone()[0] == 0
    assert writer.execute(
        "SELECT value FROM schema_meta WHERE key='caller-work'"
    ).fetchone()[0] == "keep"
    writer.commit()
    assert observer.execute(
        "SELECT value FROM schema_meta WHERE key='caller-work'"
    ).fetchone()[0] == "keep"


def test_clean_reopen_preserves_logical_state_not_claimed_physical_noop(connections):
    writer, observer = connections
    writer.execute("INSERT INTO schema_meta(key,value) VALUES ('application-marker','stable')")
    before = _logical_dump(writer)
    before_schema = _schema(writer)
    for _ in range(2):
        db.initialize(writer)
        assert _logical_dump(observer) == before
        assert _schema(observer) == before_schema
        assert observer.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert observer.execute("PRAGMA foreign_key_check").fetchall() == []


def test_strict_startup_sql_parser_handles_trigger_case_quotes_and_comments(connections):
    writer, _ = connections
    with db._startup_schema_repair(writer):
        db._execute_startup_sql(writer, """
            CREATE TABLE parser_probe (value TEXT);
            -- A comment; is not a boundary by itself.
            CREATE TRIGGER parser_probe_guard BEFORE INSERT ON parser_probe
            BEGIN
                SELECT CASE WHEN new.value='semi;colon' THEN 1
                    ELSE RAISE(ABORT, 'not allowed; here') END;
                SELECT 1; -- Still inside the trigger.
            END;
            INSERT INTO parser_probe VALUES ('semi;colon');
            -- trailing comment; without a statement
        """)
    assert writer.execute("SELECT value FROM parser_probe").fetchone()[0] == "semi;colon"
    with pytest.raises(sqlite3.IntegrityError, match="not allowed; here"):
        writer.execute("INSERT INTO parser_probe VALUES ('other')")
    with pytest.raises(sqlite3.OperationalError):
        db._execute_startup_sql(writer, "CREATE TABLE parser_probe (value TEXT);")
