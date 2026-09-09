"""Stored embedding compatibility is not established by global vec metadata."""

from __future__ import annotations

import sqlite3

import pytest

from hymem import doctor
from hymem.bootstrap import EnvConfig
from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.core.vectors import encode_vector
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.embeddings import fetch_chunk_embeddings, persist_chunk_embeddings
from hymem.extraction.embeddings import LocalHashEmbeddingClient


@pytest.fixture
def partial_switch(tmp_path):
    cfg = EnvConfig(
        root=tmp_path, llm_api_key=None, llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-v4-flash", embedding_api_key=None,
        embedding_base_url="local://feature-hash", embedding_model="new-producer",
        embedding_dim=3, embedding_backend="local_feature_hash",
        embedding_fallback_reason=None, aggregation_nodes_enabled=False,
        aggregation_digest_enabled=False,
    )
    conn = core_db.connect(HyMemConfig(root=tmp_path).db_path)
    core_db.initialize(conn)
    old = LocalHashEmbeddingClient(dim_value=3, model_name="old-producer")
    new = LocalHashEmbeddingClient(dim_value=3, model_name=cfg.embedding_model)
    conn.execute("INSERT INTO sessions(id) VALUES ('session')")
    from hymem.dreaming.chunks import Chunk, persist_chunks
    from hymem.dreaming.lossless import materialize_message_coverage
    for index in (1, 2):
        text = f"Source text number {index}"
        message_id = conn.execute(
            "INSERT INTO messages(session_id,role,content) VALUES ('session','user',?)", (text,),
        ).lastrowid
        with core_db.transaction(conn):
            materialize_message_coverage(conn, "session")
            persist_chunks(conn, [Chunk(
                f"chunk-{index}", "session", message_id, message_id,
                "test", f"user: {text}", (message_id,),
            )])
    with core_db.transaction(conn):
        assert persist_chunk_embeddings(conn, fetch_chunk_embeddings(conn, old)) == 2
    with core_db.transaction(conn):
        assert persist_chunk_embeddings(
            conn, fetch_chunk_embeddings(conn, new, exclude_ids={"chunk-2"}),
        ) == 1
    model, dim = embedding_storage_identity(new)
    assert conn.execute("SELECT value FROM schema_meta WHERE key='vec_model'").fetchone()[0] == model
    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    try:
        yield conn, cfg, new, model, dim
    finally:
        conn.close()


def test_doctor_rejects_partial_switch_even_when_global_metadata_matches(partial_switch):
    conn, cfg, new, model, dim = partial_switch
    results = doctor._check_schema_and_dim(cfg, dim, model)
    assert any(result.status == doctor.OK and result.name == "embedding identity" for result in results)
    assert any(
        result.status == doctor.FAIL and result.name == "stored embedding compatibility"
        and "incompatible=1" in result.detail for result in results
    )
    with core_db.transaction(conn):
        assert persist_chunk_embeddings(conn, fetch_chunk_embeddings(conn, new)) == 1
    fixed = doctor._check_schema_and_dim(cfg, dim, model)
    assert not any(result.status == doctor.FAIL for result in fixed)


_CURRENT = "hymem-embedding-producer-v1:" + "a" * 64
_OLD = "hymem-embedding-producer-v1:" + "b" * 64
_TABLES = (
    "chunk_embeddings", "message_embeddings", "edge_embeddings",
    "episode_embeddings", "narrative_fact_embeddings", "aggregation_node_embeddings",
)
_TYPED = {"episode_embeddings", "aggregation_node_embeddings"}


def _create_mirrors(conn):
    # Deliberately permissive fault-injection schema; the production false-green
    # regression above uses real v59 persistence. These fixtures can represent
    # damaged rows that normal write guards correctly reject.
    for table in _TABLES:
        typed = ", embedding_producer_key" if table in _TYPED else ""
        conn.execute(f"CREATE TABLE {table}(model, dim, vector_json{typed})")
    conn.execute("CREATE TABLE embedding_cache(model, dim, vector_json)")


@pytest.fixture
def mirror_conn():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    _create_mirrors(conn)
    try:
        yield conn
    finally:
        conn.close()


def _insert(conn, table, *, model=_CURRENT, dim=3, vector="[1,0,0]", typed_key=None):
    values = (model, dim, vector)
    if table in _TYPED:
        values += (model if typed_key is None else typed_key,)
    conn.execute(
        f"INSERT INTO {table} VALUES ({','.join('?' for _ in values)})", values,
    )


def _scan(conn, *, model=_CURRENT, dim=3):
    from hymem.embedding_health import scan_embedding_health

    return scan_embedding_health(conn, live_model=model, live_dim=dim)


def test_all_six_mirror_families_partition_every_stored_row(mirror_conn):
    for table in _TABLES:
        _insert(mirror_conn, table)
        _insert(mirror_conn, table, model=_OLD)
        _insert(mirror_conn, table, vector="[0,0,0]")
    health = _scan(mirror_conn)
    assert health.status == "incompatible"
    assert health.live_identity_verified is True
    assert tuple(row.table for row in health.tables) == _TABLES
    for table in health.tables:
        assert (table.total, table.current_compatible, table.incompatible, table.malformed, table.unverified) == (3, 1, 1, 1, 0)


@pytest.mark.parametrize(("model", "dim", "vector"), [
    (_CURRENT, 3, None), (_CURRENT, 3, "not-json-private-sentinel"),
    (_CURRENT, 3, "{}"), (_CURRENT, 3, "3"), (_CURRENT, 3, "[]"),
    (_CURRENT, 3, "[1,0]"), (_CURRENT, 3, "[0,0,0]"),
    (_CURRENT, 3, "[NaN,0,0]"), (_CURRENT, 3, "[Infinity,0,0]"),
    (_CURRENT, 3, "[1e308,1e308,0]"),
    (_CURRENT, 3, '["1",0,0]'), (_CURRENT, 3, "[true,0,0]"),
    (_CURRENT, 3, "b64f32:"), (_CURRENT, 3, "b64f32:!!!"),
    (_CURRENT, 3, "b64f32:AA=="), (_CURRENT, 3, b"\xff"),
    (_CURRENT, 3, encode_vector([float("nan"), 0, 0])),
    (_CURRENT, 3, encode_vector([float("inf"), 0, 0])),
    (_CURRENT, 0, "[1,0,0]"), (_CURRENT, "3", "[1,0,0]"),
    (_CURRENT, None, "[1,0,0]"), (_CURRENT, 3.0, "[1,0,0]"),
    ("https://private-sentinel", 3, "[1,0,0]"), (None, 3, "[1,0,0]"),
    (_CURRENT.encode(), 3, "[1,0,0]"),
])
def test_malformed_vectors_and_producer_fields_are_not_compatible(
    mirror_conn, model, dim, vector,
):
    _insert(mirror_conn, "chunk_embeddings", model=model, dim=dim, vector=vector)
    health = _scan(mirror_conn)
    row = health.tables[0]
    assert (row.total, row.malformed, row.current_compatible, row.incompatible) == (1, 1, 0, 0)
    assert health.status == "incompatible"
    assert "private-sentinel" not in repr(health)


@pytest.mark.parametrize("table", sorted(_TYPED))
def test_typed_producer_disagreement_is_malformed(mirror_conn, table):
    _insert(mirror_conn, table, typed_key=_OLD)
    report = next(row for row in _scan(mirror_conn).tables if row.table == table)
    assert report.malformed == 1 and report.incompatible == 0


@pytest.mark.parametrize("vector", ["[1,0,0]", encode_vector([1, 0, 0]), encode_vector([1, 0, 0]).encode()])
def test_legacy_json_and_packed_vectors_are_supported(mirror_conn, vector):
    _insert(mirror_conn, "edge_embeddings", vector=vector)
    report = _scan(mirror_conn)
    assert report.status == "compatible"
    assert sum(row.current_compatible for row in report.tables) == 1


def test_valid_old_dimension_is_incompatible_not_malformed(mirror_conn):
    _insert(mirror_conn, "chunk_embeddings", dim=2, vector="[1,0]")
    row = _scan(mirror_conn).tables[0]
    assert (row.incompatible, row.malformed) == (1, 0)


@pytest.mark.parametrize(("model", "dim"), [
    (None, None), (_CURRENT, None), (None, 3), ("private-sentinel", 3),
    (_CURRENT, True), (_CURRENT, 0), (_CURRENT, "3"),
])
def test_unknown_live_producer_is_unverified_even_with_no_rows(mirror_conn, model, dim):
    empty = _scan(mirror_conn, model=model, dim=dim)
    assert empty.status == "unverified" and not empty.live_identity_verified
    assert all(row.status == "unverified" for row in empty.tables)
    _insert(mirror_conn, "chunk_embeddings")
    _insert(mirror_conn, "chunk_embeddings", vector="[0,0,0]")
    health = _scan(mirror_conn, model=model, dim=dim)
    assert health.status == "incompatible"  # Known corruption still fails.
    assert (health.tables[0].unverified, health.tables[0].malformed) == (1, 1)
    assert health.tables[0].current_compatible == 0


def test_empty_mirrors_are_not_a_missing_source_vector_proof(mirror_conn):
    health = _scan(mirror_conn)
    assert health.status == "compatible"
    assert all(row.total == 0 for row in health.tables)
    results = doctor._check_stored_embedding_health(mirror_conn, 3, _CURRENT)
    assert "not missing-row/source-proof coverage" in results[0].detail


@pytest.mark.parametrize("fault", ["table", "column", "typed_column", "view"])
def test_missing_expected_schema_has_unknown_counts_and_fails_closed(mirror_conn, fault):
    table = "episode_embeddings" if fault == "typed_column" else "chunk_embeddings"
    if fault == "column":
        mirror_conn.execute(f"ALTER TABLE {table} RENAME COLUMN vector_json TO wrong_name")
    elif fault == "typed_column":
        mirror_conn.execute(f"ALTER TABLE {table} DROP COLUMN embedding_producer_key")
    else:
        mirror_conn.execute(f"DROP TABLE {table}")
        if fault == "view":
            mirror_conn.execute(f"CREATE VIEW {table} AS SELECT model,dim,vector_json FROM edge_embeddings")
    health = _scan(mirror_conn)
    row = next(item for item in health.tables if item.table == table)
    assert health.status == "unavailable"
    assert row.total is None and row.malformed is None
    assert doctor._check_stored_embedding_health(mirror_conn, 3, _CURRENT)[0].status == doctor.FAIL


def test_general_historical_cache_is_not_read_or_counted(mirror_conn):
    mirror_conn.execute("INSERT INTO embedding_cache VALUES ('private-sentinel', 9, 'broken')")

    def deny_cache_reads(action, arg1, *_):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and arg1 == "embedding_cache" else sqlite3.SQLITE_OK

    mirror_conn.set_authorizer(deny_cache_reads)
    health = _scan(mirror_conn)
    assert health.status == "compatible"
    assert sum(row.total for row in health.tables) == 0
    assert "private-sentinel" not in repr(health)


def test_foreign_key_violations_are_counts_not_private_row_or_table_names(mirror_conn):
    mirror_conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY)")
    for table in ("chunks", "private_sentinel_table"):
        mirror_conn.execute(f"CREATE TABLE {table}(id TEXT REFERENCES sessions(id))")
        mirror_conn.execute(f"INSERT INTO {table} VALUES ('private-sentinel-row')")
    before = "\n".join(mirror_conn.iterdump())
    health = _scan(mirror_conn)
    assert health.foreign_keys.total == 2
    assert health.foreign_keys.by_table == (("chunks", 1), ("other_tables", 1))
    results = doctor._check_stored_embedding_health(mirror_conn, 3, _CURRENT)
    assert results[1].status == doctor.FAIL
    assert "chunks=1" in results[1].detail and "other_tables=1" in results[1].detail
    assert "private" not in repr(health) + results[1].detail
    assert "\n".join(mirror_conn.iterdump()) == before


def test_foreign_key_schema_failure_is_unavailable_not_zero(mirror_conn):
    mirror_conn.execute("CREATE TABLE parent(value)")
    mirror_conn.execute("CREATE TABLE child(value REFERENCES parent(value))")
    health = _scan(mirror_conn)
    assert health.foreign_keys.status == "unavailable"
    assert health.foreign_keys.total is None
    assert doctor._check_stored_embedding_health(mirror_conn, 3, _CURRENT)[1].status == doctor.FAIL


def test_scans_stream_without_writes_or_fetchall(mirror_conn):
    mirror_conn.executemany(
        "INSERT INTO chunk_embeddings VALUES (?,3,'[1,0,0]')", [(_CURRENT,)] * 1000,
    )
    before = mirror_conn.total_changes
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_PRAGMA, sqlite3.SQLITE_TRANSACTION}
    mirror_conn.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)

    class StreamingCursor:
        def __init__(self, inner):
            self.inner = inner

        def __iter__(self):
            return iter(self.inner)

        def fetchone(self):
            return self.inner.fetchone()

        def fetchall(self):
            raise AssertionError("unbounded table materialization")

        def close(self):
            self.inner.close()

    class ReadOnlyConnection:
        @property
        def in_transaction(self):
            return mirror_conn.in_transaction

        def execute(self, *args):
            return StreamingCursor(mirror_conn.execute(*args))

        def rollback(self):
            mirror_conn.rollback()

    health = _scan(ReadOnlyConnection())
    assert health.status == "compatible" and health.tables[0].current_compatible == 1000
    assert mirror_conn.total_changes == before and not mirror_conn.in_transaction


def test_caller_owned_transaction_is_not_committed_or_rolled_back(mirror_conn):
    mirror_conn.execute("BEGIN")
    _insert(mirror_conn, "chunk_embeddings")
    assert _scan(mirror_conn).tables[0].total == 1
    assert mirror_conn.in_transaction
    mirror_conn.rollback()
    assert _scan(mirror_conn).tables[0].total == 0


def test_all_tables_and_foreign_keys_share_one_snapshot(tmp_path):
    path = tmp_path / "snapshot.sqlite"
    reader = sqlite3.connect(path, isolation_level=None)
    reader.execute("PRAGMA journal_mode=WAL")
    _create_mirrors(reader)
    writer = sqlite3.connect(path, isolation_level=None)
    changed = False

    def concurrent_commit(statement):
        nonlocal changed
        if not changed and statement.startswith("SELECT model, dim, vector_json"):
            changed = True
            _insert(writer, "message_embeddings")
            writer.execute("CREATE TABLE sessions(id PRIMARY KEY)")
            writer.execute("CREATE TABLE chunks(id REFERENCES sessions(id))")
            writer.execute("INSERT INTO chunks VALUES ('private-sentinel')")

    reader.set_trace_callback(concurrent_commit)
    try:
        snapshot = _scan(reader)
        assert changed
        assert all(table.total == 0 for table in snapshot.tables)
        assert snapshot.foreign_keys.total == 0
        reader.set_trace_callback(None)
        fresh = _scan(reader)
        assert fresh.tables[1].total == 1
        assert fresh.foreign_keys.total == 1
    finally:
        reader.close()
        writer.close()


def test_closed_database_is_unavailable_without_raw_error_text():
    conn = sqlite3.connect(":memory:")
    conn.close()
    health = _scan(conn)
    assert health.status == "unavailable"
    assert all(table.total is None for table in health.tables)


@pytest.mark.parametrize("failure", [RuntimeError("private-sentinel"), KeyboardInterrupt()])
def test_doctor_closes_connection_when_initialization_fails(partial_switch, monkeypatch, failure):
    _, cfg, _, model, dim = partial_switch

    class Connection:
        closed = False

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(doctor.core_db, "connect", lambda _: connection)

    def fail(_):
        raise failure

    monkeypatch.setattr(doctor.core_db, "initialize", fail)
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            doctor._check_schema_and_dim(cfg, dim, model)
    else:
        results = doctor._check_schema_and_dim(cfg, dim, model)
        assert results[0].status == doctor.FAIL
        assert "private-sentinel" not in results[0].detail
    assert connection.closed


@pytest.mark.parametrize("client_failed", [False, True])
def test_doctor_cli_fails_for_stale_rows_or_unverified_client(
    partial_switch, monkeypatch, capsys, client_failed,
):
    conn, cfg, new, model, dim = partial_switch
    monkeypatch.setattr(doctor, "resolve_env", lambda: cfg)
    monkeypatch.setattr(doctor, "_check_llm", lambda _: doctor._Result(doctor.OK, "LLM", "offline"))
    monkeypatch.setattr(doctor, "_check_embedding", lambda _: (
        doctor._Result(doctor.FAIL if client_failed else doctor.OK, "embeddings", "offline"),
        None if client_failed else dim, None if client_failed else model,
    ))
    assert doctor.run_doctor() == 1
    output = capsys.readouterr().out
    assert "stored embedding compatibility" in output and "All checks passed" not in output
    if client_failed:
        assert "live producer unverified" in output and "unverified=2" in output
    else:
        assert "incompatible=1" in output
        with core_db.transaction(conn):
            persist_chunk_embeddings(conn, fetch_chunk_embeddings(conn, new))
        assert doctor.run_doctor() == 0


def test_stored_health_never_logs_raw_models_urls_or_vectors(mirror_conn):
    _insert(mirror_conn, "chunk_embeddings", model="https://private-sentinel/token", vector="private-sentinel")
    results = doctor._check_stored_embedding_health(mirror_conn, 3, "private-sentinel-live")
    output = "\n".join(result.render() for result in results)
    assert "private-sentinel" not in output
    assert "https://" not in output
    assert "malformed=1" in output
