"""Losing every remote setting cannot silently overwrite its durable space."""

from __future__ import annotations

from contextlib import closing
import os
import socket
import sqlite3
import traceback

import pytest

from hymem import bootstrap, doctor, reembed
from hymem.config import HyMemConfig
from hymem.core import db
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.embeddings import fetch_chunk_embeddings, persist_chunk_embeddings
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.embeddings import LocalHashEmbeddingClient
from hymem.extraction.llm import StubLLMClient


REMOTE_KEY = "hymem-embedding-producer-v1:" + "f" * 64
MIRRORS = (
    "chunk_embeddings", "message_embeddings", "edge_embeddings",
    "episode_embeddings", "narrative_fact_embeddings",
    "aggregation_node_embeddings", "embedding_cache",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("HYMEM_") or name in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}:
            monkeypatch.delenv(name)
    monkeypatch.setenv("HYMEM_ROOT", str(tmp_path))
    monkeypatch.setenv("HYMEM_LLM_API_KEY", "synthetic-never-dispatched")
    monkeypatch.setattr(bootstrap, "_instance", None)
    monkeypatch.setattr(
        socket.socket, "connect",
        lambda *_a, **_k: pytest.fail("regression test issued a provider request"),
    )


def local_identity():
    return embedding_storage_identity(LocalHashEmbeddingClient())


def legacy_store(root, metadata=()):
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(HyMemConfig(root=root).db_path)
    conn.execute("CREATE TABLE schema_meta(key, value)")
    conn.executemany("INSERT INTO schema_meta VALUES (?, ?)", metadata)
    conn.commit()
    return conn


def forbidden_bootstrap(monkeypatch):
    def forbidden(*_a, **_k):
        pytest.fail("admission failure constructed a transport or opened HyMem")

    monkeypatch.setattr(bootstrap, "HyMem", forbidden)
    monkeypatch.setattr("hymem.contrib.openai_client.OpenAICompatibleClient", forbidden)
    monkeypatch.setattr(
        "hymem.contrib.openai_embedding_client.OpenAICompatibleEmbeddingClient", forbidden,
    )


def assert_refused(monkeypatch, root):
    forbidden_bootstrap(monkeypatch)
    path = HyMemConfig(root=root).db_path
    before = path.read_bytes() if path.is_file() else None
    with pytest.raises(RuntimeError, match="embedding") as caught:
        bootstrap.build_from_env()
    assert caught.value.__context__ is caught.value.__cause__ is None
    rendered = "".join(traceback.format_exception(caught.value))
    assert "private-sentinel" not in rendered
    result, live_dim, live_model = doctor._check_embedding(bootstrap.resolve_env())
    assert result.status == doctor.FAIL
    assert live_dim is live_model is None
    assert "private-sentinel" not in result.detail
    if before is not None:
        assert path.read_bytes() == before


@pytest.mark.parametrize("metadata", [
    [("vec_model", REMOTE_KEY), ("vec_dim", "384")],
    [("vec_model", "unknown-private-sentinel"), ("vec_dim", "384")],
    [("vec_model", None), ("vec_dim", "384")],
    [("vec_model", REMOTE_KEY)],
    [("vec_dim", "384")],
    [("vec_model", REMOTE_KEY), ("vec_model", REMOTE_KEY)],
])
def test_missing_remote_configuration_refuses_existing_identity_without_writes(
    monkeypatch, tmp_path, metadata,
):
    with closing(legacy_store(tmp_path, metadata)):
        pass
    assert_refused(monkeypatch, tmp_path)


@pytest.mark.parametrize("dim", [None, "not-a-dimension", "0384", -1, 0, "3", b"384", 384.0])
def test_malformed_or_different_local_dimension_refuses_startup(monkeypatch, tmp_path, dim):
    model, _ = local_identity()
    with closing(legacy_store(tmp_path, [("vec_model", model), ("vec_dim", dim)])):
        pass
    assert_refused(monkeypatch, tmp_path)


def test_current_remote_store_refuses_no_config_before_clients(monkeypatch, tmp_path):
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient

    client = OpenAICompatibleEmbeddingClient(
        api_key="synthetic", base_url="http://localhost:8766/v1", model="remote-test",
        dim=3, pin_dimension=True, deployment_revision="test-v1", deployment_tenant="tests",
    )
    try:
        model, dim = embedding_storage_identity(client)
        assert client.request_attempts == 0
    finally:
        client.close()
    with closing(db.connect(HyMemConfig(root=tmp_path).db_path)) as conn:
        db.initialize(conn)
        with db.embedding_mutation(conn):
            db.ensure_vec_table(conn, dim, model=model)
        assert conn.execute("SELECT value FROM schema_meta WHERE key='vec_model'").fetchone()[0] == model
    assert_refused(monkeypatch, tmp_path)


def test_readonly_admission_observes_committed_wal_metadata(monkeypatch, tmp_path):
    local_model, dim = local_identity()
    conn = legacy_store(tmp_path, [("vec_model", local_model), ("vec_dim", str(dim))])
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("UPDATE schema_meta SET value=? WHERE key='vec_model'", (REMOTE_KEY,))
        conn.commit()
        wal = HyMemConfig(root=tmp_path).db_path.with_name("hymem.sqlite-wal")
        before_wal = wal.read_bytes()
        assert before_wal
        assert_refused(monkeypatch, tmp_path)
        assert wal.read_bytes() == before_wal
        assert conn.execute("SELECT value FROM schema_meta WHERE key='vec_model'").fetchone()[0] == REMOTE_KEY
    finally:
        conn.close()


def test_local_repair_committed_only_in_wal_is_also_admitted(tmp_path):
    model, dim = local_identity()
    conn = legacy_store(tmp_path, [("vec_model", REMOTE_KEY), ("vec_dim", str(dim))])
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("UPDATE schema_meta SET value=? WHERE key='vec_model'", (model,))
        conn.commit()
        path = HyMemConfig(root=tmp_path).db_path
        wal = path.with_name("hymem.sqlite-wal")
        before_main, before_wal = path.read_bytes(), wal.read_bytes()
        assert bootstrap._embedding_configuration_error(bootstrap.resolve_env()) is None
        assert path.read_bytes() == before_main
        assert wal.read_bytes() == before_wal
    finally:
        conn.close()


@pytest.mark.parametrize("table", MIRRORS)
def test_missing_metadata_finds_remote_mirror_producer(monkeypatch, tmp_path, table):
    with closing(legacy_store(tmp_path)) as conn:
        conn.execute(f'CREATE TABLE "{table}" (model, dim)')
        conn.execute(f'INSERT INTO "{table}" VALUES (?, 384)', (REMOTE_KEY,))
        conn.commit()
    assert_refused(monkeypatch, tmp_path)


@pytest.mark.parametrize("table", ["episode_embeddings", "aggregation_node_embeddings"])
@pytest.mark.parametrize("typed", [REMOTE_KEY, None])
def test_missing_metadata_cannot_ignore_conflicting_typed_producer(monkeypatch, tmp_path, table, typed):
    model, dim = local_identity()
    with closing(legacy_store(tmp_path)) as conn:
        conn.execute(f'CREATE TABLE "{table}" (model, dim, embedding_producer_key)')
        conn.execute(f'INSERT INTO "{table}" VALUES (?, ?, ?)', (model, dim, typed))
        conn.commit()
    assert_refused(monkeypatch, tmp_path)


@pytest.mark.parametrize("typed", [False, True])
def test_mirror_identity_comparison_does_not_inherit_legacy_casefold_collation(monkeypatch, tmp_path, typed):
    model, dim = local_identity()
    with closing(legacy_store(tmp_path)) as conn:
        if typed:
            conn.execute("CREATE TABLE episode_embeddings(model, dim, embedding_producer_key TEXT COLLATE NOCASE)")
            conn.execute("INSERT INTO episode_embeddings VALUES (?, ?, ?)", (model, dim, model.upper()))
        else:
            conn.execute("CREATE TABLE chunk_embeddings(model TEXT COLLATE NOCASE, dim)")
            conn.execute("INSERT INTO chunk_embeddings VALUES (?, ?)", (model.upper(), dim))
        conn.commit()
    assert_refused(monkeypatch, tmp_path)


@pytest.mark.parametrize("damage", ["null_model", "null_dim", "text_dim", "missing_columns", "view", "shadow"])
def test_missing_metadata_fails_closed_on_uninspectable_vector_state(
    monkeypatch, tmp_path, damage,
):
    model, _ = local_identity()
    with closing(legacy_store(tmp_path)) as conn:
        if damage == "missing_columns":
            conn.execute("CREATE TABLE chunk_embeddings(legacy)")
            conn.execute("INSERT INTO chunk_embeddings VALUES (1)")
        elif damage == "view":
            conn.execute("CREATE VIEW chunk_embeddings AS SELECT 1")
        elif damage == "shadow":
            conn.execute("CREATE TABLE vec_chunks(legacy)")
        else:
            conn.execute("CREATE TABLE chunk_embeddings(model, dim)")
            conn.execute("INSERT INTO chunk_embeddings VALUES (?, ?)", (
                None if damage == "null_model" else model,
                None if damage == "null_dim" else ("384" if damage == "text_dim" else 384),
            ))
        conn.commit()
    assert_refused(monkeypatch, tmp_path)


def test_corrupt_existing_database_refuses_without_sensitive_exception_context(monkeypatch, tmp_path):
    path = HyMemConfig(root=tmp_path).db_path
    path.write_bytes(b"not a SQLite database: private-sentinel")
    assert_refused(monkeypatch, tmp_path)


def test_unreadable_database_refuses_without_sensitive_exception_context(monkeypatch, tmp_path):
    with closing(legacy_store(tmp_path)):
        pass

    def unreadable(*_a, **_k):
        raise sqlite3.OperationalError("private-sentinel")

    monkeypatch.setattr(bootstrap.sqlite3, "connect", unreadable)
    assert_refused(monkeypatch, tmp_path)


def test_preflight_closes_readonly_snapshot_and_never_calls_normal_store_open(monkeypatch, tmp_path):
    with closing(legacy_store(tmp_path, [("vec_model", REMOTE_KEY), ("vec_dim", "384")])):
        pass
    original_connect = sqlite3.connect
    handles = []
    statements = []
    closed = []

    class ObservedConnection(sqlite3.Connection):
        def close(self):
            closed.append(self.total_changes)
            super().close()

    def observed_connect(database, **kwargs):
        assert database.endswith("?mode=ro")
        assert kwargs["uri"] is True
        conn = original_connect(database, factory=ObservedConnection, **kwargs)
        conn.set_trace_callback(statements.append)
        handles.append(conn)
        return conn

    monkeypatch.setattr(bootstrap.sqlite3, "connect", observed_connect)
    monkeypatch.setattr(db, "connect", lambda *_a: pytest.fail("ordinary store opened"))
    monkeypatch.setattr(db, "initialize", lambda *_a: pytest.fail("store initialized"))
    assert_refused(monkeypatch, tmp_path)
    assert closed == [0, 0]
    assert all(sql.startswith(("SELECT", "PRAGMA", "BEGIN")) for sql in statements)
    for conn in handles:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


def test_failed_snapshot_close_cannot_admit_local_startup(monkeypatch, tmp_path):
    with closing(legacy_store(tmp_path)):
        pass
    original_connect = sqlite3.connect

    class CloseFailure(sqlite3.Connection):
        def close(self):
            super().close()
            raise RuntimeError("private-sentinel")

    monkeypatch.setattr(bootstrap.sqlite3, "connect", lambda *a, **k: original_connect(*a, factory=CloseFailure, **k))
    assert_refused(monkeypatch, tmp_path)


def test_read_deadline_expiry_refuses_even_an_otherwise_empty_store(monkeypatch, tmp_path):
    with closing(legacy_store(tmp_path)):
        pass
    ticks = iter([100.0, 103.0, 200.0, 203.0])
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: next(ticks))
    assert_refused(monkeypatch, tmp_path)


def test_sql_step_ceiling_refuses_a_partial_mirror_scan(monkeypatch, tmp_path):
    model, dim = local_identity()
    with closing(legacy_store(tmp_path)) as conn:
        conn.execute("CREATE TABLE chunk_embeddings(model, dim)")
        conn.executemany("INSERT INTO chunk_embeddings VALUES (?, ?)", [(model, dim)] * 240_000)
        conn.commit()
    # A fixed clock proves the VM budget, not the elapsed-time deadline, is
    # responsible for the refusal. The scan never returns partial success.
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: 100.0)
    assert_refused(monkeypatch, tmp_path)


@pytest.mark.parametrize("legacy", ["empty", "empty_old_mirror", "local_mirrors", "local_metadata"])
def test_empty_legacy_and_exact_local_state_admitted_without_mutation(tmp_path, legacy):
    model, dim = local_identity()
    metadata = [("vec_model", model), ("vec_dim", str(dim))] if legacy == "local_metadata" else []
    with closing(legacy_store(tmp_path, metadata)) as conn:
        if legacy == "empty_old_mirror":
            conn.execute("CREATE TABLE chunk_embeddings(legacy)")
        elif legacy == "local_mirrors":
            for table in MIRRORS:
                conn.execute(f'CREATE TABLE "{table}"(model, dim)')
                conn.execute(f'INSERT INTO "{table}" VALUES (?, ?)', (model, dim))
        elif legacy == "local_metadata":
            # Explicit local repair retains incompatible history and caches.
            conn.execute("CREATE TABLE chunk_embeddings(model, dim)")
            conn.execute("INSERT INTO chunk_embeddings VALUES (?, ?)", (REMOTE_KEY, 3))
        conn.commit()
    before = HyMemConfig(root=tmp_path).db_path.read_bytes()
    assert bootstrap._embedding_configuration_error(bootstrap.resolve_env()) is None
    assert HyMemConfig(root=tmp_path).db_path.read_bytes() == before


def test_fresh_local_and_reopen_work_with_uri_sensitive_root(monkeypatch, tmp_path):
    root = tmp_path / "root ? # %"
    monkeypatch.setenv("HYMEM_ROOT", str(root))
    monkeypatch.setattr("hymem.contrib.openai_client.OpenAICompatibleClient", lambda **_k: StubLLMClient())
    assert bootstrap._embedding_configuration_error(bootstrap.resolve_env()) is None
    assert not root.exists()
    for _ in range(2):
        memory = bootstrap.build_from_env()
        try:
            assert memory.embedding_status["backend"] == "local_feature_hash"
            model, dim = local_identity()
            with db.embedding_mutation(memory.conn):
                db.ensure_vec_table(memory.conn, dim, model=model)
        finally:
            bootstrap.shutdown_instance(memory)


def test_explicit_local_reembed_remains_available_after_missing_config_refusal(
    monkeypatch, tmp_path, capsys,
):
    path = HyMemConfig(root=tmp_path).db_path
    with closing(db.connect(path)) as conn:
        db.initialize(conn)
        conn.execute("INSERT INTO sessions(id) VALUES ('synthetic')")
        mid = conn.execute("INSERT INTO messages(session_id,role,content) VALUES ('synthetic','user','synthetic content')").lastrowid
        with db.transaction(conn):
            materialize_message_coverage(conn, "synthetic")
            persist_chunks(conn, [Chunk("test", "synthetic", mid, mid, "test", "user: synthetic content", (mid,))])
        with db.transaction(conn):
            old = LocalHashEmbeddingClient(model_name="previous-exact-producer")
            assert persist_chunk_embeddings(conn, fetch_chunk_embeddings(conn, old)) == 1
    assert bootstrap._embedding_configuration_error(bootstrap.resolve_env()) is not None
    assert reembed.main(["--db", str(path), "--apply", "--allow-local", "--json"]) == 0
    capsys.readouterr()
    assert bootstrap._embedding_configuration_error(bootstrap.resolve_env()) is None
    monkeypatch.setattr("hymem.contrib.openai_client.OpenAICompatibleClient", lambda **_k: StubLLMClient())
    memory = bootstrap.build_from_env()
    bootstrap.shutdown_instance(memory)


def test_explicit_remote_configuration_is_not_subject_to_default_local_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("HYMEM_EMBEDDING_API_KEY", "synthetic")
    monkeypatch.setenv("HYMEM_EMBEDDING_PIN_DIMENSION", "1")
    monkeypatch.setenv("HYMEM_EMBEDDING_DEPLOYMENT_REVISION", "test")
    monkeypatch.setenv("HYMEM_EMBEDDING_DEPLOYMENT_TENANT", "tests")
    with closing(legacy_store(tmp_path, [("vec_model", REMOTE_KEY), ("vec_dim", "384")])):
        pass
    monkeypatch.setattr(bootstrap.sqlite3, "connect", lambda *_a, **_k: pytest.fail("default-local guard inspected explicit remote store"))
    assert bootstrap._embedding_configuration_error(bootstrap.resolve_env()) is None
