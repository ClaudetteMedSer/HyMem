"""Explicit remote failures cannot silently replace an existing vector space."""

from __future__ import annotations

import asyncio
from contextlib import closing
import importlib
import os
from pathlib import Path
import socket
import sqlite3
import traceback
from types import SimpleNamespace

import pytest

from hymem import bootstrap, doctor
from hymem.config import HyMemConfig
from hymem.core import db
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.embeddings import fetch_chunk_embeddings, persist_chunk_embeddings
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.embeddings import LocalHashEmbeddingClient
from hymem.extraction.llm import StubLLMClient


@pytest.fixture(autouse=True)
def _isolated_offline_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("HYMEM_") or name in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}:
            monkeypatch.delenv(name)
    monkeypatch.setattr(bootstrap, "_instance", None)

    def forbidden_network(*_args, **_kwargs):
        pytest.fail("startup regression tests must not issue network requests")

    monkeypatch.setattr(socket.socket, "connect", forbidden_network)


class _OwnedLLM(StubLLMClient):
    def __init__(self, *, close_failure=False):
        super().__init__()
        self.close_calls = 0
        self.close_failure = close_failure

    def close(self):
        self.close_calls += 1
        if self.close_failure:
            raise RuntimeError("cleanup-private-sentinel")


def _environment(monkeypatch, root, fault):
    values = {
        "HYMEM_ROOT": str(root),
        "HYMEM_LLM_API_KEY": "synthetic-llm-never-dispatched",
        "HYMEM_EMBEDDING_BASE_URL": "http://embedding-server:8766/v1",
        "HYMEM_EMBEDDING_MODEL": "synthetic-remote-model",
        "HYMEM_EMBEDDING_DIM": "384",
        "HYMEM_EMBEDDING_PIN_DIMENSION": "1",
        "HYMEM_EMBEDDING_DEPLOYMENT_REVISION": "synthetic-v1",
        "HYMEM_EMBEDDING_DEPLOYMENT_TENANT": "synthetic-tests",
        "HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP": "1",
    }
    if fault == "endpoint_rejected":
        values.pop("HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP")
    elif fault == "credentials_missing":
        values["HYMEM_EMBEDDING_BASE_URL"] = "https://embeddings.example/v1"
    elif fault.startswith("HYMEM_"):
        values.pop(fault)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _seed_previous_space(root):
    root.mkdir()
    conn = db.connect(HyMemConfig(root=root).db_path)
    try:
        db.initialize(conn)
        conn.execute("INSERT INTO sessions(id) VALUES ('synthetic-session')")
        message_id = conn.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?, 'user', ?)",
            ("synthetic-session", "This is synthetic test material."),
        ).lastrowid
        with db.transaction(conn):
            materialize_message_coverage(conn, "synthetic-session")
            persist_chunks(conn, [Chunk(
                "synthetic-chunk", "synthetic-session", message_id, message_id,
                "test", "user: This is synthetic test material.", (message_id,),
            )])
        prior = LocalHashEmbeddingClient(model_name="synthetic-previous-space")
        prior_model, _ = embedding_storage_identity(prior)
        with db.transaction(conn):
            assert persist_chunk_embeddings(conn, fetch_chunk_embeddings(conn, prior)) == 1
        assert conn.execute(
            "SELECT value FROM schema_meta WHERE key='vec_model'"
        ).fetchone()[0] == prior_model
    finally:
        conn.close()


def _file_snapshot(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def _stored_rows(root):
    # The synthetic fixture is closed and checkpointed before this check.
    # Immutable mode prevents the verifier itself from creating WAL sidecars.
    uri = HyMemConfig(root=root).db_path.as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        return tuple(conn.execute("SELECT * FROM chunk_embeddings")), tuple(
            conn.execute("SELECT * FROM schema_meta ORDER BY key")
        )


def _call_surface(monkeypatch, surface):
    if surface == "bootstrap":
        memory = bootstrap.build_from_env()
        try:
            pytest.fail("invalid explicit remote configuration started successfully")
        finally:
            bootstrap.shutdown_instance(memory)
    elif surface == "mcp":
        from hymem import server

        monkeypatch.setattr(server, "_get_mcp", lambda: SimpleNamespace())
        server.main()
    else:
        honcho = importlib.import_module("hymem.honcho.app")
        monkeypatch.setattr(honcho, "_scheduler", None)

        async def enter_lifespan():
            async with honcho._lifespan(honcho.app):
                pytest.fail("Honcho advertised an invalid embedding configuration")

        asyncio.run(enter_lifespan())


@pytest.mark.parametrize("surface", ["bootstrap", "mcp", "honcho"])
@pytest.mark.parametrize("existing", [False, True], ids=["new-store", "existing-vectors"])
@pytest.mark.parametrize("fault", ["endpoint_rejected", "credentials_missing", "constructor_failure"])
def test_explicit_remote_failure_never_opens_store_or_changes_vectors(
    monkeypatch, tmp_path, surface, existing, fault,
):
    root = tmp_path / "store"
    if existing:
        _seed_previous_space(root)
    before_files = _file_snapshot(root)
    before_rows = _stored_rows(root) if existing else None
    _environment(monkeypatch, root, fault)
    calls = []
    llm = _OwnedLLM()

    def llm_factory(**_kwargs):
        calls.append("llm")
        return llm

    def broken_remote(**_kwargs):
        calls.append("embedding")
        raise RuntimeError("provider-private-sentinel https://secret.example/private")

    monkeypatch.setattr("hymem.contrib.openai_client.OpenAICompatibleClient", llm_factory)
    monkeypatch.setattr(
        "hymem.contrib.openai_embedding_client.OpenAICompatibleEmbeddingClient", broken_remote,
    )
    # Even an eager HyMem constructor is forbidden. File/row proofs below
    # independently cover the real seeded store and any sidecar files.
    monkeypatch.setattr(bootstrap, "HyMem", lambda *_a, **_k: pytest.fail("store opened"))
    with pytest.raises(RuntimeError, match="No local embedding fallback") as caught:
        _call_surface(monkeypatch, surface)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "private-sentinel" not in rendered
    assert "secret.example" not in rendered
    assert caught.value.__cause__ is caught.value.__context__ is None
    assert calls == (["llm", "embedding"] if fault == "constructor_failure" else [])
    assert llm.close_calls == (fault == "constructor_failure")
    assert bootstrap._instance is None
    assert _file_snapshot(root) == before_files
    if existing:
        assert _stored_rows(root) == before_rows
    else:
        assert not root.exists()


@pytest.mark.parametrize("missing", [
    "HYMEM_EMBEDDING_PIN_DIMENSION",
    "HYMEM_EMBEDDING_DEPLOYMENT_REVISION",
    "HYMEM_EMBEDDING_DEPLOYMENT_TENANT",
])
@pytest.mark.parametrize("existing", [False, True])
def test_incomplete_explicit_producer_identity_cannot_open_store(
    monkeypatch, tmp_path, missing, existing,
):
    root = tmp_path / "store"
    if existing:
        _seed_previous_space(root)
    before = _file_snapshot(root)
    _environment(monkeypatch, root, missing)
    monkeypatch.setattr(
        "hymem.contrib.openai_client.OpenAICompatibleClient",
        lambda **_k: pytest.fail("client constructed before identity validation"),
    )
    with pytest.raises(RuntimeError, match="remote embeddings require"):
        bootstrap.build_from_env()
    assert _file_snapshot(root) == before
    if not existing:
        assert not root.exists()


@pytest.mark.parametrize("failure_type", [RuntimeError, ImportError, ValueError])
def test_embedding_initialization_fault_redacts_context_and_preserves_cleanup(
    monkeypatch, tmp_path, failure_type,
):
    _environment(monkeypatch, tmp_path / "store", "constructor_failure")
    llm = _OwnedLLM(close_failure=True)
    monkeypatch.setattr("hymem.contrib.openai_client.OpenAICompatibleClient", lambda **_k: llm)

    def broken_remote(**_kwargs):
        raise failure_type("provider-private-sentinel")

    monkeypatch.setattr(
        "hymem.contrib.openai_embedding_client.OpenAICompatibleEmbeddingClient", broken_remote,
    )
    with pytest.raises(RuntimeError, match="could not be initialized") as caught:
        bootstrap.build_from_env()
    assert llm.close_calls == 1
    assert caught.value.__cause__ is caught.value.__context__ is None
    assert caught.value.__notes__ == ["LLM transport cleanup failed: RuntimeError"]
    assert "private-sentinel" not in "".join(traceback.format_exception(caught.value))
    assert not (tmp_path / "store").exists()


def test_intentional_default_local_startup_still_works(monkeypatch, tmp_path):
    root = tmp_path / "store"
    monkeypatch.setenv("HYMEM_ROOT", str(root))
    monkeypatch.setenv("HYMEM_LLM_API_KEY", "synthetic-llm-never-dispatched")
    llm = _OwnedLLM()
    monkeypatch.setattr("hymem.contrib.openai_client.OpenAICompatibleClient", lambda **_k: llm)
    monkeypatch.setattr(
        "hymem.contrib.openai_embedding_client.OpenAICompatibleEmbeddingClient",
        lambda **_k: pytest.fail("unconfigured remote client constructed"),
    )
    cfg = bootstrap.resolve_env()
    assert cfg.has_embedding_client is True
    result, live_dim, live_model = doctor._check_embedding(cfg)
    assert result.status == doctor.OK
    assert live_dim and live_model
    memory = bootstrap.build_from_env()
    try:
        assert memory.embedding_status["backend"] == "local_feature_hash"
        assert memory.embedding_status["fallback_reason"] is None
        assert db.schema_version(memory.conn) == db.EXPECTED_SCHEMA_VERSION
        assert HyMemConfig(root=root).db_path.is_file()
    finally:
        bootstrap.shutdown_instance(memory)
    assert llm.close_calls == 1
