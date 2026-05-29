"""Regression tests for the operational-hardening pass:
redaction, ingest size limits, query cap, embedding model/dim guard, retry."""
from __future__ import annotations

import sqlite3

import pytest

from hymem import redaction
from hymem.api import HyMem
from hymem.config import HyMemConfig
from hymem.extraction.embeddings import StubEmbeddingClient
from hymem.extraction.retry import with_retry
from hymem.query.augment import _embeddings_compatible


# ---- redaction --------------------------------------------------------------

def test_redact_covers_common_secret_shapes():
    text = (
        "key sk-ABCD1234efgh5678ijkl and AKIA1234567890ABCDEF and "
        "mail me at jane.doe@example.com or use Bearer abcdefabcdef1234567890"
    )
    out = redaction.redact(text)
    assert "sk-ABCD1234efgh5678ijkl" not in out
    assert "AKIA1234567890ABCDEF" not in out
    assert "jane.doe@example.com" not in out
    assert "abcdefabcdef1234567890" not in out
    assert "[REDACTED-API-KEY]" in out
    assert "[REDACTED-AWS-KEY]" in out
    assert "[REDACTED-EMAIL]" in out
    assert "[REDACTED-TOKEN]" in out


def test_redact_is_idempotent():
    once = redaction.redact("token sk-ABCD1234efgh5678ijkl now")
    assert redaction.redact(once) == once


def test_redact_leaves_ordinary_prose_untouched():
    prose = "We switched the build tool to uv and deploy to fly.io on push."
    assert redaction.redact(prose) == prose


def test_log_message_redacts_before_storage(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    hy.log_message("s1", "user", "my openai key is sk-ABCD1234efgh5678ijkl ok")
    row = hy.conn.execute(
        "SELECT content FROM messages WHERE session_id = 's1'"
    ).fetchone()
    assert "sk-ABCD1234efgh5678ijkl" not in row["content"]
    assert "[REDACTED-API-KEY]" in row["content"]
    hy.close()


def test_redaction_can_be_disabled(tmp_path):
    cfg = HyMemConfig(root=tmp_path, redact_secrets=False)
    hy = HyMem(cfg)
    secret = "sk-ABCD1234efgh5678ijkl"
    hy.log_message("s1", "user", f"key {secret}")
    row = hy.conn.execute(
        "SELECT content FROM messages WHERE session_id = 's1'"
    ).fetchone()
    assert secret in row["content"]
    hy.close()


# ---- ingest size limits -----------------------------------------------------

def test_oversized_message_is_truncated(tmp_path):
    cfg = HyMemConfig(root=tmp_path, max_message_chars=50, redact_secrets=False)
    hy = HyMem(cfg)
    hy.log_message("s1", "user", "x" * 500)
    row = hy.conn.execute(
        "SELECT content FROM messages WHERE session_id = 's1'"
    ).fetchone()
    assert len(row["content"]) < 500
    assert row["content"].endswith("[TRUNCATED]")
    hy.close()


def test_oversized_query_does_not_crash(tmp_path):
    cfg = HyMemConfig(root=tmp_path, max_query_chars=20)
    hy = HyMem(cfg)
    ctx = hy.augment("y" * 1000)
    assert ctx is not None
    hy.close()


# ---- embedding model/dim guard ---------------------------------------------

def _seed_chunk_embedding(conn: sqlite3.Connection, model: str, dim: int) -> None:
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s1')")
    mid = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s1', 'user', 'hello')"
    ).lastrowid
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('c1', 's1', ?, ?, 'test', 'hello')",
        (mid, mid),
    )
    conn.execute(
        "INSERT INTO chunk_embeddings(chunk_id, vector_json, model, dim) "
        "VALUES ('c1', '[0.0]', ?, ?)",
        (model, dim),
    )


def test_embeddings_compatible_matches(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _seed_chunk_embedding(conn, "stub", 16)
    assert _embeddings_compatible(conn, StubEmbeddingClient(dim_value=16)) is True
    hy.close()


def test_embeddings_incompatible_on_dim_change(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _seed_chunk_embedding(conn, "stub", 16)
    assert _embeddings_compatible(conn, StubEmbeddingClient(dim_value=8)) is False
    hy.close()


def test_embeddings_incompatible_on_model_change(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _seed_chunk_embedding(conn, "old-model", 16)
    client = StubEmbeddingClient(model_name="new-model", dim_value=16)
    assert _embeddings_compatible(conn, client) is False
    hy.close()


def test_embeddings_compatible_when_empty(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path))
    assert _embeddings_compatible(hy.conn, StubEmbeddingClient()) is True
    hy.close()


def test_embeddings_incompatible_on_mixed_corpus(tmp_path):
    """A corpus with a mix of matching and stale rows must be rejected even
    though one row matches the active client."""
    hy = HyMem(HyMemConfig(root=tmp_path))
    conn = hy.conn
    _seed_chunk_embedding(conn, "stub", 16)  # matches active client
    # A second chunk embedded by an older model still lingers in the corpus.
    mid = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s1', 'user', 'old')"
    ).lastrowid
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('c2', 's1', ?, ?, 'test', 'old')",
        (mid, mid),
    )
    conn.execute(
        "INSERT INTO chunk_embeddings(chunk_id, vector_json, model, dim) "
        "VALUES ('c2', '[0.0]', 'old-model', 16)"
    )
    assert _embeddings_compatible(conn, StubEmbeddingClient(dim_value=16)) is False
    hy.close()


# ---- retry ------------------------------------------------------------------

def test_with_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr("hymem.extraction.retry.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert with_retry(flaky, attempts=3, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_with_retry_reraises_after_exhaustion(monkeypatch):
    monkeypatch.setattr("hymem.extraction.retry.time.sleep", lambda _s: None)

    def always_fail():
        raise ValueError("down")

    with pytest.raises(ValueError, match="down"):
        with_retry(always_fail, attempts=3, base_delay=0)
