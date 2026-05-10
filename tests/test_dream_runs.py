from __future__ import annotations

import pytest

from hymem.extraction.llm import LLMRequest, StubLLMClient
from tests.conftest import make_routed_llm


def _seed_session(hy) -> str:
    sid = "s1"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "I'll set up Docker for the local dev environment.")
    hy.log_message(
        sid,
        "user",
        "No, actually we don't use Docker for local dev anymore. We switched to uv and system Python.",
    )
    hy.close_session(sid)
    return sid


def test_dream_persists_run_report(hy):
    _seed_session(hy)
    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
    ]
    markers = [{"kind": "preference", "statement": "user prefers uv"}]
    hy.set_llm(make_routed_llm(triples, markers))

    report = hy.dream()

    rows = hy.conn.execute(
        "SELECT * FROM dream_runs ORDER BY id DESC"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["ended_at"] is not None
    assert row["error"] is None
    assert row["skipped_locked"] == 0
    assert row["sessions_processed"] == report.sessions_processed
    assert row["chunks_seen"] == report.chunks_seen
    assert row["chunks_processed"] == report.chunks_processed
    assert row["triples_extracted"] == report.triples_extracted
    assert row["markers_extracted"] == report.markers_extracted


def test_dream_run_skipped_records_lock_skip(hy):
    _seed_session(hy)
    hy.conn.execute(
        "INSERT INTO run_lock(name, acquired_at, holder) "
        "VALUES ('dreaming', CURRENT_TIMESTAMP, 'other_proc')"
    )

    report = hy.dream()
    assert report.skipped_locked is True

    row = hy.conn.execute(
        "SELECT * FROM dream_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["skipped_locked"] == 1
    assert row["ended_at"] is not None


def test_recent_dream_runs_returns_dicts(hy):
    _seed_session(hy)
    hy.dream()
    hy.dream()

    rows = hy.recent_dream_runs(limit=5)
    assert isinstance(rows, list)
    assert len(rows) >= 2
    assert all(isinstance(r, dict) for r in rows)
    assert rows[0]["id"] > rows[1]["id"]
    expected_keys = {
        "id", "started_at", "ended_at",
        "sessions_processed", "chunks_seen", "chunks_processed",
        "chunks_embedded", "triples_extracted", "markers_extracted",
        "skipped_locked", "error",
    }
    assert expected_keys.issubset(rows[0].keys())


class _RaisingLLM:
    def __init__(self, message: str = "boom_llm_failure") -> None:
        self.message = message

    def complete(self, request: LLMRequest) -> str:
        raise RuntimeError(self.message)


def test_dream_records_error(hy, monkeypatch):
    _seed_session(hy)

    # Force an exception outside the per-chunk try/except so it propagates.
    from hymem.dreaming import phase2

    def _boom(*a, **kw):
        raise RuntimeError("boom_phase2_failure")

    monkeypatch.setattr(phase2, "consolidate_profile", _boom)

    with pytest.raises(RuntimeError, match="boom_phase2_failure"):
        hy.dream()

    row = hy.conn.execute(
        "SELECT * FROM dream_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["error"] is not None
    assert "boom_phase2_failure" in row["error"]
    assert row["ended_at"] is not None
