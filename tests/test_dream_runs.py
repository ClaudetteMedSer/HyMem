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


def test_refresh_lock_advances_acquired_at_for_owner(hy):
    # Seed a lock owned by us, backdated well past the TTL so any movement is
    # unambiguous, then heartbeat it.
    from hymem.dreaming.runner import _refresh_lock

    hy.conn.execute(
        "INSERT INTO run_lock(name, acquired_at, holder) "
        "VALUES ('dreaming', datetime('now', '-1 hour'), 'me')"
    )
    before = hy.conn.execute(
        "SELECT acquired_at FROM run_lock WHERE name = 'dreaming'"
    ).fetchone()["acquired_at"]

    _refresh_lock(hy.conn, "me")

    after = hy.conn.execute(
        "SELECT acquired_at FROM run_lock WHERE name = 'dreaming'"
    ).fetchone()["acquired_at"]
    # CURRENT_TIMESTAMP is strictly newer than an hour ago.
    assert after > before
    # And it is no longer stale relative to the TTL.
    stale = hy.conn.execute(
        "SELECT 1 FROM run_lock WHERE name = 'dreaming' "
        "AND acquired_at < datetime('now', '-120 seconds')"
    ).fetchone()
    assert stale is None


def test_refresh_lock_does_not_touch_other_holders_lock(hy):
    # The holder guard must prevent one process from heartbeating (and thus
    # resurrecting/stealing) a lock another process owns.
    from hymem.dreaming.runner import _refresh_lock

    hy.conn.execute(
        "INSERT INTO run_lock(name, acquired_at, holder) "
        "VALUES ('dreaming', datetime('now', '-1 hour'), 'owner_A')"
    )
    before = hy.conn.execute(
        "SELECT acquired_at FROM run_lock WHERE name = 'dreaming'"
    ).fetchone()["acquired_at"]

    _refresh_lock(hy.conn, "intruder_B")

    after = hy.conn.execute(
        "SELECT acquired_at, holder FROM run_lock WHERE name = 'dreaming'"
    ).fetchone()
    assert after["acquired_at"] == before  # untouched
    assert after["holder"] == "owner_A"    # not stolen


def _spy_refresh_lock(monkeypatch):
    """Install a spy over runner._refresh_lock; returns the list of holder args
    it was called with."""
    import hymem.dreaming.runner as runner_mod

    calls: list[str] = []
    real_refresh = runner_mod._refresh_lock

    def _spy(conn, holder):
        calls.append(holder)
        return real_refresh(conn, holder)

    monkeypatch.setattr(runner_mod, "_refresh_lock", _spy)
    return calls


def test_dream_heartbeats_lease_at_least_once(hy, monkeypatch):
    # A live dream must refresh the lease so a slow run never looks stale. The
    # heartbeat is throttled (default interval), so a fast test dream fires it
    # at least once — enough to keep acquired_at fresh.
    _seed_session(hy)
    hy.set_llm(make_routed_llm(
        [{"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1}],
        [],
    ))

    calls = _spy_refresh_lock(monkeypatch)
    report = hy.dream()

    assert report.sessions_processed >= 1
    assert len(calls) >= 1


def test_dream_heartbeats_within_a_session_when_interval_elapses(hy, monkeypatch):
    # The edge case the throttle guards: a single heavy session must heartbeat
    # *during* its chunk processing, not only at session start. With the
    # interval forced to 0, every per-chunk heartbeat fires — so a session that
    # processes at least one chunk produces strictly more than one refresh
    # (session-top + per-chunk), proving the lease can't age out mid-session.
    import hymem.dreaming.runner as runner_mod

    monkeypatch.setattr(runner_mod, "_LOCK_REFRESH_INTERVAL_SECONDS", 0)
    _seed_session(hy)
    hy.set_llm(make_routed_llm(
        [{"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1}],
        [],
    ))

    calls = _spy_refresh_lock(monkeypatch)
    report = hy.dream()

    assert report.chunks_processed >= 1
    # Session-top heartbeat + at least one per-chunk heartbeat.
    assert len(calls) >= 2


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


def test_dream_status_before_any_dream(hy):
    # Fresh DB: no chunks yet, no dream has run, no lock held.
    status = hy.dream_status()
    assert status["pending_chunks"] == 0
    assert status["total_chunks"] == 0
    assert status["prompt_version"] == hy.config.prompt_version
    assert status["in_progress"] is False
    assert status["last_run"] is None


def test_dream_status_counts_and_last_run(hy):
    _seed_session(hy)
    triples = [
        {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1},
    ]
    hy.set_llm(make_routed_llm(triples, []))

    hy.dream()

    after = hy.dream_status()
    # The dream created chunks and processed them all for the current version.
    assert after["total_chunks"] > 0
    assert after["pending_chunks"] == 0
    assert after["prompt_version"] == hy.config.prompt_version
    assert after["in_progress"] is False
    # last_run is populated and reflects the completed cycle.
    assert after["last_run"] is not None
    assert after["last_run"]["ended_at"] is not None
    assert after["last_run"]["error"] is None

    # Seed MORE chunks without dreaming → they have no processed_chunks row for
    # the current prompt_version, so they count as pending.
    hy.conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES ('extra-1', 's1', 1, 2, 'test', 'extra chunk')"
    )
    hy.conn.commit()
    bumped = hy.dream_status()
    assert bumped["total_chunks"] == after["total_chunks"] + 1
    assert bumped["pending_chunks"] == 1


def test_dream_status_pending_drops_after_dream(hy):
    _seed_session(hy)
    hy.set_llm(make_routed_llm([], []))

    # A second session whose chunks are created (and processed) by dreaming.
    sid = "s2"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "We deploy the api to fly_io for staging.")
    hy.log_message(sid, "user", "Yes, staging runs on fly_io. Keep it that way.")
    hy.close_session(sid)

    hy.dream()
    # Every chunk dreaming creates is processed for the current prompt_version.
    status = hy.dream_status()
    assert status["total_chunks"] > 0
    assert status["pending_chunks"] == 0

    # Simulate a prompt_version bump: a HyMem on the same DB but a newer
    # prompt_version sees the whole backlog as pending again (the surge this
    # status surface is meant to make transparent).
    import dataclasses

    from hymem.api import HyMem

    bumped_cfg = dataclasses.replace(
        hy.config, prompt_version=hy.config.prompt_version + "-next"
    )
    hy2 = HyMem(bumped_cfg, llm=hy._llm, embedding_client=hy._embed)
    try:
        bumped = hy2.dream_status()
        assert bumped["prompt_version"] == bumped_cfg.prompt_version
        assert bumped["pending_chunks"] == bumped["total_chunks"]
        assert bumped["pending_chunks"] > 0
    finally:
        hy2.close()


def test_dream_status_in_progress_reflects_lock(hy):
    _seed_session(hy)
    assert hy.dream_status()["in_progress"] is False

    hy.conn.execute(
        "INSERT INTO run_lock(name, acquired_at, holder) "
        "VALUES ('dreaming', CURRENT_TIMESTAMP, 'other_proc')"
    )
    hy.conn.commit()
    assert hy.dream_status()["in_progress"] is True


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
