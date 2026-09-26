"""Independent controls for bounded repair, interruption, and target fencing."""
import pytest

from hymem.deadline import DeadlineExceeded
from tests.test_summary_recovery_v63 import (
    conn, Replies, _seed, _public, _items, _no_lease, _generation, recovery,
)


def test_three_durable_attempts_are_not_reset_by_repair_or_invocation(conn):
    _seed(conn)
    llm = Replies({"summary": "z" * 509},
                  {"alternatives": ["z" * 509] * 3},
                  {"alternatives": ["z" * 509] * 3})
    original = _public(conn), _items(conn)
    first = recovery.run_summary_recovery(conn, llm, max_calls=10)
    second = recovery.run_summary_recovery(conn, llm, max_calls=10)
    third = recovery.run_summary_recovery(conn, llm, max_calls=100, max_attempts=100)
    assert (first["calls"], second["calls"], third["calls"]) == (2, 1, 0)
    assert second["exhausted"] == third["exhausted"] == 1
    assert len(llm.calls) == 3
    job = recovery._read_job(conn, "x")
    assert job["attempts"] == job["attempt_limit"] == 3
    assert job["failure_reason"] == "summary_output_cap"
    assert (_public(conn), _items(conn)) == original
    _no_lease(conn)


def test_valid_first_alternative_cannot_hide_invalid_third(conn):
    _seed(conn)
    llm = Replies({"summary": "z" * 509}, {"alternatives": [
        "The source has been retained, but the proposed operation remains pending.",
        "The proposed operation has not run.",
        "",
    ]})
    original = _public(conn), _items(conn)
    result = recovery.run_summary_recovery(conn, llm, max_calls=10)
    assert result["calls"] == 2 and result["published"] == 0
    assert result["held"] == result["remaining"] == 1
    assert (_public(conn), _items(conn)) == original
    assert recovery._read_job(conn, "x")["attempts"] == 2
    _no_lease(conn)


def test_repair_selects_whole_alternative_without_splicing(conn):
    _seed(conn)
    expected = "The proposed operation remains unexecuted; its outcome is unknown."
    llm = Replies({"summary": "z" * 509}, {"alternatives": [
        "z" * 501, expected, "No execution is confirmed.",
    ]})
    result = recovery.run_summary_recovery(conn, llm, max_calls=10)
    assert result["calls"] == 2 and result["published"] == 1
    assert conn.execute("SELECT summary FROM sessions WHERE id='x'").fetchone()[0] == expected
    _no_lease(conn)


def test_late_repair_keeps_known_cap_without_publishing(conn, monkeypatch):
    _seed(conn)
    now = [0.0]
    real = recovery.MonotonicDeadline
    monkeypatch.setattr(recovery, "MonotonicDeadline", type("Controlled", (), {
        "after": staticmethod(lambda seconds: real(seconds, clock=lambda: now[0]))}))
    def late(_request):
        now[0] = 20.0
        return {"summary": "This result arrived after the deadline."}
    llm = Replies({"summary": "z" * 509}, late)
    original = _public(conn), _items(conn)
    with pytest.raises(DeadlineExceeded) as caught:
        recovery.run_summary_recovery(conn, llm, max_calls=10, timeout_seconds=10)
    assert caught.value.summary_recovery_report["calls"] == 2
    assert caught.value.summary_recovery_report["provider_attempts"] == 2
    job = recovery._read_job(conn, "x")
    assert job["attempts"] == 2 and job["failure_reason"] == "summary_output_cap"
    assert (_public(conn), _items(conn)) == original
    _no_lease(conn)


def test_changed_target_after_primary_cannot_authorize_repair(conn):
    _seed(conn)
    original = _public(conn)
    def change(_request):
        conn.execute("UPDATE sessions SET digest_published_generation=?", (_generation("b"),))
        return {"summary": "z" * 509}
    llm = Replies(change)
    with pytest.raises(RuntimeError, match="target changed"):
        recovery.run_summary_recovery(conn, llm, max_calls=10)
    assert len(llm.calls) == 1
    assert _public(conn) == original
    _no_lease(conn)
