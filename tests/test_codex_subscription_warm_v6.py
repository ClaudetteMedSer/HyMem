"""Offline controls for bounded private failed-turn evidence."""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from benchmarks import codex_subscription_warm_v6 as warm


THREAD = "thread-a"
TURN = "turn-a"
SECRET = "private@example.com /private/prompt"


def event(method, **fields):
    return {"method": method, "params": {"threadId": THREAD, "turnId": TURN, **fields}}


def error(message=SECRET, *, retry=True, kind=None, details=None):
    return event("error", willRetry=retry, error={
        "message": message, "additionalDetails": details, "misalignment": None,
        "codexErrorInfo": kind if kind is not None else
            {"responseStreamDisconnected": {"httpStatusCode": 403}}})


START = event("item/started", item={"id": "message-a", "type": "agentMessage"})
FINAL = event("item/completed", item={"id": "message-a", "type": "agentMessage",
                                  "phase": "final_answer", "text": SECRET})
USAGE = event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 42}})
DONE = event("turn/completed", turn={"id": TURN, "status": "completed"})


def make_session(monkeypatch, events):
    session = object.__new__(warm.WarmSession)
    session.pending = deque(events)
    session.active_thread = THREAD
    session.active_turn = None
    session.stage = "turn/start"
    session.retired_threads = set()
    session.last_event = None
    session.rpc_error = None
    session.turn_observation = None
    session._observation_thread = None
    session._observation_turn = None
    session.created_at = time.monotonic()
    session.starting_events = []
    session.closed = False
    session.set_deadline = lambda deadline: None
    session.reset_private_errors()
    session.unsubscribe = lambda thread: None
    session.close = lambda: setattr(session, "closed", True)
    def rpc(self, method, params, *, preserve_notifications=False):
        assert method == "turn/start" and params["threadId"] == THREAD
        return {"turn": {"id": TURN, "status": "inProgress"}}
    monkeypatch.setattr(warm.v5.v4.v3.v2.WarmSession, "rpc", rpc)
    return session


def make_client(monkeypatch, session, sink=None):
    monkeypatch.setattr(warm.base, "inspect_preflight", lambda *args, **kwargs: {
        "auth": "chatgpt", "model": warm.base.MODEL,
        "config_isolation_admitted": True, "inference_enabled": False,
        "quota_windows": [{"remaining_percent": 75}], "_thread_id": THREAD})
    budget = warm.SharedBudget(warm.BudgetLimits(100, 100000, 1000))
    client = warm.WarmSubscriptionClient("unused", budget, "q",
        warm.BudgetLimits(100, 100000, 1000),
        session_factory=lambda *args, **kwargs: session, private_failure_sink=sink)
    return client, budget


REQUEST = SimpleNamespace(system="system", user=SECRET, temperature=0,
                          max_tokens=32, response_format="json")


class CaptureSink:
    def __init__(self, fail=False):
        self.records = []
        self.fail = fail
    def write(self, record):
        if self.fail:
            raise OSError("private path from sink")
        self.records.append(record)


def test_actual_client_retry_success_discards_evidence(monkeypatch):
    session = make_session(monkeypatch, [error(details=SECRET), START, FINAL, USAGE, DONE])
    sink = CaptureSink()
    client, budget = make_client(monkeypatch, session, sink)
    assert client.complete(REQUEST) == SECRET
    assert sink.records == []
    assert session.private_error_count == 0 and session.private_error_first is None
    assert budget.snapshot()["questions"]["q"]["known_tokens"] == 42


def test_actual_client_terminal_retry_persists_before_cleanup(monkeypatch):
    session = make_session(monkeypatch, [error("first", details=SECRET), error("last", retry=False)])
    sink = CaptureSink()
    client, budget = make_client(monkeypatch, session, sink)
    original_close = session.close
    def close():
        assert len(sink.records) == 1
        original_close()
    session.close = close
    with pytest.raises(warm.ConcurrentStop, match="fixed_other"):
        client.complete(REQUEST)
    assert session.closed and session.private_error_count == 0
    record = sink.records[0]
    assert record["error_count"] == 2
    assert record["first"]["message"] == "first"
    assert record["last"]["message"] == "last"
    assert record["first"]["will_retry"] is True
    assert record["last"]["will_retry"] is False
    assert record["first"]["http_status_code"] == 403
    assert budget.snapshot()["questions"]["q"]["usage_complete"] is False
    assert SECRET not in json.dumps(warm.serialize_failure(budget.snapshot()["first_failure"]))


def test_terminal_auth_error_is_captured(monkeypatch):
    session = make_session(monkeypatch, [error("auth denied", retry=False, kind="unauthorized")])
    sink = CaptureSink()
    client, _ = make_client(monkeypatch, session, sink)
    with pytest.raises(warm.ConcurrentStop):
        client.complete(REQUEST)
    assert sink.records[0]["first"]["error_class"] == "unauthorized"
    assert sink.records[0]["first"]["http_status_code"] is None


def test_documented_optional_error_fields_and_misalignment_are_finite():
    notice = event("error", willRetry=False, error={"message": SECRET,
        "misalignment": {"errorType": SECRET, "detailedExplanation": SECRET,
                         "steer": {"message": SECRET}}})
    captured = warm._private_error(notice, THREAD, TURN)
    assert captured["error_class"] == "unspecified"
    assert captured["misalignment_present"] is True
    assert json.dumps(captured).count(SECRET) == 1
    notice["params"]["error"]["misalignment"]["steer"] = {"message": 5}
    assert warm._private_error(notice, THREAD, TURN) is None


def test_malformed_and_foreign_errors_are_ignored_by_private_observer(monkeypatch):
    malformed = error()
    malformed["params"]["willRetry"] = "true"
    foreign = error()
    foreign["params"]["turnId"] = "other"
    for notice in (malformed, foreign):
        session = make_session(monkeypatch, [notice])
        try:
            warm.base._run_turn(session, THREAD, SECRET)
        except warm.base.SubscriptionTransportError:
            pass
        assert session.private_error_count == 0


def test_reset_and_unicode_byte_bounds(monkeypatch):
    session = make_session(monkeypatch, [error("é" * 3000, retry=False, details="🙂" * 2000)])
    with pytest.raises(warm.base.SubscriptionTransportError):
        warm.base._run_turn(session, THREAD, SECRET)
    item = session.private_error_first
    assert len(item["message"].encode()) <= warm.MAX_PRIVATE_TEXT_BYTES
    assert len(item["additional_details"].encode()) <= warm.MAX_PRIVATE_TEXT_BYTES
    assert item["message_truncated"] and item["additional_details_truncated"]
    session.pending = deque([START, FINAL, USAGE, DONE])
    assert warm.base._run_turn(session, THREAD, SECRET) == (SECRET, 42)
    assert session.private_error_count == 0


def test_sink_failure_keeps_original_accounting_and_no_public_path(monkeypatch):
    session = make_session(monkeypatch, [error(retry=False)])
    client, budget = make_client(monkeypatch, session, CaptureSink(fail=True))
    with pytest.raises(warm.ConcurrentStop, match="fixed_other"):
        client.complete(REQUEST)
    assert client.private_sink_status == "failed"
    first = budget.snapshot()["first_failure"]
    assert first["code"] == "fixed_other"
    assert first["known_usage"] is False
    public = json.dumps(warm.serialize_failure(first))
    assert SECRET not in public and "private path from sink" not in public


def _record():
    item = warm._private_error(error("message", retry=False), THREAD, TURN)
    return {"schema": "warm_private_failure_v1", "failure_code": "unexpected_notification:error",
            "error_count": 1, "first": item, "last": item}


def test_private_sink_exclusive_slots_cap_and_mode(tmp_path):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    sink = warm.PrivateFailureSink(directory)
    names = [sink.write(_record()) for _ in range(warm.MAX_PRIVATE_RECORDS)]
    assert len(set(names)) == warm.MAX_PRIVATE_RECORDS
    assert all((directory / name).stat().st_mode & 0o777 == 0o600 for name in names)
    with pytest.raises(ValueError, match="private_record_cap"):
        sink.write(_record())
    assert len(list(directory.iterdir())) == warm.MAX_PRIVATE_RECORDS


def test_private_sink_rejects_symlink_permissions_and_unsafe_slot(tmp_path):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(OSError):
        warm.PrivateFailureSink(link).write(_record())
    directory.chmod(0o755)
    with pytest.raises(ValueError, match="private_directory_unsafe"):
        warm.PrivateFailureSink(directory).write(_record())
    directory.chmod(0o700)
    (directory / "warm-private-failure-00.json").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="private_slot_unsafe"):
        warm.PrivateFailureSink(directory).write(_record())
    assert not (tmp_path / "target").exists()


def test_sink_rejects_unvalidated_fields_and_large_record(tmp_path):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    sink = warm.PrivateFailureSink(directory)
    bad = _record()
    bad["thread_id"] = THREAD
    with pytest.raises(ValueError, match="private_record_invalid"):
        sink.write(bad)
    bad = _record()
    bad["first"] = {**bad["first"], "message": "x" * 5000}
    with pytest.raises(ValueError, match="private_record_invalid"):
        sink.write(bad)
    assert list(directory.iterdir()) == []
