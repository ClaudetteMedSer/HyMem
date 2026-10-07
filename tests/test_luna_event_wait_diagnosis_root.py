"""Offline falsification controls for b06p09ba; never start a real process.

These invented wire sequences exercise the frozen parser/router. They are not
replays of private run events and cannot establish that run's upstream cause.
"""
from collections import deque
import io
import json
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_timeout_v3 as observed


PRIVATE = "INVENTED-TEXT-NOT-FOR-DIAGNOSTICS"
THREAD = "invented-active-thread"
TURN = "invented-turn"
RETIRED = "invented-retired-thread"


def event(method, **fields):
    return {"method": method, "params": {
        "threadId": THREAD, "turnId": TURN, **fields}}


def start():
    return (0.8, {"id": 1, "result": {
        "turn": {"id": TURN, "status": "inProgress"}}})


def prefix():
    return [
        start(),
        (1.0, event("thread/status/changed", status={"type": "active"})),
        (1.2, event("turn/started", turn={"id": TURN, "status": "inProgress"})),
        (1.5, event("item/started", item={"id": "user", "type": "userMessage"})),
        (2.0, event("item/completed", item={"id": "user", "type": "userMessage"})),
        (3.0, event("item/started", item={"id": "reason", "type": "reasoning"})),
        (4.08, event("item/started", item={"id": "answer", "type": "agentMessage"})),
    ]


def finish():
    return [
        (79.0, event("item/completed", item={"id": "reason", "type": "reasoning"})),
        (80.0, event("item/completed", item={"id": "answer", "type": "agentMessage",
            "phase": "final_answer", "text": PRIVATE})),
        (81.0, event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 19}})),
        (82.0, event("turn/completed", turn={"id": TURN, "status": "completed"})),
    ]


def retired_status():
    return (59.45, {"method": "thread/status/changed", "params": {
        "threadId": RETIRED, "status": {"type": "notLoaded"}}})


@pytest.fixture
def wire(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(observed.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(observed.base.subprocess, "run", lambda *a, **k:
        SimpleNamespace(stdout="codex-cli 0.158.0"))
    monkeypatch.setattr(observed.base.subprocess, "Popen", lambda *a, **k:
        SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None))
    monkeypatch.setattr(observed.base.threading, "Thread", lambda *a, **k:
        SimpleNamespace(start=lambda: None))
    original_get = observed._TimedQueue.get

    def build(rows):
        session = observed.TimeoutSession("unused", ".")
        session.active_thread = THREAD
        session.retired_threads.add(RETIRED)
        session._reader_state = "running"
        session.set_deadline(1120.0)
        pending = deque(rows)

        def scheduled_get(queue, block=True, timeout=None):
            if not queue.empty():
                return original_get(queue, block=False)
            if pending and 1000.0 + pending[0][0] <= now[0] + timeout:
                at, value = pending.popleft()
                now[0] = 1000.0 + at
                queue.put_nowait(value)
                return original_get(queue, block=False)
            now[0] += timeout
            raise observed.queue.Empty

        monkeypatch.setattr(observed._TimedQueue, "get", scheduled_get)
        return session

    return build


def run(session):
    return observed.base._run_turn(session, THREAD, PRIVATE)


def test_retired_traffic_updates_raw_clock_without_turn_progress(wire):
    session = wire(prefix() + [retired_status()])
    with pytest.raises(observed.base.SubscriptionTransportError, match="^timeout$"):
        run(session)
    snapshot = session.timeout_snapshot()
    assert snapshot["reader_enqueued"] == snapshot["reader_dequeued"] == 8
    assert snapshot["events_returned"] == 6
    assert snapshot["last_dequeue_monotonic"] == pytest.approx(1059.45)
    assert snapshot["last_event_consumed_monotonic"] == pytest.approx(1004.08)
    assert snapshot["queue_depth"] == 0 and snapshot["oldest_enqueue_monotonic"] is None
    assert snapshot["reader_state"] == "running" and snapshot["process_alive"] is True
    assert snapshot["observed"]["last_event_family"] == "item_started"
    assert snapshot["observed"]["completed_seen"] is False
    assert snapshot["observed"]["final_seen"] is False
    assert snapshot["observed"]["usage_state"] == "absent"
    assert PRIVATE not in json.dumps(snapshot)


def test_current_turn_completion_survives_retired_traffic(wire):
    session = wire(prefix() + [retired_status()] + finish())
    assert run(session) == (PRIVATE, 19)
    snapshot = session.timeout_snapshot()
    assert snapshot["events_returned"] == 10
    assert snapshot["observed"]["completed_seen"] is True
    assert snapshot["observed"]["final_seen"] is True
    assert snapshot["observed"]["usage_state"] == "positive"


def test_early_completion_is_preserved_while_start_rpc_is_pending(wire):
    rows = prefix()[1:] + finish()
    rows.append((83.0, start()[1]))
    session = wire(rows)
    assert run(session) == (PRIVATE, 19)
    assert session.timeout_snapshot()["observed"]["completed_seen"] is True


@pytest.mark.parametrize("retry", [False, True])
def test_error_is_not_silently_filtered(wire, retry):
    failure = event("error", willRetry=retry, error={
        "message": PRIVATE,
        "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 503}},
        "additionalDetails": None})
    session = wire(prefix() + [(59.45, failure)] + finish())
    if retry:
        assert run(session) == (PRIVATE, 19)
        assert session.retry_progress_count == 1
    else:
        with pytest.raises(observed.base.SubscriptionTransportError,
                           match="^unexpected_notification:error$"):
            run(session)
        assert session.timeout_snapshot()["observed"]["last_event_family"] == "error"


def test_unexpected_opted_out_delta_fails_immediately(wire):
    session = wire(prefix() + [(59.45, event("item/agentMessage/delta",
        itemId="answer", delta=PRIVATE))] + finish())
    with pytest.raises(observed.base.SubscriptionTransportError,
                       match="^notification_optout_unverified$"):
        run(session)


@pytest.mark.parametrize("missing", ["item/completed", "thread/tokenUsage/updated"])
def test_completion_without_required_output_or_usage_is_not_a_timeout(wire, missing):
    session = wire(prefix() + [row for row in finish() if row[1]["method"] != missing])
    with pytest.raises(observed.base.SubscriptionTransportError,
                       match="^incomplete_turn_or_usage$"):
        run(session)


def test_failed_completion_is_not_a_timeout(wire):
    session = wire(prefix() + [(59.45, event("turn/completed",
        turn={"id": TURN, "status": "failed"}))])
    with pytest.raises(observed.base.SubscriptionTransportError, match="^turn_failed$"):
        run(session)


@pytest.mark.parametrize("method,fields", [
    ("turn/completed", {"turn": {"id": TURN, "status": "completed"}}),
    ("item/completed", {"item": {"id": "answer", "type": "agentMessage",
        "phase": "final_answer", "text": PRIVATE}}),
    ("thread/tokenUsage/updated", {"tokenUsage": {"total": {"totalTokens": 19}}}),
])
def test_wrong_thread_semantic_events_fail_instead_of_disappearing(wire, method, fields):
    value = event(method, **fields)
    value["params"]["threadId"] = RETIRED
    session = wire(prefix() + [(59.45, value)] + finish())
    with pytest.raises(observed.base.SubscriptionTransportError) as raised:
        run(session)
    assert str(raised.value) != "timeout"


def test_completion_after_original_deadline_is_not_accepted(wire):
    session = wire(prefix() + [(at + 50, value) for at, value in finish()])
    with pytest.raises(observed.base.SubscriptionTransportError, match="^timeout$"):
        run(session)
    assert session.deadline == 1120.0
    assert session.timeout_snapshot()["observed"]["usage_state"] == "absent"


def test_malformed_stdout_is_distinguishable_from_running_empty_queue(wire):
    session = wire([])
    session.process.stdout = io.StringIO("invalid-json\n")
    session._read()
    assert session.timeout_snapshot()["reader_state"] == "stopped"
    with pytest.raises(observed.base.SubscriptionTransportError, match="protocol_failure"):
        session.receive()
