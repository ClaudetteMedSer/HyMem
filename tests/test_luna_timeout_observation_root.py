"""Root-owned timeout/privacy regressions; no process or provider is started."""
from copy import deepcopy
import io
import json
import threading
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_timeout_v1 as observed


PRIVATE = "PRIVATE-INVENTED-CONTENT"


def fake_process(*args, **kwargs):
    return SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None)


class NoThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


def client(monkeypatch, now):
    monkeypatch.setattr(observed.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(observed.base.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="codex-cli 0.158.0"))
    monkeypatch.setattr(observed.base.subprocess, "Popen", fake_process)
    monkeypatch.setattr(observed.base.threading, "Thread", NoThread)
    monkeypatch.setattr(observed.TimeoutSession, "close", lambda self: None)
    budget = observed.SharedBudget(observed.BudgetLimits(16, 160000, 600),
                                   max_in_flight=4, clock=lambda: now[0])
    result = observed.TimeoutSubscriptionClient("unused", budget, "worker-0",
                                                observed.BudgetLimits(4, 160000, 600))
    return result


def admission(session, **kwargs):
    session.active_thread = "private-thread"
    return {"auth": "chatgpt", "model": "gpt-6-luna", "inference_enabled": False,
            "config_isolation_admitted": True, "_thread_id": "private-thread",
            "quota_windows": [{"remaining_percent": 80}]}


def request():
    return SimpleNamespace(system=PRIVATE, user=PRIVATE, response_format="text",
                           max_tokens=64, temperature=0.0)


def wire_events():
    def event(method, **fields):
        return {"method": method, "params": {"threadId": "private-thread",
                  "turnId": "private-turn", **fields}}
    return [
        {"id": 1, "result": {"turn": {"id": "private-turn", "status": "inProgress"}}},
        event("item/started", item={"id": "private-item", "type": "agentMessage"}),
        event("item/completed", item={"id": "private-item", "type": "agentMessage",
                                       "phase": "final_answer", "text": PRIVATE}),
        event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 19}}),
        event("turn/completed", turn={"id": "private-turn", "status": "completed"}),
        {"id": 2, "result": {"status": "unsubscribed"}},
    ]


@pytest.mark.parametrize("missing_usage", [False, True])
def test_root_actual_parser_keeps_completion_and_usage_contract(monkeypatch, missing_usage):
    now = [9_000_000.0]
    item = client(monkeypatch, now)
    def ready(session, **kwargs):
        result = admission(session, **kwargs)
        for event in wire_events():
            if not (missing_usage and event.get("method") == "thread/tokenUsage/updated"):
                session.events.put(event)
        return result
    monkeypatch.setattr(observed.base, "inspect_preflight", ready)
    if missing_usage:
        with pytest.raises(observed.ConcurrentStop, match="incomplete_turn_or_usage"):
            item.complete(request())
    else:
        assert item.complete(request()) == PRIVATE
    record, = item.diagnostic_records()
    assert record["observed"]["final_seen"] is True
    assert record["observed"]["completed_seen"] is True
    assert record["observed"]["usage_state"] == ("absent" if missing_usage else "positive")
    assert record["known_usage"] is not missing_usage
    assert item.budget.snapshot()["known_tokens"] == (0 if missing_usage else 19)
    assert PRIVATE not in json.dumps(record)
    item.close()


@pytest.mark.parametrize("after_final", [False, True])
def test_root_expiry_records_partial_state_without_consuming_late_queue(monkeypatch, after_final):
    now = [9_000_000.0]
    item = client(monkeypatch, now)
    def ready(session, **kwargs):
        result = admission(session, **kwargs)
        for event in wire_events():
            session.events.put(event)
        return result
    original = observed.TimeoutSession.next_event
    delayed = [False]
    def next_event(session):
        if not after_final and not delayed[0]:
            now[0] += 121
            delayed[0] = True
        event = original(session)
        if after_final and event.get("method") == "item/completed":
            now[0] += 121
        return event
    monkeypatch.setattr(observed.base, "inspect_preflight", ready)
    monkeypatch.setattr(observed.TimeoutSession, "next_event", next_event)
    with pytest.raises(observed.ConcurrentStop, match="timeout"):
        item.complete(request())
    record, = item.diagnostic_records()
    assert record["failure_code"] == "timeout"
    assert record["observed"]["final_seen"] is after_final
    assert record["observed"]["completed_seen"] is False
    assert record["observed"]["usage_state"] == "absent"
    assert record["known_usage"] is False
    assert record["precleanup"]["queue_depth"] > 0
    assert record["precleanup"]["oldest_enqueue_monotonic"] < record["deadline_monotonic"]
    assert item.budget.snapshot()["usage_complete"] is False
    item.close()


def test_root_observer_preserves_frozen_pending_queue_semantics(monkeypatch):
    # The frozen router can drain already-pending notices without a receive().
    # This diagnostic must describe that behavior, not silently repair it.
    now = [9_000_000.0]
    item = client(monkeypatch, now)
    session = observed.TimeoutSession("unused", ".")
    session.active_thread = "private-thread"
    session.deadline = now[0] - 1
    event = wire_events()[1]
    session.pending.append(deepcopy(event))
    frozen_result = observed.warm.WarmSession.next_event(session)
    session.pending.append(deepcopy(event))
    diagnostic_result = session.next_event()
    assert diagnostic_result == frozen_result == event
    assert session.deadline == now[0] - 1
    item.close()


def test_root_reader_queue_fifo_and_snapshots_do_not_deadlock():
    q = observed._TimedQueue(16)
    consumed = []
    def produce():
        for index in range(1000):
            q.put(index, timeout=2)
    def consume():
        for _ in range(1000):
            consumed.append(q.get(timeout=2))
    workers = [threading.Thread(target=target, daemon=True) for target in (produce, consume)]
    for worker in workers:
        worker.start()
    for _ in range(100):
        snapshot = q.diagnostic_snapshot()
        assert 0 <= snapshot["queue_depth"] <= 16
        assert snapshot["reader_dequeued"] <= snapshot["reader_enqueued"]
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()
    assert consumed == list(range(1000))
    assert q.diagnostic_snapshot()["queue_depth"] == 0


@pytest.mark.parametrize("malformed", [False, True])
def test_root_actual_reader_preserves_payload_and_marks_only_known_state(monkeypatch, malformed):
    now = [9_000_000.0]
    item = client(monkeypatch, now)
    session = observed.TimeoutSession("unused", ".")
    session.process.stdout = io.StringIO("not-json\n" if malformed else
                                        json.dumps({"opaque": PRIVATE}) + "\n")
    session._read()
    assert session.timeout_snapshot()["reader_state"] == "stopped"
    if not malformed:
        assert session.events.get_nowait() == {"opaque": PRIVATE}
    assert session.events.get_nowait() is None
    # stopped deliberately does not claim EOF vs decoder failure distinction.
    assert PRIVATE not in json.dumps(session.timeout_snapshot())
    item.close()


def successful_record(monkeypatch):
    now = [9_000_000.0]  # Server uptime is not a one-million-second duration.
    item = client(monkeypatch, now)
    monkeypatch.setattr(observed.base, "inspect_preflight", admission)
    monkeypatch.setattr(observed.base, "_run_turn", lambda *args: (PRIVATE, 19))
    monkeypatch.setattr(observed.TimeoutSession, "unsubscribe", lambda *args: None)
    assert item.complete(request()) == PRIVATE
    record, = item.diagnostic_records()
    item.close()
    return record


def test_root_long_server_uptime_does_not_make_valid_call_fail(monkeypatch):
    record = successful_record(monkeypatch)
    assert record["known_usage"] is True and record["status"] == "success"
    assert PRIVATE not in json.dumps(record)


def test_root_startup_failure_keeps_unknown_observations(monkeypatch):
    now = [9_000_000.0]
    item = client(monkeypatch, now)
    def unavailable(*args, **kwargs):
        raise OSError(PRIVATE)
    monkeypatch.setattr(observed.base.subprocess, "Popen", unavailable)
    with pytest.raises(observed.ConcurrentStop):
        item.complete(request())
    record, = item.diagnostic_records()
    assert record["failure_code"] == "startup_failed"
    assert record["reader_state"] is None
    assert record["observed"] is None
    assert PRIVATE not in json.dumps(record)
    assert item.budget.snapshot()["turns"] == 0
    item.close()


def test_root_first_fault_survives_secondary_cleanup_failure(monkeypatch):
    now = [9_000_000.0]
    item = client(monkeypatch, now)
    monkeypatch.setattr(observed.base, "inspect_preflight", admission)
    def timeout(*args):
        now[0] += 120
        observed.base._fail("timeout")
    def bad_cleanup(*args):
        raise RuntimeError(PRIVATE)
    monkeypatch.setattr(observed.base, "_run_turn", timeout)
    monkeypatch.setattr(observed.TimeoutSession, "close", bad_cleanup)
    with pytest.raises(observed.ConcurrentStop):
        item.complete(request())
    record, = item.diagnostic_records()
    assert record["failure_code"] == "timeout"
    assert record["known_usage"] is False
    assert item.budget.snapshot()["first_failure"]["code"] == "timeout"
    monkeypatch.setattr(observed.TimeoutSession, "close", lambda self: None)
    item.close()


def test_root_preflight_consumes_existing_deadline_without_extension(monkeypatch):
    now = [9_000_000.0]
    item = client(monkeypatch, now)
    def slow_preflight(session, **kwargs):
        now[0] += 100
        return admission(session, **kwargs)
    def run_turn(session, *args):
        assert session.deadline == 9_000_120.0
        assert session.deadline - now[0] == 20
        now[0] += 21
        observed.base._fail("timeout")
    monkeypatch.setattr(observed.base, "inspect_preflight", slow_preflight)
    monkeypatch.setattr(observed.base, "_run_turn", run_turn)
    with pytest.raises(observed.ConcurrentStop):
        item.complete(request())
    record, = item.diagnostic_records()
    assert record["failure_code"] == "timeout"
    assert record["known_usage"] is False
    assert item.budget.snapshot()["usage_complete"] is False
    item.close()


def test_root_cumulative_queue_traffic_across_warm_calls_is_not_event_cap(monkeypatch):
    now = [9_000_000.0]
    item = client(monkeypatch, now)
    monkeypatch.setattr(observed.base, "inspect_preflight", admission)
    def call(session, *args):
        for _ in range(2300):
            session.events.put({"invented": PRIVATE})
            assert session.events.get()["invented"] == PRIVATE
        return PRIVATE, 19
    monkeypatch.setattr(observed.base, "_run_turn", call)
    monkeypatch.setattr(observed.TimeoutSession, "unsubscribe", lambda *args: None)
    assert item.complete(request()) == PRIVATE
    assert item.complete(request()) == PRIVATE
    records = item.diagnostic_records()
    assert len(records) == 2 and all(r["status"] == "success" for r in records)
    assert PRIVATE not in json.dumps(records)
    item.close()


@pytest.mark.parametrize("malformed", [[], {}, PRIVATE, True, -1, float("nan"), float("inf")])
def test_root_projection_total_for_all_top_level_json_fields(monkeypatch, malformed):
    record = successful_record(monkeypatch)
    for field in record:
        invalid = deepcopy(record)
        invalid[field] = malformed
        result = observed._project_record(invalid)
        assert PRIVATE not in json.dumps(result)


def test_root_projection_rejects_unknown_prose_and_returns_detached_data(monkeypatch):
    record = successful_record(monkeypatch)
    record["unexpected"] = PRIVATE
    assert observed._project_record(record) is None
    record.pop("unexpected")
    one = observed._project_record(record)
    two = observed._project_record(record)
    one["phase_seconds"]["preflight"] = 999
    assert two["phase_seconds"]["preflight"] != 999
