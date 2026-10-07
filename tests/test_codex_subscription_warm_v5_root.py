"""Independent root controls using the actual RPC queue and parser lineage."""
from collections import deque
import copy
import json
import queue
import time

import pytest

from benchmarks import codex_subscription_warm_v5 as candidate
from benchmarks import codex_subscription_warm_v4 as frozen


SECRET = "SYNTHETIC-PRIVATE-NOT-FOR-METADATA"


def error():
    return {"method": "error", "params": {"threadId": "thread", "turnId": "turn",
        "willRetry": True, "error": {"message": SECRET, "additionalDetails": None,
        "misalignment": None,
        "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 403}}}}}


def event(method, **data):
    return {"method": method, "params": {"threadId": "thread", "turnId": "turn", **data}}


def sequence():
    return [event("item/started", item={"id": "item", "type": "agentMessage"}),
        event("item/completed", item={"id": "item", "type": "agentMessage",
                                       "phase": "final_answer", "text": SECRET}),
        event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 19}}),
        event("turn/completed", turn={"id": "turn", "status": "completed"})]


def session(module, monkeypatch, events, early=None):
    s = object.__new__(module.WarmSession)
    s.created_at = time.monotonic()
    s.deadline = s.created_at + 30
    s.next_id = 0
    s.events = queue.Queue()
    s.pending = deque()
    s.stage = "startup"
    s.active_thread = "thread"
    s.active_turn = None
    s.last_event = None
    s.rpc_error = None
    s.request_id = None
    s.retired_threads = set()
    s.starting_events = []
    s.warning_targets = []
    s.initialized_result = None
    s.initialized_sent = True
    s.reset_turn_observation()
    for message in [*(early or []), {"id": 1, "result": {"turn": {
            "id": "turn", "status": "inProgress"}}}, *events]:
        s.events.put(message)
    def send(self, method, params, *, notification=False):
        assert method == "turn/start" and not notification
        self.next_id += 1
        return self.next_id
    monkeypatch.setattr(module.base.StdioSession, "send", send)
    return s


def test_real_queue_proves_old_abort_new_same_turn_recovery(monkeypatch):
    old = session(frozen, monkeypatch, sequence(), early=[error()])
    with pytest.raises(frozen.base.SubscriptionTransportError, match="unexpected_notification:error"):
        frozen.base._run_turn(old, "thread", SECRET)
    new = session(candidate, monkeypatch, sequence(), early=[error()])
    deadline = new.deadline
    assert candidate.base._run_turn(new, "thread", SECRET) == (SECRET, 19)
    assert new.next_id == 1 and new.deadline == deadline
    assert new.retry_progress_count == 1
    assert new.turn_observation["events_consumed"] == 5
    assert new.turn_observation["completed_seen"] is True
    assert new.last_event is None


@pytest.mark.parametrize("kind", ["unauthorized", "usageLimitExceeded", "rateLimitExceeded",
    "cyberPolicy", "misalignmentPolicyViolation", "sessionBudgetExceeded", "contextWindowExceeded"])
def test_explicit_access_and_quota_denials_remain_fatal(monkeypatch, kind):
    notice = error()
    notice["params"]["error"]["codexErrorInfo"] = kind
    s = session(candidate, monkeypatch, [notice, *sequence()])
    with pytest.raises(candidate.base.SubscriptionTransportError, match="unexpected_notification:error"):
        candidate.base._run_turn(s, "thread", SECRET)
    assert s.next_id == 1


@pytest.mark.parametrize("mutation", [
    lambda xs: xs.insert(1, copy.deepcopy(xs[0])),
    lambda xs: xs[0]["params"]["item"].update(type="commandExecution"),
    lambda xs: xs[1]["params"]["item"].update(id="foreign"),
    lambda xs: xs[2]["params"].update(turnId="foreign"),
    lambda xs: xs[2]["params"]["tokenUsage"]["total"].update(totalTokens=True),
    lambda xs: xs[3]["params"]["turn"].update(status="failed"),
    lambda xs: xs.pop(2),
])
def test_progress_never_weakens_lifecycle_tool_usage_or_completion_checks(monkeypatch, mutation):
    values = sequence()
    mutation(values)
    s = session(candidate, monkeypatch, [error(), *values])
    with pytest.raises(candidate.base.SubscriptionTransportError):
        candidate.base._run_turn(s, "thread", SECRET)
    assert s.next_id == 1


@pytest.mark.parametrize("field", ["params", "error", "codexErrorInfo", "httpStatusCode"])
@pytest.mark.parametrize("value", [None, True, False, [], {}, 1.5, SECRET, -1, 65536])
def test_new_validator_is_total_for_json_shapes_and_never_exports_text(field, value):
    notice = error()
    if field == "params":
        notice["params"] = value
    elif field == "error":
        notice["params"]["error"] = value
    elif field == "codexErrorInfo":
        notice["params"]["error"][field] = value
    else:
        notice["params"]["error"]["codexErrorInfo"]["responseStreamDisconnected"][field] = value
    result = candidate._validated_retry_progress(notice, "thread", "turn")
    assert SECRET not in json.dumps(result)
    assert result is None or type(result) is int


def test_timeout_does_not_reset_original_deadline(monkeypatch):
    s = session(candidate, monkeypatch, [error()])
    before = s.deadline = time.monotonic() + 0.03
    with pytest.raises(candidate.base.SubscriptionTransportError, match="timeout"):
        candidate.base._run_turn(s, "thread", SECRET)
    assert s.deadline == before and s.next_id == 1
