"""Offline controls for bounded retry progress on one admitted warm turn."""
from __future__ import annotations

import ast
import copy
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from benchmarks import codex_subscription_warm_v5 as warm


THREAD = "thread-a"
TURN = "turn-a"
SECRET = "private@example.com prompt"


def event(method, **fields):
    return {"method": method, "params": {"threadId": THREAD, "turnId": TURN, **fields}}


def retry(**changes):
    value = event("error", willRetry=True, error={"message": SECRET,
        "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 403}},
        "additionalDetails": SECRET, "misalignment": None})
    value["params"].update(changes)
    return value


START = event("item/started", item={"id": "message-a", "type": "agentMessage"})
FINAL = event("item/completed", item={"id": "message-a", "type": "agentMessage",
                                 "phase": "final_answer", "text": SECRET})
USAGE = event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 42}})
DONE = event("turn/completed", turn={"id": TURN, "status": "completed"})


class Session:
    def __init__(self, values):
        self.values = deque(values)
        self.sent = []
        self.retry_progress_count = None
        self.retry_progress_http_status = None

    def rpc(self, method, params, *, preserve_notifications=False):
        assert method == "turn/start" and preserve_notifications is True
        self.sent.append((method, params))
        return {"turn": {"id": TURN, "status": "inProgress"}}

    def next_event(self):
        if not self.values:
            raise warm.base.SubscriptionTransportError("eof")
        return self.values.popleft()


def run(values):
    session = Session(values)
    try:
        result = warm.base._run_turn(session, THREAD, "synthetic")
        failure = None
    except warm.base.SubscriptionTransportError as exc:
        result = None
        failure = exc.args[0]
    return session, result, failure


def test_pinned_parser_ast_is_identical_except_counter_and_one_branch():
    original = ast.parse(Path(warm.base.__file__).read_bytes())
    original = next(node for node in original.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_run_turn")
    patched = copy.deepcopy(warm._PATCHED_PARSER_AST)
    loop = next(node for node in patched.body if isinstance(node, ast.For))
    branch = loop.body.pop(-2)
    assert isinstance(branch, ast.If) and ast.unparse(branch.test) == "method == 'error'"
    assignments = [node for node in patched.body if isinstance(node, ast.Assign)
                   and ast.unparse(node.targets[0]).startswith(("retry_progress_", "session.retry_progress_"))]
    assert len(assignments) == 3
    patched.body = [node for node in patched.body if node not in assignments]
    assert ast.dump(patched, include_attributes=False) == ast.dump(original, include_attributes=False)


def test_matched_progress_then_final_positive_usage_and_completion():
    s, result, failure = run([retry(), START, FINAL, USAGE, DONE])
    assert failure is None and result == (SECRET, 42)
    assert len(s.sent) == 1 and s.sent[0][0] == "turn/start"
    assert s.retry_progress_count == 1 and s.retry_progress_http_status == 403
    assert SECRET not in json.dumps({"count": s.retry_progress_count,
                                     "status": s.retry_progress_http_status})


@pytest.mark.parametrize("values,code", [
    ([retry(willRetry=False)], "unexpected_notification:error"),
    ([retry(willRetry="true")], "unexpected_notification:error"),
    ([retry(threadId="foreign")], "unexpected_notification:error"),
    ([retry(turnId="foreign")], "unexpected_notification:error"),
    ([retry(error={"message": SECRET, "codexErrorInfo": "unauthorized"})],
     "unexpected_notification:error"),
    ([retry(error={"message": SECRET, "codexErrorInfo": "usageLimitExceeded"})],
     "unexpected_notification:error"),
    ([retry(error={"message": SECRET, "codexErrorInfo":
             {"responseStreamDisconnected": {"httpStatusCode": True}}})],
     "unexpected_notification:error"),
    ([retry(error={"message": SECRET, "codexErrorInfo":
             {"responseStreamDisconnected": {"httpStatusCode": 403}},
             "misalignment": {"reason": SECRET}})], "unexpected_notification:error"),
    ([retry(error={"message": SECRET, "codexErrorInfo":
             {"responseStreamDisconnected": {"httpStatusCode": 403}},
             "code": "policy"})], "unexpected_notification:error"),
    ([retry(error={"message": SECRET, "codexErrorInfo":
             {"responseStreamDisconnected": {"httpStatusCode": 403}},
             "additionalDetails": {"private": SECRET}})], "unexpected_notification:error"),
    ([retry(error={"message": SECRET, "codexErrorInfo":
             {"responseTooManyFailedAttempts": {"httpStatusCode": 403}}})],
     "unexpected_notification:error"),
    ([{"method": "error", "params": []}], "unexpected_notification:error"),
    ([retry(), event("turn/completed", turn={"id": TURN, "status": "failed"})], "turn_failed"),
    ([retry()], "eof"),
])
def test_terminal_malformed_and_eof_remain_failures(values, code):
    s, result, failure = run(values)
    assert result is None and failure == code and len(s.sent) == 1


def test_progress_is_bounded_by_notice_count_and_original_event_ceiling():
    s, result, failure = run([retry()] * (warm.MAX_RETRY_PROGRESS + 1) + [START, FINAL, USAGE, DONE])
    assert result is None and failure == "unexpected_notification:error"
    assert s.retry_progress_count == warm.MAX_RETRY_PROGRESS and len(s.sent) == 1
    s, result, failure = run([retry()] + [event("thread/status/changed")] * (warm.base.MAX_EVENTS - 1)
                             + [START, FINAL, USAGE, DONE])
    assert result is None and failure == "incomplete_turn_or_usage"
    assert len(s.values) == 4 and len(s.sent) == 1


@pytest.mark.parametrize("values,code", [
    ([retry(), START, FINAL, DONE], "incomplete_turn_or_usage"),
    ([retry(), START, FINAL, event("thread/tokenUsage/updated",
        tokenUsage={"total": {"totalTokens": 0}}), DONE], "incomplete_turn_or_usage"),
    ([retry(), START, FINAL, USAGE, event("thread/tokenUsage/updated",
        tokenUsage={"total": {"totalTokens": 1}})], "usage_regressed"),
    ([retry(), START, FINAL, FINAL], "final_message_invalid"),
    ([retry(), START, USAGE, DONE], "incomplete_turn_or_usage"),
])
def test_existing_final_and_usage_gates_remain(values, code):
    s, result, failure = run(values)
    assert result is None and failure == code and len(s.sent) == 1


def test_finite_status_validation_and_deadline_propagation():
    for detail in ({}, {"httpStatusCode": None}, {"httpStatusCode": 200}):
        notice = retry(error={"message": SECRET,
            "codexErrorInfo": {"httpConnectionFailed": detail}})
        s, result, failure = run([notice, START, FINAL, USAGE, DONE])
        assert result == (SECRET, 42) and failure is None
        assert s.retry_progress_http_status == detail.get("httpStatusCode")
    for status in (-1, 65536, "403", False):
        notice = retry(error={"message": SECRET,
            "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": status}}})
        assert run([notice])[2] == "unexpected_notification:error"
    class TimeoutSession(Session):
        def next_event(self):
            if self.values:
                return self.values.popleft()
            raise warm.base.SubscriptionTransportError("timeout")
    s = TimeoutSession([retry()])
    with pytest.raises(warm.base.SubscriptionTransportError, match="timeout"):
        warm.base._run_turn(s, THREAD, SECRET)
    assert len(s.sent) == 1 and s.retry_progress_count == 1


def test_notice_state_resets_on_each_new_turn_without_new_retry_request():
    s = Session([retry(), START, FINAL, USAGE, DONE])
    assert warm.base._run_turn(s, THREAD, "first") == (SECRET, 42)
    assert s.retry_progress_count == 1
    s.values = deque([START, FINAL, USAGE, DONE])
    assert warm.base._run_turn(s, THREAD, "second") == (SECRET, 42)
    assert s.retry_progress_count == 0 and s.retry_progress_http_status is None
    assert [method for method, _ in s.sent] == ["turn/start", "turn/start"]


def test_full_warm_client_unknown_failure_stops_global_budget(monkeypatch):
    s = Session([retry()])
    s.created_at = time.monotonic()
    s.set_deadline = lambda deadline: None
    s.close = lambda: None
    monkeypatch.setattr(warm.base, "inspect_preflight", lambda *args, **kwargs: {
        "auth": "chatgpt", "model": warm.base.MODEL,
        "config_isolation_admitted": True, "inference_enabled": False,
        "quota_windows": [{"remaining_percent": 75}], "_thread_id": THREAD})
    budget = warm.SharedBudget(warm.BudgetLimits(100, 100000, 1000))
    client = warm.WarmSubscriptionClient("unused", budget, "q",
        warm.BudgetLimits(100, 100000, 1000), session_factory=lambda *args, **kwargs: s)
    request = SimpleNamespace(system="system", user=SECRET, temperature=0,
                              max_tokens=32, response_format="json")
    with pytest.raises(warm.ConcurrentStop):
        client.complete(request)
    state = budget.snapshot()
    assert state["stopped"] is True and state["usage_complete"] is False
    assert state["questions"]["q"]["turns"] == 1
    assert len(s.sent) == 1


def test_full_warm_client_settles_one_known_success(monkeypatch):
    s = Session([retry(), START, FINAL, USAGE, DONE])
    s.created_at = time.monotonic()
    s.set_deadline = lambda deadline: None
    s.unsubscribe = lambda thread_id: None
    s.close = lambda: None
    monkeypatch.setattr(warm.base, "inspect_preflight", lambda *args, **kwargs: {
        "auth": "chatgpt", "model": warm.base.MODEL,
        "config_isolation_admitted": True, "inference_enabled": False,
        "quota_windows": [{"remaining_percent": 75}], "_thread_id": THREAD})
    budget = warm.SharedBudget(warm.BudgetLimits(100, 100000, 1000))
    client = warm.WarmSubscriptionClient("unused", budget, "q",
        warm.BudgetLimits(100, 100000, 1000), session_factory=lambda *args, **kwargs: s)
    request = SimpleNamespace(system="system", user=SECRET, temperature=0,
                              max_tokens=32, response_format="json")
    assert client.complete(request) == SECRET
    state = budget.snapshot()
    assert state["stopped"] is False and state["usage_complete"] is True
    assert state["questions"]["q"]["turns"] == 1
    assert state["questions"]["q"]["known_tokens"] == 42
    assert len(s.sent) == 1
