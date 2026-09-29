"""Offline v3 transport diagnostics; no App Server or model calls."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_warm_v3 as warm


def _error(**changes):
    event = {"method": "error", "params": {
        "threadId": "thread-1", "turnId": "turn-1", "willRetry": False,
        "error": {"message": "secret@example.com /private/secret",
                  "codexErrorInfo": "rateLimitExceeded", "additionalDetails": "secret"},
    }}
    event["params"].update(changes)
    return event


def _session(events):
    session = object.__new__(warm.WarmSession)
    session.pending = deque(events)
    session.stage = "turn/start"
    session.active_thread = "thread-1"
    session.active_turn = "turn-1"
    session.last_event = None
    session.rpc_error = None
    session.retired_threads = set()
    session.warning_targets = []
    session.rpc = lambda *args, **kwargs: {"turn": {"id": "turn-1", "status": "inProgress"}}
    return session


def test_actual_run_turn_error_is_rejected_with_safe_matched_metadata():
    session = _session([_error(willRetry=True)])
    with pytest.raises(warm.base.SubscriptionTransportError, match="unexpected_notification:error"):
        warm.base._run_turn(session, "thread-1", "synthetic user")
    assert session.stage == "turn/events"
    assert session.last_event == {"event_family": "error", "app_server_error": {
        "identity": "matched", "will_retry": True, "error_class": "rateLimitExceeded"}}
    assert "secret" not in repr(session.last_event)


def test_actual_run_turn_success_and_usage_are_unchanged():
    session = _session([
        {"method": "turn/started", "params": {"threadId": "thread-1", "turn": {"id": "turn-1"}}},
        {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread-1", "turnId": "turn-1",
            "tokenUsage": {"total": {"totalTokens": 7}}}},
        {"method": "item/started", "params": {"threadId": "thread-1", "turnId": "turn-1",
            "item": {"id": "item-1", "type": "agentMessage"}}},
        {"method": "item/completed", "params": {"threadId": "thread-1", "turnId": "turn-1",
            "item": {"id": "item-1", "type": "agentMessage", "phase": "final_answer", "text": "{}"}}},
        {"method": "turn/completed", "params": {"threadId": "thread-1", "turn": {
            "id": "turn-1", "status": "completed"}}},
    ])
    assert warm.base._run_turn(session, "thread-1", "user") == ("{}", 7)
    assert session.last_event is None and session.stage == "turn/events"


@pytest.mark.parametrize("thread,turn,identity", [
    ("foreign", "turn-1", "mismatch"), ("thread-1", "foreign", "mismatch"),
    ("", "turn-1", "invalid"), (True, "turn-1", "invalid"),
    ("thread-1", "", "invalid"), ("thread-1", False, "invalid"),
])
def test_foreign_or_invalid_identity_never_attributes_error(thread, turn, identity):
    result = warm._app_error(_error(threadId=thread, turnId=turn), "thread-1", "turn-1")
    assert result == {"identity": identity}


@pytest.mark.parametrize("info,expected", [
    ("serverOverloaded", {"error_class": "serverOverloaded"}),
    ({"httpConnectionFailed": {"httpStatusCode": 503}},
     {"error_class": "httpConnectionFailed", "http_status_code": 503}),
    ({"responseStreamDisconnected": {"httpStatusCode": None}},
     {"error_class": "responseStreamDisconnected", "http_status_code": None}),
    ({"responseStreamDisconnected": {}},
     {"error_class": "responseStreamDisconnected", "http_status_code": None}),
    ({"activeTurnNotSteerable": {"turnKind": "review"}},
     {"error_class": "activeTurnNotSteerable"}),
    ({"httpConnectionFailed": {"httpStatusCode": True}},
     {"error_class": "httpConnectionFailed", "http_status_code": "invalid"}),
    ({"httpConnectionFailed": {"httpStatusCode": 65536}},
     {"error_class": "httpConnectionFailed", "http_status_code": "invalid"}),
    ({"activeTurnNotSteerable": {"turnKind": "secret"}}, {"error_class": "invalid"}),
    ({"activeTurnNotSteerable": {"turnKind": ["review"]}}, {"error_class": "invalid"}),
    ("secret/path", {"error_class": "invalid"}),
    ({"unknown": {"message": "secret"}}, {"error_class": "invalid"}),
])
def test_structured_error_has_finite_schema(info, expected):
    event = _error()
    event["params"]["error"]["codexErrorInfo"] = info
    result = warm._app_error(event, "thread-1", "turn-1")
    assert result == {"identity": "matched", "will_retry": False, **expected}
    assert "secret" not in repr(result)


def test_malformed_error_and_retry_are_fixed_classes():
    event = _error(willRetry="yes", error={"message": 5, "codexErrorInfo": "rateLimitExceeded"})
    assert warm._app_error(event, "thread-1", "turn-1") == {
        "identity": "matched", "will_retry": "invalid", "error_class": "invalid"}
    assert warm._app_error(_error(), "thread-1", None) == {"identity": "unbound"}


@pytest.mark.parametrize("code,category", [
    (-32603, "internal_error"), (403, "other_numeric"), (True, "invalid"),
    ("-32603", "invalid"), (2**40, "invalid"),
])
def test_jsonrpc_error_only_retains_bounded_numeric_code(code, category):
    result = warm._rpc_error({"error": {"code": code, "message": "secret /path"}})
    assert result["category"] == category
    assert "secret" not in repr(result)


def test_receive_only_attributes_matching_rpc_error(monkeypatch):
    session = _session([])
    session.request_id = 4
    monkeypatch.setattr(warm.base.StdioSession, "receive", lambda self: {
        "id": 5, "error": {"code": -32603, "message": "secret"}})
    session.receive()
    assert session.rpc_error is None
    monkeypatch.setattr(warm.base.StdioSession, "receive", lambda self: {
        "id": 4, "error": {"code": -32603, "message": "secret"}})
    session.receive()
    assert session.rpc_error == {"category": "internal_error", "code": -32603}


def test_queued_error_is_attributed_only_after_turn_id_is_known():
    session = _session([_error()])
    session.active_turn = None
    session._route(_error(), "turn/start", True)
    assert session.last_event is None
    session.active_turn = "turn-1"
    with pytest.raises(warm.base.SubscriptionTransportError):
        warm.base._run_turn(session, "thread-1", "user")
    assert session.last_event["app_server_error"]["identity"] == "matched"


def test_unknown_method_is_fixed_family_and_never_public_raw():
    secret = "secret@example.com/private/path"
    session = _session([{"method": secret, "params": {"threadId": "thread-1"}}])
    with pytest.raises(warm.base.SubscriptionTransportError):
        warm.base._run_turn(session, "thread-1", "user")
    assert session.last_event == {"event_family": "unknown"}
    assert secret not in repr(warm.serialize_failure({"code": "unexpected_notification_unknown",
        **session.last_event, "private": secret}))


def test_serializer_rejects_arbitrary_nested_and_top_level_text():
    secret = "secret@example.com /private/file"
    result = warm.serialize_failure({"code": secret, "phase": secret, "rpc": secret,
        "event_family": secret, "failure_family": secret, "known_tokens": True,
        "app_server_error": {"identity": "matched", "error_class": secret,
                             "will_retry": secret, "other": secret},
        "rpc_error": {"category": secret, "code": secret}})
    assert result == {"code": "fixed_other", "phase": "unknown", "rpc": None,
                      "app_server_error": {"identity": "matched"}}
    assert secret not in repr(result)


class FakeSession:
    instances = []
    fail_close = False

    def __init__(self, binary, cwd, timeout=120):
        self.created_at = warm.time.monotonic()
        self.stage = "initialize"
        self.pending = deque()
        self.starting_events = []
        self.retired_threads = set()
        self.last_event = None
        self.rpc_error = None
        self.closed = False
        self.__class__.instances.append(self)

    def set_deadline(self, deadline):
        self.deadline = deadline

    def unsubscribe(self, thread_id):
        self.stage = "thread/unsubscribe"
        self.retired_threads.add(thread_id)

    def close(self):
        if self.fail_close:
            raise RuntimeError("secret cleanup")
        self.closed = True


def _client():
    budget = warm.SharedBudget(warm.BudgetLimits(100, 10000, 1000))
    return warm.WarmSubscriptionClient("unused", budget, "q",
        warm.BudgetLimits(100, 10000, 1000), session_factory=FakeSession)


def _request():
    return SimpleNamespace(system="system", user="user", temperature=0,
                           max_tokens=32, response_format="json")


def _admission(session, **kwargs):
    session.stage = "thread/start"
    return {"auth": "chatgpt", "model": warm.base.MODEL,
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": [{"remaining_percent": 75}], "_thread_id": "thread-1"}


def test_first_fault_survives_cleanup_and_failed_usage_is_incomplete(monkeypatch):
    FakeSession.instances.clear()
    monkeypatch.setattr(warm.base, "inspect_preflight", _admission)
    def fail_run(session, *args):
        session.stage = "turn/events"
        session.last_event = {"event_family": "error", "app_server_error": {
            "identity": "matched", "will_retry": False, "error_class": "internalServerError"}}
        warm.base._fail("unexpected_notification:error")
    monkeypatch.setattr(warm.base, "_run_turn", fail_run)
    FakeSession.fail_close = True
    item = _client()
    with pytest.raises(warm.ConcurrentStop, match="cleanup_failure"):
        item.complete(_request())
    failure = item.budget.snapshot()["first_failure"]
    assert failure["code"] == "fixed_other"
    assert failure["failure_family"] == "unexpected_notification"
    assert failure["rpc"] == "turn/events"
    assert failure["app_server_error"]["error_class"] == "internalServerError"
    assert failure["turn_admitted"] is True and failure["known_usage"] is False
    assert item.budget.snapshot()["usage_complete"] is False
    FakeSession.fail_close = False
    item.close()


def test_success_then_preflight_failure_does_not_reuse_event_detail(monkeypatch):
    FakeSession.instances.clear()
    monkeypatch.setattr(warm.base, "inspect_preflight", _admission)
    monkeypatch.setattr(warm.base, "_run_turn", lambda *args: ("{}", 7))
    item = _client()
    assert item.complete(_request()) == "{}"
    item.session.last_event = {"event_family": "error", "app_server_error": {
        "identity": "matched", "error_class": "rateLimitExceeded"}}
    monkeypatch.setattr(warm.base, "inspect_preflight", lambda *args, **kwargs: warm.base._fail("quota_floor"))
    with pytest.raises(warm.ConcurrentStop, match="quota_floor"):
        item.complete(_request())
    failure = item.budget.snapshot()["first_failure"]
    assert failure["code"] == "quota_floor" and "app_server_error" not in failure
    assert item.budget.snapshot()["known_tokens"] == 7


def test_preflight_rejected_error_keeps_only_unbound_family(monkeypatch):
    FakeSession.instances.clear()
    def fail_preflight(session, **kwargs):
        session.stage = "thread/start"
        session.last_event = {"event_family": "error", "app_server_error": {"identity": "unbound"}}
        warm.base._fail("unexpected_notification:error")
    monkeypatch.setattr(warm.base, "inspect_preflight", fail_preflight)
    item = _client()
    with pytest.raises(warm.ConcurrentStop, match="fixed_other"):
        item.complete(_request())
    failure = item.budget.snapshot()["first_failure"]
    assert failure["code"] == "fixed_other" and failure["phase"] == "preflight"
    assert failure["failure_family"] == "unexpected_notification"
    assert failure["app_server_error"] == {"identity": "unbound"}
