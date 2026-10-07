"""Offline controls for the private per-window credit proof and warm client."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from benchmarks import codex_subscription_warm_v8 as old
from benchmarks import codex_subscription_warm_v9 as warm


POSITIVE = {"hasCredits": True, "unlimited": False, "balance": "2.0"}


def rate_limits(remaining=1, credits=POSITIVE):
    return {"rateLimits": {"planType": "pro", "credits": copy.deepcopy(credits),
        "primary": {"usedPercent": 100 - remaining, "windowDurationMins": 300,
                    "resetsAt": 1790784000}}}


def admission(windows):
    return {"auth": "chatgpt", "model": warm.base.MODEL,
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": windows}


def reserved():
    budget = warm.SharedBudget(warm.BudgetLimits(16, 160000, 600), max_in_flight=4)
    budget.register("q", warm.BudgetLimits(4, 160000, 600))
    budget.reserve("q")
    return budget


def test_proof_is_bound_to_exact_issuing_window_and_original_fields():
    windows = warm.base.quota_metadata(rate_limits())
    assert windows == [{"remaining_percent": 1, "window_minutes": 300,
                        "resets_at": 1790784000}]
    assert "2.0" not in json.dumps(windows)
    for forged in (dict(windows[0]), copy.copy(windows[0]),
                   {"remaining_percent": 1, "credit_eligible": True}):
        budget = reserved()
        with pytest.raises(warm.ConcurrentStop, match="^quota_unverified$"):
            budget.before_turn("q", admission([forged]))
        assert budget.snapshot()["turns"] == 0
        budget.settle("q", used=None, turn_started=False)
        assert budget.snapshot()["reserved"] == budget.snapshot()["in_flight"] == 0
    windows[0]["window_minutes"] = 301
    budget = reserved()
    with pytest.raises(warm.ConcurrentStop, match="^quota_unverified$"):
        budget.before_turn("q", admission(windows))
    windows[0]["remaining_percent"] = 30
    budget = reserved()
    with pytest.raises(warm.ConcurrentStop, match="^quota_unverified$"):
        budget.before_turn("q", admission(windows))


def test_low_credit_is_scoped_per_snapshot_and_high_included_quota_still_works():
    high = rate_limits(80, None)["rateLimits"]
    credited = rate_limits(1)["rateLimits"]
    windows = warm.base.quota_metadata({"rateLimitsByLimitId": {
        "credited": credited, "included": high}})
    assert [window["remaining_percent"] for window in windows] == [1, 80]
    budget = reserved()
    assert budget.before_turn("q", admission(windows)) > 0
    budget.settle("q", used=17, turn_started=True)
    assert budget.snapshot()["known_tokens"] == 17
    with pytest.raises(warm.base.SubscriptionTransportError, match="^quota_floor$"):
        warm.base.quota_metadata({"rateLimitsByLimitId": {
            "credited": credited, "uncredited": rate_limits(1, None)["rateLimits"]}})


class FakeAppServer:
    def __init__(self, binary, cwd, timeout=120):
        self.created_at = time.monotonic()
        self.calls = []
        self.closed = False
        self.stage = None
        self.bound_thread = None

    def set_deadline(self, deadline):
        assert deadline > time.monotonic()

    def send(self, method, params, *, notification=False):
        self.calls.append(method)

    def rpc(self, method, params):
        self.calls.append(method)
        if method == "initialize":
            return {}
        if method == "account/read":
            return {"account": {"type": "chatgpt", "planType": "pro"}}
        if method == "model/list":
            return {"data": [{"model": warm.base.MODEL,
                "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]}
        if method == "account/rateLimits/read":
            return rate_limits()
        if method == "config/read":
            return {"config": {"forced_login_method": "chatgpt",
                "model_provider": "openai", "model": warm.base.MODEL,
                "web_search": "disabled", "project_doc_max_bytes": 0,
                "memories": {"use_memories": False, "generate_memories": False},
                "developer_instructions": "", "features": {
                    **dict.fromkeys(warm.base.DISABLED_FEATURES, False),
                    "skip_host_skill_discovery": True}, "mcp_servers": {}}}
        if method == "thread/start":
            assert params["allowProviderModelFallback"] is False
            return {"model": warm.base.MODEL, "modelProvider": "openai",
                "runtimeWorkspaceRoots": [], "instructionSources": [],
                "sandbox": {"type": "readOnly", "networkAccess": False},
                "approvalPolicy": "never", "reasoningEffort": "low",
                "thread": {"id": "thread", "ephemeral": True,
                           "environments": [], "turns": [], "path": None}}
        raise AssertionError(method)

    def bind_thread_id(self, thread_id):
        self.bound_thread = thread_id

    def unsubscribe(self, thread_id):
        assert thread_id == "thread"

    def close(self):
        self.closed = True


@pytest.mark.parametrize("module,succeeds", [(old, False), (warm, True)])
def test_actual_preflight_to_client_budget_pipeline_without_provider(monkeypatch, module, succeeds):
    observed = []
    sessions = []

    def factory(*args, **kwargs):
        session = FakeAppServer(*args, **kwargs)
        sessions.append(session)
        return session

    def fake_turn(session, thread_id, user):
        observed.append((session, thread_id, user))
        return "invented answer", 19

    monkeypatch.setattr(module.base, "_run_turn", fake_turn)
    budget = module.SharedBudget(module.BudgetLimits(16, 160000, 600), max_in_flight=4)
    client = module.WarmSubscriptionClient("unused", budget, "q",
        module.BudgetLimits(4, 160000, 600), session_factory=factory)
    request = SimpleNamespace(system="invented system", user="invented user",
                              temperature=0, max_tokens=32, response_format="text")
    if succeeds:
        assert client.complete(request) == "invented answer"
        assert len(observed) == 1
        assert budget.snapshot()["known_tokens"] == 19
        assert budget.snapshot()["turns"] == 1
        assert budget.snapshot()["reserved"] == budget.snapshot()["in_flight"] == 0
    else:
        with pytest.raises(module.ConcurrentStop, match="^quota_unverified$"):
            client.complete(request)
        assert observed == []
        assert budget.snapshot()["turns"] == 0
        assert budget.snapshot()["stop_code"] == "quota_unverified"
    assert sessions[0].bound_thread == "thread"
    assert sessions[0].calls == ["initialize", "initialized", "account/read",
        "model/list", "account/rateLimits/read", "config/read", "thread/start"]
    client.close()
    assert sessions[0].closed


def test_pinned_v8_source_and_notification_policy_are_unchanged():
    assert hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() == warm.PINNED_WARM_V8_SHA256
    assert warm.NOTIFICATION_POLICY == old.NOTIFICATION_POLICY
    assert warm.OPT_OUT_NOTIFICATION_METHODS == old.OPT_OUT_NOTIFICATION_METHODS
    assert warm.WarmSession is warm.v8.WarmSession
    assert old.concurrent.SharedBudget is not warm.SharedBudget
    with pytest.raises(old.ConcurrentStop, match="^quota_unverified$"):
        budget = old.SharedBudget(old.BudgetLimits(16, 160000, 600))
        budget.register("q", old.BudgetLimits(4, 160000, 600))
        budget.reserve("q")
        budget.before_turn("q", admission(old.base.quota_metadata(rate_limits())))
