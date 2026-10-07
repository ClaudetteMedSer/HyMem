"""Offline checks of the opt-in existing-credit billing gate."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import codex_subscription as original
from benchmarks import codex_subscription_warm_v6 as previous
from benchmarks import codex_subscription_warm_v7 as warm


def quota(remaining: int, balance: object = "2.50", *, has: bool = True,
          unlimited: bool = False) -> dict:
    return {"rateLimits": {"planType": "pro", "spendControlReached": False,
        "rateLimitReachedType": None,
        "credits": {"hasCredits": has, "unlimited": unlimited, "balance": balance},
        "primary": {"usedPercent": 100 - remaining, "windowDurationMins": 10080,
                    "resetsAt": 1791046713}}}


def test_positive_credit_admits_below_floor_without_mutation_or_balance_output():
    data = quota(1)
    before = copy.deepcopy(data)
    windows = warm.base.quota_metadata(data)
    assert windows == [{"remaining_percent": 1, "window_minutes": 10080,
                        "resets_at": 1791046713}]
    assert data == before
    assert "2.50" not in json.dumps(windows)


@pytest.mark.parametrize("balance", ["0", "-1", None, "NaN", "Infinity", "1e999999",
                                     "", "account secret", True, float("nan"), float("inf")])
def test_nonpositive_unknown_or_malformed_balance_cannot_lift_reserve(balance):
    with pytest.raises(warm.base.SubscriptionTransportError):
        warm.base.quota_metadata(quota(24, balance))


def test_included_quota_admits_without_positive_credits():
    for credits in (None, {"hasCredits": False, "unlimited": False, "balance": "0"},
                    {"hasCredits": True, "unlimited": True, "balance": "2.50"}):
        data = quota(25)
        data["rateLimits"]["credits"] = credits
        assert warm.base.quota_metadata(data)[0]["remaining_percent"] == 25
    for malformed in ({}, {"hasCredits": 1, "unlimited": False, "balance": "2.50"},
                      {"hasCredits": True, "unlimited": 0, "balance": "2.50"}):
        data = quota(25)
        data["rateLimits"]["credits"] = malformed
        with pytest.raises(warm.base.SubscriptionTransportError, match="invalid_quota"):
            warm.base.quota_metadata(data)


def test_provider_denial_and_other_bucket_floor_win_over_credit():
    denied = quota(1)
    denied["rateLimits"]["spendControlReached"] = True
    with pytest.raises(warm.base.SubscriptionTransportError, match="quota_exhausted"):
        warm.base.quota_metadata(denied)
    low = quota(1, "0", has=False)["rateLimits"]
    with pytest.raises(warm.base.SubscriptionTransportError, match="quota_floor"):
        warm.base.quota_metadata({"rateLimitsByLimitId": {
            "credited": quota(1)["rateLimits"], "uncredited": low}})


def test_credit_notifications_use_same_gate():
    event = {"method": "account/rateLimits/updated",
             "params": {"rateLimits": quota(1)["rateLimits"]}}
    warm.base._validate_account_notification(event)
    event["params"]["rateLimits"]["rateLimitReachedType"] = "rate_limit_reached"
    with pytest.raises(warm.base.SubscriptionTransportError, match="quota_exhausted"):
        warm.base._validate_account_notification(event)


def test_preflight_keeps_subscription_model_and_no_fallback_rpcs():
    class Session:
        def __init__(self):
            self.calls = []
            self.responses = {
                "initialize": {},
                "account/read": {"account": {"type": "chatgpt", "planType": "pro"}},
                "model/list": {"data": [{"model": warm.base.MODEL,
                    "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]},
                "account/rateLimits/read": quota(1),
                "config/read": {"config": {
                    "forced_login_method": "chatgpt", "model_provider": "openai",
                    "model": warm.base.MODEL, "web_search": "disabled",
                    "project_doc_max_bytes": 0,
                    "memories": {"use_memories": False, "generate_memories": False},
                    "developer_instructions": "",
                    "features": {**dict.fromkeys(warm.base.DISABLED_FEATURES, False),
                                 "skip_host_skill_discovery": True},
                    "mcp_servers": {}}},
                "thread/start": {"model": warm.base.MODEL, "modelProvider": "openai",
                    "runtimeWorkspaceRoots": [], "instructionSources": [],
                    "sandbox": {"type": "readOnly", "networkAccess": False},
                    "approvalPolicy": "never", "reasoningEffort": "low",
                    "thread": {"id": "thread", "ephemeral": True, "environments": [],
                               "turns": [], "path": None}},
            }

        def rpc(self, method, params):
            self.calls.append((method, params))
            return self.responses[method]

        def send(self, method, params, **kwargs):
            self.calls.append((method, params))

    session = Session()
    result = warm.base.inspect_preflight(session)
    assert result["quota_windows"][0]["remaining_percent"] == 1
    assert "balance" not in json.dumps(result)
    methods = [method for method, _ in session.calls]
    assert methods == ["initialize", "initialized", "account/read", "model/list",
                       "account/rateLimits/read", "config/read", "thread/start"]
    params = dict(session.calls)["thread/start"]
    assert params["model"] == "gpt-6-luna"
    assert params["allowProviderModelFallback"] is False


def test_failure_projection_and_prior_imports_stay_unchanged():
    private = "private balance 2.50"
    projected = warm.serialize_failure({"code": "quota_floor", "phase": "preflight",
        "rpc": "account/rateLimits/read", "balance": private, "message": private})
    assert private not in json.dumps(projected)
    assert projected == previous.serialize_failure({"code": "quota_floor", "phase": "preflight",
        "rpc": "account/rateLimits/read", "balance": private, "message": private})
    with pytest.raises(original.SubscriptionTransportError, match="credit_balance_present"):
        original.quota_metadata(quota(1))
    with pytest.raises(previous.base.SubscriptionTransportError, match="credit_balance_present"):
        previous.base.quota_metadata(quota(1))
    for name, digest in (("codex_subscription.py", warm.PINNED_BASE_SHA256),
                         ("codex_subscription_warm_v6.py", warm.PINNED_WARM_V6_SHA256)):
        assert hashlib.sha256((Path(original.__file__).parent / name).read_bytes()).hexdigest() == digest
    assert warm.BILLING_POLICY == "included_allowance_or_existing_finite_positive_credits_v1"
