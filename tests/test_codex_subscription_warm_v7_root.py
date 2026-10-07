"""Independent billing-policy controls; invented metadata and no inference."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import codex_subscription as original
from benchmarks import codex_subscription_warm_v7 as warm


def snapshot(remaining=50, balance="2.50", has=True):
    return {"planType": "pro", "spendControlReached": False,
            "rateLimitReachedType": None,
            "credits": {"hasCredits": has, "unlimited": False, "balance": balance},
            "primary": {"usedPercent": 100 - remaining,
                        "windowDurationMins": 10080, "resetsAt": 1791046713}}


def admit(value):
    return warm.base.quota_metadata(value)


@pytest.mark.parametrize("remaining", [0, 1, 24, 25, 100])
def test_finite_existing_credit_allows_measured_window_without_mutation(remaining):
    data = {"rateLimits": snapshot(remaining)}
    before = copy.deepcopy(data)
    assert admit(data) == [{"remaining_percent": remaining,
        "window_minutes": 10080, "resets_at": 1791046713}]
    assert data == before
    assert "balance" not in json.dumps(admit(data))


@pytest.mark.parametrize("field,value", [
    ("spendControlReached", True),
    ("rateLimitReachedType", "rate_limit_reached"),
    ("rateLimitReachedType", "workspace_owner_credits_depleted"),
    ("rateLimitReachedType", "workspace_member_usage_limit_reached"),
    ("planType", "self_serve_business_usage_based"),
])
def test_provider_denial_or_unapproved_plan_wins_over_credit(field, value):
    data = snapshot()
    data[field] = value
    with pytest.raises(warm.base.SubscriptionTransportError):
        admit({"rateLimits": data})


@pytest.mark.parametrize("balance", [None, "0", "-1", "NaN", "Infinity", "-Infinity",
    "1e999999", "", "private account text", True, 1, [], {}])
def test_unknown_invalid_or_nonpositive_credit_never_admits_below_reserve(balance):
    with pytest.raises(warm.base.SubscriptionTransportError):
        admit({"rateLimits": snapshot(1, balance)})


@pytest.mark.parametrize("credits", [None, {}, True, [],
    {"hasCredits": False, "unlimited": False, "balance": "0"},
    {"hasCredits": True, "unlimited": True, "balance": "3"}])
def test_no_finite_credit_evidence_cannot_override_reserve(credits):
    data = snapshot(24)
    data["credits"] = credits
    with pytest.raises(warm.base.SubscriptionTransportError):
        admit({"rateLimits": data})


def test_valid_zero_credits_still_allows_included_allowance():
    assert admit({"rateLimits": snapshot(30, "0", False)})[0]["remaining_percent"] == 30


def test_credit_from_one_bucket_cannot_clear_another_bucket_floor():
    with pytest.raises(warm.base.SubscriptionTransportError):
        admit({"rateLimitsByLimitId": {
            "one": snapshot(50), "two": snapshot(1, "0", False)}})


@pytest.mark.parametrize("used", [True, None, "90", float("nan"), float("inf"), -1, 101])
def test_credit_never_repairs_invalid_window(used):
    data = snapshot()
    data["primary"]["usedPercent"] = used
    with pytest.raises(warm.base.SubscriptionTransportError):
        admit({"rateLimits": data})


def test_no_windows_is_unknown_even_with_credit():
    data = snapshot()
    data["primary"] = None
    with pytest.raises(warm.base.SubscriptionTransportError):
        admit({"rateLimits": data})


def test_notification_uses_same_credit_policy_and_obeys_denial():
    data = snapshot(1)
    event = {"method": "account/rateLimits/updated", "params": {"rateLimits": data}}
    warm.base._validate_account_notification(event)
    data["spendControlReached"] = True
    with pytest.raises(warm.base.SubscriptionTransportError):
        warm.base._validate_account_notification(event)


def test_new_policy_does_not_patch_original_import_or_disk():
    with pytest.raises(original.SubscriptionTransportError, match="credit_balance_present"):
        original.quota_metadata({"rateLimits": snapshot()})
    pins = {"codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
            "codex_subscription_warm_v6.py": "98422aa251ca9482a79be5851d48ae54decc17f17b6aa1149bd6b93b618b784b"}
    for name, digest in pins.items():
        assert hashlib.sha256((Path(original.__file__).parent / name).read_bytes()).hexdigest() == digest
    assert warm.base.OVERRIDES == original.OVERRIDES
    assert warm.base.MODEL == "gpt-6-luna"
    assert warm.base.sanitized_environment({"HOME": "/safe", "OPENAI_API_KEY": "invented"}) == {"HOME": "/safe"}
