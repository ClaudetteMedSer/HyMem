"""Root-owned finite-projection and no-inference controls."""
import copy
import json

import pytest

from benchmarks import codex_subscription as base
from tools.diagnostics import luna_credit_metadata_v1 as reader


PRIVATE = "PRIVATE_ACCOUNT_NAME_OR_SECRET"


def sample(balance="0", unlimited=False):
    return {"rateLimitsByLimitId": {PRIVATE: {"planType": "pro", "limitName": PRIVATE,
        "credits": {"hasCredits": True, "unlimited": unlimited, "balance": balance},
        "primary": {"usedPercent": 20, "windowDurationMins": 300, "resetsAt": 1791046713}}}}


def report():
    gate, snapshots = reader._project(sample(), base.quota_metadata, base.SubscriptionTransportError)
    return {"schema": reader.SCHEMA, "status": "metadata_read",
        "owned_process_cleanup_verified": True, "auth": "chatgpt", "plan": "pro",
        "gate_status": gate, "snapshots": snapshots}


@pytest.mark.parametrize("balance,unlimited,category", [
    ("0", False, "zero"), ("0", True, "zero"), (None, True, "null"),
    ("10", False, "positive"), ("-1", False, "negative"),
    ("1e999", False, "positive"), ("NaN", False, "invalid"),
    (PRIVATE, False, "invalid"), (False, False, "invalid")])
def test_credits_distinguished_without_bypassing_existing_gate(balance, unlimited, category):
    gate, rows = reader._project(sample(balance, unlimited), base.quota_metadata,
                                 base.SubscriptionTransportError)
    assert gate == "credit_balance_present"
    assert rows[0]["balance"] == category and rows[0]["unlimited"] is unlimited
    assert rows[0]["windows"][0]["remaining_percent"] == 80
    assert PRIVATE not in json.dumps(rows)


@pytest.mark.parametrize("field", ["status", "auth", "plan", "gate_status"])
@pytest.mark.parametrize("bad", [[], {}])
def test_unhashable_top_fields_fail_finitely(field, bad):
    value = report()
    value[field] = bad
    assert reader._safe(value).get("status") == "ssh_unverified"


@pytest.mark.parametrize("field", ["balance", "rate_limit_reached_type"])
@pytest.mark.parametrize("bad", [[], {}])
def test_unhashable_snapshot_fields_fail_finitely(field, bad):
    value = report()
    value["snapshots"][0][field] = bad
    assert reader._safe(value).get("status") == "ssh_unverified"


@pytest.mark.parametrize("bad", [[], {}, 1, None])
def test_window_kind_fails_finitely(bad):
    value = report()
    value["snapshots"][0]["windows"][0]["kind"] = bad
    assert reader._safe(value).get("status") == "ssh_unverified"


@pytest.mark.parametrize("credits", [{}, {"hasCredits": True},
    {"hasCredits": True, "unlimited": None}, {"hasCredits": 1, "unlimited": False}])
def test_malformed_required_credit_booleans_rejected(credits):
    value = sample()
    value["rateLimitsByLimitId"][PRIVATE]["credits"] = credits
    with pytest.raises(ValueError):
        reader._project(value, base.quota_metadata, base.SubscriptionTransportError)


def test_projection_does_not_modify_or_reinterpret_source():
    value = sample()
    original = copy.deepcopy(value)
    projected = reader._project(value, base.quota_metadata, base.SubscriptionTransportError)
    assert value == original and projected[0] == "credit_balance_present"


def test_nested_private_field_rejected():
    value = report()
    assert reader._safe(value) == value
    value["snapshots"][0]["message"] = PRIVATE
    assert reader._safe(value).get("status") == "ssh_unverified"


@pytest.mark.parametrize("failure", [True, False])
def test_only_metadata_rpcs_and_cleanup(monkeypatch, failure):
    class Partial(Exception):
        pass
    calls = []
    class Session:
        def __init__(self, *args):
            pass
        def rpc(self, method):
            calls.append(method)
            if method == "account/rateLimits/read":
                if failure:
                    raise RuntimeError(PRIVATE)
                return sample()
            return {}
        def close(self):
            calls.append("close")
            return True
    helper = {"BINARY": "/unused", "_pinned_file": lambda *a, **k: b"unused",
        "_load_transport": lambda _: (None, base.quota_metadata, base.SubscriptionTransportError),
        "AllowlistedSession": Session, "PartialStartupFailure": Partial,
        "_account_projection": lambda _: ("chatgpt", "pro")}
    monkeypatch.setattr(reader, "_bound_access", lambda _: None)
    result = reader._inspect(helper, {"inspect": lambda *a: {}})
    assert calls == ["initialize", "initialized", "account/read", "account/rateLimits/read", "close"]
    assert result["owned_process_cleanup_verified"] is True
    assert result["status"] == ("rpc_unverified" if failure else "metadata_read")
    assert PRIVATE not in json.dumps(result)
