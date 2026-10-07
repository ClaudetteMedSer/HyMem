"""Independent root regression controls for the exact delta opt-out boundary."""
from collections import deque
import hashlib
import inspect
from pathlib import Path
import time

import pytest

from benchmarks import codex_subscription_warm_v7 as old
from benchmarks import codex_subscription_warm_v8 as new
from tests.test_codex_subscription_warm_v5_root import event, sequence, session, error
from tests.test_codex_subscription_warm_v7_root import snapshot


def test_root_authoritative_5000_character_answer_needs_no_fragments(monkeypatch):
    events = sequence()
    events[1]["params"]["item"]["text"] = "x" * 5000
    s = session(new, monkeypatch, events)
    s.reset_private_errors()
    assert new.base._run_turn(s, "thread", "invented") == ("x" * 5000, 19)
    assert s.turn_observation["events_consumed"] == 4


@pytest.mark.parametrize("retired", [False, True])
def test_root_cleanup_cannot_hide_a_delta(monkeypatch, retired):
    s = session(new, monkeypatch, [])
    s.pending = deque([event("item/agentMessage/delta", itemId="item", delta="invented")])
    if retired:
        s.retired_threads.add("thread")
    with pytest.raises(new.base.SubscriptionTransportError, match="^notification_optout_unverified$"):
        s.retire_pending()


@pytest.mark.parametrize("kind", ["unauthorized", "usageLimitExceeded", "rateLimitExceeded",
    "cyberPolicy", "misalignmentPolicyViolation", "sessionBudgetExceeded", "contextWindowExceeded"])
def test_root_optout_preserves_provider_denials(monkeypatch, kind):
    notice = error()
    notice["params"]["error"]["codexErrorInfo"] = kind
    s = session(new, monkeypatch, [notice, *sequence()])
    s.reset_private_errors()
    with pytest.raises(new.base.SubscriptionTransportError, match="unexpected_notification:error"):
        new.base._run_turn(s, "thread", "invented")
    assert s.next_id == 1


def test_root_private_failure_observation_survives(monkeypatch):
    notice = error()
    notice["params"]["willRetry"] = False
    s = session(new, monkeypatch, [notice])
    s.reset_private_errors()
    with pytest.raises(new.base.SubscriptionTransportError):
        new.base._run_turn(s, "thread", "invented")
    record = s.private_failure_record("unexpected_notification:error")
    assert record["error_count"] == 1
    assert record["first"]["http_status_code"] == 403


def test_root_timeout_never_becomes_success(monkeypatch):
    s = session(new, monkeypatch, [])
    s.reset_private_errors()
    s.deadline = time.monotonic() + 0.01
    deadline = s.deadline
    with pytest.raises(new.base.SubscriptionTransportError, match="timeout"):
        new.base._run_turn(s, "thread", "invented")
    assert s.deadline == deadline


@pytest.mark.parametrize("field,value", [("spendControlReached", True),
    ("rateLimitReachedType", "rate_limit_reached"), ("planType", "self_serve_business_usage_based")])
def test_root_optout_keeps_billing_denials(field, value):
    data = snapshot(0)
    assert new.base.quota_metadata({"rateLimits": data})
    data[field] = value
    with pytest.raises(new.base.SubscriptionTransportError):
        new.base.quota_metadata({"rateLimits": data})


def test_root_frozen_parser_caps_model_and_source_remain_exact():
    assert inspect.getsource(new.base._run_turn) == inspect.getsource(old.base._run_turn)
    assert new.base.MAX_EVENTS == old.base.MAX_EVENTS == 4096
    assert new.base.MAX_OUTPUT_CHARS == old.base.MAX_OUTPUT_CHARS == 1_000_000
    assert new.base.MODEL == old.base.MODEL == "gpt-6-luna"
    assert new.base.OVERRIDES == old.base.OVERRIDES
    assert hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() == new.PINNED_WARM_V7_SHA256
    assert new.base is not old.base


def test_root_notification_policy_cannot_widen_the_optout():
    assert new.OPT_OUT_NOTIFICATION_METHODS == ("item/agentMessage/delta",)
    assert new.NOTIFICATION_POLICY == "agent_message_delta_optout_v1"
    assert inspect.signature(new.WarmSubscriptionClient.__init__).parameters["session_factory"].default is new.WarmSession
