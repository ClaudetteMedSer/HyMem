"""Root reproduction: identical invented text, different stream fragmentation."""
import pytest

from benchmarks import codex_subscription_warm_v7 as accepted
from tests.test_codex_subscription_warm_v5_root import event, session


def stream(text, fragments):
    assert "".join(fragments) == text
    return [event("item/started", item={"id": "item", "type": "agentMessage"}),
        *(event("item/agentMessage/delta", itemId="item", delta=part)
          for part in fragments),
        event("item/completed", item={"id": "item", "type": "agentMessage",
            "phase": "final_answer", "text": text}),
        event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 5008}}),
        event("turn/completed", turn={"id": "turn", "status": "completed"})]


def test_same_short_output_passes_coarse_but_rejects_fine_stream(monkeypatch):
    text = "x" * 5000
    coarse = session(accepted, monkeypatch, stream(text, [text]))
    assert accepted.base._run_turn(coarse, "thread", "invented") == (text, 5008)
    fine = session(accepted, monkeypatch, stream(text, list(text)))
    deadline = fine.deadline
    with pytest.raises(accepted.base.SubscriptionTransportError,
                       match="^incomplete_turn_or_usage$"):
        accepted.base._run_turn(fine, "thread", "invented")
    assert fine.next_id == 1 and fine.deadline == deadline
    assert fine.turn_observation == {"basis": "consumed_observed_shape",
        "events_consumed": 4096, "completed_seen": False, "final_seen": False,
        "final_count": 0, "usage_update_count": 0, "usage_state": "absent",
        "last_event_family": "item_agentMessage_delta"}
    assert len(text) < accepted.base.MAX_OUTPUT_CHARS


@pytest.mark.parametrize("size,success", [(4091, True), (4092, True), (4093, False)])
def test_event_limit_boundary_in_actual_parser(monkeypatch, size, success):
    text = "x" * size
    current = session(accepted, monkeypatch, stream(text, list(text)))
    if success:
        assert accepted.base._run_turn(current, "thread", "invented") == (text, 5008)
    else:
        with pytest.raises(accepted.base.SubscriptionTransportError,
                           match="^incomplete_turn_or_usage$"):
            accepted.base._run_turn(current, "thread", "invented")
    assert current.next_id == 1
