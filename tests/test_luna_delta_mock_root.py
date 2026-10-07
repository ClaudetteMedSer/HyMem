"""Independent root tests for the exact-runtime delta diagnostic itself."""
import copy
import queue
import time

import pytest

from tools.diagnostics import luna_delta_runtime_mock_v1 as probe


def event(method, **params):
    return {"method": method, "params": {"threadId": "t", "turnId": "u", **params}}


def events(deltas):
    item = {"id": "a", "type": "agentMessage", "phase": "final_answer", "text": probe.TEXT}
    usage = {"inputTokens": 8, "outputTokens": 5000, "totalTokens": 5008}
    return [event("turn/started", turn={"id": "u", "status": "inProgress"}),
        event("item/started", item={**item, "text": ""}),
        *(event(probe.OPTOUT_METHOD, itemId="a", delta="x") for _ in range(deltas)),
        event("item/completed", item=item),
        event("thread/tokenUsage/updated", tokenUsage={"total": usage, "last": usage}),
        event("turn/completed", turn={"id": "u", "status": "completed"})]


class NoExtraReads:
    def receive(self, _deadline):
        raise AssertionError("unexpected read past completion")


def test_real_schema_final_item_text_and_5000_fragments():
    rows = []
    for case, count in (("baseline", 5000), ("opt_out", 0)):
        row = probe.empty_result(case)
        probe.consume(NoExtraReads(), events(count), time.monotonic() + 1, row, "t", "u")
        row.update(cleanup_verified=True, mock_boundary_valid=True, http_requests=1)
        assert row["final_digest_matches"] is True
        rows.append(row)
    assert probe.success_pair(rows)


@pytest.mark.parametrize("method", ["item/completed", "thread/tokenUsage/updated", "turn/completed"])
def test_foreign_identity_fails(method):
    values = events(0)
    value = next(value for value in values if value["method"] == method)
    value["params"]["threadId"] = "foreign"
    with pytest.raises(probe.ProbeFailure, match="identity"):
        probe.consume(NoExtraReads(), values, time.monotonic() + 1,
            probe.empty_result("opt_out"), "t", "u")


@pytest.mark.parametrize("tokens", [False, -1, "3", None])
def test_bad_usage_fails(tokens):
    values = copy.deepcopy(events(0))
    values[-2]["params"]["tokenUsage"]["total"]["totalTokens"] = tokens
    with pytest.raises(probe.ProbeFailure, match="usage_shape"):
        probe.consume(NoExtraReads(), values, time.monotonic() + 1,
            probe.empty_result("opt_out"), "t", "u")


def test_absolute_deadline_cannot_be_extended_by_queued_work():
    app = object.__new__(probe.AppServer)
    app.lines = queue.Queue()
    app.lines.put({"method": "invented"})
    with pytest.raises(probe.ProbeFailure, match="deadline"):
        app.receive(time.monotonic() - 1)


def test_pending_notifications_cannot_bypass_absolute_deadline():
    with pytest.raises(probe.ProbeFailure, match="deadline"):
        probe.consume(NoExtraReads(), events(0), time.monotonic() - 1,
            probe.empty_result("opt_out"), "t", "u")


def test_provider_response_contains_exact_5000_single_character_deltas():
    raw = probe.response_events()
    assert raw.count(b"event: response.output_text.delta\n") == 5000
    assert len(raw) < 2_000_000
    assert probe.OPTOUT_METHOD == "item/agentMessage/delta"
    assert probe.EVENT_LIMIT == 6000 and probe.CASE_SECONDS == 32
