"""Independent finalized-item reconstruction controls; no private data/network."""
import copy
import pytest
from benchmarks import chatgpt_plan_responses_v5 as wire
from benchmarks import chatgpt_plan_responses_v4 as old
from tests.test_chatgpt_plan_responses_root_v1 import terminal


def sequence(output_state="missing"):
    end = terminal()
    item = end["response"].pop("output")[0]
    item.update(id="invented-item", status="completed")
    if output_state == "null": end["response"]["output"] = None
    if output_state == "empty": end["response"]["output"] = []
    return [{"type": "response.output_item.done", "output_index": 2, "item": item}, end]


@pytest.mark.parametrize("state", ["missing", "null"])
def test_independent_reproduction_then_repaired_finalized_item(state):
    events = sequence(state)
    with pytest.raises(old.TransportError) as caught: old.parse_stream_events(events)
    assert caught.value.code == "invalid_output"
    result = wire.parse_stream_events(events)
    assert result.text == " invented\n" and result.total_tokens == 13
    assert " invented" not in repr(result)


@pytest.mark.parametrize("value", [[], {}, "PRIVATE_SENTINEL", False])
def test_explicit_bad_terminal_output_never_replaced(value):
    events = sequence()
    events[-1]["response"]["output"] = value
    with pytest.raises(wire.TransportError) as caught: wire.parse_stream_events(events)
    assert caught.value.code == "invalid_output"
    assert "PRIVATE_SENTINEL" not in repr(caught.value)


@pytest.mark.parametrize("kind", ["response.output_item.added", "response.output_text.delta", "response.output_text.done"])
def test_partial_or_text_only_events_cannot_replace_final_item(kind):
    events = sequence()
    events[0]["type"] = kind
    with pytest.raises(wire.TransportError): wire.parse_stream_events(events)


@pytest.mark.parametrize("field,value", [("status", "in_progress"), ("role", "user"),
                                        ("type", "function_call"), ("channel", "analysis")])
def test_nonfinal_or_unsupported_done_items_fail(field, value):
    events = sequence()
    events[0]["item"][field] = value
    with pytest.raises(wire.TransportError): wire.parse_stream_events(events)


def test_completed_event_still_required():
    with pytest.raises(wire.TransportError) as caught: wire.parse_stream_events(sequence()[:-1])
    assert caught.value.code == "missing_completion"


@pytest.mark.parametrize("change,code", [({"model": "wrong"}, "model_mismatch"),
    ({"usage": None}, "missing_usage"), ({"status": "incomplete"}, "incomplete_response"),
    ({"error": {"message": "PRIVATE_SENTINEL"}}, "response_failure")])
def test_reconstruction_does_not_skip_terminal_gates(change, code):
    events = sequence()
    events[-1]["response"].update(change)
    with pytest.raises(wire.TransportError) as caught: wire.parse_stream_events(events)
    assert caught.value.code == code and "PRIVATE_SENTINEL" not in repr(caught.value)


@pytest.mark.parametrize("index", [True, -1, None, [], "0"])
def test_invalid_done_indices_rejected(index):
    events = sequence()
    events[0]["output_index"] = index
    with pytest.raises(wire.TransportError): wire.parse_stream_events(events)


def test_duplicate_done_indices_and_identities_rejected():
    events = sequence()
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events([events[0], copy.deepcopy(events[0]), events[-1]])
    duplicate = copy.deepcopy(events[0]); duplicate["output_index"] = 4
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events([events[0], duplicate, events[-1]])


def test_finalized_items_sorted_by_index_with_gaps():
    events = sequence()
    second = copy.deepcopy(events[0])
    second["output_index"] = 0; second["item"]["id"] = "invented-earlier"
    second["item"]["content"][0]["text"] = "first:"
    assert wire.parse_stream_events([events[0], second, events[-1]]).text == "first: invented\n"


def test_done_item_snapshot_is_not_mutated_by_caller():
    events = sequence()
    def source():
        yield events[0]
        events[0]["item"]["content"][0]["text"] = "PRIVATE_SENTINEL"
        yield events[-1]
    assert wire.parse_stream_events(source()).text == " invented\n"


def test_unfinished_added_item_cannot_disappear_during_reconstruction():
    events = sequence()
    added = copy.deepcopy(events[0])
    added.update(type="response.output_item.added", output_index=4)
    added["item"].update(id="invented-unfinished", status="in_progress")
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events([added, *events])


@pytest.mark.parametrize("identity", [None, "invented-different"])
def test_done_identity_cannot_lose_or_change_known_added_binding(identity):
    events = sequence()
    added = copy.deepcopy(events[0]); added["type"] = "response.output_item.added"
    if identity is None: events[0]["item"].pop("id")
    else: events[0]["item"]["id"] = identity
    with pytest.raises(wire.TransportError): wire.parse_stream_events([added, *events])


def test_finite_reconstruction_observation_and_impossible_states():
    events = sequence("null")
    events[-1]["response"].pop("usage")
    with pytest.raises(wire.TransportError) as caught: wire.parse_stream_events(events)
    obs = caught.value.stream_observation
    assert (obs["terminal_output_state"], obs["finalized_item_count"], obs["output_reconstructed"]) == ("null", 1, True)
    assert wire._sanitize_stream_observation({**obs, "finalized_item_count": 0}) is None
    assert wire._sanitize_stream_observation({**obs, "terminal_output_state": "empty"}) is None


def test_added_after_done_cannot_rebind_identity():
    events = sequence()
    added = copy.deepcopy(events[0]); added["type"] = "response.output_item.added"
    added["item"]["id"] = "invented-conflict"
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events([events[0], added, events[-1]])


def test_second_terminal_retains_finite_failure_after_reconstruction():
    with pytest.raises(wire.TransportError) as caught:
        wire.parse_stream_events([*sequence("null"), terminal()])
    assert caught.value.code == "event_after_completion"
    assert wire._sanitize_stream_observation(caught.value.stream_observation) is not None


def test_retained_identity_bytes_bounded(monkeypatch):
    monkeypatch.setattr(wire, "MAX_WIRE_BYTES", 300)
    events = sequence()
    events[0]["item"]["id"] = "x" * 300
    with pytest.raises(wire.TransportError) as caught: wire.parse_stream_events(events)
    assert caught.value.code == "wire_limit"


def test_surrogate_identity_is_finite_invalid_event():
    events = sequence(); events[0]["item"]["id"] = "\ud800"
    with pytest.raises(wire.TransportError) as caught: wire.parse_stream_events(events)
    assert caught.value.code == "invalid_event"
