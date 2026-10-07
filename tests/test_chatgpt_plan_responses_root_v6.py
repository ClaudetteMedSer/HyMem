"""Independent empty-terminal compatibility controls; invented events only."""
import copy
import pytest
from benchmarks import chatgpt_plan_responses_v6 as wire
from benchmarks import chatgpt_plan_responses_v5 as old
from tests.test_chatgpt_plan_responses_root_v5 import sequence


@pytest.mark.parametrize("state", ["missing", "null", "empty"])
def test_finalized_evidence_is_required_and_validated(state):
    events = sequence(state)
    if state == "empty":
        with pytest.raises(old.TransportError) as caught: old.parse_stream_events(events)
        assert caught.value.code == "invalid_output"
    result = wire.parse_stream_events(events)
    assert result.text == " invented\n" and result.total_tokens == 13
    with pytest.raises(wire.TransportError): wire.parse_stream_events(events[-1:])


@pytest.mark.parametrize("change,code", [({"model": "wrong"}, "model_mismatch"),
    ({"usage": None}, "missing_usage"), ({"status": "incomplete"}, "incomplete_response"),
    ({"error": {}}, "response_failure"), ({"output": {}}, "invalid_output")])
def test_empty_reconstruction_keeps_terminal_validation(change, code):
    events = sequence("empty"); events[-1]["response"].update(change)
    with pytest.raises(wire.TransportError) as caught: wire.parse_stream_events(events)
    assert caught.value.code == code


def test_empty_reconstruction_keeps_partial_duplicate_and_refusal_rejections():
    events = sequence("empty")
    added = copy.deepcopy(events[0]); added.update(type="response.output_item.added", output_index=9)
    added["item"]["id"] = "invented-unfinished"
    for bad in ([added, *events], [events[0], events[0], events[-1]], [added, events[-1]]):
        with pytest.raises(wire.TransportError): wire.parse_stream_events(bad)
    events[0]["item"]["content"][0]["type"] = "refusal"
    with pytest.raises(wire.TransportError): wire.parse_stream_events(events)


def test_empty_state_remains_explicit_in_safe_observation():
    events = sequence("empty"); events[-1]["response"].pop("usage")
    with pytest.raises(wire.TransportError) as caught: wire.parse_stream_events(events)
    obs = caught.value.stream_observation
    assert (obs["terminal_output_state"], obs["finalized_item_count"], obs["output_reconstructed"]) == ("empty",1,True)
    assert wire._sanitize_stream_observation(obs) == obs
    assert wire._sanitize_stream_observation({**obs, "finalized_item_count": 0}) is None
    assert wire._sanitize_stream_observation({**obs, "terminal_output_state": "invalid"}) is None
