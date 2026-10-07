"""Independent no-network stream/reader contract checks for the600s experiment."""
import copy
import io
import json
import multiprocessing
import time
from types import SimpleNamespace

import pytest

from benchmarks import chatgpt_plan_responses_v11 as wire
from tests.test_chatgpt_plan_responses_root_v1 import terminal


def observed(events):
    body = b': invented keepalive\n\n' + b''.join(
        b'data: ' + json.dumps(event).encode() + b'\n\n' for event in events)
    slots = multiprocessing.get_context('spawn').RawArray('q', wire._SLOTS)
    progress = wire._Progress(slots, 600)
    decoded = list(wire._events(io.BytesIO(body), None, progress))
    value = wire._snapshot(slots, 600, 'result_wait', time.monotonic(), False)
    assert len(decoded) == len(events) and value['wire_bytes'] == len(body)
    return value, slots, progress


def test_long_fragmented_prefix_has_exact_disjoint_buckets_and_unchanged_parser():
    events = ([{'type': 'response.output_text.delta', 'delta': 'INVENTED-PRIVATE'}] * 10001
        + [{'type': 'response.reasoning_text.delta', 'delta': 'INVENTED-PRIVATE'}] * 4001
        + [{'type': 'response.reasoning_summary_text.delta', 'delta': 'INVENTED-PRIVATE'}] * 2429
        + [{'type': 'response.in_progress'}, terminal()])
    value, _, _ = observed(events)
    assert value['event_count'] == 16433
    assert value['event_buckets'] == {'output_text_delta': 10001,
        'reasoning_text_delta': 4001, 'reasoning_summary_text_delta': 2429,
        'lifecycle': 1, 'completion': 1, 'failure': 0, 'other': 0}
    assert value['completion_seen'] and value['snapshot_valid']
    assert 'INVENTED-PRIVATE' not in repr(value)
    assert wire.parse_stream_events(events) == wire._v6.parse_stream_events(events)
    with pytest.raises(wire.TransportError, match='missing_completion'):
        wire.parse_stream_events(events[:-1])


def test_failure_unknown_malformed_types_are_counted_without_disclosure():
    value, _, _ = observed([{'type': item, 'id': 'INVENTED-PRIVATE'} for item in
        ['response.failed', 'response.incomplete', 'error', 'PRIVATE-EVENT', {}, [], None, 3]])
    assert value['event_buckets']['failure'] == 3 and value['event_buckets']['other'] == 5
    assert sum(value['event_buckets'].values()) == value['event_count'] == 8
    assert not value['completion_seen']
    assert 'PRIVATE' not in repr(value)


@pytest.mark.parametrize('mutation', ['extra', 'negative', 'bool', 'huge', 'sum', 'completion', 'none'])
def test_reader_and_transport_reject_same_bucket_corruption(mutation):
    from tools.diagnostics import siwc_lme_diagnostic_progress_v13 as reader
    value, _, _ = observed([{'type': 'response.in_progress'}])
    reader.timeout_observation(value)
    bad = copy.deepcopy(value)
    if mutation == 'extra': bad['event_buckets']['PRIVATE-TYPE'] = 0
    elif mutation == 'negative': bad['event_buckets']['other'] = -1
    elif mutation == 'bool': bad['event_buckets']['other'] = False
    elif mutation == 'huge': bad['event_buckets']['other'] = 10**30
    elif mutation == 'sum': bad['event_buckets']['other'] = 1
    elif mutation == 'completion': bad['completion_seen'] = True
    elif mutation == 'none': bad['event_buckets'] = None
    assert wire.sanitize_timeout_observation(bad) is None
    with pytest.raises(ValueError, match='timeout_observation_invalid'):
        reader.timeout_observation(bad)


def test_torn_and_saturated_counters_are_unknown_and_copies_are_independent():
    from tools.diagnostics import siwc_lme_diagnostic_progress_v13 as reader
    value, slots, progress = observed([{'type': 'response.created'}])
    copied = wire.sanitize_timeout_observation(value)
    copied['event_buckets']['other'] = 100
    assert value['event_buckets']['other'] == 0
    slots[0] += 1
    torn = wire._snapshot(slots, 600, 'result_wait', time.monotonic(), None)
    assert not torn['snapshot_valid'] and torn['event_count'] is None and torn['event_buckets'] is None
    reader.timeout_observation(torn)
    slots[0] += 1
    progress.mark('parse', wire=wire.MAX_WIRE_BYTES)
    saturated = wire._snapshot(slots, 600, 'result_wait', time.monotonic(), None)
    assert saturated['event_buckets'] is None
    reader.timeout_observation(saturated)


@pytest.mark.parametrize('campaign,question,expected', [(25200,23400,593), (600,600,593), (37,900,30), (900,37,30)])
def test_admission_and_canary_share_same_outer_deadline(campaign, question, expected, monkeypatch):
    from benchmarks import chatgpt_plan_lme_v8 as bridge
    from tests.test_siwc_reservation_deadline_root_v1 import broker_for
    clock = [100.0]
    fake = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 2000.0)
    monkeypatch.setattr(bridge, 'time', fake)
    def acquire(self, *, caller_deadline):
        assert caller_deadline == 100 + min(600, campaign, question)
        clock[0] += 7
        return bridge.owner.CredentialLease('invented-not-a-token', 3000)
    broker = broker_for(bridge, monkeypatch, acquire)
    seen = []
    def response(*args, timeout):
        seen.append(timeout)
        if campaign == question == 600:
            clock[0] += 100
        return bridge.transport.Completed('invented', 2, 1, 3, 0, 0)
    budget = bridge.SharedBudget(bridge.warm.BudgetLimits(12,160000,campaign), clock=fake.monotonic)
    client = bridge.SIWCLMEClient(broker,budget,'q',bridge.warm.BudgetLimits(12,160000,question),response_call=response)
    assert client.complete(bridge.LLMRequest('invented s','invented u')) == 'invented'
    assert seen == [expected]
    assert budget.snapshot()['reserved'] == budget.snapshot()['in_flight'] == 0
    if campaign == question == 600:
        assert client.complete(bridge.LLMRequest('invented s','invented u')) == 'invented'
        assert seen == [593, 486]
