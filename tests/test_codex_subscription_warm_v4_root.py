"""Independent root protocol/privacy controls; never starts an App Server."""
from collections import deque
import hashlib
import json
from pathlib import Path
import queue

import pytest

from benchmarks import codex_subscription_warm_v4 as warm


PRIVATE = 'INVENTED_SECRET_MUST_NOT_LEAVE_SERVER'
VALUES = [None, True, False, 0, -1, 4097, 1.25, PRIVATE, [], {}, [PRIVATE], {PRIVATE: PRIVATE}]


def observation():
    return dict(basis='consumed_observed_shape', events_consumed=3,
                completed_seen=True, final_seen=True, final_count=1,
                usage_update_count=0, usage_state='absent', last_event_family='turn_completed')


@pytest.mark.parametrize('field', list(observation()))
@pytest.mark.parametrize('value', VALUES)
def test_every_json_field_shape_is_total_and_private_safe(field, value):
    payload = observation()
    payload[field] = value
    result = warm._project_observation(payload)
    assert PRIVATE not in json.dumps(result)
    result = warm.serialize_failure({'code': 'incomplete_turn_or_usage',
                                      'phase': 'run', 'rpc': 'turn/events',
                                      'turn_observation': payload})
    assert PRIVATE not in json.dumps(result)


def wire(monkeypatch, messages):
    s = object.__new__(warm.WarmSession)
    s.created_at = warm.time.monotonic()
    s.deadline = s.created_at + 30
    s.next_id = 0
    s.events = queue.Queue()
    s.pending = deque()
    s.stage = 'startup'
    s.active_thread = 'private-thread'
    s.active_turn = None
    s.last_event = None
    s.rpc_error = None
    s.request_id = None
    s.retired_threads = set()
    s.starting_events = []
    s.warning_targets = []
    s.initialized_result = None
    s.initialized_sent = True
    s.reset_turn_observation()
    for message in messages:
        s.events.put(message)

    def send(self, method, params, *, notification=False):
        self.next_id += 1
        return self.next_id

    monkeypatch.setattr(warm.base.StdioSession, 'send', send)
    return s


def ev(method, **fields):
    return {'method': method, 'params': {'threadId': 'private-thread',
                                        'turnId': 'private-turn', **fields}}


def test_real_rpc_queued_notifications_are_counted_only_when_parser_consumes(monkeypatch):
    start = ev('item/started', item={'id': 'private-item', 'type': 'agentMessage'})
    final = ev('item/completed', item={'id': 'private-item', 'type': 'agentMessage',
                                      'phase': 'final_answer', 'text': PRIVATE})
    done = ev('turn/completed', turn={'id': 'private-turn', 'status': 'completed'})
    response = {'id': 1, 'result': {'turn': {'id': 'private-turn', 'status': 'inProgress'}}}
    s = wire(monkeypatch, [start, final, response, done])
    with pytest.raises(warm.base.SubscriptionTransportError, match='incomplete_turn_or_usage'):
        warm.base._run_turn(s, 'private-thread', PRIVATE)
    assert s.next_id == 1
    assert warm._project_observation(s.turn_observation) == observation()
    assert s.last_event is None  # Existing diagnostic behavior stays unchanged.
    projected = warm.serialize_failure({'code': 'incomplete_turn_or_usage',
        'phase': 'run', 'rpc': 'turn/events', 'turn_observation': s.turn_observation})
    assert projected['turn_observation'] == observation()
    encoded = json.dumps(projected)
    assert PRIVATE not in encoded and 'private-' not in encoded


def test_source_and_budget_implementations_remain_the_pinned_originals():
    root = Path(__file__).resolve().parents[1] / 'benchmarks'
    pins = {
        'codex_subscription.py': '387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491',
        'codex_subscription_concurrent_v2.py': 'cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0',
        'codex_subscription_warm_v2.py': '9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593',
        'codex_subscription_warm_v3.py': '0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d',
    }
    for relative, expected in pins.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    assert warm.SharedBudget is warm.v3.SharedBudget
    assert warm.base._run_turn is warm.v3.base._run_turn
    assert warm.WarmSubscriptionClient.close is warm.v3.WarmSubscriptionClient.close
    assert warm.WarmSession.receive is warm.v3.WarmSession.receive
