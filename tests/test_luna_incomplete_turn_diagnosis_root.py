"""Root reproduction of the frozen parser's diagnostic ambiguity, no inference."""
import ast
import hashlib
from pathlib import Path
from typing import Any

import pytest


FROZEN = Path('/private/tmp/hymem-lme-diagnostic-offline-assembly-v4/code/benchmarks/codex_subscription.py')


def parser():
    data = FROZEN.read_bytes()
    # Match the original source pin accepted by every pilot, not a new parser.
    from benchmarks import codex_subscription as baseline
    assert hashlib.sha256(data).digest() == hashlib.sha256(Path(baseline.__file__).read_bytes()).digest()
    selected = [node for node in ast.parse(data).body
                if isinstance(node, ast.FunctionDef) and node.name in {'_run_turn', '_observed_usage'}]
    assert len(selected) == 2
    namespace = dict(Any=Any, MODEL=baseline.MODEL, MAX_EVENTS=baseline.MAX_EVENTS,
                     MAX_OUTPUT_CHARS=baseline.MAX_OUTPUT_CHARS,
                     _fail=baseline._fail, _safe_method=baseline._safe_method)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(FROZEN), 'exec'), namespace)
    return namespace['_run_turn'], baseline.SubscriptionTransportError


def event(method, **fields):
    return {'method': method, 'params': {'threadId': 'thread', 'turnId': 'turn', **fields}}


START = event('item/started', item={'id': 'answer', 'type': 'agentMessage'})
FINAL = event('item/completed', item={'id': 'answer', 'type': 'agentMessage',
                                    'phase': 'final_answer', 'text': 'invented'})
USAGE = event('thread/tokenUsage/updated', tokenUsage={'total': {'totalTokens': 42}})
DONE = event('turn/completed', turn={'id': 'turn', 'status': 'completed'})


class Events:
    def __init__(self, values):
        self.values = list(values)
        self.consumed = 0

    def rpc(self, method, params, **kwargs):
        assert method == 'turn/start'
        return {'turn': {'id': 'turn', 'status': 'inProgress'}}

    def next_event(self):
        value = self.values[self.consumed]
        self.consumed += 1
        return value


def test_valid_frozen_sequence():
    run, _ = parser()
    assert run(Events([START, FINAL, USAGE, DONE]), 'thread', 'invented') == ('invented', 42)


@pytest.mark.parametrize('sequence', [
    [START, FINAL, DONE],
    [USAGE, DONE],
    [START, FINAL, event('thread/tokenUsage/updated', tokenUsage={'total': {'totalTokens': 0}}), DONE],
    [event('thread/status/changed')] * 4096,
])
def test_distinct_faults_collapse_to_same_unannotated_error(sequence):
    run, error = parser()
    with pytest.raises(error) as exc:
        run(Events(sequence), 'thread', 'invented')
    assert exc.value.args == ('incomplete_turn_or_usage',)
    assert vars(exc.value) == {}


def test_postcompletion_usage_is_not_consumed_no_claim_this_is_live_order():
    run, error = parser()
    session = Events([START, FINAL, DONE, USAGE])
    with pytest.raises(error, match='incomplete_turn_or_usage'):
        run(session, 'thread', 'invented')
    assert session.consumed == 3
