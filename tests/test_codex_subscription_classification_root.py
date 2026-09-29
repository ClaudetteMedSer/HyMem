"""Root verification of actual inherited preflight/turn parsing with fake IO."""
from collections import deque
from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from benchmarks import codex_subscription_classification_v1 as c
from hymem.extraction import grounding_classification_v1 as g


class Protocol:
    instances = []
    failure = None
    foreign = False
    quota = 20
    entered = None
    release = None

    def __init__(self, binary, cwd, timeout=120):
        self.created_at = time.monotonic()
        self.calls = []
        self.pending = deque()
        self.starting_events = []
        self.retired_threads = set()
        self.closed = False
        self.stage = 'initialize'
        self.last_event = self.rpc_error = None
        self.active_thread = self.active_turn = None
        self.counter = 0
        self.__class__.instances.append(self)

    def set_deadline(self, deadline): self.deadline = deadline
    def send(self, method, params, **kwargs): self.calls.append((method, copy.deepcopy(params)))
    def bind_thread_id(self, tid): self.active_thread = tid
    def close(self): self.closed = True
    def unsubscribe(self, tid):
        self.stage = 'thread/unsubscribe'
        self.retired_threads.add(tid)
        self.active_thread = self.active_turn = None

    def rpc(self, method, params, **kwargs):
        self.stage = method
        self.calls.append((method, copy.deepcopy(params)))
        base = c.warm.base
        if self.failure == method:
            raise base.SubscriptionTransportError('rpc_failure:' + method)
        if method == 'initialize': return {'userAgent': 'codex/0.158.0'}
        if method == 'account/read': return {'account': {'type': 'chatgpt', 'planType': 'pro'}}
        if method == 'model/list':
            return {'data': [{'model': base.MODEL, 'supportedReasoningEfforts': [{'reasoningEffort': 'low'}]}]}
        if method == 'account/rateLimits/read':
            return {'rateLimits': {'primary': {'usedPercent': self.quota}}}
        if method == 'config/read':
            return {'config': {'forced_login_method': 'chatgpt', 'model_provider': 'openai',
                'model': base.MODEL, 'web_search': 'disabled', 'project_doc_max_bytes': 0,
                'memories': {'use_memories': False, 'generate_memories': False},
                'developer_instructions': '', 'features': {
                    **dict.fromkeys(base.DISABLED_FEATURES, False), 'skip_host_skill_discovery': True},
                'mcp_servers': {}}}
        if method == 'thread/start':
            self.counter += 1
            tid = f'thread-{self.counter}'
            return {'model': base.MODEL, 'modelProvider': 'openai', 'runtimeWorkspaceRoots': [],
                'instructionSources': [], 'sandbox': {'type': 'readOnly', 'networkAccess': False},
                'approvalPolicy': 'never', 'reasoningEffort': 'low', 'serviceTier': 'default',
                'thread': {'id': tid, 'ephemeral': True, 'environments': [], 'turns': [], 'path': None}}
        if method == 'turn/start':
            assert 'outputSchema' in params  # No unstructured fallback allowed.
            tid = params['threadId']
            turn = 'turn-' + str(self.counter)
            event_tid = 'foreign' if self.foreign else tid
            self.active_turn = turn
            self.pending.extend([
                {'method': 'turn/started', 'params': {'threadId': event_tid, 'turn': {'id': turn}}},
                {'method': 'thread/tokenUsage/updated', 'params': {'threadId': tid, 'turnId': turn,
                    'tokenUsage': {'total': {'totalTokens': 11}}}},
                {'method': 'item/started', 'params': {'threadId': tid, 'turnId': turn,
                    'item': {'id': 'item-1', 'type': 'agentMessage'}}},
                {'method': 'item/completed', 'params': {'threadId': tid, 'turnId': turn,
                    'item': {'id': 'item-1', 'type': 'agentMessage', 'phase': 'final_answer', 'text': '{}'}}},
                {'method': 'turn/completed', 'params': {'threadId': tid,
                    'turn': {'id': turn, 'status': 'completed'}}}])
            if self.entered is not None:
                self.entered.set()
                assert self.release.wait(3)
            return {'turn': {'id': turn, 'status': 'inProgress'}}
        raise AssertionError(method)

    def next_event(self):
        self.stage = 'turn/events'
        return self.pending.popleft()


@pytest.fixture(autouse=True)
def reset():
    Protocol.instances = []
    Protocol.failure = None
    Protocol.foreign = False
    Protocol.quota = 20
    Protocol.entered = Protocol.release = None


def request(object_='CairnDB'):
    return g.build_grounding_request(
        (g.Triple('Mira', 'uses', object_, 1, source_message_id=7),),
        (g.GroundingSource(7, f'Mira uses {object_}.'),))


def client(**kwargs):
    limits = c.warm.BudgetLimits(40, 10000, 1000)
    return c.ClassificationSubscriptionClient('unused', c.warm.SharedBudget(limits), 'q',
        limits, session_factory=Protocol, **kwargs)


def test_actual_preflight_and_turn_receive_exact_schema_without_policy_change():
    req, batch = request()
    item = client()
    assert item.complete_grounding(req, batch) == '{}'
    session = Protocol.instances[0]
    start = next(p for m, p in session.calls if m == 'thread/start')
    turn = next(p for m, p in session.calls if m == 'turn/start')
    assert start['baseInstructions'] == req.system
    assert start['allowProviderModelFallback'] is False
    assert start['dynamicTools'] == [] and start['ephemeral'] is True
    assert turn == {'threadId': 'thread-1', 'input': [{'type': 'text', 'text': req.user}],
        'model': c.warm.base.MODEL, 'effort': 'low', 'environments': [], 'runtimeWorkspaceRoots': [],
        'approvalPolicy': 'never', 'serviceTierForTurn': 'default',
        'sandboxPolicy': {'type': 'readOnly', 'networkAccess': False},
        'outputSchema': g.build_output_schema(batch)}
    assert item.observed_turns == 1 and item.observed_tokens == 11 and item.usage_complete
    assert item.requested_controls[-1]['max_tokens_effective'] is None
    assert session.retired_threads == {'thread-1'}
    item.close()
    assert session.closed and item.session is None


def test_warm_reuse_and_rotation_never_reuse_prior_schema():
    item = client(max_requests=2)
    batches = []
    for name in ('CairnDB', 'MapleDB', 'BirchDB'):
        req, batch = request(name)
        batches.append(batch)
        assert item.complete_grounding(req, batch) == '{}'
    assert len(Protocol.instances) == 2 and Protocol.instances[0].closed
    turns = [p for s in Protocol.instances for m, p in s.calls if m == 'turn/start']
    assert [p['outputSchema']['properties']['batch_sha256']['enum'][0] for p in turns] == [
        b.batch_sha256 for b in batches]
    assert len({json.dumps(p['outputSchema'], sort_keys=True) for p in turns}) == 3
    assert item.observed_tokens == 33 and item.observed_turns == 3
    assert item.rotations == 1
    item.close()
    assert all(s.closed for s in Protocol.instances)


@pytest.mark.parametrize('field,value', [('system', 'foreign'), ('user', 'foreign'),
    ('temperature', False), ('max_tokens', 8192), ('response_format', 'text')])
def test_request_tampering_fails_before_session_or_turn(field, value):
    req, batch = request()
    item = client()
    with pytest.raises(g.GroundingContractError, match='request:binding'):
        item.complete_grounding(replace(req, **{field: value}), batch)
    assert not Protocol.instances and item.observed_turns == 0
    item.close()


@pytest.mark.parametrize('method,known', [('account/read', True), ('thread/start', True),
    ('turn/start', False)])
def test_failed_dispatch_no_fallback_and_usage_unknown_after_admission(method, known):
    Protocol.failure = method
    item = client()
    with pytest.raises(c.warm.ConcurrentStop):
        item.complete_grounding(*request())
    assert Protocol.instances[0].closed
    turns = [p for s in Protocol.instances for m, p in s.calls if m == 'turn/start']
    assert len(turns) == int(method == 'turn/start')
    if turns: assert 'outputSchema' in turns[0]
    state = item.budget.snapshot()
    assert state['first_failure']['known_usage'] is known
    assert state['turns'] == int(method == 'turn/start')
    if not known: assert state['usage_complete'] is False
    item.close()


def test_foreign_turn_rejected_by_real_parser_and_cleans_up():
    Protocol.foreign = True
    item = client()
    with pytest.raises(c.warm.ConcurrentStop):
        item.complete_grounding(*request())
    state = item.budget.snapshot()
    assert state['first_failure']['code'] == 'turn_identity_mismatch'
    assert state['usage_complete'] is False
    assert Protocol.instances[0].closed
    item.close()


def test_quota_still_rejects_before_any_inference():
    Protocol.quota = 76
    item = client()
    with pytest.raises(c.warm.ConcurrentStop): item.complete_grounding(*request())
    assert item.observed_turns == 0
    assert not any(m == 'turn/start' for s in Protocol.instances for m, _ in s.calls)
    assert Protocol.instances[0].closed
    item.close()


def test_second_concurrent_request_cannot_overwrite_first_binding():
    Protocol.entered, Protocol.release = threading.Event(), threading.Event()
    item = client()
    first = request('CairnDB')
    second = request('MapleDB')
    errors = []
    def worker():
        try: item.complete_grounding(*first)
        except BaseException as exc: errors.append(type(exc).__name__)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert Protocol.entered.wait(3)
        with pytest.raises(c.warm.ConcurrentStop, match='concurrent_completion_rejected'):
            item.complete_grounding(*second)
    finally:
        Protocol.release.set()
        thread.join(3)
    assert not thread.is_alive()
    turns = [p for s in Protocol.instances for m, p in s.calls if m == 'turn/start']
    assert len(turns) == 1
    assert turns[0]['outputSchema'] == g.build_output_schema(first[1])
    assert turns[0]['input'][0]['text'] == first[0].user
    item.close()
    assert all(s.closed for s in Protocol.instances)


def test_bare_completion_is_never_a_schema_fallback():
    item = client()
    with pytest.raises(ValueError, match='classification_only'): item.complete(request()[0])
    assert not Protocol.instances and item.observed_turns == 0
    item.close()


def test_unbound_factory_cannot_create_unowned_process():
    item = client()
    with pytest.raises(c.warm.base.SubscriptionTransportError, match='invalid_request'):
        item.session_factory('unused', '/tmp/unused')
    assert not Protocol.instances
    item.close()


def test_invalid_request_after_warm_success_cleans_without_corrupting_prior_receipt():
    item = client()
    req, batch = request()
    item.complete_grounding(req, batch)
    receipt = copy.deepcopy(item.requested_controls)
    with pytest.raises(g.GroundingContractError, match='request:binding'):
        item.complete_grounding(replace(req, user='foreign'), batch)
    assert Protocol.instances[0].closed and item.session is None
    assert item._active_binding is None
    assert item.requested_controls == receipt
    assert item.observed_turns == 1 and item.observed_tokens == 11
    item.close()


def test_existing_session_bind_failure_clears_context_and_owns_cleanup():
    item = client()
    first = request()
    item.complete_grounding(*first)
    def fail(*args): raise ValueError('private-sentinel')
    object.__setattr__(item.session, 'bind', fail)
    with pytest.raises(c.warm.base.SubscriptionTransportError, match='invalid_request') as caught:
        item.complete_grounding(*request('OtherDB'))
    assert 'private-sentinel' not in str(caught.value)
    assert item._active_binding is None and item.session is None
    assert Protocol.instances[0].closed
    assert item.observed_turns == 1 and item.observed_tokens == 11
    item.close()


def test_new_wrapper_bind_failure_cleans_raw_before_ownership_transfer(monkeypatch):
    item = client()
    def fail(*args): raise ValueError('private-sentinel')
    monkeypatch.setattr(c._BoundSession, 'bind', fail)
    with pytest.raises(c.warm.ConcurrentStop) as caught:
        item.complete_grounding(*request())
    assert 'private-sentinel' not in str(caught.value)
    assert Protocol.instances[0].closed
    assert item._active_binding is None and item.session is None
    assert item.observed_turns == 0
    item.close()


def test_setup_cleanup_failure_retains_owner_for_explicit_cleanup_retry(monkeypatch):
    item = client()
    failed_close = [True]
    def reject_bind(*args): raise RuntimeError('private-sentinel')
    def close(raw):
        if failed_close[0]: raise RuntimeError('private-cleanup-sentinel')
        raw.closed = True
    monkeypatch.setattr(c._BoundSession, 'bind', reject_bind)
    monkeypatch.setattr(Protocol, 'close', close)
    with pytest.raises(c.warm.ConcurrentStop, match='cleanup_failure') as caught:
        item.complete_grounding(*request())
    assert 'private' not in str(caught.value)
    assert len(Protocol.instances) == 1 and not Protocol.instances[0].closed
    assert item._active_binding is None
    failed_close[0] = False
    item.close()
    assert Protocol.instances[0].closed
    assert item.budget.snapshot()['first_failure']['code'] == 'cleanup_failure'


@pytest.mark.parametrize('wrong_path', [True, False])
def test_wrong_cached_module_identity_is_rejected_before_session_creation(wrong_path):
    repo = Path(__file__).resolve().parents[1]
    module_path = repo / 'hymem/extraction/grounding_classification_v1.py'
    code = (
        'import sys,types,runpy;'
        f'sys.path.insert(0,{str(repo)!r});'
        'fake=types.ModuleType("hymem.extraction.grounding_classification_v1");'
        f'fake.__file__={str(module_path) if not wrong_path else "/private/foreign.py"!r};'
        'sys.modules[fake.__name__]=fake;'
        f'runpy.run_path({str(repo / "benchmarks/codex_subscription_classification_v1.py")!r})'
    )
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert 'pinned_classification_import_mismatch' in result.stderr
