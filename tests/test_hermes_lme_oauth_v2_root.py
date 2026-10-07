"""Root-owned account, quota, accounting and concurrency checks; no network."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import threading
import time

import pytest

from benchmarks import hermes_lme_oauth_v2 as bridge
from benchmarks import codex_subscription_staged_v6 as frozen
from hymem.extraction.llm import LLMRequest


def auth_file(path, account='invented-account', expires=None):
    claims = {'exp': expires or time.time() + 3600, 'email': 'invented@example.invalid',
              'https://api.openai.com/auth': {'chatgpt_account_id': account}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip('=')
    token = 'header.' + payload + '.signature'
    path.write_text(json.dumps({'auth_mode': 'chatgpt', 'OPENAI_API_KEY': None,
                               'tokens': {'access_token': token, 'account_id': account}}))
    path.chmod(0o600)


class MetadataSession:
    instances = []
    quota_denied = False

    def __init__(self, binary, cwd, timeout):
        self.created_at = time.monotonic()
        self.methods = []
        self.closed = False
        self.instances.append(self)

    def set_deadline(self, deadline):
        assert deadline > time.monotonic()

    def send(self, method, params, notification=False):
        self.methods.append(method)
        assert method == 'initialized' and notification

    def rpc(self, method, params):
        self.methods.append(method)
        assert method in {'initialize', 'account/read', 'model/list', 'account/rateLimits/read', 'config/read'}
        if method == 'account/read':
            return {'account': {'type': 'chatgpt', 'planType': 'pro', 'email': 'invented@example.invalid'}}
        if method == 'model/list':
            return {'data': [{'model': 'gpt-6-luna', 'supportedReasoningEfforts': [{'reasoningEffort': 'low'}]}], 'nextCursor': None}
        if method == 'account/rateLimits/read':
            return {'rateLimits': {'planType': 'pro', 'primary': {'usedPercent': 99},
                                  'spendControlReached': self.quota_denied,
                                  'credits': {'hasCredits': True, 'unlimited': False, 'balance': '2.5'}}}
        return {}

    def close(self):
        self.closed = True


def graph(tmp_path, *, workers=4):
    auth = tmp_path / 'invented-auth.json'
    auth_file(auth)
    broker = bridge.AdmissionBroker('/unused', str(auth), session_factory=MetadataSession)
    limits = bridge.warm.BudgetLimits(40, 10000, 300)
    budget = bridge.warm.SharedBudget(limits, max_in_flight=workers)
    return auth, broker, limits, budget


def test_same_credit_proof_identity_as_frozen_runner(tmp_path):
    assert bridge.warm is frozen.warm
    _, broker, cap, budget = graph(tmp_path)
    calls = []
    def transport(credentials, system, user, schema, *, timeout):
        calls.append((system, user, schema))
        return bridge.native.Completed('invented', 9, 4, 13, 3, 2)
    client = bridge.NativeLMEClient(broker, budget, 'q', cap, transport=transport)
    try:
        assert client.complete(LLMRequest(system='root system', user='root user')) == 'invented'
        state = budget.snapshot()
        assert state['known_tokens'] == 13 and state['turns'] == 1
        assert state['usage_complete'] and not state['stopped']
        assert calls == [('root system', 'root user', None)]
        assert not any('thread/' in name or 'turn/' in name for s in MetadataSession.instances for name in s.methods)
    finally:
        client.close()
        broker.close()


def test_account_change_between_calls_rejected_even_when_email_is_same(tmp_path):
    auth, broker, _, _ = graph(tmp_path)
    try:
        broker.admit(time.monotonic() + 10)
        auth_file(auth, account='other-invented-account')
        with pytest.raises(bridge.BridgeError, match='account_mismatch'):
            broker.admit(time.monotonic() + 10)
    finally:
        broker.close()


def test_explicit_quota_denial_prevents_any_http_call(tmp_path, monkeypatch):
    _, broker, cap, budget = graph(tmp_path)
    monkeypatch.setattr(MetadataSession, 'quota_denied', True)
    calls = []
    client = bridge.NativeLMEClient(broker, budget, 'q', cap, transport=lambda *a, **k: calls.append(1))
    try:
        with pytest.raises(BaseException):
            client.complete(LLMRequest(system='invented', user='invented'))
        state = budget.snapshot()
        assert calls == [] and state['turns'] == 0
        assert state['reserved'] == state['in_flight'] == 0 and state['stopped']
    finally:
        client.close()
        broker.close()


def test_failed_admitted_request_usage_is_unknown_not_zero(tmp_path):
    _, broker, cap, budget = graph(tmp_path)
    def fail(*args, **kwargs):
        raise bridge.native.TransportError('timeout')
    client = bridge.NativeLMEClient(broker, budget, 'q', cap, transport=fail)
    try:
        with pytest.raises(BaseException):
            client.complete(LLMRequest(system='invented', user='invented'))
        state = budget.snapshot()
        assert state['turns'] == 1 and state['known_tokens'] == 0
        assert state['usage_complete'] is False and state['stopped']
        assert state['reserved'] == state['in_flight'] == 0
        assert state['first_failure'] == {'code': 'timeout', 'phase': 'http',
            'turn_admitted': True, 'known_usage': False}
        summary = client.diagnostic_summary()
        assert summary['calls'] == summary['failures'] == 1
        assert summary['successes'] == 0
        assert summary['first_failure']['code'] == 'timeout'
        assert summary['timing_saturated'] is False
    finally:
        client.close()
        broker.close()


def test_admission_deadline_expiry_never_dispatches_http(tmp_path):
    _, broker, _, _ = graph(tmp_path)
    cap = bridge.warm.BudgetLimits(2, 1000, 0.01)
    budget = bridge.warm.SharedBudget(cap, max_in_flight=1)
    calls = []
    original = broker.admit
    def late(deadline):
        credentials, admission = original(deadline)
        time.sleep(0.02)
        return credentials, admission
    broker.admit = late
    client = bridge.NativeLMEClient(broker, budget, 'q', cap,
        transport=lambda *a, **k: calls.append(1))
    try:
        with pytest.raises(BaseException):
            client.complete(LLMRequest(system='invented', user='invented'))
        state = budget.snapshot()
        assert calls == [] and state['turns'] == 0
        assert state['usage_complete'] is True
        assert state['reserved'] == state['in_flight'] == 0
    finally:
        client.close()
        broker.close()


def test_watchdog_unblocks_and_reaps_actual_metadata_child(tmp_path, monkeypatch):
    managed = tmp_path / 'managed'
    managed.mkdir()
    auth = managed / 'auth.json'
    auth_file(auth)
    monkeypatch.setenv('CODEX_HOME', str(managed))
    children = []
    class BlockedSession(MetadataSession):
        def __init__(self, binary, cwd, timeout):
            super().__init__(binary, cwd, timeout)
            self.process = subprocess.Popen([sys.executable, '-I', '-B', '-c',
                'import time; time.sleep(10)'], stdout=subprocess.PIPE,
                start_new_session=True)
            children.append(self.process)
        def rpc(self, method, params):
            self.process.stdout.readline()
            raise OSError('invented blocked metadata pipe')
        def close(self):
            self.process.wait(timeout=1)
            self.process.stdout.close()
            self.closed = True
    monkeypatch.setattr(bridge.warm, 'WarmSession', BlockedSession)
    broker = bridge.AdmissionBroker('/unused', str(auth), session_factory=BlockedSession)
    start = time.monotonic()
    try:
        with pytest.raises(bridge.BridgeError, match='^timeout$'):
            broker.admit(start + 0.15)
        assert time.monotonic() - start < 1.5
        assert len(children) == 1 and children[0].poll() is not None
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)
        broker.close()


def test_four_http_requests_can_run_concurrently_after_serial_admission(tmp_path):
    _, broker, cap, budget = graph(tmp_path)
    barrier = threading.Barrier(4)
    def transport(*args, **kwargs):
        barrier.wait(timeout=3)
        return bridge.native.Completed('invented', 9, 4, 13, 3, 2)
    clients = [bridge.NativeLMEClient(broker, budget, str(i), cap, transport=transport) for i in range(4)]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            result = list(pool.map(lambda c: c.complete(LLMRequest(system='s', user='u')), clients))
        assert result == ['invented'] * 4
        state = budget.snapshot()
        assert state['turns'] == 4 and state['known_tokens'] == 52
        assert state['reserved'] == state['in_flight'] == 0
        assert state['usage_complete'] and not state['stopped']
    finally:
        for client in clients:
            client.close()
        broker.close()
