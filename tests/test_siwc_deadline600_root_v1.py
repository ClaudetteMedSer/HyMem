"""Independent invented-grant checks for the entire credential/deadline boundary."""
from pathlib import Path
import socket
import sys
import time

import pytest

from benchmarks import chatgpt_plan_lme_v7 as old_bridge
from tools.diagnostics import lme_chatgpt_plan_refresh_v2 as old_refresh
from tests.test_lme_chatgpt_plan_owner_v1 import fixture_vm_state, write


def no_network(monkeypatch):
    def deny(*a, **kw):
        pytest.fail('network forbidden for invented grant regression')
    monkeypatch.setattr(socket, 'getaddrinfo', deny)
    monkeypatch.setattr(socket.socket, 'connect', deny)


def test_real_owner_reproduces_immediate_600_deadline_rejection(tmp_path, monkeypatch):
    no_network(monkeypatch)
    state = fixture_vm_state(tmp_path)
    with old_bridge.owner.CredentialBroker(state, Path(sys.executable)) as broker:
        with pytest.raises(old_bridge.owner.OwnerError) as caught:
            broker.acquire(caller_deadline=time.monotonic() + 600)
        assert caught.value.code == 'deadline_exceeded'


def test_real_refresh_630s_lifetime_is_not_due_despite_600_plus_60_requirement(tmp_path, monkeypatch):
    import json
    no_network(monkeypatch)
    state = fixture_vm_state(tmp_path)
    credential = json.loads((state / 'credential.json').read_text())
    credential['expires_at'] = int(time.time()) + 630
    write(state / 'credential.json', credential)
    result = old_refresh.run(state, check_only=True)
    assert result['status'] == 'not_due' and result['refresh_requests'] == 0
    assert 600 < result['expiry_remaining'] <= 630 < 600 + 60


def _deadline_response_child(send, slots, credentials, request, timeout):
    from benchmarks import chatgpt_plan_lme_v8 as bridge
    from tests.test_chatgpt_plan_responses_root_v1 import terminal
    assert 599 < timeout <= 600
    assert credentials.access_token in ('invented-access', 'invented-renewed-access')
    assert request['model'] == 'gpt-5.6-luna' and request['reasoning']['effort'] == 'low'
    assert request['stream'] is True and request['store'] is False
    progress = bridge.transport._Progress(slots, timeout)
    progress.mark('child_entry')
    reply = bridge.transport.parse_stream_events([terminal()])
    progress.mark('result_ipc', ready=True, ipc=True)
    send.send(('ok', reply)); send.close()


def test_real_broker_and_default_model_transport_four_workers(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    from benchmarks import chatgpt_plan_lme_v8 as bridge
    from tools.diagnostics import siwc_lme_diagnostic_v11 as runner
    no_network(monkeypatch)
    state = fixture_vm_state(tmp_path)
    children_before = {p.pid for p in multiprocessing.active_children()}
    monkeypatch.setattr(bridge.transport, '_child', _deadline_response_child)
    with bridge.owner.CredentialBroker(state, Path(sys.executable)) as broker:
        budget = bridge.SharedBudget(bridge.warm.BudgetLimits(*runner.MAX_LIMITS['campaign']))
        limits = bridge.warm.BudgetLimits(*runner.MAX_LIMITS['question'])
        clients = [bridge.SIWCLMEClient(broker, budget, str(i), limits) for i in range(4)]
        assert all(c.response_call is bridge.transport.complete for c in clients)
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(lambda c: c.complete(bridge.LLMRequest('invented s', 'invented u')), clients)) == [' invented\n'] * 4
        source = bridge.staged_v6.classification.GroundingSource(7, 'Mira uses CairnDB.',
            source_role='user', source_peer_id='invented', source_created_at='2026-09-30')
        triple = bridge.staged_v6.classification.Triple('Mira', 'uses', 'CairnDB', 1, source_message_id=7)
        request, batch = bridge.staged_v6.staged.build_original_request((triple,), (source,))
        structured = bridge.SIWCLMEClient(broker, runner.RegistrationAlias(budget, '0', limits), '0', limits)
        assert structured.complete_stage(request, batch, 'original', False) == ' invented\n'
        snap = budget.snapshot()
        assert snap['turns'] == 5 and snap['known_tokens'] == 65 and snap['usage_complete']
        assert snap['reserved'] == snap['in_flight'] == 0
        for c in [*clients, structured]:
            assert c.diagnostic_summary()['failures'] == 0
            c.close()
    assert {p.pid for p in multiprocessing.active_children()} == children_before


def test_real_owner_refresh_subprocess_630_lifetime_then_real_transport(tmp_path, monkeypatch):
    import json
    import subprocess
    from benchmarks import chatgpt_plan_lme_v8 as bridge
    no_network(monkeypatch)
    state = fixture_vm_state(tmp_path)
    value = json.loads((state / 'credential.json').read_text())
    value['expires_at'] = int(time.time()) + 630
    write(state / 'credential.json', value)
    original_popen = subprocess.Popen
    children = []
    bootstrap = '''
import sys,importlib.util,pathlib
def deny(event,args):
    if event in ('socket.connect','socket.getaddrinfo'): raise AssertionError('network forbidden')
sys.addaudithook(deny)
spec=importlib.util.spec_from_file_location('actual_refresh3',sys.argv[1])
refresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(refresh)
def invented_http(url,form=None):
    assert url=='https://auth.openai.com/api/accounts/oauth/token'
    assert form=={'grant_type':'refresh_token','client_id':'oaiapp_invented123',
        'refresh_token':'invented-refresh','resource':'https://api.openai.com/v1'}
    return {'token_type':'Bearer','expires_in':3600,'access_token':'invented-renewed-access','refresh_token':'invented-renewed-refresh'}
refresh._http_json=invented_http
raise SystemExit(refresh.main(['refresh','--state-dir',sys.argv[2]]))
'''
    def offline_popen(command, **kwargs):
        assert command[3].endswith('/lme_chatgpt_plan_refresh_v3.py')
        assert command[4:] == ['refresh', '--state-dir', str(state)]
        assert kwargs['start_new_session'] is True
        child = original_popen([sys.executable, '-I', '-B', '-c', bootstrap, command[3], str(state)], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(bridge.owner.subprocess, 'Popen', offline_popen)
    monkeypatch.setattr(bridge.transport, '_child', _deadline_response_child)
    with bridge.owner.CredentialBroker(state, Path(sys.executable)) as broker:
        budget = bridge.SharedBudget(bridge.warm.BudgetLimits(8012, 48160000, 25200))
        client = bridge.SIWCLMEClient(broker, budget, 'q', bridge.warm.BudgetLimits(2000, 12000000, 23400))
        # Refresh work consumes the same600s deadline, so use an injected response
        # callable only here to verify its remaining allowance after real renewal.
        forwarded = []
        def response(*a, timeout):
            forwarded.append(timeout)
            assert a[0].access_token == 'invented-renewed-access'
            return bridge.transport.Completed('invented', 2, 1, 3, 0, 0)
        client.response_call = response
        assert client.complete(bridge.LLMRequest('invented s', 'invented u')) == 'invented'
        assert client.complete(bridge.LLMRequest('invented s', 'invented u')) == 'invented'
        assert len(children) == 1 and children[0].poll() == 0
        assert len(forwarded) == 2 and all(595 < x <= 600 for x in forwarded)
        assert len(list(state.glob('.refresh-attempt-*.json'))) == 1
        assert budget.snapshot()['turns'] == 2 and budget.snapshot()['known_tokens'] == 6
        assert budget.snapshot()['usage_complete']


@pytest.mark.parametrize('lifetime,due', [(601, True), (630, True), (660, True), (662, False)])
def test_actual_refresh_cutoff_covers_lease_margin(tmp_path, monkeypatch, lifetime, due):
    import json
    from tools.diagnostics import lme_chatgpt_plan_refresh_v3 as refresh
    no_network(monkeypatch)
    state = fixture_vm_state(tmp_path)
    value = json.loads((state / 'credential.json').read_text())
    value['expires_at'] = int(time.time()) + lifetime
    write(state / 'credential.json', value)
    result = refresh.run(state, check_only=True)
    assert result['status'] == ('due' if due else 'not_due')
    assert result['refresh_requests'] == 0 and not list(state.glob('.refresh-attempt-*.json'))


def test_owner_real_600_and_above_cap_and_near_wall(tmp_path, monkeypatch):
    from benchmarks import chatgpt_plan_lme_v8 as bridge
    no_network(monkeypatch)
    state = fixture_vm_state(tmp_path)
    with bridge.owner.CredentialBroker(state, Path(sys.executable)) as broker:
        assert broker.acquire(caller_deadline=time.monotonic() + 600).policy == bridge.POLICY
        with pytest.raises(bridge.owner.OwnerError) as caught:
            broker.acquire(caller_deadline=time.monotonic() + 601)
        assert caught.value.code == 'deadline_exceeded'
        budget = bridge.SharedBudget(bridge.warm.BudgetLimits(10, 10000, 37))
        def response(*a, timeout):
            assert 36 < timeout <= 37
            return bridge.transport.Completed('invented', 2, 1, 3, 0, 0)
        client = bridge.SIWCLMEClient(broker, budget, 'q', bridge.warm.BudgetLimits(10, 10000, 36.9), response_call=response)
        assert client.complete(bridge.LLMRequest('invented s', 'invented u')) == 'invented'
        assert budget.snapshot()['reserved'] == budget.snapshot()['in_flight'] == 0
