"""Real offline SQLite controls for R8 replay, budgets and isolation."""
import importlib.util
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('HYMEM_R8_SUMMARY_SOURCE',
    '/private/tmp/hymem-r8-sequential-20260926.MAx3h1/candidate')).resolve()
sys.path.insert(0, str(SOURCE))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


replay = module('r8_summary_replay_test', ROOT/'tools/diagnostics/lme_r8_summary_replay.py')
r6 = module('r8_summary_r6_test', ROOT/'tools/diagnostics/lme_r6_summary_replay.py')
support = module('r8_summary_support_test', ROOT/'tools/diagnostics/lme_summary_recovery_v1/worker.py')
api = support.import_api(SOURCE)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('offline diagnostic attempted network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)


class Client:
    def __init__(self, response=None):
        self.calls, self.request_attempts = [], 0
        self.response = response

    def memory_producer_declaration(self):
        raise NotImplementedError

    def complete(self, request):
        self.calls.append(request)
        self.request_attempts += 1
        if self.response is not None:
            return self.response
        if not request.user.startswith('{"new_material":'):
            return json.dumps({'episodes': [], 'procedures': [],
                'summary': 'The invented source remains retained and processing is pending.'})
        return json.dumps({'summary': 'The invented source remains retained and processing is pending.'})


def budget(client, cap=192):
    from hymem.deadline import MonotonicDeadline
    return replay.SharedBudget(client, MonotonicDeadline.after(60), cap=cap)


@pytest.fixture
def controls(tmp_path, monkeypatch):
    path = tmp_path.resolve()/'controls.sqlite'
    monkeypatch.setattr(r6, 'CONTROL', path)
    r6.seed_controls(api)
    conn = api.db.connect(path)
    yield conn
    conn.close()


@pytest.fixture
def retained(tmp_path):
    from hymem.dreaming.digest import digest_config_version
    from hymem.dreaming.lossless import materialize_message_coverage
    from hymem.session import open_session, append_message
    path = tmp_path.resolve()/'reference.sqlite'
    conn = api.db.connect(path)
    api.db.initialize(conn)
    for number in range(14):
        sid = f'invented-retained-{number:02}'
        open_session(conn, sid)
        append_message(conn, sid, 'user', 'Keep the original data; processing has not started.')
        last = append_message(conn, sid, 'assistant', 'I proposed inventory only; no action is complete.')
        with api.db.transaction(conn):
            materialize_message_coverage(conn, sid)
        generation = digest_config_version(prompt_version='v1', episode_prompt_version=None,
            max_chars=8000, max_tokens=3072, max_episodes=None)+'|walk='+support.digest(sid)[:32]
        conn.execute('UPDATE sessions SET digest_published_generation=?,digest_published_message_id=?,'
            'digest_cursor_prompt_version=?,digest_cursor_message_id=?,digested_message_id=?,'
            "digested_prompt_version='v1',summary_failure_reason='summary_output_cap',summary_failure_count=1 WHERE id=?",
            (generation, last, generation, last, last, sid))
    yield conn, path
    conn.close()


def test_four_actual_controls_walk_normal_digest_without_publication(controls, monkeypatch):
    from hymem.dreaming import digest
    original_extract = digest.extract_session_digest
    params = []
    def inspect_parameters(*args, **kwargs):
        params.append({k: kwargs[k] for k in replay.NORMAL_PARAMETERS})
        return original_extract(*args, **kwargs)
    monkeypatch.setattr(digest, 'extract_session_digest', inspect_parameters)
    client = Client()
    before = support.snapshot(controls)
    result = replay.normal_walk(controls, sorted(r6.control_cases()), budget(client), api, support, invented=True)
    assert result['complete_sessions'] == 4
    assert support.snapshot(controls)['full_sha256'] == before['full_sha256']
    assert all(item['summary'] and item['summary_chars'] <= 500 for item in result['sessions'])
    # This actual fixture forces repeated full-source windows and continuity.
    multi = next(item for item in result['sessions'] if item['session_id'] == 'control-multiwindow')
    assert multi['windows'] > 1
    assert len(client.calls) == sum(item['windows'] for item in result['sessions'])
    assert all('procedures' in request.system for request in client.calls)
    assert all(value == replay.NORMAL_PARAMETERS for value in params)


def test_fourteen_normal_walks_are_read_only_and_never_export_retained_text(retained, tmp_path):
    conn, path = retained
    ids = replay.targets(conn, api)
    before = support.snapshot(conn)
    readonly = api.db.connect(path)
    readonly.execute('PRAGMA query_only=ON')
    try:
        client = Client()
        private = tmp_path.resolve()/'private'
        private.mkdir()
        result = replay.normal_walk(readonly, ids, budget(client), api, support, private)
        assert result['complete_sessions'] == 14 and len(client.calls) == 14
        exported = json.dumps(result)
        assert 'invented-retained' not in exported and 'source remains retained' not in exported
        assert len(json.loads((private/'normal-context.json').read_text())) == 14
    finally:
        readonly.close()
    assert support.snapshot(conn)['full_sha256'] == before['full_sha256']


def test_retained_recovery_uses_backup_and_changes_only_summary_state(retained, tmp_path, monkeypatch):
    reference, path = retained
    clone_path = tmp_path.resolve()/'clone.sqlite'
    monkeypatch.setattr(r6, 'REFERENCE', path)
    monkeypatch.setattr(r6, 'CLONE', clone_path)
    before = support.snapshot(reference)
    r6.clone_reference()
    clone = api.db.connect(clone_path)
    try:
        client = Client()
        result = replay.explicit_recovery(clone, budget(client), api, support, replay.targets(clone, api))
        assert result['recovered_all'] and result['recovery']['published'] == 14
        assert result['all_non_summary_state_unchanged'] and len(client.calls) == 14
        assert support.snapshot(reference)['full_sha256'] == before['full_sha256']
        support.assert_unchanged(before, support.snapshot(clone))
        assert clone.execute('SELECT COUNT(*) FROM summary_recovery').fetchone()[0] == 0
    finally:
        clone.close()


def test_normal_rejection_holds_once_without_reroll(controls):
    client = Client('bad json')
    result = replay.normal_walk(controls, ['control-dense'], budget(client), api, support, invented=True)
    assert result['complete_sessions'] == 0
    assert result['sessions'][0]['failure_reasons'] == ['parse_failure']
    assert len(client.calls) == 1


def test_normal_empty_effective_tail_is_not_counted_complete(controls):
    client = Client(json.dumps({'episodes': [], 'procedures': [], 'summary': ''}))
    result = replay.normal_walk(controls, ['control-dense'], budget(client), api, support, invented=True)
    assert result['complete_sessions'] == 0
    assert result['sessions'][0]['failure_reasons'] == ['summary_validation_failure']


def test_separated_summary_failure_is_held_even_with_valid_items(controls):
    # Primary cap plus unusable repair leaves valid empty item arrays but an
    # explicit summary gap in the normal separated-publication API.
    client = Client(json.dumps({'episodes': [], 'procedures': [], 'summary': 'x'*501}))
    result = replay.normal_walk(controls, ['control-dense'], budget(client), api, support, invented=True)
    assert result['complete_sessions'] == 0
    assert result['sessions'][0]['failure_reasons']
    assert len(client.calls) == 2  # One normal call plus its stock bounded repair.


def test_real_transport_producer_and_attempt_scopes_delegate_without_network():
    client = api.client(api_key='synthetic-preflight-not-a-key', base_url=replay.ENDPOINT,
                        model=replay.MODEL, thinking='disabled')
    try:
        wrapped = budget(client)
        assert wrapped.model == replay.MODEL and wrapped.base_url == replay.ENDPOINT
        assert wrapped.transport_integrity_ok is True
        assert wrapped.effective_extra_body == {'thinking': {'type': 'disabled'}}
        original = api.producer_binding(client, declaration_hook='memory_producer_declaration')
        assert original['identity_exact'] is True
        declared = api.producer_binding(wrapped, declaration_hook='memory_producer_declaration')
        assert declared['identity_exact'] is True and declared['identity_sha256'] != original['identity_sha256']
        assert declared['declaration']['client_id'] == 'hymem.diagnostics.r8.SharedBudget'
        for field in ('model', 'endpoint_origin', 'endpoint_sha256', 'effective_request', 'retry_policy'):
            assert declared['declaration'][field] == original['declaration'][field]
        with wrapped.track_provider_attempts() as scope:
            assert scope.attempts == 0
        assert wrapped.calls == client.request_attempts == 0
        assert api.usage_snapshot(client)['calls'] == 0
    finally:
        client.close()


def test_private_capture_matches_fake_http_bodies_without_headers_or_identity_drift(tmp_path, monkeypatch):
    import httpx
    import openai
    from hymem.deadline import MonotonicDeadline
    from hymem.extraction.llm import LLMRequest
    wire = []
    reply_body = json.dumps({'id': 'invented', 'object': 'chat.completion', 'created': 0,
        'model': replay.MODEL, 'choices': [{'index': 0, 'finish_reason': 'stop',
        'message': {'role': 'assistant', 'content': 'PRIVATE-RETAINED-REPLY'}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
    def transport(request):
        wire.append(request.read())
        return httpx.Response(200, content=reply_body, headers={'X-Private-Header': 'SECRET-HEADER'}, request=request)
    factory = openai.DefaultHttpxClient
    monkeypatch.setattr(openai, 'DefaultHttpxClient', lambda **kw:
        factory(**kw, transport=httpx.MockTransport(transport)))
    installed = openai.DefaultHttpxClient
    capture = replay.WireCapture(tmp_path.resolve(), MonotonicDeadline.after(60))
    capture.phase, capture.session_sha256 = 'normal_retained', 'a'*64
    client = replay.capture_client(api, 'synthetic-preflight-not-a-key', capture)
    try:
        assert openai.DefaultHttpxClient is installed
        assert client.transport_integrity_ok is True
        original = api.producer_binding(client, declaration_hook='memory_producer_declaration')
        wrapped = replay.SharedBudget(client, capture.deadline, capture=capture)
        assert wrapped.complete(LLMRequest(system='Invented system', user='PRIVATE-RETAINED-SOURCE')) == 'PRIVATE-RETAINED-REPLY'
        assert client.transport_integrity_ok is True
        assert api.producer_binding(client, declaration_hook='memory_producer_declaration') == original
        request = json.loads((tmp_path/'001-request.json').read_text())
        response = json.loads((tmp_path/'001-response.json').read_text())
        assert base64.b64decode(request['body_base64']) == wire[0]
        assert base64.b64decode(response['body_base64']) == reply_body
        assert request['phase'] == response['phase'] == 'normal_retained'
        assert request['session_sha256'] == response['session_sha256'] == 'a'*64
        assert request['body_sha256'] == hashlib.sha256(wire[0]).hexdigest()
        exported = json.dumps(capture.public_inventory())
        assert 'PRIVATE-RETAINED' not in exported and 'SECRET-HEADER' not in exported
        assert 'SECRET-HEADER' not in json.dumps(response)
        assert 'synthetic-preflight-not-a-key' not in json.dumps(request)
        assert capture.attempts == client.request_attempts == wrapped.calls == capture.responses == 1
    finally:
        client.close()
    assert client._closed is True


def test_capture_http_cap_rejects_before_next_dispatch(tmp_path):
    import httpx
    from hymem.deadline import DeadlineExceeded, MonotonicDeadline
    capture = replay.WireCapture(tmp_path.resolve(), MonotonicDeadline.after(60))
    capture.attempts = replay.BOUNDS['http_attempts']
    request = httpx.Request('POST', replay.ENDPOINT+'/chat/completions',
        json={'model': replay.MODEL, 'thinking': {'type': 'disabled'}})
    with pytest.raises(DeadlineExceeded):
        capture.request(request)
    assert capture.attempts == replay.BOUNDS['http_attempts']
    assert list(tmp_path.iterdir()) == []


def test_recovery_capture_labels_each_actual_stock_session_and_accounts_attempts(retained, tmp_path, monkeypatch):
    import httpx
    import openai
    from hymem.deadline import MonotonicDeadline
    conn, _ = retained
    ids = replay.targets(conn, api)
    body = {'id': 'invented', 'object': 'chat.completion', 'created': 0, 'model': replay.MODEL,
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant',
            'content': json.dumps({'summary': 'The invented source remains retained and processing is pending.'})}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
    factory = openai.DefaultHttpxClient
    monkeypatch.setattr(openai, 'DefaultHttpxClient', lambda **kw: factory(**kw,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body, request=request))))
    private = tmp_path.resolve()/'capture'
    private.mkdir()
    capture = replay.WireCapture(private, MonotonicDeadline.after(60))
    client = replay.capture_client(api, 'synthetic-preflight-not-a-key', capture)
    try:
        wrapped = replay.SharedBudget(client, capture.deadline, capture=capture)
        result = replay.explicit_recovery(conn, wrapped, api, support, ids)
        assert result['recovered_all']
        assert capture.recovery_conn is None
        assert capture.attempts == capture.responses == client.request_attempts == wrapped.calls == 14
        assert result['recovery']['provider_attempts'] == 14
        exported = capture.public_inventory()
        assert {entry['phase'] for entry in exported['files'].values()} == {'explicit_recovery'}
        assert {entry['session_sha256'] for entry in exported['files'].values()} == {
            hashlib.sha256(sid.encode()).hexdigest() for sid in ids}
        assert client.transport_integrity_ok is True
    finally:
        client.close()


def test_shared_budget_caps_cross_phase_calls_and_preserves_deadline():
    from hymem.deadline import DeadlineExceeded, MonotonicDeadline
    from hymem.extraction.llm import LLMRequest
    client = Client()
    wrapped = budget(client, cap=2)
    request = LLMRequest(system='summary', user='invented')
    wrapped.complete(request)
    wrapped.complete(request)
    with pytest.raises(DeadlineExceeded):
        wrapped.complete(request)
    assert wrapped.calls == len(client.calls) == 2
    client.request_attempts = 575
    with pytest.raises(DeadlineExceeded):
        budget(client).complete(request)
    deadline = MonotonicDeadline(1, clock=lambda: 2)
    with pytest.raises(DeadlineExceeded):
        replay.SharedBudget(client, deadline).complete(request)
    assert len(client.calls) == 2


def test_real_backup_refuses_reuse(retained, tmp_path, monkeypatch):
    _, path = retained
    monkeypatch.setattr(r6, 'REFERENCE', path)
    monkeypatch.setattr(r6, 'CLONE', tmp_path.resolve()/'clone.sqlite')
    r6.clone_reference()
    with pytest.raises(FileExistsError):
        r6.clone_reference()


def test_full_offline_worker_creates_real_clones_closes_transport_and_spends_zero(
        retained, tmp_path, monkeypatch):
    reference, path = retained
    reference.close()
    work, results = tmp_path.resolve()/'work', tmp_path.resolve()/'results'
    work.mkdir()
    results.mkdir()
    monkeypatch.setattr(replay, 'REFERENCE', path)
    monkeypatch.setattr(replay, 'CLONE', work/'hymem.sqlite')
    monkeypatch.setattr(replay, 'CONTROL', work/'control.sqlite')
    monkeypatch.setattr(replay, 'RESULTS', results)
    monkeypatch.setattr(replay, 'SOURCE', SOURCE)
    monkeypatch.setattr(r6, 'REFERENCE', path)
    monkeypatch.setattr(r6, 'CLONE', replay.CLONE)
    monkeypatch.setattr(r6, 'CONTROL', replay.CONTROL)
    monkeypatch.setattr(replay, 'verify', lambda _: {'r6_summary_sha256': replay.sha(Path(r6.__file__))})
    monkeypatch.setattr(replay, 'previous_request', lambda: ({'messages': [
        {'role': 'system', 'content': api.recovery.SUMMARY_RECOVERY_SYSTEM},
        {'role': 'user', 'content': '{"prior_summary":"","new_material":"Invented offline fixture."}'}],
        'temperature': 0.0, 'max_tokens': 3072}, 'a'*64))
    monkeypatch.setattr(replay, 'load', lambda path, name, pin: support if path.name == 'worker.py' else r6)
    monkeypatch.setattr(replay.importlib.metadata, 'version', lambda name: support.RUNTIME[name])
    monkeypatch.setattr(replay.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(replay.os, 'environ', dict(os.environ))
    previous_mask = os.umask(0o077)
    try:
        assert replay.worker('offline', 'a'*64) == 0
    finally:
        os.umask(previous_mask)
    proof = json.loads((results/'offline.json').read_text())
    assert proof['status'] == 'offline_passed'
    assert proof['completion_calls'] == proof['provider_calls'] == proof['http_attempts'] == 0
    assert proof['clients_closed'] and proof['connections_closed'] and proof['threads_clean']
    assert proof['reference_unchanged'] and proof['source_unchanged']
    assert proof['previous_pins_verified'] == replay.PREVIOUS_PINS
    clone = api.db.connect(replay.CLONE)
    control = api.db.connect(replay.CONTROL)
    try:
        assert len(replay.targets(clone, api)) == 14
        assert r6.verify_controls(control) == proof['control_fixture_sha256']
    finally:
        clone.close()
        control.close()


def test_offline_source_inventory_includes_sql_resources_and_tests(tmp_path):
    for name in ('schema.sql', 'resource.json', 'test_control.py', '__pycache__/junk.pyc'):
        path = tmp_path.resolve()/name
        path.parent.mkdir(exist_ok=True)
        path.write_text('invented')
    assert set(replay.inventory(tmp_path.resolve())) == {'schema.sql', 'resource.json', 'test_control.py'}
    (tmp_path/'source_link').symlink_to(tmp_path/'schema.sql')
    with pytest.raises(RuntimeError, match='source_special_file'):
        replay.inventory(tmp_path.resolve())


class AlternativesClient(Client):
    def complete(self, request):
        self.calls.append(request)
        self.request_attempts += 1
        return json.dumps({'alternatives': [
            'Aster proposed keeping the archive unchanged until audit approval.',
            'The archive remains unchanged until approval.', 'Audit approval remains pending.']})


def test_repair_controls_use_real_windows_and_selected_continuity_without_db_writes(controls):
    client = AlternativesClient()
    before = support.snapshot(controls)
    result = replay.repair_contract_controls(controls, sorted(r6.control_cases()), budget(client),
                                             api, support, r6.control_cases())
    assert result['complete_sessions'] == 4 and result['synthetic_primary_calls'] == 0
    assert result['verification_kind'] == 'direct_repair_prompt_contract_not_stock_invocation'
    assert support.snapshot(controls)['full_sha256'] == before['full_sha256']
    multi = next(item for item in result['sessions'] if item['session_id'] == 'control-multiwindow')
    assert len(multi['windows']) > 1
    assert len(client.calls) == sum(len(item['windows']) for item in result['sessions'])
    inputs = [json.loads(json.loads(request.user)['original_generation_input']) for request in client.calls]
    assert all(set(item) == {'prior_summary', 'new_material'} for item in inputs)
    assert any(item['prior_summary'] == multi['windows'][0]['selected_overview'] for item in inputs)
    assert all(len(window['alternatives']) == 3 for item in result['sessions'] for window in item['windows'])


def test_repair_controls_budget_and_rejection_never_reroll(controls):
    from hymem.deadline import DeadlineExceeded
    client = Client('bad json')
    result = replay.repair_contract_controls(controls, ['control-dense'], budget(client), api, support, r6.control_cases())
    assert result['complete_sessions'] == 0 and len(client.calls) == 1
    assert result['sessions'][0]['windows'][0]['failure_reason'] == 'parse_failure'
    with pytest.raises(DeadlineExceeded):
        replay.repair_contract_controls(controls, ['control-multiwindow'], budget(AlternativesClient(), cap=1),
                                       api, support, r6.control_cases())


def test_exact_retained_case_one_call_keeps_input_private(tmp_path):
    client = AlternativesClient()
    wrapped = budget(client)
    wire = {'messages': [{'role': 'system', 'content': 'Original primary'},
                         {'role': 'user', 'content': '{"prior_summary":"PRIVATE_CONTEXT","new_material":"PRIVATE_SOURCE"}'}],
            'temperature': 0.0, 'max_tokens': 3072}
    result = replay.retained_repair_case(wire, 'a'*64, wrapped, api, support, tmp_path.resolve())
    assert result['passed'] and result['completion_calls'] == wrapped.calls == len(client.calls) == 1
    assert json.loads(client.calls[0].user)['original_generation_input'] == wire['messages'][1]['content']
    assert '505' in client.calls[0].system
    assert all(text not in json.dumps(result) for text in ('PRIVATE_CONTEXT', 'PRIVATE_SOURCE', 'Aster proposed'))
    assert result['database_writes'] == 0 and not result['raw_content_exported']
    assert (tmp_path/'retained-repair-case.json').is_file()


def test_previous_inventory_and_body_pins_fail_before_dispatch(tmp_path, monkeypatch):
    manifest_path = tmp_path.resolve()/'previous-manifest.json'
    manifest_path.write_text(json.dumps({'model': replay.MODEL, 'endpoint': replay.ENDPOINT}))
    previous = tmp_path.resolve()/'previous'
    (previous/'private').mkdir(parents=True)
    wire = {'model': replay.MODEL, 'messages': [
        {'role': 'system', 'content': 'Original primary'},
        {'role': 'user', 'content': '{"prior_summary":"PRIVATE","new_material":"PRIVATE"}'}],
        'thinking': {'type': 'disabled'}, 'temperature': 0.0, 'max_tokens': 3072,
        'response_format': {'type': 'json_object'}}
    raw = json.dumps(wire).encode()
    wrapper = {'phase': 'explicit_recovery', 'session_sha256': 'a'*64,
               'body_base64': base64.b64encode(raw).decode(), 'body_sha256': hashlib.sha256(raw).hexdigest()}
    path = previous/'private'/'035-request.json'
    path.write_text(json.dumps(wrapper))
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'summary': 'x'*505})}}]}
    response_raw = json.dumps(response).encode()
    response_wrapper = {**wrapper, 'status_code': 200, 'body_base64': base64.b64encode(response_raw).decode(),
                        'body_sha256': hashlib.sha256(response_raw).hexdigest()}
    response_path = previous/'private'/'035-response.json'
    response_path.write_text(json.dumps(response_wrapper))
    files = {p.name: {'sha256': replay.sha(p)} for p in (path, response_path)}
    inventory_pin = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pins = {**replay.PREVIOUS_PINS, 'manifest_sha256': replay.sha(manifest_path),
            'capture_inventory_sha256': inventory_pin, 'request_sha256': replay.sha(path),
            'request_body_sha256': wrapper['body_sha256']}
    (previous/'live.json').write_text(json.dumps({'manifest_sha256': pins['manifest_sha256'],
        'private_capture': {'files': files, 'inventory_sha256': inventory_pin}}))
    monkeypatch.setattr(replay, 'PREVIOUS', previous)
    monkeypatch.setattr(replay, 'PREVIOUS_MANIFEST', manifest_path)
    monkeypatch.setattr(replay, 'PREVIOUS_PINS', pins)
    assert replay.previous_request()[0]['messages'][1]['content'] == wire['messages'][1]['content']
    path.write_text('altered capture')
    with pytest.raises(RuntimeError, match='previous_capture_drift'):
        replay.previous_request()
