"""Root-owned verification of timeout-bound four-question integration; offline."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import chatgpt_plan_lme_v2 as bridge
from tools.diagnostics import siwc_lme_diagnostic_bundle_v3 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v3 as host
from tools.diagnostics import siwc_lme_diagnostic_launch_v3 as launch
from tools.diagnostics import siwc_lme_diagnostic_v4 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v6 as reader

REPO = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle')


def fixture(name):
    path = REPO / 'tests' / name
    spec = importlib.util.spec_from_file_location('root_timeout_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bundle, module.host, module.launch = bundle, host, launch
    module.runner, module.reader, module.bridge = runner, reader, bridge
    module.ACCEPTED, module.CODE = FROZEN, FROZEN / 'code'
    return module


def assembled(tmp_path):
    root = tmp_path / '.hymem-siwc-lme-diagnostic-rootverify0001'
    summary = bundle.assemble(repo=REPO, accepted_code=FROZEN / 'code',
        candidate=FROZEN / 'candidate', map_path=FROZEN / 'source-map.json', output=root)
    assert summary['candidate_files'] == 514 and summary['code_files'] == 26
    assert summary['model_calls'] == 0 and not summary['credential_present']
    return root


def test_genuine_541_file_archive_and_freshness(tmp_path, monkeypatch):
    root = assembled(tmp_path)
    payload = host.archive_bytes(root)
    statements = []
    for node in ast.parse(host.REMOTE).body:
        text = ast.get_source_segment(host.REMOTE, node) or ''
        if text.startswith('need(os.getuid()'):
            continue
        if text.startswith('need(regular(DATASET)'):
            break
        statements.append(node)
    else:
        raise AssertionError('safe_decoder_boundary_missing')
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(payload)))
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                 '<offline-root-decoder>', 'exec'), scope)
    assert len(scope['manifest']) == len(scope['content']) == 541
    assert not (root / 'launch-receipt.json').exists()
    with pytest.raises(ValueError):
        bundle.assemble(repo=REPO, accepted_code=FROZEN / 'code',
            candidate=FROZEN / 'candidate', map_path=FROZEN / 'source-map.json', output=root)
    (root / 'extra').write_text('invented')
    with pytest.raises(ValueError):
        host.archive_bytes(root)


def test_actual_source_only_import_receipt_size_and_v7_binding(tmp_path):
    root = assembled(tmp_path)
    script = r'''
import importlib.util,json,pathlib,sys
from types import SimpleNamespace
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
        raise AssertionError('external_operation_forbidden')
    if event == 'open' and isinstance(args[0], (str,bytes)):
        path=str(args[0])
        if path.endswith('/auth.json') or '/.codex/' in path or '/.hymem-chatgpt-plan-lme/' in path:
            raise AssertionError('credential_read_forbidden')
sys.addaudithook(audit)
root=pathlib.Path(sys.argv[1]);path=root/'code/tools/diagnostics/siwc_lme_diagnostic_v4.py'
spec=importlib.util.spec_from_file_location('isolated_siwc_timeout_root',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
assert pathlib.Path(loaded['siwc'].__file__).name == 'chatgpt_plan_lme_v2.py'
assert pathlib.Path(loaded['siwc'].transport.__file__).name == 'chatgpt_plan_responses_v7.py'
assert pathlib.Path(loaded['siwc'].transport._v6.__file__).name == 'chatgpt_plan_responses_v6.py'
questions=[{'question_id':f'invented-{i}'} for i in range(4)]
dataset=root/'absent-invented-dataset.json';original=runner._sha
runner._sha=lambda p:runner.DATASET_SHA256 if p==dataset else original(p)
loaded.update(source_only=False,dataset=dataset,questions=questions,
    prior=SimpleNamespace(SelectedQuestions=lambda *_:questions))
receipt=runner.receipt_for(root,loaded)
assert len(receipt['source_sha256']) == 26 and receipt['workers'] == 4
assert receipt['selected_count'] == 4 and len(receipt['selected_row_sha256']) == 4
assert receipt['model'] == 'gpt-5.6-luna' and receipt['store'] is False and receipt['stream'] is True
assert receipt['grant_identity_sha256'] == runner.GRANT_IDENTITY_SHA256
assert 4000 < len(runner._canonical(receipt)) <= 8192
print(json.dumps({'source_only_import':True,'receipt_bytes':len(runner._canonical(receipt)),'model_calls':0}))
'''
    done = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(root)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)['source_only_import'] is True


@pytest.mark.parametrize('name', [
    'test_recursive_prior_unit_cleanup_includes_nested_threads',
    'test_ambiguous_dispatch_consumes_one_shot',
])
def test_unchanged_cleanup_and_one_shot_controls(tmp_path, monkeypatch, name):
    getattr(fixture('test_siwc_host_root_v1.py'), name)(tmp_path, monkeypatch)


@pytest.mark.parametrize('name', [
    'test_launcher_command_fixed_caps_runtime_and_no_binary_route',
    'test_current_and_prior_unit_exclusion_includes_siwc',
    'test_reader_has_no_source_exec_or_auth_import',
])
def test_unchanged_launch_and_readonly_policies(name):
    getattr(fixture('test_siwc_host_root_v1.py'), name)()


@pytest.mark.parametrize('failure', [False, True])
def test_real_campaign_bridge2_to_reader6(tmp_path, monkeypatch, failure):
    fixtures = fixture('test_siwc_lme_runner_root_v1.py')
    run, closes, _, _ = fixtures.rig(tmp_path, monkeypatch, failure=failure, observed=True)
    result = run()
    assert closes == [True]
    check = json.loads((tmp_path / 'run/diagnostic-checkpoint.json').read_text())
    receipt = {'selected_row_sha256': check['manifest']['selected_row_sha256'],
               'source_sha256': {**runner.PINS, **runner.SIWC_PINS}}
    counts, completed, unhealthy, degraded, verified = reader.checkpoint(check, receipt)
    assert reader.terminal(result, verified, receipt, counts, completed, unhealthy) is result
    assert result['schema'] == runner.SCHEMA == reader.RUN_SCHEMA
    assert result['diagnostic_complete'] is (not failure)
    assert completed == (3 if failure else 4) and degraded == completed * 2
    assert result['budget']['usage_complete'] is (not failure)
    assert result['siwc_pilot_projection']['schema'] == 'siwc_lme_pilot_projection_v2'
    for field in ['known_tokens', 'turns']:
        changed = copy.deepcopy(result)
        changed['budget'][field] += 1
        with pytest.raises(ValueError):
            reader.terminal(changed, verified, receipt, counts, completed, unhealthy)


@pytest.mark.parametrize('state', ['clean_exit', 'failed_exit_cleaned'])
def test_cleanup_alone_never_measurement_success(tmp_path, monkeypatch, state):
    fixture('test_siwc_host_root_v1.py').test_cleanup_without_terminal_result_is_never_success(
        tmp_path, monkeypatch, state)


def observation():
    return {'child_phase': 'response_close', 'parent_timeout_site': 'result_wait',
        'last_progress_elapsed_ms': 23, 'parent_elapsed_ms': 1000,
        'timeout_allowance_ms': 1000, 'elapsed_saturated': False, 'snapshot_valid': True,
        'wire_bytes': 90, 'event_count': 1, 'completion_seen': True,
        'result_ready': False, 'result_ipc_started': False, 'child_alive_when_sampled': False}


def test_timeout_observation_survives_actual_campaign_and_reader(tmp_path, monkeypatch):
    original_init = bridge.SIWCLMEClient.__init__
    def init(self, *args, **kwargs):
        invoke = kwargs['response_call']
        def response(*inner_args, **inner_kwargs):
            try:
                return invoke(*inner_args, **inner_kwargs)
            except bridge.transport.TransportError:
                raise bridge.transport.TransportError('timeout', timeout_observation=observation()) from None
        kwargs['response_call'] = response
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(bridge.SIWCLMEClient, '__init__', init)
    fixtures = fixture('test_siwc_lme_runner_root_v1.py')
    run, closes, leases, _ = fixtures.rig(tmp_path, monkeypatch, failure=True, observed=True)
    result = run()
    assert closes == [True] and len(leases) == 5
    assert result['budget']['usage_complete'] is False and not result['diagnostic_complete']
    assert result['budget']['first_failure']['timeout_observation'] == observation()
    check = json.loads((tmp_path / 'run/diagnostic-checkpoint.json').read_text())
    receipt = {'selected_row_sha256': check['manifest']['selected_row_sha256'],
               'source_sha256': {**runner.PINS, **runner.SIWC_PINS}}
    counts, completed, unhealthy, _, verified = reader.checkpoint(check, receipt)
    assert reader.terminal(result, verified, receipt, counts, completed, unhealthy) is result
    for field in ('budget', 'siwc_observations', 'siwc_pilot_projection'):
        bad = copy.deepcopy(result)
        if field == 'budget':
            target = bad[field]['first_failure']
        elif field == 'siwc_observations':
            target = bad[field]['question.0.ordinary']['summary']['first_failure']
        else:
            target = bad[field]['questions'][0]['ordinary']['first_failure']
        target['timeout_observation']['child_phase'] = 'PRIVATE_ROOT_SENTINEL'
        with pytest.raises(ValueError):
            reader.terminal(bad, verified, receipt, counts, completed, unhealthy)


@pytest.mark.parametrize('value', [None, True, -1, 1.5, float('inf'), float('nan'),
                                  'PRIVATE_ROOT_SENTINEL', [], {}, 10**100])
def test_reader_timeout_fields_match_transport_rejection(value):
    first = {'code': 'timeout', 'phase': 'http', 'turn_admitted': True,
             'unknown_usage': True, 'timeout_observation': observation()}
    assert reader.first_failure(first) == bridge.project_first_failure(first)
    for key in ('last_progress_elapsed_ms', 'parent_elapsed_ms', 'timeout_allowance_ms',
                'wire_bytes', 'event_count'):
        changed = copy.deepcopy(first)
        changed['timeout_observation'][key] = value
        with pytest.raises((ValueError, TypeError)):
            reader.first_failure(changed)
        with pytest.raises((ValueError, TypeError)):
            bridge.project_first_failure(changed)
