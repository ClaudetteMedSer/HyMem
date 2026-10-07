"""Root verification of the revised source chain; no account or provider I/O."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import siwc_lme_diagnostic_bundle_v2 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v2 as host
from tools.diagnostics import siwc_lme_diagnostic_launch_v2 as launch
from tools.diagnostics import siwc_lme_diagnostic_v3 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v5 as reader

REPO = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle')


def prior_tests(name):
    path = REPO / 'tests' / name
    spec = importlib.util.spec_from_file_location('root_revised_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bundle, module.host, module.launch = bundle, host, launch
    module.runner, module.reader = runner, reader
    module.ACCEPTED, module.CODE = FROZEN, FROZEN / 'code'
    return module


@pytest.mark.parametrize('name', [
    'test_actual_540_file_archive_and_remote_decoder',
    'test_recursive_prior_unit_cleanup_includes_nested_threads',
    'test_ambiguous_dispatch_consumes_one_shot',
])
def test_revised_chain_actual_helpers(tmp_path, monkeypatch, name):
    getattr(prior_tests('test_siwc_host_root_v1.py'), name)(tmp_path, monkeypatch)


@pytest.mark.parametrize('name', [
    'test_launcher_command_fixed_caps_runtime_and_no_binary_route',
    'test_current_and_prior_unit_exclusion_includes_siwc',
    'test_reader_finite_projection_matches_bridge',
    'test_reader_has_no_source_exec_or_auth_import',
])
def test_revised_chain_policies(name):
    getattr(prior_tests('test_siwc_host_root_v1.py'), name)()


@pytest.mark.parametrize('key,value', [
    ('wire_observation', {'secret': 'private provider text'}),
    ('stream_observation', {'event_type': 'private provider text'}),
    ('unknown_usage', False), ('turn_admitted', 1), ('phase', 'private'),
    ('code', 'private provider text'), ('http_status', True),
])
def test_revised_reader_rejects_private_or_false_metadata(key, value):
    prior_tests('test_siwc_host_root_v1.py').test_reader_rejects_unbounded_or_untruthful_failure_metadata(key, value)


@pytest.mark.parametrize('state', ['clean_exit', 'failed_exit_cleaned'])
def test_revised_cleanup_alone_never_success(tmp_path, monkeypatch, state):
    prior_tests('test_siwc_host_root_v1.py').test_cleanup_without_terminal_result_is_never_success(tmp_path, monkeypatch, state)


@pytest.mark.parametrize('failure', [False, True])
def test_actual_new_campaign_to_new_reader(tmp_path, monkeypatch, failure):
    fixture = prior_tests('test_siwc_lme_runner_root_v1.py')
    run, closes, _, _ = fixture.rig(tmp_path, monkeypatch, failure=failure, observed=True)
    result = run()
    assert closes == [True]
    assert result['schema'] == runner.SCHEMA == reader.RUN_SCHEMA
    assert result['budget']['resource_fault'] is None
    check = json.loads((tmp_path / 'run/diagnostic-checkpoint.json').read_text())
    receipt = {'selected_row_sha256': check['manifest']['selected_row_sha256'],
               'source_sha256': {**runner.PINS, **runner.SIWC_PINS}}
    counts, completed, unhealthy, degraded, verified = reader.checkpoint(check, receipt)
    assert reader.terminal(result, verified, receipt, counts, completed, unhealthy) is result
    assert result['diagnostic_complete'] is (not failure)
    assert completed == (3 if failure else 4) and degraded == completed * 2
    assert result['budget']['usage_complete'] is (not failure)
    for field in ['known_tokens', 'turns']:
        changed = copy.deepcopy(result)
        changed['budget'][field] += 1
        with pytest.raises(ValueError):
            reader.terminal(changed, verified, receipt, counts, completed, unhealthy)
    changed = copy.deepcopy(result)
    changed['siwc_observations']['question.0.ordinary']['summary']['known_tokens'] += 1
    with pytest.raises(ValueError):
        reader.terminal(changed, verified, receipt, counts, completed, unhealthy)


def test_actual_revised_import_and_receipt_without_auth(tmp_path):
    root = prior_tests('test_siwc_host_root_v1.py').assembled(tmp_path)
    script = r'''
import importlib.util,json,pathlib,sys
from types import SimpleNamespace
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
        raise AssertionError('forbidden external operation')
sys.addaudithook(audit)
root=pathlib.Path(sys.argv[1]);path=root/'code/tools/diagnostics/siwc_lme_diagnostic_v3.py'
spec=importlib.util.spec_from_file_location('isolated_siwc_revised_root',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
questions=[{'question_id':f'invented-{i}'} for i in range(4)]
dataset=root/'absent-invented-dataset.json';original=runner._sha
runner._sha=lambda p:runner.DATASET_SHA256 if p==dataset else original(p)
loaded.update(source_only=False,dataset=dataset,questions=questions,
    prior=SimpleNamespace(SelectedQuestions=lambda *_:questions))
receipt=runner.receipt_for(root,loaded)
assert len(receipt['source_sha256'])==25 and receipt['workers']==4
assert receipt['model']=='gpt-5.6-luna' and receipt['store'] is False and receipt['stream'] is True
assert receipt['grant_identity_sha256']==runner.GRANT_IDENTITY_SHA256
assert 4000 < len(runner._canonical(receipt)) <= 8192
print(json.dumps({'source_only_import':True,'receipt_bytes':len(runner._canonical(receipt)),'model_calls':0}))
'''
    done = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(root)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    safe = json.loads(done.stdout)
    assert safe['source_only_import'] and safe['model_calls'] == 0


def test_exact_executable_deltas_are_only_contract_literal():
    pairs = [
        ('siwc_lme_diagnostic_v2.py', 'siwc_lme_diagnostic_v3.py'),
        ('siwc_lme_diagnostic_progress_v4.py', 'siwc_lme_diagnostic_progress_v5.py'),
        ('siwc_lme_diagnostic_launch_v1.py', 'siwc_lme_diagnostic_launch_v2.py'),
    ]
    old_contract = 'hymem-extraction-contract-sha256-v1:94a1adfc028694e9b1868ec8712baff38d4d7a99a9d2e3547c66819711890f95'
    new_contract = 'hymem-extraction-contract-sha256-v1:f39dd678dc4a69ef589b4cc7c4288de7393a90aedf16da611153d08d6241b510'
    for old, new in pairs:
        def definitions(name):
            raw = (REPO / 'tools/diagnostics' / name).read_text().replace(
                new_contract.split(':')[1], old_contract.split(':')[1])
            return [ast.dump(n, include_attributes=False) for n in ast.parse(raw).body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert definitions(old) == definitions(new)


@pytest.mark.parametrize('name', [
    'test_actual_four_worker_ledger_and_ten_views_reconcile',
    'test_one_failed_admitted_turn_retains_unknown_usage_and_partial_evidence',
    'test_grant_change_blocks_canary_before_any_acquire',
    'test_canonical_receipt_rejects_type_coercion_and_duplicates',
])
def test_revised_campaign_boundary_controls(tmp_path, monkeypatch, name):
    getattr(prior_tests('test_siwc_lme_runner_root_v1.py'), name)(tmp_path, monkeypatch)
