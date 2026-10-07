"""Independent root controls for the six-hour, unchanged-call-budget policy."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import chatgpt_plan_lme_v5 as bridge
from tools.diagnostics import siwc_lme_diagnostic_v7 as old_runner
from tools.diagnostics import siwc_lme_diagnostic_v8 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v10 as reader
from tools.diagnostics import siwc_lme_diagnostic_bundle_v7 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v7 as host
from tools.diagnostics import siwc_lme_diagnostic_launch_v6 as old_launch
from tools.diagnostics import siwc_lme_diagnostic_launch_v7 as launch

REPO = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle')


def fixture(name):
    spec = importlib.util.spec_from_file_location('root_sixhour_' + name[:-3], REPO / 'tests' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bundle, module.host, module.launch = bundle, host, launch
    module.runner, module.reader, module.bridge = runner, reader, bridge
    module.ACCEPTED, module.CODE = FROZEN, FROZEN / 'code'
    return module


def test_runner_exact_delta_is_wall_and_version_only():
    previous = Path(old_runner.__file__).read_text()
    revised = Path(runner.__file__).read_text()
    for new, old in [('diagnostic-v8', 'diagnostic-v7'), ('diagnostic_v8.py', 'diagnostic_v7.py'),
                     ('25_200', '14_400'), ('23_400', '12_600'), ('21_600', '10_800'),
                     ('7h 2min', '4h 2min'), ('25330', '14530')]:
        revised = revised.replace(new, old)
    assert revised == previous
    assert runner.PINS == old_runner.PINS and runner.SIWC_PINS == old_runner.SIWC_PINS
    assert runner.DIAGNOSTIC_HELPER_SHA256 == old_runner.DIAGNOSTIC_HELPER_SHA256


def test_exact_nested_caps_and_unchanged_call_budgets():
    assert runner.MAX_LIMITS == {'campaign': (8012, 48160000, 25200),
        'question': (2000, 12000000, 23400), 'canary': (12, 160000, 600)}
    assert reader.LIMITS == {key: list(value) for key, value in runner.MAX_LIMITS.items()}
    assert 21600 < 23400 < 25200 < 25330
    assert bridge.MAX_INVOCATION == bridge.transport.MAX_WALL_SECONDS == 300
    for key in runner.MAX_LIMITS:
        assert runner.MAX_LIMITS[key][:2] == old_runner.MAX_LIMITS[key][:2]
    assert runner.MAX_LIMITS['canary'] == old_runner.MAX_LIMITS['canary']


def test_actual_systemd_command_only_changes_runner_and_runtime():
    root = Path('/home/atta/.hymem-siwc-lme-diagnostic-preflight-rootverify')
    receipt = {'unit': root.name[1:] + '.service', 'runtime_path': '/invented/runtime'}
    previous = old_launch.command(root, receipt, 'a' * 64)
    revised = launch.command(root, receipt, 'a' * 64)
    normalized = [value.replace('25330s', '14530s').replace('diagnostic_v8.py', 'diagnostic_v7.py')
                  for value in revised]
    assert previous == normalized
    assert '--property=RuntimeMaxSec=25330s' in revised
    assert '--property=KillMode=control-group' in revised
    assert '--property=Restart=no' in revised


@pytest.mark.parametrize('failure', [False, True])
def test_actual_four_worker_campaign_uses_new_wall_policy_and_reconciles(tmp_path, monkeypatch, failure):
    helpers = fixture('test_siwc_lme_runner_root_v1.py')
    run, closes, _, _ = helpers.rig(tmp_path, monkeypatch, failure=failure, observed=True)
    actual_campaign = runner.run_campaign
    actual_worker = runner._question_worker
    observed = []
    def worker(*args, **kwargs):
        assert args[2].seconds == 23400 and args[6] == 21600
        observed.append(args[4])
        return actual_worker(*args, **kwargs)
    def campaign(*args, **kwargs):
        assert kwargs['indexing_seconds'] == 10800  # historical invented fixture
        kwargs['indexing_seconds'] = 21600
        return actual_campaign(*args, **kwargs)
    monkeypatch.setattr(runner, '_question_worker', worker)
    monkeypatch.setattr(runner, 'run_campaign', campaign)
    result = run()
    assert closes == [True] and sorted(observed) == [0, 1, 2, 3]
    check = json.loads((tmp_path / 'run/diagnostic-checkpoint.json').read_text())
    receipt = {'selected_row_sha256': check['manifest']['selected_row_sha256'],
               'source_sha256': {**runner.PINS, **runner.SIWC_PINS}}
    counts, complete, unhealthy, _, verified = reader.checkpoint(check, receipt)
    assert reader.terminal(result, verified, receipt, counts, complete, unhealthy) is result
    assert result['diagnostic_complete'] is (not failure)
    assert complete == (3 if failure else 4)
    assert result['budget']['usage_complete'] is (not failure)
    old = copy.deepcopy(check)
    old['manifest']['limits']['indexing_seconds'] = 10800
    old['manifest']['run_id'] = reader.canonical_hash({k: v for k, v in old['manifest'].items() if k != 'run_id'})
    old['run_id'] = old['manifest']['run_id']
    with pytest.raises(ValueError, match='checkpoint_limits_invalid'):
        reader.checkpoint(old, receipt)


@pytest.mark.parametrize('name', [
    'test_recursive_prior_unit_cleanup_includes_nested_threads',
    'test_ambiguous_dispatch_consumes_one_shot',
])
def test_recursive_cleanup_and_no_repeat(tmp_path, monkeypatch, name):
    getattr(fixture('test_siwc_host_root_v1.py'), name)(tmp_path, monkeypatch)


@pytest.mark.parametrize('state', ['clean_exit', 'failed_exit_cleaned'])
def test_cleanup_without_result_is_not_measurement_success(tmp_path, monkeypatch, state):
    fixture('test_siwc_host_root_v1.py').test_cleanup_without_terminal_result_is_never_success(
        tmp_path, monkeypatch, state)


def test_real_source_assembly_and_isolated_import_with_external_operations_denied(tmp_path):
    root = tmp_path / '.hymem-siwc-lme-diagnostic-root6hourtest'
    assembled = bundle.assemble(repo=REPO, accepted_code=FROZEN / 'code',
        candidate=FROZEN / 'candidate', map_path=FROZEN / 'source-map.json', output=root)
    assert assembled['candidate_files'] == 514 and assembled['code_files'] == 26
    assert assembled['model_calls'] == 0
    assert len(host.source_manifest(root)) == 541
    assert hashlib.sha256((root / 'code' / runner.RUNNER_RELATIVE).read_bytes()).hexdigest() == reader.RUNNER_SHA256
    script = '''
import sys, pathlib, importlib.util, json
from types import SimpleNamespace
def deny(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
        raise AssertionError('external forbidden')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if path.endswith('/auth.json') or '/.codex/' in path or '/.hymem-chatgpt-plan-lme/' in path:
            raise AssertionError('auth forbidden')
sys.addaudithook(deny)
root = pathlib.Path(sys.argv[1])
path = root / 'code/tools/diagnostics/siwc_lme_diagnostic_v8.py'
spec = importlib.util.spec_from_file_location('isolated_root_sixhour', path)
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
loaded = runner.import_source_only(root, root / 'source-map.json', runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
assert pathlib.Path(loaded['siwc'].transport.__file__).name == 'chatgpt_plan_responses_v10.py'
questions = [{'question_id': f'invented-{i}'} for i in range(4)]
dataset = root / 'absent-invented-dataset.json'; original = runner._sha
runner._sha = lambda p: runner.DATASET_SHA256 if p == dataset else original(p)
loaded.update(source_only=False, dataset=dataset, questions=questions,
              prior=SimpleNamespace(SelectedQuestions=lambda *_: questions))
receipt = runner.receipt_for(root, loaded)
assert receipt['indexing_seconds'] == 21600
assert receipt['limits']['question'] == [2000,12000000,23400]
assert receipt['limits']['campaign'] == [8012,48160000,25200]
assert receipt['limits']['canary'] == [12,160000,600]
assert receipt['invocation_seconds'] == 300 and receipt['workers'] == 4
assert len(receipt['selected_row_sha256']) == 4
assert 4000 < len(runner._canonical(receipt)) <= 8192
print(json.dumps({'source_only_import':True,'model_calls':0}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(root)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'source_only_import': True, 'model_calls': 0}
