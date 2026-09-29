"""Offline controls for one fixed dependent validator; Docker is never invoked."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import types

import pytest

BASE = Path(__file__).resolve().parents[1] / 'lme_sample8_v1'


def load(name):
    path = BASE / name
    module = types.ModuleType('sample8_finish_test_' + name.replace('/', '_').replace('.', '_'))
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)


def report(mod, passed=True):
    meter = {key: (True if key.endswith('_available') else 8) for key in mod.USAGE_FIELDS}
    return {
        'schema': 'stock-lme-sample8-postvalidation-v1',
        'status': 'scored_sample8_completed' if passed else 'validated_sample8_has_failures',
        'benchmark_completed_without_faults': passed,
        **{key: True for key in ('strict_scored_artifact_validated', 'physical_checkpoint_bound',
                                  'process_completed_cleanly', 'reader_judge_calls_measured', 'canary_accounted_separately')},
        'question_ids': mod.QUESTION_IDS, 'sample': 8, 'seed': 0, 'source_indices': mod.SOURCE_INDICES,
        'run_id': 'sha256:' + 'a' * 64, 'manifest_sha256': mod.MANIFEST_SHA,
        'archive_sha256': 'b' * 64, 'checkpoint_sha256': 'c' * 64,
        'counts': {'expected': 8, 'attempted': 8, 'unique_attempted': 8, 'total_attempts': 8,
                   'completed': 8 if passed else 7, 'failed': 0 if passed else 1, 'missing': 0},
        'scores': {'OVERALL': {'count': 8, 'accuracy': 75.0}},
        'per_question': [{'question_id': qid, 'answer_correct': index != 0, 'strict_failure': False,
                          'indexing_complete': True, 'item_indexing_healthy': True,
                          'indexing_outcome': 'success_with_summary_degradation', 'summary_healthy': False,
                          'summary_degraded_sessions': 1, 'summary_missing_sessions': 1,
                          'malformed_summaries': 0} for index, qid in enumerate(mod.QUESTION_IDS)],
        'summary_degraded_questions': 8, 'summary_degraded_sessions': 8,
        'per_role_usage': {key: copy.deepcopy(meter) for key in ('reader', 'judge', 'retrieval', 'memory_pipeline', 'canary')},
        'aggregate_paid_usage': copy.deepcopy(meter), 'elapsed_s': 7200.0,
        'global_paid_call_cap': None, 'new_provider_calls': 0,
        **{key: False for key in ('full_500_readiness_verified', 'semantic_quality_guaranteed',
                                   'representative_sample', 'official_comparable')},
    }


def inspect_object(cid, plan, status):
    args = plan['command'][plan['command'].index(plan['image']) + 1:]
    return {'Id': cid, 'Image': plan['image'], 'Name': '/' + plan['name'],
            'Config': {'Image': plan['image'], 'User': '1000:1000', 'WorkingDir': '/candidate',
                       'Entrypoint': ['/home/node/hymem-env/bin/python3'], 'Cmd': args, 'Env': ['PATH=/usr/bin']},
            'HostConfig': {'NetworkMode': plan['network'], 'ReadonlyRootfs': True, 'Privileged': False,
                           'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges'], 'Init': True,
                           'PidsLimit': 128, 'Memory': 2147483648, 'NanoCpus': 2000000000,
                           'Tmpfs': {'/tmp': 'rw,noexec,nosuid,size=64m'}, 'RestartPolicy': {'Name': 'no'}},
            'Mounts': [{'Destination': dst, 'Source': src, 'RW': rw, 'Type': 'bind'}
                       for dst, (src, rw) in plan['mounts'].items()],
            'State': {'Status': status, 'ExitCode': 0, 'OOMKilled': False, 'Pid': 123 if status == 'running' else 0},
            'Args': args, 'Path': '/home/node/hymem-env/bin/python3'}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    mod, ctl, host = load('finish_validation.py'), load('host_control.py'), load('bundle/q1_stock_host.py')
    source_root, source_path = str(ctl.ROOT), str(ctl.SOURCE)
    root = tmp_path.resolve() / 'run'
    root.mkdir()
    monkeypatch.setattr(mod, 'ROOT', root)
    monkeypatch.setattr(ctl, 'ROOT', root)
    live_plan = host.command(source_root, source_path, mod.MANIFEST_SHA, live=True)
    validation_plan = host.validation_command(source_root, source_path, mod.MANIFEST_SHA)
    # Production plans remain genuine fixed-root plans; only receipt I/O is redirected.
    host = types.SimpleNamespace(command=lambda *args, **kwargs: copy.deepcopy(live_plan),
                                 validation_command=lambda *args: copy.deepcopy(validation_plan))
    monkeypatch.setattr(ctl, 'definitions', lambda: ({'question_ids': mod.QUESTION_IDS}, host))
    monkeypatch.setattr(mod, 'load_controller', lambda: ctl)
    ctl.subprocess = types.SimpleNamespace(check_output=mod.controller_command)
    (root / 'live-container-id.json').write_text(json.dumps({'container_id': mod.LIVE_CID}))
    cid = 'd' * 64
    state = types.SimpleNamespace(mod=mod, ctl=ctl, root=root, cid=cid, calls=[],
                                  live=inspect_object(mod.LIVE_CID, live_plan, 'running'),
                                  validator=inspect_object(cid, validation_plan, 'created'),
                                  live_exit=0, validation_exit=0, timeout=None, output=json.dumps(report(mod)))

    def command(argv, *, timeout=30):
        state.calls.append((list(argv), timeout))
        if argv[:2] == ['docker', 'inspect']:
            return json.dumps([state.live if argv[2] == mod.LIVE_CID else state.validator])
        if argv[:2] == ['docker', 'wait']:
            target = argv[2]
            if state.timeout == target:
                state.timeout = None
                raise mod.FinishError('command_timeout')
            obj = state.live if target == mod.LIVE_CID else state.validator
            code = state.live_exit if target == mod.LIVE_CID else state.validation_exit
            obj['State'].update(Status='exited', ExitCode=code, Pid=0)
            return str(code) + '\n'
        if argv[:2] == ['docker', 'create']:
            assert argv == validation_plan['command']
            assert '--network' in argv and argv[argv.index('--network') + 1] == 'none'
            assert all(not writable for _, writable in validation_plan['mounts'].values())
            return cid + '\n'
        if argv[:2] == ['docker', 'start']:
            assert argv[2] == cid
            state.validator['State'].update(Status='running', Pid=456)
            return cid + '\n'
        if argv[:2] == ['docker', 'stop']:
            assert argv == ['docker', 'stop', '--time', '10', cid]
            state.validator['State'].update(Status='exited', ExitCode=137, Pid=0)
            state.validation_exit = 137
            return cid + '\n'
        if argv[:2] == ['docker', 'logs']:
            assert argv[2] == cid
            return state.output
        pytest.fail('unapproved Docker command: ' + repr(argv))

    monkeypatch.setattr(mod, 'command', command)
    return state


def test_success_runs_only_one_offline_validator_and_saves_closed_metadata(rig):
    result = rig.mod.execute()
    assert result['status'] == 'validated' and result['stage'] == 'complete'
    assert result['validation_terminal']['status'] == 'exited'
    assert result['validation_terminal']['pid'] == 0
    assert result['validation_terminal']['network'] == 'none'
    assert result['validation_terminal']['credential_mount'] is False
    assert result['cleanup_attempted'] is False
    assert json.loads((rig.root / 'finish-validation/final.json').read_text()) == result
    metadata = json.loads((rig.root / 'finish-validation/validator.json').read_text())
    assert metadata['new_provider_calls'] == 0 and metadata['summary_degraded_sessions'] == 8
    assert 'scores' not in metadata and metadata['per_question'][0]['answer_correct'] is False
    assert [cmd[:2] for cmd, _ in rig.calls].count(['docker', 'create']) == 1
    assert [cmd for cmd, _ in rig.calls if cmd[:2] == ['docker', 'start']] == [['docker', 'start', rig.cid]]
    assert not any(cmd[1] in ('rm', 'restart', 'stop', 'kill') for cmd, _ in rig.calls)
    assert (['docker', 'wait', rig.mod.LIVE_CID], 32400) in rig.calls
    assert (['docker', 'wait', rig.cid], 600) in rig.calls
    waiting = json.loads((rig.root / 'finish-validation/waiting-live.json').read_text())
    assert waiting['live_container_id'] == rig.mod.LIVE_CID
    assert waiting['manifest_sha256'] == rig.mod.MANIFEST_SHA
    assert waiting['source_and_package_verified'] is waiting['configuration_verified'] is True
    assert waiting['live_state']['status'] == 'running' and waiting['new_provider_calls'] == 0


@pytest.mark.parametrize('fault', ['record_id', 'inspect_id', 'image', 'network', 'command', 'mount', 'live_exit', 'live_oom', 'live_pid'])
def test_bad_live_identity_configuration_or_terminal_state_never_creates_validator(rig, fault):
    if fault == 'record_id': (rig.root / 'live-container-id.json').write_text(json.dumps({'container_id': 'a' * 64}))
    elif fault == 'inspect_id': rig.live['Id'] = 'a' * 64
    elif fault == 'image': rig.live['Image'] = 'sha256:' + 'a' * 64
    elif fault == 'network': rig.live['HostConfig']['NetworkMode'] = 'host'
    elif fault == 'command': rig.live['Config']['Cmd'] = ['unexpected']
    elif fault == 'mount': rig.live['Mounts'][0]['RW'] = True
    elif fault == 'live_exit': rig.live_exit = 1
    elif fault == 'live_oom': rig.live['State']['OOMKilled'] = True
    elif fault == 'live_pid':
        original = rig.ctl.inspect
        def inspect(cid, plan, expected_state=None):
            value = original(cid, plan, expected_state)
            if expected_state == 'exited': value['pid'] = 1
            return value
        rig.ctl.inspect = inspect
    result = rig.mod.execute()
    assert result['status'] == 'failed'
    assert not any(cmd[1] in ('create', 'start', 'stop', 'restart', 'kill') for cmd, _ in rig.calls)


def test_live_timeout_leaves_paid_container_untouched(rig):
    rig.timeout = rig.mod.LIVE_CID
    result = rig.mod.execute()
    assert result['failure_code'] == 'command_timeout' and result['stage'] == 'waiting_live'
    assert result['cleanup_attempted'] is False and rig.live['State']['Status'] == 'running'
    assert all(cmd[1] in ('inspect', 'wait') for cmd, _ in rig.calls)


def test_validator_timeout_stops_only_owned_offline_container_with_terminal_proof(rig):
    rig.timeout = rig.cid
    result = rig.mod.execute()
    assert result['status'] == 'failed' and result['failure_code'] == 'command_timeout'
    assert result['cleanup_attempted'] is True and 'cleanup_failed' not in result
    assert result['validation_terminal']['pid'] == 0
    assert result['validation_terminal']['exit_code'] == 137
    assert [cmd for cmd, _ in rig.calls if cmd[1] == 'stop'] == [['docker', 'stop', '--time', '10', rig.cid]]
    assert not (rig.root / 'finish-validation/validator.json').exists()


@pytest.mark.parametrize('mode', ['repeat', 'prior_validation_intent'])
def test_duplicate_invocation_and_prior_validation_cannot_resume(rig, mode):
    if mode == 'repeat':
        first = rig.mod.execute()
        before = (rig.root / 'finish-validation/final.json').read_bytes()
        rig.calls.clear()
    else:
        (rig.root / 'validation-create-intent.json').write_text('{}')
    result = rig.mod.execute()
    assert result['status'] == 'failed'
    assert not any(cmd[1] in ('create', 'start', 'stop') for cmd, _ in rig.calls)
    if mode == 'repeat':
        assert first['status'] == 'validated' and result['failure_code'] == 'duplicate_invocation'
        assert rig.calls == [] and (rig.root / 'finish-validation/final.json').read_bytes() == before


@pytest.mark.parametrize('shape', ['structured', 'exception'])
def test_validation_failure_remains_failure_without_retry(rig, shape):
    rig.validation_exit = 1
    value = report(rig.mod, passed=False) if shape == 'structured' else {
        'status': 'stock_sample8_postvalidation_failed', 'exception_type': 'RuntimeError',
        'benchmark_completed_without_faults': False, 'usage_verified': False, 'new_provider_calls': 0}
    if shape == 'structured':
        value['per_question'][0].update(indexing_outcome='failure', strict_failure=True,
                                        indexing_complete=False, item_indexing_healthy=False)
    rig.output = json.dumps(value)
    result = rig.mod.execute()
    assert result['status'] == 'validation_failed' and result['validation_terminal']['exit_code'] == 1
    assert sum(cmd[1] == 'create' for cmd, _ in rig.calls) == 1
    assert 'exception_type' not in json.loads((rig.root / 'finish-validation/validator.json').read_text())


def test_source_changed_during_wait_cannot_create_validator(rig):
    original = rig.ctl.definitions
    count = 0
    def definitions():
        nonlocal count
        count += 1
        if count == 2:
            raise AssertionError('source changed')
        return original()
    rig.ctl.definitions = definitions
    result = rig.mod.execute()
    assert result['status'] == 'failed'
    assert not any(cmd[1] in ('create', 'start', 'stop') for cmd, _ in rig.calls)


def test_unsafe_validator_config_is_neither_started_nor_stopped(rig):
    rig.validator['HostConfig']['NetworkMode'] = 'bridge'
    result = rig.mod.execute()
    assert result['status'] == 'failed' and result['cleanup_attempted'] is True
    assert result['cleanup_failed'] is True
    assert not any(cmd[1] in ('start', 'stop') for cmd, _ in rig.calls)


def test_cleanup_cannot_target_live_container(rig):
    with pytest.raises(rig.mod.FinishError, match='cleanup_ownership'):
        rig.mod.stop_owned_validator(rig.ctl, rig.mod.LIVE_CID, {'network': 'none'})
    assert rig.calls == []


@pytest.mark.parametrize('fault', ['manifest_pin', 'ids', 'archive_pin', 'proof', 'claims', 'calls',
                                  'raw_field', 'raw_nested', 'extra_usage', 'bool_counter', 'nan', 'exit_mismatch'])
def test_validator_output_is_closed_and_cannot_forge_success(rig, fault):
    value = report(rig.mod)
    if fault == 'manifest_pin': value['manifest_sha256'] = 'b' * 64
    elif fault == 'ids': value['question_ids'] = ['wrong'] * 8
    elif fault == 'archive_pin': value['archive_sha256'] = 'raw text'
    elif fault == 'proof': value['physical_checkpoint_bound'] = False
    elif fault == 'claims': value['full_500_readiness_verified'] = True
    elif fault == 'calls': value['new_provider_calls'] = 1
    elif fault == 'raw_field': value['raw_benchmark'] = 'must not escape'
    elif fault == 'raw_nested': value['per_question'][0]['answer'] = 'must not escape'
    elif fault == 'extra_usage': value['aggregate_paid_usage']['secret'] = 'must not escape'
    elif fault == 'bool_counter': value['counts']['failed'] = False
    elif fault == 'nan': value['elapsed_s'] = float('nan')
    elif fault == 'exit_mismatch': rig.validation_exit = 1
    rig.output = json.dumps(value)
    result = rig.mod.execute()
    assert result['status'] == 'failed' and result['stage'] == 'reading_validation'
    assert not (rig.root / 'finish-validation/validator.json').exists()
    assert 'must not escape' not in (rig.root / 'finish-validation/final.json').read_text()


def test_controller_pin_is_actual_and_wrong_bytes_refused(tmp_path, monkeypatch):
    mod = load('finish_validation.py')
    raw = (BASE / 'host_control.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == mod.CONTROLLER_SHA
    assert mod.load_controller().PIN == mod.MANIFEST_SHA
    monkeypatch.setattr(mod, '__file__', str(tmp_path / 'finish_validation.py'))
    (tmp_path / 'host_control.py').write_bytes(raw + b'\n')
    with pytest.raises(mod.FinishError, match='controller_pin'):
        mod.load_controller()


def test_optimized_python_refused_before_docker_or_root_access():
    result = subprocess.run([sys.executable, '-I', '-B', '-O', str(BASE / 'finish_validation.py')],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert json.loads(result.stdout)['failure_code'] == 'optimized_python_forbidden'


@pytest.mark.parametrize('fault', ['too_much_output', 'timeout', 'nonzero'])
def test_command_bounds_discard_raw_stdout_and_errors(fault):
    mod = load('finish_validation.py')
    code = {'too_much_output': "print('x' * 65537)", 'timeout': 'import time; time.sleep(2)',
            'nonzero': "import sys; print('private synthetic text'); sys.exit(2)"}[fault]
    expected = {'too_much_output': 'command_output_limit', 'timeout': 'command_timeout', 'nonzero': 'command_failed'}[fault]
    with pytest.raises(mod.FinishError, match=expected):
        mod.command([sys.executable, '-I', '-B', '-c', code], timeout=0.1 if fault == 'timeout' else 10)


def test_strict_json_rejects_duplicate_fields():
    mod = load('finish_validation.py')
    with pytest.raises(mod.FinishError, match='duplicate_json_key'):
        mod.strict_json('{"status":"passed","status":"failed"}')
