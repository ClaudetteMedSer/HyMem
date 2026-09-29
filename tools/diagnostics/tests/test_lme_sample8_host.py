"""Offline host-plan/install/census controls; no SSH, Docker or provider calls."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tarfile
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'tools/diagnostics/lme_sample8_v1'


def load(name):
    path = BASE / name
    module = types.ModuleType('sample8_host_test_' + name.replace('/', '_').replace('.', '_'))
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('network forbidden during host helper tests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)


def census():
    return json.loads((ROOT / 'docs/patches/2026-09-24-lme-sample8-census.json').read_text())


def need(value, code):
    if not value:
        raise ValueError(code)


def test_checked_census_projects_exact_metadata_only():
    source = census()
    rows = load('prepare.py').checked_census(source, load('bundle/q1_stock_run.py'), types.SimpleNamespace(need=need))
    assert rows == source['questions'] and rows is not source['questions']
    assert all(row is not original for row, original in zip(rows, source['questions']))
    assert sum(row['sessions'] for row in rows) == 398
    assert sum(row['messages'] for row in rows) == 4036


@pytest.mark.parametrize('fault', ['extra_top', 'extra_row', 'row_size', 'redistributed_size',
                                  'boolean_count', 'float_count', 'swapped_rows', 'category',
                                  'row_index', 'float_index', 'seed_bool', 'sample_bool',
                                  'provider_calls', 'raw_export', 'source_count', 'unfrozen'])
def test_checked_census_rejects_corruption_and_unbounded_fields(fault):
    value = census()
    if fault == 'extra_top': value['invented_text'] = 'must not enter artifact'
    elif fault == 'extra_row': value['questions'][0]['invented_text'] = 'must not enter artifact'
    elif fault == 'row_size': value['questions'][0]['messages'] += 1
    elif fault == 'redistributed_size':
        value['questions'][0]['sessions'] -= 1
        value['questions'][1]['sessions'] += 1
    elif fault == 'boolean_count': value['questions'][0]['sessions'] = True
    elif fault == 'float_count': value['questions'][0]['sessions'] = 49.0
    elif fault == 'swapped_rows': value['questions'].reverse()
    elif fault == 'category': value['questions'][0]['question_type'] = 'temporal-reasoning'
    elif fault == 'row_index': value['questions'][0]['source_index'] = 210
    elif fault == 'float_index': value['questions'][0]['source_index'] = 213.0
    elif fault == 'seed_bool': value['seed'] = False
    elif fault == 'sample_bool': value['sample'] = True
    elif fault == 'provider_calls': value['provider_calls'] = 1
    elif fault == 'raw_export': value['raw_content_exported'] = True
    elif fault == 'source_count': value['source_question_count'] = 499
    elif fault == 'unfrozen': value['selection_predeclared'] = False
    with pytest.raises(ValueError, match='census'):
        load('prepare.py').checked_census(value, load('bundle/q1_stock_run.py'), types.SimpleNamespace(need=need))


@pytest.mark.parametrize('mode', ['preflight', 'live', 'validation'])
def test_pure_host_plan_isolated_and_never_starts(mode):
    host, ctl = load('bundle/q1_stock_host.py'), load('host_control.py')
    plan = (host.validation_command(str(ctl.ROOT), str(ctl.SOURCE), 'a' * 64)
            if mode == 'validation' else host.command(str(ctl.ROOT), str(ctl.SOURCE), 'a' * 64, live=mode == 'live'))
    assert plan['command'][:2] == ['docker', 'create'] and plan['starts_work'] is False
    assert plan['network'] == ('bridge' if mode == 'live' else 'none')
    assert ('/run/deepseek.env' in plan['mounts']) is (mode == 'live')
    assert all(not writable for target, (_, writable) in plan['mounts'].items()
               if mode == 'validation' or target != '/results')
    assert '/var/run/docker.sock' not in plan['mounts']
    assert not set(plan['command']) & {'--privileged', '--publish', '--cap-add', '--device'}


@pytest.mark.parametrize('suffix', ['broad', 'home', '/../prod', ',dst=/escape', 'relative'])
def test_plan_rejects_out_of_scope_paths(suffix):
    host, ctl = load('bundle/q1_stock_host.py'), load('host_control.py')
    path = {'broad': '/', 'home': '/opt/stacks/hermes/instance1/home/.hermes',
            'relative': 'relative'}.get(suffix, str(ctl.ROOT) + suffix)
    with pytest.raises((ValueError, TypeError)):
        host.command(path, str(ctl.SOURCE), 'a' * 64)


def inspect_fixture(plan):
    args = plan['command'][plan['command'].index(plan['image']) + 1:]
    return {'Image': plan['image'], 'Name': '/' + plan['name'],
            'Config': {'Image': plan['image'], 'User': '1000:1000', 'WorkingDir': '/candidate',
                       'Entrypoint': ['/home/node/hymem-env/bin/python3'], 'Cmd': args, 'Env': ['PATH=/usr/bin']},
            'HostConfig': {'NetworkMode': plan['network'], 'ReadonlyRootfs': True, 'Privileged': False,
                           'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges'], 'Init': True,
                           'PidsLimit': 128, 'Memory': 2147483648, 'NanoCpus': 2000000000,
                           'Tmpfs': {'/tmp': 'rw,noexec,nosuid,size=64m'}, 'RestartPolicy': {'Name': 'no'}},
            'Mounts': [{'Destination': dst, 'Source': src, 'RW': rw, 'Type': 'bind'}
                       for dst, (src, rw) in plan['mounts'].items()],
            'State': {'Status': 'created', 'ExitCode': 0, 'OOMKilled': False, 'Pid': 0},
            'Args': args, 'Path': '/home/node/hymem-env/bin/python3'}


@pytest.mark.parametrize('fault', [None, 'writable_source', 'network', 'privileged',
                                  'credentials_env', 'command', 'image', 'restart'])
def test_real_inspector_rejects_unsafe_configuration(monkeypatch, fault):
    host, ctl = load('bundle/q1_stock_host.py'), load('host_control.py')
    plan = host.command(str(ctl.ROOT), str(ctl.SOURCE), 'a' * 64, live=True)
    obj = inspect_fixture(plan)
    if fault == 'writable_source': next(m for m in obj['Mounts'] if m['Destination'] == '/candidate')['RW'] = True
    elif fault == 'network': obj['HostConfig']['NetworkMode'] = 'host'
    elif fault == 'privileged': obj['HostConfig']['Privileged'] = True
    elif fault == 'credentials_env': obj['Config']['Env'].append('DEEPSEEK_API_KEY=invented')
    elif fault == 'command': obj['Config']['Cmd'] = ['bad']
    elif fault == 'image': obj['Image'] = 'sha256:' + '0' * 64
    elif fault == 'restart': obj['HostConfig']['RestartPolicy']['Name'] = 'always'
    monkeypatch.setattr(ctl.subprocess, 'check_output', lambda *a, **kw: json.dumps([obj]))
    if fault:
        with pytest.raises(AssertionError): ctl.inspect('b' * 64, plan, 'created')
    else:
        assert ctl.inspect('b' * 64, plan, 'created')['configuration_verified'] is True


@pytest.mark.parametrize('fault', [None, 'archive_pin', 'member_hash', 'parent_path',
                                  'symlink', 'duplicate', 'existing_root'])
def test_installer_archive_and_fresh_root_guards(tmp_path, monkeypatch, fault):
    installer = load('install.py')
    root = tmp_path.resolve() / 'install'
    body = json.dumps({'remote_run_root': str(root), 'sample': 8, 'seed': 0}).encode()
    members = [('bundle/manifest.json', body)]
    if fault == 'parent_path': members.append(('bundle/../outside', b'invented'))
    if fault == 'duplicate': members.append(('bundle/manifest.json', body))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
        for name, raw in members:
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            if fault == 'symlink':
                item.type, item.linkname, item.size = tarfile.SYMTYPE, '/invented', 0
            tar.addfile(item, io.BytesIO(raw))
    raw = buffer.getvalue()
    config = {'root': str(root), 'archive_pin': hashlib.sha256(raw).hexdigest(),
              'manifest_pin': hashlib.sha256(body).hexdigest(),
              'files': {name: hashlib.sha256(value).hexdigest() for name, value in members}}
    if fault == 'archive_pin': config['archive_pin'] = '0' * 64
    if fault == 'member_hash': config['files']['bundle/manifest.json'] = '0' * 64
    if fault == 'existing_root': root.mkdir()
    monkeypatch.setattr(sys, 'stdin', types.SimpleNamespace(buffer=io.BytesIO(raw)))
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    with contextlib.redirect_stdout(io.StringIO()):
        if fault:
            with pytest.raises((AssertionError, FileExistsError)):
                exec(compile(installer.INSTALL, 'invented-install', 'exec'), {'C': config})
        else:
            exec(compile(installer.INSTALL, 'invented-install', 'exec'), {'C': config})
            assert (root / 'bundle/manifest.json').read_bytes() == body
            assert {p.name for p in root.iterdir()} == {'bundle', 'home', 'preflight-results', 'live-results', 'install.json'}
    if fault and fault != 'existing_root': assert not root.exists()


def preflight_fixture():
    run = load('bundle/q1_stock_run.py')
    metadata = census()
    manifest = {key: metadata[key] for key in ('dataset_sha256', 'source_question_count', 'source_indices', 'sample', 'seed')}
    manifest.update(source_mapping_sha256='invented-mapping', question_ids=run.QUESTION_IDS,
                    questions=metadata['questions'], sessions=398, messages=4036)
    pre = {**manifest, 'status': 'preflight_only', 'api_calls': 0, 'source_files_verified': 231,
           'raw_data_exported': False, 'benchmark_executed': False, 'selection_predeclared': True,
           'per_question_census': [{key: row[key] for key in ('source_index', 'question_id', 'sessions', 'messages')}
                                   for row in metadata['questions']],
           'startup_probe': {'schema': 'stock-lme-sample8-pre-provider-startup-v1',
                             'status': 'passed', 'phase': 'complete', 'cleanup_failed': False,
                             'expected_question_ids': run.QUESTION_IDS, 'expected_question_count': 8,
                             'real_credentials_loaded': False, 'benchmark_execution_verified': False,
                             'private_home_created': True,
                             **{key: True for key in ('standalone_producer_identity_exact', 'runtime_transport_identity_exact',
                                  'stock_cli_pre_provider_boundary_reached', 'checkpoint_runtime_producer_matches',
                                  'cli_checkpoint_handles_closed', 'canonical_extra_body_environment_absent')},
                             **{key: 1 for key in ('runtime_clients_constructed', 'runtime_clients_closed', 'cli_checkpoints_observed')},
                             **{key: 0 for key in ('provider_completions', 'provider_http_attempts', 'provider_successful_responses',
                                                  'outbound_operations_blocked', 'completed_questions')}}}
    return manifest, pre


@pytest.mark.parametrize('fault', [None, 'wrong_ids', 'wrong_count', 'provider_calls', 'open_checkpoint',
                                  'identity_false', 'outbound', 'wrong_phase', 'raw_census'])
def test_live_admission_binds_nested_real_startup_proof(monkeypatch, fault):
    host, ctl = load('bundle/q1_stock_host.py'), load('host_control.py')
    manifest, pre = preflight_fixture()
    if fault == 'wrong_ids': pre['startup_probe']['expected_question_ids'] = ['wrong']
    elif fault == 'wrong_count': pre['startup_probe']['expected_question_count'] = 1
    elif fault == 'provider_calls': pre['startup_probe']['provider_completions'] = 17
    elif fault == 'open_checkpoint': pre['startup_probe']['cli_checkpoint_handles_closed'] = False
    elif fault == 'identity_false': pre['startup_probe']['runtime_transport_identity_exact'] = False
    elif fault == 'outbound': pre['startup_probe']['outbound_operations_blocked'] = 1
    elif fault == 'wrong_phase': pre['startup_probe']['phase'] = 'initialization'
    elif fault == 'raw_census': pre['per_question_census'][0]['sessions'] -= 1
    r5 = (ROOT / 'docs/patches/2026-09-24-lme-independent-summary-indexing-r5-manifest.json').read_bytes()
    expected, junit = json.loads(r5), b'invented-junit-already-independently-gated'
    target_id = '421c0acb36950d1e5fd6f47afc3d6ac3f5c89f8ecda90832c84a848472528d91'
    fixtures = {'receipt.json': {'manifest_sha256': ctl.R5_PIN, 'gate_passed': True, 'selected': 1044,
                                 'junit_totals': {'tests': 1044, 'failures': 0, 'errors': 0, 'skipped': 4},
                                 'observed_skip_nodeids': sorted(expected['expected_skip_nodeids']),
                                 'junit_sha256': hashlib.sha256(junit).hexdigest()},
                'supervisor.json': {'returncode': 0, 'timed_out': False, 'child_reaped': True, 'group_absent': True, 'errors': []},
                'container-id.json': {'container_id': target_id}, 'preflight-container-id.json': {'container_id': 'c' * 64},
                'preflight.json': pre}
    def read(path):
        if path.name == 'manifest.json': return r5
        if path.name == 'junit.xml': return junit
        raise AssertionError('unexpected read')
    monkeypatch.setattr(ctl, 'PIN', 'a' * 64)
    monkeypatch.setattr(ctl, 'read', read)
    monkeypatch.setattr(ctl, 'js', lambda path: fixtures[path.name])
    monkeypatch.setattr(ctl, 'inspect', lambda *a: {'exit_code': 0, 'oom_killed': False, 'pid': 0})
    monkeypatch.setattr(ctl.subprocess, 'check_output', lambda *a, **kw: json.dumps([{
        'State': {'Status': 'exited', 'ExitCode': 0, 'OOMKilled': False, 'Pid': 0}}]))
    if fault:
        with pytest.raises(AssertionError): ctl.live_gates(manifest, host)
    else:
        ctl.live_gates(manifest, host)


def test_progress_exports_only_numeric_or_closed_metadata(tmp_path, monkeypatch, capsys):
    progress = load('progress.py')
    root = tmp_path.resolve() / 'live-results'
    (root / 'benchmark').mkdir(parents=True)
    (root / 'invocation').mkdir()
    (root / 'stores').mkdir()
    marker = 'INVENTED_PRIVATE_TEXT_MUST_NOT_BE_EXPORTED'
    (root / 'invocation/stdout.bin').write_text(marker)
    (root / 'benchmark/checkpoint.json').write_text(json.dumps({
        'expected_ids': progress.IDS,
        'entries': {progress.IDS[0]: {'status': 'completed', 'attempts': 1, 'row': {'answer': marker}}},
        'execution_segments': [{'instrumentation_errors': [marker]}],
    }))
    monkeypatch.setattr(progress, 'ROOT', root)
    progress.main()
    raw = capsys.readouterr().out
    assert marker not in raw
    result = json.loads(raw)
    assert result['instrumentation_error_count'] == 1
    assert result['questions'] == {progress.IDS[0]: {'status': 'completed', 'attempts': 1}}
    assert result['progress_only'] is True and result['new_provider_calls'] == 0
    assert result['stores'] == []


def test_progress_refuses_checkpoint_symlink(tmp_path, monkeypatch):
    progress = load('progress.py')
    root = tmp_path.resolve() / 'live-results'
    (root / 'benchmark').mkdir(parents=True)
    outside = tmp_path.resolve() / 'unrelated.json'
    outside.write_text('invented unrelated content')
    (root / 'benchmark/checkpoint.json').symlink_to(outside)
    monkeypatch.setattr(progress, 'ROOT', root)
    with pytest.raises(AssertionError): progress.main()
