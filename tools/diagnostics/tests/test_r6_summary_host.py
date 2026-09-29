"""Offline controls for the one-shot summary verification container boundary."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'lme_r6_summary_host.py'
spec = importlib.util.spec_from_file_location('r6_summary_host_test', PATH)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
PIN = 'a'*64
CID = 'b'*64


def container(mode):
    plan = host.plan(mode, PIN)
    args = plan['command'][plan['command'].index(host.IMAGE)+1:]
    return {
        'Id': CID, 'Image': host.IMAGE, 'Name': '/'+plan['name'],
        'Config': {'Image': host.IMAGE, 'User': '1000:1000', 'WorkingDir': '/candidate',
                   'Entrypoint': ['/home/node/hymem-env/bin/python3'], 'Cmd': args, 'Env': []},
        'Path': '/home/node/hymem-env/bin/python3', 'Args': args,
        'HostConfig': {'NetworkMode': plan['network'], 'ReadonlyRootfs': True,
                       'Privileged': False, 'CapDrop': ['ALL'],
                       'SecurityOpt': ['no-new-privileges'], 'Init': True,
                       'PidsLimit': 128, 'Memory': 2147483648, 'NanoCpus': 2000000000,
                       'Tmpfs': {'/tmp': 'rw,noexec,nosuid,size=128m'},
                       'RestartPolicy': {'Name': 'no'}},
        'Mounts': [{'Destination': dest, 'Source': src, 'RW': rw, 'Type': 'bind'}
                   for dest, (src, rw) in plan['mounts'].items()],
        'State': {'Status': 'exited', 'ExitCode': 0, 'OOMKilled': False, 'Pid': 0},
    }


@pytest.mark.parametrize('mode', ['offline', 'live'])
def test_exact_owned_container_is_admitted(monkeypatch, mode):
    monkeypatch.setattr(host, 'run', lambda _: json.dumps([container(mode)]))
    result = host.inspect(CID, host.plan(mode, PIN))
    assert result['configuration_verified'] and result['credential_mount'] == (mode == 'live')


def test_only_disposable_diagnostics_are_writable():
    for mode in ('offline', 'live'):
        plan = host.plan(mode, PIN)
        assert {dest for dest, (_, rw) in plan['mounts'].items() if rw} == {'/results', '/work'}
        assert '/reference' in plan['mounts'] and not plan['mounts']['/reference'][1]
        assert not any('production' in src for src, _ in plan['mounts'].values())
        assert ('/run/deepseek.env' in plan['mounts']) == (mode == 'live')
        assert ('/preflight' in plan['mounts']) == (mode == 'live')
        assert plan['network'] == ('bridge' if mode == 'live' else 'none')


@pytest.mark.parametrize('section,key,value', [
    ('Config', 'User', '0:0'), ('Config', 'WorkingDir', '/reference'),
    ('Config', 'Env', ['DEEPSEEK_API_KEY=synthetic']),
    ('Config', 'Cmd', ['-c', 'pass']), ('Config', 'Entrypoint', ['/bin/sh']),
    ('HostConfig', 'NetworkMode', 'host'), ('HostConfig', 'ReadonlyRootfs', False),
    ('HostConfig', 'Privileged', True), ('HostConfig', 'CapDrop', []),
    ('HostConfig', 'SecurityOpt', []), ('HostConfig', 'Init', False),
    ('HostConfig', 'PidsLimit', 0), ('HostConfig', 'Memory', 0),
    ('HostConfig', 'NanoCpus', 0), ('HostConfig', 'Tmpfs', {}),
    ('HostConfig', 'RestartPolicy', {'Name': 'always'}),
])
def test_container_drift_rejected(monkeypatch, section, key, value):
    actual = deepcopy(container('live'))
    actual[section][key] = value
    monkeypatch.setattr(host, 'run', lambda _: json.dumps([actual]))
    with pytest.raises(AssertionError):
        host.inspect(CID, host.plan('live', PIN))


@pytest.mark.parametrize('mutation', ['writable_reference', 'missing_reference', 'extra_mount', 'wrong_source'])
def test_mount_drift_rejected(monkeypatch, mutation):
    actual = container('live')
    reference = next(m for m in actual['Mounts'] if m['Destination'] == '/reference')
    if mutation == 'writable_reference':
        reference['RW'] = True
    elif mutation == 'missing_reference':
        actual['Mounts'].remove(reference)
    elif mutation == 'extra_mount':
        actual['Mounts'].append({'Destination': '/extra', 'Source': '/extra', 'RW': False, 'Type': 'bind'})
    else:
        reference['Source'] += '-other'
    monkeypatch.setattr(host, 'run', lambda _: json.dumps([actual]))
    with pytest.raises(AssertionError):
        host.inspect(CID, host.plan('live', PIN))


def test_launch_receipt_is_exclusive(tmp_path):
    path = tmp_path/'intent.json'
    host.save(path, {'pin': PIN})
    with pytest.raises(FileExistsError):
        host.save(path, {'pin': PIN})
    assert json.loads(path.read_text()) == {'pin': PIN}
