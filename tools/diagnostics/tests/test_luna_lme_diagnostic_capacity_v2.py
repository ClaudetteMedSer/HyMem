"""Offline capacity-only controls for the versioned diagnostic candidate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('luna_lme_diagnostic_v2.py')
launcher = load('luna_lme_diagnostic_launch_v2.py')
reader = load('luna_lme_diagnostic_progress_v3.py')
assembler = load('luna_lme_diagnostic_bundle_v2.py')
preflight = load('luna_lme_diagnostic_host_preflight_v2.py')


def group_fixture(tmp_path, limit='256', current='8', peak='20', denials='0'):
    cgroup = '/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-diagnostic-test.service'
    root = tmp_path / 'cgroup'
    group = root / cgroup.lstrip('/')
    group.mkdir(parents=True)
    (group / 'memory.max').write_text('4294967296\n')
    (group / 'cpu.max').write_text('200000 100000\n')
    for name, value in [('pids.current',current),('pids.peak',peak),('pids.max',limit)]:
        (group / name).write_text(value + '\n')
    (group / 'pids.events').write_text('max ' + denials + '\n')
    proc = tmp_path / 'proc-cgroup'
    proc.write_text('0::' + cgroup + '\n')
    return root, group, proc, cgroup


def test_256_policy_is_only_task_limit_change(tmp_path, monkeypatch):
    receipt = {'unit': 'hymem-luna-lme-diagnostic-test.service'}
    command = launcher.command(tmp_path, receipt, 'a'*64)
    assert '--property=TasksMax=256' in command
    assert '--property=TasksMax=128' not in command
    for property_value in ('MemoryMax=4294967296','CPUQuota=200%',
                           'RuntimeMaxSec=14530s','TimeoutStopSec=10s',
                           'Restart=no','KillMode=control-group'):
        assert '--property=' + property_value in command
    assert command.count('--workers') == 1
    assert command[command.index('--workers') + 1] == '4'


@pytest.mark.parametrize('limit', ['128','257','max'])
def test_wrong_task_limits_fail_reader_and_runner(tmp_path, monkeypatch, limit):
    root, group, proc, cgroup = group_fixture(tmp_path, limit=limit)
    monkeypatch.setattr(reader, 'CGROUP_ROOT', root)
    assert reader._cgroup_policy(group) is False
    with pytest.raises(ValueError, match='resource_observer_unverified'):
        runner._resource_sample(cgroup, cgroup_root=root, proc_cgroup=proc)


def test_task_sample_preserves_denials_and_unavailable_not_zero(tmp_path, monkeypatch):
    root, group, proc, cgroup = group_fixture(tmp_path, denials='3')
    sample = runner._resource_sample(cgroup, cgroup_root=root, proc_cgroup=proc)
    assert sample == {'current':8,'peak':20,'limit':256,'denials':3}
    monkeypatch.setattr(reader, 'CGROUP_ROOT', root)
    assert reader._live_resource(group) == sample
    (group / 'pids.events').unlink()
    with pytest.raises(OSError):
        runner._resource_sample(cgroup, cgroup_root=root, proc_cgroup=proc)


def test_dual_constructs_both_clients_without_completion():
    created=[]
    class Ordinary:
        def __init__(self,*args,**kwargs): created.append('ordinary')
        def close(self): pass
    class Staged:
        def __init__(self,*args,**kwargs): created.append('staged')
        def close(self): pass
    class Budget:
        def snapshot(self): return {'questions': {'q-0000': {}}}
    loaded={'warm':SimpleNamespace(WarmSubscriptionClient=Ordinary),
            'staged':SimpleNamespace(StagedSubscriptionClient=Staged),
            'binary':Path('/not-executed')}
    dual=runner.make_dual(loaded,Budget(),'q-0000',object())
    assert created == ['ordinary','staged']
    dual.close()


def test_reader_rejects_missing_task_observation():
    with pytest.raises(Exception):
        reader._resource_projection(None)
    assert reader._resource_projection(None,allow_none=True) is None


def test_source_only_bundle_and_host_preflight_pin(tmp_path):
    original = Path('/private/tmp/hymem-lme-diagnostic-offline-assembly-v2')
    output = tmp_path / 'bundle'
    result = assembler.assemble(repo=ROOT.parents[1],
        accepted_code=original / 'code', candidate=original / 'candidate',
        map_path=original / 'source-map.json', output=output)
    assert result['schema'] == 'luna-lme-diagnostic-source-bundle-v2'
    assert result['candidate_files'] == 514
    assert result['dataset_present'] is False
    assert result['model_calls'] == 0
    assert len(preflight.source_manifest(output)) == 524
    assert preflight.RUNNER_SHA == launcher.RUNNER_SHA256 == reader.RUNNER_SHA256
    compile(preflight.wrapped_remote(), '<host-preflight-v2>', 'exec')
