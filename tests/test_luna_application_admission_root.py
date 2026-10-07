"""Reproduce real retained-exited service admission, no host calls."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_application_fault_host_v1 as host
from tools.diagnostics import luna_application_fault_launch_v1 as old


def revised():
    source=Path(__file__).resolve().parents[1]/'tools/diagnostics/luna_application_fault_launch_v2.py'
    spec=importlib.util.spec_from_file_location('root_revised_application_launcher',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture(tmp_path,monkeypatch,module,mutation=None):
    unit='hymem-luna-legacy-probe-Ab_cd123.service'
    expected='/user.slice/user-1000.slice/user@1000.service/app.slice/'+unit
    group=tmp_path/'cgroup'/expected.lstrip('/');group.mkdir(parents=True)
    def empty(p):
        for name,value in {'cgroup.procs':'','cgroup.threads':'','cgroup.events':'populated 0\n'}.items():
            (p/name).write_text(value)
    empty(group)
    state={'MainPID':'0','ControlGroup':'','ActiveState':'active','SubState':'exited'}
    if mutation=='live_pid':state['MainPID']='123'
    if mutation=='running':state['SubState']='running'
    if mutation=='wrong_group':state['ControlGroup']='/user.slice/elsewhere'
    if mutation=='matching_group':state['ControlGroup']=expected
    if mutation=='failed':state.update(ActiveState='failed',SubState='failed')
    if mutation=='inactive':state.update(ActiveState='inactive',SubState='dead')
    if mutation in {'nested_process','nested_thread','nested_populated'}:
        child=group/'nested';child.mkdir();empty(child)
        if mutation=='nested_process':(child/'cgroup.procs').write_text('999\n')
        if mutation=='nested_thread':(child/'cgroup.threads').write_text('999\n')
        if mutation=='nested_populated':(child/'cgroup.events').write_text('populated 1\n')
    if mutation=='symlink':(group/'linked').symlink_to(tmp_path/'absent')
    if mutation=='traversal':unit='hymem-luna-../wrong.service'
    binary=tmp_path/'binary';binary.write_bytes(b'invented-runtime')
    fake=SimpleNamespace(BINARY=str(binary),BINARY_SHA256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        CGROUP_ROOT=tmp_path/'cgroup',recursive_empty=host.recursive_empty,
        unit_for=lambda *a:'hymem-luna-application-fault-v1-containment-abcd1234.service')
    monkeypatch.setattr(module,'_regular',lambda p,*a:p.is_file())
    monkeypatch.setattr(module,'_bus_env',lambda:{})
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _:SimpleNamespace(free=30*1024**3))
    read=Path.read_text
    def read_text(p,*a,**kw):
        if str(p)=='/proc/meminfo':return 'MemAvailable: 8000000 kB\n'
        return read(p,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read_text)
    monkeypatch.setattr(module,'_unit_state',lambda *a:state)
    def run(args,**kw):
        if args[0]=='/usr/bin/docker':return SimpleNamespace(returncode=0,stdout='false\n')
        assert args[:3]==['/usr/bin/systemctl','--user','list-units']
        return SimpleNamespace(returncode=0,stdout=(unit+' loaded active exited retained\n') if args[3]=='hymem-luna*' else '')
    monkeypatch.setattr(module.subprocess,'run',run)
    return fake


def test_v1_reproduces_false_running_on_retained_empty_service(tmp_path,monkeypatch):
    fake=fixture(tmp_path,monkeypatch,old)
    with pytest.raises(ValueError,match='prior_benchmark_unit_running'):
        old.host_admission(fake,tmp_path)


@pytest.mark.parametrize('mutation',[None,'matching_group','failed','inactive'])
def test_v2_requires_independent_empty_group_for_stopped_states(tmp_path,monkeypatch,mutation):
    module=revised();fake=fixture(tmp_path,monkeypatch,module,mutation)
    module.host_admission(fake,tmp_path)


@pytest.mark.parametrize('mutation',['live_pid','running','wrong_group','nested_process','nested_thread','nested_populated','symlink','traversal'])
def test_v2_rejects_all_live_or_unverifiable_prior_units(tmp_path,monkeypatch,mutation):
    module=revised();fake=fixture(tmp_path,monkeypatch,module,mutation)
    with pytest.raises(ValueError):module.host_admission(fake,tmp_path)
