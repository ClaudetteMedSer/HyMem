"""Independent host/receipt/reader boundary tests; no external calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_application_fault_host_v1 as host
from tools.diagnostics import luna_application_fault_launch_v1 as launch
from tools.diagnostics import luna_application_fault_progress_v1 as progress
from tools.diagnostics import luna_application_fault_probe_v2 as probe
from tools.diagnostics import luna_application_fault_capture_v2 as capture
from tools.diagnostics import luna_application_fault_bundle_v1 as bundle

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')
CANDIDATE = Path('/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle')


def policy():
    return dict(ActiveState='active', SubState='running', MainPID='4321',
        ControlGroup='/user.slice/probe.service', NRestarts='0', Type='exec',
        MemoryMax='4294967296', TasksMax='256', CPUQuotaPerSecUSec='2s',
        KillMode='control-group', Restart='no', RemainAfterExit='yes',
        OOMPolicy='kill', RuntimeMaxUSec='12min 10s', TimeoutStopUSec='10s',
        UMask='0077', Result='success', ExecMainStatus='0')


def empty_group(path):
    path.mkdir(parents=True)
    for name, value in {'cgroup.procs':'', 'cgroup.threads':'',
        'cgroup.events':'populated 0\nfrozen 0\n', 'memory.max':'4294967296\n',
        'pids.max':'256\n', 'cpu.max':'200000 100000\n'}.items():
        (path/name).write_text(value)
    return path


@pytest.mark.parametrize('key,bad', [('TasksMax','512'),('MemoryMax','8589934592'),
    ('Restart','always'),('KillMode','process'),('OOMPolicy','continue'),
    ('RuntimeMaxUSec','infinity'),('TimeoutStopUSec','20s'),
    ('CPUQuotaPerSecUSec','4s'),('RemainAfterExit','no'),('UMask','0022')])
def test_exact_kernel_unit_limits(key,bad):
    values=policy(); assert host._policy_ok(values)
    values[key]=bad; assert not host._policy_ok(values)


@pytest.mark.parametrize('child_state', ['process','thread','populated','symlink'])
def test_main_exit_zero_never_proves_recursive_cleanup(tmp_path,monkeypatch,child_state):
    group=empty_group(tmp_path/'group'); child=empty_group(group/'nested')
    if child_state=='process': (child/'cgroup.procs').write_text('99\n')
    if child_state=='thread': (child/'cgroup.threads').write_text('99\n')
    if child_state=='populated': (child/'cgroup.events').write_text('populated 1\n')
    if child_state=='symlink': (child/'unexpected').symlink_to(tmp_path/'missing')
    values={**policy(),'SubState':'exited','MainPID':'0'}
    monkeypatch.setattr(host,'_unit_values',lambda _:values)
    monkeypatch.setattr(host,'_group',lambda _:group)
    result=host.terminal_runtime({'unit':'probe.service','expected_cgroup':values['ControlGroup']})
    assert result['runtime_exit']=='success' and result['unit_stopped']
    assert not result['recursive_cleanup_verified']


def receipt_fixture(tmp_path,monkeypatch):
    root=tmp_path/'.hymem-luna-application-fault-v1-abcd1234';root.mkdir(mode=0o700)
    script=root/'application-fault-host-v1.py';script.write_bytes(b'# fixture\n');script.chmod(0o600)
    monkeypatch.setattr(host,'HOST_HOME',tmp_path);monkeypatch.setattr(host,'HOST_UID',os.getuid())
    monkeypatch.setattr(host,'__file__',str(script));monkeypatch.setattr(host,'verify_sources',lambda _:'a'*64)
    receipt=host.receipt_for(root,hashlib.sha256(script.read_bytes()).hexdigest(),'probe','b'*64)
    return root,receipt


def canonical(path,value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    path.write_bytes(raw);path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('mutation',['none','float','bool','duplicate','wrong_row_index'])
def test_receipt_pins_types_and_canonical_bytes(tmp_path,monkeypatch,mutation):
    root,value=receipt_fixture(tmp_path,monkeypatch)
    if mutation=='float':value['limits']['turns']=12.0
    if mutation=='bool':value['source_question_index']=True
    if mutation=='wrong_row_index':value['source_question_index']=0
    digest=canonical(root/'launch-receipt.json',value)
    if mutation=='duplicate':
        path=root/'launch-receipt.json';data=path.read_bytes().replace(b'"one_shot":true',b'"one_shot":false,"one_shot":true')
        path.write_bytes(data);digest=hashlib.sha256(data).hexdigest()
    if mutation=='none':assert host.verify_receipt(root,digest,'probe')==value
    else:
        with pytest.raises(ValueError):host.verify_receipt(root,digest,'probe')


def test_dispatch_marker_precedes_ambiguous_systemd_start(tmp_path,monkeypatch):
    root=tmp_path;digest='a'*64
    path=root/'containment-receipt.json';path.write_bytes(b'fixture')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    receipt={'unit':'test.service'}
    fake=SimpleNamespace(verify_receipt=lambda *a:receipt,verify_sources=lambda *a:None)
    monkeypatch.setattr(launch,'_host',lambda _:fake)
    monkeypatch.setattr(launch,'host_admission',lambda *a:None)
    monkeypatch.setattr(launch,'_workdirs',lambda *a:None)
    monkeypatch.setattr(launch,'_bus_env',lambda:{})
    calls=[]
    def run(command,**kwargs):
        assert json.loads((root/'containment-attempt.json').read_text())=={'one_shot':True,'receipt_sha256':digest}
        calls.append(command);raise subprocess.TimeoutExpired(command,20)
    monkeypatch.setattr(launch.subprocess,'run',run)
    result=launch.launch(root,digest,'containment')
    assert len(calls)==1
    assert (root/'containment-attempt.json').stat().st_mode&0o777==0o600
    with pytest.raises(ValueError):launch.launch(root,digest,'containment')
    assert len(calls)==1
    command=calls[0]
    for flag in ('--property=RuntimeMaxSec=730s','--property=TimeoutStopSec=10s',
        '--property=TasksMax=256','--property=MemoryMax=4294967296',
        '--property=CPUQuota=200%','--property=KillMode=control-group','--property=Restart=no'):
        assert flag in command


def reader_fixture(tmp_path,monkeypatch,*,clean=True):
    digest='c'*64; path=tmp_path/'application-fault-host-v1.py'
    path.write_bytes(b'# synthetic pinned host\n');path.chmod(0o600)
    monkeypatch.setattr(progress,'HOST_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(progress.sys,'platform','linux')
    monkeypatch.setattr(progress.os,'getuid',lambda:1000);monkeypatch.setattr(progress.os,'geteuid',lambda:1000)
    monkeypatch.setattr(progress,'regular',lambda p,cap:p.is_file() and not p.is_symlink() and p.stat().st_size<=cap)
    fake=SimpleNamespace(_root=lambda _:None,verify_receipt=lambda *a:{'unit':'probe.service','selected_row_sha256':'0'*64},
        terminal_runtime=lambda _:dict(policy_verified=True,unit_stopped=True,group_matched=True,
            recursive_cleanup_verified=clean,runtime_exit='success'),_json_bytes=host._json_bytes,
        SCHEMA=host.SCHEMA,CAPTURE_REL=host.CAPTURE_REL,CAPTURE_SHA=host.CAPTURE_SHA,
        PROBE_REL=host.PROBE_REL,PROBE_SHA=host.PROBE_SHA)
    runtime=tmp_path/'runtime-fixture';runtime.write_bytes(b'invented-runtime-or-data')
    fake._regular=host._regular;fake.DATASET=str(runtime);fake.BINARY=str(runtime)
    fake.DATASET_SHA=hashlib.sha256(runtime.read_bytes()).hexdigest();fake.BINARY_SHA256=fake.DATASET_SHA
    def module(name,path,digest):
        if path.name=='application-fault-host-v1.py':return fake
        if path.name=='luna_application_fault_capture_v2.py':return capture
        if path.name=='luna_application_fault_probe_v2.py':return probe
        raise AssertionError('unexpected source import')
    monkeypatch.setattr(progress,'module',module)
    canonical(tmp_path/'launch-attempt.json',{'receipt_sha256':digest,'one_shot':True})
    canonical(tmp_path/'probe-execution-marker.json',{'receipt_sha256':digest,'execution_started':True})
    return digest,fake


def fault_snapshot():
    return {'schema':capture.SCHEMA,'first':{'kind':'top_level','phase':'dream',
        'exception':'runtime_error','gate_family':'none'},'cleanup':None}


def terminal_probe(*,fault=False):
    return {'schema':probe.SCHEMA,'status':'application_fault' if fault else 'inconclusive',
        'stop':'application_fault' if fault else 'none','phase':'dream' if fault else None,
        'selected_row_sha256':'0'*64,'first_snapshot':fault_snapshot() if fault else {
            'schema':capture.SCHEMA,'first':None,'cleanup':None},
        'turns':0,'known_tokens':0,'usage_complete':True,'in_flight':0,'reserved':0,
        'stages':{name:{key:0 for key in probe.COUNTS} for name in probe.STAGES},
        'resource':{'current':1,'peak':1,'limit':256,'denials':0},
        'checkpoint_durable':fault,'adapter_cleanup_ok':True,'client_cleanup_ok':True,
        'accounting_reconciled':True}


def test_missing_host_result_retains_first_fault_and_not_completion(tmp_path,monkeypatch):
    digest,_=reader_fixture(tmp_path,monkeypatch)
    private=tmp_path/'private-probe';private.mkdir(mode=0o700)
    canonical(private/'first-fault.json',fault_snapshot())
    value=progress.inspect(tmp_path,digest,'probe')
    assert value['first_fault']==fault_snapshot()
    assert value['result_verified'] is False and value['lme_completion_proved'] is False
    assert value['status']=='result_missing' and value['recursive_cleanup_verified'] is True


@pytest.mark.parametrize('clean',[True,False])
def test_reader_keeps_cleanup_distinct_from_capped_inconclusive(tmp_path,monkeypatch,clean):
    digest,_=reader_fixture(tmp_path,monkeypatch,clean=clean)
    private=tmp_path/'private-probe';private.mkdir(mode=0o700)
    value=terminal_probe();assert probe.validate_result(value,capture)
    canonical(private/'probe-result.json',value)
    canonical(tmp_path/'host-result.json',{'schema':host.SCHEMA,'mode':'probe','status':'inconclusive',
        'failure_code':None,'probe_result_present':True})
    result=progress.inspect(tmp_path,digest,'probe')
    assert result['status']=='inconclusive' and result['recursive_cleanup_verified'] is clean
    assert result['lme_completion_proved'] is False


@pytest.mark.parametrize('corruption',['private_field','duplicate','truncated'])
def test_reader_malformed_metadata_is_never_exported(tmp_path,monkeypatch,capsys,corruption):
    digest,_=reader_fixture(tmp_path,monkeypatch)
    path=tmp_path/'host-result.json'
    if corruption=='private_field':canonical(path,{'private':'NEVER-EXPORT-PRIVATE'})
    elif corruption=='duplicate':path.write_text('{"private":"NEVER-EXPORT-PRIVATE","private":0}')
    else:path.write_text('{"private":"NEVER-EXPORT-PRIVATE')
    path.chmod(0o600)
    assert progress.main(['--root',str(tmp_path),'--receipt-sha256',digest,'--mode','probe'])==1
    output=capsys.readouterr().out
    assert 'PRIVATE' not in output
    value=json.loads(output)
    assert value['status']=='incomplete_or_failed' and value['result_verified'] is False


@pytest.mark.parametrize('failure',['host_malformed','probe_malformed','late_host_fault','cleanup_fault'])
def test_later_failures_never_erase_durable_first_fault(tmp_path,monkeypatch,failure):
    digest,_=reader_fixture(tmp_path,monkeypatch)
    private=tmp_path/'private-probe';private.mkdir(mode=0o700)
    canonical(private/'first-fault.json',fault_snapshot())
    value=terminal_probe(fault=True)
    outer={'schema':host.SCHEMA,'mode':'probe','status':'application_fault',
        'failure_code':None,'probe_result_present':True}
    if failure=='cleanup_fault':
        value['status']='unverified';value['adapter_cleanup_ok']=False
        value['first_snapshot']['cleanup']={'phase':'cleanup','exception':'value_error'}
        outer['status']='unverified'
    if failure=='late_host_fault':
        outer.update(status='unverified',failure_code='terminal_containment_invalid')
    if failure=='host_malformed':outer={'private':'NEVER-EXPORT-PRIVATE'}
    canonical(private/'probe-result.json',value if failure!='probe_malformed' else {'private':'NEVER-EXPORT-PRIVATE'})
    canonical(tmp_path/'host-result.json',outer)
    result=progress.inspect(tmp_path,digest,'probe')
    assert result['first_fault']==fault_snapshot() and result['lme_completion_proved'] is False
    assert 'PRIVATE' not in json.dumps(result)
    assert result['status'] in {'incomplete_or_failed','unverified'}


@pytest.mark.parametrize('mutation',['none','private','bool','negative','denial','oom','missing'])
def test_containment_requires_finite_zero_denial_counters(tmp_path,monkeypatch,mutation):
    digest,fake=reader_fixture(tmp_path,monkeypatch)
    fake._valid_counters=host._valid_counters
    canonical(tmp_path/'containment-attempt.json',{'receipt_sha256':digest,'one_shot':True})
    canonical(tmp_path/'containment-execution-marker.json',{'receipt_sha256':digest,'execution_started':True})
    counters=dict(pids_denials=0,memory_oom=0,memory_oom_kill=0,pids_current=1,memory_current=1,memory_peak=1)
    if mutation=='private':counters['private']='PRIVATE'
    if mutation=='bool':counters['pids_current']=True
    if mutation=='negative':counters['pids_current']=-1
    if mutation=='denial':counters['pids_denials']=1
    if mutation=='oom':counters['memory_oom']=1
    if mutation=='missing':counters=None
    canonical(tmp_path/'containment-result.json',{'schema':host.SCHEMA,'mode':'containment',
        'verified':True,'model_calls':0,'resources':counters})
    if mutation=='none':assert progress.inspect(tmp_path,digest,'containment')['status']=='verified_clean'
    else:
        with pytest.raises(ValueError):progress.inspect(tmp_path,digest,'containment')


def test_source_only_bundle_has_no_data_runtime_or_launch_and_exact_import(tmp_path):
    target=tmp_path/'bundle'
    state=bundle.assemble(repo=ROOT,accepted_code=FROZEN/'code',candidate=CANDIDATE/'candidate',
        map_path=CANDIDATE/'source-map.json',output=target)
    assert state['candidate_files']==514 and state['model_calls']==0
    assert not state['dataset_present'] and not state['binary_present'] and not state['launch_receipt_present']
    program='''import importlib.util,sys,json
from pathlib import Path
root=Path(sys.argv[1]);path=root/'code/tools/diagnostics/luna_lme_diagnostic_v9.py'
spec=importlib.util.spec_from_file_location('bound',path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and loaded['questions']==[] and loaded['binary'] is None
print(json.dumps({'source_only':True,'model_calls':0}))
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',program,str(target)],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr[-5000:]
    assert json.loads(result.stdout)=={'source_only':True,'model_calls':0}
