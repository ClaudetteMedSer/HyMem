"""Root execution/failure controls with invented input and budgeted fake clients."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_staged_run_v1 as entry,luna_staged_core_v1 as core
from tools.diagnostics import luna_semantic_cases as cases
from tools.diagnostics.tests.test_luna_staged_core_v1 import FakeClient,_canary_batches


def setup_entry(monkeypatch,tmp_path):
    root=tmp_path/'run-root';root.mkdir(mode=0o700)
    writes=[]
    def write_once(path,value):
        with path.open('x') as stream:json.dump(value,stream,sort_keys=True)
        writes.append(path)
    host=SimpleNamespace(BINARY=Path('/never-execute'),write_once=write_once,
        root_valid=lambda p:p==root,HEX=__import__('re').compile(r'[0-9a-f]{64}\Z'))
    transport=core.transport;warm=transport.warm
    loaded=(host,core,object(),cases,object(),object(),None,transport,warm,warm.concurrent)
    receipt={'source_sha256':{'test-only':'1'*64}}
    monkeypatch.setattr(entry,'preflight',lambda *a,**kw:(receipt,loaded,{'candidate_files':514},()))
    reference=SimpleNamespace(_owned_absent=lambda _:True)
    monkeypatch.setattr(entry,'_reference',lambda *a:reference)
    monkeypatch.setattr(core,'derive_canary_batches',lambda **kw:_canary_batches())
    return root,host,reference,writes


@pytest.mark.parametrize('fault',['none','malformed','transport','unknown_usage','cleanup'])
@pytest.mark.parametrize('ordinal',[0,7])
def test_whole_entry_variable_turns_and_finite_failure_cleanup(monkeypatch,tmp_path,fault,ordinal):
    root,host,reference,writes=setup_entry(monkeypatch,tmp_path)
    closed=[]
    class Client(FakeClient):
        def __init__(self,key,cap,budget):
            super().__init__(key,cap,budget,malformed_at=f'unit-{ordinal:02d}' if fault=='malformed' else None,
                             fail_at=f'unit-{ordinal:02d}' if fault=='transport' else None)
        def complete_stage(self,request,batch,stage,recheck):
            if fault=='unknown_usage' and self.key==f'unit-{ordinal:02d}':
                self.budget.reserve(self.key)
                self.budget.before_turn(self.key,dict(auth='chatgpt',model=core.transport.warm.base.MODEL,
                    config_isolation_admitted=True,inference_enabled=False,
                    quota_windows=[dict(remaining_percent=90)]))
                self.budget.settle(self.key,used=None,turn_started=True)
                raise RuntimeError('PRIVATE_UNKNOWN')
            return super().complete_stage(request,batch,stage,recheck)
        def close(self):
            closed.append(self.key);super().close()
            if fault=='cleanup' and self.key==f'unit-{ordinal:02d}':raise RuntimeError('PRIVATE_CLEANUP')
    result=entry.execute(root,'1'*64,client_factory=Client,containment=lambda *a:None)
    completed=fault in ('none','malformed')
    turns=(16-(fault=='malformed')) if completed else 2*ordinal+(2 if fault=='cleanup' else 1)
    assert result['diagnostic_completed']==completed
    assert result['paid_budget']['turns']==turns
    assert result['paid_budget']['known_tokens']==100*(turns-(fault=='unknown_usage'))
    assert result['paid_budget']['usage_complete']==(fault!='unknown_usage')
    assert len(closed)==(8 if completed else ordinal+1)
    assert all(result[name] is False for name in ('completed_and_clean','semantic_accuracy_accepted','full_lme_ready'))
    assert 'PRIVATE_' not in json.dumps(result)
    assert json.loads((root/'safe-terminal.json').read_bytes())==result
    private=json.loads((root/'run/private-result.json').read_bytes())
    core.validate_public_result(private)
    assert private['paid_budget']==result['paid_budget']
    assert (root/'run/private-journal').stat().st_mode&0o077==0
    for path in (root/'run/private-journal').iterdir():assert path.stat().st_mode&0o077==0


@pytest.mark.parametrize('fault',['identity','identity_write'])
def test_default_factory_tracking_failure_always_closes_created_process(monkeypatch,tmp_path,fault):
    root,host,reference,writes=setup_entry(monkeypatch,tmp_path)
    warm=core.transport.warm;created=[];closed=[]
    class Session:
        def __init__(self,*a,**kw):self.process=SimpleNamespace(pid=8675309);created.append(self)
        def close(self):closed.append(self)
    monkeypatch.setattr(warm,'WarmSession',Session)
    def identity(pid):
        if fault=='identity':raise OSError('PRIVATE_PID_FAILURE')
        return dict(pid=pid,pgid=pid,starttime=1)
    reference._process_identity=identity
    original=host.write_once
    def write(path,value):
        if fault=='identity_write' and path.parent.name=='private-owned-processes':raise OSError('PRIVATE_WRITE_FAILURE')
        return original(path,value)
    host.write_once=write
    result=entry.execute(root,'1'*64,containment=lambda *a:None)
    assert len(created)==1 and created[0] in closed
    assert not result['diagnostic_completed'] and result['paid_budget']['turns']==0
    assert not result['completed_and_clean'] and result['stop_code'] is not None
    assert 'PRIVATE_' not in json.dumps(result)


def test_private_result_write_failure_cannot_be_reported_completed(monkeypatch,tmp_path):
    root,host,reference,writes=setup_entry(monkeypatch,tmp_path)
    original=host.write_once
    def write(path,value):
        if path.name=='private-result.json':raise OSError('PRIVATE_RESULT_FAILURE')
        return original(path,value)
    host.write_once=write
    result=entry.execute(root,'1'*64,client_factory=FakeClient,containment=lambda *a:None)
    assert result['paid_budget']['turns']==16 and result['paid_budget']['known_tokens']==1600
    assert result['diagnostic_completed'] is False
    assert result['stop_code']=='private_result_write_failure'
    assert 'PRIVATE_' not in json.dumps(result)


def test_preflight_only_does_not_derive_or_start_or_write(monkeypatch,tmp_path):
    root,host,reference,writes=setup_entry(monkeypatch,tmp_path)
    def deny(*a,**kw):raise AssertionError('must not run')
    monkeypatch.setattr(core,'derive_canary_batches',deny)
    monkeypatch.setattr(core,'run_campaign',deny)
    result=entry.execute(root,'1'*64,preflight_only=True,client_factory=deny,containment=deny)
    assert result['verified'] and result['model_calls']==0 and result['candidate_files']==514
    assert not writes and not (root/'run').exists()


def test_isolated_invalid_cli_has_finite_output_and_no_traceback(tmp_path):
    result=subprocess.run([sys.executable,'-I','-B',entry.__file__,'--root',str(tmp_path/'missing'),
        '--receipt-sha256','0'*64,'--preflight-only'],capture_output=True,text=True,timeout=15)
    assert result.returncode==1 and result.stderr==''
    output=json.loads(result.stdout)
    assert output['verified'] is False and output['model_calls']==0


def test_old_core_pin_matches_carried_sealed_bytes():
    from tools.diagnostics import luna_staged_bundle_v1 as bundle
    code=bundle.collect_sources(Path(__file__).resolve().parents[3])
    assert entry.OLD_CORE_SHA==hashlib.sha256(code['tools/diagnostics/luna_semantic_probe.py']).hexdigest()


def test_one_shot_marker_requires_boolean_before_any_control_plane_call(monkeypatch,tmp_path):
    (tmp_path/'launch-receipt.json').write_bytes(b'{}')
    (tmp_path/'launch-attempt.json').write_text(json.dumps(dict(
        receipt_sha256=hashlib.sha256(b'{}').hexdigest(),one_shot=1)))
    def deny(*a,**kw):raise AssertionError('invalid marker reached systemctl')
    monkeypatch.setattr(entry.subprocess,'run',deny)
    with pytest.raises(ValueError,match='launch_marker_invalid'):
        entry.verify_live_containment(tmp_path,dict(unit='never.service',expected_cgroup='/never'))


@pytest.mark.parametrize('fault',['none','transport','core','receipt','owner','alias'])
def test_early_pins_fail_before_import_and_bind_the_exact_transport(monkeypatch,tmp_path,fault):
    from tools.diagnostics import luna_staged_bundle_v1 as bundle
    import os,stat
    repo=Path(__file__).resolve().parents[3]
    code=bundle.collect_sources(repo)
    code['tools/diagnostics/luna_staged_run_v1.py']=Path(entry.__file__).read_bytes()
    code['tools/diagnostics/luna_staged_host_v1.py']=b'raise AssertionError("must not import here")\n'
    root=tmp_path/'.hymem-luna-staged-probe-test0000';root.mkdir(mode=0o700)
    if fault in ('transport','core'):
        name=('benchmarks/codex_subscription_staged_v1.py' if fault=='transport' else
              'tools/diagnostics/luna_staged_core_v1.py')
        code[name]+=b'\n# malicious drift\n'
    for name,raw in code.items():
        target=root/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    receipt={'source_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in code.items()}}
    raw=json.dumps(receipt).encode();(root/'launch-receipt.json').write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    if fault=='receipt':digest='0'*64
    monkeypatch.setattr(entry,'TASK_HOME_ROOT' if hasattr(entry,'TASK_HOME_ROOT') else 'HOME',tmp_path)
    original=Path.lstat
    def attributes(path,*args,**kw):
        value=original(path,*args,**kw)
        if path==root:
            return SimpleNamespace(st_mode=stat.S_IFLNK if fault=='alias' else value.st_mode,
                                   st_uid=1001 if fault=='owner' else 1000)
        return value
    monkeypatch.setattr(Path,'lstat',attributes)
    if fault=='none':assert entry._early(root,digest)==receipt
    else:
        with pytest.raises(ValueError):entry._early(root,digest)


@pytest.mark.parametrize('fault',['none','budget_binding','limits_binding','extraction_identity'])
def test_real_isolated_imports_and_postrun_workdir_policy(tmp_path,fault):
    """Only host/Linux and private-evidence seams are faked; imports are real."""
    from tools.diagnostics import luna_staged_bundle_v1 as bundle
    import shutil
    repo=Path(__file__).resolve().parents[3]
    code=bundle.collect_sources(repo)
    root=tmp_path/'isolated'
    shutil.copytree(bundle.CANDIDATE,root/'candidate')
    for name,raw in code.items():
        path=root/'code'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    source=root/'code/tools/diagnostics/luna_staged_run_v1.py'
    source.write_bytes(Path(entry.__file__).read_bytes())
    script=r'''
import hashlib,importlib.util,json,socket,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'candidate'),str(root/'code')]
def deny(*a,**kw):raise AssertionError('offline only')
socket.socket.connect=socket.create_connection=deny;subprocess.Popen=deny
from tools.diagnostics import luna_staged_run_v1 as e,luna_semantic_probe as old
pins={str(p.relative_to(root/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
      for p in (root/'code').rglob('*.py')}
receipt=dict(source_sha256=pins,binary_sha256='1'*64,
    extraction_identity=e.EXTRACTION_IDENTITY,inventory_sha256='2'*64,
    retained_sha256=dict(evidence='3'*64))
seen=[]
host=SimpleNamespace(root_valid=lambda _:True,HEX=e.HEX,strict_equal=lambda a,b:a==b,
    receipt_for=lambda *a:receipt,verify_bundle=lambda *a,**kw:seen.append(kw),
    OLD_EVIDENCE=SimpleNamespace(read_bytes=lambda:b'{}'),
    OLD_RECEIPT=SimpleNamespace(read_bytes=lambda:b'{}'),OLD_RESULT=SimpleNamespace(read_bytes=lambda:b'{}'))
e._early=lambda *a:receipt
e._module_from_file=lambda *a:host
old.verify_retained_bundle=lambda **kw:tuple(range(8))
from tools.diagnostics import luna_staged_core_v1 as core
from hymem.extraction import contract
fault=sys.argv[2]
if fault=='budget_binding':core.transport.warm.SharedBudget=type('ForeignBudget',(core.transport.warm.SharedBudget,),{})
elif fault=='limits_binding':core.transport.warm.BudgetLimits=type('ForeignLimits',(core.transport.warm.BudgetLimits,),{})
elif fault=='extraction_identity':contract.extraction_contract_identity=lambda *a:'wrong'
if fault!='none':
    try:e.preflight(root,'4'*64)
    except ValueError:pass
    else:raise AssertionError('runtime substitution accepted '+fault)
    print(json.dumps(dict(verified=True,model_calls=0)))
    sys.exit(0)
for post in (False,True):
    checked,loaded,proof,retained=e.preflight(root,'4'*64,postrun=post)
    assert checked==receipt and proof['candidate_files']==514 and len(retained)==8
    assert loaded[1].transport is loaded[7] and loaded[8] is loaded[7].warm
    assert Path(loaded[5].__file__)==root/'candidate/hymem/extraction/chunk.py'
assert seen==[dict(require_empty_workdirs=True),dict(require_empty_workdirs=False)]
print(json.dumps(dict(verified=True,model_calls=0)))
'''
    out=subprocess.run([sys.executable,'-I','-B','-c',script,str(root),fault],
                       capture_output=True,text=True,timeout=30)
    assert out.returncode==0,out.stderr
    assert json.loads(out.stdout)==dict(verified=True,model_calls=0)
