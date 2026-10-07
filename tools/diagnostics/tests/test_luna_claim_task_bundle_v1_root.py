"""Independent root integration tests; invented inputs and no model/network IO."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

import pytest

from tools.diagnostics import luna_claim_task_bundle_v1 as builder

REPO = Path(__file__).resolve().parents[3]
BASE = runpy.run_path(str(Path(__file__).with_name('test_luna_classification_bundle_v3_root.py')))
BUNDLE = Path(tempfile.mkdtemp(prefix='hymem-claim-task-root-')).resolve() / 'bundle'
builder.prepare(REPO, BUNDLE)
shutil.copytree(BASE['CANDIDATE'], BUNDLE / 'candidate')
shutil.copyfile(BASE['INVENTORY'], BUNDLE / 'candidate-source-map.json')


def execute(tail, *args):
    result = subprocess.run([sys.executable, '-I', '-B', '-c', BASE['PREAMBLE'] + tail,
        str(BUNDLE / 'candidate'), str(BUNDLE / 'code'), *args],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


SETUP = r'''
import tempfile,hashlib,importlib.util,builtins
from types import SimpleNamespace
from tools.diagnostics import luna_claim_task_run_v1 as entry,luna_claim_task_core_v1 as task
from tools.diagnostics import luna_claim_task_replay_v1 as replay,luna_semantic_probe_host as host
from tools.diagnostics import luna_semantic_probe_progress as reader
from benchmarks import codex_subscription_claim_task_v1 as transport
warm=transport.warm
actual_preflight=entry.preflight
retained=[]
first=CanaryClient()
ordinary=first.complete
def capture(request):
    raw=ordinary(request); retained.append((asdict(request),raw)); return raw
first.complete=capture
assert check.run_canary(gold,chunk,first,candidate=Path(sys.argv[1]))['passed']
assert len(retained)==8
root=Path(tempfile.mkdtemp(prefix='claim-task-whole-entry-root-'))
(root/'candidate').symlink_to(Path(sys.argv[1]),target_is_directory=True)
loaded=(host,task,core,cases,gold,chunk,check,transport,warm,warm.concurrent)
entry.preflight=lambda *a:({'source_sha256':{}},loaded,{'candidate_files':513},tuple(retained))
reference=SimpleNamespace(_owned_absent=lambda identities:True)
entry._reference=lambda *a:reference
closed=[]
class Fake:
    def __init__(self,key,cap,budget):
        self.key,self.budget=key,budget
        budget.register(key,warm.BudgetLimits(*cap))
    @property
    def observed_turns(self): return self.budget.snapshot()['questions'][self.key]['turns']
    @property
    def observed_tokens(self): return self.budget.snapshot()['questions'][self.key]['known_tokens']
    @property
    def usage_complete(self): return self.budget.snapshot()['questions'][self.key]['usage_complete']
    def complete_arm(self,arm,request,batch):
        task.contract.validate_arm_request(arm,request,batch)
        self.budget.reserve(self.key)
        self.budget.before_turn(self.key,dict(auth='chatgpt',model=warm.base.MODEL,
            config_isolation_admitted=True,inference_enabled=False,
            quota_windows=[dict(remaining_percent=90)]))
        self.budget.settle(self.key,used=None if fault=='unknown_usage' and self.key=='unit-03' else 7,turn_started=True)
        if fault=='transport' and self.key=='unit-03': raise RuntimeError('PRIVATE_TRANSPORT')
        if fault=='malformed' and self.key=='unit-03': return 'PRIVATE_MALFORMED'
        rows=[]
        for i,t in enumerate(batch.triples):
            item=dict(index=i,original=dict(state='not_established',support=None))
            if arm=='A': item['alternatives']={p:dict(state='not_established',support=None)
                for p in g.PREDICATE_ORDER if p!=t.predicate}
            rows.append(item)
        return json.dumps(dict(schema=task.contract.B_SCHEMA if arm=='B' else g.GROUNDING_CONTRACT_VERSION,
            batch_sha256=batch.batch_sha256,complete=True,classifications=rows))
    def close(self):
        closed.append(self.key)
        if fault=='cleanup' and self.key=='unit-03': raise RuntimeError('PRIVATE_CLEANUP')
def run():
    out=entry.execute(root,'1'*64,client_factory=Fake,containment=lambda *a:None)
    assert 'PRIVATE_' not in json.dumps(out)
    assert not out['completed_and_clean'] and not out['semantic_accuracy_accepted'] and not out['full_lme_ready']
    builtins._claim_task_root_preflight=entry.preflight
    shim=root/'code/tools/diagnostics/luna_claim_task_run_v1.py'
    shim.parent.mkdir(parents=True)
    shim.write_text('import builtins\npreflight=builtins._claim_task_root_preflight\n')
    return out,hashlib.sha256(shim.read_bytes()).hexdigest()
'''


@pytest.mark.parametrize('fault',['none','malformed','transport','unknown_usage','cleanup'])
def test_actual_entry_and_every_return_private_replay(fault):
    out = execute(SETUP + r'''
fault=sys.argv[3]
terminal,entry_sha=run()
proof=replay.replay(root,'1'*64,entry_sha)
assert proof['verified'] and proof['new_model_calls']==0
assert proof['complete']==(fault in ('none','malformed'))
assert terminal['diagnostic_completed']==proof['complete']
assert terminal['paid_budget']['turns']==(29 if proof['complete'] else 4)
assert terminal['paid_budget']['known_tokens']==(203 if proof['complete'] else 21 if fault=='unknown_usage' else 28)
assert terminal['paid_budget']['usage_complete']==(fault!='unknown_usage')
assert proof['returned_responses']==(29 if proof['complete'] else 3 if fault=='transport' else 4)
assert len(closed)==terminal['paid_budget']['turns']
raw=json.loads((root/'run/private-result.json').read_bytes())
assert reader._terminal_valid(terminal,raw,'1'*64,hashlib.sha256(b'{}').hexdigest())
print(json.dumps(dict(verified=True,model_calls=0)))
''', fault)
    assert out == {'verified': True, 'model_calls': 0}


@pytest.mark.parametrize('field',['known_tokens','usage_complete','reserved','in_flight','stopped','question_tokens','final_stopped'])
def test_replay_rejects_cumulative_ledger_tamper(field):
    out = execute(SETUP + r'''
fault='none'
terminal,entry_sha=run()
assert replay.replay(root,'1'*64,entry_sha)['complete']
field=sys.argv[3]
ordinal=28 if field=='final_stopped' else 5
path=next(p for p in sorted((root/'run/private-journal').glob(f'*unit-{ordinal:02d}.json'))
    if json.loads(p.read_bytes()).get('phase')=='unit_finished')
item=json.loads(path.read_bytes())
if field=='question_tokens': item['budget']['questions']['unit-02']['known_tokens']+=1
elif field=='known_tokens': item['budget'][field]+=1
elif field in ('reserved','in_flight'): item['budget'][field]=1
elif field=='usage_complete': item['budget'][field]=False
else: item['budget']['stopped']=True
path.write_text(json.dumps(item))
try: replay.replay(root,'1'*64,entry_sha)
except ValueError: pass
else: raise AssertionError('accepted ledger substitution '+field)
print(json.dumps(dict(rejected=True)))
''', field)
    assert out == {'rejected': True}


@pytest.mark.parametrize('fault',['identity','write'])
def test_default_factory_closes_process_when_tracking_fails(fault):
    assert execute(SETUP + r'''
fault=sys.argv[3]
started=[]; stopped=[]
class ProcessSession:
    def __init__(self,*a,**k):
        self.process=SimpleNamespace(pid=812345)
        started.append(True)
    def close(self): stopped.append(True)
warm.WarmSession=ProcessSession
def identity(pid):
    if fault=='identity': raise OSError('PRIVATE_IDENTITY')
    return dict(pid=pid,pgid=pid,starttime=1)
reference._process_identity=identity
write=host.write_once
def maybe_fail(path,value):
    if fault=='write' and path.parent.name=='private-owned-processes': raise OSError('PRIVATE_WRITE')
    return write(path,value)
host.write_once=maybe_fail
out=entry.execute(root,'1'*64,containment=lambda *a:None)
assert started==[True] and stopped
assert out['paid_budget']['turns']==0 and not out['diagnostic_completed']
assert not out['completed_and_clean'] and out['stop_code'] is not None
assert 'PRIVATE_' not in json.dumps(out)
print(json.dumps(dict(closed=True,model_calls=0)))
''', fault) == {'closed': True, 'model_calls': 0}


def test_startup_containment_bridge_uses_receipt_not_source_hash():
    path=BUNDLE/'adapter-v2.py'
    spec=importlib.util.spec_from_file_location('root_claim_startup',path)
    startup=importlib.util.module_from_spec(spec); spec.loader.exec_module(startup)
    from types import SimpleNamespace
    seen=[]
    module=SimpleNamespace(run=lambda *a,**k:None)
    reference=SimpleNamespace(subprocess=module,verify_live_containment=lambda *a:seen.append('contained'))
    root=Path(tempfile.mkdtemp(prefix='claim-startup-'))
    receipt={'source_sha256':{'tools/diagnostics/luna_semantic_probe_run.py':'a'*64}}
    (root/'launch-receipt.json').write_text(json.dumps(receipt))
    expected=hashlib.sha256((root/'launch-receipt.json').read_bytes()).hexdigest()
    def ref(actual_root,sha):
        assert actual_root==root and sha==expected
        return reference
    startup.contained(root,receipt,SimpleNamespace(_reference=ref))
    assert seen==['contained'] and reference.subprocess is module


@pytest.mark.parametrize('field',['extra_unit','partial_extra_field','prior_tokens','partial_turns'])
def test_partial_replay_rejects_unchecked_records(field):
    assert execute(SETUP + r'''
fault='transport'
terminal,entry_sha=run()
assert replay.replay(root,'1'*64,entry_sha)['verified']
files=sorted((root/'run/private-journal').glob('*.json'))
path=files[-1]; item=json.loads(path.read_bytes()); field=sys.argv[3]
if field=='extra_unit':
    extra=path.with_name(f'{len(files)+1:04d}-unit-04.json')
    extra.write_text(json.dumps(item))
elif field=='partial_extra_field':
    item['unreviewed']='PRIVATE'; path.write_text(json.dumps(item))
elif field=='prior_tokens':
    item['budget']['questions']['unit-00']['known_tokens']+=1
    path.write_text(json.dumps(item))
else:
    item['budget']['questions']['unit-03']['turns']=99
    path.write_text(json.dumps(item))
try: replay.replay(root,'1'*64,entry_sha)
except ValueError: pass
else: raise AssertionError('accepted partial substitution '+field)
print(json.dumps(dict(rejected=True)))
''', field) == {'rejected': True}


def test_postrun_preflight_retains_source_path_but_relaxes_only_workdir_emptiness():
    assert execute(SETUP + r'''
flags=[]
old_loaded=(host,core,cases,g,gold,chunk,check,stage,warm,warm.concurrent)
def reference_preflight(r,sha,*,require_empty_workdirs):
    flags.append(require_empty_workdirs)
    return {},old_loaded,{},tuple(retained)
reference.preflight=reference_preflight
actual_preflight(root,'1'*64)
actual_preflight(root,'1'*64,True)
assert flags==[True,False]
print(json.dumps(dict(verified=True)))
''') == {'verified': True}


def test_private_rehearsal_uses_real_candidate_with_no_calls_or_writes():
    assert execute(SETUP + r'''
builtins._claim_task_root_preflight=entry.preflight
shim=root/'code/tools/diagnostics/luna_claim_task_run_v1.py'
shim.parent.mkdir(parents=True)
shim.write_text('import builtins\npreflight=builtins._claim_task_root_preflight\n')
spec=importlib.util.spec_from_file_location('root_rehearsal',sys.argv[3])
helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
result=helper.verify(root,'1'*64,hashlib.sha256(shim.read_bytes()).hexdigest())
assert result['verified'] and result['synthetic_judgments']==29
assert result['retained_ordinary_replays']==8 and result['identical_input_pairs']==14
assert not (root/'run').exists()
assert result['model_calls']==result['files_written']==0
print(json.dumps(dict(verified=True,model_calls=0)))
''',str(REPO/'tools/diagnostics/luna_claim_task_rehearsal_root.py')) == {'verified':True,'model_calls':0}


@pytest.mark.parametrize('relative,args',[
    ('code/tools/diagnostics/luna_claim_task_run_v1.py',['--receipt-sha256','0'*64,'--preflight-only']),
    ('code/tools/diagnostics/luna_claim_task_replay_v1.py',['--receipt-sha256','0'*64,'--entry-sha256','0'*64]),
    ('code/tools/diagnostics/luna_semantic_probe_progress.py',['--receipt-sha256','0'*64,'--expected-source-pins-sha256','0'*64]),
])
def test_isolated_entrypoints_fail_finitely_before_import_or_io(relative,args):
    result=subprocess.run([sys.executable,'-I','-B',str(BUNDLE/relative),
        '--root',str(BUNDLE/'missing'),*args],capture_output=True,text=True,timeout=15)
    assert result.returncode==1 and result.stderr==''
    out=json.loads(result.stdout)
    assert out.get('verified',out.get('validated')) is False
