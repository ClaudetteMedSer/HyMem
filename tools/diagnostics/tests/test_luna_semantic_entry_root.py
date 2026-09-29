"""Independent whole-entry replay on physical candidate, entirely invented output."""
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics.tests.test_luna_semantic_canary_root import SCRIPT, CANDIDATE, ROOT


PROGRAM = SCRIPT.split('delegate = Client()')[0] + r'''
from dataclasses import asdict
import tempfile
from tools.diagnostics import luna_semantic_probe as core
from tools.diagnostics import luna_semantic_probe_run as entry
from tools.diagnostics import luna_semantic_probe_progress as reader
from tools.diagnostics import luna_semantic_probe_host as host
from tools.diagnostics import luna_semantic_cases as cases
from tools.diagnostics.tests.test_luna_semantic_probe_root import Judge
from hymem.extraction import grounding
from benchmarks import codex_subscription_warm_v3 as warm
retained=[]
first=Client()
original=first.complete
def capture(request):
    raw=original(request)
    try: wire=json.loads(request.user)
    except ValueError: wire=None
    if type(wire) is not dict or 'batch' not in wire:
        retained.append((asdict(request),raw))
    return raw
first.complete=capture
assert check.run_canary(gold,chunk,first,candidate=Path(sys.argv[1]))['passed']
assert len(retained)==8
closed=[]
class Paid:
    def __init__(self,key,cap,budget):
        self.key,self.budget=key,budget
        budget.register(key,warm.BudgetLimits(*cap))
        self.judge=(Client() if key=='canary' else Judge(cases.cases()[int(key.split('-')[-1])]))
    @property
    def observed_turns(self): return self.budget.snapshot()['questions'][self.key]['turns']
    @property
    def observed_tokens(self): return self.budget.snapshot()['questions'][self.key]['known_tokens']
    @property
    def usage_complete(self): return self.budget.snapshot()['questions'][self.key]['usage_complete']
    def complete(self,request):
        self.budget.reserve(self.key)
        self.budget.before_turn(self.key,dict(auth='chatgpt',model=warm.base.MODEL,
            config_isolation_admitted=True,inference_enabled=False,
            quota_windows=[dict(remaining_percent=90)]))
        self.budget.settle(self.key,used=9,turn_started=True)
        return self.judge.complete(request)
    def close(self): closed.append(self.key)
root=Path(tempfile.mkdtemp(prefix='hymem-semantic-entry-root-'))
receipt={'source_sha256':{}}
loaded=(host,core,cases,grounding,gold,chunk,check,stage,warm,warm.concurrent)
entry.preflight=lambda *a:(receipt,loaded,{'candidate_files':510},tuple(retained))
entry.verify_live_containment=lambda *a:None
# The actual candidate is immutable; only the synthetic temporary entry refers
# to it. Entry does not read through this link in the real preflight path.
(root/'candidate').symlink_to(Path(sys.argv[1]),target_is_directory=True)
fault=sys.argv[4]
write=host.write_once
def write_fault(path,value):
    if fault=='result_write' and path.name=='private-result.json':
        raise OSError('PRIVATE_WRITE_FAILURE')
    return write(path,value)
host.write_once=write_fault
out=entry.execute(root,'1'*64,client_factory=Paid,containment=lambda *a:None)
assert len(closed)==25 and len(set(closed))==25
assert out['paid_budget']['turns']==29
assert out['paid_budget']['known_tokens']==261
assert out['paid_budget']['in_flight']==out['paid_budget']['reserved']==0
assert out['paid_budget']['usage_complete']
assert not out['completed_and_clean']
assert 'PRIVATE_WRITE_FAILURE' not in json.dumps(out)
if fault=='none':
    assert out['core_completed'] and out['all_semantic_checks_passed']
    assert out['hybrid_replayed_ordinary_calls']==8
    assert out['hybrid_new_paid_grounding_calls']==3
    raw=json.loads((root/'run/private-result.json').read_bytes())
    import hashlib
    manifest=hashlib.sha256(b'{}').hexdigest()
    assert reader._terminal_valid(out,raw,'1'*64,manifest)
    assert len(list((root/'run/safe-progress').glob('*.json')))==25
    for mutate in ('core_type','budget_type','stop','pass_with_failure'):
        import copy
        damaged=copy.deepcopy(raw)
        if mutate=='core_type': damaged['core_completed']=1
        elif mutate=='budget_type': damaged['paid_budget']['turns']=29.0
        elif mutate=='stop': damaged['stop_code']='cleanup_failure'
        else: damaged['control_results'][0]['passed']=False
        assert not reader._terminal_valid(out,damaged,'1'*64,manifest),mutate
    for mutate in ('stop_both','schedule_both','usage_sum'):
        damaged=copy.deepcopy(raw)
        if mutate=='stop_both': damaged['stop_code']='cleanup_failure'
        elif mutate=='schedule_both': damaged['hybrid']['hybrid_schedule_complete']=False
        else: damaged['control_results'][0]['known_tokens']+=1
        projected=entry._safe_result(damaged,'1'*64,manifest,
            out['process_identity_sha256'],True,1.0,warm.serialize_failure)
        assert not reader._terminal_valid(projected,damaged,'1'*64,manifest),mutate
    # Test the separate root verifier against this complete private journal.
    # Only its already-tested preflight is replaced with an explicit test shim.
    import builtins
    from tools.diagnostics import luna_semantic_verdict_replay_root as verifier
    builtins._root_replay_preflight=entry.preflight
    shim=root/'code/tools/diagnostics/luna_semantic_probe_run.py'
    shim.parent.mkdir(parents=True)
    shim.write_text('import builtins\npreflight=builtins._root_replay_preflight\n')
    shim_sha=hashlib.sha256(shim.read_bytes()).hexdigest()
    reviewed=verifier.replay(root,'1'*64,shim_sha)
    assert reviewed['verified'] and reviewed['captured_returned_judgments_replayed']==29
    assert reviewed['new_model_calls']==0 and reviewed['all_semantic_checks_passed']
    for changed in ('ordinary_path_valid','grounding_trace','aggregate'):
        damaged=copy.deepcopy(raw)
        if changed=='aggregate':
            damaged['all_semantic_checks_passed']=False
        elif changed=='grounding_trace':
            damaged['hybrid']['canary']['grounding_trace']=[]
        else:
            damaged['hybrid']['canary']['ordinary_path_valid']=False
        (root/'run/private-result.json').write_text(json.dumps(damaged))
        try: verifier.replay(root,'1'*64,shim_sha)
        except AssertionError: pass
        else: raise AssertionError('replay accepted '+changed)
        (root/'run/private-result.json').write_text(json.dumps(raw))
else:
    assert not out['core_completed']
    assert out['stop_code']=='private_result_write_failure'
print(json.dumps({'offline':True,'units':25,'new_synthetic_turns':29,'known_synthetic_tokens':261}))
'''


@pytest.mark.parametrize('fault',['none','result_write'])
def test_root_whole_entry_synthetic_accounting_and_persistence(fault):
    result=subprocess.run([sys.executable,'-I','-B','-c',PROGRAM,
        str(CANDIDATE),str(ROOT),'second',fault],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    assert '"offline": true' in result.stdout
