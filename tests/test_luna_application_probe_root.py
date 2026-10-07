"""Independent no-inference actual adapter/probe integration controls."""
import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
FROZEN=Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')
CANDIDATE=Path('/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle')


def module(name):
    path=ROOT/'tools/diagnostics'/name
    spec=importlib.util.spec_from_file_location('root_'+path.stem,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize('field,bad', [
    ('status','budget_stop'),('turns',True),('turns',13),('known_tokens',float('nan')),
    ('in_flight',1),('reserved',1),('usage_complete',False),('accounting_reconciled',False),
    ('adapter_cleanup_ok',False),('client_cleanup_ok',False),('resource',None),
    ('status','provider_denial'),('status','transport_stop'),('status','application_fault'),
    ('checkpoint_durable',True),('selected_row_sha256','PRIVATE'),
])
def test_root_validator_does_not_label_corrupt_terminal_as_bounded_result(field,bad):
    probe=module('luna_application_fault_probe_v2.py')
    capture=module('luna_application_fault_capture_v2.py')
    value={'schema':probe.SCHEMA,'status':'inconclusive','stop':'none','phase':None,
        'selected_row_sha256':'0'*64,
        'first_snapshot':{'schema':capture.SCHEMA,'first':None,'cleanup':None},
        'turns':0,'known_tokens':0,'usage_complete':True,'in_flight':0,'reserved':0,
        'stages':{name:{key:0 for key in probe.COUNTS} for name in probe.STAGES},
        'resource':{'current':1,'peak':1,'limit':256,'denials':0},
        'checkpoint_durable':False,'adapter_cleanup_ok':True,'client_cleanup_ok':True,
        'accounting_reconciled':True}
    assert probe.validate_result(value,capture)==value
    value[field]=bad
    with pytest.raises(ValueError):probe.validate_result(value,capture)


@pytest.fixture
def bundle(tmp_path):
    bundle=tmp_path/'bundle'
    shutil.copytree(FROZEN/'code',bundle/'code')
    shutil.copytree(CANDIDATE/'candidate',bundle/'candidate')
    shutil.copyfile(CANDIDATE/'source-map.json',bundle/'source-map.json')
    for name in ('luna_lme_diagnostic_v9.py','luna_application_fault_capture_v2.py','luna_application_fault_probe_v2.py'):
        shutil.copyfile(ROOT/'tools/diagnostics'/name,bundle/'code/tools/diagnostics'/name)
    return bundle


PREFIX=r'''
import importlib.util,json,sys
from pathlib import Path
bundle=Path(sys.argv[1]);mode=sys.argv[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
runner=load('root_bound_runner',bundle/'code/tools/diagnostics/luna_lme_diagnostic_v9.py')
capture=load('root_bound_capture',bundle/'code/tools/diagnostics/luna_application_fault_capture_v2.py')
probe=load('root_bound_probe',bundle/'code/tools/diagnostics/luna_application_fault_probe_v2.py')
loaded=runner.import_source_only(bundle,bundle/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
warm=loaded['warm']
'''


def isolated(bundle,mode,body,timeout=65):
    result=subprocess.run([sys.executable,'-I','-B','-c',PREFIX+body,str(bundle),mode],
        capture_output=True,text=True,timeout=timeout)
    assert result.returncode==0,result.stderr[-12000:]
    return json.loads(result.stdout)


def test_real_dual_constructor_registers_exactly_once_without_session(bundle):
    state=isolated(bundle,'constructor',r'''
limits=warm.BudgetLimits(12,160000,600);budget=warm.SharedBudget(limits,max_in_flight=1)
directory=bundle.parent/'transport-construction';directory.mkdir(mode=0o700)
client=runner.make_dual(loaded,budget,'q-0001',limits,directory)
assert set(budget.snapshot()['questions'])=={'q-0001'}
assert client.ordinary.session is None and client.staged.session is None
client.close()
assert budget.snapshot()['turns']==0
try:budget.register('q-0001',limits)
except ValueError:pass
else:raise AssertionError('duplicate registration guard weakened')
print(json.dumps({'turns':budget.snapshot()['turns'],'slots':len(budget.snapshot()['questions'])}))
''')
    assert state=={'turns':0,'slots':1}


@pytest.mark.parametrize('mode',['cap','application_fault'])
def test_actual_candidate_open_ingest_dream_and_cleanup_with_only_invented_fake_returns(bundle,mode):
    state=isolated(bundle,mode,r'''
calls=[];closed=[];resources=[]
class Dual:
    model='gpt-6-luna'
    def __init__(self,budget,key,limits):
        self.budget,self.key=budget,key
        budget.register(key,limits)  # Same owner as the actual ordinary transport.
    def complete(self,request):
        if mode=='application_fault' and len(calls)==3:
            raise RuntimeError('PRIVATE-INVENTED-APPLICATION-FAULT')
        self.budget.reserve(self.key)
        self.budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna',
            'config_isolation_admitted':True,'inference_enabled':False,
            'quota_windows':[{'remaining_percent':100}]})
        self.budget.settle(self.key,used=10,turn_started=True);calls.append('ordinary')
        return '{"triples":[],"markers":[],"complete":true}'
    def complete_stage(self,*args):raise AssertionError('empty fixture should not need structured grounding')
    def close(self):closed.append(True)
def fake(loaded,budget,key,limits,output,*args):return Dual(budget,key,limits)
runner.make_dual=fake
# Only external provenance and provider construction are substituted here.
# The real adapter, store, ingestion, dream, counting, capture and budget run.
probe.verify_sources=lambda *args:None
selected={'question':'Invented integration question','haystack_sessions':[
    [{'role':'user','content':('Invented narrative for bounded offline validation. '*35)}]
    for i in range(12)],'haystack_session_ids':[f'invented-{i}' for i in range(12)],
    'haystack_dates':['2024-01-01T00:00:00Z']*12}
probe.select_question=lambda _loaded:selected
def resource():
    resources.append(1)
    return {'current':1,'peak':1,'limit':256,'denials':0}
result=probe.run_probe(loaded,runner,capture,output=bundle.parent/'probe-output',
    containment_verified=True,binary_sha256='0'*64,resource_check=resource)
assert closed==[True]
assert 'PRIVATE' not in json.dumps(result)
assert result['accounting_reconciled'] and result['usage_complete']
assert result['adapter_cleanup_ok'] and result['client_cleanup_ok']
print(json.dumps({'result':result,'calls':len(calls),'resources':len(resources),
    'first_file':(bundle.parent/'probe-output/first-fault.json').is_file()}))
''')
    result=state['result']
    assert state['calls']==(12 if mode=='cap' else 3),state
    assert result['turns']==state['calls'] and result['known_tokens']==state['calls']*10
    assert state['resources']>=2*state['calls']+2
    if mode=='cap':
        assert result['status']=='inconclusive',state
        assert result['stop']=='campaign_budget_exhausted'
        assert not state['first_file'] and result['first_snapshot']['first'] is None
    else:
        assert result['status']=='application_fault',state
        assert result['stop']=='application_fault' and state['first_file']
        assert result['first_snapshot']['first']['exception']=='runtime_error'
