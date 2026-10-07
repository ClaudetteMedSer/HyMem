"""Independent controls for first-fault capture on frozen source bytes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')
CAPTURE = ROOT / 'tools/diagnostics/luna_application_fault_capture_v1.py'


def _module():
    spec = importlib.util.spec_from_file_location('root_fault_capture', CAPTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHILD = r'''
import importlib.util,json,sys
from pathlib import Path
bundle=Path(sys.argv[1])
sys.path[:0]=[str(bundle/'candidate'),str(bundle/'code')]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod
cap=load('root_capture',Path(sys.argv[2]))
runner=load('root_frozen_runner',bundle/'code/tools/diagnostics/luna_lme_diagnostic_v8.py')
from hymem.extraction import chunk
from hymem.dreaming.runner import _CountingPhase1LLM,_HeartbeatLLMClient
from benchmarks.codex_subscription_warm_v9 import SharedBudget,BudgetLimits
from benchmarks.longmemeval_adapter import run_cleanup_actions
limits=BudgetLimits(12,160000,600)
loaded={'candidate':bundle/'candidate','code':bundle/'code','chunk':chunk}
results={}
for mode in ('healthy','before','after','named_impostor','cleanup'):
    budget=SharedBudget(limits,max_in_flight=1)
    budget.register('q-0001',limits)
    class Dual:
        key='q-0001'
        def __init__(self):self.budget=budget;self.calls=0
        def complete(self,request):
            if mode=='before':raise TypeError('PRIVATE-PRE-RETURN')
            budget.reserve(self.key)
            budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna',
                'config_isolation_admitted':True,'inference_enabled':False,
                'quota_windows':[{'remaining_percent':100}]})
            self.calls+=1
            budget.settle(self.key,used=10,turn_started=True)
            return '{"triples":[],"markers":[],"complete":true}'
    dual=Dual()
    accounted=runner.AccountedClient(dual,bundle/'candidate')
    memory=runner._memory_client({},accounted)
    beats=[0]
    def heartbeat():
        beats[0]+=1
        if mode in ('after','named_impostor','cleanup') and beats[0]==2:
            cls=type('ValueError',(Exception,),{}) if mode=='named_impostor' else RuntimeError
            raise cls('PRIVATE-AFTER-RETURN')
    client=_CountingPhase1LLM(_HeartbeatLLMClient(memory,heartbeat))
    original=chunk._failure
    observation=cap.FirstApplicationFaultCapture(loaded,budget)
    sentinel=False; cleanup=[]
    with observation:
        try:
            try:
                result=chunk.extract_chunk(client,'An invented sentence.',completion_call_limit=2)
            finally:
                primary=sys.exception()
                def close():
                    cleanup.append(True)
                    if mode=='cleanup':raise ValueError('PRIVATE-CLEANUP')
                run_cleanup_actions([('dream_fork_close',close)],primary_exception=primary)
        except cap.FirstApplicationFaultStop:
            sentinel=True
    state=observation.snapshot()
    copied=observation.snapshot()
    if copied['first'] is not None:
        copied['first']['exception']='PRIVATE-MUTATION'
        assert observation.snapshot()==state
    assert chunk._failure is original
    refused=False
    if sentinel:
        try:budget.reserve('q-0001')
        except BaseException:refused=True
    results[mode]={'state':state,'sentinel':sentinel,'calls':dual.calls,
        'turns':budget.snapshot()['turns'],'usage_complete':budget.snapshot()['usage_complete'],
        'refused':refused,'cleanup':len(cleanup),'reconciled':accounted.reconcile()}
encoded=json.dumps(results)
assert 'PRIVATE' not in encoded
print(encoded)
'''


@pytest.mark.skipif(not BUNDLE.is_dir(), reason='frozen bundle missing')
def test_frozen_chain_stops_before_recovery_and_preserves_primary():
    proc = subprocess.run([sys.executable, '-I', '-B', '-c', CHILD,
                           str(BUNDLE), str(CAPTURE)], capture_output=True,
                          text=True, timeout=40)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result['healthy']['state']['first'] is None
    assert result['healthy']['calls'] == 2
    for mode, label, turns in [('before', 'type_error', 0),
                               ('after', 'runtime_error', 1),
                               ('named_impostor', 'other', 1),
                               ('cleanup', 'runtime_error', 1)]:
        value = result[mode]
        assert value['sentinel'] and value['refused']
        assert value['state']['first']['exception'] == label
        assert value['turns'] == turns
        assert value['cleanup'] == 1
        assert value['usage_complete'] and value['reconciled']
    assert result['after']['state']['first']['phase'] != 'extraction_completion'


@pytest.mark.parametrize('bad', [True, False, 0, 1, [], 'PRIVATE', None])
def test_validator_rejects_non_mapping(bad):
    assert _module().validate_snapshot(bad) is None


@pytest.mark.parametrize('field', ['schema','first','cleanup'])
def test_validator_rejects_free_text_and_extra_keys(field):
    cap = _module()
    good = {'schema': cap.SCHEMA, 'first': None, 'cleanup': None}
    assert cap.validate_snapshot(good) == good
    assert cap.validate_snapshot({**good, field: 'PRIVATE'}) is None
    assert cap.validate_snapshot({**good, 'private_message': 'PRIVATE'}) is None


@pytest.mark.parametrize('field', ['kind','phase','exception','gate_family'])
@pytest.mark.parametrize('bad', [True, 1, [], {}, 'PRIVATE', None])
def test_closed_record_vocabulary(field, bad):
    cap = _module()
    record={'kind':'call_failure','phase':'unknown','exception':'other','gate_family':'none'}
    data={'schema':cap.SCHEMA,'first':{**record,field:bad},'cleanup':None}
    assert cap.validate_snapshot(data) is None
