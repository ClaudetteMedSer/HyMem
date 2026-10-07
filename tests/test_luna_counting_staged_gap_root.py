"""Root reproduction: frozen dream counter drops structured completion."""
import json
from pathlib import Path
import subprocess
import sys


BUNDLE = Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')
ROOT = Path(__file__).resolve().parents[1]


def test_nonempty_extraction_cannot_reach_staged_transport_through_counter():
    child = r'''
import importlib.util,json,sys
from pathlib import Path
bundle=Path(sys.argv[1]);root=Path(sys.argv[2])
sys.path[:0]=[str(bundle/'candidate'),str(bundle/'code')]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value);return value
runner=load('frozen_runner',bundle/'code/tools/diagnostics/luna_lme_diagnostic_v8.py')
capture=load('capture',root/'tools/diagnostics/luna_application_fault_capture_v1.py')
from hymem.extraction import chunk
from hymem.dreaming.runner import _CountingPhase1LLM,_HeartbeatLLMClient
from benchmarks.codex_subscription_warm_v9 import SharedBudget,BudgetLimits
limits=BudgetLimits(12,160000,600)
def exercise(observed):
    budget=SharedBudget(limits,max_in_flight=1);budget.register('q-0001',limits)
    class Dual:
        key='q-0001'
        def __init__(self):self.budget=budget;self.ordinary=0;self.structured=0
        def complete(self,request):
            budget.reserve(self.key)
            budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna',
                'config_isolation_admitted':True,'inference_enabled':False,
                'quota_windows':[{'remaining_percent':100}]})
            self.ordinary+=1;budget.settle(self.key,used=10,turn_started=True)
            return json.dumps({'triples':[{'subject':'user','predicate':'uses',
                'object':'sqlite','polarity':1}] if self.ordinary==1 else [],
                'markers':[],'complete':True})
        def complete_stage(self,*args):
            self.structured+=1
            raise AssertionError('must not reach staged transport in frozen chain')
    dual=Dual();accounted=runner.AccountedClient(dual,bundle/'candidate')
    memory=runner._memory_client({},accounted)
    heartbeat=_HeartbeatLLMClient(memory,lambda:None)
    client=_CountingPhase1LLM(heartbeat)
    probe=capture.FirstApplicationFaultCapture({'candidate':bundle/'candidate',
        'code':bundle/'code','chunk':chunk,'runner':runner},budget)
    if observed:
        try:
            with probe:chunk.extract_chunk(client,'The user uses sqlite.',completion_call_limit=4)
        except capture.FirstApplicationFaultStop:pass
        else:raise AssertionError('no first fault')
        failure=None
    else:
        result=chunk.extract_chunk(client,'The user uses sqlite.',completion_call_limit=4)
        failure=result.failure_reason
    return {'ordinary':dual.ordinary,'structured':dual.structured,
        'failure':failure,'observation':probe.snapshot(),'accounting':accounted.counts,
        'reconciled':accounted.reconcile(),'usage_complete':budget.snapshot()['usage_complete'],
        'counter_has_staged':hasattr(client,'complete_stage'),
        'heartbeat_has_staged':hasattr(heartbeat,'complete_stage')}
print(json.dumps({'original':exercise(False),'captured':exercise(True)}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', child,
                             str(BUNDLE), str(ROOT)], capture_output=True,
                            text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    states = json.loads(result.stdout)
    for value in states.values():
        assert value['ordinary'] == 2 and value['structured'] == 0
        assert value['reconciled'] and value['usage_complete']
        assert value['counter_has_staged'] is False
        assert value['heartbeat_has_staged'] is True
    assert states['original']['failure'] == 'call_failure'
    first = states['captured']['observation']['first']
    assert first['kind'] == 'call_failure'
    assert first['exception'] == 'attribute_error'
