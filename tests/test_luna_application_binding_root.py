"""Root source-only checks of the new loader and first-fault binding."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')
CANDIDATE = Path('/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle')


@pytest.fixture
def assembly(tmp_path):
    bundle = tmp_path / 'isolated'
    shutil.copytree(CANDIDATE / 'candidate', bundle / 'candidate')
    shutil.copytree(FROZEN / 'code', bundle / 'code')
    shutil.copyfile(CANDIDATE / 'source-map.json', bundle / 'source-map.json')
    for name in ('luna_lme_diagnostic_v9.py', 'luna_application_fault_capture_v2.py'):
        shutil.copyfile(ROOT / 'tools/diagnostics' / name, bundle / 'code/tools/diagnostics' / name)
    return bundle


PREFIX = r'''
import importlib.util,json,sys
from pathlib import Path
bundle=Path(sys.argv[1])
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
runner=load('root_verified_runner',bundle/'code/tools/diagnostics/luna_lme_diagnostic_v9.py')
cap=load('root_verified_capture',bundle/'code/tools/diagnostics/luna_application_fault_capture_v2.py')
loaded=runner.import_source_only(bundle,bundle/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] is True and loaded['dataset'] is None and loaded['binary'] is None and loaded['questions']==[]
loaded['runner']=runner
from hymem.dreaming.runner import _CountingPhase1LLM,_HeartbeatLLMClient
from hymem.extraction import chunk
warm=loaded['warm'];limits=warm.BudgetLimits(12,160000,600)
budget=warm.SharedBudget(limits,max_in_flight=1);budget.register('q-0001',limits)
'''


def run_child(assembly, script):
    return subprocess.run([sys.executable, '-I', '-B', '-c', PREFIX + script, str(assembly)], capture_output=True, text=True, timeout=40)


@pytest.mark.parametrize('impostor', [False, True])
def test_first_structured_error_keeps_accounting_checkpoint_and_cleanup_separate(assembly, impostor):
    script = r'''
class Dual:
    key='q-0001'
    def __init__(self):self.budget=budget;self.calls=0
    def settle(self):
        budget.reserve(self.key)
        budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna','config_isolation_admitted':True,
            'inference_enabled':False,'quota_windows':[{'remaining_percent':100}]})
        budget.settle(self.key,used=7,turn_started=True);self.calls+=1
    def complete(self,request):
        self.settle()
        return json.dumps({'triples':[{'subject':'user','predicate':'uses','object':'sqlite','polarity':1}]
            if self.calls==1 else [],'markers':[],'complete':True})
    def complete_stage(self,*args):
        self.settle()
        raise ERROR('PRIVATE-RAW-MESSAGE')
dual=Dual();accounted=runner.AccountedClient(dual,bundle/'candidate')
memory=runner._memory_client(loaded,accounted);client=_CountingPhase1LLM(_HeartbeatLLMClient(memory,lambda:None))
checkpoints=[];capture=cap.FirstApplicationFaultCapture(loaded,budget,on_first=checkpoints.append)
original=chunk._failure
try:
    with capture:chunk.extract_chunk(client,'The user uses sqlite.',completion_call_limit=4)
except cap.FirstApplicationFaultStop:pass
else:raise AssertionError('fault did not stop extraction')
assert chunk._failure is original
first=capture.snapshot()['first']
assert capture.record_top_level(KeyError('PRIVATE-LATER'),phase='dream') is False
assert capture.record_cleanup(ValueError('PRIVATE-CLEANUP')) is True
assert capture.snapshot()['first']==first and checkpoints[0]['cleanup'] is None
assert len(checkpoints)==1 and accounted.reconcile() and budget.snapshot()['usage_complete']
assert budget.stop_code=='application_fault' and dual.calls==3
assert client.completion_calls==3 and client.provider_attempts==3
try:budget.reserve('q-0001')
except warm.ConcurrentStop:pass
else:raise AssertionError('admission continued')
value={'capture':capture.snapshot(),'checkpoint':checkpoints[0]}
assert 'PRIVATE' not in json.dumps(value)
print(json.dumps(value))
'''
    error = "type('RuntimeError',(Exception,),{})" if impostor else 'RuntimeError'
    result = run_child(assembly, 'ERROR=' + error + '\n' + script)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['capture']['first'] == {'kind':'call_failure','phase':'accounting_boundary',
        'exception':'other' if impostor else 'runtime_error','gate_family':'none'}
    assert value['capture']['cleanup']['exception'] == 'value_error'


@pytest.mark.parametrize('tamper', [
    "loaded.pop('code')",
    "loaded['runner']=None",
    "runner.AccountedClient._call=lambda *a: None",
    "loaded['observer'].TimeoutSubscriptionClient._complete_locked=lambda *a: None",
    "loaded['staged'].StagedSubscriptionClient.complete_stage=lambda *a: None",
    "_CountingPhase1LLM.complete_stage=lambda *a: None",
])
def test_binding_rejects_missing_or_replaced_callable(assembly, tamper):
    result = run_child(assembly, tamper + "\ncap.FirstApplicationFaultCapture(loaded,budget)\n")
    assert result.returncode != 0
    assert 'capture_' in result.stderr


def test_new_cli_has_no_inference_dispatch():
    path = ROOT / 'tools/diagnostics/luna_lme_diagnostic_v9.py'
    result = subprocess.run([sys.executable,'-I','-B',str(path),'--help'], capture_output=True,text=True,timeout=10)
    assert result.returncode == 0
    assert '--run' not in result.stdout and '--receipt-sha256' not in result.stdout
    assert '--output-dir' not in result.stdout
