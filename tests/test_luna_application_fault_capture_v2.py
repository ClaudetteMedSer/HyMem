"""Offline source-binding and phase controls for the repaired isolated candidate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = Path('/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle')
FROZEN = Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')
V9 = ROOT / 'tools/diagnostics/luna_lme_diagnostic_v9.py'
V2 = ROOT / 'tools/diagnostics/luna_application_fault_capture_v2.py'
V1 = ROOT / 'tools/diagnostics/luna_application_fault_capture_v1.py'


@pytest.fixture
def assembly(tmp_path):
    if not CANDIDATE.is_dir() or not FROZEN.is_dir():
        pytest.skip('accepted source bundles unavailable')
    root = tmp_path / 'bundle'
    shutil.copytree(CANDIDATE / 'candidate', root / 'candidate')
    shutil.copytree(FROZEN / 'code', root / 'code')
    shutil.copy2(CANDIDATE / 'source-map.json', root / 'source-map.json')
    shutil.copy2(V9, root / 'code/tools/diagnostics/luna_lme_diagnostic_v9.py')
    return root


CHILD = r'''
import importlib.util,json,sys
from pathlib import Path
bundle=Path(sys.argv[1])
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
runner=load('bound_runner',bundle/'code/tools/diagnostics/luna_lme_diagnostic_v9.py')
capture=load('bound_capture',Path(sys.argv[2]))
inventory=bundle/'source-map.json'
loaded=runner.import_source_only(bundle,inventory,runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and loaded['questions']==[]
try:runner.load_verified(bundle,inventory,runner.ACCEPTED_INVENTORY_SHA256,None,None,None,runner.DIAGNOSTIC_HELPER_SHA256)
except ValueError:pass
else:raise AssertionError('source-only admitted as runnable')
from hymem.extraction import chunk,grounding_staged_v1 as staged
from hymem.dreaming.runner import _CountingPhase1LLM,_HeartbeatLLMClient
from benchmarks.codex_subscription_warm_v9 import BudgetLimits,SharedBudget
loaded['runner']=runner
limits=BudgetLimits(12,160000,600)
results={}
for mode in ('success','ordinary_after','structured_before','structured_after'):
    budget=SharedBudget(limits,max_in_flight=1);budget.register('q-0001',limits)
    class Dual:
        key='q-0001'
        def __init__(self):self.budget=budget;self.calls=[]
        def settle(self,kind):
            budget.reserve(self.key)
            budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna',
                'config_isolation_admitted':True,'inference_enabled':False,
                'quota_windows':[{'remaining_percent':100}]})
            budget.settle(self.key,used=11,turn_started=True);self.calls.append(kind)
        def complete(self,request):
            self.settle('ordinary')
            return json.dumps({'triples':[{'subject':'Mira','predicate':'uses','object':'CairnDB','polarity':1}]
                if len(self.calls)==1 else [],'markers':[],'complete':True})
        def complete_stage(self,request,batch,stage,recheck):
            staged.validate_original_request(request,batch)
            self.settle('structured')
            evidence=[{'source_message_id':batch.sources[0].source_message_id,'region':'owned','quote':'Mira uses CairnDB'}]
            support={'evidence':evidence,'checks':{name:{'state':'supported','evidence_indices':[0]}
                for name in ('attribution_and_roles','relation_and_polarity')}}
            return json.dumps({'schema':staged.ORIGINAL_SCHEMA,'batch_sha256':batch.batch_sha256,
                'complete':True,'originals':[{'index':0,'original':{'state':'supported','support':support}}]})
    dual=Dual();accounted=runner.AccountedClient(dual,bundle/'candidate')
    memory=runner._memory_client({},accounted)
    beats=[0]
    def heartbeat():
        beats[0]+=1
        if (mode=='ordinary_after' and beats[0]==2 or
            mode=='structured_before' and beats[0]==5 or
            mode=='structured_after' and beats[0]==6):
            raise RuntimeError('PRIVATE-HEARTBEAT-ERROR')
    client=_CountingPhase1LLM(_HeartbeatLLMClient(memory,heartbeat))
    probe=capture.FirstApplicationFaultCapture(loaded,budget)
    stopped=False
    with probe:
        try:chunk.extract_chunk(client,'Mira uses CairnDB.',completion_call_limit=4)
        except capture.FirstApplicationFaultStop:stopped=True
    results[mode]={'snapshot':probe.snapshot(),'stopped':stopped,'calls':dual.calls,
        'beats':beats[0],'reconciled':accounted.reconcile()}
encoded=json.dumps(results,sort_keys=True)
assert 'PRIVATE' not in encoded
print(encoded)
'''


def test_repaired_chain_heartbeat_phase_and_finite_projection(assembly):
    result = subprocess.run([sys.executable, '-I', '-B', '-c', CHILD,
                             str(assembly), str(V2)], capture_output=True,
                            text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values['success']['snapshot']['first'] is None
    assert values['success']['calls'] == ['ordinary', 'ordinary', 'structured']
    for mode in ('ordinary_after', 'structured_before', 'structured_after'):
        value = values[mode]
        assert value['stopped'] and value['reconciled']
        assert value['snapshot']['first'] == {'kind':'call_failure',
            'phase':'heartbeat_boundary','exception':'runtime_error','gate_family':'none'}
    assert values['structured_before']['calls'] == ['ordinary', 'ordinary']
    assert values['structured_after']['calls'] == ['ordinary', 'ordinary', 'structured']


def test_source_and_historical_hashes(assembly):
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == 'a4436c0187852b8f342d166bd6fcb245092c29ead127891753ce263f02ed1e18'
    assert hashlib.sha256((FROZEN/'code/tools/diagnostics/luna_lme_diagnostic_v8.py').read_bytes()).hexdigest() == '7f96f2ac53039805d8324055edcc0902d7210195e075300eca1b0fb961764f82'
    assert hashlib.sha256((assembly/'source-map.json').read_bytes()).hexdigest() == '1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd'
    for name,digest in [('hymem/dreaming/runner.py','94c36844910d962d21a08df657ba03beff16731028b83f21f9c15386074965f5'),
                        ('hymem/extraction/producer.py','d19ee1e4a61201ba12fc7fcbc13d26a017d7e2fefef26a1b67a00f98e466b695')]:
        assert hashlib.sha256((assembly/'candidate'/name).read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('tamper', [
    "loaded['staged']=None",
    "loaded['observer']=None",
    "loaded['producer']=None",
    "_HeartbeatLLMClient.complete_stage=lambda self,*args: None",
    "(bundle/'candidate/hymem/extraction/producer.py').write_bytes(b'forged')",
    "(bundle/'code/tools/diagnostics/luna_lme_diagnostic_v9.py').write_bytes(b'forged')",
])
def test_capture_refuses_forged_source_or_import(assembly, tamper):
    child = CHILD.replace("loaded['runner']=runner\n", "loaded['runner']=runner\n" + tamper + "\n")
    result = subprocess.run([sys.executable, '-I', '-B', '-c', child,
                             str(assembly), str(V2)], capture_output=True,
                            text=True, timeout=40)
    assert result.returncode != 0
    assert 'capture_' in result.stderr
    assert 'PRIVATE' not in result.stderr


def test_source_only_import_refuses_forged_map(assembly):
    (assembly/'source-map.json').write_bytes(b'{}')
    child = CHILD[:CHILD.index('from hymem.extraction import chunk')]
    result = subprocess.run([sys.executable, '-I', '-B', '-c', child,
                             str(assembly), str(V2)], capture_output=True,
                            text=True, timeout=40)
    assert result.returncode != 0
    assert 'input_identity_invalid' in result.stderr
