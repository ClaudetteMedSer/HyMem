"""Independent root controls for the isolated two-file staged proxy repair."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')


def helper():
    spec = importlib.util.spec_from_file_location('derive_root', ROOT / 'tools/diagnostics/luna_staged_proxy_candidate_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def derived(tmp_path_factory):
    output = tmp_path_factory.mktemp('root-proxy') / 'bundle'
    result = helper().assemble(FROZEN, output)
    assert result['candidate_map_sha256'] == 'f22cd2be376f019efa1d39cb6c2f1e43ffef2d7ea3bb7a07ac64479241cd4b11'
    assert result['inventory_sha256'] == '1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd'
    return output


def test_root_full_nonempty_grounded_extraction_and_call_limit(derived):
    child = r'''
import importlib.util,json,sys
from pathlib import Path
derived,frozen=map(Path,sys.argv[1:]);sys.path[:0]=[str(derived/'candidate'),str(frozen/'code')]
from hymem.extraction import chunk,grounding_staged_v1 as staged
from hymem.dreaming.runner import _CountingPhase1LLM,_HeartbeatLLMClient
from hymem.extraction.producer import _phase1_proxy_source
from benchmarks.codex_subscription_warm_v9 import BudgetLimits,SharedBudget
spec=importlib.util.spec_from_file_location('frozen_root_runner',frozen/'code/tools/diagnostics/luna_lme_diagnostic_v8.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def exercise(cap):
    limits=BudgetLimits(12,160000,600);budget=SharedBudget(limits,max_in_flight=1);budget.register('q-0001',limits)
    class Dual:
        key='q-0001'
        def __init__(self):self.budget=budget;self.calls=[]
        def settle(self,kind):
            budget.reserve(self.key)
            budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna','config_isolation_admitted':True,
                'inference_enabled':False,'quota_windows':[{'remaining_percent':100}]})
            budget.settle(self.key,used=11,turn_started=True);self.calls.append(kind)
        def complete(self,request):
            self.settle('ordinary')
            return json.dumps({'triples':[{'subject':'Mira','predicate':'uses','object':'CairnDB','polarity':1}]
                if len(self.calls)==1 else [],'markers':[],'complete':True})
        def complete_stage(self,request,batch,stage,recheck):
            assert stage=='original' and not recheck
            staged.validate_original_request(request,batch)
            self.settle('structured')
            evidence=[{'source_message_id':batch.sources[0].source_message_id,'region':'owned','quote':'Mira uses CairnDB'}]
            support={'evidence':evidence,'checks':{name:{'state':'supported','evidence_indices':[0]}
                for name in ('attribution_and_roles','relation_and_polarity')}}
            return json.dumps({'schema':staged.ORIGINAL_SCHEMA,'batch_sha256':batch.batch_sha256,
                'complete':True,'originals':[{'index':0,'original':{'state':'supported','support':support}}]})
    dual=Dual();accounted=runner.AccountedClient(dual,derived/'candidate');memory=runner._memory_client({},accounted)
    beats=[];heartbeat=_HeartbeatLLMClient(memory,lambda:beats.append(1));counter=_CountingPhase1LLM(heartbeat)
    assert _phase1_proxy_source(counter) is heartbeat and _phase1_proxy_source(heartbeat) is memory
    result=chunk.extract_chunk(counter,'Mira uses CairnDB.',completion_call_limit=cap)
    assert accounted.reconcile() and budget.snapshot()['usage_complete']
    assert counter.completion_calls==counter.provider_attempts==len(dual.calls)
    assert len(beats)==2*len(dual.calls)
    return {'failed':result.failed,'reason':result.failure_reason,'triples':len(result.triples),
        'calls':dual.calls,'stages':accounted.counts,'stop':budget.stop_code}
print(json.dumps({'success':exercise(3),'limited':exercise(2)}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', child, str(derived), str(FROZEN)], capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state['success']['failed'] is False, state
    assert state['success']['triples'] == 1
    assert state['success']['calls'] == ['ordinary', 'ordinary', 'structured']
    assert state['success']['stages']['grounding_original_initial'] == {'attempts':1,'returned':1,'turns':1,'known_tokens':11}
    assert state['limited']['failed'] and state['limited']['reason'] == 'resource_limit'
    assert state['limited']['calls'] == ['ordinary', 'ordinary']


@pytest.mark.parametrize('corruption', ['extra_file','extra_dir','symlink','fifo','changed_source','changed_map'])
def test_derivation_rejects_corrupt_source_without_creating_output(tmp_path, corruption):
    clone = tmp_path / 'input'
    shutil.copytree(FROZEN / 'candidate', clone / 'candidate')
    shutil.copyfile(FROZEN / 'source-map.json', clone / 'source-map.json')
    candidate = clone / 'candidate'
    if corruption == 'extra_file':
        (candidate / 'unexpected.txt').write_text('invented')
    elif corruption == 'extra_dir':
        (candidate / 'unexpected').mkdir()
    elif corruption == 'symlink':
        (candidate / 'unexpected').symlink_to(candidate / 'hymem')
    elif corruption == 'fifo':
        os.mkfifo(candidate / 'unexpected')
    elif corruption == 'changed_source':
        (candidate / 'hymem/extraction/producer.py').write_bytes(b'invented')
    else:
        (clone / 'source-map.json').write_bytes(b'{}')
    output = tmp_path / 'refused'
    with pytest.raises(ValueError):
        helper().assemble(clone, output)
    assert not output.exists()


def test_all_source_hashes_and_originals_preserved(derived):
    before = json.loads((FROZEN / 'source-map.json').read_text())['source_sha256']
    after = json.loads((derived / 'source-map.json').read_text())['source_sha256']
    assert len(before) == len(after) == 514
    assert {name for name in before if before[name] != after[name]} == {'hymem/dreaming/runner.py','hymem/extraction/producer.py'}
    for root, entries in [(FROZEN,before),(derived,after)]:
        for name, digest in entries.items():
            assert hashlib.sha256((root / 'candidate' / name).read_bytes()).hexdigest() == digest
