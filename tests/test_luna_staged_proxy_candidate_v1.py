"""Offline controls for the isolated, source-only staged proxy derivation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle")


def _helper():
    path = ROOT / "tools/diagnostics/luna_staged_proxy_candidate_v1.py"
    spec = importlib.util.spec_from_file_location("staged_proxy_candidate_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def derived_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("staged-proxy") / "bundle"
    _helper().assemble(FROZEN, output)
    return output


def _isolated(script: str, derived_bundle: Path):
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                             str(derived_bundle), str(FROZEN)], capture_output=True,
                            text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_exact_514_file_derivation_and_drift(tmp_path):
    module = _helper()
    output = tmp_path / "candidate-v1"
    receipt = module.assemble(FROZEN, output)
    assert receipt["candidate_files"] == 514 and receipt["model_calls"] == 0
    before = json.loads((FROZEN / "source-map.json").read_text())["source_sha256"]
    after = json.loads((output / "source-map.json").read_text())["source_sha256"]
    assert {key for key in before if before[key] != after[key]} == {module.RUNNER, module.PRODUCER}
    assert {str(path.relative_to(output / "candidate")) for path in (output / "candidate").rglob("*") if path.is_file()} == set(before)
    for key in before.keys() - {module.RUNNER, module.PRODUCER}:
        assert (output / "candidate" / key).read_bytes() == (FROZEN / "candidate" / key).read_bytes()
    for transformer, key in ((module.transform_runner, module.RUNNER),
                             (module.transform_producer, module.PRODUCER)):
        data = (FROZEN / "candidate" / key).read_bytes()
        assert transformer(data) == (output / "candidate" / key).read_bytes()
        try:
            transformer(data + b" ")
        except ValueError:
            pass
        else:
            raise AssertionError("drift accepted")


def test_real_imported_proxy_methods_and_integrity(derived_bundle):
    state = _isolated(r'''
import json,sys
from pathlib import Path
derived, frozen = map(Path, sys.argv[1:])
sys.path[:0] = [str(derived/'candidate'), str(frozen/'code')]
from hymem.dreaming.runner import _CountingPhase1LLM, _HeartbeatLLMClient
from hymem.extraction.producer import _phase1_proxy_source
class Dual:
    def __init__(self): self.calls=[]; self.fail=False
    def complete(self,request): self.calls.append(('ordinary',request)); return 'ordinary'
    def complete_stage(self,request,batch,stage,recheck):
        self.calls.append(('stage',request,batch,stage,recheck))
        if self.fail: raise ValueError('provider')
        return 'staged'
dual=Dual(); beats=[]
heartbeat=_HeartbeatLLMClient(dual,lambda:beats.append('beat'))
counter=_CountingPhase1LLM(heartbeat)
args=[object(),object(),object(),object()]
assert counter.complete_stage(*args)=='staged'
assert all(got is want for got,want in zip(dual.calls[-1][1:],args))
assert beats==['beat','beat']
dual.fail=True
try: counter.complete_stage(*args)
except ValueError: pass
else: raise AssertionError('provider failure swallowed')
assert beats==['beat']*4
assert counter.completion_calls==2 and counter.provider_attempts==2
assert counter.complete(args[0])=='ordinary'
assert counter.completion_calls==3 and counter.provider_attempts==3
assert _phase1_proxy_source(counter) is heartbeat
old=_CountingPhase1LLM.complete_stage
_CountingPhase1LLM.complete_stage=lambda self,*args:'patched'
assert _phase1_proxy_source(counter) is None
_CountingPhase1LLM.complete_stage=old
assert _phase1_proxy_source(counter) is heartbeat
counter.complete_stage=lambda *args:'patched'
assert _phase1_proxy_source(counter) is None
del counter.complete_stage
assert _phase1_proxy_source(counter) is heartbeat
old=_HeartbeatLLMClient.complete_stage
_HeartbeatLLMClient.complete_stage=lambda self,*args:'patched'
assert _phase1_proxy_source(heartbeat) is None
_HeartbeatLLMClient.complete_stage=old
assert _phase1_proxy_source(heartbeat) is dual
heartbeat.complete_stage=lambda *args:'patched'
assert _phase1_proxy_source(heartbeat) is None
del heartbeat.complete_stage
assert _phase1_proxy_source(heartbeat) is dual
print(json.dumps({'beats':len(beats),'calls':len(dual.calls)}))
''', derived_bundle)
    assert state == {"beats": 6, "calls": 3}


def test_heartbeat_lease_failure_prevents_dispatch_and_overrides_provider_error(derived_bundle):
    state = _isolated(r'''
import json,sys
from pathlib import Path
derived,frozen=map(Path,sys.argv[1:]);sys.path[:0]=[str(derived/'candidate'),str(frozen/'code')]
from hymem.dreaming.runner import _HeartbeatLLMClient
class LeaseLost(Exception):pass
class Dual:
    def __init__(self):self.calls=0
    def complete_stage(self,*args):self.calls+=1;raise ValueError('provider')
dual=Dual();beats=0
def heartbeat():
    global beats
    beats+=1
    if beats in (1,3):raise LeaseLost('lost')
client=_HeartbeatLLMClient(dual,heartbeat)
try:client.complete_stage(1,2,3,4)
except LeaseLost:pass
else:raise AssertionError('pre-call lease loss ignored')
assert dual.calls==0
try:client.complete_stage(1,2,3,4)
except LeaseLost:pass
else:raise AssertionError('post-call lease loss ignored')
assert dual.calls==1 and beats==3
print(json.dumps({'calls':dual.calls,'beats':beats}))
''', derived_bundle)
    assert state == {"calls": 1, "beats": 3}


def test_nonempty_chunk_reaches_staged_client_through_accounting_memory_chain(derived_bundle):
    state = _isolated(r'''
import importlib.util,json,sys
from pathlib import Path
derived,frozen=map(Path,sys.argv[1:]);sys.path[:0]=[str(derived/'candidate'),str(frozen/'code')]
from hymem.extraction import chunk
from hymem.dreaming.runner import _CountingPhase1LLM,_HeartbeatLLMClient
from benchmarks.codex_subscription_warm_v9 import SharedBudget,BudgetLimits
spec=importlib.util.spec_from_file_location('frozen_diagnostic_runner',
    frozen/'code/tools/diagnostics/luna_lme_diagnostic_v8.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
limits=BudgetLimits(12,160000,600);budget=SharedBudget(limits,max_in_flight=1)
budget.register('q-0001',limits)
class Dual:
    key='q-0001'
    def __init__(self):self.budget=budget;self.ordinary=0;self.structured=0
    def _settle(self):
        budget.reserve(self.key)
        budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna',
            'config_isolation_admitted':True,'inference_enabled':False,
            'quota_windows':[{'remaining_percent':100}]})
        budget.settle(self.key,used=10,turn_started=True)
    def complete(self,request):
        self._settle();self.ordinary+=1
        return json.dumps({'triples':[{'subject':'user','predicate':'uses',
            'object':'sqlite','polarity':1}] if self.ordinary==1 else [],
            'markers':[],'complete':True})
    def complete_stage(self,request,batch,stage,recheck):
        self._settle();self.structured+=1
        raise RuntimeError('reached staged fake')
dual=Dual();accounted=runner.AccountedClient(dual,derived/'candidate')
memory=runner._memory_client({},accounted)
client=_CountingPhase1LLM(_HeartbeatLLMClient(memory,lambda:None))
result=chunk.extract_chunk(client,'The user uses sqlite.',completion_call_limit=4)
assert dual.ordinary==2 and dual.structured>=1
assert result.failure_reason=='call_failure'
assert accounted.reconcile() and budget.snapshot()['usage_complete']
assert client.completion_calls==dual.ordinary+dual.structured
assert client.provider_attempts==client.completion_calls
print(json.dumps({'ordinary':dual.ordinary,'staged':dual.structured,
    'accounting':accounted.counts,'failure':result.failure_reason}))
''', derived_bundle)
    assert state["ordinary"] == 2
    assert state["staged"] >= 1
    assert "grounding_original_initial" in state["accounting"]


def test_scoped_attempts_include_raised_staged_calls(derived_bundle):
    state = _isolated(r'''
import json,sys
from contextlib import contextmanager
from pathlib import Path
derived,frozen=map(Path,sys.argv[1:]);sys.path[:0]=[str(derived/'candidate'),str(frozen/'code')]
from hymem.dreaming.runner import _CountingPhase1LLM,_HeartbeatLLMClient
from hymem.extraction.llm import ProviderAttemptTracker
class Dual:
    def __init__(self):self.tracker=None;self.fail=False
    @contextmanager
    def track_provider_attempts(self):
        self.tracker=ProviderAttemptTracker()
        yield self.tracker
    def complete_stage(self,*args):
        for _ in range(3 if self.fail else 2):self.tracker._record()
        if self.fail:raise RuntimeError('provider')
        return 'ok'
dual=Dual();counter=_CountingPhase1LLM(_HeartbeatLLMClient(dual,lambda:None))
assert counter.complete_stage(1,2,3,4)=='ok'
dual.fail=True
try:counter.complete_stage(1,2,3,4)
except RuntimeError:pass
else:raise AssertionError('raised completion swallowed')
assert counter.completion_calls==2 and counter.provider_attempts==5
print(json.dumps({'logical':counter.completion_calls,'attempts':counter.provider_attempts}))
''', derived_bundle)
    assert state == {"logical": 2, "attempts": 5}
