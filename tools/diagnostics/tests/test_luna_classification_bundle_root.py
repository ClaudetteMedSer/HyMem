"""Independent generated-bundle checks. All model outputs are invented."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from tools.diagnostics import luna_classification_bundle as derive

REPO = Path(__file__).resolve().parents[3]
SOURCE_CANDIDATE = Path('/private/tmp/hymem-classification-root-ZP2zuU/candidate')
BUNDLE = Path(tempfile.mkdtemp(prefix='hymem-classification-root-tests-')).resolve() / 'bundle'
PROOF = derive.prepare(REPO, BUNDLE)
CANDIDATE = BUNDLE / 'candidate'
INVENTORY = BUNDLE / 'map.json'
shutil.copytree(SOURCE_CANDIDATE, CANDIDATE)
shutil.copyfile(SOURCE_CANDIDATE.parent / 'map.json', INVENTORY)
CODE = BUNDLE / 'code'


def load(relative, name):
    path = BUNDLE / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOST = load('code/tools/diagnostics/luna_semantic_probe_host.py', 'root_classification_host')
ENTRY = load('code/tools/diagnostics/luna_semantic_probe_run.py', 'root_classification_entry')
READER = load('code/tools/diagnostics/luna_semantic_probe_progress.py', 'root_classification_reader')
ADAPTER = load('adapter-v2.py', 'root_classification_adapter')


def test_source_pins_limits_and_candidate_inventory():
    pins = HOST.source_pins(CODE)
    assert ADAPTER.HOST_SHA == pins['tools/diagnostics/luna_semantic_probe_host.py']
    assert ADAPTER.RUN_SHA == pins['tools/diagnostics/luna_semantic_probe_run.py']
    assert ADAPTER.READER_SHA == pins['tools/diagnostics/luna_semantic_probe_progress.py']
    assert pins['tools/diagnostics/luna_semantic_probe.py'] in (
        CODE / 'tools/diagnostics/luna_semantic_probe_run.py').read_text()
    assert HOST.INVENTORY_SHA == hashlib.sha256(INVENTORY.read_bytes()).hexdigest()
    assert HOST.LIMITS == {
        'turns':29,'known_tokens':500000,'seconds':1800,
        'control_turns':2,'control_known_tokens':100000,'control_seconds':240,
        'hybrid_turns':3,'hybrid_known_tokens':160000,'hybrid_seconds':600,
        'invocation_seconds':120,'workers':1,'units':25,
        'ordinary_replays':8,'new_judgments_max':29}
    assert HOST.POLICY['runtime_max_seconds'] == 1930
    assert HOST.POLICY['timeout_stop_seconds'] == 10
    assert HOST.POLICY['tasks_max'] == 256
    assert (CODE / 'tools/diagnostics/luna_semantic_cases.py').read_bytes() == (
        REPO / 'tools/diagnostics/luna_semantic_cases.py').read_bytes()


def execute(script, *args):
    result = subprocess.run([sys.executable,'-I','-B','-c',script,
        str(CANDIDATE),str(CODE),*args],capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


PREAMBLE = r'''
import sys,json,socket
from pathlib import Path
from dataclasses import asdict,replace
sys.path[:0]=[sys.argv[1],sys.argv[2]]
def deny(*a,**k): raise AssertionError('offline only')
socket.socket.connect=deny
socket.create_connection=deny
from hymem.extraction import grounding_classification_v1 as g,chunk
from benchmarks import extraction_canary as gold,luna_semantic_canary as check
from benchmarks import luna_semantic_stage_accounting as stage
from tools.diagnostics import luna_semantic_cases as cases,luna_semantic_probe as core

def classification(batch,choices,evidence):
    items=[]
    for i,(triple,pred,pool) in enumerate(zip(batch.triples,choices,evidence,strict=True)):
        states=['n']*22; citations=[[] for _ in states]
        if pred is not None:
            pos=g.PREDICATE_ORDER.index(pred); states[pos]='e'
            citations[pos]=list(range(len(pool)))
        else: pool=[]
        items.append(dict(index=i,states=states,evidence_pool=pool,citations=citations))
    return json.dumps(dict(schema=g.GROUNDING_CONTRACT_VERSION,batch_sha256=batch.batch_sha256,
                           complete=True,classifications=items))

class Judge:
    def __init__(self,case,behavior='expected'):
        self.case,self.behavior,self.requests=case,behavior,[]
    def complete_grounding(self,request,batch):
        g.validate_request(request,batch)
        self.requests.append((request,batch))
        choices=[]; evidence=[]
        for triple,label in zip(batch.triples,self.case.expected,strict=True):
            pred=(label.predicate if self.case.category=='correction' else
                  triple.predicate if self.case.category=='supported' else None)
            if self.behavior=='false_support': pred=triple.predicate
            if self.behavior=='reject' or self.behavior=='reject_recheck' and len(self.requests)>1:
                pred=None
            if self.behavior=='wrong_replacement' or self.behavior=='repeated_correction' and len(self.requests)>1:
                pred='avoids'
            pool=[dict(source_message_id=triple.source_message_id,region=region,quote=quote)
                  for region,quote in label.evidence]
            if self.behavior=='false_support':
                source=next(s for s in batch.sources if s.source_message_id==triple.source_message_id)
                pool=[dict(source_message_id=triple.source_message_id,region='owned',quote=source.content[:192])]
            choices.append(pred); evidence.append(pool)
        if self.behavior=='malformed': return 'PRIVATE_SYNTHETIC_DO_NOT_EXPORT'
        return classification(batch,choices,evidence)

class Budget:
    def halt(self,code): raise AssertionError(code)
class CanaryClient:
    observed_turns=observed_tokens=request_attempts=0
    usage_complete=True
    budget=Budget()
    def __init__(self,scenario='second'): self.scenario=scenario; self.seen={}
    def complete(self,request):
        self.observed_turns+=1; self.observed_tokens+=7; self.request_attempts+=1
        payloads,failures=gold._request_source_payloads(request)
        assert not failures
        triples=[]
        if 'VERIFICATION PASS' not in request.system:
            for i,e in enumerate(gold._CANARY_EXPECTED_CLAIMS):
                fragment=next((p for p in payloads if p['source_message_id']==e[6]),None)
                owned=gold._TABLE_CLAIM_ROW if i==0 else gold._PROSE_BOUNDARY_RIGHT
                if not fragment or owned not in fragment['content']: continue
                pred=e[2]
                if self.scenario in {'both','repeated_correction'} or self.scenario in {'second','false_support'} and i==1:
                    pred='uses'
                item=dict(subject=e[0],predicate=pred,object=e[3],polarity=e[5],source_message_id=e[6],
                          subject_type=e[1],object_type=e[4])
                if self.scenario=='extra_qualifier': item['temporal_scope']='invented'
                if self.scenario=='wrong_type': item['subject_type']='service'
                triples.append(item)
        markers=[dict(kind='preference',statement='invented')] if self.scenario=='extra_marker' else []
        return json.dumps(dict(complete=True,triples=triples,markers=markers))
    def complete_grounding(self,request,batch):
        g.validate_request(request,batch)
        self.observed_turns+=1; self.observed_tokens+=7; self.request_attempts+=1
        t=batch.triples[0]; mid=t.source_message_id
        i=next(i for i,e in enumerate(gold._CANARY_EXPECTED_CLAIMS) if e[6]==mid)
        self.seen[mid]=self.seen.get(mid,0)+1
        pred=gold._CANARY_EXPECTED_CLAIMS[i][2]
        if self.scenario=='false_support' and i==1: pred=t.predicate
        if self.scenario=='repeated_correction' and self.seen[mid]>1: pred='avoids'
        if self.scenario=='negative_second' and i==1: pred=None
        if self.scenario=='provider_failure' and i==1:
            self.usage_complete=False
            raise RuntimeError('PRIVATE_SYNTHETIC_DO_NOT_EXPORT')
        quote=gold._TABLE_CLAIM_ROW if i==0 else gold._PROSE_BOUNDARY_RIGHT
        return classification(batch,[pred],[[dict(source_message_id=mid,region='owned',quote=quote)]])
'''


def test_real_core_preflight_and_all_fixed_controls_without_label_leakage():
    script = PREAMBLE + r'''
proof=core.verify_local(Path(sys.argv[1]),Path(sys.argv[1]).parent/'map.json',
    cases_module=cases,grounding_module=g,canary_module=check,stage_module=stage)
assert proof['candidate_files']==513
total=0
for case in cases.cases():
    client=Judge(case); events=[]
    out=core.run_control(case,client,g,record=events.append)
    assert out['passed'],(case.case_id,out)
    assert len(client.requests)==(2 if case.category=='correction' else 1)
    for request,batch in client.requests:
        assert 'predicate' not in json.loads(request.user)['batch']['candidates'][0]
        for label in case.expected: assert label.rationale not in request.user+request.system
    assert len(events)==2*len(client.requests)
    total+=len(client.requests)
assert total==26
print(json.dumps(dict(controls=24,synthetic_calls=total)))
'''
    assert execute(script) == {'controls':24,'synthetic_calls':26}


@pytest.mark.parametrize('scenario,rechecks',[('healthy',0),('second',1),('both',2)])
def test_actual_canary_and_stage_stack(scenario,rechecks):
    script = PREAMBLE + r'''
client=CanaryClient(sys.argv[3]); ledger=stage.StageLedger(Path(sys.argv[1]))
report=check.run_canary(gold,chunk,ledger.wrap(client,'canary'),candidate=Path(sys.argv[1]))
assert report['passed'],report
assert ledger.reconcile_canary(report)
print(json.dumps(dict(calls=report['completion_calls'],rechecks=report['grounding_recheck_calls'],
                     unclassified='unclassified' in ledger.snapshot()['canary'])))
'''
    assert execute(script,scenario)=={'calls':10+rechecks,'rechecks':rechecks,'unclassified':False}


@pytest.mark.parametrize('scenario',['false_support','repeated_correction','negative_second',
    'provider_failure','extra_qualifier','extra_marker','wrong_type'])
def test_actual_canary_negative_controls(scenario):
    script = PREAMBLE + r'''
client=CanaryClient(sys.argv[3]); ledger=stage.StageLedger(Path(sys.argv[1]))
report=check.run_canary(gold,chunk,ledger.wrap(client,'canary'),candidate=Path(sys.argv[1]))
assert not report['passed']
assert 'PRIVATE_SYNTHETIC' not in json.dumps(report)
print(json.dumps(dict(rejected=True)))
'''
    assert execute(script,scenario)=={'rejected':True}


@pytest.mark.parametrize('behavior',['false_support','reject','wrong_replacement','reject_recheck',
                                   'repeated_correction','malformed'])
def test_control_failure_measurement_does_not_reroll(behavior):
    script = PREAMBLE + r'''
behavior=sys.argv[3]
category=('reject' if behavior=='false_support' else 'supported' if behavior in {'reject','malformed'} else 'correction')
case=next(c for c in cases.cases() if c.category==category)
client=Judge(case,behavior); out=core.run_control(case,client,g,record=lambda x:None)
assert not out['passed']
assert len(client.requests)==(2 if behavior in {'reject_recheck','repeated_correction'} else 1)
assert 'PRIVATE_SYNTHETIC' not in json.dumps(out)
print(json.dumps(dict(rejected=True)))
'''
    assert execute(script,behavior)=={'rejected':True}


@pytest.mark.parametrize('fault',['none','result_write'])
def test_whole_entry_private_replay_and_accounting(fault):
    from tools.diagnostics.tests.test_luna_semantic_entry_root import PROGRAM
    tail = PROGRAM[PROGRAM.index('from dataclasses import asdict'):]
    tail = tail.replace('from tools.diagnostics.tests.test_luna_semantic_probe_root import Judge\n','')
    tail = tail.replace('from hymem.extraction import grounding\n',
                        'from hymem.extraction import grounding_classification_v1 as grounding\n')
    tail = tail.replace('from benchmarks import codex_subscription_warm_v3 as warm\n',
        'from benchmarks import codex_subscription_classification_v1 as transport\nwarm=transport.warm\n')
    tail = tail.replace('    def complete(self,request):\n',
                        '    def complete_grounding(self,request,batch):\n')
    tail = tail.replace('return self.judge.complete(request)',
                        'return self.judge.complete_grounding(request,batch)')
    tail = tail.replace("{'candidate_files':510}","{'candidate_files':513}")
    tail = tail.replace('    from tools.diagnostics import luna_semantic_verdict_replay_root as verifier',
        '    import importlib.util\n'
        f'    spec=importlib.util.spec_from_file_location("root_classification_replay",{str(BUNDLE / "verdict-replay.py")!r})\n'
        '    verifier=importlib.util.module_from_spec(spec)\n'
        '    spec.loader.exec_module(verifier)')
    anchor = "    assert reviewed['new_model_calls']==0 and reviewed['all_semantic_checks_passed']\n"
    assert tail.count(anchor)==1
    tail = tail.replace(anchor,anchor+r'''
    # Swapping only the trusted original must fail even when model-facing
    # text, captured response and all public aggregate counters are unchanged.
    journal=sorted((root/'run/private-journal').glob('*control-00.json'))[0]
    raw_journal=journal.read_bytes()
    damaged=json.loads(raw_journal)
    canonical_batch=json.loads(damaged['batch'])
    canonical_batch['candidates'][0]['predicate']='avoids'
    damaged['batch']=json.dumps(canonical_batch,sort_keys=True,separators=(',',':'))
    journal.write_text(json.dumps(damaged))
    try: verifier.replay(root,'1'*64,shim_sha)
    except AssertionError: pass
    else: raise AssertionError('private replay accepted hidden-original substitution')
    journal.write_bytes(raw_journal)
''')
    program = PREAMBLE + '\nClient=CanaryClient\n' + tail
    result = execute(program,'second',fault)
    assert result == {'offline':True,'units':25,'new_synthetic_turns':29,'known_synthetic_tokens':261}


def test_actual_default_factory_reaches_owned_transport_and_stops_without_inference():
    script = PREAMBLE + r'''
import tempfile,subprocess
from tools.diagnostics import luna_semantic_probe_run as entry,luna_semantic_probe_host as host
from benchmarks import codex_subscription_classification_v1 as transport
warm=transport.warm
subprocess.Popen=deny
retained=[]
first=CanaryClient()
ordinary=first.complete
def capture(request):
    raw=ordinary(request); retained.append((asdict(request),raw)); return raw
first.complete=capture
assert check.run_canary(gold,chunk,first,candidate=Path(sys.argv[1]))['passed']
assert len(retained)==8
starts=[]
class ForcedOfflineStop(BaseException): pass
class NoInferenceSession:
    def __init__(self,binary,cwd,timeout=120):
        starts.append(dict(binary=binary,timeout=timeout))
        raise ForcedOfflineStop()
warm.WarmSession=NoInferenceSession
root=Path(tempfile.mkdtemp(prefix='classification-factory-root-'))
(root/'candidate').symlink_to(Path(sys.argv[1]),target_is_directory=True)
loaded=(host,core,cases,g,gold,chunk,check,stage,warm,warm.concurrent)
entry.preflight=lambda *args:({'source_sha256':{}},loaded,{},tuple(retained))
out=entry.execute(root,'1'*64,containment=lambda *args:None)
assert len(starts)==1, (starts,out)
assert out['stop_code'] is not None and not out['core_completed']
assert out['paid_budget']['turns']==0
assert out['client_cleanup_ok']
print(json.dumps(dict(factory_exercised=True,new_model_calls=0)))
'''
    assert execute(script)=={'factory_exercised':True,'new_model_calls':0}


# Preserve the independent previous receipt, process and control-plane fault
# assertions, applied to these generated sources rather than the old bundle.
for _file in ('test_luna_semantic_host_root.py','test_luna_semantic_reader_root.py',
              'test_luna_semantic_adapter_root.py'):
    _path=Path(__file__).with_name(_file)
    _source=_path.read_text()
    for _line in (
        'from tools.diagnostics import luna_semantic_probe_host as host',
        'from tools.diagnostics import luna_semantic_probe_run as runner',
        'from tools.diagnostics import luna_semantic_probe_progress as reader',
        'from tools.diagnostics import luna_semantic_probe_adapter_v2 as adapter',
    ):
        _source=_source.replace(_line,'')
    _source=_source.replace('luna-semantic-probe','luna-classification-probe')
    _source=_source.replace('/private/tmp/hymem-semantic-step2-v2-candidate-20260929',str(CANDIDATE))
    _source=_source.replace('/private/tmp/hymem-semantic-step2-v2-map-20260929.json',str(INVENTORY))
    _source=_source.replace('repo=Path(__file__).resolve().parents[3]','repo=_BUNDLE_CODE')
    _namespace={'__name__':__name__+'.'+_path.stem,'__file__':str(_path),
                '_BUNDLE_CODE':CODE,'host':HOST,'runner':ENTRY,'reader':READER,'adapter':ADAPTER}
    exec(compile(_source,str(_path),'exec'),_namespace)
    for _name,_value in _namespace.items():
        if _name.startswith('test_') and callable(_value):
            globals()['test_rebound_'+_path.stem+'_'+_name[5:]]=_value
        elif _name=='sealed':
            globals()[_name]=_value
