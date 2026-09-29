"""Root checks of the physical classification candidate; synthetic judgments only."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import luna_classification_candidate as builder

REPO = Path(__file__).resolve().parents[3]
BASE = Path('/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi')
ACCEPTED = Path('/private/tmp/hymem-semantic-step2-v2-candidate-20260929')
ACCEPTED_STAMP = Path('/private/tmp/hymem-semantic-step2-v2-map-20260929.json')


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp('classification-root')
    builder.prepare(BASE / 'candidate', BASE / 'headless-grounding-source-map.json',
                    ACCEPTED, ACCEPTED_STAMP, root / 'candidate', root / 'map.json', REPO)
    return root / 'candidate'


SCRIPT = r'''
import sys, json, socket
from dataclasses import asdict
sys.path.insert(0,sys.argv[1])
from hymem.extraction import chunk, grounding_classification_v1 as g
def deny(*a,**k): raise AssertionError('offline only')
socket.socket.connect=deny
socket.create_connection=deny
case=sys.argv[2]
text='Invented source for mechanical tests, not semantic accuracy.'
record=json.dumps(dict(source_message_id=71,source_role='user',source_peer_id='Mira',
    source_record_version='hymem-claim-source-v2',content=text))
claims=[dict(subject='person_'+str(i),predicate='prefers',object='database_'+str(i),
    polarity=1,source_message_id=71,value_text='five',value_numeric=5,
    value_unit='units',temporal_scope='2026',subject_type='person') for i in range(9)]
claims[0]['predicate']='uses'
if case in {'collision','conflict'}:
    claims[-1].update(subject=claims[0]['subject'],object=claims[0]['object'],
        polarity=-1 if case=='conflict' else 1)
marker=dict(kind='preference',statement='Independent invented marker.')
class Client:
    request_attempts=0
    def __init__(self): self.ground=[]; self.ordinary=0
    def complete(self,request):
        self.request_attempts+=1; self.ordinary+=1
        assert self.ordinary<=2,'no blind extraction retry'
        return json.dumps(dict(complete=True,triples=claims if self.ordinary==1 else [],
                               markers=[marker] if self.ordinary==1 else []))
    def complete_grounding(self,request,batch):
        g.validate_request(request,batch)
        self.request_attempts+=3
        self.ground.append([asdict(t) for t in batch.triples])
        assert all('predicate' not in t for t in json.loads(request.user)['batch']['candidates'])
        if case=='provider' and len(self.ground)==2: raise RuntimeError('private-test-failure')
        items=[]
        for i,t in enumerate(batch.triples):
            pred=t.predicate
            if len(self.ground)<=2 and t.subject=='person_0' and pred=='uses': pred='prefers'
            if case=='late_recheck_reject' and len(self.ground)==4: pred=None
            if case=='second_correction' and len(self.ground)==3 and i==0: pred='avoids'
            states=['n']*22; citations=[[] for _ in states]; pool=[]
            if pred is not None:
                pos=g.PREDICATE_ORDER.index(pred); states[pos]='e'; citations[pos]=[0]
                pool=[dict(source_message_id=71,region='owned',quote=text)]
            items.append(dict(index=i,states=states,evidence_pool=pool,citations=citations))
        return json.dumps(dict(schema=g.GROUNDING_CONTRACT_VERSION,
            batch_sha256=batch.batch_sha256,complete=True,classifications=items))
client=Client()
limit={'budget_second':3,'budget_recheck':4}.get(case)
result=chunk.extract_chunk(client,record,source_records=((71,record),),completion_call_limit=limit)
out=dict(failed=result.failed,calls=result.completion_calls,attempts=result.provider_attempts,
    ground=result.grounding_calls,rechecks=result.grounding_recheck_calls,
    grounding_attempts=result.grounding_provider_attempts,batches=client.ground,
    triples=[asdict(t) for t in result.triples],markers=[asdict(m) for m in result.markers],
    hints=result.entity_type_hints,details=result.failure_details)
assert 'private-test-failure' not in json.dumps(out)
if not result.failed:
    expected=[{k:v for k,v in t.items() if k!='subject_type'} for t in claims]
    expected[0]['predicate']='prefers'
    assert out['triples']==expected and out['markers']==[marker] and len(out['hints'])==9
print(json.dumps(out))
'''


def run(candidate, case):
    result = subprocess.run([sys.executable, '-I', '-B', '-c', SCRIPT, str(candidate), case],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_full_list_recheck_and_attempt_accounting(candidate):
    result = run(candidate, 'positive')
    assert not result['failed']
    assert [len(b) for b in result['batches']] == [8, 1, 8, 1]
    assert result['batches'][1] == result['batches'][3]
    assert result['calls'] == 6 and result['attempts'] == 14
    assert result['ground'] == 4 and result['rechecks'] == 2
    assert result['grounding_attempts'] == 12


@pytest.mark.parametrize('case,calls,ground,rechecks', [
    ('collision',4,2,0), ('conflict',4,2,0), ('provider',4,2,0),
    ('late_recheck_reject',6,4,2), ('second_correction',5,3,1),
    ('budget_second',3,1,0), ('budget_recheck',4,2,0),
])
def test_atomic_failure_and_no_extra_attempts(candidate, case, calls, ground, rechecks):
    result = run(candidate, case)
    assert result['failed']
    assert not result['triples'] and not result['markers'] and not result['hints']
    assert (result['calls'],result['ground'],result['rechecks']) == (calls,ground,rechecks)
    assert result['attempts'] == 2+3*ground and result['grounding_attempts'] == 3*ground


def test_candidate_delta_is_limited_and_helpers_are_exact(candidate):
    old = builder.old.inventory(BASE / 'candidate')
    new = builder.old.inventory(candidate)
    assert {k for k in old if old[k]!=new[k]} == {
        'hymem/extraction/chunk.py', 'hymem/extraction/contract.py'}
    added = set(new)-set(old)
    assert added == {'hymem/extraction/grounding.py','hymem/extraction/grounding_gate.py',
        'hymem/extraction/grounding_v2.py','hymem/extraction/grounding_classification_v1.py',
        'hymem/extraction/grounding_classification_gate_v1.py'}
    for name in added:
        assert (candidate/name).read_bytes() == (REPO/name).read_bytes()


CACHE = r'''
import sys
sys.path.insert(0,sys.argv[1])
from hymem.extraction import contract as c, grounding_classification_v1 as g
from hymem.extraction import grounding_classification_gate_v1 as gate
from hymem.extraction import grounding_v2 as v2, grounding_gate as oldgate
before=c.extraction_cache_key()
case=sys.argv[2]
if case=='prompt': g._SYSTEM+='\nIndependent test.'
elif case=='selector': g.parse_grounding_response.__code__=(lambda a,b:None).__code__
elif case=='v2_validator': v2._validate.__code__=(lambda a,b:None).__code__
elif case=='source_mapper': oldgate._source.__code__=(lambda a,b:None).__code__
elif case=='gate_import': gate.build_grounding_request=lambda a,b:None
elif case=='definition_rebinding': g.parse_grounding_response=lambda a,b:None
try: after=c.extraction_cache_key()
except (RuntimeError,ValueError): print('closed')
else:
    assert after!=before,'changed helper reused cache identity'
    print('changed')
'''


@pytest.mark.parametrize('case',['prompt','selector','v2_validator','source_mapper',
                                'gate_import','definition_rebinding'])
def test_identity_tracks_all_consumed_helpers(candidate, case):
    result=subprocess.run([sys.executable,'-I','-B','-c',CACHE,str(candidate),case],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip() in {'closed','changed'}
