"""Root checks through a fresh physical candidate, with synthetic attestations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import luna_staged_candidate_v1 as builder

REPO = Path(__file__).resolve().parents[3]
BASE = Path('/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi')
ACCEPTED = Path('/private/tmp/hymem-semantic-step2-v2-candidate-20260929')
ACCEPTED_STAMP = Path('/private/tmp/hymem-semantic-step2-v2-map-20260929.json')


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp('staged-root')
    builder.prepare(BASE / 'candidate', BASE / 'headless-grounding-source-map.json',
        ACCEPTED, ACCEPTED_STAMP, root / 'candidate', root / 'map.json', REPO)
    return root / 'candidate'


SCRIPT = r'''
import sys,json,socket
from dataclasses import asdict
sys.path.insert(0,sys.argv[1])
from hymem.extraction import chunk,grounding_staged_v1 as s,grounding_classification_v4 as g
def deny(*a,**k):raise AssertionError('offline only')
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
def assessment(triple,positive):
    if not positive:return dict(state='not_established',support=None)
    names=['attribution_and_roles','relation_and_polarity']+[
        k for k in ('value_text','value_numeric','value_unit','temporal_scope') if getattr(triple,k) is not None]
    return dict(state='supported',support=dict(
        evidence=[dict(source_message_id=71,region='owned',quote=text)],
        checks={k:dict(state='supported',evidence_indices=[0]) for k in names}))
class Client:
    request_attempts=0
    def __init__(self):self.ground=[];self.ordinary=0
    def complete(self,request):
        self.request_attempts+=1;self.ordinary+=1
        assert self.ordinary<=2,'no extraction reroll'
        return json.dumps(dict(complete=True,triples=claims if self.ordinary==1 else [],
            markers=[marker] if self.ordinary==1 else []))
    def complete_stage(self,request,bound,stage,recheck):
        assert type(recheck)is bool
        if stage=='original':s.validate_original_request(request,bound);batch=bound
        else:
            assert stage=='alternatives' and not recheck
            s.validate_alternatives_request(request,bound);batch=bound.classification_batch
        self.request_attempts+=3
        self.ground.append(dict(triples=[asdict(t) for t in batch.triples],stage=stage,recheck=recheck))
        if case=='provider' and len(self.ground)==2:raise RuntimeError('private-test-failure')
        if case=='malformed' and len(self.ground)==1:return 'private-invalid-response'
        if stage=='original':
            rows=[]
            for i,t in enumerate(batch.triples):
                positive=not(not recheck and t.subject=='person_0' and t.predicate=='uses')
                if case=='late_recheck_reject' and recheck and len(batch.triples)==1:positive=False
                if case=='second_correction' and recheck and i==0:positive=False
                rows.append(dict(index=i,original=assessment(t,positive)))
            return json.dumps(dict(schema=s.ORIGINAL_SCHEMA,batch_sha256=batch.batch_sha256,
                complete=True,originals=rows))
        rows=[dict(index=i,alternatives={p:assessment(batch.triples[i],p=='prefers')
            for p in g.PREDICATE_ORDER if p!=batch.triples[i].predicate}) for i in bound.negative_indices]
        return json.dumps(dict(schema=s.ALTERNATIVES_SCHEMA,batch_sha256=batch.batch_sha256,
            original_response_sha256='0'*64 if case=='prior_binding' else bound.original_response_sha256,
            complete=True,alternatives=rows))
client=Client()
limit={'budget_alternatives':3,'budget_second':4,'budget_recheck':5}.get(case)
result=chunk.extract_chunk(client,record,source_records=((71,record),),completion_call_limit=limit)
out=dict(failed=result.failed,calls=result.completion_calls,attempts=result.provider_attempts,
    ground=result.grounding_calls,rechecks=result.grounding_recheck_calls,
    grounding_attempts=result.grounding_provider_attempts,batches=client.ground,
    triples=[asdict(t) for t in result.triples],markers=[asdict(m) for m in result.markers],
    hints=result.entity_type_hints,details=result.failure_details)
assert 'private-test-failure' not in json.dumps(out) and 'private-invalid-response' not in json.dumps(out)
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


def test_full_extraction_path_rechecks_entire_list_and_accounts_each_stage(candidate):
    result = run(candidate, 'positive')
    assert not result['failed']
    assert [(len(b['triples']),b['stage'],b['recheck']) for b in result['batches']] == [
        (8,'original',False),(8,'alternatives',False),(1,'original',False),
        (8,'original',True),(1,'original',True)]
    assert result['batches'][2]['triples'] == result['batches'][4]['triples']
    assert (result['calls'],result['attempts'],result['ground'],result['rechecks'],
        result['grounding_attempts']) == (7,17,5,2,15)


@pytest.mark.parametrize('case,calls,ground,rechecks',[
    ('collision',5,3,0),('conflict',5,3,0),('provider',4,2,0),
    ('malformed',3,1,0),('prior_binding',4,2,0),('late_recheck_reject',7,5,2),
    ('second_correction',6,4,1),('budget_alternatives',3,1,0),
    ('budget_second',4,2,0),('budget_recheck',5,3,0),
])
def test_actual_chunk_is_atomic_and_does_not_reset_or_raise_limits(candidate,case,calls,ground,rechecks):
    result=run(candidate,case)
    assert result['failed'] and not result['triples'] and not result['markers'] and not result['hints']
    assert (result['calls'],result['ground'],result['rechecks']) == (calls,ground,rechecks)
    assert result['attempts'] == 2+3*ground and result['grounding_attempts'] == 3*ground


def inventory(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob('*') if p.is_file() and p.suffix == '.py'}


def test_physical_delta_only_two_files_and_exact_six_helpers(candidate):
    before=inventory(BASE/'candidate');after=inventory(candidate)
    assert {p for p in before if before[p]!=after[p]} == {'hymem/extraction/chunk.py','hymem/extraction/contract.py'}
    added=set(after)-set(before)
    assert added=={'hymem/extraction/'+name+'.py' for name in ('grounding','grounding_gate',
        'grounding_v2','grounding_classification_v4','grounding_staged_v1','grounding_staged_gate_v1')}
    for path in added:assert (candidate/path).read_bytes()==(REPO/path).read_bytes()
    for version in (1,2,3):
        assert not (candidate/f'hymem/extraction/grounding_classification_v{version}.py').exists()


CACHE=r'''
import sys
sys.path.insert(0,sys.argv[1])
from hymem.extraction import contract as c,grounding_staged_v1 as s,grounding_classification_v4 as g
from hymem.extraction import grounding_staged_gate_v1 as gate,grounding_v2 as v2,grounding_gate as oldgate
before=c.extraction_cache_key()
case=sys.argv[2]
if case=='original_prompt':s._ORIGINAL_SYSTEM+='\nIndependent root mutation.'
elif case=='alternative_prompt':s._ALTERNATIVES_SYSTEM+='\nIndependent root mutation.'
elif case=='original_schema':s.build_original_output_schema.__code__=(lambda a:None).__code__
elif case=='alternative_schema':s.build_alternatives_output_schema.__code__=(lambda a:None).__code__
elif case=='staged_selector':s.parse_staged_responses.__code__=(lambda *a,**k:None).__code__
elif case=='prior_validator':s._original.__code__=(lambda *a,**k:None).__code__
elif case=='classifier':g.parse_grounding_response.__code__=(lambda *a,**k:None).__code__
elif case=='v2_validator':v2._validate.__code__=(lambda *a,**k:None).__code__
elif case=='source_mapper':oldgate._source.__code__=(lambda *a,**k:None).__code__
elif case=='stage_rebinding':s.parse_original_response=lambda *a,**k:None
try:after=c.extraction_cache_key()
except (RuntimeError,ValueError):print('closed')
else:
    assert after!=before,'modified consumed helper reused cache identity'
    print('changed')
'''


@pytest.mark.parametrize('case',['original_prompt','alternative_prompt','original_schema',
    'alternative_schema','staged_selector','prior_validator','classifier','v2_validator',
    'source_mapper','stage_rebinding'])
def test_cache_identity_covers_every_new_consumed_layer(candidate,case):
    result=subprocess.run([sys.executable,'-I','-B','-c',CACHE,str(candidate),case],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip() in {'closed','changed'}
