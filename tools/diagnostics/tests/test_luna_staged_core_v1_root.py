"""Independent finite campaign faults; every model response is invented."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

from hymem.extraction import grounding_classification_v4 as g
from hymem.extraction import grounding_staged_v1 as s
from tools.diagnostics import luna_semantic_cases as cases
from tools.diagnostics import luna_staged_core_v1 as c

H = runpy.run_path(str(Path(__file__).resolve().parents[3] / 'tests/test_grounding_staged_v1_root.py'))


def canaries():
    return tuple(s.build_original_request(
        (g.Triple(subject,predicate,obj,1,source_message_id=sid),),
        (g.GroundingSource(sid,'Synthetic local input, not retained evidence.'),))[1]
        for sid,subject,predicate,obj in (
            (71001,'HyMem Canary Relay','deploys_to','Fly.io'),
            (71002,'Avery Boundary Canary','uses','PostgreSQL')))


class Journal:
    def __init__(self, fail=None): self.events=[];self.fail=fail
    def record(self,key,value):
        if value['phase']==self.fail: raise c.DiagnosticStop('private_evidence_write_failure')
        self.events.append((key,copy.deepcopy(value)))


def campaign(*,fault=None,ordinal=0,stage_fault=1,journal=None,mode='negative'):
    seen=[];created=[];closed=[]
    journal=journal or Journal()
    warm=c.transport.warm
    class Synthetic:
        def __init__(self,key,cap,budget):
            self.key,self.budget=key,budget
            self.calls=0
            assert cap==(3,100000,240)
            budget.register(key,warm.BudgetLimits(*cap));created.append(key)
        @property
        def observed_turns(self):return self.budget.snapshot()['questions'][self.key]['turns']
        @property
        def observed_tokens(self):return self.budget.snapshot()['questions'][self.key]['known_tokens']
        @property
        def usage_complete(self):return self.budget.snapshot()['questions'][self.key]['usage_complete']
        def complete_stage(self,request,bound,stage,recheck):
            if stage=='original':s.validate_original_request(request,bound);batch=bound
            else:s.validate_alternatives_request(request,bound);batch=bound.classification_batch
            self.budget.reserve(self.key)
            self.budget.before_turn(self.key,dict(auth='chatgpt',model=warm.base.MODEL,
                config_isolation_admitted=True,inference_enabled=False,
                quota_windows=[dict(remaining_percent=90)]))
            self.calls+=1
            seen.append((self.key,request,bound,stage,recheck))
            active=fault if int(self.key.rsplit('-',1)[-1])==ordinal and self.calls==stage_fault else None
            self.budget.settle(self.key,used=None if active=='unknown_usage' else 500001 if active=='overshoot' else 7,turn_started=True)
            if active=='transport':raise RuntimeError('PRIVATE_NO_EXPORT')
            if active=='malformed':return 'PRIVATE_NO_EXPORT'
            unit=c.schedule()[int(self.key.rsplit('-',1)[-1])]
            if active in ('recheck_negative','recheck_ambiguous'):
                assert stage=='original' and recheck
                raw=H['original'](batch,['not_established' if active=='recheck_negative' else 'ambiguous']*len(batch.triples))
            elif mode in ('gold','wrong_prose','half_roles','half_alternative','role_corrections'):
                case=cases.cases()[unit.index] if unit.kind=='control' else None
                if case is not None:
                    predicates=[e.predicate if case.category=='correction' else t.predicate if case.category=='supported'
                        else None for t,e in zip(batch.triples,case.expected,strict=True)]
                    pools=[[H['quote'](t.source_message_id,text,region) for region,text in e.evidence]
                        for t,e in zip(batch.triples,case.expected,strict=True)]
                    if mode in ('half_roles','half_alternative','role_corrections') and unit.index==21:
                        for i in range(2 if mode=='role_corrections' else 1):
                            predicates[i]=batch.triples[i].predicate if mode=='half_roles' else 'prefers'
                            source=next(x for x in batch.sources if x.source_message_id==batch.triples[i].source_message_id)
                            pools[i]=[H['quote'](source.source_message_id,source.content[:192])]
                else:
                    predicates=['deploys_to' if unit.kind=='table_canary' else 'uses' if mode=='wrong_prose' else 'prefers']
                    pools=[[H['quote'](t.source_message_id,batch.sources[0].content)] for t in batch.triples]
                if stage=='original':raw=H['original'](batch,[
                    'supported' if p==t.predicate else 'not_established'
                    for p,t in zip(predicates,batch.triples,strict=True)],pools)
                else:
                    _,rebound,raw=H['alternatives'](batch,json.loads(bound.original_response_canonical_json),
                        positive=[(i,p) for i,p in enumerate(predicates) if p is not None and p!=batch.triples[i].predicate],pools=pools)
                    assert rebound==bound
            elif stage=='original':raw=H['original'](batch)
            else:
                _,rebound,raw=H['alternatives'](batch,json.loads(bound.original_response_canonical_json))
                assert rebound==bound
            return json.dumps(raw)
        def close(self):
            closed.append(self.key)
            if fault=='cleanup' and int(self.key.rsplit('-',1)[-1])==ordinal:
                raise RuntimeError('PRIVATE_CLEANUP_NO_EXPORT')
    result=c.run_campaign(concurrent=warm.concurrent,warm=warm,binary='unused',
        cases_module=cases,canary_batches=canaries(),journal=journal,client_factory=Synthetic)
    assert 'PRIVATE_' not in json.dumps(result)
    assert created==closed
    return result,seen,closed,journal


def test_fixed_eight_units_and_no_semantic_success_from_clean_diagnostic():
    result,seen,closed,journal=campaign()
    assert [(u.kind,u.index) for u in c.schedule()]==[
        ('control',9),('control',12),('control',13),('control',17),('control',19),
        ('control',21),('table_canary',0),('prose_canary',1)]
    assert len(seen)==16 and len(closed)==8
    assert result['diagnostic_completed'] and result['completed_units']==8
    assert result['paid_budget']==dict(turns=16,known_tokens=112,usage_complete=True,in_flight=0,reserved=0)
    assert all(result[k] is False for k in ['semantic_accuracy_accepted','full_lme_ready',
        'completed_and_clean','process_cleanup_verified'])
    assert all(x['admitted_turns']==2 for x in result['units'])
    assert [t[3] for t in seen]==['original','alternatives']*8
    assert all(not t[4] for t in seen)
    assert seen[-2][2].triples[0].predicate=='uses'
    assert seen[-1][2].classification_batch.triples[0].predicate=='uses'
    for ordinal in range(8):
        key=f'unit-{ordinal:02d}'
        stages=[e for k,e in journal.events if k==key and e['phase']=='before_dispatch']
        assert len(stages)==2
        first,second=seen[2*ordinal:2*ordinal+2]
        assert second[2].classification_batch==first[2]
        assert second[2].original_response_sha256==s.parse_original_response(
            second[2].original_response_canonical_json,first[2]).response_sha256


def test_all_gold_controls_use_seventeen_stages_and_prose_is_corrected_not_replaced_input():
    result,seen,closed,_=campaign(mode='gold')
    assert result['diagnostic_completed'] and len(seen)==17
    assert [x['admitted_turns'] for x in result['units']]==[1,3,3,2,2,2,1,3]
    assert all(x['expected_gold_match'] for x in result['units'])
    assert result['units'][-1]['final_predicates']==['prefers']
    assert seen[-3][2].triples[0].predicate=='uses'
    assert seen[-1][2].triples[0].predicate=='prefers'
    assert seen[-3][2].sources==seen[-1][2].sources
    for key,request,bound,stage,recheck in seen:
        unit=c.schedule()[int(key.rsplit('-',1)[-1])]
        if unit.kind=='control':
            assert all(e.rationale not in request.system+request.user for e in cases.cases()[unit.index].expected)


def test_wrong_original_prose_support_is_a_gold_failure():
    result,*_=campaign(mode='wrong_prose')
    assert result['units'][-1]['outcome']=='accepted'
    assert result['units'][-1]['final_predicates']==['uses']
    assert result['units'][-1]['expected_gold_match'] is False


def test_rejected_role_pair_does_not_hide_one_false_positive():
    result,*_=campaign(mode='half_roles')
    roles=result['units'][5]
    assert roles['outcome']=='rejected'
    assert roles['expected_gold_match'] is False


@pytest.mark.parametrize('fault',['recheck_negative','recheck_ambiguous'])
def test_failed_recheck_is_reported_not_lost_to_public_validator(fault):
    result,seen,closed,_=campaign(mode='gold',fault=fault,ordinal=1,stage_fault=3)
    assert result['diagnostic_completed'] and len(seen)==17 and len(closed)==8
    assert result['units'][1]['outcome']=='rejected'
    assert result['units'][1]['expected_gold_match'] is False


@pytest.mark.parametrize('mode',['half_alternative','role_corrections'])
def test_role_pair_false_corrections_remain_measurable_and_fail_gold(mode):
    result,seen,closed,_=campaign(mode=mode)
    assert result['diagnostic_completed'] and len(closed)==8
    assert result['units'][5]['expected_gold_match'] is False
    assert result['units'][5]['outcome']==('rejected' if mode=='half_alternative' else 'accepted')


def test_known_token_overshoot_is_preserved_and_no_second_turn_admitted():
    result,seen,closed,_=campaign(fault='overshoot')
    assert len(seen)==len(closed)==1
    assert result['paid_budget']['known_tokens']==500001
    assert result['paid_budget']['turns']==1
    assert not result['diagnostic_completed'] and result['stop_code'] is not None


@pytest.mark.parametrize('ordinal',[0,3,7])
@pytest.mark.parametrize('stage_fault',[1,2])
def test_malformed_stage_ends_unit_and_never_rerolls(ordinal,stage_fault):
    result,seen,closed,_=campaign(fault='malformed',ordinal=ordinal,stage_fault=stage_fault)
    assert result['diagnostic_completed'] and result['completed_units']==len(closed)==8
    assert len(seen)==14+stage_fault
    assert result['malformed_units']==1
    assert result['units'][ordinal]['expected_gold_match'] is False
    assert result['units'][ordinal]['admitted_turns']==stage_fault


@pytest.mark.parametrize('fault',['transport','unknown_usage','cleanup'])
@pytest.mark.parametrize('ordinal',[0,7])
def test_infrastructure_failure_stops_with_accounting_not_zero(fault,ordinal):
    result,seen,closed,_=campaign(fault=fault,ordinal=ordinal)
    calls=2*ordinal+(2 if fault=='cleanup' else 1)
    assert len(seen)==calls and len(closed)==ordinal+1
    assert not result['diagnostic_completed'] and result['stop_code'] is not None
    assert result['paid_budget']['turns']==calls
    assert result['paid_budget']['known_tokens']==7*(calls-(fault=='unknown_usage'))
    assert result['paid_budget']['usage_complete']==(fault!='unknown_usage')
    if fault=='cleanup':assert result['client_cleanup_ok'] is False


@pytest.mark.parametrize('phase',['before_dispatch','response_returned','unit_finished'])
def test_private_journal_failure_stops_before_another_model_call(phase):
    result,seen,closed,_=campaign(journal=Journal(fail=phase))
    assert result['stop_code']=='private_evidence_write_failure'
    assert not result['diagnostic_completed']
    assert len(seen)=={'before_dispatch':0,'response_returned':1,'unit_finished':2}[phase]


@pytest.mark.parametrize('mutate',[
    lambda p:p.update(completed_units=True),
    lambda p:p.update(diagnostic_completed=False),
    lambda p:p['units'][0].update(ordinal=False),
    lambda p:p['units'][0].update(expected_gold_match='PRIVATE'),
    lambda p:p['units'][0].update(expected_gold_match=True),
    lambda p:p['units'][0].update(admitted_turns=True),
    lambda p:p['paid_budget'].update(turns=15),
    lambda p:p['paid_budget'].update(known_tokens=1),
    lambda p:p.update(semantic_accuracy_accepted=True),
    lambda p:p.update(diagnostic_completed=1),
    lambda p:p['units'][0]['stages'][0].update(recheck=0),
    lambda p:p['units'][0]['stages'][1].update(prior_response_sha256='Z'*64),
    lambda p:p['units'][0]['stages'][1].update(stage='original',recheck=True,prior_response_sha256=None),
    lambda p:p['units'][0]['stages'][1]['predicate_states'].pop(),
    lambda p:p['units'][0]['stages'][1]['predicate_states'].append(copy.deepcopy(p['units'][0]['stages'][1]['predicate_states'][0])),
    lambda p:p['units'][0]['stages'][1]['predicate_states'][0].update(index=1),
    lambda p:p['units'][0]['stages'][1]['predicate_states'][0].update(predicate='prefers'),
    lambda p:p.update(raw='PRIVATE'),
])
def test_public_metadata_reconciliation_and_privacy(mutate):
    result,*_=campaign();mutate(result)
    with pytest.raises(c.DiagnosticStop):c.validate_public_result(result)


def test_ambiguous_original_cannot_claim_an_alternatives_stage():
    result,*_=campaign()
    pair=result['units'][5]
    pair['stages'][0]['states'][0]='ambiguous'
    pair['stages'][1]['predicate_states']=[
        row for row in pair['stages'][1]['predicate_states'] if row['index']==1]
    with pytest.raises(c.DiagnosticStop):c.validate_public_result(result)


@pytest.mark.parametrize('mutation',['ambiguous_alternative','wrong_final','false_recheck'])
def test_corrected_success_requires_unambiguous_selection_and_supported_recheck(mutation):
    result,*_=campaign(mode='gold')
    item=result['units'][1]
    if mutation=='ambiguous_alternative':
        next(row for row in item['stages'][1]['predicate_states']
             if row['state']=='not_established')['state']='ambiguous'
    elif mutation=='wrong_final':item['final_predicates']=['uses'];item['expected_gold_match']=False
    else:item['stages'][2]['states']=['not_established'];item['expected_gold_match']=False
    with pytest.raises(c.DiagnosticStop):c.validate_public_result(result)


def test_actual_candidate_derivation_preserves_initial_batches_and_exact_ordinary_bytes(tmp_path):
    repo=Path(__file__).resolve().parents[3]
    candidate=tmp_path/'candidate';code=tmp_path/'code'
    shutil.copytree('/private/tmp/hymem-staged-v1-root-fZvNIRgs/candidate',candidate)
    files=['benchmarks/'+name+'.py' for name in ('codex_subscription_staged_v1',
        'codex_subscription_warm_v3','codex_subscription_warm_v2',
        'codex_subscription_concurrent_v2','codex_subscription')]
    files+=['tools/diagnostics/'+name+'.py' for name in ('luna_staged_core_v1','luna_semantic_probe')]
    for name in files:
        raw=(repo/name).read_bytes()
        if name=='benchmarks/codex_subscription_staged_v1.py':
            before=b'_ROOT = Path(__file__).resolve().parents[1]'
            assert raw.count(before)==1
            raw=raw.replace(before,b'_ROOT = Path(__file__).resolve().parents[2] / "candidate"')
        target=code/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    script=r'''
import json,socket,sys
from dataclasses import asdict
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2]]
def deny(*a,**k):raise AssertionError('offline only')
socket.socket.connect=deny;socket.create_connection=deny
from hymem.extraction import chunk,grounding_staged_v1 as s,grounding_classification_v4 as g
from benchmarks import extraction_canary as gold
from tools.diagnostics import luna_staged_core_v1 as c,luna_semantic_probe as probe
def support(t,quote):
    return dict(state='supported',support=dict(evidence=[dict(source_message_id=t.source_message_id,
        region='owned',quote=quote)],checks={k:dict(state='supported',evidence_indices=[0])
        for k in ('attribution_and_roles','relation_and_polarity')}))
class Capture:
    observed_turns=request_attempts=0
    def __init__(self):self.retained=[];self.initial={}
    def complete(self,request):
        self.observed_turns+=1;self.request_attempts+=1
        payloads,failures=gold._request_source_payloads(request);assert not failures
        triples=[]
        if 'VERIFICATION PASS' not in request.system:
            for i,e in enumerate(gold._CANARY_EXPECTED_CLAIMS):
                fragment=next((p for p in payloads if p['source_message_id']==e[6]),None)
                owned=gold._TABLE_CLAIM_ROW if i==0 else gold._PROSE_BOUNDARY_RIGHT
                if not fragment or owned not in fragment['content']:continue
                triples.append(dict(subject=e[0],predicate=e[2] if i==0 else 'uses',
                    object=e[3],polarity=e[5],source_message_id=e[6],subject_type=e[1],object_type=e[4]))
        raw=json.dumps(dict(complete=True,triples=triples,markers=[]))
        self.retained.append((asdict(request),raw));return raw
    def complete_stage(self,request,bound,stage,recheck):
        self.observed_turns+=1;self.request_attempts+=1
        batch=bound if stage=='original' else bound.classification_batch
        t=batch.triples[0];mid=t.source_message_id
        i=next(i for i,e in enumerate(gold._CANARY_EXPECTED_CLAIMS) if e[6]==mid)
        if not recheck:self.initial.setdefault(mid,batch)
        quote=gold._TABLE_CLAIM_ROW if i==0 else gold._PROSE_BOUNDARY_RIGHT
        wanted=gold._CANARY_EXPECTED_CLAIMS[i][2]
        if stage=='original':
            s.validate_original_request(request,bound)
            assessment=support(t,quote) if t.predicate==wanted else dict(state='not_established',support=None)
            return json.dumps(dict(schema=s.ORIGINAL_SCHEMA,batch_sha256=batch.batch_sha256,
                complete=True,originals=[dict(index=0,original=assessment)]))
        s.validate_alternatives_request(request,bound)
        return json.dumps(dict(schema=s.ALTERNATIVES_SCHEMA,batch_sha256=batch.batch_sha256,
            original_response_sha256=bound.original_response_sha256,complete=True,
            alternatives=[dict(index=0,alternatives={p:support(t,quote) if p==wanted
                else dict(state='not_established',support=None) for p in g.PREDICATE_ORDER if p!=t.predicate})]))
fake=Capture()
result=chunk.extract_chunk(fake,gold._CANARY_CONTENT,source_records=gold._source_records(),
    completion_call_limit=gold.EXTRACTION_CANARY_MAX_COMPLETION_CALLS)
assert not result.failed and len(fake.retained)==8
events=[]
batches=c.derive_canary_batches(canary_module=gold,chunk_module=chunk,semantic_probe_module=probe,
    candidate=Path(sys.argv[1]),retained=tuple(fake.retained),record=events.append)
assert [b.triples[0].predicate for b in batches]==['deploys_to','uses']
assert all(b==fake.initial[b.triples[0].source_message_id] for b in batches)
assert len([e for e in events if e.get('phase')=='ordinary_replay'])==8
assert events[-1]['synthetic_table_advancement'] is True
mutated=list(fake.retained);bad=dict(mutated[0][0]);bad['user']+=' '
mutated[0]=(bad,mutated[0][1])
try:c.derive_canary_batches(canary_module=gold,chunk_module=chunk,semantic_probe_module=probe,
    candidate=Path(sys.argv[1]),retained=tuple(mutated),record=lambda e:None)
except BaseException:pass
else:raise AssertionError('ordinary request byte drift accepted')
print(json.dumps(dict(ordinary_replays=8,initial_batches=2,model_calls=0)))
'''
    out=subprocess.run([sys.executable,'-I','-B','-c',script,str(candidate),str(code)],
        capture_output=True,text=True,timeout=45)
    assert out.returncode==0,out.stderr
    assert json.loads(out.stdout)==dict(ordinary_replays=8,initial_batches=2,model_calls=0)
