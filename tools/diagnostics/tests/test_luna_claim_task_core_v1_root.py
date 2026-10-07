"""Root independent campaign faults. Every model response is invented."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import runpy

import pytest

from tools.diagnostics import luna_claim_task_core_v1 as c
from tools.diagnostics import luna_semantic_cases as cases


def canaries():
    g=c.contract.v3
    return tuple(g.build_grounding_request(
        (g.Triple(subject,predicate,obj,1,source_message_id=sid),),
        (g.GroundingSource(sid,'Synthetic local input, not retained evidence.'),))[1]
        for sid,subject,predicate,obj in (
            (71001,'HyMem Canary Relay','deploys_to','Fly.io'),
            (71002,'Avery Boundary Canary','uses','PostgreSQL')))


class Journal:
    def __init__(self,fail=None): self.events=[]; self.fail=fail
    def record(self,key,value):
        if value['phase']==self.fail:
            raise c.DiagnosticStop('private_evidence_write_failure')
        self.events.append((key,copy.deepcopy(value)))


def campaign(*, fault=None, ordinal=0, journal=None):
    seen=[]; closed=[]
    journal=journal or Journal()
    warm=c.transport.warm
    class Synthetic:
        def __init__(self,key,cap,budget):
            self.key,self.budget=key,budget
            budget.register(key,warm.BudgetLimits(*cap))
        @property
        def observed_turns(self): return self.budget.snapshot()['questions'][self.key]['turns']
        @property
        def observed_tokens(self): return self.budget.snapshot()['questions'][self.key]['known_tokens']
        @property
        def usage_complete(self): return self.budget.snapshot()['questions'][self.key]['usage_complete']
        def complete_arm(self,arm,request,batch):
            c.contract.validate_arm_request(arm,request,batch)
            seen.append((arm,request,batch))
            index=int(self.key.rsplit('-',1)[-1])
            active=fault if index==ordinal else None
            self.budget.reserve(self.key)
            self.budget.before_turn(self.key,dict(auth='chatgpt',model=warm.base.MODEL,
                config_isolation_admitted=True,inference_enabled=False,
                quota_windows=[dict(remaining_percent=90)]))
            self.budget.settle(self.key,used=None if active=='unknown_usage' else 7,turn_started=True)
            if active=='transport':
                raise RuntimeError('PRIVATE_DO_NOT_EXPORT')
            if active=='malformed': return 'PRIVATE_DO_NOT_EXPORT'
            rows=[]
            for i,triple in enumerate(batch.triples):
                row=dict(index=i,original=dict(state='not_established',support=None))
                if arm=='A':
                    row['alternatives']={p:dict(state='not_established',support=None)
                        for p in c.contract.v3.PREDICATE_ORDER if p!=triple.predicate}
                rows.append(row)
            return json.dumps(dict(schema=c.contract.B_SCHEMA if arm=='B' else
                c.contract.v3.GROUNDING_CONTRACT_VERSION,complete=True,
                batch_sha256=batch.batch_sha256,classifications=rows))
        def close(self):
            closed.append(self.key)
            if fault=='cleanup' and int(self.key.rsplit('-',1)[-1])==ordinal:
                raise RuntimeError('PRIVATE_CLEANUP_DO_NOT_EXPORT')
    result=c.run_campaign(concurrent=warm.concurrent,warm=warm,binary='unused',
        cases_module=cases,canary_batches=canaries(),journal=journal,client_factory=Synthetic)
    assert 'PRIVATE_' not in json.dumps(result)
    return result,seen,closed,journal


def test_fixed_order_identity_and_diagnostic_not_semantic_success():
    result,seen,closed,journal=campaign()
    expected=[]
    for index in [2,3,5,6,7,9,11,12,13,17,19,21]:
        expected.extend(('control',index,arm) for arm in ('AB' if index%2==0 else 'BA'))
        if index==12: expected.append(('nominated_prefers',12,'B'))
    expected.extend([('table_canary',0,'A'),('table_canary',0,'B'),
                     ('prose_canary',1,'B'),('prose_canary',1,'A')])
    assert [(u.kind,u.index,u.arm) for u in c.schedule()]==expected
    assert len(seen)==len(set(closed))==29
    assert result['diagnostic_completed'] and result['completed_units']==29
    assert result['paid_budget']==dict(turns=29,known_tokens=203,usage_complete=True,in_flight=0,reserved=0)
    assert all(result[k] is False for k in ['semantic_accuracy_accepted','full_lme_ready',
                                          'completed_and_clean','process_cleanup_verified'])
    assert all(x['state']=='not_established' for unit in result['units'] for x in unit['result']['original'])
    for i,unit in enumerate(result['units']):
        if unit['arm']=='B':
            assert unit['result']['alternative_states'] is None
            assert unit['result']['final_verdicts'] is None
        assert unit['scope_ambiguous']==(unit['kind']=='control' and unit['index']==5)
        assert len([e for key,e in journal.events if key==f'unit-{i:02d}' and e['phase']=='before_dispatch'])==1
    for key in {(kind,index) for kind,index,_ in expected if kind!='nominated_prefers'}:
        positions=[i for i,(kind,index,_) in enumerate(expected) if (kind,index)==key]
        first,second=[seen[i] for i in positions]
        assert first[1].user==second[1].user and first[2]==second[2]
    nominated=next(i for i,x in enumerate(expected) if x[0]=='nominated_prefers')
    before=asdict(seen[nominated-1][2].triples[0]); after=asdict(seen[nominated][2].triples[0])
    assert before.pop('predicate')=='uses' and after.pop('predicate')=='prefers' and before==after
    assert seen[nominated-1][2].sources==seen[nominated][2].sources
    assert [x[2].triples[0].predicate for x in seen[-2:]]==['uses','uses']


@pytest.mark.parametrize('ordinal',[0,15,28])
def test_malformed_semantics_continue_without_reroll(ordinal):
    out,seen,closed,_=campaign(fault='malformed',ordinal=ordinal)
    assert len(seen)==len(closed)==29 and out['diagnostic_completed']
    assert out['malformed_units']==1 and out['units'][ordinal]['outcome']=='malformed'
    assert out['units'][ordinal]['result'] is None and not out['semantic_accuracy_accepted']


@pytest.mark.parametrize('fault',['transport','unknown_usage','cleanup'])
@pytest.mark.parametrize('ordinal',[0,28])
def test_terminal_failure_preserves_usage_and_stops(fault,ordinal):
    out,seen,closed,_=campaign(fault=fault,ordinal=ordinal)
    assert len(seen)==len(closed)==ordinal+1
    assert not out['diagnostic_completed'] and out['stop_code'] is not None
    assert out['paid_budget']['turns']==ordinal+1
    assert out['paid_budget']['known_tokens']==7*(ordinal if fault=='unknown_usage' else ordinal+1)
    assert out['paid_budget']['usage_complete']==(fault!='unknown_usage')
    if fault=='cleanup': assert out['client_cleanup_ok'] is False


@pytest.mark.parametrize('phase',['before_dispatch','response_returned','validated_result','unit_finished'])
def test_journal_fault_stops_and_closes(phase):
    out,seen,closed,_=campaign(journal=Journal(fail=phase))
    assert out['stop_code']=='private_evidence_write_failure'
    assert not out['diagnostic_completed']
    assert len(seen)==len(closed)==(0 if phase=='before_dispatch' else 1)


@pytest.mark.parametrize('mutate',[
    lambda p:p.update(completed_units=True),
    lambda p:p.update(diagnostic_completed=False),
    lambda p:p['units'][0].update(ordinal=False),
    lambda p:p['units'][0]['result']['original'][0].update(evidence_sha256='PRIVATE_'*8),
    lambda p:p['units'][0]['result']['original'][0].update(check_count=False),
    lambda p:p['paid_budget'].update(turns=28),
    lambda p:p['paid_budget'].update(known_tokens=1),
    lambda p:p.update(semantic_accuracy_accepted=True),
    lambda p:p['units'][0].update(label='negative'),
    lambda p:p['units'][0]['result'].update(original=[]),
])
def test_metadata_tamper_fails_closed(mutate):
    out,*_=campaign()
    mutate(out)
    with pytest.raises(c.DiagnosticStop): c.validate_public_result(out)


def test_real_candidate_derivation_keeps_original_prose_and_exact_replay():
    # Reuse only the pinned root fixture builder; all eight responses are invented
    # locally by its fake client, never loaded from a private production/run store.
    helper=runpy.run_path(str(Path(__file__).with_name('test_luna_classification_bundle_v3_root.py')))
    code=helper['CODE']; repo=helper['REPO']
    from hashlib import sha256
    adapter=(repo/'benchmarks/codex_subscription_claim_task_v1.py').read_bytes()
    adapter=adapter.replace(
        b'd3c955922219d99b359aa6619b821e7da1662e36ecea54d09b1267dce38d2c32',
        sha256((code/'benchmarks/codex_subscription_classification_v3.py').read_bytes()).hexdigest().encode())
    (code/'benchmarks/codex_subscription_claim_task_v1.py').write_bytes(adapter)
    for name in ('luna_claim_task_contract_v1.py','luna_claim_task_core_v1.py'):
        (code/'tools/diagnostics'/name).write_bytes((repo/'tools/diagnostics'/name).read_bytes())
    script=helper['PREAMBLE']+r'''
from tools.diagnostics import luna_claim_task_core_v1 as task
class Capture(CanaryClient):
    def __init__(self):
        super().__init__()
        self.retained=[]; self.initial={}
    def complete(self,request):
        raw=super().complete(request)
        self.retained.append((asdict(request),raw))
        return raw
    def complete_grounding(self,request,batch):
        self.initial.setdefault(batch.triples[0].source_message_id,batch)
        return super().complete_grounding(request,batch)
fake=Capture()
result=check.run_canary(gold,chunk,fake,candidate=Path(sys.argv[1]))
assert result['passed'] and len(fake.retained)==8
events=[]
derived=task.derive_canary_batches(canary_module=gold,chunk_module=chunk,
    semantic_probe_module=core,candidate=Path(sys.argv[1]),retained=tuple(fake.retained),
    record=events.append)
assert [b.triples[0].predicate for b in derived]==['deploys_to','uses']
assert all(b==fake.initial[b.triples[0].source_message_id] for b in derived)
assert len([e for e in events if e.get('phase')=='ordinary_replay'])==8
assert len([e for e in events if e.get('phase','').endswith('_before_dispatch')])==2
assert events[-1]['synthetic_table_advancement'] is True
mutated=list(fake.retained)
bad=dict(mutated[0][0]); bad['user']+=' '
mutated[0]=(bad,mutated[0][1])
try:
    task.derive_canary_batches(canary_module=gold,chunk_module=chunk,
        semantic_probe_module=core,candidate=Path(sys.argv[1]),retained=tuple(mutated),record=lambda e:None)
except BaseException:
    pass
else: raise AssertionError('changed ordinary bytes accepted')
print(json.dumps(dict(ordinary_replays=8,original_batches=2,model_calls=0)))
'''
    assert helper['execute'](script)==dict(ordinary_replays=8,original_batches=2,model_calls=0)
