"""Independent root tests. All judgments are invented, not model accuracy."""
from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from hymem.extraction import grounding_classification_v3 as v3
from tools.diagnostics import luna_claim_task_contract_v1 as c
from tools.diagnostics import luna_semantic_cases as cases


def sources(case):
    result = []
    for source in case.sources:
        data = asdict(source)
        data['contexts'] = tuple(v3.GroundingContext(**x) for x in data['contexts'])
        result.append(v3.GroundingSource(**data))
    return tuple(result)


def answer(batch, states, evidence=(), *, arm='B'):
    rows = []
    for i, (triple, state) in enumerate(zip(batch.triples, states, strict=True)):
        support = None
        if state == 'supported':
            checks = ['attribution_and_roles', 'relation_and_polarity'] + [
                k for k in ('value_text','value_numeric','value_unit','temporal_scope')
                if getattr(triple,k) is not None]
            pool = copy.deepcopy(list(evidence[i]))
            support = dict(evidence=pool, checks={
                k:dict(state='supported', evidence_indices=list(range(len(pool))))
                for k in checks})
        item = dict(index=i,original=dict(state=state,support=support))
        if arm == 'A':
            item['alternatives'] = ({p:dict(state='not_established',support=None)
                for p in v3.PREDICATE_ORDER if p!=triple.predicate}
                if state=='not_established' else None)
        rows.append(item)
    return dict(schema=c.B_SCHEMA if arm=='B' else v3.GROUNDING_CONTRACT_VERSION,
                complete=True,batch_sha256=batch.batch_sha256,classifications=rows)


def parse(batch, payload, arm='B'):
    return c.parse_arm_response(arm,json.dumps(payload),batch)


def sample(**qualifiers):
    triple=v3.Triple('Iris','uses','CedarTool',1,source_message_id=7,**qualifiers)
    source=v3.GroundingSource(7,'Iris uses CedarTool today at 64 ms in quiet mode.')
    request,batch=c.build_arm_request('B',(triple,),(source,))
    row=dict(source_message_id=7,region='owned',quote=source.content)
    return request,batch,answer(batch,['supported'],[[row]])


@pytest.mark.parametrize('index',range(24))
def test_exact_data_and_synthetic_frozen_controls(index):
    case=cases.cases()[index]
    a,batch=c.build_arm_request('A',case.triples,sources(case))
    b,other=c.build_arm_request('B',case.triples,sources(case))
    prior,original=v3.build_grounding_request(case.triples,sources(case))
    assert a==prior and batch==original==other
    assert a.user==b.user and a.system!=b.system
    assert (a.max_tokens,a.temperature,a.response_format)==(4096,0.0,'json')
    assert (b.max_tokens,b.temperature,b.response_format)==(4096,0.0,'json')
    assert c.build_arm_output_schema('A',batch)==v3.build_output_schema(batch)
    evidence=[]
    states=[]
    for triple,label in zip(case.triples,case.expected,strict=True):
        state='supported' if case.category=='supported' else 'not_established'
        states.append(state)
        evidence.append([dict(source_message_id=triple.source_message_id,region=r,quote=q)
                         for r,q in label.evidence])
        assert label.rationale not in b.user+b.system
    payload=answer(batch,states,evidence)
    jsonschema.Draft202012Validator(c.build_arm_output_schema('B',batch)).validate(payload)
    result=parse(batch,payload)
    assert [x.state for x in result.original]==states
    assert result.alternative_states is None and result.final_verdicts is None
    assert v3.PREDICATE_ORDER==tuple(json.loads(b.user)['batch']['predicates'])
    for arm,request in [('A',a),('B',b)]:
        c.validate_arm_request(arm,request,batch)
        with pytest.raises(v3.GroundingContractError):
            c.validate_arm_request('B' if arm=='A' else 'A',request,batch)


def test_frozen_sources_labels_and_prompt_anchor():
    assert hashlib.sha256(Path(v3.__file__).read_bytes()).hexdigest()==(
        '435c0edf52197a5ffa9e715db24156e26109bf7445ad7c9baba2f632ba7f7a76')
    assert cases.suite_sha256()=='511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925'
    with pytest.raises(RuntimeError,match='pinned_v3_prompt_drift'):
        c._replace_once('x x','x','y')
    assert 'alternative' not in c.B_SYSTEM.split('Predicate meanings:')[1].split('Return exactly')[0]


@pytest.mark.parametrize('mutate',[
    lambda p:p.update(schema=v3.GROUNDING_CONTRACT_VERSION),
    lambda p:p.update(complete=1),
    lambda p:p.update(extra='PRIVATE_SENTINEL'),
    lambda p:p.update(batch_sha256='0'*64),
    lambda p:p['classifications'][0].update(index=True),
    lambda p:p['classifications'][0].update(alternatives=None),
    lambda p:p['classifications'][0]['original'].update(support=None),
    lambda p:p['classifications'][0]['original']['support']['checks'].pop('attribution_and_roles'),
    lambda p:p['classifications'][0]['original']['support']['checks']['relation_and_polarity'].update(evidence_indices=[True]),
    lambda p:p['classifications'][0]['original']['support']['checks']['relation_and_polarity'].update(evidence_indices=[1]),
    lambda p:p['classifications'][0]['original']['support']['checks']['relation_and_polarity'].update(evidence_indices=[0,0]),
    lambda p:p['classifications'][0]['original']['support']['checks']['relation_and_polarity'].update(state='not_established'),
    lambda p:p['classifications'][0]['original']['support']['evidence'][0].update(quote='made up'),
    lambda p:p['classifications'][0]['original']['support']['evidence'][0].update(source_message_id=True),
    lambda p:p['classifications'][0]['original']['support']['evidence'][0].update(region='conversation_0'),
])
def test_fail_closed_shape_and_full_evidence(mutate):
    _,batch,payload=sample()
    mutate(payload)
    with pytest.raises(v3.GroundingContractError): parse(batch,payload)


@pytest.mark.parametrize('qualifier,value',[
    ('value_text','quiet'),('value_numeric',0.0),('value_unit','ms'),('temporal_scope','today')])
def test_each_qualifier_still_requires_check(qualifier,value):
    _,batch,payload=sample(**{qualifier:value})
    parse(batch,payload)
    del payload['classifications'][0]['original']['support']['checks'][qualifier]
    with pytest.raises(v3.GroundingContractError,match='support:checks'): parse(batch,payload)


@pytest.mark.parametrize('state',['not_established','ambiguous'])
def test_negative_is_original_only_never_fabricated_alternatives(state):
    _,batch,_=sample()
    payload=answer(batch,[state])
    result=parse(batch,payload)
    assert result.original[0].state==state
    assert result.original[0].evidence==result.original[0].checks==()
    assert result.alternative_states is None and result.final_verdicts is None
    with pytest.raises(v3.GroundingContractError): parse(batch,payload,'A')
    baseline=answer(batch,[state],arm='A')
    with pytest.raises(v3.GroundingContractError): parse(batch,baseline)
    original=parse(batch,baseline,'A')
    assert original.final_verdicts[0].status==('unsupported' if state=='not_established' else 'uncertain')


def test_parent_prefix_boundary_checks_and_immutability():
    case=cases.cases()[11]
    _,batch=c.build_arm_request('B',case.triples,sources(case))
    pool=[dict(source_message_id=112,region=r,quote=q) for r,q in case.expected[0].evidence]
    payload=answer(batch,['supported'],[pool])
    result=parse(batch,payload)
    before=copy.deepcopy(result)
    payload['classifications'][0]['original']['support']['evidence'][0]['quote']='changed'
    assert result==before
    for position,quote in [(1,'Later we discussed an unrelated router.'),(0,'not present')]:
        bad=answer(batch,['supported'],[pool])
        bad['classifications'][0]['original']['support']['evidence'][position]['quote']=quote
        with pytest.raises(v3.GroundingContractError): parse(batch,bad)
    bad=answer(batch,['supported'],[[pool[0],pool[2]]])
    with pytest.raises(v3.GroundingContractError,match='parent_required'): parse(batch,bad)


def test_bounds_arm_binding_and_fresh_schema():
    request,batch,payload=sample()
    for arm in (None,True,0,'a','AB'):
        with pytest.raises(v3.GroundingContractError): c.parse_arm_response(arm,json.dumps(payload),batch)
    for bad in (replace(request,user=request.user+' '),replace(request,temperature=False),replace(request,max_tokens=4096.0)):
        with pytest.raises(v3.GroundingContractError): c.validate_arm_request('B',bad,batch)
    with pytest.raises(v3.GroundingContractError):
        c.validate_arm_request('B',request,replace(batch,canonical_json=batch.canonical_json+' '))
    for raw in (None,{},' '*65537,'['*2000):
        with pytest.raises(v3.GroundingContractError): c.parse_arm_response('B',raw,batch)
    schema=c.build_arm_output_schema('B',batch)
    assert all(k.startswith('assessment_') for k in schema['$defs'])
    for definition in schema['properties']['classifications']['items']['anyOf']:
        assert set(definition['required'])==set(definition['properties'])=={'index','original'}
    schema['$defs'].clear()
    assert c.build_arm_output_schema('B',batch)['$defs']


def test_eight_claims_exact_order_and_missing_late_check():
    _,batch,_=sample()
    triples=tuple(replace(batch.triples[0],object=f'CedarTool{i}') for i in range(8))
    _,many=c.build_arm_request('B',triples,batch.sources)
    payload=answer(many,['not_established']*8)
    assert len(parse(many,payload).original)==8
    payload['classifications'][-1]['index']=0
    with pytest.raises(v3.GroundingContractError): parse(many,payload)
    with pytest.raises(v3.GroundingContractError):
        c.build_arm_request('B',triples+(replace(triples[0],object='extra'),),batch.sources)
