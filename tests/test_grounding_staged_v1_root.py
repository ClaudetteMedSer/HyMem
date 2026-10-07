"""Root adversarial staged-contract checks, with invented model judgments only."""
import copy
from dataclasses import asdict, replace
import itertools
import json

import jsonschema
import pytest

from hymem.extraction import grounding_staged_v1 as s
from hymem.extraction import grounding_classification_v4 as g
from tools.diagnostics import luna_semantic_cases as cases


def fixture(index=9):
    case=cases.cases()[index]
    sources=[]
    for source in case.sources:
        values=asdict(source)
        values['contexts']=tuple(g.GroundingContext(**c) for c in values['contexts'])
        sources.append(g.GroundingSource(**values))
    return s.build_original_request(case.triples,tuple(sources))


def assessment(triple,state='not_established',quotes=()):
    if state!='supported':return dict(state=state,support=None)
    checks=['attribution_and_roles','relation_and_polarity']+[
        k for k in ('value_text','value_numeric','value_unit','temporal_scope') if getattr(triple,k) is not None]
    return dict(state=state,support=dict(evidence=list(quotes),checks={k:
        dict(state='supported',evidence_indices=list(range(len(quotes)))) for k in checks}))


def envelope(schema,rows):
    out={k:copy.deepcopy(v['enum'][0]) for k,v in schema['properties'].items() if 'enum' in v}
    key=next(k for k in ('originals','alternatives','classifications') if k in schema['properties'])
    out[key]=rows
    return out


def original(batch,states=None,pools=None):
    states=states or ['not_established']*len(batch.triples)
    pools=pools or [()]*len(batch.triples)
    return envelope(s.build_original_output_schema(batch),[
        dict(index=i,original=assessment(t,state,pool)) for i,(t,state,pool) in
        enumerate(zip(batch.triples,states,pools,strict=True))])


def alternatives(batch,first,positive=(),ambiguous=(),pools=None):
    request,bound=s.build_alternatives_request(batch,json.dumps(first))
    rows=[]
    for i,t in enumerate(batch.triples):
        if first['originals'][i]['original']['state']!='not_established':continue
        pool=() if pools is None else pools[i]
        rows.append(dict(index=i,alternatives={p:assessment(t,'supported' if (i,p) in positive
            else 'ambiguous' if (i,p) in ambiguous else 'not_established',pool)
            for p in g.PREDICATE_ORDER if p!=t.predicate}))
    return request,bound,envelope(s.build_alternatives_output_schema(bound),rows)


def quote(sid,text,region='owned'):
    return dict(source_message_id=sid,region=region,quote=text)


@pytest.mark.parametrize('index',range(24))
def test_all_frozen_controls_preserve_data_and_mechanical_outcomes(index):
    req,batch=fixture(index);case=cases.cases()[index]
    original_request,_=g.build_grounding_request(batch.triples,batch.sources)
    assert req.user==original_request.user
    assert (req.temperature,req.max_tokens,req.response_format)==(0.0,4096,'json')
    assert all(x.rationale not in req.system+req.user for x in case.expected)
    pools=[[quote(t.source_message_id,text,region) for region,text in e.evidence]
        for t,e in zip(case.triples,case.expected,strict=True)]
    first=original(batch,['supported' if case.category=='supported' else 'not_established']*len(batch.triples),pools)
    jsonschema.validate(first,s.build_original_output_schema(batch))
    s.validate_original_request(req,batch)
    s.parse_original_response(json.dumps(first),batch)
    if case.category=='supported':second=None
    else:
        positives=[(i,e.predicate) for i,e in enumerate(case.expected) if case.category=='correction']
        request,bound,second=alternatives(batch,first,positive=positives,pools=pools)
        s.validate_alternatives_request(request,bound)
        jsonschema.validate(second,s.build_alternatives_output_schema(bound))
    review=s.parse_staged_responses(batch,json.dumps(first),None if second is None else json.dumps(second))
    assert [v.status in e.statuses for v,e in zip(review.verdicts,case.expected,strict=True)]==[True]*len(case.triples)


@pytest.mark.parametrize('states',itertools.product(('supported','ambiguous','not_established'),repeat=2))
def test_alternative_uniqueness_is_not_best_effort(states):
    _,batch=fixture(12);t=batch.triples[0]
    pool=[quote(t.source_message_id,batch.sources[0].content)]
    first=original(batch)
    names=('prefers','avoids')
    _,_,second=alternatives(batch,first,positive=[(0,p) for p,state in zip(names,states) if state=='supported'],
        ambiguous=[(0,p) for p,state in zip(names,states) if state=='ambiguous'],pools=[pool])
    review=s.parse_staged_responses(batch,json.dumps(first),json.dumps(second))
    count=states.count('supported')
    expected='replace_predicate' if count==1 and 'ambiguous' not in states else (
        'unsupported' if count==0 and 'ambiguous' not in states else 'uncertain')
    assert review.verdicts[0].status==expected


def test_mixed_noncontiguous_negative_indices_and_actual_prior_binding():
    source=g.GroundingSource(7,'Iris uses CedarTool. Iris prefers AsterDB.')
    triples=(g.Triple('Iris','uses','CedarTool',1,source_message_id=7),
             g.Triple('Iris','uses','AsterDB',1,source_message_id=7),
             g.Triple('Iris','prefers','AsterDB',1,source_message_id=7),
             g.Triple('Iris','avoids','AsterDB',1,source_message_id=7))
    _,batch=s.build_original_request(triples,(source,))
    pools=[[quote(7,source.content)]]*4
    first=original(batch,['supported','not_established','supported','not_established'],pools)
    req,bound,second=alternatives(batch,first,positive=[(1,'prefers'),(3,'prefers')],pools=pools)
    assert [r['index'] for r in second['alternatives']]==[1,3]
    assert len(s.parse_staged_responses(batch,json.dumps(first),json.dumps(second)).verdicts)==4
    changed=copy.deepcopy(first)
    changed['originals'][0]['original']['support']['evidence'][0]['quote']='Iris uses CedarTool.'
    s.parse_original_response(json.dumps(changed),batch)
    with pytest.raises(g.GroundingContractError):
        s.parse_staged_responses(batch,json.dumps(changed),json.dumps(second))
    second['alternatives'].reverse()
    with pytest.raises(g.GroundingContractError):
        s.parse_staged_responses(batch,json.dumps(first),json.dumps(second))


@pytest.mark.parametrize('mutate',[
    lambda x:x.update(complete=1),
    lambda x:x.update(extra='PRIVATE'),
    lambda x:x['originals'][0].update(index=True),
    lambda x:x['originals'][0].update(alternatives={}),
    lambda x:x['originals'][0]['original'].update(support={}),
])
def test_forged_original_does_not_authorize_alternatives(mutate):
    _,batch=fixture(12);first=original(batch);mutate(first)
    with pytest.raises(g.GroundingContractError):s.build_alternatives_request(batch,json.dumps(first))


@pytest.mark.parametrize('mutate',[
    lambda x:x['alternatives'][0]['alternatives'].pop('prefers'),
    lambda x:x['alternatives'][0]['alternatives'].update(uses=dict(state='not_established',support=None)),
    lambda x:x['alternatives'][0].update(index=True),
    lambda x:x['alternatives'].append(copy.deepcopy(x['alternatives'][0])),
    lambda x:x.update(complete=1),
    lambda x:x.update(batch_sha256='0'*64),
])
def test_incomplete_or_unbound_alternatives_fail_closed(mutate):
    _,batch=fixture(12);first=original(batch);_,_,second=alternatives(batch,first);mutate(second)
    with pytest.raises(g.GroundingContractError):s.parse_staged_responses(batch,json.dumps(first),json.dumps(second))


@pytest.mark.parametrize('state',['supported','ambiguous'])
def test_nonnegative_originals_never_need_or_accept_alternative_work(state):
    _,batch=fixture(12);pool=[quote(batch.triples[0].source_message_id,batch.sources[0].content)]
    first=original(batch,[state],[pool])
    with pytest.raises(g.GroundingContractError):s.build_alternatives_request(batch,json.dumps(first))
    assert s.parse_staged_responses(batch,json.dumps(first),None).verdicts[0].status==(
        'supported' if state=='supported' else 'uncertain')
    with pytest.raises(g.GroundingContractError):s.parse_staged_responses(batch,json.dumps(first),'{}')


def test_nonexistent_region_and_out_of_prefix_stay_rejected_in_both_stages():
    _,batch=fixture(19);t=batch.triples[0];source=batch.sources[0]
    pool=[quote(t.source_message_id,'Later I prefer that editor.'),
          quote(t.source_message_id,source.contexts[0].content,'conversation_0')]
    first=original(batch,['supported'],[pool])
    with pytest.raises(g.GroundingContractError,match='evidence:context_scope'):
        s.parse_original_response(json.dumps(first),batch)
    first=original(batch)
    _,_,second=alternatives(batch,first,positive=[(0,'uses')],pools=[pool])
    with pytest.raises(g.GroundingContractError,match='evidence:context_scope'):
        s.parse_staged_responses(batch,json.dumps(first),json.dumps(second))


@pytest.mark.parametrize('field,value',[('temperature',False),('max_tokens',4096.0),('system','changed'),('user','changed')])
def test_both_requests_have_exact_typed_binding(field,value):
    request,batch=fixture(12)
    with pytest.raises(g.GroundingContractError):s.validate_original_request(replace(request,**{field:value}),batch)
    request,bound,_=alternatives(batch,original(batch))
    with pytest.raises(g.GroundingContractError):s.validate_alternatives_request(replace(request,**{field:value}),bound)


@pytest.mark.parametrize('indices',[(False,),[0],(0.0,),(1,)])
def test_forged_alternatives_indices_never_reach_request_or_schema(indices):
    _,batch=fixture(12);request,bound,_=alternatives(batch,original(batch))
    damaged=replace(bound,negative_indices=indices)
    with pytest.raises(g.GroundingContractError):s.validate_alternatives_request(request,damaged)
    with pytest.raises(g.GroundingContractError):s.build_alternatives_output_schema(damaged)


def test_negative_stage_schema_requires_map_not_nullable_and_is_fresh():
    _,batch=fixture(12);_,bound,raw=alternatives(batch,original(batch))
    schema=s.build_alternatives_output_schema(bound)
    raw['alternatives'][0]['alternatives']=None
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(raw,schema)
    saved=copy.deepcopy(schema)
    schema['$defs'].clear()
    assert s.build_alternatives_output_schema(bound)==saved
    permitted={'type','properties','required','additionalProperties','enum','anyOf','items',
        'minItems','maxItems','minimum','maximum','minLength','maxLength','$schema','$defs','$ref'}
    def visit(node):
        assert type(node) is dict and set(node)<=permitted
        if node.get('type')=='object':
            assert node['additionalProperties'] is False and set(node['required'])==set(node['properties'])
        for key in ('properties','$defs'):
            for child in node.get(key,{}).values():visit(child)
        if 'items' in node:visit(node['items'])
        for child in node.get('anyOf',[]):visit(child)
    visit(saved);visit(s.build_original_output_schema(batch))


@pytest.mark.parametrize('stage',['original','alternatives'])
def test_no_qualifier_check_can_be_omitted_after_splitting(stage):
    _,batch=fixture(17);t=batch.triples[0]
    pool=[quote(t.source_message_id,batch.sources[0].content)]
    first=original(batch,['supported'],[pool])
    if stage=='original':
        first['originals'][0]['original']['support']['checks'].pop('value_numeric')
        with pytest.raises(g.GroundingContractError,match='support:checks'):
            s.parse_original_response(json.dumps(first),batch)
    else:
        first=original(batch)
        _,_,second=alternatives(batch,first,positive=[(0,'uses')],pools=[pool])
        second['alternatives'][0]['alternatives']['uses']['support']['checks'].pop('value_numeric')
        with pytest.raises(g.GroundingContractError,match='support:checks'):
            s.parse_staged_responses(batch,json.dumps(first),json.dumps(second))


def test_unselected_alternative_still_cannot_cite_invalid_evidence():
    _,batch=fixture(12);first=original(batch)
    pool=[quote(batch.triples[0].source_message_id,batch.sources[0].content)]
    _,_,second=alternatives(batch,first,positive=[(0,'prefers'),(0,'avoids')],pools=[pool])
    second['alternatives'][0]['alternatives']['avoids']['support']['evidence'][0]['region']='conversation_0'
    with pytest.raises(g.GroundingContractError,match='evidence:context_missing'):
        s.parse_staged_responses(batch,json.dumps(first),json.dumps(second))
