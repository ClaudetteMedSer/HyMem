"""Independent v4 authority controls; scripted parsing, not live accuracy."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from benchmarks import digest_retention_inventory as r
from hymem.extraction.llm import LLMRequest
from tests.digest_retention_inventory_v2_fixtures import build_cases


def make_matching(facet='constraints', *, uncertain=False, second=False):
    case = deepcopy(next(c for c in build_cases()
                         if c['pair']=='retrieval-trigger-only-condition' and c['variant']=='defective'))
    candidate = case['payload']['procedure_items'][0]['candidate']
    candidate['entities_involved'] = ['coordinator confirmation of an empty room']
    candidate['name'] = 'Open cabinet only after coordinator confirms the room is empty'
    candidate['steps'][0]['tool'] = 'cabinet-key'
    plan = r.prepare_retention_inventory(case['payload'], LLMRequest('', '', max_tokens=8192), max_calls=64)
    scope = next(s for s in plan.requests if s.kind=='procedure')
    units = [u.unit_id for u in scope.units]
    obligations = [{'facet':facet, 'unit_ids':units, 'text':'Opening requires prior room clearance.'}]
    if second:
        obligations.append({'facet':'material_facts', 'unit_ids':units, 'text':'Collect labels.'})
    parsed = r.parse_inventory(json.dumps({'obligations':obligations,'no_material_unit_ids':[],
                                         'uncertain_unit_ids':units[:1] if uncertain else []}),scope)
    assert parsed.inventory is not None
    return r.prepare_matching(scope,parsed.inventory)


def field(matching,path):
    return next(f.field_id for f in matching.inventory_scope.record_scope.fields if f.path==path)


@pytest.mark.parametrize('facet',['constraints','ordering'])
@pytest.mark.parametrize('verdict',['retained','altered'])
@pytest.mark.parametrize('paths',[
    ['/candidate/triggers/0'], ['/candidate/entities_involved/0'],
    ['/candidate/triggers/0','/candidate/entities_involved/0'],
])
def test_procedure_rule_requires_an_assertion_witness(facet,verdict,paths):
    m=make_matching(facet)
    result=r.parse_matching(json.dumps({'o0':[verdict,[field(m,p) for p in paths]]}),m)
    assert not result.structure_valid and result.judgments==()
    assert not result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize('path',[
    '/candidate/name','/candidate/description','/candidate/steps/0/action','/candidate/steps/0/tool',
])
def test_legitimate_field_roles_are_not_categorically_excluded(path):
    m=make_matching()
    result=r.parse_matching(json.dumps({'o0':['retained',[field(m,path)]]}),m)
    assert result.structure_valid and result.model_retention_satisfied
    # The guard does not prove these illustrative witnesses actually support it.
    assert not result.semantic_verified and not result.publication_authorized


def test_mixed_retrieval_and_assertion_witness_is_not_semantic_proof():
    m=make_matching()
    refs=[field(m,'/candidate/triggers/0'),field(m,'/candidate/steps/0/action')]
    result=r.parse_matching(json.dumps({'o0':['retained',refs]}),m)
    # The chosen step merely says open the cabinet. This remains a possible
    # false semantic judgment; the narrow guard must not claim to solve it.
    assert result.structure_valid and result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized


def test_uncertain_metadata_does_not_become_an_affirmative_result():
    m=make_matching()
    result=r.parse_matching(json.dumps({'o0':['uncertain',[field(m,'/candidate/triggers/0')]]}),m)
    assert result.structure_valid and not result.model_retention_satisfied


def test_one_bad_witness_rejects_entire_reply_without_salvage():
    m=make_matching(second=True)
    result=r.parse_matching(json.dumps({'o0':['retained',[field(m,'/candidate/triggers/0')]],
                                       'o1':['retained',[field(m,'/candidate/steps/1/action')]]}),m)
    assert not result.structure_valid and result.judgments==()


def test_rule_guard_does_not_redefine_material_fact_contract():
    m=make_matching('material_facts')
    result=r.parse_matching(json.dumps({'o0':['retained',[field(m,'/candidate/entities_involved/0')]]}),m)
    assert result.structure_valid and not result.semantic_verified


def test_existing_source_uncertainty_still_blocks_affirmation():
    m=make_matching(uncertain=True)
    result=r.parse_matching(json.dumps({'o0':['retained',[field(m,'/candidate/name')]]}),m)
    assert result.structure_valid and not result.model_retention_satisfied


def test_forging_field_role_cannot_bypass_bound_matching_input():
    m=make_matching()
    scope=m.inventory_scope.record_scope
    fields=tuple(replace(f,path='/candidate/description') if f.path.startswith('/candidate/triggers/') else f
                 for f in scope.fields)
    base=replace(scope.source_scope.base_scope,fields=fields)
    changed=replace(scope,source_scope=replace(scope.source_scope,base_scope=base))
    forged=replace(m,inventory_scope=replace(m.inventory_scope,record_scope=changed))
    with pytest.raises(ValueError):
        r.parse_matching(json.dumps({'o0':['retained',[field(m,'/candidate/triggers/0')]]}),forged)
