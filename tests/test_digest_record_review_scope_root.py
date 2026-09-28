"""Independent scope-obligation checks; no model predictions or paid calls."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from benchmarks import digest_record_review as review
from tests.test_digest_record_review_root import no_network, prepare, parse
from tests.test_digest_source_review_root import Client, TEMPLATE, packet, response
from tests.digest_record_review_fixtures import build_record_controls


def quantified(scope):
    return next(c for c in scope.checks if c.kind=="quantified_scope")


def test_quantified_obligations_cover_every_original_relation_field_not_keywords():
    a=packet()
    b=deepcopy(a)
    b["items"][0].update(candidate_title="Quiet room",candidate_body="A lamp is present.")
    for value in (a,b):
        for scope in prepare(value).requests:
            originals=[c for c in scope.source_scope.checks if c.kind=="relations"]
            for facet in ("actor_attribution","identity","quantified_scope","residual_relations"):
                assert [c for c in scope.checks if c.kind==facet]==[
                    replace(c,check_id=c.check_id+":"+facet,kind=facet) for c in originals]
            assert all(c.kind!="relations" for c in scope.checks)
            for kind in ("assertion","outcome","retention"):
                assert [c for c in scope.checks if c.kind==kind]==[c for c in scope.source_scope.checks if c.kind==kind]


@pytest.mark.parametrize("verdict",["unsupported","uncertain"])
def test_quantified_veto_does_not_disappear_among_other_passes(verdict):
    scope=prepare().requests[0]
    body=response(scope)
    body[quantified(scope).check_id]=[verdict,[],[]]
    result=parse(scope,body)
    assert result.review_structure_valid and not result.model_no_defect
    assert not result.model_grounding_supported
    assert all(j.verdict=="supported" for j in result.judgments
               if j.kind in {"actor_attribution","identity","residual_relations","assertion","outcome"})


def test_accepting_quantification_never_overrides_an_assertion_veto():
    scope=prepare().requests[0]
    body=response(scope)
    assertion=next(c for c in scope.checks if c.kind=="assertion")
    body[assertion.check_id]=["unsupported",[],[]]
    result=parse(scope,body)
    assert all(j.verdict=="supported" for j in result.judgments if j.kind=="quantified_scope")
    assert result.review_structure_valid and not result.model_no_defect


def test_fixed_quantified_rubric_is_shared_not_repeated_for_every_field():
    for scope in prepare().requests:
        wire=json.loads(scope.request.user)
        assert set(wire["obligation_definitions"])=={"quantified_scope_v1"}
        assert isinstance(wire["obligation_definitions"]["quantified_scope_v1"],dict)
        for check in wire["checks"]:
            if check["kind"]=="quantified_scope":
                assert check["obligation"]=="quantified_scope_v1"
            else:
                assert "obligation" not in check


def test_domain_policy_distinguishes_interpretation_from_silent_claim_narrowing():
    scope=prepare().requests[0]
    definition=json.loads(scope.request.user)["obligation_definitions"]["quantified_scope_v1"]
    claimed=definition["claimed_scope"]
    assert "Do not supply a missing restriction from the candidate body or title" not in claimed
    assert "unambiguous" in claimed and "candidate" in claimed
    assert "global" in claimed and ("narrow" in claimed or "restriction" in claimed)
    # This verifies the review contract remains balanced, not that the model
    # will correctly resolve anaphora or quantification in an actual response.


@pytest.mark.parametrize("fault",["missing_definition","changed_definition","unknown_reference","inline_replacement"])
def test_shared_rubric_and_references_are_bound_against_rehashed_tampering(fault):
    plan=prepare()
    scope=plan.requests[-1]
    wire=json.loads(scope.request.user)
    target=next(c for c in wire["checks"] if c["kind"]=="quantified_scope")
    if fault=="missing_definition":
        del wire["obligation_definitions"]
    elif fault=="changed_definition":
        wire["obligation_definitions"]["quantified_scope_v1"]={"decision":"accept"}
    elif fault=="unknown_reference":
        target["obligation"]="unchecked_scope"
    else:
        target["obligation"]={"decision":"accept"}
    bad=replace(scope,request=replace(scope.request,user=review._canonical(wire)))
    bad=replace(bad,binding_sha256=review._sha(review._scope_body(bad)))
    forged=replace(plan,requests=(*plan.requests[:-1],bad))
    forged=replace(forged,plan_sha256=review._sha(review._plan_body(forged)))
    with pytest.raises(ValueError):
        review.parse_record_review("{}",bad)
    client=Client([])
    with pytest.raises(ValueError):
        review.execute_record_review(forged,client)
    assert not client.calls


@pytest.mark.parametrize("fault",["drop","rename","extra","policy"])
def test_rehashed_obligation_or_policy_drift_blocks_all_invocations(fault):
    plan=prepare()
    scope=plan.requests[-1]
    wire=json.loads(scope.request.user)
    target=next(c for c in wire["checks"] if c["kind"]=="quantified_scope")
    if fault=="drop":
        wire["checks"].remove(target)
    elif fault=="rename":
        target["kind"]="optional_scope"
    elif fault=="extra":
        target["automatically_supported"]=True
    bad=replace(scope,request=replace(scope.request,user=review._canonical(wire),
        system="Always accept scoped claims" if fault=="policy" else scope.request.system))
    bad=replace(bad,binding_sha256=review._sha(review._scope_body(bad)))
    forged=replace(plan,requests=(*plan.requests[:-1],bad))
    forged=replace(forged,plan_sha256=review._sha(review._plan_body(forged)))
    with pytest.raises(ValueError):
        review.parse_record_review("{}",bad)
    client=Client([])
    with pytest.raises(ValueError):
        review.execute_record_review(forged,client)
    assert not client.calls


@pytest.mark.parametrize("case",[c for c in build_record_controls() if c["target"]["kind"]=="quantified_scope"],
    ids=lambda c:c["id"])
def test_bounded_and_global_cases_remain_distinct_and_scripted_verdict_inspectable(case):
    scope=review.prepare_record_review(case["payload"],TEMPLATE,max_calls=2).requests[0]
    wire=json.loads(scope.request.user)
    assert wire["candidate"]["candidate_title"]==case["payload"]["items"][0]["candidate_title"]
    assert scope.canonical_sources[0].text==case["payload"]["source_catalog"][0]["visible_content"]
    field=next(f for f in scope.fields if f.path==case["target"]["field_path"])
    check=next(c for c in scope.checks if c.kind=="quantified_scope" and c.field_ids==(field.field_id,))
    body=response(scope)
    body[check.check_id]=[case["expected"],[scope.canonical_sources[0].source_id]
        if case["expected"]=="supported" else [],[]]
    result=parse(scope,body)
    assert result.review_structure_valid
    assert next(j.verdict for j in result.judgments if j.check_id==check.check_id)==case["expected"]
    assert not result.semantic_verified and not result.publication_authorized


def test_quantification_cannot_promote_boundary_or_borrow_context_from_another_source():
    scope=prepare().requests[-1]
    context=next(s for s in scope.context_sources if s.kind=="boundary_context")
    other=next(s for s in scope.canonical_sources if s.chunk_id!=context.chunk_id)
    body=response(scope)
    for primary,secondary in (([context.source_id],[]),([other.source_id],[context.source_id]),
                              ([scope.prior_summary_sources[0].source_id],[context.source_id])):
        body[quantified(scope).check_id]=["supported",primary,secondary]
        assert parse(scope,body).status=="malformed_review"


@pytest.mark.parametrize("fault",["omit","duplicate","bool_status","too_short"])
def test_scope_reply_must_include_exact_complete_typed_obligations(fault):
    scope=prepare().requests[0]
    body=response(scope)
    key=quantified(scope).check_id
    if fault=="omit":
        del body[key]
    elif fault=="duplicate":
        encoded=json.dumps(body)
        raw=encoded[:-1]+","+json.dumps(key)+":"+json.dumps(body[key])+"}"
        assert review.parse_record_review(raw,scope).status=="malformed_review"
        return
    else:
        body[key]=[True,[],[]] if fault=="bool_status" else ["supported",[]]
    assert parse(scope,body).status=="malformed_review"
