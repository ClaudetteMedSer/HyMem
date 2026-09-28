"""Development-control plumbing, never observed semantic performance."""
from collections import Counter
from copy import deepcopy
import json
import socket

import pytest

from benchmarks import digest_record_review as review
from tests.digest_record_review_fixtures import build_record_controls
from tests.test_digest_source_review_root import TEMPLATE, response


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("record fixtures are offline development controls")
    monkeypatch.setattr(socket.socket,"connect",denied)
    monkeypatch.setattr(socket,"create_connection",denied)


def test_controls_are_nine_pairs_with_explicit_separate_target_annotations():
    cases=build_record_controls()
    assert len(cases)==18 and len({c["id"] for c in cases})==18
    assert Counter(c["target"]["kind"] for c in cases)=={
        "actor_attribution":6,"identity":6,"quantified_scope":6}
    for pair in {c["pair"] for c in cases}:
        members=[c for c in cases if c["pair"]==pair]
        assert {c["expected"] for c in members}=={"supported","unsupported"}
        assert {c["variant"] for c in members}=={"faithful","defective"}
        assert all(c["rationale"] for c in members)
        assert all("expected_scope_no_defect" not in c for c in members)


@pytest.mark.parametrize("case",build_record_controls(),ids=lambda c:c["id"])
def test_control_target_resolves_once_and_scripted_status_is_preserved(case):
    prepared=review.prepare_record_review(case["payload"],TEMPLATE,max_calls=2)
    scope=next(s for s in prepared.requests if {"kind":s.kind,"index":s.index}==case["scope"])
    field=next(f for f in scope.fields if f.path==case["target"]["field_path"])
    checks=[c for c in scope.checks if c.kind==case["target"]["kind"] and c.field_ids==(field.field_id,)]
    assert len(checks)==1
    body=response(scope)
    if case["expected"]=="unsupported":
        body[checks[0].check_id]=["unsupported",[],[]]
    result=review.parse_record_review(json.dumps(body),scope)
    target=next(j for j in result.judgments if j.check_id==checks[0].check_id)
    assert result.review_structure_valid and target.verdict==case["expected"]
    assert not result.semantic_verified and not result.publication_authorized
    assert case["id"] not in scope.request.user and case["rationale"] not in scope.request.user
    assert len(prepared.requests)==2
    assert scope.request.max_tokens==TEMPLATE.max_tokens and scope.request.temperature==TEMPLATE.temperature
    changed=deepcopy(case)
    changed["expected"]="unsupported" if case["expected"]=="supported" else "supported"
    again=review.prepare_record_review(changed["payload"],TEMPLATE,max_calls=2)
    assert prepared==again  # Annotations cannot alter requests or binding hashes.


def test_fixture_builder_returns_fresh_unshared_payloads():
    a=build_record_controls()
    b=build_record_controls()
    a[0]["payload"]["source_catalog"][0]["role"]="system"
    assert b==build_record_controls()
    assert a[1]["payload"]["source_catalog"][0]["role"]=="user"


def changed_paths(a,b,path=""):
    if type(a) is dict and type(b) is dict and a.keys()==b.keys():
        return set().union(*(changed_paths(a[k],b[k],path+"/"+k) for k in a))
    if type(a) is list and type(b) is list and len(a)==len(b):
        return set().union(*(changed_paths(x,y,path+"/"+str(i)) for i,(x,y) in enumerate(zip(a,b))))
    return set() if a==b else {path}


def test_every_pair_changes_only_its_preregisterable_mechanism():
    expected={
        "cross-message-owner":{"/source_catalog/0/role","/source_catalog/0/interpretation_only_context/role"},
        "quoted-first-person":{"/items/0/candidate_body"},
        "same-message-continuation":{"/items/0/candidate_body"},
        "canonical-name-versus-peer":{"/source_catalog/0/visible_content","/source_catalog/0/end"},
        "canonical-name-versus-workspace":{"/source_catalog/0/visible_content","/source_catalog/0/end"},
        "name-owned-by-another-person":{"/items/0/candidate_body"},
        "closed-three-member-set":{"/items/0/candidate_title"},
        "nonexhaustive-sample":{"/items/0/candidate_title"},
        "universal-domain-restriction":{"/items/0/candidate_title"},
    }
    cases=build_record_controls()
    for name,wanted in expected.items():
        a,b=[c for c in cases if c["pair"]==name]
        assert changed_paths(a["payload"],b["payload"])==wanted


def test_wire_roles_and_same_message_coordinates_survive_new_controls():
    cases={c["id"]:c for c in build_record_controls()}
    for variant,roles in (("faithful",("assistant","user")),("defective",("user","assistant"))):
        case=cases["record-cross-message-owner-"+variant]
        scope=review.prepare_record_review(case["payload"],TEMPLATE,max_calls=2).requests[0]
        record=json.loads(scope.request.user)["source_records"][0]
        assert (record["current_message"]["role"],record["boundary_context"]["message"]["role"])==roles
        assert record["current_message"]["message_id"]!=record["boundary_context"]["message"]["message_id"]
    case=cases["record-same-message-continuation-faithful"]
    scope=review.prepare_record_review(case["payload"],TEMPLATE,max_calls=2).requests[0]
    record=json.loads(scope.request.user)["source_records"][0]
    assert record["canonical_source"]["start"]==record["boundary_context"]["end"]
    assert record["current_message"]==record["boundary_context"]["message"]
