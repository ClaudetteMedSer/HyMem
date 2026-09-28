"""Source-owned retention contract tests; scripted outcomes are not model evidence."""
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from benchmarks import digest_evidence_assessment as assessment
from benchmarks import digest_source_review as review
from tests.test_digest_source_review import plan, parse, response, rehash_plan, rehash_scope
from tests.test_digest_evidence_assessment import payload, template, Scripted


def retention(scope):
    return tuple(c for c in scope.checks if c.kind == "retention")


def test_every_canonical_source_gets_exact_facets_before_unchanged_grounding():
    prepared = plan()
    assert prepared.version == "digest-source-review-v2"
    for scope in prepared.requests:
        retained = retention(scope)
        expected = [(source.source_id, facet) for source in scope.canonical_sources
                    for facet in ("material_facts", "constraints", "ordering")]
        assert [(check.source_id, check.facet) for check in retained] == expected
        assert [check.check_id for check in retained] == [f"r{i}" for i in range(len(expected))]
        assert scope.checks[:len(expected)] == retained
        assert all(check.field_ids == tuple(field.field_id for field in scope.fields)
                   for check in retained)
        old_grounding = [c for c in scope.base_scope.checks if c.kind != "retention"]
        assert [asdict(c) for c in scope.checks[len(expected):]] == [
            {**asdict(c), "facet": None} for c in old_grounding]
        assert len({c.check_id for c in scope.checks}) == len(scope.checks)
        wire = json.loads(scope.request.user)
        assert wire["checks"] == [{**asdict(c), "field_ids": list(c.field_ids)} for c in scope.checks]
        assert all(c.source_id not in {s.source_id for s in scope.context_sources}
                   for c in retained)
        with pytest.raises(FrozenInstanceError):
            retained[0].facet = "invented"


@pytest.mark.parametrize("facet", review.RETENTION_FACETS)
@pytest.mark.parametrize("status,has_witness,no_defect", [
    ("retained", True, True), ("altered", True, False),
    ("omitted", False, False), ("not_applicable", False, True),
    ("uncertain", False, False), ("uncertain", True, False),
])
def test_retention_states_bind_exact_source_facet_and_candidate_witness(facet, status, has_witness, no_defect):
    scope = plan().requests[0]
    check = next(c for c in retention(scope) if c.facet == facet)
    reply = response(scope)
    ids = [scope.fields[-1].field_id] if has_witness else []
    reply[check.check_id] = [status, ids]
    outcome = parse(scope, reply)
    assert outcome.review_structure_valid and outcome.model_grounding_supported
    assert outcome.model_no_defect is no_defect
    assert outcome.model_retention_satisfied is no_defect
    judgment = next(j for j in outcome.judgments if j.check_id == check.check_id)
    assert (judgment.source_id, judgment.facet) == (check.source_id, facet)
    assert judgment.field_ids == check.field_ids
    assert judgment.witness_field_ids == tuple(ids)
    assert judgment.witness_fields == ((scope.fields[-1],) if has_witness else ())
    assert not judgment.primary_evidence and not judgment.context_evidence
    assert not outcome.semantic_verified and not outcome.publication_authorized


@pytest.mark.parametrize("entry", [
    None, True, "retained", {}, [], ["retained"], ["retained", [], []],
    ["retained", []], ["altered", []], ["omitted", ["f0"]], ["not_applicable", ["f0"]],
    ["supported", ["f0"]], ["unsupported", []], [True, []], ["uncertain", "f0"],
    ["retained", [True]], ["retained", [0]], ["retained", [None]], ["retained", [{}]],
    ["retained", ["f0", "f0"]], ["uncertain", ["f0", "f0"]],
    ["retained", ["f999"]], ["altered", ["f999"]], ["uncertain", ["f999"]],
    ["retained", ["s0"]], ["uncertain", ["s0"]],
])
def test_bad_retention_entry_invalidates_entire_scope_without_salvage(entry):
    scope = plan().requests[0]
    reply = response(scope)
    reply[retention(scope)[0].check_id] = entry
    outcome = parse(scope, reply)
    assert outcome.status == "malformed_review" and not outcome.judgments
    assert not outcome.model_no_defect
    assert not outcome.model_grounding_supported and not outcome.model_retention_satisfied


@pytest.mark.parametrize("which", ["canonical_sources", "context_sources", "prior_summary_sources"])
def test_no_evidence_plane_ids_are_retention_witnesses(which):
    scope = plan().requests[-1]
    reply = response(scope)
    for source in getattr(scope, which):
        reply[retention(scope)[0].check_id] = ["retained", [source.source_id]]
        assert parse(scope, reply).status == "malformed_review"


def test_status_keys_cannot_override_owned_source_or_facet():
    scope = plan().requests[-1]
    check = retention(scope)[0]
    for entry in ({"status": "retained", "source_id": check.source_id, "facet": check.facet,
                   "witnesses": [scope.fields[0].field_id]},
                  ["retained", [scope.fields[0].field_id], check.source_id, check.facet]):
        reply = response(scope)
        reply[check.check_id] = entry
        assert parse(scope, reply).status == "malformed_review"


def test_source_local_results_preserve_unchanged_source_despite_other_error():
    scope = plan().requests[-1]
    reply = response(scope)
    first, second = scope.canonical_sources
    for check in retention(scope):
        if check.source_id == second.source_id:
            reply[check.check_id] = ["altered", [scope.fields[0].field_id]]
    reply["c0"] = ["unsupported", [], []]
    outcome = parse(scope, reply)
    assert outcome.review_structure_valid and not outcome.model_no_defect
    assert all(j.verdict == "retained" for j in outcome.judgments
               if j.source_id == first.source_id)
    assert all(j.verdict == "altered" for j in outcome.judgments
               if j.source_id == second.source_id)
    # Code preserves distinct judgments; it cannot prove the model will localize them.
    assert not outcome.model_grounding_supported and not outcome.model_retention_satisfied


def test_grounding_failure_does_not_change_retention_view():
    scope = plan().requests[0]
    reply = response(scope)
    reply["c0"] = ["unsupported", [], []]
    outcome = parse(scope, reply)
    assert outcome.model_retention_satisfied
    assert not outcome.model_grounding_supported and not outcome.model_no_defect


def test_missing_prohibition_may_be_omitted_with_all_emitted_fields_grounded():
    scope = plan().requests[1]
    reply = response(scope)
    constraint = next(c for c in retention(scope) if c.facet == "constraints")
    reply[constraint.check_id] = ["omitted", []]
    outcome = parse(scope, reply)
    assert outcome.model_grounding_supported and not outcome.model_retention_satisfied
    assert not outcome.model_no_defect


def test_not_applicable_is_not_proof_and_witness_is_not_proof():
    value = payload()
    value["procedure_items"][0]["candidate"]["description"] = "Never release the vent."
    scope = plan(value).requests[1]
    reply = response(scope)
    for check in retention(scope):
        reply[check.check_id] = ["not_applicable", []]
    outcome = parse(scope, reply)
    assert outcome.review_structure_valid and outcome.model_no_defect
    assert outcome.model_retention_satisfied
    assert not outcome.semantic_verified and not outcome.publication_authorized
    assert not hasattr(outcome, "model_all_supported")


def test_whole_candidate_order_and_unicode_witness_bytes_remain_exact():
    value = payload()
    actions = ["Ω-17: close", "Café e\u0301: read", "Ω-17: close"]
    value["procedure_items"][0]["candidate"]["steps"] = [
        {"order": i + 1, "action": action, "tool": None} for i, action in enumerate(actions)]
    scope = plan(value).requests[1]
    steps = tuple(f for f in scope.fields if f.path.startswith("/candidate/steps/") and f.path.endswith("/action"))
    assert [f.text for f in steps] == actions
    reply = response(scope)
    check = next(c for c in retention(scope) if c.facet == "ordering")
    reply[check.check_id] = ["altered", [f.field_id for f in steps]]
    outcome = parse(scope, reply)
    judgment = next(j for j in outcome.judgments if j.check_id == check.check_id)
    assert judgment.witness_fields == steps
    assert all(f.start == 0 and f.end == len(f.text) for f in judgment.witness_fields)
    assert [step["action"] for step in json.loads(scope.request.user)["candidate"]["candidate"]["steps"]] == actions


def test_witness_cap_checked_on_prepare_bound_and_stricter_parse_bound():
    for scope in (plan(max_evidence_per_check=1).requests[0], plan().requests[0]):
        reply = response(scope)
        reply[retention(scope)[0].check_id] = ["retained", [f.field_id for f in scope.fields[:2]]]
        assert parse(scope, reply, max_evidence_per_check=1).status == "malformed_review"


def test_empty_candidate_still_has_all_source_facets_and_no_possible_positive_witness():
    value = payload()
    value["summary_item"]["candidate_summary"] = ""
    scope = plan(value).requests[-1]
    assert not scope.fields
    assert len(scope.checks) == 3 * len(scope.canonical_sources) == 6
    reply = {c.check_id: ["omitted", []] for c in scope.checks}
    outcome = parse(scope, reply)
    assert outcome.review_structure_valid and not outcome.model_grounding_supported
    assert not outcome.model_retention_satisfied and not outcome.model_no_defect
    reply[scope.checks[0].check_id] = ["retained", ["f0"]]
    assert parse(scope, reply).status == "malformed_review"


def test_complete_new_check_cap_is_enforced_in_preparation_and_standalone_validation():
    prepared = plan()
    new_max = max(len(s.checks) for s in prepared.requests)
    old_max = max(len(s.base_scope.checks) for s in prepared.requests)
    assert new_max > old_max
    assert plan(max_checks=new_max).max_checks == new_max
    with pytest.raises(ValueError, match="complete review exceeds check cap"):
        plan(max_checks=old_max)
    scope = prepared.requests[0]
    base = replace(scope.base_scope, max_checks=len(scope.base_scope.checks))
    base = replace(base, binding_sha256=assessment._sha(assessment._scope_body(base)))
    forged = rehash_scope(replace(scope, base_scope=base))
    with pytest.raises(ValueError, match="complete review exceeds check cap"):
        parse(forged)
    client = Scripted([])
    with pytest.raises(ValueError):
        review.execute_source_review(rehash_plan(replace(prepared, max_checks=old_max)), client)
    assert not client.calls


@pytest.mark.parametrize("which,value", [("facet", "material_facts"), ("source_id", "s999"),
                                        ("field_ids", []), ("check_id", "r999")])
def test_retention_wire_cannot_be_forged_even_with_rehashed_scope(which, value):
    scope = plan().requests[0]
    wire = json.loads(scope.request.user)
    # Select constraints so replacing facet with material_facts really changes it.
    wire["checks"][1][which] = value
    forged = rehash_scope(replace(scope, request=replace(scope.request, user=review._canonical(wire))))
    with pytest.raises(ValueError):
        parse(forged)


def test_shortest_complete_output_cap_and_actual_output_length_both_enforced():
    prepared = plan()
    minimum = max(len(review._canonical(review._minimum_response(s))) for s in prepared.requests)
    bounded = plan(max_output_chars=minimum)
    client = Scripted([review._canonical(review._minimum_response(s)) for s in bounded.requests])
    result = review.execute_source_review(bounded, client)
    assert result.complete and result.review_structure_valid and not result.model_no_defect
    assert len(client.calls) == len(prepared.requests)
    scope = max(prepared.requests, key=lambda s: len(review._canonical(review._minimum_response(s))))
    with pytest.raises(ValueError, match="cannot fit parsing output cap"):
        parse(scope, max_output_chars=minimum - 1)
    # A longer valid-shaped answer still cannot exceed its configured character cap.
    assert parse(scope, response(scope), max_output_chars=minimum).status == "malformed_review"


def test_prompt_explicitly_separates_applicability_locality_and_answer_retention():
    prompt = plan().requests[0].request.system
    for required in ("SOURCE-FIRST RETENTION", "answered recommendations", "prohibitions",
                     "mandatory sequencing", "incidental narrative order", "another source",
                     "full candidate structure", "missing applicable prohibition is omitted",
                     "not mechanically verified absence", "Witness IDs do not prove preservation",
                     "EVERY material item", "all applicable material items preserved",
                     "material temporal relationships", "Not_applicable requires no applicable"):
        assert required in prompt
