"""Offline v3 instruction/contract checks, never measured model accuracy.

All replies below are scripted. A structural pass deliberately remains weaker
than correct materiality, attribution, ordering, paraphrase, or completeness.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from benchmarks import digest_retention_inventory as review
from hymem.extraction.llm import LLMRequest
from tests.digest_retention_inventory_v2_fixtures import build_cases


CASES = build_cases()


def prepared(case, *, temperature=0.0, max_tokens=8192):
    return review.prepare_retention_inventory(case["payload"],
        LLMRequest("template system", "template user", temperature=temperature,
                   max_tokens=max_tokens), max_calls=64)


def selected(case):
    return next(scope for scope in prepared(case).requests
                if {"kind": scope.kind, "index": scope.index} == case["target_scope"])


def scripted_inventory(scope, text="An explicitly stated material requirement.", *, uncertain=False):
    groups = {}
    for unit in scope.units:
        groups.setdefault(unit.source_id, []).append(unit.unit_id)
    value = {"obligations": [{"facet": "constraints", "unit_ids": refs, "text": text}
                             for refs in groups.values()],
             "no_material_unit_ids": [],
             "uncertain_unit_ids": [scope.units[0].unit_id] if uncertain else []}
    outcome = review.parse_inventory(json.dumps(value), scope)
    assert outcome.status == "valid_inventory"
    return outcome.inventory


@pytest.mark.parametrize("clauses", [
    ("Describe complete propositions, events or rules", "not merely the presence of words",
     "Reporting or quotation may itself be material"),
    ("including a justified same-message prefix", "Do not import a separate event from context",
     "even when it shares the message", "mark the unit uncertain instead of presenting a fragment"),
    ("Incidental scene-setting and metacommentary", "background label does not exempt a genuine rule",
     "A mixed unit may reference only its material obligations", "mark the unresolved unit uncertain"),
    ("both actual event endpoints", "shared prerequisite loses the original relation",
     "Obligation list position alone", "Do not invent order"),
])
def test_inventory_instructions_address_meaning_without_changing_authority(clauses):
    prompt = review._inventory_system()
    assert all(clause in prompt for clause in clauses)
    assert "No candidate is supplied" in prompt
    assert "interpretation-only, never independent new facts" in prompt
    assert "Never truncate or silently drop obligations" in prompt


@pytest.mark.parametrize("clauses", [
    ("semantically equivalent affirmative instruction", "change in grammatical form alone",
     "Do not forgive a missing qualifier", "no single field must repeat the whole obligation verbatim"),
    ("same actor, object, polarity", "modality, duration, scope, prerequisite",
     "actual event endpoints", "not another message's speaker"),
    ("do not relabel an obligation incidental", "unsupported or nonmaterial requirement",
     "use uncertain and leave it for review", "not-applicable to fit the candidate"),
    ("triggers are retrieval words or phrases", "NOT execution prerequisites",
     "NOT execution rules", "an explicit condition in an imperative name can contribute"),
])
def test_matching_instructions_preserve_strict_gate_and_correct_field_roles(clauses):
    prompt = review._matching_system()
    assert all(clause in prompt for clause in clauses)
    assert "any uncertainty blocks an affirmative overall retention result" in prompt
    assert "exactly one key per supplied obligation_id" in prompt
    assert "all mandatory qualifiers" in prompt


def test_semantic_guidance_is_versioned_not_fixture_specific():
    assert review.VERSION == "digest-retention-inventory-v4"
    prompts = review._inventory_system() + review._matching_system()
    for case in CASES:
        assert case["id"] not in prompts
        assert case["pair"] not in prompts
        for record in case["payload"]["source_catalog"]:
            assert record["chunk_id"] not in prompts
            assert record["visible_content"] not in prompts


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_instructions_do_not_filter_rewrite_or_rescope_existing_inputs(case):
    before = deepcopy(case)
    plan = prepared(case, temperature=0.375, max_tokens=3072)
    assert plan.reserved_calls == 2 * len(plan.record_plan.requests)
    assert [(scope.kind, scope.index) for scope in plan.requests] == [
        (scope.kind, scope.index) for scope in plan.record_plan.requests]
    for scope in plan.requests:
        body = json.loads(scope.request.user)
        assert set(body) == {"schema", "scope", "source_records", "location_units"}
        assert body["schema"] == review.VERSION
        assert body["location_units"] == [asdict(unit) for unit in scope.units]
        assert len(body["source_records"]) == len(scope.record_scope.source_records)
        for record, original in zip(body["source_records"], scope.record_scope.source_records):
            assert record == json.loads(json.dumps(asdict(original)))
        for source in scope.record_scope.canonical_sources:
            units = [unit for unit in scope.units if unit.source_id == source.source_id]
            assert "".join(unit.text for unit in units) == source.text
            assert units[0].start == source.start
            assert units[-1].end == source.end
        assert scope.request.temperature == 0.375 and scope.request.max_tokens == 3072
        assert scope.request.response_format == "json"
    assert case == before


def test_candidate_only_changes_still_cannot_condition_source_materiality():
    for left, right in zip(CASES[::2], CASES[1::2]):
        assert left["pair"] == right["pair"]
        a, b = selected(left), selected(right)
        assert a.request == b.request and a.units == b.units
        assert a.source_binding_sha256 == b.source_binding_sha256
        assert a.binding_sha256 != b.binding_sha256


def test_mixed_unit_needs_material_obligation_not_one_obligation_per_sentence():
    case = next(case for case in CASES if case["pair"] == "incidental-detail-omission")
    scope = selected(case)
    assert len(scope.units) == 1
    target = case["expected_source_obligations"][0]
    inventory = scripted_inventory(scope, target["description"])
    assert len(inventory.obligations) == 1
    assert inventory.no_material_unit_ids == inventory.uncertain_unit_ids == ()
    # This only demonstrates that the existing schema can express the intended
    # inventory. The stub does not establish that a model selects it correctly.
    assert not inventory.semantic_verified and not inventory.publication_authorized


@pytest.mark.parametrize("text", [
    "The source says that a notice was posted.",
    "The source contains a noun fragment.",
    "An incidental observation is incorrectly treated as mandatory here.",
    "Two actions have a prerequisite but their actual order is unresolved.",
])
def test_no_text_heuristic_turns_structural_validation_into_semantic_claim(text):
    scope = selected(CASES[0])
    inventory = scripted_inventory(scope, text)
    matching = review.prepare_matching(scope, inventory)
    field = scope.record_scope.fields[0].field_id
    scripted = {item.obligation_id: ["retained", [field]] for item in inventory.obligations}
    outcome = review.parse_matching(json.dumps(scripted), matching)
    assert outcome.structure_valid and outcome.model_retention_satisfied
    assert not outcome.semantic_verified and not outcome.publication_authorized
    # Even a semantically bad scripted inventory remains structurally valid;
    # these prompt changes add no keyword classifier or automatic proof.


@pytest.mark.parametrize("status", ["not_applicable", "optional", "incidental", "equivalent", "supported"])
def test_new_guidance_adds_no_obligation_exemption_status(status):
    scope = selected(CASES[0])
    matching = review.prepare_matching(scope, scripted_inventory(scope))
    values = {item.obligation_id: [status, []] for item in matching.inventory.obligations}
    result = review.parse_matching(json.dumps(values), matching)
    assert result.status == "malformed_matching"
    assert result.judgments == () and not result.model_retention_satisfied


def test_unresolved_materiality_cannot_be_waived_by_retained_known_content():
    scope = selected(CASES[0])
    matching = review.prepare_matching(scope, scripted_inventory(scope, uncertain=True))
    field = scope.record_scope.fields[0].field_id
    response = {item.obligation_id: ["retained", [field]] for item in matching.inventory.obligations}
    outcome = review.parse_matching(json.dumps(response), matching)
    assert outcome.structure_valid and not outcome.model_retention_satisfied


def test_matching_cannot_rewrite_frozen_inventory_to_launder_away_requirement():
    scope = selected(CASES[0])
    frozen = scripted_inventory(scope)
    changed = replace(frozen, obligations=(replace(frozen.obligations[0],
                                                   text="Now declared irrelevant."),))
    with pytest.raises(ValueError, match="frozen inventory differs"):
        review.prepare_matching(scope, changed)


def test_prompt_contract_drift_is_not_silently_accepted_as_same_bound_scope():
    scope = selected(CASES[0])
    changed = replace(scope, request=replace(scope.request,
                                            system=scope.request.system + " Changed guidance."))
    with pytest.raises(ValueError, match="inventory scope binding mismatch"):
        review.parse_inventory("{}", changed)
