"""Offline field-role and fixture-contract tests, not model-accuracy evidence.

Scripted matching replies exercise parsing only. v4 rejects procedure-rule
witnesses drawn solely from retrieval metadata; other semantic errors remain.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from benchmarks import digest_retention_inventory as review
from hymem.extraction.llm import LLMRequest
from hymem.extraction.prompts import (
    PROCEDURE_SYSTEM, SESSION_DIGEST_GRANULAR_SYSTEM, SESSION_DIGEST_SYSTEM,
)
from tests.digest_retention_inventory_fixtures import build_cases as old_cases
from tests.digest_retention_inventory_v2_fixtures import (
    FIXTURE_VERSION, TRIGGER_ONLY_PAIR, build_cases,
)


CASES = build_cases()
PAIRS = tuple(dict.fromkeys(case["pair"] for case in CASES))
ORIGINAL_CASES = {case["id"]: case for case in old_cases()}


def _scope(case):
    plan = review.prepare_retention_inventory(case["payload"],
        LLMRequest("", "", max_tokens=8192, temperature=0.0), max_calls=64)
    return next(scope for scope in plan.requests
                if {"kind": scope.kind, "index": scope.index} == case["target_scope"])


def _inventory(scope):
    # Simulated source inventory, never a semantic judgment or external gold.
    entries = []
    for record in scope.record_scope.source_records:
        source = record.canonical_source
        if source is not None:
            entries.append({"facet": "constraints", "unit_ids": [unit.unit_id
                for unit in scope.units if unit.source_id == source.source_id],
                "text": source.text})
    result = review.parse_inventory(json.dumps({"obligations": entries,
        "no_material_unit_ids": [], "uncertain_unit_ids": []}), scope)
    assert result.status == "valid_inventory"
    return result.inventory


def _diff(left, right, path=""):
    if type(left) is not type(right):
        return {path}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {path}
        return set().union(*(_diff(left[key], right[key], f"{path}/{key}") for key in left))
    if isinstance(left, list):
        if len(left) != len(right):
            return {path}
        return set().union(*(_diff(a, b, f"{path}/{i}") for i, (a, b) in enumerate(zip(left, right))))
    return set() if left == right else {path}


def _pair(name):
    return [case for case in CASES if case["pair"] == name]


def _candidate(case):
    return case["payload"]["procedure_items"][0]["candidate"]


def test_current_version_preserves_field_roles_after_semantic_guidance_change():
    assert review.VERSION == "digest-retention-inventory-v4"
    # v2 changed field roles only. v3 intentionally refines the inventory
    # instructions; claiming the old inventory-prompt hash is still current
    # would erase that experimental distinction.
    assert hashlib.sha256(review._inventory_system().encode()).hexdigest() == (
        "b9d7c47fae67563669fc8aa71a068c0b0cb436175ba2e252d651df0b8c3866f3")
    assert "Describe complete propositions, events or rules" in review._inventory_system()
    prompt = review._matching_system()
    for clause in ("name identifies the procedure", "description and the ordered steps assert",
                   "triggers are retrieval words or phrases", "NOT execution prerequisites",
                   "entities_involved lists", "NOT execution rules", "procedure name",
                   "genuine paraphrases", "exact wording is not required"):
        assert clause in prompt
    assert "all mandatory qualifiers" in prompt and "any uncertainty blocks" in prompt


def test_procedure_name_is_not_categorically_excluded_from_asserted_conditions():
    prompt = review._matching_system()
    assert "an explicit condition in an imperative name can contribute" in prompt
    assert "trigger or entity list does not establish" in prompt
    assert "trigger, entity list or procedure name does not establish" not in prompt


@pytest.mark.parametrize("prompt", [PROCEDURE_SYSTEM, SESSION_DIGEST_SYSTEM, SESSION_DIGEST_GRANULAR_SYSTEM])
def test_field_roles_match_existing_producer_contract(prompt):
    assert "triggers (list of strings): Words/phrases someone might use to ask about this procedure" in prompt
    assert "entities_involved (list of strings): Named tools, services, platforms, files involved" in prompt
    assert "description (string): 1 sentence describing what the procedure accomplishes" in prompt
    assert "order (integer): Step number starting at 1" in prompt


def test_v2_has_twelve_revised_cases_and_one_role_control_pair():
    assert len(CASES) == len({case["id"] for case in CASES}) == 14
    assert len(PAIRS) == 7 and PAIRS[-1] == TRIGGER_ONLY_PAIR
    assert not {case["id"] for case in CASES} & set(ORIGINAL_CASES)
    assert all(case["fixture_version"] == FIXTURE_VERSION
               and case["label_scope"] == "targeted_obligations_only"
               and not case["whole_scope_correctness_claim"] for case in CASES)
    assert Counter(match["expected"] for case in CASES for match in case["candidate_matches"]) == {
        "retained": 14, "omitted": 4, "altered": 2}
    assert [case["variant"] for case in _pair("incidental-detail-omission")] == [
        "faithful_detailed", "faithful_concise"]


@pytest.mark.parametrize("case", CASES[:12], ids=lambda c: c["id"])
def test_revisions_preserve_exact_old_sources_and_targeted_source_gold(case):
    old = ORIGINAL_CASES[case["revision_of"]]
    assert case["payload"]["source_catalog"] == old["payload"]["source_catalog"]
    assert case["expected_source_obligations"] == old["expected_source_obligations"]
    assert case["optional_details"] == old["optional_details"]
    assert case["target_scope"] == old["target_scope"]
    assert case["unit_chars"] == old["unit_chars"] == 256
    assert case["payload"]["summary_item"] == old["payload"]["summary_item"]
    if case["target_scope"]["kind"] != "procedure":
        assert case["payload"] == old["payload"]
    else:
        assert case["payload"]["procedure_items"][0]["cited_source_ids"] == (
            old["payload"]["procedure_items"][0]["cited_source_ids"])


def test_building_v2_does_not_mutate_or_alias_old_fixtures():
    before = old_cases()
    first, second = build_cases(), build_cases()
    first[0]["payload"]["source_catalog"][0]["visible_content"] = "mutated locally"
    first[0]["candidate_matches"][0]["expected"] = "mutated locally"
    assert old_cases() == before and build_cases() == second == CASES
    assert first[1]["payload"]["source_catalog"][0]["visible_content"] != "mutated locally"
    encoded = json.dumps(old_cases(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(encoded.encode()).hexdigest() == (
        "1cea64408a9b60fb08d0f3a9bafb5e4aafea395db86d8fd0b13b1e810fd08968")


@pytest.mark.parametrize("name", PAIRS)
def test_each_pair_changes_only_declared_candidate_assertion_paths(name):
    left, right = _pair(name)
    assert left["changed_payload_paths"] == right["changed_payload_paths"]
    assert _diff(left["payload"], right["payload"]) == set(left["changed_payload_paths"])
    assert left["payload"]["source_catalog"] == right["payload"]["source_catalog"]
    assert left["expected_source_obligations"] == right["expected_source_obligations"]
    if left["target_scope"]["kind"] == "procedure":
        assert _candidate(left)["triggers"] == _candidate(right)["triggers"]
        assert _candidate(left)["entities_involved"] == _candidate(right)["entities_involved"]
        assert all("/triggers" not in path and "/entities_involved" not in path
                   for path in left["changed_payload_paths"])


@pytest.mark.parametrize("case", [case for case in CASES if case["target_scope"]["kind"] == "procedure"],
                         ids=lambda c: c["id"])
def test_procedure_descriptions_and_order_are_schema_correct_and_gold_uses_assertions(case):
    candidate = _candidate(case)
    assert type(candidate["description"]) is str and candidate["description"].strip()
    assert [step["order"] for step in candidate["steps"]] == list(range(1, len(candidate["steps"]) + 1))
    assert all(type(step["order"]) is int and type(step["action"]) is str and step["action"].strip()
               and step["tool"] is None for step in candidate["steps"])
    assert candidate["triggers"] and all(type(value) is str and value.strip() for value in candidate["triggers"])
    assert candidate["entities_involved"] == []
    for match in case["candidate_matches"]:
        assert all(path == "/candidate/description" or path.startswith("/candidate/steps/")
                   for path in match["candidate_field_paths"])


@pytest.mark.parametrize("name", PAIRS)
def test_pair_inventory_requests_remain_byte_identical_and_candidate_blind(name):
    left, right = map(_scope, _pair(name))
    assert asdict(left.request) == asdict(right.request)
    assert left.source_binding_sha256 == right.source_binding_sha256
    assert left.binding_sha256 != right.binding_sha256
    assert left.units == right.units
    body = json.loads(left.request.user)
    assert not {"candidate", "fields", "prior_summary_sources", "candidate_matches",
                "expected_source_obligations", "fixture_version"} & set(body)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_gold_labels_never_enter_either_request(case):
    scope = _scope(case)
    changed = deepcopy(case)
    for key in set(changed) - {"payload", "target_scope"}:
        changed[key] = "GOLD-MUST-NOT-BECOME-A-REQUEST"
    assert _scope(changed) == scope
    matching = review.prepare_matching(scope, _inventory(scope))
    for request in (scope.request, matching.request):
        assert "GOLD-MUST-NOT-BECOME-A-REQUEST" not in request.user
        assert case["id"] not in request.user and case["pair"] not in request.user
        assert not {"candidate_matches", "expected_source_obligations", "fixture_version",
                    "revision_of", "whole_scope_correctness_claim"} & set(json.loads(request.user))
    fields = {field.path for field in scope.record_scope.fields}
    for match in case["candidate_matches"]:
        assert set(match["candidate_field_paths"]) <= fields


def test_paraphrase_and_opposite_condition_are_in_action_not_retrieval_metadata():
    faithful, defective = map(_candidate, _pair("paraphrase-versus-altered-condition"))
    assert faithful["steps"][0]["action"].replace("nonreceipt", "receipt") == defective["steps"][0]["action"]
    assert faithful["triggers"] == defective["triggers"] == ["replacement parcel", "dispatch a replacement"]
    assert faithful["description"] == defective["description"] == "Dispatch a replacement parcel."
    assert "nonreceipt" not in _pair("paraphrase-versus-altered-condition")[0]["payload"]["source_catalog"][0]["visible_content"]


def test_trigger_only_negative_label_is_external_and_does_not_redefine_triggers():
    faithful, defective = _pair(TRIGGER_ONLY_PAIR)
    assert _candidate(faithful)["triggers"] == _candidate(defective)["triggers"] == [
        "opening the supply cabinet after coordinator confirmation of an empty room"]
    assert "only after the coordinator confirms" in _candidate(faithful)["steps"][0]["action"]
    assert _candidate(defective)["steps"][0]["action"] == "Open the supply cabinet"
    assert _candidate(defective)["description"] == "Collect the labels from the supply cabinet."
    assert faithful["candidate_matches"][0]["expected"] == "retained"
    assert defective["candidate_matches"][0]["expected"] == "omitted"
    assert defective["candidate_matches"][0]["candidate_field_paths"] == []


def test_scripted_negative_verdict_exercises_structure_not_model_accuracy():
    scope = _scope(_pair(TRIGGER_ONLY_PAIR)[1])
    matching = review.prepare_matching(scope, _inventory(scope))
    result = review.parse_matching(json.dumps({item.obligation_id: ["omitted", []]
        for item in matching.inventory.obligations}), matching)
    assert result.structure_valid and not result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized


def test_false_trigger_only_rule_witness_is_rejected_without_semantic_authorization():
    scope = _scope(_pair(TRIGGER_ONLY_PAIR)[1])
    matching = review.prepare_matching(scope, _inventory(scope))
    trigger = next(field.field_id for field in scope.record_scope.fields if field.path == "/candidate/triggers/0")
    result = review.parse_matching(json.dumps({item.obligation_id: ["retained", [trigger]]
        for item in matching.inventory.obligations}), matching)
    # Field authority is deterministic; this does not establish model accuracy
    # or prove that a witness from an assertion field is semantically sufficient.
    assert not result.structure_valid and not result.model_retention_satisfied
    assert result.judgments == ()
    assert not result.semantic_verified and not result.publication_authorized
