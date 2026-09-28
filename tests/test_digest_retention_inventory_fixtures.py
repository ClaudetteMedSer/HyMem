"""Review fixture structure and scope authority, never score model accuracy.

Uses the established v9 source-review projection, not the new inventory API.
Targeted labels are external annotations; no scripted answer is passed off as
an end-to-end semantic success or a complete source inventory.
"""
from collections import Counter
from copy import deepcopy
import json

import pytest

from benchmarks import digest_source_review as review
from hymem.extraction.llm import LLMRequest
from tests.digest_retention_inventory_fixtures import build_cases


CASES = build_cases()
PAIRS = (
    "same-unit-two-constraints", "cross-unit-prerequisite",
    "paraphrase-versus-altered-condition", "explicit-prohibition",
    "boundary-message-attribution", "incidental-detail-omission",
)


def _pair(name):
    return [case for case in CASES if case["pair"] == name]


def _scope(case):
    plan = review.prepare_source_review(
        case["payload"], LLMRequest("", "", max_tokens=3072), max_calls=3)
    assert review._preflight(plan) == plan
    return next(scope for scope in plan.requests
                if {"kind": scope.kind, "index": scope.index} == case["target_scope"])


def _diff(left, right, path=""):
    if type(left) is not type(right):
        return {path}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {path}
        return set().union(*(_diff(left[key], right[key], f"{path}/{key}")
                             for key in left))
    if isinstance(left, list):
        if len(left) != len(right):
            return {path}
        return set().union(*(_diff(a, b, f"{path}/{i}")
                             for i, (a, b) in enumerate(zip(left, right))))
    return set() if left == right else {path}


def _mutable_ids(value):
    if isinstance(value, dict):
        return {id(value)}.union(*(_mutable_ids(item) for item in value.values()))
    if isinstance(value, list):
        return {id(value)}.union(*(_mutable_ids(item) for item in value))
    return set()


def _unit_indexes(case, obligation):
    record = next(record for record in case["payload"]["source_catalog"]
                  if record["chunk_id"] == obligation["chunk_id"])
    start = obligation["canonical_start"] - record["start"]
    stop = obligation["canonical_end"] - record["start"]
    return tuple(range(start // case["unit_chars"], (stop - 1) // case["unit_chars"] + 1))


def test_six_development_pairs_do_not_mislabel_the_two_faithful_incidental_variants():
    assert len(CASES) == len({case["id"] for case in CASES}) == 12
    assert tuple(dict.fromkeys(case["pair"] for case in CASES)) == PAIRS
    assert Counter(case["variant"] for case in CASES) == {
        "faithful": 5, "defective": 5, "faithful_detailed": 1, "faithful_concise": 1}
    assert sum(len(case["candidate_matches"]) for case in CASES) == 18
    assert Counter(label["expected"] for case in CASES
                   for label in case["candidate_matches"]) == {
        "retained": 13, "omitted": 3, "altered": 2}
    assert all(case["label_scope"] == "targeted_obligations_only"
               and case["whole_scope_correctness_claim"] is False for case in CASES)


def test_fixture_generation_is_deterministic_and_has_no_shared_mutable_gold_or_payload():
    assert build_cases() == CASES
    seen = set()
    for case in CASES + build_cases():
        identifiers = _mutable_ids(case)
        assert not seen.intersection(identifiers)
        seen.update(identifiers)


@pytest.mark.parametrize("name", PAIRS)
def test_pairs_change_only_declared_candidate_paths_with_identical_source_obligations(name):
    left, right = _pair(name)
    assert left["target_scope"] == right["target_scope"]
    assert left["change"] == right["change"]
    assert left["changed_payload_paths"] == right["changed_payload_paths"]
    assert _diff(left["payload"], right["payload"]) == set(left["changed_payload_paths"])
    assert all(path.startswith(("/items/", "/procedure_items/", "/summary_item/candidate_"))
               for path in left["changed_payload_paths"])
    assert left["payload"]["source_catalog"] == right["payload"]["source_catalog"]
    assert left["expected_source_obligations"] == right["expected_source_obligations"]
    assert left["optional_details"] == right["optional_details"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_v9_projection_preserves_full_source_coordinates_and_only_exact_witness_paths(case):
    original = deepcopy(case)
    scope = _scope(case)
    records = {record["chunk_id"]: record for record in case["payload"]["source_catalog"]}
    for canonical in scope.canonical_sources:
        record = records[canonical.chunk_id]
        assert (canonical.start, canonical.end, canonical.text) == (
            record["start"], record["end"], record["visible_content"])
        assert record["end"] - record["start"] == len(record["visible_content"])
        assert canonical.allowed_use == "support"
    obligations = {item["id"]: item for item in case["expected_source_obligations"]}
    assert len(obligations) == len(case["expected_source_obligations"])
    assert Counter(item["obligation_id"] for item in case["candidate_matches"]) == {
        identifier: 1 for identifier in obligations}
    fields = {field.path for field in scope.fields}
    for item in case["candidate_matches"]:
        assert set(item["candidate_field_paths"]) <= fields
        assert bool(item["candidate_field_paths"]) == (item["expected"] != "omitted")
    for obligation in obligations.values():
        record = records[obligation["chunk_id"]]
        start = obligation["canonical_start"] - record["start"]
        stop = obligation["canonical_end"] - record["start"]
        assert 0 <= start < stop <= len(record["visible_content"])
        assert record["visible_content"][start:stop] == obligation["canonical_quote"]
        assert obligation["facet"] in review.RETENTION_FACETS
    assert case == original


@pytest.mark.parametrize("name", PAIRS)
def test_candidate_mutations_preserve_source_authority_but_change_scope_binding(name):
    left, right = map(_scope, _pair(name))
    assert left.canonical_sources == right.canonical_sources
    assert left.context_sources == right.context_sources
    assert left.boundary_context_attributions == right.boundary_context_attributions
    assert left.prior_summary_sources == right.prior_summary_sources
    assert left.binding_sha256 != right.binding_sha256
    assert left.fields != right.fields


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_gold_and_mechanism_names_are_not_model_payload(case):
    scope = _scope(case)
    changed = deepcopy(case)
    for key in set(changed) - {"payload", "target_scope"}:
        changed[key] = "DO-NOT-SEND-FIXTURE-GOLD"
    assert _scope(changed) == scope
    assert "DO-NOT-SEND-FIXTURE-GOLD" not in scope.request.user
    assert case["id"] not in scope.request.user
    assert case["pair"] not in scope.request.user
    assert not {"expected_source_obligations", "candidate_matches", "optional_details",
                "unit_chars", "whole_scope_correctness_claim"}.intersection(
                    json.loads(scope.request.user))
    for record in case["payload"]["source_catalog"]:
        assert record["chunk_id"] == f"unit-{record['message_id']}"


@pytest.mark.parametrize("case", [case for case in CASES
                                 if case["target_scope"]["kind"] != "summary"],
                         ids=lambda case: case["id"])
def test_nonselected_summary_cannot_supply_an_omitted_constraint(case):
    scope = _scope(case)
    changed = deepcopy(case)
    for key in ("candidate_summary", "candidate_raw_summary", "prior_derived_summary"):
        changed["payload"]["summary_item"][key] = "UNRELATED-SUMMARY-SENTINEL"
    assert _scope(changed) == scope
    assert "UNRELATED-SUMMARY-SENTINEL" not in scope.request.user


def test_two_constraints_have_separate_gold_even_though_their_unit_is_the_same():
    faithful, defective = _pair("same-unit-two-constraints")
    obligations = faithful["expected_source_obligations"]
    assert len(obligations) == 2
    assert [_unit_indexes(faithful, item) for item in obligations] == [(0,), (0,)]
    assert [item["expected"] for item in faithful["candidate_matches"]] == ["retained", "retained"]
    assert [item["expected"] for item in defective["candidate_matches"]] == ["retained", "omitted"]
    assert faithful["candidate_matches"][0] == defective["candidate_matches"][0]


def test_prerequisite_crosses_codepoint_units_and_uses_natural_prose_not_hex_padding():
    faithful, defective = _pair("cross-unit-prerequisite")
    record = faithful["payload"]["source_catalog"][0]
    text = record["visible_content"]
    obligation = faithful["expected_source_obligations"][0]
    assert faithful["unit_chars"] == 256
    assert _unit_indexes(faithful, obligation) == (0, 1)
    assert obligation["canonical_start"] == 213 < 256 < obligation["canonical_end"] == 291
    assert len(text[:256].encode("utf-8")) > 256
    assert text.startswith("This note describes the label collection procedure for the café room. ")
    assert "only after the coordinator confirms the room is empty" in text
    assert faithful["payload"]["procedure_items"][0]["candidate"]["steps"] == (
        defective["payload"]["procedure_items"][0]["candidate"]["steps"])


def test_paraphrase_is_not_an_exact_copy_requirement_but_reversed_condition_is_altered():
    faithful, defective = _pair("paraphrase-versus-altered-condition")
    text = faithful["payload"]["source_catalog"][0]["visible_content"]
    candidates = [case["payload"]["procedure_items"][0]["candidate"]
                  for case in (faithful, defective)]
    assert "nonreceipt" not in text
    assert "nonreceipt" in candidates[0]["triggers"][0]
    assert candidates[0]["triggers"][0].replace("nonreceipt", "receipt") == (
        candidates[1]["triggers"][0])
    assert candidates[0]["steps"] == candidates[1]["steps"]
    assert [case["candidate_matches"][0]["expected"] for case in (faithful, defective)] == [
        "retained", "altered"]


def test_prohibition_removal_does_not_modify_the_affirmative_instruction():
    faithful, defective = _pair("explicit-prohibition")
    candidates = [case["payload"]["procedure_items"][0]["candidate"]
                  for case in (faithful, defective)]
    assert candidates[0]["steps"] == candidates[1]["steps"]
    assert candidates[0]["description"] is not None and candidates[1]["description"] is None
    assert defective["candidate_matches"][0]["candidate_field_paths"] == []


def test_same_message_and_preceding_message_context_keep_separate_speaker_authority():
    faithful, defective = _pair("boundary-message-attribution")
    scope = _scope(faithful)
    same, different = faithful["payload"]["source_catalog"]
    same_context = same["interpretation_only_context"]
    different_context = different["interpretation_only_context"]
    assert same_context["message_id"] == same["message_id"] == 3406
    assert same_context["role"] == same["role"] == "user"
    assert same_context["end"] == same["start"]
    assert same_context["content"] + same["visible_content"] == "I will carry the mosaic tray."
    assert different_context["message_id"] == 3407 != different["message_id"]
    assert different_context["role"] == "assistant" and different["role"] == "user"
    assert different["start"] == 0
    assert all(unit.allowed_use == "interpretation" for unit in scope.context_sources)
    assert [(item.message_id, item.role) for item in scope.boundary_context_attributions] == [
        (3406, "user"), (3407, "assistant")]
    assert faithful["candidate_matches"][:2] == defective["candidate_matches"][:2]
    assert defective["candidate_matches"][2]["expected"] == "altered"


def test_incidental_detail_omission_is_legitimate_and_not_a_defective_negative_control():
    detailed, concise = _pair("incidental-detail-omission")
    left = detailed["payload"]["summary_item"]["candidate_summary"]
    right = concise["payload"]["summary_item"]["candidate_summary"]
    assert left == right + " The scratch paper is yellow."
    assert detailed["candidate_matches"] == concise["candidate_matches"]
    assert concise["candidate_matches"][0]["expected"] == "retained"
    assert detailed["optional_details"] == concise["optional_details"]
    assert len(concise["optional_details"]) == 1
    assert "its color is unrelated to the decision" in concise["payload"]["source_catalog"][0]["visible_content"]
