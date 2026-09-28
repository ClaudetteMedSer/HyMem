"""Independent offline attacks; scripted replies are not semantic accuracy."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from benchmarks import digest_retention_inventory as inventory
from hymem.deadline import DeadlineExceeded
from tests.test_digest_evidence_assessment import payload, template, Scripted


def prepare(value=None, **kwargs):
    return inventory.prepare_retention_inventory(
        payload() if value is None else value, template(), max_calls=6, **kwargs)


def raw_inventory(scope):
    return {
        "obligations": [{"facet": "constraints", "unit_ids": [unit.unit_id],
                         "text": "Preserve this source-owned condition."}
                        for unit in scope.units],
        "no_material_unit_ids": [], "uncertain_unit_ids": [],
    }


def parsed(scope, raw=None):
    return inventory.parse_inventory(
        json.dumps(raw_inventory(scope) if raw is None else raw,
                   ensure_ascii=False), scope)


def match_scope(scope):
    return inventory.prepare_matching(scope, parsed(scope).inventory)


def matching_reply(scope, status="retained"):
    return {f"o{i}": [status, [] if status == "omitted" else ["f0"]]
            for i in range(len(scope.units))}


def test_inventory_is_blind_to_every_candidate_and_fallible_prior():
    original = payload()
    changed = deepcopy(original)
    changed["items"][0].update(candidate_title="CANDIDATE_TITLE_SENTINEL",
        candidate_body="CANDIDATE_BODY_SENTINEL", candidate_key_entities=["ENTITY_SENTINEL"])
    changed["procedure_items"][0]["candidate"].update(
        name="PROCEDURE_SENTINEL", description="DESCRIPTION_SENTINEL",
        triggers=["TRIGGER_SENTINEL"])
    changed["summary_item"].update(candidate_summary="SUMMARY_SENTINEL",
        candidate_raw_summary="RAW_SENTINEL", prior_derived_summary="PRIOR_SENTINEL")
    for left, right in zip(prepare(original).requests, prepare(changed).requests, strict=True):
        assert left.request == right.request
        assert left.units == right.units
        assert not any(value in right.request.user for value in (
            "CANDIDATE_TITLE_SENTINEL", "CANDIDATE_BODY_SENTINEL", "ENTITY_SENTINEL",
            "PROCEDURE_SENTINEL", "DESCRIPTION_SENTINEL", "TRIGGER_SENTINEL",
            "SUMMARY_SENTINEL", "RAW_SENTINEL", "PRIOR_SENTINEL"))
        assert replace(left.request, system=template().system, user=template().user) == template()


def test_lossless_location_units_preserve_unicode_original_offsets_and_repetition():
    value = payload()
    text = ("e\u0301🐈 same same.\n" * 80) + "Never discard either copy."
    value["source_catalog"][0].update(visible_content=text, end=8 + len(text))
    scope = prepare(value).requests[0]
    assert len(scope.units) > 1
    assert "".join(unit.text for unit in scope.units) == text
    position = 8
    for unit in scope.units:
        assert unit.start == position
        assert unit.end - unit.start == len(unit.text)
        assert unit.text == text[unit.start - 8:unit.end - 8]
        position = unit.end
    assert position == 8 + len(text)
    assert len({unit.unit_id for unit in scope.units}) == len(scope.units)


def test_boundary_and_prior_cannot_be_returned_as_canonical_obligations():
    scope = prepare().requests[-1]
    assert "Old topic persists." not in scope.request.user
    assert any(unit.text.startswith("Cedar is available") for unit in scope.units)
    assert all("Earlier " not in unit.text for unit in scope.units)
    raw = raw_inventory(scope)
    raw["obligations"][0]["unit_ids"] = ["boundary-context"]
    assert parsed(scope, raw).inventory is None
    raw["obligations"][0]["unit_ids"] = ["prior-summary"]
    assert parsed(scope, raw).inventory is None


@pytest.mark.parametrize("mutation", [
    lambda raw, units: raw["obligations"].clear(),
    lambda raw, units: raw["no_material_unit_ids"].append(units[0]),
    lambda raw, units: raw["uncertain_unit_ids"].extend([units[0], units[0]]),
    lambda raw, units: raw["obligations"][0]["unit_ids"].append(units[0]),
    lambda raw, units: raw["obligations"][0].update(facet="not_applicable"),
    lambda raw, units: raw["obligations"][0].update(text=""),
    lambda raw, units: raw["obligations"][0].update(text=" " * 3),
    lambda raw, units: raw.update(extra="not allowed"),
    lambda raw, units: raw["obligations"][0].update(extra=True),
])
def test_inventory_never_salvages_partial_or_ambiguous_coverage(mutation):
    scope = prepare().requests[-1]
    raw = raw_inventory(scope)
    mutation(raw, [unit.unit_id for unit in scope.units])
    outcome = parsed(scope, raw)
    assert outcome.inventory is None
    assert not outcome.semantic_verified and not outcome.publication_authorized


def test_two_obligations_in_one_location_unit_are_not_collapsed():
    scope = prepare().requests[1]
    raw = raw_inventory(scope)
    raw["obligations"].append({"facet": "ordering", "unit_ids": [scope.units[0].unit_id],
                               "text": "Inspection must precede verification."})
    accepted = parsed(scope, raw).inventory
    assert len(accepted.obligations) == 2
    matching = inventory.prepare_matching(scope, accepted)
    omitted = inventory.parse_matching('{"o0":["retained",["f0"]],"o1":["omitted",[]]}', matching)
    assert not omitted.model_retention_satisfied
    missing = inventory.parse_matching('{"o0":["retained",["f0"]]}', matching)
    assert not missing.model_retention_satisfied


@pytest.mark.parametrize("reply", [
    '{"o0":["retained",[]]}', '{"o0":["omitted",["f0"]]}',
    '{"o0":["retained",["s0"]]}', '{"o0":["retained",["f0","f0"]]}',
    '{"o0":["not_applicable",[]]}', '{"o0":["retained",["f0"]],"o1":["retained",["f0"]]}',
    '{"o0":["retained",["f0"]],"o0":["omitted",[]]}',
    '{"o0":["retained",[true]]}', '{"o0":["retained",[NaN]]}',
    '{"o0":["retained",["f0"]]', '{}', 'null', '[]',
])
def test_matching_rejects_unknown_missing_duplicate_and_waiver_witnesses(reply):
    result = inventory.parse_matching(reply, match_scope(prepare().requests[0]))
    assert not result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized


def test_valid_but_irrelevant_witness_remains_model_claim_not_semantic_proof():
    scope = prepare().requests[0]
    # f0 is just the availability title; it does not retain every condition.
    result = inventory.parse_matching('{"o0":["retained",["f0"]]}', match_scope(scope))
    assert result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized


def test_raw_inventory_and_parsed_objects_cannot_disagree():
    scope = prepare().requests[0]
    accepted = parsed(scope).inventory
    with pytest.raises((ValueError, TypeError)):
        inventory.prepare_matching(scope, replace(accepted, obligations=()))
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        accepted.obligations = ()


def test_rehashed_inventory_forgery_is_rebuilt_from_raw_not_trusted():
    scope = prepare().requests[0]
    accepted = parsed(scope).inventory
    obligation = replace(accepted.obligations[0], text="Forged requirement")
    forged = replace(accepted, obligations=(obligation,))
    forged = replace(forged, binding_sha256=inventory._sha(inventory._inventory_body(forged)))
    with pytest.raises(ValueError):
        inventory.prepare_matching(scope, forged)


def test_uncertain_units_prevent_all_retained_even_with_valid_matches():
    scope = prepare().requests[-1]
    raw = raw_inventory(scope)
    removed = raw["obligations"].pop()
    raw["uncertain_unit_ids"].extend(removed["unit_ids"])
    accepted = parsed(scope, raw).inventory
    matching = inventory.prepare_matching(scope, accepted)
    result = inventory.parse_matching('{"o0":["retained",["f0"]]}', matching)
    assert result.status == "valid_matching"
    assert not result.model_retention_satisfied


@pytest.mark.parametrize("verdict,witnesses", [("retained", ["f0"]), ("omitted", [])])
def test_partly_uncertain_unit_keeps_known_obligation_but_cannot_pass(verdict, witnesses):
    scope = prepare().requests[0]
    raw = raw_inventory(scope)
    raw["uncertain_unit_ids"] = [scope.units[0].unit_id]
    accepted = parsed(scope, raw)
    assert accepted.structure_valid and len(accepted.inventory.obligations) == 1
    matching = inventory.prepare_matching(scope, accepted.inventory)
    result = inventory.parse_matching(json.dumps({"o0": [verdict, witnesses]}), matching)
    assert result.structure_valid and not result.model_retention_satisfied


def test_no_material_cannot_overlap_an_uncertain_location():
    scope = prepare().requests[0]
    raw = {"obligations": [], "no_material_unit_ids": ["u0"], "uncertain_unit_ids": ["u0"]}
    assert parsed(scope, raw).inventory is None


def test_wholly_no_material_is_unassessed_not_vacuously_retained():
    scope = prepare().requests[0]
    raw = {"obligations": [], "no_material_unit_ids": [u.unit_id for u in scope.units],
           "uncertain_unit_ids": []}
    accepted = parsed(scope, raw)
    assert accepted.status == "unassessed"
    with pytest.raises(ValueError):
        inventory.prepare_matching(scope, accepted.inventory)


def test_location_coverage_does_not_prove_every_obligation_was_extracted():
    scope = prepare().requests[1]
    assert "Never delete cache" in scope.units[0].text
    raw = raw_inventory(scope)
    raw["obligations"][0].update(facet="material_facts", text="Inspect cache.")
    accepted = parsed(scope, raw)
    # The textual unit is accounted for despite the missing prohibition in the
    # MODEL's inventory. A future semantic evaluation must count this as a miss.
    assert accepted.structure_valid
    assert not accepted.semantic_verified and not accepted.publication_authorized


def test_obligation_cannot_fuse_different_canonical_source_owners():
    scope = prepare().requests[-1]
    raw = raw_inventory(scope)
    raw["obligations"] = [{"facet": "constraints", "unit_ids": [u.unit_id for u in scope.units],
                           "text": "A fused cross-source obligation."}]
    assert parsed(scope, raw).inventory is None


@pytest.mark.parametrize("raw", [
    '{"obligations":[],"obligations":[],"no_material_unit_ids":[],"uncertain_unit_ids":[]}',
    '{"obligations":NaN,"no_material_unit_ids":[],"uncertain_unit_ids":[]}',
    '{"obligations":[],"no_material_unit_ids":[],"uncertain_unit_ids":[]',
    '{"obligations":[],"no_material_unit_ids":["u0","u0"],"uncertain_unit_ids":[]}',
    'null', '[]', True, None,
])
def test_strict_inventory_json_never_accepts_duplicate_nonfinite_or_partial(raw):
    outcome = inventory.parse_inventory(raw, prepare().requests[0])
    assert outcome.inventory is None


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "6", 65])
def test_reservation_rejects_noninteger_or_out_of_range_calls(bad):
    with pytest.raises(ValueError):
        inventory.prepare_retention_inventory(payload(), template(), max_calls=bad)


def test_complete_scope_reservation_cannot_be_partially_admitted():
    with pytest.raises(ValueError):
        inventory.prepare_retention_inventory(payload(), template(), max_calls=5)


def test_client_failure_counts_attempt_and_preserves_unattempted_scope_rows():
    plan = prepare()
    client = Scripted([RuntimeError("must-not-appear-secret")])
    result = inventory.execute_retention_inventory(plan, client)
    assert result.attempted_calls == len(client.calls) == 1
    assert result.reserved_calls == 6
    assert len(result.outcomes) == 3
    assert not result.complete and not result.model_retention_satisfied
    assert "must-not-appear-secret" not in repr(result)


def test_second_stage_client_failure_counts_both_calls_and_halts_later_scopes():
    plan = prepare()
    client = Scripted([json.dumps(raw_inventory(plan.requests[0])), RuntimeError("private-error")])
    result = inventory.execute_retention_inventory(plan, client)
    assert result.attempted_calls == len(client.calls) == 2
    assert [row.attempted_calls for row in result.outcomes] == [2, 0, 0]
    assert result.outcomes[0].matching.status == "execution_error"
    assert all(row.matching.status == "skipped_after_halt" for row in result.outcomes[1:])
    assert not result.complete and "private-error" not in repr(result)


def test_malformed_matching_keeps_no_partial_judgments_and_continues_independent_scopes():
    plan = prepare()
    replies = []
    for scope in plan.requests:
        replies.extend([json.dumps(raw_inventory(scope)), '{"o0":["retained",["f0"]],"extra":[]}'])
    client = Scripted(replies)
    result = inventory.execute_retention_inventory(plan, client)
    assert result.attempted_calls == result.reserved_calls == len(client.calls) == 6
    assert result.complete and not result.structure_valid
    assert all(row.matching.judgments == () for row in result.outcomes)
    assert not result.model_retention_satisfied


def test_altered_plan_during_inventory_call_prevents_matching_call():
    plan = prepare()

    class MutatingClient:
        calls = 0

        def complete(self, request):
            self.calls += 1
            object.__setattr__(plan.requests[-1], "units", ())
            return json.dumps(raw_inventory(plan.requests[0]))

    client = MutatingClient()
    with pytest.raises(ValueError):
        inventory.execute_retention_inventory(plan, client)
    assert client.calls == 1


def test_inventory_from_another_scope_cannot_be_substituted():
    plan = prepare()
    accepted = parsed(plan.requests[0]).inventory
    with pytest.raises(ValueError):
        inventory.prepare_matching(plan.requests[1], accepted)


def test_rehashed_forged_matching_obligation_cannot_remove_an_omission():
    scope = prepare().requests[1]
    raw = raw_inventory(scope)
    raw["obligations"].append({"facet": "constraints", "unit_ids": [scope.units[0].unit_id],
                               "text": "Never delete cache."})
    accepted = parsed(scope, raw).inventory
    matching = inventory.prepare_matching(scope, accepted)
    forged_inventory = replace(accepted, obligations=accepted.obligations[:1])
    forged_inventory = replace(forged_inventory, binding_sha256=inventory._sha(
        inventory._inventory_body(forged_inventory)))
    forged = replace(matching, inventory=forged_inventory)
    forged = replace(forged, binding_sha256=inventory._sha(inventory._matching_body(forged)))
    with pytest.raises(ValueError):
        inventory.parse_matching('{"o0":["retained",["f0"]]}', forged)


@pytest.mark.parametrize("failure", [DeadlineExceeded("root timeout"), KeyboardInterrupt(), SystemExit(2)])
def test_interrupts_and_deadlines_never_become_model_failures(failure):
    client = Scripted([failure])
    with pytest.raises(type(failure)):
        inventory.execute_retention_inventory(prepare(), client)
    assert len(client.calls) == 1


def test_malformed_inventory_skips_only_dependent_match_without_retry():
    plan = prepare()
    client = Scripted(['{}', '{}', '{}'])
    result = inventory.execute_retention_inventory(plan, client)
    assert result.attempted_calls == len(client.calls) == 3
    assert result.reserved_calls == 6 and len(result.outcomes) == 3
    assert not result.model_retention_satisfied


def test_empty_canonical_inventory_never_becomes_affirmative():
    value = payload()
    for source in value["source_catalog"]:
        source.update(visible_content="", end=source["start"])
    plan = prepare(value)
    client = Scripted([])
    result = inventory.execute_retention_inventory(plan, client)
    assert not result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized


def test_plan_candidate_mutation_cannot_bypass_reconstruction():
    plan = prepare()
    scope = plan.requests[0]
    changed = replace(scope, request=replace(scope.request, user=scope.request.user + " "))
    forged = replace(plan, requests=(changed, *plan.requests[1:]))
    client = Scripted([])
    with pytest.raises(ValueError):
        inventory.execute_retention_inventory(forged, client)
    assert not client.calls
