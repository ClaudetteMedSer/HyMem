"""Summary-only API boundary tests; scripted replies are not model accuracy."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from benchmarks import digest_retention_inventory as v4
from benchmarks import digest_summary_retention as review
from hymem.deadline import DeadlineExceeded
from tests.test_digest_evidence_assessment import Scripted, payload, template


def multi_topic_payload():
    value = payload()
    text = "Inspect cache using scanner. Restart the worker using workerctl."
    value["source_catalog"][0].update(visible_content=text, end=8 + len(text))
    other = "Never remove audit snapshots."
    value["source_catalog"][1].update(visible_content=other, end=len(other))
    value["items"] = []
    value["procedure_items"] = [{
        "index": index, "candidate": {
            "name": name, "description": action + ".",
            "steps": [{"order": 1, "action": action, "tool": tool}],
            "triggers": [name.lower()], "entities_involved": [tool]},
        "cited_source_ids": ["chunk-a"],
    } for index, (name, action, tool) in enumerate((
        ("Inspect cache", "Inspect cache using scanner", "scanner"),
        ("Restart worker", "Restart the worker using workerctl", "workerctl"),
    ))]
    value["summary_item"].update(
        candidate_summary=text + " " + other,
        prior_derived_summary="PRIOR_ONLY: old unrelated deployment.")
    return value


def prepared(value=None, **kwargs):
    return review.prepare_summary_retention(
        multi_topic_payload() if value is None else value, template(), **kwargs)


def inventory_raw(plan, *, uncertain=False, empty=False):
    scope = plan.inventory_scope
    return json.dumps({
        "obligations": [] if empty else [
            {"facet": "constraints", "unit_ids": [unit.unit_id], "text": unit.text}
            for unit in scope.units],
        "no_material_unit_ids": [unit.unit_id for unit in scope.units] if empty else [],
        "uncertain_unit_ids": [scope.units[0].unit_id] if uncertain else [],
    })


def matching(plan, *, uncertain=False):
    inventory = review.parse_summary_inventory(inventory_raw(plan, uncertain=uncertain), plan)
    assert inventory.structure_valid
    return review.prepare_summary_matching(plan, inventory.inventory)


def matching_raw(scope, verdict="retained"):
    inner = scope.matching_scope
    refs = [inner.inventory_scope.record_scope.fields[0].field_id]
    return json.dumps({obligation.obligation_id: [verdict, refs if verdict == "retained" else []]
                       for obligation in inner.inventory.obligations})


def assert_scope_flags(value):
    assert value.version == "digest-summary-new-source-retention-v1"
    assert value.coverage_scope == "summary_new_sources_only"
    assert value.prior_continuity_assessed is False
    assert value.grounding_assessed is False and value.item_retention_assessed is False
    assert not value.semantic_verified and not value.publication_authorized


def test_two_narrow_procedures_are_not_exhaustive_retention_owners():
    value = multi_topic_payload()
    before = deepcopy(value)
    plan = prepared(value)
    assert_scope_flags(plan)
    assert plan.max_calls == plan.reserved_calls == 2
    assert plan.inventory_scope.kind == "summary" and plan.inventory_scope.index == 0
    # These original item scopes remain validation-only; no item retention
    # inventory is produced and they do not determine summary source coverage.
    assert [(scope.kind, scope.index) for scope in plan.validation_plan.requests] == [
        ("procedure", 0), ("procedure", 1), ("summary", 0)]
    assert plan.validation_plan.max_calls == v4.record_review.MAX_CALLS
    assert not hasattr(plan, "requests")
    assert [source.chunk_id for source in plan.inventory_scope.record_scope.canonical_sources] == [
        "chunk-a", "chunk-b"]
    assert value["procedure_items"][0]["cited_source_ids"] == ["chunk-a"]
    assert value["procedure_items"][1]["cited_source_ids"] == ["chunk-a"]
    assert value == before


def test_both_stage_requests_are_exact_existing_summary_requests():
    value = multi_topic_payload()
    plan = prepared(value)
    original = v4.prepare_retention_inventory(value, template(), max_calls=6)
    summary = next(scope for scope in original.requests if scope.kind == "summary")
    assert plan.inventory_scope == summary
    frozen = review.parse_summary_inventory(inventory_raw(plan), plan).inventory
    wrapped = review.prepare_summary_matching(plan, frozen)
    assert wrapped.matching_scope == v4.prepare_matching(summary, frozen)
    assert_scope_flags(wrapped)
    assert wrapped.plan_binding_sha256 == plan.binding_sha256


def test_first_call_excludes_candidate_and_prior_and_preserves_owned_sources():
    plan = prepared()
    wire = json.loads(plan.inventory_scope.request.user)
    assert set(wire) == {"schema", "scope", "source_records", "location_units"}
    assert "PRIOR_ONLY" not in plan.inventory_scope.request.user
    assert "REJECTED RAW SENTINEL" not in plan.inventory_scope.request.user
    assert [record["chunk_id"] for record in wire["source_records"]] == ["chunk-a", "chunk-b"]
    assert wire["source_records"][0]["current_message"]["message_id"] == 1
    assert wire["source_records"][0]["boundary_context"]["message"]["message_id"] == 1
    assert wire["source_records"][1]["current_message"]["message_id"] == 2
    for record in plan.inventory_scope.record_scope.canonical_sources:
        assert "".join(unit.text for unit in plan.inventory_scope.units
                       if unit.source_id == record.source_id) == record.text


def test_effective_summary_and_prior_are_preserved_only_in_matching():
    value = multi_topic_payload()
    plan = prepared(value)
    scope = matching(plan)
    wire = json.loads(scope.matching_scope.request.user)
    assert wire["candidate"]["candidate_summary"] == value["summary_item"]["candidate_summary"]
    assert "PRIOR_ONLY" in scope.matching_scope.request.user
    assert "REJECTED RAW SENTINEL" not in scope.matching_scope.request.user
    assert not scope.prior_continuity_assessed


def test_candidate_and_prior_mutation_cannot_change_first_stage_or_sources():
    value = multi_topic_payload()
    first = prepared(value)
    value["summary_item"].update(candidate_summary="Different candidate", prior_derived_summary="Other prior")
    value["procedure_items"][0]["candidate"]["description"] = "A different narrow description."
    second = prepared(value)
    assert first.binding_sha256 != second.binding_sha256
    assert first.inventory_scope.request == second.inventory_scope.request
    assert first.inventory_scope.units == second.inventory_scope.units


@pytest.mark.parametrize("cap", [0, 1, 3, 64, True, 2.0, "2", None])
def test_only_exact_two_call_cap_is_accepted(cap):
    with pytest.raises(ValueError):
        prepared(max_calls=cap)


@pytest.mark.parametrize("mutation", [
    lambda value: value["procedure_items"][0]["candidate"]["steps"][0].update(order=0),
    lambda value: value["procedure_items"][1].update(index=0),
    lambda value: value["procedure_items"][0].update(cited_source_ids=["missing"]),
    lambda value: value["procedure_items"][0]["candidate"].update(extra="bad"),
    lambda value: value["source_catalog"][0].update(end=99999),
    lambda value: value["summary_item"].update(candidate_raw_summary=42),
])
def test_invalid_unselected_input_is_rejected_before_preparation(mutation):
    value = multi_topic_payload()
    mutation(value)
    with pytest.raises(ValueError):
        prepared(value)


@pytest.mark.parametrize("ids", [[], ["chunk-a"], ["chunk-b"],
                                 ["chunk-b", "chunk-a"], ["chunk-a", "chunk-a"],
                                 ["chunk-a", "unknown"]])
def test_summary_new_source_authority_cannot_be_silently_rebuilt_from_item_citations(ids):
    value = multi_topic_payload()
    value["summary_item"]["new_source_ids"] = ids
    with pytest.raises(ValueError):
        prepared(value)


def test_empty_canonical_scope_is_not_a_vacuous_retention_pass():
    value = multi_topic_payload()
    for record in value["source_catalog"]:
        record.update(visible_content="", end=record["start"])
    with pytest.raises(ValueError, match="nonempty canonical text"):
        prepared(value)


def test_one_empty_record_does_not_drop_its_ownership_or_the_other_record():
    value = multi_topic_payload()
    value["source_catalog"][0].update(visible_content="", end=8)
    plan = prepared(value)
    assert len(plan.inventory_scope.record_scope.source_records) == 2
    assert {unit.chunk_id for unit in plan.inventory_scope.units} == {"chunk-b"}


def test_omission_from_summary_of_source_not_cited_by_any_item_remains_rejectable():
    value = multi_topic_payload()
    value["summary_item"]["candidate_summary"] = value["source_catalog"][0]["visible_content"]
    plan = prepared(value)
    scope = matching(plan)
    raw = json.loads(matching_raw(scope))
    omitted = next(item for item in scope.matching_scope.inventory.obligations
                   if item.source_id == plan.inventory_scope.units[-1].source_id)
    raw[omitted.obligation_id] = ["omitted", []]
    result = review.execute_summary_retention(plan, Scripted([inventory_raw(plan), json.dumps(raw)]))
    assert result.collection_complete and result.summary_structure_valid
    assert not result.summary_model_retention_satisfied
    assert_scope_flags(result)


def test_two_calls_execute_only_summary_and_never_claim_digest_acceptance():
    plan = prepared()
    scope = matching(plan)
    client = Scripted([inventory_raw(plan), matching_raw(scope)])
    result = review.execute_summary_retention(plan, client)
    assert len(client.calls) == result.attempted_calls == result.reserved_calls == 2
    assert all(json.loads(request.user)["scope"] == {"kind": "summary", "index": 0}
               for request in client.calls)
    assert result.collection_complete and result.summary_structure_valid
    assert result.summary_model_retention_satisfied
    for name in ("complete", "pass", "structure_valid", "model_retention_satisfied", "outcomes"):
        assert not hasattr(result, name)
    assert_scope_flags(result)


@pytest.mark.parametrize("raw,expected", [("{", "skipped_invalid_inventory"),
                                         (None, "skipped_invalid_inventory"),
                                         ("empty", "skipped_empty_inventory")])
def test_bad_or_empty_inventory_never_schedules_matching_or_affirms_retention(raw, expected):
    plan = prepared()
    client = Scripted([inventory_raw(plan, empty=True) if raw == "empty" else raw])
    result = review.execute_summary_retention(plan, client)
    assert result.attempted_calls == len(client.calls) == 1
    assert result.matching.status == expected
    assert not result.collection_complete
    assert not result.summary_model_retention_satisfied


def test_partial_uncertainty_and_whole_matching_response_rules_are_preserved():
    plan = prepared()
    scope = matching(plan, uncertain=True)
    result = review.parse_summary_matching(matching_raw(scope), plan, scope)
    assert result.structure_valid and not result.model_retention_satisfied
    malformed = json.loads(matching_raw(scope))
    del malformed[next(iter(malformed))]
    result = review.parse_summary_matching(json.dumps(malformed), plan, scope)
    assert result.status == "malformed_matching" and result.judgments == ()


def test_inventory_cannot_omit_a_summary_source_or_merge_different_source_owners():
    plan = prepared()
    raw = json.loads(inventory_raw(plan))
    raw["obligations"].pop()
    assert review.parse_summary_inventory(json.dumps(raw), plan).inventory is None
    raw["obligations"][0]["unit_ids"] = [unit.unit_id for unit in plan.inventory_scope.units]
    assert review.parse_summary_inventory(json.dumps(raw), plan).inventory is None


def test_raw_item_scopes_and_item_matching_cannot_enter_summary_api():
    original = v4.prepare_retention_inventory(multi_topic_payload(), template(), max_calls=6)
    item = original.requests[0]
    with pytest.raises(ValueError):
        review.parse_summary_inventory("{}", item)
    plan = prepared()
    forged = replace(plan, inventory_scope=item)
    forged = replace(forged, binding_sha256=v4._sha(review._body(forged)))
    with pytest.raises(ValueError):
        review.execute_summary_retention(forged, Scripted([]))


def test_stages_are_bound_to_the_exact_whole_validated_input():
    first = prepared()
    value = multi_topic_payload()
    value["summary_item"]["candidate_summary"] = "Different summary."
    second = prepared(value)
    scope = matching(first)
    with pytest.raises(ValueError):
        review.parse_summary_matching(matching_raw(scope), second, scope)
    with pytest.raises(ValueError):
        review.prepare_summary_matching(second, scope.matching_scope.inventory)
    with pytest.raises(ValueError):
        review.parse_summary_matching(matching_raw(scope), first, scope.matching_scope)


@pytest.mark.parametrize("name,value", [("coverage_scope", "whole_digest"),
                                       ("prior_continuity_assessed", True),
                                       ("grounding_assessed", True),
                                       ("item_retention_assessed", True),
                                       ("max_calls", True), ("reserved_calls", 3)])
def test_forged_capability_or_reservation_flags_rejected_even_with_new_hash(name, value):
    plan = prepared()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(plan, name, value)
    object.__setattr__(plan, name, value)
    object.__setattr__(plan, "binding_sha256", v4._sha(review._body(plan)))
    with pytest.raises(ValueError):
        review.execute_summary_retention(plan, Scripted([]))


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("failure", [False, True])
def test_revalidate_after_every_external_completion_even_when_client_raises(stage, failure):
    plan = prepared()
    scope = matching(plan)

    class MutatingClient:
        calls = 0

        def complete(self, request):
            self.calls += 1
            if self.calls == stage:
                object.__setattr__(plan, "grounding_assessed", True)
                if failure:
                    raise RuntimeError("private failure text")
            return inventory_raw(plan) if self.calls == 1 else matching_raw(scope)

    client = MutatingClient()
    with pytest.raises(ValueError):
        review.execute_summary_retention(plan, client)
    assert client.calls == stage


@pytest.mark.parametrize("stage", [1, 2])
def test_ordinary_client_failure_is_sanitized_and_never_retried(stage):
    plan = prepared()
    calls = [RuntimeError("PRIVATE_ERROR_TEXT")] if stage == 1 else [
        inventory_raw(plan), RuntimeError("PRIVATE_ERROR_TEXT")]
    client = Scripted(calls)
    result = review.execute_summary_retention(plan, client)
    assert result.attempted_calls == len(client.calls) == stage
    assert result.halted_reason == "client_exception" and not result.collection_complete
    assert not result.summary_structure_valid and not result.summary_model_retention_satisfied
    assert "PRIVATE_ERROR_TEXT" not in json.dumps(asdict(result))


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("error", [DeadlineExceeded("deadline"), KeyboardInterrupt(), SystemExit(1)])
def test_deadlines_and_process_interrupts_propagate_without_retry(stage, error):
    plan = prepared()
    client = Scripted([error] if stage == 1 else [inventory_raw(plan), error])
    with pytest.raises(type(error)):
        review.execute_summary_retention(plan, client)
    assert len(client.calls) == stage


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("error", [DeadlineExceeded("deadline"), KeyboardInterrupt(), SystemExit(1)])
def test_cancellation_is_not_masked_by_callback_mutating_the_plan(stage, error):
    plan = prepared()

    class CancellingClient:
        calls = 0

        def complete(self, request):
            self.calls += 1
            if self.calls == stage:
                object.__setattr__(plan, "grounding_assessed", True)
                raise error
            return inventory_raw(plan)

    client = CancellingClient()
    with pytest.raises(type(error)) as captured:
        review.execute_summary_retention(plan, client)
    assert captured.value is error
    assert client.calls == stage


def test_malformed_second_reply_completes_collection_without_semantic_acceptance():
    plan = prepared()
    result = review.execute_summary_retention(plan, Scripted([inventory_raw(plan), "{"]))
    assert result.collection_complete and result.attempted_calls == 2
    assert not result.summary_structure_valid and not result.summary_model_retention_satisfied


def test_matching_bounds_skip_is_not_a_complete_collection(monkeypatch):
    plan = prepared()

    def too_large(*args):
        raise ValueError("matching request exceeds complete input cap")

    monkeypatch.setattr(review, "prepare_summary_matching", too_large)
    client = Scripted([inventory_raw(plan)])
    result = review.execute_summary_retention(plan, client)
    assert result.matching.status == "skipped_matching_bounds"
    assert result.attempted_calls == len(client.calls) == 1
    assert not result.collection_complete and not result.summary_model_retention_satisfied


@pytest.mark.parametrize("field_name,bound", [("max_input_chars", review.MAX_INPUT_CHARS),
                                             ("max_output_chars", review.MAX_OUTPUT_CHARS)])
@pytest.mark.parametrize("bad", [0, True, "large", "above"])
def test_existing_input_output_bounds_are_not_weakened(field_name, bound, bad):
    with pytest.raises(ValueError):
        prepared(**{field_name: bound + 1 if bad == "above" else bad})
