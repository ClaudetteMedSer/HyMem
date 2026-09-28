"""Scripted contract tests; no provider calls or model-accuracy claims."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

import pytest

from benchmarks import digest_retention_inventory as inventory
from hymem.deadline import DeadlineExceeded
from tests.test_digest_evidence_assessment import payload, template, Scripted


def plan(value=None, **kwargs):
    return inventory.prepare_retention_inventory(payload() if value is None else value,
                                                 template(), max_calls=6, **kwargs)


def inventory_reply(scope, **updates):
    value = {"obligations": [
        {"facet": "constraints", "unit_ids": [unit.unit_id], "text": "A material obligation."}
        for unit in scope.units], "no_material_unit_ids": [], "uncertain_unit_ids": []}
    value.update(updates)
    return value


def accepted(scope, value=None):
    outcome = inventory.parse_inventory(json.dumps(
        inventory_reply(scope) if value is None else value, ensure_ascii=False), scope)
    assert outcome.structure_valid
    return outcome.inventory


def matching_reply(matching, verdict="retained"):
    fields = matching.inventory_scope.record_scope.fields
    refs = [fields[0].field_id] if verdict in {"retained", "altered"} else []
    return {obligation.obligation_id: [verdict, refs] for obligation in matching.inventory.obligations}


def test_reservation_exact_original_schedule_and_sampling_preserved():
    prepared = plan()
    assert prepared.reserved_calls == prepared.max_calls == 6
    assert [(s.kind, s.index) for s in prepared.requests] == [
        ("episode", 0), ("procedure", 0), ("summary", 0)]
    for scope in prepared.requests:
        assert replace(scope.request, system=template().system, user=template().user) == template()
        wire = json.loads(scope.request.user)
        assert set(wire) == {"schema", "scope", "source_records", "location_units"}
        assert "REJECTED RAW SENTINEL" not in scope.request.user
        assert "Old topic persists" not in scope.request.user
        assert scope.source_binding_sha256 == inventory._sha(asdict(scope.request))
        assert not scope.semantic_verified and not scope.publication_authorized


def test_candidate_blindness_including_full_wire_and_source_binding():
    changed = payload()
    changed["items"][0].update(candidate_body="Unrelated new candidate. 🐈",
                               candidate_title="Other", candidate_key_entities=[])
    changed["procedure_items"][0]["candidate"]["steps"][0]["action"] = "Different"
    changed["summary_item"].update(candidate_summary="New", prior_derived_summary="Changed prior")
    before, after = plan(), plan(changed)
    assert before.plan_sha256 != after.plan_sha256
    for left, right in zip(before.requests, after.requests, strict=True):
        assert left.request == right.request
        assert left.units == right.units
        assert left.source_binding_sha256 == right.source_binding_sha256
        assert left.binding_sha256 != right.binding_sha256


def test_units_lossless_offsets_codepoints_repetition_and_attribution():
    value = payload()
    source = value["source_catalog"][0]
    text = ("🐈e\u0301 ") * 150
    source.update(visible_content=text, end=8 + len(text))
    scope = plan(value).requests[0]
    assert "".join(unit.text for unit in scope.units) == text
    assert len(scope.units) == 3
    for index, unit in enumerate(scope.units):
        assert unit.unit_id == f"u{index}"
        assert unit.start == 8 + 256 * index
        assert unit.end == unit.start + len(unit.text)
        assert unit.text == text[unit.start - 8:unit.end - 8]
        assert unit.chunk_id == "chunk-a" and unit.message_id == 1
    record = json.loads(scope.request.user)["source_records"][0]
    assert record["boundary_context"]["content"] == "Earlier "
    assert record["current_message"]["message_id"] == 1
    assert all(unit.source_id == record["canonical_source"]["source_id"] for unit in scope.units)


@pytest.mark.parametrize("cap", [0, 1, 5, 65, True, 6.0])
def test_whole_schedule_reservation_fails_before_calls(cap):
    with pytest.raises(ValueError):
        inventory.prepare_retention_inventory(payload(), template(), max_calls=cap)


def test_freezes_exact_raw_bytes_and_assigns_obligation_ids_before_matching():
    scope = plan().requests[0]
    raw = json.dumps(inventory_reply(scope), indent=2)
    outcome = inventory.parse_inventory(raw, scope)
    frozen = outcome.inventory
    assert frozen.raw_json == raw
    assert frozen.raw_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert frozen.obligations[0].obligation_id == "o0"
    matching = inventory.prepare_matching(scope, frozen)
    wire = json.loads(matching.request.user)
    assert wire["model_derived_inventory"]["accepted_raw_sha256"] == frozen.raw_sha256
    assert wire["candidate"] == json.loads(scope.record_scope.request.user)["candidate"]
    assert wire["fields"] == [asdict(field) for field in scope.record_scope.fields]
    assert replace(matching.request, system=template().system, user=template().user) == template()
    assert not frozen.semantic_verified and not frozen.publication_authorized


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra=[]),
    lambda value: value["obligations"][0].update(extra="x"),
    lambda value: value["obligations"][0].update(facet="not_applicable"),
    lambda value: value["obligations"][0].update(unit_ids=["s0"]),
    lambda value: value["obligations"][0].update(unit_ids=["u0", "u0"]),
    lambda value: value["obligations"][0].update(unit_ids=[]),
    lambda value: value["obligations"][0].update(text=" "),
    lambda value: value["obligations"][0].update(text="x" * 1025),
    lambda value: value["obligations"][0].update(text=1),
    lambda value: value.update(no_material_unit_ids=["u0"]),
    lambda value: value.update(obligations=[]),
    lambda value: value.update(obligations=[value["obligations"][0]] * 129),
])
def test_entire_bad_inventory_rejected_without_salvage(mutation):
    scope = plan().requests[0]
    value = inventory_reply(scope)
    mutation(value)
    parsed = inventory.parse_inventory(json.dumps(value), scope)
    assert parsed.status == "malformed_inventory" and parsed.inventory is None
    assert not parsed.structure_valid


@pytest.mark.parametrize("raw", [None, {}, "{", "NaN", "[]", '{"obligations":[],"obligations":[]}',
                                      '"\\ud800"', " " * (inventory.MAX_OUTPUT_CHARS + 1)])
def test_malformed_json_and_output_bound_rejected(raw):
    assert inventory.parse_inventory(raw, plan().requests[0]).inventory is None


def test_one_obligation_cannot_merge_distinct_source_owners():
    scope = plan().requests[-1]
    value = inventory_reply(scope)
    value["obligations"] = [{"facet": "material_facts", "unit_ids": ["u0", "u1"], "text": "Merged"}]
    assert inventory.parse_inventory(json.dumps(value), scope).inventory is None


def test_overlapping_same_source_obligations_allowed_no_exclusion_overlap():
    scope = plan().requests[0]
    value = inventory_reply(scope)
    value["obligations"].append({"facet": "ordering", "unit_ids": ["u0"], "text": "Another condition"})
    frozen = accepted(scope, value)
    assert [item.obligation_id for item in frozen.obligations] == ["o0", "o1"]


def test_partial_unit_uncertainty_preserves_known_obligation_but_blocks_affirmation():
    scope = plan().requests[0]
    value = inventory_reply(scope, uncertain_unit_ids=["u0"])
    frozen = accepted(scope, value)
    assert frozen.obligations[0].unit_ids == frozen.uncertain_unit_ids == ("u0",)
    matching = inventory.prepare_matching(scope, frozen)
    outcome = inventory.parse_matching(json.dumps(matching_reply(matching)), matching)
    assert outcome.structure_valid
    assert not outcome.model_retention_satisfied
    assert not outcome.semantic_verified and not outcome.publication_authorized
    assert "Partial-unit uncertainty" in scope.request.system
    assert "unresolved remaining content" in matching.request.system


@pytest.mark.parametrize("obligations, uncertain", [(True, False), (False, True), (True, True)])
def test_no_material_never_overlaps_known_or_uncertain_unit(obligations, uncertain):
    scope = plan().requests[0]
    value = inventory_reply(scope, no_material_unit_ids=["u0"],
                            uncertain_unit_ids=["u0"] if uncertain else [])
    if not obligations:
        value["obligations"] = []
    outcome = inventory.parse_inventory(json.dumps(value), scope)
    assert outcome.inventory is None and outcome.status == "malformed_inventory"


def test_uncertainty_on_known_unit_cannot_hide_unaccounted_other_unit():
    scope = plan().requests[-1]
    value = inventory_reply(scope, uncertain_unit_ids=["u0"])
    value["obligations"] = value["obligations"][:1]
    outcome = inventory.parse_inventory(json.dumps(value), scope)
    assert outcome.inventory is None and not outcome.structure_valid


@pytest.mark.parametrize("plane", ["no_material_unit_ids", "uncertain_unit_ids"])
def test_empty_inventory_unassessed_even_if_location_coverage_valid(plane):
    scope = plan().requests[0]
    frozen = accepted(scope, {"obligations": [], "no_material_unit_ids": [],
                              "uncertain_unit_ids": [], plane: ["u0"]})
    with pytest.raises(ValueError, match="unassessed"):
        inventory.prepare_matching(scope, frozen)


@pytest.mark.parametrize("plane, expected", [("uncertain_unit_ids", False), ("no_material_unit_ids", True)])
def test_mixed_exclusion_is_only_model_judgment_never_semantic_authority(plane, expected):
    scope = plan().requests[-1]
    value = inventory_reply(scope)
    value["obligations"] = value["obligations"][:1]
    value[plane] = ["u1"]
    matching = inventory.prepare_matching(scope, accepted(scope, value))
    outcome = inventory.parse_matching(json.dumps(matching_reply(matching)), matching)
    assert outcome.model_retention_satisfied is expected
    assert not outcome.semantic_verified and not outcome.publication_authorized


@pytest.mark.parametrize("value", [
    {}, {"o1": ["retained", ["f0"]]}, {"o0": ["not_applicable", []]},
    {"o0": ["retained", []]}, {"o0": ["altered", []]}, {"o0": ["omitted", ["f0"]]},
    {"o0": ["retained", ["u0"]]}, {"o0": ["retained", ["f0", "f0"]]},
    {"o0": ["uncertain", [], "extra"]}, {"o0": [True, []]},
])
def test_matching_exact_obligations_status_and_witness_plane(value):
    scope = plan().requests[0]
    matching = inventory.prepare_matching(scope, accepted(scope))
    outcome = inventory.parse_matching(json.dumps(value), matching)
    assert outcome.status == "malformed_matching" and outcome.judgments == ()
    assert not outcome.model_retention_satisfied


@pytest.mark.parametrize("verdict", ["retained", "omitted", "altered", "uncertain"])
def test_every_valid_matching_verdict_is_explicit(verdict):
    scope = plan().requests[0]
    matching = inventory.prepare_matching(scope, accepted(scope))
    outcome = inventory.parse_matching(json.dumps(matching_reply(matching, verdict)), matching)
    assert outcome.structure_valid
    assert outcome.model_retention_satisfied is (verdict == "retained")


def test_valid_but_incomplete_unit_description_and_irrelevant_witness_are_not_proofs():
    scope = plan().requests[0]
    value = inventory_reply(scope)
    # Omits the canonical "not exclusive" condition while covering its unit.
    value["obligations"][0]["text"] = "Cedar is available on Aurora."
    frozen = accepted(scope, value)
    matching = inventory.prepare_matching(scope, frozen)
    # f0 title mentions availability but does not establish the whole meaning.
    result = inventory.parse_matching('{"o0":["retained",["f0"]]}', matching)
    assert result.structure_valid and result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized


def test_forged_rehashed_inventory_and_matching_are_rebuilt_from_raw():
    scope = plan().requests[0]
    frozen = accepted(scope)
    forged = replace(frozen, obligations=(replace(frozen.obligations[0], text="New obligation"),))
    forged = replace(forged, binding_sha256=inventory._sha(inventory._inventory_body(forged)))
    with pytest.raises(ValueError):
        inventory.prepare_matching(scope, forged)
    matching = inventory.prepare_matching(scope, frozen)
    forged_match = replace(matching, request=replace(matching.request, user="{}"))
    forged_match = replace(forged_match, binding_sha256=inventory._sha(inventory._matching_body(forged_match)))
    with pytest.raises(ValueError):
        inventory.parse_matching("{}", forged_match)


def test_forged_rehashed_scope_and_whole_plan_fail_before_invocation():
    prepared = plan()
    scope = prepared.requests[0]
    bad_scope = replace(scope, units=(replace(scope.units[0], start=0),))
    bad_scope = replace(bad_scope, binding_sha256=inventory._sha(inventory._scope_body(bad_scope)))
    with pytest.raises(ValueError):
        inventory.parse_inventory("{}", bad_scope)
    bad_plan = replace(prepared, requests=prepared.requests[:-1])
    bad_plan = replace(bad_plan, plan_sha256=inventory._sha(inventory._plan_body(bad_plan)))
    client = Scripted([])
    with pytest.raises(ValueError):
        inventory.execute_retention_inventory(bad_plan, client)
    assert not client.calls


def test_exact_execution_calls_reservation_and_full_outcomes():
    prepared = plan()
    values = []
    for scope in prepared.requests:
        frozen = accepted(scope)
        matching = inventory.prepare_matching(scope, frozen)
        values.extend([frozen.raw_json, json.dumps(matching_reply(matching))])
    client = Scripted(values)
    result = inventory.execute_retention_inventory(prepared, client)
    assert result.complete and result.structure_valid and result.model_retention_satisfied
    assert result.reserved_calls == result.attempted_calls == len(client.calls) == 6
    assert [outcome.attempted_calls for outcome in result.outcomes] == [2, 2, 2]
    assert not result.semantic_verified and not result.publication_authorized


def test_malformed_inventory_continues_independent_scopes_without_matching():
    prepared = plan()
    client = Scripted(["{}", "{}", "{}"])
    result = inventory.execute_retention_inventory(prepared, client)
    assert result.complete and not result.structure_valid
    assert result.reserved_calls == 6 and result.attempted_calls == 3
    assert all(item.matching.status == "skipped_invalid_inventory" for item in result.outcomes)


def test_client_exception_halts_and_accounts_remaining_scopes_without_secret():
    result = inventory.execute_retention_inventory(plan(), Scripted([RuntimeError("secret")]))
    assert not result.complete and result.halted_reason == "client_exception"
    assert result.attempted_calls == 1 and len(result.outcomes) == 3
    assert result.outcomes[0].inventory.status == "execution_error"
    assert [item.inventory.status for item in result.outcomes[1:]] == ["skipped_after_halt"] * 2
    assert "secret" not in repr(result)


@pytest.mark.parametrize("error", [DeadlineExceeded("deadline"), KeyboardInterrupt(), SystemExit(2)])
def test_deadlines_and_interrupts_propagate(error):
    client = Scripted([error])
    with pytest.raises(type(error)):
        inventory.execute_retention_inventory(plan(), client)
    assert len(client.calls) == 1


def test_empty_canonical_scopes_reserved_but_never_called_or_affirmed():
    value = payload()
    for source in value["source_catalog"]:
        source.update(visible_content="", end=source["start"])
    client = Scripted([])
    result = inventory.execute_retention_inventory(plan(value), client)
    assert result.complete and result.structure_valid
    assert result.reserved_calls == 6 and result.attempted_calls == 0 and not client.calls
    assert not result.model_retention_satisfied
    assert all(item.inventory.status == "unassessed" for item in result.outcomes)
    assert all(item.matching.status == "skipped_empty_inventory" for item in result.outcomes)
