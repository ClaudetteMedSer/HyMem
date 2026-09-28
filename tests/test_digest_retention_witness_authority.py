"""Scripted field-authority checks, not provider or semantic-accuracy evidence."""
import json

import pytest

from benchmarks import digest_retention_inventory as review
from tests.test_digest_evidence_assessment import Scripted, payload, template


RETRIEVAL_PATHS = (
    "/candidate/triggers/0", "/candidate/triggers/1",
    "/candidate/entities_involved/0",
)
ASSERTION_PATHS = (
    "/candidate/name", "/candidate/description",
    "/candidate/steps/0/action", "/candidate/steps/0/tool",
    "/candidate/steps/1/action",
)


def prepared(value=None):
    return review.prepare_retention_inventory(
        payload() if value is None else value, template(), max_calls=6)


def inventory_raw(scope, facets=("constraints",), *, uncertain=False):
    # All synthetic source units receive coverage, without assuming units are
    # atomic obligations or that this scripted text is semantically correct.
    groups = {}
    for unit in scope.units:
        groups.setdefault(unit.source_id, []).append(unit.unit_id)
    return json.dumps({
        "obligations": [
            {"facet": facet, "unit_ids": unit_ids,
             "text": "Never delete cache; inspect before verification."}
            for unit_ids in groups.values() for facet in facets],
        "no_material_unit_ids": [],
        "uncertain_unit_ids": [scope.units[0].unit_id] if uncertain else [],
    })


def matching(*, kind="procedure", facets=("constraints",), uncertain=False, value=None):
    scope = next(scope for scope in prepared(value).requests if scope.kind == kind)
    inventory = review.parse_inventory(
        inventory_raw(scope, facets, uncertain=uncertain), scope)
    assert inventory.status == "valid_inventory"
    return review.prepare_matching(scope, inventory.inventory)


def field_ids(scope, paths):
    by_path = {field.path: field.field_id
               for field in scope.inventory_scope.record_scope.fields}
    return [by_path[path] for path in paths]


def reply(scope, verdict, paths):
    refs = field_ids(scope, paths)
    return json.dumps({item.obligation_id: [verdict, refs]
                       for item in scope.inventory.obligations})


def assert_rejected(outcome):
    assert outcome.status == "malformed_matching"
    assert outcome.reason == "invalid_matching"
    assert outcome.judgments == ()
    assert not outcome.structure_valid and not outcome.model_retention_satisfied
    assert not outcome.semantic_verified and not outcome.publication_authorized


def test_field_authority_contract_has_a_distinct_version():
    assert review.VERSION == "digest-retention-inventory-v4"


@pytest.mark.parametrize("facet", ["constraints", "ordering"])
@pytest.mark.parametrize("verdict", ["retained", "altered"])
@pytest.mark.parametrize("paths", [
    (RETRIEVAL_PATHS[0],), (RETRIEVAL_PATHS[1],), (RETRIEVAL_PATHS[2],),
    RETRIEVAL_PATHS[:2], (RETRIEVAL_PATHS[0], RETRIEVAL_PATHS[2]), RETRIEVAL_PATHS,
])
def test_rule_cannot_be_witnessed_only_by_retrieval_or_entity_fields(facet, verdict, paths):
    scope = matching(facets=(facet,))
    assert_rejected(review.parse_matching(reply(scope, verdict, paths), scope))


@pytest.mark.parametrize("facet", ["constraints", "ordering"])
@pytest.mark.parametrize("verdict", ["retained", "altered"])
@pytest.mark.parametrize("path", ASSERTION_PATHS)
def test_name_description_step_and_tool_remain_authorized(facet, verdict, path):
    scope = matching(facets=(facet,))
    outcome = review.parse_matching(reply(scope, verdict, (path,)), scope)
    assert outcome.structure_valid
    assert outcome.model_retention_satisfied == (verdict == "retained")
    assert not outcome.semantic_verified and not outcome.publication_authorized


@pytest.mark.parametrize("facet", ["constraints", "ordering"])
@pytest.mark.parametrize("verdict", ["retained", "altered"])
@pytest.mark.parametrize("paths", [
    ("/candidate/description", *RETRIEVAL_PATHS),
    (*RETRIEVAL_PATHS, "/candidate/description"),
    ("/candidate/steps/0/action", "/candidate/steps/1/action"),
])
def test_cross_field_witnesses_need_not_each_repeat_a_complete_rule(facet, verdict, paths):
    scope = matching(facets=(facet,))
    outcome = review.parse_matching(reply(scope, verdict, paths), scope)
    assert outcome.structure_valid
    assert outcome.judgments[0].witness_field_ids == tuple(field_ids(scope, paths))


def test_an_explicit_imperative_name_is_not_categorically_excluded():
    value = payload()
    value["procedure_items"][0]["candidate"]["name"] = "Never delete cache"
    scope = matching(value=value)
    outcome = review.parse_matching(reply(scope, "retained", ("/candidate/name",)), scope)
    assert outcome.structure_valid and outcome.model_retention_satisfied
    assert not outcome.semantic_verified and not outcome.publication_authorized


@pytest.mark.parametrize("facet", ["constraints", "ordering"])
@pytest.mark.parametrize("paths", [(), (RETRIEVAL_PATHS[0],),
                                  (RETRIEVAL_PATHS[2],), RETRIEVAL_PATHS,
                                  ("/candidate/description",)])
def test_uncertain_can_record_retrieval_witnesses_but_stays_unresolved(facet, paths):
    scope = matching(facets=(facet,))
    outcome = review.parse_matching(reply(scope, "uncertain", paths), scope)
    assert outcome.structure_valid and not outcome.model_retention_satisfied
    assert all(judgment.verdict == "uncertain" for judgment in outcome.judgments)


@pytest.mark.parametrize("facet", ["constraints", "ordering"])
def test_omitted_keeps_its_original_empty_witness_contract(facet):
    scope = matching(facets=(facet,))
    outcome = review.parse_matching(reply(scope, "omitted", ()), scope)
    assert outcome.structure_valid and not outcome.model_retention_satisfied
    assert_rejected(review.parse_matching(reply(scope, "omitted", RETRIEVAL_PATHS), scope))


@pytest.mark.parametrize("verdict", ["retained", "altered"])
@pytest.mark.parametrize("paths", [(path,) for path in RETRIEVAL_PATHS])
def test_material_facts_are_not_reclassified_as_procedure_rules(verdict, paths):
    scope = matching(facets=("material_facts",))
    outcome = review.parse_matching(reply(scope, verdict, paths), scope)
    assert outcome.structure_valid
    assert not outcome.semantic_verified and not outcome.publication_authorized


@pytest.mark.parametrize("kind", ["episode", "summary"])
@pytest.mark.parametrize("facet", ["constraints", "ordering", "material_facts"])
@pytest.mark.parametrize("verdict", ["retained", "altered"])
def test_other_scope_contracts_remain_unchanged(kind, facet, verdict):
    scope = matching(kind=kind, facets=(facet,))
    path = scope.inventory_scope.record_scope.fields[0].path
    assert review.parse_matching(reply(scope, verdict, (path,)), scope).structure_valid


def test_any_bad_obligation_rejects_whole_reply_without_salvaging_good_matches():
    scope = matching(facets=("material_facts", "constraints", "ordering"))
    trigger = field_ids(scope, (RETRIEVAL_PATHS[0],))
    assertion = field_ids(scope, ("/candidate/steps/0/action",))
    raw = json.dumps({"o0": ["retained", trigger], "o1": ["retained", assertion],
                      "o2": ["altered", trigger]})
    assert_rejected(review.parse_matching(raw, scope))


def test_known_assertion_field_does_not_sponsor_unknown_or_duplicate_ids():
    scope = matching()
    refs = field_ids(scope, ("/candidate/description",))
    for invalid_refs in (refs + ["unknown"], refs + refs, []):
        assert_rejected(review.parse_matching(
            json.dumps({"o0": ["retained", invalid_refs]}), scope))


def test_authorized_witness_does_not_clear_source_inventory_uncertainty():
    scope = matching(uncertain=True)
    outcome = review.parse_matching(reply(scope, "retained", ("/candidate/description",)), scope)
    assert outcome.structure_valid and not outcome.model_retention_satisfied
    assert scope.inventory.uncertain_unit_ids


def test_authority_guard_still_cannot_detect_mixed_irrelevant_witness_lies():
    scope = matching()
    # "Inspect safely" does not assert "Never delete cache". A known assertion
    # field passes this authority gate anyway: there is no semantic truth test.
    assert next(field.text for field in scope.inventory_scope.record_scope.fields
                if field.path == "/candidate/description") == "Inspect safely."
    outcome = review.parse_matching(reply(scope, "retained", (
        "/candidate/description", RETRIEVAL_PATHS[0])), scope)
    assert outcome.structure_valid and outcome.model_retention_satisfied
    assert not outcome.semantic_verified and not outcome.publication_authorized


def test_rejected_procedure_matching_does_not_stop_independent_scopes():
    plan = prepared()
    scripted = []
    for scope in plan.requests:
        raw_inventory = inventory_raw(scope)
        frozen = review.parse_inventory(raw_inventory, scope).inventory
        match = review.prepare_matching(scope, frozen)
        path = (RETRIEVAL_PATHS[0] if scope.kind == "procedure"
                else scope.record_scope.fields[0].path)
        scripted.extend((raw_inventory, reply(match, "retained", (path,))))
    client = Scripted(scripted)
    result = review.execute_retention_inventory(plan, client)
    assert result.complete and result.halted_reason is None
    assert len(client.calls) == result.attempted_calls == 6
    assert [outcome.matching.status for outcome in result.outcomes] == [
        "valid_matching", "malformed_matching", "valid_matching"]
    assert not result.structure_valid and not result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized
