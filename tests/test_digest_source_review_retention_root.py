"""Independent source-owned retention checks. Scripted judgments are not gold."""
from copy import deepcopy
from dataclasses import replace
import json
import socket

import pytest

from benchmarks import digest_source_review as review
from hymem.extraction.llm import LLMRequest
from tests.digest_evidence_assessment_fixtures import build_cases
from tests.test_digest_source_review_root import Client, packet, parse, prepare, response


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("offline retention controls prohibit networking")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


FACETS = ("material_facts", "constraints", "ordering")


def retention(scope, chunk_id=None, facet=None):
    sources = {s.source_id: s for s in scope.canonical_sources}
    return [c for c in scope.checks if c.kind == "retention"
            and (chunk_id is None or sources[c.source_id].chunk_id == chunk_id)
            and (facet is None or c.facet == facet)]


def fixture_scope(name, kind):
    case = next(c for c in build_cases() if c["id"] == name)
    plan = review.prepare_source_review(case["payload"], LLMRequest("", ""), max_calls=64)
    return next(s for s in plan.requests if s.kind == kind)


def neutral(scope):
    result = response(scope)
    for check in retention(scope):
        result[check.check_id] = ["not_applicable", []]
    return result


def test_three_source_owned_obligations_precede_grounding_without_extra_calls():
    plan = prepare()
    assert len(plan.requests) == 3
    for scope in plan.requests:
        checks = retention(scope)
        assert [(c.source_id, c.facet) for c in checks] == [
            (source.source_id, facet) for source in scope.canonical_sources for facet in FACETS]
        assert tuple(checks) == scope.checks[:len(checks)]
        assert len({c.check_id for c in scope.checks}) == len(scope.checks)
        assert all(c.facet is None for c in scope.checks if c.kind != "retention")
        assert [c.check_id for c in scope.checks if c.kind != "retention"] == [
            c.check_id for c in scope.base_scope.checks if c.kind != "retention"]
        assert all(c.field_ids == tuple(f.field_id for f in scope.fields) for c in checks)


@pytest.mark.parametrize("state", ["retained", "altered"])
def test_positive_or_altered_retention_requires_actual_candidate_witnesses(state):
    scope = prepare().requests[0]
    check = retention(scope)[0]
    body = neutral(scope)
    body[check.check_id] = [state, []]
    assert parse(scope, body).status == "malformed_review"
    body[check.check_id] = [state, [scope.fields[0].field_id]]
    result = parse(scope, body)
    assert result.review_structure_valid
    found = next(j for j in result.judgments if j.check_id == check.check_id)
    assert found.witness_fields == (scope.fields[0],)
    assert found.witness_field_ids == (scope.fields[0].field_id,)
    assert found.source_id == check.source_id and found.facet == check.facet
    assert result.model_retention_satisfied == (state == "retained")


@pytest.mark.parametrize("state", ["omitted", "not_applicable"])
def test_absence_judgments_cannot_claim_presence_witnesses(state):
    scope = prepare().requests[0]
    check = retention(scope)[0]
    body = neutral(scope)
    body[check.check_id] = [state, [scope.fields[0].field_id]]
    assert parse(scope, body).status == "malformed_review"
    body[check.check_id][1] = []
    result = parse(scope, body)
    assert result.review_structure_valid
    assert result.model_retention_satisfied == (state == "not_applicable")


@pytest.mark.parametrize("bad", [
    ["retained", ["s0"]], ["retained", ["f999"]], ["retained", [True]],
    ["retained", ["f0", "f0"]], ["supported", ["f0"]],
    ["retained", ["f0"], []], ["uncertain", "f0"],
    ["not_applicable", None], ["altered"], [True, []],
])
def test_bad_retention_rejects_entire_scope_without_salvage(bad):
    scope = prepare().requests[0]
    body = neutral(scope)
    body[retention(scope)[0].check_id] = bad
    result = parse(scope, body)
    assert result.status == "malformed_review" and result.judgments == ()
    assert not result.model_no_defect


def test_prohibition_omission_is_separate_from_surviving_true_actions():
    scope = fixture_scope("assessment-prohibition-omission-defective", "procedure")
    assert not any(f.path == "/candidate/description" for f in scope.fields)
    assert "Never release the vent" in scope.canonical_sources[0].text
    body = neutral(scope)
    check = retention(scope, facet="constraints")[0]
    body[check.check_id] = ["omitted", []]
    result = parse(scope, body)
    assert result.review_structure_valid and result.model_grounding_supported
    assert not result.model_retention_satisfied and not result.model_no_defect
    judged = next(j for j in result.judgments if j.check_id == check.check_id)
    assert judged.source_id == scope.canonical_sources[0].source_id
    assert judged.witness_fields == () and judged.verdict == "omitted"
    assert judged.primary_evidence == judged.context_evidence == ()
    assert not result.semantic_verified and not result.publication_authorized


def test_explicitly_preserved_prohibition_can_use_description_field():
    scope = fixture_scope("assessment-prohibition-omission-faithful", "procedure")
    body = neutral(scope)
    field = next(f for f in scope.fields if f.path == "/candidate/description")
    body[retention(scope, facet="constraints")[0].check_id] = ["retained", [field.field_id]]
    result = parse(scope, body)
    assert result.review_structure_valid and result.model_retention_satisfied


def test_answer_omission_is_attached_to_answer_not_request():
    scope = fixture_scope("assessment-answered-outcome-omission-defective", "summary")
    body = neutral(scope)
    request = retention(scope, "audit-request", "material_facts")[0]
    answer = retention(scope, "audit-answer", "material_facts")[0]
    body[request.check_id] = ["retained", [scope.fields[0].field_id]]
    body[answer.check_id] = ["omitted", []]
    result = parse(scope, body)
    mapping = {j.check_id: j for j in result.judgments}
    assert mapping[request.check_id].verdict == "retained"
    assert mapping[answer.check_id].verdict == "omitted"
    assert result.model_grounding_supported and not result.model_retention_satisfied


def test_changed_second_record_does_not_rewrite_unchanged_first_retention():
    scope = fixture_scope("assessment-repeated-unicode-units-defective", "episode")
    sources = scope.canonical_sources
    assert len(sources) == 2
    body = neutral(scope)
    checks = retention(scope, facet="material_facts")
    witness = next(f for f in scope.fields if f.path == "/candidate_body")
    body[checks[0].check_id] = ["retained", [witness.field_id]]
    body[checks[1].check_id] = ["altered", [witness.field_id]]
    first = parse(scope, body)
    # A different judgment on the second record must not be copied to the first.
    body[checks[1].check_id] = ["uncertain", []]
    second = parse(scope, body)
    selected = lambda outcome: next(j for j in outcome.judgments if j.check_id == checks[0].check_id)
    assert selected(first) == selected(second)
    assert selected(first).verdict == "retained"
    assert not first.model_retention_satisfied and not second.model_retention_satisfied


def test_order_facet_can_bind_both_step_fields_without_swapping_them():
    scope = fixture_scope("assessment-cross-field-order-defective", "procedure")
    steps = tuple(f for f in scope.fields if f.path.endswith("/action"))
    assert len(steps) == 2
    body = neutral(scope)
    check = retention(scope, facet="ordering")[0]
    body[check.check_id] = ["altered", [f.field_id for f in steps]]
    result = parse(scope, body)
    judgment = next(j for j in result.judgments if j.check_id == check.check_id)
    assert judgment.witness_fields == steps and judgment.facet == "ordering"
    assert not result.model_no_defect


def test_empty_candidate_still_has_every_source_obligation():
    value = packet()
    value["summary_item"]["candidate_summary"] = ""
    value["summary_item"]["candidate_is_noop"] = False
    scope = prepare(value).requests[-1]
    assert scope.fields == () and len(scope.checks) == 3 * len(scope.canonical_sources)
    body = {check.check_id: ["omitted", []] for check in scope.checks}
    result = parse(scope, body)
    assert result.review_structure_valid and not result.model_retention_satisfied
    assert not result.model_no_defect
    assert parse(scope, {}).status == "malformed_review"


def test_not_applicable_is_not_proven_absence_of_material_source_facts():
    scope = fixture_scope("assessment-prohibition-omission-defective", "procedure")
    # This intentionally incorrect model judgment remains structurally valid.
    # The parser cannot prove applicability, and must never claim that it did.
    result = parse(scope, neutral(scope))
    assert result.review_structure_valid and result.model_retention_satisfied
    assert result.model_no_defect
    assert not result.semantic_verified and not result.publication_authorized
    assert not hasattr(result, "model_all_supported")


def test_extra_checks_cannot_escape_check_cap_via_smaller_base_schedule():
    plan = prepare()
    old_max = max(len(s.base_scope.checks) for s in plan.requests)
    new_max = max(len(s.checks) for s in plan.requests)
    assert new_max > old_max
    with pytest.raises(ValueError):
        prepare(max_checks=old_max)
    scope = max(plan.requests, key=lambda s: len(s.checks))
    with pytest.raises(ValueError):
        review.parse_source_review(json.dumps(neutral(scope)), scope, max_checks=old_max)
    forged = replace(plan, max_checks=old_max)
    client = Client([])
    with pytest.raises(ValueError):
        review.execute_source_review(forged, client)
    assert client.calls == []


def test_retention_witness_count_obeys_same_reference_budget():
    scope = prepare(max_evidence_per_check=1).requests[0]
    body = neutral(scope)
    body[retention(scope)[0].check_id] = ["retained", [f.field_id for f in scope.fields[:2]]]
    assert parse(scope, body).status == "malformed_review"
