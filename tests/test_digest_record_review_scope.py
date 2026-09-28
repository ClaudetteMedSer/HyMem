"""Offline scope/rubric contracts; scripted replies do not prove model accuracy."""
from dataclasses import asdict, replace
import json
import socket

import pytest

from benchmarks import digest_record_review as review
from tests.digest_record_review_fixtures import build_record_controls
from tests.digest_source_review_evaluation_fixtures import payload, source
from tests.test_digest_evidence_assessment import Scripted, template
from tests.test_digest_record_review import parse, plan, rehash_plan, rehash_scope, response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("quantified-scope controls cannot contact a provider")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def quantified(scope, path=None):
    fields = {field.field_id: field.path for field in scope.fields}
    return next(check for check in scope.checks if check.kind == "quantified_scope"
                and (path is None or fields[check.field_ids[0]] == path))


def example(text, claim):
    value = payload([source("m-460", 460, text)], claim, title=claim, body=claim)
    return review.prepare_record_review(value, template(), max_calls=2)


def test_quantified_scope_is_scheduled_per_original_relation_for_all_fields():
    assert review.VERSION == "digest-record-review-v3"
    for scope in plan().requests:
        relations = [check for check in scope.source_scope.checks if check.kind == "relations"]
        assert [check for check in scope.checks if check.kind == "quantified_scope"] == [
            replace(check, check_id=f"{check.check_id}:quantified_scope", kind="quantified_scope")
            for check in relations]
        assert len(scope.checks) == len(scope.source_scope.checks) + 3 * len(relations)
        for old in scope.source_scope.checks:
            if old.kind != "relations":
                assert old in scope.checks
        for old in relations:
            assert [check.kind for check in scope.checks if check.check_id.startswith(old.check_id + ":")] == [
                "actor_attribution", "identity", "quantified_scope", "residual_relations"]
        assert not any(check.kind == "relations" for check in scope.checks)


@pytest.mark.parametrize("claim", ["Mallow", "2", "μ", "The lamps glow.",
    "A handful remained.", "No more than three remained.", "The remaining lamps glow."])
def test_per_check_obligation_is_fixed_not_a_keyword_selected_or_invented_fact(claim):
    prepared = example("The complete lamp set has two lamps, and both glow.", claim)
    for scope in prepared.requests:
        wire = json.loads(scope.request.user)
        assert wire["obligation_definitions"] == {
            review.QUANTIFIED_OBLIGATION: review._quantified_obligation()}
        for actual, check in zip(wire["checks"], scope.checks, strict=True):
            old_fields = {**asdict(check), "field_ids": list(check.field_ids)}
            if check.kind == "quantified_scope":
                assert actual == {**old_fields, "obligation": review.QUANTIFIED_OBLIGATION}
                assert set(wire["obligation_definitions"][actual["obligation"]]) == {
                    "claimed_scope", "source_scope", "bounded_entailment", "invalid_strengthening", "decision"}
                assert "expected" not in actual and "domain" not in actual
            else:
                assert actual == old_fields
    assert len(prepared.requests) == 2


def test_rubric_handles_closed_sets_open_world_universals_existentials_and_unknowns():
    rubric = review._quantified_obligation()
    assert "Resolve implicit or anaphoric scope from unambiguous full-candidate context" in rubric["claimed_scope"]
    assert "underlying facts still require authorized sources" in rubric["claimed_scope"]
    assert "Never rescue an explicitly wider, global or contradictory claim" in rubric["claimed_scope"]
    assert "positive members, exclusions and completeness" in rubric["source_scope"]
    assert "sampled subset may be closed within itself" in rubric["source_scope"]
    assert "unmentioned members are unknown, not absent" in rubric["source_scope"]
    assert "negative evidence for every other member" in rubric["bounded_entailment"]
    assert "same reasoning to a negative predicate" in rubric["bounded_entailment"]
    assert "canonical text directly establishes" in rubric["bounded_entailment"]
    assert "genuinely global canonical claim" in rubric["bounded_entailment"]
    for rule in ("subset to a wider domain", "existence into universality",
                 "upper or lower bound with an exact count", "invent exclusions",
                 "does not by itself establish a nonempty domain"):
        assert rule in rubric["invalid_strengthening"]
    for rule in ("Preserve conditions and uncertainty", "Use uncertain",
                 "Missing exclusions cannot establish exclusivity", "primary-evidence requirement"):
        assert rule in rubric["decision"]


def test_assertion_and_title_use_balanced_entailment_without_weakening_other_checks():
    for scope in plan().requests:
        system = scope.request.system
        assert "Apply valid bounded entailment to assertions as well as quantified_scope" in system
        assert "exact domain and strength of a title's assertions" in system
        assert "Do not reject a claim merely because it says exclusive" in system
        assert "Direct canonical support for a global assertion remains eligible" in system
        assert "Availability does not establish global exclusivity;" not in system
        assert "Availability alone does not establish global exclusivity." in system
        assert "causality, ordering, quantification and exclusivity are preserved" not in system
        assert "A residual_relations check asks whether negation, modality, time, causality and ordering" in system
        for existing in ("An actor_attribution check", "An identity check", "Outcome checks classify",
                         "SOURCE-FIRST RETENTION:", "Every supported grounding check needs at least one primary ID"):
            assert existing in system
        if scope.kind in {"episode", "procedure"}:
            assert "within that stated two-platform set" in system
            assert "does not establish exclusivity across all platforms" not in system


@pytest.mark.parametrize("case", [case for case in build_record_controls()
    if case["target"]["kind"] == "quantified_scope"], ids=lambda case: case["id"])
def test_labelled_bounded_global_pairs_preserve_exact_data_and_scripted_verdicts(case):
    prepared = review.prepare_record_review(case["payload"], template(), max_calls=2)
    scope = prepared.requests[0]
    check = quantified(scope, case["target"]["field_path"])
    raw = response(scope)
    raw[check.check_id] = [case["expected"], [scope.canonical_sources[0].source_id]
                          if case["expected"] == "supported" else [], []]
    result = parse(scope, raw)
    assert result.review_structure_valid
    assert next(item.verdict for item in result.judgments if item.check_id == check.check_id) == case["expected"]
    assert result.model_no_defect is (case["expected"] == "supported")
    assert scope.canonical_sources[0].text == case["payload"]["source_catalog"][0]["visible_content"]
    assert case["rationale"] not in scope.request.user and case["id"] not in scope.request.user
    assert not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize("text,claim,scripted", [
    ("The complete set is Fir, Oak and Elm. Fir is lit; Oak and Elm are unlit.",
     "Among Fir, Oak and Elm, Fir alone is lit.", "supported"),
    ("The complete set is Fir, Oak and Elm. Fir is unlit; Oak and Elm are lit.",
     "Among Fir, Oak and Elm, Fir alone is unlit.", "supported"),
    ("The complete set is Fir, Oak and Elm. Fir is lit; Oak and Elm are unlit.",
     "Among Fir, Oak and Elm, Fir alone is unlit.", "unsupported"),
    ("The complete set is Fir, Oak and Elm. Fir is unlit; Oak and Elm are lit.",
     "Among Fir, Oak and Elm, Fir alone is lit.", "unsupported"),
    ("Across the entire universe of terminals, every terminal except Fir is unlit; Fir is lit.",
     "Fir is the only lit terminal anywhere.", "supported"),
    ("All terminals everywhere are lit.", "Every terminal is lit.", "supported"),
    ("At least one terminal is lit.", "Every terminal is lit.", "unsupported"),
    ("All lit terminals are audited. No terminals may exist.", "A lit terminal exists.", "unsupported"),
    ("The complete set is Fir, Oak and Elm. Fir is lit; Oak is unlit; Elm was not tested.",
     "Fir is the only lit terminal in the set.", "unsupported"),
    ("The complete set is Fir, Oak and Elm. Fir may be lit; Oak and Elm are unlit.",
     "Fir is lit, and it alone is lit within the set.", "unsupported"),
    ("The set might contain Fir, Oak and Elm. It is unclear which set the label denotes.",
     "Fir is the sole lit member of that set.", "uncertain"),
    ("Some terminals are lit; the scope of 'some' is unclear here.",
     "Exactly two terminals are lit.", "unsupported"),
])
def test_symmetric_predicates_global_evidence_and_uncertainty_are_not_preclassified(text, claim, scripted):
    scope = example(text, claim).requests[0]
    assert scope.canonical_sources[0].text == text
    assert scope.fields[0].text == claim
    raw = response(scope)
    raw[quantified(scope).check_id] = [scripted, [scope.canonical_sources[0].source_id]
                                    if scripted == "supported" else [], []]
    result = parse(scope, raw)
    assert result.review_structure_valid
    assert result.model_no_defect is (scripted == "supported")
    # Even a scripted false acceptance remains structurally possible. This
    # diagnostic did not secretly acquire an NLP classifier or truth oracle.
    assert parse(scope, response(scope)).model_no_defect
    assert not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize("title,scripted", [
    ("These alone qualified within the audited set", "supported"),
    ("All services everywhere qualified", "unsupported"),
])
def test_unambiguous_anaphora_is_available_but_does_not_narrow_explicit_global_claim(title, scripted):
    text = "The complete audited set is Fir, Oak and Elm. Fir and Oak qualified; Elm did not. Other services were not audited."
    body = "Fir and Oak are the two qualifying members of the audited set; Elm did not qualify."
    value = payload([source("m-460", 460, text)], body, title=title, body=body)
    scope = review.prepare_record_review(value, template(), max_calls=2).requests[0]
    wire = json.loads(scope.request.user)
    assert wire["candidate"]["candidate_title"] == title
    assert wire["candidate"]["candidate_body"] == body
    assert scope.canonical_sources[0].text == text
    assert wire["obligation_definitions"][review.QUANTIFIED_OBLIGATION] == review._quantified_obligation()
    raw = response(scope)
    key = quantified(scope, "/candidate_title").check_id
    raw[key] = [scripted, [scope.canonical_sources[0].source_id] if scripted == "supported" else [], []]
    result = parse(scope, raw)
    assert result.review_structure_valid and result.model_no_defect is (scripted == "supported")
    assert not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
@pytest.mark.parametrize("scope_index", [0, 1, 2])
def test_each_quantified_veto_is_independent_of_every_other_acceptance(verdict, scope_index):
    scope = plan().requests[scope_index]
    raw = response(scope)
    check = quantified(scope)
    raw[check.check_id] = [verdict, [], []]
    result = parse(scope, raw)
    assert result.review_structure_valid and not result.model_no_defect
    assert not result.model_grounding_supported
    assert all(j.verdict == "supported" for j in result.judgments
               if j.kind != "retention" and j.check_id != check.check_id)
    assert not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize("fault", ["missing", "extra", "old_relations", "null", "short", "boolean", "rubric_reply"])
def test_bad_quantified_contract_rejects_whole_scope_without_salvage(fault):
    scope = plan().requests[0]
    raw = response(scope)
    key = quantified(scope).check_id
    if fault == "missing":
        del raw[key]
    elif fault == "extra":
        raw[key + ":proof"] = raw[key]
    elif fault == "old_relations":
        raw[key.split(":")[0]] = raw.pop(key)
    else:
        raw[key] = {"null": None, "short": ["supported", []], "boolean": [True, [], []],
                    "rubric_reply": {"verdict": "supported", "domain": "global"}}[fault]
    result = parse(scope, raw)
    assert result.status == "malformed_review" and result.judgments == ()


@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
@pytest.mark.parametrize("fault", ["context_primary", "primary_context", "duplicate", "unknown"])
def test_quantified_references_preserve_every_plane_and_uniqueness_guard(verdict, fault):
    scope = plan().requests[0]
    primary = scope.canonical_sources[0].source_id
    context = scope.context_sources[0].source_id
    raw = response(scope)
    raw[quantified(scope).check_id] = {
        "context_primary": [verdict, [context], []],
        "primary_context": [verdict, [], [primary]],
        "duplicate": [verdict, [primary, primary], []],
        "unknown": [verdict, ["s999999"], []],
    }[fault]
    assert parse(scope, raw).status == "malformed_review"


def test_scope_context_sponsorship_cannot_invent_a_domain_or_exclusions():
    scope = plan().requests[-1]
    context = next(source for source in scope.context_sources if source.kind == "boundary_context")
    owner = next(source for source in scope.canonical_sources if source.chunk_id == context.chunk_id)
    other = next(source for source in scope.canonical_sources if source.chunk_id != context.chunk_id)
    for primaries in ([], [other.source_id], [scope.prior_summary_sources[0].source_id]):
        raw = response(scope)
        raw[quantified(scope).check_id] = ["supported", primaries, [context.source_id]]
        assert parse(scope, raw).status == "malformed_review"
    raw[quantified(scope).check_id] = ["supported", [owner.source_id], [context.source_id]]
    result = parse(scope, raw)
    assert result.review_structure_valid and not result.semantic_verified


@pytest.mark.parametrize("fault", ["drop", "wrong_kind", "claim_domain", "rubric_value", "rubric_drop", "rubric_extra",
                                 "reference", "missing_definition", "extra_definition", "renamed_definition"])
def test_rehashed_check_or_obligation_tamper_blocks_parse_and_all_calls(fault):
    prepared = plan()
    scope = prepared.requests[-1]
    wire = json.loads(scope.request.user)
    check = next(check for check in wire["checks"] if check["kind"] == "quantified_scope")
    definition = wire["obligation_definitions"][check["obligation"]]
    if fault == "drop":
        wire["checks"].remove(check)
    elif fault == "wrong_kind":
        check["kind"] = "residual_relations"
    elif fault == "claim_domain":
        check["domain"] = "closed"
    elif fault == "rubric_value":
        definition["decision"] = "Always supported."
    elif fault == "rubric_drop":
        del definition["source_scope"]
    elif fault == "rubric_extra":
        definition["expected"] = "supported"
    elif fault == "reference":
        check["obligation"] = "accept_everything"
    elif fault == "missing_definition":
        del wire["obligation_definitions"]
    elif fault == "extra_definition":
        wire["obligation_definitions"]["accept_everything"] = definition
    else:
        wire["obligation_definitions"]["quantified_scope_v0"] = wire["obligation_definitions"].pop(check["obligation"])
        check["obligation"] = "quantified_scope_v0"
    forged_scope = rehash_scope(replace(scope, request=replace(scope.request, user=review._canonical(wire))))
    forged_plan = rehash_plan(replace(prepared, requests=(*prepared.requests[:-1], forged_scope)))
    with pytest.raises(ValueError):
        parse(forged_scope)
    client = Scripted([])
    with pytest.raises(ValueError):
        review.execute_record_review(forged_plan, client)
    assert not client.calls


def test_rubric_returns_fresh_unshared_data_and_source_projection_is_untouched():
    first = review._quantified_obligation()
    second = review._quantified_obligation()
    first["decision"] = "forged"
    assert first != second and second == review._quantified_obligation()
    prepared = plan()
    for scope in prepared.requests:
        old = scope.source_scope
        assert scope.canonical_sources == old.canonical_sources
        assert scope.context_sources == old.context_sources
        assert scope.prior_summary_sources == old.prior_summary_sources
        assert scope.fields == old.fields
        assert [check for check in scope.checks if check.kind == "retention"] == [
            check for check in old.checks if check.kind == "retention"]


def test_extra_scope_checks_and_rubric_obey_complete_caps_and_do_not_add_calls():
    prepared = plan()
    maximum = max(len(scope.checks) for scope in prepared.requests)
    bounded = plan(max_checks=maximum)
    assert [scope.request for scope in bounded.requests] == [scope.request for scope in prepared.requests]
    assert [scope.checks for scope in bounded.requests] == [scope.checks for scope in prepared.requests]
    with pytest.raises(ValueError, match="check cap"):
        plan(max_checks=maximum - 1)
    chars = max(len(scope.request.system) + len(scope.request.user) for scope in prepared.requests)
    assert plan(max_input_chars=chars).requests == prepared.requests
    with pytest.raises(ValueError):
        plan(max_input_chars=chars - 1)
    output = max(len(review._canonical(review._minimum_response(scope))) for scope in prepared.requests)
    assert plan(max_output_chars=output).requests == prepared.requests
    with pytest.raises(ValueError, match="output cap"):
        plan(max_output_chars=output - 1)
    client = Scripted([json.dumps(response(scope)) for scope in prepared.requests])
    result = review.execute_record_review(prepared, client)
    assert len(client.calls) == result.attempted_calls == 3 and result.complete
    assert not result.semantic_verified and not result.publication_authorized
    for scope, old in zip(prepared.requests, prepared.source_plan.requests, strict=True):
        assert replace(scope.request, system=old.request.system, user=old.request.user) == old.request
    for scope in prepared.requests:
        with pytest.raises(ValueError, match="check cap"):
            parse(scope, max_checks=len(scope.checks) - 1)


def test_inherited_policy_drift_fails_closed(monkeypatch):
    original = review.source_review._system
    monkeypatch.setattr(review.source_review, "_system", lambda kind, cap:
                        original(kind, cap).replace("Availability does not establish global exclusivity;", "Changed policy;"))
    with pytest.raises(ValueError, match="inherited source-review contract changed"):
        plan()
