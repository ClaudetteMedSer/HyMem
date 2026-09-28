"""Fixture binding and accounting, not semantic accuracy from model calls."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from benchmarks import digest_record_review as review
from benchmarks import digest_record_review_evaluation as evaluation
from hymem.extraction.llm import LLMRequest
from tests.digest_record_review_evaluation_fixtures import build_cases
from tests.digest_source_review_evaluation_fixtures import build_cases as old_cases
from tests.test_digest_record_review_root import no_network
from tests.test_digest_record_review_fixtures import changed_paths
from tests.test_digest_source_review_root import response


TEMPLATE = LLMRequest("", "", temperature=0.0, max_tokens=3072)


def selected(case):
    plan = review.prepare_record_review(case["payload"], TEMPLATE, max_calls=3)
    return next(s for s in plan.requests
        if {"kind": s.kind, "index": s.index} == case["target_scope"])


def test_cohorts_and_pairs_are_explicit_not_pooled_with_development_cases():
    cases = build_cases()
    assert len(cases) == 30 and len({c["id"] for c in cases}) == 30
    assert Counter(c["cohort"] for c in cases) == {"fresh": 24, "replay": 6}
    assert len({c["pair"] for c in cases}) == 15
    assert Counter(c["target_scope"]["kind"] for c in cases) == {
        "episode": 24, "procedure": 4, "summary": 2}
    for name in {c["pair"] for c in cases}:
        a, b = [c for c in cases if c["pair"] == name]
        assert {a["variant"], b["variant"]} == {"faithful", "defective"}
        assert changed_paths(a["payload"], b["payload"]) == set(a["changed_payload_paths"])
        assert a["changed_payload_paths"] == b["changed_payload_paths"]
        assert a["faithful_scope_acceptance"] and a["faithful_scope_rationale"]
        assert not b["faithful_scope_acceptance"] and b["faithful_scope_rationale"] is None


def test_replay_payloads_are_byte_equivalent_with_new_explicit_facet_labels():
    originals = {c["id"]: c for c in old_cases()}
    facets = {"boundary-speaker": "actor_attribution", "opaque-identity": "identity",
              "exclusivity-scope": "quantified_scope"}
    for case in build_cases():
        if case["cohort"] != "replay":
            continue
        name = case["pair"].removeprefix("replay-")
        old = originals[f"source-review-{name}-{case['variant']}"]
        assert review._canonical(case["payload"]) == review._canonical(old["payload"])
        assert case["labels"][0]["selectors"][0]["kind"] == facets[name]
        assert case["labels"] != old["labels"]


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c["id"])
def test_labels_bind_without_leaking_and_annotation_changes_do_not_change_request(case):
    scope = selected(case)
    bound = evaluation.bind_labels(scope, case["labels"])
    assert len(bound) == len(case["labels"])
    for label in case["labels"]:
        assert label["rationale"] not in scope.request.user
        assert all(s["kind"] != "relations" for s in label["selectors"])
    assert case["id"] not in scope.request.user
    assert case["pair"] not in scope.request.user
    altered = deepcopy(case)
    altered["labels"] = []
    altered["variant"] = "annotation-only-change"
    assert selected(altered) == scope
    assert scope.request.temperature == 0.0 and scope.request.max_tokens == 3072
    assert scope.request.response_format == "json"
    # Exact exclusions have selectors and reasons, not anonymous missing gold.
    for excluded in case["excluded_checks"]:
        assert excluded["reason"]
        excluded_id = evaluation.resolve_selector(scope, excluded["selector"])
        assert all(excluded_id not in row["check_ids"] for row in bound)


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c["id"])
def test_scripted_targets_score_exactly_without_claiming_unlabelled_truth(case):
    scope = selected(case)
    body = response(scope)
    for row in evaluation.bind_labels(scope, case["labels"]):
        assert len(row["check_ids"]) == 1
        key, expected = row["check_ids"][0], row["expected"]
        if row["domain"] == "retention":
            body[key] = [expected, [scope.fields[0].field_id]
                         if expected in {"retained", "altered"} else []]
        else:
            body[key] = [expected, [scope.canonical_sources[0].source_id]
                         if expected == "supported" else [], []]
    result = evaluation.score_scope(json.dumps(body), scope, case["labels"])
    assert all(row["match"] for row in result["targets"])
    assert not result["semantic_verified"] and not result["publication_authorized"]


def test_fresh_builds_are_unshared_and_json_serializable():
    a, b = build_cases(), build_cases()
    a[0]["payload"]["source_catalog"][0]["role"] = "system"
    assert b == build_cases() and a != b
    assert json.loads(json.dumps(b)) == b


def test_selected_scope_and_original_scope_counts_are_different_on_purpose():
    plans = [review.prepare_record_review(c["payload"], TEMPLATE, max_calls=3)
             for c in build_cases()]
    assert sum(len(p.requests) for p in plans) == 58
    assert len([selected(c) for c in build_cases()]) == 30
    assert all(review._preflight(p) == p for p in plans)


def test_completed_inspection_pair_does_not_mislabel_its_outcome_as_informational():
    cases = [c for c in build_cases() if c["pair"] == "explicit-universal-support"]
    assert len(cases) == 2
    for case in cases:
        item = case["payload"]["items"][0]
        assert item["candidate_outcome"] == "resolved"
        assert "All four passed inspection." in case["payload"]["source_catalog"][0]["visible_content"]
        scope = selected(case)
        outcome = next(f for f in scope.fields if f.path == "/candidate_outcome")
        assert outcome.text == "resolved"
        assert any(c.kind == "outcome" and c.field_ids == (outcome.field_id,)
                   for c in scope.checks)
    # This correction is common to both variants, not a second manipulated axis.
    assert changed_paths(*(c["payload"] for c in cases)) == {"/items/0/candidate_title"}


def test_reported_intention_preserves_reporting_not_just_the_plan():
    faithful, defective = [c for c in build_cases() if c["pair"] == "reported-intentions"]
    assert faithful["payload"]["items"][0]["candidate_body"] == (
        "Mara told the assistant about Mara's plan to repair the mast; "
        "the assistant plans only to deliver paint.")
    assert defective["payload"]["items"][0]["candidate_body"] == (
        "The assistant told Mara about the assistant's plan to repair the mast; "
        "Mara plans only to deliver paint.")
    for case in (faithful, defective):
        record = case["payload"]["source_catalog"][0]
        assert record["role"] == "assistant"
        assert record["visible_content"] == (
            'Mara told me, "I plan to repair the mast." I plan only to deliver paint.')
    assert changed_paths(faithful["payload"], defective["payload"]) == {
        "/items/0/candidate_body"}


@pytest.mark.parametrize("case", build_cases(), ids=lambda c: c["id"])
def test_all_gold_cohort_and_acceptance_annotations_are_outside_requests(case):
    scope = selected(case)
    marker = "REVIEW_ONLY_ANNOTATION_NOT_MODEL_INPUT"
    altered = deepcopy(case)
    for key in set(case) - {"payload", "target_scope"}:
        altered[key] = marker
    assert selected(altered) == scope
    assert marker not in scope.request.user and marker not in scope.request.system
    assert case["faithful_scope_rationale"] is None or (
        case["faithful_scope_rationale"] not in scope.request.user)


def test_source_coordinates_do_not_encode_mechanism_or_gold():
    for case in build_cases():
        for record in case["payload"]["source_catalog"]:
            assert record["chunk_id"] == f"unit-{record['message_id']}"
        assert case["variant"] not in selected(case).request.user


def test_replayed_identity_cannot_import_prior_summary_as_episode_evidence():
    for case in build_cases():
        if case["pair"] != "replay-opaque-identity":
            continue
        assert case["payload"]["summary_item"]["prior_derived_summary"] == (
            "Neris previously discussed gardens.")
        scope = selected(case)
        assert scope.kind == "episode" and not scope.prior_summary_sources
        assert "Neris previously discussed gardens." not in scope.request.user


def test_incidental_summary_control_is_not_an_unlabelled_design_specification():
    faithful, defective = [c for c in build_cases() if c["pair"] == "material-decision"]
    decision = "The final design decision selected a circular dial for instrument X-74."
    incidental = "The meeting agenda was printed in silver ink."
    for case in (faithful, defective):
        assert case["payload"]["source_catalog"][0]["visible_content"] == f"{incidental} {decision}"
        assert case["labels"][1]["expected"] == "supported"
    assert faithful["payload"]["summary_item"]["candidate_summary"] == decision
    assert defective["payload"]["summary_item"]["candidate_summary"] == incidental
    assert faithful["labels"][0]["expected"] == "retained"
    assert defective["labels"][0]["expected"] == "omitted"
