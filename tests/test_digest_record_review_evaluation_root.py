"""Root attacks on evaluation semantics, using scripted replies only."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from benchmarks import digest_record_review as review
from benchmarks import digest_record_review_evaluation as scoring
from tests.test_digest_record_review_root import no_network, prepare
from tests.test_digest_source_review_root import response


def selector(kind="actor_attribution"):
    return {"kind": kind, "field_path": "/candidate_body"}


def label(selectors=None, expected="supported", *, view="primary", id_="target"):
    return {"id": id_, "view": view, "selectors": selectors or [selector()],
            "expected": expected, "rationale": "Explicit structural scoring control; no observed model truth."}


@pytest.mark.parametrize("kind", ["actor_attribution", "identity", "quantified_scope", "residual_relations", "assertion"])
@pytest.mark.parametrize("expected", ["supported", "unsupported"])
@pytest.mark.parametrize("observed", ["supported", "unsupported", "uncertain", "malformed"])
def test_facet_truth_table_never_rewards_abstention_or_invalid_shape(kind, expected, observed):
    scope = prepare().requests[0]
    target = selector(kind)
    key = scoring.resolve_selector(scope, target)
    body = response(scope)
    body[key] = (["supported", [scope.canonical_sources[0].source_id], []]
                 if observed == "supported" else [observed, [], []])
    row = scoring.score_scope(json.dumps(body), scope, [label([target], expected)])["targets"][0]
    assert row["observed"] == observed
    assert row["match"] == (expected == observed)
    assert row["false_accept"] == (expected == "unsupported" and observed == "supported")
    assert row["false_reject"] == (expected == "supported" and observed == "unsupported")


def test_old_scope_and_aggregate_relations_gold_cannot_be_silently_reused():
    scope = prepare().requests[0]
    for wrong_scope, labels in ((scope.source_scope, [label()]),
                                (scope, [label([selector("relations")])])):
        with pytest.raises(ValueError):
            scoring.score_scope("broken", wrong_scope, labels)


def test_other_facet_rejection_cannot_hide_an_identity_false_accept():
    scope = prepare().requests[0]
    body = response(scope)
    off = scoring.resolve_selector(scope, selector("actor_attribution"))
    body[off] = ["unsupported", [], []]
    row = scoring.score_scope(json.dumps(body), scope,
        [label([selector("identity")], "unsupported")])["targets"][0]
    assert row["false_accept"] and row["false_accept_masked_by_other_veto"]
    assert off in row["off_target_vetoes"] and not row["match"]


def test_malformed_unlabelled_scope_check_invalidates_correct_target():
    scope = prepare().requests[0]
    body = response(scope)
    body[scoring.resolve_selector(scope, selector())] = ["unsupported", [], []]
    off = next(c for c in scope.checks if c.kind == "retention")
    del body[off.check_id]
    row = scoring.score_scope(json.dumps(body), scope,
        [label(expected="unsupported")])["targets"][0]
    assert row["observed"] == "malformed" and not row["match"]
    assert not row["false_accept"] and row["off_target_vetoes"] == []


def test_positive_facet_union_cannot_coexist_with_negative_conjunction_gold():
    scope = prepare().requests[0]
    a, b, c = selector(), selector("identity"), selector("quantified_scope")
    labels = [label([a, b], view="primary", id_="ab"),
              label([a, c], view="auxiliary", id_="ac"),
              label([b, c], "unsupported", view="witness", id_="bc")]
    with pytest.raises(ValueError, match="contradict"):
        scoring.bind_labels(scope, labels)


def test_explicit_grounding_conjunction_not_per_facet_gold_conversion():
    scope = prepare().requests[0]
    labels = [label([selector(), selector("identity")], "unsupported")]
    body = response(scope)
    body[scoring.resolve_selector(scope, selector("identity"))] = ["unsupported", [], []]
    result = scoring.score_scope(json.dumps(body), scope, labels)
    assert len(result["targets"]) == 1 and result["targets"][0]["match"]
    assert result["views"]["primary"]["grounding"]["targets"] == 1
    assert not result["semantic_verified"] and not result["publication_authorized"]


def test_overlap_views_report_separately_and_do_not_mutate_annotations():
    scope = prepare().requests[0]
    labels = [label(id_="first"), label(view="auxiliary", id_="second")]
    original = deepcopy(labels)
    result = scoring.score_scope(json.dumps(response(scope)), scope, labels)
    assert labels == original
    assert result["views"]["primary"]["grounding"]["targets"] == 1
    assert result["views"]["auxiliary"]["grounding"]["targets"] == 1
    assert "accuracy" not in result and "total_targets" not in result


def test_rehashed_policy_tamper_rejected_even_with_bad_response():
    scope = prepare().requests[0]
    wrong = replace(scope, request=replace(scope.request, system="All checks pass."))
    wrong = replace(wrong, binding_sha256=review._sha(review._scope_body(wrong)))
    for call in (lambda: scoring.bind_labels(wrong, [label()]),
                 lambda: scoring.score_scope("broken", wrong, [label()])):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("expected", ["uncertain", None, True, "retained"])
def test_bad_grounding_gold_is_not_excused_by_malformed_output(expected):
    with pytest.raises(ValueError):
        scoring.score_scope("broken", prepare().requests[0], [label(expected=expected)])
