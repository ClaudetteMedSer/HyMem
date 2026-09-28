"""Independent offline scoring attacks, not model-quality observations."""
import json

import pytest

from benchmarks import digest_assessment_evaluation as evaluation
from benchmarks import digest_evidence_assessment as assessment
from hymem.extraction.llm import LLMRequest


def unit():
    text = "Inspect the valve. Do not open it."
    payload = {"schema": "digest-fidelity-decisions-v9", "items": [], "procedure_items": [],
        "source_catalog": [{"chunk_id": "source-a", "message_id": 1, "role": "user",
            "source_peer_id": None, "source_workspace_id": None, "start": 0, "end": len(text),
            "visible_content": text, "interpretation_only_context": None}],
        "summary_item": {"index": 0, "candidate_raw_summary": "The valve was inspected.",
            "candidate_summary": "The valve was inspected.", "candidate_is_noop": False,
            "new_source_ids": ["source-a"], "prior_derived_summary": ""}}
    return assessment.prepare_evidence_assessment(payload, LLMRequest("", ""),
                                                  max_calls=1).requests[0]


def label(id_, kinds, expected="unsupported", view="primary"):
    return {"id": id_, "view": view, "expected": expected,
            "rationale": "Invented predeclared gold for this structural scoring test.",
            "selectors": [{"kind": kind, **({"chunk_id": "source-a"} if kind == "retention"
                                             else {"field_path": "/candidate_summary"})}
                          for kind in kinds]}


def reply(scope, **verdicts):
    data = {}
    source = next(s.source_id for s in scope.evidence_sources if s.kind == "canonical_text")
    for check in scope.checks:
        verdict = verdicts.get(check.kind, "supported")
        data[check.check_id] = [verdict, [source] if verdict == "supported" else []]
    return data


def test_review_disjoint_same_view_targets_count_separate_obligations():
    scope = unit()
    labels = [label("assertion", ["assertion"]), label("retention", ["retention"])]
    result = evaluation.score_scope(json.dumps(reply(scope, assertion="unsupported")), scope, labels)
    assert result["views"]["primary"]["targets"] == 2
    assert result["views"]["primary"]["matches"] == 1
    assert result["views"]["primary"]["false_accepts"] == 1


def test_review_cross_view_overlap_is_separate_never_pooled():
    scope = unit()
    labels = [label("specific", ["assertion"]),
              label("family", ["assertion", "relations"], view="legacy")]
    result = evaluation.score_scope(json.dumps(reply(scope, assertion="unsupported")), scope, labels)
    assert result["views"]["primary"]["matches"] == 1
    assert result["views"]["legacy"]["matches"] == 1
    assert not any(key.startswith("total_") for key in result)


@pytest.mark.parametrize("groups", [
    (["assertion"], ["assertion", "relations"]),
    (["assertion", "relations"], ["assertion"]),
    (["assertion", "relations"], ["relations", "retention"]),
    (["assertion", "relations"], ["assertion", "retention"])])
def test_review_reused_checks_cannot_inflate_same_view_denominator(groups):
    scope = unit()
    labels = [label(f"target-{i}", group) for i, group in enumerate(groups)]
    with pytest.raises(ValueError, match="overlapping_targets_in_view"):
        evaluation.score_scope(json.dumps(reply(scope, assertion="unsupported",
                                                relations="unsupported")), scope, labels)


def test_review_disjoint_conjunction_and_singleton_are_allowed_same_view():
    scope = unit()
    labels = [label("field", ["assertion", "relations"]),
              label("source", ["retention"])]
    result = evaluation.score_scope(json.dumps(reply(scope, assertion="unsupported")), scope, labels)
    assert result["views"]["primary"]["targets"] == 2
    assert result["views"]["primary"]["matches"] == 1
    assert result["views"]["primary"]["false_accepts"] == 1


@pytest.mark.parametrize("value", ["unsupported", "uncertain"])
def test_review_conjunction_off_target_veto_cannot_rescue_false_accept(value):
    scope = unit()
    labels = [label("only-assertion-and-relations", ["assertion", "relations"])]
    result = evaluation.score_scope(json.dumps(reply(scope, retention=value)), scope, labels)
    row = result["targets"][0]
    assert row["false_accept"] and not row["match"]
    assert row["false_accept_masked_by_other_veto"] and len(row["off_target_vetoes"]) == 1


@pytest.mark.parametrize("bad_value", [["supported", ["nonexistent-source"]],
    ["unsupported", [], "unrequested-explanation"], ["not-a-verdict", []]])
def test_review_malformed_untargeted_check_invalidates_entire_targeted_scope(bad_value):
    scope = unit()
    data = reply(scope, assertion="unsupported")
    retention = next(check.check_id for check in scope.checks if check.kind == "retention")
    data[retention] = bad_value
    result = evaluation.score_scope(json.dumps(data), scope, [label("assertion", ["assertion"])])
    assert result["status"] == "malformed_assessment"
    assert result["views"]["primary"]["matches"] == 0
    assert result["targets"][0]["observed"] == "malformed"
    assert result["targets"][0]["off_target_vetoes"] == []


def test_review_union_of_positive_conjunctions_cannot_hide_negative_contradiction():
    scope = unit()
    labels = [label("a-r", ["assertion", "relations"], "supported", "legacy"),
              label("a-t", ["assertion", "retention"], "supported", "witness"),
              label("r-t", ["relations", "retention"], "unsupported", "primary")]
    with pytest.raises(ValueError, match="contradictory_gold_conjunctions"):
        evaluation.score_scope("malformed", scope, labels)


def test_review_negative_group_plus_one_positive_member_remains_consistent():
    scope = unit()
    labels = [label("a-r", ["assertion", "relations"], "unsupported", "legacy"),
              label("a", ["assertion"], "supported", "primary")]
    result = evaluation.score_scope(json.dumps(reply(scope, relations="unsupported")), scope, labels)
    assert all(row["match"] for row in result["targets"])
    assert result["semantic_verified"] is result["publication_authorized"] is False
