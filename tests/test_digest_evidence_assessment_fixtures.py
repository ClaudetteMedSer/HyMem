"""Offline fixture/gold integrity only; no scripted answer proves model accuracy."""
from collections import Counter
from copy import deepcopy
import json

import pytest

from benchmarks.digest_evidence_assessment import prepare_evidence_assessment
from hymem.extraction.llm import LLMRequest
from tests.digest_evidence_assessment_fixtures import build_cases


CASES = build_cases()
UNSCORED_RETENTION = {
    "assessment-causal-suffix-defective": "causal-nerys",
    "assessment-prior-only-identity-defective": "identity-zevin",
    "assessment-categorical-outcome-defective": "deferred-antenna",
}


def _plan(case):
    return prepare_evidence_assessment(case["payload"],
        LLMRequest("unused system", "unused user", "json", 4096, 0.0), max_calls=3)


def _target(case):
    matches = [scope for scope in _plan(case).requests
               if (scope.kind, scope.index) == (case["target_scope"]["kind"], case["target_scope"]["index"])]
    assert len(matches) == 1
    return matches[0]


def _resolve(scope, label):
    selector = label["selector"]
    if selector["kind"] == "retention":
        assert set(selector) == {"kind", "chunk_id"}
        sources = [source.source_id for source in scope.evidence_sources
                   if source.kind == "canonical_text" and source.chunk_id == selector["chunk_id"]]
        assert len(sources) == 1
        matches = [check for check in scope.checks
                   if check.kind == "retention" and check.source_id == sources[0]]
    else:
        assert set(selector) == {"kind", "field_path"}
        fields = [field.field_id for field in scope.fields if field.path == selector["field_path"]]
        assert len(fields) == 1
        matches = [check for check in scope.checks
                   if check.kind == selector["kind"] and check.field_ids == tuple(fields)]
    assert len(matches) == 1
    return matches[0]


def _changes(left, right, path=""):
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        assert set(left) == set(right)
        return [item for key in sorted(left) for item in _changes(left[key], right[key], path + "/" + key)]
    if isinstance(left, list):
        assert len(left) == len(right)
        return [item for index, (a, b) in enumerate(zip(left, right, strict=True))
                for item in _changes(a, b, path + "/" + str(index))]
    return [] if left == right else [path]


def test_deterministic_balanced_independent_cases():
    assert CASES == build_cases()
    assert len(CASES) == len({case["id"] for case in CASES}) == 16
    assert Counter(case["pair"] for case in CASES) == {case["pair"]: 2 for case in CASES}
    assert Counter(case["primary"]["expected"] for case in CASES) == {"supported": 8, "unsupported": 8}
    mutated = build_cases()
    mutated[0]["payload"]["source_catalog"][0]["visible_content"] = "changed"
    mutated[0]["retention_labels"][0]["expected"] = "uncertain"
    assert build_cases() == CASES
    assert mutated[1] == CASES[1]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_strict_prepare_and_all_selected_gold_resolve_once(case):
    scope = _target(case)
    labels = [case["primary"], *case["retention_labels"], *case["auxiliary_labels"]]
    seen = {}
    for label in labels:
        assert set(label) == {"selector", "expected", "rationale"}
        if label["expected"] is None:
            assert case["id"] in UNSCORED_RETENTION
            assert label["selector"] == {
                "kind": "retention", "chunk_id": UNSCORED_RETENTION[case["id"]]}
            assert label in case["retention_labels"]
            assert "Unscored before observing model outputs" in label["rationale"]
        else:
            assert label["expected"] in {"supported", "unsupported"}
        assert len(label["rationale"]) > 30
        check = _resolve(scope, label)
        if check.check_id in seen:
            assert label == seen[check.check_id]
        seen[check.check_id] = label
    assert case["primary"]["expected"] == ("supported" if case["variant"] == "faithful" else "unsupported")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_complete_canonical_retention_coverage_only(case):
    scope = _target(case)
    expected = {source.chunk_id for source in scope.evidence_sources if source.kind == "canonical_text"}
    labels = case["retention_labels"]
    assert {label["selector"]["chunk_id"] for label in labels} == expected
    assert len(labels) == len(expected)
    assert all(label["selector"]["kind"] == "retention" for label in labels)
    assert {_resolve(scope, label).check_id for label in labels} == {
        check.check_id for check in scope.checks if check.kind == "retention"}


@pytest.mark.parametrize("offset", range(0, 16, 2))
def test_pair_changes_exactly_documented_and_sources_remain_identical(offset):
    faithful, defective = CASES[offset:offset + 2]
    assert faithful["variant"] == "faithful" and defective["variant"] == "defective"
    assert faithful["pair"] == defective["pair"]
    assert faithful["target_scope"] == defective["target_scope"]
    assert faithful["primary"]["selector"] == defective["primary"]["selector"]
    assert faithful["payload"]["source_catalog"] == defective["payload"]["source_catalog"]
    assert faithful["changed_payload_paths"] == defective["changed_payload_paths"]
    assert sorted(_changes(faithful["payload"], defective["payload"])) == sorted(faithful["changed_payload_paths"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_gold_never_enters_request_and_source_offsets_remain_exact(case):
    before = deepcopy(case)
    for scope in _plan(case).requests:
        packet = json.loads(scope.request.user)
        assert set(packet) == {"schema", "scope", "candidate", "fields", "evidence_sources", "checks"}
        for label in [case["primary"], *case["retention_labels"], *case["auxiliary_labels"]]:
            assert label["rationale"] not in scope.request.user
        for key in ("primary", "retention_labels", "auxiliary_labels", "expected", "rationale"):
            assert key not in packet
    for source in case["payload"]["source_catalog"]:
        assert source["start"] == 0
        assert source["end"] == len(source["visible_content"])
    assert case == before


@pytest.mark.parametrize("name", ["answered-outcome-omission", "prohibition-omission", "incidental-versus-material"])
def test_omission_controls_preserve_separately_labeled_true_assertions(name):
    pair = [case for case in CASES if case["pair"] == name]
    assert len(pair) == 2
    assert all(case["primary"]["selector"]["kind"] == "retention" for case in pair)
    assert all(case["auxiliary_labels"] for case in pair)
    assert all(label["expected"] == "supported" for case in pair for label in case["auxiliary_labels"])
    assert all(any(label["selector"]["kind"] == "assertion" for label in case["auxiliary_labels"]) for case in pair)


def test_prior_identity_authority_stays_outside_episode_projection():
    faithful = next(case for case in CASES if case["id"] == "assessment-prior-only-identity-faithful")
    scope = _target(faithful)
    assert "Zevin" not in scope.request.user
    assert not any(source.kind == "prior_summary" for source in scope.evidence_sources)
    assert "Zevin" in faithful["payload"]["summary_item"]["prior_derived_summary"]


def test_unicode_sources_are_repeated_but_independently_addressable():
    faithful = next(case for case in CASES if case["id"] == "assessment-repeated-unicode-units-faithful")
    scope = _target(faithful)
    canonical = [source for source in scope.evidence_sources if source.kind == "canonical_text"]
    assert len(canonical) == 2
    assert len({source.source_id for source in canonical}) == 2
    assert all(source.text.count("Café check complete.") == 2 for source in canonical)
    assert "Ω-17 status: closed" in canonical[0].text
    assert "Ω-18 status: pending" in canonical[1].text


def test_order_pair_changes_cross_field_sequence_not_numeric_order():
    pair = [case for case in CASES if case["pair"] == "cross-field-order"]
    for case in pair:
        assert [step["order"] for step in case["payload"]["procedure_items"][0]["candidate"]["steps"]] == [1, 2]
        assert case["primary"]["selector"] == {"kind": "relations", "field_path": "/candidate/steps/0/action"}


def test_only_three_explicit_retention_ambiguities_are_unscored():
    excluded = {}
    scored_retention = []
    for case in CASES:
        assert case["primary"]["expected"] is not None
        assert all(label["expected"] is not None for label in case["auxiliary_labels"])
        for label in case["retention_labels"]:
            if label["expected"] is None:
                assert case["id"] not in excluded
                excluded[case["id"]] = label["selector"]["chunk_id"]
                assert "neither a credited success nor a credited failure" in label["rationale"]
            else:
                scored_retention.append(label)
    assert excluded == UNSCORED_RETENTION
    assert len(scored_retention) == 17
    assert all(label["expected"] in {"supported", "unsupported"} for label in scored_retention)
    # The label is explicit data, not inferred from the faithful/defective name:
    # some defective cases still preserve a different canonical unit correctly.
    assert any(case["variant"] == "defective" and label["expected"] == "supported"
               for case in CASES for label in case["retention_labels"])
