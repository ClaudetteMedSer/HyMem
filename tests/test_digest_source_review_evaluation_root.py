"""Independent scoring attacks. Scripted model judgments are not semantic evidence."""
from copy import deepcopy
import json
import socket

import pytest

from benchmarks import digest_source_review_evaluation as evaluation
from tests.test_digest_source_review_root import prepare, response


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("offline scorer controls prohibit networking")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def label(id_, selectors, expected, view="primary"):
    return {"id": id_, "view": view, "selectors": selectors,
            "expected": expected, "rationale": "Predeclared structural scoring control, not inferred model truth."}


def grounding(kind="assertion"):
    return {"kind": kind, "field_path": "/candidate_body"}


def retention(facet="material_facts", cid="a"):
    return {"kind": "retention", "chunk_id": cid, "facet": facet}


def score(scope, labels, data=None):
    return evaluation.score_scope(json.dumps(response(scope) if data is None else data), scope, labels)


@pytest.mark.parametrize("expected", ["retained", "omitted", "altered", "not_applicable"])
@pytest.mark.parametrize("observed", ["retained", "omitted", "altered", "not_applicable", "uncertain", "malformed"])
def test_exact_retention_status_and_error_accounting(expected, observed):
    scope = prepare().requests[0]
    target = label("retention", [retention()], expected)
    key = evaluation.resolve_selector(scope, retention())
    data = response(scope)
    if observed == "malformed":
        data[key] = ["retained", []]
    else:
        data[key] = [observed, [scope.fields[0].field_id] if observed in {"retained", "altered"} else []]
    result = score(scope, [target], data)
    row = result["targets"][0]
    assert row["domain"] == "retention"
    assert row["observed"] == observed
    assert row["match"] == (expected == observed)
    assert row["false_accept"] == (expected in {"omitted", "altered"} and observed in {"retained", "not_applicable"})
    assert row["false_reject"] == (expected in {"retained", "not_applicable"} and observed in {"omitted", "altered"})
    assert row["false_not_applicable"] == (expected != "not_applicable" and observed == "not_applicable")
    assert row["false_applicable"] == (expected == "not_applicable" and observed in {"retained", "omitted", "altered"})
    assert row["defect_kind_confusion"] == (expected in {"omitted", "altered"} and observed in {"omitted", "altered"} and observed != expected)
    bucket = result["views"]["primary"]["retention"]
    assert bucket["targets"] == 1 and bucket["matches"] == (expected == observed)
    assert not result["semantic_verified"] and not result["publication_authorized"]


def test_wrong_not_applicable_cannot_earn_positive_exact_match():
    scope = prepare().requests[0]
    key = evaluation.resolve_selector(scope, retention())
    data = response(scope)
    data[key] = ["not_applicable", []]
    row = score(scope, [label("applicable", [retention()], "retained")], data)["targets"][0]
    assert not row["match"] and row["false_not_applicable"]
    assert not row["false_accept"]  # Different error class from missing a real defect.


def test_views_and_domains_stay_separate_not_a_pooled_accuracy_number():
    scope = prepare().requests[0]
    labels = [label("g", [grounding()], "supported"),
              label("r", [retention()], "retained"),
              label("r-secondary", [retention()], "retained", view="retention")]
    result = score(scope, labels)
    assert set(result["views"]["primary"]) == {"grounding", "retention"}
    assert result["views"]["primary"]["grounding"]["targets"] == 1
    assert result["views"]["primary"]["retention"]["targets"] == 1
    assert result["views"]["retention"]["retention"]["targets"] == 1
    assert not any(k.startswith("total_") or k == "accuracy" for k in result)


@pytest.mark.parametrize("other", ["omitted", "altered", "uncertain"])
def test_off_target_rejection_cannot_rescue_target_false_accept(other):
    scope = prepare().requests[0]
    target = retention("constraints")
    off = evaluation.resolve_selector(scope, retention("ordering", "b"))
    data = response(scope)
    data[off] = [other, [scope.fields[0].field_id] if other == "altered" else []]
    row = score(scope, [label("missing-constraint", [target], "omitted")], data)["targets"][0]
    assert row["false_accept"] and not row["match"]
    assert row["false_accept_masked_by_other_veto"]
    assert off in row["off_target_vetoes"]


def test_malformed_untargeted_check_invalidates_even_a_correct_target():
    scope = prepare().requests[0]
    key = evaluation.resolve_selector(scope, grounding())
    off = evaluation.resolve_selector(scope, retention())
    data = response(scope)
    data[key] = ["unsupported", [], []]
    data[off] = ["omitted", [scope.fields[0].field_id]]
    result = score(scope, [label("actual-defect", [grounding()], "unsupported")], data)
    row = result["targets"][0]
    assert result["status"] == "malformed_review"
    assert row["observed"] == "malformed" and not row["match"]
    assert not row["false_accept"] and row["off_target_vetoes"] == []


@pytest.mark.parametrize("groups", [
    ([grounding()], [grounding(), grounding("relations")]),
    ([grounding(), grounding("relations")], [grounding()]),
])
def test_partial_same_view_overlap_rejected(groups):
    scope = prepare().requests[0]
    labels = [label(str(i), group, "unsupported") for i, group in enumerate(groups)]
    with pytest.raises(ValueError, match="overlap"):
        score(scope, labels)


@pytest.mark.parametrize("selectors", [[retention(), retention("constraints")], [grounding(), retention()]])
def test_retention_conjunction_or_mixed_domain_never_silently_coerced(selectors):
    scope = prepare().requests[0]
    with pytest.raises(ValueError):
        score(scope, [label("bad", selectors, "supported")])


@pytest.mark.parametrize("expected", [None, "uncertain", True, "supported", "unsupported"])
def test_ambiguous_or_wrong_domain_retention_gold_is_not_scored(expected):
    scope = prepare().requests[0]
    with pytest.raises(ValueError):
        evaluation.score_scope("broken", scope, [label("bad-gold", [retention()], expected)])


def test_retention_gold_conflict_across_separate_views_is_rejected():
    scope = prepare().requests[0]
    labels = [label("yes", [retention()], "retained"),
              label("no", [retention()], "not_applicable", view="retention")]
    with pytest.raises(ValueError, match="conflict"):
        score(scope, labels)


def test_grounding_conjunction_consistency_survives_positive_union_attack():
    scope = prepare().requests[0]
    a, b = grounding(), grounding("relations")
    c = {"kind": "assertion", "field_path": "/candidate_title"}
    labels = [label("ab", [a, b], "supported", "legacy"),
              label("ac", [a, c], "supported", "witness"),
              label("bc", [b, c], "unsupported", "primary")]
    with pytest.raises(ValueError, match="contradict"):
        score(scope, labels)


@pytest.mark.parametrize("bad", [
    {"kind": "retention", "chunk_id": "a"},
    {"kind": "retention", "chunk_id": "a", "facet": "all"},
    {"kind": "retention", "chunk_id": "outside", "facet": "constraints"},
    {"kind": "assertion", "field_path": "/candidate_body", "facet": "constraints"},
    {"kind": "relations", "field_path": "/missing"},
])
def test_bad_selector_does_not_fall_back_to_neighboring_check(bad):
    with pytest.raises(ValueError):
        evaluation.resolve_selector(prepare().requests[0], bad)


def test_binding_uses_coordinates_not_positional_ids_and_does_not_mutate_gold():
    scope = prepare().requests[0]
    labels = [label("r", [retention("constraints", "b")], "omitted")]
    original = deepcopy(labels)
    bound = evaluation.bind_labels(scope, labels)
    assert labels == original
    key = bound[0]["check_ids"][0]
    check = next(c for c in scope.checks if c.check_id == key)
    assert check.facet == "constraints"
    assert next(s for s in scope.canonical_sources if s.source_id == check.source_id).chunk_id == "b"


@pytest.mark.parametrize("values,expected", [([], "uncertain"), (["supported"], "supported"),
    (["unsupported", "uncertain"], "unsupported"), (["supported", "uncertain"], "uncertain")])
def test_grounding_aggregation_does_not_make_abstention_vacuous_success(values, expected):
    assert evaluation.aggregate_grounding(values) == expected
