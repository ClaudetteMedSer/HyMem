"""Scripted scorer-contract tests; no scripted answer establishes model truth."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from benchmarks import digest_source_review as review
from benchmarks import digest_source_review_evaluation as evaluation
from hymem.extraction.llm import LLMRequest


def scope():
    text = "The valve was inspected. It must not be opened."
    payload = {"schema": "digest-fidelity-decisions-v9", "items": [], "procedure_items": [],
        "source_catalog": [{"chunk_id": "one", "message_id": 1, "role": "assistant",
            "source_peer_id": None, "source_workspace_id": None, "start": 0,
            "end": len(text), "visible_content": text, "interpretation_only_context": None}],
        "summary_item": {"index": 0, "candidate_raw_summary": "The valve was inspected.",
            "candidate_summary": "The valve was inspected.", "candidate_is_noop": False,
            "new_source_ids": ["one"], "prior_derived_summary": ""}}
    return review.prepare_source_review(payload, LLMRequest("", "", max_tokens=3072),
                                         max_calls=1).requests[0]


def label(kind="assertion", expected="supported", view="primary", id_="target",
          facet="constraints"):
    selector = ({"kind": "retention", "chunk_id": "one", "facet": facet}
                if kind == "retention" else {"kind": kind, "field_path": "/candidate_summary"})
    return {"id": id_, "view": view, "selectors": [selector], "expected": expected,
            "rationale": "Independently supplied scripted-test gold, not a live model result."}


def response(unit, **changes):
    data = {}
    for check in unit.checks:
        if check.kind == "retention":
            value = changes.get(check.facet, "retained")
            witness = [unit.fields[0].field_id] if value in {"retained", "altered"} else []
            data[check.check_id] = [value, witness]
        else:
            value = changes.get(check.kind, "supported")
            source = [unit.canonical_sources[0].source_id] if value == "supported" else []
            data[check.check_id] = [value, source, []]
    return data


def score(unit, labels=None, **changes):
    return evaluation.score_scope(json.dumps(response(unit, **changes)), unit,
                                  [label()] if labels is None else labels)


@pytest.mark.parametrize("kind,facet", [("assertion", None), ("relations", None),
    *[("retention", facet) for facet in review.RETENTION_FACETS]])
def test_exact_coordinates_resolve_to_one_scheduled_check(kind, facet):
    unit = scope()
    selector = label(kind, facet=facet)["selectors"][0]
    expected = next(check.check_id for check in unit.checks
                    if check.kind == kind and check.facet == facet)
    assert evaluation.resolve_selector(unit, selector) == expected


@pytest.mark.parametrize("selector", [None, [], {}, {"kind": True}, {"kind": "grounding"},
    {"kind": "assertion", "field_path": "/absent"},
    {"kind": "outcome", "field_path": "/candidate_summary"},
    {"kind": "assertion", "field_path": ""}, {"kind": "assertion", "field_path": 1},
    {"kind": "assertion", "field_path": "/candidate_summary", "extra": None},
    {"kind": "assertion", "field_path": "/candidate_summary", "facet": "constraints"},
    {"kind": "retention", "chunk_id": "one"},
    {"kind": "retention", "chunk_id": "s0", "facet": "constraints"},
    {"kind": "retention", "chunk_id": "one", "facet": True},
    {"kind": "retention", "chunk_id": "one", "facet": "all"},
    {"kind": "retention", "chunk_id": "one", "facet": "constraints", "extra": False},
    {"kind": "retention", "chunk_id": None, "facet": "constraints"}])
def test_bad_selectors_rejected(selector):
    with pytest.raises(ValueError):
        evaluation.resolve_selector(scope(), selector)


@pytest.mark.parametrize("field,value", [("id", ""), ("id", " "), ("id", True),
    ("id", "x" * 257), ("view", "accuracy"), ("view", []), ("expected", "uncertain"),
    ("expected", None), ("expected", True), ("expected", "retained"),
    ("rationale", ""), ("rationale", "  "), ("rationale", None),
    ("rationale", "x" * 8193), ("selectors", []), ("selectors", None),
    ("selectors", [None])])
def test_invalid_gold_rejected_before_malformed_output(field, value):
    gold = label()
    gold[field] = value
    with pytest.raises(ValueError):
        evaluation.score_scope("invalid JSON", scope(), [gold])


@pytest.mark.parametrize("labels", [None, (), [], [None], [{}]])
def test_invalid_label_container(labels):
    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), labels)


@pytest.mark.parametrize("expected", ["supported", "unsupported", "uncertain", "malformed", None, True])
def test_retention_requires_exact_four_state_gold(expected):
    with pytest.raises(ValueError, match="invalid_gold"):
        evaluation.bind_labels(scope(), [label("retention", expected)])


def test_duplicate_ids_checks_and_same_view_targets_do_not_inflate_denominators():
    for labels in ([label(), label()], [label(), label(id_="another")]):
        with pytest.raises(ValueError):
            evaluation.bind_labels(scope(), labels)
    gold = label()
    gold["selectors"] *= 2
    with pytest.raises(ValueError, match="duplicate_target_check"):
        evaluation.bind_labels(scope(), [gold])


@pytest.mark.parametrize("reverse", [False, True])
def test_same_view_partial_group_overlap_rejected_in_either_order(reverse):
    group = label(id_="group")
    group["selectors"].append(label("relations")["selectors"][0])
    labels = [group, label()]
    with pytest.raises(ValueError, match="overlapping_targets_in_view"):
        evaluation.bind_labels(scope(), labels[::-1] if reverse else labels)


def test_retention_group_and_mixed_domain_group_rejected():
    gold = label("retention", "omitted")
    gold["selectors"].append(label("retention", facet="material_facts")["selectors"][0])
    with pytest.raises(ValueError, match="retention_target_requires_one_facet"):
        evaluation.bind_labels(scope(), [gold])
    gold["selectors"][1] = label()["selectors"][0]
    with pytest.raises(ValueError, match="mixed_target_domains"):
        evaluation.bind_labels(scope(), [gold])


def test_domain_only_views_reject_other_domain():
    with pytest.raises(ValueError, match="retention_view_requires"):
        evaluation.bind_labels(scope(), [label(view="retention")])
    with pytest.raises(ValueError, match="grounding_view_requires"):
        evaluation.bind_labels(scope(), [label("retention", "retained", "grounding")])


@pytest.mark.parametrize("kind,expected,conflict", [("assertion", "supported", "unsupported"),
    ("retention", "omitted", "altered"), ("retention", "retained", "not_applicable")])
def test_cross_view_overlap_consistent_but_conflicting_singletons_rejected(kind, expected, conflict):
    labels = [label(kind, expected), label(kind, expected, "auxiliary", "other")]
    bound = evaluation.bind_labels(scope(), labels)
    assert len(bound) == 2 and bound[0]["check_ids"] == bound[1]["check_ids"]
    labels[1]["expected"] = conflict
    with pytest.raises(ValueError, match="conflicting_single_check_gold"):
        evaluation.bind_labels(scope(), labels)


def test_union_of_positive_grounding_labels_rejects_negative_conjunction():
    group = label(expected="unsupported", view="legacy", id_="group")
    group["selectors"].append(label("relations")["selectors"][0])
    with pytest.raises(ValueError, match="contradictory_gold_conjunctions"):
        evaluation.bind_labels(scope(), [group, label(), label("relations", id_="relation")])
    # One positive member of a negative group is not a contradiction.
    assert len(evaluation.bind_labels(scope(), [group, label()])) == 2


def test_bound_rows_explicit_domains_and_exact_selector_ids_without_gold_mutation():
    unit = scope()
    labels = [label(), label("retention", "omitted", id_="source")]
    before = deepcopy(labels)
    bound = evaluation.bind_labels(unit, labels)
    assert bound == (
        {"id": "target", "view": "primary", "domain": "grounding", "check_ids": ("c0",),
         "expected": "supported"},
        {"id": "source", "view": "primary", "domain": "retention", "check_ids": ("r1",),
         "expected": "omitted"})
    assert labels == before


@pytest.mark.parametrize("expected", ["retained", "omitted", "altered", "not_applicable"])
@pytest.mark.parametrize("observed", ["retained", "omitted", "altered", "not_applicable", "uncertain"])
def test_exact_retention_confusion_and_unsafe_accept_categories(expected, observed):
    unit = scope()
    report = score(unit, [label("retention", expected)], constraints=observed)
    row = report["targets"][0]
    assert row["match"] is (expected == observed)
    assert row["false_accept"] is (expected in {"omitted", "altered"}
                                    and observed in {"retained", "not_applicable"})
    assert row["false_reject"] is (expected in {"retained", "not_applicable"}
                                    and observed in {"omitted", "altered"})
    assert row["false_not_applicable"] is (expected != "not_applicable" and observed == "not_applicable")
    assert row["false_applicable"] is (expected == "not_applicable"
                                       and observed in {"retained", "omitted", "altered"})
    assert row["defect_kind_confusion"] is (expected in {"omitted", "altered"}
                                            and observed in {"omitted", "altered"} and expected != observed)
    counts = report["views"]["primary"]["retention"]
    assert counts["targets"] == 1
    assert counts["confusion"] == {expected: {observed: 1}}
    assert counts["expected"] == {expected: 1} and counts["observed"] == {observed: 1}
    assert counts["matches"] == int(row["match"])
    assert counts["false_accepts"] == int(row["false_accept"])
    assert counts["false_rejects"] == int(row["false_reject"])
    assert counts["false_not_applicable"] == int(row["false_not_applicable"])
    assert counts["false_applicable"] == int(row["false_applicable"])
    assert counts["defect_kind_confusions"] == int(row["defect_kind_confusion"])


@pytest.mark.parametrize("expected", ["supported", "unsupported"])
@pytest.mark.parametrize("observed", ["supported", "unsupported", "uncertain"])
def test_grounding_confusions_do_not_inherit_retention_statuses(expected, observed):
    report = score(scope(), [label(expected=expected)], assertion=observed)
    row = report["targets"][0]
    assert row["match"] is (expected == observed)
    assert row["false_accept"] is (expected == "unsupported" and observed == "supported")
    assert row["false_reject"] is (expected == "supported" and observed == "unsupported")
    assert not any(row[key] for key in ("false_not_applicable", "false_applicable", "defect_kind_confusion"))
    assert report["views"]["primary"]["grounding"]["confusion"] == {expected: {observed: 1}}


def test_views_and_domains_have_separate_denominators_never_grandtotal():
    labels = [label(), label("retention", "omitted", id_="omission"),
              label("retention", "omitted", "retention", "same-facet")]
    report = score(scope(), labels, constraints="omitted")
    assert report["views"]["primary"]["grounding"]["matches"] == 1
    assert report["views"]["primary"]["retention"]["matches"] == 1
    assert report["views"]["retention"]["retention"]["matches"] == 1
    assert not any(key.startswith("total") for key in report)
    assert set(report["views"]["primary"]) == {"grounding", "retention"}


@pytest.mark.parametrize("verdict", ["omitted", "altered", "uncertain"])
def test_retention_veto_cannot_rescue_wrong_grounding_accept(verdict):
    report = score(scope(), [label(expected="unsupported")], constraints=verdict)
    row = report["targets"][0]
    assert row["false_accept"] and not row["match"]
    assert row["off_target_vetoes"] == ["r1"]
    assert row["false_accept_masked_by_other_veto"]
    assert report["views"]["primary"]["grounding"]["false_accepts_masked_by_other_veto"] == 1


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_grounding_veto_cannot_rescue_wrong_retention_accept(verdict):
    report = score(scope(), [label("retention", "omitted")], assertion=verdict)
    row = report["targets"][0]
    assert row["false_accept"] and not row["match"]
    assert row["off_target_vetoes"] == ["c0"] and row["false_accept_masked_by_other_veto"]


def test_retention_accepts_are_not_off_target_vetoes():
    row = score(scope(), [label(expected="unsupported")], constraints="not_applicable")["targets"][0]
    assert row["false_accept"] and row["off_target_vetoes"] == []
    assert not row["false_accept_masked_by_other_veto"]


@pytest.mark.parametrize("raw", [None, [], {}, "not json", "{}", '{"r0":["omitted",[]]}',
    '{"c0":["unsupported",[],[]],"c0":["supported",["s0"],[]]}'])
def test_malformed_scope_marks_all_domains_without_detection_credit(raw):
    report = evaluation.score_scope(raw, scope(),
        [label(expected="unsupported"), label("retention", "omitted", id_="source")])
    assert report["status"] == "malformed_review"
    for row in report["targets"]:
        assert row["observed"] == "malformed"
        assert row["off_target_vetoes"] == []
        assert not any(row[key] for key in ("match", "false_accept", "false_reject",
            "false_not_applicable", "false_applicable", "defect_kind_confusion",
            "false_accept_masked_by_other_veto"))


def test_malformed_untargeted_check_invalidates_whole_scope_without_salvage():
    unit = scope()
    data = response(unit, assertion="unsupported")
    data["r0"] = ["retained", ["unknown"]]
    report = evaluation.score_scope(json.dumps(data), unit, [label(expected="unsupported")])
    assert report["status"] == "malformed_review"
    assert report["targets"][0]["observed"] == "malformed"
    assert not report["targets"][0]["match"]


@pytest.mark.parametrize("values,expected", [([], "uncertain"), (["supported"], "supported"),
    (["supported", "unsupported"], "unsupported"), (["supported", "uncertain"], "uncertain"),
    (["unsupported", "uncertain"], "unsupported")])
def test_grounding_conjunction_preserves_abstention(values, expected):
    assert evaluation.aggregate_grounding(values) == expected


@pytest.mark.parametrize("values", [None, (), [None], [True], ["retained"], ["omitted"],
    ["malformed"], ["supported"] * (review.MAX_CHECKS + 1)])
def test_grounding_aggregate_rejects_unknown_or_unbounded_values(values):
    with pytest.raises(ValueError):
        evaluation.aggregate_grounding(values)


def test_exact_types_and_collection_bounds():
    class MappingSubclass(dict):
        pass

    class SequenceSubclass(list):
        pass

    class TextSubclass(str):
        pass

    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), SequenceSubclass([label()]))
    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), [MappingSubclass(label())])
    gold = label()
    gold["id"] = TextSubclass("not-coerced")
    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), [gold])
    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), [label()] * (review.MAX_CHECKS + 1))
    with pytest.raises(ValueError):
        evaluation.resolve_selector(scope(), {"kind": "assertion", "field_path": "x" * (review.MAX_INPUT_CHARS + 1)})


def test_scope_forgery_rejected_and_structurally_valid_wrong_answer_stays_diagnostic():
    unit = scope()
    with pytest.raises(ValueError):
        evaluation.score_scope("{}", replace(unit, binding_sha256="0" * 64), [label()])
    report = score(unit, [label("retention", "omitted")])
    assert report["status"] == "valid_review" and report["targets"][0]["false_accept"]
    assert report["semantic_verified"] is report["publication_authorized"] is False


def test_output_limit_is_enforced_without_repair_or_gold_change():
    unit, gold = scope(), [label()]
    raw = json.dumps(response(unit))
    before = deepcopy(gold)
    assert evaluation.score_scope(raw, unit, gold, max_output_chars=len(raw))["targets"][0]["match"]
    assert evaluation.score_scope(raw, unit, gold, max_output_chars=len(raw)-1)["status"] == "malformed_review"
    assert gold == before
    with pytest.raises(ValueError):
        evaluation.score_scope(raw, unit, gold, max_output_chars=True)
