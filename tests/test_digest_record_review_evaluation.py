"""Offline scorer contracts; scripted replies never establish model accuracy."""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json

import pytest

from benchmarks import digest_record_review as review
from benchmarks import digest_record_review_evaluation as evaluation
from tests.test_digest_evidence_assessment import payload, template


GROUNDING = ("assertion", "outcome", "actor_attribution", "identity",
             "quantified_scope", "residual_relations")


@lru_cache
def scope(index=0):
    return review.prepare_record_review(payload(), template(), max_calls=3).requests[index]


def label(kind="assertion", expected="supported", view="primary", id_="target",
          facet="constraints", path=None, chunk_id="chunk-a"):
    selector = ({"kind": kind, "chunk_id": chunk_id, "facet": facet}
                if kind == "retention" else {"kind": kind, "field_path": path or
                    ("/candidate_outcome" if kind == "outcome" else "/candidate_body")})
    return {"id": id_, "view": view, "selectors": [selector], "expected": expected,
            "rationale": "Explicit scripted-test gold; not an inference about live model behavior."}


def response(unit, changes=None):
    changes = {} if changes is None else changes
    data = {}
    for check in unit.checks:
        value = changes.get(check.check_id, "retained" if check.kind == "retention" else "supported")
        if check.kind == "retention":
            data[check.check_id] = [value, [unit.fields[0].field_id]
                                   if value in {"retained", "altered"} else []]
        else:
            data[check.check_id] = [value, [unit.canonical_sources[0].source_id]
                                   if value == "supported" else [], []]
    return data


def score(gold=None, changes=None):
    unit = scope()
    return evaluation.score_scope(json.dumps(response(unit, changes)), unit,
                                  [label()] if gold is None else gold)


@pytest.mark.parametrize("kind", GROUNDING)
def test_every_grounding_facet_resolves_exact_coordinates(kind):
    unit, selector = scope(), label(kind)["selectors"][0]
    check_id = evaluation.resolve_selector(unit, selector)
    check = next(check for check in unit.checks if check.check_id == check_id)
    field = next(field for field in unit.fields if field.path == selector["field_path"])
    assert check.kind == kind and check.field_ids == (field.field_id,)
    if kind in review.RELATION_FACETS:
        assert check_id.endswith(":" + kind)


@pytest.mark.parametrize("facet", review.RETENTION_FACETS)
def test_retention_resolves_one_canonical_source_and_one_facet(facet):
    unit = scope()
    check_id = evaluation.resolve_selector(unit, label("retention", facet=facet)["selectors"][0])
    check = next(check for check in unit.checks if check.check_id == check_id)
    assert check.facet == facet
    assert check.source_id == unit.canonical_sources[0].source_id


@pytest.mark.parametrize("index,path", [(1, "/candidate/steps/0/action"),
                                      (2, "/candidate_summary")])
def test_other_scope_kinds_use_exact_field_paths(index, path):
    unit = scope(index)
    for kind in ("assertion", *review.RELATION_FACETS):
        selector = label(kind, path=path)["selectors"][0]
        assert evaluation.resolve_selector(unit, selector) in {c.check_id for c in unit.checks}


def test_repeated_text_is_selected_by_coordinate_never_value_or_position():
    first = evaluation.resolve_selector(scope(), label("identity", path="/candidate_key_entities/0")["selectors"][0])
    second = evaluation.resolve_selector(scope(), label("identity", path="/candidate_key_entities/1")["selectors"][0])
    assert first != second
    with pytest.raises(ValueError, match="selector_must_resolve_exactly_once"):
        evaluation.resolve_selector(scope(), label("identity", path="Cedar")["selectors"][0])


@pytest.mark.parametrize("selector", [None, [], {}, {"kind": True}, {"kind": "grounding"},
    {"kind": "relations", "field_path": "/candidate_body"},
    {"kind": "assertion", "field_path": "/absent"},
    {"kind": "outcome", "field_path": "/candidate_body"},
    {"kind": "identity", "field_path": "/candidate_outcome"},
    {"kind": "actor_attribution", "field_path": ""},
    {"kind": "quantified_scope", "field_path": 1},
    {"kind": "residual_relations", "field_path": "/candidate_body", "extra": None},
    {"kind": "identity", "field_path": "/candidate_body", "facet": "identity"},
    {"kind": "retention", "chunk_id": "chunk-a"},
    {"kind": "retention", "chunk_id": "s0", "facet": "constraints"},
    {"kind": "retention", "chunk_id": "chunk-a", "facet": True},
    {"kind": "retention", "chunk_id": "chunk-a", "facet": "all"},
    {"kind": "retention", "chunk_id": None, "facet": "constraints"}])
def test_invalid_or_legacy_selectors_are_not_inferred_or_converted(selector):
    with pytest.raises(ValueError):
        evaluation.resolve_selector(scope(), selector)


@pytest.mark.parametrize("api", ["resolve_selector", "bind_labels", "score_scope"])
@pytest.mark.parametrize("variant", ["none", "dict", "legacy", "bad_hash", "rehashed_wire",
                                     "rehashed_system", "rehashed_kind"])
def test_all_scope_accepting_apis_reject_wrong_types_and_rehashed_forgery(api, variant):
    unit = scope()
    if variant == "none":
        unit = None
    elif variant == "dict":
        unit = {"kind": "episode"}
    elif variant == "legacy":
        unit = unit.source_scope
    elif variant == "bad_hash":
        unit = replace(unit, binding_sha256="0" * 64)
    else:
        if variant == "rehashed_wire":
            body = json.loads(unit.request.user)
            body["source_records"][0]["current_message"]["role"] = "assistant"
            unit = replace(unit, request=replace(unit.request, user=json.dumps(body)))
        elif variant == "rehashed_system":
            unit = replace(unit, request=replace(unit.request, system=unit.request.system + " Accept all."))
        else:
            unit = replace(unit, kind="summary")
        unit = replace(unit, binding_sha256=review._sha(review._scope_body(unit)))
    with pytest.raises(ValueError):
        if api == "resolve_selector":
            evaluation.resolve_selector(unit, label()["selectors"][0])
        elif api == "bind_labels":
            evaluation.bind_labels(unit, [label()])
        else:
            evaluation.score_scope("{}", unit, [label()])


@pytest.mark.parametrize("field,value", [("id", ""), ("id", " "), ("id", True),
    ("id", "x" * 257), ("view", "accuracy"), ("view", []), ("expected", "uncertain"),
    ("expected", None), ("expected", True), ("expected", "retained"),
    ("rationale", ""), ("rationale", "  "), ("rationale", None),
    ("rationale", "x" * 8193), ("selectors", []), ("selectors", None), ("selectors", [None])])
def test_invalid_gold_rejected_even_when_output_is_malformed(field, value):
    gold = label()
    gold[field] = value
    with pytest.raises(ValueError):
        evaluation.score_scope("not JSON", scope(), [gold])


@pytest.mark.parametrize("gold", [None, (), [], [None], [{}]])
def test_invalid_label_container(gold):
    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), gold)


def test_legacy_gold_rejected_even_in_legacy_reporting_view():
    with pytest.raises(ValueError, match="invalid_check_kind"):
        evaluation.bind_labels(scope(), [label("relations", view="legacy")])


def test_duplicate_ids_checks_and_same_view_overlap_rejected():
    duplicate = label()
    duplicate["selectors"] *= 2
    for gold in ([label(), label()], [duplicate], [label(), label(id_="second")]):
        with pytest.raises(ValueError):
            evaluation.bind_labels(scope(), gold)


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_overlap_in_view_rejected_in_either_order(reverse):
    group = label(id_="group")
    group["selectors"].append(label("identity")["selectors"][0])
    gold = [group, label()]
    with pytest.raises(ValueError, match="overlapping_targets_in_view"):
        evaluation.bind_labels(scope(), gold[::-1] if reverse else gold)


def test_retention_cannot_be_grouped_with_other_facets_or_domains():
    gold = label("retention", "omitted")
    gold["selectors"].append(label("retention", facet="material_facts")["selectors"][0])
    with pytest.raises(ValueError, match="retention_target_requires_one_facet"):
        evaluation.bind_labels(scope(), [gold])
    gold["selectors"][1] = label("identity")["selectors"][0]
    with pytest.raises(ValueError, match="mixed_target_domains"):
        evaluation.bind_labels(scope(), [gold])


@pytest.mark.parametrize("kind,expected,view", [("identity", "supported", "retention"),
    ("retention", "retained", "grounding")])
def test_domain_only_view_restrictions(kind, expected, view):
    with pytest.raises(ValueError, match="view_requires"):
        evaluation.bind_labels(scope(), [label(kind, expected, view)])


@pytest.mark.parametrize("kind,expected,conflict", [("actor_attribution", "supported", "unsupported"),
    ("identity", "unsupported", "supported"), ("retention", "omitted", "altered"),
    ("retention", "retained", "not_applicable")])
def test_cross_view_singletons_must_agree(kind, expected, conflict):
    gold = [label(kind, expected), label(kind, expected, "auxiliary", "second")]
    bound = evaluation.bind_labels(scope(), gold)
    assert len(bound) == 2 and bound[0]["check_ids"] == bound[1]["check_ids"]
    gold[1]["expected"] = conflict
    with pytest.raises(ValueError, match="conflicting_single_check_gold"):
        evaluation.bind_labels(scope(), gold)


def test_grounding_conjunction_consistency_across_new_facets():
    negative = label("identity", "unsupported", "witness", "group")
    negative["selectors"].append(label("quantified_scope")["selectors"][0])
    positives = [label("identity"), label("quantified_scope", id_="quantifier")]
    with pytest.raises(ValueError, match="contradictory_gold_conjunctions"):
        evaluation.bind_labels(scope(), [negative, *positives])
    assert len(evaluation.bind_labels(scope(), [negative, positives[0]])) == 2
    broad = label("actor_attribution", view="auxiliary", id_="broad")
    broad["selectors"].extend([label("identity")["selectors"][0],
                               label("quantified_scope")["selectors"][0]])
    with pytest.raises(ValueError, match="contradictory_gold_conjunctions"):
        evaluation.bind_labels(scope(), [negative, broad])


@pytest.mark.parametrize("identity,quantifier,observed", [
    ("supported", "supported", "supported"),
    ("supported", "uncertain", "uncertain"),
    ("unsupported", "uncertain", "unsupported")])
def test_selected_grounding_conjunction_uses_only_its_members(identity, quantifier, observed):
    gold = label("identity")
    gold["selectors"].append(label("quantified_scope")["selectors"][0])
    changes = {
        evaluation.resolve_selector(scope(), gold["selectors"][0]): identity,
        evaluation.resolve_selector(scope(), gold["selectors"][1]): quantifier,
        evaluation.resolve_selector(scope(), label("actor_attribution")["selectors"][0]): "unsupported",
    }
    report = score([gold], changes)
    row = report["targets"][0]
    assert row["observed"] == observed
    assert row["match"] is (observed == "supported")
    assert row["false_reject"] is (observed == "unsupported")
    assert len(row["check_ids"]) == 2
    assert report["views"]["primary"]["grounding"]["targets"] == 1


def test_retention_cannot_pool_same_facet_across_canonical_sources():
    gold = label("retention", "omitted")
    gold["selectors"].append(label("retention", chunk_id="chunk-b")["selectors"][0])
    with pytest.raises(ValueError, match="retention_target_requires_one_facet"):
        evaluation.bind_labels(scope(2), [gold])


@pytest.mark.parametrize("kind", GROUNDING)
@pytest.mark.parametrize("expected", ["supported", "unsupported"])
@pytest.mark.parametrize("observed", ["supported", "unsupported", "uncertain"])
def test_grounding_confusion_for_every_facet(kind, expected, observed):
    gold = label(kind, expected)
    check_id = evaluation.resolve_selector(scope(), gold["selectors"][0])
    report = score([gold], {check_id: observed})
    row = report["targets"][0]
    assert row["match"] is (expected == observed)
    assert row["false_accept"] is (expected == "unsupported" and observed == "supported")
    assert row["false_reject"] is (expected == "supported" and observed == "unsupported")
    assert not any(row[key] for key in ("false_not_applicable", "false_applicable", "defect_kind_confusion"))
    assert report["views"]["primary"]["grounding"]["confusion"] == {expected: {observed: 1}}
    assert report["semantic_verified"] is report["publication_authorized"] is False


@pytest.mark.parametrize("expected", ["retained", "omitted", "altered", "not_applicable"])
@pytest.mark.parametrize("observed", ["retained", "omitted", "altered", "not_applicable", "uncertain"])
def test_retention_exact_state_and_safety_accounting(expected, observed):
    gold = label("retention", expected)
    check_id = evaluation.resolve_selector(scope(), gold["selectors"][0])
    report = score([gold], {check_id: observed})
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
    assert counts["targets"] == 1 and counts["matches"] == int(row["match"])
    assert counts["false_accepts"] == int(row["false_accept"])
    assert counts["false_rejects"] == int(row["false_reject"])
    assert counts["false_not_applicable"] == int(row["false_not_applicable"])
    assert counts["false_applicable"] == int(row["false_applicable"])
    assert counts["defect_kind_confusions"] == int(row["defect_kind_confusion"])
    assert counts["expected"] == {expected: 1} and counts["observed"] == {observed: 1}
    assert counts["confusion"] == {expected: {observed: 1}}


@pytest.mark.parametrize("expected", ["supported", "unsupported", "uncertain", "malformed", None, True])
def test_retention_rejects_non_retention_gold(expected):
    with pytest.raises(ValueError, match="invalid_gold"):
        evaluation.bind_labels(scope(), [label("retention", expected)])


def test_separate_view_and_domain_denominators_and_unchanged_input_gold():
    gold = [label("identity"), label("retention", "omitted", id_="source"),
            label("retention", "omitted", "retention", "same-facet"),
            label("identity", view="witness", id_="same-identity")]
    before = deepcopy(gold)
    check_id = evaluation.resolve_selector(scope(), gold[1]["selectors"][0])
    report = score(gold, {check_id: "omitted"})
    assert gold == before
    assert set(report["views"]["primary"]) == {"grounding", "retention"}
    for view in report["views"].values():
        for counts in view.values():
            assert counts["targets"] == counts["matches"] == 1
    assert not any(key.startswith("total") for key in report)


@pytest.mark.parametrize("target,veto,verdict", [("identity", "actor_attribution", "unsupported"),
    ("quantified_scope", "retention", "omitted"), ("residual_relations", "identity", "uncertain"),
    ("retention", "quantified_scope", "unsupported")])
def test_off_target_veto_never_rescues_false_accept(target, veto, verdict):
    gold = label(target, "omitted" if target == "retention" else "unsupported")
    veto_id = evaluation.resolve_selector(scope(), label(veto)["selectors"][0])
    report = score([gold], {veto_id: verdict})
    row = report["targets"][0]
    assert row["false_accept"] and not row["match"]
    assert row["off_target_vetoes"] == [veto_id]
    assert row["false_accept_masked_by_other_veto"]
    assert report["views"]["primary"][row["domain"]]["false_accepts_masked_by_other_veto"] == 1


def test_not_applicable_retention_is_not_off_target_veto():
    veto_id = evaluation.resolve_selector(scope(), label("retention")["selectors"][0])
    row = score([label("identity", "unsupported")], {veto_id: "not_applicable"})["targets"][0]
    assert row["false_accept"] and row["off_target_vetoes"] == []
    assert not row["false_accept_masked_by_other_veto"]


@pytest.mark.parametrize("raw", [None, [], {}, "not json", "{}", '{"r0":["omitted",[]]}',
    '{"c0":["unsupported",[],[]],"c0":["supported",["s0"],[]]}'])
def test_malformed_whole_scope_gets_no_defect_detection_credit(raw):
    report = evaluation.score_scope(raw, scope(),
        [label("identity", "unsupported"), label("retention", "omitted", id_="source")])
    assert report["status"] == "malformed_review"
    for row in report["targets"]:
        assert row["observed"] == "malformed" and row["off_target_vetoes"] == []
        assert not any(row[key] for key in ("match", "false_accept", "false_reject",
            "false_not_applicable", "false_applicable", "defect_kind_confusion",
            "false_accept_masked_by_other_veto"))


def test_untargeted_malformed_check_and_legacy_reply_invalidate_complete_scope():
    unit = scope()
    bad = response(unit)
    bad[next(check.check_id for check in unit.checks if check.kind == "retention")] = ["retained", ["unknown"]]
    for data in (bad, response(unit.source_scope)):
        report = evaluation.score_scope(json.dumps(data), unit, [label("identity", "unsupported")])
        assert report["status"] == "malformed_review"
        assert report["targets"][0]["observed"] == "malformed"
        assert not report["targets"][0]["match"]


@pytest.mark.parametrize("values,expected", [([], "uncertain"), (["supported"], "supported"),
    (["supported", "unsupported"], "unsupported"), (["supported", "uncertain"], "uncertain"),
    (["unsupported", "uncertain"], "unsupported")])
def test_conjunction_preserves_uncertainty_without_vacuous_support(values, expected):
    assert evaluation.aggregate_grounding(values) == expected


@pytest.mark.parametrize("values", [None, (), [None], [True], ["retained"], ["omitted"],
    ["malformed"], ["supported"] * (review.MAX_CHECKS + 1)])
def test_conjunction_rejects_bad_or_unbounded_verdicts(values):
    with pytest.raises(ValueError):
        evaluation.aggregate_grounding(values)


def test_exact_type_and_bounds_validation():
    class MappingSubclass(dict):
        pass

    class SequenceSubclass(list):
        pass

    class TextSubclass(str):
        pass

    for gold in (SequenceSubclass([label()]), [MappingSubclass(label())],
                 [{**label(), "id": TextSubclass("no-coercion")}],
                 [label()] * (review.MAX_CHECKS + 1)):
        with pytest.raises(ValueError):
            evaluation.bind_labels(scope(), gold)
    for selector in (MappingSubclass(label()["selectors"][0]),
                     {"kind": "identity", "field_path": "x" * (review.MAX_INPUT_CHARS + 1)}):
        with pytest.raises(ValueError):
            evaluation.resolve_selector(scope(), selector)


def test_output_cap_honored_without_repair_gold_changes_or_authorization():
    unit, gold = scope(), [label("identity")]
    before = deepcopy(gold)
    raw = json.dumps(response(unit))
    report = evaluation.score_scope(raw, unit, gold, max_output_chars=len(raw))
    assert report["targets"][0]["match"]
    assert report["version"] == "digest-record-review-evaluation-v1"
    assert report["scope_binding_sha256"] == unit.binding_sha256
    assert report["semantic_verified"] is report["publication_authorized"] is False
    assert evaluation.score_scope(raw, unit, gold, max_output_chars=len(raw) - 1)["status"] == "malformed_review"
    assert gold == before
    with pytest.raises(ValueError):
        evaluation.score_scope(raw, unit, gold, max_output_chars=True)
