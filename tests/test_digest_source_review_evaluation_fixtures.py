"""Independent fixture review and scripted plumbing tests, not model accuracy.

Every semantic expectation below was reviewed from the invented source and
candidate before provider execution. Scripted answers demonstrate only that the
labelled check is selected and scored; they cannot validate a model judgment.
"""
from collections import Counter
from copy import deepcopy
import json

import pytest

from benchmarks import digest_source_review as review
from benchmarks import digest_source_review_evaluation as evaluation
from hymem.extraction.llm import LLMRequest
from tests.digest_source_review_evaluation_fixtures import (
    build_cases, context, payload, source,
)


CASES = build_cases()
PAIR_NAMES = (
    "boundary-speaker", "boundary-independent-entity", "prohibition-omission",
    "condition-omission", "answered-outcome", "source-local-status",
    "required-order", "material-chronology", "exclusivity-scope",
    "causal-inference", "opaque-identity", "incidental-versus-outcome",
)
RETENTION_PRIMARY = {
    "prohibition-omission": ("constraints", "omitted"),
    "condition-omission": ("constraints", "omitted"),
    "answered-outcome": ("material_facts", "omitted"),
    "source-local-status": ("material_facts", "altered"),
    "required-order": ("ordering", "altered"),
    "material-chronology": ("ordering", "altered"),
    "incidental-versus-outcome": ("material_facts", "omitted"),
}


def _prepare(case):
    return review.prepare_source_review(
        case["payload"], LLMRequest("", "", max_tokens=3072), max_calls=3)


def _scope(case):
    return next(scope for scope in _prepare(case).requests
                if {"kind": scope.kind, "index": scope.index} == case["target_scope"])


def _pair(name):
    return tuple(case for case in CASES if case["pair"] == name)


def _diff(left, right, path=""):
    """Leaf paths, or a container path when its membership/length changes."""
    if type(left) is not type(right):
        return {path}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {path}
        return set().union(*(_diff(left[key], right[key], f"{path}/{key}")
                             for key in left))
    if isinstance(left, list):
        if len(left) != len(right):
            return {path}
        return set().union(*(_diff(a, b, f"{path}/{i}")
                             for i, (a, b) in enumerate(zip(left, right))))
    return set() if left == right else {path}


def _mutable_ids(value):
    if isinstance(value, dict):
        return {id(value)}.union(*(_mutable_ids(item) for item in value.values()))
    if isinstance(value, list):
        return {id(value)}.union(*(_mutable_ids(item) for item in value))
    return set()


def _scripted_response(case, scope):
    """Unlabelled checks abstain; labelled checks get their predefined gold.

    Exact source/field IDs test structural plumbing, not the truth of citations.
    No result from this function is provider evidence or semantic verification.
    """
    response = {check.check_id: (["uncertain", []] if check.kind == "retention"
                                else ["uncertain", [], []])
                for check in scope.checks}
    for label in evaluation.bind_labels(scope, case["labels"]):
        for check_id in label["check_ids"]:
            verdict = label["expected"]
            if label["domain"] == "retention":
                witnesses = ([field.field_id for field in scope.fields]
                             if verdict in {"retained", "altered"} else [])
                response[check_id] = [verdict, witnesses]
            else:
                primary = ([source.source_id for source in scope.canonical_sources]
                           if verdict == "supported" else [])
                response[check_id] = [verdict, primary, []]
    return response


def test_roster_has_twelve_stable_pairs_and_independent_mutable_copies():
    assert len(CASES) == 24
    assert tuple(dict.fromkeys(case["pair"] for case in CASES)) == PAIR_NAMES
    assert len({case["id"] for case in CASES}) == 24
    assert Counter(case["variant"] for case in CASES) == {"faithful": 12, "defective": 12}
    assert build_cases() == CASES
    seen = set()
    for case in CASES + build_cases():
        identifiers = _mutable_ids(case)
        assert not seen.intersection(identifiers)
        seen.update(identifiers)


@pytest.mark.parametrize("name", PAIR_NAMES)
def test_pair_changes_exactly_the_declared_payload_paths(name):
    faithful, defective = _pair(name)
    assert [faithful["variant"], defective["variant"]] == ["faithful", "defective"]
    assert faithful["target_scope"] == defective["target_scope"]
    assert faithful["change"] == defective["change"]
    assert faithful["changed_payload_paths"] == defective["changed_payload_paths"]
    paths = faithful["changed_payload_paths"]
    assert paths and len(paths) == len(set(paths))
    assert _diff(faithful["payload"], defective["payload"]) == set(paths)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_every_scope_strictly_prepares_and_exact_target_binds(case):
    original = deepcopy(case)
    plan = _prepare(case)
    assert review._preflight(plan) == plan
    selected = [scope for scope in plan.requests
                if {"kind": scope.kind, "index": scope.index} == case["target_scope"]]
    assert len(selected) == 1
    assert len(plan.requests) == (1 if case["target_scope"]["kind"] == "summary" else 2)
    bound = evaluation.bind_labels(selected[0], case["labels"])
    assert len(bound) == len(case["labels"])
    assert all(len(row["check_ids"]) == 1 for row in bound)
    primary = [label for label in case["labels"] if label["view"] == "primary"]
    assert len(primary) == 1
    label = primary[0]
    if case["pair"] in RETENTION_PRIMARY:
        facet, negative = RETENTION_PRIMARY[case["pair"]]
        assert label["selectors"][0]["kind"] == "retention"
        assert label["selectors"][0]["facet"] == facet
        assert label["expected"] == ("retained" if case["variant"] == "faithful" else negative)
    else:
        assert label["selectors"][0]["kind"] in {"assertion", "relations"}
        assert label["expected"] == ("supported" if case["variant"] == "faithful" else "unsupported")
    assert case == original


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_exclusions_resolve_but_never_overlap_labelled_checks(case):
    scope = _scope(case)
    labelled = {check_id for label in evaluation.bind_labels(scope, case["labels"])
                for check_id in label["check_ids"]}
    excluded = []
    for entry in case["excluded_checks"]:
        assert set(entry) == {"selector", "reason"} and entry["reason"].strip()
        excluded.append(evaluation.resolve_selector(scope, entry["selector"]))
    assert len(excluded) == len(set(excluded))
    assert not labelled.intersection(excluded)
    assert len(excluded) == (2 if case["pair"] == "boundary-speaker" else 0)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_source_coordinates_and_context_windows_preserve_exact_text(case):
    for record in case["payload"]["source_catalog"]:
        assert record["end"] - record["start"] == len(record["visible_content"])
        window = record["interpretation_only_context"]
        if window is not None:
            assert window["end"] - window["start"] == len(window["content"])
            assert 0 < len(window["content"]) <= 48
            if window["message_id"] == record["message_id"]:
                assert window["end"] == record["start"]
            else:
                assert record["start"] == 0
    scope = _scope(case)
    source_map = {record["chunk_id"]: record for record in case["payload"]["source_catalog"]}
    for unit in scope.canonical_sources:
        record = source_map[unit.chunk_id]
        assert (unit.start, unit.end, unit.text) == (
            record["start"], record["end"], record["visible_content"])


def test_fixture_source_constructor_counts_unicode_codepoints_not_utf8_bytes():
    window = context(92, "Pré-Ω ")
    record = source("unicode-coordinate", 92, "café Ω-7.", context=window)
    assert record["start"] == len(window["content"]) < len(window["content"].encode())
    assert record["end"] == len(window["content"]) + len(record["visible_content"])
    case = {"payload": payload([record], "café Ω-7.")}
    scope = _prepare(case).requests[0]
    assert scope.canonical_sources[0].text == "café Ω-7."
    assert scope.canonical_sources[0].end == record["end"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_gold_and_case_metadata_do_not_enter_request_or_change_it(case):
    original = _scope(case)
    changed = deepcopy(case)
    changed["id"] = "DO-NOT-SEND-CASE-ID"
    changed["pair"] = "DO-NOT-SEND-PAIR"
    changed["variant"] = "DO-NOT-SEND-VARIANT"
    changed["labels"] = [{"rationale": "DO-NOT-SEND-GOLD"}]
    changed["excluded_checks"] = ["DO-NOT-SEND-EXCLUSION"]
    changed["change"] = "DO-NOT-SEND-CHANGE"
    assert _scope(changed) == original
    packet = json.loads(original.request.user)
    assert not {"labels", "excluded_checks", "variant", "pair", "changed_payload_paths"}.intersection(packet)
    assert case["id"] not in original.request.user
    assert case["pair"] not in original.request.user
    assert f'"{case["variant"]}"' not in original.request.user
    assert "candidate_raw_summary" not in original.request.user
    opaque = {f"unit-{record['message_id']}" for record in case["payload"]["source_catalog"]}
    assert {record["chunk_id"] for record in case["payload"]["source_catalog"]} == opaque
    assert {unit["chunk_id"] for key in ("canonical_sources", "context_sources")
            for unit in packet[key]} <= opaque
    assert {unit["chunk_id"] for unit in packet["boundary_context_attributions"]} <= opaque
    candidate = packet["candidate"]
    citation_key = "new_source_ids" if original.kind == "summary" else "cited_source_ids"
    assert set(candidate[citation_key]) <= opaque
    if original.kind != "summary":
        assert packet["prior_summary_sources"] == []


@pytest.mark.parametrize("case", [c for c in CASES if c["target_scope"]["kind"] != "summary"],
                         ids=lambda case: case["id"])
def test_nonselected_summary_cannot_rescue_selected_scope(case):
    original = _scope(case)
    changed = deepcopy(case)
    for key in ("candidate_raw_summary", "candidate_summary", "prior_derived_summary"):
        changed["payload"]["summary_item"][key] = "UNRELATED-SUMMARY-ONLY-SENTINEL"
    assert _scope(changed) == original
    assert "UNRELATED-SUMMARY-ONLY-SENTINEL" not in original.request.user


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_scripted_new_contract_answers_select_and_score_every_label(case):
    scope = _scope(case)
    report = evaluation.score_scope(json.dumps(_scripted_response(case, scope)), scope, case["labels"])
    assert report["status"] == "valid_review"
    assert len(report["targets"]) == len(case["labels"])
    assert all(row["match"] and row["observed"] == row["expected"] for row in report["targets"])
    assert not report["semantic_verified"] and not report["publication_authorized"]
    assert all(not row["false_accept"] and not row["false_reject"] for row in report["targets"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_scripted_primary_abstention_never_gets_credit_from_other_checks(case):
    scope = _scope(case)
    response = _scripted_response(case, scope)
    primary = next(label for label in evaluation.bind_labels(scope, case["labels"])
                   if label["view"] == "primary")
    for check_id in primary["check_ids"]:
        response[check_id] = (["uncertain", []] if primary["domain"] == "retention"
                              else ["uncertain", [], []])
    report = evaluation.score_scope(json.dumps(response), scope, case["labels"])
    row = next(row for row in report["targets"] if row["view"] == "primary")
    assert row["observed"] == "uncertain" and not row["match"]
    assert all(row["match"] for row in report["targets"] if row["view"] != "primary")


def test_role_flip_keeps_words_and_candidate_but_reverses_exact_speaker_metadata():
    faithful, defective = _pair("boundary-speaker")
    for case, current, preceding in ((faithful, "user", "assistant"),
                                     (defective, "assistant", "user")):
        scope = _scope(case)
        packet = json.loads(scope.request.user)
        assert packet["boundary_context_attributions"][0]["role"] == preceding
        assert packet["boundary_context_attributions"][0]["message_id"] == 1201
        assert case["payload"]["source_catalog"][0]["role"] == current
        assert scope.canonical_sources[0].message_id == 1202
        assert all(unit.allowed_use == "interpretation" for unit in scope.context_sources)
        assert "not mine" in scope.canonical_sources[0].text
    assert faithful["payload"]["items"] == defective["payload"]["items"]


def test_independent_boundary_entity_is_not_new_canonical_evidence():
    for case in _pair("boundary-independent-entity"):
        scope = _scope(case)
        canonical = scope.canonical_sources[0]
        boundary = next(unit for unit in scope.context_sources if unit.kind == "boundary_context")
        assert "Lysfjord" in boundary.text and "Lysfjord" not in canonical.text
        assert "glassblowing" in canonical.text
        assert boundary.end == canonical.start and boundary.message_id == canonical.message_id
        assert boundary.text + canonical.text == "I sailed to Lysfjord, but I want to learn glassblowing."
        assert next(label for label in case["labels"] if label["id"] == "wish")["expected"] == "retained"


def test_omitted_prohibition_and_condition_are_not_deleted_actions_or_false_claims():
    faithful, defective = _pair("prohibition-omission")
    first, second = [case["payload"]["procedure_items"][0]["candidate"] for case in (faithful, defective)]
    assert first["steps"] == second["steps"]
    assert first["description"] == "Never lift the shield during this inspection."
    assert second["description"] is None
    assert all(next(label for label in case["labels"] if label["id"] == "remaining-action")["expected"]
               == "supported" for case in (faithful, defective))
    faithful, defective = _pair("condition-omission")
    first, second = [case["payload"]["procedure_items"][0]["candidate"] for case in (faithful, defective)]
    assert first["steps"] == second["steps"]
    assert first["triggers"] == ["The enclosure is sealed"] and second["triggers"] == []
    assert "only when the enclosure is sealed" in faithful["payload"]["source_catalog"][0]["visible_content"]


def test_answer_omission_keeps_true_request_and_true_stated_claims_separate():
    faithful, defective = _pair("answered-outcome")
    assert "The final alloy-assay result was a pass." in faithful["payload"]["summary_item"]["candidate_summary"]
    assert "was a pass" not in defective["payload"]["summary_item"]["candidate_summary"]
    for case in (faithful, defective):
        labels = {label["id"]: label for label in case["labels"]}
        assert labels["request"]["expected"] == "retained"
        assert labels["stated-claims"]["expected"] == "supported"
        assert labels["answer"]["selectors"][0]["chunk_id"] != labels["request"]["selectors"][0]["chunk_id"]


def test_second_source_alteration_does_not_poison_first_source_retention():
    faithful, defective = _pair("source-local-status")
    for case in (faithful, defective):
        assert "Batch R-62 is sealed" in case["payload"]["items"][0]["candidate_body"]
        assert "Both label bands are blue" in case["payload"]["items"][0]["candidate_body"]
        assert case["payload"]["items"][0]["candidate_outcome"] == "informational"
        assert all("Label band is blue." in record["visible_content"]
                   for record in case["payload"]["source_catalog"])
        labels = {label["id"]: label for label in case["labels"]}
        assert labels["first-batch"]["expected"] == "retained"
        assert labels["first-batch"]["selectors"][0]["chunk_id"] == "unit-1208"
        assert labels["second-batch"]["selectors"][0]["chunk_id"] == "unit-1209"
    assert "R-63 is open" in faithful["payload"]["items"][0]["candidate_body"]
    assert "R-63 is sealed" in defective["payload"]["items"][0]["candidate_body"]


def test_mandatory_order_changes_actions_not_step_numbers_and_chronology_is_material():
    faithful, defective = _pair("required-order")
    first, second = [case["payload"]["procedure_items"][0]["candidate"]["steps"]
                     for case in (faithful, defective)]
    assert [step["order"] for step in first] == [step["order"] for step in second] == [1, 2]
    assert [step["action"] for step in first] == list(reversed([step["action"] for step in second]))
    faithful, defective = _pair("material-chronology")
    assert "before delivering" in faithful["payload"]["source_catalog"][0]["visible_content"]
    assert "pickup, then delivered" in faithful["payload"]["summary_item"]["candidate_summary"]
    assert "Eastgate, then completed" in defective["payload"]["summary_item"]["candidate_summary"]


def test_exclusivity_control_has_an_explicit_closed_set_not_an_availability_assumption():
    faithful, defective = _pair("exclusivity-scope")
    text = faithful["payload"]["source_catalog"][0]["visible_content"]
    assert "The only approved services are P and Q" in text
    assert "available on P and unavailable on Q" in text
    assert "does not cover other services" in text
    assert "among the approved services" in faithful["payload"]["items"][0]["candidate_title"]
    assert "across all services" in defective["payload"]["items"][0]["candidate_title"]
    assert faithful["payload"]["items"][0]["candidate_body"] == defective["payload"]["items"][0]["candidate_body"]


def test_causality_and_display_name_are_explicitly_unestablished():
    faithful, defective = _pair("causal-inference")
    assert "does not establish a causal relation" in faithful["payload"]["source_catalog"][0]["visible_content"]
    assert "does not establish a causal relation" in faithful["payload"]["items"][0]["candidate_body"]
    assert "caused" not in faithful["payload"]["items"][0]["candidate_body"]
    assert "caused" in defective["payload"]["items"][0]["candidate_body"]
    for case in _pair("opaque-identity"):
        scope = _scope(case)
        assert "Neris" not in scope.canonical_sources[0].text
        assert scope.prior_summary_sources == ()
        assert case["payload"]["source_catalog"][0]["source_peer_id"] == "Neris"
        assert case["payload"]["summary_item"]["prior_derived_summary"]
        assert not scope.boundary_context_attributions
        assert scope.canonical_sources[0].text == "I prefer pale glazes."
        assert case["payload"]["items"][0]["candidate_outcome"] == "informational"


def test_incidental_detail_is_true_but_cannot_replace_explicit_final_decision():
    faithful, defective = _pair("incidental-versus-outcome")
    assert "accepted panel M-86 without further work" in faithful["payload"]["summary_item"]["candidate_summary"]
    assert defective["payload"]["summary_item"]["candidate_summary"] == "The inspector's notebook was amber."
    for case in (faithful, defective):
        labels = {label["id"]: label for label in case["labels"]}
        assert labels["no-order"]["expected"] == "not_applicable"
        assert labels["actual-assertion"]["expected"] == "supported"
