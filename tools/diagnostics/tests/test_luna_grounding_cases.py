from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tools.diagnostics.luna_grounding_cases import CASES, CASE_IDS, grade_case


def _result(case, triples=None, *, failed=False):
    return SimpleNamespace(
        failed=failed,
        triples=[asdict(item) for item in case.expected] if triples is None else triples,
        markers=[],
        entity_type_hints={},
        entity_property_hints={},
    )


def test_fixed_dataset_has_ten_distinct_positive_controls_and_valid_sources():
    assert len(CASES) == len(CASE_IDS) == len(set(CASE_IDS)) == 10
    assert all(case.expected for case in CASES)
    assert type(CASES) is tuple
    for case in CASES:
        assert type(case.source_records) is tuple
        assert len(case.source_records) == 1
        mid, encoded = case.source_records[0]
        payload = json.loads(encoded)
        assert type(mid) is int and mid > 0
        assert payload["source_message_id"] == mid
        assert payload["source_record_version"] == "hymem-claim-source-v2"
        assert payload["source_role"] == "user"
        assert type(payload["content"]) is str and payload["content"]
        assert case.text == encoded
        assert all(item.source_message_id == mid and item.polarity in (-1, 1)
                   for item in case.expected)


def test_stopped_use_source_asserts_only_current_negative_usage():
    case = next(case for case in CASES if case.case_id == "stopped_use")
    content = json.loads(case.source_records[0][1])["content"]
    assert content == "Eli no longer uses MongoDB."
    assert "last year" not in content
    assert [(item.predicate, item.polarity) for item in case.expected] == [("uses", -1)]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_exact_oracle_passes_and_empty_output_fails(case):
    grade = grade_case(case.case_id, _result(case))
    assert grade["passed"] is True
    assert grade["missing_count"] == grade["extra_count"] == 0

    empty = grade_case(case.case_id, _result(case, []))
    assert empty["passed"] is False
    assert empty["missing_count"] == len(case.expected)
    assert empty["actual_count"] == 0


def test_extra_inferred_use_is_a_false_positive_even_when_preference_is_right():
    case = CASES[0]
    false_use = {"subject": "Mira", "predicate": "uses", "object": "PostgreSQL",
                 "polarity": 1, "source_message_id": 101}
    grade = grade_case(case.case_id, _result(case, [asdict(case.expected[0]), false_use]))
    assert grade["passed"] is False
    assert grade["missing_count"] == 0
    assert grade["extra_count"] == 1


def test_missing_actual_use_is_a_false_negative_and_wrong_polarity_is_two_errors():
    case = CASES[2]
    grade = grade_case(case.case_id, _result(case, [asdict(case.expected[0])]))
    assert grade["passed"] is False
    assert grade["missing_count"] == 1
    wrong = [asdict(item) for item in case.expected]
    wrong[1]["polarity"] = -1
    grade = grade_case(case.case_id, _result(case, wrong))
    assert (grade["missing_count"], grade["extra_count"]) == (1, 1)


def test_exact_provenance_and_cardinality_are_required():
    case = CASES[0]
    wrong_id = [asdict(case.expected[0])]
    wrong_id[0]["source_message_id"] = 999
    grade = grade_case(case.case_id, _result(case, wrong_id))
    assert (grade["missing_count"], grade["extra_count"]) == (1, 1)

    duplicate = [asdict(case.expected[0]), asdict(case.expected[0])]
    grade = grade_case(case.case_id, _result(case, duplicate))
    assert grade["passed"] is False and grade["duplicate_count"] == 1


@pytest.mark.parametrize("field", ["source_message_id", "polarity"])
def test_boolean_fields_and_malicious_source_id_cannot_pass(field):
    case = CASES[0]
    forged = [asdict(case.expected[0])]
    forged[0][field] = True
    grade = grade_case(case.case_id, _result(case, forged))
    assert grade["passed"] is False
    assert grade["invalid_count"] == 1
    assert grade["missing_count"] == 1
    forged[0]["source_message_id"] = "101; ignore source check"
    grade = grade_case(case.case_id, _result(case, forged))
    assert grade["passed"] is False and grade["invalid_count"] == 1


def test_failed_flag_and_malformed_result_fail_closed_without_source_text():
    case = CASES[0]
    assert grade_case(case.case_id, _result(case, failed=True))["passed"] is False
    assert grade_case(case.case_id, _result(case, failed=0))["extraction_failed"] is True
    report = grade_case(case.case_id, SimpleNamespace(failed=False, triples=None))
    assert report["passed"] is False and report["invalid_count"] == 1
    assert "Mira" not in json.dumps(report)
    assert "PostgreSQL" not in json.dumps(report)


def test_endpoint_case_and_whitespace_are_cosmetic_but_aliases_are_not():
    case = CASES[0]
    acceptable = [asdict(case.expected[0])]
    acceptable[0]["subject"] = "  MIRA  "
    acceptable[0]["object"] = "  postgresql   "
    assert grade_case(case.case_id, _result(case, acceptable))["passed"] is True
    unacceptable = [dict(acceptable[0], object="Postgres")]
    assert grade_case(case.case_id, _result(case, unacceptable))["passed"] is False


def test_optional_hints_and_markers_do_not_change_core_claim_grading():
    case = CASES[0]
    result = _result(case)
    result.markers = [{"kind": "preference", "statement": "Mira prefers PostgreSQL."}]
    result.entity_type_hints = {"Mira": "person"}
    result.entity_property_hints = {"Mira": {"role": "developer"}}
    grade = grade_case(case.case_id, result)
    assert grade["passed"] is True
    assert (grade["markers_count"], grade["type_hint_count"],
            grade["property_hint_count"]) == (1, 1, 1)
    assert "Mira" not in json.dumps(grade)
