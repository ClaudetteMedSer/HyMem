"""Independent frozen-control checks, not observed semantic accuracy."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tests.digest_bounded_summary_fixtures import build_cases


def test_prospective_control_file_has_not_drifted_after_root_semantic_review():
    path = Path(__file__).with_name("digest_bounded_summary_fixtures.py")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "67c230ba54e6a76bbaa79870a845e26bc1d03cd8e3d099946a511a467e36e233"


@pytest.mark.parametrize("case", build_cases(), ids=lambda x: x["case_id"])
def test_controls_keep_gold_outside_transport_and_exact_evidence_coordinates(case):
    packet, gold = case["payload"], case["gold"]
    assert packet["schema"] == "digest-fidelity-decisions-v9"
    assert set(packet) == {"schema", "source_catalog", "items", "procedure_items", "summary_item"}
    assert packet["items"] == packet["procedure_items"] == []
    assert "expected_policy_verdicts" not in json.dumps(packet)
    assert not gold["underlying_coverage_assessed"] and not gold["semantic_accuracy_observed"]
    summary = packet["summary_item"]
    assert 0 < len(summary["candidate_summary"]) <= 500
    assert len(summary["prior_derived_summary"]) <= 500
    assert summary["candidate_summary"].count(".") == 1
    assert not summary["candidate_summary"].startswith(("The user", "The assistant"))
    catalog = {r["chunk_id"]: r for r in packet["source_catalog"]}
    assert len(catalog) == len(packet["source_catalog"])
    assert summary["new_source_ids"] == list(catalog)
    for anchor in gold["source_anchors"]:
        record = catalog[anchor["chunk_id"]]
        assert record["role"] == anchor["role"]
        assert record["message_id"] == anchor["message_id"]
        start = anchor["canonical_start"] - record["start"]
        end = anchor["canonical_end"] - record["start"]
        assert record["visible_content"][start:end] == anchor["canonical_quote"]
    if summary["candidate_is_noop"]:
        assert summary["candidate_raw_summary"] == ""
        assert summary["candidate_summary"] == summary["prior_derived_summary"]
    else:
        assert summary["candidate_raw_summary"] == summary["candidate_summary"]


def test_paired_controls_hold_source_and_prior_constant_except_declared_source_mutations():
    grouped = defaultdict(list)
    for case in build_cases():
        grouped[case["family"]].append(case)
    assert len(grouped) == 6 and sum(map(len, grouped.values())) == 12
    for group in grouped.values():
        for case in group:
            if case["gold"]["source_and_prior_unchanged_within_family"]:
                assert case["payload"]["source_catalog"] == group[0]["payload"]["source_catalog"]
                assert case["payload"]["summary_item"]["prior_derived_summary"] == group[0]["payload"]["summary_item"]["prior_derived_summary"]
    left, right = grouped["recurring-outcome-bindings"]
    assert Counter(left["payload"]["summary_item"]["candidate_summary"].split()) == Counter(right["payload"]["summary_item"]["candidate_summary"].split())
    assert left["gold"]["expected_policy_verdicts"] != right["gold"]["expected_policy_verdicts"]


def test_only_complete_whole_topic_selection_changes_the_prospective_gold():
    cases = build_cases()
    changed = [c for c in cases if len(set(c["gold"]["expected_policy_verdicts"].values())) > 1]
    assert [c["variant"] for c in changed] == ["selected-complete"]
    assert changed[0]["gold"]["expected_policy_verdicts"] == {
        "legacy_material_retention": "unsupported", "bounded_highlights": "supported"}
    broken = {c["variant"] for c in cases
        if c["gold"]["expected_policy_verdicts"]["bounded_highlights"] == "unsupported"}
    assert broken == {"selected-outcome-omitted", "condition-omitted", "negation-omitted",
                      "bindings-swapped", "stale-prior-asserted", "retained-state-corrected"}


def test_builders_do_not_share_mutable_gold_or_sources():
    first, second = build_cases(), build_cases()
    original = deepcopy(second)
    first[0]["payload"]["source_catalog"][0]["visible_content"] = "mutated"
    first[0]["gold"]["source_anchors"].clear()
    assert second == original == build_cases()
