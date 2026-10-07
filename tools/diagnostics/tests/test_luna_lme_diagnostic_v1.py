"""Offline contract checks for the noncanonical LME integration."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "luna_lme_diagnostic_v1.py"
SPEC = importlib.util.spec_from_file_location("luna_lme_diagnostic_v1_tested", SOURCE)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Budget:
    def __init__(self):
        self.question = {"turns": 0, "known_tokens": 0, "usage_complete": True}
        self.halted = None

    def snapshot(self):
        return {"questions": {"q-0000": dict(self.question)}}

    def halt(self, reason):
        self.halted = reason


def test_second_registration_is_exactly_one_existing_question():
    budget = Budget()
    limits = (20, 1000, 30.0)
    alias = runner.RegistrationAlias(budget, "q-0000", limits)
    alias.register("q-0000", limits)
    with pytest.raises(ValueError, match="shared_question_registration_invalid"):
        alias.register("q-0001", limits)
    with pytest.raises(ValueError, match="shared_question_registration_invalid"):
        alias.register("q-0000", (21, 1000, 30.0))


def test_unclassified_call_is_rejected_before_paid_delegate():
    budget = Budget()
    ordinary = SimpleNamespace(complete=lambda request: pytest.fail("paid call"))
    dual = runner.DualClient(ordinary, None, budget, "q-0000")
    accounted = runner.AccountedClient(dual, Path("/nonexistent-candidate"))
    with pytest.raises(RuntimeError, match="stage_accounting_failure"):
        accounted.complete(SimpleNamespace())
    assert budget.halted == "stage_accounting_failure"


def test_diagnostic_row_keeps_unhealthy_indexing_distinct():
    class Protocol:
        @staticmethod
        def _validate_versioned_indexing(indexing, *, require_healthy, allow_failure):
            assert require_healthy and allow_failure
            return indexing["healthy"]

    loaded = {"protocol": Protocol,
              "diagnostic": SimpleNamespace(MODE="semantic_diagnostic_v1")}
    decision = {"mode": "semantic_diagnostic_v1", "admitted": True,
        "kind": "semantic_quarantine", "quarantined_chunks": 2,
        "summary_degraded_sessions": 0}
    row = {"question_id": "qid", "correct": False, "benchmark_failure": None,
        "judge_error": False, "judge_parse_valid": True, "retrieval_only": False,
        "indexing": {"outcome": "failure", "healthy": False}, "context_sha": "a" * 64}
    projected = runner.validate_diagnostic_row(loaded, row, decision, "qid")
    assert projected["strict_indexing_healthy"] is False
    assert projected["diagnostic_kind"] == "semantic_quarantine"
    assert projected["correct"] is False
    assert row["indexing"] == {"outcome": "failure", "healthy": False}
    row["indexing"] = {"outcome": "success", "healthy": True}
    with pytest.raises(ValueError, match="diagnostic_strict_health_mismatch"):
        runner.validate_diagnostic_row(loaded, row, decision, "qid")


def test_paid_campaign_requires_containment_before_output(tmp_path):
    with pytest.raises(ValueError, match="campaign_preflight_invalid"):
        runner.run_campaign({"questions": [{"question_id": "qid"}]},
            output=tmp_path / "new", campaign_limits=SimpleNamespace(seconds=100),
            canary_limits=SimpleNamespace(seconds=10),
            question_limits=SimpleNamespace(seconds=50),
            indexing_seconds=20, workers=1, helper_sha256="0" * 64,
            containment=None)
    assert not (tmp_path / "new").exists()


def test_containment_failure_precedes_output_or_client(tmp_path):
    caps = SimpleNamespace(turns=1, known_tokens=100, seconds=30)
    def reject(_loaded):
        raise ValueError("containment_invalid")
    with pytest.raises(ValueError, match="containment_invalid"):
        runner.run_campaign({"questions": [{"question_id": "qid"}]},
            output=tmp_path / "new", campaign_limits=SimpleNamespace(
                turns=10, known_tokens=1000, seconds=100),
            canary_limits=caps, question_limits=SimpleNamespace(
                turns=2, known_tokens=200, seconds=50),
            indexing_seconds=20, workers=1, helper_sha256="0" * 64,
            containment=reject)
    assert not (tmp_path / "new").exists()
    with pytest.raises(ValueError, match="containment_unverified"):
        runner.run_campaign({"questions": [{"question_id": "qid"}]},
            output=tmp_path / "new", campaign_limits=SimpleNamespace(
                turns=10, known_tokens=1000, seconds=100),
            canary_limits=caps, question_limits=SimpleNamespace(
                turns=2, known_tokens=200, seconds=50),
            indexing_seconds=20, workers=1, helper_sha256="0" * 64,
            containment=lambda _loaded: False)
    assert not (tmp_path / "new").exists()


def test_canary_gold_rejects_extra_type_hints():
    claim = ("Ada", "person", "prefers", "Tea", "drink", 1, 9)
    canary = SimpleNamespace(_CANARY_EXPECTED_CLAIMS=(claim,),
        _CANARY_OPTIONAL_TRIPLE_FIELDS=("value_text",))
    triple = SimpleNamespace(subject="Ada", predicate="prefers", object="Tea",
        polarity=1, source_message_id=9, value_text=None)
    result = SimpleNamespace(failed=False, triples=[triple], markers=[],
        duplicate_triples_collapsed=0, entity_property_hints={},
        entity_type_hints={"Ada": "person", "Tea": "drink", "Extra": "person"})
    assert runner._canary_gold(canary, result) is False
    del result.entity_type_hints["Extra"]
    assert runner._canary_gold(canary, result) is True


def test_stage_source_binding_rejects_empty_and_wrong_context():
    correct = {"source_ids": [9], "source_contents": ["claim"],
        "source_contexts": [["header"]]}
    assert runner._stage_source_context_bound([correct],
        {9: "header claim"}, {9}, {(9, "claim"): ("header",)})
    assert not runner._stage_source_context_bound([{
        **correct, "source_contents": [""]}], {9: "header claim"}, {9},
        {(9, "claim"): ("header",)})
    assert not runner._stage_source_context_bound([{
        **correct, "source_contexts": [["wrong"]]}],
        {9: "header claim"}, {9}, {(9, "claim"): ("header",)})
    assert not runner._stage_source_context_bound([{
        **correct, "source_contexts": [[]]}],
        {9: "header claim"}, {9}, {(9, "claim"): ("header",)})
    assert not runner._stage_source_context_bound([], {9: "header claim"}, {9},
        {(9, "claim"): ("header",)})


def test_ordinary_context_derivation_rejects_conflicting_source_slice():
    request = object()
    payload = {"source_message_id": 9, "content": "claim",
        "source_fragment_context": {"content": "header",
            "prelude_content": "prelude"}}
    canary = SimpleNamespace(_request_source_payloads=lambda _request: ([payload], 0))
    assert runner._ordinary_source_contexts(canary, [request]) == {
        (9, "claim"): ("header", "prelude")}
    call = 0
    def inconsistent(_request):
        nonlocal call
        call += 1
        return ([{**payload, "source_fragment_context": {"content":
            "header" if call == 1 else "wrong"}}], 0)
    canary._request_source_payloads = inconsistent
    assert runner._ordinary_source_contexts(canary, [request, request]) is None


def test_cli_run_requires_source_arguments():
    with pytest.raises(SystemExit):
        runner.main(["--run"])
