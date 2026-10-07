"""Offline tests for the inactive staged diagnostic. Model text is invented."""
from __future__ import annotations

import copy
import json

import pytest

from benchmarks import codex_subscription_staged_v1 as transport
from tools.diagnostics import luna_semantic_cases as cases
from tools.diagnostics import luna_staged_core_v1 as core


def _canary_batches():
    from hymem.extraction.grounding_v2 import GroundingSource
    from hymem.extraction.triples import Triple
    specs = (("HyMem Canary Relay", "deploys_to", "Fly.io", 9901),
             ("Avery Boundary Canary", "uses", "PostgreSQL", 9902))
    batches = []
    for subject, predicate, obj, sid in specs:
        triple = Triple(subject, predicate, obj, 1, source_message_id=sid)
        source = GroundingSource(sid, f"{subject} {predicate} {obj}", (),
                                 "user", "benchmark", "2000-01-01T00:00:00Z")
        batches.append(core.staged.build_original_request((triple,), (source,))[1])
    return tuple(batches)


def _negative():
    return {"state": "not_established", "support": None}


def _original(batch, state="not_established"):
    return json.dumps({"schema": core.staged.ORIGINAL_SCHEMA,
        "batch_sha256": batch.batch_sha256, "complete": True,
        "originals": [{"index": i, "original":
            {"state": state, "support": None}}
            for i in range(len(batch.triples))]})


def _alternatives(batch):
    return json.dumps({"schema": core.staged.ALTERNATIVES_SCHEMA,
        "batch_sha256": batch.classification_batch.batch_sha256,
        "original_response_sha256": batch.original_response_sha256,
        "complete": True, "alternatives": [{"index": i, "alternatives":
            {name: _negative() for name in core.v4.PREDICATE_ORDER
             if name != batch.classification_batch.triples[i].predicate}}
            for i in batch.negative_indices]})


def _negative_invoke(request, batch, stage, recheck):
    if stage == "original":
        return _original(batch)
    return _alternatives(batch)


def test_exact_schedule_and_negative_finite_selector():
    assert [(u.kind, u.index) for u in core.schedule()] == [
        ("control", 9), ("control", 12), ("control", 13),
        ("control", 17), ("control", 19), ("control", 21),
        ("table_canary", 0), ("prose_canary", 1)]
    unit = core.schedule()[3]
    triples, sources, _ = core._unit_input(unit, cases.cases(), _canary_batches())
    result = core.evaluate_unit(unit, triples, sources, _negative_invoke)
    assert result["outcome"] == "rejected"
    assert result["error_code"] == "unsupported"
    assert [(s["stage"], s["recheck"]) for s in result["stages"]] == [
        ("original", False), ("alternatives", False)]
    assert result["stages"][1]["prior_response_sha256"]
    assert result["stages"][1]["predicate_states"]


def test_ambiguous_has_no_alternative_and_malformed_is_separate():
    unit = core.schedule()[0]
    triples, sources, _ = core._unit_input(unit, cases.cases(), _canary_batches())
    calls = []
    def ambiguous(request, batch, stage, recheck):
        calls.append(stage)
        return _original(batch, "ambiguous")
    result = core.evaluate_unit(unit, triples, sources, ambiguous)
    assert calls == ["original"]
    assert result["outcome"] == "rejected" and result["error_code"] == "uncertain"
    malformed = core.evaluate_unit(unit, triples, sources,
        lambda request, batch, stage, recheck: "private malformed text")
    assert malformed["outcome"] == "malformed"
    assert core._gold(unit, cases.cases(), malformed) is False


@pytest.mark.parametrize("recheck_state,error_code", [
    ("not_established", "unsupported"), ("ambiguous", "uncertain")])
def test_corrected_recheck_can_reject_without_fourth_stage(recheck_state, error_code):
    unit = core.schedule()[1]
    triples, sources, _ = core._unit_input(unit, cases.cases(), _canary_batches())
    calls = []
    def invoke(request, batch, stage, recheck):
        calls.append((stage, recheck))
        if stage == "original":
            return _original(batch, recheck_state if recheck else "not_established")
        payload = json.loads(_alternatives(batch))
        source = sources[0]
        quote = source.content
        check = {"state": "supported", "evidence_indices": [0]}
        payload["alternatives"][0]["alternatives"]["prefers"] = {
            "state": "supported", "support": {"evidence": [{
                "source_message_id": source.source_message_id,
                "region": "owned", "quote": quote}], "checks": {
                    "attribution_and_roles": check,
                    "relation_and_polarity": check}}}
        return json.dumps(payload)
    result = core.evaluate_unit(unit, triples, sources, invoke)
    assert result["outcome"] == "rejected" and result["error_code"] == error_code
    assert calls == [("original", False), ("alternatives", False), ("original", True)]
    assert result["stages"][-1]["states"] == [recheck_state]
    assert core._gold(unit, cases.cases(), result) is False


class FakeClient:
    def __init__(self, key, cap, budget, *, malformed_at=None, fail_at=None,
                 failed_recheck_at=None):
        self.key, self.budget = key, budget
        self.malformed_at, self.fail_at = malformed_at, fail_at
        self.failed_recheck_at = failed_recheck_at
        self.closed = False
        budget.register(key, transport.warm.BudgetLimits(*cap))

    @property
    def observed_turns(self):
        return self.budget.snapshot()["questions"][self.key]["turns"]

    @property
    def usage_complete(self):
        return self.budget.snapshot()["questions"][self.key]["usage_complete"]

    def complete_stage(self, request, batch, stage, recheck):
        assert not self.closed
        self.budget.reserve(self.key)
        self.budget.before_turn(self.key, {"auth": "chatgpt", "model":
            transport.warm.base.MODEL, "config_isolation_admitted": True,
            "inference_enabled": False,
            "quota_windows": [{"remaining_percent": 80}]})
        self.budget.settle(self.key, used=100, turn_started=True)
        if self.fail_at == self.key:
            raise RuntimeError("PRIVATE_PROVIDER_EXCEPTION")
        if self.malformed_at == self.key:
            return "PRIVATE_MALFORMED_OUTPUT"
        if self.failed_recheck_at == self.key and stage == "alternatives":
            payload = json.loads(_alternatives(batch))
            source = batch.classification_batch.sources[0]
            check = {"state": "supported", "evidence_indices": [0]}
            payload["alternatives"][0]["alternatives"]["prefers"] = {
                "state": "supported", "support": {"evidence": [{
                    "source_message_id": source.source_message_id,
                    "region": "owned", "quote": source.content}], "checks": {
                        "attribution_and_roles": check,
                        "relation_and_polarity": check}}}
            return json.dumps(payload)
        return _original(batch) if stage == "original" else _alternatives(batch)

    def close(self):
        self.closed = True


def _run(tmp_path, *, malformed_at=None, fail_at=None, failed_recheck_at=None):
    private = tmp_path / "private"
    private.mkdir(mode=0o700, parents=True)
    result = core.run_campaign(concurrent=transport.warm.concurrent,
        warm=transport.warm, binary="unused", cases_module=cases,
        canary_batches=_canary_batches(), journal=core.PrivateJournal(private),
        client_factory=lambda key, cap, budget: FakeClient(key, cap, budget,
            malformed_at=malformed_at, fail_at=fail_at,
            failed_recheck_at=failed_recheck_at))
    return result, private


def test_all_negative_eight_units_sixteen_turns_and_private_journal(tmp_path):
    result, private = _run(tmp_path)
    assert result["diagnostic_completed"] is True
    assert result["completed_units"] == 8
    assert result["paid_budget"]["turns"] == 16
    assert all(item["admitted_turns"] == 2 for item in result["units"])
    assert all(item["outcome"] == "rejected" for item in result["units"])
    assert result["units"][3]["expected_gold_match"] is True
    assert result["units"][0]["expected_gold_match"] is False
    assert result["semantic_accuracy_accepted"] is False
    assert result["completed_and_clean"] is False
    records = [json.loads(p.read_text()) for p in sorted(private.iterdir())]
    assert sum(x["phase"] == "before_dispatch" for x in records) == 16
    assert sum(x["phase"] == "response_returned" for x in records) == 16
    assert "PRIVATE" not in json.dumps(result)


def test_malformed_continues_and_provider_failure_stops(tmp_path):
    result, _ = _run(tmp_path / "malformed", malformed_at="unit-03")
    assert result["diagnostic_completed"] is True
    assert result["paid_budget"]["turns"] == 15
    assert result["units"][3]["outcome"] == "malformed"
    assert result["units"][3]["expected_gold_match"] is False
    assert result["units"][4]["outcome"] == "rejected"
    failed, _ = _run(tmp_path / "failed", fail_at="unit-03")
    assert failed["completed_units"] == 3
    assert failed["paid_budget"]["turns"] == 7
    assert failed["stop_code"] == "infrastructure_or_runtime_failure"
    assert "PRIVATE_PROVIDER_EXCEPTION" not in json.dumps(failed)


def test_failed_recheck_is_finite_and_next_unit_runs(tmp_path):
    result, _ = _run(tmp_path, failed_recheck_at="unit-01")
    assert result["diagnostic_completed"] is True
    assert result["paid_budget"]["turns"] == 17
    assert result["units"][1]["admitted_turns"] == 3
    assert result["units"][1]["outcome"] == "rejected"
    assert result["units"][1]["error_code"] == "unsupported"
    assert result["units"][2]["outcome"] == "rejected"


def test_export_rejects_raw_and_usage_forgery(tmp_path):
    result, _ = _run(tmp_path)
    changed = copy.deepcopy(result)
    changed["units"][0]["response"] = "private"
    with pytest.raises(core.DiagnosticStop):
        core.validate_public_result(changed)
    changed = copy.deepcopy(result)
    changed["paid_budget"]["turns"] += 1
    with pytest.raises(core.DiagnosticStop):
        core.validate_public_result(changed)
