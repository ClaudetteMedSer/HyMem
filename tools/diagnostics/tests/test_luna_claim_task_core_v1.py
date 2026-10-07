"""Offline fixed-schedule and privacy checks; all model answers are invented."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import codex_subscription_claim_task_v1 as transport
from tools.diagnostics import luna_claim_task_core_v1 as core
from tools.diagnostics import luna_semantic_cases as cases


def _canary_batches():
    from hymem.extraction.grounding_v2 import GroundingSource
    from hymem.extraction.triples import Triple
    specs = (("HyMem Canary Relay", "deploys_to", "Fly.io", 9901),
             ("Avery Boundary Canary", "uses", "PostgreSQL", 9902))
    result = []
    for subject, predicate, obj, sid in specs:
        triple = Triple(subject, predicate, obj, 1, source_message_id=sid)
        source = GroundingSource(sid, f"{subject} {predicate} {obj}", (),
                                 "user", "benchmark", "2000-01-01T00:00:00Z")
        result.append(core.contract.build_arm_request("A", (triple,), (source,))[1])
    return tuple(result)


class FakeClient:
    def __init__(self, key, cap, budget, *, malformed_at=None, fail_at=None,
                 close_fail_at=None):
        self.key, self.budget = key, budget
        self.malformed_at, self.fail_at = malformed_at, fail_at
        self.close_fail_at = close_fail_at
        self.closed = False
        budget.register(key, transport.warm.BudgetLimits(*cap))

    @property
    def observed_turns(self):
        return self.budget.snapshot()["questions"][self.key]["turns"]

    @property
    def usage_complete(self):
        return self.budget.snapshot()["questions"][self.key]["usage_complete"]

    def complete_arm(self, arm, request, batch):
        assert not self.closed
        core.contract.validate_arm_request(arm, request, batch)
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
        items = []
        for index, triple in enumerate(batch.triples):
            negative = {"state": "not_established", "support": None}
            item = {"index": index, "original": negative}
            if arm == "A":
                item["alternatives"] = {name: negative for name in
                    core.contract.v3.PREDICATE_ORDER if name != triple.predicate}
            items.append(item)
        return json.dumps({"schema": core.contract.v3.GROUNDING_CONTRACT_VERSION
            if arm == "A" else core.contract.B_SCHEMA,
            "batch_sha256": batch.batch_sha256, "complete": True,
            "classifications": items})

    def close(self):
        self.closed = True
        if self.close_fail_at == self.key:
            raise RuntimeError("PRIVATE_CLEANUP_EXCEPTION")


def _run(tmp_path: Path, *, malformed_at=None, fail_at=None,
         close_fail_at=None):
    private = tmp_path / "private"
    private.mkdir(mode=0o700, parents=True)
    return core.run_campaign(concurrent=transport.warm.concurrent,
        warm=transport.warm, binary="unused", cases_module=cases,
        canary_batches=_canary_batches(), journal=core.PrivateJournal(private),
        client_factory=lambda key, cap, budget: FakeClient(key, cap, budget,
            malformed_at=malformed_at, fail_at=fail_at,
            close_fail_at=close_fail_at)), private


def test_schedule_is_exact_and_nonadaptive():
    units = core.schedule()
    assert len(units) == 29
    assert [(u.kind, u.index, u.arm) for u in units[:3]] == [
        ("control", 2, "A"), ("control", 2, "B"), ("control", 3, "B")]
    assert [(u.kind, u.index, u.arm) for u in units[14:17]] == [
        ("control", 12, "A"), ("control", 12, "B"),
        ("nominated_prefers", 12, "B")]
    assert [(u.kind, u.arm) for u in units[-4:]] == [
        ("table_canary", "A"), ("table_canary", "B"),
        ("prose_canary", "B"), ("prose_canary", "A")]


def test_complete_29_units_and_private_journal(tmp_path):
    result, directory = _run(tmp_path)
    assert result["diagnostic_completed"] is True
    assert result["paid_budget"] == {"turns": 29, "known_tokens": 2900,
        "usage_complete": True, "in_flight": 0, "reserved": 0}
    assert result["malformed_units"] == 0
    assert result["semantic_accuracy_accepted"] is False
    assert result["full_lme_ready"] is False
    assert result["completed_and_clean"] is False
    assert result["units"][16]["batch_sha256"] != result["units"][15]["batch_sha256"]
    assert result["units"][14]["batch_sha256"] == result["units"][15]["batch_sha256"]
    assert all(item["result"]["alternative_states"] is None and
        item["result"]["final_verdicts"] is None for item in result["units"]
        if item["arm"] == "B")
    records = [json.loads(path.read_text()) for path in sorted(directory.iterdir())]
    assert sum(record["phase"] == "before_dispatch" for record in records) == 29
    assert sum(record["phase"] == "response_returned" for record in records) == 29
    assert sum(record["phase"] == "unit_finished" for record in records) == 29
    first = records[0]
    assert first["phase"] == "before_dispatch"
    assert first["arm"] == "A" and first["batch_sha256"]
    assert first["output_schema_sha256"]
    assert "PRIVATE" not in json.dumps(result)


def test_malformed_judgment_continues_but_transport_failure_stops(tmp_path):
    malformed, _ = _run(tmp_path / "malformed", malformed_at="unit-03")
    assert malformed["diagnostic_completed"] is True
    assert malformed["completed_units"] == 29
    assert malformed["malformed_units"] == 1
    assert malformed["units"][3]["result"] is None
    assert malformed["units"][4]["outcome"] == "valid"
    failed, directory = _run(tmp_path / "failed", fail_at="unit-03")
    assert failed["diagnostic_completed"] is False
    assert failed["completed_units"] == 3
    assert failed["paid_budget"]["turns"] == 4
    assert failed["stop_code"] == "infrastructure_or_runtime_failure"
    assert "PRIVATE_PROVIDER_EXCEPTION" not in json.dumps(failed)
    records = [json.loads(path.read_text()) for path in sorted(directory.iterdir())]
    assert sum(record["phase"] == "before_dispatch" for record in records) == 4


def test_public_validator_rejects_raw_or_fabricated_b(tmp_path):
    result, _ = _run(tmp_path)
    result["units"][0]["private_raw"] = "leak"
    with pytest.raises(core.DiagnosticStop):
        core.validate_public_result(result)
    result["units"][0].pop("private_raw")
    b = next(item for item in result["units"] if item["arm"] == "B")
    b["result"]["alternative_states"] = []
    with pytest.raises(core.DiagnosticStop):
        core.validate_public_result(result)


def test_cleanup_failure_preserves_finite_failed_unit(tmp_path):
    result, _ = _run(tmp_path, close_fail_at="unit-03")
    assert result["completed_units"] == 4
    assert result["attempted_units"] == 4
    assert result["paid_budget"]["turns"] == 4
    assert result["stop_code"] == "cleanup_failure"
    assert result["client_cleanup_ok"] is False
    assert result["units"][3]["client_cleanup_ok"] is False
    assert result["diagnostic_completed"] is False
    assert "PRIVATE_CLEANUP_EXCEPTION" not in json.dumps(result)
