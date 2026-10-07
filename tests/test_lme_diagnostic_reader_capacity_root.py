"""Root checks using the frozen candidate's real checkpoint writer, no inference."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from tools.diagnostics import luna_lme_diagnostic_progress_v3 as reader
from tools.diagnostics.tests import test_luna_lme_diagnostic_progress_v1 as prior


@pytest.fixture
def finished(tmp_path, monkeypatch):
    monkeypatch.setattr(prior, "reader", reader)
    root, receipt, _digest = prior.prepared.__wrapped__(tmp_path, monkeypatch)
    source = Path(__file__).resolve().parents[1] / "tools/diagnostics/luna_lme_diagnostic_v2.py"
    shutil.copy2(source, root / "code/tools/diagnostics/luna_lme_diagnostic_v2.py")
    receipt["schema"] = "luna-lme-diagnostic-launch-v2"
    prior._json(root / "launch-receipt.json", receipt)
    digest = reader._sha(root / "launch-receipt.json")
    prior._json(root / "launch-attempt.json", {"receipt_sha256": digest, "one_shot": True})
    prior._json(root / "launch-command-result.json", {"returncode": 0})
    checkpoint = prior._frozen_checkpoint(root)
    terminal = prior._terminal(checkpoint)
    terminal["budget"].update(resource_fault=None, first_failure=None,
        resource_observation={"current": 4, "peak": 170, "limit": 256, "denials": 0})
    monkeypatch.setattr(reader, "_runtime", lambda _receipt: "clean_exit")
    return root, digest, terminal


def test_real_four_row_success_retains_quality_and_index_health(finished):
    root, digest, terminal = finished
    prior._json(root / "run/diagnostic-result.json", terminal)
    result = reader.inspect(root, digest)
    assert result["completed_diagnostic_and_clean"] is True
    assert result["runtime_cleanup_verified"] is True
    assert result["scored_count"] == result["selected_denominator"] == 4
    assert result["correct_count"] == 2
    assert result["strict_indexing_healthy_for_all"] is False
    assert result["summary_degraded_sessions_total"] == 1
    assert result["canary_model_gold_match"] is False
    assert result["resource_observation"]["peak"] == 170


@pytest.mark.parametrize("mutation", [
    lambda b: b.pop("resource_observation"),
    lambda b: b["resource_observation"].update(denials=1),
    lambda b: b["resource_observation"].update(limit=128),
    lambda b: b["resource_observation"].update(limit=257),
    lambda b: b["resource_observation"].update(peak=257),
    lambda b: b["resource_observation"].update(current=171),
    lambda b: b["resource_observation"].update(denials=False),
    lambda b: b.update(resource_fault="raw private text"),
    lambda b: b.update(resource_fault="resource_task_denial"),
])
def test_malformed_or_contradictory_resource_cannot_claim_success(finished, mutation):
    root, digest, terminal = finished
    mutation(terminal["budget"])
    prior._json(root / "run/diagnostic-result.json", terminal)
    with pytest.raises(ValueError):
        reader.inspect(root, digest)


@pytest.mark.parametrize("fault", ["resource_task_denial", "resource_observer_unverified"])
def test_resource_fault_never_clean_even_with_four_scored_and_exit_zero(finished, fault):
    root, digest, terminal = finished
    budget = terminal["budget"]
    budget.update(resource_fault=fault, stopped=True, stop_code=fault)
    terminal["campaign_stop"] = fault
    if fault == "resource_task_denial":
        budget["resource_observation"]["denials"] = 1
    else:
        budget["resource_observation"] = None
    prior._json(root / "run/diagnostic-result.json", terminal)
    result = reader.inspect(root, digest)
    assert result["completed_diagnostic_and_clean"] is False
    assert result["runtime_cleanup_verified"] is True
    assert result["resource_fault"] == fault


def test_first_fault_counters_cannot_exceed_terminal_history(finished):
    root, digest, terminal = finished
    budget = terminal["budget"]
    budget.update(stopped=True, stop_code="rpc_failure:thread/start")
    terminal["campaign_stop"] = "rpc_failure:thread/start"
    budget["first_failure"] = {
        "code": "rpc_failure:thread/start", "phase": "preflight", "rpc": "thread/start",
        "process_index": 1, "request_index": 1, "retired_count": 0, "queue_count": 0,
        "turn_admitted": False, "known_usage": True, "usage_complete": False,
        "rpc_error": {"category": "internal_error", "code": -32603},
        "resource_observation": {"current": 120, "peak": 150, "limit": 256, "denials": 0}}
    prior._json(root / "run/diagnostic-result.json", terminal)
    result = reader.inspect(root, digest)
    assert result["completed_diagnostic_and_clean"] is False
    assert result["first_failure"]["rpc_error_code"] == -32603
    assert result["first_failure"]["resource_observation"]["current"] == 120
    for change in ({"peak": 180}, {"denials": 1}):
        bad = copy.deepcopy(terminal)
        bad["budget"]["first_failure"]["resource_observation"].update(change)
        prior._json(root / "run/diagnostic-result.json", bad)
        with pytest.raises(ValueError, match="resource_observation_regressed"):
            reader.inspect(root, digest)
