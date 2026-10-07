"""Independent offline campaign/identity controls; no account or provider I/O."""
import ast
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from benchmarks import chatgpt_plan_lme_v1 as bridge
from hymem.extraction.llm import LLMRequest
from tools.diagnostics import siwc_lme_diagnostic_v1 as runner

_strict_path = Path('/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle/candidate/benchmarks/strictness.py')
assert hashlib.sha256(_strict_path.read_bytes()).hexdigest() == '0bef9c3371d97b6bee35c27c070d99507dc50b709b20ebe4892035cce27fd911'
_spec = importlib.util.spec_from_file_location('benchmarks._siwc_root_frozen_strictness', _strict_path)
strictness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = strictness
_spec.loader.exec_module(strictness)


def rig(tmp_path, monkeypatch, *, failure=False, wrong_grant=False, observed=False):
    dataset = tmp_path / "invented.json"
    dataset.write_text("[]")
    original_sha = runner._sha
    monkeypatch.setattr(runner, "_sha", lambda p: runner.DATASET_SHA256
                        if p == dataset else original_sha(p))
    owner = object.__new__(bridge.owner.CredentialBroker)
    owner.identity_digest = "0" * 64 if wrong_grant else runner.GRANT_IDENTITY_SHA256
    closes, leases = [], []
    monkeypatch.setattr(bridge.owner.CredentialBroker, "close", lambda _: closes.append(True))
    def acquire(self, **kw):
        leases.append(True)
        return bridge.owner.CredentialLease("invented", int(time.time()) + 900)
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire", acquire)
    original_init = bridge.SIWCLMEClient.__init__
    barrier = Barrier(4)
    def init(self, broker, budget, key, limits, **kwargs):
        def response(*args, **kw):
            if key != "canary":
                barrier.wait(timeout=5)
                if failure and key == "q-0000":
                    raise bridge.transport.TransportError("http_failure", 403, "json", "json")
            return bridge.transport.Completed("invented", 10, 2, 12, 0, 0)
        original_init(self, broker, budget, key, limits, response_call=response)
    monkeypatch.setattr(bridge.SIWCLMEClient, "__init__", init)
    class Prior:
        @staticmethod
        def atomic_private(path, value):
            path.write_text(json.dumps(value, allow_nan=False))
    loaded = {"source_only": False, "dataset": dataset,
        "questions": [{"question_id": f"invented-{i}"} for i in range(4)],
        "strictness": strictness, "diagnostic": SimpleNamespace(MODE="semantic_diagnostic_v1"),
        "siwc": bridge, "warm": bridge.warm, "prior": Prior()}
    def canary(loaded, budget, limits, directory, observations):
        client = runner.make_dual(loaded, budget, "canary", limits, directory,
                                 observations, "canary")
        try:
            assert client.complete(LLMRequest("invented", "invented")) == "invented"
            return {"structural_valid": True, "model_gold_match": False,
                "quality_failure_reason": "invented_semantic", "semantic_failure_proved": True,
                "completion_calls": 1}
        finally:
            observations.capture_pair("canary")
            client.close()
    monkeypatch.setattr(runner, "run_live_canary", canary)
    def worker(loaded, budget, limits, question, index, output, indexing_seconds, observations):
        prefix = f"question.{index}"
        client = runner.make_dual(loaded, budget, f"q-{index:04d}", limits, output,
                                 observations, prefix)
        try:
            client.complete(LLMRequest("invented", "invented"))
            return {"projection": {"question_id": question["question_id"],
                "correct": index % 2 == 0, "strict_indexing_healthy": False,
                "benchmark_failure": None, "diagnostic_kind": "semantic_quarantine",
                "quarantined_chunks": 1, "summary_degraded_sessions": 2,
                "context_sha": "a" * 64},
                "accounting": {"reader": {"attempts": 1, "returned": 1,
                    "turns": 1, "known_tokens": 12}}, "stop_code": None}
        except bridge.BridgeError:
            return {"projection": None, "accounting": None, "stop_code": "question_failure"}
        finally:
            observations.capture_pair(prefix)
            client.close()
    monkeypatch.setattr(runner, "_question_worker", worker)
    if observed:
        monkeypatch.setattr(runner, "_resource_sample", lambda _: {
            "current": 1, "peak": 9, "limit": 256, "denials": 0})
    def run():
        return runner.run_campaign(loaded, output=tmp_path / "run",
            campaign_limits=bridge.warm.BudgetLimits(*runner.MAX_LIMITS["campaign"]),
            canary_limits=bridge.warm.BudgetLimits(*runner.MAX_LIMITS["canary"]),
            question_limits=bridge.warm.BudgetLimits(*runner.MAX_LIMITS["question"]),
            indexing_seconds=10800, workers=4,
            helper_sha256=runner.DIAGNOSTIC_HELPER_SHA256,
            containment=lambda _: True, broker_factory=lambda: owner,
            resource_cgroup="/invented" if observed else None)
    return run, closes, leases, loaded


def test_actual_four_worker_ledger_and_ten_views_reconcile(tmp_path, monkeypatch):
    run, closes, leases, loaded = rig(tmp_path, monkeypatch)
    result = run()
    assert result["diagnostic_complete"] is True
    assert result["scored_count"] == 4 and result["correct_count"] == 2
    assert result["quality_accuracy_full_selected"] == .5
    assert result["strict_unhealthy_count"] == 4
    assert result["canary"]["model_gold_match"] is False
    assert result["budget"]["turns"] == 5 and result["budget"]["known_tokens"] == 60
    assert result["budget"]["usage_complete"]
    assert len(leases) == 5 and closes == [True] and "broker" not in loaded
    assert len(result["siwc_observations"]) == 10
    assert result["siwc_pilot_projection"]["aggregate"]["known_tokens"] == 60
    with pytest.raises(ValueError, match="campaign_preflight_invalid"):
        run()


def test_one_failed_admitted_turn_retains_unknown_usage_and_partial_evidence(tmp_path, monkeypatch):
    run, closes, _, _ = rig(tmp_path, monkeypatch, failure=True)
    result = run()
    assert result["diagnostic_complete"] is False
    assert result["scored_count"] == 3 and result["failed_or_unscored_count"] == 1
    assert result["quality_accuracy_full_selected"] is None
    assert result["budget"]["turns"] == 5 and result["budget"]["known_tokens"] == 48
    assert result["budget"]["usage_complete"] is False
    first = result["siwc_observations"]["question.0.ordinary"]["summary"]["first_failure"]
    assert first["code"] == "http_failure" and first["unknown_usage"] is True
    assert closes == [True]
    assert "invented" not in repr(first)


def test_grant_change_blocks_canary_before_any_acquire(tmp_path, monkeypatch):
    run, closes, leases, loaded = rig(tmp_path, monkeypatch, wrong_grant=True)
    result = run()
    assert not result["diagnostic_complete"] and result["scored_count"] == 0
    assert leases == [] and closes == [True] and "broker" not in loaded
    assert result["budget"]["turns"] == 0
    assert all(v == {"status": "unknown"} for v in result["siwc_observations"].values())


def test_canonical_receipt_rejects_type_coercion_and_duplicates(tmp_path, monkeypatch):
    expected = {"schema": runner.RECEIPT_SCHEMA, "workers": 4,
                "store": False, "model": "gpt-5.6-luna", "grant_identity_sha256": "a"*64}
    monkeypatch.setattr(runner, "receipt_for", lambda *_: expected)
    path = tmp_path / "launch-receipt.json"
    for changed in [expected, {**expected, "workers": True}, {**expected, "workers": 4.0},
                    {**expected, "store": 0}, {**expected, "model": "gpt-6-luna"}]:
        raw = runner._canonical(changed)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        (tmp_path / "launch-attempt.json").write_bytes(runner._canonical(
            {"receipt_sha256": digest, "one_shot": True}))
        if changed is expected:
            assert runner.verify_launch_receipt(tmp_path, digest, {}) == expected
        else:
            with pytest.raises(ValueError):
                runner.verify_launch_receipt(tmp_path, digest, {})
    raw = runner._canonical(expected).replace(b'"workers":4', b'"workers":4,"workers":4')
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / "launch-attempt.json").write_bytes(runner._canonical(
        {"receipt_sha256": digest, "one_shot": True}))
    with pytest.raises(ValueError):
        runner.verify_launch_receipt(tmp_path, digest, {})


def test_canary_stage_and_quality_checks_unchanged_from_accepted_runner():
    prior_path = Path(runner.__file__).with_name("luna_lme_diagnostic_v10.py")
    def definitions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    before, after = definitions(prior_path), definitions(Path(runner.__file__))
    for name in ("CanaryRecorder", "run_live_canary", "AccountedClient", "validate_diagnostic_row",
                 "RegistrationAlias", "DualClient", "verify_live_containment", "_resource_sample"):
        assert before[name] == after[name], name
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == "1b83e3d3ecfa29cdaab5f5ad5aa0b8c4da844723c280de29b952449ef6902048"
