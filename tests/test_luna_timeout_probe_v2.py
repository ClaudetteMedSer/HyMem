"""Offline source, privacy, accounting, and four-worker probe controls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import threading

import pytest

from tools.diagnostics import luna_timeout_probe_v2 as probe


REPO = Path(__file__).resolve().parents[1]


def observer_module():
    from benchmarks import codex_subscription_timeout_v2 as observer
    return observer


def sample_record(*, success=True, usage=True, code="timeout"):
    phases = {name: None for name in
        ("startup", "preflight", "turn_start", "events", "unsubscribe", "cleanup")}
    return {"schema": "warm_timeout_observation_v1",
        "status": "success" if success else "failure",
        "failure_code": None if success else code, "known_usage": usage,
        "total_seconds": 0.01, "invocation_start_monotonic": 1.0,
        "deadline_monotonic": 121.0, "remaining_at_turn_start_seconds": 119.0,
        "phase_seconds": phases, "reader_state": "running",
        "queue_depth": 0, "reader_enqueued": 1, "reader_dequeued": 1,
        "reader_enqueued_saturated": False, "reader_dequeued_saturated": False,
        "last_enqueue_monotonic": 2.0, "last_dequeue_monotonic": 2.001,
        "oldest_enqueue_monotonic": None, "last_queue_delay_seconds": 0.001,
        "last_event_consumed_monotonic": 2.001, "events_returned": 1,
        "observed": None, "process_alive": True, "precleanup": None}


def fake_factory(*, fail_id=None, tokens=100, cleanup_failure=False, barrier=None):
    observer = observer_module()
    registry = {"clients": [], "admitted": [], "closed": []}

    class FakeClient:
        def __init__(self, binary, budget, key, limits, **kwargs):
            assert binary == "/invented/binary"
            assert kwargs == {"max_requests": 16, "max_age_seconds": 300}
            self.budget, self.key = budget, key
            budget.register(key, limits)
            self.session, self.directory = None, None
            self.records = []
            registry["clients"].append(self)

        def complete(self, request):
            index = int(re.search(r"item (\d+)", request.user).group(1))
            assert request.system == probe.SYSTEM
            assert request.user == probe.USERS[index]
            try:
                self.budget.reserve(self.key)
                self.budget.before_turn(self.key, {"auth": "chatgpt", "model": "gpt-6-luna",
                    "config_isolation_admitted": True, "inference_enabled": False,
                    "quota_windows": [{"remaining_percent": 100}]})
            except BaseException:
                self.records.append(sample_record(success=False, usage=None,
                    code="fixed_other"))
                raise
            registry["admitted"].append(index)
            if barrier is not None and index < 4:
                barrier.wait(timeout=5)
            if index == fail_id:
                self.records.append(sample_record(success=False, usage=False))
                self.budget.record_first_failure("timeout", {"phase": "run",
                    "rpc": "turn/events", "turn_admitted": True, "known_usage": False,
                    "usage_complete": False, "process_index": 1,
                    "request_index": 1, "retired_count": 0, "queue_count": 0,
                    "known_tokens": self.budget.snapshot()["known_tokens"]})
                self.budget.settle(self.key, used=None, turn_started=True, failure="timeout")
                raise observer.base.SubscriptionTransportError("timeout")
            self.records.append(sample_record())
            self.budget.settle(self.key, used=tokens, turn_started=True)
            return "invented response never retained"

        def diagnostic_records(self):
            return tuple(copy.deepcopy(self.records))

        def close(self):
            registry["closed"].append(self.key)
            if cleanup_failure and self.key == "worker-0":
                raise RuntimeError("private cleanup details")

    return FakeClient, registry


def attestor(stage, record_id):
    assert stage in {"initial", "before_admission", "terminal"}
    assert record_id is None or 0 <= record_id < 16
    return {"containment": True, "denials": 0, "oom": 0}


def run_fake(**kwargs):
    factory, registry = fake_factory(**kwargs)
    result = probe.run_probe(observer_module(), observer_module().base.LLMRequest,
        "/invented/binary", attest=attestor, client_factory=factory)
    return result, registry


def test_prepare_and_isolated_import_without_inference(tmp_path):
    root = tmp_path / "probe"
    prepared = probe.prepare(root)
    assert prepared["prepared"] is True and prepared["model_calls"] == 0
    receipt = probe.verify_prepared(root, prepared["receipt_sha256"])
    assert receipt["binary_path"] == probe.BINARY_PATH
    assert receipt["binary_sha256"] == probe.BINARY_SHA256
    assert receipt["pending_host"]["one_shot_host_receipt"] == "pending"
    assert receipt["source_sha256"]["benchmarks/codex_subscription_timeout_v2.py"] == (
        "476e00bcae40c0a061a1b75b10797d4563fb9822012917382e27d9b5a6c93287")
    code = """import importlib.util,json,sys
from pathlib import Path
p=Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('isolated_probe',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
observer,request=m.load_prepared(Path(sys.argv[2]),sys.argv[3])
print(json.dumps({'ok': observer.base.MODEL=='gpt-6-luna',
                  'request': request.__name__, 'calls':0}))
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code,
        str(root / "code" / probe.SELF_RELATIVE), str(root), prepared["receipt_sha256"]],
        capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(completed.stdout) == {"ok": True, "request": "LLMRequest", "calls": 0}


def test_tamper_symlink_and_extra_file_fail_closed(tmp_path):
    root = tmp_path / "probe"
    digest = probe.prepare(root)["receipt_sha256"]
    target = root / "code" / "benchmarks/codex_subscription_warm_v9.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source_drift"):
        probe.verify_prepared(root, digest)
    target.write_bytes((REPO / "benchmarks/codex_subscription_warm_v9.py").read_bytes())
    extra = root / "code" / "extra.py"
    extra.write_text("secret")
    with pytest.raises(ValueError, match="extra_or_missing_file"):
        probe.verify_prepared(root, digest)
    extra.unlink()
    target.unlink()
    target.symlink_to(REPO / "benchmarks/codex_subscription_warm_v9.py")
    with pytest.raises(ValueError, match="source_symlink"):
        probe.verify_prepared(root, digest)


def test_loader_rejects_changed_bytes_before_execution(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    marker = tmp_path / "executed"
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    with pytest.raises(ValueError, match="module_source_drift"):
        probe._load_source("invented_probe_tamper", source, original_sha)
    assert not marker.exists()
    assert "invented_probe_tamper" not in sys.modules


def test_deny_by_default_and_no_run_cli():
    with pytest.raises(ValueError, match="containment_attestor_required"):
        probe.run_probe(observer_module(), observer_module().base.LLMRequest, "/invented/binary")
    with pytest.raises(SystemExit):
        probe.main(["--run-root", "/invented"])


def test_success_four_workers_and_finite_public_result():
    result, registry = run_fake(barrier=threading.Barrier(4))
    assert result["status"] == "observed_success"
    assert result["attempted"] == result["returned"] == result["turns"] == 16
    assert result["known_tokens"] == 1600 and result["usage_complete"] is True
    assert result["independent_recursive_cleanup_verified"] is None
    assert len(registry["clients"]) == len(registry["closed"]) == 4
    assert sorted(registry["admitted"]) == list(range(16))
    assert probe.validate_result(result, observer_module())
    encoded = json.dumps(result, sort_keys=True)
    assert "Invented timing" not in encoded
    assert "invented response" not in encoded
    assert "paper shape" not in encoded


def test_timeout_unknown_usage_first_fault_and_peers_settle():
    result, registry = run_fake(fail_id=0, barrier=threading.Barrier(4))
    assert result["status"] == "incomplete_or_failed"
    assert result["usage_complete"] is False
    assert result["reserved"] == result["in_flight"] == 0
    assert result["first_failure"]["code"] == "timeout"
    assert result["records"][0]["observation"]["known_usage"] is False
    assert result["attempted"] + result["not_attempted"] == 16
    assert len(registry["closed"]) == 4
    assert probe.validate_result(result, observer_module())


def test_known_token_overshoot_is_failed_not_clean():
    result, _ = run_fake(tokens=12_000)
    assert 160_000 < result["known_tokens"] <= 192_000
    assert result["known_token_overshoot"] is True
    assert result["status"] == "incomplete_or_failed"
    assert result["budget_stopped"] is True


def test_cleanup_failure_and_attestation_failure_are_not_clean():
    result, _ = run_fake(cleanup_failure=True)
    assert result["status"] == "incomplete_or_failed"
    assert result["client_cleanup_verified"] is False
    factory, registry = fake_factory()
    def deny(stage, record_id):
        return {"containment": stage != "initial", "denials": 0, "oom": 0}
    blocked = probe.run_probe(observer_module(), observer_module().base.LLMRequest,
        "/invented/binary", attest=deny, client_factory=factory)
    assert blocked["attempted"] == 0 and blocked["not_attempted"] == 16
    assert blocked["status"] == "incomplete_or_failed" and not registry["clients"]


def test_projection_rejects_private_injection_and_inconsistent_counts():
    result, _ = run_fake()
    bad = copy.deepcopy(result)
    bad["records"][0]["observation"]["private_prompt"] = "secret"
    assert not probe.validate_result(bad, observer_module())
    bad = copy.deepcopy(result)
    bad["first_failure"] = {"code": "timeout", "message": "secret"}
    assert not probe.validate_result(bad, observer_module())
    bad = copy.deepcopy(result)
    bad["returned"] = 15
    assert not probe.validate_result(bad, observer_module())
    bad = copy.deepcopy(result)
    bad["known_tokens"] = float("nan")
    assert not probe.validate_result(bad, observer_module())
