"""Offline source and one-worker lifecycle controls for the v3 probe."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import codex_subscription_timeout_v2 as observer
from tools.diagnostics import luna_timeout_probe_v3 as probe


def _attest(stage, index):
    assert stage in {"initial", "before_admission", "terminal"}
    assert index is None or 0 <= index < 16
    return {"containment": True, "denials": 0, "oom": 0}


def _observation(*, success=True, usage=True):
    phases = {name: None for name in
              ("startup", "preflight", "turn_start", "events", "unsubscribe", "cleanup")}
    return {"schema": "warm_timeout_observation_v1",
        "status": "success" if success else "failure",
        "failure_code": None if success else "timeout",
        "known_usage": usage, "total_seconds": 0.01,
        "invocation_start_monotonic": 1.0, "deadline_monotonic": 121.0,
        "remaining_at_turn_start_seconds": 119.0, "phase_seconds": phases,
        "reader_state": "running", "queue_depth": 0, "reader_enqueued": 1,
        "reader_dequeued": 1, "reader_enqueued_saturated": False,
        "reader_dequeued_saturated": False, "last_enqueue_monotonic": 2.0,
        "last_dequeue_monotonic": 2.001, "oldest_enqueue_monotonic": None,
        "last_queue_delay_seconds": 0.001,
        "last_event_consumed_monotonic": 2.001, "events_returned": 1,
        "observed": None, "process_alive": True, "precleanup": None}


def _factory(*, fail_at=None, swap_at=None, tokens=19):
    created = []
    class Process:
        pid = 7631
        def poll(self):
            return None
    class Session:
        def __init__(self):
            self.process = Process()
    class Client:
        def __init__(self, binary, budget, key, limits, **kwargs):
            assert binary == "/synthetic" and key == "worker-0"
            assert limits.turns == 16
            assert kwargs == {"max_requests": 16, "max_age_seconds": 300}
            budget.register(key, limits)
            self.budget, self.key = budget, key
            self.session, self.directory = None, None
            self.records = []
            self.processes_started = self.rotations = 0
            self.cold_calls = self.warm_calls = self.requests_on_process = 0
            created.append(self)

        def complete(self, request):
            index = len(self.records)
            assert request.system == probe.SYSTEM and request.user == probe.USERS[index]
            assert request.max_tokens == 1024 and request.temperature == 0.0
            self.budget.reserve(self.key)
            if self.session is None:
                self.session = Session()
                self.processes_started += 1
                self.cold_calls += 1
            else:
                self.warm_calls += 1
            self.budget.before_turn(self.key, {"auth": "chatgpt", "model": "gpt-6-luna",
                "config_isolation_admitted": True, "inference_enabled": False,
                "quota_windows": [{"remaining_percent": 100}]})
            if index + 1 == fail_at:
                self.records.append(_observation(success=False, usage=False))
                self.budget.record_first_failure("timeout", {"phase": "run",
                    "rpc": "turn/events", "turn_admitted": True,
                    "known_usage": False, "usage_complete": False,
                    "process_index": 1, "request_index": index + 1,
                    "retired_count": 0, "queue_count": 0,
                    "known_tokens": self.budget.snapshot()["known_tokens"]})
                self.budget.settle(self.key, used=None, turn_started=True,
                                   failure="timeout")
                self.session = None
                raise observer.base.SubscriptionTransportError("timeout")
            self.budget.settle(self.key, used=tokens, turn_started=True)
            self.requests_on_process += 1
            self.records.append(_observation())
            if index + 1 == swap_at:
                self.session.process = Process()
            return "synthetic answer"

        def diagnostic_records(self):
            return tuple(copy.deepcopy(self.records))

        def close(self):
            self.session = self.directory = None

    return Client, created


def _run(**kwargs):
    factory, created = _factory(**kwargs)
    result = probe.run_probe(observer, observer.base.LLMRequest, "/synthetic",
                             attest=_attest, client_factory=factory)
    assert probe.validate_result(result, observer)
    assert len(created) == 1 and created[0].session is None
    return result


def test_source_only_preparation_and_isolated_loader(tmp_path):
    root = tmp_path / "prepared"
    prepared = probe.prepare(root)
    assert prepared["model_calls"] == 0 and prepared["host_launch_authorized"] is False
    receipt = probe.verify_prepared(root, prepared["receipt_sha256"])
    assert len(receipt["source_sha256"]) == 14
    assert receipt["source_sha256"][probe.SELF_RELATIVE] == hashlib.sha256(
        Path(probe.__file__).read_bytes()).hexdigest()
    assert receipt["source_sha256"]["benchmarks/codex_subscription_timeout_v2.py"] == (
        "476e00bcae40c0a061a1b75b10797d4563fb9822012917382e27d9b5a6c93287")
    code = """import importlib.util,json,sys
from pathlib import Path
p=Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('isolated_probe',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
observer,request=m.load_prepared(Path(sys.argv[2]),sys.argv[3])
print(json.dumps({'model':observer.base.MODEL,'request':request.__name__}))
"""
    output = subprocess.run([sys.executable, "-I", "-B", "-c", code,
        str(root / "code" / probe.SELF_RELATIVE), str(root), prepared["receipt_sha256"]],
        capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(output.stdout) == {"model": "gpt-6-luna", "request": "LLMRequest"}


def test_one_process_sixteen_positions_and_private_identity():
    result = _run()
    assert result["status"] == "observed_success"
    assert result["returned"] == result["turns"] == 16
    assert result["known_tokens"] == 16 * 19
    assert result["lifecycle"] == {"processes_started": 1, "rotations": 0,
        "cold_calls": 1, "warm_calls": 15, "requests_on_process": 16,
        "identity_checks": 16, "successful_positions": list(range(1, 17)),
        "coverage_complete": True, "coverage_reason": None}
    encoded = json.dumps(result)
    assert "synthetic answer" not in encoded and "paper shape" not in encoded
    assert "pid" not in encoded and "threadId" not in encoded


def test_failure_at_fifteen_keeps_primary_fault_and_unknown_usage():
    result = _run(fail_at=15)
    assert result["status"] == "incomplete_or_failed"
    assert result["returned"] == 14 and result["attempted"] == 15
    assert result["not_attempted"] == 1 and result["turns"] == 15
    assert result["usage_complete"] is False
    assert result["first_failure"]["code"] == "timeout"
    assert result["first_failure"]["request_index"] == 15
    assert result["lifecycle"]["successful_positions"] == list(range(1, 15))
    assert result["lifecycle"]["coverage_complete"] is False


def test_same_counters_but_different_process_is_integrity_failure():
    result = _run(swap_at=4)
    assert result["status"] == "incomplete_or_failed"
    assert result["returned"] == 4 and result["not_attempted"] == 12
    assert result["first_failure"] is None
    assert result["lifecycle"]["coverage_reason"] == "identity_mismatch"
    assert result["lifecycle"]["identity_checks"] == 3


def test_validator_rejects_faked_coverage_and_bool_as_int():
    result = _run()
    for edit in (
        lambda v: v["lifecycle"].update(processes_started=True),
        lambda v: v["limits"].update(workers=True),
        lambda v: v["policy"].update(rerolls=False),
        lambda v: v["records"][14].update(sequence_position=1),
        lambda v: v["records"][14].update(identity_checked=False),
        lambda v: v["lifecycle"].update(successful_positions=[1] * 16),
        lambda v: v["lifecycle"].update(pid=7631),
    ):
        value = copy.deepcopy(result)
        edit(value)
        assert not probe.validate_result(value, observer)


def test_containment_gate_and_cli_are_fail_closed():
    with pytest.raises(ValueError, match="containment_attestor_required"):
        probe.run_probe(observer, observer.base.LLMRequest, "/synthetic")
    with pytest.raises(SystemExit):
        probe.main(["--run-root", "/synthetic"])
