"""Offline source, 600-second admission, and finite timeout-metadata controls."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from benchmarks import chatgpt_plan_lme_v8 as bridge
from tools.diagnostics import siwc_lme_diagnostic_progress_v13 as reader
from tools.diagnostics import siwc_lme_diagnostic_v11 as runner
from tools.diagnostics import siwc_lme_diagnostic_bundle_v9 as prior_bundle
from tests.test_lme_chatgpt_plan_owner_v1 import fixture_vm_state


REPO = Path(__file__).resolve().parents[3]
FROZEN = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle")
ACCEPTED_CODE = Path("/private/tmp/hymem-repaired-four-root-dtBv2p/bundle/code")
BUCKETS = ("output_text_delta", "reasoning_text_delta", "reasoning_summary_text_delta",
           "lifecycle", "completion", "failure", "other")


def test_actual_broker_admits_600_without_provider(tmp_path, monkeypatch):
    def no_network(*_args, **_kwargs):
        pytest.fail("network forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    seen = []

    def invented_response(_credentials, _system, _user, _schema, *, timeout):
        seen.append(timeout)
        return bridge.transport.Completed(
            text='{"invented":true}', input_tokens=1, output_tokens=1,
            total_tokens=2, cached_input_tokens=0, reasoning_output_tokens=0)

    with bridge.owner.CredentialBroker(fixture_vm_state(tmp_path), Path(sys.executable)) as broker:
        budget = bridge.SharedBudget(bridge.warm.BudgetLimits(8012, 48_160_000, 25_200))
        client = bridge.SIWCLMEClient(
            broker, budget, "invented-q", bridge.warm.BudgetLimits(2000, 12_000_000, 23_400),
            response_call=invented_response)
        assert client.complete(bridge.LLMRequest("invented system", "invented user")) == (
            '{"invented":true}')
        assert len(seen) == 1 and 599 < seen[0] <= 600
        summary = client.diagnostic_summary()
        assert summary["admitted_turns"] == summary["internal_http_attempts"] == 1
        assert summary["successes"] == 1 and summary["known_tokens"] == 2
        assert budget.snapshot()["reserved"] == budget.snapshot()["in_flight"] == 0


def observation(*, buckets=True):
    counts = dict.fromkeys(BUCKETS, 0) if buckets else None
    if counts is not None:
        counts["output_text_delta"] = 2
    return {"child_phase": "stream_read", "parent_timeout_site": "result_wait",
        "last_progress_elapsed_ms": 599_000, "parent_elapsed_ms": 600_000,
        "elapsed_saturated": False, "snapshot_valid": True,
        "timeout_allowance_ms": 600_000, "wire_bytes": 100, "event_count": 2,
        "event_buckets": counts, "completion_seen": False, "result_ready": False,
        "result_ipc_started": False, "child_alive_when_sampled": True}


def test_reader_reconciles_finite_buckets_and_preserves_unknown():
    good = observation()
    assert bridge.transport.sanitize_timeout_observation(good) == good
    reader.timeout_observation(good)
    unknown = observation(buckets=False)
    unknown.update(child_phase="unknown", snapshot_valid=False,
                   last_progress_elapsed_ms=None, wire_bytes=None, event_count=None,
                   completion_seen=None, result_ready=None, result_ipc_started=None)
    assert bridge.transport.sanitize_timeout_observation(unknown) == unknown
    reader.timeout_observation(unknown)
    saturated = observation(buckets=False)
    saturated["elapsed_saturated"] = True
    assert bridge.transport.sanitize_timeout_observation(saturated) == saturated
    reader.timeout_observation(saturated)
    for mutation in (
        {"event_buckets": None},
        {"event_buckets": dict(good["event_buckets"], other=1)},
        {"event_buckets": {**good["event_buckets"], "completion": 1,
                           "output_text_delta": 1}},
        {"event_buckets": {**good["event_buckets"], "other": True}},
        {"timeout_allowance_ms": 600_001},
    ):
        wrong = copy.deepcopy(good)
        wrong.update(mutation)
        assert bridge.transport.sanitize_timeout_observation(wrong) is None
        with pytest.raises(ValueError, match="timeout_observation_invalid"):
            reader.timeout_observation(wrong)


def test_source_and_policy_pins():
    assert bridge.MAX_INVOCATION == 600.0
    assert bridge.SIWCLMEClient.__init__.__kwdefaults__["response_call"] is bridge.transport.complete
    assert runner.MAX_LIMITS == {"campaign": (8012, 48_160_000, 25_200),
        "question": (2000, 12_000_000, 23_400), "canary": (12, 160_000, 600)}
    assert reader.LIMITS == {key: list(value) for key, value in runner.MAX_LIMITS.items()}
    assert reader.RUN_SCHEMA == runner.SCHEMA == "siwc-lme-semantic-diagnostic-v11"
    assert "benchmarks/chatgpt_plan_responses_v10.py" not in runner.SIWC_PINS
    assert "tools/diagnostics/lme_chatgpt_plan_owner_v2.py" not in runner.SIWC_PINS
    for relative, digest in runner.SIWC_PINS.items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == digest
    assert reader.RUNNER_SHA256 == hashlib.sha256((REPO / reader.RUNNER_RELATIVE).read_bytes()).hexdigest()
    assert reader.LAUNCHER_SHA256 == hashlib.sha256(
        (REPO / "tools/diagnostics" / reader.LAUNCHER_NAME).read_bytes()).hexdigest()


def test_source_only_isolated_closure_rejects_old_transport(tmp_path):
    if not FROZEN.is_dir() or not ACCEPTED_CODE.is_dir():
        pytest.skip("frozen local source unavailable")
    root = tmp_path.resolve() / ".hymem-siwc-lme-diagnostic-invented01"
    prior_bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
        candidate=FROZEN / "candidate", map_path=FROZEN / "source-map.json", output=root)
    code = root / "code"
    old = ("benchmarks/chatgpt_plan_lme_v7.py", "benchmarks/chatgpt_plan_responses_v10.py",
           "tools/diagnostics/siwc_lme_diagnostic_v10.py",
           "tools/diagnostics/lme_chatgpt_plan_owner_v2.py",
           "tools/diagnostics/lme_chatgpt_plan_refresh_v2.py")
    old_transport_bytes = (code / old[1]).read_bytes()
    for relative in old:
        (code / relative).unlink()
    new = ("benchmarks/chatgpt_plan_lme_v8.py", "benchmarks/chatgpt_plan_responses_v11.py",
           "tools/diagnostics/siwc_lme_diagnostic_v11.py",
           "tools/diagnostics/lme_chatgpt_plan_owner_v3.py",
           "tools/diagnostics/lme_chatgpt_plan_refresh_v3.py")
    for relative in new:
        (code / relative).write_bytes((REPO / relative).read_bytes())
    assert len(list(code.rglob("*.py"))) == 26
    script = r"""
import importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1]); code=root/'code'
source=code/'tools/diagnostics/siwc_lme_diagnostic_v11.py'
spec=importlib.util.spec_from_file_location('isolated_runner11',source)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
assert pathlib.Path(loaded['siwc'].__file__)==code/'benchmarks/chatgpt_plan_lme_v8.py'
assert pathlib.Path(loaded['siwc'].transport.__file__)==code/'benchmarks/chatgpt_plan_responses_v11.py'
assert pathlib.Path(loaded['siwc'].owner.__file__)==code/'tools/diagnostics/lme_chatgpt_plan_owner_v3.py'
assert pathlib.Path(loaded['siwc'].owner._pinned_refresh().__file__)==code/'tools/diagnostics/lme_chatgpt_plan_refresh_v3.py'
assert loaded['siwc'].SIWCLMEClient.__init__.__kwdefaults__['response_call'] is loaded['siwc'].transport.complete
print('source_only_ok')
"""
    command = [sys.executable, "-I", "-B", "-c", script, str(root)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "source_only_ok"
    (code / new[1]).write_bytes(old_transport_bytes)
    rejected = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert rejected.returncode != 0
    assert "transport_or_helper_drift" in rejected.stderr
