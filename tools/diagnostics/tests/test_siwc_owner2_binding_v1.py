"""Offline source and real-owner controls for the 300-second SIWC boundary."""
from __future__ import annotations

import hashlib
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from benchmarks import chatgpt_plan_lme_v7 as bridge
from tools.diagnostics import siwc_lme_diagnostic_bundle_v8 as previous_bundle
from tools.diagnostics import siwc_lme_diagnostic_progress_v12 as reader
from tools.diagnostics import siwc_lme_diagnostic_v10 as runner
from tests.test_lme_chatgpt_plan_owner_v1 import fixture_vm_state


REPO = Path(__file__).resolve().parents[3]
FROZEN = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle")
ACCEPTED_CODE = Path("/private/tmp/hymem-repaired-four-root-dtBv2p/bundle/code")


def test_real_broker_bridge_allows_300_without_provider(tmp_path, monkeypatch):
    def deny(*_args, **_kwargs):
        pytest.fail("network forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    state = fixture_vm_state(tmp_path)
    seen = []

    def invented_response(_credentials, _system, _user, _schema, *, timeout):
        seen.append(timeout)
        return bridge.transport.Completed(
            text='{"invented":true}', input_tokens=1, output_tokens=1,
            total_tokens=2, cached_input_tokens=0, reasoning_output_tokens=0)

    with bridge.owner.CredentialBroker(state, Path(sys.executable)) as broker:
        budget = bridge.SharedBudget(
            bridge.warm.BudgetLimits(8012, 48_160_000, 25_200))
        client = bridge.SIWCLMEClient(
            broker, budget, "invented-q",
            bridge.warm.BudgetLimits(2000, 12_000_000, 23_400),
            response_call=invented_response)
        assert client.complete(bridge.LLMRequest("invented system", "invented user")) == (
            '{"invented":true}')
        assert len(seen) == 1 and 299 < seen[0] <= 300
        summary = client.diagnostic_summary()
        assert summary["admitted_turns"] == summary["internal_http_attempts"] == 1
        assert summary["successes"] == 1 and summary["known_tokens"] == 2
        assert budget.snapshot()["reserved"] == budget.snapshot()["in_flight"] == 0
        with pytest.raises(bridge.owner.OwnerError, match="deadline_exceeded"):
            broker.acquire(caller_deadline=time.monotonic() + 300.01)


def test_exact_source_chain_and_old_owner_rejection(tmp_path):
    if not FROZEN.is_dir() or not ACCEPTED_CODE.is_dir():
        pytest.skip("frozen local source unavailable")
    root = tmp_path.resolve() / ".hymem-siwc-lme-diagnostic-invented01"
    previous_bundle.assemble(
        repo=REPO, accepted_code=ACCEPTED_CODE,
        candidate=FROZEN / "candidate", map_path=FROZEN / "source-map.json",
        output=root)
    code = root / "code"
    old = (
        "benchmarks/chatgpt_plan_lme_v6.py",
        "tools/diagnostics/siwc_lme_diagnostic_v9.py",
        "tools/diagnostics/lme_chatgpt_plan_owner_v1.py",
        "tools/diagnostics/lme_chatgpt_plan_refresh_v1.py",
    )
    old_owner_bytes = (code / old[2]).read_bytes()
    for relative in old:
        (code / relative).unlink()
    new = (
        "benchmarks/chatgpt_plan_lme_v7.py",
        "tools/diagnostics/siwc_lme_diagnostic_v10.py",
        "tools/diagnostics/lme_chatgpt_plan_owner_v2.py",
        "tools/diagnostics/lme_chatgpt_plan_refresh_v2.py",
    )
    for relative in new:
        (code / relative).write_bytes((REPO / relative).read_bytes())
    assert len(list(code.rglob("*.py"))) == 26
    script = r"""
import importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1])
source=root/'code/tools/diagnostics/siwc_lme_diagnostic_v10.py'
spec=importlib.util.spec_from_file_location('isolated_runner10',source)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
assert pathlib.Path(loaded['siwc'].__file__)==root/'code/benchmarks/chatgpt_plan_lme_v7.py'
assert pathlib.Path(loaded['siwc'].owner.__file__)==root/'code/tools/diagnostics/lme_chatgpt_plan_owner_v2.py'
assert pathlib.Path(loaded['siwc'].owner._pinned_refresh().__file__)==root/'code/tools/diagnostics/lme_chatgpt_plan_refresh_v2.py'
assert loaded['siwc'].SIWCLMEClient.__init__.__kwdefaults__['response_call'] is loaded['siwc'].transport.complete
print('source_only_ok')
"""
    command = [sys.executable, "-I", "-B", "-c", script, str(root)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "source_only_ok"
    (code / new[2]).write_bytes(old_owner_bytes)
    rejected = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert rejected.returncode != 0
    assert "transport_or_helper_drift" in rejected.stderr


def test_receipt_reader_and_wall_policy_pins():
    assert bridge.owner.REFRESH_SHA256 == hashlib.sha256(
        (REPO / "tools/diagnostics/lme_chatgpt_plan_refresh_v2.py").read_bytes()
    ).hexdigest()
    assert runner.SIWC_PINS["benchmarks/chatgpt_plan_lme_v7.py"] == hashlib.sha256(
        (REPO / "benchmarks/chatgpt_plan_lme_v7.py").read_bytes()).hexdigest()
    assert reader.RUNNER_SHA256 == hashlib.sha256(
        (REPO / reader.RUNNER_RELATIVE).read_bytes()).hexdigest()
    assert reader.LAUNCHER_SHA256 == hashlib.sha256(
        (REPO / "tools/diagnostics" / reader.LAUNCHER_NAME).read_bytes()).hexdigest()
    assert runner.MAX_LIMITS == {
        "campaign": (8012, 48_160_000, 25_200),
        "question": (2000, 12_000_000, 23_400),
        "canary": (12, 160_000, 600)}
    assert reader.LIMITS == {key: list(value) for key, value in runner.MAX_LIMITS.items()}
    assert runner.SIWC_PINS["tools/diagnostics/lme_chatgpt_plan_owner_v2.py"] == (
        hashlib.sha256((REPO / "tools/diagnostics/lme_chatgpt_plan_owner_v2.py").read_bytes()).hexdigest())
    assert runner.SIWC_PINS["tools/diagnostics/lme_chatgpt_plan_refresh_v2.py"] == (
        bridge.owner.REFRESH_SHA256)
    assert "benchmarks/chatgpt_plan_lme_v6.py" not in runner.SIWC_PINS
    assert "tools/diagnostics/lme_chatgpt_plan_owner_v1.py" not in runner.SIWC_PINS
    assert reader.RUN_SCHEMA == runner.SCHEMA == "siwc-lme-semantic-diagnostic-v10"
