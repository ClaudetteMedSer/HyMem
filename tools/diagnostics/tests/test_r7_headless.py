"""Offline controls for the fresh R7 sample-eight headless harness."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


BASE = Path(__file__).parents[1] / "lme_r7_headless"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_recipe_and_isolated_detached_plan():
    run = load("r7_headless_run", BASE / "bundle/q1_stock_run.py")
    host = load("r7_headless_host", BASE / "bundle/q1_stock_host.py")
    prepare = load("r7_headless_prepare", BASE / "prepare.py")
    assert run.SAMPLE == 8 and run.SEED == 0 and run.TIMEOUT == 32400
    assert run.SOURCE_MANIFEST_SHA == prepare.R7_PIN
    assert run.verify_selection(run.selector_from_source(
        Path("/private/tmp/hymem-r7-integration-20260925.vDa9yP/frozen-r7-final/benchmarks/longmemeval_adapter.py"))) == run.SOURCE_INDICES
    args = run.stock_arguments()
    for flag, wanted in (("--sample", "8"), ("--seed", "0"), ("--workers", "1"),
                         ("--hymem-model", "deepseek-flash"),
                         ("--indexing-max-cycles", "100"),
                         ("--indexing-timeout-s", "3600")):
        assert args[args.index(flag) + 1] == wanted
    assert "--resume-from" not in args and "--retry-failures" not in args
    for live in (False, True):
        plan = host.command(prepare.ROOT, prepare.SOURCE, "0" * 64, live=live)
        assert plan["network"] == ("bridge" if live else "none")
        assert plan["starts_work"] is False and plan["command"][1] == "create"
        assert [target for target, (_, writable) in plan["mounts"].items() if writable] == ["/results"]
        assert ("/run/deepseek.env" in plan["mounts"]) is live
        assert "/var/run/docker.sock" not in plan["mounts"]
        assert plan["mounts"]["/candidate"] == (prepare.SOURCE, False)
        assert "--read-only" in plan["command"] and "--pull" in plan["command"]
    assert hashlib.sha256((BASE / "bundle/supervised_invocation.py").read_bytes()).hexdigest() == (
        "9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc")
    assert hashlib.sha256((BASE / "bundle/transport_common.py").read_bytes()).hexdigest() == (
        "1c98c73cfb1b607113c726616058862fd6bb16a57901a57c8962ee4a0a2807a0")


@pytest.mark.parametrize("status,returncode,safe,expected", [
    ("completed", 0, True, 0), ("timeout", -15, False, 1),
])
def test_supervision_one_invocation_with_terminal_receipt(
    monkeypatch, tmp_path, status, returncode, safe, expected,
):
    run = load("r7_headless_supervision", BASE / "bundle/q1_stock_run.py")
    monkeypatch.setattr(run, "OUTPUT", tmp_path)
    monkeypatch.setattr(run, "SOURCE", tmp_path)
    monkeypatch.setattr(run, "DATA", tmp_path / "data.json")
    monkeypatch.setattr(run.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(run, "sha", lambda _path: run.DATASET_SHA)
    monkeypatch.setattr(run, "validate_package", lambda _pin: {"source_sha256": {}})
    monkeypatch.setattr(run, "verify_source", lambda _manifest: None)
    monkeypatch.setattr(run, "preflight", lambda _manifest: {"status": "preflight_only", "api_calls": 0})
    monkeypatch.setattr(run.signal, "signal", lambda *_args: None)
    calls = []
    outcome_type = dataclasses.make_dataclass("Outcome", ["status", "returncode", "safe_to_continue"])

    def fake_supervise(command, **kwargs):
        calls.append((command, kwargs))
        return outcome_type(status, returncode, safe)

    monkeypatch.setattr(run, "load_helper", lambda name: SimpleNamespace(
        supervise_invocation=fake_supervise) if name == "supervised_invocation" else None)
    assert run.supervise("0" * 64) == expected
    assert len(calls) == 1
    assert calls[0][1]["timeout_seconds"] == 32400
    assert calls[0][1]["output_limit_bytes"] == 512 * 1024 * 1024
    assert calls[0][1]["cleanup_seconds"] == 10
    assert calls[0][1]["stdin_bytes"] == run.canonical({"execute_stock_sample8": "0" * 64})
    report = json.loads((tmp_path / "supervisor-summary.json").read_text())
    assert report["outcome"]["status"] == status
    assert report["benchmark_pass_verified"] is False and report["score_verified"] is False
    assert report["global_paid_call_cap"] is None


def test_install_timeout_unknown_and_no_retry(monkeypatch, tmp_path, capsys):
    install = load("r7_headless_install", BASE / "install.py")
    archive = tmp_path / "package.tgz"
    archive.write_bytes(b"sealed-test-archive")
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"schema": "lme-r7-sample8-seal-v1",
        "source_manifest_sha256": install.R7_PIN, "manifest_sha256": "0" * 64,
        "archive": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files_sha256": {}}))
    calls = []

    def timeout(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"private-data")

    monkeypatch.setattr(install.subprocess, "run", timeout)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(sys, "argv", ["install.py", "--seal", str(seal),
        "--seal-sha256", hashlib.sha256(seal.read_bytes()).hexdigest(),
        "--receipt", str(receipt)])
    with pytest.raises(SystemExit) as caught:
        install.main()
    assert caught.value.code == 1 and len(calls) == 1
    assert calls[0][0][:2] == ["ssh", "-C"] and calls[0][1]["timeout"] == 300
    assert json.loads(capsys.readouterr().out) == {"status": "r7_sample8_install_timeout",
        "outcome": "unknown", "requires_inspection": True, "retry_attempted": False}
    assert not receipt.exists()
