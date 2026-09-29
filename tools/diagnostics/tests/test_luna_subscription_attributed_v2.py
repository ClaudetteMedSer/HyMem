"""Offline source, attribution and one-shot controls for the attributed pilot."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
DIAG = ROOT / "tools/diagnostics"
spec = importlib.util.spec_from_file_location("attributed_launch_test", DIAG / "luna_subscription_profiled_v2_launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def test_frozen_flat_bundle_loads_attributed_budget(tmp_path):
    frozen = Path("/private/tmp/hymem-r9-summary-20260927.XzIhDt")
    if not (frozen / "candidate").is_dir():
        pytest.skip("frozen candidate unavailable")
    for name in launch.PINS:
        source = (frozen / "headless-source-map.json" if name == "headless-source-map.json"
                  else (ROOT / "benchmarks" / name if name.startswith("codex_") or
                        name == "luna_stage_accounting.py" else DIAG / name))
        shutil.copyfile(source, tmp_path / name)
        assert launch.sha(tmp_path / name) == launch.PINS[name]
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    program = r'''
import hashlib, importlib.util, json, pathlib, sys, types
bundle, frozen = map(pathlib.Path, sys.argv[1:3])
spec = importlib.util.spec_from_file_location("attributed_profiled", bundle / "luna_subscription_lme_profiled_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
r = module.warm_runner
dataset = bundle / "dataset.json"
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r.old.DATASET_SHA256 = digest(dataset)
files, concurrent, _, _, _, _, _, warm = r.load_verified(
    candidate=frozen / "candidate", inventory_stamp=bundle / "headless-source-map.json",
    inventory_sha256=digest(bundle / "headless-source-map.json"),
    dataset=dataset, dataset_sha256=digest(dataset), binary=pathlib.Path("/usr/bin/true"),
    base_path=bundle / "codex_subscription.py",
    concurrent_path=bundle / "codex_subscription_concurrent_v2.py",
    warm_path=bundle / "codex_subscription_warm_v2.py")
budget = concurrent.SharedBudget(concurrent.BudgetLimits(8, 1000, 100))
assert files == 508
assert type(budget) is warm.SharedBudget
budget.record_first_failure("timeout", {"phase":"run", "rpc":None,
    "process_age_seconds":3.2, "process_index":1, "request_index":2,
    "retired_count":0, "queue_count":0, "turn_admitted":True,
    "known_usage":False, "known_tokens":5, "usage_complete":False})
assert budget.snapshot()["first_failure"]["code"] == "timeout"
assert budget.snapshot()["stop_code"] == "timeout"
r.old.experimental_canary = lambda *args, **kwargs: {"passed": False}
class Client:
    def __init__(self, budget):
        self.budget = budget
        budget.record_first_failure("timeout", {"phase":"run", "rpc":None,
            "process_age_seconds":3.2, "process_index":1, "request_index":2,
            "retired_count":0, "queue_count":0, "turn_admitted":True,
            "known_usage":False, "known_tokens":5, "usage_complete":False})
    def close(self): pass
limits = concurrent.BudgetLimits(8, 1000, 100)
result = module._accepted_run_campaign(concurrent=concurrent, request_type=object(), canary=object(),
    chunk=object(), lme=object(), protocol=object(), binary="/usr/bin/true",
    questions=[{"index":0}], output=bundle, campaign_limits=concurrent.BudgetLimits(24,3000,300),
    question_limits=concurrent.BudgetLimits(8,1000,200), canary_limits=limits,
    indexing_timeout_s=50, workers=1, client_factory=lambda key, limits, budget: Client(budget))
assert result["budget"]["first_failure"]["code"] == "timeout"
assert result["budget"]["stop_code"] == "timeout"
print(json.dumps({"files":files, "schema":module.SCHEMA}))
'''
    done = subprocess.run([sys.executable, "-I", "-c", program, str(tmp_path), str(frozen)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["schema"] == "luna-subscription-lme-profiled-v2"


def test_safe_first_failure_drops_unrecognized_text():
    spec = importlib.util.spec_from_file_location("attributed_runner_test", DIAG / "luna_subscription_lme_warm_v2.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    class Warm:
        _FIXED_CODES = frozenset({"timeout"})
        _OWN_BUDGET_CODES = frozenset()
        _RPC_METHODS = frozenset({"turn/start"})
        _EVENT_METHODS = frozenset()
    value = {"code": "timeout", "phase": "run", "rpc": "turn/start",
             "process_age_seconds": 1.5, "process_index": 1, "request_index": 2,
             "retired_count": 0, "queue_count": 0, "known_tokens": 42,
             "turn_admitted": True, "known_usage": False, "usage_complete": False,
             "secret": "private prompt"}
    safe = runner.safe_first_failure(value, Warm)
    assert safe["code"] == "timeout" and "secret" not in safe
    assert runner.safe_first_failure({**value, "code": "private prompt"}, Warm) is None
    assert runner.safe_first_failure({**value, "rpc": "private method"}, Warm) is None


def test_launcher_contract_and_one_shot(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-lme-attributed-abcd_1234"
    root.mkdir(mode=0o700)
    unit = launch.unit_for(root)
    assert unit == "hymem-luna-lme-attributed-abcd_1234.service"
    receipt = {"unit": unit}
    cmd = launch.command(root, receipt)
    assert str(root / "luna_subscription_lme_profiled_v2.py") in cmd
    assert str(root / "codex_subscription_warm_v2.py") in cmd
    for value in ("RuntimeMaxSec=14530s", "TimeoutStopSec=10s", "MemoryMax=4294967296",
                  "CPUQuota=200%", "TasksMax=128", "OOMPolicy=kill"):
        assert "--property=" + value in cmd
    assert launch.LIMITS["questions"] == launch.LIMITS["workers"] == 4
    assert launch.LIMITS["campaign-turns"] == 8012
    monkeypatch.setattr(launch, "host_admission", lambda: None)
    monkeypatch.setattr(launch, "verify_sources", lambda _: None)
    monkeypatch.setattr(launch, "receipt_for", lambda _: receipt)
    monkeypatch.setattr(launch, "sha", lambda _: "a" * 64)
    monkeypatch.setattr(launch, "HOST_ROOT", tmp_path)
    monkeypatch.setattr(launch, "HOST_UID", root.stat().st_uid)
    (root / "launch-receipt.json").write_text(json.dumps(receipt))
    calls = []
    def dispatch(*args, **kwargs):
        calls.append((root / "launch-attempt.json").exists())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launch.subprocess, "run", dispatch)
    args = ["--launch-root", str(root), "--receipt-sha256", "a" * 64]
    assert launch.main(args) == 0
    assert launch.main(args) == 1
    assert calls == [True]


def test_staged_root_is_required_for_prepare():
    assert launch.main(["--prepare"]) == 1


def test_host_admission_allows_clean_exited_prior_unit(monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "linux")
    monkeypatch.setattr(launch.os, "getuid", lambda: launch.HOST_UID)
    monkeypatch.setattr(launch.Path, "read_text", lambda _: "MemAvailable: 9000000 kB")
    monkeypatch.setattr(launch.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * 1024**3))
    calls = []
    def systemctl(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=("hymem-luna-lme-profiled-old_1234.service loaded active exited\n"
                                      if "list-units" in command else "MainPID=0\nControlGroup=\n"))
    monkeypatch.setattr(launch.subprocess, "run", systemctl)
    launch.host_admission()
    assert len(calls) == 2


def test_main_exception_recovers_safe_first_failure(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("attributed_main_test", DIAG / "luna_subscription_lme_warm_v2.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    class Warm:
        _FIXED_CODES = frozenset({"timeout"})
        _OWN_BUDGET_CODES = frozenset()
        _RPC_METHODS = frozenset({"turn/start"})
        _EVENT_METHODS = frozenset()
    concurrent = SimpleNamespace(BudgetLimits=lambda *args: SimpleNamespace())
    monkeypatch.setattr(runner, "load_verified", lambda **kwargs:
                        (508, concurrent, object(), object(), object(), object(), object(), Warm))
    monkeypatch.setattr(runner, "SelectedQuestions", lambda *args: [{"index": 0}])
    monkeypatch.setattr(runner.old, "digest", lambda _: "a" * 64)
    failure = {"code": "timeout", "phase": "run", "rpc": "turn/start",
               "process_age_seconds": 1.0, "process_index": 1, "request_index": 2,
               "retired_count": 0, "queue_count": 0, "known_tokens": 5,
               "turn_admitted": True, "known_usage": False, "usage_complete": False,
               "secret": "private prompt"}
    def fail(**kwargs):
        runner.atomic_private(kwargs["output"] / "private-progress.json", {
            "budget": {"first_failure": failure, "stop_code": "timeout"}})
        raise RuntimeError("private prompt")
    monkeypatch.setattr(runner, "run_campaign", fail)
    output = tmp_path / "run"
    options = {
        "binary": "/usr/bin/true", "base-transport": "/usr/bin/true",
        "concurrent-transport": "/usr/bin/true", "warm-transport": "/usr/bin/true",
        "candidate": str(tmp_path), "inventory-stamp": "/usr/bin/true",
        "inventory-sha256": "a" * 64, "dataset": "/usr/bin/true",
        "dataset-sha256": "a" * 64, "output-dir": str(output),
        "campaign-turns": 8012, "campaign-known-tokens": 48160000,
        "campaign-seconds": 14400, "question-turns": 2000,
        "question-known-tokens": 12000000, "question-seconds": 12600,
        "canary-turns": 12, "canary-known-tokens": 160000,
        "canary-seconds": 600, "indexing-seconds": 10800,
        "questions": 4, "workers": 4,
    }
    args = [part for key, value in options.items() for part in ("--" + key, str(value))]
    assert runner.main(args) == 1
    terminal = json.loads(capsys.readouterr().out)
    assert terminal["first_failure"]["code"] == "timeout"
    assert "secret" not in terminal["first_failure"]
    assert terminal["ok"] is False
