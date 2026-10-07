"""Independent no-provider checks for the versioned credit diagnostic wiring."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import codex_subscription_timeout_v1 as old_observer
from benchmarks import codex_subscription_timeout_v2 as observer
from hymem.extraction.llm import LLMRequest
from tools.diagnostics import luna_timeout_probe_v1 as old_probe
from tools.diagnostics import luna_timeout_probe_v2 as probe


ROOT = Path(__file__).resolve().parents[1]
RAW = {"rateLimits": {"planType": "pro", "credits": {
    "hasCredits": True, "unlimited": False, "balance": "2.0"},
    "primary": {"usedPercent": 99, "windowDurationMins": 300,
                "resetsAt": 1790784000}}}


def prior_root_controls():
    spec = importlib.util.spec_from_file_location("credit_integration_root_controls",
        ROOT / "tests/test_luna_timeout_probe_root.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("current,runner,expected", [
    (old_observer, old_probe, 0), (observer, probe, 16)])
def test_actual_four_worker_observer_parser_and_ledger_credit_path(
        monkeypatch, current, runner, expected):
    controls = prior_root_controls()
    monkeypatch.setattr(controls, "observed", current)
    factory, clients, processes, peaks = controls.setup_transport(monkeypatch)
    ready = current.base.inspect_preflight

    def credited(session, **kwargs):
        admission = ready(session, **kwargs)
        # Use the actual pinned quota parser, not fabricated admission flags.
        admission["quota_windows"] = current.base.quota_metadata(deepcopy(RAW))
        assert admission["quota_windows"][0]["remaining_percent"] == 1
        return admission

    monkeypatch.setattr(current.base, "inspect_preflight", credited)
    result = runner.run_probe(current, LLMRequest, "unused",
        attest=controls.attested, client_factory=factory)
    assert result["turns"] == result["returned"] == expected
    assert result["known_tokens"] == expected * 19
    assert result["reserved"] == result["in_flight"] == 0
    assert result["usage_complete"] is True
    assert result["client_cleanup_verified"] is True
    assert result["independent_recursive_cleanup_verified"] is None
    assert result["historical_timeout_cause_proved"] is False
    assert result["lme_readiness_proved"] is False
    if expected:
        assert len(clients) == len(processes) == 4 and max(peaks) == 4
        assert all(client.requests_on_process == 4 for client in clients)
        assert result["status"] == "observed_success"
        assert result["budget_stop_code"] is None
    else:
        assert result["status"] == "incomplete_or_failed"
        assert result["budget_stop_code"] == "quota_unverified"
    assert controls.PRIVATE not in json.dumps(result)
    assert "balance" not in json.dumps(result)


def test_exact_versioned_closure_and_unchanged_experiment():
    assert len(probe.SOURCE_PINS) + 1 == 14
    assert probe.FIXTURE_SHA256 == old_probe.FIXTURE_SHA256
    assert probe.USERS == old_probe.USERS and probe.SYSTEM == old_probe.SYSTEM
    assert probe.LIMITS == old_probe.LIMITS
    assert probe.BINARY_PATH == old_probe.BINARY_PATH
    assert probe.BINARY_SHA256 == old_probe.BINARY_SHA256
    expected = dict(old_probe.POLICY)
    expected["billing"] = observer.warm.BILLING_POLICY
    assert probe.POLICY == expected
    assert observer.SharedBudget is observer.warm.SharedBudget
    assert observer.SharedBudget is observer.warm.concurrent.SharedBudget
    assert observer.SharedBudget is not old_observer.SharedBudget
    for relative, pin in probe.SOURCE_PINS.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == pin


def test_source_only_fresh_isolated_loader_preserves_credit_proof(tmp_path):
    bundle = tmp_path / "bundle"
    prepared = probe.prepare(bundle)
    code = r'''
import hashlib, json, pathlib, sys, types
root = pathlib.Path(sys.argv[1])
path = root / "code/tools/diagnostics/luna_timeout_probe_v2.py"
source = path.read_bytes()
assert hashlib.sha256(source).hexdigest() == sys.argv[3]
module = types.ModuleType("root_fresh_credit_probe")
module.__file__ = str(path)
exec(compile(source, str(path), "exec"), module.__dict__)
observer, request = module.load_prepared(root, sys.argv[2])
assert observer.SharedBudget is observer.warm.concurrent.SharedBudget
raw = {"rateLimits": {"planType": "pro", "credits": {
    "hasCredits": True, "unlimited": False, "balance": "2.0"},
    "primary": {"usedPercent": 99, "windowDurationMins": 300,
                "resetsAt": 1790784000}}}
windows = observer.base.quota_metadata(raw)
budget = observer.SharedBudget(observer.BudgetLimits(16, 160000, 600))
budget.register("root", observer.BudgetLimits(4, 160000, 600))
budget.reserve("root")
budget.before_turn("root", {"auth": "chatgpt", "model": "gpt-6-luna",
    "config_isolation_admitted": True, "inference_enabled": False,
    "quota_windows": windows})
budget.settle("root", used=19, turn_started=True)
assert budget.snapshot()["turns"] == 1
assert budget.snapshot()["known_tokens"] == 19
assert budget.snapshot()["usage_complete"] is True
assert str(root / "code/benchmarks/codex_subscription_warm_v9.py") == observer.warm.__file__
assert windows[0]["remaining_percent"] == 1
assert "balance" not in json.dumps(windows)
print(json.dumps({"verified": True, "provider_calls": 0, "code_files": 14}))
'''
    outcome = subprocess.run([sys.executable, "-I", "-B", "-c", code,
        str(bundle), prepared["receipt_sha256"],
        hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()],
        cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert outcome.returncode == 0, outcome.stderr
    assert json.loads(outcome.stdout) == {
        "verified": True, "provider_calls": 0, "code_files": 14}


def test_actual_installer_copy_and_fresh_host_import(tmp_path):
    from tools.diagnostics import luna_timeout_install_v2 as installer
    from tools.diagnostics import luna_timeout_host_v2 as host
    bundle = tmp_path / "bundle"
    prepared = probe.prepare(bundle)
    assert prepared["receipt_sha256"] == installer.PREPARATION_SHA256
    helpers = ROOT / "tools/diagnostics"
    raw = installer.build_payload(bundle, helpers,
        hashlib.sha256((helpers / "luna_timeout_launch_v2.py").read_bytes()).hexdigest(),
        hashlib.sha256((helpers / "luna_timeout_progress_v2.py").read_bytes()).hexdigest())
    namespace = {"__name__": "root_credit_source_install"}
    exec(compile(installer.render_remote_script(raw), "<reviewed-v2-installer>", "exec"), namespace)
    output = namespace["install"](raw, home=tmp_path, uid=os.getuid(), enforce_host=False)
    root = Path(output["root"])
    assert output["code_files"] == 14 and output["model_calls"] == 0
    assert host.ROOT_PATTERN.fullmatch(root.name)
    expected, _ = namespace["validate"](raw)
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(expected)
    for path in (root, *root.rglob("*")):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    source = r'''
import hashlib, json, os, sys, types
from pathlib import Path
root = Path(sys.argv[1]); path = root / "timeout-host-v2.py"
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
module = types.ModuleType("root_fresh_credit_host"); module.__file__ = str(path)
exec(compile(raw, str(path), "exec"), module.__dict__)
module.HOST_HOME = root.parent; module.HOST_UID = os.getuid()
probe, (observer, request) = module.verify_sources(root)
assert observer.SharedBudget is observer.warm.concurrent.SharedBudget
assert observer.warm.BILLING_POLICY == probe.POLICY["billing"]
assert observer.warm.__file__ == str(root / "bundle/code/benchmarks/codex_subscription_warm_v9.py")
print(json.dumps({"source_verified": True, "model": observer.base.MODEL,
    "calls": 0, "request": request.__name__}))
'''
    checked = subprocess.run([sys.executable, "-I", "-B", "-c", source,
        str(root), installer.HOST_SHA256], capture_output=True, text=True, timeout=15)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == {"source_verified": True,
        "model": "gpt-6-luna", "calls": 0, "request": "LLMRequest"}


def test_versioned_host_accepts_underscore_in_tempfile_suffix():
    from tools.diagnostics import luna_timeout_host_v2 as host
    assert host.ROOT_PATTERN.fullmatch(".hymem-luna-timeout-v2-a_bc1234")
    assert not host.ROOT_PATTERN.fullmatch(".hymem-luna-timeout-dpunlpef")
    assert not host.ROOT_PATTERN.fullmatch(".hymem-luna-timeout-v2-../escape")


def test_historical_consumed_sources_are_unmodified():
    pins = {
        "benchmarks/codex_subscription_warm_v8.py": "0a55d44053349eb90511a597dae20f295c19bb53c734a78b5db3c41197343c21",
        "benchmarks/codex_subscription_timeout_v1.py": "9fedd5c2151bf016b03c614dff8a2d8b267cd76e73a3d005d2be80c62c1b0b93",
        "tools/diagnostics/luna_timeout_probe_v1.py": "066f7f60173b9849b18bf06845c55ee8d1ed0bd3a1cd4608e9f8d6868fa8d7d7",
        "tools/diagnostics/luna_timeout_host_v1.py": "f38c77ee73b05e1004dd9863b873665c55f568bd4e50782701aeaa48ce11d93c",
        "tools/diagnostics/luna_timeout_launch_v1.py": "74ed7eb2f16d8bab19bbbab0ed7474a56d327f2c13c5a5448d495a3e6ea0d45a",
        "tools/diagnostics/luna_timeout_progress_v1.py": "fa15e623b33b0cfb386b2e3110e505d8374b2a4041acacf74d49f0afe591c5cf",
        "tools/diagnostics/luna_timeout_install_v1.py": "98ca478b0140ddd1e72180c3d099a35af97ab200256a8ff42174225e1167fc46",
    }
    for relative, pin in pins.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == pin
