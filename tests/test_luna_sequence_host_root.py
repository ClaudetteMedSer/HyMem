"""Root-owned installed-source and terminal gate checks; no provider or SSH."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import codex_subscription_timeout_v2 as observer
from tools.diagnostics import luna_timeout_probe_v3 as probe

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "tools/diagnostics"


def source(name):
    spec = importlib.util.spec_from_file_location(name, HELPERS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_exact_closure_with_fresh_source_only_host_import(tmp_path):
    installer = source("luna_timeout_install_v3")
    host = source("luna_timeout_host_v3")
    bundle = tmp_path / "bundle"
    prepared = probe.prepare(bundle)
    assert prepared["receipt_sha256"] == installer.PREPARATION_SHA256
    raw = installer.build_payload(bundle, HELPERS,
        hashlib.sha256((HELPERS / "luna_timeout_launch_v3.py").read_bytes()).hexdigest(),
        hashlib.sha256((HELPERS / "luna_timeout_progress_v3.py").read_bytes()).hexdigest())
    namespace = {"__name__": "root_sequence_source_install"}
    exec(compile(installer.render_remote_script(raw), "<reviewed-v3-installer>", "exec"), namespace)
    output = namespace["install"](raw, home=tmp_path, uid=os.getuid(), enforce_host=False)
    root = Path(output["root"])
    assert output["code_files"] == 14 and output["model_calls"] == 0
    assert host.ROOT_PATTERN.fullmatch(root.name)
    expected, _ = namespace["validate"](raw)
    assert len(expected) == 18
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(expected)
    for path in (root, *root.rglob("*")):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    script = r'''
import hashlib, json, os, sys, types
from pathlib import Path
root = Path(sys.argv[1]); path = root / "timeout-host-v3.py"
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
module = types.ModuleType("root_fresh_sequence_host"); module.__file__ = str(path)
exec(compile(raw, str(path), "exec"), module.__dict__)
module.HOST_HOME = root.parent; module.HOST_UID = os.getuid()
probe, (observer, request) = module.verify_sources(root)
assert observer.SharedBudget is observer.warm.concurrent.SharedBudget
assert observer.warm.BILLING_POLICY == probe.POLICY["billing"]
assert observer.warm.__file__ == str(root / "bundle/code/benchmarks/codex_subscription_warm_v9.py")
assert probe.LIMITS == {"turns":16,"known_tokens":160000,"seconds":600,
    "workers":1,"calls_per_worker":16,"invocation_seconds":120}
assert module.receipt_for(root, sys.argv[2], "probe")["workers"] == 1
print(json.dumps({"source_verified":True,"calls":0,"source_files":14}))
'''
    checked = subprocess.run([sys.executable, "-I", "-B", "-c", script,
        str(root), installer.HOST_SHA256], capture_output=True, text=True, timeout=20)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == {"source_verified": True, "calls": 0, "source_files": 14}


def controls():
    spec = importlib.util.spec_from_file_location("root_sequence_controls",
        ROOT / "tests/test_luna_warm_sequence_root.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def envelope(host, candidate):
    counters = {"pids_denials":0,"memory_oom":0,"memory_oom_kill":0,
                "pids_current":10,"memory_current":100,"memory_peak":100}
    def sample(phase, index=None):
        return {"phase":phase,"index":index,
                "gate":{"containment":True,"denials":0,"oom":0},
                "resources":dict(counters)}
    return {"schema":host.SCHEMA,"mode":"probe","status":candidate["status"],
            "failure_code":None,"probe_result":candidate,
            "samples":[sample("initial"),sample("initial"),
                *[sample("before_admission",i) for i in range(candidate["attempted"])],
                sample("terminal")],"independent_recursive_cleanup_verified":None}


@pytest.mark.parametrize("case", ["success","rotation","identity","timeout","cleanup"])
def test_real_probe_validator_host_and_reader_require_coverage_and_cleanup(tmp_path, monkeypatch, case):
    host = source("luna_timeout_host_v3")
    progress = source("luna_timeout_progress_v3")
    kwargs = {"rotation":{"rotate_after":4},"identity":{"replace_after":4},
              "timeout":{"fail_at":15}}.get(case,{})
    candidate, _, _, _ = controls().run(monkeypatch, **kwargs)
    value = envelope(host, candidate)
    assert host.validate_host_result(value, probe=probe, observer=observer)
    runtime = {"policy_verified":True,"unit_stopped":True,"group_matched":True,
               "recursive_cleanup_verified":case != "cleanup","runtime_exit":"success"}
    monkeypatch.setattr(progress,"_host",lambda _:host)
    monkeypatch.setattr(progress,"_receipt",lambda *_:{"unit":"invented-unit.service"})
    monkeypatch.setattr(progress,"_attempt",lambda *_:True)
    monkeypatch.setattr(progress,"_execution",lambda *_:True)
    monkeypatch.setattr(host,"verify_sources",lambda _:(probe,(observer,object())))
    monkeypatch.setattr(host,"terminal_runtime",lambda _:runtime)
    monkeypatch.setattr(progress,"_read",lambda *_:deepcopy(value))
    (tmp_path / "probe-result.json").touch()
    result = progress.inspect(tmp_path,"0"*64,"probe")
    assert result["completed_and_clean"] is (case == "success")
    assert result["probe"]["lifecycle"] == candidate["lifecycle"]
    assert result["probe"]["counts"]["turns"] == candidate["turns"]
    assert result["probe"]["known_tokens"] == candidate["known_tokens"]
    assert result["probe"]["usage_complete"] is (case != "timeout")
    assert result["historical_timeout_cause_proved"] is False
    assert result["lme_readiness_proved"] is False
    assert "PRIVATE-SYNTHETIC" not in json.dumps(result)


def test_preserved_consumed_v2_source_hashes():
    pins = {
        "luna_timeout_probe_v2.py":"99fb78d584e7fe8ccd1fa7f7eefb6969a29f4604b23228ecc0054b95cc6b1a6c",
        "luna_timeout_host_v2.py":"2e088269bb0e675589f9e71343e1812a6df4671fdcba1bc9ff1c733e65e20158",
        "luna_timeout_launch_v2.py":"c11a60e2702f422e770762d118464b19a06c03812619bd319657537194e5422c",
        "luna_timeout_progress_v2.py":"497f6422363fd16af4ffd44ba89aac2095ac35605c39ed227ffe1de984a245f8",
        "luna_timeout_install_v2.py":"67fbdac5ef414e4eb9f291db1531f820a41bb714424698077a6f6f005f6c10df",
    }
    for name,pin in pins.items():
        assert hashlib.sha256((HELPERS/name).read_bytes()).hexdigest() == pin
