"""Offline integration and fail-closed metadata gates for observed Luna."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


DIAG = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIAG / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("luna_observed_lme")
launcher = load("luna_observed_lme_launch")
reader = load("luna_observed_lme_progress")


def failure(**extra):
    value = {"code": "fixed_other", "phase": "run", "rpc": None,
        "process_index": 1, "request_index": 1, "retired_count": 0,
        "queue_count": 0, "known_tokens": 100,
        "process_age_seconds": 2.0, "turn_admitted": True,
        "known_usage": False, "usage_complete": False}
    value.update(extra)
    return value


def test_nested_diagnostics_strict_and_finite():
    valid = failure(rpc="turn/events", event_family="error",
        failure_family="unexpected_notification",
        app_server_error={"identity": "matched", "will_retry": True,
            "error_class": "serverOverloaded"})
    assert reader.first_failure_summary(valid) == valid
    for change in (
        {"process_index": True}, {"rpc": "secret"}, {"event_family": "secret"},
        {"app_server_error": {"identity": []}},
        {"app_server_error": {"identity": "matched", "will_retry": "yes",
                              "error_class": "serverOverloaded"}},
        {"app_server_error": {"identity": "matched", "will_retry": True,
                              "error_class": "secret"}},
        {"rpc_error": {"category": "other_numeric", "code": "secret"}},
        {"arbitrary_raw": "secret"},
    ):
        assert reader.first_failure_summary({**valid, **change}) is None
    missing = dict(valid)
    del missing["known_usage"]
    assert reader.first_failure_summary(missing) is None


def test_receipt_identity_and_policy_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-lme-observed-abcdefgh"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(reader, "ROOT_PARENT", tmp_path)
    unit = "hymem-luna-lme-observed-abcdefgh.service"
    cgroup = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
    assert reader.root_identity_valid(root, unit, cgroup)
    ground = load("luna_grounding_lme_launch")
    proof = {"grounded_inventory_sha256": "a" * 64,
             "grounded_map_sha256": reader.ground.GROUNDED_MAP_SHA256}
    pins = dict(reader.PINS, **{reader.ground.DERIVED_STAMP: "a" * 64})
    monkeypatch.setattr(launcher, "HOST_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "sha", lambda path: "b" * 64 if path == ground.BINARY else hashlib.sha256(path.read_bytes()).hexdigest())
    receipt = launcher.receipt_for(root, ground, proof, pins)
    assert reader.receipt_valid(receipt, root, unit, cgroup)
    for changed in (
        dict(receipt, runner_sha256="0" * 64),
        dict(receipt, effective_warm_transport_sha256="0" * 64),
        dict(receipt, tasks_max=128),
        dict(receipt, dataset={"raw": "bad"}),
        dict(receipt, source_sha256={**pins, "codex_subscription_warm_v3.py": "0" * 64}),
    ):
        assert not reader.receipt_valid(changed, root, unit, cgroup)


def test_terminal_keeps_original_health_gates(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("observed_fixture",
        DIAG / "tests/test_luna_subscription_profiled_v2_progress.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    root, receipt, safe, result, source = fixture._fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(reader.base, "bounded_json", lambda path: source[str(path)])
    for obj in (safe, result):
        obj["schema"] = reader.SCHEMA
        obj["effective_warm_transport_sha256"] = reader.WARM_V3_SHA256
        obj["inherited_warm_transport_sha256"] = reader.WARM_V2_SHA256
    safe["runner_sha256"] = reader.RUNNER_SHA256
    safe["candidate_source_map_sha256"] = reader.ground.GROUNDED_MAP_SHA256
    assert reader.verify_terminal(root, receipt, safe, result)["validated"] is True
    safe["usage_complete"] = False
    assert reader.verify_terminal(root, receipt, safe, result)["validated"] is False
    safe["usage_complete"] = True
    result["stage_accounting"]["q-0000"]["reader"]["known_tokens"] += 1
    assert reader.verify_terminal(root, receipt, safe, result)["validated"] is False


def test_stdin_reader_fresh_root_and_private_owner(tmp_path):
    root = tmp_path / ".hymem-luna-lme-observed-abcdefgh"
    root.mkdir(mode=0o700)
    shutil.copyfile(DIAG / "luna_grounding_lme_progress.py", root / "luna_grounding_lme_progress.py")
    shutil.copyfile(DIAG / "luna_subscription_capacity_progress.py", root / "luna_subscription_capacity_progress.py")
    text = (DIAG / "luna_observed_lme_progress.py").read_text().replace(
        'ROOT_PARENT = Path("/home/atta")', f"ROOT_PARENT = Path({str(tmp_path)!r})").replace(
        "OBSERVER_UID = 1000", f"OBSERVER_UID = {os.getuid()}")
    unit = "hymem-luna-lme-observed-abcdefgh.service"
    args = [sys.executable, "-I", "-B", "-", "--root", str(root), "--unit", unit,
        "--expected-cgroup", "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "--receipt-sha256", "0" * 64]
    call = subprocess.run(args, input=text, text=True, capture_output=True, timeout=20)
    assert call.returncode == 1, call.stderr
    report = json.loads(call.stdout)
    assert report["schema"] == "luna-observed-lme-progress-v1"
    assert report["completed_and_clean"] is False
    assert report["raw_text_exported"] is False


def test_actual_flat_loader_selects_v3_client_and_budget(tmp_path):
    frozen = Path("/private/tmp/hymem-r9-summary-20260927.XzIhDt")
    derived = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi")
    if not (frozen / "candidate").is_dir() or not (derived / "candidate").is_dir():
        pytest.skip("local frozen grounding fixture unavailable")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("luna_observed_lme.py", "luna_grounding_lme.py",
                 "luna_grounding_candidate.py", "luna_subscription_lme_profiled_v2.py",
                 "luna_subscription_lme_warm_v2.py",
                 "luna_subscription_pilot.py"):
        shutil.copyfile(DIAG / name, bundle / name)
    shutil.copyfile(DIAG.parents[1] / "benchmarks/luna_stage_accounting.py",
                    bundle / "luna_stage_accounting.py")
    for name in ("codex_subscription.py", "codex_subscription_concurrent_v2.py",
                 "codex_subscription_warm_v2.py", "codex_subscription_warm_v3.py"):
        shutil.copyfile(DIAG.parents[1] / "benchmarks" / name, bundle / name)
    shutil.copyfile(frozen / "headless-source-map.json", bundle / "headless-source-map.json")
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    script = r'''
import hashlib, importlib.util, json, pathlib, sys, time
from collections import deque
from types import SimpleNamespace
b, original, grounded, stamp, dataset = map(pathlib.Path, sys.argv[1:])
s=importlib.util.spec_from_file_location("observed", b/"luna_observed_lme.py")
r=importlib.util.module_from_spec(s); s.loader.exec_module(r)
g=r.pinned("luna_grounding_lme.py",r.GROUNDED_RUNNER_SHA256,"grounded")
g.ORIGINAL_CANDIDATE=original
builder=g._pinned_module("luna_grounding_candidate.py",g.BUILDER_SHA256,"builder")
p=g._pinned_module("luna_subscription_lme_profiled_v2.py",g.PROFILED_SHA256,"profile")
g.bind_profile(p,builder,grounded,stamp,b/"headless-source-map.json",hashlib.sha256(stamp.read_bytes()).hexdigest())
r.bind(g,p)
p.warm_runner.old.DATASET_SHA256=hashlib.sha256(dataset.read_bytes()).hexdigest()
loaded=p.warm_runner.load_verified(candidate=grounded,inventory_stamp=stamp,
 inventory_sha256=hashlib.sha256(stamp.read_bytes()).hexdigest(),dataset=dataset,
 dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),binary=pathlib.Path("/usr/bin/true"),
 base_path=b/"codex_subscription.py",concurrent_path=b/"codex_subscription_concurrent_v2.py",
 warm_path=b/"codex_subscription_warm_v2.py")
v3=loaded[-1]
class FakeSession:
 def __init__(self,*args,**kwargs):
  self.created_at=time.monotonic(); self.stage="initialize"; self.pending=deque()
  self.starting_events=[]; self.retired_threads=set(); self.last_event=None; self.rpc_error=None
 def set_deadline(self,deadline): self.deadline=deadline
 def unsubscribe(self,thread_id): self.stage="thread/unsubscribe"
 def close(self): pass
def admit(session,**kwargs):
 session.stage="thread/start"
 return {"auth":"chatgpt","model":v3.base.MODEL,"config_isolation_admitted":True,
  "inference_enabled":False,"quota_windows":[{"remaining_percent":75}],"_thread_id":"thread-1"}
def fail(session,*args):
 session.stage="turn/events"
 session.last_event={"event_family":"error","app_server_error":{
  "identity":"matched","will_retry":False,"error_class":"internalServerError"}}
 v3.base._fail("unexpected_notification:error")
v3.base.inspect_preflight=admit; v3.base._run_turn=fail
limits=v3.BudgetLimits(100,10000,1000); budget=v3.SharedBudget(limits)
client=v3.WarmSubscriptionClient("unused",budget,"q",limits,session_factory=FakeSession)
try:
 client.complete(SimpleNamespace(system="s",user="u",temperature=0,max_tokens=32,response_format="json"))
except v3.ConcurrentStop: pass
else: raise AssertionError("failure was accepted")
raw=budget.snapshot()["first_failure"]
safe=p.warm_runner.safe_first_failure(raw,v3)
print(json.dumps({"files":loaded[0],"client":loaded[-1].WarmSubscriptionClient.__module__,
 "same_concurrent":loaded[1] is loaded[-1].concurrent,
 "same_stop":loaded[1].ConcurrentStop is loaded[-1].ConcurrentStop,
 "runner":p.warm_runner.RUNNER_SHA256,"schema":p.warm_runner.SCHEMA,
 "safe":safe,"usage_complete":budget.snapshot()["usage_complete"]}))
'''
    call = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(bundle),
        str(frozen / "candidate"), str(derived / "candidate"),
        str(derived / "headless-grounding-source-map.json"), str(dataset)],
        text=True, capture_output=True, timeout=45)
    assert call.returncode == 0, call.stderr
    report = json.loads(call.stdout)
    assert {key: report[key] for key in ("files", "client", "same_concurrent", "same_stop", "runner", "schema")} == {
        "files": 508, "client": "pinned_observed_warm_v3",
        "same_concurrent": True, "same_stop": True,
        "runner": launcher.RUNNER_SHA256, "schema": runner.SCHEMA}
    assert report["usage_complete"] is False
    assert report["safe"]["code"] == "fixed_other"
    assert report["safe"]["rpc"] == "turn/events"
    assert report["safe"]["app_server_error"]["error_class"] == "internalServerError"
    assert reader.first_failure_summary(report["safe"]) == report["safe"]
    assert "secret" not in json.dumps(report)


def test_one_shot_marker_before_ambiguous_dispatch(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-lme-observed-abcdefgh"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(launcher, "HOST_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "HOST_UID", os.getuid())
    (root / "empty").mkdir(mode=0o700)
    (root / "tmp").mkdir(mode=0o700)
    monkeypatch.setattr(launcher, "grounding", lambda path: None)
    class Host:
        def host_admission(self): pass
        def write_once(self, path, value):
            with path.open("x") as stream:
                json.dump(value, stream)
    base = SimpleNamespace(base_launcher=lambda path: Host())
    monkeypatch.setattr(launcher, "grounding", lambda path: base)
    monkeypatch.setattr(launcher, "verify_sources", lambda path: (base, {}, {}))
    monkeypatch.setattr(launcher, "receipt_for", lambda *args: {"unit": "observed"})
    monkeypatch.setattr(launcher, "command", lambda *args: ["true"])
    calls = []
    def ambiguous(*args, **kwargs):
        calls.append(1)
        raise subprocess.TimeoutExpired("true", 20)
    monkeypatch.setattr(launcher.subprocess, "run", ambiguous)
    receipt_path = root / "launch-receipt.json"
    receipt_path.write_text(json.dumps({"unit": "observed"}))
    args = ["--launch-root", str(root), "--receipt-sha256", launcher.sha(receipt_path)]
    assert launcher.main(args) == 1
    assert (root / "launch-attempt.json").is_file()
    assert launcher.main(args) == 1
    assert calls == [1]


def test_private_workspaces_reject_symlink_contents_and_public_mode(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-lme-observed-abcdefgh"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(launcher, "HOST_UID", os.getuid())
    (root / "empty").mkdir(mode=0o700)
    (root / "tmp").mkdir(mode=0o700)
    launcher.private_workspaces(root)
    (root / "empty" / "unexpected").write_text("x")
    with pytest.raises(ValueError, match="workspace_invalid"):
        launcher.private_workspaces(root)
    (root / "empty" / "unexpected").unlink()
    (root / "empty").chmod(0o755)
    with pytest.raises(ValueError, match="workspace_invalid"):
        launcher.private_workspaces(root)
    (root / "empty").chmod(0o700)
    (root / "tmp").rmdir()
    (root / "tmp").symlink_to(root / "empty")
    with pytest.raises(ValueError, match="workspace_invalid"):
        launcher.private_workspaces(root)


def test_launcher_verify_sources_with_real_derived_candidate(tmp_path):
    frozen = Path("/private/tmp/hymem-r9-summary-20260927.XzIhDt")
    derived = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi")
    if not (frozen / "candidate").is_dir() or not (derived / "candidate").is_dir():
        pytest.skip("local frozen grounding fixture unavailable")
    root = tmp_path / ".hymem-luna-lme-observed-abcdefgh"
    root.mkdir(mode=0o700)
    shutil.copytree(derived / "candidate", root / "candidate")
    shutil.copyfile(derived / "headless-grounding-source-map.json",
                    root / "headless-grounding-source-map.json")
    shutil.copyfile(frozen / "headless-source-map.json", root / "headless-source-map.json")
    source = load("luna_grounding_lme_launch")
    pins = launcher.expected_pins(source, root)
    for name in pins:
        if name == "headless-source-map.json":
            continue
        location = DIAG / name
        if not location.is_file():
            location = DIAG.parents[1] / "benchmarks" / name
        shutil.copyfile(location, root / name)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    script = r'''
import hashlib, importlib.util, json, pathlib, sys
root, original, dataset = map(pathlib.Path,sys.argv[1:])
s=importlib.util.spec_from_file_location("observed_launch",root/"luna_observed_lme_launch.py")
l=importlib.util.module_from_spec(s); s.loader.exec_module(l)
base=l.grounding(root); base.ORIGINAL_CANDIDATE=original; base.DATASET=dataset
base.DATASET_SHA256=hashlib.sha256(dataset.read_bytes()).hexdigest()
base.BINARY=pathlib.Path("/usr/bin/true")
l.grounding=lambda path:base
original_pinned=l.pinned
def pinned(path,digest,identity):
 module=original_pinned(path,digest,identity)
 if path.name=="luna_grounding_lme.py": module.ORIGINAL_CANDIDATE=original
 if path.name=="luna_subscription_lme_profiled_v2.py":
  module.warm_runner.old.DATASET_SHA256=base.DATASET_SHA256
 return module
l.pinned=pinned
_,proof,pins=l.verify_sources(root)
print(json.dumps({"files":proof["source_files"],"map":proof["grounded_map_sha256"],
 "v3":pins["codex_subscription_warm_v3.py"],"runner":pins[l.RUNNER_NAME]}))
'''
    call = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(root),
        str(frozen / "candidate"), str(dataset)], text=True, capture_output=True, timeout=60)
    assert call.returncode == 0, call.stderr
    result = json.loads(call.stdout)
    assert result == {"files": 508, "map": launcher.GROUNDED_MAP_SHA256,
        "v3": launcher.OBSERVED_V3_SHA256, "runner": launcher.RUNNER_SHA256}
