"""Offline controls for the bounded real-workload resource observer."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "luna_lme_resource_probe.py"
spec = importlib.util.spec_from_file_location("resource_probe_test", SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_fixed_caps_and_accepted_runner_arguments(tmp_path):
    limits = probe.LIMITS
    assert limits == {
        "campaign-turns": 176, "campaign-known-tokens": 1_500_000,
        "campaign-seconds": 900, "question-turns": 80,
        "question-known-tokens": 600_000, "question-seconds": 800,
        "canary-turns": 12, "canary-known-tokens": 160_000,
        "canary-seconds": 600, "indexing-seconds": 750,
        "questions": 4, "workers": 4, "warm-max-requests": 16,
        "warm-max-age-seconds": 300}
    args = probe.runner_args(probe.BUNDLE, tmp_path)
    assert args[args.index("--output-dir") + 1] == str(tmp_path / "run")
    assert args[args.index("--warm-transport") + 1] == str(
        probe.BUNDLE / "codex_subscription_warm_v2.py")
    assert "--model" not in args and "--prompt" not in args


def test_public_error_is_bounded_classification_only():
    event = {"id": 7, "error": {"code": -32001,
        "message": "Failed to spawn: too many threads PRIVATE_PROMPT",
        "data": {"secret": "PRIVATE_DATA"}}}
    public = probe.public_error(event)
    assert public["rpc_error_code"] == -32001
    assert public["message_class"] == "resource_limit"
    assert "PRIVATE" not in json.dumps(public)
    private = probe.private_error(event)
    assert "PRIVATE_PROMPT" in private["message"]
    assert "PRIVATE_DATA" in private["data"]
    assert probe.public_error({"id": 1, "result": []}) == {"response_shape": "other"}


def test_observer_captures_first_rpc_before_return_and_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "cgroup_tasks", lambda: {
        "pids_current": 120, "pids_peak": 125, "pids_max": 128,
        "pids_events_max": 1, "controller_threads": 9})
    observer = probe.ResourceObserver(tmp_path)
    class Session:
        stage = "thread/start"
        created_at = probe.time.monotonic() - 16
        def receive(self):
            return {"id": 5, "error": {"code": -32000,
                "message": "resource temporarily unavailable SECRET"}}
    warm = SimpleNamespace(WarmSession=Session)
    probe.instrument(warm, observer)
    event = warm.WarmSession().receive()
    assert event["id"] == 5
    first = json.loads((tmp_path / "public-first-rpc-error.json").read_text())
    assert first["method"] == "thread/start"
    assert first["tasks"]["pids_peak"] == 125
    assert "SECRET" not in json.dumps(first)
    private = json.loads((tmp_path / "private-first-rpc-error.json").read_text())
    assert "SECRET" in private["error"]["message"]
    assert (tmp_path / "private-first-rpc-error.json").stat().st_mode & 0o777 == 0o600
    warm.WarmSession().receive()
    assert observer.report()["rpc_responses"] == 2
    assert observer.report()["max_observed"]["pids_current"] == 120


def test_non_dict_result_captured_without_leaking_value(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "cgroup_tasks", lambda: None)
    observer = probe.ResourceObserver(tmp_path)
    event = {"id": 9, "result": ["PRIVATE_RESULT"]}
    observer.observe(SimpleNamespace(stage="model/list", created_at=probe.time.monotonic()), event)
    public = json.loads((tmp_path / "public-first-rpc-error.json").read_text())
    assert public["response"] == {"response_shape": "other"}
    assert "PRIVATE_RESULT" not in json.dumps(public)
    private = json.loads((tmp_path / "private-first-rpc-error.json").read_text())
    assert "PRIVATE_RESULT" in private["error"]["raw_result"]
    assert observer.report()["first_rpc_failure"]["method"] == "model/list"


def test_pinned_loaded_client_default_uses_instrumented_session(tmp_path):
    frozen = Path("/private/tmp/hymem-r9-summary-20260927.XzIhDt")
    if not (frozen / "candidate").is_dir():
        pytest.skip("frozen candidate unavailable")
    repository = Path(__file__).resolve().parents[3]
    for name, expected in probe.PINS.items():
        source = (frozen / name if name == "headless-source-map.json" else
                  repository / "benchmarks" / name if name.startswith("codex_") or
                  name == "luna_stage_accounting.py" else repository / "tools/diagnostics" / name)
        shutil.copyfile(source, tmp_path / name)
        assert probe.digest(tmp_path / name) == expected
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    program = r'''
import hashlib, importlib.util, pathlib, sys, time
bundle, frozen, wrapper = map(pathlib.Path, sys.argv[1:4])
spec = importlib.util.spec_from_file_location("actual_profiled", bundle / "luna_subscription_lme_profiled_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner = module.warm_runner
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dataset = bundle / "dataset.json"
runner.old.DATASET_SHA256 = digest(dataset)
files, *_, warm = runner.load_verified(
    candidate=frozen / "candidate", inventory_stamp=bundle / "headless-source-map.json",
    inventory_sha256=digest(bundle / "headless-source-map.json"),
    dataset=dataset, dataset_sha256=digest(dataset), binary=pathlib.Path("/usr/bin/true"),
    base_path=bundle / "codex_subscription.py",
    concurrent_path=bundle / "codex_subscription_concurrent_v2.py",
    warm_path=bundle / "codex_subscription_warm_v2.py")
assert files == 508
assert warm.WarmSubscriptionClient.__init__.__kwdefaults__["session_factory"] is warm.WarmSession
observer_spec = importlib.util.spec_from_file_location("resource_probe_actual", wrapper)
observer_module = importlib.util.module_from_spec(observer_spec)
observer_spec.loader.exec_module(observer_module)
original_class = warm.WarmSession
original_receive = original_class.receive
event = {"id": 3, "error": {"code": -32000, "message": "resource exhausted SECRET"}}
original_class.receive = lambda self: event
observer_module.instrument(warm, observer_module.ResourceObserver(bundle))
assert warm.WarmSession is original_class
assert warm.WarmSession.receive is not original_receive
assert warm.WarmSubscriptionClient.__init__.__kwdefaults__["session_factory"] is original_class
session = original_class.__new__(original_class)
session.stage = "thread/start"
session.created_at = time.monotonic()
assert session.receive() is event
assert (bundle / "private-first-rpc-error.json").exists()
assert "SECRET" not in (bundle / "public-first-rpc-error.json").read_text()
'''
    result = subprocess.run(["/opt/anaconda3/bin/python", "-I", "-c", program,
        str(tmp_path), str(frozen), str(SOURCE)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
