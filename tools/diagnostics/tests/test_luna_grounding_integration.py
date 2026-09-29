"""Offline gates for the derived four-question Luna runner and receipt."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


DIAG = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIAG / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("luna_grounding_lme")
launcher = load("luna_grounding_lme_launch")
reader = load("luna_grounding_lme_progress")
builder = load("luna_grounding_candidate")


def frozen():
    base = Path("/private/tmp/hymem-r9-summary-20260927.XzIhDt")
    if not (base / "candidate").is_dir():
        pytest.skip("local frozen candidate unavailable")
    return base / "candidate", base / "headless-source-map.json"


def candidate(tmp_path):
    source, original_stamp = frozen()
    grounded = tmp_path / "candidate"
    grounded_stamp = tmp_path / "headless-grounding-source-map.json"
    proof = builder.prepare(source, original_stamp, grounded, grounded_stamp)
    return source, original_stamp, grounded, grounded_stamp, proof


def test_exact_delta_and_wrong_map_fail_closed(tmp_path, monkeypatch):
    source, original_stamp, grounded, stamp, proof = candidate(tmp_path)
    assert proof["grounded_map_sha256"] == runner.GROUNDED_MAP_SHA256
    assert builder.verify_derived(source, grounded, stamp, original_stamp) == proof
    wrong = json.loads(stamp.read_text())
    wrong["source_sha256"]["hymem/extraction/prompts/__init__.py"] = "0" * 64
    stamp.write_text(json.dumps(wrong))
    with pytest.raises(Exception):
        builder.verify_derived(source, grounded, stamp, original_stamp)


def test_changed_other_file_and_wrong_prompt_fail_closed(tmp_path):
    source, original_stamp, grounded, stamp, _ = candidate(tmp_path)
    other = grounded / "hymem/extraction/chunk.py"
    other.write_bytes(other.read_bytes() + b"\n# drift\n")
    with pytest.raises(Exception):
        builder.verify_derived(source, grounded, stamp, original_stamp)
    other.write_bytes((source / "hymem/extraction/chunk.py").read_bytes())
    prompt = grounded / builder.PROMPT_RELATIVE
    prompt.write_bytes(prompt.read_bytes() + b"\n# drift\n")
    with pytest.raises(Exception):
        builder.verify_derived(source, grounded, stamp, original_stamp)


def test_bound_inventory_uses_explicit_new_map(tmp_path, monkeypatch):
    source, original_stamp, grounded, stamp, proof = candidate(tmp_path)
    monkeypatch.setattr(runner, "ORIGINAL_CANDIDATE", source)
    calls = []

    def original_verify(path, inventory, sha, *, expected_map_sha256=None,
                        expected_file_count=None):
        calls.append((expected_map_sha256, expected_file_count))
        return 508

    old = SimpleNamespace(verify_inventory=original_verify,
                          SOURCE_MAP_SHA256=builder.ORIGINAL_MAP_SHA256)
    warm = SimpleNamespace(old=old, RUNNER_SHA256="old", SCHEMA="old")
    profile = SimpleNamespace(warm_runner=warm)
    runner.bind_profile(profile, builder, grounded, stamp, original_stamp,
                        proof["grounded_inventory_sha256"])
    assert old.verify_inventory(grounded, stamp, proof["grounded_inventory_sha256"]) == 508
    assert calls == [(runner.GROUNDED_MAP_SHA256, 508)]
    assert old.SOURCE_MAP_SHA256 == runner.GROUNDED_MAP_SHA256
    assert warm.SCHEMA == runner.SCHEMA
    assert warm.RUNNER_SHA256 == launcher.RUNNER_SHA256
    with pytest.raises(RuntimeError, match="grounding_inventory_argument_drift"):
        old.verify_inventory(grounded, stamp, "0" * 64)


def receipt(root):
    unit = "hymem-luna-lme-grounding-abcdefgh.service"
    return {"schema": "luna-grounding-launch-v1", "root": str(root), "unit": unit,
        "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "output": "run", "candidate": str(root / "candidate"),
        "original_candidate": str(reader.ORIGINAL_CANDIDATE),
        "dataset": "/pinned/data.json", "dataset_sha256": reader.base.DATASET,
        "inventory_stamp": str(root / reader.DERIVED_STAMP),
        "inventory_sha256": "a" * 64,
        "candidate_source_map_sha256": reader.GROUNDED_MAP_SHA256,
        "runner_sha256": reader.RUNNER_SHA256,
        "launcher_sha256": reader.LAUNCHER_SHA256,
        "source_sha256": {**reader.PINS, reader.DERIVED_STAMP: "a" * 64},
        "binary": str(reader.base.BINARY), "binary_sha256": "b" * 64,
        "runtime_max_seconds": 14530, "timeout_stop_seconds": 10,
        "memory_max_bytes": 4294967296, "cpu_quota_percent": 200,
        "tasks_max": 256, "oom_policy": "kill", "kill_mode": "control-group",
        "restart": "no", "remain_after_exit": True, "model": "gpt-6-luna",
        "subscription_only": True, "reported_quota_floor_percent": 25,
        "limits": reader.base.LIMITS}


def test_receipt_policy_and_exact_source_pins(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-lme-grounding-abcdefgh"
    root.mkdir()
    monkeypatch.setattr(reader, "ROOT_PARENT", tmp_path)
    value = receipt(root)
    assert reader.root_identity_valid(root, value["unit"], value["expected_cgroup"])
    assert reader.receipt_valid(value, root, value["unit"], value["expected_cgroup"])
    for key, bad in (("tasks_max", 128), ("memory_max_bytes", 2 * 1024**3),
                     ("candidate_source_map_sha256", "0" * 64),
                     ("launcher_sha256", "0" * 64)):
        changed = dict(value, **{key: bad})
        assert not reader.receipt_valid(changed, root, value["unit"],
                                        value["expected_cgroup"])
    changed = dict(value, source_sha256={**value["source_sha256"],
                "luna_grounding_candidate.py": "0" * 64})
    assert not reader.receipt_valid(changed, root, value["unit"],
                                    value["expected_cgroup"])


def test_terminal_reuses_accepted_validator_with_exact_new_schema(monkeypatch):
    seen = []

    def accepted(root, receipt, safe, result):
        seen.append((safe["schema"], result["schema"]))
        return {"validated": True}

    monkeypatch.setattr(reader, "accepted_verify_terminal", accepted)
    assert reader.verify_terminal(None, None, {"schema": reader.SCHEMA},
                                  {"schema": reader.SCHEMA})["validated"]
    assert seen == [("luna-subscription-lme-profiled-v2",) * 2]
    assert not reader.verify_terminal(None, None,
        {"schema": "luna-subscription-lme-profiled-v2"},
        {"schema": reader.SCHEMA})["validated"]
    assert len(seen) == 1


def test_stdin_reader_bootstrap_uses_strict_root(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-luna-lme-grounding-abcdefgh"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(reader, "ROOT_PARENT", tmp_path)
    monkeypatch.setattr(reader, "OBSERVER_UID", os.getuid())
    monkeypatch.setattr(reader, "__file__", "<stdin>")
    monkeypatch.setattr(sys, "argv", ["-", "--root", str(root)])
    assert reader._base_reader_path() == root / "luna_subscription_capacity_progress.py"
    monkeypatch.setattr(sys, "argv", ["-", "--root", str(tmp_path)])
    with pytest.raises(RuntimeError, match="observer_root_argument_invalid"):
        reader._base_reader_path()


def test_reader_actual_stdin_subprocess_has_safe_failure(tmp_path):
    root = tmp_path / ".hymem-luna-lme-grounding-abcdefgh"
    root.mkdir(mode=0o700)
    shutil.copyfile(DIAG / "luna_subscription_capacity_progress.py",
                    root / "luna_subscription_capacity_progress.py")
    source = (DIAG / "luna_grounding_lme_progress.py").read_text()
    assert source.count('ROOT_PARENT = Path("/home/atta")') == 1
    assert source.count("OBSERVER_UID = 1000") == 1
    source = source.replace('ROOT_PARENT = Path("/home/atta")',
                            f"ROOT_PARENT = Path({str(tmp_path)!r})")
    source = source.replace("OBSERVER_UID = 1000", f"OBSERVER_UID = {os.getuid()}")
    unit = "hymem-luna-lme-grounding-abcdefgh.service"
    args = [sys.executable, "-I", "-B", "-", "--root", str(root),
        "--unit", unit,
        "--expected-cgroup",
        "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "--receipt-sha256", "0" * 64]
    result = subprocess.run(args, input=source, text=True, capture_output=True,
                            cwd=tmp_path, timeout=20)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["read_only"] is True and report["raw_text_exported"] is False
    assert report["completed_and_clean"] is False
    assert report["error"] == "metadata_unavailable_or_invalid"
    assert "Traceback" not in result.stderr
    unknown = subprocess.run(args + ["--unknown"], input=source, text=True,
                             capture_output=True, cwd=tmp_path, timeout=20)
    assert unknown.returncode == 2
    assert "unrecognized arguments" in unknown.stderr
    assert "Traceback" not in unknown.stderr


def test_actual_capacity_terminal_gates_survive_schema_translation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("accepted_progress_fixture",
        DIAG / "tests/test_luna_subscription_profiled_v2_progress.py")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    root, receipt_value, safe, result, source = fixture_module._fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(reader.base, "bounded_json", lambda path: source[str(path)])
    safe["schema"] = result["schema"] = reader.SCHEMA
    safe["runner_sha256"] = reader.RUNNER_SHA256
    safe["candidate_source_map_sha256"] = reader.GROUNDED_MAP_SHA256
    assert reader.verify_terminal(root, receipt_value, safe, result)["validated"] is True
    safe["candidate_source_map_sha256"] = "0" * 64
    assert reader.verify_terminal(root, receipt_value, safe, result)["validated"] is False
    safe["candidate_source_map_sha256"] = reader.GROUNDED_MAP_SHA256
    result["stage_accounting"]["q-0000"]["reader"]["known_tokens"] += 1
    assert reader.verify_terminal(root, receipt_value, safe, result)["validated"] is False
    result["stage_accounting"]["q-0000"]["reader"]["known_tokens"] -= 1
    safe["usage_complete"] = False
    assert reader.verify_terminal(root, receipt_value, safe, result)["validated"] is False
    safe["usage_complete"] = True
    result["canary"]["passed"] = False
    assert reader.verify_terminal(root, receipt_value, safe, result)["validated"] is False


def test_real_derived_candidate_loads_with_accepted_frozen_modules(tmp_path):
    source, original_stamp = frozen()
    grounded_root = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi")
    grounded = grounded_root / "candidate"
    stamp = grounded_root / "headless-grounding-source-map.json"
    if not grounded.is_dir() or not stamp.is_file():
        pytest.skip("verified derived candidate unavailable")
    dataset = tmp_path / "offline-dataset.json"
    dataset.write_text("[]")
    script = r'''
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
diag, source, original_stamp, grounded, stamp, dataset, transport = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("grounding_runner_offline", diag / "luna_grounding_lme.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ORIGINAL_CANDIDATE = source
builder = runner._pinned_module("luna_grounding_candidate.py", runner.BUILDER_SHA256,
                                "grounding_builder_offline")
profile = runner._pinned_module("luna_subscription_lme_profiled_v2.py", runner.PROFILED_SHA256,
                                "grounding_profile_offline")
inventory_sha = hashlib.sha256(stamp.read_bytes()).hexdigest()
proof = runner.bind_profile(profile, builder, grounded, stamp, original_stamp, inventory_sha)
dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
profile.warm_runner.old.DATASET_SHA256 = dataset_sha
files, _, request_type, canary, chunk, _, _, _ = profile.warm_runner.load_verified(
    candidate=grounded, inventory_stamp=stamp, inventory_sha256=inventory_sha,
    dataset=dataset, dataset_sha256=dataset_sha, binary=Path('/usr/bin/true'),
    base_path=transport/'codex_subscription.py',
    concurrent_path=transport/'codex_subscription_concurrent_v2.py',
    warm_path=transport/'codex_subscription_warm_v2.py')
profile.collector.verify_candidate(grounded)
print(json.dumps({'files':files, 'map':proof['grounded_map_sha256'],
    'runner':profile.warm_runner.RUNNER_SHA256, 'schema':profile.warm_runner.SCHEMA,
    'candidate_canary':str(Path(canary.__file__).resolve().is_relative_to(grounded.resolve())),
    'candidate_chunk':str(Path(chunk.__file__).resolve().is_relative_to(grounded.resolve()))}))
'''
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", script,
        str(DIAG), str(source), str(original_stamp), str(grounded), str(stamp),
        str(dataset), str(DIAG.parents[1] / "benchmarks")],
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {"files": 508, "map": runner.GROUNDED_MAP_SHA256,
        "runner": launcher.RUNNER_SHA256, "schema": runner.SCHEMA,
        "candidate_canary": "True", "candidate_chunk": "True"}


def test_one_shot_launch_marker_is_exclusive(tmp_path, monkeypatch, capsys):
    root = tmp_path / ".hymem-luna-lme-grounding-abcdefgh"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(launcher, "HOST_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "HOST_UID", os.getuid())
    proof = {"grounded_map_sha256": launcher.PINS.get("map", "x")}
    value = {"unit": "hymem-luna-lme-grounding-abcdefgh.service"}

    def write_once(path, data):
        with path.open("x") as stream:
            json.dump(data, stream)

    base = SimpleNamespace(host_admission=lambda: None, write_once=write_once)
    monkeypatch.setattr(launcher, "base_launcher", lambda path: base)
    monkeypatch.setattr(launcher, "verify_sources", lambda path: proof)
    monkeypatch.setattr(launcher, "receipt_for", lambda path, p: value)
    monkeypatch.setattr(launcher, "command", lambda path, p: ["true"])
    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=0))
    receipt_path = root / "launch-receipt.json"
    receipt_path.write_text(json.dumps(value))
    args = ["--launch-root", str(root), "--receipt-sha256", launcher.sha(receipt_path)]
    assert launcher.main(args) == 0
    assert launcher.main(args) == 1
    assert (root / "launch-attempt.json").exists()
    assert "never_retry_launch" in capsys.readouterr().out
