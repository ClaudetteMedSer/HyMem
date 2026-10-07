"""Offline safety controls for the repaired four-question host chain."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_lme_diagnostic_host_preflight_v9 as preflight
from tools.diagnostics import luna_lme_diagnostic_launch_v9 as launch
from tools.diagnostics import luna_lme_diagnostic_progress_v10 as reader


def test_preflight_projects_only_exact_finite_metadata():
    receipt = {"schema": "luna-lme-diagnostic-host-preflight-v9",
        "root": "/home/atta/.hymem-lme-diagnostic-preflight-abcdefgh",
        "candidate_files": 514, "code_files": 16,
        "inventory_sha256": preflight.MAP_SHA,
        "binary_sha256": reader.BINARY_SHA256,
        "dataset_sha256": reader.DATASET_SHA256,
        "preflight_verified": True, "selected_count": 4, "model_calls": 0}
    assert preflight._success_projection(receipt) == receipt
    assert preflight._success_projection({**receipt, "PRIVATE": "row text"}) is None
    for key, value in (("model_calls", False), ("selected_count", 4.0),
                       ("binary_sha256", "0" * 64), ("root", "/tmp/unsafe")):
        assert preflight._success_projection({**receipt, key: value}) is None


def test_launcher_prior_unit_requires_recursive_emptiness(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "CGROUP_ROOT", tmp_path)
    unit = "hymem-luna-prior_A.service"
    group = tmp_path / "user.slice/user-1000.slice/user@1000.service/app.slice" / unit
    group.mkdir(parents=True)
    def empty(path):
        (path / "cgroup.procs").write_text("")
        (path / "cgroup.threads").write_text("")
        (path / "cgroup.events").write_text("populated 0\n")
    empty(group)
    state = ("active", "exited", "0", "")
    assert launch._prior_unit_stopped(unit, state)
    assert launch._prior_unit_stopped(unit, (*state[:3], "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit))
    assert not launch._prior_unit_stopped(unit, (*state[:3], "/wrong/group"))
    assert not launch._prior_unit_stopped("hymem-../bad.service", state)
    child = group / "child"; child.mkdir(); empty(child)
    (child / "cgroup.procs").write_text("123\n")
    assert not launch._prior_unit_stopped(unit, state)
    (child / "cgroup.procs").write_text("")
    (child / "link").symlink_to(group, target_is_directory=True)
    assert not launch._prior_unit_stopped(unit, state)


def test_timeout_cleanup_is_independent_of_terminal_result(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"; runtime.mkdir(mode=0o700)
    with (runtime / "bus").open("wb") as bus:
        monkeypatch.setattr(reader.stat, "S_ISSOCK", lambda _mode: True)
        monkeypatch.setattr(reader.sys, "platform", "linux")
        monkeypatch.setattr(reader, "RUNTIME", runtime)
        monkeypatch.setattr(reader, "CGROUP_ROOT", tmp_path)
        monkeypatch.setattr(reader, "ROOT_UID", os.getuid())
        unit = "hymem-luna-lme-diagnostic-new.service"
        expected = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
        receipt = {"unit": unit, "expected_cgroup": expected}
        values = {"ActiveState": "failed", "SubState": "failed", "MainPID": "0",
            "ControlGroup": "", "NRestarts": "0", "Result": "timeout",
            "ExecMainStatus": "0", "MemoryMax": "4294967296", "TasksMax": "256",
            "CPUQuotaPerSecUSec": "2s", "KillMode": "control-group",
            "Restart": "no", "RemainAfterExit": "yes", "OOMPolicy": "kill",
            "RuntimeMaxUSec": "14530s", "TimeoutStopUSec": "10s"}
        output = "".join(f"{key}={value}\n" for key, value in values.items())
        monkeypatch.setattr(reader.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output))
        assert reader._failed_exit_cleanup(receipt)
        group = tmp_path / expected.lstrip("/")
        group.mkdir(parents=True)
        for node in (group,):
            (node / "cgroup.procs").write_text("")
            (node / "cgroup.threads").write_text("")
            (node / "cgroup.events").write_text("populated 0\n")
        child = group / "child"; child.mkdir()
        (child / "cgroup.procs").write_text("99\n")
        (child / "cgroup.threads").write_text("")
        (child / "cgroup.events").write_text("populated 1\n")
        assert not reader._failed_exit_cleanup(receipt)


def test_execution_marker_required_for_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-lme-diagnostic-preflight-abcdefgh"
    root.mkdir()
    (root / "launch-attempt.json").write_text(json.dumps(
        {"receipt_sha256": "a" * 64, "one_shot": True}))
    run = root / "run"; run.mkdir()
    (run / "diagnostic-checkpoint.json").write_text("{}")
    monkeypatch.setattr(reader, "_root", lambda path: path)
    monkeypatch.setattr(reader, "_receipt", lambda path, digest:
        {"selected_count": 4, "expected_cgroup": "/g", "unit": "u"})
    monkeypatch.setattr(reader, "_runtime", lambda receipt: "unverified")
    monkeypatch.setattr(reader, "_failed_exit_cleanup", lambda receipt: True)
    with pytest.raises(ValueError, match="checkpoint_without_execution_marker"):
        reader.inspect(root, "a" * 64)
