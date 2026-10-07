"""Offline controls for the one-shot invented-text access check."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_subscription_access_check_v1 as check


def test_receipt_contains_fixed_access_bounds(monkeypatch, tmp_path):
    root = tmp_path / ".hymem-lme-diagnostic-test1234"
    root.mkdir()
    root.chmod(0o700)
    monkeypatch.setattr(check, "HOST_ROOT", tmp_path)
    monkeypatch.setattr(check, "HOST_UID", root.stat().st_uid)
    monkeypatch.setattr(check, "_tool", lambda _: Path(check.__file__))
    runner = SimpleNamespace(PINS={"source": "a" * 64},
        DIAGNOSTIC_HELPER_SHA256="b" * 64, ACCEPTED_MAP_SHA256="c" * 64,
        DATASET_SHA256="d" * 64)
    launcher = SimpleNamespace(_load_runner=lambda _: runner)
    receipt = check.receipt_for(root, launcher)
    assert receipt["limits"] == [2, 32_000, 300]
    assert receipt["invocation_seconds"] == 120
    assert receipt["runtime_seconds"] == 330
    assert receipt["source_sha256"][check.TOOL_NAME] == check._sha(Path(check.__file__))
    command = check.command(root, receipt, "a" * 64)
    for required in ("--property=Restart=no", "--property=KillMode=control-group",
                     "--property=RuntimeMaxSec=330s", "--property=TimeoutStopSec=10s",
                     "--property=MemoryMax=4294967296", "--property=TasksMax=256",
                     "--property=CPUQuota=200%", "--property=OOMPolicy=kill"):
        assert required in command


def test_prepare_is_zero_inference_and_writes_once(monkeypatch, tmp_path):
    root = tmp_path / ".hymem-lme-diagnostic-test1234"
    root.mkdir()
    calls = []
    launcher = SimpleNamespace(host_admission=lambda: calls.append("admission"))
    monkeypatch.setattr(check, "_root", lambda value: value)
    monkeypatch.setattr(check, "verify_sources", lambda _: (launcher, {}))
    monkeypatch.setattr(check, "build_fixture", lambda _: calls.append("fixture"))
    monkeypatch.setattr(check, "receipt_for", lambda *_: {"schema": check.SCHEMA,
        "unit": "hymem-luna-access-check-test1234.service"})
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **k: pytest.fail("subprocess during prepare"))
    result = check.prepare(root)
    assert result["model_calls"] == 0 and calls == ["fixture", "admission"]
    assert (root / "access-receipt.json").exists()
    with pytest.raises(ValueError, match="already_prepared"):
        check.prepare(root)


def test_launch_consumes_marker_before_dispatch_and_never_retries(monkeypatch, tmp_path):
    root = tmp_path / ".hymem-lme-diagnostic-test1234"
    root.mkdir()
    for name in ("private-access", "access-empty", "access-tmp"):
        (root / name).mkdir(mode=0o700)
    launcher = SimpleNamespace(host_admission=lambda: None, _bus_env=lambda: {})
    monkeypatch.setattr(check, "_root", lambda value: value)
    monkeypatch.setattr(check, "verify_sources", lambda _: (launcher, {}))
    monkeypatch.setattr(check, "_receipt", lambda *a, **k: {"unit": "hymem-luna-access-check-test.service"})
    monkeypatch.setattr(check, "command", lambda *a: ["systemd-run"])
    calls = []
    def dispatch(*args, **kwargs):
        calls.append(json.loads((root / "access-attempt.json").read_text()))
        raise subprocess.TimeoutExpired(args[0], 20)
    monkeypatch.setattr(check.subprocess, "run", dispatch)
    with pytest.raises(subprocess.TimeoutExpired):
        check.launch(root, "a" * 64)
    assert calls == [{"receipt_sha256": "a" * 64, "one_shot": True}]
    with pytest.raises(FileExistsError):
        check.launch(root, "a" * 64)
    assert len(calls) == 1


def test_result_rejects_forged_success_and_private_text():
    good = {"schema": check.SCHEMA, "status": "transport_verified",
        "turns": 2, "known_tokens": 100, "usage_complete": True,
        "reserved": 0, "in_flight": 0, "ordinary_completed": True,
        "staged_completed": True, "schema_acknowledged": True,
        "staged_response_valid": False, "client_cleanup_verified": True,
        "resource_denials": 0, "resource_oom": 0, "containment_verified": True,
        "first_failure": None}
    assert check.validate_result(good)
    for key, bad in (("known_tokens", 32_001), ("usage_complete", False),
                     ("in_flight", 1), ("client_cleanup_verified", False),
                     ("resource_denials", 1), ("schema_acknowledged", False),
                     ("status", ["transport_verified"]), ("staged_response_valid", 1),
                     ("first_failure", {"message": "private provider text"})):
        trial = dict(good, **{key: bad})
        assert not check.validate_result(trial)


def test_recursive_cgroup_requires_unpopulated_descendants(monkeypatch, tmp_path):
    group = tmp_path / "group"
    group.mkdir()
    for name in ("cgroup.procs", "cgroup.threads"):
        (group / name).write_text("")
    (group / "cgroup.events").write_text("populated 1\n")
    monkeypatch.setattr(check, "_group_policy", lambda _: True)
    assert check._recursive_empty(group) is False
    (group / "cgroup.events").write_text("populated 0\n")
    assert check._recursive_empty(group) is True
    (group / "cgroup.procs").write_text("123\n")
    assert check._recursive_empty(group) is False
