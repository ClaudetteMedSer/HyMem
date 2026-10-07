"""Offline checks for the source-bound timeout diagnostic host chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.diagnostics import siwc_lme_diagnostic_bundle_v3 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v3 as preflight
from tools.diagnostics import siwc_lme_diagnostic_launch_v3 as launch
from tools.diagnostics import siwc_lme_diagnostic_source_install_v3 as install


REPO = Path(__file__).resolve().parents[3]
FROZEN = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle")
ACCEPTED_CODE = Path("/private/tmp/hymem-repaired-four-root-dtBv2p/bundle/code")


def test_real_frozen_source_assembly_and_manifest(tmp_path):
    if not FROZEN.is_dir() or not ACCEPTED_CODE.is_dir():
        pytest.skip("frozen local source unavailable")
    target = tmp_path / "fresh"
    report = bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
        candidate=FROZEN / "candidate", map_path=FROZEN / "source-map.json",
        output=target)
    assert report["schema"] == "siwc-lme-diagnostic-source-bundle-v3"
    assert report["candidate_files"] == 514
    assert report["candidate_map_sha256"] == "94d7b2204ed749d1b3de29c24579aeccb2730c6aa9370e2b8ebab87a7835267e"
    assert report["inventory_sha256"] == bundle.INVENTORY_SHA256
    assert report["model_calls"] == 0
    assert report["dataset_present"] is False
    assert report["credential_present"] is False
    manifest = preflight.source_manifest(target)
    assert len(manifest) == report["candidate_files"] + report["code_files"] + 1
    assert manifest["code/" + bundle.RUNNER_RELATIVE] == bundle.RUNNER_SHA256
    assert len(preflight.archive_bytes(target)) < 64 * 1024 * 1024
    (target / "code" / "unapproved.py").write_text("pass")
    with pytest.raises(ValueError, match="source_file_set_invalid"):
        preflight.source_manifest(target)


def test_frozen_inputs_and_launcher_are_hash_pinned():
    assert hashlib.sha256((REPO / bundle.RUNNER_RELATIVE).read_bytes()).hexdigest() == bundle.RUNNER_SHA256
    assert hashlib.sha256((REPO / install.SOURCE).read_bytes()).hexdigest() == install.SOURCE_SHA256
    assert bundle.FROZEN_INVENTORY_SHA256 == "1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
    assert bundle.INVENTORY_SHA256 == preflight.MAP_SHA == launch.INVENTORY_SHA256
    assert preflight.RUNNER_SHA == launch.RUNNER_SHA256 == bundle.RUNNER_SHA256
    assert "v4.py" in bundle.RUNNER_RELATIVE
    assert install.SOURCE.endswith("launch_v3.py")


def test_systemd_caps_and_one_shot_command():
    root = Path("/home/atta/.hymem-siwc-lme-diagnostic-preflight-abcdefgh")
    receipt = {"unit": "hymem-siwc-lme-diagnostic-preflight-abcdefgh.service",
        "runtime_path": "/home/atta/.hymem-siwc-runtime-v1/bin/python"}
    command = launch.command(root, receipt, "a" * 64)
    assert all(flag in command for flag in (
        "--property=Restart=no", "--property=KillMode=control-group",
        "--property=RuntimeMaxSec=14530s", "--property=TimeoutStopSec=10s",
        "--property=TasksMax=256", "--property=MemoryMax=4294967296",
        "--property=CPUQuota=200%", "--property=OOMPolicy=kill"))
    assert command[command.index("--questions") + 1] == "4"
    assert command[command.index("--workers") + 1] == "4"
    assert command[command.index("--receipt-sha256") + 1] == "a" * 64
    assert str(root / "code" / bundle.RUNNER_RELATIVE) in command
    assert command[-1] == "--run"


def test_installer_receipt_projection_is_exact():
    root = "/home/atta/.hymem-siwc-lme-diagnostic-preflight-abcdefgh"
    projected = {"schema": install.SCHEMA, "root": root,
        "unit": "hymem-siwc-lme-diagnostic-preflight-abcdefgh.service",
        "prepared": True, "model_calls": 0, "receipt_sha256": "a" * 64}
    assert install.project(projected, root=root, returncode=0) == projected
    assert install.project({**projected, "model_calls": False}, root=root, returncode=0) is None
    assert install.project({**projected, "receipt_sha256": "b" * 63}, root=root, returncode=0) is None
    assert install.project({**projected, "secret": "x"}, root=root, returncode=0) is None
    assert install.project(projected, root=root, returncode=1) is None


def test_recursive_cleanup_rejects_live_descendant(tmp_path):
    group = tmp_path / "unit"
    child = group / "nested"
    child.mkdir(parents=True)
    for node in (group, child):
        (node / "cgroup.procs").write_text("")
        (node / "cgroup.threads").write_text("")
        (node / "cgroup.events").write_text("populated 0\n")
    assert launch.recursive_empty(group)
    (child / "cgroup.procs").write_text("12345\n")
    assert not launch.recursive_empty(group)


def test_prepare_requires_fresh_root_before_source_load(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "checked_root", lambda root: root)
    monkeypatch.setattr(launch, "host_admission", lambda: None)
    monkeypatch.setattr(launch, "verify_sources", lambda root: pytest.fail("source load crossed freshness barrier"))
    (tmp_path / "launch-attempt.json").write_text("{}")
    with pytest.raises(ValueError, match="root_already_prepared"):
        launch.prepare(tmp_path)


def test_launch_requires_exact_receipt_and_unused_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "checked_root", lambda root: root)
    monkeypatch.setattr(launch, "host_admission", lambda: None)
    monkeypatch.setattr(launch, "verify_sources", lambda root: pytest.fail("source load crossed receipt barrier"))
    with pytest.raises(ValueError, match="receipt_pin_invalid"):
        launch.launch(tmp_path, "a" * 64)
    receipt = tmp_path / "launch-receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="receipt_pin_invalid"):
        launch.launch(tmp_path, "a" * 64)


def test_manifest_rejects_modified_candidate_file(tmp_path):
    if not FROZEN.is_dir() or not ACCEPTED_CODE.is_dir():
        pytest.skip("frozen local source unavailable")
    target = tmp_path / "fresh"
    bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
        candidate=FROZEN / "candidate", map_path=FROZEN / "source-map.json",
        output=target)
    stamp = json.loads((target / "source-map.json").read_text())
    relative = next(iter(stamp["source_sha256"]))
    with (target / "candidate" / relative).open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(ValueError, match="source_file_drift"):
        preflight.source_manifest(target)
