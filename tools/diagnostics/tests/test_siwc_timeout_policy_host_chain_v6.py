"""Offline checks for the approved 300-second SIWC host chain."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.diagnostics import siwc_lme_diagnostic_bundle_v6 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v6 as preflight
from tools.diagnostics import siwc_lme_diagnostic_launch_v6 as launch
from tools.diagnostics import siwc_lme_diagnostic_source_install_v6 as install


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
    assert report["schema"] == "siwc-lme-diagnostic-source-bundle-v6"
    assert report["candidate_files"] == 514
    assert report["code_files"] == 26
    assert report["candidate_map_sha256"] == "94d7b2204ed749d1b3de29c24579aeccb2730c6aa9370e2b8ebab87a7835267e"
    assert report["inventory_sha256"] == bundle.INVENTORY_SHA256
    assert report["model_calls"] == 0
    assert report["dataset_present"] is False
    assert report["credential_present"] is False
    manifest = preflight.source_manifest(target)
    assert len(manifest) == 541 == report["candidate_files"] + report["code_files"] + 1
    assert manifest["code/" + bundle.RUNNER_RELATIVE] == bundle.RUNNER_SHA256
    assert manifest["code/benchmarks/chatgpt_plan_responses_v10.py"] == (
        "a716a7e2a180c96f0f9840eb01eaeee0469302cf92911ae558d1e6c5926f88e2")
    assert "code/benchmarks/chatgpt_plan_responses_v8.py" not in manifest
    assert "code/benchmarks/chatgpt_plan_responses_v9.py" not in manifest
    assert "code/benchmarks/chatgpt_plan_responses_v7.py" not in manifest
    archive = preflight.archive_bytes(target)
    assert len(archive) < 64 * 1024 * 1024
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as stream:
        names = stream.getnames()
        assert len(names) == 542 and names[0] == "manifest.json"
        assert set(names[1:]) == set(manifest)
    (target / "code" / "unapproved.py").write_text("pass")
    with pytest.raises(ValueError, match="source_file_set_invalid"):
        preflight.source_manifest(target)


def test_frozen_inputs_and_launcher_are_hash_pinned():
    assert hashlib.sha256((REPO / bundle.RUNNER_RELATIVE).read_bytes()).hexdigest() == bundle.RUNNER_SHA256
    assert hashlib.sha256((REPO / install.SOURCE).read_bytes()).hexdigest() == install.SOURCE_SHA256
    assert bundle.FROZEN_INVENTORY_SHA256 == "1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
    assert bundle.INVENTORY_SHA256 == preflight.MAP_SHA == launch.INVENTORY_SHA256
    assert preflight.RUNNER_SHA == launch.RUNNER_SHA256 == bundle.RUNNER_SHA256
    assert bundle.RUNNER_RELATIVE.endswith("siwc_lme_diagnostic_v7.py")
    assert install.SOURCE.endswith("launch_v6.py")
    assert preflight.REMOTE.count("siwc_lme_diagnostic_v7.py") == 1
    assert "siwc-lme-diagnostic-host-preflight-v6" in preflight.REMOTE
    assert "siwc_lme_diagnostic_launch_v6.py" in install.REMOTE


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


def test_preflight_projection_rejects_changed_version_or_count():
    manifest = {"source-map.json": preflight.MAP_SHA,
        **{f"candidate/{i}": "a" * 64 for i in range(514)},
        **{f"code/{i}": "b" * 64 for i in range(26)}}
    value = {"schema": "siwc-lme-diagnostic-host-preflight-v6",
        "root": "/home/atta/.hymem-siwc-lme-diagnostic-preflight-invented01",
        "candidate_files": 514, "code_files": 26,
        "inventory_sha256": preflight.MAP_SHA,
        "runtime_sha256": "17b78e0a93175e86f9ac03141924fd7a7f0c0c52e66b34bfa0de20ffef989df1",
        "runtime_site_sha256": "2bed78ec3df853e3efe5052b30d514a2765a2097e183f4b3c32a3d53ef54806d",
        "dataset_sha256": preflight.DATASET_SHA,
        "grant_identity_sha256": "5f91fe05fb7d3b0552b7247d6a81f3fd29aae894dbcf45828b1556a0b11633dc",
        "preflight_verified": True, "selected_count": 4, "model_calls": 0}
    assert preflight._success_projection(value, manifest)
    assert not preflight._success_projection({**value, "schema": "siwc-lme-diagnostic-host-preflight-v5"}, manifest)
    assert not preflight._success_projection({**value, "code_files": 27}, manifest)
    assert not preflight._success_projection({**value, "model_calls": False}, manifest)


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


@pytest.mark.parametrize("relative", [
    "benchmarks/chatgpt_plan_responses_v10.py",
    "benchmarks/chatgpt_plan_responses_v6.py",
    "benchmarks/chatgpt_plan_lme_v5.py",
    "tools/diagnostics/siwc_lme_diagnostic_v7.py",
])
def test_manifest_rejects_transport_bridge_or_runner_drift(tmp_path, relative):
    if not FROZEN.is_dir() or not ACCEPTED_CODE.is_dir():
        pytest.skip("frozen local source unavailable")
    target = tmp_path / "fresh"
    bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
        candidate=FROZEN / "candidate", map_path=FROZEN / "source-map.json",
        output=target)
    with (target / "code" / relative).open("ab") as handle:
        handle.write(b"\n# invented drift\n")
    with pytest.raises(ValueError, match="source_identity_invalid|source_file_drift"):
        preflight.source_manifest(target)


def test_actual_isolated_source_import_and_receipt_bound(tmp_path):
    if not FROZEN.is_dir() or not ACCEPTED_CODE.is_dir():
        pytest.skip("frozen local source unavailable")
    target = tmp_path / ".hymem-siwc-lme-diagnostic-reaper0001"
    bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
        candidate=FROZEN / "candidate", map_path=FROZEN / "source-map.json",
        output=target)
    script = r'''
import importlib.util,json,pathlib,sys
from types import SimpleNamespace
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
        raise AssertionError('external_operation_forbidden')
    if event == 'open' and isinstance(args[0],(str,bytes)):
        path=str(args[0])
        if path.endswith('/auth.json') or '/.codex/' in path or '/.hymem-chatgpt-plan-lme/' in path:
            raise AssertionError('credential_read_forbidden')
sys.addaudithook(audit)
root=pathlib.Path(sys.argv[1]);path=root/'code/tools/diagnostics/siwc_lme_diagnostic_v7.py'
spec=importlib.util.spec_from_file_location('isolated_siwc_reaper_host',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
assert pathlib.Path(loaded['siwc'].__file__).name=='chatgpt_plan_lme_v5.py'
assert pathlib.Path(loaded['siwc'].transport.__file__).name=='chatgpt_plan_responses_v10.py'
assert pathlib.Path(loaded['siwc'].transport_v6.__file__).name=='chatgpt_plan_responses_v6.py'
questions=[{'question_id':f'invented-{i}'} for i in range(4)]
dataset=root/'absent-invented-dataset.json';original=runner._sha
runner._sha=lambda p:runner.DATASET_SHA256 if p==dataset else original(p)
loaded.update(source_only=False,dataset=dataset,questions=questions,
    prior=SimpleNamespace(SelectedQuestions=lambda *_:questions))
receipt=runner.receipt_for(root,loaded)
assert len(receipt['source_sha256'])==26
assert receipt['source_sha256']['benchmarks/chatgpt_plan_responses_v10.py']==runner.SIWC_PINS['benchmarks/chatgpt_plan_responses_v10.py']
assert receipt['source_sha256']['benchmarks/chatgpt_plan_responses_v6.py']==runner.SIWC_PINS['benchmarks/chatgpt_plan_responses_v6.py']
assert receipt['source_sha256']['benchmarks/chatgpt_plan_lme_v5.py']==runner.SIWC_PINS['benchmarks/chatgpt_plan_lme_v5.py']
assert receipt['model']=='gpt-5.6-luna' and receipt['reasoning']=='low'
assert receipt['store'] is False and receipt['stream'] is True
assert receipt['selected_count']==receipt['workers']==4
assert receipt['invocation_seconds']==300.0
assert 4000<len(runner._canonical(receipt))<=8192
print(json.dumps({'source_only_import':True,'receipt_bytes':len(runner._canonical(receipt)),'model_calls':0}))
'''
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(target)],
        capture_output=True, text=True, timeout=40)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["source_only_import"] is True
