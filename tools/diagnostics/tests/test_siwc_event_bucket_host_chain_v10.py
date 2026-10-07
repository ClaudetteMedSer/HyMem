"""Offline source-only checks for the exploratory 600-second SIWC host chain."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.diagnostics import siwc_lme_diagnostic_bundle_v10 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v10 as preflight
from tools.diagnostics import siwc_lme_diagnostic_launch_v10 as launch
from tools.diagnostics import siwc_lme_diagnostic_source_install_v10 as install


REPO = Path(__file__).resolve().parents[3]
FROZEN = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle")
ACCEPTED_CODE = Path("/private/tmp/hymem-repaired-four-root-dtBv2p/bundle/code")


def _assemble(tmp_path):
    if not FROZEN.is_dir() or not ACCEPTED_CODE.is_dir():
        pytest.skip("frozen accepted local source unavailable")
    target = tmp_path / ".hymem-siwc-lme-diagnostic-eventbucket0001"
    result = bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
        candidate=FROZEN / "candidate", map_path=FROZEN / "source-map.json", output=target)
    return target, result


def test_source_pins_and_immutable_caps():
    assert hashlib.sha256((REPO / bundle.RUNNER_RELATIVE).read_bytes()).hexdigest() == bundle.RUNNER_SHA256
    assert hashlib.sha256((REPO / install.SOURCE).read_bytes()).hexdigest() == install.SOURCE_SHA256
    assert bundle.RUNNER_SHA256 == preflight.RUNNER_SHA == launch.RUNNER_SHA256
    assert bundle.INVENTORY_SHA256 == preflight.MAP_SHA == launch.INVENTORY_SHA256
    assert bundle.RUNNER_RELATIVE.endswith("siwc_lme_diagnostic_v11.py")
    assert install.SOURCE.endswith("siwc_lme_diagnostic_launch_v10.py")
    assert preflight.REMOTE.count("siwc_lme_diagnostic_v11.py") == 1
    assert "siwc-lme-diagnostic-host-preflight-v10" in preflight.REMOTE
    assert "siwc_lme_diagnostic_launch_v10.py" in install.REMOTE
    root = Path("/home/atta/.hymem-siwc-lme-diagnostic-preflight-invented01")
    receipt = {"unit": "hymem-siwc-lme-diagnostic-preflight-invented01.service",
               "runtime_path": "/home/atta/.hymem-siwc-runtime-v1/bin/python"}
    command = launch.command(root, receipt, "a" * 64)
    for flag in ("--property=Restart=no", "--property=KillMode=control-group",
                 "--property=RuntimeMaxSec=25330s", "--property=TimeoutStopSec=10s",
                 "--property=TasksMax=256", "--property=MemoryMax=4294967296",
                 "--property=CPUQuota=200%", "--property=OOMPolicy=kill"):
        assert flag in command
    assert command[command.index("--questions") + 1] == "4"
    assert command[command.index("--workers") + 1] == "4"
    assert command[command.index("--receipt-sha256") + 1] == "a" * 64
    assert command[-1] == "--run"


def test_real_assembly_manifest_and_source_drift(tmp_path):
    target, result = _assemble(tmp_path)
    assert result["schema"] == "siwc-lme-diagnostic-source-bundle-v10"
    assert result["candidate_files"] == 514 and result["code_files"] == 26
    assert result["model_calls"] == 0 and result["dataset_present"] is False
    assert result["credential_present"] is False
    manifest = preflight.source_manifest(target)
    assert len(manifest) == 541
    for relative in ("benchmarks/chatgpt_plan_lme_v8.py",
                     "benchmarks/chatgpt_plan_responses_v11.py",
                     "tools/diagnostics/lme_chatgpt_plan_owner_v3.py",
                     "tools/diagnostics/lme_chatgpt_plan_refresh_v3.py"):
        assert manifest["code/" + relative] == hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
    for forbidden in ("benchmarks/chatgpt_plan_lme_v7.py",
                      "benchmarks/chatgpt_plan_responses_v10.py",
                      "tools/diagnostics/lme_chatgpt_plan_owner_v2.py",
                      "tools/diagnostics/lme_chatgpt_plan_refresh_v2.py"):
        assert "code/" + forbidden not in manifest
    archive = preflight.archive_bytes(target)
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as stream:
        names = stream.getnames()
    assert len(names) == 542 and names[0] == "manifest.json"
    assert set(names[1:]) == set(manifest)
    with (target / "code/benchmarks/chatgpt_plan_responses_v11.py").open("ab") as output:
        output.write(b"\n# invented drift\n")
    with pytest.raises(ValueError, match="source_file_drift"):
        preflight.source_manifest(target)


def test_archive_passes_actual_remote_manifest_gate_before_host_checks(tmp_path):
    target, _ = _assemble(tmp_path)
    archive = preflight.archive_bytes(target)
    assert f"RUNNER_SHA='{preflight.RUNNER_SHA}'" in preflight.REMOTE
    # Execute the unmodified remote archive-verification prefix with only
    # host UID/root rebound to invented local values. Stop before any host
    # dataset/runtime/credential inspection or staging.
    prefix = preflight.REMOTE.split("need(regular(DATASET)", 1)[0]
    prefix = prefix.replace("HOST_ROOT=Path('/home/atta')", f"HOST_ROOT=Path({str(tmp_path)!r})")
    script = "import os\nos.getuid=lambda:1000\n" + prefix + "\nprint('archive_verified')\n"
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script],
        input=archive, capture_output=True, timeout=30)
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    assert done.stdout == b"archive_verified\n"


def test_source_only_import_and_receipt_caps(tmp_path):
    target, _ = _assemble(tmp_path)
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
root=pathlib.Path(sys.argv[1]);path=root/'code/tools/diagnostics/siwc_lme_diagnostic_v11.py'
spec=importlib.util.spec_from_file_location('isolated_siwc_bucket_host',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
assert pathlib.Path(loaded['siwc'].__file__).name=='chatgpt_plan_lme_v8.py'
assert pathlib.Path(loaded['siwc'].owner.__file__).name=='lme_chatgpt_plan_owner_v3.py'
assert pathlib.Path(loaded['siwc'].owner._pinned_refresh().__file__).name=='lme_chatgpt_plan_refresh_v3.py'
assert pathlib.Path(loaded['siwc'].transport.__file__).name=='chatgpt_plan_responses_v11.py'
questions=[{'question_id':f'invented-{i}'} for i in range(4)]
dataset=root/'absent-invented-dataset.json';original=runner._sha
runner._sha=lambda p:runner.DATASET_SHA256 if p==dataset else original(p)
loaded.update(source_only=False,dataset=dataset,questions=questions,
    prior=SimpleNamespace(SelectedQuestions=lambda *_:questions))
receipt=runner.receipt_for(root,loaded)
assert len(receipt['source_sha256'])==26
assert receipt['model']=='gpt-5.6-luna' and receipt['reasoning']=='low'
assert receipt['store'] is False and receipt['stream'] is True
assert receipt['selected_count']==receipt['workers']==4
assert receipt['invocation_seconds']==600.0
assert receipt['indexing_seconds']==21600
assert receipt['limits']['question']==[2000,12000000,23400]
assert receipt['limits']['campaign']==[8012,48160000,25200]
assert receipt['limits']['canary']==[12,160000,600]
assert 4000<len(runner._canonical(receipt))<=8192
print(json.dumps({'source_only_import':True,'model_calls':0}))
'''
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(target)],
        capture_output=True, text=True, timeout=40)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"source_only_import": True, "model_calls": 0}


def test_freshness_receipt_projection_and_recursive_cleanup(tmp_path, monkeypatch):
    root = "/home/atta/.hymem-siwc-lme-diagnostic-preflight-invented01"
    projected = {"schema": install.SCHEMA, "root": root,
        "unit": "hymem-siwc-lme-diagnostic-preflight-invented01.service",
        "prepared": True, "model_calls": 0, "receipt_sha256": "a" * 64}
    assert install.project(projected, root=root, returncode=0) == projected
    assert install.project({**projected, "secret": "x"}, root=root, returncode=0) is None
    assert install.project({**projected, "model_calls": False}, root=root, returncode=0) is None
    assert install.project(projected, root=root, returncode=1) is None
    monkeypatch.setattr(launch, "checked_root", lambda root: root)
    monkeypatch.setattr(launch, "host_admission", lambda: None)
    monkeypatch.setattr(launch, "verify_sources", lambda root: pytest.fail("source load crossed freshness barrier"))
    (tmp_path / "launch-attempt.json").write_text("{}")
    with pytest.raises(ValueError, match="root_already_prepared"):
        launch.prepare(tmp_path)
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
