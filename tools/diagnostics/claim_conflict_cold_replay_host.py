"""Fresh, offline retained-response replay for the reviewed cold candidate.

The proof-v3 result authenticates only its original candidate. This stage
requires the installed one-file cold candidate and gives a new run its own
receipt. Import and configuration never contact the remote host.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "cold-replay-proof-v1"
SELF = ROOT / "claim_conflict_cold_replay_host.py"
WORKER = ROOT / "claim_conflict_proof_replay_v3.py"
CAPTURE = ROOT / "capture"
CANDIDATE = ROOT / "candidate"
WORK = ROOT / "work"
V3_ROOT = BASE / "proof-replay-v3"
V3_HOST = V3_ROOT / "claim_conflict_proof_replay_v3_host.py"
V3_HOST_SHA = "a3ddbf16ce89c35f6f6a7f44c86a1089efbbc21abfab64dcc7d073b80772d92c"
V3_WORKER = V3_ROOT / "claim_conflict_proof_replay_v3.py"
V3_WORKER_SHA = "173e9cc13c7af1b6346100892dfb24c9cd84d1a36f69291db90dc6296e360554"
V2_HOST = BASE / "proof-replay-v2/claim_conflict_proof_replay_v2_host.py"
V1_HOST = BASE / "proof-replay-v1/claim_conflict_proof_replay_host.py"
AUDIT_HOST = BASE / "proof-audit-v1/claim_conflict_proof_drift_host.py"
COLD_SUITE = BASE / "cold-replay-pytest-v1"
COLD_CANDIDATE = COLD_SUITE / "candidate"
COLD_MANIFEST = COLD_SUITE / "candidate-manifest.json"
COLD_INSTALL = COLD_SUITE / "install.json"
COLD_SUITE_HOST = COLD_SUITE / "claim_conflict_cold_pytest.py"
COLD_SUITE_HOST_SHA = "f66e270d7ac632791861ecbc8cc4657226b3acdc86aea7c8f8d8e51c8b0d0bf2"
PARENT_CANDIDATE_SHA = "cd6e7810a4c08d771c49658a5954ca71a24d9d1119bcfde3aaa4ccb86dafd694"
PARENT_PHASE1_SHA = "1037bf6d62c3981add3702f96d2ce5bd79d8ebc831f15b30f95bb3724cfd7217"
NEW_PHASE1_SHA = "bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac"
NEW_CANDIDATE_SHA = "ed889d8970c6d7827315de34342996ad55182773c238922fbbdf8510d73c43a4"
BASELINE_RESULT_SHA = "6e602bae89aee5f42be1f3f7efff87ba905a59bc134d09015ffe7a02a672c52d"
NEW_OVERLAY_SHA = "0321fd5419d8079a2cb6ea9802ef4c64df0294a3d00dc7f4204b237b67608344"
REVIEWED_FINAL_CANDIDATE = True  # Parent reviewed final candidate and 45 replay controls.
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def import_v3(path=V3_HOST):
    need(path.is_file() and not path.is_symlink() and sha(path) == V3_HOST_SHA,
         "v3_controller_pin_drift")
    spec = importlib.util.spec_from_file_location("cold_pinned_v3_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derive_candidate(parent):
    need(isinstance(parent, dict) and len(parent) == 481
         and digest(parent) == PARENT_CANDIDATE_SHA
         and parent.get("hymem/dreaming/phase1.py") == PARENT_PHASE1_SHA,
         "baseline_candidate_manifest_drift")
    expected = dict(parent)
    expected["hymem/dreaming/phase1.py"] = NEW_PHASE1_SHA
    need(digest(expected) == NEW_CANDIDATE_SHA,
         "cold_candidate_manifest_pin_drift")
    return expected


def baseline_gate(v3, baseline, shared, alias, helper):
    """Validate the original proof as original-only evidence."""
    receipt = baseline.installed(shared, alias, helper)
    result_path = V3_ROOT / "result.json"
    need(result_path.is_file() and not result_path.is_symlink()
         and sha(result_path) == BASELINE_RESULT_SHA,
         "baseline_result_pin_drift")
    result = helper.read_json(result_path)
    need(result.get("status") == "completed"
         and result.get("networked_runs_started") == 0,
         "baseline_replay_not_completed")
    stage = result.get("stages", {}).get("replay", {})
    cid = stage.get("container_id")
    need(isinstance(cid, str) and HEX64.fullmatch(cid),
         "baseline_container_missing")
    state = baseline.inspect(helper, cid, baseline.configure(helper)[1])
    need(state["status"] == "exited" and state["pid"] == 0
         and state["exit_code"] == 0 and state["oom_killed"] is False,
         "baseline_container_not_clean")
    metadata = baseline.project(stage.get("metadata"))
    baseline.verdict(metadata, receipt)
    need(receipt.get("candidate_sha256") == PARENT_CANDIDATE_SHA
         and receipt.get("phase1_sha256") == PARENT_PHASE1_SHA,
         "baseline_receipt_candidate_drift")
    return receipt


def cold_candidate_gate(v3, helper, expected):
    """Bind the new installed suite candidate, without borrowing its verdict."""
    need(COLD_MANIFEST.is_file() and not COLD_MANIFEST.is_symlink()
         and COLD_INSTALL.is_file() and not COLD_INSTALL.is_symlink()
         and COLD_SUITE_HOST.is_file() and not COLD_SUITE_HOST.is_symlink()
         and sha(COLD_SUITE_HOST) == COLD_SUITE_HOST_SHA,
         "cold_suite_not_installed")
    need(helper.read_json(COLD_MANIFEST) == expected
         and v3.inventory(COLD_CANDIDATE) == expected,
         "cold_candidate_inventory_drift")
    receipt = helper.read_json(COLD_INSTALL)
    need(receipt.get("candidate_sha256") == digest(expected)
         and receipt.get("parent_candidate_sha256") == PARENT_CANDIDATE_SHA
         and receipt.get("candidate_phase1_sha256") == NEW_PHASE1_SHA
         and receipt.get("proof_result_sha256") == BASELINE_RESULT_SHA
         and receipt.get("overlay_sha256") == NEW_OVERLAY_SHA
         and receipt.get("baseline_proof_only") is True
         and receipt.get("new_candidate_replay_verified") is False
         and receipt.get("host_sha256") == COLD_SUITE_HOST_SHA,
         "cold_suite_receipt_drift")
    return receipt


def capture_copy_gate(v3, helper, reference_sha):
    """Bind private replay inputs to the already-retained v3 capture."""
    for name in ("prepersist-001.json", "prepersist-001.sqlite"):
        need(sha(CAPTURE / name) == sha(v3.CAPTURE / name),
             "replay_capture_copy_drift")
    capture = helper.read_json(CAPTURE / "prepersist-001.json")
    snapshot_sha = sha(CAPTURE / "prepersist-001.sqlite")
    need(capture.get("source_sha256") == reference_sha
         and capture.get("database_sha256") == snapshot_sha,
         "replay_capture_binding_drift")
    return sha(CAPTURE / "prepersist-001.json"), snapshot_sha


def controller(v3_path=V3_HOST, *, v2_path=V2_HOST,
               v1_path=V1_HOST, audit_path=AUDIT_HOST):
    v3 = import_v3(v3_path)
    old = v3.controller(v2_path, v1_path=v1_path, audit_path=audit_path)
    baseline = v3.controller(v2_path, v1_path=v1_path, audit_path=audit_path)
    v2 = v3.import_pinned(v2_path, v3.PROOF_V2_HOST_SHA,
                          "cold_pinned_v2_manifest")
    old.ROOT, old.SELF, old.WORKER = ROOT, SELF, WORKER
    old.CAPTURE, old.CANDIDATE, old.WORK = CAPTURE, CANDIDATE, WORK
    old.WORKER_SHA = V3_WORKER_SHA
    old.OVERRIDE_SHAS = {
        **v2.reviewed_overrides(), "hymem/dreaming/phase1.py": NEW_PHASE1_SHA,
    }
    old.reviewed_overrides = lambda: old.OVERRIDE_SHAS
    inherited_configure, inherited_remote = old.configure, old.remote

    def pins(shared, alias, helper):
        need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
             and stat.S_IMODE(ROOT.stat().st_mode) == 0o700
             and CAPTURE.is_dir() and not CAPTURE.is_symlink()
             and stat.S_IMODE(CAPTURE.stat().st_mode) == 0o700
             and WORK.is_dir() and not WORK.is_symlink()
             and stat.S_IMODE(WORK.stat().st_mode) == 0o700,
             "cold_stage_not_private")
        for path, mode in ((SELF, 0o400), (WORKER, 0o400),
                           (CAPTURE / "prepersist-001.json", 0o600),
                           (CAPTURE / "prepersist-001.sqlite", 0o600)):
            helper.regular(path, mode=mode)
        need(sha(WORKER) == V3_WORKER_SHA, "v3_worker_copy_drift")
        original = baseline_gate(v3, baseline, shared, alias, helper)
        parent = v3.source_manifest(v2, helper)
        expected = derive_candidate(parent)
        cold_candidate_gate(v3, helper, expected)
        need(v3.inventory(CANDIDATE) == expected,
             "replay_candidate_inventory_drift")
        capture_sha, snapshot_sha = capture_copy_gate(
            v3, helper, original["reference_sha256"],
        )
        return {
            "source_files": 481,
            "candidate_sha256": digest(expected),
            "phase1_sha256": NEW_PHASE1_SHA,
            "capture_sha256": capture_sha,
            "snapshot_sha256": snapshot_sha,
            "reference_sha256": original["reference_sha256"],
            "worker_sha256": V3_WORKER_SHA,
            "host_sha256": sha(SELF),
            "baseline_candidate_sha256": PARENT_CANDIDATE_SHA,
            "baseline_result_sha256": BASELINE_RESULT_SHA,
            "baseline_install_sha256": sha(V3_ROOT / "install.json"),
            "cold_suite_install_sha256": sha(COLD_INSTALL),
            "cold_suite_manifest_sha256": sha(COLD_MANIFEST),
            "baseline_proof_only": True,
            "new_candidate_replay_verified": False,
        }

    def configure(helper):
        command, mounts = inherited_configure(helper)
        command[command.index("--name") + 1] = "hymem-cold-replay-proof-v1"
        need(command[command.index("--network") + 1] == "none"
             and command[command.index("--phase1-sha256") + 1]
             == NEW_PHASE1_SHA
             and (str(CANDIDATE), "/candidate", False) in mounts
             and (str(CAPTURE), "/capture", False) in mounts
             and (str(WORK), "/work", True) in mounts
             and (str(WORKER), "/diag/claim_conflict_proof_replay.py", False)
             in mounts
             and sum(bool(writable) for _, _, writable in mounts) == 1,
             "cold_replay_command_drift")
        return command, mounts

    def remote(action):
        if action == "remote-install":
            return remote_install(old, v3, v2, baseline)
        return inherited_remote(action)

    old.pins, old.configure, old.remote = pins, configure, remote
    return old


def remote_install(old, v3, v2, baseline):
    need(REVIEWED_FINAL_CANDIDATE, "final_cold_replay_review_pending")
    need(not CAPTURE.exists() and not CANDIDATE.exists()
         and not WORK.exists() and not WORKER.exists(),
         "cold_replay_install_already_attempted")
    shared, alias, helper = old.dependencies()
    original = baseline_gate(v3, baseline, shared, alias, helper)
    parent = v3.source_manifest(v2, helper)
    expected = derive_candidate(parent)
    cold_candidate_gate(v3, helper, expected)
    need(sha(V3_WORKER) == V3_WORKER_SHA
         and v3.inventory(v3.CANDIDATE) == parent,
         "baseline_source_or_worker_drift")
    v3.copy_private(V3_WORKER, WORKER, 0o400)
    CAPTURE.mkdir(mode=0o700)
    for name in ("prepersist-001.json", "prepersist-001.sqlite"):
        v3.copy_private(v3.CAPTURE / name, CAPTURE / name, 0o600)
    shutil.copytree(COLD_CANDIDATE, CANDIDATE, symlinks=False)
    for path in CANDIDATE.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o400)
    os.chmod(CANDIDATE, 0o700)
    WORK.mkdir(mode=0o700)
    receipt = old.pins(shared, alias, helper)
    need(receipt["baseline_result_sha256"] == BASELINE_RESULT_SHA
         and receipt["phase1_sha256"] == NEW_PHASE1_SHA
         and original["phase1_sha256"] == PARENT_PHASE1_SHA,
         "cold_replay_receipt_identity_drift")
    helper.put_json(ROOT / "install.json", receipt)
    return {"status": "installed_not_launched", **receipt}


def install():
    need(REVIEWED_FINAL_CANDIDATE, "final_cold_replay_review_pending")
    v3 = import_v3(Path(__file__).with_name(V3_HOST.name))
    local = Path(__file__)
    body = local.read_bytes()
    config = {"root": str(ROOT), "name": SELF.name,
              "size": len(body), "sha": hashlib.sha256(body).hexdigest()}
    code = """import hashlib,json,os,pathlib,sys
root=pathlib.Path(C['root'])
assert os.geteuid()==1000 and not root.exists()
raw=sys.stdin.buffer.read(C['size'])
assert len(raw)==C['size'] and hashlib.sha256(raw).hexdigest()==C['sha']
assert sys.stdin.buffer.read(1)==b''
root.mkdir(mode=0o700)
fd=os.open(root/C['name'],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
with os.fdopen(fd,'wb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps({'status':'uploaded'}))
"""
    script = "import json\nC=json.loads(" + repr(json.dumps(config)) + ")\n" + code
    uploaded = v3.ssh_json("python3 -I -B -c " + shlex.quote(script), body)
    if uploaded.get("status") != "uploaded":
        return uploaded
    return v3.ssh_json("python3 -I -B " + shlex.quote(str(SELF))
                       + " remote-install")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "launch", "status",
                                           "remote-install", "remote-launch",
                                           "remote-status", "supervise"))
    action = parser.parse_args().action
    try:
        if action not in ("status", "remote-status"):
            need(REVIEWED_FINAL_CANDIDATE, "final_cold_replay_review_pending")
        if action == "install":
            result = install()
        elif action in ("launch", "status"):
            v3 = import_v3(Path(__file__).with_name(V3_HOST.name))
            result = v3.ssh_json("python3 -I -B " + shlex.quote(str(SELF))
                                 + " remote-" + action)
        else:
            result = controller().remote(action)
    except BaseException:
        result = {"status": "operation_failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
