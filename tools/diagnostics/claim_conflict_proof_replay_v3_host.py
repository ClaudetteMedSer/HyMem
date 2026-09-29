"""Fresh network-none v64 proof replay with the audited physical-table digest.

The failed v1/v2 and successful audit stages remain immutable. This adapter
reuses the pinned v2 one-shot controller, copies only its already-pinned source
and capture into a new private stage, and requires the audit receipt first.
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
import subprocess

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "proof-replay-v3"
SELF = ROOT / "claim_conflict_proof_replay_v3_host.py"
WORKER = ROOT / "claim_conflict_proof_replay_v3.py"
WORKER_SHA = "173e9cc13c7af1b6346100892dfb24c9cd84d1a36f69291db90dc6296e360554"
WORK = ROOT / "work"
CAPTURE = ROOT / "capture"
CANDIDATE = ROOT / "candidate"
OVERRIDES = ROOT / "overrides"
PROOF_V2_ROOT = BASE / "proof-replay-v2"
PROOF_V2_HOST = PROOF_V2_ROOT / "claim_conflict_proof_replay_v2_host.py"
PROOF_V2_HOST_SHA = "e16d27c3cd989b2664a74de19bb524ae685757a9f8fc19e4452e3b3c1d5b6bce"
PROOF_V2_WORKER = PROOF_V2_ROOT / "claim_conflict_proof_replay_v2.py"
PROOF_V2_WORKER_SHA = "8de6f54cc98abaaf5ac7f328ce2b4e4c5547128e580f3f0ecd82f8d6d353ec82"
AUDIT_ROOT = BASE / "proof-audit-v1"
AUDIT_HOST = AUDIT_ROOT / "claim_conflict_proof_drift_host.py"
AUDIT_HOST_SHA = "85db82d2ef1fe508bfa2cc80ad8a8ddbe5692b99237871c8befb5dade8182153"
DRIFT_WORKER = AUDIT_ROOT / "claim_conflict_proof_drift.py"
DRIFT_WORKER_SHA = "52ed6cf0543933249fabc41cbbd99770ba10d3e4e669dc4e97fc121b85aaf08f"
AUDIT_METADATA_SHA = "50b3bc284de59422573bcb635231d9365dbfe8d8eef5366ac864aee783ca5402"
PHASE1_SHA = "1037bf6d62c3981add3702f96d2ce5bd79d8ebc831f15b30f95bb3724cfd7217"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite")


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def import_pinned(path, expected, name):
    need(isinstance(expected, str) and HEX64.fullmatch(expected)
         and path.is_file() and not path.is_symlink() and sha(path) == expected,
         "controller_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    v2 = import_pinned(PROOF_V2_HOST, PROOF_V2_HOST_SHA, "v3_proof_v2")
    audit = import_pinned(AUDIT_HOST, AUDIT_HOST_SHA, "v3_drift_audit")
    return v2, audit, v2.controller(), audit.dependencies()[2]


def audit_gate(v2, audit, helper):
    shared, proof, _ = audit.dependencies()
    receipt = audit.installed(shared, proof, helper)
    result = helper.read_json(AUDIT_ROOT / "result.json")
    need(result.get("status") == "completed"
         and result.get("networked_runs_started") == 0,
         "proof_audit_not_completed")
    stage = result.get("stages", {}).get("audit", {})
    cid = stage.get("container_id")
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "audit_container_missing")
    state = audit.inspect(helper, cid, audit.configure(helper, receipt["snapshot_sha256"])[1],
                          receipt["snapshot_sha256"])
    need(state["status"] == "exited" and state["pid"] == 0
         and state["exit_code"] == 0 and state["oom_killed"] is False,
         "audit_not_terminal_clean")
    metadata = audit.project(stage.get("metadata"))
    need(metadata["status"] == "completed"
         and metadata["snapshot_sha256"] == receipt["snapshot_sha256"]
         and metadata["phase1_sha256"] == PHASE1_SHA
         and metadata["source_unchanged"] is True
         and metadata["exact"]["schema_before"] == 63
         and metadata["exact"]["schema_after"] == 64
         and metadata["exact"]["proof_nonnull"] == 0
         and metadata["exact"]["digest_before"] != metadata["exact"]["digest_after"]
         and metadata["instrumented"]["digest_before"]
         != metadata["instrumented"]["digest_after"]
         and metadata["instrumented"]["row_comparison"]["changed"] == 0
         and metadata["instrumented"]["row_comparison"]["ordered_equal"] is True
         and metadata["instrumented"]["row_comparison"]["unordered_equal"] is True,
         "audit_source_or_rows_not_clean")
    helper.regular(AUDIT_ROOT / "work/drift-metadata.json", mode=0o600)
    need(metadata["metadata_sha256"] == AUDIT_METADATA_SHA
         and sha(AUDIT_ROOT / "work/drift-metadata.json") == AUDIT_METADATA_SHA,
         "audit_private_metadata_pin_drift")
    need(receipt["candidate_sha256"] == hashlib.sha256(json.dumps(
        source_manifest(v2, helper), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest(), "audit_candidate_identity_drift")
    return sha(AUDIT_ROOT / "result.json")


def source_manifest(v2, helper):
    shared, _, _ = v2.controller().dependencies()
    expected = shared.baseline_inventory(helper)
    expected.update(shared.OVERRIDE_SHAS)
    expected.update(v2.reviewed_overrides())
    need(len(expected) == 481 and expected.get("hymem/dreaming/phase1.py") == PHASE1_SHA,
         "v64_manifest_invalid")
    return expected


def inventory(root):
    need(root.is_dir() and not root.is_symlink(), "source_tree_missing")
    actual = {}
    for path in root.rglob("*"):
        need(not path.is_symlink(), "source_symlink")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = sha(path)
    return actual


def audit_extra_pins(v2, audit, helper):
    need(isinstance(WORKER_SHA, str) and HEX64.fullmatch(WORKER_SHA),
         "v3_worker_unreviewed")
    audit_sha = audit_gate(v2, audit, helper)
    helper.regular(PROOF_V2_WORKER, mode=0o400)
    helper.regular(DRIFT_WORKER, mode=0o400)
    need(sha(PROOF_V2_WORKER) == PROOF_V2_WORKER_SHA
         and sha(DRIFT_WORKER) == DRIFT_WORKER_SHA,
         "v3_dependency_worker_pin_drift")
    expected = source_manifest(v2, helper)
    need(inventory(PROOF_V2_ROOT / "candidate") == expected
         and inventory(CANDIDATE) == expected,
         "v3_candidate_not_identical_to_v2")
    return {"audit_result_sha256": audit_sha,
            "audit_install_sha256": sha(AUDIT_ROOT / "install.json"),
            "proof_v2_install_sha256": sha(PROOF_V2_ROOT / "install.json"),
            "candidate_sha256": hashlib.sha256(json.dumps(
                expected, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "proof_v2_worker_sha256": PROOF_V2_WORKER_SHA,
            "drift_worker_sha256": DRIFT_WORKER_SHA}


def controller(path=PROOF_V2_HOST, *, v1_path=None, audit_path=AUDIT_HOST):
    v2 = import_pinned(path, PROOF_V2_HOST_SHA, "v3_pinned_proof_v2")
    audit = import_pinned(audit_path, AUDIT_HOST_SHA, "v3_pinned_audit")
    old = v2.controller(v1_path) if v1_path is not None else v2.controller()
    old.ROOT = ROOT
    old.SELF = SELF
    old.WORKER = WORKER
    old.WORKER_SHA = WORKER_SHA
    old.WORK = WORK
    old.CAPTURE = CAPTURE
    old.CANDIDATE = CANDIDATE
    old.OVERRIDES = OVERRIDES
    old.OVERRIDE_SHAS = v2.reviewed_overrides()
    old.reviewed_overrides = v2.reviewed_overrides
    original_pins = old.pins
    original_configure = old.configure
    original_remote = old.remote

    def pins(shared, alias, helper):
        receipt = original_pins(shared, alias, helper)
        return {**receipt, **audit_extra_pins(v2, audit, helper)}

    def configure(helper):
        command, mounts = original_configure(helper)
        command[command.index("--name") + 1] = "hymem-proof-replay-v3"
        extras = [
            (str(PROOF_V2_WORKER), "/diag/claim_conflict_proof_replay_v2.py", False),
            (str(DRIFT_WORKER), "/diag/claim_conflict_proof_drift.py", False),
        ]
        mounts.extend(extras)
        position = command.index("--workdir")
        for source, destination, _ in extras:
            command[position:position] = [
                "--mount", "type=bind,src=" + source + ",dst=" + destination + ",readonly"]
            position += 2
        return command, mounts

    def remote(action):
        if action == "remote-install":
            shared, alias, helper = old.dependencies()
            return remote_install(old, v2, audit, shared, alias, helper)
        return original_remote(action)

    old.pins = pins
    old.configure = configure
    old.remote = remote
    return old


def copy_private(source, target, mode):
    need(source.is_file() and not source.is_symlink() and not target.exists(),
         "private_copy_input_invalid")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with source.open("rb") as origin, os.fdopen(fd, "wb") as output:
        shutil.copyfileobj(origin, output, length=1048576)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(target, mode)
    need(sha(target) == sha(source), "private_copy_drift")


def remote_install(old, v2, audit, shared, alias, helper):
    need(not CAPTURE.exists() and not CANDIDATE.exists() and not WORK.exists()
         and not OVERRIDES.exists(), "install_already_attempted")
    need(isinstance(WORKER_SHA, str) and HEX64.fullmatch(WORKER_SHA),
         "v3_worker_unreviewed")
    audit_gate(v2, audit, helper)
    v2_proof = v2.controller()
    v2_proof.installed(shared, alias, helper)
    expected = source_manifest(v2, helper)
    need(inventory(PROOF_V2_ROOT / "candidate") == expected,
         "v2_candidate_pin_drift")
    helper.regular(WORKER, mode=0o400)
    need(sha(WORKER) == WORKER_SHA, "v3_worker_upload_pin_drift")
    CAPTURE.mkdir(mode=0o700)
    for name in ("prepersist-001.json", "prepersist-001.sqlite"):
        copy_private(PROOF_V2_ROOT / "capture" / name, CAPTURE / name, 0o600)
    OVERRIDES.mkdir(mode=0o700)
    for relative, expected_sha in v2.reviewed_overrides().items():
        source = PROOF_V2_ROOT / "overrides" / relative
        target = OVERRIDES / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        copy_private(source, target, 0o400)
        need(sha(target) == expected_sha, "v3_override_copy_drift")
    shutil.copytree(PROOF_V2_ROOT / "candidate", CANDIDATE, symlinks=False)
    for path in CANDIDATE.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o400)
    os.chmod(CANDIDATE, 0o700)
    WORK.mkdir(mode=0o700)
    receipt = old.pins(shared, alias, helper)
    helper.put_json(ROOT / "install.json", receipt)
    return {"status": "installed_not_launched", **receipt}


def ssh_json(command, data=b""):
    try:
        completed = subprocess.run([*SSH, command], input=data,
                                   capture_output=True, timeout=180)
        need(completed.returncode == 0 and 0 < len(completed.stdout) <= 16384,
             "remote_operation_failed")
        value = json.loads(completed.stdout)
        need(isinstance(value, dict), "remote_output_invalid")
        return value
    except (subprocess.TimeoutExpired, ValueError, RuntimeError):
        return {"status": "unknown_inspect_before_retry"}


def install():
    need(isinstance(WORKER_SHA, str) and HEX64.fullmatch(WORKER_SHA),
         "v3_worker_unreviewed")
    local = Path(__file__)
    body = local.with_name("claim_conflict_proof_replay_v3.py").read_bytes()
    need(hashlib.sha256(body).hexdigest() == WORKER_SHA,
         "local_v3_worker_pin_drift")
    pieces = {SELF.name: local.read_bytes(), WORKER.name: body}
    config = {"root": str(ROOT), "files": {
        name: {"size": len(raw), "sha": hashlib.sha256(raw).hexdigest()}
        for name, raw in pieces.items()}}
    code = """import hashlib,json,os,pathlib,sys
root=pathlib.Path(C['root'])
assert os.geteuid()==1000 and not root.exists()
root.mkdir(mode=0o700)
for name,item in C['files'].items():
 raw=sys.stdin.buffer.read(item['size'])
 assert len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha']
 fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sys.stdin.buffer.read(1)==b''
print(json.dumps({'status':'uploaded'}))
"""
    script = "import json\nC=json.loads(" + repr(json.dumps(config)) + ")\n" + code
    uploaded = ssh_json("python3 -I -B -c " + shlex.quote(script),
                        b"".join(pieces.values()))
    if uploaded.get("status") != "uploaded":
        return uploaded
    return ssh_json("python3 -I -B " + shlex.quote(str(SELF)) + " remote-install")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "launch", "status",
                                           "remote-install", "remote-launch",
                                           "remote-status", "supervise"))
    action = parser.parse_args().action
    try:
        if action == "install":
            result = install()
        elif action in ("launch", "status"):
            result = ssh_json("python3 -I -B " + shlex.quote(str(SELF))
                              + " remote-" + action)
        else:
            result = controller().remote(action)
    except BaseException:
        result = {"status": "operation_failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
