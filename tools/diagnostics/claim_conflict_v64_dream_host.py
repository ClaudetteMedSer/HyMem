"""Future one-shot private v64 targeted dream; never auto-launches.

Code-only install and launch are explicit. Both are fail-closed until the
network-free cold replay and full-suite receipts are sealed and successful.
The only live traffic admitted later is the pinned targeted DeepSeek/internal
embedding diagnostic, under the inherited 64/192/512/1800 caps.
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
import subprocess

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "cold-replay-dream-v1"
SELF = ROOT / "claim_conflict_v64_dream_host.py"
WORKER = ROOT / "claim_conflict_v64_dream.py"
WORKER_SHA = "ed9d170152d643e008de1916d3b4dd7ed1e24365a80264cf59931891bdcd022f"
OLD_HOST = ROOT / "claim_conflict_alias_dream_host.py"
OLD_HOST_SHA = "a0f2c7c962185830e788aca6ddfe83bbda128cc91e6456bd627ff45f35dda671"
OLD_WORKER = ROOT / "claim_conflict_instrumented_dream.py"
OLD_WORKER_SHA = "9d6f03ef88efd01c40fd94affc9b31dd7b6b6e879066ce784f08e74600f0b863"
WORK = ROOT / "work"
CANDIDATE = ROOT / "candidate"
REFERENCE = ROOT / "reference.sqlite"
REFERENCE_ORIGINAL = BASE / "capture-next-v1/reference.sqlite"
REFERENCE_SHA = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
PROOF_ROOT = BASE / "cold-replay-proof-v1"
PROOF_SOURCE = PROOF_ROOT / "candidate"
PROOF_HOST = PROOF_ROOT / "claim_conflict_cold_replay_host.py"
PROOF_HOST_SHA = "7afdb9067ca1483a6f17d42431c737ab78f49a37bd2543bdb52c06014a7a3586"
PROOF_RESULT_SHA = "7dd4482487f4bb99700a7fc75a27b59dc1dae898b39cae6b18c8e71da184dcc1"
SUITE_ROOT = BASE / "cold-replay-pytest-v2"
SUITE_HOST = SUITE_ROOT / "claim_conflict_cold_pytest_v2.py"
SUITE_HOST_SHA = "25c9de04b32f2e6a095a5860d3ab0fd3a4ba328f880f6d91a894de0d2d6dd227"
SUITE_RESULT_SHA = "4f5725719b6cbdda61906874cc373cae6b9901547b9553dd06f0ab6ed6fd40ec"  # Independently revalidated final suite.
CANDIDATE_SHA = "ed889d8970c6d7827315de34342996ad55182773c238922fbbdf8510d73c43a4"
SUITE_OVERLAY_SHA = "5bb544abc0df397bf6b284a29d44de11b129d9c17d86b6740e04857764fc37e7"
BASELINE_PROOF_RESULT_SHA = "6e602bae89aee5f42be1f3f7efff87ba905a59bc134d09015ffe7a02a672c52d"
BASELINE_CANDIDATE_SHA = "cd6e7810a4c08d771c49658a5954ca71a24d9d1119bcfde3aaa4ccb86dafd694"
PROOF_WORKER_SHA = "173e9cc13c7af1b6346100892dfb24c9cd84d1a36f69291db90dc6296e360554"
PAYLOAD_TRANSFER_APPROVED = True  # User approved this bounded one-off private diagnostic; no recurring launch.
SHARED_HOST = BASE / "shared-embedding-dream-v1/claim_conflict_shared_embedding_host.py"
SHARED_HOST_SHA = "ca95bc8d06cc1cc9f84e80dcb91f13cdc333c681efa4e9aca5ae457d48cce1f0"
PHASE1_SHA = "bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac"
SOURCE_FILES = 481
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
         "dependency_not_reviewed_or_pin_drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    shared = import_pinned(SHARED_HOST, SHARED_HOST_SHA, "v64_dream_shared")
    proof_adapter = import_pinned(PROOF_HOST, PROOF_HOST_SHA, "v64_dream_proof_adapter")
    proof = proof_adapter.controller()
    return shared, proof, shared.helper()


def proof_gate(proof):
    need(isinstance(PROOF_RESULT_SHA, str) and HEX64.fullmatch(PROOF_RESULT_SHA),
         "cold_proof_result_unreviewed")
    shared, alias, helper = proof.dependencies()
    receipt = proof.installed(shared, alias, helper)
    result_path = PROOF_ROOT / "result.json"
    need(result_path.is_file() and not result_path.is_symlink()
         and sha(result_path) == PROOF_RESULT_SHA,
         "cold_proof_result_pin_drift")
    result = helper.read_json(result_path)
    need(result.get("status") == "completed"
         and result.get("networked_runs_started") == 0,
         "proof_replay_not_completed")
    stage = result.get("stages", {}).get("replay", {})
    cid = stage.get("container_id")
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "proof_container_id_invalid")
    state = proof.inspect(helper, cid, proof.configure(helper)[1])
    need(state["status"] == "exited" and state["pid"] == 0
         and state["exit_code"] == 0 and not state["oom_killed"],
         "proof_container_not_clean")
    proof.verdict(proof.project(stage.get("metadata")), receipt)
    need(receipt.get("source_files") == SOURCE_FILES
         and receipt.get("phase1_sha256") == PHASE1_SHA
         and receipt.get("host_sha256") == PROOF_HOST_SHA
         and receipt.get("candidate_sha256") == CANDIDATE_SHA
         and receipt.get("worker_sha256") == PROOF_WORKER_SHA
         and receipt.get("baseline_candidate_sha256") == BASELINE_CANDIDATE_SHA
         and receipt.get("baseline_result_sha256") == BASELINE_PROOF_RESULT_SHA
         and receipt.get("baseline_proof_only") is True
         and receipt.get("new_candidate_replay_verified") is False,
         "proof_candidate_identity_drift")
    return PROOF_RESULT_SHA


def suite_gate():
    need(isinstance(SUITE_RESULT_SHA, str) and HEX64.fullmatch(SUITE_RESULT_SHA),
         "full_suite_result_unreviewed")
    suite_adapter = import_pinned(SUITE_HOST, SUITE_HOST_SHA, "v64_dream_suite_adapter")
    suite = suite_adapter.configure_base()
    proof, shared, helper = suite.dependencies()
    expected = suite.pins(proof, shared, helper)
    receipt = helper.read_json(SUITE_ROOT / "install.json")
    need(receipt == expected, "suite_install_pin_drift")
    need(expected.get("host_sha256") == SUITE_HOST_SHA
         and expected.get("candidate_sha256") == CANDIDATE_SHA
         and expected.get("overlay_sha256") == SUITE_OVERLAY_SHA
         and expected.get("proof_result_sha256") == BASELINE_PROOF_RESULT_SHA
         and expected.get("parent_candidate_sha256") == BASELINE_CANDIDATE_SHA
         and expected.get("candidate_phase1_sha256") == PHASE1_SHA
         and expected.get("baseline_proof_only") is True
         and expected.get("new_candidate_replay_verified") is False,
         "suite_source_identity_drift")
    result_path = SUITE_ROOT / "result.json"
    need(result_path.is_file() and not result_path.is_symlink()
         and sha(result_path) == SUITE_RESULT_SHA,
         "full_suite_result_pin_drift")
    result = helper.read_json(result_path)
    need(result.get("status") == "passed"
         and result.get("networked_runs_started") == 0,
         "full_suite_not_passed")
    container = result.get("container", {})
    cid = container.get("container_id")
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "suite_container_id_invalid")
    state = suite.inspect(helper, cid)
    need(state["status"] == "exited" and state["pid"] == 0
         and state["exit_code"] == 0 and not state["oom_killed"],
         "suite_container_not_clean")
    metadata = suite.project_worker(result.get("metadata"))
    need(metadata["status"] == "passed"
         and metadata["full_runs_started"] == 1
         and metadata["full"]["failed"] == 0
         and metadata["full"]["errors"] == 0
         and metadata["candidate_sha256"] == expected["candidate_sha256"],
         "suite_evidence_incomplete")
    return SUITE_RESULT_SHA


def expected_inventory(shared, proof, helper):
    base = shared.baseline_inventory(helper)
    base.update(shared.OVERRIDE_SHAS)
    base.update(proof.reviewed_overrides())
    need(len(base) == SOURCE_FILES
         and base.get("hymem/dreaming/phase1.py") == PHASE1_SHA
         and "hymem/core/migrations/064_local_claim_replay_proof.sql" in base,
         "v64_source_manifest_invalid")
    need(hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")
                        ).encode()).hexdigest() == CANDIDATE_SHA,
         "v64_candidate_manifest_pin_drift")
    return base


def inventory(root):
    need(root.is_dir() and not root.is_symlink(), "candidate_missing")
    actual = {}
    for path in root.rglob("*"):
        need(not path.is_symlink(), "candidate_symlink")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = sha(path)
    return actual


def pins(shared, proof, helper):
    need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
         and stat.S_IMODE(ROOT.stat().st_mode) == 0o700
         and WORK.is_dir() and not WORK.is_symlink()
         and stat.S_IMODE(WORK.stat().st_mode) == 0o700,
         "stage_not_private")
    for path, mode in ((SELF, 0o400), (WORKER, 0o400),
                       (OLD_HOST, 0o400), (OLD_WORKER, 0o400),
                       (REFERENCE, 0o400), (helper.RUNTIME_ENV, 0o600)):
        helper.regular(path, mode=mode)
    need(sha(WORKER) == WORKER_SHA and sha(OLD_HOST) == OLD_HOST_SHA
         and sha(OLD_WORKER) == OLD_WORKER_SHA
         and sha(REFERENCE) == REFERENCE_SHA
         and sha(REFERENCE_ORIGINAL) == REFERENCE_SHA,
         "worker_or_reference_pin_drift")
    proof_sha = proof_gate(proof)
    suite_sha = suite_gate()
    expected = expected_inventory(shared, proof, helper)
    need(inventory(PROOF_SOURCE) == expected and inventory(CANDIDATE) == expected,
         "v64_candidate_copy_drift")
    runtime = helper.read_json(helper.RUNTIME_ENV)
    need(runtime.get("HYMEM_EMBEDDING_BASE_URL") == shared.EXPECTED_ENDPOINT
         and runtime.get("HYMEM_LLM_BASE_URL", "").rstrip("/")
         == "https://api.deepseek.com"
         and runtime.get("HYMEM_LLM_MODEL") == "deepseek-flash",
         "runtime_endpoint_pin_drift")
    return {"source_files": SOURCE_FILES, "phase1_sha256": PHASE1_SHA,
            "reference_sha256": REFERENCE_SHA,
            "candidate_sha256": hashlib.sha256(json.dumps(
                expected, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "proof_result_sha256": proof_sha, "suite_result_sha256": suite_sha,
            "suite_host_sha256": SUITE_HOST_SHA,
            "runtime_env_sha256": sha(helper.RUNTIME_ENV),
            "host_sha256": sha(SELF), "worker_sha256": WORKER_SHA,
            "old_host_sha256": OLD_HOST_SHA, "old_worker_sha256": OLD_WORKER_SHA}


def installed(shared, proof, helper):
    receipt = helper.read_json(ROOT / "install.json")
    need(receipt == pins(shared, proof, helper), "installed_pin_drift")
    return receipt


def controller(path=OLD_HOST):
    old = import_pinned(path, OLD_HOST_SHA, "v64_dream_pinned_supervisor")
    old.ROOT = ROOT
    old.SELF = SELF
    old.WORKER = WORKER
    old.WORK = WORK
    old.CANDIDATE = CANDIDATE
    old.REFERENCE = REFERENCE
    old.REFERENCE_SHA = REFERENCE_SHA
    old.PHASE1_SHA = PHASE1_SHA
    old.WORKER_SHA = WORKER_SHA
    old.dependencies = dependencies
    old.pins = pins
    old.installed = installed
    original_configure = old.configure
    original_remote = old.remote

    def configure(helper, mode):
        command, mounts = original_configure(helper, mode)
        extra = (str(OLD_WORKER), "/diag/claim_conflict_instrumented_dream_v1.py", False)
        mounts.append(extra)
        position = command.index("--workdir")
        command[position:position] = [
            "--mount", "type=bind,src=" + extra[0] + ",dst=" + extra[1] + ",readonly"]
        return command, mounts

    old.configure = configure

    def remote(action):
        if action in ("remote-launch", "supervise"):
            need(PAYLOAD_TRANSFER_APPROVED is True,
                 "exact_payload_transfer_not_approved")
        return original_remote(action)

    old.remote = remote
    return old


def remote_install(shared, proof, helper):
    need(not WORK.exists() and not CANDIDATE.exists() and not REFERENCE.exists(),
         "install_already_attempted")
    proof_gate(proof)
    suite_gate()
    expected = expected_inventory(shared, proof, helper)
    need(inventory(PROOF_SOURCE) == expected, "proof_source_pin_drift")
    helper.regular(REFERENCE_ORIGINAL, mode=0o400)
    need(sha(REFERENCE_ORIGINAL) == REFERENCE_SHA, "reference_pin_drift")
    for path, expected_sha in ((WORKER, WORKER_SHA), (OLD_HOST, OLD_HOST_SHA),
                               (OLD_WORKER, OLD_WORKER_SHA)):
        helper.regular(path, mode=0o400)
        need(sha(path) == expected_sha, "upload_pin_drift")
    fd = os.open(REFERENCE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600)
    with REFERENCE_ORIGINAL.open("rb") as origin, os.fdopen(fd, "wb") as target:
        shutil.copyfileobj(origin, target, length=1048576)
        target.flush()
        os.fsync(target.fileno())
    os.chmod(REFERENCE, 0o400)
    WORK.mkdir(mode=0o700)
    shutil.copytree(PROOF_SOURCE, CANDIDATE, symlinks=False)
    for path in CANDIDATE.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o400)
    os.chmod(CANDIDATE, 0o700)
    receipt = pins(shared, proof, helper)
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
    # No upload may begin while either offline result remains unreviewed.
    need(isinstance(PROOF_RESULT_SHA, str) and HEX64.fullmatch(PROOF_RESULT_SHA),
         "cold_proof_result_unreviewed")
    need(isinstance(SUITE_HOST_SHA, str) and HEX64.fullmatch(SUITE_HOST_SHA),
         "suite_host_unreviewed")
    need(isinstance(SUITE_RESULT_SHA, str) and HEX64.fullmatch(SUITE_RESULT_SHA),
         "full_suite_result_unreviewed")
    # This action only uploads reviewed diagnostic code. Server-side install
    # still gates both offline receipts before any clone or paid launch.
    local = Path(__file__)
    pieces = {
        SELF.name: local.read_bytes(),
        WORKER.name: local.with_name("claim_conflict_v64_dream.py").read_bytes(),
        OLD_HOST.name: local.with_name("claim_conflict_alias_dream_host.py").read_bytes(),
        OLD_WORKER.name: local.with_name("claim_conflict_instrumented_dream.py").read_bytes(),
    }
    need(hashlib.sha256(pieces[WORKER.name]).hexdigest() == WORKER_SHA
         and hashlib.sha256(pieces[OLD_HOST.name]).hexdigest() == OLD_HOST_SHA
         and hashlib.sha256(pieces[OLD_WORKER.name]).hexdigest() == OLD_WORKER_SHA,
         "local_diagnostic_pin_drift")
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
            old = controller()
            if action == "remote-install":
                shared, proof, helper = dependencies()
                result = remote_install(shared, proof, helper)
            else:
                result = old.remote(action)
    except BaseException:
        result = {"status": "operation_failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
