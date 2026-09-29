"""One-shot network-free audit of the retained proof-v2 upgrade drift.

The failed proof-v2 stage is immutable. This stage mounts only its sealed
candidate and captured snapshot read-only; the sole writable mount is /work.
Install and launch are explicit, separate operations.
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
import stat
import subprocess
import sys

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "proof-audit-v1"
SELF = ROOT / "claim_conflict_proof_drift_host.py"
WORKER = ROOT / "claim_conflict_proof_drift.py"
WORKER_SHA = "52ed6cf0543933249fabc41cbbd99770ba10d3e4e669dc4e97fc121b85aaf08f"
WORK = ROOT / "work"
PROOF_ROOT = BASE / "proof-replay-v2"
PROOF_HOST = PROOF_ROOT / "claim_conflict_proof_replay_v2_host.py"
PROOF_HOST_SHA = "e16d27c3cd989b2664a74de19bb524ae685757a9f8fc19e4452e3b3c1d5b6bce"
SHARED_HOST = BASE / "shared-embedding-dream-v1/claim_conflict_shared_embedding_host.py"
SHARED_HOST_SHA = "ca95bc8d06cc1cc9f84e80dcb91f13cdc333c681efa4e9aca5ae457d48cce1f0"
CANDIDATE = PROOF_ROOT / "candidate"
SNAPSHOT = PROOF_ROOT / "capture/prepersist-001.sqlite"
AUDIT_V1 = BASE / "proof-replay-v1/claim_conflict_proof_replay.py"
AUDIT_V1_SHA = "014e045de6a5591eb87aabf0760247a69cd4d5a75f5bb7462e2d404caff7a9de"
OLD_REPLAY = PROOF_ROOT / "claim_conflict_instrumented_replay.py"
OLD_REPLAY_SHA = "7b94a69f2a152f1a818ab7b5833b8d619734b3f7023de2dca89b4190f9013085"
PHASE1_SHA = "1037bf6d62c3981add3702f96d2ce5bd79d8ebc831f15b30f95bb3724cfd7217"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PUBLIC_TABLES = frozenset({
    "sessions", "messages", "chunks", "chunk_message_sources", "message_retention_coverage",
    "entity_aliases", "knowledge_graph", "kg_evidence", "kg_claim_observations",
    "kg_claim_extraction_outcomes", "processed_chunks", "phase1_generations",
    "phase1_auxiliary_outcomes", "schema_meta",
} | {"vec_" + domain + suffix
     for domain in ("chunks", "messages", "edges", "episodes", "facts")
     for suffix in ("", "_chunks", "_rowids", "_info", "_vector_chunks00")})
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


def load_pinned(path, expected, name):
    need(isinstance(expected, str) and HEX64.fullmatch(expected)
         and path.is_file() and not path.is_symlink() and sha(path) == expected,
         "dependency_pin_drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    shared = load_pinned(SHARED_HOST, SHARED_HOST_SHA, "drift_shared")
    adapter = load_pinned(PROOF_HOST, PROOF_HOST_SHA, "drift_proof_v2")
    return shared, adapter.controller(), shared.helper()


def source_gate(proof, helper):
    shared, alias, _ = proof.dependencies()
    receipt = proof.installed(shared, alias, helper)
    need(receipt.get("source_files") == 481
         and receipt.get("phase1_sha256") == PHASE1_SHA
         and receipt.get("snapshot_sha256") == sha(SNAPSHOT),
         "proof_source_pin_drift")
    result = helper.read_json(PROOF_ROOT / "result.json")
    need(result.get("status") == "failed"
         and result.get("networked_runs_started") == 0,
         "proof_v2_not_failed_terminal")
    stage = result.get("stages", {}).get("replay", {})
    cid = stage.get("container_id")
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "proof_v2_container_missing")
    state = proof.inspect(helper, cid, proof.configure(helper)[1])
    need(state["status"] == "exited" and state["pid"] == 0
         and state["exit_code"] == 1 and state["oom_killed"] is False,
         "proof_v2_not_failed_terminal")
    metadata = proof.project(stage.get("metadata"))
    need(metadata == {"status": "error", "reason_code": "proof_replay_failed",
                     "error_type": "RuntimeError", "failure_captured": True},
         "proof_v2_failure_not_target_drift")
    return receipt, sha(PROOF_ROOT / "result.json")


def pins(shared, proof, helper):
    need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
         and stat.S_IMODE(ROOT.stat().st_mode) == 0o700
         and WORK.is_dir() and not WORK.is_symlink()
         and stat.S_IMODE(WORK.stat().st_mode) == 0o700,
         "audit_stage_not_private")
    for path, mode in ((SELF, 0o400), (WORKER, 0o400),
                       (SNAPSHOT, 0o600), (AUDIT_V1, 0o400),
                       (OLD_REPLAY, 0o400)):
        helper.regular(path, mode=mode)
    need(isinstance(WORKER_SHA, str) and HEX64.fullmatch(WORKER_SHA)
         and sha(WORKER) == WORKER_SHA and sha(AUDIT_V1) == AUDIT_V1_SHA
         and sha(OLD_REPLAY) == OLD_REPLAY_SHA,
         "audit_worker_pin_drift")
    receipt, proof_result_sha = source_gate(proof, helper)
    expected = shared.baseline_inventory(helper)
    expected.update(shared.OVERRIDE_SHAS)
    expected.update(proof.reviewed_overrides())
    need(CANDIDATE.is_dir() and not CANDIDATE.is_symlink()
         and len(expected) == 481, "audit_candidate_pin_drift")
    actual = {}
    for path in CANDIDATE.rglob("*"):
        need(not path.is_symlink(), "audit_candidate_symlink")
        if path.is_file():
            actual[path.relative_to(CANDIDATE).as_posix()] = sha(path)
    need(actual == expected, "audit_candidate_inventory_drift")
    manifest_sha = hashlib.sha256(json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return {"host_sha256": sha(SELF), "worker_sha256": WORKER_SHA,
            "source_files": 481, "snapshot_sha256": receipt["snapshot_sha256"],
            "phase1_sha256": PHASE1_SHA,
            "proof_install_sha256": sha(PROOF_ROOT / "install.json"),
            "proof_result_sha256": proof_result_sha,
            "candidate_sha256": manifest_sha,
            "audit_v1_sha256": AUDIT_V1_SHA,
            "old_replay_sha256": OLD_REPLAY_SHA}


def installed(shared, proof, helper):
    receipt = helper.read_json(ROOT / "install.json")
    need(receipt == pins(shared, proof, helper), "audit_install_pin_drift")
    return receipt


def configure(helper, snapshot_sha):
    need(isinstance(snapshot_sha, str) and HEX64.fullmatch(snapshot_sha),
         "snapshot_unpinned")
    mounts = [(str(CANDIDATE), "/candidate", False),
              (str(SNAPSHOT), "/capture/source.sqlite", False),
              (str(WORKER), "/diag/claim_conflict_proof_drift.py", False),
              (str(AUDIT_V1), "/diag/claim_conflict_proof_replay_v1.py", False),
              (str(OLD_REPLAY), "/diag/claim_conflict_instrumented_replay.py", False),
              (str(WORK), "/work", True),
              (str(helper.RUNTIME), "/home/node/hymem-env", False)]
    command = ["docker", "create", "--name", "hymem-proof-audit-v1",
               "--pull", "never", "--init", "--network", "none",
               "--user", "1000:1000", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "128",
               "--memory", "2g", "--cpus", "2",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
               "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
               "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for src, dst, rw in mounts:
        command += ["--mount", "type=bind,src=" + src + ",dst=" + dst
                    + ("" if rw else ",readonly")]
    command += ["--workdir", "/candidate", "--entrypoint",
                "/home/node/hymem-env/bin/python3", helper.IMAGE,
                "-I", "-B", "/diag/claim_conflict_proof_drift.py",
                "--snapshot-sha256", snapshot_sha,
                "--phase1-sha256", PHASE1_SHA]
    return command, mounts


def inspect(helper, cid, mounts, snapshot_sha):
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "container_id_invalid")
    item = json.loads(helper.run(["docker", "inspect", cid], 30, "inspect"))[0]
    config, host, state = item["Config"], item["HostConfig"], item["State"]
    expected = configure(helper, snapshot_sha)[0]
    image_index = expected.index(helper.IMAGE)
    actual_mounts = {entry["Destination"]: (entry["Source"], entry["RW"], entry["Type"])
                     for entry in item["Mounts"]}
    need(actual_mounts == {dst: (src, rw, "bind") for src, dst, rw in mounts}
         and item["Image"] == helper.IMAGE and config["Image"] == helper.IMAGE
         and config["User"] == "1000:1000"
         and config["Entrypoint"] == ["/home/node/hymem-env/bin/python3"]
         and config["WorkingDir"] == "/candidate"
         and config["Cmd"] == expected[image_index + 1:]
         and host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True
         and host["Privileged"] is False and host["CapDrop"] == ["ALL"]
         and host["SecurityOpt"] == ["no-new-privileges"] and host["Init"] is True
         and host["Memory"] == 2147483648 and host["NanoCpus"] == 2000000000
         and host["PidsLimit"] == 128
         and host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,size=64m"}
         and host["RestartPolicy"]["Name"] == "no"
         and not any(key.startswith(("DEEPSEEK_", "OPENAI_", "HYMEM_"))
                     for key in config["Env"]), "audit_container_configuration_drift")
    return {"container_id": cid, "status": state["Status"],
            "exit_code": state["ExitCode"], "oom_killed": state["OOMKilled"],
            "pid": state["Pid"], "configuration_verified": True}


def project(raw):
    need(isinstance(raw, dict) and raw.get("status") in ("completed", "error"),
         "audit_worker_output_invalid")
    if raw["status"] == "error":
        need(raw.get("reason_code") == "drift_audit_failed"
             and type(raw.get("failure_captured")) is bool,
             "audit_error_shape_invalid")
        return {"status": "error", "reason_code": "drift_audit_failed",
                "failure_captured": raw["failure_captured"]}
    result = {"status": "completed"}
    for name in ("snapshot_sha256", "phase1_sha256", "metadata_sha256"):
        value = raw.get(name)
        need(isinstance(value, str) and HEX64.fullmatch(value), "audit_hash_invalid")
        result[name] = value
    need(raw.get("source_unchanged") is True, "audit_source_changed")
    result["source_unchanged"] = True
    result["exact"] = project_arm(raw.get("exact"), instrumented=False)
    result["instrumented"] = project_arm(raw.get("instrumented"), instrumented=True)
    return result


def project_table_id(value):
    need(isinstance(value, str) and (value in PUBLIC_TABLES or
         (value.startswith("sha256:") and HEX64.fullmatch(value[7:]))),
         "audit_table_id_invalid")
    return value


def project_table_record(raw):
    if raw is None:
        return None
    need(isinstance(raw, (list, tuple)) and len(raw) == 4
         and raw[0] in ("table", "shadow", "virtual", "view")
         and all(type(value) is int and 0 <= value <= 1000000 for value in raw[1:]),
         "audit_table_record_invalid")
    return [raw[0], *raw[1:]]


def project_arm(raw, *, instrumented):
    need(isinstance(raw, dict), "audit_arm_invalid")
    arm = {}
    for name, value in (("schema_before", 63), ("schema_after", 64),
                        ("proof_nonnull", 0)):
        need(type(raw.get(name)) is int and raw[name] == value,
             "audit_schema_proof_invalid")
        arm[name] = value
    for name in ("digest_before", "digest_after", "digest_after_repeat",
                 "inventory_before_sha256", "inventory_after_sha256"):
        value = raw.get(name)
        need(isinstance(value, str) and HEX64.fullmatch(value),
             "audit_arm_hash_invalid")
        arm[name] = value
    changes = raw.get("table_changes")
    need(isinstance(changes, list) and len(changes) <= 128,
         "audit_table_changes_invalid")
    arm["table_changes"] = []
    for item in changes:
        need(isinstance(item, dict) and item.get("schema") in ("main", "temp"),
             "audit_table_change_invalid")
        arm["table_changes"].append({
            "schema": item["schema"], "table_id": project_table_id(item.get("table_id")),
            "before": project_table_record(item.get("before")),
            "after": project_table_record(item.get("after")),
        })
    if instrumented:
        rows = raw.get("row_comparison")
        need(isinstance(rows, dict), "audit_row_comparison_missing")
        counts = {}
        for name in ("tables", "changed"):
            value = rows.get(name)
            need(type(value) is int and 0 <= value <= 2048,
                 "audit_row_count_invalid")
            counts[name] = value
        for name in ("ordered_equal", "unordered_equal"):
            need(type(rows.get(name)) is bool, "audit_row_equality_invalid")
            counts[name] = rows[name]
        ids = rows.get("changed_table_ids")
        need(isinstance(ids, list) and len(ids) <= 128
             and len(ids) == counts["changed"], "audit_changed_tables_invalid")
        counts["changed_table_ids"] = [project_table_id(value) for value in ids]
        arm["row_comparison"] = counts
    else:
        need("row_comparison" not in raw, "audit_exact_arm_not_exact")
    return arm


def supervise(shared, proof, helper):
    result = {"status": "failed", "stages": {}, "networked_runs_started": 0}
    try:
        receipt = installed(shared, proof, helper)
        helper.put_json(ROOT / "supervisor-intent.json",
                        {"networked_runs_allowed": 0, "max_containers": 1})
        command, mounts = configure(helper, receipt["snapshot_sha256"])
        helper.put_json(ROOT / "create-intent.json", {"host_sha256": sha(SELF)})
        cid = helper.run(command, 60, "create").decode().strip()
        helper.put_json(ROOT / "container.json", {"container_id": cid})
        need(inspect(helper, cid, mounts, receipt["snapshot_sha256"])["status"] == "created",
             "container_not_created")
        helper.put_json(ROOT / "start-intent.json", {"container_id": cid})
        try:
            need(helper.run(["docker", "start", cid], 60, "start").decode().strip()
                 == cid, "start_identity")
            raw = helper.run(["docker", "wait", cid], 900, "wait")
            need(re.fullmatch(rb"[0-9]{1,3}\n?", raw), "wait_shape")
        except BaseException:
            subprocess.run(["docker", "stop", "--time", "10", cid],
                           capture_output=True, timeout=30)
            need(inspect(helper, cid, mounts, receipt["snapshot_sha256"])["pid"] == 0,
                 "cleanup_unverified")
            raise
        state = inspect(helper, cid, mounts, receipt["snapshot_sha256"])
        need(state["status"] == "exited" and state["pid"] == 0
             and state["exit_code"] == int(raw) and state["oom_killed"] is False,
             "audit_terminal_state_invalid")
        metadata = project(json.loads(helper.run(["docker", "logs", cid], 30, "logs")))
        result["stages"]["audit"] = {**state, "metadata": metadata}
        if metadata["status"] == "completed":
            helper.regular(WORK / "drift-metadata.json", mode=0o600)
            need(sha(WORK / "drift-metadata.json") == metadata["metadata_sha256"],
                 "audit_private_metadata_drift")
            need(metadata["snapshot_sha256"] == receipt["snapshot_sha256"]
                 and metadata["phase1_sha256"] == PHASE1_SHA,
                 "audit_worker_source_binding_drift")
        need(state["exit_code"] == 0 and metadata["status"] == "completed"
             and metadata["source_unchanged"] is True,
             "audit_worker_not_clean")
        installed(shared, proof, helper)
        result["status"] = "completed"
    except BaseException as exc:
        result["error_type"] = (type(exc).__name__ if type(exc).__name__ in
                                ("RuntimeError", "ValueError", "OSError") else "Exception")
    helper.put_json(ROOT / "result.json", result)


def remote_install(shared, proof, helper):
    need(not WORK.exists() and not (ROOT / "install.json").exists(),
         "install_already_attempted")
    source_gate(proof, helper)
    helper.regular(WORKER, mode=0o400)
    need(isinstance(WORKER_SHA, str) and HEX64.fullmatch(WORKER_SHA)
         and sha(WORKER) == WORKER_SHA, "audit_worker_unreviewed")
    WORK.mkdir(mode=0o700)
    receipt = pins(shared, proof, helper)
    helper.put_json(ROOT / "install.json", receipt)
    return {"status": "installed_not_launched", **receipt}


def remote(action):
    shared, proof, helper = dependencies()
    if action == "remote-install":
        return remote_install(shared, proof, helper)
    if action == "remote-status":
        if (ROOT / "result.json").exists():
            return helper.read_json(ROOT / "result.json")
        return {"status": ("running_or_requires_inspection"
                           if (ROOT / "launch-intent.json").exists()
                           else "installed_not_launched")}
    if action == "supervise":
        supervise(shared, proof, helper)
        return {"status": "supervisor_finished"}
    installed(shared, proof, helper)
    need(not (ROOT / "launch-intent.json").exists(), "launch_already_attempted")
    helper.put_json(ROOT / "launch-intent.json", {"host_sha256": sha(SELF)})
    child = subprocess.Popen([sys.executable, "-I", "-B", str(SELF), "supervise"],
                             cwd=ROOT, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True)
    helper.put_json(ROOT / "launch.json", {"pid": child.pid})
    return {"status": "detached_supervisor_started", "pid": child.pid}


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
         "audit_worker_unreviewed")
    local = Path(__file__)
    body = local.with_name("claim_conflict_proof_drift.py").read_bytes()
    need(hashlib.sha256(body).hexdigest() == WORKER_SHA,
         "local_audit_worker_pin_drift")
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
            result = remote(action)
    except BaseException:
        result = {"status": "operation_failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
