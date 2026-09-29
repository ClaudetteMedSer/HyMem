"""One-shot network-none v61→v62 retained-response proof replay controller.

Install, launch and status are separate explicit actions. Import never contacts
the host. Every application override and diagnostic worker is hash-pinned.
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
import sys

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "proof-replay-v1"
SELF = ROOT / "claim_conflict_proof_replay_host.py"
WORKER = ROOT / "claim_conflict_proof_replay.py"
WORKER_SHA = "014e045de6a5591eb87aabf0760247a69cd4d5a75f5bb7462e2d404caff7a9de"
OLD_REPLAY = ROOT / "claim_conflict_instrumented_replay.py"
OLD_REPLAY_SHA = "7b94a69f2a152f1a818ab7b5833b8d619734b3f7023de2dca89b4190f9013085"
SHARED = BASE / "shared-embedding-dream-v1"
SHARED_HOST = SHARED / "claim_conflict_shared_embedding_host.py"
SHARED_HOST_SHA = "ca95bc8d06cc1cc9f84e80dcb91f13cdc333c681efa4e9aca5ae457d48cce1f0"
ALIAS_HOST = BASE / "alias-replay-v1/claim_conflict_alias_replay_host.py"
ALIAS_HOST_SHA = "c998a7179e89a55e15be4f76f996a416e9771456c38718c8f622a49d4ba120b2"
SOURCE = SHARED / "candidate"
ORIGINAL_CAPTURE = SHARED / "work/live/prepersist-001.json"
ORIGINAL_SNAPSHOT = SHARED / "work/live/prepersist-001.sqlite"
REFERENCE = BASE / "capture-next-v1/reference.sqlite"
REFERENCE_SHA = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
CAPTURE = ROOT / "capture"
OVERRIDES = ROOT / "overrides"
CANDIDATE = ROOT / "candidate"
WORK = ROOT / "work"
LOCAL_FROZEN = Path("/Users/attavanwestreenen/AGprojects/HyMem")
OVERRIDE_SHAS = {
    "hymem/core/db.py": "4c1afb8fc144a4300ac38a1deeb14363dc08e49fcd3151e7f687bfda65790ce0",
    "hymem/core/schema.sql": "1798120e80098b8f8b8defc7da868addd91322f98024d08c4e683d3b087b3d6d",
    "hymem/core/migrations/062_local_claim_replay_proof.sql": "45c79115922c8b06be4f9861b6d026204befdb28d90609d598f8d7f70bd097c6",
    "hymem/dreaming/phase1.py": "1037bf6d62c3981add3702f96d2ce5bd79d8ebc831f15b30f95bb3724cfd7217",
    "hymem/dreaming/evidence.py": "c921d035c4af5b6d0c5add8cb9eb89069b3dbef4447078470ae77dc0dbb36771",
    "hymem/dreaming/canonicalize.py": "73c1d656aaa471402732690712c9de86f82bea6e4a2bf6fef5677a1d1a4dcc18",
    "hymem/portability.py": "d50a7bc03dabcdf49cc8d6c55bae1ec17d0f72b4e31ce9b8204eca97d38cab70",
}
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite")
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


def import_pinned(path, expected, name):
    need(path.is_file() and not path.is_symlink() and sha(path) == expected,
         "controller_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    shared = import_pinned(SHARED_HOST, SHARED_HOST_SHA, "proof_shared_pins")
    alias = import_pinned(ALIAS_HOST, ALIAS_HOST_SHA, "proof_capture_pins")
    return shared, alias, shared.helper()


def reviewed_overrides():
    need(set(OVERRIDE_SHAS) == {
        "hymem/core/db.py", "hymem/core/schema.sql",
        "hymem/core/migrations/062_local_claim_replay_proof.sql",
        "hymem/dreaming/phase1.py", "hymem/dreaming/evidence.py",
        "hymem/dreaming/canonicalize.py", "hymem/portability.py",
    } and all(isinstance(value, str) and HEX64.fullmatch(value)
              for value in OVERRIDE_SHAS.values()), "proof_override_pins_unreviewed")
    return OVERRIDE_SHAS


def inventory(shared, alias, h):
    reviewed_overrides()
    alias.shared_state(shared, h)
    shared.candidate_inventory(h)
    expected = shared.baseline_inventory(h)
    expected.update(shared.OVERRIDE_SHAS)
    need(len(expected) == 480, "shared_source_inventory_invalid")
    actual_source = {}
    for path in SOURCE.rglob("*"):
        need(not path.is_symlink(), "source_symlink")
        if path.is_file():
            actual_source[path.relative_to(SOURCE).as_posix()] = sha(path)
    need(actual_source == expected, "shared_source_pin_drift")
    expected.update(OVERRIDE_SHAS)
    need(len(expected) == 481, "proof_inventory_size_invalid")
    actual = {}
    for path in CANDIDATE.rglob("*"):
        need(not path.is_symlink(), "candidate_symlink")
        if path.is_file():
            actual[path.relative_to(CANDIDATE).as_posix()] = sha(path)
    need(actual == expected, "proof_candidate_pin_drift")


def pins(shared, alias, h):
    need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
         and stat.S_IMODE(ROOT.stat().st_mode) == 0o700
         and CAPTURE.is_dir() and not CAPTURE.is_symlink()
         and stat.S_IMODE(CAPTURE.stat().st_mode) == 0o700
         and WORK.is_dir() and not WORK.is_symlink()
         and stat.S_IMODE(WORK.stat().st_mode) == 0o700,
         "stage_not_private")
    for path in (SELF, WORKER, OLD_REPLAY):
        h.regular(path, mode=0o400)
    need(sha(WORKER) == WORKER_SHA and sha(OLD_REPLAY) == OLD_REPLAY_SHA,
         "replay_worker_pin_drift")
    for path in (CAPTURE / "prepersist-001.json", CAPTURE / "prepersist-001.sqlite"):
        h.regular(path, mode=0o600)
    h.regular(REFERENCE, mode=0o400)
    need(sha(REFERENCE) == REFERENCE_SHA, "reference_pin_drift")
    for relative, expected in reviewed_overrides().items():
        path = OVERRIDES / relative
        h.regular(path, mode=0o400)
        need(sha(path) == expected, "override_upload_pin_drift")
    inventory(shared, alias, h)
    capture_hash = sha(CAPTURE / "prepersist-001.json")
    snapshot_hash = sha(CAPTURE / "prepersist-001.sqlite")
    need(capture_hash == sha(ORIGINAL_CAPTURE)
         and snapshot_hash == sha(ORIGINAL_SNAPSHOT), "capture_copy_drift")
    raw = h.read_json(CAPTURE / "prepersist-001.json")
    need(raw.get("source_sha256") == REFERENCE_SHA
         and raw.get("database_sha256") == snapshot_hash,
         "capture_source_binding_drift")
    return {"source_files": 481, "reference_sha256": REFERENCE_SHA,
            "phase1_sha256": OVERRIDE_SHAS["hymem/dreaming/phase1.py"],
            "capture_sha256": capture_hash, "snapshot_sha256": snapshot_hash,
            "worker_sha256": WORKER_SHA, "old_replay_sha256": OLD_REPLAY_SHA,
            "host_sha256": sha(SELF),
            "override_sha256": hashlib.sha256(json.dumps(
                OVERRIDE_SHAS, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()}


def installed(shared, alias, h):
    receipt = h.read_json(ROOT / "install.json")
    need(receipt == pins(shared, alias, h), "installed_pin_drift")
    return receipt


def configure(h):
    phase1_sha = OVERRIDE_SHAS["hymem/dreaming/phase1.py"]
    need(HEX64.fullmatch(phase1_sha), "phase1_unpinned")
    mounts = [(str(CANDIDATE), "/candidate", False),
              (str(WORKER), "/diag/claim_conflict_proof_replay.py", False),
              (str(OLD_REPLAY), "/diag/claim_conflict_instrumented_replay.py", False),
              (str(CAPTURE), "/capture", False),
              (str(REFERENCE), "/reference/source.sqlite", False),
              (str(WORK), "/work", True),
              (str(h.RUNTIME), "/home/node/hymem-env", False)]
    command = ["docker", "create", "--name", "hymem-proof-replay-v1",
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
                "/home/node/hymem-env/bin/python3", h.IMAGE,
                "-I", "-B", "/diag/claim_conflict_proof_replay.py",
                "--phase1-sha256", phase1_sha]
    return command, mounts


def inspect(h, cid, mounts):
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "container_id_invalid")
    item = json.loads(h.run(["docker", "inspect", cid], 30, "inspect"))[0]
    config, host, state = item["Config"], item["HostConfig"], item["State"]
    expected = configure(h)[0]
    image_index = expected.index(h.IMAGE)
    actual_mounts = {entry["Destination"]: (entry["Source"], entry["RW"], entry["Type"])
                     for entry in item["Mounts"]}
    need(actual_mounts == {dst: (src, rw, "bind") for src, dst, rw in mounts}
         and item["Image"] == h.IMAGE and config["Image"] == h.IMAGE
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
                     for key in config["Env"]), "container_configuration_drift")
    return {"container_id": cid, "status": state["Status"],
            "exit_code": state["ExitCode"], "oom_killed": state["OOMKilled"],
            "pid": state["Pid"], "configuration_verified": True}


def project_audit(raw):
    need(isinstance(raw, dict) and type(raw.get("integrity_ok")) is bool,
         "audit_shape_invalid")
    result = {"integrity_ok": raw["integrity_ok"]}
    for key in ("foreign_key_findings", "canonical_drift_findings",
                "ledger_count_mismatches", "same_generation_disagreeing_groups"):
        value = raw.get(key)
        need(type(value) is int and 0 <= value <= 100000000,
             "audit_counter_invalid")
        result[key] = value
    return result


def project_arm(raw, dedup):
    need(isinstance(raw, dict) and raw.get("status") == "completed"
         and raw.get("dedup_enabled") is dedup, "arm_status_invalid")
    result = {"status": "completed", "dedup_enabled": dedup}
    for key in ("historical_outcomes", "historical_proofs_after_upgrade",
                "published_before", "published_after_first", "published_after_repeat"):
        value = raw.get(key)
        need(type(value) is int and 0 <= value <= 100000000,
             "arm_counter_invalid")
        result[key] = value
    for key in ("semantic_digest_before_upgrade", "semantic_digest_after_upgrade",
                "proof_sha256", "database_sha256"):
        value = raw.get(key)
        need(isinstance(value, str) and HEX64.fullmatch(value), "arm_hash_invalid")
        result[key] = value
    for key in ("second_initialize_unchanged", "proof_reopen_unchanged",
                "proof_repeat_unchanged", "exact_repeat_unchanged"):
        value = raw.get(key)
        need(type(value) is bool, "arm_boolean_invalid")
        result[key] = value
    for key in ("integrity_before", "integrity_after_upgrade",
                "integrity_after_first", "integrity_after_reopen",
                "integrity_after_repeat"):
        result[key] = project_audit(raw.get(key))
    return result


def project(raw):
    need(isinstance(raw, dict) and raw.get("status") in ("completed", "error"),
         "worker_output_invalid")
    if raw["status"] == "error":
        need(raw.get("reason_code") == "proof_replay_failed"
             and raw.get("error_type") in {
                 "RuntimeError", "ValueError", "OSError", "OperationalError",
                 "IntegrityError", "TypeError", "Exception"}
             and type(raw.get("failure_captured")) is bool,
             "worker_error_projection_invalid")
        return {"status": "error", "reason_code": "proof_replay_failed",
                "error_type": raw["error_type"],
                "failure_captured": raw["failure_captured"]}
    result = {"status": "completed"}
    for key in ("capture_sha256", "snapshot_sha256", "phase1_sha256",
                "reference_sha256"):
        value = raw.get(key)
        need(isinstance(value, str) and HEX64.fullmatch(value),
             "worker_hash_invalid")
        result[key] = value
    result["dedup_on"] = project_arm(raw.get("dedup_on"), True)
    result["dedup_off"] = project_arm(raw.get("dedup_off"), False)
    return result


def clean(arm):
    return all(audit["integrity_ok"] is True and all(audit[key] == 0 for key in (
        "foreign_key_findings", "canonical_drift_findings",
        "ledger_count_mismatches", "same_generation_disagreeing_groups"))
        for audit in (arm["integrity_before"], arm["integrity_after_upgrade"],
                      arm["integrity_after_first"], arm["integrity_after_reopen"],
                      arm["integrity_after_repeat"]))


def verdict(metadata, receipt):
    need(metadata["status"] == "completed"
         and metadata["capture_sha256"] == receipt["capture_sha256"]
         and metadata["snapshot_sha256"] == receipt["snapshot_sha256"]
         and metadata["phase1_sha256"] == receipt["phase1_sha256"]
         and metadata["reference_sha256"] == receipt["reference_sha256"],
         "worker_source_identity_drift")
    for mode in ("dedup_on", "dedup_off"):
        arm = metadata[mode]
        need(clean(arm)
             and arm["historical_proofs_after_upgrade"] == 0
             and arm["semantic_digest_before_upgrade"]
             == arm["semantic_digest_after_upgrade"]
             and arm["second_initialize_unchanged"] is True
             and (arm["published_before"], arm["published_after_first"],
                  arm["published_after_repeat"]) == (0, 1, 1)
             and arm["proof_reopen_unchanged"] is True
             and arm["proof_repeat_unchanged"] is True
             and arm["exact_repeat_unchanged"] is True,
             "proof_arm_not_idempotent")


def stop(h, cid, mounts):
    subprocess.run(["docker", "stop", "--time", "10", cid],
                   capture_output=True, timeout=30)
    state = inspect(h, cid, mounts)
    need(state["status"] in ("created", "exited") and state["pid"] == 0,
         "cleanup_unverified")


def supervise(shared, alias, h):
    result = {"status": "failed", "stages": {}, "networked_runs_started": 0}
    try:
        receipt = installed(shared, alias, h)
        h.put_json(ROOT / "supervisor-intent.json",
                   {"networked_runs_allowed": 0, "max_containers": 1})
        command, mounts = configure(h)
        h.put_json(ROOT / "create-intent.json", {"host_sha256": sha(SELF)})
        cid = h.run(command, 60, "create").decode().strip()
        h.put_json(ROOT / "container.json", {"container_id": cid})
        need(inspect(h, cid, mounts)["status"] == "created",
             "container_not_created")
        h.put_json(ROOT / "start-intent.json", {"container_id": cid})
        try:
            need(h.run(["docker", "start", cid], 60, "start").decode().strip()
                 == cid, "start_identity")
            raw = h.run(["docker", "wait", cid], 900, "wait")
            need(re.fullmatch(rb"[0-9]{1,3}\n?", raw), "wait_shape")
        except BaseException:
            stop(h, cid, mounts)
            raise
        state = inspect(h, cid, mounts)
        need(state["status"] == "exited" and state["pid"] == 0
             and state["exit_code"] == int(raw) and not state["oom_killed"],
             "terminal_state_invalid")
        metadata = project(json.loads(h.run(["docker", "logs", cid], 30, "logs")))
        result["stages"]["replay"] = {**state, "metadata": metadata}
        need(state["exit_code"] == 0, "worker_failed")
        verdict(metadata, receipt)
        installed(shared, alias, h)
        result["status"] = "completed"
    except BaseException as exc:
        result["error_type"] = (type(exc).__name__ if type(exc).__name__ in
                                ("RuntimeError", "ValueError", "OSError")
                                else "Exception")
    h.put_json(ROOT / "result.json", result)


def remote(action):
    shared, alias, h = dependencies()
    if action == "remote-install":
        need(not CAPTURE.exists() and not CANDIDATE.exists() and not WORK.exists(),
             "install_already_attempted")
        reviewed_overrides()
        alias.shared_state(shared, h)
        h.regular(REFERENCE, mode=0o400)
        need(sha(REFERENCE) == REFERENCE_SHA, "reference_pin_drift")
        h.regular(OLD_REPLAY, mode=0o400)
        h.regular(WORKER, mode=0o400)
        need(sha(OLD_REPLAY) == OLD_REPLAY_SHA and sha(WORKER) == WORKER_SHA,
             "diagnostic_workers_pin_drift")
        for relative, expected in OVERRIDE_SHAS.items():
            path = OVERRIDES / relative
            h.regular(path, mode=0o400)
            need(sha(path) == expected, "override_upload_pin_drift")
        CAPTURE.mkdir(mode=0o700)
        for source in (ORIGINAL_CAPTURE, ORIGINAL_SNAPSHOT):
            target = CAPTURE / source.name
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600)
            with source.open("rb") as origin, os.fdopen(fd, "wb") as stream:
                shutil.copyfileobj(origin, stream, length=1048576)
                stream.flush()
                os.fsync(stream.fileno())
        shutil.copytree(SOURCE, CANDIDATE, symlinks=False)
        for relative in OVERRIDE_SHAS:
            target = CANDIDATE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                os.chmod(target, 0o600)
            target.write_bytes((OVERRIDES / relative).read_bytes())
        for path in CANDIDATE.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o400)
        os.chmod(CANDIDATE, 0o700)
        WORK.mkdir(mode=0o700)
        receipt = pins(shared, alias, h)
        h.put_json(ROOT / "install.json", receipt)
        return {"status": "installed_not_launched", **receipt}
    if action == "remote-status":
        if (ROOT / "result.json").exists():
            return h.read_json(ROOT / "result.json")
        return {"status": ("running_or_requires_inspection"
                           if (ROOT / "launch-intent.json").exists()
                           else "installed_not_launched")}
    if action == "supervise":
        supervise(shared, alias, h)
        return {"status": "supervisor_finished"}
    installed(shared, alias, h)
    need(not (ROOT / "launch-intent.json").exists(), "launch_already_attempted")
    h.put_json(ROOT / "launch-intent.json", {"host_sha256": sha(SELF)})
    child = subprocess.Popen([sys.executable, "-I", "-B", str(SELF), "supervise"],
                             cwd=ROOT, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True)
    h.put_json(ROOT / "launch.json", {"pid": child.pid})
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
    reviewed_overrides()
    local_host = Path(__file__)
    local_worker = local_host.with_name("claim_conflict_proof_replay.py")
    local_old_replay = local_host.with_name("claim_conflict_instrumented_replay.py")
    need(local_worker.is_file() and not local_worker.is_symlink()
         and sha(local_worker) == WORKER_SHA
         and local_old_replay.is_file() and not local_old_replay.is_symlink()
         and sha(local_old_replay) == OLD_REPLAY_SHA,
         "local_diagnostic_worker_pin_drift")
    bodies = {SELF.name: local_host.read_bytes(),
              WORKER.name: local_worker.read_bytes(),
              OLD_REPLAY.name: local_old_replay.read_bytes()}
    for relative, expected in OVERRIDE_SHAS.items():
        path = LOCAL_FROZEN / relative
        need(path.is_file() and not path.is_symlink() and sha(path) == expected,
             "local_override_pin_drift")
        bodies["overrides/" + relative] = path.read_bytes()
    config = {"root": str(ROOT), "files": {
        name: {"size": len(raw), "sha": hashlib.sha256(raw).hexdigest()}
        for name, raw in bodies.items()}}
    code = """import hashlib,json,os,pathlib,sys
root=pathlib.Path(C['root'])
assert os.geteuid()==1000 and not root.exists()
root.mkdir(mode=0o700)
for name,item in C['files'].items():
 raw=sys.stdin.buffer.read(item['size'])
 assert len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha']
 path=root/name
 path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sys.stdin.buffer.read(1)==b''
print(json.dumps({'status':'uploaded'}))
"""
    script = "import json\nC=json.loads(" + repr(json.dumps(config)) + ")\n" + code
    uploaded = ssh_json("python3 -I -B -c " + shlex.quote(script),
                        b"".join(bodies.values()))
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
