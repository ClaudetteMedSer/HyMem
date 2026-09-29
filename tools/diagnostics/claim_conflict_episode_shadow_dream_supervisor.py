"""One-shot targeted private dream after proven network-free alias replay.

Local install/launch/status are explicit. Install requires completed baseline
guard rejection and fixed idempotent publication under the pinned alias-replay
controller. No production checkout or writable production store is mounted.
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
ROOT = BASE / "alias-dream-v1"
SELF = ROOT / "claim_conflict_alias_dream_host.py"
WORKER = ROOT / "claim_conflict_instrumented_dream.py"
OVERRIDE = ROOT / "canonicalize.py"
CANDIDATE = ROOT / "candidate"
WORK = ROOT / "work"
REFERENCE = BASE / "capture-next-v1/reference.sqlite"
REFERENCE_SHA = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
PHASE1_SHA = "c7946257957d2dd6b6e00728ce2c12a0b69031376a286c722748472575339fbb"
CANONICALIZE_SHA = "6fc1f95f5945ef28a99429e0ad9336a3ea4b5c28d85c22babff0747f792018c0"
WORKER_SHA = "9d6f03ef88efd01c40fd94affc9b31dd7b6b6e879066ce784f08e74600f0b863"
LOCAL_CANONICALIZE = Path("/private/tmp/hymem-alias-idempotence-20260925.j2B8jZ/canonicalize.py")
SHARED = BASE / "shared-embedding-dream-v1"
SHARED_CANDIDATE = SHARED / "candidate"
SHARED_HOST = SHARED / "claim_conflict_shared_embedding_host.py"
SHARED_HOST_SHA = "ca95bc8d06cc1cc9f84e80dcb91f13cdc333c681efa4e9aca5ae457d48cce1f0"
ALIAS_REPLAY = BASE / "alias-replay-v2"
ALIAS_HOST = ALIAS_REPLAY / "claim_conflict_alias_replay_v2_host.py"
ALIAS_HOST_SHA = "6e740e40e6a64129acbc8c2bd002d10fe28bcce377081efef039ed967c006d2d"
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
         "dependency_host_pin_drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    shared = import_pinned(SHARED_HOST, SHARED_HOST_SHA, "alias_dream_shared")
    alias = import_pinned(ALIAS_HOST, ALIAS_HOST_SHA, "alias_dream_replay")
    h = shared.helper()
    return shared, alias, h


def replay_gate(shared, alias, h):
    alias_receipt = alias.installed(shared, h)
    need(alias_receipt.get("canonicalize_sha256") == CANONICALIZE_SHA
         and alias_receipt.get("phase1_sha256") == PHASE1_SHA
         and alias_receipt.get("source_files") == 480,
         "alias_replay_candidate_pin_drift")
    result = h.read_json(ALIAS_REPLAY / "result.json")
    need(result.get("status") == "completed"
         and result.get("networked_runs_started") == 0,
         "alias_replay_not_completed")
    stages = result.get("stages", {})
    for mode in ("baseline", "fixed"):
        stage = stages.get(mode, {})
        cid = stage.get("container_id")
        need(isinstance(cid, str) and HEX64.fullmatch(cid)
             and stage.get("status") == "exited"
             and stage.get("pid") == 0 and stage.get("exit_code") == 0
             and stage.get("oom_killed") is False
             and stage.get("configuration_verified") is True,
             "alias_replay_stage_not_terminal")
        command, mounts = alias.configure(h, mode)
        live_state = alias.inspect(h, cid, mode, mounts)
        need(live_state["status"] == "exited" and live_state["pid"] == 0
             and live_state["exit_code"] == 0 and not live_state["oom_killed"],
             "alias_replay_container_drift")
        metadata = stage.get("metadata")
        need(isinstance(metadata, dict), "alias_replay_metadata_missing")
        alias.verdict(mode, metadata, alias_receipt)
    return sha(ALIAS_REPLAY / "result.json")


def candidate_inventory(shared, h):
    shared.candidate_inventory(h)
    expected = shared.baseline_inventory(h)
    expected.update(shared.OVERRIDE_SHAS)
    expected["hymem/dreaming/canonicalize.py"] = CANONICALIZE_SHA
    need(expected["hymem/dreaming/phase1.py"] == PHASE1_SHA,
         "phase1_pin_drift")
    need(CANDIDATE.is_dir() and not CANDIDATE.is_symlink(), "candidate_missing")
    actual = {}
    for path in CANDIDATE.rglob("*"):
        need(not path.is_symlink(), "candidate_symlink")
        if path.is_file():
            actual[path.relative_to(CANDIDATE).as_posix()] = sha(path)
    need(len(actual) == 480 and actual == expected, "candidate_pin_drift")


def pins(shared, alias, h):
    need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
         and stat.S_IMODE(ROOT.stat().st_mode) == 0o700
         and WORK.is_dir() and not WORK.is_symlink()
         and stat.S_IMODE(WORK.stat().st_mode) == 0o700,
         "stage_not_private")
    for path, mode in ((SELF, 0o400), (WORKER, 0o400),
                       (OVERRIDE, 0o400), (REFERENCE, 0o400),
                       (h.RUNTIME_ENV, 0o600)):
        h.regular(path, mode=mode)
    need(sha(WORKER) == WORKER_SHA and sha(OVERRIDE) == CANONICALIZE_SHA
         and sha(REFERENCE) == REFERENCE_SHA,
         "worker_override_or_reference_pin_drift")
    candidate_inventory(shared, h)
    shared_receipt = shared.installed(h)
    need(shared_receipt["phase1_sha256"] == PHASE1_SHA,
         "shared_phase1_pin_drift")
    replay_sha = replay_gate(shared, alias, h)
    runtime = h.read_json(h.RUNTIME_ENV)
    need(runtime.get("HYMEM_EMBEDDING_BASE_URL") == shared.EXPECTED_ENDPOINT
         and runtime.get("HYMEM_LLM_BASE_URL", "").rstrip("/")
         == "https://api.deepseek.com"
         and runtime.get("HYMEM_LLM_MODEL") == "deepseek-flash",
         "runtime_endpoint_pin_drift")
    return {"source_files": 480, "phase1_sha256": PHASE1_SHA,
            "canonicalize_sha256": CANONICALIZE_SHA,
            "reference_sha256": REFERENCE_SHA,
            "alias_replay_result_sha256": replay_sha,
            "shared_install_sha256": sha(SHARED / "install.json"),
            "runtime_env_sha256": sha(h.RUNTIME_ENV),
            "host_sha256": sha(SELF), "worker_sha256": WORKER_SHA}


def installed(shared, alias, h):
    receipt = h.read_json(ROOT / "install.json")
    need(receipt == pins(shared, alias, h), "installed_pin_drift")
    return receipt


def configure(h, mode):
    need(mode in ("offline", "live"), "invalid_mode")
    mounts = [(str(CANDIDATE), "/candidate", False),
              (str(WORKER), "/diag/claim_conflict_instrumented_dream.py", False),
              (str(REFERENCE), "/reference/source.sqlite", False),
              (str(WORK), "/work", True),
              (str(h.RUNTIME), "/home/node/hymem-env", False),
              (str(h.RUNTIME_ENV), "/run/runtime-env.json", False)]
    command = ["docker", "create", "--name", "hymem-" + ROOT.name + "-" + mode,
               "--pull", "never", "--init", "--network",
               "hermes-net" if mode == "live" else "none",
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
                "-I", "-B", "/diag/claim_conflict_instrumented_dream.py",
                mode, "--env", "/run/runtime-env.json",
                "--phase1-sha256", PHASE1_SHA,
                "--max-http-attempts", "896",
                "--max-llm-http-attempts", "384",
                "--max-embedding-http-attempts", "512",
                "--deadline-seconds", "2700"]
    return command, mounts


def inspect(shared, h, cid, mode, mounts):
    # The reviewed inspector validates all security fields and the full Cmd.
    original = shared.configure
    shared.configure = lambda helper, actual_mode, _phase1: configure(helper, actual_mode)
    try:
        return shared.inspect(h, cid, mode, mounts, PHASE1_SHA)
    finally:
        shared.configure = original


def stop(shared, h, cid, mode, mounts):
    subprocess.run(["docker", "stop", "--time", "10", cid],
                   capture_output=True, timeout=30)
    state = inspect(shared, h, cid, mode, mounts)
    need(state["status"] in ("created", "exited") and state["pid"] == 0,
         "cleanup_unverified")


def supervise(shared, alias, h):
    result = {"status": "failed", "stages": {},
              "paid_live_runs_started": 0}
    try:
        receipt = installed(shared, alias, h)
        h.put_json(ROOT / "supervisor-intent.json",
                   {"paid_live_runs_allowed": 1, "max_completion_calls": 128,
                    "max_llm_http": 384, "max_embedding_http": 512})
        for mode in ("offline", "live"):
            installed(shared, alias, h)
            command, mounts = configure(h, mode)
            h.put_json(ROOT / (mode + "-create-intent.json"), {"mode": mode})
            cid = h.run(command, 60, "create").decode().strip()
            h.put_json(ROOT / (mode + "-container.json"), {"container_id": cid})
            need(inspect(shared, h, cid, mode, mounts)["status"] == "created",
                 "container_not_created")
            h.put_json(ROOT / (mode + "-start-intent.json"), {"container_id": cid})
            try:
                if mode == "live":
                    result["paid_live_runs_started"] = 1
                need(h.run(["docker", "start", cid], 60, "start").decode().strip()
                     == cid, "start_identity")
                raw = h.run(["docker", "wait", cid], 2760 if mode == "live" else 180,
                            "wait")
                need(re.fullmatch(rb"[0-9]{1,3}\n?", raw), "wait_shape")
            except BaseException:
                stop(shared, h, cid, mode, mounts)
                raise
            state = inspect(shared, h, cid, mode, mounts)
            need(state["status"] == "exited" and state["pid"] == 0
                 and state["exit_code"] == int(raw) and not state["oom_killed"],
                 "terminal_state_invalid")
            metadata = shared.project(json.loads(h.run(["docker", "logs", cid],
                                                       30, "logs")))
            result["stages"][mode] = {**state, "metadata": metadata}
            need(state["exit_code"] == 0 and metadata.get("cleanup_ok") is True
                 and metadata.get("source_unchanged") is True
                 and metadata.get("runtime_generation_verified") is True
                 and metadata.get("source_sha256") == REFERENCE_SHA
                 and metadata.get("phase1_sha256") == PHASE1_SHA,
                 "worker_not_clean")
            if mode == "offline":
                need(metadata["status"] == "ready"
                     and metadata.get("completion_calls") == 0
                     and metadata.get("http_attempts") == 0,
                     "offline_preflight_failed")
            else:
                need(metadata["status"] in ("completed", "captured_failure",
                                            "budget_stopped")
                     and metadata.get("instrumentation_capture_ok") is True
                     and metadata.get("accounting_verified") is True
                     and all(name in metadata for name in (
                         "completion_calls", "http_attempts", "llm_http_attempts",
                         "embedding_http_attempts",
                         "llm_provider_attempts_reported",
                         "embedding_provider_attempts_reported")),
                     "live_evidence_incomplete")
                if metadata["status"] == "captured_failure":
                    need(metadata.get("failure_captured") is True,
                         "failure_capture_missing")
            installed(shared, alias, h)
        result["status"] = "completed"
    except BaseException as exc:
        result["error_type"] = (type(exc).__name__ if type(exc).__name__
                                in ("RuntimeError", "ValueError", "OSError")
                                else "Exception")
    h.put_json(ROOT / "result.json", result)


def remote(action):
    shared, alias, h = dependencies()
    if action == "remote-install":
        need(not WORK.exists() and not CANDIDATE.exists(),
             "install_already_attempted")
        replay_gate(shared, alias, h)
        shared.installed(h)
        h.regular(OVERRIDE, mode=0o400)
        need(sha(OVERRIDE) == CANONICALIZE_SHA
             and sha(WORKER) == WORKER_SHA,
             "upload_pin_drift")
        WORK.mkdir(mode=0o700)
        shutil.copytree(SHARED_CANDIDATE, CANDIDATE, symlinks=False)
        target = CANDIDATE / "hymem/dreaming/canonicalize.py"
        os.chmod(target, 0o600)
        target.write_bytes(OVERRIDE.read_bytes())
        for path in CANDIDATE.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o400)
        os.chmod(CANDIDATE, 0o700)
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
    local_host = Path(__file__)
    local_worker = local_host.with_name("claim_conflict_instrumented_dream.py")
    need(sha(local_worker) == WORKER_SHA
         and LOCAL_CANONICALIZE.is_file()
         and not LOCAL_CANONICALIZE.is_symlink()
         and sha(LOCAL_CANONICALIZE) == CANONICALIZE_SHA,
         "local_worker_or_override_pin_drift")
    bodies = {SELF.name: local_host.read_bytes(),
              WORKER.name: local_worker.read_bytes(),
              OVERRIDE.name: LOCAL_CANONICALIZE.read_bytes()}
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
 fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
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

