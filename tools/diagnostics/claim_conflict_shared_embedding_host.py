"""One-shot private shared-batch dream controller; explicit install/launch only.

The explicit override pins refer to a separately reviewed local freeze.
No installation, provider call, or production operation occurs on import.
The detached supervisor owns its Docker wait/stop deadline; a host-level
SIGKILL of that supervisor is outside this controller's cleanup guarantee.
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
ROOT = BASE / "shared-embedding-dream-v1"
SELF = ROOT / "claim_conflict_shared_embedding_host.py"
WORKER = ROOT / "claim_conflict_instrumented_dream.py"
OVERRIDE_DIR = ROOT / "overrides"
CANDIDATE = ROOT / "candidate"
SOURCE = BASE / "offline-compare-v1/candidate"
REFERENCE = BASE / "capture-next-v1/reference.sqlite"
REFERENCE_SHA = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
FIRSTFIX_SHA = "31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136"
HELPER = BASE / "capture-next-v1/claim-conflict-next-host.py"
HELPER_SHA = "d6d528fb54c662dff217f243ac3890bb9d43478e45a48f2610d7060900ac7195"
WORK = ROOT / "work"
LOCAL_FROZEN = Path("/private/tmp/hymem-shared-embedding-20260925.X8jEOF/candidate")
OVERRIDE_SHAS = {
    "hymem/core/embedding_batches.py": "fc06315ef4e4cae88bc2e28b6c9202838e350525f898220253498254e7c9bb4a",
    "hymem/dreaming/embeddings.py": "690930726ba1b960af3c50e0e98617b5d03316f0cbb1c30aa245d52f217a2da6",
    "hymem/dreaming/runner.py": "fa5e9a44a66e17e31fca0b6e2c07d6313b56f27a618c4d9e2cb6ba1146934d39",
    "hymem/dreaming/phase1.py": "c7946257957d2dd6b6e00728ce2c12a0b69031376a286c722748472575339fbb",
    "hymem/dreaming/aggregate.py": "a321e0f786fad09b6e95f27176fc825738e173d8bd792808e14d0b4f25af9840",
    "hymem/reembed.py": "7a04431691ac4a78db11b01d5d2d46d2286d8fc2dca2242be85ff94c377e4ae7",
    "hymem/query/augment.py": "9cbaacbce1f9edbb8bdb0a903347a2573cdeceaa85b0a388629216110c471b5d",
    "hymem/api.py": "d041d8c26f05a27bc0fda9185708469243fba4d2751a12e06f3edf67547521b5",
}
EXPECTED_ENDPOINT = "http://embedding-server:8766/v1"
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FRAME = re.compile(r"hymem(?:/[A-Za-z_][A-Za-z_0-9]*)*/[A-Za-z_][A-Za-z_0-9]*\.py\Z")


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def reviewed_override_pins():
    need(set(OVERRIDE_SHAS) == {
        "hymem/core/embedding_batches.py", "hymem/dreaming/embeddings.py",
        "hymem/dreaming/runner.py", "hymem/dreaming/phase1.py",
        "hymem/dreaming/aggregate.py", "hymem/reembed.py",
        "hymem/query/augment.py", "hymem/api.py",
    } and all(isinstance(value, str) and HEX64.fullmatch(value)
              for value in OVERRIDE_SHAS.values()), "override_pins_unreviewed")
    return OVERRIDE_SHAS


def helper():
    need(HELPER.is_file() and not HELPER.is_symlink()
         and sha(HELPER) == HELPER_SHA, "helper_pin_drift")
    spec = importlib.util.spec_from_file_location("shared_embedding_primitives", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_inventory(h):
    reviewed_override_pins()
    h.source_inventory()
    manifest = h.read_json(h.MANIFEST)
    expected = {}
    for group in ("source_sha256", "test_sha256", "auxiliary_sha256"):
        expected.update(manifest[group])
    expected["hymem/dreaming/phase1.py"] = FIRSTFIX_SHA
    need(SOURCE.is_dir() and not SOURCE.is_symlink(), "firstfix_source_missing")
    actual = {}
    for path in SOURCE.rglob("*"):
        need(not path.is_symlink(), "firstfix_source_symlink")
        if path.is_file():
            actual[path.relative_to(SOURCE).as_posix()] = sha(path)
    need(len(actual) == 479 and actual == expected, "firstfix_source_pin_drift")
    return expected


def candidate_inventory(h):
    expected = baseline_inventory(h)
    expected.update(OVERRIDE_SHAS)
    need(CANDIDATE.is_dir() and not CANDIDATE.is_symlink(), "candidate_missing")
    actual = {}
    for path in CANDIDATE.rglob("*"):
        need(not path.is_symlink(), "candidate_symlink")
        if path.is_file():
            actual[path.relative_to(CANDIDATE).as_posix()] = sha(path)
    need(len(actual) == 480 and actual == expected, "candidate_pin_drift")


def pins(h):
    need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
         and stat.S_IMODE(ROOT.stat().st_mode) == 0o700
         and WORK.is_dir() and not WORK.is_symlink()
         and stat.S_IMODE(WORK.stat().st_mode) == 0o700, "stage_not_private")
    candidate_inventory(h)
    h.regular(REFERENCE, mode=0o400)
    h.regular(h.RUNTIME_ENV, mode=0o600)
    h.regular(SELF, mode=0o400)
    h.regular(WORKER, mode=0o400)
    need(sha(REFERENCE) == REFERENCE_SHA, "reference_pin_drift")
    runtime = h.read_json(h.RUNTIME_ENV)
    need(runtime.get("HYMEM_EMBEDDING_BASE_URL") == EXPECTED_ENDPOINT
         and runtime.get("HYMEM_LLM_BASE_URL", "").rstrip("/")
         == "https://api.deepseek.com"
         and runtime.get("HYMEM_LLM_MODEL") == "deepseek-flash",
         "runtime_endpoint_pin_drift")
    for relative, expected in OVERRIDE_SHAS.items():
        path = OVERRIDE_DIR / relative
        h.regular(path, mode=0o400)
        need(sha(path) == expected, "override_upload_pin_drift")
    return {"reference_sha256": REFERENCE_SHA,
            "phase1_sha256": OVERRIDE_SHAS["hymem/dreaming/phase1.py"],
            "source_files": 480, "manifest_sha256": h.MANIFEST_SHA,
            "override_sha256": hashlib.sha256(json.dumps(
                OVERRIDE_SHAS, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "host_sha256": sha(SELF), "worker_sha256": sha(WORKER),
            "runtime_env_sha256": sha(h.RUNTIME_ENV)}


def installed(h):
    receipt = h.read_json(ROOT / "install.json")
    need(receipt == pins(h), "installed_pin_drift")
    return receipt


def configure(h, mode, phase1_sha):
    need(mode in ("offline", "live") and HEX64.fullmatch(phase1_sha),
         "invalid_mode_or_phase1_pin")
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
                "--phase1-sha256", phase1_sha,
                "--max-http-attempts", "704",
                "--max-llm-http-attempts", "192",
                "--max-embedding-http-attempts", "512",
                "--deadline-seconds", "1800"]
    return command, mounts


def inspect(h, cid, mode, mounts, phase1_sha):
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "invalid_container_id")
    raw = h.run(["docker", "inspect", cid], 30, "inspect_full_command")
    item = json.loads(raw)[0]
    config, host, state = item["Config"], item["HostConfig"], item["State"]
    expected_cmd = configure(h, mode, phase1_sha)[0]
    image_index = expected_cmd.index(h.IMAGE)
    actual_mounts = {mount["Destination"]:
                     (mount["Source"], mount["RW"], mount["Type"])
                     for mount in item["Mounts"]}
    need(actual_mounts == {dst: (src, rw, "bind") for src, dst, rw in mounts},
         "container_mount_drift")
    need(item["Image"] == h.IMAGE and config["Image"] == h.IMAGE
         and config["User"] == "1000:1000"
         and config["Entrypoint"] == ["/home/node/hymem-env/bin/python3"]
         and config["WorkingDir"] == "/candidate"
         and config["Cmd"] == expected_cmd[image_index + 1:],
         "container_identity_or_args_drift")
    need(host["NetworkMode"] == ("hermes-net" if mode == "live" else "none")
         and host["ReadonlyRootfs"] is True and host["Privileged"] is False
         and host["CapDrop"] == ["ALL"]
         and host["SecurityOpt"] == ["no-new-privileges"]
         and host["Init"] is True and host["Memory"] == 2147483648
         and host["NanoCpus"] == 2000000000 and host["PidsLimit"] == 128
         and host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,size=64m"}
         and host["RestartPolicy"]["Name"] == "no"
         and not any(key.startswith(("DEEPSEEK_", "OPENAI_", "HYMEM_"))
                     for key in config["Env"]),
         "container_security_drift")
    return {"container_id": cid, "mode": mode, "status": state["Status"],
            "exit_code": state["ExitCode"], "oom_killed": state["OOMKilled"],
            "pid": state["Pid"], "configuration_verified": True}


def project(raw):
    need(isinstance(raw, dict)
         and raw.get("status") in ("ready", "completed", "captured_failure",
                                    "budget_stopped", "error"), "summary_shape")
    result = {"status": raw["status"]}
    for name in ("completion_calls", "http_attempts", "llm_http_attempts",
                 "embedding_http_attempts", "llm_provider_attempts_reported",
                 "embedding_provider_attempts_reported", "chunks_processed",
                 "extractions_captured", "prepersist_captured",
                 "exception_events_captured", "exception_controlflow_skipped",
                 "prompt_tokens", "completion_tokens", "total_tokens"):
        if name in raw:
            value = raw[name]
            need(type(value) is int and 0 <= value <= 100000000,
                 "summary_count_invalid")
            result[name] = value
    need(result.get("completion_calls", 0) <= 64
         and result.get("llm_http_attempts", 0) <= 192
         and result.get("embedding_http_attempts", 0) <= 512
         and result.get("http_attempts", 0) <= 704,
         "summary_budget_exceeded")
    if all(name in result for name in ("http_attempts", "llm_http_attempts",
                                       "embedding_http_attempts")):
        need(result["http_attempts"] == result["llm_http_attempts"]
             + result["embedding_http_attempts"], "summary_attempt_disagreement")
    for name in ("cleanup_ok", "source_unchanged", "runtime_generation_verified",
                 "failure_captured", "instrumentation_capture_ok",
                 "token_usage_available", "accounting_verified",
                 "exception_capture_truncated"):
        if name in raw:
            need(type(raw[name]) is bool, "summary_boolean_invalid")
            result[name] = raw[name]
    for name in ("source_sha256", "phase1_sha256", "capture_sha256",
                 "session_sha256"):
        if name in raw:
            need(isinstance(raw[name], str) and HEX64.fullmatch(raw[name]),
                 "summary_hash_invalid")
            result[name] = raw[name]
    codes = {"error_type": {"ValueError", "RuntimeError", "TypeError",
                             "KeyError", "AssertionError", "TimeoutError",
                             "ConnectionError", "OSError", "MemoryError",
                             "BudgetStop", "InstrumentationStop", "Exception"},
             "budget_reason": {"completion_budget", "http_attempt_budget",
                               "llm_http_attempt_budget",
                               "embedding_http_attempt_budget",
                               "embedding_payload_invalid", "embedding_payload_limit",
                               "prepersist_capture_budget", "cooperative_deadline"}}
    for name, allowed in codes.items():
        if name in raw:
            need(raw[name] in allowed, "summary_static_code_invalid")
            result[name] = raw[name]
    frames = raw.get("candidate_frames", [])
    need(isinstance(frames, list) and len(frames) <= 12, "summary_frames_invalid")
    for frame in frames:
        need(isinstance(frame, dict) and set(frame) == {"path", "function", "line"}
             and isinstance(frame["path"], str) and FRAME.fullmatch(frame["path"])
             and isinstance(frame["function"], str) and frame["function"].isidentifier()
             and type(frame["line"]) is int and 1 <= frame["line"] <= 100000,
             "summary_frame_invalid")
    result["candidate_frames"] = frames
    return result


def stop(h, cid, mode, mounts, phase1_sha):
    subprocess.run(["docker", "stop", "--time", "10", cid],
                   capture_output=True, timeout=30)
    state = inspect(h, cid, mode, mounts, phase1_sha)
    need(state["status"] in ("created", "exited") and state["pid"] == 0,
         "cleanup_unverified")


def supervise(h):
    result = {"status": "failed", "stages": {}, "live_runs_started": 0}
    try:
        receipt = installed(h)
        phase1_sha = receipt["phase1_sha256"]
        h.put_json(ROOT / "supervisor-intent.json",
                   {"live_runs_allowed": 1, "max_completion_calls": 64,
                    "max_llm_http": 192, "max_embedding_http": 512})
        for mode in ("offline", "live"):
            installed(h)
            command, mounts = configure(h, mode, phase1_sha)
            h.put_json(ROOT / (mode + "-create-intent.json"), {"mode": mode})
            cid = h.run(command, 60, "create").decode().strip()
            h.put_json(ROOT / (mode + "-container.json"), {"container_id": cid})
            need(inspect(h, cid, mode, mounts, phase1_sha)["status"] == "created",
                 "container_not_created")
            h.put_json(ROOT / (mode + "-start-intent.json"), {"container_id": cid})
            try:
                if mode == "live":
                    result["live_runs_started"] = 1
                need(h.run(["docker", "start", cid], 60, "start").decode().strip()
                     == cid, "start_identity")
                raw = h.run(["docker", "wait", cid], 1860 if mode == "live" else 180,
                            "wait")
                need(re.fullmatch(rb"[0-9]{1,3}\n?", raw), "wait_shape")
            except BaseException:
                stop(h, cid, mode, mounts, phase1_sha)
                raise
            state = inspect(h, cid, mode, mounts, phase1_sha)
            need(state["status"] == "exited" and state["pid"] == 0
                 and not state["oom_killed"] and state["exit_code"] == int(raw),
                 "terminal_state_invalid")
            metadata = project(json.loads(h.run(["docker", "logs", cid], 30, "logs")))
            result["stages"][mode] = {**state, "metadata": metadata}
            need(state["exit_code"] == 0 and metadata.get("cleanup_ok") is True
                 and metadata.get("source_unchanged") is True
                 and metadata.get("runtime_generation_verified") is True
                 and metadata.get("source_sha256") == REFERENCE_SHA
                 and metadata.get("phase1_sha256") == phase1_sha,
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
            installed(h)
        result["status"] = "completed"
    except BaseException as exc:
        result["error_type"] = (type(exc).__name__ if type(exc).__name__
                                in ("RuntimeError", "ValueError", "OSError")
                                else "Exception")
    h.put_json(ROOT / "result.json", result)


def remote(action):
    h = helper()
    if action == "remote-install":
        need(not WORK.exists() and not CANDIDATE.exists(), "install_already_attempted")
        baseline_inventory(h)
        for relative, expected in OVERRIDE_SHAS.items():
            path = OVERRIDE_DIR / relative
            h.regular(path, mode=0o400)
            need(sha(path) == expected, "override_upload_pin_drift")
        WORK.mkdir(mode=0o700)
        shutil.copytree(SOURCE, CANDIDATE, symlinks=False)
        for relative in OVERRIDE_SHAS:
            target = CANDIDATE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                os.chmod(target, 0o600)
            target.write_bytes((OVERRIDE_DIR / relative).read_bytes())
        for path in CANDIDATE.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o400)
        os.chmod(CANDIDATE, 0o700)
        receipt = pins(h)
        h.put_json(ROOT / "install.json", receipt)
        return {"status": "installed_not_launched", **receipt}
    if action == "remote-status":
        if (ROOT / "result.json").exists():
            return h.read_json(ROOT / "result.json")
        return {"status": ("running_or_requires_inspection"
                           if (ROOT / "launch-intent.json").exists()
                           else "installed_not_launched")}
    if action == "supervise":
        supervise(h)
        return {"status": "supervisor_finished"}
    installed(h)
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
    reviewed_override_pins()
    local_host = Path(__file__)
    local_worker = local_host.with_name("claim_conflict_instrumented_dream.py")
    need(LOCAL_FROZEN.is_dir() and not LOCAL_FROZEN.is_symlink(),
         "reviewed_local_freeze_missing")
    bodies = {SELF.name: local_host.read_bytes(),
              WORKER.name: local_worker.read_bytes()}
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
