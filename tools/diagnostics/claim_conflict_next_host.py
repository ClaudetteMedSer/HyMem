"""One-shot, detached host control for the next private chunk capture.

Local actions are install, launch, and status. No action retries an ambiguous
remote result. The remote supervisor owns both containers and their stop
deadline, even if the SSH client disconnects. Output contains metadata only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys

if sys.flags.optimize:
    raise RuntimeError("optimized_execution_forbidden")

SNAPSHOT = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
STAGE = SNAPSHOT / "capture-next-v1"
SOURCE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/verification")
MANIFEST = SOURCE.parent / "diag/manifest.json"
MANIFEST_SHA = "1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8"
RUNTIME = Path("/opt/stacks/hermes/instance1/home/hymem-env")
IMAGE = "sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5"
HOST = STAGE / "claim-conflict-next-host.py"
WORKER = STAGE / "claim_conflict_next_capture.py"
LOCAL_WORKER = Path(__file__).with_name("claim_conflict_next_capture.py")
REFERENCE = STAGE / "reference.sqlite"
EXPECTED_REFERENCE_SHA256 = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
RUNTIME_ENV = SNAPSHOT / "runtime-env.json"
WORK = STAGE / "work"
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=2", "afrodite")
CID = re.compile(r"[0-9a-f]{64}\Z")
FRAME_PATH = re.compile(r"hymem(?:/[A-Za-z_][A-Za-z_0-9]*)*/[A-Za-z_][A-Za-z_0-9]*\.py\Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError(code)


def regular(path: Path, *, mode: int | None = None) -> None:
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "unsafe_file")
    if mode is not None:
        need(stat.S_IMODE(info.st_mode) == mode, "file_mode_drift")


def put_json(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    need(len(raw) <= 8192, "oversized_receipt")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict:
    regular(path)
    value = json.loads(path.read_bytes())
    need(isinstance(value, dict), "invalid_receipt")
    return value


def source_inventory() -> dict:
    need(SOURCE.is_dir() and not SOURCE.is_symlink(), "source_tree_missing")
    need(MANIFEST.is_file() and sha(MANIFEST) == MANIFEST_SHA,
         "manifest_pin_mismatch")
    manifest = read_json(MANIFEST)
    pins: dict[str, str] = {}
    for group in ("source_sha256", "test_sha256", "auxiliary_sha256"):
        part = manifest[group]
        need(isinstance(part, dict) and not set(pins).intersection(part),
             "manifest_duplicate")
        for name, digest in part.items():
            rel = Path(name)
            need(name == rel.as_posix() and not rel.is_absolute()
                 and ".." not in rel.parts and re.fullmatch(r"[0-9a-f]{64}", digest),
                 "unsafe_manifest_entry")
        pins.update(part)
    need(len(pins) == 479, "source_file_count")
    actual: dict[str, str] = {}
    for path in SOURCE.rglob("*"):
        need(not path.is_symlink(), "source_symlink")
        if path.is_file():
            actual[path.relative_to(SOURCE).as_posix()] = sha(path)
    need(actual == pins, "source_tree_mismatch")
    return {"manifest_sha256": MANIFEST_SHA, "source_files": len(pins)}


def stage_pins() -> dict:
    need(os.geteuid() == 1000, "host_user_mismatch")
    need(STAGE.is_dir() and not STAGE.is_symlink(), "stage_missing")
    need(STAGE.stat().st_uid == 1000, "stage_owner_mismatch")
    need(WORK.is_dir() and not WORK.is_symlink()
         and stat.S_IMODE(WORK.stat().st_mode) == 0o700, "work_not_private")
    need(RUNTIME.is_dir() and not RUNTIME.is_symlink(), "runtime_missing")
    regular(REFERENCE, mode=0o400)
    regular(RUNTIME_ENV, mode=0o600)
    regular(WORKER, mode=0o400)
    need(sha(REFERENCE) == EXPECTED_REFERENCE_SHA256, "reference_pin_mismatch")
    return {**source_inventory(), "reference_sha256": sha(REFERENCE),
            "runtime_env_sha256": sha(RUNTIME_ENV),
            "worker_sha256": sha(WORKER)}


def installed() -> dict:
    receipt = read_json(STAGE / "host-install.json")
    need(receipt.get("status") == "installed_not_launched"
         and receipt.get("host_sha256") == sha(HOST), "host_pin_drift")
    current = stage_pins()
    need(all(receipt.get(key) == value for key, value in current.items()),
         "input_pin_drift")
    return receipt


def run(command: list[str], timeout: int, code: str) -> bytes:
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(code + "_timeout_inspect_before_retry") from None
    need(result.returncode == 0 and len(result.stdout) <= 65536,
         code + "_failed_inspect_before_retry")
    return result.stdout


def container_command(mode: str) -> tuple[list[str], list[tuple[str, str, bool]]]:
    need(mode in ("offline", "live"), "invalid_mode")
    mounts = [(str(SOURCE), "/candidate", False),
              (str(WORKER), "/diag/claim_conflict_next_capture.py", False),
              (str(REFERENCE), "/reference/source.sqlite", False),
              (str(WORK), "/work", True),
              (str(RUNTIME), "/home/node/hymem-env", False)]
    mounts.append((str(RUNTIME_ENV), "/run/runtime-env.json", False))
    command = ["docker", "create", "--name", "hymem-claim-conflict-" + STAGE.name + "-" + mode,
               "--pull", "never", "--init", "--network",
               "hermes-net" if mode == "live" else "none",
               "--user", "1000:1000", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "128",
               "--memory", "2g", "--cpus", "2",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
               "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
               "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for src, dst, writable in mounts:
        command.extend(["--mount", "type=bind,src=" + src + ",dst=" + dst
                        + ("" if writable else ",readonly")])
    command.extend(["--workdir", "/candidate", "--entrypoint",
                    "/home/node/hymem-env/bin/python3", IMAGE,
                    "-I", "-B", "/diag/claim_conflict_next_capture.py", mode,
                    "--env", "/run/runtime-env.json"])
    return command, mounts


def inspect_container(cid: str, mode: str, mounts: list[tuple[str, str, bool]]) -> dict:
    need(bool(CID.fullmatch(cid)), "invalid_container_id")
    obj = json.loads(run(["docker", "inspect", cid], 30, "docker_inspect"))[0]
    cfg, host, state = obj["Config"], obj["HostConfig"], obj["State"]
    actual = {m["Destination"]: (m["Source"], m["RW"], m["Type"])
              for m in obj["Mounts"]}
    need(actual == {dst: (src, rw, "bind") for src, dst, rw in mounts},
         "container_mount_drift")
    need(obj["Image"] == IMAGE and cfg["Image"] == IMAGE
         and cfg["User"] == "1000:1000"
         and cfg["Entrypoint"] == ["/home/node/hymem-env/bin/python3"]
         and cfg["WorkingDir"] == "/candidate", "container_identity_drift")
    need(cfg["Cmd"] == container_command(mode)[0][-6:],
         "container_args_drift")
    need(host["NetworkMode"] == ("hermes-net" if mode == "live" else "none")
         and host["ReadonlyRootfs"] is True and host["Privileged"] is False
         and host["CapDrop"] == ["ALL"]
         and host["SecurityOpt"] == ["no-new-privileges"] and host["Init"] is True
         and host["Memory"] == 2147483648 and host["NanoCpus"] == 2000000000
         and host["PidsLimit"] == 128
         and host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,size=64m"}
         and host["RestartPolicy"]["Name"] == "no"
         and not any(key.startswith(("DEEPSEEK_", "OPENAI_", "HYMEM_"))
                     for key in cfg["Env"]),
         "container_security_drift")
    return {"container_id": cid, "mode": mode, "status": state["Status"],
            "exit_code": state["ExitCode"], "oom_killed": state["OOMKilled"],
            "pid": state["Pid"], "configuration_verified": True}


def stage_is_unchanged(pins: dict) -> None:
    current = stage_pins()
    need(all(pins.get(key) == value for key, value in current.items()),
         "postrun_input_pin_drift")
    need(sha(HOST) == pins["host_sha256"], "postrun_host_pin_drift")


def wait_owned(cid: str, mode: str, mounts: list[tuple[str, str, bool]]) -> dict:
    deadline = 660 if mode == "live" else 180
    try:
        raw = run(["docker", "wait", cid], deadline, "docker_wait")
    except RuntimeError:
        # The host supervisor owns this exact container. Stop it on its own
        # deadline, inspect terminal state, and never start another paid run.
        subprocess.run(["docker", "stop", "--time", "10", cid],
                       capture_output=True, timeout=30)
        state = inspect_container(cid, mode, mounts)
        need(state["status"] == "exited" and state["pid"] == 0,
             "timeout_stop_unverified")
        raise RuntimeError(mode + "_deadline_stopped") from None
    need(re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n?", raw) is not None,
         "docker_wait_output_invalid")
    state = inspect_container(cid, mode, mounts)
    need(state["status"] == "exited" and state["pid"] == 0
         and state["exit_code"] == int(raw) and not state["oom_killed"],
         "container_terminal_state_invalid")
    # Worker prints only metadata JSON. Never relay Docker logs or exception
    # prose. Keep only a bounded, schema-checked summary in the receipt.
    log = run(["docker", "logs", cid], 30, "docker_logs")
    need(len(log) <= 65536, "worker_output_oversized")
    summary = json.loads(log)
    need(isinstance(summary, dict), "worker_summary_invalid")
    if summary.get("status") == "error":
        worker_status = "error"
        frames = summary.get("candidate_frames", [])
        need(isinstance(frames, list) and len(frames) <= 12,
             "candidate_frames_invalid")
        for frame in frames:
            need(isinstance(frame, dict) and set(frame) == {"path", "function", "line"}
                 and isinstance(frame["path"], str)
                 and FRAME_PATH.fullmatch(frame["path"]) is not None
                 and isinstance(frame["function"], str)
                 and frame["function"].isidentifier()
                 and isinstance(frame["line"], int)
                 and 1 <= frame["line"] <= 100000,
                 "candidate_frame_invalid")
        projection = {key: summary.get(key) for key in (
            "error_type", "stage", "completion_calls", "http_attempts")}
        projection["candidate_frames"] = frames
    else:
        worker_status = summary.get("status")
        need(isinstance(worker_status, str)
             and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", worker_status) is not None,
             "worker_status_invalid")
        if mode == "offline":
            projection = {key: summary[key] for key in (
                "source_sha256", "target_chunk_id", "source_message_ids",
                "published_count", "same_generation_observation_count",
                "expected_generation_registry_count", "runtime_generation_verified",
                "source_coverage_verified", "completion_calls", "http_attempts",
                "capture_exists")}
        else:
            projection = {key: summary.get(key) for key in (
                "triples", "dedup_vectors", "completion_calls", "http_attempts",
                "generation_key", "source_sha256", "failure_reason")}
    return {**state, "worker_status": worker_status, "metadata": projection,
            "worker_summary_sha256": hashlib.sha256(log).hexdigest()}


def supervise() -> None:
    result = {"status": "failed", "manifest_sha256": MANIFEST_SHA,
              "stages": {}, "paid_live_runs_started": 0}
    try:
        receipt = installed()
        put_json(STAGE / "supervisor-intent.json",
                 {"host_sha256": receipt["host_sha256"],
                  "manifest_sha256": MANIFEST_SHA, "paid_live_runs_allowed": 1})
        for mode in ("offline", "live"):
            stage_is_unchanged(receipt)
            command, mounts = container_command(mode)
            put_json(STAGE / (mode + "-create-intent.json"),
                     {"mode": mode, "command_sha256": hashlib.sha256(
                         json.dumps(command).encode()).hexdigest()})
            cid = run(command, 60, "docker_create").decode().strip()
            put_json(STAGE / (mode + "-container.json"), {"container_id": cid})
            created = inspect_container(cid, mode, mounts)
            need(created["status"] == "created", "container_not_created")
            put_json(STAGE / (mode + "-start-intent.json"), {"container_id": cid})
            if mode == "live":
                result["paid_live_runs_started"] = 1
            try:
                started = run(["docker", "start", cid], 60, "docker_start").decode().strip()
                need(started == cid, "docker_start_identity")
            except BaseException:
                # An ambiguous start must never leave the paid container
                # running without the supervisor's deadline.
                subprocess.run(["docker", "stop", "--time", "10", cid],
                               capture_output=True, timeout=30)
                state = inspect_container(cid, mode, mounts)
                need(state["status"] in ("created", "exited")
                     and state["pid"] == 0, "ambiguous_start_stop_unverified")
                raise
            result["stages"][mode] = wait_owned(cid, mode, mounts)
            need(result["stages"][mode]["exit_code"] == 0,
                 mode + "_worker_failed")
            if mode == "offline":
                need(result["stages"][mode]["worker_status"] == "ready"
                     and result["stages"][mode]["metadata"]["completion_calls"] == 0
                     and result["stages"][mode]["metadata"]["http_attempts"] == 0,
                     "offline_preflight_not_ready")
            if mode == "live":
                need(result["stages"][mode]["worker_status"] == "captured",
                     "live_capture_not_successful")
            stage_is_unchanged(receipt)
        result["status"] = "completed"
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        result["error_code"] = str(exc) if isinstance(exc, RuntimeError) else "worker_or_host_failure"
    put_json(STAGE / "supervisor-result.json", result)


def remote_launch() -> dict:
    receipt = installed()
    need(not (STAGE / "supervisor-intent.json").exists()
         and not (STAGE / "supervisor-result.json").exists(),
         "launch_already_attempted")
    # Launch intent is durable before detachment. An ambiguous SSH result may
    # have started the supervisor, so there is no automatic retry path.
    put_json(STAGE / "launch-intent.json",
             {"host_sha256": receipt["host_sha256"],
              "manifest_sha256": MANIFEST_SHA})
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", str(HOST), "supervise"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, cwd=str(STAGE),
        start_new_session=True, close_fds=True,
    )
    value = {"status": "detached_supervisor_started", "pid": child.pid,
             "manifest_sha256": MANIFEST_SHA}
    put_json(STAGE / "launch.json", value)
    return value


def remote_status() -> dict:
    need(HOST.is_file() and not HOST.is_symlink(), "host_missing")
    path = STAGE / "supervisor-result.json"
    if path.exists():
        return read_json(path)
    launch = STAGE / "launch.json"
    if launch.exists():
        return {"status": "running_or_requires_inspection",
                "pid": read_json(launch)["pid"],
                "manifest_sha256": MANIFEST_SHA}
    return {"status": "installed_not_launched", "manifest_sha256": MANIFEST_SHA}


INSTALL_CODE = r'''
import hashlib,json,os,pathlib,stat,sys
if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
root=pathlib.Path(C['stage'])
assert os.geteuid()==1000 and root.is_dir() and not root.is_symlink()
for name in ('claim-conflict-next-host.py','claim_conflict_next_capture.py'):
    item=C['files'][name]; raw=sys.stdin.buffer.read(item['size'])
    assert len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sys.stdin.buffer.read(1)==b''
print(json.dumps({'status':'host_and_worker_uploaded',
                  'host_sha256':C['files']['claim-conflict-next-host.py']['sha256'],
                  'worker_sha256':C['files']['claim_conflict_next_capture.py']['sha256']}))
'''


def ssh_json(command: str, *, data: bytes = b"", timeout: int = 120) -> dict:
    try:
        result = subprocess.run([*SSH, command], input=data,
                                capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "ssh_timeout", "outcome": "unknown",
                "requires_inspection": True}
    if not result.stdout or len(result.stdout) > 8192:
        return {"status": "ssh_operation_failed", "outcome": "unknown",
                "requires_inspection": True}
    try:
        value = json.loads(result.stdout)
        need(isinstance(value, dict), "invalid_remote_json")
        return value
    except (ValueError, RuntimeError):
        return {"status": "ssh_output_invalid", "outcome": "unknown",
                "requires_inspection": True}


def local_install() -> dict:
    bodies = {"claim-conflict-next-host.py": Path(__file__).read_bytes(),
              "claim_conflict_next_capture.py": LOCAL_WORKER.read_bytes()}
    cfg = {"stage": str(STAGE), "files": {
        name: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in bodies.items()}}
    script = "C=json.loads(" + repr(json.dumps(cfg)) + ")\n" + INSTALL_CODE
    uploaded = ssh_json("python3 -I -B -c " + shlex.quote("import json\n" + script),
                        data=b"".join(bodies.values()), timeout=180)
    if uploaded.get("status") != "host_and_worker_uploaded":
        return uploaded
    return ssh_json("python3 -I -B " + shlex.quote(str(HOST)) + " remote-install")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "launch", "status",
                                           "remote-install", "remote-launch",
                                           "remote-status", "supervise"))
    args = parser.parse_args()
    try:
        if args.action == "install":
            value = local_install()
        elif args.action in ("launch", "status"):
            target = "remote-launch" if args.action == "launch" else "remote-status"
            value = ssh_json("python3 -I -B " + shlex.quote(str(HOST)) + " " + target)
        elif args.action == "remote-install":
            if not WORK.exists():
                WORK.mkdir(mode=0o700)
            need(not any(WORK.iterdir()), "work_not_empty_before_install")
            pins = stage_pins()
            receipt = {"status": "installed_not_launched", "host_sha256": sha(HOST),
                       **pins}
            put_json(STAGE / "host-install.json", receipt)
            value = {"status": "installed_not_launched", "manifest_sha256": MANIFEST_SHA,
                     "source_files": pins["source_files"],
                     "host_sha256": receipt["host_sha256"]}
        elif args.action == "remote-launch":
            value = remote_launch()
        elif args.action == "remote-status":
            value = remote_status()
        else:
            supervise()
            return 0
    except BaseException as exc:
        value = {"status": "operation_failed", "error_type": type(exc).__name__,
                 "requires_inspection": True}
    print(json.dumps(value, sort_keys=True))
    return 0 if value["status"] in ("installed_not_launched",
                                    "detached_supervisor_started", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
