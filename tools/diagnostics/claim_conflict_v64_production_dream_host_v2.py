"""Local host-only staging/one-shot controller; never connects over SSH.

Root creates and seals a private bundle descriptor after reviewing every gate.
The descriptor has version/root_reviewed, host_stage/container_stage, container,
container_id/image_id, mounts (exact destination -> source/RW/type map),
source_host, interpreter, files (relative
path -> sha256), manifest/manifest_sha256, and controller_sha256. Bundle inputs
and receipts must be 0600. All descriptor paths remain private. No raw process
environment, provider content, docker output or source is emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

VERSION = "schema64-production-host-v2"
BENCHMARKS = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks")
CONTAINER_BENCHMARKS = Path("/home/node/.hermes/benchmarks")
WORKER = "claim_conflict_v64_production_dream.py"


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regular_private(path):
    info = Path(path).lstat()
    need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600,
         "private_input_required")


def write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def json_write(path, value):
    write(path, json.dumps(value, sort_keys=True, allow_nan=False).encode())


def descriptor(path, seal):
    regular_private(path)
    need(sha(path) == seal, "external_review_seal_drift")
    value = json.loads(Path(path).read_bytes())
    need(value.get("version") == VERSION and value.get("root_reviewed") is True,
         "external_review_required")
    need(value.get("controller_sha256") == sha(__file__), "controller_pin_drift")
    host = Path(value["host_stage"])
    relative = host.relative_to(BENCHMARKS)
    need(len(relative.parts) == 1 and relative.name not in ("", ".", ".."), "stage_path_invalid")
    need(Path(value["container_stage"]) == CONTAINER_BENCHMARKS / relative,
         "stage_mount_identity_invalid")
    need(value["container"] == "hermes-1" and value["interpreter"] == "/home/node/hymem-env/bin/python3",
         "runtime_identity_invalid")
    need(isinstance(value["mounts"], dict) and value["mounts"], "sealed_mounts_required")
    files = value["files"]
    need(WORKER in files and value["manifest"] in files
         and files[value["manifest"]] == value["manifest_sha256"], "bundle_incomplete")
    for name, digest in files.items():
        relative = Path(name)
        need(len(relative.parts) == 1 and relative.name == name
             and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
             "bundle_file_invalid")
    return value


def stage(bundle, seal):
    value = descriptor(bundle, seal)
    destination = Path(value["host_stage"])
    # A partial stage remains an inspection gate and is never merged/retried.
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, digest in value["files"].items():
        source = Path(bundle).parent / name
        regular_private(source)
        need(sha(source) == digest, "bundle_source_drift")
        write(destination / name, source.read_bytes())
    write(destination / "reviewed-host-bundle.json", Path(bundle).read_bytes())
    json_write(destination / "stage-receipt.json", {"version": VERSION,
        "bundle_sha256": seal, "files": len(value["files"]), "verified": True})
    return {"status": "staged", "files": len(value["files"]), "bundle_sha256": seal}


def check_staged(value, seal):
    root = Path(value["host_stage"])
    need(root.is_dir() and not root.is_symlink() and stat.S_IMODE(root.stat().st_mode) == 0o700,
         "stage_changed")
    need(sha(root / "reviewed-host-bundle.json") == seal, "stage_review_changed")
    for name, digest in value["files"].items():
        regular_private(root / name)
        need(sha(root / name) == digest, "staged_file_drift")


def mapped_path(mounts, container_path):
    path = Path(container_path)
    candidates = []
    for destination, item in mounts.items():
        parent = Path(destination)
        need(parent.is_absolute() and ".." not in parent.parts, "mount_destination_invalid")
        if path == parent or parent in path.parents:
            candidates.append((len(parent.parts), parent, item))
    need(candidates, "required_bind_mapping_missing")
    _depth, destination, item = max(candidates, key=lambda entry: entry[0])
    need(item["type"] == "bind" and Path(item["source"]).is_absolute(), "required_bind_mapping_invalid")
    return Path(item["source"]) / path.relative_to(destination)


HEALTH_PROBE = """import http.client, json
connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=5)
try:
    connection.request("GET", "/health")
    response = connection.getresponse()
    body = response.read(4097)
    if response.status != 200 or len(body) > 4096:
        raise RuntimeError("honcho_health_invalid")
    if json.loads(body) != {"status": "ok", "backend": "hymem"}:
        raise RuntimeError("honcho_health_invalid")
    print('{"backend":"hymem","status":"ok"}')
finally:
    connection.close()
"""
HEALTH_RECEIPT = b'{"backend":"hymem","status":"ok"}\n'


def honcho_health(value):
    # Address the sealed container ID, using only its sealed interpreter.
    # No proxy, redirect, provider API, file mutation, or inherited Python path.
    outcome = subprocess.run(["docker", "exec", value["container_id"],
        value["interpreter"], "-I", "-B", "-c", HEALTH_PROBE],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=10, check=True)
    need(len(outcome.stdout) <= 128 and outcome.stdout == HEALTH_RECEIPT,
         "honcho_health_invalid")


def container_identity(value):
    outcome = subprocess.run(["docker", "inspect", value["container"]],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=20, check=True)
    need(len(outcome.stdout) <= 1024 * 1024, "inspect_output_limit")
    rows = json.loads(outcome.stdout)
    need(len(rows) == 1, "container_count_invalid")
    row = rows[0]
    need(row["Id"] == value["container_id"] and row["Image"] == value["image_id"]
         and row["State"]["Running"] is True
         and row["State"].get("OOMKilled") is False,
         "container_runtime_changed")
    state = row["State"]
    if "Health" in state:
        health = state["Health"]
        need(isinstance(health, dict) and health.get("Status") == "healthy",
             "container_runtime_changed")
    mounts = {item["Destination"]: {"source": item["Source"], "rw": item["RW"], "type": item["Type"]}
              for item in row["Mounts"]}
    need(len(mounts) == len(row["Mounts"]) and mounts == value["mounts"], "sealed_mounts_changed")
    need(mapped_path(mounts, "/home/node/HyMem") == Path(value["source_host"])
         and mapped_path(mounts, value["container_stage"]) == Path(value["host_stage"]),
         "container_mount_changed")
    honcho_health(value)


def command(value, mode):
    root = Path(value["container_stage"])
    return ["docker", "exec", value["container"], value["interpreter"], "-I", "-B",
            str(root / WORKER), mode, str(root / value["manifest"]), value["manifest_sha256"]]


def launch(bundle, seal):
    value = descriptor(bundle, seal)
    check_staged(value, seal)
    container_identity(value)
    root = Path(value["host_stage"])
    intent = root / "host-launch-v1"
    intent.mkdir(mode=0o700, exist_ok=False)
    # The helper's separate supervisor still verifies candidate/gates/profile
    # and creates its durable owned invocation before releasing worker stdin.
    json_write(intent / "intent.json", {"version": VERSION, "bundle_sha256": seal,
        "status": "one_shot_intent_no_automatic_retry"})
    stdout = os.open(intent / "stdout.private", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    stderr = os.open(intent / "stderr.private", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        proc = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
            "supervise", str(Path(bundle).resolve()), seal], stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
        json_write(intent / "host-owner.json", {"pid": proc.pid, "bundle_sha256": seal})
    finally:
        os.close(stdout)
        os.close(stderr)
    return {"status": "detached", "pid": proc.pid, "bundle_sha256": seal}


def supervise(bundle, seal):
    value = descriptor(bundle, seal)
    check_staged(value, seal)
    container_identity(value)
    intent = Path(value["host_stage"]) / "host-launch-v1"
    need((intent / "intent.json").is_file(), "launch_intent_missing")
    json_write(intent / "supervisor-start.json", {"bundle_sha256": seal,
        "pid": os.getpid(), "status": "exclusive_supervisor_no_retry"})
    # No timeout kills docker-exec while leaving an unowned container process.
    # The reviewed in-container supervisor owns the worker group and deadline.
    outcome = subprocess.run(command(value, "supervise"), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    json_write(intent / "host-result.json", {"exit_code": outcome.returncode,
        "bundle_sha256": seal, "completed": outcome.returncode == 0})
    return {"status": "completed" if outcome.returncode == 0 else "failed_inspect_before_retry",
            "exit_code": outcome.returncode, "bundle_sha256": seal}


def status(bundle, seal):
    value = descriptor(bundle, seal)
    check_staged(value, seal)
    root = Path(value["host_stage"])
    result = {"status": "staged", "bundle_sha256": seal}
    for name in ("production-dream-result.json", "production-dream-supervisor.json"):
        path = root / name
        if path.exists():
            regular_private(path)
            result[name.replace(".json", "_sha256")] = sha(path)
            data = json.loads(path.read_bytes())
            if name == "production-dream-result.json":
                result["status"] = "completed" if data.get("status") == "completed" else "failed_inspect_before_retry"
                for field in ("completion_calls", "llm_http_attempts", "embedding_http_attempts",
                              "http_attempts", "prompt_tokens", "completion_tokens", "total_tokens", "owned_run_id"):
                    number = data.get(field)
                    if type(number) is int and 0 <= number <= 100000000:
                        result[field] = number
    intent = root / "host-launch-v1"
    if intent.exists() and result["status"] == "staged":
        result["status"] = "launched_inspect_owned_receipts"
    if (intent / "host-result.json").exists():
        data = json.loads((intent / "host-result.json").read_bytes())
        result["host_exit_code"] = data["exit_code"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "launch", "status", "supervise"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("bundle_sha256")
    args = parser.parse_args()
    try:
        result = globals()[args.action](args.bundle, args.bundle_sha256)
    except BaseException:
        result = {"status": "failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(1 if result["status"] == "failed_inspect_before_retry" else 0)
