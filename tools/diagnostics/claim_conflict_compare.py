"""One-shot offline replay of one captured extraction on frozen and fixed R7.

Local actions: install, launch, status. No provider credentials or network are
mounted into either replay container. An ambiguous remote result is inspected,
never retried automatically. Private capture data never enters CLI output.
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

if sys.flags.optimize:
    raise RuntimeError("optimized_execution_forbidden")

SNAPSHOT = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
CAPTURE_STAGE = SNAPSHOT / "capture-v3"
ROOT = SNAPSHOT / "offline-compare-v1"
FROZEN = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/verification")
HOST = CAPTURE_STAGE / "claim-conflict-host.py"
WORKER = ROOT / "claim_conflict_capture.py"
SELF = ROOT / "claim-conflict-compare.py"
FIXED = ROOT / "fixed-phase1.py"
CANDIDATE = ROOT / "candidate"
REFERENCE = SNAPSHOT / "source.sqlite"
PHASE1_REL = Path("hymem/dreaming/phase1.py")
OLD_SHA = "bea40b7a6565542861fadf9683dcbfd2bc70fe51dcf3f6b5bfc5ece5be6488ee"
REFERENCE_SHA = "0db99a70b01b20852ae08ebd1726be01f57b64651e8fd85f5e9ab29b1b3c6b85"
HOST_SHA = "f5035a93b8ccbc0665a4a0b19adf54a2071932a7eb2600172f9f879bf7423e16"
FIXED_SHA = "31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136"
LOCAL_FIXED = Path("/private/tmp/hymem-claim-fix-20260925.7a7aDj/candidate") / PHASE1_REL
LOCAL_WORKER = Path(__file__).with_name("claim_conflict_capture.py")
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=2", "afrodite")


def require(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError(code)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(block)
    return hashed.hexdigest()


def exclusive(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(raw) < 8192, "receipt_too_large")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def receipt(path: Path) -> dict:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "invalid_receipt")
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), "invalid_receipt_shape")
    return value


def host_module():
    require(HOST.is_file() and not HOST.is_symlink()
            and sha(HOST) == HOST_SHA, "host_helper_pin_drift")
    spec = importlib.util.spec_from_file_location("claim_conflict_host_pinned", HOST)
    require(spec is not None and spec.loader is not None, "host_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_inputs() -> tuple[object, dict]:
    require(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink(),
            "compare_stage_invalid")
    h = host_module()
    inventory = h.source_inventory()
    require(inventory["source_files"] == 479, "source_inventory_invalid")
    pins = receipt(ROOT / "install.json")
    require(sha(SELF) == pins["compare_sha256"]
            and sha(WORKER) == pins["worker_sha256"]
            and sha(FIXED) == FIXED_SHA == pins["fixed_sha256"]
            and sha(REFERENCE) == REFERENCE_SHA == pins["reference_sha256"]
            and sha(HOST) == HOST_SHA == pins["host_sha256"], "installed_input_drift")
    require(sha(FROZEN / PHASE1_REL) == OLD_SHA, "baseline_phase1_drift")
    for name in ("claim-conflict-capture.json", "claim-conflict-vectors.json"):
        source = CAPTURE_STAGE / "work" / name
        info = source.lstat()
        require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600,
                "private_capture_invalid")
        require(sha(source) == pins[name + "_sha256"], "capture_pin_drift")
    return h, pins


def prepare() -> dict:
    h, pins = verify_inputs()
    manifest = json.loads(h.MANIFEST.read_bytes())
    files = {}
    for group in ("source_sha256", "test_sha256", "auxiliary_sha256"):
        files.update(manifest[group])
    require(len(files) == 479 and files[PHASE1_REL.as_posix()] == OLD_SHA,
            "baseline_manifest_drift")
    require(not CANDIDATE.exists(), "candidate_already_prepared")
    CANDIDATE.mkdir(mode=0o700)
    for name, expected in sorted(files.items()):
        origin = FROZEN / name
        require(sha(origin) == expected and not origin.is_symlink(),
                "frozen_file_drift")
        target = CANDIDATE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        os.chmod(target, 0o400)
    target = CANDIDATE / PHASE1_REL
    os.chmod(target, 0o600)
    target.write_bytes(FIXED.read_bytes())
    os.chmod(target, 0o400)
    actual = {p.relative_to(CANDIDATE).as_posix(): sha(p)
              for p in CANDIDATE.rglob("*") if p.is_file()}
    expected = {**files, PHASE1_REL.as_posix(): FIXED_SHA}
    require(actual == expected and not any(p.is_symlink() for p in CANDIDATE.rglob("*")),
            "candidate_override_inventory_mismatch")
    for arm in ("baseline", "fixed"):
        arm_dir = ROOT / arm
        arm_dir.mkdir(mode=0o700)
        work = arm_dir / "work"
        work.mkdir(mode=0o700)
        for name in ("claim-conflict-capture.json", "claim-conflict-vectors.json"):
            source = CAPTURE_STAGE / "work" / name
            target = work / name
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(source.read_bytes())
            require(sha(target) == pins[name + "_sha256"], "private_copy_mismatch")
    value = {"status": "prepared", "manifest_sha256": h.MANIFEST_SHA,
             "source_files": 479, "baseline_phase1_sha256": OLD_SHA,
             "fixed_phase1_sha256": FIXED_SHA,
             "capture_sha256": pins["claim-conflict-capture.json_sha256"],
             "vectors_sha256": pins["claim-conflict-vectors.json_sha256"]}
    exclusive(ROOT / "prepared.json", value)
    return value


def configure(h, arm: str) -> tuple[list[str], list[tuple[str, str, bool]]]:
    require(arm in ("baseline", "fixed"), "invalid_arm")
    h.STAGE = ROOT / arm
    h.SOURCE = FROZEN if arm == "baseline" else CANDIDATE
    h.WORKER = WORKER
    h.WORK = ROOT / arm / "work"
    h.REFERENCE = REFERENCE
    command, mounts = h.container_command("replay")
    require("none" in command and "hermes-net" not in command
            and "/run/runtime-env.json" not in command
            and command[-4:] == ["-I", "-B", "/diag/claim_conflict_capture.py", "replay"],
            "replay_command_not_offline")
    return command, mounts


def verify_prepared(pins: dict) -> None:
    require(sha(CANDIDATE / PHASE1_REL) == FIXED_SHA,
            "candidate_phase1_changed")
    h = host_module()
    manifest = json.loads(h.MANIFEST.read_bytes())
    expected = {}
    for group in ("source_sha256", "test_sha256", "auxiliary_sha256"):
        expected.update(manifest[group])
    expected[PHASE1_REL.as_posix()] = FIXED_SHA
    actual = {p.relative_to(CANDIDATE).as_posix(): sha(p)
              for p in CANDIDATE.rglob("*") if p.is_file()}
    require(actual == expected and len(actual) == 479,
            "candidate_tree_changed")
    for arm in ("baseline", "fixed"):
        for name in ("claim-conflict-capture.json", "claim-conflict-vectors.json"):
            require(sha(ROOT / arm / "work" / name) == pins[name + "_sha256"],
                    "private_capture_copy_changed")


def supervise() -> None:
    result = {"status": "failed", "arms": {}, "paid_calls": 0}
    try:
        h, pins = verify_inputs()
        require(receipt(ROOT / "prepared.json")["status"] == "prepared",
                "candidate_not_prepared")
        for arm in ("baseline", "fixed"):
            verify_inputs()
            verify_prepared(pins)
            command, mounts = configure(h, arm)
            exclusive(ROOT / (arm + "-create-intent.json"),
                      {"command_sha256": digest(json.dumps(command).encode())})
            cid = h.run(command, 60, "compare_create").decode().strip()
            exclusive(ROOT / (arm + "-container.json"), {"container_id": cid})
            created = h.inspect_container(cid, "replay", mounts)
            require(created["status"] == "created", "compare_not_created")
            exclusive(ROOT / (arm + "-start-intent.json"), {"container_id": cid})
            try:
                started = h.run(["docker", "start", cid], 60, "compare_start").decode().strip()
                require(started == cid, "compare_start_identity")
            except BaseException:
                subprocess.run(["docker", "stop", "--time", "10", cid],
                               capture_output=True, timeout=30)
                raise
            stage = h.wait_owned(cid, "replay", mounts)
            result["arms"][arm] = stage
            require(stage["exit_code"] == 0 and stage["worker_status"] == "replayed",
                    "compare_replay_failed")
            verify_prepared(pins)
        result["status"] = "completed"
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        result["error_code"] = str(exc) if isinstance(exc, RuntimeError) else "offline_failure"
    exclusive(ROOT / "result.json", result)


def remote_launch() -> dict:
    verify_inputs()
    require(not (ROOT / "launch-intent.json").exists(), "already_launched")
    prepared = prepare()
    exclusive(ROOT / "launch-intent.json",
              {"prepared_sha256": sha(ROOT / "prepared.json"), "paid_calls_allowed": 0})
    child = subprocess.Popen([sys.executable, "-I", "-B", str(SELF), "supervise"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, cwd=str(ROOT),
                             start_new_session=True, close_fds=True)
    value = {"status": "detached_offline_compare_started", "pid": child.pid,
             "prepared": prepared["status"], "paid_calls": 0}
    exclusive(ROOT / "launch.json", value)
    return value


INSTALL = r'''
import hashlib,json,os,pathlib,sys
if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
root=pathlib.Path(C['root']); assert os.geteuid()==1000 and root.parent.is_dir()
root.mkdir(mode=0o700)
for name in ('claim-conflict-compare.py','claim_conflict_capture.py','fixed-phase1.py'):
    item=C['files'][name]; raw=sys.stdin.buffer.read(item['size'])
    assert len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sys.stdin.buffer.read(1)==b''
print(json.dumps({'status':'uploaded','fixed_phase1_sha256':C['files']['fixed-phase1.py']['sha256']}))
'''


def ssh_json(command: str, *, payload: bytes = b"", timeout: int = 180) -> dict:
    try:
        result = subprocess.run([*SSH, command], input=payload,
                                capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "ssh_timeout", "outcome": "unknown", "requires_inspection": True}
    if len(result.stdout) > 8192:
        return {"status": "ssh_failed", "outcome": "unknown", "requires_inspection": True}
    try:
        value = json.loads(result.stdout)
        require(isinstance(value, dict), "remote_json_not_object")
        return value
    except ValueError:
        return {"status": "ssh_output_invalid", "outcome": "unknown", "requires_inspection": True}


def local_install() -> dict:
    bodies = {"claim-conflict-compare.py": Path(__file__).read_bytes(),
              "claim_conflict_capture.py": LOCAL_WORKER.read_bytes(),
              "fixed-phase1.py": LOCAL_FIXED.read_bytes()}
    require(digest(bodies["fixed-phase1.py"]) == FIXED_SHA, "local_fixed_pin_drift")
    cfg = {"root": str(ROOT), "files": {name: {"size": len(raw),
            "sha256": digest(raw)} for name, raw in bodies.items()}}
    command = "python3 -I -B -c " + shlex.quote(
        "import json\nC=json.loads(" + repr(json.dumps(cfg)) + ")\n" + INSTALL)
    upload = ssh_json(command, payload=b"".join(bodies.values()))
    if upload.get("status") != "uploaded":
        return upload
    return ssh_json("python3 -I -B " + shlex.quote(str(SELF)) + " remote-install")


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
            remote = "remote-launch" if args.action == "launch" else "remote-status"
            value = ssh_json("python3 -I -B " + shlex.quote(str(SELF)) + " " + remote)
        elif args.action == "remote-install":
            h = host_module()
            require(h.source_inventory()["source_files"] == 479, "source_pin_drift")
            require(sha(FROZEN / PHASE1_REL) == OLD_SHA
                    and sha(FIXED) == FIXED_SHA
                    and sha(REFERENCE) == REFERENCE_SHA, "input_pin_drift")
            captured = {}
            for name in ("claim-conflict-capture.json", "claim-conflict-vectors.json"):
                path = CAPTURE_STAGE / "work" / name
                require(stat.S_IMODE(path.stat().st_mode) == 0o600, "capture_mode")
                captured[name + "_sha256"] = sha(path)
            value = {"status": "installed_not_launched", "compare_sha256": sha(SELF),
                     "worker_sha256": sha(WORKER), "fixed_sha256": FIXED_SHA,
                     "reference_sha256": sha(REFERENCE), "host_sha256": sha(HOST),
                     **captured}
            exclusive(ROOT / "install.json", value)
            value = {"status": "installed_not_launched", "source_files": 479,
                     "fixed_phase1_sha256": FIXED_SHA}
        elif args.action == "remote-launch":
            value = remote_launch()
        elif args.action == "remote-status":
            value = receipt(ROOT / "result.json") if (ROOT / "result.json").exists() else (
                {"status": "running_or_requires_inspection"} if (ROOT / "launch-intent.json").exists()
                else {"status": "installed_not_launched"})
        else:
            supervise()
            return 0
    except BaseException as exc:
        value = {"status": "operation_failed", "error_type": type(exc).__name__,
                 "requires_inspection": True}
    print(json.dumps(value, sort_keys=True))
    return 0 if value["status"] in ("installed_not_launched",
                                    "detached_offline_compare_started", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
