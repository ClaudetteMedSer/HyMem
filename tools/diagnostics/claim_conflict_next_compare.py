"""One-shot offline replay of the next captured extraction on both fixes.

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
CAPTURE_STAGE = SNAPSHOT / "capture-next-v1"
ROOT = SNAPSHOT / "offline-next-compare-v1"
FROZEN = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/verification")
HOST = CAPTURE_STAGE / "claim-conflict-next-host.py"
CAPTURE_HELPER = CAPTURE_STAGE / "claim_conflict_next_capture.py"
WORKER = ROOT / "claim_conflict_next_replay.py"
SELF = ROOT / "claim-conflict-next-compare.py"
FIX2 = ROOT / "fix2-phase1.py"
FIX1_TREE = SNAPSHOT / "offline-compare-v1/candidate"
FIX2_TREE = ROOT / "fix2-candidate"
REFERENCE = CAPTURE_STAGE / "reference.sqlite"
PHASE1_REL = Path("hymem/dreaming/phase1.py")
OLD_SHA = "bea40b7a6565542861fadf9683dcbfd2bc70fe51dcf3f6b5bfc5ece5be6488ee"
REFERENCE_SHA = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
HOST_SHA = "d6d528fb54c662dff217f243ac3890bb9d43478e45a48f2610d7060900ac7195"
CAPTURE_HELPER_SHA = "4ad2819309f51e96eb7b7195d4e215b1d53081ba9551df89cd9c657c23d3178c"
FIX1_SHA = "31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136"
FIX2_SHA = "540d257c596850b44adaacab922587b99445def7a7b6b6d7696e0739bbc2e27d"
LOCAL_FIX2 = Path(__file__).resolve().parents[2] / PHASE1_REL
LOCAL_WORKER = Path(__file__).with_name("claim_conflict_next_replay.py")
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
    spec = importlib.util.spec_from_file_location("claim_conflict_next_host_pinned", HOST)
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
            and sha(FIX2) == FIX2_SHA == pins["fix2_sha256"]
            and sha(REFERENCE) == REFERENCE_SHA == pins["reference_sha256"]
            and sha(HOST) == HOST_SHA == pins["host_sha256"]
            and sha(CAPTURE_HELPER) == CAPTURE_HELPER_SHA == pins["capture_helper_sha256"],
            "installed_input_drift")
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
    require(not FIX2_TREE.exists(), "candidate_already_prepared")
    FIX2_TREE.mkdir(mode=0o700)
    for name, expected in sorted(files.items()):
        origin = FROZEN / name
        require(sha(origin) == expected and not origin.is_symlink(),
                "frozen_file_drift")
        target = FIX2_TREE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        os.chmod(target, 0o400)
    target = FIX2_TREE / PHASE1_REL
    os.chmod(target, 0o600)
    target.write_bytes(FIX2.read_bytes())
    os.chmod(target, 0o400)
    actual = {p.relative_to(FIX2_TREE).as_posix(): sha(p)
              for p in FIX2_TREE.rglob("*") if p.is_file()}
    expected = {**files, PHASE1_REL.as_posix(): FIX2_SHA}
    require(actual == expected and not any(p.is_symlink() for p in FIX2_TREE.rglob("*")),
            "candidate_override_inventory_mismatch")
    verify_tree(FIX1_TREE, {**files, PHASE1_REL.as_posix(): FIX1_SHA})
    for arm in ("fix1", "fix2"):
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
             "source_files": 479, "fix1_phase1_sha256": FIX1_SHA,
             "fix2_phase1_sha256": FIX2_SHA,
             "capture_sha256": pins["claim-conflict-capture.json_sha256"],
             "vectors_sha256": pins["claim-conflict-vectors.json_sha256"]}
    exclusive(ROOT / "prepared.json", value)
    return value


def verify_tree(tree: Path, expected: dict[str, str]) -> None:
    require(tree.is_dir() and not tree.is_symlink(), "candidate_tree_missing")
    actual: dict[str, str] = {}
    for path in tree.rglob("*"):
        require(not path.is_symlink(), "candidate_symlink")
        if path.is_file():
            actual[path.relative_to(tree).as_posix()] = sha(path)
    require(len(actual) == 479 and actual == expected, "candidate_tree_pin_drift")


def configure(h, arm: str) -> tuple[list[str], list[tuple[str, str, bool]]]:
    require(arm in ("fix1", "fix2"), "invalid_arm")
    source = FIX1_TREE if arm == "fix1" else FIX2_TREE
    mounts = [(str(source), "/candidate", False),
              (str(WORKER), "/diag/claim_conflict_next_replay.py", False),
              (str(CAPTURE_HELPER), "/diag/claim_conflict_next_capture.py", False),
              (str(REFERENCE), "/reference/source.sqlite", False),
              (str(ROOT / arm / "work"), "/work", True),
              (str(h.RUNTIME), "/home/node/hymem-env", False)]
    command = ["docker", "create", "--name", "hymem-claim-next-" + arm,
               "--pull", "never", "--init", "--network", "none",
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
                    "/home/node/hymem-env/bin/python3", h.IMAGE,
                    "-I", "-B", "/diag/claim_conflict_next_replay.py"])
    return command, mounts


def inspect_container(h, cid: str, mounts: list[tuple[str, str, bool]]) -> dict:
    require(re.fullmatch(r"[0-9a-f]{64}", cid) is not None, "invalid_container_id")
    obj = json.loads(h.run(["docker", "inspect", cid], 30, "compare_inspect"))[0]
    cfg, host, state = obj["Config"], obj["HostConfig"], obj["State"]
    actual = {m["Destination"]: (m["Source"], m["RW"], m["Type"])
              for m in obj["Mounts"]}
    require(actual == {dst: (src, rw, "bind") for src, dst, rw in mounts},
            "compare_mount_drift")
    require(obj["Image"] == h.IMAGE and cfg["Image"] == h.IMAGE
            and cfg["User"] == "1000:1000"
            and cfg["Entrypoint"] == ["/home/node/hymem-env/bin/python3"]
            and cfg["Cmd"] == ["-I", "-B", "/diag/claim_conflict_next_replay.py"]
            and cfg["WorkingDir"] == "/candidate", "compare_identity_drift")
    require(host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True
            and host["Privileged"] is False and host["CapDrop"] == ["ALL"]
            and host["SecurityOpt"] == ["no-new-privileges"]
            and host["Init"] is True and host["Memory"] == 2147483648
            and host["NanoCpus"] == 2000000000 and host["PidsLimit"] == 128
            and host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,size=64m"}
            and host["RestartPolicy"]["Name"] == "no"
            and not any(key.startswith(("DEEPSEEK_", "OPENAI_", "HYMEM_"))
                        for key in cfg["Env"]), "compare_security_drift")
    return {"container_id": cid, "status": state["Status"],
            "exit_code": state["ExitCode"], "pid": state["Pid"],
            "oom_killed": state["OOMKilled"]}


def wait_owned(h, cid: str, mounts: list[tuple[str, str, bool]]) -> dict:
    try:
        raw = h.run(["docker", "wait", cid], 600, "compare_wait")
    except RuntimeError:
        subprocess.run(["docker", "stop", "--time", "10", cid],
                       capture_output=True, timeout=30)
        state = inspect_container(h, cid, mounts)
        require(state["status"] == "exited" and state["pid"] == 0,
                "compare_timeout_stop_unverified")
        raise RuntimeError("compare_deadline_stopped") from None
    require(re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n?", raw) is not None,
            "compare_wait_invalid")
    state = inspect_container(h, cid, mounts)
    require(state["status"] == "exited" and state["pid"] == 0
            and state["exit_code"] == int(raw) and not state["oom_killed"],
            "compare_terminal_state_invalid")
    log = h.run(["docker", "logs", cid], 30, "compare_logs")
    require(len(log) <= 65536, "compare_log_oversized")
    summary = json.loads(log)
    require(isinstance(summary, dict) and summary.get("status") in ("replayed", "error"),
            "compare_summary_invalid")
    types = {"ValueError", "RuntimeError", "FileNotFoundError", "FileExistsError",
             "TypeError", "KeyError", "IndexError", "AssertionError", "TimeoutError",
             "ConnectionError", "OSError", "MemoryError", "Exception", None}
    reasons = {None, "same_generation_observation_disagreement",
               "canonical_evidence_provenance_collision",
               "canonical_evidence_audit_collision", "canonical_claim_authority_collision",
               "missing_published_source_manifest", "duplicate_validated_claim_citation",
               "same_generation_extraction_outcome_disagreement", "unclassified_exception"}
    projection = {"status": summary["status"]}
    if summary["status"] == "replayed":
        for key, expected in (("capture_sha256", None), ("vectors_sha256", None),
                              ("source_sha256", REFERENCE_SHA)):
            value = summary.get(key)
            require(isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                    and (expected is None or value == expected),
                    "compare_capture_summary_invalid")
            projection[key] = value
        for mode in ("dedup_on", "dedup_off"):
            arm = summary[mode]
            require(arm["status"] in ("persisted", "rejected", "setup_failure",
                                      "execution_failure")
                    and arm["reason_code"] in reasons
                    and arm.get("error_type") in types
                    and arm.get("diagnostic_error_type") in types
                    and re.fullmatch(r"[0-9a-f]{64}", arm["logical_digest_before"])
                    and (arm["logical_digest_after"] is None or
                         re.fullmatch(r"[0-9a-f]{64}", arm["logical_digest_after"]))
                    and arm.get("rollback_preserved") in (True, False, None),
                    "compare_arm_summary_invalid")
            frames = arm.get("candidate_frames", [])
            require(isinstance(frames, list) and len(frames) <= 12,
                    "compare_frames_invalid")
            for frame in frames:
                require(isinstance(frame, dict) and set(frame) == {"path", "function", "line"}
                        and isinstance(frame["path"], str)
                        and h.FRAME_PATH.fullmatch(frame["path"]) is not None
                        and isinstance(frame["function"], str)
                        and frame["function"].isidentifier()
                        and len(frame["function"]) <= 96
                        and isinstance(frame["line"], int)
                        and 1 <= frame["line"] <= 100000,
                        "compare_frame_invalid")
            changed = arm.get("changed_field_sets", [])
            allowed_fields = {"polarity", "interpretation_key", "value_text",
                              "value_numeric", "value_unit", "temporal_scope"}
            require(isinstance(changed, list) and len(changed) <= 16
                    and all(isinstance(fields, list) and len(fields) <= 6
                            and all(isinstance(field, str) and field in allowed_fields
                                    for field in fields) for fields in changed),
                    "compare_changed_fields_invalid")
            count = arm.get("collision_count")
            require(count is None or isinstance(count, int) and 0 <= count <= 10000,
                    "compare_collision_count_invalid")
            projection[mode] = {key: arm.get(key) for key in (
                "status", "reason_code", "error_type", "diagnostic_error_type",
                "collision_count", "logical_digest_before", "logical_digest_after",
                "rollback_preserved")}
            projection[mode]["changed_field_sets"] = changed
            projection[mode]["candidate_frames"] = frames
    else:
        require(summary.get("reason_code") == "replay_setup_failed"
                and summary.get("error_type") in types, "compare_error_summary_invalid")
        projection.update({"reason_code": "replay_setup_failed",
                           "error_type": summary["error_type"]})
    return {**state, "worker_status": summary["status"],
            "metadata": projection,
            "worker_summary_sha256": digest(log)}


def verify_prepared(pins: dict) -> None:
    h = host_module()
    manifest = json.loads(h.MANIFEST.read_bytes())
    expected = {}
    for group in ("source_sha256", "test_sha256", "auxiliary_sha256"):
        expected.update(manifest[group])
    verify_tree(FIX1_TREE, {**expected, PHASE1_REL.as_posix(): FIX1_SHA})
    verify_tree(FIX2_TREE, {**expected, PHASE1_REL.as_posix(): FIX2_SHA})
    for arm in ("fix1", "fix2"):
        for name in ("claim-conflict-capture.json", "claim-conflict-vectors.json"):
            require(sha(ROOT / arm / "work" / name) == pins[name + "_sha256"],
                    "private_capture_copy_changed")


def conclusive(metadata: dict) -> bool:
    return all(metadata[mode]["status"] in ("persisted", "rejected")
               and (metadata[mode]["status"] != "rejected"
                    or metadata[mode]["rollback_preserved"] is True)
               for mode in ("dedup_on", "dedup_off"))


def supervise() -> None:
    result = {"status": "failed", "arms": {}, "paid_calls": 0}
    try:
        h, pins = verify_inputs()
        require(receipt(ROOT / "prepared.json")["status"] == "prepared",
                "candidate_not_prepared")
        for arm in ("fix1", "fix2"):
            verify_inputs()
            verify_prepared(pins)
            command, mounts = configure(h, arm)
            exclusive(ROOT / (arm + "-create-intent.json"),
                      {"command_sha256": digest(json.dumps(command).encode())})
            cid = h.run(command, 60, "compare_create").decode().strip()
            exclusive(ROOT / (arm + "-container.json"), {"container_id": cid})
            created = inspect_container(h, cid, mounts)
            require(created["status"] == "created", "compare_not_created")
            exclusive(ROOT / (arm + "-start-intent.json"), {"container_id": cid})
            try:
                started = h.run(["docker", "start", cid], 60, "compare_start").decode().strip()
                require(started == cid, "compare_start_identity")
            except BaseException:
                subprocess.run(["docker", "stop", "--time", "10", cid],
                               capture_output=True, timeout=30)
                raise
            stage = wait_owned(h, cid, mounts)
            result["arms"][arm] = stage
            require(stage["exit_code"] == 0 and stage["worker_status"] == "replayed",
                    "compare_replay_failed")
            require(conclusive(stage["metadata"]), "compare_arm_inconclusive")
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
for name in ('claim-conflict-next-compare.py','claim_conflict_next_replay.py','fix2-phase1.py'):
    item=C['files'][name]; raw=sys.stdin.buffer.read(item['size'])
    assert len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sys.stdin.buffer.read(1)==b''
print(json.dumps({'status':'uploaded','fix2_phase1_sha256':C['files']['fix2-phase1.py']['sha256']}))
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
    bodies = {"claim-conflict-next-compare.py": Path(__file__).read_bytes(),
              "claim_conflict_next_replay.py": LOCAL_WORKER.read_bytes(),
              "fix2-phase1.py": LOCAL_FIX2.read_bytes()}
    require(digest(bodies["fix2-phase1.py"]) == FIX2_SHA, "local_fix2_pin_drift")
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
            capture_result = receipt(CAPTURE_STAGE / "supervisor-result.json")
            require(capture_result.get("status") == "completed"
                    and capture_result.get("stages", {}).get("live", {}).get("worker_status") == "captured",
                    "capture_not_successful")
            require(sha(FROZEN / PHASE1_REL) == OLD_SHA
                    and sha(FIX2) == FIX2_SHA
                    and sha(REFERENCE) == REFERENCE_SHA
                    and sha(CAPTURE_HELPER) == CAPTURE_HELPER_SHA,
                    "input_pin_drift")
            manifest = json.loads(h.MANIFEST.read_bytes())
            expected = {}
            for group in ("source_sha256", "test_sha256", "auxiliary_sha256"):
                expected.update(manifest[group])
            verify_tree(FIX1_TREE, {**expected, PHASE1_REL.as_posix(): FIX1_SHA})
            captured = {}
            for name in ("claim-conflict-capture.json", "claim-conflict-vectors.json"):
                path = CAPTURE_STAGE / "work" / name
                require(path.is_file() and not path.is_symlink()
                        and stat.S_IMODE(path.stat().st_mode) == 0o600, "capture_mode")
                captured[name + "_sha256"] = sha(path)
            value = {"status": "installed_not_launched", "compare_sha256": sha(SELF),
                     "worker_sha256": sha(WORKER), "fix2_sha256": FIX2_SHA,
                     "reference_sha256": sha(REFERENCE), "host_sha256": sha(HOST),
                     "capture_helper_sha256": sha(CAPTURE_HELPER),
                     **captured}
            exclusive(ROOT / "install.json", value)
            value = {"status": "installed_not_launched", "source_files": 479,
                     "fix1_phase1_sha256": FIX1_SHA,
                     "fix2_phase1_sha256": FIX2_SHA}
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
