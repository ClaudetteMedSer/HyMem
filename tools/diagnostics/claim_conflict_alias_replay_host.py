"""One-shot networkless alias-guard replay: frozen baseline versus one override.

Local install/launch/status are separate. Import never contacts the host.
Both containers get only a private capture copy, pinned code, and fresh work.
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
SHARED = BASE / "shared-embedding-dream-v1"
SHARED_HOST = SHARED / "claim_conflict_shared_embedding_host.py"
SHARED_HOST_SHA = "ca95bc8d06cc1cc9f84e80dcb91f13cdc333c681efa4e9aca5ae457d48cce1f0"
SHARED_LIVE_CID = "cdbedf81c439ce817de669f4185e02ffc58e8c23ad9c96ce2e128c57bbb40396"
SHARED_CANDIDATE = SHARED / "candidate"
ORIGINAL_CAPTURE = SHARED / "work/live/prepersist-001.json"
ORIGINAL_SNAPSHOT = SHARED / "work/live/prepersist-001.sqlite"
ROOT = BASE / "alias-replay-v1"
SELF = ROOT / "claim_conflict_alias_replay_host.py"
WORKER = ROOT / "claim_conflict_instrumented_replay.py"
OVERRIDE = ROOT / "canonicalize.py"
CAPTURE = ROOT / "capture"
BASELINE_CANDIDATE = ROOT / "baseline"
FIXED_CANDIDATE = ROOT / "fixed"
WORK = ROOT / "work"
REFERENCE_SHA = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
PHASE1_SHA = "c7946257957d2dd6b6e00728ce2c12a0b69031376a286c722748472575339fbb"
BASE_CANONICALIZE_SHA = "ca5fc4e2581248501e3c69741d468fb28de5c259902aeee9acc6c7dae6d16d73"
CANONICALIZE_SHA = "6fc1f95f5945ef28a99429e0ad9336a3ea4b5c28d85c22babff0747f792018c0"
LOCAL_CANONICALIZE = Path("/private/tmp/hymem-alias-idempotence-20260925.j2B8jZ/canonicalize.py")
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


def artifact_digest(root, h):
    need(root.is_dir() and not root.is_symlink(), "shared_capture_directory_invalid")
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.suffix in (".json", ".jsonl")
                               or path.name.startswith("prepersist-")
                               and path.suffix == ".sqlite"):
            h.regular(path, mode=0o600)
            digest.update(path.name.encode("ascii"))
            digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def shared_host():
    need(SHARED_HOST.is_file() and not SHARED_HOST.is_symlink()
         and sha(SHARED_HOST) == SHARED_HOST_SHA, "shared_host_pin_drift")
    spec = importlib.util.spec_from_file_location("alias_replay_shared_pins", SHARED_HOST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_state(s, h):
    installed = s.installed(h)
    need(installed["phase1_sha256"] == PHASE1_SHA
         and installed["source_files"] == 480, "shared_candidate_pin_drift")
    receipt = h.read_json(SHARED / "result.json")
    live = receipt.get("stages", {}).get("live", {})
    metadata = live.get("metadata", {})
    need(live.get("container_id") == SHARED_LIVE_CID
         and live.get("status") == "exited" and live.get("pid") == 0
         and live.get("exit_code") == 0
         and metadata.get("status") == "captured_failure"
         and metadata.get("prepersist_captured") == 1,
         "shared_live_receipt_drift")
    need(isinstance(metadata.get("capture_sha256"), str)
         and HEX64.fullmatch(metadata["capture_sha256"])
         and artifact_digest(SHARED / "work/live", h)
         == metadata["capture_sha256"], "shared_capture_artifact_drift")
    state = json.loads(h.run(["docker", "inspect", SHARED_LIVE_CID],
                             30, "shared_container_inspect"))[0]["State"]
    need(state["Status"] == "exited" and state["Pid"] == 0
         and state["ExitCode"] == 0 and not state["OOMKilled"],
         "shared_container_not_terminal")
    for path in (ORIGINAL_CAPTURE, ORIGINAL_SNAPSHOT):
        h.regular(path, mode=0o600)
    return installed


def candidate_inventory(s, h):
    s.candidate_inventory(h)
    baseline_expected = s.baseline_inventory(h)
    baseline_expected.update(s.OVERRIDE_SHAS)
    need(baseline_expected["hymem/dreaming/canonicalize.py"] == BASE_CANONICALIZE_SHA,
         "baseline_canonicalize_pin_drift")
    expected_by_mode = {
        "baseline": baseline_expected,
        "fixed": {**baseline_expected,
                  "hymem/dreaming/canonicalize.py": CANONICALIZE_SHA},
    }
    for mode, root in (("baseline", BASELINE_CANDIDATE),
                       ("fixed", FIXED_CANDIDATE)):
        need(root.is_dir() and not root.is_symlink(), "candidate_missing")
        actual = {}
        for path in root.rglob("*"):
            need(not path.is_symlink(), "candidate_symlink")
            if path.is_file():
                actual[path.relative_to(root).as_posix()] = sha(path)
        need(len(actual) == 480 and actual == expected_by_mode[mode],
             "replay_candidate_pin_drift")


def pins(s, h):
    need(HEX64.fullmatch(CANONICALIZE_SHA), "override_unreviewed")
    need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
         and stat.S_IMODE(ROOT.stat().st_mode) == 0o700
         and WORK.is_dir() and not WORK.is_symlink()
         and stat.S_IMODE(WORK.stat().st_mode) == 0o700,
         "stage_not_private")
    for mode in ("baseline", "fixed"):
        work = WORK / mode
        need(work.is_dir() and not work.is_symlink()
             and stat.S_IMODE(work.stat().st_mode) == 0o700,
             "arm_work_not_private")
    for path, mode in ((SELF, 0o400), (WORKER, 0o400), (OVERRIDE, 0o400),
                       (CAPTURE / "prepersist-001.json", 0o600),
                       (CAPTURE / "prepersist-001.sqlite", 0o600)):
        h.regular(path, mode=mode)
    shared_state(s, h)
    candidate_inventory(s, h)
    need(sha(OVERRIDE) == CANONICALIZE_SHA
         and sha(ORIGINAL_CAPTURE) == sha(CAPTURE / "prepersist-001.json")
         and sha(ORIGINAL_SNAPSHOT) == sha(CAPTURE / "prepersist-001.sqlite"),
         "capture_or_override_pin_drift")
    raw = h.read_json(CAPTURE / "prepersist-001.json")
    snapshot_sha = sha(CAPTURE / "prepersist-001.sqlite")
    need(raw.get("source_sha256") == REFERENCE_SHA
         and raw.get("database_sha256") == snapshot_sha,
         "capture_snapshot_identity_invalid")
    return {"source_files": 480, "reference_sha256": REFERENCE_SHA,
            "phase1_sha256": PHASE1_SHA,
            "canonicalize_sha256": CANONICALIZE_SHA,
            "capture_sha256": sha(CAPTURE / "prepersist-001.json"),
            "snapshot_sha256": snapshot_sha,
            "shared_install_sha256": sha(SHARED / "install.json"),
            "host_sha256": sha(SELF), "worker_sha256": sha(WORKER)}


def installed(s, h):
    receipt = h.read_json(ROOT / "install.json")
    need(receipt == pins(s, h), "installed_pin_drift")
    return receipt


def configure(h, mode):
    need(mode in ("baseline", "fixed"), "invalid_arm")
    candidate = BASELINE_CANDIDATE if mode == "baseline" else FIXED_CANDIDATE
    mounts = [(str(candidate), "/candidate", False),
              (str(WORKER), "/diag/claim_conflict_instrumented_replay.py", False),
              (str(CAPTURE), "/capture", False),
              (str(WORK / mode), "/work", True),
              (str(h.RUNTIME), "/home/node/hymem-env", False)]
    command = ["docker", "create", "--name", "hymem-" + ROOT.name + "-" + mode,
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
                "-I", "-B", "/diag/claim_conflict_instrumented_replay.py",
                "--capture-index", "1"]
    return command, mounts


def inspect(h, cid, mode, mounts):
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "invalid_container_id")
    item = json.loads(h.run(["docker", "inspect", cid], 30, "container_inspect"))[0]
    config, host, state = item["Config"], item["HostConfig"], item["State"]
    expected_cmd = configure(h, mode)[0]
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
    need(host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True
         and host["Privileged"] is False and host["CapDrop"] == ["ALL"]
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


def project_arm(raw):
    need(isinstance(raw, dict)
         and raw.get("status") in ("completed", "inconclusive")
         and type(raw.get("dedup_enabled")) is bool,
         "arm_status_invalid")
    result = {"status": raw["status"], "dedup_enabled": raw["dedup_enabled"]}
    for field in ("published_before", "published_after_first", "published_after_repeat"):
        value = raw.get(field)
        need(value is None and field == "published_after_repeat"
             or type(value) is int and 0 <= value <= 1,
             "publication_count_invalid")
        result[field] = value
    for name in ("first", "exact_repeat"):
        item = raw.get(name)
        if item is None:
            result[name] = None
            continue
        need(isinstance(item, dict)
             and item.get("status") in ("persisted", "rejected", "execution_failure"),
             "attempt_status_invalid")
        projected = {"status": item["status"]}
        code = item.get("reason_code")
        need(code is None or code in {
            "alias_owned_state_guard", "same_generation_observation_disagreement",
            "duplicate_validated_claim_citation",
            "same_generation_extraction_outcome_disagreement",
            "canonical_evidence_provenance_collision",
            "canonical_evidence_audit_collision",
            "canonical_claim_authority_collision",
            "missing_published_source_manifest", "sqlite_integrity_rejection",
            "unclassified_exception"}, "reason_code_invalid")
        projected["reason_code"] = code
        for field in ("logical_digest_before", "logical_digest_after"):
            value = item.get(field)
            need(value is None or isinstance(value, str) and HEX64.fullmatch(value),
                 "attempt_hash_invalid")
            projected[field] = value
        for field in ("rollback_preserved", "failure_captured"):
            if field in item:
                need(item[field] is None or type(item[field]) is bool,
                     "attempt_boolean_invalid")
                projected[field] = item[field]
        for field in ("pool_count_before", "pool_count_after"):
            if field in item:
                value = item[field]
                need(value is None or type(value) is int and 0 <= value <= 1000000,
                     "attempt_count_invalid")
                projected[field] = value
        for field in ("error_type", "diagnostic_error_type"):
            if field in item and item[field] is not None:
                need(item[field] in {"ValueError", "RuntimeError", "TypeError",
                                     "KeyError", "IntegrityError", "OperationalError",
                                     "DatabaseError", "OSError", "AssertionError",
                                     "MemoryError", "Exception"},
                     "attempt_error_type_invalid")
                projected[field] = item[field]
        frames = item.get("candidate_frames", [])
        need(isinstance(frames, list) and len(frames) <= 12,
             "attempt_frames_invalid")
        for frame in frames:
            need(isinstance(frame, dict) and set(frame) == {"path", "function", "line"}
                 and isinstance(frame["path"], str) and FRAME.fullmatch(frame["path"])
                 and isinstance(frame["function"], str) and frame["function"].isidentifier()
                 and type(frame["line"]) is int and 1 <= frame["line"] <= 100000,
                 "attempt_frame_invalid")
        projected["candidate_frames"] = frames[-4:]
        result[name] = projected
    for name in ("integrity_before", "integrity_after"):
        item = raw.get(name)
        need(isinstance(item, dict) and type(item.get("integrity_ok")) is bool,
             "integrity_shape_invalid")
        result[name] = {"integrity_ok": item["integrity_ok"]}
        for field in ("foreign_key_findings", "canonical_drift_findings",
                      "ledger_count_mismatches", "same_generation_disagreeing_groups"):
            value = item.get(field)
            need(type(value) is int and 0 <= value <= 100000000,
                 "integrity_count_invalid")
            result[name][field] = value
    for name in ("exact_repeat_unchanged",):
        value = raw.get(name)
        need(value is None or type(value) is bool, "repeat_boolean_invalid")
        result[name] = value
    value = raw.get("database_sha256")
    need(isinstance(value, str) and HEX64.fullmatch(value), "arm_database_hash_invalid")
    result["database_sha256"] = value
    return result


def project(raw):
    need(isinstance(raw, dict)
         and raw.get("status") in ("replayed", "error"),
         "replay_summary_invalid")
    if raw["status"] == "error":
        need(raw.get("reason_code") == "replay_setup_failed"
             and raw.get("error_type") in {
                 "ValueError", "RuntimeError", "TypeError", "KeyError",
                 "IntegrityError", "OperationalError", "DatabaseError",
                 "OSError", "AssertionError", "MemoryError", "Exception"}
             and type(raw.get("failure_captured")) is bool,
             "replay_setup_projection_invalid")
        frames = raw.get("candidate_frames", [])
        need(isinstance(frames, list) and len(frames) <= 12,
             "replay_setup_frames_invalid")
        for frame in frames:
            need(isinstance(frame, dict) and set(frame) == {"path", "function", "line"}
                 and isinstance(frame["path"], str) and FRAME.fullmatch(frame["path"])
                 and isinstance(frame["function"], str) and frame["function"].isidentifier()
                 and type(frame["line"]) is int and 1 <= frame["line"] <= 100000,
                 "replay_setup_frame_invalid")
        return {"status": "error", "reason_code": "replay_setup_failed",
                "error_type": raw["error_type"],
                "failure_captured": raw["failure_captured"],
                "candidate_frames": frames}
    need(type(raw.get("capture_index")) is int
         and raw["capture_index"] == 1, "replay_summary_invalid")
    result = {"status": "replayed", "capture_index": 1}
    for name in ("capture_sha256", "snapshot_sha256", "phase1_sha256"):
        value = raw.get(name)
        need(isinstance(value, str) and HEX64.fullmatch(value),
             "replay_hash_invalid")
        result[name] = value
    for name in ("dedup_on", "dedup_off"):
        result[name] = project_arm(raw.get(name))
        need(result[name]["dedup_enabled"] is (name == "dedup_on"),
             "dedup_arm_identity_invalid")
    return result


def clean(arm):
    return all(audit["integrity_ok"] is True
               and all(audit[name] == 0 for name in (
                   "foreign_key_findings", "canonical_drift_findings",
                   "ledger_count_mismatches", "same_generation_disagreeing_groups"))
               for audit in (arm["integrity_before"], arm["integrity_after"]))


def verdict(mode, metadata, receipt):
    need(metadata["capture_sha256"] == receipt["capture_sha256"]
         and metadata["snapshot_sha256"] == receipt["snapshot_sha256"]
         and metadata["phase1_sha256"] == PHASE1_SHA,
         "replay_input_identity_drift")
    for name in ("dedup_on", "dedup_off"):
        arm = metadata[name]
        need(clean(arm) and arm["status"] == "completed", "arm_inconclusive")
        first, repeat = arm["first"], arm["exact_repeat"]
        need(first is not None, "first_attempt_missing")
        need(all(isinstance(first.get(name), str)
                 and HEX64.fullmatch(first[name])
                 for name in ("logical_digest_before", "logical_digest_after")),
             "first_digest_missing")
        if mode == "baseline":
            need(first["status"] == "rejected"
                 and arm["published_before"] == 0
                 and arm["published_after_first"] == 0
                 and arm["published_after_repeat"] is None
                 and first["reason_code"] == "alias_owned_state_guard"
                 and first.get("rollback_preserved") is True
                 and first.get("logical_digest_before")
                 == first.get("logical_digest_after")
                 and first.get("failure_captured") is True
                 and repeat is None, "baseline_not_alias_guard_rollback")
        else:
            need(repeat is not None and all(isinstance(repeat.get(name), str)
                 and HEX64.fullmatch(repeat[name])
                 for name in ("logical_digest_before", "logical_digest_after")),
                 "repeat_digest_missing")
            need(first["status"] == "persisted"
                 and first["logical_digest_before"] != first["logical_digest_after"]
                 and arm["published_before"] == 0
                 and arm["published_after_first"] == 1
                 and arm["published_after_repeat"] == 1
                 and repeat is not None and repeat["status"] == "persisted"
                 and arm["exact_repeat_unchanged"] is True
                 and repeat["logical_digest_before"]
                 == repeat["logical_digest_after"]
                 and first["logical_digest_after"]
                 == repeat["logical_digest_before"],
                 "fixed_not_idempotent_publication")


def stop(h, cid, mode, mounts):
    subprocess.run(["docker", "stop", "--time", "10", cid],
                   capture_output=True, timeout=30)
    state = inspect(h, cid, mode, mounts)
    need(state["status"] in ("created", "exited") and state["pid"] == 0,
         "cleanup_unverified")


def supervise(s, h):
    result = {"status": "failed", "stages": {}, "networked_runs_started": 0}
    try:
        receipt = installed(s, h)
        h.put_json(ROOT / "supervisor-intent.json",
                   {"networked_runs_allowed": 0, "capture_index": 1})
        for mode in ("baseline", "fixed"):
            installed(s, h)
            command, mounts = configure(h, mode)
            h.put_json(ROOT / (mode + "-create-intent.json"), {"mode": mode})
            cid = h.run(command, 60, "create").decode().strip()
            h.put_json(ROOT / (mode + "-container.json"), {"container_id": cid})
            need(inspect(h, cid, mode, mounts)["status"] == "created",
                 "container_not_created")
            h.put_json(ROOT / (mode + "-start-intent.json"), {"container_id": cid})
            try:
                need(h.run(["docker", "start", cid], 60, "start").decode().strip()
                     == cid, "start_identity")
                raw = h.run(["docker", "wait", cid], 300, "wait")
                need(re.fullmatch(rb"[0-9]{1,3}\n?", raw), "wait_shape")
            except BaseException:
                stop(h, cid, mode, mounts)
                raise
            state = inspect(h, cid, mode, mounts)
            need(state["status"] == "exited" and state["pid"] == 0
                 and state["exit_code"] == int(raw) and not state["oom_killed"],
                 "terminal_state_invalid")
            metadata = project(json.loads(h.run(["docker", "logs", cid], 30, "logs")))
            result["stages"][mode] = {**state, "metadata": metadata}
            need(state["exit_code"] == 0, "worker_failed")
            verdict(mode, metadata, receipt)
            installed(s, h)
        result["status"] = "completed"
    except BaseException as exc:
        result["error_type"] = (type(exc).__name__ if type(exc).__name__
                                in ("RuntimeError", "ValueError", "OSError")
                                else "Exception")
    h.put_json(ROOT / "result.json", result)


def remote(action):
    s = shared_host()
    h = s.helper()
    if action == "remote-install":
        need(HEX64.fullmatch(CANONICALIZE_SHA), "override_unreviewed")
        need(not CAPTURE.exists() and not BASELINE_CANDIDATE.exists()
             and not FIXED_CANDIDATE.exists() and not WORK.exists(),
             "install_already_attempted")
        shared_state(s, h)
        h.regular(OVERRIDE, mode=0o400)
        need(sha(OVERRIDE) == CANONICALIZE_SHA, "override_upload_pin_drift")
        CAPTURE.mkdir(mode=0o700)
        for origin in (ORIGINAL_CAPTURE, ORIGINAL_SNAPSHOT):
            target = CAPTURE / origin.name
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with origin.open("rb") as source_stream, os.fdopen(fd, "wb") as stream:
                shutil.copyfileobj(source_stream, stream, length=1048576)
                stream.flush()
                os.fsync(stream.fileno())
        shutil.copytree(SHARED_CANDIDATE, BASELINE_CANDIDATE, symlinks=False)
        shutil.copytree(SHARED_CANDIDATE, FIXED_CANDIDATE, symlinks=False)
        target = FIXED_CANDIDATE / "hymem/dreaming/canonicalize.py"
        os.chmod(target, 0o600)
        target.write_bytes(OVERRIDE.read_bytes())
        for root in (BASELINE_CANDIDATE, FIXED_CANDIDATE):
            for path in root.rglob("*"):
                os.chmod(path, 0o700 if path.is_dir() else 0o400)
            os.chmod(root, 0o700)
        WORK.mkdir(mode=0o700)
        (WORK / "baseline").mkdir(mode=0o700)
        (WORK / "fixed").mkdir(mode=0o700)
        receipt = pins(s, h)
        h.put_json(ROOT / "install.json", receipt)
        return {"status": "installed_not_launched", **receipt}
    if action == "remote-status":
        if (ROOT / "result.json").exists():
            return h.read_json(ROOT / "result.json")
        return {"status": ("running_or_requires_inspection"
                           if (ROOT / "launch-intent.json").exists()
                           else "installed_not_launched")}
    if action == "supervise":
        supervise(s, h)
        return {"status": "supervisor_finished"}
    installed(s, h)
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
    need(HEX64.fullmatch(CANONICALIZE_SHA), "override_unreviewed")
    local_host = Path(__file__)
    local_worker = local_host.with_name("claim_conflict_instrumented_replay.py")
    need(LOCAL_CANONICALIZE.is_file() and not LOCAL_CANONICALIZE.is_symlink()
         and sha(LOCAL_CANONICALIZE) == CANONICALIZE_SHA,
         "local_override_pin_drift")
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
