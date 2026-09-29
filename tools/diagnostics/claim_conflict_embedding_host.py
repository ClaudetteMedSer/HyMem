"""One-shot host controller for the isolated embedding-only private-clone check.

Local install/launch/status actions are explicit. Importing this module or
running its tests never contacts the host. Ambiguous launch outcomes are not
retried; the detached supervisor owns both container lifecycles.
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
import sqlite3
import stat
import subprocess
import sys

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "embedding-batches-v1"
SELF = ROOT / "claim_conflict_embedding_host.py"
WORKER = ROOT / "claim_conflict_embedding_verify.py"
OVERRIDE = ROOT / "embeddings.py"
HELPER = BASE / "capture-next-v1/claim-conflict-next-host.py"
HELPER_SHA = "d6d528fb54c662dff217f243ac3890bb9d43478e45a48f2610d7060900ac7195"
SOURCE = BASE / "offline-compare-v1/candidate"
CANDIDATE = ROOT / "candidate"
V2_ROOT = BASE / "instrumented-dream-v2"
V2_CID = "0fd13c946eb6243435fb9d85bdd4befb1c048d45564e1395316c829ffef84123"
V2_CAPTURE_SHA = "032eb109fa0fce0b2cb6fa3e1179f0035e4f5862aa5a65468600cadac669684f"
V2_DB = V2_ROOT / "work/live/hymem.sqlite"
REFERENCE = ROOT / "reference.sqlite"
WORK = ROOT / "work"
PHASE1_SHA = "31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136"
BASE_EMBEDDINGS_SHA = "aff6e2d71dd7e53aadffbd69d7a63b00244598187e70a94c23f7e73bc662722d"
EMBEDDINGS_SHA = "53713f956efdabdd880dfda7c926db34c37e7e4def6833b7b8866a1af4f74978"
EMBEDDING_BASE_URL = "http://embedding-server:8766/v1"
EMBEDDING_BASE_URL_SHA = "081b277dc2e1bc5cb0aa00b7d5c4688377d5a36a53840535b3e6e1e1369676ae"
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FRAME = re.compile(r"hymem(?:/[A-Za-z_][A-Za-z_0-9]*)*/[A-Za-z_][A-Za-z_0-9]*\.py\Z")


def need(condition, code):
    if not condition:
        raise RuntimeError(code)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def helper():
    need(HELPER.is_file() and not HELPER.is_symlink() and sha(HELPER) == HELPER_SHA,
         "helper_pin_drift")
    spec = importlib.util.spec_from_file_location("embedding_host_primitives", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_inventory(h):
    h.source_inventory()
    manifest = h.read_json(h.MANIFEST)
    expected = {}
    for group in ("source_sha256", "test_sha256", "auxiliary_sha256"):
        expected.update(manifest[group])
    need(expected["hymem/dreaming/embeddings.py"] == BASE_EMBEDDINGS_SHA,
         "baseline_embedding_not_git_head")
    expected["hymem/dreaming/phase1.py"] = PHASE1_SHA
    need(SOURCE.is_dir() and not SOURCE.is_symlink(), "source_missing")
    actual = {}
    for path in SOURCE.rglob("*"):
        need(not path.is_symlink(), "source_symlink")
        if path.is_file():
            actual[path.relative_to(SOURCE).as_posix()] = sha(path)
    need(len(actual) == 479 and actual == expected, "firstfix_source_pin_drift")
    return expected


def candidate_inventory(h):
    expected = original_inventory(h)
    expected["hymem/dreaming/embeddings.py"] = EMBEDDINGS_SHA
    need(CANDIDATE.is_dir() and not CANDIDATE.is_symlink(), "candidate_missing")
    actual = {}
    for path in CANDIDATE.rglob("*"):
        need(not path.is_symlink(), "candidate_symlink")
        if path.is_file():
            actual[path.relative_to(CANDIDATE).as_posix()] = sha(path)
    need(len(actual) == 479 and actual == expected, "candidate_pin_drift")


def v2_terminal(h):
    receipt = h.read_json(V2_ROOT / "result.json")
    live = receipt.get("stages", {}).get("live", {})
    need(live.get("container_id") == V2_CID and live.get("status") == "exited"
         and live.get("pid") == 0 and live.get("exit_code") == 0
         and live.get("metadata", {}).get("capture_sha256") == V2_CAPTURE_SHA,
         "v2_receipt_pin_drift")
    state = json.loads(h.run(["docker", "inspect", V2_CID], 30, "v2_inspect"))[0]["State"]
    need(state["Status"] == "exited" and state["Pid"] == 0
         and state["ExitCode"] == 0 and not state["OOMKilled"],
         "v2_container_not_terminal")
    need(V2_DB.is_file() and not V2_DB.is_symlink(), "v2_database_missing")


def backup_reference(h):
    v2_terminal(h)
    fd = os.open(REFERENCE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    source = sqlite3.connect(f"file:{V2_DB}?mode=ro", uri=True)
    target = sqlite3.connect(REFERENCE)
    try:
        source.backup(target)
        # vec0 is intentionally not loaded by the host's Python interpreter.
        # The pinned worker performs full integrity checks with vec0 loaded.
    finally:
        target.close()
        source.close()
    os.chmod(REFERENCE, 0o400)
    return sha(REFERENCE)


def pins(h):
    need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
         and stat.S_IMODE(ROOT.stat().st_mode) == 0o700
         and WORK.is_dir() and not WORK.is_symlink()
         and stat.S_IMODE(WORK.stat().st_mode) == 0o700, "stage_not_private")
    candidate_inventory(h)
    for path, mode in ((SELF, 0o400), (WORKER, 0o400), (OVERRIDE, 0o400),
                       (REFERENCE, 0o400), (h.RUNTIME_ENV, 0o600)):
        h.regular(path, mode=mode)
    runtime_values = h.read_json(h.RUNTIME_ENV)
    need(runtime_values.get("HYMEM_EMBEDDING_BASE_URL") == EMBEDDING_BASE_URL
         and hashlib.sha256(EMBEDDING_BASE_URL.encode()).hexdigest()
         == EMBEDDING_BASE_URL_SHA, "embedding_route_pin_drift")
    need(sha(OVERRIDE) == EMBEDDINGS_SHA, "override_pin_drift")
    return {"source_files": 479, "manifest_sha256": h.MANIFEST_SHA,
            "phase1_sha256": PHASE1_SHA, "embeddings_sha256": EMBEDDINGS_SHA,
            "reference_sha256": sha(REFERENCE), "host_sha256": sha(SELF),
            "worker_sha256": sha(WORKER), "runtime_env_sha256": sha(h.RUNTIME_ENV)}


def installed(h):
    receipt = h.read_json(ROOT / "install.json")
    need(receipt == pins(h), "installed_pin_drift")
    return receipt


def configure(h, mode, reference_sha):
    need(mode in ("offline", "live") and HEX64.fullmatch(reference_sha),
         "invalid_mode_or_source_pin")
    mounts = [(str(CANDIDATE), "/candidate", False),
              (str(WORKER), "/diag/claim_conflict_embedding_verify.py", False),
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
               "--env", "PYTHONDONTWRITEBYTECODE=1",
               "--env", "CLAIM_SOURCE_SHA256=" + reference_sha]
    for src, dst, rw in mounts:
        command += ["--mount", "type=bind,src=" + src + ",dst=" + dst
                    + ("" if rw else ",readonly")]
    command += ["--workdir", "/candidate", "--entrypoint",
                "/home/node/hymem-env/bin/python3", h.IMAGE,
                "-I", "-B", "/diag/claim_conflict_embedding_verify.py",
                mode, "--env", "/run/runtime-env.json"]
    return command, mounts


def inspect(h, cid, mode, mounts, source_sha):
    original = h.container_command
    h.container_command = lambda actual_mode: configure(h, actual_mode, source_sha)
    try:
        return h.inspect_container(cid, mode, mounts)
    finally:
        h.container_command = original


def projection(raw):
    need(isinstance(raw, dict)
         and raw.get("status") in ("ready", "completed", "budget_stopped",
                                    "captured_failure", "incomplete", "error"),
         "worker_summary_invalid")
    result = {"status": raw["status"]}
    for field in ("http_attempts", "provider_request_attempts", "batches_processed",
                  "vectors_persisted", "cache_hits", "remote_texts_admitted",
                  "remote_chars_admitted", "remote_utf8_bytes_admitted"):
        if field in raw:
            value = raw[field]
            need(type(value) is int and 0 <= value <= 100000000, "worker_count_invalid")
            result[field] = value
    need(result.get("http_attempts", 0) <= 128
         and result.get("provider_request_attempts", 0) <= 128,
         "embedding_http_budget_exceeded")
    for field in ("cleanup_ok", "source_unchanged", "accounting_verified",
                  "full_pending_coverage", "graph_unchanged", "failure_captured"):
        if field in raw:
            need(type(raw[field]) is bool, "worker_boolean_invalid")
            result[field] = raw[field]
    for field in ("source_sha256", "clone_sha256"):
        if field in raw:
            need(isinstance(raw[field], str) and HEX64.fullmatch(raw[field]),
                 "worker_hash_invalid")
            result[field] = raw[field]
    for field in ("before", "after"):
        if field in raw:
            item = raw[field]
            need(isinstance(item, dict), "worker_census_invalid")
            result[field] = {}
            for name in ("dimension", "scanned_batches", "scanned_chunks",
                         "pending_chunks", "cached_pending_chunks",
                         "remote_miss_texts", "remote_miss_chars",
                         "remote_miss_utf8_bytes", "planned_http_calls",
                         "uncallable_remote_batches"):
                value = item.get(name)
                need(type(value) is int and 0 <= value <= 100000000,
                     "worker_census_count_invalid")
                result[field][name] = value
            for name in ("model_sha256", "scanned_ids_sha256",
                         "pending_ids_sha256", "last_chunk_id_sha256"):
                value = item.get(name)
                need(isinstance(value, str) and HEX64.fullmatch(value),
                     "worker_census_hash_invalid")
                result[field][name] = value
    for field in ("graph_before", "graph_after"):
        if field in raw:
            item = raw[field]
            need(isinstance(item, dict) and type(item.get("integrity_ok")) is bool,
                 "worker_audit_invalid")
            result[field] = {"integrity_ok": item["integrity_ok"]}
            for name in ("foreign_key_findings", "canonical_drift_findings",
                         "ledger_count_mismatches"):
                value = item.get(name)
                need(type(value) is int and 0 <= value <= 100000000,
                     "worker_audit_count_invalid")
                result[field][name] = value
            for name in ("graph_core_sha256", "nonembedding_logical_sha256"):
                value = item.get(name)
                need(isinstance(value, str) and HEX64.fullmatch(value),
                     "worker_audit_hash_invalid")
                result[field][name] = value
    finite_codes = {
        "error_type": {"ValueError", "RuntimeError", "TypeError", "OSError",
                       "TimeoutError", "ConnectionError", "BudgetStop", "Exception"},
        "budget_reason": {"planned_payload_limit", "planned_http_budget",
                          "embedding_payload_invalid", "embedding_payload_limit",
                          "embedding_http_budget"},
        "stage": {"startup", "offline", "live", "client_identity", "live_embedding"},
    }
    for field, allowed in finite_codes.items():
        if field in raw:
            need(raw[field] in allowed, "worker_static_code_invalid")
            result[field] = raw[field]
    frames = raw.get("candidate_frames", [])
    need(isinstance(frames, list) and len(frames) <= 12, "worker_frames_invalid")
    for frame in frames:
        need(isinstance(frame, dict) and set(frame) == {"path", "function", "line"}
             and isinstance(frame["path"], str) and FRAME.fullmatch(frame["path"])
             and isinstance(frame["function"], str) and frame["function"].isidentifier()
             and type(frame["line"]) is int and 1 <= frame["line"] <= 100000,
             "worker_frame_invalid")
    result["candidate_frames"] = frames
    return result


def stop(h, cid, mode, mounts, source_sha):
    subprocess.run(["docker", "stop", "--time", "10", cid],
                   capture_output=True, timeout=30)
    state = inspect(h, cid, mode, mounts, source_sha)
    need(state["status"] in ("created", "exited") and state["pid"] == 0,
         "cleanup_unverified")


def supervise(h):
    result = {"status": "failed", "stages": {}, "embedding_live_runs_started": 0,
              "llm_live_runs_started": 0}
    try:
        receipt = installed(h)
        source_sha = receipt["reference_sha256"]
        h.put_json(ROOT / "supervisor-intent.json",
                   {"embedding_live_runs_allowed": 1, "llm_live_runs_allowed": 0})
        for mode in ("offline", "live"):
            installed(h)
            command, mounts = configure(h, mode, source_sha)
            h.put_json(ROOT / (mode + "-create-intent.json"), {"mode": mode})
            cid = h.run(command, 60, "create").decode().strip()
            h.put_json(ROOT / (mode + "-container.json"), {"container_id": cid})
            need(inspect(h, cid, mode, mounts, source_sha)["status"] == "created",
                 "not_created")
            h.put_json(ROOT / (mode + "-start-intent.json"), {"container_id": cid})
            try:
                if mode == "live":
                    result["embedding_live_runs_started"] = 1
                need(h.run(["docker", "start", cid], 60, "start").decode().strip() == cid,
                     "start_identity")
                raw = h.run(["docker", "wait", cid], 960 if mode == "live" else 180,
                            "wait")
                need(re.fullmatch(rb"[0-9]{1,3}\n?", raw), "wait_shape")
            except BaseException:
                stop(h, cid, mode, mounts, source_sha)
                raise
            state = inspect(h, cid, mode, mounts, source_sha)
            need(state["status"] == "exited" and state["pid"] == 0
                 and not state["oom_killed"] and state["exit_code"] == int(raw),
                 "terminal_state")
            metadata = projection(json.loads(h.run(["docker", "logs", cid], 30, "logs")))
            result["stages"][mode] = {**state, "metadata": metadata}
            need(state["exit_code"] == 0 and metadata.get("cleanup_ok") is True
                 and metadata.get("source_unchanged") is True
                 and metadata.get("accounting_verified") is True
                 and metadata.get("source_sha256") == source_sha,
                 "worker_not_clean")
            before_graph = metadata.get("graph_before", {})
            need(before_graph.get("integrity_ok") is True
                 and before_graph.get("foreign_key_findings") == 0
                 and before_graph.get("canonical_drift_findings") == 0
                 and before_graph.get("ledger_count_mismatches") == 0,
                 "reference_graph_audit_failed")
            if mode == "offline":
                need(metadata["status"] == "ready"
                     and metadata.get("http_attempts") == 0
                     and metadata.get("provider_request_attempts") == 0
                     and metadata["before"]["uncallable_remote_batches"] == 0
                     and metadata["before"]["planned_http_calls"] <= 128,
                     "offline_preflight_failed")
            else:
                need("http_attempts" in metadata and "provider_request_attempts" in metadata
                     and metadata["status"] in ("completed", "captured_failure",
                                            "budget_stopped"),
                     "live_diagnostic_failed")
                if metadata["status"] == "completed":
                    need(metadata.get("full_pending_coverage") is True
                         and metadata.get("graph_unchanged") is True
                         and metadata["after"]["pending_chunks"] == 0
                         and metadata["before"]["model_sha256"]
                         == metadata["after"]["model_sha256"],
                         "live_verification_incomplete")
                if metadata["status"] == "captured_failure":
                    need(metadata.get("failure_captured") is True,
                         "failure_evidence_missing")
            installed(h)
        result["status"] = "completed"
    except BaseException as exc:
        result["error_type"] = (type(exc).__name__ if type(exc).__name__
                                in ("RuntimeError", "ValueError", "OSError") else "Exception")
    h.put_json(ROOT / "result.json", result)


def remote(action):
    h = helper()
    if action == "remote-install":
        need(not WORK.exists() and not CANDIDATE.exists() and not REFERENCE.exists(),
             "install_already_attempted")
        need(sha(OVERRIDE) == EMBEDDINGS_SHA, "override_pin_drift")
        original_inventory(h)
        v2_terminal(h)
        WORK.mkdir(mode=0o700)
        shutil.copytree(SOURCE, CANDIDATE, symlinks=False)
        target = CANDIDATE / "hymem/dreaming/embeddings.py"
        os.chmod(target, 0o600)
        target.write_bytes(OVERRIDE.read_bytes())
        for path in CANDIDATE.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o400)
        os.chmod(CANDIDATE, 0o700)
        reference_sha = backup_reference(h)
        pins_now = pins(h)
        need(pins_now["reference_sha256"] == reference_sha, "backup_pin_changed")
        h.put_json(ROOT / "install.json", pins_now)
        return {"status": "installed_not_launched", **pins_now}
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
    local_host = Path(__file__)
    local_worker = local_host.with_name("claim_conflict_embedding_verify.py")
    local_override = local_host.parents[2] / "hymem/dreaming/embeddings.py"
    need(sha(local_override) == EMBEDDINGS_SHA, "local_override_pin_drift")
    head = subprocess.run(["git", "show", "HEAD:hymem/dreaming/embeddings.py"],
                          cwd=local_host.parents[2], capture_output=True, check=True)
    need(hashlib.sha256(head.stdout).hexdigest() == BASE_EMBEDDINGS_SHA,
         "baseline_git_head_drift")
    bodies = {SELF.name: local_host.read_bytes(),
              WORKER.name: local_worker.read_bytes(),
              OVERRIDE.name: local_override.read_bytes()}
    config = {"root": str(ROOT),
              "files": {name: {"size": len(raw), "sha": hashlib.sha256(raw).hexdigest()}
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
