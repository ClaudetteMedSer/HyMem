"""Offline-only, one-shot frozen-candidate pytest harness.

prepare writes a code-only local upload directory. The remote actions never
SSH: a reviewer separately uploads and invokes them. No credential, capture,
production-store, or Docker-socket mount enters the test container.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "proof-pytest-v2"
SELF = ROOT / "claim_conflict_proof_pytest.py"
WORK = ROOT / "work"
OVERLAY = ROOT / "overlay"
PROOF_ROOT = BASE / "proof-replay-v3"
CANDIDATE = PROOF_ROOT / "candidate"
PROOF_HOST = PROOF_ROOT / "claim_conflict_proof_replay_v3_host.py"
PROOF_HOST_SHA = "a3ddbf16ce89c35f6f6a7f44c86a1089efbbc21abfab64dcc7d073b80772d92c"
PROOF_RESULT_SHA = "6e602bae89aee5f42be1f3f7efff87ba905a59bc134d09015ffe7a02a672c52d"
CANDIDATE_FILES = 481
CANDIDATE_SHA = "cd6e7810a4c08d771c49658a5954ca71a24d9d1119bcfde3aaa4ccb86dafd694"
BASELINE_TEST_FILES = 241
BASELINE_TEST_MODULES = 235
BASELINE_TEST_SHA = "5a7d98683d848ae69de2b2d7b5b6d62700db4bf115db663ba422effc2bed9f9b"
REVIEWED_OVERLAY_SHA = "978b687ae717fd4a5166b7710715105f4ab092dfab9ba6d19e886b9391b65247"
MIN_COLLECTED = 7641
SHARED_HOST = BASE / "shared-embedding-dream-v1/claim_conflict_shared_embedding_host.py"
SHARED_HOST_SHA = "ca95bc8d06cc1cc9f84e80dcb91f13cdc333c681efa4e9aca5ae457d48cce1f0"
FACT_FIX = Path("/private/tmp/hymem-shared-embedding-20260925.X8jEOF/candidate/tests/test_fact_authority.py")
TEST_ROOT = Path("/private/tmp/hymem-r7-proof-v64.kEJ30D/tests")
TEST_NAMES = (
    "test_fact_authority.py", "test_claim_semantic_dedup_guard.py",
    "test_alias_registration_idempotence.py", "test_alias_registration_root_controls.py",
    "test_claim_replay_binding_regressions.py", "test_claim_replay_proof_root.py",
    "test_claim_replay_local_proof_root.py", "test_shared_embedding_bounds.py",
    "test_embedding_bounds_root_controls.py", "test_chunk_embedding_batches.py",
    "test_chunk_embedding_batches_root.py", "test_claim_replay_r7_upgrade_root.py",
    "test_summary_recovery_v63.py", "test_r7_v64_summary_preservation.py",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
COLLECT_SECONDS, FULL_SECONDS, HOST_SECONDS = 300, 5000, 5400
MOUNT_CANDIDATE, MOUNT_OVERLAY = Path("/candidate"), Path("/overlay")
CONTAINER_WORK = Path("/work")
SOURCE_MANIFEST, TEST_MANIFEST = Path("/candidate-manifest.json"), Path("/test-overlay.json")


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    need(len(raw) <= 1048576, "receipt_too_large")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def inventory(root):
    need(root.is_dir() and not root.is_symlink(), "tree_missing")
    result = {}
    for path in root.rglob("*"):
        need(not path.is_symlink(), "tree_symlink")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def load_pinned(path, expected, name):
    need(path.is_file() and not path.is_symlink() and sha(path) == expected, "dependency_pin_drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    need(isinstance(PROOF_HOST_SHA, str) and HEX64.fullmatch(PROOF_HOST_SHA)
         and isinstance(PROOF_RESULT_SHA, str) and HEX64.fullmatch(PROOF_RESULT_SHA)
         and type(CANDIDATE_FILES) is int and CANDIDATE_FILES > 480
         and isinstance(CANDIDATE_SHA, str) and HEX64.fullmatch(CANDIDATE_SHA)
         and isinstance(REVIEWED_OVERLAY_SHA, str) and HEX64.fullmatch(REVIEWED_OVERLAY_SHA),
         "corrected_v64_release_not_reviewed")
    adapter = load_pinned(PROOF_HOST, PROOF_HOST_SHA, "pytest_v3_proof_adapter")
    proof = adapter.controller()
    shared = load_pinned(SHARED_HOST, SHARED_HOST_SHA, "pytest_shared_controller")
    return proof, shared, shared.helper()


def source_pins(shared, proof, helper):
    # Only code manifests are read. Do not call either controller's installed()
    # or pins(), which also inspects private replay captures/environment files.
    expected = shared.baseline_inventory(helper)
    expected.update(shared.OVERRIDE_SHAS)
    expected.update(proof.reviewed_overrides())
    need("hymem/core/migrations/064_local_claim_replay_proof.sql" in expected,
         "v64_migration_missing")
    baseline_tests = {
        name: value for name, value in expected.items() if name.startswith("tests/")
    }
    need(len(baseline_tests) == BASELINE_TEST_FILES
         and sum(name.startswith("tests/test_") and name.endswith(".py")
                 for name in baseline_tests) == BASELINE_TEST_MODULES
         and digest(baseline_tests) == BASELINE_TEST_SHA,
         "baseline_test_inventory_drift")
    need(len(expected) == CANDIDATE_FILES and digest(expected) == CANDIDATE_SHA
         and inventory(CANDIDATE) == expected, "candidate_pin_drift")
    return expected


def replay_gate(proof, helper):
    shared, alias, _ = proof.dependencies()
    receipt = proof.installed(shared, alias, helper)
    need(receipt.get("source_files") == CANDIDATE_FILES
         and receipt.get("host_sha256") == PROOF_HOST_SHA
         and receipt.get("worker_sha256") == proof.WORKER_SHA
         and receipt.get("phase1_sha256") == proof.reviewed_overrides()["hymem/dreaming/phase1.py"]
         and receipt.get("override_sha256") == digest(proof.reviewed_overrides()),
         "proof_install_identity_drift")
    need(receipt.get("candidate_sha256") == CANDIDATE_SHA
         and isinstance(receipt.get("audit_result_sha256"), str)
         and HEX64.fullmatch(receipt["audit_result_sha256"]),
         "proof_v3_receipt_drift")
    result_path = PROOF_ROOT / "result.json"
    need(isinstance(PROOF_RESULT_SHA, str) and HEX64.fullmatch(PROOF_RESULT_SHA)
         and result_path.is_file() and not result_path.is_symlink()
         and sha(result_path) == PROOF_RESULT_SHA, "proof_v3_result_pin_drift")
    result = helper.read_json(result_path)
    need(result.get("status") == "completed" and result.get("networked_runs_started") == 0,
         "proof_replay_not_completed")
    stage = result.get("stages", {}).get("replay", {})
    cid = stage.get("container_id")
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "proof_container_missing")
    state = proof.inspect(helper, cid, proof.configure(helper)[1])
    need(state["status"] == "exited" and state["pid"] == 0 and state["exit_code"] == 0
         and state["oom_killed"] is False, "proof_container_not_clean")
    proof.verdict(proof.project(stage.get("metadata")), receipt)
    return PROOF_RESULT_SHA


def prepare(destination, test_root):
    need(not destination.exists(), "output_already_exists")
    need(test_root == TEST_ROOT and test_root.is_dir() and not test_root.is_symlink(),
         "reviewed_test_root_required")
    sources = {}
    for name in TEST_NAMES:
        source = FACT_FIX if name == "test_fact_authority.py" else test_root / name
        need(source.is_file() and not source.is_symlink(), "test_source_missing")
        sources["tests/" + name] = sha(source)
    need(digest(sources) == REVIEWED_OVERLAY_SHA, "test_overlay_source_pin_drift")
    destination.mkdir(mode=0o700)
    files = {}
    for name in TEST_NAMES:
        source = FACT_FIX if name == "test_fact_authority.py" else test_root / name
        target = destination / "overlay/tests" / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o400)
        files["tests/" + name] = sha(target)
    need(files == sources, "test_overlay_copy_drift")
    shutil.copyfile(Path(__file__), destination / SELF.name)
    (destination / SELF.name).chmod(0o400)
    write_json(destination / "test-overlay.json", files)
    return {"status": "prepared_not_uploaded", "test_overrides": len(files),
            "overlay_sha256": digest(files), "controller_sha256": sha(destination / SELF.name)}


def pins(proof, shared, helper):
    need(os.geteuid() == 1000 and not ROOT.is_symlink()
         and stat.S_IMODE(ROOT.stat().st_mode) == 0o700, "stage_not_private")
    for path in (SELF, ROOT / "test-overlay.json"):
        helper.regular(path)
    need(stat.S_IMODE(SELF.stat().st_mode) == 0o400, "controller_not_sealed")
    overlay = helper.read_json(ROOT / "test-overlay.json")
    need(set(overlay) == {"tests/" + name for name in TEST_NAMES}
         and all(isinstance(value, str) and HEX64.fullmatch(value) for value in overlay.values())
         and inventory(OVERLAY) == overlay and digest(overlay) == REVIEWED_OVERLAY_SHA,
         "test_overlay_pin_drift")
    expected = source_pins(shared, proof, helper)
    return {"candidate_sha256": digest(expected), "overlay_sha256": digest(overlay),
            "host_sha256": sha(SELF), "proof_result_sha256": replay_gate(proof, helper)}


def configure(helper):
    mounts = [(str(CANDIDATE), "/candidate", False), (str(OVERLAY), "/overlay", False),
              (str(ROOT / "candidate-manifest.json"), "/candidate-manifest.json", False),
              (str(ROOT / "test-overlay.json"), "/test-overlay.json", False),
              (str(SELF), "/diag/runner.py", False), (str(WORK), "/work", True),
              (str(helper.RUNTIME), "/home/node/hymem-env", False)]
    command = ["docker", "create", "--name", "hymem-proof-pytest-v2", "--pull", "never",
               "--init", "--network", "none", "--user", "1000:1000", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "256",
               "--memory", "2g", "--cpus", "2", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
               "--env", "HOME=/work", "--env", "TMPDIR=/work", "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for src, dst, rw in mounts:
        command += ["--mount", "type=bind,src=" + src + ",dst=" + dst + ("" if rw else ",readonly")]
    command += ["--workdir", "/work", "--entrypoint", "/home/node/hymem-env/bin/python3",
                helper.IMAGE, "-I", "-B", "/diag/runner.py", "worker"]
    return command, mounts


def inspect(helper, cid):
    need(isinstance(cid, str) and HEX64.fullmatch(cid), "container_id_invalid")
    item = json.loads(helper.run(["docker", "inspect", cid], 30, "inspect"))[0]
    config, host, state = item["Config"], item["HostConfig"], item["State"]
    command, mounts = configure(helper)
    actual = {m["Destination"]: (m["Source"], m["RW"], m["Type"]) for m in item["Mounts"]}
    need(actual == {dst: (src, rw, "bind") for src, dst, rw in mounts}
         and item["Image"] == helper.IMAGE and config["Image"] == helper.IMAGE
         and config["User"] == "1000:1000" and config["WorkingDir"] == "/work"
         and config["Entrypoint"] == ["/home/node/hymem-env/bin/python3"]
         and config["Cmd"] == command[command.index(helper.IMAGE) + 1:]
         and host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True
         and host["Privileged"] is False and host["CapDrop"] == ["ALL"]
         and host["SecurityOpt"] == ["no-new-privileges"] and host["Init"] is True
         and host["Memory"] == 2147483648 and host["NanoCpus"] == 2000000000
         and host["PidsLimit"] == 256 and host["RestartPolicy"]["Name"] == "no"
         and host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,size=64m"}
         and not any(key.startswith(("HYMEM_", "OPENAI_", "DEEPSEEK_")) for key in config["Env"]),
         "container_configuration_drift")
    return {"container_id": cid, "status": state["Status"], "pid": state["Pid"],
            "exit_code": state["ExitCode"], "oom_killed": state["OOMKilled"]}


def clean_env():
    return {"PATH": "/home/node/hymem-env/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/work",
            "TMPDIR": "/work", "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


def project_worker(raw):
    need(isinstance(raw, dict) and raw.get("status") in ("collect_failed", "tests_failed", "passed"),
         "worker_status_invalid")
    result = {"status": raw["status"]}
    for name in ("candidate_sha256", "test_inventory_sha256"):
        value = raw.get(name)
        need(isinstance(value, str) and HEX64.fullmatch(value), "worker_hash_invalid")
        result[name] = value
    need(result["candidate_sha256"] == CANDIDATE_SHA, "worker_candidate_drift")
    for name in ("application_unchanged", "pinned_source_unchanged"):
        need(raw.get(name) is True, "worker_source_changed")
        result[name] = True
    runs = raw.get("full_runs_started")
    need(type(runs) is int and runs in (0, 1), "worker_run_count_invalid")
    result["full_runs_started"] = runs
    for mode in ("collect", "full") if runs else ("collect",):
        counts = raw.get(mode)
        need(isinstance(counts, dict), "worker_counts_missing")
        result[mode] = {}
        for name in ("collected", "passed", "failed", "skipped", "errors", "exit_code"):
            value = counts.get(name)
            need(type(value) is int and 0 <= value <= 1000000, "worker_counts_invalid")
            result[mode][name] = value
    if runs:
        need(result["collect"]["exit_code"] == 0
             and result["collect"]["collected"] >= MIN_COLLECTED
             and result["full"]["collected"] == result["collect"]["collected"], "worker_collection_invalid")
    if result["status"] == "passed":
        need(runs == 1 and all(result["full"][name] == 0 for name in ("exit_code", "failed", "errors")),
             "worker_pass_not_proven")
        need(result["full"]["passed"] + result["full"]["skipped"] == result["full"]["collected"],
             "worker_test_accounting_mismatch")
    return result


def pytest_stage(mode):
    need(mode in ("collect", "full"), "invalid_pytest_stage")
    tree = CONTAINER_WORK / "tree"
    sys.path.insert(0, str(tree))
    import hymem
    import pytest
    need(Path(hymem.__file__).resolve() == tree / "hymem/__init__.py", "wrong_application_import")
    class Counts:
        def __init__(self):
            self.counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        def pytest_runtest_logreport(self, report):
            if report.failed:
                self.counts["failed" if report.when == "call" else "errors"] += 1
            elif report.skipped:
                self.counts["skipped"] += 1
            elif report.when == "call" and report.passed:
                self.counts["passed"] += 1
        def pytest_sessionfinish(self, session, exitstatus):
            write_json(CONTAINER_WORK / (mode + "-counts.json"),
                       {**self.counts, "collected": session.testscollected, "exit_code": int(exitstatus)})
    args = ["-ra", "-p", "no:cacheprovider", "--basetemp=/work/pytest-" + mode,
            "--rootdir=/work/tree", "-c", "/work/tree/pyproject.toml", "tests"]
    if mode == "collect":
        args.insert(0, "--collect-only")
    return int(pytest.main(args, plugins=[Counts()]))


def run_pytest(mode):
    with (CONTAINER_WORK / (mode + ".log")).open("xb") as log:
        child = subprocess.Popen([sys.executable, "-I", "-B", "/diag/runner.py", "pytest-" + mode],
                                 cwd=str(CONTAINER_WORK / "tree"), env=clean_env(), stdout=log, stderr=log,
                                 start_new_session=True)
        try:
            return child.wait(timeout=COLLECT_SECONDS if mode == "collect" else FULL_SECONDS)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise


def worker():
    os.umask(0o077)
    expected = json.loads(SOURCE_MANIFEST.read_text())
    overlays = json.loads(TEST_MANIFEST.read_text())
    need(len(expected) == CANDIDATE_FILES and digest(expected) == CANDIDATE_SHA
         and inventory(MOUNT_CANDIDATE) == expected, "candidate_pin_drift")
    need(inventory(MOUNT_OVERLAY) == overlays and digest(overlays) == REVIEWED_OVERLAY_SHA
         and set(overlays) == {"tests/" + name for name in TEST_NAMES}, "overlay_pin_drift")
    tree = CONTAINER_WORK / "tree"
    shutil.copytree(MOUNT_CANDIDATE, tree)
    for name in overlays:
        target = tree / name
        if target.exists():
            target.chmod(0o600)
        shutil.copyfile(MOUNT_OVERLAY / name, target)
    wanted = {**expected, **overlays}
    need(inventory(tree) == wanted, "work_tree_pin_drift")
    write_json(CONTAINER_WORK / "test-inventory.json", {k: v for k, v in wanted.items() if k.startswith("tests/")})
    result = {"status": "collect_failed", "full_runs_started": 0,
              "candidate_sha256": digest(expected), "test_inventory_sha256": digest(
                  {k: v for k, v in wanted.items() if k.startswith("tests/")})}
    collect_rc = run_pytest("collect")
    result["collect"] = json.loads((CONTAINER_WORK / "collect-counts.json").read_text())
    if collect_rc == 0 and result["collect"]["exit_code"] == 0 and result["collect"]["collected"] >= MIN_COLLECTED:
        result["full_runs_started"] = 1
        rc = run_pytest("full")
        result["full"] = json.loads((CONTAINER_WORK / "full-counts.json").read_text())
        need(result["full"]["collected"] == result["collect"]["collected"], "collection_drift")
        result["status"] = "passed" if rc == 0 and result["full"]["exit_code"] == 0 else "tests_failed"
    need(inventory(MOUNT_CANDIDATE) == expected, "candidate_changed")
    need(all((tree / name).is_file() and not (tree / name).is_symlink()
             and sha(tree / name) == value for name, value in wanted.items()), "tested_source_changed")
    need({name: value for name, value in inventory(tree).items() if name.startswith("hymem/")}
         == {name: value for name, value in expected.items() if name.startswith("hymem/")},
         "tested_application_changed")
    result["application_unchanged"] = True
    result["pinned_source_unchanged"] = True
    write_json(CONTAINER_WORK / "worker-result.json", result)
    return result


def remote(action):
    proof, shared, helper = dependencies()
    current = pins(proof, shared, helper)
    if action == "remote-install":
        WORK.mkdir(mode=0o700)
        write_json(ROOT / "candidate-manifest.json", source_pins(shared, proof, helper))
        write_json(ROOT / "install.json", current)
        return {"status": "installed_not_launched", **current}
    need(helper.read_json(ROOT / "install.json") == current, "installed_pin_drift")
    need(helper.read_json(ROOT / "candidate-manifest.json") == source_pins(shared, proof, helper),
         "installed_manifest_drift")
    need(not WORK.is_symlink() and stat.S_IMODE(WORK.stat().st_mode) == 0o700, "work_not_private")
    if action == "remote-launch":
        write_json(ROOT / "launch-intent.json", {"max_runs": 1, "network": "none"})
        child = subprocess.Popen([sys.executable, "-I", "-B", str(SELF), "supervise"],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        return {"status": "detached_supervisor_started", "pid": child.pid}
    if action == "remote-status":
        return helper.read_json(ROOT / "result.json") if (ROOT / "result.json").exists() else {"status": "not_terminal"}
    write_json(ROOT / "supervisor-intent.json", {"max_containers": 1, "timeout_seconds": HOST_SECONDS})
    result = {"status": "failed", "networked_runs_started": 0}
    cid = None
    try:
        command, _ = configure(helper)
        write_json(ROOT / "create-intent.json", {"max_containers": 1})
        created = helper.run(command, 60, "create").decode().strip()
        need(HEX64.fullmatch(created), "container_id_invalid")
        cid = created
        write_json(ROOT / "container.json", {"container_id": cid})
        need(inspect(helper, cid)["status"] == "created", "container_not_created")
        write_json(ROOT / "start-intent.json", {"container_id": cid})
        need(helper.run(["docker", "start", cid], 60, "start").decode().strip() == cid, "start_identity")
        raw = helper.run(["docker", "wait", cid], HOST_SECONDS, "wait")
        need(re.fullmatch(rb"[0-9]{1,3}\n?", raw), "wait_shape")
        state = inspect(helper, cid)
        need(state["status"] == "exited" and state["pid"] == 0 and state["exit_code"] == int(raw)
             and not state["oom_killed"], "container_not_clean")
        result["container"] = state
        result["metadata"] = project_worker(helper.read_json(WORK / "worker-result.json"))
        need(state["exit_code"] == 0 and result["metadata"]["status"] == "passed"
             and result["metadata"]["application_unchanged"] is True, "pytest_not_passed")
        need(pins(proof, shared, helper) == current, "final_pin_drift")
        result["status"] = "passed"
    except BaseException:
        if cid is not None:
            subprocess.run(["docker", "stop", "--time", "10", cid], capture_output=True, timeout=30)
            state = inspect(helper, cid)
            need(state["status"] in ("created", "exited") and state["pid"] == 0, "cleanup_unverified")
            result["container"] = state
    write_json(ROOT / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "remote-install", "remote-launch", "remote-status",
                                           "supervise", "worker", "pytest-collect", "pytest-full"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tests-root", type=Path,
                        help="Reviewed rebased v64 test snapshot directory; never the stale working tree")
    args = parser.parse_args()
    if args.action.startswith("pytest-"):
        return pytest_stage(args.action.removeprefix("pytest-"))
    try:
        if args.action == "prepare":
            need(args.output is not None and args.tests_root is not None, "output_and_test_root_required")
            result = prepare(args.output, args.tests_root)
        elif args.action == "worker":
            result = worker()
        else:
            result = remote(args.action)
    except BaseException:
        result = {"status": "failed_inspect_private_artifacts"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("prepared_not_uploaded", "installed_not_launched",
                                     "detached_supervisor_started", "passed", "not_terminal") else 1


if __name__ == "__main__":
    raise SystemExit(main())
