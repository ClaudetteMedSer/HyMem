"""Network-free controls for the v3-gated frozen-release pytest harness."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_proof_pytest as runner


def test_unsealed_v3_result_cannot_be_loaded_or_launched(monkeypatch):
    monkeypatch.setattr(runner, "PROOF_RESULT_SHA", None)
    monkeypatch.setattr(runner, "load_pinned", lambda *_: pytest.fail("loaded obsolete controller"))
    with pytest.raises(RuntimeError, match="corrected_v64_release_not_reviewed"):
        runner.dependencies()


def test_sealed_dependencies_load_only_v3_adapter(monkeypatch):
    proof, helper = object(), object()
    adapter = SimpleNamespace(controller=lambda: proof)
    shared = SimpleNamespace(helper=lambda: helper)
    calls = []
    def load(path, expected, name):
        calls.append((path, expected, name))
        return adapter if path == runner.PROOF_HOST else shared
    monkeypatch.setattr(runner, "PROOF_HOST_SHA", "a" * 64)
    monkeypatch.setattr(runner, "PROOF_RESULT_SHA", "b" * 64)
    monkeypatch.setattr(runner, "load_pinned", load)
    assert runner.dependencies() == (proof, shared, helper)
    assert calls == [
        (runner.PROOF_HOST, "a" * 64, "pytest_v3_proof_adapter"),
        (runner.SHARED_HOST, runner.SHARED_HOST_SHA, "pytest_shared_controller"),
    ]


def test_reviewed_overlay_is_exact_staged_fourteen_file_manifest():
    files = {}
    for name in runner.TEST_NAMES:
        path = runner.FACT_FIX if name == "test_fact_authority.py" else runner.TEST_ROOT / name
        assert path.is_file() and not path.is_symlink()
        files["tests/" + name] = runner.sha(path)
    assert len(files) == 14
    assert runner.digest(files) == runner.REVIEWED_OVERLAY_SHA
    assert runner.CANDIDATE_FILES == 481
    assert runner.CANDIDATE_SHA == "cd6e7810a4c08d771c49658a5954ca71a24d9d1119bcfde3aaa4ccb86dafd694"
    assert runner.ROOT.name == "proof-pytest-v2"
    assert runner.BASELINE_TEST_FILES == 241
    assert runner.BASELINE_TEST_MODULES == 235
    assert runner.BASELINE_TEST_SHA == "5a7d98683d848ae69de2b2d7b5b6d62700db4bf115db663ba422effc2bed9f9b"


def test_source_pins_require_exact_241_file_baseline_test_manifest(monkeypatch):
    test_files = {"tests/test_%03d.py" % index: "a" * 64 for index in range(235)}
    test_files.update({"tests/support_%d.py" % index: "b" * 64 for index in range(6)})
    source = {**test_files, **{
        "hymem/core/migrations/064_local_claim_replay_proof.sql": "c" * 64,
        **{"hymem/source_%03d.py" % index: "d" * 64 for index in range(239)},
    }}
    shared = SimpleNamespace(baseline_inventory=lambda _: source.copy(), OVERRIDE_SHAS={})
    proof = SimpleNamespace(reviewed_overrides=lambda: {})
    monkeypatch.setattr(runner, "inventory", lambda _: source)
    monkeypatch.setattr(runner, "CANDIDATE_SHA", runner.digest(source))
    monkeypatch.setattr(runner, "BASELINE_TEST_SHA", runner.digest(test_files))
    assert runner.source_pins(shared, proof, object()) == source
    monkeypatch.setattr(runner, "BASELINE_TEST_SHA", "0" * 64)
    with pytest.raises(RuntimeError, match="baseline_test_inventory_drift"):
        runner.source_pins(shared, proof, object())


def test_preparation_uses_only_explicit_tests_and_frozen_fact_override(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    source = repo / "tools/diagnostics/runner.py"
    source.parent.mkdir(parents=True)
    source.write_text("# controller\n")
    (repo / "tests").mkdir()
    for name in runner.TEST_NAMES:
        (repo / "tests" / name).write_text("# selected " + name)
    (repo / "tests/conftest.py").write_text("# do not overlay")
    frozen = tmp_path / "frozen_fact.py"
    frozen.write_text("# reviewed fact-only assertion change")
    monkeypatch.setattr(runner, "__file__", str(source))
    monkeypatch.setattr(runner, "FACT_FIX", frozen)
    monkeypatch.setattr(runner, "TEST_ROOT", repo / "tests")
    reviewed = {
        "tests/" + name: runner.sha(
            frozen if name == "test_fact_authority.py" else repo / "tests" / name
        )
        for name in runner.TEST_NAMES
    }
    monkeypatch.setattr(runner, "REVIEWED_OVERLAY_SHA", runner.digest(reviewed))
    output = tmp_path / "prepared"
    result = runner.prepare(output, repo / "tests")
    actual = runner.inventory(output / "overlay")
    assert result["test_overrides"] == 14
    assert set(actual) == {"tests/" + name for name in runner.TEST_NAMES}
    assert actual["tests/test_fact_authority.py"] == runner.sha(frozen)
    assert not (output / "overlay/tests/conftest.py").exists()
    assert not (output / "overlay/hymem").exists()
    assert result["overlay_sha256"] == runner.digest(actual)
    with pytest.raises(RuntimeError, match="output_already_exists"):
        runner.prepare(output, repo / "tests")
    with pytest.raises(RuntimeError, match="reviewed_test_root_required"):
        runner.prepare(tmp_path / "wrong", tmp_path / "tests")
    (repo / "tests/test_claim_replay_proof_root.py").write_text("# changed")
    with pytest.raises(RuntimeError, match="test_overlay_source_pin_drift"):
        runner.prepare(tmp_path / "changed", repo / "tests")
    assert not (tmp_path / "changed").exists()


def test_inventory_rejects_symbolic_links(tmp_path):
    (tmp_path / "source").write_text("source")
    (tmp_path / "alias").symlink_to(tmp_path / "source")
    with pytest.raises(RuntimeError, match="tree_symlink"):
        runner.inventory(tmp_path)


def _docker_fixture():
    helper = SimpleNamespace(RUNTIME=Path("/runtime"), IMAGE="sha256:" + "a" * 64)
    command, mounts = runner.configure(helper)
    item = {
        "Image": helper.IMAGE,
        "Config": {"Image": helper.IMAGE, "User": "1000:1000", "WorkingDir": "/work",
                   "Entrypoint": ["/home/node/hymem-env/bin/python3"],
                   "Cmd": command[command.index(helper.IMAGE) + 1:],
                   "Env": ["HOME=/work", "TMPDIR=/work", "PYTHONDONTWRITEBYTECODE=1"]},
        "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
                       "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"], "Init": True,
                       "Memory": 2147483648, "NanoCpus": 2000000000, "PidsLimit": 256,
                       "RestartPolicy": {"Name": "no"},
                       "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"}},
        "State": {"Status": "created", "Pid": 0, "ExitCode": 0, "OOMKilled": False},
        "Mounts": [{"Source": src, "Destination": dst, "RW": rw, "Type": "bind"}
                   for src, dst, rw in mounts],
    }
    helper.run = lambda *_: json.dumps([item]).encode()
    return helper, item, command, mounts


def test_container_has_only_private_work_writable_and_no_live_inputs():
    helper, _item, command, mounts = _docker_fixture()
    assert [(src, dst) for src, dst, rw in mounts if rw] == [(str(runner.WORK), "/work")]
    assert {dst for _, dst, _ in mounts} == {
        "/candidate", "/overlay", "/candidate-manifest.json", "/test-overlay.json",
        "/diag/runner.py", "/work", "/home/node/hymem-env",
    }
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--name") + 1] == "hymem-proof-pytest-v2"
    assert command[-4:] == ["-I", "-B", "/diag/runner.py", "worker"]
    assert runner.inspect(helper, "b" * 64)["status"] == "created"
    assert not any("KEY" in name or name.startswith("HYMEM_") for name in runner.clean_env())


@pytest.mark.parametrize("mutation", ["network", "mount", "writable_source", "command", "credential", "memory"])
def test_inspector_rejects_unsafe_container_drift(mutation):
    helper, item, _, _ = _docker_fixture()
    if mutation == "network":
        item["HostConfig"]["NetworkMode"] = "bridge"
    elif mutation == "mount":
        item["Mounts"].append({"Source": "/production", "Destination": "/prod", "RW": False, "Type": "bind"})
    elif mutation == "writable_source":
        item["Mounts"][0]["RW"] = True
    elif mutation == "command":
        item["Config"]["Cmd"][-1] = "other"
    elif mutation == "credential":
        item["Config"]["Env"].append("OPENAI_API_KEY=synthetic")
    else:
        item["HostConfig"]["Memory"] *= 2
    with pytest.raises(RuntimeError, match="container_configuration_drift"):
        runner.inspect(helper, "b" * 64)


def _worker_fixture(monkeypatch, tmp_path):
    source, overlay, work = (tmp_path / name for name in ("candidate", "overlay", "work"))
    for path in (source / "hymem", source / "tests", overlay / "tests", work):
        path.mkdir(parents=True)
    (source / "hymem/__init__.py").write_text("# pinned application")
    (source / "tests/conftest.py").write_text("# pinned fixtures")
    (source / "pyproject.toml").write_text("# pinned config")
    for name in runner.TEST_NAMES:
        (overlay / "tests" / name).write_text("# reviewed " + name)
    expected, overlays = runner.inventory(source), runner.inventory(overlay)
    source_manifest, test_manifest = tmp_path / "source.json", tmp_path / "tests.json"
    source_manifest.write_text(json.dumps(expected))
    test_manifest.write_text(json.dumps(overlays))
    for key, value in {
        "MOUNT_CANDIDATE": source, "MOUNT_OVERLAY": overlay, "CONTAINER_WORK": work,
        "SOURCE_MANIFEST": source_manifest, "TEST_MANIFEST": test_manifest,
        "CANDIDATE_FILES": len(expected), "CANDIDATE_SHA": runner.digest(expected),
        "REVIEWED_OVERLAY_SHA": runner.digest(overlays), "MIN_COLLECTED": 10,
    }.items():
        monkeypatch.setattr(runner, key, value)
    return source, work


@pytest.mark.parametrize("collect_exit,count,full_exit,expected_calls,status", [
    (1, 10, 0, ["collect"], "collect_failed"),
    (0, 9, 0, ["collect"], "collect_failed"),
    (0, 10, 0, ["collect", "full"], "passed"),
    (0, 10, 1, ["collect", "full"], "tests_failed"),
])
def test_worker_collects_before_one_full_run(monkeypatch, tmp_path, collect_exit, count,
                                            full_exit, expected_calls, status):
    source, work = _worker_fixture(monkeypatch, tmp_path)
    before, calls = runner.inventory(source), []
    def run(mode):
        calls.append(mode)
        code = collect_exit if mode == "collect" else full_exit
        runner.write_json(work / (mode + "-counts.json"), {"collected": count, "exit_code": code})
        return code
    monkeypatch.setattr(runner, "run_pytest", run)
    result = runner.worker()
    assert calls == expected_calls
    assert result["status"] == status
    assert result["full_runs_started"] == len(calls) - 1
    assert result["application_unchanged"] is True
    assert result["pinned_source_unchanged"] is True
    assert runner.inventory(source) == before
    manifest = json.loads((work / "test-inventory.json").read_text())
    assert manifest["tests/conftest.py"] == before["tests/conftest.py"]


@pytest.mark.parametrize("changed", ["hymem/__init__.py", "tests/conftest.py", "pyproject.toml"])
def test_worker_refuses_changed_pinned_application_tests_or_config(monkeypatch, tmp_path, changed):
    _, work = _worker_fixture(monkeypatch, tmp_path)
    def run(mode):
        runner.write_json(work / (mode + "-counts.json"), {"collected": 10, "exit_code": 0})
        if mode == "full":
            (work / "tree" / changed).write_text("changed")
        return 0
    monkeypatch.setattr(runner, "run_pytest", run)
    with pytest.raises(RuntimeError, match="tested_source_changed"):
        runner.worker()
    assert not (work / "worker-result.json").exists()


def _replay_fixture(monkeypatch, tmp_path):
    overrides = {"hymem/dreaming/phase1.py": "a" * 64}
    receipt = {"source_files": 499, "host_sha256": "b" * 64, "worker_sha256": "c" * 64,
               "phase1_sha256": "a" * 64, "override_sha256": runner.digest(overrides),
               "candidate_sha256": "e" * 64, "audit_result_sha256": "0" * 64}
    result = {"status": "completed", "networked_runs_started": 0,
              "stages": {"replay": {"container_id": "d" * 64, "metadata": {"valid": True}}}}
    state = {"status": "exited", "pid": 0, "exit_code": 0, "oom_killed": False}
    def verdict(metadata, actual_receipt):
        assert actual_receipt == receipt
        runner.need(metadata == {"valid": True}, "verdict_failed")
    proof = SimpleNamespace(WORKER_SHA="c" * 64, reviewed_overrides=lambda: overrides,
                            inspect=lambda *_: state, configure=lambda *_: ([], []),
                            project=lambda raw: raw, verdict=verdict,
                            dependencies=lambda: (None, None, None),
                            installed=lambda *_: receipt)
    helper = SimpleNamespace(read_json=lambda _: result)
    proof_root = tmp_path / "proof-replay-v3"
    proof_root.mkdir()
    (proof_root / "result.json").write_text("{}")
    monkeypatch.setattr(runner, "CANDIDATE_FILES", 499)
    monkeypatch.setattr(runner, "CANDIDATE_SHA", "e" * 64)
    monkeypatch.setattr(runner, "PROOF_HOST_SHA", "b" * 64)
    monkeypatch.setattr(runner, "PROOF_RESULT_SHA", "f" * 64)
    monkeypatch.setattr(runner, "PROOF_ROOT", proof_root)
    monkeypatch.setattr(runner, "sha", lambda _: "f" * 64)
    return proof, helper, receipt, result, state


def test_replay_gate_checks_reviewed_source_identity_and_actual_terminal_state(monkeypatch, tmp_path):
    proof, helper, *_ = _replay_fixture(monkeypatch, tmp_path)
    assert runner.replay_gate(proof, helper) == "f" * 64


@pytest.mark.parametrize("mutation", ["source", "host", "worker", "overrides",
                                     "candidate", "audit", "result_pin", "audit_only",
                                     "failed", "network", "running", "oom", "verdict"])
def test_replay_gate_rejects_incomplete_or_different_release(monkeypatch, tmp_path, mutation):
    proof, helper, receipt, result, state = _replay_fixture(monkeypatch, tmp_path)
    if mutation in {"source", "host", "worker", "overrides", "candidate", "audit"}:
        key = {"source": "source_files", "host": "host_sha256",
               "worker": "worker_sha256", "overrides": "override_sha256",
               "candidate": "candidate_sha256",
               "audit": "audit_result_sha256"}[mutation]
        receipt[key] = 481 if mutation == "source" else "0" * 64
        if mutation == "audit":
            receipt[key] = "not-a-hash"
    elif mutation == "result_pin":
        monkeypatch.setattr(runner, "PROOF_RESULT_SHA", "0" * 64)
    elif mutation == "audit_only":
        result["stages"] = {"audit": result["stages"]["replay"]}
    elif mutation == "failed":
        result["status"] = "failed"
    elif mutation == "network":
        result["networked_runs_started"] = 1
    elif mutation == "running":
        state.update(status="running", pid=123)
    elif mutation == "oom":
        state["oom_killed"] = True
    else:
        result["stages"]["replay"]["metadata"] = {"valid": False}
    with pytest.raises(RuntimeError):
        runner.replay_gate(proof, helper)


def _worker_result(monkeypatch):
    monkeypatch.setattr(runner, "CANDIDATE_SHA", "a" * 64)
    counts = {"collected": 7641, "passed": 7641, "failed": 0, "errors": 0, "skipped": 0, "exit_code": 0}
    return {"status": "passed", "candidate_sha256": "a" * 64, "test_inventory_sha256": "b" * 64,
            "application_unchanged": True, "pinned_source_unchanged": True, "full_runs_started": 1,
            "collect": {**counts, "passed": 0}, "full": counts}


def test_worker_projection_exports_only_static_counts_and_hashes(monkeypatch):
    raw = _worker_result(monkeypatch)
    expected = deepcopy(raw)
    raw["raw_error"] = "must not be exported"
    raw["full"]["raw_test_output"] = "must not be exported"
    assert runner.project_worker(raw) == expected


@pytest.mark.parametrize("mutation", ["missing", "negative", "bool", "small_collection", "changed_collection", "failed", "no_run", "wrong_candidate", "unaccounted"])
def test_worker_projection_cannot_claim_unproven_pass(monkeypatch, mutation):
    raw = _worker_result(monkeypatch)
    if mutation == "missing":
        del raw["full"]["errors"]
    elif mutation == "negative":
        raw["full"]["passed"] = -1
    elif mutation == "bool":
        raw["full_runs_started"] = True
    elif mutation == "small_collection":
        raw["collect"]["collected"] = 1
    elif mutation == "changed_collection":
        raw["full"]["collected"] -= 1
    elif mutation == "failed":
        raw["full"]["failed"] = 1
    elif mutation == "no_run":
        raw["full_runs_started"] = 0
    elif mutation == "wrong_candidate":
        raw["candidate_sha256"] = "c" * 64
    else:
        raw["full"]["passed"] -= 1
    with pytest.raises(RuntimeError):
        runner.project_worker(raw)


def test_pytest_timeout_terminates_then_kills_only_owned_process_group(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "CONTAINER_WORK", tmp_path)
    waits, signals, launches = [], [], []
    def wait(timeout):
        waits.append(timeout)
        if len(waits) < 3:
            raise subprocess.TimeoutExpired("synthetic", timeout)
        return -9
    child = SimpleNamespace(pid=12345, wait=wait)
    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return child
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_pytest("full")
    assert waits == [5000, 10, 10]
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert len(launches) == 1
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["env"] == runner.clean_env()


def _remote_fixture(monkeypatch, tmp_path):
    root, work = tmp_path / "stage", tmp_path / "stage/work"
    work.mkdir(mode=0o700, parents=True)
    source, current = {"hymem/__init__.py": "a" * 64}, {"sealed": True}
    (root / "install.json").write_text(json.dumps(current))
    (root / "candidate-manifest.json").write_text(json.dumps(source))
    helper = SimpleNamespace(read_json=lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "WORK", work)
    monkeypatch.setattr(runner, "dependencies", lambda: (None, None, helper))
    monkeypatch.setattr(runner, "pins", lambda *_: current)
    monkeypatch.setattr(runner, "source_pins", lambda *_: source)
    return helper, root


def test_durable_launch_intent_prevents_second_fullsuite_supervisor(monkeypatch, tmp_path):
    _, root = _remote_fixture(monkeypatch, tmp_path)
    launches = []
    def popen(*args, **kwargs):
        launches.append((args, kwargs))
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    assert runner.remote("remote-launch")["status"] == "detached_supervisor_started"
    with pytest.raises(FileExistsError):
        runner.remote("remote-launch")
    assert len(launches) == 1
    assert (root / "launch-intent.json").exists()


@pytest.mark.parametrize("create_output,stop_count", [(b"invalid-container", 0), (b"a" * 64, 1)])
def test_failed_supervision_cleans_only_a_valid_owned_container(monkeypatch, tmp_path, create_output, stop_count):
    helper, root = _remote_fixture(monkeypatch, tmp_path)
    helper.run = lambda *_: create_output
    monkeypatch.setattr(runner, "configure", lambda *_: (["create"], []))
    stops, inspections = [], []
    def inspect(*_):
        inspections.append(True)
        return {"status": "running" if len(inspections) == 1 else "exited", "pid": 0}
    monkeypatch.setattr(runner, "inspect", inspect)
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **_: stops.append(command))
    assert runner.remote("supervise")["status"] == "failed"
    assert len(stops) == stop_count
    if stops:
        assert stops == [["docker", "stop", "--time", "10", "a" * 64]]
    assert json.loads((root / "result.json").read_text())["networked_runs_started"] == 0
