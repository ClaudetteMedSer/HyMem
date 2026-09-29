"""Independent offline admission/cleanup controls; never invoke Docker or SSH."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_shared_embedding_host as host


def _worker_module():
    path = Path(__file__).parents[1] / "claim_conflict_instrumented_dream.py"
    spec = importlib.util.spec_from_file_location("shared_bounds_root_worker", path)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


@pytest.mark.parametrize("exception", [
    ValueError(), RuntimeError(), TypeError(), KeyError(), AssertionError(),
    TimeoutError(), ConnectionError(), OSError(), MemoryError(),
])
def test_every_worker_static_exception_type_survives_host_projection(exception):
    worker = _worker_module()
    code = worker.safe_type(exception)
    projected = host.project({"status": "captured_failure", "error_type": code})
    assert projected["error_type"] == code


@pytest.mark.parametrize("arguments", [
    ["--max-http-attempts", "705"],
    ["--max-llm-http-attempts", "193"],
    ["--max-embedding-http-attempts", "513"],
    ["--max-http-attempts", "0"],
    ["--max-http-attempts", "704", "--max-llm-http-attempts", "1", "--max-embedding-http-attempts", "1"],
    ["--deadline-seconds", "1801"],
    ["--phase1-sha256", "not-a-reviewed-sha"],
])
def test_worker_rejects_invalid_limits_before_setup_or_any_dispatch(monkeypatch, arguments):
    worker = _worker_module()
    reached = []
    monkeypatch.setattr(worker, "live", lambda *a, **k: reached.append(True))
    monkeypatch.setattr(worker, "WORK", Path("/must-not-inspect-work"))
    monkeypatch.setattr(worker.logging, "disable", lambda *_: None)
    monkeypatch.setattr(worker.os, "umask", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["worker", "live", *arguments])
    with pytest.raises(ValueError, match="diagnostic_limits_invalid"):
        worker.main()
    assert reached == []


def _inspector_fixture(mode="live"):
    helper = SimpleNamespace(
        RUNTIME=Path("/approved/runtime"), RUNTIME_ENV=Path("/approved/env.json"),
        IMAGE="sha256:" + "c" * 64,
    )
    phase1_sha = "d" * 64
    command, mounts = host.configure(helper, mode, phase1_sha)
    item = {
        "Image": helper.IMAGE,
        "Config": {
            "Image": helper.IMAGE, "User": "1000:1000",
            "Entrypoint": ["/home/node/hymem-env/bin/python3"],
            "WorkingDir": "/candidate",
            "Cmd": command[command.index(helper.IMAGE) + 1:],
            "Env": ["HOME=/tmp", "TMPDIR=/tmp", "PYTHONDONTWRITEBYTECODE=1"],
        },
        "HostConfig": {
            "NetworkMode": "hermes-net" if mode == "live" else "none",
            "ReadonlyRootfs": True, "Privileged": False, "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"], "Init": True,
            "Memory": 2147483648, "NanoCpus": 2000000000, "PidsLimit": 128,
            "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
            "RestartPolicy": {"Name": "no"},
        },
        "State": {"Status": "created", "ExitCode": 0, "OOMKilled": False, "Pid": 0},
        "Mounts": [{"Source": src, "Destination": dst, "RW": rw, "Type": "bind"}
                   for src, dst, rw in mounts],
    }
    return helper, phase1_sha, mounts, item


@pytest.mark.parametrize("mode", ["offline", "live"])
def test_inspector_accepts_only_configured_isolation(mode):
    helper, phase1_sha, mounts, item = _inspector_fixture(mode)
    helper.run = lambda *_: json.dumps([item]).encode()
    assert host.inspect(helper, "a" * 64, mode, mounts, phase1_sha)["configuration_verified"]
    assert [dst for _src, dst, rw in mounts if rw] == ["/work"]


@pytest.mark.parametrize("flag,unsafe", [
    ("--max-http-attempts", "705"), ("--max-llm-http-attempts", "193"),
    ("--max-embedding-http-attempts", "513"), ("--deadline-seconds", "1801"),
    ("--phase1-sha256", "e" * 64),
])
def test_inspector_rejects_any_budget_or_identity_argument_drift(flag, unsafe):
    helper, phase1_sha, mounts, item = _inspector_fixture()
    item["Config"]["Cmd"][item["Config"]["Cmd"].index(flag) + 1] = unsafe
    helper.run = lambda *_: json.dumps([item]).encode()
    with pytest.raises(RuntimeError, match="args_drift"):
        host.inspect(helper, "a" * 64, "live", mounts, phase1_sha)


@pytest.mark.parametrize("mutation", ["writable_reference", "extra_mount", "host_network"])
def test_inspector_rejects_mount_or_network_expansion(mutation):
    helper, phase1_sha, mounts, item = _inspector_fixture()
    if mutation == "writable_reference":
        next(mount for mount in item["Mounts"] if mount["Destination"] == "/reference/source.sqlite")["RW"] = True
    elif mutation == "extra_mount":
        item["Mounts"].append({"Source": "/production", "Destination": "/production", "RW": True, "Type": "bind"})
    else:
        item["HostConfig"]["NetworkMode"] = "host"
    helper.run = lambda *_: json.dumps([item]).encode()
    with pytest.raises(RuntimeError):
        host.inspect(helper, "a" * 64, "live", mounts, phase1_sha)


def test_wait_failure_stops_exact_live_container_and_does_not_authorize_retry(monkeypatch):
    helper, phase1_sha, _mounts, _item = _inspector_fixture()
    records, starts, stopped = {}, [], []

    def put_json(path, value):
        if path in records:
            raise FileExistsError("durable intent exists")
        records[path] = deepcopy(value)

    def run(command, timeout, _code):
        action = command[1]
        if action == "create":
            mode = "offline" if command[command.index("--network") + 1] == "none" else "live"
            return (("a" if mode == "offline" else "b") * 64).encode()
        cid = command[2]
        if action == "start":
            starts.append(cid)
            return cid.encode()
        if action == "wait":
            if cid == "b" * 64:
                assert timeout == 1860
                raise RuntimeError("synthetic host wait failure")
            return b"0\n"
        assert action == "logs"
        return json.dumps({
            "status": "ready", "cleanup_ok": True, "source_unchanged": True,
            "runtime_generation_verified": True, "source_sha256": host.REFERENCE_SHA,
            "phase1_sha256": phase1_sha, "completion_calls": 0, "http_attempts": 0,
        }).encode()

    helper.put_json, helper.run = put_json, run
    monkeypatch.setattr(host, "installed", lambda _: {"phase1_sha256": phase1_sha})
    monkeypatch.setattr(host, "inspect", lambda _h, cid, *args: {
        "status": "exited" if cid in starts else "created", "pid": 0,
        "oom_killed": False, "exit_code": 0,
    })
    monkeypatch.setattr(host, "stop", lambda _h, cid, *args: stopped.append(cid))
    host.supervise(helper)
    assert stopped == ["b" * 64]
    assert starts == ["a" * 64, "b" * 64]
    result = records[host.ROOT / "result.json"]
    assert result["status"] == "failed"
    assert result["live_runs_started"] == 1
    with pytest.raises(FileExistsError):
        host.supervise(helper)
    assert starts == ["a" * 64, "b" * 64]
