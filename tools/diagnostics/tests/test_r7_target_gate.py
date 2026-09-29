"""No-SSH controls for the isolated target packaging and installer boundary."""
import ast
import dataclasses
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tarfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location(
        "r7_target_under_test", Path(__file__).parents[1] / "lme_r7_target_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedded_commands_compile_without_execution(gate):
    for name in ("INSTALL", "REMOTE_DOCKER", "SUPERVISOR"):
        compile(getattr(gate, name), name, "exec")


def test_container_plan_has_only_one_writable_mount_and_no_network(gate, monkeypatch, tmp_path):
    path = tmp_path / "manifest.json"
    raw = json.dumps({"expected_nodeids": [gate.FOCUS[0] + "::test_one"],
                      "source_sha256": {"hymem/a.py": "a"},
                      "test_sha256": {gate.FOCUS[0]: "b"},
                      "auxiliary_sha256": {"README.md": "c"}}).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(gate, "MANIFEST", path)
    monkeypatch.setattr(gate, "MANIFEST_PIN", gate.sha(raw))
    cfg = gate.config("create")
    assert cfg["selected"] == 1 and cfg["files"] == 3
    assert [(target, origin) for target, origin, writable in cfg["mounts"] if writable] == [
        ("/results", gate.REMOTE_ROOT + "/results")]
    assert cfg["command"][cfg["command"].index("--network") + 1] == "none"
    assert "--read-only" in cfg["command"]
    assert all(".env" not in origin for _, origin, _ in cfg["mounts"])
    assert cfg["container_args"][-2] == gate.DATA_PIN
    assert json.loads(cfg["container_args"][-1])["supervised_invocation.py"] == gate.HELPERS["supervised_invocation.py"]
    monkeypatch.setattr(gate, "MANIFEST_PIN", "0" * 64)
    with pytest.raises(AssertionError):
        gate.config("create")


@pytest.fixture
def installer(gate, monkeypatch, tmp_path):
    raw_manifest = json.dumps({"source_sha256": {"hymem/a.py": gate.sha(b"source")},
        "test_sha256": {"tests/test_a.py": gate.sha(b"test")},
        "auxiliary_sha256": {"README.md": gate.sha(b"readme")}}).encode()
    files = {"diag/manifest.json": raw_manifest, "diag/lme_offline_gate.py": b"gate",
        "diag/r6-manifest.json": b"baseline", "diag/helper.py": b"helper",
        "verification/hymem/a.py": b"source", "verification/tests/test_a.py": b"test",
        "verification/README.md": b"readme"}
    root = tmp_path / "remote-install"
    cfg = {"root": str(root), "manifest_pin": gate.sha(raw_manifest),
        "gate_pin": gate.sha(b"gate"), "baseline_pin": gate.sha(b"baseline"),
        "helpers": {"helper.py": gate.sha(b"helper")}, "files": 3}

    def install(mutation=None):
        entries = [(name, content, tarfile.REGTYPE) for name, content in files.items()]
        if mutation:
            mutation(entries)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as writer:
            for name, content, kind in entries:
                member = tarfile.TarInfo(name)
                member.type, member.size = kind, len(content)
                writer.addfile(member, io.BytesIO(content))
        raw = buffer.getvalue()
        cfg["archive_sha256"] = hashlib.sha256(raw).hexdigest()
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw)))
        monkeypatch.setattr(gate.os, "geteuid", lambda: 1000)
        exec(compile(gate.INSTALL, "remote-installer", "exec"), {"C": cfg})
    return root, install


@pytest.mark.parametrize("defect", ["extra", "duplicate", "traversal", "symlink", "hash"])
def test_installer_rejects_unsealed_archive_before_writes(installer, defect):
    root, install = installer
    def mutate(entries):
        if defect == "extra": entries.append(("extra.py", b"extra", tarfile.REGTYPE))
        elif defect == "duplicate": entries.append(entries[0])
        elif defect == "traversal": entries.append(("../escaped", b"extra", tarfile.REGTYPE))
        elif defect == "symlink": entries.append(("alias", b"", tarfile.SYMTYPE))
        else: entries[-1] = (entries[-1][0], b"changed", tarfile.REGTYPE)
    with pytest.raises(AssertionError):
        install(mutate)
    assert not root.exists()


def test_installer_is_exclusive_and_writes_read_only_source(installer):
    root, install = installer
    install()
    receipt = json.loads((root / "results/install.json").read_text())
    assert receipt["verification_files"] == 3 and receipt["diag_files"] == 4
    assert receipt["provider_calls"] == 0
    assert (root / "verification/hymem/a.py").stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        install()


@pytest.mark.parametrize("failure", [None, "initial_integrity", "final_integrity", "producer_drift"])
def test_supervisor_terminal_status_is_fail_closed(gate, tmp_path, failure):
    producer = "sha256:b7dcd2a8a3a5107c9b0d868ecd2f3c96ab7d9f02392e20c9b967c44805673cb5"
    for label in ("r6", "r7"):
        directory = tmp_path / ("startup-" + label)
        directory.mkdir()
        (directory / "startup-report.json").write_text(json.dumps({
            "status": "passed", "provider_completions": 0, "provider_http_attempts": 0,
            "runtime_clients_constructed": 1, "runtime_clients_closed": 1,
            "cli_checkpoint_handles_closed": True,
            "producer_identity_sha256": "changed" if failure == "producer_drift" else producer}))
        (tmp_path / ("contract-" + label + ".json")).write_text('{"contract":"same"}')
    verifications = []
    def verified():
        verifications.append(True)
        if (failure == "initial_integrity" and len(verifications) == 1
                or failure == "final_integrity" and len(verifications) == 2):
            raise RuntimeError("injected integrity failure")
    module = ast.parse(gate.SUPERVISOR)
    terminal = ast.Module(body=module.body[-2:], type_ignores=[])
    report = {"status": "failed", "provider_calls": 0, "stages": {}}
    with pytest.raises(SystemExit) as caught:
        exec(compile(terminal, "terminal-control", "exec"), {
            "root": tmp_path, "report": report, "verified": verified,
            "invoke": lambda *_args: None, "sys": sys, "json": json, "args": []})
    assert caught.value.code == (0 if failure is None else 1)
    assert json.loads((tmp_path / "supervisor.json").read_text())["status"] == (
        "passed" if failure is None else "failed")


@pytest.mark.parametrize("defect", [None, "helper", "baseline", "bytecode", "symlink"])
def test_diagnostic_inventory_must_match_before_helpers_load(gate, tmp_path, defect):
    files = {"helper.py": b"# helper", "r6-manifest.json": b"{}"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    pins = {name: gate.sha(content) for name, content in files.items()}
    if defect == "helper": (tmp_path / "helper.py").write_text("# changed")
    elif defect == "baseline": (tmp_path / "r6-manifest.json").write_text('{"changed":true}')
    elif defect == "bytecode": (tmp_path / "helper.pyc").write_bytes(b"unexpected")
    elif defect == "symlink": (tmp_path / "alias").symlink_to(tmp_path / "helper.py")
    module = ast.parse(gate.SUPERVISOR)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "verify_diag")
    namespace = {"diag": tmp_path, "diag_pins": pins, "hashlib": hashlib}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "diag-check", "exec"), namespace)
    if defect is None:
        assert namespace["verify_diag"]() == files
    else:
        with pytest.raises(AssertionError): namespace["verify_diag"]()


@pytest.mark.parametrize("command", ["host", "INSTALL", "REMOTE_DOCKER", "SUPERVISOR"])
def test_optimized_execution_refuses_before_side_effects(gate, command):
    args = ([str(Path(gate.__file__)), "install"] if command == "host"
            else ["-c", getattr(gate, command)])
    result = subprocess.run([sys.executable, "-I", "-B", "-O", *args],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "RuntimeError: optimized_execution_forbidden" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("defect", ["timeout", "nonzero", "unsafe_cleanup", "terminal_receipt"])
def test_bad_invocation_stops_remaining_stages(gate, tmp_path, defect):
    outcome_type = dataclasses.make_dataclass("Outcome", ["status", "returncode", "safe_to_continue", "terminal_receipt_written"])
    outcomes = {
        "timeout": outcome_type("timeout", -15, True, True),
        "nonzero": outcome_type("failed", 1, True, True),
        "unsafe_cleanup": outcome_type("completed", 0, False, True),
        "terminal_receipt": outcome_type("failed", 0, False, False),
    }
    invocations = []
    def supervise(*args, **kwargs):
        invocations.append(args)
        return outcomes[defect]
    module = ast.parse(gate.SUPERVISOR)
    invoke = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "invoke")
    report = {"status": "failed", "provider_calls": 0, "stages": {}}
    namespace = {"root": tmp_path, "report": report, "stages": report["stages"],
        "verified": lambda: None, "sys": sys, "json": json, "args": [],
        "env": {}, "supervise_invocation": supervise, "dataclasses": dataclasses}
    with pytest.raises(SystemExit) as caught:
        exec(compile(ast.Module(body=[invoke, *module.body[-2:]], type_ignores=[]), "invoke-control", "exec"), namespace)
    assert caught.value.code == 1
    assert len(invocations) == 1
    assert set(report["stages"]) == {"tests"}
    assert json.loads((tmp_path / "supervisor.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("defect", [None, "network", "baseline_writable", "candidate_writable", "command", "image", "credential"])
def test_start_inspects_configuration_before_side_effects(gate, monkeypatch, tmp_path, defect):
    manifest = tmp_path / "manifest.json"
    raw = json.dumps({"expected_nodeids": [gate.FOCUS[0] + "::test_one"],
        "source_sha256": {}, "test_sha256": {}, "auxiliary_sha256": {}}).encode()
    manifest.write_bytes(raw)
    monkeypatch.setattr(gate, "MANIFEST", manifest)
    monkeypatch.setattr(gate, "MANIFEST_PIN", gate.sha(raw))
    cid = "a" * 64
    cfg = gate.config("start", cid)
    cfg["root"] = str(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    (results / "container-id.json").write_text(json.dumps({"container_id": cid}))
    obj = {"Image": cfg["image"], "Path": "/home/node/hymem-env/bin/python3", "Args": cfg["container_args"],
        "Mounts": [{"Destination": target, "Source": origin, "RW": writable, "Type": "bind"}
                   for target, origin, writable in cfg["mounts"]],
        "Config": {"Image": cfg["image"], "User": "1000:1000", "WorkingDir": "/verification",
                   "Entrypoint": ["/home/node/hymem-env/bin/python3"], "Cmd": cfg["container_args"],
                   "Env": ["TMPDIR=/results/tmp"]},
        "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
            "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"], "Init": True,
            "Memory": 2147483648, "NanoCpus": 2000000000, "PidsLimit": 128,
            "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"}, "RestartPolicy": {"Name": "no"}},
        "State": {"Status": "created", "ExitCode": 0, "OOMKilled": False, "Pid": 0}}
    if defect == "network": obj["HostConfig"]["NetworkMode"] = "bridge"
    elif defect in {"baseline_writable", "candidate_writable"}:
        target = "/baseline" if defect == "baseline_writable" else "/verification"
        next(item for item in obj["Mounts"] if item["Destination"] == target)["RW"] = True
    elif defect == "command": obj["Config"]["Cmd"] = ["changed"]
    elif defect == "image": obj["Image"] = "changed"
    elif defect == "credential": obj["Config"]["Env"].append("DEEPSEEK_API_KEY=synthetic-test")
    starts = []
    def check_output(command, **kwargs):
        if command == ["docker", "inspect", cid]: return json.dumps([obj])
        assert command == ["docker", "start", cid]
        starts.append(cid)
        obj["State"]["Status"] = "running"
        return cid
    monkeypatch.setattr(gate.subprocess, "check_output", check_output)
    if defect is None:
        exec(compile(gate.REMOTE_DOCKER, "start-control", "exec"), {"C": cfg})
        assert starts == [cid]
        assert (results / "start-intent.json").is_file()
    else:
        with pytest.raises(AssertionError):
            exec(compile(gate.REMOTE_DOCKER, "start-control", "exec"), {"C": cfg})
        assert starts == []
        assert not (results / "start-intent.json").exists()


@pytest.mark.parametrize("action", ["install", "create", "start", "status"])
def test_ssh_timeout_reports_unknown_without_command_or_retry(
    gate, monkeypatch, tmp_path, capsys, action,
):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    monkeypatch.setattr(gate, "config", lambda *args: {"action": action})
    monkeypatch.setattr(gate, "archive", lambda: b"sealed-archive")
    argv = [str(gate.__file__), action]
    if action in {"start", "status"}:
        argv += ["--container-id", "a" * 64]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def timeout(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(
            cmd=command, timeout=kwargs["timeout"],
            output=b"partial-private-output", stderr=b"huge-inline-code-traceback",
        )

    monkeypatch.setattr(gate.subprocess, "run", timeout)
    with pytest.raises(SystemExit) as caught:
        gate.main()
    assert caught.value.code == 1
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:1] == ["ssh"]
    assert gate.SSH_OPTIONS == (
        "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2",
    )
    assert command[1:1 + len(gate.SSH_OPTIONS)] == list(gate.SSH_OPTIONS)
    assert command[1 + len(gate.SSH_OPTIONS)] == "afrodite"
    assert kwargs["timeout"] == (300 if action == "install" else 120)
    assert kwargs["input"] == (b"sealed-archive" if action == "install" else b"")
    assert kwargs["capture_output"] is True
    output = capsys.readouterr()
    assert output.err == ""
    report = json.loads(output.out)
    assert report == {
        "status": "target_gate_timeout", "action": action, "outcome": "unknown",
        "requires_inspection": True, "retry_attempted": False,
        "timeout_seconds": kwargs["timeout"],
    }
    assert "partial-private-output" not in output.out
    assert "huge-inline-code-traceback" not in output.out
    assert not list(tmp_path.rglob("*.json"))
