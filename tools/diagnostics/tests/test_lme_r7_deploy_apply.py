"""Local-only controls for the staged Hermes1 deployment helper."""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from tools.diagnostics import lme_r7_deploy_apply as deploy


SETUPTOOLS_PIN = "062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922"
WHEEL_PIN = "708e7481cc80179af0e556bbf0cc00b8444c7321e2700b8d8580231d13017248"


def test_host_and_embedded_remote_compile_without_execution():
    with open(deploy.__file__, encoding="utf-8") as stream:
        compile(stream.read(), deploy.__file__, "exec")
    compile("CONFIG={}\n" + deploy.REMOTE, "<remote>", "exec")
    assert "--network','none'" in deploy.REMOTE
    assert "'--name','hymem-r7-deploy-'+label" in deploy.REMOTE
    assert "['docker','stop','-t','120',container]" in deploy.REMOTE
    assert "'--no-deps','--no-index','--no-cache-dir','--find-links'" in deploy.REMOTE
    assert "'PRAGMA user_version'" not in deploy.REMOTE


def test_backup_db_creates_consistent_copy_then_requests_runtime_verification(tmp_path):
    """Execute the embedded backup function with its real imports and fake Docker."""
    module = ast.parse(deploy.REMOTE)
    selected = [
        node for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.FunctionDef) and node.name == "backup_db"
    ]
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<backup>", "exec"), scope)
    source_db = tmp_path / "live.sqlite"
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE fact(value TEXT)")
        conn.execute("INSERT INTO fact VALUES ('preserved')")
    stage = tmp_path / "stage"
    stage.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    verified = []

    def docker(script, **kwargs):
        assert "db.connect(pathlib.Path('/backup/pre-deploy.sqlite'))" in script
        assert "db.initialize" not in script
        assert kwargs["source_host"] == candidate
        assert kwargs["rw_stage"] is True
        assert kwargs["label"] == "stop-backup"
        with sqlite3.connect(stage / "pre-deploy.sqlite") as conn:
            assert conn.execute("SELECT value FROM fact").fetchone() == ("preserved",)
        verified.append(True)
        return b'{"schema":61,"integrity_ok":true,"foreign_keys_ok":true}'

    scope.update(stage=stage, db_path=source_db, source=candidate,
                 docker_python=docker,
                 need=lambda ok, code: ok or (_ for _ in ()).throw(RuntimeError(code)),
                 sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    result = scope["backup_db"]()
    assert verified == [True]
    assert result["schema_version"] == 61
    assert result["sha256"] == hashlib.sha256(
        (stage / "pre-deploy.sqlite").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("pid,progress,allowed", [
    (23202, False, True),
    (23203, False, False),
    (23202, True, False),
])
def test_stop_idle_guard_requires_same_honcho_pid_and_idle_http_status(
    pid, progress, allowed,
):
    module = ast.parse(deploy.REMOTE)
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "confirm_idle")
    scope = {"C": {"health_port": 8000, "idle_confirmed_pid": 23202},
             "container": "hermes-1", "json": json,
             "need": lambda ok, code: ok or (_ for _ in ()).throw(RuntimeError(code))}

    def run(command, timeout, code):
        if command[:2] == ["docker", "top"]:
            return f"PID COMMAND\n{pid} /home/node/hymem-env/bin/hymem-honcho\n".encode()
        assert command[:3] == ["docker", "exec", "hermes-1"]
        assert "/health" in command[-1] and "/dream-status" in command[-1]
        return json.dumps({"health": {"status": "ok", "backend": "hymem"},
                           "in_progress": progress}).encode()

    scope["run"] = run
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 "<idle>", "exec"), scope)
    if allowed:
        scope["confirm_idle"]()
    else:
        with pytest.raises(RuntimeError):
            scope["confirm_idle"]()


@pytest.mark.parametrize("arguments", [
    ["stop"],
    ["stop", "--idle-confirmed-pid", "23202"],
    ["stop", "--idle-confirmed-pid", "23202", "--health-port", "0"],
    ["install"],
    ["install", "--setuptools-sha256", "bad", "--wheel-sha256", WHEEL_PIN],
])
def test_required_stop_and_offline_wheel_evidence_fail_before_ssh(
    monkeypatch, arguments,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("SSH called without required evidence")

    monkeypatch.setattr(deploy.subprocess, "run", forbidden)
    with pytest.raises(SystemExit) as stopped:
        deploy.main(arguments)
    assert stopped.value.code == 2


def test_one_stage_invocation_reports_only_sanitized_receipt(
    monkeypatch, capsys,
):
    calls = []

    def ssh(command, *, capture_output, timeout):
        calls.append((command, capture_output, timeout))
        receipt = {"status": "passed", "manifest_sha256": deploy.MANIFEST_PIN,
                   "source_files_verified": 479}
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt).encode(),
                               stderr=b"RAW_REMOTE_SECRET")

    monkeypatch.setattr(deploy.subprocess, "run", ssh)
    assert deploy.main([
        "install", "--setuptools-sha256", SETUPTOOLS_PIN,
        "--wheel-sha256", WHEEL_PIN,
    ]) == 0
    command, capture_output, timeout = calls.pop()
    assert not calls and command[0] == "ssh" and command[-2] == "afrodite"
    assert capture_output is True and timeout == deploy.TIMEOUT["install"]
    remote_argv = shlex.split(command[-1])
    assert remote_argv[:4] == ["python3", "-I", "-B", "-c"]
    assert SETUPTOOLS_PIN in remote_argv[-1] and WHEEL_PIN in remote_argv[-1]
    output = capsys.readouterr()
    assert json.loads(output.out)["source_files_verified"] == 479
    assert "RAW_REMOTE_SECRET" not in output.out + output.err


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "invalid_receipt"])
def test_transport_or_receipt_failure_does_not_print_remote_bytes(
    monkeypatch, capsys, failure,
):
    def ssh(command, *, capture_output, timeout):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout, output=b"RAW_REMOTE_SECRET")
        if failure == "nonzero":
            return SimpleNamespace(returncode=255, stdout=b"RAW_REMOTE_SECRET",
                                   stderr=b"RAW_REMOTE_SECRET")
        return SimpleNamespace(returncode=0, stdout=b"RAW_REMOTE_SECRET",
                               stderr=b"RAW_REMOTE_SECRET")

    monkeypatch.setattr(deploy.subprocess, "run", ssh)
    with pytest.raises(SystemExit):
        deploy.main(["start"])
    output = capsys.readouterr()
    assert "RAW_REMOTE_SECRET" not in output.out + output.err
