"""Local-only checks for the second-phase Honcho repair helper."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
import venv

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "honcho_sqlite_install_recover.py"
SPEC = importlib.util.spec_from_file_location("honcho_sqlite_install_recover", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def _pure_functions():
    names = {"need", "sha_bytes", "receipt", "atomic_source", "valid_command"}
    nodes = [node for node in ast.parse(repair.REMOTE).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"hashlib": hashlib, "json": json, "os": os, "Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<remote-pure>", "exec"), namespace)
    return namespace


def _remote_namespace(*names: str):
    selected = set(names) | {"need", "sha_bytes", "sha", "receipt", "atomic_source"}
    nodes = [node for node in ast.parse(repair.REMOTE).body
             if isinstance(node, ast.FunctionDef) and node.name in selected]
    assert {node.name for node in nodes} == selected
    namespace = {"hashlib": hashlib, "json": json, "os": os, "Path": Path,
                 "stat": stat, "subprocess": subprocess}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<remote-functions>", "exec"), namespace)
    return namespace


def test_command_shape_accepts_only_exact_module_invocation() -> None:
    valid = _pure_functions()["valid_command"]
    assert valid([b"/usr/bin/python3.11", b"-m", b"hymem.honcho.app"])
    assert not valid([b"/usr/bin/python3.11", b"-m", b"hymem.honcho.app", b"--reload"])
    assert not valid([b"/usr/bin/python3.11", b"-c", b"print('x')"])
    assert not valid([b"/bin/sh", b"-m", b"hymem.honcho.app"])
    assert not valid([b"/home/node/hymem-env/bin/hymem-honcho"])


def test_expected_port_accepts_proc_environment_bytes_only() -> None:
    expected_port = _remote_namespace("expected_port")["expected_port"]
    assert expected_port({b"HYMEM_HONCHO_PORT": b"8765"}) == 8765
    assert expected_port({}) == 8765
    for raw in (b"", b"87x5", b"-1", b"\xd9\xa8\xd9\xa7\xd9\xa6\xd9\xa5", "8765"):
        with pytest.raises(RuntimeError, match="port_invalid"):
            expected_port({b"HYMEM_HONCHO_PORT": raw})
    with pytest.raises(RuntimeError, match="port_changed"):
        expected_port({b"HYMEM_HONCHO_PORT": b"8766"})


def test_atomic_install_file_and_receipt_are_exact_and_no_overwrite(tmp_path) -> None:
    functions = _pure_functions()
    target = tmp_path / "db.py"
    target.write_bytes(b"original")
    functions["atomic_source"](target, b"candidate", 0o640)
    assert target.read_bytes() == b"candidate"
    assert target.stat().st_mode & 0o777 == 0o640
    assert not (tmp_path / "db.py.sqlite-repair-tmp").exists()
    path = tmp_path / "install.json"
    functions["receipt"](path, {"candidate": "sha256"})
    assert json.loads(path.read_text()) == {"candidate": "sha256"}
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(RuntimeError, match="receipt_already_exists"):
        functions["receipt"](path, {"candidate": "changed"})


def test_remote_command_is_fixed_and_stderr_is_not_exposed(monkeypatch, capsys) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=2, stdout="", stderr="private secret")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert repair.run_remote("status", repair.DEFAULT_STAGE) == 1
    output = capsys.readouterr().out
    assert "private secret" not in output
    command, kwargs = calls[0]
    assert command[:2] == ["ssh", "-C"]
    assert "docker exec -i -u node hermes-1 /usr/bin/python3.11" in command[-1]
    assert json.loads(kwargs["input"]) == {"stage": repair.DEFAULT_STAGE}


def test_wrapped_remote_classifies_only_fixed_guard_codes() -> None:
    ast.parse(repair.wrapped_remote())
    result = subprocess.run(
        ["python3.11", "-c", repair.wrapped_remote(), "invalid-action"],
        input=json.dumps({"stage": repair.DEFAULT_STAGE}), text=True,
        capture_output=True, timeout=5,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"error": "unknown_action", "type": "RuntimeError"}
    assert "Traceback" not in result.stdout


def test_every_explicit_guard_code_is_in_fixed_allowlist() -> None:
    codes = set()
    for node in ast.walk(ast.parse(repair.REMOTE)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "need" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)):
            codes.add(node.args[1].value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "RuntimeError" and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
            codes.add(node.args[0].value)
    assert codes <= repair.SAFE_GUARD_CODES


def test_replaying_venv_argv_with_real_executable_preserves_prefix(tmp_path) -> None:
    venv.EnvBuilder(with_pip=False).create(tmp_path / "venv")
    venv_python = tmp_path / "venv/bin/python"
    real_executable = os.path.realpath(venv_python)
    code = "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix,sys.executable]))"
    direct = subprocess.run([str(venv_python), "-c", code], capture_output=True,
                            text=True, check=True, timeout=5)
    replay = subprocess.run([str(venv_python), "-c", code], executable=real_executable,
                            capture_output=True, text=True, check=True, timeout=5)
    assert json.loads(replay.stdout) == json.loads(direct.stdout)
    assert json.loads(replay.stdout)[0] != json.loads(replay.stdout)[1]


def test_recovery_refuses_stage_drift_and_existing_intent_before_signal(tmp_path) -> None:
    namespace = _remote_namespace("recover")
    signals = []
    namespace["os"] = SimpleNamespace(kill=lambda *args: signals.append(args))
    namespace["validate_stage"] = lambda: (_ for _ in ()).throw(RuntimeError("stage_candidate_mismatch"))
    with pytest.raises(RuntimeError, match="stage_candidate_mismatch"):
        namespace["recover"]()
    assert signals == []

    namespace["STAGE"] = tmp_path
    namespace["ORIGINAL"] = "original"
    namespace["CANDIDATE"] = "candidate"
    namespace["HELPER"] = "helper"
    namespace["validate_stage"] = lambda: ({}, {})
    namespace["json_file"] = lambda path: {
        "original_sha256": "original", "candidate_db_sha256": "candidate",
        "helper_sha256": "helper",
    }
    (tmp_path / "recover-intent.json").write_text("{}")
    with pytest.raises(RuntimeError, match="recovery_already_attempted"):
        namespace["recover"]()
    assert signals == []

    (tmp_path / "recover-intent.json").unlink()
    db = tmp_path / "db.py"
    helper = tmp_path / "serialized_sqlite.py"
    db.write_bytes(b"drifted source")
    helper.write_bytes(b"helper")
    namespace["LIVE_DB"] = db
    namespace["LIVE_HELPER"] = helper
    with pytest.raises(RuntimeError, match="live_sources_changed"):
        namespace["recover"]()
    assert signals == []

    namespace["sha"] = lambda path: "candidate" if path == db else "helper"
    namespace["original_identity"] = lambda: {"pid": 12381}
    namespace["command_and_environment"] = lambda: ([b"python"], {}, "cmdsha")
    namespace["expected_port"] = lambda env: 8765
    namespace["port_owners"] = lambda port: {99999}
    namespace["PID"] = 12381
    with pytest.raises(RuntimeError, match="port_ownership_changed"):
        namespace["recover"]()
    assert signals == []


@pytest.mark.parametrize("foreign_drift", [False, True])
def test_install_rollback_only_restores_exact_candidate(tmp_path, foreign_drift) -> None:
    namespace = _remote_namespace("install")
    stage = tmp_path / "stage"
    live = tmp_path / "live"
    (stage / "hymem/core").mkdir(parents=True)
    (live / "hymem/core").mkdir(parents=True)
    original, candidate, helper = b"original-source", b"candidate-source", b"helper-source"
    db = live / "hymem/core/db.py"
    helper_path = live / "hymem/core/serialized_sqlite.py"
    db.write_bytes(original)
    os.chmod(db, 0o600)
    (stage / "original-db.py").write_bytes(original)
    (stage / "hymem/core/db.py").write_bytes(candidate)
    (stage / "hymem/core/serialized_sqlite.py").write_bytes(helper)
    namespace.update({
        "STAGE": stage, "LIVE_DB": db, "LIVE_HELPER": helper_path,
        "UID": os.geteuid(), "PID": 12381, "START": 215405984,
        "ORIGINAL": hashlib.sha256(original).hexdigest(),
        "CANDIDATE": hashlib.sha256(candidate).hexdigest(),
        "HELPER": hashlib.sha256(helper).hexdigest(),
        "validate_stage": lambda: ({}, {}), "original_identity": lambda: {},
        "command_and_environment": lambda: ([b"python"], {}, "cmdsha"),
        "expected_port": lambda env: 8765, "port_owners": lambda port: {12381},
        "venv_replay_ok": lambda: True,
    })
    original_receipt = namespace["receipt"]

    def fail_at_install_receipt(path, value):
        if path.name == "install.json":
            if foreign_drift:
                db.write_bytes(b"foreign-source")
            raise OSError("simulated receipt failure")
        return original_receipt(path, value)

    namespace["receipt"] = fail_at_install_receipt
    with pytest.raises(OSError, match="simulated receipt failure"):
        namespace["install"]()
    assert db.read_bytes() == (b"foreign-source" if foreign_drift else original)
    assert not helper_path.exists()
    assert (stage / "install-failed.json").is_file()
