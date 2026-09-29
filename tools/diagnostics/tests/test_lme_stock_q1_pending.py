"""Local-only controls for recovered history and the UNSEALED successor."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "tools/diagnostics/lme_stock_q1"
PENDING = BASE / "pending"
PREVIOUS = BASE / "previous-v3"
HISTORICAL_MANIFEST = "e6be83fee57c238c88d989bc7d8f49fc2016e62e7642636dc4cfd240f5f1868d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network forbidden during local package preparation")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        monkeypatch.setattr(socket, name, forbidden)


def test_recovered_predecessor_matches_all_seven_historical_pins():
    assert sha(PREVIOUS / "manifest.json") == HISTORICAL_MANIFEST
    manifest = json.loads((PREVIOUS / "manifest.json").read_text())
    expected = {"manifest.json": HISTORICAL_MANIFEST, **manifest["helper_sha256"]}
    assert {p.name for p in PREVIOUS.iterdir()} == set(expected)
    assert len(expected) == 7
    assert {name: sha(PREVIOUS / name) for name in expected} == expected


def test_pending_is_unsealed_and_preserves_stock_recipe_and_supervisor():
    assert not (PENDING / "manifest.json").exists()
    old = load(PREVIOUS / "q1_stock_run.py", "q1_previous_recipe")
    new = load(PENDING / "q1_stock_run.py", "q1_pending_recipe")
    assert new.stock_arguments() == old.stock_arguments()
    assert new.TIMEOUT == old.TIMEOUT == 5400
    for name in ("transport_common.py", "supervised_invocation.py"):
        assert (PENDING / name).read_bytes() == (PREVIOUS / name).read_bytes()
    assert (PENDING / "lme_q1_startup_preflight.py").read_bytes() == (
        ROOT / "tools/diagnostics/lme_q1_startup_preflight.py").read_bytes()
    with pytest.raises(FileNotFoundError):
        new.PACKAGE = PENDING
        new.validate_package("a" * 64)


@pytest.mark.parametrize("count", [1, 3, 231])
def test_source_inventory_is_exact_but_not_fixed_at_230(tmp_path, monkeypatch, count):
    run = load(PENDING / "q1_stock_run.py", "q1_pending_inventory")
    source = tmp_path / "source"
    source.mkdir()
    files = {}
    for index in range(count):
        path = source / f"source_{index}.py"
        path.write_text("# invented source inventory\n")
        files[path.name] = sha(path)
    monkeypatch.setattr(run, "SOURCE", source)
    monkeypatch.setattr(run, "selector_from_source", lambda _: object())
    monkeypatch.setattr(run, "derive_seed", lambda _: 53)
    run.verify_source({"source_sha256": files})
    (source / "unexpected.bin").write_bytes(b"\x00\xff")
    with pytest.raises(RuntimeError, match="source_inventory_drift"):
        run.verify_source({"source_sha256": files})


def test_source_links_outside_hymem_are_rejected(tmp_path, monkeypatch):
    run = load(PENDING / "q1_stock_run.py", "q1_pending_links")
    source = tmp_path / "source"
    source.mkdir()
    (source / "extra").symlink_to(tmp_path)
    monkeypatch.setattr(run, "SOURCE", source)
    with pytest.raises(RuntimeError, match="source_symlink"):
        run.verify_source({"source_sha256": {"anything.py": "a" * 64}})


CHILD = r'''
import hashlib,json,pathlib,sys,types
def block(event,args):
    if event in {"socket.connect","socket.getaddrinfo","socket.gethostbyname","socket.gethostbyaddr","socket.getnameinfo","socket.sendto","socket.sendmsg"}:
        raise AssertionError("unexpected network")
sys.addaudithook(block)
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
before=sorted(str(p.relative_to(root)) for p in root.rglob("*"))
path=root/("q1_stock_run.py" if mode=="host" else "q1_stock_validate.py")
module=types.ModuleType("q1_test_loader"); module.__file__=str(path)
exec(compile(path.read_bytes(),str(path),"exec"),module.__dict__)
for iteration in range(2):
    if mode=="host":
        target=root/"q1_stock_host.py"
        loaded=module.load_verified_module(target,hashlib.sha256(target.read_bytes()).hexdigest(),"q1_verified_host")
        assert callable(loaded.command)
    else:
        pin=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()
        assert callable(module.load_run_helper(pin).verify_source)
after=sorted(str(p.relative_to(root)) for p in root.rglob("*"))
assert before==after
assert not any("__pycache__" in name for name in after)
print(json.dumps({"loads":2,"inventory_unchanged":True,"mode":mode}))
'''


@pytest.mark.parametrize("mode", ["host", "postvalidator"])
def test_two_real_subprocess_loads_without_B_flag_never_write_cache(tmp_path, mode):
    # No -B, no PYTHONDONTWRITEBYTECODE: this reproduces the original host
    # inspector regression unless the helper compiles verified bytes itself.
    for name in ("q1_stock_run.py", "q1_stock_host.py", "q1_stock_validate.py"):
        (tmp_path / name).write_bytes((PENDING / name).read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps({
        "helper_sha256": {"q1_stock_run.py": sha(tmp_path / "q1_stock_run.py")},
    }))
    completed = subprocess.run([sys.executable, "-I", "-c", CHILD, str(tmp_path), mode],
                               env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
                               capture_output=True, text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"loads": 2, "inventory_unchanged": True, "mode": mode}


@pytest.mark.parametrize("fault", ["hash", "symlink"])
def test_verified_import_refuses_untrusted_bytes(tmp_path, fault):
    run = load(PENDING / "q1_stock_run.py", "q1_pending_bad_import")
    real = tmp_path / "real.py"
    real.write_text("raise AssertionError('untrusted code executed')\n")
    path = real
    pin = sha(real)
    if fault == "hash":
        pin = "0" * 64
    else:
        path = tmp_path / "link.py"
        path.symlink_to(real)
    with pytest.raises(RuntimeError, match="helper_(drift|path)"):
        run.load_verified_module(path, pin, "q1_never_executed")


def test_worker_clears_extra_body_before_stock_cli_without_reading_real_key(monkeypatch):
    run = load(PENDING / "q1_stock_run.py", "q1_pending_worker")
    pin = "a" * 64
    monkeypatch.setattr(run.sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(
        run.canonical({"execute_stock_q1": pin}))))
    monkeypatch.setattr(run.sys, "path", list(sys.path))
    monkeypatch.setattr(run.sys, "argv", [])
    monkeypatch.setattr(run.os, "environ", {"HYMEM_LLM_EXTRA_BODY": "unwanted"})
    monkeypatch.setattr(run, "validate_package", lambda _: {})
    checks = []
    monkeypatch.setattr(run, "verify_source", lambda _: checks.append(True))
    monkeypatch.setattr(run, "load_helper", lambda _: types.SimpleNamespace(read_key=lambda: "synthetic-key"))
    called = []
    def stock(path, *, run_name):
        assert "HYMEM_LLM_EXTRA_BODY" not in run.os.environ
        assert run.os.environ["DEEPSEEK_API_KEY"] == "synthetic-key"
        assert run.sys.argv == run.stock_arguments()
        called.append((path, run_name))
    monkeypatch.setattr(run.runpy, "run_path", stock)
    run.worker(pin)
    assert checks == [True, True] and called == [(run.stock_arguments()[0], "__main__")]


def test_preflight_invokes_real_startup_interface_and_reports_dynamic_count(tmp_path, monkeypatch):
    run = load(PENDING / "q1_stock_run.py", "q1_pending_preflight")
    data = tmp_path / "longmemeval_s_cleaned.json"
    sessions = [[{}] * 11 for _ in range(43)] + [[{}] * 6]
    rows = [{} for _ in range(500)]
    rows[210] = {"question_id": "09ba9854", "haystack_sessions": sessions}
    data.write_text(json.dumps(rows))
    monkeypatch.setattr(run, "DATA", data)
    monkeypatch.setattr(run, "OUTPUT", tmp_path)
    monkeypatch.setattr(run, "sha", lambda _: run.DATASET_SHA)
    monkeypatch.setattr(run, "verify_source", lambda _: lambda rows, **kwargs: [rows[210]])
    calls = []
    def startup(**kwargs):
        calls.append(kwargs)
        return {"status": "passed", "provider_completions": 0}
    monkeypatch.setattr(run, "load_helper", lambda name: types.SimpleNamespace(run_probe=startup))
    manifest = {"source_sha256": {"a.py": "a" * 64, "b.py": "b" * 64}}
    result = run.preflight(manifest)
    assert result["source_files_verified"] == 2
    assert result["startup_probe"]["status"] == "passed"
    assert calls == [{"source": run.SOURCE, "arguments": run.stock_arguments(), "output": tmp_path}]
    assert result["benchmark_executed"] is False and result["api_calls"] == 0

