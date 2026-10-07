"""Independent exact-source assembly tests; no host or provider calls."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.diagnostics import luna_timeout_install_v1 as installer
from tools.diagnostics import luna_timeout_probe_v1 as probe


BUNDLE = None
HELPERS = Path(__file__).resolve().parents[1] / "tools/diagnostics"


@pytest.fixture(scope="module", autouse=True)
def accepted_bundle(tmp_path_factory):
    global BUNDLE
    BUNDLE = tmp_path_factory.mktemp("root-source-preparation") / "bundle"
    prepared = probe.prepare(BUNDLE)
    assert prepared["receipt_sha256"] == installer.PREPARATION_SHA256


def payload(bundle=None):
    return installer.build_payload(bundle or BUNDLE, HELPERS,
        hashlib.sha256((HELPERS / "luna_timeout_launch_v1.py").read_bytes()).hexdigest(),
        hashlib.sha256((HELPERS / "luna_timeout_progress_v1.py").read_bytes()).hexdigest())


def remote(raw):
    namespace = {"__name__": "root_install_test"}
    exec(compile(installer.render_remote_script(raw), "<reviewed-installer>", "exec"), namespace)
    return namespace


def test_root_actual_source_copy_has_exact_inventory_and_private_modes(tmp_path):
    raw = payload()
    namespace = remote(raw)
    result = namespace["install"](raw, home=tmp_path, uid=os.getuid(), enforce_host=False)
    root = Path(result["root"])
    assert result["model_calls"] == 0
    expected, _ = namespace["validate"](raw)
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(expected)
    for path in (root, *root.rglob("*")):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    source = """import hashlib,os,sys,types,json
from pathlib import Path
root=Path(sys.argv[1]); path=root/'timeout-host-v1.py'; raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
m=types.ModuleType('root_isolated_host'); m.__file__=str(path)
exec(compile(raw,str(path),'exec'),m.__dict__)
m.HOST_HOME=root.parent; m.HOST_UID=os.getuid()
probe,(observer,request)=m.verify_sources(root)
print(json.dumps({'source_verified':True,'model':observer.base.MODEL,'calls':0,'request':request.__name__}))
"""
    checked = subprocess.run([sys.executable, "-I", "-B", "-c", source, str(root), installer.HOST_SHA256],
        capture_output=True, text=True, check=True, timeout=15)
    assert json.loads(checked.stdout) == {"source_verified": True, "model": "gpt-6-luna", "calls": 0, "request": "LLMRequest"}


@pytest.mark.parametrize("mutation", ["path", "extra", "duplicate", "content"])
def test_root_rehashed_malformed_payload_rejected_before_install(tmp_path, mutation):
    value = json.loads(payload())
    if mutation == "path":
        value["files"][0]["path"] = "../escape.py"
    elif mutation == "extra":
        value["files"].append(value["files"][0])
    elif mutation == "duplicate":
        value["files"][1] = value["files"][0]
    else:
        value["files"][0]["base64"] = base64.b64encode(b"arbitrary unapproved source").decode()
        value["files"][0]["sha256"] = hashlib.sha256(b"arbitrary unapproved source").hexdigest()
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    namespace = remote(raw)
    with pytest.raises(ValueError):
        namespace["install"](raw, home=tmp_path, uid=os.getuid(), enforce_host=False)
    assert not list(tmp_path.iterdir())


def test_root_extra_empty_bundle_directory_rejected(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copied)
    (copied / "unused").mkdir()
    with pytest.raises(ValueError):
        payload(copied)
