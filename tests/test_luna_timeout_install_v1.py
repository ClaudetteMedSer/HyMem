"""Offline controls for the source-only Luna timeout installer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

import pytest
from tools.diagnostics import luna_timeout_probe_v1 as probe


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools/diagnostics/luna_timeout_install_v1.py"


def module():
    spec = importlib.util.spec_from_file_location("luna_timeout_install_v1", SOURCE)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def fixture(tmp_path):
    mod = module()
    bundle = tmp_path / "bundle"
    prepared = probe.prepare(bundle)
    assert prepared["receipt_sha256"] == mod.PREPARATION_SHA256
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    actual = SOURCE.parent
    host = (actual / "luna_timeout_host_v1.py").read_bytes()
    launcher = (actual / "luna_timeout_launch_v1.py").read_bytes()
    reader = b"raise RuntimeError('source unexpectedly executed')\n"
    (helpers / "luna_timeout_host_v1.py").write_bytes(host)
    (helpers / "luna_timeout_launch_v1.py").write_bytes(launcher)
    (helpers / "luna_timeout_progress_v1.py").write_bytes(reader)
    pins = (hashlib.sha256(launcher).hexdigest(),
            hashlib.sha256(reader).hexdigest())
    return mod, bundle, helpers, pins


def payload(fixture):
    mod, bundle, helpers, pins = fixture
    return mod.build_payload(bundle, helpers, *pins)


def remote(mod, data):
    namespace = {"__name__": "offline_installer_test"}
    exec(compile(mod.render_remote_script(data), "<remote-installer>", "exec"),
         namespace)
    return namespace


def test_import_has_no_host_effects(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("subprocess called")))
    mod = module()
    assert mod.HELPERS["timeout-host-v1.py"] == "luna_timeout_host_v1.py"


def test_exact_install_source_only_private_modes_and_fresh_root(fixture, tmp_path):
    mod, _, _, _ = fixture
    data = payload(fixture)
    host = remote(mod, data)
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    first = host["install"](data, home, os.getuid(), enforce_host=False)
    second = host["install"](data, home, os.getuid(), enforce_host=False)
    assert first["installed"] is True and first["code_files"] == 13
    assert first["bundle_files"] == 14 and first["preparation_receipts"] == 1
    assert first["helpers"] == 3 and first["model_calls"] == 0
    assert first["root"] != second["root"]
    for receipt in (first, second):
        root = Path(receipt["root"])
        assert root.parent == home and root.name.startswith(".hymem-luna-timeout-")
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        files = set()
        for path in root.rglob("*"):
            assert not path.is_symlink()
            if path.is_dir():
                assert stat.S_IMODE(path.stat().st_mode) == 0o700
            else:
                assert stat.S_IMODE(path.stat().st_mode) == 0o600
                files.add(path.relative_to(root).as_posix())
        expected = {"bundle/code/" + name for name in mod.SOURCE_NAMES}
        expected |= {"bundle/local-preparation-receipt.json"}
        expected |= set(mod.HELPERS)
        assert files == expected
    assert (Path(first["root"]) / "timeout-progress-v1.py").read_bytes().startswith(
        b"raise RuntimeError")


@pytest.mark.parametrize("change", ["missing", "extra", "emptydir", "symlink", "tamper"])
def test_local_bundle_rejects_invalid_closure(fixture, tmp_path, change):
    mod, bundle, helpers, pins = fixture
    target = bundle / "code" / "benchmarks" / "codex_subscription.py"
    if change == "missing":
        target.unlink()
    elif change == "extra":
        (bundle / "secret.txt").write_text("secret")
    elif change == "emptydir":
        (bundle / "code" / "unexpected").mkdir()
    elif change == "symlink":
        target.unlink()
        target.symlink_to(tmp_path / "elsewhere")
    else:
        target.write_bytes(target.read_bytes() + b"# changed\n")
    with pytest.raises(ValueError):
        mod.build_payload(bundle, helpers, *pins)


def test_helper_hash_rejected(fixture):
    mod, bundle, helpers, pins = fixture
    with pytest.raises(ValueError, match="helper_drift"):
        mod.build_payload(bundle, helpers, "0" * 64, pins[1])
    (helpers / "luna_timeout_progress_v1.py").write_text("changed")
    with pytest.raises(ValueError, match="helper_drift"):
        mod.build_payload(bundle, helpers, *pins)


def test_remote_rejects_tamper_duplicate_and_path_escape(fixture):
    mod, _, _, _ = fixture
    good = payload(fixture)
    host = remote(mod, good)
    with pytest.raises(ValueError, match="payload_drift"):
        host["validate"](good + b" ")
    altered = json.loads(good)
    altered["files"][0]["path"] = "../escape.py"
    bad = json.dumps(altered, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="file_path_invalid"):
        remote(mod, bad)["validate"](bad)
    duplicate = good.replace(b'"schema":"luna-timeout-install-v1"',
                             b'"schema":"luna-timeout-install-v1","schema":"x"')
    assert duplicate != good
    with pytest.raises(ValueError, match="duplicate_json_key"):
        remote(mod, duplicate)["validate"](duplicate)


def test_host_identity_and_existing_file_rejected(fixture, tmp_path):
    mod, _, _, _ = fixture
    data = payload(fixture)
    host = remote(mod, data)
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="host_user_invalid"):
        host["install"](data, home, os.getuid(), enforce_host=True)
    path = tmp_path / "existing"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        host["write_once"](path, b"replacement")
    assert path.read_bytes() == b"original"
