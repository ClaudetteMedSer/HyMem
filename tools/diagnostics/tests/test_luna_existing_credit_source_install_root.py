"""Network-free safety checks for the existing-credit source installer."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "luna_existing_credit_source_install_root.py"
SPEC = importlib.util.spec_from_file_location("luna_existing_credit_source_install_root", SOURCE)
assert SPEC and SPEC.loader
install = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install)


ROOT = "/home/atta/.hymem-lme-diagnostic-preflight-abcdefgh"
SECRET = "private provider text /private/access-result.json"


def success(kind="access"):
    prefix = "hymem-luna-access-check-" if kind == "access" else "hymem-luna-lme-diagnostic-"
    return {"schema": install.SCHEMA, "prepared": True, "model_calls": 0,
            "root": ROOT, "unit": prefix + "preflight-abcdefgh.service",
            "kind": kind, "receipt_sha256": "a" * 64}


@pytest.mark.parametrize("change", [
    {"status": {"private": SECRET}},
    {"schema": SECRET}, {"prepared": 1}, {"model_calls": False},
    {"model_calls": 1}, {"root": SECRET}, {"kind": "pilot"},
    {"unit": SECRET}, {"receipt_sha256": SECRET},
    {"extra": SECRET},
])
def test_success_projection_rejects_untrusted_or_private_values(change):
    value = success()
    value.update(change)
    assert install._success_projection(value, root=ROOT, kind="access", returncode=0) is None


def test_success_projection_accepts_only_exact_kind_and_unit():
    assert install._success_projection(success(), root=ROOT, kind="access", returncode=0) == success()
    assert install._success_projection(success("pilot"), root=ROOT, kind="pilot", returncode=0) == success("pilot")
    assert install._success_projection(success(), root=ROOT, kind="access", returncode=1) is None


def test_main_projects_remote_stdout_and_never_echoes_private_json(monkeypatch, capsys):
    repo = Path(install.__file__).resolve().parents[2]
    pinned = {}
    for name, (relative, _) in install.FILES.items():
        pinned[name] = (relative, hashlib.sha256((repo / relative).read_bytes()).hexdigest())
    monkeypatch.setattr(install, "FILES", pinned)
    monkeypatch.setattr(sys, "argv", ["installer", "--root", ROOT, "--kind", "access"])
    calls = []

    def remote(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": SECRET}), stderr=SECRET)

    monkeypatch.setattr(install.subprocess, "run", remote)
    assert install.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {"status": "installation_unverified",
                                  "never_repeat_automatically": True}
    assert SECRET not in output
    assert len(calls) == 1 and calls[0][0] == "ssh"


def test_main_accepts_only_exact_remote_success(monkeypatch, capsys):
    repo = Path(install.__file__).resolve().parents[2]
    pinned = {name: (relative, hashlib.sha256((repo / relative).read_bytes()).hexdigest())
              for name, (relative, _) in install.FILES.items()}
    monkeypatch.setattr(install, "FILES", pinned)
    monkeypatch.setattr(sys, "argv", ["installer", "--root", ROOT, "--kind", "pilot"])
    monkeypatch.setattr(install.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=json.dumps(success("pilot")), stderr=SECRET))
    assert install.main() == 0
    assert json.loads(capsys.readouterr().out) == success("pilot")


def _remote_inventory_check(root: Path, *, extra: bool) -> None:
    root.mkdir(mode=0o700)
    (root / "candidate").mkdir(mode=0o700)
    (root / "code").mkdir(mode=0o700)
    (root / "source-map.json").write_text("{}")
    if extra:
        (root / "safe-terminal.json").write_text(SECRET)
    payload = {"root": str(root), "kind": "access", "files": {
        name: {"data": base64.b64encode(name.encode()).decode(),
               "sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in install.FILES}}
    # Run the actual remote pre-write gate against a temporary local directory.
    # Substitute only the host path and UID assertions; no SSH or prepare runs.
    pre_write = install.REMOTE.split('for name,record in PAYLOAD["files"].items():', 1)[0]
    pre_write = pre_write.replace("/home/atta/", re.escape(str(root.parent)) + "/")
    pre_write = pre_write.replace("os.getuid()==1000", "os.getuid()==os.getuid()")
    pre_write = pre_write.replace("root.stat().st_uid==1000", "root.stat().st_uid==os.getuid()")
    exec("PAYLOAD=" + repr(payload) + "\n" + pre_write, {})


def test_remote_gate_requires_exact_fresh_staging_entries(tmp_path):
    root = tmp_path / ".hymem-lme-diagnostic-preflight-abcdefgh"
    _remote_inventory_check(root, extra=False)
    with pytest.raises(AssertionError):
        _remote_inventory_check(tmp_path / ".hymem-lme-diagnostic-preflight-ijklmnop", extra=True)
