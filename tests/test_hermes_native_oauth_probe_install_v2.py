"""Offline controls for the pinned v2 native OAuth source installer."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import hermes_native_oauth_probe_install_v2 as installer


def receipt():
    return {"schema": installer.SCHEMA, "prepared": True, "model_calls": 0,
        "root": installer.ROOT, "unit": installer.UNIT, "receipt_sha256": "a" * 64}


def test_exact_five_sources_pinned_and_remote_prepares_only():
    repo = Path(__file__).resolve().parents[1]
    assert set(installer.FILES) == {
        "code/benchmarks/hermes_codex_responses_v1.py",
        "code/benchmarks/hermes_codex_responses_v2.py",
        "code/benchmarks/hermes_lme_oauth_v2.py",
        "hermes_native_oauth_probe_v2.py",
        "luna_lme_diagnostic_launch_v9.py"}
    assert len(installer.FILES) == 5
    for relative, digest in installer.FILES.values():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == digest
    remote = installer.REMOTE
    assert '"--prepare-root"' in remote
    assert '"--launch-root"' not in remote and '"--run-root"' not in remote
    assert "os.O_EXCL" in remote and "os.O_NOFOLLOW" in remote
    assert 'entry.name for entry in root.iterdir()' in remote
    assert 'private_dir(root/"code/benchmarks")' in remote
    assert 'hashlib.sha256(data).hexdigest()==digest' in remote
    assert 'stat.S_IMODE(info.st_mode)==0o700' in remote
    assert 'stat.S_IMODE(source_map.st_mode)==0o600' in remote
    assert 'assert set(PAYLOAD["files"])==set(expected)' in remote
    assert 'and set(result)=={"schema","prepared","model_calls","root","unit","receipt_sha256"}' in remote
    assert 'result["schema"]=="hermes-native-oauth-probe-v2"' in remote


@pytest.mark.parametrize("change", [
    {"model_calls": 1}, {"model_calls": False}, {"prepared": 1},
    {"receipt_sha256": "A" * 64}, {"receipt_sha256": "a" * 63},
    {"unit": "another.service"}, {"root": "/other"},
    {"schema": "hermes-native-oauth-probe-install-v1"}, {"private": "leak"},
])
def test_projection_rejects_mutation(change):
    good = receipt()
    assert installer._projection(good, 0) == good
    assert installer._projection({**good, **change}, 0) is None
    assert installer._projection(good, 1) is None


def test_main_single_exact_bounded_command_and_finite_projection(monkeypatch, capsys):
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps(receipt()), stderr="private secret", returncode=0)
    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    assert installer.main() == 0
    assert json.loads(capsys.readouterr().out) == receipt()
    assert len(calls) == 1
    command, options = calls[0]
    assert command == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "afrodite", "python3 -I -B -"]
    assert options["capture_output"] is True and options["timeout"] == 120
    assert options["text"] is True
    assert options["input"].startswith("PAYLOAD=")
    assert installer.ROOT in options["input"]
    assert '"--prepare-root"' in options["input"]
    assert "private secret" not in capsys.readouterr().out


def test_source_drift_blocks_ssh(monkeypatch, capsys):
    monkeypatch.setitem(installer.FILES, "hermes_native_oauth_probe_v2.py",
        ("tools/diagnostics/hermes_native_oauth_probe_v2.py", "0" * 64))
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("SSH must not run")))
    assert installer.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "installation_unverified", "never_repeat_automatically": True}


@pytest.mark.parametrize("stdout", [
    '{"schema":"x","schema":"y"}', "private output", "NaN",
    json.dumps({**receipt(), "extra": "private"}),
    json.dumps({**receipt(), "model_calls": True}),
])
def test_malformed_or_mutated_remote_response_never_leaks(monkeypatch, capsys, stdout):
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=stdout, stderr="private secret", returncode=0))
    assert installer.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "installation_unverified", "never_repeat_automatically": True}
