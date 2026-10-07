"""Offline controls for the pinned, one-shot source installer."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tools.diagnostics import hermes_native_oauth_probe_install_v1 as installer


def receipt():
    return {"schema": installer.SCHEMA, "prepared": True, "model_calls": 0,
        "root": installer.ROOT,
        "unit": "hymem-luna-native-oauth-probe-preflight-wszo8v4u.service",
        "receipt_sha256": "a" * 64}


def test_exact_four_sources_are_pinned_and_remote_is_prepare_only():
    repo = Path(__file__).resolve().parents[1]
    assert set(installer.FILES) == {
        "code/benchmarks/hermes_codex_responses_v1.py",
        "code/benchmarks/hermes_lme_oauth_v1.py",
        "hermes_native_oauth_probe_v1.py",
        "luna_lme_diagnostic_launch_v9.py"}
    for relative, digest in installer.FILES.values():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == digest
    remote = installer.REMOTE
    assert '"--prepare-root"' in remote
    assert '"--launch-root"' not in remote and '"--run-root"' not in remote
    assert "os.O_EXCL" in remote and "os.O_NOFOLLOW" in remote
    assert 'entry.name for entry in root.iterdir()' in remote
    assert 'private_dir(root/"code/benchmarks")' in remote
    assert 'hashlib.sha256(data).hexdigest()==digest' in remote


def test_projection_requires_exact_finite_receipt():
    good = receipt()
    assert installer._projection(good, 0) == good
    for bad in ({**good, "model_calls": 1}, {**good, "model_calls": False},
                {**good, "prepared": 1}, {**good, "receipt_sha256": "A" * 64},
                {**good, "unit": "another.service"}, {**good, "private": "leak"}):
        assert installer._projection(bad, 0) is None
    assert installer._projection(good, 1) is None


def test_main_uses_single_bounded_ssh_and_exports_only_projection(monkeypatch, capsys):
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps(receipt()), stderr="secret", returncode=0)
    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    assert installer.main() == 0
    assert json.loads(capsys.readouterr().out) == receipt()
    assert len(calls) == 1
    command, options = calls[0]
    assert command == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "afrodite", "python3 -I -B -"]
    assert options["capture_output"] is True and options["timeout"] == 120
    assert installer.ROOT in options["input"]


def test_source_drift_blocks_ssh_and_failure_is_fixed_safe_error(monkeypatch, capsys):
    monkeypatch.setitem(installer.FILES, "hermes_native_oauth_probe_v1.py",
        ("tools/diagnostics/hermes_native_oauth_probe_v1.py", "0" * 64))
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("SSH must not run")))
    assert installer.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "installation_unverified", "never_repeat_automatically": True}


def test_remote_malformed_or_duplicate_receipt_never_leaks(monkeypatch, capsys):
    responses = ['{"schema":"x","schema":"y"}', 'private output',
                 json.dumps({**receipt(), "extra": "secret"})]
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout=responses.pop(0), stderr="secret", returncode=0)
    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    for _ in range(3):
        assert installer.main() == 1
        assert json.loads(capsys.readouterr().out) == {
            "status": "installation_unverified", "never_repeat_automatically": True}
