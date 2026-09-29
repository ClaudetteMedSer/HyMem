"""Network-free controls of explicit upload actions and output privacy."""
import importlib.util
from pathlib import Path
import types
import pytest

DIAG = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "episode_shadow_uploader", DIAG / "claim_conflict_episode_shadow_dream_upload.py")
uploader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uploader)


def test_exact_four_pins_and_bad_local_file_refused(tmp_path):
    pieces = uploader.local_payload()
    assert set(pieces) == set(uploader.FILES)
    for name, raw in pieces.items():
        (tmp_path / name).write_bytes(raw)
    (tmp_path / uploader.HOST).write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="pin_drift"):
        uploader.local_payload(tmp_path)


def test_install_explicitly_uploads_before_install(monkeypatch):
    calls = []
    def fake(command, data=b""):
        calls.append((command, data))
        return {"status": "uploaded" if data else "installed_not_launched"}
    monkeypatch.setattr(uploader, "ssh_json", fake)
    assert uploader.run("install")["status"] == "installed_not_launched"
    assert len(calls) == 2
    assert calls[0][1] == b"".join(uploader.local_payload().values())
    assert "remote-install" in calls[1][0]
    assert all("remote-launch" not in command for command, _data in calls)


def test_unknown_upload_never_installs_or_retries(monkeypatch):
    calls = []
    def fake(*args):
        calls.append(args)
        return {"status": "unknown_inspect_before_retry"}
    monkeypatch.setattr(uploader, "ssh_json", fake)
    assert uploader.run("install")["status"] == "unknown_inspect_before_retry"
    assert len(calls) == 1


@pytest.mark.parametrize("action", ["launch", "status"])
def test_other_actions_do_not_upload(monkeypatch, action):
    calls = []
    def fake(command, data=b""):
        calls.append((command, data))
        return {"status": "installed_not_launched"}
    monkeypatch.setattr(uploader, "ssh_json", fake)
    uploader.run(action)
    assert len(calls) == 1 and calls[0][1] == b""
    assert "remote-" + action in calls[0][0]


def test_output_never_exports_remote_private_payload(monkeypatch):
    raw = b'{"status":"completed","private":"secret","stages":{"private":"secret"}}'
    monkeypatch.setattr(uploader.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=raw))
    assert uploader.ssh_json("unused") == {"status": "completed"}
    bad = b'{"status":"secret-private-marker"}'
    monkeypatch.setattr(uploader.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=bad))
    assert uploader.ssh_json("unused") == {"status": "unknown_inspect_before_retry"}


def test_existing_stage_and_readonly_exclusive_writes_are_required():
    assert "not os.path.lexists(root)" in uploader.REMOTE
    assert "root.mkdir(mode=0o700)" in uploader.REMOTE
    assert "os.O_EXCL|os.O_NOFOLLOW,0o400" in uploader.REMOTE
    assert "stage" not in uploader.FILES


def test_remote_existing_stage_refused_without_overwrite(tmp_path):
    import subprocess
    import sys
    stage = tmp_path / "existing"
    stage.mkdir(mode=0o700)
    sentinel = stage / "sentinel"
    sentinel.write_bytes(b"preserved")
    config = {"root": str(stage), "files": {}}
    script = ("import os\nos.geteuid=lambda:1000\nC=" + repr(config)
              + "\n" + uploader.REMOTE)
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                          capture_output=True)
    assert done.returncode != 0
    assert sentinel.read_bytes() == b"preserved"
    assert list(stage.iterdir()) == [sentinel]


def test_remote_host_tamper_rejected_before_exec(tmp_path):
    import hashlib
    import subprocess
    import sys
    root = tmp_path / "stage"
    root.mkdir(mode=0o700)
    path = root / uploader.HOST
    path.write_bytes(b"tampered")
    path.chmod(0o400)
    config = {"root": str(root), "files": {
        uploader.HOST: hashlib.sha256(b"accepted").hexdigest()},
        "host": uploader.HOST, "action": "remote-launch"}
    script = ("import os\nos.geteuid=lambda:1000\n"
              "os.execv=lambda *args: (_ for _ in ()).throw(AssertionError('EXEC_CALLED'))\n"
              "C=" + repr(config) + "\n" + uploader.EXEC_REMOTE)
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                          capture_output=True)
    assert done.returncode != 0
    assert b"remote_pin_admission_failed" in done.stderr
    assert b"EXEC_CALLED" not in done.stderr


@pytest.mark.parametrize("status,expected", [
    ("operation_failed_inspect_before_retry", 1),
    ("unknown_inspect_before_retry", 1), ("failed", 1),
    ("detached_supervisor_started", 0)])
def test_cli_failure_exit_code(monkeypatch, capsys, status, expected):
    import sys
    monkeypatch.setattr(sys, "argv", ["uploader", "status"])
    monkeypatch.setattr(uploader, "run", lambda action: {"status": status})
    assert uploader.main() == expected
    assert status in capsys.readouterr().out
