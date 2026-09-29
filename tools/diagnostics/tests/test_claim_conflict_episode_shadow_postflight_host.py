from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from tools.diagnostics import claim_conflict_episode_shadow_postflight_host as host
from tools.diagnostics import claim_conflict_episode_shadow_postflight_install as installer


def test_configure_network_none_readonly_entire_source_no_env():
    seal = "a" * 64
    command, mounts = host.configure(SimpleNamespace(RUNTIME=Path("/runtime")), seal)
    assert command[command.index("--network") + 1] == "none"
    assert "--env" not in command and "--env-file" not in command
    assert [dst for _, dst, rw in mounts if rw] == ["/work"]
    assert (str(host.ROOT / "work/live"), "/private-dream", False) in mounts
    assert all("runtime-env" not in src and "runtime-env" not in dst for src, dst, _ in mounts)
    assert command[-2:] == ["--source-sha256", seal]
    assert "--read-only" in command


@pytest.mark.parametrize("change", [{"pid": 1}, {"exit_code": 1}, {"status": "running"}, {"oom_killed": True}, {"configuration_verified": False}])
def test_seal_cannot_accept_unfinished_worker(change):
    state = dict(status="exited", pid=0, exit_code=0, oom_killed=False, configuration_verified=True)
    state.update(change)
    with pytest.raises(RuntimeError, match="container_not_clean"):
        host.clean(state)


def test_required_seal_and_source_pin(tmp_path):
    value = {name: "a" * 64 for name in installer.SEAL_FIELDS}
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps(value))
    seal.chmod(0o400)
    assert host.read_seal(seal, host.sha(seal)) == value
    source = tmp_path / "hymem.sqlite"
    source.write_bytes(b"sealed")
    with pytest.raises(RuntimeError, match="pin_drift"):
        host.closed_source(source, "b" * 64)
    Path(str(source) + "-wal").write_bytes(b"pending")
    with pytest.raises(RuntimeError, match="unsealed_sidecar"):
        host.closed_source(source, host.sha(source))


def test_installer_local_pins_match():
    assert set(installer.files()) == set(installer.PINS)


def test_install_does_not_require_finished_source(monkeypatch):
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    installer.dispatch(SimpleNamespace(action="install"))
    payload = json.loads(calls[0][1]["input"])
    assert set(payload) == {"root", "files"}
    assert "work/live/hymem.sqlite" not in installer.REMOTE_INSTALL
    assert "result.json" not in installer.REMOTE_INSTALL


@pytest.mark.parametrize("action", ["seal", "run"])
def test_no_ready_or_explicit_seal_never_calls_ssh(monkeypatch, action):
    def unexpected(*args, **kwargs):
        raise AssertionError("should not invoke SSH")
    monkeypatch.setattr(installer.subprocess, "run", unexpected)
    args = SimpleNamespace(action=action, ready=False, seal_json=None, seal_sha256=None)
    with pytest.raises(ValueError, match="explicit_ready_required"):
        installer.dispatch(args)
    args.ready = True
    with pytest.raises(ValueError, match="explicit_seal"):
        installer.dispatch(args)


def test_seal_uploads_only_explicit_pins_without_running(tmp_path, monkeypatch):
    source = tmp_path / "seal.json"
    value = {name: "a" * 64 for name in installer.SEAL_FIELDS}
    source.write_text(json.dumps(value))
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    installer.dispatch(SimpleNamespace(action="seal", ready=True, seal_json=source))
    command, options = calls[0]
    assert "--prepare-seal" in command[-1]
    assert "--ready" in command[-1]
    assert json.loads(options["input"]) == value


def test_host_prepare_seal_validates_readiness_before_writing(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    value = {name: "a" * 64 for name in installer.SEAL_FIELDS}
    import hashlib
    from io import StringIO
    seal_sha = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    monkeypatch.setattr(host, "STAGE", stage)
    monkeypatch.setattr(host.sys, "stdin", StringIO(json.dumps(value)))
    monkeypatch.setattr(host.sys, "argv", ["host", "--seal", str(stage / "seal.json"), "--seal-sha256", seal_sha, "--ready", "--prepare-seal"])
    def pending(_seal):
        raise RuntimeError("container_not_clean")
    monkeypatch.setattr(host, "ready", pending)
    with pytest.raises(RuntimeError, match="container_not_clean"):
        host.main()
    assert not (stage / "seal.json").exists()
