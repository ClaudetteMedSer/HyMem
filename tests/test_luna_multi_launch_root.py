"""Independent offline launch controls; never invoke systemd or a model."""
import importlib.util
import json
from pathlib import Path
import types

SOURCE = Path(__file__).resolve().parents[1] / 'tools/diagnostics/luna_subscription_multi_launch.py'
spec = importlib.util.spec_from_file_location('multi_launch_root_controls', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_exact_envelope_and_isolated_command():
    root = Path('/home/atta/.hymem-luna-lme-multi-test')
    cmd = m.command(root, {'unit': 'hymem-luna-lme-multi-test.service'})
    assert '--property=Restart=no' in cmd
    assert '--property=KillMode=control-group' in cmd
    assert '--property=RuntimeMaxSec=14530s' in cmd
    assert '--property=MemoryMax=4294967296' in cmd
    assert '--property=OOMPolicy=kill' in cmd
    env = cmd.index('/usr/bin/env')
    assert cmd[env:env+6] == ['/usr/bin/env', '-i', 'HOME=/home/atta',
        'PATH=/usr/local/bin:/usr/bin:/bin', 'TMPDIR=' + str(root / 'tmp'), '/usr/bin/python3']
    assert cmd[env+6:env+8] == ['-I', '-B']
    assert '--resume' not in cmd and '--api-key' not in cmd
    assert cmd[cmd.index('--questions')+1] == '2'
    assert cmd[cmd.index('--workers')+1] == '2'
    assert cmd[cmd.index('--campaign-turns')+1] == '4012'
    assert cmd[cmd.index('--campaign-known-tokens')+1] == '24160000'
    assert cmd[cmd.index('--output-dir')+1] == str(root / 'run')


def test_marker_never_overwrites(tmp_path):
    path = tmp_path / 'launch-attempt.json'
    m.write_once(path, {'one_shot': True})
    import pytest
    with pytest.raises(FileExistsError):
        m.write_once(path, {'one_shot': False})
    assert json.loads(path.read_text()) == {'one_shot': True}
    assert path.stat().st_mode & 0o777 == 0o600


def test_host_admission_refuses_running_peer(monkeypatch):
    monkeypatch.setattr(m.sys, 'platform', 'linux')
    monkeypatch.setattr(m.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(m.Path, 'read_text', lambda _: 'MemAvailable: 9000000 kB')
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda _: types.SimpleNamespace(free=30*1024**3))
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(
        stdout='hymem-luna-other.service loaded active running\n'))
    import pytest
    with pytest.raises(ValueError, match='prior_unit_running'):
        m.host_admission()


def test_host_admission_allows_terminal_peers(monkeypatch):
    monkeypatch.setattr(m.sys, 'platform', 'linux')
    monkeypatch.setattr(m.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(m.Path, 'read_text', lambda _: 'MemAvailable: 9000000 kB')
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda _: types.SimpleNamespace(free=30*1024**3))
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(
        stdout='hymem-luna-other.service loaded active exited\n'))
    m.host_admission()
