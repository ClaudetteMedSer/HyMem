"""Offline controls for the four-question profiled launcher; no model or systemd call."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'tools/diagnostics/luna_subscription_profiled_launch.py'
SPEC = importlib.util.spec_from_file_location('profiled_launch_root_controls', SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_exact_envelope_and_private_command():
    root = Path('/home/atta/.hymem-luna-lme-profiled-abcd1234')
    receipt = {'unit': m.unit_for(root)}
    cmd = m.command(root, receipt)
    for property_ in ('Restart=no', 'KillMode=control-group', 'RemainAfterExit=yes',
                      'RuntimeMaxSec=14530s', 'TimeoutStopSec=10s',
                      'MemoryMax=4294967296', 'CPUQuota=200%', 'TasksMax=128',
                      'OOMPolicy=kill', 'UMask=0077'):
        assert '--property=' + property_ in cmd
    env = cmd.index('/usr/bin/env')
    assert cmd[env:env + 9] == ['/usr/bin/env', '-i', 'HOME=/home/atta',
        'PATH=/usr/local/bin:/usr/bin:/bin', 'TMPDIR=' + str(root / 'tmp'),
        '/usr/bin/python3', '-I', '-B', str(root / 'luna_subscription_lme_profiled.py')]
    assert '--resume' not in cmd and '--api-key' not in cmd
    assert '--endpoint' not in cmd and '--model' not in cmd
    assert '--concurrent-transport' in cmd and '--warm-transport' in cmd
    for key, value in m.LIMITS.items():
        assert cmd[cmd.index('--' + key) + 1] == str(value)
    assert cmd[cmd.index('--output-dir') + 1] == str(root / 'run')
    assert '--property=StandardOutput=file:' + str(root / 'safe-terminal.json') in cmd
    assert '--property=StandardError=file:' + str(root / 'private-launch-stderr.log') in cmd


def test_receipt_pins_and_root_unit_identity(monkeypatch):
    monkeypatch.setattr(m, 'sha', lambda _: 'a' * 64)
    root = Path('/home/atta/.hymem-luna-lme-profiled-abcd1234')
    receipt = m.receipt_for(root)
    assert receipt['schema'] == 'luna-profiled-launch-v1'
    assert receipt['unit'] == 'hymem-luna-lme-profiled-abcd1234.service'
    assert receipt['source_sha256'] == m.PINS
    assert receipt['runner_sha256'] == m.PINS['luna_subscription_lme_profiled.py']
    assert receipt['model'] == 'gpt-6-luna'
    assert receipt['subscription_only'] is True
    assert receipt['reported_quota_floor_percent'] == 25
    assert receipt['limits']['questions'] == receipt['limits']['workers'] == 4
    assert receipt['limits']['campaign-turns'] == 8012
    assert receipt['limits']['campaign-known-tokens'] == 48160000
    with pytest.raises(ValueError, match='root_invalid'):
        m.unit_for(Path('/home/atta/.hymem-luna-lme-profiled-../bad'))


def test_consumed_marker_never_overwrites(tmp_path):
    path = tmp_path / 'launch-attempt.json'
    m.write_once(path, {'one_shot': True})
    with pytest.raises(FileExistsError):
        m.write_once(path, {'one_shot': False})
    assert json.loads(path.read_text()) == {'one_shot': True}
    assert path.stat().st_mode & 0o777 == 0o600


def _admission(monkeypatch, listed, shown='MainPID=0\nControlGroup=\n'):
    monkeypatch.setattr(m.sys, 'platform', 'linux')
    monkeypatch.setattr(m.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(m.Path, 'read_text', lambda _: 'MemAvailable: 9000000 kB')
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda _: SimpleNamespace(free=30 * 1024**3))
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=listed if 'list-units' in cmd else shown)
    monkeypatch.setattr(m.subprocess, 'run', run)
    return calls


def test_host_admission_rejects_running_peer(monkeypatch):
    _admission(monkeypatch, 'hymem-luna-other.service loaded active running\n')
    with pytest.raises(ValueError, match='prior_unit_running'):
        m.host_admission()


def test_host_admission_rejects_nonempty_exited_peer(monkeypatch):
    _admission(monkeypatch, 'hymem-luna-other.service loaded active exited\n',
               'MainPID=0\nControlGroup=/user.slice/leftover\n')
    with pytest.raises(ValueError, match='prior_unit_not_empty'):
        m.host_admission()


def test_host_admission_allows_empty_exited_peer(monkeypatch):
    calls = _admission(monkeypatch, 'hymem-luna-other.service loaded active exited\n')
    m.host_admission()
    assert len(calls) == 2


def test_launch_consumes_marker_before_dispatch(monkeypatch, tmp_path, capsys):
    # Patch the fixed host boundary to use a synthetic root; exercise the
    # launch ordering without invoking systemd or loading candidate sources.
    root = tmp_path / '.hymem-luna-lme-profiled-abcd1234'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(m, 'host_admission', lambda: None)
    monkeypatch.setattr(m, 'verify_sources', lambda _: None)
    monkeypatch.setattr(m, 'receipt_for', lambda _: {'unit': m.unit_for(root)})
    monkeypatch.setattr(m, 'sha', lambda _: 'a' * 64)
    monkeypatch.setattr(m, 'HOST_ROOT', tmp_path)
    monkeypatch.setattr(m, 'HOST_UID', root.stat().st_uid)
    (root / 'launch-receipt.json').write_text(json.dumps({'unit': m.unit_for(root)}))
    observed = []
    def dispatch(*args, **kwargs):
        observed.append((root / 'launch-attempt.json').is_file())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess, 'run', dispatch)
    assert m.main(['--launch-root', str(root), '--receipt-sha256', 'a' * 64]) == 0
    assert observed == [True]
    assert m.main(['--launch-root', str(root), '--receipt-sha256', 'a' * 64]) == 1
    assert observed == [True]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['never_retry_launch'] is True
