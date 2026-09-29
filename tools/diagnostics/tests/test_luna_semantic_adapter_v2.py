"""Offline checks for the versioned semantic startup adapter."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).parents[1] / 'luna_semantic_probe_adapter_v2.py'
spec = importlib.util.spec_from_file_location('semantic_adapter_v2_tested', SOURCE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_bus_missing_and_wrong_owner():
    mode = SimpleNamespace(st_mode=0o40700, st_uid=1000)
    sock = SimpleNamespace(st_mode=0o140600, st_uid=999)
    original_lstat = Path.lstat
    def missing(path):
        if path == adapter.RUNTIME:
            raise FileNotFoundError
        return original_lstat(path)
    def states(path):
        if path == adapter.RUNTIME:
            return mode
        if path == adapter.BUS:
            return sock
        return original_lstat(path)
    with patch.object(adapter.os, 'getuid', return_value=1000), \
         patch.object(Path, 'lstat', missing):
        with pytest.raises(FileNotFoundError):
            adapter.bus_environment()
    with patch.object(adapter.os, 'getuid', return_value=1000), \
         patch.object(Path, 'lstat', states):
        with pytest.raises(ValueError, match='trusted_user_bus_invalid'):
            adapter.bus_environment()


def test_containment_scopes_bus_to_systemctl_and_restores_module():
    calls = []
    def raw(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0)
    module = SimpleNamespace(run=raw)
    def verify(root, receipt):
        assert fake_run.subprocess is not module
        fake_run.subprocess.run(['/usr/bin/systemctl', '--user', 'show', 'unit.service',
            '--property=ActiveState', '--no-pager'], capture_output=True)
        with pytest.raises(ValueError, match='containment_command_invalid'):
            fake_run.subprocess.run(['/usr/bin/systemctl', '--user', 'stop', 'unit.service'])
    fake_run = SimpleNamespace(subprocess=module, verify_live_containment=verify)
    with patch.object(adapter, 'bus_environment', return_value={'XDG_RUNTIME_DIR': '/run/user/1000'}):
        adapter.contained(Path('/unused'), {'unit': 'unit.service'}, fake_run)
    assert fake_run.subprocess is module
    assert calls[0][1]['env'] == {'XDG_RUNTIME_DIR': '/run/user/1000'}


def test_mode_and_adapter_source_pinned(tmp_path):
    root = tmp_path
    (root / 'launch-receipt.json').write_bytes(b'receipt')
    (root / adapter.ADAPTER).write_bytes(SOURCE.read_bytes())
    base = adapter.digest(root / 'launch-receipt.json')
    sidecar = {'schema': 'luna-semantic-probe-adapter-v2',
               'base_receipt_sha256': base,
               'adapter_sha256': adapter.digest(root / adapter.ADAPTER),
               'entry': adapter.ADAPTER, 'entry_action': 'smoke',
               'bus_scope': 'systemctl-only',
               'startup_failure_schema': 'luna-semantic-probe-startup-failure-v2'}
    (root / adapter.SIDECAR).write_text(json.dumps(sidecar))
    sidecar_sha = adapter.digest(root / adapter.SIDECAR)
    adapter.verify(root, base, sidecar['adapter_sha256'], 'smoke', sidecar_sha)
    with pytest.raises(ValueError, match='adapter_pin_invalid'):
        adapter.verify(root, base, sidecar['adapter_sha256'], 'inference', sidecar_sha)
    sidecar['entry_action'] = 'inference'
    (root / adapter.SIDECAR).write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match='adapter_pin_invalid'):
        adapter.verify(root, base, sidecar['adapter_sha256'], 'inference', sidecar_sha)
    (root / adapter.ADAPTER).write_bytes(b'changed')
    with pytest.raises(ValueError, match='adapter_pin_invalid'):
        adapter.verify(root, base, sidecar['adapter_sha256'], None, sidecar_sha)


def test_launch_uses_exact_sealed_entry_and_mode(tmp_path):
    root = tmp_path
    old = str(root / 'code/tools/diagnostics/luna_semantic_probe_run.py')
    captured = []
    def host_launch(_root, _sha, dispatch):
        command = ['/usr/bin/systemd-run', '--user', '--property=MemoryMax=4294967296',
                   '/usr/bin/env', '-i', 'HOME=/home/atta', '/usr/bin/python3',
                   '-I', '-B', old, '--root', str(root), '--receipt-sha256', 'pin']
        dispatch(command, capture_output=True, timeout=20)
        return {'launched': True}
    fake_host = SimpleNamespace(launch=host_launch)
    with patch.object(adapter, 'verify', return_value={}), \
         patch.object(adapter.subprocess, 'run', side_effect=lambda c, **k: captured.append(c)):
        adapter.launch(root, 'pin', 'adapterpin', 'sidecarpin', fake_host, 'smoke')
    index = captured[0].index(str(root / adapter.ADAPTER))
    assert captured[0][index:index + 2] == [str(root / adapter.ADAPTER), 'smoke']
    assert 'XDG_RUNTIME_DIR=' not in ' '.join(captured[0])
    assert captured[0][-2:] == ['--adapter-receipt-sha256', 'sidecarpin']


def test_reader_startup_failure_is_zero_admission_and_failed_cleanup(tmp_path):
    root = tmp_path
    receipt = {'unit': 'unit.service', 'expected_cgroup': '/group',
               'source_sha256': {}, 'binary_sha256': 'bin'}
    (root / 'launch-receipt.json').write_text(json.dumps(receipt))
    (root / 'launch-attempt.json').write_text(json.dumps(
        {'receipt_sha256': 'receipt', 'one_shot': True}))
    (root / 'launch-admission.json').write_text(json.dumps({
        'receipt_sha256': 'receipt', 'old_luna_stopped': True,
        'old_deepseek_stopped': True, 'memory_floor_bytes': 6 * 1024**3,
        'disk_floor_bytes': 20 * 1024**3, 'memory_floor_met': True,
        'disk_floor_met': True}))
    terminal = {'schema': 'luna-semantic-probe-startup-failure-v2',
                'receipt_sha256': 'receipt', 'adapter_sha256': 'adapter',
                'zero_admission_proof': 'run_directory_never_created',
                'paid_turns': 0, 'known_tokens': 0, 'model_calls': 0,
                'completed_and_clean': False,
                'stop_code': 'pre_inference_startup_failure'}
    (root / 'safe-terminal.json').write_text(json.dumps(terminal))
    policy = {'available': True, 'active_state': 'failed', 'main_pid_zero': True,
              'n_restarts_zero': True, 'cgroup_empty': True,
              'resource_policy_verified': True}
    progress = SimpleNamespace(subprocess=SimpleNamespace(run=lambda *a, **k: None),
                               _systemd=lambda *a: policy)
    host = SimpleNamespace(root_valid=lambda r: True, strict_equal=lambda a, b: a == b,
        receipt_for=lambda *a: receipt, verify_bundle=lambda *a, **k: None,
        regular=lambda p: p.is_file())
    with patch.object(adapter, 'verify', return_value={'entry_action': 'inference'}), \
         patch.object(adapter, 'load_source', return_value=host):
        result = adapter.observe(root, 'receipt', 'adapter', 'sidecar', progress)
    assert result['zero_admission_verified'] is True
    assert result['model_calls'] == 0
    assert result['failed_unit_cleanup_verified'] is True
    assert result['completed_and_clean'] is False
    (root / 'run').mkdir()
    with patch.object(adapter, 'verify', return_value={'entry_action': 'inference'}), \
         patch.object(adapter, 'load_source', return_value=host):
        with pytest.raises(AttributeError):
            adapter.observe(root, 'receipt', 'adapter', 'sidecar', progress)


def test_preflight_failure_persists_zero_admission_only_before_run(tmp_path):
    root = tmp_path
    (root / 'launch-attempt.json').write_text('{}')
    def fail(*args, **kwargs):
        raise ValueError('synthetic_preflight_failure')
    run = SimpleNamespace(execute=fail)
    def write_once(path, value):
        path.write_text(json.dumps(value))
    host = SimpleNamespace(root_valid=lambda r: True, write_once=write_once)
    def loader(path, name, expected):
        return run if expected == adapter.RUN_SHA else host
    with patch.object(adapter, 'verify', return_value={}), \
         patch.object(adapter, 'load_source', side_effect=loader):
        assert adapter.entry(root, 'receipt', 'inference', 'adapter', 'sidecar') == 1
    terminal = json.loads((root / 'safe-terminal.json').read_text())
    assert terminal['model_calls'] == terminal['paid_turns'] == 0
    assert terminal['zero_admission_proof'] == 'run_directory_never_created'
    (root / 'safe-terminal.json').unlink()
    (root / 'run').mkdir()
    with patch.object(adapter, 'verify', return_value={}), \
         patch.object(adapter, 'load_source', side_effect=loader):
        assert adapter.entry(root, 'receipt', 'inference', 'adapter', 'sidecar') == 1
    assert not (root / 'safe-terminal.json').exists()


def test_smoke_calls_exact_v1_preflight_and_containment_without_execute(tmp_path):
    root = tmp_path
    record = []
    host = SimpleNamespace(write_once=lambda path, value: record.append(value))
    run = SimpleNamespace(preflight=lambda r, h: ({'unit': 'unit.service'}, (), {}, ()),
                          execute=lambda *a, **k: pytest.fail('inference reached'))
    def loader(path, name, expected):
        return run if expected == adapter.RUN_SHA else host
    with patch.object(adapter, 'verify', return_value={}), \
         patch.object(adapter, 'load_source', side_effect=loader), \
         patch.object(adapter, 'contained', side_effect=lambda *a: record.append('policy')):
        assert adapter.entry(root, 'receipt', 'smoke', 'adapter', 'sidecar') == 0
    assert record[0] == 'policy'
    assert record[1]['policy_verified'] is True
    assert record[1]['model_calls'] == 0
