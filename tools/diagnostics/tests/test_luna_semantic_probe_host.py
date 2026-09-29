"""Offline faults for the semantic host and metadata observer; no service launch."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_semantic_probe_host as host
from tools.diagnostics import luna_semantic_probe_progress as progress
from tools.diagnostics import luna_semantic_probe_run as runner


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(host, 'TASK_HOME_ROOT', tmp_path)
    monkeypatch.setattr(host, 'UID', os.getuid())
    root = tmp_path / '.hymem-luna-semantic-probe-offline01'
    root.mkdir(mode=0o700)
    pins = dict(host.ACCEPTED)
    pins.update({name: '1' * 64 for name in host.NEW})
    receipt = host.receipt_for(root, pins, '2' * 64)
    host.write_once(root / 'launch-receipt.json', receipt)
    monkeypatch.setattr(host, 'verify_bundle', lambda *a, **k: None)
    monkeypatch.setattr(host, 'host_admission', lambda: None)
    return root, receipt


def test_receipt_type_drift_rejected_before_dispatch(sealed):
    root, receipt = sealed
    changed = copy.deepcopy(receipt)
    changed['policy']['tasks_max'] = 256.0
    (root / 'launch-receipt.json').write_text(json.dumps(changed))
    called = []
    with pytest.raises(ValueError, match='receipt_invalid'):
        host.launch(root, host.sha(root / 'launch-receipt.json'),
                    dispatch=lambda *a, **k: called.append(1))
    assert called == []


def test_launch_consumed_even_if_dispatch_raises(sealed):
    root, _ = sealed
    attempts = []
    def fail(*a, **k):
        attempts.append(1)
        raise OSError('PRIVATE_COMMAND_TEXT')
    digest = host.sha(root / 'launch-receipt.json')
    out = host.launch(root, digest, dispatch=fail)
    assert out['dispatch_ambiguous'] and out['never_retry']
    assert 'PRIVATE_' not in str(out)
    with pytest.raises(FileExistsError):
        host.launch(root, digest, dispatch=fail)
    assert attempts == [1]


def test_preflight_mode_does_not_create_run_or_call_campaign(monkeypatch, tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    monkeypatch.setattr(runner, 'preflight', lambda *a: (
        {}, (), {'candidate_files': 510}, tuple(range(8))))
    out = runner.execute(root, '3' * 64, preflight_only=True)
    assert out['verified'] and out['model_calls'] == 0
    assert not (root / 'run').exists()


def test_progress_record_is_finite_and_rejects_extra_fields(tmp_path):
    root = tmp_path / 'root'
    folder = root / 'run/safe-progress'
    folder.mkdir(parents=True)
    good = {'schema': 'luna-semantic-probe-unit-progress-v1',
            'finished_units': 1, 'paid_turns': 1, 'known_tokens': 17,
            'usage_complete': True, 'stop_present': False}
    (folder / '00.json').write_text(json.dumps(good))
    assert progress._journal_progress(root) == {
        'finished_units': 1, 'paid_turns': 1, 'known_tokens': 17,
        'usage_complete': True}
    good['private_response'] = 'DO_NOT_EXPORT'
    (folder / '00.json').write_text(json.dumps(good))
    with pytest.raises(ValueError, match='progress_record_invalid'):
        progress._journal_progress(root)


def test_systemd_requires_empty_expected_cgroup_even_with_blank_property(monkeypatch):
    unit = 'hymem-luna-semantic-probe-offline01.service'
    cgroup = '/user.slice/user-1000.slice/user@1000.service/app.slice/' + unit
    payload = '\n'.join(f'{k}={v}' for k, v in {
        'ActiveState': 'active', 'SubState': 'exited', 'MainPID': '0',
        'ControlGroup': '', 'NRestarts': '0', 'Result': 'success',
        'OOMPolicy': 'kill', 'ExecMainStatus': '0', 'MemoryMax': '4294967296',
        'TasksMax': '256', 'CPUQuotaPerSecUSec': '2s',
        'KillMode': 'control-group', 'Restart': 'no',
        'RemainAfterExit': 'yes', 'RuntimeMaxUSec': '32min 10s',
        'TimeoutStopUSec': '10s'}.items())
    monkeypatch.setattr(progress.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=payload))
    # On local machines the remote cgroup path is absent; the shape still
    # verifies that a successful exit alone is insufficient for clean status.
    ok = progress._systemd(unit, cgroup)
    assert ok['clean'] and ok['main_pid_zero'] and ok['n_restarts_zero']


def test_terminal_rejects_unknown_usage_and_extra_attributes():
    budget = {'turns': 0, 'known_tokens': 0, 'usage_complete': False,
              'in_flight': 0, 'reserved': 0}
    result = {'schema': 'luna-semantic-probe-v1', 'completed_units': 0,
              'core_completed': False, 'all_semantic_checks_passed': False,
              'paid_budget': budget, 'control_results': [], 'hybrid': None,
              'false_support_claims': 0, 'false_rejections': 0,
              'missed_recoveries': 0, 'recheck_failures': 0,
              'malformed_units': 0, 'client_cleanup_ok': True}
    terminal = {'schema': 'luna-semantic-probe-terminal-v1',
        'receipt_sha256': '1' * 64, 'source_sha256': '2' * 64,
        'process_identity_sha256': hashlib.sha256(b'[]').hexdigest(),
        'process_groups_absent_at_entry_exit': True,
        'core_completed': False, 'all_semantic_checks_passed': False,
        'completed_and_clean': False, 'completed_units': 0,
        'control_outcomes': {name: 0 for name in
            ('passed', 'false_support', 'false_rejection', 'missed_recovery',
             'malformed', 'recheck_failed')},
        'false_support_claims': 0, 'false_rejections': 0,
        'missed_recoveries': 0, 'recheck_failures': 0, 'malformed_units': 0,
        'hybrid_replayed_ordinary_calls': None,
        'hybrid_new_paid_grounding_calls': None, 'hybrid_passed': None,
        'paid_budget': budget, 'client_cleanup_ok': True,
        'stop_code': None, 'first_failure': None, 'elapsed_seconds': 1.0}
    assert progress._terminal_valid(terminal, result, '1' * 64, '2' * 64)
    terminal['extra'] = 'PRIVATE'
    assert not progress._terminal_valid(terminal, result, '1' * 64, '2' * 64)


def test_private_result_failure_keeps_known_paid_usage_in_safe_terminal(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    root.mkdir()
    class Journal:
        def __init__(self, directory):
            self.directory = directory
        def record(self, *args):
            pass
    budget = {'turns': 1, 'known_tokens': 123, 'usage_complete': False,
              'in_flight': 0, 'reserved': 0}
    result = {'schema': 'luna-semantic-probe-v1', 'control_results': [],
        'hybrid': None, 'false_support_claims': 0, 'false_rejections': 0,
        'missed_recoveries': 0, 'recheck_failures': 0, 'malformed_units': 0,
        'completed_units': 0, 'paid_budget': budget, 'first_failure': None,
        'client_cleanup_ok': True, 'stop_code': 'transport_or_budget_stop',
        'core_completed': False, 'all_semantic_checks_passed': False}
    core = SimpleNamespace(PrivateJournal=Journal, run_campaign=lambda **kwargs: result)
    warm = SimpleNamespace(WarmSession=object, WarmSubscriptionClient=object,
                           BudgetLimits=object, serialize_failure=lambda value: None)
    monkeypatch.setattr(runner, 'preflight', lambda *a: (
        {'source_sha256': {}}, (host, core, None, None, None, None, None, None,
                               warm, None), {}, ()))
    original = host.write_once
    def write(path, value):
        if path.name == 'private-result.json':
            raise OSError('PRIVATE_PATH_DETAIL')
        return original(path, value)
    monkeypatch.setattr(host, 'write_once', write)
    safe = runner.execute(root, '1' * 64, containment=lambda *a: None)
    assert safe['paid_budget'] == budget
    assert safe['stop_code'] == 'private_result_write_failure'
    assert not safe['core_completed'] and not safe['completed_and_clean']
    assert 'PRIVATE_' not in json.dumps(safe)
    assert json.loads((root / 'safe-terminal.json').read_text()) == safe
