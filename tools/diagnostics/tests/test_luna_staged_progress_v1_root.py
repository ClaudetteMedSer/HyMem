"""Root observer controls: real runner/replay journals and invented OS metadata."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_staged_progress_v1 as reader
from tools.diagnostics import luna_staged_run_v1 as entry, luna_staged_replay_v1 as replay
from tools.diagnostics.tests.test_luna_staged_replay_v1_root import execute_fixture


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def gone(*args):
    raise ProcessLookupError


def observer_fixture(monkeypatch, tmp_path, **kwargs):
    root, entry_sha, terminal = execute_fixture(monkeypatch, tmp_path, **kwargs)
    receipt, loaded, proof, retained = entry.preflight(root, '1'*64)
    host = loaded[0]
    receipt.update(unit='invented.service', expected_cgroup='/invented')
    receipt['source_sha256']['tools/diagnostics/luna_staged_run_v1.py'] = entry_sha
    source_sha = hashlib.sha256(canonical(receipt['source_sha256'])).hexdigest()
    host.strict_equal = lambda a,b: canonical(a) == canonical(b)
    monkeypatch.setattr(entry, 'TASK_HOME_ROOT', tmp_path)
    reference = SimpleNamespace(_memory_available_bytes=lambda: 8*1024**3,
        _age_seconds=lambda p: 0 if p.exists() else None)
    original_load = reader._load
    def load(name, path):
        if path.name == 'luna_staged_run_v1.py':
            return entry
        if path.name == 'luna_staged_replay_v1.py':
            return replay
        if path.name == 'luna_classification_progress_reference_v3.py':
            return reference
        return original_load(name, path)
    monkeypatch.setattr(reader, '_load', load)
    monkeypatch.setattr(reader, '_early', lambda *a: receipt)
    identities = [dict(pid=900000+i, pgid=900000+i, starttime=100+i,
        index=i, unit_key=f'unit-{i:02d}') for i in range(terminal['attempted_units'])]
    owned = root / 'run/private-owned-processes'
    for i, value in enumerate(identities):
        (owned / f'{i:04d}.json').write_bytes(canonical(value))
    terminal['source_sha256'] = source_sha
    terminal['process_identity_sha256'] = hashlib.sha256(canonical(identities)).hexdigest()
    (root/'safe-terminal.json').write_bytes(canonical(terminal))
    (root/'launch-attempt.json').write_bytes(canonical(dict(receipt_sha256='1'*64, one_shot=True)))
    (root/'launch-admission.json').write_bytes(canonical(dict(receipt_sha256='1'*64,
        old_luna_stopped=True, old_deepseek_stopped=True, memory_floor_bytes=6*1024**3,
        disk_floor_bytes=20*1024**3, memory_floor_met=True, disk_floor_met=True)))
    service = dict(available=True, clean=True, active_state='active', sub_state='exited',
        main_pid_zero=True, n_restarts_zero=True, oom_killed=False,
        cgroup_empty=True, resource_policy_verified=True)
    def observe():
        return reader.observe(root, '1'*64, source_sha, systemd=lambda *a: service,
            proc_root=tmp_path/'proc', killpg=gone)
    return root, terminal, service, observe


@pytest.mark.parametrize('fault', ['none', 'malformed', 'transport', 'unknown_usage', 'cleanup', 'overshoot'])
def test_real_journals_cleanup_and_quality_remain_distinct(monkeypatch, tmp_path, fault):
    root, terminal, service, observe = observer_fixture(monkeypatch, tmp_path, fault=fault)
    out = observe()
    assert out['terminal_validated'] and out['replay_validated']
    assert out['owned_processes_verified'] and out['owned_process_groups_absent']
    assert out['completed_and_clean'] == (fault in ('none', 'malformed'))
    assert out['paid_turns'] == terminal['paid_budget']['turns']
    assert out['known_tokens'] == terminal['paid_budget']['known_tokens']
    assert out['usage_complete'] == (fault != 'unknown_usage')
    assert out['malformed_units'] == (fault == 'malformed')
    assert not out['semantic_accuracy_accepted'] and not out['full_lme_ready']
    assert 'PRIVATE_' not in json.dumps(out)


def test_gold_success_still_not_a_model_quality_or_full_benchmark_claim(monkeypatch, tmp_path):
    root, terminal, service, observe = observer_fixture(monkeypatch, tmp_path, mode='gold')
    out = observe()
    assert out['completed_and_clean'] and out['expected_gold_matches'] == 8
    assert out['paid_turns'] == 17 and out['owned_process_count'] == 8
    assert not out['semantic_accuracy_accepted'] and not out['full_lme_ready']


@pytest.mark.parametrize('fault', ['empty', 'missing_key', 'extra_key', 'duplicate', 'boolean',
    'reorder', 'tracking_failure', 'symlink', 'extra_file', 'terminal_hash', 'live_pid', 'live_group'])
def test_ownership_faults_never_pass(monkeypatch, tmp_path, fault):
    root, terminal, service, observe = observer_fixture(monkeypatch, tmp_path)
    directory = root/'run/private-owned-processes'
    if fault == 'empty':
        for path in directory.iterdir():
            path.unlink()
        terminal['process_identity_sha256'] = hashlib.sha256(canonical([])).hexdigest()
    elif fault == 'terminal_hash':
        terminal['process_identity_sha256'] = '0'*64
    elif fault == 'tracking_failure':
        (directory/'tracking-failure.json').write_bytes(canonical(dict(tracking_failed=True, unit_key='unit-03')))
    elif fault == 'extra_file':
        (directory/'EXTRA').write_text('{}')
    elif fault == 'symlink':
        path = directory/'0001.json'
        path.unlink()
        path.symlink_to(directory/'0000.json')
    elif fault == 'live_pid':
        proc = tmp_path/'proc/900000'; proc.mkdir(parents=True)
        (proc/'stat').write_text('900000 (worker with spaces) '+ ' '.join(['0']*19 + ['100']))
    elif fault == 'live_group':
        monkeypatch.setattr(sys.modules[__name__], 'gone', lambda *a: None)
    else:
        path = directory/'0003.json'
        value = json.loads(path.read_bytes())
        if fault == 'missing_key':
            value['unit_key'] = 'unit-02'
        elif fault == 'extra_key':
            value['unit_key'] = 'unit-08'
        elif fault == 'duplicate':
            value['pid'] = 900002
        elif fault == 'boolean':
            value['starttime'] = True
        elif fault == 'reorder':
            value['index'] = 2
        path.write_bytes(canonical(value))
        # Bind the edited ledger to test its own semantics, not just its checksum.
        terminal['process_identity_sha256'] = hashlib.sha256(canonical(
            [json.loads(p.read_bytes()) for p in sorted(directory.iterdir())])).hexdigest()
    (root/'safe-terminal.json').write_bytes(canonical(terminal))
    try:
        out = observe()
    except ValueError:
        return
    assert not out['completed_and_clean']


@pytest.mark.parametrize('fault', ['progress_usage', 'progress_stop', 'progress_prefix_tokens',
    'progress_prefix_turns', 'progress_missing', 'terminal_tokens', 'terminal_extra',
    'terminal_boolean', 'replay_response', 'marker_bool', 'admission_bool', 'unit_failed'])
def test_independent_integrity_sources_each_veto_completion(monkeypatch, tmp_path, fault):
    root, terminal, service, observe = observer_fixture(monkeypatch, tmp_path)
    if fault.startswith('progress_'):
        path = root/'run/safe-progress/03.json'
        value = json.loads(path.read_bytes())
        if fault == 'progress_usage':
            value['usage_complete'] = False
        elif fault == 'progress_stop':
            value['stop_present'] = True
        elif fault == 'progress_prefix_tokens':
            value['known_tokens'] += 1
        elif fault == 'progress_prefix_turns':
            value['paid_turns'] += 1
        elif fault == 'progress_missing':
            path.unlink()
        if fault != 'progress_missing':
            path.write_bytes(canonical(value))
    elif fault.startswith('terminal_'):
        if fault == 'terminal_tokens':
            terminal['paid_budget']['known_tokens'] += 1
        elif fault == 'terminal_extra':
            terminal['unreviewed'] = 'PRIVATE'
        else:
            terminal['completed_and_clean'] = 0
        (root/'safe-terminal.json').write_bytes(canonical(terminal))
    elif fault == 'replay_response':
        path = next(p for p in sorted((root/'run/private-journal').iterdir())
            if json.loads(p.read_bytes()).get('phase') == 'response_returned')
        value = json.loads(path.read_bytes()); value['response'] = 'PRIVATE'
        path.write_bytes(canonical(value))
    elif fault == 'marker_bool':
        (root/'launch-attempt.json').write_bytes(canonical(dict(receipt_sha256='1'*64, one_shot=1)))
    elif fault == 'admission_bool':
        path = root/'launch-admission.json'; value = json.loads(path.read_bytes())
        value['old_deepseek_stopped'] = 1; path.write_bytes(canonical(value))
    else:
        service.update(clean=False, active_state='failed')
    try:
        out = observe()
    except ValueError:
        return
    assert not out['completed_and_clean']


@pytest.mark.parametrize('field', ['diagnostic_completed', 'completed_and_clean',
    'semantic_accuracy_accepted', 'full_lme_ready'])
def test_failure_terminal_does_not_accept_integers_as_booleans(field):
    value = dict(reader.FAILURE, receipt_sha256='1'*64)
    assert reader._failure_valid(value, '1'*64)
    value[field] = 0
    assert not reader._failure_valid(value, '1'*64)


def test_partial_terminal_failure_does_not_invent_zero_usage(monkeypatch, tmp_path):
    root, terminal, service, observe = observer_fixture(monkeypatch, tmp_path, fault='transport')
    (root/'safe-terminal.json').write_bytes(canonical(dict(reader.FAILURE, receipt_sha256='1'*64)))
    service.update(clean=False, active_state='failed')
    out = observe()
    assert out['failure_terminal_validated'] and not out['completed_and_clean']
    assert out['usage_complete'] is None
    assert out['paid_turns'] is None or out['paid_turns'] > 0


def test_isolated_invalid_cli_is_private_and_finite(tmp_path):
    out = subprocess.run([sys.executable, '-I', '-B', reader.__file__, '--root',
        str(tmp_path/'PRIVATE_MISSING'), '--receipt-sha256', '0'*64,
        '--expected-source-pins-sha256', '0'*64], capture_output=True, text=True, timeout=15)
    assert out.returncode == 1 and out.stderr == '' and 'PRIVATE_' not in out.stdout
    value = json.loads(out.stdout)
    assert value['validated'] is False and not value['completed_and_clean']


@pytest.mark.parametrize('fault', ['none', 'source', 'manifest', 'receipt', 'symlink'])
def test_early_pins_are_verified_before_import(tmp_path, fault):
    root = tmp_path/'early'; root.mkdir()
    pins = {}
    for name in reader.EARLY:
        relative = 'tools/diagnostics/'+name
        path = root/'code'/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b'raise AssertionError("never import during early verification")\n'
        path.write_bytes(raw)
        pins[relative] = hashlib.sha256(raw).hexdigest()
    receipt = dict(source_sha256=pins)
    (root/'launch-receipt.json').write_bytes(canonical(receipt))
    digest = hashlib.sha256(canonical(receipt)).hexdigest()
    source = hashlib.sha256(canonical(pins)).hexdigest()
    if fault == 'source':
        (root/'code/tools/diagnostics'/reader.EARLY[0]).write_text('PRIVATE')
    elif fault == 'manifest':
        source = '0'*64
    elif fault == 'receipt':
        digest = '0'*64
    elif fault == 'symlink':
        path = root/'code/tools/diagnostics'/reader.EARLY[0]
        path.unlink()
        path.symlink_to(path.with_name(reader.EARLY[1]))
    if fault == 'none':
        assert reader._early(root, digest, source) == receipt
    else:
        with pytest.raises(ValueError):
            reader._early(root, digest, source)


def test_systemd_proxy_is_scoped_and_restored_on_failure(monkeypatch):
    original = SimpleNamespace()
    scoped = SimpleNamespace()
    reference = SimpleNamespace(subprocess=original)
    monkeypatch.setattr(reader, 'subprocess', scoped)
    def check(unit, group):
        assert reference.subprocess is scoped and (unit, group) == ('u', 'g')
        raise ValueError('controlled')
    reference._systemd = check
    with pytest.raises(ValueError, match='controlled'):
        reader._systemd('u', 'g', reference)
    assert reference.subprocess is original
