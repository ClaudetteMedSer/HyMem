"""Offline observer controls with invented records and no service launch."""
from __future__ import annotations

import json

import pytest

from tools.diagnostics import luna_staged_progress_v1 as observer
from tools.diagnostics import luna_staged_run_v1 as run
from tools.diagnostics import luna_staged_core_v1 as core


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _result():
    return {'schema': core.SCHEMA, 'units': [], 'completed_units': 0,
        'attempted_units': 1, 'malformed_units': 0,
        'paid_budget': {'turns': 1, 'known_tokens': 0, 'usage_complete': False,
                        'in_flight': 1, 'reserved': 1},
        'first_failure': None, 'client_cleanup_ok': False,
        'stop_code': 'infrastructure_or_runtime_failure',
        'diagnostic_completed': False, 'semantic_accuracy_accepted': False,
        'full_lme_ready': False, 'completed_and_clean': False,
        'process_cleanup_verified': False}


def test_terminal_exact_projection_and_private_result_write_failure():
    result = _result()
    terminal = run._safe_result(result, 'a' * 64, 'b' * 64, 'c' * 64,
                                False, 1.125)
    assert observer._terminal_valid(terminal, result, 'a' * 64, 'b' * 64, run)
    assert terminal['paid_budget']['usage_complete'] is False
    changed = dict(terminal, expected_gold_matches=1)
    assert not observer._terminal_valid(changed, result, 'a' * 64, 'b' * 64, run)
    changed = dict(terminal, process_groups_absent_at_entry_exit=1)
    assert not observer._terminal_valid(changed, result, 'a' * 64, 'b' * 64, run)
    failure = {**observer.FAILURE, 'receipt_sha256': 'a' * 64}
    assert observer._failure_valid(failure, 'a' * 64)
    assert not observer._terminal_valid(failure, None, 'a' * 64, 'b' * 64, run)
    assert not observer._failure_valid(dict(failure, paid_budget={'turns': 0}), 'a' * 64)


def test_variable_process_count_and_scheduled_ownership(tmp_path):
    identities = [
        {'pid': 101, 'pgid': 201, 'starttime': 301, 'index': 0, 'unit_key': 'unit-00'},
        {'pid': 102, 'pgid': 202, 'starttime': 302, 'index': 1, 'unit_key': 'unit-00'},
        {'pid': 103, 'pgid': 203, 'starttime': 303, 'index': 2, 'unit_key': 'unit-01'},
    ]
    for i, item in enumerate(identities):
        _write(tmp_path / 'run/private-owned-processes' / f'{i:04d}.json', item)
    digest = observer._sha(observer._canonical(identities))
    def absent(_pgid, _signal):
        raise ProcessLookupError
    owned = observer._owned(tmp_path, digest, 2, proc_root=tmp_path / 'proc',
                            killpg=absent)
    assert owned == {'verified': True, 'absent': True, 'count': 3,
                     'dispatched_covered': True, 'tracking_failure': False}
    assert not observer._owned(tmp_path, '0' * 64, 2, proc_root=tmp_path / 'proc',
                               killpg=absent)['verified']
    assert not observer._owned(tmp_path, None, None, proc_root=tmp_path / 'proc',
                               killpg=absent)['dispatched_covered']
    _write(tmp_path / 'run/private-owned-processes/0002.json',
           dict(identities[2], pgid=202))
    with pytest.raises(ValueError, match='owned_identity_invalid'):
        observer._owned(tmp_path, digest, 2, proc_root=tmp_path / 'proc', killpg=absent)


def test_live_pid_group_and_tracking_failure_fail_closed(tmp_path):
    item = {'pid': 101, 'pgid': 201, 'starttime': 301, 'index': 0,
            'unit_key': 'unit-00'}
    _write(tmp_path / 'run/private-owned-processes/0000.json', item)
    digest = observer._sha(observer._canonical([item]))
    def present(_pgid, _signal):
        return None
    assert not observer._owned(tmp_path, digest, 1, proc_root=tmp_path / 'proc',
                               killpg=present)['absent']
    proc = tmp_path / 'proc/101/stat'
    proc.parent.mkdir(parents=True)
    proc.write_text('101 (warm session) ' + ' '.join(['S'] + ['0'] * 18 + ['301']))
    def absent(_pgid, _signal):
        raise ProcessLookupError
    assert not observer._owned(tmp_path, digest, 1, proc_root=tmp_path / 'proc',
                               killpg=absent)['absent']
    proc.unlink()
    _write(tmp_path / 'run/private-owned-processes/tracking-failure.json',
           {'tracking_failed': True, 'unit_key': 'unit-00'})
    owned = observer._owned(tmp_path, digest, 1, proc_root=tmp_path / 'proc',
                            killpg=absent)
    assert not owned['verified'] and not owned['absent'] and owned['tracking_failure']


def test_progress_partial_tail_and_unknown_usage(tmp_path):
    directory = tmp_path / 'run/safe-progress'
    first = {'schema': 'luna-staged-probe-unit-progress-v1',
             'finished_units': 1, 'paid_turns': 1, 'known_tokens': 7,
             'usage_complete': True, 'stop_present': False}
    second = dict(first, finished_units=2, usage_complete=False, stop_present=True)
    _write(directory / '00.json', first)
    _write(directory / '01.json', second)
    value = observer._progress(tmp_path)
    assert value['finished_units'] == 2 and value['paid_turns'] == 1
    assert value['usage_complete'] is False
    _write(directory / '02.json', dict(second, finished_units=3))
    with pytest.raises(ValueError, match='progress_after_stop_invalid'):
        observer._progress(tmp_path)
    (directory / '02.json').unlink()
    _write(directory / '01.json', dict(second, known_tokens=6))
    with pytest.raises(ValueError, match='progress_record_invalid'):
        observer._progress(tmp_path)


def test_progress_journal_exact_budget_binding(tmp_path):
    first = {'schema': 'luna-staged-probe-unit-progress-v1',
             'finished_units': 1, 'paid_turns': 2, 'known_tokens': 11,
             'usage_complete': True, 'stop_present': False}
    second = dict(first, finished_units=2, paid_turns=3, known_tokens=15)
    budget = lambda record: {'turns': record['paid_turns'],
        'known_tokens': record['known_tokens'],
        'usage_complete': record['usage_complete'],
        'stopped': record['stop_present']}
    _write(tmp_path / 'run/private-journal/0001-unit-00.json',
           {'phase': 'unit_finished', 'budget': budget(first)})
    _write(tmp_path / 'run/private-journal/0002-unit-01.json',
           {'phase': 'unit_finished', 'budget': budget(second)})
    assert observer._journal_progress_matches(tmp_path, [first, second])
    assert not observer._journal_progress_matches(tmp_path,
        [first, dict(second, known_tokens=16)])
