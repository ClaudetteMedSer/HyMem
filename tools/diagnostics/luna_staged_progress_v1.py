"""Read-only, finite observer for the sealed eight-unit staged probe.

The public report contains only counters, finite labels, hashes and service
metadata. Private response and source text never enter the report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

sys.dont_write_bytecode = True

SCHEMA = 'luna-staged-probe-progress-v1'
HEX = re.compile(r'[0-9a-f]{64}\Z')
EARLY = ('luna_staged_host_v1.py', 'luna_staged_run_v1.py',
         'luna_staged_core_v1.py', 'luna_staged_replay_v1.py',
         'luna_staged_progress_v1.py',
         'luna_classification_progress_reference_v3.py')
FAILURE = {'schema': 'luna-staged-probe-terminal-failure-v1',
           'diagnostic_completed': False, 'completed_and_clean': False,
           'semantic_accuracy_accepted': False, 'full_lme_ready': False,
           'paid_budget': None, 'stop_code': 'entrypoint_failure'}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def _read(path: Path, cap: int):
    state = path.lstat()
    _require(stat.S_ISREG(state.st_mode) and 0 <= state.st_size <= cap,
             'record_invalid')
    return json.loads(path.read_bytes())


def _optional(path: Path, cap: int):
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    return _read(path, cap)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, 'module_load_invalid')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _early(root: Path, receipt_sha: str, source_sha: str):
    _require(type(receipt_sha) is str and HEX.fullmatch(receipt_sha) is not None and
             type(source_sha) is str and HEX.fullmatch(source_sha) is not None,
             'digest_invalid')
    receipt_path = root / 'launch-receipt.json'
    raw = receipt_path.read_bytes() if stat.S_ISREG(receipt_path.lstat().st_mode) else b''
    _require(_sha(raw) == receipt_sha, 'receipt_pin_invalid')
    receipt = json.loads(raw)
    pins = receipt.get('source_sha256')
    _require(type(pins) is dict and _sha(_canonical(pins)) == source_sha,
             'source_manifest_invalid')
    for name in EARLY:
        relative = 'tools/diagnostics/' + name
        expected = pins.get(relative)
        path = root / 'code' / relative
        _require(type(expected) is str and HEX.fullmatch(expected) is not None and
                 stat.S_ISREG(path.lstat().st_mode) and _sha(path.read_bytes()) == expected,
                 'early_source_pin_invalid')
    return receipt


def _progress(root: Path) -> dict:
    directory = root / 'run/safe-progress'
    try:
        state = directory.lstat()
    except FileNotFoundError:
        return {'finished_units': 0, 'paid_turns': None, 'known_tokens': None,
                'usage_complete': None, 'last_path': directory / 'missing',
                'records': []}
    _require(stat.S_ISDIR(state.st_mode), 'progress_directory_invalid')
    files = sorted(directory.iterdir())
    _require(len(files) <= 8, 'progress_count_invalid')
    previous_turns = previous_tokens = 0
    last = None
    records = []
    prior_stop = prior_unknown = False
    for index, path in enumerate(files):
        _require(path.name == f'{index:02d}.json', 'progress_sequence_invalid')
        value = _read(path, 10_000)
        _require(type(value) is dict and set(value) == {'schema', 'finished_units',
            'paid_turns', 'known_tokens', 'usage_complete', 'stop_present'} and
            value['schema'] == 'luna-staged-probe-unit-progress-v1' and
            type(value['finished_units']) is int and value['finished_units'] == index + 1 and
            type(value['paid_turns']) is int and previous_turns <= value['paid_turns'] <= 29 and
            type(value['known_tokens']) is int and previous_tokens <= value['known_tokens'] and
            type(value['usage_complete']) is bool and
            type(value['stop_present']) is bool, 'progress_record_invalid')
        _require(index == 0 or not prior_stop, 'progress_after_stop_invalid')
        _require(index == 0 or not prior_unknown, 'progress_after_unknown_usage_invalid')
        _require(value['paid_turns'] - previous_turns <= 3,
                 'progress_unit_turns_invalid')
        prior_stop = value['stop_present']
        prior_unknown = not value['usage_complete']
        previous_turns, previous_tokens, last = value['paid_turns'], value['known_tokens'], path
        records.append(value)
    return {'finished_units': len(files), 'paid_turns': previous_turns if last else None,
            'known_tokens': previous_tokens if last else None,
            'usage_complete': value['usage_complete'] if last else None,
            'last_path': last or directory / 'missing', 'records': records}


def _journal_progress_matches(root: Path, records: list[dict]) -> bool:
    """The replay verifies this journal; compare every safe counter to it."""
    try:
        directory = root / 'run/private-journal'
        _require(stat.S_ISDIR(directory.lstat().st_mode), 'journal_directory_invalid')
        finished = []
        for path in sorted(directory.iterdir()):
            if re.fullmatch(r'\d{4}-unit-\d{2}\.json', path.name) is None:
                continue
            value = _read(path, 2_000_000)
            if type(value) is dict and value.get('phase') == 'unit_finished':
                budget = value.get('budget')
                _require(type(budget) is dict, 'journal_budget_invalid')
                finished.append({'schema': 'luna-staged-probe-unit-progress-v1',
                    'finished_units': len(finished) + 1,
                    'paid_turns': budget['turns'],
                    'known_tokens': budget['known_tokens'],
                    'usage_complete': budget['usage_complete'],
                    'stop_present': budget['stopped']})
        return _canonical(finished) == _canonical(records)
    except BaseException:
        return False


def _owned(root: Path, expected_sha: str | None, attempted: int | None,
           *, proc_root: Path = Path('/proc'), killpg=None) -> dict:
    directory = root / 'run/private-owned-processes'
    try:
        state = directory.lstat()
    except FileNotFoundError:
        return {'verified': False, 'absent': False, 'count': 0,
                'dispatched_covered': False, 'tracking_failure': False}
    _require(stat.S_ISDIR(state.st_mode), 'owned_directory_invalid')
    files = sorted(directory.iterdir())
    marker = directory / 'tracking-failure.json'
    tracking_failure = marker in files
    if tracking_failure:
        failure = _read(marker, 10_000)
        _require(type(failure) is dict and set(failure) ==
                 {'tracking_failed', 'unit_key'} and
                 failure['tracking_failed'] is True and
                 type(failure['unit_key']) is str and
                 failure['unit_key'] in {f'unit-{i:02d}' for i in
                                          range(8 if attempted is None else attempted)},
                 'tracking_marker_invalid')
        files.remove(marker)
    _require(len(files) <= 256, 'owned_count_invalid')
    identities = []
    seen_pid = set()
    seen_group = set()
    keys = {f'unit-{i:02d}' for i in range(8 if attempted is None else attempted)}
    for index, path in enumerate(files):
        _require(path.name == f'{index:04d}.json', 'owned_sequence_invalid')
        item = _read(path, 10_000)
        _require(type(item) is dict and set(item) ==
                 {'pid', 'pgid', 'starttime', 'index', 'unit_key'} and
                 type(item['index']) is int and item['index'] == index and
                 all(type(item[k]) is int and item[k] > 0 for k in
                     ('pid', 'pgid', 'starttime')) and
                 type(item['unit_key']) is str and item['unit_key'] in keys and
                 item['pid'] not in seen_pid and item['pgid'] not in seen_group,
                 'owned_identity_invalid')
        seen_pid.add(item['pid'])
        seen_group.add(item['pgid'])
        identities.append(item)
    digest_valid = expected_sha is not None and _sha(_canonical(identities)) == expected_sha
    absent = True
    check_group = killpg or os.killpg
    for item in identities:
        try:
            raw = (proc_root / str(item['pid']) / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            if int(fields[19]) == item['starttime']:
                absent = False
        except FileNotFoundError:
            pass
        except (OSError, IndexError, ValueError):
            absent = False
        try:
            check_group(item['pgid'], 0)
            absent = False
        except ProcessLookupError:
            pass
        except PermissionError:
            absent = False
    covered = attempted is not None and all(
        any(item['unit_key'] == key for item in identities) for key in keys)
    return {'verified': digest_valid and not tracking_failure,
            'absent': absent and not tracking_failure, 'count': len(identities),
            'dispatched_covered': covered, 'tracking_failure': tracking_failure}


def _terminal_valid(terminal, result, receipt_sha, source_sha, entry):
    try:
        _require(type(terminal) is dict and type(result) is dict and
                 terminal.get('schema') == 'luna-staged-probe-terminal-v1' and
                 terminal.get('receipt_sha256') == receipt_sha and
                 terminal.get('source_sha256') == source_sha and
                 type(terminal.get('process_identity_sha256')) is str and
                 HEX.fullmatch(terminal['process_identity_sha256']) is not None and
                 type(terminal.get('process_groups_absent_at_entry_exit')) is bool and
                 type(terminal.get('elapsed_seconds')) in (int, float) and
                 math.isfinite(terminal['elapsed_seconds']) and
                 0 <= terminal['elapsed_seconds'] <= 2000,
                 'terminal_shape_invalid')
        projected = entry._safe_result(result, receipt_sha, source_sha,
            terminal['process_identity_sha256'],
            terminal['process_groups_absent_at_entry_exit'],
            terminal['elapsed_seconds'])
        return _canonical(projected) == _canonical(terminal)
    except BaseException:
        return False


def _failure_valid(terminal, receipt_sha):
    try:
        return type(terminal) is dict and _canonical(terminal) == _canonical(
            {**FAILURE, 'receipt_sha256': receipt_sha})
    except BaseException:
        return False


def _systemd(unit: str, cgroup: str, reference) -> dict:
    """Use the pinned reference's independent service and resource checks."""
    original = reference.subprocess
    reference.subprocess = subprocess
    try:
        return reference._systemd(unit, cgroup)
    finally:
        reference.subprocess = original


def observe(root: Path, receipt_sha: str, expected_source_pins_sha256: str,
            *, systemd=None, proc_root: Path = Path('/proc'), killpg=None) -> dict:
    root = Path(root)
    receipt = _early(root, receipt_sha, expected_source_pins_sha256)
    entry = _load('pinned_staged_entry_for_observer',
        root / 'code/tools/diagnostics/luna_staged_run_v1.py')
    receipt, loaded, proof, _ = entry.preflight(root, receipt_sha, postrun=True)
    host = loaded[0]
    reference = _load('pinned_staged_reference_for_observer',
        root / 'code/tools/diagnostics/luna_classification_progress_reference_v3.py')
    replay = _load('pinned_staged_replay_for_observer',
        root / 'code/tools/diagnostics/luna_staged_replay_v1.py')
    marker = _optional(root / 'launch-attempt.json', 10_000)
    admission = _optional(root / 'launch-admission.json', 10_000)
    launched = host.strict_equal(marker, {'receipt_sha256': receipt_sha,
                                           'one_shot': True})
    expected_admission = {'receipt_sha256': receipt_sha,
        'old_luna_stopped': True, 'old_deepseek_stopped': True,
        'memory_floor_bytes': 6 * 1024**3, 'disk_floor_bytes': 20 * 1024**3,
        'memory_floor_met': True, 'disk_floor_met': True}
    admission_verified = host.strict_equal(admission, expected_admission) if launched else False
    _require((marker is None and admission is None) or
             (launched and admission_verified), 'launch_admission_invalid')
    service = ((systemd or (lambda unit, cgroup: _systemd(unit, cgroup, reference)))(
                   receipt['unit'], receipt['expected_cgroup'])
               if launched else {'available': False, 'clean': False,
                                 'active_state': 'not_launched'})
    _require(type(service) is dict and type(service.get('clean')) is bool and
             type(service.get('active_state')) is str,
             'service_state_invalid')
    terminal = _optional(root / 'safe-terminal.json', 30_000)
    result = _optional(root / 'run/private-result.json', 300_000)
    valid = _terminal_valid(terminal, result, receipt_sha,
                            expected_source_pins_sha256, entry)
    failure_terminal = _failure_valid(terminal, receipt_sha)
    _require(terminal is None or valid or failure_terminal or
             type(terminal) is dict, 'terminal_record_invalid')
    attempted = result['attempted_units'] if valid else None
    owned = _owned(root, terminal['process_identity_sha256'] if valid else None,
                   attempted, proc_root=proc_root, killpg=killpg)
    proof_replay = None
    if valid:
        try:
            proof_replay = replay.replay(root, receipt_sha,
                receipt['source_sha256']['tools/diagnostics/luna_staged_run_v1.py'])
        except BaseException:
            pass
    replay_validated = (type(proof_replay) is dict and
                        proof_replay.get('verified') is True)
    progress = _progress(root)
    progress_verified = bool(valid and replay_validated and
        _journal_progress_matches(root, progress['records']) and
        progress['finished_units'] in
            (result['completed_units'], result['completed_units'] + 1))
    if progress_verified and result['diagnostic_completed']:
        turns = tokens = 0
        progress_verified = progress['finished_units'] == 8
        for item, record in zip(result['units'], progress['records']):
            turns += item['admitted_turns']
            tokens += item['known_tokens']
            if (record['paid_turns'] != turns or record['known_tokens'] != tokens or
                    record['usage_complete'] is not True or
                    record['stop_present'] is not False):
                progress_verified = False
    budget = terminal['paid_budget'] if valid else None
    clean = bool(valid and launched and admission_verified and
        terminal['diagnostic_completed'] and replay_validated and
        proof_replay.get('complete') is True and progress_verified and
        terminal['process_groups_absent_at_entry_exit'] and
        owned['verified'] and owned['absent'] and owned['dispatched_covered'] and
        service['clean'] and terminal['client_cleanup_ok'] and
        budget['usage_complete'] and budget['in_flight'] == budget['reserved'] == 0)
    failed_service_cleanup = bool(service.get('available') is True and
        service.get('active_state') == 'failed' and
        service.get('main_pid_zero') is True and
        service.get('n_restarts_zero') is True and
        service.get('cgroup_empty') is True and
        service.get('resource_policy_verified') is True)
    return {'schema': SCHEMA, 'receipt_verified': True,
        'source_pins_verified': True, 'candidate_inventory_verified': True,
        'retained_bundle_verified': True, 'candidate_files': proof['candidate_files'],
        'launched': launched, 'admission_verified': admission_verified,
        'unit': receipt['unit'], 'unit_state': service['active_state'],
        'unit_sub_state': service.get('sub_state'), 'unit_cleanup_verified': service['clean'],
        'failed_unit_cleanup_verified': failed_service_cleanup,
        'resource_policy_verified': service.get('resource_policy_verified'),
        'main_pid_zero': service.get('main_pid_zero'),
        'cgroup_empty': service.get('cgroup_empty'),
        'n_restarts_zero': service.get('n_restarts_zero'),
        'oom_killed': service.get('oom_killed'),
        'owned_processes_verified': owned['verified'],
        'owned_process_groups_absent': owned['absent'],
        'owned_process_count': owned['count'],
        'dispatched_processes_covered': owned['dispatched_covered'],
        'process_tracking_failure': owned['tracking_failure'],
        'terminal_validated': valid, 'failure_terminal_validated': failure_terminal,
        'replay_validated': replay_validated,
        'replay_complete': proof_replay['complete'] if replay_validated else False,
        'replayed_returned_responses': proof_replay['returned_responses'] if replay_validated else None,
        'replayed_stages': proof_replay['replayed_stages'] if replay_validated else None,
        'replay_result_sha256': proof_replay['private_result_sha256'] if replay_validated else None,
        'progress_verified': progress_verified,
        'phase': 'terminal' if valid else 'terminal_failure' if failure_terminal else
                 'running' if launched and service.get('active_state') == 'active' and
                 service.get('sub_state') == 'running' and
                 service.get('main_pid_zero') is False else
                 'ended_without_valid_terminal' if launched else 'prepared',
        'completed_units': terminal['completed_units'] if valid else None,
        'finished_unit_records': progress['finished_units'],
        'attempted_units': terminal['attempted_units'] if valid else None,
        'paid_turns': budget['turns'] if valid else progress['paid_turns'],
        'known_tokens': budget['known_tokens'] if valid else progress['known_tokens'],
        'usage_complete': budget['usage_complete'] if valid else None,
        'settled_prefix_usage_complete': progress['usage_complete'],
        'diagnostic_completed': terminal['diagnostic_completed'] if valid else False,
        'malformed_units': terminal['malformed_units'] if valid else None,
        'expected_gold_matches': terminal['expected_gold_matches'] if valid else None,
        'unit_outcomes': [{key: item[key] for key in
            ('ordinal', 'kind', 'index', 'outcome', 'error_code',
             'expected_gold_match', 'admitted_turns')}
            for item in result['units']] if valid else None,
        'stop_code': terminal['stop_code'] if valid or failure_terminal else None,
        'first_failure': terminal['first_failure'] if valid else None,
        'memory_available_bytes': reference._memory_available_bytes(),
        'disk_free_bytes': shutil.disk_usage(entry.TASK_HOME_ROOT).free,
        'safe_progress_age_seconds': reference._age_seconds(progress['last_path']),
        'terminal_age_seconds': reference._age_seconds(root / 'safe-terminal.json'),
        'private_stderr_age_seconds': reference._age_seconds(root / 'private-launch-stderr.log'),
        'completed_and_clean': clean, 'semantic_accuracy_accepted': False,
        'full_lme_ready': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--expected-source-pins-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(observe(args.root, args.receipt_sha256,
              args.expected_source_pins_sha256), sort_keys=True, allow_nan=False))
        return 0
    except BaseException:
        print(json.dumps({'schema': SCHEMA, 'validated': False,
            'completed_and_clean': False, 'semantic_accuracy_accepted': False,
            'full_lme_ready': False, 'reason': 'observer_validation_failed'},
            sort_keys=True))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
