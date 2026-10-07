"""Finite, source-bound observer for the claim-task diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def _json(path: Path, cap: int = 300_000):
    state = path.lstat()
    if not stat.S_ISREG(state.st_mode) or state.st_size > cap:
        raise ValueError('record_invalid')
    return json.loads(path.read_bytes())


def _systemd(unit: str, cgroup: str) -> dict:
    from tools.diagnostics import luna_classification_progress_reference_v3 as reference
    original = reference.subprocess
    reference.subprocess = subprocess
    try:
        return reference._systemd(unit, cgroup)
    finally:
        reference.subprocess = original


def _journal_progress(root: Path) -> dict:
    directory = root / 'run/safe-progress'
    if not directory.is_dir() or directory.is_symlink():
        return {'finished_units': 0, 'paid_turns': None,
                'known_tokens': None, 'usage_complete': None}
    files = sorted(directory.iterdir())
    if len(files) > 29:
        raise ValueError('progress_count_invalid')
    last = None
    for index, path in enumerate(files):
        if path.name != f'{index:02d}.json':
            raise ValueError('progress_sequence_invalid')
        value = _json(path, 10_000)
        if (type(value) is not dict or set(value) != {'schema', 'finished_units',
                'paid_turns', 'known_tokens', 'usage_complete', 'stop_present'} or
                value['schema'] != 'luna-claim-task-probe-unit-progress-v1' or
                type(value['finished_units']) is not int or value['finished_units'] != index + 1 or
                type(value['paid_turns']) is not int or not 0 <= value['paid_turns'] <= 29 or
                type(value['known_tokens']) is not int or value['known_tokens'] < 0 or
                type(value['usage_complete']) is not bool or
                type(value['stop_present']) is not bool):
            raise ValueError('progress_record_invalid')
        last = value
    return {'finished_units': len(files),
            'paid_turns': last['paid_turns'] if last else None,
            'known_tokens': last['known_tokens'] if last else None,
            'usage_complete': last['usage_complete'] if last else None}


def _terminal_valid(terminal: dict, result: dict, receipt_sha: str,
                    manifest_sha: str) -> bool:
    from tools.diagnostics import luna_claim_task_run_v1 as entry
    from tools.diagnostics import luna_claim_task_core_v1 as core
    try:
        core.validate_public_result(result)
        if (type(terminal) is not dict or terminal.get('schema') !=
                'luna-claim-task-probe-terminal-v1' or
                terminal.get('receipt_sha256') != receipt_sha or
                terminal.get('source_sha256') != manifest_sha or
                terminal.get('completed_and_clean') is not False or
                terminal.get('semantic_accuracy_accepted') is not False or
                terminal.get('full_lme_ready') is not False or
                type(terminal.get('process_groups_absent_at_entry_exit')) is not bool or
                type(terminal.get('process_identity_sha256')) is not str or
                re.fullmatch(r'[0-9a-f]{64}', terminal['process_identity_sha256']) is None or
                type(terminal.get('elapsed_seconds')) not in (int, float) or
                not 0 <= terminal['elapsed_seconds'] <= 2000):
            return False
        expected = entry._safe_result(result, receipt_sha, manifest_sha,
            terminal['process_identity_sha256'],
            terminal['process_groups_absent_at_entry_exit'],
            terminal['elapsed_seconds'])
        return json.dumps(expected, sort_keys=True, allow_nan=False) == json.dumps(
            terminal, sort_keys=True, allow_nan=False)
    except BaseException:
        return False


def observe(root: Path, receipt_sha: str, expected_manifest_sha: str,
            *, systemd=None) -> dict:
    receipt_path = root / 'launch-receipt.json'
    if (not receipt_path.is_file() or receipt_path.is_symlink() or
            hashlib.sha256(receipt_path.read_bytes()).hexdigest() != receipt_sha):
        raise ValueError('receipt_pin_invalid')
    early = json.loads(receipt_path.read_bytes())
    pins = early.get('source_sha256')
    if type(pins) is not dict:
        raise ValueError('manifest_invalid')
    for name in ('luna_semantic_probe_host.py', 'luna_claim_task_run_v1.py',
                 'luna_claim_task_core_v1.py', 'luna_claim_task_replay_v1.py',
                 'luna_classification_progress_reference_v3.py'):
        relative = 'tools/diagnostics/' + name
        path = root / 'code' / relative
        expected = pins.get(relative)
        if (type(expected) is not str or re.fullmatch(r'[0-9a-f]{64}', expected) is None or
                not path.is_file() or path.is_symlink() or
                hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('early_source_pin_invalid')
    sys.path.insert(0, str(root / 'code'))
    sys.path.insert(0, str(root / 'candidate'))
    from tools.diagnostics import luna_semantic_probe_host as host
    from tools.diagnostics import luna_classification_progress_reference_v3 as reference
    from tools.diagnostics import luna_claim_task_replay_v1 as replay
    if (not host.root_valid(root) or not host.HEX.fullmatch(receipt_sha) or
            not host.HEX.fullmatch(expected_manifest_sha) or
            host.sha(root / 'launch-receipt.json') != receipt_sha):
        raise ValueError('receipt_or_root_invalid')
    receipt = _json(root / 'launch-receipt.json')
    if not host.strict_equal(receipt, host.receipt_for(root,
            receipt['source_sha256'], receipt['binary_sha256'])):
        raise ValueError('receipt_invalid')
    host.verify_bundle(root, receipt, require_empty_workdirs=False)
    manifest = hashlib.sha256(json.dumps(receipt['source_sha256'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if manifest != expected_manifest_sha:
        raise ValueError('manifest_invalid')
    marker = _json(root / 'launch-attempt.json') if host.regular(root / 'launch-attempt.json') else None
    admission = _json(root / 'launch-admission.json') if host.regular(root / 'launch-admission.json') else None
    expected_admission = {'receipt_sha256': receipt_sha, 'old_luna_stopped': True,
        'old_deepseek_stopped': True, 'memory_floor_bytes': 6 * 1024**3,
        'disk_floor_bytes': 20 * 1024**3, 'memory_floor_met': True,
        'disk_floor_met': True}
    launched = host.strict_equal(marker, {'receipt_sha256': receipt_sha, 'one_shot': True})
    if marker is not None and not launched or launched and not host.strict_equal(admission, expected_admission):
        raise ValueError('launch_admission_invalid')
    unit = ((systemd or reference._systemd)(receipt['unit'], receipt['expected_cgroup'])
            if launched else {'available': False, 'clean': False, 'active_state': 'not_launched'})
    terminal = _json(root / 'safe-terminal.json') if host.regular(root / 'safe-terminal.json') else None
    result = _json(root / 'run/private-result.json') if host.regular(root / 'run/private-result.json') else None
    valid = terminal is not None and result is not None and _terminal_valid(
        terminal, result, receipt_sha, manifest)
    owned = reference._owned(root, terminal['process_identity_sha256'] if valid else None)
    replay_proof = None
    if valid:
        try:
            replay_proof = replay.replay(root, receipt_sha,
                receipt['source_sha256']['tools/diagnostics/luna_claim_task_run_v1.py'])
        except BaseException:
            pass
    replay_validated = type(replay_proof) is dict and replay_proof.get('verified') is True
    clean = bool(valid and terminal['diagnostic_completed'] and replay_validated and
        replay_proof.get('complete') is True and
        terminal['process_groups_absent_at_entry_exit'] and owned['verified'] and
        owned['absent'] and owned['count'] == 29 and unit['clean'] and launched and
        terminal['client_cleanup_ok'] and terminal['paid_budget']['usage_complete'] and
        terminal['paid_budget']['turns'] == 29 and
        terminal['paid_budget']['in_flight'] == terminal['paid_budget']['reserved'] == 0)
    progress = _journal_progress(root)
    failed_unit_cleanup = bool(unit.get('available') is True and
        unit.get('active_state') == 'failed' and unit.get('main_pid_zero') is True and
        unit.get('n_restarts_zero') is True and unit.get('cgroup_empty') is True and
        unit.get('resource_policy_verified') is True)
    last = (root / 'run/safe-progress' / f"{progress['finished_units'] - 1:02d}.json"
            if progress['finished_units'] else root / 'run/safe-progress/missing')
    return {'schema': 'luna-claim-task-probe-progress-v1',
            'receipt_verified': True, 'source_pins_verified': True,
            'candidate_inventory_verified': True, 'retained_bundle_verified': True,
            'launched': launched, 'unit': receipt['unit'],
            'unit_state': unit['active_state'], 'unit_cleanup_verified': unit['clean'],
            'failed_unit_cleanup_verified': failed_unit_cleanup,
            'unit_sub_state': unit.get('sub_state'),
            'resource_policy_verified': unit.get('resource_policy_verified'),
            'main_pid_zero': unit.get('main_pid_zero'),
            'cgroup_empty': unit.get('cgroup_empty'),
            'n_restarts_zero': unit.get('n_restarts_zero'),
            'oom_killed': unit.get('oom_killed'),
            'owned_processes_verified': owned['verified'],
            'owned_process_groups_absent': owned['absent'],
            'owned_process_count': owned['count'],
            'terminal_validated': valid, 'replay_validated': replay_validated,
            'replay_complete': replay_proof['complete'] if replay_validated else False,
            'replayed_returned_responses': replay_proof['returned_responses'] if replay_validated else None,
            'replay_result_sha256': replay_proof['private_result_sha256'] if replay_validated else None,
            'phase': 'terminal' if valid else 'running' if launched and
                unit.get('active_state') == 'active' and
                unit.get('sub_state') == 'running' and
                unit.get('main_pid_zero') is False else 'ended_without_valid_terminal' if launched else 'prepared',
            'completed_units': terminal['completed_units'] if valid else progress['finished_units'],
            'paid_turns': terminal['paid_budget']['turns'] if valid else progress['paid_turns'],
            'known_tokens': terminal['paid_budget']['known_tokens'] if valid else progress['known_tokens'],
            'usage_complete': terminal['paid_budget']['usage_complete'] if valid else progress['usage_complete'],
            'diagnostic_completed': terminal['diagnostic_completed'] if valid else False,
            'malformed_units': terminal['malformed_units'] if valid else None,
            'stop_code': terminal['stop_code'] if valid else None,
            'first_failure': terminal['first_failure'] if valid else None,
            'memory_available_bytes': reference._memory_available_bytes(),
            'disk_free_bytes': shutil.disk_usage(host.TASK_HOME_ROOT).free,
            'safe_progress_age_seconds': reference._age_seconds(last),
            'terminal_age_seconds': reference._age_seconds(root / 'safe-terminal.json'),
            'private_stderr_age_seconds': reference._age_seconds(root / 'private-launch-stderr.log'),
            'admission_verified': host.strict_equal(admission, expected_admission) if launched else False,
            'completed_and_clean': clean, 'semantic_accuracy_accepted': False,
            'full_lme_ready': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--expected-source-pins-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(observe(Path(args.root), args.receipt_sha256,
            args.expected_source_pins_sha256), sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps({'schema': 'luna-claim-task-probe-progress-v1',
            'validated': False, 'completed_and_clean': False,
            'semantic_accuracy_accepted': False, 'full_lme_ready': False,
            'reason': 'observer_validation_failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
