"""Read-only, metadata-only observer for a sealed semantic diagnostic.

The caller must supply the receipt digest and canonical source-manifest digest.
No private request, response, source text, or credential is returned.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

PARENT = Path('/home/atta')
UID = 1000
ROOT_NAME = re.compile(r'\.hymem-luna-semantic-probe-[A-Za-z0-9_-]{8,}\Z')
HEX = re.compile(r'[0-9a-f]{64}\Z')
STOP_CODES = {None, 'transport_or_budget_stop', 'cleanup_failure',
              'private_evidence_write_failure', 'infrastructure_or_runtime_failure',
              'preunit_accounting_invalid', 'postunit_accounting_invalid',
              'hybrid_accounting_invalid', 'hybrid_stage_accounting_invalid',
              'private_result_write_failure', 'finite_other'}
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _read_json(path: Path):
    if not regular(path) or path.stat().st_size > 2_000_000:
        raise ValueError('json_path_invalid')
    for attempt in range(2):
        try:
            return json.loads(path.read_bytes())
        except (OSError, UnicodeError, ValueError):
            if attempt:
                raise ValueError('json_unreadable') from None
    raise ValueError('json_unreadable')


def _host(root: Path, receipt: dict, expected_manifest_sha: str):
    pins = receipt.get('source_sha256')
    if type(pins) is not dict:
        raise ValueError('source_manifest_invalid')
    raw = json.dumps(pins, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha:
        raise ValueError('expected_source_pins_invalid')
    relative = 'tools/diagnostics/luna_semantic_probe_host.py'
    source = root / 'code' / relative
    if not regular(source) or sha(source) != pins.get(relative):
        raise ValueError('host_source_invalid')
    spec = importlib.util.spec_from_file_location('pinned_semantic_host_for_reader', source)
    if spec is None or spec.loader is None:
        raise ValueError('host_import_invalid')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def _systemd(unit: str, cgroup: str) -> dict:
    props = ('ActiveState,SubState,MainPID,ControlGroup,NRestarts,Result,OOMPolicy,ExecMainStatus,'
             'MemoryMax,TasksMax,CPUQuotaPerSecUSec,KillMode,Restart,RemainAfterExit,'
             'RuntimeMaxUSec,TimeoutStopUSec')
    result = subprocess.run(['/usr/bin/systemctl', '--user', 'show', unit,
        '--property=' + props, '--no-pager'], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return {'available': False, 'clean': False, 'active_state': 'unknown'}
    rows = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    required = set(props.split(','))
    if set(rows) != required:
        return {'available': False, 'clean': False, 'active_state': 'unknown'}
    active = rows['ActiveState']
    sub = rows['SubState']
    group = rows['ControlGroup']
    if group not in {'', cgroup}:
        return {'available': True, 'clean': False, 'active_state': 'invalid'}
    policy = (rows['MemoryMax'] == '4294967296' and rows['TasksMax'] == '256'
              and rows['CPUQuotaPerSecUSec'] in {'2s', '2.000s'}
              and rows['KillMode'] == 'control-group' and rows['Restart'] == 'no'
              and rows['RemainAfterExit'] == 'yes' and rows['OOMPolicy'] == 'kill'
              and rows['RuntimeMaxUSec'] in {'32min 10s', '32min 10.000s', '1930s'}
              and rows['TimeoutStopUSec'] in {'10s', '10.000s'})
    cgroup_empty = True
    path = Path('/sys/fs/cgroup' + cgroup)
    if path.exists():
        try:
            events = dict(line.split() for line in (path / 'cgroup.events').read_text().splitlines())
            cgroup_empty = events.get('populated') == '0'
        except (OSError, ValueError):
            cgroup_empty = False
    clean = (active in {'active', 'inactive'} and sub in {'exited', 'dead'}
             and rows['MainPID'] == '0' and rows['NRestarts'] == '0'
             and rows['Result'] == 'success' and rows['OOMPolicy'] == 'kill'
             and rows['ExecMainStatus'] == '0' and cgroup_empty and policy)
    return {'available': True, 'clean': clean,
            'active_state': active if active in {'active', 'inactive', 'failed', 'activating', 'deactivating'} else 'invalid',
            'sub_state': sub if sub in {'exited', 'dead', 'running', 'start', 'stop', 'failed'} else 'invalid',
            'main_pid_zero': rows['MainPID'] == '0', 'n_restarts_zero': rows['NRestarts'] == '0',
            'oom_killed': rows['Result'] == 'oom-kill', 'cgroup_empty': cgroup_empty,
            'resource_policy_verified': policy}


def _owned(root: Path, expected_sha: str | None) -> dict:
    path = root / 'run/private-owned-processes'
    if not path.is_dir() or path.is_symlink():
        return {'verified': False, 'absent': False, 'count': 0}
    identities = []
    files = sorted(path.iterdir())
    for i, file in enumerate(files):
        if file.name != f'{i:04d}.json':
            return {'verified': False, 'absent': False, 'count': len(identities)}
        value = _read_json(file)
        if (type(value) is not dict or set(value) != {'pid', 'pgid', 'starttime', 'index'}
                or type(value['index']) is not int or value['index'] != i or any(type(value[k]) is not int or value[k] <= 0
                                             for k in ('pid', 'pgid', 'starttime'))):
            return {'verified': False, 'absent': False, 'count': len(identities)}
        identities.append(value)
    raw = json.dumps(identities, sort_keys=True, separators=(',', ':')).encode()
    if expected_sha is None or hashlib.sha256(raw).hexdigest() != expected_sha:
        return {'verified': False, 'absent': False, 'count': len(identities)}
    absent = True
    for identity in identities:
        # /proc confirms original PID identity; killpg tests the owned group.
        proc = Path(f"/proc/{identity['pid']}/stat")
        try:
            stat_text = proc.read_text()
            fields = stat_text[stat_text.rfind(')') + 2:].split()
            if int(fields[19]) == identity['starttime']:
                absent = False
        except FileNotFoundError:
            pass
        except (OSError, IndexError, ValueError):
            absent = False
        try:
            import os
            os.killpg(identity['pgid'], 0)
            absent = False
        except ProcessLookupError:
            pass
        except PermissionError:
            absent = False
    return {'verified': True, 'absent': absent, 'count': len(identities)}


def _terminal_valid(terminal: dict, result: dict, receipt_sha: str,
                    manifest_sha: str, failure_serializer=None) -> bool:
    if failure_serializer is None:
        from benchmarks import codex_subscription_warm_v3 as warm
        failure_serializer = warm.serialize_failure
    keys = {'schema', 'receipt_sha256', 'source_sha256', 'process_identity_sha256',
            'process_groups_absent_at_entry_exit', 'core_completed',
            'all_semantic_checks_passed', 'completed_and_clean', 'completed_units',
            'control_outcomes', 'false_support_claims', 'false_rejections',
            'missed_recoveries', 'recheck_failures', 'malformed_units',
            'hybrid_replayed_ordinary_calls', 'hybrid_new_paid_grounding_calls',
            'hybrid_passed', 'paid_budget', 'client_cleanup_ok', 'stop_code',
            'first_failure', 'elapsed_seconds'}
    if (type(terminal) is not dict or set(terminal) != keys or
            terminal['schema'] != 'luna-semantic-probe-terminal-v1' or
            terminal['receipt_sha256'] != receipt_sha or
            terminal['source_sha256'] != manifest_sha or
            terminal['completed_and_clean'] is not False or
            type(result) is not dict or result.get('schema') != 'luna-semantic-probe-v1'):
        return False
    if any(type(terminal[k]) is not bool for k in
           ('process_groups_absent_at_entry_exit', 'core_completed',
            'all_semantic_checks_passed', 'client_cleanup_ok')):
        return False
    if type(terminal['completed_units']) is not int or not 0 <= terminal['completed_units'] <= 25:
        return False
    if (type(result.get('completed_units')) is not int or
            any(type(result.get(k)) is not bool for k in
                ('core_completed', 'all_semantic_checks_passed', 'client_cleanup_ok')) or
            terminal['completed_units'] != result.get('completed_units') or
            terminal['core_completed'] != result.get('core_completed') or
            terminal['all_semantic_checks_passed'] != result.get('all_semantic_checks_passed')):
        return False
    budget = terminal['paid_budget']
    if (type(budget) is not dict or set(budget) != {'turns', 'known_tokens',
            'usage_complete', 'in_flight', 'reserved'} or
            json.dumps(budget, sort_keys=True) != json.dumps(result.get('paid_budget'), sort_keys=True)
            or type(budget['turns']) is not int or not 0 <= budget['turns'] <= 29
            or type(budget['known_tokens']) is not int or budget['known_tokens'] < 0
            or type(budget['usage_complete']) is not bool
            or any(type(budget[k]) is not int or budget[k] < 0 for k in ('in_flight', 'reserved'))):
        return False
    controls = result.get('control_results')
    if type(controls) is not list or len(controls) > 24:
        return False
    names = ('passed', 'false_support', 'false_rejection', 'missed_recovery', 'malformed', 'recheck_failed')
    if type(terminal['control_outcomes']) is not dict or not set(terminal['control_outcomes']) == set(names) or any(type(terminal['control_outcomes'][name]) is not int for name in names) or terminal['control_outcomes'] != {name: sum(type(x) is dict and x.get('outcome') == name for x in controls) for name in names}:
        return False
    for key in ('false_support_claims', 'false_rejections', 'missed_recoveries',
                'recheck_failures', 'malformed_units'):
        if type(terminal[key]) is not int or terminal[key] < 0 or terminal[key] != result.get(key):
            return False
    hybrid = result.get('hybrid')
    for terminal_key, result_key in (('hybrid_replayed_ordinary_calls', 'replayed_ordinary_calls'),
                                     ('hybrid_new_paid_grounding_calls', 'new_paid_grounding_calls'),
                                     ('hybrid_passed', 'passed')):
        if type(terminal[terminal_key]) is not type(None if hybrid is None else hybrid.get(result_key)) or terminal[terminal_key] != (None if hybrid is None else hybrid.get(result_key)):
            return False
    if (terminal['client_cleanup_ok'] != result.get('client_cleanup_ok') or
            type(terminal['elapsed_seconds']) not in (int, float) or
            not 0 <= terminal['elapsed_seconds'] <= 2000 or
            type(terminal['process_identity_sha256']) is not str or
            HEX.fullmatch(terminal['process_identity_sha256']) is None):
        return False
    if ((terminal['stop_code'] is not None and
         (type(terminal['stop_code']) is not str or terminal['stop_code'] not in STOP_CODES)) or
            (terminal['first_failure'] is not None and type(terminal['first_failure']) is not dict)):
        return False
    failure = terminal['first_failure']
    if (type(result.get('first_failure')) not in (dict, type(None)) or
            failure_serializer(failure) != failure or
            json.dumps(failure, sort_keys=True) != json.dumps(
                failure_serializer(result.get('first_failure')), sort_keys=True)):
        return False
    private_stop = result.get('stop_code')
    projected_stop = private_stop if type(private_stop) in (str, type(None)) and private_stop in STOP_CODES else 'finite_other'
    if terminal['stop_code'] != projected_stop:
        return False
    if (terminal['all_semantic_checks_passed'] !=
            (len(controls) == 24 and all(type(x) is dict and x.get('passed') is True
                                         for x in controls) and hybrid is not None
             and hybrid.get('passed') is True)):
        return False
    if terminal['core_completed']:
        if (terminal['completed_units'] != 25 or len(controls) != 24 or
                hybrid is None or budget['in_flight'] != 0 or budget['reserved'] != 0 or
                not budget['usage_complete'] or budget['turns'] > 29 or
                terminal['stop_code'] is not None or
                result.get('stop_code') is not None or
                hybrid.get('hybrid_schedule_complete') is not True or
                not terminal['client_cleanup_ok']):
            return False
        if (type(hybrid) is not dict or
                any(type(item) is not dict or
                    type(item.get('new_calls')) is not int or
                    type(item.get('new_admitted_turns')) is not int or
                    item['new_calls'] != item['new_admitted_turns'] or
                    type(item.get('known_tokens')) is not int or
                    item['known_tokens'] < 0 or
                    item.get('usage_complete') is not True or
                    item.get('client_cleanup_ok') is not True
                    for item in controls) or
                type(hybrid.get('new_paid_grounding_calls')) is not int or
                type(hybrid.get('new_admitted_turns')) is not int or
                hybrid['new_paid_grounding_calls'] != hybrid['new_admitted_turns'] or
                type(hybrid.get('known_tokens')) is not int or
                hybrid['known_tokens'] < 0 or
                hybrid.get('paid_known_tokens') != hybrid['known_tokens'] or
                hybrid.get('usage_complete') is not True or
                hybrid.get('client_cleanup_ok') is not True or
                sum(item['new_admitted_turns'] for item in controls) +
                    hybrid['new_admitted_turns'] != budget['turns'] or
                sum(item['known_tokens'] for item in controls) +
                    hybrid['known_tokens'] != budget['known_tokens']):
            return False
    if terminal['core_completed'] and terminal['all_semantic_checks_passed']:
        if (budget['turns'] != 29 or terminal['hybrid_replayed_ordinary_calls'] != 8
                or terminal['hybrid_new_paid_grounding_calls'] != 3
                or any(terminal[k] != 0 for k in ('false_support_claims',
                    'false_rejections', 'missed_recoveries', 'recheck_failures',
                    'malformed_units'))):
            return False
    return True


def _journal_progress(root: Path) -> dict:
    journal = root / 'run/safe-progress'
    if not journal.is_dir() or journal.is_symlink():
        return {'finished_units': 0, 'paid_turns': None, 'known_tokens': None,
                'usage_complete': None}
    finished = 0
    budget = None
    files = sorted(journal.iterdir())
    for i, path in enumerate(files):
        if not regular(path) or path.name != f'{i:02d}.json':
            raise ValueError('progress_sequence_invalid')
        value = _read_json(path)
        if (type(value) is not dict or set(value) != {'schema', 'finished_units',
                'paid_turns', 'known_tokens', 'usage_complete', 'stop_present'}
                or value['schema'] != 'luna-semantic-probe-unit-progress-v1'
                or type(value['finished_units']) is not int or value['finished_units'] != i + 1
                or any(type(value[k]) is not int or value[k] < 0 for k in ('paid_turns', 'known_tokens'))
                or value['paid_turns'] > 29 or type(value['usage_complete']) is not bool
                or type(value['stop_present']) is not bool):
            raise ValueError('progress_record_invalid')
        finished += 1
        budget = value
    if finished > 25:
        raise ValueError('journal_unit_count_invalid')
    return {'finished_units': finished,
            'paid_turns': budget['paid_turns'] if budget else None,
            'known_tokens': budget['known_tokens'] if budget else None,
            'usage_complete': budget['usage_complete'] if budget else None}


def _age_seconds(path: Path):
    try:
        if not regular(path):
            return None
        return round(max(0.0, min(time.time() - path.stat().st_mtime, 1_000_000_000.0)), 3)
    except OSError:
        return None


def _memory_available_bytes():
    try:
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(memory['MemAvailable'].split()[0]) * 1024
    except (OSError, ValueError, KeyError, IndexError):
        return None


def observe(root: Path, receipt_sha: str, expected_manifest_sha: str,
            *, systemd=_systemd) -> dict:
    if (not root.is_absolute() or root.parent != PARENT or
            ROOT_NAME.fullmatch(root.name) is None or
            not HEX.fullmatch(receipt_sha) or not HEX.fullmatch(expected_manifest_sha)):
        raise ValueError('arguments_invalid')
    state = root.lstat()
    if not stat.S_ISDIR(state.st_mode) or state.st_uid != UID or state.st_mode & 0o077:
        raise ValueError('root_invalid')
    path = root / 'launch-receipt.json'
    if not regular(path) or sha(path) != receipt_sha:
        raise ValueError('receipt_pin_invalid')
    receipt = _read_json(path)
    host = _host(root, receipt, expected_manifest_sha)
    if not host.strict_equal(receipt, host.receipt_for(root, receipt['source_sha256'], receipt['binary_sha256'])):
        raise ValueError('receipt_invalid')
    host.verify_bundle(root, receipt, require_empty_workdirs=False)
    sys.path.insert(0, str(root / 'code'))
    sys.path.insert(0, str(root / 'candidate'))
    warm_path = root / 'code/benchmarks/codex_subscription_warm_v3.py'
    spec = importlib.util.spec_from_file_location('pinned_semantic_warm_for_reader', warm_path)
    if spec is None or spec.loader is None:
        raise ValueError('warm_import_invalid')
    warm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warm)
    launched = _read_json(root / 'launch-attempt.json') if regular(root / 'launch-attempt.json') else None
    if launched is not None and not host.strict_equal(launched, {'receipt_sha256': receipt_sha, 'one_shot': True}):
        raise ValueError('launch_marker_invalid')
    admission = _read_json(root / 'launch-admission.json') if regular(root / 'launch-admission.json') else None
    expected_admission = {'receipt_sha256': receipt_sha, 'old_luna_stopped': True,
        'old_deepseek_stopped': True, 'memory_floor_bytes': 6 * 1024**3,
        'disk_floor_bytes': 20 * 1024**3, 'memory_floor_met': True,
        'disk_floor_met': True}
    if launched is not None and not host.strict_equal(admission, expected_admission):
        raise ValueError('launch_admission_invalid')
    unit = systemd(receipt['unit'], receipt['expected_cgroup']) if launched else {'available': False, 'clean': False, 'active_state': 'not_launched'}
    terminal = _read_json(root / 'safe-terminal.json') if regular(root / 'safe-terminal.json') else None
    result = _read_json(root / 'run/private-result.json') if regular(root / 'run/private-result.json') else None
    validated = terminal is not None and result is not None and _terminal_valid(
        terminal, result, receipt_sha, expected_manifest_sha, warm.serialize_failure)
    owned = _owned(root, terminal['process_identity_sha256'] if validated else None)
    journal = _journal_progress(root)
    last_progress = (root / 'run/safe-progress' / f"{journal['finished_units'] - 1:02d}.json"
                     if journal['finished_units'] else root / 'run/safe-progress/missing')
    complete_clean = bool(validated and terminal['core_completed'] and
        terminal['all_semantic_checks_passed'] and
        terminal['process_groups_absent_at_entry_exit'] and owned['verified'] and
        owned['absent'] and owned['count'] > 0 and unit['clean'] and launched is not None and
        terminal['client_cleanup_ok'] and terminal['paid_budget']['usage_complete']
        and terminal['paid_budget']['in_flight'] == terminal['paid_budget']['reserved'] == 0)
    return {'schema': 'luna-semantic-probe-progress-v1',
            'receipt_verified': True, 'source_pins_verified': True,
            'candidate_inventory_verified': True, 'retained_bundle_verified': True,
            'launched': launched is not None, 'unit': receipt['unit'],
            'unit_state': unit['active_state'], 'unit_cleanup_verified': unit['clean'],
            'unit_sub_state': unit.get('sub_state'),
            'owned_processes_verified': owned['verified'],
            'owned_process_groups_absent': owned['absent'],
            'owned_process_count': owned['count'], 'terminal_validated': validated,
            'phase': ('terminal' if validated else 'running' if launched and
                      unit.get('active_state') == 'active' and
                      unit.get('sub_state') == 'running' and
                      unit.get('main_pid_zero') is False else
                      'ended_without_valid_terminal' if launched else 'prepared'),
            'completed_units': terminal['completed_units'] if validated else journal['finished_units'],
            'paid_turns': terminal['paid_budget']['turns'] if validated else journal['paid_turns'],
            'known_tokens': terminal['paid_budget']['known_tokens'] if validated else journal['known_tokens'],
            'usage_complete': terminal['paid_budget']['usage_complete'] if validated else journal['usage_complete'],
            'elapsed_seconds': terminal['elapsed_seconds'] if validated else None,
            'memory_available_bytes': _memory_available_bytes(),
            'disk_free_bytes': shutil.disk_usage(PARENT).free,
            'safe_progress_age_seconds': _age_seconds(last_progress),
            'terminal_age_seconds': _age_seconds(root / 'safe-terminal.json'),
            'private_stderr_age_seconds': _age_seconds(root / 'private-launch-stderr.log'),
            'admission_verified': host.strict_equal(admission, expected_admission) if launched else False,
            'memory_floor_bytes': 6 * 1024**3,
            'disk_floor_bytes': 20 * 1024**3,
            'old_runs_stopped_at_launch': host.strict_equal(admission, expected_admission) if launched else False,
            'stop_present': (terminal['stop_code'] is not None if validated else
                             bool(journal['finished_units'] and _read_json(root / 'run/safe-progress' / f"{journal['finished_units'] - 1:02d}.json")['stop_present'])),
            'stop_code': terminal['stop_code'] if validated else None,
            'first_failure_code': terminal['first_failure']['code'] if validated and terminal['first_failure'] is not None else None,
            'first_failure': terminal['first_failure'] if validated else None,
            'control_outcomes': terminal['control_outcomes'] if validated else None,
            'false_support_claims': terminal['false_support_claims'] if validated else None,
            'false_rejections': terminal['false_rejections'] if validated else None,
            'missed_recoveries': terminal['missed_recoveries'] if validated else None,
            'recheck_failures': terminal['recheck_failures'] if validated else None,
            'malformed_units': terminal['malformed_units'] if validated else None,
            'hybrid_replayed_ordinary_calls': terminal['hybrid_replayed_ordinary_calls'] if validated else None,
            'hybrid_new_paid_grounding_calls': terminal['hybrid_new_paid_grounding_calls'] if validated else None,
            'core_completed': terminal['core_completed'] if validated else False,
            'all_semantic_checks_passed': terminal['all_semantic_checks_passed'] if validated else False,
            'completed_and_clean': complete_clean,
            'semantic_fix_accepted': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--expected-source-pins-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        out = observe(Path(args.root), args.receipt_sha256,
                      args.expected_source_pins_sha256)
        print(json.dumps(out, sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps({'schema': 'luna-semantic-probe-progress-v1',
                          'validated': False, 'completed_and_clean': False,
                          'reason': 'observer_validation_failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
