"""Sealed semantic probe entrypoint. ``--preflight-only`` never starts inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

def safe_first_failure(value, serializer=None):
    if serializer is None:
        from benchmarks import codex_subscription_warm_v3 as warm
        serializer = warm.serialize_failure
    return serializer(value)


def _paths(root: Path):
    code = root / 'code'
    # Candidate must win all hymem and original canary imports. The mini code
    # bundle supplies only the new diagnostic modules and accepted transport.
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(root / 'candidate'))
    return code


def preflight(root: Path, receipt_sha: str) -> tuple[dict, object, object, tuple]:
    # Verify the receipt and both modules before importing their code.
    if (not root.is_absolute() or root.parent != Path('/home/atta') or
            re.fullmatch(r'\.hymem-luna-semantic-probe-[A-Za-z0-9_-]{8,}', root.name) is None or
            re.fullmatch(r'[0-9a-f]{64}', receipt_sha) is None):
        raise ValueError('root_or_receipt_invalid')
    for path in (Path('/home/atta'), root, root / 'code', root / 'code/tools',
                 root / 'code/tools/diagnostics'):
        state = path.lstat()
        if not stat.S_ISDIR(state.st_mode):
            raise ValueError('path_alias_or_symlink')
    if root.lstat().st_uid != 1000 or root.lstat().st_mode & 0o077:
        raise ValueError('root_private_invalid')
    receipt_path = root / 'launch-receipt.json'
    if not stat.S_ISREG(receipt_path.lstat().st_mode) or hashlib.sha256(receipt_path.read_bytes()).hexdigest() != receipt_sha:
        raise ValueError('receipt_pin_invalid')
    early_receipt = json.loads(receipt_path.read_bytes())
    pins = early_receipt.get('source_sha256')
    if type(pins) is not dict:
        raise ValueError('source_manifest_invalid')
    for relative, expected in (
        ('tools/diagnostics/luna_semantic_probe.py',
         '3ccb7ec12f93f8502fe8ed1448a071ac977dd334d17821f661913d634c27b334'),
        ('tools/diagnostics/luna_semantic_probe_host.py',
         pins.get('tools/diagnostics/luna_semantic_probe_host.py'))):
        source = root / 'code' / relative
        if (type(expected) is not str or re.fullmatch(r'[0-9a-f]{64}', expected) is None
                or not stat.S_ISREG(source.lstat().st_mode)
                or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
            raise ValueError('early_source_drift')
    import importlib.util
    host_spec = importlib.util.spec_from_file_location('pinned_semantic_host_for_entry',
        root / 'code/tools/diagnostics/luna_semantic_probe_host.py')
    if host_spec is None or host_spec.loader is None:
        raise ValueError('host_import_invalid')
    host = importlib.util.module_from_spec(host_spec)
    host_spec.loader.exec_module(host)
    core_spec = importlib.util.spec_from_file_location('pinned_semantic_core_for_entry',
        root / 'code/tools/diagnostics/luna_semantic_probe.py')
    if core_spec is None or core_spec.loader is None:
        raise ValueError('core_import_invalid')
    core = importlib.util.module_from_spec(core_spec)
    core_spec.loader.exec_module(core)
    if not host.root_valid(root) or not host.HEX.fullmatch(receipt_sha):
        raise core.ProbeStop('root_or_receipt_invalid')
    receipt_path = root / 'launch-receipt.json'
    if not host.regular(receipt_path) or host.sha(receipt_path) != receipt_sha:
        raise core.ProbeStop('receipt_pin_invalid')
    receipt = json.loads(receipt_path.read_bytes())
    if not host.strict_equal(receipt, host.receipt_for(root, receipt['source_sha256'], receipt['binary_sha256'])):
        raise core.ProbeStop('receipt_invalid')
    host.verify_bundle(root, receipt)
    if host.sha(root / 'code/tools/diagnostics/luna_semantic_probe_run.py') != receipt['source_sha256']['tools/diagnostics/luna_semantic_probe_run.py']:
        raise core.ProbeStop('entrypoint_drift')
    _paths(root)
    from tools.diagnostics import luna_semantic_cases as cases
    from hymem.extraction import grounding, chunk
    from benchmarks import extraction_canary as canary
    from benchmarks import luna_semantic_canary as semantic
    from benchmarks import luna_semantic_stage_accounting as stage
    from benchmarks import codex_subscription_warm_v3 as warm
    concurrent = warm.concurrent
    if Path(warm.__file__).resolve() != (root / 'code/benchmarks/codex_subscription_warm_v3.py').resolve():
        raise core.ProbeStop('warm_import_drift')
    proof = core.verify_local(root / 'candidate', root / 'candidate-source-map.json',
                              cases_module=cases, grounding_module=grounding,
                              canary_module=semantic, stage_module=stage)
    if (proof['candidate_files'] != 510 or concurrent is not warm.concurrent or
            concurrent.SharedBudget is not warm.SharedBudget or
            concurrent.BudgetLimits is not warm.BudgetLimits or
            not callable(getattr(warm.SharedBudget, 'record_first_failure', None))):
        raise core.ProbeStop('transport_or_candidate_invalid')
    retained = core.verify_retained_bundle(
        evidence=host.OLD_EVIDENCE.read_bytes(),
        original_receipt=host.OLD_RECEIPT.read_bytes(),
        original_result=host.OLD_RESULT.read_bytes(),
        evidence_sha256_from_new_receipt=receipt['retained_sha256']['evidence'])
    if Path(canary.__file__).resolve() != (root / 'candidate/benchmarks/extraction_canary.py').resolve() or Path(chunk.__file__).resolve() != (root / 'candidate/hymem/extraction/chunk.py').resolve():
        raise core.ProbeStop('canary_import_drift')
    return receipt, (host, core, cases, grounding, canary, chunk, semantic, stage, warm, concurrent), proof, retained


def verify_live_containment(root: Path, receipt: dict) -> None:
    """Require the actual service and effective cgroup limits before inference."""
    from tools.diagnostics import luna_semantic_probe as core
    marker = root / 'launch-attempt.json'
    from tools.diagnostics import luna_semantic_probe_host as host
    expected_marker = {'receipt_sha256': host.sha(root / 'launch-receipt.json'),
                       'one_shot': True}
    if not host.regular(marker) or not host.strict_equal(json.loads(marker.read_bytes()), expected_marker):
        raise core.ProbeStop('launch_marker_invalid')
    props = ('ActiveState,SubState,MainPID,ControlGroup,NRestarts,MemoryMax,TasksMax,'
             'CPUQuotaPerSecUSec,KillMode,Restart,RemainAfterExit,OOMPolicy,'
             'RuntimeMaxUSec,TimeoutStopUSec')
    shown = subprocess.run(['/usr/bin/systemctl', '--user', 'show', receipt['unit'],
        '--property=' + props, '--no-pager'], capture_output=True, text=True,
        check=True, timeout=10)
    rows = dict(line.split('=', 1) for line in shown.stdout.splitlines() if '=' in line)
    if (set(rows) != set(props.split(',')) or rows['ActiveState'] != 'active'
            or rows['SubState'] != 'running' or rows['MainPID'] != str(os.getpid())
            or rows['ControlGroup'] != receipt['expected_cgroup']
            or rows['NRestarts'] != '0' or rows['MemoryMax'] != '4294967296'
            or rows['TasksMax'] != '256' or rows['KillMode'] != 'control-group'
            or rows['Restart'] != 'no' or rows['RemainAfterExit'] != 'yes'
            or rows['OOMPolicy'] != 'kill'
            or rows['CPUQuotaPerSecUSec'] not in {'2s', '2.000s'}
            or rows['RuntimeMaxUSec'] not in {'32min 10s', '32min 10.000s', '1930s'}
            or rows['TimeoutStopUSec'] not in {'10s', '10.000s'}):
        raise core.ProbeStop('service_policy_invalid')
    cgroup = receipt['expected_cgroup']
    groups = Path('/proc/self/cgroup').read_text().splitlines()
    if f'0::{cgroup}' not in groups:
        raise core.ProbeStop('cgroup_identity_invalid')
    path = Path('/sys/fs/cgroup' + cgroup)
    if ((path / 'memory.max').read_text().strip() != '4294967296'
            or (path / 'pids.max').read_text().strip() != '256'):
        raise core.ProbeStop('effective_resource_policy_invalid')
    cpu = (path / 'cpu.max').read_text().split()
    if len(cpu) != 2 or not all(x.isdecimal() for x in cpu) or int(cpu[0]) != 2 * int(cpu[1]):
        raise core.ProbeStop('effective_cpu_policy_invalid')


def _process_identity(pid: int) -> dict:
    raw = Path(f'/proc/{pid}/stat').read_text()
    # comm can contain spaces and parentheses; fields after its final ')' are
    # stable Linux stat fields starting with state (field 3).
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'pgid': int(fields[2]), 'starttime': int(fields[19])}


def _owned_absent(identities: list[dict]) -> bool:
    for item in identities:
        try:
            current = _process_identity(item['pid'])
            if current['starttime'] == item['starttime']:
                return False
            # A reused PID cannot prove its old process group absent.
            os.killpg(item['pgid'], 0)
            return False
        except ProcessLookupError:
            pass
        except PermissionError:
            return False
        except FileNotFoundError:
            try:
                os.killpg(item['pgid'], 0)
                return False
            except ProcessLookupError:
                pass
            except PermissionError:
                return False
    return True


def _safe_result(core_result: dict, receipt_sha: str, source_sha: str, process_sha: str,
                 process_absent: bool, elapsed: float, failure_serializer=None) -> dict:
    # This projection has only fixed labels, numeric counters and hashes. It
    # deliberately carries no fixture, source, request or response text.
    controls = core_result['control_results']
    budget = core_result['paid_budget']
    return {'schema': 'luna-semantic-probe-terminal-v1',
            'receipt_sha256': receipt_sha,
            'source_sha256': source_sha,
            'process_identity_sha256': process_sha,
            'process_groups_absent_at_entry_exit': process_absent,
            'core_completed': core_result['core_completed'],
            'all_semantic_checks_passed': core_result['all_semantic_checks_passed'],
            'completed_and_clean': False,
            'completed_units': core_result['completed_units'],
            'control_outcomes': {name: sum(x['outcome'] == name for x in controls)
                                 for name in ('passed', 'false_support', 'false_rejection',
                                              'missed_recovery', 'malformed', 'recheck_failed')},
            'false_support_claims': core_result['false_support_claims'],
            'false_rejections': core_result['false_rejections'],
            'missed_recoveries': core_result['missed_recoveries'],
            'recheck_failures': core_result['recheck_failures'],
            'malformed_units': core_result['malformed_units'],
            'hybrid_replayed_ordinary_calls': None if core_result['hybrid'] is None else core_result['hybrid']['replayed_ordinary_calls'],
            'hybrid_new_paid_grounding_calls': None if core_result['hybrid'] is None else core_result['hybrid']['new_paid_grounding_calls'],
            'hybrid_passed': None if core_result['hybrid'] is None else core_result['hybrid']['passed'],
            'paid_budget': {key: budget[key] for key in ('turns', 'known_tokens',
                                                         'usage_complete', 'in_flight', 'reserved')},
            'client_cleanup_ok': core_result['client_cleanup_ok'],
            'stop_code': core_result['stop_code'] if core_result['stop_code'] in
                         {None, 'transport_or_budget_stop', 'cleanup_failure',
                          'private_evidence_write_failure', 'infrastructure_or_runtime_failure',
                          'preunit_accounting_invalid', 'postunit_accounting_invalid',
                          'hybrid_accounting_invalid', 'hybrid_stage_accounting_invalid'}
                         else 'finite_other',
            'first_failure': safe_first_failure(core_result['first_failure'], failure_serializer),
            'elapsed_seconds': round(elapsed, 3)}


def execute(root: Path, receipt_sha: str, *, preflight_only: bool = False,
            client_factory=None, containment=verify_live_containment) -> dict:
    started = time.monotonic()
    receipt, loaded, proof, retained = preflight(root, receipt_sha)
    if preflight_only:
        return {'schema': 'luna-semantic-probe-preflight-v1', 'verified': True,
                'receipt_sha256': receipt_sha, 'candidate_files': proof['candidate_files'],
                'retained_ordinary_completions': len(retained), 'model_calls': 0}
    host, core, cases, grounding, canary, chunk, semantic, stage, warm, concurrent = loaded
    containment(root, receipt)
    run = root / 'run'
    run.mkdir(mode=0o700)
    journal_path = run / 'private-journal'
    journal_path.mkdir(mode=0o700)
    progress_path = run / 'safe-progress'
    progress_path.mkdir(mode=0o700)
    owned_path = run / 'private-owned-processes'
    owned_path.mkdir(mode=0o700)
    identities: list[dict] = []

    class TrackingSession(warm.WarmSession):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            finally:
                process = getattr(self, 'process', None)
                if process is not None:
                    try:
                        identity = _process_identity(process.pid)
                    except BaseException:
                        host.write_once(owned_path / 'tracking-failure.json',
                                        {'tracking_failed': True})
                        raise
                    identity['index'] = len(identities)
                    host.write_once(owned_path / f'{len(identities):04d}.json', identity)
                    identities.append(identity)

    def factory(key, cap, budget):
        return warm.WarmSubscriptionClient(str(host.BINARY), budget, key,
            warm.BudgetLimits(*cap), session_factory=TrackingSession,
            max_requests=16, max_age_seconds=300)

    class ProgressJournal(core.PrivateJournal):
        def __init__(self, directory):
            super().__init__(directory)
            self.finished = 0

        def record(self, unit, value):
            super().record(unit, value)
            if value.get('phase') == 'unit_finished':
                state = value['budget']
                host.write_once(progress_path / f'{self.finished:02d}.json', {
                    'schema': 'luna-semantic-probe-unit-progress-v1',
                    'finished_units': self.finished + 1,
                    'paid_turns': state['turns'],
                    'known_tokens': state['known_tokens'],
                    'usage_complete': state['usage_complete'],
                    'stop_present': state['stopped']})
                self.finished += 1

    result = core.run_campaign(concurrent=concurrent, warm=warm,
        binary=str(host.BINARY), cases_module=cases, grounding_module=grounding,
        journal=ProgressJournal(journal_path), retained=retained,
        canary_module=canary, chunk_module=chunk, semantic_module=semantic,
        stage_module=stage, candidate=root / 'candidate',
        client_factory=client_factory or factory)
    private_result_written = True
    try:
        host.write_once(run / 'private-result.json', result)
    except BaseException:
        private_result_written = False
    identity_sha = hashlib.sha256(json.dumps(identities, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    absent = _owned_absent(identities)
    source_sha = hashlib.sha256(json.dumps(receipt['source_sha256'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    safe = _safe_result(result, receipt_sha, source_sha, identity_sha, absent,
                        time.monotonic() - started, warm.serialize_failure)
    if not private_result_written:
        safe['core_completed'] = False
        safe['completed_and_clean'] = False
        safe['stop_code'] = 'private_result_write_failure'
    host.write_once(root / 'safe-terminal.json', safe)
    return safe


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(args.root)
    try:
        out = execute(root, args.receipt_sha256, preflight_only=args.preflight_only)
        if args.preflight_only:
            print(json.dumps(out, sort_keys=True))
        return 0 if args.preflight_only or out['core_completed'] else 1
    except BaseException:
        # Raw exception text is never sent to stdout or the safe terminal.
        try:
            from tools.diagnostics import luna_semantic_probe_host as host
            if host.root_valid(root) and (root / 'run').is_dir() and not (root / 'safe-terminal.json').exists():
                host.write_once(root / 'safe-terminal.json', {
                    'schema': 'luna-semantic-probe-terminal-failure-v1',
                    'receipt_sha256': args.receipt_sha256 if host.HEX.fullmatch(args.receipt_sha256) else None,
                    'completed_and_clean': False, 'core_completed': False,
                    'all_semantic_checks_passed': False, 'stop_code': 'entrypoint_failure'})
        except BaseException:
            pass
        if args.preflight_only:
            print(json.dumps({'verified': False, 'model_calls': 0, 'reason': 'preflight_failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
