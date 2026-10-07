"""Sealed, finite staged-contract probe entrypoint.

The accepted host checks every source byte before this module imports the
candidate. This entry repeats that boundary before any candidate import or
paid call. It never invokes the prior semantic probe's preflight or campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True

TASK_HOME_ROOT = Path('/home/atta')
NAME = re.compile(r'\.hymem-luna-staged-probe-[A-Za-z0-9_-]{8,}\Z')
HEX = re.compile(r'[0-9a-f]{64}\Z')
EXTRACTION_IDENTITY = ('hymem-extraction-contract-sha256-v1:'
    '94a1adfc028694e9b1868ec8712baff38d4d7a99a9d2e3547c66819711890f95')
CORE_SHA = 'a389f6e8a522d878a140234ac9c040f2e3daab77bfc7cbaeadf0f4d5b12676bd'
TRANSPORT_SHA = 'c483975d53ca3523708cb589a68a8b0523664312ea259057e48ee01e7dad6ca8'
OLD_CORE_SHA = 'ddc556a8f524e8c40609de11923ee42bf4126dd4752c318c7727fc60bae59028'


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('module_load_invalid')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _early(root: Path, receipt_sha: str) -> dict:
    if (not root.is_absolute() or root.parent != TASK_HOME_ROOT or NAME.fullmatch(root.name) is None or
            type(receipt_sha) is not str or HEX.fullmatch(receipt_sha) is None):
        raise ValueError('root_or_receipt_invalid')
    for path in (TASK_HOME_ROOT, root, root / 'code', root / 'code/tools',
                 root / 'code/tools/diagnostics'):
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise ValueError('path_alias_or_symlink')
    if root.lstat().st_uid != 1000 or root.lstat().st_mode & 0o077:
        raise ValueError('root_private_invalid')
    receipt_path = root / 'launch-receipt.json'
    if (not stat.S_ISREG(receipt_path.lstat().st_mode) or
            _sha(receipt_path.read_bytes()) != receipt_sha):
        raise ValueError('receipt_pin_invalid')
    receipt = json.loads(receipt_path.read_bytes())
    pins = receipt.get('source_sha256')
    if type(pins) is not dict:
        raise ValueError('source_manifest_invalid')
    # The host may be new, but its exact bytes must be committed by the sealed
    # receipt before executing it. The accepted core and transport have fixed
    # independent pins, so a forged receipt cannot replace them.
    early = {
        'tools/diagnostics/luna_staged_host_v1.py': pins.get(
            'tools/diagnostics/luna_staged_host_v1.py'),
        'tools/diagnostics/luna_staged_run_v1.py': pins.get(
            'tools/diagnostics/luna_staged_run_v1.py'),
        'tools/diagnostics/luna_staged_core_v1.py': CORE_SHA,
        'benchmarks/codex_subscription_staged_v1.py': TRANSPORT_SHA,
    }
    for relative, expected in early.items():
        path = root / 'code' / relative
        if (type(expected) is not str or HEX.fullmatch(expected) is None or
                pins.get(relative) != expected or
                not stat.S_ISREG(path.lstat().st_mode) or
                _sha(path.read_bytes()) != expected):
            raise ValueError('early_source_drift')
    return receipt


def _reference(root: Path, receipt_sha: str):
    """Load only accepted process helpers; never its preflight or runner."""
    receipt = _early(root, receipt_sha)
    relative = 'tools/diagnostics/luna_semantic_probe_run.py'
    path = root / 'code' / relative
    expected = receipt['source_sha256'].get(relative)
    if (type(expected) is not str or not HEX.fullmatch(expected) or
            not stat.S_ISREG(path.lstat().st_mode) or _sha(path.read_bytes()) != expected):
        raise ValueError('reference_pin_invalid')
    return _module_from_file('pinned_staged_process_helpers', path)


def _paths(root: Path) -> None:
    # Python -I excludes ambient search paths. Candidate must win original
    # extraction imports, while code supplies the diagnostic and transport.
    for path in (root / 'code', root / 'candidate'):
        if str(path) in sys.path:
            sys.path.remove(str(path))
    sys.path.insert(0, str(root / 'code'))
    sys.path.insert(0, str(root / 'candidate'))


def _at(module, path: Path) -> bool:
    return Path(getattr(module, '__file__', '')).resolve() == path.resolve()


def preflight(root: Path, receipt_sha: str, postrun: bool = False):
    receipt = _early(root, receipt_sha)
    host = _module_from_file('pinned_staged_host_for_entry',
        root / 'code/tools/diagnostics/luna_staged_host_v1.py')
    if (not host.root_valid(root) or not host.HEX.fullmatch(receipt_sha) or
            not host.strict_equal(receipt, host.receipt_for(
                root, receipt['source_sha256'], receipt['binary_sha256']))):
        raise ValueError('receipt_invalid')
    host.verify_bundle(root, receipt, require_empty_workdirs=not postrun)
    if receipt['extraction_identity'] != EXTRACTION_IDENTITY:
        raise ValueError('extraction_identity_invalid')
    _paths(root)
    from tools.diagnostics import luna_staged_core_v1 as core
    from tools.diagnostics import luna_semantic_probe as old_core
    from tools.diagnostics import luna_semantic_cases as cases
    from benchmarks import extraction_canary as canary
    from benchmarks import codex_subscription_staged_v1 as transport
    from hymem.extraction import chunk
    from hymem.extraction import contract as extraction_contract
    semantic = None
    warm = transport.warm
    concurrent = warm.concurrent
    code = root / 'code'
    candidate = root / 'candidate'
    wanted = ((core, code / 'tools/diagnostics/luna_staged_core_v1.py'),
              (old_core, code / 'tools/diagnostics/luna_semantic_probe.py'),
              (cases, code / 'tools/diagnostics/luna_semantic_cases.py'),
              (canary, candidate / 'benchmarks/extraction_canary.py'),
              (chunk, candidate / 'hymem/extraction/chunk.py'),
              (extraction_contract, candidate / 'hymem/extraction/contract.py'),
              (transport, code / 'benchmarks/codex_subscription_staged_v1.py'),
              (warm, code / 'benchmarks/codex_subscription_warm_v3.py'))
    if not all(_at(module, path) for module, path in wanted):
        raise ValueError('import_path_invalid')
    if (receipt['source_sha256']['tools/diagnostics/luna_staged_core_v1.py'] != CORE_SHA or
            _sha(Path(core.__file__).read_bytes()) != CORE_SHA or
            receipt['source_sha256']['tools/diagnostics/luna_semantic_probe.py'] != OLD_CORE_SHA or
            _sha(Path(old_core.__file__).read_bytes()) != OLD_CORE_SHA or
            concurrent is not warm.concurrent or transport.warm is not warm or
            core.transport is not transport or core.staged is not transport.staged or
            core.v4 is not transport.classification or core.gate is not transport.gate or
            transport.staged.v4 is not transport.classification or
            transport.gate._staged is not transport.staged or
            concurrent.SharedBudget is not warm.SharedBudget or
            concurrent.BudgetLimits is not warm.BudgetLimits or
            not callable(getattr(warm.SharedBudget, 'record_first_failure', None))):
        raise ValueError('module_binding_invalid')
    # The source-bound transport pins the accepted source gate, staged parsers,
    # inherited warm transport and their transitive imports at import time.
    if (transport._ROOT != candidate or
            transport.classification._build_v2 is not transport.v2.build_grounding_request or
            transport.gate._source_gate is not transport.source_gate or
            extraction_contract.extraction_contract_identity() != EXTRACTION_IDENTITY or
            cases.suite_sha256() != core.CASE_SHA256):
        raise ValueError('transport_or_fixture_invalid')
    for name, (relative, digest) in transport._PINS.items():
        module = sys.modules.get(name)
        if (module is None or not _at(module, candidate / relative) or
                _sha(Path(module.__file__).read_bytes()) != digest):
            raise ValueError('candidate_contract_invalid')
    if (not _at(transport.grounding, candidate / 'hymem/extraction/grounding.py') or
            cases.GroundingSource is not transport.grounding.GroundingSource or
            not all(type(source) is transport.grounding.GroundingSource for case in cases.cases()
                    for source in case.sources)):
        raise ValueError('fixture_binding_invalid')
    retained = old_core.verify_retained_bundle(
        evidence=host.OLD_EVIDENCE.read_bytes(),
        original_receipt=host.OLD_RECEIPT.read_bytes(),
        original_result=host.OLD_RESULT.read_bytes(),
        evidence_sha256_from_new_receipt=receipt['retained_sha256']['evidence'])
    if len(retained) != 8:
        raise ValueError('retained_invalid')
    proof = {'candidate_files': 514,
             'inventory_sha256': receipt['inventory_sha256'],
             'fixture_label_sha256': core.CASE_SHA256,
             'extraction_identity': EXTRACTION_IDENTITY}
    return receipt, (host, core, old_core, cases, canary, chunk, semantic,
                     transport, warm, concurrent), proof, retained


def verify_live_containment(root: Path, receipt: dict) -> None:
    """Use the accepted containment policy with this run's staged receipt."""
    marker = root / 'launch-attempt.json'
    expected = {'receipt_sha256': _sha((root / 'launch-receipt.json').read_bytes()),
                'one_shot': True}
    if not stat.S_ISREG(marker.lstat().st_mode):
        raise ValueError('launch_marker_invalid')
    observed = json.loads(marker.read_bytes())
    if (type(observed) is not dict or set(observed) != set(expected) or
            observed['receipt_sha256'] != expected['receipt_sha256'] or
            observed['one_shot'] is not True):
        raise ValueError('launch_marker_invalid')
    props = ('ActiveState,SubState,MainPID,ControlGroup,NRestarts,MemoryMax,TasksMax,'
             'CPUQuotaPerSecUSec,KillMode,Restart,RemainAfterExit,OOMPolicy,'
             'RuntimeMaxUSec,TimeoutStopUSec')
    shown = subprocess.run(['/usr/bin/systemctl', '--user', 'show', receipt['unit'],
        '--property=' + props, '--no-pager'], capture_output=True, text=True,
        check=True, timeout=10)
    rows = dict(line.split('=', 1) for line in shown.stdout.splitlines() if '=' in line)
    if (set(rows) != set(props.split(',')) or rows['ActiveState'] != 'active' or
            rows['SubState'] != 'running' or rows['MainPID'] != str(os.getpid()) or
            rows['ControlGroup'] != receipt['expected_cgroup'] or
            rows['NRestarts'] != '0' or rows['MemoryMax'] != '4294967296' or
            rows['TasksMax'] != '256' or rows['KillMode'] != 'control-group' or
            rows['Restart'] != 'no' or rows['RemainAfterExit'] != 'yes' or
            rows['OOMPolicy'] != 'kill' or
            rows['CPUQuotaPerSecUSec'] not in {'2s', '2.000s'} or
            rows['RuntimeMaxUSec'] not in {'32min 10s', '32min 10.000s', '1930s'} or
            rows['TimeoutStopUSec'] not in {'10s', '10.000s'}):
        raise ValueError('service_policy_invalid')
    cgroup = receipt['expected_cgroup']
    if f'0::{cgroup}' not in Path('/proc/self/cgroup').read_text().splitlines():
        raise ValueError('cgroup_identity_invalid')
    path = Path('/sys/fs/cgroup' + cgroup)
    if ((path / 'memory.max').read_text().strip() != '4294967296' or
            (path / 'pids.max').read_text().strip() != '256'):
        raise ValueError('effective_resource_policy_invalid')
    cpu = (path / 'cpu.max').read_text().split()
    if len(cpu) != 2 or not all(x.isdecimal() for x in cpu) or int(cpu[0]) != 2 * int(cpu[1]):
        raise ValueError('effective_cpu_policy_invalid')


_STOPS = frozenset((None, 'private_evidence_write_failure', 'preunit_accounting_invalid',
    'postunit_accounting_invalid', 'cleanup_failure', 'infrastructure_or_runtime_failure',
    'transport_or_budget_stop', 'canary_batch_binding_invalid', 'schedule_invalid',
    'stage_limit', 'correction_integrity_failure'))


def _safe_result(result: dict, receipt_sha: str, source_sha: str,
                 process_sha: str, absent: bool, elapsed: float) -> dict:
    from tools.diagnostics import luna_staged_core_v1 as core
    core.validate_public_result(result)
    budget = result['paid_budget']
    return {'schema': 'luna-staged-probe-terminal-v1',
            'receipt_sha256': receipt_sha, 'source_sha256': source_sha,
            'process_identity_sha256': process_sha,
            'process_groups_absent_at_entry_exit': absent,
            'diagnostic_completed': result['diagnostic_completed'],
            'completed_and_clean': False, 'semantic_accuracy_accepted': False,
            'full_lme_ready': False, 'completed_units': result['completed_units'],
            'attempted_units': result['attempted_units'],
            'malformed_units': result['malformed_units'],
            'expected_gold_matches': sum(x['expected_gold_match'] for x in result['units']),
            'paid_budget': {key: budget[key] for key in
                ('turns', 'known_tokens', 'usage_complete', 'in_flight', 'reserved')},
            'client_cleanup_ok': result['client_cleanup_ok'],
            'stop_code': result['stop_code'] if result['stop_code'] in _STOPS else 'finite_other',
            'first_failure': result['first_failure'],
            'elapsed_seconds': round(elapsed, 3)}


def execute(root: Path, receipt_sha: str, *, preflight_only: bool = False,
            client_factory=None, containment=None) -> dict:
    started = time.monotonic()
    receipt, loaded, proof, retained = preflight(root, receipt_sha)
    if preflight_only:
        return {'schema': 'luna-staged-probe-preflight-v1', 'verified': True,
                'receipt_sha256': receipt_sha, 'candidate_files': proof['candidate_files'],
                'retained_ordinary_completions': len(retained), 'model_calls': 0}
    host, core, old_core, cases, canary, chunk, semantic, transport, warm, concurrent = loaded
    (containment or verify_live_containment)(root, receipt)
    run = root / 'run'
    run.mkdir(mode=0o700)
    journal_path = run / 'private-journal'
    journal_path.mkdir(mode=0o700)
    progress_path = run / 'safe-progress'
    progress_path.mkdir(mode=0o700)
    owned_path = run / 'private-owned-processes'
    owned_path.mkdir(mode=0o700)
    identities = []
    reference = _reference(root, receipt_sha)

    # A single client may create several warm sessions as it rotates. Each
    # process is recorded before its first turn, keyed to its scheduled unit.
    def factory(key, cap, budget):
        class TrackingSession(warm.WarmSession):
            def __init__(self, *args, **kwargs):
                try:
                    super().__init__(*args, **kwargs)
                finally:
                    process = getattr(self, 'process', None)
                    if process is not None:
                        try:
                            identity = reference._process_identity(process.pid)
                            identity['index'] = len(identities)
                            identity['unit_key'] = key
                            host.write_once(owned_path / f'{len(identities):04d}.json', identity)
                            identities.append(identity)
                        except BaseException:
                            try:
                                host.write_once(owned_path / 'tracking-failure.json',
                                                {'tracking_failed': True, 'unit_key': key})
                            finally:
                                self.close()
                            raise

        return transport.StagedSubscriptionClient(str(host.BINARY), budget, key,
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
                    'schema': 'luna-staged-probe-unit-progress-v1',
                    'finished_units': self.finished + 1,
                    'paid_turns': state['turns'],
                    'known_tokens': state['known_tokens'],
                    'usage_complete': state['usage_complete'],
                    'stop_present': state['stopped']})
                self.finished += 1

    journal = ProgressJournal(journal_path)
    batches = core.derive_canary_batches(canary_module=canary, chunk_module=chunk,
        semantic_probe_module=old_core, candidate=root / 'candidate', retained=retained,
        record=lambda event: journal.record('derivation', event))
    result = core.run_campaign(concurrent=concurrent, warm=warm,
        binary=str(host.BINARY), cases_module=cases, canary_batches=batches,
        journal=journal, client_factory=client_factory or factory)
    core.validate_public_result(result)
    private_written = True
    try:
        host.write_once(run / 'private-result.json', result)
    except BaseException:
        private_written = False
    identity_sha = _sha(json.dumps(identities, sort_keys=True,
        separators=(',', ':')).encode())
    absent = reference._owned_absent(identities)
    source_sha = _sha(json.dumps(receipt['source_sha256'], sort_keys=True,
        separators=(',', ':')).encode())
    safe = _safe_result(result, receipt_sha, source_sha, identity_sha, absent,
                        time.monotonic() - started)
    if not private_written:
        safe['diagnostic_completed'] = False
        safe['stop_code'] = 'private_result_write_failure'
    host.write_once(root / 'safe-terminal.json', safe)
    return safe


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = execute(args.root, args.receipt_sha256,
                         preflight_only=args.preflight_only)
        if args.preflight_only:
            print(json.dumps(result, sort_keys=True))
        return 0 if args.preflight_only or result['diagnostic_completed'] else 1
    except BaseException:
        try:
            # Failure after creating a run directory has unknown usage. The
            # terminal intentionally contains no fabricated zero counters.
            receipt, loaded, _, _ = preflight(args.root, args.receipt_sha256,
                                               postrun=True)
            host = loaded[0]
            if (args.root / 'run').is_dir() and not (args.root / 'safe-terminal.json').exists():
                host.write_once(args.root / 'safe-terminal.json', {
                    'schema': 'luna-staged-probe-terminal-failure-v1',
                    'receipt_sha256': args.receipt_sha256,
                    'diagnostic_completed': False, 'completed_and_clean': False,
                    'semantic_accuracy_accepted': False, 'full_lme_ready': False,
                    'paid_budget': None, 'stop_code': 'entrypoint_failure'})
        except BaseException:
            pass
        if args.preflight_only:
            print(json.dumps({'verified': False, 'model_calls': 0,
                              'reason': 'preflight_failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
