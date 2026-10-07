"""Sealed claim-task entry. The accepted v3 entry supplies read-only preflight and containment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def _reference(root: Path, receipt_sha: str):
    path = root / 'launch-receipt.json'
    if (not path.is_file() or path.is_symlink() or
            hashlib.sha256(path.read_bytes()).hexdigest() != receipt_sha):
        raise ValueError('receipt_pin_invalid')
    receipt = json.loads(path.read_bytes())
    source = root / 'code/tools/diagnostics/luna_semantic_probe_run.py'
    expected = receipt['source_sha256']['tools/diagnostics/luna_semantic_probe_run.py']
    if (not source.is_file() or source.is_symlink() or
            hashlib.sha256(source.read_bytes()).hexdigest() != expected):
        raise ValueError('reference_pin_invalid')
    spec = importlib.util.spec_from_file_location('pinned_claim_task_reference_entry', source)
    if spec is None or spec.loader is None:
        raise ValueError('reference_load_invalid')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(root: Path, receipt_sha: str, postrun: bool = False):
    reference = _reference(root, receipt_sha)
    receipt, loaded, proof, retained = reference.preflight(root, receipt_sha,
        require_empty_workdirs=not postrun)
    from tools.diagnostics import luna_claim_task_core_v1 as core
    from benchmarks import codex_subscription_claim_task_v1 as transport
    host, old_core, cases, grounding, canary, chunk, semantic, stage, warm, concurrent = loaded
    if (core.transport is not transport or
            transport.warm.concurrent is None or
            transport.contract is not core.contract):
        raise ValueError('claim_transport_binding_invalid')
    core.validate_public_result  # Pin import and fail before inference if unavailable.
    return receipt, (host, core, old_core, cases, canary, chunk, semantic,
                     transport, transport.warm, transport.warm.concurrent), proof, retained


def _safe_result(result: dict, receipt_sha: str, source_sha: str,
                 process_sha: str, absent: bool, elapsed: float) -> dict:
    from tools.diagnostics import luna_claim_task_core_v1 as core
    from benchmarks import codex_subscription_claim_task_v1 as transport
    core.validate_public_result(result)
    return {'schema': 'luna-claim-task-probe-terminal-v1',
            'receipt_sha256': receipt_sha, 'source_sha256': source_sha,
            'process_identity_sha256': process_sha,
            'process_groups_absent_at_entry_exit': absent,
            'diagnostic_completed': result['diagnostic_completed'],
            'completed_and_clean': False,
            'semantic_accuracy_accepted': False, 'full_lme_ready': False,
            'completed_units': result['completed_units'],
            'attempted_units': result['attempted_units'],
            'malformed_units': result['malformed_units'],
            'paid_budget': result['paid_budget'],
            'client_cleanup_ok': result['client_cleanup_ok'],
            'stop_code': result['stop_code'],
            'first_failure': transport.warm.serialize_failure(result['first_failure']),
            'elapsed_seconds': round(elapsed, 3)}


def execute(root: Path, receipt_sha: str, *, preflight_only: bool = False,
            client_factory=None, containment=None) -> dict:
    started = time.monotonic()
    receipt, loaded, proof, retained = preflight(root, receipt_sha)
    if preflight_only:
        return {'schema': 'luna-claim-task-probe-preflight-v1', 'verified': True,
                'receipt_sha256': receipt_sha, 'candidate_files': proof['candidate_files'],
                'retained_ordinary_completions': len(retained), 'model_calls': 0}
    host, core, old_core, cases, canary, chunk, semantic, transport, warm, concurrent = loaded
    reference = _reference(root, receipt_sha)
    (containment or reference.verify_live_containment)(root, receipt)
    run = root / 'run'
    run.mkdir(mode=0o700)
    journal_path = run / 'private-journal'
    journal_path.mkdir(mode=0o700)
    progress_path = run / 'safe-progress'
    progress_path.mkdir(mode=0o700)
    owned_path = run / 'private-owned-processes'
    owned_path.mkdir(mode=0o700)
    identities = []

    class TrackingSession(warm.WarmSession):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            finally:
                process = getattr(self, 'process', None)
                if process is not None:
                    try:
                        identity = reference._process_identity(process.pid)
                    except BaseException:
                        try:
                            host.write_once(owned_path / 'tracking-failure.json', {'tracking_failed': True})
                        finally:
                            self.close()
                        raise
                    identity['index'] = len(identities)
                    try:
                        host.write_once(owned_path / f'{len(identities):04d}.json', identity)
                    except BaseException:
                        self.close()
                        raise
                    identities.append(identity)

    def factory(key, cap, budget):
        return transport.ClaimTaskSubscriptionClient(str(host.BINARY), budget, key,
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
                    'schema': 'luna-claim-task-probe-unit-progress-v1',
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
    identity_sha = hashlib.sha256(json.dumps(identities, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    absent = reference._owned_absent(identities)
    source_sha = hashlib.sha256(json.dumps(receipt['source_sha256'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    safe = _safe_result(result, receipt_sha, source_sha, identity_sha, absent,
                        time.monotonic() - started)
    if not private_written:
        safe['diagnostic_completed'] = False
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
        return 0 if args.preflight_only or out['diagnostic_completed'] else 1
    except BaseException:
        try:
            from tools.diagnostics import luna_semantic_probe_host as host
            if host.root_valid(root) and (root / 'run').is_dir() and not (root / 'safe-terminal.json').exists():
                host.write_once(root / 'safe-terminal.json', {
                    'schema': 'luna-claim-task-probe-terminal-failure-v1',
                    'receipt_sha256': args.receipt_sha256 if host.HEX.fullmatch(args.receipt_sha256) else None,
                    'completed_and_clean': False, 'diagnostic_completed': False,
                    'semantic_accuracy_accepted': False, 'full_lme_ready': False,
                    'stop_code': 'entrypoint_failure'})
        except BaseException:
            pass
        if args.preflight_only:
            print(json.dumps({'verified': False, 'model_calls': 0, 'reason': 'preflight_failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
