"""One existing Luna extraction canary; no LME question or production store."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

CANDIDATE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate')
BINARY = Path('/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex')
INVENTORY = Path('/home/atta/.hymem-luna-lme-v2-yeds3h_l/headless-source-map.json')
DATASET = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner-sha256', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / 'luna_subscription_lme_profiled.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == args.runner_sha256
    spec = importlib.util.spec_from_file_location('root_profiled_canary', source)
    profiled = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = profiled
    spec.loader.exec_module(profiled)
    runner = profiled.warm_runner
    parts = runner.load_verified(candidate=CANDIDATE, inventory_stamp=INVENTORY,
        inventory_sha256='852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb',
        dataset=DATASET, dataset_sha256='d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442',
        binary=BINARY, base_path=root / 'codex_subscription.py',
        concurrent_path=root / 'codex_subscription_concurrent_v2.py',
        warm_path=root / 'codex_subscription_warm.py')
    files, concurrent, request_type, canary, chunk, lme, protocol, warm = parts
    sessions = []
    class Tracked(warm.WarmSession):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            sessions.append(self)
    budget = warm.SharedBudget(warm.BudgetLimits(12, 160000, 600), max_in_flight=1)
    delegate = warm.WarmSubscriptionClient(str(BINARY), budget, 'canary',
        warm.BudgetLimits(12, 160000, 590), session_factory=Tracked)
    ledger = profiled.collector.StageLedger(CANDIDATE)
    client = ledger.wrap(delegate, 'canary')
    result, failure, cleanup = None, None, True
    started = time.monotonic()
    try:
        with (root / 'private-canary.log').open('x') as log:
            os.chmod(root / 'private-canary.log', 0o600)
            with redirect_stdout(log), redirect_stderr(log):
                result = runner.old.experimental_canary(canary, chunk, client,
                    evidence=lambda value: runner.atomic_private(root / 'private-canary-evidence.json', value))
    except BaseException:
        failure = 'canary_or_transport_failure'
    finally:
        try:
            client.close()
        except BaseException:
            cleanup = False
        for session in sessions:
            try:
                session.close()
                os.killpg(session.process.pid, 0)
                cleanup = False
            except ProcessLookupError:
                pass
            except BaseException:
                cleanup = False
    state = budget.snapshot()
    reconciled = ledger.reconcile(state)
    ok = (failure is None and result is not None and result.get('passed') is True
          and cleanup and reconciled and state['usage_complete'] and not state['stopped']
          and state['in_flight'] == state['reserved'] == 0)
    report = {'ok': ok, 'failure': failure, 'canary': result, 'budget': state,
        'stages': ledger.snapshot(), 'stage_accounting_reconciled': reconciled,
        'transport': runner.warm_metrics(delegate), 'source_files': files,
        'cleanup': cleanup, 'process_count': len(sessions),
        'wall_seconds': time.monotonic() - started, 'lme_questions_run': 0,
        'internal_http_attempts': None}
    runner.atomic_private(root / 'safe-canary-result.json', report)
    print(json.dumps(report, sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print(json.dumps({'ok': False, 'code': 'probe_setup_failed'}))
        raise SystemExit(1)
