"""One approved production dream, supervised on Hermes1, no automatic retry.

Only metadata is emitted. The worker inherits the effective HyMem environment
from the current Honcho process; credentials never enter a receipt or argv.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import sys

SOURCE = Path('/home/node/HyMem')
STAGE = Path('/home/node/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/deploy-v1')
SELF = STAGE / 'production-dream.py'
PHASE1_SHA = '31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136'
SUPERVISOR_SHA = '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'
GENERATION = 'hymem-phase1-generation-v1:6075085e12e32e1b790e49b99b8c3bb50718b18be0f28762d1582c58ee8e35eb'
TARGET = 'chk_12238afe485d5b2f4e7975b0d20b096ec20414d9'
sys.path.insert(0, str(SOURCE))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > 65536:
        raise RuntimeError('metadata_receipt_too_large')
    fd = os.open(STAGE / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def verify():
    if (sha(SOURCE / 'hymem/dreaming/phase1.py') != PHASE1_SHA
            or sha(STAGE / 'supervised_invocation.py') != SUPERVISOR_SHA
            or Path(__file__).resolve() != SELF
            or not STAGE.is_dir()):
        raise RuntimeError('production_dream_input_drift')
    installed = json.loads((STAGE / 'dream-worker-installed.json').read_bytes())
    if installed['sha256'] != sha(SELF):
        raise RuntimeError('dream_worker_pin_drift')


def runtime_env():
    pids = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'hymem.honcho' in args or any(a.rsplit(b'/', 1)[-1] == b'hymem-honcho' for a in args):
            pids.append(proc)
    if len(pids) != 1:
        raise RuntimeError('honcho_process_count')
    values = {}
    for item in (pids[0] / 'environ').read_bytes().split(b'\0'):
        if b'=' in item:
            key, value = item.split(b'=', 1)
            if key.startswith(b'HYMEM_'):
                values[os.fsdecode(key)] = os.fsdecode(value)
    if (values.get('HYMEM_LLM_MODEL') != 'deepseek-flash'
            or values.get('HYMEM_LLM_BASE_URL', '').rstrip('/') != 'https://api.deepseek.com'
            or not values.get('HYMEM_LLM_API_KEY')
            or values.get('HYMEM_ROOT') != '/home/node/.hermes'):
        raise RuntimeError('production_runtime_identity_changed')
    return {'PATH': '/home/node/hymem-env/bin:/usr/bin:/bin', 'HOME': '/home/node',
            'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1', **values}


def snapshot(hy):
    status = hy.dream_status()
    fields = ('in_progress', 'pending_chunks', 'pending_digests', 'pending_profiles',
              'pending_facts', 'pending_aggregation', 'quarantined_chunks',
              'quarantined_digests', 'quarantined_profiles', 'quarantined_facts',
              'coverage_integrity_failures', 'summary_degraded_sessions',
              'summary_missing_sessions', 'summary_healthy')
    return {key: status.get(key) for key in fields}


def worker():
    logging.disable(logging.CRITICAL)
    # Wait for the parent's durable ownership receipt before opening the store.
    if sys.stdin.buffer.read() != b'approved-production-dream-v1':
        raise RuntimeError('production_dream_authorization_missing')
    verify()
    from hymem.bootstrap import build_from_env, shutdown_instance
    from hymem.deadline import MonotonicDeadline
    from benchmarks.strictness import usage_snapshot
    result = {'status': 'failed', 'cleanup_ok': False}
    hy = None
    try:
        hy = build_from_env()
        if hy._phase1_generation['generation_key'] != GENERATION:
            raise RuntimeError('production_generation_changed')
        result['before'] = snapshot(hy)
        if result['before']['in_progress']:
            raise RuntimeError('production_dream_already_active')
        result['previous_run_id'] = hy.conn.execute('SELECT MAX(id) FROM dream_runs').fetchone()[0]
        report = hy.dream(deadline=MonotonicDeadline.after(1800))
        result['report'] = {key: value for key, value in asdict(report).items()
                            if value is None or type(value) in (int, float, bool)}
        result['after'] = snapshot(hy)
        runs = hy.conn.execute(
            'SELECT id,started_at,ended_at,error FROM dream_runs '
            'WHERE id>? AND skipped_locked=0 ORDER BY id',
            (result['previous_run_id'],)).fetchall()
        if report.skipped_locked or len(runs) != 1:
            raise RuntimeError('production_dream_run_ownership_ambiguous')
        last = runs[0]
        result['run'] = {'id': last['id'], 'started_at': last['started_at'],
                         'ended_at': last['ended_at'], 'has_error': last['error'] is not None}
        result['target_current_publications'] = hy.conn.execute(
            'SELECT COUNT(*) FROM current_phase1_publications WHERE chunk_id=? AND phase1_generation_key=?',
            (TARGET, GENERATION)).fetchone()[0]
        result['status'] = ('completed' if not report.skipped_locked and last['ended_at']
                            and last['error'] is None and last['id'] > result['previous_run_id']
                            and result['target_current_publications'] == 1 else 'unverified')
    except BaseException as exc:
        result['error_type'] = type(exc).__name__
    finally:
        if hy is not None:
            try:
                result['usage'] = usage_snapshot(hy._llm)
            except BaseException as exc:
                result['usage_error_type'] = type(exc).__name__
                result['status'] = 'failed'
            finally:
                try:
                    result['cleanup_ok'] = shutdown_instance(hy)
                except BaseException as exc:
                    result['cleanup_error_type'] = type(exc).__name__
        if not result['cleanup_ok']:
            result['status'] = 'failed'
        save('dream-result.json', result)
    return 0 if result['status'] == 'completed' else 1


def supervise():
    verify()
    spec = importlib.util.spec_from_file_location('claim_dream_supervisor', STAGE / 'supervised_invocation.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    env = runtime_env()
    def cancelled(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, cancelled)
    outcome = module.supervise_invocation(
        [sys.executable, '-I', '-B', str(SELF), 'worker'], cwd=SOURCE, env=env,
        output_dir=STAGE / 'dream-invocation', timeout_seconds=1860,
        cleanup_seconds=10, stdin_bytes=b'approved-production-dream-v1',
        output_limit_bytes=2 * 1024 * 1024,
    )
    save('dream-supervisor.json', asdict(outcome))
    return 0 if outcome.status == 'completed' and outcome.safe_to_continue else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('supervise', 'worker'))
    args = parser.parse_args()
    try:
        rc = supervise() if args.mode == 'supervise' else worker()
    except BaseException as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        rc = 1
    raise SystemExit(rc)
