"""One-shot, read-only closed metadata poller. Run remotely through SSH stdin.

No scheduling, provider calls, retries, process mutation, or raw log output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
import types

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky')
ROOT = BASE / 'lme-r9-full500-headless-v1'
CONT = BASE / 'lme-r9-full500-continuation-v1'
FINAL = BASE / 'lme-r9-full500-finalizer-v1'
SOURCE = BASE / 'lme-r9-full-suite-v1/candidate'
USAGE_KEYS = ('calls', 'calls_available', 'request_attempts', 'request_attempts_available',
              'successful_responses', 'successful_responses_available', 'prompt_tokens',
              'completion_tokens', 'total_tokens', 'token_usage_available', 'cost_usd', 'cost_available')
CANARY_STATUSES = frozenset(('pending', 'passed', 'failed', 'not_run_no_pending', 'skipped_non_comparable'))
CANARY_FAILURES = frozenset(('branch_incomplete', 'call_failure', 'clean_empty', 'contract_failure',
    'incomplete_response', 'input_contract_failure', 'internal_validation_failure', 'item_validation_failure',
    'output_limit_exceeded', 'parse_failure', 'resource_limit', 'response_conflict', 'shape_failure',
    'source_coverage_failure', 'supported_claim_evidence_missing', 'supported_claim_missing',
    'unexpected_canary_output', 'unspecified_failure'))

def require(condition):
    if not condition:
        raise ValueError('metadata_integrity_check_failed')

def read(path):
    require(path.resolve() == path and path.is_file() and not path.is_symlink())
    require(path.stat().st_size <= 256 * 1024 * 1024)
    return path.read_bytes()

def js(path):
    return json.loads(read(path))

def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0)
    return value

def usage(meter):
    require(type(meter) is dict)
    out = {}
    for key in USAGE_KEYS:
        value = meter.get(key)
        require(value is None or type(value) is bool or type(value) in (int, float))
        if type(value) in (int, float):
            number(value)
        out[key] = value
    return out

def terminal_projection(data, manifest, pin):
    require(data['manifest_sha256'] == pin and data['question_ids'] == manifest['question_ids'])
    require(data['status'] in ('scored_full500_completed', 'validated_full500_has_failures'))
    out = {'status': data['status'], 'counts': {}}
    for key in ('expected', 'attempted', 'completed', 'failed', 'missing', 'unique_attempted', 'total_attempts'):
        require(type(data['counts'][key]) is int)
        out['counts'][key] = number(data['counts'][key])
    require(out['counts']['expected'] == 500)
    for key in ('strict_scored_artifact_validated', 'physical_checkpoint_bound',
                'process_completed_cleanly', 'benchmark_completed_without_faults',
                'reader_judge_calls_measured', 'canary_accounted_separately'):
        require(type(data[key]) is bool)
        out[key] = data[key]
    out['elapsed_s'] = number(data['elapsed_s'])
    out['aggregate_paid_usage'] = usage(data['aggregate_paid_usage'])
    out['per_role_usage'] = {role: usage(data['per_role_usage'][role])
                             for role in ('reader', 'judge', 'memory_pipeline', 'retrieval', 'canary')
                             if role in data['per_role_usage']}
    rows = data['per_question']
    require(type(rows) is list and len(rows) == 500
            and [r['question_id'] for r in rows] == manifest['question_ids'])
    require(all(type(row['answer_correct']) is bool for row in rows))
    out['correct'] = sum(row['answer_correct'] for row in rows)
    return out

def continuation_projection(result):
    require(type(result) is dict and result['status'] in ('operator_pending', 'offline_validation_finished'))
    out = {'status': result['status']}
    for key in ('paid_runs_started', 'validation_runs_started', 'live_exit_code', 'validation_exit_code'):
        value = result.get(key)
        require(value is None or type(value) is int and 0 <= value <= (1 if 'runs_started' in key else 255))
        out[key] = value
    value = result.get('operator_inspection_required', False)
    require(type(value) is bool)
    out['operator_inspection_required'] = value
    return out

def process_projection(owned, proc_stat):
    pid = owned['pid']
    require(type(pid) is int and pid > 0 and pid == owned['process_group_id'] == owned['session_id'])
    out = {'process_present': proc_stat is not None}
    if proc_stat is not None:
        fields = proc_stat.rpartition(')')[2].split()
        require(int(fields[19]) == owned['start_ticks'] and int(fields[2]) == int(fields[3]) == pid)
        out.update(identity_verified=True, process_live=fields[0] not in ('Z', 'X'))
    return out

def project(manifest, checkpoint):
    """Pure projection; only pinned IDs, booleans, counters and usage leave here."""
    ids = manifest['question_ids']
    require(manifest.get('schema') == 'stock-lme-r9-full500-candidate-regression-v1')
    require(manifest.get('question_count') == 500 and type(manifest.get('question_count')) is int
            and manifest.get('sample') == 0 and type(manifest.get('sample')) is int
            and manifest.get('source_indices') == list(range(500)))
    require(type(ids) is list and len(ids) == len(set(ids)) == 500)
    require(all(type(q) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', q) for q in ids))
    out = {'expected_questions': 500, 'inflight_usage': None,
           'usage_scope': 'checkpoint_snapshot', 'checkpoint_present': checkpoint is not None}
    if checkpoint is None:
        return out
    require(checkpoint['expected_ids'] == ids and checkpoint['scored'] is True
            and checkpoint['verdict_key'] == 'correct'
            and checkpoint['status'] in ('running', 'complete'))
    entries = checkpoint['entries']
    require(type(entries) is dict and set(entries) <= set(ids))
    counts = dict(completed=0, failed=0, correct=0, healthy_indexing=0,
                  summary_missing_sessions=0, summary_degraded_sessions=0, malformed_summaries=0)
    summary_metadata_questions = 0
    for qid, entry in entries.items():
        require(type(entry) is dict and entry['status'] in ('completed', 'failed'))
        row = entry['row']
        require(type(row) is dict and row['question_id'] == qid
                and (row.get('correct') is None or type(row['correct']) is bool))
        counts[entry['status']] += 1
        counts['correct'] += row.get('correct') is True
        indexing = row.get('indexing') or {}
        counts['healthy_indexing'] += indexing.get('complete') is True and indexing.get('healthy') is True
        summary = (indexing.get('final_status') or {}).get('summary_health') or {}
        summary_metadata_questions += all(type(summary.get(key)) is int for key in
            ('summary_missing_sessions', 'summary_degraded_sessions', 'malformed_summaries'))
        for key in ('summary_missing_sessions', 'summary_degraded_sessions', 'malformed_summaries'):
            value = summary.get(key)
            if value is not None:
                require(type(value) is int)
                counts[key] += number(value)
    require(checkpoint['status'] != 'complete' or len(entries) == 500)
    out.update(checkpoint_status=checkpoint['status'], counts=counts,
               uncheckpointed_questions=500-len(entries),
               summary_metadata_questions=summary_metadata_questions)
    segments = checkpoint.get('execution_segments', [])
    require(type(segments) is list)
    if segments:
        require(len(segments) == 1 and type(segments[0]) is dict)
        segment = segments[0]
        out['elapsed_s'] = number(segment['elapsed_s'])
        out['per_role_usage'] = {role: usage(segment[role + '_usage'])
                                 for role in ('reader', 'judge', 'memory_pipeline', 'retrieval')
                                 if role + '_usage' in segment}
        canary = segment.get('extraction_canary')
        if canary is not None:
            require(type(canary) is dict and canary.get('status') in CANARY_STATUSES)
            out['canary_status'] = canary['status']
            if canary['status'] == 'failed':
                require(canary.get('failure_reason') in CANARY_FAILURES)
                out['canary_failure_reason'] = canary['failure_reason']
            if type(canary.get('usage')) is dict:
                out['canary_usage'] = usage(canary['usage'])
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--controller-sha256', required=True)
    args = parser.parse_args()
    require(all(re.fullmatch('[0-9a-f]{64}', pin) for pin in
                (args.manifest_sha256, args.controller_sha256)))
    raw = read(ROOT / 'host_control.py')
    require(hashlib.sha256(raw).hexdigest() == args.controller_sha256)
    controller = types.ModuleType('verified_progress_controller')
    controller.__file__ = str(ROOT / 'host_control.py')
    exec(compile(raw, controller.__file__, 'exec'), controller.__dict__)
    controller.ROOT, controller.SOURCE, controller.PIN = ROOT, SOURCE, args.manifest_sha256
    manifest, host = controller.definitions()
    checkpoint_path = ROOT / 'live-results/benchmark/checkpoint.json'
    out = project(manifest, js(checkpoint_path) if checkpoint_path.exists() else None)
    out.update(source_and_runtime_verified=True, raw_content_exported=False,
               manifest_sha256=args.manifest_sha256,
               disk_free_bytes=shutil.disk_usage(ROOT).free)
    if checkpoint_path.exists():
        out['checkpoint_age_s'] = max(0, time.time()-checkpoint_path.stat().st_mtime)
    for mode in ('live', 'validation'):
        receipt = ROOT / (mode + '-container-id.json')
        if receipt.exists():
            cid = js(receipt)['container_id']
            plan = (host.command(str(ROOT), str(SOURCE), args.manifest_sha256, live=True)
                    if mode == 'live' else host.validation_command(str(ROOT), str(SOURCE), args.manifest_sha256))
            state = controller.inspect(cid, plan)
            out[mode] = {k: state[k] for k in ('status', 'exit_code', 'oom_killed', 'pid', 'configuration_verified')}
            if (mode == 'validation' and state['configuration_verified'] is True
                    and state['status'] == 'exited' and state['exit_code'] == state['pid'] == 0
                    and state['oom_killed'] is False):
                reply = subprocess.run(['docker', 'logs', cid], capture_output=True, timeout=30)
                require(reply.returncode == 0 and len(reply.stdout) <= 32 * 1024 * 1024)
                out['terminal_validation'] = terminal_projection(json.loads(reply.stdout), manifest, args.manifest_sha256)
    for label, metadata_root in (('continuation', CONT), ('finalizer', FINAL)):
        result_path = metadata_root / 'result.json'
        if result_path.exists():
            out[label + '_result'] = continuation_projection(js(result_path))
        owned_path = metadata_root / 'launch-owned.json'
        if owned_path.exists():
            owned = js(owned_path)
            require(type(owned['pid']) is int and owned['pid'] > 0)
            proc = Path('/proc') / str(owned['pid']) / 'stat'
            for key, value in process_projection(owned, proc.read_text() if proc.exists() else None).items():
                out[label + '_' + key] = value
    out['private_invocation_logs'] = {}
    for mode in ('live', 'validation'):
        for name in ('stdout.bin', 'stderr.bin'):
            path = ROOT / (mode + '-results') / 'invocation' / name
            if path.exists():
                require(path.resolve() == path and path.is_file() and not path.is_symlink())
                stat = path.stat()
                out['private_invocation_logs'][mode + '_' + name] = {'bytes': stat.st_size, 'mtime': stat.st_mtime}
    print(json.dumps(out, sort_keys=True, allow_nan=False))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'status': 'metadata_verification_failed', 'raw_content_exported': False}))
        raise SystemExit(1)
