"""Verifier-only v2: read-only postvalidation of the SAME sealed stock Q1 run.

Run in a separate network=none container with results/source/package mounted
read-only and no credentials. Only the bounded metadata report leaves stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types


def load_run_helper(manifest_sha):
    # This verifier is mounted separately at /audit; the paid run's original
    # sealed package at /diag remains byte-identical and fully validated.
    package = Path('/diag')
    manifest_path = package / 'manifest.json'
    require(manifest_path.resolve() == manifest_path
            and stat.S_ISREG(manifest_path.lstat().st_mode), 'postvalidation_manifest_path')
    raw_manifest = manifest_path.read_bytes()
    require(hashlib.sha256(raw_manifest).hexdigest() == manifest_sha, 'postvalidation_manifest_pin')
    manifest = json.loads(raw_manifest)
    path = package / 'q1_stock_run.py'
    require(path.resolve() == path and stat.S_ISREG(path.lstat().st_mode), 'postvalidation_helper_path')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest['helper_sha256']['q1_stock_run.py'],
            'postvalidation_helper_pin')
    module = types.ModuleType('q1_stock_run')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def read_json(path):
    if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('postvalidation_non_regular_path')
    return json.loads(path.read_text())


def require(value, code):
    if not value:
        raise RuntimeError(code)


def check_recipe(data, run):
    config = data['config']
    expected = {
        'sample': 1, 'seed': run.SEED, 'workers': 1, 'top_k': 15,
        'auto_ability': True, 'permissive_default': True, 'no_dream': False,
        'embeddings': False, 'aggregation_nodes': False, 'episode_granularity': False,
        'retrieval_only': False, 'distill': False, 'indexing_require_healthy': True,
        'source_order_validated': True, 'dataset_expected_count': 500,
        'scored_run': True, 'label_free_answer_path': True, 'prereg': None,
        'indexing_max_cycles': 100, 'indexing_timeout_s': 3600.0,
        'judge_protocol': 'legacy-custom', 'hymem_thinking': 'disabled',
    }
    for field, expected_value in expected.items():
        value = config.get(field)
        require(value == expected_value and (
            type(value) is type(expected_value)
            or type(expected_value) is float and type(value) is int), 'postvalidation_recipe')
    require(config.get('scales', config.get('scale')) == 'S'
            and data['manifest']['protocol_split'] == 'full'
            and data['manifest']['expected_count'] == 1
            and data['manifest']['data_hash'] == 'sha256:' + run.DATASET_SHA,
            'postvalidation_dataset_recipe')
    disabled = {'thinking': {'type': 'disabled'}}
    for role in ('reader', 'judge', 'memory_pipeline'):
        identity = data['models'][role]
        require(identity['model'] == run.MODEL
                and identity['endpoint_origin'] == run.ENDPOINT
                and identity['endpoint_sha256'] == 'sha256:' + hashlib.sha256(
                    run.ENDPOINT.encode()).hexdigest(), 'postvalidation_model')
        field = 'effective_extra_body' if role == 'memory_pipeline' else 'extra_body'
        require(identity[field] == disabled, 'postvalidation_thinking')



def canonical(value):
    """Compare JSON evidence without Python's bool/int/float equality coercion."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_physical_checkpoint(snapshot, rows, *, question_id, reconcile_results,
                                 bounded_failure_text):
    """Bind physical fields through the stock checkpoint's explicit projection.

    AtomicCheckpoint.record persists the raw row; reconcile_results adds
    strict_failure for the archive. Failed entries additionally project their
    recorded failure and a False verdict, just as AtomicCheckpoint.reconcile
    does; they can never produce a clean-completion report. No other field is
    removed or normalized, including IDs and indexing content.
    This fresh, one-attempt Q1 gate also binds the otherwise unarchived history.
    """
    require(snapshot['expected_ids'] == [question_id]
            and snapshot['status'] == 'complete'
            and snapshot['scored'] is True
            and snapshot['verdict_key'] == 'correct'
            and type(snapshot['entries']) is dict
            and set(snapshot['entries']) == {question_id},
            'postvalidation_physical_checkpoint')
    entry = snapshot['entries'][question_id]
    require(type(entry) is dict and entry.get('status') in ('completed', 'failed'),
            'postvalidation_physical_entry')
    failed = entry['status'] == 'failed'
    fields = {'status', 'attempts', 'row', 'attempt_history'}
    if failed:
        fields.add('failure')
    require(set(entry) == fields
            and type(entry['attempts']) is int and entry['attempts'] == 1,
            'postvalidation_physical_entry')
    raw = entry['row']
    require(type(raw) is dict and raw.get('question_id') == question_id
            and 'correct' in raw
            and (raw['correct'] is None or type(raw['correct']) is bool),
            'postvalidation_physical_row')
    failure = raw.get('benchmark_failure')
    require((failure is None or type(failure) is str)
            and failed is (bool(failure) or raw['correct'] is None)
            and (not failed or entry['failure'] == bounded_failure_text(
                failure or 'no_valid_prediction_verdict')),
            'postvalidation_physical_status')
    if 'strict_failure' in raw:
        require(type(raw['strict_failure']) is bool and raw['strict_failure'] is failed,
                'postvalidation_physical_failure_flag')
    history = entry['attempt_history']
    require(type(history) is list and len(history) == 1
            and type(history[0]) is dict
            and set(history[0]) == {'attempt', 'status', 'failure', 'row'}
            and type(history[0]['attempt']) is int and history[0]['attempt'] == 1
            and history[0]['status'] == entry['status']
            and history[0]['failure'] == entry.get('failure')
            and canonical(history[0]['row']) == canonical(raw),
            'postvalidation_physical_history')
    ledger_row = dict(raw)
    if failed:
        ledger_row['correct'] = False
        ledger_row['benchmark_failure'] = entry['failure']
    projected = list(reconcile_results([question_id], [ledger_row]).rows)
    # Only the documented checkpoint/reconciler-derived fields above may differ.
    # Canonical comparison binds every remaining field without type coercion.
    require(canonical(projected) == canonical(rows),
            'postvalidation_physical_checkpoint')


def summarize(data, validated, snapshot, supervisor, *, manifest_sha, archive_sha,
              checkpoint_sha, checkpoint_attestation, reconcile_results, bounded_failure_text, run):
    """Additional run-specific fences after the stock strict validator passes."""
    check_recipe(data, run)
    rows = validated['rows']
    require(len(rows) == 1 and rows[0]['question_id'] == run.QUESTION_ID,
            'postvalidation_question')
    validate_physical_checkpoint(snapshot, rows, question_id=run.QUESTION_ID,
                                 reconcile_results=reconcile_results,
                                 bounded_failure_text=bounded_failure_text)
    require(checkpoint_attestation(snapshot, rows) == data['execution']['checkpoint'],
            'postvalidation_checkpoint_binding')
    counts = validated['counts']
    segments = data['execution']['segments']
    require(len(segments) == 1 and segments[0]['status'] == 'complete'
            and segments[0]['attempted_attempts'] == 1
            and counts['expected'] == counts['attempted'] == counts['unique_attempted']
            == counts['total_attempts'] == 1
            and snapshot['entries'][run.QUESTION_ID]['attempts'] == 1,
            'postvalidation_resumed_or_incomplete')
    require(supervisor['manifest_sha256'] == manifest_sha
            and supervisor['source_and_dataset_unchanged'] is True
            and supervisor['exception_type'] is None
            and supervisor['postrun_exception_type'] is None,
            'postvalidation_supervisor_identity')
    outcome = supervisor['outcome']
    process_ok = (outcome['status'] == 'completed'
                  and type(outcome['returncode']) is int and outcome['returncode'] == 0
                  and all(outcome.get(field) is True for field in (
                      'safe_to_continue', 'cleanup_complete', 'child_reaped',
                      'group_absent_after_reap', 'terminal_receipt_written'))
                  and outcome.get('errors') == [] and outcome.get('cleanup_warnings') == [])
    segment = segments[0]
    usage = {}
    for role in ('reader', 'judge', 'retrieval', 'memory_pipeline'):
        meter = segment[role + '_usage']
        # The stock validator has checked every availability flag, finite count,
        # token reconciliation and cross-role ownership. Preserve unavailable
        # values as null rather than making up zeros or a dollar estimate.
        usage[role] = {field: meter[field] for field in (
            'calls', 'calls_available', 'request_attempts', 'request_attempts_available',
            'successful_responses', 'successful_responses_available',
            'prompt_tokens', 'completion_tokens', 'total_tokens', 'token_usage_available',
            'cost_usd', 'cost_available')}
    reader_judge_measured = all(
        usage[role].get(flag) is True and type(usage[role].get(field)) is int
        and usage[role][field] >= 1
        for role in ('reader', 'judge')
        for field, flag in (('calls', 'calls_available'),
                            ('request_attempts', 'request_attempts_available'))
    )
    indexing = rows[0].get('indexing') or {}
    passed = (process_ok and not segment.get('instrumentation_errors')
              and reader_judge_measured
              and counts['completed'] == 1 and counts['failed'] == counts['missing'] == 0
              and indexing.get('complete') is True and indexing.get('healthy') is True
              and indexing.get('outcome') in ('success', 'success_with_summary_degradation')
              and type(rows[0].get('correct')) is bool)
    return {
        'schema': 'stock-q1-postvalidation-v2',
        'status': 'scored_q1_completed' if passed else 'validated_q1_has_failures',
        'benchmark_completed_without_faults': passed,
        'strict_scored_artifact_validated': True,
        'physical_checkpoint_bound': True, 'process_completed_cleanly': process_ok,
        'reader_judge_calls_measured': reader_judge_measured,
        'question_id': run.QUESTION_ID, 'run_id': validated['run_id'],
        'manifest_sha256': manifest_sha, 'archive_sha256': archive_sha,
        'checkpoint_sha256': checkpoint_sha, 'counts': counts,
        'answer_correct': rows[0].get('correct'), 'scores': validated['scores'],
        'indexing_outcome': indexing.get('outcome'),
        'item_indexing_healthy': indexing.get('healthy'),
        'summary_healthy': indexing.get('summary_healthy'),
        'summary_degraded_questions': validated['summary_degraded_questions'],
        'summary_degraded_sessions': validated['summary_degraded_sessions'],
        'per_role_usage': usage, 'elapsed_s': validated['elapsed_s'],
        'global_paid_call_cap': None, 'new_provider_calls': 0,
        'full_500_readiness_verified': False, 'semantic_quality_guaranteed': False,
        'representative_sample': False, 'official_comparable': False,
    }


def validate(manifest_sha):
    run = load_run_helper(manifest_sha)
    manifest = run.validate_package(manifest_sha)
    run.verify_source(manifest)
    require(run.sha(run.DATA) == run.DATASET_SHA, 'postvalidation_dataset_drift')
    sys.path[:0] = [str(run.SOURCE), str(run.SOURCE / 'benchmarks')]
    # These imports define validators only; no registry connection, benchmark
    # adapter main(), LLM client or checkpoint writer is constructed.
    from benchmarks.lme_registry import _load_registry_artifact
    from benchmarks.lme_protocol import validate_strict_artifact
    from benchmarks.archive_evidence import checkpoint_attestation
    from benchmarks.strictness import reconcile_results, bounded_failure_text
    benchmark = run.OUTPUT / 'benchmark'
    pointer = benchmark / 'longmemeval-v2-hymem.json'
    raw_pointer = read_json(pointer)
    require(set(raw_pointer) == {'archive', 'run_id', 'artifact_digest'}
            and type(raw_pointer['archive']) is str
            and Path(raw_pointer['archive']).name == raw_pointer['archive'],
            'postvalidation_pointer')
    archive = benchmark / raw_pointer['archive']
    read_json(archive)  # Refuse a symlink before the stock pointer reader opens it.
    data, actual_name, _digest, compatibility = _load_registry_artifact(pointer)
    require(actual_name == archive.name
            and compatibility == 'pointer-target-digest-validated', 'postvalidation_archive_binding')
    validated = validate_strict_artifact(data, path=archive, require_scored=True)
    checkpoint = benchmark / 'checkpoint.json'
    supervisor = read_json(run.OUTPUT / 'supervisor-summary.json')
    require(read_json(run.OUTPUT / 'invocation/terminal.json') == supervisor['outcome'],
            'postvalidation_terminal_receipt_binding')
    result = summarize(
        data, validated, read_json(checkpoint), supervisor,
        manifest_sha=manifest_sha, archive_sha=run.sha(archive), checkpoint_sha=run.sha(checkpoint),
        checkpoint_attestation=checkpoint_attestation, reconcile_results=reconcile_results,
        bounded_failure_text=bounded_failure_text, run=run,
    )
    run.validate_package(manifest_sha)
    run.verify_source(manifest)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    os.environ.clear()
    os.environ.update({'PATH': '/home/node/hymem-env/bin:/usr/bin:/bin',
                       'HOME': '/tmp', 'PYTHONDONTWRITEBYTECODE': '1',
                       'PYTHONNOUSERSITE': '1', 'LANG': 'C.UTF-8'})
    try:
        result = validate(args.manifest_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result['benchmark_completed_without_faults'] else 1
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({'status': 'stock_q1_postvalidation_failed',
                          'exception_type': type(exc).__name__,
                          'benchmark_completed_without_faults': False,
                          'usage_verified': False, 'new_provider_calls': 0}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
