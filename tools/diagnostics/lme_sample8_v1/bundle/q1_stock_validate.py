"""Offline, read-only validation of one fresh eight-question stock LME run.

Requires pinned source/package/results mounted read-only, network=none, and no
credentials. Only bounded identifiers, health, verdicts and accounting escape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import types

SAMPLE = 8
SEED = 0
SOURCE_INDICES = [213, 262, 329, 339, 370, 372, 392, 400]
QUESTION_IDS = ['c18a7dc8', 'gpt4_e061b84f', 'gpt4_483dd43c', 'gpt4_7de946e7',
                '945e3d21', '71315a70', '72e3ee87', 'e61a7584']
USAGE_FIELDS = (
    'calls', 'calls_available', 'request_attempts', 'request_attempts_available',
    'successful_responses', 'successful_responses_available', 'prompt_tokens',
    'completion_tokens', 'total_tokens', 'token_usage_available',
    'cost_usd', 'cost_available',
)


def require(value, code):
    if not value:
        raise RuntimeError(code)


def canonical(value):
    """JSON equality without Python's bool/int/float equality coercion."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_json(path):
    require(path.resolve() == path and stat.S_ISREG(path.lstat().st_mode),
            'postvalidation_non_regular_path')
    return json.loads(path.read_text())


def load_run_helper(manifest_sha):
    package = Path('/diag')
    path = package / 'manifest.json'
    require(path.resolve() == path and stat.S_ISREG(path.lstat().st_mode),
            'postvalidation_manifest_path')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest_sha, 'postvalidation_manifest_pin')
    manifest = json.loads(raw)
    path = package / 'q1_stock_run.py'
    require(path.resolve() == path and stat.S_ISREG(path.lstat().st_mode),
            'postvalidation_helper_path')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest['helper_sha256']['q1_stock_run.py'],
            'postvalidation_helper_pin')
    module = types.ModuleType('sample8_stock_run')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def expected_ids(manifest, run):
    require(manifest.get('schema') == 'stock-lme-sample8-development-run-preparation-v1'
            and type(manifest.get('sample')) is int and manifest['sample'] == SAMPLE
            and type(manifest.get('seed')) is int and manifest['seed'] == SEED
            and canonical(manifest.get('source_indices')) == canonical(SOURCE_INDICES)
            and type(run.SAMPLE) is int and run.SAMPLE == SAMPLE
            and type(run.SEED) is int and run.SEED == SEED
            and canonical(run.SOURCE_INDICES) == canonical(SOURCE_INDICES),
            'postvalidation_selection')
    ids = manifest.get('question_ids')
    require(type(ids) is list and len(ids) == SAMPLE
            and all(type(qid) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}', qid)
                    for qid in ids) and len(set(ids)) == SAMPLE and ids == QUESTION_IDS,
            'postvalidation_expected_ids')
    return list(ids)


def check_recipe(data, run):
    expected = {
        'sample': SAMPLE, 'seed': SEED, 'workers': 1, 'top_k': 15,
        'auto_ability': True, 'permissive_default': True, 'no_dream': False,
        'embeddings': False, 'aggregation_nodes': False, 'episode_granularity': False,
        'retrieval_only': False, 'distill': False, 'indexing_require_healthy': True,
        'source_order_validated': True, 'dataset_expected_count': 500,
        'scored_run': True, 'label_free_answer_path': True, 'prereg': None,
        'indexing_max_cycles': 100, 'indexing_timeout_s': 3600.0,
        'judge_protocol': 'legacy-custom', 'hymem_thinking': 'disabled',
    }
    config = data['config']
    for field, wanted in expected.items():
        value = config.get(field)
        require(value == wanted and (type(value) is type(wanted)
                or type(wanted) is float and type(value) is int), 'postvalidation_recipe')
    require(config.get('scales', config.get('scale')) == 'S'
            and data['manifest']['protocol_split'] == 'full'
            and type(data['manifest']['expected_count']) is int
            and data['manifest']['expected_count'] == SAMPLE
            and data['manifest']['data_hash'] == 'sha256:' + run.DATASET_SHA,
            'postvalidation_dataset_recipe')
    for role in ('reader', 'judge', 'memory_pipeline'):
        identity = data['models'][role]
        require(identity['model'] == run.MODEL and identity['endpoint_origin'] == run.ENDPOINT
                and identity['endpoint_sha256'] == 'sha256:' + hashlib.sha256(
                    run.ENDPOINT.encode()).hexdigest(), 'postvalidation_model')
        field = 'effective_extra_body' if role == 'memory_pipeline' else 'extra_body'
        require(identity[field] == {'thinking': {'type': 'disabled'}}, 'postvalidation_thinking')


def validate_physical_checkpoint(snapshot, rows, *, question_ids, reconcile_results,
                                 bounded_failure_text):
    """Bind every physical raw row through the genuine stock projection.

    The only differences allowed are stock's derived strict_failure and, for a
    failed entry, bounded failure text/False verdict. Histories are bound too.
    """
    require(snapshot['expected_ids'] == question_ids and snapshot['status'] == 'complete'
            and snapshot['scored'] is True and snapshot['verdict_key'] == 'correct'
            and type(snapshot['entries']) is dict
            and set(snapshot['entries']) == set(question_ids),
            'postvalidation_physical_checkpoint')
    ledger_rows = []
    for qid in question_ids:
        entry = snapshot['entries'][qid]
        require(type(entry) is dict and entry.get('status') in ('completed', 'failed'),
                'postvalidation_physical_entry')
        failed = entry['status'] == 'failed'
        fields = {'status', 'attempts', 'row', 'attempt_history'} | ({'failure'} if failed else set())
        require(set(entry) == fields and type(entry['attempts']) is int and entry['attempts'] == 1,
                'postvalidation_physical_entry')
        raw = entry['row']
        require(type(raw) is dict and raw.get('question_id') == qid and 'correct' in raw
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
        require(type(history) is list and len(history) == 1 and type(history[0]) is dict
                and set(history[0]) == {'attempt', 'status', 'failure', 'row'}
                and type(history[0]['attempt']) is int and history[0]['attempt'] == 1
                and history[0]['status'] == entry['status']
                and history[0]['failure'] == entry.get('failure')
                and canonical(history[0]['row']) == canonical(raw),
                'postvalidation_physical_history')
        projected = dict(raw)
        if failed:
            projected['correct'] = False
            projected['benchmark_failure'] = entry['failure']
        ledger_rows.append(projected)
    projected_rows = list(reconcile_results(question_ids, ledger_rows).rows)
    require(canonical(projected_rows) == canonical(rows), 'postvalidation_physical_checkpoint')


def usage_projection(meter):
    # The strict stock validator admits the meter first; this boundary also
    # refuses text, nonfinite values, and unknown output fields in its report.
    result = {field: meter[field] for field in USAGE_FIELDS}
    require(all(value is None or type(value) in (bool, int, float) for value in result.values())
            and all(type(value) is not float or math.isfinite(value) for value in result.values()),
            'postvalidation_usage_projection')
    return result


def summarize(data, validated, snapshot, supervisor, *, manifest, manifest_sha,
              archive_sha, checkpoint_sha, checkpoint_attestation, reconcile_results,
              bounded_failure_text, aggregate_usage_snapshots, run):
    """Run-specific fences after stock validate_strict_artifact succeeds."""
    ids = expected_ids(manifest, run)
    check_recipe(data, run)
    rows = validated['rows']
    require(type(rows) in (list, tuple) and len(rows) == SAMPLE
            and [row.get('question_id') for row in rows] == ids, 'postvalidation_question')
    validate_physical_checkpoint(snapshot, rows, question_ids=ids,
                                 reconcile_results=reconcile_results,
                                 bounded_failure_text=bounded_failure_text)
    require(canonical(checkpoint_attestation(snapshot, rows))
            == canonical(data['execution']['checkpoint']), 'postvalidation_checkpoint_binding')
    counts = validated['counts']
    require(canonical(counts) == canonical(snapshot['counts']), 'postvalidation_counts_binding')
    segments = data['execution']['segments']
    require(len(segments) == 1 and segments[0]['status'] == 'complete'
            and type(segments[0]['attempted_attempts']) is int
            and segments[0]['attempted_attempts'] == SAMPLE
            and all(type(counts.get(field)) is int and counts[field] == SAMPLE
                    for field in ('expected', 'attempted', 'unique_attempted', 'total_attempts')),
            'postvalidation_resumed_or_incomplete')
    require(supervisor['manifest_sha256'] == manifest_sha
            and supervisor['source_and_dataset_unchanged'] is True
            and supervisor['exception_type'] is None and supervisor['postrun_exception_type'] is None,
            'postvalidation_supervisor_identity')
    outcome = supervisor['outcome']
    process_ok = (outcome['status'] == 'completed'
                  and type(outcome['returncode']) is int and outcome['returncode'] == 0
                  and all(outcome.get(field) is True for field in (
                      'safe_to_continue', 'cleanup_complete', 'child_reaped',
                      'group_absent_after_reap', 'terminal_receipt_written'))
                  and outcome.get('errors') == [] and outcome.get('cleanup_warnings') == [])
    segment = segments[0]
    usage = {role: usage_projection(segment[role + '_usage'])
             for role in ('reader', 'judge', 'retrieval', 'memory_pipeline')}
    # Pinned stock source accounts this dedicated-client canary separately from
    # all four role meters. No mutable result/fixture defines that ownership.
    canary = segment['extraction_canary']
    require(canary['status'] == 'passed' and canary['client_closed'] is True,
            'postvalidation_canary')
    usage['canary'] = usage_projection(canary['usage'])
    total_usage = usage_projection(aggregate_usage_snapshots(usage.values()))
    reader_judge_measured = all(
        usage[role].get(flag) is True and type(usage[role].get(field)) is int
        and usage[role][field] >= counts['completed']
        for role in ('reader', 'judge')
        for field, flag in (('calls', 'calls_available'), ('request_attempts', 'request_attempts_available')))
    metadata = []
    for row in rows:
        indexing = row.get('indexing') or {}
        summary = (indexing.get('final_status') or {}).get('summary_health') or {}
        metadata.append({
            'question_id': row['question_id'], 'answer_correct': row.get('correct'),
            'strict_failure': row.get('strict_failure'),
            'indexing_complete': indexing.get('complete'), 'item_indexing_healthy': indexing.get('healthy'),
            'indexing_outcome': indexing.get('outcome'), 'summary_healthy': indexing.get('summary_healthy'),
            **{field: summary.get(field) for field in (
                'summary_degraded_sessions', 'summary_missing_sessions', 'malformed_summaries')},
        })
    passed = (process_ok and not segment.get('instrumentation_errors') and reader_judge_measured
              and counts['completed'] == SAMPLE and counts['failed'] == counts['missing'] == 0
              and all(item['indexing_complete'] is True and item['item_indexing_healthy'] is True
                      and item['indexing_outcome'] in ('success', 'success_with_summary_degradation')
                      and item['strict_failure'] is False and type(item['answer_correct']) is bool
                      for item in metadata))
    return {
        'schema': 'stock-lme-sample8-postvalidation-v1',
        'status': 'scored_sample8_completed' if passed else 'validated_sample8_has_failures',
        'benchmark_completed_without_faults': passed, 'strict_scored_artifact_validated': True,
        'physical_checkpoint_bound': True, 'process_completed_cleanly': process_ok,
        'reader_judge_calls_measured': reader_judge_measured,
        'question_ids': ids, 'sample': SAMPLE, 'seed': SEED, 'source_indices': SOURCE_INDICES,
        'run_id': validated['run_id'], 'manifest_sha256': manifest_sha,
        'archive_sha256': archive_sha, 'checkpoint_sha256': checkpoint_sha,
        'counts': counts, 'scores': validated['scores'], 'per_question': metadata,
        'summary_degraded_questions': validated['summary_degraded_questions'],
        'summary_degraded_sessions': validated['summary_degraded_sessions'],
        'per_role_usage': usage, 'aggregate_paid_usage': total_usage,
        'canary_accounted_separately': True, 'elapsed_s': validated['elapsed_s'],
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
    from benchmarks.lme_registry import _load_registry_artifact
    from benchmarks.lme_protocol import validate_strict_artifact
    from benchmarks.archive_evidence import checkpoint_attestation
    from benchmarks.strictness import reconcile_results, bounded_failure_text, aggregate_usage_snapshots
    benchmark = run.OUTPUT / 'benchmark'
    pointer = benchmark / 'longmemeval-v2-hymem.json'
    raw_pointer = read_json(pointer)
    require(set(raw_pointer) == {'archive', 'run_id', 'artifact_digest'}
            and type(raw_pointer['archive']) is str
            and Path(raw_pointer['archive']).name == raw_pointer['archive'], 'postvalidation_pointer')
    archive = benchmark / raw_pointer['archive']
    read_json(archive)
    data, actual_name, _digest, compatibility = _load_registry_artifact(pointer)
    require(actual_name == archive.name and compatibility == 'pointer-target-digest-validated',
            'postvalidation_archive_binding')
    validated = validate_strict_artifact(data, path=archive, require_scored=True)
    checkpoint = benchmark / 'checkpoint.json'
    supervisor = read_json(run.OUTPUT / 'supervisor-summary.json')
    require(read_json(run.OUTPUT / 'invocation/terminal.json') == supervisor['outcome'],
            'postvalidation_terminal_receipt_binding')
    result = summarize(
        data, validated, read_json(checkpoint), supervisor, manifest=manifest,
        manifest_sha=manifest_sha, archive_sha=run.sha(archive), checkpoint_sha=run.sha(checkpoint),
        checkpoint_attestation=checkpoint_attestation, reconcile_results=reconcile_results,
        bounded_failure_text=bounded_failure_text, aggregate_usage_snapshots=aggregate_usage_snapshots, run=run,
    )
    run.validate_package(manifest_sha)
    run.verify_source(manifest)
    require(run.sha(run.DATA) == run.DATASET_SHA, 'postvalidation_dataset_drift')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    os.environ.clear()
    os.environ.update({'PATH': '/home/node/hymem-env/bin:/usr/bin:/bin', 'HOME': '/tmp',
                       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'LANG': 'C.UTF-8'})
    try:
        result = validate(args.manifest_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result['benchmark_completed_without_faults'] else 1
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({'status': 'stock_sample8_postvalidation_failed',
                          'exception_type': type(exc).__name__,
                          'benchmark_completed_without_faults': False,
                          'usage_verified': False, 'new_provider_calls': 0}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
