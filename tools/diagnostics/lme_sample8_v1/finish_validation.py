"""One-shot dependent offline validation of the already-running sample-eight run.

No scheduler, paid work, reruns, container deletion, or raw-log persistence.
Install beside the exact pinned host_control.py; invoke with python3 -I -B and
no arguments. The exclusive finish-validation directory is a permanent latch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import time
import types

ROOT = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/sample8-stock-v1')
MANIFEST_SHA = 'fa471c38440dd70e8527b75171b1af8fc9b48ee98081df9738b45093f41efffc'
CONTROLLER_SHA = '424fcfe119c0662ceecc4dff49a753e1b96500b1d0e030117415eee389efa28e'
LIVE_CID = '56729d5659b7f8807956ff8a7552d548afaf9471449bf6e5c33c03baa6b42a87'
QUESTION_IDS = ['c18a7dc8', 'gpt4_e061b84f', 'gpt4_483dd43c', 'gpt4_7de946e7',
                '945e3d21', '71315a70', '72e3ee87', 'e61a7584']
SOURCE_INDICES = [213, 262, 329, 339, 370, 372, 392, 400]
LIVE_WAIT_SECONDS = 32400
VALIDATION_WAIT_SECONDS = 600
OUTPUT_LIMIT = 65536
COUNT_FIELDS = ('expected', 'attempted', 'unique_attempted', 'total_attempts', 'completed', 'failed', 'missing')
USAGE_FIELDS = ('calls', 'calls_available', 'request_attempts', 'request_attempts_available',
                'successful_responses', 'successful_responses_available', 'prompt_tokens',
                'completion_tokens', 'total_tokens', 'token_usage_available', 'cost_usd', 'cost_available')
REPORT_FIELDS = frozenset(('schema', 'status', 'benchmark_completed_without_faults',
    'strict_scored_artifact_validated', 'physical_checkpoint_bound', 'process_completed_cleanly',
    'reader_judge_calls_measured', 'question_ids', 'sample', 'seed', 'source_indices', 'run_id',
    'manifest_sha256', 'archive_sha256', 'checkpoint_sha256', 'counts', 'scores', 'per_question',
    'summary_degraded_questions', 'summary_degraded_sessions', 'per_role_usage', 'aggregate_paid_usage',
    'canary_accounted_separately', 'elapsed_s', 'global_paid_call_cap', 'new_provider_calls',
    'full_500_readiness_verified', 'semantic_quality_guaranteed', 'representative_sample', 'official_comparable'))


class FinishError(RuntimeError):
    pass


def require(value, code):
    if not value:
        raise FinishError(code)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(FinishError('nonfinite_json')))


def command(argv, *, timeout=30):
    """Bound stdout in memory; stderr is never copied or persisted.

    A timeout only kills this Docker CLI client, never the existing paid run.
    """
    result = bytearray()
    deadline = time.monotonic() + timeout
    with subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, 'command_timeout')
                    for key, _ in selector.select(min(remaining, 1)):
                        block = os.read(key.fileobj.fileno(), 8192)
                        if not block:
                            selector.unregister(key.fileobj)
                            continue
                        require(len(result) + len(block) <= OUTPUT_LIMIT, 'command_output_limit')
                        result.extend(block)
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
            require(process.returncode == 0, 'command_failed')
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            raise
    return bytes(result).decode('utf-8')


def controller_command(argv, *, text):
    require(text is True and len(argv) == 3 and argv[:2] == ['docker', 'inspect'], 'controller_command')
    raw = command(argv)
    values = strict_json(raw)
    require(type(values) is list and len(values) == 1 and values[0]['Id'] == argv[2], 'inspect_container_id')
    return raw


def load_controller():
    require(__debug__, 'optimized_python_forbidden')
    path = Path(__file__).resolve().with_name('host_control.py')
    require(path.resolve() == path and stat.S_ISREG(path.lstat().st_mode), 'controller_path')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONTROLLER_SHA, 'controller_pin')
    module = types.ModuleType('sample8_finish_verified_controller')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require(module.ROOT == ROOT and module.PIN == MANIFEST_SHA, 'controller_identity')
    module.subprocess = types.SimpleNamespace(check_output=controller_command)
    return module


def integer(value, maximum):
    return type(value) is int and 0 <= value <= maximum


def usage_projection(value):
    require(type(value) is dict and set(value) == set(USAGE_FIELDS), 'usage_fields')
    for key, item in value.items():
        if key.endswith('_available'):
            require(type(item) is bool, 'usage_availability')
        elif key == 'cost_usd':
            require(item is None or type(item) in (int, float) and math.isfinite(item) and item >= 0, 'usage_cost')
        else:
            require(item is None or integer(item, 10**12), 'usage_counter')
    return dict(value)


def project_validator(raw, exit_code):
    """A closed, text-free projection, never a substitute for the stock validator."""
    require(len(raw.encode()) <= OUTPUT_LIMIT, 'validator_output_limit')
    value = strict_json(raw)
    require(type(value) is dict and type(value.get('new_provider_calls')) is int
            and value['new_provider_calls'] == 0, 'validator_calls')
    if value.get('status') == 'stock_sample8_postvalidation_failed':
        require(set(value) == {'status', 'exception_type', 'benchmark_completed_without_faults',
                              'usage_verified', 'new_provider_calls'}
                and value['benchmark_completed_without_faults'] is False
                and value['usage_verified'] is False and exit_code != 0, 'validator_failure_receipt')
        # Deliberately exclude even exception names from the saved boundary.
        return {key: value[key] for key in ('status', 'benchmark_completed_without_faults',
                                           'usage_verified', 'new_provider_calls')}
    require(set(value) == REPORT_FIELDS and value['schema'] == 'stock-lme-sample8-postvalidation-v1', 'validator_schema')
    require(value['manifest_sha256'] == MANIFEST_SHA and value['question_ids'] == QUESTION_IDS
            and type(value['sample']) is int and value['sample'] == 8
            and type(value['seed']) is int and value['seed'] == 0
            and value['source_indices'] == SOURCE_INDICES
            and all(type(i) is int for i in value['source_indices']), 'validator_identity')
    for field in ('archive_sha256', 'checkpoint_sha256'):
        require(type(value[field]) is str and re.fullmatch('[0-9a-f]{64}', value[field]), 'validator_artifact_pin')
    require(type(value['run_id']) is str and re.fullmatch('sha256:[0-9a-f]{64}', value['run_id']), 'validator_run_id')
    passed = value['benchmark_completed_without_faults']
    require(type(passed) is bool and passed is (exit_code == 0)
            and value['status'] == ('scored_sample8_completed' if passed else 'validated_sample8_has_failures'), 'validator_exit_verdict')
    for field in ('strict_scored_artifact_validated', 'physical_checkpoint_bound', 'canary_accounted_separately'):
        require(value[field] is True, 'validator_proof')
    for field in ('process_completed_cleanly', 'reader_judge_calls_measured'):
        require(type(value[field]) is bool and (not passed or value[field] is True), 'validator_proof')
    for field in ('full_500_readiness_verified', 'semantic_quality_guaranteed', 'representative_sample', 'official_comparable'):
        require(value[field] is False, 'validator_scope')
    require(value['global_paid_call_cap'] is None, 'validator_scope')
    counts = value['counts']
    require(type(counts) is dict and set(counts) == set(COUNT_FIELDS)
            and all(integer(counts[k], 8) for k in COUNT_FIELDS)
            and all(counts[k] == 8 for k in COUNT_FIELDS[:4])
            and counts['completed'] + counts['failed'] + counts['missing'] == 8
            and (not passed or counts['completed'] == 8), 'validator_counts')
    rows = value['per_question']
    flags = ('answer_correct', 'strict_failure', 'indexing_complete', 'item_indexing_healthy', 'summary_healthy')
    counters = ('summary_degraded_sessions', 'summary_missing_sessions', 'malformed_summaries')
    fields = {'question_id', 'indexing_outcome', *flags, *counters}
    require(type(rows) is list and len(rows) == 8, 'validator_questions')
    for qid, row in zip(QUESTION_IDS, rows):
        require(type(row) is dict and set(row) == fields and row['question_id'] == qid
                and all(row[k] is None or type(row[k]) is bool for k in flags)
                and all(row[k] is None or integer(row[k], 398) for k in counters)
                and row['indexing_outcome'] in (None, 'success', 'success_with_summary_degradation', 'failure'), 'validator_question_metadata')
        if passed:
            require(type(row['answer_correct']) is bool and row['strict_failure'] is False
                    and row['indexing_complete'] is True and row['item_indexing_healthy'] is True
                    and row['indexing_outcome'] in ('success', 'success_with_summary_degradation'), 'validator_question_success')
    for field, cap in (('summary_degraded_questions', 8), ('summary_degraded_sessions', 398)):
        require(integer(value[field], cap), 'validator_summary_counts')
    require(type(value['elapsed_s']) in (int, float) and math.isfinite(value['elapsed_s'])
            and 0 <= value['elapsed_s'] <= LIVE_WAIT_SECONDS, 'validator_elapsed')
    roles = value['per_role_usage']
    require(type(roles) is dict and set(roles) == {'reader', 'judge', 'retrieval', 'memory_pipeline', 'canary'}, 'validator_usage_roles')
    projected = {k: v for k, v in value.items() if k not in ('scores', 'per_role_usage', 'aggregate_paid_usage')}
    projected['per_role_usage'] = {k: usage_projection(v) for k, v in roles.items()}
    projected['aggregate_paid_usage'] = usage_projection(value['aggregate_paid_usage'])
    return projected


def write_new(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    require(len(raw) <= OUTPUT_LIMIT, 'receipt_limit')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw)


def clean_exit(state):
    require(state['status'] == 'exited' and type(state['exit_code']) is int
            and state['exit_code'] == 0 and state['oom_killed'] is False
            and type(state['pid']) is int and state['pid'] == 0, 'live_not_clean')


def stop_owned_validator(ctl, cid, plan):
    require(cid != LIVE_CID and plan['network'] == 'none', 'cleanup_ownership')
    state = ctl.inspect(cid, plan)
    if state['status'] not in ('exited', 'created'):
        require(command(['docker', 'stop', '--time', '10', cid], timeout=20).strip() == cid, 'cleanup_stop')
        command(['docker', 'wait', cid], timeout=10)
    state = ctl.inspect(cid, plan)
    require(state['status'] in ('exited', 'created') and type(state['pid']) is int
            and state['pid'] == 0 and state['oom_killed'] is False, 'cleanup_terminal')
    return state


def execute():
    """Run once. All failure artifacts remain; a later invocation never resumes."""
    state_dir = None
    stage = 'admission'
    result = {'schema': 'sample8-dependent-validation-v1', 'status': 'failed',
              'live_container_id': LIVE_CID, 'manifest_sha256': MANIFEST_SHA,
              'new_provider_calls': 0, 'raw_logs_saved': False, 'validation_container_id': None,
              'validation_terminal': None, 'cleanup_attempted': False}
    try:
        ctl = load_controller()
        require(ROOT.resolve() == ROOT and ROOT.is_dir(), 'run_root')
        candidate = ROOT / 'finish-validation'
        candidate.mkdir(mode=0o700)  # Permanent, exclusive latch even after failure.
        state_dir = candidate
        write_new(state_dir / 'intent.json', result)
        manifest, host = ctl.definitions()
        require(manifest['question_ids'] == QUESTION_IDS, 'manifest_questions')
        require(ctl.js(ROOT / 'live-container-id.json') == {'container_id': LIVE_CID}, 'live_container_id')
        live_plan = host.command(str(ROOT), str(ctl.SOURCE), MANIFEST_SHA, live=True)
        initial = ctl.inspect(LIVE_CID, live_plan)
        require(initial['status'] in ('running', 'exited'), 'live_state')
        stage = 'waiting_live'
        write_new(state_dir / 'waiting-live.json', {
            'schema': 'sample8-dependent-validation-wait-v1', 'live_container_id': LIVE_CID,
            'manifest_sha256': MANIFEST_SHA, 'source_and_package_verified': True,
            'configuration_verified': True, 'live_state': initial,
            'maximum_wait_seconds': LIVE_WAIT_SECONDS, 'new_provider_calls': 0})
        wait_code = command(['docker', 'wait', LIVE_CID], timeout=LIVE_WAIT_SECONDS).strip()
        require(wait_code == '0', 'live_wait_exit')
        clean_exit(ctl.inspect(LIVE_CID, live_plan, 'exited'))
        manifest, host = ctl.definitions()  # Recheck sealed source after the wait.
        stage = 'creating_validation'
        for name in ('validation-create-intent.json', 'validation-container-id.json',
                     'validation-created.json', 'validation-start-intent.json'):
            path = ROOT / name
            require(not path.exists() and not path.is_symlink(), 'validation_already_attempted')
        plan = host.validation_command(str(ROOT), str(ctl.SOURCE), MANIFEST_SHA)
        require(plan['network'] == 'none' and plan['new_provider_calls'] == 0
                and '/run/deepseek.env' not in plan['mounts']
                and all(writable is False for _, writable in plan['mounts'].values())
                and plan['command'][:2] == ['docker', 'create']
                and '/diag/q1_stock_validate.py' in plan['command'], 'validation_plan')
        ctl.save('validation-create-intent.json', {'manifest_sha256': MANIFEST_SHA, 'plan': plan})
        cid = command(plan['command']).strip()
        require(re.fullmatch('[0-9a-f]{64}', cid) and cid != LIVE_CID, 'validation_container_id')
        result['validation_container_id'] = cid
        ctl.save('validation-container-id.json', {'container_id': cid})
        ctl.save('validation-created.json', ctl.inspect(cid, plan, 'created'))
        stage = 'starting_validation'
        clean_exit(ctl.inspect(LIVE_CID, live_plan, 'exited'))
        ctl.inspect(cid, plan, 'created')
        ctl.save('validation-start-intent.json', {'container_id': cid, 'manifest_sha256': MANIFEST_SHA})
        require(command(['docker', 'start', cid]).strip() == cid, 'validation_start')
        stage = 'waiting_validation'
        code = command(['docker', 'wait', cid], timeout=VALIDATION_WAIT_SECONDS).strip()
        require(re.fullmatch('[0-9]{1,3}', code), 'validation_wait_exit')
        final = ctl.inspect(cid, plan, 'exited')
        require(final['pid'] == 0 and final['oom_killed'] is False
                and type(final['exit_code']) is int and final['exit_code'] == int(code), 'validation_exit')
        result['validation_terminal'] = final
        stage = 'reading_validation'
        metadata = project_validator(command(['docker', 'logs', cid]), final['exit_code'])
        write_new(state_dir / 'validator.json', metadata)
        result['status'] = 'validated' if metadata['benchmark_completed_without_faults'] else 'validation_failed'
        stage = 'complete'
    except Exception as exc:
        result['failure_code'] = str(exc) if type(exc) is FinishError else (
            'duplicate_invocation' if type(exc) is FileExistsError else 'operation_failed')
        if result['validation_container_id'] is not None and result['validation_terminal'] is None:
            result['cleanup_attempted'] = True
            try:
                result['validation_terminal'] = stop_owned_validator(ctl, result['validation_container_id'], plan)
            except Exception:
                result['cleanup_failed'] = True
    result['stage'] = stage
    if state_dir is not None:
        write_new(state_dir / 'final.json', result)
    return result


def main():
    require(len(sys.argv) == 1, 'arguments_forbidden')
    result = execute()
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
