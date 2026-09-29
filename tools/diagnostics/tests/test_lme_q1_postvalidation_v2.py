"""Offline regression against the frozen R5 writer, not invented equal rows.

Run with HYMEM_Q1_VERIFIER_SOURCE pointing at the independently frozen R5 tree.
No provider, saved benchmark payload, or production store is used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'tools/diagnostics/lme_stock_q1'
R5_MANIFEST = ROOT / 'docs/patches/2026-09-24-lme-independent-summary-indexing-r5-manifest.json'
R5_PIN = 'c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'
OLD_PIN = 'c4e3ccdef2d0a558d43be1d800192dd94f747bbf233a8fbfbe7c09154e601db3'


def load(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('network forbidden during verifier regression')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'getnameinfo'):
        monkeypatch.setattr(socket, name, forbidden)


@pytest.fixture(scope='module')
def modules():
    source = Path(os.environ['HYMEM_Q1_VERIFIER_SOURCE']).resolve()
    raw = R5_MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == R5_PIN
    pins = json.loads(raw)['source_sha256']
    for relative, pin in pins.items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == pin
    sys.path.insert(0, str(source))
    from benchmarks import strictness, archive_evidence
    assert Path(strictness.__file__).resolve() == source / 'benchmarks/strictness.py'
    assert Path(archive_evidence.__file__).resolve() == source / 'benchmarks/archive_evidence.py'
    old_path = BASE / 'pending/q1_stock_validate.py'
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == OLD_PIN
    old = load(old_path, 'q1_original_postvalidation_regression')
    new = load(BASE / 'postvalidation_v2/q1_stock_validate.py', 'q1_v2_postvalidation_regression')
    run = load(BASE / 'pending/q1_stock_run.py', 'q1_run_postvalidation_regression')
    yield types.SimpleNamespace(strict=strictness, evidence=archive_evidence, old=old, new=new, run=run)
    sys.path.remove(str(source))
    for relative, pin in pins.items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == pin


@pytest.fixture
def make_case(tmp_path, modules):
    def create(*, correct=True, failed=False, explicit_flag=False, row_failure=None,
               summary_degraded=False):
        run = modules.run
        config = {
            'sample': 1, 'seed': run.SEED, 'workers': 1, 'top_k': 15,
            'auto_ability': True, 'permissive_default': True, 'no_dream': False,
            'embeddings': False, 'aggregation_nodes': False, 'episode_granularity': False,
            'retrieval_only': False, 'distill': False, 'indexing_require_healthy': True,
            'source_order_validated': True, 'dataset_expected_count': 500,
            'scored_run': True, 'label_free_answer_path': True, 'prereg': None,
            'indexing_max_cycles': 100, 'indexing_timeout_s': 3600.0,
            'judge_protocol': 'legacy-custom', 'hymem_thinking': 'disabled', 'scales': 'S',
        }
        identities = {}
        for role in ('reader', 'judge', 'memory_pipeline'):
            identities[role] = {
                'model': run.MODEL, 'endpoint_origin': run.ENDPOINT,
                'endpoint_sha256': 'sha256:' + hashlib.sha256(run.ENDPOINT.encode()).hexdigest(),
                ('effective_extra_body' if role == 'memory_pipeline' else 'extra_body'):
                    {'thinking': {'type': 'disabled'}},
            }
        manifest = modules.strict.build_manifest(
            benchmark='LongMemEval', code_sha256='sha256:' + 'c' * 64,
            data_sha256='sha256:' + run.DATASET_SHA, config=config, models=identities,
            seed=run.SEED, expected_ids=[run.QUESTION_ID], protocol_split='full',
        )
        meter = modules.strict.usage_snapshot(types.SimpleNamespace(
            call_count=1, request_attempt_count=1, successful_response_count=1,
            prompt_tokens=3, completion_tokens=2, total_tokens=5,
        ))
        # Use the provider-independent fields that the existing postvalidator
        # reads. This fixture tests physical evidence, not a fake strict archive.
        meter.update(calls=1, calls_available=True, request_attempts=1,
                     request_attempts_available=True, successful_responses=1,
                     successful_responses_available=True)
        segment = {'segment_id': 'synthetic-segment', 'status': 'complete', 'attempted_attempts': 1,
                   **{role + '_usage': deepcopy(meter)
                      for role in ('reader', 'judge', 'retrieval', 'memory_pipeline')}}
        row = {
            'question_id': run.QUESTION_ID, 'answer': 'Invented answer, not benchmark text.',
            'correct': correct, 'score': 1,
            'indexing': {'complete': True, 'healthy': True, 'outcome': 'success',
                         'summary_healthy': True},
        }
        if summary_degraded:
            row['indexing'].update(outcome='success_with_summary_degradation', summary_healthy=False)
        if explicit_flag:
            row['strict_failure'] = False
        if row_failure:
            row['benchmark_failure'] = row_failure
        path = tmp_path / ('failed' if failed else str(correct))
        path.mkdir()
        checkpoint = path / 'checkpoint.json'
        # Real stock APIs emit the unequal raw and reconciled representations.
        with modules.strict.AtomicCheckpoint(checkpoint, manifest=manifest,
                                             expected_ids=[run.QUESTION_ID]) as ledger:
            if failed:
                ledger.record(run.QUESTION_ID, row=None, failure='synthetic_failure',
                              execution_segment=segment)
            else:
                ledger.record(run.QUESTION_ID, row=row, execution_segment=segment)
            snapshot = ledger.finalize()
            rows = list(ledger.reconcile().rows)
        assert json.loads(checkpoint.read_text()) == snapshot
        data = {'config': config, 'models': identities, 'manifest': manifest,
                'execution': {'segments': snapshot['execution_segments'],
                              'checkpoint': modules.evidence.checkpoint_attestation(snapshot, rows)}}
        validated = {'rows': rows, 'counts': snapshot['counts'], 'run_id': manifest['run_id'],
                     'scores': {'overall': int(bool(correct) and not failed)}, 'elapsed_s': 1,
                     'summary_degraded_questions': int(summary_degraded),
                     'summary_degraded_sessions': 2 if summary_degraded else 0}
        supervisor = {'manifest_sha256': 'a' * 64, 'source_and_dataset_unchanged': True,
                      'exception_type': None, 'postrun_exception_type': None,
                      'outcome': {'status': 'completed', 'returncode': 0, 'safe_to_continue': True,
                                  'cleanup_complete': True, 'child_reaped': True,
                                  'group_absent_after_reap': True, 'terminal_receipt_written': True,
                                  'errors': [], 'cleanup_warnings': []}}
        return {'data': data, 'validated': validated, 'snapshot': snapshot, 'supervisor': supervisor,
                'manifest_sha': 'a' * 64, 'archive_sha': 'b' * 64,
                'checkpoint_sha': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'checkpoint_attestation': modules.evidence.checkpoint_attestation, 'run': run}
    return create


def evaluate(modules, case):
    return modules.new.summarize(**case, reconcile_results=modules.strict.reconcile_results,
                                bounded_failure_text=modules.strict.bounded_failure_text)


@pytest.mark.parametrize('correct', [True, False])
def test_real_writer_exposes_old_false_rejection_and_v2_accepts(modules, make_case, correct):
    case = make_case(correct=correct)
    before = modules.new.canonical(case['snapshot'])
    raw = case['snapshot']['entries'][modules.run.QUESTION_ID]['row']
    assert 'strict_failure' not in raw and case['validated']['rows'][0]['strict_failure'] is False
    with pytest.raises(RuntimeError, match='postvalidation_physical_checkpoint'):
        modules.old.summarize(**case)
    result = evaluate(modules, case)
    assert result['schema'] == 'stock-q1-postvalidation-v2'
    assert result['benchmark_completed_without_faults'] is True
    assert result['answer_correct'] is correct
    assert result['new_provider_calls'] == 0 and result['full_500_readiness_verified'] is False
    assert modules.new.canonical(case['snapshot']) == before


def test_valid_failed_row_remains_failure_not_false_readiness(modules, make_case):
    result = evaluate(modules, make_case(failed=True))
    assert result['status'] == 'validated_q1_has_failures'
    assert result['benchmark_completed_without_faults'] is False
    assert result['counts']['failed'] == 1 and result['per_role_usage']['reader']['calls'] == 1


def test_consistent_preexisting_derived_flag_is_accepted(modules, make_case):
    assert evaluate(modules, make_case(explicit_flag=True))['benchmark_completed_without_faults'] is True


def test_real_writer_summary_degradation_preserves_healthy_item_completion(modules, make_case):
    result = evaluate(modules, make_case(summary_degraded=True))
    assert result['benchmark_completed_without_faults'] is True
    assert result['item_indexing_healthy'] is True and result['summary_healthy'] is False
    assert result['indexing_outcome'] == 'success_with_summary_degradation'
    assert result['summary_degraded_questions'] == 1 and result['summary_degraded_sessions'] == 2


@pytest.mark.parametrize('correct,row_failure', [(True, 'synthetic_failure'),
                                               (None, None), (None, 'synthetic_failure')])
def test_real_failed_row_verdict_derivation_never_claims_readiness(modules, make_case, correct, row_failure):
    case = make_case(correct=correct, row_failure=row_failure)
    entry = case['snapshot']['entries'][modules.run.QUESTION_ID]
    assert entry['status'] == 'failed' and entry['row']['correct'] is correct
    assert case['validated']['rows'][0]['correct'] is False
    result = evaluate(modules, case)
    assert result['status'] == 'validated_q1_has_failures'
    assert result['benchmark_completed_without_faults'] is False


@pytest.mark.parametrize('mutation', [
    'answer', 'correct', 'indexing', 'score', 'extra_field', 'missing_field', 'id_whitespace',
    'correct_integer', 'indexing_integer', 'score_float', 'flag_conflict', 'flag_integer',
])
def test_every_changed_raw_field_still_rejected_with_matching_history(modules, make_case, mutation):
    case = make_case()
    entry = case['snapshot']['entries'][modules.run.QUESTION_ID]
    raw = entry['row']
    if mutation == 'answer': raw['answer'] = 'Different invented answer.'
    elif mutation == 'correct': raw['correct'] = False
    elif mutation == 'indexing': raw['indexing']['outcome'] = 'not_success'
    elif mutation == 'score': raw['score'] = 2
    elif mutation == 'extra_field': raw['unarchived_extra'] = True
    elif mutation == 'missing_field': del raw['answer']
    elif mutation == 'id_whitespace': raw['question_id'] += ' '
    elif mutation == 'correct_integer': raw['correct'] = 1
    elif mutation == 'indexing_integer': raw['indexing']['healthy'] = 1
    elif mutation == 'score_float': raw['score'] = 1.0
    elif mutation == 'flag_conflict': raw['strict_failure'] = True
    elif mutation == 'flag_integer': raw['strict_failure'] = 0
    entry['attempt_history'][0]['row'] = deepcopy(raw)
    with pytest.raises(RuntimeError, match='postvalidation_physical_'):
        evaluate(modules, case)


@pytest.mark.parametrize('mutation', [
    'snapshot_status', 'entry_status', 'attempt_count', 'attempt_bool', 'history_empty',
    'history_attempt', 'history_attempt_bool', 'history_status', 'history_failure',
    'history_answer', 'history_extra', 'entry_extra', 'entry_missing', 'extra_entry',
    'scored', 'verdict_key', 'archive_flag_integer',
])
def test_status_attempt_history_and_archive_marker_tampering_rejected(modules, make_case, mutation):
    case = make_case()
    snapshot = case['snapshot']
    entry = snapshot['entries'][modules.run.QUESTION_ID]
    event = entry['attempt_history'][0]
    if mutation == 'snapshot_status': snapshot['status'] = 'running'
    elif mutation == 'entry_status': entry['status'] = 'failed'
    elif mutation == 'attempt_count': entry['attempts'] = 2
    elif mutation == 'attempt_bool': entry['attempts'] = True
    elif mutation == 'history_empty': entry['attempt_history'] = []
    elif mutation == 'history_attempt': event['attempt'] = 2
    elif mutation == 'history_attempt_bool': event['attempt'] = True
    elif mutation == 'history_status': event['status'] = 'failed'
    elif mutation == 'history_failure': event['failure'] = 'invented_failure'
    elif mutation == 'history_answer': event['row']['answer'] = 'Altered history.'
    elif mutation == 'history_extra': event['unexpected'] = True
    elif mutation == 'entry_extra': entry['unexpected'] = True
    elif mutation == 'entry_missing': del entry['row']
    elif mutation == 'extra_entry': snapshot['entries']['unexpected'] = deepcopy(entry)
    elif mutation == 'scored': snapshot['scored'] = 1
    elif mutation == 'verdict_key': snapshot['verdict_key'] = 'other'
    elif mutation == 'archive_flag_integer': case['validated']['rows'][0]['strict_failure'] = 0
    with pytest.raises(RuntimeError, match='postvalidation_physical_'):
        evaluate(modules, case)


@pytest.mark.parametrize('mutation', ['recipe', 'attestation', 'cleanup', 'reader_usage', 'indexing_health'])
def test_existing_nonphysical_gates_are_preserved(modules, make_case, mutation):
    case = make_case()
    if mutation == 'recipe':
        case['data']['config']['indexing_require_healthy'] = False
        with pytest.raises(RuntimeError, match='postvalidation_recipe'):
            evaluate(modules, case)
        return
    if mutation == 'attestation':
        case['data']['execution']['checkpoint']['state_sha256'] = 'sha256:' + '0' * 64
        with pytest.raises(RuntimeError, match='postvalidation_checkpoint_binding'):
            evaluate(modules, case)
        return
    if mutation == 'cleanup': case['supervisor']['outcome']['cleanup_complete'] = False
    elif mutation == 'reader_usage':
        case['data']['execution']['segments'][0]['reader_usage']['calls_available'] = False
        case['data']['execution']['checkpoint'] = modules.evidence.checkpoint_attestation(
            case['snapshot'], case['validated']['rows'])
    elif mutation == 'indexing_health':
        # A consistently archived unhealthy row remains an explicit failed gate.
        row = case['validated']['rows'][0]
        row['indexing']['healthy'] = False
        raw = case['snapshot']['entries'][modules.run.QUESTION_ID]['row']
        raw['indexing']['healthy'] = False
        case['snapshot']['entries'][modules.run.QUESTION_ID]['attempt_history'][0]['row'] = deepcopy(raw)
        case['data']['execution']['checkpoint'] = modules.evidence.checkpoint_attestation(case['snapshot'], [row])
    assert evaluate(modules, case)['benchmark_completed_without_faults'] is False
