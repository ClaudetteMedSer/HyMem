"""Eight-row verifier controls against genuine frozen R5 checkpoint APIs.

Run with HYMEM_Q1_VERIFIER_SOURCE set to the frozen R5 tree. Source IDs are
public metadata; all row contents and outputs below are invented, offline data.
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
HELPER = ROOT / 'tools/diagnostics/lme_sample8_v1/bundle/q1_stock_validate.py'
R5_MANIFEST = ROOT / 'docs/patches/2026-09-24-lme-independent-summary-indexing-r5-manifest.json'
R5_PIN = 'c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'
OLD_HELPER = ROOT / 'tools/diagnostics/lme_stock_q1/postvalidation_v2/q1_stock_validate.py'
OLD_PIN = 'e322d4b0c9fd3f0c455d409186b4788a9287f076f27bb8729a65bb4d0c0bc9cb'


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
    assert hashlib.sha256(OLD_HELPER.read_bytes()).hexdigest() == OLD_PIN
    sys.path.insert(0, str(source))
    from benchmarks import strictness, archive_evidence
    assert Path(strictness.__file__).resolve() == source / 'benchmarks/strictness.py'
    assert Path(archive_evidence.__file__).resolve() == source / 'benchmarks/archive_evidence.py'
    verifier = types.ModuleType('sample8_verifier_test')
    verifier.__file__ = str(HELPER)
    exec(compile(HELPER.read_bytes(), str(HELPER), 'exec'), verifier.__dict__)
    run = types.SimpleNamespace(SAMPLE=8, SEED=0, SOURCE_INDICES=list(verifier.SOURCE_INDICES),
                                MODEL='deepseek-flash', ENDPOINT='https://api.deepseek.com',
                                DATASET_SHA='d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442')
    yield types.SimpleNamespace(v=verifier, strict=strictness, evidence=archive_evidence, run=run)
    sys.path.remove(str(source))
    for relative, pin in pins.items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == pin
    assert hashlib.sha256(OLD_HELPER.read_bytes()).hexdigest() == OLD_PIN


@pytest.fixture
def make_case(tmp_path, modules):
    def create(*, wrong_index=None, failed_index=None, degraded_index=None, correct_override=True,
               failure_override=None, explicit_flag=False):
        run, v = modules.run, modules.v
        ids = list(v.QUESTION_IDS)
        package = {'schema': 'stock-lme-sample8-development-run-preparation-v1',
                   'sample': 8, 'seed': 0, 'source_indices': list(v.SOURCE_INDICES), 'question_ids': ids}
        config = {
            'sample': 8, 'seed': 0, 'workers': 1, 'top_k': 15,
            'auto_ability': True, 'permissive_default': True, 'no_dream': False,
            'embeddings': False, 'aggregation_nodes': False, 'episode_granularity': False,
            'retrieval_only': False, 'distill': False, 'indexing_require_healthy': True,
            'source_order_validated': True, 'dataset_expected_count': 500,
            'scored_run': True, 'label_free_answer_path': True, 'prereg': None,
            'indexing_max_cycles': 100, 'indexing_timeout_s': 3600.0,
            'judge_protocol': 'legacy-custom', 'hymem_thinking': 'disabled', 'scales': 'S',
        }
        models = {role: {
            'model': run.MODEL, 'endpoint_origin': run.ENDPOINT,
            'endpoint_sha256': 'sha256:' + hashlib.sha256(run.ENDPOINT.encode()).hexdigest(),
            ('effective_extra_body' if role == 'memory_pipeline' else 'extra_body'):
                {'thinking': {'type': 'disabled'}},
        } for role in ('reader', 'judge', 'memory_pipeline')}
        manifest = modules.strict.build_manifest(
            benchmark='LongMemEval', code_sha256='sha256:' + 'c' * 64,
            data_sha256='sha256:' + run.DATASET_SHA, config=config, models=models,
            seed=run.SEED, expected_ids=ids, protocol_split='full')
        meter = modules.strict.usage_snapshot(types.SimpleNamespace(
            call_count=8, request_attempts=8, successful_responses=8,
            prompt_tokens=24, completion_tokens=16, total_tokens=40))
        segment = {'segment_id': 'synthetic-segment', 'status': 'complete', 'attempted_attempts': 8,
                   **{role + '_usage': deepcopy(meter)
                      for role in ('reader', 'judge', 'retrieval', 'memory_pipeline')},
                   'extraction_canary': {'status': 'passed', 'client_closed': True, 'usage': deepcopy(meter)}}
        checkpoint = tmp_path / 'checkpoint.json'
        with modules.strict.AtomicCheckpoint(checkpoint, manifest=manifest, expected_ids=ids) as ledger:
            for i, qid in enumerate(ids):
                healthy_summary = i != degraded_index
                row = {'question_id': qid, 'answer': f'Invented answer {i}; not benchmark content.',
                       'correct': False if i == wrong_index else correct_override, 'score': 1,
                       'indexing': {'complete': True, 'healthy': True,
                                    'outcome': 'success' if healthy_summary else 'success_with_summary_degradation',
                                    'summary_healthy': healthy_summary,
                                    'final_status': {'summary_health': {
                                        'summary_degraded_sessions': 0 if healthy_summary else 2,
                                        'summary_missing_sessions': 0 if healthy_summary else 1,
                                        'malformed_summaries': 0}}}}
                if explicit_flag: row['strict_failure'] = False
                if failure_override: row['benchmark_failure'] = failure_override
                if i == failed_index:
                    ledger.record(qid, row=None, failure='invented_failure', execution_segment=segment)
                else:
                    ledger.record(qid, row=row, execution_segment=segment)
            snapshot = ledger.finalize()
            rows = list(ledger.reconcile().rows)
        assert json.loads(checkpoint.read_text()) == snapshot
        data = {'config': config, 'models': models, 'manifest': manifest,
                'execution': {'segments': snapshot['execution_segments'],
                              'checkpoint': modules.evidence.checkpoint_attestation(snapshot, rows)}}
        validated = {'rows': rows, 'counts': deepcopy(snapshot['counts']), 'run_id': manifest['run_id'],
                     'scores': {'OVERALL': {'count': 8, 'accuracy': 100.0}}, 'elapsed_s': 1.0,
                     'summary_degraded_questions': int(degraded_index is not None),
                     'summary_degraded_sessions': 2 if degraded_index is not None else 0}
        supervisor = {'manifest_sha256': 'a' * 64, 'source_and_dataset_unchanged': True,
                      'exception_type': None, 'postrun_exception_type': None,
                      'outcome': {'status': 'completed', 'returncode': 0, 'safe_to_continue': True,
                                  'cleanup_complete': True, 'child_reaped': True,
                                  'group_absent_after_reap': True, 'terminal_receipt_written': True,
                                  'errors': [], 'cleanup_warnings': []}}
        return dict(data=data, validated=validated, snapshot=snapshot, supervisor=supervisor,
                    manifest=package, manifest_sha='a' * 64, archive_sha='b' * 64,
                    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    checkpoint_attestation=modules.evidence.checkpoint_attestation,
                    reconcile_results=modules.strict.reconcile_results,
                    bounded_failure_text=modules.strict.bounded_failure_text,
                    aggregate_usage_snapshots=modules.strict.aggregate_usage_snapshots, run=run)
    return create


def reattest(case, modules):
    case['data']['execution']['checkpoint'] = modules.evidence.checkpoint_attestation(
        case['snapshot'], case['validated']['rows'])


@pytest.mark.parametrize('wrong_index', [None, 0, 7])
def test_eight_real_checkpoint_rows_accept_wrong_answers_without_infrastructure_failure(modules, make_case, wrong_index):
    case = make_case(wrong_index=wrong_index)
    before = modules.v.canonical(case['snapshot'])
    assert all('strict_failure' not in e['row'] for e in case['snapshot']['entries'].values())
    result = modules.v.summarize(**case)
    assert result['benchmark_completed_without_faults'] is True
    assert result['status'] == 'scored_sample8_completed' and len(result['per_question']) == 8
    assert result['question_ids'] == modules.v.QUESTION_IDS
    assert [r['answer_correct'] for r in result['per_question']] == [i != wrong_index for i in range(8)]
    assert result['aggregate_paid_usage']['calls'] == 40  # Four roles plus separate canary.
    assert result['aggregate_paid_usage']['total_tokens'] == 200
    assert result['aggregate_paid_usage']['cost_usd'] is None
    assert result['new_provider_calls'] == 0 and result['full_500_readiness_verified'] is False
    assert 'Invented answer' not in json.dumps(result)
    assert modules.v.canonical(case['snapshot']) == before


def test_degradation_is_visible_and_does_not_waive_item_health(modules, make_case):
    case = make_case(degraded_index=7)
    result = modules.v.summarize(**case)
    assert result['benchmark_completed_without_faults'] is True
    last = result['per_question'][-1]
    assert last['item_indexing_healthy'] is True and last['summary_healthy'] is False
    assert last['summary_degraded_sessions'] == 2 and last['summary_missing_sessions'] == 1
    assert result['summary_degraded_questions'] == 1 and result['summary_degraded_sessions'] == 2


@pytest.mark.parametrize('failed_index', [0, 4, 7])
def test_failed_question_never_claims_clean_completion(modules, make_case, failed_index):
    result = modules.v.summarize(**make_case(failed_index=failed_index))
    assert result['benchmark_completed_without_faults'] is False
    assert result['status'] == 'validated_sample8_has_failures'
    assert result['counts']['failed'] == 1
    assert result['per_question'][failed_index]['strict_failure'] is True


@pytest.mark.parametrize('correct,failure', [(True, 'synthetic'), (None, None), (None, 'synthetic')])
def test_real_failed_verdict_normalization_remains_failure(modules, make_case, correct, failure):
    result = modules.v.summarize(**make_case(correct_override=correct, failure_override=failure))
    assert result['benchmark_completed_without_faults'] is False and result['counts']['failed'] == 8


def test_consistent_preexisting_derived_flags_are_valid(modules, make_case):
    assert modules.v.summarize(**make_case(explicit_flag=True))['benchmark_completed_without_faults']


@pytest.mark.parametrize('index', [0, 3, 7])
@pytest.mark.parametrize('mutation', ['answer', 'correct', 'indexing', 'score_type', 'extra', 'missing', 'qid', 'flag', 'flag_type'])
def test_every_physical_row_is_bound_even_when_history_matches(modules, make_case, index, mutation):
    case = make_case()
    raw = case['snapshot']['entries'][modules.v.QUESTION_IDS[index]]['row']
    if mutation == 'answer': raw['answer'] = 'Different invented answer'
    elif mutation == 'correct': raw['correct'] = False
    elif mutation == 'indexing': raw['indexing']['healthy'] = 1
    elif mutation == 'score_type': raw['score'] = 1.0
    elif mutation == 'extra': raw['unarchived'] = True
    elif mutation == 'missing': del raw['answer']
    elif mutation == 'qid': raw['question_id'] += ' '
    elif mutation == 'flag': raw['strict_failure'] = True
    elif mutation == 'flag_type': raw['strict_failure'] = 0
    case['snapshot']['entries'][modules.v.QUESTION_IDS[index]]['attempt_history'][0]['row'] = deepcopy(raw)
    with pytest.raises(RuntimeError, match='postvalidation_physical_'):
        modules.v.summarize(**case)


@pytest.mark.parametrize('mutation', ['attempt', 'attempt_bool', 'history_empty', 'history_attempt',
    'history_attempt_bool', 'history_row', 'history_status', 'history_extra', 'entry_extra',
    'extra_entry', 'missing_entry', 'swapped_rows', 'duplicate_row', 'missing_row', 'extra_row',
    'order', 'scored', 'verdict_key', 'snapshot_status', 'archive_flag_type'])
def test_ledger_history_inventory_and_order_tampering_fails(modules, make_case, mutation):
    case = make_case()
    snapshot = case['snapshot']; entry = snapshot['entries'][modules.v.QUESTION_IDS[-1]]
    history = entry['attempt_history'][0]; rows = case['validated']['rows']
    if mutation == 'attempt': entry['attempts'] = 2
    elif mutation == 'attempt_bool': entry['attempts'] = True
    elif mutation == 'history_empty': entry['attempt_history'] = []
    elif mutation == 'history_attempt': history['attempt'] = 2
    elif mutation == 'history_attempt_bool': history['attempt'] = True
    elif mutation == 'history_row': history['row']['answer'] = 'Different history'
    elif mutation == 'history_status': history['status'] = 'failed'
    elif mutation == 'history_extra': history['other'] = True
    elif mutation == 'entry_extra': entry['other'] = True
    elif mutation == 'extra_entry': snapshot['entries']['extra'] = deepcopy(entry)
    elif mutation == 'missing_entry': del snapshot['entries'][modules.v.QUESTION_IDS[0]]
    elif mutation == 'swapped_rows': rows[0], rows[1] = rows[1], rows[0]
    elif mutation == 'duplicate_row': rows[-1] = deepcopy(rows[0])
    elif mutation == 'missing_row': rows.pop()
    elif mutation == 'extra_row': rows.append(deepcopy(rows[0]))
    elif mutation == 'order': snapshot['expected_ids'].reverse()
    elif mutation == 'scored': snapshot['scored'] = 1
    elif mutation == 'verdict_key': snapshot['verdict_key'] = 'other'
    elif mutation == 'snapshot_status': snapshot['status'] = 'running'
    elif mutation == 'archive_flag_type': rows[-1]['strict_failure'] = 0
    with pytest.raises(RuntimeError, match='postvalidation_(physical_|question)'):
        modules.v.summarize(**case)


@pytest.mark.parametrize('mutation', ['sample', 'seed', 'source_indices', 'ids', 'ids_duplicate', 'ids_order',
    'recipe', 'recipe_bool', 'model', 'thinking', 'attestation', 'counts', 'count_bool', 'segments', 'segment_attempt_bool'])
def test_run_identity_recipe_counts_and_attestation_fences(modules, make_case, mutation):
    case = make_case(); package = case['manifest']; data = case['data']
    if mutation == 'sample': package['sample'] = 1
    elif mutation == 'seed': package['seed'] = False
    elif mutation == 'source_indices': package['source_indices'][0] += 1
    elif mutation == 'ids': package['question_ids'][-1] = 'unexpected'
    elif mutation == 'ids_duplicate': package['question_ids'][-1] = package['question_ids'][0]
    elif mutation == 'ids_order': package['question_ids'].reverse()
    elif mutation == 'recipe': data['config']['indexing_require_healthy'] = False
    elif mutation == 'recipe_bool': data['config']['workers'] = True
    elif mutation == 'model': data['models']['reader']['model'] = 'other'
    elif mutation == 'thinking': data['models']['judge']['extra_body'] = {}
    elif mutation == 'attestation': data['execution']['checkpoint']['state_sha256'] = 'sha256:' + '0' * 64
    elif mutation == 'counts': case['validated']['counts']['completed'] = 7
    elif mutation == 'count_bool': case['validated']['counts']['failed'] = False
    elif mutation == 'segments': data['execution']['segments'].append(deepcopy(data['execution']['segments'][0]))
    elif mutation == 'segment_attempt_bool': data['execution']['segments'][0]['attempted_attempts'] = True
    with pytest.raises(RuntimeError, match='postvalidation_'):
        modules.v.summarize(**case)


@pytest.mark.parametrize('mutation', ['cleanup', 'returncode', 'reader_usage', 'judge_usage', 'instrumentation', 'indexing'])
def test_bad_nonphysical_evidence_never_reports_clean(modules, make_case, mutation):
    case = make_case(); segment = case['data']['execution']['segments'][0]
    if mutation == 'cleanup': case['supervisor']['outcome']['cleanup_complete'] = False
    elif mutation == 'returncode': case['supervisor']['outcome']['returncode'] = False
    elif mutation in ('reader_usage', 'judge_usage'): segment[mutation]['calls'] = 7
    elif mutation == 'instrumentation': segment['instrumentation_errors'] = ['private_detail']
    elif mutation == 'indexing':
        qid = modules.v.QUESTION_IDS[-1]
        entry = case['snapshot']['entries'][qid]
        entry['row']['indexing']['healthy'] = False
        entry['attempt_history'][0]['row'] = deepcopy(entry['row'])
        case['validated']['rows'][-1]['indexing']['healthy'] = False
    reattest(case, modules)
    assert modules.v.summarize(**case)['benchmark_completed_without_faults'] is False


def test_unknown_canary_tokens_and_cost_remain_unknown_in_total(modules, make_case):
    case = make_case()
    usage = case['data']['execution']['segments'][0]['extraction_canary']['usage']
    usage.update(prompt_tokens=None, completion_tokens=None, total_tokens=None, token_usage_available=False)
    reattest(case, modules)
    result = modules.v.summarize(**case)
    assert result['aggregate_paid_usage']['calls'] == 40
    assert result['aggregate_paid_usage']['token_usage_available'] is False
    assert result['aggregate_paid_usage']['total_tokens'] is None
    assert result['aggregate_paid_usage']['cost_usd'] is None


@pytest.mark.parametrize('mutation', ['status', 'closed', 'text', 'nan'])
def test_untrusted_canary_or_usage_cannot_leak_or_pass(modules, make_case, mutation):
    case = make_case(); canary = case['data']['execution']['segments'][0]['extraction_canary']
    if mutation == 'status': canary['status'] = 'failed'
    elif mutation == 'closed': canary['client_closed'] = False
    elif mutation == 'text': canary['usage']['cost_usd'] = 'PRIVATE TEXT'
    elif mutation == 'nan': canary['usage']['cost_usd'] = float('nan')
    # Attestation independently refuses nonfinite data before summarize.
    if mutation == 'nan':
        with pytest.raises((ValueError, RuntimeError)):
            reattest(case, modules)
        return
    reattest(case, modules)
    with pytest.raises(RuntimeError, match='postvalidation_'):
        modules.v.summarize(**case)


@pytest.mark.parametrize('degraded,wrong_last', [(False, False), (True, True)])
def test_actual_strict_archive_validator_to_eight_row_postvalidator(
    modules, make_case, tmp_path, degraded, wrong_last,
):
    """Complete real validator path; only content/provider execution is invented.

    No validators, checkpoint writers, attesters or reconciliation functions are
    mocked. Published fixture metadata uses the predeclared pinned dataset
    identity solely to exercise the exact recipe, never to claim a real run.
    """
    from tests import test_lme_protocol_hardening as fixtures
    from tests.archive_evidence_fixtures import healthy_convergence
    from benchmarks import lme_protocol as protocol
    from hymem.contrib.endpoint_policy import secret_free_endpoint_identity
    from hymem.contrib.openai_client import openai_compatible_producer_declaration
    from hymem.extraction.producer import producer_binding_from_typed_declaration

    case = make_case()
    run, ids = modules.run, modules.v.QUESTION_IDS
    artifact = fixtures.make_artifact()
    config, models = artifact['config'], artifact['models']
    config.update(case['data']['config'])
    config.update(
        dataset_revision=protocol.LME_S_DATASET_REVISION,
        dataset_url=protocol.LME_S_DATASET_URL,
        dataset_sha256=protocol.LME_S_DATASET_SHA256,
        source_ids_hash=protocol.LME_S_SOURCE_IDS_HASH,
        source_qtype_counts=deepcopy(protocol.LME_S_QTYPE_COUNTS),
    )
    endpoint = secret_free_endpoint_identity(run.ENDPOINT, label='invented fixture')
    disabled = {'thinking': {'type': 'disabled'}}
    for role, prefix in [('reader', 'answer'), ('judge', 'judge'), ('memory_pipeline', 'hymem')]:
        models[role].update(provider='deepseek', model=run.MODEL, **endpoint)
        config.update({prefix + '_model': run.MODEL,
                       **{prefix + '_' + key: value for key, value in endpoint.items()}})
        if role != 'memory_pipeline':
            models[role]['extra_body'] = deepcopy(disabled)
            config[prefix + '_extra_body_obj'] = deepcopy(disabled)
    pipeline = models['memory_pipeline']
    pipeline.update(thinking_mode='disabled', effective_extra_body=deepcopy(disabled),
                    deployment_revision_sha256=None, deployment_tenant_sha256=None)
    pipeline['aggregation_producer'] = producer_binding_from_typed_declaration(
        openai_compatible_producer_declaration(
            model=run.MODEL, endpoint=run.ENDPOINT, thinking_mode='disabled',
            effective_extra_body=disabled, transport_package_version=pipeline['transport_package_version'],
            request_timeout_seconds=pipeline['request_timeout_seconds'],
            deployment_revision_sha256=None, deployment_tenant_sha256=None,
            require_consistent_thinking=True), declaration_hook='aggregation_producer_declaration')
    rows, indexing_runs = [], []
    prototype = artifact['per_question'][0]
    for i, qid in enumerate(ids):
        row = deepcopy(prototype)
        row.update(question_id=qid, question=f'Invented integration question {i}',
                   answer='Invented reference', hypothesis='Invented hypothesis',
                   correct=not (wrong_last and i == 7),
                   judge_raw='no' if wrong_last and i == 7 else 'yes')
        raw_summary = healthy_convergence(config)
        if degraded and i == 7:
            raw_summary.update(summary_healthy=False, outcome='success_with_summary_degradation')
            raw_summary['final_status'].update(
                summary_healthy=False, summary_degraded_sessions=2, summary_missing_sessions=1)
        row['indexing'] = protocol.canonicalize_lme_indexing_summary(raw_summary)
        rows.append(row)
        indexing_runs.append({'question_id': qid, 'summary': deepcopy(row['indexing'])})
    segment = artifact['execution']['segments'][0]
    segment.update(attempted_attempts=8, model_identities=models,
                   reader_usage=fixtures._usage(calls=8, total_tokens=80),
                   judge_usage=fixtures._usage(calls=8, total_tokens=16),
                   memory_pipeline_usage=fixtures._usage(calls=8, total_tokens=160),
                   indexing_runs=indexing_runs, latest_indexing=deepcopy(indexing_runs[-1]),
                   extraction_canary=fixtures._passed_canary(pipeline))
    artifact['manifest'] = modules.strict.build_manifest(
        benchmark='LongMemEval', code_sha256=modules.strict.content_hash('invented fixture code'),
        data_sha256=config['dataset_sha256'], config=config, models=models,
        seed=run.SEED, expected_ids=ids, protocol_split='full')
    checkpoint = tmp_path / 'full-validator.checkpoint.json'
    with modules.strict.AtomicCheckpoint(checkpoint, manifest=artifact['manifest'], expected_ids=ids) as ledger:
        for row in rows:
            ledger.record(row['question_id'], row=row, execution_segment=segment)
        snapshot = ledger.finalize()
        published = list(ledger.reconcile().rows)
    artifact.update(per_question=published, scores=protocol._scores_from_rows(published),
                    result_digest=modules.strict.content_hash(published),
                    abstention_diagnostics=protocol._abstention_from_rows(published),
                    conditional_judged_only={'accuracy': sum(r['correct'] for r in published) / 8, 'count': 8})
    artifact['execution'].update(counts=deepcopy(snapshot['counts']),
                                 segments=deepcopy(snapshot['execution_segments']),
                                 checkpoint=modules.evidence.checkpoint_attestation(snapshot, published))
    archive = tmp_path / 'invented-eight-question-archive.json'
    archive.write_text(json.dumps(artifact))
    loaded = json.loads(archive.read_text())
    # This is the actual frozen validator, with all its checks enabled.
    validated = protocol.validate_strict_artifact(loaded, path=archive, require_scored=True)
    case.update(data=loaded, validated=validated, snapshot=json.loads(checkpoint.read_text()),
                archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest(),
                checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    report = modules.v.summarize(**case)
    assert report['benchmark_completed_without_faults'] is True
    assert report['counts']['completed'] == 8 and report['counts']['failed'] == 0
    assert report['per_question'][-1]['answer_correct'] is not wrong_last
    assert report['summary_degraded_questions'] == int(degraded)
    assert report['summary_degraded_sessions'] == (2 if degraded else 0)
    assert report['aggregate_paid_usage']['calls'] == 32
    assert report['aggregate_paid_usage']['total_tokens'] == 276
    assert 'Invented' not in json.dumps(report)
