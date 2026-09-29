"""Offline R6 regression harness controls; invented fixtures, no provider calls.

Set HYMEM_R6_SOURCE to the exact frozen full source/test tree. Application code
and all historical diagnostic files remain unchanged by these tests.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
NEW = ROOT / 'tools/diagnostics/lme_r6_regression_v1'
FROZEN = Path(os.environ.get('HYMEM_R6_SOURCE', '/private/tmp/hymem-r6-sequential-20260925.6dSIIq/frozen-r6-final')).resolve()
MANIFEST = ROOT / 'docs/patches/2026-09-25-lme-r6-final-frozen-manifest.json'
OBSOLETE_C1_PIN = 'c1f9dc072f408ee0a64d7ac6d688d96de0828a700309fdbde30a66dccb2a98bb'


def load(name):
    path = NEW / name
    module = types.ModuleType('r6_test_' + name.replace('/', '_').replace('.', '_'))
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('network forbidden in offline regression controls')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'getnameinfo'):
        monkeypatch.setattr(socket, name, forbidden)


@pytest.fixture(scope='module')
def package():
    prepare = load('prepare.py')
    manifest, payload = prepare.build(FROZEN, MANIFEST)
    return manifest, payload


def test_new_seal_contains_only_exact_frozen_sources_and_pinned_support(package):
    manifest, payload = package
    original = json.loads(MANIFEST.read_bytes())
    assert manifest['source_sha256'] == original['source_sha256']
    assert manifest['source_files'] == 231
    assert {p.removeprefix('candidate/') for p in payload if p.startswith('candidate/')} == set(original['source_sha256'])
    assert not any(p.startswith('candidate/tests/') for p in payload)
    assert manifest['canary_version'] == 'hymem-phase1-extraction-canary-v19'
    assert manifest['source_split_policy_version'] == 'hymem-source-semantic-split-v11'
    assert manifest['global_paid_call_cap'] is None
    assert manifest['question_attempts'] == manifest['canary_suites'] == 1
    assert manifest['summary_recovery_in_stock_digest'] is False
    for name, pin in load('prepare.py').SUPPORT.items():
        assert hashlib.sha256(payload['bundle/' + name]).hexdigest() == pin


@pytest.mark.parametrize('boundary', ['sealer', 'installer', 'controller', 'runner'])
def test_real_obsolete_c1_package_is_rejected_at_each_source_boundary(monkeypatch, boundary):
    old_root = FROZEN.parent / 'lme-regression-v1'
    old_manifest = old_root / 'input-manifest.json'
    assert hashlib.sha256(old_manifest.read_bytes()).hexdigest() == OBSOLETE_C1_PIN
    package_pin = hashlib.sha256((old_root / 'bundle/manifest.json').read_bytes()).hexdigest()
    final_pin = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert final_pin != OBSOLETE_C1_PIN
    if boundary == 'sealer':
        module = load('prepare.py')
        assert module.SOURCE_PIN == final_pin
        with pytest.raises(RuntimeError, match='source_manifest_pin'):
            module.build(FROZEN, old_manifest)
    elif boundary == 'installer':
        module = load('install.py')
        assert module.SOURCE_PIN == final_pin
        seal = old_root.parent / (old_root.name + '-seal.json')
        monkeypatch.setattr(sys, 'argv', ['install.py', '--seal', str(seal), '--seal-sha256',
            hashlib.sha256(seal.read_bytes()).hexdigest(), '--receipt', '/unused'])
        monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: pytest.fail('stale source reached SSH'))
        with pytest.raises(RuntimeError, match='seal_identity'):
            module.main()
    elif boundary == 'controller':
        module = load('host_control.py')
        assert module.SOURCE_PIN == final_pin
        monkeypatch.setattr(module, 'ROOT', old_root)
        monkeypatch.setattr(module, 'SOURCE', old_root / 'candidate')
        with pytest.raises(RuntimeError, match='source_manifest_pin'):
            module.definitions(package_pin)
    else:
        module = load('bundle/q1_stock_run.py')
        assert module.SOURCE_MANIFEST_SHA == final_pin
        monkeypatch.setattr(module, 'PACKAGE', old_root / 'bundle')
        with pytest.raises(RuntimeError, match='manifest_source_contract'):
            module.validate_package(package_pin)


def test_installer_transfer_timeout_is_bounded_at_600_seconds(tmp_path, monkeypatch):
    module = load('install.py')
    archive = tmp_path / 'invented.tgz'
    archive.write_bytes(b'invented-transfer-payload')
    value = {'schema': 'r6-lme-failed-question-seal-v1', 'source_manifest_sha256': module.SOURCE_PIN,
             'archive': str(archive), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
             'manifest_sha256': 'a' * 64, 'files_sha256': {}}
    seal = tmp_path / 'invented-seal.json'
    seal.write_text(json.dumps(value))
    monkeypatch.setattr(sys, 'argv', ['install.py', '--seal', str(seal), '--seal-sha256',
        hashlib.sha256(seal.read_bytes()).hexdigest(), '--receipt', str(tmp_path / 'receipt.json')])
    calls = []
    def transfer(argv, **kwargs):
        calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode=0, stdout=b'{"status":"invented-transfer-only"}')
    monkeypatch.setattr(module.subprocess, 'run', transfer)
    module.main()
    assert len(calls) == 1 and calls[0][0][:2] == ['ssh', 'afrodite']
    assert calls[0][1]['timeout'] == 600
    assert calls[0][1]['input'] == b'invented-transfer-payload'


def test_original_stock_selection_and_flags_change_only_sample_count():
    run, pre = load('bundle/q1_stock_run.py'), load('bundle/lme_q1_startup_preflight.py')
    selector = run.selector_from_source(FROZEN / 'benchmarks/longmemeval_adapter.py')
    assert run.verify_selection(selector) == [329]
    old = types.ModuleType('old_sample8_recipe')
    path = ROOT / 'tools/diagnostics/lme_sample8_v1/bundle/q1_stock_run.py'
    exec(compile(path.read_bytes(), str(path), 'exec'), old.__dict__)
    expected = old.stock_arguments()
    expected[expected.index('--sample') + 1] = '1'
    assert run.stock_arguments() == expected
    assert (run.SAMPLE, run.SEED, run.TIMEOUT, run.QUESTION_IDS, run.QUESTION_CENSUS) == (
        1, 0, 5400, ['gpt4_483dd43c'], [(52, 532)])
    pre.request_recipe(run.stock_arguments())


@pytest.mark.parametrize('extra', [['--retry-failures'], ['--resume-from', '/old'],
    ['--skip-extraction-canary'], ['--no-dream'], ['--embeddings'], ['--auto-ability'], ['trailing']])
def test_forbidden_or_duplicate_recipe_refused(extra):
    with pytest.raises(RuntimeError):
        load('bundle/lme_q1_startup_preflight.py').request_recipe(load('bundle/q1_stock_run.py').stock_arguments() + extra)


@pytest.mark.parametrize('flag,value', [('--sample', '8'), ('--seed', '1'), ('--workers', '2'),
    ('--indexing-max-cycles', '101'), ('--indexing-timeout-s', '3601'), ('--judge-model', 'other')])
def test_recipe_drift_refused(flag, value):
    args = load('bundle/q1_stock_run.py').stock_arguments()
    args[args.index(flag) + 1] = value
    with pytest.raises(RuntimeError):
        load('bundle/lme_q1_startup_preflight.py').request_recipe(args)


@pytest.mark.parametrize('fault', ['extra', 'symlink', 'bytes'])
def test_exact_source_inventory_rejects_corruption(tmp_path, fault):
    seal = load('prepare.py')
    tree = tmp_path / 'source'
    tree.mkdir()
    (tree / 'a.py').write_bytes(b'original')
    mapping = {'a.py': hashlib.sha256(b'original').hexdigest()}
    if fault == 'extra': (tree / '.hidden').write_bytes(b'unknown')
    elif fault == 'symlink': (tree / 'link').symlink_to(tree / 'a.py')
    else: (tree / 'a.py').write_bytes(b'drift')
    with pytest.raises(RuntimeError): seal.exact_tree(tree, mapping)


@pytest.mark.parametrize('mode', ['preflight', 'live', 'validation'])
def test_host_plans_have_no_production_mounts_and_never_start(mode):
    host = load('bundle/q1_stock_host.py')
    ctl = load('host_control.py')
    plan = (host.validation_command(str(ctl.ROOT), str(ctl.SOURCE), 'a' * 64) if mode == 'validation'
            else host.command(str(ctl.ROOT), str(ctl.SOURCE), 'a' * 64, live=mode == 'live'))
    assert plan['command'][:2] == ['docker', 'create'] and plan['starts_work'] is False
    assert plan['network'] == ('bridge' if mode == 'live' else 'none')
    assert ('/run/deepseek.env' in plan['mounts']) is (mode == 'live')
    assert all(not writable for target, (_, writable) in plan['mounts'].items()
               if mode == 'validation' or target != '/results')
    assert plan['mounts']['/candidate'] == (str(ctl.SOURCE), False)
    assert '/var/run/docker.sock' not in plan['mounts']


def admission_case(tmp_path, monkeypatch):
    ctl = load('host_control.py')
    monkeypatch.setattr(ctl, 'ROOT', tmp_path)
    (tmp_path / 'gates').mkdir()
    refs = {}
    for i in range(4):
        body = json.dumps({'invented_gate': i}).encode()
        name = 'gates/gate-' + str(i) + '.json'
        (tmp_path / name).write_bytes(body)
        refs[name] = hashlib.sha256(body).hexdigest()
    value = {'schema': 'r6-parent-regression-admission-v1', 'source_manifest_sha256': ctl.SOURCE_PIN,
             'all_three_fixes_parent_accepted': True, 'full_offline_gate_passed': True,
             'semantic_controls_parent_accepted': True, 'production_changes': False, 'receipts': refs}
    return ctl, value


@pytest.mark.parametrize('fault', [None, 'false', 'extra', 'pin', 'traversal', 'symlink', 'missing', 'oldsource', 'obsolete_c1'])
def test_parent_admission_is_pinned_closed_and_binds_all_evidence(tmp_path, monkeypatch, fault):
    ctl, value = admission_case(tmp_path, monkeypatch)
    if fault == 'false': value['all_three_fixes_parent_accepted'] = False
    elif fault == 'extra': value['unexpected'] = 'text'
    elif fault == 'oldsource': value['source_manifest_sha256'] = '0' * 64
    elif fault == 'obsolete_c1': value['source_manifest_sha256'] = OBSOLETE_C1_PIN
    elif fault == 'traversal': value['receipts']['gates/../outside.json'] = '0' * 64
    elif fault == 'missing': value['receipts'].pop('gates/gate-0.json')
    elif fault == 'symlink':
        target = tmp_path / 'gates/gate-0.json'
        target.unlink()
        target.symlink_to(tmp_path / 'gates/gate-1.json')
    (tmp_path / 'admission.json').write_text(json.dumps(value))
    pin = hashlib.sha256((tmp_path / 'admission.json').read_bytes()).hexdigest()
    if fault == 'pin': pin = '0' * 64
    if fault:
        with pytest.raises(RuntimeError): ctl.admission(pin)
    else:
        assert ctl.admission(pin) == value


@pytest.mark.parametrize('fault', [None, 'duplicate', 'symlink', 'archive_pin', 'member_pin', 'existing_root'])
def test_installer_only_accepts_exact_fresh_package(tmp_path, monkeypatch, package, capsys, fault):
    installer = load('install.py')
    manifest, payload = deepcopy(package)
    target = tmp_path / 'fresh-package'
    manifest.update(remote_run_root=str(target), remote_source=str(target / 'candidate'))
    payload['bundle/manifest.json'] = json.dumps(manifest, sort_keys=True).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
        entries = list(payload.items())
        if fault == 'duplicate': entries.append(entries[0])
        for name, body in entries:
            item = tarfile.TarInfo(name)
            item.size = len(body)
            if fault == 'symlink' and name == 'input-manifest.json':
                item.type, item.linkname, item.size = tarfile.SYMTYPE, '/outside', 0
            archive.addfile(item, io.BytesIO(body))
    raw = output.getvalue()
    config = {'root': str(target), 'archive_pin': hashlib.sha256(raw).hexdigest(),
              'manifest_pin': hashlib.sha256(payload['bundle/manifest.json']).hexdigest(),
              'source_pin': installer.SOURCE_PIN,
              'files': {name: hashlib.sha256(body).hexdigest() for name, body in payload.items()}}
    if fault == 'archive_pin': config['archive_pin'] = '0' * 64
    elif fault == 'member_pin': config['files']['candidate/hymem/api.py'] = '0' * 64
    elif fault == 'existing_root': target.mkdir()
    monkeypatch.setattr(sys, 'stdin', types.SimpleNamespace(buffer=io.BytesIO(raw)))
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    if fault:
        with pytest.raises((RuntimeError, FileExistsError)):
            exec(compile(installer.INSTALL, '<reviewed installer>', 'exec'), {'C': config})
        assert not (target / 'install.json').exists()
    else:
        exec(compile(installer.INSTALL, '<reviewed installer>', 'exec'), {'C': config})
        receipt = json.loads(capsys.readouterr().out)
        assert receipt['source_files'] == 231 and receipt['provider_calls'] == 0
        assert receipt['status'] == 'installed_not_started'
        assert not (target / 'candidate/tests').exists()


def test_old_worker_authorization_is_rejected_before_key(monkeypatch):
    run = load('bundle/q1_stock_run.py')
    monkeypatch.setattr(sys, 'stdin', types.SimpleNamespace(buffer=io.BytesIO(run.canonical({'execute_stock_sample8': 'a' * 64}))))
    monkeypatch.setattr(run, 'validate_package', lambda *_: pytest.fail('authorization must precede package/credentials'))
    with pytest.raises(RuntimeError, match='worker_authorization'):
        run.worker('a' * 64)


def test_supervisor_receives_absolute_deadline_and_exclusive_results(tmp_path, monkeypatch):
    run = load('bundle/q1_stock_run.py')
    observed = []
    @dataclass
    class Outcome:
        status: str = 'completed'
        returncode: int = 0
        safe_to_continue: bool = True
    def invoke(argv, **kwargs):
        observed.append((argv, kwargs))
        return Outcome()
    monkeypatch.setattr(run, 'OUTPUT', tmp_path)
    monkeypatch.setattr(run.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(run.time, 'monotonic', lambda: 123.0)
    monkeypatch.setattr(run.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(run, 'validate_package', lambda _: {})
    monkeypatch.setattr(run, 'verify_source', lambda _: None)
    monkeypatch.setattr(run, 'preflight', lambda _: {'api_calls': 0})
    monkeypatch.setattr(run, 'sha', lambda _: run.DATASET_SHA)
    monkeypatch.setattr(run, 'load_helper', lambda _: types.SimpleNamespace(supervise_invocation=invoke))
    assert run.supervise('a' * 64) == 0
    assert len(observed) == 1
    kwargs = observed[0][1]
    assert kwargs['deadline_expires_at'] == 5523.0
    assert kwargs['timeout_seconds'] == 5400 and kwargs['cleanup_seconds'] == 10
    assert kwargs['stdin_bytes'] == run.canonical({'execute_r6_failed_question_v1': 'a' * 64})
    assert json.loads((tmp_path / 'supervisor-summary.json').read_text())['global_paid_call_cap'] is None
    with pytest.raises(FileExistsError): run.supervise('a' * 64)
    assert len(observed) == 1


def test_validation_launch_accepts_failed_live_exit_and_never_restarts_it(tmp_path, monkeypatch):
    ctl, host = load('host_control.py'), load('bundle/q1_stock_host.py')
    monkeypatch.setattr(ctl, 'ROOT', tmp_path)
    monkeypatch.setattr(ctl, 'SOURCE', Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/invented/candidate'))
    # Keep the pure plan's required remote path while redirecting only receipt I/O.
    monkeypatch.setattr(host, 'command', lambda *a, **kw: {'mode': 'live' if kw.get('live') else 'preflight'})
    monkeypatch.setattr(host, 'validation_command', lambda *a: {'mode': 'validation'})
    monkeypatch.setattr(ctl, 'definitions', lambda _: ({}, host))
    live_id, validator_id = 'b' * 64, 'c' * 64
    (tmp_path / 'live-container-id.json').write_text(json.dumps({'container_id': live_id}))
    (tmp_path / 'validation-container-id.json').write_text(json.dumps({'container_id': validator_id}))
    calls = []
    def inspect(cid, plan, pin, expected_state=None):
        if cid == live_id:
            assert expected_state == 'exited'
            return {'pid': 0, 'exit_code': 1, 'status': 'exited'}
        return {'pid': 0, 'status': 'created'}
    monkeypatch.setattr(ctl, 'inspect', inspect)
    monkeypatch.setattr(ctl, 'command', lambda argv: calls.append(argv) or validator_id)
    monkeypatch.setattr(sys, 'argv', ['host_control.py', 'start', 'validation',
        '--manifest-sha256', 'a' * 64, '--container-id', validator_id])
    ctl.main()
    assert calls == [['docker', 'start', validator_id]]
    with pytest.raises(FileExistsError): ctl.main()
    assert calls == [['docker', 'start', validator_id]]


def test_real_stock_startup_is_zero_call_one_id_and_closes_handles(tmp_path):
    data, output = tmp_path / 'data', tmp_path / 'output'
    data.mkdir(); output.mkdir()
    rows = []
    for i in range(500):
        sid = 'invented-session-' + str(i)
        rows.append({'question_id': 'gpt4_483dd43c' if i == 329 else 'invented-' + str(i),
            'question_type': 'single-session-user', 'question': 'What color is the invented kite?',
            'answer': 'blue', 'question_date': '2026/01/02 (Fri) 12:00',
            'haystack_sessions': [[{'role': 'user', 'content': 'My invented kite is blue.'}]],
            'haystack_session_ids': [sid], 'answer_session_ids': [sid],
            'haystack_dates': ['2026/01/01 (Thu) 12:00']})
    (data / 'longmemeval_s_cleaned.json').write_text(json.dumps(rows))
    helper = NEW / 'bundle/lme_q1_startup_preflight.py'
    result = subprocess.run([sys.executable, '-I', '-B', str(helper), '--source', str(FROZEN),
        '--data-dir', str(data), '--output', str(tmp_path / 'probe'),
        '--question-ids-json', '["gpt4_483dd43c"]'],
        env={'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path), 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'passed' and report['expected_question_ids'] == ['gpt4_483dd43c']
    assert report['expected_question_count'] == 1 and report['cli_checkpoint_handles_closed'] is True
    assert report['provider_completions'] == report['provider_http_attempts'] == report['outbound_operations_blocked'] == 0
    assert report['real_credentials_loaded'] is False


@pytest.fixture(scope='module')
def app():
    sys.path.insert(0, str(FROZEN))
    from benchmarks import strictness, archive_evidence, lme_protocol
    from tests import test_lme_protocol_hardening as fixtures
    for module in (strictness, archive_evidence, lme_protocol, fixtures):
        assert Path(module.__file__).resolve().is_relative_to(FROZEN)
    yield types.SimpleNamespace(strict=strictness, evidence=archive_evidence, protocol=lme_protocol, fixtures=fixtures)
    sys.path.remove(str(FROZEN))


def real_case(tmp_path, app, mode):
    """Use the real stock writer, attester, reconciler and strict archive reader."""
    from tests.archive_evidence_fixtures import healthy_convergence
    from hymem.contrib.endpoint_policy import secret_free_endpoint_identity
    from hymem.contrib.openai_client import openai_compatible_producer_declaration
    from hymem.extraction.producer import producer_binding_from_typed_declaration
    run, verifier = load('bundle/q1_stock_run.py'), load('bundle/q1_stock_validate.py')
    failed = mode == 'failed'
    artifact = app.fixtures._indexing_failure_artifact() if failed else app.fixtures.make_artifact()
    config, models = artifact['config'], artifact['models']
    config.update(sample=1, seed=0, workers=1, top_k=15, auto_ability=True, permissive_default=True,
        no_dream=False, embeddings=False, aggregation_nodes=False, episode_granularity=False,
        retrieval_only=False, distill=False, indexing_require_healthy=True, source_order_validated=True,
        dataset_expected_count=500, scored_run=True, label_free_answer_path=True, prereg=None,
        indexing_max_cycles=100, indexing_timeout_s=3600.0, judge_protocol='legacy-custom',
        hymem_thinking='disabled', scales='S', dataset_revision=app.protocol.LME_S_DATASET_REVISION,
        dataset_url=app.protocol.LME_S_DATASET_URL, dataset_sha256=app.protocol.LME_S_DATASET_SHA256,
        source_ids_hash=app.protocol.LME_S_SOURCE_IDS_HASH, source_qtype_counts=deepcopy(app.protocol.LME_S_QTYPE_COUNTS))
    endpoint = secret_free_endpoint_identity(run.ENDPOINT, label='invented fixture')
    disabled = {'thinking': {'type': 'disabled'}}
    for role, prefix in [('reader', 'answer'), ('judge', 'judge'), ('memory_pipeline', 'hymem')]:
        models[role].update(provider='deepseek', model=run.MODEL, **endpoint)
        config.update({prefix + '_model': run.MODEL, **{prefix + '_' + key: value for key, value in endpoint.items()}})
        if role != 'memory_pipeline':
            models[role]['extra_body'] = deepcopy(disabled)
            config[prefix + '_extra_body_obj'] = deepcopy(disabled)
    pipeline = models['memory_pipeline']
    pipeline.update(thinking_mode='disabled', effective_extra_body=deepcopy(disabled),
                    deployment_revision_sha256=None, deployment_tenant_sha256=None)
    pipeline['aggregation_producer'] = producer_binding_from_typed_declaration(
        openai_compatible_producer_declaration(model=run.MODEL, endpoint=run.ENDPOINT,
            thinking_mode='disabled', effective_extra_body=disabled,
            transport_package_version=pipeline['transport_package_version'],
            request_timeout_seconds=pipeline['request_timeout_seconds'],
            deployment_revision_sha256=None, deployment_tenant_sha256=None,
            require_consistent_thinking=True), declaration_hook='aggregation_producer_declaration')
    row = deepcopy(artifact['per_question'][0])
    row['question_id'] = run.QUESTION_IDS[0]
    if not failed:
        row.update(question='Invented integration question', answer='Invented reference',
                   hypothesis='Invented hypothesis', correct=mode != 'wrong', judge_raw='no' if mode == 'wrong' else 'yes')
        summary = healthy_convergence(config)
        if mode == 'degraded':
            summary.update(summary_healthy=False, outcome='success_with_summary_degradation')
            summary['final_status'].update(summary_healthy=False, summary_degraded_sessions=2, summary_missing_sessions=1)
        row['indexing'] = app.protocol.canonicalize_lme_indexing_summary(summary)
    # AtomicCheckpoint must receive the raw row; reconcile supplies strict_failure.
    row.pop('strict_failure', None)
    segment = artifact['execution']['segments'][0]
    indexing = {'question_id': row['question_id'], 'summary': deepcopy(row['indexing'])}
    segment.update(attempted_attempts=1, model_identities=models,
        reader_usage=app.fixtures._usage(calls=0 if failed else 1, total_tokens=0 if failed else 10),
        judge_usage=app.fixtures._usage(calls=0 if failed else 1, total_tokens=0 if failed else 2),
        memory_pipeline_usage=app.fixtures._usage(calls=1, total_tokens=20),
        indexing_runs=[indexing], latest_indexing=deepcopy(indexing),
        extraction_canary=app.fixtures._passed_canary(pipeline))
    artifact['manifest'] = app.strict.build_manifest(benchmark='LongMemEval',
        code_sha256=app.strict.content_hash('invented fixture code'), data_sha256=config['dataset_sha256'],
        config=config, models=models, seed=0, expected_ids=run.QUESTION_IDS, protocol_split='full')
    checkpoint = tmp_path / 'checkpoint.json'
    with app.strict.AtomicCheckpoint(checkpoint, manifest=artifact['manifest'], expected_ids=run.QUESTION_IDS) as ledger:
        ledger.record(row['question_id'], row=row, execution_segment=segment)
        snapshot = ledger.finalize()
        published = list(ledger.reconcile().rows)
    completed = 0 if failed else 1
    artifact.update(per_question=published, scores=app.protocol._scores_from_rows(published),
        result_digest=app.strict.content_hash(published), abstention_diagnostics=app.protocol._abstention_from_rows(published),
        conditional_judged_only={'accuracy': None if failed else float(mode != 'wrong'), 'count': completed})
    artifact['execution'].update(counts=deepcopy(snapshot['counts']), segments=deepcopy(snapshot['execution_segments']),
                                checkpoint=app.evidence.checkpoint_attestation(snapshot, published))
    archive = tmp_path / 'invented-archive.json'
    archive.write_text(json.dumps(artifact))
    validated = app.protocol.validate_strict_artifact(json.loads(archive.read_text()), path=archive, require_scored=True)
    supervisor = {'manifest_sha256': 'a' * 64, 'source_and_dataset_unchanged': True,
        'exception_type': None, 'postrun_exception_type': None,
        'outcome': {'status': 'completed', 'returncode': 0, 'safe_to_continue': True,
                    'cleanup_complete': True, 'child_reaped': True, 'group_absent_after_reap': True,
                    'terminal_receipt_written': True, 'errors': [], 'cleanup_warnings': []}}
    case = dict(data=artifact, validated=validated, snapshot=json.loads(checkpoint.read_text()), supervisor=supervisor,
        manifest={'schema': 'r6-lme-failed-question-preparation-v1', 'sample': 1, 'seed': 0,
                  'source_indices': [329], 'question_ids': run.QUESTION_IDS},
        manifest_sha='a' * 64, archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest(),
        checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_attestation=app.evidence.checkpoint_attestation, reconcile_results=app.strict.reconcile_results,
        bounded_failure_text=app.strict.bounded_failure_text, aggregate_usage_snapshots=app.strict.aggregate_usage_snapshots, run=run)
    return verifier, case


@pytest.mark.parametrize('mode', ['healthy', 'wrong', 'degraded', 'failed'])
def test_actual_strict_archive_checkpoint_and_result_semantics(tmp_path, app, mode):
    verifier, case = real_case(tmp_path, app, mode)
    report = verifier.summarize(**case)
    assert report['strict_scored_artifact_validated'] is True
    assert report['physical_checkpoint_bound'] is True
    assert report['benchmark_completed_without_faults'] is (mode != 'failed')
    assert report['counts']['total_attempts'] == 1
    assert report['counts']['failed'] == int(mode == 'failed')
    assert report['summary_degraded_questions'] == int(mode == 'degraded')
    assert report['full_500_readiness_verified'] is False and report['global_paid_call_cap'] is None
    assert 'Invented' not in json.dumps(report)
    if mode == 'wrong': assert report['per_question'][0]['answer_correct'] is False
    if mode == 'failed': assert report['per_question'][0]['strict_failure'] is True


@pytest.mark.parametrize('fault', ['attempt', 'row', 'history', 'canary_version', 'split_version'])
def test_physical_or_canary_evidence_tampering_cannot_pass(tmp_path, app, fault):
    verifier, case = real_case(tmp_path, app, 'healthy')
    entry = case['snapshot']['entries']['gpt4_483dd43c']
    if fault == 'attempt': entry['attempts'] = 2
    elif fault == 'row': entry['row']['correct'] = False
    elif fault == 'history': entry['attempt_history'][0]['row']['correct'] = False
    else:
        field = 'version' if fault == 'canary_version' else 'source_split_policy_version'
        case['data']['execution']['segments'][0]['extraction_canary'][field] = 'old-version'
    with pytest.raises(RuntimeError): verifier.summarize(**case)
