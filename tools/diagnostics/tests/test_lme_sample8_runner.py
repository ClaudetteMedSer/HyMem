"""Offline successor-runner controls using invented content; no paid calls.

Set HYMEM_SAMPLE8_SOURCE to the frozen R5 tree for the real startup controls.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "tools/diagnostics/lme_sample8_v1/bundle"
OLD = ROOT / "tools/diagnostics/lme_stock_q1/pending"
INDICES = [213, 262, 329, 339, 370, 372, 392, 400]
IDS = ['c18a7dc8', 'gpt4_e061b84f', 'gpt4_483dd43c', 'gpt4_7de946e7',
       '945e3d21', '71315a70', '72e3ee87', 'e61a7584']
CENSUS = [(49, 509), (46, 467), (52, 532), (49, 533),
          (48, 489), (53, 514), (51, 496), (50, 496)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(filename):
    path = BUNDLE / filename
    module = types.ModuleType('sample8_' + filename.removesuffix('.py'))
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('network forbidden during offline diagnostic verification')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'getnameinfo'):
        monkeypatch.setattr(socket, name, forbidden)


def frozen_source():
    return Path(os.environ.get('HYMEM_SAMPLE8_SOURCE', str(ROOT))).resolve()


def test_fixed_draw_matches_real_pinned_stock_selector_without_labels():
    run = load('q1_stock_run.py')
    selector = run.selector_from_source(frozen_source() / 'benchmarks/longmemeval_adapter.py')
    assert run.verify_selection(selector) == INDICES
    assert 210 not in INDICES
    assert run.SAMPLE == 8 and run.SEED == 0 and run.TIMEOUT == 32400
    assert run.QUESTION_IDS == IDS and run.QUESTION_CENSUS == CENSUS
    with pytest.raises(RuntimeError, match='selected_positions_drift'):
        run.verify_selection(lambda rows, **kwargs: list(reversed(INDICES)))


def test_recipe_has_only_fixed_selection_changes_from_prior_q1():
    run = load('q1_stock_run.py')
    old = types.ModuleType('old_q1')
    exec(compile((OLD / 'q1_stock_run.py').read_bytes(), str(OLD / 'q1_stock_run.py'), 'exec'), old.__dict__)
    expected = old.stock_arguments()
    expected[expected.index('--sample') + 1] = '8'
    expected[expected.index('--seed') + 1] = '0'
    assert run.stock_arguments() == expected
    startup = load('lme_q1_startup_preflight.py')
    startup.request_recipe(run.stock_arguments())
    assert 'HYMEM_LLM_EXTRA_BODY' not in run.environment()
    assert 'DEEPSEEK_API_KEY' not in run.environment()
    for seed in (True, 1, 53, '0'):
        with pytest.raises(RuntimeError, match='seed_drift'):
            run.stock_arguments(seed)


@pytest.mark.parametrize('flag', ['--api-key', '--hymem-api-key', '--answer-api-key',
                                  '--judge-api-key', '--resume-from', '--retry-failures',
                                  '--skip-extraction-canary', '--no-dream', '--api-key=fake'])
def test_forbidden_recipe_rejected_before_source_import(flag):
    startup = load('lme_q1_startup_preflight.py')
    argv = load('q1_stock_run.py').stock_arguments()
    with pytest.raises(RuntimeError, match='q1_preflight_forbidden_recipe'):
        startup.request_recipe(argv + [flag, 'synthetic-only'])


@pytest.mark.parametrize('flag,value', [('--sample', '1'), ('--seed', '53'), ('--workers', '2'),
                                       ('--indexing-max-cycles', '101'), ('--indexing-timeout-s', '3601'),
                                       ('--judge-protocol', 'official'), ('--top-k', '20')])
def test_noncanonical_fixed_recipe_rejected(flag, value):
    startup = load('lme_q1_startup_preflight.py')
    argv = load('q1_stock_run.py').stock_arguments()
    argv[argv.index(flag) + 1] = value
    with pytest.raises(RuntimeError, match='sample8_preflight_fixed_recipe'):
        startup.request_recipe(argv)


@pytest.mark.parametrize('extra', [['--embeddings'], ['--auto-ability'], ['trailing'],
                                 ['--retrieval-only'], ['--aggregation-nodes']])
def test_unknown_duplicate_or_trailing_arguments_rejected(extra):
    startup = load('lme_q1_startup_preflight.py')
    with pytest.raises(RuntimeError, match='sample8_preflight_'):
        startup.request_recipe(load('q1_stock_run.py').stock_arguments() + extra)


@pytest.mark.parametrize('ids', [IDS[:-1], list(reversed(IDS)), IDS[:-1] + [IDS[0]],
                               ['invented'] * 8, [True] + IDS[1:], tuple(IDS)])
def test_expected_ids_are_exact_in_both_helpers(ids):
    with pytest.raises(RuntimeError, match='question_ids_contract'):
        load('q1_stock_run.py').question_ids_from_manifest({'question_ids': ids})
    with pytest.raises(RuntimeError, match='sample8_preflight_expected_ids'):
        load('lme_q1_startup_preflight.py').validate_expected_ids(ids)


def manifest_for(run):
    return {'schema': 'stock-lme-sample8-development-run-preparation-v1',
            'dataset_sha256': run.DATASET_SHA, 'sample': 8, 'seed': 0,
            'source_indices': list(INDICES), 'question_ids': list(IDS),
            'stock_arguments': run.stock_arguments(), 'supervision_seconds': 32400,
            'source_sha256': {'invented.py': 'a' * 64}, 'helper_sha256': {}}


@pytest.mark.parametrize('field,value', [('sample', True), ('sample', 7), ('seed', False),
                                       ('seed', 53), ('source_indices', [213] * 8),
                                       ('source_indices', [213.0] + INDICES[1:]),
                                       ('supervision_seconds', 5400), ('supervision_seconds', 32400.0),
                                       ('question_ids', list(reversed(IDS)))])
def test_manifest_contract_rejects_drift(tmp_path, monkeypatch, field, value):
    run = load('q1_stock_run.py')
    manifest = manifest_for(run)
    monkeypatch.setattr(run, 'PACKAGE', tmp_path)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    assert run.validate_package(sha(path)) == manifest
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='manifest_contract|question_ids_contract'):
        run.validate_package(sha(path))


def synthetic_rows(*, full_census=False):
    rows = []
    ids_by_index = dict(zip(INDICES, IDS))
    sizes = dict(zip(INDICES, CENSUS)) if full_census else {}
    for index in range(500):
        sessions, messages = sizes.get(index, (1, 1))
        counts = [messages // sessions + (i < messages % sessions) for i in range(sessions)]
        session_ids = [f'invented-{index}-{i}' for i in range(sessions)]
        rows.append({'question_id': ids_by_index.get(index, f'invented-{index:04d}'),
                     'question_type': 'single-session-user', 'question': 'What is the invented kite color?',
                     'answer': 'blue', 'question_date': '2026/01/02 (Fri) 12:00',
                     'haystack_sessions': [[{'role': 'user', 'content': 'My invented kite is blue.'}]
                                           * count for count in counts],
                     'haystack_session_ids': session_ids, 'answer_session_ids': [session_ids[0]],
                     'haystack_dates': ['2026/01/01 (Thu) 12:00'] * sessions})
    return rows


@pytest.mark.parametrize('fault', [None, 'extra', 'symlink', 'hash'])
def test_exact_source_inventory_before_selection(tmp_path, monkeypatch, fault):
    run = load('q1_stock_run.py')
    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'invented.py'
    path.write_text('# invented, never executed\n')
    files = {path.name: sha(path)}
    monkeypatch.setattr(run, 'SOURCE', source)
    monkeypatch.setattr(run, 'selector_from_source', lambda _: lambda rows, **kwargs: list(INDICES))
    if fault == 'extra':
        (source / 'unlisted.bin').write_bytes(b'\x00')
    if fault == 'symlink':
        (source / 'outside').symlink_to(tmp_path)
    if fault == 'hash':
        files[path.name] = 'a' * 64
    if fault:
        with pytest.raises(RuntimeError, match='source_inventory_drift|source_symlink|source_drift'):
            run.verify_source({'source_sha256': files})
    else:
        selector = run.verify_source({'source_sha256': files})
        assert selector(list(range(500)), sample=8, seed=0) == INDICES


@pytest.mark.parametrize('fault', [None, 'id', 'census', 'selector', 'dataset_count'])
def test_preflight_census_bound_to_all_eight_original_positions(tmp_path, monkeypatch, fault):
    run = load('q1_stock_run.py')
    rows = synthetic_rows(full_census=True)
    if fault == 'id':
        rows[213]['question_id'] = 'substituted'
    if fault == 'census':
        rows[213]['haystack_sessions'].pop()
    if fault == 'dataset_count':
        rows.pop()
    data = tmp_path / 'longmemeval_s_cleaned.json'
    data.write_text(json.dumps(rows))
    monkeypatch.setattr(run, 'DATA', data)
    monkeypatch.setattr(run, 'OUTPUT', tmp_path)
    monkeypatch.setattr(run, 'sha', lambda _: run.DATASET_SHA)
    selector = lambda source, **kwargs: [source[index] for index in INDICES]
    if fault == 'selector':
        selector = lambda source, **kwargs: [source[index] for index in reversed(INDICES)]
    monkeypatch.setattr(run, 'verify_source', lambda _: selector)
    calls = []
    def startup(**kwargs):
        calls.append(kwargs)
        return {'status': 'passed', 'provider_completions': 0}
    monkeypatch.setattr(run, 'load_helper', lambda _: types.SimpleNamespace(run_probe=startup))
    if fault:
        with pytest.raises(RuntimeError, match='question_id_drift|question_census_drift|sample_drift'):
            run.preflight(manifest_for(run))
        assert calls == []
    else:
        result = run.preflight(manifest_for(run))
        assert result['question_ids'] == IDS and result['source_indices'] == INDICES
        assert result['sessions'] == 398 and result['messages'] == 4036
        assert result['api_calls'] == 0 and result['benchmark_executed'] is False
        assert calls == [{'source': run.SOURCE, 'arguments': run.stock_arguments(),
                          'output': tmp_path, 'expected_question_ids': IDS}]


def test_worker_calls_only_stock_cli_with_clean_environment(monkeypatch):
    run = load('q1_stock_run.py')
    pin = 'a' * 64
    monkeypatch.setattr(run.sys, 'stdin', types.SimpleNamespace(buffer=io.BytesIO(
        run.canonical({'execute_stock_sample8': pin}))))
    monkeypatch.setattr(run.sys, 'path', list(sys.path))
    monkeypatch.setattr(run.sys, 'argv', [])
    monkeypatch.setattr(run.os, 'environ', {'HYMEM_LLM_EXTRA_BODY': 'unwanted', 'HTTP_PROXY': 'unwanted'})
    monkeypatch.setattr(run, 'validate_package', lambda _: {})
    checks = []
    monkeypatch.setattr(run, 'verify_source', lambda _: checks.append(True))
    monkeypatch.setattr(run, 'load_helper', lambda _: types.SimpleNamespace(read_key=lambda: 'synthetic-key'))
    called = []
    def stock(path, *, run_name):
        assert 'HYMEM_LLM_EXTRA_BODY' not in run.os.environ and 'HTTP_PROXY' not in run.os.environ
        assert run.os.environ['DEEPSEEK_API_KEY'] == 'synthetic-key'
        assert run.sys.argv == run.stock_arguments()
        called.append((path, run_name))
    monkeypatch.setattr(run.runpy, 'run_path', stock)
    run.worker(pin)
    assert checks == [True, True] and called == [(run.stock_arguments()[0], '__main__')]


def test_old_worker_authorization_is_rejected_before_credentials(monkeypatch):
    run = load('q1_stock_run.py')
    pin = 'a' * 64
    monkeypatch.setattr(run.sys, 'stdin', types.SimpleNamespace(buffer=io.BytesIO(
        run.canonical({'execute_stock_q1': pin}))))
    monkeypatch.setattr(run, 'load_helper', lambda _: pytest.fail('credential helper reached'))
    with pytest.raises(RuntimeError, match='worker_authorization'):
        run.worker(pin)


CHILD = r'''
import importlib.util,json,os,pathlib,sys
helper,source,data,output = map(pathlib.Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('sample8_startup',helper)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
output.mkdir(); before=dict(os.environ)
report=module.run_probe(source=source,arguments=module.stock_arguments(source,data,output),
                        output=output,expected_question_ids=module.QUESTION_IDS)
assert dict(os.environ)==before
blocked=[]
for event in ['socket.connect','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr',
              'socket.getnameinfo','socket.sendto','socket.sendmsg']:
    try: sys.audit(event,'invented-only')
    except RuntimeError as exc:
        assert str(exc)=='q1_preflight_outbound_forbidden'; blocked.append(event)
report['test_events_blocked']=len(blocked)
print(json.dumps(report,sort_keys=True))
'''


def test_real_stock_cli_creates_exact_eight_id_checkpoint_without_provider_calls(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'longmemeval_s_cleaned.json').write_text(json.dumps(synthetic_rows()))
    result = subprocess.run([sys.executable, '-I', '-B', '-c', CHILD,
                             str(BUNDLE / 'lme_q1_startup_preflight.py'), str(frozen_source()),
                             str(data), str(tmp_path / 'output')],
                            env={'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path),
                                 'HYMEM_LLM_EXTRA_BODY': 'ambient-invented',
                                 'DEEPSEEK_API_KEY': 'ambient-invented-key'},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert result.stderr == '' and 'ambient-invented' not in result.stdout
    report = json.loads(result.stdout)
    assert report['status'] == 'passed' and report['expected_question_ids'] == IDS
    assert report['expected_question_count'] == 8 and report['test_events_blocked'] == 7
    for field in ('runtime_transport_identity_exact', 'checkpoint_runtime_producer_matches',
                  'stock_cli_pre_provider_boundary_reached', 'cli_checkpoint_handles_closed'):
        assert report[field] is True
    for field in ('provider_completions', 'provider_http_attempts', 'provider_successful_responses',
                  'completed_questions'):
        assert report[field] == 0
    assert report['cleanup_failed'] is False and report['real_credentials_loaded'] is False
    snapshot = json.loads((tmp_path / 'output/cli-preflight/checkpoint.json').read_text())
    assert snapshot['expected_ids'] == IDS and snapshot['entries'] == {}
    assert snapshot['execution_segments'] == [] and snapshot['status'] == 'running'
