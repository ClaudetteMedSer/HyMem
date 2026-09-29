"""Prepare or supervise the single failed-question R6 regression; inert on import.

Raw data, credentials, private logs and kept stores stay on the remote host.
Process completion is not interpreted as a passed indexing/benchmark result.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import stat
import sys
import time
import types

SOURCE = Path('/candidate')
PACKAGE = Path('/diag')
DATA = Path('/data/longmemeval_s_cleaned.json')
OUTPUT = Path('/results')
PYTHON = '/home/node/hymem-env/bin/python3'
MODEL = 'deepseek-flash'
ENDPOINT = 'https://api.deepseek.com'
TIMEOUT = 5400
SAMPLE = 1
SEED = 0
SOURCE_INDICES = [329]
QUESTION_IDS = ['gpt4_483dd43c']
QUESTION_CENSUS = [(52, 532)]
SOURCE_MANIFEST_SHA = 'bd8d0f3a8fb40bd6b77e7ca6579c8e5e8ee78733bea2bb22df71bc4b2c12eaa2'
CANARY_VERSION = 'hymem-phase1-extraction-canary-v19'
SPLIT_VERSION = 'hymem-source-semantic-split-v11'
DATASET_SHA = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
SELECTOR_SHA = '2f94272ef1383b1f5a7530644d28d1bdc9c7001231cd46297af392f33e74dc43'


def require(value, code):
    if not value:
        raise RuntimeError(code)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode()


def save(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def selector_from_source(path):
    """Execute only the pinned label-blind selector, not the adapter module."""
    text = path.read_text()
    tree = ast.parse(text)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'select_label_blind_questions']
    require(len(functions) == 1, 'selector_count')
    function = functions[0]
    require(hashlib.sha256(ast.get_source_segment(text, function).encode()).hexdigest()
            == SELECTOR_SHA, 'selector_source_drift')
    namespace = {'hashlib': hashlib, 'BenchmarkIntegrityError': ValueError}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function.name]


def verify_selection(selector):
    """Check the preregistered label-blind draw; never search seeds or labels."""
    selected = selector(list(range(500)), sample=SAMPLE, seed=SEED)
    require(selected == SOURCE_INDICES, 'selected_positions_drift')
    return selected


def question_ids_from_manifest(manifest):
    ids = manifest.get('question_ids')
    require(type(ids) is list and len(ids) == SAMPLE
            and all(type(value) is str and value and value.strip() == value
                    for value in ids)
            and len(set(ids)) == SAMPLE and ids == QUESTION_IDS, 'question_ids_contract')
    return list(ids)


def stock_arguments(seed=SEED):
    require(type(seed) is int and seed == SEED, 'seed_drift')
    body = '{"thinking":{"type":"disabled"}}'
    return [
        '/candidate/benchmarks/longmemeval_adapter.py',
        '--scales', 'S', '--sample', str(SAMPLE), '--seed', str(seed), '--workers', '1',
        '--top-k', '15', '--auto-ability', '--permissive-default',
        '--hymem-model', MODEL, '--hymem-base-url', ENDPOINT,
        '--hymem-thinking', 'disabled',
        '--answer-model', MODEL, '--answer-base-url', ENDPOINT,
        '--answer-extra-body', body,
        '--judge-model', MODEL, '--judge-base-url', ENDPOINT,
        '--judge-extra-body', body, '--judge-protocol', 'legacy-custom',
        '--no-prereg', '--protocol-split', 'full',
        '--indexing-max-cycles', '100', '--indexing-timeout-s', '3600',
        '--indexing-require-healthy', '--keep-db',
        '--data-dir', '/data', '--results-dir', '/results/benchmark',
        '--checkpoint', '/results/benchmark/checkpoint.json',
    ]


def environment():
    # Never inherit operator overrides, proxies, API keys or producer settings.
    return {'PATH': '/home/node/hymem-env/bin:/usr/bin:/bin', 'HOME': '/tmp',
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
            'TMPDIR': '/results/stores', 'LANG': 'C.UTF-8'}


def validate_package(manifest_sha):
    require(type(manifest_sha) is str and re.fullmatch('[0-9a-f]{64}', manifest_sha), 'manifest_pin')
    manifest_path = PACKAGE / 'manifest.json'
    require(sha(manifest_path) == manifest_sha, 'manifest_drift')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema'] == 'r6-lme-failed-question-preparation-v1', 'manifest_schema')
    require(manifest['approved_source_manifest_sha256'] == SOURCE_MANIFEST_SHA
            and manifest['source_files'] == 231 and len(manifest['source_sha256']) == 231
            and manifest['canary_version'] == CANARY_VERSION
            and manifest['source_split_policy_version'] == SPLIT_VERSION
            and manifest['global_paid_call_cap'] is None
            and manifest['rerolls_allowed'] is False and manifest['resume_allowed'] is False,
            'manifest_source_contract')
    require(manifest['dataset_sha256'] == DATASET_SHA
            and type(manifest['sample']) is int and manifest['sample'] == SAMPLE
            and type(manifest['seed']) is int and manifest['seed'] == SEED
            and type(manifest['source_indices']) is list
            and all(type(value) is int for value in manifest['source_indices'])
            and manifest['source_indices'] == SOURCE_INDICES
            and manifest['stock_arguments'] == stock_arguments()
            and type(manifest['supervision_seconds']) is int
            and manifest['supervision_seconds'] == TIMEOUT, 'manifest_contract')
    require(set(path.name for path in PACKAGE.iterdir())
            == {'manifest.json', *manifest['helper_sha256']}, 'package_inventory')
    for name, expected in manifest['helper_sha256'].items():
        require(type(name) is str and Path(name).name == name
                and type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'helper_entry')
        path = PACKAGE / name
        require(path.resolve() == path and not path.is_symlink()
                and stat.S_ISREG(path.lstat().st_mode) and sha(path) == expected, 'helper_drift')
    question_ids_from_manifest(manifest)
    return manifest


def verify_source(manifest):
    files = manifest['source_sha256']
    require(type(files) is dict and bool(files), 'source_inventory')
    require(SOURCE.resolve() == SOURCE and SOURCE.is_dir()
            and not SOURCE.is_symlink(), 'source_root')
    actual = set()
    def traversal_failed(_error):
        raise RuntimeError('source_traversal_failed')
    # Inspect every entry, including root modules, binary assets, hidden files,
    # and directory links. Do not follow symlinks or ignore unknown extensions.
    for directory, directories, filenames in os.walk(
            SOURCE, followlinks=False, onerror=traversal_failed):
        for name in directories + filenames:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), 'source_symlink')
            require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'source_non_regular')
            if stat.S_ISREG(mode):
                actual.add(path.relative_to(SOURCE).as_posix())
    require(actual == set(files), 'source_inventory_drift')
    for name, expected in files.items():
        require(type(name) is str and name == Path(name).as_posix()
                and not Path(name).is_absolute() and '..' not in Path(name).parts
                and type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected),
                'source_manifest_entry')
        path = SOURCE / name
        require(path.resolve() == path and path.is_relative_to(SOURCE)
                and not path.is_symlink() and sha(path) == expected, 'source_drift')
    selector = selector_from_source(SOURCE / 'benchmarks/longmemeval_adapter.py')
    verify_selection(selector)
    return selector


def preflight(manifest):
    selector = verify_source(manifest)
    require(DATA.resolve() == DATA and not DATA.is_symlink() and sha(DATA) == DATASET_SHA, 'dataset_drift')
    # Streaming loader preserves the original full dataset; the stock CLI will
    # read this same file. Never create a one-row replacement dataset.
    import ijson
    with DATA.open('rb') as stream:
        rows = list(ijson.items(stream, 'item'))
    selected = selector(rows, sample=SAMPLE, seed=SEED)
    require(len(rows) == 500
            and selected == [rows[index] for index in SOURCE_INDICES], 'sample_drift')
    question_ids = question_ids_from_manifest(manifest)
    require([row.get('question_id') for row in selected] == question_ids, 'question_id_drift')
    census = []
    for source_index, target in zip(SOURCE_INDICES, selected):
        sessions = target.get('haystack_sessions')
        require(type(sessions) is list and sessions
                and all(type(session) is list for session in sessions), 'question_census_shape')
        census.append({'source_index': source_index, 'question_id': target['question_id'],
                       'sessions': len(sessions), 'messages': sum(map(len, sessions))})
    require([(row['sessions'], row['messages']) for row in census] == QUESTION_CENSUS,
            'question_census_drift')
    total_sessions = sum(row['sessions'] for row in census)
    total_messages = sum(row['messages'] for row in census)
    # Stock CLI reads this same full dataset again. Release this copy first.
    del sessions, target, rows, selected
    startup = load_helper('lme_q1_startup_preflight')
    try:
        startup_report = startup.run_probe(
            source=SOURCE, arguments=stock_arguments(), output=OUTPUT,
            expected_question_ids=question_ids)
    except BaseException as exc:
        failure_report = getattr(exc, 'q1_preflight_report', None)
        if type(failure_report) is dict:
            try:
                save(OUTPUT / 'preflight-startup-failure.json', failure_report)
            except BaseException:
                exc.add_note('r6_regression_preflight_startup_failure_receipt_failed')
        raise
    return {'status': 'preflight_only', 'dataset_sha256': DATASET_SHA,
            'source_mapping_sha256': hashlib.sha256(canonical(manifest['source_sha256'])).hexdigest(),
            'source_files_verified': len(manifest['source_sha256']), 'source_question_count': 500,
            'source_indices': list(SOURCE_INDICES), 'question_ids': question_ids,
            'seed': SEED, 'sample': SAMPLE, 'selection_predeclared': True,
            'per_question_census': census, 'sessions': total_sessions,
            'messages': total_messages, 'api_calls': 0,
            'raw_data_exported': False, 'benchmark_executed': False,
            'startup_probe': startup_report}


def load_verified_module(path, expected_sha256, module_name):
    """Compile verified bytes, never consult or create an import bytecode cache.

    Host-side plan consumers can use the same function after verifying this
    runner's bytes against the independently pinned package manifest.
    """
    require(path.resolve() == path and not path.is_symlink()
            and stat.S_ISREG(path.lstat().st_mode), 'helper_path')
    raw = path.read_bytes()
    require(type(expected_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_sha256)
            and hashlib.sha256(raw).hexdigest() == expected_sha256, 'helper_drift')
    require(type(module_name) is str and re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', module_name), 'helper_module')
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(path), 'exec'), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def load_helper(name):
    require(type(name) is str and re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', name), 'helper_name')
    manifest = json.loads((PACKAGE / 'manifest.json').read_bytes())
    expected = manifest['helper_sha256'][name + '.py']
    return load_verified_module(PACKAGE / (name + '.py'), expected, 'q1_' + name)


def cancel(_signum, _frame):
    raise KeyboardInterrupt('r6_regression_shutdown')


def worker(manifest_sha):
    # Supervisor releases the small authorization frame only after recording
    # ownership of this process group. Credentials never travel in argv/stdin.
    payload = sys.stdin.buffer.read(256)
    require(payload == canonical({'execute_r6_failed_question_v1': manifest_sha}), 'worker_authorization')
    manifest = validate_package(manifest_sha)
    verify_source(manifest)
    common = load_helper('transport_common')
    key = common.read_key()
    os.environ.clear()
    os.environ.update(environment())
    require('HYMEM_LLM_EXTRA_BODY' not in os.environ, 'canonical_extra_body_environment')
    os.environ['DEEPSEEK_API_KEY'] = key
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'benchmarks')]
    sys.argv = stock_arguments()
    # Execute the actual pinned CLI unchanged. No monkeypatched provider,
    # validation, input selection, retry policy, semantic output or scorer.
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        verify_source(manifest)


def supervise(manifest_sha):
    expiry = time.monotonic() + TIMEOUT
    manifest = validate_package(manifest_sha)
    require(os.geteuid() == 1000 and OUTPUT.resolve() == OUTPUT, 'isolated_runtime_required')
    for name in ('benchmark', 'stores'):
        (OUTPUT / name).mkdir(mode=0o700)
    save(OUTPUT / 'preflight.json', preflight(manifest))
    supervisor = load_helper('supervised_invocation')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    outcome = None
    failure = None
    source_unchanged = False
    try:
        outcome = supervisor.supervise_invocation(
            [PYTHON, '-I', '-B', '/diag/q1_stock_run.py', 'worker', '--manifest-sha256', manifest_sha],
            cwd=SOURCE, env=environment(), output_dir=OUTPUT / 'invocation',
            timeout_seconds=TIMEOUT, deadline_expires_at=expiry,
            cleanup_seconds=10, output_limit_bytes=256 * 1024 * 1024,
            stdin_bytes=canonical({'execute_r6_failed_question_v1': manifest_sha}),
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        post_failure = None
        try:
            validate_package(manifest_sha)
            verify_source(manifest)
            require(sha(DATA) == DATASET_SHA, 'postrun_dataset_drift')
            source_unchanged = True
        except BaseException as exc:
            post_failure = exc
        try:
            save(OUTPUT / 'supervisor-summary.json', {
                'status': 'stock_process_finished' if outcome is not None else 'supervisor_failed',
                'outcome': asdict(outcome) if outcome is not None else None,
                'exception_type': type(failure).__name__ if failure is not None else None,
                'postrun_exception_type': type(post_failure).__name__ if post_failure is not None else None,
                'manifest_sha256': manifest_sha,
                'source_and_dataset_unchanged': source_unchanged,
                'benchmark_pass_verified': False, 'score_verified': False,
                'global_paid_call_cap': None, 'deployment_performed': False,
            })
        except BaseException as exc:
            if failure is not None:
                failure.add_note('stock_q1_terminal_receipt_failed')
            elif post_failure is not None:
                post_failure.add_note('stock_q1_terminal_receipt_failed')
            else:
                raise
        if post_failure is not None:
            if failure is not None:
                failure.add_note('stock_q1_postrun_identity_failed')
            else:
                raise post_failure
    return 0 if (outcome.status == 'completed' and outcome.returncode == 0
                 and outcome.safe_to_continue and source_unchanged) else 1


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'supervise', 'worker'))
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.manifest_sha256)
        return 0
    os.environ.clear()
    os.environ.update(environment())
    if args.action == 'preflight':
        manifest = validate_package(args.manifest_sha256)
        value = preflight(manifest)
        save(OUTPUT / 'preflight.json', value)
        print(json.dumps(value, sort_keys=True))
        return 0
    return supervise(args.manifest_sha256)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, Exception) as exc:
        # Private child logs contain the CLI's own diagnostics. Never echo
        # exception values, credentials or dataset content from this wrapper.
        print(json.dumps({'status': 'r6_regression_wrapper_failed', 'exception_type': type(exc).__name__}))
        raise SystemExit(1)
