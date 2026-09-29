"""Seal the frozen R6 source and one-question helpers offline; never launch."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import types

SOURCE_PIN = 'bd8d0f3a8fb40bd6b77e7ca6579c8e5e8ee78733bea2bb22df71bc4b2c12eaa2'
ROOT = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/r6-lme-failed-question-v1'
NAMES = {'q1_stock_run.py', 'q1_stock_host.py', 'q1_stock_validate.py',
         'lme_q1_startup_preflight.py', 'supervised_invocation.py', 'transport_common.py', 'RUNBOOK.md'}
SUPPORT = {'supervised_invocation.py': '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc',
           'transport_common.py': '1c98c73cfb1b607113c726616058862fd6bb16a57901a57c8962ee4a0a2807a0'}


def need(value, code):
    if not value:
        raise RuntimeError('r6_seal_' + code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: need(False, 'nonfinite_json'))


def regular(path):
    need(path.is_absolute() and path.resolve() == path, 'path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        need(stat.S_ISREG(info.st_mode) and info.st_size <= 32 * 1024 * 1024, 'regular')
        return stream.read()


def exact_tree(root, mapping):
    need(root.is_absolute() and root.resolve() == root and root.is_dir(), 'tree_root')
    need(type(mapping) is dict and bool(mapping), 'mapping')
    for name, pin in mapping.items():
        p = PurePosixPath(name)
        need(type(name) is str and p.as_posix() == name and not p.is_absolute()
             and '..' not in p.parts and all(not part.startswith('.') for part in p.parts)
             and type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'mapping_entry')
    actual = set()
    def failed(_error):
        need(False, 'tree_traversal')
    for directory, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in directories + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            need(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'tree_nonregular')
            if stat.S_ISREG(mode):
                actual.add(path.relative_to(root).as_posix())
    need(actual == set(mapping), 'tree_inventory')
    captured = {name: regular(root / name) for name in mapping}
    need(all(sha(body) == mapping[name] for name, body in captured.items()), 'tree_hash')
    return captured


def definitions(path):
    module = types.ModuleType('r6_seal_' + path.stem)
    module.__file__ = str(path)
    exec(compile(regular(path), str(path), 'exec'), module.__dict__)
    return module


def constant(raw, name):
    matches = [node.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    need(len(matches) == 1, 'constant_count')
    return ast.literal_eval(matches[0])


def build(tree, source_manifest):
    raw = regular(source_manifest)
    need(sha(raw) == SOURCE_PIN, 'source_manifest_pin')
    source = decode(raw)
    need(source['schema'] == 'lme-r6-sequential-fixes-frozen-inventory-v1', 'source_schema')
    expected = {}
    for group in ('source_sha256', 'test_sha256', 'auxiliary_sha256'):
        need(not set(expected).intersection(source[group]), 'overlap')
        expected.update(source[group])
    captured = exact_tree(tree, expected)
    pins = source['source_sha256']
    need(len(pins) == 231, 'source_count')
    base = Path(__file__).resolve().parent / 'bundle'
    need({p.name for p in base.iterdir()} == NAMES, 'helper_inventory')
    helpers = {name: regular(base / name) for name in NAMES}
    need(all(sha(helpers[name]) == pin for name, pin in SUPPORT.items()), 'support_drift')
    run, host = definitions(base / 'q1_stock_run.py'), definitions(base / 'q1_stock_host.py')
    run.verify_selection(run.selector_from_source(tree / 'benchmarks/longmemeval_adapter.py'))
    need(constant(captured['benchmarks/extraction_canary.py'], 'EXTRACTION_CANARY_VERSION')
         == run.CANARY_VERSION, 'canary_version')
    need(constant(captured['hymem/extraction/chunk.py'], 'SOURCE_RECORD_SPLIT_POLICY_VERSION')
         == run.SPLIT_VERSION, 'split_version')
    manifest = {
        'schema': 'r6-lme-failed-question-preparation-v1', 'source_revision': 'r6',
        'approved_source_manifest_sha256': SOURCE_PIN,
        'source_sha256': pins, 'source_files': len(pins),
        'source_mapping_sha256': sha(run.canonical(pins)),
        'source_inventory_policy': 'exact-manifest-regular-files-no-symlinks',
        'helper_sha256': {name: sha(body) for name, body in helpers.items()},
        'dataset_sha256': run.DATASET_SHA, 'sample': 1, 'seed': 0,
        'source_indices': [329], 'question_ids': ['gpt4_483dd43c'],
        'questions': [{'source_index': 329, 'question_id': 'gpt4_483dd43c',
                       'question_type': 'temporal-reasoning', 'sessions': 52, 'messages': 532}],
        'sessions': 52, 'messages': 532, 'source_question_count': 500,
        'selection_reason': 'fixed_prior_sample8_indexing_failure_regression',
        'selector_sha256': run.SELECTOR_SHA, 'stock_arguments': run.stock_arguments(),
        'supervision_seconds': run.TIMEOUT, 'cleanup_seconds': 10,
        'indexing_max_cycles': 100, 'indexing_timeout_seconds': 3600,
        'requested_model': run.MODEL, 'endpoint': run.ENDPOINT, 'thinking': 'disabled',
        'canary_version': run.CANARY_VERSION, 'source_split_policy_version': run.SPLIT_VERSION,
        'canary_suites': 1, 'question_attempts': 1,
        'docker_image': host.IMAGE, 'runtime_host_path': host.RUNTIME,
        'remote_run_root': ROOT, 'remote_source': ROOT + '/candidate',
        'global_paid_call_cap': None, 'rerolls_allowed': False, 'resume_allowed': False,
        'raw_content_exported': False, 'production_memory_mounted': False,
        'production_changes': False, 'paid_calls_by_preparation': 0,
        'full_500_readiness_verified': False,
        'summary_recovery_in_stock_digest': False,
    }
    host.command(ROOT, ROOT + '/candidate', '0' * 64)
    payload = {'input-manifest.json': raw, 'bundle/manifest.json': encoded(manifest),
               **{'candidate/' + name: captured[name] for name in pins},
               **{'bundle/' + name: body for name, body in helpers.items()}}
    return manifest, payload


def exclusive(path, body):
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    need(path.parent.resolve() == path.parent, 'output_parent')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, 'wb') as out:
        out.write(body)
        out.flush()
        os.fsync(out.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, payload = build(args.tree, args.manifest)
    need(args.output.is_absolute() and args.output.resolve() == args.output
         and not args.output.is_relative_to(args.tree)
         and not args.output.is_relative_to(Path(__file__).resolve().parent), 'output_scope')
    args.output.mkdir(mode=0o755)
    for name, body in sorted(payload.items()):
        exclusive(args.output / name, body)
    files = {name: sha(body) for name, body in payload.items()}
    exact_tree(args.output, files)
    archive = args.output.parent / (args.output.name + '.tgz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for name in sorted(files):
                tar.add(args.output / name, arcname=name, recursive=False)
    receipt = {'schema': 'r6-lme-failed-question-seal-v1',
               'manifest_sha256': sha(payload['bundle/manifest.json']),
               'source_manifest_sha256': SOURCE_PIN, 'source_files': len(manifest['source_sha256']),
               'archive_sha256': sha(regular(archive)), 'files_sha256': files,
               'output': str(args.output), 'archive': str(archive), 'paid_calls': 0}
    seal_path = args.output.parent / (args.output.name + '-seal.json')
    seal_raw = encoded(receipt)
    exclusive(seal_path, seal_raw)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files_sha256'}
                     | {'seal': str(seal_path), 'seal_sha256': sha(seal_raw)}, sort_keys=True))


if __name__ == '__main__':
    main()
