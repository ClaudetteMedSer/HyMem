"""Freeze/package a reviewed R4-only integration; exclusive additive outputs."""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

DOCS = Path(__file__).resolve().parent
R3_NAME = '2026-09-19-lme-independent-summary-indexing-r3-manifest.json'
R3_PIN = '573fd0f5d763adbbd9f26df7fa249a546e2980304d303771ea76fcec4526aa66'
P14 = '2026-09-24-lme-14-current-model-history.patch'
M14 = '2026-09-24-lme-independent-summary-indexing-r4-manifest.json'
AUX = '2026-09-24-lme-r4-auxiliary-model-default.patch'
RECEIPT = '2026-09-24-lme-r4-exact-reconstruction.json'
NEW_TESTS = {'tests/test_current_deepseek_service.py',
             'tests/test_lme_current_model_startup.py', 'tests/test_lme_historical_admission.py'}
SOURCE_CHANGES = {'benchmarks/' + name + '.py' for name in (
    'beam_adapter', 'coref_eval', 'fact_probe', 'lme_canary_watch', 'lme_protocol',
    'lme_registry', 'locomo_adapter', 'longmemeval_adapter', 'msc_adapter',
    'multihop_miner', 'rerank_ab', 'rules_compliance')} | {
    'hymem/bootstrap.py', 'hymem/server.py', 'hymem/contrib/model_policy.py',
    'hymem/contrib/openai_client.py'}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, pin=None):
    assert path.is_file() and not path.is_symlink() and path.resolve() == path
    raw = path.read_bytes()
    assert pin is None or sha(raw) == pin, ('hash', str(path))
    return raw

def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()

def write_new(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)

def delta(before, after):
    parts = []
    for name, new in sorted(after.items()):
        old = before.get(name, b'')
        if old == new:
            continue
        assert (not old or old.endswith(b'\n')) and new.endswith(b'\n')
        parts.extend(difflib.unified_diff(old.decode().splitlines(keepends=True),
            new.decode().splitlines(keepends=True), fromfile='a/' + name if name in before else '/dev/null',
            tofile='b/' + name, n=3, lineterm='\n'))
    return ''.join(parts).encode()

def method(raw, cls, function):
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return ast.dump(next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == function),
                    include_attributes=False)

COLLECT = r'''
import json,os,pathlib,socket,sys
root,output=map(pathlib.Path,sys.argv[1:])
os.environ.clear()
os.environ.update(PATH='/opt/anaconda3/bin:/usr/bin:/bin',HOME=str(output.parent),
    TMPDIR=str(output.parent),PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
os.chdir(root);sys.path.insert(0,str(root))
def audit(event,args):
    if event in {'socket.connect','socket.getaddrinfo','socket.sendto'}:
        raise PermissionError('collection is provider-free')
sys.addaudithook(audit)
import pytest
class Nodes:
    def pytest_collection_finish(self,session):
        nodes=sorted(item.nodeid for item in session.items)
        assert len(nodes)==len(set(nodes))
        with output.open('x') as stream:json.dump(nodes,stream)
raise SystemExit(pytest.main(['tests','--collect-only','-q','-p','no:cacheprovider'],plugins=[Nodes()]))
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r3', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    r3 = json.loads(read(DOCS / R3_NAME, R3_PIN))
    old_pins = {**r3['source_sha256'], **r3['test_sha256'], **r3['auxiliary_sha256']}
    assert len(old_pins) == 459 and not NEW_TESTS.intersection(old_pins)
    before = {name: read(args.r3 / name, pin) for name, pin in old_pins.items()}
    expected = set(old_pins) | NEW_TESTS
    actual = {p.relative_to(args.candidate).as_posix() for p in args.candidate.rglob('*') if p.is_file()}
    assert expected <= actual
    assert all('__pycache__' in Path(name).parts and name.endswith('.pyc') for name in actual - expected)
    after = {name: read(args.candidate / name) for name in expected}
    source = {name: sha(after[name]) for name in r3['source_sha256']}
    tests = {name: sha(after[name]) for name in sorted(set(r3['test_sha256']) | NEW_TESTS)}
    auxiliary = {name: sha(after[name]) for name in r3['auxiliary_sha256']}
    changed_source = sorted(name for name in source if source[name] != old_pins[name])
    assert set(changed_source) == SOURCE_CHANGES
    assert [name for name in auxiliary if auxiliary[name] != old_pins[name]] == ['README.md']
    assert method(before['hymem/contrib/openai_client.py'], 'OpenAICompatibleClient', 'complete') == method(
        after['hymem/contrib/openai_client.py'], 'OpenAICompatibleClient', 'complete')
    assert method(before['benchmarks/longmemeval_adapter.py'], 'LLMClient', 'chat') == method(
        after['benchmarks/longmemeval_adapter.py'], 'LLMClient', 'chat')
    assert b'HYMEM_LLM_EXTRA_BODY' in after['hymem/contrib/openai_client.py']
    tree = args.candidate.parent / 'verification-tree-r4'
    assert not tree.exists()
    tree.mkdir(mode=0o700)
    for name, raw in sorted(after.items()):
        target = tree / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_new(target, raw)
        target.chmod(0o400)
    node_path = tree.parent / 'r4-collected-nodeids.json'
    log = tree.parent / 'r4-collection.log'
    with log.open('xb') as stream:
        process = subprocess.run([sys.executable, '-I', '-B', '-c', COLLECT, str(tree), str(node_path)],
            env={}, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    assert process.returncode == 0, str(log)
    nodes = json.loads(read(node_path))
    assert len(nodes) == len(set(nodes)) and len(nodes) >= len(r3['expected_nodeids'])
    for name, raw in after.items():
        assert read(tree / name, sha(raw)) == raw
    chain = {**r3['prior_patch_sha256'], r3['patch']: r3['patch_sha256']}
    assert len(chain) == 14
    for name, pin in chain.items():
        read(DOCS / name, pin)
    app_before = {name: before[name] for name in set(r3['source_sha256']) | set(r3['test_sha256'])}
    app_after = {name: after[name] for name in set(source) | set(tests)}
    patch = delta(app_before, app_after)
    aux_patch = delta({'README.md': before['README.md']}, {'README.md': after['README.md']})
    helper = DOCS / '2026-09-24-reconstruct-independent-summary-r3.py'
    spec = importlib.util.spec_from_file_location('exact_durable_reconstruction', helper)
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    reconstructed, changed = exact.apply_exact(patch, app_before, app_after)
    assert reconstructed == app_after and len(reconstructed) == 454
    aux_before = {name: before[name] for name in auxiliary}
    aux_final, changed_aux = exact.apply_exact(aux_patch, aux_before, auxiliary)
    assert changed_aux == ['README.md']
    reconstructed.update(aux_final)
    assert reconstructed == after and len(reconstructed) == 462
    manifest = dict(schema='lme-r4-current-model-history-20260924-v1', revision='r4',
        base_commit=r3['base_commit'], prior_manifest=R3_NAME, prior_manifest_sha256=R3_PIN,
        prior_patch_sha256=chain, source_sha256=source, test_sha256=tests, auxiliary_sha256=auxiliary,
        expected_nodeids=nodes, selected_tests=sorted({node.split('::', 1)[0] for node in nodes}),
        source_files=230, test_files=224, auxiliary_files=8, combined_files=462,
        changed_source=changed_source, changed_existing_tests=sorted(name for name in r3['test_sha256']
            if tests[name] != old_pins[name]), added_tests=sorted(NEW_TESTS), changed_auxiliary=['README.md'],
        patch=P14, patch_sha256=sha(patch), auxiliary_patch=AUX, auxiliary_patch_sha256=sha(aux_patch),
        auxiliary_patch_test_only=True, auxiliary_patch_deployment_authorized=False,
        requested_service_not_immutable_weights=True, sdk_completion_and_raw_chat_unchanged=True,
        extra_body_extension_preserved=True, original_r3_summary_contract_preserved=True,
        generator_sha256=sha(read(Path(__file__))), exact_parser_sha256=sha(read(helper)),
        fresh_collection_only=True, tests_executed_by_packaging=False, full_suite_pass_claimed=False,
        old_vanished_receipts_are_not_new_evidence=True, api_calls=0, deployment_performed=False,
        source_tree=str(tree), benchmark_readiness_claimed=False,
        scope='R3 to R4 model/history integration only. All462 files reconstruct exactly. Node IDs are fresh collection metadata, not passing tests. README is separate test-only auxiliary, never an application patch.')
    manifest_raw = encode(manifest)
    receipt = dict(schema='r4-exact-reconstruction-20260924-v1', status='all462_file_bytes_reconstructed',
        manifest=M14, manifest_sha256=sha(manifest_raw), patch=P14, patch_sha256=sha(patch),
        auxiliary_patch=AUX, auxiliary_patch_sha256=sha(aux_patch), auxiliary_patch_test_only=True,
        prior_patch_hashes_rechecked=14, prior_r3_manifest_sha256=R3_PIN, source_files_verified=230,
        test_files_verified=224, auxiliary_files_verified=8, combined_files_verified=462,
        collected_nodeids=len(nodes), source_tree=str(tree), tests_executed=False, api_calls=0,
        full_suite_pass_claimed=False, benchmark_readiness_claimed=False, deployment_performed=False,
        old_vanished_receipts_reused=False, parser_sha256=sha(read(helper)))
    payloads = {P14: patch, M14: manifest_raw, AUX: aux_patch, RECEIPT: encode(receipt)}
    assert all(not (DOCS / name).exists() for name in payloads)
    for name, raw in payloads.items():
        write_new(DOCS / name, raw)
    print(json.dumps({'status':'r4_packaged_and_reconstructed_not_tested', 'tree':str(tree),
        'artifact_sha256':{name:sha(raw) for name,raw in payloads.items()},
        'collected_nodeids':len(nodes)}, sort_keys=True))

if __name__ == '__main__':
    main()
