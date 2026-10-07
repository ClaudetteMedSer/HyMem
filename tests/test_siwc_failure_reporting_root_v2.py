"""Independent source-delta and frozen-protocol checks; no live requests."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import sys

from tools.diagnostics import siwc_lme_diagnostic_v2 as runner


ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle')


def test_consumed_sources_unchanged():
    pins = {
        'tools/diagnostics/siwc_lme_diagnostic_v1.py':
            '9366f40cf218581b51b877fd230e530c1e699cca0b76036e4304886c08c698f3',
        'tools/diagnostics/siwc_lme_diagnostic_progress_v3.py':
            '89e39169a81d9ee810e54fb4c143753906aed282f50527b923cca6cc504a5279',
        'benchmarks/lme_diagnostic.py':
            'b07cdb4d26ad2f95ab236fe8af4a1665d19c376ffaf077e90a561ce500ac1181',
    }
    for relative, expected in pins.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_no_runtime_policy_or_execution_changes():
    old = ast.parse((ROOT / 'tools/diagnostics/siwc_lme_diagnostic_v1.py').read_text())
    new = ast.parse((ROOT / 'tools/diagnostics/siwc_lme_diagnostic_v2.py').read_text())
    assert len(new.body) == len(old.body) + 2
    old_nodes = {getattr(n, 'name', ''): n for n in old.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new_nodes = {getattr(n, 'name', ''): n for n in new.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert set(new_nodes) - set(old_nodes) == {'_worker_failure_code'}
    for name, node in old_nodes.items():
        if name != '_question_worker':
            assert ast.dump(node) == ast.dump(new_nodes[name]), name
    # The only worker execution delta is the value assigned to its report code.
    def normalize(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'reason' for t in node.targets):
                node.value = ast.Constant(value='reporting-code')
        return ast.dump(tree)
    assert normalize(old_nodes['_question_worker']) == normalize(new_nodes['_question_worker'])
    def assignments(tree):
        return {ast.dump(n.targets[0]): ast.dump(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    a, b = assignments(old), assignments(new)
    allowed = {ast.dump(ast.Name(id=name, ctx=ast.Store())) for name in ('SCHEMA', 'RUNNER_RELATIVE', 'WORKER_EXCEPTION_TYPES')}
    assert {key for key in set(a) | set(b) if a.get(key) != b.get(key)} == allowed


def test_frozen_projection_adversarial_controls():
    script = r'''
import copy
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

repo, frozen = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('root_runner', repo / 'tools/diagnostics/siwc_lme_diagnostic_v2.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
loaded = r.import_source_only(frozen, frozen / 'source-map.json', r.ACCEPTED_INVENTORY_SHA256)
def blocked(*args, **kwargs):
    raise AssertionError('offline operation forbidden')
socket.socket = socket.create_connection = blocked
subprocess.Popen = blocked
p, s, lme = loaded['protocol'], loaded['strictness'], loaded['lme']
raw = dict(cycles=0, max_cycles=100, timeout_s=10800.0, elapsed_s=10800.0,
    complete=False, healthy=False, failure_reason='timeout_during_cycle',
    reports=[], final_status={}, quarantined={})
base = p.canonicalize_lme_indexing_summary(raw)
fault = lme.IndexingConvergenceError('private message', {'secret': 'private'})
checks = 0
def verify(value, expected, exc=fault):
    global checks
    result = r._worker_failure_code(loaded, SimpleNamespace(last_indexing_summary=value), exc)
    assert result == expected
    assert s.bounded_failure_text(result) == result
    assert 'private' not in result and '\n' not in result
    checks += 1
for code in ('timeout_before_cycle', 'timeout_during_cycle', 'timeout_after_cycle'):
    value = copy.deepcopy(base)
    value['failure']['code'] = code
    verify(value, 'indexing_failure:' + code)
    value['elapsed_s'] = 10799.0
    verify(value, 'worker_failure:IndexingConvergenceError')
for key, value in (('cycles', -1), ('cycles', True), ('elapsed_s', float('nan')),
        ('elapsed_s', float('inf')), ('elapsed_s', -1), ('timeout_s', 0),
        ('timeout_s', float('nan')), ('healthy', True), ('complete', True),
        ('schema', 'unknown'), ('outcome', 'success'), ('reports', [None]),
        ('cleanup_errors', [{'stage': 'secret', 'exception_type': 'ValueError'}])):
    bad = copy.deepcopy(base)
    bad[key] = value
    verify(bad, 'worker_failure:IndexingConvergenceError')
for value in (None, [], 'private', {}, {'failure': {'code': 'timeout_during_cycle'}}):
    verify(value, 'worker_failure:IndexingConvergenceError')
for code in ('private', 'timeout_during_cycle:private', 'timeout_during_cycle\n', None, True, 1):
    bad = copy.deepcopy(base)
    bad['failure']['code'] = code
    verify(bad, 'worker_failure:IndexingConvergenceError')
bad = copy.deepcopy(base)
bad['private_extra'] = 'private'
verify(bad, 'worker_failure:IndexingConvergenceError')
for name in r.WORKER_EXCEPTION_TYPES:
    exc = type(name, (Exception,), {})('private')
    verify(base, 'worker_failure:' + name, exc)
for name in ('PrivateFault', 'ValueError\nprivate', 'private.' * 100):
    exc = type(name, (Exception,), {})('private')
    verify(base, 'worker_failure:Exception', exc)
for code in ('worker_runtime_failure', 'not_started_after_campaign_stop'):
    assert s.bounded_failure_text(code) == 'unspecified_failure'
    checks += 1
assert r._worker_failure_code(loaded, None, fault) == 'worker_failure:IndexingConvergenceError'
print(json.dumps({'checks': checks + 1, 'provider_calls': 0}))
'''
    done = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(ROOT), str(FROZEN)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-3000:]
    import json
    result = json.loads(done.stdout)
    assert result == {'checks': 52, 'provider_calls': 0}
