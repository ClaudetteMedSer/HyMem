"""Root-owned tests of generated sources; all inference is invented/offline."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from tools.diagnostics import luna_semantic_policy_bundle_v2 as derive
from tools.diagnostics.tests.test_luna_semantic_entry_root import PROGRAM as ORIGINAL_PROGRAM


REPO = Path(__file__).resolve().parents[3]
CANDIDATE = Path('/private/tmp/hymem-semantic-policy-v2.aBQTUn6q/candidate')
INVENTORY = CANDIDATE.parent / 'map.json'
# This is test-only generated source, never dispatched to a service/provider.
BUNDLE = Path(tempfile.mkdtemp(prefix='hymem-policy-root-bundle-')).resolve() / 'bundle'
PROOF = derive.prepare(REPO, BUNDLE)
CODE = BUNDLE / 'code'


def load(relative, name):
    path = BUNDLE / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOST = load('code/tools/diagnostics/luna_semantic_probe_host.py', 'root_policy_host')
RUNNER = load('code/tools/diagnostics/luna_semantic_probe_run.py', 'root_policy_runner')
READER = load('code/tools/diagnostics/luna_semantic_probe_progress.py', 'root_policy_reader')
ADAPTER = load('adapter-v2.py', 'root_policy_adapter')
SEMANTIC = load('code/benchmarks/luna_semantic_canary.py', 'root_policy_canary')


def test_generated_pins_and_resource_caps_are_exact():
    pins = HOST.source_pins(CODE)
    assert len(pins) == 16
    assert ADAPTER.HOST_SHA == pins['tools/diagnostics/luna_semantic_probe_host.py']
    assert ADAPTER.RUN_SHA == pins['tools/diagnostics/luna_semantic_probe_run.py']
    assert ADAPTER.READER_SHA == pins['tools/diagnostics/luna_semantic_probe_progress.py']
    early_entry = (CODE / 'tools/diagnostics/luna_semantic_probe_run.py').read_text()
    assert pins['tools/diagnostics/luna_semantic_probe.py'] in early_entry
    assert '3ccb7ec12f93f8502fe8ed1448a071ac977dd334d17821f661913d634c27b334' not in early_entry
    assert HOST.LIMITS == {
        'turns': 29, 'known_tokens': 500000, 'seconds': 1800,
        'control_turns': 2, 'control_known_tokens': 100000, 'control_seconds': 240,
        'hybrid_turns': 3, 'hybrid_known_tokens': 160000, 'hybrid_seconds': 600,
        'invocation_seconds': 120, 'workers': 1, 'units': 25,
        'ordinary_replays': 8, 'new_judgments_max': 29}
    assert HOST.INVENTORY_SHA == hashlib.sha256(INVENTORY.read_bytes()).hexdigest()
    assert 'source-grounding-v1' not in (CODE / 'benchmarks/luna_semantic_canary.py').read_text()


def test_generator_cannot_write_inside_its_frozen_candidate(tmp_path, monkeypatch):
    frozen = tmp_path / 'frozen-candidate'
    frozen.mkdir()
    monkeypatch.setattr(derive, 'CANDIDATE', frozen)
    monkeypatch.setattr(derive, 'INVENTORY', frozen.with_name('map.json'))
    monkeypatch.setattr(derive, '_validate_candidate', lambda: None)
    with pytest.raises(ValueError):
        derive.prepare(REPO, frozen / 'forbidden-new-bundle')
    assert list(frozen.iterdir()) == []


def test_physical_candidate_passes_real_readonly_core_preflight():
    script = r'''
import sys, json, socket
from pathlib import Path
sys.path[:0] = sys.argv[1:3]
def deny(*a,**k): raise AssertionError('no network')
socket.socket.connect = deny
from tools.diagnostics import luna_semantic_probe as core, luna_semantic_cases as cases
from hymem.extraction import grounding
from benchmarks import luna_semantic_canary as canary, luna_semantic_stage_accounting as stage
result=core.verify_local(Path(sys.argv[1]),Path(sys.argv[3]),cases_module=cases,
    grounding_module=grounding,canary_module=canary,stage_module=stage)
assert result['candidate_files']==510
assert grounding.GROUNDING_CONTRACT_VERSION=='source-grounding-v2'
print(json.dumps(result))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script,
        str(CANDIDATE), str(CODE), str(INVENTORY)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('fault', ['none', 'result_write'])
def test_whole_entry_replay_preserves_29_call_accounting_and_failure_metadata(fault):
    judge_path = Path(__file__).with_name('test_luna_semantic_probe_root.py')
    source = judge_path.read_text()
    judge_node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Judge')
    judge = ast.get_source_segment(source, judge_node).replace('source-grounding-v1', 'source-grounding-v2')
    program = ORIGINAL_PROGRAM.replace('source-grounding-v1', 'source-grounding-v2')
    program = program.replace('luna-semantic-probe', 'luna-semantic-policy-v2')
    before = 'sys.path[:0] = [sys.argv[1], sys.argv[2]]'
    assert program.count(before) == 1
    program = program.replace(before, before[:-1] + ', ' + repr(str(REPO)) + ']')
    before = 'from tools.diagnostics.tests.test_luna_semantic_probe_root import Judge'
    assert program.count(before) == 1
    program = program.replace(before, judge)
    before = '    from tools.diagnostics import luna_semantic_verdict_replay_root as verifier'
    assert program.count(before) == 1
    program = program.replace(before,
        '    import importlib.util\n'
        f'    spec=importlib.util.spec_from_file_location("root_policy_replay",{str(BUNDLE / "verdict-replay.py")!r})\n'
        '    verifier=importlib.util.module_from_spec(spec)\n'
        '    spec.loader.exec_module(verifier)')
    result = subprocess.run([sys.executable, '-I', '-B', '-c', program,
        str(CANDIDATE), str(CODE), 'second', fault], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert '"new_synthetic_turns": 29' in result.stdout


# Rebind existing root-owned host/reader/canary/adapter controls in memory.
# All predicates, mutation cases, evidence/gold and expected counts are retained.
for _file in ('test_luna_semantic_host_root.py', 'test_luna_semantic_reader_root.py',
              'test_luna_semantic_canary_root.py', 'test_luna_semantic_adapter_root.py'):
    _path = Path(__file__).with_name(_file)
    _source = _path.read_text()
    _bindings = {
        'from tools.diagnostics import luna_semantic_probe_host as host': 'host',
        'from tools.diagnostics import luna_semantic_probe_run as runner': 'runner',
        'from tools.diagnostics import luna_semantic_probe_progress as reader': 'reader',
        'from tools.diagnostics import luna_semantic_probe_adapter_v2 as adapter': 'adapter',
        'from benchmarks import luna_semantic_canary as semantic': 'semantic',
    }
    for _line in _bindings:
        _source = _source.replace(_line, '')
    _source = _source.replace('source-grounding-v1', 'source-grounding-v2')
    _source = _source.replace('luna-semantic-probe', 'luna-semantic-policy-v2')
    _source = _source.replace('/private/tmp/hymem-semantic-step2-v2-candidate-20260929', str(CANDIDATE))
    _source = _source.replace('/private/tmp/hymem-semantic-step2-v2-map-20260929.json', str(INVENTORY))
    _source = _source.replace('ROOT = Path(__file__).resolve().parents[3]', 'ROOT = _BUNDLE_CODE')
    _source = _source.replace('repo=Path(__file__).resolve().parents[3]', 'repo=_BUNDLE_CODE')
    _namespace = {'__name__': __name__ + '.' + _path.stem, '__file__': str(_path),
        '_BUNDLE_CODE': CODE, 'host': HOST, 'runner': RUNNER, 'reader': READER,
        'adapter': ADAPTER, 'semantic': SEMANTIC}
    exec(compile(_source, str(_path), 'exec'), _namespace)
    for _name, _value in _namespace.items():
        if _name.startswith('test_') and callable(_value):
            globals()['test_rebound_' + _path.stem + '_' + _name[5:]] = _value
        elif _name in ('sealed', 'healthy_report'):
            globals()[_name] = _value
