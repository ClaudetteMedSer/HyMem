"""Root tests for the zero-inference input rehearsal; synthetic local evidence."""
import builtins
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

from tools.diagnostics import luna_staged_rehearsal_root as rehearsal
from tools.diagnostics import luna_staged_run_v1 as entry, luna_staged_core_v1 as core
from tools.diagnostics.tests.test_luna_staged_run_v1_root import setup_entry
from tools.diagnostics.tests.test_luna_staged_core_v1 import _canary_batches


@pytest.mark.parametrize('fault', ['none', 'entry_pin', 'launched', 'run_exists', 'original_changed'])
def test_synthetic_rehearsal_frozen_inputs_no_disk_output(monkeypatch, tmp_path, fault):
    root, host, reference, writes = setup_entry(monkeypatch, tmp_path)
    batches = _canary_batches()
    def derive(**kw):
        for i in range(8):
            kw['record'](dict(phase='ordinary_replay', ordinal=i))
        return batches[::-1] if fault == 'original_changed' else batches
    monkeypatch.setattr(core, 'derive_canary_batches', derive)
    monkeypatch.setattr(builtins, '_staged_rehearsal_test_preflight', entry.preflight, raising=False)
    path = root / 'code/tools/diagnostics/luna_staged_run_v1.py'
    path.parent.mkdir(parents=True)
    path.write_text('import builtins\npreflight=builtins._staged_rehearsal_test_preflight\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if fault == 'entry_pin':
        digest = '0'*64
    if fault == 'launched':
        (root / 'launch-attempt.json').write_text('{}')
    if fault == 'run_exists':
        (root / 'run').mkdir()
    if fault != 'none':
        with pytest.raises((ValueError, AssertionError, core.DiagnosticStop)):
            rehearsal.verify(root, '1'*64, digest)
    else:
        out = rehearsal.verify(root, '1'*64, digest)
        assert out['verified'] and out['model_calls'] == out['files_written'] == 0
        assert out['synthetic_judgments'] == 16 and out['retained_ordinary_replays'] == 8
        assert out['exact_bound_batch_pairs'] == 8 and not out['semantic_model_accuracy_measured']
    assert writes == []


def test_cli_invalid_path_is_finite_without_side_effects(tmp_path):
    out = subprocess.run([sys.executable, '-I', '-B', rehearsal.__file__,
        '--root', str(tmp_path/'PRIVATE_MISSING'), '--receipt-sha256', '0'*64,
        '--entry-sha256', '0'*64], capture_output=True, text=True, timeout=10)
    assert out.returncode == 1 and out.stderr == '' and 'PRIVATE_' not in out.stdout
    value = json.loads(out.stdout)
    assert value['verified'] is False and value['model_calls'] == 0


def test_physical_candidate_requests_rehearse_in_isolated_process(tmp_path):
    from tools.diagnostics import luna_staged_bundle_v1 as bundle
    repo = Path(__file__).resolve().parents[3]
    candidate = tmp_path/'candidate'
    shutil.copytree(bundle.CANDIDATE, candidate)
    code = tmp_path/'code'
    for name, raw in bundle.collect_sources(repo).items():
        path = code/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    # Reuse the reviewed invented ordinary-response fixture, not production data.
    tree = ast.parse(Path(__file__).with_name('test_luna_staged_core_v1_root.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
        n.name == 'test_actual_candidate_derivation_preserves_initial_batches_and_exact_ordinary_bytes')
    script = next(ast.literal_eval(n.value) for n in function.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'script' for t in n.targets))
    script += r'''
import builtins,hashlib,importlib.util,tempfile,subprocess
from types import SimpleNamespace
from tools.diagnostics import luna_semantic_cases as cases
subprocess.Popen=deny
root=Path(tempfile.mkdtemp(prefix='staged-rehearsal-physical-root-'))
(root/'candidate').symlink_to(Path(sys.argv[1]),target_is_directory=True)
loaded=(SimpleNamespace(),c,probe,cases,gold,chunk,None,c.transport,
        c.transport.warm,c.transport.warm.concurrent)
builtins._root_physical_preflight=lambda *a:({},loaded,dict(candidate_files=514),tuple(fake.retained))
shim=root/'code/tools/diagnostics/luna_staged_run_v1.py';shim.parent.mkdir(parents=True)
shim.write_text('import builtins\npreflight=builtins._root_physical_preflight\n')
spec=importlib.util.spec_from_file_location('root_private_rehearsal',sys.argv[3])
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
result=helper.verify(root,'1'*64,hashlib.sha256(shim.read_bytes()).hexdigest())
assert result['verified'] and result['model_calls']==result['files_written']==0
assert result['retained_ordinary_replays']==8 and result['synthetic_judgments']==16
print(json.dumps(result,sort_keys=True))
'''
    out = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(candidate),
        str(code), rehearsal.__file__], capture_output=True, text=True, timeout=45)
    assert out.returncode == 0, out.stderr
    value = json.loads(out.stdout.splitlines()[-1])
    assert value['verified'] and value['candidate_files'] == 514
    assert value['model_calls'] == value['files_written'] == 0
