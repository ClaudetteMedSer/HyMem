"""Registry CLI must work without an editable install or inherited Python path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run(argv, *, cwd, home):
    return subprocess.run([sys.executable, '-B', *argv], cwd=cwd,
        env={'PATH': os.defpath, 'HOME': str(home), 'PYTHONDONTWRITEBYTECODE': '1'},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)


@pytest.mark.parametrize('isolated', [False, True])
def test_direct_cli_help_without_site_packages_from_unrelated_cwd(tmp_path, isolated):
    args = ['-I', '-S'] if isolated else ['-E', '-s', '-S']
    result = run([*args, str(ROOT / 'benchmarks/lme_registry.py'), '--help'], cwd=tmp_path, home=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'LongMemEval' in result.stdout
    assert not list(tmp_path.rglob('*.db')) and not list(tmp_path.rglob('*.sqlite'))


def test_package_cli_help_without_site_packages(tmp_path):
    result = run(['-E', '-s', '-S', '-m', 'benchmarks.lme_registry', '--help'], cwd=ROOT, home=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'LongMemEval' in result.stdout


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('exception', ['ImportError', 'ValueError'])
def test_internal_import_failure_is_not_masked_by_fallback(tmp_path, direct, exception):
    # Fail only the registry's package-relative strictness import. A broad
    # exception fallback would incorrectly proceed via top-level strictness.
    code = '''import builtins,json,runpy,sys
root,direct,kind=sys.argv[1:]
sys.path.insert(0,root)
sys.path.insert(1,root+'/benchmarks')
original=builtins.__import__
attempts=[]
def injected(name,globals=None,locals=None,fromlist=(),level=0):
    attempts.append([name,level])
    if name=='strictness' and level==1:
        raise (ImportError if kind=='ImportError' else ValueError)('internal_dependency_sentinel')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=injected
sys.argv=[root+'/benchmarks/lme_registry.py','--help']
try:
    if direct=='true': runpy.run_path(sys.argv[0],run_name='__main__')
    else: runpy.run_module('benchmarks.lme_registry',run_name='__main__')
except (ImportError,ValueError) as error:
    assert str(error)=='internal_dependency_sentinel'
    assert not any(name=='strictness' and level==0 for name,level in attempts)
    print(json.dumps({'error_type':type(error).__name__,'sentinel_preserved':True}))
else:
    raise AssertionError('internal dependency failure was masked')
'''
    result = run(['-I', '-S', '-c', code, str(ROOT), str(direct).lower(), exception], cwd=tmp_path, home=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'error_type': exception, 'sentinel_preserved': True}
