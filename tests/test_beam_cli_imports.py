"""BEAM entry points share checkout-local LME helpers without editable hooks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[1]
_PYTHON = [sys.executable, "-E", "-s", "-S", "-B"]
# Expose installed dependencies without running site/.pth editable-install hooks.
# In particular, use the real requests package; this test must not supply a stub.
_DEPENDENCY_PATHS = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})


def _run(code, *, cwd, dependencies=True):
    bootstrap = """
import sys
assert 'site' not in sys.modules
def forbid_network(event, args):
    if event in {'socket.connect', 'socket.sendto'}:
        raise AssertionError('BEAM import/CLI checks must not access the network')
sys.addaudithook(forbid_network)
"""
    if dependencies:
        bootstrap += f"sys.path.extend({_DEPENDENCY_PATHS!r})\n"
    return subprocess.run(
        [*_PYTHON, "-c", textwrap.dedent(bootstrap) + textwrap.dedent(code)],
        cwd=cwd,
        env={key: value for key, value in os.environ.items()
             if not key.startswith(("HYMEM_", "OPENAI_", "DEEPSEEK_"))},
        capture_output=True, text=True, timeout=60,
    )


@pytest.fixture(scope="module")
def clean_beam_checkout(tmp_path_factory):
    work = tmp_path_factory.mktemp("beam-checkout")
    checkout = work / "checkout"
    outside = work / "unrelated-cwd"
    outside.mkdir()
    for package in ("benchmarks", "hymem"):
        for source in (_REPO / package).rglob("*"):
            if source.is_file() and source.suffix in {".py", ".sql"}:
                destination = checkout / source.relative_to(_REPO)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    probe = _run("""
        import importlib.util
        assert importlib.util.find_spec('hymem') is None
        assert importlib.util.find_spec('benchmarks') is None
        import requests
        assert requests.__file__ and requests.__version__
        assert not any(name.startswith('__editable__') for name in sys.modules)
    """, cwd=outside)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    return checkout, outside


def test_beam_cli_help_matches_across_entry_points(clean_beam_checkout):
    checkout, outside = clean_beam_checkout
    outputs = []
    for invocation in ("absolute", "relative", "module"):
        if invocation == "module":
            code = """
                import runpy
                sys.argv = ['benchmarks.beam_adapter', '--help']
                runpy.run_module('benchmarks.beam_adapter', run_name='__main__', alter_sys=True)
            """
        else:
            target = str(checkout / "benchmarks/beam_adapter.py") if invocation == "absolute" else "benchmarks/beam_adapter.py"
            code = f"""
                import runpy
                from pathlib import Path
                sys.path.insert(0, str(Path({target!r}).resolve().parent))
                sys.argv = [{target!r}, '--help']
                runpy.run_path({target!r}, run_name='__main__')
            """
        result = _run(code, cwd=outside if invocation == "absolute" else checkout)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "HyMem BEAM Benchmark (direct API)" in result.stdout
        assert result.stderr == ""
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] == outputs[2]


def test_beam_imports_share_helpers_and_code_identity(clean_beam_checkout):
    checkout, _ = clean_beam_checkout
    identities = []
    for name, cwd in (("benchmarks.beam_adapter", checkout), ("beam_adapter", checkout / "benchmarks")):
        result = _run(f"""
            import importlib
            import json
            from pathlib import Path
            beam = importlib.import_module({name!r})
            from benchmarks import longmemeval_adapter as lme, lme_protocol, strictness
            import hymem
            import requests
            root = Path({str(checkout)!r})
            assert Path(beam.__file__).resolve() == root / 'benchmarks/beam_adapter.py'
            assert Path(lme.__file__).resolve() == root / 'benchmarks/longmemeval_adapter.py'
            assert Path(lme_protocol.__file__).resolve() == root / 'benchmarks/lme_protocol.py'
            assert Path(hymem.__file__).resolve() == root / 'hymem/__init__.py'
            assert 'longmemeval_adapter' not in sys.modules
            symbols = ('PINNED_DEEPSEEK_MODEL', 'THINKING_DISABLED', '_detect_ability', '_detect_ability_safe', '_render_answer_context')
            for symbol in symbols:
                assert getattr(beam, symbol) is getattr(lme, symbol), symbol
            assert beam.BenchmarkIntegrityError is lme.BenchmarkIntegrityError is strictness.BenchmarkIntegrityError
            assert beam.http is lme.http is requests
            assert set(strictness.python_file_imported_symbols(
                Path(beam.__file__), module_names=('benchmarks.longmemeval_adapter',)
            )) == set(symbols)
            print(json.dumps({{'code': beam.beam_code_hash(), 'judge_prompt': beam.BEAM_OFFICIAL_JUDGE_PROMPT_HASH}}))
        """, cwd=cwd)
        assert result.returncode == 0, result.stdout + result.stderr
        identities.append(json.loads(result.stdout))
    assert identities[0] == identities[1]


def test_beam_does_not_mask_missing_real_dependency(clean_beam_checkout):
    checkout, _ = clean_beam_checkout
    result = _run("""
        import importlib.util
        assert importlib.util.find_spec('requests') is None
        try:
            import benchmarks.beam_adapter
        except ModuleNotFoundError as error:
            assert error.name == 'requests', error
        else:
            raise AssertionError('Missing requests must remain an import error')
        assert 'longmemeval_adapter' not in sys.modules
    """, cwd=checkout, dependencies=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_beam_code_identity_tracks_the_actual_imported_lme_helper(clean_beam_checkout):
    checkout, _ = clean_beam_checkout
    result = _run("""
        from pathlib import Path
        from benchmarks import beam_adapter as beam, longmemeval_adapter as lme
        source_path = Path(lme.__file__)
        original = source_path.read_text(encoding='utf-8')
        before = beam.beam_code_hash()
        try:
            source_path.write_text(
                original + '\\ndef unused_beam_import_probe():\\n    return 1\\n',
                encoding='utf-8',
            )
            assert beam.beam_code_hash() == before
            marker = '    return detected\\n'
            assert original.count(marker) == 1
            source_path.write_text(
                original.replace(marker, '    return None\\n'), encoding='utf-8',
            )
            assert beam.beam_code_hash() != before
        finally:
            source_path.write_text(original, encoding='utf-8')
    """, cwd=checkout)
    assert result.returncode == 0, result.stdout + result.stderr
