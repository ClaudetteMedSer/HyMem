"""Offline accounting against the frozen LME candidate, with no provider calls."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
FROZEN = Path('/private/tmp/hymem-lme-diagnostic-offline-assembly-v3')


def load(name):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('luna_lme_diagnostic_v3.py')
launcher = load('luna_lme_diagnostic_launch_v3.py')
reader = load('luna_lme_diagnostic_progress_v4.py')
assembler = load('luna_lme_diagnostic_bundle_v3.py')
preflight = load('luna_lme_diagnostic_host_preflight_v3.py')


class Budget:
    def __init__(self):
        self.turns = self.tokens = 0
        self.halted = None

    def snapshot(self):
        return {'questions': {'q': {'turns': self.turns, 'known_tokens': self.tokens}}}

    def halt(self, reason):
        self.halted = reason


class Delegate:
    def __init__(self, answer='[]', error=False):
        self.budget = Budget()
        self.key = 'q'
        self.answer = answer
        self.error = error
        self.called = 0

    def complete(self, request):
        self.called += 1
        self.budget.turns += 1
        self.budget.tokens += 7
        if self.error:
            raise RuntimeError('provider_failure')
        return self.answer

    def complete_stage(self, request, batch, stage, recheck):
        return self.complete(request)


def test_unknown_and_invalid_staged_fail_before_dispatch():
    delegate = Delegate()
    client = runner.AccountedClient(delegate, FROZEN / 'candidate')
    with pytest.raises(RuntimeError, match='stage_accounting_failure'):
        client.complete(object())
    assert delegate.called == 0
    assert delegate.budget.halted == 'stage_accounting_failure'
    with pytest.raises(RuntimeError, match='stage_accounting_failure'):
        client.complete_stage(object(), object(), 'original', False)
    assert delegate.called == 0


def test_error_settlement_and_hidden_mismatch():
    delegate = Delegate(error=True)
    client = runner.AccountedClient(delegate, FROZEN / 'candidate')
    # Reader/judge are exercised via their actual frozen entry points below;
    # this local failure path checks the same accounted dispatch settlement.
    def ordinary_call():
        return client._call('reader', lambda: delegate.complete(object()))
    with pytest.raises(RuntimeError, match='provider_failure'):
        ordinary_call()
    assert client.counts['reader'] == {'attempts': 1, 'returned': 0,
        'turns': 1, 'known_tokens': 7}
    assert client.reconcile()
    delegate.budget.tokens += 1
    assert not client.reconcile()


def test_versioned_source_policy_and_bundle(tmp_path):
    assert runner.MAX_LIMITS == {'campaign': (8012, 48_160_000, 14_400),
        'question': (2000, 12_000_000, 12_600), 'canary': (12, 160_000, 600)}
    assert launcher.RUNNER_SHA256 == reader.RUNNER_SHA256 == assembler.RUNNER_SHA256 == preflight.RUNNER_SHA
    assert launcher.RUNNER_SHA256 == runner._sha(ROOT / 'luna_lme_diagnostic_v3.py')
    command = launcher.command(tmp_path, {'unit': 'hymem-luna-lme-diagnostic-test.service'}, 'a'*64)
    assert '--property=TasksMax=256' in command
    assert command[command.index('--workers') + 1] == '4'
    assert 'luna_lme_diagnostic_v3.py' in ' '.join(command)
    assert reader.PINS['tools/diagnostics/luna_lme_diagnostic_v3.py'] == launcher.RUNNER_SHA256
    output = tmp_path / 'bundle'
    manifest = assembler.assemble(repo=REPO, accepted_code=FROZEN / 'code',
        candidate=FROZEN / 'candidate', map_path=FROZEN / 'source-map.json', output=output)
    assert manifest['schema'] == 'luna-lme-diagnostic-source-bundle-v3'
    assert manifest['candidate_files'] == 514 and manifest['model_calls'] == 0
    assert len(preflight.source_manifest(output)) == 524
    remote = ast.parse(preflight.REMOTE)
    remote_constants = {target.id: ast.literal_eval(node.value)
        for node in remote.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
        and target.id in {'RUNNER', 'RUNNER_SHA'}}
    assert remote_constants == {'RUNNER': 'code/tools/diagnostics/luna_lme_diagnostic_v3.py',
        'RUNNER_SHA': runner._sha(ROOT / 'luna_lme_diagnostic_v3.py')}
    compile(preflight.wrapped_remote(), '<host-preflight-v3>', 'exec')


def test_actual_frozen_candidate_accounting():
    if not (FROZEN / 'candidate').is_dir():
        pytest.skip('accepted source-only frozen candidate unavailable')
    script = Path(__file__).with_name('verify_lme_attribution_candidate_v3.py')
    result = subprocess.run([sys.executable, '-B', str(script)], capture_output=True,
        text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
