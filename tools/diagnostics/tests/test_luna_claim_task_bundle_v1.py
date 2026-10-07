"""Offline bundle and terminal boundaries; no provider or host dispatch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_claim_task_bundle_v1 as bundle
from tools.diagnostics import luna_claim_task_core_v1 as core
from tools.diagnostics import luna_claim_task_run_v1 as entry
from tools.diagnostics import luna_claim_task_progress_v1 as progress
from tools.diagnostics.tests.test_luna_claim_task_core_v1 import FakeClient, _canary_batches
from tools.diagnostics import luna_semantic_cases as cases
from benchmarks import codex_subscription_claim_task_v1 as transport

REPO = Path(__file__).resolve().parents[3]


def test_immutable_bundle_derives_new_family_and_exact_caps(tmp_path):
    target = tmp_path / 'bundle'
    proof = bundle.prepare(REPO, target)
    receipt = json.loads((target / 'derivation-receipt.json').read_bytes())
    assert proof['model_calls'] == 0 and proof['launched'] is False
    assert receipt['schema'] == 'luna-claim-task-probe-bundle-v1'
    assert receipt['candidate_files'] == 513
    for name, digest in receipt['output_sha256'].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
    host = (target / 'code/tools/diagnostics/luna_semantic_probe_host.py').read_text()
    startup = (target / 'adapter-v2.py').read_text()
    transport = (target / 'code/benchmarks/codex_subscription_claim_task_v1.py').read_text()
    assert "'units': 29" in host
    assert "'control_turns': 1" in host
    assert "'memory_max_bytes': 4294967296" in host
    assert "'runtime_max_seconds': 1930" in host
    assert "'quota_floor_percent': 25" in host
    assert "'model': 'gpt-6-luna'" in host
    assert 'luna_claim_task_run_v1.py' in startup
    assert "result['diagnostic_completed']" in startup
    assert bundle.sha((target / 'code/benchmarks/codex_subscription_classification_v3.py').read_bytes()) in transport
    assert 'd3c955922219d99b359aa6619b821e7da1662e36ecea54d09b1267dce38d2c32' not in transport


def test_source_pin_and_candidate_validation_fail_closed(monkeypatch, tmp_path):
    wrong = dict(bundle.FROZEN)
    wrong['tools/diagnostics/luna_claim_task_core_v1.py'] = '0' * 64
    monkeypatch.setattr(bundle, 'FROZEN', wrong)
    with pytest.raises(ValueError, match='source_pin_invalid'):
        bundle.prepare(REPO, tmp_path / 'wrong-source')
    from tools.diagnostics import luna_classification_bundle_v3 as v3
    monkeypatch.setattr(v3, 'CANDIDATE', tmp_path / 'missing-candidate')
    with pytest.raises(ValueError):
        v3.validate_candidate()


def test_checked_bindings_reject_drift():
    with pytest.raises(ValueError, match='binding_count_invalid'):
        bundle.one(b'prefix', 'missing', 'replacement')
    with pytest.raises(ValueError, match='binding_count_invalid'):
        bundle.one(b'old old', 'old', 'new')


def test_terminal_projection_has_no_semantic_success_or_raw_text():
    # The accepted core validator rejects made-up public fields before export.
    with pytest.raises(core.DiagnosticStop):
        entry._safe_result({'schema': core.SCHEMA, 'private_raw': 'secret'},
                           '0' * 64, '1' * 64, '2' * 64, True, 1.0)
    assert progress._terminal_valid({'schema': 'luna-claim-task-probe-terminal-v1',
        'raw_response': 'secret'}, {}, '0' * 64, '1' * 64) is False


def test_replay_and_reader_are_read_only_by_contract():
    text = (REPO / 'tools/diagnostics/luna_claim_task_replay_v1.py').read_text()
    assert 'new_model_calls' in text and 'sys.addaudithook(audit)' in text
    assert 'parse_arm_response' in text and 'output_schema_sha256' in text
    reader = (REPO / 'tools/diagnostics/luna_claim_task_progress_v1.py').read_text()
    assert "owned['count'] == 29" in reader
    assert 'replay.replay(root, receipt_sha' in reader
    assert "'semantic_accuracy_accepted': False" in reader


def _fake_entry(monkeypatch, tmp_path):
    root = tmp_path / 'private-root'
    root.mkdir(mode=0o700)
    def write_once(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, sort_keys=True)
    host = SimpleNamespace(BINARY=Path('/no-binary'), write_once=write_once)
    receipt = {'source_sha256': {'fixture': '0' * 64}}
    loaded = (host, core, object(), cases, object(), object(), object(),
              transport, transport.warm, transport.warm.concurrent)
    monkeypatch.setattr(entry, 'preflight', lambda *_: (receipt, loaded,
        {'candidate_files': 513}, tuple(range(8))))
    monkeypatch.setattr(entry, '_reference', lambda *_: SimpleNamespace(
        _owned_absent=lambda _: True))
    monkeypatch.setattr(core, 'derive_canary_batches', lambda **_: _canary_batches())
    return root


def test_entry_with_fake_transport_runs_all_29_and_keeps_malformed_separate(monkeypatch, tmp_path):
    root = _fake_entry(monkeypatch, tmp_path)
    result = entry.execute(root, 'a' * 64, containment=lambda *_: None,
        client_factory=lambda key, cap, budget: FakeClient(key, cap, budget,
            malformed_at='unit-03'))
    assert result['diagnostic_completed'] is True
    assert result['completed_units'] == result['attempted_units'] == 29
    assert result['malformed_units'] == 1
    assert result['paid_budget']['turns'] == 29
    assert result['completed_and_clean'] is False
    assert result['semantic_accuracy_accepted'] is False
    assert result['full_lme_ready'] is False
    assert 'PRIVATE' not in (root / 'safe-terminal.json').read_text()
    private = json.loads((root / 'run/private-result.json').read_bytes())
    assert private['units'][3]['outcome'] == 'malformed'


def test_entry_default_factory_selects_claim_task_transport(monkeypatch, tmp_path):
    root = _fake_entry(monkeypatch, tmp_path)
    called = []
    def fake_adapter(binary, budget, key, cap, **kwargs):
        called.append((binary, key, cap))
        return FakeClient(key, (cap.turns, cap.known_tokens,
                                cap.seconds), budget)
    monkeypatch.setattr(transport, 'ClaimTaskSubscriptionClient', fake_adapter)
    result = entry.execute(root, 'a' * 64, containment=lambda *_: None)
    assert result['diagnostic_completed'] is True
    assert len(called) == 29
    assert all(item[0] == '/no-binary' for item in called)


def test_generated_startup_containment_bridge_uses_receipt_hash(monkeypatch, tmp_path):
    target = tmp_path / 'bundle'
    bundle.prepare(REPO, target)
    source = target / 'adapter-v2.py'
    spec = importlib.util.spec_from_file_location('claim_task_generated_startup_test', source)
    startup = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(startup)
    root = tmp_path / 'runtime'
    root.mkdir()
    (root / 'launch-receipt.json').write_text('{}')
    receipt = {'unit': 'fixed.service'}
    calls = []
    def original(args, **kwargs):
        calls.append((args, kwargs))
        return object()
    reference = SimpleNamespace(subprocess=SimpleNamespace(run=original))
    def verify(root_arg, receipt_arg):
        reference.subprocess.run(['/usr/bin/systemctl', '--user', 'show',
            'fixed.service', '--property=MainPID', '--no-pager'])
    reference.verify_live_containment = verify
    def reference_loader(root_arg, receipt_hash):
        assert root_arg == root
        assert receipt_hash == hashlib.sha256(b'{}').hexdigest()
        return reference
    monkeypatch.setattr(startup, 'bus_environment', lambda: {'BUS': 'scoped'})
    startup.contained(root, receipt, SimpleNamespace(_reference=reference_loader))
    assert calls == [(['/usr/bin/systemctl', '--user', 'show',
                      'fixed.service', '--property=MainPID', '--no-pager'],
                     {'env': {'BUS': 'scoped'}})]


def test_generated_startup_reader_accepts_exact_zero_call_failure(monkeypatch, tmp_path):
    target = tmp_path / 'bundle'
    bundle.prepare(REPO, target)
    spec = importlib.util.spec_from_file_location('claim_task_startup_observer_test',
                                                  target / 'adapter-v2.py')
    startup = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(startup)
    root = tmp_path / 'runtime'
    root.mkdir()
    receipt_sha = '1' * 64
    adapter_sha = '2' * 64
    sidecar_sha = '3' * 64
    receipt = {'unit': 'fixed.service', 'expected_cgroup': '/fixed',
               'source_sha256': {}, 'binary_sha256': '4' * 64}
    (root / 'launch-receipt.json').write_text(json.dumps(receipt))
    marker = {'receipt_sha256': receipt_sha, 'one_shot': True}
    admission = {'receipt_sha256': receipt_sha, 'old_luna_stopped': True,
        'old_deepseek_stopped': True, 'memory_floor_bytes': 6 * 1024**3,
        'disk_floor_bytes': 20 * 1024**3, 'memory_floor_met': True,
        'disk_floor_met': True}
    (root / 'launch-attempt.json').write_text(json.dumps(marker))
    (root / 'launch-admission.json').write_text(json.dumps(admission))
    terminal = {'schema': 'luna-claim-task-probe-startup-failure-v2',
        'receipt_sha256': receipt_sha, 'adapter_sha256': adapter_sha,
        'zero_admission_proof': 'run_directory_never_created',
        'paid_turns': 0, 'known_tokens': 0, 'model_calls': 0,
        'completed_and_clean': False, 'semantic_accuracy_accepted': False,
        'full_lme_ready': False, 'stop_code': 'pre_inference_startup_failure'}
    (root / 'safe-terminal.json').write_text(json.dumps(terminal))
    host = SimpleNamespace(root_valid=lambda _: True,
        strict_equal=lambda a, b: a == b,
        receipt_for=lambda *_: receipt,
        verify_bundle=lambda *_args, **_kwargs: None,
        regular=lambda path: path.is_file())
    monkeypatch.setattr(startup, 'verify', lambda *_: {'entry_action': 'inference'})
    monkeypatch.setattr(startup, 'load_source', lambda *_: host)
    monkeypatch.setattr(startup, 'bus_environment', lambda: {'BUS': 'scoped'})
    reader = SimpleNamespace(subprocess=SimpleNamespace(run=lambda *_a, **_k: None),
        _systemd=lambda *_: {'available': True, 'active_state': 'failed',
            'main_pid_zero': True, 'n_restarts_zero': True,
            'cgroup_empty': True, 'resource_policy_verified': True})
    out = startup.observe(root, receipt_sha, adapter_sha, sidecar_sha, reader)
    assert out['phase'] == 'startup_failed'
    assert out['zero_admission_verified'] is True
    assert out['failed_unit_cleanup_verified'] is True
    assert out['completed_and_clean'] is False
    assert out['semantic_accuracy_accepted'] is False
    assert out['full_lme_ready'] is False
