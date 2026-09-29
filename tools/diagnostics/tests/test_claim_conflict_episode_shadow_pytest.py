"""Local, credential-free controls for the new offline-suite adapter."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_episode_shadow_pytest as suite


def test_prepare_has_exact_code_only_overlay_and_refuses_rerun(tmp_path):
    output = tmp_path / 'prepared'
    result = suite.prepare(output)
    assert result['status'] == 'prepared_not_uploaded'
    assert result['test_overrides'] == 21
    assert result['expected_collected'] == 7764
    assert suite.digest(json.loads((output / 'test-overlay.json').read_text())) == suite.OVERLAY_SHA
    assert set(p.name for p in output.iterdir()) == {'overlay', 'test-overlay.json', 'reviewed-candidate.json', suite.SELF.name}
    assert suite.sha(output / suite.SELF.name) == suite.sha(Path(suite.__file__))
    with pytest.raises(RuntimeError, match='output_already_exists'):
        suite.prepare(output)


def test_overlay_preserves_eighteen_prior_pins_and_adds_three_exact_sources():
    sources, manifest = suite.test_sources()
    assert len(manifest) == 21
    assert all(sources['tests/' + name].parent == suite.TEST_ROOT for name in suite.NEW_TEST_NAMES)
    assert suite.digest({name: manifest[name] for name in ('tests/' + n for n in suite.PRIOR_TEST_NAMES)}) == suite.PRIOR_OVERLAY_SHA
    assert suite.digest(manifest) == suite.OVERLAY_SHA


def test_mutated_test_rejected_before_creating_destination(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, 'OVERLAY_SHA', '0' * 64)
    with pytest.raises(RuntimeError, match='test_source_pin_drift'):
        suite.prepare(tmp_path / 'absent')
    assert not (tmp_path / 'absent').exists()


def test_prepare_copy_failure_cleans_only_new_artifact(monkeypatch, tmp_path):
    prior = tmp_path / 'prior-receipt'
    prior.write_text('preserve')
    def fail(*_):
        raise OSError('copy failed')
    monkeypatch.setattr(suite.shutil, 'copyfile', fail)
    with pytest.raises(OSError):
        suite.prepare(tmp_path / 'partial')
    assert not (tmp_path / 'partial').exists()
    assert prior.read_text() == 'preserve'


def test_remote_review_gate_has_no_side_effects(monkeypatch):
    monkeypatch.setattr(suite, 'REVIEWED_REMOTE_EXECUTION', False)
    with pytest.raises(RuntimeError, match='episode_suite_root_review_pending'):
        suite.remote_install(SimpleNamespace())


def test_pin_checked_before_import_no_host_container_fallback(monkeypatch, tmp_path):
    bad = tmp_path / 'bad.py'
    bad.write_text('raise AssertionError("executed")')
    monkeypatch.setattr(suite, 'BASE_CONTAINER_SUITE', bad)
    with pytest.raises(RuntimeError, match='base_suite_pin_drift'):
        suite.load_base('container')
    monkeypatch.setattr(suite, 'BASE_SUITE', tmp_path / 'missing.py')
    with pytest.raises(RuntimeError, match='base_suite_pin_drift'):
        suite.load_base('host')
    assert suite.load_base('local').FULL_SECONDS == 5000


def test_configuration_is_source_only_network_none_and_versioned():
    base = suite.configure_base('local')
    helper = SimpleNamespace(RUNTIME=Path('/runtime'), IMAGE='sha256:' + 'a' * 64)
    command, mounts = base.configure(helper)
    assert command[command.index('--name') + 1] == 'hymem-episode-shadow-pytest-v1'
    assert command[command.index('--network') + 1] == 'none'
    assert '--read-only' in command and '--cap-drop' in command
    assert [(dest, writable) for _, dest, writable in mounts if writable] == [('/work', True)]
    assert set(dest for _, dest, _ in mounts) == {'/candidate', '/overlay', '/candidate-manifest.json', '/test-overlay.json', '/diag/runner.py', '/work', '/home/node/hymem-env', '/diag/base_suite.py'}
    assert str(suite.CANDIDATE) in [src for src, _, _ in mounts]
    assert base.ROOT == suite.ROOT and base.ROOT.name == 'episode-shadow-pytest-v1'
    assert base.FULL_SECONDS == 10800 and base.HOST_SECONDS == 11200


def test_manifest_requires_exact_application_inventory(monkeypatch):
    manifest = {**suite.APPLICATION_PINS, **{'source/%03d.py' % n: 'a' * 64 for n in range(479)}}
    monkeypatch.setattr(suite, 'CANDIDATE_SHA', suite.digest(manifest))
    assert suite.validate_candidate(manifest) == manifest
    manifest['hymem/core/db.py'] = 'b' * 64
    with pytest.raises(RuntimeError, match='episode_candidate_pin_drift'):
        suite.validate_candidate(manifest)


def test_prepare_binds_every_application_and_baseline_test_source(monkeypatch, tmp_path):
    manifest = json.loads(suite.LOCAL_MANIFEST.read_text())
    assert len(manifest) == 481 and suite.digest(manifest) == suite.CANDIDATE_SHA
    monkeypatch.setattr(suite, 'LOCAL_CANDIDATE', tmp_path)
    with pytest.raises(RuntimeError, match='local_candidate_source_pin_drift'):
        suite.prepare(tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


def test_exact_collection_required_before_full_run(monkeypatch, tmp_path):
    # Stub only the inherited child runner; exercise the adapter's boundary.
    pinned = suite.load_base('local')
    pinned.run_pytest = lambda _: 0
    pinned.CONTAINER_WORK = tmp_path
    monkeypatch.setattr(suite, 'load_base', lambda _: pinned)
    base = suite.configure_base('local')
    (tmp_path / 'collect-counts.json').write_text(json.dumps({'collected': 7765}))
    with pytest.raises(RuntimeError, match='episode_collection_count_drift'):
        base.run_pytest('collect')
    (tmp_path / 'collect-counts.json').write_text(json.dumps({'collected': 7764}))
    assert base.run_pytest('collect') == 0


def test_collection_and_one_run_receipt_accounting():
    base = suite.configure_base('local')
    counts = {'collected': 7764, 'passed': 7760, 'skipped': 4, 'failed': 0, 'errors': 0, 'exit_code': 0}
    raw = {'status': 'passed', 'candidate_sha256': suite.CANDIDATE_SHA,
           'test_inventory_sha256': suite.EFFECTIVE_TEST_SHA, 'application_unchanged': True,
           'pinned_source_unchanged': True, 'full_runs_started': 1,
           'collect': {**counts, 'passed': 0}, 'full': counts}
    assert base.project_worker(raw)['full']['passed'] == 7760
    raw['full_runs_started'] = 2
    with pytest.raises(RuntimeError, match='worker_run_count_invalid'):
        base.project_worker(raw)


def test_replay_gate_requires_exact_success_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, 'REPLAY_ROOT', tmp_path)
    path = tmp_path / 'result.json'
    result = {'status': 'verified', **{name: True for name in ('source_unchanged', 'exact_vectors_preserved', 'semantic_rows_unchanged', 'repeat_noop', 'reopen_aligned', 'health_clean')}}
    path.write_text(json.dumps(result))
    monkeypatch.setattr(suite, 'REPLAY_RESULT_SHA', suite.sha(path))
    base = suite.configure_base('local')
    helper = SimpleNamespace(read_json=lambda path: json.loads(path.read_text()))
    assert base.replay_gate(None, helper) == suite.sha(path)
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='episode_replay_receipt_pin_drift'):
        base.replay_gate(None, helper)
