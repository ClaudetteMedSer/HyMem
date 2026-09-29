"""No-SSH controls for the detached, validation-only R7 finalizer."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


HELPER = Path(__file__).parents[1] / 'lme_r7_headless_finalizer.py'
LIVE = 'a' * 64
VALIDATION = 'b' * 64


@pytest.fixture
def finalizer():
    spec = importlib.util.spec_from_file_location('r7_finalizer_under_test', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controller_cannot_start_paid_live_container(finalizer, monkeypatch):
    monkeypatch.setattr(finalizer.subprocess, 'run', lambda *_a, **_k: pytest.fail('subprocess forbidden'))
    with pytest.raises(RuntimeError, match='forbidden_finalizer_action'):
        finalizer.controller('start', 'live', LIVE)
    with pytest.raises(RuntimeError, match='forbidden_finalizer_action'):
        finalizer.controller('create', 'live')


def test_failed_live_run_still_gets_one_offline_validation(finalizer, monkeypatch):
    calls, reports = [], []
    monkeypatch.setattr(finalizer, 'checked_remote', lambda: None)
    monkeypatch.setattr(finalizer, 'owned_live', lambda cid: calls.append(('owned', cid)))
    monkeypatch.setattr(finalizer, 'read', lambda _path: (
        ('{"live_container_id":"' + LIVE + '","manifest_sha256":"' + finalizer.PIN +
         '","validation_runs_allowed":1,"new_paid_runs_allowed":0}').encode()))

    def controller(action, mode, cid=None):
        calls.append((action, mode, cid))
        if (action, mode) == ('status', 'live'):
            return {'status': 'exited', 'pid': 0, 'exit_code': 1}
        if (action, mode) == ('create', 'validation'):
            return {'status': 'created', 'container_id': VALIDATION}
        if (action, mode) == ('start', 'validation'):
            return {'status': 'running'}
        return {'status': 'exited', 'pid': 0, 'exit_code': 0}

    monkeypatch.setattr(finalizer, 'controller', controller)
    monkeypatch.setattr(finalizer, 'docker_wait', lambda cid, _timeout: 1 if cid == LIVE else 0)
    monkeypatch.setattr(finalizer, 'receipt', lambda name, value: reports.append((name, value)))
    finalizer.worker(LIVE)
    assert [step for step in calls if step[0] in ('create', 'start')] == [
        ('create', 'validation', None), ('start', 'validation', VALIDATION)]
    assert reports[0][0] == 'finalizer-result.json'
    result = reports[0][1]
    assert result['status'] == 'offline_validation_finished'
    assert result['live_exit_code'] == 1 and result['validation_exit_code'] == 0
    assert result['new_paid_runs_started'] == 0 and result['score_verified'] is False


def test_live_wait_timeout_never_creates_validation(finalizer, monkeypatch):
    calls, reports = [], []
    monkeypatch.setattr(finalizer, 'checked_remote', lambda: None)
    monkeypatch.setattr(finalizer, 'owned_live', lambda _cid: None)
    monkeypatch.setattr(finalizer, 'read', lambda _path: (
        ('{"live_container_id":"' + LIVE + '","manifest_sha256":"' + finalizer.PIN +
         '","validation_runs_allowed":1,"new_paid_runs_allowed":0}').encode()))
    monkeypatch.setattr(finalizer, 'controller', lambda *args: calls.append(args))
    monkeypatch.setattr(finalizer, 'docker_wait', lambda *_args: (_ for _ in ()).throw(
        subprocess.TimeoutExpired(['docker', 'wait'], 9 * 3600 + 60)))
    monkeypatch.setattr(finalizer, 'receipt', lambda name, value: reports.append((name, value)))
    finalizer.worker(LIVE)
    assert calls == []
    assert reports[0][1]['status'] == 'outcome_requires_inspection'
    assert reports[0][1]['error_type'] == 'TimeoutExpired'
    assert reports[0][1]['validation_container_id'] is None


def test_duplicate_launch_does_not_spawn_second_worker(finalizer, monkeypatch, tmp_path):
    monkeypatch.setattr(finalizer, 'ROOT', tmp_path)
    monkeypatch.setattr(finalizer, 'checked_remote', lambda: None)
    monkeypatch.setattr(finalizer, 'owned_live', lambda _cid: None)
    monkeypatch.setattr(finalizer, 'controller', lambda *_args: {'status': 'running'})
    monkeypatch.setattr(finalizer, 'receipt', lambda *_args: (_ for _ in ()).throw(FileExistsError()))
    monkeypatch.setattr(finalizer.subprocess, 'Popen', lambda *_a, **_k: pytest.fail('duplicate worker'))
    with pytest.raises(FileExistsError):
        finalizer.remote_launch(LIVE)


def test_launch_is_detached_without_credentials_or_paid_action(finalizer, monkeypatch, tmp_path):
    monkeypatch.setattr(finalizer, 'ROOT', tmp_path)
    monkeypatch.setattr(finalizer, 'checked_remote', lambda: None)
    monkeypatch.setattr(finalizer, 'owned_live', lambda _cid: None)
    monkeypatch.setattr(finalizer, 'controller', lambda *_args: {'status': 'running'})
    writes, starts = [], []
    monkeypatch.setattr(finalizer, 'receipt', lambda name, value: writes.append((name, value)))

    def popen(command, **kwargs):
        starts.append((command, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(finalizer.subprocess, 'Popen', popen)
    result = finalizer.remote_launch(LIVE)
    assert result['status'] == 'detached_finalizer_started'
    assert [name for name, _ in writes] == ['finalizer-intent.json', 'finalizer-launch.json']
    assert writes[0][1]['new_paid_runs_allowed'] == 0
    assert len(starts) == 1 and starts[0][1]['start_new_session'] is True
    assert starts[0][1]['stdout'] is subprocess.DEVNULL
    assert starts[0][1]['stderr'] is subprocess.DEVNULL
    assert starts[0][0][-2:] == ['worker', LIVE]
