"""Independent runner failure controls; no network or provider invocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / 'luna_subscription_canary.py'
    spec = importlib.util.spec_from_file_location('root_canary_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def harness(runner, tmp_path, monkeypatch, *, calls=2, fails_at=None):
    processes = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.process = SimpleNamespace(pid=900000 + len(processes),
                poll=lambda: 0 if self.closed else None)
            processes.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(runner, '_group_absent',
        lambda pid: processes[pid - 900000].closed)

    class Client:
        def __init__(self, binary, session_factory):
            self.session_factory = session_factory
            self.observed_turns = 0
            self.observed_tokens = None
            self.usage_complete = True

        def preflight(self):
            session = self.session_factory('/unused', '/unused')
            session.close()
            return dict(config_isolation_admitted=True, auth='chatgpt',
                        model='gpt-6-luna', inference_enabled=False)

        def complete(self, request):
            session = self.session_factory('/unused', '/unused')
            try:
                self.observed_turns += 1
                if self.observed_turns == fails_at:
                    self.usage_complete = False
                    raise RuntimeError('deliberate private transport failure')
                self.observed_tokens = (self.observed_tokens or 0) + 17
                return '{}'
            finally:
                session.close()

    def gate(canary, chunk, client, evidence):
        for _ in range(calls):
            client.complete(SimpleNamespace(system='synthetic', user='synthetic',
                temperature=0.0, max_tokens=32, response_format='json'))
        return dict(schema='luna-experimental-canary-v2', passed=True)

    report = dict(ok=False, stop_code=None)
    kwargs = dict(pilot=SimpleNamespace(experimental_canary=gate),
        transport=SimpleNamespace(StdioSession=Session, CodexSubscriptionClient=Client),
        canary=None, chunk=None, binary='/unused', output=tmp_path,
        deadline=runner.time.monotonic() + 30, report=report)
    return kwargs, report, processes


def test_runner_failure_retains_partial_usage(runner, tmp_path, monkeypatch):
    kwargs, report, processes = harness(runner, tmp_path, monkeypatch, fails_at=2)
    with pytest.raises(RuntimeError):
        runner.run_canary(**kwargs)
    assert report['ok'] is False
    assert report['known_tokens'] == 17
    assert report['observed_turns'] == 2
    assert report['usage_complete'] is False
    assert report['app_server_cleanup_complete'] is True
    assert all(p.closed for p in processes)


def test_runner_rejects_twenty_fifth_call(runner, tmp_path, monkeypatch):
    kwargs, report, processes = harness(runner, tmp_path, monkeypatch, calls=25)
    with pytest.raises(runner.CanaryStop, match='call_cap'):
        runner.run_canary(**kwargs)
    assert report['ok'] is False
    assert report['attempted_calls'] == report['observed_turns'] == 24
    assert report['known_tokens'] == 408
    assert len(processes) == 25  # preflight plus exactly 24 invocations
    assert all(p.closed for p in processes)


def test_session_journal_failure_closes_created_child(runner, tmp_path, monkeypatch):
    kwargs, report, processes = harness(runner, tmp_path, monkeypatch)
    def fail_journal(*args, **kwargs):
        raise OSError('simulated disk full')
    monkeypatch.setattr(runner, '_journal', fail_journal)
    with pytest.raises(Exception):
        runner.run_canary(**kwargs)
    assert len(processes) == 1
    assert processes[0].closed
    assert report['ok'] is False


def test_final_progress_failure_cannot_leave_success(runner, tmp_path, monkeypatch):
    kwargs, report, processes = harness(runner, tmp_path, monkeypatch)
    save = runner._private_json
    def fail_finished(path, value):
        if value.get('phase') == 'finished':
            raise OSError('simulated disk full')
        return save(path, value)
    monkeypatch.setattr(runner, '_private_json', fail_finished)
    with pytest.raises(Exception):
        runner.run_canary(**kwargs)
    assert report['ok'] is False
    assert report['known_tokens'] == 34
    assert report['observed_turns'] == 2
    assert all(p.closed for p in processes)
