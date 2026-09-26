"""Public command safety; worker mechanics have independent regressions."""
import json
from types import SimpleNamespace

import pytest

from hymem import bootstrap, recover_summaries
from hymem.core import db
from hymem.dreaming.summary_state import record_summary_failure
from tests.test_summary_frontier_v62 import _published


def _environment(monkeypatch, root):
    monkeypatch.setattr(bootstrap, 'resolve_env', lambda: SimpleNamespace(root=root))


def test_inspection_makes_no_provider_or_store_write(hy, monkeypatch, capsys):
    _published(hy.conn)
    with db.transaction(hy.conn):
        record_summary_failure(hy.conn, 'x', 'summary_output_cap')
    _environment(monkeypatch, hy.config.root)
    monkeypatch.setattr(bootstrap, 'build_from_env', lambda: pytest.fail('inspection built provider'))
    before = list(hy.conn.iterdump())
    assert recover_summaries.main([]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'inspect' and result['status'] == 'degraded'
    assert result['health_before']['summary_degraded_sessions'] == 1
    assert list(hy.conn.iterdump()) == before


def test_inspection_of_healthy_store_is_complete(hy, monkeypatch, capsys):
    _published(hy.conn)
    _environment(monkeypatch, hy.config.root)
    assert recover_summaries.main([]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'complete'


def test_missing_store_refuses_creation_and_provider(tmp_path, monkeypatch, capsys):
    _environment(monkeypatch, tmp_path)
    monkeypatch.setattr(bootstrap, 'build_from_env', lambda: pytest.fail('missing store built provider'))
    assert recover_summaries.main(['--apply']) == 1
    assert not list(tmp_path.iterdir())
    assert json.loads(capsys.readouterr().out)['status'] == 'error'


def test_old_schema_refuses_upgrade_or_provider(hy, monkeypatch, capsys):
    hy.conn.execute("UPDATE schema_meta SET value='61' WHERE key='schema_version'")
    _environment(monkeypatch, hy.config.root)
    monkeypatch.setattr(bootstrap, 'build_from_env', lambda: pytest.fail('old store built provider'))
    before = list(hy.conn.iterdump())
    assert recover_summaries.main(['--apply']) == 1
    assert list(hy.conn.iterdump()) == before
    assert json.loads(capsys.readouterr().out)['status'] == 'error'


@pytest.mark.parametrize('args', [
    ['--max-calls', '0'], ['--max-calls', '101'], ['--max-attempts', '-1'],
    ['--max-chars', '0'], ['--max-tokens', '32769'], ['--timeout-seconds', 'nan'],
    ['--timeout-seconds', 'inf'], ['--session-id', ''],
])
def test_invalid_bounds_fail_before_environment_or_provider(args, monkeypatch):
    monkeypatch.setattr(bootstrap, 'resolve_env', lambda: pytest.fail('invalid bounds reached environment'))
    with pytest.raises(SystemExit) as caught:
        recover_summaries.main(args)
    assert caught.value.code == 2


@pytest.mark.parametrize('failure', [None, RuntimeError('SECRET provider output'), KeyboardInterrupt('SECRET interrupt')])
def test_apply_delegates_exact_budget_and_always_closes_owner(hy, monkeypatch, capsys, failure):
    _published(hy.conn)
    _environment(monkeypatch, hy.config.root)
    events = []
    def run(**kwargs):
        events.append(kwargs)
        if failure is not None:
            raise failure
        return dict(calls=0, advanced=0, published=0, held=0, exhausted=0, remaining=0,
                    provider_attempts=0, provider_attempts_exact=True)
    owner = SimpleNamespace(recover_summaries=run)
    monkeypatch.setattr(bootstrap, 'build_from_env', lambda: owner)
    monkeypatch.setattr(bootstrap, 'shutdown_instance', lambda value: events.append(value))
    args = ['--apply', '--max-calls', '2', '--max-attempts', '4', '--max-chars', '4000',
            '--max-tokens', '500', '--timeout-seconds', '30', '--session-id', 'x']
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt) as caught:
            recover_summaries.main(args)
        assert caught.value is failure
    else:
        assert recover_summaries.main(args) == (1 if failure else 0)
    assert events == [dict(max_calls=2, max_attempts=4, max_chars=4000, max_tokens=500,
                           timeout_seconds=30.0, session_id='x'), owner]
    assert 'SECRET' not in capsys.readouterr().out


def test_cleanup_failure_preserves_primary_interrupt(hy, monkeypatch, capsys):
    _published(hy.conn)
    _environment(monkeypatch, hy.config.root)
    primary = KeyboardInterrupt('primary secret')
    def run(**kwargs):
        raise primary
    def close(value):
        raise RuntimeError('cleanup secret')
    monkeypatch.setattr(bootstrap, 'build_from_env', lambda: SimpleNamespace(recover_summaries=run))
    monkeypatch.setattr(bootstrap, 'shutdown_instance', close)
    with pytest.raises(KeyboardInterrupt) as caught:
        recover_summaries.main(['--apply'])
    assert caught.value is primary
    assert 'RuntimeError' in str(primary.__notes__)
    assert 'cleanup secret' not in str(primary.__notes__)
    assert not capsys.readouterr().out


def test_fatal_worker_costs_survive_sanitized_cli_receipt(hy, monkeypatch, capsys):
    _published(hy.conn)
    _environment(monkeypatch, hy.config.root)
    failure = RuntimeError('secret provider text')
    accounting = dict(calls=1, provider_attempts=2, provider_attempts_exact=True,
                      advanced=0, published=0, held=0, exhausted=0, remaining=1)
    failure.summary_recovery_report = accounting
    def run(**kwargs):
        raise failure
    monkeypatch.setattr(bootstrap, 'build_from_env', lambda: SimpleNamespace(recover_summaries=run))
    monkeypatch.setattr(bootstrap, 'shutdown_instance', lambda _: True)
    assert recover_summaries.main(['--apply']) == 1
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result['recovery'] == accounting
    assert 'secret provider text' not in output


def test_cleanup_interrupt_is_not_swallowed_after_a_reported_failure(hy, monkeypatch):
    _published(hy.conn)
    _environment(monkeypatch, hy.config.root)
    def run(**kwargs):
        raise RuntimeError('ordinary reported error')
    cancellation = KeyboardInterrupt('cancel cleanup')
    def close(value):
        raise cancellation
    monkeypatch.setattr(bootstrap, 'build_from_env', lambda: SimpleNamespace(recover_summaries=run))
    monkeypatch.setattr(bootstrap, 'shutdown_instance', close)
    with pytest.raises(KeyboardInterrupt) as caught:
        recover_summaries.main(['--apply'])
    assert caught.value is cancellation


@pytest.mark.parametrize('changes', [{'calls': True}, {'provider_attempts': -1},
                                   {'provider_attempts_exact': 1}, {'secret': 'hidden'}])
def test_untrusted_accounting_cannot_leak_into_cli_receipt(changes):
    accounting = dict(calls=1, provider_attempts=1, provider_attempts_exact=True,
                      advanced=0, published=0, held=1, exhausted=0, remaining=1)
    accounting.update(changes)
    assert recover_summaries._safe_recovery_report(accounting) is None
