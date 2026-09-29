"""Offline controls for Fix3 isolation, strict success, and paid accounting."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'lme_r6_summary_replay.py'
SPEC = importlib.util.spec_from_file_location('r6_summary_replay_test', PATH)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('offline summary diagnostic attempted network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'getnameinfo'):
        monkeypatch.setattr(socket, name, forbidden)


@pytest.fixture
def receipt():
    return dict(status='recovered_all', stock_invocations=1,
        health_after=dict(summary_healthy=True, summary_degraded_sessions=0,
                          summary_missing_sessions=0, malformed_summaries=0),
        recovery=dict(calls=10, provider_attempts=10, provider_attempts_exact=True,
                      advanced=10, published=10, held=0, exhausted=0, remaining=0),
        changes=dict(published_sessions=10, recovered_sessions=10, private_partial_sessions=0, held_sessions=0),
        usage=dict(calls=10, calls_available=True, successful_responses=10, successful_responses_available=True,
                   request_attempts=10, request_attempts_available=True, token_usage_available=True,
                   prompt_tokens=100, completion_tokens=20, total_tokens=120,
                   latency_s=0.2, latency_available=True, cost_usd=None, cost_available=False))


def test_old_honest_degradation_cannot_pass_effectiveness(receipt):
    assert replay.effective(receipt, 10)
    receipt['status'] = 'honestly_degraded'
    receipt['health_after'].update(summary_healthy=False, summary_degraded_sessions=10, summary_missing_sessions=10)
    receipt['recovery'].update(advanced=0, published=0, held=10, remaining=10)
    receipt['changes'].update(published_sessions=0, recovered_sessions=0, held_sessions=10)
    replay.accounting(receipt, replay.RETAINED_BOUNDS)
    assert not replay.effective(receipt, 10)


@pytest.mark.parametrize('group,key,value', [
    ('health_after', 'summary_healthy', 1), ('health_after', 'summary_degraded_sessions', True),
    ('health_after', 'summary_missing_sessions', 1), ('health_after', 'malformed_summaries', 1),
    ('recovery', 'published', 9), ('recovery', 'remaining', 1), ('recovery', 'held', 1),
    ('recovery', 'exhausted', 1), ('changes', 'private_partial_sessions', 1),
    ('changes', 'published_sessions', 9), ('changes', 'held_sessions', 1),
])
def test_partial_or_malformed_outcomes_fail_effectiveness(receipt, group, key, value):
    receipt[group][key] = value
    assert not replay.effective(receipt, 10)


@pytest.mark.parametrize('mutation', ['bool_count', 'negative', 'over_budget', 'hidden_attempt', 'inexact',
    'outcome_sum', 'unavailable_attempts', 'bad_token_sum', 'token_float', 'known_cost_null', 'unknown_cost_value'])
def test_accounting_rejects_forged_or_ambiguous_receipts(receipt, mutation):
    r, u = receipt['recovery'], receipt['usage']
    if mutation == 'bool_count': r['published'] = True
    elif mutation == 'negative': r['held'] = -1
    elif mutation == 'over_budget': r['provider_attempts'] = u['request_attempts'] = 301
    elif mutation == 'hidden_attempt': u['request_attempts'] = 11
    elif mutation == 'inexact': r['provider_attempts_exact'] = False
    elif mutation == 'outcome_sum': r['advanced'] = 9
    elif mutation == 'unavailable_attempts': u['request_attempts_available'] = False
    elif mutation == 'bad_token_sum': u['total_tokens'] = 121
    elif mutation == 'token_float': u['prompt_tokens'] = 100.0
    elif mutation == 'known_cost_null': u['cost_available'] = True
    elif mutation == 'unknown_cost_value': u['cost_usd'] = 1.0
    with pytest.raises(RuntimeError):
        replay.accounting(receipt, replay.RETAINED_BOUNDS)


def test_aggregate_has_one_total_budget_and_does_not_invent_missing_work(receipt):
    second = deepcopy(receipt)
    second['recovery']['calls'] = 7
    second['usage']['request_attempts'] = 8
    assert replay.aggregate((receipt, second)) == dict(completion_calls=17, completion_calls_exact=True,
                                                      http_attempts=18, http_attempts_exact=True)
    receipt['recovery'] = None
    total = replay.aggregate((receipt, second))
    assert total['completion_calls'] is None and total['completion_calls_exact'] is False
    assert total['http_attempts'] == 18
    assert replay.aggregate(({'stock_invocations': 0},))['completion_calls'] == 0
    unknown = replay.aggregate(({'stock_invocations': 1},))
    assert unknown['http_attempts'] is None and unknown['http_attempts_exact'] is False


def test_control_bounds_restore_after_success_and_exception(receipt, monkeypatch):
    support = SimpleNamespace(BOUNDS=replay.RETAINED_BOUNDS,
                              run_checked=lambda *a, **k: deepcopy(receipt))
    conn = SimpleNamespace(execute=lambda _: SimpleNamespace(fetchone=lambda: (0,)))
    original = support.BOUNDS
    evidence = {}
    monkeypatch.setattr(replay, 'output_metadata', lambda *args: {
        'failure_reason_counts': {}, 'private_draft_chars': [], 'published_summary_chars': [20] * 10})
    replay.run_phase(conn, None, None, support, replay.CONTROL_BOUNDS, 10, evidence, before={})
    assert evidence['effectiveness_passed'] and support.BOUNDS is original
    def fail(*args, **kwargs):
        assert support.BOUNDS == replay.CONTROL_BOUNDS
        kwargs['evidence']['stock_invocations'] = 1
        kwargs['evidence']['usage'] = receipt['usage']
        raise RuntimeError('synthetic rejection after paid work')
    support.run_checked = fail
    with pytest.raises(RuntimeError):
        replay.run_phase(conn, None, None, support, replay.CONTROL_BOUNDS, 4, evidence, before={})
    assert support.BOUNDS is original and evidence['usage']['request_attempts'] == 10


def test_output_metadata_tracks_publication_by_exact_session_without_exporting_private_data():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, auto_summary TEXT, frontier INTEGER)')
        conn.execute('CREATE TABLE summary_recovery (failure_reason TEXT, draft TEXT)')
        conn.executemany('INSERT INTO sessions VALUES (?,?,?)', [
            ('PRIVATE-healthy', 'An existing healthy overview stays unchanged.', 1),
            ('PRIVATE-recovered', 'same ten chars', 1)])
        before = {'sessions': {row['id']: dict(row) for row in conn.execute('SELECT * FROM sessions')}}
        # A fresh proof can publish identical wording. Detect its changed
        # frontier by exact session identity, and exclude the healthy row.
        conn.execute("UPDATE sessions SET frontier=2 WHERE id='PRIVATE-recovered'")
        conn.execute('INSERT INTO summary_recovery VALUES (?,?)', ('summary_output_cap', '🧭' * 12))
        support = SimpleNamespace(public_projection=lambda row: {
            key: row[key] for key in ('auto_summary', 'frontier')})
        result = replay.output_metadata(conn, before, support)
        assert result == dict(failure_reason_counts={'summary_output_cap': 1},
                              private_draft_chars=[12], published_summary_chars=[14])
        assert 'PRIVATE' not in json.dumps(result) and 'same ten chars' not in json.dumps(result)
        conn.execute("UPDATE summary_recovery SET failure_reason='PRIVATE-unexpected-reason'")
        with pytest.raises(RuntimeError, match='metadata_failure_reason'):
            replay.output_metadata(conn, before, support)
    finally:
        conn.close()


@pytest.fixture
def wal_source(tmp_path, monkeypatch):
    reference = tmp_path.resolve() / 'reference.sqlite'
    conn = sqlite3.connect(reference, isolation_level=None)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA wal_autocheckpoint=0')
    conn.execute('CREATE TABLE invented (id INTEGER PRIMARY KEY, value TEXT)')
    conn.executemany('INSERT INTO invented(value) VALUES (?)', [('WAL-only value ' + 'x' * 4096,)] * 800)
    monkeypatch.setattr(replay, 'REFERENCE', reference)
    monkeypatch.setattr(replay, 'CLONE', tmp_path.resolve() / 'clone.sqlite')
    yield conn
    conn.close()


def test_paged_backup_pins_read_transaction_and_is_fresh(wal_source, monkeypatch):
    real_connect = sqlite3.connect
    observed = []
    class Tracked(sqlite3.Connection):
        def backup(self, target, *, pages, progress):
            assert self.in_transaction and pages == 256
            def check(status, remaining, total):
                observed.append(remaining)
                progress(status, remaining, total)
            return super().backup(target, pages=pages, progress=check)
    def connect(path, *args, **kwargs):
        if kwargs.get('uri'):
            assert path.endswith('?mode=ro') and 'immutable' not in path
            kwargs['factory'] = Tracked
        return real_connect(path, *args, **kwargs)
    monkeypatch.setattr(replay.sqlite3, 'connect', connect)
    before = replay.reference_files()
    replay.clone_reference()
    clone = real_connect(replay.CLONE)
    try:
        assert clone.execute('SELECT COUNT(*) FROM invented').fetchone()[0] == 800
    finally:
        clone.close()
    assert len(observed) > 1 and observed[-1] == 0
    after = replay.reference_files()
    # Unlike the real read-only bind mount, this local test has a writable
    # original and an existing writable SHM mapping in the same process.
    # SQLite may update SHM housekeeping, but never the database or WAL.
    assert {k: v for k, v in after.items() if not k.endswith('-shm')} == {
        k: v for k, v in before.items() if not k.endswith('-shm')}
    assert set(after) == set(before) and replay.REFERENCE.name + '-shm' in after
    with pytest.raises(FileExistsError):
        replay.clone_reference()


@pytest.mark.parametrize('fault', ['samefile', 'hardlink', 'symlink', 'sidecar'])
def test_clone_aliases_and_stale_sidecars_are_rejected(wal_source, monkeypatch, fault):
    if fault == 'samefile': monkeypatch.setattr(replay, 'CLONE', replay.REFERENCE)
    elif fault == 'hardlink': os.link(replay.REFERENCE, replay.CLONE)
    elif fault == 'symlink': replay.CLONE.symlink_to(replay.REFERENCE)
    else: Path(str(replay.CLONE) + '-wal').write_bytes(b'stale')
    with pytest.raises((FileExistsError, RuntimeError)):
        replay.clone_reference()
    assert wal_source.execute('SELECT COUNT(*) FROM invented').fetchone()[0] == 800


@pytest.fixture
def candidate_api():
    source = Path(os.environ.get('HYMEM_Q1_VERIFIER_SOURCE', PATH.parents[2])).resolve()
    support = replay.load(PATH.parent / 'lme_summary_recovery_v1/worker.py',
                          'r6_summary_test_support', replay.SUPPORT_SHA)
    api = support.import_api(source)
    return support, api


def test_four_real_source_controls_include_dense_and_multiwindow_replay(tmp_path, monkeypatch, candidate_api):
    support, api = candidate_api
    monkeypatch.setattr(replay, 'CONTROL', tmp_path.resolve() / 'control.sqlite')
    replay.seed_controls(api)
    conn = api.db.connect(replay.CONTROL)
    try:
        before, _, health = support.verify_baseline(conn, api, expected_degraded=4)
        assert health['summary_degraded_sessions'] == health['summary_missing_sessions'] == 4
        assert replay.verify_controls(conn) == replay.digest(replay.control_cases())
        counts = {}
        for row in conn.execute('SELECT id,digest_published_message_id FROM sessions').fetchall():
            sid, target = row
            job = dict(session_id=sid, target_message_id=target, cursor_message_id=None,
                       cursor_partial_message_id=None, cursor_offset=0, draft='')
            count = 0
            while True:
                request, after = api.recovery._request(conn, job, 8000, 3072)
                assert request.max_tokens == 3072 and json.loads(request.user)['new_material']
                count += 1
                assert count <= 24
                if after == (target, None, 0): break
                job.update(cursor_message_id=after[0], cursor_partial_message_id=after[1], cursor_offset=after[2])
            counts[sid] = count
        assert counts['control-dense'] == 1
        assert counts['control-multiwindow'] >= 3
        assert sum(counts.values()) <= 24
        assert support.snapshot(conn)['full_sha256'] == before['full_sha256']
        # Source is deliberately immutable. Adding an unrelated session is
        # enough to prove that the exact control census rejects tampering.
        from hymem.session import open_session
        open_session(conn, 'unexpected-control')
        with pytest.raises(RuntimeError, match='control_sessions'):
            replay.verify_controls(conn)
    finally:
        conn.close()


@pytest.mark.parametrize('mode', ['valid', 'empty', 'truncated', 'overlong', 'partial_hold'])
def test_actual_stock_recovery_and_sdk_accounting_on_invented_sources(tmp_path, monkeypatch, candidate_api, mode):
    support, api = candidate_api
    import openai
    calls = []
    expected_summaries = {}
    expected_drafts = []
    class Resource:
        def create(self, **wire):
            calls.append(wire)
            assert wire['model'] == replay.MODEL and wire['max_tokens'] == 3072
            assert wire['temperature'] == 0.0
            envelope = json.loads(wire['messages'][1]['content'])
            material, prior = envelope['new_material'], envelope['prior_summary']
            if 'Cedar' in material:
                case = 'control-dense'
                value = ('Elena approved Cedar\'s read-only inventory with a 240-euro ceiling. '
                         'Deleting originals is prohibited. The inventory has not run and is paused for her checksum review.')
            elif 'Omar' in material:
                case = 'control-injection'
                value = ('Omar\'s export remains failed and originals are intact. Error inspection is pending. '
                         'The log\'s success and deletion instructions are untrusted; no repair or deletion was executed.')
            elif 'Iris' in material:
                case = 'control-uncertainty'
                value = ('Iris confirmed two absent receipts; five remain unresolved. Stale cache is only her hypothesis '
                         'and the cause is unproved. Refresh/comparison were proposed, not executed; no receipt was restored.')
            elif 'Lantern' in material or 'Lantern' in prior:
                case = 'control-multiwindow'
                value = ('Ada owns Lantern; originals must be preserved. Noel updated the target from 12 to 14 files. '
                         'Only a dry-run inventory exists: real restore is pending and the deadline is unknown.'
                         if 'target is now 14' in material else
                         'Ada owns Lantern\'s restore plan. The initial target is 12 files; originals must be preserved and deletion is prohibited.')
            else:
                pytest.fail('unrecognized invented source window')
            assert 10 <= len(value) <= 500
            expected_summaries[case] = value
            if mode in ('empty', 'truncated', 'overlong'):
                value = 'x' * 501 if mode == 'overlong' else ''
            elif mode == 'partial_hold' and prior:
                expected_drafts.append(prior)
                value = 'x' * 501
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason='length' if mode == 'truncated' else 'stop',
                message=SimpleNamespace(content=json.dumps({'summary': value})))],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8))
    class SDK:
        def __init__(self, **options):
            self._client = options['http_client']
            self.chat = SimpleNamespace(completions=Resource())
        def close(self):
            self._client.close()
    monkeypatch.setattr(openai, 'OpenAI', SDK)
    monkeypatch.delenv('HYMEM_LLM_EXTRA_BODY', raising=False)
    monkeypatch.setattr(replay, 'CONTROL', tmp_path.resolve() / 'control.sqlite')
    replay.seed_controls(api)
    conn = api.db.connect(replay.CONTROL)
    client = api.client(api_key='synthetic-not-a-key', base_url=replay.ENDPOINT,
                        model=replay.MODEL, thinking='disabled')
    evidence = {'stock_invocations': 0}
    before = support.snapshot(conn)
    try:
        replay.run_phase(conn, client, api, support, replay.CONTROL_BOUNDS, 4, evidence, before=before)
        assert evidence['stock_invocations'] == 1 and client._closed
        assert evidence['recovery']['provider_attempts_exact'] is True
        assert evidence['recovery']['provider_attempts'] == evidence['usage']['request_attempts'] == len(calls)
        assert evidence['usage']['total_tokens'] == 8 * len(calls)
        assert support.BOUNDS == replay.RETAINED_BOUNDS
        support.assert_unchanged(before, support.snapshot(conn))
        if mode == 'valid':
            assert evidence['output_metadata'] == dict(failure_reason_counts={}, private_draft_chars=[],
                published_summary_chars=sorted(len(value) for value in expected_summaries.values()))
            assert evidence['effectiveness_passed'] and evidence['recovery']['published'] == 4
            assert len(calls) == evidence['recovery']['advanced'] == 6
            assert sum(bool(json.loads(wire['messages'][1]['content'])['prior_summary']) for wire in calls) == 2
            assert 'Noel updated the target from 12 to 14' in conn.execute(
                "SELECT auto_summary FROM sessions WHERE id='control-multiwindow'").fetchone()[0]
        elif mode == 'partial_hold':
            assert not evidence['effectiveness_passed'] and len(calls) == 5
            assert evidence['recovery']['held'] == evidence['changes']['private_partial_sessions'] == 1
            assert evidence['recovery']['published'] == 3
            assert evidence['output_metadata'] == dict(failure_reason_counts={'summary_output_cap': 1},
                private_draft_chars=[len(expected_drafts[0])],
                published_summary_chars=sorted(len(value) for sid, value in expected_summaries.items()
                                                if sid != 'control-multiwindow'))
            assert len(expected_drafts[0]) > 0
        else:
            reason = {'empty': 'summary_validation_failure', 'truncated': 'output_truncated',
                      'overlong': 'summary_output_cap'}[mode]
            assert evidence['output_metadata'] == dict(failure_reason_counts={reason: 4},
                private_draft_chars=[0, 0, 0, 0], published_summary_chars=[])
            assert not evidence['effectiveness_passed'] and len(calls) == 4
            assert evidence['recovery']['held'] == 4 and evidence['recovery']['published'] == 0
            assert support.snapshot(conn)['sessions'] == before['sessions']
            assert {row[0] for row in conn.execute('SELECT attempts FROM summary_recovery')} == {1}
        assert conn.execute('SELECT 1 FROM run_lock').fetchone() is None and not conn.in_transaction
        serialized = json.dumps(evidence['output_metadata'])
        assert all(sid not in serialized and value not in serialized for sid, value in expected_summaries.items())
    finally:
        client.close()
        conn.close()


def test_manifest_requires_exact_new_source_and_unchanged_support(tmp_path, monkeypatch):
    source = tmp_path.resolve() / 'candidate'
    source.mkdir()
    for i in range(231): (source / f'source{i}.py').write_bytes(b'# source\n')
    diag = tmp_path.resolve() / 'diag'
    diag.mkdir()
    (diag / 'worker.py').write_bytes(PATH.read_bytes())
    support_dir = tmp_path.resolve() / 'support'
    support_dir.mkdir()
    monkeypatch.setattr(replay, 'SOURCE', source)
    monkeypatch.setattr(replay, 'DIAG', diag)
    original_sha = replay.sha
    def sha(path):
        if path == Path('/support/worker.py'): return replay.SUPPORT_SHA
        if path == Path('/support/supervised_invocation.py'): return replay.SUPERVISOR_SHA
        return original_sha(path)
    monkeypatch.setattr(replay, 'sha', sha)
    manifest = dict(schema=replay.SCHEMA, source_sha256={p.name: original_sha(p) for p in source.iterdir()},
                    support_sha256=replay.SUPPORT_SHA, supervisor_sha256=replay.SUPERVISOR_SHA,
                    worker_sha256=original_sha(PATH))
    (diag / 'manifest.json').write_bytes(replay.encoded(manifest))
    pin = original_sha(diag / 'manifest.json')
    assert replay.verify(pin) == manifest
    (source / 'source0.py').write_bytes(b'# changed\n')
    with pytest.raises(RuntimeError, match='source_inventory_drift'): replay.verify(pin)
