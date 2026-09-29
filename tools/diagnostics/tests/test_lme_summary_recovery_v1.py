"""Synthetic, offline controls for the one-shot benchmark-clone recovery worker."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ['HYMEM_Q1_VERIFIER_SOURCE']).resolve()
sys.path.insert(0, str(SOURCE))
from tests.test_typed_response_recovery import sdk_factory
from tests.test_completion_response_admission import response
from tests.test_summary_recovery_v63 import _seed
from hymem.core import db

PATH = ROOT / 'tools/diagnostics/lme_summary_recovery_v1/worker.py'
worker = types.ModuleType('summary_recovery_worker_test')
worker.__file__ = str(PATH)
exec(compile(PATH.read_bytes(), str(PATH), 'exec'), worker.__dict__)
API = worker.import_api(SOURCE)


@pytest.fixture(scope='module', autouse=True)
def source_pins():
    raw = (ROOT / 'docs/patches/2026-09-24-lme-independent-summary-indexing-r5-manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == worker.R5_PIN
    pins = json.loads(raw)['source_sha256']
    assert Path(db.__file__).resolve() == SOURCE / 'hymem/core/db.py'
    def verify():
        assert all(hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() == pin
                   for path, pin in pins.items())
    verify()
    yield
    verify()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('unexpected network in synthetic recovery test')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'getnameinfo'):
        monkeypatch.setattr(socket, name, forbidden)


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / 'hymem.sqlite')
    db.initialize(connection)
    for number in range(10):
        _seed(connection, f'synthetic-{number:02}')
    yield connection
    connection.close()


def valid_reply(wire):
    assert wire['model'] == 'deepseek-flash'
    assert wire['max_tokens'] == 3072 and wire['temperature'] == 0.0
    return response(content=json.dumps({'summary': 'Both invented source messages remain represented.'}))


def test_all_ten_recover_through_actual_worker_sdk_boundary(conn, sdk_factory):
    client, calls = sdk_factory(valid_reply)
    before = worker.snapshot(conn)
    receipt = worker.run_checked(conn, client, API)
    after = worker.snapshot(conn)
    assert receipt['status'] == 'recovered_all' and len(calls) == 10
    assert receipt['changes']['recovered_sessions'] == 10
    assert receipt['health_after']['summary_healthy'] is True
    assert receipt['recovery']['calls'] == receipt['recovery']['published'] == 10
    assert receipt['usage']['request_attempts'] == 10
    assert receipt['usage']['token_usage_available'] is True
    assert receipt['client_closed'] is True and client._closed is True
    assert before['full_sha256'] != after['full_sha256']
    worker.assert_unchanged(before, after)


@pytest.mark.parametrize('mode', ['empty', 'length'])
def test_rejections_remain_honest_one_attempt_per_session_no_reroll(conn, sdk_factory, mode):
    value = response(content=json.dumps({'summary': ''})) if mode == 'empty' else response(
        content='unpublished partial', finish='length')
    client, calls = sdk_factory(lambda _: value)
    before = worker.snapshot(conn)
    receipt = worker.run_checked(conn, client, API)
    assert receipt['status'] == 'honestly_degraded' and len(calls) == 10
    assert receipt['recovery']['held'] == receipt['health_after']['summary_degraded_sessions'] == 10
    assert receipt['changes']['published_sessions'] == 0
    assert worker.snapshot(conn)['sessions'] == before['sessions']
    assert {row[0] for row in conn.execute('SELECT attempts FROM summary_recovery')} == {1}
    assert receipt['usage']['total_tokens'] == 80
    assert receipt['usage']['calls'] == (10 if mode == 'empty' else 0)


def test_budget_exhaustion_leaves_partial_draft_private(conn, sdk_factory, monkeypatch):
    # Only this synthetic control lowers the fixed production bound to force a
    # partial state. The real manifest-bound worker always uses 100/8000.
    monkeypatch.setitem(worker.BOUNDS, 'max_calls', 1)
    monkeypatch.setitem(worker.BOUNDS, 'max_chars', 300)
    _seed(conn, 'aaa-long-synthetic', long=True)
    client, calls = sdk_factory(valid_reply)
    before = worker.snapshot(conn)
    receipt = worker.run_checked(conn, client, API, expected_degraded=11)
    assert receipt['status'] == 'honestly_degraded' and len(calls) == 1
    assert receipt['changes']['private_partial_sessions'] == 1
    assert receipt['changes']['published_sessions'] == 0
    assert worker.snapshot(conn)['sessions'] == before['sessions']


def test_provider_fault_retains_unknown_paid_cost_and_closes(conn, sdk_factory):
    client, calls = sdk_factory(lambda _: RuntimeError('PRIVATE_PROVIDER_DIAGNOSTIC'))
    receipt = worker.run_checked(conn, client, API)
    assert receipt['status'] == 'error' and len(calls) == 3
    assert receipt['recovery']['calls'] == 1 and receipt['recovery']['provider_attempts'] == 3
    assert receipt['usage']['token_usage_available'] is False
    assert receipt['usage']['total_tokens'] is None and receipt['usage']['cost_usd'] is None
    assert receipt['client_closed'] and client._closed
    assert 'PRIVATE_PROVIDER_DIAGNOSTIC' not in json.dumps(receipt)


def test_unknown_baseline_refuses_before_dispatch(conn, sdk_factory):
    client, calls = sdk_factory(valid_reply)
    with pytest.raises(worker.DiagnosticFailure, match='unexpected_summary_baseline'):
        worker.run_checked(conn, client, API, expected_degraded=9)
    assert calls == []


@pytest.mark.parametrize('table', ['messages', 'episodes', 'knowledge_graph',
                                  'message_retention_coverage', 'phase1_generations', 'run_lock'])
def test_any_non_summary_table_change_is_rejected(conn, table):
    before = worker.snapshot(conn)
    altered = deepcopy(before)
    altered['tables'][table]['sha256'] = '0' * 64
    with pytest.raises(worker.DiagnosticFailure, match='non_summary_table_changed'):
        worker.assert_unchanged(before, altered)


def test_item_frontier_change_is_rejected(conn):
    before = worker.snapshot(conn)
    altered = deepcopy(before)
    altered['stable_sessions']['sha256'] = '0' * 64
    with pytest.raises(worker.DiagnosticFailure, match='item_or_source_frontier_changed'):
        worker.assert_unchanged(before, altered)


def test_schema_change_is_rejected(conn):
    before = worker.snapshot(conn)
    altered = deepcopy(before)
    altered['schema_sha256'] = '0' * 64
    with pytest.raises(worker.DiagnosticFailure, match='schema_changed'):
        worker.assert_unchanged(before, altered)


def test_curated_summary_is_preserved_with_successful_auto_recovery(conn, sdk_factory):
    conn.execute("UPDATE sessions SET summary='Invented curator text.',summary_source='operator'")
    client, _ = sdk_factory(valid_reply)
    receipt = worker.run_checked(conn, client, API)
    assert receipt['status'] == 'recovered_all'
    assert {row[0] for row in conn.execute('SELECT summary FROM sessions')} == {'Invented curator text.'}


def test_postrun_tamper_retains_paid_accounting_in_outer_receipt(conn, sdk_factory, monkeypatch):
    client, calls = sdk_factory(valid_reply)
    def reject(*args):
        raise worker.DiagnosticFailure('synthetic_audit_failure')
    monkeypatch.setattr(worker, 'assert_unchanged', reject)
    receipt = {}
    with pytest.raises(worker.DiagnosticFailure, match='synthetic_audit_failure'):
        worker.run_checked(conn, client, API, evidence=receipt)
    assert receipt['recovery']['calls'] == receipt['usage']['request_attempts'] == len(calls) == 10
    assert client._closed


def test_reference_snapshot_includes_live_wal_and_backup_equivalence(conn, tmp_path):
    # Keep the writer open so WAL content is required. Never immutable=1.
    source = tmp_path / 'hymem.sqlite'
    assert Path(str(source) + '-wal').exists()
    reader = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, isolation_level=None)
    reader.row_factory = sqlite3.Row
    destination = sqlite3.connect(tmp_path / 'clone.sqlite', isolation_level=None)
    destination.row_factory = sqlite3.Row
    try:
        reader.execute('PRAGMA query_only=ON')
        reader.backup(destination)
        assert worker.snapshot(reader)['full_sha256'] == worker.snapshot(destination)['full_sha256']
    finally:
        destination.close()
        reader.close()


def test_bounds_and_report_dont_claim_semantic_or_benchmark_success():
    assert worker.BOUNDS == dict(max_calls=100, max_attempts=3, max_chars=8000,
                                 max_tokens=3072, timeout_seconds=1800)
    assert API.retry_attempts == 3
    assert worker.SUMMARY_FIELDS == {
        'summary', 'summary_source', 'auto_summary', 'auto_summary_generation',
        'auto_summary_message_id', 'auto_summary_partial_message_id', 'auto_summary_message_offset',
        'summary_failure_reason', 'summary_failure_count',
    }


def test_real_sdk_preflight_proves_identity_without_provider_work(conn):
    from hymem.extraction.producer import phase1_generation_binding, register_phase1_generation
    client = API.client(api_key='synthetic-preflight-not-a-key', base_url=worker.ENDPOINT,
                        model=worker.MODEL, thinking='disabled')
    try:
        with db.transaction(conn):
            register_phase1_generation(conn, phase1_generation_binding('v1', client))
        before = worker.snapshot(conn)['full_sha256']
        receipt = worker.preflight_checked(conn, client, API)
        assert receipt['status'] == 'preflight_passed'
        assert receipt['stock_invocations'] == receipt['provider_calls'] == 0
        assert receipt['usage']['calls'] == receipt['usage']['request_attempts'] == 0
        assert worker.snapshot(conn)['full_sha256'] == before
        assert client._closed
    finally:
        client.close()


def test_real_sdk_preflight_refuses_unbound_producer_without_calls(conn):
    client = API.client(api_key='synthetic-preflight-not-a-key', base_url=worker.ENDPOINT,
                        model=worker.MODEL, thinking='disabled')
    try:
        with pytest.raises(worker.DiagnosticFailure, match='q1_producer_mismatch'):
            worker.preflight_checked(conn, client, API)
        assert client.call_count == client.request_attempts == 0
    finally:
        client.close()


def test_exact_authorization_frame_is_accepted():
    worker.read_authority(io.BytesIO(worker.encoded(worker.AUTHORIZATION) + b'\n'))


@pytest.mark.parametrize('frame', [b'', b'{}', b'x' * 256, b'null', b'{"execute_summary_recovery":true}'])
def test_invalid_authorization_frames_are_rejected(frame):
    with pytest.raises(worker.DiagnosticFailure, match='missing_execution_authority'):
        worker.read_authority(io.BytesIO(frame))


def test_duplicate_or_trailing_authorization_is_rejected():
    frame = worker.encoded(worker.AUTHORIZATION)
    duplicate = frame[:-1] + b',' + frame[1:]
    for value in (duplicate, frame + b'{}'):
        with pytest.raises(worker.DiagnosticFailure, match='missing_execution_authority'):
            worker.read_authority(io.BytesIO(value))


@pytest.fixture
def source_inventory(tmp_path, monkeypatch):
    source = tmp_path / 'candidate'
    source.mkdir()
    pins = {}
    for number in range(231):
        path = source / f'synthetic_{number}.py'
        path.write_bytes(b'# invented source\n')
        pins[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.json'
    manifest.write_bytes(worker.encoded({'source_sha256': pins}))
    monkeypatch.setattr(worker, 'R5_PIN', hashlib.sha256(manifest.read_bytes()).hexdigest())
    return source, manifest


def test_source_verification_accepts_exact_regular_inventory(source_inventory):
    worker.verify_source(*source_inventory)


@pytest.mark.parametrize('kind', ['fifo', 'symlink', 'directory_symlink'])
def test_source_special_files_are_rejected_before_reads(source_inventory, kind):
    source, manifest = source_inventory
    extra = source / 'unexpected'
    if kind == 'fifo': os.mkfifo(extra)
    elif kind == 'directory_symlink': extra.symlink_to(source.parent, target_is_directory=True)
    else: extra.symlink_to(source / 'synthetic_0.py')
    with pytest.raises(worker.DiagnosticFailure, match='source_special_file'):
        worker.verify_source(source, manifest)


def test_source_walk_errors_are_not_silently_ignored(source_inventory, monkeypatch):
    def broken(root, *, followlinks, onerror):
        onerror(OSError('synthetic permission denial'))
        return iter(())
    monkeypatch.setattr(worker.os, 'walk', broken)
    with pytest.raises(worker.DiagnosticFailure, match='source_traversal_error'):
        worker.verify_source(*source_inventory)
