"""One bounded stock summary-recovery invocation on an isolated benchmark clone.

No dream, migration, reader, judge, or production operation is performed.
Only bounded hashes/counters leave this process; source and summaries stay local.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import sqlite3
import stat
import sys
import threading

R5_PIN = 'c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'
MODEL = 'deepseek-flash'
ENDPOINT = 'https://api.deepseek.com'
BOUNDS = dict(max_calls=100, max_attempts=3, max_chars=8000, max_tokens=3072,
              timeout_seconds=1800)
RUNTIME = {'openai': '2.53.0', 'httpx': '0.28.1', 'requests': '2.34.2'}
AUTHORIZATION = {'execute_summary_recovery': 'summary-recovery-live-diagnostic-v1:' + R5_PIN}
SUMMARY_FIELDS = frozenset({
    'summary', 'summary_source', 'auto_summary', 'auto_summary_generation',
    'auto_summary_message_id', 'auto_summary_partial_message_id',
    'auto_summary_message_offset', 'summary_failure_reason', 'summary_failure_count',
})


class DiagnosticFailure(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(value, label):
    if not value:
        raise DiagnosticFailure(label)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def regular(path):
    require(path.is_absolute() and path.resolve() == path
            and stat.S_ISREG(path.lstat().st_mode), 'non_regular_path')
    return path


def verify_source(source, manifest_path):
    raw = regular(manifest_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == R5_PIN, 'r5_manifest_pin')
    pins = json.loads(raw)['source_sha256']
    require(len(pins) == 231 and source.resolve() == source and source.is_dir(), 'source_shape')
    actual = {}
    def traversal_error(_exc):
        raise DiagnosticFailure('source_traversal_error')
    for directory, names, files in os.walk(source, followlinks=False, onerror=traversal_error):
        for name in names + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'source_special_file')
            if stat.S_ISREG(mode):
                actual[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == pins, 'source_inventory_drift')
    return digest(pins)


def read_key(path):
    require(os.geteuid() == 1000 and path.resolve() == path, 'credential_mount')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == 1000
                and stat.S_IMODE(info.st_mode) == 0o600 and info.st_size <= 65536,
                'credential_permissions')
        raw = handle.read(65537)
    require(len(raw) <= 65536, 'credential_size')
    values = []
    for line in raw.decode().splitlines():
        match = re.fullmatch(r'[ \t]*(?:export[ \t]+)?DEEPSEEK_API_KEY[ \t]*=(.*)', line)
        if match:
            parts = shlex.split(match[1], comments=True, posix=True)
            require(len(parts) == 1 and re.fullmatch(r'[A-Za-z0-9_-]{20,200}', parts[0]),
                    'credential_shape')
            values.append(parts[0])
    require(len(values) == 1 and values[0] != 'dummy-loopback-only', 'credential_count')
    return values[0]


def cell(value):
    if value is None:
        return ['null']
    if type(value) is bytes:
        return ['blob', base64.b64encode(value).decode('ascii')]
    require(type(value) in (str, int, float), 'unsupported_sqlite_cell')
    require(type(value) is not float or math.isfinite(value), 'nonfinite_sqlite_cell')
    return [type(value).__name__, value]


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def table_fingerprint(conn, name, *, exclude=()):
    columns = [row[1] for row in conn.execute('PRAGMA table_info(' + quote(name) + ')')]
    selected = [column for column in columns if column not in exclude]
    require(selected, 'empty_table_projection')
    # Rowid is an identity too. WITHOUT ROWID tables already expose their key.
    declaration = conn.execute('SELECT sql FROM sqlite_schema WHERE type=\'table\' AND name=?',
                               (name,)).fetchone()[0] or ''
    rowid = [] if re.search(r'\bWITHOUT\s+ROWID\b', declaration, re.I) else ['_rowid_']
    selection = ','.join(quote(column) for column in rowid + selected)
    rows = sorted(encoded([cell(value) for value in row])
                  for row in conn.execute('SELECT ' + selection + ' FROM ' + quote(name)))
    hasher = hashlib.sha256(encoded(rowid + selected))
    for row in rows:
        hasher.update(str(len(row)).encode() + b':' + row)
    return {'rows': len(rows), 'sha256': hasher.hexdigest()}


def snapshot(conn):
    require(not conn.in_transaction, 'snapshot_caller_transaction')
    conn.execute('BEGIN')
    try:
        schema = [tuple(row) for row in conn.execute(
            'SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name')]
        tables = {row[1]: table_fingerprint(conn, row[1]) for row in schema if row[0] == 'table'}
        sessions = {row['id']: dict(row) for row in conn.execute('SELECT * FROM sessions')}
        stable = table_fingerprint(conn, 'sessions', exclude=SUMMARY_FIELDS)
        return {'schema_sha256': digest(schema), 'tables': tables,
                'stable_sessions': stable, 'sessions': sessions,
                'full_sha256': digest({'schema': schema, 'tables': tables})}
    finally:
        conn.execute('ROLLBACK')


def assert_unchanged(before, after):
    require(before['schema_sha256'] == after['schema_sha256'], 'schema_changed')
    require(before['stable_sessions'] == after['stable_sessions'], 'item_or_source_frontier_changed')
    require(set(before['tables']) == set(after['tables']), 'table_inventory_changed')
    for table in before['tables']:
        if table not in {'sessions', 'summary_recovery'}:
            require(before['tables'][table] == after['tables'][table], 'non_summary_table_changed')


def integrity(conn):
    require([row[0] for row in conn.execute('PRAGMA integrity_check')] == ['ok'], 'integrity_check')
    require(conn.execute('PRAGMA foreign_key_check').fetchone() is None, 'foreign_key_check')


def public_projection(row):
    return {key: row[key] for key in SUMMARY_FIELDS}


def audit_summary_changes(conn, before, after, initial_states, report, api):
    changed = recovered = partial = held = 0
    require(set(before['sessions']) == set(after['sessions']), 'session_inventory_changed')
    for sid, original in before['sessions'].items():
        current = after['sessions'][sid]
        state = api.classify_summary_state(conn, sid)
        require(not state['malformed'], 'malformed_summary_after')
        was_degraded = initial_states[sid]['degraded']
        did_change = encoded(public_projection(original)) != encoded(public_projection(current))
        require(was_degraded or not did_change, 'healthy_summary_changed')
        job = api.recovery._read_job(conn, sid)
        if job is not None:
            require(was_degraded, 'unexpected_private_job')
            api.recovery._validate_job(conn, job)
            require(job['target_message_id'] == original['digest_published_message_id']
                    and job['target_generation'] == original['digest_published_generation'],
                    'private_target_changed')
            partial += int(api.recovery._position(job) != (None, None, 0))
            held += int(job['failure_reason'] is not None)
        if did_change:
            require(state['summary_healthy'] and job is None, 'partial_summary_published')
            require(current['auto_summary_generation'] == original['digest_published_generation']
                    and current['auto_summary_message_id'] == original['digest_published_message_id']
                    and current['auto_summary_partial_message_id'] is None
                    and current['auto_summary_message_offset'] == 0
                    and current['summary_failure_reason'] is None
                    and current['summary_failure_count'] == 0, 'publication_frontier')
            # Reproduce the stock writer's operator/legacy non-overwrite rule.
            auto_owned = (original['summary'] is None or (
                original['summary_source'] == 'auto' and original['summary'] == original['auto_summary']))
            expected_summary = current['auto_summary'] if auto_owned else original['summary']
            expected_owner = ('auto' if auto_owned else 'operator'
                              if original['summary_source'] in ('auto', None) else original['summary_source'])
            require(current['summary'] == expected_summary and current['summary_source'] == expected_owner,
                    'curated_summary_changed')
        changed += int(did_change)
        recovered += int(was_degraded and state['summary_healthy'])
    if report is not None:
        require(report['published'] == changed == recovered, 'publication_accounting')
    return {'published_sessions': changed, 'recovered_sessions': recovered,
            'private_partial_sessions': partial, 'held_sessions': held}


def verify_baseline(conn, api, *, expected_degraded=10):
    """Read-only pre-paid store admission; shared by preflight and live worker."""
    integrity(conn)
    require(api.db.schema_version(conn) == 63, 'schema_version')
    before = snapshot(conn)
    require(before['tables']['summary_recovery']['rows'] == 0
            and before['tables']['run_lock']['rows'] == 0, 'not_fresh_recovery_clone')
    initial = {sid: api.classify_summary_state(conn, sid) for sid in before['sessions']}
    health_before = api.durable_summary_status(conn)
    require(health_before['summary_degraded_sessions'] == expected_degraded
            and health_before['malformed_summaries'] == 0, 'unexpected_summary_baseline')
    for sid, state in initial.items():
        if state['degraded']:
            row = before['sessions'][sid]
            require(row['digest_published_message_id'] is not None
                    and row['digest_published_message_id'] == row['coverage_message_id'],
                    'unindexed_summary_target')
    return before, initial, health_before


def run_checked(conn, client, api, *, expected_degraded=10, evidence=None):
    """Exercise the actual worker exactly once; tests inject only the provider."""
    before, initial, health_before = verify_baseline(conn, api, expected_degraded=expected_degraded)
    report = error = None
    close_error = None
    evidence = {} if evidence is None else evidence
    initial_threads = {thread.ident for thread in threading.enumerate()}
    try:
        evidence['stock_invocations'] = 1
        report = api.recovery.run_summary_recovery(conn, client, **BOUNDS)
    except BaseException as exc:
        error = type(exc).__name__
        report = getattr(exc, 'summary_recovery_report', None)
    finally:
        usage = api.usage_snapshot(client)
        evidence['usage'] = usage
        evidence['recovery'] = api.safe_report(report)
        try:
            client.close()
        except BaseException as exc:
            close_error = type(exc).__name__
    report = api.safe_report(report)
    require(report is not None and report['calls'] <= BOUNDS['max_calls'], 'recovery_accounting')
    require(report['provider_attempts_exact'] is True
            and usage['request_attempts_available'] is True
            and report['provider_attempts'] == usage['request_attempts']
            and usage['request_attempts'] <= BOUNDS['max_calls'] * api.retry_attempts,
            'provider_attempt_accounting')
    require(usage['calls_available'] is True and usage['successful_responses_available'] is True
            and usage['calls'] == usage['successful_responses'] <= report['calls'], 'admitted_accounting')
    if error is None:
        require(report['calls'] == report['advanced'] + report['held'], 'outcome_accounting')
    require(not conn.in_transaction and conn.execute('SELECT 1 FROM run_lock').fetchone() is None,
            'lease_cleanup')
    require(not any(thread.ident not in initial_threads for thread in threading.enumerate()),
            'thread_cleanup')
    integrity(conn)
    after = snapshot(conn)
    assert_unchanged(before, after)
    changes = audit_summary_changes(conn, before, after, initial, report, api)
    health_after = api.durable_summary_status(conn)
    require(report['remaining'] == health_after['summary_degraded_sessions'] or error is not None,
            'remaining_accounting')
    return {'status': 'error' if error or close_error else
            'recovered_all' if health_after['summary_healthy'] else 'honestly_degraded',
            'exception_type': error, 'client_close_exception_type': close_error,
            'client_closed': close_error is None, 'lease_and_threads_clean': True,
            'health_before': health_before, 'health_after': health_after,
            'recovery': report, 'usage': usage, 'changes': changes,
            'integrity_check': 'ok', 'foreign_key_violations': 0,
            'all_non_summary_state_unchanged': True,
            'clone_before_sha256': before['full_sha256'], 'clone_after_sha256': after['full_sha256']}


def import_api(source):
    sys.path[:0] = [str(source), str(source / 'benchmarks')]
    from types import SimpleNamespace
    from hymem.core import db
    from hymem.dreaming import summary_recovery
    from hymem.dreaming.summary_state import classify_summary_state
    from hymem.dreaming.status import durable_summary_status
    from hymem.recover_summaries import _safe_recovery_report
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.extraction.producer import producer_binding_for_declaration
    from hymem.extraction.retry import DEFAULT_RETRY_ATTEMPTS
    from benchmarks.strictness import usage_snapshot
    require(Path(db.__file__).resolve() == source / 'hymem/core/db.py', 'import_source_drift')
    return SimpleNamespace(db=db, recovery=summary_recovery,
        classify_summary_state=classify_summary_state, durable_summary_status=durable_summary_status,
        safe_report=_safe_recovery_report, client=OpenAICompatibleClient, usage_snapshot=usage_snapshot,
        producer_binding=producer_binding_for_declaration, retry_attempts=DEFAULT_RETRY_ATTEMPTS)


def verify_client(conn, client, api):
    require(client.model == MODEL and client.base_url == ENDPOINT
            and client.effective_extra_body == {'thinking': {'type': 'disabled'}}
            and client.transport_integrity_ok is True, 'client_identity')
    binding = api.producer_binding(client, declaration_hook='memory_producer_declaration')
    require(binding['identity_exact'] is True, 'inexact_memory_producer')
    producers = {row[0] for row in conn.execute('SELECT DISTINCT producer_identity_sha256 FROM phase1_generations')}
    require(producers == {binding['identity_sha256']}, 'q1_producer_mismatch')
    return binding['identity_sha256']


def save(path, result):
    require(path.is_absolute() and path.parent.resolve() == path.parent
            and path.parent.is_dir(), 'receipt_path')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(encoded(result) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_authority(stream):
    # Exact canonical JSON prevents duplicate keys, trailing frames or strings
    # containing executable material. The supervisor writes the frame then EOF.
    raw = stream.read(256)
    require(len(raw) < 256 and raw.strip() == encoded(AUTHORIZATION), 'missing_execution_authority')


def preflight_checked(conn, client, api):
    before, _states, health = verify_baseline(conn, api)
    identity = verify_client(conn, client, api)
    usage = api.usage_snapshot(client)
    require(all(usage[name + '_available'] is True and type(usage[name]) is int
                and usage[name] == 0
                for name in ('calls', 'request_attempts', 'successful_responses')),
            'preflight_provider_work')
    client.close()
    require(snapshot(conn)['full_sha256'] == before['full_sha256'], 'preflight_mutated_store')
    return {'status': 'preflight_passed', 'producer_identity_sha256': identity,
            'health_before': health, 'usage': usage, 'client_closed': True,
            'stock_invocations': 0, 'provider_calls': 0,
            'clone_before_sha256': before['full_sha256'], 'clone_after_sha256': before['full_sha256'],
            'all_non_summary_state_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'r5-manifest', 'clone', 'reference', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    os.environ.clear()
    os.environ.update(PATH='/home/node/hymem-env/bin:/usr/bin:/bin', HOME='/tmp',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', LANG='C.UTF-8')
    logging.disable(logging.CRITICAL)
    result = {'schema': 'summary-recovery-live-diagnostic-v1', 'r5_manifest_sha256': R5_PIN,
              'model': MODEL, 'endpoint': ENDPOINT, 'bounds': BOUNDS,
              'stock_invocations': 0, 'original_store_modified': False,
              'benchmark_rerun': False, 'full_500_readiness_verified': False,
              'semantic_quality_guaranteed': False}
    conn = client = None
    try:
        require(not args.receipt.exists(), 'receipt_already_exists')
        require((args.credential_file is None) == args.preflight, 'credential_mode')
        if not args.preflight:
            read_authority(sys.stdin.buffer)
        require(regular(args.clone) != regular(args.reference)
                and not os.path.samefile(args.clone, args.reference), 'clone_aliases_reference')
        result['source_inventory_sha256'] = verify_source(args.source, args.r5_manifest)
        require({name: importlib.metadata.version(name) for name in RUNTIME} == RUNTIME, 'runtime_drift')
        api = import_api(args.source)
        # Read-only source snapshot includes WAL content; no immutable-mode shortcut.
        reference = sqlite3.connect(args.reference.as_uri() + '?mode=ro', uri=True, isolation_level=None)
        reference.row_factory = sqlite3.Row
        try:
            reference.execute('PRAGMA query_only=ON')
            reference_before = snapshot(reference)
        finally:
            reference.close()
        conn = api.db.connect(args.clone)
        require(snapshot(conn)['full_sha256'] == reference_before['full_sha256'], 'clone_source_mismatch')
        # Preflight is network-disabled and never mounts or reads real credentials.
        key = 'synthetic-preflight-not-a-key' if args.preflight else read_key(args.credential_file)
        client = api.client(api_key=key, base_url=ENDPOINT,
                            model=MODEL, thinking='disabled')
        result['producer_identity_sha256'] = verify_client(conn, client, api)
        result['max_http_attempts'] = BOUNDS['max_calls'] * api.retry_attempts
        result.update(preflight_checked(conn, client, api) if args.preflight else
                      run_checked(conn, client, api, evidence=result))
        if result['client_closed']:
            client = None  # run_checked has closed the transport.
        conn.close()
        conn = None
        reference = sqlite3.connect(args.reference.as_uri() + '?mode=ro', uri=True, isolation_level=None)
        reference.row_factory = sqlite3.Row
        try:
            reference.execute('PRAGMA query_only=ON')
            require(snapshot(reference)['full_sha256'] == reference_before['full_sha256'],
                    'reference_store_changed')
        finally:
            reference.close()
        require(verify_source(args.source, args.r5_manifest) == result['source_inventory_sha256'],
                'source_changed')
        result['reference_store_unchanged_verified'] = True
        result['source_unchanged_verified'] = True
    except BaseException as exc:
        result['status'] = 'error'
        result['exception_type'] = type(exc).__name__
        result['error_code'] = exc.code if type(exc) is DiagnosticFailure else 'diagnostic_failed'
        if client is not None:
            result['usage'] = api.usage_snapshot(client)
    finally:
        for owner in (client, conn):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as exc:
                    result['status'] = 'error'
                    result['cleanup_exception_type'] = type(exc).__name__
        try:
            save(args.receipt, result)
        except BaseException as exc:
            result['status'] = 'error'
            result['receipt_exception_type'] = type(exc).__name__
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result['status'] == 'error' else 0


if __name__ == '__main__':
    raise SystemExit(main())
