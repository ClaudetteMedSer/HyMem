"""Root-owned independent boundaries for the stopped o83ipyar reader."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest


ROOT = Path(__file__).parents[3]
SOURCE = ROOT / 'tools/diagnostics/siwc_lme_stopped_store_counts_v1.py'
SPEC = importlib.util.spec_from_file_location('root_stopped_counts', SOURCE)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def functions():
    scope = {name: getattr(tool, name) for name in (
        'ROOT', 'RECEIPT_SHA', 'SCHEMA', 'INDEXING_SCHEMA', 'REASONS', 'PENDING',
        'MALFORMED', 'QUARANTINED', 'REPORT_COUNTS', 'REPORT_OPTIONAL_COUNTS',
        'REPORT_FLAGS', 'MAX_COUNT', 'MAX_ROWS', 'MAX_OUTPUT_BYTES')}
    source = tool.READER.read_bytes()
    assert hashlib.sha256(source).hexdigest() == tool.READER_SHA
    scope['READER_SOURCE'] = source.decode()
    exec(tool.REMOTE.split('\ngated()\n', 1)[0], scope)
    return scope


def zero_summary():
    return dict(schema=tool.INDEXING_SCHEMA, outcome='failure', complete=False,
                healthy=False, failure={'code': 'timeout_during_cycle'},
                max_cycles=100, timeout_s=10800, elapsed_s=10801,
                cycles=0, reports=[], final_status=None, cleanup_errors=[])


def test_real_pinned_functions_and_nullable_zero_cycle():
    scope = functions()
    projected = scope['indexing'](zero_summary())
    assert projected['last_completed_status'] is None
    assert projected['completed_report_first'] is None
    assert projected['status_may_predate_interrupted_cycle'] is True
    assert not any(projected['completed_report_totals'].values())


@pytest.mark.parametrize('key,value', [
    ('runtime_cleanup_verified', False), ('known_tokens', 0),
    ('usage_complete', False), ('known_turns', 3839), ('resource_fault', 'oom'),
    ('scored_count', 1), ('owner_failure', 'failure'),
])
def test_recorded_terminal_corruption_stops_before_receipt(key, value):
    scope = functions()
    evidence = json.loads((ROOT / 'docs/plans/2026-10-02-siwc-o83ipyar-terminal-metadata.json').read_text())
    evidence[key] = value
    scope['reader']['inspect'] = lambda *_: evidence
    scope['reader']['receipt'] = lambda *_: pytest.fail('gate must precede later reads')
    with pytest.raises(ValueError):
        scope['gated']()


@pytest.mark.parametrize('key,value', [
    ('timing_saturated', True), ('failures', 1), ('successes', 0),
    ('first_failure', {'private': 'never export'}), ('known_tokens', 0),
])
def test_observer_corruption_stops_before_receipt(key, value):
    scope = functions()
    evidence = json.loads((ROOT / 'docs/plans/2026-10-02-siwc-o83ipyar-terminal-metadata.json').read_text())
    evidence['siwc_observations']['question.0.ordinary']['summary'][key] = value
    scope['reader']['inspect'] = lambda *_: evidence
    scope['reader']['receipt'] = lambda *_: pytest.fail('observer gate must precede later reads')
    with pytest.raises(ValueError):
        scope['gated']()


def test_sql_authorizer_forbids_private_column_and_db_is_unchanged(tmp_path):
    scope = functions()
    scope['reader']['ROOT_UID'] = os.getuid()
    database = tmp_path / 'test.sqlite'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE chunk_extraction_attempts(attempts, last_failure_reason, last_failure_details)')
        conn.execute('INSERT INTO chunk_extraction_attempts VALUES (1,?,?)',
                     ('NOT-A-REASON-PRIVATE', 'PRIVATE-DETAIL'))
    before = hashlib.sha256(database.read_bytes()).digest()
    result = scope['retry_counts'](database)
    assert result['by_attempt_and_last_reason']['1'] == {'other': 1}
    assert 'PRIVATE' not in json.dumps(result)
    assert hashlib.sha256(database.read_bytes()).digest() == before
    with sqlite3.connect(database) as conn:
        conn.execute('ALTER TABLE chunk_extraction_attempts RENAME TO hidden')
        conn.execute('CREATE VIEW chunk_extraction_attempts AS SELECT attempts,last_failure_details AS last_failure_reason FROM hidden')
    with pytest.raises(sqlite3.DatabaseError):
        scope['retry_counts'](database)


@pytest.mark.parametrize('suffix', ['-wal', '-shm', '-journal'])
def test_all_nonempty_sidecars_fail_closed(tmp_path, suffix):
    scope = functions()
    scope['reader']['ROOT_UID'] = os.getuid()
    database = tmp_path / 'test.sqlite'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE chunk_extraction_attempts(attempts,last_failure_reason)')
    Path(str(database) + suffix).write_bytes(b'nonempty')
    with pytest.raises(ValueError, match='sqlite_sidecar_nonempty'):
        scope['retry_counts'](database)


def test_constants_only_remote_failure_and_local_no_echo(monkeypatch, capsys):
    payload = tool._payload()
    assert "except BaseException:" in payload
    compile(payload, '<pinned-stopped-reader>', 'exec')
    monkeypatch.setattr(tool, '_invoke', lambda _: b'{"secret":"PRIVATE-DO-NOT-EMIT"}')
    assert tool.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        'schema': tool.SCHEMA, 'status': 'inspection_unavailable'}


def test_report_field_projection_matches_exact_frozen_protocol():
    protocol = Path('/private/tmp/hymem-siwc-deadline300-h7CPPm/bundle/candidate/benchmarks/lme_protocol.py')
    assignments = {}
    for node in ast.parse(protocol.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {'_INDEXING_REPORT_FIELDS', 'LME_INDEXING_SUMMARY_VERSION'}:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
    assert assignments['LME_INDEXING_SUMMARY_VERSION'] == tool.INDEXING_SCHEMA
    assert set(assignments['_INDEXING_REPORT_FIELDS']) == set(
        tool.REPORT_COUNTS + tool.REPORT_OPTIONAL_COUNTS + tool.REPORT_FLAGS + ('aggregation_blocking',))


@pytest.mark.parametrize('size', [12, 32769])
def test_actual_local_subprocess_capture_is_bounded_and_discards_stderr(monkeypatch, size):
    real = tool.subprocess.Popen
    def local_only(command, **kwargs):
        assert command == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           '-o', 'ConnectionAttempts=1', 'afrodite', '/usr/bin/python3 -I -B -']
        assert kwargs['stderr'] == tool.subprocess.DEVNULL
        return real([sys.executable, '-I', '-B', '-c',
                     "import sys; sys.stdin.buffer.read(); sys.stderr.write('PRIVATE'); "
                     f"sys.stdout.write('x'*{size})"], **kwargs)
    monkeypatch.setattr(tool.subprocess, 'Popen', local_only)
    if size > tool.MAX_OUTPUT_BYTES:
        with pytest.raises(ValueError, match='inspection_output_oversize'):
            tool._invoke('invented payload')
    else:
        assert tool._invoke('invented payload') == b'x' * size
