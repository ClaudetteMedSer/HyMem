"""Read-only, bounded R7 sample-eight progress census for SSH stdin.

No application imports, raw log reads, record text, credentials or production
paths. Numeric snapshots are progress only, never final benchmark validation.
"""
import json
from pathlib import Path
import sqlite3
import stat
import sys
import time

if sys.flags.optimize:
    raise RuntimeError('optimized_execution_forbidden')

ROOT = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-v64-sample8-headless-v1/live-results')
IDS = ['c18a7dc8','gpt4_e061b84f','gpt4_483dd43c','gpt4_7de946e7',
       '945e3d21','71315a70','72e3ee87','e61a7584']


def regular(path):
    assert path.resolve() == path and stat.S_ISREG(path.lstat().st_mode)
    assert path.is_relative_to(ROOT)
    return path


def main():
    assert ROOT.resolve() == ROOT and ROOT.is_dir()
    result = {'progress_only': True, 'new_provider_calls': 0, 'files': {}, 'stores': []}
    for relative in ('invocation/stdout.bin', 'invocation/stderr.bin',
                     'invocation/terminal.json', 'supervisor-summary.json',
                     'benchmark/checkpoint.json'):
        path = ROOT / relative
        if path.exists():
            info = regular(path).stat()
            result['files'][relative] = {
                'bytes': info.st_size, 'age_seconds': round(max(0, time.time() - info.st_mtime), 1)}
    paths = list((ROOT / 'stores').glob('hymem-lme-*/hymem.sqlite'))
    checkpoint = ROOT / 'benchmark/checkpoint.json'
    if checkpoint.exists():
        assert regular(checkpoint).stat().st_size <= 16 * 1024 * 1024
        value = json.loads(checkpoint.read_bytes())
        assert value['expected_ids'] == IDS and set(value['entries']) <= set(IDS)
        result['questions'] = {qid: {
            'status': entry['status'] if entry['status'] in ('completed','failed','pending','running') else 'unknown',
            'attempts': entry['attempts'] if type(entry['attempts']) is int else None,
        } for qid,entry in value['entries'].items()}
        result['instrumentation_error_count'] = sum(len(segment.get('instrumentation_errors',[]))
            for segment in value['execution_segments'])
    assert len(paths) <= 9  # Eight stock questions plus, at most, a startup probe.
    for path in sorted(paths):
        regular(path)
        counts = {}
        connection = None
        try:
            connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0.5)
            deadline = time.monotonic() + 1.0
            connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            connection.execute('PRAGMA query_only=ON')
            connection.execute('BEGIN')
            for table in ('sessions', 'messages', 'chunks', 'processed_chunks',
                          'knowledge_graph', 'episodes', 'dream_runs', 'chunk_extraction_attempts'):
                counts[table] = connection.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
            counts['completed_dreams'] = connection.execute(
                'SELECT COUNT(*) FROM dream_runs WHERE ended_at IS NOT NULL').fetchone()[0]
            counts['chunk_attempt_rows_at_least_3'] = connection.execute(
                'SELECT COUNT(*) FROM chunk_extraction_attempts WHERE attempts >= 3').fetchone()[0]
            for field in ('digest_quarantined', 'facts_quarantined', 'profile_quarantined'):
                counts['sessions_' + field] = connection.execute(
                    'SELECT COUNT(*) FROM sessions WHERE ' + field + ' = 1').fetchone()[0]
            counts['sessions_with_summary_failure_marker'] = connection.execute(
                'SELECT COUNT(*) FROM sessions WHERE summary_failure_reason IS NOT NULL').fetchone()[0]
            counts['episodes_matching_published_generation'] = connection.execute(
                'SELECT COUNT(*) FROM episodes e JOIN sessions s ON s.id = e.session_id '
                'WHERE e.digest_generation = s.digest_published_generation').fetchone()[0]
            counts['sessions_with_published_item_frontier'] = connection.execute(
                'SELECT COUNT(*) FROM sessions WHERE digest_published_message_id IS NOT NULL').fetchone()[0]
        except sqlite3.Error as exc:
            counts['snapshot_error_type'] = type(exc).__name__
        finally:
            if connection is not None:
                connection.close()
        result['stores'].append(counts)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

