"""Bounded, source-pinned retained-chunk replay; original store is read-only.

No dream, publication, migrations, benchmark scoring, or production mounts.
Offline mode proves contiguous exact source coverage before any paid work.
Live mode makes one extraction invocation (96 completions / 288 HTTP maximum).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import types

CHUNK = 'chk_d9c7f58f71d38ae508b690115183b0000dd34416'
RECORDS_SHA = '245ca6720283a891c2c21b45e5f2eae51c2e326e297d84c9a3fe3f39f4a6ce6f'
SUPPORT_SHA = 'bd8cc72c9bec26e391632af0f133d226e00f367807120b7a9a395e4dc55bf7a5'
SUPERVISOR_SHA = '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'
MODEL = 'deepseek-flash'
ENDPOINT = 'https://api.deepseek.com'
BOUNDS = {'completion_calls': 96, 'http_attempts': 288, 'timeout_seconds': 1200}


def need(value, code):
    if not value:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name, pin):
    need(sha(path) == pin, 'support_pin')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def verify(pin):
    manifest = Path('/diag/manifest.json')
    need(sha(manifest) == pin, 'manifest_pin')
    value = json.loads(manifest.read_text())
    need(value['schema'] == 'r6-fix2-retained-chunk-package-v1', 'manifest_schema')
    source = Path('/candidate')
    actual = {}
    for path in source.rglob('*'):
        need(not path.is_symlink(), 'source_symlink')
        if path.is_file():
            actual[path.relative_to(source).as_posix()] = sha(path)
        else:
            need(path.is_dir(), 'source_special_file')
    need(actual == value['source_sha256'], 'source_inventory_drift')
    need(sha(Path(__file__).resolve()) == value['worker_sha256'], 'worker_drift')
    return value


def source_records():
    from hymem.core.db import register_read_authority_functions
    from hymem.dreaming.phase1 import _claim_sources_for_chunk, _claim_source_record
    from hymem.dreaming.chunks import Chunk
    path = Path('/reference/hymem.sqlite')
    files = [path] + [Path(str(path)+suffix) for suffix in ('-wal', '-shm')
                      if Path(str(path)+suffix).exists()]
    before = {p.name: sha(p) for p in files}
    conn = sqlite3.connect(path.as_uri()+'?mode=ro'+('&immutable=1' if len(files)==1 else ''),
                           uri=True, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        register_read_authority_functions(conn)
        row = conn.execute('SELECT * FROM chunks WHERE id=?', (CHUNK,)).fetchone()
        need(row is not None, 'missing_chunk')
        chunk = Chunk(**{k: row[k] for k in ('id', 'session_id', 'start_message_id',
                                            'end_message_id', 'salience_reason', 'text')})
        sources = _claim_sources_for_chunk(conn, chunk)
        records = tuple((s.message_id, _claim_source_record(s)) for s in sources)
        need(bool(records) and hashlib.sha256('\n'.join(s for _, s in records).encode()).hexdigest()
             == RECORDS_SHA, 'canonical_records_drift')
    finally:
        conn.close()
    return chunk, records, files, before


def partition_proof(records, api):
    unit = api._ExtractionUnit(text='\n'.join(s for _, s in records), source_records=records)
    leaves, failure = api._prepartition(unit)
    need(failure is None and bool(leaves), 'prepartition_still_fails')
    originals = {mid: json.loads(encoded) for mid, encoded in records}
    positions = {mid: 0 for mid in originals}
    metadata = []
    for leaf, depth in leaves:
        need(depth <= api._MAX_SPLIT_DEPTH and len(leaf.text) <= api._MAX_UNSPLITTABLE_INPUT_CHARS,
             'partition_limits')
        parts = []
        for record in leaf.source_records:
            payload = api._source_payload(record)
            need(payload is not None and record[0] in originals, 'partition_payload')
            original = originals[record[0]]
            start = payload.get('source_content_start', 0)
            end = payload.get('source_content_end', len(original['content']))
            need(start == positions[record[0]] and end > start
                 and payload['content'] == original['content'][start:end], 'partition_gap_overlap')
            need(all(payload[k] == original[k] for k in original if k != 'content'
                     and k != 'source_record_version'), 'partition_scope_drift')
            for key in ('source_fragment_context', 'source_boundary_context'):
                context = payload.get(key)
                if context:
                    need(context['content'] == original['content'][context['source_content_start']:
                         context['source_content_end']], 'invented_context')
                    if 'prelude_content' in context:
                        need(context['prelude_content'] == original['content'][context['prelude_source_content_start']:
                             context['prelude_source_content_end']], 'invented_prelude')
            positions[record[0]] = end
            parts.append({'message_id': record[0], 'start': start, 'end': end,
                          'has_table_context': payload.get('source_fragment_context') is not None})
        metadata.append({'encoded_chars': len(leaf.text), 'depth': depth, 'parts': parts})
    need(all(positions[mid] == len(value['content']) for mid, value in originals.items()), 'source_omitted')
    need(len(leaves) <= api._MAX_PREPARTITION_LEAVES, 'leaf_limit')
    return metadata


def environment():
    return dict(PATH='/home/node/hymem-env/bin:/usr/bin:/bin', HOME='/tmp',
                PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', LANG='C.UTF-8')


def worker(mode, pin):
    os.environ.clear()
    os.environ.update(environment())
    logging.disable(logging.CRITICAL)
    manifest = verify(pin)
    support = load(Path('/support/worker.py'), 'retained_replay_support', SUPPORT_SHA)
    need({name: importlib.metadata.version(name) for name in support.RUNTIME} == support.RUNTIME, 'runtime_drift')
    sys.path.insert(0, '/candidate')
    from hymem.extraction import chunk as api
    from hymem.extraction.contract import extraction_contract_binding
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.deadline import MonotonicDeadline, DeadlineBoundLLMClient, use_deadline
    from benchmarks.strictness import usage_snapshot
    result = {'schema': 'r6-fix2-retained-chunk-replay-v1', 'mode': mode, 'manifest_sha256': pin,
              'chunk_id': CHUNK, 'model': MODEL, 'endpoint': ENDPOINT, 'bounds': BOUNDS,
              'extraction_invocations': 0, 'benchmark_rerun': False, 'production_changes': False}
    client = None
    files = before = None
    initial_threads = {t.ident for t in threading.enumerate()}
    try:
        chunk, records, files, before = source_records()
        result['partition'] = partition_proof(records, api)
        result['source_records_sha256'] = RECORDS_SHA
        result['canonical_sources_verified'] = result['exact_partition_verified'] = True
        result['contract_identity'] = extraction_contract_binding()['identity']
        # The archived R5 run must stay inspectable under the successor, but
        # its old extraction contract must not be admitted as current output.
        from benchmarks.lme_registry import _load_registry_artifact
        from benchmarks.lme_protocol import validate_archived_artifact, validate_strict_artifact
        from benchmarks.strictness import BenchmarkIntegrityError
        pointer = Path('/legacy-results/longmemeval-v2-hymem.json')
        archived, name, _digest, compatibility = _load_registry_artifact(pointer)
        need(compatibility == 'pointer-target-digest-validated' and Path(name).name == name,
             'historical_pointer')
        archive_path = pointer.parent/name
        need(sha(archive_path) == '020e84e99ffb758d8fd97a5fc82a411db2387116ef30ed2bda4b1b7f232168f9',
             'historical_archive_pin')
        archived_result = validate_archived_artifact(archived, path=archive_path, require_scored=True)
        need(archived_result['counts']['completed'] == 7 and archived_result['counts']['failed'] == 1,
             'historical_outcome_changed')
        try:
            validate_strict_artifact(archived, path=archive_path, require_scored=True)
        except BenchmarkIntegrityError:
            result['old_contract_not_current'] = True
        else:
            raise RuntimeError('historical_contract_admitted_live')
        result['saved_r5_archive_validated'] = True
        if mode == 'live':
            authority = sys.stdin.buffer.read(256)
            need(authority == (pin+'\n').encode(), 'missing_execution_authority')
        key = 'synthetic-preflight-not-a-key' if mode == 'offline' else support.read_key(Path('/run/deepseek.env'))
        client = OpenAICompatibleClient(api_key=key, base_url=ENDPOINT, model=MODEL, thinking='disabled')
        need(client.transport_integrity_ok is True and client.effective_extra_body == {'thinking': {'type': 'disabled'}},
             'transport_drift')
        if mode == 'live':
            result['extraction_invocations'] = 1
            deadline = MonotonicDeadline.after(BOUNDS['timeout_seconds'])
            with use_deadline(deadline):
                extracted = api.extract_chunk(DeadlineBoundLLMClient(client, deadline), chunk.text,
                                              source_records=records, completion_call_limit=BOUNDS['completion_calls'])
            result['extraction'] = {key: getattr(extracted, key) for key in (
                'failed', 'failure_reason', 'failure_details', 'completion_calls', 'provider_attempts',
                'initial_prepartition_leaves', 'duplicate_triples_collapsed')}
            result['extraction'].update(triples=len(extracted.triples), markers=len(extracted.markers))
            need(all(t.source_message_id in {mid for mid, _ in records} for t in extracted.triples), 'invalid_citation')
            usage = usage_snapshot(client)
            need(usage['calls_available'] and usage['request_attempts_available']
                 and usage['calls'] <= extracted.completion_calls <= BOUNDS['completion_calls']
                 and extracted.provider_attempts == usage['request_attempts'] <= BOUNDS['http_attempts'],
                 'accounting_mismatch')
            result['status'] = 'extraction_failed' if extracted.failed else 'extraction_completed'
        else:
            need(client.call_count == client.request_attempts == 0, 'offline_calls')
            result['status'] = 'offline_passed'
        result['usage'] = usage_snapshot(client)
    except BaseException as exc:
        result['status'] = 'error'
        result['exception_type'] = type(exc).__name__
        if client is not None:
            result['usage'] = usage_snapshot(client)
    finally:
        if client is not None:
            try:
                client.close()
                result['client_closed'] = True
            except BaseException as exc:
                result.update(status='error', close_exception_type=type(exc).__name__)
        try:
            need(files is not None and before == {p.name: sha(p) for p in files}, 'reference_changed')
            verify(pin)
            need(not any(t.ident not in initial_threads for t in threading.enumerate()), 'threads_remain')
            result.update(reference_unchanged=True, source_unchanged=True, threads_clean=True)
        except BaseException as exc:
            result.update(status='error', postcheck_exception_type=type(exc).__name__)
        support.save(Path('/results')/(mode+'.json'), result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] in ('offline_passed', 'extraction_completed') else 1


def supervise(pin):
    verify(pin)
    support = load(Path('/support/worker.py'), 'retained_replay_support', SUPPORT_SHA)
    supervisor = load(Path('/support/supervised_invocation.py'), 'retained_replay_supervisor', SUPERVISOR_SHA)
    def cancel(_signal, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    outcome = supervisor.supervise_invocation(
        [sys.executable, '-I', '-B', '/diag/worker.py', 'live', '--manifest-sha256', pin],
        cwd=Path('/candidate'), env=environment(), output_dir=Path('/results/invocation'),
        timeout_seconds=1260, cleanup_seconds=10, output_limit_bytes=2*1024*1024,
        stdin_bytes=(pin+'\n').encode())
    verify(pin)
    support.save(Path('/results/supervisor.json'), {'manifest_sha256': pin, 'outcome': asdict(outcome)})
    return 0 if outcome.returncode == 0 and outcome.safe_to_continue else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('offline', 'live', 'supervise'))
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    return supervise(args.manifest_sha256) if args.mode == 'supervise' else worker(args.mode, args.manifest_sha256)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({'status': 'diagnostic_failed', 'exception_type': type(exc).__name__}))
        raise SystemExit(1)
