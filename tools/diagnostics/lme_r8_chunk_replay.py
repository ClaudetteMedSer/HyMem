"""One source tree, two retained chunks, one bounded diagnostic invocation.

Private wire bodies never belong in exported outcomes. No database writes.
Package construction and live launch are separate explicit host actions.
"""
from __future__ import annotations

import argparse
import base64
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

CHUNKS = ('chk_aace7a800d186c4351ce09307040592724ff7239',
          'chk_57346bc06395cd17a1f1bfd38b6701a3de5e5b8e')
MODEL = 'deepseek-flash'
ENDPOINT = 'https://api.deepseek.com'
BOUNDS = {'completion_calls': 96, 'http_attempts': 288, 'timeout_seconds': 1200}


def need(condition, code):
    if not condition:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    result = {}
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'source_symlink')
        need(path.is_file() or path.is_dir(), 'source_special_file')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def load(path, name, pin):
    need(sha(path) == pin, 'support_pin')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def verify(pin):
    path = Path('/diag/manifest.json')
    need(sha(path) == pin, 'manifest_pin')
    manifest = json.loads(path.read_text())
    need(manifest['schema'] == 'r8-retained-chunks-v1', 'manifest_schema')
    need(manifest['bounds'] == BOUNDS and set(manifest['chunks']) == set(CHUNKS), 'manifest_scope')
    need(inventory(Path('/candidate')) == manifest['source_sha256'], 'source_inventory_drift')
    need(sha(Path(__file__)) == manifest['worker_sha256'], 'worker_drift')
    for name, digest in manifest['reference_sha256'].items():
        need(name in ('hymem.sqlite', 'hymem.sqlite-wal', 'hymem.sqlite-shm'), 'reference_filename')
        need(sha(Path('/reference') / name) == digest, 'reference_drift')
    actual = {p.name for p in Path('/reference').glob('hymem.sqlite*')}
    need(actual == set(manifest['reference_sha256']), 'reference_inventory')
    return manifest


def source_records(chunk_id, expected):
    from hymem.core.db import register_read_authority_functions
    from hymem.dreaming.phase1 import _claim_sources_for_chunk, _claim_source_record
    from hymem.dreaming.chunks import Chunk
    path = Path('/reference/hymem.sqlite')
    conn = sqlite3.connect(path.as_uri() + '?mode=ro' +
                          ('&immutable=1' if not Path(str(path)+'-wal').exists() else ''),
                          uri=True, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        register_read_authority_functions(conn)
        row = conn.execute('SELECT * FROM chunks WHERE id=?', (chunk_id,)).fetchone()
        need(row is not None, 'missing_chunk')
        chunk = Chunk(**{k: row[k] for k in ('id', 'session_id', 'start_message_id',
                                           'end_message_id', 'salience_reason', 'text')})
        records = tuple((s.message_id, _claim_source_record(s))
                        for s in _claim_sources_for_chunk(conn, chunk))
        need(bool(records) and hashlib.sha256('\n'.join(s for _, s in records).encode()).hexdigest()
             == expected, 'canonical_records_drift')
        return chunk, records
    finally:
        conn.close()


def partition_proof(records, api):
    originals = {mid: json.loads(encoded) for mid, encoded in records}
    need(len(originals) == len(records), 'duplicate_source_id')
    leaves, failure = api._prepartition(api._ExtractionUnit(
        text='\n'.join(s for _, s in records), source_records=records))
    need(failure is None and bool(leaves), 'prepartition_failed')
    positions = {mid: 0 for mid in originals}
    metadata = []
    for leaf, depth in leaves:
        need(depth <= api._MAX_SPLIT_DEPTH and
             len(leaf.text) <= api._MAX_UNSPLITTABLE_INPUT_CHARS, 'partition_limits')
        parts = []
        for record in leaf.source_records:
            payload = api._source_payload(record)
            need(payload is not None and record[0] in originals, 'partition_payload')
            original = originals[record[0]]
            start = payload.get('source_content_start', 0)
            end = payload.get('source_content_end', len(original['content']))
            need(start == positions[record[0]] and end > start and
                 payload['content'] == original['content'][start:end], 'partition_gap_overlap')
            need(all(payload.get(k) == original[k] for k in original
                     if k not in ('content', 'source_record_version')), 'partition_scope_drift')
            for key in ('source_fragment_context', 'source_boundary_context'):
                context = payload.get(key)
                if context:
                    need(context['content'] == original['content'][context['source_content_start']:
                         context['source_content_end']], 'invented_context')
                    if 'prelude_content' in context:
                        need(context['prelude_content'] == original['content'][context['prelude_source_content_start']:
                             context['prelude_source_content_end']], 'invented_prelude')
            positions[record[0]] = end
            parts.append({'message_id': record[0], 'start': start, 'end': end})
        metadata.append({'depth': depth, 'encoded_chars': len(leaf.text), 'parts': parts})
    need(all(positions[mid] == len(payload['content']) for mid, payload in originals.items()), 'source_omitted')
    need(len(leaves) <= api._MAX_PREPARTITION_LEAVES, 'leaf_limit')
    return metadata


class WireCapture:
    """SDK HTTP event hooks: bodies only, never headers or credentials."""
    def __init__(self, root, deadline):
        self.root, self.deadline = root, deadline
        self.attempts = 0
        self.chunk_id = None

    def save(self, name, value):
        with (self.root/name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)

    def request(self, request):
        self.deadline.check()
        need(str(request.url) == ENDPOINT + '/chat/completions', 'wire_endpoint')
        need(self.attempts < BOUNDS['http_attempts'], 'http_cap')
        self.attempts += 1
        request.extensions['r8_capture_index'] = self.attempts
        body = request.read()
        payload = json.loads(body)
        need(payload['model'] == MODEL and payload['thinking'] == {'type': 'disabled'}, 'wire_config')
        self.save(f'{self.attempts:03d}-request.json', {'chunk_id': self.chunk_id,
                  'body_base64': base64.b64encode(body).decode(), 'body_sha256': hashlib.sha256(body).hexdigest()})

    def response(self, response):
        body = response.read()
        index = response.request.extensions['r8_capture_index']
        self.save(f'{index:03d}-response.json', {'status_code': response.status_code,
                  'body_base64': base64.b64encode(body).decode(), 'body_sha256': hashlib.sha256(body).hexdigest()})


def environment():
    return dict(PATH='/home/node/hymem-env/bin:/usr/bin:/bin', HOME='/tmp',
                PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', LANG='C.UTF-8')


def worker(mode, pin):
    os.environ.clear()
    os.environ.update(environment())
    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    manifest = verify(pin)
    support = load(Path('/support/worker.py'), 'r8_support', manifest['support_sha256'])
    need({n: importlib.metadata.version(n) for n in support.RUNTIME} == support.RUNTIME, 'runtime_drift')
    sys.path.insert(0, '/candidate')
    from hymem.extraction import chunk as api
    from hymem.extraction.contract import extraction_contract_binding
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.deadline import MonotonicDeadline, DeadlineBoundLLMClient, use_deadline
    from benchmarks.strictness import usage_snapshot
    initial_threads = {t.ident for t in threading.enumerate()}
    result = {'schema': 'r8-retained-chunk-outcome-v1', 'mode': mode,
              'manifest_sha256': pin, 'source_label': manifest['source_label'],
              'bounds': BOUNDS, 'extraction_invocations': 0, 'chunks': []}
    client = None
    try:
        prepared = []
        for chunk_id in CHUNKS:
            chunk, records = source_records(chunk_id, manifest['chunks'][chunk_id])
            proof = partition_proof(records, api)
            result['chunks'].append({'chunk_id': chunk_id, 'source_records_sha256': manifest['chunks'][chunk_id],
                                     'partition': proof, 'exact_partition_verified': True})
            prepared.append((chunk, records))
        result['contract_identity'] = extraction_contract_binding()['identity']
        if mode == 'live':
            need(sys.stdin.buffer.read(256) == (pin+'\n').encode(), 'execution_authority')
        deadline = MonotonicDeadline.after(BOUNDS['timeout_seconds'])
        capture = WireCapture(Path('/results/private'), deadline)
        if mode == 'live':
            private = Path('/results/private')
            private.mkdir(mode=0o700)
        import openai
        factory = openai.DefaultHttpxClient
        # The ordinary SDK client is constructed with public hooks before
        # HyMem seals it, identically in offline/live modes.
        openai.DefaultHttpxClient = lambda **kw: factory(**kw, event_hooks={
            'request': [capture.request], 'response': [capture.response]})
        try:
            key = support.read_key(Path('/run/deepseek.env')) if mode == 'live' else 'synthetic-preflight-not-a-key'
            client = OpenAICompatibleClient(api_key=key, base_url=ENDPOINT, model=MODEL, thinking='disabled')
        finally:
            openai.DefaultHttpxClient = factory
        need(client.transport_integrity_ok and client.effective_extra_body == {'thinking': {'type': 'disabled'}}, 'transport_drift')
        if mode == 'live':
            calls = 0
            with use_deadline(deadline):
                for index, (chunk, records) in enumerate(prepared):
                    deadline.check()
                    need(calls < BOUNDS['completion_calls'], 'completion_cap')
                    capture.chunk_id = chunk.id
                    result['extraction_invocations'] += 1
                    extracted = api.extract_chunk(DeadlineBoundLLMClient(client, deadline),
                        chunk.text, source_records=records,
                        completion_call_limit=BOUNDS['completion_calls']-calls)
                    calls += extracted.completion_calls
                    # failure_details may contain corpus/provider excerpts: private only.
                    support.save(private/f'{index+1}-extraction.json', {
                        'chunk_id': chunk.id, 'failure_details': extracted.failure_details,
                        'triples': [asdict(t) for t in extracted.triples],
                        'markers': [asdict(m) for m in extracted.markers]})
                    result['chunks'][index]['extraction'] = {k: getattr(extracted, k) for k in (
                        'failed', 'failure_reason', 'completion_calls', 'provider_attempts',
                        'initial_prepartition_leaves', 'duplicate_triples_collapsed')}
                    result['chunks'][index]['extraction'].update(triples=len(extracted.triples), markers=len(extracted.markers))
                    need(all(t.source_message_id in {mid for mid, _ in records} for t in extracted.triples), 'invalid_citation')
                    need(calls <= BOUNDS['completion_calls'] and client.request_attempts == capture.attempts <= BOUNDS['http_attempts'], 'accounting_mismatch')
            result['status'] = 'live_completed'
            result['extraction_success'] = all(not c['extraction']['failed'] for c in result['chunks'])
        else:
            need(client.call_count == client.request_attempts == 0, 'offline_calls')
            result['status'] = 'offline_passed'
        result['usage'] = usage_snapshot(client)
    except BaseException as exc:
        result.update(status='error', exception_type=type(exc).__name__)
    finally:
        if client is not None:
            try:
                result['usage'] = usage_snapshot(client)
                client.close()
                result['client_closed'] = True
            except BaseException as exc:
                result.update(status='error', close_exception_type=type(exc).__name__)
        try:
            verify(pin)
            need(not any(t.ident not in initial_threads for t in threading.enumerate()), 'threads_remain')
            result.update(reference_unchanged=True, source_unchanged=True, threads_clean=True)
        except BaseException as exc:
            result.update(status='error', postcheck_exception_type=type(exc).__name__)
        support.save(Path('/results')/(mode+'.json'), result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] in ('offline_passed', 'live_completed') else 1


def supervise(pin):
    manifest = verify(pin)
    support = load(Path('/support/worker.py'), 'r8_support', manifest['support_sha256'])
    supervisor = load(Path('/support/supervised_invocation.py'), 'r8_supervisor', manifest['supervisor_sha256'])
    def cancel(_signal, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    outcome = supervisor.supervise_invocation(
        [sys.executable, '-I', '-B', '/diag/worker.py', 'live', '--manifest-sha256', pin],
        cwd=Path('/candidate'), env=environment(), output_dir=Path('/results/invocation'),
        timeout_seconds=1200, cleanup_seconds=10, output_limit_bytes=2*1024*1024,
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
