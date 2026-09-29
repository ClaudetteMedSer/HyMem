"""Offline exact-request replay of one sealed R8 source and its private capture.

Mount the original package at /diag,/support,/candidate, its reference read-only
at /reference, and live-results read-only at /capture. Run network-none, uid1000,
read-only Docker, with no credential mount. Pass reviewed helper/manifest/capture
inventory SHA256 pins; stdin must be the manifest pin plus newline. Print only
sanitized statistics. Never transport /capture/private off Afrodite.
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
import sys
import types

CHUNKS = ('chk_aace7a800d186c4351ce09307040592724ff7239',
          'chk_57346bc06395cd17a1f1bfd38b6701a3de5e5b8e')


class AuditMismatch(BaseException):
    """Bypass extraction's provider-error recovery on replay mismatches."""


def need(value, code):
    if not value:
        raise AuditMismatch(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def capture_inventory(root):
    need(root.resolve() == root and root.is_dir(), 'capture_path')
    result = {}
    for path in root.rglob('*'):
        need(not path.is_symlink() and (path.is_dir() or path.is_file()), 'capture_special_file')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    return result


def body(path):
    wrapper = json.loads(path.read_text())
    raw = base64.b64decode(wrapper['body_base64'], validate=True)
    need(digest(raw) == wrapper['body_sha256'], 'capture_body_hash')
    return wrapper, json.loads(raw)


def sequence(root):
    requests = sorted(root.glob('*-request.json'))
    responses = sorted(root.glob('*-response.json'))
    need(bool(requests) and len(requests) == len(responses) <= 288, 'capture_pair_count')
    pairs = []
    previous = -1
    for index, (request, response) in enumerate(zip(requests, responses), 1):
        need(request.name == f'{index:03d}-request.json' and response.name == f'{index:03d}-response.json', 'capture_order')
        metadata, wire = body(request)
        response_metadata, answer = body(response)
        chunk_id = metadata['chunk_id']
        need(chunk_id in CHUNKS and CHUNKS.index(chunk_id) >= previous, 'capture_chunk_order')
        previous = CHUNKS.index(chunk_id)
        need(wire.get('model') == 'deepseek-flash' and wire.get('thinking') == {'type':'disabled'}, 'capture_wire_config')
        need(response_metadata['status_code'] == 200, 'unsupported_http_retry')
        pairs.append((chunk_id, wire, answer))
    need({p[0] for p in pairs} == set(CHUNKS), 'capture_chunk_missing')
    return pairs


class ReplayClient:
    def __init__(self, pairs):
        self.pairs = pairs
        self.position = 0
        self.chunk_id = None
        self.successful_responses = 0

    def complete(self, request):
        need(self.position < len(self.pairs) and self.position < 96, 'replay_extra_request')
        chunk_id, wire, answer = self.pairs[self.position]
        expected = {'model':'deepseek-flash',
                    'messages':[{'role':'system','content':request.system},
                                {'role':'user','content':request.user}],
                    'temperature':request.temperature, 'max_tokens':request.max_tokens,
                    'thinking':{'type':'disabled'}}
        if request.response_format == 'json':
            expected['response_format'] = {'type':'json_object'}
        need(chunk_id == self.chunk_id and wire == expected, 'replay_request_mismatch')
        self.position += 1
        choices = answer.get('choices')
        if type(choices) is not list or len(choices) != 1:
            raise RuntimeError('recorded_response_choice_count')
        choice = choices[0]
        if choice.get('finish_reason') != 'stop':
            # This diagnostic replays the reviewed all-200/stop captures only.
            # In particular, length has a dedicated maintained-client error;
            # generic RuntimeError would reproduce the wrong extraction path.
            raise AuditMismatch('unsupported_recorded_finish_reason')
        content = choice.get('message', {}).get('content')
        if type(content) is not str:
            raise RuntimeError('recorded_response_content_type')
        self.successful_responses += 1
        return content


def audit(pin, capture_pin, helper_pin):
    need(digest(Path(__file__).read_bytes()) == helper_pin, 'helper_pin')
    need(sys.stdin.buffer.read(256) == (pin+'\n').encode(), 'audit_authority')
    os.environ.clear()
    os.environ.update(PATH='/home/node/hymem-env/bin:/usr/bin:/bin', HOME='/tmp',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', LANG='C.UTF-8')
    logging.disable(logging.CRITICAL)
    raw = Path('/diag/manifest.json').read_bytes()
    need(digest(raw) == pin, 'manifest_pin')
    manifest = json.loads(raw)
    path = Path('/diag/worker.py')
    need(digest(path.read_bytes()) == manifest['worker_sha256'], 'worker_pin')
    worker = types.ModuleType('r8_original_worker')
    worker.__file__ = str(path)
    sys.modules[worker.__name__] = worker
    exec(compile(path.read_bytes(), str(path), 'exec'), worker.__dict__)
    worker.verify(pin)
    support = worker.load(Path('/support/worker.py'), 'r8_audit_support', manifest['support_sha256'])
    need({n:importlib.metadata.version(n) for n in support.RUNTIME} == support.RUNTIME, 'runtime_drift')
    root = Path('/capture')
    before = capture_inventory(root)
    need(digest(encoded(before)) == capture_pin, 'capture_inventory_pin')
    original = json.loads((root/'live.json').read_text())
    need(original['manifest_sha256'] == pin and original['status'] == 'live_completed', 'capture_source_binding')
    pairs = sequence(root/'private')
    client = ReplayClient(pairs)
    sys.path.insert(0, '/candidate')
    from hymem.extraction import chunk as api
    from hymem.deadline import MonotonicDeadline, use_deadline
    calls = 0
    stats = []
    with use_deadline(MonotonicDeadline.after(1200)):
        for index, cid in enumerate(CHUNKS):
            chunk, records = worker.source_records(cid, manifest['chunks'][cid])
            worker.partition_proof(records, api)
            client.chunk_id = cid
            need(calls < 96, 'replay_completion_cap')
            extracted = api.extract_chunk(client, chunk.text, source_records=records, completion_call_limit=96-calls)
            calls += extracted.completion_calls
            actual_private = {'chunk_id':cid, 'failure_details':extracted.failure_details,
                              'triples':[asdict(t) for t in extracted.triples],
                              'markers':[asdict(m) for m in extracted.markers]}
            expected_private = json.loads((root/'private'/f'{index+1}-extraction.json').read_text())
            need(encoded(actual_private) == encoded(expected_private), 'replay_extraction_mismatch')
            actual = {k:getattr(extracted,k) for k in ('failed','failure_reason','completion_calls',
                'provider_attempts','initial_prepartition_leaves','duplicate_triples_collapsed')}
            actual.update(triples=len(extracted.triples),markers=len(extracted.markers))
            expected = original['chunks'][index]
            need(expected['chunk_id'] == cid and actual == expected['extraction'], 'replay_counters_mismatch')
            stats.append({'chunk_id':cid,'completion_calls':extracted.completion_calls,
                          'provider_attempts':extracted.provider_attempts,'failed':extracted.failed,
                          'exact_private_result_sha256':digest(encoded(actual_private))})
    need(client.position == len(pairs), 'replay_unconsumed_responses')
    worker.verify(pin)
    need(before == capture_inventory(root), 'capture_changed')
    return {'status':'audit_passed','manifest_sha256':pin,'capture_inventory_sha256':capture_pin,
            'source_label':manifest['source_label'],'completion_calls':calls,
            'recorded_http_attempts':len(pairs),'admitted_responses':client.successful_responses,
            'provider_calls':0,'exact_requests_verified':True,'private_results_verified':True,
            'reference_unchanged':True,'source_unchanged':True,'capture_unchanged':True,'chunks':stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--capture-inventory-sha256',required=True,
        help='SHA256 of sorted compact JSON map relative capture filename -> SHA256; encoded() format')
    parser.add_argument('--helper-sha256',required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.manifest_sha256,args.capture_inventory_sha256,args.helper_sha256),sort_keys=True))
    except BaseException as exc:
        # AuditMismatch contains only static codes; never export parser errors.
        print(json.dumps({'status':'audit_failed','code':str(exc) if type(exc) is AuditMismatch else 'audit_exception',
                          'exception_type':type(exc).__name__},sort_keys=True))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
