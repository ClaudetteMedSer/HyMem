"""R8 normal-digest and explicit-recovery replay, without original writes.

Retained summaries stay private on Afrodite. Only invented control summaries
are exported for human semantic review. One shared budget covers every phase.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, replace
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

SCHEMA = 'r8-summary-repair-verification-package-v2'
BOUNDS = {'completion_calls': 192, 'http_attempts': 576, 'timeout_seconds': 2700}
NORMAL_PARAMETERS = {'max_chars': 12000, 'max_tokens': 3072, 'granular': False,
    'max_episodes': 12, 'separate_summary': True, 'prior_summary_is_stale': False}
RECOVERY_PARAMETERS = {'max_attempts': 3, 'max_chars': 8000, 'max_tokens': 3072}
SUPPORT_SHA = 'bd8cc72c9bec26e391632af0f133d226e00f367807120b7a9a395e4dc55bf7a5'
SUPERVISOR_SHA = '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'
MODEL, ENDPOINT = 'deepseek-flash', 'https://api.deepseek.com'
SOURCE, DIAG, RESULTS = Path('/candidate'), Path('/diag'), Path('/results')
REFERENCE, CLONE, CONTROL = Path('/reference/hymem.sqlite'), Path('/work/hymem.sqlite'), Path('/work/control.sqlite')
PREVIOUS = Path('/previous')
PREVIOUS_MANIFEST = Path('/previous-manifest.json')
PREVIOUS_PINS = {
    'manifest_sha256': 'cf341386e467e7879cbabaa8380b761c3e905ec674757d3990842dbf0f81e67c',
    'capture_inventory_sha256': 'fd645967f1596857eeb1621d73bc82d4dd3349991c94a20fd6e35e6a3779b325',
    'request_file': '035-request.json',
    'request_sha256': 'ad07d82fac2f0b65c8fd2415f2bf656dfedb54177396e9ae1536d73b267afb78',
    'request_body_sha256': '807dc2b13fbcfd1fd6940149f4472144926549288a5c3172e62d8916fdc35a36',
    'returned_chars': 505,
}


def need(value, code):
    if not value:
        raise RuntimeError(code)


def sha(path):
    need(path.resolve() == path and path.is_file() and not path.is_symlink(), 'regular_file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    result = {}
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        # Only generated Python/test caches are excluded; SQL, resources and
        # tests are part of the producer inventory too.
        if any(p in ('__pycache__', '.pytest_cache') for p in rel.parts) or path.suffix in ('.pyc', '.pyo'):
            continue
        need(not path.is_symlink() and (path.is_file() or path.is_dir()), 'source_special_file')
        if path.is_file():
            result[rel.as_posix()] = sha(path)
    return result


def load(path, name, pin):
    need(sha(path) == pin, 'support_pin')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def verify(pin):
    need(sha(DIAG/'manifest.json') == pin, 'manifest_pin')
    value = json.loads((DIAG/'manifest.json').read_text())
    need(value['schema'] == SCHEMA and value['bounds'] == BOUNDS
         and value['retained_targets'] == 14 and value['control_targets'] == 4
         and value['normal_parameters'] == NORMAL_PARAMETERS
         and value['recovery_parameters'] == RECOVERY_PARAMETERS
         and value['previous_pins'] == PREVIOUS_PINS
         and value['repair_control_targets'] == 4, 'manifest_scope')
    need(inventory(SOURCE) == value['source_sha256'], 'source_inventory_drift')
    need(sha(Path(__file__).resolve()) == value['worker_sha256'], 'worker_drift')
    need(sha(Path('/support/worker.py')) == value['support_sha256'] == SUPPORT_SHA
         and sha(Path('/support/supervised_invocation.py')) == value['supervisor_sha256'] == SUPERVISOR_SHA,
         'support_contract')
    need(sha(Path('/support/r6_summary.py')) == value['r6_summary_sha256'], 'controls_drift')
    need({p.name: sha(p) for p in REFERENCE.parent.glob('hymem.sqlite*')}
         == value['reference_sha256'], 'reference_drift')
    previous_request()
    return value


def captured_body(path):
    wrapper = json.loads(path.read_text())
    raw = base64.b64decode(wrapper['body_base64'], validate=True)
    need(hashlib.sha256(raw).hexdigest() == wrapper['body_sha256'], 'previous_body_hash')
    return wrapper, json.loads(raw)


def previous_request():
    """Verify the previous private inventory and exact observed primary input."""
    need(sha(PREVIOUS_MANIFEST) == PREVIOUS_PINS['manifest_sha256'], 'previous_manifest_pin')
    previous_manifest = json.loads(PREVIOUS_MANIFEST.read_text())
    need(previous_manifest['model'] == MODEL and previous_manifest['endpoint'] == ENDPOINT,
         'previous_route')
    live = json.loads((PREVIOUS/'live.json').read_text())
    need(live['manifest_sha256'] == PREVIOUS_PINS['manifest_sha256'], 'previous_receipt_binding')
    files = live['private_capture']['files']
    raw = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    need(hashlib.sha256(raw).hexdigest() == PREVIOUS_PINS['capture_inventory_sha256']
         == live['private_capture']['inventory_sha256'], 'previous_inventory_pin')
    for name, entry in files.items():
        need(Path(name).name == name and re_match_capture_name(name), 'previous_filename')
        need(sha(PREVIOUS/'private'/name) == entry['sha256'], 'previous_capture_drift')
    path = PREVIOUS/'private'/PREVIOUS_PINS['request_file']
    need(sha(path) == PREVIOUS_PINS['request_sha256'], 'previous_request_pin')
    wrapper, wire = captured_body(path)
    need(wrapper['body_sha256'] == PREVIOUS_PINS['request_body_sha256']
         and wrapper['phase'] == 'explicit_recovery', 'previous_request_binding')
    need(set(wire) == {'model', 'messages', 'temperature', 'max_tokens', 'thinking', 'response_format'}
         and wire['model'] == MODEL and wire['thinking'] == {'type': 'disabled'}
         and wire['response_format'] == {'type': 'json_object'}
         and wire['temperature'] == 0.0 and wire['max_tokens'] == 3072
         and len(wire['messages']) == 2
         and [m['role'] for m in wire['messages']] == ['system', 'user']
         and all(set(m) == {'role', 'content'} and type(m['content']) is str for m in wire['messages']),
         'previous_wire_contract')
    envelope = json.loads(wire['messages'][1]['content'])
    need(set(envelope) == {'prior_summary', 'new_material'}
         and all(type(v) is str for v in envelope.values()), 'previous_primary_input')
    response_wrapper, response = captured_body(PREVIOUS/'private'/'035-response.json')
    need(response_wrapper['status_code'] == 200 and response_wrapper['phase'] == 'explicit_recovery'
         and response_wrapper['session_sha256'] == wrapper['session_sha256']
         and len(response['choices']) == 1 and response['choices'][0]['finish_reason'] == 'stop',
         'previous_primary_response')
    content = response['choices'][0]['message']['content']
    parsed = json.loads(content)
    need(set(parsed) == {'summary'} and type(parsed['summary']) is str
         and len(parsed['summary'].strip()) == PREVIOUS_PINS['returned_chars'], 'previous_cap_observation')
    return wire, wrapper['session_sha256']


def re_match_capture_name(name):
    import re
    return re.fullmatch(r'[0-9]{3}-(request|response)\.json', name) is not None


def retained_primary_request(wire):
    from hymem.extraction.llm import LLMRequest
    return LLMRequest(system=wire['messages'][0]['content'], user=wire['messages'][1]['content'],
                      temperature=wire['temperature'], max_tokens=wire['max_tokens'], response_format='json')


def read_reference():
    from hymem.core.db import register_read_authority_functions
    conn = sqlite3.connect(REFERENCE.as_uri()+'?mode=ro', uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    conn.execute('PRAGMA temp_store=MEMORY')
    register_read_authority_functions(conn)
    return conn


def targets(conn, api, expected=14):
    rows = conn.execute('SELECT * FROM sessions ORDER BY id').fetchall()
    selected = [row['id'] for row in rows if api.classify_summary_state(conn, row['id'])['degraded']]
    need(len(selected) == expected, 'target_count')
    for row in rows:
        if row['id'] in selected:
            need(row['summary_failure_reason'] == 'summary_output_cap'
                 and row['auto_summary'] is None
                 and row['digest_published_message_id'] == row['coverage_message_id']
                 and row['coverage_message_id'] is not None, 'retained_target_state')
    need(conn.execute('SELECT COUNT(*) FROM summary_recovery').fetchone()[0] == 0
         and conn.execute('SELECT COUNT(*) FROM run_lock').fetchone()[0] == 0, 'fresh_clone_required')
    return selected


class SharedBudget:
    """Same logical/HTTP envelope and absolute deadline across all phases."""
    def __init__(self, client, deadline, cap=192, http_cap=576, capture=None):
        self.client, self.deadline = client, deadline
        self.cap, self.http_cap, self.calls = cap, http_cap, 0
        self.capture = capture

    def __getattr__(self, key):
        return getattr(self.client, key)

    def memory_producer_declaration(self):
        # Custom diagnostics are not maintained transparent producer proxies.
        # Declare our own implementation honestly while preserving the exact
        # provider's route, request and retry semantics. The package pins this
        # wrapper code; never claim its identity equals the unwrapped client.
        original = self.client.memory_producer_declaration()
        implementation = hashlib.sha256(
            (sha(Path(__file__).resolve())+'|'+original.implementation).encode()).hexdigest()
        return replace(original, client_id='hymem.diagnostics.r8.SharedBudget',
                       implementation='r8-summary-budget-v1:sha256:'+implementation)

    def complete(self, request):
        from hymem.deadline import DeadlineExceeded
        self.deadline.check()
        # A verified shipped completion can make at most three HTTP attempts.
        # Reserve its whole retry envelope before dispatch, including errors.
        if self.calls >= self.cap or self.client.request_attempts + 3 > self.http_cap:
            raise DeadlineExceeded('diagnostic_budget_exhausted')
        self.calls += 1
        if self.capture is not None:
            self.capture.select_recovery_session()
        result = self.client.complete(request)
        self.deadline.check()
        need(self.client.request_attempts <= self.http_cap, 'http_budget')
        return result


class WireCapture:
    """Reviewed R8 body-only hooks, installed before transport sealing."""
    def __init__(self, root, deadline):
        self.root, self.deadline = root, deadline
        self.attempts, self.responses = 0, 0
        self.phase, self.session_sha256 = None, None
        self.recovery_conn = None
        self.files = {}

    def select_recovery_session(self):
        if self.recovery_conn is not None:
            # The admitted clone has no jobs before this single stock run.
            # Stock processes sessions serially and inserts each new job as
            # it starts; its newest row is the currently dispatched session,
            # including subsequent windows of that same private job.
            row = self.recovery_conn.execute(
                'SELECT session_id FROM summary_recovery ORDER BY rowid DESC LIMIT 1').fetchone()
            need(row is not None, 'capture_recovery_session')
            self.session_sha256 = hashlib.sha256(row[0].encode()).hexdigest()

    def save(self, index, kind, body, **metadata):
        name = f'{index:03d}-{kind}.json'
        value = {'phase': self.phase, 'session_sha256': self.session_sha256,
                 'body_base64': base64.b64encode(body).decode(),
                 'body_sha256': hashlib.sha256(body).hexdigest(), **metadata}
        with (self.root/name).open('x') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
        self.files[name] = {'sha256': sha(self.root/name), 'phase': value['phase'],
            'session_sha256': value['session_sha256'], 'body_sha256': value['body_sha256']}

    def request(self, request):
        from hymem.deadline import DeadlineExceeded
        self.deadline.check()
        need(str(request.url) == ENDPOINT+'/chat/completions', 'wire_endpoint')
        if self.attempts >= BOUNDS['http_attempts']:
            raise DeadlineExceeded('diagnostic_http_budget_exhausted')
        need(self.phase in ('normal_retained', 'explicit_recovery', 'normal_controls',
                            'repair_contract_controls', 'retained_repair_case')
             and isinstance(self.session_sha256, str) and len(self.session_sha256) == 64, 'capture_label')
        body = request.read()
        payload = json.loads(body)
        need(payload['model'] == MODEL and payload['thinking'] == {'type': 'disabled'}, 'wire_config')
        self.attempts += 1
        request.extensions['r8_capture_index'] = self.attempts
        request.extensions['r8_capture_phase'] = self.phase
        request.extensions['r8_capture_session'] = self.session_sha256
        self.save(self.attempts, 'request', body)

    def response(self, response):
        body = response.read()
        index = response.request.extensions['r8_capture_index']
        self.save(index, 'response', body, status_code=response.status_code,
                  phase=response.request.extensions['r8_capture_phase'],
                  session_sha256=response.request.extensions['r8_capture_session'])
        self.responses += 1

    def public_inventory(self):
        need(all(sha(self.root/name) == value['sha256'] for name, value in self.files.items()), 'capture_drift')
        encoded = json.dumps(self.files, sort_keys=True, separators=(',', ':')).encode()
        return {'requests': self.attempts, 'responses': self.responses, 'files': self.files,
                'inventory_sha256': hashlib.sha256(encoded).hexdigest()}


def capture_client(api, key, capture):
    import openai
    factory = openai.DefaultHttpxClient
    # Public hooks enter the owned transport before HyMem seals its identity.
    # Restore the constructor immediately, including construction failures.
    openai.DefaultHttpxClient = lambda **kw: factory(**kw, event_hooks={
        'request': [capture.request], 'response': [capture.response]})
    try:
        return api.client(api_key=key, base_url=ENDPOINT, model=MODEL, thinking='disabled')
    finally:
        openai.DefaultHttpxClient = factory


def normal_walk(conn, session_ids, client, api, support, private_root=None, *, invented=False):
    """Call normal digest from the beginning, carrying only private context."""
    from hymem.dreaming.digest import extract_session_digest
    from hymem.dreaming.lossless import lossless_cursor_is_valid
    before = support.snapshot(conn)
    evidence, private = [], []
    for sid in session_ids:
        if client.capture is not None:
            client.capture.phase = 'normal_controls' if invented else 'normal_retained'
            client.capture.session_sha256 = hashlib.sha256(sid.encode()).hexdigest()
        cursor, summary, windows = (None, None, 0), None, 0
        reasons, stages = [], []
        complete = False
        while True:
            result = extract_session_digest(conn, sid, client, **NORMAL_PARAMETERS,
                since_message_id=cursor[0], partial_message_id=cursor[1],
                since_message_offset=cursor[2], prior_summary=summary)
            if result is None:
                complete = True
                break
            windows += 1
            summary_failure = getattr(result, 'summary_failure_reason', None)
            if result.parse_failed or result.failure_reason is not None or summary_failure is not None:
                reasons.append(summary_failure or result.failure_reason or 'unspecified_failure')
                stages.append(getattr(result, 'failure_stage', None))
                break  # No automatic reroll of a held window.
            after = (result.covered_message_id, result.partial_message_id, result.next_message_offset)
            need(after != cursor and lossless_cursor_is_valid(conn, sid, *after), 'normal_cursor_progress')
            cursor = after
            if result.summary is not None:
                summary = result.summary
            if result.caught_up:
                tail = conn.execute('SELECT coverage_message_id FROM sessions WHERE id=?', (sid,)).fetchone()[0]
                need(cursor == (tail, None, 0), 'normal_tail_mismatch')
                if not isinstance(summary, str) or not summary.strip():
                    reasons.append('summary_validation_failure')
                    stages.append('normal_tail')
                    break
                complete = True
                break
        item = {'session_sha256': hashlib.sha256(sid.encode()).hexdigest(), 'windows': windows,
                'complete': complete, 'summary_chars': len(summary or ''),
                'summary_sha256': hashlib.sha256((summary or '').encode()).hexdigest(),
                'failure_reasons': reasons, 'failure_stages': stages}
        if invented:
            item.update(session_id=sid, summary=summary)
        evidence.append(item)
        private.append({'session_id': sid, 'cursor': cursor, 'summary': summary})
    after = support.snapshot(conn)
    need(after['full_sha256'] == before['full_sha256'], 'normal_digest_database_write')
    if private_root is not None:
        support.save(private_root/'normal-context.json', private)
    return {'parameters': NORMAL_PARAMETERS, 'sessions': evidence,
            'complete_sessions': sum(e['complete'] for e in evidence),
            'database_unchanged': True}


def explicit_recovery(conn, client, api, support, session_ids):
    before = support.snapshot(conn)
    initial = {sid: api.classify_summary_state(conn, sid) for sid in before['sessions']}
    report = None
    if client.capture is not None:
        client.capture.phase = 'explicit_recovery'
        client.capture.recovery_conn = conn
    try:
        # One stock invocation: it holds a rejected session once, never rerolls.
        remaining = client.cap - client.calls
        need(remaining > 0, 'recovery_budget_empty')
        report = api.recovery.run_summary_recovery(conn, client,
            max_calls=min(100, remaining), **RECOVERY_PARAMETERS,
            timeout_seconds=max(0.001, client.deadline.remaining()))
    finally:
        if client.capture is not None:
            client.capture.recovery_conn = None
        after = support.snapshot(conn)
        support.assert_unchanged(before, after)
        support.integrity(conn)
        need(not conn.in_transaction and conn.execute('SELECT 1 FROM run_lock').fetchone() is None,
             'recovery_lease_cleanup')
    safe = api.safe_report(report)
    changes = support.audit_summary_changes(conn, before, after, initial, safe, api)
    health = api.durable_summary_status(conn)
    return {'parameters': RECOVERY_PARAMETERS, 'recovery': safe, 'changes': changes, 'health_after': health,
            'all_non_summary_state_unchanged': True,
            'recovered_all': safe['published'] == len(session_ids) and safe['remaining'] == 0
                and safe['held'] == safe['exhausted'] == 0 and health['summary_healthy'] is True}


def repair_contract_controls(conn, session_ids, client, api, support, criteria):
    """Direct actual repair prompts, not synthetic primary or stock recovery."""
    from hymem.dreaming.lossless import lossless_cursor_is_valid
    from hymem.extraction.jsonio import loads_exact_or_fenced
    before = support.snapshot(conn)
    evidence = []
    for sid in session_ids:
        tail = conn.execute('SELECT coverage_message_id FROM sessions WHERE id=?', (sid,)).fetchone()[0]
        job = dict(session_id=sid, target_message_id=tail, cursor_message_id=None,
                   cursor_partial_message_id=None, cursor_offset=0, draft='')
        windows, complete = [], False
        while True:
            request, after = api.recovery._request(conn, job, RECOVERY_PARAMETERS['max_chars'],
                                                   RECOVERY_PARAMETERS['max_tokens'])
            if client.capture is not None:
                client.capture.phase = 'repair_contract_controls'
                client.capture.session_sha256 = hashlib.sha256(sid.encode()).hexdigest()
            # None is the repair contract's generic feedback. No invented
            # paid primary, failed length, or durable attempt is represented.
            raw = client.complete(api.recovery._cap_recovery_request(request))
            selected, failure = api.recovery._parse_repair_alternatives(raw)
            data = loads_exact_or_fenced(raw) if isinstance(raw, str) and len(raw) <= 65536 else None
            alternatives = data.get('alternatives') if isinstance(data, dict) else None
            if not isinstance(alternatives, list) or not all(type(v) is str for v in alternatives):
                alternatives = None
            windows.append({'alternatives': alternatives, 'selected_overview': selected,
                            'failure_reason': failure, 'review_criteria': criteria[sid]['review']})
            if failure is not None:
                break
            need(after != api.recovery._position(job) and lossless_cursor_is_valid(conn, sid, *after),
                 'repair_control_progress')
            if after == (tail, None, 0):
                complete = True
                break
            job.update(cursor_message_id=after[0], cursor_partial_message_id=after[1],
                       cursor_offset=after[2], draft=selected)
        evidence.append({'session_id': sid, 'complete': complete, 'windows': windows})
    need(support.snapshot(conn)['full_sha256'] == before['full_sha256'], 'repair_control_database_write')
    return {'verification_kind': 'direct_repair_prompt_contract_not_stock_invocation',
            'synthetic_primary_calls': 0, 'sessions': evidence,
            'complete_sessions': sum(item['complete'] for item in evidence), 'database_unchanged': True}


def retained_repair_case(wire, session_sha256, client, api, support, private_root):
    request = retained_primary_request(wire)
    repair = api.recovery._cap_recovery_request(request, PREVIOUS_PINS['returned_chars'])
    if client.capture is not None:
        client.capture.phase, client.capture.session_sha256 = 'retained_repair_case', session_sha256
    before = client.calls
    raw = client.complete(repair)
    selected, failure = api.recovery._parse_repair_alternatives(raw)
    need(client.calls == before + 1, 'retained_repair_call_accounting')
    support.save(private_root/'retained-repair-case.json', {'raw': raw, 'selected': selected,
                                                          'failure_reason': failure})
    return {'verification_kind': 'direct_repair_prompt_on_exact_recorded_primary_input',
            'previous_pins': PREVIOUS_PINS, 'completion_calls': 1, 'passed': failure is None,
            'failure_reason': failure, 'selected_chars': len(selected or ''),
            'selected_sha256': hashlib.sha256((selected or '').encode()).hexdigest(),
            'database_writes': 0, 'raw_content_exported': False}


def environment():
    return dict(PATH='/home/node/hymem-env/bin:/usr/bin:/bin', HOME='/tmp',
                PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', LANG='C.UTF-8')


def worker(mode, pin):
    os.umask(0o077)
    os.environ.clear()
    os.environ.update(environment())
    logging.disable(logging.CRITICAL)
    result = {'schema': 'r8-summary-repair-verification-v2', 'mode': mode, 'manifest_sha256': pin,
              'bounds': BOUNDS, 'retained_targets': 14, 'control_targets': 4,
              'normal_parameters': NORMAL_PARAMETERS, 'recovery_parameters': RECOVERY_PARAMETERS,
              'production_changes': False, 'benchmark_rerun': False,
              'semantic_review_required': True, 'stock_invocations': 0}
    client = budget = capture = support = reference = clone = controls = None
    original = None
    threads = {t.ident for t in threading.enumerate()}
    try:
        need(os.geteuid() == 1000, 'isolated_user')
        manifest = verify(pin)
        previous_wire, previous_session = previous_request()
        result['previous_pins_verified'] = PREVIOUS_PINS
        support = load(Path('/support/worker.py'), 'r8_summary_support', SUPPORT_SHA)
        r6 = load(Path('/support/r6_summary.py'), 'r8_summary_controls', manifest['r6_summary_sha256'])
        r6.REFERENCE, r6.CLONE, r6.CONTROL = REFERENCE, CLONE, CONTROL
        need({n: importlib.metadata.version(n) for n in support.RUNTIME} == support.RUNTIME, 'runtime_drift')
        api = support.import_api(SOURCE)
        need(callable(api.recovery._parse_repair_alternatives), 'repair_alternatives_contract')
        prepared_repair = api.recovery._cap_recovery_request(
            retained_primary_request(previous_wire), PREVIOUS_PINS['returned_chars'])
        need(json.loads(prepared_repair.user) == {
            'original_generation_input': previous_wire['messages'][1]['content']}, 'retained_exact_input')
        result['retained_repair_request_sha256'] = support.digest(asdict(prepared_repair))
        need(api.retry_attempts == 3, 'provider_retry_contract')
        reference = read_reference()
        support.integrity(reference)
        ids = targets(reference, api)
        original = support.snapshot(reference)
        reference.close()
        reference = None
        if mode == 'offline':
            r6.clone_reference()
            r6.seed_controls(api)
        else:
            need(sys.stdin.buffer.read(256) == (pin+'\n').encode(), 'missing_execution_authority')
            proof = json.loads((Path('/preflight')/'offline.json').read_text())
            need(proof['status'] == 'offline_passed' and proof['manifest_sha256'] == pin
                 and proof['previous_pins_verified'] == PREVIOUS_PINS
                 and proof['provider_calls'] == proof['http_attempts'] == proof['completion_calls'] == 0
                 and all(proof[k] is True for k in ('source_unchanged', 'reference_unchanged',
                       'clients_closed', 'connections_closed', 'threads_clean')), 'offline_proof')
        clone, controls = api.db.connect(CLONE), api.db.connect(CONTROL)
        need(support.snapshot(clone)['full_sha256'] == original['full_sha256'], 'clone_mismatch')
        need(not os.path.samefile(CLONE, REFERENCE) and not os.path.samefile(CONTROL, REFERENCE)
             and not os.path.samefile(CLONE, CONTROL), 'clone_alias')
        result['control_fixture_sha256'] = r6.verify_controls(controls)
        result['controls_before_sha256'] = support.snapshot(controls)['full_sha256']
        result['clone_before_sha256'] = original['full_sha256']
        result['normal_source_targets_sha256'] = support.digest(ids)
        from hymem.deadline import MonotonicDeadline, use_deadline
        deadline = MonotonicDeadline.after(BOUNDS['timeout_seconds'])
        private = RESULTS/'private'
        private.mkdir(mode=0o700)
        capture = WireCapture(private, deadline)
        key = ('synthetic-preflight-not-a-key' if mode == 'offline'
               else support.read_key(Path('/run/deepseek.env')))
        client = capture_client(api, key, capture)
        need(client.model == MODEL and client.base_url == ENDPOINT and client.transport_integrity_ok
             and client.effective_extra_body == {'thinking': {'type': 'disabled'}}, 'client_identity')
        budget = SharedBudget(client, deadline, capture=capture)
        raw_binding = api.producer_binding(client, declaration_hook='memory_producer_declaration')
        bounded_binding = api.producer_binding(budget, declaration_hook='memory_producer_declaration')
        need(raw_binding['identity_exact'] is True and bounded_binding['identity_exact'] is True
             and all(bounded_binding['declaration'][field] == raw_binding['declaration'][field]
                     for field in ('model', 'endpoint_origin', 'endpoint_sha256', 'effective_request', 'retry_policy')),
             'wrapper_producer_identity')
        result['producer_identity_sha256'] = raw_binding['identity_sha256']
        result['diagnostic_producer_identity_sha256'] = bounded_binding['identity_sha256']
        if mode == 'offline':
            result.update(status='offline_passed', completion_calls=0, provider_calls=0, http_attempts=0)
        else:
            need(proof['producer_identity_sha256'] == result['producer_identity_sha256']
                 and proof['diagnostic_producer_identity_sha256'] == result['diagnostic_producer_identity_sha256']
                 and proof['clone_before_sha256'] == result['clone_before_sha256']
                 and proof['controls_before_sha256'] == result['controls_before_sha256']
                 and proof['control_fixture_sha256'] == result['control_fixture_sha256']
                 and proof['normal_source_targets_sha256'] == result['normal_source_targets_sha256'], 'offline_context_drift')
            need(proof['retained_repair_request_sha256'] == result['retained_repair_request_sha256'],
                 'retained_repair_preflight_drift')
            result['stock_invocations'] = 1
            with use_deadline(deadline):
                reference = read_reference()
                result['normal_retained'] = normal_walk(reference, ids, budget, api, support, private)
                reference.close()
                reference = None
                result['explicit_recovery'] = explicit_recovery(clone, budget, api, support, ids)
                control_ids = sorted(r6.control_cases())
                result['normal_controls'] = normal_walk(controls, control_ids, budget, api, support, invented=True)
                for item in result['normal_controls']['sessions']:
                    item['review_criteria'] = r6.control_cases()[item['session_id']]['review']
                result['repair_contract_controls'] = repair_contract_controls(
                    controls, control_ids, budget, api, support, r6.control_cases())
                result['retained_repair_case'] = retained_repair_case(
                    previous_wire, previous_session, budget, api, support, private)
            result['status'] = ('effectiveness_passed_review_pending'
                if result['normal_retained']['complete_sessions'] == 14
                and result['normal_controls']['complete_sessions'] == 4
                and result['explicit_recovery']['recovered_all']
                and result['repair_contract_controls']['complete_sessions'] == 4
                and result['retained_repair_case']['passed'] else 'honestly_degraded')
    except BaseException as exc:
        result.update(status='error', exception_type=type(exc).__name__)
    finally:
        cleanup = []
        if client is not None:
            try:
                result['usage'] = api.usage_snapshot(client)
                calls = budget.calls if budget is not None else 0
                result.update(completion_calls=calls, provider_calls=result['usage']['calls'],
                              http_attempts=result['usage']['request_attempts'])
                need(result['usage']['calls_available'] is True
                     and result['usage']['request_attempts_available'] is True
                     and result['usage']['successful_responses_available'] is True
                     and all(type(result['usage'][field]) is int for field in
                             ('calls', 'request_attempts', 'successful_responses'))
                     and 0 <= result['provider_calls'] == result['usage']['successful_responses'] <= calls <= 192
                     and 0 <= result['http_attempts'] <= min(576, calls*3), 'accounting')
                need(capture.attempts == result['http_attempts'], 'capture_attempt_accounting')
            except BaseException as exc:
                cleanup.append('accounting:'+type(exc).__name__)
            try:
                client.close()
            except BaseException as exc:
                cleanup.append('client:'+type(exc).__name__)
        result['clients_closed'] = not any(c.startswith('client:') for c in cleanup)
        if capture is not None:
            try:
                result['private_capture'] = capture.public_inventory()
            except BaseException as exc:
                cleanup.append('capture:'+type(exc).__name__)
        if mode == 'offline' and support is not None and clone is not None and controls is not None:
            try:
                need(support.snapshot(clone)['full_sha256'] == result['clone_before_sha256']
                     and support.snapshot(controls)['full_sha256'] == result['controls_before_sha256'],
                     'offline_clone_changed')
                result['offline_clones_unchanged'] = True
            except BaseException as exc:
                cleanup.append('clone:'+type(exc).__name__)
        for conn in (reference, clone, controls):
            if conn is not None:
                try:
                    conn.close()
                except BaseException as exc:
                    cleanup.append('connection:'+type(exc).__name__)
        result['connections_closed'] = not any(c.startswith('connection:') for c in cleanup)
        try:
            check = read_reference()
            try:
                need(original is not None and support.snapshot(check)['full_sha256'] == original['full_sha256'],
                     'reference_changed')
            finally:
                check.close()
            verify(pin)
            previous_request()
            result['reference_unchanged'] = result['source_unchanged'] = True
            need(not any(t.ident not in threads for t in threading.enumerate()), 'threads_remaining')
            result['threads_clean'] = True
        except BaseException as exc:
            cleanup.append('postcheck:'+type(exc).__name__)
        result['cleanup_errors'] = cleanup
        if cleanup:
            result['status'] = 'error'
        if support is not None:
            support.save(RESULTS/(mode+'.json'), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] in ('offline_passed', 'effectiveness_passed_review_pending') else 1


def supervise(pin):
    manifest = verify(pin)
    support = load(Path('/support/worker.py'), 'r8_supervise_support', SUPPORT_SHA)
    supervisor = load(Path('/support/supervised_invocation.py'), 'r8_summary_supervisor', SUPERVISOR_SHA)
    def cancel(_signal, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    outcome = None
    failure = None
    try:
        outcome = supervisor.supervise_invocation(
            [sys.executable, '-I', '-B', '/diag/worker.py', 'live', '--manifest-sha256', pin],
            cwd=SOURCE, env=environment(), output_dir=RESULTS/'invocation', timeout_seconds=2700,
            cleanup_seconds=10, output_limit_bytes=2*1024*1024, stdin_bytes=(pin+'\n').encode())
    except BaseException as exc:
        failure = type(exc).__name__
    unchanged = False
    try:
        verify(pin)
        unchanged = True
    except BaseException as exc:
        failure = failure or type(exc).__name__
    support.save(RESULTS/'supervisor.json', {'manifest_sha256': pin, 'outcome': asdict(outcome),
        'package_and_source_unchanged': unchanged, 'exception_type': failure} if outcome else
        {'manifest_sha256': pin, 'outcome': None, 'package_and_source_unchanged': unchanged, 'exception_type': failure})
    return 0 if failure is None and outcome is not None and outcome.status == 'completed' and outcome.returncode == 0 and outcome.safe_to_continue else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('offline', 'live', 'supervise'))
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    return supervise(args.manifest_sha256) if args.mode == 'supervise' else worker(args.mode, args.manifest_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
