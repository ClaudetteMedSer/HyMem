"""Small R9 diagnostic primitives; never persist normal extraction results."""
from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

FAILURE_STAGES=frozenset({'preflight','verify','support_import','source_import','backup_normal','backup_recovery',
    'connect_normal','connect_recovery','initialize_normal','initialize_recovery','select_target','snapshot_clones',
    'seed_controls','snapshot_controls','client_create','producer_identity','capture_primary','capture_controls',
    'normal','source_exact_repair','explicit_recovery','control_0','control_1','control_2','control_3','audit'})
EXCEPTION_CLASSES=frozenset({'RuntimeError','ValueError','TypeError','KeyError','AttributeError','ImportError',
    'ModuleNotFoundError','PermissionError','FileNotFoundError','FileExistsError','OSError','OperationalError',
    'DatabaseError','IntegrityError','DeadlineExceeded','KeyboardInterrupt','SystemExit'})
FRAME_FILES=frozenset({'worker.py','core.py','db.py','digest.py','summary_recovery.py','summary_state.py',
    'lme_r6_summary_replay.py','r6_summary.py','r8_summary.py','openai_client.py','deadline.py'})

def failure_evidence(exc):
    frames=[]; trace=exc.__traceback__
    while trace is not None:
        filename=Path(trace.tb_frame.f_code.co_filename).name
        frames.append(dict(file=filename if filename in FRAME_FILES else 'other',line=trace.tb_lineno))
        trace=trace.tb_next
    return dict(exception_class=type(exc).__name__ if type(exc).__name__ in EXCEPTION_CLASSES else 'other',
                frames=frames[-16:])

TARGET = 'd1fca62fab454a5a2c6a521bfc4c584219549044a3f7b24ab391ad13063482b2'
BOUNDS = dict(completion_calls=12, http_attempts=36, timeout_seconds=900, invocation_timeout_seconds=120)
PHASE_CAPS = dict(normal=2, source_exact_repair=1, explicit_recovery=3,
                  control_0=1, control_1=1, control_2=1, control_3=1)
NORMAL = dict(max_chars=12000, max_tokens=3072, granular=False,
              max_episodes=12, separate_summary=True, prior_summary_is_stale=False)

def need(value, label):
    if not value:
        raise RuntimeError(label)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()

def select(conn):
    rows = conn.execute('SELECT * FROM sessions').fetchall()
    matches = [r for r in rows if hashlib.sha256(r['id'].encode()).hexdigest() == TARGET]
    need(len(matches) == 1, 'target_selection')
    row = matches[0]
    need(row['auto_summary'] is None and row['summary_failure_reason'] == 'summary_output_cap'
         and row['summary_failure_count'] == 1
         and row['digest_published_message_id'] == row['coverage_message_id'] == 300,
         'target_state')
    messages = conn.execute('SELECT id,content FROM messages WHERE session_id=? ORDER BY id',
                            (row['id'],)).fetchall()
    need([m['id'] for m in messages] == list(range(293,301))
         and sum(len(m['content']) for m in messages) == 6996, 'target_source')
    need(conn.execute('SELECT 1 FROM summary_recovery WHERE session_id=?',
                     (row['id'],)).fetchone() is None, 'target_recovery_job')
    return row['id']

class CapturedRequest(BaseException):
    """Escapes product Exception handlers before any paid completion."""
    def __init__(self, request):
        self.request = request

def capture_normal(conn, sid, native, extract, *, max_chars=12000):
    class Callback:
        def __getattr__(self, name):
            return getattr(native, name)
        def complete(self, request):
            raise CapturedRequest(request)
    try:
        extract(conn, sid, Callback(), **{**NORMAL,'max_chars':max_chars})
    except CapturedRequest as caught:
        return caught.request
    raise RuntimeError('normal_request_not_dispatched')

def source_exact_repair(primary, builder):
    # This helper consumes only trimmed length. X is never a model response.
    repair = builder(primary, 'X' * 503)
    need(json.loads(repair.user) == {'original_generation_input': primary.user},
         'repair_source_exactness')
    need(repair.max_tokens == primary.max_tokens and repair.temperature == primary.temperature
         and repair.response_format == primary.response_format,
         'repair_limits')
    return repair

class Budget:
    def __init__(self, client, deadline, worker_hash, capture):
        self.client, self.deadline, self.worker_hash, self.capture = client, deadline, worker_hash, capture
        self.calls, self.phase, self.phase_calls = 0, None, {}
        self.phase_end = 0
    def __getattr__(self, key):
        return getattr(self.client, key)
    def memory_producer_declaration(self):
        original = self.client.memory_producer_declaration()
        return replace(original, client_id='hymem.diagnostics.r9.Budget',
            implementation='r9-summary-budget-v1:sha256:' +
            hashlib.sha256((self.worker_hash + '|' + original.implementation).encode()).hexdigest())
    def start(self, phase):
        need(phase in PHASE_CAPS and phase not in self.phase_calls, 'phase_single_use')
        self.phase, self.phase_end = phase, time.monotonic() + 120
        self.phase_calls[phase] = 0
        self.capture.phase = phase
    def complete(self, request):
        self.deadline.check()
        need(time.monotonic() < self.phase_end, 'invocation_deadline')
        need(self.phase in PHASE_CAPS and self.phase_calls[self.phase] < PHASE_CAPS[self.phase]
             and self.calls < 12 and self.client.request_attempts + 3 <= 36, 'completion_budget')
        self.calls += 1
        self.phase_calls[self.phase] += 1
        from hymem.deadline import MonotonicDeadline, use_deadline
        remaining = min(self.deadline.remaining(), self.phase_end-time.monotonic())
        with use_deadline(MonotonicDeadline.after(remaining)):
            result = self.client.complete(request)
        self.deadline.check()
        need(time.monotonic() < self.phase_end and self.client.request_attempts <= 36, 'postcall_bounds')
        return result

def parse_projection(raw, parser, *, invented=False):
    selected, failure = parser(raw)
    data = None
    try:
        data = json.loads(raw) if isinstance(raw, str) and len(raw) <= 65536 else None
    except (ValueError, TypeError):
        pass
    alternatives = data.get('alternatives') if isinstance(data, dict) else None
    options = alternatives if isinstance(alternatives, list) else []
    def typename(value):
        return {str:'str',dict:'dict',list:'list',int:'int',float:'float',bool:'bool',type(None):'NoneType'}.get(type(value),'other')
    allowed_failures={'summary_output_cap','summary_validation_failure','parse_failure','shape_failure',
                      'summary_shape_failure','output_truncated'}
    failure = failure if failure is None or failure in allowed_failures else 'unrecognized_failure'
    result = dict(raw_type=typename(raw), raw_chars=len(raw) if isinstance(raw,str) else None,
                  parsed_type=typename(data), option_types=[typename(v) for v in options],
                  option_lengths=[len(v) if isinstance(v,str) else None for v in options],
                  selected_chars=len(selected) if isinstance(selected,str) else None,
                  selected_sha256=hashlib.sha256(selected.encode()).hexdigest() if isinstance(selected,str) else None,
                  failure_reason=failure)
    if invented:
        result.update(alternatives=alternatives, selected=selected)
    return result

def backup(reference, destination, *, metrics=None):
    need(not destination.exists() and not any(Path(str(destination)+s).exists() for s in ('-wal','-shm')),
         'fresh_clone_required')
    fd = os.open(destination, os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW, 0o600)
    os.close(fd)
    source = target = None
    expiry = time.monotonic()+120
    metrics={} if metrics is None else metrics
    metrics.update(callbacks=0,status=None,remaining_pages=None,total_pages=None)
    def progress(status,remaining,total):
        metrics.update(callbacks=metrics['callbacks']+1,status=status,remaining_pages=remaining,total_pages=total)
        need(time.monotonic()<expiry,'backup_deadline')
    try:
        source = sqlite3.connect(reference.as_uri()+'?mode=ro', uri=True,isolation_level=None,timeout=5)
        target = sqlite3.connect(destination)
        source.execute('PRAGMA query_only=ON')
        source.execute('PRAGMA temp_store=MEMORY')
        # Pin one read snapshot before backup: autocommit backup can restart
        # from page zero on each intervening WAL commit.
        source.execute('BEGIN')
        source.execute('SELECT COUNT(*) FROM sqlite_schema').fetchone()
        source.backup(target, pages=256, progress=progress)
    finally:
        try:
            if target is not None: target.close()
        finally:
            if source is not None:
                try: source.rollback()
                finally: source.close()
