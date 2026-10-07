"""Private, read-only q-0001 held-attempt census for stopped pilot afit9i7d.

Only fixed, finite counters cross SSH. This checks whether a conservative
superset of the production held rows equals the reported 79; it cannot recover
the discarded question exception or make a new benchmark decision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tools.diagnostics import luna_instrumented_failure_metadata_v1 as v1
from tools.diagnostics import luna_instrumented_failure_metadata_v2 as v2


SCHEMA = "luna-instrumented-held-census-v1"
V1_SHA = "a892dbd759ab0ebd5bfacf81eee0b7e5e00f5de530cf1cc340453f235d164826"
V2_SHA = "6a81b7feb31849e08dc4c860b21a59c1867074b9220c2068e93eb188de251ffd"
HELPER_SHA = "b07cdb4d26ad2f95ab236fe8af4a1665d19c376ffaf077e90a561ce500ac1181"
REPORTED_HELD = 79
RETRY_BOUND = 3  # Frozen candidate HyMemConfig default; the runner does not override it.
CATEGORIES = ("nonsemantic_reason", "invalid_details", "unsupported_details")
# Frozen candidate hymem/extraction/chunk.py _FAILURE_REASONS (SHA256 c644513152e2ddefbe0d0d18dce7d597d18b4ba78ce275dd04ad4b2133ee7b92).
REASON_CODES = ("branch_incomplete", "call_failure", "contract_failure",
    "grounding_failure", "incomplete_response", "input_contract_failure", "internal_validation_failure",
    "item_validation_failure", "output_limit_exceeded", "parse_failure",
    "resource_limit", "response_conflict", "shape_failure",
    "source_coverage_failure", "unspecified_failure", "other")

PROJECTION = r'''
import json
import sqlite3
import time
from contextlib import closing
from urllib.parse import quote

_ALLOWED_TABLES = {'phase1_generations', 'chunk_extraction_attempts', 'chunks',
                   'chunk_extraction_terminal_losses'}

def _authorize(action, arg1, arg2, dbname, source):
    if source is not None:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and dbname in {None, 'main'} and arg1 in _ALLOWED_TABLES:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and arg2 in {'count', 'sum', 'typeof', 'length'}:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_TRANSACTION and arg1 in {'BEGIN', 'COMMIT', 'ROLLBACK'}:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY

def _detail_category(reason, details, classifier):
    if type(reason) is not str or reason not in classifier['_SEMANTIC_REASONS'] | classifier['_BRANCH_REASONS']:
        return 'nonsemantic_reason'
    if type(details) is not str:
        return 'invalid_details'
    try:
        parsed = json.loads(details)
    except (TypeError, ValueError):
        return 'invalid_details'
    if (type(parsed) is not list or len(parsed) > 32
            or any(type(item) is not str or len(item) > 160
                   or classifier['_DETAIL'].fullmatch(item) is None
                   or item.endswith('diagnostics:truncated')
                   or item.endswith('diagnostic:invalid') for item in parsed)):
        return 'invalid_details'
    return 'unsupported_details'

def _census(root, namespace, report, base, classifier, final_projector):
    q = base['questions']['q-0001']
    indexing, final = q['indexing'], q['indexing']['final_status']
    if (q['checkpoint_failure_code'] != 'unspecified_failure'
            or indexing.get('complete') is not True
            or indexing.get('outcome') != 'failure'
            or indexing.get('healthy') is not False
            or indexing.get('failure_code') != 'quarantined_extraction'
            or indexing.get('cleanup_error_count') != 0
            or final is None
            or final['quarantined']['quarantined_chunks'] != REPORTED_HELD
            or any(final['pending'].values()) or any(final['malformed'].values())
            or any(n for k, n in final['quarantined'].items() if k != 'quarantined_chunks')
            or final['terminal_loss']['chunks'] != 0
            or final['coverage_integrity']['failures'] != 0
            or final['summary_health']['summary_healthy'] is not True):
        raise ValueError('q1_gate_invalid')
    path = root/'run'/'q-0001'/'private-indexing.json'
    summary = namespace['_read'](path, root, 262144)
    if type(summary) is not dict or type(summary.get('final_status')) is not dict:
        raise ValueError('summary_invalid')
    if final_projector(summary['final_status']) != final:
        raise ValueError('summary_changed')
    if summary['final_status'].get('in_progress') is not False:
        raise ValueError('status_in_progress')
    generation = summary['final_status'].get('phase1_generation_key')
    prefix = 'hymem-phase1-generation-v1:'
    if (type(generation) is not str or len(generation) != len(prefix) + 64
            or not generation.startswith(prefix)
            or any(ch not in '0123456789abcdef' for ch in generation[len(prefix):])):
        raise ValueError('generation_invalid')
    db = root/'run'/'q-0001'/'hymem.sqlite'
    if (db.is_symlink() or not namespace['_file'](db, root, 4_294_967_296)
            or db.stat().st_size < 4096):
        raise ValueError('store_invalid')
    wal = db.with_name(db.name + '-wal')
    if wal.is_symlink() or (wal.exists() and (
            not namespace['_file'](wal, root, 4_294_967_296)
            or wal.stat().st_size != 0)):
        raise ValueError('wal_unverified')
    deadline = time.monotonic() + 8.0
    uri = 'file:' + quote(str(db), safe='/') + '?mode=ro&immutable=1'
    with closing(sqlite3.connect(uri, uri=True, timeout=0.1)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        conn.execute('PRAGMA trusted_schema=OFF')
        conn.execute('PRAGMA temp_store=MEMORY')
        conn.set_authorizer(_authorize)
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        conn.execute('BEGIN')
        cache_rows = conn.execute(
            'SELECT extraction_cache_key FROM phase1_generations WHERE generation_key=? LIMIT 2',
            (generation,)).fetchall()
        if len(cache_rows) != 1 or type(cache_rows[0][0]) is not str or not cache_rows[0][0]:
            raise ValueError('cache_key_unverified')
        cache_key = cache_rows[0][0]
        # Same production held conditions, omitting only current_phase1_publications.
        # A view may call HyMem UDFs; this script registers none and denies views.
        where = """FROM chunk_extraction_attempts a JOIN chunks c ON c.id=a.chunk_id
            WHERE c.chunk_kind='extraction'
              AND (c.salience_reason IS NULL OR c.salience_reason<>'short_session_fallback')
              AND c.source_manifest_version='claim-source-manifest-v1'
              AND c.source_manifest_count>0 AND a.attempts>=?
              AND a.prompt_version=? AND a.phase1_generation_key=?
              AND NOT EXISTS (SELECT 1 FROM chunk_extraction_terminal_losses loss
                              WHERE loss.chunk_id=c.id)"""
        params = (RETRY_BOUND, cache_key, generation)
        checked = conn.execute("""SELECT COUNT(*) AS n,
            SUM(CASE WHEN typeof(a.last_failure_reason)='text'
                      AND length(a.last_failure_reason)<=128
                      AND typeof(a.last_failure_details)='text'
                      AND length(a.last_failure_details)<=8192
                THEN 1 ELSE 0 END) AS bounded """ + where, params).fetchone()
        if (type(checked['n']) is not int or not 0 <= checked['n'] <= 1000
                or (checked['bounded'] if checked['bounded'] is not None else 0)
                   != checked['n']):
            raise ValueError('census_input_bound')
        rows = conn.execute("""SELECT a.last_failure_reason, a.last_failure_details,
                                      COUNT(*) AS n """ + where + """
            GROUP BY a.last_failure_reason, a.last_failure_details
            LIMIT 1001""", params).fetchall()
        if len(rows) > 1000 or time.monotonic() >= deadline:
            raise ValueError('census_bound')
        accepted = 0
        rejected = 0
        categories = {key: 0 for key in CATEGORIES}
        reasons = {key: 0 for key in REASON_CODES}
        for row in rows:
            if time.monotonic() >= deadline:
                raise ValueError('census_deadline')
            n, reason, details = row['n'], row['last_failure_reason'], row['last_failure_details']
            if type(n) is not int or not 0 < n <= 1000:
                raise ValueError('row_count_invalid')
            reasons[reason if type(reason) is str and reason in reasons else 'other'] += n
            if classifier['_semantic_reason'](reason, details):
                accepted += n
            else:
                rejected += n
                categories[_detail_category(reason, details, classifier)] += n
        total = accepted + rejected
        if total > 1000:
            raise ValueError('census_bound')
        conn.execute('ROLLBACK')
    equal = total == REPORTED_HELD
    return {'schema': SCHEMA, 'terminal_and_cleanup_verified': True,
            'source_receipt_verified': True, 'question': 'q-0001',
            'reported_held': REPORTED_HELD, 'superset_count': total,
            'superset_equals_reported': equal, 'semantic_accepted': accepted,
            'semantic_rejected': rejected, 'rejection_categories': categories,
            'reason_counts': reasons,
            'diagnostic_decision_if_exact': (rejected == 0 if equal else None)}
'''

REMOTE = r'''
namespace = {'__name__': 'reviewed_reader', '__file__': '<reviewed-reader>'}
exec(compile(SOURCE, '<reviewed-reader>', 'exec'), namespace)
root = namespace['Path'](ROOT)
report = namespace['inspect'](root, RECEIPT_SHA)
scope = {'INDEXING_CODES': INDEXING_CODES, 'EXCEPTION_TYPES': EXCEPTION_TYPES,
         'RECEIPT_SHA': RECEIPT_SHA, 'SCHEMA': V1_SCHEMA}
exec(compile(V1_PROJECTION, '<frozen-v1-projection>', 'exec'), scope)
base = scope['_project'](root, namespace, report)
extension = {'PENDING': PENDING, 'MALFORMED': MALFORMED,
             'QUARANTINED': QUARANTINED, 'SUMMARY_COUNTS': SUMMARY_COUNTS,
             'MAX_COUNT': MAX_COUNT, 'V1_SCHEMA': V1_SCHEMA, 'SCHEMA': V2_SCHEMA,
             'V1_FIELDS': scope['_fields']}
exec(compile(V2_EXTENSION, '<frozen-v2-extension>', 'exec'), extension)
base = extension['_extend'](root, namespace, report, base)
helper_path = root/'code'/'benchmarks'/'lme_diagnostic.py'
if not namespace['_file'](helper_path, root, 2000000):
    raise ValueError('helper_file_invalid')
helper_bytes = helper_path.read_bytes()
if namespace['_sha'](helper_path) != HELPER_SHA or helper_bytes != HELPER_SOURCE:
    raise ValueError('helper_pin_invalid')
classifier = {'__name__': 'pinned_pure_classifier'}
exec(compile(HELPER_SOURCE, '<pinned-pure-classifier>', 'exec'), classifier)
projection = {'SCHEMA': SCHEMA, 'REPORTED_HELD': REPORTED_HELD,
              'RETRY_BOUND': RETRY_BOUND, 'CATEGORIES': CATEGORIES,
              'REASON_CODES': REASON_CODES}
exec(compile(PROJECTION, '<bounded-held-census>', 'exec'), projection)
print(json.dumps(projection['_census'](root, namespace, report, base, classifier,
                                     extension['_final']),
                 sort_keys=True, separators=(',', ':'), allow_nan=False))
'''


def _validated(value: object) -> dict | None:
    if type(value) is not dict or set(value) != {
            "schema", "terminal_and_cleanup_verified", "source_receipt_verified",
            "question", "reported_held", "superset_count",
            "superset_equals_reported", "semantic_accepted", "semantic_rejected",
            "rejection_categories", "reason_counts", "diagnostic_decision_if_exact"}:
        return None
    if (value["schema"] != SCHEMA or value["terminal_and_cleanup_verified"] is not True
            or value["source_receipt_verified"] is not True
            or value["question"] != "q-0001" or value["reported_held"] != REPORTED_HELD):
        return None
    for name in ("superset_count", "semantic_accepted", "semantic_rejected"):
        if type(value[name]) is not int or not 0 <= value[name] <= 100000:
            return None
    categories = value["rejection_categories"]
    reasons = value["reason_counts"]
    if (type(categories) is not dict or set(categories) != set(CATEGORIES)
            or any(type(categories[k]) is not int or not 0 <= categories[k] <= 100000
                   for k in CATEGORIES)
            or sum(categories.values()) != value["semantic_rejected"]
            or value["semantic_accepted"] + value["semantic_rejected"] != value["superset_count"]):
        return None
    if (type(reasons) is not dict or set(reasons) != set(REASON_CODES)
            or any(type(reasons[k]) is not int or not 0 <= reasons[k] <= 1000
                   for k in REASON_CODES)
            or sum(reasons.values()) != value["superset_count"]):
        return None
    exact = value["superset_count"] == REPORTED_HELD
    return value if (type(value["superset_equals_reported"]) is bool
        and value["superset_equals_reported"] is exact
        and value["diagnostic_decision_if_exact"] is
            ((value["semantic_rejected"] == 0) if exact else None)) else None


def _payload(source: bytes, helper: bytes) -> str:
    return ("import json\nSOURCE=" + repr(source.decode("utf-8"))
        + "\nHELPER_SOURCE=" + repr(helper)
        + "\nROOT=" + repr(v1.ROOT) + "\nRECEIPT_SHA=" + repr(v1.RECEIPT_SHA)
        + "\nV1_SCHEMA=" + repr(v1.SCHEMA) + "\nV2_SCHEMA=" + repr(v2.SCHEMA)
        + "\nSCHEMA=" + repr(SCHEMA) + "\nHELPER_SHA=" + repr(HELPER_SHA)
        + "\nINDEXING_CODES=" + repr(v1.INDEXING_CODES)
        + "\nEXCEPTION_TYPES=" + repr(v1.EXCEPTION_TYPES)
        + "\nV1_PROJECTION=" + repr(v1.PROJECTION)
        + "\nPENDING=" + repr(v2.PENDING) + "\nMALFORMED=" + repr(v2.MALFORMED)
        + "\nQUARANTINED=" + repr(v2.QUARANTINED)
        + "\nSUMMARY_COUNTS=" + repr(v2.SUMMARY_COUNTS)
        + "\nMAX_COUNT=" + repr(v2.MAX_COUNT)
        + "\nV2_EXTENSION=" + repr(v2.EXTENSION)
        + "\nREPORTED_HELD=" + repr(REPORTED_HELD)
        + "\nRETRY_BOUND=" + repr(RETRY_BOUND)
        + "\nCATEGORIES=" + repr(CATEGORIES)
        + "\nREASON_CODES=" + repr(REASON_CODES)
        + "\nPROJECTION=" + repr(PROJECTION) + "\n" + REMOTE)


def main() -> int:
    try:
        for path, digest in ((Path(v1.__file__), V1_SHA), (Path(v2.__file__), V2_SHA),
                             (v1.READER, v1.READER_SHA)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("source_pin_invalid")
        helper_path = Path(__file__).resolve().parents[2] / "benchmarks/lme_diagnostic.py"
        helper = helper_path.read_bytes()
        if hashlib.sha256(helper).hexdigest() != HELPER_SHA:
            raise ValueError("helper_pin_invalid")
        payload = _payload(v1.READER.read_bytes(), helper)
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite",
            "/usr/bin/python3 -I -B -"], input=payload, text=True,
            capture_output=True, timeout=30)
        if result.returncode != 0 or len(result.stdout) > 2048:
            raise ValueError("remote_unavailable")
        decoded = json.loads(result.stdout, parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("nonfinite_metadata")))
        safe = _validated(decoded)
        if safe is None:
            raise ValueError("projection_invalid")
        print(json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        print(json.dumps({"schema": SCHEMA, "status": "metadata_unavailable"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
