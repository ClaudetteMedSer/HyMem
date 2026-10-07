"""Read-only, source-bound projection of the stopped instrumented pilot's failures.

Private JSON is opened only on Afrodite after the pinned reader verifies the
receipt, terminal and independent cleanup. No private JSON or stderr is exported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "luna-instrumented-failure-metadata-v1"
READER = Path(__file__).with_name("luna_lme_diagnostic_progress_v9.py")
READER_SHA = "ad83c8c5fc5239646a3502fd91d68256645d8bc7428157e6765b8afcee974eb5"
ROOT = "/home/atta/.hymem-lme-diagnostic-preflight-afit9i7d"
RECEIPT_SHA = "769811ec0c398474b544295a0ff406229e920ff178293591a0312eeee9548921"

# Frozen candidate benchmarks/lme_protocol.py's _INDEXING_FAILURE_CODES.
INDEXING_CODES = frozenset({
    "phase1_producer_unavailable", "timeout_before_cycle", "timeout_during_cycle",
    "cycle_exception", "malformed_status_shape", "malformed_pending_backlog",
    "malformed_quarantine_state", "malformed_terminal_loss_state",
    "malformed_coverage_integrity_state", "malformed_aggregation_failure_report",
    "malformed_cycle_failure_report", "coverage_integrity_failure",
    "malformed_durable_state", "terminal_extraction_source_loss",
    "quarantined_extraction", "timeout_after_cycle", "max_cycles_exhausted",
})
EXCEPTION_TYPES = frozenset({"ValueError", "RuntimeError", "TypeError", "KeyError",
    "IndexingConvergenceError", "BenchmarkIntegrityError", "ConcurrentStop",
    "OperationalError", "IntegrityError", "TimeoutError", "OSError",
    "AssertionError", "MemoryError", "CancelledError"})

# This is stdlib-only code shipped with the pinned reader. It has no dynamic
# filename, key, or output field selection from private data.
PROJECTION = r'''
import math

def _code(value, allowed):
    return value if type(value) is str and value in allowed else 'other'

def _number(value, limit=1000000000):
    if type(value) is int and 0 <= value <= limit:
        return value
    if type(value) is float and math.isfinite(value) and 0 <= value <= limit:
        return value
    return None

def _fields(value):
    if type(value) is not dict:
        return {'object': False}
    out = {'object': True}
    for key in ('status', 'outcome'):
        if key in value:
            out[key] = _code(value[key], {'running','complete','success',
                'success_with_summary_degradation','failure','rejected'})
    for key in ('complete','healthy','summary_healthy','admitted'):
        if type(value.get(key)) is bool:
            out[key] = value[key]
    for key in ('cycles','max_cycles','elapsed_s','timeout_s',
                'quarantined_chunks','summary_degraded_sessions',
                'summary_missing_sessions'):
        if key in value:
            out[key] = _number(value[key])
    if 'kind' in value:
        out['kind'] = _code(value['kind'], {'rejected','strict_healthy',
            'summary_degradation','semantic_quarantine'})
    if 'failure' in value:
        failure = value['failure']
        out['failure_code'] = (_code(failure.get('code'), INDEXING_CODES)
            if type(failure) is dict else 'other')
        if type(failure) is dict and failure.get('exception_type') is not None:
            out['exception_type'] = _code(failure['exception_type'], EXCEPTION_TYPES)
    if 'cleanup_errors' in value:
        out['cleanup_error_count'] = (len(value['cleanup_errors'])
            if type(value['cleanup_errors']) is list
            and len(value['cleanup_errors']) <= 1000000 else None)
    return out

def _project(root, namespace, report):
    if not (type(report) is dict
            and report.get('status') == 'terminal_incomplete_or_unclean'
            and report.get('runtime_cleanup_verified') is True
            and report.get('campaign_stop') == 'question_failure'
            and report.get('budget_stop_code') == 'question_failure'
            and report.get('selected_denominator') == 4
            and report.get('scored_count') == 0
            and report.get('known_turns') == 4586
            and report.get('known_tokens') == 33246386
            and report.get('usage_complete') is True
            and report.get('first_failure') is None
            and type(report.get('resource_observation')) is dict
            and report['resource_observation'].get('denials') == 0):
        raise ValueError('terminal_gate_invalid')
    checkpoint = namespace['_read'](root/'run'/'diagnostic-checkpoint.json',root,2000000)
    receipt = namespace['_receipt'](root, RECEIPT_SHA)
    namespace['_checkpoint'](checkpoint, receipt)
    counts, completed, _, _ = namespace['_counts'](checkpoint)
    if counts is None or completed != 0 or counts['failed'] != 4:
        raise ValueError('checkpoint_gate_invalid')
    out = {'schema': SCHEMA, 'terminal_and_cleanup_verified': True,
           'source_receipt_verified': True, 'questions': {}}
    entries = checkpoint['entries']
    ids = checkpoint['expected_ids']
    for index in range(4):
        label = 'q-%04d' % index
        entry = entries.get(ids[index])
        if type(entry) is not dict or entry.get('status') != 'failed':
            raise ValueError('checkpoint_gate_invalid')
        row = entry.get('row')
        failure = row.get('benchmark_failure') if type(row) is dict else None
        # The writer's finite per-question stop vocabulary.
        checkpoint_code = _code(failure, {'indexing_rejected','question_failure',
            'worker_runtime_failure','not_started_after_campaign_stop',
            'unspecified_failure'})
        question = {'checkpoint_failure_code': checkpoint_code}
        directory = root/'run'/label
        row_path = directory/'private-row.json'
        if row_path.is_symlink():
            raise ValueError('private_file_invalid')
        question['private_row_present'] = (namespace['_file'](row_path, root, 262144)
            if row_path.exists() else False)
        if row_path.exists() and not question['private_row_present']:
            raise ValueError('private_file_invalid')
        for public, filename in (('indexing','private-indexing.json'),
                                 ('diagnostic_indexing','private-diagnostic-indexing.json')):
            path = directory/filename
            if path.is_symlink():
                raise ValueError('private_file_invalid')
            if not path.exists():
                question[public] = {'present': False}
                continue
            value = namespace['_read'](path, root, 262144)
            question[public] = {'present': True, **_fields(value)}
        out['questions'][label] = question
    return out
'''

REMOTE = r'''
namespace = {'__name__': 'reviewed_reader', '__file__': '<reviewed-reader>'}
exec(compile(SOURCE, '<reviewed-reader>', 'exec'), namespace)
root = namespace['Path'](ROOT)
# inspect verifies the source receipt and terminal independently before any
# per-question private path can be opened.
report = namespace['inspect'](root, RECEIPT_SHA)
scope = {'INDEXING_CODES': INDEXING_CODES, 'EXCEPTION_TYPES': EXCEPTION_TYPES,
         'RECEIPT_SHA': RECEIPT_SHA,
         'SCHEMA': SCHEMA}
exec(compile(PROJECTION, '<bounded-projection>', 'exec'), scope)
print(json.dumps(scope['_project'](root, namespace, report), sort_keys=True,
                 separators=(',', ':'), allow_nan=False))
'''


def _validated(value: object) -> dict | None:
    if type(value) is not dict or set(value) != {"schema", "terminal_and_cleanup_verified",
            "source_receipt_verified", "questions"}:
        return None
    if (value["schema"] != SCHEMA or value["terminal_and_cleanup_verified"] is not True
            or value["source_receipt_verified"] is not True
            or type(value["questions"]) is not dict
            or set(value["questions"]) != {f"q-{i:04d}" for i in range(4)}):
        return None
    allowed_fields = {"object", "status", "outcome", "complete", "healthy",
        "summary_healthy", "admitted", "cycles", "max_cycles", "elapsed_s",
        "timeout_s", "quarantined_chunks", "summary_degraded_sessions",
        "summary_missing_sessions", "kind", "failure_code", "exception_type",
        "cleanup_error_count"}
    for question in value["questions"].values():
        if (type(question) is not dict or set(question) != {
                "checkpoint_failure_code", "private_row_present", "indexing",
                "diagnostic_indexing"}
                or question["checkpoint_failure_code"] not in {
                    "indexing_rejected", "question_failure", "worker_runtime_failure",
                    "not_started_after_campaign_stop", "unspecified_failure", "other"}
                or type(question["private_row_present"]) is not bool):
            return None
        for key in ("indexing", "diagnostic_indexing"):
            item = question[key]
            if type(item) is not dict or type(item.get("present")) is not bool:
                return None
            if not item["present"]:
                if set(item) != {"present"}:
                    return None
                continue
            if not set(item).issubset(allowed_fields | {"present"}):
                return None
            if type(item.get("object")) is not bool:
                return None
            if item["object"] is False and set(item) != {"present", "object"}:
                return None
            if "status" in item and (type(item["status"]) is not str
                    or item["status"] not in {"running", "complete", "success",
                        "success_with_summary_degradation", "failure", "rejected", "other"}):
                return None
            if "outcome" in item and (type(item["outcome"]) is not str
                    or item["outcome"] not in {"running", "complete", "success",
                        "success_with_summary_degradation", "failure", "rejected", "other"}):
                return None
            if "kind" in item and (type(item["kind"]) is not str
                    or item["kind"] not in {"rejected", "strict_healthy",
                        "summary_degradation", "semantic_quarantine", "other"}):
                return None
            if "failure_code" in item and (type(item["failure_code"]) is not str
                    or item["failure_code"] not in INDEXING_CODES | {"other"}):
                return None
            if "exception_type" in item and (type(item["exception_type"]) is not str
                    or item["exception_type"] not in EXCEPTION_TYPES | {"other"}):
                return None
            for field in ("complete", "healthy", "summary_healthy", "admitted"):
                if field in item and type(item[field]) is not bool:
                    return None
            for field, number in item.items():
                if field in {"cycles", "max_cycles", "elapsed_s", "timeout_s",
                        "quarantined_chunks", "summary_degraded_sessions",
                        "summary_missing_sessions", "cleanup_error_count"}:
                    if number is not None and (type(number) not in (int, float)
                            or not 0 <= number <= 1_000_000_000):
                        return None
    return value


def main() -> int:
    try:
        source = READER.read_bytes()
        if hashlib.sha256(source).hexdigest() != READER_SHA:
            raise ValueError("reader_pin_mismatch")
        payload = ("import json\nSOURCE=" + repr(source.decode("utf-8")) + "\n"
            + "ROOT=" + repr(ROOT) + "\nRECEIPT_SHA=" + repr(RECEIPT_SHA) + "\n"
            + "SCHEMA=" + repr(SCHEMA) + "\nINDEXING_CODES=" + repr(INDEXING_CODES)
            + "\nEXCEPTION_TYPES=" + repr(EXCEPTION_TYPES)
            + "\nPROJECTION=" + repr(PROJECTION) + "\n" + REMOTE)
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite",
            "/usr/bin/python3 -I -B -"], input=payload, text=True,
            capture_output=True, timeout=30)
        if result.returncode != 0 or len(result.stdout) > 32768:
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
