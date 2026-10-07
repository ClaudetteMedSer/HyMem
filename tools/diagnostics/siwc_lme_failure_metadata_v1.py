"""Bounded, read-only projection of one stopped SIWC diagnostic's indexing metadata.

The host side executes the pinned progress reader, then opens only four fixed
private-indexing.json paths after terminal, cleanup, receipt and checkpoint gates.
This is a metadata projection, not full canonical protocol validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "siwc-lme-failure-metadata-v1"
READER = Path(__file__).with_name("siwc_lme_diagnostic_progress_v3.py")
READER_SHA = "89e39169a81d9ee810e54fb4c143753906aed282f50527b923cca6cc504a5279"
ROOT = "/home/atta/.hymem-siwc-lme-diagnostic-preflight-d2hxx626"
RECEIPT_SHA = "09d74cf8e9e4999542d130411e0d40a05670c2690701c58ee43a5157123a94e3"
INDEXING_SCHEMA = "hymem-lme-indexing-summary-v6"

# Frozen candidate benchmarks/lme_protocol.py's indexing failure vocabulary.
INDEXING_CODES = frozenset({
    "phase1_producer_unavailable", "timeout_before_cycle", "timeout_during_cycle",
    "cycle_exception", "malformed_status_shape", "malformed_pending_backlog",
    "malformed_quarantine_state", "malformed_terminal_loss_state",
    "malformed_coverage_integrity_state", "malformed_aggregation_failure_report",
    "malformed_cycle_failure_report", "coverage_integrity_failure",
    "malformed_durable_state", "terminal_extraction_source_loss",
    "quarantined_extraction", "timeout_after_cycle", "max_cycles_exhausted",
})
EXCEPTION_TYPES = frozenset({
    "ValueError", "RuntimeError", "TypeError", "KeyError",
    "IndexingConvergenceError", "BenchmarkIntegrityError", "ConcurrentStop",
    "OperationalError", "IntegrityError", "TimeoutError", "OSError",
    "AssertionError", "MemoryError", "CancelledError",
})
PENDING = (
    "pending_source_materialization", "pending_chunks", "pending_digests",
    "pending_profiles", "pending_facts", "pending_aggregation",
    "pending_chunk_embeddings", "pending_message_embeddings", "pending_edge_embeddings",
    "pending_episode_embeddings", "pending_fact_embeddings",
)
MALFORMED = (
    "malformed_source_materialization", "malformed_digests", "malformed_profiles",
    "malformed_facts", "malformed_summaries",
)
QUARANTINED = (
    "quarantined_chunks", "quarantined_digests", "quarantined_profiles",
    "quarantined_facts", "quarantined_facts_malformed",
)
SUMMARY = ("summary_degraded_sessions", "summary_missing_sessions", "malformed_summaries")
CHECKPOINT_CODES = frozenset({
    "indexing_rejected", "question_failure", "worker_runtime_failure",
    "not_started_after_campaign_stop", "unspecified_failure",
})
MAX_COUNT = 2_147_483_647
MAX_SECONDS = 1_000_000


PROJECTION = r'''
import math

def _count(value):
    if type(value) is not int or not 0 <= value <= MAX_COUNT:
        raise ValueError('metadata_count_invalid')
    return value

def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= MAX_SECONDS:
        raise ValueError('metadata_seconds_invalid')
    return value

def _counts(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('metadata_counts_shape_invalid')
    return {key: _count(value[key]) for key in keys}

def _finite_code(value, allowed):
    return value if type(value) is str and value in allowed else 'other'

def _final(value):
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError('final_status_invalid')
    pending = _counts(value.get('pending'), PENDING)
    malformed = _counts(value.get('malformed'), MALFORMED)
    quarantined = _counts(value.get('quarantined'), QUARANTINED)
    terminal = value.get('terminal_loss')
    coverage = value.get('coverage_integrity')
    summary = value.get('summary_health')
    if type(terminal) is not dict or type(coverage) is not dict:
        raise ValueError('final_status_invalid')
    if type(summary) is not dict or set(summary) != set(SUMMARY) | {'summary_healthy'}:
        raise ValueError('summary_health_invalid')
    health = {key: _count(summary[key]) for key in SUMMARY}
    flag = summary['summary_healthy']
    if (type(flag) is not bool or flag is not (
            health['summary_degraded_sessions'] == 0 and health['malformed_summaries'] == 0)
            or health['summary_missing_sessions'] > health['summary_degraded_sessions']
            or health['malformed_summaries'] != malformed['malformed_summaries']):
        raise ValueError('summary_health_invalid')
    health['summary_healthy'] = flag
    return {'pending': pending, 'malformed': malformed, 'quarantined': quarantined,
        'terminal_loss': {'chunks': _count(terminal.get('chunks'))},
        'coverage_integrity': {'failures': _count(coverage.get('failures'))},
        'summary_health': health}

def _indexing(value):
    if type(value) is not dict or value.get('schema') != INDEXING_SCHEMA:
        raise ValueError('indexing_schema_invalid')
    result = {'present': True, 'outcome': _finite_code(value.get('outcome'),
        {'success', 'success_with_summary_degradation', 'failure'}),
        'cycles': _count(value.get('cycles')),
        'max_cycles': _count(value.get('max_cycles')),
        'elapsed_s': _seconds(value.get('elapsed_s')),
        'timeout_s': _seconds(value.get('timeout_s')),
        'final_status': _final(value.get('final_status'))}
    if result['max_cycles'] != 100 or result['cycles'] > result['max_cycles'] or result['timeout_s'] != 10800:
        raise ValueError('indexing_bounds_invalid')
    for key in ('complete', 'healthy'):
        if type(value.get(key)) is not bool:
            raise ValueError('indexing_boolean_invalid')
        result[key] = value[key]
    failure = value.get('failure')
    if failure is None:
        result['failure_code'] = None
        result['exception_type'] = None
    elif type(failure) is dict:
        result['failure_code'] = _finite_code(failure.get('code'), INDEXING_CODES)
        kind = failure.get('exception_type')
        result['exception_type'] = None if kind is None else _finite_code(kind, EXCEPTION_TYPES)
    else:
        raise ValueError('indexing_failure_invalid')
    errors = value.get('cleanup_errors')
    if type(errors) is not list or len(errors) > MAX_COUNT:
        raise ValueError('cleanup_errors_invalid')
    result['cleanup_error_count'] = len(errors)
    return result

def _project(root, reader):
    report = reader['inspect'](root, RECEIPT_SHA)
    if not (type(report) is dict
            and report.get('status') == 'terminal_incomplete_or_unclean'
            and report.get('runtime_cleanup_verified') is True
            and report.get('completed_diagnostic_and_clean') is False
            and type(report.get('selected_denominator')) is int and report['selected_denominator'] == 4
            and type(report.get('scored_count')) is int and report['scored_count'] == 0
            and type(report.get('failed_count')) is int and report['failed_count'] == 4
            and type(report.get('known_turns')) is int and report['known_turns'] == 3554
            and type(report.get('known_tokens')) is int and report['known_tokens'] == 11837159
            and report.get('usage_complete') is True
            and report.get('campaign_stop') == 'question_failure'
            and report.get('budget_stop_code') == 'question_failure'
            and report.get('first_failure') is None
            and report.get('owner_failure') is None
            and report.get('resource_fault') is None
            and type(report.get('resource_observation')) is dict
            and type(report['resource_observation'].get('denials')) is int
            and report['resource_observation']['denials'] == 0):
        raise ValueError('terminal_gate_invalid')
    receipt = reader['receipt'](root, RECEIPT_SHA)
    run = root/'run'
    if not run.is_dir() or run.is_symlink() or not run.resolve().is_relative_to(root):
        raise ValueError('run_directory_invalid')
    path = run/'diagnostic-checkpoint.json'
    checkpoint = reader['read'](path, root, 2_000_000)
    counts, completed, _, _, check = reader['checkpoint'](checkpoint, receipt)
    if (counts is None or completed != 0 or counts['failed'] != 4
            or counts['expected'] != 4 or counts['missing'] != 0):
        raise ValueError('checkpoint_gate_invalid')
    ids = check['expected_ids']
    entries = check['entries']
    out = {'schema': SCHEMA, 'terminal_and_cleanup_verified': True,
           'source_receipt_verified': True, 'questions': {}}
    for index in range(4):
        label = 'q-%04d' % index
        entry = entries.get(ids[index])
        if type(entry) is not dict or entry.get('status') != 'failed':
            raise ValueError('checkpoint_entry_invalid')
        directory = root/'run'/label
        if not directory.is_dir() or directory.is_symlink() or not directory.resolve().is_relative_to(root):
            raise ValueError('private_directory_invalid')
        private = directory/'private-indexing.json'
        if private.is_symlink():
            raise ValueError('private_file_invalid')
        item = {'checkpoint_failure_code': _finite_code(entry.get('failure'), CHECKPOINT_CODES)}
        item['indexing'] = (_indexing(reader['read'](private, root, 262144))
            if private.exists() else {'present': False})
        out['questions'][label] = item
    return out
'''

REMOTE = r'''
reader = {'__name__': 'reviewed_reader', '__file__': '<reviewed-reader>'}
exec(compile(SOURCE, '<reviewed-reader>', 'exec'), reader)
scope = {'RECEIPT_SHA': RECEIPT_SHA, 'SCHEMA': SCHEMA,
    'INDEXING_SCHEMA': INDEXING_SCHEMA, 'INDEXING_CODES': INDEXING_CODES,
    'EXCEPTION_TYPES': EXCEPTION_TYPES, 'PENDING': PENDING,
    'MALFORMED': MALFORMED, 'QUARANTINED': QUARANTINED,
    'SUMMARY': SUMMARY, 'CHECKPOINT_CODES': CHECKPOINT_CODES,
    'MAX_COUNT': MAX_COUNT, 'MAX_SECONDS': MAX_SECONDS}
exec(compile(PROJECTION, '<bounded-projection>', 'exec'), scope)
root = reader['Path'](ROOT)
print(json.dumps(scope['_project'](root, reader), sort_keys=True,
    separators=(',', ':'), allow_nan=False))
'''


def _validated(value: object) -> dict | None:
    if (type(value) is not dict or set(value) != {
            "schema", "terminal_and_cleanup_verified", "source_receipt_verified", "questions"}
            or value["schema"] != SCHEMA
            or value["terminal_and_cleanup_verified"] is not True
            or value["source_receipt_verified"] is not True
            or type(value["questions"]) is not dict
            or set(value["questions"]) != {f"q-{i:04d}" for i in range(4)}):
        return None
    for question in value["questions"].values():
        if (type(question) is not dict or set(question) != {"checkpoint_failure_code", "indexing"}
                or type(question["checkpoint_failure_code"]) is not str
                or question["checkpoint_failure_code"] not in CHECKPOINT_CODES | {"other"}):
            return None
        item = question["indexing"]
        if type(item) is not dict or type(item.get("present")) is not bool:
            return None
        if item["present"] is False:
            if set(item) != {"present"}:
                return None
            continue
        if set(item) != {"present", "outcome", "cycles", "max_cycles", "elapsed_s",
                "timeout_s", "final_status", "complete", "healthy", "failure_code",
                "exception_type", "cleanup_error_count"}:
            return None
        if (type(item["outcome"]) is not str or item["outcome"] not in {
                "success", "success_with_summary_degradation", "failure", "other"}
                or type(item["complete"]) is not bool or type(item["healthy"]) is not bool
                or type(item["cycles"]) is not int or not 0 <= item["cycles"] <= MAX_COUNT
                or type(item["max_cycles"]) is not int or item["max_cycles"] != 100
                or item["cycles"] > item["max_cycles"]
                or type(item["cleanup_error_count"]) is not int
                or not 0 <= item["cleanup_error_count"] <= MAX_COUNT):
            return None
        for key in ("elapsed_s", "timeout_s"):
            number = item[key]
            if type(number) not in (int, float) or not 0 <= number <= MAX_SECONDS:
                return None
        if item["timeout_s"] != 10800:
            return None
        if item["failure_code"] is not None and (type(item["failure_code"]) is not str
                or item["failure_code"] not in INDEXING_CODES | {"other"}):
            return None
        if item["failure_code"] is None and item["exception_type"] is not None:
            return None
        if item["exception_type"] is not None and (type(item["exception_type"]) is not str
                or item["exception_type"] not in EXCEPTION_TYPES | {"other"}):
            return None
        final = item["final_status"]
        if final is None:
            continue
        if type(final) is not dict or set(final) != {"pending", "malformed", "quarantined",
                "terminal_loss", "coverage_integrity", "summary_health"}:
            return None
        for label, keys in (("pending", PENDING), ("malformed", MALFORMED),
                            ("quarantined", QUARANTINED), ("terminal_loss", ("chunks",)),
                            ("coverage_integrity", ("failures",))):
            counts = final[label]
            if (type(counts) is not dict or set(counts) != set(keys)
                    or any(type(counts[key]) is not int or not 0 <= counts[key] <= MAX_COUNT
                           for key in keys)):
                return None
        health = final["summary_health"]
        if (type(health) is not dict or set(health) != set(SUMMARY) | {"summary_healthy"}
                or any(type(health[key]) is not int or not 0 <= health[key] <= MAX_COUNT
                       for key in SUMMARY)
                or type(health["summary_healthy"]) is not bool
                or health["summary_missing_sessions"] > health["summary_degraded_sessions"]
                or health["malformed_summaries"] != final["malformed"]["malformed_summaries"]
                or health["summary_healthy"] is not (
                    health["summary_degraded_sessions"] == 0 and health["malformed_summaries"] == 0)):
            return None
    return value


def main() -> int:
    try:
        source = READER.read_bytes()
        if hashlib.sha256(source).hexdigest() != READER_SHA:
            raise ValueError("reader_pin_mismatch")
        payload = ("import json\nSOURCE=" + repr(source.decode("utf-8")) + "\n"
            + "ROOT=" + repr(ROOT) + "\nRECEIPT_SHA=" + repr(RECEIPT_SHA)
            + "\nSCHEMA=" + repr(SCHEMA) + "\nINDEXING_SCHEMA=" + repr(INDEXING_SCHEMA)
            + "\nINDEXING_CODES=" + repr(INDEXING_CODES)
            + "\nEXCEPTION_TYPES=" + repr(EXCEPTION_TYPES)
            + "\nPENDING=" + repr(PENDING) + "\nMALFORMED=" + repr(MALFORMED)
            + "\nQUARANTINED=" + repr(QUARANTINED) + "\nSUMMARY=" + repr(SUMMARY)
            + "\nCHECKPOINT_CODES=" + repr(CHECKPOINT_CODES)
            + "\nMAX_COUNT=" + repr(MAX_COUNT) + "\nMAX_SECONDS=" + repr(MAX_SECONDS)
            + "\nPROJECTION=" + repr(PROJECTION) + "\n" + REMOTE)
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite",
            "/usr/bin/python3 -I -B -"], input=payload, text=True,
            capture_output=True, timeout=30)
        if result.returncode != 0 or len(result.stdout.encode("utf-8")) > 32768:
            raise ValueError("remote_unavailable")
        decoded = json.loads(result.stdout, parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("nonfinite_metadata")), object_pairs_hook=_unique_fields)
        safe = _validated(decoded)
        if safe is None:
            raise ValueError("projection_invalid")
        print(json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except (OSError, UnicodeError, subprocess.SubprocessError, ValueError, TypeError):
        print(json.dumps({"schema": SCHEMA, "status": "metadata_unavailable"}, sort_keys=True))
        return 1


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_projection_field")
        value[key] = item
    return value


if __name__ == "__main__":
    raise SystemExit(main())
