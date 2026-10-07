"""Finite grounding subreason census of the stopped, consumed SIWC pilot.

Only fixed category counts leave the host. A row contributes at most once to
each distinct code, even when its bounded details repeat that code. Therefore
code counts can exceed row counts when a branch contains several failure codes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "siwc-lme-retry-reasons-v2"
ROOT = "/home/atta/.hymem-siwc-lme-diagnostic-preflight-d2hxx626"
METADATA = Path(__file__).with_name("siwc_lme_failure_metadata_v1.py")
METADATA_SHA = "f5baf130c27138f3f248c17483f6d89ea9b320b49ff7c8c7219d7f7e5666b7b6"
READER = Path(__file__).with_name("siwc_lme_diagnostic_progress_v3.py")
READER_SHA = "89e39169a81d9ee810e54fb4c143753906aed282f50527b923cca6cc504a5279"
V1 = Path(__file__).with_name("siwc_lme_retry_reasons_v1.py")
V1_SHA = "b27044d6c9c457f39be8cedfe9b712ba9f94420880f17f1767b6f5be52d09ea1"
REASONS = (
    "branch_incomplete", "call_failure", "contract_failure", "grounding_failure",
    "incomplete_response", "input_contract_failure", "internal_validation_failure",
    "item_validation_failure", "output_limit_exceeded", "parse_failure",
    "resource_limit", "response_conflict", "shape_failure",
    "source_coverage_failure", "unspecified_failure", "other",
)
MAX_ROWS = 100_000
MAX_DETAILS_CHARS = 8192
MAX_DETAILS_ITEMS = 32
MAX_DETAIL_CHARS = 160
MAX_OUTPUT_BYTES = 32768

# Literal codes from grounding_staged_gate_v1.py, grounding_staged_v1.py,
# grounding_classification_v4.py, grounding_v2.py, and grounding_gate.py.
# Contract errors pass through the gate as contract:<code with : replaced by _>.
# No stored detail can extend this vocabulary. The two dynamic source/context
# metadata paths in grounding_v2.py are enumerated explicitly.
GATE_CODES = (
    "context_nested_scope", "context_scope", "correction_collision",
    "correction_conflict", "correction_mutation", "source_invalid",
    "source_offset", "sources_total_bounds", "support_integrity", "triple_source",
    "verdict_uncertain", "verdict_unsupported", "contract_verdict_correction",
)
CONTRACT_CODES = (
    "alternatives_count", "alternatives_coverage", "alternatives_index",
    "alternatives_none_required", "alternatives_prior_binding",
    "alternatives_required", "alternatives_shape", "alternatives_unexpected",
    "alternatives_batch_binding", "alternatives_batch_type",
    "assessment_negative_support", "assessment_shape", "assessment_state",
    "batch_binding", "batch_serialization", "batch_type", "batch_unicode",
    "check_indices", "check_shape", "check_state",
    "classification_alternatives", "classification_index", "classification_shape",
    "context_content", "context_id", "context_metadata", "context_parent_metadata",
    "context_parent_missing", "context_parent_prefix", "context_parent_region",
    "context_parent_unexpected", "context_prefix", "context_region", "context_type",
    "evidence_context_missing", "evidence_context_scope", "evidence_duplicate",
    "evidence_global_bounds", "evidence_owned_required", "evidence_parent_required",
    "evidence_parent_scope", "evidence_quote", "evidence_quote_missing",
    "evidence_region", "evidence_shape", "evidence_source",
    "original_index", "original_shape", "request_binding",
    "response_binding", "response_bounds", "response_correction_flag",
    "response_count", "response_depth", "response_incomplete",
    "response_schema", "response_shape", "source_content", "source_contexts",
    "source_duplicate_id", "source_id", "source_legacy_scope",
    "source_metadata", "source_type", "sources_bounds", "sources_total_bounds",
    "support_checks", "support_evidence_bounds", "support_shape",
    "support_unreferenced_evidence", "triple_numeric", "triple_polarity",
    "triple_predicate", "triple_qualifier", "triple_source", "triple_text",
    "triple_type", "triples_bounds", "verdict_correction", "verdict_evidence",
    "verdict_evidence_count", "verdict_index", "verdict_negative_shape",
    "verdict_predicate", "verdict_shape", "verdict_status",
)
CODES = tuple(sorted({"grounding:" + code for code in GATE_CODES} |
                     {"grounding:contract_" + code for code in CONTRACT_CODES}))
FALLBACKS = ("other", "invalid_details", "no_grounding_code", "truncated_details")
CATEGORIES = (*CODES, *FALLBACKS)

# Exact accepted v1 reason x attempt result for the four stopped databases.
# Each triple is (grounding_failure, branch_incomplete) for attempts 1, 2, 3.
# Every other v1 reason must be zero. Reconciliation precedes detail reading.
EXPECTED = {
    "q-0000": ((8, 0), (9, 0), (12, 1)),
    "q-0001": ((4, 1), (5, 0), (12, 2)),
    "q-0002": ((6, 0), (1, 0), (21, 1)),
    "q-0003": ((6, 1), (6, 0), (12, 0)),
}

REMOTE = r'''
import hashlib
import json
import os
import sqlite3
import stat
import time
from pathlib import Path
from urllib.parse import quote

reader = {'__name__': 'pinned_progress', '__file__': '<pinned-progress>'}
exec(compile(READER_SOURCE, '<pinned-progress>', 'exec'), reader)
metadata = {'__name__': 'pinned_metadata', '__file__': '<pinned-metadata>'}
exec(compile(METADATA_SOURCE, '<pinned-metadata>', 'exec'), metadata)
exec(compile(metadata['PROJECTION'], '<pinned-metadata-projection>', 'exec'), metadata)
root = Path(ROOT)
gate = metadata['_project'](root, reader)
if metadata['_validated'](gate) is None:
    raise ValueError('metadata_gate_invalid')

def _stat(path, *, directory=False, absent=False):
    try:
        info = path.lstat()
    except FileNotFoundError:
        if absent:
            return None
        raise
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if (not kind(info.st_mode) or info.st_uid != reader['ROOT_UID']
            or info.st_mode & 0o022 or (not directory and info.st_nlink != 1)):
        raise ValueError('unsafe_path')
    return info

def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)

def _digest(path, expected):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        if (_identity(opened) != _identity(expected)
                or not 0 < opened.st_size <= 512 * 1024 * 1024):
            raise ValueError('database_changed')
        remaining = opened.st_size
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            while remaining:
                block = handle.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError('database_changed')
                digest.update(block)
                remaining -= len(block)
            if handle.read(1):
                raise ValueError('database_changed')
        if _identity(os.fstat(fd)) != _identity(expected):
            raise ValueError('database_changed')
    finally:
        os.close(fd)
    return digest.digest()

def _sidecars(path):
    for suffix in ('-wal', '-journal', '-shm'):
        side = Path(str(path) + suffix)
        info = _stat(side, absent=True)
        if info is not None and info.st_size != 0:
            raise ValueError('sqlite_sidecar_nonempty')

def _reason_query():
    return ('SELECT CASE WHEN typeof(attempts)=\'integer\' AND attempts=1 THEN 1 '
            'WHEN typeof(attempts)=\'integer\' AND attempts=2 THEN 2 '
            'WHEN typeof(attempts)=\'integer\' AND attempts=3 THEN 3 ELSE 0 END, '
            'CASE ' + ' '.join(
                "WHEN typeof(last_failure_reason)='text' AND "
                "last_failure_reason COLLATE BINARY = '%s' THEN '%s'" % (reason, reason)
                for reason in REASONS[:-1])
            + " ELSE 'other' END, COUNT(*) FROM chunk_extraction_attempts GROUP BY 1,2")

def _details_query():
    return ("SELECT CASE WHEN "
            "typeof(last_failure_details)='text' AND "
            "length(CAST(last_failure_details AS BLOB))<=%d THEN last_failure_details "
            "ELSE NULL END FROM chunk_extraction_attempts" % MAX_DETAILS_CHARS)

def _detail_codes(value):
    # Two split levels are possible in this frozen extraction path. Keep a
    # small further margin, but reject arbitrary nesting as an unknown code.
    if type(value) is not str or len(value) > MAX_DETAIL_CHARS:
        return None
    for _ in range(4):
        if value.startswith('left.'):
            value = value[5:]
        elif value.startswith('right.'):
            value = value[6:]
        else:
            break
    if value.startswith('left.') or value.startswith('right.'):
        return 'other'
    if value in CODES:
        return value
    if value in {'left:grounding_failure', 'right:grounding_failure',
                 'left:branch_incomplete', 'right:branch_incomplete'}:
        return None
    return 'other'

def _categories(details):
    if type(details) is not str or len(details) > MAX_DETAILS_CHARS:
        return {'invalid_details'}
    try:
        items = json.loads(details)
    except (ValueError, TypeError, RecursionError, OverflowError):
        return {'invalid_details'}
    if (type(items) is not list or len(items) > MAX_DETAILS_ITEMS
            or any(type(item) is not str or len(item) > MAX_DETAIL_CHARS
                   for item in items)):
        return {'invalid_details'}
    found = set()
    for item in items:
        if item == 'diagnostics:truncated':
            found.add('truncated_details')
        elif item == 'diagnostic:invalid':
            found.add('invalid_details')
        else:
            classified = _detail_codes(item)
            if classified is not None:
                found.add(classified)
    if not any(item in CODES or item == 'other' for item in found):
        found.add('no_grounding_code')
    return found

def _census(path, label):
    parent = path.parent
    _stat(parent, directory=True)
    if parent.parent != root / 'run' or not parent.resolve().is_relative_to(root):
        raise ValueError('database_directory_invalid')
    before = _stat(path)
    if not 0 < before.st_size <= 512 * 1024 * 1024:
        raise ValueError('database_size_invalid')
    _sidecars(path)
    digest = _digest(path, before)
    if _identity(_stat(path)) != _identity(before):
        raise ValueError('database_changed')
    start = time.monotonic()
    steps = [0]
    connection = sqlite3.connect('file:' + quote(str(path)) + '?mode=ro&immutable=1',
                                 uri=True, timeout=1)
    try:
        connection.enable_load_extension(False)
        def authorizer(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and database == 'main' and (arg1, arg2) in {
                    ('chunk_extraction_attempts', 'attempts'),
                    ('chunk_extraction_attempts', 'last_failure_reason'),
                    ('chunk_extraction_attempts', 'last_failure_details')}:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and arg2 in {'typeof', 'count', 'length'}:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        connection.set_authorizer(authorizer)
        def progress():
            steps[0] += 1000
            return 1 if steps[0] > 20_000_000 or time.monotonic() - start > 15 else 0
        connection.set_progress_handler(progress, 1000)
        histogram = {str(attempt): {reason: 0 for reason in REASONS} for attempt in (1,2,3)}
        total = 0
        for attempt, reason, count in connection.execute(_reason_query()):
            if (type(attempt) is not int or attempt not in (1,2,3)
                    or type(reason) is not str or reason not in REASONS
                    or type(count) is not int or not 0 < count <= MAX_ROWS):
                raise ValueError('invalid_retry_bucket')
            total += count
            if total > MAX_ROWS or histogram[str(attempt)][reason]:
                raise ValueError('retry_row_bound')
            histogram[str(attempt)][reason] = count
        expected = EXPECTED[label]
        if (total != sum(sum(pair) for pair in expected)
                or any(histogram[str(attempt)]['grounding_failure'] != pair[0]
                       or histogram[str(attempt)]['branch_incomplete'] != pair[1]
                       or any(value for reason, value in histogram[str(attempt)].items()
                              if reason not in {'grounding_failure', 'branch_incomplete'})
                       for attempt, pair in enumerate(expected, 1))):
            raise ValueError('v1_histogram_changed')
        counts = {category: 0 for category in CATEGORIES}
        seen = 0
        for (details,) in connection.execute(_details_query()):
            seen += 1
            if seen > MAX_ROWS or seen > total:
                raise ValueError('retry_row_bound')
            for category in _categories(details):
                counts[category] += 1
        if seen != total:
            raise ValueError('retry_row_count_changed')
    finally:
        connection.close()
    if _identity(_stat(path)) != _identity(before) or _digest(path, before) != digest:
        raise ValueError('database_changed')
    _sidecars(path)
    return {'rows': total, 'categories': {key: value for key, value in counts.items() if value}}

result = {'schema': SCHEMA, 'source_receipt_terminal_cleanup_verified': True,
          'v1_histogram_reconciled': True, 'questions': {}}
for index in range(4):
    label = 'q-%04d' % index
    result['questions'][label] = _census(root / 'run' / label / 'hymem.sqlite', label)
encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
if len(encoded.encode('utf-8')) > MAX_OUTPUT_BYTES:
    raise ValueError('output_bound')
print(encoded)
'''


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_projection_field")
        result[key] = value
    return result


def _validated(value: object) -> dict | None:
    if (type(value) is not dict or set(value) != {
            "schema", "source_receipt_terminal_cleanup_verified",
            "v1_histogram_reconciled", "questions"}
            or value["schema"] != SCHEMA
            or value["source_receipt_terminal_cleanup_verified"] is not True
            or value["v1_histogram_reconciled"] is not True
            or type(value["questions"]) is not dict
            or set(value["questions"]) != set(EXPECTED)):
        return None
    for label, item in value["questions"].items():
        if (type(item) is not dict or set(item) != {"rows", "categories"}
                or type(item["rows"]) is not int
                or item["rows"] != sum(sum(pair) for pair in EXPECTED[label])
                or type(item["categories"]) is not dict
                or not set(item["categories"]) <= set(CATEGORIES)):
            return None
        if any(type(count) is not int or not 0 < count <= item["rows"]
               for count in item["categories"].values()):
            return None
        if sum(item["categories"].values()) < item["rows"]:
            return None
    return value


def _pinned(path: Path, expected: str) -> str:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("source_pin_mismatch")
    return source.decode("utf-8")


def _payload() -> str:
    _pinned(V1, V1_SHA)
    metadata = _pinned(METADATA, METADATA_SHA)
    reader = _pinned(READER, READER_SHA)
    return ("import json\n" + "METADATA_SOURCE=" + repr(metadata) + "\n"
            + "READER_SOURCE=" + repr(reader) + "\n"
            + "ROOT=" + repr(ROOT) + "\nSCHEMA=" + repr(SCHEMA) + "\n"
            + "REASONS=" + repr(REASONS) + "\nMAX_ROWS=" + repr(MAX_ROWS) + "\n"
            + "MAX_DETAILS_CHARS=" + repr(MAX_DETAILS_CHARS) + "\n"
            + "MAX_DETAILS_ITEMS=" + repr(MAX_DETAILS_ITEMS) + "\n"
            + "MAX_DETAIL_CHARS=" + repr(MAX_DETAIL_CHARS) + "\n"
            + "MAX_OUTPUT_BYTES=" + repr(MAX_OUTPUT_BYTES) + "\n"
            + "EXPECTED=" + repr(EXPECTED) + "\n"
            + "CODES=" + repr(CODES) + "\nCATEGORIES=" + repr(CATEGORIES) + "\n"
            + REMOTE)


def main() -> int:
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite",
            "/usr/bin/python3 -I -B -"], input=_payload(), text=True,
            capture_output=True, timeout=90)
        if result.returncode != 0 or len(result.stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError("remote_unavailable")
        decoded = json.loads(result.stdout, object_pairs_hook=_unique_fields,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        safe = _validated(decoded)
        if safe is None:
            raise ValueError("projection_invalid")
        print(json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except (OSError, UnicodeError, subprocess.SubprocessError, ValueError, TypeError):
        print(json.dumps({"schema": SCHEMA, "status": "reasons_unavailable"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
