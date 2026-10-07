"""Closed retry-reason census for the single consumed SIWC LME pilot.

The host receives source-pinned Python on SSH stdin. Only finite histograms
leave it; this tool never prints a SQLite error, raw row, or host stderr.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "siwc-lme-retry-reasons-v1"
ROOT = "/home/atta/.hymem-siwc-lme-diagnostic-preflight-d2hxx626"
METADATA = Path(__file__).with_name("siwc_lme_failure_metadata_v1.py")
METADATA_SHA = "f5baf130c27138f3f248c17483f6d89ea9b320b49ff7c8c7219d7f7e5666b7b6"
READER = Path(__file__).with_name("siwc_lme_diagnostic_progress_v3.py")
READER_SHA = "89e39169a81d9ee810e54fb4c143753906aed282f50527b923cca6cc504a5279"
# Frozen candidate hymem/extraction/chunk.py _FAILURE_REASONS, source SHA256
# c644513152e2ddefbe0d0d18dce7d597d18b4ba78ce275dd04ad4b2133ee7b92.
REASONS = (
    "branch_incomplete", "call_failure", "contract_failure", "grounding_failure",
    "incomplete_response", "input_contract_failure", "internal_validation_failure",
    "item_validation_failure", "output_limit_exceeded", "parse_failure",
    "resource_limit", "response_conflict", "shape_failure",
    "source_coverage_failure", "unspecified_failure", "other",
)
MAX_ROWS = 100_000


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

def _census(path):
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
    query = ('SELECT CASE WHEN typeof(attempts)=\'integer\' AND attempts=1 THEN 1 '
             'WHEN typeof(attempts)=\'integer\' AND attempts=2 THEN 2 '
             'WHEN typeof(attempts)=\'integer\' AND attempts=3 THEN 3 ELSE 0 END, '
             'CASE '
             + ' '.join("WHEN typeof(last_failure_reason)='text' AND "
                        "last_failure_reason COLLATE BINARY = '%s' THEN '%s'"
                        % (reason, reason) for reason in REASONS[:-1])
             + " ELSE 'other' END, COUNT(*) FROM chunk_extraction_attempts GROUP BY 1,2")
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
                    ('chunk_extraction_attempts', 'last_failure_reason')}:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and arg2 in {'typeof', 'count'}:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        connection.set_authorizer(authorizer)
        def progress():
            steps[0] += 1000
            return 1 if steps[0] > 20_000_000 or time.monotonic() - start > 15 else 0
        connection.set_progress_handler(progress, 1000)
        counts = {str(attempt): {reason: 0 for reason in REASONS} for attempt in (1,2,3)}
        total = 0
        for attempt, reason, count in connection.execute(query):
            if (type(attempt) is not int or attempt not in (1,2,3)
                    or type(reason) is not str or reason not in REASONS
                    or type(count) is not int or not 0 < count <= MAX_ROWS):
                raise ValueError('invalid_retry_bucket')
            total += count
            if total > MAX_ROWS or counts[str(attempt)][reason]:
                raise ValueError('retry_row_bound')
            counts[str(attempt)][reason] = count
    finally:
        connection.close()
    if _identity(_stat(path)) != _identity(before) or _digest(path, before) != digest:
        raise ValueError('database_changed')
    _sidecars(path)
    return {'rows': total, 'by_attempt': counts}

result = {'schema': SCHEMA, 'source_receipt_terminal_cleanup_verified': True,
          'questions': {}}
for index in range(4):
    label = 'q-%04d' % index
    result['questions'][label] = _census(root / 'run' / label / 'hymem.sqlite')
print(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))
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
            "schema", "source_receipt_terminal_cleanup_verified", "questions"}
            or value["schema"] != SCHEMA
            or value["source_receipt_terminal_cleanup_verified"] is not True
            or type(value["questions"]) is not dict
            or set(value["questions"]) != {f"q-{i:04d}" for i in range(4)}):
        return None
    for item in value["questions"].values():
        if (type(item) is not dict or set(item) != {"rows", "by_attempt"}
                or type(item["rows"]) is not int or not 0 <= item["rows"] <= MAX_ROWS
                or type(item["by_attempt"]) is not dict
                or set(item["by_attempt"]) != {"1", "2", "3"}):
            return None
        total = 0
        for buckets in item["by_attempt"].values():
            if (type(buckets) is not dict or set(buckets) != set(REASONS)
                    or any(type(count) is not int or not 0 <= count <= MAX_ROWS
                           for count in buckets.values())):
                return None
            total += sum(buckets.values())
        if total != item["rows"]:
            return None
    return value


def _pinned(path: Path, expected: str) -> str:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("source_pin_mismatch")
    return source.decode("utf-8")


def _payload() -> str:
    metadata = _pinned(METADATA, METADATA_SHA)
    reader = _pinned(READER, READER_SHA)
    return ("import json\n" + "METADATA_SOURCE=" + repr(metadata) + "\n"
            + "READER_SOURCE=" + repr(reader) + "\n"
            + "ROOT=" + repr(ROOT) + "\nSCHEMA=" + repr(SCHEMA) + "\n"
            + "REASONS=" + repr(REASONS) + "\nMAX_ROWS=" + repr(MAX_ROWS) + "\n" + REMOTE)


def main() -> int:
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite",
            "/usr/bin/python3 -I -B -"], input=_payload(), text=True,
            capture_output=True, timeout=90)
        if result.returncode != 0 or len(result.stdout.encode("utf-8")) > 8192:
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
