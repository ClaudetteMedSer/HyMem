"""Closed, read-only log counters for the stopped afit9i7d instrumented pilot.

The warning comes from frozen chunk.py's single_attempt exception handler.
These counts corroborate observed call failures; they cannot recover the
original question exception or distinguish its phase.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tools.diagnostics import luna_instrumented_failure_metadata_v1 as v1


SCHEMA = "luna-instrumented-error-type-census-v1"
V1_SHA = "a892dbd759ab0ebd5bfacf81eee0b7e5e00f5de530cf1cc340453f235d164826"
CHUNK_SHA = "c644513152e2ddefbe0d0d18dce7d597d18b4ba78ce275dd04ad4b2133ee7b92"
CHUNK = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle/candidate/hymem/extraction/chunk.py")
ERROR_TYPES = (
    "ValueError", "RuntimeError", "TypeError", "AttributeError", "KeyError",
    "TimeoutError", "OSError", "OperationalError", "IntegrityError",
    "ProgrammingError", "DatabaseError", "NotSupportedError", "PermissionError",
    "MemoryError", "RecursionError", "IndexError", "AssertionError",
    "ConcurrentStop", "DreamLeaseLost", "other",
)
MAX_LOG_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 8192
READ_BYTES = 65536
DEADLINE_SECONDS = 20

PROJECTION = r'''
import os
import re
import stat
import time

_WARNING = b'chunk_extraction.call_failure error_type='
_CLASS_NAME = re.compile(rb'[A-Za-z_][A-Za-z0-9_]{0,127}\Z')

def _census(root, namespace, base):
    if (type(base) is not dict or base.get('schema') != V1_SCHEMA
            or base.get('terminal_and_cleanup_verified') is not True
            or base.get('source_receipt_verified') is not True):
        raise ValueError('v1_gate_invalid')
    source = root/'candidate'/'hymem'/'extraction'/'chunk.py'
    if not namespace['_file'](source, root, 2_000_000):
        raise ValueError('candidate_source_invalid')
    if namespace['_sha'](source) != CHUNK_SHA:
        raise ValueError('candidate_source_invalid')
    path = root/'private-diagnostic-run.log'
    if not namespace['_file'](path, root, MAX_LOG_BYTES):
        raise ValueError('log_invalid')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode)
                or not 0 <= before.st_size <= MAX_LOG_BYTES):
            raise ValueError('log_invalid')
        counts = {name: 0 for name in ERROR_TYPES}
        total = 0
        size = 0
        line = b''
        deadline = time.monotonic() + DEADLINE_SECONDS
        def accept(value):
            nonlocal total
            if len(value) > MAX_LINE_BYTES:
                raise ValueError('log_line_limit')
            if value.startswith(_WARNING):
                # Full-line warning only. Dynamic class names are never keys.
                suffix = value[len(_WARNING):]
                if _CLASS_NAME.fullmatch(suffix) is None:
                    return
                name = suffix.decode('ascii')
                key = name if name in counts and name != 'other' else 'other'
                counts[key] += 1
                total += 1
        while True:
            if time.monotonic() > deadline:
                raise ValueError('log_deadline')
            block = os.read(fd, READ_BYTES)
            if not block:
                break
            size += len(block)
            if size > MAX_LOG_BYTES or size > before.st_size:
                raise ValueError('log_size_changed')
            parts = (line + block).split(b'\n')
            for completed in parts[:-1]:
                if time.monotonic() > deadline:
                    raise ValueError('log_deadline')
                accept(completed)
            line = parts[-1]
            if len(line) > MAX_LINE_BYTES:
                raise ValueError('log_line_limit')
        if line:
            accept(line)
        after = os.fstat(fd)
        if (size != before.st_size or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ino != before.st_ino):
            raise ValueError('log_changed')
        return {'schema': SCHEMA, 'terminal_and_cleanup_verified': True,
            'source_receipt_verified': True, 'candidate_source_verified': True,
            'total_call_failure_warnings': total, 'error_type_counts': counts}
    finally:
        os.close(fd)
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
extension = {'V1_SCHEMA': V1_SCHEMA, 'SCHEMA': SCHEMA, 'CHUNK_SHA': CHUNK_SHA,
             'ERROR_TYPES': ERROR_TYPES, 'MAX_LOG_BYTES': MAX_LOG_BYTES,
             'MAX_LINE_BYTES': MAX_LINE_BYTES, 'READ_BYTES': READ_BYTES,
             'DEADLINE_SECONDS': DEADLINE_SECONDS}
exec(compile(PROJECTION, '<bounded-log-census>', 'exec'), extension)
print(json.dumps(extension['_census'](root, namespace, base),
                 sort_keys=True, separators=(',', ':'), allow_nan=False))
'''


def _validated(value: object) -> dict | None:
    if type(value) is not dict or set(value) != {
            "schema", "terminal_and_cleanup_verified", "source_receipt_verified",
            "candidate_source_verified", "total_call_failure_warnings",
            "error_type_counts"}:
        return None
    if (value["schema"] != SCHEMA or value["terminal_and_cleanup_verified"] is not True
            or value["source_receipt_verified"] is not True
            or value["candidate_source_verified"] is not True):
        return None
    counts = value["error_type_counts"]
    if (type(counts) is not dict or set(counts) != set(ERROR_TYPES)
            or any(type(counts[name]) is not int or not 0 <= counts[name] <= MAX_LOG_BYTES
                   for name in ERROR_TYPES)):
        return None
    total = value["total_call_failure_warnings"]
    if type(total) is not int or total != sum(counts.values()) or total > MAX_LOG_BYTES:
        return None
    return value


def main() -> int:
    try:
        if hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest() != V1_SHA:
            raise ValueError("v1_pin_mismatch")
        source = v1.READER.read_bytes()
        if hashlib.sha256(source).hexdigest() != v1.READER_SHA:
            raise ValueError("reader_pin_mismatch")
        if hashlib.sha256(CHUNK.read_bytes()).hexdigest() != CHUNK_SHA:
            raise ValueError("candidate_pin_mismatch")
        payload = ("import json\nSOURCE=" + repr(source.decode("utf-8")) + "\n"
            + "ROOT=" + repr(v1.ROOT) + "\nRECEIPT_SHA=" + repr(v1.RECEIPT_SHA)
            + "\nV1_SCHEMA=" + repr(v1.SCHEMA) + "\nSCHEMA=" + repr(SCHEMA)
            + "\nINDEXING_CODES=" + repr(v1.INDEXING_CODES)
            + "\nEXCEPTION_TYPES=" + repr(v1.EXCEPTION_TYPES)
            + "\nV1_PROJECTION=" + repr(v1.PROJECTION)
            + "\nCHUNK_SHA=" + repr(CHUNK_SHA) + "\nERROR_TYPES=" + repr(ERROR_TYPES)
            + "\nMAX_LOG_BYTES=" + repr(MAX_LOG_BYTES)
            + "\nMAX_LINE_BYTES=" + repr(MAX_LINE_BYTES)
            + "\nREAD_BYTES=" + repr(READ_BYTES)
            + "\nDEADLINE_SECONDS=" + repr(DEADLINE_SECONDS)
            + "\nPROJECTION=" + repr(PROJECTION) + "\n" + REMOTE)
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite",
            "/usr/bin/python3 -I -B -"], input=payload, text=True,
            capture_output=True, timeout=40)
        if result.returncode != 0 or len(result.stdout) > 8192:
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
