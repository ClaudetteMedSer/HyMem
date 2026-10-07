"""Closed counters for both retained warning locations of stopped afit9i7d.

The source-defined warning may reach the runner log or systemd's fixed stderr
file. Counts corroborate observed call failures only; zero counts do not
explain the original question failure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tools.diagnostics import luna_instrumented_error_type_census_v1 as prior


SCHEMA = "luna-instrumented-error-type-census-v2"
PRIOR_SHA = "d8aff9369a6586079f5a8d4f12faf350828f7af2e01b3aeeda559057897cd6c8"
LAUNCHER = Path(__file__).with_name("luna_lme_diagnostic_launch_v8.py")
LAUNCHER_SHA = "6c620e446e6dbee4c1a68d1105dadba64510d3beed2a92f3b58f4b79185ef504"
FILE_LABELS = ("diagnostic_run", "launch_stderr")
FILE_NAMES = ("private-diagnostic-run.log", "private-launch-stderr.log")

EXTENSION = r'''
import os
import re
import stat
import time

_WARNING = b'chunk_extraction.call_failure error_type='
_PREFIX = b'WARNING:hymem.extraction.chunk:'
_CLASS_NAME = re.compile(rb'[A-Za-z_][A-Za-z0-9_]{0,127}\Z')

def _scan(path, root, namespace, deadline):
    if not namespace['_file'](path, root, MAX_LOG_BYTES):
        raise ValueError('log_invalid')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= MAX_LOG_BYTES:
            raise ValueError('log_invalid')
        counts = {name: 0 for name in ERROR_TYPES}
        bare = {name: 0 for name in ERROR_TYPES}
        size = 0
        line = b''
        def accept(value):
            if len(value) > MAX_LINE_BYTES:
                raise ValueError('log_line_limit')
            is_bare = value.startswith(_WARNING)
            if is_bare:
                suffix = value[len(_WARNING):]
            elif value.startswith(_PREFIX + _WARNING):
                suffix = value[len(_PREFIX) + len(_WARNING):]
            else:
                return
            if _CLASS_NAME.fullmatch(suffix) is None:
                return
            name = suffix.decode('ascii')
            key = name if name in counts and name != 'other' else 'other'
            counts[key] += 1
            if is_bare:
                bare[key] += 1
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
        return {'total_call_failure_warnings': sum(counts.values()),
                'error_type_counts': counts}, bare
    finally:
        os.close(fd)

def _extend(root, namespace, baseline):
    if (type(baseline) is not dict or baseline.get('schema') != PRIOR_SCHEMA
            or baseline.get('terminal_and_cleanup_verified') is not True
            or baseline.get('source_receipt_verified') is not True
            or baseline.get('candidate_source_verified') is not True):
        raise ValueError('prior_gate_invalid')
    deadline = time.monotonic() + DEADLINE_SECONDS
    files = {}
    for label, name in zip(FILE_LABELS, FILE_NAMES, strict=True):
        files[label], bare = _scan(root/name, root, namespace, deadline)
        if label == 'diagnostic_run' and bare != baseline.get('error_type_counts'):
            raise ValueError('prior_log_changed')
    counts = {name: sum(files[label]['error_type_counts'][name]
                        for label in FILE_LABELS) for name in ERROR_TYPES}
    return {'schema': SCHEMA, 'terminal_and_cleanup_verified': True,
            'source_receipt_verified': True, 'candidate_source_verified': True,
            'files': files, 'total_call_failure_warnings': sum(counts.values()),
            'error_type_counts': counts}
'''

REMOTE = r'''
namespace = {'__name__': 'reviewed_reader', '__file__': '<reviewed-reader>'}
exec(compile(SOURCE, '<reviewed-reader>', 'exec'), namespace)
root = namespace['Path'](ROOT)
report = namespace['inspect'](root, RECEIPT_SHA)
scope = {'INDEXING_CODES': INDEXING_CODES, 'EXCEPTION_TYPES': EXCEPTION_TYPES,
         'RECEIPT_SHA': RECEIPT_SHA, 'SCHEMA': METADATA_SCHEMA}
exec(compile(METADATA_PROJECTION, '<frozen-metadata-projection>', 'exec'), scope)
base = scope['_project'](root, namespace, report)
prior_scope = {'V1_SCHEMA': METADATA_SCHEMA, 'SCHEMA': PRIOR_SCHEMA,
    'CHUNK_SHA': CHUNK_SHA, 'ERROR_TYPES': ERROR_TYPES,
    'MAX_LOG_BYTES': MAX_LOG_BYTES, 'MAX_LINE_BYTES': MAX_LINE_BYTES,
    'READ_BYTES': READ_BYTES, 'DEADLINE_SECONDS': DEADLINE_SECONDS}
exec(compile(PRIOR_PROJECTION, '<frozen-prior-census>', 'exec'), prior_scope)
baseline = prior_scope['_census'](root, namespace, base)
extension = {'SCHEMA': SCHEMA, 'PRIOR_SCHEMA': PRIOR_SCHEMA,
    'ERROR_TYPES': ERROR_TYPES, 'FILE_LABELS': FILE_LABELS,
    'FILE_NAMES': FILE_NAMES, 'MAX_LOG_BYTES': MAX_LOG_BYTES,
    'MAX_LINE_BYTES': MAX_LINE_BYTES, 'READ_BYTES': READ_BYTES,
    'DEADLINE_SECONDS': DEADLINE_SECONDS}
exec(compile(EXTENSION, '<two-fixed-log-census>', 'exec'), extension)
print(json.dumps(extension['_extend'](root, namespace, baseline),
                 sort_keys=True, separators=(',', ':'), allow_nan=False))
'''


def _valid_counts(value: object, limit: int) -> bool:
    if type(value) is not dict or set(value) != {
            "total_call_failure_warnings", "error_type_counts"}:
        return False
    counts = value["error_type_counts"]
    if (type(counts) is not dict or set(counts) != set(prior.ERROR_TYPES)
            or any(type(counts[name]) is not int
                   or not 0 <= counts[name] <= limit
                   for name in prior.ERROR_TYPES)):
        return False
    total = value["total_call_failure_warnings"]
    return (type(total) is int and total == sum(counts.values())
        and 0 <= total <= limit)


def _validated(value: object) -> dict | None:
    if type(value) is not dict or set(value) != {
            "schema", "terminal_and_cleanup_verified", "source_receipt_verified",
            "candidate_source_verified", "files", "total_call_failure_warnings",
            "error_type_counts"}:
        return None
    if (value["schema"] != SCHEMA or value["terminal_and_cleanup_verified"] is not True
            or value["source_receipt_verified"] is not True
            or value["candidate_source_verified"] is not True
            or type(value["files"]) is not dict
            or set(value["files"]) != set(FILE_LABELS)):
        return None
    if not _valid_counts({key: value[key] for key in (
            "total_call_failure_warnings", "error_type_counts")},
            2 * prior.MAX_LOG_BYTES):
        return None
    if any(not _valid_counts(value["files"][label], prior.MAX_LOG_BYTES)
           for label in FILE_LABELS):
        return None
    if any(value["error_type_counts"][name] != sum(
            value["files"][label]["error_type_counts"][name]
            for label in FILE_LABELS) for name in prior.ERROR_TYPES):
        return None
    return value


def main() -> int:
    try:
        if hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest() != PRIOR_SHA:
            raise ValueError("prior_pin_mismatch")
        if hashlib.sha256(Path(prior.v1.__file__).read_bytes()).hexdigest() != prior.V1_SHA:
            raise ValueError("metadata_pin_mismatch")
        source = prior.v1.READER.read_bytes()
        if hashlib.sha256(source).hexdigest() != prior.v1.READER_SHA:
            raise ValueError("reader_pin_mismatch")
        if hashlib.sha256(prior.CHUNK.read_bytes()).hexdigest() != prior.CHUNK_SHA:
            raise ValueError("candidate_pin_mismatch")
        if hashlib.sha256(LAUNCHER.read_bytes()).hexdigest() != LAUNCHER_SHA:
            raise ValueError("launcher_pin_mismatch")
        payload = ("import json\nSOURCE=" + repr(source.decode("utf-8")) + "\n"
            + "ROOT=" + repr(prior.v1.ROOT)
            + "\nRECEIPT_SHA=" + repr(prior.v1.RECEIPT_SHA)
            + "\nMETADATA_SCHEMA=" + repr(prior.v1.SCHEMA)
            + "\nPRIOR_SCHEMA=" + repr(prior.SCHEMA)
            + "\nSCHEMA=" + repr(SCHEMA)
            + "\nINDEXING_CODES=" + repr(prior.v1.INDEXING_CODES)
            + "\nEXCEPTION_TYPES=" + repr(prior.v1.EXCEPTION_TYPES)
            + "\nMETADATA_PROJECTION=" + repr(prior.v1.PROJECTION)
            + "\nCHUNK_SHA=" + repr(prior.CHUNK_SHA)
            + "\nERROR_TYPES=" + repr(prior.ERROR_TYPES)
            + "\nMAX_LOG_BYTES=" + repr(prior.MAX_LOG_BYTES)
            + "\nMAX_LINE_BYTES=" + repr(prior.MAX_LINE_BYTES)
            + "\nREAD_BYTES=" + repr(prior.READ_BYTES)
            + "\nDEADLINE_SECONDS=" + repr(prior.DEADLINE_SECONDS)
            + "\nPRIOR_PROJECTION=" + repr(prior.PROJECTION)
            + "\nFILE_LABELS=" + repr(FILE_LABELS)
            + "\nFILE_NAMES=" + repr(FILE_NAMES)
            + "\nEXTENSION=" + repr(EXTENSION) + "\n" + REMOTE)
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite",
            "/usr/bin/python3 -I -B -"], input=payload, text=True,
            capture_output=True, timeout=60)
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
