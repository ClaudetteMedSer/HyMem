#!/usr/bin/env python3
"""Diagnostic adapter for closed, hash-sealed SQLite main files only.

The host must establish that the paid worker exited with PID zero before launch.
Immutable reads cannot include outstanding WAL transactions; sidecars and main
file pins are therefore checked on both sides of every backup.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat
import traceback

V1_FILE = Path('/diag/postflight_v1.py')
V1_SHA256 = '1ca6318fdac58d7b0715547baa8b80d4119c9a1a2012fb9f58bf96c3a443b117'
SOURCE = Path('/private-dream/hymem.sqlite')
BASELINE = Path('/reference/source.sqlite')
SOURCE_SHA256 = 'f4f9cb8ec8247d27ab2981af58ec0c044fb76cc71356f2e6a757540d4eae59ae'
WORK = Path('/work')


def digest(path):
    with path.open('rb') as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def regular(path, mode=None):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError('sealed_input_not_regular')
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        raise ValueError('sealed_input_mode_invalid')
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError('sealed_input_parent_symlink')


def sealed(path, expected, mode=None):
    regular(path, mode)
    for suffix in ('-wal', '-journal'):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            regular(sidecar)
            if sidecar.stat().st_size:
                raise ValueError('sealed_input_nonempty_sidecar')
    if digest(path) != expected:
        raise ValueError('sealed_input_pin_drift')


def backup_reader(pins, work):
    """Build the sole replacement callback; no global SQLite changes."""
    def backup(source, target):
        if source not in pins:
            raise ValueError('sealed_input_path_invalid')
        expected, mode, name = pins[source]
        if target != work / name:
            raise ValueError('sealed_output_path_invalid')
        if (not work.is_dir() or work.is_symlink()
                or stat.S_IMODE(work.stat().st_mode) != 0o700):
            raise ValueError('private_work_directory_invalid')
        sealed(source, expected, mode)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        try:
            origin = sqlite3.connect(source.as_uri() + '?mode=ro&immutable=1', uri=True)
            try:
                copy = sqlite3.connect(target)
                try:
                    origin.backup(copy)
                finally:
                    copy.close()
            finally:
                origin.close()
        finally:
            sealed(source, expected, mode)
    return backup


def load_v1(path=V1_FILE):
    regular(path)
    if digest(path) != V1_SHA256:
        raise RuntimeError('postflight_v1_pin_drift')
    spec = importlib.util.spec_from_file_location('sealed_postflight_v1', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('postflight_v1_unloadable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.load_audit

    def load_audit(path=module.AUDIT_FILE):
        audit = original(path)  # Retains the exact audit helper pin.
        audit.backup_readonly = backup_reader({
            BASELINE: (audit.REFERENCE_SHA, 0o400, 'baseline.sqlite'),
            SOURCE: (SOURCE_SHA256, None, 'dream.sqlite'),
        }, WORK)
        return audit

    module.load_audit = load_audit
    return module


def main():
    os.umask(0o077)
    if (not WORK.is_dir() or WORK.is_symlink()
            or stat.S_IMODE(WORK.stat().st_mode) != 0o700 or any(WORK.iterdir())):
        raise ValueError('private_work_directory_not_fresh')
    return load_v1().main()


if __name__ == '__main__':
    try:
        code = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink() and stat.S_IMODE(WORK.stat().st_mode) == 0o700:
                raw = json.dumps({'type': type(exc).__name__,
                                  'traceback': ''.join(traceback.format_exception(exc))},
                                 sort_keys=True, ensure_ascii=True).encode()
                fd = os.open(WORK / 'postflight-failure.json',
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                captured = True
        except BaseException:
            pass
        safe = type(exc).__name__ if type(exc).__name__ in {
            'ValueError', 'RuntimeError', 'TypeError', 'OperationalError',
            'DatabaseError', 'OSError', 'KeyError'} else 'Exception'
        print(json.dumps({'status': 'error', 'error_type': safe,
                          'failure_captured': captured}, sort_keys=True))
        code = 1
    raise SystemExit(code)
