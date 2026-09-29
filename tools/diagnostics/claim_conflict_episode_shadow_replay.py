#!/usr/bin/env python3
"""Offline, clone-only surplus episode-vector replay; stdout contains metadata only.

Mount the pinned candidate at /candidate read-only, the entire closed source
directory at /private-dream read-only (including SQLite sidecar siblings),
and the pinned store audit at /diag.
Supply an empty private /work. Run with Docker --network none and an outer
timeout; the lexical application deadline is cooperative, not preemptive.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sqlite3
import stat
import struct
import sys
import traceback

SOURCE = Path('/private-dream/hymem.sqlite')
SOURCE_SHA = 'f4f9cb8ec8247d27ab2981af58ec0c044fb76cc71356f2e6a757540d4eae59ae'
WORK = Path('/work')
AUDIT = Path('/diag/claim_conflict_store_audit.py')
AUDIT_SHA = 'e2efe365c5aedbbe88d86d521dd37b821759afc5fb21c21a6662e6a8ce567f42'
TIMEOUT_SECONDS = 60


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def regular_path(path):
    """Reject symlinks in every path component, including parent directories."""
    path = path.absolute()
    for component in (path, *path.parents):
        require(not component.is_symlink(), 'symlink_input')
    require(stat.S_ISREG(path.lstat().st_mode), 'nonregular_input')


def sha(path):
    regular_path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_guard(path, expected_sha):
    regular_path(path)
    for suffix in ('-wal', '-shm', '-journal'):
        require(not os.path.lexists(str(path) + suffix), 'source_sidecar_present')
    require(sha(path) == expected_sha, 'source_pin_drift')
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns)


def clone_immutable(source, target, expected_sha):
    before = source_guard(source, expected_sha)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    with contextlib.closing(sqlite3.connect(
        source.absolute().as_uri() + '?mode=ro&immutable=1', uri=True,
    )) as origin, contextlib.closing(sqlite3.connect(target)) as copy:
        origin.execute('PRAGMA query_only=ON')
        origin.backup(copy)
    require(source_guard(source, expected_sha) == before, 'source_changed_during_backup')
    return before


def load_audit():
    require(sha(AUDIT) == AUDIT_SHA, 'audit_pin_drift')
    spec = importlib.util.spec_from_file_location('episode_shadow_pinned_audit', AUDIT)
    require(spec is not None and spec.loader is not None, 'audit_unloadable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(value):
    """Typed, length-delimited encoding avoids value/string/hash ambiguity."""
    if value is None:
        return b'N'
    if isinstance(value, bytes):
        tag, raw = b'B', value
    elif isinstance(value, str):
        tag, raw = b'S', value.encode('utf-8')
    elif isinstance(value, int):
        tag, raw = b'I', str(value).encode('ascii')
    elif isinstance(value, float):
        tag, raw = b'F', struct.pack('>d', value)
    else:
        raise RuntimeError('unsupported_sqlite_value')
    return tag + struct.pack('>Q', len(raw)) + raw


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def semantic_fingerprint(conn):
    """All logical rows, including other FTS/vec shadows; exact own exclusion.

    Row order is irrelevant; duplicate row hashes retain multiplicity. Include
    rowid for rowid tables so identity-only edits cannot evade the gate.
    sqlite_master is included separately and without any approved exclusion.
    """
    listing = conn.execute('PRAGMA main.table_list').fetchall()
    # sqlite-vec lazily reports vector_chunks00 as an ordinary table. Use the
    # exact single-embedding vec0 storage inventory, never a prefix exclusion.
    own = {'vec_episodes_info', 'vec_episodes_chunks', 'vec_episodes_rowids',
           'vec_episodes_vector_chunks00'}
    actual_own = {row[1] for row in listing if row[0] == 'main'
                  and row[1].startswith('vec_episodes_')}
    require(actual_own == own, 'episode_storage_inventory_changed')
    require(any(row[1] == 'vec_episodes' and row[2] == 'virtual' for row in listing),
            'episode_virtual_table_missing')
    payload = {}
    for schema, name, kind, _ncol, without_rowid, _strict in listing:
        if schema != 'main' or kind == 'view' or name == 'sqlite_schema':
            continue
        if name == 'vec_episodes' or name in own:
            continue
        columns = [row[1] for row in conn.execute(f'PRAGMA main.table_xinfo({quoted(name)})')]
        # Virtual-table rowid may be an alias. Explicitly retain it anyway.
        selection = ('rowid,' if not without_rowid else '') + '*'
        rows = [hashlib.sha256(b''.join(encoded(value) for value in row)).digest()
                for row in conn.execute(f'SELECT {selection} FROM {quoted(name)}')]
        payload[name] = [columns, len(rows), hashlib.sha256(b''.join(sorted(rows))).hexdigest()]
    schema_rows = sorted(b''.join(encoded(v) for v in row) for row in conn.execute(
        'SELECT type,name,tbl_name,rootpage,sql FROM sqlite_master'))
    payload['$sqlite_master'] = hashlib.sha256(b''.join(schema_rows)).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def vector_snapshot(conn, db):
    from hymem.dreaming.aggregate import load_clusterable_episodes
    dim_row = conn.execute("SELECT value FROM schema_meta WHERE key='vec_dim'").fetchone()
    model_row = conn.execute("SELECT value FROM schema_meta WHERE key='vec_model'").fetchone()
    require(dim_row is not None and model_row is not None, 'vector_metadata_missing')
    dim, model = int(dim_row[0]), model_row[0]
    import re
    require(dim > 0 and isinstance(model, str) and re.fullmatch(
        r'hymem-embedding-producer-v1:[0-9a-f]{64}', model) is not None,
        'vector_metadata_invalid')
    expected = {}
    for episode in load_clusterable_episodes(
        conn, max_rowid=None, embedding_model=model, embedding_dim=dim,
    ):
        vec = db._finite_vec(episode['vector'], dim)
        require(vec is not None, 'invalid_authoritative_vector')
        expected[int(episode['rowid'])] = db._pack_vector(vec)
    actual = {int(row[0]): bytes(row[1]) for row in conn.execute(
        'SELECT rowid,embedding FROM vec_episodes')}
    require(all(actual.get(key) == value for key, value in expected.items()),
            'missing_or_different_vector')
    return expected, actual


def replay(source=SOURCE, work=WORK, expected_sha=SOURCE_SHA):
    require(work.is_dir() and not work.is_symlink(), 'work_invalid')
    for parent in work.absolute().parents:
        require(not parent.is_symlink(), 'work_parent_symlink')
    require(stat.S_IMODE(work.stat().st_mode) == 0o700 and not any(work.iterdir()),
            'work_not_fresh_private')
    audit = load_audit()
    source_identity = source_guard(source, expected_sha)
    target = work / 'episode-shadow.sqlite'
    clone_immutable(source, target, expected_sha)
    from hymem.core import db
    from hymem.deadline import MonotonicDeadline, use_deadline
    conn = db.connect(target)
    try:
        require(db.schema_version(conn) == 64, 'schema_not_64')
        require(db._load_vec_extension(conn), 'vector_extension_unavailable')
        before_health = audit.integrity(conn)
        require(audit.is_clean(before_health), 'pre_repair_health_failed')
        expected, actual = vector_snapshot(conn, db)
        require(len(actual) == 44 and len(expected) == 36
                and len(actual.keys() - expected.keys()) == 8, 'snapshot_counts_changed')
        before = semantic_fingerprint(conn)
        deadline = MonotonicDeadline.after(TIMEOUT_SECONDS)
        with use_deadline(deadline):
            require(db.prune_extra_episode_vectors(conn) is True, 'repair_refused')
            retained_expected, retained = vector_snapshot(conn, db)
            require(retained_expected == expected and retained == expected,
                    'repair_vector_mismatch')
            require(semantic_fingerprint(conn) == before, 'semantic_rows_changed')
            require(db.prune_extra_episode_vectors(conn) is False, 'repeat_not_noop')
            require(semantic_fingerprint(conn) == before, 'repeat_semantic_rows_changed')
            require(vector_snapshot(conn, db) == (expected, expected), 'repeat_vectors_changed')
        require(audit.is_clean(audit.integrity(conn)), 'post_repair_health_failed')
    finally:
        conn.close()
    conn = db.connect(target)
    try:
        require(db._load_vec_extension(conn), 'reopen_vector_extension_unavailable')
        require(db.vec_episodes_aligned(conn), 'reopen_alignment_failed')
        require(vector_snapshot(conn, db) == (expected, expected), 'reopen_vectors_changed')
        require(semantic_fingerprint(conn) == before, 'reopen_semantic_rows_changed')
        require(audit.is_clean(audit.integrity(conn)), 'reopen_health_failed')
    finally:
        conn.close()
    require(source_guard(source, expected_sha) == source_identity, 'source_changed')
    return {'status': 'verified', 'source_sha256': expected_sha,
            'source_unchanged': True, 'before_actual': 44, 'valid_vectors': 36,
            'removed_surplus': 8, 'after_actual': 36, 'exact_vectors_preserved': True,
            'semantic_sha256': before, 'semantic_rows_unchanged': True,
            'repeat_noop': True, 'reopen_aligned': True, 'health_clean': True}


def main():
    sys.path.insert(0, '/candidate')
    logging.disable(logging.CRITICAL)
    # Library output stays in private evidence, including unexpected prints.
    try:
        # replay requires empty /work; keep suppression in memory until finish.
        import io
        private_output = io.StringIO()
        with contextlib.redirect_stdout(private_output), contextlib.redirect_stderr(private_output):
            report = replay()
        code = 0
    except BaseException as exc:
        captured = False
        try:
            fd = os.open(WORK / 'episode-shadow-failure.txt', os.O_WRONLY | os.O_CREAT
                         | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w') as stream:
                stream.write(private_output.getvalue())
                stream.write(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            captured = True
        except BaseException:
            pass
        report = {'status': 'error', 'reason_code': 'episode_shadow_replay_failed',
                  'failure_captured': captured}
        code = 1
    print(json.dumps(report, sort_keys=True, separators=(',', ':')))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
