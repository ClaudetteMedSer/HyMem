#!/usr/bin/env python3
"""Clone-only rehearsal for the specifically reviewed orphan at rowid 1058.

There is deliberately no production apply mode or caller-selected write DB.
The reference commits the complete native chunk row, not merely its rowid.
Archive roundtrip and full-baseline backup restoration are separate proofs:
the latter retains the original known FK fault, without disabling foreign keys
or pretending an archive-only reinsert into the runtime schema was tested.
Limits: 512 MiB source, two million rows per audited table/posting stream,
16 MiB per captured cell, 64 derived entity links, cooperative timeout. Fresh
private artifacts are retained on success or failure; no cleanup is automatic.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import sys

from hymem.core import db
from hymem.deadline import DeadlineExceeded, MonotonicDeadline, use_deadline

TARGET_ROWID = 1058
MAX_ROWS = 2_000_000
MAX_CELL_BYTES = 16 * 1024 * 1024
MAX_DATABASE_BYTES = 512 * 1024 * 1024
_REFERENCE = re.compile(r"[a-f0-9]{64}\Z")


class Refused(RuntimeError):
    """Only code-owned constant reasons cross the command boundary."""


def _q(name):
    return '"' + name.replace('"', '""') + '"'


def _native(value):
    if value is None:
        return ["null"]
    if type(value) is int:
        return ["integer", str(value)]
    if type(value) is float:
        return ["real", value.hex()]
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_CELL_BYTES:
            raise Refused("cell_limit")
        return ["text", value]
    if isinstance(value, bytes):
        if len(value) > MAX_CELL_BYTES:
            raise Refused("cell_limit")
        return ["blob", base64.b64encode(value).decode("ascii")]
    raise Refused("unsupported_native_value")


def _encoded(values):
    return json.dumps([_native(value) for value in values], ensure_ascii=True,
                      separators=(",", ":")).encode("ascii")


def reference_fingerprint(conn):
    """Read-only complete-row fingerprint; callers keep it in private files."""
    cursor = conn.execute("SELECT rowid AS quarantine_rowid,* FROM chunks WHERE rowid=?", (TARGET_ROWID,))
    row = cursor.fetchone()
    if row is None:
        raise Refused("target_absent")
    names = [column[0] for column in cursor.description]
    return hashlib.sha256(_encoded(names) + b"\n" + _encoded(row)).hexdigest()


def _new_file(path):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)


def _readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _backup(source, destination, deadline):
    _new_file(destination)
    clone = sqlite3.connect(destination.resolve().as_uri() + "?mode=rw", uri=True, isolation_level=None)
    try:
        source.backup(clone, pages=128, progress=lambda *_: deadline.check())
    finally:
        clone.close()


def _load_vectors(conn):
    if any("using vec0" in (row[0] or "").lower() for row in conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table'",
    )):
        # The core helper may log transport-loader exception text. The CLI
        # suppresses logging globally and returns only constant failure codes.
        if not db._load_vec_extension(conn):
            raise Refused("vector_extension_unavailable")
        conn.enable_load_extension(False)


def _tables(conn):
    # SQLite's own classification excludes FTS/vec implementation shadows,
    # not a guessed prefix that could hide a physical dependency table.
    return sorted((row[1], row[2], bool(row[4])) for row in conn.execute("PRAGMA main.table_list")
                  if row[1] != "sqlite_schema" and row[2] in ("table", "virtual"))


def _rows(conn, table, without_rowid=False):
    names = [row[1] for row in conn.execute(f"PRAGMA main.table_info({_q(table)})")]
    # vec0 exposes its rowid as a declared column; avoid projecting it twice.
    columns = names if without_rowid or "rowid" in names else ["rowid", *names]
    order = ",".join(_q(name) for name in (names if without_rowid else ["rowid"]))
    cursor = conn.execute(f"SELECT {','.join(_q(name) for name in columns)} FROM {_q(table)} ORDER BY {order}")
    return columns, cursor


def _bounded(cursor, deadline):
    for count, row in enumerate(cursor, 1):
        deadline.check()
        if count > MAX_ROWS:
            raise Refused("row_limit")
        yield row


def _schema(conn):
    return tuple(tuple(row) for row in conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name",
    ))


def _vec_storage_tables(conn):
    """Recognize only the loaded vec0 module's narrow maintained layout.

    Some sqlite-vec versions do not mark their storage tables as SQLite
    'shadow' tables. Their packed allocation pages change when one vector is
    deleted; the virtual rowid/vector projection is the logical invariant.
    Unknown tables sharing a prefix are NOT excluded from inspection.
    """
    result = set()
    layouts = {
        "_chunks": ("chunk_id", "size", "validity", "rowids"),
        "_rowids": ("rowid", "id", "chunk_id", "chunk_offset"),
        "_vector_chunks00": ("rowid", "vectors"),
    }
    for table, kind, _without in _tables(conn):
        if kind != "virtual":
            continue
        definition = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0] or ""
        if "using vec0" not in definition.lower():
            continue
        if table not in ("vec_chunks", "vec_messages", "vec_edges", "vec_episodes", "vec_facts"):
            raise Refused("unreviewed_vector_table")
        for suffix, columns in layouts.items():
            storage = table + suffix
            actual = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({_q(storage)})"))
            if actual != columns or conn.execute(f"PRAGMA foreign_key_list({_q(storage)})").fetchall():
                raise Refused("unreviewed_vector_storage")
            result.add(storage)
    return result


def _fts_vocab(conn, table):
    # Temporary virtual views expose actual postings, unlike external-content
    # SELECT which can hide stale index entries by reading the content table.
    conn.execute("DROP TABLE IF EXISTS temp.quarantine_vocab")
    conn.execute(f"CREATE VIRTUAL TABLE temp.quarantine_vocab USING fts5vocab(main,{_q(table)},instance)")
    return conn.execute("SELECT term,doc,col,offset FROM temp.quarantine_vocab ORDER BY term,doc,col,offset")


def _snapshot(conn, deadline, *, omit_chunk=None):
    result = {}
    vector_storage = _vec_storage_tables(conn)
    for table, kind, without in _tables(conn):
        if table in vector_storage:
            continue
        definition = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0] or ""
        if kind == "virtual" and "using fts5" in definition.lower():
            columns, cursor = ["term", "doc", "col", "offset"], _fts_vocab(conn, table)
            skip_index = 1 if table == "chunks_fts" else None
            skip_value = TARGET_ROWID
        else:
            columns, cursor = _rows(conn, table, without)
            field = {"chunks": "rowid", "chunk_embeddings": "chunk_id", "entity_mentions": "chunk_id", "vec_chunks": "rowid"}.get(table)
            skip_index = columns.index(field) if field in columns else None
            skip_value = omit_chunk if table in ("chunk_embeddings", "entity_mentions") else TARGET_ROWID
        count, digest = 0, hashlib.sha256(_encoded(columns))
        for row in _bounded(cursor, deadline):
            if omit_chunk is not None and skip_index is not None and row[skip_index] == skip_value:
                continue
            digest.update(b"\n" + _encoded(row))
            count += 1
        result[table] = (count, digest.hexdigest())
        if kind == "virtual" and "using fts5" in definition.lower():
            # Include empty-document membership too; it has no token postings.
            docsize = table + "_docsize"
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (docsize,)).fetchone():
                columns, cursor = _rows(conn, docsize)
                count, digest = 0, hashlib.sha256(_encoded(columns))
                for row in _bounded(cursor, deadline):
                    if omit_chunk is not None and table == "chunks_fts" and row[0] == TARGET_ROWID:
                        continue
                    digest.update(b"\n" + _encoded(row))
                    count += 1
                result[docsize] = (count, digest.hexdigest())
    conn.execute("DROP TABLE IF EXISTS temp.quarantine_vocab")
    return result


def _json_mentions(value, identities, depth=0):
    if depth > 64:
        raise Refused("json_depth_limit")
    if isinstance(value, str):
        return value in identities
    if isinstance(value, list):
        return any(_json_mentions(item, identities, depth + 1) for item in value)
    if isinstance(value, dict):
        return any(key in identities or _json_mentions(item, identities, depth + 1) for key, item in value.items())
    return False


def _preflight(conn, reference, deadline):
    if db.schema_version(conn) != 59 or db.EXPECTED_SCHEMA_VERSION != 59:
        raise Refused("reviewed_schema_required")
    if reference_fingerprint(conn) != reference:
        raise Refused("reference_mismatch")
    target = conn.execute("SELECT rowid,* FROM chunks WHERE rowid=?", (TARGET_ROWID,)).fetchone()
    chunk, session = target["id"], target["session_id"]
    if (not isinstance(chunk, str) or not chunk or not isinstance(session, str) or not session
            or target["chunk_kind"] != "extraction" or target["source_manifest_version"] is not None
            or target["source_manifest_count"] is not None):
        raise Refused("target_not_unproven_extraction")
    if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session,)).fetchone():
        raise Refused("parent_present")
    faults = [tuple(row) for row in conn.execute("PRAGMA foreign_key_check")]
    chunk_fk = [row for row in conn.execute("PRAGMA foreign_key_list(chunks)")
                if row[2] == "sessions" and row[3] == "session_id" and row[4] == "id"]
    if len(chunk_fk) != 1 or faults != [("chunks", TARGET_ROWID, "sessions", chunk_fk[0][0])]:
        raise Refused("unexpected_foreign_key_faults")
    if conn.execute("SELECT 1 FROM messages WHERE session_id=? OR id BETWEEN ? AND ? LIMIT 1",
                    (session, target["start_message_id"], target["end_message_id"])).fetchone():
        raise Refused("raw_source_present")
    vectors = conn.execute("SELECT rowid,* FROM chunk_embeddings WHERE chunk_id=?", (chunk,)).fetchall()
    if len(vectors) != 1:
        raise Refused("expected_single_durable_vector")
    if conn.execute("SELECT COUNT(*) FROM entity_mentions WHERE chunk_id=?", (chunk,)).fetchone()[0] > 64:
        raise Refused("entity_mentions_limit")
    identities = {chunk, session}
    for table, kind, without in _tables(conn):
        if kind != "table":
            continue
        columns, cursor = _rows(conn, table, without)
        # Inspect declared chunk references AND physical identity/JSON payloads,
        # including material/aggregate dependencies absent from foreign keys.
        foreign_columns = {row[3] for row in conn.execute(f"PRAGMA foreign_key_list({_q(table)})")
                           if row[2] in ("chunks", "sessions")}
        for row in _bounded(cursor, deadline):
            values = dict(zip(columns, row))
            if table == "chunks" and values.get("rowid") == TARGET_ROWID:
                continue
            if table in ("chunk_embeddings", "entity_mentions") and values.get("chunk_id") == chunk:
                continue
            for column, value in values.items():
                if isinstance(value, str):
                    if value in identities:
                        raise Refused("dependent_identity_present")
                    if value.lstrip().startswith(("[", "{")):
                        _native(value)  # bound JSON parsing before allocation
                        try:
                            parsed = json.loads(value)
                        except (ValueError, RecursionError):
                            if any(part in column.lower() for part in ("source", "chunk", "session", "member", "input")):
                                raise Refused("unreadable_dependency_payload") from None
                        else:
                            if _json_mentions(parsed, identities):
                                raise Refused("dependent_json_present")
                elif column in foreign_columns and value is not None:
                    raise Refused("nontext_dependency_identity")
                elif (type(value) is int and "message_id" in column.lower()
                      and target["start_message_id"] <= value <= target["end_message_id"]):
                    raise Refused("dependent_source_message_present")
    # Explicitly consult the same effective material views used by stock
    # delete triggers, not just their underlying physical tables.
    for view in ("aggregation_visible_episode_sources", "aggregation_enabled_profile_sources", "aggregation_enabled_kg_sources"):
        if conn.execute(f"SELECT 1 FROM {_q(view)} WHERE source_coverage_chunk_id=? LIMIT 1", (chunk,)).fetchone():
            raise Refused("aggregation_dependency_present")
    return target


def _capture(conn, target):
    captured = {}
    for table, field, value in (("chunks", "rowid", TARGET_ROWID),
                                ("chunk_embeddings", "chunk_id", target["id"]),
                                ("entity_mentions", "chunk_id", target["id"]),
                                ("vec_chunks", "rowid", TARGET_ROWID)):
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            continue
        columns, _ = _rows(conn, table)
        rows = conn.execute(f"SELECT {','.join(_q(name) for name in columns)} FROM {_q(table)} WHERE {_q(field)}=? ORDER BY rowid", (value,)).fetchall()
        captured[table] = (columns, [tuple(row) for row in rows])
    return captured


def _archive(path, captured):
    _new_file(path)
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, isolation_level=None)
    try:
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table, (columns, rows) in captured.items():
                # No affinity: TEXT/BLOB/INTEGER/REAL/NULL remain native, and
                # explicit original rowids survive the archive roundtrip.
                conn.execute(f"CREATE TABLE {_q(table)} ({','.join(_q(name) + (' INTEGER PRIMARY KEY' if name == 'rowid' else '') for name in columns)})")
                conn.executemany(f"INSERT INTO {_q(table)} VALUES ({','.join('?' for _ in columns)})", rows)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
    finally:
        conn.close()
    check = _readonly(path)
    try:
        for table, (columns, rows) in captured.items():
            current = [tuple(row) for row in check.execute(f"SELECT {','.join(_q(name) for name in columns)} FROM {_q(table)}")]
            if [_encoded(row) for row in current] != [_encoded(row) for row in rows]:
                raise Refused("archive_roundtrip_failed")
        if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise Refused("archive_integrity_failed")
    finally:
        check.close()
    with path.open("rb") as saved:
        os.fsync(saved.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _integrity(conn):
    if [tuple(row) for row in conn.execute("PRAGMA integrity_check")] != [("ok",)]:
        raise Refused("database_integrity_failed")
    for table, kind, _without in _tables(conn):
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0] or ""
        if kind == "virtual" and "using fts5" in sql.lower():
            # rank=1 is wrong for intentionally selective external-content
            # indexes (coverage must not enter extraction FTS). Validate FTS
            # structure here and exact postings/doc membership via snapshots.
            conn.execute(f"INSERT INTO {_q(table)}({_q(table)}) VALUES ('integrity-check')")


def rehearse(source_path, rehearsal_dir, reference_sha256, *, timeout_seconds=120):
    if not isinstance(reference_sha256, str) or _REFERENCE.fullmatch(reference_sha256) is None:
        raise Refused("invalid_reference")
    deadline = MonotonicDeadline.after(timeout_seconds)
    if timeout_seconds > 3600:
        raise Refused("invalid_timeout")
    destination = Path(rehearsal_dir)
    # No caller can point deletion at an existing store or reused directory.
    os.mkdir(destination, 0o700)
    baseline, working, archive, restored = [destination / name for name in (
        "baseline.sqlite", "working.sqlite", "quarantine.sqlite", "baseline-restored.sqlite",
    )]
    source = _readonly(source_path)
    source.execute("PRAGMA query_only=ON")
    conn = None
    try:
        with use_deadline(deadline):
            if source.execute("PRAGMA page_count").fetchone()[0] * source.execute("PRAGMA page_size").fetchone()[0] > MAX_DATABASE_BYTES:
                raise Refused("database_size_limit")
            _backup(source, baseline, deadline)
            source.close()
            source = None
            base = _readonly(baseline)
            try:
                _backup(base, working, deadline)
            finally:
                base.close()
            # Core UDF/authorization wiring, on the freshly created WORKING
            # clone only. Never initialize, migrate, rebuild, or weaken guards.
            conn = db.connect(working)
            _load_vectors(conn)
            conn.set_progress_handler(lambda: int(deadline.expired), 10000)
            target = _preflight(conn, reference_sha256, deadline)
            schema_before = _schema(conn)
            before = _snapshot(conn, deadline)
            expected = _snapshot(conn, deadline, omit_chunk=target["id"])
            _integrity(conn)
            captured = _capture(conn, target)
            _archive(archive, captured)  # committed+read-back+fsynced FIRST
            # Baseline restoration is a full SQLite backup, deliberately not
            # an archive-only insert of an orphan under relaxed foreign keys.
            base = _readonly(baseline)
            try:
                _backup(base, restored, deadline)
            finally:
                base.close()
            restore_check = _readonly(restored)
            try:
                _load_vectors(restore_check)
                if (_schema(restore_check) != schema_before or _snapshot(restore_check, deadline) != before
                        or reference_fingerprint(restore_check) != reference_sha256):
                    raise Refused("baseline_restore_failed")
                if len(restore_check.execute("PRAGMA foreign_key_check").fetchall()) != 1:
                    raise Refused("baseline_restore_fault_changed")
            finally:
                restore_check.close()
            with db.transaction(conn):
                _preflight(conn, reference_sha256, deadline)
                if _capture(conn, target) != captured:
                    raise Refused("target_changed_before_delete")
                # Only this optional disposable shadow is explicitly removed;
                # stock FK/FTS/source lifecycle triggers perform the rest.
                if "vec_chunks" in captured:
                    conn.execute("DELETE FROM vec_chunks WHERE rowid=?", (TARGET_ROWID,))
                conn.execute("DELETE FROM chunks WHERE rowid=? AND id=?", (TARGET_ROWID, target["id"]))
                if _schema(conn) != schema_before or _snapshot(conn, deadline) != expected:
                    raise Refused("unexpected_logical_delta")
                if conn.execute("PRAGMA foreign_key_check").fetchall():
                    raise Refused("remaining_foreign_key_faults")
                _integrity(conn)
                if conn.execute("SELECT 1 FROM chunk_embeddings WHERE chunk_id=?", (target["id"],)).fetchone():
                    raise Refused("semantic_candidate_remains")
                if conn.execute("SELECT 1 FROM chunks_fts WHERE rowid=?", (TARGET_ROWID,)).fetchone():
                    raise Refused("fts_candidate_remains")
            # Verify committed data, not merely the pre-COMMIT projection.
            if _schema(conn) != schema_before or _snapshot(conn, deadline) != expected:
                raise Refused("postcommit_verification_failed")
            return {
                "status": "verified_clone_only", "quarantined_chunks": 1,
                "quarantined_durable_vectors": 1,
                "quarantined_entity_mentions": len(captured["entity_mentions"][1]),
                "quarantined_shadow_vectors": len(captured.get("vec_chunks", ([], []))[1]),
                "archive_native_roundtrip_verified": True,
                "full_baseline_restore_verified": True,
                "archive_only_runtime_reinsert_tested": False,
                "source_writes": 0, "remaining_foreign_key_faults": 0,
            }
    finally:
        if source is not None:
            source.close()
        if conn is not None:
            conn.set_progress_handler(None, 0)
            conn.close()


class _SafeParser(argparse.ArgumentParser):
    def error(self, _message):
        raise Refused("invalid_arguments")


def main(argv=None):
    logging.disable(logging.CRITICAL)
    parser = _SafeParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--rehearsal-dir", required=True, type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    try:
        args = parser.parse_args(argv)
        result = rehearse(args.source, args.rehearsal_dir, args.reference_sha256,
                          timeout_seconds=args.timeout_seconds)
    except Refused as exc:
        result = {"status": "refused", "reason": str(exc), "source_writes": 0}
    except (KeyboardInterrupt, DeadlineExceeded):
        result = {"status": "cancelled", "reason": "cancelled_or_deadline", "source_writes": 0}
    except Exception:
        result = {"status": "refused", "reason": "rehearsal_failed", "source_writes": 0}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "verified_clone_only" else 1


if __name__ == "__main__":
    sys.exit(main())
