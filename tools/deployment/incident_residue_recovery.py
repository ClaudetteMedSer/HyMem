#!/usr/bin/env python3
"""Clone-only quarantine of the two uncited Sep10 ownership-rollback uploads.

Never edits its input, binds sessions, rewrites proofs, replays messages, clears
failure ledgers, or installs a production database. A complete original SQLite
backup and native typed-row receipt remain private and recoverable. Adoption of
the checked clone requires a separately reviewed offline deployment.

The existing orphan rehearsal supplies bounded native encoding, SQLite table
classification and actual FTS posting inspection. It is loaded by sibling path,
not through an ambient tools package. Both files must accompany deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys

from hymem.core import db
from hymem.coverage_health import scan_coverage_health
from hymem.deadline import MonotonicDeadline
from hymem.dreaming.lossless import (
    COVERAGE_VALIDATION_COLUMNS, COVERAGE_VALIDATION_JOINS,
    validate_message_coverage_row,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION

_spec = importlib.util.spec_from_file_location(
    "_incident_native_archive", Path(__file__).with_name("rehearse_orphan_quarantine.py"))
if _spec is None or _spec.loader is None:
    raise RuntimeError("archive_helper_unavailable")
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)
Refused = native.Refused
REVIEWED_SCHEMA = 61
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_GUARD = "message_lossless_stream_delete_guard"


@dataclass(frozen=True)
class Pins:
    message_ids: tuple[int, int]
    content_sha256: tuple[str, str]
    session_sha256: str
    membership_sha256: str
    store_sha256: str
    schema_sha256: str


def _hash(value):
    return hashlib.sha256(native._encoded(value)).hexdigest()


def _validate_pins(pins):
    if (len(pins.message_ids) != 2 or len(set(pins.message_ids)) != 2
            or any(type(mid) is not int or mid <= 0 for mid in pins.message_ids)
            or tuple(sorted(pins.message_ids)) != tuple(pins.message_ids)
            or len(pins.content_sha256) != 2
            or any(not isinstance(pin, str) or not _HEX.fullmatch(pin) for pin in (
                *pins.content_sha256, pins.session_sha256, pins.membership_sha256,
                pins.store_sha256, pins.schema_sha256))):
        raise Refused("invalid_pins")


def _regular(path):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise Refused("absolute_nonsymlink_path_required")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise Refused("single_link_regular_file_required")


def _schema_hash(conn):
    return _hash([native._encoded(row).decode("ascii") for row in native._schema(conn)])


def _snapshot(conn, deadline, target=None):
    """Hash every logical row/posting, optionally applying the exact expected delta."""
    result = {}
    vector_storage = native._vec_storage_tables(conn)
    mids = set(target["message_ids"]) if target else set()
    cids = set(target["chunk_ids"]) if target else set()
    crows = set(target["chunk_rowids"]) if target else set()

    def skip(table, row):
        if table in ("messages", "message_embeddings", "message_retention_coverage"):
            return row.get("id" if table == "messages" else "message_id") in mids
        if table == "chunks":
            return row.get("id") in cids
        if table == "session_peers":
            return bool(target and row.get("session_id") == target["session_id"])
        if table in ("vec_messages", "messages_fts", "messages_fts_docsize"):
            return row.get("doc", row.get("rowid")) in mids
        if table in ("chunks_fts", "message_coverage_fts", "chunks_fts_docsize", "message_coverage_fts_docsize"):
            return row.get("doc", row.get("rowid")) in crows
        return False

    def stream(table, columns, cursor):
        count, digest = 0, hashlib.sha256(native._encoded(columns))
        for row in native._bounded(cursor, deadline):
            values = dict(zip(columns, row))
            if target and skip(table, values):
                continue
            if target and table == "sessions" and values.get("id") == target["session_id"]:
                values["coverage_message_id"] = target["remaining_frontier"]
            digest.update(b"\n" + native._encoded([values[column] for column in columns]))
            count += 1
        result[table] = (count, digest.hexdigest())

    for table, kind, without in native._tables(conn):
        if table in vector_storage:
            continue
        definition = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0] or ""
        if kind == "virtual" and "using fts5" in definition.lower():
            stream(table, ["term", "doc", "col", "offset"], native._fts_vocab(conn, table))
            docsize = table + "_docsize"
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (docsize,)).fetchone():
                columns, cursor = native._rows(conn, docsize)
                stream(docsize, columns, cursor)
        else:
            columns, cursor = native._rows(conn, table, without)
            stream(table, columns, cursor)
    conn.execute("DROP TABLE IF EXISTS temp.quarantine_vocab")
    return result


def _snapshot_hash(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _open_ro(path):
    _regular(path)
    conn = native._readonly(path)
    try:
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, native.MAX_CELL_BYTES)
        native._load_vectors(conn)
        conn.execute("BEGIN")
        return conn
    except BaseException:
        conn.close()
        raise


def discover_pins(source, message_ids):
    """Read-only hash discovery, without exposing source/session/peer values.

    Pin discovery is not approval. Review the incident identities and content
    hashes independently before handing these pins to recover().
    """
    source = Path(source)
    conn = _open_ro(source)
    try:
        if len(message_ids) != 2 or any(type(mid) is not int or mid <= 0 for mid in message_ids):
            raise Refused("exactly_two_message_ids_required")
        rows = [conn.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone() for mid in message_ids]
        if any(row is None for row in rows) or rows[0]["session_id"] != rows[1]["session_id"]:
            raise Refused("incident_targets_not_present")
        session = rows[0]["session_id"]
        members = conn.execute("SELECT * FROM session_peers WHERE session_id=? ORDER BY workspace_id,peer_id", (session,)).fetchall()
        pins = Pins(tuple(message_ids), tuple(hashlib.sha256(row["content"].encode()).hexdigest() for row in rows),
                    hashlib.sha256(session.encode()).hexdigest(),
                    _hash([native._encoded(row).decode("ascii") for row in members]),
                    _snapshot_hash(_snapshot(conn, MonotonicDeadline.after(120))), _schema_hash(conn))
        _validate_pins(pins)
        return pins
    finally:
        conn.rollback()
        conn.close()


def _audit(conn, target_ids=(), deadline=None):
    """Validate every retained proof; only exact ownership-only targets may fail."""
    invalid = []
    joins = COVERAGE_VALIDATION_JOINS.replace("JOIN chunks c", "LEFT JOIN chunks c").replace(
        "     AND raw.session_id = mc.source_session_id", "")
    for row in conn.execute(f"SELECT {COVERAGE_VALIDATION_COLUMNS} FROM message_retention_coverage mc {joins}"):
        if deadline is not None:
            deadline.check()
        try:
            validate_message_coverage_row(row)
        except (RuntimeError, ValueError, TypeError, KeyError, IndexError):
            if row["message_id"] not in target_ids:
                raise Refused("unrelated_invalid_coverage") from None
            # A validation-only substitution proves that the original bytes,
            # peer membership and proof all match; it never updates SQLite.
            corrected = dict(row)
            corrected["bound_workspace_id"] = row["source_workspace_id"]
            try:
                validate_message_coverage_row(corrected)
            except Exception:
                raise Refused("target_not_ownership_only") from None
            invalid.append(row["message_id"])
    if sorted(invalid) != sorted(target_ids):
        raise Refused("unexpected_invalid_coverage_set")
    if conn.execute("SELECT count(*) FROM coverage_integrity_failures").fetchone()[0]:
        raise Refused("recorded_coverage_failure_requires_separate_review")
    if conn.execute("SELECT count(*) FROM sessions WHERE coverage_message_id IS NOT NULL AND "
                    "(typeof(coverage_message_id) != 'integer' OR coverage_message_id < 0)").fetchone()[0]:
        raise Refused("invalid_coverage_frontier")
    if conn.execute("SELECT 1 FROM messages m JOIN sessions s ON s.id=m.session_id "
                    "WHERE m.id <= s.coverage_message_id AND NOT EXISTS ("
                    "SELECT 1 FROM message_retention_coverage c WHERE c.message_id=m.id "
                    "AND c.source_session_id=m.session_id AND c.coverage_version=?) LIMIT 1",
                    (LOSSLESS_COVERAGE_VERSION,)).fetchone():
        raise Refused("missing_coverage_proof")


def _json_references(value, mids, cids, depth=0):
    if depth > 64:
        raise Refused("dependency_json_depth_limit")
    if isinstance(value, dict):
        return any(key in cids or _json_references(item, mids, cids, depth + 1) for key, item in value.items())
    if isinstance(value, list):
        return any(_json_references(item, mids, cids, depth + 1) for item in value)
    return (type(value) is int and value in mids) or (isinstance(value, str) and value in cids)


def _preflight(conn, pins, deadline):
    deadline.check()
    conn.set_progress_handler(lambda: int(deadline.expired), 10000)
    if db.EXPECTED_SCHEMA_VERSION != REVIEWED_SCHEMA or db.schema_version(conn) != REVIEWED_SCHEMA:
        raise Refused("reviewed_schema_61_required")
    if _schema_hash(conn) != pins.schema_sha256 or _snapshot_hash(_snapshot(conn, deadline)) != pins.store_sha256:
        raise Refused("snapshot_pin_mismatch")
    if list(conn.execute("PRAGMA foreign_key_check")) or conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise Refused("preexisting_sqlite_integrity_failure")
    messages = [dict(conn.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone() or {}) for mid in pins.message_ids]
    if any(not row for row in messages):
        raise Refused("target_absent")
    session_id = messages[0]["session_id"]
    session = dict(conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone() or {})
    if (not session or session["source_workspace_id"] is not None
            or hashlib.sha256(session_id.encode()).hexdigest() != pins.session_sha256
            or any(row["session_id"] != session_id or not row["source_peer_id"] or not row["source_workspace_id"] for row in messages)
            or tuple(hashlib.sha256(row["content"].encode()).hexdigest() for row in messages) != tuple(pins.content_sha256)):
        raise Refused("incident_source_pin_or_shape_mismatch")
    placeholders = ",".join("?" for _ in pins.message_ids)
    coverage = conn.execute(f"SELECT * FROM message_retention_coverage WHERE message_id IN ({placeholders}) ORDER BY message_id", pins.message_ids).fetchall()
    if len(coverage) != 2 or any(row["coverage_version"] != LOSSLESS_COVERAGE_VERSION for row in coverage):
        raise Refused("exactly_two_ordered_proofs_required")
    chunks = conn.execute(f"SELECT rowid,* FROM chunks WHERE id IN (SELECT chunk_id FROM message_retention_coverage WHERE message_id IN ({placeholders})) ORDER BY rowid", pins.message_ids).fetchall()
    if len(chunks) != 2 or any(row["source_manifest_version"] is not None or row["source_manifest_count"] is not None for row in chunks):
        raise Refused("unexpected_target_chunks")
    members = conn.execute("SELECT * FROM session_peers WHERE session_id=? ORDER BY workspace_id,peer_id", (session_id,)).fetchall()
    if not 1 <= len(members) <= 2 or _hash([native._encoded(row).decode("ascii") for row in members]) != pins.membership_sha256:
        raise Refused("membership_pin_or_count_mismatch")
    remaining = conn.execute(f"SELECT max(message_id) FROM message_retention_coverage WHERE source_session_id=? AND message_id NOT IN ({placeholders}) AND coverage_version=?", (session_id, *pins.message_ids, LOSSLESS_COVERAGE_VERSION)).fetchone()[0]
    if remaining is None or remaining >= min(pins.message_ids) or session["coverage_message_id"] != max(pins.message_ids):
        raise Refused("incident_must_be_unconsumed_stream_tail")
    for column, value in session.items():
        if "message_id" in column and column != "coverage_message_id" and value is not None:
            if type(value) is not int or value > remaining:
                raise Refused("consumer_frontier_reaches_incident")
    if conn.execute(f"SELECT 1 FROM messages WHERE session_id=? AND id NOT IN ({placeholders}) AND (source_peer_id IS NOT NULL OR source_workspace_id IS NOT NULL OR id>?)", (session_id, *pins.message_ids, remaining)).fetchone():
        raise Refused("remaining_attributed_or_new_source")
    if conn.execute(f"SELECT 1 FROM message_retention_coverage WHERE source_session_id=? AND message_id NOT IN ({placeholders}) AND (source_peer_id IS NOT NULL OR source_workspace_id IS NOT NULL)", (session_id, *pins.message_ids)).fetchone():
        raise Refused("remaining_attributed_proof")
    target = {"message_ids": list(pins.message_ids), "session_id": session_id,
              "chunk_ids": [row["id"] for row in chunks], "chunk_rowids": [row["rowid"] for row in chunks],
              "remaining_frontier": remaining, "membership_count": len(members)}
    mids, cids = set(pins.message_ids), set(target["chunk_ids"])
    vector_storage = native._vec_storage_tables(conn)
    for table, kind, without in native._tables(conn):
        if kind != "table" or table in vector_storage:
            continue
        columns, cursor = native._rows(conn, table, without)
        fk_fields = {row[3] for row in conn.execute(f"PRAGMA foreign_key_list({native._q(table)})") if row[2] in ("messages", "chunks", "message_retention_coverage")}
        for row in native._bounded(cursor, deadline):
            values = dict(zip(columns, row))
            if ((table == "messages" and values.get("id") in mids)
                    or (table == "chunks" and values.get("id") in cids)
                    or (table in ("message_retention_coverage", "message_embeddings") and values.get("message_id") in mids)):
                continue
            for column, value in values.items():
                if table == "sessions" and values.get("id") == session_id and column == "coverage_message_id":
                    continue
                if (isinstance(value, str) and value in cids) or (
                        type(value) is int and value in mids and ("message_id" in column or column in fk_fields)):
                    raise Refused("dependent_source_reference")
                if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
                    native._native(value)
                    try:
                        parsed = json.loads(value)
                    except (ValueError, RecursionError):
                        if any(part in column for part in ("source", "chunk", "message", "input")):
                            raise Refused("unreadable_dependency_payload") from None
                    else:
                        if _json_references(parsed, mids, cids):
                            raise Refused("dependent_json_reference")
            if (table != "chunks" or values.get("id") not in cids) and values.get("session_id") == session_id:
                start, end = values.get("start_message_id"), values.get("end_message_id")
                if type(start) is int and type(end) is int and start <= max(mids) and end >= min(mids):
                    raise Refused("dependent_source_range")
    _audit(conn, pins.message_ids, deadline)
    guard = conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (_GUARD,)).fetchone()
    if guard is None:
        raise Refused("ordered_stream_guard_missing")
    target["guard_sql"] = guard[0]
    return target


def _write_json(path, value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != payload:
        raise Refused("archive_receipt_roundtrip_failed")


def _sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _capture_rows(conn, target, deadline):
    captured = {}
    for table in ("sessions", "messages", "chunks", "message_retention_coverage", "message_embeddings", "session_peers", "vec_messages"):
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            continue
        columns, cursor = native._rows(conn, table)
        rows = []
        for row in native._bounded(cursor, deadline):
            values = dict(zip(columns, row))
            key = {"sessions": "id", "messages": "id", "chunks": "id", "message_retention_coverage": "message_id", "message_embeddings": "message_id", "session_peers": "session_id", "vec_messages": "rowid"}[table]
            ids = ([target["session_id"]] if table in ("sessions", "session_peers") else target["chunk_ids"] if table == "chunks" else target["message_ids"])
            if values[key] in ids:
                rows.append([native._native(value) for value in row])
        captured[table] = {"columns": columns, "rows": rows}
    return captured


def recover(source, bundle, pins, *, _checkpoint=lambda stage: None):
    """Make a checked recovery clone; never alter or replace source.

    A ready.json receipt is the only success marker. A failure retains the
    private backup and any incomplete clone for inspection, never auto-cleans.
    _checkpoint is solely a fault-injection seam for rollback tests.
    """
    _validate_pins(pins)
    source, bundle = Path(source), Path(bundle)
    _regular(source)
    if (not bundle.is_absolute() or bundle.parent.resolve(strict=True) != bundle.parent
            or bundle == source or bundle.is_symlink()):
        raise Refused("private_absolute_bundle_required")
    deadline = MonotonicDeadline.after(300)
    if bundle.exists():
        if not bundle.is_dir() or bundle.stat().st_mode & 0o077:
            raise Refused("private_bundle_permissions_required")
        _regular(bundle / "ready.json")
        receipt = json.loads((bundle / "ready.json").read_text())
        if (receipt.get("version") != "incident-residue-recovery-v1"
                or receipt.get("pins") != json.loads(json.dumps(asdict(pins)))
                or receipt.get("before") != pins.store_sha256):
            raise Refused("existing_bundle_pin_mismatch")
        _regular(bundle / "rows.json")
        if hashlib.sha256((bundle / "rows.json").read_bytes()).hexdigest() != receipt.get("rows_sha256"):
            raise Refused("existing_archive_receipt_changed")
        for path, allowed in ((source, {receipt["before"], receipt["after"]}),
                              (bundle / "original.sqlite", {receipt["before"]}),
                              (bundle / "repaired.sqlite", {receipt["after"]})):
            conn = _open_ro(path)
            try:
                if _snapshot_hash(_snapshot(conn, deadline)) not in allowed or _schema_hash(conn) != pins.schema_sha256:
                    raise Refused("existing_bundle_changed")
            finally:
                conn.rollback()
                conn.close()
        return {"status": "already_verified", "quarantined_messages": 2}
    source_conn = _open_ro(source)
    try:
        if source_conn.execute("PRAGMA page_count").fetchone()[0] * source_conn.execute("PRAGMA page_size").fetchone()[0] > native.MAX_DATABASE_BYTES:
            raise Refused("database_size_limit")
        _preflight(source_conn, pins, deadline)
        os.mkdir(bundle, 0o700)
        _sync(bundle.parent)
        original, repaired = bundle / "original.sqlite", bundle / "repaired.sqlite"
        native._backup(source_conn, original, deadline)
        _sync(original)
        _sync(bundle)
    finally:
        source_conn.rollback()
        source_conn.close()
    archive = _open_ro(original)
    try:
        target = _preflight(archive, pins, deadline)
        expected = _snapshot(archive, deadline, target)
        _write_json(bundle / "rows.json", {"version": "incident-native-quarantine-v1", "pins": asdict(pins), "rows": _capture_rows(archive, target, deadline)})
        native._backup(archive, repaired, deadline)
    finally:
        archive.rollback()
        archive.close()
    _checkpoint("archived")
    conn = db.connect(repaired)
    try:
        native._load_vectors(conn)
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, native.MAX_CELL_BYTES)
        conn.set_progress_handler(lambda: int(deadline.expired), 10000)
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            _preflight(conn, pins, deadline)
            conn.execute(f"DROP TRIGGER {native._q(_GUARD)}")
            _checkpoint("guard_suspended")
            for mid in pins.message_ids:
                if db.has_vec_table(conn, "vec_messages"):
                    conn.execute("DELETE FROM vec_messages WHERE rowid=?", (mid,))
                conn.execute("DELETE FROM message_embeddings WHERE message_id=?", (mid,))
                conn.execute("DELETE FROM message_retention_coverage WHERE message_id=?", (mid,))
                conn.execute("DELETE FROM messages WHERE id=?", (mid,))
            for cid in target["chunk_ids"]:
                conn.execute("DELETE FROM chunks WHERE id=?", (cid,))
            conn.execute("DELETE FROM session_peers WHERE session_id=?", (target["session_id"],))
            conn.execute("UPDATE sessions SET coverage_message_id=? WHERE id=?", (target["remaining_frontier"], target["session_id"]))
            conn.execute(target["guard_sql"])
            _checkpoint("rows_quarantined")
            _audit(conn, deadline=deadline)
            if _schema_hash(conn) != pins.schema_sha256 or _snapshot(conn, deadline) != expected:
                raise Refused("unexpected_repair_delta")
            if list(conn.execute("PRAGMA foreign_key_check")) or conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise Refused("repaired_sqlite_integrity_failure")
            _checkpoint("before_commit")
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
    finally:
        conn.close()
    # Ordinary application startup must neither fail nor quietly repair more
    # rows or guards. It runs only on our private clone, never the source.
    reopened = db.connect(repaired)
    try:
        native._load_vectors(reopened)
        db.initialize(reopened)
        if _schema_hash(reopened) != pins.schema_sha256 or _snapshot(reopened, deadline) != expected:
            raise Refused("reopen_changed_repaired_clone")
    finally:
        reopened.close()
    health = scan_coverage_health(repaired)
    if health.status != "valid" or not health.complete:
        raise Refused("repaired_coverage_audit_not_green")
    _checkpoint("reopened")
    _sync(repaired)
    _write_json(bundle / "ready.json", {"version": "incident-residue-recovery-v1", "pins": asdict(pins),
                                        "before": pins.store_sha256, "after": _snapshot_hash(expected),
                                        "rows_sha256": hashlib.sha256((bundle / "rows.json").read_bytes()).hexdigest(),
                                        "quarantined_messages": 2, "removed_stale_memberships": target["membership_count"]})
    _sync(bundle)
    return {"status": "verified_clone", "quarantined_messages": 2, "production_writes": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--discover", nargs=2, type=int, metavar=("MESSAGE_1", "MESSAGE_2"))
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        if args.discover and not args.pins and not args.bundle:
            print(json.dumps(asdict(discover_pins(args.source, args.discover)), sort_keys=True))
        elif args.pins and args.bundle and not args.discover:
            _regular(args.pins)
            print(json.dumps(recover(args.source, args.bundle, Pins(**json.loads(args.pins.read_text()))), sort_keys=True))
        else:
            raise Refused("choose_discovery_or_clone_recovery")
    except Refused as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}))
        return 1
    except Exception:
        print(json.dumps({"status": "refused", "reason": "recovery_unavailable"}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
