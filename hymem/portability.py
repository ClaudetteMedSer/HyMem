"""Memory export / import (improv item G).

Emits the canonical HyMem state as JSON Lines — one record per line, each
``{"type": <kind>, "record": {...}}`` — preceded by a ``_meta`` header. The
format is stable and human-inspectable, suitable for backups, project-to-
project migration, and feeding external tooling. Stays in-process; no service
layer.

Import is additive and idempotent: rows are INSERT-OR-IGNOREd in
dependency order (sessions before their chunks/episodes/procedures), so
re-importing the same file is a no-op and importing into a populated DB merges
rather than clobbers. Best used against a fresh database. The autoincrement
ids of knowledge_graph / profile_entries are dropped on import so they don't
collide with rows already in the target — those tables dedupe on their natural
unique keys ((s,p,o) and text).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from hymem.core import db as core_db

log = logging.getLogger("hymem.portability")

EXPORT_VERSION = 1

# (kind, table, columns) in export order. Import re-orders so a row's
# referenced session always lands first.
_EXPORT_SPEC: list[tuple[str, str, list[str]]] = [
    ("session", "sessions", ["id", "started_at", "ended_at", "summary"]),
    ("chunk", "chunks", [
        "id", "session_id", "start_message_id", "end_message_id",
        "salience_reason", "text", "created_at",
    ]),
    ("episode", "episodes", [
        "id", "session_id", "title", "summary", "participants",
        "start_message_id", "end_message_id", "outcome", "key_entities",
        "created_at",
    ]),
    ("procedure", "procedures", [
        "id", "session_id", "name", "description", "steps", "triggers",
        "entities_involved", "confidence", "status", "created_at",
    ]),
    ("edge", "knowledge_graph", [
        "id", "subject_canonical", "predicate", "object_canonical",
        "pos_evidence", "neg_evidence", "first_seen", "last_seen",
        "last_reinforced", "status", "derived",
    ]),
    ("profile_entry", "profile_entries", [
        "id", "kind", "text", "pos_evidence", "neg_evidence",
        "first_seen", "last_updated",
    ]),
]

_TABLE_BY_KIND = {kind: table for kind, table, _ in _EXPORT_SPEC}
# Sessions must import before rows that FK-reference them.
_IMPORT_ORDER = ["session", "chunk", "episode", "procedure", "edge", "profile_entry"]
# Autoincrement-id tables: drop the id on import so it can't collide with rows
# already present; they dedupe on their natural unique key instead.
_DROP_ID_ON_IMPORT = {"edge", "profile_entry"}


def export_jsonl(conn, path: str | Path) -> dict[str, int]:
    """Write the canonical state to `path` as JSON Lines. Returns per-kind
    row counts."""
    path = Path(path)
    counts: dict[str, int] = {}
    with path.open("w", encoding="utf-8") as f:
        meta = {
            "type": "_meta",
            "format": "hymem-jsonl",
            "version": EXPORT_VERSION,
            "schema_version": core_db.schema_version(conn),
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for kind, table, cols in _EXPORT_SPEC:
            rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
            for r in rows:
                record = {c: r[c] for c in cols}
                f.write(
                    json.dumps({"type": kind, "record": record}, ensure_ascii=False)
                    + "\n"
                )
            counts[kind] = len(rows)
    log.info("export.done path=%s counts=%s", path, counts)
    return counts


def import_jsonl(conn, path: str | Path) -> dict[str, int]:
    """Load a JSON Lines export into the DB (additive, INSERT-OR-IGNORE).
    Returns per-kind counts of rows actually inserted. Caller is responsible
    for invalidating query caches afterwards."""
    path = Path(path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind in _TABLE_BY_KIND:
                grouped[kind].append(obj["record"])

    inserted: dict[str, int] = {}
    with core_db.transaction(conn):
        for kind in _IMPORT_ORDER:
            table = _TABLE_BY_KIND[kind]
            drop_id = kind in _DROP_ID_ON_IMPORT
            n = 0
            for record in grouped.get(kind, []):
                cols = [c for c in record if not (drop_id and c == "id")]
                if not cols:
                    continue
                placeholders = ", ".join("?" * len(cols))
                cur = conn.execute(
                    f"INSERT OR IGNORE INTO {table}({', '.join(cols)}) "
                    f"VALUES ({placeholders})",
                    [record[c] for c in cols],
                )
                n += cur.rowcount
            inserted[kind] = n
    log.info("import.done path=%s inserted=%s", path, inserted)
    return inserted
