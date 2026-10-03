"""Read-only, snapshot-consistent compatibility audit of stored vector mirrors.

This is not a missing-embedding or source-authority audit. Compatible means
only that a stored row's producer metadata and vector numerics agree with the
verified live space. Archived rows are included; absent rows, content hashes,
source proofs and whether a row is eligible for repair are not established.
The historical general embedding_cache is deliberately outside this audit.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from importlib.resources import files

from hymem.core.vectors import decode_vector


MIRROR_TABLES = (
    "chunk_embeddings", "message_embeddings", "edge_embeddings",
    "episode_embeddings", "narrative_fact_embeddings", "aggregation_node_embeddings",
)
_TYPED_PRODUCER_TABLES = frozenset({
    "episode_embeddings", "aggregation_node_embeddings",
})
_PRODUCER_KEY = re.compile(r"hymem-embedding-producer-v1:[0-9a-f]{64}\Z")
# Names come from shipped code, not the database. Even a hostile custom table
# name cannot leak credentials into the foreign-key diagnostic.
_SAFE_FK_TABLES = frozenset(re.findall(
    r"CREATE TABLE IF NOT EXISTS (\w+)\s*\(",
    files("hymem.core").joinpath("schema.sql").read_text(encoding="utf-8"),
))


@dataclass(frozen=True)
class MirrorHealth:
    table: str
    status: str
    total: int | None
    current_compatible: int | None
    incompatible: int | None
    malformed: int | None
    unverified: int | None
    error_code: str | None = None


@dataclass(frozen=True)
class ForeignKeyHealth:
    status: str
    total: int | None
    by_table: tuple[tuple[str, int], ...] = ()
    error_code: str | None = None


@dataclass(frozen=True)
class EmbeddingHealth:
    live_identity_verified: bool
    tables: tuple[MirrorHealth, ...]
    foreign_keys: ForeignKeyHealth

    @property
    def status(self) -> str:
        if any(table.status == "unavailable" for table in self.tables):
            return "unavailable"
        if any(table.incompatible or table.malformed for table in self.tables):
            return "incompatible"
        return "compatible" if self.live_identity_verified else "unverified"


def _exact_key(value: object) -> bool:
    return type(value) is str and _PRODUCER_KEY.fullmatch(value) is not None


def _positive_dimension(value: object) -> bool:
    return type(value) is int and value > 0


def _valid_vector(encoded: object, dim: int) -> bool:
    if not isinstance(encoded, (str, bytes)):
        return False
    try:
        vector = decode_vector(encoded)
        if type(vector) is not list or len(vector) != dim:
            return False
        if any(type(item) not in (int, float) for item in vector):
            return False
        values = [float(item) for item in vector]
        if not all(math.isfinite(item) for item in values):
            return False
        norm = math.sqrt(sum(item * item for item in values))
        return math.isfinite(norm) and norm > 0.0
    except (AttributeError, TypeError, ValueError, OverflowError, RecursionError):
        return False


def _unavailable(table: str, code: str) -> MirrorHealth:
    # Partial scans never masquerade as exact whole-table counts.
    return MirrorHealth(table, "unavailable", None, None, None, None, None, code)


def _scan_mirror(conn, table, *, live_model, live_dim, verified) -> MirrorHealth:
    cursor = None
    try:
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
        ).fetchone()
        if present is None:
            return _unavailable(table, "missing_table")
        required = {"model", "dim", "vector_json"}
        if table in _TYPED_PRODUCER_TABLES:
            required.add("embedding_producer_key")
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if not required <= columns:
            return _unavailable(table, "missing_columns")
        typed = "embedding_producer_key" if table in _TYPED_PRODUCER_TABLES else "model"
        cursor = conn.execute(f'SELECT model, dim, vector_json, {typed} FROM "{table}"')
        total = current = incompatible = malformed = unverified = 0
        # Stream one row/vector at a time: memory depends on the largest
        # individual vector, never on corpus size. No row IDs/text are loaded.
        for model, dim, vector, typed_key in cursor:
            total += 1
            if (
                not _exact_key(model) or not _positive_dimension(dim)
                or not _exact_key(typed_key) or typed_key != model
                or not _valid_vector(vector, dim)
            ):
                malformed += 1
            elif not verified:
                unverified += 1
            elif model != live_model or dim != live_dim:
                incompatible += 1
            else:
                current += 1
        status = (
            "incompatible" if incompatible or malformed
            else "compatible" if verified else "unverified"
        )
        return MirrorHealth(table, status, total, current, incompatible, malformed, unverified)
    except (sqlite3.Error, UnicodeError):
        return _unavailable(table, "schema_or_read_failure")
    finally:
        if cursor is not None:
            cursor.close()


def _scan_foreign_keys(conn) -> ForeignKeyHealth:
    counts: dict[str, int] = {}
    cursor = None
    try:
        cursor = conn.execute("PRAGMA foreign_key_check")
        for row in cursor:
            name = row[0] if row[0] in _SAFE_FK_TABLES else "other_tables"
            counts[name] = counts.get(name, 0) + 1
        total = sum(counts.values())
        return ForeignKeyHealth(
            "violations" if total else "valid", total, tuple(sorted(counts.items())),
        )
    except (sqlite3.Error, UnicodeError):
        return ForeignKeyHealth("unavailable", None, error_code="foreign_key_check_failed")
    finally:
        if cursor is not None:
            cursor.close()


def scan_embedding_health(
    conn: sqlite3.Connection, *, live_model: str | None, live_dim: int | None,
) -> EmbeddingHealth:
    """Count every stored mirror under one read snapshot, without provider work.

    The caller supplies the exact identity returned by live producer validation;
    a display model label or absent/invalid identity leaves compatibility
    unverified. Counts are mutually exclusive; schema/read errors use unknown
    counts, not healthy zeros. An existing caller-owned transaction is retained.
    Otherwise a deferred read transaction is opened and rolled back after all
    table/FK scans. No migrations, vector rewrites, UDFs or connection close.
    """
    verified = _exact_key(live_model) and _positive_dimension(live_dim)
    owns_snapshot = False
    try:
        owns_snapshot = not conn.in_transaction
        if owns_snapshot:
            conn.execute("BEGIN")
        try:
            tables = tuple(
                _scan_mirror(
                    conn, table, live_model=live_model, live_dim=live_dim, verified=verified,
                ) for table in MIRROR_TABLES
            )
            foreign_keys = _scan_foreign_keys(conn)
        finally:
            if owns_snapshot and conn.in_transaction:
                conn.rollback()
        return EmbeddingHealth(verified, tables, foreign_keys)
    except (sqlite3.Error, UnicodeError):
        return EmbeddingHealth(
            verified, tuple(_unavailable(table, "snapshot_unavailable") for table in MIRROR_TABLES),
            ForeignKeyHealth("unavailable", None, error_code="snapshot_unavailable"),
        )
