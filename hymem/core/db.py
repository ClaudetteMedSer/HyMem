from __future__ import annotations

import contextlib
import contextvars
import functools
import asyncio
import hashlib
import json
import logging
import math
import re
import sqlite3
import struct
import threading
from importlib.resources import files
from pathlib import Path
from typing import Iterator

from hymem.deadline import check_current_deadline
from hymem.core.message_records import (
    encode_message_record,
    message_content_hash,
    message_record_matches_raw_source,
    message_record_matches_source,
    message_record_proof_valid,
)
from hymem.core.graph import live_edge_predicate
from hymem.core.time import (
    normalize_iso_timestamp,
    register_sqlite_time_functions,
    validate_event_clock,
)
from hymem.core.vectors import decode_vector

log = logging.getLogger("hymem.core.db")

EXPECTED_SCHEMA_VERSION = 59
_EVIDENCE_MUTATION_KEYS: contextvars.ContextVar[
    frozenset[tuple[int, int, int]]
] = contextvars.ContextVar("hymem_evidence_mutation_keys", default=frozenset())
_EVIDENCE_HISTORY_KEYS: contextvars.ContextVar[
    frozenset[tuple[int, int, int]]
] = contextvars.ContextVar("hymem_evidence_history_keys", default=frozenset())
_EVIDENCE_DESTRUCTIVE_KEYS: contextvars.ContextVar[
    frozenset[tuple[int, int, int]]
] = contextvars.ContextVar("hymem_evidence_destructive_keys", default=frozenset())
_PHASE1_GENERATION_PRUNE_KEYS: contextvars.ContextVar[
    frozenset[tuple[int, int, int]]
] = contextvars.ContextVar(
    "hymem_phase1_generation_prune_keys", default=frozenset()
)
_EMBEDDING_MUTATION_KEYS: contextvars.ContextVar[
    frozenset[tuple[int, int, int]]
] = contextvars.ContextVar("hymem_embedding_mutation_keys", default=frozenset())
_TRANSACTION_LEASE_FENCES: contextvars.ContextVar[
    tuple[tuple[tuple[int, int, int], str, str], ...]
] = contextvars.ContextVar("hymem_transaction_lease_fences", default=())


class LeaseOwnershipLost(BaseException):
    """Control-flow signal raised when a fenced writer no longer owns its lease.

    This deliberately derives directly from :class:`BaseException`, like the
    deadline signal. Dreaming has best-effort ``except Exception`` recovery
    paths which write retry/quarantine state; lease loss must pass through all
    of them without publishing a replacement failure on behalf of an obsolete
    owner. The outer runner records only sanitized lifecycle telemetry.
    """


def _connection_authority_key(conn: sqlite3.Connection) -> tuple[int, int, int]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return id(conn), threading.get_ident(), id(task) if task is not None else 0


def activate_transaction_lease_fence(
    conn: sqlite3.Connection,
    *,
    name: str,
    holder: str,
) -> contextvars.Token:
    """Fence subsequent :func:`transaction` calls to one exact lease token.

    The authority is lexical and connection/thread/task scoped. It does not
    grant ownership: every transaction proves the token from ``run_lock`` only
    after acquiring SQLite's writer lock, and proves it again immediately
    before commit. Callers must reset the returned token in ``finally``.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("lease name must be a non-empty string")
    if not isinstance(holder, str) or not holder:
        raise ValueError("lease holder must be a non-empty string")
    key = _connection_authority_key(conn)
    active = _TRANSACTION_LEASE_FENCES.get()
    if any(existing_key == key for existing_key, _, _ in active):
        raise RuntimeError("transaction lease fence already active")
    return _TRANSACTION_LEASE_FENCES.set((*active, (key, name, holder)))


def deactivate_transaction_lease_fence(token: contextvars.Token) -> None:
    """Reset a token returned by :func:`activate_transaction_lease_fence`."""

    _TRANSACTION_LEASE_FENCES.reset(token)


def _transaction_lease_fence(
    conn: sqlite3.Connection,
) -> tuple[str, str] | None:
    key = _connection_authority_key(conn)
    for existing_key, name, holder in reversed(_TRANSACTION_LEASE_FENCES.get()):
        if existing_key == key:
            return name, holder
    return None


def _assert_transaction_lease_owned(conn: sqlite3.Connection) -> None:
    fence = _transaction_lease_fence(conn)
    if fence is None:
        return
    name, holder = fence
    owned = conn.execute(
        "SELECT 1 FROM run_lock WHERE name = ? AND holder = ?",
        (name, holder),
    ).fetchone()
    if owned is None:
        # Never include the random holder token in an exception or log string.
        raise LeaseOwnershipLost(f"{name} lease ownership lost")


def _load_schema() -> str:
    return (files("hymem.core") / "schema.sql").read_text(encoding="utf-8")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _v43_domain_present(conn: sqlite3.Connection) -> bool:
    """Return whether the complete external-provenance domain exists.

    A few supported historical migration fixtures intentionally contain only
    the table under test.  They must still advance their version marker, but
    cannot install cross-table v43 constraints.  Real HyMem stores and fresh
    schema bootstraps contain this complete set.
    """
    return all(
        _table_exists(conn, table)
        for table in (
            "sessions", "messages", "chunks", "message_retention_coverage",
            "peers", "kg_evidence", "episodes", "procedures",
            "profile_staging", "temporal_mentions", "narrative_facts",
            "chunk_message_sources", "user_profile", "kg_claim_observations",
        )
    )


def _v47_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether the extraction scheduling domain can accept migration v47.

    Historical migration tests include intentionally sparse schemas.  They may
    advance the version marker, but must not execute triggers or a data backfill
    against tables that do not exist.
    """
    if not all(
        _table_exists(conn, table)
        for table in (
            "sessions", "messages", "chunks", "message_retention_coverage",
            "chunk_message_sources", "processed_chunks",
        )
    ):
        return False
    required = {
        "sessions": {"id", "coverage_message_id"},
        "messages": {"id", "session_id", "role", "content", "created_at"},
        "chunks": {
            "id", "session_id", "start_message_id", "end_message_id",
            "salience_reason", "text", "chunk_kind",
            "source_manifest_version", "source_manifest_count",
        },
        "message_retention_coverage": {
            "message_id", "source_session_id", "source_role",
            "source_created_at", "chunk_id", "message_content_hash",
            "hash_version", "record_version", "coverage_version",
        },
        "chunk_message_sources": {
            "chunk_id", "ordinal", "source_message_id", "source_session_id",
            "source_coverage_chunk_id", "source_coverage_version",
        },
        "processed_chunks": {"chunk_id", "prompt_version"},
    }
    return all(
        columns.issubset({
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        })
        for table, columns in required.items()
    )


def _v48_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether the local Phase-1 retry ledger can accept v48 diagnostics."""
    if not _table_exists(conn, "chunk_extraction_attempts"):
        return False
    columns = {
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(chunk_extraction_attempts)"
        ).fetchall()
    }
    return {
        "chunk_id", "prompt_version", "attempts", "last_failure_at",
    }.issubset(columns)


def _v49_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether per-cycle extraction-call attribution can be installed."""
    return _table_exists(conn, "dream_runs")


def _v50_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether durable coverage-integrity health state can be installed."""
    if not all(_table_exists(conn, table) for table in ("sessions", "dream_runs")):
        return False
    session_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    return "id" in session_columns


def _v51_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether aggregation run attribution can accept the v51 contract."""
    return _table_exists(conn, "dream_runs")


def _v52_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether source-materialization acknowledgements can be installed."""
    if not all(_table_exists(conn, table) for table in ("sessions", "chunks")):
        return False
    session_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    chunk_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
    }
    return (
        {"id", "coverage_message_id"}.issubset(session_columns)
        and {"id", "session_id", "chunk_kind"}.issubset(chunk_columns)
    )


def _v53_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether the complete Phase-1 publication domain can accept v53."""

    return all(
        _table_exists(conn, table)
        for table in (
            "processed_chunks", "chunk_extraction_attempts",
            "kg_claim_extraction_outcomes", "kg_claim_observations",
            "behavioral_markers",
        )
    )


def _v53_generation_bindings_present(conn: sqlite3.Connection) -> bool:
    """Whether every v53 publication table has its producer binding column.

    ``_v53_domain_present`` intentionally describes the *pre-migration*
    domain and therefore cannot require columns which migration 053 has not
    added yet.  Runtime guards and registry pruning need the stronger,
    post-migration shape check: a stamped historical repair can temporarily
    recreate one of these tables in its older shape.
    """

    if not _v53_domain_present(conn):
        return False
    return all(
        "phase1_generation_key" in {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for table in (
            "processed_chunks", "chunk_extraction_attempts",
            "kg_claim_extraction_outcomes", "kg_claim_observations",
            "behavioral_markers",
        )
    )


def _v54_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether the complete producer-scoped auxiliary domain can accept v54."""

    return _v53_generation_bindings_present(conn) and all(
        _table_exists(conn, table)
        for table in (
            "entity_types", "entity_properties", "profile_entries", "rules",
        )
    )


def _v55_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether the historical aggregation domain can accept typed proofs."""

    return _v45_domain_present(conn)


def _v55_aggregation_bindings_present(
    conn: sqlite3.Connection, *, allow_v56: bool = False,
    allow_v57: bool = False,
) -> bool:
    """Reject stamped v55 stores with a partial/lookalike proof boundary."""

    if not _v55_aggregation_storage_present(
        conn, allow_v56=allow_v56, allow_v57=allow_v57,
    ):
        return False

    required_triggers = {
        "aggregation_source_header_insert_guard",
        "aggregation_source_header_update_guard",
        "aggregation_source_bound_update_guard",
        "aggregation_source_occurrence_insert_guard",
        "aggregation_source_occurrence_update_guard",
        "aggregation_source_occurrence_delete_unpublishes",
        "aggregation_input_insert_guard", "aggregation_input_update_guard",
        "aggregation_input_delete_unpublishes",
        "aggregation_input_source_insert_guard",
        "aggregation_input_source_update_guard",
        "aggregation_input_source_delete_unpublishes",
        "aggregation_publication_update_guard",
    }
    if allow_v56:
        required_triggers |= {
            "aggregation_generation_node_update_guard",
            "aggregation_generation_publication_insert_guard",
            "aggregation_generations_insert_guard",
            "aggregation_generations_update_guard",
            "aggregation_generations_delete_guard",
        }
    if allow_v57:
        required_triggers |= {
            "aggregation_material_node_update_guard",
            "aggregation_material_publication_insert_guard",
            "aggregation_material_epochs_insert_guard",
            "aggregation_material_epochs_update_guard",
            "aggregation_material_epochs_delete_guard",
        }
    fts_triggers = {
        "aggregation_nodes_fts_insert",
        "aggregation_nodes_fts_delete",
        "aggregation_nodes_fts_update",
    }
    expected_by_table = {
        "aggregation_nodes": {
            "aggregation_source_header_insert_guard",
            "aggregation_source_header_update_guard",
            "aggregation_source_bound_update_guard",
        } | fts_triggers,
        "aggregation_node_source_occurrences": {
            "aggregation_source_occurrence_insert_guard",
            "aggregation_source_occurrence_update_guard",
            "aggregation_source_occurrence_delete_unpublishes",
        },
        "aggregation_node_inputs": {
            "aggregation_input_insert_guard",
            "aggregation_input_update_guard",
            "aggregation_input_delete_unpublishes",
        },
        "aggregation_node_input_sources": {
            "aggregation_input_source_insert_guard",
            "aggregation_input_source_update_guard",
            "aggregation_input_source_delete_unpublishes",
        },
        "aggregation_publication_state": {
            "aggregation_publication_update_guard",
        },
    }
    if allow_v56:
        expected_by_table["aggregation_nodes"].add(
            "aggregation_generation_node_update_guard"
        )
        expected_by_table["aggregation_publication_state"].add(
            "aggregation_generation_publication_insert_guard"
        )
        expected_by_table["aggregation_generations"] = {
            "aggregation_generations_insert_guard",
            "aggregation_generations_update_guard",
            "aggregation_generations_delete_guard",
        }
    if allow_v57:
        expected_by_table["aggregation_nodes"].add(
            "aggregation_material_node_update_guard"
        )
        expected_by_table["aggregation_publication_state"].add(
            "aggregation_material_publication_insert_guard"
        )
        expected_by_table["aggregation_material_epochs"] = {
            "aggregation_material_epochs_insert_guard",
            "aggregation_material_epochs_update_guard",
            "aggregation_material_epochs_delete_guard",
        }
    trigger_rows = conn.execute(
        "SELECT name,tbl_name,sql FROM sqlite_master WHERE type='trigger'"
    ).fetchall()
    actual_by_table = {
        table: {
            str(row["name"])
            for row in trigger_rows
            if str(row["tbl_name"]) == table
        }
        for table in expected_by_table
    }
    if actual_by_table != expected_by_table:
        return False
    script = files("hymem.core.migrations").joinpath(
        "055_aggregation_typed_provenance.sql"
    ).read_text(encoding="utf-8")
    v56_script = files("hymem.core.migrations").joinpath(
        "056_aggregation_generation_identity.sql"
    ).read_text(encoding="utf-8") if allow_v56 else ""
    v57_script = files("hymem.core.migrations").joinpath(
        "057_aggregation_material_epoch.sql"
    ).read_text(encoding="utf-8") if allow_v57 else ""
    statements = _split_sql_statements(script)
    def normalized(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().rstrip(";")
    for name in required_triggers:
        source_script = (
            v57_script if name.startswith("aggregation_material")
            else v56_script if name.startswith("aggregation_generation")
            or name.startswith("aggregation_generations") else script
        )
        expected = next(
            statement for statement in _split_sql_statements(source_script)
            if re.match(rf"\s*CREATE\s+TRIGGER\s+{name}\b", statement, re.I)
        )
        actual = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
            (name,),
        ).fetchone()
        if actual is None or normalized(actual["sql"]) != normalized(expected):
            return False
    schema_statements = _split_sql_statements(_load_schema())
    for name in fts_triggers:
        expected = next(
            statement for statement in schema_statements
            if re.match(
                rf"\s*CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?{name}\b",
                statement,
                re.I,
            )
        )
        actual = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
            (name,),
        ).fetchone()
        if actual is None or normalized(
            re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", actual["sql"], flags=re.I)
        ) != normalized(
            re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", expected, flags=re.I)
        ):
            return False
    return True


def _v55_aggregation_storage_present(
    conn: sqlite3.Connection, *, allow_v56: bool = False,
    allow_v57: bool = False,
) -> bool:
    if not _v55_domain_present(conn):
        return False
    marker = conn.execute(
        "SELECT value FROM schema_meta WHERE "
        "key='aggregation_typed_provenance_schema'"
    ).fetchone()
    if marker is None or marker["value"] != "55":
        return False
    expected_node_xinfo = (
        ("id", "TEXT", 0, None, 1, 0),
        ("title", "TEXT", 1, None, 0, 0),
        ("summary", "TEXT", 1, None, 0, 0),
        ("member_episode_ids", "TEXT", 1, "'[]'", 0, 0),
        ("session_ids", "TEXT", 1, "'[]'", 0, 0),
        ("n_members", "INTEGER", 1, "0", 0, 0),
        ("n_sessions", "INTEGER", 1, "0", 0, 0),
        ("created_at", "TIMESTAMP", 0, "CURRENT_TIMESTAMP", 0, 0),
        ("level", "INTEGER", 1, "0", 0, 0),
        ("is_root", "INTEGER", 1, "0", 0, 0),
        ("source_manifest_version", "TEXT", 0, None, 0, 0),
        ("source_manifest_count", "INTEGER", 1, "0", 0, 0),
        ("source_manifest_hash", "TEXT", 0, None, 0, 0),
        ("source_manifest_complete", "BOOLEAN", 1, "0", 0, 0),
        ("input_fingerprint", "TEXT", 0, None, 0, 0),
        ("input_manifest_version", "TEXT", 0, None, 0, 0),
        ("input_manifest_count", "INTEGER", 1, "0", 0, 0),
        ("input_manifest_hash", "TEXT", 0, None, 0, 0),
        ("input_manifest_complete", "BOOLEAN", 1, "0", 0, 0),
        ("node_kind", "TEXT", 0, None, 0, 0),
        ("output_hash", "TEXT", 0, None, 0, 0),
        ("publication_id", "TEXT", 0, None, 0, 0),
        ("build_config_version", "TEXT", 0, None, 0, 0),
    ) + ((
        ("aggregation_generation_key", "TEXT", 0, None, 0, 0),
        ("aggregation_request_hash", "TEXT", 0, None, 0, 0),
    ) if allow_v56 else ()) + ((
        ("aggregation_material_epoch_key", "TEXT", 0, None, 0, 0),
    ) if allow_v57 else ())
    actual_node_xinfo = tuple(
        (str(row["name"]), str(row["type"]), int(row["notnull"]),
         row["dflt_value"], int(row["pk"]), int(row["hidden"]))
        for row in conn.execute("PRAGMA table_xinfo(aggregation_nodes)")
    )
    if actual_node_xinfo != expected_node_xinfo:
        return False
    node_expected = {
        "input_manifest_version": ("TEXT", 0, None),
        "input_manifest_count": ("INTEGER", 1, "0"),
        "input_manifest_hash": ("TEXT", 0, None),
        "input_manifest_complete": ("BOOLEAN", 1, "0"),
        "node_kind": ("TEXT", 0, None),
        "output_hash": ("TEXT", 0, None),
        "publication_id": ("TEXT", 0, None),
        "build_config_version": ("TEXT", 0, None),
    }
    if allow_v56:
        node_expected["aggregation_generation_key"] = ("TEXT", 0, None)
        node_expected["aggregation_request_hash"] = ("TEXT", 0, None)
    if allow_v57:
        node_expected["aggregation_material_epoch_key"] = ("TEXT", 0, None)
    node_info = {
        str(row["name"]): (str(row["type"]), int(row["notnull"]), row["dflt_value"])
        for row in conn.execute("PRAGMA table_info(aggregation_nodes)")
    }
    if any(node_info.get(name) != shape for name, shape in node_expected.items()):
        return False
    required_tables = {
        "aggregation_node_inputs", "aggregation_node_input_sources",
        "aggregation_node_source_occurrences", "aggregation_publication_state",
    }
    if allow_v56:
        required_tables.add("aggregation_generations")
    if allow_v57:
        required_tables.add("aggregation_material_epochs")
    if not all(_table_exists(conn, table) for table in required_tables):
        return False
    expected_shapes = {
        "aggregation_node_inputs": (
            ("node_id", "TEXT", 1, None, 1),
            ("ordinal", "INTEGER", 1, None, 2),
            ("input_kind", "TEXT", 1, None, 0),
            ("source_key", "TEXT", 1, None, 0),
            ("source_ref_json", "TEXT", 1, None, 0),
            ("payload_hash", "TEXT", 1, None, 0),
            ("authority_hash", "TEXT", 1, None, 0),
            ("source_manifest_count", "INTEGER", 1, None, 0),
            ("source_manifest_hash", "TEXT", 1, None, 0),
        ),
        "aggregation_node_input_sources": (
            ("node_id", "TEXT", 1, None, 1),
            ("input_ordinal", "INTEGER", 1, None, 2),
            ("source_ordinal", "INTEGER", 1, None, 3),
            ("source_message_id", "INTEGER", 1, None, 0),
            ("source_session_id", "TEXT", 1, None, 0),
            ("source_role", "TEXT", 1, None, 0),
            ("source_peer_id", "TEXT", 0, None, 0),
            ("source_workspace_id", "TEXT", 0, None, 0),
            ("source_created_at", "TIMESTAMP", 0, None, 0),
            ("source_coverage_chunk_id", "TEXT", 1, None, 0),
            ("source_coverage_version", "TEXT", 1, None, 0),
            ("source_content_hash", "TEXT", 1, None, 0),
        ),
        "aggregation_node_source_occurrences": (
            ("node_id", "TEXT", 1, None, 1),
            ("ordinal", "INTEGER", 1, None, 2),
            ("source_message_id", "INTEGER", 1, None, 0),
            ("source_session_id", "TEXT", 1, None, 0),
            ("source_role", "TEXT", 1, None, 0),
            ("source_peer_id", "TEXT", 0, None, 0),
            ("source_workspace_id", "TEXT", 0, None, 0),
            ("source_created_at", "TIMESTAMP", 0, None, 0),
            ("source_coverage_chunk_id", "TEXT", 1, None, 0),
            ("source_coverage_version", "TEXT", 1, None, 0),
            ("source_content_hash", "TEXT", 1, None, 0),
        ),
        "aggregation_publication_state": (
            ("id", "INTEGER", 0, None, 1),
            ("publication_id", "TEXT", 1, None, 0),
            ("config_version", "TEXT", 1, None, 0),
            ("cluster_min_members", "INTEGER", 1, None, 0),
            ("cluster_min_sessions", "INTEGER", 1, None, 0),
            ("anchor_fact_cap", "INTEGER", 1, None, 0),
            ("root_node_id", "TEXT", 0, None, 0),
            ("node_count", "INTEGER", 1, None, 0),
            ("node_set_hash", "TEXT", 1, None, 0),
            ("published_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
        ) + ((
            ("aggregation_generation_key", "TEXT", 0, None, 0),
            ("request_contract_sha256", "TEXT", 0, None, 0),
        ) if allow_v56 else ()) + ((
            ("aggregation_material_epoch_key", "TEXT", 0, None, 0),
            ("material_revision", "INTEGER", 0, None, 0),
            ("node_embedding_count", "INTEGER", 0, None, 0),
            ("node_embedding_set_hash", "TEXT", 0, None, 0),
        ) if allow_v57 else ()),
    }
    if allow_v56:
        expected_shapes["aggregation_generations"] = (
            ("generation_key", "TEXT", 0, None, 1),
            ("material_config_version", "TEXT", 1, None, 0),
            ("producer_identity_sha256", "TEXT", 1, None, 0),
            ("identity_exact", "BOOLEAN", 1, None, 0),
            ("reuse_scope", "TEXT", 1, None, 0),
            ("binding_json", "TEXT", 1, None, 0),
            ("created_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
        )
    for table, expected in expected_shapes.items():
        actual = tuple(
            (str(row["name"]), str(row["type"]), int(row["notnull"]),
             row["dflt_value"], int(row["pk"]))
            for row in conn.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            return False
    input_fks = tuple(
        tuple(row) for row in conn.execute(
            "PRAGMA foreign_key_list(aggregation_node_inputs)"
        )
    )
    source_fks = tuple(
        tuple(row) for row in conn.execute(
            "PRAGMA foreign_key_list(aggregation_node_input_sources)"
        )
    )
    if input_fks != ((0, 0, "aggregation_nodes", "node_id", "id",
                      "NO ACTION", "CASCADE", "NONE"),):
        return False
    expected_source_fks = (
        (0, 0, "message_retention_coverage", "source_message_id", "message_id", "NO ACTION", "RESTRICT", "NONE"),
        (0, 1, "message_retention_coverage", "source_coverage_chunk_id", "chunk_id", "NO ACTION", "RESTRICT", "NONE"),
        (0, 2, "message_retention_coverage", "source_coverage_version", "coverage_version", "NO ACTION", "RESTRICT", "NONE"),
        (1, 0, "aggregation_node_inputs", "node_id", "node_id", "NO ACTION", "CASCADE", "NONE"),
        (1, 1, "aggregation_node_inputs", "input_ordinal", "ordinal", "NO ACTION", "CASCADE", "NONE"),
    )
    if source_fks != expected_source_fks:
        return False
    flat_source_fks = tuple(
        tuple(row) for row in conn.execute(
            "PRAGMA foreign_key_list(aggregation_node_source_occurrences)"
        )
    )
    if flat_source_fks != (
        (0, 0, "message_retention_coverage", "source_message_id", "message_id", "NO ACTION", "RESTRICT", "NONE"),
        (0, 1, "message_retention_coverage", "source_coverage_chunk_id", "chunk_id", "NO ACTION", "RESTRICT", "NONE"),
        (0, 2, "message_retention_coverage", "source_coverage_version", "coverage_version", "NO ACTION", "RESTRICT", "NONE"),
        (1, 0, "aggregation_nodes", "node_id", "id", "NO ACTION", "CASCADE", "NONE"),
    ):
        return False
    if allow_v56:
        node_generation_fks = tuple(
            tuple(row) for row in conn.execute(
                "PRAGMA foreign_key_list(aggregation_nodes)"
            )
        )
        publication_generation_fks = tuple(
            tuple(row) for row in conn.execute(
                "PRAGMA foreign_key_list(aggregation_publication_state)"
            )
        )
        expected_generation_fk = (
            0, 0, "aggregation_generations", "aggregation_generation_key",
            "generation_key", "NO ACTION", "RESTRICT", "NONE",
        )
        expected_node_fks = (expected_generation_fk,)
        expected_publication_fks = (expected_generation_fk,)
        if allow_v57:
            expected_material_fk = (
                0, 0, "aggregation_material_epochs",
                "aggregation_material_epoch_key", "material_epoch_key",
                "NO ACTION", "RESTRICT", "NONE",
            )
            # SQLite numbers ALTER-added foreign keys newest-first.
            expected_generation_fk = (
                1, 0, "aggregation_generations", "aggregation_generation_key",
                "generation_key", "NO ACTION", "RESTRICT", "NONE",
            )
            expected_node_fks = (expected_material_fk, expected_generation_fk)
            expected_publication_fks = (
                expected_material_fk, expected_generation_fk,
            )
        if (
            node_generation_fks != expected_node_fks
            or publication_generation_fks != expected_publication_fks
        ):
            return False
    # Index names and every index_xinfo field are part of the owned domain.
    # Comparing only column sets would collapse duplicate indexes and ignore a
    # forged collation, descending key, expression, or auxiliary row.
    def xinfo(*rows: tuple[int, int, str | None, int, str, int]) -> tuple[
        tuple[int, int, str | None, int, str, int], ...
    ]:
        return rows

    expected_indexes = {
        "aggregation_nodes": {
            "sqlite_autoindex_aggregation_nodes_1": (
                1, "pk", 0,
                xinfo((0, 0, "id", 0, "BINARY", 1),
                      (1, -1, None, 0, "BINARY", 0)),
            ),
        },
        "aggregation_node_inputs": {
            "sqlite_autoindex_aggregation_node_inputs_1": (
                1, "pk", 0,
                xinfo((0, 0, "node_id", 0, "BINARY", 1),
                      (1, 1, "ordinal", 0, "BINARY", 1),
                      (2, -1, None, 0, "BINARY", 0)),
            ),
            "sqlite_autoindex_aggregation_node_inputs_2": (
                1, "u", 0,
                xinfo((0, 0, "node_id", 0, "BINARY", 1),
                      (1, 2, "input_kind", 0, "BINARY", 1),
                      (2, 3, "source_key", 0, "BINARY", 1),
                      (3, -1, None, 0, "BINARY", 0)),
            ),
        },
        "aggregation_node_input_sources": {
            "sqlite_autoindex_aggregation_node_input_sources_1": (
                1, "pk", 0,
                xinfo((0, 0, "node_id", 0, "BINARY", 1),
                      (1, 1, "input_ordinal", 0, "BINARY", 1),
                      (2, 2, "source_ordinal", 0, "BINARY", 1),
                      (3, -1, None, 0, "BINARY", 0)),
            ),
            "sqlite_autoindex_aggregation_node_input_sources_2": (
                1, "u", 0,
                xinfo((0, 0, "node_id", 0, "BINARY", 1),
                      (1, 1, "input_ordinal", 0, "BINARY", 1),
                      (2, 4, "source_session_id", 0, "BINARY", 1),
                      (3, 3, "source_message_id", 0, "BINARY", 1),
                      (4, -1, None, 0, "BINARY", 0)),
            ),
            "idx_aggregation_input_source_occurrence": (
                0, "c", 0,
                xinfo((0, 4, "source_session_id", 0, "BINARY", 1),
                      (1, 3, "source_message_id", 0, "BINARY", 1),
                      (2, -1, None, 0, "BINARY", 0)),
            ),
        },
        "aggregation_node_source_occurrences": {
            "sqlite_autoindex_aggregation_node_source_occurrences_1": (
                1, "pk", 0,
                xinfo((0, 0, "node_id", 0, "BINARY", 1),
                      (1, 1, "ordinal", 0, "BINARY", 1),
                      (2, -1, None, 0, "BINARY", 0)),
            ),
            "sqlite_autoindex_aggregation_node_source_occurrences_2": (
                1, "u", 0,
                xinfo((0, 0, "node_id", 0, "BINARY", 1),
                      (1, 3, "source_session_id", 0, "BINARY", 1),
                      (2, 2, "source_message_id", 0, "BINARY", 1),
                      (3, -1, None, 0, "BINARY", 0)),
            ),
            "idx_aggregation_source_occurrence": (
                0, "c", 0,
                xinfo((0, 3, "source_session_id", 0, "BINARY", 1),
                      (1, 2, "source_message_id", 0, "BINARY", 1),
                      (2, -1, None, 0, "BINARY", 0)),
            ),
        },
        "aggregation_publication_state": {},
    }
    if allow_v56:
        expected_indexes["aggregation_generations"] = {
            "sqlite_autoindex_aggregation_generations_1": (
                1, "pk", 0,
                xinfo((0, 0, "generation_key", 0, "BINARY", 1),
                      (1, -1, None, 0, "BINARY", 0)),
            ),
        }
    for table, expected in expected_indexes.items():
        actual: dict[str, tuple[object, ...]] = {}
        for index in conn.execute(f"PRAGMA index_list({table})"):
            name = str(index["name"])
            rows = tuple(
                (
                    int(row["seqno"]), int(row["cid"]), row["name"],
                    int(row["desc"]), str(row["coll"]), int(row["key"]),
                )
                for row in conn.execute(f"PRAGMA index_xinfo('{name}')")
            )
            actual[name] = (
                int(index["unique"]), str(index["origin"]),
                int(index["partial"]), rows,
            )
        if actual != expected:
            return False

    # Exact CREATE SQL catches weakened/missing CHECK and UNIQUE clauses which
    # PRAGMA column/FK metadata cannot reveal.
    script = files("hymem.core.migrations").joinpath(
        "055_aggregation_typed_provenance.sql"
    ).read_text(encoding="utf-8")
    statements = _split_sql_statements(script)
    def normalized(value: str) -> str:
        value = re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", value, flags=re.I)
        value = re.sub(r"--[^\n]*", "", value)
        return re.sub(r"\s+", " ", value).strip().rstrip(";")
    exact_v55_tables = [
        "aggregation_node_inputs", "aggregation_node_input_sources",
    ]
    if not allow_v56:
        exact_v55_tables.append("aggregation_publication_state")
    for table in exact_v55_tables:
        expected = next(
            statement for statement in statements
            if re.match(
                rf"\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{table}\b",
                statement, re.I,
            )
        )
        actual_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if actual_row is None or normalized(actual_row["sql"]) != normalized(expected):
            return False
    if allow_v56:
        expected_registry = next(
            statement for statement in _split_sql_statements(
                files("hymem.core.migrations").joinpath(
                    "056_aggregation_generation_identity.sql"
                ).read_text(encoding="utf-8")
            )
            if re.match(r"\s*CREATE\s+TABLE\s+aggregation_generations\b", statement, re.I)
        )
        actual_registry = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='aggregation_generations'"
        ).fetchone()
        if (
            actual_registry is None
            or normalized(actual_registry["sql"]) != normalized(expected_registry)
        ):
            return False
    index_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND "
        "name='idx_aggregation_input_source_occurrence'"
    ).fetchone()
    expected_index = next(
        statement for statement in statements
        if re.match(r"\s*CREATE\s+INDEX.*idx_aggregation_input_source_occurrence\b", statement, re.I)
    )
    if index_row is None or normalized(index_row["sql"]) != normalized(expected_index):
        return False
    schema_statements = _split_sql_statements(_load_schema())
    flat_expected = next(
        statement for statement in schema_statements
        if re.match(
            r"\s*CREATE\s+TABLE.*aggregation_node_source_occurrences\b",
            statement, re.I,
        )
    )
    flat_actual = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND "
        "name='aggregation_node_source_occurrences'"
    ).fetchone()
    if flat_actual is None or normalized(flat_actual["sql"]) != normalized(flat_expected):
        return False
    fresh_node_expected = next(
        statement for statement in schema_statements
        if re.match(r"\s*CREATE\s+TABLE.*aggregation_nodes\b", statement, re.I)
    )
    # A real v54->v55 upgrade has the same exact columns/constraints but
    # SQLite serializes ALTER-added columns after the original table CHECK.
    # Construct that one canonical alternate spelling from the owned sources.
    marker = fresh_node_expected.index("    input_manifest_version TEXT,")
    legacy_check = fresh_node_expected.index(
        "    CHECK (\n        (source_manifest_complete", marker
    )
    legacy_node_sql = fresh_node_expected[:marker] + fresh_node_expected[legacy_check:]
    reference = sqlite3.connect(":memory:")
    try:
        reference.execute(legacy_node_sql)
        for statement in statements:
            if re.match(
                r"\s*ALTER\s+TABLE\s+aggregation_nodes\s+ADD\s+COLUMN\b",
                statement, re.I,
            ):
                reference.execute(statement)
        if allow_v56:
            v56_statements = _split_sql_statements(
                files("hymem.core.migrations").joinpath(
                    "056_aggregation_generation_identity.sql"
                ).read_text(encoding="utf-8")
            )
            for statement in v56_statements:
                if re.match(
                    r"\s*ALTER\s+TABLE\s+aggregation_nodes\s+ADD\s+COLUMN\b",
                    statement, re.I,
                ):
                    reference.execute(statement)
        if allow_v57:
            v57_statements = _split_sql_statements(
                files("hymem.core.migrations").joinpath(
                    "057_aggregation_material_epoch.sql"
                ).read_text(encoding="utf-8")
            )
            for statement in v57_statements:
                if re.match(
                    r"\s*ALTER\s+TABLE\s+aggregation_nodes\s+ADD\s+COLUMN\b",
                    statement, re.I,
                ):
                    reference.execute(statement)
        migrated_node_expected = reference.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND "
            "name='aggregation_nodes'"
        ).fetchone()[0]
    finally:
        reference.close()
    node_actual = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND "
        "name='aggregation_nodes'"
    ).fetchone()
    if node_actual is None or normalized(node_actual["sql"]) not in {
        normalized(fresh_node_expected), normalized(migrated_node_expected),
    }:
        return False
    return True


def _v55_domain_footprint_present(conn: sqlite3.Connection) -> bool:
    """Distinguish intentionally sparse fixtures from damaged v55 stores."""

    objects = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger','index')"
        )
    }
    marker = conn.execute(
        "SELECT value FROM schema_meta WHERE "
        "key='aggregation_typed_provenance_schema'"
    ).fetchone()
    if marker is not None:
        return True
    if objects & {
        "aggregation_node_inputs", "aggregation_node_input_sources",
        "aggregation_node_source_occurrences", "aggregation_publication_state",
        "aggregation_input_insert_guard",
        "aggregation_source_header_insert_guard",
        "idx_aggregation_input_source_occurrence",
        "idx_aggregation_source_occurrence", "aggregation_nodes_fts",
    }:
        return True
    if _table_exists(conn, "aggregation_nodes"):
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(aggregation_nodes)")
        }
        return bool(columns & {
            "input_manifest_version", "input_manifest_hash", "node_kind",
            "output_hash", "publication_id", "build_config_version",
        })
    return False


def _v56_generation_storage_present(
    conn: sqlite3.Connection, *, allow_v57: bool = False,
) -> bool:
    """Require v56 tables, columns, FKs, and canonical registry rows."""

    marker = conn.execute(
        "SELECT value FROM schema_meta WHERE key='aggregation_generation_schema'"
    ).fetchone()
    if marker is None or marker["value"] != "56":
        return False
    if not _v55_aggregation_storage_present(
        conn, allow_v56=True, allow_v57=allow_v57,
    ):
        return False
    expected_columns = {
        "aggregation_build_health": {
            "last_success_generation_key", "pending_generation_key",
            "last_failure_generation_key",
        },
        "dream_runs": {"aggregation_generation_key"},
    }
    for table, required in expected_columns.items():
        columns = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if not required.issubset(columns):
            return False
        fks = {
            (str(row["from"]), str(row["table"]), str(row["to"]),
             str(row["on_delete"]))
            for row in conn.execute(f"PRAGMA foreign_key_list({table})")
        }
        if any(
            (column, "aggregation_generations", "generation_key", "RESTRICT")
            not in fks
            for column in required
        ):
            return False
    malformed = conn.execute(
        "SELECT 1 FROM aggregation_generations WHERE "
        "hymem_aggregation_generation_registry_row_is_valid("
        "generation_key,material_config_version,producer_identity_sha256,"
        "identity_exact,reuse_scope,binding_json)<>1 LIMIT 1"
    ).fetchone()
    return malformed is None


def _v56_generation_bindings_present(
    conn: sqlite3.Connection, *, allow_v57: bool = False,
) -> bool:
    """Require the complete exact aggregation-generation boundary."""

    return (
        _v56_generation_storage_present(conn, allow_v57=allow_v57)
        and _v55_aggregation_bindings_present(
            conn, allow_v56=True, allow_v57=allow_v57,
        )
    )


def _v57_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether the complete aggregation-material boundary can be installed."""

    required_columns = {
        "embedding_cache": {
            "text_hash", "model", "vector_json", "dim", "created_at",
        },
        "message_embeddings": {
            "message_id", "source_coverage_chunk_id",
            "source_coverage_version", "text_hash", "model", "vector_json",
            "dim", "created_at",
        },
        "chunk_embeddings": {
            "chunk_id", "model", "vector_json", "dim", "text_hash",
            "created_at",
        },
        "edge_embeddings": {
            "edge_text", "model", "vector_json", "dim", "created_at",
        },
        "episode_embeddings": {
            "episode_id", "model", "vector_json", "dim", "text_hash",
            "created_at",
        },
        "narrative_fact_embeddings": {
            "fact_id", "model", "vector_json", "dim", "text_hash",
            "created_at",
        },
        "aggregation_node_embeddings": {
            "node_id", "model", "vector_json", "dim", "text_hash",
            "created_at",
        },
        "sessions": {"id", "digest_published_generation", "source_workspace_id"},
        "messages": {
            "id", "session_id", "role", "content", "created_at",
            "source_peer_id", "source_workspace_id",
        },
        "peers": {"id", "workspace_id", "role"},
        "session_peers": {"session_id", "workspace_id", "peer_id"},
        "chunks": {
            "id", "session_id", "start_message_id", "end_message_id", "text",
            "chunk_kind", "source_manifest_version", "source_manifest_count",
        },
        "message_retention_coverage": {
            "message_id", "source_session_id", "source_role", "source_peer_id",
            "source_workspace_id", "source_created_at", "chunk_id",
            "message_content_hash", "hash_version", "record_version",
            "coverage_version",
        },
        "episode_source_occurrences": {
            "episode_id", "ordinal", "source_message_id", "source_session_id",
            "source_role", "source_created_at", "source_content_hash",
            "source_peer_id", "source_workspace_id", "source_coverage_chunk_id",
            "source_coverage_version",
        },
        "episodes": {
            "id", "session_id", "start_message_id", "end_message_id",
            "title", "summary", "key_entities",
            "digest_slice_key", "digest_generation", "source_manifest_version",
            "source_manifest_count", "source_manifest_hash",
            "source_manifest_complete",
        },
        "aggregation_nodes": {"id", "publication_id"},
        "aggregation_publication_state": {"publication_id"},
        "aggregation_build_health": {
            "id", "last_success_config_version", "last_success_at",
            "pending_config_version", "pending_attempts",
            "pending_caught_exceptions", "pending_fusion_failures",
            "first_pending_at", "last_attempt_at",
            "total_caught_exceptions", "total_fusion_failures",
            "superseded_pending_configs", "last_failure_config_version",
            "last_failure_kind", "last_failure_at",
            "last_success_generation_key", "pending_generation_key",
            "last_failure_generation_key",
        },
        "dream_runs": {"id"},
        "user_profile": {
            "slot", "slot_key", "value", "confidence", "valid_at",
            "invalid_at", "source_message_id", "source_session_id",
            "source_created_at",
        },
        "knowledge_graph": {
            "id", "subject_canonical", "predicate", "object_canonical",
            "derived", "status", "pos_evidence", "neg_evidence", "first_seen",
            "last_seen", "last_reinforced", "valid_at", "invalid_at",
        },
        "kg_evidence": {
            "id", "edge_id", "chunk_id", "polarity", "surface_subject",
            "surface_object", "value_text", "value_numeric", "value_unit",
            "temporal_scope", "source_role", "evidence_kind", "evidence_weight",
            "weight_source", "extraction_prompt_version", "extracted_at",
            "source_message_id", "source_session_id", "source_created_at",
            "source_event_at",
            "source_peer_id", "source_workspace_id", "source_coverage_chunk_id",
            "source_coverage_version", "provenance_status", "interpretation_key",
            "revision", "is_current", "superseded_at", "superseded_reason",
            "published_at",
        },
        "kg_evidence_signals": {
            "edge_id", "signal_key", "signal_kind", "polarity",
            "evidence_weight", "counts_toward_confidence",
        },
        "kg_edge_lifecycle": {
            "edge_id", "event_kind", "direction", "event_at",
            "source_evidence_id", "created_at",
        },
        "kg_claim_observations": {
            "chunk_id", "edge_id", "source_session_id", "source_message_id",
            "evidence_kind", "polarity", "prompt_version", "prompt_generation",
            "evidence_id", "interpretation_key", "observed_at",
            "phase1_generation_key",
        },
        "kg_claim_extraction_outcomes": {
            "chunk_id", "prompt_version", "prompt_generation", "result_hash",
            "succeeded_at", "phase1_generation_key",
        },
        "phase1_generations": {
            "generation_key", "extraction_cache_key",
            "producer_identity_sha256", "identity_exact", "reuse_scope",
            "binding_json", "created_at",
        },
    }
    # v57 extends the exact v56 publication/generation boundary; a sparse
    # lookalike with only v45-era tables must be skipped, never admitted into
    # ALTER/trigger DDL that assumes the complete v56 columns and guards.
    if not _v56_generation_bindings_present(
        conn, allow_v57=_v57_domain_footprint_present(conn),
    ):
        return False
    for table, required in required_columns.items():
        if not _table_exists(conn, table):
            return False
        columns = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if not required.issubset(columns):
            return False
    if not _v57_embedding_logical_keys_present(conn):
        return False
    if not _v57_source_logical_keys_present(conn):
        return False
    return True


_V57_EMBEDDING_LOGICAL_KEYS = {
    "embedding_cache": ("text_hash", "model"),
    "message_embeddings": ("message_id",),
    "chunk_embeddings": ("chunk_id",),
    "edge_embeddings": ("edge_text",),
    "episode_embeddings": ("episode_id",),
    "narrative_fact_embeddings": ("fact_id",),
    "aggregation_node_embeddings": ("node_id",),
}
_V57_EMBEDDING_ROWID_ALIAS_KEYS = {
    ("message_embeddings", ("message_id",)),
    ("narrative_fact_embeddings", ("fact_id",)),
}


# v57 joins and invalidators assume these historical source coordinates are
# unique.  A column-complete CTAS/lookalike without its key constraints can
# duplicate one logical episode/source/evidence during capture, or make a
# maintained UPSERT silently append another row.  Preflight therefore owns
# the minimum logical-key boundary it consumes instead of accepting columns
# alone.  Tables with two entries require both independent keys.
_V57_SOURCE_LOGICAL_KEYS = {
    "sessions": (("id",),),
    "messages": (("id",),),
    "peers": (("id", "workspace_id"),),
    "session_peers": (("session_id", "workspace_id", "peer_id"),),
    "chunks": (("id",),),
    "message_retention_coverage": (
        ("message_id", "chunk_id", "coverage_version"),
    ),
    "episode_source_occurrences": (
        ("episode_id", "ordinal"),
        ("episode_id", "source_session_id", "source_message_id"),
    ),
    "episodes": (("id",),),
    "user_profile": (("id",),),
    "knowledge_graph": (
        ("id",),
        ("subject_canonical", "predicate", "object_canonical"),
    ),
    "kg_evidence": (("id",),),
    "kg_evidence_signals": (
        ("id",), ("edge_id", "signal_kind", "signal_key"),
    ),
    "kg_edge_lifecycle": (("id",), ("edge_id", "event_key")),
    "kg_claim_observations": ((
        "chunk_id", "edge_id", "source_session_id", "source_message_id",
        "evidence_kind",
    ),),
    "kg_claim_extraction_outcomes": (("chunk_id",),),
    "phase1_generations": (("generation_key",),),
}
_V57_SOURCE_AUTOINCREMENT_KEYS = {
    ("messages", ("id",)),
    ("user_profile", ("id",)),
    ("knowledge_graph", ("id",)),
    ("kg_evidence", ("id",)),
    ("kg_evidence_signals", ("id",)),
    ("kg_edge_lifecycle", ("id",)),
}

_V57_EMBEDDING_RUNTIME_COLUMNS = {
    "embedding_cache": {
        "text_hash", "model", "vector_json", "dim", "created_at",
    },
    "message_embeddings": {
        "message_id", "source_coverage_chunk_id", "source_coverage_version",
        "text_hash", "model", "vector_json", "dim", "created_at",
    },
    "chunk_embeddings": {
        "chunk_id", "model", "vector_json", "dim", "text_hash", "created_at",
    },
    "edge_embeddings": {
        "edge_text", "model", "vector_json", "dim", "created_at",
    },
    "episode_embeddings": {
        "episode_id", "model", "vector_json", "dim", "text_hash", "created_at",
    },
    "narrative_fact_embeddings": {
        "fact_id", "model", "vector_json", "dim", "text_hash", "created_at",
    },
    "aggregation_node_embeddings": {
        "node_id", "model", "vector_json", "dim", "text_hash", "created_at",
    },
}


def _v57_embedding_runtime_columns_present(conn: sqlite3.Connection) -> bool:
    for table, required in _V57_EMBEDDING_RUNTIME_COLUMNS.items():
        if not _table_exists(conn, table):
            return False
        columns = {
            str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if not required.issubset(columns):
            return False
    return True


def _v57_embedding_logical_keys_present(conn: sqlite3.Connection) -> bool:
    """Require the exact conflict key used by every maintained mirror writer."""

    for table, expected in _V57_EMBEDDING_LOGICAL_KEYS.items():
        if not _table_exists(conn, table):
            return False
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        primary = tuple(
            str(row["name"])
            for row in sorted(info, key=lambda row: int(row["pk"]) or 2**31)
            if int(row["pk"]) > 0
        )
        rowid_required = (table, expected) in _V57_EMBEDDING_ROWID_ALIAS_KEYS
        matched = rowid_required and _integer_rowid_alias_is_exact(
            conn, table, expected[0], require_autoincrement=False,
        )
        for index in conn.execute(f"PRAGMA index_list({table})").fetchall():
            if rowid_required:
                continue
            if int(index["unique"]) != 1 or int(index["partial"]) != 0:
                continue
            if _index_xinfo_shape_is_exact(
                conn, str(index["name"]), expected,
            ):
                matched = True
                break
        if not matched:
            return False
    return True


def _v57_source_logical_keys_present(conn: sqlite3.Connection) -> bool:
    """Require every source identity v57 treats as one canonical row."""

    for table, expected_keys in _V57_SOURCE_LOGICAL_KEYS.items():
        if not _table_exists(conn, table):
            return False
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        primary = tuple(
            str(row["name"])
            for row in sorted(info, key=lambda row: int(row["pk"]) or 2**31)
            if int(row["pk"]) > 0
        )
        unique_keys: set[tuple[str, ...]] = set()
        auto_key = next((
            expected for expected in expected_keys
            if (table, expected) in _V57_SOURCE_AUTOINCREMENT_KEYS
        ), None)
        if auto_key is not None and _integer_rowid_alias_is_exact(
            conn, table, auto_key[0], require_autoincrement=True,
        ):
            unique_keys.add(auto_key)
        for index in conn.execute(f"PRAGMA index_list({table})").fetchall():
            if int(index["unique"]) != 1 or int(index["partial"]) != 0:
                continue
            name = str(index["name"])
            for expected in expected_keys:
                if (table, expected) in _V57_SOURCE_AUTOINCREMENT_KEYS:
                    continue
                if _index_xinfo_shape_is_exact(conn, name, expected):
                    unique_keys.add(expected)
        if any(expected not in unique_keys for expected in expected_keys):
            return False
    return True


def _integer_rowid_alias_is_exact(
    conn: sqlite3.Connection, table: str, column: str, *,
    require_autoincrement: bool,
) -> bool:
    """Distinguish a rowid alias from UNIQUE or historic ``PK DESC``."""

    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    primary = tuple(
        str(row["name"])
        for row in sorted(info, key=lambda row: int(row["pk"]) or 2**31)
        if int(row["pk"]) > 0
    )
    if primary != (column,):
        return False
    target = next(row for row in info if str(row["name"]) == column)
    if str(target["type"]).upper() != "INTEGER":
        return False
    # ``INTEGER PRIMARY KEY DESC`` creates a pk-origin index and is not the
    # rowid alias despite table_info reporting the same type/pk ordinal.
    if any(
        str(index["origin"]) == "pk"
        for index in conn.execute(f"PRAGMA index_list({table})").fetchall()
    ):
        return False
    if require_autoincrement:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if row is None or re.search(
            rf'(?<!\w)"?{re.escape(column)}"?\s+INTEGER\s+PRIMARY\s+KEY\s+'
            r'AUTOINCREMENT\b',
            str(row["sql"]), re.I,
        ) is None:
            return False
    return True


def _v57_material_bindings_present(
    conn: sqlite3.Connection, *, validate_triggers: bool = True,
) -> bool:
    """Reject partial/lookalike v57 clocks, registries, columns and guards."""

    marker = conn.execute(
        "SELECT value FROM schema_meta WHERE "
        "key='aggregation_material_epoch_schema'"
    ).fetchone()
    if marker is None or marker["value"] != "57":
        return False
    if not (
        _v56_generation_bindings_present(conn, allow_v57=True)
        if validate_triggers
        else _v56_generation_storage_present(conn, allow_v57=True)
    ):
        return False
    if not _v57_embedding_runtime_columns_present(conn):
        return False
    expected_columns = {
        "episode_embeddings": {
            "embedding_producer_key": ("TEXT", 0, None),
        },
        "aggregation_node_embeddings": {
            "embedding_producer_key": ("TEXT", 0, None),
        },
        "aggregation_nodes": {
            "aggregation_material_epoch_key": ("TEXT", 0, None),
        },
        "aggregation_publication_state": {
            "aggregation_material_epoch_key": ("TEXT", 0, None),
            "material_revision": ("INTEGER", 0, None),
            "node_embedding_count": ("INTEGER", 0, None),
            "node_embedding_set_hash": ("TEXT", 0, None),
        },
        "aggregation_build_health": {
            "last_success_material_epoch_key": ("TEXT", 0, None),
            "pending_material_epoch_key": ("TEXT", 0, None),
            "last_failure_material_epoch_key": ("TEXT", 0, None),
            "attempt_serial": ("INTEGER", 1, "0"),
            "pending_attempt_token": ("INTEGER", 0, None),
        },
        "dream_runs": {
            "aggregation_material_epoch_key": ("TEXT", 0, None),
        },
    }
    for table, required in expected_columns.items():
        if not _table_exists(conn, table):
            return False
        columns = {
            str(row["name"]): (
                str(row["type"]), int(row["notnull"]), row["dflt_value"],
            )
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if any(columns.get(name) != shape for name, shape in required.items()):
            return False
    malformed_health = conn.execute(
        "SELECT 1 FROM aggregation_build_health WHERE id<>1 OR "
        "typeof(attempt_serial)<>'integer' OR attempt_serial<0 OR "
        "attempt_serial>9223372036854775806 OR NOT (("
        "pending_config_version IS NULL AND pending_generation_key IS NULL "
        "AND pending_material_epoch_key IS NULL "
        "AND pending_attempt_token IS NULL) OR ("
        "pending_config_version IS NOT NULL "
        "AND pending_generation_key IS NOT NULL "
        "AND pending_attempt_token=attempt_serial "
        "AND pending_attempt_token BETWEEN 1 AND 9223372036854775806)) "
        "LIMIT 1"
    ).fetchone()
    if malformed_health is not None:
        return False
    if not all(_table_exists(conn, table) for table in (
        "aggregation_material_clock", "aggregation_material_epochs",
    )):
        return False
    clock = conn.execute(
        "SELECT id,revision,clock_schema FROM aggregation_material_clock"
    ).fetchall()
    if (
        len(clock) != 1
        or clock[0]["id"] != 1
        or not isinstance(clock[0]["revision"], int)
        or isinstance(clock[0]["revision"], bool)
        or not 0 <= clock[0]["revision"] <= 9223372036854775806
        or clock[0]["clock_schema"]
        != "hymem-aggregation-material-clock-v1"
    ):
        return False
    malformed = conn.execute(
        "SELECT 1 FROM aggregation_material_epochs WHERE "
        "hymem_aggregation_material_registry_row_is_valid("
        "material_epoch_key,material_revision,config_version,snapshot_sha256,"
        "embedding_producer_key,identity_exact,reuse_scope,binding_json)<>1 "
        "LIMIT 1"
    ).fetchone()
    if malformed is not None:
        return False
    for table in (
        "embedding_cache", "message_embeddings", "chunk_embeddings",
        "edge_embeddings", "episode_embeddings",
        "narrative_fact_embeddings", "aggregation_node_embeddings",
    ):
        if not _table_exists(conn, table):
            return False
        producer_clause = (
            " OR embedding_producer_key IS NOT model"
            if table in {"episode_embeddings", "aggregation_node_embeddings"}
            else ""
        )
        unsafe = conn.execute(
            f"SELECT 1 FROM {table} WHERE typeof(model)<>'text' "
            "OR length(model)<>92 "
            "OR substr(model,1,28)<>'hymem-embedding-producer-v1:' "
            "OR substr(model,29) GLOB '*[^0-9a-f]*'"
            + producer_clause + " LIMIT 1"
        ).fetchone()
        if unsafe is not None:
            return False
    if not _v57_embedding_logical_keys_present(conn):
        return False
    if not _v57_source_logical_keys_present(conn):
        return False
    unsafe_vec_model = conn.execute(
        "SELECT 1 FROM schema_meta WHERE key='vec_model' AND ("
        "typeof(value)<>'text' OR length(value)<>92 OR "
        "substr(value,1,28)<>'hymem-embedding-producer-v1:' OR "
        "substr(value,29) GLOB '*[^0-9a-f]*') LIMIT 1"
    ).fetchone()
    if unsafe_vec_model is not None:
        return False
    script = files("hymem.core.migrations").joinpath(
        "057_aggregation_material_epoch.sql"
    ).read_text(encoding="utf-8")
    # Exact trigger SQL is part of the freshness boundary. Merely finding a
    # same-named no-op trigger would otherwise make a stamped store look safe.
    def normalized(value: str) -> str:
        value = re.sub(r"--[^\n]*", "", value)
        return re.sub(r"\s+", " ", value).strip().rstrip(";")
    statements = _split_sql_statements(script)
    expected_trigger_sql = {
        match.group(1): statement
        for statement in statements
        if (match := re.match(
            r"\s*CREATE\s+TRIGGER\s+(\w+)\b", statement, flags=re.I,
        ))
    }
    if validate_triggers:
        for name, expected in expected_trigger_sql.items():
            actual = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
                (name,),
            ).fetchone()
            if actual is None or normalized(actual["sql"]) != normalized(expected):
                return False
        expected_index_sql = {
            match.group(1): statement
            for statement in statements
            if (match := re.match(
                r"\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)\b",
                statement,
                flags=re.I,
            ))
        }
        for name, expected in expected_index_sql.items():
            actual = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                (name,),
            ).fetchone()
            if actual is None or normalized(actual["sql"]) != normalized(expected):
                return False
        expected_view_sql = {
            match.group(1): statement
            for statement in statements
            if (match := re.match(
                r"\s*CREATE\s+VIEW\s+(\w+)\b", statement, flags=re.I,
            ))
        }
        for name, expected in expected_view_sql.items():
            actual = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='view' AND name=?",
                (name,),
            ).fetchone()
            if actual is None or normalized(actual["sql"]) != normalized(expected):
                return False

    # Every v57 ALTER constraint must survive serialization exactly. This
    # catches nullable/lookalike keys and weakened digest/FK CHECKs while the
    # v55/v56 validators continue to own the pre-v57 portion of each table.
    actual_table_sql = {
        table: normalized(conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()["sql"])
        for table in expected_columns
    }
    for statement in statements:
        match = re.match(
            r"\s*ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+([\s\S]+)\Z",
            statement.rstrip(";"), flags=re.I,
        )
        if match is None:
            continue
        table, definition = match.groups()
        if table in actual_table_sql and normalized(definition) not in actual_table_sql[table]:
            return False

    expected_material_fks = {
        "aggregation_nodes": {"aggregation_material_epoch_key"},
        "aggregation_publication_state": {"aggregation_material_epoch_key"},
        "aggregation_build_health": {
            "last_success_material_epoch_key", "pending_material_epoch_key",
            "last_failure_material_epoch_key",
        },
        "dream_runs": {"aggregation_material_epoch_key"},
    }
    for table, columns in expected_material_fks.items():
        fks = {
            (str(row["from"]), str(row["table"]), str(row["to"]),
             str(row["on_delete"]))
            for row in conn.execute(f"PRAGMA foreign_key_list({table})")
        }
        if any(
            (column, "aggregation_material_epochs", "material_epoch_key", "RESTRICT")
            not in fks for column in columns
        ):
            return False

    # Exact registry/clock SQL and index inventory catch weakened CHECKs,
    # substituted uniqueness, or lookalike tables.
    for table in ("aggregation_material_clock", "aggregation_material_epochs"):
        expected = next(
            statement for statement in statements
            if re.match(rf"\s*CREATE\s+TABLE\s+{table}\b", statement, re.I)
        )
        actual = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if actual is None or normalized(actual["sql"]) != normalized(expected):
            return False
    clock_indexes = conn.execute(
        "PRAGMA index_list(aggregation_material_clock)"
    ).fetchall()
    if (
        len(clock_indexes) != 0
        and not (
            len(clock_indexes) == 1
            and int(clock_indexes[0]["unique"]) == 1
            and str(clock_indexes[0]["origin"]) == "pk"
        )
    ):
        return False
    registry_indexes = conn.execute(
        "PRAGMA index_list(aggregation_material_epochs)"
    ).fetchall()
    if (
        len(registry_indexes) != 1
        or int(registry_indexes[0]["unique"]) != 1
        or str(registry_indexes[0]["origin"]) != "pk"
        or tuple(
            str(row["name"])
            for row in conn.execute(
                f"PRAGMA index_info('{registry_indexes[0]['name']}')"
            )
        ) != ("material_epoch_key",)
    ):
        return False
    return True


def _v57_domain_footprint_present(conn: sqlite3.Connection) -> bool:
    marker = conn.execute(
        "SELECT value FROM schema_meta WHERE "
        "key='aggregation_material_epoch_schema'"
    ).fetchone()
    if marker is not None:
        return True
    if any(_table_exists(conn, table) for table in (
        "aggregation_material_clock", "aggregation_material_epochs",
    )):
        return True
    return any(
        _table_exists(conn, table) and column in {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        for table, column in (
            ("episode_embeddings", "embedding_producer_key"),
            ("aggregation_nodes", "aggregation_material_epoch_key"),
            ("aggregation_publication_state", "material_revision"),
            ("aggregation_build_health", "pending_material_epoch_key"),
            ("dream_runs", "aggregation_material_epoch_key"),
        )
    )


def _v56_domain_footprint_present(conn: sqlite3.Connection) -> bool:
    marker = conn.execute(
        "SELECT value FROM schema_meta WHERE key='aggregation_generation_schema'"
    ).fetchone()
    if marker is not None:
        return True
    if _table_exists(conn, "aggregation_generations"):
        return True
    for table, column in (
        ("aggregation_nodes", "aggregation_generation_key"),
        ("aggregation_nodes", "aggregation_request_hash"),
        ("aggregation_publication_state", "aggregation_generation_key"),
        ("aggregation_publication_state", "request_contract_sha256"),
        ("aggregation_build_health", "pending_generation_key"),
        ("dream_runs", "aggregation_generation_key"),
    ):
        if _table_exists(conn, table) and column in {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }:
            return True
    return False


def _v54_domain_footprint_present(conn: sqlite3.Connection) -> bool:
    """Distinguish a sparse test fixture from a damaged stamped v54 store."""

    if _v53_generation_bindings_present(conn):
        return True
    if any(
        _table_exists(conn, table)
        for table in (
            "phase1_auxiliary_outcomes", "entity_type_observations",
            "entity_property_observations", "entity_mention_observations",
            "profile_entry_marker_evidence", "profile_marker_decisions",
            "rule_marker_evidence", "rule_marker_decisions",
        )
    ):
        return True
    for table, column in (
        ("entity_types", "origin"),
        ("entity_properties", "origin"),
        ("profile_entries", "source"),
    ):
        if _table_exists(conn, table) and column in {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }:
            return True
    return False


def _v54_auxiliary_tables_present(conn: sqlite3.Connection) -> bool:
    return all(
        _table_exists(conn, table)
        for table in (
            "phase1_auxiliary_outcomes", "entity_type_observations",
            "entity_property_observations", "entity_mention_observations",
            "profile_entry_marker_evidence", "profile_marker_decisions",
            "rule_marker_evidence", "rule_marker_decisions",
        )
    )


_V54_TABLE_SHAPES: dict[str, tuple[tuple[object, ...], ...]] = {
    "entity_types": (
        ("entity_canonical", "TEXT", 1, None, 1),
        ("type", "TEXT", 1, None, 2),
        ("confidence", "REAL", 1, "1.0", 0),
        ("source_chunk_id", "TEXT", 0, None, 0),
        ("origin", "TEXT", 1, "'legacy_unattributed'", 0),
    ),
    "entity_properties": (
        ("entity_canonical", "TEXT", 1, None, 1),
        ("key", "TEXT", 1, None, 2),
        ("value", "TEXT", 1, None, 0),
        ("source_chunk_id", "TEXT", 0, None, 0),
        ("updated_at", "TIMESTAMP", 0, "CURRENT_TIMESTAMP", 0),
        ("origin", "TEXT", 1, "'legacy_unattributed'", 0),
    ),
    "profile_entries": (
        ("id", "INTEGER", 0, None, 1),
        ("kind", "TEXT", 1, None, 0),
        ("text", "TEXT", 1, None, 0),
        ("pos_evidence", "INTEGER", 1, "1", 0),
        ("neg_evidence", "INTEGER", 1, "0", 0),
        ("first_seen", "TIMESTAMP", 0, "CURRENT_TIMESTAMP", 0),
        ("last_updated", "TIMESTAMP", 0, "CURRENT_TIMESTAMP", 0),
        ("source", "TEXT", 1, "'legacy_unattributed'", 0),
    ),
    "rules": (
        ("id", "INTEGER", 0, None, 1),
        ("text", "TEXT", 1, None, 0),
        ("scope", "TEXT", 1, "'always_on'", 0),
        ("trigger_entities", "TEXT", 1, "'[]'", 0),
        ("source", "TEXT", 1, "'user'", 0),
        ("pos_evidence", "INTEGER", 1, "1", 0),
        ("neg_evidence", "INTEGER", 1, "0", 0),
        ("valid_at", "TIMESTAMP", 0, None, 0),
        ("invalid_at", "TIMESTAMP", 0, None, 0),
        ("status", "TEXT", 1, "'active'", 0),
        ("created_at", "TIMESTAMP", 0, "CURRENT_TIMESTAMP", 0),
    ),
    "behavioral_markers": (
        ("id", "INTEGER", 0, None, 1),
        ("kind", "TEXT", 1, None, 0),
        ("statement", "TEXT", 1, None, 0),
        ("chunk_id", "TEXT", 1, None, 0),
        ("created_at", "TIMESTAMP", 0, "CURRENT_TIMESTAMP", 0),
        ("consolidated_at", "TIMESTAMP", 0, None, 0),
        ("phase1_generation_key", "TEXT", 0, None, 0),
    ),
    "phase1_auxiliary_outcomes": (
        ("chunk_id", "TEXT", 1, None, 1),
        ("phase1_generation_key", "TEXT", 1, None, 2),
        ("extraction_cache_key", "TEXT", 1, None, 0),
        ("auxiliary_contract_key", "TEXT", 1, None, 0),
        ("result_hash", "TEXT", 1, None, 0),
        ("entity_type_count", "INTEGER", 1, None, 0),
        ("entity_property_count", "INTEGER", 1, None, 0),
        ("entity_mention_count", "INTEGER", 1, None, 0),
        ("marker_count", "INTEGER", 1, None, 0),
        ("published_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
    ),
    "entity_type_observations": (
        ("chunk_id", "TEXT", 1, None, 1),
        ("entity_canonical", "TEXT", 1, None, 2),
        ("type", "TEXT", 1, None, 3),
        ("confidence", "REAL", 1, "1.0", 0),
        ("phase1_generation_key", "TEXT", 1, None, 4),
        ("observed_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
    ),
    "entity_property_observations": (
        ("chunk_id", "TEXT", 1, None, 1),
        ("entity_canonical", "TEXT", 1, None, 2),
        ("key", "TEXT", 1, None, 3),
        ("value", "TEXT", 1, None, 0),
        ("phase1_generation_key", "TEXT", 1, None, 4),
        ("observed_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
    ),
    "entity_mention_observations": (
        ("chunk_id", "TEXT", 1, None, 1),
        ("entity_canonical", "TEXT", 1, None, 2),
        ("phase1_generation_key", "TEXT", 1, None, 3),
        ("observed_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
    ),
    "profile_entry_marker_evidence": (
        ("profile_entry_id", "INTEGER", 1, None, 1),
        ("marker_id", "INTEGER", 1, None, 2),
        ("phase1_generation_key", "TEXT", 1, None, 0),
        ("created_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
    ),
    "profile_marker_decisions": (
        ("marker_id", "INTEGER", 0, None, 1),
        ("phase1_generation_key", "TEXT", 1, None, 0),
        ("profile_policy_key", "TEXT", 1, None, 0),
        ("decision", "TEXT", 1, None, 0),
        ("profile_entry_id", "INTEGER", 1, None, 0),
        ("decided_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
    ),
    "rule_marker_evidence": (
        ("rule_id", "INTEGER", 1, None, 1),
        ("marker_id", "INTEGER", 1, None, 2),
        ("phase1_generation_key", "TEXT", 1, None, 0),
        ("created_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
    ),
    "rule_marker_decisions": (
        ("marker_id", "INTEGER", 0, None, 1),
        ("phase1_generation_key", "TEXT", 1, None, 0),
        ("routing_key", "TEXT", 1, None, 0),
        ("decision", "TEXT", 1, None, 0),
        ("rule_id", "INTEGER", 0, None, 0),
        ("decided_at", "TIMESTAMP", 1, "CURRENT_TIMESTAMP", 0),
    ),
}

_V54_FOREIGN_KEYS: dict[str, frozenset[tuple[str, ...]]] = {
    "entity_types": frozenset({
        ("source_chunk_id", "chunks", "id", "NO ACTION", "SET NULL", "NONE"),
    }),
    "entity_properties": frozenset({
        ("source_chunk_id", "chunks", "id", "NO ACTION", "SET NULL", "NONE"),
    }),
    "profile_entries": frozenset(),
    "rules": frozenset(),
    "behavioral_markers": frozenset({
        ("chunk_id", "chunks", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
    }),
    "phase1_auxiliary_outcomes": frozenset({
        ("chunk_id", "chunks", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
    }),
    "entity_type_observations": frozenset({
        ("chunk_id", "chunks", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
    }),
    "entity_property_observations": frozenset({
        ("chunk_id", "chunks", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
    }),
    "entity_mention_observations": frozenset({
        ("chunk_id", "chunks", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
    }),
    "profile_entry_marker_evidence": frozenset({
        ("profile_entry_id", "profile_entries", "id", "NO ACTION", "CASCADE", "NONE"),
        ("marker_id", "behavioral_markers", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
    }),
    "profile_marker_decisions": frozenset({
        ("marker_id", "behavioral_markers", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
        ("profile_entry_id", "profile_entries", "id", "NO ACTION", "CASCADE", "NONE"),
    }),
    "rule_marker_evidence": frozenset({
        ("rule_id", "rules", "id", "NO ACTION", "CASCADE", "NONE"),
        ("marker_id", "behavioral_markers", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
    }),
    "rule_marker_decisions": frozenset({
        ("marker_id", "behavioral_markers", "id", "NO ACTION", "CASCADE", "NONE"),
        ("phase1_generation_key", "phase1_generations", "generation_key",
         "NO ACTION", "RESTRICT", "NONE"),
        ("rule_id", "rules", "id", "NO ACTION", "SET NULL", "NONE"),
    }),
}


# These indexes are part of v54's authority boundary, rather than optional
# query tuning.  In particular, the two marker-evidence indexes make a
# producer marker's Phase-2 decision singular and the partial behavioral
# marker index makes exact Phase-1 replay idempotent.  Keep the complete
# expected shape here so both the pre-stamp migration check and startup healer
# use the same definition.
_V54_INDEX_SPECS: tuple[
    tuple[str, str, tuple[str, ...], int, int, str | None, str], ...
] = (
    (
        "idx_phase1_auxiliary_generation", "phase1_auxiliary_outcomes",
        ("phase1_generation_key", "chunk_id"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_phase1_auxiliary_generation "
        "ON phase1_auxiliary_outcomes(phase1_generation_key,chunk_id)",
    ),
    (
        "idx_entity_type_observations_lookup", "entity_type_observations",
        ("type", "entity_canonical", "phase1_generation_key"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_entity_type_observations_lookup "
        "ON entity_type_observations(type,entity_canonical,phase1_generation_key)",
    ),
    (
        "idx_entity_type_observations_entity", "entity_type_observations",
        ("entity_canonical", "phase1_generation_key"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_entity_type_observations_entity "
        "ON entity_type_observations(entity_canonical,phase1_generation_key)",
    ),
    (
        "idx_entity_property_observations_lookup",
        "entity_property_observations",
        ("key", "value", "entity_canonical", "phase1_generation_key"),
        0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_entity_property_observations_lookup "
        "ON entity_property_observations(key,value,entity_canonical,phase1_generation_key)",
    ),
    (
        "idx_entity_property_observations_entity",
        "entity_property_observations",
        ("entity_canonical", "key", "phase1_generation_key"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_entity_property_observations_entity "
        "ON entity_property_observations(entity_canonical,key,phase1_generation_key)",
    ),
    (
        "idx_entity_mention_observations_entity",
        "entity_mention_observations",
        ("entity_canonical", "phase1_generation_key"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_entity_mention_observations_entity "
        "ON entity_mention_observations(entity_canonical,phase1_generation_key)",
    ),
    (
        "idx_profile_marker_generation", "profile_entry_marker_evidence",
        ("phase1_generation_key", "marker_id"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_profile_marker_generation "
        "ON profile_entry_marker_evidence(phase1_generation_key,marker_id)",
    ),
    (
        "idx_profile_marker_one_decision", "profile_entry_marker_evidence",
        ("marker_id",), 1, 0, None,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_marker_one_decision "
        "ON profile_entry_marker_evidence(marker_id)",
    ),
    (
        "idx_profile_marker_decision_generation", "profile_marker_decisions",
        ("phase1_generation_key", "marker_id"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_profile_marker_decision_generation "
        "ON profile_marker_decisions(phase1_generation_key,marker_id)",
    ),
    (
        "idx_rule_marker_generation", "rule_marker_evidence",
        ("phase1_generation_key", "marker_id"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_rule_marker_generation "
        "ON rule_marker_evidence(phase1_generation_key,marker_id)",
    ),
    (
        "idx_rule_marker_one_decision", "rule_marker_evidence",
        ("marker_id",), 1, 0, None,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_rule_marker_one_decision "
        "ON rule_marker_evidence(marker_id)",
    ),
    (
        "idx_rule_marker_decision_generation", "rule_marker_decisions",
        ("phase1_generation_key", "marker_id"), 0, 0, None,
        "CREATE INDEX IF NOT EXISTS idx_rule_marker_decision_generation "
        "ON rule_marker_decisions(phase1_generation_key,marker_id)",
    ),
    (
        "idx_behavioral_marker_generation_identity", "behavioral_markers",
        ("chunk_id", "phase1_generation_key", "kind", "statement"),
        1, 1, "phase1_generation_keyisnotnull",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_behavioral_marker_generation_identity "
        "ON behavioral_markers(chunk_id,phase1_generation_key,kind,statement) "
        "WHERE phase1_generation_key IS NOT NULL",
    ),
)


def _index_xinfo_shape_is_exact(
    conn: sqlite3.Connection, name: str, columns: tuple[str, ...],
) -> bool:
    xinfo = conn.execute(f'PRAGMA index_xinfo("{name}")').fetchall()
    key_shape = tuple(
        (str(row["name"]), int(row["desc"]), str(row["coll"]))
        for row in xinfo if int(row["key"]) == 1
    )
    if key_shape != tuple((column, 0, "BINARY") for column in columns):
        return False
    # A normal rowid-table index has one non-key rowid payload entry.  Requiring
    # this also rejects expression indexes which can report deceptively similar
    # names via less exact PRAGMA surfaces on older SQLite builds.
    non_key_shape = tuple(
        (int(row["cid"]), row["name"], int(row["desc"]), str(row["coll"]))
        for row in xinfo if int(row["key"]) == 0
    )
    return non_key_shape == ((-1, None, 0, "BINARY"),)


def _v54_index_is_exact(
    conn: sqlite3.Connection,
    spec: tuple[str, str, tuple[str, ...], int, int, str | None, str],
) -> bool:
    name, table, columns, unique, partial, predicate, _statement = spec
    metadata = next((
        row for row in conn.execute(f"PRAGMA index_list({table})").fetchall()
        if str(row["name"]) == name
    ), None)
    if metadata is None:
        return False
    if int(metadata["unique"]) != unique or int(metadata["partial"]) != partial:
        return False
    if not _index_xinfo_shape_is_exact(conn, name, columns):
        return False
    sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,),
    ).fetchone()
    if sql_row is None or sql_row["sql"] is None:
        return False
    ddl = re.sub(r"/\*.*?\*/", "", str(sql_row["sql"]), flags=re.DOTALL)
    ddl = re.sub(r"--[^\r\n]*", "", ddl)
    normalized_sql = "".join(ddl.lower().split()).replace('"', "")
    actual_predicate = normalized_sql.partition("where")[2] or None
    return actual_predicate == predicate


def _v54_table_shape_is_exact(conn: sqlite3.Connection, table: str) -> bool:
    xinfo = conn.execute(f"PRAGMA table_xinfo({table})").fetchall()
    # table_info omits generated/hidden columns. Such a column can impose a
    # narrowing NOT NULL/generated constraint while the visible authority
    # shape looks canonical, so require an entirely ordinary column set.
    if any(int(row["hidden"] or 0) != 0 for row in xinfo):
        return False
    actual = tuple(
        (str(row["name"]), str(row["type"]).upper(), int(row["notnull"]),
         row["dflt_value"], int(row["pk"]))
        for row in xinfo
    )
    if actual != _V54_TABLE_SHAPES[table]:
        return False
    foreign_keys = frozenset(
        (
            str(row["from"]), str(row["table"]), str(row["to"]),
            str(row["on_update"]), str(row["on_delete"]), str(row["match"]),
        )
        for row in conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    )
    if foreign_keys != _V54_FOREIGN_KEYS[table]:
        return False
    pk_columns = tuple(
        str(row["name"])
        for row in sorted(
            conn.execute(f"PRAGMA table_info({table})").fetchall(),
            key=lambda item: int(item["pk"]) or 2**31,
        )
        if int(row["pk"]) > 0
    )
    # INTEGER PRIMARY KEY aliases rowid and has no backing index. Every other
    # v54 primary identity must be ASC/BINARY exactly; COLLATE NOCASE changes
    # replay identity even though table_info reports the same columns.
    integer_rowid_pk = bool(
        len(pk_columns) == 1
        and next(
            str(row["type"]).upper()
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            if str(row["name"]) == pk_columns[0]
        ) == "INTEGER"
    )
    if pk_columns and not integer_rowid_pk:
        pk_indexes = [
            row for row in conn.execute(f"PRAGMA index_list({table})").fetchall()
            if str(row["origin"]) == "pk"
        ]
        if (
            len(pk_indexes) != 1
            or int(pk_indexes[0]["unique"]) != 1
            or int(pk_indexes[0]["partial"]) != 0
            or not _index_xinfo_shape_is_exact(
                conn, str(pk_indexes[0]["name"]), pk_columns,
            )
        ):
            return False
    sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone()
    ddl = str(sql_row["sql"] or "") if sql_row is not None else ""
    # A lookalike table must not satisfy an authority constraint by placing
    # its spelling in a SQL comment.  sqlite_master preserves comments from
    # CREATE TABLE, so remove both forms before matching the canonical checks.
    ddl = re.sub(r"/\*.*?\*/", "", ddl, flags=re.DOTALL)
    ddl = re.sub(r"--[^\r\n]*", "", ddl)
    normalized_sql = "".join(ddl.lower().split())
    required_checks = {
        "entity_types": (
            "check(originin('user','legacy_unattributed'))",
        ),
        "entity_properties": (
            "check(originin('user','legacy_unattributed'))",
        ),
        "profile_entries": (
            "check(kindin('preference','avoidance','style','context'))",
            "check(sourcein('user','agent_inferred','legacy_unattributed'))",
        ),
        "rules": (
            "check(scopein('always_on','contextual'))",
            "check(sourcein('user','agent_inferred'))",
            "check(statusin('active','retracted'))",
        ),
        "behavioral_markers": (
            "check(kindin('correction','preference','rejection','style'))",
        ),
        "phase1_auxiliary_outcomes": (
            "check(substr(result_hash,1,7)='sha256:'andlength(result_hash)=71"
            "andsubstr(result_hash,8)notglob'*[^0-9a-f]*')",
            "check(entity_type_count>=0)",
            "check(entity_property_count>=0)",
            "check(entity_mention_count>=0)",
            "check(marker_count>=0)",
        ),
        "entity_type_observations": (
            "check(length(trim(entity_canonical))>0andhymem_entity_canonical_is_normalized(entity_canonical)=1)",
            "check(length(trim(type))>0)",
            "check(typeof(confidence)in('integer','real')andconfidence>=0.0andconfidence<=1.0)",
        ),
        "entity_property_observations": (
            "check(length(trim(entity_canonical))>0andhymem_entity_canonical_is_normalized(entity_canonical)=1)",
            "check(length(trim(key))>0)",
        ),
        "entity_mention_observations": (
            "check(length(trim(entity_canonical))>0andhymem_entity_canonical_is_normalized(entity_canonical)=1)",
        ),
        "profile_marker_decisions": (
            "check(decisionin('materialized','manual_authority','identity_conflict'))",
            "check(length(trim(profile_policy_key))>0)",
        ),
        "rule_marker_decisions": (
            "check(decisionin('routed','no_rule'))",
            "check((decision='routed'andrule_idisnotnull)or(decision='no_rule'andrule_idisnull))",
            "check(length(trim(routing_key))>0)",
        ),
    }
    expected_checks = required_checks.get(table, ())
    # Required-fragment checks alone are fail-open: a lookalike can retain all
    # canonical constraints and append a narrowing CHECK (for example
    # ``type='person'``), silently changing which exact producer projections
    # can be replayed.  Reject both missing and additional CHECK clauses.
    return (
        "collate" not in normalized_sql
        and normalized_sql.count("check(") == len(expected_checks)
        and all(fragment in normalized_sql for fragment in expected_checks)
    )


def _v54_triggers_are_compatible(conn: sqlite3.Connection) -> bool:
    """Reject unowned mutation hooks on producer-authority tables.

    Known v54 guards may be absent or stale on a stamped store because startup
    deliberately heals them from the migration source.  Unknown hooks cannot
    be healed safely: they run inside the publisher's private mutation window
    and could alter an authenticated projection after its hash is computed.
    """
    script = files("hymem.core.migrations").joinpath(
        "054_phase1_auxiliary_provenance.sql"
    ).read_text(encoding="utf-8")
    expected = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"CREATE\s+TRIGGER\s+IF\s+NOT\s+EXISTS\s+(\w+)\s+"
            r"(?:BEFORE|AFTER)\s+.*?\s+ON\s+(\w+)\b",
            script,
            flags=re.IGNORECASE | re.DOTALL,
        )
    }
    authority_tables = tuple(_V54_TABLE_SHAPES)
    placeholders = ",".join("?" for _ in authority_tables)
    actual = conn.execute(
        "SELECT name,tbl_name FROM sqlite_master WHERE type='trigger' "
        f"AND tbl_name IN ({placeholders})",
        authority_tables,
    ).fetchall()
    return all(
        expected.get(str(row["name"])) == str(row["tbl_name"])
        for row in actual
    )


def _v54_auxiliary_bindings_present(
    conn: sqlite3.Connection, *, include_indexes: bool = True,
) -> bool:
    if not _v54_domain_present(conn):
        return False
    required_tables = (
        "phase1_auxiliary_outcomes", "entity_type_observations",
        "entity_property_observations", "entity_mention_observations",
        "profile_entry_marker_evidence",
        "profile_marker_decisions", "rule_marker_evidence",
        "rule_marker_decisions",
    )
    if not all(_table_exists(conn, table) for table in required_tables):
        return False
    if not all(
        _v54_table_shape_is_exact(conn, table) for table in _V54_TABLE_SHAPES
    ):
        return False
    if not _v54_triggers_are_compatible(conn):
        return False
    required_columns = {
        "entity_types": {"origin"},
        "entity_properties": {"origin"},
        "profile_entries": {"source"},
        "phase1_auxiliary_outcomes": {
            "chunk_id", "phase1_generation_key", "extraction_cache_key",
            "auxiliary_contract_key", "result_hash", "entity_type_count",
            "entity_property_count", "entity_mention_count", "marker_count",
            "published_at",
        },
        "entity_type_observations": {
            "chunk_id", "entity_canonical", "type", "confidence",
            "phase1_generation_key", "observed_at",
        },
        "entity_property_observations": {
            "chunk_id", "entity_canonical", "key", "value",
            "phase1_generation_key", "observed_at",
        },
        "entity_mention_observations": {
            "chunk_id", "entity_canonical", "phase1_generation_key",
            "observed_at",
        },
        "profile_entry_marker_evidence": {
            "profile_entry_id", "marker_id", "phase1_generation_key",
            "created_at",
        },
        "profile_marker_decisions": {
            "marker_id", "phase1_generation_key", "profile_policy_key",
            "decision", "profile_entry_id", "decided_at",
        },
        "rule_marker_evidence": {
            "rule_id", "marker_id", "phase1_generation_key", "created_at",
        },
        "rule_marker_decisions": {
            "marker_id", "phase1_generation_key", "routing_key", "decision",
            "rule_id", "decided_at",
        },
    }
    columns_ok = all(
        columns.issubset({
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        })
        for table, columns in required_columns.items()
    )
    if not columns_ok:
        return False
    for table, columns in (
        ("profile_entries", ("text",)), ("rules", ("text",)),
    ):
        unique_indexes = [
            row for row in conn.execute(f"PRAGMA index_list({table})").fetchall()
            if int(row["unique"]) == 1
        ]
        if (
            len(unique_indexes) != 1
            or int(unique_indexes[0]["partial"]) != 0
            or str(unique_indexes[0]["origin"]) != "u"
            or not _index_xinfo_shape_is_exact(
                conn, str(unique_indexes[0]["name"]), columns,
            )
        ):
            return False
    allowed_named_unique = {
        (str(spec[1]), str(spec[0]))
        for spec in _V54_INDEX_SPECS if int(spec[3]) == 1
    }
    for table in _V54_TABLE_SHAPES:
        for row in conn.execute(f"PRAGMA index_list({table})").fetchall():
            if int(row["unique"]) != 1:
                continue
            origin = str(row["origin"])
            name = str(row["name"])
            if origin == "pk":
                continue
            if origin == "u" and table in {"profile_entries", "rules"}:
                # The preceding exact-one check validates its BINARY/ASC text
                # identity, so this is the single canonical table constraint.
                continue
            if origin == "c" and (table, name) in allowed_named_unique:
                continue
            # Extra uniqueness narrows replay identity despite every canonical
            # index still being present; it is authority-shape corruption.
            return False
    return (
        not include_indexes
        or all(_v54_index_is_exact(conn, spec) for spec in _V54_INDEX_SPECS)
    )


def register_read_authority_functions(conn: sqlite3.Connection) -> None:
    """Install connection-local functions used by authoritative read SQL.

    SQLite UDFs do not travel with a database file.  Core connections and
    read-only benchmark/probe connections must therefore install the same
    timestamp and Phase-1 producer authority predicates before executing
    ``live_edge_predicate()`` or other publication-aware selectors.
    """

    from hymem.extraction.producer import (
        phase1_generation_runtime_authorized,
        producer_generation_runtime_authorized,
    )
    from hymem.dreaming.canonicalize import normalize as normalize_entity

    conn.create_function(
        "hymem_phase1_generation_is_authorized",
        2,
        phase1_generation_runtime_authorized,
        deterministic=False,
    )
    conn.create_function(
        "hymem_producer_generation_is_authorized",
        2,
        producer_generation_runtime_authorized,
        deterministic=False,
    )
    conn.create_function(
        "hymem_entity_canonical_is_normalized",
        1,
        lambda value: int(
            isinstance(value, str)
            and bool(value)
            and normalize_entity(value) == value
        ),
        deterministic=True,
    )
    register_current_phase1_generation(conn, None)
    register_current_rule_routing(conn, None)
    register_current_profile_policy(conn)
    register_current_auxiliary_contract(conn)
    register_sqlite_time_functions(conn)


def register_current_phase1_generation(
    conn: sqlite3.Connection, generation_key: str | None,
) -> None:
    """Scope behavioral reads on ``conn`` to one selected Phase-1 producer.

    ``None`` is the neutral low-level/read-only posture: every otherwise valid
    durable generation remains visible. A configured :class:`HyMem` instance
    installs its selected key, so preserved A evidence is not silently exposed
    as B output while B is pending or failed.
    """

    from hymem.extraction.producer import phase1_generation_runtime_authorized

    if generation_key is not None and not isinstance(generation_key, str):
        raise TypeError("Phase-1 generation key must be text or None")

    def current(key: object, identity_exact: object) -> int:
        return int(
            phase1_generation_runtime_authorized(key, identity_exact) == 1
            and (generation_key is None or key == generation_key)
        )

    conn.create_function(
        "hymem_phase1_generation_is_current",
        2,
        current,
        deterministic=False,
    )


def register_current_rule_routing(
    conn: sqlite3.Connection, routing_key: str | None,
) -> None:
    """Select the exact marker-to-rule producer for authoritative reads.

    ``None`` is the neutral low-level posture and accepts the single physical
    completed decision retained for each otherwise-current marker. Configured
    HyMem connections pass their exact policy/model key and therefore fail
    closed while a changed router is pending or failed.
    """

    if routing_key is not None and not isinstance(routing_key, str):
        raise TypeError("rule routing key must be text or None")

    conn.create_function(
        "hymem_rule_routing_is_current",
        1,
        lambda key: int(routing_key is None or key == routing_key),
        deterministic=False,
    )


def register_current_profile_policy(conn: sqlite3.Connection) -> None:
    """Register the deterministic marker-to-profile implementation identity."""

    from hymem.dreaming.phase2 import (
        PROFILE_MATERIALIZATION_POLICY_KEY,
        validate_profile_materialization_policy,
    )

    validate_profile_materialization_policy()

    conn.create_function(
        "hymem_profile_policy_is_current",
        1,
        lambda key: int(key == PROFILE_MATERIALIZATION_POLICY_KEY),
        deterministic=False,
    )


def register_current_auxiliary_contract(conn: sqlite3.Connection) -> None:
    """Register the current whole-response auxiliary publication contract."""

    from hymem.dreaming.phase1_auxiliary import (
        CURRENT_AUXILIARY_CONTRACT_KEY,
        validate_auxiliary_contract_implementation,
    )

    validate_auxiliary_contract_implementation()

    conn.create_function(
        "hymem_auxiliary_contract_is_current",
        1,
        lambda key: int(key == CURRENT_AUXILIARY_CONTRACT_KEY),
        deterministic=False,
    )


def connect(path: Path) -> sqlite3.Connection:
    from hymem.dreaming.lossless import coverage_chunk_id
    from hymem.extraction.producer import (
        phase1_generation_registry_row_is_valid,
    )
    from hymem.dreaming.aggregation_generation import (
        aggregation_generation_registry_row_is_valid,
    )
    from hymem.dreaming.aggregation_material import (
        aggregation_material_registry_row_is_valid,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    register_read_authority_functions(conn)
    conn.create_function(
        "hymem_phase1_generation_registry_row_is_valid",
        6,
        phase1_generation_registry_row_is_valid,
        deterministic=True,
    )
    conn.create_function(
        "hymem_aggregation_generation_registry_row_is_valid",
        6,
        aggregation_generation_registry_row_is_valid,
        deterministic=True,
    )
    conn.create_function(
        "hymem_aggregation_material_registry_row_is_valid",
        8,
        aggregation_material_registry_row_is_valid,
        deterministic=True,
    )
    # Used by the v37 coverage-ledger guard triggers. Keeping the hash function
    # in SQLite makes direct SQL obey the same source-lifecycle invariant as
    # the Python API: durable proof cannot be dropped after raw deletion.
    conn.create_function(
        "hymem_message_content_hash",
        2,
        message_content_hash,
        deterministic=True,
    )
    conn.create_function(
        "hymem_message_record_proof_valid",
        4,
        message_record_proof_valid,
        deterministic=True,
    )
    conn.create_function(
        "hymem_message_record_matches_source",
        10,
        message_record_matches_source,
        deterministic=True,
    )
    conn.create_function(
        "hymem_message_record_matches_raw_source",
        11,
        message_record_matches_raw_source,
        deterministic=True,
    )
    conn.create_function(
        "hymem_evidence_mutation_authorized",
        0,
        lambda: 1 if _connection_authority_key(conn) in _EVIDENCE_MUTATION_KEYS.get() else 0,
        deterministic=False,
    )
    conn.create_function(
        "hymem_evidence_history_authorized",
        0,
        lambda: 1 if _connection_authority_key(conn) in _EVIDENCE_HISTORY_KEYS.get() else 0,
        deterministic=False,
    )
    conn.create_function(
        "hymem_evidence_destructive_authorized",
        0,
        lambda: 1
        if _connection_authority_key(conn) in _EVIDENCE_DESTRUCTIVE_KEYS.get()
        else 0,
        deterministic=False,
    )
    conn.create_function(
        "hymem_phase1_generation_prune_authorized",
        0,
        lambda: 1
        if _connection_authority_key(conn)
        in _PHASE1_GENERATION_PRUNE_KEYS.get()
        else 0,
        deterministic=False,
    )
    conn.create_function(
        "hymem_embedding_mutation_authorized",
        0,
        lambda: 1
        if _connection_authority_key(conn) in _EMBEDDING_MUTATION_KEYS.get()
        else 0,
        deterministic=False,
    )
    conn.create_function(
        "hymem_message_record",
        3,
        lambda message_id, role, content: encode_message_record(
            message_id=int(message_id), role=str(role), content=str(content)
        ),
        deterministic=True,
    )
    conn.create_function(
        "hymem_coverage_chunk_id",
        2,
        lambda session_id, message_id: coverage_chunk_id(
            str(session_id), int(message_id)
        ),
        deterministic=True,
    )
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    # WAL is set here (not just in schema.sql) so it is active before any
    # schema creation or migration runs. journal_mode persists on the file;
    # synchronous is per-connection and must be set every time.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    # Derived embedding routes were historically stored in cache labels. v57
    # physically scrubs those rows; secure deletion must be active before its
    # migration DELETE so removed cell payloads are not left in database pages.
    conn.execute("PRAGMA secure_delete = ON")
    secure_delete = conn.execute("PRAGMA secure_delete").fetchone()
    if secure_delete is None or int(secure_delete[0]) != 1:
        raise RuntimeError("SQLite secure_delete could not be enabled")
    return conn


@contextlib.contextmanager
def read_snapshot(path: Path) -> Iterator[sqlite3.Connection]:
    """Open one isolated, query-only SQLite snapshot and close it promptly.

    Status polling must not BEGIN/COMMIT on HyMem's shared read connection:
    another thread may own its transaction. A dedicated read-only connection
    gives every multi-query health report one coherent WAL snapshot without
    acquiring a writer lock. Read-side predicates still receive their pure
    timestamp and Phase-1 generation-authorization functions.
    """
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(
        uri,
        uri=True,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    register_read_authority_functions(conn)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException as primary:
        for label, cleanup in (
            ("snapshot rollback", lambda: conn.execute("ROLLBACK") if conn.in_transaction else None),
            ("snapshot close", conn.close),
        ):
            try:
                cleanup()
            except BaseException as cleanup_error:
                with contextlib.suppress(AttributeError, TypeError):
                    primary.add_note(
                        f"{label} failed: {type(cleanup_error).__name__}"
                    )
        raise
    else:
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        finally:
            conn.close()


@contextlib.contextmanager
def evidence_mutation(conn: sqlite3.Connection) -> Iterator[None]:
    """Authorize one tightly scoped internal evidence/lifecycle rewrite."""
    key = _connection_authority_key(conn)
    token = _EVIDENCE_MUTATION_KEYS.set(_EVIDENCE_MUTATION_KEYS.get() | {key})
    try:
        yield
    finally:
        _EVIDENCE_MUTATION_KEYS.reset(token)


@contextlib.contextmanager
def embedding_mutation(conn: sqlite3.Connection) -> Iterator[None]:
    """Authorize one internal durable embedding-cache/mirror write."""

    key = _connection_authority_key(conn)
    token = _EMBEDDING_MUTATION_KEYS.set(
        _EMBEDDING_MUTATION_KEYS.get() | {key}
    )
    try:
        yield
    finally:
        _EMBEDDING_MUTATION_KEYS.reset(token)


def embedding_writer(function):
    """Wrap a maintained persistence function in embedding write authority."""

    @functools.wraps(function)
    def authorized(conn: sqlite3.Connection, *args, **kwargs):
        with embedding_mutation(conn):
            return function(conn, *args, **kwargs)

    return authorized


@contextlib.contextmanager
def evidence_history_mutation(conn: sqlite3.Connection) -> Iterator[None]:
    """Authorize a validated restore of immutable historical ledger rows.

    Normal runtime writers may only cite current evidence. Portability and
    complete-history merge paths additionally need to restore references to
    retired revisions without pretending those revisions are current. This
    narrower flag is meaningful only while ``evidence_mutation`` is also held.
    """
    key = _connection_authority_key(conn)
    token = _EVIDENCE_HISTORY_KEYS.set(_EVIDENCE_HISTORY_KEYS.get() | {key})
    try:
        with evidence_mutation(conn):
            yield
    finally:
        _EVIDENCE_HISTORY_KEYS.reset(token)


@contextlib.contextmanager
def evidence_destructive_mutation(conn: sqlite3.Connection) -> Iterator[None]:
    """Authorize explicit opt-in destruction of published graph history."""
    key = _connection_authority_key(conn)
    token = _EVIDENCE_DESTRUCTIVE_KEYS.set(
        _EVIDENCE_DESTRUCTIVE_KEYS.get() | {key}
    )
    try:
        with evidence_mutation(conn):
            yield
    finally:
        _EVIDENCE_DESTRUCTIVE_KEYS.reset(token)


@contextlib.contextmanager
def _phase1_generation_pruning(
    conn: sqlite3.Connection,
) -> Iterator[None]:
    """Authorize only guarded removal of unreferenced inexact identities."""

    key = _connection_authority_key(conn)
    token = _PHASE1_GENERATION_PRUNE_KEYS.set(
        _PHASE1_GENERATION_PRUNE_KEYS.get() | {key}
    )
    try:
        yield
    finally:
        _PHASE1_GENERATION_PRUNE_KEYS.reset(token)


def prune_unreferenced_inexact_phase1_generations(
    conn: sqlite3.Connection,
    *,
    limit: int = 256,
) -> int:
    """Remove bounded process-only identities after every audit ref vanishes.

    Exact identities remain immutable forever. Inexact identities remain while
    any cache, attempt, outcome, observation, or marker references them. The
    private lexical authority and exact-row check keep arbitrary direct deletes
    fail closed; the five declared foreign keys independently reject referenced
    rows while this helper's selection proves them unreferenced up front.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Phase-1 generation prune limit must be positive")
    reference_tables = [
        "processed_chunks", "chunk_extraction_attempts",
        "kg_claim_extraction_outcomes", "kg_claim_observations",
        "behavioral_markers",
    ]
    reference_tables.extend(
        table for table in (
            "phase1_auxiliary_outcomes", "entity_type_observations",
            "entity_property_observations", "entity_mention_observations",
            "profile_entry_marker_evidence",
            "profile_marker_decisions", "rule_marker_evidence",
            "rule_marker_decisions",
        )
        if _table_exists(conn, table)
    )
    absence = " ".join(
        f"AND NOT EXISTS (SELECT 1 FROM {table} row "
        " WHERE row.phase1_generation_key=generation.generation_key)"
        for table in reference_tables
    )
    before = conn.total_changes
    with _phase1_generation_pruning(conn):
        conn.execute(
            "DELETE FROM phase1_generations WHERE generation_key IN ("
            "SELECT generation.generation_key FROM phase1_generations generation "
            "WHERE generation.identity_exact=0 "
            + absence + " "
            "ORDER BY generation.created_at,generation.generation_key LIMIT ?"
            ")",
            (limit,),
        )
    return conn.total_changes - before


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except ImportError:
        return False
    except Exception as exc:
        log.info("sqlite-vec failed to load (%s); using Python cosine search", exc)
        return False


def initialize(conn: sqlite3.Connection) -> None:
    # schema.sql is an additive bootstrap and would otherwise recreate a
    # deleted v54 authority table before integrity validation, concealing
    # material data loss. Sparse historical fixtures have no v54 footprint and
    # remain eligible for the compatibility bootstrap below.
    if _table_exists(conn, "schema_meta") and schema_version(conn) >= 54:
        v54_base_tables_present = all(
            _table_exists(conn, table)
            for table in (
                "entity_types", "entity_properties", "profile_entries",
                "rules", "behavioral_markers", "phase1_generations",
            )
        )
        if _v54_domain_footprint_present(conn) and (
            not v54_base_tables_present
            or not _v54_auxiliary_tables_present(conn)
        ):
            raise RuntimeError(
                "schema v54 producer-scoped auxiliary domain is incomplete"
            )
    if (
        _table_exists(conn, "schema_meta")
        and schema_version(conn) == 55
        and _v55_domain_footprint_present(conn)
        and not _v55_aggregation_storage_present(conn)
    ):
        raise RuntimeError("schema v55 aggregation provenance domain is incomplete")
    if (
        _table_exists(conn, "schema_meta")
        and schema_version(conn) >= 56
            and _v56_domain_footprint_present(conn)
            and not _v56_generation_storage_present(
                conn,
                allow_v57=_v57_domain_footprint_present(conn),
            )
    ):
        raise RuntimeError("schema v56 aggregation generation domain is incomplete")
    if (
        _table_exists(conn, "schema_meta")
        and schema_version(conn) >= 57
        and _v57_domain_footprint_present(conn)
        and not _v57_material_bindings_present(conn, validate_triggers=False)
    ):
        raise RuntimeError("schema v57 aggregation material domain is incomplete")
    conn.executescript(_load_schema())
    _load_vec_extension(conn)
    cur = schema_version(conn)
    if cur > EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {cur} is newer than code expects ({EXPECTED_SCHEMA_VERSION}). "
            f"Downgrading is not supported. Use a newer version of HyMem."
        )
    _run_migrations(conn)


def _install_evidence_revision_guards(conn: sqlite3.Connection) -> None:
    """Install the canonical v40 evidence guards from their owned migration.

    Schema v42 deliberately refreshes these two long provenance guards because
    their original timestamp comparison used SQLite's broader grammar. Keeping
    one source definition avoids a second large trigger copy drifting again.
    """
    script = files("hymem.core.migrations").joinpath(
        "040_claim_source_provenance.sql"
    ).read_text(encoding="utf-8")
    for name in (
        "kg_evidence_v40_insert_guard",
        "kg_evidence_v40_update_guard",
    ):
        marker = f"CREATE TRIGGER IF NOT EXISTS {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _install_evidence_publication_guards(conn: sqlite3.Connection) -> None:
    """Install/heal schema-42 publication guards from one SQL definition."""
    script = files("hymem.core.migrations").joinpath(
        "042_evidence_publication_clock.sql"
    ).read_text(encoding="utf-8")
    for name in (
        "kg_evidence_published_at_insert_guard",
        "kg_evidence_published_at_update_guard",
        "kg_evidence_v40_delete_guard",
        "kg_edge_lifecycle_update_guard",
        "kg_edge_lifecycle_delete_guard",
        "kg_lifecycle_dependencies_update_guard",
        "kg_lifecycle_dependencies_delete_guard",
    ):
        marker = f"CREATE TRIGGER {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _install_aggregation_source_guards(conn: sqlite3.Connection) -> None:
    """Install/heal the schema-appropriate aggregation proof boundary."""

    legacy_script = files("hymem.core.migrations").joinpath(
        "045_aggregation_source_provenance.sql"
    ).read_text(encoding="utf-8")
    episode_names = (
        "episode_source_header_insert_guard",
        "episode_source_header_update_guard",
        "episode_source_bound_update_guard",
        "episode_source_occurrence_insert_guard",
        "episode_source_occurrence_update_guard",
        "episode_source_occurrence_delete_unpublishes",
    )
    legacy_aggregation_names = (
        "aggregation_source_header_insert_guard",
        "aggregation_source_header_update_guard",
        "aggregation_source_bound_update_guard",
        "aggregation_source_occurrence_insert_guard",
        "aggregation_source_occurrence_update_guard",
        "aggregation_source_occurrence_delete_unpublishes",
    )
    for name in episode_names:
        marker = f"CREATE TRIGGER IF NOT EXISTS {name}"
        start = legacy_script.index(marker)
        end = legacy_script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(legacy_script[start:end])
    if schema_version(conn) < 55:
        for name in legacy_aggregation_names:
            marker = f"CREATE TRIGGER IF NOT EXISTS {name}"
            start = legacy_script.index(marker)
            end = legacy_script.index("\nEND;", start) + len("\nEND;")
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            conn.executescript(legacy_script[start:end])
        return
    typed_script = files("hymem.core.migrations").joinpath(
        "055_aggregation_typed_provenance.sql"
    ).read_text(encoding="utf-8")
    typed_names = legacy_aggregation_names + (
        "aggregation_input_insert_guard", "aggregation_input_update_guard",
        "aggregation_input_delete_unpublishes",
        "aggregation_input_source_insert_guard",
        "aggregation_input_source_update_guard",
        "aggregation_input_source_delete_unpublishes",
        "aggregation_publication_update_guard",
    )
    for name in typed_names:
        marker = f"CREATE TRIGGER {name}"
        start = typed_script.index(marker)
        end = typed_script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(typed_script[start:end])


def _install_aggregation_generation_guards(conn: sqlite3.Connection) -> None:
    """Heal v56's canonical registry and generation publication guards."""

    script = files("hymem.core.migrations").joinpath(
        "056_aggregation_generation_identity.sql"
    ).read_text(encoding="utf-8")
    for name in (
        "aggregation_generations_insert_guard",
        "aggregation_generations_update_guard",
        "aggregation_generations_delete_guard",
        "aggregation_generation_node_update_guard",
        "aggregation_generation_publication_insert_guard",
    ):
        marker = f"CREATE TRIGGER {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _install_aggregation_material_support_objects(
    conn: sqlite3.Connection,
) -> bool:
    """Heal v57-owned indexes/views before any dependent trigger can run.

    A SQLite trigger may remain installed after a referenced view is dropped.
    DML then fails while evaluating the dangling trigger.  Views therefore
    have to be restored before startup normalization touches profile/evidence
    rows; indexes are repaired in the same pass because their exact shapes are
    part of the v57 hot-path boundary.  Return whether anything changed so the
    caller can withdraw authority that may have escaped a missing object.
    """

    script = files("hymem.core.migrations").joinpath(
        "057_aggregation_material_epoch.sql"
    ).read_text(encoding="utf-8")
    changed = False
    for statement in _split_sql_statements(script):
        match = re.match(
            r"\s*CREATE\s+(?:UNIQUE\s+)?(INDEX|VIEW)\s+(\w+)\b",
            statement,
            flags=re.I,
        )
        if match is None:
            continue
        kind, name = match.groups()
        object_type = kind.lower()
        actual = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type=? AND name=?",
            (object_type, name),
        ).fetchone()

        def normalized(value: str) -> str:
            value = re.sub(r"--[^\n]*", "", value)
            return re.sub(r"\s+", " ", value).strip().rstrip(";")

        if actual is not None and normalized(actual["sql"]) == normalized(statement):
            continue
        # A same-named object of another kind is a malformed boundary, not a
        # healable omission: do not drop arbitrary application data here.
        collision = conn.execute(
            "SELECT type FROM sqlite_master WHERE name=?", (name,),
        ).fetchone()
        if collision is not None and str(collision["type"]) != object_type:
            raise RuntimeError(
                f"schema v57 {object_type} boundary is malformed"
            )
        conn.execute(f"DROP {kind.upper()} IF EXISTS {name}")
        conn.execute(statement)
        changed = True
    return changed


def _install_aggregation_material_guards(conn: sqlite3.Connection) -> None:
    """Heal every canonical v57 material support object and trigger."""

    _install_aggregation_material_support_objects(conn)
    script = files("hymem.core.migrations").joinpath(
        "057_aggregation_material_epoch.sql"
    ).read_text(encoding="utf-8")
    for statement in _split_sql_statements(script):
        match = re.match(
            r"\s*CREATE\s+TRIGGER\s+(\w+)\b", statement, flags=re.I,
        )
        if match is None:
            continue
        conn.execute(f"DROP TRIGGER IF EXISTS {match.group(1)}")
        conn.execute(statement)


def _install_fact_authority_guards(conn: sqlite3.Connection) -> None:
    """Install/heal v46's authoritative fact publication guards."""

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_outcome_before_cursor ON "
        "fact_extraction_outcomes(session_id,cursor_before_message_id,"
        "cursor_before_partial_message_id,cursor_before_offset)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_outcome_after_cursor ON "
        "fact_extraction_outcomes(session_id,cursor_after_message_id,"
        "cursor_after_partial_message_id,cursor_after_offset)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_outcome_chain_order ON "
        "fact_extraction_outcomes(session_id,"
        "COALESCE(cursor_before_partial_message_id,cursor_before_message_id,-1),"
        "CASE WHEN cursor_before_partial_message_id IS NULL THEN 1 ELSE 0 END,"
        "cursor_before_offset,slice_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_outcome_replay_v46 ON "
        "fact_extraction_outcomes(session_id,source_manifest_complete,"
        "prompt_version)"
    )

    script = files("hymem.core.migrations").joinpath(
        "046_narrative_fact_authority.sql"
    ).read_text(encoding="utf-8")
    names = (
        "session_workspace_binding_guard",
        "fact_outcome_insert_guard",
        "fact_outcome_header_guard",
        "fact_outcome_bound_guard",
        "fact_outcome_result_guard",
        "fact_outcome_delete_guard",
        "fact_revision_insert_guard",
        "fact_revision_update_guard",
        "fact_revision_delete_guard",
        "fact_source_occurrence_insert_guard",
        "fact_source_occurrence_update_guard",
        "fact_source_occurrence_delete_guard",
        "narrative_fact_authority_insert_guard",
        "narrative_fact_authority_update_guard",
        "narrative_fact_bound_update_guard",
        "narrative_fact_lifecycle_projection_guard",
        "narrative_fact_delete_guard",
        "narrative_fact_lifecycle_insert_guard",
        "narrative_fact_lifecycle_update_guard",
        "narrative_fact_lifecycle_delete_guard",
    )
    for name in names:
        marker = f"CREATE TRIGGER IF NOT EXISTS {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        _apply_migration_sql(conn, script[start:end])


def _install_external_peer_guards(conn: sqlite3.Connection) -> None:
    """Heal schema-43 external author/provenance guards on every startup."""
    script = files("hymem.core.migrations").joinpath(
        "043_external_peer_provenance.sql"
    ).read_text(encoding="utf-8")
    for name in (
        "message_retention_coverage_delete_guard",
        "message_retention_coverage_update_guard",
        "session_workspace_binding_guard",
        "session_peer_binding_insert_guard",
        "session_peer_binding_update_guard",
        "session_peer_delete_guard",
        "peer_identity_update_guard",
        "peer_identity_delete_guard",
        "message_external_provenance_insert_guard",
        "message_external_provenance_update_guard",
        "message_lossless_source_update_guard",
        "message_coverage_peer_insert_guard",
        "message_coverage_peer_update_guard",
        "kg_evidence_v43_peer_insert_guard",
        "kg_evidence_v43_peer_update_guard",
    ):
        marker = f"CREATE TRIGGER {name}"
        start = script.index(marker)
        end = script.index("\nEND;", start) + len("\nEND;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _ensure_message_coverage_fts(conn: sqlite3.Connection) -> None:
    """Heal and exactly rebuild the durable-message search shadow."""
    definition = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='message_coverage_fts'"
    ).fetchone()
    definition_sql = "" if definition is None else "".join(
        str(definition["sql"] or "").lower().split()
    )
    columns = [
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(message_coverage_fts)"
        ).fetchall()
    ] if definition is not None else []
    valid_shape = bool(
        columns == ["content"]
        and "usingfts5(content,content='',tokenize='porterunicode61')"
        in definition_sql
    )
    if definition is not None and not valid_shape:
        for name in (
            "message_coverage_fts_insert",
            "message_coverage_fts_delete",
            "message_coverage_fts_update_delete",
            "message_coverage_fts_update_insert",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("DROP TABLE message_coverage_fts")
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS message_coverage_fts USING fts5(
            content,
            content='',
            tokenize='porter unicode61'
        );
        DROP TRIGGER IF EXISTS message_coverage_fts_insert;
        DROP TRIGGER IF EXISTS message_coverage_fts_delete;
        DROP TRIGGER IF EXISTS message_coverage_fts_update_delete;
        DROP TRIGGER IF EXISTS message_coverage_fts_update_insert;
        INSERT INTO message_coverage_fts(message_coverage_fts)
        VALUES('delete-all');
        INSERT INTO message_coverage_fts(rowid, content)
        SELECT rowid, json_extract(text, '$.content') FROM chunks
        WHERE chunk_kind = 'coverage'
          AND json_valid(text)
          AND json_type(text, '$.content') = 'text';
        CREATE TRIGGER message_coverage_fts_insert
        AFTER INSERT ON chunks
        WHEN new.chunk_kind = 'coverage'
         AND json_valid(new.text)
         AND json_type(new.text, '$.content') = 'text' BEGIN
            INSERT INTO message_coverage_fts(rowid, content)
            VALUES (new.rowid, json_extract(new.text, '$.content'));
        END;
        CREATE TRIGGER message_coverage_fts_delete
        AFTER DELETE ON chunks
        WHEN old.chunk_kind = 'coverage'
         AND json_valid(old.text)
         AND json_type(old.text, '$.content') = 'text' BEGIN
            INSERT INTO message_coverage_fts(
                message_coverage_fts, rowid, content
            ) VALUES (
                'delete', old.rowid, json_extract(old.text, '$.content')
            );
        END;
        CREATE TRIGGER message_coverage_fts_update_delete
        AFTER UPDATE OF text, chunk_kind ON chunks
        WHEN old.chunk_kind = 'coverage'
         AND json_valid(old.text)
         AND json_type(old.text, '$.content') = 'text' BEGIN
            INSERT INTO message_coverage_fts(
                message_coverage_fts, rowid, content
            ) VALUES (
                'delete', old.rowid, json_extract(old.text, '$.content')
            );
        END;
        CREATE TRIGGER message_coverage_fts_update_insert
        AFTER UPDATE OF text, chunk_kind ON chunks
        WHEN new.chunk_kind = 'coverage'
         AND json_valid(new.text)
         AND json_type(new.text, '$.content') = 'text' BEGIN
            INSERT INTO message_coverage_fts(rowid, content)
            VALUES (new.rowid, json_extract(new.text, '$.content'));
        END;
        """
    )


def _ensure_narrative_facts_fts(conn: sqlite3.Connection) -> None:
    """Heal and exactly rebuild the authoritative-current fact shadow.

    FTS5's external-content ``rebuild`` command mirrors every row in the
    content table.  That is deliberately wrong for narrative facts: legacy
    projections have no source proof and retracted projections are historical
    state.  Even if downstream joins reject them, indexing those documents
    changes BM25 corpus statistics.  Rebuild explicitly from the authoritative
    active subset and reinstall transition-aware triggers on every open.
    """

    definition = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='narrative_facts_fts'"
    ).fetchone()
    definition_sql = "" if definition is None else "".join(
        str(definition["sql"] or "").lower().split()
    )
    columns = [
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(narrative_facts_fts)"
        ).fetchall()
    ] if definition is not None else []
    valid_shape = bool(
        columns == ["text"]
        and (
            "usingfts5(text,content='narrative_facts',"
            "content_rowid='id',tokenize='porterunicode61')"
        ) in definition_sql
    )

    owner = not conn.in_transaction
    if owner:
        conn.execute("BEGIN IMMEDIATE")
    try:
        for name in (
            "narrative_facts_fts_insert",
            "narrative_facts_fts_delete",
            "narrative_facts_fts_update",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        if definition is not None and not valid_shape:
            conn.execute("DROP TABLE narrative_facts_fts")
        _apply_migration_sql(
            conn,
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS narrative_facts_fts USING fts5(
                text,
                content='narrative_facts',
                content_rowid='id',
                tokenize='porter unicode61'
            );
            INSERT INTO narrative_facts_fts(narrative_facts_fts)
            VALUES('delete-all');
            INSERT INTO narrative_facts_fts(rowid, text)
            SELECT id, text
            FROM narrative_facts
            WHERE source_outcome_key IS NOT NULL
              AND lifecycle_status = 'active'
              AND invalid_at IS NULL;
            CREATE TRIGGER narrative_facts_fts_insert
            AFTER INSERT ON narrative_facts
            WHEN new.source_outcome_key IS NOT NULL
             AND new.lifecycle_status = 'active'
             AND new.invalid_at IS NULL BEGIN
                INSERT INTO narrative_facts_fts(rowid, text)
                VALUES (new.id, new.text);
            END;
            CREATE TRIGGER narrative_facts_fts_delete
            AFTER DELETE ON narrative_facts
            WHEN old.source_outcome_key IS NOT NULL
             AND old.lifecycle_status = 'active'
             AND old.invalid_at IS NULL BEGIN
                INSERT INTO narrative_facts_fts(
                    narrative_facts_fts, rowid, text
                ) VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER narrative_facts_fts_update
            AFTER UPDATE OF text, source_outcome_key, lifecycle_status,
                            invalid_at
            ON narrative_facts BEGIN
                INSERT INTO narrative_facts_fts(
                    narrative_facts_fts, rowid, text
                )
                SELECT 'delete', old.id, old.text
                WHERE old.source_outcome_key IS NOT NULL
                  AND old.lifecycle_status = 'active'
                  AND old.invalid_at IS NULL;
                INSERT INTO narrative_facts_fts(rowid, text)
                SELECT new.id, new.text
                WHERE new.source_outcome_key IS NOT NULL
                  AND new.lifecycle_status = 'active'
                  AND new.invalid_at IS NULL;
            END;
            """,
        )
    except Exception:
        if owner:
            conn.execute("ROLLBACK")
        raise
    else:
        if owner:
            conn.execute("COMMIT")


def _ensure_post_migration_runtime_guards(conn: sqlite3.Connection) -> None:
    """Heal latest triggers only after their owning columns/tables exist."""
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()
    }
    chunk_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
    } if "chunks" in tables else set()
    if (
        schema_version(conn) >= 38
        and {"chunks", "chunks_fts"}.issubset(tables)
        and {"chunk_kind", "text"}.issubset(chunk_columns)
    ):
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks
            WHEN new.chunk_kind = 'extraction' BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks
            WHEN old.chunk_kind = 'extraction' BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.rowid, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update_delete
            AFTER UPDATE OF text, chunk_kind ON chunks
            WHEN old.chunk_kind = 'extraction' BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.rowid, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update_insert
            AFTER UPDATE OF text, chunk_kind ON chunks
            WHEN new.chunk_kind = 'extraction' BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            """
        )
    if (
        schema_version(conn) >= 40
        and {
            "chunks", "chunk_message_sources", "kg_evidence",
            "kg_evidence_signals", "kg_claim_observations",
            "kg_edge_lifecycle", "kg_lifecycle_dependencies",
            "knowledge_graph",
        }.issubset(tables)
    ):
        observation_columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(kg_claim_observations)"
            ).fetchall()
        }
        generation_conflict_scope = (
            "AND existing.phase1_generation_key "
            "IS new.phase1_generation_key"
            if "phase1_generation_key" in observation_columns
            else ""
        )
        conn.executescript(
            f"""
            DROP TRIGGER IF EXISTS kg_evidence_signals_v40_insert_guard;
            DROP TRIGGER IF EXISTS kg_evidence_signals_v40_update_guard;
            DROP TRIGGER IF EXISTS kg_evidence_signals_v40_delete_guard;
            DROP TRIGGER IF EXISTS kg_claim_observations_insert_guard;
            DROP TRIGGER IF EXISTS kg_edge_lifecycle_insert_guard;
            DROP TRIGGER IF EXISTS kg_lifecycle_dependencies_insert_guard;
            CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_insert_guard
            BEFORE INSERT ON kg_evidence_signals
            WHEN hymem_evidence_mutation_authorized() <> 1
              OR (new.signal_kind = 'manual_retraction' AND new.polarity <> -1) BEGIN
                SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
            END;
            CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_update_guard
            BEFORE UPDATE ON kg_evidence_signals
            WHEN hymem_evidence_mutation_authorized() <> 1
              OR (new.signal_kind = 'manual_retraction' AND new.polarity <> -1) BEGIN
                SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
            END;
            CREATE TRIGGER IF NOT EXISTS kg_evidence_signals_v40_delete_guard
            BEFORE DELETE ON kg_evidence_signals
            WHEN hymem_evidence_mutation_authorized() <> 1 BEGIN
                SELECT RAISE(ABORT, 'kg evidence signals are internally managed');
            END;
            CREATE TRIGGER IF NOT EXISTS kg_claim_observations_insert_guard
            BEFORE INSERT ON kg_claim_observations
            BEGIN
                SELECT RAISE(ABORT, 'claim observations are internally managed')
                WHERE hymem_evidence_mutation_authorized() <> 1;
                SELECT RAISE(ABORT, 'claim observation lacks canonical evidence')
                WHERE NOT (
                  EXISTS (
                    SELECT 1 FROM chunk_message_sources cms
                    JOIN chunks c ON c.id = cms.chunk_id
                    WHERE cms.chunk_id = new.chunk_id
                      AND cms.source_session_id = new.source_session_id
                      AND cms.source_message_id = new.source_message_id
                      AND c.source_manifest_version = 'claim-source-manifest-v1'
                  )
                  AND EXISTS (
                    SELECT 1 FROM kg_evidence ev
                    WHERE ev.id = new.evidence_id
                      AND ev.edge_id = new.edge_id
                      AND ev.source_session_id = new.source_session_id
                      AND ev.source_message_id = new.source_message_id
                      AND ev.evidence_kind = new.evidence_kind
                      AND ev.polarity = new.polarity
                      AND ev.interpretation_key = new.interpretation_key
                      AND ev.provenance_status = 'canonical'
                      AND (ev.is_current = 1
                           OR hymem_evidence_history_authorized() = 1)
                      AND ev.revision > 0
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM kg_claim_observations existing
                    WHERE existing.edge_id = new.edge_id
                      AND existing.source_session_id = new.source_session_id
                      AND existing.source_message_id = new.source_message_id
                      AND existing.evidence_kind = new.evidence_kind
                      AND existing.prompt_generation = new.prompt_generation
                      {generation_conflict_scope}
                      AND (existing.polarity <> new.polarity
                           OR existing.interpretation_key <> new.interpretation_key)
                  )
                );
            END;
            CREATE TRIGGER IF NOT EXISTS kg_edge_lifecycle_insert_guard
            BEFORE INSERT ON kg_edge_lifecycle
            BEGIN
                SELECT RAISE(ABORT, 'knowledge graph lifecycle events are internally managed')
                WHERE hymem_evidence_mutation_authorized() <> 1;
                SELECT RAISE(ABORT, 'invalid knowledge graph lifecycle event')
                WHERE NOT (
                  new.event_at = COALESCE(
                    hymem_normalize_iso_timestamp(new.event_at),
                    '0001-01-01T00:00:00.000Z'
                  )
                  AND (
                    (new.event_kind = 'claim_assertion' AND new.direction = 1
                     AND new.source_evidence_id IS NOT NULL
                     AND new.dependency_count = 0)
                    OR (new.event_kind = 'manual_retraction'
                        AND new.direction = -1 AND new.dependency_count = 0)
                    OR (new.event_kind = 'value_supersession'
                        AND new.direction = -1 AND new.dependency_count > 0
                        AND new.source_evidence_id IS NULL)
                    OR (new.event_kind = 'phase3_retraction'
                        AND new.direction = -1 AND new.dependency_count > 0
                        AND new.source_evidence_id IS NULL)
                    OR (new.event_kind = 'legacy_state'
                        AND new.source_evidence_id IS NULL
                        AND new.dependency_count = 0)
                  )
                  AND (new.source_evidence_id IS NULL OR EXISTS (
                    SELECT 1 FROM kg_evidence ev
                    WHERE ev.id = new.source_evidence_id
                      AND ev.edge_id = new.edge_id
                      AND ev.provenance_status = 'canonical'
                      AND (ev.is_current = 1
                           OR hymem_evidence_history_authorized() = 1)
                      AND ev.polarity = new.direction
                      AND ev.source_event_at = new.event_at
                  ))
                );
            END;
            CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_insert_guard
            BEFORE INSERT ON kg_lifecycle_dependencies
            BEGIN
                SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed')
                WHERE hymem_evidence_mutation_authorized() <> 1;
                SELECT RAISE(ABORT, 'invalid lifecycle evidence dependency')
                WHERE NOT EXISTS (
                  SELECT 1
                  FROM kg_edge_lifecycle lifecycle
                  JOIN kg_evidence ev ON ev.id = new.evidence_id
                  WHERE lifecycle.id = new.lifecycle_id
                    AND lifecycle.direction = -1
                    AND (ev.is_current = 1
                         OR hymem_evidence_history_authorized() = 1)
                    AND (
                      (lifecycle.event_kind = 'phase3_retraction'
                       AND lifecycle.edge_id = ev.edge_id AND ev.polarity = -1)
                      OR (lifecycle.event_kind = 'value_supersession'
                          AND ev.polarity = 1
                          AND EXISTS (
                            SELECT 1
                            FROM knowledge_graph loser
                            JOIN knowledge_graph winner
                              ON winner.subject_canonical = loser.subject_canonical
                             AND winner.predicate = loser.predicate
                             AND winner.object_canonical <> loser.object_canonical
                            WHERE loser.id = lifecycle.edge_id
                              AND winner.id = ev.edge_id
                              AND winner.derived = 0
                          ))
                    )
                );
            END;
            CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_update_guard
            BEFORE UPDATE ON kg_lifecycle_dependencies
            WHEN hymem_evidence_mutation_authorized() <> 1 BEGIN
                SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed');
            END;
            CREATE TRIGGER IF NOT EXISTS kg_lifecycle_dependencies_delete_guard
            BEFORE DELETE ON kg_lifecycle_dependencies
            WHEN hymem_evidence_mutation_authorized() <> 1 BEGIN
                SELECT RAISE(ABORT, 'lifecycle dependencies are internally managed');
            END;
            """
        )
    if (
        schema_version(conn) >= 41
        and {
            "chunks", "kg_evidence", "kg_claim_observations",
            "kg_claim_extraction_outcomes",
        }.issubset(tables)
        and {"source_manifest_version", "source_manifest_count"}.issubset(
            chunk_columns
        )
    ):
        # ``schema.sql`` deliberately uses IF NOT EXISTS so it remains safe as
        # a pre-migration bootstrap against old stores.  Once v41 is stamped,
        # replace the durable publication guards unconditionally: this heals a
        # process that created an earlier same-named trigger definition before
        # the final migration contract was installed.
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS kg_claim_extraction_outcomes_insert_guard;
            DROP TRIGGER IF EXISTS kg_claim_extraction_outcomes_update_guard;
            DROP TRIGGER IF EXISTS kg_claim_extraction_outcomes_delete_guard;
            DROP TRIGGER IF EXISTS chunk_source_manifest_header_update_guard;
            CREATE TRIGGER kg_claim_extraction_outcomes_insert_guard
            BEFORE INSERT ON kg_claim_extraction_outcomes
            WHEN hymem_evidence_mutation_authorized() <> 1
              OR length(trim(new.prompt_version)) = 0
              OR new.prompt_generation < 0
              OR substr(new.result_hash, 1, 7) <> 'sha256:'
              OR length(new.result_hash) <> 71
              OR substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'
            BEGIN
                SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
            END;
            CREATE TRIGGER kg_claim_extraction_outcomes_update_guard
            BEFORE UPDATE ON kg_claim_extraction_outcomes
            WHEN hymem_evidence_mutation_authorized() <> 1
              OR length(trim(new.prompt_version)) = 0
              OR new.prompt_generation < 0
              OR substr(new.result_hash, 1, 7) <> 'sha256:'
              OR length(new.result_hash) <> 71
              OR substr(new.result_hash, 8) GLOB '*[^0-9a-f]*'
            BEGIN
                SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
            END;
            CREATE TRIGGER kg_claim_extraction_outcomes_delete_guard
            BEFORE DELETE ON kg_claim_extraction_outcomes
            WHEN hymem_evidence_mutation_authorized() <> 1
            BEGIN
                SELECT RAISE(ABORT, 'claim extraction outcomes are internally managed');
            END;
            CREATE TRIGGER chunk_source_manifest_header_update_guard
            BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
            WHEN old.source_manifest_version IS NOT NULL
             AND (new.source_manifest_version IS NOT old.source_manifest_version
                  OR new.source_manifest_count IS NOT old.source_manifest_count)
             AND NOT (
                  new.source_manifest_version IS NULL
                  AND new.source_manifest_count IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM kg_evidence ev WHERE ev.chunk_id = old.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM kg_claim_observations observation
                      WHERE observation.chunk_id = old.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM kg_claim_extraction_outcomes outcome
                      WHERE outcome.chunk_id = old.id
                  )
             )
            BEGIN
                SELECT RAISE(ABORT, 'published chunk source manifest header is immutable');
            END;
            """
        )
    evidence_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(kg_evidence)").fetchall()
    } if "kg_evidence" in tables else set()
    if schema_version(conn) >= 42 and "published_at" in evidence_columns:
        # Heal the write-once publication boundary on every startup. A missing
        # trigger must not turn a stamped v42 store into mutable history.
        _install_evidence_revision_guards(conn)
        _install_evidence_publication_guards(conn)
    if (
        schema_version(conn) >= 43
        and {
            "sessions", "messages", "chunks", "message_retention_coverage",
            "peers", "session_peers", "kg_evidence", "episodes",
            "procedures", "profile_staging", "temporal_mentions",
            "narrative_facts", "chunk_message_sources", "user_profile",
            "kg_claim_observations",
        }.issubset(tables)
        and {"source_peer_id", "source_workspace_id"}.issubset(evidence_columns)
    ):
        _ensure_message_coverage_fts(conn)
        _install_external_peer_guards(conn)
    if schema_version(conn) >= 45 and _v45_domain_present(conn):
        _install_aggregation_source_guards(conn)
        if schema_version(conn) >= 56:
            _install_aggregation_generation_guards(conn)
        if schema_version(conn) >= 57:
            guards_were_current = _v57_material_bindings_present(conn)
            if not guards_were_current:
                # A missing/weakened invalidator means mutations may have
                # escaped the freshness clock.  Healing the SQL alone cannot
                # retroactively prove the standing publication, so withdraw
                # it and its success attestation before accepting the repair.
                conn.execute("DELETE FROM aggregation_publication_state")
                conn.execute(
                    "UPDATE aggregation_build_health SET "
                    "last_success_config_version=NULL,"
                    "last_success_generation_key=NULL,"
                    "last_success_material_epoch_key=NULL,"
                    "last_success_at=NULL WHERE id=1"
                )
            _install_aggregation_material_guards(conn)
    if schema_version(conn) >= 46 and _v46_domain_present(conn):
        _ensure_narrative_facts_fts(conn)
        _install_fact_authority_guards(conn)
    if (
        schema_version(conn) >= 47
        and _v47_domain_present(conn)
        and _table_exists(conn, "chunk_extraction_terminal_losses")
    ):
        _install_terminal_chunk_loss_guards(conn)
    if schema_version(conn) >= 52 and _v52_domain_present(conn):
        _install_source_materialization_guards(conn)
    if (
        schema_version(conn) >= 53
        and _table_exists(conn, "phase1_generations")
        and _v53_generation_bindings_present(conn)
    ):
        _validate_phase1_generation_registry(conn)
        _install_phase1_generation_guards(conn)
        prune_unreferenced_inexact_phase1_generations(conn)
    if (
        schema_version(conn) >= 54
        and not _v54_domain_present(conn)
        and _v53_generation_bindings_present(conn)
    ):
        raise RuntimeError("schema v54 producer-scoped auxiliary domain is incomplete")
    if schema_version(conn) >= 54 and _v54_domain_present(conn):
        if not _v54_auxiliary_bindings_present(conn, include_indexes=False):
            raise RuntimeError(
                "schema v54 producer-scoped auxiliary domain is incomplete"
            )
        from hymem.dreaming.phase1_auxiliary import (
            validate_phase1_auxiliary_registry,
        )

        validate_phase1_auxiliary_registry(conn)
        _install_phase1_auxiliary_guards(conn)
        if not _v54_auxiliary_bindings_present(conn):
            raise RuntimeError(
                "schema v54 producer-scoped auxiliary indexes could not be healed"
            )
    if schema_version(conn) == 55 and _v55_domain_present(conn):
        if not _v55_aggregation_bindings_present(conn):
            raise RuntimeError(
                "schema v55 aggregation provenance boundary is malformed"
            )
    if schema_version(conn) >= 56 and _v55_domain_present(conn):
        if not _v56_generation_bindings_present(
            conn, allow_v57=schema_version(conn) >= 57,
        ):
            raise RuntimeError(
                "schema v56 aggregation generation boundary is malformed"
            )
    if schema_version(conn) >= 57 and _v57_domain_footprint_present(conn):
        if not _v57_domain_present(conn):
            raise RuntimeError(
                "schema v57 aggregation material prerequisites are malformed"
            )
        if not _v57_material_bindings_present(conn):
            raise RuntimeError(
                "schema v57 aggregation material boundary is malformed"
            )


def _install_phase1_generation_guards(conn: sqlite3.Connection) -> None:
    """Heal canonical/immutable producer registry guards on every startup."""

    conn.executescript(
        """
        DROP TRIGGER IF EXISTS phase1_generations_insert_guard;
        DROP TRIGGER IF EXISTS phase1_generations_update_guard;
        DROP TRIGGER IF EXISTS phase1_generations_delete_guard;
        CREATE TRIGGER phase1_generations_insert_guard
        BEFORE INSERT ON phase1_generations
        WHEN hymem_phase1_generation_registry_row_is_valid(
            new.generation_key,new.extraction_cache_key,
            new.producer_identity_sha256,new.identity_exact,
            new.reuse_scope,new.binding_json
        ) <> 1
        BEGIN
            SELECT RAISE(ABORT, 'invalid Phase-1 generation registry row');
        END;
        CREATE TRIGGER phase1_generations_update_guard
        BEFORE UPDATE ON phase1_generations
        BEGIN
            SELECT RAISE(ABORT, 'Phase-1 generation registry is immutable');
        END;
        CREATE TRIGGER phase1_generations_delete_guard
        BEFORE DELETE ON phase1_generations
        WHEN hymem_phase1_generation_prune_authorized() <> 1
          OR old.identity_exact <> 0
        BEGIN
            SELECT RAISE(ABORT, 'Phase-1 generation registry is immutable');
        END;
        """
    )


def _install_phase1_auxiliary_guards(conn: sqlite3.Connection) -> None:
    """Heal v54's authority views, indexes, and mutation guards."""

    script = files("hymem.core.migrations").joinpath(
        "054_phase1_auxiliary_provenance.sql"
    ).read_text(encoding="utf-8")
    for spec in _V54_INDEX_SPECS:
        name, _table, _columns, _unique, _partial, _predicate, statement = spec
        if not _v54_index_is_exact(conn, spec):
            conn.execute(f'DROP INDEX IF EXISTS "{name}"')
        conn.execute(statement)
    view_start = script.index("DROP VIEW IF EXISTS current_phase1_publications;")
    trigger_start = script.index(
        "CREATE TRIGGER IF NOT EXISTS profile_marker_evidence_lineage_guard"
    )
    conn.executescript(script[view_start:trigger_start])
    trigger_names = (
        "profile_marker_evidence_lineage_guard",
        "profile_marker_decision_insert_guard",
        "rule_marker_evidence_lineage_guard",
        "phase1_auxiliary_outcome_insert_guard",
        "phase1_auxiliary_outcome_update_guard",
        "phase1_auxiliary_outcome_delete_guard",
        "entity_type_observation_insert_guard",
        "entity_type_observation_update_guard",
        "entity_type_observation_delete_guard",
        "entity_property_observation_insert_guard",
        "entity_property_observation_update_guard",
        "entity_property_observation_delete_guard",
        "entity_mention_observation_insert_guard",
        "entity_mention_observation_update_guard",
        "entity_mention_observation_delete_guard",
        "profile_marker_evidence_update_guard",
        "profile_marker_evidence_delete_guard",
        "profile_marker_decision_update_guard",
        "profile_marker_decision_delete_guard",
        "rule_marker_evidence_update_guard",
        "rule_marker_evidence_delete_guard",
        "rule_marker_decision_insert_guard",
        "rule_marker_decision_update_guard",
        "rule_marker_decision_delete_guard",
        "behavioral_marker_producer_insert_guard",
        "behavioral_marker_semantic_update_guard",
        "behavioral_marker_delete_guard",
        "linked_profile_semantic_update_guard",
        "linked_profile_delete_guard",
        "linked_rule_semantic_update_guard",
        "linked_rule_delete_guard",
        "entity_type_authority_insert_guard",
        "entity_type_authority_update_guard",
        "entity_property_authority_insert_guard",
        "entity_property_authority_update_guard",
        "profile_entry_domain_insert_guard",
        "profile_entry_domain_update_guard",
        "rule_domain_insert_guard",
        "rule_domain_update_guard",
    )
    for name in trigger_names:
        marker = f"CREATE TRIGGER IF NOT EXISTS {name}"
        start = script.index(marker)
        end = script.index("END;", start) + len("END;")
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(script[start:end])


def _validate_phase1_generation_registry(conn: sqlite3.Connection) -> None:
    """Reject a stamped store whose producer identity registry was forged."""

    invalid = conn.execute(
        "SELECT generation_key FROM phase1_generations "
        "WHERE hymem_phase1_generation_registry_row_is_valid("
        "generation_key,extraction_cache_key,producer_identity_sha256,"
        "identity_exact,reuse_scope,binding_json)<>1 LIMIT 1"
    ).fetchone()
    if invalid is not None:
        raise RuntimeError("Phase-1 generation registry integrity check failed")


def _install_source_materialization_guards(conn: sqlite3.Connection) -> None:
    """Heal the extraction-cache invalidation boundary on every startup."""

    session_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    if not {
        "source_materialized_message_id",
        "source_materialization_config_version",
    }.issubset(session_columns):
        return
    conn.executescript(
        """
        DROP TRIGGER IF EXISTS extraction_chunk_delete_invalidates_source_materialization;
        CREATE TRIGGER extraction_chunk_delete_invalidates_source_materialization
        AFTER DELETE ON chunks
        WHEN old.chunk_kind = 'extraction'
        BEGIN
            UPDATE sessions
            SET source_materialized_message_id = NULL,
                source_materialization_config_version = NULL
            WHERE id = old.session_id;
        END;
        """
    )


def _install_terminal_chunk_loss_guards(conn: sqlite3.Connection) -> None:
    """Heal v47's fail-closed scheduling guards on every startup."""
    conn.executescript(
        """
        DROP TRIGGER IF EXISTS processed_chunks_terminal_loss_insert_guard;
        CREATE TRIGGER processed_chunks_terminal_loss_insert_guard
        BEFORE INSERT ON processed_chunks
        WHEN EXISTS (
            SELECT 1 FROM chunk_extraction_terminal_losses loss
            WHERE loss.chunk_id = new.chunk_id
        ) BEGIN
            SELECT RAISE(ABORT, 'terminal extraction loss cannot be marked processed');
        END;
        DROP TRIGGER IF EXISTS processed_chunks_terminal_loss_update_guard;
        CREATE TRIGGER processed_chunks_terminal_loss_update_guard
        BEFORE UPDATE OF chunk_id, prompt_version ON processed_chunks
        WHEN EXISTS (
            SELECT 1 FROM chunk_extraction_terminal_losses loss
            WHERE loss.chunk_id = new.chunk_id
        ) BEGIN
            SELECT RAISE(ABORT, 'terminal extraction loss cannot be marked processed');
        END;
        DROP TRIGGER IF EXISTS chunk_extraction_terminal_loss_clear_processed;
        CREATE TRIGGER chunk_extraction_terminal_loss_clear_processed
        AFTER INSERT ON chunk_extraction_terminal_losses BEGIN
            DELETE FROM processed_chunks WHERE chunk_id = new.chunk_id;
        END;
        DROP TRIGGER IF EXISTS chunk_extraction_terminal_loss_insert_guard;
        CREATE TRIGGER chunk_extraction_terminal_loss_insert_guard
        BEFORE INSERT ON chunk_extraction_terminal_losses
        WHEN NOT EXISTS (
            SELECT 1 FROM chunks c
            WHERE c.id = new.chunk_id
              AND c.chunk_kind = 'extraction'
              AND COALESCE(c.salience_reason, '') <> 'short_session_fallback'
              AND c.source_manifest_version IS NULL
              AND c.source_manifest_count IS NULL
        ) BEGIN
            SELECT RAISE(ABORT, 'terminal extraction loss requires an unmanifested extraction chunk');
        END;
        DROP TRIGGER IF EXISTS chunk_extraction_terminal_loss_update_guard;
        CREATE TRIGGER chunk_extraction_terminal_loss_update_guard
        BEFORE UPDATE ON chunk_extraction_terminal_losses BEGIN
            SELECT RAISE(ABORT, 'terminal extraction loss is immutable');
        END;
        DROP TRIGGER IF EXISTS chunk_extraction_terminal_loss_manifest_guard;
        CREATE TRIGGER chunk_extraction_terminal_loss_manifest_guard
        BEFORE UPDATE OF source_manifest_version, source_manifest_count ON chunks
        WHEN EXISTS (
            SELECT 1 FROM chunk_extraction_terminal_losses loss
            WHERE loss.chunk_id = old.id
        ) AND (
            new.source_manifest_version IS NOT old.source_manifest_version
            OR new.source_manifest_count IS NOT old.source_manifest_count
        ) BEGIN
            SELECT RAISE(ABORT, 'terminal extraction loss must be explicitly resolved');
        END;
        """
    )


_MIGRATION_NAME_RE = re.compile(r"^(\d+)")
# Errors raised when a forward-only migration re-applies against a schema.sql
# database that already has the object. Tolerated so migrations stay no-ops.
_IDEMPOTENT_ERROR_MARKERS = ("duplicate column name", "already exists")


def _discover_migrations() -> list[tuple[int, object]]:
    """Return (version, traversable) for every NNN_*.sql under migrations/,
    sorted ascending by the leading integer."""
    pkg = files("hymem.core.migrations")
    found: list[tuple[int, object]] = []
    for entry in pkg.iterdir():
        name = entry.name
        if not name.endswith(".sql"):
            continue
        match = _MIGRATION_NAME_RE.match(name)
        if match is None:
            continue
        found.append((int(match.group(1)), entry))
    found.sort(key=lambda item: item[0])
    return found


def _split_sql_statements(script: str) -> list[str]:
    """Split a migration script into individual statements, treating a
    ``CREATE TRIGGER ... BEGIN ... END;`` block as one statement (its internal
    semicolons must not split it). Full-line ``--`` comments are dropped.

    Only FULL-LINE comments are stripped, so a semicolon inside a TRAILING
    ``--`` comment still terminates the statement and cuts a CREATE TABLE in
    half ("incomplete input"). Keep migration end-of-line comments
    semicolon-free; schema.sql has no such constraint (executescript hands the
    whole file to SQLite, which parses comments properly).
    """
    body = "\n".join(
        line for line in script.splitlines() if not line.strip().startswith("--")
    )
    parts = re.split(r"(\bBEGIN\b|\bEND\b|;)", body, flags=re.IGNORECASE)
    statements: list[str] = []
    buf: list[str] = []
    depth = 0
    for part in parts:
        token = part.strip().lower()
        if token == "begin":
            depth += 1
            buf.append(part)
        elif token == "end":
            depth = max(0, depth - 1)
            buf.append(part)
        elif part == ";" and depth == 0:
            buf.append(part)
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(part)
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _apply_migration_sql(conn: sqlite3.Connection, script: str) -> None:
    """Execute a migration script statement-by-statement, tolerating the
    idempotency errors a forward-only migration raises on an up-to-date DB
    (duplicate column / object already exists)."""
    for stmt in _split_sql_statements(script):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if any(m in str(exc).lower() for m in _IDEMPOTENT_ERROR_MARKERS):
                continue
            raise


def _v40_sql_is_complete(conn: sqlite3.Connection) -> bool:
    """Recognize the fully installed v40 DDL after a pre-stamp crash.

    Migration 040 rebuilds ``kg_evidence``. Replaying that rebuild after the
    SQL and data hook committed but before ``schema_meta`` was stamped would
    erase canonical provenance and revision history. Partial installs do not
    satisfy this deliberately strict shape/object check and are safe to
    rebuild from their still-pre-hook ledger.
    """
    def columns(table: str) -> set[str]:
        return {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    evidence_columns = {
        "source_message_id", "source_session_id", "source_created_at",
        "source_event_at", "source_coverage_chunk_id",
        "source_coverage_version", "provenance_status", "is_current",
        "superseded_at", "superseded_reason", "revision",
        "interpretation_key",
    }
    chunk_columns = {"source_manifest_version", "source_manifest_count"}
    if not evidence_columns.issubset(columns("kg_evidence")):
        return False
    if not chunk_columns.issubset(columns("chunks")):
        return False
    required_objects = {
        ("table", "chunk_message_sources"),
        ("table", "kg_edge_lifecycle"),
        ("table", "kg_lifecycle_dependencies"),
        ("table", "kg_claim_observations"),
        ("trigger", "kg_evidence_v40_insert_guard"),
        ("trigger", "kg_evidence_v40_update_guard"),
        ("trigger", "chunk_source_manifest_header_update_guard"),
        ("trigger", "kg_edge_lifecycle_insert_guard"),
        ("trigger", "kg_evidence_signals_v40_insert_guard"),
        ("trigger", "kg_lifecycle_dependencies_update_guard"),
        ("trigger", "kg_lifecycle_dependencies_delete_guard"),
        ("index", "idx_evidence_canonical_identity"),
    }
    found = {
        (str(row["type"]), str(row["name"]))
        for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN "
            "('table','trigger','index')"
        ).fetchall()
    }
    return required_objects.issubset(found)


def _prepare_v40_legacy_shape(conn: sqlite3.Connection) -> None:
    """Complete only the legacy columns required by migration 040.

    The migration test matrix intentionally includes sparse but supported
    historical stores. SQL cannot conditionally ``ALTER`` a missing table, so
    this small preflight handles both an absent graph and old graph rows that
    predate bi-temporal columns.
    """
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "knowledge_graph" in tables:
        graph_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(knowledge_graph)").fetchall()
        }
        additions = {
            "pos_evidence": "INTEGER NOT NULL DEFAULT 0",
            "neg_evidence": "INTEGER NOT NULL DEFAULT 0",
            "first_seen": "TIMESTAMP",
            "last_seen": "TIMESTAMP",
            "valid_at": "TIMESTAMP",
            "invalid_at": "TIMESTAMP",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "derived": "BOOLEAN NOT NULL DEFAULT 0",
        }
        for name, declaration in additions.items():
            if name not in graph_columns:
                conn.execute(
                    f"ALTER TABLE knowledge_graph ADD COLUMN {name} {declaration}"
                )


def _v40_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether this historical store actually contains the graph domain."""
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    return {
        "knowledge_graph", "kg_evidence", "chunks", "sessions",
        "message_retention_coverage",
    }.issubset(tables)


def _v45_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether a partial historical store contains the aggregation domain."""

    required_tables = {
        "episodes", "aggregation_nodes", "message_retention_coverage",
    }
    if not all(_table_exists(conn, table) for table in required_tables):
        return False
    episode_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(episodes)").fetchall()
    }
    node_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(aggregation_nodes)").fetchall()
    }
    coverage_info = conn.execute(
        "PRAGMA table_info(message_retention_coverage)"
    ).fetchall()
    coverage_columns = {str(row["name"]) for row in coverage_info}
    coverage_pk = {
        str(row["name"]): int(row["pk"])
        for row in coverage_info if int(row["pk"] or 0) > 0
    }
    # Pre-v45 base columns used by the migration triggers.  New v45 columns
    # may already exist on a fresh schema bootstrap and are deliberately not
    # required here.
    return {
        "id", "session_id", "title", "summary",
    }.issubset(episode_columns) and {
        "id", "title", "summary", "member_episode_ids", "session_ids",
        "n_members", "n_sessions", "level", "is_root",
    }.issubset(node_columns) and {
        "message_id", "source_session_id", "source_role", "source_peer_id",
        "source_workspace_id", "source_created_at", "chunk_id",
        "coverage_version", "message_content_hash",
    }.issubset(coverage_columns) and coverage_pk == {
        "message_id": 1, "chunk_id": 2, "coverage_version": 3,
    }


def _v46_domain_present(conn: sqlite3.Connection) -> bool:
    """Whether a partial historical store contains the complete facts domain."""

    if not all(_table_exists(conn, table) for table in (
        "sessions", "narrative_facts", "narrative_fact_embeddings",
        "message_retention_coverage", "chunks",
    )):
        return False
    session_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    fact_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(narrative_facts)").fetchall()
    }
    embedding_columns = {
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(narrative_fact_embeddings)"
        ).fetchall()
    }
    coverage_info = conn.execute(
        "PRAGMA table_info(message_retention_coverage)"
    ).fetchall()
    coverage_columns = {str(row["name"]) for row in coverage_info}
    coverage_pk = {
        str(row["name"]): int(row["pk"])
        for row in coverage_info if int(row["pk"] or 0) > 0
    }
    return {
        "id", "facts_message_id",
    }.issubset(session_columns) and {
        "id", "session_id", "start_message_id", "end_message_id", "text",
        "fact_date", "entities", "prompt_version", "valid_at", "invalid_at",
        "created_at",
    }.issubset(fact_columns) and {
        "fact_id", "vector_json", "model", "dim", "text_hash", "created_at",
    }.issubset(embedding_columns) and {
        "message_id", "source_session_id", "source_role", "source_peer_id",
        "source_workspace_id", "source_created_at", "chunk_id",
        "coverage_version", "message_content_hash",
    }.issubset(coverage_columns) and coverage_pk == {
        "message_id": 1, "chunk_id": 2, "coverage_version": 3,
    }


def _v46_sql_is_complete(conn: sqlite3.Connection) -> bool:
    """Recognize a fully applied v46 domain whose version stamp is stale.

    Replaying the v46 table rebuild would preserve the current fact projection
    but drop its lifecycle ledger. This deliberately strict structural check
    permits only guard healing plus an atomic stamp when all corrected tables
    and identities are already present.
    """

    required_columns = {
        "sessions": {
            "facts_cursor_message_id", "facts_cursor_partial_message_id",
            "facts_cursor_offset", "facts_cursor_prompt_version",
            "facts_retry_count", "facts_retry_config_version",
            "facts_quarantined",
        },
        "fact_extraction_outcomes": {
            "slice_key", "session_id", "prompt_version", "input_hash",
            "generation", "outcome_status", "result_hash",
            "source_manifest_version", "source_manifest_count",
            "source_manifest_hash", "source_manifest_complete", "succeeded_at",
        },
        "fact_extraction_source_occurrences": {
            "slice_key", "ordinal", "source_message_id", "source_session_id",
            "source_coverage_chunk_id", "source_coverage_version",
            "source_content_hash",
        },
        "fact_extraction_revisions": {
            "slice_key", "generation", "prompt_version", "outcome_status",
            "result_hash", "succeeded_at",
        },
        "narrative_facts": {
            "id", "source_outcome_key", "fact_key", "current_generation",
            "lifecycle_status", "created_at",
        },
        "narrative_fact_lifecycle": {
            "fact_id", "generation", "direction", "event_at",
            "prompt_version", "result_hash", "recorded_at",
        },
        "narrative_fact_embeddings": {
            "fact_id", "vector_json", "model", "dim", "text_hash",
            "created_at",
        },
    }
    for table, expected in required_columns.items():
        if not _table_exists(conn, table):
            return False
        actual = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if not expected.issubset(actual):
            return False
    unique_indexes: set[tuple[str, ...]] = set()
    for index in conn.execute("PRAGMA index_list(narrative_facts)").fetchall():
        if int(index["unique"] or 0) != 1:
            continue
        unique_indexes.add(tuple(
            str(row["name"])
            for row in conn.execute(
                f"PRAGMA index_info({index['name']})"
            ).fetchall()
        ))
    return (
        ("source_outcome_key", "fact_key") in unique_indexes
        and ("session_id", "start_message_id", "text") not in unique_indexes
        and _table_exists(conn, "narrative_facts_fts")
    )


def _seed_v40_legacy_lifecycle(conn: sqlite3.Connection) -> None:
    """Preserve pre-v40 materialized lifecycle without inventing a source."""
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if not {"knowledge_graph", "kg_edge_lifecycle"}.issubset(tables):
        return
    accepted_at = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
    rows = conn.execute(
        "SELECT id,status,valid_at,invalid_at,first_seen,last_seen "
        "FROM knowledge_graph WHERE derived=0 ORDER BY id"
    ).fetchall()
    with evidence_mutation(conn):
        for row in rows:
            is_open = row["status"] == "active" and row["invalid_at"] is None
            raw_event = (
                (row["valid_at"] or row["first_seen"])
                if is_open
                else (row["invalid_at"] or row["last_seen"] or row["first_seen"])
            )
            if raw_event is None:
                event_at = "0001-01-01T00:00:00.000Z"
            else:
                try:
                    event_at = normalize_iso_timestamp(
                        raw_event,
                        context="pre-v40 lifecycle",
                    )
                    validate_event_clock(
                        conn,
                        event_at,
                        accepted_at,
                        context="pre-v40 lifecycle",
                    )
                except ValueError:
                    # Never turn SQLite-only Julian/calendar interpretations or
                    # unsupported future snapshots into durable valid history.
                    continue
            conn.execute(
                "INSERT OR IGNORE INTO kg_edge_lifecycle("
                "edge_id,event_key,event_kind,direction,event_at,details) "
                "VALUES (?,'legacy-state','legacy_state',?,?,?)",
                (
                    row["id"],
                    1 if is_open else -1,
                    event_at,
                    "pre-v40 lifecycle snapshot - exact transition provenance unavailable",
                ),
            )
    # Do not rewrite the pre-v40 materialized interval here.  Older migrations
    # deliberately preserved their historical timestamp spelling (for example
    # a date-only ``2024-01-01``).  The lifecycle snapshot stores a normalized
    # event key, while portability canonicalizes an equivalent representation
    # in-memory.  Operational claim events call the reducer themselves.


def _normalize_v40_portable_keys(conn: sqlite3.Connection) -> None:
    """Heal unreleased v40 keys that accidentally embedded local row ids."""
    from hymem.dreaming.evidence import (
        _interpretation_key,
        recanonicalize_lifecycle_keys,
    )

    candidates = conn.execute(
        """
        SELECT * FROM kg_evidence
        WHERE interpretation_key = 'legacy-migrated-v1'
           OR interpretation_key = 'legacy-unspecified'
           OR interpretation_key GLOB 'legacy-row:[0-9]*'
        ORDER BY id
        """
    ).fetchall()
    with evidence_history_mutation(conn):
        for row in candidates:
            semantic_key = _interpretation_key(
                polarity=int(row["polarity"]),
                evidence_weight=int(row["evidence_weight"]),
                weight_source=row["weight_source"],
                source_role=row["source_role"],
                surface_subject=row["surface_subject"],
                surface_object=row["surface_object"],
                value_text=row["value_text"],
                value_numeric=row["value_numeric"],
                value_unit=row["value_unit"],
                temporal_scope=row["temporal_scope"],
            )
            if semantic_key == row["interpretation_key"]:
                continue
            conn.execute(
                "UPDATE kg_evidence SET interpretation_key=? WHERE id=?",
                (semantic_key, row["id"]),
            )
            conn.execute(
                "UPDATE kg_claim_observations SET interpretation_key=? "
                "WHERE evidence_id=?",
                (semantic_key, row["id"]),
            )

        signals = conn.execute(
            "SELECT * FROM kg_evidence_signals ORDER BY id"
        ).fetchall()
        for row in signals:
            key = str(row["signal_key"])
            portable = key
            if re.fullmatch(r"legacy:(positive|negative):[0-9]+", key):
                portable = key.rsplit(":", 1)[0]
            elif re.fullmatch(r"edge:[0-9]+:polarity:-?1", key):
                portable = "runtime-unattributed:polarity:" + key.rsplit(":", 1)[-1]
            if portable == key:
                continue
            collision = conn.execute(
                "SELECT * FROM kg_evidence_signals WHERE edge_id=? "
                "AND signal_kind=? AND signal_key=?",
                (row["edge_id"], row["signal_kind"], portable),
            ).fetchone()
            if collision is not None:
                semantic = (
                    "polarity", "evidence_weight", "counts_toward_confidence",
                    "details", "created_at",
                )
                if any(collision[field] != row[field] for field in semantic):
                    raise RuntimeError("legacy evidence signal key collision")
                conn.execute(
                    "DELETE FROM kg_evidence_signals WHERE id=?", (row["id"],)
                )
            else:
                conn.execute(
                    "UPDATE kg_evidence_signals SET signal_key=? WHERE id=?",
                    (portable, row["id"]),
                )
    # Key healing is intentionally metadata-only.  Recomputing every interval
    # on startup would mutate timestamp spellings established by historical
    # migrations even when their temporal meaning is unchanged.
    recanonicalize_lifecycle_keys(conn)


def _migrate_v57_embedding_identities(conn: sqlite3.Connection) -> None:
    """Remove legacy embedding identities that persisted opaque URL paths.

    Pre-v57 labels do not bind implementation or producer instance and a
    caller-controlled label may itself contain an opaque endpoint. Every row
    here is a derived cache, so an atomic purge and later re-embedding is the
    only exact, confidentiality-preserving upgrade.
    """

    tables = (
        "embedding_cache", "message_embeddings", "chunk_embeddings",
        "edge_embeddings", "episode_embeddings",
        "narrative_fact_embeddings", "aggregation_node_embeddings",
    )
    conn.execute("PRAGMA secure_delete=ON")
    secure_delete = conn.execute("PRAGMA secure_delete").fetchone()
    if secure_delete is None or int(secure_delete[0]) != 1:
        raise RuntimeError("v57 embedding route scrub requires secure_delete")
    # The absence of live rows says nothing about free pages, a prior WAL, or
    # already-deleted pre-v57 labels. Schedule the physical rewrite on every
    # actual upgrade from a pre-v57 domain; this marker is committed with the
    # logical purge so a crash cannot skip the out-of-transaction completion.
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
        "('embedding_route_scrub_pending','v57-known-live-rows')"
    )
    # Pre-v57 rows identify only a caller-controlled model label. Even an
    # OpenAI-compatible legacy string lacks implementation, dimension policy,
    # and exact producer semantics, while custom labels may themselves contain
    # secrets. These tables are all derived caches; purge is the only exact and
    # confidentiality-preserving migration. The next maintenance pass rebuilds
    # them under the producer-bound safe storage key.
    for table in tables:
        if not _table_exists(conn, table):
            continue
        columns = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if "model" not in columns:
            continue
        conn.execute(f"DELETE FROM {table}")

    conn.execute("DELETE FROM schema_meta WHERE key='vec_model'")
    for table in (
        "vec_chunks", "vec_messages", "vec_edges", "vec_episodes", "vec_facts",
    ):
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"DELETE FROM {table}")

    # DB-wide postcondition for every known durable embedding identity column.
    for table in tables:
        if _table_exists(conn, table) and "model" in {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }:
            if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                raise RuntimeError("legacy embedding identity survived v57")
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
        "('embedding_route_identity_schema',"
        "'hymem-openai-compatible-embedding-space-v2')"
    )


def _complete_v57_embedding_route_scrub(conn: sqlite3.Connection) -> None:
    """Finish the crash-resumable physical scrub of known legacy route rows."""

    pending = conn.execute(
        "SELECT value FROM schema_meta WHERE key='embedding_route_scrub_pending'"
    ).fetchone()
    if pending is None:
        return
    if pending["value"] != "v57-known-live-rows":
        raise RuntimeError("embedding route scrub marker is malformed")
    if conn.in_transaction:
        raise RuntimeError("embedding route scrub requires no transaction")
    secure_delete = conn.execute("PRAGMA secure_delete").fetchone()
    if secure_delete is None or int(secure_delete[0]) != 1:
        raise RuntimeError("embedding route scrub requires secure_delete")

    # VACUUM rewrites the main file, removing pre-migration free-page/history
    # copies. It can renumber implicit rowids, so repair the FTS shadows before
    # declaring the active database family clean. Episode vectors use stable
    # id-derived keys and the legacy vector mirrors were already purged.
    conn.execute("VACUUM")
    resync_rowid_shadows(conn)
    checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is None or int(checkpoint[0]) != 0:
        raise RuntimeError("embedding route WAL scrub is busy")
    with transaction(conn):
        conn.execute(
            "DELETE FROM schema_meta WHERE key='embedding_route_scrub_pending'"
        )
    checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is None or int(checkpoint[0]) != 0:
        raise RuntimeError("embedding route WAL finalization is busy")


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply every migration file whose version exceeds the DB's
    schema_version, bumping schema_version after each so an interrupted run
    resumes cleanly. Migrations are idempotent, so a fresh schema.sql database
    (which starts at version 1) runs them all as no-ops up to the latest."""
    cur = schema_version(conn)
    for version, entry in _discover_migrations():
        if version <= cur:
            continue
        apply_v40 = version != 40 or _v40_domain_present(conn)
        apply_version = apply_v40 and (
            version != 42
            or all(_table_exists(conn, table) for table in (
                "kg_evidence", "kg_claim_observations",
                "kg_claim_extraction_outcomes", "kg_edge_lifecycle",
            ))
        )
        if version == 43:
            apply_version = apply_version and _v43_domain_present(conn)
        if version == 45:
            apply_version = apply_version and _v45_domain_present(conn)
        if version == 46:
            apply_version = apply_version and _v46_domain_present(conn)
        if version == 47:
            apply_version = apply_version and _v47_domain_present(conn)
        if version == 48:
            apply_version = apply_version and _v48_domain_present(conn)
        if version == 49:
            apply_version = apply_version and _v49_domain_present(conn)
        if version == 50:
            apply_version = apply_version and _v50_domain_present(conn)
        if version == 51:
            apply_version = apply_version and _v51_domain_present(conn)
        if version == 52:
            apply_version = apply_version and _v52_domain_present(conn)
        if version == 53:
            apply_version = apply_version and _v53_domain_present(conn)
        if version == 54:
            apply_version = apply_version and _v54_domain_present(conn)
        if version == 55:
            apply_version = apply_version and _v55_domain_present(conn)
        if version == 56:
            apply_version = apply_version and _v55_domain_present(conn)
        if version == 57:
            apply_version = apply_version and _v57_domain_present(conn)
            if (
                not apply_version
                and _v56_generation_bindings_present(
                    conn,
                    allow_v57=_v57_domain_footprint_present(conn),
                )
            ):
                # A complete v56 aggregation domain is not an intentionally
                # sparse fixture. Missing v57 source keys/columns are damage;
                # stop at v56 instead of silently stamping past the boundary.
                raise RuntimeError(
                    "schema v57 aggregation material preflight is malformed"
                )
        if version == 58:
            apply_version = apply_version and {
                "source_workspace_id", "digest_cursor_prompt_version"
            }.issubset({row[1] for row in conn.execute("PRAGMA table_info(sessions)")})
        if version == 59:
            apply_version = apply_version and _table_exists(conn, "procedures")
        if version == 40 and apply_v40 and not _v40_sql_is_complete(conn):
            _prepare_v40_legacy_shape(conn)
        if version == 46 and apply_version and _v46_sql_is_complete(conn):
            # SQL may have committed before an old process published the
            # version marker. Never replay the destructive facts-table rebuild
            # over an already-authoritative lifecycle ledger.
            with transaction(conn):
                _install_fact_authority_guards(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) "
                    "VALUES ('schema_version', ?)", (str(version),),
                )
            log.info("recognized complete schema v%d (%s)", version, entry.name)
            continue
        if (
            version == 57
            and apply_version
            and _v57_material_bindings_present(
                conn, validate_triggers=False,
            )
        ):
            # A stale lower version marker can coexist with an otherwise exact
            # v57 storage tail whose owned view/index/trigger was removed while
            # reconstructing an older fixture or after an interrupted repair.
            # Restore only the canonical owned DDL before deciding whether the
            # complete boundary can be recognized.  The storage validator has
            # already rejected unsafe vector rows and malformed tables/FKs.
            with transaction(conn):
                _install_aggregation_material_guards(conn)
        if (
            version == 57
            and apply_version
            and _v57_material_bindings_present(conn)
        ):
            # An earlier process may have committed the complete v57 boundary
            # before publishing schema_version.  Replaying ALTER/CREATE over
            # that exact tail is neither idempotent nor necessary; recognize
            # it atomically just like the v46 rebuild boundary.
            with transaction(conn):
                # Exact live rows are already producer-bound, but a stale
                # schema stamp cannot prove that pre-v57 free pages/WAL never
                # held a caller-labelled route. Schedule the same resumable
                # physical scrub without deleting the validated live rows.
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
                    "('embedding_route_scrub_pending','v57-known-live-rows')"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
                    "('embedding_route_identity_schema',"
                    "'hymem-openai-compatible-embedding-space-v2')"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
                    "('schema_version',?)", (str(version),),
                )
            log.info("recognized complete schema v%d (%s)", version, entry.name)
            continue
        if version == 46 and apply_version:
            # v46 replaces the v26 facts table to remove its lossy legacy
            # UNIQUE key. DDL and version publication must be one crash-atomic
            # unit: an interrupted DROP/rename can never strand half a domain
            # while schema_meta still advertises v45.
            with transaction(conn):
                with evidence_mutation(conn):
                    _apply_migration_sql(
                        conn, entry.read_text(encoding="utf-8")
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) "
                    "VALUES ('schema_version', ?)", (str(version),),
                )
            log.info("migrated schema to v%d (%s)", version, entry.name)
            continue
        if version == 54 and apply_version:
            # v54 adds columns before publishing several mutually dependent
            # ledgers/views. Run both DDL and schema stamp in one SQLite
            # transaction; an injected crash cannot strand a v53 stamp with a
            # half-added column that makes the migration unreplayable.
            with transaction(conn):
                with evidence_mutation(conn):
                    _apply_v54_auxiliary_migration(
                        conn, entry.read_text(encoding="utf-8")
                    )
                # CREATE TABLE/INDEX IF NOT EXISTS must never bless a
                # pre-created lookalike.  Validate the exact keys, FKs,
                # authority CHECKs, and load-bearing index predicates while
                # the migration is still rollback-able, before publishing the
                # v54 schema stamp.
                if not _v54_auxiliary_bindings_present(conn):
                    raise RuntimeError(
                        "schema v54 producer-scoped auxiliary domain is malformed"
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
                    "('schema_version',?)",
                    (str(version),),
                )
            log.info("migrated schema to v%d (%s)", version, entry.name)
            continue
        if version == 55 and apply_version:
            # Columns, child ledgers, guards, invalidation, and the schema
            # stamp are one crash-atomic publication-boundary upgrade.
            with transaction(conn):
                _apply_migration_sql(conn, entry.read_text(encoding="utf-8"))
                # Fresh schema.sql already carries the v56 extension columns
                # and registry, while its v56 guards are deliberately installed
                # by the next crash-atomic migration. Validate the complete
                # extended boundary at v56 rather than requiring not-yet-run
                # triggers here.
                if (
                    not _v56_domain_footprint_present(conn)
                    and not _v55_aggregation_bindings_present(conn)
                ):
                    raise RuntimeError(
                        "schema v55 aggregation provenance domain is malformed"
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
                    "('schema_version',?)", (str(version),),
                )
            log.info("migrated schema to v%d (%s)", version, entry.name)
            continue
        if version == 56 and apply_version:
            # Registry, foreign keys, legacy-publication withdrawal, and the
            # schema stamp are one atomic generation-boundary upgrade.
            with transaction(conn):
                _apply_migration_sql(conn, entry.read_text(encoding="utf-8"))
                if _v57_material_bindings_present(
                    conn, validate_triggers=False,
                ):
                    # A reconstructed pre-v56 source table can temporarily
                    # remove a dependent v57 view/trigger while leaving the
                    # exact v57 storage tail in place.  Now that v56 has
                    # restored every dependency, heal that owned DDL before
                    # validating the extended v56 boundary.
                    _install_aggregation_material_guards(conn)
                # A deliberately downgraded/recovery fixture may already
                # carry the complete v57 tail.  Validate the v56 prefix
                # against that extended shape instead of mistaking the
                # owned v57 columns/triggers for foreign schema drift.
                if not _v56_generation_bindings_present(
                    conn,
                    allow_v57=_v57_domain_footprint_present(conn),
                ):
                    raise RuntimeError(
                        "schema v56 aggregation generation domain is malformed"
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
                    "('schema_version',?)", (str(version),),
                )
            log.info("migrated schema to v%d (%s)", version, entry.name)
            continue
        if version == 57 and apply_version:
            # Clock, registry, source invalidators, legacy route scrubbing and
            # the schema stamp are one crash-atomic freshness-boundary upgrade.
            with transaction(conn):
                # Purge every pre-v57, caller-labelled vector before installing
                # v57's invalidation and write-authority triggers.  Besides
                # keeping the upgrade replayable, this coalesces a large legacy
                # cache into the one migration transaction instead of appending
                # one material-clock revision per deleted episode vector.
                _migrate_v57_embedding_identities(conn)
                _apply_migration_sql(conn, entry.read_text(encoding="utf-8"))
                if not _v57_material_bindings_present(conn):
                    raise RuntimeError(
                        "schema v57 aggregation material domain is malformed"
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES "
                    "('schema_version',?)", (str(version),),
                )
            log.info("migrated schema to v%d (%s)", version, entry.name)
            continue
        if apply_version and (version != 40 or not _v40_sql_is_complete(conn)):
            # Forward migrations are an internal, transactionally owned
            # rewrite.  This also lets a fresh bootstrap replay pre-v40 signal
            # seed statements after the latest schema has installed v40's
            # direct-SQL mutation guards.
            with evidence_mutation(conn):
                _apply_migration_sql(conn, entry.read_text(encoding="utf-8"))
        if version == 39:
            # The DDL is idempotent, so complete its Python data hook before
            # publishing the version marker. A crash here leaves v38 stamped
            # and startup safely replays both pieces instead of permanently
            # skipping canonical provenance materialization.
            _backfill_v39_message_coverage(conn)
        if version == 40 and apply_v40:
            # SQL establishes the guarded ledger first. This data hook can
            # then recognize only source chunks whose exact historical builder
            # text is reproducible from validated v38 artifacts. A crash keeps
            # v39 stamped and safely replays the idempotent hook on restart.
            _backfill_v40_chunk_manifests(conn)
            _seed_v40_legacy_lifecycle(conn)
        if version == 41:
            # Non-empty v40 observation sets prove that a source-validated
            # extraction was published. Old processed markers alone do not
            # prove an empty success, so the hook deliberately does not invent
            # empty authority. Keeping this before the version stamp gives the
            # Python data phase the same crash-replay semantics as v39/v40.
            _backfill_v41_claim_extraction_outcomes(conn)
        if version == 47 and apply_version:
            _backfill_v47_terminal_chunk_losses(conn)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )
        log.info("migrated schema to v%d (%s)", version, entry.name)
    if schema_version(conn) >= 40 and _v40_domain_present(conn):
        _normalize_v40_portable_keys(conn)
    if schema_version(conn) >= 54:
        _drop_phase1_auxiliary_views(conn)
    if schema_version(conn) >= 41 and _table_exists(
        conn, "kg_claim_extraction_outcomes"
    ):
        _ensure_v41_claim_extraction_outcome_shape(conn)
        _refresh_v41_claim_extraction_outcomes(conn)
    if schema_version(conn) >= 57 and _v57_domain_present(conn):
        # A dropped v57 view leaves its dependent triggers present but
        # unexecutable.  Restore owned support objects before profile/evidence
        # normalization performs any DML.  Since writes may have escaped a
        # missing/weakened object, withdraw the standing authority first.
        if not _v57_material_bindings_present(conn):
            conn.execute("DELETE FROM aggregation_publication_state")
            conn.execute(
                "UPDATE aggregation_build_health SET "
                "last_success_config_version=NULL,"
                "last_success_generation_key=NULL,"
                "last_success_material_epoch_key=NULL,"
                "last_success_at=NULL WHERE id=1"
            )
        _install_aggregation_material_support_objects(conn)
    if schema_version(conn) >= 39 and _table_exists(conn, "user_profile"):
        _ensure_profile_active_invariants(conn)
    _ensure_post_migration_runtime_guards(conn)
    _complete_v57_embedding_route_scrub(conn)


def _drop_phase1_auxiliary_views(conn: sqlite3.Connection) -> None:
    """Remove v54 dependents before a historical table-shape repair."""

    for name in (
        "current_rules", "current_profile_entries",
        "current_entity_mentions", "current_entity_properties",
        "current_entity_types",
        "current_phase1_publications",
    ):
        conn.execute(f"DROP VIEW IF EXISTS {name}")


def _apply_v54_auxiliary_migration(
    conn: sqlite3.Connection, script: str,
) -> None:
    """Apply v54 idempotently inside its caller-owned transaction."""

    additions = (
        (
            "entity_types", "origin",
            "ALTER TABLE entity_types ADD COLUMN origin TEXT NOT NULL "
            "DEFAULT 'legacy_unattributed' CHECK (origin IN "
            "('user','legacy_unattributed'))",
        ),
        (
            "entity_properties", "origin",
            "ALTER TABLE entity_properties ADD COLUMN origin TEXT NOT NULL "
            "DEFAULT 'legacy_unattributed' CHECK (origin IN "
            "('user','legacy_unattributed'))",
        ),
        (
            "profile_entries", "source",
            "ALTER TABLE profile_entries ADD COLUMN source TEXT NOT NULL "
            "DEFAULT 'legacy_unattributed' CHECK (source IN "
            "('user','agent_inferred','legacy_unattributed'))",
        ),
    )
    _drop_phase1_auxiliary_views(conn)
    for table, column, ddl in additions:
        columns = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(ddl)
    body_start = script.index(
        "CREATE TABLE IF NOT EXISTS phase1_auxiliary_outcomes"
    )
    _apply_migration_sql(conn, script[body_start:])


def _backfill_v47_terminal_chunk_losses(conn: sqlite3.Connection) -> None:
    """Recover every provable legacy manifest, then terminalize the rest.

    v39 only needed to cover sessions referenced by legacy profile facts.  A
    store can therefore reach v46 with old extraction chunks whose raw messages
    still exist but whose coverage was never materialized.  Those chunks are
    recoverable and must not be mislabeled as data loss.  Cover only sessions
    that actually contain an unmanifested extraction chunk, retry v40's exact
    historical-builder reconstruction, and mark only the remainder.
    """
    from hymem.dreaming.chunks import record_unrecoverable_chunk_losses
    from hymem.dreaming.lossless import backfill_all_message_coverage

    with transaction(conn):
        sessions = conn.execute(
            "SELECT DISTINCT session_id FROM chunks "
            "WHERE chunk_kind='extraction' "
            "AND COALESCE(salience_reason, '') <> 'short_session_fallback' "
            "AND source_manifest_version IS NULL "
            "AND source_manifest_count IS NULL ORDER BY session_id"
        ).fetchall()
        for row in sessions:
            backfill_all_message_coverage(conn, row["session_id"])

    # This routine reproduces only the recognized historical one-user or
    # assistant+user builder bytes and validates every coverage proof.  It does
    # not guess source membership from a numeric range or from chunk prose.
    _backfill_v40_chunk_manifests(conn)

    with transaction(conn):
        newly_terminal = record_unrecoverable_chunk_losses(conn)
    if newly_terminal:
        log.warning(
            "chunk_extraction.source_manifest_terminal_loss chunks=%d "
            "action=excluded_from_future_budgets",
            newly_terminal,
        )


def _backfill_v39_message_coverage(conn: sqlite3.Connection) -> None:
    """Give upgraded profile provenance an exportable canonical source.

    Migration 039 can recover source ids from surviving raw USER rows. Cover
    every surviving raw row in each affected session before moving its ordered
    frontier, so upgrade→export→import is lossless without claiming a sparse
    one-message stream.
    """
    from hymem.dreaming.lossless import backfill_all_message_coverage

    with transaction(conn):
        sessions = conn.execute(
            "SELECT DISTINCT source_session_id AS session_id "
            "FROM user_profile WHERE source_session_id IS NOT NULL "
            "AND source_message_id IS NOT NULL ORDER BY source_session_id"
        ).fetchall()
        for row in sessions:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (row["session_id"],)
            ).fetchone()
            if exists is not None:
                backfill_all_message_coverage(conn, row["session_id"])


def _backfill_v40_chunk_manifests(conn: sqlite3.Connection) -> None:
    """Recover exact membership for recognized legacy chunk-builder output.

    The historical salience/baseline builder emitted either ``user: ...`` or
    one assistant endpoint followed by one user endpoint. We reproduce those
    bytes from immutable coverage; anything else remains deliberately
    unmanifested. Only a one-message chunk can also attribute an old claim
    exactly, so paired chunks replay under the current citation-capable prompt
    instead of guessing.
    """
    from hymem.dreaming.lossless import validate_message_coverage_artifact
    from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
    from hymem.dreaming.evidence import (
        claim_assertion_event_key,
        prompt_generation,
    )

    with transaction(conn):
        rows = conn.execute(
            """
            SELECT id, session_id, start_message_id, end_message_id, text
            FROM chunks
            WHERE chunk_kind = 'extraction'
              AND source_manifest_version IS NULL
              AND COALESCE(salience_reason, '') <> 'short_session_fallback'
              AND session_id IS NOT NULL
              AND start_message_id IS NOT NULL
              AND end_message_id IS NOT NULL
            ORDER BY id
            """
        ).fetchall()
        for chunk in rows:
            ids = [int(chunk["start_message_id"])]
            if int(chunk["end_message_id"]) != ids[0]:
                ids.append(int(chunk["end_message_id"]))
            proofs = []
            valid = True
            for message_id in ids:
                proof_row = conn.execute(
                    """
                    SELECT chunk_id FROM message_retention_coverage
                    WHERE message_id = ? AND source_session_id = ?
                      AND coverage_version = ?
                    """,
                    (message_id, chunk["session_id"], LOSSLESS_COVERAGE_VERSION),
                ).fetchone()
                if proof_row is None:
                    valid = False
                    break
                try:
                    proof = validate_message_coverage_artifact(
                        conn, message_id=message_id,
                        chunk_id=proof_row["chunk_id"],
                        coverage_version=LOSSLESS_COVERAGE_VERSION,
                    )
                except (RuntimeError, TypeError, ValueError):
                    valid = False
                    break
                proofs.append(proof)
            if not valid or not proofs:
                continue
            if len(proofs) == 1:
                expected_text = f"{proofs[0].role}: {proofs[0].content}"
                roles_valid = proofs[0].role == "user"
            else:
                expected_text = (
                    f"assistant: {proofs[0].content}\nuser: {proofs[1].content}"
                )
                roles_valid = proofs[0].role == "assistant" and proofs[1].role == "user"
            if not roles_valid or chunk["text"] != expected_text:
                continue
            conn.executemany(
                """
                INSERT INTO chunk_message_sources(
                    chunk_id, ordinal, source_message_id, source_session_id,
                    source_coverage_chunk_id, source_coverage_version
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["id"], ordinal, proof.message_id,
                        proof.session_id, proof.chunk_id,
                        LOSSLESS_COVERAGE_VERSION,
                    )
                    for ordinal, proof in enumerate(proofs)
                ],
            )
            conn.execute(
                "UPDATE chunks SET source_manifest_version = ?, "
                "source_manifest_count = ? WHERE id = ?",
                ("claim-source-manifest-v1", len(proofs), chunk["id"]),
            )
        # Promote old claim rows only when the manifested chunk has exactly one
        # source AND that prospective source identity occurs exactly once. Do
        # this in a second pass so two ambiguous chunks cannot make the first
        # arrival authoritative.
        candidates = conn.execute(
            """
            SELECT ev.id, ev.edge_id, ev.chunk_id, ev.evidence_kind,
                   ev.polarity, ev.extraction_prompt_version, ev.extracted_at,
                   ev.interpretation_key,
                   cms.source_message_id, cms.source_session_id,
                   cms.source_coverage_chunk_id, cms.source_coverage_version,
                   mc.source_role, mc.source_created_at
            FROM kg_evidence ev
            JOIN chunks c ON c.id = ev.chunk_id
            JOIN chunk_message_sources cms
              ON cms.chunk_id = c.id AND cms.ordinal = 0
            JOIN message_retention_coverage mc
              ON mc.message_id = cms.source_message_id
             AND mc.chunk_id = cms.source_coverage_chunk_id
             AND mc.coverage_version = cms.source_coverage_version
            WHERE ev.provenance_status = 'legacy_unattributed'
              AND c.source_manifest_version = 'claim-source-manifest-v1'
              AND c.source_manifest_count = 1
            ORDER BY ev.id
            """
        ).fetchall()
        groups: dict[tuple[int, str, str, int], list[sqlite3.Row]] = {}
        for row in candidates:
            key = (
                int(row["edge_id"]), row["evidence_kind"],
                row["source_session_id"], int(row["source_message_id"]),
            )
            groups.setdefault(key, []).append(row)
        for rows_for_source in groups.values():
            if len(rows_for_source) != 1:
                continue
            row = rows_for_source[0]
            try:
                event_at = normalize_iso_timestamp(
                    row["source_created_at"],
                    context="legacy claim source",
                )
            except ValueError:
                # SQLite accepts bare Julian numbers, impossible calendar
                # dates, and other shapes that the public clock does not. Such
                # old rows remain explicitly unattributed rather than gaining
                # invented canonical history during startup healing.
                continue
            # A legacy row whose source clock leads its extraction clock cannot
            # be safely promoted: doing so would let a future assertion become
            # today's materialized state. Keep it explicitly unattributed so
            # schema healing remains available rather than bricking startup.
            from hymem.core.time import validate_event_clock

            try:
                validate_event_clock(
                    conn,
                    event_at,
                    row["extracted_at"],
                    context="legacy claim promotion",
                )
            except ValueError:
                continue
            with evidence_mutation(conn):
                conn.execute(
                    """
                    UPDATE kg_evidence
                    SET source_message_id = ?, source_session_id = ?,
                        source_role = ?, source_created_at = ?, source_event_at = ?,
                        source_coverage_chunk_id = ?, source_coverage_version = ?,
                        provenance_status = 'canonical'
                    WHERE id = ?
                    """,
                    (
                        row["source_message_id"], row["source_session_id"],
                        row["source_role"], row["source_created_at"], event_at,
                        row["source_coverage_chunk_id"],
                        row["source_coverage_version"], row["id"],
                    ),
                )
            with evidence_mutation(conn):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO kg_claim_observations(
                        chunk_id, edge_id, source_session_id, source_message_id,
                        evidence_kind, polarity, prompt_version, prompt_generation,
                        evidence_id, interpretation_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["chunk_id"], row["edge_id"], row["source_session_id"],
                        row["source_message_id"], row["evidence_kind"],
                        row["polarity"],
                        row["extraction_prompt_version"] or "pre-v40",
                        prompt_generation(
                            row["extraction_prompt_version"] or "pre-v40"
                        ),
                        row["id"], row["interpretation_key"],
                    ),
                )
            if int(conn.execute(
                "SELECT polarity FROM kg_evidence WHERE id = ?", (row["id"],)
            ).fetchone()[0]) == 1:
                from hymem.dreaming.bitemporal import record_lifecycle_event

                record_lifecycle_event(
                    conn,
                    edge_id=int(row["edge_id"]),
                    event_key=claim_assertion_event_key(
                        row["source_session_id"], row["source_message_id"],
                        row["evidence_kind"], 1,
                    ),
                    event_kind="claim_assertion",
                    direction=1,
                    event_at=event_at,
                    source_evidence_id=int(row["id"]),
                )


def _backfill_v41_claim_extraction_outcomes(conn: sqlite3.Connection) -> None:
    """Backfill only non-empty, coherent v40 chunk publications."""
    if not _table_exists(conn, "kg_claim_extraction_outcomes") or not _table_exists(
        conn, "kg_claim_observations"
    ):
        return
    from hymem.dreaming.evidence import claim_observation_result_hash

    chunks = conn.execute(
        "SELECT DISTINCT chunk_id FROM kg_claim_observations ORDER BY chunk_id"
    ).fetchall()
    with transaction(conn), evidence_mutation(conn):
        for item in chunks:
            chunk_id = str(item["chunk_id"])
            authority = conn.execute(
                "SELECT DISTINCT prompt_version,prompt_generation "
                "FROM kg_claim_observations WHERE chunk_id=?",
                (chunk_id,),
            ).fetchall()
            if len(authority) != 1:
                # A normal whole-chunk publication has one prompt authority.
                # Mixed legacy rows are not enough proof to fabricate one.
                continue
            row = authority[0]
            conn.execute(
                "INSERT OR IGNORE INTO kg_claim_extraction_outcomes("
                "chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at) "
                "VALUES (?,?,?,?,COALESCE((SELECT MAX(observed_at) "
                "FROM kg_claim_observations WHERE chunk_id=?),CURRENT_TIMESTAMP))",
                (
                    chunk_id, row["prompt_version"], row["prompt_generation"],
                    claim_observation_result_hash(conn, chunk_id), chunk_id,
                ),
            )


def _ensure_v41_claim_extraction_outcome_shape(conn: sqlite3.Connection) -> None:
    """Heal an early stamped-v41 outcome FK from CASCADE to RESTRICT.

    Empty successful publications are the only durable proof that a newer
    extraction intentionally returned no claims. Cascading their row with a
    direct chunk deletion would permit a stale portable snapshot to resurrect
    those claims. Rebuild only the single table when its FK action is wrong;
    the runtime guard installer below then restores every dependent trigger.
    """
    foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(kg_claim_extraction_outcomes)"
    ).fetchall()
    chunk_fk_is_restrict = any(
        row["from"] == "chunk_id"
        and row["table"] == "chunks"
        and row["to"] == "id"
        and str(row["on_delete"]).upper() == "RESTRICT"
        for row in foreign_keys
    )
    base_columns = {
        "chunk_id", "prompt_version", "prompt_generation", "result_hash",
        "succeeded_at",
    }
    actual_columns = {
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(kg_claim_extraction_outcomes)"
        ).fetchall()
    }
    allowed_columns = {frozenset(base_columns)}
    if _table_exists(conn, "phase1_generations"):
        allowed_columns.add(frozenset({*base_columns, "phase1_generation_key"}))
    if frozenset(actual_columns) not in allowed_columns:
        raise RuntimeError("unsupported claim extraction outcome table shape")
    source_has_phase1_generation = "phase1_generation_key" in actual_columns
    # A stamped v53 store can still carry the released early-v41 CASCADE
    # shape after a partial/manual historical reconstruction.  When the v53
    # registry exists, this repair must converge the rebuilt table to the full
    # current shape even if its source table predates the binding column.
    target_has_phase1_generation = (
        source_has_phase1_generation
        or _table_exists(conn, "phase1_generations")
    )
    phase1_fk_is_restrict = any(
        row["from"] == "phase1_generation_key"
        and row["table"] == "phase1_generations"
        and row["to"] == "generation_key"
        and str(row["on_delete"]).upper() == "RESTRICT"
        for row in foreign_keys
    )

    if chunk_fk_is_restrict and (
        not target_has_phase1_generation
        or (source_has_phase1_generation and phase1_fk_is_restrict)
    ):
        if target_has_phase1_generation:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_claim_outcomes_phase1_generation "
                "ON kg_claim_extraction_outcomes(phase1_generation_key,chunk_id)"
            )
        return

    if (
        chunk_fk_is_restrict
        and target_has_phase1_generation
        and not source_has_phase1_generation
    ):
        # The chunk FK already has the safe released action, but an early
        # stamped reconstruction omitted v53's nullable producer binding.
        # Adding it in place preserves the outcome row and all unrelated table
        # metadata while converging to the full v53 schema.
        conn.execute(
            "ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN "
            "phase1_generation_key TEXT REFERENCES "
            "phase1_generations(generation_key) ON DELETE RESTRICT"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claim_outcomes_phase1_generation "
            "ON kg_claim_extraction_outcomes(phase1_generation_key,chunk_id)"
        )
        return

    conn.execute("SAVEPOINT hymem_heal_v41_outcome_fk")
    try:
        for trigger in (
            "kg_claim_extraction_outcomes_insert_guard",
            "kg_claim_extraction_outcomes_update_guard",
            "kg_claim_extraction_outcomes_delete_guard",
            "chunk_source_manifest_header_update_guard",
            "kg_evidence_published_at_insert_guard",
            "kg_evidence_published_at_update_guard",
            "kg_evidence_v40_delete_guard",
            "kg_edge_lifecycle_update_guard",
            "kg_edge_lifecycle_delete_guard",
            "kg_lifecycle_dependencies_update_guard",
            "kg_lifecycle_dependencies_delete_guard",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute(
            "ALTER TABLE kg_claim_extraction_outcomes "
            "RENAME TO kg_claim_extraction_outcomes_v41_old"
        )
        conn.execute(
            "CREATE TABLE kg_claim_extraction_outcomes("
            "chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE RESTRICT,"
            "prompt_version TEXT NOT NULL,"
            "prompt_generation INTEGER NOT NULL CHECK(prompt_generation >= 0),"
            "result_hash TEXT NOT NULL,"
            "succeeded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            + (
                ",phase1_generation_key TEXT REFERENCES "
                "phase1_generations(generation_key) ON DELETE RESTRICT"
                if target_has_phase1_generation else ""
            )
            + ")"
        )
        generation_column = (
            ",phase1_generation_key" if source_has_phase1_generation else ""
        )
        conn.execute(
            "INSERT INTO kg_claim_extraction_outcomes("
            "chunk_id,prompt_version,prompt_generation,result_hash,succeeded_at"
            + generation_column + ") SELECT chunk_id,prompt_version,"
            "prompt_generation,result_hash,succeeded_at" + generation_column
            + " FROM kg_claim_extraction_outcomes_v41_old"
        )
        conn.execute("DROP TABLE kg_claim_extraction_outcomes_v41_old")
        if target_has_phase1_generation:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_claim_outcomes_phase1_generation "
                "ON kg_claim_extraction_outcomes(phase1_generation_key,chunk_id)"
            )
    except BaseException:
        conn.execute("ROLLBACK TO hymem_heal_v41_outcome_fk")
        conn.execute("RELEASE hymem_heal_v41_outcome_fk")
        raise
    conn.execute("RELEASE hymem_heal_v41_outcome_fk")


def _refresh_v41_claim_extraction_outcomes(conn: sqlite3.Connection) -> None:
    """Keep v41 hashes aligned with intentional v40 key normalization."""
    from hymem.dreaming.evidence import refresh_claim_extraction_outcomes

    rows = conn.execute(
        "SELECT chunk_id FROM kg_claim_extraction_outcomes ORDER BY chunk_id"
    ).fetchall()
    refresh_claim_extraction_outcomes(
        conn, [str(row["chunk_id"]) for row in rows]
    )


def _ensure_profile_active_invariants(conn: sqlite3.Connection) -> None:
    """Heal legacy rows, then install v39 profile shape/source guards."""
    owner = not conn.in_transaction
    if owner:
        conn.execute("BEGIN IMMEDIATE")
    try:
        from hymem.dreaming.lossless import covered_messages_after
        from hymem.dreaming.user_profile import reconcile_profile_intervals

        # Reopen may be healing rows written by an older binary (or by direct
        # SQL while guards were absent).  Remove the previous guard set before
        # doing that repair, then reinstall it in this same transaction below.
        # Otherwise a now-invalid legacy tuple can make its own cleanup fail.
        for trigger_name in (
            "user_profile_shape_insert_guard",
            "user_profile_shape_update_guard",
            "user_profile_source_insert_guard",
            "user_profile_source_update_guard",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")

        # Canonicalizing relationship keys can collapse case/whitespace
        # variants, so remove the old case-sensitive guard inside this same
        # transaction before healing their interval chain.
        conn.execute(
            "DROP INDEX IF EXISTS idx_user_profile_one_active_relationship"
        )
        # Empty legacy assertions cannot carry memory. Remove them before the
        # write guards make their invalid shape immutable; retain every
        # non-empty assertion.
        conn.execute("DELETE FROM user_profile WHERE trim(value) = ''")
        conn.execute(
            "UPDATE user_profile SET slot_key = NULL "
            "WHERE slot <> 'relationship' AND slot_key IS NOT NULL"
        )
        for row in conn.execute(
            "SELECT id, slot_key FROM user_profile WHERE slot = 'relationship'"
        ).fetchall():
            key = (
                row["slot_key"].strip().lower()
                if isinstance(row["slot_key"], str) and row["slot_key"].strip()
                else "[legacy-unknown]"
            )
            if key != row["slot_key"]:
                conn.execute(
                    "UPDATE user_profile SET slot_key = ? WHERE id = ?",
                    (key, row["id"]),
                )

        # Never bless partial or unverifiable legacy provenance. A durable
        # tuple survives only when the producer-bounded USER artifact proves
        # it. The nullable live FK is retained only when that same raw row is
        # still an exact copy; retention may later SET NULL without erasing the
        # durable source tuple.
        for row in conn.execute(
            "SELECT id, evidence_message_id, source_message_id, "
            "source_session_id, source_created_at FROM user_profile"
        ).fetchall():
            source_mid = row["source_message_id"]
            source_session = row["source_session_id"]
            proof = None
            if isinstance(source_mid, int) and isinstance(source_session, str):
                covered = covered_messages_after(
                    conn,
                    source_session,
                    source_mid - 1,
                    limit=1,
                    roles=frozenset({"user"}),
                    through_message_id=source_mid,
                )
                if (
                    covered
                    and covered[0].message_id == source_mid
                    and covered[0].source_created_at == row["source_created_at"]
                ):
                    proof = covered[0]
            if proof is None:
                if any(
                    row[name] is not None
                    for name in (
                        "evidence_message_id", "source_message_id",
                        "source_session_id", "source_created_at",
                    )
                ):
                    conn.execute(
                        "UPDATE user_profile SET evidence_message_id = NULL, "
                        "source_message_id = NULL, source_session_id = NULL, "
                        "source_created_at = NULL WHERE id = ?",
                        (row["id"],),
                    )
                continue
            evidence_mid = row["evidence_message_id"]
            if evidence_mid is not None:
                live = conn.execute(
                    "SELECT session_id, role, content, created_at FROM messages "
                    "WHERE id = ?",
                    (evidence_mid,),
                ).fetchone()
                if (
                    evidence_mid != source_mid
                    or live is None
                    or live["session_id"] != source_session
                    or live["role"] != "user"
                    or live["content"] != proof.content
                    or live["created_at"] != proof.source_created_at
                ):
                    conn.execute(
                        "UPDATE user_profile SET evidence_message_id = NULL "
                        "WHERE id = ?",
                        (row["id"],),
                    )
        keys = conn.execute(
            "SELECT DISTINCT slot, slot_key FROM user_profile "
            "WHERE slot IN ('name','role','employer','location','age_birthday') "
            "OR slot = 'relationship'"
        ).fetchall()
        for row in keys:
            reconcile_profile_intervals(conn, row["slot"], row["slot_key"])
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profile_one_active_singleton "
            "ON user_profile(slot) WHERE invalid_at IS NULL AND "
            "slot IN ('name','role','employer','location','age_birthday')"
        )
        # COALESCE also protects malformed/legacy NULL relationship keys;
        # SQLite's ordinary UNIQUE semantics otherwise allow unlimited NULLs.
        conn.execute(
            "CREATE UNIQUE INDEX idx_user_profile_one_active_relationship "
            "ON user_profile(slot, lower(trim(COALESCE(slot_key, '')))) "
            "WHERE invalid_at IS NULL AND slot = 'relationship'"
        )
        shape_check = (
            "new.slot NOT IN ('role','name','employer','location','language',"
            "'relationship','possession','age_birthday','health_condition',"
            "'recurring_activity') OR "
            "trim(new.value) = '' OR "
            "new.confidence IS NULL OR NOT (new.confidence >= 0.0 "
            "AND new.confidence <= 1.0) OR "
            "(new.slot = 'relationship' AND "
            " (new.slot_key IS NULL OR trim(new.slot_key) = '' "
            "OR new.slot_key <> lower(trim(new.slot_key)))) OR "
            "(new.slot <> 'relationship' AND new.slot_key IS NOT NULL)"
        )
        for operation, suffix in (("INSERT", "insert"), ("UPDATE", "update")):
            conn.execute(
                f"CREATE TRIGGER user_profile_shape_{suffix}_guard "
                f"BEFORE {operation} ON user_profile WHEN {shape_check} "
                "BEGIN SELECT RAISE(ABORT, 'invalid user_profile shape'); END"
            )
        source_valid = """
            (
                new.source_message_id IS NOT NULL
                AND new.source_session_id IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM message_retention_coverage mc
                    JOIN sessions s ON s.id = mc.source_session_id
                    JOIN chunks c ON c.id = mc.chunk_id
                    WHERE mc.message_id = new.source_message_id
                      AND mc.source_session_id = new.source_session_id
                      AND mc.source_role = 'user'
                      AND mc.source_created_at IS new.source_created_at
                      AND mc.coverage_version = 'dream-lossless-message-v1'
                      AND s.coverage_message_id IS NOT NULL
                      AND mc.message_id <= s.coverage_message_id
                      AND c.chunk_kind = 'coverage'
                      AND c.session_id = mc.source_session_id
                      AND mc.chunk_id = hymem_coverage_chunk_id(
                          mc.source_session_id, mc.message_id
                      )
                      AND c.start_message_id = mc.message_id
                      AND c.end_message_id = mc.message_id
                      AND json_valid(c.text)
                      AND json_extract(c.text, '$.id') = mc.message_id
                      AND json_extract(c.text, '$.role') = 'user'
                      AND json_extract(c.text, '$.record_version') =
                          mc.record_version
                      AND hymem_message_record_proof_valid(
                          c.text, mc.message_content_hash,
                          mc.hash_version, mc.record_version
                      ) = 1
                      AND (
                          new.evidence_message_id IS NULL
                          OR EXISTS (
                              SELECT 1 FROM messages m
                              WHERE m.id = new.evidence_message_id
                                AND m.id = new.source_message_id
                                AND m.session_id = new.source_session_id
                                AND m.role = 'user'
                                AND m.created_at IS new.source_created_at
                                AND m.content = json_extract(c.text, '$.content')
                          )
                      )
                )
            )
        """
        conn.execute(
            "CREATE TRIGGER user_profile_source_insert_guard "
            "BEFORE INSERT ON user_profile "
            f"WHEN NOT ({source_valid}) "
            "BEGIN SELECT RAISE(ABORT, 'invalid user_profile provenance'); END"
        )
        # Existing unattributed rows are a supported legacy state. They may be
        # updated (for example interval reconciliation or confidence healing)
        # only while remaining unattributed; new rows must always carry a
        # producer-bounded USER source through the INSERT guard above.
        legacy_unchanged = """
            old.evidence_message_id IS NULL
            AND old.source_message_id IS NULL
            AND old.source_session_id IS NULL
            AND old.source_created_at IS NULL
            AND new.evidence_message_id IS NULL
            AND new.source_message_id IS NULL
            AND new.source_session_id IS NULL
            AND new.source_created_at IS NULL
        """
        conn.execute(
            "CREATE TRIGGER user_profile_source_update_guard "
            "BEFORE UPDATE ON user_profile "
            f"WHEN NOT (({source_valid}) OR ({legacy_unchanged})) "
            "BEGIN SELECT RAISE(ABORT, 'invalid user_profile provenance'); END"
        )
    except Exception:
        if owner:
            conn.execute("ROLLBACK")
        raise
    else:
        if owner:
            conn.execute("COMMIT")


_VEC_TABLES = frozenset({
    "vec_chunks", "vec_messages", "vec_edges", "vec_episodes", "vec_facts",
})


def _ensure_vec_table_named(conn: sqlite3.Connection, name: str, dim: int) -> bool:
    """Ensure one vec0 shadow and report whether it was newly created."""
    if name not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {name}")
    existed = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0(embedding float[{dim}])"
    )
    return not existed


def ensure_vec_table(
    conn: sqlite3.Connection, dim: int, *, model: str
) -> None:
    """Ensure all vec0 shadows exist for one exact vector-space identity.

    The virtual tables share ``vec_dim``/``vec_model`` metadata, so on a
    dimension or known-model change they are dropped and rebuilt in lockstep, then
    backfilled from their JSON mirror tables (chunk_embeddings /
    edge_embeddings / episode_embeddings).
    """
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("vector dimension must be a positive integer")
    if (
        not isinstance(model, str)
        or re.fullmatch(r"hymem-embedding-producer-v1:[0-9a-f]{64}", model)
        is None
    ):
        raise ValueError("vector model id must be an exact producer key")
    if not _load_vec_extension(conn):
        return
    try:
        existing_dim = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
        ).fetchone()
        existing_model = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_model'"
        ).fetchone()
        try:
            stored_dim = int(existing_dim["value"]) if existing_dim else None
        except (TypeError, ValueError, OverflowError):
            stored_dim = None
        identity_matches = bool(
            stored_dim == dim
            and existing_model is not None
            and existing_model["value"] == model
        )
        if identity_matches:
            # The durable mirrors are authoritative and every normal persist
            # writes its one shadow row.  Re-scanning every corpus on every
            # message append made hot ingestion quadratic.  Backfill only a
            # shadow that had to be created; explicit resync remains the repair
            # path for physical-row drift.
            for table, backfill in (
                ("vec_chunks", _backfill_vec),
                ("vec_edges", _backfill_vec_edges),
                ("vec_messages", _backfill_vec_messages),
                ("vec_episodes", _backfill_vec_episodes),
                ("vec_facts", _backfill_vec_facts),
            ):
                if _ensure_vec_table_named(conn, table, dim):
                    backfill(conn, dim, model=model)
            return
        conn.execute("DELETE FROM schema_meta WHERE key IN ('vec_dim','vec_model')")
        for stale in (
            "vec_chunks", "vec_messages", "vec_edges", "vec_episodes",
            "vec_facts",
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"DROP TABLE IF EXISTS {stale}")
        _ensure_vec_table_named(conn, "vec_chunks", dim)
        _ensure_vec_table_named(conn, "vec_messages", dim)
        _ensure_vec_table_named(conn, "vec_edges", dim)
        _ensure_vec_table_named(conn, "vec_episodes", dim)
        _ensure_vec_table_named(conn, "vec_facts", dim)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('vec_dim', ?)",
            (str(dim),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) "
            "VALUES ('vec_model', ?)",
            (model,),
        )
        _backfill_vec(conn, dim, model=model)
        _backfill_vec_messages(conn, dim, model=model)
        _backfill_vec_edges(conn, dim, model=model)
        _backfill_vec_episodes(conn, dim, model=model)
        _backfill_vec_facts(conn, dim, model=model)
    except sqlite3.OperationalError:
        log.info("vec tables unavailable; using Python cosine search")


def _backfill_vec(
    conn: sqlite3.Connection, dim: int, *, model: str
) -> None:
    rows = conn.execute(
        "SELECT c.rowid, e.vector_json FROM chunk_embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "WHERE c.chunk_kind = 'extraction' AND e.dim = ? "
        "AND e.model = ? ORDER BY c.rowid",
        (dim, model),
    ).fetchall()
    if not rows:
        return
    count = conn.execute("SELECT COUNT(*) AS c FROM vec_chunks").fetchone()["c"]
    if count >= len(rows):
        return

    for r in rows:
        try:
            vec = decode_vector(r["vector_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        vec = _finite_vec(vec, dim)
        if vec is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (r["rowid"], _pack_vector(vec)),
        )
    log.info("backfilled vec_chunks with %d existing embeddings", len(rows))


def _backfill_vec_edges(
    conn: sqlite3.Connection, dim: int, *, model: str
) -> None:
    """Populate vec_edges (rowid = knowledge_graph.id) from cached edge vectors.

    Best-effort: embed_pending_edges is the authoritative refresh. This handles
    cold-start, dim changes, and pre-v6 DBs.
    """
    rows = conn.execute(
        f"""
        SELECT kg.id AS edge_id,
               kg.subject_canonical || ' ' || kg.predicate || ' '
                   || kg.object_canonical AS edge_text
        FROM knowledge_graph kg
        WHERE {live_edge_predicate('kg')}
        """
    ).fetchall()
    if not rows:
        return
    have = conn.execute("SELECT COUNT(*) AS c FROM vec_edges").fetchone()["c"]
    if have >= len(rows):
        return
    for r in rows:
        emb = conn.execute(
            "SELECT vector_json,dim,model FROM edge_embeddings WHERE edge_text = ?",
            (r["edge_text"],),
        ).fetchone()
        if emb is None:
            continue
        try:
            stored_dim = int(emb["dim"])
        except (TypeError, ValueError, OverflowError):
            continue
        if stored_dim != dim:
            continue
        if emb["model"] != model:
            continue
        try:
            vec = decode_vector(emb["vector_json"])
        except (AttributeError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(vec, (list, tuple)) or len(vec) != dim:
            continue
        try:
            vector = [float(value) for value in vec]
            finite = all(math.isfinite(value) for value in vector)
            norm = math.sqrt(sum(value ** 2 for value in vector))
        except (TypeError, ValueError, OverflowError):
            finite = False
            norm = 0.0
            vector = []
        if not finite or not math.isfinite(norm) or norm <= 0.0:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO vec_edges(rowid, embedding) VALUES (?, ?)",
            (r["edge_id"], _pack_vector(vector)),
        )
    log.info("backfilled vec_edges from %d edge rows", len(rows))


def _backfill_vec_episodes(
    conn: sqlite3.Connection, dim: int, *, model: str
) -> None:
    """Populate vec_episodes from the exact clusterable episode universe."""

    # The sqlite-vec shadow is an acceleration of aggregation's proof-valid
    # universe, not merely every generation-visible durable mirror. Keeping
    # these definitions identical prevents a retracted/broken source proof
    # from becoming a permanent extra shadow row and disabling KNN forever.
    from hymem.dreaming.aggregate import load_clusterable_episodes

    episodes = load_clusterable_episodes(
        conn, max_rowid=None, embedding_model=model, embedding_dim=dim,
    )
    rows = [episode for episode in episodes if episode["vector"] is not None]
    if not rows:
        return
    for r in rows:
        rowid = int(r["rowid"])
        vec = _finite_vec(r["vector"], dim)
        if vec is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (rowid, _pack_vector(vec)),
        )
    log.info("backfilled vec_episodes from %d episode rows", len(rows))


def _backfill_vec_facts(
    conn: sqlite3.Connection, dim: int, *, model: str
) -> None:
    """Populate vec_facts (rowid = narrative_facts.id, an INTEGER PRIMARY KEY,
    so VACUUM-stable like vec_edges) from the JSON mirror on cold start / dim
    change. Suppresses its own missing-table error so ensure_vec_table keeps
    working against a pre-v26 store."""
    with contextlib.suppress(sqlite3.OperationalError):
        rows = conn.execute(
            "SELECT fact_id, vector_json FROM narrative_fact_embeddings "
            "WHERE dim = ? AND model = ?",
            (dim, model),
        ).fetchall()
        if not rows:
            return
        have = conn.execute("SELECT COUNT(*) AS c FROM vec_facts").fetchone()["c"]
        if have >= len(rows):
            return
        for r in rows:
            try:
                vec = decode_vector(r["vector_json"])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            vec = _finite_vec(vec, dim)
            if vec is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO vec_facts(rowid, embedding) VALUES (?, ?)",
                (r["fact_id"], _pack_vector(vec)),
            )
        log.info("backfilled vec_facts from %d fact rows", len(rows))


def _backfill_vec_messages(
    conn: sqlite3.Connection, dim: int, *, model: str
) -> None:
    """Populate the stable message-id vec0 shadow from its durable mirror."""
    query = (
        "SELECT message_id, vector_json FROM message_embeddings "
        "WHERE dim = ? AND model = ?"
    )
    params: tuple[object, ...] = (dim, model)
    try:
        rows = conn.execute(query + " ORDER BY message_id", params).fetchall()
    except sqlite3.OperationalError:
        return
    for row in rows:
        try:
            decoded = decode_vector(row["vector_json"])
        except (AttributeError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
        vector = _finite_vec(decoded, dim)
        if vector is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO vec_messages(rowid, embedding) VALUES (?, ?)",
            (row["message_id"], _pack_vector(vector)),
        )


def _finite_vec(value: object, dim: int) -> list[float] | None:
    """Return one exact, finite, non-zero vector or ``None``."""
    if not isinstance(value, (list, tuple)) or len(value) != dim:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    return vector if math.isfinite(norm) and norm > 0.0 else None


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def episode_vector_rowid(episode_id: str) -> int:
    """Return a portable, VACUUM-stable sqlite-vec key for an episode."""

    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode id is malformed")
    digest = hashlib.sha256(
        b"hymem-episode-vector-rowid-v1\x00" + episode_id.encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return value or 1


def episode_vector_rowids(episode_ids) -> dict[str, int]:
    """Project episode ids to vec rowids, failing closed on any collision."""

    result: dict[str, int] = {}
    owners: dict[int, str] = {}
    for raw_episode_id in episode_ids:
        episode_id = str(raw_episode_id)
        key = episode_vector_rowid(episode_id)
        previous = owners.get(key)
        if previous is not None and previous != episode_id:
            raise RuntimeError("episode vector rowid collision")
        owners[key] = episode_id
        result[episode_id] = key
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rowid-shadow integrity. chunks / aggregation_nodes have TEXT
# primary keys, so their rowids are implicit — and SQLite's VACUUM may RENUMBER
# implicit rowids (it compacts freelist gaps). Everything keyed on those rowids
# from the outside silently decouples when that happens: the external-content
# FTS tables (chunks_fts / episodes_fts / aggregation_nodes_fts) start joining
# match hits to the wrong content rows. Episode vectors use a stable
# id-derived integer key; episodes_fts still mirrors the implicit source rowid
# and is rebuilt here after VACUUM. This quietly degrades plain FTS retrieval.
# messages_fts (content_rowid='id', INTEGER PRIMARY KEY), vec_edges (rowid =
# knowledge_graph.id, INTEGER PRIMARY KEY), and narrative_facts_fts/vec_facts
# (rowid = narrative_facts.id, INTEGER PRIMARY KEY) are VACUUM-stable and need
# nothing.
# ─────────────────────────────────────────────────────────────────────────────

_ROWID_FTS_TABLES = ("chunks_fts", "episodes_fts", "aggregation_nodes_fts")


def resync_rowid_shadows(conn: sqlite3.Connection) -> None:
    """Rebuild every index keyed on an implicit (renumberable) rowid from its
    content table's CURRENT rowids. Episode vectors are also rebuilt onto the
    v57 deterministic id-derived key, healing shadows created by older code.
    Idempotent, and cheap next to the VACUUM that makes it necessary."""
    # FTS5's external-content `rebuild` command indiscriminately indexes every
    # chunks row.  Coverage artifacts are durable source storage, not search
    # documents, so rebuild that table explicitly from extraction rows only.
    with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text) "
            "SELECT rowid, text FROM chunks WHERE chunk_kind = 'extraction'"
        )
    with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
        conn.execute(
            "INSERT INTO message_coverage_fts(message_coverage_fts) "
            "VALUES('delete-all')"
        )
        conn.execute(
            "INSERT INTO message_coverage_fts(rowid, content) "
            "SELECT rowid, json_extract(text, '$.content') FROM chunks "
            "WHERE chunk_kind = 'coverage' AND json_valid(text) "
            "AND json_type(text, '$.content') = 'text'"
        )
    # Like coverage chunks, unpublished episode generations must stay out of
    # the physical FTS corpus: even filtered result rows would otherwise alter
    # BM25 IDF/ranking. Rebuild episode postings from the publication marker.
    with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
        conn.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('delete-all')")
        conn.execute(
            "INSERT INTO episodes_fts(rowid, title, summary) "
            "SELECT e.rowid, e.title, e.summary FROM episodes e "
            "JOIN sessions s ON s.id = e.session_id "
            "WHERE e.digest_generation IS NULL "
            "OR e.digest_generation = s.digest_published_generation"
        )
    for fts in (
        name for name in _ROWID_FTS_TABLES
        if name not in {"chunks_fts", "episodes_fts"}
    ):
        with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
            conn.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')")
    # A connection that already owns vec virtual tables has the module loaded.
    # Re-loading the extension from inside the caller's repair transaction can
    # fail on some sqlite-vec builds even though those tables are usable.
    if (
        not has_vec_table(conn, table="vec_episodes")
        and not _load_vec_extension(conn)
    ):
        return
    dim_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
    ).fetchone()
    if not dim_row:
        return
    try:
        dim = int(dim_row["value"])
    except (TypeError, ValueError, OverflowError):
        log.warning("rowid shadow resync skipped: invalid vec_dim metadata")
        return
    model_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'vec_model'"
    ).fetchone()
    model = (
        str(model_row["value"])
        if model_row is not None and isinstance(model_row["value"], str)
        and model_row["value"]
        else None
    )
    if (
        model is None
        or re.fullmatch(r"hymem-embedding-producer-v1:[0-9a-f]{64}", model)
        is None
    ):
        # A dimension without one exact producer identity cannot safely
        # select any durable mirror rows.  Drop the derived shadows rather
        # than rebuilding a mixed vector space.
        for table in _VEC_TABLES:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("DELETE FROM schema_meta WHERE key IN ('vec_dim','vec_model')")
        log.warning("rowid vector shadow resync skipped: vec_model is unavailable")
        return
    for table, backfill in (
        ("vec_chunks", _backfill_vec),
        ("vec_episodes", _backfill_vec_episodes),
    ):
        with contextlib.suppress(sqlite3.OperationalError):
            # Some sqlite-vec builds refuse DROP while a virtual-table cursor
            # was used earlier in this connection. Clearing first still makes
            # the repair exact if that DROP is unavailable.
            conn.execute(f"DELETE FROM {table}")
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            _ensure_vec_table_named(conn, table, dim)
            backfill(conn, dim, model=model)
    log.info("rowid shadows resynced (fts rebuilt, vec_chunks/vec_episodes refilled)")


def vec_episodes_aligned(conn: sqlite3.Connection, sample: int = 8) -> bool:
    """Check every deterministic episode vector key against JSON mirrors."""
    del sample
    if not _load_vec_extension(conn) or not has_vec_table(conn, table="vec_episodes"):
        return True
    dim_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
    ).fetchone()
    if not dim_row:
        return True
    try:
        dim = int(dim_row["value"])
    except (TypeError, ValueError, OverflowError):
        log.warning("vec_episodes alignment unverifiable: invalid vec_dim metadata")
        return True
    if dim <= 0:
        log.warning("vec_episodes alignment unverifiable: invalid vec_dim metadata")
        return True
    try:
        model_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='vec_model'"
        ).fetchone()
        model = model_row["value"] if model_row is not None else None
        if (
            not isinstance(model, str)
            or re.fullmatch(r"hymem-embedding-producer-v1:[0-9a-f]{64}", model)
            is None
        ):
            return False
        from hymem.dreaming.aggregate import load_clusterable_episodes

        episodes = load_clusterable_episodes(
            conn, max_rowid=None, embedding_model=model, embedding_dim=dim,
        )
        expected: dict[int, bytes] = {}
        for episode in episodes:
            vec = _finite_vec(episode["vector"], dim)
            if vec is not None:
                expected[int(episode["rowid"])] = _pack_vector(vec)
        actual_rows = conn.execute(
            "SELECT rowid,embedding FROM vec_episodes ORDER BY rowid"
        ).fetchall()
        actual_keys = {int(row["rowid"]) for row in actual_rows}
        if actual_keys != set(expected):
            return False
        actual = {
            int(row["rowid"]): bytes(row["embedding"])
            for row in actual_rows
        }
        for rowid in expected:
            if actual.get(rowid) != expected[rowid]:
                return False
    except sqlite3.OperationalError:
        return True    # unverifiable ≠ misaligned; never resync on a probe error
    except (
        AttributeError, UnicodeError, json.JSONDecodeError, TypeError,
        ValueError, OverflowError,
    ):
        return False
    return True


def heal_rowid_shadows(conn: sqlite3.Connection) -> bool:
    """Probe vec_episodes alignment and, on a proven mismatch, resync every
    rowid shadow. Returns True when a repair ran. Called by the dream runner
    before aggregation so stores skewed by pre-fix VACUUMs heal on their next
    dream instead of their next VACUUM."""
    if vec_episodes_aligned(conn):
        return False
    log.warning(
        "vec_episodes misaligned with episodes rowids (post-VACUUM renumber); "
        "resyncing all rowid shadows"
    )
    resync_rowid_shadows(conn)
    return True


def vec_search(
    conn: sqlite3.Connection,
    query_vector: list[float],
    top_k: int,
    *,
    table: str = "vec_chunks",
) -> list[tuple[int, float]]:
    if table not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {table}")
    if not _load_vec_extension(conn):
        return []
    try:
        rows = conn.execute(
            f"""
            SELECT rowid, distance
            FROM {table}
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (_pack_vector(query_vector), top_k),
        ).fetchall()
        return [(int(r["rowid"]), float(r["distance"])) for r in rows]
    except (sqlite3.OperationalError, TypeError):
        return []


def vec_search_strict(
    conn: sqlite3.Connection,
    query_vector: list[float],
    top_k: int,
    *,
    table: str = "vec_chunks",
) -> list[tuple[int, float]]:
    """Status-bearing vec search for correctness-sensitive materializers.

    Unlike :func:`vec_search`, absence/query failure is not conflated with a
    legitimate empty result. Aggregation catches this and explicitly switches
    to its exact path, recording that actual mode in the material snapshot.
    """

    if table not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {table}")
    if not _load_vec_extension(conn):
        raise RuntimeError("sqlite-vec extension unavailable")
    if not has_vec_table(conn, table=table):
        raise RuntimeError(f"{table} is unavailable")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("vec top_k must be a positive integer")
    rows = conn.execute(
        f"SELECT rowid,distance FROM {table} "
        "WHERE embedding MATCH ? AND k=? ORDER BY distance",
        (_pack_vector(query_vector), top_k),
    ).fetchall()
    result: list[tuple[int, float]] = []
    for row in rows:
        rowid, distance = int(row["rowid"]), float(row["distance"])
        if not math.isfinite(distance):
            raise RuntimeError("sqlite-vec returned a non-finite distance")
        result.append((rowid, distance))
    return sorted(result, key=lambda item: (item[1], item[0]))


def has_vec_table(conn: sqlite3.Connection, table: str = "vec_chunks") -> bool:
    if table not in _VEC_TABLES:
        raise ValueError(f"unknown vec table: {table}")
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    # A benchmark deadline is lexical and absent in ordinary operation.  Check
    # both sides of the transaction so work that crosses the bound rolls back
    # instead of committing late semantic state.
    check_current_deadline()
    conn.execute("BEGIN IMMEDIATE")
    try:
        # The ownership proof is deliberately inside the rollback-protected
        # writer transaction. A contender cannot replace the row between this
        # check and COMMIT, and an assertion failure cannot strand BEGIN open.
        _assert_transaction_lease_owned(conn)
        yield conn
        check_current_deadline()
        _assert_transaction_lease_owned(conn)
        # Keep COMMIT inside the protected region. SQLite may leave a
        # transaction open when commit itself faults; that stranded writer
        # would otherwise absorb a later lease release into the same failed
        # transaction and block every independent process.
        conn.execute("COMMIT")
    except BaseException as primary:
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        except BaseException as rollback_error:
            # Cleanup evidence is deliberately bounded to its type so a
            # provider/custom connection cannot leak text or replace the
            # original body/deadline/lease/commit exception.
            with contextlib.suppress(AttributeError, TypeError):
                primary.add_note(
                    "transaction rollback failed: "
                    f"{type(rollback_error).__name__}"
                )
        raise


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    return int(row["value"]) if row else 0


def backfill_entity_mentions(conn: sqlite3.Connection) -> None:
    """Idempotent backfill: populate entity_mentions from existing chunks if empty.

    No-op if the table already has rows or if there are no chunks.
    """
    has_mentions = conn.execute(
        "SELECT 1 FROM entity_mentions LIMIT 1"
    ).fetchone()
    if has_mentions:
        return
    chunk_count = conn.execute(
        "SELECT COUNT(*) AS c FROM chunks WHERE chunk_kind = 'extraction'"
    ).fetchone()["c"]
    if not chunk_count:
        return

    from hymem.dreaming.mentions import index_chunk_mentions

    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE chunk_kind = 'extraction'"
    ).fetchall()
    total = 0
    for row in rows:
        total += index_chunk_mentions(conn, row["id"], row["text"])
    log.info("backfilled entity_mentions: chunks=%d mentions=%d", len(rows), total)
