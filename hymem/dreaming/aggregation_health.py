"""Durable, bounded health state for the aggregation build boundary.

The runner marks an enabled build pending *before* entering aggregation code.
Only a returned result with zero fusion failures acknowledges that attempt.
This makes process death, a total exception, and a contained partial fusion
failure all fail closed across restarts while retaining only hashes, enums,
counters, and timestamps.
"""
from __future__ import annotations

import sqlite3
from typing import Any


MAX_AGGREGATION_HEALTH_COUNT = 2_147_483_647
MAX_AGGREGATION_ATTEMPT_TOKEN = 9_223_372_036_854_775_806


def _validated_attempt_token(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_AGGREGATION_ATTEMPT_TOKEN
    ):
        raise ValueError("aggregation attempt token is malformed")
    return value


def require_pending_aggregation_attempt(
    conn: sqlite3.Connection,
    config_version: str,
    generation_key: str,
    attempt_token: int,
    *,
    material_epoch_key: str | None = None,
) -> None:
    """Fence one caller to the exact still-active build attempt."""

    token = _validated_attempt_token(attempt_token)
    row = conn.execute(
        "SELECT pending_material_epoch_key FROM aggregation_build_health "
        "WHERE id=1 AND pending_config_version=? AND pending_generation_key=? "
        "AND pending_attempt_token=?",
        (config_version, generation_key, token),
    ).fetchone()
    if row is None or (
        material_epoch_key is not None
        and row["pending_material_epoch_key"] != material_epoch_key
    ):
        raise RuntimeError("aggregation build attempt was superseded")


def bind_pending_aggregation_material(
    conn: sqlite3.Connection,
    config_version: str,
    generation_key: str,
    attempt_token: int,
    material_epoch_key: str,
) -> None:
    """Attach a captured v57 epoch to the matching active health attempt."""

    token = _validated_attempt_token(attempt_token)
    row = conn.execute(
        "SELECT pending_material_epoch_key FROM aggregation_build_health "
        "WHERE id=1 AND pending_config_version=? AND pending_generation_key=? "
        "AND pending_attempt_token=?",
        (config_version, generation_key, token),
    ).fetchone()
    if row is None:
        raise RuntimeError("aggregation build attempt was superseded")
    prior = row["pending_material_epoch_key"]
    if prior is not None and prior != material_epoch_key:
        raise RuntimeError("aggregation pending material identity changed")
    cursor = conn.execute(
        "UPDATE aggregation_build_health SET pending_material_epoch_key=? "
        "WHERE id=1 AND pending_config_version=? AND pending_generation_key=? "
        "AND pending_attempt_token=?",
        (material_epoch_key, config_version, generation_key, token),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("aggregation build attempt was superseded")


def begin_aggregation_build(
    conn: sqlite3.Connection, config_version: str, *, generation_binding: object,
) -> int:
    """Atomically mark one build attempt pending before any fallible work."""

    # Publication state, not operational health, is the read authority. The
    # build marker and publication withdrawal land in the same transaction at
    # the runner boundary, before any fallible provider work. Historical rows
    # remain physical cache/audit material, but no old or partial tree can be
    # mistaken for the outcome of this attempt.
    from hymem.dreaming.aggregation_generation import register_aggregation_generation

    generation = register_aggregation_generation(conn, generation_binding)
    generation_key = str(generation["generation_key"])
    if generation["contract"]["material_config_version"] != config_version:
        raise ValueError("aggregation health generation/config mismatch")
    conn.execute("DELETE FROM aggregation_publication_state")

    row = conn.execute(
        """
        INSERT INTO aggregation_build_health(
            id, pending_config_version, pending_generation_key, pending_attempts,
            pending_caught_exceptions, pending_fusion_failures,
            first_pending_at, last_attempt_at, attempt_serial,
            pending_attempt_token
        ) VALUES (1, ?, ?, 1, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1, 1)
        ON CONFLICT(id) DO UPDATE SET
            superseded_pending_configs = MIN(
                aggregation_build_health.superseded_pending_configs +
                CASE
                    WHEN aggregation_build_health.pending_config_version IS NOT NULL
                     AND (aggregation_build_health.pending_config_version
                          <> excluded.pending_config_version
                          OR aggregation_build_health.pending_generation_key
                          IS NOT excluded.pending_generation_key)
                    THEN 1 ELSE 0
                END,
                ?
            ),
            pending_attempts = CASE
                WHEN aggregation_build_health.pending_config_version = excluded.pending_config_version
                 AND aggregation_build_health.pending_generation_key
                     IS excluded.pending_generation_key
                THEN MIN(aggregation_build_health.pending_attempts + 1, ?)
                ELSE 1
            END,
            pending_caught_exceptions = CASE
                WHEN aggregation_build_health.pending_config_version = excluded.pending_config_version
                 AND aggregation_build_health.pending_generation_key
                     IS excluded.pending_generation_key
                THEN aggregation_build_health.pending_caught_exceptions
                ELSE 0
            END,
            pending_fusion_failures = CASE
                WHEN aggregation_build_health.pending_config_version = excluded.pending_config_version
                 AND aggregation_build_health.pending_generation_key
                     IS excluded.pending_generation_key
                THEN aggregation_build_health.pending_fusion_failures
                ELSE 0
            END,
            first_pending_at = CASE
                WHEN aggregation_build_health.pending_config_version = excluded.pending_config_version
                 AND aggregation_build_health.pending_generation_key
                     IS excluded.pending_generation_key
                THEN aggregation_build_health.first_pending_at
                ELSE CURRENT_TIMESTAMP
            END,
            last_attempt_at = CURRENT_TIMESTAMP,
            pending_config_version = excluded.pending_config_version,
            pending_generation_key = excluded.pending_generation_key,
            pending_material_epoch_key = NULL,
            attempt_serial = aggregation_build_health.attempt_serial + 1,
            pending_attempt_token = aggregation_build_health.attempt_serial + 1
        RETURNING pending_attempt_token
        """,
        (
            config_version, generation_key,
            MAX_AGGREGATION_HEALTH_COUNT,
            MAX_AGGREGATION_HEALTH_COUNT,
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError("aggregation build attempt was not created")
    return _validated_attempt_token(row["pending_attempt_token"])


def record_aggregation_build_failure(
    conn: sqlite3.Connection,
    config_version: str,
    generation_key: str,
    attempt_token: int,
    *,
    caught_exceptions: int = 0,
    fusion_failures: int = 0,
    material_epoch_key: str | None = None,
) -> None:
    """Accumulate safe failure counters for the active pending attempt."""

    if (
        isinstance(caught_exceptions, bool)
        or isinstance(fusion_failures, bool)
        or not isinstance(caught_exceptions, int)
        or not isinstance(fusion_failures, int)
        or caught_exceptions < 0
        or fusion_failures < 0
        or caught_exceptions + fusion_failures <= 0
    ):
        raise ValueError("aggregation failure counters must be non-negative and nonzero")
    failure_kind = (
        "exception_and_fusion"
        if caught_exceptions and fusion_failures
        else "exception" if caught_exceptions else "fusion_failure"
    )
    token = _validated_attempt_token(attempt_token)
    cursor = conn.execute(
        """
        UPDATE aggregation_build_health SET
            pending_caught_exceptions = MIN(
                pending_caught_exceptions + ?, ?
            ),
            pending_fusion_failures = MIN(
                pending_fusion_failures + ?, ?
            ),
            total_caught_exceptions = MIN(
                total_caught_exceptions + ?, ?
            ),
            total_fusion_failures = MIN(
                total_fusion_failures + ?, ?
            ),
            last_failure_config_version = ?,
            last_failure_generation_key = ?,
            last_failure_material_epoch_key = COALESCE(
                ?, pending_material_epoch_key
            ),
            last_failure_kind = ?,
            last_failure_at = CURRENT_TIMESTAMP
        WHERE id = 1 AND pending_config_version = ?
          AND pending_generation_key = ? AND pending_attempt_token = ?
        """,
        (
            caught_exceptions,
            MAX_AGGREGATION_HEALTH_COUNT,
            fusion_failures,
            MAX_AGGREGATION_HEALTH_COUNT,
            caught_exceptions,
            MAX_AGGREGATION_HEALTH_COUNT,
            fusion_failures,
            MAX_AGGREGATION_HEALTH_COUNT,
            config_version,
            generation_key,
            material_epoch_key,
            failure_kind,
            config_version,
            generation_key,
            token,
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("aggregation failure has no matching pending build")


def complete_aggregation_build(
    conn: sqlite3.Connection, config_version: str, generation_key: str,
    attempt_token: int, *,
    expected_node_count: int,
    material_epoch_key: str | None = None,
    embedding_client: object | None = None,
) -> None:
    """Acknowledge only an already-published, exact structural build result.

    The producer is the sole component allowed to mint publication authority,
    including the legitimate empty set.  Health completion merely verifies
    that exact authority against the returned node count; it must never repair
    or fabricate publication for a clean-returning partial/injected builder.
    """

    if (
        isinstance(expected_node_count, bool)
        or not isinstance(expected_node_count, int)
        or expected_node_count < 0
    ):
        raise ValueError("aggregation success node count must be non-negative")
    owned_transaction = not conn.in_transaction
    token = _validated_attempt_token(attempt_token)
    try:
        if owned_transaction:
            conn.execute("BEGIN IMMEDIATE")
        pending = conn.execute(
            "SELECT pending_config_version,pending_generation_key,"
            "pending_attempt_token "
            "FROM aggregation_build_health WHERE id=1"
        ).fetchone()
        if (
            pending is None
            or pending["pending_config_version"] != config_version
            or pending["pending_generation_key"] != generation_key
            or pending["pending_attempt_token"] != token
        ):
            raise RuntimeError("aggregation success has no matching pending build")

        from hymem.dreaming.aggregation_provenance import (
            load_current_aggregation_publication,
        )

        if material_epoch_key is None:
            material_row = conn.execute(
                "SELECT aggregation_material_epoch_key "
                "FROM aggregation_publication_state WHERE id=1"
            ).fetchone()
            material_epoch_key = (
                str(material_row["aggregation_material_epoch_key"])
                if material_row is not None
                and material_row["aggregation_material_epoch_key"] is not None
                else None
            )
        if material_epoch_key is None:
            raise RuntimeError(
                "aggregation success requires an exact current publication"
            )

        publication = load_current_aggregation_publication(
            conn, expected_generation_key=generation_key,
            expected_material_epoch_key=material_epoch_key,
            embedding_client=embedding_client,
        )
        if (
            publication is None
            or publication.config_version != config_version
            or len(publication.nodes) != expected_node_count
        ):
            raise RuntimeError(
                "aggregation success requires an exact current publication"
            )
        cursor = conn.execute(
            """
            UPDATE aggregation_build_health SET
                last_success_config_version = ?,
                last_success_generation_key = ?,
                last_success_material_epoch_key = ?,
                last_success_at = CURRENT_TIMESTAMP,
                pending_config_version = NULL,
                pending_generation_key = NULL,
                pending_attempt_token = NULL,
                pending_material_epoch_key = NULL,
                pending_attempts = 0,
                pending_caught_exceptions = 0,
                pending_fusion_failures = 0,
                first_pending_at = NULL,
                last_attempt_at = NULL
            WHERE id = 1 AND pending_config_version = ?
              AND pending_generation_key = ? AND pending_attempt_token = ?
            """,
            (
                config_version, generation_key, material_epoch_key,
                config_version, generation_key, token,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("aggregation success has no matching pending build")
        if owned_transaction:
            conn.execute("COMMIT")
    except BaseException:
        if owned_transaction and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def invalidate_aggregation_success(conn: sqlite3.Connection) -> None:
    """Invalidate a local success attestation after portable source changes.

    Aggregation nodes and this health ledger are deliberately not portable.
    Import callers preserve bounded audit counters but clear the success proof,
    so a configured destination must run a clean local build before it can be
    reported healthy.
    """

    conn.execute(
        "UPDATE aggregation_build_health SET "
        "last_success_config_version=NULL,last_success_at=NULL,"
        "last_success_generation_key=NULL,"
        "last_success_material_epoch_key=NULL WHERE id=1"
    )
    conn.execute("DELETE FROM aggregation_publication_state")


def aggregation_health_status(
    conn: sqlite3.Connection,
    *,
    enabled: bool,
    config_version: str | None,
    generation_key: str | None = None,
    embedding_client: object | None = None,
) -> dict[str, Any]:
    """Return the current config's health plus bounded historical counters."""

    if not isinstance(enabled, bool):
        raise ValueError("aggregation enabled state must be boolean")
    if enabled and not config_version:
        raise ValueError("enabled aggregation requires a config version")
    row = conn.execute(
        "SELECT * FROM aggregation_build_health WHERE id=1"
    ).fetchone()
    values = dict(row) if row is not None else {}
    pending_config = values.get("pending_config_version")
    last_success_config = values.get("last_success_config_version")
    from hymem.dreaming.aggregation_provenance import (
        load_current_aggregation_publication,
    )

    publication = (
        load_current_aggregation_publication(
            conn, expected_generation_key=generation_key,
            embedding_client=embedding_client,
        )
        if enabled else None
    )
    publication_matches = bool(
        publication is not None and publication.config_version == config_version
        and (generation_key is None or publication.generation_key == generation_key)
    )
    pending_generation = values.get("pending_generation_key")
    last_success_generation = values.get("last_success_generation_key")
    last_success_material = values.get("last_success_material_epoch_key")
    active_pending = bool(
        enabled and pending_config == config_version
        and (generation_key is None or pending_generation == generation_key)
    )
    pending = int(
        enabled
        and (
            active_pending
            or last_success_config != config_version
            or (
                generation_key is not None
                and last_success_generation != generation_key
            )
            or not publication_matches
            or (
                publication is not None
                and last_success_material != publication.material_epoch_key
            )
        )
    )
    return {
        "aggregation_enabled": enabled,
        "aggregation_config_version": config_version if enabled else None,
        "aggregation_generation_key": generation_key if enabled else None,
        "aggregation_publication_generation_key": (
            publication.generation_key if publication is not None else None
        ),
        "aggregation_publication_generation": (
            dict(publication.generation_binding)
            if publication is not None else None
        ),
        "aggregation_material_epoch_key": (
            publication.material_epoch_key if publication is not None else None
        ),
        "aggregation_material_revision": (
            publication.material_revision if publication is not None else None
        ),
        "aggregation_material_binding": (
            dict(publication.material_binding)
            if publication is not None else None
        ),
        "pending_aggregation": pending,
        "aggregation_active_build_attempts": (
            int(values.get("pending_attempts", 0)) if active_pending else 0
        ),
        "aggregation_active_caught_exceptions": (
            int(values.get("pending_caught_exceptions", 0))
            if active_pending else 0
        ),
        "aggregation_active_fusion_failures": (
            int(values.get("pending_fusion_failures", 0))
            if active_pending else 0
        ),
        "aggregation_total_caught_exceptions": int(
            values.get("total_caught_exceptions", 0)
        ),
        "aggregation_total_fusion_failures": int(
            values.get("total_fusion_failures", 0)
        ),
        "aggregation_superseded_pending_configs": int(
            values.get("superseded_pending_configs", 0)
        ),
        "aggregation_last_success_config_version": last_success_config,
        "aggregation_last_success_generation_key": (
            last_success_generation if enabled else None
        ),
        "aggregation_last_success_material_epoch_key": (
            last_success_material if enabled else None
        ),
        "aggregation_last_success_at": values.get("last_success_at"),
        "aggregation_last_failure_config_version": values.get(
            "last_failure_config_version"
        ),
        "aggregation_last_failure_generation_key": values.get(
            "last_failure_generation_key"
        ),
        "aggregation_last_failure_material_epoch_key": values.get(
            "last_failure_material_epoch_key"
        ) if enabled else None,
        "aggregation_pending_material_epoch_key": values.get(
            "pending_material_epoch_key"
        ) if active_pending else None,
        "aggregation_last_failure_kind": values.get("last_failure_kind"),
        "aggregation_last_failure_at": values.get("last_failure_at"),
        "aggregation_stale_pending_config_version": (
            pending_config
            if pending_config is not None and pending_config != config_version
            else None
        ),
        "aggregation_stale_pending_generation_key": (
            pending_generation
            if pending_generation is not None
            and generation_key is not None
            and pending_generation != generation_key else None
        ),
    }
