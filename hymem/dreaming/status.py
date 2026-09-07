"""Read-only, current-policy dreaming backlog classification.

The runner owns all writes.  This module mirrors its cursor/rebuild/retry
decisions over one caller-owned SQLite read snapshot so operator, MCP, and
benchmark completion cannot infer success from Phase-1 chunks alone.
"""

from __future__ import annotations

import sqlite3

from hymem.config import HyMemConfig
from hymem.dreaming.chunks import source_materialization_config_version
from hymem.dreaming.digest import (
    active_episode_prompt_version,
    digest_config_version,
    digest_generation_is_recognized,
    digest_generation_matches_config,
    digest_retry_policy_version,
    digest_retry_is_quarantined,
    digest_retry_state_is_valid,
)
from hymem.dreaming.facts import (
    fact_cursor_retry_unit_key,
    fact_session_authority_is_valid,
    facts_config_version,
    facts_generation_is_recognized,
    facts_retry_policy_version,
    facts_retry_state_is_valid,
    facts_tail_message_id,
    next_fact_outcome_for_replay,
)
from hymem.dreaming.lossless import (
    COVERAGE_INTEGRITY_CONFIG_VERSION,
    lossless_cursor_is_valid,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.dreaming.user_profile import (
    PROFILE_PROMPT_VERSION,
    profile_config_version,
    profile_generation_is_recognized,
    profile_generation_matches_config,
    profile_retry_policy_version,
    profile_retry_is_quarantined,
    profile_retry_state_is_valid,
    profile_user_tail_message_id,
)


DREAM_STATUS_SCHEMA_VERSION = "hymem-dream-status-v6"
DURABLE_PENDING_FIELDS = (
    "pending_source_materialization",
    "pending_chunks",
    "pending_digests",
    "pending_profiles",
    "pending_facts",
    "pending_aggregation",
)
DURABLE_MALFORMED_FIELDS = (
    "malformed_source_materialization",
    "malformed_digests",
    "malformed_profiles",
    "malformed_facts",
)

# Exact v6 health projection used by completion-claim consumers. V3 made
# Phase-1 completion producer-generation-aware; a v2 prompt-only zero cannot
# be relabelled as complete under another model. V5 applies the same boundary
# to aggregation publications and exposes their validated stored binding. The status
# payload intentionally carries additional observability/configuration fields,
# so consumers cannot reject every extension.  They must, however, require
# these fields and fail closed on any unclassified health-like extension.
DREAM_STATUS_BLOCKING_COUNT_FIELDS = (
    "pending_source_materialization",
    "pending_chunks",
    "pending_digests",
    "pending_profiles",
    "pending_facts",
    "quarantined_chunks",
    "quarantined_digests",
    "quarantined_profiles",
    "quarantined_facts",
    "quarantined_facts_malformed",
    "terminal_loss_chunks",
    "coverage_integrity_failures",
    "malformed_source_materialization",
    "malformed_digests",
    "malformed_profiles",
    "malformed_facts",
    "pending_aggregation",
)
DREAM_STATUS_BOOLEAN_GATE_FIELDS = ("in_progress",)
# Phase-1 backlog counts are authoritative only when evaluated against an
# exact configured producer generation.  These are metadata, not numeric
# pending counters.
DREAM_STATUS_PHASE1_AUTHORITY_FIELDS = (
    "phase1_backlog_status",
    "pending_chunks_authoritative",
    "phase1_generation_key",
)
DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS = (
    "aggregation_generation_key",
    "aggregation_publication_generation_key",
    "aggregation_last_success_generation_key",
)
DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS = (
    "aggregation_material_epoch_key",
    "aggregation_last_success_material_epoch_key",
)
DREAM_STATUS_HEALTH_DETAIL_FIELDS = (
    "terminal_loss_reasons",
    "coverage_integrity_failure_reasons",
    "coverage_integrity_failure_details",
    "coverage_integrity_failure_details_truncated",
    "coverage_integrity_config_version",
    "aggregation_publication_generation",
    "aggregation_material_binding",
)

# These current payload fields are required bounded diagnostics/configuration,
# not independent completion gates.  The blocking current aggregation state is
# represented by pending_aggregation above.
DREAM_STATUS_DIAGNOSTIC_COUNT_FIELDS = (
    "aggregation_active_build_attempts",
    "aggregation_active_caught_exceptions",
    "aggregation_active_fusion_failures",
    "aggregation_total_caught_exceptions",
    "aggregation_total_fusion_failures",
    "aggregation_superseded_pending_configs",
    "extraction_provider_attempt_budget",
)
DREAM_STATUS_DIAGNOSTIC_BOOLEAN_FIELDS = ("aggregation_enabled",)
DREAM_STATUS_DIAGNOSTIC_CONFIG_VERSION_FIELDS = (
    "aggregation_config_version",
    "aggregation_last_success_config_version",
    "aggregation_last_failure_config_version",
    "aggregation_stale_pending_config_version",
)
DREAM_STATUS_DIAGNOSTIC_TIMESTAMP_FIELDS = (
    "aggregation_last_success_at",
    "aggregation_last_failure_at",
)
DREAM_STATUS_DIAGNOSTIC_ENUM_FIELDS = ("aggregation_last_failure_kind",)
DREAM_STATUS_DIAGNOSTIC_NULLABLE_TEXT_FIELDS = (
    *DREAM_STATUS_DIAGNOSTIC_CONFIG_VERSION_FIELDS,
    *DREAM_STATUS_DIAGNOSTIC_TIMESTAMP_FIELDS,
    *DREAM_STATUS_DIAGNOSTIC_ENUM_FIELDS,
    *DREAM_STATUS_AGGREGATION_AUTHORITY_FIELDS,
    *DREAM_STATUS_AGGREGATION_MATERIAL_AUTHORITY_FIELDS,
    "aggregation_last_failure_generation_key",
    "aggregation_stale_pending_generation_key",
    "aggregation_pending_material_epoch_key",
    "aggregation_last_failure_material_epoch_key",
)
DREAM_STATUS_SANCTIONED_HEALTH_DIAGNOSTIC_FIELDS = (
    *DREAM_STATUS_DIAGNOSTIC_COUNT_FIELDS,
    *DREAM_STATUS_DIAGNOSTIC_BOOLEAN_FIELDS,
    *DREAM_STATUS_DIAGNOSTIC_NULLABLE_TEXT_FIELDS,
)
DREAM_STATUS_RECOGNIZED_HEALTH_FIELDS = frozenset((
    "dream_status_schema",
    *DREAM_STATUS_BLOCKING_COUNT_FIELDS,
    *DREAM_STATUS_BOOLEAN_GATE_FIELDS,
    *DREAM_STATUS_PHASE1_AUTHORITY_FIELDS,
    *DREAM_STATUS_HEALTH_DETAIL_FIELDS,
    *DREAM_STATUS_SANCTIONED_HEALTH_DIAGNOSTIC_FIELDS,
    "aggregation_material_revision",
))

# Broad on purpose: an added field capable of changing a health conclusion
# must either join the explicit schema above or force consumers to report the
# snapshot as unverified under the unchanged schema marker.
DREAM_STATUS_HEALTH_NAME_FRAGMENTS = (
    "pending",
    "malformed",
    "quarantin",
    "budget",
    "coverage_integrity",
    "fail",
    "failure",
    "error",
    "exception",
    "loss",
    "health",
    "unhealthy",
    "stuck",
    "blocked",
    "blocking",
    "complete",
    "success",
    "ready",
    "remaining",
    "retry",
    "skipped",
    "timeout",
    "deadline",
    "degraded",
    "dirty",
    "active",
    "backlog",
    "queue",
    "lag",
    "owed",
    "lock",
    "invalid",
    "corrupt",
    "poison",
    "gate",
    "in_progress",
)


def is_health_like_dream_status_field(field_name: object) -> bool:
    """Return whether an extension name could alter a completion claim."""

    if not isinstance(field_name, str):
        return True
    normalized = field_name.strip().lower()
    return any(
        fragment in normalized
        for fragment in DREAM_STATUS_HEALTH_NAME_FRAGMENTS
    )


def _int_or_none(value: object) -> bool:
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value >= 1
    )


def _cursor_shape_is_valid(
    cursor_message_id: object,
    partial_message_id: object,
    offset: object,
) -> bool:
    return bool(
        _int_or_none(cursor_message_id)
        and _int_or_none(partial_message_id)
        and isinstance(offset, int)
        and not isinstance(offset, bool)
        and offset >= 0
        and ((partial_message_id is None) == (offset == 0))
    )


def _source_materialization_status(
    conn: sqlite3.Connection, cfg: HyMemConfig, rows: list[sqlite3.Row]
) -> dict[str, int]:
    current_producer = source_materialization_config_version(
        min_chars=cfg.salience_min_chars
    )
    pending = 0
    malformed = 0
    for row in rows:
        session_id = row["id"]
        coverage_tail = row["coverage_message_id"]
        materialized_tail = row["source_materialized_message_id"]
        producer = row["source_materialization_config_version"]

        raw_tail_row = conn.execute(
            "SELECT MAX(id) AS message_id FROM messages WHERE session_id=?",
            (session_id,),
        ).fetchone()
        raw_tail = raw_tail_row["message_id"] if raw_tail_row else None
        has_uncovered_raw = bool(
            raw_tail is not None
            and (
                coverage_tail is None
                or not _int_or_none(coverage_tail)
                or int(raw_tail) > int(coverage_tail)
            )
        )

        state_malformed = bool(
            not _int_or_none(coverage_tail)
            or not _int_or_none(materialized_tail)
            or (producer is not None and (
                not isinstance(producer, str) or not producer.strip()
            ))
            or (materialized_tail is not None and coverage_tail is None)
            or (
                materialized_tail is not None
                and coverage_tail is not None
                and _int_or_none(materialized_tail)
                and _int_or_none(coverage_tail)
                and int(materialized_tail) > int(coverage_tail)
            )
        )
        if coverage_tail is not None and _int_or_none(coverage_tail):
            state_malformed = state_malformed or not lossless_cursor_is_valid(
                conn, session_id, int(coverage_tail), None, 0
            )
            # A live raw row below the producer frontier is an independent,
            # indexed witness that its current exact proof must exist. This
            # catches sparse/manual frontier advancement without rescanning
            # retained-away history.
            missing = conn.execute(
                "SELECT 1 FROM messages m WHERE m.session_id=? AND m.id<=? "
                "AND NOT EXISTS ("
                " SELECT 1 FROM message_retention_coverage mc "
                " JOIN chunks c ON c.id=mc.chunk_id "
                " WHERE mc.message_id=m.id "
                " AND mc.source_session_id=m.session_id "
                " AND mc.coverage_version=? "
                " AND c.chunk_kind='coverage'"
                ") LIMIT 1",
                (session_id, int(coverage_tail), LOSSLESS_COVERAGE_VERSION),
            ).fetchone()
            state_malformed = state_malformed or missing is not None

        has_covered_source = coverage_tail is not None
        acknowledgement_current = bool(
            has_covered_source
            and not state_malformed
            and producer == current_producer
            and materialized_tail == coverage_tail
        )
        if has_uncovered_raw or (has_covered_source and not acknowledgement_current):
            pending += 1
        if state_malformed:
            malformed += 1
    return {
        "pending_source_materialization": pending,
        "malformed_source_materialization": malformed,
    }


def _digest_status(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    rows: list[sqlite3.Row],
) -> dict[str, int]:
    episode_prompt = active_episode_prompt_version(
        cfg.episode_granularity_enabled
    )
    current_config = digest_config_version(
        prompt_version=cfg.prompt_version,
        episode_prompt_version=episode_prompt,
        max_chars=cfg.dream_digest_max_chars,
        max_tokens=cfg.dream_digest_max_tokens,
        max_episodes=(
            cfg.dream_max_episodes_per_session
            if cfg.episode_granularity_enabled else None
        ),
    )
    pending = 0
    quarantined = 0
    malformed = 0
    for row in rows:
        stored = row["digest_cursor_prompt_version"]
        published = row["digest_published_generation"]
        cursor = row["digest_cursor_message_id"]
        partial = row["digest_cursor_partial_message_id"]
        offset = row["digest_cursor_offset"]
        coverage_tail = row["coverage_message_id"]

        shape_valid = _cursor_shape_is_valid(cursor, partial, offset)
        stored_recognized = stored is None or digest_generation_is_recognized(stored)
        published_recognized = (
            published is None or digest_generation_is_recognized(published)
        )
        retry_valid = digest_retry_state_is_valid(
            row["digest_retry_count"],
            row["digest_retry_config_version"],
            row["digest_quarantined"],
        )
        state_malformed = bool(
            not shape_valid
            or not stored_recognized
            or not published_recognized
            or not retry_valid
            or (stored is None and (cursor is not None or partial is not None or offset != 0))
            or (published is not None and stored is None)
        )

        cursor_current = bool(
            shape_valid and digest_generation_matches_config(stored, current_config)
        )
        cursor_invalid = bool(
            cursor_current
            and not lossless_cursor_is_valid(
                conn, row["id"], cursor, partial, int(offset),
            )
        )
        state_malformed = state_malformed or cursor_invalid
        if cursor_invalid:
            cursor_current = False

        published_current = bool(
            row["digested_prompt_version"] == cfg.prompt_version
            and row["episodes_prompt_version"] == episode_prompt
            and digest_generation_matches_config(published, current_config)
            and published == stored
        )
        caught_up = bool(
            coverage_tail is None
            or (
                cursor_current
                and partial is None
                and int(offset) == 0
                and cursor is not None
                and cursor == coverage_tail
            )
        )
        requires_rebuild = bool(
            cursor_current
            and not published_current
            and (caught_up or stored == published)
        )
        work_exists = bool(
            coverage_tail is not None and (not caught_up or requires_rebuild)
        )
        retry_key = digest_retry_policy_version(
            current_config,
            max_attempts=cfg.digest_extraction_max_attempts,
            rebuild_from=(stored if requires_rebuild or cursor_invalid else None),
            invalidated_stamp=(
                ("invalid-cursor" if cursor_invalid else row["digested_prompt_version"])
                if requires_rebuild or cursor_invalid else None
            ),
        )
        active_quarantine = bool(work_exists and digest_retry_is_quarantined(
            row["digest_retry_count"],
            row["digest_retry_config_version"],
            retry_key=retry_key,
            max_attempts=cfg.digest_extraction_max_attempts,
        ))
        if work_exists and not active_quarantine:
            pending += 1
        if active_quarantine:
            quarantined += 1
        if state_malformed:
            malformed += 1
    return {
        "pending_digests": pending,
        "quarantined_digests": quarantined,
        "malformed_digests": malformed,
    }


def _profile_status(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    rows: list[sqlite3.Row],
) -> dict[str, int]:
    if not cfg.profile_extraction_enabled:
        return {
            "pending_profiles": 0,
            "quarantined_profiles": 0,
            "malformed_profiles": 0,
        }
    current_config = profile_config_version(
        max_chars=cfg.dream_digest_max_chars,
        max_items=cfg.profile_max_items_per_session,
        redact_values=cfg.redact_secrets,
    )
    pending = 0
    quarantined = 0
    malformed = 0
    for row in rows:
        stored = row["profile_cursor_prompt_version"]
        published = row["profile_published_generation"]
        cursor = row["profile_cursor_message_id"]
        partial = row["profile_cursor_partial_message_id"]
        offset = row["profile_cursor_offset"]
        shape_valid = _cursor_shape_is_valid(cursor, partial, offset)
        retry_valid = profile_retry_state_is_valid(
            row["profile_retry_count"],
            row["profile_retry_config_version"],
            row["profile_quarantined"],
        )
        state_malformed = bool(
            not shape_valid
            or (stored is not None and not profile_generation_is_recognized(stored))
            or (
                published is not None
                and not profile_generation_is_recognized(published)
            )
            or not retry_valid
            or (stored is None and (cursor is not None or partial is not None or offset != 0))
            or (published is not None and stored is None)
        )
        try:
            tail = profile_user_tail_message_id(conn, row["id"])
        except (RuntimeError, TypeError, ValueError):
            tail = None
            state_malformed = True
        cursor_current = bool(
            shape_valid and profile_generation_matches_config(stored, current_config)
        )
        cursor_invalid = bool(
            cursor_current
            and not lossless_cursor_is_valid(
                conn,
                row["id"],
                cursor,
                partial,
                int(offset),
                roles=frozenset({"user"}),
            )
        )
        state_malformed = state_malformed or cursor_invalid
        if cursor_invalid:
            cursor_current = False
        published_current = bool(
            row["profile_prompt_version"] == PROFILE_PROMPT_VERSION
            and profile_generation_matches_config(published, current_config)
            and published == stored
        )
        caught_up = bool(
            tail is None
            or (
                cursor_current
                and partial is None
                and int(offset) == 0
                and cursor is not None
                and cursor == tail
            )
        )
        requires_rebuild = bool(
            cursor_current
            and not published_current
            and (caught_up or stored == published)
        )
        # Even an empty USER stream owes a current zero-call publication marker.
        work_exists = bool(
            (tail is None and not published_current)
            or (tail is not None and (not caught_up or requires_rebuild))
        )
        retry_key = profile_retry_policy_version(
            current_config,
            max_attempts=cfg.profile_extraction_max_attempts,
            rebuild_from=(stored if requires_rebuild or cursor_invalid else None),
            invalidated_stamp=(
                ("invalid-cursor" if cursor_invalid else row["profile_prompt_version"])
                if requires_rebuild or cursor_invalid else None
            ),
        )
        active_quarantine = bool(
            tail is not None
            and work_exists
            and profile_retry_is_quarantined(
                row["profile_retry_count"],
                row["profile_retry_config_version"],
                retry_key=retry_key,
                max_attempts=cfg.profile_extraction_max_attempts,
            )
        )
        if work_exists and not active_quarantine:
            pending += 1
        if active_quarantine:
            quarantined += 1
        if state_malformed:
            malformed += 1
    return {
        "pending_profiles": pending,
        "quarantined_profiles": quarantined,
        "malformed_profiles": malformed,
    }


def _fact_status(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    rows: list[sqlite3.Row],
) -> dict[str, int]:
    if not cfg.facts_extraction_enabled:
        return {
            "pending_facts": 0,
            "quarantined_facts": 0,
            "quarantined_facts_malformed": 0,
            "malformed_facts": 0,
        }
    current_config = facts_config_version(cfg)
    pending = 0
    quarantined = 0
    quarantined_malformed = 0
    malformed = 0
    for row in rows:
        cursor = row["facts_cursor_message_id"]
        partial = row["facts_cursor_partial_message_id"]
        offset = row["facts_cursor_offset"]
        marker = row["facts_cursor_prompt_version"]
        retry_valid = facts_retry_state_is_valid(
            row["facts_retry_count"],
            row["facts_retry_config_version"],
            row["facts_quarantined"],
        )
        state_malformed = bool(
            not _cursor_shape_is_valid(cursor, partial, offset)
            or (marker is not None and not facts_generation_is_recognized(marker))
            or not retry_valid
            or row["facts_message_id"] != cursor
        )
        cursor_valid = bool(
            not state_malformed
            and lossless_cursor_is_valid(
                conn,
                row["id"],
                cursor,
                partial,
                int(offset),
                roles=frozenset({"user", "assistant"}),
            )
        )
        state_malformed = state_malformed or not cursor_valid
        try:
            tail = facts_tail_message_id(conn, row["id"])
            stale_slice = (
                next_fact_outcome_for_replay(conn, row["id"], current_config)
                if cursor_valid else None
            )
        except (RuntimeError, TypeError, ValueError):
            tail = None
            stale_slice = None
            state_malformed = True
        caught_up = bool(
            cursor_valid
            and partial is None
            and (tail is None or cursor == tail)
        )
        if cursor_valid and stale_slice is None:
            # Completion is authoritative, not just coordinate-shaped: verify
            # every bounded manifest/result/lifecycle proof before reporting a
            # current fact chain as healthy. This streams in source order and
            # makes no provider calls, but is intentionally O(fact history).
            state_malformed = (
                state_malformed
                or not fact_session_authority_is_valid(conn, row["id"])
            )
        model_work_exists = bool(
            cursor_valid
            and not state_malformed
            and (stale_slice is not None or not caught_up)
        )
        local_stamp_owed = bool(
            cursor_valid and not state_malformed
            and stale_slice is None and caught_up
            and marker != current_config
        )
        work_exists = bool(model_work_exists or local_stamp_owed)
        safe_offset = (
            int(offset)
            if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0
            else 0
        )
        retry_unit = stale_slice or fact_cursor_retry_unit_key(
            row["id"], cursor if _int_or_none(cursor) else None,
            partial if _int_or_none(partial) else None, safe_offset
        )
        retry_key = facts_retry_policy_version(
            cfg, replay_slice_key=retry_unit
        )
        # A current-unit retry cannot legitimately remain after all provider
        # work and the publication marker are complete.  It is neither a
        # quarantine nor latent work the runner can heal, so fail closed.
        if (
            not model_work_exists
            and marker == current_config
            and retry_valid
            and row["facts_retry_config_version"] == retry_key
            and int(row["facts_retry_count"] or 0) > 0
        ):
            state_malformed = True
        active_quarantine = bool(
            model_work_exists
            and not state_malformed
            and retry_valid
            and cfg.facts_extraction_max_attempts > 0
            and row["facts_retry_config_version"] == retry_key
            and int(row["facts_retry_count"] or 0)
            >= cfg.facts_extraction_max_attempts
            and int(row["facts_quarantined"] or 0) == 1
        )
        if work_exists and not active_quarantine:
            pending += 1
        if active_quarantine:
            quarantined += 1
        if row["facts_quarantined"] == 1 and (
            state_malformed or not retry_valid
        ):
            quarantined_malformed += 1
        if state_malformed:
            malformed += 1
    return {
        "pending_facts": pending,
        "quarantined_facts": quarantined,
        "quarantined_facts_malformed": quarantined_malformed,
        "malformed_facts": malformed,
    }


def durable_fact_work_status(
    conn: sqlite3.Connection, cfg: HyMemConfig
) -> dict[str, int]:
    """Return the one authoritative fact work/quarantine classification."""

    rows = conn.execute(
        "SELECT id,facts_message_id,facts_cursor_message_id,"
        "facts_cursor_partial_message_id,facts_cursor_offset,"
        "facts_cursor_prompt_version,facts_retry_count,"
        "facts_retry_config_version,facts_quarantined "
        "FROM sessions ORDER BY id"
    ).fetchall()
    return _fact_status(conn, cfg, rows)


def durable_dream_work_status(
    conn: sqlite3.Connection, cfg: HyMemConfig
) -> dict[str, int | str]:
    """Return every durable non-embedding work class for one read snapshot.

    Counts use sessions as their unit. A unit appears in ``pending_*`` only
    while the current policy can retry it; its exact active quarantine moves it
    to the corresponding quarantine counter. Malformed state is additive and
    never silently disappears from health merely because no ordinary tail is
    visible. Disabled optional subsystems return exact zeroes.
    """

    rows = conn.execute(
        "SELECT id,coverage_message_id,source_materialized_message_id,"
        "source_materialization_config_version,"
        "digested_prompt_version,episodes_prompt_version,"
        "digest_cursor_message_id,digest_cursor_partial_message_id,"
        "digest_cursor_offset,digest_cursor_prompt_version,"
        "digest_published_generation,digest_retry_count,"
        "digest_retry_config_version,digest_quarantined,"
        "profile_prompt_version,profile_cursor_message_id,"
        "profile_cursor_partial_message_id,profile_cursor_offset,"
        "profile_cursor_prompt_version,profile_published_generation,"
        "profile_retry_count,profile_retry_config_version,profile_quarantined,"
        "facts_message_id,facts_cursor_message_id,"
        "facts_cursor_partial_message_id,facts_cursor_offset,"
        "facts_cursor_prompt_version,facts_retry_count,"
        "facts_retry_config_version,facts_quarantined FROM sessions ORDER BY id"
    ).fetchall()
    return {
        "dream_status_schema": DREAM_STATUS_SCHEMA_VERSION,
        **_source_materialization_status(conn, cfg, rows),
        **_digest_status(conn, cfg, rows),
        **_profile_status(conn, cfg, rows),
        **_fact_status(conn, cfg, rows),
    }
