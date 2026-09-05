"""Dependency-neutral knowledge-graph SQL semantics."""

from __future__ import annotations

import re

from hymem.core.time import EVENT_CLOCK_SKEW_SECONDS


_SQL_ALIAS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SQL_COLUMN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\Z"
)


def bounded_graph_clock_sql(column: str) -> str:
    """Return a canonical graph-clock value only when usable *now*.

    Ranking and destructive maintenance must not let malformed or scheduled
    future legacy metadata outrank a valid present/past coordinate.  The raw
    SQL identifier is restricted because this fragment is interpolated into
    otherwise-static queries.
    """
    if _SQL_COLUMN.fullmatch(column) is None:
        raise ValueError("graph clock column must be a SQL identifier")
    cutoff = (
        "strftime('%Y-%m-%dT%H:%M:%fZ','now',"
        f"'+{EVENT_CLOCK_SKEW_SECONDS} seconds')"
    )
    return (
        f"CASE WHEN hymem_timestamp_at_or_before({column},{cutoff})=1 "
        f"THEN hymem_normalize_iso_timestamp({column}) ELSE NULL END"
    )


def graph_clock_order_sql(column: str, *, descending: bool = True) -> str:
    """Stable instant-aware ordering fragment; unusable clocks sort last."""
    direction = "DESC" if descending else "ASC"
    bounded = bounded_graph_clock_sql(column)
    return f"({bounded} IS NOT NULL) DESC, {bounded} {direction}"


def live_edge_predicate(
    alias: str | None = None,
    *,
    include_derived: bool = False,
    require_positive_majority: bool = True,
) -> str:
    """Return the canonical SQL predicate for a live graph edge.

    ``alias`` is restricted to a SQL identifier because callers interpolate
    this fragment into static queries. Direct observations are the default; an
    intentional inference/count consumer must explicitly opt in.
    """
    if alias is not None and not _SQL_ALIAS.fullmatch(alias):
        raise ValueError("knowledge-graph SQL alias must be an identifier")
    prefix = f"{alias}." if alias else ""
    outer_id = f"{alias}.id" if alias else "knowledge_graph.id"
    parts = [
        f"{prefix}status = 'active'",
        f"{prefix}invalid_at IS NULL",
    ]
    if require_positive_majority:
        parts.append(f"{prefix}pos_evidence > {prefix}neg_evidence")
    parts.extend([
        # NULL retains pre-valid-time/manual compatibility. A non-NULL value is
        # normalized and compared by the same Python parser used at ingestion;
        # SQLite's permissive Julian-date grammar never sees the raw value.
        f"({prefix}valid_at IS NULL OR ("
        f"hymem_timestamp_at_or_before("
        f"{prefix}valid_at, "
        f"strftime('%Y-%m-%dT%H:%M:%fZ', 'now', "
        f"'+{EVENT_CLOCK_SKEW_SECONDS} seconds')) = 1))",
        # A canonical extraction is authority only after its whole-chunk
        # outcome was durably published. Legacy/synthetic evidence has no such
        # producer contract. Requiring every current canonical row to have a
        # matching publication makes partial/corrupt claim writes fail closed.
        f"NOT EXISTS ("
        f"SELECT 1 FROM kg_evidence hymem_ev "
        f"WHERE hymem_ev.edge_id={outer_id} "
        f"AND hymem_ev.provenance_status='canonical' "
        f"AND hymem_ev.is_current=1 "
        f"AND (hymem_normalize_iso_timestamp(hymem_ev.published_at) IS NULL "
        f"OR hymem_event_clock_is_valid("
        f"hymem_ev.source_event_at,hymem_ev.extracted_at)<>1 "
        f"OR hymem_timestamp_at_or_before("
        f"hymem_ev.extracted_at,hymem_ev.published_at)<>1 "
        f"OR hymem_timestamp_at_or_before("
        f"hymem_ev.published_at,"
        f"strftime('%Y-%m-%dT%H:%M:%fZ', 'now', "
        f"'+{EVENT_CLOCK_SKEW_SECONDS} seconds'))<>1 "
        f"OR (hymem_ev.polarity=1 AND NOT EXISTS ("
        f"SELECT 1 FROM kg_edge_lifecycle hymem_lifecycle "
        f"WHERE hymem_lifecycle.edge_id=hymem_ev.edge_id "
        f"AND hymem_lifecycle.source_evidence_id=hymem_ev.id "
        f"AND hymem_lifecycle.event_kind='claim_assertion' "
        f"AND hymem_lifecycle.direction=1 "
        f"AND hymem_lifecycle.event_at=hymem_ev.source_event_at "
        f"AND hymem_event_clock_is_valid("
        f"hymem_lifecycle.event_at,hymem_lifecycle.created_at)=1 "
        f"AND hymem_timestamp_at_or_before("
        f"hymem_ev.extracted_at,hymem_lifecycle.created_at)=1 "
        f"AND hymem_timestamp_at_or_before("
        f"hymem_lifecycle.created_at,hymem_ev.published_at)=1 "
        f"AND hymem_timestamp_at_or_before("
        f"hymem_lifecycle.created_at,"
        f"strftime('%Y-%m-%dT%H:%M:%fZ', 'now', "
        f"'+{EVENT_CLOCK_SKEW_SECONDS} seconds'))=1)) "
        f"OR NOT EXISTS ("
        f"SELECT 1 FROM kg_claim_observations hymem_obs "
        f"JOIN kg_claim_extraction_outcomes hymem_out "
        f"ON hymem_out.chunk_id=hymem_obs.chunk_id "
        f"AND hymem_out.prompt_version=hymem_obs.prompt_version "
        f"AND hymem_out.prompt_generation=hymem_obs.prompt_generation "
        f"WHERE hymem_obs.evidence_id=hymem_ev.id "
        f"AND hymem_obs.edge_id=hymem_ev.edge_id "
        f"AND hymem_obs.source_session_id=hymem_ev.source_session_id "
        f"AND hymem_obs.source_message_id=hymem_ev.source_message_id "
        f"AND hymem_obs.evidence_kind=hymem_ev.evidence_kind "
        f"AND hymem_obs.polarity=hymem_ev.polarity "
        f"AND hymem_obs.interpretation_key=hymem_ev.interpretation_key "
        f"AND hymem_normalize_iso_timestamp(hymem_obs.observed_at) IS NOT NULL "
        f"AND hymem_normalize_iso_timestamp(hymem_out.succeeded_at) IS NOT NULL "
        f"AND hymem_timestamp_at_or_before("
        f"hymem_ev.extracted_at,hymem_obs.observed_at)=1 "
        f"AND hymem_timestamp_at_or_before("
        f"hymem_ev.published_at,hymem_out.succeeded_at)=1 "
        f"AND hymem_timestamp_gap_within("
        f"hymem_obs.observed_at,hymem_out.succeeded_at,"
        f"{EVENT_CLOCK_SKEW_SECONDS})=1 "
        f"AND hymem_timestamp_at_or_before("
        f"hymem_out.succeeded_at,"
        f"strftime('%Y-%m-%dT%H:%M:%fZ', 'now', "
        f"'+{EVENT_CLOCK_SKEW_SECONDS} seconds'))=1)))",
    ])
    if not include_derived:
        parts.append(f"{prefix}derived = 0")
    return " AND ".join(parts)
