"""Auditable current-state and valid-time knowledge-graph reads.

The write side stores two different clocks:

* ``kg_edge_lifecycle.event_at`` is source/valid time -- when the claim was
  true in the represented world.
* lifecycle ``created_at`` and evidence ``extracted_at``/``superseded_at`` are
  transaction time -- when a particular evidence/lifecycle revision was
  available to HyMem.

This module keeps those clocks separate.  The small SQL predicate helper is
also the single definition used by live readers so ``status='active'`` cannot
silently drift away from the invalidation and evidence-majority guards.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

from hymem.core.graph import live_edge_predicate
from hymem.core.time import (
    EVENT_CLOCK_SKEW_SECONDS,
    event_clock_is_valid,
    normalize_iso_timestamp,
    timestamp_at_or_before,
)
from hymem.dreaming.canonicalize import resolve
from hymem.dreaming.lossless import validate_message_coverage_artifact


GRAPH_CITATION_LIMIT = 5


@dataclass(frozen=True)
class GraphEvidenceCitation:
    """One exact source behind a graph fact.

    Source identity fields are nullable only for explicitly unattributed
    historical rows.  Current public ``GraphFact`` retrieval deliberately
    selects canonical sources, so its populated citations have real values.
    ``source_event_at`` is valid time; ``recorded_at`` is transaction time.
    """

    evidence_id: int
    evidence_kind: str
    source_role: str | None
    source_session_id: str | None
    source_message_id: int | None
    source_event_at: str | None
    source_created_at: str | None
    temporal_scope: str | None
    recorded_at: str
    coverage_chunk_id: str | None
    coverage_version: str | None
    extraction_chunk_id: str
    currently_authoritative: bool
    authoritative_at_recorded_time: bool
    provenance_status: str
    source_peer_id: str | None = None
    source_workspace_id: str | None = None


@dataclass(frozen=True)
class AsOfGraphFact:
    """A direct fact valid at one source-time coordinate.

    This intentionally has no confidence/counter fields: today's materialized
    evidence counters would be false metadata for a historical transaction
    slice. ``citations`` are the positive assertion sources in the interval as
    known at ``recorded_at`` (or under current authority when it is ``None``).
    Intervals are half-open: ``valid_at <= t < invalid_at``. Entity names are
    projected onto today's canonical topology because alias merges are not
    transaction-versioned; ``recorded_at`` versions authority, not topology.
    """

    edge_id: int
    subject: str
    predicate: str
    object: str
    valid_at: str
    invalid_at: str | None
    as_of: str
    recorded_at: str | None
    citations: list[GraphEvidenceCitation] = field(default_factory=list)
    derived: bool = False


def normalize_public_timestamp(value: str, *, field_name: str) -> str:
    """Validate an ISO-8601 API timestamp and normalize it to UTC milliseconds.

    Date-only values mean midnight UTC.  Naive date-times are interpreted as
    UTC (the same deterministic convention used for host message timestamps);
    timezone-aware values are converted to UTC.  Malformed and boolean-like
    inputs are rejected rather than mapped to the ancient internal sentinel.
    """
    try:
        return normalize_iso_timestamp(value, context=field_name)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp") from exc


def _current_authoritative_evidence(
    conn: sqlite3.Connection,
    *,
    edge_id: int | None = None,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[sqlite3.Row]:
    """Load current canonical evidence whose publication is authoritative.

    Scope predicates are part of the SQL candidate population. The subsequent
    clock/outcome check is the same one lifecycle replay uses, so malformed or
    unpublished rows cannot influence scoped counts, confidence, or ranking.
    """
    clauses = [
        "ev.provenance_status = 'canonical'",
        "ev.is_current = 1",
    ]
    params: list[object] = []
    if edge_id is not None:
        clauses.append("ev.edge_id = ?")
        params.append(int(edge_id))
    if source_session_id is not None:
        clauses.append("ev.source_session_id = ?")
        params.append(source_session_id)
    if source_workspace_id is not None:
        clauses.append("ev.source_workspace_id = ?")
        params.append(source_workspace_id)
    if source_peer_id is not None:
        clauses.append("ev.source_peer_id = ?")
        params.append(source_peer_id)
    rows = conn.execute(
        f"""
        SELECT ev.id AS evidence_id, ev.id, ev.edge_id, ev.polarity,
               ev.evidence_weight, ev.source_role, ev.source_peer_id,
               ev.source_workspace_id, ev.source_session_id,
               ev.source_message_id, ev.source_created_at,
               ev.source_coverage_chunk_id, ev.source_coverage_version,
               ev.surface_subject, ev.surface_object,
               ev.provenance_status, ev.is_current, ev.extracted_at,
               ev.published_at, ev.superseded_at, ev.source_event_at,
               julianday(hymem_normalize_iso_timestamp(ev.source_event_at))
                   AS event_jd,
               kg.subject_canonical AS s, kg.predicate AS p,
               kg.object_canonical AS o, kg.derived,
               (
                 SELECT MIN(hymem_normalize_iso_timestamp(
                                  outcome.succeeded_at))
                 FROM kg_claim_observations observation
                 JOIN kg_claim_extraction_outcomes outcome
                   ON outcome.chunk_id=observation.chunk_id
                  AND outcome.prompt_version=observation.prompt_version
                  AND outcome.prompt_generation=observation.prompt_generation
                 WHERE observation.evidence_id=ev.id
                   AND observation.edge_id=ev.edge_id
                   AND observation.source_session_id=ev.source_session_id
                   AND observation.source_message_id=ev.source_message_id
                   AND observation.evidence_kind=ev.evidence_kind
                   AND observation.polarity=ev.polarity
                   AND observation.interpretation_key=ev.interpretation_key
                   AND hymem_normalize_iso_timestamp(
                         observation.observed_at) IS NOT NULL
                   AND hymem_normalize_iso_timestamp(
                         outcome.succeeded_at) IS NOT NULL
                   AND hymem_timestamp_at_or_before(
                         ev.extracted_at, observation.observed_at
                       ) = 1
                   AND hymem_timestamp_gap_within(
                         observation.observed_at, outcome.succeeded_at,
                         {EVENT_CLOCK_SKEW_SECONDS}
                       ) = 1
               ) AS current_publication_at
        FROM kg_evidence ev
        JOIN knowledge_graph kg ON kg.id = ev.edge_id
        WHERE {' AND '.join(clauses)}
        ORDER BY ev.edge_id, ev.id
        """,
        params,
    ).fetchall()
    present_cutoff = conn.execute(
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')"
    ).fetchone()[0]
    return [
        row for row in rows
        if _evidence_authoritative_at(
            row, None, present_cutoff=present_cutoff
        )
    ]


def validated_current_evidence(
    conn: sqlite3.Connection,
    *,
    edge_id: int | None = None,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[sqlite3.Row]:
    """Current canonical evidence whose immutable source proof also verifies.

    This is the shared authority projection for read ranking and conservative
    write routing. Publication/lifecycle validation happens in
    :func:`_current_authoritative_evidence`; this final pass binds every row to
    its exact retained source occurrence and rejects malformed event clocks.
    """
    valid: list[sqlite3.Row] = []
    for row in _current_authoritative_evidence(
        conn,
        edge_id=edge_id,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    ):
        try:
            proof = validate_message_coverage_artifact(
                conn,
                message_id=int(row["source_message_id"]),
                chunk_id=row["source_coverage_chunk_id"],
                coverage_version=row["source_coverage_version"],
            )
        except (RuntimeError, TypeError, ValueError, sqlite3.Error):
            continue
        if (
            proof.session_id != row["source_session_id"]
            or proof.role != row["source_role"]
            or proof.source_created_at != row["source_created_at"]
            or proof.source_peer_id != row["source_peer_id"]
            or proof.source_workspace_id != row["source_workspace_id"]
            or row["event_jd"] is None
        ):
            continue
        try:
            event_jd = float(row["event_jd"])
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(event_jd):
            continue
        valid.append(row)
    return valid


def validated_confidence_signal_totals(
    conn: sqlite3.Connection,
    *,
    edge_ids: list[int] | tuple[int, ...] | set[int] | None = None,
) -> dict[int, tuple[int, int]]:
    """Return strict, present-time confidence-signal totals by edge."""
    clauses = [
        "counts_toward_confidence=1",
        "typeof(edge_id)='integer'",
        "typeof(polarity)='integer'",
        "polarity IN (-1,1)",
        "typeof(evidence_weight)='integer'",
        "evidence_weight>=1",
        "hymem_normalize_iso_timestamp(created_at) IS NOT NULL",
        "hymem_timestamp_at_or_before("
        "created_at,strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds'))=1",
    ]
    params: list[object] = []
    if edge_ids is not None:
        ids = sorted({int(edge_id) for edge_id in edge_ids})
        if not ids:
            return {}
        clauses.append("edge_id IN (" + ",".join("?" for _ in ids) + ")")
        params.extend(ids)
    rows = conn.execute(
        "SELECT edge_id,"
        "COALESCE(SUM(CASE WHEN polarity=1 THEN evidence_weight ELSE 0 END),0) "
        "AS positive,"
        "COALESCE(SUM(CASE WHEN polarity=-1 THEN evidence_weight ELSE 0 END),0) "
        "AS negative FROM kg_evidence_signals WHERE "
        + " AND ".join(clauses)
        + " GROUP BY edge_id ORDER BY edge_id",
        params,
    ).fetchall()
    return {
        int(row["edge_id"]): (int(row["positive"]), int(row["negative"]))
        for row in rows
    }


def current_positive_citations(
    conn: sqlite3.Connection,
    edge_id: int,
    *,
    limit: int = GRAPH_CITATION_LIMIT,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[GraphEvidenceCitation]:
    """Return bounded positive authority for the edge's current open interval.

    Merely filtering ``kg_evidence.is_current`` is insufficient: a still-current
    assertion may belong to an interval that a manual/phase-3 event closed.
    Replaying eligible lifecycle events ensures assert A -> retract -> reassert
    B cites B, never A.
    """
    if limit <= 0:
        return []
    state = current_positive_state(
        conn,
        int(edge_id),
        limit=limit,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    return state[1] if state is not None else []


def current_positive_state(
    conn: sqlite3.Connection,
    edge_id: int,
    *,
    limit: int = GRAPH_CITATION_LIMIT,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> tuple[str, list[GraphEvidenceCitation]] | None:
    """Replay a current open interval inside one exact provenance scope.

    Scope is applied to evidence and dependent lifecycle events before state
    reduction. A close caused jointly by a different peer/workspace therefore
    cannot erase this scope's independent assertion history.
    """
    if limit <= 0:
        return None
    events = _eligible_lifecycle_events(
        conn,
        int(edge_id),
        cutoff=None,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    state_open = False
    opened_at: str | None = None
    evidence_ids: list[int] = []
    for event in events:
        if event.direction == 1:
            if not state_open:
                evidence_ids = []
                opened_at = event.event_at
            state_open = True
            if event.source_evidence_id is not None:
                evidence_ids.append(event.source_evidence_id)
        else:
            state_open = False
            opened_at = None
            evidence_ids = []
    if not state_open or opened_at is None:
        return None
    citations = _citations_by_id(
        conn,
        list(dict.fromkeys(evidence_ids)),
        authoritative_at_recorded_time=True,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
        limit=limit,
    )
    citations = [
        citation
        for citation in citations
        if citation.currently_authoritative
        and citation.provenance_status == "canonical"
    ][:limit]
    return (opened_at, citations) if citations else None


def facts_at(
    conn: sqlite3.Connection,
    valid_time: str,
    *,
    recorded_at: str | None = None,
    entity: str | None = None,
) -> list[AsOfGraphFact]:
    """Return direct facts valid at ``valid_time`` from lifecycle history.

    With ``recorded_at=None`` this uses today's authority eligibility, exactly
    like lifecycle materialization: superseded source revisions and stale
    dependencies are ignored.  Supplying ``recorded_at`` asks the independent
    authority-cutoff question: only events/revisions that existed at that
    cutoff and had not yet been superseded participate. Results are always
    projected onto the *current* canonical entity topology; merges rewrite
    graph endpoints and are not transaction-versioned, so this is deliberately
    not a literal historical topology snapshot.

    The materialized ``knowledge_graph.status``/``valid_at`` cache is never used
    to decide historical truth.  Retraction therefore closes an interval
    without erasing an earlier snapshot, and re-assertion opens a new interval.
    This lifecycle history is intentionally broader than ordinary current
    retrieval, whose live predicate also requires active/open materialized
    state and ``pos_evidence > neg_evidence``. A tie that has not produced a
    persisted close remains in lifecycle history but is hidden by current
    ``augment``/timeline/count reads.
    """
    target = normalize_public_timestamp(valid_time, field_name="valid_time")
    cutoff = (
        normalize_public_timestamp(recorded_at, field_name="recorded_at")
        if recorded_at is not None
        else None
    )
    canon = resolve(conn, entity) if entity is not None else None
    sql = """
        SELECT id, subject_canonical, predicate, object_canonical
        FROM knowledge_graph
        WHERE derived = 0
    """
    params: list[object] = []
    if canon is not None:
        sql += " AND (subject_canonical = ? OR object_canonical = ?)"
        params.extend((canon, canon))
    sql += " ORDER BY subject_canonical, predicate, object_canonical, id"

    results: list[AsOfGraphFact] = []
    for edge in conn.execute(sql, params).fetchall():
        events = _eligible_lifecycle_events(conn, int(edge["id"]), cutoff=cutoff)
        snapshot = _interval_at(events, target)
        if snapshot is None:
            continue
        valid_at, invalid_at, evidence_ids = snapshot
        citations = _citations_by_id(
            conn,
            evidence_ids,
            authoritative_at_recorded_time=True,
        )
        results.append(
            AsOfGraphFact(
                edge_id=int(edge["id"]),
                subject=edge["subject_canonical"],
                predicate=edge["predicate"],
                object=edge["object_canonical"],
                valid_at=valid_at,
                invalid_at=invalid_at,
                as_of=target,
                recorded_at=cutoff,
                citations=citations,
            )
        )
    return results


@dataclass(frozen=True)
class _LifecycleEvent:
    event_at: str
    event_key: str
    event_kind: str
    direction: int
    source_evidence_id: int | None
    causal_order: tuple[object, ...]


def _eligible_lifecycle_events(
    conn: sqlite3.Connection,
    edge_id: int,
    *,
    cutoff: str | None,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[_LifecycleEvent]:
    rows = conn.execute(
        """
        SELECT id, event_at, event_key, event_kind, direction,
               source_evidence_id, dependency_count, created_at
        FROM kg_edge_lifecycle
        WHERE edge_id = ?
        ORDER BY id
        """,
        (edge_id,),
    ).fetchall()
    present_cutoff = conn.execute(
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')"
    ).fetchone()[0]
    evidence_rows = {
        int(row["id"]): row
        for row in conn.execute(
            f"""
            SELECT id, edge_id, provenance_status, source_session_id,
                   source_message_id, evidence_kind, revision,
                   interpretation_key, chunk_id, is_current, extracted_at,
                   published_at, superseded_at, source_event_at,
                   source_peer_id, source_workspace_id,
                   (
                     SELECT MIN(hymem_normalize_iso_timestamp(
                                      outcome.succeeded_at))
                     FROM kg_claim_observations observation
                     JOIN kg_claim_extraction_outcomes outcome
                       ON outcome.chunk_id=observation.chunk_id
                      AND outcome.prompt_version=observation.prompt_version
                      AND outcome.prompt_generation=observation.prompt_generation
                     WHERE observation.evidence_id=ev.id
                       AND observation.edge_id=ev.edge_id
                       AND observation.source_session_id=ev.source_session_id
                       AND observation.source_message_id=ev.source_message_id
                       AND observation.evidence_kind=ev.evidence_kind
                       AND observation.polarity=ev.polarity
                       AND observation.interpretation_key=ev.interpretation_key
                       AND hymem_normalize_iso_timestamp(
                             observation.observed_at) IS NOT NULL
                       AND hymem_normalize_iso_timestamp(
                             outcome.succeeded_at) IS NOT NULL
                       AND hymem_timestamp_at_or_before(
                             ev.extracted_at,
                             observation.observed_at
                           ) = 1
                       AND hymem_timestamp_gap_within(
                             observation.observed_at,
                             outcome.succeeded_at,
                             {EVENT_CLOCK_SKEW_SECONDS}
                           ) = 1
                   ) AS current_publication_at
            FROM kg_evidence ev
            WHERE edge_id = ? OR id IN (
                SELECT dependency.evidence_id
                FROM kg_lifecycle_dependencies dependency
                JOIN kg_edge_lifecycle lifecycle
                  ON lifecycle.id = dependency.lifecycle_id
                WHERE lifecycle.edge_id = ?
            )
            """,
            (edge_id, edge_id),
        ).fetchall()
    }
    dependencies: dict[int, list[int]] = {}
    for row in conn.execute(
        """
        SELECT dependency.lifecycle_id, dependency.evidence_id
        FROM kg_lifecycle_dependencies dependency
        JOIN kg_edge_lifecycle lifecycle ON lifecycle.id = dependency.lifecycle_id
        WHERE lifecycle.edge_id = ?
        ORDER BY dependency.lifecycle_id, dependency.evidence_id
        """,
        (edge_id,),
    ).fetchall():
        dependencies.setdefault(int(row["lifecycle_id"]), []).append(
            int(row["evidence_id"])
        )

    scoped = any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    )

    def evidence_in_scope(evidence: sqlite3.Row | None) -> bool:
        if evidence is None:
            return False
        if scoped and evidence["provenance_status"] != "canonical":
            return False
        return bool(
            (source_session_id is None
             or evidence["source_session_id"] == source_session_id)
            and (source_peer_id is None
                 or evidence["source_peer_id"] == source_peer_id)
            and (source_workspace_id is None
                 or evidence["source_workspace_id"] == source_workspace_id)
        )

    def eligible_evidence(evidence_id: int) -> bool:
        evidence = evidence_rows.get(evidence_id)
        return evidence_in_scope(evidence) and _evidence_authoritative_at(
            evidence, cutoff, present_cutoff=present_cutoff
        )

    canonical_authority_exists = any(
        evidence["provenance_status"] == "canonical"
        and evidence_in_scope(evidence)
        and (
            (cutoff is None and int(evidence["is_current"]) == 1)
            or (cutoff is not None and eligible_evidence(evidence_id))
        )
        for evidence_id, evidence in evidence_rows.items()
        if int(evidence["edge_id"]) == edge_id
    )

    out: list[_LifecycleEvent] = []
    for row in rows:
        source_id = (
            int(row["source_evidence_id"])
            if row["source_evidence_id"] is not None
            else None
        )
        dependency_ids = dependencies.get(int(row["id"]), [])
        if scoped:
            # Scope before validating clocks. Corrupt history owned solely by
            # another provenance partition cannot suppress an otherwise valid
            # local fact. Source-less manual retractions are deliberate global
            # operator authority; legacy/unattributed state proves nothing in
            # a scoped view.
            if row["event_kind"] == "legacy_state":
                continue
            if source_id is not None and not evidence_in_scope(
                evidence_rows.get(source_id)
            ):
                continue
            if dependency_ids and not all(
                evidence_in_scope(evidence_rows.get(item))
                for item in dependency_ids
            ):
                continue
            if (
                source_id is None
                and not dependency_ids
                and row["event_kind"] != "manual_retraction"
            ):
                continue
        event_at = _parse_stored_timestamp(row["event_at"])
        if event_at is None:
            # Lifecycle rows can predate today's guard triggers or be corrupted
            # by direct SQL. A malformed valid-time coordinate proves no
            # interval and must fail closed on both current and historical reads.
            return []
        if not timestamp_at_or_before(event_at, present_cutoff):
            # Scheduled future authority is unsupported. This also protects
            # historical reads from externally injected future lifecycle rows.
            return []
        event_recorded_at = _parse_stored_timestamp(row["created_at"])
        if event_recorded_at is None or not timestamp_at_or_before(
            event_recorded_at, present_cutoff
        ):
            return []
        if cutoff is not None and event_recorded_at > cutoff:
            continue
        if source_id is not None:
            source_evidence = evidence_rows.get(source_id)
            source_extracted = (
                _parse_stored_timestamp(source_evidence["extracted_at"])
                if source_evidence is not None else None
            )
            source_published = (
                _parse_stored_timestamp(source_evidence["published_at"])
                if source_evidence is not None
                and source_evidence["provenance_status"] == "canonical"
                else None
            )
            if not eligible_evidence(source_id):
                if (
                    cutoff is None
                    and source_evidence is not None
                    and int(source_evidence["is_current"]) == 1
                ):
                    return []
                continue
            if (
                source_extracted is None
                or source_published is None
                or source_extracted > event_recorded_at
                or event_recorded_at > source_published
            ):
                return []
        if int(row["dependency_count"]) != len(dependency_ids):
            return []
        if dependency_ids:
            if not all(eligible_evidence(item) for item in dependency_ids):
                if cutoff is None and any(
                    item in evidence_rows
                    and int(evidence_rows[item]["is_current"]) == 1
                    for item in dependency_ids
                ):
                    return []
                continue
            dependency_clocks = []
            for item in dependency_ids:
                evidence = evidence_rows.get(item)
                if evidence is None:
                    return []
                field = (
                    "published_at"
                    if evidence["provenance_status"] == "canonical"
                    else "extracted_at"
                )
                dependency_clocks.append(
                    _parse_stored_timestamp(evidence[field])
                )
            if any(
                clock is None or clock > event_recorded_at
                for clock in dependency_clocks
            ):
                return []
        if row["event_kind"] == "manual_retraction":
            prefix = "manual-retraction:"
            signal_key = (
                row["event_key"][len(prefix):]
                if row["event_key"].startswith(prefix) else None
            )
            signal = conn.execute(
                "SELECT created_at FROM kg_evidence_signals "
                "WHERE edge_id=? AND signal_kind='manual_retraction' "
                "AND signal_key=? AND polarity=-1 "
                "AND counts_toward_confidence=1",
                (edge_id, signal_key),
            ).fetchone()
            signal_at = _parse_stored_timestamp(
                signal["created_at"] if signal is not None else None
            )
            if signal_at is None or signal_at > event_recorded_at:
                return []
        if row["event_kind"] == "legacy_state" and canonical_authority_exists:
            continue

        if source_id is not None:
            causal = _evidence_order(evidence_rows.get(source_id))
        elif row["event_kind"] in {"phase3_retraction", "value_supersession"}:
            causal = max(
                (_evidence_order(evidence_rows.get(item)) for item in dependency_ids),
                default=(-2, "missing"),
            )
        elif row["event_kind"] == "manual_retraction":
            causal = (2, row["event_key"])
        else:
            causal = (-1, row["event_key"])
        out.append(
            _LifecycleEvent(
                event_at=event_at,
                event_key=row["event_key"],
                event_kind=row["event_kind"],
                direction=int(row["direction"]),
                source_evidence_id=source_id,
                causal_order=causal,
            )
        )
    out.sort(
        key=lambda event: (
            event.event_at,
            event.causal_order,
            0 if event.event_kind == "claim_assertion" else 1,
            event.event_key,
        )
    )
    return out


def _evidence_authoritative_at(
    evidence: sqlite3.Row,
    cutoff: str | None,
    *,
    present_cutoff: str,
) -> bool:
    published = (
        _parse_stored_timestamp(evidence["published_at"])
        if evidence["provenance_status"] == "canonical"
        else None
    )
    current_publication = (
        _parse_stored_timestamp(evidence["current_publication_at"])
        if evidence["provenance_status"] == "canonical"
        else None
    )
    extracted = _parse_stored_timestamp(evidence["extracted_at"])
    superseded = (
        _parse_stored_timestamp(evidence["superseded_at"])
        if evidence["superseded_at"] is not None
        else None
    )
    if extracted is None:
        return False
    canonical = evidence["provenance_status"] == "canonical"
    if canonical:
        if (
            published is None
            or event_clock_is_valid(
                evidence["source_event_at"], evidence["extracted_at"]
            ) != 1
            or extracted > published
        ):
            return False
    if evidence["superseded_at"] is not None and superseded is None:
        return False
    if canonical and superseded is not None and published > superseded:
        return False
    if cutoff is None:
        if int(evidence["is_current"]) != 1 or not timestamp_at_or_before(
            extracted, present_cutoff
        ):
            return False
        if not canonical:
            return True
        return bool(
            current_publication is not None
            and published <= current_publication
            and timestamp_at_or_before(published, present_cutoff) == 1
            and timestamp_at_or_before(current_publication, present_cutoff) == 1
        )
    return (
        extracted <= cutoff
        and (published is None or published <= cutoff)
        and (superseded is None or cutoff < superseded)
    )


def _interval_at(
    events: list[_LifecycleEvent], target: str
) -> tuple[str, str | None, list[int]] | None:
    state_open = False
    opened_at: str | None = None
    assertion_ids: list[int] = []
    split = 0
    for split, event in enumerate(events):
        if event.event_at > target:
            break
        if event.direction == 1:
            if not state_open:
                opened_at = event.event_at
                assertion_ids = []
            state_open = True
            if event.source_evidence_id is not None:
                assertion_ids.append(event.source_evidence_id)
        else:
            state_open = False
            opened_at = None
            assertion_ids = []
    else:
        split = len(events)

    if not state_open or opened_at is None:
        return None

    invalid_at: str | None = None
    # If the loop broke, ``split`` points at the first event after target.
    for event in events[split:]:
        if event.direction == -1:
            invalid_at = event.event_at
            break
        if event.source_evidence_id is not None:
            # Future assertions reinforce the interval but are not citations for
            # knowledge at the requested valid-time coordinate.
            continue
    return opened_at, invalid_at, list(dict.fromkeys(assertion_ids))


def _evidence_order(evidence: sqlite3.Row | None) -> tuple[object, ...]:
    if evidence is None:
        return (-2, "missing")
    if evidence["provenance_status"] == "canonical":
        return (
            1,
            evidence["source_session_id"],
            int(evidence["source_message_id"]),
            evidence["evidence_kind"],
            int(evidence["revision"]),
            evidence["interpretation_key"],
        )
    return (
        0,
        evidence["chunk_id"],
        evidence["evidence_kind"],
        int(evidence["revision"]),
        evidence["interpretation_key"],
    )


def _citations_by_id(
    conn: sqlite3.Connection,
    evidence_ids: list[int],
    *,
    authoritative_at_recorded_time: bool,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
    limit: int = GRAPH_CITATION_LIMIT,
) -> list[GraphEvidenceCitation]:
    if not evidence_ids:
        return []
    placeholders = ",".join("?" for _ in evidence_ids)
    scope_clauses: list[str] = []
    scope_params: list[object] = []
    if source_session_id is not None:
        scope_clauses.append("source_session_id = ?")
        scope_params.append(source_session_id)
    if source_workspace_id is not None:
        scope_clauses.append("source_workspace_id = ?")
        scope_params.append(source_workspace_id)
    if source_peer_id is not None:
        scope_clauses.append("source_peer_id = ?")
        scope_params.append(source_peer_id)
    scope_sql = (
        " AND " + " AND ".join(scope_clauses) if scope_clauses else ""
    )
    rows = conn.execute(
        f"""
        SELECT id, evidence_kind, source_role, source_peer_id,
               source_workspace_id, source_session_id,
               source_message_id, source_event_at, source_created_at,
               temporal_scope, extracted_at, published_at,
               source_coverage_chunk_id,
               source_coverage_version, chunk_id, is_current,
               provenance_status, interpretation_key
        FROM kg_evidence
        WHERE id IN ({placeholders}) AND polarity = 1 {scope_sql}
        """,
        [*evidence_ids, *scope_params],
    ).fetchall()
    by_id = {
        int(row["id"]): _citation(
            row,
            authoritative_at_recorded_time=authoritative_at_recorded_time,
        )
        for row in rows
    }
    rows_by_id = {int(row["id"]): row for row in rows}
    ordered: list[GraphEvidenceCitation] = []
    seen_occurrences: set[tuple[object, ...]] = set()
    for item in reversed(evidence_ids):
        row = rows_by_id.get(item)
        citation = by_id.get(item)
        if row is None or citation is None:
            continue
        if row["provenance_status"] == "canonical":
            from hymem.dreaming.lossless import validate_message_coverage_artifact

            try:
                proof = validate_message_coverage_artifact(
                    conn,
                    message_id=int(row["source_message_id"]),
                    chunk_id=row["source_coverage_chunk_id"],
                    coverage_version=row["source_coverage_version"],
                )
            except (RuntimeError, TypeError, ValueError, sqlite3.Error):
                # Read paths independently fail closed even if an operator
                # dropped write guards and forged a provenance tuple.
                continue
            if (
                proof.session_id != row["source_session_id"]
                or proof.role != row["source_role"]
                or proof.source_created_at != row["source_created_at"]
                or proof.source_peer_id != row["source_peer_id"]
                or proof.source_workspace_id != row["source_workspace_id"]
            ):
                continue
        if row["provenance_status"] == "canonical":
            # Independent branches can carry the same immutable evidence
            # occurrence with different local revision/interval handles. Keep
            # both ledger rows (retirement is append-only), but cite the source
            # occurrence once. Reversed lifecycle order prefers the terminal
            # representative when both are eligible at a historical cutoff.
            occurrence = (
                "canonical", row["source_session_id"],
                int(row["source_message_id"]), row["evidence_kind"],
                row["source_workspace_id"], row["source_peer_id"],
                row["interpretation_key"], row["chunk_id"],
                _parse_stored_timestamp(row["extracted_at"]),
                _parse_stored_timestamp(row["published_at"]),
            )
        else:
            occurrence = ("legacy", int(row["id"]))
        if occurrence in seen_occurrences:
            continue
        seen_occurrences.add(occurrence)
        ordered.append(citation)
    return ordered[:limit]


def _citation(
    row: sqlite3.Row, *, authoritative_at_recorded_time: bool = True
) -> GraphEvidenceCitation:
    return GraphEvidenceCitation(
        evidence_id=int(row["id"]),
        evidence_kind=row["evidence_kind"],
        source_role=row["source_role"],
        source_peer_id=row["source_peer_id"],
        source_workspace_id=row["source_workspace_id"],
        source_session_id=row["source_session_id"],
        source_message_id=(
            int(row["source_message_id"])
            if row["source_message_id"] is not None
            else None
        ),
        source_event_at=row["source_event_at"],
        source_created_at=row["source_created_at"],
        temporal_scope=row["temporal_scope"],
        recorded_at=normalize_iso_timestamp(
            (
                row["published_at"]
                if row["provenance_status"] == "canonical"
                else row["extracted_at"]
            ),
            context="stored citation transaction",
        ),
        coverage_chunk_id=row["source_coverage_chunk_id"],
        coverage_version=row["source_coverage_version"],
        extraction_chunk_id=row["chunk_id"],
        currently_authoritative=bool(row["is_current"]),
        authoritative_at_recorded_time=authoritative_at_recorded_time,
        provenance_status=row["provenance_status"],
    )


def _parse_stored_timestamp(value: object) -> str | None:
    """Normalize persisted transaction time, failing closed when unknown.

    A malformed/missing transaction coordinate cannot prove that an event or
    evidence revision existed at a caller-supplied historical cutoff. Current
    authority reads do not consult this helper and retain legacy compatibility.
    """
    if value is None:
        return None
    try:
        return normalize_public_timestamp(str(value), field_name="stored timestamp")
    except ValueError:
        return None
