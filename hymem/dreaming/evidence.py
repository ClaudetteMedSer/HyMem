"""Auditable, idempotent evidence accounting for knowledge-graph edges.

``knowledge_graph.pos_evidence`` and ``neg_evidence`` are cached totals.  The
source of truth is the union of:

* ``kg_evidence``: message-chunk-backed extraction, reinforcement, and decay
  observations; and
* ``kg_evidence_signals``: events without a chunk, such as an explicit host
  retraction, plus quarantined legacy deltas retained for audit.

Derived edges are deliberately excluded.  Their ``1/0`` counters are computed
confidence placeholders rebuilt by ``inference.py``, not observed evidence.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import hashlib
import json
import functools
from dataclasses import dataclass
from typing import Iterable

from hymem.core.time import earliest_timestamp_spelling, normalize_iso_timestamp


log = logging.getLogger("hymem.dreaming.evidence")

_PROMPT_GENERATION_RE = re.compile(r"(?:^|[^0-9])v?(\d+)$", re.IGNORECASE)


def _atomic_evidence_helper(savepoint: str):
    """Make a public multi-statement ledger helper autocommit-safe."""
    def decorate(func):
        @functools.wraps(func)
        def wrapped(conn: sqlite3.Connection, *args, **kwargs):
            conn.execute(f"SAVEPOINT {savepoint}")
            try:
                result = func(conn, *args, **kwargs)
            except BaseException:
                conn.execute(f"ROLLBACK TO {savepoint}")
                conn.execute(f"RELEASE {savepoint}")
                raise
            conn.execute(f"RELEASE {savepoint}")
            return result
        return wrapped
    return decorate


def prompt_generation(prompt_version: str) -> int:
    """Return the explicit trailing generation, or conservative generation 0."""
    match = _PROMPT_GENERATION_RE.search(prompt_version.strip())
    return int(match.group(1)) if match else 0


def claim_result_hash(semantic_rows: Iterable[Iterable[object]]) -> str:
    """Hash a complete claim-result set using only portable semantics.

    Local edge/evidence rowids and transaction timestamps are deliberately
    excluded.  Entity/edge merges must refresh this projection because the
    edge natural tuple is part of what a chunk asserted.
    """
    rows = sorted(
        [list(row) for row in semantic_rows],
        key=lambda row: json.dumps(
            row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ),
    )
    encoded = json.dumps(
        rows, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def claim_observation_result_hash(
    conn: sqlite3.Connection, chunk_id: str
) -> str:
    """Return the portable semantic hash of one chunk's full observation set."""
    rows = conn.execute(
        """
        SELECT kg.subject_canonical, kg.predicate, kg.object_canonical,
               observation.source_session_id, observation.source_message_id,
               observation.evidence_kind, observation.polarity,
               observation.interpretation_key
        FROM kg_claim_observations observation
        JOIN knowledge_graph kg ON kg.id = observation.edge_id
        WHERE observation.chunk_id = ?
        """,
        (chunk_id,),
    ).fetchall()
    return claim_result_hash(tuple(row) for row in rows)


def claim_extraction_prompt_is_stale(
    conn: sqlite3.Connection, *, chunk_id: str, prompt_version: str
) -> bool:
    """Return whether a newer successful whole-chunk result already exists."""
    row = conn.execute(
        "SELECT prompt_generation FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()
    return bool(
        row is not None
        and int(row["prompt_generation"]) > prompt_generation(prompt_version)
    )


def _publish_chunk_evidence(conn: sqlite3.Connection, chunk_id: str) -> int:
    """Write each observed revision's immutable first publication clock."""
    outcome = conn.execute(
        "SELECT prompt_version,prompt_generation,succeeded_at "
        "FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()
    if outcome is None:
        return 0
    from hymem.core.db import evidence_mutation

    pending = conn.execute(
        """
        SELECT 1
        FROM kg_claim_observations observation
        JOIN kg_evidence ev ON ev.id=observation.evidence_id
        WHERE observation.chunk_id=?
          AND observation.prompt_version=?
          AND observation.prompt_generation=?
          AND ev.provenance_status='canonical'
          AND ev.published_at IS NULL
        LIMIT 1
        """,
        (chunk_id, outcome["prompt_version"], outcome["prompt_generation"]),
    ).fetchone()
    if pending is not None:
        # A staged revision may be repaired after the outcome row was first
        # written (for example after lifecycle reconciliation). Re-publish the
        # mutable whole-chunk authority at one fresh coordinate; immutable
        # clocks already present on older evidence remain untouched.
        published_at = normalize_iso_timestamp(
            conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0],
            context="claim publication",
        )
        with evidence_mutation(conn):
            conn.execute(
                "UPDATE kg_claim_observations SET observed_at=? "
                "WHERE chunk_id=? AND prompt_version=? "
                "AND prompt_generation=?",
                (
                    published_at, chunk_id, outcome["prompt_version"],
                    int(outcome["prompt_generation"]),
                ),
            )
            conn.execute(
                "UPDATE kg_claim_extraction_outcomes SET succeeded_at=? "
                "WHERE chunk_id=?",
                (published_at, chunk_id),
            )
    else:
        published_at = normalize_iso_timestamp(
            outcome["succeeded_at"], context="claim publication"
        )

    with evidence_mutation(conn):
        cur = conn.execute(
            """
            UPDATE kg_evidence
            SET published_at = ?
            WHERE published_at IS NULL
              AND provenance_status = 'canonical'
              AND id IN (
                  SELECT observation.evidence_id
                  FROM kg_claim_observations observation
                  WHERE observation.chunk_id = ?
                    AND observation.prompt_version = ?
                    AND observation.prompt_generation = ?
              )
            """,
            (
                published_at, chunk_id, outcome["prompt_version"],
                int(outcome["prompt_generation"]),
            ),
        )
    return int(cur.rowcount or 0)


@_atomic_evidence_helper("hymem_record_claim_extraction_outcome")
def record_claim_extraction_outcome(
    conn: sqlite3.Connection, *, chunk_id: str, prompt_version: str
) -> bool:
    """Publish the latest successful, source-validated result for a chunk.

    A higher prompt generation replaces the prior complete result, including
    with an empty set. A lower generation is stale. Equal generations must
    describe the exact same portable result set; prompt spelling and timestamp
    then converge under a stable total order.
    """
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("claim extraction prompt version must be nonempty")
    generation = prompt_generation(prompt_version)
    manifest = conn.execute(
        "SELECT source_manifest_version,source_manifest_count," 
        "(SELECT COUNT(*) FROM chunk_message_sources member "
        " WHERE member.chunk_id=chunks.id) AS member_count "
        "FROM chunks WHERE id=?",
        (chunk_id,),
    ).fetchone()
    if (
        manifest is None
        or manifest["source_manifest_version"] != "claim-source-manifest-v1"
        or not int(manifest["source_manifest_count"] or 0)
        or int(manifest["source_manifest_count"]) != int(manifest["member_count"])
    ):
        raise ValueError("claim extraction outcome requires a published source manifest")
    invalid_observation = conn.execute(
        """
        SELECT 1
        FROM kg_claim_observations observation
        JOIN kg_evidence ev ON ev.id=observation.evidence_id
        WHERE observation.chunk_id=? AND (
            observation.prompt_version<>?
            OR observation.prompt_generation<>?
            OR ev.provenance_status<>'canonical'
            OR ev.edge_id<>observation.edge_id
            OR ev.source_session_id<>observation.source_session_id
            OR ev.source_message_id<>observation.source_message_id
            OR ev.evidence_kind<>observation.evidence_kind
            OR ev.polarity<>observation.polarity
            OR ev.interpretation_key<>observation.interpretation_key
            OR NOT EXISTS (
                SELECT 1 FROM chunk_message_sources member
                WHERE member.chunk_id=observation.chunk_id
                  AND member.source_session_id=observation.source_session_id
                  AND member.source_message_id=observation.source_message_id
                  AND member.source_coverage_chunk_id=ev.source_coverage_chunk_id
                  AND member.source_coverage_version=ev.source_coverage_version
            )
        ) LIMIT 1
        """,
        (chunk_id, prompt_version, generation),
    ).fetchone()
    if invalid_observation is not None:
        raise ValueError(
            "claim extraction outcome disagrees with its observation authority"
        )
    result_hash = claim_observation_result_hash(conn, chunk_id)
    existing = conn.execute(
        "SELECT prompt_version,prompt_generation,result_hash,succeeded_at "
        "FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()
    if existing is not None:
        old_generation = int(existing["prompt_generation"])
        if generation < old_generation:
            return False
        if generation == old_generation and existing["result_hash"] != result_hash:
            raise ValueError(
                "same prompt generation claim extraction outcomes disagree"
            )
        if generation == old_generation:
            winner_version = max(str(existing["prompt_version"]), prompt_version)
            if winner_version == existing["prompt_version"]:
                return bool(_publish_chunk_evidence(conn, chunk_id))
            from hymem.core.db import evidence_mutation

            with evidence_mutation(conn):
                conn.execute(
                    "UPDATE kg_claim_extraction_outcomes SET prompt_version=? "
                    "WHERE chunk_id=?",
                    (winner_version, chunk_id),
                )
                conn.execute(
                    "UPDATE kg_claim_observations SET prompt_version=? "
                    "WHERE chunk_id=? AND prompt_generation=?",
                    (winner_version, chunk_id, generation),
                )
            _publish_chunk_evidence(conn, chunk_id)
            return True
    from hymem.core.db import evidence_history_mutation

    with evidence_history_mutation(conn):
        if existing is None:
            conn.execute(
                "INSERT INTO kg_claim_extraction_outcomes(" 
                "chunk_id,prompt_version,prompt_generation,result_hash) "
                "VALUES (?,?,?,?)",
                (chunk_id, prompt_version, generation, result_hash),
            )
        else:
            conn.execute(
                "UPDATE kg_claim_extraction_outcomes SET prompt_version=?,"
                "prompt_generation=?,result_hash=?,succeeded_at=CURRENT_TIMESTAMP "
                "WHERE chunk_id=?",
                (prompt_version, generation, result_hash, chunk_id),
            )
    _publish_chunk_evidence(conn, chunk_id)
    return True


def refresh_claim_extraction_outcomes(
    conn: sqlite3.Connection, chunk_ids: Iterable[str]
) -> None:
    """Refresh hashes after an authorized edge-natural/provenance merge."""
    ids = sorted({str(chunk_id) for chunk_id in chunk_ids if chunk_id})
    if not ids:
        return
    from hymem.core.db import evidence_mutation

    with evidence_mutation(conn):
        for chunk_id in ids:
            conn.execute(
                "UPDATE kg_claim_extraction_outcomes SET result_hash=? "
                "WHERE chunk_id=?",
                (claim_observation_result_hash(conn, chunk_id), chunk_id),
            )


def claim_retirement_authority(
    conn: sqlite3.Connection,
    *,
    source_session_id: str,
    source_message_id: int,
) -> tuple[str, str] | None:
    """Return the deterministic newest chunk outcome that omitted a source."""
    rows = conn.execute(
        """
        SELECT outcome.prompt_version,outcome.prompt_generation,
               outcome.succeeded_at,outcome.chunk_id,
               hymem_normalize_iso_timestamp(outcome.succeeded_at)
                   AS normalized_succeeded_at
        FROM kg_claim_extraction_outcomes outcome
        JOIN chunk_message_sources member ON member.chunk_id=outcome.chunk_id
        WHERE member.source_session_id=? AND member.source_message_id=?
          AND hymem_normalize_iso_timestamp(outcome.succeeded_at) IS NOT NULL
        """,
        (source_session_id, int(source_message_id)),
    ).fetchall()
    if not rows:
        return None

    def rank(row) -> tuple:
        return (
            int(row["prompt_generation"]), str(row["prompt_version"]),
            row["normalized_succeeded_at"], str(row["succeeded_at"]),
            str(row["chunk_id"]),
        )

    winner = max(rows, key=rank)
    return (
        str(winner["normalized_succeeded_at"]),
        f"successful_reextract:{winner['prompt_version']}",
    )


def evidence_natural_identity(
    conn: sqlite3.Connection, evidence_id: int
) -> tuple[object, ...]:
    """Return an ID-independent identity for a durable evidence revision."""
    item = conn.execute(
        """
        SELECT ev.provenance_status, ev.source_session_id,
               ev.source_message_id, ev.chunk_id, ev.evidence_kind,
               ev.revision, ev.interpretation_key,
               kg.subject_canonical, kg.predicate, kg.object_canonical
        FROM kg_evidence ev
        JOIN knowledge_graph kg ON kg.id = ev.edge_id
        WHERE ev.id = ?
        """,
        (int(evidence_id),),
    ).fetchone()
    if item is None:
        raise ValueError("evidence revision is missing")
    edge = (
        item["subject_canonical"], item["predicate"], item["object_canonical"]
    )
    if item["provenance_status"] == "canonical":
        return (
            *edge, "canonical", item["source_session_id"],
            int(item["source_message_id"]), item["evidence_kind"],
            int(item["revision"]), item["interpretation_key"],
        )
    return (
        *edge, "legacy", item["chunk_id"], item["evidence_kind"],
        int(item["revision"]), item["interpretation_key"],
    )


def evidence_natural_key(conn: sqlite3.Connection, evidence_id: int) -> str:
    """Hash :func:`evidence_natural_identity` for portable event keys."""
    encoded = json.dumps(
        evidence_natural_identity(conn, evidence_id),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def claim_assertion_event_key(
    source_session_id: str,
    source_message_id: int,
    evidence_kind: str,
    revision: int,
) -> str:
    """Return the portable identity of one sourced positive transition."""
    return (
        f"claim-assertion:{source_session_id}:{int(source_message_id)}:"
        f"{evidence_kind}:r{int(revision)}"
    )


def evidence_cause_key(conn: sqlite3.Connection, evidence_id: int) -> str:
    """Return a compact ID-independent dependency identity."""
    row = conn.execute(
        "SELECT provenance_status,source_session_id,source_message_id,"
        "chunk_id,evidence_kind,revision FROM kg_evidence WHERE id=?",
        (int(evidence_id),),
    ).fetchone()
    if row is None:
        raise ValueError("lifecycle cause evidence is missing")
    if row["provenance_status"] == "canonical":
        return (
            f"canonical:{row['source_session_id']}:{int(row['source_message_id'])}:"
            f"{row['evidence_kind']}:r{int(row['revision'])}"
        )
    return (
        f"legacy:{row['chunk_id']}:{row['evidence_kind']}:"
        f"r{int(row['revision'])}"
    )


def phase3_retraction_event_key(
    conn: sqlite3.Connection, evidence_ids: Iterable[int]
) -> str:
    """Bind a phase-3 decision to its exact portable cause set."""
    cause_keys = sorted({
        evidence_cause_key(conn, int(evidence_id)) for evidence_id in evidence_ids
    })
    encoded = json.dumps(cause_keys, separators=(",", ":")).encode("utf-8")
    return "phase3-retraction:" + hashlib.sha256(encoded).hexdigest()


def value_supersession_event_key(
    conn: sqlite3.Connection,
    *,
    loser_edge_id: int,
    winner_evidence_id: int,
    event_at: str,
) -> str:
    """Bind a typed-value close to natural loser/winner identities."""
    loser = conn.execute(
        "SELECT subject_canonical,predicate FROM knowledge_graph WHERE id=?",
        (int(loser_edge_id),),
    ).fetchone()
    winner = conn.execute(
        "SELECT kg.object_canonical FROM kg_evidence ev "
        "JOIN knowledge_graph kg ON kg.id=ev.edge_id WHERE ev.id=?",
        (int(winner_evidence_id),),
    ).fetchone()
    if loser is None or winner is None:
        raise ValueError("value-supersession identity is missing")
    return (
        f"value-supersession:{loser['subject_canonical']}:{loser['predicate']}:"
        f"{winner['object_canonical']}:{event_at}:"
        f"{evidence_natural_key(conn, int(winner_evidence_id))}"
    )


def manual_retraction_event_key(signal_key: str) -> str:
    """Bind a manual lifecycle transition to its confidence signal."""
    return f"manual-retraction:{signal_key}"


def recanonicalize_lifecycle_keys(conn: sqlite3.Connection) -> None:
    """Rebind derived lifecycle identities after a controlled graph rewrite.

    Canonical/entity merges can change an edge natural key or an evidence
    revision number. Decision keys must follow those portable identities; a
    stale pre-merge key would let the same causal event be appended twice.
    """
    rows = conn.execute(
        "SELECT * FROM kg_edge_lifecycle ORDER BY id"
    ).fetchall()
    desired: dict[int, str] = {}
    for row in rows:
        lifecycle_id = int(row["id"])
        kind = row["event_kind"]
        if kind == "claim_assertion":
            source = conn.execute(
                "SELECT source_session_id,source_message_id,evidence_kind,revision "
                "FROM kg_evidence WHERE id=?",
                (row["source_evidence_id"],),
            ).fetchone()
            if source is None:
                raise ValueError("claim lifecycle source disappeared during merge")
            desired[lifecycle_id] = claim_assertion_event_key(
                source["source_session_id"], source["source_message_id"],
                source["evidence_kind"], source["revision"],
            )
        elif kind in {"phase3_retraction", "value_supersession"}:
            dependencies = [
                int(item["evidence_id"])
                for item in conn.execute(
                    "SELECT evidence_id FROM kg_lifecycle_dependencies "
                    "WHERE lifecycle_id=? ORDER BY evidence_id", (lifecycle_id,),
                ).fetchall()
            ]
            if kind == "phase3_retraction":
                desired[lifecycle_id] = phase3_retraction_event_key(
                    conn, dependencies
                )
            elif len(dependencies) == 1:
                desired[lifecycle_id] = value_supersession_event_key(
                    conn,
                    loser_edge_id=int(row["edge_id"]),
                    winner_evidence_id=dependencies[0],
                    event_at=row["event_at"],
                )
            else:
                raise ValueError("value lifecycle lost its unique cause")
        elif kind == "manual_retraction":
            prefix = "manual-retraction:"
            if not str(row["event_key"]).startswith(prefix):
                raise ValueError("manual lifecycle key lost its signal binding")
            desired[lifecycle_id] = str(row["event_key"])
        elif kind == "legacy_state":
            desired[lifecycle_id] = str(row["event_key"])

    from hymem.core.db import evidence_history_mutation

    with evidence_history_mutation(conn):
        # Vacate every changing key first so deterministic swaps/collapses do
        # not depend on local row order.
        for row in rows:
            lifecycle_id = int(row["id"])
            if desired.get(lifecycle_id) != row["event_key"]:
                conn.execute(
                    "UPDATE kg_edge_lifecycle SET event_key=? WHERE id=?",
                    (f"__hymem_rekey__:{lifecycle_id}", lifecycle_id),
                )
        for row in rows:
            lifecycle_id = int(row["id"])
            wanted = desired.get(lifecycle_id, row["event_key"])
            current = conn.execute(
                "SELECT * FROM kg_edge_lifecycle WHERE id=?", (lifecycle_id,),
            ).fetchone()
            if current is None or current["event_key"] == wanted:
                continue
            collision = conn.execute(
                "SELECT * FROM kg_edge_lifecycle WHERE edge_id=? AND event_key=?",
                (current["edge_id"], wanted),
            ).fetchone()
            if collision is not None:
                dependencies = {
                    int(item["evidence_id"])
                    for item in conn.execute(
                        "SELECT evidence_id FROM kg_lifecycle_dependencies "
                        "WHERE lifecycle_id=?", (lifecycle_id,),
                    ).fetchall()
                }
                collision_dependencies = {
                    int(item["evidence_id"])
                    for item in conn.execute(
                        "SELECT evidence_id FROM kg_lifecycle_dependencies "
                        "WHERE lifecycle_id=?", (collision["id"],),
                    ).fetchall()
                }
                fields = (
                    "event_kind", "direction", "event_at", "source_evidence_id",
                    "dependency_count", "details",
                )
                if (
                    any(current[field] != collision[field] for field in fields)
                    or dependencies != collision_dependencies
                ):
                    raise ValueError("canonical lifecycle key collision during merge")
                earliest_created_at = earliest_timestamp_spelling(
                    collision["created_at"], current["created_at"]
                )
                conn.execute(
                    "UPDATE kg_edge_lifecycle SET created_at=? WHERE id=?",
                    (earliest_created_at, collision["id"]),
                )
                conn.execute(
                    "DELETE FROM kg_edge_lifecycle WHERE id=?", (lifecycle_id,)
                )
            else:
                conn.execute(
                    "UPDATE kg_edge_lifecycle SET event_key=? WHERE id=?",
                    (wanted, lifecycle_id),
                )


@dataclass(frozen=True)
class EvidenceMutation:
    """Result of recording one chunk-backed assertion."""

    inserted: bool
    polarity_changed: bool
    evidence_id: int

    @property
    def contribution_changed(self) -> bool:
        return self.inserted or self.polarity_changed


def _interpretation_key(
    *,
    polarity: int,
    evidence_weight: int,
    weight_source: str,
    source_role: str | None,
    surface_subject: str | None,
    surface_object: str | None,
    value_text: str | None,
    value_numeric: float | None,
    value_unit: str | None,
    temporal_scope: str | None,
) -> str:
    payload = {
        "evidence_weight": evidence_weight,
        "polarity": polarity,
        "source_role": source_role,
        # Surface spellings are retained on the immutable revision for audit,
        # but are not part of semantic prompt authority.  Canonical merges can
        # legitimately collapse ``redis_db`` and ``redis_database`` observations
        # from the same source/generation without creating a contradictory
        # interpretation of the claim.
        "temporal_scope": temporal_scope,
        "value_numeric": value_numeric,
        "value_text": value_text,
        "value_unit": value_unit,
        "weight_source": weight_source,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _edge_ids(conn: sqlite3.Connection, edge_ids: Iterable[int] | None) -> list[int]:
    if edge_ids is not None:
        return list(dict.fromkeys(int(edge_id) for edge_id in edge_ids))
    return [
        int(row["id"])
        for row in conn.execute(
            "SELECT id FROM knowledge_graph WHERE derived = 0 ORDER BY id"
        ).fetchall()
    ]


def ledger_counts(conn: sqlite3.Connection, edge_id: int) -> tuple[int, int]:
    """Return confidence-bearing positive/negative totals from provenance."""
    row = conn.execute(
        """
        SELECT
            COALESCE((
                SELECT SUM(evidence_weight) FROM kg_evidence
                WHERE edge_id = ? AND polarity = 1 AND is_current = 1
            ), 0) + COALESCE((
                SELECT SUM(evidence_weight) FROM kg_evidence_signals
                WHERE edge_id = ? AND polarity = 1
                  AND counts_toward_confidence = 1
            ), 0) AS pos,
            COALESCE((
                SELECT SUM(evidence_weight) FROM kg_evidence
                WHERE edge_id = ? AND polarity = -1 AND is_current = 1
            ), 0) + COALESCE((
                SELECT SUM(evidence_weight) FROM kg_evidence_signals
                WHERE edge_id = ? AND polarity = -1
                  AND counts_toward_confidence = 1
            ), 0) AS neg
        """,
        (edge_id, edge_id, edge_id, edge_id),
    ).fetchone()
    return int(row["pos"]), int(row["neg"])


def capture_unattributed_counts(
    conn: sqlite3.Connection,
    edge_ids: Iterable[int],
    *,
    reason: str,
) -> None:
    """Make direct/manual cached counts explicit before mutating an edge.

    HyMem's public paths now write provenance first, but callers and older tests
    may still seed ``knowledge_graph`` directly.  Rather than silently erasing
    those counts when a later operation reconciles the cache, this function
    converts any positive delta into a confidence-bearing, visibly
    ``runtime_unattributed`` signal.  A cache below its ledger is repaired up to
    the ledger; evidence rows are never discarded to accommodate a stale cache.
    """
    ids = _edge_ids(conn, edge_ids)
    from hymem.core.db import evidence_mutation

    with evidence_mutation(conn):
        for edge_id in ids:
            edge = conn.execute(
                "SELECT pos_evidence, neg_evidence, derived FROM knowledge_graph WHERE id = ?",
                (edge_id,),
            ).fetchone()
            if edge is None or edge["derived"]:
                continue
            expected_pos, expected_neg = ledger_counts(conn, edge_id)
            for polarity, actual, expected in (
                (1, int(edge["pos_evidence"]), expected_pos),
                (-1, int(edge["neg_evidence"]), expected_neg),
            ):
                delta = actual - expected
                if delta <= 0:
                    continue
                conn.execute(
                    """
                    INSERT INTO kg_evidence_signals(
                        edge_id, signal_key, signal_kind, polarity,
                        evidence_weight, counts_toward_confidence, details
                    ) VALUES (?, ?, 'runtime_unattributed', ?, ?, 1, ?)
                    ON CONFLICT(edge_id, signal_kind, signal_key) DO UPDATE SET
                        evidence_weight = evidence_weight + excluded.evidence_weight,
                        details = excluded.details
                    """,
                    (
                        edge_id,
                        f"runtime-unattributed:polarity:{polarity}",
                        polarity,
                        delta,
                        reason,
                    ),
                )
                log.warning(
                    "kg_evidence.runtime_unattributed edge_id=%d polarity=%d weight=%d reason=%s",
                    edge_id,
                    polarity,
                    delta,
                    reason,
                )
    reconcile_edge_counts(conn, ids)


def reconcile_edge_counts(
    conn: sqlite3.Connection, edge_ids: Iterable[int] | None = None
) -> int:
    """Rebuild cached totals from trusted provenance; derived edges are skipped."""
    changed = 0
    for edge_id in _edge_ids(conn, edge_ids):
        edge = conn.execute(
            "SELECT pos_evidence, neg_evidence, derived FROM knowledge_graph WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if edge is None or edge["derived"]:
            continue
        pos, neg = ledger_counts(conn, edge_id)
        if (int(edge["pos_evidence"]), int(edge["neg_evidence"])) == (pos, neg):
            continue
        conn.execute(
            "UPDATE knowledge_graph SET pos_evidence = ?, neg_evidence = ? WHERE id = ?",
            (pos, neg, edge_id),
        )
        changed += 1
    return changed


def count_mismatches(conn: sqlite3.Connection) -> list[dict[str, int]]:
    """Return non-derived edges whose cached totals disagree with provenance."""
    mismatches: list[dict[str, int]] = []
    for edge_id in _edge_ids(conn, None):
        row = conn.execute(
            "SELECT pos_evidence, neg_evidence FROM knowledge_graph WHERE id = ?",
            (edge_id,),
        ).fetchone()
        expected_pos, expected_neg = ledger_counts(conn, edge_id)
        if (int(row["pos_evidence"]), int(row["neg_evidence"])) != (
            expected_pos,
            expected_neg,
        ):
            mismatches.append(
                {
                    "edge_id": edge_id,
                    "pos_evidence": int(row["pos_evidence"]),
                    "neg_evidence": int(row["neg_evidence"]),
                    "expected_pos": expected_pos,
                    "expected_neg": expected_neg,
                }
            )
    return mismatches


@_atomic_evidence_helper("hymem_record_chunk_evidence")
def record_chunk_evidence(
    conn: sqlite3.Connection,
    *,
    edge_id: int,
    chunk_id: str,
    evidence_kind: str,
    polarity: int,
    evidence_weight: int,
    weight_source: str,
    prompt_version: str | None = None,
    source_role: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
    surface_subject: str | None = None,
    surface_object: str | None = None,
    value_text: str | None = None,
    value_numeric: float | None = None,
    value_unit: str | None = None,
    temporal_scope: str | None = None,
    source_message_id: int | None = None,
    source_session_id: str | None = None,
    source_created_at: str | None = None,
    source_event_at: str | None = None,
    source_coverage_chunk_id: str | None = None,
    source_coverage_version: str | None = None,
) -> EvidenceMutation:
    """Record one current assertion per exact claim source.

    Canonical v40 assertions key on ``(edge, source session, source message,
    evidence kind)``. Distinct messages in one extraction chunk are independent
    proofs; replaying the same message is not. Calls without source provenance
    remain an explicit legacy/internal compatibility path keyed by chunk.
    """
    if polarity not in (-1, 1):
        raise ValueError("polarity must be -1 or 1")
    if evidence_weight < 1:
        raise ValueError("evidence_weight must be >= 1")
    interpretation_key = _interpretation_key(
        polarity=polarity,
        evidence_weight=evidence_weight,
        weight_source=weight_source,
        source_role=source_role,
        surface_subject=surface_subject,
        surface_object=surface_object,
        value_text=value_text,
        value_numeric=value_numeric,
        value_unit=value_unit,
        temporal_scope=temporal_scope,
    )

    canonical = source_message_id is not None
    if canonical:
        if (
            isinstance(source_message_id, bool)
            or not isinstance(source_message_id, int)
            or source_message_id < 1
            or not isinstance(source_session_id, str)
            or not source_session_id.strip()
            or source_role not in {"user", "assistant", "system", "tool"}
            or (source_peer_id is None) != (source_workspace_id is None)
            or (
                source_peer_id is not None
                and (
                    not isinstance(source_peer_id, str)
                    or not source_peer_id.strip()
                    or not isinstance(source_workspace_id, str)
                    or not source_workspace_id.strip()
                )
            )
            or (
                source_created_at is not None
                and not isinstance(source_created_at, str)
            )
            or not isinstance(source_event_at, str)
            or not source_event_at
            or not isinstance(source_coverage_chunk_id, str)
            or not source_coverage_chunk_id
            or not isinstance(source_coverage_version, str)
            or not source_coverage_version
        ):
            raise ValueError("canonical evidence requires complete source provenance")
        from hymem.core.time import normalize_iso_timestamp

        normalized_source_event_at = normalize_iso_timestamp(
            source_event_at,
            context="canonical evidence source_event_at",
        )
        try:
            normalized_source_created_at = normalize_iso_timestamp(
                source_created_at,
                context="canonical evidence source_created_at",
            )
        except ValueError:
            # Only phase-1's documented legacy-source path may turn unknown
            # historical message time into the ancient canonical coordinate.
            if normalized_source_event_at != "0001-01-01T00:00:00.000Z":
                raise
            normalized_source_created_at = normalized_source_event_at
        if normalized_source_event_at != normalized_source_created_at:
            raise ValueError(
                "canonical evidence source event does not match source created_at"
            )
        # Raw source_created_at remains exact citation metadata. Every ordering,
        # lifecycle, and clock comparison uses this canonical coordinate.
        source_event_at = normalized_source_event_at
    elif any(
        value is not None
        for value in (
            source_session_id, source_created_at, source_event_at,
            source_coverage_chunk_id, source_coverage_version, source_role,
            source_peer_id, source_workspace_id,
        )
    ):
        raise ValueError("legacy evidence cannot carry partial source provenance")

    capture_unattributed_counts(
        conn, [edge_id], reason=f"before {evidence_kind} evidence write"
    )
    if canonical:
        existing = conn.execute(
            """
            SELECT id, polarity, evidence_weight, interpretation_key, extracted_at,
                   source_created_at, source_event_at, source_peer_id,
                   source_workspace_id
            FROM kg_evidence
            WHERE edge_id = ? AND source_session_id = ?
              AND source_message_id = ? AND evidence_kind = ?
              AND provenance_status = 'canonical' AND is_current = 1
            """,
            (edge_id, source_session_id, source_message_id, evidence_kind),
        ).fetchone()
        if existing is not None and (
            existing["source_created_at"] != source_created_at
            or existing["source_event_at"] != source_event_at
            or existing["source_peer_id"] != source_peer_id
            or existing["source_workspace_id"] != source_workspace_id
        ):
            raise ValueError("canonical evidence source provenance collides")
        max_revision = int(
            conn.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM kg_evidence "
                "WHERE edge_id = ? AND source_session_id = ? "
                "AND source_message_id = ? AND evidence_kind = ? "
                "AND provenance_status = 'canonical'",
                (edge_id, source_session_id, source_message_id, evidence_kind),
            ).fetchone()[0]
        )
    else:
        existing = conn.execute(
            """
            SELECT id, polarity, evidence_weight, interpretation_key
            FROM kg_evidence
            WHERE edge_id = ? AND chunk_id = ? AND evidence_kind = ?
              AND provenance_status = 'legacy_unattributed'
              AND is_current = 1
            """,
            (edge_id, chunk_id, evidence_kind),
        ).fetchone()
        max_revision = int(
            conn.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM kg_evidence "
                "WHERE edge_id = ? AND chunk_id = ? AND evidence_kind = ? "
                "AND provenance_status = 'legacy_unattributed'",
                (edge_id, chunk_id, evidence_kind),
            ).fetchone()[0]
        )

    semantically_changed = (
        existing is not None
        and existing["interpretation_key"] != interpretation_key
    )
    inserted = existing is None or semantically_changed
    polarity_changed = existing is not None and int(existing["polarity"]) != polarity
    observation_recorded_at: str | None = None
    if canonical:
        # Persist and validate against one exact transaction coordinate. Source
        # event time may lead it only by the documented small clock-skew bound;
        # otherwise a future-dated message would corrupt current graph state.
        observation_recorded_at = (
            conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
            if inserted
            else existing["extracted_at"]
        )
        from hymem.core.time import validate_event_clock

        validate_event_clock(
            conn,
            source_event_at,
            observation_recorded_at,
            context="canonical evidence",
        )
    elif inserted:
        observation_recorded_at = conn.execute(
            "SELECT CURRENT_TIMESTAMP"
        ).fetchone()[0]
    if semantically_changed:
        from hymem.core.db import evidence_mutation

        with evidence_mutation(conn):
            conn.execute(
                "UPDATE kg_evidence SET is_current = 0, "
                "superseded_at = CURRENT_TIMESTAMP, "
                "superseded_reason = 'source_reinterpreted' WHERE id = ?",
                (existing["id"],),
            )
    if inserted:
        from hymem.core.db import evidence_mutation

        with evidence_mutation(conn):
            conn.execute(
                """
            INSERT INTO kg_evidence(
                edge_id, chunk_id, evidence_kind, polarity, evidence_weight,
                weight_source, extraction_prompt_version, source_role,
                source_peer_id, source_workspace_id,
                surface_subject, surface_object, value_text, value_numeric,
                value_unit, temporal_scope, source_message_id,
                source_session_id, source_created_at, source_event_at,
                source_coverage_chunk_id, source_coverage_version,
                provenance_status, interpretation_key, revision, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                edge_id,
                chunk_id,
                evidence_kind,
                polarity,
                evidence_weight,
                weight_source,
                prompt_version,
                source_role,
                source_peer_id,
                source_workspace_id,
                surface_subject,
                surface_object,
                value_text,
                value_numeric,
                value_unit,
                temporal_scope,
                source_message_id,
                source_session_id,
                source_created_at,
                source_event_at,
                source_coverage_chunk_id,
                source_coverage_version,
                "canonical" if canonical else "legacy_unattributed",
                interpretation_key,
                max_revision + 1,
                observation_recorded_at,
                ),
            )

    reconcile_edge_counts(conn, [edge_id])
    if canonical:
        from hymem.dreaming.bitemporal import (
            recompute_edge_interval,
            record_lifecycle_event,
        )

        current = conn.execute(
            """
            SELECT id, revision FROM kg_evidence
            WHERE edge_id = ? AND source_session_id = ?
              AND source_message_id = ? AND evidence_kind = ?
              AND provenance_status = 'canonical' AND is_current = 1
            """,
            (edge_id, source_session_id, source_message_id, evidence_kind),
        ).fetchone()
        if current is not None and polarity == 1:
            record_lifecycle_event(
                conn,
                edge_id=edge_id,
                event_key=claim_assertion_event_key(
                    source_session_id, source_message_id,
                    evidence_kind, current["revision"],
                ),
                event_kind="claim_assertion",
                direction=1,
                event_at=source_event_at,
                source_evidence_id=int(current["id"]),
            )
        else:
            recompute_edge_interval(conn, edge_id)
    current_id = int(current["id"] if canonical else conn.execute(
        "SELECT id FROM kg_evidence WHERE edge_id = ? AND chunk_id = ? "
        "AND evidence_kind = ? AND provenance_status = 'legacy_unattributed' "
        "AND is_current = 1",
        (edge_id, chunk_id, evidence_kind),
    ).fetchone()["id"])
    return EvidenceMutation(
        inserted=inserted,
        polarity_changed=polarity_changed,
        evidence_id=current_id,
    )


def begin_chunk_extraction_reconciliation(
    conn: sqlite3.Connection, *, chunk_id: str, prompt_version: str
) -> set[int]:
    """Replace one chunk's observation set after a validated success.

    Canonical proofs can be shared by overlapping chunks, so deleting this
    chunk's authority does not retire them yet. Legacy chunk-owned extraction
    rows cannot be attributed to another authority and are retired immediately.
    The caller records the replacement observations, then calls
    :func:`finalize_chunk_extraction_reconciliation` in the same transaction.
    """
    observation_rows = conn.execute(
        "SELECT DISTINCT edge_id FROM kg_claim_observations WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchall()
    legacy_rows = conn.execute(
        "SELECT id, edge_id FROM kg_evidence WHERE chunk_id = ? "
        "AND evidence_kind = 'extraction' AND is_current = 1 "
        "AND provenance_status = 'legacy_unattributed'",
        (chunk_id,),
    ).fetchall()
    edge_ids = {
        int(row["edge_id"]) for row in [*observation_rows, *legacy_rows]
    }
    from hymem.core.db import evidence_mutation

    with evidence_mutation(conn):
        conn.execute(
            "DELETE FROM kg_claim_observations WHERE chunk_id = ?", (chunk_id,)
        )
        conn.execute(
            """
            UPDATE kg_evidence
            SET is_current = 0, superseded_at = CURRENT_TIMESTAMP,
                superseded_reason = ?
            WHERE chunk_id = ? AND evidence_kind = 'extraction' AND is_current = 1
              AND provenance_status = 'legacy_unattributed'
            """,
            (f"successful_reextract:{prompt_version}", chunk_id),
        )
    reconcile_edge_counts(conn, edge_ids)
    return edge_ids


def record_claim_observation(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    edge_id: int,
    source_session_id: str,
    source_message_id: int,
    polarity: int,
    prompt_version: str,
    evidence_id: int,
    evidence_kind: str = "extraction",
) -> None:
    """Attach one validated chunk authority to a globally deduped proof."""
    generation = prompt_generation(prompt_version)
    evidence_row = conn.execute(
        "SELECT interpretation_key FROM kg_evidence WHERE id = ?",
        (evidence_id,),
    ).fetchone()
    if evidence_row is None:
        raise ValueError("claim observation evidence is missing")
    interpretation_key = str(evidence_row["interpretation_key"])
    conflict = conn.execute(
        """
        SELECT 1 FROM kg_claim_observations
        WHERE edge_id = ? AND source_session_id = ? AND source_message_id = ?
          AND evidence_kind = ? AND prompt_generation = ?
          AND (polarity <> ? OR interpretation_key <> ?)
        LIMIT 1
        """,
        (
            edge_id, source_session_id, source_message_id, evidence_kind,
            generation, polarity, interpretation_key,
        ),
    ).fetchone()
    if conflict is not None:
        raise ValueError("same-generation claim observations disagree")
    from hymem.core.db import evidence_mutation

    with evidence_mutation(conn):
        conn.execute(
            """
        INSERT INTO kg_claim_observations(
            chunk_id, edge_id, source_session_id, source_message_id,
            evidence_kind, polarity, prompt_version, prompt_generation
            , evidence_id, interpretation_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(
            chunk_id, edge_id, source_session_id, source_message_id, evidence_kind
        ) DO UPDATE SET
            polarity = excluded.polarity,
            prompt_version = excluded.prompt_version,
            prompt_generation = excluded.prompt_generation,
            evidence_id = excluded.evidence_id,
            interpretation_key = excluded.interpretation_key,
            observed_at = CURRENT_TIMESTAMP
            """,
            (
                chunk_id, edge_id, source_session_id, source_message_id,
                evidence_kind, polarity, prompt_version, generation,
                evidence_id, interpretation_key,
            ),
        )


@_atomic_evidence_helper("hymem_finalize_chunk_extraction")
def finalize_chunk_extraction_reconciliation(
    conn: sqlite3.Connection, edge_ids: Iterable[int]
) -> None:
    """Retire orphaned proofs and converge state after chunk replacement."""
    from hymem.dreaming.bitemporal import recompute_edge_interval

    ids = _edge_ids(conn, edge_ids)
    reconciliation_at = normalize_iso_timestamp(
        conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0],
        context="claim reconciliation transaction",
    )
    if ids:
        placeholders = ",".join("?" for _ in ids)
        dependents = conn.execute(
            f"""
            SELECT DISTINCT lifecycle.edge_id
            FROM kg_edge_lifecycle lifecycle
            JOIN kg_lifecycle_dependencies dependency
              ON dependency.lifecycle_id = lifecycle.id
            JOIN kg_evidence cause ON cause.id = dependency.evidence_id
            WHERE cause.edge_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        ids = list(dict.fromkeys([*ids, *(int(row[0]) for row in dependents)]))
    for edge_id in ids:
        groups = conn.execute(
            """
            SELECT source_session_id, source_message_id, evidence_kind,
                   MAX(prompt_generation) AS winning_generation
            FROM kg_claim_observations
            WHERE edge_id = ?
            GROUP BY source_session_id, source_message_id, evidence_kind
            """,
            (edge_id,),
        ).fetchall()
        for group in groups:
            interpretations = conn.execute(
                """
                SELECT DISTINCT polarity, interpretation_key
                FROM kg_claim_observations
                WHERE edge_id = ? AND source_session_id = ?
                  AND source_message_id = ? AND evidence_kind = ?
                  AND prompt_generation = ?
                """,
                (
                    edge_id, group["source_session_id"],
                    group["source_message_id"], group["evidence_kind"],
                    group["winning_generation"],
                ),
            ).fetchall()
            if len(interpretations) != 1:
                raise ValueError(
                    "same-generation claim observations disagree semantically"
                )
            desired = int(interpretations[0]["polarity"])
            desired_key = str(interpretations[0]["interpretation_key"])
            selected = conn.execute(
                """
                SELECT evidence_id FROM kg_claim_observations
                WHERE edge_id = ? AND source_session_id = ?
                  AND source_message_id = ? AND evidence_kind = ?
                  AND prompt_generation = ?
                  AND polarity = ? AND interpretation_key = ?
                ORDER BY chunk_id, prompt_version, evidence_id LIMIT 1
                """,
                (
                    edge_id, group["source_session_id"],
                    group["source_message_id"], group["evidence_kind"],
                    group["winning_generation"], desired, desired_key,
                ),
            ).fetchone()
            current = conn.execute(
                """
                SELECT id, polarity, interpretation_key FROM kg_evidence
                WHERE edge_id = ? AND source_session_id = ?
                  AND source_message_id = ? AND evidence_kind = ?
                  AND provenance_status = 'canonical' AND is_current = 1
                """,
                (
                    edge_id, group["source_session_id"],
                    group["source_message_id"], group["evidence_kind"],
                ),
            ).fetchone()
            if (
                current is not None
                and int(current["polarity"]) == desired
                and current["interpretation_key"] == desired_key
            ):
                continue
            if current is not None:
                from hymem.core.db import evidence_mutation

                with evidence_mutation(conn):
                    conn.execute(
                        "UPDATE kg_evidence SET is_current = 0, "
                        "superseded_at = CURRENT_TIMESTAMP, "
                        "superseded_reason = 'lower_prompt_authority' WHERE id = ?",
                        (current["id"],),
                    )
            source_revision = conn.execute(
                "SELECT * FROM kg_evidence WHERE id = ?",
                (selected["evidence_id"],),
            ).fetchone()
            if source_revision is None:
                raise RuntimeError("claim observation has no matching evidence revision")
            next_revision = int(conn.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM kg_evidence "
                "WHERE edge_id = ? AND source_session_id = ? "
                "AND source_message_id = ? AND evidence_kind = ? "
                "AND provenance_status = 'canonical'",
                (
                    edge_id, group["source_session_id"],
                    group["source_message_id"], group["evidence_kind"],
                ),
            ).fetchone()[0]) + 1
            from hymem.core.db import evidence_mutation

            with evidence_mutation(conn):
                cur = conn.execute(
                    """
                INSERT INTO kg_evidence(
                    edge_id, chunk_id, polarity, surface_subject,
                    surface_object, value_text, value_numeric, value_unit,
                    temporal_scope, source_role, evidence_kind,
                    source_peer_id, source_workspace_id,
                    evidence_weight, weight_source, extraction_prompt_version,
                    extracted_at, source_message_id, source_session_id,
                    source_created_at, source_event_at,
                    source_coverage_chunk_id, source_coverage_version,
                    provenance_status, interpretation_key, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, 'canonical', ?, ?)
                    """,
                    (
                    edge_id, source_revision["chunk_id"], desired,
                    source_revision["surface_subject"],
                    source_revision["surface_object"],
                    source_revision["value_text"],
                    source_revision["value_numeric"],
                    source_revision["value_unit"],
                    source_revision["temporal_scope"],
                    source_revision["source_role"],
                    source_revision["evidence_kind"],
                    source_revision["source_peer_id"],
                    source_revision["source_workspace_id"],
                    source_revision["evidence_weight"],
                    source_revision["weight_source"],
                    source_revision["extraction_prompt_version"],
                    reconciliation_at,
                    source_revision["source_message_id"],
                    source_revision["source_session_id"],
                    source_revision["source_created_at"],
                    source_revision["source_event_at"],
                    source_revision["source_coverage_chunk_id"],
                    source_revision["source_coverage_version"],
                    desired_key, next_revision,
                    ),
                )
            replacement_id = int(cur.lastrowid)
            # The replacement is created after the chunk outcome that exposed
            # the global prompt winner. Re-publish every whole-chunk authority
            # that carries this winning interpretation at one exact clock, so
            # the replacement is never left staged and other observations in
            # those chunks cannot drift outside the bounded publication gap.
            # Immutable publication clocks on prior evidence are untouched.
            from hymem.core.db import evidence_history_mutation

            with evidence_history_mutation(conn):
                conn.execute(
                    """
                    UPDATE kg_claim_observations
                    SET evidence_id=?, observed_at=?
                    WHERE edge_id=? AND source_session_id=?
                      AND source_message_id=? AND evidence_kind=?
                      AND polarity=? AND interpretation_key=?
                    """,
                    (
                        replacement_id, reconciliation_at, edge_id,
                        group["source_session_id"],
                        group["source_message_id"], group["evidence_kind"],
                        desired, desired_key,
                    ),
                )
                authority_chunks = [
                    str(row["chunk_id"])
                    for row in conn.execute(
                        "SELECT DISTINCT chunk_id FROM kg_claim_observations "
                        "WHERE evidence_id=? ORDER BY chunk_id",
                        (replacement_id,),
                    ).fetchall()
                ]
                if not authority_chunks:
                    raise ValueError(
                        "reconciled claim revision has no chunk authority"
                    )
                placeholders = ",".join("?" for _ in authority_chunks)
                coherent_outcomes = int(conn.execute(
                    "SELECT COUNT(*) FROM kg_claim_extraction_outcomes outcome "
                    "WHERE outcome.chunk_id IN (" + placeholders + ") "
                    "AND EXISTS (SELECT 1 FROM kg_claim_observations observation "
                    "WHERE observation.chunk_id=outcome.chunk_id "
                    "AND observation.evidence_id=? "
                    "AND observation.prompt_version=outcome.prompt_version "
                    "AND observation.prompt_generation="
                    "outcome.prompt_generation)",
                    (*authority_chunks, replacement_id),
                ).fetchone()[0])
                if coherent_outcomes != len(authority_chunks):
                    raise ValueError(
                        "reconciled claim revision lacks a successful outcome"
                    )
                conn.execute(
                    "UPDATE kg_claim_observations SET observed_at=? "
                    "WHERE chunk_id IN (" + placeholders + ")",
                    (reconciliation_at, *authority_chunks),
                )
                conn.execute(
                    "UPDATE kg_claim_extraction_outcomes SET succeeded_at=? "
                    "WHERE chunk_id IN (" + placeholders + ")",
                    (reconciliation_at, *authority_chunks),
                )
            if desired == 1:
                from hymem.dreaming.bitemporal import record_lifecycle_event

                record_lifecycle_event(
                    conn,
                    edge_id=edge_id,
                    event_key=claim_assertion_event_key(
                        group["source_session_id"], group["source_message_id"],
                        group["evidence_kind"], next_revision,
                    ),
                    event_kind="claim_assertion",
                    direction=1,
                    event_at=source_revision["source_event_at"],
                    source_evidence_id=replacement_id,
                    recorded_at=reconciliation_at,
                )
            with evidence_mutation(conn):
                conn.execute(
                    "UPDATE kg_evidence SET published_at=? WHERE id=?",
                    (reconciliation_at, replacement_id),
                )
        from hymem.core.db import evidence_mutation

        orphaned = conn.execute(
            """
            SELECT id,source_session_id,source_message_id,source_event_at
            FROM kg_evidence
            WHERE edge_id = ? AND evidence_kind = 'extraction'
              AND provenance_status = 'canonical' AND is_current = 1
              AND NOT EXISTS (
                  SELECT 1 FROM kg_claim_observations observation
                  WHERE observation.edge_id = kg_evidence.edge_id
                    AND observation.source_session_id = kg_evidence.source_session_id
                    AND observation.source_message_id = kg_evidence.source_message_id
                    AND observation.evidence_kind = kg_evidence.evidence_kind
                    AND observation.polarity = kg_evidence.polarity
              )
            """,
            (edge_id,),
        ).fetchall()
        with evidence_mutation(conn):
            for orphan in orphaned:
                authority = claim_retirement_authority(
                    conn,
                    source_session_id=orphan["source_session_id"],
                    source_message_id=int(orphan["source_message_id"]),
                )
                superseded_at, reason = authority or (
                    orphan["source_event_at"],
                    "successful_reextract:no_current_authority",
                )
                conn.execute(
                    "UPDATE kg_evidence SET is_current=0,superseded_at=?,"
                    "superseded_reason=? WHERE id=?",
                    (superseded_at, reason, orphan["id"]),
                )
        reconcile_edge_counts(conn, [edge_id])
        recomputed = recompute_edge_interval(conn, edge_id)
        positive = conn.execute(
            """
            SELECT 1 FROM kg_evidence
            WHERE edge_id = ? AND is_current = 1 AND polarity = 1
            UNION ALL
            SELECT 1 FROM kg_evidence_signals
            WHERE edge_id = ? AND polarity = 1
              AND counts_toward_confidence = 1
            LIMIT 1
            """,
            (edge_id, edge_id),
        ).fetchone()
        if not recomputed and positive is None:
            terminal = conn.execute(
                """
                SELECT COALESCE(
                    MAX(CASE WHEN provenance_status = 'canonical'
                                  AND polarity = 1
                             THEN hymem_normalize_iso_timestamp(
                                  source_event_at) END),
                    (SELECT hymem_normalize_iso_timestamp(valid_at)
                     FROM knowledge_graph WHERE id = ?),
                    '0001-01-01T00:00:00.000Z'
                ) AS valid_at,
                COALESCE(
                    MAX(CASE WHEN provenance_status = 'canonical'
                             THEN hymem_normalize_iso_timestamp(
                                  source_event_at) END),
                    (SELECT hymem_normalize_iso_timestamp(valid_at)
                     FROM knowledge_graph WHERE id = ?),
                    '0001-01-01T00:00:00.000Z'
                ) AS event_at
                FROM kg_evidence WHERE edge_id = ?
                """,
                (edge_id, edge_id, edge_id),
            ).fetchone()
            conn.execute(
                "UPDATE knowledge_graph SET status = 'retracted', "
                "valid_at = ?, invalid_at = MAX(?, ?) "
                "WHERE id = ? AND derived = 0",
                (
                    terminal["valid_at"], terminal["valid_at"],
                    terminal["event_at"], edge_id,
                ),
            )


@_atomic_evidence_helper("hymem_record_signal")
def record_signal(
    conn: sqlite3.Connection,
    *,
    edge_id: int,
    signal_key: str,
    signal_kind: str,
    polarity: int,
    evidence_weight: int = 1,
    details: str | None = None,
) -> bool:
    """Record an idempotent non-chunk signal and refresh the cached totals."""
    if polarity not in (-1, 1):
        raise ValueError("polarity must be -1 or 1")
    if evidence_weight < 1:
        raise ValueError("evidence_weight must be >= 1")
    if signal_kind == "manual_retraction" and polarity != -1:
        raise ValueError("manual_retraction signals must have negative polarity")
    capture_unattributed_counts(conn, [edge_id], reason=f"before {signal_kind} signal")
    from hymem.core.db import evidence_mutation

    existing = conn.execute(
        "SELECT polarity,evidence_weight,counts_toward_confidence,details "
        "FROM kg_evidence_signals WHERE edge_id=? AND signal_kind=? "
        "AND signal_key=?",
        (edge_id, signal_kind, signal_key),
    ).fetchone()
    expected = (polarity, evidence_weight, 1, details)
    if existing is not None and tuple(existing) != expected:
        raise ValueError("evidence signal key collides with different state")
    inserted = existing is None
    if inserted:
        with evidence_mutation(conn):
            conn.execute(
            """
            INSERT INTO kg_evidence_signals(
                edge_id, signal_key, signal_kind, polarity, evidence_weight,
                counts_toward_confidence, details
            ) VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (edge_id, signal_key, signal_kind, polarity, evidence_weight, details),
            )
    reconcile_edge_counts(conn, [edge_id])
    if signal_kind == "manual_retraction":
        from hymem.dreaming.bitemporal import (
            manual_retraction_event_at,
            record_lifecycle_event,
        )

        signal = conn.execute(
            "SELECT id, created_at FROM kg_evidence_signals "
            "WHERE edge_id = ? AND signal_kind = ? AND signal_key = ?",
            (edge_id, signal_kind, signal_key),
        ).fetchone()
        # Use the same exact clamp validated by the lifecycle wire contract.
        # Manual events sort after sourced events at an equal instant, making
        # the public intent deterministic without an imperative status write.
        event_at = manual_retraction_event_at(
            conn, edge_id, signal["created_at"]
        )
        record_lifecycle_event(
            conn,
            edge_id=edge_id,
            event_key=manual_retraction_event_key(signal_key),
            event_kind="manual_retraction",
            direction=-1,
            event_at=event_at,
            details=details,
        )
    return inserted


@_atomic_evidence_helper("hymem_move_edge_provenance")
def move_edge_provenance(
    conn: sqlite3.Connection, survivor_id: int, member_ids: Iterable[int]
) -> None:
    """Move complete revision/authority/lifecycle history to a survivor.

    Current-source collisions are retired, never overwritten. Historical
    revision-number collisions are deterministically renumbered, while an
    exactly identical imported revision is retained once and all foreign keys
    are remapped before its duplicate is removed.
    """
    members = [edge_id for edge_id in _edge_ids(conn, member_ids) if edge_id != survivor_id]
    if not members:
        return
    all_ids = [survivor_id, *members]
    capture_unattributed_counts(conn, all_ids, reason="before edge merge")

    placeholders = ",".join("?" for _ in members)
    rows = conn.execute(
        f"SELECT * FROM kg_evidence WHERE edge_id IN ({placeholders}) "
        "ORDER BY provenance_status, source_session_id, source_message_id, "
        "chunk_id, evidence_kind, revision, interpretation_key, id",
        members,
    ).fetchall()
    id_map: dict[int, int] = {}
    duplicate_ids: list[int] = []
    dependent_edge_ids: set[int] = set()
    merge_at = normalize_iso_timestamp(
        conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0],
        context="canonical edge merge transaction",
    )
    prompt_authority: dict[
        tuple[str, int, str], tuple[tuple[int, str], set[int], str]
    ] = {}
    authority_rows = conn.execute(
        "SELECT observation.source_session_id,observation.source_message_id,"
        "observation.evidence_kind,observation.polarity,"
        "observation.interpretation_key,observation.prompt_generation,"
        "observation.evidence_id,observation.observed_at,"
        "outcome.succeeded_at "
        "FROM kg_claim_observations observation "
        "JOIN kg_claim_extraction_outcomes outcome "
        "ON outcome.chunk_id=observation.chunk_id "
        "AND outcome.prompt_version=observation.prompt_version "
        "AND outcome.prompt_generation=observation.prompt_generation "
        "WHERE observation.edge_id IN ("
        + ",".join("?" for _ in all_ids) + ") "
        "AND hymem_event_clock_is_valid(outcome.succeeded_at,?)=1",
        (*all_ids, merge_at),
    ).fetchall()
    grouped_authority: dict[tuple[str, int, str], list[sqlite3.Row]] = {}
    for observation in authority_rows:
        key = (
            str(observation["source_session_id"]),
            int(observation["source_message_id"]),
            str(observation["evidence_kind"]),
        )
        grouped_authority.setdefault(key, []).append(observation)
    for key, observations in grouped_authority.items():
        generation = max(int(item["prompt_generation"]) for item in observations)
        winners = [
            item for item in observations
            if int(item["prompt_generation"]) == generation
        ]
        semantics = {
            (int(item["polarity"]), str(item["interpretation_key"]))
            for item in winners
        }
        if len(semantics) != 1:
            raise ValueError(
                "canonical edge merge found conflicting same-generation "
                "prompt authority"
            )
        clocks = []
        for item in winners:
            try:
                clocks.append(normalize_iso_timestamp(
                    item["succeeded_at"], context="edge merge publication"
                ))
            except ValueError:
                continue
        prompt_authority[key] = (
            next(iter(semantics)),
            {int(item["evidence_id"]) for item in winners},
            max(clocks) if clocks else merge_at,
        )

    def identity_where(row: sqlite3.Row) -> tuple[str, tuple[object, ...]]:
        if row["provenance_status"] == "canonical":
            return (
                "source_session_id = ? AND source_message_id = ?",
                (row["source_session_id"], row["source_message_id"]),
            )
        return ("chunk_id = ?", (row["chunk_id"],))

    def first_occurrence_rank(row: sqlite3.Row) -> tuple[object, ...]:
        try:
            extracted = normalize_iso_timestamp(
                row["extracted_at"], context="edge merge extraction"
            )
            validity = 0
        except ValueError:
            extracted = "9999-12-31T23:59:59.999Z"
            validity = 1
        audit = json.dumps(
            [
                row["chunk_id"], row["surface_subject"],
                row["surface_object"], row["extraction_prompt_version"],
                str(row["extracted_at"] or ""),
            ],
            ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        )
        return validity, extracted, str(row["extracted_at"] or ""), audit

    from hymem.core.db import evidence_history_mutation

    with evidence_history_mutation(conn):
        # Resolve the global prompt winner before moving either side.  This is
        # the state that would have existed had both surface forms already
        # shared one canonical edge.  In particular, the successful outcome
        # (not its slightly earlier observation) is the transaction boundary
        # at which a losing interpretation may close.
        for (
            source_session_id, source_message_id, evidence_kind
        ), (desired, _winner_ids, authority_at) in prompt_authority.items():
            current_rows = conn.execute(
                "SELECT id,polarity,interpretation_key,published_at "
                "FROM kg_evidence WHERE edge_id IN ("
                + ",".join("?" for _ in all_ids) + ") "
                "AND source_session_id=? AND source_message_id=? "
                "AND evidence_kind=? AND provenance_status='canonical' "
                "AND is_current=1",
                (
                    *all_ids, source_session_id, source_message_id,
                    evidence_kind,
                ),
            ).fetchall()
            for current_row in current_rows:
                semantic = (
                    int(current_row["polarity"]),
                    str(current_row["interpretation_key"]),
                )
                if semantic == desired:
                    continue
                close_at = authority_at
                try:
                    close_at = max(
                        close_at,
                        normalize_iso_timestamp(
                            current_row["published_at"],
                            context="edge merge loser publication",
                        ),
                    )
                except ValueError:
                    pass
                conn.execute(
                    "UPDATE kg_evidence SET is_current=0,superseded_at=?,"
                    "superseded_reason='lower_prompt_authority' WHERE id=?",
                    (close_at, current_row["id"]),
                )

        for original_row in rows:
            # Prompt-authority reduction above may have retired this member.
            # Re-read it so collision handling never reasons from the stale
            # pre-reduction snapshot.
            row = conn.execute(
                "SELECT * FROM kg_evidence WHERE id=?", (original_row["id"],)
            ).fetchone()
            if row is None:
                continue
            identity_sql, identity_params = identity_where(row)
            exact = conn.execute(
                f"""
                SELECT * FROM kg_evidence
                WHERE edge_id = ? AND provenance_status = ?
                  AND evidence_kind = ? AND {identity_sql}
                  AND revision = ? AND interpretation_key = ?
                  AND polarity = ?
                ORDER BY id LIMIT 1
                """,
                (
                    survivor_id, row["provenance_status"], row["evidence_kind"],
                    *identity_params, row["revision"], row["interpretation_key"],
                    row["polarity"],
                ),
            ).fetchone()
            if exact is not None and bool(exact["is_current"]) != bool(
                row["is_current"]
            ):
                # A retired revision is append-only and cannot be revived.
                # Keep the open branch copy as a distinct, renumbered
                # revision instead of deleting it into the retired survivor
                # (or choosing a direction-dependent keeper).  Finalization
                # can then select the already-published current copy without
                # fabricating a merge-time replacement.
                exact = None
            if exact is not None:
                immutable = (
                    "polarity", "evidence_weight", "weight_source", "source_role",
                    "value_text", "value_numeric", "value_unit", "temporal_scope",
                    "source_message_id", "source_session_id", "source_created_at",
                    "source_event_at", "source_coverage_chunk_id",
                    "source_coverage_version", "provenance_status",
                    "interpretation_key", "evidence_kind",
                )
                if any(exact[field] != row[field] for field in immutable):
                    raise ValueError(
                        "canonical edge merge found conflicting evidence audit"
                    )
                published = []
                for candidate in (exact["published_at"], row["published_at"]):
                    if candidate is None:
                        continue
                    try:
                        published.append(normalize_iso_timestamp(
                            candidate, context="edge merge publication"
                        ))
                    except ValueError:
                        continue
                earliest_published = min(published) if published else None
                first = (
                    row
                    if first_occurrence_rank(row) < first_occurrence_rank(exact)
                    else exact
                )
                semantic = (
                    int(row["polarity"]), str(row["interpretation_key"])
                )
                authority = (
                    prompt_authority.get((
                        str(row["source_session_id"]),
                        int(row["source_message_id"]),
                        str(row["evidence_kind"]),
                    ))
                    if row["provenance_status"] == "canonical"
                    else None
                )
                # Exact branch copies are one logical revision.  Union their
                # authority interval: a surviving winning observation keeps
                # it current; otherwise the interval closes when the last
                # coherent branch copy closed, never at merge wall time.
                merged_current = bool(exact["is_current"])
                superseded_at = None
                superseded_reason = None
                if not merged_current:
                    retirement_candidates: list[tuple[str, str, str]] = []
                    for candidate in (exact, row):
                        raw = candidate["superseded_at"]
                        if raw is None:
                            continue
                        try:
                            retirement_candidates.append((
                                normalize_iso_timestamp(
                                    raw, context="edge merge retirement"
                                ),
                                str(raw),
                                str(candidate["superseded_reason"] or ""),
                            ))
                        except ValueError:
                            continue
                    if authority is not None and authority[0] != semantic:
                        retirement_candidates.append((
                            authority[2], authority[2],
                            "lower_prompt_authority",
                        ))
                    if retirement_candidates:
                        (
                            _normalized_retirement,
                            superseded_at,
                            superseded_reason,
                        ) = max(retirement_candidates)
                    else:
                        # Preserve a deterministic malformed legacy close as
                        # malformed (and therefore fail-closed); never launder
                        # it into an invented valid timestamp.
                        raw_candidates = sorted(
                            (
                                str(candidate["superseded_at"]),
                                str(candidate["superseded_reason"] or ""),
                            )
                            for candidate in (exact, row)
                            if candidate["superseded_at"] is not None
                        )
                        if raw_candidates:
                            superseded_at, superseded_reason = raw_candidates[-1]
                        else:
                            superseded_at = exact["superseded_at"]
                            superseded_reason = exact["superseded_reason"]
                conn.execute(
                    "UPDATE kg_evidence SET chunk_id=?,surface_subject=?,"
                    "surface_object=?,extraction_prompt_version=?,"
                    "extracted_at=?,published_at=?,is_current=?,"
                    "superseded_at=?,superseded_reason=? WHERE id=?",
                    (
                        first["chunk_id"], first["surface_subject"],
                        first["surface_object"],
                        first["extraction_prompt_version"],
                        first["extracted_at"], earliest_published,
                        1 if merged_current else 0,
                        superseded_at, superseded_reason, exact["id"],
                    ),
                )
                id_map[int(row["id"])] = int(exact["id"])
                duplicate_ids.append(int(row["id"]))
                continue

            revision = int(row["revision"])
            revision_collision = conn.execute(
                f"""
                SELECT 1 FROM kg_evidence
                WHERE edge_id = ? AND provenance_status = ?
                  AND evidence_kind = ? AND {identity_sql} AND revision = ?
                """,
                (
                    survivor_id, row["provenance_status"], row["evidence_kind"],
                    *identity_params, revision,
                ),
            ).fetchone()
            if revision_collision is not None:
                revision = int(conn.execute(
                    f"""
                    SELECT COALESCE(MAX(revision), 0) FROM kg_evidence
                    WHERE edge_id = ? AND provenance_status = ?
                      AND evidence_kind = ? AND {identity_sql}
                    """,
                    (
                        survivor_id, row["provenance_status"],
                        row["evidence_kind"], *identity_params,
                    ),
                ).fetchone()[0]) + 1

            if int(row["is_current"]):
                current_collision = conn.execute(
                    f"""
                    SELECT id,polarity,interpretation_key,published_at
                    FROM kg_evidence
                    WHERE edge_id = ? AND provenance_status = ?
                      AND evidence_kind = ? AND {identity_sql}
                      AND is_current = 1
                    """,
                    (
                        survivor_id, row["provenance_status"],
                        row["evidence_kind"], *identity_params,
                    ),
                ).fetchone()
                if current_collision is not None:
                    incoming_wins = False
                    authority_at = merge_at
                    if row["provenance_status"] == "canonical":
                        authority = prompt_authority.get((
                            str(row["source_session_id"]),
                            int(row["source_message_id"]),
                            str(row["evidence_kind"]),
                        ))
                        if authority is not None:
                            desired, winner_ids, authority_at = authority
                            incoming_semantic = (
                                int(row["polarity"]),
                                str(row["interpretation_key"]),
                            )
                            collision_semantic = (
                                int(current_collision["polarity"]),
                                str(current_collision["interpretation_key"]),
                            )
                            incoming_wins = (
                                incoming_semantic == desired
                                and (
                                    collision_semantic != desired
                                    or int(row["id"]) in winner_ids
                                    and int(current_collision["id"])
                                    not in winner_ids
                                )
                            )
                    loser = current_collision if incoming_wins else row
                    publication = loser["published_at"]
                    if publication is not None:
                        try:
                            authority_at = max(
                                authority_at,
                                normalize_iso_timestamp(
                                    publication,
                                    context="edge merge loser publication",
                                ),
                            )
                        except ValueError:
                            pass
                    conn.execute(
                        "UPDATE kg_evidence SET is_current=0,superseded_at=?,"
                        "superseded_reason='lower_prompt_authority' WHERE id=?",
                        (authority_at, loser["id"]),
                    )
            conn.execute(
                "UPDATE kg_evidence SET edge_id = ?, revision = ? WHERE id = ?",
                (survivor_id, revision, row["id"]),
            )
            id_map[int(row["id"])] = int(row["id"])

        # Move observations before deleting exact duplicate revisions.
        observations = conn.execute(
            f"SELECT rowid, * FROM kg_claim_observations "
            f"WHERE edge_id IN ({placeholders}) ORDER BY chunk_id, rowid",
            members,
        ).fetchall()
        observation_chunk_ids = {
            str(observation["chunk_id"]) for observation in observations
        }
        for observation in observations:
            mapped_evidence = id_map.get(
                int(observation["evidence_id"]), int(observation["evidence_id"])
            )
            collision = conn.execute(
                """
                SELECT rowid, polarity, prompt_version, prompt_generation,
                       evidence_id, interpretation_key
                FROM kg_claim_observations
                WHERE chunk_id = ? AND edge_id = ?
                  AND source_session_id = ? AND source_message_id = ?
                  AND evidence_kind = ?
                """,
                (
                    observation["chunk_id"], survivor_id,
                    observation["source_session_id"],
                    observation["source_message_id"],
                    observation["evidence_kind"],
                ),
            ).fetchone()
            if collision is not None:
                expected = (
                    int(observation["polarity"]), observation["prompt_version"],
                    int(observation["prompt_generation"]), mapped_evidence,
                    observation["interpretation_key"],
                )
                actual = (
                    int(collision["polarity"]), collision["prompt_version"],
                    int(collision["prompt_generation"]),
                    int(collision["evidence_id"]),
                    collision["interpretation_key"],
                )
                if actual != expected:
                    raise ValueError(
                        "canonical edge merge found conflicting claim authority"
                    )
                conn.execute(
                    "DELETE FROM kg_claim_observations WHERE rowid = ?",
                    (observation["rowid"],),
                )
            else:
                conn.execute(
                    "UPDATE kg_claim_observations "
                    "SET edge_id = ?, evidence_id = ? WHERE rowid = ?",
                    (survivor_id, mapped_evidence, observation["rowid"]),
                )

        # Revision numbers originate in independent edge histories. Reassign
        # them from portable chronology after observations have converged, so
        # choosing either alias as the survivor produces the same ordinals.
        # Retired/open copies of an otherwise identical interpretation remain
        # distinct, with the retired interval ordered first; this preserves
        # append-only retirement while avoiding a merge-time replacement.
        canonical_bases = conn.execute(
            "SELECT DISTINCT source_session_id,source_message_id,evidence_kind "
            "FROM kg_evidence WHERE edge_id=? "
            "AND provenance_status='canonical' ORDER BY source_session_id,"
            "source_message_id,evidence_kind",
            (survivor_id,),
        ).fetchall()
        for base in canonical_bases:
            base_params = (
                survivor_id, base["source_session_id"],
                base["source_message_id"], base["evidence_kind"],
            )
            revisions = conn.execute(
                "SELECT ev.*,MIN(observation.prompt_generation) "
                "AS observed_generation FROM kg_evidence ev "
                "LEFT JOIN kg_claim_observations observation "
                "ON observation.evidence_id=ev.id WHERE ev.edge_id=? "
                "AND ev.source_session_id=? AND ev.source_message_id=? "
                "AND ev.evidence_kind=? AND ev.provenance_status='canonical' "
                "GROUP BY ev.id",
                base_params,
            ).fetchall()

            def revision_clock(candidate: sqlite3.Row) -> tuple[int, str, str]:
                try:
                    return (
                        0,
                        normalize_iso_timestamp(
                            candidate["extracted_at"],
                            context="edge merge revision extraction",
                        ),
                        str(candidate["extracted_at"]),
                    )
                except ValueError:
                    return (1, "9999-12-31T23:59:59.999Z", str(
                        candidate["extracted_at"]
                    ))

            ordered_revisions = sorted(
                revisions,
                key=lambda candidate: (
                    int(candidate["observed_generation"])
                    if candidate["observed_generation"] is not None
                    else prompt_generation(
                        str(candidate["extraction_prompt_version"] or "")
                    ),
                    str(candidate["interpretation_key"]),
                    int(candidate["polarity"]),
                    1 if int(candidate["is_current"]) else 0,
                    revision_clock(candidate),
                    str(candidate["chunk_id"]),
                    str(candidate["published_at"] or ""),
                    str(candidate["superseded_at"] or ""),
                ),
            )
            revision_changes = [
                (int(candidate["id"]), wanted)
                for wanted, candidate in enumerate(ordered_revisions, start=1)
                if int(candidate["revision"]) != wanted
            ]
            if revision_changes:
                temporary = max(
                    int(candidate["revision"]) for candidate in revisions
                ) + len(revisions) + 1
                for offset, (evidence_id, _wanted) in enumerate(
                    revision_changes
                ):
                    conn.execute(
                        "UPDATE kg_evidence SET revision=? WHERE id=?",
                        (temporary + offset, evidence_id),
                    )
                for evidence_id, wanted in revision_changes:
                    conn.execute(
                        "UPDATE kg_evidence SET revision=? WHERE id=?",
                        (wanted, evidence_id),
                    )

        lifecycle_rows = conn.execute(
            f"SELECT * FROM kg_edge_lifecycle WHERE edge_id IN ({placeholders}) "
            "ORDER BY event_at, event_key, id",
            members,
        ).fetchall()
        manual_event_origins = {
            (
                int(lifecycle["edge_id"]),
                str(lifecycle["event_key"])[len("manual-retraction:"):],
            ): int(lifecycle["id"])
            for lifecycle in lifecycle_rows
            if lifecycle["event_kind"] == "manual_retraction"
            and str(lifecycle["event_key"]).startswith("manual-retraction:")
        }
        manual_event_snapshots = {
            (
                int(lifecycle["edge_id"]),
                str(lifecycle["event_key"])[len("manual-retraction:"):],
            ): (
                lifecycle["event_at"], lifecycle["details"],
                int(lifecycle["direction"]), lifecycle["created_at"],
            )
            for lifecycle in lifecycle_rows
            if lifecycle["event_kind"] == "manual_retraction"
            and str(lifecycle["event_key"]).startswith("manual-retraction:")
        }

        for lifecycle in lifecycle_rows:
            lifecycle_id = int(lifecycle["id"])
            mapped_source = (
                id_map.get(int(lifecycle["source_evidence_id"]))
                if lifecycle["source_evidence_id"] is not None
                else None
            )
            event_key = str(lifecycle["event_key"])
            if lifecycle["event_kind"] == "claim_assertion" and mapped_source:
                source = conn.execute(
                    "SELECT source_session_id, source_message_id, evidence_kind, "
                    "revision FROM kg_evidence WHERE id = ?",
                    (mapped_source,),
                ).fetchone()
                event_key = claim_assertion_event_key(
                    source["source_session_id"], source["source_message_id"],
                    source["evidence_kind"], source["revision"],
                )
            collision = conn.execute(
                "SELECT * FROM kg_edge_lifecycle "
                "WHERE edge_id = ? AND event_key = ?",
                (survivor_id, event_key),
            ).fetchone()
            if collision is not None:
                source_dependencies = {
                    id_map.get(int(item["evidence_id"]), int(item["evidence_id"]))
                    for item in conn.execute(
                        "SELECT evidence_id FROM kg_lifecycle_dependencies "
                        "WHERE lifecycle_id = ?",
                        (lifecycle_id,),
                    ).fetchall()
                }
                target_dependencies = {
                    int(item["evidence_id"])
                    for item in conn.execute(
                        "SELECT evidence_id FROM kg_lifecycle_dependencies "
                        "WHERE lifecycle_id = ?",
                        (collision["id"],),
                    ).fetchall()
                }
                same = (
                    collision["event_kind"] == lifecycle["event_kind"]
                    and int(collision["direction"]) == int(lifecycle["direction"])
                    and collision["event_at"] == lifecycle["event_at"]
                    and collision["source_evidence_id"] == mapped_source
                    and collision["details"] == lifecycle["details"]
                    and source_dependencies == target_dependencies
                )
                if same:
                    target_lifecycle = int(collision["id"])
                    created_candidates: list[str] = []
                    for candidate in (
                        collision["created_at"], lifecycle["created_at"]
                    ):
                        try:
                            created_candidates.append(normalize_iso_timestamp(
                                candidate,
                                context="edge merge lifecycle transaction",
                            ))
                        except ValueError:
                            continue
                    earliest_created = (
                        min(created_candidates)
                        if created_candidates
                        else min(
                            (
                                collision["created_at"], lifecycle["created_at"]
                            ),
                            key=lambda value: (str(value), value is None),
                        )
                    )
                    if collision["created_at"] != earliest_created:
                        conn.execute(
                            "UPDATE kg_edge_lifecycle SET created_at=? WHERE id=?",
                            (earliest_created, target_lifecycle),
                        )
                    conn.execute(
                        "DELETE FROM kg_edge_lifecycle WHERE id = ?",
                        (lifecycle_id,),
                    )
                    continue
                suffix = hashlib.sha256(json.dumps(
                    [
                        lifecycle["event_kind"], lifecycle["direction"],
                        lifecycle["event_at"],
                        (
                            evidence_natural_identity(conn, mapped_source)
                            if mapped_source is not None else None
                        ),
                        sorted(
                            evidence_natural_identity(conn, item)
                            for item in source_dependencies
                        ),
                        lifecycle["details"],
                    ],
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                event_key = f"{event_key}:merge:{suffix}"
            conn.execute(
                "UPDATE kg_edge_lifecycle SET edge_id = ?, event_key = ?, "
                "source_evidence_id = ? WHERE id = ?",
                (survivor_id, event_key, mapped_source, lifecycle_id),
            )
            dependencies = conn.execute(
                "SELECT evidence_id FROM kg_lifecycle_dependencies "
                "WHERE lifecycle_id = ?",
                (lifecycle_id,),
            ).fetchall()
            for dependency in dependencies:
                old_evidence = int(dependency["evidence_id"])
                mapped = id_map.get(old_evidence, old_evidence)
                if mapped != old_evidence:
                    conn.execute(
                        "UPDATE OR IGNORE kg_lifecycle_dependencies "
                        "SET evidence_id = ? WHERE lifecycle_id = ? "
                        "AND evidence_id = ?",
                        (mapped, lifecycle_id, old_evidence),
                    )
                    conn.execute(
                        "DELETE FROM kg_lifecycle_dependencies "
                        "WHERE lifecycle_id = ? AND evidence_id = ?",
                        (lifecycle_id, old_evidence),
                    )

        for evidence_id in duplicate_ids:
            mapped_evidence = id_map[evidence_id]
            affected_lifecycle = conn.execute(
                """
                SELECT DISTINCT lifecycle.id, lifecycle.edge_id
                FROM kg_lifecycle_dependencies dependency
                JOIN kg_edge_lifecycle lifecycle
                  ON lifecycle.id = dependency.lifecycle_id
                WHERE dependency.evidence_id = ?
                """,
                (evidence_id,),
            ).fetchall()
            for affected in affected_lifecycle:
                dependent_edge_ids.add(int(affected["edge_id"]))
                conn.execute(
                    "UPDATE OR IGNORE kg_lifecycle_dependencies "
                    "SET evidence_id = ? WHERE lifecycle_id = ? "
                    "AND evidence_id = ?",
                    (mapped_evidence, affected["id"], evidence_id),
                )
                conn.execute(
                    "DELETE FROM kg_lifecycle_dependencies "
                    "WHERE lifecycle_id = ? AND evidence_id = ?",
                    (affected["id"], evidence_id),
                )
                conn.execute(
                    "UPDATE kg_edge_lifecycle SET dependency_count = ("
                    "SELECT COUNT(*) FROM kg_lifecycle_dependencies "
                    "WHERE lifecycle_id = ?) WHERE id = ?",
                    (affected["id"], affected["id"]),
                )
            conn.execute("DELETE FROM kg_evidence WHERE id = ?", (evidence_id,))

        signal_rows = conn.execute(
            f"SELECT * FROM kg_evidence_signals "
            f"WHERE edge_id IN ({placeholders}) ORDER BY signal_kind, signal_key, id",
            members,
        ).fetchall()
        for signal in signal_rows:
            lifecycle_id = (
                manual_event_origins.get(
                    (int(signal["edge_id"]), str(signal["signal_key"]))
                )
                if signal["signal_kind"] == "manual_retraction" else None
            )
            incoming_pair = manual_event_snapshots.get(
                (int(signal["edge_id"]), str(signal["signal_key"]))
            )
            collision = conn.execute(
                "SELECT * FROM kg_evidence_signals WHERE edge_id = ? "
                "AND signal_kind = ? AND signal_key = ?",
                (survivor_id, signal["signal_kind"], signal["signal_key"]),
            ).fetchone()
            if collision is not None:
                semantic = (
                    int(signal["polarity"]), int(signal["evidence_weight"]),
                    int(signal["counts_toward_confidence"]), signal["details"],
                    signal["created_at"],
                )
                existing_semantic = (
                    int(collision["polarity"]),
                    int(collision["evidence_weight"]),
                    int(collision["counts_toward_confidence"]),
                    collision["details"], collision["created_at"],
                )
                collision_lifecycle = (
                    conn.execute(
                        "SELECT event_at,details,direction,created_at "
                        "FROM kg_edge_lifecycle WHERE edge_id=? AND event_key=?",
                        (
                            survivor_id,
                            manual_retraction_event_key(signal["signal_key"]),
                        ),
                    ).fetchone()
                    if signal["signal_kind"] == "manual_retraction" else None
                )
                existing_pair = (
                    tuple(collision_lifecycle)
                    if collision_lifecycle is not None else None
                )
                if semantic == existing_semantic and incoming_pair == existing_pair:
                    conn.execute(
                        "DELETE FROM kg_evidence_signals WHERE id = ?",
                        (signal["id"],),
                    )
                    # Exact signal/event pairs collapse as a unit. The earlier
                    # lifecycle pass normally removed the duplicate already;
                    # this covers a legacy pair whose old key was noncanonical.
                    if lifecycle_id is not None and conn.execute(
                        "SELECT 1 FROM kg_edge_lifecycle WHERE id=?",
                        (lifecycle_id,),
                    ).fetchone() is not None:
                        conn.execute(
                            "DELETE FROM kg_edge_lifecycle WHERE id=?",
                            (lifecycle_id,),
                        )
                    continue
                suffix = hashlib.sha256(json.dumps(
                    [
                        signal["signal_kind"], signal["signal_key"], *semantic,
                        incoming_pair,
                    ],
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                signal_key = f"{signal['signal_key']}:merge:{suffix}"
            else:
                signal_key = signal["signal_key"]
            conn.execute(
                "UPDATE kg_evidence_signals SET edge_id = ?, signal_key = ? "
                "WHERE id = ?",
                (survivor_id, signal_key, signal["id"]),
            )
            if signal["signal_kind"] == "manual_retraction":
                if lifecycle_id is not None:
                    conn.execute(
                        "UPDATE kg_edge_lifecycle SET event_key=? WHERE id=?",
                        (manual_retraction_event_key(signal_key), lifecycle_id),
                    )

        recanonicalize_lifecycle_keys(conn)
        refresh_claim_extraction_outcomes(conn, observation_chunk_ids)

    finalize_chunk_extraction_reconciliation(conn, [survivor_id])
    from hymem.dreaming.bitemporal import recompute_edge_interval

    for dependent_edge_id in sorted(dependent_edge_ids - {survivor_id}):
        recompute_edge_interval(conn, dependent_edge_id)
    reconcile_edge_counts(conn, all_ids)
