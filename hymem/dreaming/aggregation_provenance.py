"""Exact source manifests for episodes and RAPTOR aggregation nodes.

Message ranges are presentation metadata, not provenance.  This module carries
the exact ordered occurrence set selected by an episode through every RAPTOR
level and binds each occurrence to the immutable lossless-message artifact that
actually stores it.  It deliberately lives below the query package: dreaming,
portability, and scoped retrieval all validate the same neutral proof type and
the query layer translates it to its public ``SourceOccurrence`` DTO.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Mapping

from hymem.dreaming.lossless import validate_message_coverage_artifact
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION


EPISODE_SOURCE_MANIFEST_VERSION = "episode-source-manifest-v1"
AGGREGATION_SOURCE_MANIFEST_VERSION = "aggregation-source-manifest-v1"
_CLAIM_SOURCE_MANIFEST_VERSION = "claim-source-manifest-v1"


@dataclass(frozen=True)
class BoundSourceOccurrence:
    """One exact source turn plus the durable artifact that proves it."""

    message_id: int
    session_id: str
    role: str
    source_peer_id: str | None
    source_workspace_id: str | None
    source_created_at: str | None
    coverage_chunk_id: str
    coverage_version: str
    content_hash: str

    @property
    def occurrence_identity(self) -> tuple[str, int]:
        return self.session_id, self.message_id

    def manifest_record(self, ordinal: int) -> dict[str, object]:
        return {
            "ordinal": ordinal,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "source_peer_id": self.source_peer_id,
            "source_workspace_id": self.source_workspace_id,
            "source_created_at": self.source_created_at,
            "coverage_chunk_id": self.coverage_chunk_id,
            "coverage_version": self.coverage_version,
            "content_hash": self.content_hash,
        }


def source_manifest_hash(
    version: str, occurrences: Iterable[BoundSourceOccurrence]
) -> str:
    """Hash an ordered manifest, including ordinals and ownership metadata."""

    payload = {
        "version": version,
        "sources": [
            occurrence.manifest_record(index)
            for index, occurrence in enumerate(occurrences)
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def aggregation_input_fingerprint(
    items: Iterable[Mapping[str, object]], *, extra_inputs: Iterable[str] = ()
) -> str:
    """Bind a fusion cache key to every effective prompt member.

    Member order is significant because it is the order rendered to the model.
    A member's exact title, summary, and source-manifest state participate, so an
    in-place episode rewrite cannot reuse a summary fused from older bytes.
    ``extra_inputs`` binds non-tree prompt material such as the root facts block.
    """

    members = []
    for item in items:
        members.append({
            "id": item.get("id"),
            # Level-0 rendering prefixes each episode with its session id.
            "session_id": item.get("session_id"),
            "title": item.get("title"),
            "summary": item.get("summary"),
            "source_manifest_hash": item.get("source_manifest_hash"),
            "source_provenance_complete": bool(
                item.get("source_provenance_complete", False)
            ),
        })
    payload = {
        "version": "aggregation-input-v1",
        "members": members,
        "extra_inputs": list(extra_inputs),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def combine_source_occurrences(
    groups: Iterable[Iterable[BoundSourceOccurrence]],
) -> tuple[BoundSourceOccurrence, ...]:
    """Canonical exact union, rejecting conflicting ownership.

    Citation order is model output and therefore is not provenance.  Sort the
    validated occurrence identities instead so reversed/duplicated citations
    publish the same manifest and do not spuriously re-key the aggregation
    cache.  Message ids are the store's global chronology; the remaining
    fields are stable identity tie-breakers for defensive portability.
    """

    result: list[BoundSourceOccurrence] = []
    seen: dict[tuple[str, int], BoundSourceOccurrence] = {}
    for group in groups:
        for occurrence in group:
            key = occurrence.occurrence_identity
            previous = seen.get(key)
            if previous is None:
                seen[key] = occurrence
            elif previous != occurrence:
                raise ValueError("one source occurrence has conflicting provenance")
    result.extend(seen.values())
    result.sort(
        key=lambda item: (
            item.message_id,
            item.session_id,
            item.coverage_chunk_id,
        )
    )
    return tuple(result)


def _bound_coverage_occurrence(
    conn: sqlite3.Connection,
    *,
    message_id: object,
    chunk_id: object,
    coverage_version: object,
) -> BoundSourceOccurrence:
    if (
        not isinstance(message_id, int)
        or isinstance(message_id, bool)
        or not isinstance(chunk_id, str)
        or not chunk_id
        or coverage_version != LOSSLESS_COVERAGE_VERSION
    ):
        raise ValueError("source manifest has an invalid coverage identity")
    proof = validate_message_coverage_artifact(
        conn,
        message_id=message_id,
        chunk_id=chunk_id,
        coverage_version=str(coverage_version),
    )
    row = conn.execute(
        "SELECT message_content_hash FROM message_retention_coverage "
        "WHERE message_id=? AND chunk_id=? AND coverage_version=?",
        (message_id, chunk_id, coverage_version),
    ).fetchone()
    if row is None or not isinstance(row["message_content_hash"], str):
        raise RuntimeError("coverage proof lacks a content hash")
    return BoundSourceOccurrence(
        message_id=proof.message_id,
        session_id=proof.session_id,
        role=proof.role,
        source_peer_id=proof.source_peer_id,
        source_workspace_id=proof.source_workspace_id,
        source_created_at=proof.source_created_at,
        coverage_chunk_id=proof.chunk_id,
        coverage_version=str(coverage_version),
        content_hash=row["message_content_hash"],
    )


def resolve_cited_episode_sources(
    conn: sqlite3.Connection,
    session_id: str,
    chunk_ids: Iterable[str],
) -> tuple[BoundSourceOccurrence, ...] | None:
    """Resolve cited chunks without expanding or trusting numeric ranges.

    Coverage chunks contribute their one canonical occurrence.  Extraction
    chunks contribute only their published, complete claim-source manifest.
    A legacy item with no citations remains explicitly unattributed.  Once an
    item cites anything, however, every citation is an authority claim: a bad
    proof raises so the caller's episode/cursor transaction rolls back instead
    of permanently advancing past a transiently corrupt source.
    """

    cited = list(chunk_ids)
    if not cited:
        return None
    if any(not isinstance(item, str) or not item for item in cited):
        raise ValueError("episode source citations must be non-empty chunk ids")
    groups: list[tuple[BoundSourceOccurrence, ...]] = []
    for chunk_id in cited:
        chunk = conn.execute(
            "SELECT id,session_id,start_message_id,end_message_id,text,"
            "chunk_kind,source_manifest_version,source_manifest_count "
            "FROM chunks WHERE id=?",
            (chunk_id,),
        ).fetchone()
        if chunk is None or chunk["session_id"] != session_id:
            raise ValueError("episode source chunk crosses a session boundary")
        if chunk["chunk_kind"] == "coverage":
            proof_row = conn.execute(
                "SELECT message_id,coverage_version "
                "FROM message_retention_coverage "
                "WHERE chunk_id=? AND source_session_id=? "
                "AND coverage_version=? ORDER BY message_id",
                (chunk_id, session_id, LOSSLESS_COVERAGE_VERSION),
            ).fetchall()
            if len(proof_row) != 1:
                raise ValueError("episode coverage citation is not canonical")
            occurrence = _bound_coverage_occurrence(
                conn,
                message_id=proof_row[0]["message_id"],
                chunk_id=chunk_id,
                coverage_version=proof_row[0]["coverage_version"],
            )
            groups.append((occurrence,))
            continue
        if (
            chunk["chunk_kind"] != "extraction"
            or chunk["source_manifest_version"]
            != _CLAIM_SOURCE_MANIFEST_VERSION
            or not isinstance(chunk["source_manifest_count"], int)
            or isinstance(chunk["source_manifest_count"], bool)
            or int(chunk["source_manifest_count"]) <= 0
        ):
            raise ValueError("episode extraction citation has no complete manifest")
        source_rows = conn.execute(
            "SELECT ordinal,source_message_id,source_session_id,"
            "source_coverage_chunk_id,source_coverage_version "
            "FROM chunk_message_sources WHERE chunk_id=? ORDER BY ordinal",
            (chunk_id,),
        ).fetchall()
        declared = int(chunk["source_manifest_count"])
        if (
            len(source_rows) != declared
            or any(
                row["ordinal"] != expected
                for expected, row in enumerate(source_rows)
            )
        ):
            raise ValueError("episode extraction citation has a corrupt manifest")
        resolved: list[BoundSourceOccurrence] = []
        for row in source_rows:
            occurrence = _bound_coverage_occurrence(
                conn,
                message_id=row["source_message_id"],
                chunk_id=row["source_coverage_chunk_id"],
                coverage_version=row["source_coverage_version"],
            )
            if occurrence.session_id != row["source_session_id"]:
                raise ValueError("episode extraction source ownership mismatches")
            resolved.append(occurrence)
        if (
            not resolved
            or any(item.session_id != session_id for item in resolved)
            or resolved[0].message_id != chunk["start_message_id"]
            or resolved[-1].message_id != chunk["end_message_id"]
            or any(
                left.message_id >= right.message_id
                for left, right in zip(resolved, resolved[1:])
            )
            or "\n".join(f"{item.role}: {item.content}" for item in (
                validate_message_coverage_artifact(
                    conn,
                    message_id=source.message_id,
                    chunk_id=source.coverage_chunk_id,
                    coverage_version=source.coverage_version,
                )
                for source in resolved
            )) != chunk["text"]
        ):
            raise ValueError("episode extraction source manifest is inconsistent")
        groups.append(tuple(resolved))
    combined = combine_source_occurrences(groups)
    if not combined:
        raise ValueError("episode source manifest is empty")
    return combined


_SOURCE_COLUMNS = (
    "ordinal,source_message_id,source_session_id,source_role,source_peer_id,"
    "source_workspace_id,source_created_at,source_coverage_chunk_id,"
    "source_coverage_version,source_content_hash"
)


def _insert_source_rows(
    conn: sqlite3.Connection,
    *,
    table: str,
    parent_column: str,
    parent_id: str,
    occurrences: tuple[BoundSourceOccurrence, ...],
) -> None:
    for ordinal, occurrence in enumerate(occurrences):
        conn.execute(
            f"INSERT INTO {table}({parent_column},{_SOURCE_COLUMNS}) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                parent_id,
                ordinal,
                occurrence.message_id,
                occurrence.session_id,
                occurrence.role,
                occurrence.source_peer_id,
                occurrence.source_workspace_id,
                occurrence.source_created_at,
                occurrence.coverage_chunk_id,
                occurrence.coverage_version,
                occurrence.content_hash,
            ),
        )


def unpublish_episode_source_manifest(
    conn: sqlite3.Connection, episode_id: str
) -> None:
    """Clear a manifest header and children before replacing episode bytes."""

    if not conn.in_transaction:
        raise RuntimeError("episode source manifest replacement requires a transaction")
    conn.execute(
        "UPDATE episodes SET source_manifest_version=NULL,"
        "source_manifest_count=0,source_manifest_hash=NULL,"
        "source_manifest_complete=0 WHERE id=?",
        (episode_id,),
    )
    conn.execute(
        "DELETE FROM episode_source_occurrences WHERE episode_id=?", (episode_id,)
    )


def persist_episode_source_manifest(
    conn: sqlite3.Connection,
    episode_id: str,
    occurrences: tuple[BoundSourceOccurrence, ...] | None,
) -> None:
    """Replace one episode's manifest atomically with its episode UPSERT."""

    canonical = (
        combine_source_occurrences((occurrences,)) if occurrences else None
    )
    # Unpublish before touching children.  A reader on another connection sees
    # either the old committed manifest or the new committed manifest, never a
    # complete header paired with a partially replaced child set.
    unpublish_episode_source_manifest(conn, episode_id)
    if canonical is None:
        return
    # This helper is a publication boundary, not merely an INSERT loop.  Accept
    # equivalent caller ordering/duplicates but persist only the one canonical
    # exact union that the loader validates; conflicting duplicate ownership
    # raises before any child is published and rolls back the caller's tx.
    _insert_source_rows(
        conn,
        table="episode_source_occurrences",
        parent_column="episode_id",
        parent_id=episode_id,
        occurrences=canonical,
    )
    conn.execute(
        "UPDATE episodes SET source_manifest_version=?,source_manifest_count=?,"
        "source_manifest_hash=?,source_manifest_complete=1 WHERE id=?",
        (
            EPISODE_SOURCE_MANIFEST_VERSION,
            len(canonical),
            source_manifest_hash(EPISODE_SOURCE_MANIFEST_VERSION, canonical),
            episode_id,
        ),
    )


def _load_bound_rows(
    conn: sqlite3.Connection,
    *,
    table: str,
    parent_column: str,
    parent_id: str,
) -> tuple[BoundSourceOccurrence, ...] | None:
    rows = conn.execute(
        f"SELECT {_SOURCE_COLUMNS} FROM {table} "
        f"WHERE {parent_column}=? ORDER BY ordinal",
        (parent_id,),
    ).fetchall()
    occurrences: list[BoundSourceOccurrence] = []
    for expected_ordinal, row in enumerate(rows):
        if row["ordinal"] != expected_ordinal:
            return None
        try:
            proof = _bound_coverage_occurrence(
                conn,
                message_id=row["source_message_id"],
                chunk_id=row["source_coverage_chunk_id"],
                coverage_version=row["source_coverage_version"],
            )
        except (RuntimeError, TypeError, ValueError, sqlite3.Error):
            return None
        stored = BoundSourceOccurrence(
            message_id=row["source_message_id"],
            session_id=row["source_session_id"],
            role=row["source_role"],
            source_peer_id=row["source_peer_id"],
            source_workspace_id=row["source_workspace_id"],
            source_created_at=row["source_created_at"],
            coverage_chunk_id=row["source_coverage_chunk_id"],
            coverage_version=row["source_coverage_version"],
            content_hash=row["source_content_hash"],
        )
        if stored != proof:
            return None
        occurrences.append(stored)
    try:
        if combine_source_occurrences((occurrences,)) != tuple(occurrences):
            return None
    except ValueError:
        return None
    return tuple(occurrences)


def load_episode_source_manifest(
    conn: sqlite3.Connection, episode_id: str
) -> tuple[BoundSourceOccurrence, ...] | None:
    """Return an episode's complete, coverage-valid manifest or ``None``."""

    row = conn.execute(
        "SELECT session_id,source_manifest_version,source_manifest_count,"
        "source_manifest_hash,source_manifest_complete FROM episodes WHERE id=?",
        (episode_id,),
    ).fetchone()
    if (
        row is None
        or row["source_manifest_complete"] != 1
        or row["source_manifest_version"] != EPISODE_SOURCE_MANIFEST_VERSION
        or not isinstance(row["source_manifest_count"], int)
        or isinstance(row["source_manifest_count"], bool)
        or int(row["source_manifest_count"]) <= 0
    ):
        return None
    occurrences = _load_bound_rows(
        conn,
        table="episode_source_occurrences",
        parent_column="episode_id",
        parent_id=episode_id,
    )
    if (
        occurrences is None
        or len(occurrences) != int(row["source_manifest_count"])
        or any(item.session_id != row["session_id"] for item in occurrences)
        or row["source_manifest_hash"]
        != source_manifest_hash(EPISODE_SOURCE_MANIFEST_VERSION, occurrences)
    ):
        return None
    return occurrences


def persist_aggregation_source_manifest(
    conn: sqlite3.Connection,
    node_id: str,
    *,
    occurrences: tuple[BoundSourceOccurrence, ...] | None,
    input_fingerprint: str,
) -> None:
    """Replace one node's flattened source manifest without partial publish."""

    if not conn.in_transaction:
        raise RuntimeError("aggregation source manifest publication requires a transaction")
    canonical = (
        combine_source_occurrences((occurrences,)) if occurrences else None
    )
    # The fingerprint is part of a published node's immutable prompt input.
    # Unpublish first, then it can be replaced without weakening the SQL guard.
    conn.execute(
        "UPDATE aggregation_nodes SET source_manifest_version=?,"
        "source_manifest_count=0,source_manifest_hash=NULL,"
        "source_manifest_complete=0 WHERE id=?",
        (AGGREGATION_SOURCE_MANIFEST_VERSION, node_id),
    )
    conn.execute(
        "DELETE FROM aggregation_node_source_occurrences WHERE node_id=?", (node_id,)
    )
    conn.execute(
        "UPDATE aggregation_nodes SET input_fingerprint=? WHERE id=?",
        (input_fingerprint, node_id),
    )
    if canonical is None:
        return
    _insert_source_rows(
        conn,
        table="aggregation_node_source_occurrences",
        parent_column="node_id",
        parent_id=node_id,
        occurrences=canonical,
    )
    conn.execute(
        "UPDATE aggregation_nodes SET source_manifest_version=?,"
        "source_manifest_count=?,source_manifest_hash=?,"
        "source_manifest_complete=1,input_fingerprint=? WHERE id=?",
        (
            AGGREGATION_SOURCE_MANIFEST_VERSION,
            len(canonical),
            source_manifest_hash(AGGREGATION_SOURCE_MANIFEST_VERSION, canonical),
            input_fingerprint,
            node_id,
        ),
    )


def load_aggregation_source_manifest(
    conn: sqlite3.Connection,
    node_id: str,
    *,
    validate_level0_input: bool = True,
    _seen: frozenset[str] = frozenset(),
) -> tuple[BoundSourceOccurrence, ...] | None:
    """Return a complete node manifest, rejecting stale tree inputs."""

    if node_id in _seen:
        return None
    seen = _seen | {node_id}

    row = conn.execute(
        "SELECT id,level,is_root,member_episode_ids,session_ids,n_members,n_sessions,"
        "input_fingerprint,source_manifest_version,source_manifest_count,"
        "source_manifest_hash,source_manifest_complete "
        "FROM aggregation_nodes WHERE id=?",
        (node_id,),
    ).fetchone()
    if (
        row is None
        or row["source_manifest_complete"] != 1
        or row["source_manifest_version"] != AGGREGATION_SOURCE_MANIFEST_VERSION
        or not isinstance(row["source_manifest_count"], int)
        or isinstance(row["source_manifest_count"], bool)
        or int(row["source_manifest_count"]) <= 0
        or not isinstance(row["input_fingerprint"], str)
        or len(row["input_fingerprint"]) != 71
        or not row["input_fingerprint"].startswith("sha256:")
        or not isinstance(row["level"], int)
        or isinstance(row["level"], bool)
        or int(row["level"]) < 0
        or row["is_root"] not in (0, 1)
        or (row["level"] == 0 and row["is_root"] != 0)
    ):
        return None
    try:
        member_ids = json.loads(row["member_episode_ids"])
        session_ids = json.loads(row["session_ids"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(member_ids, list)
        or not member_ids
        or any(not isinstance(item, str) or not item for item in member_ids)
        or len(member_ids) != len(set(member_ids))
        or row["n_members"] != len(member_ids)
        or not isinstance(session_ids, list)
        or any(not isinstance(item, str) or not item for item in session_ids)
        or session_ids != sorted(set(session_ids))
        or row["n_sessions"] != len(session_ids)
    ):
        return None
    occurrences = _load_bound_rows(
        conn,
        table="aggregation_node_source_occurrences",
        parent_column="node_id",
        parent_id=node_id,
    )
    if (
        occurrences is None
        or len(occurrences) != int(row["source_manifest_count"])
        or sorted({item.session_id for item in occurrences}) != session_ids
        or row["source_manifest_hash"]
        != source_manifest_hash(AGGREGATION_SOURCE_MANIFEST_VERSION, occurrences)
    ):
        return None

    if validate_level0_input:
        descriptors: list[dict[str, object]] = []
        groups: list[tuple[BoundSourceOccurrence, ...]] = []
        for member_id in member_ids:
            if row["level"] > 0:
                child = conn.execute(
                    "SELECT id,level,title,summary,source_manifest_hash "
                    "FROM aggregation_nodes WHERE id=?",
                    (member_id,),
                ).fetchone()
                if child is not None:
                    if (
                        not isinstance(child["level"], int)
                        or isinstance(child["level"], bool)
                        or int(child["level"]) >= int(row["level"])
                    ):
                        return None
                    sources = load_aggregation_source_manifest(
                        conn,
                        member_id,
                        validate_level0_input=True,
                        _seen=seen,
                    )
                    if sources is None:
                        return None
                    descriptors.append({
                        "id": child["id"],
                        "session_id": None,
                        "title": child["title"],
                        "summary": child["summary"],
                        "source_manifest_hash": child["source_manifest_hash"],
                        "source_provenance_complete": True,
                    })
                    groups.append(sources)
                    continue
            episode = conn.execute(
                "SELECT e.id,e.session_id,e.title,e.summary,"
                "e.source_manifest_hash "
                "FROM episodes e JOIN sessions s ON s.id=e.session_id "
                "WHERE e.id=? AND (e.digest_generation IS NULL OR "
                "e.digest_generation=s.digest_published_generation)",
                (member_id,),
            ).fetchone()
            if episode is None:
                return None
            sources = load_episode_source_manifest(conn, member_id)
            if sources is None:
                return None
            descriptors.append({
                "id": episode["id"],
                # Session ids are rendered only by the level-0 cluster prompt.
                "session_id": (
                    episode["session_id"] if row["level"] == 0 else None
                ),
                "title": episode["title"],
                "summary": episode["summary"],
                "source_manifest_hash": episode["source_manifest_hash"],
                "source_provenance_complete": True,
            })
            groups.append(sources)
        try:
            expected_occurrences = combine_source_occurrences(groups)
        except ValueError:
            return None
        expected_fingerprint = aggregation_input_fingerprint(
            descriptors,
            extra_inputs=("(none)",) if row["is_root"] == 1 else (),
        )
        if (
            occurrences != expected_occurrences
            or row["input_fingerprint"] != expected_fingerprint
        ):
            return None
    return occurrences
