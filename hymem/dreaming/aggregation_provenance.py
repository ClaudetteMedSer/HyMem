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
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from hymem.core.time import EVENT_CLOCK_SKEW_SECONDS
from hymem.dreaming.lossless import validate_message_coverage_artifact
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.extraction.llm import LLMRequest
from hymem.extraction.embeddings import embedding_text_hash
from hymem.extraction.prompts import (
    AGGREGATE_SYSTEM, AGGREGATE_USER_TEMPLATE,
    DIGEST_SYSTEM, DIGEST_USER_TEMPLATE,
    ROLLUP_SYSTEM, ROLLUP_USER_TEMPLATE,
)


EPISODE_SOURCE_MANIFEST_VERSION = "episode-source-manifest-v1"
AGGREGATION_SOURCE_MANIFEST_VERSION = "aggregation-source-manifest-v1"
AGGREGATION_INPUT_MANIFEST_VERSION = "aggregation-input-manifest-v1"
AGGREGATION_INPUT_SOURCE_VERSION = "aggregation-input-source-v1"
AGGREGATION_OUTPUT_VERSION = "aggregation-output-v2"
AGGREGATION_REQUEST_VERSION = "aggregation-request-v1"
AGGREGATION_PUBLICATION_VERSION = "aggregation-publication-v4"
AGGREGATION_MAX_TITLE_CHARS = 300
AGGREGATION_MAX_SUMMARY_CHARS = 2000
AGGREGATION_PUBLICATION_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# These salts are provenance semantics, not merely prompt-cache decoration.
# The validator recomputes every node id from its typed inputs, so producer and
# reader must share one dependency-neutral definition.
AGGREGATION_CLUSTER_SALT = "cluster.v5"
AGGREGATION_ROLLUP_SALT = "rollup.v4"
AGGREGATION_ROOT_SALT = "root.v5"
_CLAIM_SOURCE_MANIFEST_VERSION = "claim-source-manifest-v1"
_EMBEDDING_CLIENT_UNSET = object()


def _sha256_payload(version: str, payload: object) -> str:
    encoded = json.dumps(
        {"version": version, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def aggregation_canonical_json(value: object) -> str:
    """Canonical durable JSON shared by aggregation producers and readers."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def aggregation_publication_timestamp_is_canonical(value: object) -> bool:
    """Accept only SQLite's bounded, injection-safe UTC clock rendering."""

    if not isinstance(value, str) or len(value) != 19:
        return False
    try:
        parsed = datetime.strptime(value, AGGREGATION_PUBLICATION_TIMESTAMP_FORMAT)
    except ValueError:
        return False
    return parsed.strftime(AGGREGATION_PUBLICATION_TIMESTAMP_FORMAT) == value


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


@dataclass(frozen=True)
class AggregationInputProof:
    """One typed effective prompt input and its exact lossless sources.

    ``source_ref`` is an explicit authoritative identity.  It is never
    resolved by title, summary, numeric range, similarity, or a table rowid.
    ``rendered_text`` is the exact normal (non-lossy) member/anchor text used
    by the fusion prompt; aggregation no longer publishes shrink-retry output.
    """

    kind: str
    source_ref: Mapping[str, object]
    rendered_text: str
    authority_hash: str
    occurrences: tuple[BoundSourceOccurrence, ...]
    phase1_generation_keys: tuple[str, ...] = ()

    @property
    def source_ref_json(self) -> str:
        return aggregation_canonical_json(dict(self.source_ref))

    @property
    def source_key(self) -> str:
        return _sha256_payload(
            "aggregation-input-key-v1",
            {"kind": self.kind, "source_ref": dict(self.source_ref)},
        )

    @property
    def payload_hash(self) -> str:
        return _sha256_payload("aggregation-input-payload-v1", self.rendered_text)

    @property
    def source_hash(self) -> str:
        return source_manifest_hash(
            AGGREGATION_INPUT_SOURCE_VERSION, self.occurrences
        )

    @property
    def proof_hash(self) -> str:
        return _sha256_payload(
            "aggregation-typed-input-proof-v1",
            self.manifest_record(0, include_ordinal=False),
        )

    def manifest_record(
        self, ordinal: int, *, include_ordinal: bool = True,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "input_kind": self.kind,
            "source_key": self.source_key,
            "source_ref_json": self.source_ref_json,
            "payload_hash": self.payload_hash,
            "authority_hash": self.authority_hash,
            "source_manifest_count": len(self.occurrences),
            "source_manifest_hash": self.source_hash,
        }
        if include_ordinal:
            record["ordinal"] = ordinal
        if self.phase1_generation_keys:
            record["phase1_generation_keys"] = list(self.phase1_generation_keys)
        return record


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


def aggregation_input_manifest_hash(
    inputs: Sequence[AggregationInputProof],
) -> str:
    """Bind ordered typed inputs, authoritative identities, and proof sets."""

    return _sha256_payload(
        AGGREGATION_INPUT_MANIFEST_VERSION,
        [item.manifest_record(index) for index, item in enumerate(inputs)],
    )


def aggregation_typed_input_fingerprint(
    inputs: Sequence[AggregationInputProof],
) -> str:
    """The cache/node fingerprint for the exact effective prompt inputs."""

    return _sha256_payload(
        "aggregation-input-v2",
        {
            "input_manifest_version": AGGREGATION_INPUT_MANIFEST_VERSION,
            "input_manifest_hash": aggregation_input_manifest_hash(inputs),
            "proof_hashes": [item.proof_hash for item in inputs],
        },
    )


def aggregation_llm_request_hash(request: LLMRequest) -> str:
    """Hash the exact immutable request object handed to ``complete``."""

    if not isinstance(request, LLMRequest):
        raise ValueError("aggregation request must be an LLMRequest")
    if not isinstance(request.system, str) or not isinstance(request.user, str):
        raise ValueError("aggregation request prompts must be text")
    if request.response_format not in {"json", "text"}:
        raise ValueError("aggregation request response format is invalid")
    if (
        not isinstance(request.max_tokens, int)
        or isinstance(request.max_tokens, bool)
        or request.max_tokens <= 0
    ):
        raise ValueError("aggregation request max_tokens is invalid")
    if (
        not isinstance(request.temperature, (int, float))
        or isinstance(request.temperature, bool)
        or not math.isfinite(request.temperature)
    ):
        raise ValueError("aggregation request temperature is invalid")
    return _sha256_payload(AGGREGATION_REQUEST_VERSION, {
        "system": request.system,
        "user": request.user,
        "response_format": request.response_format,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    })


def aggregation_fusion_max_tokens(prompt: str) -> int:
    return min(8192, 2048 + len(prompt) // 2)


def aggregation_llm_request(
    inputs: Sequence[AggregationInputProof], *, node_kind: str,
) -> LLMRequest:
    """Reconstruct the exact request from validated, ordered typed inputs."""

    members = [
        item for item in inputs
        if item.kind in {"episode", "aggregation_node"}
    ]
    anchors = [item for item in inputs if item not in members]
    text = "\n\n---\n\n".join(item.rendered_text for item in members)
    if node_kind == "cluster":
        if anchors:
            raise ValueError("cluster request cannot carry anchors")
        system = AGGREGATE_SYSTEM
        user = AGGREGATE_USER_TEMPLATE.format(text=text)
    elif node_kind == "rollup":
        if anchors:
            raise ValueError("rollup request cannot carry anchors")
        system = ROLLUP_SYSTEM
        user = ROLLUP_USER_TEMPLATE.format(text=text)
    elif node_kind == "root":
        facts = (
            "\n".join(f"- {item.rendered_text}" for item in anchors)
            if anchors else "(none)"
        )
        system = DIGEST_SYSTEM
        user = DIGEST_USER_TEMPLATE.format(facts=facts, text=text)
    else:
        raise ValueError("invalid aggregation request kind")
    return LLMRequest(
        system=system, user=user, response_format="json",
        max_tokens=aggregation_fusion_max_tokens(user), temperature=0.0,
    )


def aggregation_output_hash(
    node_kind: str, title: str, summary: str,
    request_hash: str | None = None,
) -> str:
    """Bind a node's own model-produced bytes at every hierarchy level."""

    return _sha256_payload(
        AGGREGATION_OUTPUT_VERSION,
        {
            "node_kind": node_kind, "title": title, "summary": summary,
            "aggregation_request_hash": request_hash,
        },
    )


def aggregation_output_is_canonical(title: object, summary: object) -> bool:
    """Whether persisted fusion bytes are in the producer's output language."""

    return bool(
        isinstance(title, str)
        and isinstance(summary, str)
        and title
        and summary
        and title.strip() == title
        and summary.strip() == summary
        and len(title) <= AGGREGATION_MAX_TITLE_CHARS
        and len(summary) <= AGGREGATION_MAX_SUMMARY_CHARS
    )


def aggregation_node_id(
    member_ids: Sequence[str], *, node_kind: str, input_fingerprint: str,
    generation_key: str | None = None, request_hash: str | None = None,
) -> str:
    """Recomputable node identity over typed-input provenance.

    Member order is already bound by ``input_fingerprint``; sorting here keeps
    the historical content-defined id property while the fingerprint prevents
    an order-changing prompt from aliasing the old output.
    """

    salts = {
        "cluster": AGGREGATION_CLUSTER_SALT,
        "rollup": AGGREGATION_ROLLUP_SALT,
        "root": AGGREGATION_ROOT_SALT,
    }
    if node_kind not in salts:
        raise ValueError("invalid aggregation node kind")
    payload = (
        f"{salts[node_kind]}::{'|'.join(sorted(member_ids))}::"
        f"input={input_fingerprint}::generation={generation_key}::"
        f"request={request_hash}"
    )
    # v55 intentionally retires the legacy 64-bit truncated SHA-1 namespace.
    # A full SHA-256 identity makes collision an infeasible integrity event;
    # any actual key collision is still rejected by the exact proof validator.
    return "agg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def aggregation_authority_hash(kind: str, payload: object) -> str:
    """Stable typed hash of the authoritative row(s) behind one input."""

    return _sha256_payload(
        "aggregation-input-authority-v2", {"kind": kind, "authority": payload}
    )


def aggregation_publication_node_set_hash(rows: Iterable[Mapping[str, object]]) -> str:
    """Hash the exact structural/output headers of a candidate publication."""

    fields = (
        "id", "title", "summary", "member_episode_ids", "session_ids",
        "n_members", "n_sessions", "level", "is_root", "node_kind",
        "output_hash", "input_fingerprint", "input_manifest_version",
        "input_manifest_count", "input_manifest_hash",
        "input_manifest_complete",
        "source_manifest_version", "source_manifest_count",
        "source_manifest_hash", "source_manifest_complete",
        "build_config_version", "aggregation_generation_key",
        "aggregation_request_hash", "aggregation_material_epoch_key",
    )
    normalized = sorted(
        ({field: row.get(field) for field in fields} for row in rows),
        key=lambda item: str(item.get("id")),
    )
    return _sha256_payload("aggregation-node-set-v2", normalized)


def aggregation_node_embedding_set_hash(
    rows: Iterable[Mapping[str, object]],
) -> str:
    """Commit to every retrieval-effective published node vector."""

    records = []
    for row in rows:
        vector_json = row.get("vector_json")
        vector_digest = (
            "sha256:" + hashlib.sha256(str(vector_json).encode("utf-8")).hexdigest()
            if isinstance(vector_json, str) else None
        )
        records.append({
            "node_id": row.get("node_id"),
            "model": row.get("model"),
            "dimension": row.get("dim"),
            "text_hash": row.get("text_hash"),
            "embedding_producer_key": row.get("embedding_producer_key"),
            "vector_sha256": vector_digest,
        })
    return _sha256_payload(
        "aggregation-node-embedding-set-v1",
        sorted(records, key=lambda item: str(item["node_id"])),
    )


def aggregation_publication_id(
    *, config_version: str, root_id: str | None, node_count: int,
    node_set_hash: str, published_at: str, cluster_min_members: int,
    cluster_min_sessions: int, anchor_fact_cap: int,
    generation_key: str | None = None,
    request_contract_sha256: str | None = None,
    material_epoch_key: str | None = None,
    material_revision: int | None = None,
    node_embedding_count: int | None = None,
    node_embedding_set_hash: str | None = None,
) -> str:
    if not aggregation_publication_timestamp_is_canonical(published_at):
        raise ValueError("aggregation publication timestamp is not canonical")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (cluster_min_members, cluster_min_sessions)
    ):
        raise ValueError("aggregation cluster minima are malformed")
    if (
        isinstance(anchor_fact_cap, bool)
        or not isinstance(anchor_fact_cap, int)
        or anchor_fact_cap < 0
    ):
        raise ValueError("aggregation anchor cap is malformed")
    return _sha256_payload(
        AGGREGATION_PUBLICATION_VERSION,
        {
            "config_version": config_version,
            "root_id": root_id,
            "node_count": node_count,
            "node_set_hash": node_set_hash,
            "published_at": published_at,
            "cluster_min_members": cluster_min_members,
            "cluster_min_sessions": cluster_min_sessions,
            "anchor_fact_cap": anchor_fact_cap,
            "aggregation_generation_key": generation_key,
            "aggregation_material_epoch_key": material_epoch_key,
            "material_revision": material_revision,
            "node_embedding_count": node_embedding_count,
            "node_embedding_set_hash": node_embedding_set_hash,
            "request_contract_sha256": request_contract_sha256,
        },
    )


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
    _memo: dict[tuple[int, str, str], BoundSourceOccurrence] | None = None,
) -> BoundSourceOccurrence:
    if (
        not isinstance(message_id, int)
        or isinstance(message_id, bool)
        or not isinstance(chunk_id, str)
        or not chunk_id
        or coverage_version != LOSSLESS_COVERAGE_VERSION
    ):
        raise ValueError("source manifest has an invalid coverage identity")
    cache_key = (message_id, chunk_id, str(coverage_version))
    if _memo is not None and cache_key in _memo:
        return _memo[cache_key]
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
    result = BoundSourceOccurrence(
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
    if _memo is not None:
        _memo[cache_key] = result
    return result


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
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
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
                _memo=_coverage_memo,
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
    conn: sqlite3.Connection, episode_id: str, *,
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
    _manifest_memo: dict[
        str, tuple[BoundSourceOccurrence, ...] | None
    ] | None = None,
    _episode_row: Mapping[str, object] | sqlite3.Row | None = None,
) -> tuple[BoundSourceOccurrence, ...] | None:
    """Return an episode's complete, coverage-valid manifest or ``None``."""

    if _manifest_memo is not None and episode_id in _manifest_memo:
        return _manifest_memo[episode_id]

    row = _episode_row
    if row is None:
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
        if _manifest_memo is not None:
            _manifest_memo[episode_id] = None
        return None
    occurrences = _load_bound_rows(
        conn,
        table="episode_source_occurrences",
        parent_column="episode_id",
        parent_id=episode_id,
        _coverage_memo=_coverage_memo,
    )
    if (
        occurrences is None
        or len(occurrences) != int(row["source_manifest_count"])
        or any(item.session_id != row["session_id"] for item in occurrences)
        or row["source_manifest_hash"]
        != source_manifest_hash(EPISODE_SOURCE_MANIFEST_VERSION, occurrences)
    ):
        result = None
    else:
        result = occurrences
    if _manifest_memo is not None:
        _manifest_memo[episode_id] = result
    return result


_HASH_PREFIX_LENGTH = 71
_AGGREGATION_INPUT_KINDS = frozenset({
    "episode", "aggregation_node", "user_profile",
    "knowledge_graph", "narrative_fact",
})
_AGGREGATION_ANCHOR_KINDS = frozenset({
    "user_profile", "knowledge_graph", "narrative_fact",
})


def _valid_hash(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == _HASH_PREFIX_LENGTH
        and value.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def _valid_build_config(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 92
        and value.startswith("aggregation-build-config-v1:")
        and all(char in "0123456789abcdef" for char in value[28:])
    )


def episode_authority_hash(row: Mapping[str, object]) -> str:
    return aggregation_authority_hash("episode", {
        key: row.get(key) for key in (
            "id", "session_id", "title", "summary", "digest_slice_key",
            "digest_generation", "key_entities", "source_manifest_version",
            "source_manifest_count", "source_manifest_hash",
            "source_manifest_complete",
        )
    })


def aggregation_node_authority_hash(row: Mapping[str, object]) -> str:
    return aggregation_authority_hash("aggregation_node", {
        key: row.get(key) for key in (
            "id", "title", "summary", "member_episode_ids", "session_ids",
            "n_members", "n_sessions", "level", "is_root", "node_kind",
            "output_hash", "input_fingerprint", "input_manifest_version",
            "input_manifest_count", "input_manifest_hash",
            "input_manifest_complete",
            "source_manifest_version", "source_manifest_count",
            "source_manifest_hash", "source_manifest_complete",
            "build_config_version", "aggregation_generation_key",
            "aggregation_request_hash",
        )
    })


def make_aggregation_input_proof(
    *, kind: str, source_ref: Mapping[str, object], rendered_text: str,
    authority_payload: object | None = None, authority_hash: str | None = None,
    occurrences: Iterable[BoundSourceOccurrence],
    phase1_generation_keys: Iterable[str] = (),
) -> AggregationInputProof:
    """Canonical constructor used by producers and typed anchor loaders."""

    if kind not in _AGGREGATION_INPUT_KINDS:
        raise ValueError("invalid aggregation input kind")
    if not isinstance(rendered_text, str):
        raise ValueError("aggregation input text must be a string")
    canonical = combine_source_occurrences((tuple(occurrences),))
    if not canonical:
        raise ValueError("aggregation input proof is empty")
    computed_authority = (
        authority_hash
        if authority_hash is not None
        else aggregation_authority_hash(kind, authority_payload)
    )
    if not _valid_hash(computed_authority):
        raise ValueError("aggregation input authority hash is malformed")
    # Force strict/canonical JSON now rather than at persistence time.
    canonical_ref = json.loads(aggregation_canonical_json(dict(source_ref)))
    if not isinstance(canonical_ref, dict):
        raise ValueError("aggregation input reference must be an object")
    raw_generation_keys = tuple(phase1_generation_keys)
    if any(
        not isinstance(key, str) or not key or "\x00" in key
        for key in raw_generation_keys
    ):
        raise ValueError("aggregation input Phase-1 scope is malformed")
    generation_keys = tuple(sorted(set(raw_generation_keys)))
    return AggregationInputProof(
        kind=kind,
        source_ref=canonical_ref,
        rendered_text=rendered_text,
        authority_hash=computed_authority,
        occurrences=canonical,
        phase1_generation_keys=generation_keys,
    )


def episode_input_proof(
    conn: sqlite3.Connection, episode_id: str, *, with_session_prefix: bool,
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
    _manifest_memo: dict[
        str, tuple[BoundSourceOccurrence, ...] | None
    ] | None = None,
    _proof_memo: dict[
        tuple[str, bool], AggregationInputProof | None
    ] | None = None,
) -> AggregationInputProof | None:
    proof_key = (episode_id, with_session_prefix)
    if _proof_memo is not None and proof_key in _proof_memo:
        return _proof_memo[proof_key]
    row = conn.execute(
        "SELECT e.*,s.digest_published_generation FROM episodes e "
        "JOIN sessions s ON s.id=e.session_id WHERE e.id=? AND "
        "(e.digest_generation IS NULL OR "
        " e.digest_generation=s.digest_published_generation)",
        (episode_id,),
    ).fetchone()
    sources = load_episode_source_manifest(
        conn, episode_id, _coverage_memo=_coverage_memo,
        _manifest_memo=_manifest_memo,
    )
    if row is None or sources is None:
        if _proof_memo is not None:
            _proof_memo[proof_key] = None
        return None
    rendered = f"{row['title']}\n{row['summary']}"
    if with_session_prefix:
        rendered = f"[{row['session_id']}] {rendered}"
    values = dict(row)
    result = make_aggregation_input_proof(
        kind="episode", source_ref={"id": row["id"]},
        rendered_text=rendered, authority_hash=episode_authority_hash(values),
        occurrences=sources,
    )
    if _proof_memo is not None:
        _proof_memo[proof_key] = result
    return result


def _profile_anchor_from_row(
    conn: sqlite3.Connection, row: sqlite3.Row,
    *,
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
    _source_memo: dict[
        tuple[int, str], BoundSourceOccurrence | None
    ] | None = None,
) -> AggregationInputProof | None:
    message_id = row["source_message_id"]
    session_id = row["source_session_id"]
    if (
        not isinstance(message_id, int) or isinstance(message_id, bool)
        or not isinstance(session_id, str) or not session_id
        or row["invalid_at"] is not None
    ):
        return None
    source_key = (message_id, session_id)
    memoized = _source_memo is not None and source_key in _source_memo
    source = _source_memo.get(source_key) if memoized else None
    if not memoized:
        proof_row = conn.execute(
            "SELECT chunk_id,coverage_version FROM message_retention_coverage "
            "WHERE message_id=? AND source_session_id=? AND "
            "coverage_version=? ORDER BY chunk_id",
            (message_id, session_id, LOSSLESS_COVERAGE_VERSION),
        ).fetchall()
        if len(proof_row) == 1:
            try:
                source = _bound_coverage_occurrence(
                    conn, message_id=message_id,
                    chunk_id=proof_row[0]["chunk_id"],
                    coverage_version=proof_row[0]["coverage_version"],
                    _memo=_coverage_memo,
                )
            except (RuntimeError, TypeError, ValueError, sqlite3.Error):
                source = None
        if _source_memo is not None:
            _source_memo[source_key] = source
    if source is None:
        return None
    if (
        source.session_id != session_id or source.role != "user"
        or source.source_created_at != row["source_created_at"]
    ):
        return None
    slot_key = row["slot_key"]
    rendered = (
        f"user {row['slot']}({slot_key}) {row['value']}"
        if slot_key else f"user {row['slot']} {row['value']}"
    )
    ref = {
        "slot": row["slot"], "slot_key": slot_key, "value": row["value"],
        "source_session_id": session_id, "source_message_id": message_id,
    }
    authority = {
        **ref, "confidence": row["confidence"], "valid_at": row["valid_at"],
        "invalid_at": row["invalid_at"],
        "source_created_at": row["source_created_at"],
    }
    try:
        return make_aggregation_input_proof(
            kind="user_profile", source_ref=ref, rendered_text=rendered,
            authority_payload=authority, occurrences=(source,),
        )
    except (TypeError, ValueError):
        return None


def load_profile_anchor_inputs(
    conn: sqlite3.Connection, cap: int,
    *,
    _anchor_memo: dict[
        tuple[str, int], AggregationInputProof | None
    ] | None = None,
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
    _source_memo: dict[
        tuple[int, str], BoundSourceOccurrence | None
    ] | None = None,
) -> list[AggregationInputProof]:
    """Return only exact active profile anchors; invalid rows do not use cap."""

    if cap <= 0:
        return []
    slot_order = (
        "name", "role", "employer", "location", "age_birthday", "language",
        "relationship", "possession", "health_condition", "recurring_activity",
    )
    order = {slot: index for index, slot in enumerate(slot_order)}
    slot_case = "CASE slot " + " ".join(
        f"WHEN '{slot}' THEN {index}" for index, slot in enumerate(slot_order)
    ) + f" ELSE {len(slot_order)} END"
    try:
        cursor = conn.execute(
            "SELECT id,slot,slot_key,value,confidence,valid_at,invalid_at,"
            "source_message_id,source_session_id,source_created_at "
            "FROM user_profile WHERE invalid_at IS NULL ORDER BY "
            + slot_case
            + ",COALESCE(slot_key,''),confidence DESC,value,"
              "source_session_id,source_message_id,source_created_at,id"
        )
    except sqlite3.OperationalError:
        return []
    # Validate before applying the cap, then break every semantic tie with
    # portable source/proof coordinates.  SQLite row order (and user_profile's
    # local surrogate allocation) must never decide which exact anchor crosses
    # a capped root prompt after export/import.
    candidates: list[tuple[tuple[object, ...], AggregationInputProof]] = []
    boundary_prefix: tuple[object, ...] | None = None
    batch_size = max(32, min(256, cap * 4))
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        stop = False
        for row in rows:
            prefix = (
                order.get(str(row["slot"]), len(order)),
                str(row["slot_key"] or ""),
                -float(row["confidence"]),
                str(row["value"]),
                str(row["source_session_id"]),
                int(row["source_message_id"])
                if isinstance(row["source_message_id"], int)
                and not isinstance(row["source_message_id"], bool)
                else -1,
                str(row["source_created_at"]),
            )
            if boundary_prefix is not None and prefix > boundary_prefix:
                stop = True
                break
            memo_key = ("user_profile", int(row["id"]))
            if _anchor_memo is not None and memo_key in _anchor_memo:
                proof = _anchor_memo[memo_key]
            else:
                proof = _profile_anchor_from_row(
                    conn, row, _coverage_memo=_coverage_memo,
                    _source_memo=_source_memo,
                )
                if _anchor_memo is not None:
                    _anchor_memo[memo_key] = proof
            if proof is not None:
                source = proof.occurrences[0]
                candidates.append((
                    (
                        *prefix,
                        source.source_peer_id or "",
                        source.source_workspace_id or "",
                        source.coverage_chunk_id,
                        source.coverage_version,
                        source.content_hash,
                        proof.authority_hash,
                    ),
                    proof,
                ))
                if len(candidates) == cap:
                    boundary_prefix = prefix
        if stop:
            break
    candidates.sort(key=lambda item: item[0])
    return [proof for _key, proof in candidates[:cap]]


def _evidence_phase1_generation_keys(
    conn: sqlite3.Connection, evidence_rows: Sequence[sqlite3.Row],
) -> tuple[str, ...] | None:
    """Resolve the exact current producer generations supporting evidence.

    This is the generation projection of ``_phase1_publication_sql`` in
    ``query.graph_state``.  Keeping it attached to the selected typed anchor
    prevents the global Phase-1 registry from becoming unrelated root input.
    """

    keys: set[str] = set()
    for evidence in evidence_rows:
        rows = conn.execute(
            """
            SELECT DISTINCT observation.phase1_generation_key
            FROM kg_claim_observations observation
            JOIN kg_claim_extraction_outcomes outcome
              ON outcome.chunk_id=observation.chunk_id
             AND outcome.prompt_version=observation.prompt_version
             AND outcome.prompt_generation=observation.prompt_generation
            JOIN phase1_generations generation
              ON generation.generation_key=outcome.phase1_generation_key
             AND generation.extraction_cache_key=outcome.prompt_version
            WHERE observation.evidence_id=?
              AND observation.edge_id=?
              AND observation.source_session_id=?
              AND observation.source_message_id=?
              AND observation.evidence_kind=?
              AND observation.polarity=?
              AND observation.interpretation_key=?
              AND outcome.phase1_generation_key IS NOT NULL
              AND outcome.phase1_generation_key=observation.phase1_generation_key
              AND hymem_phase1_generation_is_current(
                    generation.generation_key,generation.identity_exact)=1
              AND hymem_normalize_iso_timestamp(observation.observed_at)
                    IS NOT NULL
              AND hymem_normalize_iso_timestamp(outcome.succeeded_at)
                    IS NOT NULL
              AND hymem_timestamp_at_or_before(
                    ?,observation.observed_at)=1
              AND hymem_timestamp_gap_within(
                    observation.observed_at,outcome.succeeded_at,?)=1
            ORDER BY observation.phase1_generation_key
            """,
            (
                evidence["id"], evidence["edge_id"],
                evidence["source_session_id"], evidence["source_message_id"],
                evidence["evidence_kind"], evidence["polarity"],
                evidence["interpretation_key"], evidence["extracted_at"],
                EVENT_CLOCK_SKEW_SECONDS,
            ),
        ).fetchall()
        if not rows:
            return None
        keys.update(str(row["phase1_generation_key"]) for row in rows)
    return tuple(sorted(keys))


def _knowledge_graph_anchor_from_row(
    conn: sqlite3.Connection, edge: sqlite3.Row,
    *,
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
) -> AggregationInputProof | None:
    # Confidence influenced by an unattributed/manual signal cannot be called
    # exact message-backed authority.
    if conn.execute(
        "SELECT 1 FROM kg_evidence_signals WHERE edge_id=? "
        "AND counts_toward_confidence=1 LIMIT 1", (edge["id"],),
    ).fetchone() is not None:
        return None
    from hymem.query.graph_state import validated_current_evidence

    authoritative = validated_current_evidence(conn, edge_id=int(edge["id"]))
    evidence_rows = conn.execute(
        "SELECT * FROM kg_evidence WHERE edge_id=? AND is_current=1 "
        "ORDER BY source_session_id,source_message_id,evidence_kind,revision,"
        "interpretation_key",
        (edge["id"],),
    ).fetchall()
    if (
        not evidence_rows
        or {int(item["id"]) for item in evidence_rows}
           != {int(item["id"]) for item in authoritative}
        or not any(int(item["polarity"]) == 1 for item in evidence_rows)
        or any(item["provenance_status"] != "canonical" for item in evidence_rows)
    ):
        return None
    phase1_generation_keys = _evidence_phase1_generation_keys(
        conn, evidence_rows,
    )
    if phase1_generation_keys is None:
        return None
    occurrences: list[BoundSourceOccurrence] = []
    authority_evidence: list[dict[str, object]] = []
    evidence_fields = (
        "polarity", "surface_subject", "surface_object", "value_text",
        "value_numeric", "value_unit", "temporal_scope", "source_role",
        "source_peer_id", "source_workspace_id", "evidence_kind",
        "evidence_weight", "weight_source", "extraction_prompt_version",
        "extracted_at", "published_at", "source_message_id",
        "source_session_id", "source_created_at", "source_event_at",
        "source_coverage_chunk_id", "source_coverage_version",
        "provenance_status", "interpretation_key", "revision", "is_current",
        "superseded_at", "superseded_reason",
    )
    for item in evidence_rows:
        try:
            occurrence = _bound_coverage_occurrence(
                conn, message_id=item["source_message_id"],
                chunk_id=item["source_coverage_chunk_id"],
                coverage_version=item["source_coverage_version"],
                _memo=_coverage_memo,
            )
        except (RuntimeError, TypeError, ValueError, sqlite3.Error):
            return None
        if (
            occurrence.session_id != item["source_session_id"]
            or occurrence.role != item["source_role"]
            or occurrence.source_peer_id != item["source_peer_id"]
            or occurrence.source_workspace_id != item["source_workspace_id"]
            or occurrence.source_created_at != item["source_created_at"]
        ):
            return None
        occurrences.append(occurrence)
        authority_evidence.append({field: item[field] for field in evidence_fields})
    ref = {
        "subject": edge["subject_canonical"], "predicate": edge["predicate"],
        "object": edge["object_canonical"],
    }
    edge_fields = (
        "subject_canonical", "predicate", "object_canonical", "pos_evidence",
        "neg_evidence", "first_seen", "last_seen", "last_reinforced",
        "valid_at", "invalid_at", "status", "derived",
    )
    authority = {
        "edge": {field: edge[field] for field in edge_fields},
        "evidence": authority_evidence,
    }
    try:
        return make_aggregation_input_proof(
            kind="knowledge_graph", source_ref=ref,
            rendered_text=(
                f"{edge['subject_canonical']} {edge['predicate']} "
                f"{edge['object_canonical']}"
            ), authority_payload=authority, occurrences=occurrences,
            phase1_generation_keys=phase1_generation_keys,
        )
    except (TypeError, ValueError):
        return None


def load_knowledge_graph_anchor_inputs(
    conn: sqlite3.Connection, cap: int,
    *,
    _anchor_memo: dict[
        tuple[str, int], AggregationInputProof | None
    ] | None = None,
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
) -> list[AggregationInputProof]:
    """Return valid direct KG anchors, filtering proof failures before cap."""

    if cap <= 0:
        return []
    from hymem.core.graph import graph_clock_order_sql

    try:
        cursor = conn.execute(
            "SELECT * FROM knowledge_graph kg WHERE kg.derived=0 "
            "AND kg.status='active' AND kg.invalid_at IS NULL "
            "AND kg.pos_evidence>kg.neg_evidence "
            "AND (kg.valid_at IS NULL OR hymem_timestamp_at_or_before("
            "kg.valid_at,strftime('%Y-%m-%dT%H:%M:%fZ','now',"
            f"'+{EVENT_CLOCK_SKEW_SECONDS} seconds'))=1) "
            "ORDER BY kg.pos_evidence-kg.neg_evidence DESC,"
            f"{graph_clock_order_sql('kg.last_seen')},"
            "kg.subject_canonical,kg.predicate,kg.object_canonical"
        )
    except sqlite3.OperationalError:
        return []
    result: list[AggregationInputProof] = []
    batch_size = max(32, min(256, cap * 4))
    while len(result) < cap:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            memo_key = ("knowledge_graph", int(row["id"]))
            if _anchor_memo is not None and memo_key in _anchor_memo:
                proof = _anchor_memo[memo_key]
            else:
                proof = _knowledge_graph_anchor_from_row(
                    conn, row, _coverage_memo=_coverage_memo,
                )
                if _anchor_memo is not None:
                    _anchor_memo[memo_key] = proof
            if proof is not None:
                result.append(proof)
                if len(result) >= cap:
                    break
    return result


def narrative_fact_anchor_input(
    conn: sqlite3.Connection, source_outcome_key: str, fact_key: str,
) -> AggregationInputProof | None:
    """Resolve a narrative-fact anchor through its typed outcome authority."""

    from hymem.dreaming.facts import load_fact_source_manifest

    row = conn.execute(
        "SELECT * FROM narrative_facts WHERE source_outcome_key=? AND fact_key=? "
        "AND lifecycle_status='active' AND invalid_at IS NULL",
        (source_outcome_key, fact_key),
    ).fetchone()
    if row is None:
        return None
    sources = load_fact_source_manifest(conn, int(row["id"]))
    if sources is None:
        return None
    ref = {"source_outcome_key": source_outcome_key, "fact_key": fact_key}
    authority = {key: row[key] for key in (
        "session_id", "start_message_id", "end_message_id", "text",
        "fact_date", "entities", "prompt_version", "valid_at", "invalid_at",
        "source_outcome_key", "fact_key", "current_generation",
        "lifecycle_status",
    )}
    return make_aggregation_input_proof(
        kind="narrative_fact", source_ref=ref,
        rendered_text=str(row["text"]), authority_payload=authority,
        occurrences=sources,
    )


def load_root_anchor_inputs(
    conn: sqlite3.Connection, cap: int, *,
    _anchor_memo: dict[
        tuple[str, int], AggregationInputProof | None
    ] | None = None,
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
    _profile_source_memo: dict[
        tuple[int, str], BoundSourceOccurrence | None
    ] | None = None,
) -> list[AggregationInputProof]:
    """Profile-first root anchors with invalid sources excluded before cap."""

    profiles = load_profile_anchor_inputs(
        conn, cap, _anchor_memo=_anchor_memo,
        _coverage_memo=_coverage_memo, _source_memo=_profile_source_memo,
    )
    return profiles + load_knowledge_graph_anchor_inputs(
        conn, max(0, cap - len(profiles)), _anchor_memo=_anchor_memo,
        _coverage_memo=_coverage_memo,
    )


def _insert_input_source_rows(
    conn: sqlite3.Connection, node_id: str, input_ordinal: int,
    occurrences: Sequence[BoundSourceOccurrence],
) -> None:
    for source_ordinal, occurrence in enumerate(occurrences):
        conn.execute(
            "INSERT INTO aggregation_node_input_sources("
            "node_id,input_ordinal,source_ordinal,source_message_id,"
            "source_session_id,source_role,source_peer_id,source_workspace_id,"
            "source_created_at,source_coverage_chunk_id,source_coverage_version,"
            "source_content_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                node_id, input_ordinal, source_ordinal, occurrence.message_id,
                occurrence.session_id, occurrence.role, occurrence.source_peer_id,
                occurrence.source_workspace_id, occurrence.source_created_at,
                occurrence.coverage_chunk_id, occurrence.coverage_version,
                occurrence.content_hash,
            ),
        )


def persist_aggregation_source_manifest(
    conn: sqlite3.Connection,
    node_id: str,
    *,
    occurrences: tuple[BoundSourceOccurrence, ...] | None = None,
    input_fingerprint: str | None = None,
    inputs: Sequence[AggregationInputProof] | None = None,
    node_kind: str | None = None,
    publication_id: str | None = None,
    build_config_version: str | None = None,
    aggregation_generation_key: str | None = None,
    aggregation_material_epoch_key: str | None = None,
) -> None:
    """Atomically publish one node's complete typed proof.

    The legacy ``occurrences``/``input_fingerprint`` arguments remain accepted
    only to unpublish historical rows.  A complete v55 publication requires
    typed ``inputs`` and derives both flattened sources and fingerprint from
    them, so a caller cannot make the two ledgers disagree.
    """

    if not conn.in_transaction:
        raise RuntimeError("aggregation source manifest publication requires a transaction")
    typed_inputs = tuple(inputs or ())
    # v55/v56 callers did not have to repeat the v57 material identity when
    # republishing the proof of an already-bound physical node.  Resolve that
    # omission from the immutable row, but never let an explicit mismatch be
    # replaced silently.  Do this before withdrawing the old manifest so a
    # malformed request cannot partially mutate the row outside a managed
    # transaction.
    existing_identity = conn.execute(
        "SELECT aggregation_material_epoch_key FROM aggregation_nodes WHERE id=?",
        (node_id,),
    ).fetchone()
    if aggregation_material_epoch_key is None and existing_identity is not None:
        stored_material_key = existing_identity["aggregation_material_epoch_key"]
        if stored_material_key is not None:
            aggregation_material_epoch_key = str(stored_material_key)
    canonical = (
        combine_source_occurrences(item.occurrences for item in typed_inputs)
        if typed_inputs else None
    )
    derived_fingerprint = (
        aggregation_typed_input_fingerprint(typed_inputs)
        if typed_inputs else input_fingerprint
    )
    conn.execute(
        "UPDATE aggregation_nodes SET source_manifest_version=?,"
        "source_manifest_count=0,source_manifest_hash=NULL,"
        "source_manifest_complete=0,input_manifest_version=NULL,"
        "input_manifest_count=0,input_manifest_hash=NULL,"
        "input_manifest_complete=0 WHERE id=?",
        (AGGREGATION_SOURCE_MANIFEST_VERSION, node_id),
    )
    conn.execute("DELETE FROM aggregation_node_inputs WHERE node_id=?", (node_id,))
    conn.execute(
        "DELETE FROM aggregation_node_source_occurrences WHERE node_id=?", (node_id,)
    )
    if not typed_inputs or canonical is None:
        conn.execute(
            "UPDATE aggregation_nodes SET input_fingerprint=?,node_kind=?,"
            "publication_id=?,build_config_version=?,"
            "aggregation_generation_key=?,aggregation_material_epoch_key=? "
            "WHERE id=?",
            (
                derived_fingerprint, node_kind, publication_id,
                build_config_version, aggregation_generation_key,
                aggregation_material_epoch_key, node_id,
            ),
        )
        return
    if (
        node_kind not in {"cluster", "rollup", "root"}
        or not _valid_hash(publication_id)
        or not _valid_build_config(build_config_version)
    ):
        raise ValueError("aggregation publication identity is malformed")
    from hymem.dreaming.aggregation_generation import (
        aggregation_generation_key_is_shaped,
    )
    if not aggregation_generation_key_is_shaped(aggregation_generation_key):
        raise ValueError("aggregation generation identity is malformed")
    from hymem.dreaming.aggregation_material import (
        load_registered_aggregation_material_epoch,
    )
    material = load_registered_aggregation_material_epoch(
        conn, aggregation_material_epoch_key, allow_inexact=True,
    )
    if (
        material is None
        or material["config_version"] != build_config_version
    ):
        raise ValueError("aggregation material identity is malformed")
    output = conn.execute(
        "SELECT title,summary,aggregation_request_hash "
        "FROM aggregation_nodes WHERE id=?", (node_id,),
    ).fetchone()
    if output is None:
        raise ValueError("aggregation node disappeared before proof publication")
    output_digest = aggregation_output_hash(
        node_kind, output["title"], output["summary"],
        output["aggregation_request_hash"],
    )
    conn.execute(
        "UPDATE aggregation_nodes SET input_fingerprint=?,node_kind=?,"
        "output_hash=?,publication_id=?,build_config_version=?,"
        "aggregation_generation_key=?,aggregation_material_epoch_key=? "
        "WHERE id=?",
        (
            derived_fingerprint, node_kind, output_digest, publication_id,
            build_config_version, aggregation_generation_key,
            aggregation_material_epoch_key, node_id,
        ),
    )
    for ordinal, item in enumerate(typed_inputs):
        record = item.manifest_record(ordinal)
        conn.execute(
            "INSERT INTO aggregation_node_inputs("
            "node_id,ordinal,input_kind,source_key,source_ref_json,payload_hash,"
            "authority_hash,source_manifest_count,source_manifest_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                node_id, ordinal, record["input_kind"], record["source_key"],
                record["source_ref_json"], record["payload_hash"],
                record["authority_hash"], record["source_manifest_count"],
                record["source_manifest_hash"],
            ),
        )
        _insert_input_source_rows(conn, node_id, ordinal, item.occurrences)
    _insert_source_rows(
        conn, table="aggregation_node_source_occurrences",
        parent_column="node_id", parent_id=node_id, occurrences=canonical,
    )
    conn.execute(
        "UPDATE aggregation_nodes SET source_manifest_version=?,"
        "source_manifest_count=?,source_manifest_hash=?,"
        "source_manifest_complete=1,input_manifest_version=?,"
        "input_manifest_count=?,input_manifest_hash=?,"
        "input_manifest_complete=1 WHERE id=?",
        (
            AGGREGATION_SOURCE_MANIFEST_VERSION, len(canonical),
            source_manifest_hash(AGGREGATION_SOURCE_MANIFEST_VERSION, canonical),
            AGGREGATION_INPUT_MANIFEST_VERSION, len(typed_inputs),
            aggregation_input_manifest_hash(typed_inputs), node_id,
        ),
    )


def _load_input_sources(
    conn: sqlite3.Connection, node_id: str, input_ordinal: int, *,
    _coverage_memo: dict[
        tuple[int, str, str], BoundSourceOccurrence
    ] | None = None,
) -> tuple[BoundSourceOccurrence, ...] | None:
    rows = conn.execute(
        "SELECT source_ordinal,source_message_id,source_session_id,source_role,"
        "source_peer_id,source_workspace_id,source_created_at,"
        "source_coverage_chunk_id,source_coverage_version,source_content_hash "
        "FROM aggregation_node_input_sources WHERE node_id=? AND input_ordinal=? "
        "ORDER BY source_ordinal",
        (node_id, input_ordinal),
    ).fetchall()
    result: list[BoundSourceOccurrence] = []
    for expected, row in enumerate(rows):
        if row["source_ordinal"] != expected:
            return None
        try:
            proof = _bound_coverage_occurrence(
                conn, message_id=row["source_message_id"],
                chunk_id=row["source_coverage_chunk_id"],
                coverage_version=row["source_coverage_version"],
                _memo=_coverage_memo,
            )
        except (RuntimeError, TypeError, ValueError, sqlite3.Error):
            return None
        stored = BoundSourceOccurrence(
            message_id=row["source_message_id"], session_id=row["source_session_id"],
            role=row["source_role"], source_peer_id=row["source_peer_id"],
            source_workspace_id=row["source_workspace_id"],
            source_created_at=row["source_created_at"],
            coverage_chunk_id=row["source_coverage_chunk_id"],
            coverage_version=row["source_coverage_version"],
            content_hash=row["source_content_hash"],
        )
        if stored != proof:
            return None
        result.append(stored)
    try:
        canonical = combine_source_occurrences((result,))
    except ValueError:
        return None
    return canonical if canonical == tuple(result) else None


def _strict_ref(raw: object, keys: frozenset[str]) -> dict[str, object] | None:
    try:
        value = json.loads(raw) if isinstance(raw, str) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or frozenset(value) != keys:
        return None
    if aggregation_canonical_json(value) != raw:
        return None
    return value


def _resolve_anchor_input(
    conn: sqlite3.Connection, kind: str, ref: dict[str, object], *,
    context: "_AggregationValidationContext | None" = None,
) -> AggregationInputProof | None:
    if kind == "user_profile":
        expected = frozenset({
            "slot", "slot_key", "value", "source_session_id", "source_message_id",
        })
        if frozenset(ref) != expected:
            return None
        rows = conn.execute(
            "SELECT id,slot,slot_key,value,confidence,valid_at,invalid_at,"
            "source_message_id,source_session_id,source_created_at "
            "FROM user_profile WHERE slot=? AND slot_key IS ? AND value=? "
            "AND source_session_id=? AND source_message_id=? AND invalid_at IS NULL",
            (
                ref["slot"], ref["slot_key"], ref["value"],
                ref["source_session_id"], ref["source_message_id"],
            ),
        ).fetchall()
        if len(rows) != 1:
            return None
        memo_key = ("user_profile", int(rows[0]["id"]))
        if context is not None and memo_key in context.anchors:
            return context.anchors[memo_key]
        proof = _profile_anchor_from_row(
            conn, rows[0],
            _coverage_memo=context.coverage if context is not None else None,
            _source_memo=(
                context.profile_sources if context is not None else None
            ),
        )
        if context is not None:
            context.anchors[memo_key] = proof
        return proof
    if kind == "knowledge_graph":
        if frozenset(ref) != frozenset({"subject", "predicate", "object"}):
            return None
        from hymem.core.graph import live_edge_predicate

        rows = conn.execute(
            f"SELECT * FROM knowledge_graph kg WHERE kg.subject_canonical=? "
            "AND kg.predicate=? AND kg.object_canonical=? AND kg.derived=0 AND "
            f"{live_edge_predicate('kg')}",
            (ref["subject"], ref["predicate"], ref["object"]),
        ).fetchall()
        if len(rows) != 1:
            return None
        memo_key = ("knowledge_graph", int(rows[0]["id"]))
        if context is not None and memo_key in context.anchors:
            return context.anchors[memo_key]
        proof = _knowledge_graph_anchor_from_row(
            conn, rows[0],
            _coverage_memo=context.coverage if context is not None else None,
        )
        if context is not None:
            context.anchors[memo_key] = proof
        return proof
    if kind == "narrative_fact":
        if frozenset(ref) != frozenset({"source_outcome_key", "fact_key"}):
            return None
        if not all(isinstance(value, str) and value for value in ref.values()):
            return None
        return narrative_fact_anchor_input(
            conn, str(ref["source_outcome_key"]), str(ref["fact_key"])
        )
    return None


@dataclass(frozen=True)
class _ValidatedAggregationNode:
    row: Mapping[str, object]
    inputs: tuple[AggregationInputProof, ...]
    occurrences: tuple[BoundSourceOccurrence, ...]


@dataclass(frozen=True)
class ValidatedAggregationPublication:
    """The complete current publication after validating every declared row."""

    publication_id: str
    config_version: str
    cluster_min_members: int
    cluster_min_sessions: int
    anchor_fact_cap: int
    root_node_id: str | None
    published_at: str
    generation_key: str
    generation_binding: Mapping[str, object]
    request_contract_sha256: str
    material_epoch_key: str
    material_binding: Mapping[str, object]
    material_revision: int
    node_embedding_count: int
    node_embedding_set_hash: str
    nodes: Mapping[str, _ValidatedAggregationNode]


@dataclass
class _AggregationValidationContext:
    """Exact proof memo scoped to one coherent SQLite snapshot."""

    nodes: dict[str, _ValidatedAggregationNode]
    coverage: dict[tuple[int, str, str], BoundSourceOccurrence]
    episode_manifests: dict[
        str, tuple[BoundSourceOccurrence, ...] | None
    ]
    episode_proofs: dict[
        tuple[str, bool], AggregationInputProof | None
    ]
    generations: dict[str, Mapping[str, object] | None]
    materials: dict[str, Mapping[str, object] | None]
    anchors: dict[tuple[str, int], AggregationInputProof | None]
    profile_sources: dict[
        tuple[int, str], BoundSourceOccurrence | None
    ]

    @classmethod
    def empty(cls) -> "_AggregationValidationContext":
        return cls({}, {}, {}, {}, {}, {}, {}, {})


def _validate_aggregation_node(
    conn: sqlite3.Connection, node_id: str, *, seen: frozenset[str],
    expected_generation_key: str | None = None,
    expected_material_epoch_key: str | None = None,
    context: _AggregationValidationContext | None = None,
) -> _ValidatedAggregationNode | None:
    if context is None:
        context = _AggregationValidationContext.empty()
    if node_id in seen:
        return None
    if node_id in context.nodes:
        return context.nodes[node_id]
    row_obj = conn.execute(
        "SELECT * FROM aggregation_nodes WHERE id=?", (node_id,),
    ).fetchone()
    if row_obj is None:
        return None
    row = dict(row_obj)
    from hymem.dreaming.aggregation_generation import (
        load_registered_aggregation_generation,
    )
    generation_key = row.get("aggregation_generation_key")
    generation_cache_key = str(generation_key)
    if generation_cache_key not in context.generations:
        context.generations[generation_cache_key] = (
            load_registered_aggregation_generation(
                conn, generation_key,
                allow_inexact=expected_generation_key is not None,
            )
        )
    generation = context.generations[generation_cache_key]
    if (
        generation is None
        or (
            expected_generation_key is not None
            and generation_key != expected_generation_key
        )
        or generation["contract"]["material_config_version"]
        != row.get("build_config_version")
    ):
        return None
    from hymem.dreaming.aggregation_material import (
        load_registered_aggregation_material_epoch,
    )
    material_key = row.get("aggregation_material_epoch_key")
    material_cache_key = str(material_key)
    if material_cache_key not in context.materials:
        context.materials[material_cache_key] = (
            load_registered_aggregation_material_epoch(
                conn, material_key,
                allow_inexact=expected_material_epoch_key is not None,
            )
        )
    material = context.materials[material_cache_key]
    if (
        material is None
        or (
            expected_material_epoch_key is not None
            and material_key != expected_material_epoch_key
        )
        or material["config_version"] != row.get("build_config_version")
    ):
        return None
    if (
        row.get("source_manifest_complete") != 1
        or row.get("source_manifest_version") != AGGREGATION_SOURCE_MANIFEST_VERSION
        or not isinstance(row.get("source_manifest_count"), int)
        or isinstance(row.get("source_manifest_count"), bool)
        or int(row["source_manifest_count"]) <= 0
        or row.get("input_manifest_complete") != 1
        or row.get("input_manifest_version") != AGGREGATION_INPUT_MANIFEST_VERSION
        or not isinstance(row.get("input_manifest_count"), int)
        or isinstance(row.get("input_manifest_count"), bool)
        or int(row["input_manifest_count"]) <= 0
        or not _valid_hash(row.get("source_manifest_hash"))
        or not _valid_hash(row.get("input_manifest_hash"))
        or not _valid_hash(row.get("input_fingerprint"))
        or not _valid_hash(row.get("output_hash"))
        or not _valid_hash(row.get("aggregation_request_hash"))
        or not _valid_hash(row.get("publication_id"))
        or not _valid_build_config(row.get("build_config_version"))
        or row.get("node_kind") not in {"cluster", "rollup", "root"}
        or not isinstance(row.get("level"), int)
        or isinstance(row.get("level"), bool)
        or int(row["level"]) < 0
        or row.get("is_root") not in (0, 1)
        or not isinstance(row.get("title"), str)
        or not isinstance(row.get("summary"), str)
        or not aggregation_output_is_canonical(
            row.get("title"), row.get("summary")
        )
        or row["output_hash"] != aggregation_output_hash(
            str(row["node_kind"]), str(row["title"]), str(row["summary"]),
            str(row["aggregation_request_hash"]),
        )
    ):
        return None
    kind = str(row["node_kind"])
    level = int(row["level"])
    if not (
        (kind == "cluster" and level == 0 and row["is_root"] == 0)
        or (kind == "rollup" and level > 0 and row["is_root"] == 0)
        or (kind == "root" and level > 0 and row["is_root"] == 1)
    ):
        return None
    try:
        member_ids = json.loads(row["member_episode_ids"])
        session_ids = json.loads(row["session_ids"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(member_ids, list) or not member_ids
        or any(not isinstance(value, str) or not value for value in member_ids)
        or row["n_members"] != len(member_ids)
        or not isinstance(session_ids, list)
        or any(not isinstance(value, str) or not value for value in session_ids)
        or session_ids != sorted(set(session_ids))
        or row["n_sessions"] != len(session_ids)
        or row["member_episode_ids"] != aggregation_canonical_json(member_ids)
        or row["session_ids"] != aggregation_canonical_json(session_ids)
    ):
        return None
    if kind == "rollup" and len(member_ids) < 2:
        # Producer rollups are actual fusions. A singleton is passed through
        # unchanged and can never truthfully claim rollup output authority.
        return None
    occurrences = _load_bound_rows(
        conn, table="aggregation_node_source_occurrences",
        parent_column="node_id", parent_id=node_id,
        _coverage_memo=context.coverage,
    )
    if (
        occurrences is None
        or len(occurrences) != int(row["source_manifest_count"])
        or row["source_manifest_hash"] != source_manifest_hash(
            AGGREGATION_SOURCE_MANIFEST_VERSION, occurrences
        )
        or sorted({item.session_id for item in occurrences}) != session_ids
    ):
        return None
    stored_inputs = conn.execute(
        "SELECT * FROM aggregation_node_inputs WHERE node_id=? ORDER BY ordinal",
        (node_id,),
    ).fetchall()
    if (
        len(stored_inputs) != int(row["input_manifest_count"])
        or any(item["ordinal"] != index for index, item in enumerate(stored_inputs))
    ):
        return None
    validated_inputs: list[AggregationInputProof] = []
    typed_members: list[str] = []
    typed_member_keys: list[tuple[str, str]] = []
    child_levels: list[int] = []
    anchors_started = False
    next_seen = seen | {node_id}
    for input_row in stored_inputs:
        input_kind = input_row["input_kind"]
        if input_kind not in _AGGREGATION_INPUT_KINDS:
            return None
        expected_keys = {
            "episode": frozenset({"id"}),
            "aggregation_node": frozenset({"id"}),
            "user_profile": frozenset({
                "slot", "slot_key", "value", "source_session_id",
                "source_message_id",
            }),
            "knowledge_graph": frozenset({"subject", "predicate", "object"}),
            "narrative_fact": frozenset({"source_outcome_key", "fact_key"}),
        }[input_kind]
        ref = _strict_ref(input_row["source_ref_json"], expected_keys)
        stored_sources = _load_input_sources(
            conn, node_id, int(input_row["ordinal"]),
            _coverage_memo=context.coverage,
        )
        if ref is None or stored_sources is None or not stored_sources:
            return None
        expected: AggregationInputProof | None
        if input_kind == "episode":
            if anchors_started or not isinstance(ref.get("id"), str):
                return None
            expected = episode_input_proof(
                conn, str(ref["id"]), with_session_prefix=(kind == "cluster"),
                _coverage_memo=context.coverage,
                _manifest_memo=context.episode_manifests,
                _proof_memo=context.episode_proofs,
            )
            typed_members.append(str(ref["id"]))
            typed_member_keys.append(("episode", str(ref["id"])))
        elif input_kind == "aggregation_node":
            if anchors_started or kind == "cluster" or not isinstance(ref.get("id"), str):
                return None
            child = _validate_aggregation_node(
                conn, str(ref["id"]), seen=next_seen,
                expected_generation_key=str(generation_key),
                expected_material_epoch_key=str(material_key),
                context=context,
            )
            if (
                child is None or int(child.row["level"]) >= level
                or child.row["publication_id"] != row["publication_id"]
                or child.row["build_config_version"] != row["build_config_version"]
                or child.row["aggregation_generation_key"] != generation_key
                or child.row["aggregation_material_epoch_key"] != material_key
            ):
                return None
            expected = make_aggregation_input_proof(
                kind="aggregation_node", source_ref={"id": child.row["id"]},
                rendered_text=f"{child.row['title']}\n{child.row['summary']}",
                authority_hash=aggregation_node_authority_hash(child.row),
                occurrences=child.occurrences,
            )
            typed_members.append(str(ref["id"]))
            typed_member_keys.append(("aggregation_node", str(ref["id"])))
            child_levels.append(int(child.row["level"]))
        else:
            anchors_started = True
            if kind != "root":
                return None
            expected = _resolve_anchor_input(
                conn, str(input_kind), ref, context=context,
            )
        if expected is None or expected.occurrences != stored_sources:
            return None
        expected_record = expected.manifest_record(int(input_row["ordinal"]))
        # Phase-1 generations are a typed-proof/material-scope commitment, not
        # a column in ``aggregation_node_inputs``.  They participate in
        # ``proof_hash`` and ``input_manifest_hash`` below; compare here only
        # the canonical projection actually persisted in the typed-input row.
        # Indexing ``sqlite3.Row`` with the proof-only field made every root
        # containing a KG anchor fail validation with ``IndexError``.
        persisted_fields = (
            "ordinal", "input_kind", "source_key", "source_ref_json",
            "payload_hash", "authority_hash", "source_manifest_count",
            "source_manifest_hash",
        )
        if any(
            input_row[key] != expected_record[key]
            for key in persisted_fields
        ):
            return None
        validated_inputs.append(expected)
    if (
        typed_members != member_ids
        or len(typed_members) != int(row["n_members"])
        or len(typed_member_keys) != len(set(typed_member_keys))
    ):
        return None
    if kind == "cluster" and any(
        item.kind != "episode" for item in validated_inputs
    ):
        return None
    if kind != "cluster" and level != max(child_levels, default=0) + 1:
        # Level is structural input, not decorative ranking metadata. A
        # pass-through episode/level-0 child lives at the conceptual base, and
        # each rollup/root is exactly one level above its highest typed child.
        return None
    try:
        combined = combine_source_occurrences(
            item.occurrences for item in validated_inputs
        )
    except ValueError:
        return None
    typed_tuple = tuple(validated_inputs)
    try:
        expected_request_hash = aggregation_llm_request_hash(
            aggregation_llm_request(typed_tuple, node_kind=kind)
        )
    except (TypeError, ValueError):
        return None
    if (
        combined != occurrences
        or row["input_manifest_hash"] != aggregation_input_manifest_hash(typed_tuple)
        or row["input_fingerprint"] != aggregation_typed_input_fingerprint(typed_tuple)
        or row["aggregation_request_hash"] != expected_request_hash
        or row["id"] != aggregation_node_id(
            member_ids, node_kind=kind,
            input_fingerprint=str(row["input_fingerprint"]),
            generation_key=str(generation_key),
            request_hash=str(row["aggregation_request_hash"]),
        )
    ):
        return None
    result = _ValidatedAggregationNode(row, typed_tuple, occurrences)
    context.nodes[node_id] = result
    return result


def load_aggregation_node_proof(
    conn: sqlite3.Connection, node_id: str, *,
    expected_generation_key: str | None = None,
    expected_material_epoch_key: str | None = None,
) -> _ValidatedAggregationNode | None:
    """Central structural validator, evaluated in one coherent DB snapshot."""

    owned_snapshot = not conn.in_transaction
    try:
        if owned_snapshot:
            conn.execute("BEGIN")
        result = _validate_aggregation_node(
            conn, node_id, seen=frozenset(),
            expected_generation_key=expected_generation_key,
            expected_material_epoch_key=expected_material_epoch_key,
            context=_AggregationValidationContext.empty(),
        )
        if owned_snapshot:
            conn.execute("COMMIT")
        return result
    except (RuntimeError, TypeError, ValueError, sqlite3.Error):
        if owned_snapshot and conn.in_transaction:
            conn.execute("ROLLBACK")
        return None


def load_aggregation_source_manifest(
    conn: sqlite3.Connection,
    node_id: str,
    *,
    validate_level0_input: bool = True,
    _seen: frozenset[str] = frozenset(),
) -> tuple[BoundSourceOccurrence, ...] | None:
    """Return exact typed node sources, or ``None`` on any structural defect.

    ``validate_level0_input`` is retained for source compatibility but can no
    longer weaken validation.  Every hierarchy level proves its own typed
    manifest; callers needing material text use :func:`load_aggregation_node_proof`.
    """

    del validate_level0_input, _seen
    proof = load_aggregation_node_proof(conn, node_id)
    return proof.occurrences if proof is not None else None


def load_published_aggregation_root(
    conn: sqlite3.Connection, *, expected_config_version: str | None = None,
    expected_cluster_min_members: int | None = None,
    expected_cluster_min_sessions: int | None = None,
    expected_anchor_fact_cap: int | None = None,
    expected_generation_key: str | None = None,
    embedding_client: object = _EMBEDDING_CLIENT_UNSET,
) -> _ValidatedAggregationNode | None:
    """Load the sole structurally published root, never newest-row guessing."""

    publication = load_current_aggregation_publication(
        conn, expected_config_version=expected_config_version,
        expected_cluster_min_members=expected_cluster_min_members,
        expected_cluster_min_sessions=expected_cluster_min_sessions,
        expected_anchor_fact_cap=expected_anchor_fact_cap,
        expected_generation_key=expected_generation_key,
        embedding_client=embedding_client,
    )
    if publication is None or publication.root_node_id is None:
        return None
    return publication.nodes.get(publication.root_node_id)


def load_current_aggregation_publication(
    conn: sqlite3.Connection, *, expected_config_version: str | None = None,
    expected_cluster_min_members: int | None = None,
    expected_cluster_min_sessions: int | None = None,
    expected_anchor_fact_cap: int | None = None,
    expected_generation_key: str | None = None,
    expected_material_epoch_key: str | None = None,
    embedding_client: object = _EMBEDDING_CLIENT_UNSET,
) -> ValidatedAggregationPublication | None:
    """Validate the one current node set in a coherent SQLite snapshot.

    Physical nodes are history/cache, not read authority.  This validates every
    row named by the singleton before exposing any one of them, so a malformed
    orphan, an unpublished candidate, or a digest-disabled poison row cannot be
    reached through a search/provider path.
    """

    owned_snapshot = not conn.in_transaction
    try:
        if owned_snapshot:
            conn.execute("BEGIN")
        state_rows = conn.execute(
            "SELECT * FROM aggregation_publication_state ORDER BY id"
        ).fetchall()
        if not state_rows:
            result = None
        elif len(state_rows) != 1 or state_rows[0]["id"] != 1:
            result = None
        else:
            state = dict(state_rows[0])
            publication_rows = [dict(row) for row in conn.execute(
                "SELECT * FROM aggregation_nodes WHERE publication_id=? ORDER BY id",
                (state["publication_id"],),
            ).fetchall()]
            root_id = state.get("root_node_id")
            stored_min_members = state.get("cluster_min_members")
            stored_min_sessions = state.get("cluster_min_sessions")
            stored_anchor_cap = state.get("anchor_fact_cap")
            generation_key = state.get("aggregation_generation_key")
            from hymem.dreaming.aggregation_generation import (
                load_registered_aggregation_generation,
            )
            generation = load_registered_aggregation_generation(
                conn, generation_key,
                allow_inexact=expected_generation_key is not None,
            )
            from hymem.dreaming.aggregation_material import (
                aggregation_anchor_phase1_generation_keys,
                aggregation_phase1_scope_identity,
                current_aggregation_material_revision,
                disabled_aggregation_phase1_scope_identity,
                embedding_execution_identity,
                load_registered_aggregation_material_epoch,
                verify_aggregation_material_epoch,
            )
            material_key = state.get("aggregation_material_epoch_key")
            live_embedding_supplied = embedding_client is not _EMBEDDING_CLIENT_UNSET
            material = load_registered_aggregation_material_epoch(
                conn, material_key, allow_inexact=live_embedding_supplied,
            )
            try:
                current_material_revision = current_aggregation_material_revision(conn)
            except RuntimeError:
                current_material_revision = -1
            material_ok = bool(
                material is not None
                and material_key == material.get("material_epoch_key")
                and state.get("material_revision")
                == material.get("material_revision")
                == current_material_revision
                and material.get("config_version") == state.get("config_version")
                and (
                    expected_material_epoch_key is None
                    or material_key == expected_material_epoch_key
                )
            )
            if material_ok and not live_embedding_supplied:
                # Omitting a live vector-space identity is never authority to
                # serve a tree built in some historical producer space.  The
                # sole durable no-client contract is the explicitly disabled
                # producer captured in the material epoch.
                material_ok = (
                    material["embedding_producer"].get("declaration")
                    == {"kind": "disabled"}
                    and material.get("embedding_dimension") is None
                )
            if material_ok and live_embedding_supplied:
                if (
                    material["embedding_producer"].get("declaration")
                    == {"kind": "disabled"}
                    and material.get("embedding_dimension") is None
                ):
                    # The configured client did not participate in this tree:
                    # entity/exact clustering and lexical serving are valid
                    # independently of its current vector space.
                    pass
                else:
                    try:
                        live_binding, _live_model, live_dim = (
                            embedding_execution_identity(embedding_client)
                        )
                    except (RuntimeError, TypeError, ValueError):
                        material_ok = False
                    else:
                        material_ok = bool(
                            live_binding == material["embedding_producer"]
                            and live_dim == material["embedding_dimension"]
                        )
            horizon = material.get("fresh_until") if material is not None else None
            if material_ok and horizon is not None:
                now = conn.execute(
                    "SELECT CURRENT_TIMESTAMP AS now"
                ).fetchone()["now"]
                material_ok = isinstance(now, str) and now < horizon
            request_contract_sha256 = state.get("request_contract_sha256")
            expected_minima = (
                expected_cluster_min_members, expected_cluster_min_sessions,
            )
            if (
                generation is None
                or not material_ok
                or (
                    expected_generation_key is not None
                    and generation_key != expected_generation_key
                )
                or generation["contract"]["material_config_version"]
                != state.get("config_version")
                or request_contract_sha256
                != generation["contract"]["request_policy_sha256"]
            ):
                result = None
            elif (
                expected_config_version is not None
                and (
                    not _valid_build_config(expected_config_version)
                    or state.get("config_version") != expected_config_version
                )
            ):
                result = None
            elif root_id is not None and not isinstance(root_id, str):
                result = None
            elif (
                any(value is None for value in expected_minima)
                and any(value is not None for value in expected_minima)
            ):
                result = None
            elif expected_cluster_min_members is not None and any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in expected_minima
            ):
                result = None
            elif (
                not _valid_hash(state.get("publication_id"))
                or not _valid_hash(state.get("node_set_hash"))
                or not _valid_build_config(state.get("config_version"))
                or not isinstance(state.get("node_count"), int)
                or isinstance(state.get("node_count"), bool)
                or int(state["node_count"]) < 0
                or not isinstance(state.get("material_revision"), int)
                or isinstance(state.get("material_revision"), bool)
                or int(state["material_revision"]) < 0
                or not isinstance(state.get("node_embedding_count"), int)
                or isinstance(state.get("node_embedding_count"), bool)
                or int(state["node_embedding_count"]) < 0
                or not _valid_hash(state.get("node_embedding_set_hash"))
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in (stored_min_members, stored_min_sessions)
                )
                or isinstance(stored_anchor_cap, bool)
                or not isinstance(stored_anchor_cap, int)
                or stored_anchor_cap < 0
                or (
                    expected_anchor_fact_cap is not None
                    and (
                        isinstance(expected_anchor_fact_cap, bool)
                        or not isinstance(expected_anchor_fact_cap, int)
                        or expected_anchor_fact_cap < 0
                        or stored_anchor_cap != expected_anchor_fact_cap
                    )
                )
                or (
                    expected_cluster_min_members is not None
                    and (
                        stored_min_members != expected_cluster_min_members
                        or stored_min_sessions != expected_cluster_min_sessions
                    )
                )
                or not aggregation_publication_timestamp_is_canonical(
                    state.get("published_at")
                )
            ):
                result = None
            else:
                set_hash = aggregation_publication_node_set_hash(publication_rows)
                expected_publication = aggregation_publication_id(
                    config_version=str(state["config_version"]),
                    root_id=root_id,
                    node_count=int(state["node_count"]), node_set_hash=set_hash,
                    published_at=str(state["published_at"]),
                    cluster_min_members=int(stored_min_members),
                    cluster_min_sessions=int(stored_min_sessions),
                    anchor_fact_cap=int(stored_anchor_cap),
                    generation_key=str(generation_key),
                    request_contract_sha256=str(request_contract_sha256),
                    material_epoch_key=str(material_key),
                    material_revision=int(state["material_revision"]),
                    node_embedding_count=int(state["node_embedding_count"]),
                    node_embedding_set_hash=str(state["node_embedding_set_hash"]),
                )
                validated: dict[str, _ValidatedAggregationNode] = {}
                validation_context = _AggregationValidationContext.empty()
                validation_context.generations[str(generation_key)] = generation
                validation_context.materials[str(material_key)] = material
                for raw in publication_rows:
                    proof = _validate_aggregation_node(
                        conn, str(raw["id"]), seen=frozenset(),
                        expected_generation_key=str(generation_key),
                        expected_material_epoch_key=str(material_key),
                        context=validation_context,
                    )
                    if proof is None:
                        validated = {}
                        break
                    validated[str(raw["id"])] = proof
                roots = {
                    node_id for node_id, proof in validated.items()
                    if proof.row["node_kind"] == "root"
                }
                shape_ok = (
                    len(publication_rows) == int(state["node_count"])
                    and len(validated) == len(publication_rows)
                    and state["node_set_hash"] == set_hash
                    and state["publication_id"] == expected_publication
                    and all(
                        proof.row["build_config_version"] == state["config_version"]
                        and proof.row["publication_id"] == state["publication_id"]
                        and proof.row["aggregation_generation_key"] == generation_key
                        and proof.row["aggregation_material_epoch_key"] == material_key
                        for proof in validated.values()
                    )
                    and ((root_id is None and not roots) or roots == {root_id})
                    and (
                        root_id is None
                        or sum(
                            item.kind in _AGGREGATION_ANCHOR_KINDS
                            for item in validated[root_id].inputs
                        ) <= int(stored_anchor_cap)
                    )
                    and all(
                        proof.row["node_kind"] != "cluster"
                        or (
                            len(proof.inputs) >= int(stored_min_members)
                            and len({
                                source.session_id
                                for item in proof.inputs
                                for source in item.occurrences
                            }) >= int(stored_min_sessions)
                        )
                        for proof in validated.values()
                    )
                )
                # With a root, every declared row must be reachable by exact
                # typed child references.  Without a root (digest disabled),
                # only independently useful level-0 clusters may be published.
                if shape_ok and root_id is not None:
                    reachable: set[str] = set()
                    stack = [root_id]
                    while stack:
                        current = stack.pop()
                        if current in reachable or current not in validated:
                            continue
                        reachable.add(current)
                        stack.extend(
                            str(item.source_ref["id"])
                            for item in validated[current].inputs
                            if item.kind == "aggregation_node"
                        )
                    shape_ok = reachable == set(validated)
                elif shape_ok:
                    shape_ok = all(
                        proof.row["node_kind"] == "cluster"
                        for proof in validated.values()
                    )
                if shape_ok:
                    # A publication is a partitioned tree/forest, never a DAG.
                    # Reusing a child node or episode under multiple parents
                    # would amplify prompt material while the flattened source
                    # union silently deduplicated it.
                    node_indegree = {node_id: 0 for node_id in validated}
                    episode_parent: set[str] = set()
                    for proof in validated.values():
                        for item in proof.inputs:
                            source_id = item.source_ref.get("id")
                            if item.kind == "aggregation_node":
                                child_id = str(source_id)
                                if child_id not in node_indegree:
                                    shape_ok = False
                                    break
                                node_indegree[child_id] += 1
                            elif item.kind == "episode":
                                episode_id = str(source_id)
                                if episode_id in episode_parent:
                                    shape_ok = False
                                    break
                                episode_parent.add(episode_id)
                        if not shape_ok:
                            break
                    if shape_ok and root_id is not None:
                        shape_ok = all(
                            degree == (0 if node_id == root_id else 1)
                            for node_id, degree in node_indegree.items()
                        )
                    elif shape_ok:
                        shape_ok = all(degree == 0 for degree in node_indegree.values())
                if shape_ok:
                    from hymem.core.vectors import decode_vector
                    from hymem.dreaming.aggregation_material import (
                        aggregation_anchor_records_sha256,
                    )

                    embedding_rows = [dict(row) for row in conn.execute(
                        "SELECT embedding.* FROM aggregation_node_embeddings embedding "
                        "JOIN aggregation_nodes node ON node.id=embedding.node_id "
                        "WHERE node.publication_id=? ORDER BY embedding.node_id",
                        (state["publication_id"],),
                    ).fetchall()]
                    expected_embedding_ids = {
                        node_id for node_id, proof in validated.items()
                        if proof.row["node_kind"] == "cluster"
                    } if material["node_embedding_policy"] == "required" else set()
                    actual_embedding_ids: set[str] = set()
                    for embedding in embedding_rows:
                        node_id = str(embedding["node_id"])
                        proof = validated.get(node_id)
                        try:
                            decoded = decode_vector(embedding["vector_json"])
                            vector_ok = bool(
                                isinstance(decoded, list)
                                and len(decoded) == material["embedding_dimension"]
                                and all(math.isfinite(float(value)) for value in decoded)
                                and math.sqrt(sum(float(value) ** 2 for value in decoded)) > 0
                            )
                        except (TypeError, ValueError, OverflowError):
                            vector_ok = False
                        if (
                            proof is None
                            or proof.row["node_kind"] != "cluster"
                            or node_id in actual_embedding_ids
                            or embedding["model"]
                            != material["embedding_producer"]["producer_key"]
                            or embedding["embedding_producer_key"] != embedding["model"]
                            or embedding["dim"] != material["embedding_dimension"]
                            or embedding["text_hash"] != embedding_text_hash(
                                f"{proof.row['title']}\n{proof.row['summary']}"
                            )
                            or not vector_ok
                        ):
                            shape_ok = False
                            break
                        actual_embedding_ids.add(node_id)
                    shape_ok = bool(
                        shape_ok
                        and actual_embedding_ids == expected_embedding_ids
                        and len(embedding_rows) == int(state["node_embedding_count"])
                        and aggregation_node_embedding_set_hash(embedding_rows)
                        == state["node_embedding_set_hash"]
                    )
                    if shape_ok:
                        selected_anchors = (
                            load_root_anchor_inputs(
                                conn, int(stored_anchor_cap),
                                _anchor_memo=validation_context.anchors,
                                _coverage_memo=validation_context.coverage,
                                _profile_source_memo=(
                                    validation_context.profile_sources
                                ),
                            )
                            if material["root_anchor_policy"] == "enabled"
                            else []
                        )
                        rooted_anchors = tuple(
                            item for item in (
                                validated[root_id].inputs if root_id is not None else ()
                            ) if item.kind in _AGGREGATION_ANCHOR_KINDS
                        )
                        shape_ok = bool(
                            (root_id is None or tuple(selected_anchors) == rooted_anchors)
                            and len(selected_anchors) == material["anchor_count"]
                            and aggregation_anchor_records_sha256(selected_anchors)
                            == material["anchors_sha256"]
                        )
                        if shape_ok:
                            selected_generation_keys = (
                                aggregation_anchor_phase1_generation_keys(
                                    selected_anchors,
                                )
                            )
                            selected_scope = (
                                aggregation_phase1_scope_identity(
                                    conn,
                                    generation_keys=selected_generation_keys,
                                )[0]
                                if selected_generation_keys
                                else disabled_aggregation_phase1_scope_identity()[0]
                            )
                            shape_ok = bool(
                                selected_scope
                                == material["phase1_scope_sha256"]
                            )
                result = ValidatedAggregationPublication(
                    publication_id=str(state["publication_id"]),
                    config_version=str(state["config_version"]),
                    cluster_min_members=int(stored_min_members),
                    cluster_min_sessions=int(stored_min_sessions),
                    anchor_fact_cap=int(stored_anchor_cap),
                    root_node_id=root_id,
                        published_at=str(state["published_at"]),
                        generation_key=str(generation_key),
                        generation_binding=generation,
                        request_contract_sha256=str(request_contract_sha256),
                    material_epoch_key=str(material_key),
                    material_binding=material,
                    material_revision=int(state["material_revision"]),
                    node_embedding_count=int(state["node_embedding_count"]),
                    node_embedding_set_hash=str(state["node_embedding_set_hash"]),
                    nodes=validated,
                ) if shape_ok else None
        if result is not None:
            final_state = conn.execute(
                "SELECT publication_id,aggregation_material_epoch_key,"
                "material_revision,CURRENT_TIMESTAMP AS now "
                "FROM aggregation_publication_state WHERE id=1"
            ).fetchone()
            fresh_until = result.material_binding.get("fresh_until")
            if (
                final_state is None
                or final_state["publication_id"] != result.publication_id
                or final_state["aggregation_material_epoch_key"]
                != result.material_epoch_key
                or final_state["material_revision"] != result.material_revision
                or current_aggregation_material_revision(conn)
                != result.material_revision
                or (
                    fresh_until is not None
                    and (
                        not isinstance(final_state["now"], str)
                        or final_state["now"] >= fresh_until
                    )
                )
            ):
                result = None
        if owned_snapshot:
            conn.execute("COMMIT")
            if result is not None:
                # Root inputs intentionally do not mutate the episode/vector
                # material clock.  Close the first snapshot, then take one
                # short independent snapshot that reselects the exact capped
                # anchors and consumed Phase-1 scope alongside the singleton.
                # This catches a root write that committed while the larger
                # recursive proof validation was running without duplicating
                # the selection algorithm in SQL triggers.
                conn.execute("BEGIN")
                post = conn.execute(
                    "SELECT publication_id,aggregation_material_epoch_key,"
                    "material_revision,CURRENT_TIMESTAMP AS now "
                    "FROM aggregation_publication_state WHERE id=1"
                ).fetchone()
                post_embedding_client = (
                    embedding_client if live_embedding_supplied else None
                )
                try:
                    verify_aggregation_material_epoch(
                        conn, result.material_binding,
                        embedding_client=post_embedding_client,
                        anchor_cap=result.anchor_fact_cap,
                    )
                except (RuntimeError, TypeError, ValueError, sqlite3.Error):
                    result = None
                else:
                    fresh_until = result.material_binding.get("fresh_until")
                    if (
                        post is None
                        or post["publication_id"] != result.publication_id
                        or post["aggregation_material_epoch_key"]
                        != result.material_epoch_key
                        or post["material_revision"] != result.material_revision
                        or (fresh_until is not None and post["now"] >= fresh_until)
                    ):
                        result = None
                conn.execute("COMMIT")
        return result
    except (RuntimeError, TypeError, ValueError, sqlite3.Error):
        if owned_snapshot and conn.in_transaction:
            conn.execute("ROLLBACK")
        return None


def load_current_aggregation_node_proof(
    conn: sqlite3.Connection, node_id: str, *,
    expected_config_version: str | None = None,
    expected_cluster_min_members: int | None = None,
    expected_cluster_min_sessions: int | None = None,
    expected_anchor_fact_cap: int | None = None,
    expected_generation_key: str | None = None,
    embedding_client: object = _EMBEDDING_CLIENT_UNSET,
) -> _ValidatedAggregationNode | None:
    """Return a node only when it belongs to the wholly valid publication."""

    publication = load_current_aggregation_publication(
        conn, expected_config_version=expected_config_version,
        expected_cluster_min_members=expected_cluster_min_members,
        expected_cluster_min_sessions=expected_cluster_min_sessions,
        expected_anchor_fact_cap=expected_anchor_fact_cap,
        expected_generation_key=expected_generation_key,
        embedding_client=embedding_client,
    )
    return publication.nodes.get(node_id) if publication is not None else None
