"""Canonical encoding and fingerprints for durable raw-message records."""

from __future__ import annotations

import hashlib
import json

from hymem.core.time import normalize_iso_timestamp


MESSAGE_CONTENT_HASH_VERSION = "sha256-role-content-v1"
MESSAGE_RECORD_VERSION = "hymem-message-jsonl-v1"
MESSAGE_PROVENANCE_HASH_VERSION = "sha256-message-provenance-v2"
MESSAGE_PROVENANCE_RECORD_VERSION = "hymem-message-jsonl-v2"


def _external_source_timestamp_is_valid(value: object) -> bool:
    """Whether ``value`` is an attributable source-time coordinate."""
    if not isinstance(value, str):
        return False
    try:
        normalize_iso_timestamp(value, context="external message source")
    except ValueError:
        return False
    return True


def message_content_hash(role: str, content: str) -> str:
    """Fingerprint the exact stored role/content pair."""
    payload = json.dumps(
        {"content": content, "role": role},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode_message_record(*, message_id: int, role: str, content: str) -> str:
    """Encode one source message as an unambiguous canonical JSONL record."""
    return json.dumps(
        {
            "content": content,
            "id": int(message_id),
            "record_version": MESSAGE_RECORD_VERSION,
            "role": role,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def encode_provenance_message_record(
    *,
    message_id: int,
    session_id: str,
    role: str,
    content: str,
    source_created_at: str,
    source_peer_id: str,
    source_workspace_id: str,
) -> str:
    """Encode an external message while binding its exact source identity.

    ``messages.created_at`` is the caller's normalized event time when one was
    supplied, and otherwise the database acceptance time.  Keeping its exact
    stored spelling in the record makes both interpretations tamper evident.
    """
    if not _external_source_timestamp_is_valid(source_created_at):
        raise ValueError(
            "external message source_created_at must be a valid ISO timestamp"
        )
    return json.dumps(
        {
            "content": content,
            "id": int(message_id),
            "record_version": MESSAGE_PROVENANCE_RECORD_VERSION,
            "role": role,
            "session_id": session_id,
            "source_created_at": source_created_at,
            "source_peer_id": source_peer_id,
            "source_workspace_id": source_workspace_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def message_provenance_hash(
    *,
    message_id: int,
    session_id: str,
    role: str,
    content: str,
    source_created_at: str,
    source_peer_id: str,
    source_workspace_id: str,
) -> str:
    """Fingerprint the complete v2 external-source record."""
    record = encode_provenance_message_record(
        message_id=message_id,
        session_id=session_id,
        role=role,
        content=content,
        source_created_at=source_created_at,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    return hashlib.sha256(record.encode("utf-8")).hexdigest()


def canonical_message_record(
    *,
    message_id: int,
    session_id: str,
    role: str,
    content: str,
    source_created_at: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
) -> tuple[str, str, str, str]:
    """Return canonical bytes, hash, hash version, and record version.

    Legacy/native messages deliberately retain the frozen v1 representation.
    An attributed message always uses v2; a partial identity is never encoded.
    """
    if (source_peer_id is None) != (source_workspace_id is None):
        raise ValueError("external message provenance must be a complete pair")
    if source_peer_id is None:
        record = encode_message_record(
            message_id=message_id,
            role=role,
            content=content,
        )
        return (
            record,
            message_content_hash(role, content),
            MESSAGE_CONTENT_HASH_VERSION,
            MESSAGE_RECORD_VERSION,
        )
    if not _external_source_timestamp_is_valid(source_created_at):
        raise ValueError(
            "external message source_created_at must be a valid ISO timestamp"
        )
    record = encode_provenance_message_record(
        message_id=message_id,
        session_id=session_id,
        role=role,
        content=content,
        source_created_at=source_created_at,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    return (
        record,
        hashlib.sha256(record.encode("utf-8")).hexdigest(),
        MESSAGE_PROVENANCE_HASH_VERSION,
        MESSAGE_PROVENANCE_RECORD_VERSION,
    )


def message_record_proof_valid(
    chunk_text: object,
    message_hash: object,
    hash_version: object,
    record_version: object,
) -> int:
    """SQLite-friendly self-consistency check for a v1 or v2 record proof.

    Relational metadata is checked separately by the coverage/evidence guards
    and by ``validate_message_coverage_artifact``.  This helper lets historical
    v40 SQL accept v2 records without referring to columns that did not exist
    when that migration ran.
    """
    if not all(
        isinstance(value, str)
        for value in (chunk_text, message_hash, hash_version, record_version)
    ):
        return 0
    for line in chunk_text.split("\n"):
        try:
            payload = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if (
            hash_version == MESSAGE_CONTENT_HASH_VERSION
            and record_version == MESSAGE_RECORD_VERSION
            and set(payload) == {"content", "id", "record_version", "role"}
            and isinstance(payload.get("id"), int)
            and not isinstance(payload.get("id"), bool)
            and isinstance(payload.get("content"), str)
            and isinstance(payload.get("role"), str)
        ):
            canonical = encode_message_record(
                message_id=payload["id"],
                role=payload["role"],
                content=payload["content"],
            )
            if (
                line == canonical
                and message_hash
                == message_content_hash(payload["role"], payload["content"])
            ):
                return 1
        if (
            hash_version == MESSAGE_PROVENANCE_HASH_VERSION
            and record_version == MESSAGE_PROVENANCE_RECORD_VERSION
            and set(payload) == {
                "content", "id", "record_version", "role", "session_id",
                "source_created_at", "source_peer_id", "source_workspace_id",
            }
            and isinstance(payload.get("id"), int)
            and not isinstance(payload.get("id"), bool)
            and all(
                isinstance(payload.get(field), str)
                for field in (
                    "content", "role", "session_id", "source_peer_id",
                    "source_workspace_id",
                )
            )
            and _external_source_timestamp_is_valid(
                payload.get("source_created_at")
            )
        ):
            canonical = encode_provenance_message_record(
                message_id=payload["id"],
                session_id=payload["session_id"],
                role=payload["role"],
                content=payload["content"],
                source_created_at=payload["source_created_at"],
                source_peer_id=payload["source_peer_id"],
                source_workspace_id=payload["source_workspace_id"],
            )
            if (
                line == canonical
                and message_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            ):
                return 1
    return 0


def message_record_matches_source(
    chunk_text: object,
    message_id: object,
    session_id: object,
    role: object,
    source_created_at: object,
    source_peer_id: object,
    source_workspace_id: object,
    message_hash: object,
    hash_version: object,
    record_version: object,
) -> int:
    """Whether a proof binds the supplied relational source tuple exactly."""
    if (
        not isinstance(chunk_text, str)
        or not isinstance(message_id, int)
        or isinstance(message_id, bool)
        or not isinstance(session_id, str)
        or not isinstance(role, str)
        or (source_peer_id is not None and not isinstance(source_peer_id, str))
        or (
            source_workspace_id is not None
            and not isinstance(source_workspace_id, str)
        )
    ):
        return 0
    if source_peer_id is not None and not _external_source_timestamp_is_valid(
        source_created_at
    ):
        return 0
    for line in chunk_text.split("\n"):
        try:
            payload = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(
            payload.get("content"), str
        ):
            continue
        try:
            canonical, expected_hash, expected_hash_version, expected_record_version = (
                canonical_message_record(
                    message_id=message_id,
                    session_id=session_id,
                    role=role,
                    content=payload["content"],
                    source_created_at=source_created_at,
                    source_peer_id=source_peer_id,
                    source_workspace_id=source_workspace_id,
                )
            )
        except (TypeError, ValueError):
            return 0
        if (
            line == canonical
            and message_hash == expected_hash
            and hash_version == expected_hash_version
            and record_version == expected_record_version
        ):
            return 1
    return 0


def message_record_matches_raw_source(
    chunk_text: object,
    message_id: object,
    session_id: object,
    role: object,
    content: object,
    source_created_at: object,
    source_peer_id: object,
    source_workspace_id: object,
    message_hash: object,
    hash_version: object,
    record_version: object,
) -> int:
    """Whether a proof binds an exact raw-message tuple.

    Unlike :func:`message_record_matches_source`, this variant receives the
    authoritative raw content instead of discovering content inside the
    artifact.  Coverage write guards use it when the raw row still exists so
    a self-consistent replacement artifact cannot disagree with that row.
    """
    if (
        not isinstance(chunk_text, str)
        or not isinstance(message_id, int)
        or isinstance(message_id, bool)
        or not isinstance(session_id, str)
        or not isinstance(role, str)
        or not isinstance(content, str)
        or (source_peer_id is not None and not isinstance(source_peer_id, str))
        or (
            source_workspace_id is not None
            and not isinstance(source_workspace_id, str)
        )
    ):
        return 0
    if source_peer_id is not None and not _external_source_timestamp_is_valid(
        source_created_at
    ):
        return 0
    try:
        canonical, expected_hash, expected_hash_version, expected_record_version = (
            canonical_message_record(
                message_id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                source_created_at=source_created_at,
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
        )
    except (TypeError, ValueError):
        return 0
    return int(
        canonical in chunk_text.split("\n")
        and message_hash == expected_hash
        and hash_version == expected_hash_version
        and record_version == expected_record_version
    )


def chunk_contains_message_record(*, chunk_text: str, record: str) -> bool:
    """Whether ``record`` is present as one exact physical JSONL line."""
    # JSONL is delimited by ASCII LF. ``str.splitlines()`` also splits valid
    # JSON string characters such as U+0085, U+2028, and U+2029, which would
    # make an otherwise exact Unicode message impossible to prove/read.
    return any(line == record for line in chunk_text.split("\n"))
