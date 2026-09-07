"""Producer-bound publication and validation for Phase-1 auxiliaries.

Claims have their own append-only evidence ledger.  Entity hints and behavioral
markers are a separate whole-response projection: an omitted item in a later
successful response is meaningful, while a failed response must change
nothing.  This module gives that projection one canonical hash and exact
chunk/producer lineage, then exposes small helpers shared by live persistence,
startup validation, and portability.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sqlite3
import textwrap
from collections.abc import Mapping, Sequence

from hymem.dreaming import canonicalize
from hymem.extraction.markers import Marker

AUXILIARY_CONTRACT_V0 = "phase1-auxiliary-contract-v0"
AUXILIARY_CONTRACT_V1 = "phase1-auxiliary-contract-v1"
AUXILIARY_CONTRACT_V2 = "phase1-auxiliary-contract-v2"
CURRENT_AUXILIARY_CONTRACT_KEY = AUXILIARY_CONTRACT_V2
# Backward-compatible import name. Current-policy code below deliberately uses
# the explicit alias so future policies retain V0/V1/V2's immutable dispatch.
AUXILIARY_CONTRACT_KEY = CURRENT_AUXILIARY_CONTRACT_KEY
# This is deliberately pinned rather than silently recomputed into the
# contract key.  Any edit to the v2 canonical projection/framing/publication
# must fail the sentinel and be released as a new contract key while retaining
# the immutable v0/v1 validator for durable history.
AUXILIARY_POLICY_SHA256 = (
    "sha256:874f92901cb335d36209ee5bb8b5b29f5c0c02cdf1d2bfca03ae436e6f01a8a2"
)


def _hash_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _hash_payload_v0_v1(payload: Mapping[str, object]) -> str:
    """Frozen JSON/hash framing for durable auxiliary contracts v0 and v1."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _canonical_auxiliary_result_v0(
    *,
    chunk_id: str,
    phase1_generation_key: str,
    extraction_cache_key: str,
    entity_types: Sequence[tuple[str, str, float]],
    entity_properties: Sequence[tuple[str, str, str]],
    entity_mentions: Sequence[str],
    markers: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Immutable canonicalizer for supported historical contract v0.

    V0 predates producer-bound mention observations. Keep its byte framing
    independent from the current helper: a new contract field must never
    reinterpret already-published history. ``entity_mentions`` is accepted
    only to keep dispatch call sites uniform and is intentionally ignored.
    """

    del entity_mentions
    normalized_types = sorted({
        (str(entity), str(type_name), float(confidence))
        for entity, type_name, confidence in entity_types
    })
    normalized_properties = sorted({
        (str(entity), str(key), str(value))
        for entity, key, value in entity_properties
    })
    normalized_markers = sorted({
        (str(kind), str(statement)) for kind, statement in markers
    })
    if any(not math.isfinite(item[2]) for item in normalized_types):
        raise ValueError("entity type confidence must be finite")
    payload: dict[str, object] = {
        "chunk_id": str(chunk_id),
        "entity_properties": [list(item) for item in normalized_properties],
        "entity_types": [list(item) for item in normalized_types],
        "extraction_cache_key": str(extraction_cache_key),
        "auxiliary_contract_key": AUXILIARY_CONTRACT_V0,
        "markers": [list(item) for item in normalized_markers],
        "phase1_generation_key": str(phase1_generation_key),
        "version": "phase1-auxiliary-result-v0",
    }
    return {
        "payload": payload,
        "result_hash": _hash_payload_v0_v1(payload),
        "entity_type_count": len(normalized_types),
        "entity_property_count": len(normalized_properties),
        "entity_mention_count": 0,
        "marker_count": len(normalized_markers),
    }


def _canonical_auxiliary_result_v1(
    *,
    chunk_id: str,
    phase1_generation_key: str,
    extraction_cache_key: str,
    entity_types: Sequence[tuple[str, str, float]],
    entity_properties: Sequence[tuple[str, str, str]],
    entity_mentions: Sequence[str],
    markers: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Frozen rowid/time-independent canonicalizer for contract v1."""

    normalized_types = sorted({
        (str(entity), str(type_name), float(confidence))
        for entity, type_name, confidence in entity_types
    })
    normalized_properties = sorted({
        (str(entity), str(key), str(value))
        for entity, key, value in entity_properties
    })
    normalized_markers = sorted({
        (str(kind), str(statement)) for kind, statement in markers
    })
    normalized_mentions = sorted({str(entity) for entity in entity_mentions})
    if any(not math.isfinite(item[2]) for item in normalized_types):
        raise ValueError("entity type confidence must be finite")
    payload: dict[str, object] = {
        "chunk_id": str(chunk_id),
        "entity_properties": [list(item) for item in normalized_properties],
        "entity_types": [list(item) for item in normalized_types],
        "entity_mentions": normalized_mentions,
        "extraction_cache_key": str(extraction_cache_key),
        "auxiliary_contract_key": AUXILIARY_CONTRACT_V1,
        "markers": [list(item) for item in normalized_markers],
        "phase1_generation_key": str(phase1_generation_key),
        "version": "phase1-auxiliary-result-v1",
    }
    return {
        "payload": payload,
        "result_hash": _hash_payload_v0_v1(payload),
        "entity_type_count": len(normalized_types),
        "entity_property_count": len(normalized_properties),
        "entity_mention_count": len(normalized_mentions),
        "marker_count": len(normalized_markers),
    }


def _canonical_auxiliary_result_v2(
    *,
    chunk_id: str,
    phase1_generation_key: str,
    extraction_cache_key: str,
    entity_types: Sequence[tuple[str, str, float]],
    entity_properties: Sequence[tuple[str, str, str]],
    entity_mentions: Sequence[str],
    markers: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Current auxiliary framing with Unicode-safe canonical identities."""

    normalized_types = sorted({
        (str(entity), str(type_name), float(confidence))
        for entity, type_name, confidence in entity_types
    })
    normalized_properties = sorted({
        (str(entity), str(key), str(value))
        for entity, key, value in entity_properties
    })
    normalized_markers = sorted({
        (str(kind), str(statement)) for kind, statement in markers
    })
    normalized_mentions = sorted({str(entity) for entity in entity_mentions})
    if any(not math.isfinite(item[2]) for item in normalized_types):
        raise ValueError("entity type confidence must be finite")
    payload: dict[str, object] = {
        "chunk_id": str(chunk_id),
        "entity_properties": [list(item) for item in normalized_properties],
        "entity_types": [list(item) for item in normalized_types],
        "entity_mentions": normalized_mentions,
        "extraction_cache_key": str(extraction_cache_key),
        "auxiliary_contract_key": AUXILIARY_CONTRACT_V2,
        "markers": [list(item) for item in normalized_markers],
        "phase1_generation_key": str(phase1_generation_key),
        "version": "phase1-auxiliary-result-v2",
    }
    return {
        "payload": payload,
        "result_hash": _hash_payload(payload),
        "entity_type_count": len(normalized_types),
        "entity_property_count": len(normalized_properties),
        "entity_mention_count": len(normalized_mentions),
        "marker_count": len(normalized_markers),
    }


_AUXILIARY_CANONICALIZERS = {
    AUXILIARY_CONTRACT_V0: _canonical_auxiliary_result_v0,
    AUXILIARY_CONTRACT_V1: _canonical_auxiliary_result_v1,
    AUXILIARY_CONTRACT_V2: _canonical_auxiliary_result_v2,
}
SUPPORTED_AUXILIARY_CONTRACT_KEYS = frozenset(_AUXILIARY_CANONICALIZERS)


def canonical_auxiliary_result(
    *,
    chunk_id: str,
    phase1_generation_key: str,
    extraction_cache_key: str,
    auxiliary_contract_key: str = CURRENT_AUXILIARY_CONTRACT_KEY,
    entity_types: Sequence[tuple[str, str, float]],
    entity_properties: Sequence[tuple[str, str, str]],
    entity_mentions: Sequence[str],
    markers: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Dispatch to the immutable canonicalizer recorded by the outcome."""

    canonicalizer = _AUXILIARY_CANONICALIZERS.get(auxiliary_contract_key)
    if canonicalizer is None:
        raise ValueError("unsupported Phase-1 auxiliary contract")
    return canonicalizer(
        chunk_id=chunk_id,
        phase1_generation_key=phase1_generation_key,
        extraction_cache_key=extraction_cache_key,
        entity_types=entity_types,
        entity_properties=entity_properties,
        entity_mentions=entity_mentions,
        markers=markers,
    )


def _canonical_projection(
    conn: sqlite3.Connection,
    entity_type_hints: Mapping[str, str],
    entity_property_hints: Mapping[str, Mapping[str, str]],
    markers: Sequence[Marker],
) -> tuple[
    list[tuple[str, str, float]],
    list[tuple[str, str, str]],
    list[tuple[str, str]],
]:
    # Multiple surfaces can resolve to one canonical within the same response.
    # Multi-label types are meaningful and retained.  Two distinct values for
    # the same canonical property are not orderable evidence, though: omit that
    # optional property rather than making dict/alias iteration pick a winner.
    types = sorted({
        (canonicalize.resolve(conn, entity), str(type_name), 1.0)
        for entity, type_name in entity_type_hints.items()
    })
    property_groups: dict[tuple[str, str], set[str]] = {}
    for entity, values in entity_property_hints.items():
        canonical = canonicalize.resolve(conn, entity)
        for key, value in values.items():
            property_groups.setdefault((canonical, str(key)), set()).add(
                str(value)
            )
    properties = sorted(
        (entity, key, next(iter(values)))
        for (entity, key), values in property_groups.items()
        if len(values) == 1
    )
    marker_rows = sorted({(marker.kind, marker.statement) for marker in markers})
    return types, properties, marker_rows


def _publish_phase1_auxiliaries(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    phase1_generation_key: str,
    extraction_cache_key: str,
    entity_type_hints: Mapping[str, str],
    entity_property_hints: Mapping[str, Mapping[str, str]],
    entity_mentions: Sequence[str],
    markers: Sequence[Marker],
) -> None:
    """Set-reconcile one successful chunk/generation auxiliary projection.

    The caller owns the surrounding transaction and must already have
    published the matching claim outcome.  Other producer generations remain
    immutable history; only this generation's set is replaced.  Thus a failed
    producer switch leaves A physically intact, while B's successful omission
    is represented by an authenticated empty set.
    """

    if not phase1_generation_key:
        raise ValueError("Phase-1 auxiliary publication requires a generation")
    generation = conn.execute(
        "SELECT extraction_cache_key FROM phase1_generations "
        "WHERE generation_key=?",
        (phase1_generation_key,),
    ).fetchone()
    claim = conn.execute(
        "SELECT prompt_version,phase1_generation_key "
        "FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()
    if (
        generation is None
        or generation["extraction_cache_key"] != extraction_cache_key
        or claim is None
        or claim["prompt_version"] != extraction_cache_key
        or claim["phase1_generation_key"] != phase1_generation_key
    ):
        raise ValueError("auxiliary publication does not match claim authority")

    types, properties, marker_rows = _canonical_projection(
        conn, entity_type_hints, entity_property_hints, markers
    )

    # Process-instance identities cannot be trusted after restart and must not
    # become an unbounded durable registry merely because exact histories are
    # preserved.  Once this chunk has a newer successful publication, purge
    # only superseded inexact auxiliary history; durable exact producers keep
    # their replay/audit rows.
    retired_inexact = [
        str(row["phase1_generation_key"])
        for row in conn.execute(
            "SELECT DISTINCT auxiliary.phase1_generation_key "
            "FROM phase1_auxiliary_outcomes auxiliary "
            "JOIN phase1_generations generation "
            "ON generation.generation_key=auxiliary.phase1_generation_key "
            "WHERE auxiliary.chunk_id=? "
            "AND auxiliary.phase1_generation_key<>? "
            "AND generation.identity_exact=0",
            (chunk_id, phase1_generation_key),
        ).fetchall()
    ]
    for retired_key in retired_inexact:
        conn.execute(
            "DELETE FROM behavioral_markers WHERE chunk_id=? "
            "AND phase1_generation_key=?",
            (chunk_id, retired_key),
        )
        conn.execute(
            "DELETE FROM entity_type_observations WHERE chunk_id=? "
            "AND phase1_generation_key=?",
            (chunk_id, retired_key),
        )
        conn.execute(
            "DELETE FROM entity_property_observations WHERE chunk_id=? "
            "AND phase1_generation_key=?",
            (chunk_id, retired_key),
        )
        conn.execute(
            "DELETE FROM entity_mention_observations WHERE chunk_id=? "
            "AND phase1_generation_key=?",
            (chunk_id, retired_key),
        )
        conn.execute(
            "DELETE FROM phase1_auxiliary_outcomes WHERE chunk_id=? "
            "AND phase1_generation_key=?",
            (chunk_id, retired_key),
        )

    existing_types = sorted(
        (
            str(row["entity_canonical"]),
            str(row["type"]),
            float(row["confidence"]),
        )
        for row in conn.execute(
            "SELECT entity_canonical,type,confidence "
            "FROM entity_type_observations WHERE chunk_id=? "
            "AND phase1_generation_key=? ORDER BY entity_canonical,type",
            (chunk_id, phase1_generation_key),
        ).fetchall()
    )
    if existing_types != types:
        conn.execute(
            "DELETE FROM entity_type_observations "
            "WHERE chunk_id=? AND phase1_generation_key=?",
            (chunk_id, phase1_generation_key),
        )
        conn.executemany(
            "INSERT INTO entity_type_observations("
            "chunk_id,entity_canonical,type,confidence,phase1_generation_key) "
            "VALUES (?,?,?,?,?)",
            [
                (chunk_id, entity, type_name, confidence, phase1_generation_key)
                for entity, type_name, confidence in types
            ],
        )
    # Keep the pre-v54 tables as explicitly non-authoritative compatibility
    # history for tooling that still inspects them.  Never overwrite a manual
    # row: current readers use the ledgers/views below, not this lossy cache.
    conn.executemany(
        "INSERT INTO entity_types("
        "entity_canonical,type,confidence,source_chunk_id,origin) "
        "VALUES (?,?,?,?,'legacy_unattributed') "
        "ON CONFLICT(entity_canonical,type) DO UPDATE SET "
        "confidence=excluded.confidence,"
        "source_chunk_id=excluded.source_chunk_id "
        "WHERE entity_types.origin<>'user' AND ("
        "entity_types.confidence IS NOT excluded.confidence OR "
        "entity_types.source_chunk_id IS NOT excluded.source_chunk_id)",
        [
            (entity, type_name, confidence, chunk_id)
            for entity, type_name, confidence in types
        ],
    )

    normalized_mentions = sorted({
        canonicalize.resolve(conn, entity) for entity in entity_mentions
    })
    existing_mentions = [
        str(row["entity_canonical"])
        for row in conn.execute(
            "SELECT entity_canonical FROM entity_mention_observations "
            "WHERE chunk_id=? AND phase1_generation_key=? "
            "ORDER BY entity_canonical",
            (chunk_id, phase1_generation_key),
        ).fetchall()
    ]
    if existing_mentions != normalized_mentions:
        conn.execute(
            "DELETE FROM entity_mention_observations WHERE chunk_id=? "
            "AND phase1_generation_key=?",
            (chunk_id, phase1_generation_key),
        )
        conn.executemany(
            "INSERT INTO entity_mention_observations("
            "chunk_id,entity_canonical,phase1_generation_key) VALUES (?,?,?)",
            [
                (chunk_id, entity, phase1_generation_key)
                for entity in normalized_mentions
            ],
        )
    existing_properties = sorted(
        (str(row["entity_canonical"]), str(row["key"]), str(row["value"]))
        for row in conn.execute(
            "SELECT entity_canonical,key,value "
            "FROM entity_property_observations WHERE chunk_id=? "
            "AND phase1_generation_key=? ORDER BY entity_canonical,key",
            (chunk_id, phase1_generation_key),
        ).fetchall()
    )
    if existing_properties != properties:
        conn.execute(
            "DELETE FROM entity_property_observations "
            "WHERE chunk_id=? AND phase1_generation_key=?",
            (chunk_id, phase1_generation_key),
        )
        conn.executemany(
            "INSERT INTO entity_property_observations("
            "chunk_id,entity_canonical,key,value,phase1_generation_key) "
            "VALUES (?,?,?,?,?)",
            [
                (chunk_id, entity, key, value, phase1_generation_key)
                for entity, key, value in properties
            ],
        )
    conn.executemany(
        "INSERT INTO entity_properties("
        "entity_canonical,key,value,source_chunk_id,origin,updated_at) "
        "VALUES (?,?,?,?,'legacy_unattributed',CURRENT_TIMESTAMP) "
        "ON CONFLICT(entity_canonical,key) DO UPDATE SET "
        "value=excluded.value,source_chunk_id=excluded.source_chunk_id,"
        "updated_at=CURRENT_TIMESTAMP "
        "WHERE entity_properties.origin<>'user' AND ("
        "entity_properties.value IS NOT excluded.value OR "
        "entity_properties.source_chunk_id IS NOT excluded.source_chunk_id)",
        [
            (entity, key, value, chunk_id)
            for entity, key, value in properties
        ],
    )

    desired_markers = set(marker_rows)
    existing = conn.execute(
        "SELECT id,kind,statement FROM behavioral_markers "
        "WHERE chunk_id=? AND phase1_generation_key=? ORDER BY id",
        (chunk_id, phase1_generation_key),
    ).fetchall()
    retained: dict[tuple[str, str], int] = {}
    for row in existing:
        identity = (str(row["kind"]), str(row["statement"]))
        if identity not in desired_markers or identity in retained:
            conn.execute("DELETE FROM behavioral_markers WHERE id=?", (row["id"],))
        else:
            retained[identity] = int(row["id"])
    for kind, statement in marker_rows:
        if (kind, statement) not in retained:
            cursor = conn.execute(
                "INSERT INTO behavioral_markers("
                "kind,statement,chunk_id,phase1_generation_key) "
                "VALUES (?,?,?,?)",
                (kind, statement, chunk_id, phase1_generation_key),
            )
            retained[(kind, statement)] = int(cursor.lastrowid)

    # A retained marker can predate the v54 association ledger, or its derived
    # profile row may have been removed by old cap logic.  Make that gap
    # replayable instead of trusting consolidated_at by itself.
    for marker_id in retained.values():
        has_profile = conn.execute(
            "SELECT 1 FROM profile_marker_decisions WHERE marker_id=?",
            (marker_id,),
        ).fetchone()
        if has_profile is None:
            conn.execute(
                "UPDATE behavioral_markers SET consolidated_at=NULL WHERE id=?",
                (marker_id,),
            )

    canonical = canonical_auxiliary_result(
        chunk_id=chunk_id,
        phase1_generation_key=phase1_generation_key,
        extraction_cache_key=extraction_cache_key,
        auxiliary_contract_key=CURRENT_AUXILIARY_CONTRACT_KEY,
        entity_types=types,
        entity_properties=properties,
        entity_mentions=normalized_mentions,
        markers=marker_rows,
    )
    conn.execute(
        "INSERT INTO phase1_auxiliary_outcomes("
        "chunk_id,phase1_generation_key,extraction_cache_key,"
        "auxiliary_contract_key,result_hash,"
        "entity_type_count,entity_property_count,entity_mention_count,"
        "marker_count) VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(chunk_id,phase1_generation_key) DO UPDATE SET "
        "extraction_cache_key=excluded.extraction_cache_key,"
        "auxiliary_contract_key=excluded.auxiliary_contract_key,"
        "result_hash=excluded.result_hash,"
        "entity_type_count=excluded.entity_type_count,"
        "entity_property_count=excluded.entity_property_count,"
        "entity_mention_count=excluded.entity_mention_count,"
        "marker_count=excluded.marker_count,"
        "published_at=CASE WHEN "
        "phase1_auxiliary_outcomes.result_hash=excluded.result_hash "
        "THEN phase1_auxiliary_outcomes.published_at ELSE CURRENT_TIMESTAMP END",
        (
            chunk_id,
            phase1_generation_key,
            extraction_cache_key,
            CURRENT_AUXILIARY_CONTRACT_KEY,
            canonical["result_hash"],
            canonical["entity_type_count"],
            canonical["entity_property_count"],
            canonical["entity_mention_count"],
            canonical["marker_count"],
        ),
    )


_AUXILIARY_POLICY_FUNCTIONS = (
    _hash_payload,
    _hash_payload_v0_v1,
    canonicalize.normalize,
    canonicalize.resolve,
    _canonical_projection,
    _canonical_auxiliary_result_v2,
    _publish_phase1_auxiliaries,
)
_AUXILIARY_CANONICAL_FUNCTIONS = (
    canonicalize.normalize,
    canonicalize.resolve,
)
_AUXILIARY_CANONICAL_REGEXES = tuple(
    (pattern.pattern, int(pattern.flags))
    for pattern in (
        canonicalize._LEADING_ARTICLES,
        canonicalize._TRAILING_PAREN,
        canonicalize._NON_ALNUM,
        canonicalize._NON_UNICODE_WORD,
    )
)
_AUXILIARY_CANONICAL_SETTINGS = (
    canonicalize._MAX_UNICODE_CANONICAL_CHARS,
)
_AUXILIARY_POLICY_IMPORT_SHA256 = _hash_payload({
    "callables": [
        textwrap.dedent(inspect.getsource(function))
        .replace("\r\n", "\n").replace("\r", "\n").strip()
        for function in _AUXILIARY_POLICY_FUNCTIONS
    ],
    "canonical_regexes": [list(item) for item in _AUXILIARY_CANONICAL_REGEXES],
    "canonical_settings": list(_AUXILIARY_CANONICAL_SETTINGS),
    "dependency_names": [
        "_hash_payload", "_hash_payload_v0_v1",
        "canonicalize.normalize", "canonicalize.resolve",
        "_canonical_projection",
        "_canonical_auxiliary_result_v2", "_publish_phase1_auxiliaries",
    ],
})


def auxiliary_policy_sha256() -> str:
    """Return a runtime-version-independent current policy identity.

    The producer registry's AST hash is ideal for a same-runtime LLM client,
    but Python adds fields to its AST model across supported interpreter
    releases.  A durable on-disk contract cannot vary for identical source, so
    this sentinel hashes normalized source text plus the transitive regex
    constants used by canonicalization.
    """

    dependencies = (
        _hash_payload,
        _hash_payload_v0_v1,
        canonicalize.normalize,
        canonicalize.resolve,
        _canonical_projection,
        _canonical_auxiliary_result_v2,
        _publish_phase1_auxiliaries,
    )
    canonical_functions = (canonicalize.normalize, canonicalize.resolve)
    canonical_regexes = tuple(
        (pattern.pattern, int(pattern.flags))
        for pattern in (
            canonicalize._LEADING_ARTICLES,
            canonicalize._TRAILING_PAREN,
            canonicalize._NON_ALNUM,
            canonicalize._NON_UNICODE_WORD,
        )
    )
    canonical_settings = (canonicalize._MAX_UNICODE_CANONICAL_CHARS,)
    if (
        dependencies != _AUXILIARY_POLICY_FUNCTIONS
        or canonical_functions != _AUXILIARY_CANONICAL_FUNCTIONS
        or canonical_regexes != _AUXILIARY_CANONICAL_REGEXES
        or canonical_settings != _AUXILIARY_CANONICAL_SETTINGS
    ):
        return _hash_payload({"integrity": "runtime-drift"})
    return _AUXILIARY_POLICY_IMPORT_SHA256


def validate_auxiliary_contract_implementation() -> None:
    """Fail closed if current v2 policy changed without contract evolution."""

    actual = auxiliary_policy_sha256()
    if actual != AUXILIARY_POLICY_SHA256:
        raise RuntimeError(
            "Phase-1 auxiliary v2 implementation changed without a new "
            f"contract key (expected {AUXILIARY_POLICY_SHA256}, got {actual})"
        )


def publish_phase1_auxiliaries(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    phase1_generation_key: str,
    extraction_cache_key: str,
    entity_type_hints: Mapping[str, str],
    entity_property_hints: Mapping[str, Mapping[str, str]],
    entity_mentions: Sequence[str],
    markers: Sequence[Marker],
) -> None:
    """Publish through the same lexical authority as the claim ledger."""

    from hymem.core.db import evidence_mutation

    validate_auxiliary_contract_implementation()
    with evidence_mutation(conn):
        _publish_phase1_auxiliaries(
            conn,
            chunk_id=chunk_id,
            phase1_generation_key=phase1_generation_key,
            extraction_cache_key=extraction_cache_key,
            entity_type_hints=entity_type_hints,
            entity_property_hints=entity_property_hints,
            entity_mentions=entity_mentions,
            markers=markers,
        )


def validate_phase1_auxiliary_outcome(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    phase1_generation_key: str,
) -> bool:
    """Validate one durable outcome against its complete child projection."""

    outcome = conn.execute(
        "SELECT * FROM phase1_auxiliary_outcomes "
        "WHERE chunk_id=? AND phase1_generation_key=?",
        (chunk_id, phase1_generation_key),
    ).fetchone()
    if outcome is None:
        return False
    generation = conn.execute(
        "SELECT extraction_cache_key FROM phase1_generations "
        "WHERE generation_key=?",
        (phase1_generation_key,),
    ).fetchone()
    if (
        generation is None
        or generation["extraction_cache_key"] != outcome["extraction_cache_key"]
        or outcome["auxiliary_contract_key"]
        not in SUPPORTED_AUXILIARY_CONTRACT_KEYS
    ):
        return False
    raw_types = conn.execute(
            "SELECT entity_canonical,type,confidence "
            "FROM entity_type_observations WHERE chunk_id=? "
            "AND phase1_generation_key=? ORDER BY entity_canonical,type",
            (chunk_id, phase1_generation_key),
        ).fetchall()
    if any(
        not isinstance(row["entity_canonical"], str)
        or not row["entity_canonical"].strip()
        or canonicalize.normalize(row["entity_canonical"])
        != row["entity_canonical"]
        or not isinstance(row["type"], str)
        or not row["type"].strip()
        or isinstance(row["confidence"], bool)
        or not isinstance(row["confidence"], (int, float))
        or not math.isfinite(float(row["confidence"]))
        or not 0.0 <= float(row["confidence"]) <= 1.0
        for row in raw_types
    ):
        return False
    types = [
        (row["entity_canonical"], row["type"], float(row["confidence"]))
        for row in raw_types
    ]
    raw_properties = conn.execute(
            "SELECT entity_canonical,key,value "
            "FROM entity_property_observations WHERE chunk_id=? "
            "AND phase1_generation_key=? ORDER BY entity_canonical,key",
            (chunk_id, phase1_generation_key),
        ).fetchall()
    if any(
        not isinstance(row["entity_canonical"], str)
        or not row["entity_canonical"].strip()
        or canonicalize.normalize(row["entity_canonical"])
        != row["entity_canonical"]
        or not isinstance(row["key"], str)
        or not row["key"].strip()
        or not isinstance(row["value"], str)
        for row in raw_properties
    ):
        return False
    properties = [
        (row["entity_canonical"], row["key"], row["value"])
        for row in raw_properties
    ]
    raw_mentions = conn.execute(
            "SELECT entity_canonical FROM entity_mention_observations "
            "WHERE chunk_id=? AND phase1_generation_key=? "
            "ORDER BY entity_canonical",
            (chunk_id, phase1_generation_key),
        ).fetchall()
    if any(
        not isinstance(row["entity_canonical"], str)
        or not row["entity_canonical"].strip()
        or canonicalize.normalize(row["entity_canonical"])
        != row["entity_canonical"]
        for row in raw_mentions
    ):
        return False
    mentions = [str(row["entity_canonical"]) for row in raw_mentions]
    if (
        outcome["auxiliary_contract_key"] == AUXILIARY_CONTRACT_V0
        and mentions
    ):
        # The historical contract had no mention projection. A row here is
        # therefore orphaned semantic data even though its FK identity exists.
        return False
    raw_markers = conn.execute(
            "SELECT kind,statement FROM behavioral_markers WHERE chunk_id=? "
            "AND phase1_generation_key=? ORDER BY kind,statement",
            (chunk_id, phase1_generation_key),
        ).fetchall()
    if any(
        row["kind"] not in {"correction", "preference", "rejection", "style"}
        or not isinstance(row["statement"], str)
        or not row["statement"].strip()
        for row in raw_markers
    ):
        return False
    markers = [(row["kind"], row["statement"]) for row in raw_markers]
    canonical = canonical_auxiliary_result(
        chunk_id=chunk_id,
        phase1_generation_key=phase1_generation_key,
        extraction_cache_key=outcome["extraction_cache_key"],
        auxiliary_contract_key=str(outcome["auxiliary_contract_key"]),
        entity_types=types,
        entity_properties=properties,
        entity_mentions=mentions,
        markers=markers,
    )
    return bool(
        outcome["result_hash"] == canonical["result_hash"]
        and int(outcome["entity_type_count"]) == canonical["entity_type_count"]
        and int(outcome["entity_property_count"])
        == canonical["entity_property_count"]
        and int(outcome["entity_mention_count"])
        == canonical["entity_mention_count"]
        and int(outcome["marker_count"]) == canonical["marker_count"]
    )


def refresh_phase1_auxiliary_outcomes(
    conn: sqlite3.Connection,
    identities: Sequence[tuple[str, str]],
) -> None:
    """Rehash publications after an authorized canonical entity merge."""

    from hymem.core.db import evidence_mutation

    with evidence_mutation(conn):
        for chunk_id, generation_key in sorted(set(identities)):
            outcome = conn.execute(
                "SELECT extraction_cache_key,auxiliary_contract_key "
                "FROM phase1_auxiliary_outcomes "
                "WHERE chunk_id=? AND phase1_generation_key=?",
                (chunk_id, generation_key),
            ).fetchone()
            if outcome is None:
                continue
            types = [
                (row["entity_canonical"], row["type"], float(row["confidence"]))
                for row in conn.execute(
                    "SELECT entity_canonical,type,confidence "
                    "FROM entity_type_observations WHERE chunk_id=? "
                    "AND phase1_generation_key=?",
                    (chunk_id, generation_key),
                ).fetchall()
            ]
            properties = [
                (row["entity_canonical"], row["key"], row["value"])
                for row in conn.execute(
                    "SELECT entity_canonical,key,value "
                    "FROM entity_property_observations WHERE chunk_id=? "
                    "AND phase1_generation_key=?",
                    (chunk_id, generation_key),
                ).fetchall()
            ]
            mentions = [
                str(row["entity_canonical"])
                for row in conn.execute(
                    "SELECT entity_canonical FROM entity_mention_observations "
                    "WHERE chunk_id=? AND phase1_generation_key=?",
                    (chunk_id, generation_key),
                ).fetchall()
            ]
            markers = [
                (row["kind"], row["statement"])
                for row in conn.execute(
                    "SELECT kind,statement FROM behavioral_markers "
                    "WHERE chunk_id=? AND phase1_generation_key=?",
                    (chunk_id, generation_key),
                ).fetchall()
            ]
            canonical = canonical_auxiliary_result(
                chunk_id=chunk_id,
                phase1_generation_key=generation_key,
                extraction_cache_key=str(outcome["extraction_cache_key"]),
                auxiliary_contract_key=str(outcome["auxiliary_contract_key"]),
                entity_types=types,
                entity_properties=properties,
                entity_mentions=mentions,
                markers=markers,
            )
            conn.execute(
                "UPDATE phase1_auxiliary_outcomes SET result_hash=?,"
                "entity_type_count=?,entity_property_count=?,"
                "entity_mention_count=?,marker_count=? "
                "WHERE chunk_id=? AND phase1_generation_key=?",
                (
                    canonical["result_hash"], canonical["entity_type_count"],
                    canonical["entity_property_count"],
                    canonical["entity_mention_count"],
                    canonical["marker_count"], chunk_id, generation_key,
                ),
            )


def validate_phase1_auxiliary_registry(conn: sqlite3.Connection) -> None:
    """Fail store-open when a published auxiliary ledger was corrupted."""

    invalid_base_hint = conn.execute(
        "SELECT 1 FROM entity_types WHERE "
        "hymem_entity_canonical_is_normalized(entity_canonical)<>1 "
        "OR length(trim(type))=0 "
        "OR typeof(confidence) NOT IN ('integer','real') "
        "OR confidence<0.0 OR confidence>1.0 "
        "OR origin NOT IN ('user','legacy_unattributed') "
        "OR (origin='user' AND source_chunk_id IS NOT NULL) "
        "UNION ALL "
        "SELECT 1 FROM entity_properties WHERE "
        "hymem_entity_canonical_is_normalized(entity_canonical)<>1 "
        "OR length(trim(key))=0 OR typeof(value)<>'text' "
        "OR origin NOT IN ('user','legacy_unattributed') "
        "OR (origin='user' AND source_chunk_id IS NOT NULL) "
        "LIMIT 1"
    ).fetchone()
    if invalid_base_hint is not None:
        raise RuntimeError("entity hint authority row is invalid")

    invalid_profile = conn.execute(
        "SELECT 1 FROM profile_entries WHERE length(trim(text))=0 "
        "OR kind NOT IN ('preference','avoidance','style','context') "
        "OR source NOT IN ('user','agent_inferred','legacy_unattributed') "
        "OR typeof(pos_evidence)<>'integer' OR pos_evidence<0 "
        "OR typeof(neg_evidence)<>'integer' OR neg_evidence<0 "
        "OR (first_seen IS NOT NULL AND (typeof(first_seen)<>'text' "
        "OR length(trim(first_seen))=0)) "
        "OR (last_updated IS NOT NULL AND (typeof(last_updated)<>'text' "
        "OR length(trim(last_updated))=0)) "
        "LIMIT 1"
    ).fetchone()
    if invalid_profile is not None:
        raise RuntimeError("profile entry domain is invalid")
    for rule in conn.execute(
        "SELECT text,scope,trigger_entities,source,status,pos_evidence,"
        "neg_evidence,valid_at,invalid_at,created_at FROM rules ORDER BY id"
    ).fetchall():
        try:
            triggers = json.loads(rule["trigger_entities"])
        except (TypeError, ValueError, json.JSONDecodeError):
            triggers = None
        if (
            not isinstance(rule["text"], str)
            or not rule["text"].strip()
            or rule["scope"] not in {"always_on", "contextual"}
            or rule["source"] not in {"user", "agent_inferred"}
            or rule["status"] not in {"active", "retracted"}
            or not isinstance(rule["pos_evidence"], int)
            or int(rule["pos_evidence"]) < 0
            or not isinstance(rule["neg_evidence"], int)
            or int(rule["neg_evidence"]) < 0
            or not isinstance(triggers, list)
            or any(
                not isinstance(item, str)
                or not item.strip()
                or canonicalize.normalize(item) != item
                for item in (triggers or [])
            )
            or (rule["scope"] == "always_on" and bool(triggers))
            or (rule["scope"] == "contextual" and not triggers)
            or any(
                value is not None
                and (not isinstance(value, str) or not value.strip())
                for value in (
                    rule["valid_at"], rule["invalid_at"], rule["created_at"],
                )
            )
        ):
            raise RuntimeError("rule domain is invalid")

    rows = conn.execute(
        "SELECT chunk_id,phase1_generation_key "
        "FROM phase1_auxiliary_outcomes ORDER BY chunk_id,phase1_generation_key"
    ).fetchall()
    for row in rows:
        if not validate_phase1_auxiliary_outcome(
            conn,
            chunk_id=str(row["chunk_id"]),
            phase1_generation_key=str(row["phase1_generation_key"]),
        ):
            raise RuntimeError("Phase-1 auxiliary publication integrity check failed")

    # A chunk may retain several exact historical auxiliary generations while
    # the claim table intentionally holds only its latest successful one.  Do
    # not misclassify those old branches as corrupt. A producer-bound claim
    # must match one complete auxiliary branch; a repaired legacy claim whose
    # generation is NULL is explicitly untrusted and remains hidden/pending.
    # No claim at all is corruption in a stamped v54 store.
    invalid_current_lineage = conn.execute(
        "SELECT 1 FROM (SELECT DISTINCT chunk_id "
        "FROM phase1_auxiliary_outcomes) history "
        "LEFT JOIN kg_claim_extraction_outcomes claim "
        "ON claim.chunk_id=history.chunk_id "
        "WHERE claim.chunk_id IS NULL OR ("
        "claim.phase1_generation_key IS NOT NULL AND NOT EXISTS ("
        "SELECT 1 FROM phase1_auxiliary_outcomes current "
        "WHERE current.chunk_id=claim.chunk_id "
        "AND current.phase1_generation_key=claim.phase1_generation_key "
        "AND current.extraction_cache_key=claim.prompt_version)) LIMIT 1"
    ).fetchone()
    if invalid_current_lineage is not None:
        raise RuntimeError("Phase-1 auxiliary current claim lineage is invalid")

    orphan = conn.execute(
        "SELECT 1 FROM entity_type_observations observation "
        "WHERE NOT EXISTS (SELECT 1 FROM phase1_auxiliary_outcomes outcome "
        "WHERE outcome.chunk_id=observation.chunk_id AND "
        "outcome.phase1_generation_key=observation.phase1_generation_key) "
        "UNION ALL "
        "SELECT 1 FROM entity_property_observations observation "
        "WHERE NOT EXISTS (SELECT 1 FROM phase1_auxiliary_outcomes outcome "
        "WHERE outcome.chunk_id=observation.chunk_id AND "
        "outcome.phase1_generation_key=observation.phase1_generation_key) "
        "UNION ALL "
        "SELECT 1 FROM entity_mention_observations observation "
        "WHERE NOT EXISTS (SELECT 1 FROM phase1_auxiliary_outcomes outcome "
        "WHERE outcome.chunk_id=observation.chunk_id AND "
        "outcome.phase1_generation_key=observation.phase1_generation_key) "
        "UNION ALL "
        "SELECT 1 FROM behavioral_markers marker "
        "WHERE marker.phase1_generation_key IS NOT NULL AND NOT EXISTS ("
        "SELECT 1 FROM phase1_auxiliary_outcomes outcome "
        "WHERE outcome.chunk_id=marker.chunk_id AND "
        "outcome.phase1_generation_key=marker.phase1_generation_key) "
        "LIMIT 1"
    ).fetchone()
    if orphan is not None:
        raise RuntimeError("Phase-1 auxiliary child has no publication")

    duplicate_marker = conn.execute(
        "SELECT 1 FROM behavioral_markers "
        "WHERE phase1_generation_key IS NOT NULL "
        "GROUP BY chunk_id,phase1_generation_key,kind,statement "
        "HAVING COUNT(*)<>1 LIMIT 1"
    ).fetchone()
    if duplicate_marker is not None:
        raise RuntimeError("duplicate producer-bound behavioral marker identity")

    invalid_profile_link = conn.execute(
        "SELECT 1 FROM profile_entry_marker_evidence link "
        "LEFT JOIN behavioral_markers marker ON marker.id=link.marker_id "
        "LEFT JOIN profile_entries entry ON entry.id=link.profile_entry_id "
        "WHERE marker.id IS NULL OR entry.id IS NULL "
        "OR marker.phase1_generation_key IS NOT link.phase1_generation_key "
        "OR entry.source<>'agent_inferred' OR entry.text<>marker.statement "
        "OR entry.kind<>CASE marker.kind "
        "WHEN 'preference' THEN 'preference' "
        "WHEN 'rejection' THEN 'avoidance' WHEN 'style' THEN 'style' "
        "ELSE 'context' END LIMIT 1"
    ).fetchone()
    if invalid_profile_link is not None:
        raise RuntimeError("profile marker evidence lineage is invalid")

    invalid_profile_decision = conn.execute(
        "SELECT 1 FROM profile_marker_decisions decision "
        "LEFT JOIN behavioral_markers marker ON marker.id=decision.marker_id "
        "LEFT JOIN profile_entries entry "
        "ON entry.id=decision.profile_entry_id "
        "LEFT JOIN profile_entry_marker_evidence link "
        "ON link.marker_id=decision.marker_id "
        "WHERE marker.id IS NULL OR entry.id IS NULL "
        "OR length(trim(decision.profile_policy_key))=0 "
        "OR marker.phase1_generation_key IS NOT "
        "decision.phase1_generation_key "
        "OR (decision.decision='materialized' AND ("
        "entry.source<>'agent_inferred' OR entry.text<>marker.statement "
        "OR entry.kind<>CASE marker.kind "
        "WHEN 'preference' THEN 'preference' "
        "WHEN 'rejection' THEN 'avoidance' WHEN 'style' THEN 'style' "
        "ELSE 'context' END OR link.profile_entry_id IS NOT entry.id "
        "OR link.phase1_generation_key IS NOT "
        "decision.phase1_generation_key)) "
        "OR (decision.decision='manual_authority' AND ("
        "entry.source<>'user' OR entry.text<>marker.statement)) "
        "OR (decision.decision='identity_conflict' AND ("
        "entry.source='user' OR entry.text<>marker.statement "
        "OR entry.kind=CASE marker.kind "
        "WHEN 'preference' THEN 'preference' "
        "WHEN 'rejection' THEN 'avoidance' WHEN 'style' THEN 'style' "
        "ELSE 'context' END)) "
        "OR (decision.decision<>'materialized' AND link.marker_id IS NOT NULL) "
        "LIMIT 1"
    ).fetchone()
    if invalid_profile_decision is not None:
        raise RuntimeError("profile marker decision lineage is invalid")

    missing_profile_decision = conn.execute(
        "SELECT 1 FROM profile_entry_marker_evidence link "
        "WHERE NOT EXISTS (SELECT 1 FROM profile_marker_decisions decision "
        "WHERE decision.marker_id=link.marker_id "
        "AND decision.profile_entry_id=link.profile_entry_id "
        "AND decision.phase1_generation_key=link.phase1_generation_key "
        "AND decision.decision='materialized') LIMIT 1"
    ).fetchone()
    if missing_profile_decision is not None:
        raise RuntimeError("profile marker evidence lacks its exact decision")

    invalid_rule_link = conn.execute(
        "SELECT 1 FROM rule_marker_evidence link "
        "LEFT JOIN behavioral_markers marker ON marker.id=link.marker_id "
        "LEFT JOIN rules rule ON rule.id=link.rule_id "
        "LEFT JOIN rule_marker_decisions decision "
        "ON decision.marker_id=link.marker_id "
        "WHERE marker.id IS NULL OR rule.id IS NULL "
        "OR marker.phase1_generation_key IS NOT link.phase1_generation_key "
        "OR rule.source<>'agent_inferred' OR decision.decision<>'routed' "
        "OR decision.rule_id IS NOT link.rule_id "
        "OR decision.phase1_generation_key IS NOT link.phase1_generation_key "
        "LIMIT 1"
    ).fetchone()
    if invalid_rule_link is not None:
        raise RuntimeError("rule marker evidence lineage is invalid")

    invalid_decision = conn.execute(
        "SELECT 1 FROM rule_marker_decisions decision "
        "LEFT JOIN behavioral_markers marker ON marker.id=decision.marker_id "
        "LEFT JOIN rules rule ON rule.id=decision.rule_id "
        "WHERE marker.id IS NULL "
        "OR length(trim(decision.routing_key))=0 "
        "OR marker.phase1_generation_key IS NOT decision.phase1_generation_key "
        "OR (decision.decision='routed' AND (rule.id IS NULL "
        "OR rule.source<>'agent_inferred')) "
        "OR (decision.decision='no_rule' AND (decision.rule_id IS NOT NULL "
        "OR EXISTS (SELECT 1 FROM rule_marker_evidence link "
        "WHERE link.marker_id=decision.marker_id))) "
        "OR (decision.decision='routed' AND NOT EXISTS ("
        "SELECT 1 FROM rule_marker_evidence link "
        "WHERE link.marker_id=decision.marker_id "
        "AND link.rule_id=decision.rule_id "
        "AND link.phase1_generation_key=decision.phase1_generation_key)) "
        "LIMIT 1"
    ).fetchone()
    if invalid_decision is not None:
        raise RuntimeError("rule marker decision lineage is invalid")


def add_manual_entity_type(
    conn: sqlite3.Connection,
    entity: str,
    type_name: str,
    *,
    confidence: float = 1.0,
) -> None:
    """Explicitly publish a user-authored type, never inferred from NULL FKs."""

    if not isinstance(entity, str) or not entity.strip():
        raise ValueError("entity must be a non-empty string")
    if not isinstance(type_name, str) or not type_name.strip():
        raise ValueError("entity type must be a non-empty string")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("entity type confidence must be finite and in [0, 1]")
    canonical = canonicalize.resolve(conn, entity.strip())
    conn.execute(
        "INSERT INTO entity_types("
        "entity_canonical,type,confidence,source_chunk_id,origin) "
        "VALUES (?,?,?,NULL,'user') ON CONFLICT(entity_canonical,type) "
        "DO UPDATE SET confidence=excluded.confidence,source_chunk_id=NULL,"
        "origin='user'",
        (canonical, type_name.strip(), float(confidence)),
    )


def add_manual_entity_property(
    conn: sqlite3.Connection,
    entity: str,
    key: str,
    value: str,
) -> None:
    """Explicitly publish a user-authored property with priority over models.

    Empty string is a deliberate, portable property value (for example an
    explicitly blank configuration field); identifiers are always stripped
    and non-empty.
    """

    if not isinstance(entity, str) or not entity.strip():
        raise ValueError("entity must be a non-empty string")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("entity property key must be a non-empty string")
    if not isinstance(value, str):
        raise ValueError("entity property value must be a string")
    canonical = canonicalize.resolve(conn, entity.strip())
    conn.execute(
        "INSERT INTO entity_properties("
        "entity_canonical,key,value,source_chunk_id,origin,updated_at) "
        "VALUES (?,?,?,NULL,'user',CURRENT_TIMESTAMP) "
        "ON CONFLICT(entity_canonical,key) DO UPDATE SET "
        "value=excluded.value,source_chunk_id=NULL,origin='user',"
        "updated_at=CURRENT_TIMESTAMP",
        (canonical, key.strip(), value),
    )
