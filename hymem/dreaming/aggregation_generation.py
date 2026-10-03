"""Exact, secret-free generation identity for aggregation LLM material.

The aggregation config hash identifies data-shaping settings and prompt bytes;
this module adds the effective LLM producer and executable request/parser/
recovery contract. Durable reuse is authorized only when both agree. Clients
without a validated producer declaration receive a process-instance identity,
so the exact live object may reuse its own work but a restart cannot.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from typing import Any

from hymem.config import HyMemConfig
from hymem.extraction.producer import (
    authorize_inexact_producer_generation,
    canonical_callable_sha256,
    canonical_module_sha256,
    producer_binding_for_declaration,
    producer_generation_runtime_authorized,
    validate_aggregation_producer_binding,
)

_CANONICAL_CALLABLE_SHA256 = canonical_callable_sha256
_CANONICAL_MODULE_SHA256 = canonical_module_sha256


AGGREGATION_GENERATION_SCHEMA = "hymem-aggregation-generation-v1"
AGGREGATION_GENERATION_KEY_PREFIX = f"{AGGREGATION_GENERATION_SCHEMA}:"
AGGREGATION_MATERIAL_ALGORITHM_VERSION = "aggregation-material-algorithm-v1"
AGGREGATION_REQUEST_CONTRACT_VERSION = "aggregation-llm-request-contract-v2"
_GENERATION_KEY_RE = re.compile(
    rf"{re.escape(AGGREGATION_GENERATION_KEY_PREFIX)}[0-9a-f]{{64}}\Z"
)
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONFIG_RE = re.compile(r"aggregation-build-config-v1:[0-9a-f]{64}\Z")
_MAX_BINDING_BYTES = 40_960


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def aggregation_generation_contract(cfg: HyMemConfig) -> dict[str, str]:
    """Return the current executable aggregation generation contract."""

    # Imports are intentionally lazy: aggregate.py calls this module from the
    # build boundary, after its definitions are complete.
    from hymem.dreaming import aggregate as implementation
    from hymem.dreaming import aggregation_provenance as provenance
    from hymem.core import graph as graph_semantics
    from hymem.extraction import jsonio as json_parser
    from hymem.dreaming.aggregation_provenance import (
        AGGREGATION_MAX_SUMMARY_CHARS,
        AGGREGATION_MAX_TITLE_CHARS,
        aggregation_input_manifest_hash,
        aggregation_fusion_max_tokens,
        aggregation_llm_request,
        aggregation_llm_request_hash,
        aggregation_node_id,
        aggregation_output_hash,
        aggregation_output_is_canonical,
        aggregation_publication_id,
        aggregation_typed_input_fingerprint,
        combine_source_occurrences,
        load_knowledge_graph_anchor_inputs,
        load_profile_anchor_inputs,
        load_root_anchor_inputs,
        make_aggregation_input_proof,
        persist_aggregation_source_manifest,
    )
    prompts = {
        "cluster_system": implementation.AGGREGATE_SYSTEM,
        "cluster_user": implementation.AGGREGATE_USER_TEMPLATE,
        "rollup_system": implementation.ROLLUP_SYSTEM,
        "rollup_user": implementation.ROLLUP_USER_TEMPLATE,
        "root_system": implementation.DIGEST_SYSTEM,
        "root_user": implementation.DIGEST_USER_TEMPLATE,
    }
    request_policy = {
        "version": AGGREGATION_REQUEST_CONTRACT_VERSION,
        "response_format": "json",
        "temperature": 0.0,
        "max_tokens": "min(8192,2048+len(rendered_user_prompt)//2)",
        "max_tokens_floor": 2048,
        "max_tokens_ceiling": 8192,
        "max_title_chars": AGGREGATION_MAX_TITLE_CHARS,
        "max_summary_chars": AGGREGATION_MAX_SUMMARY_CHARS,
        "parse": "loads_lenient-object",
        "validation": "nonempty-stripped-title-summary-bounded-v1",
        "recovery": "one-identical-request-reroll-on-structural-ceiling-cut",
        "shrink": "disabled",
        "json_ceiling_cut_policy": json_parser.JSON_CEILING_CUT_POLICY_VERSION,
        "json_delimiters": json_parser._DELIMS,
        "json_whitespace": sorted(json_parser._JSON_WHITESPACE),
        "json_simple_escapes": sorted(json_parser._JSON_SIMPLE_ESCAPES),
        "json_hex_digits": sorted(json_parser._JSON_HEX_DIGITS),
        "json_opening_fence_pattern": json_parser._OPENING_JSON_FENCE.pattern,
        "json_opening_fence_flags": json_parser._OPENING_JSON_FENCE.flags,
        "json_prefix_states": [
            json_parser._PREFIX_COMPLETE,
            json_parser._PREFIX_INCOMPLETE,
            json_parser._PREFIX_INVALID,
        ],
    }
    callable_surface = _CANONICAL_CALLABLE_SHA256(
        implementation.aggregation_config_version,
        implementation.load_clusterable_episodes,
        implementation.generate_candidate_pairs,
        implementation._cosine,
        implementation._jaccard,
        implementation._linked,
        implementation.cluster_episodes,
        implementation.select_clusters,
        implementation._is_cut_id,
        implementation._content_defined_groups,
        implementation._norm_entity,
        implementation._node_id,
        implementation._stable_sample,
        implementation._centroid,
        implementation._forecast_rebuild,
        implementation._leaf_fingerprint,
        implementation._items_text,
        implementation._fusion_max_tokens,
        implementation._fusion_request,
        implementation._llm_fuse,
        implementation._summarize_cluster,
        implementation._build_digest_levels,
        implementation._node_frontier_item,
        implementation._candidate_node_row,
        implementation._reusable_fusion,
        implementation.load_aggregation_node_proof,
        implementation.loads_lenient,
        implementation.is_ceiling_cut,
        json_parser._JSONContainerFrame,
        json_parser._new_json_container_frame,
        json_parser._reject_duplicate_object_keys,
        json_parser._reject_nonfinite_constant,
        json_parser._finite_float,
        json_parser.loads_strict_json,
        json_parser.loads_exact_or_fenced,
        json_parser._json_container_starts,
        json_parser._scan_json_container_prefix,
        json_parser._close_container,
        json_parser._scan_json_string,
        json_parser._scan_json_literal,
        json_parser._scan_json_number,
        json_parser._candidate_spans,
        json_parser._span,
        json_parser._matches,
        json_parser._unwrap_envelope,
        combine_source_occurrences,
        make_aggregation_input_proof,
        load_profile_anchor_inputs,
        load_knowledge_graph_anchor_inputs,
        load_root_anchor_inputs,
        # Anchor selection imports these at execution. Bind their loaded
        # implementations as well as the caller, including rebound helpers.
        graph_semantics.anchor_edge_order_sql,
        graph_semantics.graph_clock_order_sql,
        graph_semantics.bounded_graph_clock_sql,
        persist_aggregation_source_manifest,
        aggregation_typed_input_fingerprint,
        aggregation_input_manifest_hash,
        aggregation_fusion_max_tokens,
        aggregation_llm_request,
        aggregation_llm_request_hash,
        aggregation_node_id,
        aggregation_output_hash,
        aggregation_output_is_canonical,
        aggregation_publication_id,
    )
    executable = _digest({
        "module_source_sha256": _CANONICAL_MODULE_SHA256(
            implementation, provenance, json_parser, graph_semantics,
        ),
        "runtime_callable_surface_sha256": callable_surface,
    })
    return {
        "material_config_version": implementation.aggregation_config_version(cfg),
        "material_algorithm_version": AGGREGATION_MATERIAL_ALGORITHM_VERSION,
        "prompts_sha256": _digest(prompts),
        "request_policy_sha256": _digest(request_policy),
        "implementation_sha256": executable,
    }


def aggregation_generation_binding_for_contract(
    contract: Mapping[str, str], client: object,
) -> dict[str, Any]:
    """Bind a producer to one lifecycle-cached executable contract."""

    # Reuse the historical validator's closed contract shape without hashing
    # callable source again. The caller obtained this mapping from
    # ``aggregation_generation_contract`` in the same lifecycle.
    if not isinstance(contract, Mapping) or set(contract) != {
        "material_config_version", "material_algorithm_version",
        "prompts_sha256", "request_policy_sha256", "implementation_sha256",
    }:
        raise ValueError("aggregation generation contract shape is invalid")
    producer = producer_binding_for_declaration(
        client, declaration_hook="aggregation_producer_declaration"
    )
    payload = {
        "schema": AGGREGATION_GENERATION_SCHEMA,
        "contract": dict(contract),
        "producer": producer,
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    binding = {
        **payload,
        "generation_key": AGGREGATION_GENERATION_KEY_PREFIX
        + hashlib.sha256(encoded).hexdigest(),
    }
    if not producer["identity_exact"]:
        authorize_inexact_producer_generation(client, binding["generation_key"])
    return validate_aggregation_generation_binding(binding)


def aggregation_generation_binding(
    cfg: HyMemConfig, client: object,
) -> dict[str, Any]:
    """Bind the actual LLM producer to the exact aggregation contract."""

    return aggregation_generation_binding_for_contract(
        aggregation_generation_contract(cfg), client,
    )


def validate_aggregation_generation_binding(value: object) -> dict[str, Any]:
    """Validate a historical generation envelope without adopting its claims."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema", "contract", "producer", "generation_key",
    }:
        raise ValueError("aggregation generation binding shape is invalid")
    if value.get("schema") != AGGREGATION_GENERATION_SCHEMA:
        raise ValueError("aggregation generation binding schema is invalid")
    contract = value.get("contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "material_config_version", "material_algorithm_version",
        "prompts_sha256", "request_policy_sha256", "implementation_sha256",
    }:
        raise ValueError("aggregation generation contract shape is invalid")
    config_version = contract.get("material_config_version")
    if not isinstance(config_version, str) or _CONFIG_RE.fullmatch(config_version) is None:
        raise ValueError("aggregation generation config identity is invalid")
    algorithm = contract.get("material_algorithm_version")
    if (
        not isinstance(algorithm, str)
        or not algorithm
        or len(algorithm) > 128
        or "\x00" in algorithm
    ):
        raise ValueError("aggregation algorithm identity is invalid")
    for field in (
        "prompts_sha256", "request_policy_sha256", "implementation_sha256",
    ):
        digest = contract.get(field)
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise ValueError("aggregation contract digest is invalid")
    producer = validate_aggregation_producer_binding(value.get("producer"))
    payload = {
        "schema": value["schema"],
        "contract": dict(contract),
        "producer": producer,
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_key = AGGREGATION_GENERATION_KEY_PREFIX + hashlib.sha256(
        encoded
    ).hexdigest()
    if value.get("generation_key") != expected_key:
        raise ValueError("aggregation generation key disagrees with its binding")
    return {**payload, "generation_key": expected_key}


def validate_current_aggregation_generation_binding(
    value: object, *, cfg: HyMemConfig,
) -> dict[str, Any]:
    binding = validate_aggregation_generation_binding(value)
    if binding["contract"] != aggregation_generation_contract(cfg):
        raise ValueError("aggregation generation uses another executable contract")
    return binding


def canonical_aggregation_generation_json(value: object) -> str:
    binding = validate_aggregation_generation_binding(value)
    encoded = json.dumps(
        binding, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > _MAX_BINDING_BYTES:
        raise ValueError("aggregation generation binding is too large")
    return encoded


def aggregation_generation_key_is_shaped(value: object) -> bool:
    return isinstance(value, str) and _GENERATION_KEY_RE.fullmatch(value) is not None


def register_aggregation_generation(
    conn: sqlite3.Connection, value: object,
) -> dict[str, Any]:
    binding = validate_aggregation_generation_binding(value)
    producer = binding["producer"]
    if not producer["identity_exact"] and not producer_generation_runtime_authorized(
        binding["generation_key"], False
    ):
        raise ValueError("inexact aggregation generation has no live client authority")
    encoded = canonical_aggregation_generation_json(binding)
    expected = (
        binding["contract"]["material_config_version"],
        producer["identity_sha256"],
        1 if producer["identity_exact"] else 0,
        producer["reuse_scope"],
        encoded,
    )
    row = conn.execute(
        "SELECT material_config_version,producer_identity_sha256,identity_exact,"
        "reuse_scope,binding_json FROM aggregation_generations "
        "WHERE generation_key=?",
        (binding["generation_key"],),
    ).fetchone()
    if row is not None:
        if tuple(row) != expected:
            raise ValueError("aggregation generation registry collision")
        return binding
    conn.execute(
        "DELETE FROM aggregation_generations WHERE identity_exact=0 "
        "AND NOT EXISTS (SELECT 1 FROM aggregation_nodes node "
        "WHERE node.aggregation_generation_key=aggregation_generations.generation_key) "
        "AND NOT EXISTS (SELECT 1 FROM aggregation_publication_state publication "
        "WHERE publication.aggregation_generation_key=aggregation_generations.generation_key) "
        "AND NOT EXISTS (SELECT 1 FROM aggregation_build_health health WHERE "
        "health.last_success_generation_key=aggregation_generations.generation_key OR "
        "health.pending_generation_key=aggregation_generations.generation_key OR "
        "health.last_failure_generation_key=aggregation_generations.generation_key) "
        "AND NOT EXISTS (SELECT 1 FROM dream_runs run "
        "WHERE run.aggregation_generation_key=aggregation_generations.generation_key)"
    )
    conn.execute(
        "INSERT INTO aggregation_generations("
        "generation_key,material_config_version,producer_identity_sha256,"
        "identity_exact,reuse_scope,binding_json) VALUES (?,?,?,?,?,?)",
        (binding["generation_key"], *expected),
    )
    return binding


def load_registered_aggregation_generation(
    conn: sqlite3.Connection, generation_key: object, *,
    allow_inexact: bool = False,
) -> dict[str, Any] | None:
    """Load one registry row under explicit process-identity authority.

    Unscoped/historical readers accept durable declarations only. A caller
    that already proved the live expected generation may opt into an inexact
    row for that exact process instance.
    """

    if not aggregation_generation_key_is_shaped(generation_key):
        return None
    row = conn.execute(
        "SELECT material_config_version,producer_identity_sha256,identity_exact,"
        "reuse_scope,binding_json FROM aggregation_generations "
        "WHERE generation_key=?",
        (generation_key,),
    ).fetchone()
    if row is None:
        return None
    try:
        if aggregation_generation_registry_row_is_valid(
            generation_key, row["material_config_version"],
            row["producer_identity_sha256"], row["identity_exact"],
            row["reuse_scope"], row["binding_json"],
        ) != 1:
            return None
        binding = validate_aggregation_generation_binding(
            json.loads(row["binding_json"])
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return None
    if not binding["producer"]["identity_exact"] and not allow_inexact:
        return None
    if not producer_generation_runtime_authorized(
        binding["generation_key"], row["identity_exact"]
    ):
        return None
    return binding


def aggregation_generation_registry_row_is_valid(
    generation_key: object, material_config_version: object,
    producer_identity_sha256: object, identity_exact: object,
    reuse_scope: object, binding_json: object,
) -> int:
    try:
        if not isinstance(binding_json, str) or len(
            binding_json.encode("utf-8")
        ) > _MAX_BINDING_BYTES:
            return 0

        def reject_duplicate_keys(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = item
            return result

        decoded = json.loads(
            binding_json, object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {item}")
            ),
        )
        binding = validate_aggregation_generation_binding(decoded)
        producer = binding["producer"]
        expected = (
            binding["generation_key"],
            binding["contract"]["material_config_version"],
            producer["identity_sha256"],
            1 if producer["identity_exact"] else 0,
            producer["reuse_scope"],
            canonical_aggregation_generation_json(binding),
        )
        return int(expected == (
            generation_key, material_config_version, producer_identity_sha256,
            identity_exact, reuse_scope, binding_json,
        ))
    except (TypeError, ValueError, UnicodeError, OverflowError):
        return 0
