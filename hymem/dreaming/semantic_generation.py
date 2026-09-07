"""Loaded implementation and effective producer identity for memory tiers.

Configuration-only historical stamps remain readable, but a running producer
never treats them as current. Unknown clients get the existing process-instance
nonce, not an unjustified assumption that their Phase-1 route is also memory's.
"""
from __future__ import annotations

import hashlib
import json
import types

from hymem.extraction.producer import (
    canonical_module_sha256,
    canonical_module_slice_sha256,
    canonical_callable_sha256,
    producer_binding_for_declaration,
)


def semantic_generation_suffix(tier: str, client: object | None) -> str:
    if client is None:
        return ""
    from hymem.dreaming import (
        canonicalize, digest, episodes, facts, lossless, procedures, summary,
        user_profile,
    )
    from hymem.extraction import jsonio
    from hymem import redaction
    from hymem.dreaming import runner

    if tier == "digest":
        modules = (digest, episodes, procedures, summary)
        dispatch = (
            "extract_session_digest", "persist_episodes", "persist_procedures",
            "publish_digest_procedures",
            "persist_auto_session_summary", "stage_digest_extraction",
            "load_digest_staged_summary", "load_completed_digest_slices",
            "digest_staging_cursor_is_valid",
        )
    elif tier == "profile":
        modules = (user_profile, redaction)
        dispatch = ("extract_user_profile", "stage_profile_extraction", "publish_profile_generation")
    elif tier == "facts":
        modules = (facts, canonicalize)
        dispatch = ("extract_facts", "reextract_fact_outcome", "persist_facts")
    else:
        raise ValueError("unknown memory tier")
    payload = {
        "schema": "hymem-semantic-generation-v1",
        "tier": tier,
        "producer": producer_binding_for_declaration(
            client, declaration_hook="memory_producer_declaration",
        ),
        "implementation": canonical_module_sha256(*modules, jsonio),
        # A defining module alone misses an imported alias rebound at its real
        # call site. Bind those loaded aliases as well as runner dispatch.
        "call_sites": canonical_callable_sha256(
            semantic_generation_suffix,
            runner._run_dreaming,
            *(getattr(runner, name) for name in dispatch),
            *(value for module in (*modules, jsonio, lossless)
              for value in vars(module).values()
              if isinstance(value, types.FunctionType)
              or (isinstance(value, type)
                  and getattr(value, "__module__", "").startswith("hymem."))),
        ),
        "source": canonical_module_slice_sha256(
            lossless, "covered_messages_after", "validate_message_coverage_artifact",
        ),
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "|semantic=sha256:" + hashlib.sha256(encoded).hexdigest()
