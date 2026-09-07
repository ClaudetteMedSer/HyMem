"""Canonical identity for the Phase-1 extraction publication contract.

``HyMemConfig.prompt_version`` remains the human-operated generation used by
session digest/profile code.  It is not, by itself, sufficient authority for a
Phase-1 cache hit: prompt wording, validators, parsing, and recovery behavior
can change while somebody forgets to bump that label.

This module derives one deterministic identity from the actual rendered prompt
bytes plus the executable acceptance/recovery surface.  The logical module
names below are stable; filesystem paths and Python object reprs never enter
the digest.  Phase-1 database keys use :func:`extraction_cache_key`, so a code
change cannot silently reuse a row written under another contract even when
the public prompt label was left unchanged.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from hymem.extraction import chunk as chunk_module
from hymem.extraction import jsonio as jsonio_module
from hymem.extraction import llm as llm_module
from hymem.extraction import markers as markers_module
from hymem.extraction import prompts as prompts_module
from hymem.extraction import retry as retry_module
from hymem.extraction import triples as triples_module
from hymem.contrib.implementation_identity import import_time_source_sha256


ACTIVE_EXTRACTION_PROMPT_VERSION = "v20"
EXTRACTION_CONTRACT_SCHEMA = "hymem-extraction-contract-sha256-v1"
EXTRACTION_CACHE_SCHEMA = "hymem-extraction-cache-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXTRACTION_CONTRACT_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)
_LOADED_MODULE_SLICE_SHA256 = None


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _text_digest(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


class _StripDocstrings(ast.NodeTransformer):
    """Remove non-executable text/type-check branches from source identity."""

    def _strip(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip

    def visit_If(self, node: ast.If):  # noqa: N802 - ast visitor API
        test = node.test
        type_checking = (
            isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
        ) or (
            isinstance(test, ast.Attribute)
            and isinstance(test.value, ast.Name)
            and test.value.id == "typing"
            and test.attr == "TYPE_CHECKING"
        )
        if type_checking:
            # ``typing.TYPE_CHECKING`` is always false at runtime.  Preserve a
            # rare executable ``else`` branch while excluding annotations and
            # imports reachable only by static analyzers.
            replacement = []
            for statement in node.orelse:
                visited = self.visit(statement)
                if visited is None:
                    continue
                replacement.extend(
                    visited if isinstance(visited, list) else [visited]
                )
            return replacement
        return self.generic_visit(node)


@functools.lru_cache(maxsize=32)
def _normalized_python_source(source: str) -> str:
    """Canonical AST: insensitive to comments, docstrings, paths, and layout."""

    tree = ast.parse(source.replace("\r\n", "\n").replace("\r", "\n"))
    tree = _StripDocstrings().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _statement_bindings(node: ast.stmt) -> tuple[str, ...]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return tuple(
            alias.asname or alias.name.split(".", 1)[0]
            for alias in node.names
        )
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)

    names: list[str] = []

    def collect(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                collect(item)

    for target in targets:
        collect(target)
    return tuple(names)


@functools.lru_cache(maxsize=64)
def _normalized_module_slice(source: str, roots: tuple[str, ...]) -> str:
    """Canonical transitive AST slice rooted at runtime-relevant globals."""

    tree = ast.parse(source.replace("\r\n", "\n").replace("\r", "\n"))
    bindings: dict[str, list[int]] = {}
    for index, statement in enumerate(tree.body):
        for name in _statement_bindings(statement):
            bindings.setdefault(name, []).append(index)

    missing = sorted(set(roots) - set(bindings))
    if missing:
        raise ValueError(
            "extraction contract source roots are absent: " + ",".join(missing)
        )

    selected: set[int] = {
        index
        for index, statement in enumerate(tree.body)
        if isinstance(statement, ast.ImportFrom)
        and statement.module == "__future__"
    }
    pending = list(roots)
    expanded: set[str] = set()
    while pending:
        name = pending.pop()
        if name in expanded:
            continue
        expanded.add(name)
        for index in bindings.get(name, ()):
            if index in selected:
                continue
            selected.add(index)
            dependencies = {
                child.id
                for child in ast.walk(tree.body[index])
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
            }
            pending.extend(sorted(dependencies & set(bindings)))

    sliced = ast.Module(
        body=[tree.body[index] for index in sorted(selected)],
        type_ignores=[],
    )
    sliced = _StripDocstrings().visit(sliced)
    ast.fix_missing_locations(sliced)
    return json.dumps(
        {
            "roots": sorted(set(roots)),
            "ast": ast.dump(
                sliced, annotate_fields=True, include_attributes=False
            ),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _module_source_digest(module: ModuleType, *roots: str) -> str:
    """Bind import-time source and current loaded code, never late disk state."""

    # Producer installs this exact helper object after its circular import of
    # this module completes.  A direct import of ``contract`` may reach here
    # first, in which case completing that import performs the same install.
    # Later mutation of producer's public helper name cannot relabel or break
    # the already-loaded extraction contract.
    helper = _LOADED_MODULE_SLICE_SHA256
    if helper is None:
        from hymem.extraction import producer as _producer  # noqa: F401

        helper = _LOADED_MODULE_SLICE_SHA256
    if not callable(helper):
        raise RuntimeError("loaded extraction identity helper is unavailable")
    return helper(module, *roots)


def _contract_components(prompt_version: str) -> dict[str, Any]:
    # Resolve through the real publication consumers: a rebound module alias
    # must not be represented by a separate import of its original definition.
    from hymem.dreaming import phase1, phase1_auxiliary

    canonicalize_module = phase1.canonicalize
    auxiliary_canonicalize_module = phase1_auxiliary.canonicalize
    if (
        not isinstance(prompt_version, str)
        or not prompt_version
        or prompt_version != prompt_version.strip()
        or "\x00" in prompt_version
        or len(prompt_version) > 128
    ):
        raise ValueError("extraction prompt version is malformed")
    integrity = getattr(chunk_module, "chunk_extraction_support_integrity", None)
    frozen_integrity = getattr(
        chunk_module, "_CHUNK_EXTRACTION_INTEGRITY_FUNCTION", None,
    )
    if (
        integrity is not frozen_integrity
        or not callable(frozen_integrity)
        or not frozen_integrity()
    ):
        raise RuntimeError("chunk extraction helper integrity changed")

    # These are the exact deterministic system prompts used by benchmark
    # stores. Retraction feedback is audit data and has no dynamic prompt slot.
    # Read through the exact names bound in ``chunk``.  This matters under
    # hot-patching/mutation tests too: hashing another import of the same
    # original constant would be a parallel claim, not the active request.
    primary = chunk_module.build_chunk_extraction_system()
    empty_verification = chunk_module.build_chunk_empty_verification_system()
    omission_verification = (
        chunk_module.build_chunk_omission_verification_system()
    )

    return {
        "schema": EXTRACTION_CONTRACT_SCHEMA,
        "contract_implementation": EXTRACTION_CONTRACT_IMPLEMENTATION_SHA256,
        "prompt_version": prompt_version,
        "canonicalization_policy": canonicalize_module.CANONICALIZATION_POLICY_VERSION,
        "canonicalization_unicode_version": canonicalize_module.CANONICAL_UNICODE_VERSION,
        "auxiliary_canonicalization_policy": (
            auxiliary_canonicalize_module.CANONICALIZATION_POLICY_VERSION
        ),
        "prompt_bytes": {
            "primary_system": _text_digest(primary),
            "empty_verification_system": _text_digest(empty_verification),
            "omission_verification_system": _text_digest(
                omission_verification
            ),
            "primary_user_template": _text_digest(
                chunk_module.CHUNK_EXTRACTION_USER_TEMPLATE
            ),
            "omission_user_template": _text_digest(
                chunk_module.CHUNK_OMISSION_VERIFICATION_USER_TEMPLATE
            ),
        },
        "prompt_input_policy": {
            "extraction_feedback": (
                prompts_module.EXTRACTION_FEEDBACK_PROMPT_POLICY_VERSION
            ),
        },
        "acceptance_vocabulary": {
            # The validator's bound alias is authoritative; prompt rendering
            # is already covered byte-for-byte above.
            "predicates": list(triples_module.ALLOWED_PREDICATES),
            "entity_types": sorted(triples_module._VALID_TYPES),
            "triple_required_keys": sorted(
                triples_module._COMBINED_REQUIRED_KEYS
            ),
            "triple_optional_keys": sorted(
                triples_module._COMBINED_OPTIONAL_KEYS
            ),
            "triple_max_properties": triples_module._COMBINED_MAX_PROPERTIES,
            "marker_kinds": list(markers_module._ALLOWED_KINDS),
            "source_message_id_required_for_source_records": True,
        },
        "recovery_policy": {
            "json_ceiling_cut": (
                jsonio_module.JSON_CEILING_CUT_POLICY_VERSION
            ),
            "source_record_split": (
                chunk_module.SOURCE_RECORD_SPLIT_POLICY_VERSION
            ),
            "table_fragment_context": (
                chunk_module.SOURCE_FRAGMENT_CONTEXT_VERSION
            ),
            "prose_boundary_context": (
                chunk_module.SOURCE_BOUNDARY_CONTEXT_VERSION
            ),
            "max_prose_boundary_context_chars": (
                chunk_module._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS
            ),
            "conversation_context": chunk_module.SOURCE_CONVERSATION_CONTEXT_VERSION,
            "max_conversation_context_records": chunk_module._MAX_CONVERSATION_CONTEXT_RECORDS,
            "max_conversation_context_chars": chunk_module._MAX_CONVERSATION_CONTEXT_CHARS,
            "max_conversation_context_encoded_chars": chunk_module._MAX_CONVERSATION_CONTEXT_ENCODED_CHARS,
            "max_conversation_context_applicability_chars": chunk_module._MAX_CONVERSATION_CONTEXT_APPLICABILITY_CHARS,
            "clean_empty": chunk_module.CLEAN_EMPTY_RECOVERY_POLICY_VERSION,
            "retry_attempts": retry_module.DEFAULT_RETRY_ATTEMPTS,
            "max_completion_calls": (
                chunk_module.MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK
            ),
            "max_leaf_input_chars": chunk_module._MAX_LEAF_INPUT_CHARS,
            "max_prepartition_leaves": chunk_module._MAX_PREPARTITION_LEAVES,
            "max_split_depth": chunk_module._MAX_SPLIT_DEPTH,
            "min_fragment_content_chars": (
                chunk_module._MIN_FRAGMENT_CONTENT_CHARS
            ),
            "max_triples_per_response": chunk_module._MAX_TRIPLES_PER_RESPONSE,
            "max_markers_per_response": chunk_module._MAX_MARKERS_PER_RESPONSE,
            "explicit_cue_pattern": chunk_module._EXPLICIT_EXTRACTION_CUE.pattern,
            "explicit_cue_flags": chunk_module._EXPLICIT_EXTRACTION_CUE.flags,
        },
        # Transitive AST slices close the gap between the enumerated knobs and
        # executable behavior without coupling expensive extraction cache
        # invalidation to unrelated prompt families or test clients. Logical
        # roots, never filesystem paths or object reprs, enter the digest.
        "executable_source": {
            "canonicalization": _module_source_digest(
                canonicalize_module, "normalize", "resolve",
            ),
            "auxiliary_canonicalization": _module_source_digest(
                auxiliary_canonicalize_module, "normalize", "resolve",
            ),
            "chunk": _module_source_digest(
                chunk_module, "extract_chunk", "safe_failure_diagnostics"
            ),
            "jsonio": _module_source_digest(
                jsonio_module,
                "is_ceiling_cut",
                "loads_exact_or_fenced",
                "loads_strict_json",
            ),
            "llm": _module_source_digest(
                llm_module, "LLMRequest", "measure_provider_attempts"
            ),
            "markers": _module_source_digest(
                markers_module,
                "Marker",
                "markers_from_list",
                "normalize_combined_marker_item",
            ),
            "prompt_builders": _module_source_digest(
                prompts_module,
                "CHUNK_EXTRACTION_USER_TEMPLATE",
                "CHUNK_OMISSION_VERIFICATION_USER_TEMPLATE",
                "build_chunk_empty_verification_system",
                "build_chunk_extraction_system",
                "build_chunk_omission_verification_system",
            ),
            "retry": _module_source_digest(
                retry_module, "DEFAULT_RETRY_ATTEMPTS", "with_retry"
            ),
            "triples": _module_source_digest(
                triples_module,
                "Triple",
                "normalize_combined_triple_item",
                "triples_from_list",
            ),
        },
    }


def extraction_contract_identity(
    prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> str:
    """Return the canonical identity of the currently executing contract."""

    encoded = json.dumps(
        _contract_components(prompt_version),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{EXTRACTION_CONTRACT_SCHEMA}:" + hashlib.sha256(encoded).hexdigest()


def extraction_contract_binding(
    prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> dict[str, str]:
    """Small credential-free binding suitable for configs and artifacts."""

    return {
        "schema": EXTRACTION_CONTRACT_SCHEMA,
        "prompt_version": prompt_version,
        "identity": extraction_contract_identity(prompt_version),
    }


def validate_extraction_contract_binding(
    value: object,
    *,
    expected_prompt_version: str | None = None,
) -> dict[str, str]:
    """Require an exact binding to the code that is executing now."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema", "prompt_version", "identity",
    }:
        raise ValueError("extraction contract binding shape is invalid")
    prompt_version = value.get("prompt_version")
    if expected_prompt_version is not None and prompt_version != expected_prompt_version:
        raise ValueError("extraction contract prompt version disagrees")
    if not isinstance(prompt_version, str):
        raise ValueError("extraction contract prompt version is malformed")
    expected = extraction_contract_binding(prompt_version)
    actual = dict(value)
    if actual != expected:
        raise ValueError("extraction contract identity disagrees with runtime")
    return actual


def extraction_cache_key(
    prompt_version: str = ACTIVE_EXTRACTION_PROMPT_VERSION,
) -> str:
    """Namespace Phase-1 durable state by the derived contract identity.

    The function is intentionally idempotent for internal call chains.  A
    namespaced key from another executable contract is rejected rather than
    re-namespaced or treated as a human prompt label.
    """

    prefix = f"{EXTRACTION_CACHE_SCHEMA}:"
    if isinstance(prompt_version, str) and prompt_version.startswith(prefix):
        remainder = prompt_version[len(prefix):]
        digest, separator, public_version = remainder.partition(":")
        if not separator or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("extraction cache key is malformed")
        expected = extraction_cache_key(public_version)
        if prompt_version != expected:
            raise ValueError("extraction cache key belongs to another contract")
        return prompt_version

    identity = extraction_contract_identity(prompt_version)
    digest = identity.rsplit(":", 1)[-1]
    return f"{prefix}{digest}:{prompt_version}"


def validate_effective_config_extraction_contract(
    effective_config: object,
) -> dict[str, str]:
    """Bind a serialized/dataclass HyMem config to the runtime contract."""

    if isinstance(effective_config, Mapping):
        prompt_version = effective_config.get("prompt_version")
        binding = effective_config.get("extraction_contract")
    else:
        prompt_version = getattr(effective_config, "prompt_version", None)
        binding = getattr(effective_config, "extraction_contract", None)
    if not isinstance(prompt_version, str):
        raise ValueError("effective HyMem extraction prompt version is absent")
    return validate_extraction_contract_binding(
        binding, expected_prompt_version=prompt_version
    )
