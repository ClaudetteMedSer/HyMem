"""Exact, secret-free freshness identity for aggregation source material.

The v56 generation key answers *who/what executable request produced the LLM
text*.  This module answers the independent question *which current material
and retrieval-generation conditions were consumed*.  The identities compose;
neither is permitted to stand in for the other.

Material snapshots retain only counts and cryptographic commitments. Episode,
profile, graph, source, vector, and opaque endpoint bytes never enter the
registry JSON. A monotonically increasing SQLite revision cheaply fences the
episode universe and effective vector space. Root/profile/KG material instead
uses the one canonical capped selector and exact source proofs at capture,
every provider boundary, publication commit, and public read; broad SQL
triggers must not withdraw a tree for an unselected root candidate.
"""
from __future__ import annotations

import hashlib
import dis
import inspect
import json
import re
import sqlite3
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from hymem.extraction.producer import (
    _unknown_producer,
    canonical_callable_sha256,
    exact_callable_sha256,
)
from hymem.core.time import EVENT_CLOCK_SKEW_SECONDS


EMBEDDING_PRODUCER_BINDING_SCHEMA = "hymem-embedding-producer-binding-v1"
PUBLIC_EMBEDDING_IDENTITY_SCHEMA = "hymem-public-embedding-identity-v1"
CUSTOM_EMBEDDING_DECLARATION_SCHEMA = (
    "hymem-custom-embedding-producer-declaration-v2"
)
EMBEDDING_PRODUCER_KEY_PREFIX = "hymem-embedding-producer-v1:"
AGGREGATION_MATERIAL_EPOCH_SCHEMA = "hymem-aggregation-material-epoch-v1"
AGGREGATION_MATERIAL_EPOCH_KEY_PREFIX = (
    f"{AGGREGATION_MATERIAL_EPOCH_SCHEMA}:"
)
AGGREGATION_BLOCKING_CONTRACT = "hymem-aggregation-blocking-v1"
AGGREGATION_MATERIAL_CLOCK_SCHEMA = "hymem-aggregation-material-clock-v1"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EMBEDDING_KEY_RE = re.compile(
    rf"{re.escape(EMBEDDING_PRODUCER_KEY_PREFIX)}[0-9a-f]{{64}}\Z"
)
_MATERIAL_KEY_RE = re.compile(
    rf"{re.escape(AGGREGATION_MATERIAL_EPOCH_KEY_PREFIX)}[0-9a-f]{{64}}\Z"
)
_CONFIG_RE = re.compile(r"aggregation-build-config-v1:[0-9a-f]{64}\Z")
_MAX_BINDING_BYTES = 32_768
_PUBLIC_ATTESTATION_RE = re.compile(r"[A-Za-z0-9_.:@+-]{1,256}\Z")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def aggregation_anchor_records(
    anchors: Sequence[object],
) -> list[dict[str, object]]:
    """Canonical ordered projection of selected typed root anchors."""

    return [
        {
            "ordinal": ordinal,
            "kind": getattr(anchor, "kind"),
            "source_key": getattr(anchor, "source_key"),
            "proof_sha256": getattr(anchor, "proof_hash"),
        }
        for ordinal, anchor in enumerate(anchors)
    ]


def aggregation_anchor_records_sha256(anchors: Sequence[object]) -> str:
    return _digest(aggregation_anchor_records(anchors))


def aggregation_phase1_scope_identity(
    conn: sqlite3.Connection,
    *,
    generation_keys: Sequence[str] | None = None,
) -> tuple[str, bool, str]:
    """Commit the exact Phase-1 generations supporting selected material.

    ``None`` retains the compatibility meaning of the whole current registry.
    A concrete sequence, including the empty sequence, scopes the commitment
    to generations actually consumed by selected KG anchors.  This prevents
    an unrelated extractor registration from re-keying an episode-only or
    profile-only aggregation publication.
    """

    try:
        if generation_keys is None:
            rows = conn.execute(
                "SELECT generation_key,identity_exact,reuse_scope "
                "FROM phase1_generations "
                "WHERE hymem_phase1_generation_is_current("
                "generation_key,identity_exact)=1 ORDER BY generation_key"
            ).fetchall()
        else:
            raw_keys = tuple(generation_keys)
            if any(
                not isinstance(key, str) or not key or "\x00" in key
                for key in raw_keys
            ):
                raise ValueError("aggregation Phase-1 generation key is malformed")
            keys = tuple(sorted(set(raw_keys)))
            if not keys:
                rows = []
            else:
                placeholders = ",".join("?" for _ in keys)
                rows = conn.execute(
                    "SELECT generation_key,identity_exact,reuse_scope "
                    "FROM phase1_generations WHERE generation_key IN ("
                    + placeholders
                    + ") AND hymem_phase1_generation_is_current("
                      "generation_key,identity_exact)=1 ORDER BY generation_key",
                    keys,
                ).fetchall()
                if tuple(str(row["generation_key"]) for row in rows) != keys:
                    # A selected anchor may never inherit authority from a
                    # missing or no-longer-current producer generation.
                    return _digest([
                        {"generation_key": key, "unavailable": True}
                        for key in keys
                    ]), False, "process_instance"
    except sqlite3.OperationalError:
        rows = []
    records = [
        {"generation_key": str(row["generation_key"]),
         "identity_exact": int(row["identity_exact"]),
         "reuse_scope": str(row["reuse_scope"])}
        for row in rows
    ]
    exact = all(
        record["identity_exact"] == 1
        and record["reuse_scope"] == "durable"
        for record in records
    )
    return _digest(records), exact, "durable" if exact else "process_instance"


def disabled_aggregation_phase1_scope_identity() -> tuple[str, bool, str]:
    """Canonical inert Phase-1 scope for a root-anchor-disabled material."""

    return _digest([]), True, "durable"


def aggregation_anchor_phase1_generation_keys(
    anchors: Sequence[object],
) -> tuple[str, ...]:
    """Return the canonical producer-generation set consumed by KG anchors."""

    keys: set[str] = set()
    for anchor in anchors:
        raw = getattr(anchor, "phase1_generation_keys", ())
        if not isinstance(raw, tuple) or any(
            not isinstance(key, str) or not key or "\x00" in key
            for key in raw
        ):
            raise ValueError("aggregation anchor Phase-1 scope is malformed")
        keys.update(raw)
    return tuple(sorted(keys))


def aggregation_phase1_scope_sha256(conn: sqlite3.Connection) -> str:
    """Compatibility helper returning the exact Phase-1 scope commitment."""

    return aggregation_phase1_scope_identity(conn)[0]


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _safe_model_digest(value: object) -> str:
    from hymem.contrib.endpoint_policy import validate_public_attestation

    public_value = validate_public_attestation(
        value, label="embedding public identity", max_bytes=4096,
    )
    return "sha256:" + hashlib.sha256(public_value.encode("utf-8")).hexdigest()


def _exact_embedding_binding(declaration: Mapping[str, object]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": EMBEDDING_PRODUCER_BINDING_SCHEMA,
        "identity_exact": True,
        "reuse_scope": "durable",
        "declaration": dict(declaration),
    }
    encoded = _canonical_json(payload).encode("utf-8")
    return validate_embedding_producer_binding({
        **payload,
        "producer_key": EMBEDDING_PRODUCER_KEY_PREFIX
        + hashlib.sha256(encoded).hexdigest(),
    })


def configured_openai_embedding_producer_binding(
    *,
    base_url: str,
    request_model: str,
    dimension: int,
    pin_dimension: bool,
    deployment_revision: str | None,
    deployment_tenant: str | None,
) -> dict[str, Any]:
    """Resolve the exact public producer binding used by strict runtimes.

    This performs no network or credential lookup.  Generic endpoint/model
    aliases are not durable authority: a pinned dimension plus explicit
    non-secret deployment and semantic-tenant commitments are required.
    """

    from hymem.contrib import endpoint_policy
    from hymem.contrib.endpoint_policy import secret_free_endpoint_identity
    from hymem.contrib.openai_embedding_client import (
        embedding_attestation_sha256,
        openai_compatible_embedding_identity,
        openai_embedding_transport_policy_sha256,
        OPENAI_EMBEDDING_IMPLEMENTATION_SHA256,
    )

    if not pin_dimension:
        raise ValueError("durable remote embeddings require a pinned dimension")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise ValueError("embedding dimension must be a positive integer")
    endpoint = secret_free_endpoint_identity(
        base_url,
        label="embedding",
        allow_insecure_internal_env=endpoint_policy.EMBEDDING_INTERNAL_HTTP_ENV,
    )
    revision_sha256 = embedding_attestation_sha256(
        deployment_revision, label="deployment revision",
    )
    tenant_sha256 = embedding_attestation_sha256(
        deployment_tenant, label="deployment tenant",
    )
    if revision_sha256 is None or tenant_sha256 is None:
        raise ValueError(
            "durable remote embeddings require deployment revision and tenant attestations"
        )
    model_identity = openai_compatible_embedding_identity(base_url, request_model)
    return _exact_embedding_binding({
        "kind": "openai_compatible",
        "implementation_sha256": OPENAI_EMBEDDING_IMPLEMENTATION_SHA256,
        # ``model_identity`` is a library-produced envelope containing a safe
        # canonical origin.  Validate the caller-controlled model before that
        # envelope is constructed, then hash the internal envelope directly;
        # public-attestation labels themselves deliberately reject URL syntax.
        "model_identity_sha256": "sha256:" + hashlib.sha256(
            model_identity.encode("utf-8")
        ).hexdigest(),
        "request_model_sha256": _safe_model_digest(request_model),
        "endpoint_origin": endpoint["endpoint_origin"],
        "endpoint_sha256": endpoint["endpoint_sha256"],
        "configured_dimension": dimension,
        "dimension_policy": "pinned",
        "deployment_revision_sha256": revision_sha256,
        "deployment_tenant_sha256": tenant_sha256,
        "transport_policy_sha256": openai_embedding_transport_policy_sha256(),
    })


def _custom_embedding_declaration(client: object) -> dict[str, Any] | None:
    """Validate the explicit opt-in authority contract for custom clients."""

    hook_name = "embedding_producer_declaration"
    class_hook = inspect.getattr_static(type(client), hook_name, None)
    if class_hook is None:
        return None
    if (
        inspect.getattr_static(type(client), "__getattribute__", None)
        is not object.__getattribute__
        or inspect.getattr_static(type(client), "__getattr__", None) is not None
    ):
        raise ValueError("custom embedding attribute dispatch is not attestable")
    try:
        state = object.__getattribute__(client, "__dict__")
    except (AttributeError, TypeError):
        state = {}
    if hook_name in state or "embed" in state:
        raise ValueError("custom embedding implementation is instance-shadowed")
    hook = getattr(client, hook_name, None)
    embed = inspect.getattr_static(type(client), "embed", None)
    if not callable(hook) or not isinstance(
        embed, (type(lambda: None), staticmethod, classmethod),
    ) or not isinstance(
        class_hook, (type(lambda: None), staticmethod, classmethod),
    ):
        raise ValueError("custom embedding declaration is unavailable")
    raw = hook()
    fields = {
        "schema", "implementation", "implementation_revision",
        "deployment_revision", "deployment_tenant", "model_revision",
        "request_policy", "dimension", "network_free",
    }
    if type(raw) is not dict or set(raw) != fields:
        raise ValueError("custom embedding declaration shape is malformed")
    if raw.get("schema") != CUSTOM_EMBEDDING_DECLARATION_SCHEMA:
        raise ValueError("custom embedding declaration schema is unsupported")
    if raw.get("network_free") is not True:
        raise ValueError(
            "custom network embedding identity requires a maintained route contract"
        )
    try:
        if object.__getattribute__(client, "network_free") is not True:
            raise ValueError(
                "custom embedding network posture disagrees with its declaration"
            )
    except AttributeError as exc:
        raise ValueError("custom embedding network posture is unavailable") from exc
    for field in fields - {"schema", "dimension", "network_free"}:
        value = raw.get(field)
        if (
            type(value) is not str
            or _PUBLIC_ATTESTATION_RE.fullmatch(value) is None
        ):
            raise ValueError(
                f"custom embedding {field.replace('_', ' ')} is not a public attestation"
            )
        _safe_model_digest(value)
    dimension = raw.get("dimension")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        or getattr(client, "dim", None) != dimension
    ):
        raise ValueError("custom embedding declared dimension is malformed")
    try:
        embed_function = (
            embed.__func__ if isinstance(embed, (staticmethod, classmethod))
            else embed
        )
        pending_codes = [embed_function.__code__]
        while pending_codes:
            code = pending_codes.pop()
            if any(
                instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
                for instruction in dis.get_instructions(code)
            ):
                # Sampling a class/instance helper hash before and after one
                # call cannot detect a helper ABA. Custom exact execution is
                # therefore deliberately limited to a closed direct callable;
                # stateful transports must use a maintained sealed client.
                raise ValueError(
                    "custom embedding indirect attribute dispatch is not attestable"
                )
            pending_codes.extend(
                item for item in code.co_consts
                if isinstance(item, type(code))
            )
        implementation_sha256 = _digest({
            "embed": exact_callable_sha256(embed),
            "declaration": canonical_callable_sha256(class_hook),
        })
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("custom embedding implementation is not attestable") from exc
    return {
        "kind": "custom_attested",
        "implementation_sha256": implementation_sha256,
        "declared_implementation_sha256": _safe_model_digest(
            raw["implementation"]
        ),
        "implementation_revision_sha256": _safe_model_digest(
            raw["implementation_revision"]
        ),
        "deployment_revision_sha256": _safe_model_digest(
            raw["deployment_revision"]
        ),
        "deployment_tenant_sha256": _safe_model_digest(
            raw["deployment_tenant"]
        ),
        "model_revision_sha256": _safe_model_digest(raw["model_revision"]),
        "request_policy_sha256": _safe_model_digest(raw["request_policy"]),
        "dimension": dimension,
        "network_free": True,
    }


def _known_embedding_declaration(client: object) -> dict[str, Any] | None:
    """Return maintained, implementation-bound producer metadata.

    Exact class equality is deliberate. A subclass can override ``embed`` and
    therefore cannot inherit the base implementation's durable authority.
    """

    from hymem.contrib import openai_embedding_client as openai_embedding_module
    from hymem.extraction import embeddings as embeddings_module
    from hymem.contrib.openai_embedding_client import (
        OPENAI_EMBEDDING_IMPLEMENTATION_SHA256,
    )
    from hymem.extraction.embeddings import (
        CachedEmbeddingClient,
        LOCAL_HASH_EMBEDDING_IMPLEMENTATION_SHA256,
        LocalHashEmbeddingClient,
        MAPPED_STUB_EMBEDDING_IMPLEMENTATION_SHA256,
        MappedStubEmbeddingClient,
        STUB_EMBEDDING_IMPLEMENTATION_SHA256,
        StubEmbeddingClient,
        maintained_embedding_class_integrity,
        mapped_stub_embedding_config,
        normalize_text,
    )
    OpenAICompatibleEmbeddingClient = (
        openai_embedding_module._OPENAI_EMBEDDING_MAINTAINED_CLASS
    )
    LocalHashEmbeddingClient = (
        embeddings_module._LOCAL_HASH_EMBEDDING_MAINTAINED_CLASS
    )
    StubEmbeddingClient = embeddings_module._STUB_EMBEDDING_MAINTAINED_CLASS
    MappedStubEmbeddingClient = (
        embeddings_module._MAPPED_STUB_EMBEDDING_MAINTAINED_CLASS
    )
    # Python permits a non-data class method/property to be shadowed on one
    # instance. Exact-class checking alone would then bless arbitrary vector
    # code under the maintained implementation digest.
    try:
        instance_state = object.__getattribute__(client, "__dict__")
    except (AttributeError, TypeError):
        instance_state = {}
    maintained_types = (
        OpenAICompatibleEmbeddingClient,
        LocalHashEmbeddingClient,
        MappedStubEmbeddingClient,
        StubEmbeddingClient,
    )
    if type(client) in maintained_types and (
        inspect.getattr_static(type(client), "__getattribute__", None)
        is not object.__getattribute__
        or inspect.getattr_static(type(client), "__getattr__", None) is not None
    ):
        return None
    if type(client) in maintained_types and any(name in instance_state for name in (
        "embed", "model", "dim", "request_model", "endpoint_identity",
        "configured_dim", "dimension_policy", "deployment_revision_sha256",
        "deployment_tenant_sha256", "transport_policy_sha256",
        "transport_integrity_ok", "dimension_integrity_ok",
        "_features", "_embed_with_locked_transport",
        "_verify_transport_integrity", "_transport_state",
        "_http_transport_state", "_ssl_context_state",
        "_canonical_transport_value", "_record_observed_dimension",
        "_accept_observed_dimension",
        "_config",
    )):
        return None

    if type(client) in (
        LocalHashEmbeddingClient, StubEmbeddingClient,
        MappedStubEmbeddingClient,
    ):
        if not maintained_embedding_class_integrity(client):
            return None

    if type(client) is OpenAICompatibleEmbeddingClient:
        if not client.transport_integrity_ok:
            raise RuntimeError("embedding transport identity changed")
        if not client.dimension_integrity_ok:
            raise RuntimeError("embedding dimension integrity was contradicted")
        endpoint = client.endpoint_identity
        deployment_revision = client.deployment_revision_sha256
        deployment_tenant = client.deployment_tenant_sha256
        # A route+alias cannot prove immutable weights: credentials, org, or a
        # provider-side alias may select another deployment without changing
        # the URL/model.  Only an explicit non-secret revision/tenant
        # commitment makes this producer durable; otherwise the generic
        # unknown-client path assigns process-instance scope.
        if (
            deployment_revision is None
            or deployment_tenant is None
            or client.dimension_policy != "pinned"
        ):
            return None
        if (
            not isinstance(endpoint, Mapping)
            or set(endpoint) != {"endpoint_origin", "endpoint_sha256"}
            or not isinstance(endpoint.get("endpoint_origin"), str)
            or not _valid_digest(endpoint.get("endpoint_sha256"))
        ):
            raise ValueError("embedding endpoint identity is malformed")
        return {
            "kind": "openai_compatible",
            "implementation_sha256": OPENAI_EMBEDDING_IMPLEMENTATION_SHA256,
            # ``client.model`` is a maintained typed envelope containing the
            # already-sanitized endpoint origin.  Caller-controlled request
            # model and endpoint fields are independently validated below.
            "model_identity_sha256": "sha256:" + hashlib.sha256(
                client.model.encode("utf-8")
            ).hexdigest(),
            "request_model_sha256": _safe_model_digest(client.request_model),
            "endpoint_origin": endpoint["endpoint_origin"],
            "endpoint_sha256": endpoint["endpoint_sha256"],
            "configured_dimension": client.configured_dim,
            "dimension_policy": client.dimension_policy,
            "deployment_revision_sha256": deployment_revision,
            "deployment_tenant_sha256": deployment_tenant,
            "transport_policy_sha256": client.transport_policy_sha256,
        }
    if type(client) is LocalHashEmbeddingClient:
        # These fixed fields affect retrieval collision handling and strict
        # status/receipt claims even though they do not change vector bytes.
        # A mutated client must not keep the exact maintained producer key
        # while claiming another quality/backend/network posture.
        try:
            if (
                object.__getattribute__(client, "backend")
                != "local_feature_hash"
                or object.__getattribute__(client, "quality") != "lexical"
                or object.__getattribute__(client, "network_free") is not True
            ):
                return None
        except (AttributeError, TypeError):
            return None
        model_descriptor = inspect.getattr_static(
            LocalHashEmbeddingClient, "model", None,
        )
        dim_descriptor = inspect.getattr_static(
            LocalHashEmbeddingClient, "dim", None,
        )
        if not isinstance(model_descriptor, property) or not isinstance(
            dim_descriptor, property
        ):
            return None
        return {
            "kind": "local_feature_hash",
            "implementation_sha256": LOCAL_HASH_EMBEDDING_IMPLEMENTATION_SHA256,
            "model_identity_sha256": _safe_model_digest(client.model),
            "dimension": client.dim,
            "runtime_sha256": _digest({
                "python_implementation": sys.implementation.name,
                "python_version": list(sys.version_info[:3]),
                "unicode_database": unicodedata.unidata_version,
            }),
        }
    if type(client) is StubEmbeddingClient:
        model_descriptor = inspect.getattr_static(
            StubEmbeddingClient, "model", None,
        )
        dim_descriptor = inspect.getattr_static(
            StubEmbeddingClient, "dim", None,
        )
        if not isinstance(model_descriptor, property) or not isinstance(
            dim_descriptor, property
        ):
            return None
        return {
            "kind": "test_stub",
            "implementation_sha256": STUB_EMBEDDING_IMPLEMENTATION_SHA256,
            "model_identity_sha256": _safe_model_digest(client.model),
            "dimension": client.dim,
        }
    if type(client) is MappedStubEmbeddingClient:
        config = mapped_stub_embedding_config(client)
        if config is None:
            return None
        return {
            "kind": "mapped_test_stub",
            "implementation_sha256": (
                MAPPED_STUB_EMBEDDING_IMPLEMENTATION_SHA256
            ),
            "model_identity_sha256": _safe_model_digest(config.model),
            "dimension": config.dimension,
            "vector_policy_sha256": _digest({
                "vectors": config.vectors,
                "default": config.default,
                "quality": config.quality,
                "fail_on": config.fail_on,
            }),
            "quality": config.quality,
            "network_free": True,
        }
    return _custom_embedding_declaration(client)


def _embedding_identity_source(client: object) -> object:
    """Unwrap only privately registered or exact maintained cache proxies."""

    from hymem.extraction import embeddings as embeddings_module
    from hymem.extraction.embeddings import cached_embedding_proxy_integrity
    CachedEmbeddingClient = embeddings_module._CACHED_EMBEDDING_MAINTAINED_CLASS
    from hymem.extraction.producer import _phase1_proxy_source

    seen: set[int] = set()
    while True:
        if id(client) in seen:
            raise ValueError("embedding producer proxy chain is cyclic")
        seen.add(id(client))
        source = _phase1_proxy_source(client)
        if source is not None:
            client = source
            continue
        if type(client) is CachedEmbeddingClient:
            if not cached_embedding_proxy_integrity(client):
                return client
            client = object.__getattribute__(client, "_inner")
            continue
        return client


def embedding_producer_binding(client: object | None) -> dict[str, Any]:
    """Resolve one embedding producer without persisting caller-controlled data.

    Unknown implementations get the same conservative live-object nonce used
    for unknown LLM clients. Their vectors/publications may be reused by that
    exact object during this process, never by an unscoped/restarted reader.
    """

    if client is None:
        declaration: dict[str, Any] | None = {"kind": "disabled"}
        exact = True
        scope = "durable"
        opaque_identity = None
    else:
        client = _embedding_identity_source(client)
        declaration = _known_embedding_declaration(client)
        if declaration is None:
            unknown = _unknown_producer(client)
            exact = False
            scope = "process_instance"
            opaque_identity = unknown["identity_sha256"]
            # Model/dimension values affect execution but arbitrary labels are
            # hashed before persistence so a custom client cannot smuggle a
            # route or credential into the material registry.
            try:
                identity_client = client
                from hymem.extraction.producer import (
                    _registered_phase1_proxy_delegate,
                )
                registered_source = _registered_phase1_proxy_delegate(client)
                if registered_source is not None:
                    identity_client = registered_source
                try:
                    from hymem.contrib import (
                        openai_embedding_client as openai_embedding_module,
                    )
                    OpenAICompatibleEmbeddingClient = (
                        openai_embedding_module._OPENAI_EMBEDDING_MAINTAINED_CLASS
                    )
                except ImportError:
                    OpenAICompatibleEmbeddingClient = None  # type: ignore[assignment,misc]
                if (
                    OpenAICompatibleEmbeddingClient is not None
                    and type(identity_client) is OpenAICompatibleEmbeddingClient
                ):
                    # Adaptive or unattested maintained OpenAI clients are
                    # process-only, but their internal model envelope still
                    # contains a safe origin and is not a raw public label.
                    _safe_model_digest(identity_client.request_model)
                    model_digest = "sha256:" + hashlib.sha256(
                        identity_client.model.encode("utf-8")
                    ).hexdigest()
                else:
                    model_digest = _safe_model_digest(identity_client.model)
                dimension = identity_client.dim
            except Exception as exc:
                raise ValueError("embedding client identity is unavailable") from exc
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError("embedding client dimension is malformed")
            declaration = None
            unknown_payload = {
                "opaque_instance_sha256": opaque_identity,
                "model_identity_sha256": model_digest,
            }
        else:
            exact = True
            scope = "durable"
            opaque_identity = None

    payload: dict[str, Any] = {
        "schema": EMBEDDING_PRODUCER_BINDING_SCHEMA,
        "identity_exact": exact,
        "reuse_scope": scope,
        "declaration": declaration,
    }
    if not exact:
        payload["opaque_instance"] = unknown_payload
    encoded = _canonical_json(payload).encode("utf-8")
    result = {
        **payload,
        "producer_key": EMBEDDING_PRODUCER_KEY_PREFIX
        + hashlib.sha256(encoded).hexdigest(),
    }
    return validate_embedding_producer_binding(result)


def validate_embedding_producer_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("embedding producer binding is malformed")
    exact = value.get("identity_exact")
    if not isinstance(exact, bool):
        raise ValueError("embedding producer exactness is malformed")
    expected_keys = {
        "schema", "identity_exact", "reuse_scope", "declaration",
        "producer_key",
    } | ({"opaque_instance"} if not exact else set())
    if set(value) != expected_keys:
        raise ValueError("embedding producer binding shape is malformed")
    if value.get("schema") != EMBEDDING_PRODUCER_BINDING_SCHEMA:
        raise ValueError("embedding producer binding schema is unsupported")
    if value.get("reuse_scope") != (
        "durable" if exact else "process_instance"
    ):
        raise ValueError("embedding producer reuse scope is malformed")
    declaration = value.get("declaration")
    if exact:
        if not isinstance(declaration, Mapping):
            raise ValueError("exact embedding producer lacks a declaration")
        kind = declaration.get("kind")
        allowed = {
            "disabled": {"kind"},
            "local_feature_hash": {
                "kind", "implementation_sha256", "model_identity_sha256",
                "dimension", "runtime_sha256",
            },
            "test_stub": {
                "kind", "implementation_sha256", "model_identity_sha256",
                "dimension",
            },
            "mapped_test_stub": {
                "kind", "implementation_sha256", "model_identity_sha256",
                "dimension", "vector_policy_sha256", "quality",
                "network_free",
            },
            "openai_compatible": {
                "kind", "implementation_sha256", "model_identity_sha256",
                "request_model_sha256", "endpoint_origin", "endpoint_sha256",
                "configured_dimension", "dimension_policy",
                "deployment_revision_sha256", "deployment_tenant_sha256",
                "transport_policy_sha256",
            },
            "custom_attested": {
                "kind", "implementation_sha256",
                "declared_implementation_sha256",
                "implementation_revision_sha256",
                "deployment_revision_sha256",
                "deployment_tenant_sha256", "model_revision_sha256",
                "request_policy_sha256", "dimension", "network_free",
            },
        }
        if kind not in allowed or set(declaration) != allowed[kind]:
            raise ValueError("embedding producer declaration shape is malformed")
        for key, item in declaration.items():
            if key.endswith("sha256") and not _valid_digest(item):
                raise ValueError("embedding producer digest is malformed")
        if kind != "disabled":
            dimension = declaration.get(
                "configured_dimension" if kind == "openai_compatible"
                else "dimension"
            )
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError("embedding producer dimension is malformed")
        if kind == "mapped_test_stub" and (
            declaration.get("quality") not in {"semantic", "lexical"}
            or declaration.get("network_free") is not True
        ):
            raise ValueError("mapped stub embedding posture is malformed")
        if kind == "openai_compatible":
            origin = declaration.get("endpoint_origin")
            try:
                from hymem.contrib.endpoint_policy import (
                    validate_recorded_endpoint_origin,
                )

                validate_recorded_endpoint_origin(
                    origin, label="embedding producer",
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("embedding producer endpoint is malformed") from exc
            if declaration.get("dimension_policy") != "pinned":
                raise ValueError("embedding producer endpoint is malformed")
        if kind == "custom_attested" and declaration.get("network_free") is not True:
            raise ValueError("custom embedding network posture is malformed")
    else:
        if declaration is not None:
            raise ValueError("inexact embedding producer claims a declaration")
        opaque = value.get("opaque_instance")
        if (
            not isinstance(opaque, Mapping)
            or set(opaque) != {
                "opaque_instance_sha256", "model_identity_sha256",
            }
            or not _valid_digest(opaque.get("opaque_instance_sha256"))
            or not _valid_digest(opaque.get("model_identity_sha256"))
        ):
            raise ValueError("inexact embedding producer identity is malformed")
    payload = {
        key: value[key] for key in value if key != "producer_key"
    }
    expected = EMBEDDING_PRODUCER_KEY_PREFIX + hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if value.get("producer_key") != expected:
        raise ValueError("embedding producer key disagrees with its binding")
    return {**payload, "producer_key": expected}


def public_embedding_identity(
    binding: Mapping[str, object],
    dimension: int | None,
    *,
    fallback_policy: str,
    fallback_reason: str | None,
    transport_security: str,
) -> dict[str, Any]:
    """Return one closed, route-secret-free artifact identity."""

    producer = validate_embedding_producer_binding(binding)
    declaration = producer["declaration"]
    kind = declaration.get("kind") if isinstance(declaration, Mapping) else None
    if kind == "disabled":
        backend, quality, network_free, configured = (
            "none", "none", True, False,
        )
        vector_space_key = None
    elif kind == "local_feature_hash":
        backend, quality, network_free, configured = (
            "local_feature_hash", "lexical", True, True,
        )
        vector_space_key = producer["producer_key"]
    elif kind == "openai_compatible":
        backend, quality, network_free, configured = (
            "openai_compatible", "semantic", False, True,
        )
        vector_space_key = producer["producer_key"]
    else:
        raise ValueError("embedding producer is not supported by strict artifacts")
    return validate_public_embedding_identity({
        "schema": PUBLIC_EMBEDDING_IDENTITY_SCHEMA,
        "configured": configured,
        "backend": backend,
        "quality": quality,
        "network_free": network_free,
        "vector_space_key": vector_space_key,
        "dimension": dimension,
        "identity_exact": producer["identity_exact"],
        "reuse_scope": producer["reuse_scope"],
        "producer_binding": producer,
        "fallback_policy": fallback_policy,
        "fallback_reason": fallback_reason,
        "transport_security": transport_security,
    })


def validate_public_embedding_identity(value: object) -> dict[str, Any]:
    """Validate the exact public embedding identity shared by benchmarks."""

    fields = {
        "schema", "configured", "backend", "quality", "network_free",
        "vector_space_key", "dimension", "identity_exact", "reuse_scope",
        "producer_binding", "fallback_policy", "fallback_reason",
        "transport_security",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("public embedding identity shape is malformed")
    if value.get("schema") != PUBLIC_EMBEDDING_IDENTITY_SCHEMA:
        raise ValueError("public embedding identity schema is unsupported")
    producer = validate_embedding_producer_binding(value.get("producer_binding"))
    if producer["identity_exact"] is not True or producer["reuse_scope"] != "durable":
        raise ValueError("public embedding identity is not durable exact")
    declaration = producer["declaration"]
    kind = declaration.get("kind") if isinstance(declaration, Mapping) else None
    expected = {
        "disabled": (False, "none", "none", True, None, None, "none", None, "none"),
        "local_feature_hash": (
            True, "local_feature_hash", "lexical", True,
            producer["producer_key"], declaration.get("dimension"),
            "none", None, "local-no-network",
        ),
        "openai_compatible": (
            True, "openai_compatible", "semantic", False,
            producer["producer_key"], declaration.get("configured_dimension"),
            "fail-closed", None, None,
        ),
    }
    if kind not in expected:
        raise ValueError("public embedding backend is unsupported")
    observed = (
        value.get("configured"), value.get("backend"), value.get("quality"),
        value.get("network_free"), value.get("vector_space_key"),
        value.get("dimension"), value.get("fallback_policy"),
        value.get("fallback_reason"),
        (
            value.get("transport_security")
            if kind != "openai_compatible" else None
        ),
    )
    if observed != expected[kind]:
        raise ValueError("public embedding identity disagrees with producer")
    if (
        value.get("identity_exact") is not True
        or value.get("reuse_scope") != "durable"
    ):
        raise ValueError("public embedding authority is malformed")
    if kind == "openai_compatible":
        security = value.get("transport_security")
        origin = declaration.get("endpoint_origin")
        try:
            from hymem.contrib.endpoint_policy import (
                validate_recorded_endpoint_origin,
            )

            _endpoint, derived_security = validate_recorded_endpoint_origin(
                origin, label="embedding producer",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("public embedding endpoint is malformed") from exc
        if security != derived_security:
            raise ValueError("public embedding transport posture disagrees")
    return {key: value[key] for key in fields}


def embedding_binding_model_dimension(
    binding: Mapping[str, Any], client: object | None,
) -> tuple[str | None, int | None]:
    """Return live storage model/dim after proving ``client`` matches binding."""

    validated = validate_embedding_producer_binding(binding)
    if client is None:
        if validated["declaration"] != {"kind": "disabled"}:
            raise ValueError("enabled embedding binding has no live client")
        return None, None
    if embedding_producer_binding(client) != validated:
        raise RuntimeError("embedding producer identity changed")
    try:
        dimension = client.dim
    except Exception as exc:
        raise RuntimeError("embedding producer identity changed") from exc
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise RuntimeError("embedding producer identity changed")
    return str(validated["producer_key"]), dimension


def embedding_storage_identity(client: object) -> tuple[str, int]:
    """Return the universal secret-free durable vector-space key and dimension.

    All embedding caches and mirrors use this key in their historical ``model``
    column. It is deliberately more exact than the client protocol's display
    label and prevents same-label producers from sharing vectors.
    """

    source = _embedding_identity_source(client)
    binding = embedding_producer_binding(source)
    if not binding["identity_exact"]:
        # A process-instance nonce prevents cross-object collisions, but it
        # cannot observe a hidden route/code switch inside the same arbitrary
        # object.  Such a producer may be called through the in-memory cache
        # wrapper's explicit no-cache path, but it has no authority to select
        # or populate durable mirrors/shared caches.
        raise RuntimeError(
            "inexact embedding producer has no durable storage identity"
        )
    try:
        dimension = source.dim
    except Exception as exc:
        raise RuntimeError("embedding client identity is unavailable") from exc
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise RuntimeError("embedding client has an invalid dimension")
    return str(binding["producer_key"]), int(dimension)


def embedding_execution_identity(
    client: object | None,
) -> tuple[dict[str, Any], str, int | None]:
    """Atomically bracket the mutable producer declaration and dimension."""

    if client is None:
        binding = embedding_producer_binding(None)
        return binding, str(binding["producer_key"]), None
    source = _embedding_identity_source(client)
    before = embedding_producer_binding(source)
    try:
        dimension = source.dim
    except Exception as exc:
        raise RuntimeError("embedding client identity is unavailable") from exc
    after = embedding_producer_binding(source)
    try:
        final_dimension = source.dim
    except Exception as exc:
        raise RuntimeError("embedding client identity is unavailable") from exc
    if (
        before != after
        or dimension != final_dimension
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise RuntimeError("embedding client identity changed during snapshot")
    return before, str(before["producer_key"]), int(dimension)


def aggregation_material_binding(
    *,
    material_revision: int,
    config_version: str,
    episode_ceiling_rowid: int | None,
    episode_records: Sequence[Mapping[str, object]],
    anchor_records: Sequence[Mapping[str, object]],
    blocking: Mapping[str, object],
    embedding_binding: Mapping[str, object],
    embedding_dimension: int | None,
    node_embedding_required: bool,
    root_anchors_enabled: bool,
    phase1_scope_sha256: str,
    phase1_scope_identity_exact: bool,
    phase1_scope_reuse_scope: str,
    fresh_until: str | None = None,
) -> dict[str, Any]:
    """Mint the compact commitment to one already-captured material bundle."""

    if (
        isinstance(material_revision, bool)
        or not isinstance(material_revision, int)
        or material_revision < 0
    ):
        raise ValueError("aggregation material revision is malformed")
    if not isinstance(config_version, str) or _CONFIG_RE.fullmatch(config_version) is None:
        raise ValueError("aggregation material config identity is malformed")
    if (
        episode_ceiling_rowid is not None
        and (
            isinstance(episode_ceiling_rowid, bool)
            or not isinstance(episode_ceiling_rowid, int)
            or episode_ceiling_rowid < 0
        )
    ):
        raise ValueError("aggregation episode ceiling is malformed")
    producer = validate_embedding_producer_binding(embedding_binding)
    if embedding_dimension is not None and (
        isinstance(embedding_dimension, bool)
        or not isinstance(embedding_dimension, int)
        or embedding_dimension <= 0
    ):
        raise ValueError("aggregation embedding dimension is malformed")
    if (producer["declaration"] == {"kind": "disabled"}) != (
        embedding_dimension is None
    ):
        raise ValueError("aggregation embedding policy/dimension disagrees")
    if not isinstance(node_embedding_required, bool):
        raise ValueError("node embedding policy is malformed")
    if not isinstance(root_anchors_enabled, bool):
        raise ValueError("root anchor policy is malformed")
    if not _valid_digest(phase1_scope_sha256):
        raise ValueError("Phase-1 visibility scope is malformed")
    if not isinstance(phase1_scope_identity_exact, bool) or (
        phase1_scope_reuse_scope
        != ("durable" if phase1_scope_identity_exact else "process_instance")
    ):
        raise ValueError("Phase-1 visibility scope authority is malformed")
    for label, timestamp in (("freshness", fresh_until),):
        if timestamp is None:
            continue
        try:
            from datetime import datetime
            parsed = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"aggregation {label} clock is malformed") from exc
        if parsed.strftime("%Y-%m-%d %H:%M:%S") != timestamp:
            raise ValueError(f"aggregation {label} clock is malformed")
    episodes_sha256 = _digest(list(episode_records))
    anchors_sha256 = _digest(list(anchor_records))
    blocking_sha256 = _digest(dict(blocking))
    payload = {
        "schema": AGGREGATION_MATERIAL_EPOCH_SCHEMA,
        "material_revision": material_revision,
        "config_version": config_version,
        "episode_ceiling_rowid": episode_ceiling_rowid,
        "episode_count": len(episode_records),
        "episodes_sha256": episodes_sha256,
        "anchor_count": len(anchor_records),
        "anchors_sha256": anchors_sha256,
        "blocking_sha256": blocking_sha256,
        "embedding_producer": producer,
        "embedding_dimension": embedding_dimension,
        "node_embedding_policy": (
            "required" if node_embedding_required else "disabled"
        ),
        "root_anchor_policy": "enabled" if root_anchors_enabled else "disabled",
        "phase1_scope_sha256": phase1_scope_sha256,
        "phase1_scope_identity_exact": phase1_scope_identity_exact,
        "phase1_scope_reuse_scope": phase1_scope_reuse_scope,
        "fresh_until": fresh_until,
    }
    encoded = _canonical_json(payload).encode("utf-8")
    binding = {
        **payload,
        "snapshot_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }
    with_snapshot = _canonical_json(binding).encode("utf-8")
    binding["material_epoch_key"] = (
        AGGREGATION_MATERIAL_EPOCH_KEY_PREFIX
        + hashlib.sha256(with_snapshot).hexdigest()
    )
    return validate_aggregation_material_binding(binding)


def validate_aggregation_material_binding(value: object) -> dict[str, Any]:
    expected_keys = {
        "schema", "material_revision", "config_version",
        "episode_ceiling_rowid", "episode_count", "episodes_sha256",
        "anchor_count", "anchors_sha256", "blocking_sha256",
        "embedding_producer", "embedding_dimension", "node_embedding_policy",
        "root_anchor_policy",
        "phase1_scope_sha256", "phase1_scope_identity_exact",
        "phase1_scope_reuse_scope",
        "snapshot_sha256",
        "fresh_until", "material_epoch_key",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("aggregation material binding shape is malformed")
    if value.get("schema") != AGGREGATION_MATERIAL_EPOCH_SCHEMA:
        raise ValueError("aggregation material schema is unsupported")
    for key in ("material_revision", "episode_count", "anchor_count"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError("aggregation material count is malformed")
    ceiling = value.get("episode_ceiling_rowid")
    if ceiling is not None and (
        isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 0
    ):
        raise ValueError("aggregation material ceiling is malformed")
    if not isinstance(value.get("config_version"), str) or _CONFIG_RE.fullmatch(
        str(value["config_version"])
    ) is None:
        raise ValueError("aggregation material config is malformed")
    for key in (
        "episodes_sha256", "anchors_sha256", "blocking_sha256",
        "snapshot_sha256",
    ):
        if not _valid_digest(value.get(key)):
            raise ValueError("aggregation material digest is malformed")
    if value.get("node_embedding_policy") not in {"required", "disabled"}:
        raise ValueError("aggregation node embedding policy is malformed")
    if value.get("root_anchor_policy") not in {"enabled", "disabled"}:
        raise ValueError("aggregation root anchor policy is malformed")
    if not _valid_digest(value.get("phase1_scope_sha256")):
        raise ValueError("aggregation Phase-1 scope is malformed")
    phase1_exact = value.get("phase1_scope_identity_exact")
    if not isinstance(phase1_exact, bool) or value.get(
        "phase1_scope_reuse_scope"
    ) != ("durable" if phase1_exact else "process_instance"):
        raise ValueError("aggregation Phase-1 scope authority is malformed")
    if value.get("root_anchor_policy") == "disabled":
        disabled_scope, disabled_exact, disabled_reuse = (
            disabled_aggregation_phase1_scope_identity()
        )
        if (
            value.get("anchor_count") != 0
            or value.get("anchors_sha256") != _digest([])
            or value.get("phase1_scope_sha256") != disabled_scope
            or phase1_exact is not disabled_exact
            or value.get("phase1_scope_reuse_scope") != disabled_reuse
        ):
            raise ValueError("disabled aggregation root authority is malformed")
    for label, timestamp in (("freshness", value.get("fresh_until")),):
        if timestamp is None:
            continue
        try:
            from datetime import datetime
            parsed = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"aggregation {label} clock is malformed") from exc
        if parsed.strftime("%Y-%m-%d %H:%M:%S") != timestamp:
            raise ValueError(f"aggregation {label} clock is malformed")
    producer = validate_embedding_producer_binding(value.get("embedding_producer"))
    dimension = value.get("embedding_dimension")
    if dimension is not None and (
        isinstance(dimension, bool) or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise ValueError("aggregation embedding dimension is malformed")
    if (producer["declaration"] == {"kind": "disabled"}) != (dimension is None):
        raise ValueError("aggregation embedding policy/dimension disagrees")
    payload = {
        key: value[key]
        for key in value
        if key not in {"snapshot_sha256", "material_epoch_key"}
    }
    payload["embedding_producer"] = producer
    snapshot = "sha256:" + hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if value.get("snapshot_sha256") != snapshot:
        raise ValueError("aggregation material snapshot digest disagrees")
    with_snapshot = {**payload, "snapshot_sha256": snapshot}
    key = AGGREGATION_MATERIAL_EPOCH_KEY_PREFIX + hashlib.sha256(
        _canonical_json(with_snapshot).encode("utf-8")
    ).hexdigest()
    if value.get("material_epoch_key") != key:
        raise ValueError("aggregation material key disagrees")
    return {**with_snapshot, "material_epoch_key": key}


def canonical_aggregation_material_json(value: object) -> str:
    encoded = _canonical_json(validate_aggregation_material_binding(value))
    if len(encoded.encode("utf-8")) > _MAX_BINDING_BYTES:
        raise ValueError("aggregation material binding is too large")
    return encoded


def aggregation_material_registry_row_is_valid(
    material_epoch_key: object,
    material_revision: object,
    config_version: object,
    snapshot_sha256: object,
    embedding_producer_key: object,
    identity_exact: object,
    reuse_scope: object,
    binding_json: object,
) -> int:
    try:
        if not isinstance(binding_json, str) or len(
            binding_json.encode("utf-8")
        ) > _MAX_BINDING_BYTES:
            return 0

        def reject_duplicates(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = item
            return result

        decoded = json.loads(
            binding_json,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {item}")
            ),
        )
        binding = validate_aggregation_material_binding(decoded)
        producer = binding["embedding_producer"]
        material_exact = bool(
            producer["identity_exact"]
            and binding["phase1_scope_identity_exact"]
        )
        expected = (
            binding["material_epoch_key"], binding["material_revision"],
            binding["config_version"], binding["snapshot_sha256"],
            producer["producer_key"],
            1 if material_exact else 0,
            "durable" if material_exact else "process_instance",
            canonical_aggregation_material_json(binding),
        )
        return int(expected == (
            material_epoch_key, material_revision, config_version,
            snapshot_sha256, embedding_producer_key, identity_exact,
            reuse_scope, binding_json,
        ))
    except (TypeError, ValueError, UnicodeError, OverflowError):
        return 0


def register_aggregation_material_epoch(
    conn: sqlite3.Connection, value: object,
) -> dict[str, Any]:
    binding = validate_aggregation_material_binding(value)
    producer = binding["embedding_producer"]
    material_exact = bool(
        producer["identity_exact"] and binding["phase1_scope_identity_exact"]
    )
    encoded = canonical_aggregation_material_json(binding)
    expected = (
        binding["material_revision"], binding["config_version"],
        binding["snapshot_sha256"], producer["producer_key"],
        1 if material_exact else 0,
        "durable" if material_exact else "process_instance", encoded,
    )
    row = conn.execute(
        "SELECT material_revision,config_version,snapshot_sha256,"
        "embedding_producer_key,identity_exact,reuse_scope,binding_json "
        "FROM aggregation_material_epochs WHERE material_epoch_key=?",
        (binding["material_epoch_key"],),
    ).fetchone()
    if row is not None:
        if tuple(row) != expected:
            raise ValueError("aggregation material registry collision")
        return binding
    conn.execute(
        "INSERT INTO aggregation_material_epochs("
        "material_epoch_key,material_revision,config_version,snapshot_sha256,"
        "embedding_producer_key,identity_exact,reuse_scope,binding_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (binding["material_epoch_key"], *expected),
    )
    return binding


def load_registered_aggregation_material_epoch(
    conn: sqlite3.Connection, material_epoch_key: object, *,
    allow_inexact: bool = False,
) -> dict[str, Any] | None:
    if (
        not isinstance(material_epoch_key, str)
        or _MATERIAL_KEY_RE.fullmatch(material_epoch_key) is None
    ):
        return None
    row = conn.execute(
        "SELECT material_revision,config_version,snapshot_sha256,"
        "embedding_producer_key,identity_exact,reuse_scope,binding_json "
        "FROM aggregation_material_epochs WHERE material_epoch_key=?",
        (material_epoch_key,),
    ).fetchone()
    if row is None or aggregation_material_registry_row_is_valid(
        material_epoch_key, *tuple(row)
    ) != 1:
        return None
    try:
        binding = validate_aggregation_material_binding(json.loads(row["binding_json"]))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not (
            binding["embedding_producer"]["identity_exact"]
            and binding["phase1_scope_identity_exact"]
        )
        and not allow_inexact
    ):
        return None
    return binding


def current_aggregation_material_revision(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT revision,clock_schema FROM aggregation_material_clock WHERE id=1"
    ).fetchone()
    if (
        row is None
        or isinstance(row["revision"], bool)
        or not isinstance(row["revision"], int)
        or row["revision"] < 0
        or row["revision"] > 9223372036854775806
        or row["clock_schema"] != "hymem-aggregation-material-clock-v1"
        or conn.execute(
            "SELECT COUNT(*) AS count FROM aggregation_material_clock"
        ).fetchone()["count"] != 1
    ):
        raise RuntimeError("aggregation material clock is unavailable")
    return int(row["revision"])


def aggregation_material_fresh_until(
    conn: sqlite3.Connection, *, evaluated_at: str,
) -> str | None:
    """Return the earliest no-write transition of time-gated KG authority."""

    minima: list[str] = []
    for table, column, scope in (
        ("knowledge_graph", "valid_at", "derived=0 AND status='active' AND invalid_at IS NULL"),
        ("knowledge_graph", "last_seen", "derived=0 AND status='active' AND invalid_at IS NULL"),
        ("kg_evidence", "published_at", "provenance_status='canonical' AND is_current=1"),
        ("kg_edge_lifecycle", "created_at", "event_kind='claim_assertion' AND direction=1"),
        (
            "kg_claim_extraction_outcomes", "succeeded_at",
            "EXISTS (SELECT 1 FROM kg_claim_observations observation "
            "WHERE observation.chunk_id=kg_claim_extraction_outcomes.chunk_id "
            "AND observation.prompt_version="
            "kg_claim_extraction_outcomes.prompt_version "
            "AND observation.prompt_generation="
            "kg_claim_extraction_outcomes.prompt_generation)",
        ),
    ):
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone() is None:
            continue
        available = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in available:
            continue
        row = conn.execute(
            f"SELECT MIN(strftime('%Y-%m-%d %H:%M:%S',"
            f"hymem_normalize_iso_timestamp({column}),"
            f"'-{EVENT_CLOCK_SKEW_SECONDS} seconds')) "
            f"AS fresh_until FROM {table} WHERE {scope} AND "
            f"hymem_normalize_iso_timestamp({column}) IS NOT NULL AND "
            f"hymem_timestamp_at_or_before({column},"
            f"strftime('%Y-%m-%dT%H:%M:%fZ',?,"
            f"'+{EVENT_CLOCK_SKEW_SECONDS} seconds'))=0",
            (evaluated_at,),
        ).fetchone()
        value = row["fresh_until"] if row is not None else None
        if isinstance(value, str) and value:
            minima.append(value)
    return min(minima) if minima else None


def verify_aggregation_material_epoch(
    conn: sqlite3.Connection,
    binding: Mapping[str, object],
    *,
    embedding_client: object | None,
    anchor_cap: int | None = None,
) -> None:
    """Cheap build/read fence: clock plus mutable producer declaration."""

    validated = validate_aggregation_material_binding(binding)
    if current_aggregation_material_revision(conn) != validated["material_revision"]:
        raise RuntimeError("aggregation source material changed during build")
    horizon = validated.get("fresh_until")
    if horizon is not None:
        current = conn.execute("SELECT CURRENT_TIMESTAMP AS now").fetchone()["now"]
        if not isinstance(current, str) or current >= horizon:
            raise RuntimeError("aggregation source material time horizon expired")
    producer, _model, dimension = embedding_execution_identity(embedding_client)
    if producer != validated["embedding_producer"]:
        raise RuntimeError("aggregation embedding producer changed during build")
    if dimension != validated["embedding_dimension"]:
        raise RuntimeError("aggregation embedding dimension changed during build")
    if anchor_cap is not None:
        from hymem.dreaming.aggregation_provenance import load_root_anchor_inputs

        anchors = (
            load_root_anchor_inputs(conn, anchor_cap)
            if validated["root_anchor_policy"] == "enabled"
            else []
        )
        if (
            len(anchors) != validated["anchor_count"]
            or aggregation_anchor_records_sha256(anchors)
            != validated["anchors_sha256"]
        ):
            raise RuntimeError("aggregation root anchor selection changed during build")
        selected_generation_keys = aggregation_anchor_phase1_generation_keys(
            anchors,
        )
        live_phase1_scope = (
            aggregation_phase1_scope_identity(
                conn, generation_keys=selected_generation_keys,
            )[0]
            if selected_generation_keys
            else disabled_aggregation_phase1_scope_identity()[0]
        )
        if live_phase1_scope != validated["phase1_scope_sha256"]:
            raise RuntimeError(
                "aggregation Phase-1 visibility scope changed during build"
            )
