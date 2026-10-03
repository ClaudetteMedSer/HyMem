from __future__ import annotations

import math
import os
import re
import threading
import time
import hashlib
import weakref
import importlib
import inspect
import json
import ssl
from collections.abc import Mapping
from typing import Sequence

from hymem.contrib.endpoint_policy import (
    EMBEDDING_INTERNAL_HTTP_ENV,
    resolve_embedding_api_key,
    secret_free_endpoint_identity,
    validate_public_attestation,
    validate_http_endpoint,
)
from hymem.deadline import DeadlineExceeded, current_deadline


DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 10.0
OPENAI_COMPATIBLE_EMBEDDING_IDENTITY_SCHEMA = (
    "hymem-openai-compatible-embedding-space-v2"
)


def _loaded_distribution_version(module_name: str) -> str | None:
    """Read version metadata from the loaded implementation, never dist-info."""

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    for attribute in ("__version__", "VERSION"):
        value = getattr(module, attribute, None)
        if type(value) is str and value.strip() == value and value:
            return value
    return None


# Captured while this owner module is imported.  Reading mutable dist-info at
# first use can make an old, already-loaded SDK claim a freshly deployed
# package version.  The HTTP stack and TLS runtime jointly determine the wire
# behaviour, so all are part of the maintained transport commitment.
OPENAI_TRANSPORT_RUNTIME_VERSIONS = (
    ("openai", _loaded_distribution_version("openai")),
    ("httpx", _loaded_distribution_version("httpx")),
    ("httpcore", _loaded_distribution_version("httpcore")),
    ("openssl", ssl.OPENSSL_VERSION),
)
OPENAI_TRANSPORT_RUNTIME_EXACT = all(
    type(value) is str and bool(value)
    for _name, value in OPENAI_TRANSPORT_RUNTIME_VERSIONS
)


def embedding_attestation_sha256(
    value: str | None, *, label: str,
) -> str | None:
    """Hash one operator-asserted public deployment commitment.

    A digest is a commitment, not secrecy: short or guessable values can be
    recovered by dictionary search.  Callers therefore assert that this is
    public routing/revision metadata.  Obvious credential syntax is rejected
    defensively, but no validator can prove an arbitrary label is non-secret.
    """

    if value is None:
        return None
    public_value = validate_public_attestation(
        value, label=f"embedding {label}", max_bytes=4096,
    )
    return "sha256:" + hashlib.sha256(public_value.encode("utf-8")).hexdigest()


def openai_embedding_transport_policy_sha256() -> str:
    """Commit the maintained SDK/request policy without endpoint secrets."""

    payload = {
        "schema": "hymem-openai-transport-policy-v2",
        "runtime_versions": dict(OPENAI_TRANSPORT_RUNTIME_VERSIONS),
        "request": "embeddings.create=input,model;dimensions=omitted",
        "max_retries": 0,
        "timeout": "per-client-v1",
        "httpx_trust_env": False,
        "ambient_proxy": "disabled",
        "ambient_organization": "disabled",
        "ambient_project": "disabled",
        "httpcore_policy_seal": "v1",
        "tls_policy_seal": "ssl-context-fields-v1",
    }
    return "sha256:" + hashlib.sha256(json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def is_loopback_embedding_url(base_url: str) -> bool:
    """Whether an HTTP endpoint is unambiguously local to this machine."""
    try:
        endpoint = validate_http_endpoint(base_url, label="embedding")
    except (TypeError, ValueError):
        return False
    return endpoint.is_loopback


def is_official_openai_embedding_url(base_url: str) -> bool:
    """Whether it is safe to inherit the general ``OPENAI_API_KEY``."""
    try:
        endpoint = validate_http_endpoint(base_url, label="embedding")
    except (TypeError, ValueError):
        return False
    return endpoint.official_provider == "openai"


def validate_embedding_base_url(base_url: str) -> None:
    """Reject unsafe embedding transports and ambiguous request routes."""
    validate_http_endpoint(
        base_url,
        label="embedding",
        allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
    )


def safe_embedding_base_url(base_url: str) -> str:
    """Return an endpoint label safe for diagnostics and logs.

    Request endpoints never accept userinfo, query parameters, or fragments.
    A valid URL is therefore already credential-free; malformed/ambiguous input
    becomes one constant and no attacker- or operator-controlled bytes are
    echoed from an error path.
    """
    if base_url == "local://feature-hash":
        return base_url
    try:
        return secret_free_endpoint_identity(
            base_url,
            label="embedding",
            allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
        )["endpoint_origin"]
    except (TypeError, ValueError, OverflowError):
        return "<invalid embedding URL>"


def openai_compatible_embedding_identity(base_url: str, model: str) -> str:
    """Stable secret-free vector-space id for endpoint and request model.

    Model labels are not globally unique: two gateways may expose the same
    label with unrelated weights/tokenization. The canonical origin remains
    visible for diagnosis, while a digest of the *full* canonical endpoint
    distinguishes opaque tenant/deployment paths without persisting them.
    Credentials, queries, and fragments are rejected before identity exists.
    """
    if type(base_url) is not str or type(model) is not str:
        raise ValueError("embedding endpoint and model must be non-empty")
    raw_url = base_url
    raw_model = validate_public_attestation(
        model, label="embedding request model", max_bytes=4096,
    )
    if not raw_url or not raw_model:
        raise ValueError("embedding endpoint and model must be non-empty")
    # Validate before constructing identity: an endpoint with query routing or
    # URL credentials must not mint a reusable vector-space label.
    endpoint = validate_http_endpoint(
        raw_url,
        label="embedding",
        allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
    )
    identity = secret_free_endpoint_identity(
        endpoint.url,
        label="embedding",
        allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
    )
    return (
        f"{OPENAI_COMPATIBLE_EMBEDDING_IDENTITY_SCHEMA}:"
        f"{identity['endpoint_origin']}::"
        f"endpoint-sha256={identity['endpoint_sha256'][7:]}::"
        f"model={raw_model}"
    )


_OPENAI_EMBEDDING_EXECUTION_ATTRIBUTES = (
    "embed", "_embed_with_locked_transport", "_verify_transport_integrity",
    "_transport_state", "_http_transport_state", "_ssl_context_state",
    "_canonical_transport_value", "_record_observed_dimension",
    "_accept_observed_dimension", "close", "model", "request_model",
    "endpoint_identity", "deployment_revision_sha256",
    "deployment_tenant_sha256", "transport_policy_sha256", "backend",
    "quality", "network_free", "dim", "configured_dim", "observed_dim",
    "dimension_policy", "dimension_integrity_ok", "__getattribute__",
)
_OPENAI_EMBEDDING_ORIGINAL_IMPLEMENTATION_GUARD: tuple[object, ...] | None = None
_OPENAI_EMBEDDING_CLOSE_TARGETS: weakref.WeakKeyDictionary[object, object] = (
    weakref.WeakKeyDictionary()
)
_OPENAI_EMBEDDING_CLOSE_TARGETS_LOCK = threading.Lock()


class OpenAICompatibleEmbeddingClient:
    """EmbeddingClient backed by any OpenAI-compatible HTTP endpoint.

    Works with OpenAI and embedding-capable OpenAI-compatible endpoints (for
    example Together or a local vLLM deployment).  A chat-compatible endpoint
    is not necessarily embedding-compatible; in particular no LLM credential
    is implicitly reused here.
    All constructor arguments fall back to environment variables so the server
    can be configured entirely via the shell environment.

    Environment variables (all optional if arguments are passed directly):
        HYMEM_EMBEDDING_API_KEY   — purpose-bound API key. OPENAI_API_KEY is
                                    inherited only for the exact official
                                    api.openai.com HTTPS origin; loopback uses
                                    a non-secret dummy local key.
        HYMEM_EMBEDDING_BASE_URL  — base URL (default: https://api.openai.com/v1)
        HYMEM_EMBEDDING_MODEL     — model name (default: text-embedding-3-small)
        HYMEM_EMBEDDING_DIM       — declared dim (default: 1536); the actual
                                    dim is read from the API response on the
                                    first call unless pin_dimension is enabled.
        HYMEM_EMBEDDING_TIMEOUT_SECONDS
                                  — per-request timeout (default: 10 seconds).
                                    SDK retries are disabled.
        HYMEM_EMBEDDING_PIN_DIMENSION
                                  — required by high-level durable HyMem use.
        HYMEM_EMBEDDING_DEPLOYMENT_REVISION
        HYMEM_EMBEDDING_DEPLOYMENT_TENANT
                                  — operator-asserted PUBLIC labels required
                                    for durable remote producer identity.
        HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP
                                  — explicit 1/true/yes/on opt-in for an HTTP
                                    private-IP or internal service-DNS endpoint.
                                    Public HTTP remains forbidden.

    ``pin_dimension=True`` is an explicit benchmark/integrity mode.  It keeps
    the configured dimension authoritative, rejects every contradictory
    provider response, and latches that observation for callers that need to
    prevent publication after a best-effort path caught the immediate error.
    The low-level client remains adaptive by default: a direct uncached call,
    or ``CachedEmbeddingClient``'s process-only bypass, may accept the first
    coherent observed dimension and will never cache or persist it.  HyMem's
    durable ingestion/query/bootstrap paths require a pinned dimension plus
    public deployment revision and tenant attestations and fail before a
    provider call when that exact authority is absent.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        timeout: float | None = None,
        pin_dimension: bool = False,
        deployment_revision: str | None = None,
        deployment_tenant: str | None = None,
    ) -> None:
        resolved_base = (
            base_url
            or os.environ.get("HYMEM_EMBEDDING_BASE_URL")
            or "https://api.openai.com/v1"
        )
        endpoint, resolved_key = resolve_embedding_api_key(
            resolved_base, explicit_key=api_key
        )

        try:
            openai_module = importlib.import_module("openai")
            OpenAI = openai_module.OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install 'hymem[server]'"
            ) from e
        self._model = (
            model
            or os.environ.get("HYMEM_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )
        env_dim = os.environ.get("HYMEM_EMBEDDING_DIM")
        try:
            resolved_dim = dim if dim is not None else (int(env_dim) if env_dim else 1536)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding dimension must be a positive integer") from exc
        if (
            isinstance(resolved_dim, bool)
            or not isinstance(resolved_dim, int)
            or resolved_dim <= 0
        ):
            raise ValueError("embedding dimension must be a positive integer")
        if not isinstance(pin_dimension, bool):
            raise ValueError("pin_dimension must be a boolean")
        if type(self._model) is not str or not self._model.strip():
            raise ValueError("embedding model id must be non-empty")
        self._model = self._model.strip()
        revision = deployment_revision or os.environ.get(
            "HYMEM_EMBEDDING_DEPLOYMENT_REVISION"
        )
        if revision is not None and (
            type(revision) is not str or not revision.strip()
            or "\x00" in revision
        ):
            raise ValueError("embedding deployment revision must be non-empty")
        self._deployment_revision_sha256 = embedding_attestation_sha256(
            revision, label="deployment revision",
        )
        tenant = deployment_tenant or os.environ.get(
            "HYMEM_EMBEDDING_DEPLOYMENT_TENANT"
        )
        if tenant is not None and (
            type(tenant) is not str or not tenant.strip() or "\x00" in tenant
        ):
            raise ValueError("embedding deployment tenant must be non-empty")
        # This is an explicit, non-secret semantic-routing attestation.  We do
        # not derive it from API keys or ambient SDK organization/project
        # variables: doing so would either persist secrets or silently bless a
        # credential-routed deployment under the same vector-space key.
        self._deployment_tenant_sha256 = embedding_attestation_sha256(
            tenant, label="deployment tenant",
        )
        self._transport_policy_sha256 = openai_embedding_transport_policy_sha256()
        env_timeout = os.environ.get("HYMEM_EMBEDDING_TIMEOUT_SECONDS")
        try:
            resolved_timeout = (
                timeout
                if timeout is not None
                else (
                    float(env_timeout)
                    if env_timeout is not None
                    else DEFAULT_EMBEDDING_TIMEOUT_SECONDS
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding timeout must be a positive finite number") from exc
        if (
            isinstance(resolved_timeout, bool)
            or not isinstance(resolved_timeout, (int, float))
            or not math.isfinite(float(resolved_timeout))
            or float(resolved_timeout) <= 0.0
        ):
            raise ValueError("embedding timeout must be a positive finite number")
        self._identity = openai_compatible_embedding_identity(
            endpoint.url, self._model
        )
        self._endpoint_identity = secret_free_endpoint_identity(
            endpoint.url,
            label="embedding",
            allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
        )
        # Keep the operator's requested dimension separate from the provider's
        # observed dimension.  General HyMem deployments retain the historical
        # adaptive behaviour through ``dim``; strict benchmark builders opt in
        # to a fail-closed pin without making receipt identity phase-dependent.
        self._configured_dim = resolved_dim
        self._dim = resolved_dim
        self._observed_dim: int | None = None
        self._pin_dimension = pin_dimension
        self._dimension_mismatch_observed = False
        self._dimension_lock = threading.Lock()
        self._transport_lock = threading.RLock()
        # Public/status/receipt metadata exposes only the origin. The exact
        # opaque route remains committed by ``endpoint_sha256`` and is kept by
        # the private SDK transport, never in durable model/status fields.
        self.base_url = self._endpoint_identity["endpoint_origin"]
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.input_count = 0
        self.input_characters = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.cost_usd = None
        self.token_usage_available = False
        self._usage_complete = True
        self._usage_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._request_timeout = float(resolved_timeout)
        # Keep exactly one retry layer on the latency-sensitive query and
        # post-commit ingestion paths.  The SDK default retries and multi-minute
        # timeout would otherwise multiply into an effectively unbounded stall.
        client_kwargs: dict[str, object] = {
            "api_key": resolved_key,
            "base_url": endpoint.url,
            # Disable SDK ambient organization/project routing. These values
            # are intentionally empty public policy, never credentials.
            "organization": "",
            "project": "",
            "timeout": float(resolved_timeout),
            "max_retries": 0,
        }
        default_http_client = getattr(
            openai_module, "DefaultHttpxClient", None,
        )
        self._owned_transport_policy = callable(default_http_client)
        self._owned_http_client = None
        if self._owned_transport_policy:
            # Exact producer authority never inherits HTTPS_PROXY/ALL_PROXY or
            # another ambient route. Such URLs may contain credentials and are
            # intentionally absent from every durable binding and diagnostic.
            self._owned_http_client = default_http_client(
                timeout=float(resolved_timeout), trust_env=False,
            )
            client_kwargs["http_client"] = self._owned_http_client
        try:
            self._client = OpenAI(**client_kwargs)
            self._transport_object = self._client
            self._http_transport_object = getattr(self._client, "_client", None)
            self._embedding_resource = self._client.embeddings
            self._embedding_create_descriptor = inspect.getattr_static(
                type(self._embedding_resource), "create", None,
            )
            self._embedding_resource_has_client_backref = hasattr(
                self._embedding_resource, "_client"
            )
            self._embedding_instance_create = (
                getattr(self._embedding_resource, "create", None)
                if self._embedding_create_descriptor is None else None
            )
            self._implementation_guard = (
                _OPENAI_EMBEDDING_ORIGINAL_IMPLEMENTATION_GUARD
            )
            self._transport_guard = self._transport_state()
            with _OPENAI_EMBEDDING_CLOSE_TARGETS_LOCK:
                _OPENAI_EMBEDDING_CLOSE_TARGETS[self] = self._client
        except BaseException as primary:
            # Constructor failure transfers no ownership to the caller. Close
            # the transport we created without replacing the causal exception.
            owned = self._owned_http_client
            if owned is not None:
                try:
                    close = getattr(owned, "close", None)
                    if callable(close):
                        close()
                except Exception as cleanup_error:
                    try:
                        primary.add_note(
                            "owned embedding transport cleanup failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    except Exception:
                        pass
            raise

    @staticmethod
    def _canonical_transport_value(value: object) -> object:
        """Snapshot SDK request state without rendering provider objects.

        ``openai.Omit`` values intentionally compare by identity.  Deep-copying
        a headers mapping therefore made a pristine client fail its own seal.
        Canonical typed tags make those sentinels stable while retaining exact
        in-memory comparisons for credential/header primitives.  Unknown
        objects are bound by identity only and are never stringified, logged,
        hashed, or persisted.
        """

        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return ("value", type(value).__name__, value)
        value_type = type(value)
        type_tag = (value_type.__module__, value_type.__qualname__)
        if type_tag[0].startswith("openai") and type_tag[1].endswith("Omit"):
            return ("sdk-sentinel", *type_tag)
        if isinstance(value, Mapping):
            items = [
                (
                    OpenAICompatibleEmbeddingClient._canonical_transport_value(key),
                    OpenAICompatibleEmbeddingClient._canonical_transport_value(item),
                )
                for key, item in value.items()
            ]
            items.sort(key=lambda pair: repr(pair[0]))
            return ("mapping", type_tag, tuple(items))
        if isinstance(value, (tuple, list)):
            return (
                "sequence", type_tag,
                tuple(
                    OpenAICompatibleEmbeddingClient._canonical_transport_value(item)
                    for item in value
                ),
            )
        return ("opaque-object", type_tag, id(value))

    @classmethod
    def _ssl_context_state(cls, context: object) -> object:
        """Bind stable TLS policy without formatting certificate/key state."""

        if context is None:
            return cls._canonical_transport_value(None)
        return (
            "ssl-context",
            (type(context).__module__, type(context).__qualname__),
            id(context),
            tuple(
                (name, cls._canonical_transport_value(getattr(context, name, None)))
                for name in (
                    "check_hostname", "verify_mode", "verify_flags",
                    "minimum_version", "maximum_version", "options",
                    "security_level", "hostname_checks_common_name",
                    "post_handshake_auth", "keylog_filename",
                )
            ),
        )

    @classmethod
    def _http_transport_state(cls, transport: object) -> object:
        """Bind one maintained HTTPTransport's non-live routing policy."""

        if transport is None:
            return cls._canonical_transport_value(None)
        pool = getattr(transport, "_pool", None)
        return (
            "http-transport",
            (type(transport).__module__, type(transport).__qualname__),
            id(transport),
            (type(pool).__module__, type(pool).__qualname__),
            id(pool),
            cls._ssl_context_state(getattr(pool, "_ssl_context", None)),
            tuple(
                (name, cls._canonical_transport_value(getattr(pool, name, None)))
                for name in (
                    "_proxy", "_max_connections",
                    "_max_keepalive_connections", "_keepalive_expiry",
                    "_http1", "_http2", "_retries", "_local_address",
                    "_uds", "_network_backend", "_socket_options",
                )
            ),
        )

    def _transport_state(self) -> tuple[object, ...]:
        """Process-only effective SDK state; never serialize or log it.

        This seals accidental or non-concurrent private-state drift inside the
        maintained OpenAI/httpx object graph.  It is not a sandbox against
        hostile code running in the same interpreter; DNS/proxy/provider-side
        semantics remain covered by TLS plus the operator's explicit immutable
        deployment/tenant attestation.
        """

        transport = self._client
        http_client = getattr(transport, "_client", None)
        mounts = getattr(http_client, "_mounts", None)
        mount_state = (
            tuple(sorted(
                (id(key), self._http_transport_state(value))
                for key, value in mounts.items()
            ))
            if isinstance(mounts, Mapping) else None
        )
        hooks = getattr(http_client, "_event_hooks", None)
        hook_state = (
            tuple(sorted(
                (str(key), tuple(id(callback) for callback in callbacks))
                for key, callbacks in hooks.items()
            ))
            if isinstance(hooks, Mapping) else None
        )
        def collection_state(value: object) -> object:
            try:
                if hasattr(value, "multi_items"):
                    value = list(value.multi_items())
                elif hasattr(value, "items"):
                    value = list(value.items())
            except Exception:
                pass
            return self._canonical_transport_value(value)

        timeout = getattr(http_client, "timeout", None)
        timeout_state = tuple(
            self._canonical_transport_value(getattr(timeout, field, None))
            for field in ("connect", "read", "write", "pool")
        )
        return (
            str(getattr(transport, "base_url", "")),
            getattr(transport, "api_key", None),
            getattr(transport, "organization", None),
            getattr(transport, "project", None),
            self._canonical_transport_value(
                getattr(transport, "default_headers", None)
            ),
            self._canonical_transport_value(
                getattr(transport, "default_query", None)
            ),
            self._http_transport_state(getattr(http_client, "_transport", None)),
            id(getattr(http_client, "_auth", None)),
            mount_state,
            hook_state,
            str(getattr(http_client, "base_url", "")),
            collection_state(getattr(http_client, "headers", None)),
            collection_state(getattr(http_client, "cookies", None)),
            collection_state(getattr(http_client, "params", None)),
            timeout_state,
            self._canonical_transport_value(
                getattr(http_client, "follow_redirects", None)
            ),
            self._canonical_transport_value(
                getattr(http_client, "max_redirects", None)
            ),
            self._canonical_transport_value(
                getattr(http_client, "trust_env", None)
            ),
        )

    @property
    def transport_integrity_ok(self) -> bool:
        try:
            resource = self._client.embeddings
            return bool(
                self._owned_transport_policy
                and OPENAI_TRANSPORT_RUNTIME_EXACT
                and self._closed is False
                and self._transport_policy_sha256
                    == openai_embedding_transport_policy_sha256()
                and self._implementation_guard is not None
                and inspect.getattr_static(
                    type(self), "__getattribute__", None,
                ) is object.__getattribute__
                and inspect.getattr_static(type(self), "__getattr__", None) is None
                and tuple(
                    inspect.getattr_static(type(self), name, None)
                    for name in _OPENAI_EMBEDDING_EXECUTION_ATTRIBUTES
                ) == self._implementation_guard
                and self._client is self._transport_object
                and getattr(self._client, "_client", None)
                is self._http_transport_object
                and resource is self._embedding_resource
                and (
                    not self._embedding_resource_has_client_backref
                    or getattr(resource, "_client", None) is self._client
                )
                and (
                    (
                        self._embedding_create_descriptor is None
                        and getattr(resource, "create", None)
                        is self._embedding_instance_create
                    )
                    or (
                        self._embedding_create_descriptor is not None
                        and "create" not in getattr(resource, "__dict__", {})
                    )
                )
                and inspect.getattr_static(type(resource), "create", None)
                is self._embedding_create_descriptor
                and self._transport_state() == self._transport_guard
            )
        except Exception:
            return False

    def _verify_transport_integrity(self) -> None:
        if not self.transport_integrity_ok:
            raise RuntimeError("embedding transport identity changed")

    @property
    def model(self) -> str:
        """Persisted vector-space identity (provider endpoint + model)."""
        return self._identity

    @property
    def request_model(self) -> str:
        """Model label sent to the remote API."""
        return self._model

    @property
    def endpoint_identity(self) -> dict[str, str]:
        """Credential-free route commitment safe for durable artifacts."""

        return dict(self._endpoint_identity)

    @property
    def deployment_revision_sha256(self) -> str | None:
        """Non-secret commitment required for cross-process vector reuse."""

        return self._deployment_revision_sha256

    @property
    def deployment_tenant_sha256(self) -> str | None:
        """Explicit non-secret tenant/routing commitment, when supplied."""

        return self._deployment_tenant_sha256

    @property
    def transport_policy_sha256(self) -> str:
        return self._transport_policy_sha256

    @property
    def backend(self) -> str:
        return "openai_compatible"

    @property
    def quality(self) -> str:
        return "semantic"

    @property
    def network_free(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        with self._dimension_lock:
            return self._dim

    @property
    def configured_dim(self) -> int:
        """Immutable dimension requested when the transport was constructed."""

        return self._configured_dim

    @property
    def observed_dim(self) -> int | None:
        """Most recently observed non-empty provider vector length, if any."""

        with self._dimension_lock:
            return self._observed_dim

    @property
    def dimension_policy(self) -> str:
        """Whether provider dimensions adapt at runtime or are pinned."""

        return "pinned" if self._pin_dimension else "adaptive"

    @property
    def dimension_integrity_ok(self) -> bool:
        """Latched integrity state used by reusable benchmark builders."""

        with self._dimension_lock:
            return not self._dimension_mismatch_observed

    def _record_observed_dimension(self, dimension: int) -> None:
        with self._dimension_lock:
            self._observed_dim = dimension
            if self._pin_dimension and dimension != self._configured_dim:
                # This is deliberately latched.  Hot-ingest embeddings are a
                # best-effort path and may catch a provider exception; a later
                # successful retry must not let that build publish a reusable
                # receipt after any contradictory vector-space observation.
                self._dimension_mismatch_observed = True

    def _accept_observed_dimension(self, dimension: int) -> None:
        self._record_observed_dimension(dimension)
        if self._pin_dimension and dimension != self._configured_dim:
            raise RuntimeError(
                "embedding provider dimension differs from the pinned "
                "configured dimension"
            )
        with self._dimension_lock:
            self._dim = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Hold the process-only seal across dispatch, response decoding,
        # dimension acceptance, and the final integrity fence. The helper's
        # existing inner acquisition is reentrant and keeps the request path
        # easy to audit without opening a parse-time mutation window.
        with self._transport_lock:
            return self._embed_with_locked_transport(texts)

    def _embed_with_locked_transport(
        self, texts: Sequence[str],
    ) -> list[list[float]]:
        # Keep identity-check + dispatch + post-check one maintained critical
        # section.  The SDK resource used for dispatch is the exact frozen
        # object certified above; resolving ``client.embeddings`` again would
        # reopen a proxy/ABA route between proof and request.
        with self._transport_lock:
            self._verify_transport_integrity()
            payload = list(texts)
            if any(type(text) is not str for text in payload):
                raise TypeError("embedding inputs must be exact strings")
            if not payload:
                return []
            request: dict[str, object] = {
                "model": self._model,
                "input": payload,
            }
            deadline = current_deadline()
            if deadline is not None:
                request["timeout"] = deadline.cap_timeout(
                    self._request_timeout
                )
            with self._usage_lock:
                self.request_attempts += 1
                self.input_count += len(payload)
                self.input_characters += sum(len(text) for text in payload)
            started = time.monotonic()
            try:
                if deadline is not None:
                    deadline.check()
                resp = self._embedding_resource.create(**request)
                # A non-conforming transport may return after its timeout.
                # Never validate/cache/persist that late vector batch.
                if deadline is not None:
                    deadline.check()
                self._verify_transport_integrity()
            except (Exception, DeadlineExceeded):
                with self._usage_lock:
                    self._usage_complete = False
                    self.token_usage_available = False
                raise
            finally:
                with self._usage_lock:
                    self.total_latency_s += time.monotonic() - started
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            valid_usage = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and value >= 0
                for value in (prompt_tokens, total_tokens)
            )
            with self._usage_lock:
                self.call_count += 1
                self.successful_responses += 1
                if valid_usage:
                    self.prompt_tokens += prompt_tokens
                    self.total_tokens += total_tokens
                else:
                    self._usage_complete = False
                self.token_usage_available = (
                    self.successful_responses > 0 and self._usage_complete
                )
            data = list(resp.data)
            if len(data) != len(payload):
                raise RuntimeError(
                    f"embedding provider returned {len(data)} vectors for "
                    f"{len(payload)} inputs"
                )
            indices = [getattr(item, "index", None) for item in data]
            has_any_index = any(index is not None for index in indices)
            has_complete_indices = all(
                isinstance(index, int) and not isinstance(index, bool)
                for index in indices
            )
            if has_any_index and not has_complete_indices:
                raise RuntimeError(
                    "embedding provider returned partial/malformed indices"
                )
            if has_complete_indices:
                if sorted(indices) != list(range(len(payload))):
                    raise RuntimeError(
                        "embedding provider returned invalid response indices"
                    )
                data.sort(key=lambda item: item.index)
            vectors: list[list[float]] = []
            resolved_dim: int | None = None
            for item in data:
                raw = getattr(item, "embedding", None)
                if not isinstance(raw, (list, tuple)) or not raw:
                    raise RuntimeError(
                        "embedding provider returned a malformed vector"
                    )
                if self._pin_dimension:
                    # Length is observable even when numeric decoding later
                    # fails. Record every vector so both orders of a mixed-
                    # shape batch latch a contradiction before the general
                    # shape error.
                    self._record_observed_dimension(len(raw))
                try:
                    vector = [float(value) for value in raw]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        "embedding provider returned a malformed vector"
                    ) from exc
                norm = math.sqrt(sum(value * value for value in vector))
                if (
                    not all(math.isfinite(value) for value in vector)
                    or not math.isfinite(norm) or norm <= 0.0
                ):
                    raise RuntimeError(
                        "embedding provider returned a non-finite/zero vector"
                    )
                if resolved_dim is None:
                    resolved_dim = len(vector)
                elif len(vector) != resolved_dim:
                    self._record_observed_dimension(len(vector))
                    raise RuntimeError(
                        "embedding provider returned mixed vector dimensions"
                    )
                vectors.append(vector)
            if resolved_dim is not None:
                self._accept_observed_dimension(resolved_dim)
            self._verify_transport_integrity()
            return vectors

    def close(self) -> None:
        """Release the SDK's underlying HTTP connection pool exactly once."""

        with self._close_lock:
            if self._closed:
                return
            with self._transport_lock:
                self._closed = True
                with _OPENAI_EMBEDDING_CLOSE_TARGETS_LOCK:
                    target = _OPENAI_EMBEDDING_CLOSE_TARGETS.pop(
                        self, self._client,
                    )
                close = getattr(target, "close", None)
                if callable(close):
                    close()


# Freeze the maintained class surface after its definition, before callers can
# construct an instance.  Capturing it in ``__init__`` would bless a class
# monkeypatch made before construction under the same module-derived producer
# key.  The runtime guard is process-only and is never serialized.
_OPENAI_EMBEDDING_ORIGINAL_IMPLEMENTATION_GUARD = tuple(
    inspect.getattr_static(
        OpenAICompatibleEmbeddingClient, name, None,
    )
    for name in _OPENAI_EMBEDDING_EXECUTION_ATTRIBUTES
)

# Compose independently self-captured import-time commitments.  In particular,
# endpoint_policy may have been loaded long before this module; rereading its
# current disk file here could label old live endpoint code with a new digest.
from hymem.contrib.endpoint_policy import ENDPOINT_POLICY_IMPLEMENTATION_SHA256
from hymem.contrib.implementation_identity import (
    compose_import_time_sha256,
    import_time_source_sha256,
)

OPENAI_EMBEDDING_IMPLEMENTATION_SHA256 = compose_import_time_sha256(
    import_time_source_sha256(__file__),
    ENDPOINT_POLICY_IMPLEMENTATION_SHA256,
)
_OPENAI_EMBEDDING_MAINTAINED_CLASS = OpenAICompatibleEmbeddingClient
