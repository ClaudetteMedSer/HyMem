from __future__ import annotations

import hashlib
import inspect
import math
import re
import threading
import weakref
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, Sequence


def normalize_text(text: str) -> str:
    """Canonical form used by the local lexical feature encoder."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def embedding_text_hash(text: str) -> str:
    """Versioned fingerprint of the exact bytes sent to an embedder.

    Earlier releases hashed ``normalize_text(text)`` while sending ``text``
    unchanged.  Case/whitespace-sensitive providers could therefore reuse a
    vector produced for different input.  The version prefix prevents those
    legacy cache rows from being mistaken for exact-input fingerprints.
    """
    if type(text) is not str:
        raise TypeError("embedding input must be an exact string")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256-exact-input-v1:{digest}"


def _exact_embedding_text_payload(texts: Sequence[str]) -> list[str]:
    """Materialize exactly the immutable strings used for key and dispatch.

    A ``str`` subclass can compare/hash like one value while overriding
    ``__str__`` to render another.  Accepting it would let a cache key describe
    bytes different from those embedded by a maintained provider.  Plain
    strings have one stable Unicode value for both operations.
    """

    payload = list(texts)
    if any(type(text) is not str for text in payload):
        raise TypeError("embedding inputs must be exact strings")
    return payload


class EmbeddingClient(Protocol):
    """Hermes wires whatever embedding backend it wants behind this interface."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    @property
    def model(self) -> str: ...
    @property
    def dim(self) -> int: ...


@dataclass
class LocalHashEmbeddingClient:
    """Small, deterministic, dependency-free embedding fallback.

    This is deliberately identified as a *lexical* feature-hash backend.  It
    gives a default installation a real, finite vector path without starting a
    service, importing a model runtime, or making a network request.  Operators
    with a local/remote semantic model can configure the OpenAI-compatible
    client explicitly; the distinct model id prevents the two vector spaces
    from ever being mixed.

    Word features carry most of the weight while character trigrams provide a
    little robustness to inflection, accents, and typos.  Feature hashing keeps
    memory bounded and makes results byte-for-byte deterministic across Python
    processes (unlike ``hash()``).
    """

    dim_value: int = 384
    model_name: str = "hymem-local-feature-hash-v1"
    fallback_reason: str | None = None

    # Optional observability attributes consumed by HyMem/query status.  They
    # are intentionally not part of EmbeddingClient's minimal protocol.
    backend: str = field(default="local_feature_hash", init=False)
    quality: str = field(default="lexical", init=False)
    network_free: bool = field(default=True, init=False)
    call_count: int = field(default=0, init=False)
    request_attempts: int = field(default=0, init=False)
    successful_responses: int = field(default=0, init=False)
    input_count: int = field(default=0, init=False)
    input_characters: int = field(default=0, init=False)
    total_latency_s: float = field(default=0.0, init=False)
    token_usage_available: bool = field(default=False, init=False)
    _usage_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.dim_value, bool) or not isinstance(self.dim_value, int) or self.dim_value <= 0:
            raise ValueError("local embedding dimension must be a positive integer")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("local embedding model id must be non-empty")

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def dim(self) -> int:
        return self.dim_value

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        normalized = normalize_text(text)
        words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        features: list[tuple[str, float]] = [
            (f"w:{word}", 1.0) for word in words
        ]
        for word in words:
            padded = f"^{word}$"
            features.extend(
                (f"c3:{padded[i:i + 3]}", 0.35)
                for i in range(max(0, len(padded) - 2))
            )
        # Empty/whitespace-only inputs still need a valid non-zero vector so a
        # caller can deterministically abstain/rank instead of receiving NaN.
        return features or [("<empty>", 1.0)]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        started = time.monotonic()
        payload = _exact_embedding_text_payload(texts)
        with self._usage_lock:
            self.call_count += 1
            self.input_count += len(payload)
            self.input_characters += sum(len(text) for text in payload)
        vectors: list[list[float]] = []
        for text in payload:
            vector = [0.0] * self.dim_value
            # Class-qualify the maintained helper.  Attribute dispatch through
            # ``self`` would let an instance/class ``__getattribute__`` hook
            # redirect vector generation while the exact declaration still
            # attested the real ``LocalHashEmbeddingClient._features``.
            for feature, weight in LocalHashEmbeddingClient._features(text):
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.dim_value
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[index] += sign * weight
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
        with self._usage_lock:
            self.total_latency_s += time.monotonic() - started
        return vectors


@dataclass
class StubEmbeddingClient:
    """Deterministic test stub: hashes each text into a fixed-dim normalized vector.

    Cosine similarity between identical strings is 1.0; different strings are
    typically near 0.
    """

    model_name: str = "stub"
    dim_value: int = 16
    calls: list[list[str]] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def dim(self) -> int:
        return self.dim_value

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = _exact_embedding_text_payload(texts)
        out: list[list[float]] = []
        for t in payload:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(h[i % len(h)] / 255.0) - 0.5 for i in range(self.dim_value)]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        self.calls.append(payload)
        return out


class _MappedStubConfig(NamedTuple):
    model: str
    dimension: int
    vectors: tuple[tuple[str, tuple[float, ...]], ...]
    default: tuple[float, ...]
    quality: str
    fail_on: str | None
    connection: object | None


_MAPPED_STUB_CONFIGS: weakref.WeakKeyDictionary[
    object, _MappedStubConfig
] = weakref.WeakKeyDictionary()
_MAPPED_STUB_CONFIGS_LOCK = threading.Lock()


class MappedStubEmbeddingClient:
    """Maintained deterministic mapped-vector stub used by integration tests.

    The execution policy is copied into an immutable module-owned record at
    construction and every call dispatches through that record.  Mutable call
    telemetry is deliberately separate.  This gives semantic integration
    tests a behavior-controlled exact producer without granting arbitrary
    custom objects durable authority merely because they self-declare it.
    """

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]] | None = None,
        *,
        model: str = "semantic-test-v1",
        dim: int = 3,
        default: Sequence[float] | None = None,
        quality: str = "semantic",
        fail_on: str | None = None,
        conn: object | None = None,
    ) -> None:
        if type(model) is not str or not model:
            raise ValueError("mapped stub model must be a non-empty string")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("mapped stub dimension must be positive")
        if quality not in {"semantic", "lexical"}:
            raise ValueError("mapped stub quality is invalid")
        if fail_on is not None and type(fail_on) is not str:
            raise ValueError("mapped stub failure marker must be text")
        source = {} if vectors is None else vectors
        if not isinstance(source, Mapping) or any(
            type(key) is not str for key in source
        ):
            raise ValueError("mapped stub vectors must have string keys")

        def exact_vector(value: Sequence[float]) -> tuple[float, ...]:
            try:
                result = tuple(float(item) for item in value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("mapped stub vector is malformed") from exc
            if len(result) != dim or not all(math.isfinite(item) for item in result):
                raise ValueError("mapped stub vector dimension is malformed")
            return result

        frozen_vectors = tuple(sorted(
            (key, exact_vector(value)) for key, value in source.items()
        ))
        if default is None:
            generated = tuple([0.0, 1.0, 0.0][:dim])
            if len(generated) != dim:
                generated = tuple(1.0 if index == 0 else 0.0 for index in range(dim))
            frozen_default = generated
        else:
            frozen_default = exact_vector(default)
        self.calls: list[list[str]] = []
        self.transaction_states: list[bool] = []
        with _MAPPED_STUB_CONFIGS_LOCK:
            _MAPPED_STUB_CONFIGS[self] = _MappedStubConfig(
                model=model,
                dimension=dim,
                vectors=frozen_vectors,
                default=frozen_default,
                quality=quality,
                fail_on=fail_on,
                connection=conn,
            )

    def _config(self) -> _MappedStubConfig:
        with _MAPPED_STUB_CONFIGS_LOCK:
            config = _MAPPED_STUB_CONFIGS.get(self)
        if config is None:
            raise RuntimeError("mapped stub execution identity is unavailable")
        return config

    @property
    def model(self) -> str:
        return MappedStubEmbeddingClient._config(self).model

    @property
    def dim(self) -> int:
        return MappedStubEmbeddingClient._config(self).dimension

    @property
    def backend(self) -> str:
        return "recording"

    @property
    def quality(self) -> str:
        return MappedStubEmbeddingClient._config(self).quality

    @property
    def network_free(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        config = MappedStubEmbeddingClient._config(self)
        payload = _exact_embedding_text_payload(texts)
        self.calls.append(payload)
        connection = config.connection
        if connection is not None:
            self.transaction_states.append(bool(
                getattr(connection, "in_transaction", False)
            ))
        if config.fail_on is not None and any(
            config.fail_on in text for text in payload
        ):
            raise RuntimeError("provider unavailable")
        mapped = dict(config.vectors)
        return [list(mapped.get(text, config.default)) for text in payload]


def mapped_stub_embedding_config(client: object) -> _MappedStubConfig | None:
    """Return the frozen execution record for one exact maintained stub."""

    if type(client) is not MappedStubEmbeddingClient:
        return None
    with _MAPPED_STUB_CONFIGS_LOCK:
        return _MAPPED_STUB_CONFIGS.get(client)


_MAPPED_STUB_RUNTIME_GUARD = (
    id(_MAPPED_STUB_CONFIGS), id(_MAPPED_STUB_CONFIGS_LOCK),
    mapped_stub_embedding_config,
)


# Freeze module-level helpers that maintained execution dispatches through.
# Class descriptor guards alone cannot observe rebinding a referenced global.
_EMBEDDING_MODULE_HELPER_GUARD = (
    normalize_text,
    _exact_embedding_text_payload,
)


# Capture the maintained dispatch surface at module initialization.  Runtime
# monkeypatches made before a client instance is constructed must not become a
# newly "exact" implementation merely because that instance snapshots the
# already-patched class.  Vector-affecting methods/properties are bound here;
# fixed retrieval/status fields are validated separately at every identity
# boundary.
_MAINTAINED_EMBEDDING_CLASS_GUARDS: dict[type[object], tuple[object, ...]] = {
    LocalHashEmbeddingClient: tuple(
        inspect.getattr_static(LocalHashEmbeddingClient, name, None)
        for name in ("embed", "_features", "model", "dim", "__getattribute__")
    ),
    StubEmbeddingClient: tuple(
        inspect.getattr_static(StubEmbeddingClient, name, None)
        for name in ("embed", "model", "dim", "__getattribute__")
    ),
    MappedStubEmbeddingClient: tuple(
        inspect.getattr_static(MappedStubEmbeddingClient, name, None)
        for name in (
            "embed", "_config", "model", "dim", "backend", "quality",
            "network_free", "__getattribute__",
        )
    ),
}


def maintained_embedding_class_integrity(client: object) -> bool:
    """Return whether an in-tree embedding class retains its frozen dispatch.

    This is a process-integrity check, not a sandbox for hostile concurrent
    code.  It closes stable/non-concurrent instance and class monkeypatches
    before a cache or durable mirror can reuse the maintained producer key.
    """

    cls = type(client)
    expected = _MAINTAINED_EMBEDDING_CLASS_GUARDS.get(cls)
    if expected is None:
        return False
    if cls is LocalHashEmbeddingClient:
        names = ("embed", "_features", "model", "dim", "__getattribute__")
    elif cls is MappedStubEmbeddingClient:
        names = (
            "embed", "_config", "model", "dim", "backend", "quality",
            "network_free", "__getattribute__",
        )
    else:
        names = ("embed", "model", "dim", "__getattribute__")
    try:
        return bool(
            (
                normalize_text,
                _exact_embedding_text_payload,
            ) == _EMBEDDING_MODULE_HELPER_GUARD
            and (
                cls is not MappedStubEmbeddingClient
                or _MAPPED_STUB_RUNTIME_GUARD == (
                    id(_MAPPED_STUB_CONFIGS),
                    id(_MAPPED_STUB_CONFIGS_LOCK),
                    mapped_stub_embedding_config,
                )
            )
            and
            inspect.getattr_static(cls, "__getattribute__", None)
            is object.__getattribute__
            and inspect.getattr_static(cls, "__getattr__", None) is None
            and tuple(
                inspect.getattr_static(cls, name, None) for name in names
            ) == expected
        )
    except Exception:
        return False


class CachedEmbeddingClient:
    """LRU-cache only exact vectors by producer key, dimension, and text.

    A durable producer declaration commits implementation, deployment/route
    attestations, model policy, and dimension; only that exact identity may
    select or populate the cache.  Undeclared/adaptive clients take a fresh
    process-only path on every call and receive no cache or durable-mirror
    authority.  This cuts cold-query latency on repeated user-message embeds
    inside one `augment()` call and across follow-up turns without assuming a
    caller-controlled display model label identifies vector semantics.

    Batch behaviour: split the input into cache hits and misses, forward only
    the misses to the wrapped client, then re-stitch results in input order so
    callers see the same ordering they passed in.

    Thread-safe under a single lock; contention is light because the wrapped
    embedding API call dominates over the cache check.
    """

    def __init__(self, inner: "EmbeddingClient", *, max_size: int = 128) -> None:
        self._inner = inner
        self._max_size = max_size
        self._cache: OrderedDict[
            tuple[str, int, str], list[float]
        ] = OrderedDict()
        self._lock = threading.Lock()
        # Serialize wrapper calls across the identity snapshot, provider call,
        # and cache publication. A dynamic client may update ``dim`` from its
        # first response; concurrent calls must never observe that transition
        # halfway through and label one vector with another call's identity.
        self._call_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._hits = 0
        self._misses = 0
        from hymem.extraction.producer import _register_phase1_producer_proxy

        _register_phase1_producer_proxy(self, inner)

    def _verified_inner(self):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        return _verified_phase1_proxy_delegate(self)

    @property
    def model(self) -> str:
        return self._verified_inner().model

    @property
    def dim(self) -> int:
        return self._verified_inner().dim

    @property
    def backend(self) -> str:
        return str(getattr(self._verified_inner(), "backend", "configured"))

    @property
    def quality(self) -> str:
        return str(getattr(self._verified_inner(), "quality", "semantic"))

    @property
    def network_free(self) -> bool:
        return bool(getattr(self._verified_inner(), "network_free", False))

    @property
    def fallback_reason(self) -> str | None:
        try:
            value = getattr(self._verified_inner(), "fallback_reason", None)
        except Exception:
            return None
        return value if isinstance(value, str) and value else None

    # Benchmark/accounting observability delegates to the actual provider.
    # These are properties (not snapshots) so cumulative usage stays current.
    @property
    def call_count(self):
        return getattr(self._verified_inner(), "call_count", None)

    @property
    def request_attempts(self):
        return getattr(self._verified_inner(), "request_attempts", None)

    @property
    def successful_responses(self):
        return getattr(self._verified_inner(), "successful_responses", None)

    @property
    def input_count(self):
        return getattr(self._verified_inner(), "input_count", None)

    @property
    def input_characters(self):
        return getattr(self._verified_inner(), "input_characters", None)

    @property
    def prompt_tokens(self):
        return getattr(self._verified_inner(), "prompt_tokens", None)

    @property
    def total_tokens(self):
        return getattr(self._verified_inner(), "total_tokens", None)

    @property
    def total_latency_s(self):
        return getattr(self._verified_inner(), "total_latency_s", None)

    @property
    def cost_usd(self):
        return getattr(self._verified_inner(), "cost_usd", None)

    @property
    def token_usage_available(self):
        return bool(getattr(self._verified_inner(), "token_usage_available", False))

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = _exact_embedding_text_payload(texts)
        with self._call_lock:
            with self._close_lock:
                if self._closed:
                    raise RuntimeError("embedding client is closed")
            return self._embed_serialized(payload)

    def close(self) -> None:
        """Close the wrapped provider at most once.

        The call lock keeps an in-flight provider request from racing transport
        teardown.  Marking the wrapper closed before delegating also makes a
        provider close failure terminal: a second lifecycle owner must not
        retry an HTTP-pool close whose partial effects are unknown.
        """

        with self._call_lock:
            with self._close_lock:
                if self._closed:
                    return
                inner = self._verified_inner()
                self._closed = True
            close = getattr(inner, "close", None)
            if callable(close):
                close()

    def _embed_serialized(self, texts: Sequence[str]) -> list[list[float]]:
        texts = _exact_embedding_text_payload(texts)
        if not texts:
            return []
        from hymem.dreaming.aggregation_material import (
            embedding_execution_identity,
            embedding_storage_identity,
        )
        inner = self._verified_inner()

        binding_before, _producer_key, declared_dim = (
            embedding_execution_identity(inner)
        )
        if not binding_before["identity_exact"]:
            # A process nonce distinguishes objects, but it cannot observe a
            # hidden route/implementation switch inside the same arbitrary
            # object.  Therefore an inexact producer gets neither memory-cache
            # hits nor durable reuse authority: every call reaches the live
            # provider and is bracketed only for detectable identity drift.
            fresh = inner.embed(list(texts))
            binding_after, _after_key, final_dim = (
                embedding_execution_identity(inner)
            )
            adaptive_dimension_transition = False
            if final_dim != declared_dim:
                try:
                    from hymem.contrib.openai_embedding_client import (
                        OpenAICompatibleEmbeddingClient,
                    )

                    adaptive_dimension_transition = bool(
                        type(inner) is OpenAICompatibleEmbeddingClient
                        and inner.dimension_policy == "adaptive"
                        and inner.dimension_integrity_ok is True
                        and inner.transport_integrity_ok is True
                    )
                except Exception:
                    adaptive_dimension_transition = False
            if binding_after != binding_before or (
                final_dim != declared_dim and not adaptive_dimension_transition
            ):
                raise RuntimeError(
                    "embedding client identity changed during provider call"
                )
            if len(fresh) != len(texts):
                raise RuntimeError(
                    f"embedding client returned {len(fresh)} vectors for "
                    f"{len(texts)} inputs"
                )
            if final_dim is None:
                raise RuntimeError("embedding client dimension is unavailable")
            validated = [
                self._validated_vector(raw, expected_dim=final_dim)
                for raw in fresh
            ]
            if any(vector is None for vector in validated):
                raise RuntimeError(
                    "embedding client returned a non-finite/zero vector"
                )
            with self._lock:
                self._misses += len(texts)
            return [list(vector) for vector in validated if vector is not None]

        model_id, dim_id = embedding_storage_identity(inner)
        out: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        with self._lock:
            for i, t in enumerate(texts):
                key = (model_id, dim_id, t)
                cached = self._cache.get(key)
                if cached is not None:
                    valid_cached = self._validated_vector(
                        cached, expected_dim=dim_id
                    )
                    if valid_cached is None:
                        # A wrapped dynamic client may have discovered a new
                        # dimension since this entry was cached, or a caller
                        # may have mutated a previously returned list.  Never
                        # replay a malformed/stale cache entry.
                        del self._cache[key]
                        miss_indices.append(i)
                        miss_texts.append(t)
                    else:
                        self._cache.move_to_end(key)
                        out[i] = list(valid_cached)
                        self._hits += 1
                else:
                    miss_indices.append(i)
                    miss_texts.append(t)

        if miss_texts:
            model_before, dim_before = embedding_storage_identity(inner)
            if model_before != model_id or dim_before != dim_id:
                raise RuntimeError(
                    "embedding client identity changed before provider call"
                )
            fresh = inner.embed(miss_texts)
            model_after, dim_after = embedding_storage_identity(inner)
            if len(fresh) != len(miss_texts):
                raise RuntimeError(
                    f"embedding client returned {len(fresh)} vectors for "
                    f"{len(miss_texts)} inputs"
                )
            if (
                model_after != model_before
                or isinstance(dim_after, bool)
                or not isinstance(dim_after, int)
                or dim_after <= 0
            ):
                raise RuntimeError("embedding client changed/invalidated its identity")
            validated: list[list[float]] = []
            for raw in fresh:
                vector = self._validated_vector(raw, expected_dim=dim_after)
                if vector is None:
                    raise RuntimeError("embedding client returned a non-finite/zero vector")
                validated.append(vector)
            if dim_after != dim_before and any(value is not None for value in out):
                # Cached hits were admitted under the old declaration. Do not
                # mix them with the newly discovered vector space; clear and
                # let the next call retry under the corrected identity.
                with self._lock:
                    self._cache.clear()
                raise RuntimeError("embedding dimension changed with cached batch hits")
            with self._lock:
                for idx, t, vec in zip(miss_indices, miss_texts, validated):
                    out[idx] = vec
                    self._cache[(model_after, dim_after, t)] = list(vec)
                    self._cache.move_to_end((model_after, dim_after, t))
                    self._misses += 1
                    while len(self._cache) > self._max_size:
                        self._cache.popitem(last=False)

        if any(vector is None for vector in out):
            raise RuntimeError("embedding cache failed to preserve batch cardinality")
        model_final, dim_final = embedding_storage_identity(inner)
        expected_final_dim = dim_after if miss_texts else dim_id
        if model_final != model_id or dim_final != expected_final_dim:
            with self._lock:
                self._cache.clear()
            raise RuntimeError("embedding client identity changed during cache operation")
        # Return fresh lists so a caller cannot corrupt a cached vector by
        # mutating the object it received.
        return [list(vector) for vector in out if vector is not None]

    @staticmethod
    def _validated_vector(
        raw: object, *, expected_dim: int
    ) -> list[float] | None:
        if (
            isinstance(expected_dim, bool)
            or not isinstance(expected_dim, int)
            or expected_dim <= 0
            or not isinstance(raw, (list, tuple))
            or len(raw) != expected_dim
        ):
            return None
        try:
            vector = [float(value) for value in raw]
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in vector):
            return None
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if math.isfinite(norm) and norm > 0.0 else None


class PinnedEmbeddingClient:
    """Maintained transparent wrapper for strict dimension/lifecycle fencing.

    ``error_type`` changes only the public exception vocabulary; the vector
    producer remains the exact invoked inner client.  Keeping this class in
    the core module gives proxy identity an exact, module-initialized type and
    dispatch guard instead of trusting an importable registration helper.
    """

    def __init__(
        self, inner: object, *, expected_dimension: int,
        owned_transport: object | None = None,
        error_type: type[Exception] = RuntimeError,
    ) -> None:
        if not isinstance(error_type, type) or not issubclass(error_type, Exception):
            raise TypeError("embedding wrapper error_type must be an exception type")
        self._inner = inner
        self._owned_transport = owned_transport
        self._owned_transport_target = owned_transport
        self._closed = False
        self._lifecycle_lock = threading.RLock()
        self._error_type = error_type
        self.expected_dimension = int(expected_dimension)
        self._initial_model = getattr(inner, "model")
        from hymem.extraction.producer import _register_phase1_producer_proxy

        _register_phase1_producer_proxy(self, inner)
        with _PINNED_CLOSE_TARGETS_LOCK:
            _PINNED_CLOSE_TARGETS[self] = owned_transport

    def __getattr__(self, name: str):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        return getattr(_verified_phase1_proxy_delegate(self), name)

    @property
    def model(self):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        return getattr(_verified_phase1_proxy_delegate(self), "model")

    @property
    def dim(self):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        return getattr(_verified_phase1_proxy_delegate(self), "dim")

    def _failure(self, message: str) -> Exception:
        return self._error_type(message)

    def embed(self, texts):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        with self._lifecycle_lock:
            if self._closed:
                raise self._failure("embedding client is closed")
            if (
                self.model != self._initial_model
                or self.dim != self.expected_dimension
                or getattr(
                    _verified_phase1_proxy_delegate(self),
                    "dimension_integrity_ok", True,
                ) is not True
            ):
                raise self._failure(
                    "embedding client identity differs from manifested model/dimension"
                )
            inner = _verified_phase1_proxy_delegate(self)
            vectors = inner.embed(texts)
            if (
                self.model != self._initial_model
                or self.dim != self.expected_dimension
                or getattr(inner, "dimension_integrity_ok", True) is not True
                or any(
                    not isinstance(vector, (list, tuple))
                    or len(vector) != self.expected_dimension
                    for vector in vectors
                )
            ):
                raise self._failure(
                    "embedding provider returned a dimension/identity that differs "
                    "from the benchmark manifest"
                )
            return vectors

    def close(self) -> None:
        """Close only the transport whose ownership was explicitly transferred."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            with _PINNED_CLOSE_TARGETS_LOCK:
                target = _PINNED_CLOSE_TARGETS.pop(self, None)
            close = getattr(target, "close", None)
            if callable(close):
                close()


_PINNED_CLOSE_TARGETS: weakref.WeakKeyDictionary[object, object] = (
    weakref.WeakKeyDictionary()
)
_PINNED_CLOSE_TARGETS_LOCK = threading.Lock()


_CACHED_EMBEDDING_GUARD_NAMES = (
    "embed", "_embed_serialized", "_validated_vector", "_verified_inner",
    "close", "model",
    "dim", "backend", "quality", "network_free", "fallback_reason",
    "__getattribute__", "__getattr__",
)
_CACHED_EMBEDDING_ORIGINAL_GUARD = tuple(
    inspect.getattr_static(CachedEmbeddingClient, name, None)
    for name in _CACHED_EMBEDDING_GUARD_NAMES
)

_PINNED_EMBEDDING_GUARD_NAMES = (
    "embed", "complete", "model", "dim", "backend", "quality",
    "network_free", "close", "__getattribute__", "__getattr__",
    "_embed_serialized", "_validated_vector", "_failure",
)
_PINNED_EMBEDDING_ORIGINAL_GUARD = tuple(
    inspect.getattr_static(PinnedEmbeddingClient, name, None)
    for name in _PINNED_EMBEDDING_GUARD_NAMES
)


def cached_embedding_proxy_integrity(client: object) -> bool:
    """Prove the maintained cache still transparently invokes its inner.

    Durable producer identity belongs to the inner client.  We may unwrap the
    cache only while its invocation, validation, metadata delegation, and
    lifecycle dispatch retain the module-initialized descriptors and no
    per-instance method/property shadows have been installed.
    """

    if type(client) is not CachedEmbeddingClient:
        return False
    try:
        state = object.__getattribute__(client, "__dict__")
        return bool(
            (
                normalize_text,
                _exact_embedding_text_payload,
            ) == _EMBEDDING_MODULE_HELPER_GUARD
            and
            object.__getattribute__(client, "_closed") is False
            and object.__getattribute__(client, "_inner") is not client
            and not any(name in state for name in _CACHED_EMBEDDING_GUARD_NAMES)
            and tuple(
                inspect.getattr_static(CachedEmbeddingClient, name, None)
                for name in _CACHED_EMBEDDING_GUARD_NAMES
            ) == _CACHED_EMBEDDING_ORIGINAL_GUARD
        )
    except (AttributeError, TypeError):
        return False


# These commitments are captured while the owning module is imported, at the
# same lifecycle boundary that creates the executable class objects above.
# Re-reading mutable source files later could otherwise label already-loaded
# OLD bytecode with a NEW on-disk implementation digest during a rolling
# deploy, allowing cross-process cache reuse between different algorithms.
from hymem.extraction.producer import canonical_callable_sha256 as _callable_sha256

LOCAL_HASH_EMBEDDING_IMPLEMENTATION_SHA256 = _callable_sha256(
    LocalHashEmbeddingClient._features,
    LocalHashEmbeddingClient.embed,
    inspect.getattr_static(LocalHashEmbeddingClient, "model").fget,
    inspect.getattr_static(LocalHashEmbeddingClient, "dim").fget,
    normalize_text,
    _exact_embedding_text_payload,
)
STUB_EMBEDDING_IMPLEMENTATION_SHA256 = _callable_sha256(
    StubEmbeddingClient.embed,
    inspect.getattr_static(StubEmbeddingClient, "model").fget,
    inspect.getattr_static(StubEmbeddingClient, "dim").fget,
    _exact_embedding_text_payload,
)
MAPPED_STUB_EMBEDDING_IMPLEMENTATION_SHA256 = _callable_sha256(
    MappedStubEmbeddingClient.__init__, MappedStubEmbeddingClient._config,
    MappedStubEmbeddingClient.embed,
    inspect.getattr_static(MappedStubEmbeddingClient, "model").fget,
    inspect.getattr_static(MappedStubEmbeddingClient, "dim").fget,
    mapped_stub_embedding_config,
    _exact_embedding_text_payload,
)

_LOCAL_HASH_EMBEDDING_MAINTAINED_CLASS = LocalHashEmbeddingClient
_STUB_EMBEDDING_MAINTAINED_CLASS = StubEmbeddingClient
_MAPPED_STUB_EMBEDDING_MAINTAINED_CLASS = MappedStubEmbeddingClient
_CACHED_EMBEDDING_MAINTAINED_CLASS = CachedEmbeddingClient
_PINNED_EMBEDDING_MAINTAINED_CLASS = PinnedEmbeddingClient
