from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, Sequence


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
    if not isinstance(text, str):
        raise TypeError("embedding input must be text")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256-exact-input-v1:{digest}"


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
        payload = list(texts)
        with self._usage_lock:
            self.call_count += 1
            self.input_count += len(payload)
            self.input_characters += sum(len(str(text)) for text in payload)
        vectors: list[list[float]] = []
        for text in payload:
            vector = [0.0] * self.dim_value
            for feature, weight in self._features(str(text)):
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
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(h[i % len(h)] / 255.0) - 0.5 for i in range(self.dim_value)]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        self.calls.append(list(texts))
        return out


class CachedEmbeddingClient:
    """Wraps an EmbeddingClient with an LRU cache keyed on (model, text).

    Embeddings are pure functions of (model, text), so the cache needs no TTL:
    changing the embedding model produces a different key, and a model change
    requires re-embedding the corpus anyway (vec table dimension drift is
    caught by hymem-doctor). Cuts cold-query latency on repeated user-message
    embeds inside one `augment()` call (Source 2 KNN + chunk vector search
    share one embed) and across follow-up turns within a session.

    Batch behaviour: split the input into cache hits and misses, forward only
    the misses to the wrapped client, then re-stitch results in input order so
    callers see the same ordering they passed in.

    Thread-safe under a single lock; contention is light because the wrapped
    embedding API call dominates over the cache check.
    """

    def __init__(self, inner: "EmbeddingClient", *, max_size: int = 128) -> None:
        self._inner = inner
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._lock = threading.Lock()
        # Serialize wrapper calls across the identity snapshot, provider call,
        # and cache publication. A dynamic client may update ``dim`` from its
        # first response; concurrent calls must never observe that transition
        # halfway through and label one vector with another call's identity.
        self._call_lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def backend(self) -> str:
        return str(getattr(self._inner, "backend", "configured"))

    @property
    def quality(self) -> str:
        return str(getattr(self._inner, "quality", "semantic"))

    @property
    def network_free(self) -> bool:
        return bool(getattr(self._inner, "network_free", False))

    @property
    def fallback_reason(self) -> str | None:
        try:
            value = getattr(self._inner, "fallback_reason", None)
        except Exception:
            return None
        return value if isinstance(value, str) and value else None

    # Benchmark/accounting observability delegates to the actual provider.
    # These are properties (not snapshots) so cumulative usage stays current.
    @property
    def call_count(self):
        return getattr(self._inner, "call_count", None)

    @property
    def request_attempts(self):
        return getattr(self._inner, "request_attempts", None)

    @property
    def successful_responses(self):
        return getattr(self._inner, "successful_responses", None)

    @property
    def input_count(self):
        return getattr(self._inner, "input_count", None)

    @property
    def input_characters(self):
        return getattr(self._inner, "input_characters", None)

    @property
    def prompt_tokens(self):
        return getattr(self._inner, "prompt_tokens", None)

    @property
    def total_tokens(self):
        return getattr(self._inner, "total_tokens", None)

    @property
    def total_latency_s(self):
        return getattr(self._inner, "total_latency_s", None)

    @property
    def cost_usd(self):
        return getattr(self._inner, "cost_usd", None)

    @property
    def token_usage_available(self):
        return bool(getattr(self._inner, "token_usage_available", False))

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with self._call_lock:
            return self._embed_serialized(texts)

    def _embed_serialized(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model_id = self._inner.model
        dim_id = self._inner.dim
        if (
            not isinstance(model_id, str) or not model_id
            or isinstance(dim_id, bool) or not isinstance(dim_id, int)
            or dim_id <= 0
        ):
            raise RuntimeError("embedding client has an invalid identity")
        out: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        with self._lock:
            for i, t in enumerate(texts):
                key = (model_id, t)
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
            model_before = self._inner.model
            dim_before = self._inner.dim
            if model_before != model_id or dim_before != dim_id:
                raise RuntimeError(
                    "embedding client identity changed before provider call"
                )
            fresh = self._inner.embed(miss_texts)
            model_after = self._inner.model
            dim_after = self._inner.dim
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
                    self._cache[(model_after, t)] = list(vec)
                    self._cache.move_to_end((model_after, t))
                    self._misses += 1
                    while len(self._cache) > self._max_size:
                        self._cache.popitem(last=False)

        if any(vector is None for vector in out):
            raise RuntimeError("embedding cache failed to preserve batch cardinality")
        model_final = self._inner.model
        dim_final = self._inner.dim
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
