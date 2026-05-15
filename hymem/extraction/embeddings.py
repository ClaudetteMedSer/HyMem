from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, Sequence


class EmbeddingClient(Protocol):
    """Hermes wires whatever embedding backend it wants behind this interface."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    @property
    def model(self) -> str: ...
    @property
    def dim(self) -> int: ...


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
        self._hits = 0
        self._misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model_id = self._inner.model
        out: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        with self._lock:
            for i, t in enumerate(texts):
                key = (model_id, t)
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    out[i] = cached
                    self._hits += 1
                else:
                    miss_indices.append(i)
                    miss_texts.append(t)

        if miss_texts:
            fresh = self._inner.embed(miss_texts)
            with self._lock:
                for idx, t, vec in zip(miss_indices, miss_texts, fresh):
                    out[idx] = vec
                    self._cache[(model_id, t)] = vec
                    self._cache.move_to_end((model_id, t))
                    self._misses += 1
                    while len(self._cache) > self._max_size:
                        self._cache.popitem(last=False)

        # Every slot is filled by construction.
        return [v for v in out if v is not None]
