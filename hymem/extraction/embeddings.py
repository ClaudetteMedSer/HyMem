from __future__ import annotations

import hashlib
import math
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
