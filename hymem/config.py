from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HyMemConfig:
    root: Path
    """Directory holding hymem.sqlite, MEMORY.md, USER.md."""

    salience_min_chars: int = 30
    """Minimum chunk size before extraction is attempted."""

    fts_top_k: int = 5
    graph_top_k_per_entity: int = 3
    embedding_max_scan: int = 5000

    decay_window_days: int = 30
    decay_factor: float = 0.9
    retract_threshold: float = 0.15

    profile_max_entries: int = 16
    insights_max_entries: int = 12

    prompt_version: str = "v4"

    dream_budget: int = 50
    """Maximum number of chunks to process per dreaming cycle."""

    max_chunks: int = 50000
    """Soft cap on total stored chunks. Excess unreferenced chunks are pruned."""

    retention_days: int = 90
    """Chunks newer than this are always kept regardless of graph references."""

    rerank_ambiguity_threshold: float = 0.6
    """Minimum RRF score drop between #1/#2 results to consider them clear
    (skip reranking). Higher = more reranking."""

    @property
    def db_path(self) -> Path:
        return self.root / "hymem.sqlite"

    @property
    def memory_md_path(self) -> Path:
        return self.root / "MEMORY.md"

    @property
    def user_md_path(self) -> Path:
        return self.root / "USER.md"
