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

    graph_semantic_top_k: int = 10
    """KNN candidates pulled from vec_edges during semantic graph lookup."""

    graph_predicate_top_k: int = 10
    """Edges pulled per predicate-routed query."""

    graph_top_k: int = 8
    """Final number of GraphFacts returned by augment()."""

    graph_recency_half_life_days: float = 30.0
    """Half-life for edge recency decay: weight = exp(-days_since_last_seen / half_life)."""

    graph_recency_recent_days: float = 7.0
    """Edges with days_since_last_seen <= this emit a recency_Nd reason code."""

    graph_predicate_boost: float = 1.5
    """Score multiplier applied to edges whose predicate matches a routed predicate."""

    graph_entity_match_boost: float = 0.1
    """Additive bonus added to similarity for entity-anchored edges in the
    no-predicate fallback. Keeps entity matches relevant without letting them
    override semantic ranking."""

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
