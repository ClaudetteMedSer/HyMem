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

    graph_token_overlap_weight: float = 0.5
    """Score multiplier for entity-anchored edges in the fallback path when the
    anchoring entity was reached only via token-overlap expansion (and not via
    direct entity match or type expansion). Keeps fuzzy entity links present
    without letting them outrank direct hits."""

    graph_token_overlap_threshold: int = 20
    """A token segment shared by more than this many canonicals is considered
    too common to drive token-overlap expansion (e.g. `system`, `service`)."""

    graph_token_overlap_max_per_entity: int = 5
    """Max token-overlap expansions allowed per matched canonical."""

    decay_window_days: int = 30
    decay_factor: float = 0.9
    retract_threshold: float = 0.15

    zombie_neg_threshold: int = 2
    """Negative-dominance offset in the auto-retract rule
    `neg_evidence >= 2 * pos_evidence + zombie_neg_threshold`. At pos=0 this
    reduces to `neg >= threshold` (catches classic zombies); at pos=1 it
    fires at neg=threshold+2 (catches edges where one positive is buried
    under many negatives). Keep small; raising shields more edges from
    retraction."""

    reinforce_window_days: int = 30
    """Window for soft positive reinforcement from co-mention. Symmetric to
    decay_window_days."""

    profile_max_entries: int = 16
    insights_max_entries: int = 12

    prompt_version: str = "v6"

    dream_budget: int = 50
    """Maximum number of chunks to process per dreaming cycle."""

    dream_baseline_budget: int = 10
    """If the salience tier leaves budget unspent, drain up to this many
    non-salience-marked chunks (newest first) per cycle. Guarantees every chunk
    eventually flows through extraction even if it didn't trip the regexes."""

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
