from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_evidence_role_weights() -> dict[str, int]:
    # Weight a positive evidence row by the role of its chunk's first message.
    # The chunker prefixes a user turn with the preceding assistant turn, so an
    # assistant-prefixed chunk (the common case) keeps weight 1 — unchanged from
    # the historical +1 — while a chunk the *user* opens (an unprompted, self-
    # initiated assertion, not an agreement with possibly-confabulated assistant
    # context) counts double. Roles absent from the map fall back to 1.
    return {"user": 2}


def _default_predicate_half_life_days() -> dict[str, float]:
    # Tiered eligibility windows for phase-3 decay. Longer = decays slower.
    #   ~90d  preference/avoidance — a user's "prefers uv" holds long after the
    #         last mention.
    #   ~60d  structural / dependency — a dependency or composition rarely
    #         changes without an explicit conversation, so it shouldn't accrue
    #         soft-contradiction negatives and get retracted before it's
    #         reinforced.
    # Volatile runtime predicates (uses, runs_on, deploys_to, connects_to,
    # configured_with, ...) are intentionally absent and fall back to the
    # global decay_window_days.
    return {
        "prefers": 90.0,
        "avoids": 90.0,
        "rejects": 90.0,
        "depends_on": 60.0,
        "requires_version": 60.0,
        "part_of": 60.0,
        "implements": 60.0,
    }


@dataclass(frozen=True)
class HyMemConfig:
    root: Path
    """Directory holding hymem.sqlite, MEMORY.md, USER.md."""

    salience_min_chars: int = 30
    """Minimum chunk size before extraction is attempted."""

    # ---- ingest limits & privacy ------------------------------------------
    redact_secrets: bool = True
    """When True, message content is scrubbed for high-confidence secrets
    (API keys, tokens, private keys, credentials in URLs, emails) before it is
    written to SQLite, so the on-disk store never holds the raw value. Chunks
    are derived from already-redacted messages, so this one chokepoint covers
    the whole pipeline. Set False to store verbatim."""

    max_message_chars: int = 100_000
    """Hard cap on a single logged message. Longer content is truncated (with a
    marker appended) before storage so a pathological turn can't bloat the DB or
    stall extraction. 0 disables the cap."""

    max_query_chars: int = 10_000
    """Hard cap on a user message passed to augment(). Longer queries are
    truncated before FTS/embedding so recall latency stays bounded. 0 disables."""

    working_memory_turns: int = 10
    """Number of most-recent raw turns for the active session that augment()
    returns as a working-memory tier, so facts stated this session are
    recallable before any dream runs. 0 disables."""

    fts_top_k: int = 5

    message_fts_top_k: int = 5
    """Number of raw-message keyword hits augment() returns in `message_hits`,
    via a direct FTS5 path over the `messages` table (user/assistant turns
    only). Complements `fts_top_k`, which searches dreamed *chunks*: raw turns
    are searchable the moment they are logged, so facts are recallable across
    sessions and before any dream consolidates them — the gap chunk-FTS leaves.
    0 disables the path."""

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

    predicate_half_life_days: dict[str, float] = field(
        default_factory=_default_predicate_half_life_days
    )
    """Per-predicate eligibility window (in days) for phase-3 decay. An active,
    unreinforced edge is only considered for a negative-evidence bump once it
    hasn't been reinforced for this many days. Sticky predicates (prefers /
    avoids / rejects) use a longer window so they decay slower than volatile
    runtime predicates. Predicates absent from the map fall back to
    `decay_window_days`."""

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

    prompt_version: str = "v8"

    dream_budget: int = 50
    """Maximum number of chunks to process per dreaming cycle."""

    dream_baseline_budget: int = 10
    """If the salience tier leaves budget unspent, drain up to this many
    non-salience-marked chunks (newest first) per cycle. Guarantees every chunk
    eventually flows through extraction even if it didn't trip the regexes."""

    dream_digest_max_tokens: int = 3072
    """max_tokens for the batched per-session digest call (episodes + summary +
    procedures in one JSON object). Larger than the 1024 LLMRequest default
    because the combined output is roughly three responses' worth."""

    dream_digest_max_chars: int = 12000
    """Char cap on the session text fed to the digest call (the larger of the
    pre-batching episode/procedure caps)."""

    max_chunks: int = 50000
    """Soft cap on total stored chunks. Excess unreferenced chunks are pruned."""

    retention_days: int = 90
    """Chunks newer than this are always kept regardless of graph references.
    Also the age window for pruning old episodes and stale procedures."""

    message_retention_days: int = 90
    """Raw messages of a session are pruned once the session is older than this
    AND carries a non-null summary. The summary gate means this is a no-op in
    stub/no-LLM deployments where summaries are never generated, so the only
    copy of unreconstructable data is never destroyed."""

    tombstone_retention_days: int = 30
    """Retracted knowledge_graph edges (and their cascaded kg_evidence) are
    hard-deleted once last_seen is older than this. Active/derived edges are
    untouched (derived edges are rebuilt each cycle)."""

    dream_runs_keep: int = 500
    """Max dream_runs rows retained; older rows are pruned (newest kept)."""

    extraction_feedback_keep: int = 200
    """Max extraction_feedback rows retained (newest kept). Comfortably above
    the 10 the runner injects as negative examples."""

    vacuum_after_prune: bool = True
    """Run VACUUM after a dream cycle whose sweeps deleted rows, to return freed
    pages to the OS (plain DELETE leaves the file size flat)."""

    vacuum_min_pruned: int = 100
    """Minimum rows pruned in a cycle before VACUUM fires, so trivial sweeps
    don't pay the full-rewrite cost."""

    rerank_ambiguity_threshold: float = 0.6
    """Minimum RRF score drop between #1/#2 results to consider them clear
    (skip reranking). Higher = more reranking."""

    rerank_top_k: int = 20
    """Size of the candidate pool sent to the reranker. After rerank the
    list is truncated to ``fts_top_k``. Larger gives the reranker more room
    to reorder beyond the top of the fused list at the cost of latency
    (LLM tokens or cross-encoder forward passes)."""

    rerank_model: str = "llm"
    """Rerank backend. ``"llm"`` reuses the host-provided LLM client (one
    extra request per query). ``"cross-encoder"`` uses a local
    sentence-transformers cross-encoder if installed; otherwise falls back
    to the LLM (or to the un-reranked candidates if no LLM is wired)."""

    rerank_cross_encoder_model: str = "mixedbread-ai/mxbai-rerank-base-v1"
    """HuggingFace model id used when ``rerank_model="cross-encoder"``."""

    hedge_confidence_threshold: float = 0.75
    """Below this Laplace-smoothed confidence, a GraphFact is flagged
    `hedge_recommended` so consumers can soften phrasing
    ("you may use X" vs "you use X")."""

    hedge_min_evidence: int = 3
    """A GraphFact with fewer than this many total evidence rows
    (pos + neg) is flagged `hedge_recommended` regardless of confidence —
    one early extraction shouldn't read as assertive context indefinitely."""

    evidence_role_weights: dict[str, int] = field(
        default_factory=_default_evidence_role_weights
    )
    """Per-role positive-evidence weight, keyed on the role of a chunk's first
    message. Applied when a positive triple bumps `pos_evidence` (phase 1) and
    when co-mention reinforcement fires (phase 3). Roles absent from the map use
    weight 1, so the change is a no-op for assistant-prefixed chunks."""

    triple_dedup_enabled: bool = True
    """When True and an embedding client is available, a brand-new triple is
    checked against existing same-predicate edges by vector similarity before a
    new edge is created (see `triple_dedup_cosine_threshold`)."""

    triple_dedup_cosine_threshold: float = 0.97
    """Cosine-similarity cutoff for `triple_dedup_enabled`. A new
    `(subject, predicate, object)` whose embedded triple text is at least this
    similar to an existing active edge with the *same predicate* attaches its
    evidence to that edge instead of spawning a near-duplicate (e.g. `app uses
    uv` vs `app uses uv_pip`). Kept tight — only collapses genuine siblings."""

    triple_dedup_lexical_ratio: float = 0.85
    """Lexical-sibling guard for dedup, applied *in addition* to the cosine
    threshold. A near-duplicate edge must also share its subject or object with
    the candidate exactly, and the differing entity must be lexically similar —
    share an underscore token, be a substring, or have a difflib ratio at least
    this high. Stops false merges of short, embedding-close-but-distinct names
    (`redis` vs `redash`) that the cosine gate alone would collapse."""

    behavioral_dedup_cosine_threshold: float = 0.90
    """Default cosine cutoff for the *retroactive* behavioral-edge dedup report
    (`HyMem.behavioral_duplicate_report`). Looser than
    `triple_dedup_cosine_threshold` because behavioral objects are paraphrases
    (`concise` / `brevity` / `short answers`) rather than tool-name siblings, and
    the report intentionally drops the lexical gate. Report-only today — no edge
    is merged automatically; a human reviews proposals before any apply step."""

    procedure_stale_confidence_factor: float = 0.5
    """Multiplier applied to a procedure's `confidence` when
    `mark_procedure_stale` flags it. The status flip already removes it from
    `_procedure_search`; the confidence haircut records the negative signal so
    a later re-extraction starts from a discounted prior rather than 1.0.
    Procedures rot faster than triples — a stale runbook is actively
    misleading — so the signal is retained even after the row is hidden."""

    @property
    def db_path(self) -> Path:
        return self.root / "hymem.sqlite"

    @property
    def memory_md_path(self) -> Path:
        return self.root / "MEMORY.md"

    @property
    def user_md_path(self) -> Path:
        return self.root / "USER.md"
