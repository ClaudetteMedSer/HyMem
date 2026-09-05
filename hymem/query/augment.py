from __future__ import annotations

import json
import heapq
import logging
import math
import re
import sqlite3
import struct
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field, replace

from hymem.config import HyMemConfig
from hymem.core.graph import graph_clock_order_sql, live_edge_predicate
from hymem.core.vectors import decode_vector
from hymem.dreaming.aggregate import Digest, load_digest
from hymem.dreaming.aggregation_provenance import (
    BoundSourceOccurrence,
    load_aggregation_source_manifest,
)
from hymem.dreaming.lossless import (
    COVERAGE_VALIDATION_COLUMNS,
    COVERAGE_VALIDATION_JOINS,
    validate_message_coverage_row,
)
from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
from hymem.dreaming.facts import load_fact_source_manifests
from hymem.dreaming.user_profile import ProfileEntry, load_profile
from hymem.extraction.embeddings import EmbeddingClient, embedding_text_hash
from hymem.extraction.llm import LLMClient
from hymem.query.coref import QueryRewrite, rewrite_query
from hymem.query.entities import GraphCount, count_relations, match_known_entities
from hymem.query.fusion import (
    FusedEvidence,
    PackedContext,
    SourceOccurrence,
    enrich_context_provenance,
    fuse_context,
    scope_context_in_place,
)
from hymem.query.graph_state import (
    GraphEvidenceCitation,
    _current_authoritative_evidence,
    current_positive_citations,
    current_positive_state,
    validated_confidence_signal_totals,
    validated_current_evidence,
)
from hymem.query.intent import detect_ability_signal
from hymem.query.predicate_routing import route_predicates
from hymem.query.rerank import rerank as run_rerank
from hymem.rules import Rule, load_rules
from hymem.session import Message, recent_messages

log = logging.getLogger("hymem.query.augment")
_QUERY_VECTOR_UNSET = object()


@dataclass
class GraphFact:
    subject: str
    predicate: str
    object: str
    confidence: float
    pos_evidence: int
    neg_evidence: int
    derived: bool = False
    why_retrieved: list[str] = field(default_factory=list)
    score: float = 0.0
    hedge_recommended: bool = False
    """Set by `_graph_lookup` from `cfg.hedge_confidence_threshold` and
    `cfg.hedge_min_evidence`. Hermes (or any consumer) reads this to decide
    whether to soften phrasing — HyMem only signals, never rewrites the
    fact text."""
    edge_id: int | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    citations: list[GraphEvidenceCitation] = field(default_factory=list)
    """Bounded exact source records. Empty means provenance is unavailable;
    consumers must never synthesize an author or session in that case."""


def format_graph_fact_sources(fact: GraphFact) -> str:
    """Render bounded graph provenance without inventing missing identities."""
    if not fact.citations:
        return "source unavailable"
    labels: list[str] = []
    for citation in fact.citations:
        role = citation.source_role or "unavailable"
        session = citation.source_session_id or "unavailable"
        message = (
            str(citation.source_message_id)
            if citation.source_message_id is not None
            else "unavailable"
        )
        event_at = citation.source_event_at or "unavailable"
        peer = citation.source_peer_id or "unavailable"
        workspace = citation.source_workspace_id or "unavailable"
        labels.append(
            f"evidence {citation.evidence_id}: peer={peer}, workspace={workspace}, "
            f"role={role}, session={session}, "
            f"message={message}, event={event_at}"
        )
    return "; ".join(labels)


@dataclass
class EpisodeHit:
    episode_id: str
    session_id: str
    title: str
    summary: str
    score: float
    score_kind: str = "bm25"
    """Source of the score: "bm25" (FTS only), "vec" (semantic only), or
    "rrf" (reciprocal-rank-fused FTS + vec)."""
    why_retrieved: list[str] = field(default_factory=list)
    """Short reason chips (e.g. `episode_fts("postgres pool")`,
    `episode_rrf(fts+vec, 0.0240)`) mirroring `GraphFact.why_retrieved`, so a
    consumer can quote why an episode surfaced instead of guessing."""
    source_occurrences: tuple[SourceOccurrence, ...] = ()
    """Exact source turns behind this summary; empty when unavailable."""
    source_provenance_complete: bool = False


@dataclass
class FactHit:
    """One authoritative narrative fact (schema v46) — a self-contained
    one-sentence statement extracted at dream time, the middle granularity
    between a knowledge-graph triple and an episode summary. `fact_date` is an
    explicit ISO date the conversation wrote, or None (undated facts stay
    undated — never a session-date fallback). `entities` are canonical ids.
    Numeric start/end ids are descriptive only. ``source_occurrences`` carries
    the exact lossless proof; facts with an incomplete/corrupt proof or a
    non-current lifecycle projection never reach this DTO."""

    fact_id: int
    text: str
    fact_date: str | None
    entities: list[str]
    session_id: str
    score: float
    score_kind: str = "bm25"
    why_retrieved: list[str] = field(default_factory=list)
    """Short reason chips (e.g. `fact_fts("fly deploy")`,
    `fact_rrf(fts+vec, 0.0240)`), mirroring the other tiers."""
    source_occurrences: tuple[SourceOccurrence, ...] = ()
    source_provenance_complete: bool = False


@dataclass
class AggregationNodeHit:
    """A Phase-2 RAPTOR cross-session aggregation node — one fused summary over a
    cluster of episodes that spanned several sessions. Surfaced as an ADDITIVE
    tier (only when `cfg.aggregation_nodes_enabled`) so a multi-session synthesis
    question can read the through-line from one summary instead of re-fusing
    dozens of raw turns. `member_episode_ids`/`session_ids` let the host trace
    the node back to its sources."""

    node_id: str
    title: str
    summary: str
    member_episode_ids: list[str]
    session_ids: list[str]
    score: float
    score_kind: str = "bm25"
    why_retrieved: list[str] = field(default_factory=list)
    source_occurrences: tuple[SourceOccurrence, ...] = ()
    """Union of exact turns behind the node's published member episodes."""
    source_provenance_complete: bool = False


@dataclass
class FtsHit:
    chunk_id: str
    session_id: str
    text: str
    score: float
    score_kind: str = "bm25"
    why_retrieved: list[str] = field(default_factory=list)
    """Short reason chips (e.g. `fts_match("postgres pool")`,
    `vec_topk(sim=0.82)`, `rrf(fts+vec, 0.0240)`, `reranked`)."""
    source_occurrences: tuple[SourceOccurrence, ...] = ()
    """Authoritative manifest occurrences, or validated range fallback."""
    source_provenance_complete: bool = False


@dataclass
class MessageHit:
    """An exact message occurrence surfaced by lexical or semantic retrieval.

    Live rows are searched through ``messages``/FTS; the durable coverage
    corpus preserves the same occurrence after opt-in raw pruning.

    Distinct from `FtsHit` (which carries dreamed *chunk* text) on purpose: the
    two are different granularities and their BM25 scores aren't comparable, so
    they stay in separate lists and the host weaves them together. `created_at`
    and `message_id` are exposed so a consumer can prefer the most recent
    statement when a fact was updated (BM25 alone is recency-blind)."""

    message_id: int
    session_id: str
    role: str
    text: str
    score: float
    created_at: str = ""
    score_kind: str = "bm25"
    why_retrieved: list[str] = field(default_factory=list)
    """Short reason chips, mirroring `FtsHit` (e.g. `message_fts("postgres pool")`)."""
    enumerates_items: bool = False
    """True only on aggregate (MR) hits whose turn enumerates *several* items in
    one message ("a shirt, jeans and boots"). The distinct-turn count treats such
    a turn as one event, so this flag tells the host LLM that this turn under-
    counts the true item-count and should be re-read. False for non-aggregate
    hits and for plain single-item turns."""
    source_peer_id: str | None = None
    source_workspace_id: str | None = None
    source_occurrences: tuple[SourceOccurrence, ...] = ()


@dataclass(frozen=True)
class SemanticStatus:
    """Observable state of the one query-embedding attempt.

    ``quality`` distinguishes the dependency-free lexical feature hash from a
    configured semantic model. Failures never suppress lexical retrieval.
    """

    configured: bool = False
    attempted: bool = False
    available: bool = False
    backend: str = "none"
    quality: str = "none"
    model: str | None = None
    dim: int | None = None
    reason: str = "no_embedding_client"
    fallback_reason: str | None = None


@dataclass
class ProcedureHit:
    procedure_id: str
    session_id: str
    name: str
    description: str
    steps: list[dict]
    score: float
    why_retrieved: list[str] = field(default_factory=list)
    """Short reason chips (e.g. `procedure_fts("deploy staging")`)."""


@dataclass
class TemporalEvent:
    """One dated event for the temporal-reasoning (TR) path, already normalized
    to a sortable date so the host LLM reads a chronology instead of hunting for
    dates in retrieval noise.

    `date` is ISO `YYYY-MM-DD` for fully-resolved dates; for a year-less message
    mention it falls back to the date portion of the source turn's event time
    (`created_at`) so the event still has a sort key. `source` names where the
    event came from: "message" (an explicit date written in a turn), "graph" (a
    direct knowledge-graph edge with a source-valid ``valid_at``), or
    "session-date" (the turn's `created_at` for a matched turn that carried no
    content-date — a *when-discussed* anchor, NOT necessarily when-it-happened,
    so the host must not use it for duration math the way a content-date is).
    Only *dated* items appear here; undated graph facts stay in `graph_facts`."""

    date: str
    text: str
    source: str
    why_retrieved: list[str] = field(default_factory=list)


@dataclass
class AugmentedContext:
    """Structured context for the host (Hermes) to assemble into its prompt.

    Hermes decides ordering, headers, and token budget — HyMem only returns
    the pieces. This keeps prompt assembly out of the memory module.

    `fts_hits[i].score` carries different units depending on `score_kind`:
        - "bm25": SQLite FTS5 BM25 score (lower = better, often negative)
        - "rrf":  reciprocal rank fusion score from FTS+vector merge (higher = better)

    `recent_turns` is the working-memory tier: the last N raw turns of the
    active session, included so the host can surface within-session facts that
    have not yet been consolidated by dreaming. It is populated only when a
    `session_id` is passed to `augment()`; otherwise it stays empty.

    `message_hits` is the raw-message hybrid tier: lexical hits plus semantic
    scoring over exact durable user/assistant coverage occurrences. Unlike
    `recent_turns` it is query-relevant and can span sessions; unlike
    `fts_hits` it reaches turns that dreaming never promoted to extraction
    chunks and survives opt-in raw pruning.
    """

    user_md: str = ""
    memory_md: str = ""
    graph_facts: list[GraphFact] = field(default_factory=list)
    """Current knowledge-graph facts selected for this query.

    This is a declared part of the context contract (rather than an attribute
    attached by ``augment``) so empty/pre-retrieval contexts serialize and
    introspect exactly like populated ones.
    """
    fts_hits: list[FtsHit] = field(default_factory=list)
    message_hits: list[MessageHit] = field(default_factory=list)
    semantic_status: SemanticStatus = field(default_factory=SemanticStatus)
    """Backend identity and outcome for the single query-vector attempt."""
    total_message_matches: int = 0
    """Candidate count for "how many X" questions when `augment()` ran in
    aggregation mode (`ability="MR"`); 0 otherwise. It is the number of distinct
    **user** turns matching the query after conservative restatement dedup —
    assistant echoes are excluded so an action isn't double-counted. It is a
    *candidate* answer, not gospel: one turn may state several items (or none),
    so the host's LLM verifies it against `message_hits` (the matching turns,
    chronological). The count stays exact even when `message_hits` is capped at
    `message_fts_aggregate_cap` (so `total_message_matches >= len(message_hits)`)."""
    enumeration_turns: int = 0
    """Of the matching MR turns, how many enumerate *several* items in one message
    ("a shirt, jeans and boots"). Populated only under `ability="MR"`; 0 otherwise.

    This is the over-count provenance the aggregate path can offer WITHOUT entity
    typing (a graph-native typed count was ruled out: the type vocabulary is tech-
    only, so consumer categories like clothing/food never get typed). When this is
    0, the one-turn-one-item assumption holds and `total_message_matches` is the
    answer. When nonzero, that many turns each state multiple items, so the true
    count is HIGHER than `total_message_matches`; the host LLM should re-read the
    `message_hits` carrying `enumerates_items=True` and tally items, not turns.
    Computed over the full deduped match set, so it stays exact under the cap."""
    graph_count: GraphCount | None = None
    """EXACT graph-native count for the IN-DOMAIN slice of an `ability="MR"`
    question, or None. Populated only when the query maps cleanly onto the
    in-vocab type/predicate/entity machinery (see `hymem.query.count_routing`):
    "how many services depend on redis?" → distinct typed subjects;
    "how many databases do we use?" → distinct typed objects. None when the query
    is out-of-vocab, un-typed, or ambiguous, or when neither edge orientation
    yields any match (a zero-with-no-evidence result is suppressed rather than
    surfaced as a misleading "exact 0") — the cases the graph cannot answer.

    Precedence contract for the host: when `graph_count` is present it is the
    EXACT, dedup-correct answer (a `COUNT(DISTINCT ...)` over active edges) and
    should be preferred over `total_message_matches`, which is only the lexical
    *candidate* (it counts raw user turns and can over/under-count). The two
    coexist on purpose: the keyword candidate always runs as the fallback for
    out-of-vocab / consumer-domain / un-typed questions where `graph_count` is
    None, and it also lets the host cross-check the exact number against the
    matching turns. `graph_count.counted` names which side was tallied so the host
    can phrase the answer unambiguously."""
    episodes: list[EpisodeHit] = field(default_factory=list)
    facts: list[FactHit] = field(default_factory=list)
    """Current proof-valid narrative facts (schema v46), retrieved by an
    active-authority-only FTS shadow plus optional vectors and RRF. The native
    tier has its own candidate budget; final fusion/dedup and token packing are
    intentionally shared, so source-equivalent facts do not duplicate a turn
    and lower-priority evidence cannot claim unlimited prompt space. Retracted
    facts remain in immutable lifecycle history but leave search. Empty when
    disabled, capped at zero, or authority cannot be proven."""
    aggregation_nodes: list[AggregationNodeHit] = field(default_factory=list)
    """Phase-2 RAPTOR cross-session cluster summaries (see `AggregationNodeHit`).
    Empty unless `cfg.aggregation_nodes_enabled` is set AND the dream has built
    nodes — an additive tier that never displaces episode/chunk/message hits."""
    procedures: list[ProcedureHit] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    recent_turns: list[Message] = field(default_factory=list)
    user_profile: list[ProfileEntry] = field(default_factory=list)
    """ACTIVE typed user-profile rows (schema v18) — the small, always-relevant
    identity tier (name, role, employer, location, relationship(person), ...)
    extracted from USER turns under a closed slot vocabulary. ADDITIVE like the
    aggregation tier: it is loaded by a standalone SELECT and never consumes a
    slot from message/chunk/episode/graph retrieval, so a populated profile
    cannot crowd gold turns out of any other tier. Empty until a dream has
    extracted profile facts, when `cfg.profile_extraction_enabled` is False, or
    on a pre-v18 store. Capped at `cfg.profile_context_cap`, identity slots
    first; values are already redaction-scrubbed at persist time."""
    rules: list[Rule] = field(default_factory=list)
    """ACTIVE `always_on` Rules (schema v23) — standing behavioral imperatives
    ("always run the tests before pushing", "never suggest Docker") loaded into
    EVERY call when `cfg.rules_enabled` is set. `always_on` rules load
    unconditionally; `contextual` rules load only when a `trigger_entities`
    member overlaps `matched_entities`. ADDITIVE like the profile/aggregation
    tiers: a standalone SELECT (`hymem.rules.load_rules`) that never consumes a
    slot from any retrieval tier. Capped at `cfg.rules_context_cap`, `always_on`
    first. Empty when `rules_enabled` is False or on a pre-v23 store. Values are
    redaction-scrubbed at persist time. Deliberately NOT in the RAPTOR digest
    anchor (additional_planning.md §0)."""
    digest: Digest | None = None
    """The standing whole-store root digest (see `HyMem.digest()`), populated
    only when `cfg.augment_include_digest` is True — the Stage-5 convenience
    for single-call hosts that cannot make a separate `digest()` call. None by
    default: the digest is standing, dream-time context, so most hosts should
    fetch it once per dream rather than on every augment. Like the profile and
    aggregation tiers it is purely additive — loaded by its own SELECT, never
    a retrieval competitor, cannot crowd any other tier. Render it with
    `digest.as_context_block()` for the canonical staleness-stamped form."""
    temporal_events: list[TemporalEvent] = field(default_factory=list)
    """Dated events in chronological (date-ascending) order, populated only when
    `augment()` runs with `ability="TR"`. It merges explicit dates extracted from
    raw messages (`temporal_mentions`) with current direct knowledge-graph edges
    carrying lifecycle-derived source ``valid_at`` (never ingestion time), so a
    temporal-reasoning question ("how
    long between X and Y?", "what happened first?") receives a ready-made
    timeline. Only dated items appear; undated graph facts stay in `graph_facts`.
    Empty for every other ability and for DBs without the temporal index."""
    detected_ability: str | None = None
    """The ability HyMem *inferred* from the query text when the host supplied no
    explicit `ability` hint (None when the host supplied one, or when nothing was
    inferred). Provenance only — it mirrors the `why_retrieved` chips' role of
    making a routing decision debuggable: it lets a consumer see that, e.g., a
    plain "how many cards did I add?" was auto-routed to the MR aggregate path
    without the caller passing `ability="MR"`. An *explicit* host hint always
    wins and leaves this None, so host-supplied and inferred shaping stay
    distinguishable."""
    coref: QueryRewrite | None = None
    """E5 provenance: the anaphora/ellipsis rewrite applied to the query before
    ANY retrieval tier ran (`hymem/query/coref.py`), or None when no rewrite was
    attempted (no `session_id`, or `cfg.coref_enabled` is False). Populated even
    when nothing changed — `coref.changed` is False and `coref.rule` names the
    abstain reason ("self_contained", "no_turns", "no_referent"), so "why did
    this follow-up retrieve nothing?" is answerable from the result object alone.
    Same observability contract as `detected_rule`. The rewrite only ever APPENDS
    resolved referents, so every tier still saw all of the original tokens."""
    detected_rule: str | None = None
    """WHICH router rule produced `detected_ability` (e.g. "tr_recency",
    "mr_count"), or the abstain reason ("none"/"empty"/"non_str") when nothing was
    inferred. None when the host supplied an explicit `ability` hint (no inference
    ran). This is the observability half of the routing decision: `detected_ability`
    says WHAT was inferred, `detected_rule` says WHY — so a production misroute
    ("why did this plain question get MR-shaped?") is diagnosable from the result
    object alone, without re-running the patterns. See `detect_ability_signal`."""
    fused_evidence: list[FusedEvidence] = field(default_factory=list)
    """Additive global rank/dedup view; native tier lists stay available."""
    packed_context: PackedContext | None = None
    """Most recent item-level render, including budget/truncation metadata."""
    retrieval_query: str = ""
    """Exact post-coreference query used to center presentation excerpts."""
    fusion_source_session_id: str | None = None
    fusion_source_peer_id: str | None = None
    fusion_source_workspace_id: str | None = None
    """Ownership boundary reapplied whenever the additive fusion is rebuilt."""
    count_message_hits: list[MessageHit] = field(default_factory=list)
    """Bounded exact turns supporting ``total_message_matches``.

    Kept separately from relevance ``message_hits`` so prompt packing can make
    the candidate count conditional on at least one actual counted turn while
    retaining both sources additively.
    """


# Ability hints a host (e.g. a BEAM harness) may pass to shape retrieval to the
# question type. The label is known to the host, not inferred here. Only "MR"
# (aggregation) is wired in Phase 1; the rest are reserved for follow-up shaping.
_ABILITIES = frozenset({"IE", "MR", "TR", "SUM", "IF", "KU"})


def _normalize_ability(ability: str | None) -> str | None:
    """Uppercase + validate an ability hint; unknown/empty values become None so
    callers degrade to the default (un-shaped) retrieval path."""
    if not ability:
        return None
    normalized = ability.strip().upper()
    return normalized if normalized in _ABILITIES else None


def augment(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    user_message: str,
    *,
    embedding_client: EmbeddingClient | None = None,
    llm: LLMClient | None = None,
    token_overlap_index: dict[str, list[str]] | None = None,
    session_id: str | None = None,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
    ability: str | None = None,
) -> AugmentedContext:
    if source_peer_id is not None and source_workspace_id is None:
        raise ValueError(
            "source_workspace_id is required when source_peer_id is provided"
        )
    scoped_request = any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    )
    # The explicit host hint always wins (the host knows the question type). Only
    # when it is None/absent/unknown do we infer one from the query text, so the
    # MR/TR shaping that the BEAM harness gets from a ground-truth label also
    # fires in real conversations where no label is passed. The inferred code is
    # re-validated through the same `_ABILITIES` gate so it can never widen the
    # accepted set. `detected_ability` records the inference for debuggability
    # (left None when the host supplied the hint, so the two stay distinguishable).
    ability = _normalize_ability(ability)
    detected: str | None = None
    detected_rule: str | None = None
    if ability is None:
        signal = detect_ability_signal(user_message)
        detected_rule = signal.rule
        detected = _normalize_ability(signal.ability)
        ability = detected
        if detected is not None:
            log.debug("ability.auto_detected=%s rule=%s (no host hint)",
                      detected, detected_rule)
    ctx = AugmentedContext()
    ctx.detected_ability = detected
    ctx.detected_rule = detected_rule
    if cfg.user_md_path.exists():
        ctx.user_md = cfg.user_md_path.read_text(encoding="utf-8")
    if cfg.memory_md_path.exists():
        ctx.memory_md = cfg.memory_md_path.read_text(encoding="utf-8")

    # Working-memory tier: the last N raw turns of the active session, so facts
    # stated this session are recallable before any dream has consolidated them.
    # `conn` here is the READ connection; recent_messages is a plain SELECT.
    if session_id is not None and cfg.working_memory_turns > 0:
        ctx.recent_turns = recent_messages(conn, session_id, cfg.working_memory_turns)
        if scoped_request:
            ctx.recent_turns = [
                message for message in ctx.recent_turns
                if (
                    (source_session_id is None
                     or message.session_id == source_session_id)
                    and (source_peer_id is None
                         or message.source_peer_id == source_peer_id)
                    and (source_workspace_id is None
                         or message.source_workspace_id == source_workspace_id)
                )
            ]

    # E5 anaphora/ellipsis resolution — the ONLY place the query text is
    # rewritten, and it happens BEFORE every retrieval tier so all of them
    # benefit from one fix (a pronoun-only follow-up defeats BM25, vectors,
    # entity matching and predicate routing simultaneously). The rewrite APPENDS
    # the resolved referents and never replaces, so each tier below still sees
    # every original token — the additive invariant at the query level.
    #
    # Ability detection deliberately ran on the ORIGINAL text above: the router's
    # patterns are about question SHAPE ("how many", "how long between"), which a
    # referent clause cannot change but could confuse.
    #
    # Needs a session to have any antecedent, so it is inert when the host passes
    # no `session_id`. `ctx.recent_turns` is reused when the working-memory window
    # already covers `coref_max_turns` (the default 10 ≥ 6), so the common path
    # adds no second SELECT.
    query = user_message
    if cfg.coref_enabled and session_id is not None and cfg.coref_max_turns > 0:
        coref_turns = (
            ctx.recent_turns[-cfg.coref_max_turns:]
            if len(ctx.recent_turns) >= cfg.coref_max_turns
            else recent_messages(conn, session_id, cfg.coref_max_turns)
        )
        if scoped_request:
            coref_turns = [
                message for message in coref_turns
                if (
                    (source_session_id is None
                     or message.session_id == source_session_id)
                    and (source_peer_id is None
                         or message.source_peer_id == source_peer_id)
                    and (source_workspace_id is None
                         or message.source_workspace_id == source_workspace_id)
                )
            ]
        ctx.coref = rewrite_query(
            user_message, coref_turns, cfg=cfg, conn=conn, llm=llm
        )
        if ctx.coref.changed:
            query = ctx.coref.rewritten
            log.debug("coref.applied rule=%s query=%r", ctx.coref.rule, query)

    # One validated query vector is shared by every semantic tier below.  A
    # provider or shape failure is recorded and all lexical tiers continue.
    query_vector: list[float] | None = None
    if embedding_client is not None:
        query_vector, ctx.semantic_status = _query_embedding_with_status(
            embedding_client, query
        )
    # A failed/abstained one-shot query embedding is a hard semantic stop for
    # this augment call.  In particular, do not let individual tiers inspect
    # optional client metadata after the central guard has already found it to
    # be malformed: custom ``model``/``dim``/observability properties may
    # themselves raise.  Lexical retrieval remains fully available below.
    semantic_client = embedding_client if query_vector is not None else None

    # P4 typed user-profile tier: always-relevant identity facts, loaded by its
    # own SELECT so it is purely ADDITIVE — no other tier's top-k budget is
    # touched (mirrors how the TR/aggregation tiers layer on). load_profile
    # degrades to [] on a pre-v18 store.
    if cfg.profile_extraction_enabled:
        ctx.user_profile = load_profile(conn, cap=cfg.profile_context_cap)

    # Stage-5 single-call convenience: ship the standing root digest alongside
    # the per-query tiers. Off by default — the digest only changes at dream
    # time, so most hosts fetch it once via HyMem.digest() instead of here.
    if cfg.augment_include_digest:
        ctx.digest = load_digest(conn)

    # Pull a wider candidate pool when reranking is likely so the reranker
    # has room to reorder beyond the top-fts_top_k window; the final result
    # is still trimmed to fts_top_k after rerank.
    candidate_k = max(cfg.fts_top_k, cfg.rerank_top_k)
    fts = _fts_search(
        conn, query, top_k=candidate_k,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    vec: list[FtsHit] = []
    if semantic_client is not None:
        vec = _vector_search(
            conn,
            semantic_client,
            query,
            top_k=candidate_k,
            max_scan=cfg.embedding_max_scan,
            query_vector=query_vector,
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
        )
        ctx.fts_hits = _rrf_merge(fts, vec, top_k=candidate_k)
    else:
        ctx.fts_hits = fts

    if scoped_request:
        # Composite text must cross the exact-manifest + cryptographic coverage
        # boundary before any LLM/cross-encoder reranker can observe it. The
        # same validation runs again at final assembly because the context DTO
        # remains mutable by design.
        enrich_context_provenance(conn, ctx)
        scope_context_in_place(
            ctx,
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
        )
        safe_chunk_ids = {hit.chunk_id for hit in ctx.fts_hits}
        fts = [hit for hit in fts if hit.chunk_id in safe_chunk_ids]
        vec = [hit for hit in vec if hit.chunk_id in safe_chunk_ids]

    rerank_enabled = (
        cfg.rerank_model == "cross-encoder" or llm is not None
    )
    if rerank_enabled and should_rerank(fts, vec, ctx.fts_hits, cfg.rerank_ambiguity_threshold):
        log.debug("rerank.triggered model=%s", cfg.rerank_model)
        ctx.fts_hits = run_rerank(
            query,
            list(ctx.fts_hits[: cfg.rerank_top_k]),
            top_k=cfg.fts_top_k,
            model=cfg.rerank_model,
            llm=llm,
            cross_encoder_model=cfg.rerank_cross_encoder_model,
        )
        # rerank() reorders via dataclasses.replace, preserving why_retrieved;
        # tag the survivors so the chip trail shows the rerank step. New list
        # per hit to avoid mutating the (shared) pre-rerank chip list.
        for hit in ctx.fts_hits:
            if hit.score_kind == "reranked":
                hit.why_retrieved = [*hit.why_retrieved, "reranked"]
    else:
        log.debug("rerank.skipped")
        ctx.fts_hits = ctx.fts_hits[: cfg.fts_top_k]

    # Raw-message hybrid tier: FTS5 plus durable occurrence vectors, reaching
    # turns that were never extraction-chunked or whose raw rows were pruned.
    # Kept separate from fts_hits — different granularity, non-comparable BM25.
    def _relevance_message_hits() -> list[MessageHit]:
        # This raw-message tier is the dominant recovery source (most gold turns
        # come back here, not via dreamed chunks), yet it was historically
        # returned in raw BM25 keyword order — so a turn that is the right answer
        # but only a weak lexical match sat below the cut. Mirror the chunk tier:
        # pull a wider candidate pool, then rerank it down to message_fts_top_k so
        # a semantically-relevant turn in the BM25 tail can be lifted into view.
        msg_candidate_k = (
            max(cfg.message_fts_top_k, cfg.rerank_top_k)
            if (rerank_enabled and cfg.rerank_message_hits)
            else cfg.message_fts_top_k
        )
        # Workspace-scoped Honcho traffic is covered atomically at ingestion,
        # so its durable corpus is the one authoritative occurrence index.
        # This prevents raw and retained BM25 pools from crowding one another
        # before the caller's limit and keeps results stable across pruning.
        if source_workspace_id is not None:
            msg_hits = _coverage_message_fts_search(
                conn,
                query,
                top_k=msg_candidate_k,
                source_session_id=source_session_id,
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
        else:
            msg_hits = _message_fts_search(
                conn,
                query,
                top_k=msg_candidate_k,
                source_session_id=source_session_id,
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
        if source_workspace_id is None and any(
            value is not None
            for value in (
                source_session_id, source_peer_id, source_workspace_id,
            )
        ):
            durable_hits = _coverage_message_fts_search(
                conn,
                query,
                top_k=max(msg_candidate_k, 1),
                source_session_id=source_session_id,
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
            seen_occurrences = {
                (hit.session_id, hit.message_id) for hit in msg_hits
            }
            msg_hits.extend(
                hit for hit in durable_hits
                if (hit.session_id, hit.message_id) not in seen_occurrences
            )
            msg_hits.sort(key=lambda hit: (hit.score, hit.message_id))
        semantic_hits = _coverage_message_vector_search(
            conn,
            query,
            top_k=msg_candidate_k,
            embedding_client=semantic_client,
            query_vector=query_vector,
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
        )
        msg_hits = _merge_message_lexical_semantic(
            msg_hits, semantic_hits, top_k=msg_candidate_k
        )
        # Only pay the rerank call when it can actually change the surviving set
        # (pool deeper than the cut). A pool at/below the cut would only reorder
        # turns the host already sees, not lift a new one in — not worth a call.
        if (
            rerank_enabled
            and cfg.rerank_message_hits
            and len(msg_hits) > cfg.message_fts_top_k
        ):
            log.debug("rerank.message.triggered model=%s", cfg.rerank_model)
            msg_hits = run_rerank(
                query,
                list(msg_hits),
                top_k=cfg.message_fts_top_k,
                model=cfg.rerank_model,
                llm=llm,
                cross_encoder_model=cfg.rerank_cross_encoder_model,
            )
            for hit in msg_hits:
                if hit.score_kind == "reranked":
                    hit.why_retrieved = [*hit.why_retrieved, "reranked"]
        else:
            msg_hits = msg_hits[: cfg.message_fts_top_k]
        return msg_hits

    # ability="MR" layers a deterministic candidate count (distinct deduped user
    # turns) onto the tier. The crucial design choice is REPLACE vs ADDITIVE:
    #  * legacy (mr_aggregate_additive=False): the aggregate OWNS message_hits, so
    #    a *mis-routed* count question (the router over-fires "how many X" on
    #    single-session lookups it can't distinguish from text) loses relevance
    #    retrieval entirely — the dominant real-world cost of MR over-detection.
    #  * additive (default): relevance retrieval still runs and fills message_hits;
    #    the COUNT (total_message_matches + the exact in-domain graph_count) is
    #    layered ON TOP. A false-positive MR detection then costs nothing — the
    #    question still gets full (reranked) retrieval — while a genuine count
    #    question keeps its number. This mirrors how TR is already additive, and
    #    sidesteps the un-fixable text ambiguity ("how many books have I read" is
    #    a real count in production, a lookup on this benchmark — indistinguishable
    #    by regex, so precision can't be tightened without cutting real MR recall).
    mr_aggregate = ability == "MR" and cfg.message_fts_aggregate_cap > 0
    if mr_aggregate and not cfg.mr_aggregate_additive:
        (
            ctx.count_message_hits,
            ctx.total_message_matches,
            ctx.enumeration_turns,
        ) = _message_fts_aggregate(
            conn, query, cap=cfg.message_fts_aggregate_cap,
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
        )
        ctx.message_hits = list(ctx.count_message_hits)
        ctx.graph_count = _maybe_graph_count(conn, query)
    else:
        if cfg.message_fts_top_k > 0:
            ctx.message_hits = _relevance_message_hits()
        if mr_aggregate:
            # Count layered on top of relevance retrieval (message_hits already
            # set above). The graph gives an EXACT typed count for the in-domain
            # slice; any failure leaves graph_count=None and the keyword candidate
            # count stands. Its bounded exact evidence turns are retained in
            # ``count_message_hits`` alongside relevance hits, so packing can
            # keep a candidate count coherent with at least one counted source.
            (
                ctx.count_message_hits,
                ctx.total_message_matches,
                ctx.enumeration_turns,
            ) = _message_fts_aggregate(
                conn, query, cap=cfg.message_fts_aggregate_cap,
                source_session_id=source_session_id,
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
            ctx.graph_count = _maybe_graph_count(conn, query)

    # ability="TR" (temporal reasoning) builds a date-ordered event list so the
    # host LLM reads a chronology instead of finding dates in noise. It merges
    # explicit dates extracted from raw messages (temporal_mentions) with dated
    # knowledge-graph edges, ordered date-ascending. Only populated for TR; the
    # other tiers above (graph/fts/messages) still run unchanged.
    if ability == "TR" and not scoped_request:
        ctx.temporal_events = _temporal_events(
            conn, query, ctx.message_hits, ctx.fts_hits, top_k=cfg.fts_top_k
        )

    if not scoped_request:
        ctx.episodes = _episode_search(
            conn, query,
            top_k=cfg.fts_top_k,
            embedding_client=semantic_client,
            query_vector=query_vector,
        )

    # Schema-v46 narrative facts: a separate native candidate budget feeds the
    # shared final fusion/packing stage. Only active, exact-manifest authority
    # crosses search/provider hooks; retracted history stays audit-only.
    if cfg.facts_enabled and cfg.facts_top_k > 0:
        ctx.facts = _fact_search(
            conn, query,
            top_k=cfg.facts_top_k,
            embedding_client=semantic_client,
            query_vector=query_vector,
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
        )

    # Phase-2 RAPTOR additive tier: cross-session cluster summaries. Off by
    # default; only runs when the layer is enabled AND the routed ability is in
    # `aggregation_inject_abilities` (default TR-only — the G4 A/B showed broad
    # injection reshuffles ranking against gold message hits everywhere except
    # temporal reasoning). Never displaces the tiers above — it layers a
    # synthesis view on top.
    # Stage 4a adds a STRICT-OR second condition: a starved query (raw signal
    # below `aggregation_fallback_min_hits`) fires the tier too, on the licence
    # that nodes cannot crowd what is not there. It only ever ADDS firings —
    # `_aggregation_tier_fires` is evaluated unchanged and is never relaxed.
    if cfg.aggregation_nodes_enabled:
        by_ability = _aggregation_tier_fires(cfg, ability)
        by_fallback = _sparse_signal_fires(cfg, ctx)
        if by_ability or by_fallback:
            ctx.aggregation_nodes = _aggregation_search(
                conn, query,
                top_k=cfg.aggregation_top_k,
                embedding_client=semantic_client,
                query_vector=query_vector,
                max_scan=cfg.embedding_max_scan,
                source_session_id=source_session_id,
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
            # Provenance: chip ONLY the firings the fallback actually caused.
            # An ability-gated firing on a thin query is still an ability
            # firing — the two modes have completely different expected
            # effects, and without the split no later A/B can separate them.
            if by_fallback and not by_ability:
                chip = f"sparse_fallback(raw={_raw_signal_count(ctx)})"
                ctx.aggregation_nodes = [
                    replace(h, why_retrieved=[*h.why_retrieved, chip])
                    for h in ctx.aggregation_nodes
                ]

    # ability="IF" (instruction/step recall) pulls a wider procedure set, since
    # procedures — ordered step-by-step workflows — are the natural fit for
    # "what steps did I take to implement X?" The host still decides ordering.
    proc_top_k = cfg.procedure_top_k_if if ability == "IF" else cfg.fts_top_k
    if not scoped_request:
        ctx.procedures = _procedure_search(conn, query, top_k=proc_top_k)

    matched = match_known_entities(conn, query)
    type_expanded, expansion_info = _expand_entities_by_type(conn, matched)
    # Free-text type/property expansion: the user may ask "what build tools
    # do we use?" without naming any specific entity. Map type/property
    # keywords in the message to canonicals tagged with that type or
    # property; merge into the entity set so Source 1 of the graph lookup
    # picks them up.
    query_type_expanded, query_type_info = _expand_entities_from_query(
        conn, query
    )
    overlap_expanded, overlap_info = _expand_entities_by_token_overlap(
        conn, matched,
        max_per_entity=cfg.graph_token_overlap_max_per_entity,
        common_token_threshold=cfg.graph_token_overlap_threshold,
        token_index=token_overlap_index,
    )
    combined = list(type_expanded)
    for e in query_type_expanded:
        if e not in combined:
            combined.append(e)
        # Surface the type label that justified the addition through the same
        # `entity_type:` reason channel as direct-entity type expansion.
        expansion_info.setdefault(e, query_type_info[e])
    for e in overlap_expanded:
        if e not in combined:
            combined.append(e)
    ctx.matched_entities = combined

    # Idea B `always_on` Rules tier — loaded here (after matched_entities is
    # resolved) so `contextual` rules can gate on entity overlap; `always_on`
    # rules inject unconditionally. Its own SELECT, so purely additive — no
    # other tier's budget is touched. Degrades to [] on a pre-v23 store.
    if cfg.rules_enabled:
        ctx.rules = load_rules(conn, ctx.matched_entities, cap=cfg.rules_context_cap)

    routed = route_predicates(query)
    ctx.graph_facts = _graph_lookup(
        conn, cfg, query, ctx.matched_entities, expansion_info, routed,
        overlap_info=overlap_info,
        embedding_client=semantic_client,
        query_vector=query_vector,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    if any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    ):
        # Global alias/type topology has no peer/workspace provenance. Expose
        # only literal endpoint matches from facts that survived exact scoped
        # authority, so unrelated workspace data cannot leak through metadata.
        ctx.matched_entities = sorted({
            endpoint
            for fact in ctx.graph_facts
            for endpoint, marker in (
                (fact.subject, "entity_match:subject"),
                (fact.object, "entity_match:object"),
            )
            if marker in fact.why_retrieved
            or _query_mentions_canonical(query, endpoint)
        })
    # Add exact source manifests/ranges only after all native tier queries have
    # finished, then build the additive global view.  This never changes the
    # tier DTO ordering or performs model work; it gives hosts one calibrated,
    # occurrence-deduped stream without taking the drill-down API away.
    ctx.retrieval_query = query
    ctx.fusion_source_session_id = source_session_id
    ctx.fusion_source_peer_id = source_peer_id
    ctx.fusion_source_workspace_id = source_workspace_id
    enrich_context_provenance(conn, ctx)
    scope_context_in_place(
        ctx,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    ctx.aggregation_nodes = ctx.aggregation_nodes[:cfg.aggregation_top_k]
    ctx.fused_evidence = fuse_context(
        ctx,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    return ctx


def should_rerank(
    fts_hits: list[FtsHit],
    vec_hits: list[FtsHit],
    fused: list[FtsHit],
    threshold: float,
) -> bool:
    if not fts_hits or not vec_hits:
        return False

    if fts_hits and vec_hits:
        if fts_hits[0].chunk_id == vec_hits[0].chunk_id:
            return False

    if len(fused) < 2:
        return False

    score_1 = fused[0].score
    score_2 = fused[1].score
    if score_1 <= 0:
        return False
    drop = 1.0 - (score_2 / score_1)
    if drop > threshold:
        return False

    return True


_FTS_SAFE = re.compile(r"[^A-Za-z0-9_\- ]+")


def _fold_diacritics(text: str) -> str:
    """Strip combining diacritics so a query token matches the FTS index.

    The FTS5 `unicode61` tokenizer folds diacritics when it builds the index
    ("café" is stored as "cafe"), but `_FTS_SAFE` below is an ASCII-only
    whitelist — so without this step it would SHRED an accented query token
    before it ever reaches FTS ("café" → "caf", "coördinatie" → "co rdinatie"),
    and the Dutch/loanword query would silently match nothing while the index
    holds the folded form. NFKD splits each precomposed letter into base + mark;
    dropping the combining marks (`unicodedata.combining`) yields the same ASCII
    base the index stored, so query and index agree. This covers the full Dutch
    diacritic set (ë ï ö é ü á è …); precomposed Latin letters with no canonical
    decomposition (ø ß æ — not used in Dutch) are left for the ASCII strip and
    are a known out-of-scope residual."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def _fts_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[FtsHit]:
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    if not cleaned:
        return []
    # Build an OR query across tokens so partial matches still surface results.
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

    scope_sql, scope_params = _chunk_manifest_scope_sql(
        "c",
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    chip = f'fts_match("{" ".join(tokens)}")'
    scoped = any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    )
    batch_size = max(32, max(1, top_k) * 2) if scoped else max(0, top_k)
    if batch_size <= 0:
        return []
    offset = 0
    accepted: list[FtsHit] = []
    # A malformed store must not turn one scoped request into an unbounded
    # validation walk.  The cap counts raw candidates, while ``top_k`` counts
    # only candidates that pass the exact manifest + coverage proof.
    raw_scan_cap = max(1024, top_k * 64)
    scanned = 0
    while len(accepted) < top_k and scanned < raw_scan_cap:
        try:
            # bm25() is an FTS5 built-in; schema.sql declares chunks_fts with
            # fts5. Scoped reads page past superficially valid but corrupt
            # manifests; only cryptographically reconstructed hits are accepted.
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.session_id, c.text,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ? AND c.chunk_kind = 'extraction'
                      {scope_sql}
                ORDER BY score, c.id
                LIMIT ? OFFSET ?
                """,
                (
                    fts_query, *scope_params,
                    min(batch_size, raw_scan_cap - scanned), offset,
                ),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            break
        batch = [
            FtsHit(
                chunk_id=row["chunk_id"],
                session_id=row["session_id"],
                text=row["text"],
                score=float(row["score"]),
                why_retrieved=[chip],
            )
            for row in rows
        ]
        if scoped:
            temporary = AugmentedContext(fts_hits=batch)
            enrich_context_provenance(conn, temporary)
            scope_context_in_place(
                temporary,
                source_session_id=source_session_id,
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
            batch = temporary.fts_hits
        accepted.extend(batch)
        offset += len(rows)
        scanned += len(rows)
        if len(rows) < batch_size:
            break
    if scoped and len(accepted) < top_k and scanned >= raw_scan_cap:
        log.warning(
            "augment.scoped_fts_validation_scan_exhausted "
            "accepted=%d top_k=%d scanned=%d",
            len(accepted), top_k, scanned,
        )
    return accepted[:top_k]


def _message_scope_sql(
    *,
    source_session_id: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
) -> tuple[str, tuple[object, ...]]:
    """Build an author/session predicate for raw-message candidate SQL.

    The predicate is interpolated only from static fragments and is applied
    before BM25 ordering and LIMIT.  This prevents high-scoring messages from
    another peer/session from filling the candidate pool and producing a
    false-empty scoped result after post-filtering.
    """
    clauses: list[str] = []
    params: list[object] = []
    if source_session_id is not None:
        clauses.append("m.session_id = ?")
        params.append(source_session_id)
    if source_workspace_id is not None:
        clauses.append("m.source_workspace_id = ?")
        params.append(source_workspace_id)
    if source_peer_id is not None:
        clauses.append("m.source_peer_id = ?")
        params.append(source_peer_id)
    return (
        (" AND " + " AND ".join(clauses)) if clauses else "",
        tuple(params),
    )


def _derived_scope_sql(
    alias: str,
    start_column: str,
    end_column: str,
    *,
    source_session_id: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
) -> tuple[str, tuple[object, ...]]:
    """Pre-rank ownership filter for composite artifacts.

    Every turn in the artifact's actual range must carry the requested external
    ownership; post-query provenance validation still fails closed on damaged
    proofs. Applying this before ORDER/LIMIT prevents another workspace's high
    BM25 rows from crowding out an eligible scoped result.
    """
    clauses: list[str] = []
    params: list[object] = []
    if source_session_id is not None:
        clauses.append(f"{alias}.session_id = ?")
        params.append(source_session_id)
    for column, value in (
        ("source_workspace_id", source_workspace_id),
        ("source_peer_id", source_peer_id),
    ):
        if value is None:
            continue
        range_sql = (
            "mc.source_session_id = " + alias + ".session_id "
            f"AND mc.message_id BETWEEN {alias}.{start_column} "
            f"AND {alias}.{end_column}"
        )
        clauses.append(
            f"EXISTS (SELECT 1 FROM message_retention_coverage mc "
            f"WHERE {range_sql} AND mc.{column} = ?)"
        )
        params.append(value)
        clauses.append(
            f"NOT EXISTS (SELECT 1 FROM message_retention_coverage mc "
            f"WHERE {range_sql} AND mc.{column} IS NOT ?)"
        )
        params.append(value)
    return (
        (" AND " + " AND ".join(clauses)) if clauses else "",
        tuple(params),
    )


def _chunk_manifest_scope_sql(
    alias: str,
    *,
    source_session_id: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
) -> tuple[str, tuple[object, ...]]:
    """Pre-rank composite ownership using exact manifest membership.

    Numeric ranges are metadata, not source manifests: skipped message ids and
    multi-author turns make a BETWEEN predicate both over- and under-inclusive.
    This SQL gate keeps wrong-scope candidates from occupying LIMIT slots;
    cryptographic coverage validation and exact text reconstruction run before
    any candidate text can reach a reranker.
    """
    if not any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    ):
        return "", ()
    clauses = [
        f"{alias}.source_manifest_version = 'claim-source-manifest-v1'",
        f"{alias}.source_manifest_count > 0",
        "(SELECT COUNT(*) FROM chunk_message_sources cms "
        f"WHERE cms.chunk_id={alias}.id)={alias}.source_manifest_count",
        "NOT EXISTS (SELECT 1 FROM chunk_message_sources cms "
        f"WHERE cms.chunk_id={alias}.id "
        f"AND cms.source_session_id IS NOT {alias}.session_id)",
    ]
    params: list[object] = []
    if source_session_id is not None:
        clauses.append(f"{alias}.session_id = ?")
        params.append(source_session_id)
    for column, value in (
        ("source_workspace_id", source_workspace_id),
        ("source_peer_id", source_peer_id),
    ):
        if value is None:
            continue
        clauses.append(
            "NOT EXISTS ("
            "SELECT 1 FROM chunk_message_sources cms "
            "LEFT JOIN message_retention_coverage mc "
            "ON mc.message_id=cms.source_message_id "
            "AND mc.source_session_id=cms.source_session_id "
            "AND mc.chunk_id=cms.source_coverage_chunk_id "
            "AND mc.coverage_version=cms.source_coverage_version "
            f"WHERE cms.chunk_id={alias}.id "
            f"AND (mc.message_id IS NULL OR mc.{column} IS NOT ?)"
            ")"
        )
        params.append(value)
    return " AND " + " AND ".join(clauses), tuple(params)


def _message_fts_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[MessageHit]:
    """Direct BM25 keyword search over raw `messages` (user/assistant turns).

    Mirrors `_fts_search` but targets `messages_fts` instead of `chunks_fts`, so
    it reaches turns dreaming never chunked. Returns [] (not an error) if the
    table is absent — e.g. a DB migrated by older code — so retrieval degrades to
    chunk-FTS rather than failing."""
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    if not cleaned:
        return []
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

    scope_sql, scope_params = _message_scope_sql(
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    try:
        rows = conn.execute(
            f"""
            SELECT m.id, m.session_id, m.role, m.content, m.created_at,
                   m.source_peer_id, m.source_workspace_id,
                   bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            WHERE messages_fts MATCH ? {scope_sql}
            ORDER BY score, m.id
            LIMIT ?
            """,
            (fts_query, *scope_params, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    chip = f'message_fts("{" ".join(tokens)}")'
    return [
        MessageHit(
            message_id=int(r["id"]),
            session_id=r["session_id"],
            role=r["role"],
            text=r["content"],
            score=float(r["score"]),
            created_at=r["created_at"] or "",
            source_peer_id=r["source_peer_id"],
            source_workspace_id=r["source_workspace_id"],
            why_retrieved=[chip],
        )
        for r in rows
    ]


def _coverage_message_fts_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[MessageHit]:
    """Search exact retained messages after their raw rows have been pruned.

    Coverage chunks are canonical one-message JSONL artifacts indexed in a
    dedicated corpus. The author/session predicate is applied in SQL before
    the bounded candidate scan; every survivor is then independently validated
    before its decoded content can reach a caller.
    """
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    tokens = [token for token in cleaned.split() if len(token) >= 2]
    if not tokens or top_k <= 0:
        return []
    fts_query = " OR ".join(f'"{token}"' for token in tokens)
    clauses = [
        "message_coverage_fts MATCH ?",
        "c.chunk_kind = 'coverage'",
        "mc.coverage_version = ?",
        "mc.source_role IN ('user', 'assistant')",
        "typeof(mc.message_id) = 'integer'",
    ]
    params: list[object] = [fts_query, LOSSLESS_COVERAGE_VERSION]
    if source_session_id is not None:
        clauses.append("mc.source_session_id = ?")
        params.append(source_session_id)
    if source_workspace_id is not None:
        clauses.append("mc.source_workspace_id = ?")
        params.append(source_workspace_id)
    if source_peer_id is not None:
        clauses.append("mc.source_peer_id = ?")
        params.append(source_peer_id)
    chip = f'coverage_fts("{" ".join(tokens)}")'
    hits: list[MessageHit] = []
    seen: set[tuple[str, int]] = set()
    batch_size = max(64, top_k * 8)
    offset = 0
    while True:
        try:
            rows = conn.execute(
                f"""
                SELECT {COVERAGE_VALIDATION_COLUMNS}
                FROM message_retention_coverage mc
                {COVERAGE_VALIDATION_JOINS}
                JOIN message_coverage_fts
                  ON message_coverage_fts.rowid = c.rowid
                WHERE {' AND '.join(clauses)}
                ORDER BY mc.message_id, mc.chunk_id, mc.coverage_version
                LIMIT ? OFFSET ?
                """,
                (*params, batch_size, offset),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            break
        offset += len(rows)
        for row in rows:
            try:
                proof = validate_message_coverage_row(row)
            except (RuntimeError, TypeError, ValueError):
                continue
            occurrence = (proof.session_id, proof.message_id)
            if occurrence in seen:
                continue
            seen.add(occurrence)
            folded_text = _fold_diacritics(proof.content).lower()
            token_counts = [folded_text.count(token.lower()) for token in tokens]
            distinct = sum(count > 0 for count in token_counts)
            occurrences = sum(min(count, 4) for count in token_counts)
            phrase_bonus = int(cleaned.lower() in folded_text)
            scoped_score = -float(2 * distinct + occurrences + phrase_bonus)
            hits.append(MessageHit(
                message_id=proof.message_id,
                session_id=proof.session_id,
                role=proof.role,
                text=proof.content,
                score=scoped_score,
                created_at=proof.source_created_at or "",
                score_kind="coverage_lexical",
                source_peer_id=proof.source_peer_id,
                source_workspace_id=proof.source_workspace_id,
                why_retrieved=[chip],
            ))
    hits.sort(key=lambda hit: (hit.score, hit.message_id))
    return hits[:top_k]


def _coverage_message_vector_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int,
    embedding_client: EmbeddingClient | None,
    query_vector: list[float] | None,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[MessageHit]:
    """Exact cosine search over validated durable message occurrences.

    The JSON mirror is authoritative because it carries model/dimension and a
    coverage FK. Scoring that validated mirror makes sqlite-vec availability,
    stale physical rowids, and raw-message pruning unable to change results.
    Scope is applied before ranking and no finite ANN/pre-score cutoff can hide
    the best eligible occurrence. The cursor is consumed row-by-row and only
    ``top_k`` candidates are retained, bounding Python-side memory while exact
    scoring remains O(scoped corpus).
    """
    if (
        top_k <= 0 or not isinstance(query, str) or not query.strip()
        or embedding_client is None or query_vector is None
        or not query_vector
    ):
        return []
    try:
        model = embedding_client.model
        dim = embedding_client.dim
    except Exception:
        return []
    if (
        not isinstance(model, str) or not model
        or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
        or len(query_vector) != dim
    ):
        return []

    clauses = [
        "mc.coverage_version = ?",
        "mc.source_role IN ('user', 'assistant')",
        "source_session.coverage_message_id IS NOT NULL",
        "typeof(mc.message_id) = 'integer'",
        "mc.message_id <= source_session.coverage_message_id",
        "me.model = ?",
        "me.dim = ?",
    ]
    params: list[object] = [LOSSLESS_COVERAGE_VERSION, model, dim]
    if source_session_id is not None:
        clauses.append("mc.source_session_id = ?")
        params.append(source_session_id)
    if source_workspace_id is not None:
        clauses.append("mc.source_workspace_id = ?")
        params.append(source_workspace_id)
    if source_peer_id is not None:
        clauses.append("mc.source_peer_id = ?")
        params.append(source_peer_id)
    try:
        rows = conn.execute(
            f"""
            SELECT {COVERAGE_VALIDATION_COLUMNS},
                   me.vector_json AS message_vector_json,
                   me.text_hash AS embedding_text_hash
            FROM message_embeddings me
            JOIN message_retention_coverage mc
              ON mc.message_id = me.message_id
             AND mc.chunk_id = me.source_coverage_chunk_id
             AND mc.coverage_version = me.source_coverage_version
            {COVERAGE_VALIDATION_JOINS}
            WHERE {' AND '.join(clauses)}
            """,
            tuple(params),
        )
    except sqlite3.OperationalError:
        return []

    qnorm = math.sqrt(sum(value * value for value in query_vector))
    if not math.isfinite(qnorm) or qnorm <= 0.0:
        return []
    # Each side of the join is unique on message_id, so no unbounded seen-set
    # is needed. ``rank`` is lower-is-better and preserves deterministic ties.
    selected: list[tuple[tuple[float, str, int], MessageHit]] = []
    for row in rows:
        try:
            proof = validate_message_coverage_row(row)
        except (RuntimeError, TypeError, ValueError):
            continue
        expected_hash = embedding_text_hash(proof.content)
        if row["embedding_text_hash"] != expected_hash:
            continue
        if not _quality_allows_candidate(
            embedding_client, query, proof.content
        ):
            continue
        vector = _decode_finite_vector(
            row["message_vector_json"], expected_dim=dim
        )
        if vector is None:
            continue
        vnorm = math.sqrt(sum(value * value for value in vector))
        similarity = sum(
            left * right for left, right in zip(query_vector, vector)
        ) / (qnorm * vnorm)
        if not math.isfinite(similarity) or similarity <= 0.0:
            continue
        similarity = min(1.0, similarity)
        hit = MessageHit(
                message_id=proof.message_id,
                session_id=proof.session_id,
                role=proof.role,
                text=proof.content,
                score=similarity,
                created_at=proof.source_created_at or "",
                score_kind="semantic",
                source_peer_id=proof.source_peer_id,
                source_workspace_id=proof.source_workspace_id,
                why_retrieved=[f"message_vec(sim={similarity:.3f})"],
            )
        rank = (-similarity, hit.session_id, hit.message_id)
        if len(selected) < top_k:
            selected.append((rank, hit))
            continue
        worst_index = max(range(len(selected)), key=lambda index: selected[index][0])
        if rank < selected[worst_index][0]:
            selected[worst_index] = (rank, hit)
    selected.sort(key=lambda item: item[0])
    return [hit for _, hit in selected]


def _merge_message_lexical_semantic(
    lexical: list[MessageHit], semantic: list[MessageHit], *, top_k: int
) -> list[MessageHit]:
    """Preserve lexical winners and use vectors to fill otherwise-empty slots.

    The measured local encoder is useful for true vocabulary gaps but noisy in
    deep lists.  Semantic corroboration is therefore annotated on lexical hits,
    while semantic-only occurrences append without demoting a strong BM25 hit.
    Global cross-tier fusion remains a separate policy concern.
    """
    if top_k <= 0:
        return []
    semantic_by_key = {
        (hit.session_id, hit.message_id): hit for hit in semantic
    }
    merged: list[MessageHit] = []
    seen: set[tuple[str, int]] = set()
    for hit in lexical:
        key = (hit.session_id, hit.message_id)
        corroboration = semantic_by_key.get(key)
        if corroboration is not None:
            hit = replace(
                hit,
                why_retrieved=[
                    *hit.why_retrieved, *corroboration.why_retrieved,
                    "message_lexical_preserved",
                ],
            )
        merged.append(hit)
        seen.add(key)
        if len(merged) >= top_k:
            return merged
    for hit in semantic:
        key = (hit.session_id, hit.message_id)
        if key in seen:
            continue
        merged.append(hit)
        seen.add(key)
        if len(merged) >= top_k:
            break
    return merged


# High-frequency function words dropped when building an aggregation FTS query,
# so a full question ("how many project cards did I add?") matches on content
# terms only instead of OR-ing in noise like "do"/"have"/"my". English + Dutch
# (the project's Dutch-prioritized Latin-script scope). Only used by the
# aggregate path; the normal _message_fts_search keeps its len>=2, no-filter
# tokenization so ability=None behavior is byte-for-byte unchanged.
_AGG_STOPWORDS = frozenset({
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "my", "your", "our", "their",
    "its", "in", "on", "at", "to", "for", "of", "with", "from", "by", "as",
    "about", "what", "how", "when", "where", "which", "who", "whom", "why",
    "can", "will", "would", "should", "could", "may", "might", "must",
    "i", "me", "we", "you", "they", "them", "us", "this", "that", "these",
    "those", "total", "many", "much", "across", "all", "any", "new", "old",
    "after", "before", "and", "or", "but", "not",
    # Dutch
    "de", "het", "een", "ik", "je", "jij", "wij", "ze", "zij", "hij", "u",
    "heb", "hebt", "heeft", "hebben", "had", "hadden", "ben", "bent", "zijn",
    "hoeveel", "wat", "hoe", "wanneer", "waar", "welke", "wie", "waarom",
    "op", "van", "met", "voor", "naar", "aan", "bij", "uit", "over", "om",
    "tot", "door", "en", "of", "maar", "niet", "alle", "alles", "veel",
    "totaal", "nieuw", "nieuwe", "na", "mijn", "jouw", "ons", "onze", "hun",
    "dit", "dat", "deze", "die",
})


def _aggregate_tokens(query: str) -> list[str]:
    """Build content tokens for an aggregation FTS query: drop stopwords and
    tokens shorter than 3 chars. Falls back to the normal len>=2 tokenization if
    that empties the set (a question made entirely of stop/short words), so the
    query is never empty."""
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    if not cleaned:
        return []
    parts = cleaned.split()
    filtered = [t for t in parts if len(t) >= 3 and t.lower() not in _AGG_STOPWORDS]
    if filtered:
        return filtered
    return [t for t in parts if len(t) >= 2]


# Upper bound on user turns scanned for an aggregation count, so the count is
# exact for realistic conversations even when the returned evidence is capped at
# the (smaller) message_fts_aggregate_cap. Generous vs. any plausible "how many"
# answer; not a config knob to keep the surface lean.
_MR_COUNT_SCAN = 5000


def _dedup_key(text: str) -> str:
    """Conservative normalization for restatement dedup: lowercase + collapse
    whitespace, nothing else. Two turns collapse only if their text is otherwise
    identical — so turns differing by *any* content token (a number, an entity)
    stay distinct and the count never folds distinct events together. This is the
    safe side of the Phase 1.5 trade-off (under-merge, never under-count)."""
    return " ".join(text.lower().split())


# Coordinating conjunctions that join enumerated items inside one turn, across
# the project's Latin-script scope (English + Dutch prioritized, plus the common
# German/French/Spanish forms a multilingual user may switch into). Matched only
# as whole words (the regex below uses \b) so "android"/"sander" don't trip the
# "and"-lookalike. Kept as a comment-level note; the split regex is the source
# of truth so the two never drift.
#
# A turn enumerating >= this many items is flagged as an over-count risk: the
# distinct-turn count treats it as one event, but it states several. Two list
# segments (one comma/conjunction split) is the smallest unambiguous list.
_ENUM_MIN_SEGMENTS = 2


def _enumerates_items(text: str) -> bool:
    """Heuristic: does this single turn enumerate *several* items?

    The MR over-count failure mode is one turn listing many things ("I have a
    shirt, jeans, and boots") — the aggregate count tallies it as one matching
    turn while the true item count is three. We can't resolve that count without
    typing (see Phase A), but we can deterministically *flag* the turn so the
    host LLM knows turn-count and item-count diverge here and re-reads the text.

    Detection is intentionally conservative — a structural list signal, not NLP:
    we split on commas, semicolons, and standalone coordinating conjunctions
    (`and`/`en`/`und`/…), and report True only when >= `_ENUM_MIN_SEGMENTS`
    non-empty segments survive. A plain sentence ("I added a blue shirt") has one
    segment and is never flagged; "shirt, jeans and boots" yields three. This
    biases toward *under*-flagging (a missed list just falls back to the existing
    one-turn-one-item assumption), never toward inventing enumerations."""
    lowered = text.lower()
    # Split on list punctuation first, then on standalone conjunction words.
    rough = re.split(r"[,;]|\band\b|\ben\b|\bund\b|\bet\b|\by\b|\be\b", lowered)
    segments = [s for s in (seg.strip() for seg in rough) if s]
    # Re-validate that any conjunction split was a real word boundary: the regex
    # already enforces \b, but guard the empty/whitespace artifacts above.
    return len(segments) >= _ENUM_MIN_SEGMENTS


def _message_fts_aggregate(
    conn: sqlite3.Connection,
    query: str,
    *,
    cap: int,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> tuple[list[MessageHit], int, int]:
    """Counting retrieval for MR-style "how many X" questions.

    The LLM can't reliably tally across a wall of hits, so this does the
    deterministic part in SQL/Python and hands back a *candidate count* plus the
    evidence behind it:

      - restricts to **user** turns (assistant echoes double-count actions),
      - drops query stopwords so the match is on content terms (`_aggregate_tokens`),
      - collapses literal restatements via `_dedup_key` (conservative),
      - returns the distinct count + those turns (chronological), evidence capped
        at `cap` while the returned count stays exact (scanned up to
        `_MR_COUNT_SCAN`).

    The count is a *candidate* answer, not gospel — one turn may state several
    items, or none — so the host's LLM verifies it against the returned turns.

    Over-count provenance: because a graph-native typed count is not viable on
    this schema (the entity-type vocabulary is tech-only, so consumer/personal
    categories like "clothing" never get typed — see `_enumerates_items` and the
    module note), we can't *resolve* the "one turn lists three items" case. We
    instead surface it: the third return value is the number of distinct evidence
    turns that *enumerate* multiple items, and each such hit carries
    `enumerates_items=True`. A nonzero value tells the host LLM that turn-count
    undercounts the true item-count and it should re-read those turns rather than
    trust the distinct-turn tally. When zero, one-turn-one-item holds and the
    candidate count is the answer.

    Returns ([], 0, 0) on no content tokens or if `messages_fts` is absent."""
    tokens = _aggregate_tokens(query)
    if not tokens:
        return [], 0, 0
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

    scope_sql, scope_params = _message_scope_sql(
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    try:
        rows = conn.execute(
            f"""
            SELECT m.id, m.session_id, m.role, m.content, m.created_at,
                   m.source_peer_id, m.source_workspace_id,
                   bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            WHERE messages_fts MATCH ? AND m.role = 'user' {scope_sql}
            ORDER BY m.created_at, m.id
            LIMIT ?
            """,
            (fts_query, *scope_params, _MR_COUNT_SCAN),
        ).fetchall()
    except sqlite3.OperationalError:
        return [], 0, 0

    seen: set[str] = set()
    deduped: list[sqlite3.Row] = []
    for r in rows:
        key = _dedup_key(r["content"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    count = len(deduped)
    # Count over the FULL deduped set (not just the capped evidence) so the
    # over-count signal stays exact alongside `count`, mirroring how the count
    # itself survives the cap.
    enumeration_turns = sum(1 for r in deduped if _enumerates_items(r["content"]))
    chip = f'message_fts_aggregate("{" ".join(tokens)}")'
    hits = [
        MessageHit(
            message_id=int(r["id"]),
            session_id=r["session_id"],
            role=r["role"],
            text=r["content"],
            score=float(r["score"]),
            created_at=r["created_at"] or "",
            score_kind="aggregate",
            why_retrieved=[chip],
            enumerates_items=_enumerates_items(r["content"]),
            source_peer_id=r["source_peer_id"],
            source_workspace_id=r["source_workspace_id"],
        )
        for r in deduped[:cap]
    ]
    return hits, count, enumeration_turns


def _maybe_graph_count(
    conn: sqlite3.Connection, query: str
) -> GraphCount | None:
    """EXACT graph count for the in-domain slice of an MR query, else None.

    Bridges the MR query to `count_relations` via `count_routing.plan_count`,
    which only emits a plan when the query maps cleanly onto in-vocab
    type/predicate/entity (see that module for the side heuristic). The anchor
    entity is taken from `match_known_entities` — the DIRECT matches only, so a
    type-expansion canonical can never masquerade as a user-named anchor and
    pivot the count on an entity the user never mentioned.

    Degrades to None on ANY failure (routing or the count itself) so the keyword
    aggregate path — which has already run — is never disturbed. `count_relations`
    already swallows OperationalError; the broad except here guards the routing
    layer too, keeping augment() robust against a malformed query/graph."""
    # Imported lazily to avoid a circular import: count_routing imports
    # detect_query_types from this module.
    from hymem.query.count_routing import plan_count

    try:
        matched = match_known_entities(conn, query)
        plan = plan_count(query, matched)
        if plan is None:
            return None
        result = _resolve_graph_count(conn, plan)
        # Suppress a zero-with-no-evidence result. An in-vocab mapping that finds
        # no edges is almost always a wrong-direction or absent-data case; a
        # surfaced "exact 0" would mislead the host more than the keyword
        # candidate it would shadow, so fall back to that candidate instead. A
        # genuine "zero of type X" is rare and the keyword path still answers it.
        if result.count == 0 and not result.entities:
            return None
        return result
    except Exception:  # noqa: BLE001 — additive layer must never break augment()
        log.debug("graph_count routing failed; leaving graph_count=None", exc_info=True)
        return None


def _resolve_graph_count(conn: sqlite3.Connection, plan: "CountPlan") -> GraphCount:
    """Run `count_relations` for a plan, recovering from a wrong-direction guess.

    The router (`plan_count`) has to guess which side carries the counted type for
    an *anchored* question, and it defaults to typing the **subject** ("how many
    services depend on redis" → services are subjects). But natural phrasings just
    as often type the **object** ("how many databases does billing use" → the
    databases are the objects of `billing uses …`). When the default orientation
    finds nothing, we retry the mirror orientation (type on the object side,
    anchor as the subject) and prefer whichever actually has edges.

    Only anchored, subject-typed plans carry this ambiguity. We try the mirror
    solely when the primary orientation is empty, so a primary that *does* find
    edges is trusted as-is and no both-orientations-non-empty ambiguity can arise.
    Unanchored plans (no entity to pivot on) have a single unambiguous orientation
    and are returned directly."""
    primary = count_relations(
        conn,
        count=plan.count,
        predicates=plan.predicates,
        subject=plan.subject,
        object=plan.object,
        subject_type=plan.subject_type,
        object_type=plan.object_type,
    )
    # Mirror only applies to the anchored, subject-typed shape the router emits
    # (count="subject", subject_type=T, object=anchor). If it already found edges,
    # trust it; otherwise flip the type to the object side and the anchor to the
    # subject side and see if that orientation is the one the graph holds.
    anchored_subject_typed = (
        plan.count == "subject"
        and plan.subject_type is not None
        and plan.object is not None
    )
    if not anchored_subject_typed or primary.count > 0:
        return primary
    mirror = count_relations(
        conn,
        count="object",
        predicates=plan.predicates,
        subject=plan.object,
        object_type=plan.subject_type,
    )
    return mirror if mirror.count > 0 else primary


# Predicates whose edges describe a datable event/adoption ("we started using X
# on …"), so a dated edge is worth surfacing on the TR timeline. Confidence-
# bearing relational predicates; excludes structural ones (part_of, contains,
# equivalent_to) that rarely carry a meaningful event date.
_TR_EDGE_PREDICATES = frozenset({
    "uses", "depends_on", "deploys_to", "implements", "replaces",
    "requires_version", "runs_on", "connects_to", "configured_with",
})


def _temporal_event_date(normalized: str | None, created_at: str) -> str | None:
    """Pick a sortable date for an event, preferring a fully-resolved date.

    A year-less message mention (`normalized` is None) still has ordering signal
    via the turn's event time, so we fall back to the date portion of
    `created_at` (ISO strings sort lexicographically, so the leading 10 chars are
    the YYYY-MM-DD prefix). Returns None only when nothing datable remains, so
    the caller can drop it (the TR list must stay date-only)."""
    if normalized:
        return normalized
    if created_at and len(created_at) >= 10 and created_at[4] == "-":
        return created_at[:10]
    return None


def _temporal_message_events(
    conn: sqlite3.Connection, query: str, *, top_k: int
) -> list[TemporalEvent]:
    """Query-relevant dated message mentions, ordered date-ascending.

    Restricts the FTS-matched messages to those carrying an extracted date
    (`temporal_mentions`), so the list is purely chronological evidence. Degrades
    to [] — never raises — when `temporal_mentions` or `messages_fts` is absent
    (pre-v14 DB), mirroring `_message_fts_search`'s OperationalError tolerance."""
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    tokens = [t for t in cleaned.split() if len(t) >= 2]

    # When the query carries content tokens, scope the timeline to messages that
    # match them; an empty/stopword-only query falls back to the whole index so a
    # bare "what happened first?" still returns a chronology.
    try:
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            rows = conn.execute(
                """
                SELECT tm.normalized_date, tm.raw_text, tm.surrounding_text,
                       tm.created_at, m.id AS message_id
                FROM temporal_mentions tm
                JOIN messages_fts ON messages_fts.rowid = tm.message_id
                JOIN messages m ON m.id = tm.message_id
                WHERE messages_fts MATCH ?
                """,
                (fts_query,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT normalized_date, raw_text, surrounding_text,
                       created_at, message_id
                FROM temporal_mentions
                """
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    events: list[tuple[str, TemporalEvent]] = []
    for r in rows:
        date = _temporal_event_date(r["normalized_date"], r["created_at"] or "")
        if date is None:
            continue
        chip = (
            "temporal_mention"
            if r["normalized_date"]
            else "temporal_mention(event_time)"
        )
        events.append((
            date,
            TemporalEvent(
                date=date,
                text=r["surrounding_text"] or r["raw_text"],
                source="message",
                why_retrieved=[chip],
            ),
        ))
    events.sort(key=lambda e: e[0])
    return [ev for _, ev in events[:top_k]]


def _temporal_graph_events(
    conn: sqlite3.Connection, entities: list[str], *, top_k: int
) -> list[TemporalEvent]:
    """Dated knowledge-graph edges for matched entities, ordered date-ascending.

    ``knowledge_graph.valid_at`` is the sort coordinate; ingestion-time
    ``first_seen`` is never a fallback. A scope may annotate the text only when
    it belongs to current canonical positive evidence, preventing an older
    retired revision from leaking into today's chronology. Only direct,
    live-current datable predicates are considered. Returns [] when no entity
    matched and tolerates pre-provenance databases."""
    if not entities:
        return []

    pred_placeholders = ",".join("?" * len(_TR_EDGE_PREDICATES))
    ent_placeholders = ",".join("?" * len(entities))
    try:
        current = live_edge_predicate("kg")
        rows = conn.execute(
            f"""
            SELECT kg.id, kg.subject_canonical AS s, kg.predicate AS p,
                   kg.object_canonical AS o,
                   hymem_normalize_iso_timestamp(kg.valid_at) AS valid_at
            FROM knowledge_graph kg
            WHERE {current}
              AND kg.valid_at IS NOT NULL
              AND kg.predicate IN ({pred_placeholders})
              AND (kg.subject_canonical IN ({ent_placeholders})
                   OR kg.object_canonical IN ({ent_placeholders}))
            """,
            list(_TR_EDGE_PREDICATES) + entities + entities,
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    events: list[tuple[str, TemporalEvent]] = []
    for r in rows:
        # Scope is explanatory text, never the chronology coordinate. The
        # lifecycle-derived valid_at is the source-time fact onset.
        citations = current_positive_citations(conn, int(r["id"]), limit=1)
        if not citations:
            # A legacy/materialized timestamp has no exact source proof and may
            # be ingestion-derived; omitting it is safer than asserting a false
            # chronology coordinate.
            continue
        scope = (citations[0].temporal_scope or "").strip()
        date = _temporal_event_date(None, r["valid_at"] or "")
        if date is None:
            continue
        fact = f"{r['s']} {r['p']} {r['o']}"
        text = f"{fact} ({scope})" if scope else fact
        chips = ["edge_valid_at"]
        if scope:
            chips.append("current_temporal_scope")
        events.append((
            date,
            TemporalEvent(
                date=date, text=text, source="graph", why_retrieved=chips
            ),
        ))
    events.sort(key=lambda e: e[0])
    return [ev for _, ev in events[:top_k]]


# Secondary TR source (session-date events) is capped tight so a broad FTS match
# can't flood the chronology and bury the primary content-date / graph evidence.
_SESSION_EVENT_CAP = 5


def _truncate(text: str, limit: int = 200) -> str:
    text = (text or "").strip()
    return text[: limit - 3] + "..." if len(text) > limit else text


def _temporal_hits_events(
    conn: sqlite3.Connection,
    message_hits: list["MessageHit"],
    fts_hits: list["FtsHit"],
    *,
    cap: int = _SESSION_EVENT_CAP,
) -> list[TemporalEvent]:
    """Session-date anchors derived from the ALREADY-RETRIEVED evidence — not a
    fresh keyword pass.

    The turn's `created_at` (the session date threaded in at ingest) anchors a
    question even when the user never restated a date in the prose — the common
    LongMemEval shape (grounding lives in session metadata). Earlier this was a
    separate FTS pass keyed on the query tokens, which inherited the question's
    vocabulary: a turn saying "Walk for Hunger" was invisible to a question about
    a "charity event". Sourcing the anchors from what the retriever ALREADY
    surfaced fixes that — `fts_hits` are the *semantic* chunk tier (embeddings),
    so they recall the answer-bearing turn even when keywords miss.

    Two evidence sources, both emitted as `source="session-date"` (a *when-
    discussed* anchor, NOT event-time for duration math):
    (1) `message_hits` — raw turns carrying `created_at` + `message_id` directly;
    (2) `fts_hits` — dreamed chunks, mapped chunk -> `start_message_id` ->
        `messages.created_at` for the chunk's session date.
    Guard rails: a turn/chunk-range that already carries a `temporal_mention`
    (a content-date) is skipped — those are authoritative and handled by
    `_temporal_message_events`, so no double-count and no dilution. Capped at
    `cap`, in retrieval-relevance order (message_hits before chunk hits), then
    presented date-ascending. Degrades to [] on a pre-v14 DB (no
    temporal_mentions table)."""
    # The TR feature is gated on temporal_mentions; absent it, degrade to [] like
    # the rest of the path (preserves the pre-v14 contract).
    try:
        dated = {
            r["message_id"]
            for r in conn.execute("SELECT message_id FROM temporal_mentions")
        }
    except sqlite3.OperationalError:
        return []

    candidates: list[TemporalEvent] = []
    seen: set[int] = set()

    # (1) Raw-message hits: provenance is on the hit object itself.
    for h in message_hits:
        mid = getattr(h, "message_id", None)
        if mid is None or mid in dated or mid in seen:
            continue
        date = _temporal_event_date(None, getattr(h, "created_at", "") or "")
        if date is None:
            continue
        seen.add(mid)
        candidates.append(TemporalEvent(
            date=date, text=_truncate(getattr(h, "text", "")),
            source="session-date", why_retrieved=["message_hit(discussed)"],
        ))

    # (2) Semantic chunk hits: map chunk -> message range -> session date. Skip a
    # chunk whose range already carries a content-date (covered by the primary).
    chunk_ids = [c for c in (getattr(h, "chunk_id", None) for h in fts_hits) if c]
    if chunk_ids:
        ph = ",".join("?" * len(chunk_ids))
        try:
            rows = conn.execute(
                f"""
                SELECT c.id, c.start_message_id, c.text,
                       (SELECT m.created_at FROM messages m
                        WHERE m.id = c.start_message_id) AS created_at,
                       EXISTS(SELECT 1 FROM temporal_mentions tm
                              WHERE tm.message_id BETWEEN c.start_message_id
                                                      AND c.end_message_id) AS has_date
                FROM chunks c
                WHERE c.id IN ({ph})
                """,
                chunk_ids,
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        # Preserve the fts_hits relevance order (the SQL IN-clause doesn't).
        by_id = {r["id"]: r for r in rows}
        for cid in chunk_ids:
            r = by_id.get(cid)
            if r is None or r["has_date"]:
                continue
            mid = r["start_message_id"]
            if mid in seen:
                continue
            date = _temporal_event_date(None, r["created_at"] or "")
            if date is None:
                continue
            seen.add(mid)
            candidates.append(TemporalEvent(
                date=date, text=_truncate(r["text"]),
                source="session-date", why_retrieved=["chunk_hit(discussed)"],
            ))

    # Cap in relevance order (message_hits first), then present date-ascending.
    return sorted(candidates[:cap], key=lambda e: e.date)


_ISO_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _looks_iso(value: str) -> bool:
    """True if `value` starts with a YYYY-MM-DD prefix (so its first 10 chars are
    a usable sort key). Used to decide whether a free-text temporal_scope can be
    ordered against ISO dates."""
    return bool(_ISO_DATE_PREFIX.match(value))


def _temporal_events(
    conn: sqlite3.Connection,
    query: str,
    message_hits: list["MessageHit"],
    fts_hits: list["FtsHit"],
    *,
    top_k: int,
) -> list[TemporalEvent]:
    """Merge dated message mentions, dated graph edges, and session-date anchors
    into one chronology.

    The primary sources (message mentions by FTS over the query, graph edges by
    the query's matched entities) are gathered, sorted date-ascending, and capped
    at `top_k`. Session-date anchors — the `created_at` of *dateless* turns the
    retriever already surfaced (`message_hits` + semantic `fts_hits`) — are then
    appended *additively* (beyond `top_k`, capped at `_SESSION_EVENT_CAP`), so
    metadata-grounded questions get an anchor without the content-date chronology
    being evicted (see `_temporal_hits_events`). Sourcing them from the retrieved
    hits rather than a fresh keyword pass means semantic recall reaches the
    answer-bearing turn even when the question's vocabulary doesn't match it.
    The union is re-sorted date-ascending. Returns [] gracefully on a pre-v14 DB
    (no temporal_mentions table) — the underlying helpers swallow the
    OperationalError — so the TR path never raises."""
    msg_events = _temporal_message_events(conn, query, top_k=top_k)
    entities = match_known_entities(conn, query)
    graph_events = _temporal_graph_events(conn, entities, top_k=top_k)
    primary = msg_events + graph_events
    primary.sort(key=lambda e: e.date)
    primary = primary[:top_k]

    # Session-date anchors are ALWAYS added, not gated on a thin primary. In a
    # large haystack the content-date timeline is mostly scraped noise (the FTS
    # always finds *some* dated turns), while the answer-bearing turn is often
    # dateless-but-session-dated — so the anchor must survive even when content-
    # dates are plentiful. An earlier `< 2` gate was self-defeating: it never
    # fired because the haystack always yields ≥2 content-dates. They are added
    # *additively* (beyond `top_k`) and labelled when-discussed, so they never
    # EVICT the content-date chronology that the passing questions rely on.
    session_events = _temporal_hits_events(conn, message_hits, fts_hits)
    merged = primary + session_events
    merged.sort(key=lambda e: e.date)
    return merged


def _embeddings_compatible(conn: sqlite3.Connection, embedder: EmbeddingClient) -> bool:
    """Guard against querying chunk embeddings written by a *different* model or
    dimension than the active embedder. A dim mismatch is caught per-vector in
    the python path, but the vec0 fast path has no such check, and a model swap
    that keeps the same dim would otherwise return silent garbage. When the
    stored model/dim disagree with the live client, skip vector search (FTS
    still runs) and log so the operator can re-embed.

    Checks every distinct (model, dim) pair, not just one row: a mixed corpus
    (some rows from an old model, some from the new) must still be treated as
    incompatible unless *all* stored vectors match the active client — otherwise
    a single matching first row would wave through a partially-stale corpus."""
    rows = conn.execute(
        "SELECT DISTINCT model, dim FROM chunk_embeddings"
    ).fetchall()
    if not rows:
        return True  # nothing stored yet — nothing to mismatch
    mismatched = [
        (r["model"], r["dim"])
        for r in rows
        if r["model"] != embedder.model or r["dim"] != embedder.dim
    ]
    if mismatched:
        log.warning(
            "vector search skipped: stored embeddings include model/dim %s but "
            "active client is model=%s dim=%s; re-embed to enable semantic recall",
            mismatched, embedder.model, embedder.dim,
        )
        return False
    return True


def _vector_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
    query_vector: object = _QUERY_VECTOR_UNSET,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[FtsHit]:
    # Durable rows carry the vector-space identity; vec0 does not. Use one
    # exact scorer for both environments so stale physical rowids and
    # same-dimension model swaps cannot change results.
    return _python_cosine_search(
        conn, embedder, query, top_k=top_k, max_scan=max_scan,
        query_vector=query_vector,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )


def _vec_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    query_vector: object = _QUERY_VECTOR_UNSET,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[FtsHit]:
    return _python_cosine_search(
        conn, embedder, query, top_k=top_k, max_scan=max(top_k, 5000),
        query_vector=query_vector,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )


def _python_cosine_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
    query_vector: object = _QUERY_VECTOR_UNSET,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[FtsHit]:
    try:
        model = embedder.model
        dim = embedder.dim
    except Exception:
        return []
    if (
        not isinstance(model, str) or not model
        or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
    ):
        return []
    scope_sql, scope_params = _chunk_manifest_scope_sql(
        "c",
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    scan_limit = max(0, int(max_scan))
    if scan_limit <= 0:
        return []
    scoped = any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    )
    rows: list[sqlite3.Row] = []
    if not scoped:
        rows = conn.execute(
            f"""
            SELECT c.id AS chunk_id, c.session_id, c.text, e.vector_json,
                   e.text_hash
            FROM chunk_embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            WHERE c.chunk_kind = 'extraction' AND e.model = ? AND e.dim = ?
            {scope_sql}
            ORDER BY c.created_at DESC, c.id
            LIMIT ?
            """,
            (model, dim, *scope_params, scan_limit),
        ).fetchall()
    else:
        # ``max_scan`` counts proof-valid vectors, not superficially matching
        # rows. Otherwise ``max_scan`` forged/newer manifests can occupy every
        # slot and starve an older valid scoped hit. Page with bounded memory,
        # validate each batch before any model-quality hook/reranker sees text,
        # and retain a separate raw-row ceiling for corrupt-store DoS safety.
        page_size = max(32, min(256, scan_limit * 2))
        raw_scan_cap = max(1024, scan_limit * 64)
        offset = 0
        scanned = 0
        while len(rows) < scan_limit and scanned < raw_scan_cap:
            batch = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.session_id, c.text, e.vector_json,
                       e.text_hash
                FROM chunk_embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                WHERE c.chunk_kind = 'extraction' AND e.model = ? AND e.dim = ?
                {scope_sql}
                ORDER BY c.created_at DESC, c.id
                LIMIT ? OFFSET ?
                """,
                (
                    model, dim, *scope_params,
                    min(page_size, raw_scan_cap - scanned), offset,
                ),
            ).fetchall()
            if not batch:
                break
            temporary = AugmentedContext(fts_hits=[
                FtsHit(
                    chunk_id=row["chunk_id"], session_id=row["session_id"],
                    text=row["text"], score=0.0, score_kind="vec",
                )
                for row in batch
            ])
            enrich_context_provenance(conn, temporary)
            scope_context_in_place(
                temporary,
                source_session_id=source_session_id,
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
            safe_ids = {hit.chunk_id for hit in temporary.fts_hits}
            rows.extend(
                row for row in batch
                if row["chunk_id"] in safe_ids
                and row["text_hash"] == embedding_text_hash(row["text"])
                and _decode_finite_vector(
                    row["vector_json"], expected_dim=dim
                ) is not None
            )
            offset += len(batch)
            scanned += len(batch)
            if len(batch) < page_size:
                break
        rows = rows[:scan_limit]
        if len(rows) < scan_limit and scanned >= raw_scan_cap:
            log.warning(
                "augment.scoped_vector_validation_scan_exhausted "
                "accepted=%d max_scan=%d scanned=%d",
                len(rows), scan_limit, scanned,
            )
    if not rows:
        return []

    if query_vector is _QUERY_VECTOR_UNSET:
        qvec = _query_embedding(embedder, query)
    elif isinstance(query_vector, list):
        qvec = _finite_vector(query_vector, expected_dim=dim)
    else:
        qvec = None
    if qvec is None:
        return []
    qnorm = math.sqrt(sum(x * x for x in qvec))

    scored: list[tuple[float, sqlite3.Row]] = []
    for r in rows:
        expected_hash = embedding_text_hash(r["text"])
        if r["text_hash"] != expected_hash:
            continue
        if not _quality_allows_candidate(embedder, query, r["text"]):
            continue
        vec = _decode_finite_vector(r["vector_json"], expected_dim=dim)
        if vec is None:
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec))
        sim = dot / (qnorm * vnorm)
        if not math.isfinite(sim) or sim <= 0.0:
            continue
        scored.append((sim, r))
    scored.sort(key=lambda item: (-item[0], str(item[1]["chunk_id"])))

    return [
        FtsHit(
            chunk_id=r["chunk_id"],
            session_id=r["session_id"],
            text=r["text"],
            score=float(sim),
            score_kind="vec",
            why_retrieved=[f"vec_topk(sim={sim:.3f})"],
        )
        for sim, r in scored[:top_k]
    ]


def _rrf_merge(
    fts: list[FtsHit], vec: list[FtsHit], *, top_k: int, k: int = 60
) -> list[FtsHit]:
    # Reciprocal rank fusion: score = sum(1 / (k + rank)) across each list.
    by_id: dict[str, FtsHit] = {}
    scores: dict[str, float] = {}
    fts_ids = {h.chunk_id for h in fts}
    vec_ids = {h.chunk_id for h in vec}
    for rank, hit in enumerate(fts, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.chunk_id, hit)
    for rank, hit in enumerate(vec, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        replace(
            by_id[cid], score=score, score_kind="rrf",
            why_retrieved=_rrf_chips(by_id[cid].why_retrieved, cid, fts_ids, vec_ids, score),
        )
        for cid, score in ordered[:top_k]
    ]


def _rrf_chips(
    base: list[str], item_id: str, fts_ids: set[str], vec_ids: set[str], score: float
) -> list[str]:
    """Carry the kept hit's source chip (fts_match / vec_topk) and append a
    fused `rrf(<sources>, <score>)` chip naming which lists contributed."""
    if item_id in fts_ids and item_id in vec_ids:
        sources = "fts+vec"
    elif item_id in fts_ids:
        sources = "fts"
    else:
        sources = "vec"
    return [*base, f"rrf({sources}, {score:.4f})"]


_EDGE_SELECT = """
    SELECT id, subject_canonical AS s, predicate AS p, object_canonical AS o,
           pos_evidence AS pos, neg_evidence AS neg, derived,
           hymem_normalize_iso_timestamp(valid_at) AS valid_at,
           hymem_normalize_iso_timestamp(invalid_at) AS invalid_at,
           CASE WHEN hymem_timestamp_at_or_before(
                         last_seen,
                         strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')
                     ) = 1
                THEN MAX(
                    0.0,
                    julianday('now')
                      - julianday(hymem_normalize_iso_timestamp(last_seen))
                )
                ELSE 36500.0
           END AS days_since
    FROM knowledge_graph
"""


def _edge_provenance_scope_sql(
    *,
    source_session_id: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
    edge_alias: str = "knowledge_graph",
) -> tuple[str, tuple[object, ...]]:
    """Require a current canonical positive citation in the requested scope."""
    clauses: list[str] = []
    params: list[object] = []
    if source_session_id is not None:
        clauses.append("ev.source_session_id = ?")
        params.append(source_session_id)
    if source_workspace_id is not None:
        clauses.append("ev.source_workspace_id = ?")
        params.append(source_workspace_id)
    if source_peer_id is not None:
        clauses.append("ev.source_peer_id = ?")
        params.append(source_peer_id)
    if not clauses:
        return "", ()
    return (
        " AND EXISTS ("
        "SELECT 1 FROM kg_evidence ev "
        f"WHERE ev.edge_id = {edge_alias}.id "
        "AND ev.provenance_status = 'canonical' "
        "AND ev.polarity = 1 AND ev.is_current = 1 "
        "AND ev.published_at IS NOT NULL AND "
        + " AND ".join(clauses)
        + ")",
        tuple(params),
    )


def _legacy_lifecycle_is_compatible(
    conn: sqlite3.Connection, edge_id: int
) -> bool:
    """Validate the narrow materialized-edge compatibility path.

    A row with no canonical evidence is either a native/manual insertion used
    by the public low-level API and historical callers, or an explicitly
    migrated legacy edge.  No lifecycle is valid for a native row; migrated
    rows may carry only well-formed ``legacy_state`` events.  Any sourced,
    malformed, or mixed lifecycle means the materialized cache cannot be used
    as truth and the edge fails closed.
    """
    rows = conn.execute(
        "SELECT event_kind,event_at,created_at FROM kg_edge_lifecycle "
        "WHERE edge_id=? ORDER BY event_at,event_key",
        (edge_id,),
    ).fetchall()
    if not rows:
        return True
    return all(
        row["event_kind"] == "legacy_state"
        and conn.execute(
            "SELECT hymem_event_clock_is_valid(?,?)",
            (row["event_at"], row["created_at"]),
        ).fetchone()[0] == 1
        for row in rows
    )


def _current_graph_rows(
    conn: sqlite3.Connection,
) -> list[dict[str, object]]:
    """Build the unscoped current graph from durable authority.

    Canonical edges are reconstructed from validated evidence and lifecycle;
    materialized ``knowledge_graph`` counters/status never decide their truth.
    Rows with *no canonical history* retain the documented native/legacy
    compatibility path and are labeled so callers cannot mistake missing
    citations for canonical provenance.
    """
    canonical_history = {
        int(row[0])
        for row in conn.execute(
            "SELECT DISTINCT edge_id FROM kg_evidence "
            "WHERE provenance_status='canonical'"
        ).fetchall()
    }
    evidence_by_edge: dict[int, list[sqlite3.Row]] = {}
    for row in validated_current_evidence(conn):
        evidence_by_edge.setdefault(int(row["edge_id"]), []).append(row)

    signal_totals = validated_confidence_signal_totals(conn)

    now_row = conn.execute("SELECT julianday('now')").fetchone()
    now_jd = float(now_row[0]) if now_row is not None else 0.0
    if not math.isfinite(now_jd):
        now_jd = 0.0

    current: list[dict[str, object]] = []
    for edge_id in sorted(canonical_history):
        evidence_rows = evidence_by_edge.get(edge_id, [])
        if not evidence_rows:
            continue
        state = current_positive_state(
            conn, edge_id, limit=max(5, len(evidence_rows))
        )
        validated_ids = {int(row["evidence_id"]) for row in evidence_rows}
        citations = [
            citation for citation in (state[1] if state is not None else [])
            if citation.evidence_id in validated_ids
        ]
        valid_coordinates = [
            citation.source_event_at for citation in citations
            if citation.source_event_at is not None
        ]
        if state is None or not citations or not valid_coordinates:
            continue
        signal_pos, signal_neg = signal_totals.get(edge_id, (0, 0))
        positive = signal_pos + sum(
            int(row["evidence_weight"])
            for row in evidence_rows if int(row["polarity"]) == 1
        )
        negative = signal_neg + sum(
            int(row["evidence_weight"])
            for row in evidence_rows if int(row["polarity"]) == -1
        )
        if positive <= negative or positive < 0 or negative < 0:
            continue
        kg = conn.execute(
            _EDGE_SELECT + " WHERE id=? AND derived=0", (edge_id,)
        ).fetchone()
        if kg is None:
            continue
        last_event_jd = max(float(row["event_jd"]) for row in evidence_rows)
        days_since = max(0.0, now_jd - last_event_jd)
        if not math.isfinite(days_since):
            continue
        current.append({
            "id": edge_id,
            "s": str(kg["s"]),
            "p": str(kg["p"]),
            "o": str(kg["o"]),
            "pos": positive,
            "neg": negative,
            "derived": bool(kg["derived"]),
            "valid_at": min(valid_coordinates),
            "invalid_at": None,
            "days_since": days_since,
            "citations": citations[:5],
            "authority_kind": "canonical",
        })

    legacy_kinds = {
        int(row["edge_id"]): "legacy"
        for row in conn.execute(
            "SELECT DISTINCT edge_id FROM kg_evidence "
            "WHERE provenance_status<>'canonical'"
        ).fetchall()
    }
    compatibility_rows = conn.execute(
        _EDGE_SELECT
        + f" WHERE {live_edge_predicate()} "
          "ORDER BY subject_canonical,predicate,object_canonical"
    ).fetchall()
    for row in compatibility_rows:
        edge_id = int(row["id"])
        if edge_id in canonical_history or not _legacy_lifecycle_is_compatible(
            conn, edge_id
        ):
            continue
        try:
            days_since = max(0.0, float(row["days_since"]))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(days_since):
            continue
        current.append({
            "id": edge_id,
            "s": str(row["s"]),
            "p": str(row["p"]),
            "o": str(row["o"]),
            "pos": max(0, int(row["pos"])),
            "neg": max(0, int(row["neg"])),
            "derived": bool(row["derived"]),
            "valid_at": row["valid_at"],
            "invalid_at": row["invalid_at"],
            "days_since": days_since,
            "citations": [],
            "authority_kind": legacy_kinds.get(edge_id, "native"),
        })
    current.sort(key=lambda row: (row["s"], row["p"], row["o"]))
    return current


def _recency_weight(days_since: object, half_life_days: object) -> float:
    """Finite conservative recency decay for persisted/configured values."""
    try:
        days = max(0.0, float(days_since))
        half_life = float(half_life_days)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(days) or not math.isfinite(half_life) or half_life <= 0:
        return 0.0
    value = math.exp(-days / half_life)
    return value if math.isfinite(value) else 0.0


def _nonnegative_int_config(value: object) -> int:
    """Return a safe cardinality/depth config, disabling malformed values."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _nonnegative_finite_config(value: object) -> float | None:
    """Return a finite non-negative numeric config or ``None`` when invalid."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _graph_lookup(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    query: str,
    entities: list[str],
    expansion_info: dict[str, str],
    routed: frozenset[str],
    *,
    overlap_info: dict[str, str] | None = None,
    embedding_client: EmbeddingClient | None = None,
    query_vector: object = _QUERY_VECTOR_UNSET,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[GraphFact]:
    """Hybrid edge ranker over one authoritative, comparable candidate pool.

    Entity, semantic, predicate, recency, and multi-hop sources contribute
    candidates before the shared confidence/recency/relevance score and final
    natural-key tie-break. Semantic relevance is additive on routed queries,
    so having a valid vector can improve an edge but can never penalize it.

    With no embedding client the semantic source is skipped and the score
    collapses to confidence × recency × predicate_boost — close to the prior
    entity-anchored leaderboard behaviour.

    When no predicate routes (`routed` is empty), the ranker switches into a
    fallback with per-candidate scoring:
      - Entity-anchored candidates score by confidence × recency. Edge-level
        embeddings are too noisy ("rejects working" matches "projects working"
        for surface reasons) to compete with a deterministic entity hit, so
        Source 2 is skipped entirely when entities matched.
      - Semantic-only candidates (only present when no entity matched) score
        by similarity × confidence × recency.
      - Otherwise (no signal at all) score collapses to confidence × recency
        over a recent-edges seed so something graph-shaped is still shown.
    """
    graph_limit = _nonnegative_int_config(cfg.graph_top_k)
    if graph_limit == 0:
        return []
    if any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    ):
        return _scoped_graph_lookup(
            conn,
            cfg,
            query,
            entities,
            expansion_info,
            routed,
            overlap_info=overlap_info,
            embedding_client=embedding_client,
            query_vector=query_vector,
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
        )

    fallback = not routed
    overlap_info = overlap_info or {}
    graph_rows = _current_graph_rows(conn)
    rows_by_id = {int(row["id"]): row for row in graph_rows}
    candidates: dict[tuple[str, str, str], dict] = {}

    def _ensure(row: sqlite3.Row | dict[str, object]) -> dict:
        key = (row["s"], row["p"], row["o"])
        c = candidates.get(key)
        if c is None:
            c = {
                "edge_id": int(row["id"]),
                "s": row["s"],
                "p": row["p"],
                "o": row["o"],
                "pos": int(row["pos"]),
                "neg": int(row["neg"]),
                "derived": bool(row["derived"]),
                "valid_at": row["valid_at"],
                "invalid_at": row["invalid_at"],
                "days_since": (
                    float(row["days_since"]) if row["days_since"] is not None else 0.0
                ),
                "semantic_score": 0.0,
                "semantic_retrieved": False,
                "entity_match": False,
                "entity_types": set(),
                "overlap_tokens": set(),
                "direct_anchor": False,
                "multihop_score": 0.0,
                "hop": 1,
                "citations": list(row.get("citations", []))
                    if isinstance(row, dict) else [],
                "authority_kind": row.get("authority_kind", "native")
                    if isinstance(row, dict) else "native",
            }
            candidates[key] = c
        return c

    # Source 1 — entity-anchored (always).
    for entity in entities:
        rows = [
            row for row in graph_rows
            if row["s"] == entity or row["o"] == entity
        ]
        for r in rows:
            c = _ensure(r)
            c["entity_match"] = True
            if entity in expansion_info:
                c["entity_types"].add(expansion_info[entity])
            if entity in overlap_info:
                c["overlap_tokens"].add(overlap_info[entity])
            else:
                # Any non-overlap anchor (direct match or type expansion) means
                # the edge isn't *only* surfaced via token-overlap — full weight.
                c["direct_anchor"] = True

    # Source 2 — semantic KNN. Skipped in the fallback path when entities
    # matched: edge-level embeddings are short and noisy, so they crowd out
    # entity-anchored candidates when the user named a known entity.
    skip_semantic = fallback and bool(entities)
    if embedding_client is not None and not skip_semantic:
        for edge_id, semantic_score in _semantic_edge_hits(
            conn, cfg, embedding_client, query,
            allowed_edge_ids=frozenset(rows_by_id),
            query_vector=query_vector,
        ):
            row = rows_by_id.get(edge_id)
            if row is None:
                continue
            try:
                semantic_score = float(semantic_score)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(semantic_score) or semantic_score <= 0.0:
                continue
            semantic_score = min(1.0, semantic_score)
            c = _ensure(row)
            c["semantic_score"] = max(c["semantic_score"], semantic_score)
            c["semantic_retrieved"] = True

    # Source 3 — predicate-routed.
    if routed:
        rows = [row for row in graph_rows if row["p"] in routed]
        for r in rows:
            _ensure(r)

    # Source 4 — multi-hop expansion from DIRECTLY-anchored entities only.
    # Chaining a fuzzy (token-overlap) link N times produces garbage, so seeds
    # exclude overlap-only anchors. Additive: dedups against Sources 1/3 on
    # (s, p, o) via `_ensure`, so a bridged edge that is also a direct/routed hit
    # keeps its stronger native score and multi-hop never double-counts.
    if cfg.graph_multihop_enabled:
        direct_seeds = [e for e in entities if e not in overlap_info]
        for d in _multihop_edges(
            conn, cfg, direct_seeds,
            edge_rows=graph_rows,
        ).values():
            c = _ensure(d["row"])
            c["multihop_score"] = max(c["multihop_score"], d["path_score"])
            c["hop"] = d["hop"]

    # Recency-only seeding: if the fallback path has no candidates at all
    # (no entity match, no semantic hit), pull a small set of recent active
    # edges so the graph_facts list isn't empty when something could be shown.
    if fallback and not candidates:
        for row in graph_rows:
            _ensure(row)

    results: list[GraphFact] = []
    for c in candidates.values():
        positive = max(0, int(c["pos"]))
        negative = max(0, int(c["neg"]))
        confidence = (positive + 1.0) / (positive + negative + 2.0)
        recency_weight = _recency_weight(
            c["days_since"], cfg.graph_recency_half_life_days
        )
        semantic_score = (
            min(1.0, max(0.0, float(c["semantic_score"])))
            if math.isfinite(float(c["semantic_score"])) else 0.0
        )
        in_routed = c["p"] in routed
        # A candidate reached ONLY via multi-hop chaining (not a direct entity
        # anchor, semantic hit, or routed predicate) scores by its compounding
        # path_score — always below a 1-hop hit by construction (decay < 1), so
        # the additive invariant holds: bridged edges add, never displace.
        multihop_only = (
            c["multihop_score"] > 0.0
            and not c["entity_match"]
            and not c["semantic_retrieved"]
            and not in_routed
        )

        why: list[str] = []
        if multihop_only:
            path_score = float(c["multihop_score"])
            score = (
                path_score * recency_weight
                if math.isfinite(path_score) and path_score > 0.0 else 0.0
            )
            why.append(f"fallback:multihop:{c['hop']}hop")
            why.append(f"score:multihop(path={max(0.0, path_score):.3f})")
        elif fallback:
            if c["entity_match"]:
                overlap_only = not c["direct_anchor"]
                if overlap_only:
                    try:
                        overlap_weight = float(cfg.graph_token_overlap_weight)
                    except (TypeError, ValueError, OverflowError):
                        overlap_weight = 0.0
                    if not math.isfinite(overlap_weight):
                        overlap_weight = 0.0
                    overlap_weight = min(1.0, max(0.0, overlap_weight))
                    score = overlap_weight * confidence * recency_weight
                    why.append("fallback:entity_anchored:overlap")
                    why.append(f"score:overlap(x{overlap_weight:.2f})")
                    for tok in sorted(c["overlap_tokens"]):
                        why.append(f"overlap_via:{tok}")
                else:
                    # Semantic lookup is skipped for a named-entity fallback,
                    # so the direct anchor needs no artificial multiplier.
                    score = confidence * recency_weight
                    why.append("fallback:entity_anchored")
            elif semantic_score > 0:
                score = semantic_score * confidence * recency_weight
                why.append("fallback:semantic")
                why.append(f"score:semantic(x{semantic_score:.3f})")
            else:
                score = confidence * recency_weight
                why.append("fallback:recency")
        else:
            base = confidence * recency_weight
            relevance = 1.0
            if c["entity_match"]:
                if c["direct_anchor"]:
                    relevance += 1.0
                    why.append("score:entity(+1.00base)")
                else:
                    try:
                        overlap_weight = float(cfg.graph_token_overlap_weight)
                    except (TypeError, ValueError, OverflowError):
                        overlap_weight = 0.0
                    if not math.isfinite(overlap_weight):
                        overlap_weight = 0.0
                    overlap_weight = min(1.0, max(0.0, overlap_weight))
                    relevance += overlap_weight
                    why.append(f"score:overlap(+{overlap_weight:.2f}base)")
            if c["semantic_retrieved"] and semantic_score > 0.0:
                # Semantic evidence is additive: observing a valid similarity
                # can improve a routed candidate, never penalize it relative to
                # an otherwise identical candidate with no stored vector.
                relevance += semantic_score
                why.append(f"score:semantic(+{semantic_score:.3f}base)")
            try:
                configured_boost = float(cfg.graph_predicate_boost)
            except (TypeError, ValueError, OverflowError):
                configured_boost = 1.0
            predicate_boost = (
                max(0.0, configured_boost)
                if math.isfinite(configured_boost) and in_routed else 1.0
            )
            score = base * relevance * predicate_boost
            if in_routed:
                why.append(f"score:predicate(x{predicate_boost:.2f})")

        if not math.isfinite(score) or score < 0.0:
            score = 0.0
        why.append(
            f"score:base(confidence={confidence:.3f},recency={recency_weight:.3f})"
        )

        if c["semantic_retrieved"] and semantic_score > 0.0:
            why.append(f"semantic_{semantic_score:.2f}")
        if in_routed:
            why.append(f"predicate:{c['p']}")
        for entity_type in sorted(c["entity_types"]):
            why.append(f"entity_type:{entity_type}")
        recent_days = _nonnegative_finite_config(cfg.graph_recency_recent_days)
        if recent_days is not None and c["days_since"] <= recent_days:
            why.append(f"recency_{round(c['days_since'])}d")
        if c["entity_match"]:
            why.append("entity_match")

        if c["authority_kind"] != "canonical":
            why.append(f"compat:materialized_{c['authority_kind']}")

        total_evidence = positive + negative
        hedge = (
            confidence < cfg.hedge_confidence_threshold
            or total_evidence < cfg.hedge_min_evidence
        )
        results.append(
            GraphFact(
                subject=c["s"],
                predicate=c["p"],
                object=c["o"],
                confidence=confidence,
                pos_evidence=positive,
                neg_evidence=negative,
                derived=c["derived"],
                why_retrieved=why,
                score=score,
                hedge_recommended=hedge,
                edge_id=c["edge_id"],
                valid_at=c["valid_at"],
                invalid_at=c["invalid_at"],
                citations=list(c["citations"]),
            )
        )

    results.sort(key=lambda fact: (
        -fact.score,
        fact.subject,
        fact.predicate,
        fact.object,
    ))
    return results[:graph_limit]


def _query_mentions_canonical(query: str, canonical: str) -> bool:
    """Conservative Unicode-aware literal mention without global alias state."""
    # ``[^\W_]`` means any Unicode word character except underscore. It keeps
    # Greek, CJK, Cyrillic, and other scripts intact while treating canonical
    # separators (``_``, punctuation, whitespace) uniformly as boundaries.
    def words(value: str) -> list[str]:
        folded = _fold_diacritics(value).casefold()
        return re.findall(r"[^\W_]+", folded, flags=re.UNICODE)

    query_words = words(query)
    entity_words = words(canonical)
    if not query_words or not entity_words:
        return False
    width = len(entity_words)
    return any(
        query_words[index:index + width] == entity_words
        for index in range(len(query_words) - width + 1)
    )


def _scoped_graph_lookup(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    query: str,
    entities: list[str],
    expansion_info: dict[str, str],
    routed: frozenset[str],
    *,
    overlap_info: dict[str, str] | None,
    embedding_client: EmbeddingClient | None,
    query_vector: object,
    source_session_id: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
) -> list[GraphFact]:
    """Derive graph truth from one provenance partition before ranking.

    ``knowledge_graph`` is a global materialized cache. Its status, counters,
    interval, and recency can all be changed by another Honcho peer/workspace,
    so a scoped API must not use them. This path starts from exact canonical
    evidence, validates every retained source, replays only in-scope lifecycle
    events, computes weighted confidence/recency locally, and only then applies
    ``graph_top_k``.
    """
    graph_limit = _nonnegative_int_config(cfg.graph_top_k)
    if graph_limit == 0:
        return []
    rows = validated_current_evidence(
        conn,
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    evidence_by_edge: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        evidence_by_edge.setdefault(int(row["edge_id"]), []).append(row)
    if not evidence_by_edge:
        return []

    now_jd = float(conn.execute("SELECT julianday('now')").fetchone()[0])
    scoped_rows: list[dict[str, object]] = []
    surface_forms: dict[str, set[str]] = {}
    for edge_id in sorted(evidence_by_edge):
        evidence_rows = evidence_by_edge[edge_id]
        exemplar = evidence_rows[0]
        positive = sum(
            int(row["evidence_weight"])
            for row in evidence_rows if int(row["polarity"]) == 1
        )
        negative = sum(
            int(row["evidence_weight"])
            for row in evidence_rows if int(row["polarity"]) == -1
        )
        if positive <= negative:
            continue
        state = current_positive_state(
            conn,
            edge_id,
            limit=max(5, len(evidence_rows)),
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
        )
        validated_ids = {int(row["evidence_id"]) for row in evidence_rows}
        citations = [
            citation for citation in (state[1] if state is not None else [])
            if citation.evidence_id in validated_ids
        ]
        valid_coordinates = [
            citation.source_event_at for citation in citations
            if citation.source_event_at is not None
        ]
        if state is None or not citations or not valid_coordinates:
            continue
        last_event_jd = max(float(row["event_jd"]) for row in evidence_rows)
        days_since = max(0.0, now_jd - last_event_jd)
        if not math.isfinite(days_since):
            continue
        subject = str(exemplar["s"])
        object_ = str(exemplar["o"])
        scoped_rows.append({
            "id": edge_id,
            "s": subject,
            "p": str(exemplar["p"]),
            "o": object_,
            "pos": positive,
            "neg": negative,
            "derived": bool(exemplar["derived"]),
            "valid_at": min(valid_coordinates),
            "invalid_at": None,
            "days_since": days_since,
            "citations": citations[:5],
            "authority_kind": "canonical",
        })
        open_positive_ids = {citation.evidence_id for citation in citations}
        for row in evidence_rows:
            if (
                int(row["evidence_id"]) not in open_positive_ids
                or int(row["polarity"]) != 1
            ):
                continue
            for canonical, surface_field in (
                (subject, "surface_subject"),
                (object_, "surface_object"),
            ):
                surface = row[surface_field]
                if isinstance(surface, str) and surface.strip():
                    surface_forms.setdefault(canonical, set()).add(surface)

    if not scoped_rows:
        return []
    scoped_rows.sort(key=lambda row: (row["s"], row["p"], row["o"]))
    rows_by_id = {int(row["id"]): row for row in scoped_rows}
    local_entities = sorted({
        str(row[field])
        for row in scoped_rows
        for field in ("s", "o")
        if _query_mentions_canonical(query, str(row[field]))
        or any(
            _query_mentions_canonical(query, surface)
            for surface in surface_forms.get(str(row[field]), set())
        )
    })

    fallback = not routed
    semantic_scores: dict[int, float] = {}
    if embedding_client is not None and not (fallback and local_entities):
        for edge_id, score in _semantic_edge_hits(
            conn, cfg, embedding_client, query,
            allowed_edge_ids=frozenset(rows_by_id),
            query_vector=query_vector,
        ):
            try:
                score = float(score)
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                edge_id in rows_by_id
                and math.isfinite(score)
                and score > 0.0
            ):
                semantic_scores[int(edge_id)] = min(1.0, score)

    candidate_ids: set[int] = set()
    for row in scoped_rows:
        edge_id = int(row["id"])
        if row["s"] in local_entities or row["o"] in local_entities:
            candidate_ids.add(edge_id)
        if row["p"] in routed:
            candidate_ids.add(edge_id)
    candidate_ids.update(semantic_scores)

    multihop: dict[int, tuple[float, int]] = {}
    if cfg.graph_multihop_enabled and local_entities:
        for item in _multihop_edges(
            conn, cfg, local_entities, edge_rows=scoped_rows
        ).values():
            edge_id = int(item["row"]["id"])
            candidate_ids.add(edge_id)
            multihop[edge_id] = (float(item["path_score"]), int(item["hop"]))

    if fallback and not local_entities and not semantic_scores:
        candidate_ids.update(rows_by_id)
    if not candidate_ids:
        return []

    try:
        configured_boost = float(cfg.graph_predicate_boost)
    except (TypeError, ValueError, OverflowError):
        configured_boost = 1.0
    facts: list[GraphFact] = []
    for edge_id in sorted(candidate_ids):
        row = rows_by_id.get(edge_id)
        if row is None:
            continue
        positive = int(row["pos"])
        negative = int(row["neg"])
        confidence = (positive + 1.0) / (positive + negative + 2.0)
        recency = _recency_weight(
            row["days_since"], cfg.graph_recency_half_life_days
        )
        base = confidence * recency
        anchored = [
            entity for entity in local_entities
            if entity == row["s"] or entity == row["o"]
        ]
        predicate_match = row["p"] in routed
        semantic = semantic_scores.get(edge_id, 0.0)
        multihop_state = multihop.get(edge_id)
        multihop_only = (
            multihop_state is not None
            and not anchored
            and not predicate_match
            and semantic <= 0.0
        )
        why = ["scoped:canonical_evidence"]
        if multihop_only:
            path_score, hop = multihop_state
            score = path_score * recency
            why.extend((
                f"fallback:multihop:{hop}hop",
                f"score:multihop(path={path_score:.3f})",
            ))
        elif fallback:
            if anchored:
                score = base
                why.append("fallback:entity_anchored")
            elif semantic > 0.0:
                score = base * semantic
                why.extend((
                    "fallback:semantic",
                    f"score:semantic(x{semantic:.3f})",
                ))
            else:
                score = base
                why.append("fallback:recency")
        else:
            relevance = 1.0
            if anchored:
                relevance += 1.0
                why.append("score:entity(+1.00base)")
            if semantic > 0.0:
                relevance += semantic
                why.append(f"score:semantic(+{semantic:.3f}base)")
            predicate_boost = (
                max(0.0, configured_boost)
                if predicate_match and math.isfinite(configured_boost) else 1.0
            )
            score = base * relevance * predicate_boost
            if predicate_match:
                why.append(f"score:predicate(x{predicate_boost:.2f})")
        if not math.isfinite(score) or score < 0.0:
            score = 0.0
        why.append(
            f"score:base(confidence={confidence:.3f},recency={recency:.3f})"
        )
        if semantic > 0.0:
            why.append(f"semantic_{semantic:.2f}")
        if predicate_match:
            why.append(f"predicate:{row['p']}")
        if anchored:
            why.append("entity_match")
            if row["s"] in anchored:
                why.append("entity_match:subject")
            if row["o"] in anchored:
                why.append("entity_match:object")
        recent_days = _nonnegative_finite_config(cfg.graph_recency_recent_days)
        if (
            recent_days is not None
            and float(row["days_since"]) <= recent_days
        ):
            why.append(f"recency_{round(float(row['days_since']))}d")
        total = positive + negative
        facts.append(GraphFact(
            subject=str(row["s"]),
            predicate=str(row["p"]),
            object=str(row["o"]),
            confidence=confidence,
            pos_evidence=positive,
            neg_evidence=negative,
            derived=bool(row["derived"]),
            why_retrieved=why,
            score=score,
            hedge_recommended=(
                confidence < cfg.hedge_confidence_threshold
                or total < cfg.hedge_min_evidence
            ),
            edge_id=edge_id,
            valid_at=str(row["valid_at"]),
            invalid_at=None,
            citations=list(row["citations"]),
        ))
    facts.sort(key=lambda fact: (
        -fact.score, fact.subject, fact.predicate, fact.object
    ))
    return facts[:graph_limit]


def _multihop_edges(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    seeds: list[str],
    *,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
    edge_rows: list[dict[str, object]] | None = None,
) -> dict[tuple[str, str, str], dict]:
    """Read-only BFS outward from seed entities up to `cfg.graph_multihop_max_hops`
    edges, returning the bridging edges that 1-hop retrieval (Source 1) misses.

    Each returned edge carries a compounding `path_score` = product over the chain
    of (smoothed_confidence × `graph_multihop_decay`), so a longer chain is strictly
    weaker than a shorter one. Only edges at hop >= 2 are returned: the first BFS
    round traverses the seeds' own 1-hop edges, which Source 1 already retrieves —
    re-emitting them would double-count. (The Idea-A sketch's `range(1, max_hops)`
    stopped one round short and mislabeled those 1-hop edges as hop-2; the loop
    below runs `max_hops` rounds and emits from round 2 on, so `max_hops=2` reaches
    the true 2-hop bridge — verified by tests/test_multihop.py.)

    Returns `{(s, p, o): {"row": sqlite3.Row, "path_score": float, "hop": int}}`.
    """
    max_hops = _nonnegative_int_config(cfg.graph_multihop_max_hops)
    if max_hops < 2 or not seeds:
        return {}

    # The unscoped path supplies its already validated authority projection.
    # Direct callers get the same projection instead of independently trusting
    # materialized status/counters. Scoped graph lookup supplies its validated
    # local projection, so unrelated workspace topology cannot create bridges.
    if edge_rows is None and not any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    ):
        edge_rows = _current_graph_rows(conn)

    try:
        decay = float(cfg.graph_multihop_decay)
        min_score = float(cfg.graph_multihop_min_score)
    except (TypeError, ValueError, OverflowError):
        return {}
    if (
        not math.isfinite(decay)
        or not math.isfinite(min_score)
        or decay <= 0.0
        or decay >= 1.0
    ):
        return {}
    min_score = max(0.0, min_score)

    seeds_set = set(seeds)
    reached: dict[str, float] = {s: 1.0 for s in seeds}  # node -> best path score
    out: dict[tuple[str, str, str], dict] = {}
    frontier = list(seeds)
    scope_sql, scope_params = _edge_provenance_scope_sql(
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )

    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        if edge_rows is not None:
            frontier_set = set(frontier)
            rows = sorted(
                (
                    row for row in edge_rows
                    if row["s"] in frontier_set or row["o"] in frontier_set
                ),
                key=lambda row: (row["s"], row["p"], row["o"]),
            )
        else:
            ph = ",".join("?" * len(frontier))
            rows = conn.execute(
                _EDGE_SELECT
                + f"""
                WHERE {live_edge_predicate()}
                  AND (subject_canonical IN ({ph}) OR object_canonical IN ({ph}))
                  {scope_sql}
                ORDER BY subject_canonical,predicate,object_canonical
                """,
                [*frontier, *frontier, *scope_params],
            ).fetchall()

        next_scores: dict[str, float] = {}
        for r in rows:
            positive = max(0, int(r["pos"]))
            negative = max(0, int(r["neg"]))
            conf = (positive + 1.0) / (positive + negative + 2.0)
            # Never emit a seed-incident edge: it is 1-hop from a seed and Source 1
            # already has it (emitting would double-count / mislabel it as a bridge).
            seed_incident = r["s"] in seeds_set or r["o"] in seeds_set
            for near, far in ((r["s"], r["o"]), (r["o"], r["s"])):
                if near not in reached:
                    continue
                path_score = reached[near] * conf * decay
                if not math.isfinite(path_score) or path_score < min_score:
                    continue
                if hop >= 2 and not seed_incident:  # emit true bridges only
                    key = (r["s"], r["p"], r["o"])
                    prev = out.get(key)
                    if prev is None or path_score > prev["path_score"]:
                        out[key] = {"row": r, "path_score": path_score, "hop": hop}
                if far not in reached or path_score > reached[far]:
                    reached[far] = path_score
                    if path_score > next_scores.get(far, 0.0):
                        next_scores[far] = path_score
        # Advance the strongest frontier nodes into the next hop — but apply the
        # hub guard first. A super-hub (degree > graph_multihop_hub_degree_max) is
        # REACHED (it stays in `reached`, so an edge INTO it can still emit) but is
        # never EXPANDED: fanning out from it would make every one of its leaves a
        # 2-hop "bridge" of every other leaf (`road_trip ← user → driving_trip`),
        # flooding graph_top_k with hub-mediated non-bridges and diluting the true
        # bridge out of recall. A genuine intermediate (degree ~2) is far below the
        # cap, so real chains still bridge. `<= 0` disables the guard.
        candidates = sorted(next_scores, key=lambda node: (-next_scores[node], node))
        if cfg.graph_multihop_hub_degree_max > 0 and candidates:
            probe = candidates
            degrees = _active_degrees(conn, probe, edge_rows=edge_rows)
            candidates = [
                n
                for n in probe
                if degrees.get(n, 0) <= cfg.graph_multihop_hub_degree_max
            ]
        frontier = candidates
    return out


def _active_degrees(
    conn: sqlite3.Connection,
    nodes: list[str],
    *,
    edge_rows: list[dict[str, object]] | None = None,
) -> dict[str, int]:
    """Active degree (count of active edges where the node is subject OR object)
    for each node, in a single query — the hub guard's fan-out test in
    `_multihop_edges`. A node absent from the result has degree 0. (A self-loop
    edge, subject==object, is counted twice; those are effectively absent in this
    graph, so degree == incident-edge count in practice.)"""
    if not nodes:
        return {}
    if edge_rows is not None:
        wanted = set(nodes)
        degrees = {node: 0 for node in wanted}
        for row in edge_rows:
            if row["s"] in wanted:
                degrees[str(row["s"])] += 1
            if row["o"] in wanted:
                degrees[str(row["o"])] += 1
        return degrees
    ph = ",".join("?" * len(nodes))
    rows = conn.execute(
        f"""
        SELECT node, COUNT(*) AS deg FROM (
            SELECT subject_canonical AS node FROM knowledge_graph
             WHERE {live_edge_predicate()} AND subject_canonical IN ({ph})
            UNION ALL
            SELECT object_canonical AS node FROM knowledge_graph
             WHERE {live_edge_predicate()} AND object_canonical IN ({ph})
        ) GROUP BY node
        """,
        nodes + nodes,
    ).fetchall()
    return {r["node"]: r["deg"] for r in rows}


def _recency_edges(
    conn: sqlite3.Connection,
    limit: int,
    *,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[sqlite3.Row]:
    """Pull the most recent active edges by confidence × recency.

    Used by the no-predicate fallback when neither entity match nor semantic
    KNN produced any candidates, so something graph-shaped is still returned.
    """
    scope_sql, scope_params = _edge_provenance_scope_sql(
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    return conn.execute(
        _EDGE_SELECT
        + f"""
        WHERE {live_edge_predicate()} {scope_sql}
        ORDER BY (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) DESC,
                 {graph_clock_order_sql('last_seen')}, id
        LIMIT ?
        """,
        (*scope_params, limit),
    ).fetchall()


def _semantic_edge_hits(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    embedder: EmbeddingClient,
    query: str,
    *,
    allowed_edge_ids: frozenset[int] | None = None,
    query_vector: object = _QUERY_VECTOR_UNSET,
) -> list[tuple[int, float]]:
    """Return compatible edge similarities in a common finite ``(0, 1]`` unit.

    The durable JSON mirror carries model/dimension identity; ``vec_edges``
    does not.  We therefore use vec0 only when its row-id coverage is complete,
    and always validate/score the matching durable vectors.  A stale vec row,
    mixed-model corpus, or malformed vector can never consume the semantic
    candidate cap.  All compatible candidates are returned because applying a
    semantic-only LIMIT before confidence/recency scoring can discard the true
    final winner.
    """
    from hymem.core import db as core_db

    try:
        if int(cfg.graph_semantic_top_k) <= 0:
            return []
    except (TypeError, ValueError, OverflowError):
        return []
    if allowed_edge_ids is None:
        allowed_edge_ids = frozenset(
            int(row["id"]) for row in _current_graph_rows(conn)
        )
    rows = _compatible_edge_embedding_rows(conn, embedder, allowed_edge_ids)
    if not rows:
        return []
    if query_vector is _QUERY_VECTOR_UNSET:
        qvec = _query_embedding(embedder, query)
    elif isinstance(query_vector, list):
        qvec = _finite_vector(query_vector, expected_dim=embedder.dim)
    else:
        qvec = None
    if qvec is None:
        return []

    if core_db._load_vec_extension(conn) and core_db.has_vec_table(
        conn, table="vec_edges"
    ):
        dim_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='vec_dim'"
        ).fetchone()
        try:
            vec_dim = int(dim_row[0]) if dim_row is not None else -1
            physical_count = int(
                conn.execute("SELECT COUNT(*) FROM vec_edges").fetchone()[0]
            )
        except (TypeError, ValueError, OverflowError, sqlite3.Error):
            vec_dim = -1
            physical_count = 0
        if vec_dim == len(qvec) and physical_count > 0:
            hits = core_db.vec_search(
                conn, qvec, physical_count, table="vec_edges"
            )
            complete_ids: set[int] = set()
            for edge_id, distance in hits:
                similarity = _distance_to_unit_similarity(distance)
                if similarity is not None:
                    complete_ids.add(int(edge_id))
            expected_ids = {int(row["edge_id"]) for row in rows}
            if not expected_ids.issubset(complete_ids):
                log.warning(
                    "vec_edges coverage is stale; exact durable edge scan used"
                )
        # Scoring below deliberately uses the durable vectors even on the vec0
        # branch: that is the only place model/dimension metadata exists, and it
        # makes sqlite-vec and fallback ranking mathematically identical.
        return _score_edge_vectors(
            rows, qvec, top_k=None, embedder=embedder, query=query
        )

    return _python_cosine_edge_search(
        conn, embedder, query,
        top_k=cfg.graph_semantic_top_k, max_scan=cfg.embedding_max_scan,
        query_vector=qvec,
        allowed_edge_ids=allowed_edge_ids,
        return_all=True,
    )


def _python_cosine_edge_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
    query_vector: list[float] | None = None,
    allowed_edge_ids: frozenset[int] | None = None,
    return_all: bool = False,
) -> list[tuple[int, float]]:
    """Exact durable cosine fallback for graph edges.

    ``max_scan`` is retained for API compatibility but is intentionally not a
    pre-ranking LIMIT: a recency-bounded scan can silently omit the best
    semantic candidate.  The graph source is already bounded by its persisted
    edge corpus, and final ``graph_top_k`` is applied after shared scoring.
    """
    if allowed_edge_ids is None:
        allowed_edge_ids = frozenset(
            int(row["id"]) for row in _current_graph_rows(conn)
        )
    rows = _compatible_edge_embedding_rows(conn, embedder, allowed_edge_ids)
    if not rows:
        return []
    qvec = query_vector if query_vector is not None else _query_embedding(embedder, query)
    if qvec is None:
        return []
    limit = None if return_all else max(0, int(top_k))
    return _score_edge_vectors(
        rows, qvec, top_k=limit, embedder=embedder, query=query
    )


def _compatible_edge_embedding_rows(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    allowed_edge_ids: frozenset[int],
) -> list[sqlite3.Row]:
    """Load only vectors produced by the active embedding identity."""
    try:
        model = embedder.model
        dim = embedder.dim
    except Exception:
        return []
    if (
        not isinstance(model, str)
        or not model
        or isinstance(dim, bool)
        or not isinstance(dim, int)
        or dim <= 0
        or not allowed_edge_ids
    ):
        return []
    rows = conn.execute(
        """
        SELECT kg.id AS edge_id, kg.subject_canonical AS s,
               kg.predicate AS p, kg.object_canonical AS o,
               e.vector_json,e.model,e.dim
        FROM knowledge_graph kg
        JOIN edge_embeddings e
          ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate || ' '
                           || kg.object_canonical
        WHERE e.model=? AND e.dim=?
        ORDER BY kg.subject_canonical,kg.predicate,kg.object_canonical
        """,
        (model, dim),
    ).fetchall()
    return [row for row in rows if int(row["edge_id"]) in allowed_edge_ids]


def _query_embedding(
    embedder: EmbeddingClient, query: str
) -> list[float] | None:
    """Embed once and validate against the identity *after* the response.

    OpenAI-compatible clients may learn their true dimension from the first
    response, so snapshotting ``dim`` before ``embed`` rejects a valid first
    call and then mysteriously succeeds on the second.
    """
    return _query_embedding_with_status(embedder, query)[0]


def _query_embedding_with_status(
    embedder: EmbeddingClient, query: str
) -> tuple[list[float] | None, SemanticStatus]:
    backend = _safe_embedding_attr(embedder, "backend", "configured")
    quality = _safe_embedding_attr(embedder, "quality", "semantic")
    fallback_reason_value = _safe_embedding_attr(embedder, "fallback_reason", "")
    fallback_reason = fallback_reason_value or None
    if not isinstance(query, str) or not query.strip():
        return None, SemanticStatus(
            configured=True,
            attempted=False,
            available=False,
            backend=backend,
            quality=quality,
            reason="blank_query",
            fallback_reason=fallback_reason,
        )
    model: str | None = None
    dim: int | None = None
    try:
        model_before = embedder.model
        dim_before = embedder.dim
        model = model_before if isinstance(model_before, str) else None
        dim = (
            dim_before
            if isinstance(dim_before, int) and not isinstance(dim_before, bool)
            else None
        )
        if (
            not isinstance(model_before, str) or not model_before
            or isinstance(dim_before, bool)
            or not isinstance(dim_before, int)
            or dim_before <= 0
        ):
            raise ValueError("invalid embedding identity")
    except Exception as exc:
        log.warning(
            "semantic retrieval unavailable for this query: backend=%s error=%s",
            backend, type(exc).__name__,
        )
        return None, SemanticStatus(
            configured=True, attempted=False, available=False,
            backend=backend, quality=quality, model=model, dim=dim,
            reason="invalid_identity", fallback_reason=fallback_reason,
        )

    try:
        batch = embedder.embed([query])
    except Exception as exc:
        log.warning(
            "semantic retrieval unavailable for this query: backend=%s error=%s",
            backend, type(exc).__name__,
        )
        return None, SemanticStatus(
            configured=True, attempted=True, available=False,
            backend=backend, quality=quality, model=model, dim=dim,
            reason="provider_error", fallback_reason=fallback_reason,
        )

    try:
        model_after = embedder.model
        expected_dim = embedder.dim
        model = model_after if isinstance(model_after, str) else model
        dim = (
            expected_dim
            if isinstance(expected_dim, int) and not isinstance(expected_dim, bool)
            else dim
        )
        if len(batch) != 1:
            raise ValueError("wrong batch cardinality")
        if (
            model_after != model_before
            or isinstance(expected_dim, bool)
            or not isinstance(expected_dim, int)
            or expected_dim <= 0
        ):
            raise ValueError("invalid or changing embedding identity")
        vector = _finite_vector(batch[0], expected_dim=expected_dim)
        if vector is None:
            raise ValueError("malformed query vector")
    except Exception as exc:
        log.warning(
            "semantic retrieval unavailable for this query: backend=%s error=%s",
            backend, type(exc).__name__,
        )
        return None, SemanticStatus(
            configured=True,
            attempted=True,
            available=False,
            backend=backend,
            quality=quality,
            model=model,
            dim=dim,
            reason="malformed_vector",
            fallback_reason=fallback_reason,
        )
    return vector, SemanticStatus(
        configured=True,
        attempted=True,
        available=True,
        backend=backend,
        quality=quality,
        model=model,
        dim=dim,
        reason="ready",
        fallback_reason=fallback_reason,
    )


def _finite_vector(value: object, *, expected_dim: int) -> list[float] | None:
    if (
        isinstance(expected_dim, bool)
        or not isinstance(expected_dim, int)
        or expected_dim <= 0
        or not isinstance(value, (list, tuple))
        or len(value) != expected_dim
    ):
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    return vector if math.isfinite(norm) and norm > 0.0 else None


def _decode_finite_vector(value: object, *, expected_dim: int) -> list[float] | None:
    try:
        decoded = decode_vector(value)  # type: ignore[arg-type]
    except (AttributeError, UnicodeError, ValueError, TypeError, struct.error):
        return None
    return _finite_vector(decoded, expected_dim=expected_dim)


def _resolved_query_vector(
    embedder: EmbeddingClient,
    query: str,
    supplied: object,
    *,
    expected_dim: int,
) -> list[float] | None:
    """Resolve a direct-call vector or reuse augment's one-shot result.

    ``None`` is an explicit failed/abstained result and must never trigger a
    second provider call. Only the private sentinel means the caller omitted a
    shared vector (legacy/direct helper use).
    """
    if supplied is _QUERY_VECTOR_UNSET:
        return _query_embedding(embedder, query)
    if not isinstance(supplied, list):
        return None
    return _finite_vector(supplied, expected_dim=expected_dim)


def _durable_cosine(
    query_vector: list[float], stored_vector: object
) -> float | None:
    vector = _decode_finite_vector(
        stored_vector, expected_dim=len(query_vector)
    )
    if vector is None:
        return None
    qnorm = math.sqrt(sum(value * value for value in query_vector))
    vnorm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(qnorm) or not math.isfinite(vnorm) or qnorm <= 0 or vnorm <= 0:
        return None
    similarity = sum(
        left * right for left, right in zip(query_vector, vector)
    ) / (qnorm * vnorm)
    if not math.isfinite(similarity) or similarity <= 0.0:
        return None
    return min(1.0, similarity)


_LOCAL_LEXICAL_STOPWORDS = frozenset({
    "and", "are", "but", "for", "from", "has", "have", "into", "not",
    "that", "the", "their", "then", "this", "was", "were", "what",
    "when", "where", "which", "with", "you", "your",
})


def _safe_embedding_attr(
    embedder: EmbeddingClient, name: str, default: str
) -> str:
    try:
        value = getattr(embedder, name, default)
        return default if value is None else str(value)
    except Exception:
        return default


def _quality_allows_candidate(
    embedder: EmbeddingClient, query: str, candidate_text: str
) -> bool:
    """Fail closed on feature-hash collisions from the local lexical backend.

    A true semantic model may bridge vocabulary gaps.  The dependency-free
    fallback cannot: positive cosine without actual word/near-word overlap is
    only a hash collision and must never be presented as semantic evidence.
    """
    if _safe_embedding_attr(embedder, "quality", "semantic") != "lexical":
        return True

    def words(text: str) -> list[str]:
        folded = _fold_diacritics(text).casefold()
        return [
            token for token in re.findall(r"[^\W_]+", folded, flags=re.UNICODE)
            if len(token) >= 3 and token not in _LOCAL_LEXICAL_STOPWORDS
        ]

    query_words = words(query)
    candidate_words = words(candidate_text)
    if not query_words or not candidate_words:
        return False
    if set(query_words) & set(candidate_words):
        return True

    def grams(word: str) -> set[str]:
        padded = f"^{word}$"
        return {padded[index:index + 3] for index in range(len(padded) - 2)}

    for left in query_words:
        if len(left) < 4:
            continue
        left_grams = grams(left)
        for right in candidate_words:
            if len(right) < 4:
                continue
            right_grams = grams(right)
            overlap = len(left_grams & right_grams)
            if overlap >= 2 and overlap / min(len(left_grams), len(right_grams)) >= 0.6:
                return True
    return False


def _score_edge_vectors(
    rows: list[sqlite3.Row],
    query_vector: list[float],
    *,
    top_k: int | None,
    embedder: EmbeddingClient | None = None,
    query: str = "",
) -> list[tuple[int, float]]:
    """Score cosine in one normalized unit and natural-key tie order."""
    qnorm = math.sqrt(sum(item * item for item in query_vector))
    if not math.isfinite(qnorm) or qnorm <= 0.0:
        return []
    scored: list[tuple[int, float, str, str, str]] = []
    for row in rows:
        edge_text = f"{row['s']} {row['p']} {row['o']}"
        if embedder is not None and not _quality_allows_candidate(
            embedder, query, edge_text
        ):
            continue
        vector = _decode_finite_vector(
            row["vector_json"], expected_dim=len(query_vector)
        )
        if vector is None:
            continue
        vnorm = math.sqrt(sum(item * item for item in vector))
        cosine = sum(
            left * right for left, right in zip(query_vector, vector)
        ) / (qnorm * vnorm)
        if not math.isfinite(cosine):
            continue
        # Cosine is already a higher-is-better unit. Orthogonal/negative
        # vectors are not semantic evidence and must not enter the candidate
        # pool under a misleading positive score.
        if cosine <= 0.0:
            continue
        similarity = min(1.0, cosine)
        scored.append((
            int(row["edge_id"]), similarity,
            str(row["s"]), str(row["p"]), str(row["o"]),
        ))
    scored.sort(key=lambda item: (-item[1], item[2], item[3], item[4]))
    if top_k is not None:
        scored = scored[:top_k]
    return [(edge_id, similarity) for edge_id, similarity, *_ in scored]


def _distance_to_unit_similarity(distance: object) -> float | None:
    """Convert a non-negative vec0 distance to finite higher-is-better units."""
    try:
        value = float(distance)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    similarity = 1.0 / (1.0 + value)
    return min(1.0, max(0.0, similarity))


def _episode_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 3,
    embedding_client: EmbeddingClient | None = None,
    query_vector: object = _QUERY_VECTOR_UNSET,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[EpisodeHit]:
    """Episode retrieval. Always runs the FTS path; when an embedding client
    is configured *and* vec_episodes has rows, also runs semantic KNN over
    title+summary embeddings and RRF-fuses the two ranked lists.

    Falls back to FTS-only (with score_kind="bm25") when there's no embedder,
    no vec_episodes table, or vec returns nothing — preserving the original
    behavior for clients that haven't dreamed any episode embeddings yet.
    """
    from hymem.core import db as core_db

    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    scope_sql, scope_params = _derived_scope_sql(
        "e", "start_message_id", "end_message_id",
        source_session_id=source_session_id,
        source_peer_id=source_peer_id,
        source_workspace_id=source_workspace_id,
    )
    fts_hits: list[EpisodeHit] = []
    if cleaned:
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            episode_chip = f'episode_fts("{" ".join(tokens)}")'
            try:
                rows = conn.execute(
                    f"""SELECT e.id, e.session_id, e.title, e.summary, bm25(episodes_fts) AS score
                       FROM episodes_fts
                       JOIN episodes e ON e.rowid = episodes_fts.rowid
                       JOIN sessions s ON s.id = e.session_id
                       WHERE episodes_fts MATCH ?
                         AND (e.digest_generation IS NULL
                              OR e.digest_generation = s.digest_published_generation)
                         {scope_sql}
                       ORDER BY score, e.id
                       LIMIT ?""",
                    (fts_query, *scope_params, top_k * 2),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for r in rows:
                fts_hits.append(
                    EpisodeHit(
                        episode_id=r["id"],
                        session_id=r["session_id"],
                        title=r["title"],
                        summary=r["summary"],
                        score=float(r["score"]),
                        score_kind="bm25",
                        why_retrieved=[episode_chip],
                    )
                )

    vec_hits: list[EpisodeHit] = []
    if embedding_client is not None:
        try:
            model = embedding_client.model
            dim = embedding_client.dim
            rows = conn.execute(
                f"""
                SELECT e.id, e.session_id, e.title, e.summary,
                       ee.vector_json, ee.text_hash
                FROM episode_embeddings ee
                JOIN episodes e ON e.id = ee.episode_id
                JOIN sessions s ON s.id = e.session_id
                WHERE ee.model=? AND ee.dim=?
                  AND (e.digest_generation IS NULL
                       OR e.digest_generation = s.digest_published_generation)
                  {scope_sql}
                ORDER BY e.id
                """,
                (model, dim, *scope_params),
            ).fetchall()
        except (AttributeError, sqlite3.OperationalError, TypeError, ValueError):
            rows = []
            dim = -1
        qvec = _resolved_query_vector(
            embedding_client, query, query_vector, expected_dim=dim
        )
        scored: list[tuple[float, sqlite3.Row]] = []
        if qvec is not None:
            for row in rows:
                candidate_text = f"{row['title']}\n{row['summary']}"
                if row["text_hash"] != embedding_text_hash(candidate_text):
                    continue
                if not _quality_allows_candidate(
                    embedding_client, query, candidate_text
                ):
                    continue
                similarity = _durable_cosine(qvec, row["vector_json"])
                if similarity is not None:
                    scored.append((similarity, row))
        scored.sort(key=lambda item: (-item[0], str(item[1]["id"])))
        vec_hits = [
            EpisodeHit(
                episode_id=row["id"],
                session_id=row["session_id"],
                title=row["title"],
                summary=row["summary"],
                score=float(similarity),
                score_kind="vec",
                why_retrieved=[f"episode_vec(sim={similarity:.3f})"],
            )
            for similarity, row in scored[:top_k * 2]
        ]

    if not vec_hits:
        return fts_hits[:top_k]
    if not fts_hits:
        return vec_hits[:top_k]
    return _rrf_merge_episodes(fts_hits, vec_hits, top_k=top_k)


def _rrf_merge_episodes(
    fts: list[EpisodeHit],
    vec: list[EpisodeHit],
    *,
    top_k: int,
    k: int = 60,
) -> list[EpisodeHit]:
    """RRF over two ranked episode lists. Mirrors `_rrf_merge` (for chunks)
    but keyed on episode_id."""
    by_id: dict[str, EpisodeHit] = {}
    scores: dict[str, float] = {}
    for rank, hit in enumerate(fts, start=1):
        scores[hit.episode_id] = scores.get(hit.episode_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.episode_id, hit)
    for rank, hit in enumerate(vec, start=1):
        scores[hit.episode_id] = scores.get(hit.episode_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.episode_id, hit)
    fts_ids = {h.episode_id for h in fts}
    vec_ids = {h.episode_id for h in vec}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        replace(
            by_id[eid], score=score, score_kind="rrf",
            why_retrieved=_episode_rrf_chips(
                by_id[eid].why_retrieved, eid, fts_ids, vec_ids, score
            ),
        )
        for eid, score in ordered[:top_k]
    ]


def _episode_rrf_chips(
    base: list[str], item_id: str, fts_ids: set[str], vec_ids: set[str], score: float
) -> list[str]:
    if item_id in fts_ids and item_id in vec_ids:
        sources = "fts+vec"
    elif item_id in fts_ids:
        sources = "fts"
    else:
        sources = "vec"
    return [*base, f"episode_rrf({sources}, {score:.4f})"]


def _fact_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int,
    embedding_client: EmbeddingClient | None = None,
    query_vector: object = _QUERY_VECTOR_UNSET,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[FactHit]:
    """Retrieve only lifecycle-current, cryptographically sourced facts.

    Ownership is filtered against exact manifest rows in SQL before candidate
    ranking. Every survivor then crosses the full coverage/hash/lifecycle
    validator before any external quality hook can inspect its text.
    """
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    scope_clauses = [
        "f.source_outcome_key IS NOT NULL",
        "f.lifecycle_status='active'",
        "f.invalid_at IS NULL",
        "o.source_manifest_complete=1",
        "o.source_manifest_version='fact-source-manifest-v1'",
        "o.source_manifest_count > 0",
    ]
    scope_params: list[object] = []
    if source_session_id is not None:
        scope_clauses.append(
            "NOT EXISTS (SELECT 1 FROM fact_extraction_source_occurrences fs "
            "WHERE fs.slice_key=o.slice_key AND fs.source_session_id<>?)"
        )
        scope_params.append(source_session_id)
    if source_peer_id is not None:
        scope_clauses.append(
            "NOT EXISTS (SELECT 1 FROM fact_extraction_source_occurrences fs "
            "WHERE fs.slice_key=o.slice_key AND fs.source_peer_id IS NOT ?)"
        )
        scope_params.append(source_peer_id)
    if source_workspace_id is not None:
        scope_clauses.append(
            "NOT EXISTS (SELECT 1 FROM fact_extraction_source_occurrences fs "
            "WHERE fs.slice_key=o.slice_key AND fs.source_workspace_id IS NOT ?)"
        )
        scope_params.append(source_workspace_id)
    scope_sql = " AND ".join(scope_clauses)
    page_size = max(64, int(top_k) * 8)
    outcome_proof_cache: OrderedDict[
        str, tuple[BoundSourceOccurrence, ...] | None
    ] = OrderedDict()
    chain_proof_cache: OrderedDict[str, bool | None] = OrderedDict()

    def trim_proof_caches(rows: list[sqlite3.Row]) -> None:
        for row in rows:
            outcome_key = str(row["source_outcome_key"])
            if outcome_key in outcome_proof_cache:
                outcome_proof_cache.move_to_end(outcome_key)
            session_key = str(row["session_id"])
            if session_key in chain_proof_cache:
                chain_proof_cache.move_to_end(session_key)
        while len(outcome_proof_cache) > 64:
            outcome_proof_cache.popitem(last=False)
        while len(chain_proof_cache) > 256:
            chain_proof_cache.popitem(last=False)

    def prove_page(
        rows: list[sqlite3.Row],
    ) -> dict[int, tuple[BoundSourceOccurrence, ...]]:
        # Bounded LRUs avoid both all-corpus retention and rescanning one long
        # session's committed chain on every page.
        resolved = load_fact_source_manifests(
            conn, [int(row["id"]) for row in rows],
            outcome_cache=outcome_proof_cache,
            chain_cache=chain_proof_cache,
        )
        trim_proof_caches(rows)
        return resolved

    fts_hits: list[FactHit] = []
    if cleaned:
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            fact_chip = f'fact_fts("{" ".join(tokens)}")'
            last_score: float | None = None
            last_fts_id = 0
            while len(fts_hits) < top_k * 2:
                take = page_size
                cursor_sql = ""
                cursor_params: tuple[object, ...] = ()
                if last_score is not None:
                    cursor_sql = (
                        "AND (narrative_facts_fts.rank > ? OR "
                        "(narrative_facts_fts.rank = ? AND f.id > ?))"
                    )
                    cursor_params = (last_score, last_score, last_fts_id)
                try:
                    rows = conn.execute(
                        f"""SELECT f.id, f.session_id, f.text, f.fact_date, f.entities,
                                  f.source_outcome_key,
                                  narrative_facts_fts.rank AS score
                           FROM narrative_facts_fts
                           JOIN narrative_facts f
                             ON f.id = narrative_facts_fts.rowid
                           JOIN fact_extraction_outcomes o
                             ON o.slice_key=f.source_outcome_key
                           WHERE narrative_facts_fts MATCH ? AND {scope_sql}
                           {cursor_sql}
                           ORDER BY narrative_facts_fts.rank, f.id
                           LIMIT ?""",
                        (
                            fts_query, *scope_params, *cursor_params, take,
                        ),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                if not rows:
                    break
                last_score = float(rows[-1]["score"])
                last_fts_id = int(rows[-1]["id"])
                page_proofs = prove_page(rows)
                for r in rows:
                    proof = page_proofs.get(int(r["id"]))
                    if proof is None:
                        continue
                    fts_hits.append(_fact_row_hit(
                        r, float(r["score"]), "bm25", [fact_chip],
                        source_occurrences=_query_source_occurrences(proof),
                    ))
                    if len(fts_hits) >= top_k * 2:
                        break
                if len(rows) < take:
                    break

    vec_hits: list[FactHit] = []
    if embedding_client is not None:
        try:
            model = embedding_client.model
            dim = embedding_client.dim
            rows = []
        except (AttributeError, sqlite3.OperationalError, TypeError, ValueError):
            rows = []
            dim = -1
        qvec = _resolved_query_vector(
            embedding_client, query, query_vector, expected_dim=dim
        )
        keep = max(1, int(top_k) * 2)
        scored_heap: list[
            tuple[
                float, int, int, sqlite3.Row,
                tuple[BoundSourceOccurrence, ...],
            ]
        ] = []
        if qvec is not None:
            last_vector_id = 0
            while True:
                take = page_size
                try:
                    rows = conn.execute(
                        f"""
                        SELECT f.id, f.session_id, f.text, f.fact_date,
                               f.entities, f.source_outcome_key,
                               fe.vector_json, fe.text_hash
                        FROM narrative_fact_embeddings fe
                        JOIN narrative_facts f ON f.id=fe.fact_id
                        JOIN fact_extraction_outcomes o
                          ON o.slice_key=f.source_outcome_key
                        WHERE fe.model=? AND fe.dim=? AND f.id>? AND {scope_sql}
                        ORDER BY f.id LIMIT ?
                        """,
                        (
                            model, dim, last_vector_id, *scope_params, take,
                        ),
                    ).fetchall()
                except sqlite3.OperationalError:
                    break
                if not rows:
                    break
                last_vector_id = int(rows[-1]["id"])
                page_proofs = prove_page(rows)
                for row in rows:
                    fact_id = int(row["id"])
                    proof = page_proofs.get(fact_id)
                    if proof is None:
                        continue
                    if row["text_hash"] != embedding_text_hash(row["text"]):
                        continue
                    if not _quality_allows_candidate(
                        embedding_client, query, row["text"]
                    ):
                        continue
                    similarity = _durable_cosine(qvec, row["vector_json"])
                    if similarity is not None:
                        # The first two fields make the min-heap root the worst
                        # survivor (low similarity, then high id).  Fact ids are
                        # unique, so the remaining fields never participate in
                        # tuple ordering.
                        entry = (similarity, -fact_id, fact_id, row, proof)
                        if len(scored_heap) < keep:
                            heapq.heappush(scored_heap, entry)
                        elif entry[:2] > scored_heap[0][:2]:
                            heapq.heapreplace(scored_heap, entry)
                if len(rows) < take:
                    break
        scored_heap.sort(key=lambda item: (-item[0], item[2]))
        vec_hits = [
            _fact_row_hit(
                row, float(similarity), "vec",
                [f"fact_vec(sim={similarity:.3f})"],
                source_occurrences=_query_source_occurrences(proof),
            )
            for similarity, _neg_id, _fact_id, row, proof
            in scored_heap[:top_k * 2]
        ]

    if not vec_hits:
        return fts_hits[:top_k]
    if not fts_hits:
        return vec_hits[:top_k]
    return _rrf_merge_facts(fts_hits, vec_hits, top_k=top_k)


def _fact_row_hit(
    r: sqlite3.Row, score: float, score_kind: str, chips: list[str],
    *, source_occurrences: tuple[SourceOccurrence, ...] = (),
) -> FactHit:
    try:
        entities = json.loads(r["entities"] or "[]")
    except (json.JSONDecodeError, TypeError):
        entities = []
    return FactHit(
        fact_id=int(r["id"]),
        text=r["text"],
        fact_date=r["fact_date"],
        entities=[e for e in entities if isinstance(e, str)],
        session_id=r["session_id"],
        score=score,
        score_kind=score_kind,
        why_retrieved=chips,
        source_occurrences=source_occurrences,
        source_provenance_complete=bool(source_occurrences),
    )


def _rrf_merge_facts(
    fts: list[FactHit],
    vec: list[FactHit],
    *,
    top_k: int,
    k: int = 60,
) -> list[FactHit]:
    """RRF over two ranked fact lists, keyed on fact_id — the
    `_rrf_merge_episodes` pattern."""
    by_id: dict[int, FactHit] = {}
    scores: dict[int, float] = {}
    for rank, hit in enumerate(fts, start=1):
        scores[hit.fact_id] = scores.get(hit.fact_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.fact_id, hit)
    for rank, hit in enumerate(vec, start=1):
        scores[hit.fact_id] = scores.get(hit.fact_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.fact_id, hit)
    fts_ids = {h.fact_id for h in fts}
    vec_ids = {h.fact_id for h in vec}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    out: list[FactHit] = []
    for fid, score in ordered[:top_k]:
        if fid in fts_ids and fid in vec_ids:
            sources = "fts+vec"
        elif fid in fts_ids:
            sources = "fts"
        else:
            sources = "vec"
        base = by_id[fid]
        out.append(replace(
            base, score=score, score_kind="rrf",
            why_retrieved=[*base.why_retrieved, f"fact_rrf({sources}, {score:.4f})"],
        ))
    return out


def _aggregation_tier_fires(cfg: HyMemConfig, ability: str | None) -> bool:
    """Whether the aggregation tier runs for this query's (normalized) ability.
    An empty `aggregation_inject_abilities` means every query (broad mode, for
    A/B re-runs); otherwise the ability must be in the allowlist — so with the
    default `("TR",)` an unrouted question (ability None) gets no nodes."""
    allowed = cfg.aggregation_inject_abilities
    if not allowed:
        return True
    return ability is not None and ability in {a.upper() for a in allowed}


def _raw_signal_count(ctx: AugmentedContext) -> int:
    """How much RAW evidence the tiers above the aggregation gate produced, as
    one named definition so thinness lives in exactly one testable place.

    Counts `message_hits + fts_hits`; `episodes` are deliberately EXCLUDED —
    see `aggregation_fallback_min_hits` for the recorded reasoning and for why
    the alternative must be decided on an argument, not on a better number."""
    return len(ctx.message_hits) + len(ctx.fts_hits)


def _sparse_signal_fires(cfg: HyMemConfig, ctx: AugmentedContext) -> bool:
    """Stage 4a: whether raw retrieval is thin enough to fire the node tier as a
    fallback. Strict `<`, so a query landing exactly ON the threshold does NOT
    fire — the knob reads as "fewer than N hits is starved". `<= 0` disables,
    which is the shipped default, so the condition is inert until set."""
    if cfg.aggregation_fallback_min_hits <= 0:
        return False
    return _raw_signal_count(ctx) < cfg.aggregation_fallback_min_hits


def _aggregation_row_hit(
    row: sqlite3.Row, *, score: float, score_kind: str, chip: str,
    source_occurrences: tuple[SourceOccurrence, ...] = (),
) -> AggregationNodeHit:
    """Build an `AggregationNodeHit` from an aggregation_nodes row, decoding the
    JSON member/session id lists defensively (a malformed list degrades to [])."""
    try:
        member_ids = json.loads(row["member_episode_ids"] or "[]")
    except (ValueError, TypeError):
        member_ids = []
    try:
        session_ids = json.loads(row["session_ids"] or "[]")
    except (ValueError, TypeError):
        session_ids = []
    return AggregationNodeHit(
        node_id=row["id"],
        title=row["title"],
        summary=row["summary"],
        member_episode_ids=member_ids,
        session_ids=session_ids,
        score=float(score),
        score_kind=score_kind,
        why_retrieved=[chip],
        source_occurrences=source_occurrences,
        source_provenance_complete=bool(source_occurrences),
    )


def _query_source_occurrences(
    occurrences: tuple[BoundSourceOccurrence, ...],
) -> tuple[SourceOccurrence, ...]:
    """Translate the neutral durable proof into the public query DTO."""

    return tuple(
        SourceOccurrence(
            item.session_id,
            item.message_id,
            item.source_peer_id,
            item.source_workspace_id,
        )
        for item in occurrences
    )


def _scoped_aggregation_sources(
    conn: sqlite3.Connection,
    node_id: str,
    *,
    source_session_id: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
) -> tuple[SourceOccurrence, ...] | None:
    """Validate one composite before its text enters ranking or model hooks."""

    occurrences = load_aggregation_source_manifest(conn, node_id)
    if occurrences is None:
        return None
    if not all(
        (source_session_id is None or item.session_id == source_session_id)
        and (source_peer_id is None or item.source_peer_id == source_peer_id)
        and (
            source_workspace_id is None
            or item.source_workspace_id == source_workspace_id
        )
        for item in occurrences
    ):
        return None
    return _query_source_occurrences(occurrences)


def _aggregation_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 3,
    embedding_client: EmbeddingClient | None = None,
    max_scan: int = 5000,
    query_vector: object = _QUERY_VECTOR_UNSET,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[AggregationNodeHit]:
    """Phase-2 RAPTOR node retrieval. FTS over node title+summary, plus (when an
    embedder is present) a Python-cosine scan over `aggregation_node_embeddings`
    — the node count is small and the tier is off by default, so no vec0 table is
    maintained. The two ranked lists are RRF-fused, mirroring `_episode_search`.
    Returns [] cleanly when no node table/rows exist (un-dreamed clients).
    Only level-0 nodes are candidates: the v17 rollup/digest levels are
    host-facing standing context (`HyMem.digest()`), not retrieval competitors."""
    if top_k <= 0:
        return []
    scoped = any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    )
    candidate_k = max(1, top_k * 2)
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    fts_hits: list[AggregationNodeHit] = []
    if cleaned:
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            chip = f'agg_fts("{" ".join(tokens)}")'
            batch_size = max(32, candidate_k) if scoped else candidate_k
            raw_scan_cap = max(1024, candidate_k * 64)
            offset = scanned = 0
            while len(fts_hits) < candidate_k and scanned < raw_scan_cap:
                try:
                    rows = conn.execute(
                        """SELECT n.id, n.title, n.summary,
                                  n.member_episode_ids, n.session_ids,
                                  bm25(aggregation_nodes_fts) AS score
                           FROM aggregation_nodes_fts
                           JOIN aggregation_nodes n
                             ON n.rowid = aggregation_nodes_fts.rowid
                           WHERE aggregation_nodes_fts MATCH ? AND n.level = 0
                           ORDER BY score, n.id
                           LIMIT ? OFFSET ?""",
                        (
                            fts_query,
                            min(batch_size, raw_scan_cap - scanned),
                            offset,
                        ),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                if not rows:
                    break
                for row in rows:
                    sources: tuple[SourceOccurrence, ...] = ()
                    if scoped:
                        validated = _scoped_aggregation_sources(
                            conn,
                            row["id"],
                            source_session_id=source_session_id,
                            source_peer_id=source_peer_id,
                            source_workspace_id=source_workspace_id,
                        )
                        if validated is None:
                            continue
                        sources = validated
                    fts_hits.append(_aggregation_row_hit(
                        row,
                        score=row["score"],
                        score_kind="bm25",
                        chip=chip,
                        source_occurrences=sources,
                    ))
                    if len(fts_hits) >= candidate_k:
                        break
                offset += len(rows)
                scanned += len(rows)
                if len(rows) < batch_size:
                    break
            if scoped and len(fts_hits) < candidate_k and scanned >= raw_scan_cap:
                log.warning(
                    "augment.scoped_aggregation_fts_validation_scan_exhausted "
                    "accepted=%d top_k=%d scanned=%d",
                    len(fts_hits), candidate_k, scanned,
                )

    vec_hits: list[AggregationNodeHit] = []
    if embedding_client is not None:
        rows: list[sqlite3.Row] = []
        proof_by_id: dict[str, tuple[SourceOccurrence, ...]] = {}
        try:
            model = embedding_client.model
            dim = embedding_client.dim
            scan_limit = max(0, int(max_scan))
            if not scoped:
                rows = conn.execute(
                    """
                    SELECT n.id, n.title, n.summary, n.member_episode_ids,
                           n.session_ids, ne.vector_json, ne.text_hash
                    FROM aggregation_node_embeddings ne
                    JOIN aggregation_nodes n ON n.id = ne.node_id
                    WHERE n.level = 0 AND ne.model = ? AND ne.dim = ?
                    ORDER BY n.created_at DESC, n.id
                    LIMIT ?
                    """,
                    (model, dim, scan_limit),
                ).fetchall()
            elif scan_limit > 0:
                # Count only proof-valid, in-scope rows against max_scan.  A
                # newer corrupt/mixed-owner prefix cannot starve an older safe
                # node, and candidate text reaches no provider quality hook
                # until its exact manifest passes.
                page_size = max(32, min(256, scan_limit * 2))
                raw_scan_cap = max(1024, scan_limit * 64)
                offset = scanned = 0
                while len(rows) < scan_limit and scanned < raw_scan_cap:
                    batch = conn.execute(
                        """
                        SELECT n.id, n.title, n.summary, n.member_episode_ids,
                               n.session_ids, ne.vector_json, ne.text_hash
                        FROM aggregation_node_embeddings ne
                        JOIN aggregation_nodes n ON n.id = ne.node_id
                        WHERE n.level = 0 AND ne.model = ? AND ne.dim = ?
                        ORDER BY n.created_at DESC, n.id
                        LIMIT ? OFFSET ?
                        """,
                        (
                            model,
                            dim,
                            min(page_size, raw_scan_cap - scanned),
                            offset,
                        ),
                    ).fetchall()
                    if not batch:
                        break
                    for row in batch:
                        sources = _scoped_aggregation_sources(
                            conn,
                            row["id"],
                            source_session_id=source_session_id,
                            source_peer_id=source_peer_id,
                            source_workspace_id=source_workspace_id,
                        )
                        if sources is None:
                            continue
                        proof_by_id[str(row["id"])] = sources
                        rows.append(row)
                        if len(rows) >= scan_limit:
                            break
                    offset += len(batch)
                    scanned += len(batch)
                    if len(batch) < page_size:
                        break
                if len(rows) < scan_limit and scanned >= raw_scan_cap:
                    log.warning(
                        "augment.scoped_aggregation_vector_validation_scan_exhausted "
                        "accepted=%d max_scan=%d scanned=%d",
                        len(rows), scan_limit, scanned,
                    )
        except (AttributeError, sqlite3.OperationalError, TypeError, ValueError):
            rows = []
        if rows:
            qvec = _resolved_query_vector(
                embedding_client, query, query_vector,
                expected_dim=dim,
            )
            scored: list[tuple[float, sqlite3.Row]] = []
            if qvec is not None:
                for r in rows:
                    candidate_text = f"{r['title']}\n{r['summary']}"
                    if r["text_hash"] != embedding_text_hash(candidate_text):
                        continue
                    if not _quality_allows_candidate(
                        embedding_client, query, candidate_text
                    ):
                        continue
                    similarity = _durable_cosine(qvec, r["vector_json"])
                    if similarity is not None:
                        scored.append((similarity, r))
            scored.sort(key=lambda item: (-item[0], str(item[1]["id"])))
            vec_hits = [
                _aggregation_row_hit(
                    r,
                    score=sim,
                    score_kind="vec",
                    chip=f"agg_vec(sim={sim:.3f})",
                    source_occurrences=proof_by_id.get(str(r["id"]), ()),
                )
                for sim, r in scored[:candidate_k]
            ]

    if not vec_hits:
        return fts_hits[:top_k]
    if not fts_hits:
        return vec_hits[:top_k]
    return _rrf_merge_aggregation(fts_hits, vec_hits, top_k=top_k)


def _rrf_merge_aggregation(
    fts: list[AggregationNodeHit],
    vec: list[AggregationNodeHit],
    *,
    top_k: int,
    k: int = 60,
) -> list[AggregationNodeHit]:
    """RRF over two ranked aggregation-node lists. Mirrors `_rrf_merge_episodes`,
    keyed on node_id."""
    by_id: dict[str, AggregationNodeHit] = {}
    scores: dict[str, float] = {}
    for rank, hit in enumerate(fts, start=1):
        scores[hit.node_id] = scores.get(hit.node_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.node_id, hit)
    for rank, hit in enumerate(vec, start=1):
        scores[hit.node_id] = scores.get(hit.node_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(hit.node_id, hit)
    fts_ids = {h.node_id for h in fts}
    vec_ids = {h.node_id for h in vec}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    out: list[AggregationNodeHit] = []
    for nid, score in ordered[:top_k]:
        if nid in fts_ids and nid in vec_ids:
            sources = "fts+vec"
        elif nid in fts_ids:
            sources = "fts"
        else:
            sources = "vec"
        out.append(
            replace(
                by_id[nid], score=score, score_kind="rrf",
                why_retrieved=[
                    *by_id[nid].why_retrieved, f"agg_rrf({sources}, {score:.4f})"
                ],
            )
        )
    return out


def _procedure_search(conn: sqlite3.Connection, query: str, top_k: int = 3) -> list[ProcedureHit]:
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    if not cleaned:
        return []
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

    try:
        rows = conn.execute(
            """SELECT p.id, p.session_id, p.name, p.description, p.steps,
                      bm25(procedures_fts) AS score
               FROM procedures_fts
               JOIN procedures p ON p.rowid = procedures_fts.rowid
               WHERE procedures_fts MATCH ?
                 AND p.status = 'active'
               ORDER BY score
               LIMIT ?""",
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    chip = f'procedure_fts("{" ".join(tokens)}")'
    result: list[ProcedureHit] = []
    for r in rows:
        try:
            steps = json.loads(r["steps"]) if r["steps"] else []
        except json.JSONDecodeError:
            steps = []
        result.append(ProcedureHit(
            procedure_id=r["id"],
            session_id=r["session_id"],
            name=r["name"],
            description=r["description"] or "",
            steps=steps,
            score=float(r["score"]),
            why_retrieved=[chip],
        ))
    return result


def build_token_overlap_index(
    conn: sqlite3.Connection,
    *,
    write_conn: sqlite3.Connection | None = None,
) -> dict[str, list[str]]:
    """Map every underscore-segment token to the active canonicals containing
    it. Caller-cacheable; rebuild after a dream cycle that may have added,
    retracted, or merged edges.

    On a warm database the persistent ``token_overlap_index`` table is read
    directly (O(index rows) instead of O(active edges)). When the table is
    empty — cold start, post-migration, or after runner invalidated it —
    the function falls back to the full canonical scan and, if *write_conn* is
    provided, persists the result so the next cold start is fast.

    Public (no leading underscore) so callers — HyMem instances, background
    workers — can build, stash, and pass it back through `augment()` to avoid
    re-scanning the canonical set on every query. At a few hundred canonicals
    the scan is sub-millisecond; at tens of thousands it begins to matter.
    """
    persisted = conn.execute(
        f"""
        SELECT token_index.token, token_index.canonical
        FROM token_overlap_index token_index
        WHERE EXISTS (
            SELECT 1 FROM knowledge_graph kg
            WHERE {live_edge_predicate('kg')}
              AND (kg.subject_canonical = token_index.canonical
                   OR kg.object_canonical = token_index.canonical)
        )
        """
    ).fetchall()
    if persisted:
        by_token: dict[str, list[str]] = {}
        for r in persisted:
            by_token.setdefault(r["token"], []).append(r["canonical"])
        return by_token

    current = live_edge_predicate()
    rows = conn.execute(
        f"SELECT DISTINCT subject_canonical AS c FROM knowledge_graph WHERE {current} "
        "UNION "
        f"SELECT DISTINCT object_canonical FROM knowledge_graph WHERE {current}"
    ).fetchall()
    by_token = {}
    for r in rows:
        c = r["c"]
        for tok in c.split("_"):
            if tok:
                by_token.setdefault(tok, []).append(c)

    if write_conn is not None and by_token:
        # isolation_level=None means autocommit — wrap explicitly so a partial
        # write doesn't leave a half-populated table that subsequent cold starts
        # would trust as complete.
        write_conn.execute("BEGIN IMMEDIATE")
        try:
            write_conn.executemany(
                "INSERT OR IGNORE INTO token_overlap_index(token, canonical) VALUES (?, ?)",
                [(tok, c) for tok, canons in by_token.items() for c in canons],
            )
            write_conn.execute("COMMIT")
        except Exception:
            write_conn.execute("ROLLBACK")
            raise

    return by_token


def _expand_entities_by_token_overlap(
    conn: sqlite3.Connection,
    entities: list[str],
    *,
    max_per_entity: int = 5,
    common_token_threshold: int = 20,
    token_index: dict[str, list[str]] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Find other canonicals sharing a rare token segment with matched entities.

    A matched canonical like `atta_van_westreenen` has segments `atta`, `van`,
    `westreenen`. For each segment, look up other canonicals containing that
    token as a complete underscore-delimited segment — so `atta_projects`
    surfaces via the `atta` token while keeping single-prefix collisions like
    `attach_handler` out (different segment). Common tokens appearing in more
    than `common_token_threshold` canonicals (`system`, `service`, `data`) are
    dropped as noise. Returns the new entities to consider for Source 1, plus
    an `{entity: token}` map for the `overlap_via:` reason code.

    Pass `token_index` to reuse a prebuilt token-segment index (see
    `build_token_overlap_index`) so repeated `augment()` calls do not re-scan
    the canonical set. Falls back to an on-the-fly build when None.

    Returns ([], {}) when no input is multi-segment or no co-occurrence is
    found in the active graph. Single-token canonicals never trigger expansion
    — there is nothing to overlap on.
    """
    if not entities:
        return [], {}

    multi_token = [e for e in entities if "_" in e]
    if not multi_token:
        return [], {}

    by_token = token_index if token_index is not None else build_token_overlap_index(conn)

    matched_set = set(entities)
    seen: set[str] = set(matched_set)
    expansions: list[str] = []
    overlap_info: dict[str, str] = {}

    for entity in multi_token:
        added = 0
        for tok in entity.split("_"):
            if not tok or added >= max_per_entity:
                continue
            holders = by_token.get(tok, [])
            if len(holders) > common_token_threshold:
                continue  # common-token noise (`system`, `data`, …)
            for c in holders:
                if c in seen:
                    continue
                seen.add(c)
                expansions.append(c)
                overlap_info[c] = tok
                added += 1
                if added >= max_per_entity:
                    break
    return expansions, overlap_info


# Maps category-style query phrases to the entity-type labels emitted by the
# extraction prompt. Lets "what build tools do we use?" pull every canonical
# tagged `package_manager` even when no specific package manager is named.
# Phrases match against a normalised, lowercased copy of the user message.
_TYPE_QUERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "package_manager": (
        "package manager", "package management", "build tool", "build tools",
        "dependency manager", "dependency management",
    ),
    "database": ("database", "databases", "datastore", "data store"),
    "language": ("programming language", "languages", "language stack"),
    "framework": ("framework", "frameworks", "web framework"),
    "service": ("service", "services", "microservice", "microservices"),
    "container": ("container", "containers", "containerization", "containerisation"),
    "platform": ("platform", "platforms", "cloud platform"),
    "testing_framework": ("test framework", "testing framework", "test runner", "test tooling"),
    "ci_tool": ("ci tool", "ci tools", "ci/cd", "ci pipeline", "continuous integration"),
    "monitoring_tool": ("monitoring", "observability", "metrics tool"),
    "config_file": ("config file", "configuration file", "config files"),
    "identity_provider": ("identity provider", "auth provider", "sso", "single sign-on"),
    "message_broker": ("message broker", "message queue", "queue system", "pubsub", "pub/sub"),
    "protocol": ("protocol", "protocols"),
    "environment": ("environment", "environments", "deployment environment"),
    "api": ("api", "apis"),
    "library": ("library", "libraries", "dependency", "dependencies"),
    "tool": ("tooling", "dev tool", "dev tools"),
}

# Maps category-style query phrases to entity_properties (key, value) filters.
# Lets a question about "build tools" also surface entities the LLM tagged
# with `category=build_tool` even if they have no entity_types row.
_PROPERTY_QUERY_KEYWORDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("category", "build_tool"): ("build tool", "build tools"),
    ("category", "database"): ("database", "databases"),
    ("category", "testing"): ("test framework", "testing framework", "test runner"),
    ("category", "deployment"): ("deployment", "deploy tool"),
    ("category", "observability"): ("monitoring", "observability"),
}


def detect_query_types(user_message: str) -> set[str]:
    """Return the in-vocab entity-type labels a category-style query names.

    Single source of truth for "which `entity_types` label does this query target"
    — it scans the lowercased message against the SAME `_TYPE_QUERY_KEYWORDS`
    phrase map that `_expand_entities_from_query` uses for graph_facts expansion,
    so the type vocabulary never forks between the retrieval and counting paths.
    The graph-count router (`hymem.query.count_routing`) consumes this to decide a
    target subject_type/object_type; `_expand_entities_from_query` reuses it below
    so the detection logic lives in exactly one place."""
    msg = user_message.lower()
    return {
        type_label
        for type_label, phrases in _TYPE_QUERY_KEYWORDS.items()
        if any(p in msg for p in phrases)
    }


def _expand_entities_from_query(
    conn: sqlite3.Connection,
    user_message: str,
    *,
    max_per_type: int = 10,
) -> tuple[list[str], dict[str, str]]:
    """Pull canonicals whose type or property matches a category-style query.

    Scans the lowercased message for the configured keyword phrases. For each
    hit, returns up to ``max_per_type`` canonicals tagged with that type (via
    ``entity_types``) or carrying the matching ``(key, value)`` pair in
    ``entity_properties``. Returns ``(canonicals, {canonical: type_label})``
    where the label is the type or ``"key=value"`` rendering of the property.
    """
    msg = user_message.lower()
    matched_types: set[str] = detect_query_types(user_message)

    matched_props: list[tuple[str, str]] = []
    for (key, value), phrases in _PROPERTY_QUERY_KEYWORDS.items():
        if any(p in msg for p in phrases):
            matched_props.append((key, value))

    if not matched_types and not matched_props:
        return [], {}

    seen: set[str] = set()
    out: list[str] = []
    info: dict[str, str] = {}

    for type_label in sorted(matched_types):
        rows = conn.execute(
            "SELECT entity_canonical FROM entity_types WHERE type = ? LIMIT ?",
            (type_label, max_per_type),
        ).fetchall()
        for r in rows:
            ent = r["entity_canonical"]
            if ent in seen:
                continue
            seen.add(ent)
            out.append(ent)
            info[ent] = type_label

    for key, value in matched_props:
        rows = conn.execute(
            "SELECT entity_canonical FROM entity_properties WHERE key = ? AND value = ? LIMIT ?",
            (key, value, max_per_type),
        ).fetchall()
        for r in rows:
            ent = r["entity_canonical"]
            if ent in seen:
                continue
            seen.add(ent)
            out.append(ent)
            info[ent] = f"{key}={value}"

    return out, info


def _expand_entities_by_type(
    conn: sqlite3.Connection,
    entities: list[str],
    max_expanded: int = 10,
) -> tuple[list[str], dict[str, str]]:
    """For matched entities, find other entities of the same type.

    Returns (all_entities, expansion_info) where expansion_info maps each
    expanded entity to the type label that surfaced it (used for the
    `entity_type:` reason code).
    """
    if not entities:
        return entities, {}

    placeholders = ",".join("?" * len(entities))
    type_rows = conn.execute(
        f"SELECT DISTINCT type FROM entity_types WHERE entity_canonical IN ({placeholders})",
        entities,
    ).fetchall()
    if not type_rows:
        return entities, {}

    types = [r[0] for r in type_rows]
    type_placeholders = ",".join("?" * len(types))

    expanded_rows = conn.execute(
        f"""SELECT DISTINCT entity_canonical, type FROM entity_types
            WHERE type IN ({type_placeholders})
            AND entity_canonical NOT IN ({placeholders})
            LIMIT ?""",
        types + entities + [max_expanded],
    ).fetchall()

    expansion_info: dict[str, str] = {}
    expanded: list[str] = []
    for r in expanded_rows:
        ent = r["entity_canonical"]
        if ent not in expansion_info:
            expansion_info[ent] = r["type"]
            expanded.append(ent)

    return entities + expanded, expansion_info
