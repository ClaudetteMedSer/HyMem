from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field, replace

from hymem.config import HyMemConfig
from hymem.core.vectors import decode_vector
from hymem.dreaming.aggregate import Digest, load_digest
from hymem.dreaming.user_profile import ProfileEntry, load_profile
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.query.coref import QueryRewrite, rewrite_query
from hymem.query.entities import GraphCount, count_relations, match_known_entities
from hymem.query.intent import detect_ability_signal
from hymem.query.predicate_routing import route_predicates
from hymem.query.rerank import rerank as run_rerank
from hymem.rules import Rule, load_rules
from hymem.session import Message, recent_messages

log = logging.getLogger("hymem.query.augment")


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


@dataclass
class MessageHit:
    """A raw session-log turn surfaced by direct FTS5 keyword search over the
    `messages` table — the path that reaches content not (yet) consolidated into
    chunks by dreaming, including turns from other sessions.

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
    dated knowledge-graph edge — its `temporal_scope`, else its `first_seen`), or
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

    `message_hits` is the raw-message keyword tier: BM25 hits from a direct FTS5
    search over the `messages` table (user/assistant turns). Unlike
    `recent_turns` it is query-relevant and spans *all* sessions, and unlike
    `fts_hits` it reaches turns that dreaming never chunked. This closes the
    "search raw messages by keyword" gap that chunk-only FTS leaves.
    """

    user_md: str = ""
    memory_md: str = ""
    fts_hits: list[FtsHit] = field(default_factory=list)
    message_hits: list[MessageHit] = field(default_factory=list)
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
    raw messages (`temporal_mentions`) with dated knowledge-graph edges (their
    `temporal_scope`, else `first_seen`), so a temporal-reasoning question ("how
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
    ability: str | None = None,
) -> AugmentedContext:
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
        ctx.coref = rewrite_query(
            user_message, coref_turns, cfg=cfg, conn=conn, llm=llm
        )
        if ctx.coref.changed:
            query = ctx.coref.rewritten
            log.debug("coref.applied rule=%s query=%r", ctx.coref.rule, query)

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
    fts = _fts_search(conn, query, top_k=candidate_k)
    vec: list[FtsHit] = []
    if embedding_client is not None:
        vec = _vector_search(
            conn,
            embedding_client,
            query,
            top_k=candidate_k,
            max_scan=cfg.embedding_max_scan,
        )
        ctx.fts_hits = _rrf_merge(fts, vec, top_k=candidate_k)
    else:
        ctx.fts_hits = fts

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

    # Raw-message keyword tier: direct FTS5 over the session log, reaching turns
    # that were never chunked (low salience, not-yet-dreamed, or other sessions).
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
        msg_hits = _message_fts_search(conn, query, top_k=msg_candidate_k)
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
            ctx.message_hits,
            ctx.total_message_matches,
            ctx.enumeration_turns,
        ) = _message_fts_aggregate(
            conn, query, cap=cfg.message_fts_aggregate_cap
        )
        ctx.graph_count = _maybe_graph_count(conn, query)
    else:
        if cfg.message_fts_top_k > 0:
            ctx.message_hits = _relevance_message_hits()
        if mr_aggregate:
            # Count layered on top of relevance retrieval (message_hits already
            # set above). The graph gives an EXACT typed count for the in-domain
            # slice; any failure leaves graph_count=None and the keyword candidate
            # count stands. The aggregate's own evidence turns are discarded — the
            # reranked relevance turns are the better evidence view.
            (
                _,
                ctx.total_message_matches,
                ctx.enumeration_turns,
            ) = _message_fts_aggregate(
                conn, query, cap=cfg.message_fts_aggregate_cap
            )
            ctx.graph_count = _maybe_graph_count(conn, query)

    # ability="TR" (temporal reasoning) builds a date-ordered event list so the
    # host LLM reads a chronology instead of finding dates in noise. It merges
    # explicit dates extracted from raw messages (temporal_mentions) with dated
    # knowledge-graph edges, ordered date-ascending. Only populated for TR; the
    # other tiers above (graph/fts/messages) still run unchanged.
    if ability == "TR":
        ctx.temporal_events = _temporal_events(
            conn, query, ctx.message_hits, ctx.fts_hits, top_k=cfg.fts_top_k
        )

    ctx.episodes = _episode_search(
        conn, query,
        top_k=cfg.fts_top_k,
        embedding_client=embedding_client,
    )

    # Phase-2 RAPTOR additive tier: cross-session cluster summaries. Off by
    # default; only runs when the layer is enabled AND the routed ability is in
    # `aggregation_inject_abilities` (default TR-only — the G4 A/B showed broad
    # injection reshuffles ranking against gold message hits everywhere except
    # temporal reasoning). Never displaces the tiers above — it layers a
    # synthesis view on top.
    if cfg.aggregation_nodes_enabled and _aggregation_tier_fires(cfg, ability):
        ctx.aggregation_nodes = _aggregation_search(
            conn, query,
            top_k=cfg.aggregation_top_k,
            embedding_client=embedding_client,
            max_scan=cfg.embedding_max_scan,
        )

    # ability="IF" (instruction/step recall) pulls a wider procedure set, since
    # procedures — ordered step-by-step workflows — are the natural fit for
    # "what steps did I take to implement X?" The host still decides ordering.
    proc_top_k = cfg.procedure_top_k_if if ability == "IF" else cfg.fts_top_k
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
        embedding_client=embedding_client,
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


def _fts_search(conn: sqlite3.Connection, query: str, *, top_k: int) -> list[FtsHit]:
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    if not cleaned:
        return []
    # Build an OR query across tokens so partial matches still surface results.
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

    try:
        # bm25() is an FTS5 built-in; schema.sql declares chunks_fts with fts5.
        # If the table is ever migrated away from FTS5, this query will raise
        # OperationalError and fall through to the empty-results path below.
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.session_id, c.text, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    chip = f'fts_match("{" ".join(tokens)}")'
    return [
        FtsHit(
            chunk_id=r["chunk_id"],
            session_id=r["session_id"],
            text=r["text"],
            score=float(r["score"]),
            why_retrieved=[chip],
        )
        for r in rows
    ]


def _message_fts_search(
    conn: sqlite3.Connection, query: str, *, top_k: int
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

    try:
        rows = conn.execute(
            """
            SELECT m.id, m.session_id, m.role, m.content, m.created_at,
                   bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            WHERE messages_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, top_k),
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
            why_retrieved=[chip],
        )
        for r in rows
    ]


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
    conn: sqlite3.Connection, query: str, *, cap: int
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

    try:
        rows = conn.execute(
            """
            SELECT m.id, m.session_id, m.role, m.content, m.created_at,
                   bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            WHERE messages_fts MATCH ? AND m.role = 'user'
            ORDER BY m.created_at, m.id
            LIMIT ?
            """,
            (fts_query, _MR_COUNT_SCAN),
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

    Revives the long-dead `kg_evidence.temporal_scope`: an edge's most specific
    recorded scope wins, else the edge's `first_seen` provides an event date.
    Only datable relational predicates (`_TR_EDGE_PREDICATES`) are considered, so
    structural edges (part_of, contains) don't clutter the timeline. Returns []
    when no entity matched. Tolerant of a missing `temporal_scope` column."""
    if not entities:
        return []

    pred_placeholders = ",".join("?" * len(_TR_EDGE_PREDICATES))
    ent_placeholders = ",".join("?" * len(entities))
    try:
        rows = conn.execute(
            f"""
            SELECT kg.subject_canonical AS s, kg.predicate AS p,
                   kg.object_canonical AS o, kg.first_seen,
                   (SELECT ev.temporal_scope FROM kg_evidence ev
                    WHERE ev.edge_id = kg.id AND ev.temporal_scope IS NOT NULL
                    ORDER BY ev.extracted_at DESC LIMIT 1) AS scope
            FROM knowledge_graph kg
            WHERE kg.status = 'active'
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
        # Prefer an extracted temporal_scope that *looks* like an ISO date so the
        # merged list stays sortable; otherwise fall back to first_seen. A free-
        # text scope ("last quarter") is kept as the event text but not as the
        # sort key, since it can't be ordered against ISO dates.
        scope = (r["scope"] or "").strip()
        scope_date = scope[:10] if _looks_iso(scope) else None
        date = scope_date or _temporal_event_date(None, r["first_seen"] or "")
        if date is None:
            continue
        fact = f"{r['s']} {r['p']} {r['o']}"
        text = f"{fact} ({scope})" if scope and not scope_date else fact
        chip = "temporal_scope" if scope_date else "edge_first_seen"
        events.append((
            date,
            TemporalEvent(
                date=date, text=text, source="graph", why_retrieved=[chip]
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
) -> list[FtsHit]:
    from hymem.core import db as core_db

    if not _embeddings_compatible(conn, embedder):
        return []
    if core_db.has_vec_table(conn):
        return _vec_search(conn, embedder, query, top_k=top_k)
    return _python_cosine_search(conn, embedder, query, top_k=top_k, max_scan=max_scan)


def _vec_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
) -> list[FtsHit]:
    from hymem.core import db as core_db

    qvec = embedder.embed([query])[0]
    hits = core_db.vec_search(conn, qvec, top_k)

    result: list[FtsHit] = []
    for chunk_rowid, distance in hits:
        row = conn.execute(
            "SELECT id AS chunk_id, session_id, text FROM chunks WHERE rowid = ?",
            (chunk_rowid,),
        ).fetchone()
        if row:
            sim = 1.0 / (1.0 + distance)
            result.append(
                FtsHit(
                    chunk_id=row["chunk_id"],
                    session_id=row["session_id"],
                    text=row["text"],
                    score=float(sim),
                    score_kind="vec",
                    why_retrieved=[f"vec_topk(sim={sim:.3f})"],
                )
            )
    return result


def _python_cosine_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
) -> list[FtsHit]:
    rows = conn.execute(
        """
        SELECT c.id AS chunk_id, c.session_id, c.text, e.vector_json
        FROM chunk_embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        (max_scan,),
    ).fetchall()
    if not rows:
        return []

    qvec = embedder.embed([query])[0]
    qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0

    scored: list[tuple[float, sqlite3.Row]] = []
    for r in rows:
        vec = decode_vector(r["vector_json"])
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
        sim = dot / (qnorm * vnorm)
        scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)

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

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
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
           (julianday('now') - julianday(last_seen)) AS days_since
    FROM knowledge_graph
"""


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
) -> list[GraphFact]:
    """Hybrid edge ranker: gathers candidates from entity matches, semantic KNN,
    and predicate routing, then scores by semantic × confidence × recency × boost.

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
    fallback = not routed
    overlap_info = overlap_info or {}
    candidates: dict[tuple[str, str, str], dict] = {}

    def _ensure(row: sqlite3.Row) -> dict:
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
            }
            candidates[key] = c
        return c

    # Source 1 — entity-anchored (always).
    for entity in entities:
        rows = conn.execute(
            _EDGE_SELECT
            + """
            WHERE status = 'active'
              AND (subject_canonical = ? OR object_canonical = ?)
            ORDER BY (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) DESC,
                     last_reinforced DESC
            LIMIT ?
            """,
            (entity, entity, cfg.graph_top_k_per_entity),
        ).fetchall()
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
            conn, cfg, embedding_client, query
        ):
            row = conn.execute(
                _EDGE_SELECT + " WHERE id = ? AND status = 'active'",
                (edge_id,),
            ).fetchone()
            if row is None:
                continue
            c = _ensure(row)
            c["semantic_score"] = max(c["semantic_score"], semantic_score)
            c["semantic_retrieved"] = True

    # Source 3 — predicate-routed.
    if routed:
        pred_placeholders = ",".join("?" * len(routed))
        rows = conn.execute(
            _EDGE_SELECT
            + f"""
            WHERE status = 'active' AND predicate IN ({pred_placeholders})
            ORDER BY (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) DESC,
                     last_seen DESC
            LIMIT ?
            """,
            list(routed) + [cfg.graph_predicate_top_k],
        ).fetchall()
        for r in rows:
            _ensure(r)

    # Source 4 — multi-hop expansion from DIRECTLY-anchored entities only.
    # Chaining a fuzzy (token-overlap) link N times produces garbage, so seeds
    # exclude overlap-only anchors. Additive: dedups against Sources 1/3 on
    # (s, p, o) via `_ensure`, so a bridged edge that is also a direct/routed hit
    # keeps its stronger native score and multi-hop never double-counts.
    if cfg.graph_multihop_enabled:
        direct_seeds = [e for e in entities if e not in overlap_info]
        for d in _multihop_edges(conn, cfg, direct_seeds).values():
            c = _ensure(d["row"])
            c["multihop_score"] = max(c["multihop_score"], d["path_score"])
            c["hop"] = d["hop"]

    # Recency-only seeding: if the fallback path has no candidates at all
    # (no entity match, no semantic hit), pull a small set of recent active
    # edges so the graph_facts list isn't empty when something could be shown.
    if fallback and not candidates:
        for row in _recency_edges(conn, cfg.graph_top_k):
            _ensure(row)

    results: list[GraphFact] = []
    for c in candidates.values():
        confidence = (c["pos"] + 1.0) / (c["pos"] + c["neg"] + 2.0)
        recency_weight = math.exp(-c["days_since"] / cfg.graph_recency_half_life_days)
        semantic_score = c["semantic_score"]
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
            score = c["multihop_score"] * recency_weight
            why.append(f"fallback:multihop:{c['hop']}hop")
        elif fallback:
            if c["entity_match"]:
                overlap_only = not c["direct_anchor"]
                if overlap_only:
                    score = (
                        cfg.graph_token_overlap_weight
                        * confidence
                        * recency_weight
                    )
                    why.append("fallback:entity_anchored:overlap")
                    for tok in sorted(c["overlap_tokens"]):
                        why.append(f"overlap_via:{tok}")
                else:
                    score = confidence * recency_weight
                    why.append("fallback:entity_anchored")
            elif semantic_score > 0:
                score = semantic_score * confidence * recency_weight
                why.append("fallback:semantic")
            else:
                score = confidence * recency_weight
                why.append("fallback:recency")
        else:
            predicate_boost = cfg.graph_predicate_boost if in_routed else 1.0
            score = (
                confidence
                * recency_weight
                * (semantic_score if semantic_score > 0 else 1.0)
                * predicate_boost
            )

        if c["semantic_retrieved"]:
            why.append(f"semantic_{max(0.0, semantic_score):.2f}")
        if in_routed:
            why.append(f"predicate:{c['p']}")
        for entity_type in sorted(c["entity_types"]):
            why.append(f"entity_type:{entity_type}")
        if c["days_since"] <= cfg.graph_recency_recent_days:
            why.append(f"recency_{round(c['days_since'])}d")
        if c["entity_match"]:
            why.append("entity_match")

        total_evidence = c["pos"] + c["neg"]
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
                pos_evidence=c["pos"],
                neg_evidence=c["neg"],
                derived=c["derived"],
                why_retrieved=why,
                score=score,
                hedge_recommended=hedge,
            )
        )

    results.sort(key=lambda f: f.score, reverse=True)
    return results[: cfg.graph_top_k]


# Hard safety bound on BFS frontier width per hop. Not a tuning knob — a hub
# node (e.g. `uv`) could otherwise explode the frontier and blow the query
# latency budget; `graph_multihop_min_score` is the primary bound, this is the
# backstop. Keep only the highest-scoring frontier nodes into the next hop.
_MULTIHOP_FRONTIER_CAP = 256


def _multihop_edges(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    seeds: list[str],
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
    if cfg.graph_multihop_max_hops < 2 or not seeds:
        return {}

    seeds_set = set(seeds)
    reached: dict[str, float] = {s: 1.0 for s in seeds}  # node -> best path score
    out: dict[tuple[str, str, str], dict] = {}
    frontier = list(seeds)

    for hop in range(1, cfg.graph_multihop_max_hops + 1):
        if not frontier:
            break
        ph = ",".join("?" * len(frontier))
        rows = conn.execute(
            _EDGE_SELECT
            + f"""
            WHERE status = 'active'
              AND (subject_canonical IN ({ph}) OR object_canonical IN ({ph}))
            """,
            frontier + frontier,
        ).fetchall()

        next_scores: dict[str, float] = {}
        for r in rows:
            conf = (r["pos"] + 1.0) / (r["pos"] + r["neg"] + 2.0)
            # Never emit a seed-incident edge: it is 1-hop from a seed and Source 1
            # already has it (emitting would double-count / mislabel it as a bridge).
            seed_incident = r["s"] in seeds_set or r["o"] in seeds_set
            for near, far in ((r["s"], r["o"]), (r["o"], r["s"])):
                if near not in reached:
                    continue
                path_score = reached[near] * conf * cfg.graph_multihop_decay
                if path_score < cfg.graph_multihop_min_score:
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
        candidates = sorted(next_scores, key=next_scores.get, reverse=True)
        if cfg.graph_multihop_hub_degree_max > 0 and candidates:
            probe = candidates[: _MULTIHOP_FRONTIER_CAP * 2]
            degrees = _active_degrees(conn, probe)
            candidates = [
                n
                for n in probe
                if degrees.get(n, 0) <= cfg.graph_multihop_hub_degree_max
            ]
        frontier = candidates[:_MULTIHOP_FRONTIER_CAP]
    return out


def _active_degrees(
    conn: sqlite3.Connection, nodes: list[str]
) -> dict[str, int]:
    """Active degree (count of active edges where the node is subject OR object)
    for each node, in a single query — the hub guard's fan-out test in
    `_multihop_edges`. A node absent from the result has degree 0. (A self-loop
    edge, subject==object, is counted twice; those are effectively absent in this
    graph, so degree == incident-edge count in practice.)"""
    if not nodes:
        return {}
    ph = ",".join("?" * len(nodes))
    rows = conn.execute(
        f"""
        SELECT node, COUNT(*) AS deg FROM (
            SELECT subject_canonical AS node FROM knowledge_graph
             WHERE status = 'active' AND subject_canonical IN ({ph})
            UNION ALL
            SELECT object_canonical AS node FROM knowledge_graph
             WHERE status = 'active' AND object_canonical IN ({ph})
        ) GROUP BY node
        """,
        nodes + nodes,
    ).fetchall()
    return {r["node"]: r["deg"] for r in rows}


def _recency_edges(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Pull the most recent active edges by confidence × recency.

    Used by the no-predicate fallback when neither entity match nor semantic
    KNN produced any candidates, so something graph-shaped is still returned.
    """
    return conn.execute(
        _EDGE_SELECT
        + """
        WHERE status = 'active'
        ORDER BY (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) DESC,
                 last_seen DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _semantic_edge_hits(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    embedder: EmbeddingClient,
    query: str,
) -> list[tuple[int, float]]:
    """Return (edge_id, semantic_score) pairs for edges similar to the query."""
    from hymem.core import db as core_db

    if core_db._load_vec_extension(conn) and core_db.has_vec_table(
        conn, table="vec_edges"
    ):
        qvec = embedder.embed([query])[0]
        hits = core_db.vec_search(
            conn, qvec, cfg.graph_semantic_top_k, table="vec_edges"
        )
        return [(edge_id, 1.0 / (1.0 + distance)) for edge_id, distance in hits]
    return _python_cosine_edge_search(
        conn, embedder, query,
        top_k=cfg.graph_semantic_top_k, max_scan=cfg.embedding_max_scan,
    )


def _python_cosine_edge_search(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    query: str,
    *,
    top_k: int,
    max_scan: int,
) -> list[tuple[int, float]]:
    rows = conn.execute(
        """
        SELECT kg.id AS edge_id, e.vector_json
        FROM knowledge_graph kg
        JOIN edge_embeddings e
          ON e.edge_text = kg.subject_canonical || ' ' || kg.predicate || ' '
                           || kg.object_canonical
        WHERE kg.status = 'active'
        ORDER BY kg.last_seen DESC
        LIMIT ?
        """,
        (max_scan,),
    ).fetchall()
    if not rows:
        return []

    qvec = embedder.embed([query])[0]
    qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0

    scored: list[tuple[float, int]] = []
    for r in rows:
        vec = decode_vector(r["vector_json"])
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
        sim = dot / (qnorm * vnorm)
        scored.append((sim, r["edge_id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(edge_id, sim) for sim, edge_id in scored[:top_k]]


def _episode_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 3,
    embedding_client: EmbeddingClient | None = None,
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
    fts_hits: list[EpisodeHit] = []
    if cleaned:
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            episode_chip = f'episode_fts("{" ".join(tokens)}")'
            try:
                rows = conn.execute(
                    """SELECT e.id, e.session_id, e.title, e.summary, bm25(episodes_fts) AS score
                       FROM episodes_fts
                       JOIN episodes e ON e.rowid = episodes_fts.rowid
                       WHERE episodes_fts MATCH ?
                       ORDER BY score
                       LIMIT ?""",
                    (fts_query, top_k * 2),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for r in rows:
                fts_hits.append(
                    EpisodeHit(
                        episode_id=r["id"],
                        session_id=r["session_id"],
                        title=r["title"],
                        summary=r["summary"][:300],
                        score=float(r["score"]),
                        score_kind="bm25",
                        why_retrieved=[episode_chip],
                    )
                )

    vec_hits: list[EpisodeHit] = []
    if (
        embedding_client is not None
        and core_db._load_vec_extension(conn)
        and core_db.has_vec_table(conn, table="vec_episodes")
    ):
        qvec = embedding_client.embed([query])[0]
        try:
            hit_rows = core_db.vec_search(
                conn, qvec, top_k * 2, table="vec_episodes"
            )
        except Exception:
            hit_rows = []
        for rowid, distance in hit_rows:
            r = conn.execute(
                "SELECT id, session_id, title, summary FROM episodes WHERE rowid = ?",
                (rowid,),
            ).fetchone()
            if r is None:
                continue
            sim = 1.0 / (1.0 + distance)
            vec_hits.append(
                EpisodeHit(
                    episode_id=r["id"],
                    session_id=r["session_id"],
                    title=r["title"],
                    summary=r["summary"][:300],
                    score=float(sim),
                    score_kind="vec",
                    why_retrieved=[f"episode_vec(sim={sim:.3f})"],
                )
            )

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
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
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


def _aggregation_tier_fires(cfg: HyMemConfig, ability: str | None) -> bool:
    """Whether the aggregation tier runs for this query's (normalized) ability.
    An empty `aggregation_inject_abilities` means every query (broad mode, for
    A/B re-runs); otherwise the ability must be in the allowlist — so with the
    default `("TR",)` an unrouted question (ability None) gets no nodes."""
    allowed = cfg.aggregation_inject_abilities
    if not allowed:
        return True
    return ability is not None and ability in {a.upper() for a in allowed}


def _aggregation_row_hit(
    row: sqlite3.Row, *, score: float, score_kind: str, chip: str
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
        summary=row["summary"][:600],
        member_episode_ids=member_ids,
        session_ids=session_ids,
        score=float(score),
        score_kind=score_kind,
        why_retrieved=[chip],
    )


def _aggregation_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 3,
    embedding_client: EmbeddingClient | None = None,
    max_scan: int = 5000,
) -> list[AggregationNodeHit]:
    """Phase-2 RAPTOR node retrieval. FTS over node title+summary, plus (when an
    embedder is present) a Python-cosine scan over `aggregation_node_embeddings`
    — the node count is small and the tier is off by default, so no vec0 table is
    maintained. The two ranked lists are RRF-fused, mirroring `_episode_search`.
    Returns [] cleanly when no node table/rows exist (un-dreamed clients).
    Only level-0 nodes are candidates: the v17 rollup/digest levels are
    host-facing standing context (`HyMem.digest()`), not retrieval competitors."""
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(query)).strip()
    fts_hits: list[AggregationNodeHit] = []
    if cleaned:
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            chip = f'agg_fts("{" ".join(tokens)}")'
            try:
                rows = conn.execute(
                    """SELECT n.id, n.title, n.summary, n.member_episode_ids,
                              n.session_ids, bm25(aggregation_nodes_fts) AS score
                       FROM aggregation_nodes_fts
                       JOIN aggregation_nodes n ON n.rowid = aggregation_nodes_fts.rowid
                       WHERE aggregation_nodes_fts MATCH ? AND n.level = 0
                       ORDER BY score
                       LIMIT ?""",
                    (fts_query, top_k * 2),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            fts_hits = [
                _aggregation_row_hit(r, score=r["score"], score_kind="bm25", chip=chip)
                for r in rows
            ]

    vec_hits: list[AggregationNodeHit] = []
    if embedding_client is not None:
        try:
            rows = conn.execute(
                """
                SELECT n.id, n.title, n.summary, n.member_episode_ids,
                       n.session_ids, ne.vector_json
                FROM aggregation_node_embeddings ne
                JOIN aggregation_nodes n ON n.id = ne.node_id
                WHERE n.level = 0
                ORDER BY n.created_at DESC
                LIMIT ?
                """,
                (max_scan,),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if rows:
            qvec = embedding_client.embed([query])[0]
            qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0
            scored: list[tuple[float, sqlite3.Row]] = []
            for r in rows:
                vec = decode_vector(r["vector_json"])
                if len(vec) != len(qvec):
                    continue
                dot = sum(a * b for a, b in zip(qvec, vec))
                vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
                scored.append((dot / (qnorm * vnorm), r))
            scored.sort(key=lambda x: x[0], reverse=True)
            vec_hits = [
                _aggregation_row_hit(
                    r, score=sim, score_kind="vec", chip=f"agg_vec(sim={sim:.3f})"
                )
                for sim, r in scored[:top_k * 2]
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
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
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
        "SELECT token, canonical FROM token_overlap_index"
    ).fetchall()
    if persisted:
        by_token: dict[str, list[str]] = {}
        for r in persisted:
            by_token.setdefault(r["token"], []).append(r["canonical"])
        return by_token

    rows = conn.execute(
        "SELECT DISTINCT subject_canonical AS c FROM knowledge_graph WHERE status='active' "
        "UNION "
        "SELECT DISTINCT object_canonical FROM knowledge_graph WHERE status='active'"
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


