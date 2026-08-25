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

    message_fts_aggregate_cap: int = 50
    """Counting path for `ability="MR"` ("how many X across all my requests?").
    Set to 0 to disable — `ability="MR"` then uses the normal top-k message path.
    When > 0, augment() returns distinct, deduped **user** turns chronologically
    in `message_hits` (capped at this value) and the candidate count in
    `total_message_matches` (exact even when the evidence is capped). The cap
    bounds only the *evidence* turns the host sees, not the count. Defaults to 50:
    keyword counting fits *lexical* "how many" questions (the common case); the
    host's LLM still verifies the candidate count against the returned turns, so
    a purely semantic "how many different ways…" question degrades to LLM tallying
    rather than returning a wrong number."""

    mr_aggregate_additive: bool = True
    """When True (default), an `ability="MR"` detection LAYERS the candidate count
    (`total_message_matches` + exact in-domain `graph_count`) on top of the normal
    reranked relevance `message_hits`, instead of REPLACING them with the
    aggregate's chronological distinct-user turns. This neutralises the retrieval
    cost of MR over-detection: the router can't tell a real cross-session count
    ("how many times did I deploy") from a single-session lookup phrased as a count
    ("how many books have I read") — the two are textually identical — so a false
    positive used to swap a lookup onto the count-only path and lose relevance
    retrieval. Additive mode means a mis-routed question still gets full retrieval
    while a genuine count keeps its number. Set False for the legacy replace-mode
    (aggregate owns `message_hits`). Mirrors the already-additive TR path. The
    exact count is unchanged either way — only which turns are shown as evidence."""

    procedure_top_k_if: int = 10
    """Procedure budget when augment() is called with `ability="IF"`
    (instruction/step recall — "what steps did I take to implement X?").
    Procedures are the natural fit for IF, so IF-tagged queries pull a wider set
    of `ProcedureHit`s than the default `fts_top_k`. Other abilities keep
    `fts_top_k`."""

    aggregation_nodes_enabled: bool = False
    """Master switch for the Phase-2 RAPTOR cross-session aggregation layer (off
    by default). When True, dreaming clusters episodes across sessions
    (connected components over embedding-OR-entity overlap) and fuses each
    multi-session cluster into one `aggregation_nodes` summary, and augment()
    surfaces matching nodes in `ctx.aggregation_nodes` as an ADDITIVE tier (it
    never replaces episode/chunk/message retrieval). It targets the multi-session
    *synthesis* residual: a question whose answer is spread one-fact-per-session
    can be read off a single cluster summary instead of re-fusing dozens of raw
    turns. Off → zero extra LLM cost at dream time and zero behavior change at
    query time. The front-run co-location probe that gated this build lives in
    `benchmarks/raptor_cluster_probe.py`."""

    aggregation_emb_threshold: float = 0.55
    """Cosine threshold above which two episodes link into the same cluster
    (the embedding arm of the OR). Default from the passing probe grid point
    (emb≥0.55 OR ent≥0.50), which co-located 87% of mappable gold."""

    aggregation_ent_threshold: float = 0.50
    """Jaccard(key_entities) threshold above which two episodes link (the entity
    arm of the OR). Catches the named-thing continuity embeddings miss."""

    aggregation_max_cluster_size: int = 15
    """Hard cap on connected-component size before fusion — the Stage-3a
    chaining guard (0 = uncapped). The OR-link rule chains transitively
    (A~B, B~C puts A with C even when they share nothing), and on the prod
    store the probe (benchmarks/cluster_size_probe.py, 2026-06-12) found ONE
    component of 348 episodes spanning 61 sessions — fusing that yields a mush
    summary. Components larger than the cap are split deterministically into
    recency-ordered windows of at most this many episodes (full windows
    anchored at the OLDEST end so between-dream arrivals only re-fuse the
    newest tail window — see cluster_episodes — and the undersized newest
    window is dropped by the normal min-members/min-sessions policy). 0
    translates to None (uncapped) at the
    `cluster_episodes` call sites — matching the `aggregation_digest_anchor_facts`
    "0 disables" house style."""

    aggregation_blocking_top_k: int = 24
    """Candidate-blocking KNN width for the cosine arm of episode clustering
    (Stage 3b). All-pairs Python cosine is O(n²) per dream — on the prod box
    (2026-06-12) 395 episodes meant 77,815 pair tests at 4.04s, past the 2s
    budget. With blocking, the entity arm tests exactly the pairs sharing >= 1
    key entity (lossless: Jaccard >= 0.5 needs a shared entity) and the cosine
    arm tests each episode against its top-k `vec_episodes` neighbors only.
    0 disables blocking entirely (exact all-pairs — what
    benchmarks/cluster_size_probe.py measures); with k >= n-1 blocking is
    exact, so small stores lose nothing. Stores without sqlite_vec fall back
    to exact all-pairs automatically."""

    aggregation_min_sessions: int = 2
    """Only fuse clusters spanning at least this many DISTINCT sessions. The
    whole point is *cross-session* synthesis; a single-session cluster adds
    nothing the per-session episode/summary doesn't already give, so it is
    skipped (no LLM call, no node)."""

    aggregation_min_members: int = 2
    """Minimum episodes in a cluster before it is fused. Singletons are skipped."""

    aggregation_max_members: int = 12
    """Cap on episodes fed to one aggregation summary call, bounding context and
    cost. Largest clusters are truncated to their first `max_members` episodes in
    message order; the node records the full membership regardless."""

    aggregation_top_k: int = 3
    """Number of aggregation nodes augment() returns in `ctx.aggregation_nodes`
    when the layer is enabled."""

    aggregation_digest_enabled: bool = True
    """Sub-switch (active only when `aggregation_nodes_enabled` is True): after
    the level-0 cluster nodes are built, recursively roll the tree up — cluster
    the level-0 nodes plus the episodes no cluster absorbed, fuse each group
    into a level-N node, repeat — until one ROOT digest node remains: the
    standing "what do you know about me?" summary `HyMem.digest()` returns.
    This is the consumption model the G4 A/Bs argued FOR: the digest is host-
    facing standing context (e.g. system-prompt injection), not a retrieval
    competitor — levels >= 1 never enter the query-time tier, so it cannot
    crowd message hits. Fusion calls are reuse-cached by member-set hash, so a
    dream over a stable store rebuilds the tree without new LLM calls."""

    aggregation_digest_max_leaves: int = 256
    """Cap on pass-through episodes (those outside every kept cluster) admitted
    into the digest tree, keeping the most recent. Bounds first-build LLM cost
    on a large backlog store (~leaves/11 fusion calls worst case); level-0
    nodes are always included."""

    aggregation_digest_anchor_facts: int = 20
    """Max ACTIVE, non-derived knowledge-graph edges injected into the root
    digest fusion as a VERIFIED FACTS ground-truth block (0 disables). The
    summaries the root fuses are machine-generated and can crystallize
    hallucinated identity details in the reuse cache (the "Acme Corp" incident);
    graph edges are extracted directly from conversation evidence, so they give
    the model true identity/preference signals to use instead of a vacuum to
    fill, and an explicit license to drop summary claims that conflict. The
    root node's cache id includes a hash of this block, so the digest
    regenerates whenever the anchor facts change."""

    aggregation_inject_abilities: tuple[str, ...] = ("TR",)
    """Abilities for which the aggregation tier fires at query time (empty tuple
    = every query, the broad mode). The G4 LME A/B (500q, seed 0) showed broad
    injection is net-harmful: nodes recover NO messages the message/chunk tiers
    missed — they only reshuffle ranking, crowding gold turns out of the answer
    pool (KU −9.0pp, SS-P −3.4pp) while helping only temporal reasoning
    (TR +3.0pp, −4 ranking misses). So by default the tier only fires for
    TR-routed questions — the one ability with a verified mechanism — and is a
    no-op elsewhere. Additive-safe under routing errors: a TR false positive
    merely adds a summary tier (never displaces other tiers), a false negative
    is identical to the layer being off."""

    augment_include_digest: bool = False
    """When True, `augment()` also loads the standing root digest into
    `ctx.digest` so a single-call host gets the whole-store summary without a
    second `HyMem.digest()` round-trip. Default False to keep `augment()` lean:
    the digest is *standing* context — it does not change per query, so a host
    that assembles its own system prompt should fetch it once per dream via
    `HyMem.digest()` (or the `hymem_digest` MCP tool) instead of paying the
    load on every turn. Purely additive either way: the digest is never a
    retrieval tier and consumes no other tier's budget (the Stage-5 delivery
    decision in benchmarks/raptor_digest_plan.md)."""

    profile_extraction_enabled: bool = True
    """Master switch for the typed user-profile tier (schema v18, Stage 1 /
    P4). Default ON since profile.v2 PASSED the on-box hand-scored precision
    gate (~95% adjusted, zero v1 bleed-throughs, 2026-06-12 — see
    benchmarks/raptor_digest_plan.md Stage 1; profile.v1 had FAILED it at ~8%,
    which kept this False until the v2 re-gate). Any material prompt change
    re-enters the same gate: bump PROFILE_PROMPT_VERSION, flip this False,
    re-score ≥0.9 on the box, then re-enable. When True, dreaming runs one
    extra LLM call per
    dreamed session over the session's USER turns only, extracting facts into
    the CLOSED slot vocabulary (role, name, employer, location, language,
    relationship(person), possession, age_birthday, health_condition,
    recurring_activity — enforced by validation AND a table CHECK, so the LLM
    can never invent a slot). Rows are bi-temporal (valid_at/invalid_at, the
    v15 knowledge-graph semantics): single-valued slots supersede on conflict,
    relationship supersedes per person, the rest accumulate. Consumed
    ADDITIVELY by three readers — the root digest's VERIFIED FACTS anchor
    (profile rows above graph edges), augment()'s `ctx.user_profile` tier
    (never displaces other tiers), and `HyMem.profile()`. The call has its own
    per-session skip-guard (sessions.profile_prompt_version, schema v19), so
    re-dreaming unchanged sessions still costs zero tail calls while a
    PROFILE_PROMPT_VERSION bump alone re-extracts. False → no extraction call
    and an always-empty augment tier."""

    profile_max_items_per_session: int = 16
    """Cap on validated profile items accepted from one per-session extraction
    call. A runaway response (the LLM tagging every sentence as a fact) is
    truncated here before any row is written; 16 comfortably covers a genuine
    identity-dense session."""

    profile_context_cap: int = 24
    """Max ACTIVE profile rows augment() returns in `ctx.user_profile`. The
    tier is meant to stay small and always-relevant; multi-valued slots
    (possession, recurring_activity, ...) accumulate over a long history, so
    the cap keeps the tier from bloating host prompts. Identity slots sort
    first, so they always survive the cut. `HyMem.profile()` is uncapped."""

    rules_enabled: bool = True
    """Idea B: load the `always_on` Rules tier (`hymem/rules.py`) into
    `ctx.rules` on every `augment()` call — standing behavioral imperatives
    ("always run the tests before pushing", "never suggest Docker") that are
    rules, not facts/preferences. `always_on` rules inject unconditionally;
    `contextual` rules inject only on a `trigger_entities`/`matched_entities`
    overlap. Purely additive (own SELECT, never a retrieval competitor). ON by
    default since the box adherence gate cleared 2026-07-27 (answerer
    deepseek-v4-flash, INDEPENDENT judge gpt-oss-120b, threshold 0.8, ON>OFF —
    `benchmarks/rules_compliance.py`). Inert until rules exist: an empty `rules`
    table (or a pre-v23 store) yields an empty tier, so the default adds no cost
    until a host calls `add_rule()`."""
    rules_context_cap: int = 16
    """Max ACTIVE rules `augment()` returns in `ctx.rules` — `always_on` first,
    then triggered `contextual`, stable by insertion order. Bounds the fixed
    per-call token cost the rules tier adds to every context."""
    rules_extraction_enabled: bool = False
    """Idea B write-side: during Phase-2 consolidation, route the imperative,
    durable sub-slice of `behavioral_markers` (rejection/style, and imperative-
    shaped corrections) into the `rules` table as `source='agent_inferred'`
    rules — so a standing instruction the user *stated in passing* ("stop
    suggesting Docker") becomes an always_on rule without a `add_rule()` call
    and WITHOUT a new LLM call (it reuses markers the dream already extracted).
    Off by default: an auto-rule injects into EVERY context, so a false positive
    is costly — flip only after the extraction precision probe clears on real
    dream markers (`benchmarks/rule_extraction_probe.py`; precision is the gated
    metric, not recall). Independent of `rules_enabled` (read side)."""
    rules_extraction_mode: str = "lexical"
    """Idea B write-side routing instrument (`hymem/rules_extract.py`), active only
    when `rules_extraction_enabled`. `lexical` = the deterministic classifier
    (`rule_scope_for_marker`, ~14% precision on real markers — the baseline arm).
    `llm` = an LLM durability tag decides (route iff standing AND confidence ≥
    `rules_extraction_confidence_min`); ONE batched call per dream, and it returns
    a canonical rule form that collapses paraphrases. `llm_fastpath` = trust a
    lexical imperative modal as standing (no call) and send only the ambiguous
    rest to the LLM. Arms are compared head-to-head by
    `benchmarks/rule_extraction_experiment.py`; the winning arm is what flips the
    write-side default. `lexical` keeps the current behaviour with no LLM cost."""
    rules_extraction_confidence_min: float = 0.75
    """Min LLM standing-confidence to mint a rule in `llm`/`llm_fastpath` mode.
    Higher = more precise, fewer rules. Swept by the extraction experiment to the
    point where precision clears 0.90 at max recall; the sweep result sets this."""
    rules_extraction_batch_size: int = 20
    """Markers per durability call in `llm`/`llm_fastpath` mode. A real judge is
    accurate on a small batch but collapses to all-non-standing on a large one
    (deepseek-v4-flash: 100% at 10, 0% at 111), so a dream's markers are judged in
    sub-batches of this size. Lower if a weaker judge still collapses."""

    facts_enabled: bool = False
    """Campaign E / E1 master switch (read side): surface the narrative-facts
    tier in `ctx.facts` — self-contained one-sentence statements extracted at
    dream time (schema v26), the middle granularity between a knowledge-graph
    triple and an episode summary. ADDITIVE like the profile/rules tiers: its
    own FTS/vec SELECT over `narrative_facts`, never a slot from any other
    tier's budget, so a populated facts store cannot crowd gold turns out of
    message/chunk/episode retrieval. Superseded facts (invalid_at set — E6's
    job) leave the tier but stay in the DB for audit. Degrades to [] on a
    pre-v26 store. False → the tier is always empty; extraction is governed
    separately by `facts_extraction_enabled`."""

    facts_extraction_enabled: bool = True
    """Campaign E / E1 write side: during dreaming, run ONE extra LLM call per
    session tail extracting narrative facts (`hymem/dreaming/facts.py`),
    append-only under the `facts.v2` prompt that cleared the G-F1b
    extraction-faithfulness gate (2026-08-02, 123/123 strict — the gate that
    kept this tier unbuilt while deepseek-v4-flash scored 0.55-0.76). Coverage
    follows the v24 watermark pattern with its own column
    (`sessions.facts_message_id`): steady-state re-dreams of unchanged
    sessions cost zero calls, a parse failure holds the watermark and retries,
    and a FACTS_PROMPT_VERSION bump extracts forward only (covered ranges are
    never re-extracted). Any material prompt change re-enters the
    `benchmarks/fact_probe.py` gate at ≥0.90 before this stays True."""

    facts_top_k: int = 8
    """Max FactHits `augment()` returns in `ctx.facts` (0 disables the tier's
    retrieval without touching extraction). FTS + optional vec KNN, RRF-fused
    like the episode tier."""

    dream_max_facts_per_session: int = 8
    """Cap on validated facts accepted from one per-session-tail extraction
    call; a runaway reply is truncated here before any row is written. 8 is
    the G-F1b density gate's criterion-3 threshold (median facts/session ≤8,
    measured median 7.0-8.0), so the cap and the gate agree on what a
    reasonable session yields."""

    coref_enabled: bool = True
    """Campaign E / E5: resolve anaphora + ellipsis in the query BEFORE any
    retrieval tier runs (`hymem/query/coref.py`). A follow-up like "what did she
    say about that?" or "en de prijs?" defeats every tier at once — BM25 has only
    stopwords, the vector points nowhere, no entity matches — because the referent
    is in the previous turn, not the query. The rewrite APPENDS the resolved
    referents (`query + " (context: medflow)"`) and never replaces, so retrieval
    keeps every original token and can only gain signal. ON by default: the
    default path is stdlib-only (no LLM call), it needs a `session_id` to have any
    recent turns to resolve against, and it fires only when a trigger matches AND
    a referent resolves — a self-contained query is returned byte-identical."""
    coref_max_turns: int = 6
    """Recent turns of the active session searched for the referent. Small on
    purpose: the antecedent of "that" is nearly always the thing just discussed,
    and a wide window pulls in stale entities that only dilute the appended
    clause."""
    coref_llm_enabled: bool = False
    """Optional Stage-2 fallback: when the heuristic fires but the graph knows no
    entity in the recent-turn window, spend ONE tiny `LLMClient` call to rewrite
    the follow-up as a standalone question. Off by default — the salient-token
    fallback covers the same cases for free, and this puts an LLM call on the
    query path (latency + cost on every under-specified follow-up). Only the
    TERMS the model introduces are used, and they are still appended, so even a
    hallucinated rewrite cannot remove a token the original query matched on."""

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

    graph_multihop_enabled: bool = False
    """Source 4 of `_graph_lookup`: query-time read-only BFS from directly-anchored
    entities across ALL predicates (vs. offline `inference.py`, which only chains
    `depends_on`). Bridges edges 1-hop retrieval misses — e.g.
    `atta —part_of→ medflow —deploys_to→ fly.io` from the seed `atta`. Additive
    (never filters; multi-hop-only candidates are discounted below any direct hit).
    Off by default — flip only after the bridging-edge recall probe clears the LME
    guard (Track A / `additional_planning.md` Idea A; tests/test_multihop.py)."""
    graph_multihop_max_hops: int = 2
    """Max chain length (edges from a seed) explored by Source 4. `2` reaches the
    bridge two edges out (the cheap first step, most of the gain); `3` goes deeper.
    < 2 disables the traversal entirely."""
    graph_multihop_decay: float = 0.5
    """Per-hop score multiplier (< 1) so a longer chain is strictly weaker than a
    shorter one — BrainDB's compounding edge-weight model. A fresh 2-hop chain
    scores ~conf²·decay² ≈ 0.5²·0.5² so it never outranks a 1-hop direct hit."""
    graph_multihop_min_score: float = 0.05
    """Prune frontier paths whose compounding score falls below this — bounds
    fan-out from hub nodes and keeps the query path LLM-free and fast."""
    graph_multihop_hub_degree_max: int = 32
    """Hub guard: a node whose active degree (edges where it is subject OR object)
    exceeds this is REACHED but never EXPANDED — the BFS will not fan out through
    it. In a personal-memory star every leaf is 2 hops from every other leaf via
    the `user` hub (degree = all facts about the user), so expanding hubs emits a
    flood of hub-mediated non-bridges (`road_trip ← user → driving_trip`) that
    dilute the true bridge out of `graph_top_k`. A genuine intermediate entity
    (`medflow`, degree ~2) stays well below this, so real chains still bridge. Set
    ≤ 0 to disable the guard (raw traversal). Only active when multi-hop is on."""

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

    value_supersession_enabled: bool = True
    """Schema v15 bi-temporal consumer. When True, the dream cycle closes the
    validity interval of an OLDER value-edge the moment a NEWER one supersedes it:
    among active, non-derived edges sharing subject + predicate but differing on a
    *typed-value* object (a number / percentage / count, a DATE, or a VERSION —
    classified from the object string; `kg_evidence.value_numeric`/`value_unit`
    only refine a bare number's missing unit), the edge with the earlier
    `valid_at` is retracted and its `invalid_at` set to the newer edge's
    `valid_at`. This is single-assertion supersession — one authoritative update
    suffices — distinct from the evidence-accumulation retract rule above which
    needs repeated negatives. Typed-value scoping means multi-valued facts (a
    project using many tools) are never collapsed. Default ON since the LME
    guard cleared (2026-07-02, 500q A/B: score-neutral +0.8pp overall, zero
    false positives across all firings — a footprint the benchmark cannot size,
    so the flip is a correctness call: a superseded edge left active is wrong
    context for the digest anchor, profile, ask() and conflicts()). Every firing
    logs `bitemporal.supersede ...` at INFO — a `prefers`/multi-valued row there
    is the rollback signal. Set False to disable.
    See `hymem/dreaming/value_supersession.py`.

    Watch note (2026-08-25, following the invalid_at re-assert fix in
    phase1._upsert_triple): the rollback signal is "a `prefers`/multi-valued
    row" — NOT "the same pair fires twice." A positive re-mention of a
    value-superseded edge now clears its invalid_at and reactivates it, so
    old and new value are BOTH active until the next dream re-retracts the
    loser by valid_at. That is pre-existing behaviour (the status flip already
    did it); the fix only makes invalid_at agree with it. Consequence: the
    `bitemporal.supersede` line can legitimately fire twice on the same pair.
    Do NOT roll back the feature on a double firing — that is the feature
    converging, not the guard failing.
    """

    reinforce_window_days: int = 30
    """Window for soft positive reinforcement from co-mention. Symmetric to
    decay_window_days."""

    profile_max_entries: int = 16
    insights_max_entries: int = 12

    prompt_version: str = "v9"

    dream_budget: int = 50
    """Maximum number of chunks to process per dreaming cycle."""

    dream_baseline_budget: int = 10
    """If the salience tier leaves budget unspent, drain up to this many
    non-salience-marked chunks (newest first) per cycle. Guarantees every chunk
    eventually flows through extraction even if it didn't trip the regexes."""

    chunk_extraction_max_attempts: int = 3
    """Consecutive failed extractions before a chunk is given up on and marked
    done anyway. A failed extraction is held for retry rather than marked, and
    the runner spends a budget slot on a chunk BEFORE knowing whether it will
    fail — so without a bound, a permanently-broken chunk consumes one of
    `dream_budget`'s slots on every dream forever, starving new chunks. Giving
    up logs at WARNING with the attempt count; set to 0 to retry indefinitely."""

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
    Also the age window for pruning stale procedures. Episodes are governed by
    episode_retention_days instead."""

    episode_retention_days: int = 0
    """Age window for pruning old episodes; 0 (the default) keeps them forever.
    Episodes are the leaves of the aggregation/digest tree, which full-rebuilds
    from live episodes each dream — deleting them makes the digest forget, so
    this is decoupled from retention_days. Set to a positive number of days to
    restore age-based pruning."""

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

    rerank_message_hits: bool = True
    """Also rerank the raw-message keyword tier (`message_hits`), not just the
    chunk tier (`fts_hits`). When True, augment() pulls a `rerank_top_k`-wide
    BM25 candidate pool from the `messages` table and reranks it down to
    `message_fts_top_k`, so a semantically-relevant turn sitting in the BM25
    tail can be lifted into the cut. This tier is the dominant recovery source
    on BEAM (most gold turns come back here, never via dreamed chunks), yet it
    was historically returned in raw BM25 order — the bulk of ranking-loss
    misses. Costs one extra rerank call per query when a reranker is wired and
    the pool exceeds `message_fts_top_k`. Set False to restore the raw-BM25
    behaviour (no extra rerank cost). No effect on the MR aggregate path, which
    counts rather than ranks."""

    ask_max_context_chars: int = 8000
    """Char budget for the rendered memory-context block `HyMem.ask()` sends to
    the synthesis LLM. The block is assembled most-authoritative-first (profile
    → graph facts → dated raw turns → ... → recent turns) and truncated from
    the tail, so tightening the budget sheds the softest evidence first.
    0 disables the cap."""

    ask_max_tokens: int = 1024
    """max_tokens for the single synthesis completion behind `HyMem.ask()`."""

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
