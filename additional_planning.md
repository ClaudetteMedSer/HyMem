# Additional Planning

Two ideas borrowed from [BrainDB](https://github.com/dimknaf/braindb), adapted to
HyMem's embedded, edge-typed architecture, plus the episode-granularity plan
(added 2026-07-02, see [Plan C](#plan-c--episode-granularity-in-dreaming)):

- **Idea A** — query-time multi-hop graph traversal with compounding edge weights.
- **Idea B** — `always_on` Rules as a first-class node type.
- **Plan C** — decision-grained episode extraction in dreaming.

Ideas A and B have been checked against the current RAPTOR/aggregation
architecture (see [§0](#0-raptor-interference-check)) and are clear to build.
Plan C is sequenced BEHIND the RAPTOR Stage 3c flip decision (see its
sequencing constraint).

> Reviewed for staleness 2026-07-02: `aggregate.py` line refs updated after the
> Option B snapshot fix landed (commit 8b36501); schema still v21; suite at 695
> after the day's landings (`episode_retention_days`, value-supersession v3
> version class, `HyMem.ask()`).

---

## 0. RAPTOR interference check

RAPTOR in HyMem is the **aggregation tier**: `dreaming/aggregate.py` builds the
tree at dream time, `HyMem.digest()` / `load_digest` serve the root, and
query-side consumption is `_aggregation_search` (`query/augment.py`), gated by
`cfg.aggregation_nodes_enabled` + `_aggregation_tier_fires` (default abilities
`("TR",)`). It operates over episodes/clusters; its only contact with
`knowledge_graph` is the digest's `_anchor_facts` block
(`aggregate.py:530`), which reads `status='active' AND derived=0 AND
invalid_at IS NULL` edges **at dream time**.

**Idea A is clear.** Multi-hop lives entirely inside `_graph_lookup`
(`query/augment.py:1432`), writes only to `ctx.graph_facts`, and is read-only —
it materializes no edges. `_aggregation_search` writes a different ctx field and
shares no state. The digest anchor never sees query-time expansion. Zero
interference.

**Idea B is clear, with one constraint.** A new `rules` table is migration v22
(schema is at `EXPECTED_SCHEMA_VERSION = 21`, `core/db.py:17`); migrations are
sequential/idempotent and don't touch the aggregation tables. The new
`ctx.rules` tier is independent of `_aggregation_search`.
**Constraint:** do NOT feed rules into `_anchor_facts`. That block's content
hashes into the RAPTOR root digest's cache id (`aggregate.py:~544`), so wiring
rules in would couple every rule edit to digest regeneration. Keep rules a
parallel augment tier only.

---

## Idea A — Query-time multi-hop traversal with compounding edge weights

### Motivation

The only multi-hop today is *offline* in `dreaming/inference.py`: a dream-time
BFS that materializes `derived=1` edges, and **only for `depends_on` chains +
the `uses→depends_on` cross-hop**. Every other predicate (`part_of`, `owns`,
`located_in`, `participates_in`, `runs_on`, `deploys_to`, …) is never
transitively connected. So a question like *"where is the project Atta works on
deployed?"* needs `atta —part_of→ medflow —deploys_to→ fly.io`, which neither the
offline closure (wrong predicates) nor `_graph_lookup` Source 1 (1-hop entity
anchor) bridges. BrainDB's 3-hop compounding traversal fills exactly this gap.

**Key decision:** this is a query-time, **read-only** expansion — Source 4 of
`_graph_lookup` — not new materialized edges. Per the additive-MR invariant
([project_mr_filter_killed]), it only ever *adds* candidates, never filters.

### Sketch

New helper near `_recency_edges` in `query/augment.py`:

```python
def _multihop_edges(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    seeds: list[str],
) -> dict[tuple[str, str, str], dict]:
    """BFS outward from seed entities up to cfg.graph_multihop_max_hops.

    Each frontier edge carries a path score = product of per-hop
    (smoothed_confidence × cfg.graph_multihop_decay). The decay (<1) makes a
    longer chain strictly weaker than a shorter one — BrainDB's compounding
    edge-weight model. Only hop>=2 edges are returned (hop-1 edges are already
    Source 1; re-emitting double-counts).
    """
    if cfg.graph_multihop_max_hops < 2 or not seeds:
        return {}

    reached: dict[str, float] = {s: 1.0 for s in seeds}   # node -> best path score
    out: dict[tuple[str, str, str], dict] = {}
    frontier = list(seeds)

    for hop in range(1, cfg.graph_multihop_max_hops):
        if not frontier:
            break
        ph = ",".join("?" * len(frontier))
        rows = conn.execute(
            _EDGE_SELECT + f"""
            WHERE status = 'active'
              AND (subject_canonical IN ({ph}) OR object_canonical IN ({ph}))
            """,
            frontier + frontier,
        ).fetchall()

        next_frontier: list[str] = []
        for r in rows:
            conf = (r["pos"] + 1.0) / (r["pos"] + r["neg"] + 2.0)
            for near, far in ((r["s"], r["o"]), (r["o"], r["s"])):
                if near not in reached:
                    continue
                path_score = reached[near] * conf * cfg.graph_multihop_decay
                if path_score < cfg.graph_multihop_min_score:
                    continue
                key = (r["s"], r["p"], r["o"])
                prev = out.get(key)
                if prev is None or path_score > prev["path_score"]:
                    d = _ensure_dict_from_row(r)   # same shape as _ensure()
                    d["path_score"] = path_score
                    d["hop"] = hop + 1
                    out[key] = d
                if far not in reached or path_score > reached[far]:
                    reached[far] = path_score
                    next_frontier.append(far)
        frontier = next_frontier
    return out
```

Wiring into `_graph_lookup`, right after Source 1 — seeded by **direct** entity
matches only (not overlap-only anchors; chaining a fuzzy link N times produces
garbage):

```python
    # Source 4 — multi-hop expansion from directly-anchored entities.
    if cfg.graph_multihop_enabled and not fallback_only_overlap:
        direct_seeds = [e for e in entities if e not in (overlap_info or {})]
        for key, d in _multihop_edges(conn, cfg, direct_seeds).items():
            c = candidates.get(key) or _ensure(_row_from(d))
            c["multihop_score"] = max(c.get("multihop_score", 0.0), d["path_score"])
            c["hop"] = d["hop"]
```

Scoring loop — multi-hop-only candidates get a discounted score and an honest
reason code so `why_retrieved` stays truthful:

```python
        elif c.get("multihop_score", 0.0) > 0:    # fallback path, reached only via chain
            score = c["multihop_score"] * recency_weight
            why.append(f"fallback:multihop:{c['hop']}hop")
```

### Config knobs (matching the existing `graph_*` style)

```python
    graph_multihop_enabled: bool = False
    """Source 4 of _graph_lookup: query-time BFS from matched entities across
    ALL predicates (vs. offline inference.py, which only chains depends_on).
    Off by default — flip after the recall probe clears the LME guard."""
    graph_multihop_max_hops: int = 3        # BrainDB's depth; 2 is the cheap first step
    graph_multihop_decay: float = 0.5       # per-hop multiplier (<1) so longer chains lose
    graph_multihop_min_score: float = 0.05  # prune frontier paths below this (bounds fan-out)
```

### Risks
- **Fan-out / latency.** A hub like `uv` explodes the frontier;
  `graph_multihop_min_score` + a per-hop frontier-width cap bound it. The query
  path must stay LLM-free and fast — measure p95 `augment()` latency.
- **No double-counting derived edges.** Offline `depends_on` closure already
  exists; dedup Source 4 against `candidates` from Sources 1/3 on `(s,p,o)`.
- **Decay vs. Laplace confidence.** Fresh edges start at 0.5, so a 3-hop chain
  is `~0.5³ × decay³ ≈ 0.002` — correctly invisible vs 1-hop. Verify
  `min_score` doesn't prune everything; the probe set tells you.

### Validation — bridging-edge recall@k (primary), LME guard (non-regression)

End-to-end LME/BEAM accuracy is too noisy to attribute a multi-hop change to
([project_lme_variance_band]: strict LME swings ~4 q/category, per-category
deltas under ~±5pp are noise). A retrieval-recall probe is far more sensitive
and tests the mechanism directly.

**Primary metric — bridging-edge recall@k.** Adapt `benchmarks/gold_rank_probe.py`:
build a probe set of questions needing a 2-/3-hop chain, label the *bridging
edge* (the hop-2/hop-3 `(s,p,o)` 1-hop retrieval misses), and measure whether it
appears in `graph_facts[:graph_top_k]`.

1. **Build the probe set (~60–100 items), two sources:**
   - **Synthetic, controlled** — seed a DB with known chains
     (`atta —part_of→ medflow —deploys_to→ fly.io`), ask the bridging question,
     assert recall. Ground truth + deterministic decay/min_score tests. Lives in
     `tests/` as real pytest (mirrors the P4 box-gate style).
   - **Mined from LME/BEAM** — filter multi-session-reasoning items to those
     needing a cross-predicate hop; hand-label the bridge. Use
     `benchmarks/longmemeval_adapter.py` / `benchmarks/beam_adapter.py`.

2. **A/B protocol** (same DB, `graph_multihop_enabled` off → on):
   - **bridging-edge recall@8** (primary) — must rise on multi-hop items.
   - **recall@8 on a 1-hop control set** — must NOT drop (multi-hop must not
     crowd out direct hits; the additive invariant as a metric).
   - **p50/p95 `augment()` latency** — cost gate; budget e.g. p95 < 1.5×
     baseline before any LME run.

3. **Then** run the full LME guard (canonical 70.0% full-dream
   [project_lme_canonical_fulldream], MS floor 51.9) as **non-regression only**,
   not a tuning signal. Tune `decay`/`min_score`/`max_hops` against the recall
   probe; LME just confirms nothing broke.

**Sweep:** `max_hops ∈ {2,3}`, `decay ∈ {0.4,0.5,0.6}`,
`min_score ∈ {0.02,0.05,0.1}`, scored recall@8 vs p95 latency — pick the Pareto
knee. `max_hops=2` is the likely first ship (cheapest, most of the gain).

---

## Idea B — `always_on` Rules as a first-class node type

### Motivation

HyMem's "always loaded" layer is scattered across the MEMORY.md / USER.md
auto-sections (`augment.py:322`), `profile_entries`, and the closed-vocabulary
`user_profile` slots. None is a clean standing imperative ("always run tests
before pushing", "never suggest Docker") — they're facts/preferences, not rules,
and they compete for the capped `insights_max_entries=12` / `profile_max_entries=16`
budgets. BrainDB's `always_on` rule node is a dedicated abstraction injected into
every context call. This is a cleaner home for the imperative subset.

### Sketch — schema (migration v22)

```sql
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    scope TEXT NOT NULL DEFAULT 'always_on'
        CHECK (scope IN ('always_on','contextual')),
    -- contextual rules inject only when a trigger entity matches;
    -- always_on inject every augment() call (BrainDB semantics)
    trigger_entities TEXT NOT NULL DEFAULT '[]',   -- JSON, for scope='contextual'
    source TEXT NOT NULL DEFAULT 'user',           -- user | agent_inferred
    pos_evidence INTEGER NOT NULL DEFAULT 1,
    neg_evidence INTEGER NOT NULL DEFAULT 0,
    valid_at TIMESTAMP,                            -- bi-temporal, like knowledge_graph / user_profile
    invalid_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active','retracted')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(scope, status, invalid_at);
```

### Sketch — augment + extraction

```python
# augment.py, AugmentedContext
    rules: list[Rule] = field(default_factory=list)

# augment(), after matched entities computed (~line 531)
    if cfg.rules_enabled:
        ctx.rules = _load_rules(conn, ctx.matched_entities, cap=cfg.rules_context_cap)
```

- `always_on` rules load unconditionally (the whole point); `contextual` rules
  load iff their trigger overlaps `ctx.matched_entities`.
- **Extraction:** route a sub-slice of existing `behavioral_markers`
  (`correction`/`rejection` that are imperative + durable) during Phase-2
  consolidation — no new LLM call, preserving one-call-per-chunk discipline.
- **Direct API:** `hy.add_rule(text, scope=...)` + an MCP tool, since rules are
  often *told*, not inferred. Follow the `HyMem.ask()` / `hymem_ask` pattern
  (landed 2026-07-02: `hymem/query/ask.py` + `server.py`) for the API-plus-tool
  pairing.
- **Supersession** reuses `bitemporal.py` interval-closing — a contradicting
  rule closes the prior's `invalid_at` rather than overwriting.

### Honcho surface
Rules ride along in `GET .../context` and `peers/{pid}/card`, ahead of MEMORY.md
— matching how BrainDB's `always_on` injects into every context call.

### Constraint (from §0)
Do **not** add rules to `_anchor_facts` (`aggregate.py:530`). That block hashes
into the RAPTOR root digest cache id; coupling rule edits to digest regeneration
is undesired. Rules stay a parallel augment tier.

### Validation — compliance gate, NOT a benchmark number

Rules are behavioral imperatives; LME/BEAM have no questions testing "did the
agent obey a standing instruction." Forcing it onto those suites measures noise.
Mirror the P4 profile-tier box gate ([project_p4_profile_tier], ~95% pass):

1. **Synthetic compliance harness.** Fixed (rule, probe-message,
   expected-behavior) triples — e.g. rule "never suggest Docker" + a probe that
   tempts it. Run rules-injected vs not; LLM-judge (Claude) scores *adherence*
   (rule present in context + response respects it). A pass/fail box gate, not a
   leaderboard.
2. **Mechanical pytest (no LLM):** `always_on` rules appear in every
   `augment()` ctx regardless of query; `contextual` rules appear iff trigger
   matched; a contradicting rule closes the prior `invalid_at` (reuse bitemporal
   tests); retracted rules never surface.
3. **Cost watch.** Rules add fixed token cost to every call — log it; "every
   context call" is the hot path.

---

## Plan C — Episode granularity in dreaming

*(added 2026-07-02 — the "point 5" carry-over from the competitive/architecture
review)*

### Motivation

- **BEAM floor post-mortem** (benchmarks/beam_investigation_notes.md): EO/SUM
  failures traced to episodes too ABSTRACT ("developed budget tracker with
  Flask, added auth") to decompose into the rubric's event sequence. The one EO
  question that flipped did so exactly when detailed episodes led the context —
  mechanism verified, magnitude unmeasurable at sample=3. Episode granularity is
  a CORE dreaming concern, not an adapter knob.
- **Digest coverage gap**: 65/91 sessions covered on the prod store (~42%
  episode-recall gap upstream) — the digest tree can only fuse what episodes
  carry.
- **Rate-distortion framing** ("Remember the Decision, Not the Description",
  arXiv 2605.10870): store *decisions and outcomes* at retrieval granularity.
  The `episodes.outcome` field already exists; the failure is granularity, not
  schema.
- **Retention decoupling** (`episode_retention_days = 0`, landed 2026-07-02):
  episodes are now permanent, so each one should carry more retrievable signal —
  and `HyMem.ask()` consumes them directly in its evidence block.

### Sketch

- Prompt-side change to the per-session digest call
  (`hymem/dreaming/digest.py`; bounded by `dream_digest_max_chars` 12000 /
  `dream_digest_max_tokens` 3072): decompose into event-grained episodes — one
  episode per decision/change/outcome, concrete values (names, numbers, dates,
  versions) in the summary, target roughly 3–8 per substantive session instead
  of 1–2 blobs. Keep the outcome discipline (outcome REQUIRED when the session
  reached one).
- New `dream_max_episodes_per_session` config cap (mirror
  `profile_max_items_per_session = 16`) so a runaway response is truncated
  before persistence.
- **Version the prompt + re-extraction guard.** Mirror the
  `PROFILE_PROMPT_VERSION` / fusion-salt lesson: a granularity change must
  invalidate prior extractions or old blob episodes silently persist (UPSERT
  refreshes only matching message ranges). Reuse the per-session skip-guard
  pattern (`sessions.profile_prompt_version`, schema v19) with an episodes
  prompt version so unchanged sessions cost zero tail calls.

### Validation — box gate + probes, explicitly NOT BEAM EO

The variance-band discipline applies twice over: BEAM EO at sample=3 has a
±12.5pp/category noise floor (measured, via the CR control), so the aggregate
cannot see this change. Gates, in order:

1. **Mechanical pytest** (StubLLMClient): multi-episode persistence per
   session, cap enforcement, stable-id UPSERT semantics on re-dream, prompt
   version guard.
2. **Qualitative box gate** (mirror the P4 profile gate, pass ≥ 0.9): hand-score
   ~20 sessions' episodes on (a) traceability — every episode grounded in actual
   turns, no invented outcomes; (b) granularity — decision-level with concrete
   values; (c) session coverage. `HyMem.ask()` doubles as a cheap spot-check
   harness here.
3. **`benchmarks/episode_coverage_probe.py`** before/after — coverage should
   rise from the ~42% gap.
4. **LME full guard as NON-REGRESSION only** (canonical 70.0 full-dream, MS
   floor 51.9). Not a tuning signal.
5. **Cost watch**: more episodes ⇒ more `vec_episodes` embeddings + a larger
   clustering input each dream. Blocking (`aggregation_blocking_top_k = 24`)
   bounds the pair tests; measure dream wall-clock and fusion-call counts
   before/after.

### Sequencing constraint (hard)

Run AFTER the RAPTOR Stage 3c flip decision. The flip gate is a quiescent-dream
node-reuse watch (~90%+ hold, no spikes) following the Option B snapshot-ceiling
fix (commit 8b36501). A granularity change rewrites episode membership
wholesale — under the watch it is indistinguishable from the spurious-rebuild
failure mode it exists to rule out. Land the flip (or the no-flip verdict)
first, then granularity, then re-verify reuse once on the new episode set.

---

## Recommended sequencing

1. **Idea A at `max_hops=2`** first — measurable, attacks a concrete retrieval
   gap, clean go/no-go from the recall probe. Build the synthetic multi-hop
   probe as pytest **before** touching `_graph_lookup` (the variance-band
   discipline requires a gate before tuning).
2. **Idea B** second — lower implementation risk (purely additive context), but
   the payoff is behavioral and only validatable via the compliance gate, so it
   won't move headline numbers and shouldn't be judged by them.
3. **Plan C** independently of A/B but strictly after the RAPTOR flip decision
   (see its sequencing constraint) — A/B don't touch episodes, so they can
   proceed while the reuse watch runs.
