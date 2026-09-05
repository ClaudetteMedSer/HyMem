# Additional Planning

> **Historical engineering ledger.** LongMemEval numbers such as 70.0 and old
> Honcho/Hindsight comparisons below document earlier local/sample decisions;
> they are not current official-comparable claims. The strict LME harness now
> pins dataset/evaluator identities, labels full-set work development-only, and
> has no newly rerun HyMem score. Current external figures, if cited, require
> protocol caveats: Honcho vendor 90.4 (Haiku 4.5) / 92.6 (Gemini 3 Pro),
> Hindsight official March 2026 94.6 single-query, Mnemosyne 98.9
> Recall@All@5 on 100 items (not end-to-end), and BEAM-100K 65.2 separately.

Two ideas borrowed from [BrainDB](https://github.com/dimknaf/braindb), adapted to
HyMem's embedded, edge-typed architecture, plus the episode-granularity plan
(added 2026-07-02, see [Plan C](#plan-c--episode-granularity-in-dreaming)),
plus the narrative-facts campaign (added 2026-07-30, see
[Campaign E](#campaign-e--the-narrative-facts-roadmap-added-2026-07-30)),
plus the MindCache state-anchor expansion plan (added 2026-08-14, see
[Plan D](#plan-d--state-anchor-expansion-borrowed-from-mindcache-added-2026-08-14)),
plus the Grove Memory borrows (added 2026-08-18, see
[Plan E](#plan-e--grove-borrows-borrowed-from-grove-memory-added-2026-08-18)):

- **Idea A** — query-time multi-hop graph traversal with compounding edge weights.
- **Idea B** — `always_on` Rules as a first-class node type.
- **Plan C** — decision-grained episode extraction in dreaming.
- **Plan D** — query-time state-anchor expansion (borrowed from MindCache).
- **Plan E** — Grove borrows: labeled wildcard, recovery gauge, trajectory
  resurfacing, null-model gate (borrowed from Grove Memory).

Ideas A and B have been checked against the current RAPTOR/aggregation
architecture (see [§0](#0-raptor-interference-check)) and are clear to build.
Plan C was sequenced BEHIND the RAPTOR Stage 3c flip decision — **UNBLOCKED
2026-08-26**, when G-FLIP passed 7/7 on the re-anchored window and
`aggregation_nodes_enabled` flipped to True. Its OTHER precondition is
unchanged and still binding: the episode rewrite must clear
`benchmarks/fact_probe.py`'s faithfulness bar on the candidate model before
it ships. It must also not overlap the post-flip verification dream.

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

### STATUS 2026-07-25 — steps 1–2 BUILT (probe + feature), default OFF

- **Feature** landed in `hymem/query/augment.py`: `_multihop_edges` (read-only
  BFS) + Source 4 wiring (direct seeds only, dedups Sources 1/3 on `(s,p,o)`) +
  a discounted-scoring branch emitting `fallback:multihop:{n}hop`. Four config
  knobs in `hymem/config.py` (`graph_multihop_{enabled=False,max_hops=2,
  decay=0.5,min_score=0.05}`). **Two sketch bugs fixed** (probe-before-code paid
  off): the BFS `range(1, max_hops)` was one round short — with `max_hops=2` it
  never reached the worked example's own bridge — corrected to `range(1,
  max_hops+1)` emitting from round 2; and the seed's 1-hop edges were re-emitted
  as hop-2 (only the `entity_match` dedup masked it) — now an explicit
  seed-incident filter drops them. Compounding verified exactly: hop-2 = conf²·
  decay² = 1/9, hop-3 = conf³·decay³ = 1/27 (fresh 3-hop chains sit ~min_score).
- **Probe — synthetic half**: `tests/test_multihop.py` (10 pytest, all green in
  the 720-test suite). Encodes G-A1 as assertions: bridge recall 0→present on
  flip, 1-hop control non-regression (same score, no multihop chip), decay/
  min_score/depth/dedup determinism.
- **Probe — mined half**: `benchmarks/multihop_probe.py` (+ `_example.json`
  schema/demo). LLM/embedding-free `gold_rank_probe.py`-style harness; consumes
  a labeled probe JSON (fresh-seed OR `--store <built.sqlite>` read-only), runs
  the off→on recall@k A/B per set + p50/p95 `_graph_lookup` latency, prints the
  G-A1 advisory (multihop rose / control held / p95<1.5× with a sub-1ms noise
  floor). Sweep points drive via `--max-hops/--decay/--min-score`; `--json` for
  the sweep loop.
- **Miner**: `benchmarks/multihop_miner.py` — pre-fills the labeled probe set so
  Phase A is a verify pass, not authoring. Reuses `_multihop_edges` (so a
  proposed bridge is exactly what Source 4 fetches) + `match_known_entities`,
  ranks candidate edges by gold-answer token overlap, and auto-sorts each MR/TR
  question into multihop (a hop≥2 bridge explains the answer) / control (a direct
  1-hop edge does) / dropped. Emits probe-compatible items + `_`-hints
  (`_gold/_hop/_answer_overlap/_alt_bridges`). Gold-for-labeling is legitimate
  (ground truth), not read at retrieval time. **Two modes:**
  (1) **`--store`** — mine an existing dreamed store, LLM-free, seconds.
  (2) **`--lme-data`** — per-question: rebuild + **dream each question's own
  haystack to completion** (loops `dream()` until `not budget_exhausted`), then
  mine it. Sidesteps the `dream_budget=50`-per-cycle under-dream that made a
  combined LME store yield a false-empty graph (37 edges / 4 subjects → false 0%
  G-A1); faithful to LME's isolated per-question retrieval. Emits a self-contained
  fresh-seed `edges` block (probe needs no `--store`); prints per-question dream
  health (avg edges/store) to catch a non-extracting dream LLM. Dreams via
  `--dream-model` (default v4-flash, thinking-disabled; `stub` = plumbing test).
  Both modes share `_mine_question`. End-to-end verified: store→probe PASS,
  per-question plumbing + self-contained edges→probe PASS.
- **Guard flag**: `longmemeval_adapter.py --graph-multihop` (+
  `--graph-multihop-max-hops/--decay/--min-score`) wires the swept knobs into the
  adapter config and records them in run metadata — makes G-A2 runnable.

### STATUS 2026-07-26 — box G-A1 on LME FAILED → real cause = hub-dilution → hub guard added, G-A1 PASSES locally

The box ran the per-question miner on 40 MR/TR questions (dreams **healthy** —
200+ edges / 30+ subjects each; the `dream_budget` loop worked), then G-A1 on the
self-contained slice: **0/4 bridges, FAIL**. The box's stated cause ("BFS is
subject→object only, object-only seeds have no outgoing edges") is **wrong** —
`_multihop_edges` traverses `((s,o),(o,s))` (augment.py:1709), so it is already
bidirectional and an object-only seed *does* reach `user`. The **real** cause:

- **Hub dilution.** Every LME "bridge" runs through the `user` super-hub
  (`road_trip ← user → driving_trip`). A leaf seed reaches `user` in 1 hop; hop 2
  expands `user`, which is incident to *hundreds* of edges (worse in the merged
  slice, which unions 40 questions' `user` edges). All emit as ~equal-weight
  hop-2 candidates; `graph_top_k=8` truncates; the true bridge is buried. **Not
  "can't reach" — "reaches everything and keeps 8."**
- **The deeper truth:** a path through a super-hub is not a bridge — it holds for
  *every pair* of things the user mentioned. **LME's personal-memory star has ~0
  genuine (non-hub) intermediate-entity bridges** (40 questions → 4, all
  hub-mediated). LME **cannot validate Track A** regardless of dream quality; this
  confirms the standing prediction empirically. Hub paths are also net-negative in
  production (flood query-irrelevant `user` edges).

**Fix — hub guard (degree-cap):** new knob `graph_multihop_hub_degree_max=32`
(config.py) + `_active_degrees` helper. A node whose active degree exceeds the cap
is **reached but never expanded** — edges INTO it can still emit, but the BFS never
fans OUT through it. Genuine intermediates (`medflow`, degree 2) stay far below the
cap so real chains still bridge; the `user` hub (degree hundreds) is inert. `≤0`
disables. **Deliberately does NOT rescue the 4 LME items** (they aren't bridges) —
mechanism > score, no tuning toward them.

- **pytest** (`tests/test_multihop.py`, now 14, all green in the 724-suite): guard
  blocks fan-out through a degree-40 hub; guard-off floods ~39 siblings (proves the
  guard is what suppresses it); low-degree intermediate still bridges with a hub in
  the same store; end-to-end leaf-seed of a star yields **no** `fallback:multihop`
  fact (ON == OFF on a star → inert, not harmful).
- **Local G-A1 substrate**: `benchmarks/multihop_genuine_bridges.json` — 5 genuine
  low-degree chains + 4 controls, needs **no box, no LME**. `multihop_probe.py`
  reads **G-A1 PASS**: multihop 0→100%, control 100→100%, latency sub-floor. This
  is the canonical mechanism gate; LME/BEAM slices only measure *substrate
  richness*, which for personal-memory graphs is ~0.

- **PENDING (box, now optional/Hermes-scoped)**: the mechanism gate is CLOSED
  locally. The remaining reads are substrate-dependent and belong on a **Hermes
  production graph** (which has genuine intermediate entities), not LME: mine that
  graph → G-A1 on real bridges → sweep → G-A2 non-regression (vs 68.4%). Ship
  default stays `False`; on LME the feature is provably inert with the guard on.

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

**Superseded 2026-09-04:** that historical design was removed from the Honcho
surface. Process-global rules have no workspace/peer ownership proof and must
not cross a tenant-scoped route. Rules remain available through native HyMem
and MCP APIs only until they carry explicit workspace, peer, and session
provenance; Honcho representation/context/chat/card now exclude their text.

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

### STATUS 2026-07-26 — core mechanism BUILT (probe-first), default OFF

The mechanical compliance gate and the tier it depends on are landed; the
LLM-adherence half stays a box gate.

- **Schema**: migration `023_rules.sql` + the `rules` table in `schema.sql`;
  `EXPECTED_SCHEMA_VERSION` 22 → **23**. (The §0 note above is stale — it was
  written when head was 21 and called this "v22"; head had already moved to 22,
  so the rules table is **v23**.) Bi-temporal (`valid_at`/`invalid_at` +
  `status`), `text UNIQUE`, `scope` and `source` CHECK-constrained.
- **Feature** `hymem/rules.py`: `Rule` + `load_rules` (always_on unconditional;
  contextual gated on `trigger_entities`∩`matched_entities`, canonicalized both
  sides; always_on-first, capped; degrades to `[]` pre-v23) + `add_rule`
  (redaction-scrubbed persist; `text`-UPSERT reinforces `pos_evidence` and
  revives a retracted row; `supersedes` closes a prior interval) + `retract_rule`.
- **Wiring**: `AugmentedContext.rules` + a `load_rules` call in `augment()`
  right after `matched_entities` resolves, gated by `cfg.rules_enabled`
  (default **False**) + `cfg.rules_context_cap=16`. Purely additive (own SELECT,
  no tier's budget touched). **NOT** in `_anchor_facts` (§0 honoured).
- **API**: `HyMem.add_rule(text, scope=, trigger_entities=, source=,
  supersedes=)` / `HyMem.retract_rule(id)` — the *told-not-inferred* direct path,
  mirroring `HyMem.ask()`.
- **Render wiring** (added 2026-07-27): `render_context` in `hymem/query/ask.py`
  emits a `=== STANDING RULES (always follow) ===` section FIRST (never shed by
  tail-truncation), and `_system_prompt(has_rules)` appends `ASK_RULES_DIRECTIVE`
  (obey-not-quote) to `ASK_PROMPT_V1` ONLY when rules are present — so the
  versioned base prompt is byte-identical for every current (no-rules) consumer.
  Without this the answerer never saw the rules; it is what makes adherence
  measurable. Covered by 3 more pytests (render-first, directive-iff-rules,
  ask() end-to-end).
- **Gate — mechanical (`tests/test_rules.py`, 12 green in the 736-suite)**:
  always_on injects on every query; default-OFF stays empty; contextual fires
  iff trigger overlaps; always_on ranks before contextual; supersession +
  retraction interval-close and stop surfacing; re-assert reinforces without
  duplicating; pre-v23 degrades; rules render first + directive composes. E2E
  smoke: fresh store → v23, rule surfaces on an unrelated query.
- **Gate — LLM adherence (BOX-TESTABLE NOW)**: `benchmarks/rules_compliance.py`
  + `benchmarks/rules_compliance_runbook.md`. Self-contained (6 rule/tempting-probe
  triples, no LME/BEAM, no dream). Runs `ask()` ON vs OFF per triple, LLM-judges
  compliance, and gates on THREE checks: (1) ON adherence ≥ `--threshold` (0.8),
  (2) **ON > OFF** (the rule caused it, not the base model), (3) rule present in
  every ON / no OFF context (mechanical invariant). Stub-mode plumbing verified;
  needs a box run with a real answerer + an INDEPENDENT judge.
- **GATE CLEARED 2026-07-27 → default FLIPPED ON.** Box ran
  `benchmarks/rules_compliance.py` with answerer `deepseek-v4-flash` + an
  INDEPENDENT judge `openai/gpt-oss-120b` (OpenRouter), threshold 0.8 — PASS on
  all three checks (ON≥0.8, ON>OFF, rule-present invariant). `rules_enabled`
  default `False`→`True` in `hymem/config.py` (docstring updated); the
  `test_default_off_no_rules_tier` assertion was repurposed to the new default-ON
  behaviour + a `test_rules_tier_can_be_disabled` opt-out test (local suite 737
  green; box 651). The tier is INERT until a host calls `add_rule()` (empty
  `rules` table → empty tier), so ON-by-default adds no cost on stores without rules.
### STATUS 2026-07-27 — extraction routing + surfaces BUILT; 3-gate scorecard

Idea B is now backed by three data-driven gates (see
`benchmarks/rules_compliance_runbook.md`): **adherence** (CLEARED, above),
**extraction precision**, **overhead**.

- **Extraction routing (write side)**: `rules.route_markers_to_rules` +
  `rule_scope_for_marker`, wired into runner Phase-2 (gated `cfg.rules_extraction_enabled`,
  default **OFF**), reusing already-extracted markers → NO new LLM call. Policy:
  `style` routes on kind; `rejection`/`correction` require an imperative cue
  (`_DIRECTIVE_RE`); `preference` never routes. `DreamReport.rules_extracted`
  reports the count. Emits `source='agent_inferred'` rules; idempotent via
  add_rule's text-UPSERT.
- **Extraction PRECISION gate** (`benchmarks/rule_extraction_probe.py`,
  deterministic, runs anywhere): scores `rule_scope_for_marker` on a hand-labeled
  marker set. **Data-driven tuning loop paid off**: first run **85% (FAIL)** — the
  3 FPs were rejection *one-offs* ("rejected the Tuesday meeting"); tightened the
  policy to require the directive cue for rejection too (past-tense forms
  excluded) → **100% precision / 100% recall (PASS)**. Precision is gated
  (default ≥0.90); a false rule pollutes every call. Box must re-validate on REAL
  dream markers (`--labels markers.json`) before flipping the write-side default.
- **OVERHEAD gate** (`benchmarks/rules_overhead.py`, deterministic): ON/empty
  adds **0 rendered chars** (proves the ON-by-default read side is free until
  `add_rule()`); ON/full (16 rules) = 1138 chars / **0.034ms p95** overhead. PASS.
- **Surfaces**: `HyMem.add_rule/rules/retract_rule`; MCP `hymem_add_rule` +
  `hymem_list_rules` (`server.py`); Honcho — active rules lead the peer card +
  peer/session context ahead of MEMORY.md (`honcho/app.py::_rules_block`).
  **Superseded 2026-09-04:** the Honcho clause is retained here as history, but
  unowned global rules are now native/MCP-only and are excluded from every
  workspace-scoped Honcho read until a provenance model exists.
- **Gates — mechanical (`tests/test_rules.py`, 21 green; full suite 746)**: adds
  classifier + end-to-end dream-routing (routes imperative markers, skips
  one-offs, respects the write-side gate) + `list_rules` + MCP tools + Honcho
  card/context surfacing + inert-when-empty.
- **PENDING**: (1) box re-validate extraction precision on real dream markers →
  flip `rules_extraction_enabled` ON. (2) optionally sweep the adherence probe
  with a larger/adversarial probe set for the competitive writeup.

### STATUS 2026-07-27 — lexical extraction FAILED on real markers → LLM-durability-tag extractor + A/B experiment engine prototyped

The write-side precision gate ran on **real** dream markers and **failed at 8.3%**
(1 TP / 11 FP / 37 FN). Two rounds of lexical tightening confirmed a ~14% ceiling:
`rejects?`/`refuses?` fired on 100% of rejection markers (kind-restatement leak;
removed, FP 73→11 + present-tense one-offs added to `rule_extraction_probe.py` as
local regression guards), and the residual FPs are one-off *corrections* carrying
incidental modals ("X was Dutch **instead** of English", "**should** be automatic").
**Standing-vs-one-off is semantic, not lexical** — the gate correctly blocks the
flip; further regex trimming is a dead end. Decision: `rules_extraction_enabled`
**stays OFF**; auto-extraction is **R&D**, not a launch blocker. Shippable value =
read side (ON, adherence-cleared) + told-path surfaces (`add_rule` via API/MCP/Honcho).

**Prototype (option b + a):**
- `hymem/rules_extract.py` — batched **LLM durability tagger** (`judge_durability_batch`,
  ONE call per dream; asks standing-rule-vs-one-off; returns standing/confidence/
  **canonical rule** which collapses paraphrases so `add_rule`'s UPSERT `pos_evidence`
  becomes a cross-session recurrence counter). `route_decisions` dispatches
  `cfg.rules_extraction_mode` ∈ `lexical`|`llm`|`llm_fastpath`, gated on
  `rules_extraction_confidence_min`; every failure degrades to "don't mint".
- Wired live: `route_markers_to_rules(conn, cfg, llm=)` ← runner passes `llm`.
  Config `rules_extraction_mode` (default `lexical`) + `rules_extraction_confidence_min`.
- **Experiment engine** `benchmarks/rule_extraction_experiment.py` — scores every
  arm (mode × τ-sweep × promotion{immediate/and/recurrence/or_highconf} × N) in one
  judgment pass; `--sim` fake-judge offline, real `--answer-model` on box; precision
  gated 0.90, prints best arm at max recall. Spec + decision rules =
  `benchmarks/rules_extraction_ab.md` (E1 instrument, E2 τ-frontier, E3 repetition,
  E4 independent-judge, E5 cost, E6 LME non-regression).
- **Offline `--sim` findings** (synthetic, NOT real numbers — mechanics only):
  llm 100% vs lexical 70%; **`llm_fastpath` DOMINATED (77%) — the lexical shortcut
  re-imports the FP leak**; repetition gating is a **robustness** lever (restores
  precision a noisy judge loses, at a recall cost), so build schema v24
  (`status='provisional'`) ONLY if E3 on real markers earns it.
- Tests: `tests/test_rules_extract.py` (13); full suite 759 passed.

### STATUS 2026-07-28 — write-side auto-injection CLOSED; candidate-SUGGESTION shipped

The box ran the experiment on real markers across the full pipeline. Findings, in
order:

1. **Instrument confirmed, gate not cleared.** The LLM tagger is **7×** lexical on
   precision (59% vs 8% on the rule-enriched set) at ~95% recall — but 59% is short
   of 0.90, and drops to ~45% on the natural-base-rate 614-marker set (a base-rate
   effect, the *more* honest number).
2. **A kind-filter LEAK, fixed.** The full run first read **5.8%** because
   `route_decisions`/`judge_corpus` sent EVERY kind to the tagger — 1,768
   `preference` markers flooded it. The `_RULE_KINDS` gate lived only in the lexical
   classifier; the LLM path bypassed it. Fixed with `rules.is_rule_eligible_kind`,
   a single eligibility gate BOTH paths call BEFORE the tagger (durability ≠
   eligibility; a durable preference still belongs in the profile tier).
3. **Adjudication was inconclusive, not confirmatory.** An independent blind judge
   (gpt-oss-120b) to check whether the tagger's "FPs" were mislabels scored **50% on
   its controls** — chance — tripping the distrust guard. So "0/25 overturned" is
   not evidence; the labels stayed suspect.
4. **The labels WERE the problem (`--by-kind`).** Per-kind at τ=0.90 the tagger is
   `rejection` **69%**/99%, `style` "0%", `correction` ~0%. All 90 gold rules were
   labeled `rejection`; `style`/`correction` had ZERO — impossible. Eyeballing
   settled it: the `style` markers ("be concise", "class-level skills", "active
   updates") ARE durable directives *mislabeled* not-rule (and already live in the
   profile as preferences → auto-minting them = duplication); `correction` markers
   are genuinely one-off. Corrected precision ~73%. The FPs are **not concentrated
   in a kind** — no `_RULE_KINDS` lever helps.
5. **Repetition-gating has no validation corpus.** The policy layer clears 0.90 only
   at ~3–5% recall (nothing recurs) on LME (star topology), MSC (preference-shaped),
   AND real Honcho data. Only Honcho could show recurring imperatives, and it shows
   them sparse — suggesting the sparsity is intrinsic (rules are said once, expected
   to hold).

**Decision: auto-*injection* is closed as a dead-end on available data.** The tagger
is a strong high-recall *detector*, so the answer is **candidate SUGGESTION** —
`HyMem.suggest_rules()` + MCP `hymem_suggest_rules` (read-only; ranked, de-duped
`RuleCandidate`s with marker/session counts + `already_active`; the human confirms
via `add_rule`, which also makes the profile-vs-rules tier-placement call). Built +
tested 2026-07-28 (`tests/test_rules.py`, full suite 775). `rules_extraction_enabled`
stays OFF. Repetition-gating / schema v24 is NOT built — WATCH ITEM: as the Honcho
store grows, re-check imperative recurrence (`msc_adapter.py --probe-mode recurrence`
→ `rule_extraction_experiment.py --policy-from-canonical`); sparse → close v24
permanently, dense → the E3 path reopens. Full scorecard:
`benchmarks/rules_compliance_runbook.md`; experiment design: `rules_extraction_ab.md`.

---

## Plan C — Episode granularity in dreaming

*(added 2026-07-02 — the "point 5" carry-over from the competitive/architecture
review)*

> **~~SUBSUMED 2026-07-30 by Campaign E, Step 4 (E1 build, below).~~
> UN-SUBSUMED same day — E1 was banked dead by G-F1 (see Campaign E, Step 1).**
> The `narrative_facts` artifact class that was to have absorbed this plan's
> decision-grained granularity goal does not exist and will not be built:
> extraction faithfulness measured 0.55–0.76 against a required 0.90, twice,
> across a deliberate prompt revision. *(Amended later the same day: revival
> gate `G-F1b` — one pre-registered new-dreamer arm, Campaign E Step 1 — is now
> authorized; if it passes and Step 4 builds, the subsumption comes back into
> force. The warning below applies to G-F1b's candidate model identically:
> Plan C's episode rewrite must clear the same faithfulness bar on THAT model
> before it runs.)*
>
> **Read that as evidence about THIS plan too, not just about E1.** Plan C is
> also a generative rewrite — it asks a model to re-cut episodes at decision
> granularity — and G-F1 is the only measurement the repo has of what that model
> does when asked to produce self-contained items from session turns. It invented
> biographical detail at roughly 1 claim per 3–4 items, reproduced the same
> inventions under a differently-worded prompt, and confabulated over the
> char-cap truncation boundary. Plan C's version is *harder*, not easier: it
> rewrites the artifact retrieval already depends on, where E1 only added a new
> one. **Do not run the episode-prompt version without first clearing
> `benchmarks/fact_probe.py`'s faithfulness bar** (stratified sample,
> correct-answer control, ≥0.90) on episode rewrites specifically. The
> granularity motivation below (BEAM EO/SUM post-mortem, rate-distortion
> framing) still stands — the *diagnosis* was never in question, only the
> generative remedy. **RAPTOR flip-watch constraint DISCHARGED 2026-08-26**
> (G-FLIP PASS 7/7, `aggregation_nodes_enabled` flipped True). The
> faithfulness constraint above is unchanged and still binding. One residual
> ordering rule survives the discharge: this plan rewrites episode membership,
> so it must not OVERLAP the post-flip verification dream — schedule it as its
> own effort and re-verify aggregation reuse once after it lands.
>
> > **STATUS 2026-08-30 — the MODEL-level concern is discharged; the
> > EPISODE-REWRITE bar is not, and the difference is the whole point of how it
> > was worded.** G-F1 re-ran on v4-flash against full sources and passed at
> > **faithfulness 1.00 (98/98)**, which settles what the paragraph above was
> > actually afraid of: the 0.55-0.76 that made this warning urgent was an
> > instrument artifact, the "1 invented claim per 3-4 items" never happened,
> > and the confabulation-over-the-truncation-boundary mode was the recorder's,
> > not the model's.
> >
> > But the bar as banked reads "on episode rewrites **specifically**", and that
> > run measured the NARRATIVE-FACTS extractor — `fact_probe.py` is G-F1, and
> > the four criteria reported (gold in <= 5 facts, faithfulness, median
> > facts/session, control median) are G-F1's own. Facts extraction and episode
> > re-cutting are different generative tasks against the same turns, and this
> > plan's text already argues its own is HARDER: it rewrites the artifact
> > retrieval depends on, where E1 only added a new one. Accepting a facts
> > result as the episode result is the reuse-another-benchmark's-driver trap
> > that has now cost this project three times (MSC parity, deixis,
> > answerability).
> >
> > **The bar is also unmeasurable as written, and that is a defect in the
> > banked text rather than a reason to waive it.** There is no episode-rewrite
> > prompt to score — Plan C is unbuilt — so "clear the bar before you run it"
> > is circular. **Resolution: Plan C proceeds PROBE-FIRST**, the pattern this
> > repo already uses (Idea A/B both shipped core-mechanism-first, default OFF).
> > Build the episode-rewrite prompt and a faithfulness probe over ITS output;
> > measure >= 0.90 on a stratified sample with a correct-answer control on
> > EPISODE REWRITES; only then does any default change become discussable.
> > Nothing ships on the strength of the facts number.
> >
> > What the facts result legitimately buys: the prior is now "this model is
> > faithful when asked to produce self-contained items from session turns",
> > where before it was "this model invents at 1-in-3.5". That changes the
> > expected value of building the probe enough to justify building it. It does
> > not pre-score it.
> >
> > **BUILT 2026-08-30 (stage 1, probe-first, default OFF; UNCOMMITTED — no
> > benchmark or LLM spend was incurred).** What exists now:
> >
> > * `SESSION_DIGEST_GRANULAR_SYSTEM` / `_USER_TEMPLATE` — a SECOND digest
> >   prompt beside the shipping one (which is untouched, byte for byte),
> >   pinned by `EPISODE_GRANULAR_PROMPT_VERSION = "episodes.granular.v1"` in
> >   `hymem/dreaming/digest.py`. Its user closer is deliberately unique so
> >   stubs, probes and call counts can tell the arms apart.
> > * `episode_granularity_enabled = False` and
> >   `dream_max_episodes_per_session = 12` (a runaway bound, NOT the 3-8
> >   target — the cap applies to the granular arm only, since capping the
> >   shipping prompt would be a default change).
> > * Schema **v35**: `sessions.episodes_prompt_version`, the v19 stamp
> >   pattern. NULL = the shipping blob prompt, which is what every pre-v35
> >   store already reads, so an untouched store never re-extracts; flipping
> >   the flag re-digests each session ONCE and then returns to zero tail
> >   calls; flipping it back re-digests once under the blob prompt.
> > * Two persist changes the granularity forces: the episode id carries a
> >   title hash on the granular arm (several decisions of one session
> >   legitimately cite the same chunk, so they share a message range and would
> >   otherwise collide onto ONE row), and the re-read window is SUPERSEDED on
> >   either side of a granularity CHANGE (UPSERT refreshes only matching
> >   ranges, so rows written under the other id shape would otherwise survive
> >   beside the new ones). The second was built granular-arm-only and CORRECTED
> >   2026-08-31 — see the revert-hole note below.
> > * `benchmarks/episode_probe.py` (G-EP1) + 31 offline tests
> >   (`tests/test_episode_granularity.py`, `tests/test_episode_probe.py`).
> >
> > **G-EP1, pre-registered here before any run.** FLIP-DISCUSSABLE iff all of:
> > (1) faithfulness hand-score >= 0.90 on the stratified sample; (2) median
> > episodes per SUBSTANTIVE session in [3, 8] on the target arm; (3) >= 60% of
> > episodes carry a concrete value in the summary; (4) >= 90% session
> > coverage; (5) the correct-answer control median <=
> > `dream_max_episodes_per_session`. Above a 2% parse-failure rate the run is
> > INCOMPLETE, never FAIL (truncation biases the criteria in opposite
> > directions — the G-F1b ceiling, same reasoning). Criteria 2-4 are
> > mechanical and the probe computes them; criterion 1 is a hand-read and the
> > probe reports INCOMPLETE until `--faithfulness` supplies it, so a subset of
> > criteria can never print PASS. A PASS makes the flip DISCUSSABLE, not
> > automatic: the LME full guard (non-regression only), the dream cost watch
> > and the no-overlap rule against a RAPTOR verification dream are separate
> > and still stand.
> >
> > Two instrument decisions worth banking. The probe scores the CHUNK-shaped
> > digest input, not raw turns, because that is the corpus the feature reads —
> > scoring the convenient shape is the mistake `dreaming/facts.py` documents
> > from the other side. And the dump records the LITERAL prompt string the
> > extractor sent, hashed at send time and re-hashed on read
> > (`assert_full_source`), so the `[:4000]` class of defect is a hard failure
> > rather than an invisible one.
>
> > **CORRECTION 2026-08-31 — the flip contract had an asymmetric hole, found
> > box-side by reading the code against its own docstring.** `supersede_window`
> > was passed only when `episode_granularity_enabled` is True, so flipping ON
> > was clean and flipping OFF was not: the blob rewrite writes bare-range ids,
> > the granular rows it replaces carry `range#titlehash`, the two shapes cannot
> > collide, UPSERT refreshes nothing, and the store is left serving BOTH
> > granularities of the same conversation — the exact mixed store the
> > supersession exists to prevent, in the direction nobody looked. The config
> > docstring meanwhile asserted that the revert superseded. Damage was nil (the
> > flag has never been on anywhere), but a false claim in a docstring is how a
> > property gets banked as verified.
> >
> > Fixed in `f256443`, keyed on the session's STAMP rather than on the flag.
> > That distinction is the substance of the fix, not a detail: passing the
> > window unconditionally would also have closed the hole, and would have made
> > every blob re-dream destructive-in-window on stores that never opted in — a
> > silent default change smuggled in under a default-OFF feature, which is the
> > thing `validate_episode_items(max_items=None)` was written two files over to
> > avoid. A store that has only ever run the shipping prompt reads NULL, passes
> > no window, and keeps the additive re-dream it has always had.
> >
> > Two further stale claims in the same docstring, both inherited from the
> > pre-backout brief: episodes are stamped on the SESSION only (one column; a
> > per-row copy was dropped as derived state that can disagree with its source,
> > per migration 035), and the revert has two documented gaps, both in the
> > over-KEEPING direction — an empty blob reply supersedes nothing (an empty
> > extraction must never delete a previous one's work) and a granular row
> > reaching past the re-read window lies outside it and survives.
> >
> > **Process finding, and it is the reusable part.** The pre-existing revert
> > test asserted the prompt and the stamp; neither can see a surviving row, and
> > the file's own header lists five things it pins, of which "the store does not
> > end up mixed" is not one. A test suite written from a feature's INTENT
> > covers the direction the feature is meant to run. The two new tests are
> > mutation-checked in BOTH directions — the row-level one fails against the old
> > wiring, the wiring one fails against the old wiring AND against an
> > unconditional window, which is what pins the inertness leg rather than
> > asserting it. Offline only; no spend. Suite 1331 passed (+2), same 8
> > `sqlite_vec`-absent failures on the verifying interpreter, 0 on the box.
>
> **STATUS 2026-08-31 — G-EP1 RAN: PASS.** `episode_probe.py`, 20 sessions/arm
> (40 extraction calls, seed 0, 23 miss + 23 control questions, 10 gold-bearing
> / 10 filler per arm), model deepseek-v4-flash @ api.deepseek.com/v1, source
> `longmemeval-v2-hymem-20260607T164031Z-seed0.json` (the G-F1 instrumented
> run, so the selection is the banked readside §2.1 rule and the arms match
> G-F1's question set), dataset `longmemeval_s_cleaned.json`. All five criteria:
> faithfulness **0.98** (67/68 episodes strict, hand-read counts-only against
> the hash-verified `extractor_input`), median episodes/substantive session
> **4.0** (in 3-8), concrete-value share **68%** (≥60%), coverage **100%**
> (≥90%), control median **4.0** (≤12). Parse failures **0**. Verdict: PASS —
> the flip is now DISCUSSABLE, not automatic (LME full guard, dream cost watch
> and the no-overlap rule against a RAPTOR verification dream are separate and
> still stand).
>
> Two findings from the run, both instrument-level.
>
> **The first 20/arm run was UNREADABLE — and the probe's own diagnostic was
> wrong about why.** 21/40 parse failures (52.5% > 2% ceiling → INCOMPLETE as
> designed). The probe advised "re-run at a higher --max-tokens"; the true
> cause was that `deepseek-v4-flash` is a REASONING model: as called without
> `extra_body`, it burns the full 8192-token output budget on
> `reasoning_content` and returns `content==""` (finish=length, content_len=0,
> reasoning_len≈32k), which `loads_lenient` turns into a parse failure. The
> adapter's own docstring already documents the fix — post-2026-07-24 the
> provider prepends thinking tokens unless the request carries
> `{"thinking":{"type":"disabled"}}` (longmemeval_adapter.py:327-330) — and
> `episode_probe.py` supports it via `--extra-body`; the first invocation
> simply didn't pass it. Re-run with the flag: 0 parse failures. Higher
> --max-tokens would have been the wrong remedy (it re-spends the budget on
> reasoning and can still end length-stopped with empty content).
>
> **The parse-failure row discards the raw response, so a run can fail 52%
> without one shard of evidence about the failure class.** The dump stores
> `extractor_input` (what was sent) but nothing of what came back; the
> diagnostic had to be inferred and was wrong. One line — record the raw
> response's `finish_reason` and length (or a fixed-size tail) on parse
> failure only — would have made this diagnosis immediate. **Suggested, not
> applied** (attn: user gate — box spend and instrument edits are both the
> user's call; the probe is unchanged).
>
> **Disagreement between G-MON and G-MON-b, live and resolved by design, not by
> waiver.** The G-MON first read (2026-08-27) banks a family-routing rule: a
> below-bar append with `leaf_changed=1` goes to the family question rather
> than being counted as a regression — and G-MON-b's literal bar (amp ≤ 4.5)
> would fire on exactly those rows (family amps 6.3-7.3 exceed the bar by
> construction). #1336 (2026-08-30 16:36 UTC, post-flip, leaf_changed=1, reuse
> 77.7%) is such a row: below-bar append, amp 7.7, residual 0. Read under G-MON
> it is family-routed (no regression finding); read under G-MON-b's literal
> bar it fires. The banked text anticipated this: "if the two disagree on a
> future row, that disagreement is itself the datum." The record: the family
> row is not a regression finding, and G-LD1 — scored on this same row, first
> leaf_changed=1 carrying a non-NULL delta (add 6 / rem 6, rollup 19) —
> CONFIRMED the LARGE leaf-delta branch (rollup≥15 → added+removed≥5; observed
> 12≥5; falsifier not fired). So the disagreement resolved in the direction
> the family instrument predicted, and the G-LD1 ladder held its first
> pre-dated test.
>
> **The transferred pre-dated family test also already CONFIRMED — on #1327,
> before the transfer was written.** "First leaf_changed=1 row stamped after
> 2026-08-27 09:43:38 UTC" resolves to #1327 (08-27 10:09:21 UTC): level0=3
> (=l0miss, exact), rollup=18 (predicted 15-18), root=1, self-check 3+18+1=22
> = built-reused, residual 0 → reading (a), benign digest cascade, family
> CLOSED. The falsifier (level0≈17) never fired on any row. Status honesty:
> #1327 predates the transfer commit, so this is a blind-status confirmation
> (the doc's knowledge stops at #1324 — corroborating trace), not full
> pre-dated strength; the family question did not need #1336 or #1341 to
> close. #1334 (v34's first post-deploy dream) reports NULL deltas by contract
> and #1336 is the first scorable row — so G-LD1's first pre-dated test is
> #1336 at full strength (banked 15:04:21 UTC in d3f6178, scored 16:36:34
> UTC).
>
> **Faithfulness hand-score detail (the 0.98, not 1.00).** 68 episodes in the
> sample; 67 fully grounded. The one exception: the grocery-expense session
> (9aaed6a3__answer_353d3c6d_1) — episode "Add skincare expense at Sephora"
> carries "$227.99 and total expenses to $302.99" in its summary, and those two
> numbers are NOT literally in the extractor input: the input's last turn is
> the user's "$50 at Sephora", and the assistant's reply with the running
> totals is past the chunk boundary. The extractor COMPUTED the totals from
> grounded operands ($177.99+$50=$227.99; $252.99+$50=$302.99 — both
> arithmetic steps the assistant itself performs two turns earlier), and the
> arithmetic is correct. Classified: derived-and-correct, NOT confabulation.
> The strict spelling of the bar ("every name, number, date and version must
> appear in the input") counts it as a miss, so it is scored at 0.98 rather
> than 1.00 — the conservative reading; even at 1.00 the verdict is the same
> (≥0.90).
>
> **The instrument fix is now APPLIED (`b7748c1`), and it is the same one
> the run above suggested.** `CapturingLLM` records `reply_chars` (the true
> full length) plus a bounded, explicitly-diagnostic `reply_head`; the dump
> row carries both and the backend error; `summarize` splits the failures
> into empty vs non-empty; and `report` picks the remedy from the recording
> instead of from an assumption, saying "cause UNKNOWN" outright when the
> failing replies were not recorded. A note also prints BEFORE the spend
> whenever `--extra-body` is empty, which is the cheapest place to catch the
> invocation that started this. Two tests, mutation-checked in both
> directions: dropping the recording fails one, counting recorded replies
> over all rows fails the other. Offline only, no spend. Suite 1333 passed
> (+2), same 8 `sqlite_vec`-absent failures on the verifying interpreter.
>
> One defect found while checking the fix against its own output, and it is
> the reusable part: the first cut counted recorded replies over ALL rows, so
> a single recorded SUCCESS made a set of unrecorded FAILURES look diagnosed
> and the report printed a confident wrong cause. That is the E3 shape — a
> diagnostic that returns a constant when broken — reappearing inside the fix
> for a diagnostic that returned a constant when broken. Both counts are now
> over the FAILING rows only, and the mixed case is pinned by a test.
>
> **Second finding, NOT acted on.** `longmemeval_adapter._call` raises on
> `content is None` but passes `content == ""` straight through as data. The
> probe only noticed because a parse failure is countable there; on the
> canonical LoCoMo/LME/MSC paths the same empty reply becomes an empty answer
> and is scored as a WRONG one rather than counted as a failure. Deliberately
> left alone: widening that guard changes retry behaviour in shared scoring
> code mid-campaign, which needs its own reason and its own pre-registration,
> not a drive-by on the way past. Recorded here so it is not rediscovered as
> a mystery — and it bounds how far the v4-flash migration story can be
> called settled.
>
> **LEVER BUILT 2026-08-31 (`--episode-granularity`), and the pass that built it
> found a live contamination in the same adapter.** The flip's own LME
> non-regression guard could not be executed: `episode_granularity_enabled`
> appeared nowhere under `benchmarks/`, so the granular arm was unreachable from
> the CLI and the guard would have run the blob prompt in both arms — a clean
> null produced by an instrument that never touched the lever, which is the
> unreachable-code-path shape from the diagnostic-controls memo. The flag is
> write-side: it changes what the dream extracts, so the arm MUST be built with
> `--fresh` and without `--no-dream`, and a run reusing a store dreamt under the
> other prompt measures the store, not the lever. The help text says so.
>
> **The contamination: `benchmarks/longmemeval_adapter.py` never pinned
> `aggregation_nodes_enabled`.** It set only the True leg, so when the library
> default flipped False -> True on 2026-08-26 this adapter silently gained the
> aggregation layer + digest — while every run through it was still being
> compared to a 68.4 baseline that ran without them. `beam_adapter` and
> `msc_adapter` were pinned that day (`52adfe5`, `4853f08`); this one was
> missed, and the banked record ("the three benchmark adapters were silent
> default-config consumers and now pin it False") asserted a state that was true
> of two files out of three. The doc was checked against itself rather than
> against the code, which is the same failure the `4853f08` write-up called a
> doc/code form mismatch — caught there, repeated here on the file that carries
> the frozen baseline.
>
> Both levers are now pinned in BOTH positions, unconditionally, and three tests
> assert it. The aggregation test deliberately also asserts that the LIBRARY
> default is True: that is what makes "adapter reads False" evidence of a pin
> rather than of an inherited default, and if the default ever moves back the
> test SHOULD fail rather than keep reading green while testing nothing. Both
> mutation-checked — restoring the conditional form fails one, dropping the
> granularity override fails the other. Suite 1336 passed (+3 on this
> interpreter; the new file skips where `requests` is absent and runs on the
> box).
>
> **Consequence for the guard, and it is not small.** Any LME run through this
> adapter dated after 2026-08-26 was an aggregation-ON run under an
> aggregation-OFF label. That is a comparability question about those runs, not
> a finding about the layer, and it should be settled by checking dates before
> any of them is used as a baseline for the flip.

> **SETTLED 2026-09-01 — the contamination has NO victims, and the check that
> says so is now in the ledger rather than in this paragraph.** The window is
> bounded by two commits: `52adfe5` (2026-08-26T16:26:57Z, library default
> False -> True) and `2247074` (2026-08-30T20:50:00Z, the adapter pins the
> lever both ways). **Zero of the 74 rows in `lme_runs.db` overlap it.** Every
> LME run either predates the flip or postdates the pin; nothing in the ledger
> was silently aggregation-ON.
>
> The two rows that could have been are the flip's own guard arms, and they
> clear the pin by **26 minutes** — which is only knowable because the check
> does not use `run_date`. That stamp is written when the archive is, i.e. at
> the END of the run; the version of the code that executed is decided at the
> START. `guard-epg-off` ends 2026-08-30T23:59:24Z after 9,807s, so it started
> 21:15:56Z — 26 minutes after `2247074`. (`guard-epg-on` started 00:04:33Z,
> five minutes after the OFF arm finished, which independently confirms the
> stamp is an end stamp: read as start stamps the two arms would overlap.) A
> check keyed on `run_date` would have returned the same verdict here for the
> wrong reason, and the wrong verdict on the first run that straddles a commit.
>
> **`lme_registry.py audit`** is the build. It exists because
> `aggregation_nodes_enabled = 0` in this DB is three claims wearing one value
> — RECORDED by the run's config block, ASSERTED afterwards via `--set`, or
> ABSENT — and only the first is a measurement. The other two state intent, and
> inside the window the code did not follow intent. All three are therefore
> CONTRADICTED in-window, ABSENT included: NULL there is not missing
> information, because the library default supplies it, and reporting UNKNOWN
> would leave a known aggregation-ON run available as an OFF baseline on the
> grounds that nobody wrote it down. Exit 1 on any contradicted row.
>
> **The empty window is reported as a count, not as silence** — "runs whose
> execution overlapped that window: 0". A clean audit and an audit that is not
> wired up print the same thing otherwise, which is the §10.3 shape from the
> gold-delta protocol. For the same reason the 15 tests are led by the ones
> that put a row INSIDE the window and require it to fire; four mutations were
> checked and each is caught by its intended test (collapse the window to a
> point: 7 fail; key the test on the end stamp: the straddling-run test fails;
> read an in-window ABSENT as UNKNOWN: 1; let an analyst-set on any column
> taint every column: 1).
>
> **The adapter-count claim is also now checked against the code rather than
> against itself**, which is what the paragraph above says was never done.
> `beam_adapter.py:1125` and `msc_adapter.py:271` both pin unconditionally;
> `longmemeval_adapter.py:545` does since `2247074`. The fourth entry point,
> `locomo_adapter.py`, contains no mention of the lever — and is NOT a fourth
> gap: it constructs `MSCAdapter` (`locomo_adapter.py:513`) and inherits that
> pin. All LoCoMo runs are covered by the MSC pin. Offline, no spend. Suite
> 1644 passed (+15).

> **STATUS 2026-09-01 — the LME full guard HAS ALREADY BEEN RUN, and this
> document does not record it.** Everything above prices the guard as an
> unspent ~1,000-reader-call, 2-3h job. Both arms are on the box and have been
> since 2026-08-31: `guard-epg-off-20260830T235924Z.json` (500 answer + 500
> judge calls, 9,807s) and `guard-epg-on-20260831T031130Z.json` (11,216s),
> sequential, deepseek-v4-flash both sides, `sample=0` full 500.
>
> | | OVERALL | KU | MS | SSA | SSP | SSU | TR |
> |---|---|---|---|---|---|---|---|
> | OFF | **71.0** | 74.4 | 53.4 | 67.9 | 73.3 | 94.3 | 75.2 |
> | ON | **71.0** | 73.1 | 54.1 | 66.1 | 83.3 | 97.1 | 72.2 |
>
> Against gate 4 as pre-registered ("LME full guard as NON-REGRESSION only
> (canonical 70.0 full-dream, MS floor 51.9). Not a tuning signal.") the ON arm
> clears both: 71.0 >= 70.0 and MS 54.1 >= 51.9. Comparability to the canonical
> is established rather than assumed — the audit above puts both arms
> post-`2247074`, aggregation pinned OFF, which is the same layer state as the
> pre-flip baseline.
>
> **And the gate is NOT discharged, because the pair cannot evidence its own
> contrast.** Both arms ran BEFORE `6543ee6` (2026-08-31T06:48:34Z) taught the
> adapter to write the levers into the config block. The two blocks are
> byte-identical except `elapsed_s` and `total_tokens`; the registry's 0/1 is an
> analyst `--set`; the stems are what the operator remembers typing. Nothing
> inside either file says which arm it is.
>
> Everything else in the artifacts was checked and none of it separates the
> readings:
>
> * `num_memories` differs on 120/500 rows — but it is **capped at 45 and
>   saturated on 352/500 rows in BOTH arms**, so on 70% of the sample it cannot
>   move however many episodes the dream cut. The 120 differing rows are
>   symmetric (51 up, 69 down, mean −0.05, z = −0.62), which is what two
>   `--fresh` re-dreams of ONE prompt also look like.
> * `num_sessions`, `num_messages`, `recall_ceiling`, `n_facts`,
>   `gold_in_facts`, `detected_ability`: identical on all 500 by construction.
> * 331/500 answers differ and 52 questions flip — reader/judge churn at this
>   model's known rate, and equally consistent with either reading.
>
> **71.0 vs 71.0 is what a real null looks like. It is also what two runs of the
> same arm look like.** The artifacts cannot tell them apart, so this pair
> cannot discharge a non-regression gate on the granularity lever however clean
> the numbers are. That is the unreachable-code-path shape from
> `docs/diagnostic_controls.md` one level up: not an instrument that never
> touched the lever, but a pair of results that cannot show whether it did —
> the precise hazard named when `--episode-granularity` was built ("a clean null
> produced by an instrument that never touched the lever").
>
> **Built, so the next pair is checked before its scores are read:**
> `run_registry.arm_evidence()` (shared core, so BEAM and LoCoMo get it too)
> and `lme_registry.py arms A.json B.json --lever K`. Three outcomes —
> EVIDENCED (both blocks record the lever and differ), SAME_ARM (both record
> it and agree), UNEVIDENCED (either block is silent) — plus the confounds, the
> other keys that also moved, because an evidenced contrast is still not a clean
> one. It reads artifacts only: a stem or a `--set` is the claim under test, not
> evidence for it. Run on the real pair it returns UNEVIDENCED and exit 1.
> 11 tests, 3 mutations checked (treat an absent lever as a difference: 4 fail;
> report SAME_ARM as EVIDENCED: 1; stop ignoring the timing keys: 1).
>
> **What this costs to fix is a re-run, and that is the user's call, not this
> document's.** Post-`6543ee6` the config block records the lever, so a repeat
> pair would be self-evidencing. The banked reading until then: the guard's
> numbers are recorded above and are *encouraging* — nothing suggests a
> regression — but gate 4 stays OPEN, and the flip still needs it. Offline, no
> spend. Suite 1655 passed (+11).

> **STATUS 2026-09-01 — gate 3 RAN, free, on both corpora, and it cannot be
> moved by this flip.** `episode_coverage_probe.py` is read-only and LLM-less,
> so the "before" half of gate 3 cost nothing. It was never going to say what
> the gate expected.
>
> | store | sessions | with ≥1 episode | coverage | never_dreamed | dreamed_zero_short | dreamed_zero_long |
> |---|---|---|---|---|---|---|
> | prod `~/.hermes/hymem.sqlite` | 110 | 86 | **78.2%** | 9 | 15 | **0** |
> | 10 surviving LME stores | 476 | 105 | **22.1%** | dominant (97.3% on the store audited in full) | 1 | **0** |
>
> **`dreamed_zero_long` is ZERO on both.** That bucket is the extraction-recall
> problem — a substantial session the extractor read and returned nothing for —
> and it is the only one a better episode prompt could fix. It has no
> population. On prod every uncovered session is either a test stub
> (`final-upload-test`, `ws:smoke-test`, `verify-redact`, …) or literally empty:
> all 9 `never_dreamed` sessions hold **zero messages**, so no prompt and no
> scheduler can create an episode from them.
>
> **On LME the gap is real and it is not the prompt's.** 371 of 476 sessions —
> holding **3,902 of 4,986 messages** — carry no digest at all. The reason is in
> `dream_runs`: exactly **one cycle** per store, `chunks_processed 29 / 56`,
> `sessions_processed 14`. HyMem's dream is incremental by design — each cycle
> spends a chunk budget (`dream_budget = 50`, `dream_baseline_budget = 10`) on
> the most salient candidates and leaves the rest for the next cycle. Every
> benchmark adapter bulk-ingests a haystack and then dreams **once**. The
> undigested tail is the designed shape of that harness, not a runner bug, and
> re-cutting the episode prompt cannot reach a session the dream never opened.
>
> **This is the larger finding, and it is not about Plan C.** Every
> episode-, digest- and aggregation-dependent result ever measured on LME was
> measured over roughly **a fifth of the corpus**. It also bounds the guard
> above independently of the arm-evidence problem: even had the ON arm
> demonstrably run granular, the lever could only touch the ~22% of sessions
> that were digested at all, so 71.0 vs 71.0 is the expected reading rather
> than an informative null. Recorded here, not acted on — widening the dream
> budget or dreaming to convergence changes what every banked LME number means
> and needs its own pre-registration, not a drive-by.
>
> **Built:** the probe now reads `dream_runs` and prints which of the two
> readings of `never_dreamed` the evidence supports — NOT A GAP (every such
> session is empty; the prod answer), ONE CYCLE OVER A BULK INGEST (the LME
> answer), RUNNER GAP (many cycles and content still undigested; the
> mechanical fix the verdict guide currently offers as the only reading),
> NEVER RAN, or UNAVAILABLE on a store predating the table. It deliberately
> reports **no chunk ratio**: `chunks_seen` is the candidate pool re-counted
> every cycle, so on prod the sum reads `425/330255` and
> `processed/seen` looks like a coverage fraction while being nothing of the
> kind. A caveat does not survive being quoted, so the number is not produced.
>
> One process note worth keeping. The first cut of the test pinning that
> absence ran against the `probe_store` fixture, which has never dreamed —
> `dream_history` returns None there, the assertions were skipped, and the test
> **passed against the mutation that put the ratio back**. It was caught by
> mutation-checking rather than by review: a vacuous test for a vacuity check,
> found the only way that shape ever is. Four mutations now, each caught.
> Offline, no spend. Suite 1667 passed (+12).

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
   rise from the ~42% gap. **RAN 2026-09-01 (free): the gate cannot move.**
   `dreamed_zero_long` — the only bucket a better episode prompt can fix — is
   ZERO on prod and on all ten surviving LME stores. See the STATUS 2026-09-01
   block above.
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
3. **Plan C** independently of A/B. The RAPTOR flip decision it waited on
   RESOLVED 2026-08-26 (PASS → flipped), so the gate is discharged; what
   remains is the faithfulness bar and the no-overlap rule against the
   post-flip verification dream (see its sequencing constraint).

---

## Campaign E — the narrative-facts roadmap (added 2026-07-30)

*Origin: the 2026-07-30 competitive review (HyMem vs Hindsight/Honcho), integrated
with its own critique the same day. Supersedes nothing above; Plan C was briefly
subsumed by Step 4 (E1 build) and UN-subsumed the same day when G-F1 cancelled it.
Same standing contract as every plan in this
file: front-run gate before any build, additive-only, mechanism > score, nothing
reads oracle labels, per-category LME deltas under ~±5pp are noise.*

### Scoreboard — state as of 2026-08-25 (read this first; the per-step blocks below are the chronological record)

| Item | State | Verdict in one line |
|---|---|---|
| **E1 narrative facts** | **AUTHORITATIVE LIFECYCLE UPGRADE 2026-09-04 (schema v46); historical Step 5 result retained** | The v26 append-only/range-only design was replaced by exact lossless occurrence manifests, durable empty/failure/retry outcomes, revisioned retract/resurrect lifecycle, prompt/config replay on immutable source-unit boundaries, active-authority-only FTS/vector search, and v10 portability. Extraction and retrieval now both default on so paid work is visible; the old benchmark result remains evidence for future empirical gating, not a reason to ship a hidden write-only tier. |
| **E5 anaphora** | **SHIPPED ✓** | `hymem/query/coref.py`, on by default; 31/31 resolution, 0/12 no-harm. The hedge that paid. |
| **E3 rerank A/B (M1+M2)** | **RUN 2026-07-30/31 — VERDICT IN, NOTHING FLIPPED** | **M1 FAIL on latency arithmetic** (CE is 10.7× *slower* than the API on CPU vs a required ≥10× faster; ~9.5ms/candidate is unreachable — mxbai 108× off, bge 37× off) and its quality row is **unmeasured**, not parity (it ran on the vacuous handset). **M2 = parity, not a bge win** (NL R@1 +4/20 at p≥0.125, EN −2/15; fails the pre-registered effect size once rescaled to n=20). bge's real edge is latency (3–4× on CPU, language-flat). **Decision: keep both defaults as they are** — Step 6 closes unexercised. |
| **E6 supersession over facts** | **NARROW SOURCE-UNIT LIFECYCLE BUILT IN v46; CROSS-SOURCE E6 NOT BUILT** | Successful replay authoritatively replaces only the same exact extraction unit: omissions retract and later identical payloads resurrect deterministically, with immutable revision/lifecycle history. The old typed-value/date heuristic across distinct sessions is explicitly rejected as unsafe and remains unimplemented; simultaneous contradictory facts from different source units can coexist, and Step 14 must gate the default-on tier's distraction risk empirically. |
| **E4 temporal boost (query-side)** | **NOT BUILT — closed by decomposition, then by selectivity arithmetic** | LME gate technically PASSED (8.8% / 90.9% / 0) but the same rules fail across corpora on the speech-time/event-time axis, and selective fire rate lands under criterion 1 on both. **Carry-forward `G-E4b` (ingest-side `valid_at`) PRE-REGISTERED 2026-07-31 — and its Step 0 pre-check closes the query-range consumer for free: ceiling ≈0.22% of queries (LME), ~20× under the bar.** The only live path is Fork B (an existing `valid_at` reader — supersession / recency-dating), which is a different feature and must be argued on its own terms. |
| **E2 per-entity observations** *(Campaign E's E2 — not Grove's recovery gauge)* | **DORMANT (still double-blocked, but one blocker is now moving)** | Needs flip-watch green AND a new faithfulness result clearing `fact_probe.py`'s bar. The flip-watch leg is no longer *untestable*: schema v30 (persisted leaf watermark) and v31 (structural rebuild forecast) closed the observability holes, criterion 6 (keying integrity) was banked 2026-08-09, and the first post-v31 dream `#1182` read clean (predicted=9, residual=0, reuse 91.7%). The gate is now **PENDING on accrual** — it needs every verdict row to carry a populated `aggregation_keying_residual` — not blocked on a defect. The faithfulness leg is untouched. |
| **E7 usage feedback** | **OPEN, ungated** | Artifact-agnostic long game; no front-run designed yet. |

**Campaign E is closed as a scored campaign.** Every scored item is either
shipped (E5), run-and-verdicted (E1 Step 5, E3), closed by argument (E4),
dormant on a named blocker (Campaign-E E2, E7), or rejected as unsafe
(cross-source E6). Schema v46's
same-unit replay lifecycle is not evidence that cross-source contradictions are
resolved; the default-on retrieval change remains subject to Step 14's empirical
non-regression gate.

**Next actions, in order** (2026-08-25, superseding the 2026-07-30 list): the live
work is now Plan D + Grove E2, sequenced in that plan's own section below.
(1) ~~run `benchmarks/recovery_probe.py` read-only on the box **and** on a LoCoMo
`--db-dir` store~~ — **DONE 2026-08-25, RE-RUN 2026-08-27.** `anchor_delta = 0`
on both stores both times, but the second run found the probe's premise broken by
`8c6925c`; the Plan D licence stands **re-based onto a selectivity argument** and
now carries a tripwire. Read the STATUS 2026-08-27 block before starting (2), and
carry the selectivity framing — not "the clause is inert" — into Plan D's text;
~~(2) build Plan D behind a default-OFF flag and run the shadow
probe over both corpora; (3) if C1 proceeds, the scored LoCoMo A/B under the
OFF-arm stratification specced in Plan D.~~ **STEPS 2-3 FORECLOSED 2026-08-26:
the shadow probe RAN and Plan D closed C1 FAIL-mechanism** (0.0-1.35% against a
5% bar; the separate-cap legs reject H0 at p = 0.022). There is no A/B to run.
Grove E1 is **deferred** — see the note in the Plan E section.

**Next actions, in order (2026-08-31, superseding the 2026-08-27 list).** Items
1 and 3 below are now closed or blocked, which left **G-EP1 as the only live
item that could move anything** — everything else on this list is passive
accrual or foreclosed.

**AMENDED, same day — G-EP1 RAN and PASSED, so this list is now empty of live
items.** Its first invocation aborted at 52.5% parse failures (no
`--extra-body '{"thinking":{"type":"disabled"}}'`, a reasoning model returning
empty content) and read INCOMPLETE by the ceiling, exactly as designed: it never
scored, so re-running it was the same pre-registration and not a revival. The
re-run with the flag returned 0 parse failures and PASS on all five criteria
(faithfulness 0.98). The instrument now records what came BACK, so the next run
that fails this way names its own cause rather than being attributed to
truncation by inference. What that PASS buys is bounded and unchanged: the flip
is DISCUSSABLE, not automatic — the LME full guard (non-regression only), the
dream cost watch, and the no-overlap rule against a RAPTOR verification dream
are separate gates and all three still stand. **The next decision is the FLIP
itself, and it needs those three discharged, not a fourth probe.**

**Gate status 2026-08-31 (box-side read).** No-overlap: DISCHARGED (#1317 is the
banked RAPTOR post-flip verification dream; a post-flip reuse re-verify is still
owed AFTER the granularity flip lands). Dream cost watch: BASELINE READY (dreams
1342/1343 at 47s/58s wall clock, input 1167→1168, reuse 91-100%, 0 fusion
failures) — the watch is one-time re-digest cost plus steady-state delta,
measurable before/after, free. LME full guard: was BLOCKED ON A MISSING LEVER,
then UNBLOCKED — `--episode-granularity` exists as of this entry. **Superseded
2026-09-01: the guard was RUN on 2026-08-30/31 (both arms, 71.0 vs 71.0) and
the gate is still OPEN, because the pair cannot evidence which arm was which.
See the STATUS 2026-09-01 block in Plan C.** The guard is
~1,000 reader calls plus dream calls, ~2-3h, and both arms must be `--fresh`
because the lever is write-side. **Before it spends, check the aggregation
contamination above: the OFF arm must be built by the same pinned adapter, and
any pre-existing post-2026-08-26 run must not be reused as its baseline.**

**Gate status 2026-09-01 (supersedes the block above).** Three of the flip's
five gates are now settled and two need spend, so the flip is decidable only
behind a spend decision that is the user's to make.

| gate | state |
|---|---|
| 1. mechanical pytest | **PASS** — 31 offline tests, unchanged |
| 2. qualitative box gate (G-EP1) | **PASS** — faithfulness 0.98, all five criteria |
| 3. `episode_coverage_probe` before/after | **RETIRED — no population.** `dreamed_zero_long` is 0 on prod and on all ten LME stores; the bucket a better episode prompt could fix does not exist. The gate cannot discriminate and must not be counted as passed |
| 4. LME full guard (non-regression) | **OPEN.** Already RUN 2026-08-30/31 (71.0 vs 71.0, clearing 70.0 / MS 51.9) but UNEVIDENCED — the arms cannot show which arm they were. Post-`6543ee6` a repeat pair is self-evidencing; that is ~1,000 reader calls |
| 5. dream cost watch | **OPEN, needs one dream.** Baseline banked (dreams 1342/1343, 47s/58s, reuse 91-100%); the "after" leg is a re-digest under the granular prompt, which is LLM spend |
| no-overlap ordering rule | **DISCHARGED** (#1317); a post-flip reuse re-verify is still owed AFTER the flip lands |

**Gate 3's retirement is the one that changes the argument, not just the
scoreboard.** It was the only gate that would have shown the flip DOING
something rather than not breaking something — 2 and 4 are a faithfulness bar
and a non-regression bar, both satisfiable by a change with no effect at all.
With 3 retired, nothing on this list can distinguish "granularity helps" from
"granularity is inert", and the LME coverage finding above says the lever can
reach at most ~22% of that corpus anyway. **A flip decision taken on gates
1/2/4/5 alone would be a decision that the feature is harmless, not that it is
worth having**, and this plan's own standing contract is mechanism > score.
Whoever takes it next should either design a gate with a live population or
flip it on the honest ground that it is a shape improvement whose benchmark
effect is below what this harness can resolve.

1. ~~**RUN, box-side:** `benchmarks/fact_probe.py` faithfulness.~~ **RAN
   2026-08-30 — G-F1 PASS on v4-flash at full source, faithfulness 1.00
   (98/98).** It closed the `[:4000]` open item outright (see the G-F1b block)
   but did NOT clear Plan C's bar, which is banked "on episode rewrites
   specifically" and measured the facts extractor instead. Plan C proceeds
   probe-first; see its STATUS 2026-08-30 block. The no-overlap constraint was
   satisfied by construction (#1317 was already banked).
2. **ACCRUE, passive:** G-MON / G-MON-b on post-flip append rows, and `G-LD1`
   (the v34 leaf-delta prediction) on the first leaf-changed rows carrying a
   non-NULL delta. Both are reads of dreams that happen anyway; neither needs a
   run of its own. Note v34 reports NULL on its first post-deploy dream by
   contract.
3. ~~**RUN, free and already pre-registered:** the LoCoMo leg of the E1 judge
   artifact.~~ **BLOCKED 2026-08-30 — INPUTS LOST, and it is no longer free.**
   See the note below.

**The LoCoMo judge-artifact leg is BLOCKED, and the price must not be
re-quoted as zero.** The leg is a zero-LLM re-analysis of the banked Step-5
paired runs (`locomo_e1_on.json` / `locomo_e1_off.json`), and those dumps no
longer exist: untracked files on a box that was restore-stamped 2026-08-09, with
the `/tmp` db-dir wiped on container restart — the same loss class that took the
July `locomo_dbs_emb` stores. Searched and not found in git (any branch, commit
or loose object), on GitHub, in `~/.hermes/benchmarks/`, or on the Mac. The
flips dump alone would not suffice even if it survived: it carries correctness
flags, not the `ai_answer` texts the refusal check reads.

Re-running the pair costs ~3,200 reader calls (n=800 x 2 arms x answer+judge,
~4h ON / ~1.5h OFF). **Recommendation: bank as blocked, do not re-run**, on
three grounds. (a) It cannot change a default: E1's `read off` rests on zero
measured BENEFIT, not on the harm, so removing the harm entirely still leaves
nothing to turn on. (b) The expected direction is attenuation, not reversal —
the LME artifact rate applied to -2.9pp lands near z = -2.1, still significant.
(c) This document already priced recovering a pair at 1,600 reader calls as
"not proportionate"; this is twice that, for a confirmatory read. The record
therefore stays exactly where the honesty rule put it — **"costs on LoCoMo in
this regime"** — and that wording is now load-bearing rather than provisional.

**Historical 2026-08-25 sequencing note (superseded by v46):** E6 was then
unblocked but would have closed `invalid_at` on a tier whose READ default was
off (E1 Step 5's `read off, write on`) — instrumenting a path nothing consumed
was the unreachable-code-path trap. **E7** is ungated with no front-run designed.
**The `_anchor_facts` squeeze fix** was refused at S1-C1 and stays refused.

### E0. Evidence base and the six review constraints

**The finding.** Four instruments, three corpora, independent investigations all
terminate at the same wall: LME MS banks ~20 synthesis misses, LoCoMo ~137 of
its answerable-miss bucket, MSC 24/35, BEAM KU/CR/EO answer-side. P0 measured
the reader at ~20% of the gap to Hindsight (72.6 vs judge-matched 68.4). The
residual is **evidence packaging**: readers fuse pre-fused facts fine but fail
to fuse ~45 raw turn fragments. HyMem's store is structurally richer than
Hindsight's (bi-temporal KG, supersession, locked vocabulary, procedures, rules);
what it lacks is the middle-granularity unit — self-contained, dated,
entity-tagged **narrative facts** — between the atomic triple and the abstract
episode. Hindsight's paper (arXiv 2512.12818) is the template + existence proof
for that unit, **not** evidence: zero ablations, borrowed baselines under a
different judge, blank token budgets, LoCoMo adversarial omitted, and their
benchmark answer step is single-shot (the agentic reflect loop is a product
feature, not what produced their numbers). E1's case stands on HyMem's own
probes alone.

**The six constraints the 2026-07-30 review imposed (all adopted):**

1. **Mechanism criterion is the gate.** "MS ≥ +5pp on one LME A/B" is inside the
   churn floor (100% reader-side; judge churn measured <1.5% → majority-of-3
   judging buys nothing). Offline mechanism results decide build/no-build;
   LME A/Bs are **non-regression only**; scored confirmation lives on LoCoMo
   n=800 (±1.6pp) + MSC.
2. **Historical v26 constraint (superseded by v46): facts were per-range
   immutable, append-only, and version-tagged.** V46 instead makes the exact
   source unit immutable while publishing immutable result revisions and
   lifecycle events; the active projection changes only through validated
   replay of that same unit. It deliberately does not infer cross-source
   semantic supersession.
3. **E2 (per-entity observations) does NOT get that exemption** — it is
   synthetic over a mutable set, i.e. the aggregation-node pattern under watch.
   E2 is gated on the flip-watch turning green.
4. **E3 (cross-encoder rerank) is two changes, not one.** Backend flip
   (LLM → CE) and model swap (mxbai English → bge-m3 multilingual) are measured
   separately; adoption is ONE deliberate rebaseline after E1's scored runs.
5. **E4 needs a relative-date resolver first.** `dreaming/dates.py` is
   explicit-date-only by design ("resolving them requires an anchor date");
   "what did we decide last week" fails at the mention layer otherwise.
6. **Hindsight's numbers are demoted to corroboration.** Argue every item from
   HyMem's own evidence.

**Churn floors (read every delta against these):** LME per-category ±~5pp /
~4 questions; LoCoMo ±1.6pp @ n=800 (±3.2 @ n=200); MSC ~±4pp @ n=100;
sampling band ≠ churn floor (LoCoMo ±7.4pp @ n=151 answerable across samples).

---

### Step 1 — E1 front-run probe (`benchmarks/fact_probe.py`)

> **BUILT + RUN 2026-07-30 → G-F1 FAILED, E1 BANKED DEAD — then revival gate
> `G-F1b` OPENED later the same day** (verdict AND the G-F1b protocol at the end
> of this block; the probe itself is retained as the repo's faithfulness
> instrument and is the instrument G-F1b reuses).
>
> **BUILT 2026-07-30, UNRUN** (no LLM spend yet, and no banked source run on this
> box — `~/.hermes/benchmarks/` is empty here, so the gate must be executed on the
> Hermes box). `benchmarks/fact_probe.py` + `tests/test_fact_probe.py` (25 tests);
> `--sim` end-to-end verified. Two deviations from the sketch below, both
> deliberate and documented in the module docstring:
> 1. **No per-question store rebuild.** The `--inspect-floor` pattern exists to
>    reproduce a RETRIEVAL state; this probe exercises no HyMem tier (it indexes
>    into its own FTS5 table, the shape migration 026 will create) so a rebuild +
>    dream per question would cost hundreds of calls and measure nothing extra.
>    The production query path (`_FTS_SAFE`, `_fold_diacritics`) IS imported, so
>    the tokenization under test is the real one. Consequence: the selection
>    rule's "gold survived into the sent context" clause is inherited from the
>    source run's instrumentation rather than re-verified live, so the recovered
>    set can be slightly WIDER than the banked 20 — the probe prints the recovered
>    n for reconciliation.
> 2. **Cost is `questions × sessions`, not ~40 calls.** Extraction is one call per
>    session (as specified) and an LME haystack carries tens of sessions per
>    question; the plan's ~40-call budget corresponds to one call per QUESTION.
>    `--cost` prints the call count and exits; `--max-sessions` caps sessions per
>    question (label-free — keeps the most recent). **Pick the budget before
>    running.**
>
> `FACTS_PROMPT_V1` lives in the probe, NOT in `extraction/prompts/` — a front-run
> gate must not land a core change before it clears. Step 4 moves the text
> verbatim behind `FACTS_PROMPT_VERSION`. The gate reports **INCOMPLETE** (never
> PASS) until `--faithfulness` supplies the hand-score, so three-of-four can't
> read as a pass.
>
> **First box trials, 2026-07-30.** Trial 0 (`--sim`, free): selection recovered
> **28** candidates vs the banked ~20 — inside the documented widening (the
> "gold survived into context" clause is inherited, not re-verified), so read the
> gate as a fraction. Trial 2 (`--cost`): **2,759 extraction calls across 56
> questions** — capped to `--max-questions 10` (~500 calls). Trial 1 (E3 M2)
> blocked: OOM installing torch for sentence-transformers.
>
> **Trial 3 returned a hard 0% density, and it was an INSTRUMENT BUG, not a
> finding — and NOT store contamination** (the probe opens no HyMem store: it
> builds a fresh `:memory:` FTS5 index per question over that question's own
> `haystack_sessions`, BM25-only, no vector path, so UltraChat cannot reach it).
> The real cause: criterion 1 asked whether a returned fact CONTAINED the gold
> TURN, but a narrative fact is a rewrite of its turn *by prompt design*, so the
> check could never fire on real output. `--sim` read 82% only because canned
> "facts" are verbatim turns. **Fixed 2026-07-30:** "reachable" is now
> **gold-session provenance** (a returned fact was extracted from an answer-
> bearing session), with answer-string containment as corroboration and the old
> verbatim check retained as a diagnostic lower bound. The **0.60 threshold is
> unchanged** — the instrument was repaired, not the gate — and because that
> distinction only holds if the repair is itself controlled, the MS-hit control
> arm is now measured and printed beside the misses, with the report declaring
> the gate UNREAD when both arms return the same extreme or the control fails to
> beat the misses. `--rescore <dump>.json` recomputes all readings from a prior
> run's stored facts at ZERO LLM cost, so the fix does not re-spend Trial 3.
>
> **Rescore of Trial 3: provenance 9/10 misses (90%) vs 10/10 control.** Three
> follow-ups, all instrument-side, all fixed 2026-07-30 at zero LLM cost:
> 1. **The gap is ONE question at n=10.** Discrimination is now judged in
>    questions, not percentage points (at n=10, 10pp *is* one row — the LME
>    churn-floor discipline applied to this probe), and the report says so.
>    The 90% still reads against the 60% threshold; the miss-vs-control CONTRAST
>    does not read at this n.
> 2. **Answer-string containment was 0 on BOTH arms** — including questions the
>    live pipeline answered correctly, whose facts demonstrably reach the answer.
>    That is a non-firing check (LME `answer` fields are prose, so exact substring
>    never matches), not evidence that facts drop values. Auto-labelled as such.
> 3. **The faithfulness sample was unstratified**, so it drew ~20 UltraChat/
>    ShareGPT sessions. NOT store contamination — LongMemEval *builds* haystacks
>    by padding gold sessions with UltraChat/ShareGPT distractors (~50 filler to
>    ~5 gold), so a uniform draw is ~all filler by construction, and the probe
>    reads them from the dataset file, never from a store. Hand-scoring it would
>    have measured faithfulness on the padding. `build_faithfulness_sample` now
>    reserves half the budget for gold-bearing sessions, tags every entry with
>    `stratum`, and refuses (loudly) when no gold-bearing session made the sample.
>
> **Faithfulness hand-read (stratified, 10 gold-bearing sessions, ~45 facts):
> ~22–40% strict pass — FAILS the ≥0.90 criterion at both ends of the range.**
> Two distinct causes, and only one is the model's:
> - **Dates: the probe's own bug.** `validate_facts` stamped the SESSION date
>   onto any fact the model returned as `null` — converting a correctly-undated
>   fact into a confident specific date, contradicting both the prompt ("never
>   guess one") and the E1 schema (`fact_date` = explicit dates only; relative
>   references are E4's job). The hand-read scored those as model hallucinations;
>   they were the validator's. **Fallback removed**; `audit_fact_dates` splits
>   dated facts into `model_supplied` vs `injected_or_coincident` so a hand-score
>   already done on a pre-fix dump can be re-attributed at zero cost. Dates must
>   NOT be scored on the v1 dump — the run cannot attribute them.
> - **Content invention: the model's, and it alone sinks the gate.** Loyalty
>   programme, GPA 3.6 + Dean's list + sister at Stanford, named goats, Dr.
>   Johnson, 10K/20K ride, brand recommendations — none in source. Excusing dates
>   entirely still lands ~0.6, far below 0.90.
>
> **This is the pre-authorized `FACTS_PROMPT_V2` case: two VISIBLE prompt defects
> (not score-fitting).** (1) v1's date clause licensed invention — "if the session
> states one *or the session date applies*"; (2) "2 to 8 facts for a substantive
> session" read as a quota with a FLOOR of 2, and the cheapest way to satisfy a
> floor on a thin session is to invent — which matches the observed failures.
> V2 (`--prompt-version v2`): invention ban stated first and as the only rule that
> matters, explicit dates only, no quota, `[]` named as a good answer, assistant
> suggestions explicitly not user facts, and the session date is no longer shown
> to the model at all. v1 retained verbatim so the two are comparable arms.
>
> **G-F1 is now one decision, not a scoring question: spend the single allowed
> iteration (~500 calls, re-extraction — a rescore cannot recover this) or bank
> E1 dead. A second failure banks it dead by the pre-registered rule.**
>
> ---
>
> ## **G-F1 VERDICT 2026-07-30: FAIL. E1 IS BANKED DEAD.**
>
> The single allowed `FACTS_PROMPT_V2` iteration was spent and hand-scored:
> **21–29 of 38 facts pass strict (55–76%) on the same 10 gold-bearing sessions.
> The generous end of the range is still 14pp below the required 0.90.** By the
> pre-registered rule (`one prompt iteration allowed on a visible prompt defect;
> a second failure banks E1 dead — record and stop`), **E1 is closed. Step 4 is
> cancelled. No third prompt.**
>
> **V2 did what it was justified to do, and that is exactly why the verdict is
> clean.** Both visible defects are gone from the failure set: no quota-floor
> padding, no session-date stamping, facts materially more concise (38 facts vs
> ~45 for the same sessions). The improvement from ~0.6 (dates excused) to
> ~0.55–0.76 is inside hand-score noise at n=38, i.e. **not a real move.** The
> two worst V1 sessions reproduce their hallucinations verbatim under V2:
> `answer_35c5419d_3` (GPA 3.6 / Dean's list / sister at Stanford, 1/4) and
> `answer_f56e6152_1` (goats Billy & Nanny, cheese/yogurt, leash training,
> chicken feed, 2/6). **Identical inventions from a differently-worded prompt on
> the same source is the signature of a model-side failure mode, not a prompt
> defect** — which is precisely what the one-iteration rule exists to discriminate,
> and it discriminated. There is no third visible defect to fix.
>
> **Secondary failure mode found, and it is not prompt-fixable either:**
> sessions 3 and 10 (`answer_85a77c48_2` facts 7–8, `answer_a21f3697_2`
> "Foundations of Yoga" / MWF 7am / phone reminders) fail in the **truncated
> tail** — the char-cap drops turns, and the model confabulates over the cut
> rather than stopping. Raising the cap trades directly against extraction cost,
> which is the ~500-calls-per-10-questions number that made this probe expensive
> in the first place.
>
> **What the verdict does and does not say.** It kills E1 *as specified* — a
> generative narrative-fact artifact written into the store by an extraction
> pass. It does NOT invalidate the density half of the thesis: provenance
> reachability read 90% on misses, and the middle-granularity retrieval unit may
> still be worth having. But it cannot be *generated*, because at 0.55–0.76
> faithfulness a fact tier writes ~1 invented biographical claim per 3–4 facts
> into permanent memory, where supersession (E6) would then defend it and the
> reader would cite it with confidence. **A retrieval tier that fabricates
> beats no tier only if you never read it.** Note also that the 90% provenance
> was measured on facts that include invented content — BM25 will happily match
> a fabricated fact into the top-5 — so it is provenance-correct, not
> quality-correct, and does not survive as an independent finding.
>
> **Campaign E survives on its hedge, as designed.** E5 shipped and passed
> (31/31, 0/12 no-harm). E3 (M1/M2) is unaffected — it reranks what retrieval
> already found and generates nothing. E2/E4/E6/E7 were all specified over the
> facts artifact; **E6 (supersession over facts) dies with E1**, E2/E4/E7 need
> re-specification over episodes/observations before any of them can be costed.
>
> **Retained assets:** `benchmarks/fact_probe.py` + tests stay in-tree. It is the
> only instrument in the repo that measures *extraction faithfulness* against
> source turns with a stratified sample and a correct-answer control, and any
> future generative-write proposal (E2 observations included) must clear the same
> bar before it writes to the store. The V1/V2 dumps are the fixture. Cost of the
> whole G-F1 decision: ~1,000 extraction calls, one dead build avoided.
>
> ---
>
> **REVIVAL GATE `G-F1b` — OPENED 2026-07-30 (user decision): change the dream
> model and re-run the gate.** Rationale on record: the narrative-fact tier is
> judged the **largest available lever against Hindsight** — the E0 thesis says
> the middle-granularity unit is the one structural thing HyMem lacks, and
> nothing else in the campaign addresses it. So E1 is **SUSPENDED, not dead**,
> pending exactly one new-model arm. This is what the "revival requires a NEW
> faithfulness result" clause anticipates — G-F1 discriminated a *model-side*
> failure, so a model change is the responsive action, not score-fitting.
> Protocol, fixed in advance:
>
> 1. **Gate first, migration second.** The gate run is identical whether the swap
>    ends up wholesale (new project dreamer) or scoped (a facts-pass-only model
>    via the pluggable `LLMClient`), so no migration decision is needed before
>    the result exists. A wholesale swap is a separate migration event with the
>    deepseek-precedent price tag (full re-pair, new frozen baselines, one-time
>    RAPTOR refusion) — priced on its own merits, not smuggled in under E1.
> 2. **ONE pre-registered candidate, chosen on grounds independent of this
>    gate.** Natural candidate: `gpt-oss-120b` — already the project's P0
>    reader-parity model and independent judge, so it is a model already trusted
>    and paid for; no judge circularity because faithfulness is HAND-scored.
>    Iterating models until one clears is the multiple-comparisons version of
>    score-fitting and is not permitted.
> 3. **The run:** `fact_probe.py --prompt-version v2 --model <candidate>
>    --max-questions 10` on the same stratified gold-bearing sample (~500
>    extraction calls; `--cost` first to confirm), hand-scored strict against the
>    same 0.90 bar. No new plumbing needed — `--model`/`--api-key`/
>    `--extra-body` already exist.
> 4. **Borderline rule:** at n≈40 facts the hand-score has real width (V2's own
>    read spanned 21–29/38). A result in ~0.85–0.95 is EXPAND-THE-SAMPLE, not a
>    verdict in either direction.
> 5. **Read the truncation-tail mode separately:** does the candidate stop at
>    the char-cap cut or confabulate past it (the sessions-3/10 pattern)? A model
>    that stops cleanly moots the secondary failure mode; one that doesn't
>    re-opens the cap-vs-cost trade.
> 6. **PASS** reopens Step 4 as specced (the historical plan put E6 behind it;
>    cross-source E6 was later rejected) and supplies the
>    faithfulness result E2/Plan C are gated on (the flip-watch still blocks E2
>    independently). **FAIL** banks E1 dead on a second model class — record and
>    stop; at that point the finding is about the task, not the model.
> ### G-F1b — the one allowed revival read (added 2026-07-31, PRE-REGISTERED)
>
> **What this is.** E1's bank-death rule allows revival only on a NEW faithfulness
> result (never a re-read of the 0.55–0.76 one). G-F1b is that read: `fact_probe.py`
> re-run on the same instrumented emb-ON 2026-06-07 source run
> (`longmemeval-v2-hymem-20260607T164031Z-seed0.json` + `longmemeval_s_cleaned.json`,
> pair verified 500/500 gold-turn counts) with extraction model `openai/gpt-oss-120b`
> @ OpenRouter, prompt arm v2, `--max-tokens 4096`. Same four criteria, same
> thresholds — the mechanism is unchanged, only the extractor differs.
>
> **Pre-registered readings, BEFORE the numbers exist (2026-07-31):**
> 1. **Parse-failure ceiling (in code, `_MAX_PARSE_FAILURE_RATE = 0.02`):** if
>    `parse_failures/calls > 2%`, verdict is **INCOMPLETE** (re-run at a higher
>    `--max-tokens`), never FAIL. Rationale: truncation biases the four criteria in
>    opposite directions — criterion 1 (gold reachable) gets harder, criteria 3/4
>    (median upper bounds) get easier — so a truncation-heavy run is
>    indistinguishable from an honest FAIL without this counter. Pinned by
>    `test_parse_failure_ceiling_is_incomplete_never_fail`.
> 2. **Reasoning asymmetry — constrains what a PASS means.** The v4-flash arm ran
>    with thinking disabled; gpt-oss-120b CANNOT disable reasoning (OpenRouter
>    400: "Reasoning is mandatory for this endpoint"). A PASS means "this
>    extractor, WITH mandatory reasoning, clears 0.90" — it does NOT license the
>    inference "gpt-oss-120b is more faithful than v4-flash" (confounded with
>    reasoning on/off). Any dreamer-swap argument is a separate migration event
>    with its own price, per the deepseek precedent.
> 3. **Budget is ~2× the priced figure, deliberately accepted.** 1002 extraction
>    calls at `--max-questions 10` (the plan's ~500 assumed one call per question;
>    the honest count is `questions × sessions` ≈ 50/question), plus 4096-cap
>    reasoning output ≈ 3.4× v4-flash's per-call spend → total ≈ 6–7× the priced
>    cost. Criteria are fractions so the gate reads fine at this n; faithfulness
>    (the underpowered criterion) widens if the budget is cut to
>    `--max-questions 6` (601 calls).
>
> **Instrument patches this read required (all default-preserving, suite green):**
> `--base-url` (probe hardcoded DeepSeek), `--max-tokens` (1200 default; gpt-oss-120b
> burned it on CoT and returned `content=null` — verified 4096 completes), and a
> null-content guard (`raw is None` → counted parse failure, not a row crash).
> The 2026-07-30 V1/V2 dumps are NOT on this box, so the v4-flash parse-failure
> rate cannot be re-derived here; if it was non-zero, the 0.55–0.76 carries the
> same truncation asterisk.
>
> **SMOKE SERIES 2026-07-31 (n=1 question per arm, 98 calls each) — config ladder:**
>
> | config | parse failures | ceiling | gold production | notes |
> |---|---|---|---|---|
> | 1200 (probe default) | crashed on `content=null` | — | — | CoT ate the whole budget; `raw.startswith` on None |
> | 4096, run 1 | 6/98 = 6.1% | EXCEEDED | 2/4 (50%) | no retry on 200-null-content → one flap = one failure; both `_2` gold sessions 0 facts |
> | 4096, run 2 (retry fix) | 2/98 = 2.04% | EXCEEDED (0.04pp) | 2/4 (50%, different pair) | gold production run-to-run non-deterministic → stochastic CoT budget exhaustion, not session properties |
> | 8192 (retry fix) | **0/98 = 0%** | **CLEARED** | **4/4 (100%)** | median facts/session 8.0 — exactly at the criterion-3 cap; over-extraction is now the live risk of this config |
>
> **Decisions banked:** full run at `--max-tokens 8192`, `--workers 4`. The
> 4096-vs-8192 gap is the difference between a ~50% gold-production rate and
> 100% — the faithfulness sample (which can only fill from gold sessions that
> produced facts) is the binding constraint, and 8192 is the config that fills it.
> The `--rescore --faithfulness-sample 40` expansion stays free if the hand-score
> lands 0.85–0.95. n=6 vs n=10: n=6 yields ~28 gold sessions (≥20, the 40-sample
> requirement) at the observed 100% production; n=10's 400 extra calls buy no
> additional evidence on the criterion that failed twice.
>
> **Criterion 3 is non-discriminating at the 8192 config (recorded 2026-08-01,
> before the run).** Smoke density over ~98 sessions read median 8.0 miss / 8.0
> control — the criterion's threshold (≤8) sits at the measured expected value,
> so its outcome is decided by sampling, not by extraction quality. Mechanism:
> density is jointly set by the prompt's "2 to 8 facts" instruction and the
> token cap; raising 4096→8192 (required for parse integrity, 0/98 vs 6.1%/2.04%)
> mechanically raised mean output 207→322. The 4096 densities were partly a
> truncation artifact; 8.0 is the honest number. Accordingly: a run that fails
> only criterion 3 at median 9, with criteria 1, 2 and 4 passing, is recorded as
> a PASS on the scored question plus a separate store-cost note — 8–9 is a 12%
> write-volume difference, not a project-deciding quantity. A median ≥11 is a
> genuine over-extraction FAIL and is read as one. The cap is not lowered to
> move this number in either case.
>
> **G-F1b VERDICT 2026-08-01: PASS — faithfulness 123/123 (1.00) on the healed
> sample. E1 is UN-BANKED; Step 4 reopens as specced.** The 2026-07-31 run's
> hand-score surfaced an INSTRUMENT BUG, not a model failure: the dump recorded
> `render_session(messages)[:4000]` while the extractor saw the full 12,000-char
> input (the `[:4000]` truncation had been in `fact_probe.py` since its first
> commit, 9c172d8 — before both v4-flash hand-reads). Fixed upstream (961a5f5:
> dump stores the exact extractor input; `build_faithfulness_sample` re-renders
> from the dataset, so `--rescore` heals existing dumps). All 123 facts re-scored
> against the full inputs: the 50 previously-unresolvable facts are verbatim in
> chars 4000–12000 — including the pre-flagged "inventions" (Anglo American
> Diavik = the source's own claim, preserved faithfully; "neutral theme" = a
> genuine later change of mind, supersession-correct; the Yamaha FZ6R
> $3,500/Jan-20/Craigslist, page-250 Nightingale, MWF-7am yoga, black Lenovo bag
> — all verbatim). Criterion 5 reads clean: two true mid-word 12k cuts, zero
> facts reference content past the boundary; the sessions-3/10 confabulation
> mode does not reproduce on gpt-oss-120b — the cap-vs-cost trade is mooted.
> Two disclosed quibbles (E09 pronoun, E15 one-word completion) dock nothing —
> even at 0.99 the verdict is identical. Per pre-registered step 6: **PASS
> reopens Step 4 as specced; the historical plan said E6 revived behind it
> (cross-source E6 was later rejected), and the faithfulness result
> E2/Plan C are gated on is supplied** (the flip-watch still blocks E2
> independently). Wholesale-vs-scoped migration decision is now live, priced
> separately per protocol step 1. **Open item:** the same `[:4000]` trap would
> have flipped this 1.00 into an apparent 0.59 — squarely inside G-F1's
> 0.55–0.76 — so if the v1/v2 dumps survive anywhere, the same free rescore
> settles whether v4-flash actually invented the GPA-3.6/Stanford and
> Billy-&-Nanny content. Not on this box (confirmed); recorded open.
>
> > **OPEN ITEM CLOSED 2026-08-30 — by a stronger route than the one proposed.**
> > The v1/v2 dumps never turned up, so the free rescore was never available.
> > G-F1 was instead re-RUN on v4-flash against full sources (prompt v2,
> > 20-session stratified sample, 950 extraction calls, 0 parse failures) and
> > **PASSES all four criteria**: provenance 10/10 (100%) vs a >= 60% bar,
> > **faithfulness 1.00 (98/98)**, median 3.0 facts/session (<= 8), control
> > median 4.0 (<= 12). The dump was verified to carry complete extractor input
> > (min 6,454 / max 12,000 chars, no slice markers), and the pre-flagged
> > "invention" session `answer_35c5419d_3` — GPA 3.6 / Dean's list / Stanford —
> > **re-scores verbatim-verified against the full source**.
> >
> > **So the 0.55-0.76 was an instrument artifact end to end, on BOTH models.**
> > The `[:4000]` trap did not merely depress one gpt-oss-120b run; it laid a
> > false floor under the model this project then migrated away from partly on
> > faithfulness grounds. v4-flash is not a confabulator, and E1's original
> > "banked dead by G-F1" was a reading of a broken recorder rather than of a
> > model. A fresh run is the stronger closure anyway: a rescore of surviving
> > dumps would have inherited whatever else those dumps got wrong.
> >
> > **What this does NOT change.** E1's shipped verdict stays `read off, write
> > on`. That came from Step 5's SCORED LoCoMo measurement (-2.9pp on the fired
> > subset, z = -2.40) and from zero measured benefit anywhere — both
> > independent of G-F1, which only ever gated whether E1 got BUILT. A
> > faithfulness result cannot revive a read side that was turned off for
> > costing.
> >
> > Attribution was audited beyond substring containment: association-risky
> > claims (Tybee, Topsail, Fish Factory, meal plan / Coach $800 / gym, and a
> > $10 train fare the user was REPORTING A FRIEND'S CLAIM about) all read clean
> > against the actual turns, the reported-speech case attributed correctly
> > rather than asserted. The "recovered 23 vs banked 20" warning is the
> > documented widening — the gate is a fraction and was run as one — and the
> > probe's own docstring already calls it conservative for the gate.
> >
> > Artifacts (box): `~/.hermes/benchmarks/facts_v4flash_fullsrc_20260830.json`,
> > `..._scored.json`.

**Idea.** Extract narrative facts with a draft prompt from the haystacks of the
~20 banked LME MS synthesis misses plus an equal-sized control of MS hits, and
measure the two things the build can't measure later: (a) does a fact tier
deliver the question's gold inside ≤5 dense items (vs ~45 raw slots), (b) is
the extraction faithful (≥0.9 hand-score — every value/name/date traceable to
a turn, the profile.v2 gate pattern). Probe extraction is append-only by
construction, so its artifacts are reusable as test fixtures in Step 4.

**Architecture.**
- Selection: the instrumented emb-ON floor-audit run JSON (2026-06-07) in
  `~/.hermes/benchmarks/`; the readside plan §2.1 selection rule (MS non-`_abs`,
  `correct=false`, `recall_ceiling=true`, no `"none"` in `gold_turn_tiers`,
  gold survived into the sent context) + equal random `correct=true` MS control.
- Per qid: rebuild the per-question temp DB (the `--inspect-floor` pattern,
  `_inspect_floor_questions` in `longmemeval_adapter.py`), ingest haystack,
  run **one fact-extraction call per session** (draft `FACTS_PROMPT_V1`: input
  = session turns char-capped; output JSON list of 2–8 facts, each
  `{text, date ISO|null, entities[]}`; rules: self-contained without the source
  turn, values/names/dates verbatim, one fact per exchange/decision/outcome,
  no invention, skip smalltalk), store into a temp FTS5 table, retrieve top-5
  with the question.
- Gold check: **containment** (port `_gold_in_pool` from
  `longmemeval_adapter.py`: one string contains the other or a shared 40-char
  prefix) against the ≤5 returned facts. Per question: gold covered? facts
  extracted total? Same for control.
- Faithfulness: dump all extracted facts to JSON; hand-score a 20-session
  sample.
- `--sim` stub mode (canned extraction) verifying selection → rebuild → FTS →
  gold-check plumbing offline, mirroring `multihop_probe.py`'s plumbing tests.

**Pre-registered gate G-F1 (decides build/no-build; score-free).**
BUILD iff: gold-in-≤5-facts on **≥60%** of banked misses AND faithfulness
**≥0.9** AND median facts/session ≤ 8 AND control shows no systematic
over-extraction (>12/session). One prompt iteration (`FACTS_PROMPT_V2`) allowed
on a visible prompt defect; a second failure banks E1 dead — record and stop.

**Tests:** probe-only; the `--sim` plumbing test + a fixture asserting the
selection rule reproduces the banked n on the 2026-06-07 JSON.

---

### Step 2 (parallel, day one) — E5 anaphora resolver (`hymem/query/coref.py`)

> **BUILT + GATE PASSED 2026-07-30.** `hymem/query/coref.py`, wired in `augment()`
> before every tier, `ctx.coref` provenance, three config flags
> (`coref_enabled=True`, `coref_max_turns=6`, `coref_llm_enabled=False`).
> Tests: `tests/test_coref.py` (27) + `tests/test_coref_eval.py` (5); suite 856.
> **Gate (`benchmarks/coref_eval.py`, set `benchmarks/coref_eval_set.json`):
> resolution 31/31 = 100% (graph path 10/10, salient-token path 21/21; pronoun
> 14/14, ellipsis 10/10, demonstrative 7/7), no-harm control 0/12 rewrites →
> PASS.** No LME run, by design.
>
> Two notes on how the gate was reached, so the number is readable:
> - `rewrite_query` takes an optional `conn` (not in the plan's signature) because
>   `match_known_entities` — the plan's own resolution rule — is a graph lookup.
>   Without a store it falls back to salient tokens; both paths are gated
>   separately and both clear.
> - The FIRST gate run scored 100% resolution but **2/12 control false fires**, both
>   real precision defects: the content-token ceilings were too generous and the
>   stopword table was missing interlocutor words ("we"/"ik"), which inflated the
>   count on ordinary questions. Fixed by tightening the ceilings to the observed
>   distribution (pronoun 3, demonstrative 3, ellipsis 2 + a 5-token total ceiling
>   — "wat hebben we afgesproken over het prijsexperiment?" has only 2 content
>   tokens yet is a complete question) and extending the stopwords. Resolution
>   stayed 30+/30+, i.e. **no recall was traded for the precision**. The eval set
>   is now a standing regression test (`test_coref_eval.py`), so a future loosening
>   that re-introduces a false fire fails the suite.

**Idea.** P3, the real-life lever LME is blind to: follow-up queries carry
pronouns/ellipsis ("what did she say about that?", "en de prijs?") and every
retrieval tier misses simultaneously. Resolve against the session's recent
turns BEFORE retrieval. Zero LME delta by construction; production value only —
it is the campaign's hedge: it ships value even if G-F1 kills E1.

**Architecture.**
- New `hymem/query/coref.py`:
  `rewrite_query(query, recent_turns, *, cfg, llm=None) -> QueryRewrite` with
  `QueryRewrite{rewritten: str, changed: bool, rule: str, resolved: dict}`.
- **Stage 1, heuristic (stdlib, EN+NL):** fire on (a) pronoun-dominant queries
  (standalone it/that/she/he/they/dit/dat/ze/hij with no known-entity match),
  (b) ellipsis follow-ups (≤4 content tokens, no entity match), (c)
  demonstratives ("that project", "die tool"). Resolve referents from the last
  `cfg.coref_max_turns` recent turns: entities via `match_known_entities` over
  those turns first, else salient noun tokens of the last user turn.
- **APPEND, never replace:** `rewritten = original + " (context: <referents>)"`.
  Retrieval keeps the original tokens AND gains the resolved ones — the
  additive invariant applied at the query level.
- **Stage 2, optional LLM fallback** (`cfg.coref_llm_enabled`, default False):
  one tiny "rewrite as a standalone question" call when the heuristic fires at
  low confidence; `LLMClient` Protocol only.
- Wiring: in `augment()` right after ability detection, before ALL retrieval
  tiers; only when `session_id` is passed (needs `recent_turns`); every tier
  consumes `rewritten`. `ctx.coref` records the rewrite for observability
  (same contract as `detected_rule`).
- Config: `coref_enabled: bool = True` (heuristic is cheap and safe),
  `coref_max_turns: int = 6`, `coref_llm_enabled: bool = False`.

**Gate:** a hand-built 30-item eval set (EN+NL pronoun/ellipsis follow-ups with
known referents) — heuristic resolution ≥80% with zero rewrites on
self-contained queries (the no-harm control). No LME run.

**Tests (`tests/test_coref.py`):** pronoun query + recent turns → rewritten
contains referent; self-contained query → byte-identical; no `session_id` →
inert; entity-matched query unchanged; Dutch pronouns; append-not-replace
(original tokens preserved verbatim); LLM fallback gating (stub call counts);
`coref_enabled=False` → inert; `ctx.coref` observability fields.

---

### Step 3 (parallel, offline) — E3 reranker measurements (`benchmarks/rerank_ab.py`)

> **BUILT 2026-07-30 — RUN 2026-07-30/31, VERDICT IN, NOTHING FLIPPED. Read the
> STATUS block at the end of this step first; the spec below is the original
> pre-registration, kept verbatim.** `benchmarks/rerank_ab.py` + `tests/test_rerank_ab.py` (17 tests);
> `--sim` end-to-end verified, and the hand-set is validated as an instrument by
> the suite (every gold verbatim in its corpus AND reachable inside the BM25 pool —
> a hand-set whose gold is unreachable measures retrieval, not rerank).
>
> - **M2 needed a Dutch set that did not exist**, so `benchmarks/rerank_handset.json`
>   was written: two blocks (`nl`, `en`), each one shared ~40-turn corpus + 15
>   questions whose gold is a verbatim turn, with on-topic distractors sharing the
>   question's vocabulary. One corpus per block (not per question) is what makes
>   ranking competitive. The `en` block doubles as M1's default source and as M2's
>   English regression control; `--dataset` runs M1 on an LME MS slice instead.
> - **Hard guard:** `cross_encoder_rerank` degrades to returning the pool UNCHANGED
>   when sentence-transformers/the model is missing, which would post a plausible
>   parity result for a reranker that never ran. The script refuses to run in that
>   state (exit 2) rather than reporting it.
> - Pool caveat, stated in the docstring: the candidate pool is a wide BM25 sweep
>   of the raw-message tier (no dream → no chunks; no embedding server → no vec),
>   so ABSOLUTE ranks carry the `gold_rank_probe.py` bias. The comparison is still
>   sound because both arms are handed the identical pool object — read the output
>   as "how well does backend X order this pool", never as a production rank.
> - Nothing flips: adoption stays Step 6.

**Idea.** Two SEPARATE measurements, both offline, neither touching a frozen
baseline: **M1** cross-encoder backend vs LLM backend (shipping model
`mixedbread-ai/mxbai-rerank-base-v1`); **M2** model swap mxbai →
`bge-reranker-v2-m3` (the Dutch/multilingual constraint). Adoption is deferred
to Step 6; this step only produces the decision data.

**Architecture.**
- New probe (or `gold_rank_probe.py` extension): for a fixed question set (LME
  MS slice for M1 + a Dutch hand-set for M2), pull the production candidate
  pool (BM25+vec fused, `rerank_top_k` wide), rerank with each backend in turn.
- Measure: gold-rank distribution (median, ≤15 share), pairwise rank
  correlation, p50/p95 rerank latency (CE local vs LLM round-trip), LLM tokens
  spent per query.
- **M1 gate (pre-registered):** CE parity = median gold rank within 1 position
  of LLM rerank AND ≤15 share within 2pp AND latency ≥10× better. **M2 gate:**
  bge ≥ mxbai on the Dutch set with English median-rank regression ≤1 position.
- `--sim` mode with a deterministic fake reranker for plumbing.

**Tests:** probe plumbing + rank-correlation/latency math unit tests; no core
changes.

#### STATUS 2026-07-30/31 — BOTH MEASUREMENTS RUN; M1 FAIL, M2 PARITY; **no default changed**

**Decision (user, 2026-07-31): keep the shipping config exactly as it is** —
`rerank_model="llm"` and `rerank_cross_encoder_model="mixedbread-ai/mxbai-rerank-base-v1"`.
Step 6 closes unexercised. Neither measurement produced a reason to move that
survives its own pre-registered criterion.

**M1 (CE backend vs LLM) — FAIL, and the quality row is UNMEASURED.**

| arm | p50 | p95 | gold rank (median/mean) |
|---|---|---|---|
| CE `mxbai-rerank-base-v1` (CPU) | 29.7s | 41.0s | 1.0 / 1.0 |
| LLM `deepseek-v4-flash` | 2.9s | 3.8s | 1.0 / 1.1 |

Gate required latency **≥10× better**; observed **0.09× (10.7× worse)**. This
closes on arithmetic, not on model choice: beating a 3.8s p95 by 10× means
≤0.38s for 40 candidates ≈ **9.5ms/candidate**. mxbai is 108× off that, bge 37×
off. **No cross-encoder reaches it on CPU, so trying more models is pointless** —
the avenue closes without further runs.

**Do not record M1 as "quality parity".** Both arms scored 1.0/1.0 because they
ran on the *original* handset, where BM25 alone already put gold at rank 1 on all
30 questions (below). Every rank cell was at ceiling, so the quality criterion
could not fail. M1 was never re-run on handset v3. The honest record is
**FAIL on latency, quality untested.** The CE's non-latency case (no API
dependency, no token cost, offline operation) was never in this gate's criteria
and is untouched by the failure.

*The re-run condition, for the record:* GPU hardware plausibly closes the 9.5ms
budget, but that is a **hypothesis — no CE was ever benchmarked on a GPU here**.
Reviving M1 means re-running the gate as written on that hardware, not inheriting
this verdict with the sign flipped.

**M2 (mxbai → bge-reranker-v2-m3) — PARITY, not a bge win.** Run on handset v3.

| | mxbai R@1 | bge R@1 | Δ | mean gold rank |
|---|---|---|---|---|
| NL (n=20) | 11/20 | 15/20 | **+4** | 2.25 → 1.25 |
| EN (n=15) | 12/15 | 10/15 | **−2** | 1.57 → 1.60 |

Found: NL 20/20 both; EN 14/15 → 15/15. Across the 11 hard items (BM25 gold rank
≥10) **mxbai wins overall**: bge takes nl-08 and nl-18, mxbai takes nl-06, en-04,
en-06. Three reasons this is parity and not a result:

1. **The effect size was pre-registered in items at n=15** (4 items = 26.7pp).
   At n=20 the same effect is **5.3 items**; observed 4 → fails as written.
   Changing the denominator without rescaling the criterion silently loosens it.
2. **McNemar:** the most favourable discordant split consistent with 11 vs 15
   (bge 4, mxbai 0) gives **p = 0.125**; a 5–1 split gives p ≈ 0.22. Same
   "prior, not a result" territory as E4's 4/0 one-directional misses.
3. **The five items added between the n=15 and n=20 runs netted Δ=0.** Models are
   deterministic, so the first 15 are unchanged; under a real ~27pp effect five
   items should have widened the gap by ~1.3. The resolution test came back
   negative.

M2 was specified as **non-inferiority**, so parity is a legitimate and available
conclusion — bge is not worse. What it does not license is banking "+4 on Dutch
R@1" as a justification.

**bge's one real, measured advantage is latency: 3–4× faster than mxbai on CPU**
(NL p50 48.3s→11.4s, p95 59.6s→13.9s; EN p50 26.6s→11.3s), with a visible
mechanism — **mxbai's English-first tokenizer costs 1.82× on Dutch while bge is
language-flat at 1.01×**. That argues for bge *within* the CE path. It was still
not adopted, because (a) the CE path is dormant (`rerank_model` defaults to
`"llm"`; `augment.py` routes to the CE only on `cfg.rerank_model ==
"cross-encoder"`), so the flip would change nothing for anyone on the default
config, and (b) it **triples the opt-in download, ~0.7 GB → ~2.3 GB**, against
the hardening initiative's pip-install-ease goal. A dormant setting is not worth
a footprint regression on an unmeasured quality row. *If the CE backend is ever
adopted (i.e. M1 re-run and passing on GPU), revisit bge in the same commit — on
latency + parity, not on the Dutch R@1 number.*

**Why no LME non-regression guard was run.** It would have been **vacuous by
construction**: LME runs the default config, the CE path never executes, so the
run returns "no change" whatever the CE model id says. That is a confident
constant, not a measurement — the same failure class as the ceiling handset and
the degenerate ≤15 criterion below. A meaningful guard would need
`rerank_model="cross-encoder"` in both arms, i.e. a scored run of the backend M1
just closed. **Generalize: before spending a run on a non-regression guard, check
that the code path under test is actually reachable from the config the guard
runs.**

**Three instrument defects, all found mid-flight; two fixed, one open.**

1. **The vacuous handset (fixed).** BM25 alone put gold at **median rank 1.0 on
   all 30 original questions** — every rank measurement in the first M1 and M2
   runs was measuring nothing, because the rerankers had no work to do. Caught
   from the ceiling signature (all 8 rank cells identical at 1.0) *before* the
   baseline confirmed it. The handset's distractors were never the problem;
   **question-to-gold vocabulary overlap** was. Rewritten as v3 (30→35 questions;
   NL targets compounds `verbindingspool`/`connection pool`, separable verbs,
   register shifts, synonyms; EN matched for difficulty), and gated as an
   *instrument* on a BM25 front-run independent of both arms: **median gold rank
   ≥4, zero items at rank 1, all gold reachable in the pool.** Achieved NL 4.0 /
   EN 5.0, 0 at rank 1, 15/15 found both. **Lesson: gate the measuring device on
   a baseline arm before running the arms under test** — the same discipline
   G-E4a needed and the reason the first two E3 runs were thrown away.
2. **The `id()` overlap bug (fixed).** `run_arm()` keyed its overlap dict on
   `id(h)`, the Python object address. Both `cross_encoder_rerank` and
   `llm_rerank` return `replace(hit, ...)` — new objects, new addresses — so
   every ranked hit mapped to `-1` and overlap read as zero. Caught from the
   arithmetic: two arms each drawing 15 from the same 40 share ~5.6 by chance, so
   an observed <3 is **below chance**, which two rerankers scoring the same
   documents cannot produce. **Below-chance overlap is an anti-correlation
   signature, i.e. a bug, never "the arms disagree."** Fixed to `h.message_id`
   (stable across the copy) + 2 regression tests, one of which asserts the old
   keying *fails*, so the bug cannot silently return. Suite 19 green.
   **Blast radius is benchmark-local — the product does not share the defect:**
   `augment.py` keys on domain IDs throughout (`r["id"]`, `hit.chunk_id`,
   `hit.episode_id`), as does `dreaming/aggregate.py`. Verified, not assumed.
3. **The degenerate ≤15 criterion (OPEN — fix before any E3 re-run).** With
   `--pool 40 --top-k 15`, "≤15 share" is arithmetically identical to "Found":
   every returned item has rank ≤15 by construction. The M1 gate's second clause
   therefore carried no information in either run. **top-k must be < pool for the
   share metric to mean anything** — set `--top-k 5` (or widen the pool) before
   the criterion is read again.

**Carry-forward.** The single reusable finding here is not about rerankers: it is
that **three separate measurements in this step — the ceiling handset, the
degenerate ≤15 share, and the proposed LME guard — were each incapable of
failing.** A gate that cannot fail returns a confident constant and reads as a
pass. Check reachability and headroom on the instrument before spending anything
on the arms. See [docs/diagnostic_controls.md](docs/diagnostic_controls.md).

---

### Step 4 — E1 build: the narrative-facts artifact ~~(gated on G-F1)~~ ~~SUSPENDED, re-gated on G-F1b~~ **BUILT 2026-08-02**

> **BUILT 2026-08-02 on the G-F1b PASS** (123/123 strict faithfulness,
> gpt-oss-120b; the gate that cancelled this step on 2026-07-30 and revived it
> 2026-08-02). Landed as specced below — **schema v26** (`narrative_facts` +
> `narrative_facts_fts` + `narrative_fact_embeddings` + `vec_facts`,
> `sessions.facts_message_id`, `dream_runs.facts_extracted/fact_failures`),
> `hymem/dreaming/facts.py`, the additive `ctx.facts` tier with RRF fusion,
> the `=== FACTS (verified past events) ===` block in `ask()` and the MCP
> `hymem_augment` rendering, 22 tests in `tests/test_facts.py`, full suite
> green. **Deltas from the spec, all deliberate:**
>
> 1. **Extraction reads RAW MESSAGES, not chunks.** The spec said "the
>    session's undigested tail (chunks above the watermark)". The gate scored
>    extraction over raw `role: content` turns, and chunks are a different
>    corpus (salience-floored, assistant prefixes duplicated across chunks,
>    short turns dropped). Reading chunks would have shipped a pipeline the
>    gate never measured. Watermark semantics are unchanged — it is a message
>    id either way.
> 2. **`FACTS_PROMPT_VERSION` is `facts.v2`, not `facts.v1`** — v2 is the arm
>    that cleared the gate; v1 (the arm that failed G-F1) never ships.
> 3. **The watermark is the ONLY facts skip-guard** (no
>    `facts_prompt_version` session stamp, unlike digest/profile). Forward-only
>    versioning means a bump must NOT re-read covered ranges, so a
>    version-keyed guard would be actively wrong.
> 4. **An oversized single turn is COVERED, not re-read.** Truncation keeps
>    the head, so no later dream can read more of that turn; holding the
>    watermark (the digest's behaviour) would re-spend one call per dream
>    forever and store nothing. The dropped tail is logged
>    (`facts.oversized_turn`), never silent.
>
> **Not done, deliberately:** narrative facts are absent from
> `portability.py`'s export spec — which already omits `user_profile` (v18),
> `rules` (v23) and the aggregation layer, so this joins the tracked
> export-gap phase rather than being fixed inconsistently here. **Step 5
> (scored confirmation: LoCoMo n=800 `--fresh`, MSC, LME non-regression) is
> now the open item** — the build is unmeasured on any benchmark. The historical
> plan said E6 revived
> behind this step; E2 stays blocked on the flip-watch independently.

> **Superseded 2026-09-04 by schema v46.** The block above is historical. The
> current contract uses exact lossless source occurrences and resumable
> character offsets; successful empty units advance, malformed/over-cap units
> hold and retry, and no oversized tail is discarded. Prompt/config changes
> replay every stored unit on its original exact coordinates before new tail
> units append. Facts have immutable result revisions plus retract/resurrect
> lifecycle history, active-authority-only FTS/vector readers, and atomic v10
> portability. Both extraction and retrieval default on.

**Historical v26 idea.** Dream-time extraction of self-contained narrative facts, stored
immutably (append-only, version-tagged), served as an additive retrieval tier
and as the lead evidence block in `ask()`. Subsumes Plan C's granularity goal
as a NEW artifact class — episode membership is untouched, so the Plan C
sequencing constraint (RAPTOR flip-watch) does not apply. The historical v26
plan described this as unblocking E6; cross-source E6 was later rejected, and
v46's exact same-unit replay does not unblock or implement it.

**Schema (migration `026_narrative_facts.sql`; `EXPECTED_SCHEMA_VERSION` 25→26;
table also added to the `schema.sql` fresh-DB baseline — indexes/ALTERs only in
the migration per the documented gotcha):**

```sql
CREATE TABLE IF NOT EXISTS narrative_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_message_id INTEGER NOT NULL,
    end_message_id INTEGER NOT NULL,
    text TEXT NOT NULL,                    -- self-contained narrative; IMMUTABLE
    fact_date TEXT,                        -- ISO or NULL (explicit dates; relatives = E4)
    entities TEXT NOT NULL DEFAULT '[]',   -- JSON array of canonical names
    prompt_version TEXT NOT NULL,          -- 'facts.v1' provenance tag
    valid_at TEXT,                         -- bi-temporal, mirrors knowledge_graph
    invalid_at TEXT,                       -- the ONLY mutable field (E6 closes it)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, start_message_id, text)
);
-- + narrative_facts_fts (FTS5 over text, triggers, mirroring episodes_fts)
-- + narrative_fact_embeddings JSON cache (content-addressed like edge_embeddings)
-- + vec_facts vec0 table when sqlite-vec is present
-- + sessions.facts_message_id INTEGER  (facts watermark, mirrors v24 digested_message_id)
-- + dream_runs: facts_extracted INTEGER, fact_failures INTEGER (mirrors v25)
```

**Extraction — new `hymem/dreaming/facts.py`:**
- `extract_facts(conn, session_id, llm, cfg, *, since_message_id)` reads the
  session's undigested tail (chunks starting above the facts watermark — the
  v24 watermark pattern, own column so facts and digest advance independently),
  makes ONE LLM call per session tail (`FACTS_PROMPT_V1` + user template in
  `extraction/prompts/__init__.py`, versioned module constant
  `FACTS_PROMPT_VERSION` like `PROFILE_PROMPT_VERSION`), returns validated
  items. Validation mirrors `validate_episode_items`: non-empty text ≤600
  chars, date ISO-or-null, entities through `canonicalize.normalize`, cap
  `cfg.dream_max_facts_per_session` (default 8). Over-cap arrays reject as a
  whole and hold the exact cursor for adaptive retry; they are never truncated.
- Persist one authoritative source outcome plus exact occurrence manifest.
  Successful replay publishes a new immutable result revision, retracts facts
  omitted from the replacement set, and resurrects identical payload keys
  deterministically without erasing history.
- **Prompt/config bumps replay first.** Every committed source unit is
  re-extracted on its stored exact boundaries/input; old units are never
  repartitioned under a changed character cap. Only after the replay walk and
  full authority audit may the new version append tail units.
- Idempotency: a quiescent current store makes no extraction call; duplicate
  first-pass/replay workers are generation/cursor-CAS checked and cannot fork
  or regress the ledger.
- Embeddings: batch-embed fact texts OUTSIDE the write lock (the phase1
  lock-free pattern); content-addressed `embedding_cache` reuse is free.
- `DreamReport.facts_extracted` / `fact_failures` persisted to `dream_runs`.

**Retrieval — additive tier in `query/augment.py`:**
- `FactHit` dataclass `{fact_id, text, fact_date, entities, session_id, score,
  why_retrieved, source_occurrences}`; `ctx.facts: list[FactHit]` has its own
  native candidate budget, then shares occurrence dedup and the finite final
  prompt packer with higher-priority standing/current evidence.
- `_fact_search()`: FTS5 over `narrative_facts_fts` (same `_FTS_SAFE` +
  `_fold_diacritics` query path as the other six FTS sites) + optional vec KNN
  + RRF. Only current rows with a complete, committed exact source/result/
  lifecycle proof can rank or cross a provider hook; retracted history stays
  in the DB but outside the active-only FTS shadow. Cap `cfg.facts_top_k`
  (default 8, 0 disables).
  `facts_enabled` (default True) is the master switch;
  `facts_extraction_enabled` (default True) gates the write side separately.
- Wired after `_fts_search`/`_vector_search`, before graph lookup; v1 does NOT
  feed entity matching (minimal diff).
- `ask.py render_context`: new section `=== FACTS (verified past events) ===`
  placed after KNOWN FACTS and ABOVE CONVERSATION EVIDENCE — facts lead, raw
  turns stay below as verification backup (the Acme lesson: a summary is never
  the only copy). Line form `- [2023-11-30] text` (`undated` fallback),
  snippet-capped; tail-truncation sheds it before STANDING RULES/USER PROFILE.
- Surfaces at landing: `HyMem.augment()`/`ask()` + the `hymem_augment` MCP
  rendering. Honcho `/search` response shaping is a documented follow-up, not
  part of this step.

**Risks:** extraction quality still needs empirical benchmark gating; immutable
history grows on long-lived sessions (bounded units/revisions/items); and every
native tier ultimately competes inside one finite token pack. Exact-occurrence
dedup and priority-aware packing prevent a source-equivalent fact from silently
displacing standing/current evidence, but do not create unlimited context.

**Tests (`tests/test_facts.py`, StubLLMClient + stub embeddings, mirroring
`test_digest.py`):**
1. extraction→persist round-trip (rows, canonical entities, watermark advanced
   to covered `end_message_id`);
2. new messages append exact cursor-successor units; immutable historical
   revisions/events remain byte-identical;
3. idempotent re-dream: 0 new rows, 0 extraction calls (stub call count),
   watermark stable;
4. prompt/config bump replays every old unit on its exact stored coordinates,
   with successful empty replacement retracting its prior active payloads;
5. parse failure → `fact_failures` +1, watermark held, retry succeeds;
6. cursor/generation CAS makes identical concurrent submission idempotent and
   rejects divergent first-pass or stale-replay workers;
7. tier surfaces matching facts; non-matching query → empty; `invalid_at` set →
   never surfaces, row retained;
8. **additive invariant:** with facts present, `message_hits`/`fts_hits`/
   `episodes`/`graph_facts` are byte-identical to the `facts_enabled=False`
   control;
9. pre-v26 store → tier degrades to [], extraction skipped, no crash;
10. `render_context`: FACTS above CONVERSATION EVIDENCE, below KNOWN FACTS;
    date rendering; snippet cap; truncation order (rules/profile survive, facts
    shed first among evidence);
11. `ask()` end-to-end sees the facts block;
12. both config flags off → no extraction, empty tier;
13. migration 026 on a pre-v26 DB: table + watermark + dream_runs columns
    created, existing data untouched, version-guard error path intact.

---

### Step 5 — scored confirmation (box; protocol, not a build) — **RAN 2026-08-04/05, VERDICT: READ OFF, WRITE ON**

> **REINSTATED 2026-08-02** — Step 4 built, so this ran as written below. The
> facts tier shipped defaulted ON and unmeasured on every benchmark; the
> pre-registered reading order (mechanism BEFORE score) is what kept the result
> readable. **The protocol below is the pre-registration; the verdict block that
> follows it is the result. Read them in that order.**

- **LoCoMo, n=800, `--fresh` (core changed → stores rebuilt), seed 0, canonical
  3×/24k.** Pre-registered reads, in order: (a) mechanism — facts rendered in
  contexts? gold-in-facts rate on the miss set (the instrument must move before
  the score is read); (b) multi-hop + temporal READ TOGETHER (cat-1 is ~113/800
  → ±4pp alone); (c) all-800 net vs the ±1.6pp churn floor — a lever moving
  both populations is gated on all 800 (the Step-3 lesson).
- **MSC recall run**, same protocol, treated as secondary (band ±4pp @ n=100).
- **LME full-500 seed-0 `--auto-ability` full-dream: NON-REGRESSION ONLY**
  (hold 70.0 ± band; MS floor 51.9; per-category <±5pp = noise). Do not tune
  against it.
- All runs: config recorded in metadata (adapter contract), paired reads via
  `locomo_flip.py` / `compare_recall.py`.

#### STATUS 2026-08-05 — RESULT: E1 measurably COSTS on LoCoMo; verdict `read off, write on`

Instrument: `benchmarks/facts_ab.py` (fire rate → McNemar FIRED vs NOT-FIRED,
the latter a built-in negative control).

**LoCoMo n=800 — the battery's only measured signal.** Fired-subset McNemar
**−2.9pp (b=10/c=24, z=−2.40, p=0.024)** with a **flat not-fired control**
(+0.9pp, p=0.45), so the arms differ by the facts flag and nothing else. The
all-800 net (−1.4pp, z=−1.72) had read as non-significant — the pre-registered
fired-subset read is what made the cost visible, and it is the reason that read
exists.

**Mechanism is DISTRACTION, not crowding.** The facts block is appended *before*
`total_chars=0`, so it adds context and displaces nothing: the ON arm saw a
strict SUPERSET of the OFF arm's context and still lost. This retires the
assumption that "additive, displaces nothing" is a harm-free property — it is
exactly the property this tier had while costing 2.9pp. **Every future additive
query-side tier inherits a RELEVANCE-PRECISION bar on top of faithfulness.**

**LME paired 2026-08-05 — NO benefit, and structurally unreadable.** Fire rate
99.8% ⇒ the not-fired control is n=1, i.e. **vacuous**; `gold_in_facts` only
5.1%; the MS sign is NEGATIVE (−6.6pp), opposite the retracted +3.0. The 66-flip
hand-check puts **100% of the net in COUNT/SUM questions**: an unconditional
top-8 with no relevance floor reads as a COMPLETE enumeration, so the model
counts the 8 facts instead of the raw turns below (both `_abs` abstentions lost
2/2; 2 of 4 gold-in-facts rows lost, including a current-vs-previous PB pair).
Self-containment strips the completeness / ordering / supersession structure
those questions need.

**Judge artifact found in the same dump.** 6/66 flips have BOTH arms refusing yet
score discordant (5 lost / 1 gained = −4 of −12), because the judge credits a
gold value recited incidentally *inside a refusal* — a judge-side `_lex_match`
trap. It leaves LME null either way but **strengthens** the count/sum reading
(residual −8 total, count/sum −11). **It is UNRUN on LoCoMo**, where the same
rate would land near z≈−2.1, so the record must not harden past *"costs on
LoCoMo in this regime."*

**MSC / BEAM / LME-MS were all under-powered ⇒ UNMEASURED, not null.**

**Historical Step-5 verdict (superseded by the v46 authority/default
alignment): read off, write on.** It used `facts_enabled=False` and
`facts_extraction_enabled=True`. This rested on **zero
measured benefit**, not on the harm — the store keeps filling so the tier can be
revived without a backfill. **Any revival must pre-register the count-gate
prediction.** E2/Plan C inherit the relevance-precision bar. The old cross-source
E6 heuristic is not revived merely because lifecycle fields exist: it remains
rejected as unsafe. Schema v46 now ships both sides on and supports only exact
same-unit replay; future benchmark work may revisit ranking/defaults but must not
silently pay for hidden memory.

Collateral: the `[:4000]` recorder bug had turned 50 faithful facts into
"inventions" (50/50 resolved faithful) ⇒ **G-F1's v4-flash 0.55–0.76 still needs
the same full-source recheck before the migration story is settled.**

---

### Step 6 — E3 adoption (one deliberate rebaseline; after Step 5)

> **CLOSED UNEXERCISED 2026-07-31 — no rebaseline spent, no default changed.**
> The precondition ("if M1+M2 passed") was not met: M1 FAILED on latency
> arithmetic and M2 read as parity rather than a win (Step 3 STATUS). Both
> defaults stand: `rerank_model="llm"`, `rerank_cross_encoder_model=mxbai`.
> The one deliberate rebaseline the review priced is therefore **unspent and
> still available** for a future item. Reinstate this step only if M1 is re-run
> and passes on GPU hardware — and if it does, carry the bge swap into that same
> commit on latency+parity grounds.

If M1+M2 passed (Step 3): flip `rerank_model` default `"llm"`→`"cross-encoder"`
and (if M2 passed) `rerank_cross_encoder_model` → `bge-reranker-v2-m3` in ONE
commit — the single deliberate rebaseline the review priced. LME/LoCoMo
non-regression is one shared confirmation run (Step 5 is cancelled, so there
are no runs to ride). Update the frozen-baseline table
(`longmemeval_roadmap.md` §1) with the new numbers. Suite green; adjust any
test pinning the old default.

---

### Step 7 — E2 per-entity observations (GATED on flip-watch green)

**Idea.** Hindsight's observation network: per-entity preference-neutral
summaries ("MedFlow: what it is, status, key decisions"), regenerated at dream
time only when the entity's evidence set changed. THE entity-centric query
shape ("where do we stand on X?") in real life.

**Architecture.**
- Migration `027_entity_observations.sql`:
  `entity_observations(canonical TEXT PRIMARY KEY, summary TEXT, source_hash
  TEXT, updated_at)` + FTS shadow. `source_hash` = member-set hash of inputs →
  reuse cache (unchanged entities cost zero calls on re-dream).
- `hymem/dreaming/observations.py`: per dream, entities whose evidence
  (episodes + facts + active edges mentioning them) changed since `source_hash`
  AND degree ≥ `observations_min_evidence` get ONE regeneration call
  (`OBSERVE_PROMPT_V1`, evidence-bound clauses reused from root.v4), cap
  `observations_max_per_dream` (default 20, most-recently-mentioned first).
- Augment tier `ctx.observations`, gated `observations_enabled` (default False
  until gated) AND `matched_entities` non-empty — the natural gate, no router.
  Additive SELECT (`canonical IN matched_entities`); rendered in `ask()` below
  USER PROFILE.
- **Constraint (mirrors the rules §0 constraint): NOT in `_anchor_facts`** —
  the digest cache must not couple to observation churn.

**Gate:** flip-watch green (prerequisite) + qualitative hand-read of 20
entities on the prod store + mechanical tests. **Tests:** regenerate-on-change
only; reuse-cache hit on quiescent re-dream; empty `matched_entities` → empty
tier; additive-invariant control; cap enforcement; pre-v27 degradation;
not-in-anchor assertion.

---

### Historical Step 8 — E4, E6, E7 (cross-source E6 later rejected)

**E4 temporal-range boost.**

> **FRONT-RUN RUN 2026-07-30 → G-E4a FAILS 2 of 3 on LoCoMo. DO NOT BUILD (b)
> AS SPECIFIED.** Instrument: `benchmarks/reldate_probe.py` (+ 34 tests), which
> carries a prototype resolver so the gate runs before `reldates.py` exists —
> the `fact_probe.py`/`FACTS_PROMPT_V1` pattern. No LLM, no embeddings, no
> store; seconds to run. Pre-registered G-E4a = fire rate ≥5% AND range
> precision ≥90% AND zero control fires.
>
> | population | n | fired | rate | vague-only | precision | control |
> |---|---|---|---|---|---|---|
> | locomo-questions (gating) | 1986 | 24 | **1.2%** | 61 (3.1%) | **20.8%** | **0 ✓** |
> | locomo-turns (non-gating) | 5882 | 375 | 6.4% | 217 (3.7%) | — | 0 |
>
> **Criterion 1 — fire rate 1.2% vs 5%.** This is the Track A verdict again: a
> correct boost that no query triggers is dead code with a config flag. Note
> `vague_only` (61) **outnumbers** resolvable (24) 2.5:1 — most temporal intent
> in the corpus is "recently"/"a while back"/"the other day", which carries no
> arithmetic. **No resolver improvement reaches that bucket**, so the ceiling
> here is ~3% even with a perfect parser, i.e. the criterion cannot be met by
> building harder.
>
> **Criterion 2 — precision 20.8%, and the WHY is the real finding.** The
> arithmetic is right and the axis is wrong. Misses are **one-directional: gold
> AFTER the range 15, BEFORE 4** (3 of those 4 are future `next month/year`
> windows). Worked example: *"Where did Caroline move from 4 years ago?"*
> resolves to 2020 — correctly — and the gold turn is dated **2023-06-09**,
> because that is when she *said* it. **A range boost matches an item's SPEECH
> time; a relative expression in a question is about EVENT time.** Every
> `n_units_ago` miss is that mismatch. It is a bi-temporal gap, not a resolver
> bug, and no amount of resolver work closes it. (A resolver *accuracy* problem
> would scatter misses on both sides — which is precisely why the probe measures
> the direction rather than arguing it.)
>
> **Criterion 3 — control PASSES 0/1901.** The one criterion E4 would have to
> get right to be safe, it gets right. Two false-fire classes were found and
> fixed during the run, both mine, both now regression-tested: `"the last week
> **of** August 2023"` is an ABSOLUTE construction that was resolving to a
> window a year off, and a stated in-text anchor (`"…as mentioned on November 6,
> 2023"`) was being ignored so the *anchor's* error was reported as the
> *resolver's* imprecision — the same instrument-defect class that made the G-F1
> date reading unusable.
>
> **What survives, and it is not a rescue.** Content-side fires at **6.4%** —
> above the gate — while the query side does not. Relative dates are abundant
> *inside stored turns* ("I moved here last week") and rare in questions. That
> is a different feature: **normalizing relative mentions at INGEST**, where the
> anchor is known exactly (the turn's own timestamp) and the resolved date is
> EVENT time, which is the axis the misses show is needed. It also lands on an
> artifact HyMem already has — bi-temporal `valid_at` (Phase 1, landed) — rather
> than on `messages.created_at`. **This is a candidate, not a decision:** it
> must clear its own front-run (what share of resolved mentions get a `valid_at`
> that retrieval can use?) before it costs anything.
>
> **LME arm RUN 2026-07-30 → fire rate 9.4% ✓, control 0 ✓, precision
> UNMEASURED. Verdict = INCOMPLETE, not PASS.** Criterion 1 genuinely clears on
> LME, and the corpus difference is real: LME ships an explicit `question_date`
> per question, so relative expressions have a usable anchor that LoCoMo's
> annotator-written questions mostly lack. **9.4% is the resolvable rate as
> reported** — `fired` already excludes vague-only rows, so subtracting
> `vague_only` (3.8%) from it double-counts.
>
> Two instrument defects the LME run exposed, both fixed + regression-tested:
> - **The probe printed PASS on 2 of 3 criteria.** `precision is None` was
>   written as satisfying criterion 2 — an UNMEASURED criterion counted as a met
>   one. `fact_probe.py` reports INCOMPLETE in exactly this situation and this
>   probe now agrees: three states (FAIL > INCOMPLETE > PASS), and a failing
>   *measured* criterion still outranks INCOMPLETE.
> - **"No gold in LME" is a LOADER bug, not a corpus property.** LongMemEval
>   stamps read `2023/05/20 (Sat) 02:21`; `load_lme` sliced `[:10]` and demanded
>   ISO, so every `haystack_dates` entry failed and every gold list came back
>   empty — the criterion most likely to fail reported as "n/a". `normalize_date`
>   now accepts slash/dot/prose forms, and the report distinguishes "zero rows
>   carry gold AT ALL" (loader failure, loud) from "no fired question had gold"
>   (corpus property). **LME does have dated gold; re-run to measure criterion 2.**
>
> Also added: **fire rate by question category**, because a rate carried by one
> annotator-designed category ("temporal-reasoning") is a property of the
> benchmark's mix, not of how people ask. The warning requires the top category
> to carry ≥60% of fires AND be ≥2× over-represented, so it does not trip on the
> largest category simply being largest (LoCoMo's category 4 is 42% of questions
> and 62% of fires — size, not concentration).
>
> **LME RE-RUN 2026-07-30 with the fixed loader → G-E4a FAIL, but a DIFFERENT
> failure from LoCoMo's.** n=500, fired 47 (**9.4% ✓**), vague-only 19, control
> **0 ✓**, **precision 80.9% (n=47) ✗**. By category: multi-session 15.8%,
> temporal-reasoning 14.3%, single-session-preference 10.0%, single-session-user
> 2.9%, knowledge-update 2.6% — the concentration warning correctly stays quiet,
> and **multi-session leading means E4 is NOT a temporal-reasoning-only
> feature**, which was the live worry about the 9.4%.
>
> **The mechanism is NOT the LoCoMo one, and the distinction decides what
> happens next.** LME misses split **5 after / 4 before — balanced.** LoCoMo's
> were 15/4, one-directional, which is the speech-time/event-time signature.
> Balanced misses mean the resolver is inaccurate on particular expressions: a
> different failure with a different, *fixable* remedy. The probe's axis warning
> requires `after ≥ 3× before` and correctly did not fire — but its SILENCE was
> read as agreement with the LoCoMo diagnosis, so the balanced case now names
> itself in the report (regression-tested).
>
> **Size of the gap:** 38/47. Reaching 90% needs **4 more questions**; σ at
> n=47 is 5.7pp, so 90% sits **1.59σ** from 80.9%. The criterion fails as
> pre-registered — that stands, and widening the floor post-hoc because the
> number came in close would be exactly the score-fitting the campaign contract
> forbids. But it is not a decisive failure, and the misses are readable for
> free (`--verbose`).
>
> **PRE-REGISTERED BEFORE READING THEM** (the G-F1 rule, transplanted): **one**
> resolver revision is allowed, and only on a VISIBLE defect class in the misses
> — an unhandled construction, a wrong window for a specific form. Tuning
> `_UNITS` tolerances until the score clears is NOT a visible defect and is not
> permitted. **A second failure banks E4's query-side boost dead.** Two defect
> classes of exactly this kind were already found and fixed mid-run (absolute
> `"last week OF August"`, ignored in-text anchor), so the prior that a third
> exists is reasonable rather than hopeful.
>
> **REVISION 1 SPENT 2026-07-30** (the one allowed; a second failure banks E4's
> query-side dead). Two visible defect classes from the miss hand-read, no
> threshold tuning:
> - **Directional qualifiers.** `before X` / `since X` denote HALF-OPEN
>   intervals and the resolver emitted a point window at X. Now
>   `before_*` → `[0001-01-01, end]`, `since_*` → `[start, anchor]`, applied
>   BEFORE the prospective check so `"before tomorrow"` stays retrospective.
>   Cue search is bounded (40 chars, nearest cue wins) so a `since` in another
>   clause cannot reach across the sentence.
> - **Prospective windows are a THIRD category, not a fire.** `tonight`,
>   `tomorrow`, `next week` resolve forward; no stored past item can fall in
>   them, so boosting is cost with no upside. They are excluded from the fire
>   rate AND from the control — a prospective question is neither a retrieval
>   range nor a marker-free question, and **this was a real hole: the LME
>   control read 0/453 partly because two `tonight` questions were sitting in
>   the FIRED bucket.** `has_temporal_language` still returns True for them.
>
> **A defect in the revision itself, caught by the existing tests:** bare
> `"from"` was in the forward-cue list, so `"the plan from last month"` became
> an open-ended range — and one such range then covered its gold by accident
> and *raised* precision. Removed. **A cue that widens a window can only help
> the score, so it must clear a higher bar than one that narrows it.**
>
> LoCoMo after revision: 24 → 21 fires, precision 20.8% → 23.8%, verdict
> unchanged (its failure is the axis mismatch plus a 1.1% fire rate, neither of
> which a resolver revision touches). Suite 934.
>
> **HOW THE RE-RUN MUST BE READ, fixed in advance.** Expected LME arithmetic:
> 3 prospective fires leave the denominator and 2 directional misses become
> hits → ~40/44 ≈ 90.9%. **That is NOT a meaningful pass.** σ at n=44 is 4.3pp,
> so 90.9% sits 0.2σ from the line; separating 80.9% from 90% at p<0.05 needs
> ~150 fired questions ≈ 1,600 LME questions and the corpus has 500. **The
> precision criterion is underpowered on this corpus by construction** (the
> variance-band lesson: the band scales √(p/n), so a small n needs a large
> delta). Most of the gain is denominator, not accuracy. **Decide E4 on the
> miss DECOMPOSITION, not on the threshold crossing** — irreducible axis misses
> ~4% of fires, prospective false fires ~6%, fixable construction gaps ~4%. The
> thing that killed E4 on LoCoMo is marginal on LME.
>
> **Status: (a) `hymem/query/reldates.py` NOT written; (b) augment wiring NOT
> written.** ~~The build now hangs on ONE free re-run:~~
> `python reldate_probe.py --dataset <lme_s>.json` with the fixed loader.
> Criterion 2 is the one LoCoMo failed at 20.8% on a **corpus-independent**
> mechanism, and LME's `haystack_dates` are session dates — SPEECH time, the
> same axis — so the prior is that it fails there too. If it does, E4's
> query-side boost is banked and the ingest-side candidate (relative mentions →
> bi-temporal `valid_at`) is what carries forward. If precision clears 90%, E4
> is a genuine PASS on LME and the LoCoMo failure is corpus mix.
>
> ---
>
> **RE-RUN 2026-07-30 → G-E4a reads PASS on LME (fire 8.8% ✓, precision 90.9%
> ✓ n=44, control 0 ✓) and still FAILS on LoCoMo (1.1% / 23.8%). Per the
> reading fixed in advance, the threshold crossing is NOT the verdict — the
> decomposition is, and it says the resolver is FINISHED while the ceiling is
> ARCHITECTURAL. Still not built.**
>
> All 4 surviving LME misses are **gold AFTER the range — one-directional**,
> the axis signature. Every balanced (resolver-error) miss the previous run
> showed is gone; revision 1 consumed exactly the fixable half. What remains is
> the speech-time/event-time gap, in the corpus that was supposed to be the
> favourable one.
>
> **The cross-corpus confirmation is the finding, not the LME number.** n=4
> one-directional is p=0.125 on its own — a prior, not a result (the probe now
> says so; see the small-n caveat). But the misses concentrate on the same two
> rules in both corpora:
>
> | rule | LoCoMo precision | LME misses |
> |---|---|---|
> | `calendar_last` ("last month") | 33% (3/9) | 3 of 4 |
> | `n_units_ago` ("a few months ago") | **0% (0/9)** | 1 of 4 |
> | everything else | 100% (2/2) | 0 |
>
> Two corpora, opposite verdicts, **the same two constructions failing** — and
> they are the two most common resolvable expressions. That is not corpus mix.
> Mechanism prediction it implies, checkable free from the existing JSON:
> **precision should decay with lookback distance**, because the further back
> the window, the more room between when a thing happened and when it was
> mentioned. `calendar_last` and `n_units_ago` are the long-lookback rules;
> `within_last_n` and `since_*` are the short ones and are the ones at 100%.
>
> **The gate measured the wrong quantity, and this is the honest gap.** Range
> precision = "the window contains the gold date." It says nothing about how
> many NON-gold items the window also contains. A month-wide window over a
> corpus of session dates may cover 10–20% of the store, in which case a ×1.5
> boost has no discriminative power at any precision. **Selectivity — the share
> of the corpus inside a fired range — is the quantity that decides whether the
> boost helps retrieval at all, and G-E4a never measured it.** It is free to
> measure (dataset-side, no store, no LLM) and it is decision-blocking for ANY
> range boost, including the ingest-side successor.
>
> **Disposition: E4's query-side boost over `created_at` is NOT built.** Not
> "banked dead" in E1's sense — the pre-registered gate was satisfied and the
> revision budget was spent honestly. It fails a different test: the two rules
> that carry the feature are the two that miss, and the miss is on the axis
> that a query-side boost cannot reach. **The carry-forward is the ingest-side
> candidate** (relative mentions inside turns → bi-temporal `valid_at`, where
> the anchor is the turn's own timestamp and the output is EVENT time), which
> content-side fire rates have pointed at since the first run: 5.2% on LoCoMo
> turns, above the gate, against 1.1% on its questions. It needs its own
> front-run, and that front-run must carry a selectivity criterion.
>
> **Instrument additions this run (REPORTING only — no resolver change, so no
> revision budget spent):** per-rule precision with a "broken dominant rule"
> warning (an aggregate can clear 90% while the most common construction is
> wrong and rare rules carry the mean), and a small-n caveat on the
> one-directional warning (4/0 arises by chance 12% of the time; 6/0 is the
> p<0.05 point). Suite 936.
>
> ---
>
> **SELECTIVITY 2026-07-30 — the measurement the gate was missing, and it
> CLOSES E4's query side on arithmetic rather than on judgement.** Built into
> `reldate_probe.py` (loaders now carry `corpus_dates`; no second script, no
> JSON round-trip, `normalize_date` shared). For each fired range: what share
> of that row's corpus falls inside the window.
>
> **Correction to the block above, forced by the per-rule denominators.** With
> `n` per rule now known, LME per-rule precision is `within_last_n` 19/19,
> `n_units_ago` 13/14 (93%), `calendar_last` 3/6 (50%), rest 5/5. So "the same
> two rules fail in both corpora" was half wrong: **`calendar_last` fails in
> both (33% / 50%), but `n_units_ago` is 0/9 on LoCoMo and 13/14 on LME.** The
> cross-corpus contradiction sits on ONE rule, not two.
>
> That rule is also the only one that was ever going to justify the feature,
> and selectivity is what settles it. LME medians: `within_last_n` 100%,
> `calendar_this` 100%, `since_n_units_ago` 100%, `before_day_word` 100%;
> `calendar_last` 14%, `n_units_ago` 10%. **Every precise rule is
> non-selective and every selective rule is imprecise.** 27 of 44 fired ranges
> (61%) cover >20% of their haystack; 22 cover 100%.
>
> **Precision and selectivity are independent, and a boost needs both.** A
> window can be 100% precise and 100% non-selective at once — "this year" over
> a corpus that is entirely one year is perfectly precise and boosts the whole
> store. G-E4a measured only the first, so it graded a global boost as a
> retrieval signal.
>
> **BOTH TAILS ARE DEAD, and a median cannot tell them apart** — this is the
> refinement the standalone script did not have. 0% selectivity is not the good
> end of the scale: a window covering NOTHING boosts nothing. On LoCoMo
> questions 13 of 21 fires are empty, which is the same event as `n_units_ago`
> scoring 0/9 precision — the window lands before the conversation starts. The
> probe now splits fires into narrow / wide / **empty** and counts only
> `0 < selectivity ≤ 20%` as a fire the feature gets credit for.
>
> **The arithmetic that closes it.** Selective fire rate = narrow fires ÷ n:
> **LoCoMo 5/1986 = 0.3%** (raw 1.1%); **LME 17/500 ≈ 3.4%** by the 61%-wide
> figure, or 4.0% if only the four 100%-median rules are dropped. Both routes
> land **below the pre-registered 5% criterion 1**. E4 does not clear its own
> gate once non-discriminating fires are removed — no judgement call, no
> retro-fitted threshold, just criterion 1 read against fires that could
> actually reorder something. The probe prints this counterfactual explicitly
> rather than silently re-scoring G-E4a.
>
> **The ingest-side carry-forward is NOT killed by this — do not carry the
> query-side reading over.** LoCoMo turns: raw 5.2% → 1.8% "selective", with
> 192 of 307 windows empty. But an empty window on the CONTENT side means the
> mentioned event predates the corpus, and a `valid_at` written from it is
> still a correct write; it is a coverage question, not a discrimination one.
> **What its front-run must ask: what share of resolved mentions land where a
> QUERY range can reach them** — i.e. measured against the query range
> distribution, not corpus density. The probe now prints that distinction on
> non-gating populations so the 5.2% cannot be read as a green light.
> Suite 940.

(a) New `hymem/query/reldates.py`: stdlib-only
relative-date resolver, EN+NL (yesterday, N days/weeks/months ago, last
week/month/year, "twee weken geleden", "vorige week", "between X and Y"),
anchored to a `now` argument (harness passes `question_date`, production
wall-clock); returns `[start, end]` or None. ~150 lines + a pattern-matrix
test file. (b) augment wiring: when a range resolves, items whose dates fall in
range get a score boost ×`cfg.temporal_boost` (default 1.5) on the
message/facts/temporal tiers with a `in_range:YYYY-MM-DD..YYYY-MM-DD` why-code.
**Boost, never filter** — out-of-range item sets stay identical (additive
invariant test). **Tests:** resolver matrix EN+NL with fixed anchor; boost-only
semantics; no-range query → byte-identical path; why codes; TR chronology
unchanged.

#### G-E4b — ingest-side `valid_at`: front-run PRE-REGISTRATION (written 2026-07-31, before any measurement)

*Nothing built, nothing run. This registers the criteria BEFORE the numbers, per
the standing contract. Zero LLM cost: `reldate_probe.py` already carries the
resolver, both loaders, `corpus_dates` and the narrow/wide/empty split, so C1 and
C4 are countable from existing JSON and only C2 needs hand-scoring.*

**STEP 0 — the free arithmetic pre-check, and it must run FIRST.** The
carry-forward names the consumer as *"where a QUERY range can reach them."* That
consumer has already been measured, and it is nearly dead: selective query fire
rate **0.3% LoCoMo / 3.4% LME**. For an ingest-written `valid_at` to change a
query's result you need BOTH (i) that query to fire a *selective* range and (ii)
the target item to be one that gained a `valid_at` from a resolved relative
mention (content-side fire 5.2% LoCoMo turns / 6.4% raw). Treating them as
independent and counting every co-occurrence as a win — both generous —

> **ceiling on the share of queries this can affect = 3.4% × 6.4% ≈ 0.22% (LME);
> 0.3% × 5.2% ≈ 0.02% (LoCoMo).**

**That is one to two orders of magnitude below criterion 1's 5% bar, so Fork A
below closes on arithmetic — before a probe is written, not after.** This is
E4's query side closing a second time by the same mechanism, and the reason it
was worth computing rather than assuming: *a correct feature that nothing
triggers is dead code with a config flag* (the Track A verdict, third
occurrence).

**The fork this forces, to be resolved before any criterion is measured:**

- **Fork A — consumer is query ranges: CLOSED by the pre-check above.** Record
  and stop. Do not build a probe to rediscover a multiplication.
- **Fork B — consumer is an existing `valid_at` reader.** `valid_at` is not
  only for range queries, and the shipped readers do not need one: typed-value
  **`value_supersession`** (orders versions by date), the **recency-dating
  clause over `message_hits`** (the largest shipped dating lever: +5.4pp overall
  / +11.5pp KU strict), and bi-temporal invalidation. **If the ingest write has
  value it is here, not in range retrieval.** Name exactly ONE consumer before
  measuring anything — E3's lesson, applied prospectively: confirm the path
  under test is reachable from the config the gate runs, or the gate returns a
  confident constant.

**Pre-registered prediction, recorded so neither outcome can be rationalized
afterwards: Fork A closes; if this proceeds at all it proceeds as Fork B.** And
Fork B is a *different feature* from the one E4 carried forward — it must be
argued on its own terms, not inherited on E4's momentum.

**Criteria, applicable only once Fork B names a live consumer:**

- **C1 — INCREMENTAL coverage, not raw fire rate. ≥5%.** `dreaming/dates.py`
  already extracts explicit dates, so the measurable is the share of turns
  gaining a `valid_at` they do **not** already have. The raw 5.2%/6.4%
  double-counts every relative mention sitting beside an explicit date in the
  same turn, and would read as a green light for work already done.
- **C2 — CORRECTNESS ≥95%, deliberately HIGHER than the query side's 90%**,
  on a hand-labelled sample of ≥50 resolved mentions stratified by rule. The
  asymmetry is structural: a query-side boost is transient and self-correcting,
  an **ingest write is permanent and supersession will defend it** — E1's
  finding generalized (a store write that fabricates beats no write only if you
  never read it). One class of error does disappear here (the anchor is the
  turn's own timestamp, exactly known, so G-E4a's anchor-error class is gone),
  but **`calendar_last` is 33%/50% precise across both corpora and that is a
  resolver defect which travels to ingest unchanged.** Per-rule reporting is
  mandatory; a rule under bar is **excluded from writing**, never averaged away
  by the rules at 100%.
- **C3 — ABSTENTION / no-harm, zero tolerance.** Unresolvable or ambiguous →
  write NOTHING; an undated turn stays undated. This is the `validate_facts`
  lesson restated (stamping the session date onto null-dated facts manufactured
  confident wrong data). Control population: turns with no relative mention must
  receive zero writes.
- **C4 — REACHABILITY (replaces selectivity).** Share of new `valid_at` writes
  that measurably change the named consumer's output. Measured, not assumed —
  this criterion exists because three separate E3 measurements read as PASS
  while being incapable of failing.

**Revision budget: one**, on a visible defect class only, same rule as
G-F1/G-E4a — threshold tuning is not a visible defect. A second failure banks
the ingest side.

**~~E6 cross-source supersession over facts~~ — CANCELLED 2026-07-30 with E1;
later REJECTED rather than implemented by v46.**
Worth keeping the reason visible: supersession is the
mechanism that would have *defended* a fabricated fact — closing the older,
correct row in favour of a newer invented one — so at 0.55–0.76 faithfulness E6
was not merely unbuildable, it was the amplifier. That is also why E6 must
not be conflated with v46's narrower authoritative replay. The old heuristic is
retained below as a rejected design record, not an intended implementation.

**Rejected E6 design (not built):** Extend the
`value_supersession.py` classify→group→compare pipeline to `narrative_facts`:
typed-value classification over fact text+entities, group by (entity, attribute
cue), compare `fact_date`; a newer contradicting fact closes the older fact's
`invalid_at` (the one mutable field). Conservative: cross-session with distinct
dates only (the LME same-date lesson); multi-valued never fires; audit line
`facts.supersede subj=.. old=.. new=..`; `facts_supersession_enabled` default
False → flip after a clean audit, mirroring the v3.1 guard. **Tests:**
cross-session numeric update closes old fact (tier stops surfacing, row
retained); same-date never fires; multi-valued never fires; flag-off no-op;
audit emitted.

**What v46 actually guarantees:** a successful correction of one previously
committed exact source unit publishes a new result generation atomically;
payloads omitted by that generation retract and later-identical payloads can
resurrect, while all prior revisions/events remain auditable. It never treats a
fact from another session or source unit as a correction, so conflicting
current facts may coexist. That precision avoids invented attribution but does
not solve the old E6 problem or eliminate the measured distraction risk.

**E7 usage-signal feedback** (long game). `ask()` post-pass: stdlib
token-overlap between the answer text and rendered items →
`retrieval_feedback(item_kind, item_id, used, query_hash)` (migration 028);
dream-time aggregation → bounded ranking prior ×[0.8..1.2] on the relevant
tier. Invisible to benchmarks; compounds in deployment. **Tests:** overlap
detector fixtures; boost bounds; cold-store no-op; zero writes when `ask()` is
unused.

---

### Campaign sequencing and run budget

| # | Item | Depends on | LLM cost | Gate |
|---|------|-----------|----------|------|
| 1 | E1 probe (`fact_probe.py`) — **RUN, VERDICT IN** | — | ~1,000 calls (v1 + v2) | **G-F1 FAILED on v4-flash** — faithfulness 0.55–0.76 vs 0.90, twice ✗ |
| 1b | **`G-F1b` revival gate — NEXT LLM SPEND** (one new-dreamer arm; protocol in Step 1) | ONE pre-registered candidate model (natural: gpt-oss-120b) | ~500 calls + hand-score | same criteria as G-F1 (faithfulness ≥0.90 strict); 0.85–0.95 at n≈40 → expand sample; **FAIL = E1 dead for real** |
| 2 | E5 anaphora — **BUILT, GATE PASSED, SHIPPED** | — (parallel day one) | none (heuristic) | 31-item eval **100%**, no-harm **0/12** ✓ |
| 3 | E3 measurements M1+M2 — **RUN, VERDICT IN** | — (parallel, offline) | M1 LLM arm only | **M1 ✗ latency (0.09× vs ≥10×; quality untested — ceiling handset); M2 = parity, not a win (NL +4/20 at p≥0.125, fails the effect size rescaled to n=20; EN −2/15)** |
| 4 | E1 build — **SUSPENDED** (spec retained verbatim) | **G-F1b PASS** | build only | builds iff G-F1b ✓ |
| 5 | Scored confirmation — **SUSPENDED with Step 4** | Step 4 | box runs | reinstated as written iff Step 4 builds |
| 6 | E3 adoption — **CLOSED UNEXERCISED**; nothing flipped, rebaseline unspent | Step 3 only (Step 5 suspended) | **0 (not spent)** | precondition unmet: M1 ✗, M2 parity — revisit only if M1 is re-run and passes on GPU |
| 7 | E2 observations — **needs re-spec** (was over facts) | flip-watch green **+ a faithfulness result (a G-F1b PASS supplies it)** | capped per dream | must clear `fact_probe.py`'s bar first |
| 8 | E4 — **query-side CLOSED on arithmetic**, not built; carry-forward **`G-E4b` now PRE-REGISTERED (2026-07-31)**, Fork A (query-range consumer) closed by its free Step 0 pre-check, Fork B (existing `valid_at` reader) unstarted. This historical row said E6 would revive with E1; cross-source E6 was later rejected and v46 implements only same-unit replay. | Fork B must name ONE live consumer before anything is measured | none spent (probe is LLM-free) | raw gate: LoCoMo 1.1%/23.8% ✗, LME 8.8%/90.9% ✓ — **but SELECTIVE fire rate 0.3% / ~3.4%, both under criterion 1**; **G-E4b Step 0 ceiling ≈0.22% (LME) / 0.02% (LoCoMo) of queries ⇒ Fork A dead**; C1 ≥5% incremental, **C2 ≥95%** (higher than query side: ingest writes are permanent), C3 zero-tolerance abstention, C4 reachability |

**Post-G-F1 campaign state (amended 2026-07-30 when `G-F1b` opened).**
Campaign E's generative half is **SUSPENDED, not closed**: G-F1's verdict is
conditional on deepseek-v4-flash as extractor, and one new-dreamer arm is
authorized on the recorded judgment that the narrative-fact tier is the largest
available lever against Hindsight (the E0 thesis — no other campaign item
addresses the middle-granularity gap). **G-F1b is the campaign's next LLM
spend**; a PASS re-activates Steps 4→5 and supplies E2's faithfulness
prerequisite; a FAIL banks E1 dead on a second model class — record and stop.
~~Alongside it: **E3** (M1 needs an API key, M2
blocked on the torch OOM) is now the only unblocked scored item, and it is
independent of everything E1 touched — it reorders a pool retrieval already
built and writes nothing.~~ **E3 RAN 2026-07-30/31 and is CLOSED with no config
change** (M1 ✗ on CPU latency arithmetic; M2 parity — Step 3 STATUS), so
**G-F1b is now the campaign's only live scored item**, not merely its next LLM
spend. ~~**E4** (temporal-range boost) was specified over facts' `fact_date` but is
re-specifiable over episode/message dates, which already exist and are not
model-generated; that is the cheapest surviving item.~~ **E4 front-run RAN
2026-07-30 and FAILED G-E4a** (see Step 8): message/episode dates are SPEECH
time and the queries ask about EVENT time, so re-specifying it over them was
the wrong move — measured, not guessed, at zero LLM cost.
**E7** is artifact-agnostic (it scores whatever the tier returned). **E2** is
the one to be careful with: per-entity observations are generative writes, so it
inherits G-F1's finding directly and must clear the same bar before it costs
anything.

Hard rules carried into this campaign: never suppress-filter on a routed
signal (boost ≠ filter); native tiers keep separate candidate budgets but share
final fusion/token packing; any material facts prompt/config change bumps its
generation and replays immutable exact source units before new tail work;
profile changes re-gate (profile precedent); judge posture frozen; LME A/Bs are
non-regression confirmations, never tuning signals.

---

## Plan D — state-anchor expansion (borrowed from MindCache, added 2026-08-14)

Reviewed 2026-08-14 against MindCache `collapsed_tree`
(faisalhussain-devs/MindCache): their decision-anchor retrieval seeds a
secondary lexical expansion from the currently-ACTIVE decision set, pulling
supporting rows that share no vector overlap with the query but overlap with
the current state. Full implementation plan (TDD tasks, exact files, stop
conditions): [`docs/plans/2026-08-14-state-anchor-expansion.md`](docs/plans/2026-08-14-state-anchor-expansion.md).

**Adaptation to HyMem:** the "current state" is not an LLM status label — it is
the bitemporal active edge set, the SAME predicate as the digest anchor
(`aggregate.py:829`, `status='active' AND derived=0 AND invalid_at IS NULL`).
Seed terms = subject canonical + predicate + object canonical (+ typed values),
run through existing `_fts_search` / `_vec_search` / `_rrf_merge`. Additive
only, budget-capped (≤5 rows / ≤400 tokens), never reorders or suppresses
existing rows. Zero schema change, zero prompt change, zero dream spend.

> **CORRECTIONS 2026-08-25, before any build — banked ahead of the numbers.**
>
> **1. The predicate quoted above is INCOMPLETE.** `_anchor_facts`
> (`aggregate.py:801-835`) leads with **profile rows** (via `load_profile`,
> which outrank graph edges and consume part of `cap`), and its edge clause also
> carries `AND pos_evidence > neg_evidence` with
> `ORDER BY pos_evidence - neg_evidence DESC, last_seen DESC, id LIMIT ?`. Seeding
> from the three-clause version quoted here would seed from a **different set than
> the digest actually uses**. Copy the full clause; seed profile rows too, as a
> separate source with its own counter so the probe can attribute hits.
>
> **2. Typed values are not on the edge row.** They live in `kg_evidence`
> (`value_text` / `value_numeric` / `value_unit` / `temporal_scope`), so
> "(+ typed values)" costs a JOIN against the C3 cost line. Skip them in v1.
>
> **3. This plan is COUPLED to Grove E2 and cannot be gated independently.** The
> `invalid_at IS NULL` clause is redundant with `status='active'` *except* on
> re-asserted edges — the exact population Grove E2 measures (see the amendment
> in the Plan E → E2 section). So the clause is a deliberate decision here, not a
> copy: run E2's Stage 0 probe first, and if its `anchor_delta > 0`, drop the
> clause in the anchor selector with the rationale recorded. A state anchor whose
> job is tracking what is *currently* true should not silently bar facts that were
> retracted and re-confirmed.
>
> > **RESOLVED 2026-08-27 — keep the clause, on a re-based rationale.**
> > `anchor_delta = 0` on both stores on both runs, so the copy is licensed. But
> > the 2026-08-27 re-run found the E2 premise broken by `8c6925c` (the writer
> > now clears `invalid_at` on any positive mention), so the licence rests on
> > **selectivity** — tombstoned edges are ~0.077% of active edges against a
> > ~0.376% anchor budget share, and the fix shrinks that further — **not** on
> > "the clause is inert". Copy the predicate verbatim, and copy this rationale
> > with it. **Tripwire:** if a measured store ever shows `retracted_share`
> > approaching `edge_budget_share`, the clause becomes live and this decision
> > reopens. See the STATUS 2026-08-27 block in the Plan E -> E2 section.
>
> **4. C2 and C4 are NOT harm gates.** As pre-registered below they assert
> "additive, displaces nothing, existing rows byte-identical" — which is
> **exactly the property the narrative-facts tier had while costing 2.9pp on
> LoCoMo** (Campaign E Step 5; mechanism was DISTRACTION, not crowding). Only the
> scored A/B can clear harm. Likewise **C1 is a proceed-gate, not a flip-gate**:
> reachability has already failed to convert here once (the LoCoMo
> message-vector probe bridged 3/54 true vocabulary gaps).
>
> **5. `select_anchor_edges` must NOT copy the SHARED CAP — separate caps for
> profile and edges (banked 2026-08-25, before the shadow probe ran).** The Grove
> E2 Stage 0 probe found `_anchor_facts` returns early at `aggregate.py:823-824`
> once profile rows fill the cap: on the box that is 22 profile rows against
> cap=20, so the edge budget is **0 of 8754 active edges**, and on LoCoMo conv-26
> it is 4 slots for 55 edges. A shared cap is correct for `_anchor_facts` — it is
> budgeting ONE PROMPT BLOCK, where profile rows genuinely should outrank edges.
> It is wrong for an anchor SELECTOR, which is a seed source: starving the edge
> seeds is not prioritization, it makes the tier inert.
>
> **This is a gate-validity issue, not a preference.** Copying the shared cap
> would have the shadow probe measure a tier seeded from 4 edges (LoCoMo) or none
> (box), C1 would read near zero, and Plan D would close FAIL-mechanism because
> of the digest's block budget rather than because state anchors do not work —
> a confident wrong answer, and the cheapest kind to avoid. `select_anchor_edges`
> therefore takes `edge_cap` and `profile_cap` independently and never returns
> early. Everything else in the predicate is copied verbatim.
>
> **6. The scored A/B needs an OFF-arm stratification and an A/A control.** The
> obvious "fired = the tier added a row the OFF arm lacked" conditions on the
> treatment arm's own output, routes the distraction harm into the control, and
> makes the run read VOID rather than FAIL. Full design in the Plan D + Grove E2
> plan (Item 3).

**RAPTOR interference — CLEAR (mirrors Idea A):** query-time read of active
edges; writes nothing; `_aggregation_search` writes a different ctx field and
shares no state; digest anchor read is dream-time and unaffected.

**Sequencing:** independent of Campaign E; probe is LLM-free, runs in parallel
with G-F1b without touching the dream budget.

**Pre-registered gate (verdict must cite numbers; band arithmetic applies):**
- C1 mechanism: anchored-only hit rate ≥ 5% of sampled queries (gold row
  reachable ONLY via expansion).
- C2 harm: 0 wrong-state pulls (invalidated/superseded edges excluded by
  construction; unit-tested incl. value_supersession'd edges).
- C3 cost: 0 LLM calls, ≤1 vector call/query, ≤5 rows/≤400 tokens, +100ms
  latency budget.
- C4 non-interference: existing rows never reordered; non-target categories
  within ±2σ on answerable.
- Verdicts: PASS → flip `state_anchor_enabled`; FAIL-mechanism → close, no
  score-chasing; UNMEASURED → extend sample once or keep shadow.
- MindCache's own evidence for anchors is "Observed in Development" (manual) —
  the reason this gate exists before any build spend beyond the LLM-free probe.

#### STATUS 2026-08-25 — **CLOSED, FAIL-mechanism** (LoCoMo conv-26, n=74)

Built D1-D4 (`hymem/query/state_anchor.py`, `benchmarks/state_anchor_probe.py`,
22 tests). Task 6 never built — the gate closed first. **Zero LLM spend.**

**The first run was an artifact and must not be cited.** It reported 2.7%
anchored-only reach. Both of those hits were **provenance-circular**: the gold
chunk WAS the chunk the seed edge had been extracted from, so the anchor
"reached" it by tautology. The probe as first written omitted two of D4's four
pre-registered corrections — the circularity exclusion and the baseline-coverage
denominator — and `test_probe_reports_the_anchored_only_hit` used
`c-evidence`, the seed edge's own evidence chunk, as gold. **The single test
certifying the mechanism was itself the tautology.** Both corrections were added
and control-verified (revert circularity exclusion →
`test_a_provenance_circular_gold_does_not_score` fails; revert the headroom
denominator → `test_a_saturated_baseline_reports_zero_headroom` fails).

**Corrected matrix** (headroom = queries whose gold the baseline does NOT
already hold; `sep` = the banked separate-cap config, `shared` = the digest's
profile-first squeeze):

| leg | hit_rate | within headroom | circular | wrong-state | p95 |
|---|---|---|---|---|---|
| sep + FTS | **0.0%** | 0/22 | 2 | 0 | 7.3ms |
| shared + FTS | 1.35% | 1/22 | 0 | 0 | 6.9ms |
| sep + vec | **0.0%** | 0/18 | 2 | 0 | 235ms |
| shared + vec | 1.35% | 1/18 | 1 | 0 | 183ms |

**The ceiling rules out the saturation reading.** `c1_ceiling` = 29.7% (FTS) /
24.3% (vec): the baseline covers 70.3% of gold (52/74), leaving 22 queries with
real headroom. So the low number is not the LME 99.8% pattern — there WAS room
and the mechanism did not reach it. This is what makes the run interpretable;
without the denominator the first number was unreadable in both directions.

**Verdict against the banked gate.** C1 **FAIL-mechanism**: 0.0-1.35% against a
5% bar on sampled queries. On the pre-registered denominator the separate-cap
legs REJECT H0 (true rate >= 5%) at exact-binomial p = 0.022 — a supported
FAIL, not merely a number under a line. The shared-cap legs (1 hit) do not
reject (p = 0.11), and they are the gate-invalid config anyway. C2 **PASS**: 0
wrong-state in all four combos. C3 **mixed**: FTS clean (0 LLM, <=5 rows /
298 tokens, 7.3ms); both vec legs blow the +100ms budget at 183-235ms p95, so
C3 holds for FTS only. C4 **never measured and never will be** — it gates the
Task-6 A/B, which FAIL-mechanism forecloses. That is the correct outcome, not
an open gap.

> **Do NOT re-read this on the headroom denominator.** Within headroom nothing
> rejects at any leg (0/22 → p = 0.32; 0/18 → p = 0.40). That denominator was
> not banked; adopting it after seeing the numbers would be post-hoc, and it is
> underpowered in both directions. The gate closes on its banked terms.

> **A prediction of mine that the data falsified.** Plan D correction 5 (banked
> 2026-08-25, above) argued the shared cap would starve the edge seed and make
> C1 read near zero for the wrong reason. The gate-VALIDITY half of that
> argument stands — a tier seeded from 0 edges cannot be tested. The empirical
> half was wrong: separate caps produced FEWER non-circular hits (0) than the
> shared cap (1), not more. The difference is one question and neither is a
> signal, so the honest reading is that the cap policy did not matter here at
> all — but the correction did not rescue the mechanism, and the record should
> not imply it might have.

**Third failure of reachability-to-convert in this project** — after the LoCoMo
message-vector probe (3/54 true vocabulary gaps bridged) and narrative facts
(perfectly additive, cost 2.9pp). C1 was pre-registered as a proceed-gate that
could only ever kill; it did.

**Known limits of this close, stated rather than discovered later.** (a) One
store, one conversation — the LME and BEAM legs of the pre-registered sample
never ran. (b) Only the CHUNK corpus was probed; `expand_over_messages` exists
and is tested but the probe never calls it, so D5's per-corpus requirement is
satisfied for one corpus of two. (c) Two blind spots named in the plan remain
structural: an anchor that makes rows the reader ALREADY had interpretable
scores 0 here, as does an anchor edge that simply IS the answer.

**Code kept, not deleted.** `hymem/query/state_anchor.py` imports into nothing
in production (Task 6 unbuilt) so it costs no runtime path, and
`test_the_predicate_matches_the_digest_anchor_row_for_row` is independently
useful: it fails if `_anchor_facts`' predicate ever drifts, which is a live
concern during the criterion-6 accrual window.

**Rejected MindCache items (recorded 2026-08-14, do not revisit without new
evidence):** topic taxonomy (graph already encodes structure), Leiden
partitioning (overkill at our scale), input denoiser (recall risk — could drop
exactly the turns benchmarks probe), query-time constitution prompt
(prompting artifact; authority order already covered by origin/confidence
mechanics), 600MB lazy cross-encoder (rerank tier already exists).

---

## Plan E — Grove borrows (borrowed from Grove Memory, added 2026-08-18)

Reviewed 2026-08-18 against Grove Memory ("Living Memory", Phoenix Grove
Systems, 2026, pgsgrove.com/papers/living-memory): a research-intelligence
memory with use-weighted retrieval strength (ACT-R family + access-diversity
weighting), principled exploration (labeled wildcards, UCB-selected), and
consent-gated schema formation. Evidence caveat: the paper withholds ALL
numeric parameters (§10) and reports zero quantitative results — "full-scale
semantic evaluation... in progress"; verified behaviors are qualitative.
Verdict on their evidence: **UNMEASURED**. We borrow mechanisms as hypotheses
to gate, never as validated results. Full implementation plan (TDD tasks,
exact files, stop conditions):
[`docs/plans/2026-08-18-grove-borrows.md`](docs/plans/2026-08-18-grove-borrows.md).

Four items survived the review; each is a separate, independently gated tier.

### E1 — Labeled wildcard slot in augment (mode-gated exploration)

> **DEFERRED 2026-08-25 — not built, and the selection mechanism as specced below
> is NOT IMPLEMENTABLE.** Two independent reasons, both structural:
>
> 1. **No substrate for the counter.** There is no retrieval/surface log table in
>    the schema, and `HyMem.augment()` runs on a connection with
>    `PRAGMA query_only = ON` (`hymem/api.py:191`) — the query path *cannot
>    write*. The "zero-prior-surface counter over recent augment calls" has
>    nowhere to live.
> 2. **The process-local alternative is a defect we already paid to fix.** A
>    module global keyed to nothing is exactly `_LAST_LEAF_SET`: readable only
>    within one process lifetime (so NULL on ~175/187 rows on a box that starts a
>    fresh process per dream) *and* silently cross-contaminating when one process
>    serves two stores. Schema v30 moved that watermark into the store;
>    re-introducing the same shape on the query path would re-open it.
>
> **Sequenced behind Plan D's verdict.** D is the same *kind* of bet — an additive
> query-side tier — and Campaign E Step 5 showed one of those can cost 2.9pp while
> displacing nothing. D's result tells us whether this class can pay here at all
> before a second one is built.
>
> **If revived, it needs two changes to this spec:** (a) a store-derived dormancy
> proxy (low `pos_evidence`, old `last_seen`, never surfaced in an episode) in
> place of UCB — deterministic, no counter, no writes; and (b) a **third placebo
> arm** in the gate (a random dormant row), because the C1 below cannot separate
> "the selection works" from "adding any extra row works". Without the placebo the
> gate reads PASS for an inert selector.

**Adaptation:** after the RRF merge in `query/augment.py`, when
`cfg.augment_wildcard_mode != "off"`, reserve exactly ONE result slot for a
relevant-but-dormant row: drawn from the shortlist tail / dormant band
(never-or-rarely surfaced, derived=0, low recency), appended AFTER the normal
rows, stamped `wildcard: true` in the row plus a `wildcards` note in the
augmented context. Never reorders, never suppresses, never displaces a normal
hit (paper P3: bounded tilt). Mode-gated: `off` (default, precision) /
`warm` (synthesis) / `hot` (digest-adjacent) — mirrors the paper's
mode-dependent temperatures. Selection is NOT random: candidates carry an
uncertainty bonus proportional to how under-observed they are (UCB-lite;
without a retrieval log, "under-observed" is approximated by a zero-prior-
surface counter over recent augment calls).

**RAPTOR interference — CLEAR (mirrors Idea A / Plan D):** query-time,
read-only, writes nothing; `_aggregation_search` writes a different ctx
field; digest anchor read is dream-time and unaffected.

**Sequencing:** independent of Campaign E; probe is LLM-free (lexical/vector
only), runs in parallel with G-F1b without touching the dream budget.

**Pre-registered gate:**
- C1 mechanism: in ≥5% of warm-mode sampled queries the wildcard row is
  relevant-but-dormant (gold-reachable only via the wildcard slot; shadow
  probe against `off` baseline).
- C2 harm: 0 — wildcard never displaces a normal row (test: normal rows
  byte-identical vs `off` mode), never reorders, absent in `off`.
- C3 cost: 0 LLM calls; ≤1 extra FTS/vec tail read; ≤1 row / ≤200 tokens.
- C4 non-interference: per-category answerable deltas within ±2σ on
  non-target categories.
- Verdicts: PASS → flip warm-mode default; FAIL-mechanism → close, no
  score-chasing; UNMEASURED → extend sample once or keep shadow.

### E2 — Recovery-rate gauge (read-only instrument)

> **AMENDED 2026-08-25, before any build.** Reading the actual bitemporal code
> changed three things about the spec below. Recorded here so the amendments are
> banked ahead of the numbers, not fitted to them.
>
> **(a) The signal exists with no schema change — as an ARTIFACT, not a design.**
> `phase1.py:654-666` resurrects a retracted edge to `status='active'` but never
> clears `invalid_at`, and **nothing in `hymem/` ever clears it on
> `knowledge_graph`** (only `rules.py:407` does, on a different table). So
>
> > **PREMISE RETRACTED 2026-08-27 — the bolded clause is FALSE** as of commit
> > `8c6925c` (2026-08-25 20:09 UTC). See the STATUS block at the end of this
> > section. Note also that this amendment's **"Decision: measure, do not fix"**
> > below was overtaken ~11 hours later the same day by that commit, from a
> > different thread, and the two were never linked.
> 
> `status='active' AND invalid_at IS NOT NULL AND derived=0` *is* the
> recovered population. It is also a live defect: `_anchor_facts`
> (`aggregate.py:828-830`) is the **only** query in the codebase that reads
> `invalid_at` on this table, so a re-confirmed fact is authoritative at query
> time and **invisible to the digest anchor**; and because `stamp_invalidation`
> (`bitemporal.py:76`) and `value_supersession` are both write-once, a later
> re-retraction leaves the *first* retraction's date in place permanently.
> **Decision: measure, do not fix** — the fix would change `_anchor_facts`'
> output, rekeying the root digest on every store and injecting a
> deploy-refusion into the live criterion-6 accrual.
>
> **(b) Stage 0 is a read-only probe, and the headline number is a COUNTERFACTUAL
> DIFF, not a row count.** A raw count of the recovered population overstates the
> impact, because `_anchor_facts` also requires `pos_evidence > neg_evidence`,
> takes only the top `cap`, and lets profile rows consume part of that cap — most
> recovered edges fail the evidence margin anyway. Run the real anchor query
> twice, with and without the `invalid_at IS NULL` clause, and diff the rendered
> lists: **`anchor_delta`** is the gate. `anchor_delta == 0` means the clause is
> inert and there is nothing to fix. Gate Stage 1 (the in-dream gauge) on it —
> instrumenting a population known to be empty is the unreachable-code-path trap.
> Pre-register a *rate*, not `> 0`: ordinary decay-then-re-mention churn satisfies
> any `> 0` bar on any store. Hand-verify ≥3 rows against `kg_evidence`
> polarity/`extracted_at` ordering, or the number cannot distinguish genuine
> recovery from value oscillation.
>
> **(c) Instrument the TRANSITION, not the stock; and one obvious confound counter
> is dead on arrival.** A phase-3 read of "currently active, previously closed" is
> a slowly-drifting cumulative stock that will be misread as a per-run rate — count
> the flip at `phase1.py:662` instead. And **do not use `invalid_at = last_seen` as
> a migration-015 backfill counter**: the same UPDATE that creates a recovery sets
> `last_seen = CURRENT_TIMESTAMP`, so on the numerator that counter reads 0 by
> construction. Detect backfill against the evidence trail instead (a genuine
> closure has a matching `kg_evidence` row with `polarity = -1`).
>
> Full build + measurement plan: see the Plan D + Grove E2 plan (Item 1).

**Adaptation:** count, over the store's bitemporal history, the fraction of
currently-active facts (`status='active' AND derived=0`) that were previously
retracted/superseded (`invalid_at` set on an earlier edge with the same
subject/predicate/object triple) and later re-asserted with fresh evidence.
Report `recovery_rate` in the dream summary log (runner output), NOT in the
digest payload — digest content hashes into the RAPTOR root cache id (Idea B
constraint). Pure SQL + counters; zero behavioral change. facts_ab-style
instrumentation: measures whether the retraction gate is too aggressive (the
paper's "system tunes its own doubt" — gauge ONLY, **no auto-tuning**;
auto-tuning conflicts with pre-registration culture).

**RAPTOR interference — CLEAR:** read-only SQL at dream time; touches no
aggregation table, no cache id, no augment path.

**Sequencing:** independent; LLM-free; can land any time.

**Pre-registered gate:**
- C1 mechanism: gauge reports a numeric fraction; unit test over a synthetic
  supersede→re-assert fixture returns the hand-computed value.
- C2 harm: 0 — read-only; store unchanged (test asserts no writes).
- C3 cost: one indexed query per dream.
- C4 non-interference: digest cache id byte-identical before/after.
- Verdicts: PASS → keep reporting in the dream log; FAIL-mechanism → fix.
  NO auto-tuning ever.

#### STATUS 2026-08-25 — Stage 0 RAN on two stores; **Grove E2 Stage 1 CLOSED, FAIL-mechanism**

Instrument: `benchmarks/recovery_probe.py` (read-only, LLM-free, `mode=ro`,
`PRAGMA data_version` guard). Artifacts:
`~/.hermes/benchmarks/recovery_probe_{box,locomo26}_20260825.json`.

| Leg | cap split | `anchor_delta` | recovered | verdict |
|---|---|---|---|---|
| Box `~/.hermes/hymem.sqlite` @ cap 20 (production shape) | 22 profile + **0 edge** | 0 | 2 of 8754 active (6 retracted, 0 unstamped) | **VACUOUS** |
| **Box @ cap 100 — the measured evidence, supersedes the row above** | 22 profile + **78 edge** | **0** (arms 78/78 identical) | 2 of 8754 | **INERT (strong)** |
| LoCoMo conv-26 (fresh) | 16 profile + 4 edge | 0 (both arms byte-identical) | 0 of 55 | **INERT-EMPTY** |

**The close rests on BOTH legs of the argument.** Re-running the box at the cap
the VACUOUS verdict itself prescribes gave a 78-fact edge budget with the 2
recovered edges present and `anchor_delta` still 0 — the strong INERT branch: a
non-empty recovery population the clause nonetheless costs nothing. And there is
no `evidence_backed` row anywhere: the box's 2 recovered edges are both
`no_negative_evidence` (pos>0, neg=0, no evidence trail — backfill or
cascade-deleted evidence, not quotable), and LoCoMo has 0 of 0. An in-dream gauge
would instrument an empty set on both stores, which is the unreachable-code-path
trap. **Plan D may copy the `_anchor_facts` predicate verbatim** — the coupling
recorded in the Plan D section is resolved in the "clause is inert" direction.

**Instrument amended mid-read (recorded because the amendment came AFTER seeing a
number).** The box leg exposed a degenerate criterion in the probe's own verdict
function: it returned the pre-registered `INERT` reading whenever
`anchor_delta == 0`, without checking the delta *could* have been non-zero. With
20 profile rows against `cap=20` the edge budget is 0 and the diff is 0 by
arithmetic, so the probe was reporting a result on a store that had measured
nothing. Two verdicts added, each with a binding negative control:
**`VACUOUS`** (`edge_budget <= 0` — cannot answer) and **`INERT-EMPTY`**
(`recovered == 0` — nothing to bar, a correct close for that store that does not
generalise to one where retractions fire). The strong `INERT` now requires a
non-empty recovered population that the clause still fails to bar. The
production-shape leg produced no strong INERT, but **the cap-100 re-run did**,
which is why the close is firmer than the first reading suggested. The honest
scope caveat still stands:
LoCoMo conv-26 is a one-conversation single-dream store where no retraction ever
fired, and the box could not measure the delta at production cap.

**Collateral finding, separate issue — the box digest anchor contains ZERO graph
edges.** `_anchor_facts` gives profile rows the whole cap first and returns early
when `remaining <= 0` (`aggregate.py:820-823`). The box has **22** active profile
rows against `aggregation_digest_anchor_facts = 20`, so **none of its 8754 active
edges reach the VERIFIED FACTS block**, and the graph has been silently squeezed
out of the root digest as the profile grew. `facts_hash` (`aggregate.py:1241`) is
computed over the returned block, so on this store the root digest cache id is
keyed on the profile ALONE — a graph change cannot invalidate it through the fact
block. The docstring's language is *prioritization* ("profile rows lead", "graph
edges fill the remainder"); the early return silently converts that into
*exclusion* the moment the profile meets the cap, which is an unreported
behaviour change on the LIVE digest. LoCoMo conv-26 shows the same shape in
miniature (4 edge slots for 55 edges). Unrelated to `invalid_at`; it is server
code and needs its own pre-registered gate — do NOT hotfix it in this branch.

#### STATUS 2026-08-27 — Stage 0 RE-RUN; **the premise broke underfoot, and the close is RE-BASED**

Two read-only re-runs, no torn-snapshot retries, artifacts preserved to
`~/.hermes/benchmarks/`.

| Leg | cap split | `anchor_delta` | recovered | retracted | verdict |
|---|---|---|---|---|---|
| LoCoMo conv-26 | 16 profile + 4 edge, anchor 4/4 | 0 | 0 of 55 | 0 | **INERT-EMPTY** (identical to Aug 25 — ingested snapshot, no dreams land there) |
| Box @ cap 20 (production shape) | 20 profile + **0 edge** | 0 | — | — | **VACUOUS** (degeneracy guard fires, as Aug 25) |
| Box @ cap 60 | 26 profile + **34 edge**, anchor 34/34 both arms | **0** | **0 of 9047** | **7** (all stamped, all inside the 30-day prune window) | see below |

**The Aug-25 strong-INERT reading is no longer reproducible, and the cause is
one of our own commits.** On Aug 25 the box read `recovered = 2` — edges 1872
and 2840, both `no_negative_evidence`, closed 2026-05-29. Today both are active
with `invalid_at` NULL and nothing re-retracted them. **`8c6925c` (2026-08-25
20:09 UTC, "phase1/phase3: clear invalid_at on re-assert and reinforce") cleared
them**; its own commit message names edge 1872 as the corrupted row. The box
pulled it with v33 (~17:30 UTC Aug 26) and the dreams since (#1319-#1324) ran
the fixed code — both rows self-healed via soft reinforcement (`pos_evidence`
+49/+37), which also explains the otherwise-odd absence of any new `kg_evidence`
rows: reinforcement increments counters without inserting evidence.

**What is retracted, precisely.**
1. The premise sentence in amendment (a) and in the probe's docstring
   ("nothing in `hymem/` ever clears it"). Corrected in both places.
2. The **reproducibility** of the strong-`INERT` branch — *not* the Aug-25
   measurement, which was validly taken on the store as it then stood. Post-fix
   the `recovered` stock self-drains on any live store, so the probe will report
   `INERT-EMPTY` in the healthy steady state forever. The strong branch requires
   a non-empty recovered population and is now **unreachable, not unmet**. That
   is the degenerate criterion the probe's own verdict function was amended to
   guard against, arriving one level up.

So **neither store now establishes "the clause is inert" as a mechanism fact**:
conv-26 is empty because it never retracted an edge, and the box is empty
because the fix healed the only two recoveries it had. Banking the
unreachable-code-path conclusion on today's numbers would bank it against the
wrong evidence base.

**What survives, on a NEW argument.** The clause still costs **0** anchor facts
on both stores under direct measurement — but the licence for Plan D must now
rest on **selectivity**, not on the writer never clearing the stamp. At cap 60
the box has 34 edge slots against 9,047 active edges and 7 tombstoned ones: the
barred population is **0.077%** of the edges against a **0.376%** budget share —
5x below it — and both arms returned the same 34 edges, so no tombstoned edge
came close to ranking in. This is the **E4 lesson repeated**: precision and
selectivity are independent, and precision alone never licensed the clause. The
clause is precise (it bars genuinely-retracted edges) and non-selective (the
barred set is far too small to reach the budget). Crucially, `8c6925c` pushes
this argument in the *favourable* direction — it shrinks the tombstone
population further — so the re-basing does not weaken the conclusion, it changes
what the conclusion is made of.

**Verdict: Plan D's "copy the `_anchor_facts` predicate verbatim" licence
STANDS, re-based from inertness to selectivity, and now carries a tripwire it
did not have before** — if `retracted_share` ever approaches `edge_budget_share`
on a measured store, the clause becomes live and Plan D inherits a real
decision instead of a free copy. Grove E2 Stage 1 stays CLOSED, but its stated
ground changes from "FAIL-mechanism, the population is empty" to "FAIL-mechanism
*by selectivity*, the population cannot reach the budget". Anyone re-reading the
Aug-25 close should read this block with it.

**The consequence amendment (a) feared did NOT materialise.** (a) declined the
`_anchor_facts` fix because it "would change `_anchor_facts`' output, rekeying
the root digest on every store and injecting a deploy-refusion into the live
criterion-6 accrual". `8c6925c` fixed the *writer* rather than the reader and
produced that same output change — arriving on the box after G-FLIP's 11:15
Aug-26 window had already closed. The first post-pull dream (#1319) came back
clean at 92.2% with `residual = 0`. No refusion spike.

**Process finding, and it is the reusable one:** a "measure, do not fix"
decision banked in a plan section does not bind a commit landing from another
thread. It was recorded as an *intent* rather than as a named code path, so
nothing connected `phase1._upsert_triple` to the E2 premise that depended on it.
Such decisions need to name the function they are protecting.

**Not done, deliberately:** no hand-verify (`evidence_backed = 0`, so the >=3-row
check does not apply); the probe's verdict function is **not** re-tuned to make
`INERT-EMPTY` mean more than it does; and no selectivity criterion is
retro-scored onto the Aug-25 artifacts.

#### STATUS 2026-08-25 — gate RAN, **fix REFUSED at S1-C1**; the defect stays open

Gate built and run: `benchmarks/digest_squeeze_probe.py` (Stage 0, read-only,
zero LLM) and `benchmarks/digest_squeeze_dump.py` (Stage 1 instrument,
read-only, LLM-LESS — prints prompts for hand-scoring, writes nothing, has no
output-path argument), 31 tests. The `_anchor_facts` fix was deliberately NOT
built: it was gated on Stage 1, and Stage 1 refused it.

**Stage 0 (box, cap 20).** 23 active profile rows fill the cap → edge budget 0
→ early return fires → the digest fuses **profile-only** (20 facts). Under
separate budgets the block would hold 40 facts. `EDGES_RESTORED = 20`. Verdict
**SQUEEZED**. conv-26: 16 profile → 4 edge slots, block 20 → 36, **NOT-SQUEEZED
but NOT a null control** (the shared cap still costs it 16 edges).

> **Two corrections to the numbers this section previously carried.**
> **(a) 8754 was the wrong population.** The anchor-eligible count adds
> `invalid_at IS NULL AND pos_evidence > neg_evidence`: **8,380 eligible of
> 8,772 active non-derived**, 392 excluded. Keyed on the wrong count, a store
> full of ineligible edges reads non-VACUOUS while restoration is structurally
> zero — the guard-encoding finding.
> **(b) The delivery figure is 20, not 8,380.** `edge_cap` is 20, so the pool
> only decides which 20 win the `pos_evidence - neg_evidence` ordering. Every
> "0 of 8,754 edges reach the digest" framing is true but implies the fix
> delivers thousands of facts. It delivers **twenty**.

**Stage 1 (box, hand-scored locally, counts only — digest text is personal
conversation content and never left the box).** ARM A is the byte-for-byte
production prompt, pinned by a fidelity test against what a real
`build_aggregation_nodes` dream hands the LLM, so this scores what ships today.

| # | criterion | bar | measured | result |
|---|---|---|---|---|
| S1-C1 | grounding gain | >=3/10 | **2/10** | **FAIL** |
| S1-C2 | faithfulness | 0 | 0/10 | PASS |
| S1-C3 | no regression | 0 | 0/10 | PASS |
| S1-C4 | cost | <=40 lines | 40/40 | PASS, **zero headroom** |

Both C1 gains sit in project/organization structure — the only material
difference between the arms; the other 8 slots are identity/activity claims
identical in both. **Verdict: the additive fix does not ship.** It failed by
ONE claim-slot at n=10, and the gate's banked honesty note applies in full:
this refuses, it does not demonstrate uselessness. Moving the bar now would be
post-hoc.

> **The reserved-edge-floor fallback is dead too, and for a stronger reason.**
> It DISPLACES the profile tail to make room for the same 20 lines that just
> failed to earn their place — taking on displacement harm for a benefit
> measured at 2/10, when the non-displacing variant already scored 0/10 on both
> harm criteria. Do not fall back to it.

**What S1 did NOT measure, and what therefore stays open.** The gate scored
digest *text* at one instant. It cannot see the second, independent
consequence, which both contracts state explicitly:
`config.py:205-206` — *"The root node's cache id includes a hash of this block,
so the digest regenerates whenever the anchor facts change"*; `aggregate.py:
1235-1239` — *"a changed graph (new fact, supersession) regenerates the digest
even when the tree's membership is unchanged ... the price of NOT doing it is a
digest pinned to stale ground truth."* On the box `anchor_facts` is profile-only,
so **a graph change cannot regenerate the root digest through the fact block.
That price is being paid silently today**, and S1-C1's refusal does not touch
it: staleness needs a criterion observed ACROSS a graph change, not two blocks
compared at one instant. Reopening on that basis needs its own pre-registration,
not a re-run of this one.

**S1-C4 sits ON its bar** (40 of 40) with an unbounded profile side (23 rows,
multi-valued slots accumulate). Whatever revives this must re-size the cap pair
first.

**Coupling for whenever a fix does land:** `recovery_probe.measure_recovery`'s
`remaining = cap - n_profile` mirrors the early return and must change in the
SAME commit, or Grove E2's `anchor_delta` silently goes wrong. The two new
probes have no such coupling (the Stage 0 probe imports production
`_anchor_facts` for its CURRENT arm rather than re-implementing it).

#### G-DS1 — digest-staleness gate, PRE-REGISTERED 2026-08-26, **CONDITIONAL / not runnable on the box today**

> **RE-MEASURED 2026-09-01 (free, read-only): still not runnable, and moving
> further away.** `digest_squeeze_probe.py` on the box store reads **28 active
> profile rows** against cap 20 → `remaining = 0` → **edge budget 0**, so
> G-DS1's population condition (c) `edge_budget > 0` is empty. The three
> readings taken so far are 22 (probe docstring), 23 (this pre-registration,
> 2026-08-26) and 28 (today) — which **confirms the banked mechanism
> longitudinally rather than by argument**: `SINGLE_VALUED_SLOTS` does not
> bound the accumulating slots, so "once a store's active profile crosses the
> cap it never comes back" is now an observed trend over ~a week, not a
> prediction.
>
> Same run, for the record: **9,536 eligible edges** (of 10,071 active
> non-derived) and **zero** reach the digest anchor; the separate-budget
> counterfactual would restore 20 of them; and 8 profile rows are additionally
> dropped by `load_profile`'s own tail cap — the separate, still-open defect the
> fix would not repair. S1-C1's refusal (2/10 vs a bar of >=3/10) is untouched
> by this; what has changed is only that the population it was refused over has
> grown by five rows.

Written before any fix exists, so it can never be shaped around one. Filed
**blocked**, not open: with F1 refused, F2 dead and F3 (below) incoherent, there
is currently no live fix for it to accept and no population on the box to run it
against.

**Established by reading the code, not by measurement.** `aggregate.py:801-836`
— profile leads, `remaining = cap - len(profile)`, `if remaining <= 0: return
profile`. `load_profile(conn, cap=20)` truncates at 20
(`user_profile.py:343`), so 23 active rows give `n_profile=20` → `remaining=0` →
ZERO graph edges. `aggregate.py:1240-1242` — `facts_hash =
sha1(facts_block)[:12]`, `root_id = _node_id(member_ids,
salt=f"{_ROOT_SALT}|{facts_hash}")`. So the root cache id is a function of
(member set, profile rows) ONLY: a graph change with stable membership and
stable profile leaves `root_id` untouched, the cache hits, nothing regenerates.
Both contracts' promise is broken. **This is a proof and needs no gate.**

**Correction to the framing this defect was carried under.** "A graph change
cannot regenerate the digest — the price both contracts say they are paying to
avoid" is right about the broken promise and wrong about the price, and the
difference decides which fix is even coherent. Under saturation the graph edges
are NOT IN THE FUSION INPUT AT ALL. The cached digest is therefore not stale
relative to its inputs — it is correctly cached over inputs that never contained
the graph. Invalidating the cache key on a graph change would re-run the fusion
with a BYTE-IDENTICAL prompt, paying a fresh LLM call for nondeterministic
variation on the same content.

**Consequence: staleness is DOWNSTREAM of the squeeze, not independent of it.**
It is a real, observable phenomenon only where edges reach the block, i.e. where
`edge_budget > 0`. This kills the one fix shape S1 had not already refused:

| fix | status |
|---|---|
| F1 separate budgets | **REFUSED** — S1-C1, 2/10 vs bar >=3/10 |
| F2 reserved edge floor | **DEAD, stronger reason** — displacement harm for a 2/10 benefit |
| F3 salt-only / cache-key (fold a graph-state hash into the root salt without changing the block) | **INCOHERENT under saturation** — identical prompt, pure LLM churn |

**The gate itself.**

*Population.* Consecutive dream pairs where (a) the root's member set is
identical across the pair, (b) the active profile row set is identical, and
(c) **`edge_budget > 0` in both dreams**. Condition (c) is what makes the gate
self-scoping: the population is EMPTY on the box today, non-empty on conv-26
(16 profile → 4 edge slots), and becomes non-empty on the box only if a cap fix
lands or the profile is pruned below 20.

*Arms.* Treatment = pairs where the top-`remaining` eligible edge slice changed.
Control = pairs where it did not.

*Criterion.* Treatment: `root_id` changes on >= 90% of pairs. Control: `root_id`
changes on **0** pairs.

*Predictions.* Under a working fix, treatment ~ N/N and control 0/N. Under the
defect **both arms read 0** — the arms are indistinguishable, and that
indistinguishability IS the finding. Control showing ANY change voids the run
(filter (a) or (b) is broken): the built-in vacuity detector, in the shape the
diagnostic-controls lesson demands.

*Gate on `root_id`, never on digest text* — text can coincidentally match; the
cache id is the mechanism.

*Instrument, per dream (read-only, zero LLM, counts-only reporting — digest text
never leaves the box):* `n_profile`, `edge_budget`, `facts_block`, `facts_hash`,
`root_id`, root member-set hash, and a hash of the top-N eligible edges under
the production ordering.

*Floor.* n>=5 treatment pairs, n>=3 control, mirroring the criterion-6 floor.

**Sequencing.** Land nothing here before G-FLIP resolves. Every candidate fix
changes `facts_hash` behaviour, which changes root regeneration frequency, which
changes aggregation reuse% — the metric the flip watch is accruing on right now.
A fix landed mid-watch contaminates the window the same way the launcher env
loss killed the last one.

**What this actually reopens.** Not a separate defect with its own repair: the
SQUEEZE, with staleness as an additional argument in that gate's favour. Any
revival must re-size the cap pair first (S1-C4 sits at 40/40, zero headroom) and
must move `recovery_probe.measure_recovery`'s `remaining = cap - n_profile`
(`benchmarks/recovery_probe.py:184-188`) in the SAME commit — that mirror is
currently CORRECT (`_anchor_edge_facts` guards `limit <= 0 -> []`), and an
F1/F2-shaped fix is exactly what breaks it.

### E3 — Trajectory-based resurfacing for retracted facts

**Adaptation:** when dreaming's re-assertion path re-proposes a
previously-retracted/superseded triple (same subject/predicate/object, fresh
evidence), attach the prior retraction reason(s) as `retraction_history` on
the new candidate — the paper's "resurfaces carrying its prior refutations as
questions". Reuses the EXISTING bitemporal predicate (`invalid_at`, retracted
status); the old edge stays invalid — never resurrected in place, no ported
status labels (house rule). Hook: the re-assert path in `dreaming/phase1.py`
(contradiction handling, ~line 669) and/or the evidence-accumulation retract
in `query/conflicts.py`.

**RAPTOR interference — CLEAR:** dream-time read of the old edge's retraction
record + a field on the new candidate; aggregation tables untouched; digest
anchor predicate unchanged.

**Sequencing:** BEHIND Campaign E — touches the dreaming phase-1 re-assert
path that Campaign E also modifies; lands after those changes settle, and its
own probe runs shadow-first.

**Pre-registered gate:**
- C1 mechanism: ≥80% of re-asserted triples with a prior retraction carry
  `retraction_history` (unit + fixture test).
- C2 harm: 0 — no behavioral change when no prior retraction exists; old
  edges never re-activated (asserted).
- C3 cost: one indexed lookup per re-assertion; 0 LLM calls (reason text read
  from the existing record, not regenerated).
- C4 non-interference: contradiction/supersession counters and digest content
  unchanged.
- Verdicts: PASS → keep; FAIL-mechanism → close; UNMEASURED → extend sample.

### E4 — Null-model threshold for the consolidation gate

**Adaptation:** candidate clusters surfaced by `consolidate_insights`
(`dreaming/runner.py:723`, phase2) must beat a null model: domain-label
shuffling (episode/domain membership permuted) recomputes the detection
metric distribution; candidates below the calibrated α (default 0.05, or the
measured spurious-cluster rate) are not surfaced. Statistical, offline,
LLM-free. Operationalizes the paper's false-discovery-scaling defense —
"coincidence grows superlinearly with corpus size" — for the RAPTOR
aggregation tier's own gate.

**RAPTOR interference — CLEAR with a sequence tie:** the gate SUPPRESSES
surface candidates only; never adds content, never changes digest content or
cache ids, never touches augment. But it sits ON the aggregation gate that
the Stage 3c flip decision governs — sequence E4's probe BEHIND the 3c
decision (same dependency as Plan C).

**Pre-registered gate:**
- C1 mechanism: null-model test rejects ≥95% of spurious clusters on a
  synthetic shuffled fixture; real surfaced clusters survive.
- C2 harm: 0 — suppression only, additive nowhere.
- C3 cost: offline shuffle+recompute, bounded runtime budget per dream.
- C4 non-interference: digest root and cache id byte-identical; surfaced
  cluster SET is a subset of today's (never a superset).
- Verdicts: PASS → flip gate on; FAIL-mechanism → close; UNMEASURED → keep
  shadow.

#### STATUS 2026-09-01 — front-run RAN (free, read-only), **FAIL-mechanism on two counts**

The RAPTOR sequence tie is discharged (Stage 3c flipped 2026-08-26), and E4 is
"statistical, offline, LLM-free", so its front-run cost nothing.
`benchmarks/consolidation_null_probe.py` on the box store, 70 eligible
`depends_on` edges, 49 subjects, 58 objects, **9 observed hubs**.

**1. The proposed null cannot move the statistic.** E4 specifies "domain-label
shuffling (episode/domain membership permuted)". The hub rule is `GROUP BY
object_canonical HAVING COUNT(*) >= 2` — a function of the OBJECT degree
distribution and nothing else. Permuting subject or domain labels rearranges
who sits under each object and leaves every degree exactly where it was. Over
**2000 permutations the statistic takes one distinct value: {9}**. A gate
calibrated on that null accepts and rejects precisely what it would have
without one, and prints a calibrated-looking α while doing it. Measured rather
than argued, because that is the difference between this and the retracted §4.2
in the gold-delta protocol.

**2. Against a null that DOES reach the statistic, the graph is below it.**
Reassigning each edge's object uniformly over the observed vocabulary, 20,000
trials: null mean **19.74**, p05/median/p95 **17/20/23**, observed **9**,
P(null ≤ observed) = 0.0000. The store is *more dispersed* than chance — most
objects are named once. There is no excess of shared dependencies for a
false-discovery gate to filter, and suppressing at α = 0.05 would delete all
nine candidates, the real ones included. **The premise E4 imports — "coincidence
grows superlinearly with corpus size" — describes a regime this corpus is not
in.**

**And a finding about the detector itself, which outlives E4.** "Objects with
≥ 2 edges" is not a concentration measure, though "hub" invites reading it as
one. It peaks at an even 2-pairing and falls off on both sides (60-63 edges,
3000 trials): a star of 3 objects scores 3 against a null of 6.00; a 2-regular
graph scores 30 against 17.90; a matching scores 0 against 15.80. **A genuine
hub — one object everything depends on — makes this statistic SMALLER.** So
EXCESS here would mean "more evenly paired than chance", not "more clustered
than chance", which is a mismatch with the coincidental-*cluster* framing E4
was built on. Pinned by a test, because the verdict cannot be read correctly
without it.

**Verdict: FAIL-mechanism → CLOSE.** Not "no effect measured" — the specified
null is inert by construction and the corpus is on the wrong side of a null
that works. Revival needs a new statistic (distinct-subject counts, or a real
concentration measure) and a corpus where the deficit has reversed, argued in a
fresh pre-registration. C2/C3/C4 were never reached and are not evidence of
anything. The probe stays as the instrument that would notice the reversal.

*(Also pinned, latent on the box today: `COUNT(*) >= 2` counts EDGES, so one
subject with two edges to the same object renders as "a shared dependency of:
X, X". All nine current hubs have distinct subjects, so this is not biting —
the test exists so a future store that hits it is not read as a real hub.)*

This is the third Grove item to close FAIL-mechanism on its own front-run (E2
2026-08-25, E3 on population, E4 today), which is the borrow discipline working
rather than failing.

**Rejected Grove items (recorded 2026-08-18, do not revisit without new
evidence):** dual-space structural-signature distillation (protocol
unpublished, quality ceiling unproven at scale — the paper's own limitation;
would touch frozen judge posture), schema taxonomy/annotator lifecycle
(changes ingestion encoding; consent-gated promotion machinery unjustified at
HyMem's scale), full UCB exploration machinery (behavior under skewed query
distributions uncharacterized — the paper's own limitation; the
labeled-wildcard discipline is the transferable part, E1), auto-tuning of the
recovery gauge (conflicts with pre-registered gate culture; gauge-only per
E2), checkpointed worldview re-derivation/diff (bitemporal store already
provides audit; full re-derivation cost unjustified), MDL compression gate
(specified but not operationalized even by the paper; no corpus-encoding
machinery to apply it to).

---

## Judge instrument audit + the `invalid_at` lifecycle (added 2026-08-25)

Two threads ran on 2026-08-25 after Plan D, Grove E1/E2 and the digest-squeeze
gate all closed. They are recorded together because the second was found while
choosing what should follow the first, and the first is the reason the second
was fixed rather than deferred a second time.

Neither thread changed a canonical number. That is the point of both.

### Part 1 — the judge instrument (D1-D4)

**Why it was audited at all.** `longmemeval_adapter.judge_answer` is the single
function behind LoCoMo 68.2%, LME 68.4% and MSC ~84.0% — `locomo_adapter.py:421`
and `msc_adapter.py:502` both import it. (BEAM is NOT in the blast radius:
`beam_adapter.py:725` defines its own rubric judge returning a dict.) Until this
run it was two lines, and it **discarded `raw`**, so its defect rates were not
merely unmeasured but unmeasurABLE from any stored run. Recording `raw` is the
whole reason `benchmarks/judge_audit.py` exists. All four criteria were
pre-registered in that module's docstring before its first LLM call; the
docstring is the authoritative ledger and this section is the summary.

**The run.** LME, 500 rows, 500 judgeable, 30 `_abs`, 112 non-`_abs` refusals,
0 judge-side `LLM_ERROR`, v4-flash at temp 0.0 / `max_tokens=10`.

| # | criterion | bar | measured | band |
|---|---|---|---|---|
| C1 | misscore (shipping vs reference rule disagree) | ≥1.0% MATERIAL | **0.00%** (0), C1b 0 | IMMATERIAL — *as a lower bound* |
| C2 | non-compliant replies (indicator only) | no bar | **0.00%** (0) | every reply a bare yes/no |
| C3 | refusal scored CORRECT (non-`_abs`) | ≥2.0% MATERIAL | **2.13%** raw → **1.87%** corrected | **WATCH-at-bar** |
| C4 | arm refusal-rate asymmetry | ≥1.0pp | **NOT RUN** | blocked |

**C3 is recorded as WATCH-at-bar, NOT as "below material".** Raw 10/470 has a
Wilson 95% CI of **1.16–3.87%**; the hand-check FP rate 3/25 has **4.17–29.96%**.
The 2.0% bar sits inside both. Breakeven FP for MATERIAL is exactly **6.00% =
1.5 of 25**, so **one hand-scored row moves the verdict**:

| FPs / 25 | corrected C3 | band |
|---|---|---|
| 0 | 2.128% | MATERIAL |
| 1 | 2.043% | MATERIAL |
| 2 | 1.957% | WATCH |
| 3 | 1.872% | WATCH |

> **The verdict was flipped by a defect in the instrument, not by the data.**
> `write_handcheck_sample` drew both arms from ALL rows while `build_report`
> divided by non-`_abs` only — 6 of 25 slots spent on rows the criterion never
> counts, on a run where the reader refused 28 of 30 abstention questions. The
> pre-fix sample gave FP 1/19 = 5.26% → **2.016% = MATERIAL**. Redrawing from
> the population the rate is divided by gave 3/25 = 12% → **1.872% = WATCH**.
> Divisor/population parity is not bookkeeping; it decided this verdict.

**C3's numerator decomposed (hand-read, all 10 rows).** Every one is the
containment criterion doing exactly what its prompt says — **zero genuine judge
errors**, so no judge-prompt change was warranted on this evidence. 5 pass the
strict recitation test. The other **5 are `recites_gold` FALSE NEGATIVES**, and
the mechanism is worse than "strictness biases down": the `len(t) > 2` filter
**discards the numerals** — the entire payload of a temporal-reasoning gold —
while **mandating the trailing "also acceptable" gloss**. One row states both
"22 days" and "21 days" verbatim and still fails; one fails on the single word
"taking"; the preference row is a paraphrase the rubric's "recalls and utilizes
personal information" clause credits.

> **`recites_gold` token rule — PRE-REGISTERED AS ITS OWN GATE, not folded in.**
> It also sizes `free_precheck`'s spend-licence ceiling. With the five FNs
> counted, the 2026-08-25 ceiling numerator is **≥21/470 = ≥4.47%**, not 16/470
> = 3.40%. Direction is **conservative for the licence** — an under-count can
> only wrongly REFUSE a spend, never wrongly license one — and this spend was
> REACHABLE under either count. Any loosening changes the licence arithmetic as
> well as the C3 footnote. Gate it separately. **UNRUN.**

*(Pre-registration above is banked verbatim and is not edited. Resolution, added
after the fact: the gate was built 2026-08-26, RAN the same day, and **FAILED
R2** — 2 FP against a bar of ≤1. The alias stayed at `recites_gold_v1` and the
rule was not re-tuned. Note the direction argument quoted above holds only for
v1: loosening inverts it, which is why R2/R3 existed at all. See "A5 RAN".)*

#### D2 — CLOSED BY FIX, in its only inert window

`judge_answer` now calls `longmemeval_adapter.parse_judge_verdict`: word-boundary
`\byes\b`/`\bno\b`, first verdict token wins, negated affirmatives and the
`[LLM_ERROR: ...]` sentinel score False, no verdict token fails closed.

**Landed on the strength of being a PROVEN no-op**, not on the strength of being
right: C2 = 0.00% over 500 recorded replies, C1b = 0 negated-yes, 0 sentinels.
Each of the three rules is inert on the corpus on its own evidence.
`judge_audit.py --verify-parse` re-scored all 500 stored replies under the frozen
legacy rule and the live one: **0 verdicts change, 0 could have changed**
(349 compliant-yes / 151 compliant-no). No canonical number moved; LoCoMo, LME
and MSC are **not** re-baselined.

> **Why now and not at the next migration.** deepseek-chat's hard-deprecation
> already forced one judge migration and cost 1.6pp of judge harshness. At the
> next verbose judge, the decision rule and the data would change in the same
> step with no way to separate them. The insurance is only free once.

**Deliberately NOT changed, both pinned by test:**

- **"yes and no" still scores True.** Not merely "criterion, not parse": rule 3
  IS `judge_audit.reference_verdict`, banked pre-run, and that identity is the
  **only warrant that C1 = 0.00% certifies this function** rather than something
  merely like it. `reference_verdict("yes and no")` → True. Flipping it breaks
  the certification chain. Resolving a hedging judge is D1 territory.
- **A truncated fragment carrying a bare "yes"** ("...whether a yes would be")
  still scores True. Needs the reply's structure, and `max_tokens=10` is part of
  the frozen comparability contract.
- **Residual, legacy-identical, no regression:** stacked modifiers
  ("never really a yes") pass the tightened negation regex. Unfixed family
  member, not drift. No measured pressure to widen a decision rule.

**D3 — HALF-fixed, deliberately.** Containment rode along (provably inert, 0
sentinels): a sentinel can no longer score CORRECT, which closes the
`[LLM_ERROR: unexpected token 'yes']` hole the legacy rule scored True.
**Visibility did not**: `judge_answer` returns a bare bool with no channel for
"the judge never answered", and giving it one decides what five call sites across
three adapters do **mid-outage** — not two lines, and not inert. Stays pinned as
a defect.

**C4 — blocked, recorded not dropped.** No scored LoCoMo run pair exists on the
box; the only conv-26 artifact is a `--diag-only` dump (`correct=null`, empty
`ai_answer`, no reader calls by construction) which would have audited nothing.
Recovering the pair is 1,600 reader calls — not proportionate.

#### Instrument lessons that generalise beyond this audit

1. **A "we verified it changed nothing" claim needs the denominator of things
   that COULD have changed.** `--verify-parse` reports flips AND
   `rows_that_could_flip`, because on an all-compliant corpus 0 flips *cannot
   fail* to be 0. Without the split it is a certificate signed by an instrument
   that never met the surface it certifies — the E3 trap reappearing inside the
   verification of a fix for it. Pinned by
   `test_a_zero_flip_result_on_a_compliant_corpus_is_reported_as_VACUOUS`.
2. **A baseline copy must be frozen, and the freeze needs a test.**
   `judge_audit.shipping_verdict` is now the pre-fix rule that actually produced
   the three canonical numbers. Re-syncing it to production "for consistency"
   would make the before/after diff a constant zero **by construction**. One
   test exists solely to fail on a well-meant tidy-up.
3. **A counter and a decision rule need different tightness.** The audit's
   `_NEGATED_YES` over-matches on purpose (inflating a lower-bound bucket is
   conservative); the landed rule cannot (over-matching marks correct answers
   wrong). The join test asserts **both** halves so a re-sync fails loudly.
4. **Negative controls by reversion, one guard at a time.** Seven guards, each
   reverted individually; each failed exactly its own test. **This is how a
   vacuous test of mine was caught**: `test_verify_parse_reads_raw_not_the_
   recorded_verdict` used a fixture that passed under *both* the correct
   implementation and the bug it was named for. Rewritten with a deliberately
   corrupt fixture. A guard whose test cannot fail is unguarded.

**Commits:** `5acb05b` (C3 `_abs` cross-tab + `c3_spend_licence`) → `2ec2ed8`
(hand-check population parity) → `9408b76` (free-path writer delegates) →
`454d393` (post-run ledger) → `4369fe2` (the parse fix) → `5884adf`
(`recites_gold` gate pre-registered) → `a025151` (certification-chain note).

### Part 2 — `knowledge_graph.invalid_at` was never cleared on re-assert

Found while sizing what should follow the audit. Previously flagged in the
digest-squeeze plan as a sibling defect deferred on a deploy-cost claim that was
itself retracted as overstated by a large factor.

**The defect.** `phase1._upsert_triple` resurrected a retracted edge with
`status = CASE WHEN status='retracted' THEN 'active' ELSE status END` — and
`invalid_at` appeared **nowhere in `phase1.py`**. Across all of `hymem/`, the
only site that ever cleared an `invalid_at` was `rules.py:407`, for the `rules`
table; `knowledge_graph.invalid_at` had **no clearing path at all**
(`bitemporal.py` has `stamp_validity` / `stamp_invalidation` and no inverse).

**Two live readers consume exactly the violated conjunction:**
`query/state_anchor.py:70` and `dreaming/aggregate.py:829`, both
`status='active' AND derived=0 AND invalid_at IS NULL`. So an
asserted → contradicted → **re-asserted** edge was permanently invisible to
retrieval and to the digest's VERIFIED FACTS — in the one state where its
evidence is strongest.

**Second writer with the same hole:** `phase3.reinforce` cannot resurrect (its
SELECT filters retracted out) but bumps positive evidence on already-corrupted
active rows — reinforcing guaranteed invisibility.

**Sizing (free, counts only, before any fix).** Box: **2 rows**, both
`derived=0`, i.e. exactly the population both readers consume; conv-26: 0
(latent there). Not hypothetical and not historical: edge 1872 carried
`last_reinforced = 2026-08-25 18:12:59` — phase3 was bumping it into invisibility
**that evening**. Both rows are `user rejects <X>` — explicit user rejections of
system behaviour, suppressed since 2026-05-29.

> **Claim corrected in passing:** the "growing since the 2026-07-02 supersession
> flip" framing is NOT evidenced by this store — both stamps predate it. The
> supported claim is narrower and sufficient: nothing ever cleared the field, so
> the population is **monotonic**.

**The fix (`8c6925c`, TDD, 3 RED → GREEN).** `invalid_at = NULL` in phase1's
positive branch and in `phase3.reinforce`, mirroring `rules.py`'s re-assert
contract. **Unconditional, not gated on `status='retracted'`**, so a row that
already died corrupted self-heals on its next positive mention — pinned so it
cannot be "safely" re-gated later. Negative polarity untouched (a negative
mention must not resurrect a closed edge), pinned. 4 tests in
`tests/test_bitemporal.py`.

**Live-store repair, 2 → 0.** Authorised on three checks, in this order:
(1) **the invariant holds by construction** — all three `invalid_at` writers
(`api.py:805-816`, `phase3.py:127-133`, `value_supersession.py:233-240`) set
`status='retracted'` in the same breath, so `active AND invalid_at IS NOT NULL`
is unreachable through any correct path and every matching row is corruption;
(2) **the one reading that would make the UPDATE destructive is unrepresentable
here** — a general bi-temporal model can hold an active fact with a closed
world-time interval ("lived in Berlin 2019–2021"), but in this schema
`invalid_at` is only ever stamped *as part of* a retraction; (3) **it clears a
cache, not a fact** — `stamp_invalidation` re-derives the date from `kg_evidence`,
which the UPDATE never touched. Backup via `.backup`, **not `VACUUM INTO`**
(VACUUM renumbers rowids, and rowid drift against the vec/FTS shadows is one of
the four causes behind the RAPTOR reuse failure). Anchor-eligible count
8896 → 8898, effective immediately rather than on next mention.

**Unstated benefit, worth keeping findable.** `value_supersession.py:237` is
`COALESCE(invalid_at, ?)` — write-once. Before this fix, a
resurrected-then-re-superseded edge kept its **first** close date, so the
interval claimed "invalid since T1" while the edge had actually been valid again
between T1 and T2. Clearing on re-assert means the COALESCE now sees NULL and
stamps a fresh, correct date. The fix repaired the interval semantics for
oscillating edges, one layer below the visibility bug it was written for.

**Watch note landed (`910df08`, `config.py:463`).** A positive re-mention now
clears `invalid_at` and reactivates the old value, so both values are active
until the next dream's supersession re-retracts the loser by `valid_at`.
`bitemporal.supersede` can therefore legitimately fire **twice on the same
pair**. The rollback signal for that watch is *a `prefers`/multi-valued row* —
**NOT** a double firing. Do not roll the feature back on a double firing: that is
the guard converging, not failing.

#### A both-active probe was proposed and REFUSED — on the same trap

Post-dream both-active would be **0 by construction**: `supersede_competing_
values` runs at `runner.py:715`, *after* phase1 persist (283-387) and
`phase3.reinforce` (699), so a re-mention is resolved inside the same pass that
created it. A probe sampling after a completed dream **cannot report non-zero** —
a ceiling instrument, the exact shape the vacuity split above exists to prevent.

It is degenerate from the other side too: `value_supersession.py:195-196` skips
every object `_classify_object` cannot parse, so free text never competes. A
naive "count same subj/pred active pairs" would return a large **permanent**
population that is not a defect in any sense. To be correct the probe would have
to reuse supersession's own grouping — at which point it *is* supersession, run
immediately after itself, returning 0.

**The signal that would actually matter is RE-RETRACTION** — the same edge
superseded across more than one dream — and it needs no new code: the
`bitemporal.supersede` audit line already carries `subj`/`pred`/`old`, so
repeated triples across dream runs are the count. Build an instrument only if
that is ever non-trivial, and build it against observed cases.

### What stays open after 2026-08-25

| Item | State |
|---|---|
| **D1 / C3** | **WATCH-at-bar, band NOT resolved.** Pre-registration says record, do not re-baseline on it alone. Reviving needs a new gate; note the numerator is containment-by-design with 0 judge errors, so a judge-prompt fix has no measured defect to fix. |
| **`recites_gold` token rule** | Pre-registered `5884adf`, **UNRUN**, free. Changes the licence arithmetic, not just the C3 footnote. *(State as of 2026-08-25. Built 2026-08-26; RAN the same day and **FAILED R2** — see "A5 RAN" below. Closed, alias stays v1.)* |
| **C4 arm asymmetry** | Blocked on a scored LoCoMo run pair that does not exist. |
| **D3 visibility** | Pinned defect. Needs a channel `judge_answer`'s bool return does not have. |
| **E6 supersession over facts** | **Historical status, superseded 2026-09-04:** v46 safely implements same-source-unit replacement history, but the cross-session typed-value/date heuristic described by E6 is rejected and not built. |
| **Grove E3** | Pre-registered, unbuilt; its "behind Campaign E" sequencing tie is released now that Campaign E is closed as a scored campaign. |
| **Grove E4** | Pre-registered, unbuilt; still tied to the Stage 3c flip decision (same dependency as Plan C), NOT to Campaign E. |
| **`main`** | 209 commits behind `Beam-optimisation`, which now holds Campaign E, narrative facts, schema v19→v26 and this audit under a name that no longer describes it. |

---

## D3 closed + the `recites_gold` gate built (added 2026-08-26)

Both threads the 2026-08-25 audit left open as *cheap and well-evidenced*. Both
are LLM-free. Neither moves a canonical number, and one of them is asserted to
move none rather than claimed to.

**Machine note, load-bearing for what follows.** The dev machine is not the box.
`~/.hermes` does not exist there, and no `judge_audit.json` or scored LME run
file is present — those live on `Afrodite.MedSerPBAS`. So Part A's CODE is
built and tested; Part A's VERDICT is unrun and can only run where the banked
replies are. This is the same shape that blocks C4, and it is stated rather than
worked around.

### Part A — the `recites_gold` token rule (built, gated; RAN 2026-08-26 → **R2 FAIL**)

The pre-registration (`5884adf`) recorded that the C3 numerator's 5 false
negatives came from the instrument's token rule, not the judge, and said to gate
any loosening on its own. That is now built.

**v1 is frozen under its own name** with the same never-re-sync contract as
`shipping_verdict` — it produced C3 10/470, the 5-of-10 decomposition, and the
16/470 = 3.40% ceiling that licensed the 2026-08-25 spend. **v2** keeps the
verbatim fast path, strips the widening gloss from gold, keeps numerals as
content, and allows `RECITE_ALPHA_COVERAGE` slack on prose.
**`recites_gold` still aliases v1.** The alias IS the flip, and it never
happened: A5 ran on 2026-08-26 and failed R2. What follows is the state as
built and banked, *before* the verdict; the verdict is recorded below.

> **The safe direction reverses, which is the whole reason this is separate.**
> The banked note argues v1's under-count "can only wrongly REFUSE a spend,
> never wrongly license one". Loosening inverts that: `free_precheck`'s ceiling
> numerator is this function, so v2 can wrongly LICENSE one. v2 therefore holds
> numerals at ZERO tolerance and buys slack only on prose — and R2/R3, arms v1
> never needed, are the gate.

Bars, banked before v2 scored a row: **R1** recall on fixtures, **R2** precision
by hand-check of newly-flagged rows (FP ≤ 1), **R3** a free shuffled-gold
negative control (v2's mismatched-pair rate ≤ v1's + 2pp AND true-pair ≥ 3× its
own shuffled rate), **R4** the licence restated under both rules, reported not
barred. **R1 alone is explicitly not the gate** — it re-finds the rows the rule
was written from.

`--verify-recitation` is the free instrument, and it carries the vacuity split
for the same reason `--verify-parse` does: a row whose gold appears VERBATIM in
the answer is decided by the fast path BOTH rules share, so on such a corpus
"0 changes" cannot fail to be 0. `token_rule_consulted` is the denominator that
makes the number readable. Both directions of change are reported, because the
rules are **not nested** — v2 is looser on gloss and prose but stricter on
numerals, so it un-flags rows v1 flagged.

> **A fixture the tests rejected, worth keeping.** The first R1 arm included a
> "short numeral" row. It does not belong there: the banked note records two v1
> defects in one breath and they point OPPOSITE ways — discarding numerals makes
> v1 more PERMISSIVE, mandating the gloss makes it more RESTRICTIVE. Only the
> second can produce a false negative. Reading "two defects" as "two defects in
> the same direction" is the kind of error that survives a docstring and dies to
> a fixture that has to assert BOTH `v1 is False` and `v2 is True`.

**What A5 must do on the box:** run `--verify-recitation judge_audit.json`, hand
-check the `.recitation.json` sample it writes, and record R1–R4 with numbers.
PASS flips one line; FAIL keeps v1 and closes. Do not re-tune the rule to make
it pass.

> **DONE 2026-08-26.** It ran, and it FAILED R2 (2 FP against a bar of ≤1).
> The alias stayed at v1 and the rule was not re-tuned. Numbers in "A5 RAN"
> below.

### Part B — D3, the judge-outage channel (CLOSED)

`judge_answer` returned a bare bool, so a judge that never answered and a judge
that said "no" were the same value at the call site. An outage streak deflated
the arm it hit, silently, across all three canonical numbers — on a stack that
has already lived through one provider hard-deprecation and one outage streak.

**The channel is a sibling, not a modification.** `judge_answer_raw` returns
`(verdict, raw)`; `judge_answer` is now `judge_answer_raw(...)[0]`,
byte-identical. That matters: rule 3 of `parse_judge_verdict` is IDENTICAL to
`judge_audit.reference_verdict`, banked pre-run, and that identity is the entire
warrant for C1 = 0.00% certifying *this* function rather than something merely
like it. `judge_scored` holds the policy in one place so six call sites across
three adapters cannot each decide it differently.

**The denominators were the actual work** — this is what "not two lines" meant.
`sum(r["correct"] for r in rows) / len(rows)` appears at ~15 sites and TypeErrors
on the first unscored row. Unscored rows are dropped ONCE per reporting entry
point rather than guarded fifteen times.

> **Where an unscored row does the most damage, if it is coerced.** In the
> abstention arm it reads as "the reader answered a question it should have
> refused" — a hallucination finding manufactured by a timeout. In the recall
> diagnostics it lands in the miss decomposition and is attributed to whatever
> its `recall_ceiling` says — a retrieval finding, manufactured the same way.
> Both are the diagnostic-controls lesson: a broken device returns a confident
> constant.

**Scope extended, deliberately, to the run-file instruments.** `locomo_flip.py`
would have TypeError'd; it now drops an unscored row from BOTH arms or neither,
because dropping it from one leaves the comparison unpaired on exactly the rows
an outage touched — **the C4 arm-asymmetry void condition arriving through the
back door**. `locomo_audit.py` no longer hands one to the synthesis-bucket
hand-check as a reader failure. `facts_ab.py` had already reached the same rule
independently ("abstention/unjudged: no verdict to pair") and is now pinned so a
tidy-up cannot remove it on the grounds that `correct` is always a bool.

**Inertness is asserted, not assumed.** 0 judge sentinels over 500 replies means
the filter is the identity on a clean run, and a test says so. Nothing is
re-baselined.

**`judge_raw` is now persisted** on every verdict-recording row (~10 tokens).
This narrows `judge_audit.py`'s own reason to exist: `--run --spend` re-judges
because `raw` was discarded, and on any run made after today it is not. Future
audits of such runs should re-score STORED replies. C4 stays blocked on the old
pair; it is unblocked for the next one.

#### The negative-control probe lied, and that is the finding

All ten new D3 guards were reverted one at a time. Three read UNGUARDED. **Two
were real test gaps** (the abstention filter and LoCoMo's `judge_error` field —
the latter because a substring check on the file passed while one of two record
builders in the same module had lost it; replaced by an ast invariant that every
dict recording a verdict alongside its question carries the channel). **The
third was the probe itself**: stale `.pyc` bytecode meant pytest re-ran the
unbroken module. `PYTHONDONTWRITEBYTECODE=1` fixed it and the guard turned out
to be guarded all along.

> A negative-control device returning a confident wrong constant is precisely
> the failure mode negative controls exist to catch. Any future reversion probe
> in this repo runs with bytecode caching off, and reports which guards it broke
> rather than only which tests failed.

### Open after 2026-08-26

| Item | State |
|---|---|
| **D1 / C3** | Unchanged: WATCH-at-bar, band NOT resolved. |
| **`recites_gold` gate** | **CLOSED — R2 FAIL.** Ran free on Afrodite the same day; 2 FP against a bar of ≤1. Alias stays v1, rule not re-tuned. See "A5 RAN" below. |
| **C4 arm asymmetry** | Still blocked on the old pair; **unblocked for the next one** now that `judge_raw` is persisted. |
| **D3 visibility** | **CLOSED.** |
| **E6, Grove E3/E4, digest staleness** | **Grove E4 CLOSED 2026-09-01, FAIL-mechanism** — its specified null leaves the gated statistic invariant (one distinct value over 2000 permutations), and against a null that reaches it the box store sits below chance (9 vs 19.74). **Digest staleness (G-DS1) re-measured 2026-09-01: still not runnable**, 28 active profile rows vs cap 20 (22 → 23 → 28). Cross-source E6 is rejected/unbuilt (v46's same-unit replay is a narrower authority mechanism); Grove E3's measured population is ~2 rows on the box and 0 on conv-26, and nothing consumes `retraction_history` — the shape that closed Grove E2 FAIL-mechanism. |
| **`main`** | Now 214 commits behind `Beam-optimisation` and 4 ahead: a real merge, not a fast-forward. |

### A5 RAN — `recites_gold` v2 CLOSED, R2 FAIL (2026-08-26, Afrodite)

Gate ran free on `lme_audit_spend.json` (500 records, zero LLM calls).
**Verdict: R2 FAIL → alias stays `recites_gold_v1`. Rejected, not re-tuned.**

| # | bar | measured | |
|---|---|---|---|
| R1 | fixtures recover the FN mechanisms | 3 real rows LOST by v2 are all the numeral defect; v1 wrong on all 3 | confirmed |
| R2 | **FP ≤ 1 of newly-flagged** | **2 FP** (`f420262c`, `4dfccbf8`) | **FAIL** |
| R3 | shuffled ≤ v1+2.0pp AND true ≥ 3× shuffled | 4.02% vs 4.22%; 52.00% true = 13× | PASS |
| R4 | ceiling, reported | 3.40% → 4.04% nominal → **3.62% honest** | reported |

R2 fails on both readings of "sample": 2 of 12 newly-flagged, and 2 of the 3
refusal-arm rows. **No reading passes.**

**Two waivers argued and refused.** (a) "report not barred" is R4's clause,
verbatim and only R4's; R2's is "Bar: FP <= 1", and reading the waiver one row
up the table turns the gate into its own confirmation pass. (b) "the FPs only
inflate the upper bound" is the hazard R2 was written to catch — the banked
direction argument ("an under-count can only wrongly refuse a spend, never
wrongly license one") holds for the STRICT rule and **reverses** under
loosening, and the ceiling is what licenses the spend.

**The decomposition is worse than the bar, and is the keeper finding.** 8 of
v2's 12 new flags are non-refusal prose rows that do not move the ceiling.
On the refusal arm — the only population the licence reads — v2 is **1 TP / 3**.
Gains land where they do not count; errors concentrate where they do.

**The licence prediction is falsified independently.** `5884adf` pre-registered
the numerator at **≥21 (≥4.47%)** with the five FNs counted. v2 measured 19
(4.04%) nominal, 17 (3.62%) honest — below on both. Even waiving R2 entirely,
the flip never reaches the numerator that motivated the work: **the spend stays
unlicensed either way**, so the flip would cost the frozen baseline and buy
nothing. R3's 13× separation stands as a real result about v2's mechanism; it
is not a licence, because R3 was never the arm in question.

**Revival = a NEW pre-registration written before scoring**, not a re-tune.
The question A5 surfaced: does the refusal arm need a different rule from the
prose arm (≈10/12 precision overall vs 1/3 there)? Do not answer it by moving
`RECITE_ALPHA_COVERAGE` until that gate exists.

Also fixed: `judge_scored`'s docstring said five call sites (six), and
`judge_audit.py` said four `recites_gold` callers (three).

> **PRE-REGISTRATION 2026-09-02 — gates 4 and 5 run at 10:01Z, and this block
> is committed before they do.** The user authorized the spend for both. What
> follows is fixed before any number exists; the scorers are committed with it.
>
> **Gate 4 — the LME full guard, re-run so it can evidence itself.** Two arms,
> sequential, identical but for one flag:
>
> ```
> --scales S --sample 0 --seed 0 --workers 8 --top-k 15 --auto-ability
> --permissive-default --answer-model deepseek-v4-flash
> --answer-base-url https://api.deepseek.com
> --answer-extra-body '{"thinking":{"type":"disabled"}}'
> --judge-model deepseek-v4-flash
> --judge-extra-body '{"thinking":{"type":"disabled"}}'
> ```
>
> plus `--episode-granularity` on the ON arm and nothing else. That reproduces
> the 2026-08-30/31 pair's config exactly (`workers: 8`, `sample: 0`, full 500)
> so the new numbers are comparable to the old ones, and to the canonical.
>
> **The pre-flight was run offline and is the reason this spends at all.** The
> real parser was driven with both command lines and the namespaces diffed:
> they differ in exactly `episode_granularity` and nothing else, and
> `run_registry.arm_evidence()` on the config blocks those args produce returns
> **EVIDENCED, confounds []**. So the failure that voided the last pair is
> excluded *before* the API time, not diagnosed after it. Nothing about the
> scores was pre-flighted, and nothing could be.
>
> **The bar is unchanged and is NON-REGRESSION ONLY** — canonical 70.0 OVERALL,
> MS floor 51.9. Order of reading is fixed here: `arm_evidence` first, scores
> second. If the pair reads UNEVIDENCED or SAME_ARM the run is INCOMPLETE and
> the scores are not read at all, because a number that cannot say which arm
> produced it is the thing this re-run exists to stop. A PASS discharges gate 4
> and nothing else; it remains a bar a change with no effect also clears, which
> is the point gate 3's retirement made and this run does not alter.
>
> **Gate 5 — the dream cost watch, on a SNAPSHOT.** `benchmarks/
> dream_cost_watch.py`, four legs against one sqlite-backup copy of the
> production store: `settle` (OFF, bring to steady state), `before` (OFF, the
> baseline), `migrate` (ON, the one-time re-digest), `after` (ON, the steady
> state under test). The gated comparison is `after` vs `before`; `migrate` is
> reported, not gated, because a one-off is a price and not a regression.
>
> It runs on a copy because the production store dreams on a schedule (rows
> 1394-1397 all landed 2026-09-01). Flipping the lever on it would rewrite live
> episode rows to take a reading, and let the scheduled dreamer interleave with
> the measurement — mutating the user's memory AND getting a worse number for
> it.
>
> Criteria live in `dream_cost_watch.evaluate` and are committed with this
> block: the migrate leg must have sent the GRANULAR digest prompt and no blob
> one (attributed by prompt identity at the call, never by the `granularity`
> field the runner wrote — gate 4's lesson, applied before it could recur);
> new stamps/granular calls in [0.90, 1.0]; `after` digest calls <= `before`
> + 5; `after` wall clock within max(2x, +60s) of `before`; zero digest and
> fusion failures on both granular legs. 19 tests, 12 mutations checked.
>
> **Two things the pre-flight found that would have made the run lie.** A dress
> rehearsal with a stub LLM (no network, no spend) digested **40 of 110
> sessions**, not 110: `run_dreaming` mints no fallback chunk for a session
> with no user/assistant content and `continue`s before the digest, so 70 empty
> stubs can never carry a stamp however the lever is set. The first cut of the
> stamp criterion was keyed on the session count and would have read FAIL at
> 36% coverage — charging the flip for the store's own shape. The denominator
> is now granular calls SENT. Separately, the first cut of the two stamp
> criteria was one inequality written twice with different constants; no test
> could distinguish them and deleting either changed nothing. Mutation-checking
> found it, not review, and it is now a genuine two-sided bound (calls that
> land no stamp; stamps with no call behind them) with a test for each side.
>
> A third fact worth banking regardless of the gate: 61 sessions carry a
> `digested_prompt_version` but are no longer digestible. They were digested
> when they had content and no longer yield a chunk or a fallback. That is
> recorded in the census as `digested` and gated on nothing.
>
> **ADDENDUM, same day, still before the run — gate 4's scorer now exists and
> enforces the order.** `benchmarks/guard_score.py`. The block above fixed the
> reading order in prose ("`arm_evidence` first, scores second"); prose is what
> the last pair also had. `report()` now asks `arm_evidence()` and RETURNS
> before computing any accuracy when the pair is UNEVIDENCED or SAME_ARM, so
> the refusal is structural rather than a discipline. Run against the
> 2026-08-30/31 pair it prints INCOMPLETE and exit 2, and **71.0 does not
> appear in its output at all** — which is the property, since a number
> already seen cannot be un-seen, and one step for the numbers plus a separate
> step for provenance is exactly how that pair came to be quoted in three
> documents before anyone asked which arm it was.
>
> It also reports what the previous guard structurally could not: the FIRED
> subset, using the per-question `n_episodes` that `6543ee6` added. The E1
> lesson is the reason — all-800 net read NULL there while the subset where
> the tier reached the reader read -2.9pp (p=0.024), so an unconditional
> all-500 net is not evidence of no effect unless the tier can be shown to
> have reached the reader.
>
> **The subset's limit is stated in the instrument and printed on every run,
> because it is weaker than it looks.** `n_episodes` counts episodes handed to
> the reader, and this lever changes how episodes are CUT — so two arms can
> hand over the same NUMBER of different episodes. The subset is a LOWER BOUND
> on the questions the lever touched, not the set of them. It is reported and
> never gated, and the caveat prints loudest in the case that most invites
> over-reading: an EMPTY fired subset, which is equally consistent with "the
> lever changed nothing" and "the lever re-cut every episode without moving a
> single count". 16 tests, 9 mutations checked (score anyway after warning: 3
> fail; treat SAME_ARM as evidenced: 3; orient by argv order: 1; read the bar
> on the OFF arm: 4; drop either bar: 3 and 1; read a missing n_episodes as no
> effect: 6; define fired as counts that MATCH: 2; drop the caveat: 2).

> **STATUS 2026-09-02 — gate 5 RAN and PASSED, 10:01-10:13Z, all seven
> criteria.** Four legs on a sqlite-backup snapshot of the production store
> (110 sessions, 101 digested, 1218 episodes at snapshot time).
>
> | leg | granular | elapsed | calls | digest blob/gran | episodes |
> |---|---|---|---|---|---|
> | settle | False | 263.9s | 9 | 0/0 | 1218 |
> | before | False | 163.6s | 5 | 0/0 | 1218 |
> | migrate | True | 289.7s | 90 | **0/39** | 1239 |
> | after | True | **34.4s** | 1 | **0/0** | 1239 |
>
> **The architectural claim the gate exists to test holds exactly.** Flipping
> the lever re-digests each session once and then returns to zero tail calls:
> `after` made 0 digest calls against `before`'s 0, and ran in 34.4s against
> 163.6s. The steady state is not merely bounded, it is cheaper than the
> baseline cycle. One-time migration cost: **39 digest calls, 290s, 566,189
> prompt chars** — the whole price of the flip on this store.
>
> The migrate leg sent 39 granular digest prompts and **zero blob ones**, and
> 39 new stamps landed for 39 calls. Both stamp criteria are tight (100.0%,
> and no stamp without a call). Zero digest failures, zero fusion failures on
> both granular legs.
>
> **What the run does NOT establish, stated because the number invites the
> error.** `migrate` created 112 episodes across 39 sessions (2.87 per
> session) while the store's total moved only 1218 → 1239 (+21): supersession
> replaced roughly 91 rows. On those 39 sessions the count went 240 → 261.
> **240 is accumulated state, not one blob pass's output** — the store has
> been dreaming under the blob prompt for months — so no per-pass blob rate
> can be read off it, and the tempting "blob makes 6.15/session, granular
> makes 6.69" comparison is between two different kinds of number. The
> measurement that would settle it is a blob re-digest of the same 39 sessions
> on a fresh snapshot (~39 digest calls); it was not run.
>
> **What it does establish, and it bears on the flip decision.** Per-session
> episode counts on those 39 sessions moved median **3.0 → 5.0** (mean 6.15 →
> 6.69). Both medians sit inside G-EP1's criterion 2 band, "median episodes per
> substantive session in [3, 8] **on the target arm**". That criterion was
> pre-registered against the target arm alone with no baseline arm measured —
> so on this corpus it is satisfied by the store's blob-era shape as well, and
> its PASS is not by itself evidence that granularity changed the shape. Same
> defect class as gate 3 and as the 2026-08-30 guard pair: a bar that a
> no-effect change also clears. It does not retract G-EP1's PASS, which was
> measured elsewhere and included a faithfulness hand-score that this does not
> touch; it says criterion 2 specifically cannot carry the discrimination
> weight the flip argument has been putting on it.
>
> **Unrelated finding, banked separately because it is about the live store
> and not about Plan C.** The blob legs logged 11 `chunk_extraction.parse_failure`
> events and **2 chunks abandoned after 3 attempts with `content_lost=1`**
> (`chk_c7aaa821…`, `chk_9a7e13b7…`), on raw payloads of 24-28KB. Both
> occurred on `settle`/`before`, i.e. the OFF arm, so the granularity lever did
> not cause them; they are the production store's existing condition and would
> have gone on happening unobserved. `chunk_extraction_failures` is in-memory
> and NOT persisted to `dream_runs` by design, so nothing on the box records
> that this content was dropped. Worth its own item.

> **STATUS 2026-09-02 — gate 4 DID NOT RUN. Both arms were the OFF arm. The
> cost was 5.5h and ~2,000 reader calls, and the cause was a bug in the runner
> script, not in the adapter or the lever.**
>
> `guard2_rerun.sh` defined `run_arm () { label="$1"; shift; ... "$@"; }` and
> called `run_arm on`. `on` was consumed as the label, `shift` left `"$@"`
> empty, and `--episode-granularity` was never passed. Both artifacts record
> `episode_granularity_enabled=False`
> (`…20260902T125748Z…`, `…20260902T154415Z…`; 9,850s and 9,986s, 500 answer
> calls each).
>
> **The pre-flight did not catch it because the pre-flight tested the wrong
> thing.** It drove the real parser with both command lines and proved they
> differed in exactly `episode_granularity` — the command lines *intended*.
> Nothing checked the argv the *script* built. That gap is the whole defect:
> "the command I designed" and "the command the harness constructs" are
> different objects, and only the second one spends money.
>
> **`guard_score.py` did exactly what it was built for**, one day after being
> built for it: `[SAME_ARM] both arms recorded episode_granularity_enabled=False
> -- this pair is not an A/B on that lever, whatever it is named`, INCOMPLETE,
> exit 2, no accuracy computed. Without it this would have been read as a
> null on the lever and very likely banked as gate 4 PASS: two arms named
> off/on, 500 questions each, plausible scores, and a bar it clears.
>
> **What the spend did buy — the harness's test-retest floor at n=500, which
> the project has never measured.** Two runs of the IDENTICAL arm:
>
> | | KU | MS | SSA | SSP | SSU | TR | **OVERALL** |
> |---|---|---|---|---|---|---|---|
> | run 1 | 70.5 | 53.4 | 64.3 | 66.7 | 94.3 | 71.4 | **68.6** |
> | run 2 | 69.2 | 53.4 | 62.5 | 70.0 | 92.9 | 74.4 | **69.0** |
> | delta | −1.3 | 0.0 | −1.8 | **+3.3** | −1.4 | **+3.0** | +0.4 |
>
> **42 of 500 questions (8.4%) flip verdict between two runs of the same arm**
> (20 correct→wrong, 22 wrong→correct). Per-ability swings of ±3pp are pure
> noise on n=30-133 cells. Read every past per-ability A/B claim in this
> document against that: a ±3pp per-ability movement is not a finding, and
> several banked ones are that size. OVERALL is far steadier (+0.4).
>
> **This also retires the last excuse for the 2026-08-30 pair.** Its 71.0 vs
> 71.0 was tight enough to look like a clean null; two runs of one arm differ
> by 0.4 and flip 8.4% of questions, so exact agreement was luck, and the
> pair was uninformative for a second, independent reason.
>
> **AND IT RAISES A GATE-4 PROBLEM THAT MUST BE SETTLED BEFORE ANY RE-RUN.**
> Gate 4's bar is the canonical **70.0 OVERALL**. Today's OFF arm — the
> baseline, the same configuration — scores **68.6 and 69.0. The baseline no
> longer clears the bar the ON arm is required to clear.** MS is 53.4 in both
> and still clears its 51.9 floor, so the problem is OVERALL specifically.
> Either the canonical has drifted (model, endpoint, or dataset-side) or 70.0
> was always inside the noise band of a run that happened to land high. A
> re-run of gate 4 against an unreachable bar would fail for reasons that have
> nothing to do with episode granularity, and would cost another 5.5h to say
> so. **Gate 4 must be re-baselined before it is re-run.** The two OFF arms
> above are, unintentionally, most of the material for doing that offline.
>
> **Fixed: `benchmarks/assert_arm.py`** — parses the argv the runner is about
> to execute with the adapter's OWN parser and fails if the lever does not
> match the arm's label. It is `arm_evidence` moved from after the run to
> before it: one reads the config block a run wrote, the other the argv a run
> is about to use. `guard2_rerun.sh` now builds the argv ONCE into an array,
> asserts it, and executes that same array. Run against the argv that cost
> this run it exits 1. 20 tests, 5 mutations checked.
>
> One of those tests was itself vacuous and green: it captured ambient
> `sys.argv` as its "before", and an earlier test in the same file had already
> leaked, so the captured value WAS the leaked value. It passed against the
> mutation that deletes the restore. It uses a sentinel now. Third time this
> shape has been caught by mutation-checking rather than by reading, and the
> second time inside a module written to prevent it.

> **RE-BASELINING 2026-09-02 — gate 4's bar is a coin flip, and both of its
> constants come from a model that no longer exists. Offline, no spend.**
> `benchmarks/lme_noise_model.py`, 13 tests, 7 mutations checked.
>
> **Provenance first.** Gate 4 is pre-registered as `OVERALL >= 70.0` with an
> `MS floor 51.9`. Both numbers come from **one run** —
> `longmemeval-v2-hymem-20260610T094858Z-seed0.json`, 2026-06-10, `deepseek-chat`
> for BOTH answer and judge. `deepseek-chat` was **hard-deprecated 2026-07-24**
> (`hymem/bootstrap.py:23`). The canonical is not reproducible, and it is being
> used as a floor for runs on a different model.
>
> **Two independent noise estimates, which agree.** The failed run's accident
> supplied what the project never had: two runs of an IDENTICAL arm over the
> same 500 questions.
>
> | estimate | basis | SD of one run |
> |---|---|---|
> | PAIRED | 42/500 discordant questions, one same-arm pair | **0.92pp** |
> | ERA | 9 comparable full-500 v4-flash runs, 2026-07-27 → 09-02 | **1.17pp** |
>
> ERA is the larger, as it must be — it contains real drift as well as churn.
>
> **How the bars behave against a change that does nothing:**
>
> | bar | era mean (n=9) | position | P(inert arm FAILS) |
> |---|---|---|---|
> | OVERALL >= 70.0 | 70.27, SD 1.17 | **+0.23 SD — dead centre** | **41%** |
> | MS >= 51.9 | 54.30, SD 2.41 | −1.00 SD | 16% |
>
> Jointly, **an inert lever fails gate 4 about half the time.** A central
> estimate is being used as a floor. That is the same defect as gate 3 and the
> guard pair, in its third form: not a bar a no-effect change passes, but a bar
> a no-effect change fails at random — equally uninformative, and more
> expensive, because it invites a re-run.
>
> **The harness's resolution, which no re-baselining can fix.** McNemar on the
> paired null: 42 discordant, so |b−c| must exceed 1.96·√42 ≈ 12.7 questions to
> reject at α=.05 — **a minimum detectable effect of 2.54pp**. Gate 4 cannot
> distinguish a 2pp regression from nothing, however the bar is set, and this
> is set by CHURN, not sample size: LME-S has only 500 questions, so n cannot
> be raised. **The only lever on resolution is the 8.4% verdict churn itself**
> (greedy decoding, a deterministic judge, or judging twice and keeping
> agreement). That would buy more than any bar change.
>
> **Proposed replacement, to be pre-registered before any re-run.** Drop both
> absolute constants; both arms already run contemporaneously and share all
> 500 ids, so the comparison should be PAIRED:
>
> 1. Score by McNemar on paired per-question outcomes, ON vs the OFF arm of
>    the SAME session — never against a historical constant.
> 2. **REGRESSION** iff ON is worse and the test rejects at α=.05.
> 3. Otherwise **NO REGRESSION DETECTED, and the MDE is reported in the
>    verdict** — the claim is "no regression larger than 2.5pp", never "no
>    regression". A gate that cannot state its own resolution is how "71.0 vs
>    71.0" became evidence in the first place.
> 4. MS keeps a floor only as the same paired test on the MS subset; the
>    absolute 51.9 goes, having no reproducible basis.
>
> This does not make gate 4 cheaper — still two arms of 500 — and it does not
> make the flip motivated. It makes a gate 4 result mean something when it
> arrives, which the pre-registered version would not have.

> **CHURN DECOMPOSITION 2026-09-03 — there is no churn fix, and the one I
> recommended does not exist. Offline, no spend.**
> `benchmarks/churn_decompose.py`, 53 tests, 11 mutations checked.
>
> The re-baselining above closed with "the only lever on resolution is the
> 8.4% verdict churn itself (greedy decoding, a deterministic judge, or
> judging twice and keeping agreement)". Two of those three were already in
> place and the third is worthless. **That recommendation is retracted, on a
> measurement rather than an argument.**
>
> The two same-arm artifacts record the answer text and the raw judge reply
> per question, so the churn splits offline and free.
>
> **Stage 1 — the judge is not the problem.** On **181** of 500 questions the
> two runs produced byte-identical answer text. The judge changed its mind on
> **none** of them. All 42 flips had different answer text.
>
> | | same answer | different answer |
> |---|---|---|
> | verdict held (458) | 181 | 277 |
> | verdict flipped (42) | **0** | 42 |
>
> The power check is what makes that readable: answer identity runs at 40%
> among concordant questions and 0% among discordant, so it is strongly
> associated with the flip rather than uninformative. Zero out of 181 is
> reported as an interval, not a rate — Clopper-Pearson one-sided 95% puts
> the judge's flip rate at **≤ 1.6%**, which is the honest claim. Judging
> twice and keeping agreement would buy nothing measurable, and the reader
> and judge already run at `temperature=0.0`.
>
> **Stage 2 — nor is it our retrieval.** Splitting the 42 by whether the
> reader's recorded context also moved: **11 moved, 31 did not.** The raw
> count invites "retrieval churn is ours, go fix it". The matched control
> refuses it. Against **non-flips whose answer text also moved** — where the
> reader's output moved and the verdict held anyway — a moved fingerprint
> runs at **35%**, against **26%** among the flips. Retrieval movement is if
> anything *under*-represented among flips. Retrieval churn is real and is
> not what flips verdicts.
>
> Using *all* concordant questions as the control would have read 65%
> identical and made retrieval look guilty; it is the wrong denominator,
> because it includes the stable questions where retrieval is stable for the
> same reason the answer is.
>
> **So the residue is provider-side non-determinism at temperature=0.0,
> which no flag of ours removes.** MDE **2.54pp** is a property of this
> harness, not a tuning parameter. LME-S caps n at 500, so it cannot be
> bought down with more questions either. Any plan that needs to resolve a
> sub-2.5pp effect on LME-S needs a different instrument, not a better run.
>
> **One gap closed for next time.** Stage 2's split is a pair of bounds, not
> a partition, because the artifact records COUNTS (`n_episodes`, `n_facts`,
> `ability_used`) and two runs can hand the reader the same NUMBER of
> different episodes — the same lower-bound caveat `guard_score.fired_subset`
> carries. `context_sha` (commit `5acedf7`) hashes the rendered reader prompt
> onto every row at no extra call, so a future paired run answers this
> exactly. `churn_decompose` reads it when both runs carry it and **refuses**
> a pair where only one does: every fingerprint would differ and the split
> would describe the two schemas rather than the two runs.

> **STATUS 2026-09-03 — gate 4's scorer is now the paired one. Offline, no
> spend.** `benchmarks/guard_score.py`, 48 tests, 16 mutations checked.
>
> The replacement pre-registered above is built. `CANONICAL_OVERALL` and
> `MS_FLOOR` are deleted; the contrast is exact McNemar against the OFF arm
> of the same session; REGRESSION requires ON to be worse **and** the test to
> reject; and the negative verdict reports **"no regression larger than X
> pp"** with X computed from the run in hand.
>
> Two things the absolute bars were hiding:
>
> 1. A moved `answer_model`, `judge_model`, `scale`, `sample` or `seed` is
>    now **INCOMPLETE**, not a noted confound. Paired scoring assumes the
>    arms differ in the lever alone; two answer models is a different
>    experiment, and no verdict is computed.
> 2. **The multi-session subset resolves only 6.3pp** (n=121, 15 discordant)
>    against OVERALL's 2.5pp. The retired 51.9 floor was being read on a
>    subset that cannot see a 5pp regression, and said nothing about it.
>
> Verified end to end: the real same-arm pair still reads INCOMPLETE/exit 2,
> and under an injected lever label that known-null pair reads NO REGRESSION
> DETECTED at exactly the 2.54pp `lme_noise_model` derives independently.
>
> **Gate 4 remains OPEN and un-run.** What changed is that a result would now
> mean something; the spend is still two arms of 500 and still unauthorised.

> **CONCENTRATION WORK-UP 2026-09-03 — the fired-indicator gate 4 already uses
> is blind on 84% of the run. Offline, no spend.**
> `benchmarks/concentration_model.py`, 27 tests, 20 mutations checked.
>
> Churn cannot be reduced and LME-S caps n at 500, so the last lever on gate
> 4's resolution is spending the 500 questions better. McNemar rejects when
> the net questions moved exceeds `Z*sqrt(D)`, and D counts only DISCORDANT
> questions — so if the lever's effect sits inside a subset S, scoring S
> keeps the whole numerator and discards the churn outside it:
>
>     gain = sqrt(D_all / D_S) ≈ 1/sqrt(f),   break-even leakage = 1 - sqrt(f)
>
> where **f is the fraction of the run the lever actually moves**. Churn is
> near-uniform on the calibration pair (7.8% inside the fired subset vs 8.5%
> outside), which is the warrant for the proportionality.
>
> | f | MDE | gain | break-even leakage |
> |---|---|---|---|
> | 5% | 0.57pp | 4.47x | 78% |
> | 25% | 1.27pp | 2.00x | 50% |
> | 50% | 1.80pp | 1.41x | 29% |
> | 100% | 2.54pp | 1.00x | 0% |
>
> **`f` is a property of the LEVER and no same-arm pair can measure it.** The
> null pair's firing rate is what the indicator does with NO lever set — the
> contamination floor, not f. An earlier draft of this module quoted the gain
> off that rate, which reads the denominator from the wrong population; the
> output is a curve for that reason.
>
> **THE FINDING. `n_episodes` saturates.** 421 of 500 questions sit at exactly
> **10** episodes — the retrieval cap. On **84%** of the run the indicator
> `guard_score.fired_subset` uses is a CONSTANT and can never fire, however
> the lever cuts episodes. Its subset is not "the questions the lever touched"
> but "the questions that fell below the retrieval cap", which is a fact about
> retrieval depth. The existing LOWER BOUND caveat badly understated this and
> has been rewritten in `guard_score.py`.
>
> Corroborating, weakly: the one granularity=True artifact
> (`20260831T101051Z`, 10 questions) shares all 10 ids with the OFF arm and
> `n_episodes` differs on **0 of 10** — eight of them pinned at 10. It is a
> 10-question probe with `sample` and `workers` also moved, so it evidences
> nothing about the lever; it is quoted only as a second sighting of the cap.
>
> **What this means for gate 4.** Concentration is real and worth up to 4x,
> but only for a NARROW lever (f <= 25%), and it cannot be applied to episode
> granularity today because there is no usable fired-indicator for it: the
> count-based one is blind on five-sixths of the run and `context_sha` did not
> exist when these runs were taken. **Do not buy a subset-scored gate-4 re-run
> expecting better resolution.** The first run carrying `context_sha` (5acedf7)
> gets a true fired set for free, and this module then measures the gain
> instead of projecting it.

> **RETRIEVAL-ONLY MODE 2026-09-03 — measure `f` without the reader, and a
> correction to what that saves. Offline build, no spend.**
> `--retrieval-only`, `benchmarks/fired_fraction.py`, 35 new tests, 20
> mutations checked.
>
> `concentration_model` left one number undetermined and decisive: **f**, the
> fraction of questions the lever moves. At f <= 25% a subset-scored gate 4
> resolves twice as finely and is worth buying; at f >= 75% it resolves no
> better and gate 4 should be retired rather than re-run. f is a property of
> RETRIEVAL — it does not depend on what the reader says or how the judge
> scores it — so it can be measured without either.
>
> **The mode.** `--retrieval-only` retrieves, builds the reader's prompt,
> hashes it, and sends nothing. Two guarantees are structural rather than
> branch-deep: the reader and judge become `PoisonLLM` objects that RAISE if
> reached, and distillation (which is part of retrieval and genuinely fires)
> gets its own counted client so the reader's can be poisoned. The prompt is
> rendered in exactly one place, so the cheap run fingerprints byte-identically
> what the expensive one would have sent — tested across every branch that
> swaps the system prompt or the memory list.
>
> The artifact carries `retrieval_only` and a measured `retrieval_cost` block,
> and **every scorer refuses it**: `guard_score`, `concentration_model` and
> `churn_decompose` all reject a verdict-free artifact rather than reading
> `correct: None` as a miss. A 0% arm beside a 69% arm looks like a
> catastrophic regression rather than a category error, and the refusal is
> what stops that. (Three refusals were written with no test; the mutation
> sweep found all three.)
>
> **THE CORRECTION. It is not nearly free, and I said it would be.** The mode
> skips the reader and the judge — but not the DREAM, and episode granularity
> is a **dream-time** lever, so the dream is the one thing that cannot be
> skipped when measuring its f. From the archive: no-dream 500-question runs
> take **~0.2h**, dreamed ones **2.2–2.8h**, and the no-dream runs already
> include all 500 answer and 500 judge calls. **Dreaming is ~93% of the wall
> clock.** Dropping the reader and judge saves ~7% of a run, not 90%.
> (`distill` is False in the gate-4 config, so that caveat at least does not
> apply: the mode makes literally zero API calls of its own.)
>
> **The saving is in `n`, not in the mode.** f is a proportion and the
> decision it feeds is coarse, so `fired_fraction` reports a **Wilson
> interval** rather than a point: run the smallest n whose interval clears the
> threshold and escalate only if it straddles. **50 questions per arm is ~4%
> of a full gate-4 pair** and separates f=10% from f=80% decisively.
>
> **Recommended next step, when spend is authorised:** two 50-question
> `--retrieval-only` arms (~0.5h total). If f is broad, gate 4 cannot resolve
> this lever at any price and should be retired; if narrow, a subset-scored
> gate 4 is worth the 5.5h. Either way the 5.5h is only spent once the cheap
> measurement says it would mean something.

> **PRE-REGISTRATION 2026-09-03 — the f probe. Authorised spend; written and
> committed BEFORE the run.**
>
> **Question.** What fraction of questions does `--episode-granularity`
> actually move? Everything downstream turns on it: at f <= 25% a
> subset-scored gate 4 resolves ~2x more finely and is worth its 5.5h; at
> f >= 75% it resolves no better and gate 4 should be RETIRED for this lever
> rather than re-run.
>
> **Design.** Two arms, `--retrieval-only`, 50 questions, seed 0. Verified
> offline before scheduling: the loader draws the same 50 ids on both arms
> (identical across two loads, 50 unique, stratified 5 each across 10 types).
> Both arms dream — `--no-dream` is NOT set, and must not be: episode
> granularity is a dream-time lever, so skipping the dream would measure
> nothing. `distill` is False, so the arms make **zero API calls of their own**.
>
> Argv, identical but for the last flag, built once into an array and asserted
> against its own label before anything spends:
>
>     --scales S --sample 50 --seed 0 --workers 8 --top-k 15
>     --auto-ability --permissive-default --retrieval-only
>     --answer-model deepseek-v4-flash
>     --answer-base-url https://api.deepseek.com
>     --answer-extra-body '{"thinking":{"type":"disabled"}}'
>     --judge-model deepseek-v4-flash
>     --judge-extra-body '{"thinking":{"type":"disabled"}}'
>     --data-dir /home/node/.hermes/benchmarks
>     [ARM ON ONLY: --episode-granularity]
>
> Asserted through the adapter's OWN parser, on the array the runner executes:
> `retrieval_only=True` on both, `episode_granularity` False/True, and
> `sample=50 seed=0 workers=8 top_k=15 no_dream=False distill=False`. Both
> passed at 06:58Z, ~3h before the run.
>
> **Scorer.** `benchmarks/fired_fraction.py`, committed at `88c32fc` before
> any of this was scheduled. It refuses a non-retrieval-only artifact, refuses
> a pair that cannot evidence its arms, and refuses to fall back to the count
> fields when `context_sha` is absent.
>
> **DECISION RULE, fixed in advance.** Read the Wilson 95% interval, not the
> point:
>
> 1. interval entirely **>= 0.75** -> **BROAD**. Concentration buys nothing;
>    recommend retiring gate 4 for this lever rather than re-running it.
> 2. interval entirely **<= 0.25** -> **NARROW**. A subset-scored gate 4 is
>    worth the 5.5h, and `concentration_model` then measures the gain instead
>    of projecting it.
> 3. otherwise -> **INCONCLUSIVE AT n=50**. Escalate n or accept a middling
>    gain; do NOT read the point estimate as if it had settled it.
>
> **What this cannot establish.** That the lever moved the reader's input on a
> question is not evidence it moved the ANSWER, still less the verdict. f
> bounds what a subset-scored gate could SEE. A result here is never a result
> about whether episode granularity helps.
>
> **Expected cost.** 0 answer calls, 0 judge calls, 0 distill calls. Wall clock
> is dream-dominated: ~2.74h per 500 dreamed questions scales to **~0.28h per
> arm, ~0.55h total**. The measured `retrieval_cost` block and elapsed_s will
> be checked against this, and a large miss is itself a finding.

> **INCIDENT 2026-09-03 — the f probe ran, completed its retrieval, and was
> destroyed by a print statement. My defect; no result.**
>
> Arm OFF dreamed and fingerprinted all 50 questions in **876s** (against a
> ~0.28h estimate, so the cost model was right), then died in the run summary:
>
>     AttributeError: 'PoisonLLM' object has no attribute 'call_count'
>
> `main()` reads `answer_llm.call_count` in two places. I guarded the artifact
> block with `getattr` and missed the summary print. `set -e` then aborted
> before arm ON started. **No artifact was written and the run produced
> nothing.** The dream tokens are spent and unrecoverable.
>
> **Four fixes, all of which the incident earned:**
>
> 1. `PoisonLLM` now carries `call_count = 0` and `total_tokens = 0`. A
>    stand-in that is not a drop-in only relocates the failure; guarding one
>    reader and missing another is precisely what happened. A test now
>    extracts every `answer_llm.<attr>` / `judge_llm.<attr>` read in the module
>    and asserts the stand-in has each, so the class is pinned, not the
>    instance.
> 2. **A SAFETY DUMP before any reporting.** Everything expensive happens in
>    the loop; everything after it is presentation, and five diagnostic passes
>    sat between the loop and the artifact write. The rows are now dumped to
>    `partial-<stamp>-seed<n>.json` immediately, carrying the config keys a
>    scorer needs to pair the arms. 876s of dreaming should not be lost to a
>    format string.
> 3. `judge_error` is False under `--retrieval-only`. The log said
>    "⚠ UNSCORED (judge error): 50" for a run that made **zero judge calls** —
>    a judge never called did not error, and that is the same vacuity as
>    "0 judge errors" over a run that made none.
> 4. The progress line reports no accuracy under the flag. "Acc: 0.0%" there
>    is not a low score; it is a measurement never taken.
>
> **A SEPARATE AND WORSE FINDING — my mutation harness was poisoning the
> bytecode cache.** Chasing an impossible test failure (`all()` evaluating
> False while the function returned True) showed the container executing a
> `.pyc` whose `is_retrieval_only` called `any`, not `all`. The sweep writes a
> mutant, imports it, then restores the original; `any(` and `all(` are the
> **same length**, so when the restore lands in the same second Python's
> mtime+size check passes and the stale mutant bytecode survives the sweep.
>
> Every result run after such a mutation was therefore suspect, including a
> "1891 passed" suite and one sweep's clean bill. All caches were purged, every
> harness now runs under `PYTHONDONTWRITEBYTECODE=1`, and **all seven sweeps
> and the full suite were re-run clean**: 1891 passed, 1 skipped, no survivors.
> Three sweeps had also gone stale against later refactors and were
> re-anchored rather than left claiming coverage they no longer checked.
>
> This is the session's defect class turned on my own tooling: a verification
> step that silently stopped verifying.
>
> **Verified without spending.** A one-question `--no-dream --retrieval-only`
> run exercises all of `main()` and makes **zero API calls**. It now completes:
> artifact written, `retrieval_cost` all zeros, 0 judge-error flags (was 50),
> every row carrying a `context_sha`, and the safety dump present. That check
> is free and should have been run before 10:01Z.
>
> **Gate: a re-run is new spend and is NOT authorised by the original
> approval.** The probe must be re-run from scratch — both arms — to produce
> the f the pre-registration asks for.

> **RESULT 2026-09-03 — f = 94%. Gate 4 cannot be rescued for this lever.**
> Two 50-question `--retrieval-only` arms, 12:38–12:56Z. Read against the
> decision rule pre-registered at `aeb0b24`, unmodified.
>
> **f = 47/50 = 94% (Wilson 95% CI 84%–98%).** The interval lies entirely
> above 0.75, so by rule 1: **BROAD**.
>
> | | |
> |---|---|
> | concentration gain | **1.03x** |
> | MDE 2.54pp would become | **2.46pp** |
> | break-even leakage | 3% |
>
> Subset-scoring gate 4 buys **8 hundredths of a percentage point**. The last
> lever on this gate's resolution is spent.
>
> **Run integrity.** `assert_arm` passed both dests on both arms before either
> spent; each artifact was verified against its arm label rather than picked by
> `ls -1t`; safety dumps written on both; arm evidence EVIDENCED (A=False,
> B=True); 50 rows and 50 `context_sha` on each side; **0 answer calls, 0 judge
> calls, 0 distill calls, 0 tokens**. Elapsed 0.28h and 0.29h against a 0.28h
> estimate — the cost model was right.
>
> **The saturation finding, confirmed on a real A/B.** On this pair:
>
> | indicator | fires on |
> |---|---|
> | `context_sha` (the rendered prompt, hashed) | **47/50 = 94%** |
> | `n_episodes` differs (what `guard_score.fired_subset` uses) | 9/50 = 18% |
> | both arms pinned at the cap of 10 episodes | 40/50 = 80% |
>
> The count indicator misses **83% of the fired set**. The lever changes what
> the episodes CONTAIN, not how many are handed over, and the count saturates
> at the retrieval cap on 80% of questions here (84% predicted from the
> 2026-09-02 runs). Had f been measured with the count indicator it would have
> read 18% — squarely in the "NARROW, concentrate, buy the 5.5h run" band, and
> wrong. `context_sha` is what makes the difference between the two.
>
> **WHERE THIS LEAVES GATE 4.** Every lever on its resolution is now measured
> and exhausted: the bars were miscalibrated (an inert arm failed them ~50% of
> the time); MDE 2.54pp is set by churn, not n; the churn is not the judge
> (<=1.6%) and not our retrieval (under-represented among flips), leaving
> provider-side non-determinism at temperature 0; LME-S caps n at 500; and
> concentration buys 1.03x. **LME-S cannot resolve an episode-granularity
> effect smaller than ~2.5pp, at any price.**
>
> That is not the same as "gate 4 is worthless". As a NON-REGRESSION gate it
> can still answer one question honestly: *is there a regression larger than
> 2.5pp?* If that is the question, one paired run answers it and the paired
> scorer now reports the resolution alongside the verdict. What it can never be
> is a tuning signal, or a warrant for "granularity is harmless".
>
> **RECOMMENDATION: retire gate 4 as a decision gate for episode granularity.**
> Run it once if a >2.5pp blow-up is the worry; do not run it to decide whether
> the feature helps, because it cannot. Deciding that needs a different
> instrument, and f=94% says the lever is a broad intervention, so an
> instrument that can see broad effects is the thing to look for.
>
> **What f does NOT establish**, restated because it is the easiest thing to
> over-read: 94% of prompts moved. Not one answer, and not one verdict.
