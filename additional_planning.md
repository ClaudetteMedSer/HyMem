# Additional Planning

Two ideas borrowed from [BrainDB](https://github.com/dimknaf/braindb), adapted to
HyMem's embedded, edge-typed architecture, plus the episode-granularity plan
(added 2026-07-02, see [Plan C](#plan-c--episode-granularity-in-dreaming)),
plus the narrative-facts campaign (added 2026-07-30, see
[Campaign E](#campaign-e--the-narrative-facts-roadmap-added-2026-07-30)):

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
> across a deliberate prompt revision.
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
> generative remedy. Both remaining sequencing constraints are unchanged: this
> plan touches episode membership, so it stays gated on the RAPTOR flip-watch.

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

---

## Campaign E — the narrative-facts roadmap (added 2026-07-30)

*Origin: the 2026-07-30 competitive review (HyMem vs Hindsight/Honcho), integrated
with its own critique the same day. Supersedes nothing above; Plan C is subsumed
by Step 4 (E1 build) as noted there. Same standing contract as every plan in this
file: front-run gate before any build, additive-only, mechanism > score, nothing
reads oracle labels, per-category LME deltas under ~±5pp are noise.*

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
2. **Facts are per-range IMMUTABLE, append-only, version-tagged.** A closed
   range's facts never change → no re-fusion, no resample, no poison-cascade:
   the entire aggregation reuse bug class (the third flip-watch failed on the
   deepseek outage streak; watch still red) is sidestepped by construction.
   The ONE mutable field is `invalid_at` (closing it is itself append-only).
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

> **BUILT + RUN 2026-07-30 → G-F1 FAILED, E1 BANKED DEAD** (verdict at the end of
> this block; the probe itself is retained as the repo's faithfulness instrument).
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

> **BUILT 2026-07-30, UNRUN** (M1 needs an API key for the LLM arm; both
> measurements need `sentence-transformers` + the two CE models, unavailable on
> this box). `benchmarks/rerank_ab.py` + `tests/test_rerank_ab.py` (17 tests);
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

---

### Step 4 — E1 build: the narrative-facts artifact ~~(gated on G-F1)~~ CANCELLED

> **CANCELLED 2026-07-30 — G-F1 FAILED TWICE. Do not build this.** Faithfulness
> 0.55–0.76 vs a required 0.90, with the same inventions reproducing across the
> one allowed prompt revision. Full verdict and its reasoning: Step 1 above.
> The spec below is kept verbatim as the record of what was gated and why the
> gate was worth running — **it is not a backlog item.** Reviving any part of it
> requires a NEW faithfulness result, not a re-reading of this one. E6
> (supersession over facts) is cancelled with it; E2/E4/E7 must be re-specified
> over an artifact that exists.

**Idea.** Dream-time extraction of self-contained narrative facts, stored
immutably (append-only, version-tagged), served as an additive retrieval tier
and as the lead evidence block in `ask()`. Subsumes Plan C's granularity goal
as a NEW artifact class — episode membership is untouched, so the Plan C
sequencing constraint (RAPTOR flip-watch) does not apply. Unblocks E6.

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
  `cfg.dream_max_facts_per_session` (default 8, truncate).
- Persist **append-only**: `INSERT OR IGNORE` keyed on the UNIQUE constraint;
  no UPDATE path for `text`/`entities`; `invalid_at` starts NULL. Watermark
  advances only on successful parse; parse failure → `DreamReport.fact_failures
  += 1`, watermark held, retried next dream (the v25 `digest_failures`/
  `parse_failed` contract).
- **Prompt bumps extract FORWARD ONLY**: new ranges are tagged with the new
  `FACTS_PROMPT_VERSION`; covered ranges are never re-extracted. (Review
  constraint 2 — this is what makes E1 safe while the flip-watch is red.)
- Idempotency: re-dream of a quiescent store = 0 extraction calls (watermark at
  tail) + 0 new rows (UNIQUE key).
- Embeddings: batch-embed fact texts OUTSIDE the write lock (the phase1
  lock-free pattern); content-addressed `embedding_cache` reuse is free.
- `DreamReport.facts_extracted` / `fact_failures` persisted to `dream_runs`.

**Retrieval — additive tier in `query/augment.py`:**
- `FactHit` dataclass `{fact_id, text, fact_date, entities, session_id, score,
  why_retrieved}`; `ctx.facts: list[FactHit]` with the standard additive-tier
  docstring (own SELECT/FTS, never consumes another tier's budget, degrades to
  [] on a pre-v26 store).
- `_fact_search()`: FTS5 over `narrative_facts_fts` (same `_FTS_SAFE` +
  `_fold_diacritics` query path as the other six FTS sites) + optional vec KNN
  + RRF; `WHERE invalid_at IS NULL` (superseded facts leave the tier but stay
  in the DB for audit); cap `cfg.facts_top_k` (default 8, 0 disables).
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

**Risks:** extraction quality on deepseek (G-F1 faithfulness gate is the
mitigation); growth on long-lived sessions (watermark + caps); version mixing
at retrieval (harmless — additive evidence, duplicates impossible by
append-only + UNIQUE).

**Tests (`tests/test_facts.py`, StubLLMClient + stub embeddings, mirroring
`test_digest.py`):**
1. extraction→persist round-trip (rows, canonical entities, watermark advanced
   to covered `end_message_id`);
2. append-only: new messages arrive, re-dream → old rows byte-identical, new
   rows only for the new range;
3. idempotent re-dream: 0 new rows, 0 extraction calls (stub call count),
   watermark stable;
4. forward-only versioning: bump `FACTS_PROMPT_VERSION` → new range under new
   tag, covered ranges untouched;
5. parse failure → `fact_failures` +1, watermark held, retry succeeds;
6. UNIQUE dedup on re-submitted range;
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

### Step 5 — scored confirmation (box; protocol, not a build)

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

---

### Step 6 — E3 adoption (one deliberate rebaseline; after Step 5)

If M1+M2 passed (Step 3): flip `rerank_model` default `"llm"`→`"cross-encoder"`
and (if M2 passed) `rerank_cross_encoder_model` → `bge-reranker-v2-m3` in ONE
commit — the single deliberate rebaseline the review priced. LME/LoCoMo
non-regression rides Step 5's runs where the config diff is orthogonal, else
one shared confirmation run. Update the frozen-baseline table
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

### Step 8 — E4, E6, E7 (production track, independent)

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
> **Status: (a) `hymem/query/reldates.py` NOT written; (b) augment wiring NOT
> written.** One cheap confirmation is outstanding before E4 is formally banked:
> `python reldate_probe.py --dataset <lme_s>.json` on the box (free, no LLM).
> LME questions carry an explicit `question_date`, so it is the fairer test of
> criterion 1 — but criterion 2's axis mismatch is architectural and will
> reproduce.

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

**~~E6 supersession over facts~~ — CANCELLED 2026-07-30 with E1** (its target
artifact does not exist). Worth keeping the reason visible: supersession is the
mechanism that would have *defended* a fabricated fact — closing the older,
correct row in favour of a newer invented one — so at 0.55–0.76 faithfulness E6
was not merely unbuildable, it was the amplifier. Spec retained below only as
the record of the intended design.

**E6 supersession over facts** (after Step 4). Extend the
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
| 1 | E1 probe (`fact_probe.py`) — **RUN, VERDICT IN** | — | ~1,000 calls (v1 + v2) | **G-F1 FAILED** — faithfulness 0.55–0.76 vs 0.90, twice ✗ |
| 2 | E5 anaphora — **BUILT, GATE PASSED, SHIPPED** | — (parallel day one) | none (heuristic) | 31-item eval **100%**, no-harm **0/12** ✓ |
| 3 | E3 measurements M1+M2 — **BUILT, UNRUN** | — (parallel, offline) | M1 LLM arm only | M1/M2 pre-registered parity |
| ~~4~~ | ~~E1 build~~ — **CANCELLED** (G-F1) | — | — | — |
| ~~5~~ | ~~Scored confirmation~~ — **CANCELLED** (nothing to confirm) | — | — | — |
| 6 | E3 adoption (one rebaseline) | Step 3 only (Step 5 gone) | ≤1 shared run | offline parity held; baselines re-frozen |
| 7 | E2 observations — **needs re-spec** (was over facts) | flip-watch green **+ a new faithfulness result** | capped per dream | must clear `fact_probe.py`'s bar first |
| 8 | E4 — **G-E4a FAILED 2/3, not built** (E7 open; **E6 cancelled with E1**) | E7: none | none spent (probe is LLM-free) | E4 fire rate 1.2% vs 5%, precision 20.8% vs 90% ✗ |

**Post-G-F1 campaign state.** Campaign E's generative half is closed; its
retrieval half is what remains. Live work: **E3** (M1 needs an API key, M2
blocked on the torch OOM) is now the only unblocked scored item, and it is
independent of everything E1 touched — it reorders a pool retrieval already
built and writes nothing. ~~**E4** (temporal-range boost) was specified over facts' `fact_date` but is
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
signal (boost ≠ filter); additive tiers never touch another tier's budget; any
material prompt change bumps its version constant AND extracts forward-only
(facts) or re-gates (profile precedent); judge posture frozen; LME A/Bs are
non-regression confirmations, never tuning signals.
