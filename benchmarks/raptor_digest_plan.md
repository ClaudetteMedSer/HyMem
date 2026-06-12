# RAPTOR / Digest carry-over plan

*Written 2026-06-11. Companion to `longmemeval_roadmap.md` (which stays the LME record);
this doc tracks the RAPTOR→digest product thread after RAPTOR was closed as an LME lever.
Same contract as the roadmap: front-run gate before any build, additive-only, mechanism >
score, nothing reads oracle labels.*

---

## 0. Where we are (state as of 2026-06-11, branch `Beam-optimisation`)

**Landed and verified:**
- Schema v17: `aggregation_nodes.level`/`is_root`; levels ≥1 NEVER enter query-time
  retrieval (level-0 filter in `_aggregation_search`) — the G4 crowding mechanism is
  structurally impossible.
- `HyMem.digest()` → exported `Digest` dataclass (`title`, `summary`, `n_sessions`,
  `n_sessions_total`, `generated_at`). Root built by recursive rollup (consecutive-chunk
  fallback guarantees convergence; even leaf sampling across the backlog, cap 256).
- Fusion reuse cache keyed by member-set hash + prompt-version salt
  (`cluster.v2` / `rollup.v2` / `root.v4` in `hymem/dreaming/aggregate.py`).
- Store-grounded anchor: `_anchor_facts()` injects top active non-derived KG edges as a
  VERIFIED FACTS block into the root fusion (`aggregation_digest_anchor_facts`, default
  20); block hash joins the root cache id so graph changes regenerate the digest.
- Query tier: ability-gated (`aggregation_inject_abilities`, default `("TR",)`),
  adapter flags `--aggregation-nodes` / `--aggregation-broad`.
- Prod findings: breadth fix confirmed (7+ threads vs single vignette); coverage 65/91
  pinned to the episode-extraction gap; "Acme Corp" hallucination traced to a poisoned
  cached rollup → salts bumped, anchor built, **v4 not yet verified on the box**.

**Standing invariants (hard-won, do not re-litigate):**
1. No more LME A/Bs on aggregation variants — the benchmark is structurally unable to
   reward them (recall already closed by message-FTS; summaries can only reshuffle).
2. Additive-only consumption: nodes/digest must never compete with raw turns for slots.
3. **Salt-bump rule:** any material fusion-prompt change MUST bump that level's salt, or
   cached artifacts outlive the fix (the Acme lesson).
4. Per-category LME deltas under ~±5pp are noise; only trust a mechanism.

---

## Stage 0 — Verify root.v4 on the box (run FIRST tomorrow, ~15 min)

The salt bumps mean the next dream regenerates every fusion (~13 calls on the 91-session
store — trivial). Checklist:

- [ ] Re-dream with the layer enabled; confirm `aggregate.built nodes=N reused=0` in the
      log (full regeneration — proves the poisoned rollup is evicted).
- [ ] `hy.digest()`: "Acme Corp" / invented identity GONE; breadth retained (7+ threads).
- [ ] Inspect what the anchor actually injected:
      `SELECT subject_canonical, predicate, object_canonical FROM knowledge_graph
       WHERE status='active' AND derived=0 AND invalid_at IS NULL
       ORDER BY pos_evidence - neg_evidence DESC LIMIT 20;`
      Expectation: tech edges present, personal identity (bedrijfsarts, name) likely
      ABSENT — that absence is the Stage 1 motivation, record it.
- [ ] Coverage attribution: `SELECT COUNT(DISTINCT session_id) FROM episodes;` vs total
      sessions. If ≈65, the digest is faithful and the 26-session gap is Stage 2's.
- [ ] Second dream, no changes → `reused=N` (all fusions), digest byte-identical.

**Gate:** if v4 still shows invented identity after cache eviction + anchor, do NOT
prompt-tune further — that's evidence the model invents at episode level (upstream of
the tree); jump to Stage 1/2 diagnostics instead.

---

## Stage 1 — P4: typed user-profile tier (feeds the anchor true identity)

**Problem:** the anchor can only inject what the graph knows, and the 18-predicate
vocabulary is tech-domain. "User is a bedrijfsarts in Amsterdam named Atta" never becomes
an edge — so the digest is honest but identity-thin, and the LME "14-floor" of incidental
personal facts stays unreachable. This is the highest-EV next build: it serves the digest,
real Hermes recall, AND the benchmark floor with one tier.

**Design sketch (additive, schema-constrained — the SAFE version of the shelved
open-ended incidental extraction):**
- New dream-phase extraction over USER turns only, closed slot vocabulary:
  `role`, `name`, `employer`, `location`, `language`, `relationship(person)`,
  `possession`, `age/birthday`, `health_condition` (sensitive — see redaction note),
  `recurring_activity`. Each: `{slot, value, evidence_message_id, confidence}`.
- New table `user_profile` (schema v18) — slot rows, bi-temporal like KG
  (`valid_at`/`invalid_at`, supersession on slot conflict mirrors P2 semantics).
  One LLM call per dreamed session batch (piggyback on the existing phase-1 batching).
- Consumers: (a) `_anchor_facts()` prepends profile rows to VERIFIED FACTS (profile >
  graph edges in priority); (b) a `ctx.user_profile` additive tier in `augment()` —
  small, always-relevant, cannot crowd; (c) `HyMem.profile()` accessor for hosts.
- Redaction: respect the existing `redaction.py` path; health slots must honor it.

**Front-run gate (free, offline):** before building, run the extraction PROMPT manually
over ~20 sessions from the prod store export and count precision/recall of slot
assertions by hand. Closed vocabulary should make precision ≥0.9; if it doesn't, stop.

**Tests:** extraction fixture-driven (StubLLM); supersession (new employer invalidates
old); anchor priority order; redaction; `profile()` empty without dream.

**Est:** the largest stage — prompt + migration + dream phase + 3 consumers. One focused
session.

---

## Stage 2 — Episode-extraction coverage (the 65/91 gap; dual payoff)

**Problem:** 26/91 sessions produced no episodes, so neither the digest nor the
aggregation clusters can ever cover them. Same root cause as the banked LME finding
(42% of MS misses had no gold episode). Fixing it widens the digest AND is the one
RAPTOR-adjacent lever that would also move the benchmark.

**Probe first (free, LLM-less, ~30 min):** on the prod store, characterize the 26
sessions: length distribution (are they short/single-exchange?), content type (tool
noise? pure Q&A?), `sessions.digested_version` (were they dreamed at all, or dreamed and
yielded zero?). Three different causes → three different fixes:
- never dreamed → scheduler/runner bug, fix is mechanical;
- dreamed, zero episodes, short sessions → episode extractor's minimum-content threshold
  too strict, or prompt refuses thin sessions → relax + "a single substantive exchange is
  an episode" instruction;
- dreamed, zero episodes, long sessions → extraction prompt recall problem → worth a
  dedicated fix + re-run of the banked MS coverage numbers.

**Gate:** build nothing until the probe says which bucket dominates.

**Est:** probe is small; the fix depends on the bucket (mechanical → small; prompt recall
→ medium, needs offline before/after on the same sessions).

---

## Stage 3 — Production enablement path (flip the default for Hermes)

The layer is still `aggregation_nodes_enabled = False` in prod. Before flipping:

**3a. Chaining guard (quality).** Connected-components over OR-links chains
transitively → one mega-cluster yields mush summaries. *Probe first:* cluster-size
distribution on the prod store (pure-Python, no LLM — reuse
`benchmarks/raptor_cluster_probe.py` loaders). If max cluster ≪ 15 episodes, skip the
guard entirely. If mega-clusters exist: cap component size (split by recency window at
cap, or require BOTH emb AND ent agreement to grow a component past the cap). Salt-bump
`cluster.v3` if fusion inputs change.

**3b. O(n²) scaling.** `cluster_episodes` is all-pairs Python cosine, rebuilt per dream
— fine at 10² episodes, death at 10⁴. Fix: candidate blocking — entity inverted index
(entity → episode ids; only pairs sharing an entity get the Jaccard test) + embedding
top-k neighbors via the existing vec path for the cosine arm. Keep the pure clusterer's
contract (probe re-exports it) — blocking only generates the candidate pair list.
*Gate:* time the current build on the prod store first; if < 2s, defer this stage.

**3c. Flip criteria.** Enable on the Hermes server when: v4 digest verified (Stage 0),
3a probe clean or guard built, dream-time cost measured and acceptable (log
`nodes=/reused=` over a week — steady-state should be near-full reuse). The QUERY tier
stays TR-gated; enabling the layer is primarily for the digest.

---

## Stage 4 — Query-time consumption v2 (only after Stage 3)

**4a. Sparse-signal fallback injection.** Fire the node tier when raw retrieval is THIN
(e.g. `len(message_hits) + len(fts_hits) < threshold`, or top BM25 score below floor)
instead of / in addition to ability gating. Principled: nodes appear exactly when there
is nothing for them to crowd, covering vague/global/cold-start queries ("what do we know
about my projects?") that LME never asks. Config: `aggregation_fallback_min_hits`
(0 disables). Tests offline; no LME run (invariant 1).

**4b. Drill-down API.** `HyMem.expand_node(node_id)` → member episodes (and for ≥1
levels, child nodes) — the RAPTOR tree-traversal read. `member_episode_ids` already
persisted; this is a thin read-only accessor + tests. Lets a host show "why does my
digest say X" with provenance — pairs well with Hermes UI later.

---

## Stage 5 — Digest delivery into Hermes (product wiring)

- Embedded host: inject `digest().summary` into the system prompt; refresh after each
  dream. Decide staleness display (`generated_at` is exposed).
- `server.py` (MCP): add a `digest` tool (8th tool) returning the dataclass fields.
- `honcho` adapter: map digest to whatever Honcho's user-representation endpoint
  expects (it's the natural analogue of Honcho's "dialectic" user model).
- Open design question for tomorrow: should `augment()` ALSO return the digest (e.g.
  `ctx.digest`) so single-call hosts get it? Leaning yes-but-optional
  (`cfg.augment_include_digest`, default False) to keep `augment()` lean.

---

## Suggested order for tomorrow

| # | Item | Type | Effort | Why this order |
|---|------|------|--------|----------------|
| 1 | Stage 0 v4 verification | box run | ~15 min | everything else reads its evidence |
| 2 | Stage 2 probe (26 sessions) | offline probe | ~30 min | free, decides Stage 2 build |
| 3 | Stage 3a probe (cluster sizes) | offline probe | ~15 min | free, decides guard |
| 4 | Stage 1 P4 profile tier | build | rest of day | highest EV; gate with the manual precision check first |
| 5 | Stage 2 fix | build | depends on probe | dual payoff once bucket known |
| 6 | Stage 3b/3c, 4, 5 | build | later | gated on the above |

The three probes together cost ~an hour and de-risk every build decision — run all three
before writing any feature code.
