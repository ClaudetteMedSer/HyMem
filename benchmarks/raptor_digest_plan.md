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

**RESULT (box run 2026-06-12): v4 verified, gate passed in spirit — proceed to Stage 1.**
- Acme demoted from stated-as-fact (root.v3) to "noted but not verified" (root.v4) — the
  evidence-bound clause works at all levels. NOT omitted: the graph has no contradictory
  identity edge for it to lose to. "Suspect" ≠ "omit" → clean omission needs a true
  identity edge (e.g. `(atta_van_westreenen, works_as, bedrijfsarts)`) so "senior
  engineer" contradicts. That is exactly the Stage 1 / P4 motivation — recorded.
- Anchor grounding confirmed: digest now names real graph entities
  (claudette_med_ser, agent37_containers, composio_git_hub) vs v3's generic names.
- Breadth retained and improved: 10+ threads (v3 had 7), more specific detail.
- Gate reading: the model is NOT inventing at episode level — it correctly applies
  "identity claim with no verified fact → suspect". No further prompt-tuning.
- Still outstanding (non-blocking): coverage-attribution query and the second-dream
  `reused=N` / byte-identical check.

The salt bumps mean the next dream regenerates every fusion (~13 calls on the 91-session
store — trivial). Checklist:

- [x] Re-dream with the layer enabled; confirm `aggregate.built nodes=N reused=0` in the
      log (full regeneration — proves the poisoned rollup is evicted).
- [x] `hy.digest()`: "Acme Corp" / invented identity GONE; breadth retained (7+ threads).
      *(Partial: demoted to "noted but not verified", not gone — see RESULT above.)*
- [x] Inspect what the anchor actually injected:
      `SELECT subject_canonical, predicate, object_canonical FROM knowledge_graph
       WHERE status='active' AND derived=0 AND invalid_at IS NULL
       ORDER BY pos_evidence - neg_evidence DESC LIMIT 20;`
      Expectation: tech edges present, personal identity (bedrijfsarts, name) likely
      ABSENT — that absence is the Stage 1 motivation, record it.
- [x] Coverage attribution: 66/92 sessions covered (71.7%) — confirmed via the Stage 2
      probe on 2026-06-12; the 26-session gap is Stage 2's (see its RESULT below).
- [x] Second dream, no changes → `reused=N` (all fusions), digest byte-identical.
      *(Confirmed 2026-06-12 post-build-wave: third dream pass 34/34 reused,
      digest byte-identical — the second pass legitimately regenerated because
      profile.v2 rows entered the anchor between passes. Stage 0 fully CLOSED.)*

**Gate:** if v4 still shows invented identity after cache eviction + anchor, do NOT
prompt-tune further — that's evidence the model invents at episode level (upstream of
the tree); jump to Stage 1/2 diagnostics instead.

---

## Stage 1 — P4: typed user-profile tier (feeds the anchor true identity)

**STATUS (2026-06-12): BUILT, on-box gate pending.** Schema v18 (`user_profile`,
closed-vocab CHECK), `profile.v1` extraction over USER turns (piggybacks the
per-session digest skip-guard — zero tail calls on unchanged re-dreams), bi-temporal
supersession (single-valued slots + relationship-per-person supersede; rest
accumulate; re-assertion reinforces, never duplicates), redaction at the persist
chokepoint. All three consumers wired: `_anchor_facts()` (profile rows above graph
edges, combined cap, feeds the root cache hash so profile changes regenerate the
digest), additive `ctx.user_profile` in `augment()`, `HyMem.profile()`. Config:
`profile_extraction_enabled` (True) / `profile_max_items_per_session` (16) /
`profile_context_cap` (24). 23 tests green; full suite 584 passed.
**The front-run precision gate still applies, now as a POST-build enable gate on the
box:** run `python benchmarks/profile_prompt_dump.py <prod store> --sessions 20`,
paste the rendered prompts into the box LLM, hand-score slot precision; ≥0.9 →
re-dream and check `hy.digest()` for Acme's clean omission (the anchor now has a
true identity edge to prefer); <0.9 → set `profile_extraction_enabled=False` and
revise `profile.v1`/validation before any prod dream.

**GATE RESULT (box run 2026-06-12): precision ~8% (≈8/98) — gate NOT passed.**
31 sessions, 98 rows extracted. Correct: ~7–9 (name, language, location,
relationship(titus skrabanja)=supervisor). Failures, by mode:
1. `health_condition` bleed — role descriptions ("works as bedrijfsarts with
   verzuimbegeleiding") and PATIENT facts ("chronische vermoeidheid",
   "patient finally booked psychologist") landed as user health.
2. `employer` misresolution — GitHub org (ClaudetteMedSer) taken as employer;
   the real employer (O3) missed.
3. `possession` over-extraction — 20+ rows enumerating individual GitHub repos;
   redundant `recurring_activity` rows. ~85 of 98 rows are noise.
4. Over-conservative on real facts — `role=bedrijfsarts` and `employer=O3` are
   explicit in prior_memory text but were not extracted (`role=developer` came
   out instead — incomplete).
**Root cause:** the prompt can't distinguish "facts ABOUT the user" from "facts
IN the user's context" — prior_memory_file dumps carry tech context (repos,
agent config) and clinical reflections (patients), and v1 extracts from both.
**Despite the gate failure, the ANCHOR MECHANISM is verified:** regenerated
digest titled "Atta Van Westreenen — Developer Profile", ZERO mention of Acme
Corp — real identity facts outcompeted the hallucinated one. "Suspect → omit"
confirmed end-to-end; what failed is extraction precision, not consumption.

**Response (built 2026-06-12, this branch): profile.v2.** Per the gate contract:
- `profile_extraction_enabled` default flipped to **False** — no prod dream
  extracts until the v2 prompt re-passes the ≥0.9 gate.
- `profile.v2` prompt rewrite targeting the four failure modes above (the
  aboutness test: user-vs-context; clinician-patient exclusion; employer ≠
  GitHub org; possession = durable real-world items only; pasted memory text
  that explicitly describes the user IS extractable — channel ≠ aboutness).
- Migration 019 (schema v19): purge the failed v1 rows (`DELETE FROM
  user_profile` — regenerable, v18 never released) + per-session
  `sessions.profile_prompt_version` stamp, decoupling the profile skip-guard
  from the digest guard so a prompt-version bump re-extracts on already-digested
  sessions (v1 sharing the digest guard would have made v2 unreachable without
  a full re-dream under a bumped digest prompt).
**Re-gate procedure (box):** same as before — `profile_prompt_dump.py`,
hand-score ≥0.9, only then set `profile_extraction_enabled=True` and dream.

**RE-GATE RESULT (box run 2026-06-12): profile.v2 PASSED — Stage 1 CLOSED.**
Raw 83.7%, adjusted ~95% (≈123/129; the raw score penalized verbose-but-correct
role variants like "aios bedrijfsgeneeskunde" against a too-strict GT set).
Every v1 failure mode fixed and verified: ZERO health_condition bleeds, ZERO
GitHub-org employers, ZERO repo possessions, and the previously-missed
role=bedrijfsarts / employer=O3 now extracted (the aboutness inline example
working). Zero Acme/senior-engineer extraction — the beam-test session handled
correctly. Residual imperfections (NOT worth a v3 re-gate cycle):
- 2× relationship without slot_key ("supervisor" sans "titus skrabanja") —
  validation drops these at persist time, so they cost recall, not precision;
  a one-line prompt nudge IF a v3 ever happens for other reasons.
- 1× employer="not explicitly stated" — LLM emitted meta-reasoning as a value;
  genuine but singular.
**Default flipped back ON (`profile_extraction_enabled=True`) per the gate
contract.** Any future material prompt change re-enters the same gate: bump
PROFILE_PROMPT_VERSION, default False, re-score ≥0.9, re-enable.
End-to-end stabilization confirmed: profile rows entered the anchor → root
regenerated once (expected — anchor hash changed), then third dream 34/34
reused, digest byte-identical, Acme omitted across all three passes.

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

**Probe READY (2026-06-12):** `benchmarks/episode_coverage_probe.py` — read-only
(sqlite mode=ro), buckets every zero-episode session into never_dreamed /
dreamed_zero_short / dreamed_zero_long and prints the verdict guide. Run on the box:
`python benchmarks/episode_coverage_probe.py <prod store> --json coverage_audit.json`.

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

**PROBE RESULT (box run 2026-06-12): never_dreamed dominates — mechanical fix.**
Coverage 71.7% (66/92). Of 26 uncovered: 23 never_dreamed (88.5%), 3
dreamed_zero_short (11.5%), 0 dreamed_zero_long. The 23 are test/WebSocket/
diagnostic sessions with 0–4 messages. No prompt work needed.

**Root cause (located in code):** both chunk tiers only mint chunks from USER
turns that clear `salience_min_chars` or a trigger regex
(`hymem/dreaming/chunks.py`); a session whose user turns are all short produces
zero chunks in both tiers, and the runner's `if not chunks and not baseline:
continue` skips the entire per-session tail — digest never runs,
`digested_prompt_version` stays NULL forever.

**Fix (built 2026-06-12, this branch): short-session fallback chunk.** When a
session reaches the tail with zero chunks in both tiers but has ≥1 user/assistant
message, mint ONE fallback chunk spanning the whole session
(`salience_reason='short_session_fallback'`, char-capped), persist + embed it
like any other chunk, and let the normal digest tail run (episodes get a valid
evidence chunk id; `digested_prompt_version` gets stamped; skip-guard then makes
re-dreams free). Phase-1 triple extraction is NOT run on fallback chunks — the
goal is digest/episode coverage, not graph growth from diagnostic noise.
Truly empty sessions (zero user/assistant messages) still skip. Expected effect
on the box: next dream digests the 23 sessions (23 one-off LLM calls); honest
residue moves to dreamed_zero_short where the digest legitimately found nothing.

---

## Stage 3 — Production enablement path (flip the default for Hermes)

The layer is still `aggregation_nodes_enabled = False` in prod. Before flipping:

**3a. Chaining guard (quality).** *Probe READY (2026-06-12):*
`benchmarks/cluster_size_probe.py` — read-only, reuses the production
clusterer/loader verbatim and the production thresholds (0.55/0.50), histogram +
largest-cluster membership + verdict at `--cap 15`. Run on the box:
`python benchmarks/cluster_size_probe.py <prod store> --json cluster_sizes.json`.

Connected-components over OR-links chains
transitively → one mega-cluster yields mush summaries. *Probe first:* cluster-size
distribution on the prod store (pure-Python, no LLM — reuse
`benchmarks/raptor_cluster_probe.py` loaders). If max cluster ≪ 15 episodes, skip the
guard entirely. If mega-clusters exist: cap component size (split by recency window at
cap, or require BOTH emb AND ent agreement to grow a component past the cap). Salt-bump
`cluster.v3` if fusion inputs change.

**PROBE RESULT (box run 2026-06-12): mega-cluster confirmed — build the guard.**
348 episodes in ONE component at cap 15, spanning 61 sessions
(2026-05-07-hymem-setup through Branch-Update-Verification-Reviewed) — a single
giant component via transitive chaining through the embedding arm.

**Guard (built 2026-06-12, this branch): recency-window split at cap.**
`cluster_episodes` gains `max_cluster_size` (config
`aggregation_max_cluster_size`, default 15; 0 = uncapped, translated to None
at the call sites): components larger than the cap are split deterministically
into recency-ordered windows of ≤ cap (the plan's sanctioned option — the
BOTH-arms alternative is union-find order-dependent). Recency signal is
`start_message_id` (store-wide AUTOINCREMENT = true cross-session ingestion
clock; session_id is lexicographic, NOT chronological on the prod store); full
windows align to the most-recent end so the one possibly-undersized window
holds the OLDEST episodes — downstream min-members/min-sessions filtering then
drops the least recent slice, never the newest. Applied at BOTH call sites
(level-0 `select_clusters` and the rollup loop — a frontier mega-component
would otherwise fuse from a truncation of itself). Salt bumped
`cluster.v2 → cluster.v3` (membership semantics changed); `rollup.v2`/`root.v4`
unchanged — their cache ids already key on member sets, which change
naturally. The probe keeps measuring RAW chaining (calls the clusterer
uncapped) so it stays the diagnostic for the guard.

**3b. O(n²) scaling.** `cluster_episodes` is all-pairs Python cosine, rebuilt per dream
— fine at 10² episodes, death at 10⁴. Fix: candidate blocking — entity inverted index
(entity → episode ids; only pairs sharing an entity get the Jaccard test) + embedding
top-k neighbors via the existing vec path for the cosine arm. Keep the pure clusterer's
contract (probe re-exports it) — blocking only generates the candidate pair list.
*Gate:* time the current build on the prod store first; if < 2s, defer this stage.

**TIMING RESULT (box run 2026-06-12): 395 episodes, 77,815 pairs, 4.04s > 2s —
gate fired, do NOT defer.** Build the blocking as designed. Notes pinned before
the build:
- The entity arm is EXACT under blocking: Jaccard ≥ 0.5 requires ≥1 shared
  entity, so the inverted index loses nothing.
- The cosine arm is approximate (top-k KNN via `vec_episodes`): with
  k ≥ n−1 it is exact, so small stores lose nothing; at scale a missed link
  means both endpoints already had ≥k closer neighbors — and the cap-15
  windowing makes marginal membership drift immaterial anyway.
- NO salt bump: blocking changes membership, not the fusion prompt; cache ids
  already key on member-set hashes, so changed clusters regenerate naturally
  and unchanged clusters keep their valid cached fusions. (The 3a bump was for
  the cap semantics; the salt rule is about prompt-level staleness.)
- No `vec_episodes` table (sqlite_vec absent) → fall back to exact all-pairs,
  today's behavior; embedded small stores are unaffected.

**BUILT + VERIFIED (2026-06-12, this branch).** `cluster_episodes` gains
`candidate_pairs` (None = exact all-pairs — the probe contract);
`generate_candidate_pairs(conn, episodes, emb_top_k)` builds the set (entity
inverted index + KNN k+1 over `vec_episodes`, rowid-mapped; returns None →
exact fallback when top_k≤0, the vec extension won't load, or the table is
absent). Config: `aggregation_blocking_top_k = 24` (0 disables). Wired through
`select_clusters(…, conn=None)`; the rollup loop stays exact (frontier items
are few and not in vec_episodes). Salts untouched per the no-bump rationale.
Verified independently on a synthetic 400-episode store (40-center vectors,
sparse entities): candidates = 10.0% of all-pairs, 1.675s → 0.258s (6.5×),
components byte-identical, k≥n−1 exactness holds.

**BOX RE-TIME (2026-06-12): 4.04s → 1.27s (3.2×), pairs 79,401 → 19,561
(24.6%), components IDENTICAL — under the 2s gate. Stage 3b CLOSED.**
The 3.2× vs the synthetic 6.5× is denser real-world entity overlap (the exact
entity arm keeps 25% of pairs vs 10% synthetic); the KNN cosine arm is the
part that scales, so headroom grows with store size. Operational note from the
re-time: a bare `core_db.connect()` does NOT load the vec extension on its own
— the blocking guard then correctly degrades to exact all-pairs (by design,
but worth knowing when timing/probing by hand: initialize the store the way
HyMem does, or you measure the fallback path).

**3c. Flip criteria.** Enable on the Hermes server when: v4 digest verified (Stage 0),
3a probe clean or guard built, dream-time cost measured and acceptable (log
`nodes=/reused=` over a week — steady-state should be near-full reuse). The QUERY tier
stays TR-gated; enabling the layer is primarily for the digest.
*Status 2026-06-12: criteria 1 and 2 MET (Stage 0 closed; guard built and
verified ≤15 on the box). Steady-state reuse already looks right (34/34 on the
no-change pass) — what remains is the week-scale cost observation, then flip.*

**INSTRUMENTATION LANDED (2026-06-13, this branch).** The `nodes=/reused=`
signal this criterion watches was only reaching a `log.info()` at
`aggregate.py:711`, and the MCP server runs without `logging.basicConfig`, so
production dropped it silently — we had the flat-34 circumstantial evidence but
not the exact reused count. Fixed at the source instead of the log:
`build_aggregation_nodes` now returns `AggregationResult(nodes, reused)`, and
the runner persists BOTH into `dream_runs` per cycle (`aggregation_nodes_built`,
`aggregation_nodes_reused` — schema v20, migration 020). `built` was likewise
computed-then-dropped before this; both gaps closed. So the week-scale dataset
now accrues durably as a queryable column rather than a scraped log line —
captured forward from deploy (the June 11-12 runs stay unrecoverable: full-
replace keeps no prior-membership baseline to diff, so no post-hoc DB
reconstruction was possible). Secondary: added `logging.basicConfig` (env-tuned
via `HYMEM_LOG_LEVEL`, default INFO) to the `hymem-server` entry point, which
also closes the parallel blind spot where the `aggregate.build_failure`
exception was being dropped. 663 tests green.

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

**4b BUILT (2026-06-12, this branch — pulled forward of the 3c flip since it
touches nothing query-time; 657 tests green).**
- `expand_node(conn, node_id)` in `aggregate.py` → `NodeExpansion`: the node's
  own fields plus its members resolved ONE level down — `child_nodes`
  (`NodeChild`) and `episodes` (`NodeMemberEpisode`, carrying session_id +
  start/end message ids so a host can walk a digest claim down to the raw
  turns). Member order = persisted fusion-input order. `missing_member_ids`
  keeps the read honest instead of silently shrinking (only reachable via
  store surgery; pinned by test).
- Entry point: `Digest` gained `node_id` (the root's id; additive,
  default ""), so the traversal starts from `digest().node_id` — or from an
  `AggregationNodeHit.node_id` out of the query tier.
- `HyMem.expand_node()` reads via `read_conn`; returns None for unknown/stale
  ids (node ids are rebuilt each dream). No LLM, no schema change, no salt
  (read-only delivery). Exported: `NodeExpansion`/`NodeChild`/
  `NodeMemberEpisode` from the package root.
- NOT built into MCP/Honcho surfaces — per the plan this is the embedded/UI
  read; wire a tool/endpoint only when a concrete Hermes UI consumer exists.

---

## Stage 5 — Digest delivery into Hermes (product wiring)

**BUILT (2026-06-12, this branch). All four surfaces wired; 652 tests green.**

- Embedded host: inject `digest().summary` into the system prompt; refresh after each
  dream. Decide staleness display (`generated_at` is exposed).
- `server.py` (MCP): add a `digest` tool (8th tool) returning the dataclass fields.
- `honcho` adapter: map digest to whatever Honcho's user-representation endpoint
  expects (it's the natural analogue of Honcho's "dialectic" user model).
- Open design question for tomorrow: should `augment()` ALSO return the digest (e.g.
  `ctx.digest`) so single-call hosts get it? Leaning yes-but-optional
  (`cfg.augment_include_digest`, default False) to keep `augment()` lean.

**What was built, and the decisions taken:**
- **Staleness display DECIDED via one canonical render:** `Digest.as_context_block()`
  — `## title` + summary + a single provenance footer
  `(Memory digest covering N of M sessions; generated <generated_at>.)`.
  Every delivery surface uses this method, so the staleness decision lives in
  exactly one place. Embedded-host pattern documented on `HyMem.digest()`:
  inject the block, re-fetch after each `dream()` (the digest only changes at
  dream time — nothing to poll between dreams).
- **MCP:** `hymem_digest` (8th tool) returns `as_context_block()`, or an
  explanatory "not built yet" message instead of an error when no root exists.
- **Honcho:** new `_peer_representation()` helper = digest block ABOVE USER.md,
  consumed by BOTH user-representation surfaces (`GET .../peers/{pid}/card` and
  `peer_representation` in `GET .../peers/{pid}/context`). Degrades to plain
  USER.md when no digest exists (today's behavior, pinned by test).
- **Open question RESOLVED as leaned: yes-but-optional.**
  `cfg.augment_include_digest` (default False) → `ctx.digest: Digest | None` on
  `AugmentedContext`. Additive like the profile/aggregation tiers (own SELECT,
  never a retrieval competitor); default-off keeps `augment()` lean since the
  digest is standing dream-time context, not per-query context.
- NOT touched: salts (no fusion-prompt change — delivery only), retrieval tiers,
  and the Stage-3c flip (`aggregation_nodes_enabled` stays False in prod until
  the week-scale cost watch passes; these surfaces all degrade gracefully to
  "no digest" until then).

---

## Next box run — COMPLETED 2026-06-12, all gates passed

1. ✓ Auto-migrated to v19, v1 rows purged.
2. ✓ Coverage 66→71/92 (77.2%); never_dreamed 23→9 — the 9 are ALL 0-message
   WebSocket placeholders, which the fallback correctly skips (working as
   designed). dreamed_zero_short now 12: diagnostic/trigger sessions (1–4 msg
   tech content) where the fallback chunk exists but the episode extractor
   legitimately titles nothing — arguably correct; covering them would need an
   episode-extraction prompt tweak (a "single substantive exchange is an
   episode" instruction), parked as an OPTIONAL Stage 2b below.
   ✓ Cluster guard: level-0 nodes all ≤15 members (was one 348-component);
   probe still reports the raw 352-mega-cluster by design (it measures
   uncapped chaining).
3. ✓ profile.v2 re-gate PASSED ~95% adjusted (see Stage 1 RE-GATE RESULT);
   default flipped back ON.
4. ✓ Reuse check: third dream 34/34 reused, digest byte-identical; Acme
   omitted across all three passes. Tests on box: 621 passed / 2 skipped.

**Optional Stage 2b (parked, decide later):** the 12 dreamed_zero_short
diagnostic sessions. Current behavior is defensible (no episode in "diagnostic
trigger" content); only build the prompt tweak if digest breadth over those
sessions ever matters. Same front-run contract: offline before/after on those
12 sessions first.

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
