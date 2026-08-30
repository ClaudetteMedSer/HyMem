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
drops the least recent slice, never the newest.
**REVERSED 2026-07-05 (dream runs 685-693): windows now anchor at the OLDEST
end.** Newest-end alignment shifted every window boundary whenever an episode
joined the mega-component between dreams, re-keying all ~24 windows plus the
rollup chain above them — reuse collapsed to ~30% on every episode-adding
dream (the constant ~13 reused nodes were the small clusters outside the
component). The 678/680 rowid ceiling (8b36501) couldn't catch this: it guards
mid-build arrivals, not between-dream ones. Oldest-anchored windows confine an
append to the still-filling newest tail window; the undersized-newest slice a
min-size filter drops stays retrievable as episodes and enters the digest as
leftover leaves. No salt bump (blocking precedent: member-set hashes re-key
naturally). Applied at BOTH call sites
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

**ENABLE PATH LANDED (2026-06-17, this branch).** The June-13 columns were
accruing `0/0` every cycle because the layer was never actually ON in prod:
`build_from_env` threaded only `root` into `HyMemConfig`, so the server always
ran the `aggregation_nodes_enabled = False` default — there was no env var to
flip it. (The flat-34 "steady-state reuse" was the June-12 one-shot demo
script's nodes, not a production signal — production had never built any.) Three
fixes so the week-scale data can actually start:
- `HYMEM_AGGREGATION_NODES_ENABLED` (+ `HYMEM_AGGREGATION_DIGEST_ENABLED` to
  isolate level-0 node-build cost from digest-rollup cost) are now resolved in
  `resolve_env`/`build_from_env` via an overrides dict — an *unset* var stays
  `None` and the dataclass default wins, so the shipped default stays off until
  the flip, and a future default change stays authoritative.
- A startup `log.info` confirms when the layer is enabled (and the effective
  digest state), so a deploy can verify the flag took.
- Re-added the counts to the `dream.end` log line (`agg_nodes=/agg_reused=`) —
  they were persisted to `dream_runs` but absent from the log, so the grep-able
  signal this criterion describes did not exist.
*Status: criteria 1 and 2 still MET. Criterion 3 (week-scale cost) is now
actually collectable — the clock starts at the next server deploy with
`HYMEM_AGGREGATION_NODES_ENABLED=true`. Read it right: the first dream reports
~all-built / ~zero-reused (full-replace wipes the stale demo nodes, whose
member-set hashes no longer match the grown store); reuse climbs toward
near-full on runs 2..N — that ratio is the flip signal. The flip itself stays
the one-line config-default change, now backed by data rather than the demo
pass.*

**FLIP-WATCH RESULT — v4 run, 2026-08-26** (window `--since 2026-08-08`, `--check-episodes` on, 135 window rows, 17 verdict rows, floor ≥5 met).
**G-FLIP: FAIL.** 14/17 at reuse ≥90%. Offenders: **#1178 = 88.8%, #1180 = 87.0%,
#1183 = 79.2%** (all Aug-8 deploy-day rows; #1178/#1180 ran pre-v31). Zero
blocking-flips, zero failure rows, zero unclassifiable rows. Criterion 6 n=3
(#1182–#1184, all residual=0 — clean, but under the 5-row floor).
Verdict recorded as FAIL; NOT banked as a pass on any criterion — the window
is contaminated (see env-parity note below). CSV
`~/.hermes/benchmarks/flipwatch_2026-08.csv`, full write-up
`benchmarks/flipwatch_result.md`.

**APPEND-CAPABILITY CORRECTION (2026-08-26).** The "store grew 106 / append
genuinely exercised" line was a window-aggregate instrument artifact, and it
was wrong. The honest quantities: **(a) verdict-set-visible append = +15
episodes** — summed `aggregation_input_episodes` deltas over the surviving
aggregation rows (#1169–#1184), all at churn **+1**; **(b) 91 episodes**
arrived during the dead era (`episodes.created_at >= 2026-08-09 22:16` = 91,
matching the #1303 delta) and never entered any verdict row's delta. So the
append criterion is exercised at granularity +1 only; the +91 backlog churn
was untested by the verdict set. The only sub-bar v31 append evidence remains
**#1183 = 79.2%** — which raises, not lowers, the stakes on its amplification
analysis (RESOLVED below: forecast-exact, discharged). Same class as the
third watch's paper pass: a window-aggregate check
cannot see a defect that splits the window in two.

**#1303 — env-parity artifact row (recorded, NOT excused).** `2026-08-26
09:43:19–09:45:55`: input 1044→1135 (delta +91), built=103, reused=68
(66.0%), failures=0, blocking=knn, level0-missed=3, leaf-changed=1,
**predicted=35 == rebuilt=35, residual=0**. It ran on the flag-carrying MCP
path BEFORE the launcher fix, so it is not "the settling dream" of the
re-enabled layer and gets no excusal label. Recorded because it is real
aggregation data: criterion 6 held (residual 0) through a +91 append, and the
structural forecast was exact. With it, criterion 6 has n=4 populated values
(all residual=0) — still under the 5-row floor.

**ENV-PARITY DEFECT — the real reason the watch died (this is the keystone).**
Three executors carried two different layer states:
| path | flag | since |
|---|---|---|
| honcho (post-restart.sh launch) | OFF — `config.py:112` default, var absent; `build_from_env` reads `os.environ` only, no dotenv | restart 2026-08-09 22:16 |
| periodic `hymem-dream.sh` | OFF — `HyMemConfig` default; it sources `/home/node/.hymem.env` which did not exist | all along |
| MCP server (`hymem-server-wrapper:14`) | ON — `export HYMEM_AGGREGATION_NODES_ENABLED=true` | wrapper changed 2026-08-10 19:48 |

Consequence: **118 consecutive no-agg rows (#1185–#1302, 2026-08-09 22:16 →
2026-08-26 08:58)** — silent exclusions, indistinguishable from "nothing to
do" in the classifier (`flipwatch_classify.py:377-380`). This is the second
time an unset env var has zeroed this watch (cf. the 2026-06-13 0/0 columns
and the 2026-06-17 "ENABLE PATH LANDED" note; the flat-34 "steady-state
reuse" was a demo script, not production). The fragility: `False` default =
unset var silently disables the layer, and the flip that removes that
fragility is gated on the watch the fragility keeps killing.

**FIX LANDED 2026-08-26** — all three paths now agree, flag ON:
1. `post-restart.sh` honcho launch: var added above the env block (bash -n
   clean; honcho restarted, `/proc/<pid>/environ` + `/health` verified).
2. `/home/node/.hymem.env` created with the var (the file the periodic script
   sources — it did not exist).
3. `hymem-dream.sh`: exports the var before sourcing (with a comment; layered
   env-line comments break `env` — keep them OUTSIDE continuation lists).

**PRE-REGISTRATION — post-fix window (banked 2026-08-26, BEFORE any post-fix
row exists).** The first dreams after the fix are ordinary append rows: no
new label, no "settling"/"deploy-refusion" excusal (no salt bump), they are IN
the verdict. Re-anchor the watch as follows:
- `--since 2026-08-26 11:15` — after the last env-split row (#1303 ended
  09:45:55) and after all three paths agree.
- `--restart-reason "env-parity defect fixed 2026-08-26: all three trigger
  paths (post-restart.sh, hymem-dream.sh, MCP wrapper) now pass
  HYMEM_AGGREGATION_NODES_ENABLED=true; rows before 2026-08-26 11:15 are
  artifacts of the split (honcho+cron OFF since 2026-08-09 22:16, MCP ON
  since 2026-08-10 19:48)"`.
- **Excuse reuse%, never criterion 6.** Reuse% gates COST, residual gates
  CORRECTNESS. A low-reuse append row is a cost signal, not a defect. A
  positive `aggregation_keying_residual` is a defect with no tolerance,
  regardless of reuse. Prediction: **residual == 0 on every aggregation row
  in the new window** (correctness half).
- **COST HALF IS ALSO PREDICTED: reuse% ≥ 90 on the first incremental
  append.** Rationale: the +91 backlog was consumed by #1303 OUTSIDE the
  window, so the row this session kicks off is the first incremental append
  the 90% bar has ever seen on v31+v32 code — the cost mechanism is
  untested post-fix. If this first row lands reuse < 90% WITH residual == 0,
  that is NOT a settling artifact and NOT a keying defect: it is a
  **windowing-confinement finding** (per the 2026-07-05 reversal, an
  append's churn must stay confined to the still-filling newest tail
  window; a small Δ re-keying many member sets means the confinement
  invariant is leaking) and it routes to the plan's windowing re-analysis,
  not to "extend and ignore".
- Criterion 6 floor: **n=4 carried, not n=0** — see the carry rule below.
  The banked floor is n≥5 populated; with the carry, ONE more
  residual=0 row reaches it; a bank/pass verdict must not be drawn at
  n=4.

**CRITERION-6 CARRY RULE (banked 2026-08-26, before the new window's first
row).** How much of the pre-fix window crosses the `--since 2026-08-26 11:15`
restart boundary — split by quantity KIND, matching the instrument's own
cost/correctness split:
- **Reuse% does NOT cross.** It is a cost rate; the pre-fix era is not
  representative of the fixed-launcher regime (env split, MCP-path-only ON).
  Post-fix reuse below 90% is the gate's verdict, not a carry-over.
- **Criterion 6 (keying integrity — `aggregation_keying_residual`) DOES
  cross.** It is a correctness invariant: structural, not sampled; it reads
  the whole keying in one dream; there is no variance band to absorb. A
  correctly-instrumented row is valid evidence regardless of which window
  it fell in, and the restart cause (env parity) has no bearing on whether
  a keyed id matched.
- Accounting: n=4 carried (#1182, #1183, #1184, #1303 — all residual=0) +
  one fresh post-fix residual=0 row = floor n≥5. The new window opens at
  criterion-6 n=4, not n=0; this is the cost/correctness distinction the
  gate already draws, written beside the pre-registration — deliberately
  NOT a silent default in the classifier.

**#1183 — CORRECTNESS discharged, COST still open (re-resolved 2026-08-26
after an over-reach was pulled back).** One fact only:
`aggregation_keying_residual == 0` on all four populated rows (#1182,
#1183, #1184, #1303). By definition `residual = rebuilt − predicted` and
`rebuilt = built − reused`, so residual == 0 ⟺ rebuilt == predicted —
four rows are **four instances of one fact** (criterion 6 passing), not
independent corroboration of a second claim. What it discharges: every
cache miss on those rows was a genuine member-set change — the
salt/hash/rowid-shadow class criterion 6 exists to catch is clean.
What it does NOT discharge: why 22 member sets changed on a Δ=1 append
(#1183: input_eps 1042→1043, l0miss=3, leaf_changed=1, built 108→106,
reused 99→84, 79.2%). That is the COST question, and it is live: the
2026-07-05 oldest-anchored windowing reversal exists to confine an
append's churn to the still-filling newest tail window — 22 rebuilds from
one arrival reads like the confinement invariant leaking. The recorded
`aggregation_leaf_changed=1` splits it partially (some of the 22 may be
leaf_term — unclustered episodes entering the digest rollup, which move
on every arrival — rather than the tree proper), but the flag is binary
and no per-term decomposition is persisted, so that split is qualitative.
The plan's own routing for this row class: REOPEN THE WINDOWING ANALYSIS
with the row's `input_eps`/`built` delta as starting evidence — NOT
"discharge". **Pre-registered bound (fallback ONLY — dead letter inside
the deciding window):** for rows WITHOUT a forecast, A(Δ=1) ≤ 4.0·l0miss
(observed Δ=1 max, n≥10; the docstring's "~3.3 at #1158" is one member of
that band). Every post-v32 row carries a forecast, so the envelope only
ever applies OUTSIDE the deciding window — kept as a fallback, NOT
counted as a satisfied precondition.
**Consequence for the old window:** the three below-bar offenders carry no
keying-defect evidence (correctness clean on all four rows). The COST
half is left to the new window: the 90% bar on incremental appends is
UNTESTED on v31+v32 code (the +91 backlog was consumed by #1303 outside
the window) — the first post-fix row tests it, outcome pre-registered
above (reuse ≥ 90 predicted; < 90 with residual == 0 = windowing finding).
The old window's FAIL stays banked as-is.

**INSTRUMENT GAP — BUILT 2026-08-26, committed separately (see the
flipwatch-v32 commits).** (b)
one nullable column set from the effective config at dream start
(`aggregation_effective`), plus a classifier label that hard-FAILs on a
no-agg streak (≥5 consecutive, tail-of-window) instead of silently excluding.
`MIN_VERDICT_ROWS` is implemented — the sanity floor is fine; the hole is
that `built == 0` is indistinguishable from "layer switched off".

### FLIP-WATCH RESULT — v6 run, re-anchored window, 2026-08-26: **G-FLIP PASS → FLIPPED**

Window `--since 2026-08-26 11:15` (the pre-registered re-anchor; restart reason
= env-parity fix). 5 verdict rows. **All seven checks green.**

| # | check | bar | measured | result |
|---|---|---|---|---|
| 1 | every verdict row ≥ 90% | 5/5 | 5/5, min **91.3%** | PASS |
| 2 | blocking-flip rows | 0 | 0 | PASS |
| 3 | failure-attributed rows | ≤2 | 0 | PASS |
| 4 | unclassifiable low-reuse rows | 0 | 0 | PASS |
| 5 | sanity floor | ≥5 | 5 | PASS |
| 6 | dead-watch streak (v5 guard) | <5 | longest **0** | PASS |
| 7 | keying integrity (`residual > 0`) | 0 | 0 | PASS |

| row | time | kind | reuse | rebuild | resid |
|---|---|---|---|---|---|
| #1309 | 11:45 | append, eps Δ+1 | 92.2% | 2.7/miss | 0 |
| #1310, #1311 | — | quiescent | 100% | — | 0 |
| #1312 | 14:30 | append, eps Δ+1 | 91.3% | 3.0/miss | 0 |
| #1313 | 14:44 | quiescent | 100% | — | 0 |

**The cost half — the open question this window existed to answer — holds.**
The pre-registered prediction was reuse ≥ 90 on an incremental append; both
appends cleared it (92.2, 91.3) at 2.7–3.0 rebuilds per level-0 miss, the
classic Δ=1 band (2.2–4.0). **No #1183-class leak in-window**: nothing
resembling 7.3/miss appeared, so the windowing-confinement question raised by
#1183 stays open as a historical row and is NOT reproduced by current code.
Criterion 6 reaches n=5 (4 carried + fresh rows), all residual 0.

**Advisory, recorded and deliberately NOT treated as a bar: thin append
coverage.** Only #1312 counts as window-internal append evidence (#1309's Δ
straddles the anchor, so the banked split-window rule credits it to neither
side). One counted append is thin. It is recorded as an advisory because
**raising a bar after seeing a PASS is post-hoc tightening — structurally the
same move as the loosening the A5 verdict refused two waivers for.** The
pre-registered criteria (including quiescent rows counting toward
`MIN_VERDICT_ROWS`) were banked before the numbers existed and are met. The
correct response to thin evidence is to keep reading, not to withhold the
flip: the classifier keeps running on the post-flip window against the same
bar, and an append below 90% later is a finding to act on — actionable
precisely because the bar was not moved to avoid it.

**FLIPPED 2026-08-26** — `hymem/config.py:112`
`aggregation_nodes_enabled: bool = False → True`.

Two scope limits written into the docstring, because the flip is narrower than
it looks:

- **Near no-op for prod.** The box has run with the layer ON via
  `HYMEM_AGGREGATION_NODES_ENABLED=1` on all three launch paths since the
  env-parity fix earlier the same day. What the flip removes is the
  FRAGILITY — an unset env var no longer silently disables the layer, which
  is what zeroed this watch twice (2026-06-17, 2026-08-09). **Corollary: the
  post-flip verification dream is a WEAK test.** The effective config is
  unchanged, so it cannot fail for flip-related reasons; treat a refusion
  there as evidence that something ELSE moved (salt bump, member-set shift),
  not as the flip being validated. Post-flip the env var becomes an explicit
  OFF switch (`bootstrap.py:86` semantics need no change).
- **No benchmark regime changes.** `msc_adapter` (reused wholesale by
  `locomo_adapter`) and `beam_adapter` were default-config consumers and now
  pin `overrides["aggregation_nodes_enabled"] = False` explicitly (dict-item
  form — that is the string to grep for), so the LoCoMo / MSC / BEAM
  canonical baselines stay comparable to every run behind them.
  `longmemeval_adapter:532` already pinned it True. Moving a benchmark onto
  the shipped config is a pre-registered scored decision, never a side effect
  of a default change — the same argument that keeps the LoCoMo canonical
  deliberately narrow.

Downstream: Plan C **UNBLOCKED** (`additional_planning.md`); Stage 4a remains
gated on its own build spec below, not on this flip.

### The leaf-changed family — instrument BUILT 2026-08-26 (schema v33)

Three rows now share one signature, and the post-flip verification dream made
it a population rather than a curiosity:

| row | reuse | leaf_changed | l0miss | rebuilt | rebuilds/miss | residual |
|---|---|---|---|---|---|---|
| #1183 | 79.2% | 1 | 3 | 22 | 7.3 | 0 |
| #1307 | 81.4% | 1 | 3 | 19 | 6.3 | 0 |
| #1317 (post-flip) | 80.6% | 1 | 3 | 20 | 6.7 | 0 |

Against a Δ=1 band of 2.2–4.0, and against #1309/#1312 which sat inside it at
2.7/3.0. **Residual 0 settles CORRECTNESS on all three** — every rebuild was a
genuine member-set change, so the salt/hash/rowid-shadow class is clean. It
says nothing about COST, which is what the reuse bar gates and what these rows
are low on. Reading the residual as a discharge is the move this plan already
forbids for this row class (see the #1183 paragraph above): the standing
"leaf-set change re-keys a subtree, structural not a regression" explanation
may well be right, but **`aggregation_leaf_changed` is BINARY and cannot
distinguish**

- (a) the leaf term explains all 20 rebuilds — benign digest cascade, from
- (b) the leaf term explains 3 and the tree leaked 17 — the 2026-07-05
  oldest-anchored windowing confinement failing.

So the family stayed arguable across three separate readings, which is the
real cost: not a wrong answer, an *undecidable* one.

**Fix: split the rebuild by tree level** (`aggregation_rebuilt_level0` /
`_rollup` / `_root`, migration 033). Level-0 rebuilds track episode arrivals
into clusters; level ≥1 interior rebuilds track the digest leaf set shifting
and cascading up; the root keys on membership OR the anchor-facts hash, so it
is counted separately rather than smeared into the cascade term. Derived from
the build's own `rows` with no new state, no extra pass and nothing fitted —
the same move v31 made for keying. The three counts sum to `built - reused` by
construction, so the instrument is self-checking; a split that stops summing is
lying, and the test asserts it end-to-end through the store.

**On the next leaf-changed row this decides the family outright:** l0=3 with
rollup≈16 closes it benign; l0≈17 reopens the windowing analysis with the row
as its starting evidence. No bar, no accrual, one dream.

Deliberately NOT done: no threshold, no gate criterion, no classifier change.
This adds an observation channel to a question that was being answered
qualitatively — it does not answer it in advance, and it must not be read as
having pre-judged which branch is true.

**First live v33 row — #1319, 2026-08-26 17:37:58.** `rebuilt_level0/rollup/root
= 3 / 4 / 1`, and `3 + 4 + 1 = 8 = predicted = built - reused`. The
self-check asserted twice offline (`test_decomposition_sums_to_actual` at unit
level, `test_decomposition_is_persisted_on_the_dream_run` through the store)
holds end-to-end on the instrument's first production row. It is a
`leaf_changed=0` row, so it calibrates the CLEAN baseline, not the family.

**PRE-REGISTERED PREDICTION for the next `leaf_changed=1` row — banked
2026-08-27 09:43:38 UTC (commit `ea6b188`). The clause originally written here,
"before that row exists", is FALSE — see "Timing correction" below.** Writing it
down now is the whole point:
the argument below is available *before* the measurement, so if it is simply
recited afterwards the reading is a confirmation, not a finding. Two facts
already in hand constrain the answer:

| family | leaf_changed | l0miss | rebuilt | /miss |
|---|---|---|---|---|
| #1318, #1319 | 0 | 3 | 8 | 2.7 |
| #1183, #1307, #1317 | 1 | 3 | 19-22 | 6.3-7.3 |

`l0miss` is **3 in BOTH families**, so the ~12-rebuild gap cannot be coming from
episode arrivals into clusters — that input is constant across the split. And
#1319 gives the clean-row calibration directly: `rebuilt_level0 = 3 = l0miss`,
exactly. Structurally this is what the tree shape predicts, since the digest
leaf set is level-0 nodes plus leftover *episodes*, and which leftovers get
sampled does not touch cluster membership.

- **Prediction:** `rebuilt_level0 ~= 3` (tracking `l0miss`, unchanged across
  families), `rollup ~= 15-18`, `root = 1` => reading **(a)**, benign digest
  cascade, family CLOSED.
- **Falsifier, and it is sharp:** `rebuilt_level0 ~= 17` means the 2026-07-05
  oldest-anchored windowing confinement is leaking => reading **(b)**, and the
  windowing analysis reopens with that row as its starting evidence.

A middle result (`level0` in 6-12) falsifies neither cleanly and should be
recorded as such rather than rounded to the nearer branch.

#### Timing correction — the target row already existed. Read 2026-08-27.

`dream_runs.started_at` is `CURRENT_TIMESTAMP` (UTC). **Row #1324, the first
`leaf_changed=1` row after #1319, ran 08:26:38 UTC — 77 minutes BEFORE the
prediction was committed at 09:43:38 UTC.** The sentence above claimed the
prediction predated the row. It did not.

What actually held is weaker, and the difference matters: the prediction was
written *without sight of* #1324 (this document's own knowledge stops at #1319
and never mentions 1320-1324, which is the corroborating trace). That is a
**blind** read, not a **pre-dated** one — and blindness is not third-party
verifiable, where a timestamp ordering is. So #1324 is recorded at the weaker
status it earns, and it does not discharge the pre-registration.

**The pre-dated test transfers forward:** it is now the first `leaf_changed=1`
row stamped after 2026-08-27 09:43:38 UTC, i.e. strictly after #1324. The
prediction's text is unchanged and is not re-tuned against #1324's values —
re-fitting it now is exactly the move the A5 verdict refuses.

#### Finding — #1324 SPLITS the prediction. Unclassed, and informative anyway.

`level0 = 3` (`l0miss = 3` — exact), `rollup = 8`, `root = 1`, `rebuilt = 12`
(3+8+1 = 12, the v33 self-check holds on its second family row), `residual = 0`,
reuse 88.3%.

- The **structurally argued** component landed exactly: `rebuilt_level0 = 3 =
  l0miss`, the same identity #1319 gave on the clean side. That component was
  the one derived from tree shape rather than from the observed gap.
- The **magnitude** component missed: `rollup = 8` against a predicted 15-18.
- **The sharp falsifier did not fire.** `level0 ~= 17` would have meant the
  2026-07-05 oldest-anchored windowing confinement is leaking; `level0 = 3`
  says it is not, for this row.
- The middle clause (`level0` in 6-12) does not apply either.

Per the rule banked above, this is **unclassed, not rounded to the nearer
branch**. Reading (b) is dead *for this row*; reading (a) is not established,
because (a)'s predicted cascade size did not reproduce.

**What it does establish: `leaf_changed=1` is not a monolith.** At *constant*
`l0miss = 3`, `rebuilt` now forms a monotone ladder — 8 (#1318/#1319, clean) ->
12 (#1324) -> 19-22 (#1183/#1307/#1317) — and **every bit of that variance sits
in the `rollup` term**, which is precisely what the v33 split was built to
expose. Family 1's rollup of 15-18 did not reproduce in family 2.

The flag is binary; the cascade it is standing in for is clearly continuous.
That reframes the open question from "why do leaf-changed rows rebuild more?"
to "what sets the *size* of the leaf-set shift?" — and the honest answer is that
no instrument currently measures that size. Candidate next observation channel,
NOT built here and NOT pre-judged: a count of the leaf-set delta itself,
derived from the build's own rows the way v33's split was, so the continuous
driver is measured instead of inferred from a flag.

#### v34 BUILT 2026-08-27 — leaf-set delta (`aggregation_leaf_added/_removed`)

The channel named above as a candidate is now built, on the same terms v33 was:
derived from a set already in memory against the watermark the store already
keeps, no extra pass, no new query, **no threshold and no gate criterion**. It
does not answer the family question in advance and must not be read as having
pre-judged it.

Migration 034 supersedes an explicit decision in 030, which stored a
fingerprint rather than the id list because "the comparison only ever tests
equality". That was true when written. #1324 made it false — two rows carrying
the same flag value differ by ~8 rebuilds, so the flag abbreviates a continuous
quantity.

**Self-checking on two independent identities**, which is what makes it an
instrument rather than a number:

1. `leaf_changed = 1` **iff** `added + removed > 0`. The v29 flag and the v34
   counts reach the same comparison by different routes — hash equality vs set
   difference — so a disagreement means one is broken. Logged as
   `aggregate.leafdelta_disagreement`, not raised: an instrument that aborts
   the dream it is measuring costs more than the reading is worth.
2. `added - removed = n_leaves - previous n_leaves`, using the count v30
   already persists.

**NULL is unattributed**, as in v29/v31/v33 — a pre-v34 watermark row has no
predecessor id list, and an empty set would make every leaf look newly added
and manufacture a large delta out of a store that never moved. **The first
dream after this deploys therefore reports NULL and the second reports
numbers.** A NULL on the first post-deploy row is the contract working, not a
failure.

12 tests. Three of them exist because the first drafts were wrong in ways that
would have passed:

- the end-to-end persistence test originally seeded *clustered pairs*, which
  leave nothing over, so the leaf set was EMPTY and identity (1) reduced to
  `0 == int(0 > 0)` — satisfied by an implementation returning zeros
  unconditionally. It now seeds singletons and asserts a **swap** (1 added, 1
  removed), so the identity is exercised at 1 rather than satisfied at 0.
- `test_a_swap_is_the_case_n_leaves_cannot_see` pins the case a cheaper
  count-delta channel reports as 0.
- `test_the_disagreement_warning_can_actually_fire` forges a violation
  (matching id list, corrupted fingerprint) to prove the self-check is
  reachable — the E3 lesson that an unreachable guard reads clean regardless.

Both mutants confirm the tests bite: always-zero fails 3, and returning
`frozenset()` instead of NULL fails 3 including both counterfeit guards.

#### PRE-REGISTERED PREDICTION `G-LD1` — banked in the SAME COMMIT that builds v34

**This one is genuinely pre-dated and the ordering is checkable**: v34 reports
NULL on its first post-deploy dream, so no row this prediction can be scored
against exists anywhere yet — not on the box, not in principle. That is the
property #1324 turned out to lack, and it is worth having explicitly rather
than by luck.

The ladder to be explained, at constant `l0miss = 3`:

| rows | leaf_changed | rollup |
|---|---|---|
| #1318, #1319 | 0 | 4 |
| #1324 | 1 | 8 |
| #1183, #1307, #1317 | 1 | 15-18 |

**Prediction:** the rollup term is driven by **how many** leaves moved, so
`rollup - 4` rises monotonically with `added + removed`. Concretely, over the
next leaf-changed rows: a row with `rollup ~ 8` carries a SMALL delta
(`added + removed <= 3`), and a row with `rollup >= 15` carries a LARGE one
(`added + removed >= 5`).

**Falsifier, and it is sharp in both directions:** a row with `rollup >= 15` at
`added + removed <= 2`, or a row with `rollup ~ 8` at `added + removed >= 8`.
Either says the *count* of moved leaves is not the driver — and the live
alternative is then **which** leaves moved (their position in the tree, so one
leaf under a heavy subtree re-keys more than four under light ones), which is a
different mechanism needing its own instrument rather than a bigger sample.

A result inside neither branch is recorded as unclassed, exactly as #1324 was,
and NOT rounded to the nearer one.

### The flip gate vs the regression monitor — SEPARATED 2026-08-27

**The post-flip window is structurally unpassable as anchored, and that is a
signal problem, not a reason to touch #1317.**

Criterion 1 is "*every* verdict row >= 90%". #1317 sits in the window at 80.6%
and was correctly NOT re-anchored away. The consequence is arithmetic: this
watch returns FAIL on every future read regardless of what new rows do. #1318
and #1319 came back textbook-clean (92.2%, 2.7/miss, residual 0) and the
classifier still says FAIL; two more clean rows will not change it. That is the
confident-constant failure mode in a new dress — an instrument whose output no
longer varies with the thing it is measuring (see the diagnostic-controls
lesson: a gate that cannot register the difference between "fine" and
"regressed" is not reading anything).

The cause is role reuse, not a defect. The seven criteria were designed to
authorize a **one-time decision** ("is it safe to flip?"). They are now being
run verbatim as an **ongoing regression monitor**, which is a different job with
a different natural criterion. The resolution is to name the two roles apart:

- **G-FLIP, the flip gate: CLOSED, PASSED 2026-08-26, historical.** It did its
  job on the 11:15 window. It is not re-run, not re-anchored, and not re-scored.
  Its verdict does not change if later rows are worse — that is what makes a
  later bad row a *finding* rather than a retroactive re-litigation.
- **G-MON, the regression monitor: pre-registered here, 2026-08-27, distinct
  from G-FLIP.** Criterion: **no NEW below-bar append row** — i.e. every
  `append`-classified verdict row stamped after the flip commit (52adfe5) is
  >= 90% reuse with `residual = 0`. Rows predating the flip stay on the record
  and are excluded from G-MON's *scoring* while remaining visible in the
  listing. One below-bar append after the flip is a finding to act on; a
  `leaf_changed=1` row below the bar is routed to the family question above
  (which now has its own instrument) rather than counted as a regression, since
  that population was already known to be low-reuse *before* the flip.

#1317 stays exactly where it is under either framing — it is pre-flip evidence
for a known open question, and it is not an append. What changes is that G-MON
can register a difference again, which G-FLIP-as-monitor no longer can.

**Deliberately NOT done:** #1317 is not dropped, the window is not re-anchored,
and G-FLIP's bar is not moved. Re-anchoring to clear a failing row after seeing
it fail is post-hoc tightening's mirror image and is refused on the same grounds
the A5 verdict refused two waivers.

#### G-MON, first read 2026-08-27: the criterion FIRED on #1320.

| row | class | leaf | reuse | l0miss | level0/rollup/root | residual | /miss |
|---|---|---|---|---|---|---|---|
| #1319 | append | 0 | 92.2% | 3 | 3 / 4 / 1 | 0 | 2.7 |
| **#1320** | **append** | **0** | **87.5%** | **5** | **5 / 7 / 1** | **0** | **2.6** |
| #1321-#1323 | append | 0 | clean | - | - | 0 | - |
| #1324 | append | 1 | 88.3% | 3 | 3 / 8 / 1 | 0 | 4.0 |

#1324 routes to the family instrument per G-MON's own text and is **not** a
regression reading. #1320 is a below-bar `append` row stamped after the flip
commit, which is exactly the condition G-MON pre-registered as "a finding to act
on". **The finding fired. It is recorded as fired.**

It is also birth-dated: #1320 ran ~24h before G-MON was written, so the
criterion applies to it retroactively. That is not grounds to exclude it —
excluding a row because the criterion was written after it is the same move as
re-anchoring a window to clear #1317, and it is refused for the same reason.

**Acting on it — diagnosis: #1320 is the most rebuild-efficient row on record,
and it fails anyway.** Per-miss amplification is **2.6**, below #1318/#1319's
clean 2.7 and far below the family's 6.3-7.3. Nothing is amplifying. The reuse
deficit is proportional to `l0miss = 5` — more leaves genuinely changed.

**And that exposes a defect in the criterion itself, by arithmetic rather than
by hindsight.** `built` is ~103 on all three measured rows (8/92.2%, 13/87.5%,
12/88.3% all invert to 103 within rounding), so at this tree size reuse% is a
linear restatement of `rebuilt`, and the 90% bar is exactly `rebuilt <= 10`.
Since `rebuilt = amplification x l0miss`, at the clean amplification of ~2.7 the
bar is breached by `l0miss >= 4` on its own:

> **G-MON's 90% reuse bar is arithmetically equivalent to "no more than three
> changed leaves per dream" — a cap on how much the STORE changed, not on how
> efficiently the tree rebuilt.** A busier day fails it with a perfectly healthy
> cascade.

This derivation needs no knowledge of which direction #1320 went; it follows
from the definition of reuse% and was available when G-MON was written. It was
missed. Recorded as an amendment prompted by a failing row, because that is what
it is.

**Correction, forward-only — G-MON-b, pre-registered 2026-08-27, first scored on
rows stamped after this entry.** Criterion: per-miss amplification
`rebuilt / l0miss <= 4.5` on `append` rows with `residual = 0`, which is
`l0miss`-invariant and therefore measures the thing G-MON was built to catch.
The 4.5 bar sits above the clean band (2.6-2.7) and #1324 (4.0) and below the
family (6.3-7.3) — it is set from the *pre-existing* calibration, not from
#1320.

**Deliberately NOT done:** G-MON's raw 90% bar is **not** retracted, not
loosened, and not re-scored. #1320 stays a fired finding. G-MON-b runs
*alongside* it, so the substitution is visible in the record rather than
silently applied — a criterion swapped in after a failure has to be readable as
a swap, or the next reader cannot tell a fix from a fudge. If the two disagree
on a future row, that disagreement is itself the datum.

---

## Stage 4 — Query-time consumption v2 (only after Stage 3)

**Stage 3 RESOLVED 2026-08-26 (G-FLIP PASS -> flipped), so Stage 4 is open.**
4a BUILT 2026-08-27 (inert default); 4b BUILT 2026-06-12. Stage 4 is complete
as specced — what remains is a *flip* decision for 4a's threshold, which is a
separate pre-registered scored decision and is NOT taken here.

**4a. Sparse-signal fallback injection.** Fire the node tier when raw retrieval is THIN
(e.g. `len(message_hits) + len(fts_hits) < threshold`, or top BM25 score below floor)
instead of / in addition to ability gating. Principled: nodes appear exactly when there
is nothing for them to crowd, covering vague/global/cold-start queries ("what do we know
about my projects?") that LME never asks. Config: `aggregation_fallback_min_hits`
(0 disables). Tests offline; no LME run (invariant 1).

**4a BUILT 2026-08-27 — ships INERT (`aggregation_fallback_min_hits = 0`).**
The spec's own gating note ("do not land a default change before the 3c flip
resolves") is DISCHARGED: 3c resolved PASS and flipped on 2026-08-26, and 4a
lands with a default of 0 anyway, so no default behaviour changes on any store.
Built exactly to the spec below — `_raw_signal_count` / `_sparse_signal_fires`
(`augment.py`), strict OR at the gate, `sparse_fallback(raw=N)` chip applied
only when `by_fallback and not by_ability`, the episodes-excluding variant of
thinness. All 8 matrix rows plus 3 unit rows on the predicates: **11 tests,
suite 1287 passed / 8 failed**, the 8 being the pre-existing local
`sqlite_vec`-missing set, byte-identical to the baseline measured by `git stash`
in the same session (1276 passed before). No LME run, invariant 1 intact.

Two spec points that survived contact with the code and are worth reading as
banked, because both are the kind of thing that silently rots:

- **Attribution, not just firing.** Matrix row 4 was extended with a row 4b —
  a TR query on a *starved* store, where BOTH conditions are true at once.
  `by_ability` wins the attribution there, so a cold-start TR query cannot be
  miscounted as fallback evidence when the later A/B is read. Row 4 alone
  would have passed with the chip smearing.
- **Rows 1 and 5 are the landing licence.** Row 1 pins the inert default
  (firing set identical to TR-only), row 5 pins that the master switch
  dominates at any fallback value. Together they are why this could land during
  the post-flip monitor window without perturbing it.

*Original spec, retained verbatim as the build record.* Code sites confirmed against HEAD: the gate to
amend is `augment.py:584`
(`if cfg.aggregation_nodes_enabled and _aggregation_tier_fires(cfg, ability):`),
the ability predicate is `_aggregation_tier_fires` (`augment.py:2209-2218`), and
the knob belongs beside `aggregation_inject_abilities` (`config.py:208`). The
tier ordering already works: `ctx.fts_hits` (set at `augment.py:441-466`),
`ctx.message_hits` (:531) and `ctx.episodes` (:560) are all populated before the
aggregation gate at :584, so the thinness test reads live counts with no
reordering.

*Shape.* New `aggregation_fallback_min_hits: int = 0` (0 disables — 4a ships
inert). Gate becomes
`if cfg.aggregation_nodes_enabled and (_aggregation_tier_fires(cfg, ability) or _sparse_signal_fires(cfg, ctx))`.

Three properties, ordered by how easy they are to get wrong:

1. **Strict OR, never a relaxation.** The G4 A/B is what pinned
   `aggregation_inject_abilities` to TR-only: broad injection reshuffles ranking
   against gold turns. 4a's whole licence is "nodes appear when there is nothing
   to crowd", so the fallback must be an INDEPENDENT condition that only ever
   adds firings on starved queries. It must never widen, soften or bypass the
   ability gate for a query that already has hits.
2. **Thinness is one named function.** `_raw_signal_count(ctx)`, so the
   definition lives in one place and is directly testable. Ship this section's
   spec (`len(message_hits) + len(fts_hits)`) as the default.
3. **Provenance on the firing mode.** A fallback firing tags its hits
   (`why_retrieved += ["sparse_fallback(raw=N)"]`). Ability-gated and
   fallback-gated firings have completely different expected effects; without
   the chip no later A/B can separate them. Cheap now, unrecoverable later.

> **Pre-registered open choice, flagged not decided:** whether `ctx.episodes`
> joins the thinness count. The readings point opposite ways — EXCLUDING
> episodes (this section's text) fires the tier when a dreamed store already has
> session summaries covering the query, i.e. when nodes are REDUNDANT rather
> than crowding; INCLUDING them means the fallback almost never fires on a
> mature store, i.e. 4a is inert in practice rather than merely default-off.
> Ship the excluding variant, record the including variant with its reason, and
> decide any switch on a stated argument. **Do not decide it by trying both and
> keeping the better number.**

*Reachability caveat — belongs in the code, not only here.* With
`aggregation_nodes_enabled=False` (production, and staying so until the flip)
the entire 4a path is UNREACHABLE in production. That is intended, but this
project has been burned by a guard reading PASS because its path was unreachable
from the config under test (see the E3 gates lesson). So: every 4a test must
construct a config with BOTH the master switch and the fallback ON — a test that
exercises 4a with the layer off tests nothing and passes regardless — and the
`aggregation_fallback_min_hits` docstring must say it is subordinate to
`aggregation_nodes_enabled` and inert until the 3c flip.

*Test matrix (offline, zero LLM, invariant 1 respected).*

| # | config | expect |
|---|---|---|
| 1 | fallback=0 (default), layer ON | firing set identical to today's TR-only — the inert-default regression guard |
| 2 | layer ON, fallback=2, ability=None, 0 raw hits, nodes exist | fires; hits carry the `sparse_fallback` chip |
| 3 | layer ON, fallback=2, ability=None, 3 raw hits | does NOT fire |
| 4 | layer ON, fallback=2, ability="TR", 10 raw hits | fires via the ability gate; chip attributes to ability, not fallback |
| 5 | **layer OFF**, fallback=5, 0 raw hits | does NOT fire — the master switch dominates |
| 6 | fallback firing vs same query fallback off | `message_hits` / `fts_hits` / `episodes` BYTE-IDENTICAL |
| 7 | raw hits == min_hits exactly | does not fire (strict `<`), documented |
| 8 | fallback fires, zero nodes built | returns `[]`, no error |

Test 6 is what earns the feature: it turns "nodes appear when there is nothing
to crowd" from a claim into a mechanical assertion. Tests 1 and 5 together are
what let 4a land DURING the flip watch without touching it.

*Cost.* The fallback runs `_aggregation_search` (FTS + a vec scan bounded by
`embedding_max_scan`) on starved queries. No LLM call. Assert the call count,
not wall-clock.

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
