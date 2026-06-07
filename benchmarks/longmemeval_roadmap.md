# LongMemEval / BEAM Retrieval Roadmap

Single source of truth for HyMem retrieval-quality work against LongMemEval (LME)
and BEAM. **Read this before proposing a change — most "obvious" ideas have already
been tried; the dead-ends section says which and why.** Supersedes the original
`~/.claude/plans/serene-fluttering-sparrow.md` (that was only the first ability-shaping
phase). Running narrative log lives in memory `project_beam_retrieval.md`; this file
is the scannable ledger.

**Overriding constraint (never relax):** a change only counts if it carries to
real-world HyMem / Hermes, not just the LME score. Anything that reads the oracle
`question_type` label (which production does not have) is benchmark-gaming and is
rejected on sight.

**Standing process:** do NOT commit or push — the user commits themselves. All work
lives on branch `Beam-optimisation`, uncommitted.

---

## 1. Current baseline (the config to A/B against)

**Shipping config = validated defaults, no flags except the run controls:**

| Knob | Value | Notes |
|---|---|---|
| ordering | **message-first** (default) | raw turns lead; graph_facts demoted to a confidence-ranked tail. `--graph-facts-first` is the A/B opt-out only. |
| `rerank_message_hits` | `True` | LLM-rerank both the chunk and raw-message tiers |
| `rerank_model` | `"llm"` | reuses the host deepseek client; cross-encoder path exists but unused |
| `mr_aggregate_additive` | `True` | MR layers a count on top of relevance retrieval (never replaces it) |
| `message_fts_aggregate_cap` | `50` | exact count regardless of cap |
| dream | **ON** | vindicated — net-neutral on LME overall, kept for the cross-session KG (product differentiator LME can't see) + the TR edge |
| embedding_client | **NONE** by default; `--embeddings` wires it | semantic vec recall is score-neutral on LME (65.0→65.0, see L1) — wired for production faithfulness, not a score lever; off in the headline run |
| adapter top_k overrides | `message_fts_top_k=15, fts_top_k=10, graph_top_k=10` | set in `HyMemAdapter.open()` |
| reference-now | `question_date` prepended | `--now` source logged per run |

**Run the baseline:**
```bash
# Oracle (question_type drives shaping) — the headline number
python benchmarks/longmemeval_adapter.py --sample 0 --seed 0 --workers 8
# Production-truth (real detect_ability router drives shaping)
python benchmarks/longmemeval_adapter.py --sample 0 --seed 0 --workers 4 --auto-ability
```
`--auto-ability` OOMs at `--workers 8` (dual 50-session haystacks → ~16 concurrent
dream cycles); use `--workers 4`.

**Authoritative full-500 seed-0 matrix (default-flip live):**

| Config | Overall | KU | MS | SS-A | SS-P | SS-U | TR |
|---|---|---|---|---|---|---|---|
| Oracle, no-dream | 65.0 | 61.5 | 48.1 | 60.7 | 70.0 | 91.4 | 70.7 |
| **Oracle, full-dream, msg-first (BASELINE)** | **65.0** | 60.3 | 46.6 | 62.5 | 73.3 | 88.6 | 72.9 |
| Oracle, full-dream, graph-facts-first | 63.6 | 61.5 | 50.4† | 57.1 | 66.7 | 78.6 | 72.2 |
| Auto-ability, no-dream | 58.6 | 56.4 | 39.1 | 60.7 | 13.3 | 92.9 | 70.7 |
| **Auto-ability, full-dream (PRODUCTION-TRUTH)** | **60.2** | 58.4 | 42.5 | 63.4 | 11.7 | 93.6 | 71.1 |

† extraction non-determinism, not mechanism — the `--graph-facts-first` flag does
**not** touch the MR/MS path (MS→MR ∈ TASK_RECALL → message-first in both columns).

**How to read the two baselines:**
- **65.0% (oracle)** is the headline / category-shaping ceiling.
- **60.2% (auto-ability)** is the honest production number. The −4.8pp gap is **~77%
  the SS-P harness artifact** (30 Q × 61.6pp / 500 = 3.7pp). Net SS-P out and the
  production router matches oracle within noise. Do not quote −4.8pp as a retrieval
  regression — it isn't one (see dead-end D4).
- Published SOTA for reference: Hindsight 89.4 / **HyMem 65.0** / Honcho 63.8 /
  LIGHT 51.7 / RAG 48.5. The gap to Hindsight is concentrated in **MS** (multi-session).

---

## 2. DONE — landed changes & tests (the "don't redo this" ledger)

All landed on `Beam-optimisation`. Full suite green at each step (~470 tests).

### Retrieval infrastructure
- **`messages_fts` raw-turn keyword tier (schema v13).** FTS5 over `messages.content`,
  porter/unicode61, user/assistant only, live triggers at ingest. Surfaced as
  `AugmentedContext.message_hits`, knob `message_fts_top_k`. **The single biggest
  historical win** (overall 15.2→38.0 on the early BEAM run). Raw turns are searchable
  before any dream consolidates them.
- **`created_at` end-to-end (was a live data-loss bug).** `append_message` dropped the
  timestamp → every row got ingestion-time, making all chronological ordering wrong and
  TR structurally unanswerable. Threaded through `log_message(s)`, `app.py`,
  conditional INSERT. Test: `test_add_messages_persists_supplied_created_at`.
- **`temporal_mentions` table + date extraction (schema v14).** `dreaming/dates.py`
  (stdlib EN+NL months, ISO/numeric, >12-rule), `dreaming/temporal.py` (per-message
  indexing). `augment(ability="TR")` → `temporal_events: list[TemporalEvent]`.
- **Session-date TR anchors (always-on, additive).** `_temporal_hits_events` sources
  anchors from already-retrieved `message_hits`/`fts_hits` (carries `created_at`),
  appended beyond `top_k` so they never evict content-dates. Tests:
  `test_tr_session_anchors_coexist_without_evicting_content_dates`,
  `test_tr_chunk_hit_anchors_via_retrieved_chunk`.
- **Message-tier reranking (alley #2).** `message_hits` now pulls a `rerank_top_k`-wide
  pool and reranks down to `message_fts_top_k` (was raw BM25). Flag `rerank_message_hits`
  (default True). Targets the ranking-dominant losses (esp. MS). Tests:
  `test_message_tier_reranks_wider_pool_then_trims`, `..._disabled_keeps_raw_bm25`.

### Ability routing (the production carry-over layer)
- **`detect_ability(query) → MR|TR|None` (`query/intent.py`).** stdlib regex, EN+NL.
  The anti-gaming guardrail: MR/TR shaping fires in production WITHOUT the oracle label.
  Explicit host `ability` always wins; detection only fills None.
- **Router TR broadening (rounds 1 & 2).** Rebuilt `_TR_ORDER` (WH-opener + intervening
  noun + adverbial ordinal, determiner-guarded for EN+NL), new `_TR_DISTANCE` ("N units
  ago", "last week"), `_TR_RECENCY` ("when did I last/first…"), `_TR_DURATION` perfect-aux.
  **TR router recall 38% → 92%, precision 86%.** +80-odd cases in `test_intent.py`.
- **`benchmarks/router_eval.py`.** Zero-LLM sweep of `detect_ability` vs oracle over the
  full dataset (seconds on the box) + per-intent residual-miss listing. Tight tuning loop
  decoupled from the expensive retrieval+judge run.

### MR / counting
- **`message_fts_aggregate_cap` lexical-count path.** user-only filter, EN+NL stopword
  drop, restatement dedup. +45pp on lexical-count MR; semantic-count MR degrades
  gracefully (host LLM verifies). Default 50.
- **Additive-MR (`mr_aggregate_additive`, default True).** MR now LAYERS a count on top
  of (reranked) relevance retrieval instead of REPLACING it. Makes a false-positive MR
  route near-harmless — the root fix for the 62 NONE→MR router FPs (which are textually
  identical to real MR and thus un-fixable by regex). Tests:
  `test_mr_additive_layers_count_on_relevance_retrieval`,
  `test_mr_false_positive_keeps_relevance_retrieval`.
- **In-domain graph-native COUNT (`count_relations` / `plan_count`).** Exact
  `COUNT(DISTINCT subject|object)` over the KG, mirror-orientation retry, abstains on
  zero-evidence. Fires only on the tech vocabulary → helps real-Hermes in-domain use,
  **None on all LME** (consumer domain) by design.

### Ordering / dream
- **Message-first default ordering (the carry-over fix).** Inverted the default so
  raw turns lead and graph_facts demote to a confidence-ranked tail for EVERY ability.
  Closed the −14.3pp SS-user dream regression (graph_facts were crowding the answer turn
  at the top of the context). `--graph-facts-first` retained as A/B opt-out only.
- **Diagnosed dream as net-neutral, not harmful.** The −2.2pp was 100% an ordering
  artifact; message-first ties no-dream at 65.0 while keeping the TR edge and the
  cross-session KG. Dream stays ON.

### Adapter / harness & diagnostics
- **`--workers N` (ThreadPoolExecutor over per-question temp DBs).** Byte-identical
  results, lock-guarded counters. Full-500 2h13m → 14m with `--no-dream --workers 8`.
- **`--no-dream`, `--seed` (default 0, paired runs), `--sample 0` (full 500).**
- **`question_date` reference-now.** Prepends "Today's date is …"; fixed "how many days
  ago" TR misses. Source logged (`question_date|haystack_max|none`).
- **Recall-ceiling diagnostic.** Splits every miss into retrieval-loss vs ranking-loss
  via gold-turn-in-pool check (captured before the top_k cut, tier-attributed).
  Verdict banked: **190 ranking misses vs 37 retrieval — ranking is the bottleneck.**
- **Router shadow / `--auto-ability` diagnostic.** Records oracle vs detected ability on
  every run; `--auto-ability` makes the inferred label drive shaping (production score).
- **Immutable per-run archive** `…-{UTCstamp}-seed{N}.json` (stops runs clobbering each
  other).

---

## 3. Validated dead-ends — DO NOT re-chase

- **D1. Graph-native COUNT for LME MR.** KG predicates/types are tech-locked (18
  predicates, 24 types); LME MR is consumer-domain → structurally inexpressible.
  graph_count is None on all 500 by design. (In-domain Hermes counting IS built and
  works — different use case.)
- **D2. LLM "count across long context" for semantic MR.** Empirically false for
  deepseek — it can't tally. Lexical-count helps; semantic MR ("how many *ways*") is a
  reader problem, not retrieval. MR keyword ceiling is real.
- **D3. Category-aware ordering (message-first for IE/PF, graph-facts-first for MS).**
  Built on a phantom: MS→MR ∈ TASK_RECALL, so the ordering flag never touches MS — the
  apparent MS delta is dream non-determinism. No category is proven to want graph-facts-
  first. Don't add category routing keyed on the oracle label (violates the constraint).
- **D4. "Fix" the SS-P auto-ability crater.** It's a HARNESS artifact, not retrieval:
  oracle gives SS-P the permissive `ANSWERING_PREFERENCE_PROMPT`; the router can't emit
  PF so production falls to the strict default prompt → the model refuses on
  recommendation questions. Ceiling unchanged (gold retrieved). The adapter's
  `answer_question` prompt-pick is code Hermes never runs. **Rejected fixes:**
  question_type→ability fallback (reads the oracle label — pure gaming); a 4th PF router
  class (buys real Hermes nothing — PF touches only the benchmark prompt). The only
  honest option is a permissive default prompt, but it endangers the `*_abs` abstention
  slice — run it only with `*_abs` broken out so the trade is visible.
- **D5. Disabling dream to "win" on LME.** LME is single-conversation-haystack; the
  cross-session KG dream builds is invisible to it. "No-dream wins" is a statement about
  LME's blind spot, not HyMem — would gut the product differentiator.
- **D6. SS-preference ceiling (67%) as a retrieval lever.** Judge/ceiling-bound, not
  retrieval. Settled.
- **D7. TR FP regex-guarding.** Additive shaping already makes TR FPs near-harmless and
  TR is strong (92% recall); guarding risks the recall.
- **D8. Q45-class entity-precision misses** ("model grabs niece vs cousin"). Reader
  precision, not a retrieval/temporal gap. Low ROI at current n.

---

## 4. Open levers — prioritized (all carry-over-clean)

The recall-ceiling verdict frames everything: **ranking & cross-session synthesis is
the bottleneck, not recall.** MS is the prize — 27% of the set, lowest score (~46.6%),
ceiling ~96%, each 10pp ≈ +2.66pp overall, and exactly where Hindsight leads.

- **L1. Wire a real embedding client — DONE 2026-06-07, score-neutral but diagnostically
  decisive.** Added `--embeddings` flag (`HyMemAdapter.open()` builds
  `OpenAICompatibleEmbeddingClient()`, env-driven so it drives the SAME local FastEmbed
  ONNX server Hermes uses: `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, `:8766`).
  **Result: overall 65.0 → 65.0 (0.0pp). Embeddings do NOT move the LME score.** Per-cat
  within noise (KU +3.8, SS-A +1.8, TR −1.5, SS-P −6.6, MS/SS-U flat). Recovered-by tiers:
  103 both (FTS+vec fusion), 355 raw-message, 5 FTS-only, **no vec-only bucket** → vec
  recall is *redundant* with FTS on LME; it adds candidates, not unique recall. **The win
  is the diagnostic, not the score:** with embeddings ON, retrieval loss is near-zero
  everywhere (MS 6 retrieval vs 65 ranking, SS-U 0 vs 8, TR 13 vs 25) — so the
  "ranking/synthesis is the bottleneck, not recall" thesis is now PROVEN, not hypothesized.
  Lever closed. Embeddings stay wired (production-faithful) but are not the score lever.
  → **redirects all remaining effort to ranking (L2).**
- **L2. Reranker — the top lever (ranking, not recall). FRONT-RUN PROBE LANDED 2026-06-07.**
  Mechanism (verified [augment.py:335-339](../hymem/query/augment.py#L335-L339),
  [:456-480](../hymem/query/augment.py#L456-L480)): the *message* tier (dominant, 355
  hits) reranker is ALREADY firing — NOT ambiguity-gated, runs whenever pool > cut. Only
  the *chunk* tier is gated by `should_rerank`, which now mostly SKIPS (FTS≈vec agree) and
  recovers ~2 turns = noise. The message tier pulls `max(message_fts_top_k=15,
  rerank_top_k=20)=20` BM25 candidates and reranks to 15.
  - **PROBE RESULT (`gold_rank_probe.py --category multi-session --sample 0 --seed 0`,
    n=133 MS):** median gold BM25 rank = **2**. Distribution: **≤15: 122 (92%)**, 16-20: 4
    (3%), 21-40: 2, 41-60: 1, 61+: 4, **NOT-in-BM25: 0**. The reranker already SEES 126/133
    gold as candidates at the default budget. This **kills the L2a hypothesis**: widening
    20→40 newly reaches just 2 turns, →60 one more — noise. The 65 MS ranking misses are
    NOT "gold below the rerank window"; they are **gold seen-but-demoted** — the LLM
    reranker is pushing gold OUT of its own top-15. The lever is the reranker's *judgment*,
    not its *budget*.
  - **L2a — widen budget (`--rerank-top-k 40/60`). KILLED by the probe.** 3 turns / 133 is
    not worth the compute. Flag stays for completeness; do not run it for MS.
  - **L2c — message-tier reranker OFF (`--no-rerank-message-hits`, raw BM25 top-15). MEASURED
    2026-06-07 (no-dream, seed 0, w8) — RERANKER IS NET-POSITIVE; L2 CHAPTER CLOSED.**
    Result vs baseline (rerank ON): **overall 65.0→64.2 (−0.8), MS 48.1→43.6 (−4.5pp),
    SS-U flat, TR −0.8. MS ranking misses 60→64 (+4), retrieval misses 9→11 (+2).** Turning
    the reranker OFF makes MS WORSE → the "reranker demotes gold" hypothesis is **dead**; the
    reranker LIFTS semantically-relevant turns raw BM25 leaves below the cut. **Keep the
    message-tier reranker on (shipped default). L2b cross-encoder is moot** (no incumbent
    harm to fix). **CONFOUND (sharpens the claim):** `--no-rerank-message-hits` also narrows
    the candidate pool 20→15 ([augment.py:326-330](../hymem/query/augment.py#L326-L330)),
    so L2c removed BOTH the wider window AND reordering. The **+2 retrieval misses are the
    fingerprint of the narrowing** — exactly the 4 gold turns the probe found at BM25 rank
    16-20 (between the 15-cut and the 20-window), which raw-BM25@15 never pulls. So the
    honest claim is "the message-tier reranker AS CONFIGURED (20-window + LLM reorder) is
    net-positive for MS," NOT "the LLM's ranking judgment saves MS." A clean reorder-only
    test (`rerank_top_k=15` ON vs OFF) is NOT worth running — the shipped config is the
    decision. (Side note: KU −4 ranking misses without the reranker — within noise but a
    consistent hint the reranker may slightly *harm* KU; category-specific, parked.)
    - **Attribution footnote when reading the L2c delta:** `--no-rerank-message-hits` turns
      off ONLY the message tier; the *chunk* tier reranker ([augment.py:296](../hymem/query/augment.py#L296),
      gated by `should_rerank`, not by this flag) still fires in BOTH the baseline and the
      L2c column. Its *expected* contribution therefore CANCELS in the paired delta (no
      bias) — but the LLM reranker is stochastic and the chunk tier swings ~2 turns, so that
      VARIANCE lands in the delta as noise. **Rule: treat any MS delta within ~±2-3 turns as
      noise floor, not message-tier signal.** (`should_rerank` mostly skips on LME — FTS≈vec
      agree on the top hit — so the chunk tier rarely fires, shrinking the floor; but it can.)
  - **L2b — cross-encoder (`--rerank-model cross-encoder`). MOOT after L2c.** L2c showed the
    LLM reranker is net-POSITIVE (it doesn't demote gold), so there's no incumbent harm for a
    stronger model to fix; a cross-encoder might still squeeze out the residual but it's not
    the lever — the residual MS misses are downstream (→ L3). Flag stays wired; revisit only
    if L3 proves the misses are reranker-judgment-bound (they aren't, per the L3 probe below).
    (When/if used: English-only mxbai-base — production multilingual/Dutch needs `bge-reranker-v2-m3`.)
  - Adapter flags wired: `--rerank-top-k`, `--rerank-model`, `--rerank-message-hits /
    --no-rerank-message-hits` (all persisted in result `config` for `compare_recall.py`).
  - **Reads the sweep:** `compare_recall.py base.json msgRRoff.json [crossenc.json]` diffs
    `recall_diagnostics`, per-category `miss_ranking` + a TOTAL row, Δ-vs-baseline (↓ = win).
  - **Probe caveat (documented in its docstring):** BM25 rank is message-FTS only = a LOWER
    BOUND on combined-pool rank (chunk+graph+MR fuse before the cut), so the gate is
    pessimistic by design — safe for a "run it" decision.
  - **Redundancy-closure (embeddings chapter — CLOSED for MS):** NOT-in-BM25 = **0** for MS:
    BM25 alone reaches 100% of MS gold at some rank → **zero vec-only recovery**, embeddings
    are fully redundant for MS (matches L1's no-vec-only-bucket). Caveat: the probe is
    message-FTS only, so a 0 here is conservative on the "keep embeddings" side (chunk
    embeddings could in principle recover a turn) — but L1's real embeddings-on run already
    showed no vec-only bucket, so the two agree: vec is droppable for LME on MS. (Other
    categories not yet probed; TR especially may differ — probe before generalizing.)
- **L3. MS coverage at the 15-slot message cut — the next lever (NOT the 45-slot assembly).**
  **Mechanism CORRECTED 2026-06-07 (the earlier "crowded out of the final context" story was
  wrong — verified against the code):** the answer context cut is `top_k*3 = 45`
  ([longmemeval_adapter.py:531](longmemeval_adapter.py#L531)) and the MR/TASK_RECALL assembly
  places `message_hits` FIRST ([:518-522](longmemeval_adapter.py#L518-L522)) with a total
  retrieved set ~≤45 — so message hits **always survive the 45-slot cut**; chunk/graph can
  never crowd them out there (only the low-confidence chunk/graph TAIL is ever trimmed). The
  real point of loss is the **15-slot `message_fts_top_k` cut inside augment()**: a multi-
  session answer needs N gold turns across N sessions, and one verbose session can monopolize
  the 15 message slots, squeezing out sibling-session gold.
  - **CRITICAL — the "ranking miss" label conflates two failures for MS.** The recall
    diagnostic sets `recall_ceiling=True` if *ANY* gold turn is in the pool (`_gold_in_pool`
    is an any-match) — correct for single-session questions (one gold turn = answer), but MS
    needs SEVERAL. A question needing 3 turns with 1 present + 2 squeezed out still scores
    ceiling=True → labeled "ranking miss" → but it's a **coverage loss**, not a ranking loss.
    So the ~64 MS ranking misses are a MIX of **coverage-short** (L3-fixable in HyMem) and
    **fully-covered-but-model-fails** (synthesis — answer-side, out of scope like SS-pref).
    Only the coverage-short slice is ours to fix.
  - **FRONT-RUN GATE before building anything (same discipline that killed L2a):**
    `gold_rank_probe.py --coverage --category multi-session --sample 0 --seed 0 --join-run
    <baseline.json>`. LLM-free; per MS question it measures `n_gold`, `n_in_cut` (gold turns
    inside the 15-slot window), coverage ratio, and the window's **session concentration**
    (distinct sessions + max-session-share — the monopoly signal). `--join-run` reads the
    baseline result JSON, isolates the category's ranking misses (correct=False &
    recall_ceiling=True) and **splits them coverage-short vs fully-covered in a single table
    row + verdict.** Decision: mostly coverage-short → L3 has teeth (fix below); mostly
    fully-covered → L3 won't help, the residual is synthesis (bank it, out of scope).
    - Probe caveat (documented): coverage is raw-BM25 **message-only** (the probe doesn't
      dream, so no chunks; and no rerank) → a LOWER bound on real coverage, so coverage-short
      is mildly over-counted (biases toward "L3 has teeth" — the extra-experiment direction).
  - **The fix, IF coverage-short dominates** (don't build until the gate says so): cheapest
    is widening `message_fts_top_k` (MS can afford to evict the chunk/graph tail under the
    45-cut); fancier is session-diversity-aware selection within the window (cap per-session
    slots so a verbose session can't monopolize). The probe's max-session-share tells you
    which: high share → diversity-pack; gold simply at rank >15 → just widen the cut.
  - **MEASURED 2026-06-07 — L3 IS DEAD in both forms (the gate + two free front-runs killed
    it before any feature build).** Coverage gate on 60 MS ranking misses: 36 coverage-short
    (60%), 24 fully-covered/synthesis (40%). But the coverage-short slice does NOT respond to
    either L3 fix:
    - **`--cut-sweep 20,25,30` (widen projection, free — `gold_ranks` is cut-independent):**
      cut=30 recovers only **8 of 36**; **15 are floor** (≥1 gold turn NOT in the message BM25
      pool at any depth); the other 13 sit at rank 31+. Recovery curve is shallow, floor is large
      → not a window-size problem.
    - **`--pack-sim 2,3,4 --pack-pool 60` (diversity-pack simulation, free — reorders a FIXED
      15-slot window over a deep pool, no answer-side crowding):** recovers **1 of 36** at every
      cap. The 21 non-floor misses are **intra-session** lexical-rank failures — the gold turn is
      deep in its OWN session's BM25 ranking, not squeezed out by a sibling session. Diversity-pack
      can't touch intra-session ranking (scenario C). Same for widening.
    - **Decomposition of the 60 MS ranking misses:** 15 floor (message-FTS recall gap → chunks/
      embeddings, the ONLY slice with a live lever) · 20 deep-lexical (intra-session BM25; reranker
      already mitigates, residual is a lexical ceiling) · 24 synthesis (answer-side, bank it) ·
      1 widen/pack-recoverable (noise). **L3 (message-window packing/widening) banked as dead.**
  - **NEXT LEVER — the floor audit (the 15).** The probe floor is message-only (no dream → no
    chunks), so the real post-chunk floor is unknown. Instrumented the adapter's `per_question`
    block with **per-gold-turn fused-pool membership** (`gold_turns_in_pool` + `gold_turn_tiers`,
    `"none"` = unrecovered turn) so a single baseline re-run + `gold_rank_probe.py --coverage
    --join-run <instrumented.json> --floor-audit` splits the 15 into **PHANTOM** (chunks/embeddings
    already rescue → floor disappears) vs **REAL** recall gap (chunks miss them too → narrow,
    targetable chunk/embedding-recall lever on exactly those qids). `recall_ceiling`'s any-match
    can't answer this; the per-turn record can.
- **L4. Permissive/abstention-aware default answer prompt (harness, optional, gated).**
  Only if a clean banked benchmark number is wanted — run WITH the `*_abs` slice broken
  out (see D4). Not load-bearing for the HyMem conclusion.

**Sequencing:** L1 done (recall ruled out). L2a KILLED by the gold-rank probe (92% MS gold
already at BM25 ≤15). L2c DONE → reranker is net-positive (−4.5pp MS when OFF), keep it, L2b
moot → **L2 chapter closed.** L3 DONE → **DEAD in both forms** (cut-sweep recovers ≤8/36,
pack-sim 1/36; the 21 non-floor MS misses are intra-session lexical-rank, untouchable by
window packing/widening). NOW: **floor audit** — instrument per-gold-turn fused-pool membership,
re-run baseline once, `--floor-audit` splits the 15 floor misses into phantom (chunks rescue)
vs real recall gap (the only remaining retrieval lever, narrowed to ~15 qids). Every step
data-gated by a free LLM-less probe before any feature build — the discipline that killed L2a,
L2b, and both halves of L3.

---

## 5. Methodology notes (so results stay comparable)

- **Always `--seed 0` (or `--sample 0` for full 500).** Pre-seed runs drew different
  stratified subsets — no run was paired. Per-category deltas at n≈17 are ±1-2 Q of noise.
- **Mechanism > score.** A category lift is only trusted when the mechanism is verified
  (e.g. temporal_events non-empty), because unseeded score swings are sample variance.
- **Two numbers, always.** Report oracle (ceiling) AND `--auto-ability` (production).
  A gain that only shows under the oracle label is fiction until the router reproduces it.
- **Run the cheap router sweep first.** `router_eval.py` confirms a routing change in
  seconds before spending the expensive retrieval+judge run.

---

## 6. Still pending (larger, durable — not LME-gated)

- Bi-temporal edges (Zep/Graphiti `valid_at`/`invalid_at`).
- RAPTOR-style aggregation nodes in dreaming (staleness via `digested_version`).
- Relative-date parsing ("twee weken geleden") — needs `dateparser`, deferred against
  the zero-dependency hardening goal.
- `messages_fts` not carried by export/import.
- Tokenizer `porter` (English) vs Dutch-first scope.
