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
- **Dutch FTS diacritic fix 2026-06-08 (carry-clean bug fix; query side only, no migration,
  no dep).** The FTS5 `unicode61` index FOLDS diacritics ("café"→"cafe"), but the query
  sanitizer `_FTS_SAFE` was an ASCII-only whitelist that SHREDDED accented query tokens before
  they reached FTS ("café"→"caf" = 0 hits; "coördinatie"→"co" OR "rdinatie" = 0 hits). So every
  accented Dutch/loanword query silently missed while the index held the folded form — a real
  retrieval bug, Dutch-prioritized. Fix: new `_fold_diacritics()` (NFKD + drop combining marks)
  applied at ALL SIX FTS query sites (chunks/messages/aggregate/temporal/episodes/procedures),
  so query and index agree. Recall is now accent-INSENSITIVE both directions. Covers the full
  Dutch diacritic set (ë ï ö é ü á è); precomposed non-decomposing Latin letters (ø ß æ — not
  Dutch) are a known out-of-scope residual. `entities.py` was already accent-safe (Unicode
  `_TOKEN` + `normalize`), so the bug was isolated to `_FTS_SAFE`. Tests:
  `test_accented_query_recalls_accented_message`, `test_accent_insensitive_both_directions`;
  full retrieval suite (208) green. **MEASUREMENT CAVEAT: there is NO Dutch eval set (LME/BEAM
  are English), so Dutch FTS is validated by correctness + unit tests, not a benchmark delta.**
  **BIGGER DUTCH LEVER STILL OPEN (needs a decision): stemming (boeken≠boek).** Porter is
  English-only and stdlib `sqlite3` can't register a custom FTS5 tokenizer (needs apsw/C ext,
  fights pip-install ease), so Dutch inflection isn't conflated. Options: query-side morphological
  expansion (driver-agnostic, fragile against the Porter index) vs. replace Porter with a
  bilingual scheme (migration + English-stemming risk). **DECISION 2026-06-08: DEFERRED until a
  Dutch eval set exists** — tuning stemming with no benchmark to measure it would be blind work
  (the same measure-first discipline applied to D4/router). The diacritic fix stands as the
  complete, high-confidence Dutch FTS deliverable for now.
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
- **Router TR polish (round 3) 2026-06-08.** Targeted the router-eval TR residual misses where
  TR SHAPING actually helps (carry-clean, not oracle-chasing). (1) **Activity durations:** added
  spend/take verbs (`_TR_DUR_VERB`, EN+NL) to `_TR_DUR_END` so a single-activity span with no
  between/after/ago anchor routes TR — "how many days did I spend on my camping trip", "how many
  days did it take to finish X" (were `mr_count`), "how long did I take to finish X and Y" (was
  `none` via `_TR_HOWLONG`). "last/lasted" deliberately excluded (collides with "last week").
  (2) **`_TR_AGE`:** "how old was I WHEN I moved" → age-at-event = two-point temporal calc (rule
  `tr_age`); guarded by the "when/toen" clause so "how old is my laptop" stays None. (3) **`_MR_TOTAL_CUE`
  guard (rule `mr_total`, checked FIRST):** an explicit "in total"/"altogether" sum cue on a count
  opener is aggregation across events (MR), so "how many hours have I spent … in total" (oracle MR,
  was mis-firing TR via the new spend verb) correctly stays MR — this resolves the TR/MR duration
  overlap DEFERRED from the precision round. Predicted on real eval rows: 4 TR misses recovered + 1
  MR mis-fire corrected; existing overlap guards (between-anchor TR, count-with-timeframe MR,
  duration-to-now TR) all hold. **SKIPPED (TR shaping wouldn't help → would be oracle-chasing):**
  recurring time-of-day ("what time do I wake up on Tuesdays"), superlative-over-period ("which
  airline did I fly most in March"), count-before-event ("how many charity events before X" — left
  to additive-MR). +5 test groups; full suite green. **MEASURED (box, S/500): TR recall 92→95%
  (+3pp, the 4 targeted spend/take/age rows flipped to TR exactly); MR recall 83→81% (−2pp, 3
  duration-shaped oracle-MR Qs shifted to TR); TR precision 82→79% (+4 FP across tr_howlong/
  tr_duration/tr_age); `mr_total` works (2 FP, both on abstention Qs).** **VERDICT: ACCEPT — the
  TR precision drop is a TAXONOMY ARTIFACT, not production harm, confirming the same pattern as the
  `mr_count` precision round.** The 4 TR FPs ("how long to assemble the IKEA bookshelf", "how old
  when grandma gave me the necklace") are linguistically-correct duration/age routings scored as FPs
  ONLY because the oracle labels them single-session by ANSWER-LOCATION, which production cannot see.
  TR shaping is ADDITIVE (augment.py: "the other tiers (graph/fts/messages) still run unchanged" — TR
  only appends `temporal_events`), so a TR FP is near-harmless like an additive-MR FP: a short/empty
  timeline layered on otherwise-full retrieval; the 3 MR→TR shifts keep full normal retrieval too.
  **GENERALIZED LESSON (now seen TWICE): router precision against LME systematically UNDERSTATES
  production quality — the labels encode answer-location the router can't access, and both MR and TR
  shaping are additive so misroutes don't suppress retrieval. Production-truth = end-to-end accuracy
  (full LME run), NOT router precision. Do NOT tighten tr_age/tr_duration to chase it — the only
  suppressor is reading answer-location = gaming.** Router work COMPLETE.
- **`benchmarks/router_eval.py`.** Zero-LLM sweep of `detect_ability` vs oracle over the
  full dataset (seconds on the box) + per-intent residual-miss listing. Tight tuning loop
  decoupled from the expensive retrieval+judge run.
- **Router hardening 2026-06-08 (production robustness + observability; NOT yet the
  precision/recall round).** Carry-clean — `detect_ability` IS the production path, so all
  of this lands in real Hermes. (a) **Robustness:** `_prepare()` now NFC-normalises input
  (composed vs decomposed Dutch diacritics match one way), tolerates non-str/blank input as
  an abstain instead of raising (it sits on every augment() hot path — must never crash the
  host on a malformed turn), and clips the scan to `_MAX_SCAN_CHARS=4096` so the lazy
  `[\s\S]*?` TR bridges can't rescan a multi-KB paste to EOS at every start position (the
  intent opener always sits at the question start, so a bounded prefix is both where the
  signal lives AND a worst-case cost cap). (b) **Observability:** new `AbilitySignal(ability,
  rule)` + `detect_ability_signal()` name the firing branch (tr_duration/tr_howlong/tr_order/
  tr_recency/mr_count/tr_distance, or none/empty/non_str); `detect_ability` is now a thin
  wrapper. `augment()` records `ctx.detected_rule` alongside `detected_ability`, so a
  production misroute ("why did this get MR-shaped?") is diagnosable from the result object
  alone. `router_eval.py` prints the firing rule per residual-miss/FP and a by-rule FP tally
  (names the worst-over-firing pattern). +5 hardening + 1 observability test groups in
  `test_intent.py` (non-str/blank/oversized/near-miss-no-hang/NFC + per-rule signal); all
  green on the Mac.
- **Router precision/recall round 2026-06-08 (measured on box, S/500).** Eval: MR recall 74%
  (89/121) precision 55% (89/163); TR recall 92% precision 82%; 89 FPs, **`by rule: mr_count=70`**.
  **KEY FINDING — the 55% MR precision is a MEASUREMENT ARTIFACT, not a defect; HOLD `mr_count`,
  do NOT chase precision.** "MR target" = `multi-session` ONLY (per `QUESTION_TYPE_TO_ABILITY`);
  the 70 `mr_count` FPs are `single-session-user`/`-preference` (→IE/PF→NONE) questions that are
  textually COUNT questions ("how many playlists do I have", "how much did I spend on a handbag")
  — oracle calls them NONE-target purely by ANSWER-LOCATION (answer lives in one session), a label
  production does NOT have. "How many playlists" is the same string whether the answer sits in one
  session or ten. Tightening `mr_count` to suppress them = overfitting to the session-location
  oracle label = the exact gaming the carry-over constraint forbids, AND costs real MR recall, AND
  undoes additive-MR (`mr_aggregate_additive`, default True) which already makes a false MR route
  near-harmless (count layers on relevance retrieval). The real MR metric is end-to-end accuracy
  under additive-MR (the full LME run), not router precision. **ACTION TAKEN — recall only:** new
  `_MR_AGGREGATE` pattern + rule `mr_aggregate` for aggregation phrasing with NO count opener
  ("total amount I spent" — `_MR_COUNT` needs "amount OF"; "what percentage"; "on average"; EN+NL),
  same MR aggregate path, separate rule so its recall/FP is measurable independently. +3 test groups
  (aggregation positives, own-rule reporting, precision guards); all green. **DEFERRED to TR polish:**
  the TR/MR duration-overlap misses ("how many days did I spend on my camping trip" → mr_count steals
  oracle TR) — genuinely ambiguous (inverse "how many hours have I spent...in total" is oracle MR), so
  it belongs in the TR round, not a half-fix here. **MEASURED (box, S/500): clean recall-only win.
  MR recall 74→83% (+9pp, 12 misses recovered: "total amount", "difference in price", "average
  age/GPA", "percentage discount"); MR precision 55→57% (UP — recall gain outpaced FP cost); TR
  flat 92%/82%; `mr_count` FPs held at exactly 70; `mr_aggregate` added only 2 FPs (total 89→91).**
  The lone notable `mr_aggregate` FP is an `_abs` Q ("total cost of my headphones and the iPad") —
  harmless: it's genuinely aggregation-shaped, and on an abstention Q additive-MR just layers a
  candidate count that comes back empty (reinforcing, not breaking, abstention). The remaining 20
  MR misses are NONE-type (no aggregation marker) or the deferred TR/MR duration overlap. **Router
  hardening COMPLETE** (robustness + observability + precision/recall); residual MR recall lives in
  the duration overlap, owned by the TR-polish round.

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

### Answer-context shaping
- **KU recency-dating lever 2026-06-09 (prompt-only, SHIPPED, carry-clean).** Two changes in
  `answer_question`: (1) date-stamp each `message_hit` with its `created_at` in the answer context
  (`[MEM 2023-11-30] …`); raw turns are the only tier carrying `created_at` (FACT/fts/episode stay
  undated, graph dating deferred). (2) Append a **value-aware recency clause** to BOTH default
  prompts (`ANSWERING_SYSTEM_PROMPT` strict + `ANSWERING_PERMISSIVE_PROMPT`): *"use the value from
  the most recent memory that actually states it; a later memory that only mentions the topic without
  giving the value does not override it."* **Why it works (probed by `ku_probe.py` before any run):**
  strict KU misses are "present-but-not-latest" — the new value IS retrieved, but a later turn merely
  re-mentions the topic without the value (37/37 spoiler turns tangential, 0 stale re-assertions), so
  naive latest-date-wins is wrong; value-aware recency fixes it. Dating helps the model pick among what
  it already SEES. **MEASURED (full 500, seed 0, auto-ability, no-dream):** STRICT **overall +5.4pp
  (58.6→64.0), KU +11.5pp (56.4→67.9)** — both outside the variance band, real; TR +6.0, MS +5.3 also
  gained (date stamps add recency awareness), SS-U flat 92.9 (clause inert without a multi-date conflict
  = cleanest no-regression evidence); no regressions. PERMISSIVE (shipping) **overall +2.2pp
  (65.4→67.6), KU +12.8pp (62.8→75.6)** — the clause STACKS on permissive, abstention held 70%. **Carry
  to production:** HyMem already exposes `MessageHit.created_at` (docstring names this use case), so
  Hermes date-stamps its context + carries the clause; no HyMem change — the adapter is the reference impl.

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
  **WIRED 2026-06-08 (`--permissive-default`, adapter-only, not yet run):** new
  `ANSWERING_PERMISSIVE_PROMPT` (preference-style but abstention-guarded — keeps "say I
  don't have enough information" + "do not invent specific facts") replaces the strict
  `ANSWERING_SYSTEM_PROMPT` in the **default/unknown-ability `else` branch** of
  `answer_question` (and the MR no-user-messages fallback) when the flag is set; PF/MR/TR
  branches untouched. So a label-free SS-P question (router → None → else branch) gets the
  permissive prompt without reading the oracle label. `compute_abstention_scores` +
  `print_abstention_scores` break every category into **answerable vs `_abs`** (overall +
  per-base-category), persisted as `abstention_diagnostics` in the run JSON and printed
  after the score table — so the SS-P recovery and any abstention regression are both
  visible in one A/B. Banner shows `Default answer prompt: PERMISSIVE|STRICT`; config
  records `permissive_default`. **A/B to run:** strict vs `--permissive-default`, same
  fixed seed (`--auto-ability` to score the production path); accept only if answerable
  (esp. SS-P) lifts WITHOUT sinking the abstention row. Adapter-side prompt change only —
  does NOT alter the load-bearing HyMem conclusion (real Hermes picks its own posture).
  **MEASURED 2026-06-08 (full 500, seed 0, --auto-ability, strict vs permissive):
  OVERALL 60.2 → 65.6 (+5.4pp). SS-P 10.0 → 63.3 (+53.3pp, the target) — 16 F→T flips
  (refusal→correct, all "I don't have enough information"→bridged), 0 T→F. Positive
  side-effects everywhere: KU +3.9, SS-A +3.6, SS-U +2.8 (92.9→95.7), MS +2.3, TR +0.7.**
  Clean win on the answerable axis; the abstention guard ("do not invent specific facts")
  held (0 T→F on SS-P). **GATING CAVEAT — the abstention axis is UNMEASURED: the `_cleaned`
  S dataset has ZERO `_abs` questions (all 500 answerable), so the answerable-vs-abstention
  report shows n/a for abstention and CANNOT confirm the permissive default doesn't trade
  refusals for hallucinations on unanswerable queries.** Since a permissive default's whole
  risk lives on the abstention axis and production (Hermes) is full of unanswerable queries,
  this is banked as an LME-ANSWERABLE win, NOT yet cleared to become a Hermes default.
  **Loader hardened 2026-06-08:** `load_longmemeval_data` now derives the `_abs`
  question_type from the official LongMemEval `question_id` suffix (official data flags
  abstention on the id, not the type — judge + report key on the type), and the banner
  prints whether abstention questions are present so a guard-rail-blind run is obvious.
  **NEXT to actually clear it: re-run the A/B on an abstention-bearing dataset** (the
  original/un-cleaned LongMemEval_S, or LongMemEval_M/oracle that retain `_abs`) and require
  the abstention row to hold, not just answerable to lift.
  **GATE CLEARED — MEASURED 2026-06-08 (abstention-bearing set, 500 incl. 30 `_abs`, seed 0,
  --auto-ability, strict vs permissive):** OVERALL 58.4 → 65.2 (+6.8pp). SS-P 13.3 → 56.7
  (+43.4pp, 14 F→T / 1 T→F = net +13). Answerable ALL 57.9 → 65.1. **Abstention ALL held
  flat 66.7 → 66.7 (20/30 both runs).** The aggregate "hold" is a CANCELLATION, not a clean
  pass: 1 `_abs` regressed (`gpt4_372c3eed_abs`, MS — strict correctly answered "4",
  permissive invented an Arcadia-High→UCLA-CS narrative) and a different `_abs` improved,
  netting zero. The guard held **100% on SS-U and TR `_abs` both runs** (single-turn
  user-fact questions — where a hallucinated fact is most damaging); the one leak was on
  **multi-session synthesis**, the predictable soft spot (a permissive prompt licenses the
  cross-session bridging that MS over-extends into confabulation). **Verdict: cleared for
  the LME headline; cleared-with-documented-caveat as a Hermes default — abstention guard is
  tight on single-turn facts, leaky on multi-hop synthesis. Do NOT call the guard "perfectly
  tight"; the MS case forbids it. If shipped as Hermes default, either tighten the guard for
  the multi-hop case or accept MS abstention as the known monitored residual.**
- **D5. Disabling dream to "win" on LME.** LME is single-conversation-haystack; the
  cross-session KG dream builds is invisible to it. "No-dream wins" is a statement about
  LME's blind spot, not HyMem — would gut the product differentiator.
- **D6. SS-preference ceiling (67%) as a retrieval lever.** Judge/ceiling-bound, not
  retrieval. Settled.
- **D7. TR FP regex-guarding.** Additive shaping already makes TR FPs near-harmless and
  TR is strong (92% recall); guarding risks the recall.
- **D8. Q45-class entity-precision misses** ("model grabs niece vs cousin"). Reader
  precision, not a retrieval/temporal gap. Low ROI at current n.
- **D9. KU ranking / top_k / budget / rerank as a lever.** `ku_probe.py --ranking` (72 KU items,
  the recall-ceiling "ranking miss" disambiguated by reconstructing the final answer context offline)
  found **B=0** — the gold turn reaches the answer context in 70/72, IDENTICALLY under raw BM25 and
  the LLM reranker. So KU's residual is NOT truncation and NOT reranker-demotion (`rerank_message_hits`
  gains nothing, no B→C flips). The ~23 remaining KU misses are **synthesis** (gold in context, model
  still wrong — a few are judge noise per [[project_lme_variance_band]], real core ~17-21) + **2 recall
  misses**. Don't spend a run on KU ranking/top_k/budget/rerank — exhausted. The dating lever (§2) is
  the KU win; further KU gains would need conflict-resolution prompt strength or the 2 recall misses,
  both low-yield.

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
  - **FLOOR AUDIT — MEASURED 2026-06-07, MS RETRIEVAL STORY CLOSED.** Instrumented the adapter's
    `per_question` block with **per-gold-turn fused-pool membership** (`gold_turns_in_pool` +
    `gold_turn_tiers`, `"none"` = unrecovered turn — `recall_ceiling`'s any-match can't answer this).
    Ran `--floor-audit` on an instrumented baseline, embeddings OFF then ON. **Result: 0 PHANTOM both
    ways (15/15 REAL off, 14/14 REAL on).** Every floor gold turn evades EVERY tier (message FTS +
    chunk FTS + vector) — "new retrieval path needed / fundamentally unrecoverable" is now MEASURED.
    The embeddings-on re-audit was forced by a methodology catch: L1's "no vec-only bucket" was an
    ANY-MATCH metric, blind to whether the *specific* per-turn floor turns were vec-recoverable (same
    conflation trap, one level up). Head-to-head OFF→ON: ranking misses 66→58 (−8), coverage-short
    38→38 (identical), floor 15→14 → **embeddings re-rank (lift deep-but-pooled gold), they don't
    recall (no new gold in pool)**. **Final MS miss breakdown (emb ON, n=58): 14 REAL floor (~0 banked
    headroom) · ~24 deep-lexical (reranker+emb already extracted most, residual = lexical ceiling) · 20
    synthesis (answer-side, bank).** Retrieval headroom on MS ≈ 0 (perfect floor fix ≈ 9 correct ≈
    1.8pp overall, only IF a buildable path exists AND it carries to real Hermes — least-likely slice
    to generalize). **VERDICT: MS is ranking/synthesis-bound, not recall-bound. Retrieval ceiling
    reached.** (Adapter DX fix: `--embeddings` now works with zero env setup — local FastEmbed
    defaults passed explicitly, `HYMEM_EMBEDDING_*` still overrides.)
  - **FLOOR INSPECTOR built (`--inspect-floor <instrumented.json>`) — characterize the 14.** Turns
    "unrecoverable" from a count into a named failure mode per question: reads the run's floor qids
    (ranking miss + ≥1 `"none"` tier), then ingests/dreams/searches each and dumps the question, the
    unrecovered gold turn(s) + haystack location + raw msg-FTS rank, the question↔gold salient-token
    overlap (the VOCAB-GAP signal), and what ranked instead. Heuristic mode tag (VOCAB GAP / WEAK
    OVERLAP / IMPLICIT) + tally. Run WITH `--embeddings` + full dream to reproduce the audited floor.
    Decides whether a new path (paraphrase/HyDE/query-expansion vs turn-linking) is worth building —
    weighed against carry-over (an LME-only fix is out of scope).
  - **FLOOR INSPECTOR — MEASURED 2026-06-07, FLOOR BANKED AS UNRECOVERABLE.** Tally over 88 floor gold
    turns / 35 questions: **VOCAB GAP 60 · WEAK OVERLAP 28 · IMPLICIT 0.** Zero IMPLICIT is the load-
    bearing number — there is NO hidden multi-hop/turn-linking band the token heuristic was missing, so
    the KG-bridge path (the one lever that would have both carried to Hermes AND moved the floor) has no
    target here. But the dumps reframe the 60 "VOCAB GAP" as a **sparse-signal / signal-to-noise problem,
    not a vocabulary problem**: the gold fact is a single incidental phrase inside a long, topically-
    unrelated user turn (Q "how many years older is grandma than me" → gold turn is 90% Europe travel
    advice, "is 32 considered young or old?" buried in it; Q "how many model kits" → "trying enamel washes
    on my 1/72 B-29" in passing; Q "items of clothing to pick up" → "pick up my dry cleaning for the navy
    blazer"). The heuristic reads zero token overlap because the turn's DOMINANT content is off-topic — the
    link is purely contextual (same user, different subject). **No retriever HyMem can build — lexical,
    semantic, or graph — can distinguish an incidental "32" in a travel message from the rest of the
    haystack; the bottleneck is SNR *within the gold turn itself*.** The only fix is a reader that spots
    the needle in a 90%-off-topic turn — answer-side, not retrieval; and not even a HyMem lever. **VERDICT:
    the 14-floor is fundamentally unrecoverable by retrieval. MS retrieval is fully closed.** (One option
    explicitly considered and set aside: surface incidental facts at DREAM/extraction time into structured
    memory — that IS a HyMem-native, carry-over-clean lever in principle, but these are counting/aggregation
    Qs over incidental side-details with weak Hermes carry-over and high over-extraction risk; not worth it
    now. Logged as a candidate, not a plan.)
- **L4. Permissive/abstention-aware default answer prompt (harness, optional, gated).**
  Only if a clean banked benchmark number is wanted — run WITH the `*_abs` slice broken
  out (see D4). Not load-bearing for the HyMem conclusion.
- **L5. MR dead-filter fix — the active MS lever 2026-06-09 (carry-clean, one-region fix).**
  **The bug:** `answer_question` builds `context` from the FULL `memories` (incl. assistant turns),
  freezes it, THEN — too late — reassigns `memories` to a user-only filter that nothing reads
  ([longmemeval_adapter.py:696](longmemeval_adapter.py#L696) freeze, [:713](longmemeval_adapter.py#L713)
  dead reassign). So MR-shaped questions see assistant-polluted context despite the MR preamble claiming
  "assistant echoes excluded" (line 665) — the preamble lies about what the model sees. **Probed before
  fixing (`ku_probe.py --ms`, n=121, both raw BM25 and reranker ON):** ability distribution **MR=98 (81%),
  None=13, TR=10** — MR is the overwhelming share of MS, so the blast radius is huge. A/B/C: **A 8-10,
  B=0, C 111-113** — gold reaches context, residual is in-context. Dead-filter what-if over the 98
  MR-detected (avg **53% assistant noise** in context): **SAFE 92 (every gold turn is a user message —
  filter keeps gold, drops noise) · RISK 0 (no gold is an assistant turn that would be dropped) · n/a 6
  (gold not retrieved anyway = A-bucket).** SAFE dominates, RISK is exactly zero, over half the MR context
  is wasted assistant echo. **The fix:** apply the user-only filter to the iteration list BEFORE the
  context-build loop (not after the freeze), keeping the existing "no user messages → fall back to default
  prompt + unfiltered context" guard. **Carry-clean:** the filter is keyed on the production `detect_ability`
  → MR route (not the oracle label) and on message role, both of which Hermes has. Reranker side-note: the
  LLM reranker slightly HELPS MS (A 10→8), unlike KU where it was neutral — consistent with
  `rerank_message_hits` being net-positive for MR.
  - **KILLED — MEASURED 2026-06-09 (full 500, seed 0, auto-ability, strict + permissive, dating-only vs
    dating+filter). REVERTED.** The filter is **neutral on its MS target** (permissive 45.1→45.1; strict
    44.4→42.9, within the [[project_lme_variance_band]]) — confirming the 53% assistant noise was NOT
    hurting the model; it ignored/padded with it. But **overall slips −1.4pp in BOTH postures** and the
    damage lands on **non-MR-target categories**, uniformly negative, largest on **SS-A (−5.4 strict /
    −7.2 permissive)**. **Root cause (the probe's blind spot):** under `--auto-ability` the filter fires
    on the *route* (`detect_ability == MR`), and that route has FALSE POSITIVES — count-shaped questions
    whose true category is SS-A/KU/SS-P (documented: `mr_count` has ~70 FPs, all single-session). For an
    FP-routed question the filter **SUBTRACTS** the assistant turns from context, and **SS-A =
    single-session-ASSISTANT** means the gold answer can live in an assistant turn → stripping assistant
    turns drops the gold. SS-A negative in BOTH postures with the largest magnitude is the fingerprint.
    **This breaks the additive-MR safety property** (§2 additive-MR / D3): MR false-positives are only
    tolerable because a false MR route keeps FULL relevance retrieval and just layers a count on top. A
    user-only context filter makes MR **subtractive**, reintroducing exactly the harm additive-MR was
    built to neutralize. **The `ku_probe.py --ms` RISK=0 was CATEGORY-conditioned (measured over true-MS
    questions, where gold is a user turn by nature), not ROUTE-conditioned — it never sampled the MR-FP
    population from SS-A/SS-P/KU.** **LESSON (bank it): never SUPPRESS/filter retrieval keyed on a routed
    ability that has false positives — only ADD on it (the additive-MR invariant). The dead user-only
    filter is harmless precisely BECAUSE it's dead; "fixing" it is a regression.** The dead no-op stays
    as-is in the adapter; the dating clause (§2) is the real MS-adjacent lever. Cheap post-hoc confirm
    (no new run): in the run JSON, isolate `detected_ability==MR` ∧ oracle `single-session-assistant`
    and check the T→F flips between the two runs.
  - **CONFIRMED 2026-06-09 (the post-hoc check, no new run).** 7 SS-A questions mis-route to MR; the
    filter flips ALL 4 that were previously correct (Speyer tourism phone number; French-omelette egg
    count; HAMT avg framerate; Chiefs-vs-Jaguars score) check->wrong; the 3 already-wrong stay wrong.
    These are "remind me what you told me about X?" questions — gold lives in a past ASSISTANT turn; the
    router reads them as count/aggregate-shaped ("how many eggs", "average framerate"), MR strips the
    assistant turns, gold gone. The full SS-A -5.4/-7.2 regression = these 4 flips. **Key corollary:
    pre-filter these 4 were MR-routed AND correct — additive-MR already absorbed the mis-route (full
    retrieval kept, answer found). So there is NOTHING to fix in the router; the route was never the
    problem.** A router "fix" to stop MR-routing "how many eggs" would (a) read answer-location = gaming
    (per the §2 router rounds / twice-seen lesson: `mr_count` FPs are single-session count-shaped strings
    the router CANNOT distinguish from real MR — same surface form), and (b) cost real MR recall — both
    already-rejected dead-ends. **The only error was making MR subtractive. Resolution: MR shaping stays
    ADDITIVE-ONLY (count layered on full retrieval); the dead user-only filter stays dead. No filter, no
    router change — done.**

**Sequencing:** L1 done (recall ruled out). L2a KILLED by the gold-rank probe (92% MS gold
already at BM25 ≤15). L2c DONE → reranker is net-positive (−4.5pp MS when OFF), keep it, L2b
moot → **L2 chapter closed.** L3 DONE → **DEAD in both forms** (cut-sweep recovers ≤8/36,
pack-sim 1/36; the 21 non-floor MS misses are intra-session lexical-rank, untouchable by
window packing/widening). NOW: **floor audit** — instrument per-gold-turn fused-pool membership,
re-run baseline once, `--floor-audit` splits the 15 floor misses into phantom (chunks rescue)
vs real recall gap. **DONE 2026-06-07: floor inspected → all sparse-signal, 0 IMPLICIT →
retrieval is fully closed on MS.** The remaining headroom is ALL answer-side. The next (and only
clean) answer-side lever is **D4's permissive default prompt with the `*_abs` slice broken out**
(SS-P crater, ~3.7pp ceiling, prompt change not a HyMem change) — pursue ONLY if a banked LME
number is wanted; the higher carry-over work is the production levers (router hardening, Dutch
FTS, TR polish). Every retrieval step was data-gated by a free LLM-less probe before any feature
build — the discipline that killed L2a, L2b, and both halves of L3, and now closes the floor.

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

- Bi-temporal edges (Zep/Graphiti `valid_at`/`invalid_at`). **Phase 1 (schema v15
  columns) LANDED; the supersession wiring is P2 below.**
- RAPTOR-style aggregation nodes in dreaming (staleness via `digested_version`).
  **GATE PASSED → BUILT (off by default).** `raptor_cluster_probe.py` on the MS
  ranking misses (grid emb≥0.55 OR ent≥0.50): of 53 misses, 31 had gold turns that
  became episodes; **27/31 (87%) co-located all gold episodes in ONE cluster**
  (mean 1.13 gold-clusters vs 16.8 clusters/question) — one node summary bundles the
  synthesis inputs. The other 22/53 (42%) have NO gold episode at all = a dream
  **coverage** gap (episode-extraction recall), a separate prior lever, NOT a
  clustering gap — does not block the build. Built: schema v16 (`aggregation_nodes`,
  `aggregation_node_embeddings`, migration `016`), `hymem/dreaming/aggregate.py`
  (canonical clusterer; probe re-exports it), runner step, additive off-by-default
  retrieval tier (`cfg.aggregation_nodes_enabled` → `ctx.aggregation_nodes`),
  `tests/test_aggregate.py` (10 offline). **Remaining: G4 = the enabled-vs-disabled
  LME A/B on the box (must not regress the 70.0 baseline / 51.9 MS floor) before it
  ships on by default.** Only clusters spanning ≥2 sessions with ≥2 episodes are
  summarized (singletons/single-session cost no LLM call); nodes are full-rebuilt each
  dream (content-hash id → unchanged cluster reuses its cached embedding).
- Relative-date parsing ("twee weken geleden") — needs `dateparser`, deferred against
  the zero-dependency hardening goal.
- `messages_fts` not carried by export/import.
- Tokenizer `porter` (English) vs Dutch-first scope. **Unblocked by P5 below.**

### Candidate levers — proposed 2026-06-10, NOT yet probed/built

Same contract as §4: front-run gate before any build; additive-only (the MR-filter
lesson); nothing reads the oracle label. None re-chase D1–D9. Roughly EV-ordered.

- **P0 (measurement, run first). Reader-parity run.** One full-500 seed-0 run with a
  stronger answer model through the existing pluggable client — same config, same judge
  posture. Decides how much of the 19pp gap to Hindsight (89.4) is reader strength vs
  architecture: even PERFECT MS only reaches ~82 from the 70.0 canonical baseline, so
  the gap is distributed and the reader is the dominant unmeasured variable (D2/D8/KU
  residual are all documented deepseek reader weaknesses). Report the reader alongside
  the number — condition-honesty, not gaming.
- **P1. Question-conditioned fact distillation at read time (map-reduce reader).** Before
  the final answer call, map over retrieved hits ("extract any statement relevant to
  {question}, else NONE"), then answer over the distilled list. Targets THREE banked
  buckets at once: the 14 sparse-signal floor (each turn read individually → the
  incidental "32" gets spotted — the floor inspector's "only fix is a reader that spots
  the needle"), MS synthesis (~20: fuse ~15 one-line facts, not 45 raw slots — RAPTOR's
  benefit without the clustering bet), and D2's can't-tally (tallying a short extracted
  list is an easier task). Question-conditioned + transient sidesteps the over-extraction
  risk that shelved write-time incidental extraction. Cost: N small LLM calls/query —
  gate on route (MR/TR) or high hit-count. Additive (distilled facts join, never replace,
  raw hits). Sequence vs RAPTOR: fallback if co-location kills it, complement if not.
  **Free front-run: dry-run offline on the 20 banked MS synthesis misses.**
- **P2. Bi-temporal KU supersession (wire the landed v15 columns).** Dream-time
  contradiction detection: new fact conflicts with stored (same subject/predicate,
  different value) → stamp old edge `invalid_at`; retrieval demotes/excludes invalidated
  facts. Converts KU correctness from prompt-side hope (the §2 recency clause — the
  reader must apply it) into a property of the store — load-bearing for real Hermes,
  where the reader prompt isn't ours and conversations span months. Measurable target:
  the ~17–21 KU conflict-resolution residual (D9).
- **P3. Query rewriting for anaphora (the real-life lever LME is blind to).** Every LME
  question is self-contained; real Hermes queries aren't ("what did she say about
  that?"). Resolve pronouns/ellipsis against recent turns BEFORE the retrieval tiers —
  raw-query FTS gets pronouns, vec gets vagueness, so both tiers miss today. Standard
  conversational-RAG move, additive, through the existing client Protocol (cheap
  heuristic pass first, LLM fallback). No LME delta by construction (D5-style blind
  spot) — production value only, likely worth more there than any remaining LME point.
- **P4. Typed user-profile tier (bounded incidental-fact extraction).** The SAFE version
  of the shelved floor-inspector option: extract only schema-constrained first-person
  assertions at dream time (ages, names, relationships, possessions, preferences,
  locations) — closed vocabulary keeps precision high, unlike open-ended incidental
  extraction. Where most of the 14-floor lives, and the memory feature users actually
  notice ("you remembered my daughter's name"). Honcho-style user representation, native.
- **P5. Dutch mini eval set (unblocks the deferred stemming decision).** Machine-translate
  a stratified ~100-Q LME slice (questions + haystacks) → `LME-NL-mini`. Not
  publication-grade; exists solely so Dutch FTS work (stemming: boeken≠boek, §2
  diacritics follow-up) stops being blind — measure-first applied to creating the
  measure. Cheap (a few dollars of MT).
- **P6. Cross-encoder rerank for production latency (not a score play).** Shipping config
  spends a full LLM round-trip reranking the message tier on EVERY query. L2c proved
  reranking is net-positive; nothing proved the LLM must do it. A/B `--rerank-model
  cross-encoder` (already wired; `bge-reranker-v2-m3` for multilingual per L2b note) —
  accept if quality holds at ~50ms local. Latency/cost lever for Hermes; LME-neutral
  expected.
- **P7. Usage-signal feedback (longer-term).** Track which retrieved memories the answer
  actually relied on (reader cites hit IDs, or verbatim-overlap detection) → small
  ranking prior: retrieved-but-ignored decays, cited boosts. Invisible to any benchmark;
  pays off in long-running deployments.
