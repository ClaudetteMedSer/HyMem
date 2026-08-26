# Read-side synthesis plan — flip-watch → P0 reader parity → P1 bounded reflect

*Written 2026-07-24, branch `Beam-optimisation`. Companion to
`benchmarks/raptor_digest_plan.md` (RAPTOR product thread),
`benchmarks/longmemeval_roadmap.md` (LME record — P0/P1 are its candidate
levers), and `additional_planning.md` (Idea A/B, Plan C). Same standing
contract: front-run gate before any build, additive-only, mechanism > score,
nothing reads oracle labels at decision time, per-category LME deltas under
~±5pp are noise (strict runs swing ~4 questions/category on identical config).*

**Motivating context (from the 2026-07 Hindsight review):** Hindsight
(Vectorize.io, arXiv 2512.12818) reports 91.4% LongMemEval (MS 79.7) on an
OSS-120B-class reader with an agentic ≤10-iteration "reflect" answer loop.
HyMem's canonical is 70.0% full-dream (MS floor 51.9) on deepseek-chat with a
single-shot answerer. Three independent closed HyMem investigations agree the
residual is **answer-side synthesis, not retrieval**: LME retrieval CLOSED
(floor audit: 14-floor unrecoverable, MS decomposition banks 20 synthesis
misses), all three BEAM floors closed as answer-side across 5 architectures,
and the MS recall-ceiling verdict. The gap is also reader-confounded: the
five-way BEAM answerer A/B swung ±8pp from answerer choice alone. Hence, in
order: **Phase 0** read the RAPTOR flip-watch (free, unblocks Plan C),
**Phase 1** P0 reader-parity run (one measurement run — decides how much of
the ~21pp gap is reader vs architecture), **Phase 2** P1 question-conditioned
fact distillation (a bounded single-step "reflect"). **Track A** (Idea A
multi-hop, `max_hops=2`) runs in parallel — it touches only `query/augment.py`,
never episodes, so it cannot disturb the watch.

**What we deliberately do NOT import from Hindsight:** the Postgres/server
shape (embedded-first is HyMem's identity), and a belief/opinion tier
(Hindsight deprecated theirs — independent validation of never building one).
Idea B (rules ≙ their directives network) and P6 (cross-encoder rerank) stay
backlog: real, but they move no number this plan is gated on.

---

## Phase 0 — RAPTOR Stage 3c flip decision (box read, zero LLM cost)

The four reuse fixes landed 2026-07-12 in `bb96057` (leftover-resample
amplifier, fusion-failure leak/poison cascade, VACUUM rowid-renumber vs
vec/FTS shadows, ceiling-after-drain hole; salts bumped `cluster.v4` /
`rollup.v3`; schema v22 adds the attribution columns). ~12 days of
`dream_runs` rows have accrued since. Post-fix repro worst case was 87%
(pre-fix 29–50%). This phase READS that data and takes the flip decision the
raptor plan's Stage 3c has been blocked on. No build until 0.4.

### 0.1 Pull the watch window

On the box (store = `~/.hermes/hymem.sqlite` for the server user; read-only):

```bash
sqlite3 "file:$HOME/.hermes/hymem.sqlite?mode=ro" -header -column "
SELECT id, started_at,
       aggregation_nodes_built   AS built,
       aggregation_nodes_reused  AS reused,
       CASE WHEN aggregation_nodes_built > 0
            THEN ROUND(100.0 * aggregation_nodes_reused / aggregation_nodes_built, 1)
            END                  AS reuse_pct,
       aggregation_fusion_failures AS failures,
       aggregation_input_episodes  AS input_eps,
       aggregation_blocking        AS blocking,
       skipped_locked, error
FROM dream_runs
WHERE started_at >= '2026-07-12'
ORDER BY id;"
```

Save the raw table into this doc's RESULT block (and the same dump as CSV next
to the run artifacts, e.g. `~/.hermes/benchmarks/flipwatch_2026-07.csv`).

### 0.2 Classify every row (the attribution contract from migration 022)

Work top to bottom; each row gets exactly one label:

1. **`deploy-refusion` (expected once):** the first post-deploy dream after the
   `cluster.v4`/`rollup.v3` salt bumps legitimately rebuilds everything —
   `reused ≈ 0`, `built ≈ full tree`. Identify it from the data (first row
   after the v22 migration where built is high and reused ~0); exactly ONE such
   row is expected. Excluded from the verdict.
2. **`failure-attributed`:** `failures > 0`. LLM-flakiness event, not
   membership churn; each fail→heal transition costs one low-reuse run.
   Excluded from the reuse verdict but COUNTED separately (see gate).
3. **`blocking-flip`:** `blocking` mode differs from the previous row
   (`'knn'` ↔ `'exact:<reason>'`). Environment split between trigger paths
   (one missing sqlite-vec) — the smoking gun from the July hunt. This is an
   ops bug to fix (unify the environments), not clustering churn. Excluded
   from the verdict, blocks the flip until fixed.
4. **`append`:** `input_eps` grew vs the previous row (new episodes entered
   the store between dreams). Under the oldest-anchored windows this must now
   confine churn to the newest tail window — these rows are IN the verdict.
5. **`quiescent`:** none of the above. IN the verdict.

### 0.3 Gate G-FLIP (numeric, from the banked flip criterion)

Over all `quiescent` + `append` rows in the window:

- **PASS** iff every such row has `reuse_pct ≥ 90` (append dreams explicitly
  included — this was the failure mode both prior watches died on), AND there
  are zero `blocking-flip` rows, AND `failure-attributed` rows are ≤ 2 in the
  window (occasional transport flakiness is tolerable; a streak means the
  fusion path is unhealthy and reuse numbers are unreadable), AND there is no
  unclassifiable low-reuse row (a row < 90% that fits none of labels 1–4 means
  a fifth cause exists — that is an automatic FAIL regardless of averages).
- **FAIL** otherwise. On FAIL: do NOT flip; append the classified table + the
  offending rows to `raptor_digest_plan.md` Stage 3c as a third failed-watch
  RESULT, and route by label — `blocking-flip` → fix env parity (sqlite-vec on
  both trigger paths), `failures` streak → LLM transport/retry work, an
  unclassifiable row → reopen the windowing analysis with the row's
  `input_eps`/`built` delta as the starting evidence. Plan C stays blocked.

Sanity floor: if the window contains < 5 usable (quiescent+append) rows,
extend the watch instead of deciding — the gate needs at least 5 verdict rows.

### 0.4 On PASS — the flip (one-line build + doc ripple)

> **DONE 2026-08-26 — G-FLIP PASSED (7/7, re-anchored window, 5 verdict rows,
> min reuse 91.3%) and every step below is executed.** Result table + verdict
> banked in `raptor_digest_plan.md` (Stage 3c FLIP-WATCH RESULT, v6). Steps 1-3
> and 5 landed in one commit; step 4 (post-flip verification on the box) is the
> only item still pending an observation.
>
> Two amendments the flip made to this section as written:
> - **Step 4 is a WEAK test, not a confirmation.** Prod has run with the layer
>   ON via env on all three launch paths since the same-day env-parity fix, so
>   the effective config is unchanged and the verification dream cannot fail for
>   flip-related reasons. High reuse there is near-guaranteed and is not
>   evidence for the flip; a refusion means something else moved.
> - **Step 3 needed a fourth item this section did not list.** The flip changes
>   every DEFAULT-CONFIG consumer, and three benchmark adapters were exactly
>   that: `msc_adapter` (reused wholesale by `locomo_adapter`) and
>   `beam_adapter` never pinned the flag, so the one-line change would have
>   silently switched the layer + digest ON for the LoCoMo / MSC / BEAM dream
>   path and broken comparability with the canonical baselines behind them.
>   Both now pin `overrides["aggregation_nodes_enabled"] = False` in the same
>   commit (dict-item form — that is the string to grep for).
>   `longmemeval_adapter:532` already pinned True. Generalize the lesson: a
>   config-DEFAULT flip has a blast radius the size of its default-config
>   consumer set, and that set is not what the gate measured.


1. `hymem/config.py:112` — `aggregation_nodes_enabled: bool = False` → `True`,
   docstring updated to record the flip date + gate evidence pointer.
   Bootstrap semantics already do the right thing (`hymem/bootstrap.py:86`: an
   UNSET `HYMEM_AGGREGATION_NODES_ENABLED` leaves `None` and the dataclass
   default wins), so after the flip the env var becomes an explicit OFF switch
   — no bootstrap change needed. Verify the startup `log.info` still reports
   the effective state.
2. Tests: full suite green locally; adjust any test pinning the old default
   (expect a handful asserting the layer is off by default).
3. Docs: `raptor_digest_plan.md` Stage 3c → RESULT block with the classified
   table + "FLIPPED <date>"; `hymem/Hermes_instruction.md` checklist item 2
   (env var no longer required at/after this build — keep as legacy note);
   `additional_planning.md` Plan C sequencing constraint → "UNBLOCKED <date>".
4. Post-flip verification on the box after deploy: next dream's `dream_runs`
   row shows the layer on with high reuse (no salt was bumped by the flip, so
   NO refusion is expected — a refusion here is itself a red flag).
5. Plan C (episode granularity) is hereby unblocked but remains OUT OF SCOPE
   for this plan — it rewrites episode membership and must not overlap the
   post-flip verification dream. Schedule it as its own effort per
   `additional_planning.md`, and re-verify reuse once after it lands.

**Deliverable:** classified watch table + verdict banked in
`raptor_digest_plan.md`; on PASS, the one-line flip committed + Plan C
unblocked in writing.

---

## Phase 1 — P0 reader-parity run (one measurement run, run before P1's A/B)

**Question this answers:** how much of the ~21pp gap to Hindsight's 91.4 is
reader strength vs architecture? Even perfect MS only reaches ~82 from 70.0,
so the gap is distributed — and D2/D8/KU residuals are all documented
deepseek reader weaknesses. One run, condition-honest (reader reported next to
the number), no tuning.

### 1.1 Build: unlock the answer client's base URL (small adapter change)

`benchmarks/longmemeval_adapter.py` — the `LLMClient` hardcodes
`self.base_url = DEEPSEEK_BASE_URL` (line ~284), so `--answer-model` alone
cannot reach a non-DeepSeek endpoint. Changes:

- `LLMClient.__init__(model, api_key, base_url=DEEPSEEK_BASE_URL)`.
- New flags: `--answer-base-url` (default `DEEPSEEK_BASE_URL`) and
  `--answer-api-key` (default: fall back to `--api-key`). Thread them into
  the ANSWER client only.
- **The judge client stays exactly as-is: `deepseek-chat` against
  `DEEPSEEK_BASE_URL`, same per-type judge prompts.** Judge posture is the
  comparability contract with the canonical 70.0 — never vary it in this plan.
- Record `answer_base_url` in the output `metadata` block (answer_model is
  already recorded) so the run is self-describing.
- Regression check: with the new flags absent, a `--sample 2` smoke run is
  byte-path-identical to today (defaults preserved).

### 1.2 Choose the parity reader (decision rule, not taste)

First choice: the same class Hindsight rode — **`gpt-oss-120b` via any
OpenAI-compatible endpoint** (this is what makes the comparison
apples-to-apples). Fallback if unavailable/impractical: the strongest
OpenAI-compatible reader accessible, reported honestly as "strong-reader",
not "parity". Pin: temperature path unchanged (adapter uses 0.0 for answers),
`max_tokens=1024` unchanged.

### 1.3 The run (box)

Reproduce the canonical config EXACTLY except the answer endpoint. Do not
reconstruct the config from memory — read the canonical run's `metadata`
block from its results JSON in `~/.hermes/benchmarks/` (the 70.0 full-dream
500q seed-0 run) and mirror every flag (`--sample 0 --seed 0`, full dream —
no `--no-dream`, embeddings/permissive/rerank flags as canonical). One run:

```bash
python benchmarks/longmemeval_adapter.py \
  --sample 0 --seed 0 --workers 8 \
  --answer-model <parity-model> \
  --answer-base-url <endpoint> --answer-api-key $KEY \
  <remaining flags copied verbatim from canonical metadata>
```

### 1.4 Reading the result (no gate — this is measurement)

Bank in `longmemeval_roadmap.md` under P0, alongside the reader identity:

- **Overall vs 70.0**, per-category deltas (only ≥5pp meaningful), and the
  abstention slice broken out.
- **Attribution split:** `architecture gap := 91.4 − (HyMem @ parity reader)`;
  `reader gap := (HyMem @ parity reader) − 70.0`.
- Decision guidance downstream: if the parity reader lands ≥ ~80, most of the
  gap was reader — P1's job is the residual MS/synthesis slice and its value
  should be judged at parity (see 2.5). If it lands ≈ 70–73, the gap is
  architectural — P1's stakes rise, and its A/B under deepseek is already the
  honest test. Either way P1 proceeds; its own go/no-go is the free dry-run
  (2.2), not this number.

**Cost note:** 500 answer calls at ~8–16k context on the parity endpoint +
500 deepseek judge calls. Budget before running; the judge side is unchanged.

---

## Phase 2 — P1 question-conditioned fact distillation ("bounded reflect")

**Mechanism:** before the final answer call, map over the retrieved hits with
a small extraction call each — "extract any statement relevant to
{question}, else NONE" — then answer over the distilled list PLUS the raw
hits (additive; distilled facts join, never replace — the MR-filter lesson is
an invariant). This is the single-iteration approximation of Hindsight's ≤10
reflect iterations, and it targets three banked buckets at once: the 14-floor
sparse-signal misses (each turn read individually → the incidental "32" gets
spotted), the ~20 MS synthesis misses (fuse ~15 one-line facts, not 45 raw
slots), and D2's can't-tally (tallying a short extracted list is easier).
Question-conditioned + transient sidesteps the over-extraction risk that
shelved write-time incidental extraction.

### 2.1 Recover the 20 banked MS synthesis misses (free, box artifacts)

Source: the instrumented emb-ON floor-audit run JSON (2026-06-07) in
`~/.hermes/benchmarks/` — the run whose banked decomposition is
"n=58: 14 REAL floor · ~24 deep-lexical · 20 synthesis". Selection rule if
the qid list must be re-derived from `per_question`:

- `question_type` = multi-session (non-`_abs`), `correct = false`,
  `recall_ceiling = true` (gold was in the pool → not retrieval),
- NOT floor: no `"none"` entry in `gold_turn_tiers`,
- NOT deep-lexical: the gold turn survived into the context actually sent to
  the answerer. This last split isn't a stored field — verify it in the
  dry-run harness by re-rendering the context (same `answer_question` builder,
  same char caps) and checking the gold turn text is inside it. Rows where
  gold fell below the context cut are deep-lexical → exclude.

If the recovered set ≠ 20, reconcile against the banked decomposition and
record the actual n — the gate below scales as a fraction, not an absolute.

### 2.2 Free front-run gate: offline dry-run on the banked misses

Build a dry-run mode on the adapter mirroring the `--inspect-floor` pattern
(`_inspect_floor_questions`, line ~1331 — it already knows how to rebuild a
per-question temp DB, ingest, dream, and search from a results JSON):

- `--distill-dryrun <instrumented.json>`: for each selected qid — rebuild,
  retrieve (same config as the source run), apply the distillation map, answer
  over `[DISTILLED EVIDENCE]` + raw context, judge with the standard deepseek
  judge. Also run the SAME harness over an equal-sized random control sample
  of MS HITS (`correct = true`) from the same JSON, to catch regressions.
- Distillation prompt, versioned constant in the adapter:

  ```
  DISTILL_PROMPT_V1: From the memory excerpt below, extract every statement
  relevant to this question, quoting concrete values, names, and dates
  verbatim. One line per statement. If nothing is relevant, reply exactly
  NONE.
  Question: {question}
  ```

  One call per rendered hit (per-hit is the Hindsight-faithful shape and the
  cost is trivial at dry-run scale; batching 8–10 excerpts per call is a
  recorded COST KNOB for Hermes, not a v1 variable). `NONE` replies are
  dropped; kept lines render as a `[DISTILLED EVIDENCE — extracted per-turn,
  verify against the memories below]` block ABOVE the raw memories block.
  Distillation reads only the question + hit text — label-free by
  construction.
- Log per question: flip status, distill calls made, lines kept, tokens.

**Gate G-P1a:** ≥ 25% of the banked synthesis misses flip to correct
(≥5/20), AND control regressions ≤ 1, AND a hand-read of every flipped
answer shows no newly invented fact (the judge can be charitable; the
hand-read is the honesty check). One prompt iteration (`DISTILL_PROMPT_V2`,
salt-style version bump) is allowed if the first attempt fails on a visible
prompt defect; a second failure banks P1 as dead — no full run, record the
verdict in `longmemeval_roadmap.md` and stop this phase.

### 2.3 Adapter integration for the A/B

- `--distill` flag on the adapter: in `_evaluate_one_question`, after
  retrieval and before `answer_question`, map `DISTILL_PROMPT_V<n>` over the
  memories that will enter the context (message/fts/episode content; skip
  `graph_fact` lines — already atomic), then pass the kept lines through to
  `answer_question` as a new additive block rendered above the memories
  (mirror how `aggregation_nodes` render as a non-competing block — that
  crowding lesson cost KU −9.0pp once already).
- Gating (a COST control, not a quality filter, and label-free): distill fires
  when the ability in use ∈ {MR, TR} OR the rendered hit count ≥ 12;
  otherwise the question runs untouched. Record fired/not per question.
- Instrumentation into `per_question`: `distill_fired`, `distill_calls`,
  `distill_kept`, plus token counts into metadata.

### 2.4 Full LME A/B (one run; baseline is banked)

- ON arm: `--sample 0 --seed 0 --distill`, canonical reader (deepseek-chat)
  and canonical config (copied from the canonical run metadata, as in 1.3).
- OFF arm: the banked canonical 70.0 run itself (same seed → paired). Only
  if any canonical-config flag has drifted since, rerun OFF first.

**Gate G-P1b (ship signal):**
- Primary: **MS strict ≥ +5pp** vs paired baseline (above the ±5pp category
  noise band; MS floor is 51.9).
- Secondary (all must hold): overall ≥ baseline − 1pp; no category worse than
  −5pp; the broken-out abstention slice within noise (distillation must not
  convert honest "I don't know" into confabulation — check `*_abs` rows
  explicitly).
- Diagnostics to bank either way: flips among the 2.1 qids (did the mechanism
  hit its named targets, or did unrelated questions move — mechanism > score),
  D2/tally deltas, distill cost per question.

### 2.5 Optional third run — P1 under the parity reader

Run ONLY if G-P1b passed AND Phase 1 showed reader-dominance (parity reader
≥ ~80): repeat the ON arm with the 1.1 flags pointing at the parity reader.
This is the honest Hindsight comparison (their number = strong reader + reflect
loop; this run = strong reader + bounded reflect) and tells whether
distillation's value survives a reader that can already spot needles. Bank all
four cells that now exist (reader × distill) in the roadmap.

### 2.6 Productization into `HyMem.ask()` (only after G-P1b passes)

`hymem/query/ask.py` is the natural home — its own docstring concedes
synthesis is the bottleneck, and `ASK_PROMPT_V1` is already the versioned
single-call contract. Additive, config-gated, Protocol-only:

- Config (`hymem/config.py`): `ask_distill_enabled: bool = False` (stays
  False until a Hermes cost read), `ask_distill_min_hits: int = 12`,
  `ask_distill_max_calls: int = 24` (hard cap on map fan-out).
- `DISTILL_PROMPT_V<n>` moves to `ask.py` as a versioned constant (same
  version-bump discipline as `ASK_PROMPT_V1` / the fusion salts).
- Flow in `ask()`: when enabled and the hit count clears the floor, map over
  `message_hits`/`fts_hits`/`episodes` snippets (respecting
  `_SNIPPET_CHARS`), drop `NONE`s, render kept lines as a
  `=== DISTILLED EVIDENCE ===` section ABOVE `CONVERSATION EVIDENCE` in
  `render_context` ordering; the raw tiers stay untouched below it. The
  existing `_truncate_block` budget still applies to the whole.
- N+1 LLM calls per `ask()` when it fires — log `distill_calls`/`kept` at
  debug, and expose call count on `Answer` so hosts can meter it.
- Tests (StubLLMClient, mirroring the existing ask tests): fires only above
  the hit floor; respects `max_calls`; `NONE` filtering; additive rendering
  order; disabled by default; truncation interaction; no distill call when
  the store is empty.
- The no-shipped-backend rule holds throughout: everything speaks the
  `LLMClient` Protocol.

**Deliverables:** dry-run verdict + A/B numbers banked in
`longmemeval_roadmap.md` P1; on pass, `ask()` distillation landed default-OFF
with tests; a Hermes cost read (tokens/latency per fired `ask()`) filed
before any default flip.

---

## Track A (parallel) — Idea A multi-hop traversal at `max_hops=2`

Fully sketched in `additional_planning.md` §Idea A (code, config knobs,
risks). It touches only `_graph_lookup` in `query/augment.py` — no episodes,
no dream path — so it may proceed during Phase 0's watch and alongside
Phases 1–2. Its value is production/Hermes recall (LME retrieval is CLOSED);
judge it by the bridging-edge probe ONLY. Sequence, with gates:

1. **Probe before code:** synthetic bridging-chain pytest (seeded DB with
   known 2-hop chains, e.g. `atta —part_of→ medflow —deploys_to→ fly.io`)
   PLUS a mined LME/BEAM multi-hop slice with hand-labeled bridging edges
   (~60–100 items total, per the sketch). The probe adapts
   `benchmarks/gold_rank_probe.py`.
2. **Build:** `_multihop_edges` + Source-4 wiring + the four `graph_multihop_*`
   config knobs exactly as sketched (default OFF; seeds = direct entity
   matches only; hop≥2 edges only; honest `fallback:multihop:{n}hop` reason
   codes).
3. **Gate G-A1 (A/B on the same DB, off→on):** bridging-edge recall@8 rises
   on the multi-hop set; recall@8 on the 1-hop control set does NOT drop
   (the additive invariant as a metric); p95 `augment()` latency < 1.5×
   baseline.
4. **Sweep** `max_hops ∈ {2,3}` × `decay ∈ {0.4,0.5,0.6}` ×
   `min_score ∈ {0.02,0.05,0.1}` against the probe, pick the Pareto knee
   (`max_hops=2` is the expected ship).
5. **Gate G-A2:** one full LME guard run as NON-regression only (vs 70.0, MS
   floor 51.9) — never a tuning signal.
6. Ship default stays `False`; enable for Hermes via config once G-A1/G-A2
   hold.

---

## Sequencing and run budget

| # | Item | Depends on | LLM cost | Gate |
|---|------|-----------|----------|------|
| 0.1–0.3 | Flip-watch read + verdict | bb96057 deployed (done) | none | G-FLIP |
| 0.4 | Flip + Plan C unblock | G-FLIP pass | none | post-flip dream check |
| 1.1 | Adapter `--answer-base-url` | — | none | smoke `--sample 2` |
| 1.3 | P0 parity run | 1.1 | 1 full-500 run | none (measurement) |
| 2.1–2.2 | Synthesis-miss recovery + dry-run | box artifacts | ~40 small Qs | G-P1a |
| 2.3–2.4 | `--distill` + LME A/B | G-P1a pass; 1.3 done (for reading) | 1 full-500 run | G-P1b |
| 2.5 | Parity×distill cell | G-P1b pass + reader-dominant P0 | 1 full-500 run (optional) | — |
| 2.6 | `ask()` productization | G-P1b pass | none (Stub tests) | Hermes cost read |
| A | Idea A probe→build→sweep | none (parallel) | probe-only | G-A1, G-A2 |

Hard rules carried from the banked record: never suppress-filter on a routed
ability (additive-MR invariant); no LME A/Bs on aggregation variants; any
material prompt change bumps its version constant; judge posture is frozen;
Plan C only after the flip verdict AND never overlapping a reuse-verification
dream.
