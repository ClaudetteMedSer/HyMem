# Pre-registration: BEAM gold-delta rejudge (B−A) — 2026-08-31 (v0.2)

Status: **EXECUTED 2026-08-31** — amendments incorporated, run complete,
readout in §8. Branch: Beam-optimisation.
Supersedes the draft "three-arm + pin" pre-registration (v0.1, unexecutable: no
`--answer-extra-body` / `--judge-extra-body` on beam_adapter — amendment (c)).
Supersedes v0.1.1 §5 swap request (t=0.5 fixed cutoff — amendment (d)).
This is the resequenced Phase 1: the gold delta, alone, via a judge-only rejudge
of the existing anchor.

## 1. Question (what this decides)

Does scoring the June-comparable anchor with the **real gold** (GOLD_FIELDS)
instead of the **empty legacy ideal** change the scores materially?

- **Falsifier (H0):** pooled B−A on the 8 affected abilities is within the
  no-op band. Then the June record (OVERALL 54.2 = the 16:50Z artifact's
  0.5416) stands as-is and the empty-ideal defect cost nothing measurable.
- **Reject H0:** pooled B−A outside the band → the record's meaning changed →
  rebasing is required (fresh gold-on baseline; old canonical retires as a
  comparison point, never read as regression/improvement).

## 2. Design (accepted amendments)

- **A** = `results_20260831T165039Z.json` (16:50Z sized run, `judge_gold` off,
  judge=deepseek-chat). **B** = judge-only rejudge of A with gold on.
- Answerer output **byte-fixed** (stored row `answer`), model byte-fixed at the
  anchor's actual model (deepseek-chat both sides — the anchor's JUDGE was
  deepseek-chat, not the pinned v4-flash; pinning is Phase 2).
- Judge path: `judge_answer` unchanged (temp 0.0, max_tokens 512, same regex
  parse) — B differs from A in **one variable only**: the IDEAL ANSWER field.
- **No** answer calls, **no** ingest/dream, **no** extra_body plumbing, **no**
  `_call` change (the falsy-content raise is Phase 2), **no** competitor-table
  read, **no** score-table read other than the falsifier.
- Review amendments incorporated: (d1) continuous = primary, binarized =
  companion; (d2) t chosen from A's marginal between mass points, not fixed
  0.5; (d3) B-side parse-failure = ABORT (void), not keep-prior; (d4) ABS/CR
  promoted from band to GATE.

## 3. Verified facts (evidence, 2026-08-31)

- Rows live under `conversations[].questions`: 8 convs × 20 = 160 rows.
  Per-row keys: ability(short), question, answer, ideal_answer, rubric(list),
  score(float), scores(list), gold_kind, probe. No `question_id`, no `gold_text`
  — gold comes from a fresh dataset parse, matched on `(ability, question)`.
- Uniqueness: 0 duplicate `(ability, question)` pairs in-artifact and 0 across
  convs → the match key is unambiguous.
- Reparse determinism: 160/160 rows identical (question/ideal_answer/rubric/
  gold_kind/ability); 0/128 pool golds empty after reparse.
- ABS/CR controls: gold_text == stored ideal_answer on 32/32 → identical
  prompts, pure judge-churn control arm (n=32, no built-in churn assumption).
- Anchor parse-clean: 0/160 rows with the silent-0 signature ("non-empty
  rubric + scores==[]"); 0/160 empty rubrics. (A-side rate = 0/160.)
- gold_kind by ability: response on ABS/CR/EO/IE/KU/MR/TR, compliance_spec on
  IF/PF, summary on SUM — all 16 rows per ability.
- Anchor metadata verified: judge_model=deepseek-chat, answer_model=deepseek-chat,
  sample=8, scales=[100K], top_k=10, judge_calls=160/answer_calls=160,
  elapsed_s=5471.6, date=2026-08-31T16:50:39.701802+00:00. **metadata does NOT
  record judge_gold** — A's gold-off status rests on the flag default
  (judge_gold=False, :1251-1256). B MUST record `judge_gold: true` plus
  `a_date` and `gap_hours` (the A→B exposure window — its own alias-drift
  footprint, see §5 gate).
- A per-ability values (pre-read, for the falsifier table only): ABS .7500,
  CR .4219, EO .2944, IE .3542, IF .6875, KU .5625, MR .5042, PF .8750,
  SUM .3724, TR .5938; OVERALL .54158 (= 54.2, the record-comparison number).
- **A pool score marginal (pool n=128): 0.0×45, 0.125×1, 0.2×3, 0.25×3,
  0.333×3, 0.4×2, 0.5×5, 0.5556×1, 0.6×5, 0.6667×2, 0.75×1, 0.8×2, 0.8889×1,
  1.0×54 — every value = k/n for its own n (0 non-k/n); rubric lengths pool
  {1:46, 2:37, 3:9, 4:7, 5:22, 6:4, 8:1, 9:2}; control {1:16, 4:16}.**
- **Binarization cutoff (amendment d2), chosen from A's marginal BEFORE B:**
  0.5 is a mass point (5/128 rows; even-length rubrics). Semantic: "satisfied"
  = at least half the rubric items (the judge's own construct — mean over 0–1
  items). t = (m⁻ + 0.5)/2 = (0.4 + 0.5)/2 = **0.45** → the 5 atomic rows are
  satisfied in A; no score is a k/n for any n ≤ 9 at 0.45, so the cutoff cannot
  land on a mass point. Alternative (strict majority > 0.5): t = (0.5 +
  0.5556)/2 = 0.52778 → those 5 rows unsatisfied. Only the companion's D/net
  changes; the continuous primary is unaffected. **Pre-registered: t = 0.45.**
- LME `_rejudge_run` port source: :2300–2399 (keep-prior-on-judge-error,
  `_rejudged` flag, `-rejudged-<model>-<stamp>.json`, stamps rejudged_from /
  rejudge_original_judge / judge_model). LME guardrail :2327–2328
  (v4-flash-without-extra_body warning) ported as an ABORT in rejudge mode
  (no extra_body mechanism here: bare shallow-v4-flash as judge → refuse).
- REJUDGE GUARD: reparse must reproduce 160/160 stored rows (ability,
  question, ideal_answer, rubric) or the run aborts — dataset drift cannot
  silently change the gold.
- `judge_answer`: `total = sum(scores)/len(scores)` (:1200), rubric prompt
  "score 0-1 each" (:1189); `re.search(r'{...}', ...)` (:1196) cannot match
  nested JSON; no-rubric (:1180) and except (:1204) paths return the SAME
  `{"score": 0.0, "scores": []}` — silent-0, indistinguishable from real 0.0
  in the score column (raw is the only discriminator, which is why the
  rejudge captures it).
- Message-construction byte-equality snapshot taken 2026-08-31 BEFORE any
  code change (stub LLM on a real KU row + real control row, exact current
  construction, 812 bytes) → /tmp/judge_messages_snapshot.json. The
  return_raw test asserts byte-equality against this snapshot (see §4.5).
- Registry `path.stem[:16]` bug is GENERAL — three registries: lme :184,
  beam :139, locomo :114. Every non-DOC row carries a truncated stem:
  `longmemeval-v2-h` (×all LME), `beam-v14-prefere` / `beam-v15-mr-tr-f` /
  `beam-v16-mr-tr-f`, `locomo_conv26_di`. Rejudge row id=41 additionally
  inherits source run_date + total_tokens/elapsed_s (1718606 / 8404.22
  byte-identical to id=42).

## 4. Procedure

1. **Canary (pre-registered, hard-fail):** one call via the EXACT rejudge
   client path (model=deepseek-chat, temp 0.0, max_tokens=512 — the judge
   path's own ceiling, so a thinking-alias regression reproduces the
   `content == ""` / finish=length shape rather than a false-positive from a
   tiny token budget): assert `content.strip()` non-empty and no `[LLM_ERROR`
   prefix; on violation print the raw repr, exit 1. **Asserts non-empty, not
   non-null** — the trap shape. (The pin-phase v4-flash canary is a separate,
   later item; this canary protects the deepseek-chat alias map used by B.)
2. Load artifact A.
3. Re-parse dataset (`load_beam_conversations(["100K"], 8)`); run the 160/160
   reproduction guard (above). Print A→B gap: gap_hours = (B start − A
   metadata date 2026-08-31T16:50:39.701802+00:00) — stored in B metadata.
4. Match each row to its parsed question via `(ability, question)`; ideal =
   `gold_text` (pool + controls). For the control rows the ideal equals the
   stored legacy ideal by construction.
5. **Refactor + byte-equality test (executed before the rejudge run):**
   extract message assembly into `_judge_messages(question, ideal, rubric,
   ai_answer)` (verbatim move, no reordering); `judge_answer` gains
   `return_raw: bool = False` (positional/keyword unchanged for existing
   callers; message bytes identical. Test
   `tests/test_judge_messages_byte_identical.py`: for the SAME fixed inputs
   used for the snapshot, assert `_judge_messages(...)` == snapshot messages
   (deep equality AND json-dump byte equality). Existing callers unchanged.
6. Judge all 160 rows via `judge_answer(..., return_raw=True)`; per row store
   score, scores, judge_raw, judge_error.
7. **Error classes — two, with different consequences:**
   (a) **Silent-0 parse-failure** (signature: non-empty rubric AND
   `scores == []` — covers both no-rubric-return and except path): ABORT the
   run, print row ids, exit 2, write no verdict artifact. B's rate must be
   ≤ A's (0/160) per amendment (d3) — a clean anchor does not transfer when
   the arm changes the prompt; a rising rate would read as "gold lowers
   scores" in the exact direction the experiment believes.
   (b) **Explicit error** (raw empty / `[LLM_ERROR` prefix — noisy, not
   stealthy): keep the anchor score, `_rejudged=False`, exclude from δ̄ and D,
   count and print (mirror LME :2338-2349). Ceiling: >5% of the pool (7/128)
   → run INVALID for the falsifier; report, don't interpret.
8. Output: `results_20260831T165039Z-rejudged-deepseek-chat-<stamp>.json`
   next to A. Metadata: `date` = rejudge exec time (NOT source's),
   `elapsed_s` = own, `answer_calls` = 0, `judge_calls` = own, `a_date` (A's
   metadata date, verbatim), `gap_hours`, `judge_gold: true`,
   `rejudged_from`, `rejudge_original_judge` ("deepseek-chat"). Per row:
   `score` (new), `score_original` (A's), `scores`, `judge_raw`,
   `judge_error`, `_rejudged`. No registry insertion (beam pattern is
   beam-*.json; read from artifact).

## 5. Read protocol (fixed before counts — never re-ranked post-hoc)

- **Pool = 8 affected abilities** (EO, IE, IF, KU, MR, PF, SUM, TR), n=128.
  ABS/CR = **control arm**, n=32 — the gate, not merely a band.
- **PRIMARY (continuous — the quantity under test IS a continuous mean;
  0.5416, not a pass rate; the control supports a per-row SD far better than
  a flip rate — a 0-flip rule-of-three upper bound ≈ 9.4% would be too wide
  to falsify anything):**
  δ̄ = mean(B − A) over the 128 pool rows; SE = SD_ctl/√128 where SD_ctl = SD
  of the 32 control-arm deltas (byte-identical prompts → pure judge churn on
  the continuous scale; a fortiori the H0 estimate, uncontaminated by the
  gold-on change). Band = 2·SE. **Inside → H0 holds; outside → meaning
  changed.** δ̄_pp = δ̄×100; report numerically.
- **COMPANION (binarized, catches atom-level verdict flips the mean can
  smooth over):** t = 0.45 (pre-registered from A's marginal, §3). D =
  #(t_B ≠ t_A); gained/lost; net_q = gained − lost. Band = 2√D (binomial,
  LME closed form). net=+3q@D=43 ↔ ±2.62pp precedent; at n=128 one flip ≈
  0.78pp → knife-edge; same-form band per LME guard practice.
- **Verdict (OR):** record-needs-rebasing ⇔ primary outside its band OR
  companion outside its band. Both inside → record stands. STATED REASON,
  because unwritten it reads as an accident: OR'ing two tests inflates the
  false-positive rate toward 2α, which is CONSERVATIVE in the direction of
  declaring a rebase needed — and that asymmetry is the correct one, because
  wrongly trusting a contaminated record costs more than an unnecessary
  rebase.
- **GATE (amendments d4):** before any verdict, check the control arm:
  |δ̄_ctl| > 2·SD_ctl/√32 → VOID. ABS/CR sends byte-identical prompts in both
  arms; any control delta is judge churn OR alias drift (deepseek-chat →
  v4-flash non-thinking is a server-side map; no model-identity field in the
  response can catch it) with no way to separate them — so a control delta
  outside its own band makes B−A UNATTRIBUTABLE, void regardless of the 128
  rows. Degenerate SD_ctl == 0: fall back to flip gate (any control flip →
  void) — deterministic judge means even one flip is movement. Report
  D_ctl/(flips, count + rule-of-three 9.4% bound as context) descriptively.
  The A→B wall-clock gap (stored as gap_hours) is the exposure window for
  this drift.
- **Per-ability: descriptive only** (n=16; one flip = 6.25pp knife-edge).
  Table: ability, A, B, Δpp, flips, own D. IF/PF compliance_spec golds are
  in the pool per the 8-ability directive; the per-ability table allows
  reading them separately (spec-as-ideal is a different quantity than
  response-as-ideal — probe-side exclusion precedent).
- No top-up. The pool is fixed by the anchor; a larger n would be a fresh
  run = new baseline, not a refinement.

## 6. Registry fix (its own pre-registered change, same PR series — GENERAL)

Verified: `path.stem[:16]` is broken for **ALL** non-DOC stems in three
registries (rejudge row is where it surfaced, not what's wrong): lme :184,
beam :139, locomo :114 — `longmemeval-v2-h`, `beam-v14-prefere`,
`locomo_conv26_di` etc. Rejudge rows additionally inherit source run_date +
total_tokens + elapsed_s (id=41: 1718606 / 8404.22 byte-identical to source
id=42).
1. Shared helper `stem_source_date(stem) -> str | None`: first
   `\d{8}T\d{6}Z` stamp in the stem; **None** when the stem has no stamp
   (beam/locomo stems never do — `beam-v14-preference-fix.json`,
   `locomo_conv26_diag.json`; LME stems always do — `-20260602T134049Z-seed0`).
2. `kind=rejudge`: run_date = **last** stamp in the stem (the rejudge exec
   time); source_date = first stamp; total_tokens = NULL; elapsed_s = NULL
   (inherited-but-wrong is worse than missing).
3. Future `_rejudge_run` writes its own `date`/stats into the rejudge artifact
   (stops the inheritance at the source); the registry rule doubles as the
   safety net for the existing row.
4. Backfill: all THREE registries, ALL rows — recompute source_date by the
   helper; WHERE differs → UPDATE (lme: not-null, stamps always present;
   beam/locomo → NULL where no stamp). Verify against the file (read-back,
   not trust-the-write). Beam/locomo rows nobody suspected (e.g. id=3/4
   `beam-v1?-mr-tr-f`) are covered by the same sweep.

## 7. Cost & non-actions

- 160 judge calls, zero answer calls; ~5 min dataset reparse + ~10-20 min
  judge pass (160 × ~3-5 s). No HyMem, no dream, no store writes.
- NOT in this phase: extra_body plumbing, pin-phase v4-flash canary, model
  pin, arm A (alias delta), digit-filter/ABS pre-registered-out rules (already
  banked; KU/TR reuse rules unchanged), competitor table, beam_runs.db
  ingestion of results_*.json (pattern is beam-*.json; read from artifact).
- Registry fix §6 is its own pre-registered change; it does not block this
  run (B's readout is artifact-based).

## 8. Executed results (2026-08-31, run B)

Artifact: `/home/node/hymem_beam/results_20260831T165039Z-rejudged-deepseek-chat-20260831T200531Z.json`
(metadata: judge_gold=true, answer_calls=0, judge_calls=161 incl. canary,
gap_hours=3.17, a_date=2026-08-31T16:50:39Z, rejudged_from=results_20260831T165039Z.json).

- Canary OK (217 chars content on the exact judge path, temp 0.0, max_tokens 512).
- 0/160 silent-0; 0/160 explicit LLM_ERROR; 160/160 rejudged; 291s; 24/128 pool
  rows moved, 104 exact-zero deltas.
- **Control arm ABS/CR: 32/32 per-row identical (SD_ctl = 0.0000).** The judge is
  SCORE-DETERMINISTIC at temp 0.0 on byte-identical prompts — verified
  post-hot by 4/4 fresh re-judges reproducing B's stored scores and raw lengths
  (104/128 pool zeros corroborate). SD_ctl=0 is a measurement, not a bug.
- CONTROL GATE: |δ̄_ctl|=0 within any band; flip-gate fallback (SD=0): 0 flips
  → PASS. gap window 3.17 h, no alias drift detectable.
- PRIMARY (continuous): δ̄ = −0.582pp, band = ±0.000pp (deterministic judge →
  zero-width SE) → OUTSIDE.
- COMPANION (t=0.45): D=12 (3 gained, 9 lost), net=−6, band=2√12=6.93 → INSIDE.
- **VERDICT (OR): REBASE REQUIRED.** Without SD_ctl, the OR fires on the zero-
  width primary alone; per-ability deltas are heterogeneous (EO −7.57pp, SUM
  −4.37pp vs IF/KU +6.25pp) — meaning changed, and asymmetric per ability.
- per-ability (n=16): A→B — EO 29.44→21.88, IE 35.42→36.46, IF 68.75→75.00,
  KU 56.25→62.50, MR 50.42→47.29, PF 87.50→84.38, SUM 37.24→32.86, TR
  59.38→59.38; ABS 75.00, CR 42.19 unchanged. OVERALL 54.16 → 53.69 = −0.47pp.
- Productively: 5/128 A rows sat at exactly 0.5 (the t=0.5 mass point); 2 of
  them flipped in B (both MR 0.5→0.25). D is 12 at BOTH t=0.45 and t=0.5 on
  this sample — the tie rule was inert here, but the histogram rule chose t
  without conditioning on the outcome, which is the discipline that matters.
- Next per plan: §6 registry general fix, then model-pin pre-registration.
