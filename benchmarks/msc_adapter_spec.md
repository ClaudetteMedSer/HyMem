# MSC (Multi-Session Chat) adapter — design spec

**Why.** LongMemEval is single-shot / star-topology: within a question's haystack,
a fact appears in one place and never recurs across sessions. That one property is
what makes three things **untestable** today — repetition-gated rule promotion (E3
died at 5% recall because ~nothing recurs), the `suggest_rules()` recurrence signal
(`session_count` has nothing to separate), and Track-A multi-hop (star topology →
~0 genuine bridges, see [[project_track_a_multihop]]). MSC has genuine cross-session
structure: two speakers chat over N sessions and their personas *accumulate and
restate*. An MSC adapter turns those three bets into measurable numbers.

This spec mirrors `longmemeval_adapter.py` section-for-section so the two adapters
share machinery and conventions; it **reuses** that file's proven parts by import
rather than reimplementing them.

> **STATUS 2026-07-28 — BUILT (`benchmarks/msc_adapter.py`, ~430 lines).** Data
> contract verified against the real MemGPT/MSC-Self-Instruct schema (§1). Both
> probe modes implemented; `--sim` mechanics green (loader, date synthesis,
> gold-session location, marker-dump labeling all validated offline). Real numbers
> need the box. **Key finding the build already surfaced:** MSC's durable content is
> *preference/fact-shaped* ("I have two dogs") → `preference`-kind markers →
> excluded from rules by `is_rule_eligible_kind`, so the recurrence/E3 path is
> **largely inert on MSC for the RULES tier** — MSC exercises the *profile* tier's
> cross-session accumulation, not rule durability. So MSC's strong, honest fit is
> **`recall`** (cross-session fact retrieval — the capability LME can't isolate);
> validating repetition-gating *for rules* still needs a corpus of recurring
> *imperatives*, which no available benchmark provides. The `recurrence` mode prints
> this explicitly (a NOTE when 0 rule-eligible markers appear).

> **STATUS 2026-07-28 (later) — FIRST RUN + PARITY FIXES.** First real `recall` run
> (100 samples, v4-flash answer+judge): **42.0%**, E1 decay 54% at 1-back → 33-40%
> at 2-4-back. Analysis showed the number was a FLOOR, not an architecture verdict —
> the MSC answer path was not at parity with LME's: (1) no `top_k*3` at the pipeline
> layer (the LME driver's multiplier — the same silent drop as the BEAM June
> regression); (2) `search()` cut `[:10]` after 15 message_hits, so consolidated
> tiers NEVER reached the reader; (3) the P4 `user_profile` tier — the tier MSC
> content is shaped for — plus episodes/graph/procedures/aggregation nodes were
> unread. **All fixed in the adapter:** LME-parity tier collection + ordering,
> `top_k*3`, profile prepended ADDITIVELY (never consumes a raw-turn slot),
> aggregation nodes/temporal events/graph_count passed through to
> `answer_question`, `--embeddings` fallbacks now import the LME `LOCAL_EMBED_*`
> constants (same local server posture), and `--dream-per-session` implemented.
> **Step-0 diagnostics added:** per-question `gold_in_context` / `gold_in_pool`
> (lexical, τ=0.6) with a per-distance table and a miss decomposition
> (retrieval vs ranking/cut vs synthesis/judge) — this split decides Step 2.
> Core (`hymem/`) deliberately untouched pending the parity re-run.

> **STATUS 2026-07-28 (parity re-run) — 65.0% (+23pp; the 42% was harness).**
> Same 100 samples/seed 0/frozen posture. E1 flattened (69.2/70.0/55.6/66.1 at
> 1/2/3/4-back; gold-in-ctx 100/100/89/89%); profile tier populated (6.2/q, zero
> empty). Miss decomposition (35): **synthesis/judge 24 (69%)**, retrieval 9
> (26%, all at 3-4-back — the FTS-paraphrase tail), ranking/cut 2 (6%). At n=100
> (σ≈4.8pp) 65.0 is statistically indistinguishable from LME's 68.4 — the
> "cross-session recall is harder" read is retracted. Next: audit the 24
> synthesis misses (recall `--out` + `--dump-context` added for exactly this),
> then a gpt-oss-120b reader-parity probe (LME P0 method, judge frozen). Prompt
> fixes deliberately deferred until the audit says which failure mode dominates.

---

## 1. Data contract (VERIFIED 2026-07-28 against MemGPT/MSC-Self-Instruct)

The concrete, downloadable artifact is **MemGPT/MSC-Self-Instruct** (HuggingFace,
500 rows) — a QA derivative of the ParlAI MSC data with clean recall labels. Actual
per-row schema (confirmed via the HF datasets-server `first-rows` API):

```
example = {
  "previous_dialogs": [ { "dialog": [ {"text": str}, ... ],   # the multi-session
                          "personas": [ [str], [str] ],       #   history to ingest
                          "time_num": int, "time_unit": str,  #   inter-session gap
                          "time_back": str }, ... ],
  "self_instruct": { "B": <question>, "A": <gold answer> },   # the recall probe
  "personas": [ [str], [str] ],                               # current-session personas
  "init_personas": [ [str], [str] ],                          # session-1 personas
  "personas_update1": [str, ...], "personas_update2": [str, ...],   # evolving snapshots
  "dialog": [ {text, id, convai2_id, rating}, ... ],          # the CURRENT session
  "metadata": { "initial_data_id": str, "session_id": int },
}
```

Findings that shaped the build (deltas from the original spec assumption):
- **Turns carry only `text`** inside `previous_dialogs[i].dialog` — no speaker field.
  Speakers alternate, so the loader assigns roles by index parity (`--start-role`).
- **Persona snapshots are CUMULATIVE**, not per-session deltas — so "present in
  `personas_update1`" means "known since session 2", NOT "restated in session 2".
  Genuine restatement is therefore derived by lexical matching each persona fact
  back to the turns of each session (`_lex_match`), the §4 approach — not by
  snapshot diffing.
- **Real relative gaps exist** (`time_num`/`time_unit`, e.g. 5 "hours"), so session
  dates are synthesized from them (monotonic, ordering-preserving) rather than a
  fixed `--session-gap-days` (kept as the fallback).
- **`self_instruct` is ready-made recall QA** — no probe generation needed for
  `recall` mode; `.B` is the question, `.A` the gold.

`load_msc_data(path, sample, seed)` mirrors `load_longmemeval_data`: JSON/JSONL in,
sampled+seeded **normalized** examples out (`{id, sessions, session_dates, question,
answer, persona_facts, n_sessions}`) so downstream code never sees raw field drift.
Still to confirm before a box run: license/redistribution for committing a sampled
slice to `hymem_beam/data/`.

---

## 2. MSC → HyMem mapping (mirror `HyMemAdapter`)

`MSCAdapter(db_path, …)` mirrors `HyMemAdapter` — isolated temp DB per dialogue,
same `open()` → `HyMemConfig(root=…, **overrides)` + `OpenAICompatibleClient`, same
`--keep-db`, same `fork()`-based `dream_and_wait`. Two MSC-specific rules:

- **One HyMem session per MSC session — never merge.** LME chunks at 50 msgs/session
  (`f"{sess_id}_{i//50}"`); MSC sessions are ~10–14 turns, so keep exactly one
  `log_messages(dialogue_id + "_s{i}", turns)` call per MSC session. `session_count`
  in `suggest_rules()`/policy metrics is only meaningful if session boundaries are
  preserved — **this is the whole point of the corpus.**
- **Synthesize monotonic session dates.** MSC has no timestamps; pass each session a
  `created_at` spaced by `--session-gap-days` (default 7) so the recency levers and
  bi-temporal supersession (`valid_at`/`invalid_at`) see real temporal separation,
  exactly as LME's `session_date` feeds `created_at`.

Ingest one speaker's view per dialogue (default **A**; `--speaker`): HyMem is a
single-user memory, so we model speaker A's memory of the conversation. B's turns are
ingested as the interlocutor (role `assistant`), A's as `user` — matching how HyMem
extracts *the user's* markers/profile.

---

## 3. The pipeline (mirror `evaluate_question` / `_evaluate_one_question`)

Per dialogue, in an isolated DB, parallelizable over `--workers`:
1. `ingest_sessions(sessions, dates)` — §2 mapping.
2. `dream_and_wait()` — one dream after the last session (or `--dream-per-session` to
   dream after each, which better mimics a live store and lets `pos_evidence`
   accumulate across dreams; default single-dream for speed, flag to compare).
3. derive/attach **probes** (§4).
4. per probe: `search`/`answer`/`judge` (recall & adherence modes) or a pure
   structural read (recurrence mode).
5. `compute_scores` per mode, printed with the LME report scaffolding.

---

## 4. Probe derivation — the ONE decision that needs sign-off

MSC is natively a *generation* benchmark, so an adapter must define what it probes.
Three modes, `--probe-mode`:

**A. `recall` (mirrors an LME accuracy number).** From persona facts, auto-generate a
question per fact ("What does the user … {fact predicate}?") asked *after all
sessions are ingested*. Judge with the reused `judge_answer`. Tests cross-session
fact/preference accumulation — the IE/MR analogue, but genuinely multi-session.

**B. `recurrence` (THE de-blocker — cheap, no answerer/judge).** For each persona fact
labeled with the sessions it's stated in (per-session annotations, §1.1), compare the
ground-truth stated-in-N-sessions against HyMem's own `suggest_rules().session_count`
for the matching canonical policy. Metric: does `session_count` separate
stated-once from stated-many (AUC / rank-correlation)? **This is the exact test E3
couldn't run on LME** — it validates (or kills) the recurrence signal that
repetition-gating and the suggestion ranking both rest on. Needs no LLM answerer/judge
(only the tagger), so it's the cheapest and most decisive experiment.

**C. `adherence` (read-side rules).** Take a persona preference, inject it as a rule
via `add_rule`, and measure whether a held-out probe response respects it ON vs OFF —
the standardized, multi-session extension of the bespoke `rules_compliance.py`.

**Fallback if §1.1 per-session annotations are absent:** derive recurrence by
embedding-matching each accumulated persona fact back to each session's turns
(fact "appears" in session i if max cosine to a turn ≥ τ). Weaker ground truth —
label it as heuristic-derived in the output so results are read with appropriate
caution — but it unblocks E2/E3 without hand-labeling.

> **Recommendation: build B first, then A, then C.** B is the reason MSC is worth
> adding (it de-blocks Idea B repetition-gating + the suggestion signal), needs no
> answer/judge LLM, and is a small amount of code. A and C reuse the LME answer/judge
> path almost verbatim. **Your call on the default `--probe-mode` and whether to
> assume §1.1 annotations or start on the embedding-match fallback.**

---

## 5. Experiments (each verifies a specific open decision)

| # | Experiment | Mode | Verifies |
|---|-----------|------|----------|
| **E1** | Cross-session recall vs session distance | recall | HyMem retrieves a session-1 fact when probed after session N (the capability LME's star topology can't isolate) |
| **E2** | Recurrence-signal validity | recurrence | Does `session_count` separate durable from one-off? → **unblocks repetition-gating / schema v24 and the `suggest_rules` ranking** |
| **E3** | Repetition-gated promotion precision | recurrence | Re-run `rule_extraction_experiment.py` policy layer on REAL cross-session recurrence — the run that died at 5% recall on LME |
| **E4** | Multi-hop bridges | recall + `--graph-multihop` | Do genuine persona chains (A hikes → A near mountains) give multi-hop non-zero lift where the star gave ~0? |
| **E5** | Rule adherence, multi-session | adherence + `--rules`/`--no-rules` | Does an injected persona rule shift responses ON vs OFF, at conversation scale |

E3 literally feeds MSC-derived `(kind, statement, session_id, is_rule)` into the
existing experiment engine with `--policy-from-canonical` — no new scoring code, and
it's the honest retry of the corpus-artifact result in [[project_idea_b_rules]].

---

## 6. CLI surface (mirror LME flags; add MSC-specific)

Reused verbatim: `--sample --seed --workers --answer-model --answer-base-url
--answer-api-key --judge-model --data-dir --keep-db --no-dream --embeddings
--graph-multihop --rules/--no-rules --rules-extraction --value-supersession`.

MSC-specific:
```
--probe-mode {recall,recurrence,adherence}   default: recurrence
--speaker {A,B}            default A       (whose memory we model)
--session-gap-days N       default 7       (synthetic inter-session spacing)
--dream-per-session        flag            (dream after each session, not just last)
--recurrence-tau FLOAT     default 0.75    (tagger τ for the recurrence probe)
--annotations {persona,embedding-match}     default persona (falls back per §4)
```

---

## 7. Build plan / file skeleton (`benchmarks/msc_adapter.py`)

Reuse from `longmemeval_adapter.py` by import (don't reimplement): `LLMClient`,
`OpenAICompatibleClient` wiring, `judge_answer`/`get_judge_prompt`, `answer_question`,
`compute_scores`, the report printers.

New:
```
load_msc_data(path, sample, seed)                    # §1 loader + normalizer
class MSCAdapter(HyMemAdapter-shaped)                # §2: open/ingest/dream/search
derive_probes(dialogue, mode, annotations)           # §4: the three modes + fallback
score_recurrence(candidates, gold_sessions)          # §4B: AUC / rank-corr
evaluate_dialogue(dialogue, args, answer_llm, judge) # §3 per-dialogue driver
main()                                               # mirrors LME main(): sample→
                                                     #   pool workers→ per-mode report
```

Est. ~500–700 lines (vs LME's 2510 — MSC needs no oracle/floor/distill/rejudge
machinery). Offline `--sim` path (SimJudge-style fake answerer) so mechanics run with
no API, matching the experiment engine's offline discipline.

---

## 8. Risks / honest caveats

- **§1.1 is load-bearing.** No per-session persona annotations → E2/E3 fall back to
  heuristic recurrence (weaker, labeled as such). Confirm before promising E3 numbers.
- **Single-user modeling.** HyMem models one speaker; cross-speaker reasoning ("what
  did B say about X") is out of scope by design (matches the product).
- **Synthetic dates are a modeling choice**, not ground truth — fine for exercising
  recency/supersession mechanics, not for absolute temporal-accuracy claims.
- **Comparability.** Keep the answerer/judge posture frozen to the LME baseline
  (deepseek-v4-flash answerer, gpt-oss-120b judge) so MSC and LME numbers sit in the
  same frame; MSC is an ADDITIVE probe, not a replacement.
```
