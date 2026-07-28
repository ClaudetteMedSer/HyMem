# Idea B write-side — auto-extraction A/B design & experiment plan

**Problem.** The deterministic lexical classifier caps at ~14% precision on real
dream markers (`rule_extraction_probe.py`, 2026-07-27): standing-vs-one-off is
*semantic*, not lexical. An auto-rule injects into **every** call, so precision is
the gate (≥ 0.90). This doc defines the candidate designs and the experiments
that pick the winner from data, not taste.

## The design space (A/B knobs)

| Knob | Values | Where | What it controls |
|------|--------|-------|------------------|
| **MODE** | `lexical` · `llm` · `llm_fastpath` | `config.rules_extraction_mode` → `rules_extract.route_decisions` | the routing instrument |
| **τ** (tau) | 0.5 … 0.9 | `config.rules_extraction_confidence_min` | LLM standing-confidence to mint |
| **PROMOTION** | `immediate` · `and` · `recurrence` · `or_highconf` | experiment (option A; needs schema v24 to ship) | repetition-gated promotion |
| **N** | 1 … 3 | experiment | min distinct sessions for recurrence |

- **`lexical`** — current classifier, no LLM. The baseline/control arm.
- **`llm`** — one batched durability call per dream decides; route iff `standing`
  and `confidence ≥ τ`. Returns a **canonical** rule form so paraphrases collapse
  to one `rules.text` (this is what makes `pos_evidence` a recurrence counter).
- **`llm_fastpath`** — trust a lexical imperative modal as standing (no call),
  send only the ambiguous rest to the LLM. Cheapest LLM arm.
- **PROMOTION** (per-policy, option A): `immediate` = mint on first sighting;
  `and` = classifier AND recurs ≥N sessions; `recurrence` = recurs ≥N (classifier
  ignored — rescues modal-less standing policies); `or_highconf` = high-confidence
  single shot OR recurrence.

## The engine

`benchmarks/rule_extraction_experiment.py` scores **every** arm against one
labeled corpus in a single judgment pass (one batched LLM call per mode; all τ/N
points derive analytically). `--sim` swaps in a deterministic fake judge so the
mechanics run offline; the box passes a real `--answer-model` for real numbers.

## Experiments & decision rules

Precision is gated at 0.90 everywhere; recall is reported. Pick the **simplest**
arm that clears the gate at acceptable recall (Occam: prefer `immediate` over a
promotion scheme, `llm` over `llm_fastpath`, unless the data earns the complexity).

**E1 — Instrument beats the baseline.** Does any LLM arm clear 0.90 where lexical
can't?
```bash
python benchmarks/rule_extraction_experiment.py --labels real_markers.json \
    --answer-model deepseek-v4-flash --answer-base-url <url>
```
Decision: if `llm` clears 0.90 and `lexical` doesn't → the semantic instrument is
justified. (Offline `--sim`: lexical 70%, llm 100% — mechanics confirmed.)

**E2 — τ sweep (precision/recall frontier).** Read the per-marker table: the
lowest τ that holds precision ≥ 0.90 maximizes recall; raise τ for margin against
judge drift. Decision: set `rules_extraction_confidence_min` to the elbow. Watch
for `llm_fastpath` **losing** to `llm` — the lexical shortcut re-imports the FP
leak (offline: fastpath 77% vs llm 100%). If so, `llm_fastpath` is out.

**E3 — Repetition value.** Does per-policy promotion beat `immediate`? Compare the
`immediate` row to the best `and`/`recurrence`/`or_highconf` row.
Decision:
- If `immediate` already clears 0.90 at good recall → **do NOT build repetition**
  (no schema v24). Ship `llm` + immediate.
- If the judge is noisy and `immediate` precision dips below 0.90 but `and`/`or_highconf`
  restores it → repetition is a **robustness** lever worth the schema cost. Pick
  the N with the best recall at precision ≥ 0.90.
```bash
# stress it: repetition should recover precision a noisy judge loses
python benchmarks/rule_extraction_experiment.py --sim --sim-noise 0.15
```

**E4 — Judge robustness (independent).** Re-run E1/E2 with a *different* judge
model (`--answer-model` = an independent LLM) and compare the chosen arm's
precision. Decision: the τ/mode choice must clear 0.90 under **both** judges, or
lower τ / add repetition until it does. Mirrors the adherence gate's
answerer≠judge discipline.

**E5 — Cost.** One batched call per dream (Phase-2), not per marker. Record tokens
in/out for the real run. Decision: `llm_fastpath` only earns its keep if E2 shows
it matches `llm` precision AND saves materially on tokens — otherwise `llm` wins
on simplicity.

**E6 — End-to-end non-regression (LME).** With the chosen arm wired, re-run the
LME write-side guard (`longmemeval_adapter.py --rules-extraction`, mode set in
config). Decision: overall must stay within the variance band vs baseline — auto-
rules must not poison factual recall. (The lexical arm already passed this at
+1.4pp; the LLM arm must re-clear it because it mints *more* rules.)

## From experiment to production

1. E1–E5 on real markers pick `(mode, τ)` and whether repetition is needed.
2. Wire the winner: set `rules_extraction_mode` / `rules_extraction_confidence_min`
   defaults. If repetition wins, add schema **v24** (a `status='provisional'`
   state: rules accumulate `pos_evidence` but stay out of `load_rules` until
   promoted at N), and a promotion step in `route_markers_to_rules`.
3. E6 gate, then flip `rules_extraction_enabled` default ON + commit + record the
   numbers in `additional_planning.md` §Idea B and the runbook scorecard.

Until then: `rules_extraction_enabled` stays **OFF**; the shipped value is the
read side (ON) + told-path surfaces. This is R&D, not a launch blocker.
