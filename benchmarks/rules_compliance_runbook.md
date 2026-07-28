# Idea B (`always_on` Rules) — data-driven scorecard + box runbook

**Audience:** whoever runs the Hermes box. Idea B gives standing behavioral
imperatives ("always run the tests before pushing", "never suggest Docker") a
first-class home injected into every context call. To beat other memory systems
on this, the claim is backed by three measurable gates, not vibes:

| Gate | Question | Metric | Where it runs | Status |
|------|----------|--------|---------------|--------|
| **Adherence** | Does the model OBEY a rule in context? | ON>OFF compliance, ON≥0.8 | box (needs LLM) | **CLEARED** 2026-07-27 (judge gpt-oss-120b) → read side default ON |
| **Extraction precision** | Are auto-inferred rules CORRECT? | precision ≥ 0.90 | anywhere (deterministic) / box (LLM tagger) | **AUTO-INJECTION CLOSED 2026-07-28.** Lexical FAILED (8.3%, ~14% ceiling). The LLM durability tagger works where it matters (`rejection` 69% prec / 99% recall) but overall can't clear 0.90 — the drag is degenerate labels (`style` markers are durable directives *mislabeled* not-rule) + genuine one-off semantics; repetition-gating has no corpus to validate on (nothing recurs). **Superseded by the candidate-SUGGESTION surface** (`suggest_rules`): tagger proposes, human confirms = precision gate. Write side stays OFF. |
| _LME non-regression_ | Do auto-rules HURT factual recall? | ON ≥ OFF within variance | box (LME) | **MET** 2026-07-27 (`--rules-extraction` A/B = +1.4pp flat). Necessary, not sufficient — LME is blind to rule *correctness*, so it does NOT substitute for the precision gate. |
| **Overhead** | What does the tier COST per call? | 0 chars when empty; p95 Δ ≤ 1ms | anywhere (deterministic) | **PASS** locally (0 chars empty; 1138 chars / 0.034ms at 16 rules) |

Two switches, gated independently:
- `rules_enabled` (read side) — **ON** (adherence gate cleared). Inert until rules exist.
- `rules_extraction_enabled` (write side) — **OFF** until the extraction gate clears on *real* dream markers (below).

---

## Gate 1 — Adherence (LLM, box)
Does the answering model actually obey a rule that's in context? Mirrors the P4
profile box gate.
```bash
python -m pytest tests/test_rules.py -q            # mechanical prereq: expect all green
pip install 'hymem[server]'
export HYMEM_LLM_API_KEY=...                        # or DEEPSEEK_API_KEY / OPENAI_API_KEY
python benchmarks/rules_compliance.py \
    --answer-model deepseek-v4-flash \
    --judge-model  openai/gpt-oss-120b --judge-base-url <openrouter-url> \
    --verbose
```
> Use an INDEPENDENT judge (answerer ≠ judge). PASS = ON adherence ≥ 0.8 **and**
> ON > OFF **and** rule present in every ON / no OFF context. This cleared
> 2026-07-27 (answerer deepseek-v4-flash, judge gpt-oss-120b), so `rules_enabled`
> ships ON. Re-run only if the render/prompt wiring changes.

## Gate 2 — Extraction precision (deterministic; box validates on real markers)
`route_markers_to_rules` promotes the imperative sub-slice of behavioral markers
into `agent_inferred` rules during dreaming. A false positive becomes a rule
injected into EVERY call, so **precision is the gate** (recall is reported only).

The classifier is deterministic, so the probe runs anywhere:
```bash
python benchmarks/rule_extraction_probe.py          # built-in labeled set → PASS at 100%
```
**But the built-in set is hand-written.** Before flipping `rules_extraction_enabled`
ON, validate the classifier against the markers YOUR dream LLM actually produces:
```bash
# 1. Dream over real sessions, then dump the markers the LLM produced:
sqlite3 -json "$HYMEM_ROOT/hymem.sqlite" \
  "SELECT kind, statement FROM behavioral_markers" > markers.json
# 2. Hand-label each with "is_rule": true/false (is it a STANDING imperative,
#    or a one-off fact/preference?), then:
python benchmarks/rule_extraction_probe.py --labels markers.json --verbose
```
PASS (precision ≥ 0.90) on real markers → flip `rules_extraction_enabled` default
to `True` and record the precision + n here. FAIL → read the FP list.

**Real-marker result (2026-07-27): FAILED at 8.3% — lexical approach exhausted.**
Two rounds of classifier tightening (`rejects?`/`refuses?` removed, cutting FP
73→11) confirmed the ceiling is ~14%: the residual FPs are one-off *corrections*
carrying incidental modals ("X was Dutch **instead** of English", "…**should** be
automatic"). Standing-vs-one-off is a SEMANTIC distinction a word list can't make,
so further directive-vocabulary trimming is a dead end. **Do NOT keep tuning the
regex.** The built-in probe stays 100% as a deterministic regression guard.

**Semantic layer built + tested (2026-07-28) → AUTO-INJECTION CLOSED.** The LLM
durability tagger (`hymem/rules_extract.py`, the `llm` routing mode) is the
semantic instrument the lexical ceiling demanded. On real markers it is **7×**
lexical (59% vs 8% on the enriched set; ~45–73% on the natural-base-rate set), and
the per-KIND breakdown (`rule_extraction_experiment.py --by-kind`) localizes what's
left:

| kind | llm precision | recall | reading |
|------|--------------|--------|---------|
| `rejection` | **69%** | 99% | the signal-bearing kind — the tagger works |
| `style` | "0%" | — | **labeling artifact**: the markers ARE durable directives ("be concise", "class-level skills") *mislabeled* `is_rule=false`; they also duplicate the profile tier |
| `correction` | ~0% | — | genuinely one-off (2.4% FP rate) — tagger already conservative |

The FPs are **not concentrated in a kind** (so no `_RULE_KINDS` surgery helps) —
they're bad `style` labels plus `rejection`'s genuine one-offs ("rejects LoCoMo"),
which only *recurrence* could separate. But **repetition-gating has no corpus to
validate on**: the policy layer clears 0.90 only at ~3–5% recall (nothing recurs)
on LME (star topology), MSC (preference-shaped — see `msc_adapter_spec.md`), *and*
real Honcho session data. Of the three, only Honcho could show recurring
imperatives and it shows them sparse — a hypothesis (one real data point) that
standing rules are said once and expected to hold, so the sparsity is intrinsic.

**Decision: auto-*injection* is a dead-end on available data; the answer is
candidate SUGGESTION** — the tagger reliably *finds* standing directives (high
recall), so surface them and let the human confirm (which also makes the
profile-vs-rules *tier-placement* call a classifier can't). See `HyMem.suggest_rules`
/ `hymem_suggest_rules` below. Repetition-gating (schema v24) is not built —
gated behind a WATCH ITEM: if imperative recurrence turns dense as the Honcho store
grows, the E3 path reopens (`msc_adapter.py --probe-mode recurrence` dumps straight
into `rule_extraction_experiment.py`).

## Gate 3 — Overhead (deterministic, the cost watch)
Rules ride on the hot path, so the tier must be near-free.
```bash
python benchmarks/rules_overhead.py --reps 400
```
PASS = ON-but-empty adds 0 rendered chars (the default is free until `add_rule()`),
and a saturated rulebook (cap=16) stays within the p95 latency budget. Local:
1138 chars / 0.034ms p95 at 16 rules.

---

## Surfaces (already wired)
- **Direct/MCP**: `HyMem.add_rule()`/`rules()`/`retract_rule()`; MCP tools
  `hymem_add_rule` + `hymem_list_rules` (registered in `hymem/server.py`).
- **Candidate suggestion** (the write-side answer): `HyMem.suggest_rules()` + MCP
  `hymem_suggest_rules` — read-only, runs the durability tagger over unconsolidated
  markers, returns ranked de-duped `RuleCandidate`s (text / scope / confidence /
  marker & session counts / `already_active`) for a human to adopt via `add_rule`.
  Nothing auto-persists; the confirming human is the precision + tier-placement gate.
  Flow: log → `suggest_rules` → `add_rule` the good ones → dream.
- **Honcho**: active rules lead the peer card + peer/session context
  representation, ahead of MEMORY.md (`hymem/honcho/app.py::_rules_block`).
- **Ask renderer**: rules render first in `ask()` context + an obey-directive is
  appended to the system prompt when rules are present.

## Decide
- Read side is ON. Leave it — inert on rule-less stores.
- **Write-side auto-injection stays OFF — CLOSED as a dead-end on available data.**
  Lexical can't reach 0.90 (semantic); the LLM tagger works on `rejection` (69%)
  but the overall gate is blocked by mislabeled `style` + genuine one-offs, and
  repetition-gating has no corpus with enough recurrence to gate on. Do not flip,
  do not trim the regex, do not do `_RULE_KINDS` surgery.
- **Ship the value that's ready:** read side + told-path (`add_rule`) + the
  **candidate-suggestion** surface (`suggest_rules` / `hymem_suggest_rules`) — the
  tagger triages, the human is the gate. No 0.90 auto-gate to clear.
- **WATCH ITEM (the one live thread):** as the Honcho store grows, re-check whether
  standing imperatives recur across sessions (`msc_adapter.py --probe-mode
  recurrence` → `rule_extraction_experiment.py --policy-from-canonical`). Stays
  sparse → close repetition-gating + schema v24 permanently. Turns dense → the E3
  auto-promotion path reopens with real data.
