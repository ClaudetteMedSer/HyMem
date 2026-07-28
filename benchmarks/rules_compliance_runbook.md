# Idea B (`always_on` Rules) — data-driven scorecard + box runbook

**Audience:** whoever runs the Hermes box. Idea B gives standing behavioral
imperatives ("always run the tests before pushing", "never suggest Docker") a
first-class home injected into every context call. To beat other memory systems
on this, the claim is backed by three measurable gates, not vibes:

| Gate | Question | Metric | Where it runs | Status |
|------|----------|--------|---------------|--------|
| **Adherence** | Does the model OBEY a rule in context? | ON>OFF compliance, ON≥0.8 | box (needs LLM) | **CLEARED** 2026-07-27 (judge gpt-oss-120b) → read side default ON |
| **Extraction precision** | Are auto-inferred rules CORRECT? | precision ≥ 0.90 | anywhere (deterministic) | **FAILED on real markers** 2026-07-27 — 8.3% (1 TP / 11 FP / 37 FN); lexical approach exhausted (~14% ceiling). Write side STAYS OFF; auto-extraction is now R&D. Built-in set still 100% (deterministic guard). |
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
regex.** The path to ≥0.90 is a semantic layer — either repetition-gated routing
(deterministic, needs cross-session recurrence + provenance) or an LLM `standing`
tag folded into marker extraction — tracked in `additional_planning.md` §Idea B.
The built-in probe stays 100% as a deterministic regression guard for the regex.

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
- **Honcho**: active rules lead the peer card + peer/session context
  representation, ahead of MEMORY.md (`hymem/honcho/app.py::_rules_block`).
- **Ask renderer**: rules render first in `ask()` context + an obey-directive is
  appended to the system prompt when rules are present.

## Decide
- Read side is ON. Leave it — inert on rule-less stores.
- **Write side stays OFF.** Gate 2 failed on real markers (8.3%); the lexical
  classifier can't reach ≥0.90. Do not flip, do not keep trimming the regex.
- **Ship the value that's ready:** read side + the told-path surfaces above
  (`add_rule` via API/MCP/Honcho) — explicit rules have no precision problem.
- **Auto-extraction is R&D** (repetition-gated vs LLM `standing` tag). It re-enters
  this gate only when a semantic layer exists; then re-run Gate 2 on real markers.
