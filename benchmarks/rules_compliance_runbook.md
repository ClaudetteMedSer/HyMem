# Idea B (`always_on` Rules) — data-driven scorecard + box runbook

**Audience:** whoever runs the Hermes box. Idea B gives standing behavioral
imperatives ("always run the tests before pushing", "never suggest Docker") a
first-class home injected into every context call. To beat other memory systems
on this, the claim is backed by three measurable gates, not vibes:

| Gate | Question | Metric | Where it runs | Status |
|------|----------|--------|---------------|--------|
| **Adherence** | Does the model OBEY a rule in context? | ON>OFF compliance, ON≥0.8 | box (needs LLM) | **CLEARED** 2026-07-27 (judge gpt-oss-120b) → read side default ON |
| **Extraction precision** | Are auto-inferred rules CORRECT? | precision ≥ 0.90 | anywhere (deterministic) | **PASS** locally 100% — pending box run on real markers → write side default OFF |
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
to `True` and record the precision + n here. FAIL → read the FP list; the fix is
the classifier in `hymem/rules.py::rule_scope_for_marker` (e.g. tighten the
directive-cue vocabulary), NOT the probe. Local tuning already caught the
rejection one-off leak (85%→100%) this way.

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
- Flip **write side** (`rules_extraction_enabled`) only after Gate 2 passes on
  real dream markers. Then commit and record the numbers in
  `additional_planning.md` §Idea B STATUS.
