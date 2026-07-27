# Idea B (`always_on` Rules) — box runbook: the LLM-adherence gate

**Audience:** whoever runs the Hermes box. The feature is built and **default
OFF**; this is the one gate it needs before the default can flip — does the
answering model actually OBEY a standing rule? It mirrors the P4 profile box gate
(a pass/fail adherence gate, NOT a benchmark number). Copy-paste from the repo root.

## What's already done (nothing to build)
- Feature: `hymem/rules.py` + `ctx.rules` in `augment()` (gated on
  `cfg.rules_enabled`, default False) + `HyMem.add_rule()`/`retract_rule()` +
  rules rendered FIRST in `hymem/query/ask.py` with an obey-directive appended to
  the system prompt when rules are present. Schema v23 (`rules` table).
- Mechanical gate (no LLM, runs anywhere):
  ```bash
  python -m pytest tests/test_rules.py -q          # expect 12 passed
  ```
- Adherence harness: `benchmarks/rules_compliance.py` (self-contained — 6
  rule/tempting-probe triples, no LME/BEAM data, no dream).

## Prerequisites
- An OpenAI-compatible LLM endpoint + key. Install the client extra once:
  ```bash
  pip install 'hymem[server]'
  export HYMEM_LLM_API_KEY=...          # or DEEPSEEK_API_KEY / OPENAI_API_KEY
  ```
- Optional dry run (no API — proves the pipeline, not adherence):
  ```bash
  python benchmarks/rules_compliance.py --answer-model stub --judge-model stub
  # expect: rule@ON all "yes", the "rule present in every ON / no OFF" check ✓.
  ```

## Run the gate
```bash
python benchmarks/rules_compliance.py \
    --answer-model deepseek-v4-flash \
    --judge-model  deepseek-v4-flash \
    --verbose
```
For a machine-readable line instead: add `--json`. To point at a different
answerer (gpt-oss, a local vLLM, Claude): `--answer-model <name>
--answer-base-url <url> [--answer-api-key <key>]` (same trio for `--judge-*`).

> **Judge independence.** The harness defaults answerer == judge for
> convenience, but a model grading its own output can flatter itself. For the
> gate of record, prefer a DIFFERENT (ideally stronger/neutral) judge, e.g.
> `--answer-model deepseek-v4-flash --judge-model gpt-oss-120b
> --judge-base-url <gpt-oss-url>`. Hold the judge fixed across any re-runs.

## How the harness works
Per triple it asks the SAME question twice on a fresh store:
- **ON** (`rules_enabled=True`) — the rule is in context + the obey-directive.
- **OFF** (`rules_enabled=False`) — no standing rule; the base model's default.

A judge scores each answer `comply` / `violate` / `unclear`. The OFF arm is the
control: it isolates whether the RULE caused compliance or the model already
behaved that way.

## Read the result — the gate PASSES iff all three hold
1. **ON adherence ≥ threshold** (`--threshold`, default 0.8) — the model obeys
   when told.
2. **ON > OFF** — compliance is caused by the rule, not the base model's habits.
   *A rule that "passes" only because OFF also complies has proven nothing.*
3. **rule present in every ON context and no OFF context** — the mechanical
   invariant; a ✗ here is a wiring regression, not a model result.

Exit code is 0 on PASS, 1 on FAIL (0 for a stub run — its gate is meaningless).

## Decide
- **PASS** → the tier is safe to enable. Flip `rules_enabled` default to `True`
  in `hymem/config.py`, bank the adherence table + the answerer/judge/threshold
  used in `additional_planning.md` §Idea B STATUS, then commit.
- **FAIL on (1)** → the model ignores rules even in-context; a system-prompt/
  directive change is needed (test as its own A/B — recall the DeepSeek procedural
  regression, keep it simple). Default stays OFF.
- **FAIL on (2) only** → the probe set is too easy (base model already complies).
  Sharpen the probes to tempt violation harder (`--probes my_probes.json`,
  same `{id,rule,question,watch}` schema) and re-run; not a feature failure.
- **FAIL on (3)** → a rendering/wiring regression — re-run the mechanical pytest
  and fix before trusting any LLM numbers.

The ship default stays `False` until (1)+(2)+(3) hold on a run with an
independent judge.
