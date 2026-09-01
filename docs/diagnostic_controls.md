# Diagnostic controls

**Every measuring device in this repo gets a control arm before its number is
read.** Not the thing being measured — the *device*. This document is the
standing discipline behind the benchmark instruments, the pre-registered gates,
and the test suite's own guards.

It exists because the rule has been violated in all three places, each time
producing a confident number that was wrong, and each time costing more to
retract than the control would have cost to run.

---

## 1. The rule

A gold-surface diagnostic is a **classifier**, and a classifier's output is
uninterpretable without a population where the answer is known by construction.

That population is usually free. For a retrieval/delivery check it is
**questions the reader answered CORRECTLY** — those had the evidence delivered,
by definition. The check's failure rate on that population *is* its error rate,
and only the **excess** over it is evidence of anything.

> Ask of any diagnostic, before reading its number: **what does this return when
> it is broken?** Then check that case first.

A device that returns a confident constant when broken is indistinguishable
from a device that works, unless you have deliberately arranged a case where the
two disagree.

---

## 2. Measured control rates

Measured 2026-07-29/30 across the benchmark triad, after a LoCoMo miss
decomposition sent the levers queue the wrong way for four turns.

| benchmark | check | kind | control rate | direction of error |
|---|---|---|---|---|
| LoCoMo | `_lex_match` τ=0.6 vs render | soft overlap | **11% FP** | over-credits delivery ⇒ inflates SYNTHESIS, blames the reader |
| MSC | `_lex_match` τ=0.6 vs render | soft overlap | **13% FP** | same; measured excess only +2pp ⇒ bucket is genuine |
| LME | `_gold_in_pool` vs pool | containment/prefix | **≤3% FN** | under-credits ⇒ inflates RETRIEVAL, blames the retriever |
| LME | per-turn `gold_turn_tiers` | containment | **12% FN** | same |

τ=0.6 soft overlap misfires roughly **1 in 8** on genuinely-delivered evidence,
consistently across two corpora. Treat ~11–13% as the baseline any `_lex_match`
audit subtracts. LME's ≤3% is an *upper* bound: some correct answers come from
parametric knowledge with no retrieval at all, which inflates the control.

**Why LoCoMo was the outlier.** LME compares the gold turn against the **pool**,
where `message_hits` expose the raw turn verbatim — containment is the right
test for that haystack. LoCoMo compared against the whole **render** (raw turns
+ consolidated memories + profile entries) with **soft** overlap: wrong test,
wrong haystack, so it fired on any neighbour sharing content words. LoCoMo has
exact evidence text in `evidence_map` and raw turns in its render, so porting
`_gold_in_pool`-style containment would buy 11% → ~3%.

**What the 11% cost:** a retracted "≤+2.5pp retrieval ceiling" and three
reversed lever orderings.

---

## 3. Three structural traps

All three return a confident constant when broken.

### (a) Nested surfaces make a bucket unreachable

`render ⊆ top_k ⊆ pool`, so `gold_in_context=True` **forces**
`gold_in_topk=True`. An "in top_k, lost the cut" bucket is unreachable whenever
tier hits fit inside the cut — it reads 0 by construction, and 0 looks like a
finding.

### (b) SQLite external-content FTS5 answers from the wrong table

`WHERE rowid=?` on `messages_fts` is answered from `messages` and reports
**unindexed rows as present**. Test membership with a `MATCH` on the row's own
rare tokens instead.

### (c) A scored artifact that records less than the model's input

`fact_probe.py` sent 12,000 chars to the model but stored `source_turns[:4000]`
— since its first commit. Every fact faithfully taken from chars 4000–12000
scored as an **invention**.

The signature: "hallucinations" that are hyper-specific, mutually consistent,
and reproduce verbatim across prompt rewrites. Faithful extraction does exactly
that too, which is why it survived — it contaminated G-F1's model-side verdict,
not just one run.

**Magnitude,** measured on the healed G-F1b re-read (2026-08-02): **50 of 50**
facts the truncated record scored as inventions were verbatim in the unrecorded
region. The trap flipped a 1.00 result to an apparent 0.59.

> The scored artifact must **byte-equal** the model input. A faithfulness read
> that never CONFIRMS an invention against full source has measured nothing.

---

## 4. The same rule applies to gates

A gate that **cannot fail** returns a confident constant and reads as a PASS.
Campaign E hit this three times in one step (2026-07-30/31). All three were
caught by arithmetic, before or instead of a run.

| # | failure | signature |
|---|---|---|
| a | **Ceiling instrument** — the rerank handset's BM25 baseline already put gold at median rank 1.0 on all 30 questions, so both arms of M1/M2 scored 1.0/1.0 | every cell of the results table identical, at the maximum |
| b | **Degenerate criterion** — with `--pool 40 --top-k 15`, "≤15 share" is arithmetically identical to "Found", because every returned item has rank ≤15 by construction | two metrics that never disagree |
| c | **Unreachable code path** — a proposed LME guard on `rerank_cross_encoder_model` returns zero delta whatever the value, since `rerank_model` defaults to `"llm"` and `augment.py` routes to the CE only on `== "cross-encoder"` | a clean no-op that looks like a clean pass |

**Two checks, both free:**

1. **Gate the measuring DEVICE on a baseline arm independent of the arms under
   test**, before spending anything. (E3's handset v3 front-run: BM25 median
   ≥4, zero at rank 1, all gold reachable.)
2. **Before any non-regression run, confirm the code path under test is
   reachable from the config the guard actually runs.**

**Fourth, related: below-chance agreement is a bug signature, never a finding.**
Two arms each drawing 15 from the same 40 share ~5.6 by chance, so E3's observed
<3 was anti-correlation — an `id()`-keyed overlap dict vs `replace()`-copied
hits — not "the arms disagree".

### The vacuity split

A gate reporting "0 changes" must also report **how many rows could have
changed**. If a row is decided by a fast path both arms share, it is immune to
the change under test, and on a corpus of such rows "0 changes" **cannot fail to
be 0**.

`judge_audit.py`'s `--verify-parse` and `--verify-recitation` both carry this
denominator (`token_rule_consulted`), and it is what makes their certificates
readable rather than merely reassuring.

---

## 5. And to the test suite's own guards

House discipline: **revert each guard one at a time and confirm it fails exactly
its own test.** A guard whose test cannot fail is unguarded.

Run on 18 guards across the D3 / `recites_gold` work (2026-08-26). Three read
UNGUARDED:

- **Two were real test gaps.** One: a substring check on a file passed while one
  of **two** record builders in the same module had lost the field. Replaced
  with an `ast` invariant — every dict recording a verdict alongside its
  question carries the channel — rather than a second substring.
- **The third was the probe lying.** Stale `.pyc` bytecode meant pytest
  re-imported the unbroken module, so a fully guarded guard reported as
  unguarded.

> **Run any reversion probe in this repo with `PYTHONDONTWRITEBYTECODE=1`, and
> have it report which GUARDS it broke, not only which tests failed.**

A negative-control device returning a confident wrong constant is the exact
failure mode negative controls exist to catch. The rule at the top of this
document eats its own tail if you let it.

---

## 6. Read the instrument, not the empty space

`pyproject.toml` sets `addopts = "-q"`. A hand-typed `-q` therefore stacks to
`-qq`, and the **pass-count summary line disappears**.

Failures still print under `-qq`, so the device cannot return a false PASS — but
it stops returning a positive one, and:

> "I saw no failures" is not "I saw 162 pass."

Read the count off an explicit `-o addopts=""`:

```bash
python -m pytest tests/test_judge_instrument.py -o addopts=""
# 162 passed in 0.16s        <- the positive line
```

This generalises past pytest. An absence of alarm is not a measurement; find
the affirmative signal the device emits when it is working, and read that.

---

## 7. Applying it — the checklist

Before reading any diagnostic number:

- [ ] What does this device return when it is **broken**? Have I seen that case?
- [ ] Is there a population where the answer is known **by construction** — and
      have I run the device on it?
- [ ] Is the scored artifact **byte-identical** to what the model saw?
- [ ] Can the metric's buckets all be **reached**, or does nesting force one to 0?
- [ ] Does the two-metric criterion have a case where the metrics **disagree**?
- [ ] Is the code path under test **reachable** from the config being run?
- [ ] For a "no change" result: what is the **vacuity denominator**?
- [ ] Am I reading a **positive signal**, or the absence of a negative one?

---

## 8. Instruments

All read-only, no model calls.

| instrument | what it does |
|---|---|
| `benchmarks/locomo_audit.py` | strict re-score with a **mandatory** correct-answer control; three-surface split via `--topk-dump` |
| `benchmarks/locomo_adapter.py --diag-only` | retrieval + render, no reader; deliberately refuses to print `0.0%` for a run that measured nothing |
| `benchmarks/locomo_index_probe.py` | NOT INGESTED / NOT INDEXED / NOT MATCHED / MATCHED against the persisted stores |
| `benchmarks/judge_audit.py --verify-parse` | parse-rule flip check, with the vacuity denominator |
| `benchmarks/judge_audit.py --verify-recitation` | `recites_gold` v1↔v2 diff, both ceilings, shuffled-gold control arm |
| `benchmarks/facts_ab.py` | McNemar on the FIRED subset, with the NOT-FIRED subset as a built-in negative control |
| `benchmarks/locomo_flip.py` | paired flip classifier; drops an unscored row from **both** arms or neither; refuses to call a pair a re-judge when neither arm recorded a reader answer |

---

## 9. Provenance

This discipline is applied in place throughout `benchmarks/judge_audit.py`'s
docstrings and in `additional_planning.md`'s campaign records. Those are the
local applications; this document is the general statement.
