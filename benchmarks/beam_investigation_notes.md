# BEAM Floor-Category Investigation Notes

**Branch:** `Beam-optimisation` · **Adapter:** [`beam_adapter.py`](beam_adapter.py) · **Last updated:** 2026-06-13

Running log of the BEAM scoring investigation: what moved the score, what
turned out to be noise, and what the per-category failures actually are at the
code level. Kept in-tree so we can circle back. (The "×3 multiplier" regression
that preceded this work is its own story — summarized under *Background* below.)

---

## TL;DR — current state

- **Honest BEAM score = 52.1%** (per-conversation **isolated** stores, sample ≈ 10).
  The 59.8% "shared-store" number is inflated by cross-persona contamination and
  should **not** be reported or chased — see *The isolation experiment*.
- **Solid, validated wins** (far above noise): ability-routed answering prompts —
  **CR 8→54, IF 25→67**. These are real and committed.
- **Remaining floor at the honest number:** EO ~8%, SUM ~38%, KU 50%, PF 75%.
- **EO/SUM** are a retrieval-architecture problem (top-k relevance is the wrong
  tool for "coverage" questions), *not* an answering problem.
- **KU/PF** dropped under isolation because shared mode was leaning on
  cross-persona facts; extraction itself is fine (code-verified). The lever-vs-
  ceiling check **ran: 10/10 KU zeros have the gold fact in their own turns** — so
  it's a **ranking/recall** lever, not a ceiling. Root blocker: KU leads its
  context with *undated, deduped* graph facts, so dedup keeps an arbitrary value
  and the recency clause can't act. See *Single-conversation fact graph*.

---

## Score progression

| Stage | Overall | Notes |
|---|---|---|
| v16 (Jun 1, `a988f94`) | 47.4% | 30-memory context (`top_k×3`), raw prompts, no dating |
| + ×3 restore | 50.4% | multiplier had been silently dropped; restored |
| + dating + ability prompts | 59.1% | real event dates + CR/IF/EO/SUM routed prompts (sample=3) |
| + EO/SUM episode coverage | ~59% | CR/EO moves within noise at sample=3 |
| **isolated, sample≈10** | **52.1%** | **the honest number**; shared-store equivalent = 59.8% |

SOTA reference (100K): Hindsight 73.4, Mnemosyne 65.2, Honcho 63.0, LIGHT 35.8, RAG 32.3.

---

## Background — the "×3" regression (closed)

A June "regression" (51% → ~36%) was traced to the `top_k × 3` answer-context
multiplier being silently dropped in a refactor (`dea8d94`), shrinking the answer
context from 30 memories to 10. Restoring it recovered the score. No model
change, no contamination, no outlier — just a dropped multiplier and a
misattributed commit message. See git history around the `×3` restore commit.

---

## What worked: ability-routed answering prompts

**Diagnosis (from per-question records):** every floor-category failure was a
prompt/task mismatch, not a retrieval miss:

- **CR (contradiction resolution):** model answered definitively instead of
  surfacing the conflict. The default prompt's recency clause (prefer the newest
  value — correct for KU) is exactly wrong for CR.
- **IF (instruction following):** rubrics are *format* checks (code blocks,
  version numbers, numeric codes). Answers were substantive but missed the format.
- **EO (event ordering):** model couldn't order shuffled, relevance-ranked
  snippets.
- **SUM (summarization):** rubrics list 4–6 specific facts; model gave vague
  summaries or wrongly abstained.

**Fix (landed, committed):** route a dedicated answering prompt per ability —
CR surfaces both sides with dates then resolves; IF follows standing format
instructions and reproduces specifics verbatim; EO/SUM get coverage-oriented
prompts and 2× context budget. `IE/KU/ABS/PF/MR` routing left untouched
(additive principle).

**Result:** CR **8.3 → 54.2**, IF **25 → 66.7** — both far above the noise floor,
unambiguous wins.

### Event dating (enabling fix)

Every BEAM message carries a `time_anchor` (e.g. `March-15-2024`), one per
session block, ~3 blocks/conversation spanning weeks. The adapter had been
**discarding it** — all `[MEM]` tags showed ingestion day, so the KU recency
clause ran on meaningless dates and EO had no ordering signal. Now parsed
(`_parse_time_anchor`), propagated per block, and passed to `log_messages` as
the event time (with a per-turn second offset so intra-session order survives
date-only granularity).

---

## EO/SUM are a coverage problem, not a ceiling

The box initially called EO a "retrieval recall ceiling." The data refutes that:
the model systematically returned **planning/scheduling** content for *every* EO
question — a systematic category swap, not scattered near-misses. That's the
signature of relevance-ranking a vague ordering query: "what order did I build
the features" lexically matches the planning turns, not the implementation turns
("added transaction error handling").

**Root bug found:** episodes (per-session "what happened" summaries) were sliced
off EO entirely — `search()` built `message_hits` first, appended episodes to the
leftover, then `[:top_k]` (=30); with ≥30 message hits, episodes never survived.
EO was answered from raw turns only.

**Coverage lever (landed, committed):** for EO/SUM, lead with `episode_hits[:8]`
then the message timeline, and reserve the overview so the slice can't eat it.
For EO, the chrono-sort keeps episodes leading (survive char budget) then the
dated raw-turn timeline. TR ordering left byte-identical (separate branch).

**Result:** EO 6.7 → 13.3 (one question jumped; per-question trace confirms the
mechanism — old answer listed planning events, new answer listed real
implementation milestones from an episode). SUM unchanged.

**Why it's only partial — the real bottleneck:** episode summaries at this scale
are too *abstract* ("developed budget tracker with Flask, added auth") to
decompose into a rubric's fine event sequence. When episodes lack granularity the
model falls back to raw turns and planning turns win the ranking again. **Episode
granularity is a core *dreaming* concern, not an adapter knob** — it should be
driven by LME + qualitative eval (where it helps everything), not tuned against a
handful of EO questions.

**Rejected idea:** "feed ALL raw turns chronologically, bypass relevance." The
BEAM scale labels (100K/500K/1M/10M) are *token counts of full history*; the
benchmark exists because that doesn't fit. The answer budget (~4k tokens) holds
~30 of ~188 turns at 100K, and it's hopeless above. The feasible steelman is
**per-block sessions + a session-scoped chronological feed** (see *Open levers*).

---

## The isolation experiment — why 52.1%, not 59.8%

Run at sample ≈ 10 in **both** modes:

| Category | Shared | Isolated | Δ |
|---|---|---|---|
| KU | 83.3 | 50.0 | **−33** |
| PF | 100 | 75.0 | **−25** |
| TR | 33.3 | 60.0 | **+27** |
| EO | 13.3 | 8.2 | −5 (noise) |
| SUM | ~37 | ~38 | flat |
| **Overall** | **59.8** | **52.1** | −7.7 |

The overall barely moved while categories swung ±30pp in **opposite directions** —
the headline number is a poor steering signal; the structure is all per-category.

**Verified: BEAM conversations are independent personas** (conv1 = Craig Baker,
49M, budget app; conv2 = Christina Baker, 56F, weather app — each row has its own
`user_profile`). Therefore:

- The shared-store advantage on KU/PF is **cross-persona contamination**, not
  real-world utility. The order-concentration (later conversations are the ones
  isolation breaks) is the contamination signature — later questions ride an
  accumulated multi-persona graph.
- **TR is the internal control.** The *same* cross-conversation reach that helps
  KU/PF (more facts) hurts TR (wrong-persona dates polluting the timeline, −27pp).
  You can't bank the KU/PF gains as "utility" while writing off the TR loss as
  contamination — it's one mechanism.
- So **52.1% is the valid number.** 59.8% transfers to no sane deployment
  (single-user: no other users in the graph; multi-user: isolate per user).

Cross-conversation accumulation *is* a genuine HyMem feature — but for the
single-user-multi-session case, which **LME** measures, not BEAM. Do **not** ship
shared-store-across-users to chase the 59.8 (privacy leak + non-transferable).

---

## Single-conversation fact graph — does isolation "starve" it?

**Code-verified verdict: the strong claim is false.** Triple extraction is
per-chunk via LLM (`phase1.extract_chunk_results`; `run_dreaming` just loops the
sessions' chunks) — **store-size independent.** A conversation's own facts become
graph edges identically whether isolated or shared. "Most specific facts never
get extracted" does not hold.

What isolation **actually** removes (all code-grounded):

1. **Aggregation/digest tier is hard-dead.** `aggregation_min_sessions = 2`
   ([`config.py`](../hymem/config.py) L159) + the adapter ingests one session per
   conversation ⇒ it never fires. *But* the adapter only reads
   `result.graph_facts` — it never consumes aggregation nodes — so this doesn't
   hit the KU score directly.
2. **Confidence flattens.** Confidence = Laplace `(pos+1)/(pos+neg+2)`
   (`phase2.py`). Shared-mode dreaming accumulates `pos` on matching edges via the
   cross-cycle dedup pool (recurring facts → pos 7+); isolation leaves
   single-mention facts at pos=1 → 0.67. `_graph_lookup` ranks by
   `confidence × recency × semantic`, so one ranking axis goes flat in isolation.
3. **Cross-persona corroboration vanishes.** `_graph_lookup`
   ([`augment.py`](../hymem/query/augment.py) L533) gathers candidates across the
   whole edge store with **no session filter**; shared-mode KU/PF answers were
   boosted by other personas' facts. This is the contamination, correctly removed.

**So the KU/PF drop is loss of (mostly illegitimate) cross-persona corroboration,
not an extraction gap.**

**The check ran — verdict: PRESENT, 10/10.** Every isolated KU zero has the gold
fact in its *own* conversation turns. This is a **retrieval-ranking/recall**
problem, not a contamination ceiling. (My earlier lean toward *absent* was wrong;
the data corrected it.) The signature: 9/10 the model confidently returns a
*sibling* value — same entity, wrong number (gold 78% coverage → returned 65%;
gold 5 interviews → returned 3; gold April 22 11AM → returned April 21 3PM). 1/10
(conv 4) it abstained having seen two wrong candidates but not the gold one.

**Precise mechanism (code-grounded), sharper than "confidence-axis problem":**
in isolation confidence is *flat* (all single-mention facts = 0.67), so the wrong
value isn't winning on higher confidence — there's no gradient. The real blocker
is structural:

1. **KU leads its context with `graph_facts`, not raw turns.** KU is in the `else`
   branch ([beam_adapter.py](beam_adapter.py) L490-503); `TASK_RECALL` (L355)
   excludes it. Graph facts come first.
2. **Graph facts are deduped to one value per attribute** — when the persona
   stated an attribute twice ("70% after 12 problems" → "95% after 15"), dreaming
   kept *one* edge, and with confidence flat it kept an essentially arbitrary one.
3. **Graph facts are undated**, so the value-aware recency clause is *structurally
   inert* on the exact tier that leads KU answers — `[FACT]` gets no date, only
   `[MEM]` raw turns do (L558-562, "graph dating deferred"). The gold-bearing dated
   turn sits *below* the facts and may be sliced by the 30-cap.

This also explains why event-dating moved **LME**-KU hard (+11.5pp) but
**BEAM**-KU "held exactly" at 83.3: the dating lever's reach is tier-limited to
`message_hits`; the tier that leads BEAM-KU is still dark.

**Correction to the "entity-match + recency" prescription:** recency only fixes
the *update/reschedule* cases (newest wins). It *hurts* the *qualifier* cases
("after 15 problems = 95%" may precede "after 20 = 90%"); there the question's
qualifier is the key, not recency. Robust signal = entity-match + **qualifier**-
match, recency as tiebreak only when the question asks "current/now". The
unifying failure is that retrieval surfaces *one* value and the answerer commits
to it → durable fix is **recall** (all candidate values, each dated+qualified)
more than rerank.

**Tag classification RAN — verdict: 8/8 `[MEM]`, not `[FACT]`.** (My `[FACT]`-first
prior was wrong; the graph tier isn't in play here.) And the pattern is uniform:
every zero is an **old-value-vs-updated-value** case where the conversation
contains *both* and the model returns the **stale** one:

| Conv | Wrong (stale) | Gold (updated) | Pattern |
|---|---|---|---|
| 2 | 65% coverage (T114) | 78% (T128) | improved |
| 6 | 3 interviews (T62) | 5 (T92) | got more |
| 7 | 45 sources (T82) | 52 (T110) | added more |
| 8 | Apr 21 3PM (T92) | Apr 22 11AM (T80+) | rescheduled |
| 9 | 3:00 PM | 4:30 PM | rescheduled |
| 10 | 1,800 words (T242) | **1,350 (T64)** | reduced — **stale mentioned LATER** |
| 10 | Apr 20 (T86) | Apr 25 (T168) | extended |

**Refined mechanism (code-grounded):** the box's "older turn's higher *confidence*
drowns the date" isn't what the code does — all `message_hits` get a **constant
`confidence: 0.7`** ([beam_adapter.py](beam_adapter.py) L392), no gradient. They
tie on confidence and keep **augment()'s relevance order**; **date plays zero role
in `search()` ranking** — it's only a display tag for the answerer (L393-395,
L561). So the failure is: relevance surfaces the stale turn at/above the updated
one, and either the updated turn falls below the 30-cap (recall miss → recency
clause never gets both) or both survive but the clause isn't honored. The box's
*location* (search() ordering) and *axis* (recency over the flat default) are
right; the lever is relevance-vs-recency, not confidence-vs-recency.

**A turn-recency reorder was then tried, run, and falsified.** I predicted "7/8
clean forward-recency, conv 10 the residual." The box run showed the **opposite**:
a Borda recency-blend on `message_hits` went net **−3** (fixed only conv 8, broke
conv 1/3/4/6 that had been working). The retrospective pattern isn't a rare
residual — it's the **dominant** mode. The gold turn *asserts* the update ("updated
to April 5"); it is usually **older** than a later turn that merely *references*
the stale value ("per the April 1 deadline"). Mention-recency pulls the reference
up and slices the assertion, so the answerer sees only the stale value. Conv 8 was
the lone case where mention-order happened to coincide with validity.

**Conclusion: turn-level recency is the wrong axis.** The recency that matters is
fact-*validity* recency — when the fact became true, not when it was last
mentioned. The answerer's recency clause already resolves updates correctly **when
both values reach it** (that's why conv 1/3/4/6 worked before the blend); the blend
broke that by rearranging which turns reach the answerer. The real lever is the
schema-v15 bi-temporal `valid_at`/`invalid_at` path (extract update semantics, set
validity, prefer latest-valid at retrieval) — core, LME-validated, *not* an adapter
reorder. See *Open levers*.

---

## Open levers / next steps

- [x] **Present-vs-absent check** on isolated KU zeros — **DONE: 10/10 present.**
      It's a ranking/recall lever, not a ceiling.
- [x] **KU `[FACT]`-vs-`[MEM]` tag classification** — **DONE: 8/8 `[MEM]`,** all
      old-vs-updated, model returns the stale value. Graph tier not in play.
- [x] **KU/PF recency-blended message_hit ordering (adapter)** — **TRIED &
      REVERTED.** Box run: net **−3** (fixed 1 KU zero — conv 8, the only case where
      turn-recency aligned with validity; **broke 4 previously-working cases**:
      conv 1 165→150, conv 3 Apr5→Apr1, conv 4 15→10 problems, conv 6 7→5 women).
      Mechanism of the regression: the gold turn *asserts* the update ("updated to
      April 5") and is usually **older** than a later turn that merely *references*
      the stale value ("per the April 1 deadline"); mention-recency pulls the
      reference up and slices the assertion. The answerer's recency clause already
      resolves updates correctly **when both values reach it** — the blend broke
      that by rearranging which turns reach the answerer. **Turn-level recency is
      the wrong axis.** Reverted; tombstone comment left in `search()`.
- [~] **Fact-validity recency via schema-v15 `valid_at`/`invalid_at`** — the
      dominant KU failure mode (not a residual). LME-driven, core, not BEAM-hand-fit.
      Progress (all local + deterministic; LME A/B pending on the box):
      - **Step 0 (done):** repro test `tests/test_ku_update_supersession_repro.py`
        pins the gap — a single update emits only positive evidence, so the old
        edge stays active. Settled the routing: the real extraction prompt is
        tech-stack-framed and unreliable on value updates; even when it emits a
        `replaces`/`-1`, nothing acts on it (read-only at query time; one negative
        can't trip phase3). The detector + correctness discriminator already exist
        in `query/conflicts.py` (`_competing_objects`, functional predicates) but
        are read-only.
      - **Step 1 (landed, flag default OFF):** `hymem/dreaming/value_supersession.py`
        — a dream-cycle consumer that retracts the older-`valid_at` edge among
        competing **typed-value** objects (numeric/%/count via
        `kg_evidence.value_numeric`, `value_unit` as compatibility key), closing
        `invalid_at` at the winner's `valid_at`. Gated by
        `cfg.value_supersession_enabled`. Typed-value scoping keeps multi-valued
        facts safe. Test flips green with the flag on; full suite green.
      - **A/B is now runnable:** `longmemeval_adapter.py` has a `--value-supersession`
        flag (wires `value_supersession_enabled`). Protocol: full-dream, `--sample 500`
        (KU needs its ~70 items), 3 paired repeats, change only this one bit vs the
        canonical 70.0 baseline. Read OVERALL (G4 guard, hold ~70 ±1.5pp) +
        knowledge-update strict (target, needs >±5pp to clear noise) + collateral
        watch on multi-session / single-session-preference / knowledge-update_abs.
        **Pre-check before interpreting:** confirm dream log `bitemporal.value_superseded
        count>0` — if 0, real extraction isn't populating `value_numeric` and a null
        result means discriminator-too-narrow, not lever-fails.
      - **Then:** decide the date/version representation and the default flip.
- [ ] *(superseded)* turn-recency reorder and the block-tag sub-check — both moot;
      mention-order is the wrong axis (see the reverted blend above).
- [ ] **Per-block sessions:** split each conversation's ~3 time-anchored blocks
      into distinct HyMem sessions. Helps **EO** (clean per-session timelines) and
      restores the aggregation tier — but only helps KU/PF *if* the adapter starts
      consuming aggregation/digest output (it currently doesn't). Does **not** by
      itself add evidence accumulation.
- [ ] **Episode granularity** (core dreaming): the real EO/SUM bottleneck. Drive
      via LME + qualitative eval, not BEAM tuning.
- [ ] **MR nit:** one observed failure was the model describing four features but
      never stating "Four." Low-risk prompt hardening; not worth chasing at small
      sample.

---

## Methodology lessons (don't relearn these)

- **Noise floor is large.** At sample=3, CR moved **±12.5pp on code that doesn't
  touch CR** (pure LLM nondeterminism; temp 0 ≠ deterministic on deepseek). At
  sample≈10 it's ≈ ±7pp. A category delta below the floor is not a signal — only
  CR/IF (≈+45) and the isolation swings (±27–33) clear it. Same lesson as the LME
  variance band.
- **Overall is a poor steering metric** when categories move in opposite
  directions (isolation: ±30pp per category, ~8pp overall). Work per-category at
  the mechanism level.
- **Keep effective config in saved artifacts.** The whole ×3 regression hid for
  days because the decisive variable (context width) appeared in no run output.
  The adapter now records `context_memories` and full per-question records
  (`answer`/`ideal`/`rubric`/`scores`) so runs can be re-judged and diffed.
- **BEAM measures per-conversation memory.** Its conversations are independent
  personas; it cannot validly reward cross-conversation recall. Use LME for that.
