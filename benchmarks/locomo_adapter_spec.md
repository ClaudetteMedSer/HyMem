# LoCoMo adapter — design spec

**Why.** LoCoMo ("Evaluating Very Long-Term Conversational Memory of LLM Agents",
Maharana et al., ACL 2024) is the third leg of the benchmark triad and covers what
the other two structurally can't:

| | LME | MSC | LoCoMo |
|---|---|---|---|
| sessions/history | 40-60, star topology (fact stated once) | 2-5, genuine recurrence | **19-32, genuine recurrence** |
| turns/history | ~hundreds | ~20-60 | **369-689** |
| timestamps | real session dates | none (synthesized) | **real per-session date-times** |
| adversarial class | `_abs` (6% of qs) | none (all answerable) | **cat-5 = 22% — premise-swap traps** |
| multi-hop labels | none | none | **cat-1 with per-hop evidence ids** |

Three specific draws: (1) **cat-1 multi-hop with multi-turn evidence annotations**
is the first labeled target for Track-A `--graph-multihop` A/Bs on a non-star
corpus ([[project_track_a_multihop]]: LME's star topology made the feature inert);
(2) **cat-5 adversarial** exercises the abstention posture MSC forced us to
override — the two benchmarks now hold opposite ends of the answerability contract;
(3) 30-session histories at real timescales (May→Oct 2023) stress the recency
lever, supersession, and dream consolidation at a depth MSC never reaches. It is
also the de-facto industry comparison corpus (Mem0, Zep, MemGPT re-evals all
report on it).

> **STATUS 2026-07-28 — CANONICAL 70.2% answerable / 76.0% overall** (n=200,
> seed 0, `--top-k` ×3 aperture + `--embeddings` + `--max-context-chars 24000`,
> reader thinking OFF). Arc: 53.0 → 59.6 (3× aperture) → 65.6 (embeddings at 3×)
> → **70.2** (24k budget). Everything above was adapter/config-side; the core was
> never touched — the same pattern as the MSC arc ([[project_msc_benchmark]]).
> Residual at canonical: 13 retrieval misses (query→turn paraphrase tail,
> `--name-prefix` territory) and 32 synthesis. **Read every delta here against
> the churn floor in §8 — it is larger than several of the levers that produced
> this number.**
>
> Build validation (2026-07-28, still current): data contract verified against the
> REAL `data/locomo10.json` from snap-research/locomo (not a paper description —
> the file has quirks a spec-first build would have missed, §1); `--sim` fixture
> end-to-end (all 5 categories); real-file loader (10 convs / 1986 QA, category
> counts match raw inspection, dates parse monotonically, **1979/1986 questions
> have fully locatable evidence turns**); real conv-26 ingest with StubLLM;
> `--db-dir` persistence + reuse; `--workers` parallelism.

---

## 1. Data contract (VERIFIED 2026-07-28 against snap-research locomo10.json)

`locomo10.json`: 10 conversations, 19-32 sessions each, 1986 QA rows.
Fetch (CC **BY-NC** 4.0 — kept out of git via `.gitignore`; decide on
redistribution before ever committing it):

```
curl -L -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

```
conversation = {
  "sample_id": "conv-26",
  "conversation": {
    "speaker_a": str, "speaker_b": str,
    "session_<N>": [ {"speaker": str, "dia_id": "D<N>:<t>", "text": str,
                      # photo-share turns additionally carry:
                      "img_url": [str], "blip_caption": str, "query": str}, ...],
    "session_<N>_date_time": "1:56 pm on 8 May, 2023", ... },
  "qa": [ {"question", "answer": str|int, "evidence": ["D1:3",...], "category": 1-4},
          {"question", "adversarial_answer": str, "evidence", "category": 5}, ...],
  "observation" / "session_summary" / "event_summary": ...,  # oracle aids — unused
}
```

Empirical quirks (all present in the real file; the loader absorbs each):
- `category` is **sometimes a string** (`'5'`) → int coercion.
- cat-5 `evidence` is a **string repr of a list** (`"['D2:3']"`) → regex
  `D\d+:\d+` extraction over `str(evidence)`.
- answers: 1536 str, **6 int**, 444 None (cat-5); **2 rows carry BOTH** `answer`
  and `adversarial_answer` (treated as cat-5).
- 1226 turns carry photo shares → `blip_caption` appended in-line
  (`… [shared a photo: <caption>]`), the standard text-only treatment.
- 4 QA rows have no evidence ids; 3 more have partially unlocatable ids →
  `gold_distance=-1`, tracked, excluded from nothing except distance rows.

Categories (counts): **1 = multi-hop (282), 2 = temporal (321), 3 = open-domain
inference (96), 4 = single-hop (841), 5 = adversarial (446)**. Confirmed by
sampling, not assumed: cat-5 rows are cat-4 questions with the **speaker/premise
swapped** and `adversarial_answer` holding the *trap* answer (e.g. cat-4 "What did
Melanie realize after the charity race?" → "self-care is important"; cat-5 "What
did **Caroline** realize after **her** charity race?" → trap "self-care is
important"). Correct cat-5 behavior is saying the information isn't there.

---

## 2. The MSC lesson ×3, restated up front

The entire MSC climb (42.0 → 84.0) was adapter-side contract restatement
([[project_msc_benchmark]]). LoCoMo states all three on day one:

- **Feeding parity.** Retrieval is `MSCAdapter.search` reused WHOLESALE (one
  shared implementation, no third copy): `top_k*3` at the pipeline layer,
  message-first tier ordering, additive P4 profile tier, full pre-truncation
  pool. The ×3 multiplier alone was the BEAM June regression AND MSC's first
  23pp.
- **Deixis.** LoCoMo questions are **third-person by name** ("What did Caroline
  research?") while memories carry `[user]`/`[assistant]` tags — the exact MSC
  bug class (+14pp there), except here attribution swaps aren't just misses:
  **cat-5 exists to punish them**. `locomo_perspective_clause(a, b)` states the
  per-conversation name↔role mapping via `extra_system` (additive; LME postures
  byte-identical).
- **Answerability — deliberately NOT the MSC clause.** Cats 1-4 are answerable
  by construction, but cat-5's whole point is that abstention is CORRECT. The
  LME base prompt's abstention permission is load-bearing here (it IS the cat-5
  pass behavior), so the canonical posture has **no blanket answerability
  clause**, and cat-5 is judged with the LME `_abs` abstention judge.
  `--answerable-clause` exists as an A/B lever but is **label-leaky** — it
  conditions the prompt on the very label cat-5 measures, something LME itself
  never does (its `_abs` questions get the same answer prompt as everything
  else) — so it can never be canonical; the report brands runs that use it.

---

## 3. Mapping decisions

- **Speakers.** HyMem is single-user: `speaker_a` = `[user]` (flip with
  `--user-speaker b`), partner = `[assistant]`. One `log_messages` call per
  LoCoMo session — session boundaries preserved (19-32 HyMem sessions/conv).
- **Dates are REAL.** `session_N_date_time` ("1:56 pm on 8 May, 2023") parses
  with `%I:%M %p on %d %B, %Y` → `created_at` stamps; verified monotonic across
  all 10 conversations, no synthesis needed (unlike MSC). Unparseable → previous
  + 1 day, ordering preserved. `question_date` = last session's date (the "now"
  for relative-date math).
- **Label-routing only where LME has precedent** (its canonical run reads the
  oracle question type for ability routing): cat 2 → ability `"TR"` (TR prompt +
  temporal_events chronology + question_date — the whole time-anchor stack);
  cat 3 → `permissive_default` (the D4 posture: open-domain questions require
  world-knowledge bridging *by construction*; abstention guard intact) + a short
  commit-to-best-inference style clause. Cats 1/4/5 → default prompt. **No MR
  routing anywhere** ([[project_mr_filter_killed]] — the MR user-only filter is
  a suppress-filter and stays dead).
- **Judges** (all reused verbatim from LME): cat 1 → `multi-session`, cat 2 →
  `temporal-reasoning` (off-by-one-day tolerance fits LoCoMo's date golds),
  cats 3/4 → `single-session-user`, cat 5 → the `_abs` abstention judge with a
  constructed explanation that NAMES the trap answer so the judge fails a reader
  that fell for the premise swap.
- **question_type strings** `multi-hop / temporal / open-domain / single-hop /
  adversarial_abs` flow through `compute_scores` (per-category table for free)
  and `compute_abstention_scores` (the `_abs` suffix keeps the answerable-vs-
  abstention split working untouched).

---

## 4. Diagnostics (stronger than MSC's — evidence-annotation-based)

LoCoMo annotates the gold turns (`evidence` dia_ids), so Step-0 diagnostics use
the **exact gold turn text**, not MSC's answer-text heuristic (lexical τ=0.6
remains only to absorb the 600-char context truncation):

- `gold_in_context` — ALL evidence turns surfaced to the reader (multi-hop needs
  every hop; partial surfacing tracked in `evidence_in_context_frac`).
- `gold_in_pool` — all evidence retrievable pre-truncation → miss decomposition
  splits retrieval vs ranking/cut vs synthesis/judge, answerable cats only
  (cat-5 "evidence" is the trap source, not a gold location).
- `gold_distance` — sessions back to the FARTHEST-back evidence turn, reported
  in buckets (1 / 2-3 / 4-7 / 8-15 / 16+) since LoCoMo distances run 1..32 —
  the long-range version of MSC's E1 table.
- Per-conversation accuracy table (10 rows) — a cross-conversation contamination
  / difficulty check, and profile-tier population stats.

---

## 5. Operational shape (where LoCoMo differs from MSC)

Ingest+dream over 19-32 sessions is the expensive step and is **per
conversation, not per question** (10 stores serve 1986 questions):

- `--db-dir DIR` persists per-conversation stores (`DIR/conv-26/hymem.sqlite`)
  and **reuses them on later runs** (skips ingest+dream) — QA/prompt iterations
  don't re-pay ingestion. The MSC arc took 5 scored passes; on LoCoMo each
  re-ingest avoided saves ~10 conv × ~25 dreams of LLM calls. A reused store is
  only valid for the same core/schema — `--fresh` (or clearing the dir) after
  core changes, and any core-change regression run MUST be `--fresh`.
- `--workers N` parallelizes **conversations** (each owns its store; ≤10
  useful). QA within a conversation stays sequential (one SQLite connection).
- `--sample N` = global seeded QA cap (random ≈ category-proportional at
  n≥100); `--categories 1,2,4` / `--convs conv-26,…` for targeted slices.
- `--sim` is fully offline including the report (local `compute_scores`
  fallback — importing the LME module pulls in `requests` at module level).

Suggested first box run (before any full-1986 spend):

```
python benchmarks/locomo_adapter.py --data benchmarks/data/locomo10.json \
  --sample 200 --seed 0 --workers 10 --db-dir /tmp/locomo_dbs \
  --answer-extra-body '{"thinking":{"type":"disabled"}}' \
  --judge-extra-body  '{"thinking":{"type":"disabled"}}' \
  --out /tmp/locomo_results_v1.json
```

then the miss decomposition decides the next lever, per the LME/MSC discipline.

---

## 6. Levers queue (pre-registered, so post-hoc tuning stays honest)

Verdicts as of 2026-07-28. "Net" = flip-script net questions on n=200; compare
against the §8 churn floor (net ≈ ±4, ~10 questions moving) before reading a
delta as a signal.

- **L1 `--graph-multihop` on cat-1** — ran, null (31.6% cat-1 OFF vs ON). **Not
  closed: instrument mismatch, not a verdict.** Multi-hop feeds `graph_facts`
  rendered as three-word `"s p o"` triples, which cannot clear the `_lex_match`
  τ=0.6 evidence check against full turn text — the A/B had no path to a
  positive on the diagnostic. Re-run needs a triple-aware surfacing check.
- **L2 `--name-prefix`** — UNRUN, now the top candidate: it targets the 13-miss
  retrieval residual (query→turn paraphrase gaps) that survives at canonical.
  Ingest-side → requires `--fresh`, so it re-pays ingest+dream.
- **L3 `--answerable-clause`** — label-leaky, never canonical (§2); unrun.
- **L4 `--dream-per-session`** — unrun.
- **L5 `--embeddings`** — **VERDICT REVERSED.** First read (52.3% at the default
  aperture) called it dead; it was bottlenecked by a fixed `[:30]` cut discarding
  the pool gain. At 3× aperture it is **+6.0pp (59.6 → 65.6)** and part of
  canonical. The near-miss: this is the same mistake as the BEAM June regression
  — judging a pool-widening lever through a binding downstream cut.
- **L6 retrieval aperture + context budget** (added after the runs; the largest
  lever on this corpus). The MSC-sized constants (15/10/10, rerank pool 20)
  surface ~2% of a 369-689-turn LoCoMo history. `--top-k` ×3 = **+6.6pp**;
  ×5 = **−3.3pp, within the churn floor → no dilution demonstrated** (§8).
  `--max-context-chars` must scale WITH the aperture: at the 8000 default the
  render loop truncated the extra evidence away and it scored as *synthesis*
  loss, which is what made the first "reader is drowning" read wrong. 24k = **+4.6pp**.
- **L7 reader thinking** — enabled at 3×/24k scored WORSE in every category. Even
  if the magnitude is inside the floor, there is no measured benefit, so OFF
  stays canonical (and it is the cheaper setting).

## 7. Risks / honest caveats

- **Contamination.** locomo10 is public since early 2024 and heavily
  re-benchmarked; readers may know it. The per-conversation accuracy table and
  the retrieval-OFF floor (`--no-dream` + tiny `--top-k`) are the checks.
- **Judge frame.** LoCoMo's official metric is F1 (+ their own LLM-judge
  variants); ours is the LME yes/no judge. Numbers are comparable WITHIN the
  HyMem triad (that's the point — one frame across LME/MSC/LoCoMo), NOT against
  published LoCoMo tables. Never quote ours next to Mem0/Zep's without the
  caveat.
- **Cat-3 is tiny** (96 total; ~10 in a 200-sample) — per-category swings there
  are noise, same discipline as [[project_lme_variance_band]].
- **Single-user modeling.** Facts about the partner live only in `[assistant]`
  raw turns (no profile tier) — expect a user/partner asymmetry on cats 1/4;
  the perspective clause makes it attributable, not invisible.
- **License.** CC BY-NC 4.0 — local benchmark use is fine; the data file stays
  gitignored, redistribution is an explicit decision not a default.

## 8. The churn floor (MEASURED — read every delta against it)

Two runs of the **identical** config, differing only by `--dump-context` (which
records the rendered string and cannot change an answer), scored 70.2% and 67.5%
answerable: **7 broken / 3 fixed, ~10 of 200 questions moving, net −4.**

That is the floor. It is not a temperature setting — answer generation and
judging are both already at `temperature=0.0`
(`longmemeval_adapter.py:978`, `:1059`); this is server-side nondeterminism at
temp 0 and cannot be tuned away client-side.

Consequences, all load-bearing:

- **A single A/B cannot resolve anything smaller than ~net 5 on n=200.** The 5×
  aperture arm (net −5, 9 broken / 4 fixed) is one question past the floor — its
  "dilution confirmed" reading does not survive. Dilution remains *unobserved on
  this corpus at any setting*, not disproven.
- **Two runs of the same config are not interchangeable as an A arm.** A
  canonical-vs-replicate baseline mix-up put 2.0pp of drift inside one reported
  delta. Flip comparisons must name which file is A.
- **`locomo_flip.py` is the instrument, not the accuracy line** — but its counts
  need the floor subtracted too. Rule of thumb for calling a lever real on
  n=200: broken count ≥ 2× floor (≥14) **or** net ≥ |8|, ideally with a
  mechanism visible in the four-surface bucket migration.
- **Not a LoCoMo property — MEASURED on MSC too.** Per-question churn is ~5% here
  and **4% on MSC** (replicate 2026-07-28: 3 fixed / 1 broken, 84.0 → **86.0**),
  matching the ~4-questions/category band [[project_lme_variance_band]] recorded
  on LME. Two consequences for the MSC baseline: the churn *rate* is comparable
  across the triad, but at n=100 it buys a **wider** pp band (2σ ≈ ±4pp vs ±3.2pp
  at n=200 — sd ≈ √(p/n)), and 84.0 is now known to be one draw, not a center.
  Two draws put the mean near **85 ± ~1.4pp**, so an MSC regression gate written
  as "≥ 84.0" passes a no-op change about half the time. Restate it as a band.
- **A floor measured from one replicate is itself one draw.** MSC's 4 movers
  carry ≈ ±2 of counting noise; 4% and 5% are not distinguishable at this number
  of reruns. Treat both as "~5%", not as a ranking of the two benchmarks.
- **Splitting reader churn from judge churn — BUILT.**
  `locomo_adapter.py --rejudge RESULTS.json` re-judges a stored `--out` file with
  the same judge, no ingest and no re-answering, and writes a flip-compatible
  copy. Every flip it reports is judge nondeterminism with the reader held
  byte-identical; `locomo_flip.py` detects that case and relabels its
  "dilution" line accordingly. If the judge owns most of the floor, majority-of-3
  judging shrinks it for all three benchmarks at once. Cat-5 golds round-trip
  through `_gold_for_judge()` (shared with the live path) so the re-judge sees
  the same abstention explanation, not a reconstruction — otherwise the run would
  measure prompt drift. LME's own `--rejudge`
  (`longmemeval_adapter.py:1901`) is envelope-shaped (`{config, per_question}`,
  `hypothesis`) and cannot read LoCoMo's bare list, hence the separate path.
