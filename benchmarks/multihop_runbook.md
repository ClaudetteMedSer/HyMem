# Track A (multi-hop) — box runbook: what to do next

**Audience:** whoever runs the Hermes box. Everything below is copy-pasteable
from the repo root. The feature is already built and default-OFF; this runbook
takes it through its two gates (**G-A1** recall probe → **G-A2** LME
non-regression) and, on pass, flips it on.

> ### ⚠️ UPDATE 2026-07-26 — read this first; it changes what the box should do
> A box run took the miner→probe path on 40 LME MR/TR questions with **healthy
> dreams** (200+ edges/question) and got **G-A1 FAIL, 0/4 bridges**. Root cause is
> now understood and **fixed in the feature**, and it changes the plan below:
> - **LME cannot validate Track A.** LME's graph is a personal-memory **star**
>   centred on `user`; its only 2-hop paths run through that super-hub
>   (`road_trip ← user → driving_trip`), which is a *hub-mediated non-bridge*, true
>   of every pair of things the user mentioned. Genuine intermediate-entity bridges
>   are ~0 in LME regardless of dream quality. **Do not spend box time hand-labeling
>   an LME multi-hop slice — the substrate isn't bridge-shaped.**
> - **Hub guard added** (`graph_multihop_hub_degree_max=32`): super-hubs are reached
>   but never expanded, so on a star the feature is provably **inert** (ON == OFF),
>   and genuine low-degree bridges still fire.
> - **The mechanism gate is now CLOSED locally**, no box/LME needed:
>   ```bash
>   python benchmarks/multihop_probe.py --probe benchmarks/multihop_genuine_bridges.json --verbose
>   # expect: multihop 0→100%, control held, ── G-A1 advisory: PASS ──
>   ```
> - **The box's remaining role is substrate-scoped to Hermes.** Phases A–E below
>   apply **only to a Hermes production graph** (which has real intermediate
>   entities), not LME. On LME there is nothing left to gate.

## State (already done, nothing to build)
- Feature: `hymem/query/augment.py` (`_multihop_edges` + Source 4 + `_active_degrees`
  hub guard) + `hymem/config.py`
  (`graph_multihop_{enabled=False,max_hops=2,decay=0.5,min_score=0.05,hub_degree_max=32}`).
  **Default OFF** — no behaviour change until you flip it.
- Synthetic ground-truth: `tests/test_multihop.py` (14 tests, incl. 4 hub-guard).
  Sanity-check:
  ```bash
  python -m pytest tests/test_multihop.py -q          # expect 14 passed
  ```
- Recall probe harness: `benchmarks/multihop_probe.py`. Substrates:
  `multihop_genuine_bridges.json` (**canonical local G-A1**, PASSES today) and
  `multihop_probe_example.json` (schema demo).
- Probe-set miner: `benchmarks/multihop_miner.py` (pre-fills the labeled set from a
  run's questions + a store, so Phase A is a verify pass, not authoring).
- LME guard arm: `benchmarks/longmemeval_adapter.py --graph-multihop` (new flag;
  `--help` lists it and the three knob overrides).

## The four remaining steps (in order)
| Step | What | Gate | Cost |
|------|------|------|------|
| A | Build the labeled probe set (miner pre-fills → you verify) | — (human input) | mostly auto |
| B | Run the recall probe | **G-A1** | seconds |
| C | Sweep the Pareto knee | — (tuning) | minutes |
| D | Full LME guard, OFF vs ON | **G-A2** | 2 runs |
| E | Decide + flip | — | one line |

---

## Mandatory Phase-1 extraction canary

LME, BEAM, LoCoMo, and MSC now run one deterministic extraction preflight per
pending run configuration, before any scored conversation/question store is
opened. The probe uses a dedicated `OpenAICompatibleClient` with the exact
memory-pipeline endpoint, model, and thinking policy, then calls the production
`extract_chunk()` path with valid `claim-source-v2` records. Canary policy v17
(`hymem-phase1-extraction-canary-v17`) is pinned to extraction prompt v20 and
binds the terminal clean-empty recovery policy as well as the block-aware,
canonical-Markdown-table-safe source-split v9 policy. V9 keeps exact content
and offsets while allowing a heading- or colon-introduced canonical table to
split with separately labelled same-source prelude plus header context. It also
gives a right-hand prose continuation an exact, bounded preceding same-source
window so a claim crossing a sentence or paragraph cut remains visible. That
window is context-only, applies to a bounded fragment prefix, adds no calls,
and cannot independently own an extracted item. The policy, effective HyMem
config, report, checkpoint, and registry all carry the same SHA-256 contract
identity, mechanically derived from the actual prompt bytes and executable
validation/recovery behavior; a hand-edited version label cannot bless drift.

The deterministic fixture
(`hymem-phase1-context-paths-four-leaf-v7`) has two source records over 5,000
characters each, so production prepartition must create exactly four initial
fragments. The
first record is an ATX-heading-introduced canonical table. Its only supported
row is in the later continuation, where authoritative content contains neither
the heading nor header; the actual provider request must carry the exact,
separately labelled prelude and canonical header context. The second record
separates a preference subject/predicate clause from its PostgreSQL object
clause at a proven paragraph boundary; only the later fragment's exact bounded
`source_boundary_context` completes it. Request/response-path evidence records
both non-self-contained continuations, their exact contexts, proves each exact
claim was emitted on its context-bearing continuation and never another leaf,
records both source IDs, and reconciles primary/empty/omission request counts
with total completions.

The prose record also contains clearly labelled non-factual list and fenced-code
controls. Both blocks contain concrete schema-like decoy tokens, but split the
relation labels so they neither state a relationship nor trigger cue recovery.
A pass requires each complete block to reach the provider intact at least twice,
zero request-fragment boundaries inside either span, and no decoy output. The
policy-bound `hymem-phase1-markdown-atom-split-probe-v1` additionally invokes
the real source splitter on each exact control without a provider; both must
remain atomic. These checks exercise block protection without adding provider
spend. The probe otherwise uses the same Phase-1 prompt, source-ID contract,
JSON/item validators, non-empty omission verification, and retry behavior as
dream indexing.

The two expected claims are fixed and manifested: the table-continuation claim
`(HyMem Canary Relay:service, deploys_to, Fly.io:platform, +1)` at source
message ID `9271604311`, and the boundary-spanning prose claim
`(Avery Boundary Canary:person, prefers, PostgreSQL:database, +1)` at source
message ID `9271604312`. Empty service/platform cells and inert hashes supply
density without creating additional relationships. The report carries the
fixture hash, initial-leaf count, exact execution path, and two canonical
matched-claim evidence records, not only a success label. A pass means exactly
those two final triples, zero markers, and zero coalesced duplicates; any
additional schema-valid output fails closed. The run aborts before ingest if
either bounded pass fails, validation fails, or either exact supported
claim/type is absent or misattributed after empty/omission verification. The
canary never constructs a HyMem database and its client is closed after the
probe, so it cannot contaminate a benchmark store.

The normal four-leaf path costs eight logical completions: four primary calls,
two empty verifications, and two omission verifications. Recovery is limited to
24 logical calls and 72 provider attempts under the shipped three-attempt
transport policy; the production per-chunk limit remains 96 logical calls.

Canary attempts/tokens are recorded separately under `extraction_canary` with
`usage_accounting=excluded_from_scored_usage_dedicated_memory_pipeline_client`;
they are not added to reader, judge, or memory-pipeline scored usage. LME and
BEAM persist this in their execution segment. MSC/LoCoMo include it in recall
rows; MSC recurrence also writes `<out>.extraction-canary.json`. Offline
`--sim` and the existing `--no-dream` development path print and record
`skipped_non_comparable`; neither is evidence that extraction works. A terminal
checkpoint/resume with no pending indexing work records the distinct comparable
`not_run_no_pending` zero-call state and does not spend another canary. Shared
validation rejects unknown/missing fields, non-finite or inconsistent usage,
partial/duplicate claim evidence, and a canary model, endpoint, thinking mode,
or effective request body that differs from the memory pipeline. BEAM and LME
repeat this validation when strict artifacts enter their registries; MSC and
LoCoMo validate before work and bind the report to emitted rows/provenance.

Do not start an expensive run if this line is absent or not `PASS`:

```text
Extraction canary: PASS (N provider attempt(s); excluded from scored memory-pipeline usage)
```

---

## Phase A — Build the labeled probe set

The probe measures **bridging-edge recall@8**: does the edge that 1-hop
retrieval MISSES (the hop-2/hop-3 bridge) appear in `graph_facts[:8]` once
multi-hop is on. To measure that you need a JSON of labeled items. Schema and a
worked example live in `benchmarks/multihop_probe_example.json` and the
`multihop_probe.py` module docstring. Each item:

```json
{"id": "...", "set": "multihop"|"control", "route": false,
 "question": "...", "seeds": ["<canonical entity>"], "bridge": ["s","p","o"]}
```

- **`set`**: `"multihop"` = answer needs a ≥2-hop chain; the `bridge` is the far
  edge. `"control"` = a 1-hop direct hit (bridge is an edge *incident to a
  seed*) — this set guards the additive invariant (must NOT drop).
- **`seeds`**: canonical entities the question anchors on (the direct match).
- **`bridge`**: the exact `(subject_canonical, predicate, object_canonical)`
  triple that counts as a recall hit. **It must match the store's canonical form
  and use a canonical predicate** (list at the bottom of this file).
- **`route: false`** forces the entity-anchored fallback path where multi-hop
  earns its keep. Leave it `false` unless you are deliberately testing a routed
  question (then Source 3 may already fetch the bridge and multi-hop shows no
  lift there — which is correct).

Target **~60–100 items total**, roughly half multihop / half control.

### Path 1 (recommended) — mine, then verify

The **miner** (`benchmarks/multihop_miner.py`) does the labeling for you and
leaves you a *verification* pass. It reuses the feature's own traversal to
propose bridges, so a proposed bridge is exactly what Source 4 would fetch, and
uses the gold answer to auto-sort each question into `multihop` / `control`.
(Using gold here is legitimate — it LABELS the ground-truth set; the probe still
measures recall with the feature blind to the label.)

**Two miner modes — pick one:**
- **`--lme-data` (per-question, recommended, LLM-bound):** rebuilds and dreams
  *each question's own haystack* to completion, then mines it. Small store per
  question → one/few dream cycles fully drain it, so it **sidesteps the
  dream_budget under-dream entirely** and is faithful to how LME retrieves
  (isolated per-question store). Emits a self-contained `edges` block → probe with
  **no `--store`**. This is the path that avoids step 1's combined-store hazard —
  jump to step 2b.
  ```bash
  python benchmarks/multihop_miner.py \
    --lme-data <longmemeval.json> --types multi-session --limit 40 \
    --dream-model deepseek-v4-flash --out SLICE.json
  # per-question dream stats print per item; watch the "avg edges/store" line —
  # if it's tiny, the dream LLM isn't extracting (check thinking-disable).
  python benchmarks/multihop_probe.py --probe SLICE.json --verbose   # NO --store
  ```
  Cost: one independently converged store per question (at least ~40 dream
  cycles for a 40-question slice; more when a bounded tier leaves durable work).
  Requires the box's extraction LLM (deepseek-v4-flash, thinking disabled).
- **`--store` (mine an existing dreamed store, LLM-free, seconds):** faster, but
  you must supply a store already dreamed to completion (step 1 below).

1. **[store mode only] Get a store** with dreamed edges for these questions. LME
   haystacks are
   per-question, so there is no single ready-made store — build one combined
   store: ingest the selected questions' sessions into one `hymem.sqlite` and
   dream it (reuse the adapter's ingest path), **or** point at a persistent
   Hermes store.
   > **CRITICAL — dream to completion.** `dream()` processes at most
   > `cfg.dream_budget` (=50) Phase-1 chunks per call. Recursive extraction is
   > additionally bounded by `cfg.dream_extraction_provider_attempt_budget`
   > (=200 actual provider requests; 0 disables this ceiling). Reaching either
   > bound with work pending reports `report.budget_exhausted = True`; the
   > independently capped short-turn baseline does the same when its positive
   > allowance is spent with actionable work left (0 intentionally disables
   > that tier). The call
   > bound is separately visible as
   > `report.extraction_provider_attempt_budget_exhausted`. It never interrupts
   > a chunk, so the shipped client's sole three-attempt layer (SDK retries
   > disabled) can overshoot by at most one chunk's 288-HTTP-request hard
   > envelope. Digest/profile/fact calls do not count
   > against it and their later-session tails still advance. A 100k-message
   > combined store is thousands
   > of chunks, so a single `dream()` call consolidates ~1% of it (~20 episodes,
   > ~37 edges) — which produces a **false 0% G-A1** (nothing to bridge). Loop
   > until the **durable** backlog is drained. A non-exhausted individual report
   > is not proof of completion: a failed attempt can remain pending. Use the
   > shared bounded convergence helper, which also rejects lock stalls,
   > timeouts, quarantine, terminal source loss, and lossless coverage whose
   > integrity could not be established. This state is non-pending and fails
   > immediately: resolve a transient cause or repair/rebuild persistent
   > artifact damage, then start a new bounded convergence call. One complete
   > successful source walk clears its durable health row.
   > ```python
   > from benchmarks.strictness import converge_indexing
   >
   > indexing = converge_indexing(
   >     dream_hy.dream,
   >     status=dream_hy.dream_status,
   >     max_cycles=1000,
   >     timeout_s=3600,
   >     require_healthy=True,
   > )
   > ```
   > Current strict evidence requires the coherent
   > `hymem-dream-status-v5` + `hymem-benchmark-indexing-status-v3`
   > snapshot; LME serializes it through `hymem-lme-indexing-summary-v4`.
   > Prefer repeated bounded cycles over a huge one-shot chunk budget, which no
   > longer predicts provider cost by itself.
   > You only need ~60–100 probe items, so a **bounded ~30–40-question subset**
   > dreamed to completion is enough and far cheaper than all 248.
   Confirm the store is actually dreamed (hundreds+ of edges, many subjects — not
   tens):
   ```bash
   sqlite3 STORE.sqlite "SELECT COUNT(*) edges,
     COUNT(DISTINCT subject_canonical) subjects
     FROM knowledge_graph WHERE status='active';"
   ```
   Even fully dreamed, LME is personal-memory (user-centric), so genuine
   cross-entity bridges are a minority — a small multihop set is expected; the
   real G-A1 substrate is a Hermes production store.
2. **[store mode] Mine** (LLM-free, seconds). `--from` takes a results JSON (uses
   its `per_question`: question + gold `answer` + `question_type`) or a bare
   `[{id,question,answer}]` list:
   ```bash
   python benchmarks/multihop_miner.py \
     --from ~/.hermes/benchmarks/<run>.json \
     --store STORE.sqlite --out SLICE.json
   # prints: scanned N → multihop M, control C, dropped D (no-seed …)
   ```
   Both modes mine MR+TR types by default (`--types` to change), enumerate bridges
   at `--max-hops 3 --min-score 0.01` (broad — the probe/sweep tunes later), and
   write probe-ready items with `_`-prefixed hints (`_gold`, `_hop`,
   `_answer_overlap`, `_alt_bridges`).
3. **Verify** `SLICE.json` by hand (both modes) — this is the human step, now short:
   - For each `multihop` item, confirm `bridge` is the answer-bearing edge. If a
     different edge is right, swap in one from `_alt_bridges`; if none fit (the
     store lacks the fact, or the question isn't really multi-hop), delete the item.
   - Spot-check `control` items: the direct edge should plainly answer the question.
   - `dropped` questions (no edge overlapped the gold, or no seed matched) aren't
     emitted — add any real ones by hand using the SQL in Path 2 if you want them.
   - The `_`-prefixed fields can stay; `multihop_probe.py` ignores unknown keys.
4. Because you mined against the real store, **run the probe with `--store` and
   omit any `edges` block**.

### Path 2 (manual add / fix, or when a store isn't handy)

To add an item by hand (or check what the miner dropped), inspect the store
directly. For a candidate seeded on entity `E`:
```bash
# 1-hop neighbours of E (Source 1 — NOT bridges; these are control edges):
sqlite3 STORE.sqlite "SELECT subject_canonical,predicate,object_canonical
  FROM knowledge_graph WHERE status='active'
    AND (subject_canonical='E' OR object_canonical='E');"

# 2-hop bridges: edges incident to a neighbour N but NOT to E.
sqlite3 STORE.sqlite "SELECT subject_canonical,predicate,object_canonical
  FROM knowledge_graph WHERE status='active'
    AND (subject_canonical='N' OR object_canonical='N')
    AND subject_canonical!='E' AND object_canonical!='E';"
```
The answer-relevant row from the second query is the `bridge`; `E` is the `seed`.
Record verbatim (canonical strings, exact predicate).

### Path 3 (offline sanity, lower fidelity) — hand-authored fresh-seed chains

Write ~40–60 chains directly in an `edges` block + `items` (as in the example
JSON), drawn from real LME/BEAM content but using canonical predicates. No store
needed (`multihop_probe.py` seeds a temp store from `edges`). This is basically a
bigger pytest — use it to smoke the harness or if a real store isn't handy, but
prefer the mined path (1) for the decision.

---

## Phase B — G-A1 recall read

```bash
# mined / manual against the real store (Paths 1–2):
python benchmarks/multihop_probe.py --probe SLICE.json --store STORE.sqlite --verbose
# fresh-seed (Path 3, edges block in the JSON):
python benchmarks/multihop_probe.py --probe SLICE.json --verbose
```

**Read the output — G-A1 passes iff all three hold:**
1. **multihop recall RISES** off → on (the mechanism works on real bridges).
2. **control recall HELD** (Δ ≥ 0 — the additive invariant; multi-hop must not
   displace direct hits). A drop here is a real bug — stop and report.
3. **p95 latency on < 1.5× off** (or "below 1ms floor — not gated": a
   sub-millisecond lookup can't blow a budget; on a real store the number is
   meaningful).

The script prints a `── G-A1 advisory: PASS/FAIL ──` banner — **advisory**; the
final call is yours reading the numbers.

**If multihop recall does NOT rise:** most likely the bridges are deeper than
`max_hops=2` or pruned by `min_score`. Don't conclude failure yet — that's what
Phase C tunes. Re-run a couple of items with `--max-hops 3 --min-score 0.01
--verbose` to confirm the bridge is reachable at all before sweeping.

---

## Phase C — Sweep the Pareto knee

Grid: `max_hops ∈ {2,3}` × `decay ∈ {0.4,0.5,0.6}` × `min_score ∈ {0.02,0.05,0.1}`.
Use `--json` (pure JSON on stdout, logs on stderr) and collect:

```bash
: > sweep.jsonl
for H in 2 3; do for D in 0.4 0.5 0.6; do for M in 0.02 0.05 0.1; do
  python benchmarks/multihop_probe.py --probe SLICE.json --store STORE.sqlite \
    --max-hops $H --decay $D --min-score $M --latency-reps 100 --json \
    2>/dev/null >> sweep.jsonl
done; done; done

# Rank by multihop recall, then by lowest p95, keeping only control-held rows:
python - <<'PY'
import json
rows = [json.loads(l) for l in open("sweep.jsonl")]
ok = [r for r in rows if r["gate"]["control_held"] and r["gate"]["latency_ok"]]
ok.sort(key=lambda r: (-r["multihop"]["recall_on"], r["latency_ms"]["on_p95"]))
for r in ok[:10]:
    c = r["config"]
    print(f"hops={c['max_hops']} decay={c['decay']} min={c['min_score']:<5} "
          f"mh_recall={r['multihop']['recall_on']:5.1f}% "
          f"ctl={r['control']['recall_on']:5.1f}% p95_on={r['latency_ms']['on_p95']:.2f}ms")
PY
```

**Pick the knee:** the config with the best multihop recall that still holds the
control set and the latency budget. **Prefer `max_hops=2`** — if depth 3 only
adds a point or two of recall for extra latency, ship 2 (cheapest, most of the
gain — the plan's expected ship). Record the chosen `(max_hops, decay, min_score)`.

---

## Phase D — G-A2 full LME non-regression guard

**This is a PAIRED test: run the identical LME config twice, OFF then ON, same
seed/reader/judge, differing only by `--graph-multihop`.** Multi-hop needs
dreamed edges, so **full dream** (do NOT pass `--no-dream`).

> **Reader/judge note (post-deprecation).** The banked **68.4%** baseline was
> *answered* by the now-deprecated `deepseek-chat` and only re-judged under
> v4-flash — you cannot reproduce its answer side, so it is a **historical
> reference, not the paired control**. The paired control is your own fresh OFF
> arm below, run under the go-forward reader. Use `deepseek-v4-flash` (thinking
> disabled) for BOTH answer and judge so the pair is internally matched.

First copy the canonical flags from a baseline run's metadata so you mirror
everything except the lever:

```bash
jq '.metadata' ~/.hermes/benchmarks/<canonical-baseline>.json   # read; mirror the flags below
```

```bash
COMMON="--sample 0 --seed 0 --workers 8 \
  --answer-model deepseek-v4-flash --answer-extra-body {\"thinking\":{\"type\":\"disabled\"}} \
  --judge-model  deepseek-v4-flash --judge-extra-body  {\"thinking\":{\"type\":\"disabled\"}} \
  --hymem-model deepseek-v4-flash --hymem-thinking disabled \
  <every other flag copied verbatim from the baseline metadata: embeddings / \
   permissive-default / rerank / value-supersession / etc.>"

# OFF arm (paired control):
python benchmarks/longmemeval_adapter.py $COMMON

# ON arm (swept knobs from Phase C):
python benchmarks/longmemeval_adapter.py $COMMON \
  --graph-multihop --graph-multihop-max-hops <H> \
  --graph-multihop-decay <D> --graph-multihop-min-score <M>
```

The ON run records `graph_multihop` + `graph_multihop_knobs` in its metadata, so
it is self-describing.

**G-A2 passes (non-regression only — NOT a tuning signal) iff:**
- **overall ON ≥ OFF − 1pp** (within the ±category noise band), AND
- **no category worse than −5pp** vs the OFF arm, AND
- **MS not below its floor** (MS is where multi-hop could plausibly help; it must
  at minimum not regress).

Multi-hop is additive and gated below direct hits, so the expected result is
neutral-to-slightly-positive. A real regression means a bridge edge is crowding
the answer context — investigate before shipping.

---

## Phase E — Decide and flip

- **Both gates pass** → enable it. Two options:
  - *Production only:* leave the library default OFF and pass `--graph-multihop`
    (+ knobs) wherever Hermes runs the query path / set the config field in the
    Hermes config. Safest.
  - *New default:* in `hymem/config.py`, `graph_multihop_enabled: False → True`
    (and set the swept `max_hops/decay/min_score` as the new defaults); update
    the docstring with the flip date + gate evidence; run the full suite
    (`python -m pytest -q`) and fix any test pinning the old default.
  - Bank the G-A1 table + G-A2 deltas + chosen knobs in
    `additional_planning.md` §Idea A (STATUS block) and `benchmarks/longmemeval_roadmap.md`.
- **G-A1 fails** (recall doesn't rise even after the sweep) → the bridges LME
  needs aren't the cross-predicate kind this fills; bank as a dead lever in the
  roadmap, leave the code default-OFF (it's still valid for Hermes recall). Stop.
- **G-A2 fails** (LME regresses) → do NOT ship; record the regressing categories,
  leave default OFF. The recall win may still justify a Hermes-only enable, but
  not a default flip.

---

## Reference

**Canonical predicate vocabulary** (schema CHECK — `bridge`/`edges` must use
these): `uses, depends_on, prefers, rejects, avoids, replaces, conflicts_with,
deploys_to, part_of, equivalent_to, implements, contains, configured_with,
requires_version, runs_on, connects_to, generates, tested_by, owns, located_in,
participates_in, has_attribute`.

**Gate thresholds at a glance**
- G-A1: multihop recall@8 ↑ ; control recall@8 Δ ≥ 0 ; p95_on < 1.5× p95_off.
- G-A2: overall ON ≥ OFF−1pp ; no category < −5pp ; MS ≥ floor. Non-regression only.

**Files:** feature `hymem/query/augment.py` + `hymem/config.py` · miner
`benchmarks/multihop_miner.py` (pre-fills the probe set) · probe
`benchmarks/multihop_probe.py` (`--json` for sweeps) · guard flag in
`benchmarks/longmemeval_adapter.py` · ground truth `tests/test_multihop.py` ·
design `additional_planning.md` §Idea A.
