# Track A (multi-hop) — box runbook: what to do next

**Audience:** whoever runs the Hermes box. Everything below is copy-pasteable
from the repo root. The feature is already built and default-OFF; this runbook
takes it through its two gates (**G-A1** recall probe → **G-A2** LME
non-regression) and, on pass, flips it on.

## State (already done, nothing to build)
- Feature: `hymem/query/augment.py` (`_multihop_edges` + Source 4) +
  `hymem/config.py` (`graph_multihop_{enabled=False,max_hops=2,decay=0.5,min_score=0.05}`).
  **Default OFF** — no behaviour change until you flip it.
- Synthetic ground-truth: `tests/test_multihop.py` (10 tests). Sanity-check:
  ```bash
  python -m pytest tests/test_multihop.py -q          # expect 10 passed
  ```
- Recall probe harness: `benchmarks/multihop_probe.py` (+ `multihop_probe_example.json`).
- LME guard arm: `benchmarks/longmemeval_adapter.py --graph-multihop` (new flag;
  `--help` lists it and the three knob overrides).

## The four remaining steps (in order)
| Step | What | Gate | Cost |
|------|------|------|------|
| A | Build the labeled probe set | — (human input) | manual |
| B | Run the recall probe | **G-A1** | seconds |
| C | Sweep the Pareto knee | — (tuning) | minutes |
| D | Full LME guard, OFF vs ON | **G-A2** | 2 runs |
| E | Decide + flip | — | one line |

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

### Path 1 (recommended for the real G-A1) — mine + introspect a built store

This is what makes G-A1 mean more than the pytest: real edges, real canonical
forms, real graph density. Use an existing dreamed LME/BEAM store (the ones your
LME runs build) or build one.

1. **Pick a store** with dreamed edges. Any `hymem.sqlite` from a full-dream LME
   run works. Confirm it has edges:
   ```bash
   sqlite3 STORE.sqlite "SELECT COUNT(*) FROM knowledge_graph WHERE status='active';"
   ```
2. **Shortlist multi-hop questions.** From the LME set, pick multi-session
   questions whose answer plausibly needs chaining two facts across sessions via
   *different* predicates (e.g. "where is the project X works on deployed?" →
   `X —part_of→ P —deploys_to→ where`). ~30–50 of them.
3. **Find the canonical seed + bridge for each** by inspecting the store. For a
   candidate seeded on entity `E`:
   ```bash
   # 1-hop neighbours of E (these are Source 1 — NOT bridges):
   sqlite3 STORE.sqlite "SELECT subject_canonical,predicate,object_canonical
     FROM knowledge_graph WHERE status='active'
       AND (subject_canonical='E' OR object_canonical='E');"

   # 2-hop bridges: edges incident to a neighbour N but NOT to E.
   # (replace N with each neighbour from the query above)
   sqlite3 STORE.sqlite "SELECT subject_canonical,predicate,object_canonical
     FROM knowledge_graph WHERE status='active'
       AND (subject_canonical='N' OR object_canonical='N')
       AND subject_canonical!='E' AND object_canonical!='E';"
   ```
   The answer-relevant row from the second query is your `bridge`; `E` is the
   `seed`. Record verbatim (canonical strings, exact predicate).
4. **Build the control set** (equal size): questions answered by a *direct* edge
   of a seed — `bridge` = a row from the first query (incident to `E`).
5. Assemble everything into `SLICE.json` (same shape as the example). Because you
   run against the real store, **omit the `edges` block** and pass `--store`.

> **Labour-saver:** if hand-labeling 60–100 items is too much, ask the assistant
> to build the **miner** — a script that pre-filters LME/BEAM multi-session items
> to multi-hop candidates and pre-fills seeds/bridge suggestions from the store,
> so you only verify a short list. Not built yet (offered).

### Path 2 (fast sanity, lower fidelity) — hand-authored fresh-seed chains

Write ~40–60 chains directly in an `edges` block + `items` (as in the example
JSON), drawn from real LME/BEAM content but using canonical predicates. No store
needed (`multihop_probe.py` seeds a temp store from `edges`). This is basically a
bigger pytest — use it to smoke the harness or if a real store isn't handy, but
prefer Path 1 for the decision.

---

## Phase B — G-A1 recall read

```bash
# Path 1 (real store):
python benchmarks/multihop_probe.py --probe SLICE.json --store STORE.sqlite --verbose
# Path 2 (fresh-seed):
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

**Files:** feature `hymem/query/augment.py` + `hymem/config.py` · probe
`benchmarks/multihop_probe.py` (`--json` for sweeps) · guard flag in
`benchmarks/longmemeval_adapter.py` · ground truth `tests/test_multihop.py` ·
design `additional_planning.md` §Idea A.
