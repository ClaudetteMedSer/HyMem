# Full-S run stopped: failure census and offline diagnosis

## Scope and disposition

After the user asked for progress, nine of 500 questions had finished, all with
`indexing_failure:quarantined_extraction`; Q10 was in progress. The user approved
stopping and investigating. Only the benchmark adapter was interrupted with
SIGINT, after verifying its PID, exact executable source path, checkpoint,
working directory and process group. The detached supervisor recorded exit -2
at **2026-09-18 20:40:35 UTC**, after 20,624 seconds (5 h 43 m 44 s).
Both benchmark and supervisor processes are now absent. No restart, resume,
deployment, production-memory mutation or additional paid diagnostic occurred.

Run directory inside Hermes1:
`/home/node/.hermes/benchmarks/lme-full-20260918-5HQ61m`.
Host prefix: `/opt/stacks/hermes/instance1/home`.
Checkpoint, logs, frozen source/data and all ten question stores remain there.
The interrupted checkpoint still says `running`; the exit receipt is the
authoritative process disposition. Q10 is unscored and has one unfinished dream
audit row. Do not relabel this partial run complete or silently resume it.

Preserved receipt hashes:

- `results/checkpoint.json`:
  `40dbf12ba53f3a3380453c3d378b474ecbfd23029da7d798725655934831d12f`.
- `benchmark.log`:
  `e506cfec8b959a146f2fdb02aea9f316edb2bd7e696a91136c359f447dcc85ee`.

Read-only checks of all ten closed benchmark databases: SQLite quick-check OK,
zero foreign-key violations, no WAL files, and approximately 95 MB total. Exact
hashes and detailed metadata are in private remote `diagnosis-census.json`.
All 225 frozen source/config hashes still match the launch manifest. Hermes1,
Hermes2 and Hermes3 retain their previous start times; Honcho `/health` is 200/ok.

## What actually failed

| Questions | Terminal indexing gate | Durable evidence |
| --- | --- | --- |
| 1–6 and 9 | Fact quarantine | Ten sessions exhausted six attempts |
| 7 | Digest quarantine | Two sessions exhausted six attempts |
| 8 | Digest and chunk quarantine | Five digest sessions and one chunk |

`quarantined_extraction` is a shared benchmark reason covering chunks, digests,
profiles and facts. It does not mean nine chunk failures. None of the nine
finished questions reached the reader/judge. They remain score-zero failure
rows; the prior fatal failure-envelope exception did not recur.

The recorded usage for those nine finished attempts was 8,965 memory-pipeline
calls plus eight canary calls, excluding Q10's interrupted activity. No exact
dollar cost is established here. Despite terminal failures, the preserved dream
reports across the ten stores record 1,775 fact items and 5,314 triples. This is
not a blanket zero-output model failure.

## 1. Digest: confirmed output-contract mismatch

There are **769 `digest.summary_output_cap` warnings**, 756 in the nine finished
questions and 13 in Q10. The deployed prompt asks for an updated summary retaining
earlier and new information but does not state the validator's 500-character
limit. Longer summaries cause rejection of the whole digest and hold its cursor.

This is directly linked to all seven terminal digest quarantines:

- Six sessions had six consecutive summary-cap rejections.
- The seventh had one episode-item rejection and five summary-cap rejections.
- The cap-rejected summaries in these chains ranged from 502 to 684 characters.

The ten logged `digest.shape_failure` events are separate incidents, usually a
missing `procedures` key. They were not the terminal reason for these seven
quarantines. The current frozen extractor has no later local semantic verifier;
those uninstalled experiments cannot have caused this run's failures.

The retry policy shrinks source windows for every failure type:
12,000 → 6,000 → 3,000 → 1,500 → 750 → 375 characters, while preserving the prior
summary and keeping a 3,072-token response allowance. It neither explains the
500-character output cap nor specifically repairs the rejected summary.

Historical source references: `af6a615:hymem/dreaming/digest.py:93`, `:169`,
`:548`, `:602`; prompts at `hymem/extraction/prompts/__init__.py:564` and `:648`.
Independent historical-code synthetic controls verify the 500/501 boundary and
separate shape, summary-shape and item failures; no paid calls.

## 2. Facts: confirmed defects, incomplete retrospective attribution

The frozen facts prompt asks for a bare JSON array, while the transport selects
JSON-object mode. The parser accepts a bare array or exactly `{"facts": [...]}`,
but rejects additional wrapper keys. It also rejects more than eight facts,
fact text over 600 characters, over 64 entities, or entity text over 200
characters. Those capacity limits are not communicated in the deployed prompt.

All ten quarantined fact sessions exhausted the actual six-attempt policy;
their fact cursors remained at the start. No fact extraction/persistence/replay
exception or invalid-state event appears in this run's logs. Some affected
sessions are small: one consists of 99- and 669-character turns, another of
41- and 980-character turns. Large excerpts alone do not explain the pattern.

However, the frozen code collapses invalid envelopes, item validation, capacity
rejection and malformed/truncated JSON into `parse_failed=True`, without logging
or persisting the rejection subtype or raw output. Consequently, the exact
cause of each of these ten historical fact failures is **not recoverable from
the retained evidence**. The contract defects are proven; claiming all ten
were caused specifically by the item cap or envelope would overstate evidence.

Root independently ran three controls against the exact frozen runtime: eight
valid items accepted, nine rejected, and an otherwise valid object with an extra
transport-shaped key rejected. A separate agent checked ten historical parser
controls, including text/entity/date limits. All were offline and synthetic.

Historical source references: `af6a615:hymem/dreaming/facts.py:832–903`,
`hymem/contrib/openai_client.py:655`, `hymem/dreaming/runner.py:2395`.
The local facts object/capacity repairs exist outside the deployed snapshot;
they must be reviewed and tested as a narrow candidate, not assumed installed.

## 3. Chunks: exact zero-model reproduction

All three chunks with retained failed-attempt rows were rebuilt from validated
canonical source manifests and replayed using the frozen extractor. A client
that forbids model calls and a network-denial guard were installed. Each produced
the identical stored `resource_limit / split:no_admissible_semantic_boundary`
failure with **zero model calls and zero SQLite changes**:

| Question | Chunk prefix | Stored attempts | Owned source content lengths |
| --- | --- | --- | --- |
| 1 | `chk_fba7a143` | 1, held | 4,105 + 8 |
| 8 | `chk_642e936a` | 3, quarantined | 4,027 + 15 |
| 9 | `chk_1c6511c2` | 2, held | 3,474 + 410 |

Thus these six recorded failed attempts occurred before generation, not because
DeepSeek declined or exhausted output tokens. They hit the conservative source
partition/context policy and 4,000 encoded-character leaf bound. Even Q9's two
individually fitting records cannot safely separate under the current contextual
suffix policy. Exact source hashes, record lengths, protected spans and replay
results are retained in `diagnosis-chunk-replay.json`; no source text left Afrodite.

An independent synthetic counterexample demonstrates one general limitation:
25 short bullets split successfully as a bare list, but adding a colon-led
introduction or heading makes the whole list indivisible and produces a
zero-call resource failure. Do not assume that is the exact block type of every
retained source without further structural tracing. The confirmed attribution
is local prepartition failure, not a model/output failure.

## Remediation boundary and proposed order

This investigation made no application changes. The narrow deployed repair fixed
the old clean-empty veto and failure-envelope crash; it did not establish broader
fact/digest convergence. Treating its offline acceptance as enough justification
for a full 500-question run was too optimistic, especially given known excluded
fact/digest contract work.

Recommended next implementation, one independently reviewed agent/fix at a time:

1. Align facts JSON-object instructions and advertised capacities; preserve
   strict coverage/cursor guards. Persist bounded rejection subtypes, not secrets
   or raw outputs. Test valid empty, overflow, extra keys, malformed fields and
   unchanged cursors before any paid verification.
2. Align digest summary requirements with its bound and use reason-specific
   recovery. Test prior-summary preservation, exact 500/501 behavior, all seven
   retained terminal chains and correct quarantine accounting. Do not deploy the
   entire unaccepted local verifier architecture or silently truncate summaries.
3. Extend safe source/context partitioning for the exact three retained chunks.
   Preserve introductions, negation, attribution and one-sided context ownership;
   retain the existing unsafe-split negative controls. More model retries cannot
   fix a zero-call partition failure.
4. Verify the combined isolated candidate over all ten retained question stores
   and synthetic controls offline. Any subsequent bounded live test should cover
   these failure families explicitly, with complete rejection/usage accounting.
   Do not restart full500 merely because the chunk-only canary passes.

No fixes are claimed complete here, and the stopped run has not been resumed.
