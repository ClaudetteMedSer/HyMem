# Observation-only Luna LME diagnostic

## Purpose and acceptance

Root accepted the separately implemented v3 diagnostic adapter and observed
runner/launcher/reader after independently reviewing the code and running 606
offline tests. See `2026-09-28-luna-transport-failure-observability.md`.

The preceding four-question grounding run failed with its underlying error
classification discarded. This fresh same-four-question run is justified to
capture that finite classification if it recurs, or establish validated
completion under unchanged gates. The observation-only repair is not evidence
that the live fault is fixed. A pass is not a measured reliability improvement.
There is no automatic reroll on either failure or completion.

Only transport diagnostics and their source-bound reporting wrappers change.
The grounded memory candidate, questions/order, prompts, model, authentication,
strict canary, indexing/summary gates and scoring remain unchanged. No original
run is resumed. No production changes, raw text/log/store/credential export,
paired-grounding rerun, full-500 or cap increases are authorized here.

## Frozen sources

- Adapter: `0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d`
- Runner: `2062a8816e8e22214a1e7febc4898322210c381abdf9bb0f34e802d61d111550`
- Launcher: `1f503eae5121306a9ca70a6d1286e7b1cb898186ac49979403ff835367e3df3e`
- Reader: `4c8862733d3950d8437fe2ad2376349ff0ce2b9935ea5561a79c52bfd611f303`
- Candidate inventory: `896b89d56f393bf386cef7b400bbb1450306555ccb6d05f4b6c787e616bf104f`
- Candidate source map: `217036b8089911c352cdf5994ef2b37915b5c68d2622a23c0642d264487dbe62`

## Budgets and isolation

GPT-6 Luna subscription only; 25% reported quota floor, no API-key fallback or
purchased credits. Four questions/four workers; campaign 8,012 turns /
48,160,000 known tokens / 14,400 seconds. Each question: 2,000 turns / 12,000,000
known tokens / 12,600 seconds, indexing 10,800 seconds. Canary: 12 turns /
160,000 known tokens / 600 seconds. Invocation deadline 120 seconds. Warm
process rotation 16 requests or 300 seconds; fresh ephemeral thread per call.

Systemd runtime 14,530 seconds plus ten-second stop bound, TasksMax 256,
MemoryMax 4 GiB, CPUQuota 200%, OOMPolicy kill, KillMode control-group,
Restart no, RemainAfterExit yes. Admission requires 6 GiB available memory,
20 GiB disk and no older running pilot. Private empty working directory/tmp,
source checks before dispatch, and one-shot marker consumed before launching.
Never retry an ambiguous launch. The server cleanup bound is laptop-independent.

## Status

Accepted for preparation. No new model calls or launch yet. Record immutable
root/unit/receipt identities and the monitoring command before dispatch.

## Prepared immutable identity and read-only monitoring

Prepared once, with zero model calls. Root independently exercised the real
SSH-stdin reader: receipt, all source pins, dataset and candidate inventory
verified; no output, processes or active unit existed. Unit policy is naturally
unavailable before creation, not a failed live-policy check. Disk 215.667 GiB.

- Root: `/home/atta/.hymem-luna-lme-observed-4e8qycex`
- Unit: `hymem-luna-lme-observed-4e8qycex.service`
- Cgroup: `/user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-observed-4e8qycex.service`
- Receipt SHA256: `ec4b060b2122c742c4e0dac6d95037ae04693ee65d5866523a0f9b4a194fadfc`
- Staging directory: `/home/atta/.hymem-luna-observed-bundle-Mr676UBv`

Verify the reader SHA256 above, then use this exact metadata-only command:

```sh
shasum -a 256 tools/diagnostics/luna_observed_lme_progress.py
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  /usr/bin/python3 -I -B - \
  --root /home/atta/.hymem-luna-lme-observed-4e8qycex \
  --unit hymem-luna-lme-observed-4e8qycex.service \
  --expected-cgroup /user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-observed-4e8qycex.service \
  --receipt-sha256 ec4b060b2122c742c4e0dac6d95037ae04693ee65d5866523a0f9b4a194fadfc \
  < tools/diagnostics/luna_observed_lme_progress.py
```

The monitor must be updated before the one-shot launch. While active, polling
is strictly read-only: no new model calls, source/model/budget changes, restarts,
resumes, rerolls or production changes. Never relaunch a missing unit.
Report completed questions, terminal outcomes, integrity/policy/process/OOM/
task-denial problems, disk below 20 GiB, or both log and progress inactivity
over twelve minutes. Retry a temporarily unreadable JSON snapshot once.

Require `completed_and_clean=true` plus independently empty unit/cgroup before
claiming clean completion. Separate correctness from indexing/summary health.
Missing health is unknown, and in-flight/failed usage incomplete, not zero.
Only terminal stage reconciliation is definitive. Report finite first-failure
metadata and cleanup on failure; do not infer the old cause merely from this
run's outcome. External quota/auth limitation: pause and ask for direction.
Preserve all old runs and the stopped DeepSeek monitor. Never rerun the accepted
paired grounding diagnostic. A terminal failure returns to evidence-driven
sequential Sol repair/root review, not an automatic reroll; a clean completion
ends and pauses this heartbeat. This is neither canonical API accuracy nor a
full-500 result.

## One-shot dispatch and startup

Dispatched once on September 28 at approximately **21:31:06 UTC**, after the
monitor update returned ACTIVE and the prepared launch/receipt hashes were
rechecked. The launch command returned 0 with `never_retry=true`.

At 12.26 seconds, root's exact metadata-only reader verified live source,
dataset, inventory and receipt pins; the unit was active/running in the exact
cgroup, with correct resource/cleanup policy and zero restarts. Canary was
in progress (two settled calls, 16,881 known tokens); no failure recorded.
No question had started or completed yet. Observed peak tasks 42 of 256, zero
task-limit denials, disk 215.666 GiB. This is startup evidence, not end-to-end
success. Usage remains incomplete during the live campaign.

## Terminal canary failure

The same run stopped after **30.018 seconds**, at the strict canary, before any
question started. No reroll occurred. Eight admitted/returned calls, **69,130
known tokens**, complete usage and reconciled stage accounting; no transport
first-failure event. Safe terminal reason `canary_failed`. The reader returned
`completed_and_clean=false`. Root independently verified MainPID 0, empty
cgroup header, zero expected-cgroup processes, zero restarts and policy still
valid; exit status 1. Removed kernel counters are unknown, not zero. Disk
215.667 GiB. The original transport cause remains unresolved.

A receipt-bound, finite-only read of retained canary metadata found one of two
required core claims matched, four initial leaves and eight calls. Both path
checks were false. All four optional type slots were absent, with zero invalid
or wrong types and no final hint mismatch; omitted optional types are allowed
and are not the failure. Result SHA256:
`0730f970ec291473121bde497f855bfa4e49bda2b7b108a60d7efd7e55058147`.

Separate read-only Sol diagnosis agrees that these metadata alone cannot
distinguish model omission/invalid output from an extraction/validator defect.
Root is performing a network-disabled, byte-exact replay of the eight saved
responses using the verified frozen candidate. Only finite gate/path counts
may leave the host. No new model calls or candidate changes are authorized by
this diagnostic step. A root-run synthetic control on this actual candidate
still accepts the two exact claims and rejects an unsupported additional
predicate (one offline test passed).

## Offline diagnosis: predicate substitution, not a missing response

Root's on-host replay consumed all eight retained responses with exact canonical
UTF-8 equality of every regenerated request field and reproduced the entire
recorded canary grade. Network connects, process creation and file writes were
blocked. No model calls, raw evidence export or changes to run artifacts.
The replay helper initially rejected the source inventory's existing 0664 mode
inside its private 0700 parent; after independently verifying its hash and
mode, source-only reads allow that existing mode while evidence still requires
private file permissions. This was an offline diagnostic setup issue, not a
run or candidate defect.

Extraction returned two valid-structure triples, no markers and no extraction
failure. The table triple matches all five core fields. The prose triple has
the expected subject, object, polarity and source-message ID, but the predicate
does not match. All source IDs and context paths are correct; the sole execution
path delta is prose exact-context recognized emissions: zero versus one.
The prose primary returned one triple and its omission pass returned none.
The canary correctly rejected the wrong relationship. Type hints and transport
did not cause this failure. No final indexing/summary/answer scores exist.

The accepted generic prompt already requires independent predicate support and
preservation of preference semantics. The application structurally validates
predicates, but the subsequent omission-only pass cannot reject or replace an
unsupported primary predicate. Another wording change has no proven deterministic
defect to repair here. The old transport incident remains unresolved; the new
diagnostic never reached its question workload.

Do not reroll. The proposed next step is a separately scoped source-grounded
semantic validation contract, with invented positive/negative controls and a
new honest canary/extraction identity, followed by separate Sol implementation
and root verification. That is broader than observation-only reporting or the
accepted generic prompt change; request direction before implementation or
additional live diagnostics. Production, all old runs and frozen receipts stay
unchanged. Existing gates must not be relaxed to accept the substituted claim.
