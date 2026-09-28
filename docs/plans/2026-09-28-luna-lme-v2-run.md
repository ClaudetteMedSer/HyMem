# Fresh Luna LME v2 pilot

The user requested “Run the LME using Luna” after the experimental canary
repair passed. Start with the existing reviewed one-question runner; a
non-blocking question asks whether to expand afterward to eight or all 500.
No answer to that expansion question is implied by launching this pilot.

This is a new one-shot run, not a retry/resumption of the failed v1 pilot or
cancelled DeepSeek R9. It has its own single run-entry canary and shared budget
across canary, memory, reader and judge. Prior diagnostic usage remains separate.
No production changes, API-key route, purchased-credit use, provider fallback,
automatic reroll or modification of frozen candidate code is authorized here.

## Verified inputs and launch

- Exact GPT-6 Luna through official Codex 0.158.0 App Server, saved ChatGPT auth.
- Pilot SHA256 `0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0`.
- Transport SHA256 `387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491`.
- Inventory file SHA256 `852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb`;
  compact source-map SHA256 `35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51`;
  all 508 frozen candidate files verified.
- Dataset SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
  Source-first question, index 0: 53 sessions / 550 messages. The full dataset
  is kept unchanged; the runner streams only its first item.
- Root reran 96 pilot/transport/frozen-runtime tests; all passed. A separate
  Sol launch review found no technical blocker for this fresh-run scope.
- Actual no-inference Afrodite preflight verified subscription/configuration
  admission, 81% weekly quota remaining, and complete preflight process cleanup.

`/tmp` is a 7.6 GiB RAM-backed filesystem, not the server's main disk. The initial
20 GiB space check stopped before any model call. Output instead uses a fresh
private directory on the disk-backed home filesystem (216.14 GiB free at launch).
No existing data was deleted. Benchmark temporary files also use this directory.

Private remote root: `/home/atta/.hymem-luna-lme-v2-yeds3h_l`.
Output: `run-v2`; runner terminal metadata: `safe-terminal.json`.
Unit: `hymem-luna-lme-v2-20260928-yeds3hl.service`.
Launch receipt SHA256:
`a9578531cede94accf74e9f336a0425a95907e222dbc5f3cd83c3de1f2fb2203`.

The unit launched active/running (initial PID 2764452). Root verified
`KillMode=control-group`, `Restart=no`, `NRestarts=0`, `RuntimeMaxSec=5390`,
`TimeoutStopSec=10`, private umask, and exact run cgroup. The unit preserves
terminal status with `RemainAfterExit=yes`; active/exited is not running work.

## Limits and interpretation

One question, sequential; at most 1,200 model turns, stop before another turn
at 4 million observed tokens, at most 90 minutes overall, 120 seconds per
invocation, and at least 25% quota remaining in every reported account window.
The server enforces the overall wall/cleanup bound independently of this laptop.

The canary is experimental v2. Retrieval uses top-k 15, lexical/FTS-only,
aggregation nodes and episode granularity off, no distillation, Luna memory,
reader and `legacy-custom` judge. Healthy indexing AND summary coverage remain
mandatory. Treat completion/index health and answer correctness separately.
Subscription temperature/output-cap/JSON-mode are not equivalent to the API
controls; this is not a canonical official-model score. Internal HTTP attempts
are unknown. Recorded token totals exclude unreported in-flight work.

## Polling

Use only reviewed metadata readers. Do not export logs, model/benchmark text,
stores or credentials. Do not launch, resume, reroll, change limits/models or
modify production in a polling turn. The cancelled R9 monitor remains paused.
Terminal success needs private offline validation and cleanup checks, not just
exit status. The Sol-implemented read-only reader passed **14 tests**, including
six independent root controls; root also exercised it against the running unit.
Its SHA256 is
`422b745bb3dedec0007e75a31b452628e30e90a68c46a95b8073395018d9b05f`.
The exact launch recipe below reads only bounded metadata; terminal validation
privately rechecks result/row/index-health and source/dataset integrity. It
exports no raw benchmark/model text. Only `completed_and_clean=true` establishes
clean completion, not `terminal.validated` alone. A scored incorrect answer is
valid benchmark execution, not an execution failure.

Verify the reader hash locally, then run from the project directory:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  python3 -I -B - \
  --root /home/atta/.hymem-luna-lme-v2-yeds3h_l \
  --unit hymem-luna-lme-v2-20260928-yeds3hl.service \
  --expected-cgroup /user.slice/user-1000.slice/user@1000.service/app.slice/hymem-luna-lme-v2-20260928-yeds3hl.service \
  --pilot /home/atta/.hymem-luna-lme-v2-yeds3h_l/luna_subscription_pilot.py \
  --transport /home/atta/.hymem-luna-lme-v2-yeds3h_l/codex_subscription.py \
  --candidate /opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate \
  --inventory-stamp /home/atta/.hymem-luna-lme-v2-yeds3h_l/headless-source-map.json \
  --inventory-sha256 852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb \
  --dataset /opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json \
  --launch-receipt /home/atta/.hymem-luna-lme-v2-yeds3h_l/launch-receipt.json \
  --launch-receipt-sha256 a9578531cede94accf74e9f336a0425a95907e222dbc5f3cd83c3de1f2fb2203 \
  < tools/diagnostics/luna_subscription_lme_progress.py
```

The first verified reader snapshot showed a passing canary, question started,
54 observed turns / 406,250 known tokens with another invocation in flight,
exact live unit/cgroup, no restarts and 216.129 GiB free. Indexing outcome and
answer correctness were not yet available. Missing/in-flight usage is unknown,
not zero. No terminal failure was present in that snapshot.

For background checks, remain quiet on routine progress. Alert on failure,
source/metadata integrity loss, incorrect unit/cgroup policy, disk under 20 GiB,
or both progress and log timestamps unchanged for over 12 minutes while the
unit claims to run. A boundary-race/temporarily unreadable JSON snapshot should
be retried once without changing the experiment. On clean completion or
terminal failure, report validated health/correctness/known usage/cleanup with
limitations and pause the new pilot heartbeat. Never resume the old R9 monitor.
Local heartbeat polling requires this app/computer available; the run itself
and server-enforced limits are independent of the laptop.

The app heartbeat `monitor-luna-lme-pilot` is PAUSED after terminal failure.
It previously checked at a 10-minute interval and reported only completion,
failure or required action.
The older `finish-lme-validation` heartbeat was inspected and remains PAUSED.

## Terminal diagnosis

No question completed: indexing timed out during the third cycle after
3,605.86 seconds. The two completed cycles each processed 50 chunks with no
chunk-extraction failure. Terminal accounting: 472 turns, 3,341,389 known tokens,
usage complete, cleanup complete. The reported 178 pending units were the
end-of-cycle-two snapshot. A later query-only census under the same producer
found 170 pending chunks, three pending digests and three missing/degraded
summaries, without quarantine, malformed state, coverage failure or terminal
loss. No answer correctness is available. No automatic rerun is authorized by
this receipt. See `2026-09-28-luna-concurrency-repair.md` for the subsequent fix.
