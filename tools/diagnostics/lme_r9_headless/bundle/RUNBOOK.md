# Isolated R9 fixed-eight development integration test

Preparation is offline and launches nothing. Source is the exact 508-file
R9 candidate inventory 35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51.
Source and run root are fixed under the reviewed benchmark base:
lme-r9-full-suite-v1/candidate and lme-r9-sample8-headless-v1.
Every new output is exclusive; historical artifacts stay intact.

Seal with prepare.py --tree, --manifest (plain exact source mapping), --census,
--runtime-seal, --output, --approved-source-manifest-sha256,
--remote-root and --remote-source. The archive includes byte-pinned controller,
continuation and bundle. No candidate files are edited or deployed.

Controller create/start/status preflight/live/validation requires --root and
--manifest-sha256. Create and start are separate exclusive actions. Preflight
uses network none and no real credentials, reaches the genuine stock CLI
startup boundary and must verify producer identity and zero provider calls.

Live start requires --candidate-gate-sha256 for exact bytes of
root-reviewed-candidate-gate.json. The R9 gate binds precisely suite and
summary manifest/receipt/supervisor files, with exact R9 source equality.
The suite is the unchanged r8-offline-full-suite-v1 harness: manifest c8f2cae8,
7940 collected, 7936 passed, four skipped, no failures/errors or provider calls.
The targeted paid R9 source-exact summary capsule is manifest f6f8aff6, nine
calls, normal 417 characters, selected direct repair 282 characters, and one
healthy explicit-recovery publication in three calls, with clean supervision,
cleanup and unchanged source/reference/non-summary data.

Admission is for the next development integration test. Root reviewed selected
invented control outputs as faithful with no invented claims; this review scope
does not establish independent semantic review of the private real case. The uncertainty selected
output omits the unexecuted refresh/comparison proposal. Complete source-claim
retention is not established; strong coverage checklist passed is false.
No retained R8 extraction evidence is carried forward. This gate does not claim
the historical R8 full-control guarantees or full500 coverage.

Keep all eight IDs/indices/census, dataset, seed zero, workers one, top-k 15,
deepseek-flash, endpoint and disabled thinking unchanged. Canary v20 and source
split v11 identities remain. Limits remain 100 indexing cycles, 3600 seconds
for indexing per question, nine-hour supervision and 512 MiB private logs.
There is no overall provider call/token/spend cap: global paid call cap is null.
No seed search, reroll, resume or failed-question retry is allowed.

The host continuation is separately detached by the operator after root review,
as UID1000, with a private fresh lme-r9-headless-continuation-v1 root. Its config
binds the fresh prepared manifest/controller, source pin, 508 files, 7940 tests,
the exact suite/preflight container IDs, the two reviewed evidence roots/pins
and the exact semantic_review object in host_control.py. Run continuation.py
with --config, --config-sha256 and --self-sha256; its bytes are also sealed in
the preparation manifest. It writes an exclusive intent before any dispatch.

Continuation waits for the reviewed suite, checks evidence and genuine
preflight, starts one paid fixed-eight container, then starts one read-only
network-none offline validation container after live exit, including a known
exited/PID0 OOM or nonzero result. OOM is never reported as live success.
Ambiguous dispatch or timeout requires operator inspection and never retries.
The continuation persists on the host without the laptop session. It never
kills/removes containers or deploys. It preserves raw text/credentials/stores
on the remote host; validation and scoring are distinct from healthy indexing.
The unchanged validator permits success_with_summary_degradation and reports
degraded-question/session counts. Report these explicitly alongside validation
status; offline validation success does not imply every summary is healthy.
