# Isolated R9 full500 development integration test

Preparation is offline and launches nothing. Source is the exact 508-file R9
candidate inventory 35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51.
The fixed source is lme-r9-full-suite-v1/candidate; the fresh run root is
lme-r9-full500-headless-v1 under the reviewed benchmark base. Outputs are
exclusive and no source application files are changed.

Seal with prepare.py --tree, --manifest, --census, --runtime-seal,
--prior-sample8-validation, --output, --approved-source-manifest-sha256,
--remote-root and --remote-source. The all500 census contains only ordered IDs,
source indices, categories, session counts and message counts. The sealed prior
validation report is outside bundle; package inventory remains strict.

Stock CLI selection is --sample 0, seed zero, exactly source indices 0..499.
Keep workers one, top-k 15, deepseek-flash, disabled thinking, legacy-custom
judge, full protocol split, no preregistration, 100 indexing cycles and 3600
seconds per question. Keep-db uses fresh isolated stores and checkpoint.
Outer supervision is 2592000 seconds (30 days), private log cap 512 MiB.
There is no overall provider call/token/spend cap. No reroll, resume or retry
is allowed. Full500 legacy-custom results are not officially comparable.

Controller create/start/status preflight/live/validation uses --root and
--manifest-sha256. Creation and start are separate exclusive actions.
Network-none preflight uses no credentials and reaches the genuine stock CLI
pre-provider boundary with all500 ordered IDs, exact producer identity and zero
provider operations. Live start additionally requires a pinned root-reviewed
candidate gate and successful genuine preflight.

Admission binds the unchanged full suite (7940 collected, 7936 passed, four
skipped, zero errors/provider calls), plus the exact same-source sample8
manifest 2ba18ff05ec1b679ea5f74cb8f3153558e042bc070811a9cc7a7476b71649b80,
its successful sealed offline validation report and clean live supervisor.
Every prior sample8 question must have complete healthy indexing and zero
missing/degraded/malformed summaries. The old manifest-pinned controller
revalidates its package/runtime/source and exact validation container mounts,
image, command, exited/PID0/noOOM state. Actual closed metadata Docker stdout
must match the sealed prior report. This admission makes no semantic guarantee.

The host continuation uses a private fresh lme-r9-full500-continuation-v1 root
and pinned config, controller and self hashes. It writes an exclusive intent,
waits for the full suite, admits evidence/preflight, starts one paid full500
container and waits at most 2592060 seconds. After a confirmed terminal live
exit it starts exactly one read-only network-none offline validator, including
after a nonzero exit or OOM. Failed or incomplete artifacts remain failure
evidence. Ambiguous dispatch and timeout stop for operator inspection without
automatic retry. The controller and continuation never kill/remove containers.

Validation binds the genuine strict scored archive and physical one-attempt
checkpoint to all500 IDs, stock recipe, producer canary and role accounting.
Reader/judge usage is measured and canary usage is accounted separately.
Raw case content, provider text, credentials and stores remain private on host.
Validation permits healthy completion with reported summary degradation;
report degradation explicitly. All exported reports are closed metadata.
