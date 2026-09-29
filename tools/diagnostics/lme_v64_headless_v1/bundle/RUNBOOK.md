# Fresh schema64 sample-eight

Preparation is offline. Never restart or resume historical runs.
Prepare with --tree, --manifest, --census, --runtime-seal and --output.
Source manifest identity is compact sorted JSON SHA256 (481 files).

Runtime seal schema lme-v64-runtime-seal-v1 has schema, image, runtime_root,
entries and root_reviewed=true, canonical pretty JSON plus newline.
Each exact recursive entry has mode/uid/gid and one of sha256, link (readlink
target), directory=true. Root reviews all effective runtime files and links;
the pinned image binds interpreter targets external to the runtime mount.
The controller compares all entries, modes, ownership, files and link targets,
without traversing links. Hidden files and additional entries are included.

Controller requires --manifest-sha256 from the fresh seal. Live start requires
--deployed-gate-sha256 for root-reviewed-deployed-gate.json, with schema
lme-v64-root-reviewed-deployed-gate-v1, root_reviewed=true, status=passed,
source_manifest_sha256, schema_version=64, preparation_manifest_sha256,
runtime_seal_sha256; true complete_suite_passed,
bounded_paid_dream_postflight_passed, migration_rehearsal_passed,
deployed_source_verified, postdeploy_checks_passed,
production_environment_preserved. evidence_sha256 binds suite, paid_postflight,
migration, deployment and postdeploy. Root reads all receipts before issuing it.

Only live mounts credentials; source/runtime/dataset mounts are read-only.
No production DB or Docker socket. Preflight/validation network=none.
No restart policy. Nine-hour supervision and 512MiB logs remain unchanged.
Protocol, selected eight IDs/seed0, workers1, separate canary metering,
one-attempt histories and archive/checkpoint validation are preserved.

Operational gate requires zero failed/missing, healthy indexing and zero
summary degradation. Score and summary health are distinct outcomes.
Full500 needs a separately sized supervisor and validator.

Use lme_v64_headless_finalizer_v1.py with --manifest-sha256 for fresh detached
finalization. It pins this controller and never starts a paid live container.
Do not invoke the historical R7 finalizer with this bundle.
