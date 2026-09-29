# Prepared production controller, awaiting external review and gates

This document records preparation only. Neither this file nor staging grants
launch approval. Full-suite, private-paid independent postflight, deployment
start, production preflight and separate role-profile receipts must be verified
and externally sealed by root. No production action was performed to prepare
these helpers.

The host helper is `tools/diagnostics/claim_conflict_v64_production_dream_host.py`.
The worker is `tools/diagnostics/claim_conflict_v64_production_dream.py`.
The unchanged reviewed metering adapter is
`tools/diagnostics/claim_conflict_episode_shadow_dream_instrumented.py` and the
owned-process supervisor is `benchmarks/supervised_invocation.py`.

The private host bundle descriptor schema is documented in the host helper's
module docstring. Its controller SHA binds the host helper bytes. The descriptor
must be mode 0600 and externally SHA sealed. All flat bundle file names and bytes
are SHA sealed; these include the fresh worker, reviewed meter/supervisor, exact
481-file manifest and normalized gate receipts. The worker's independently
sealed launch manifest binds the exact candidate, helper paths and bytes, target
session hash, generation, current effective role environment hash and budgets.
Descriptors and live environment details remain on the host and are never
committed here. No helper creates a verified gate receipt for itself.

Host staging is a fresh single directory directly below
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks`; it maps to the same name
below `/home/node/.hermes/benchmarks` in Hermes1. Stage and launch fences are
exclusive: a partial stage, stale intent or existing supervisor-start receipt
requires inspection, never automatic retry. Container `hermes-1`, its ID, image
ID, complete sealed mount map, running state and healthy state are checked before
launch. Path mapping uses the longest matching sealed bind ancestor, including
an exact reviewed home bind if present, and rejects any unsealed nested mount.
The census interpreter is `/home/node/hymem-env/bin/python3`. The worker
resolves and hashes the frozen target's session, rechecks schema 64/open durable
runs/dream lease, then uses only the official targeted dream API. Its counters
observe original producer/client function code and do not replace those methods.

Usage on the host, after root supplies the private reviewed bundle and its SHA:

```text
python -I -B claim_conflict_v64_production_dream_host.py stage PRIVATE_BUNDLE SHA256
python -I -B claim_conflict_v64_production_dream_host.py launch PRIVATE_BUNDLE SHA256
python -I -B claim_conflict_v64_production_dream_host.py status PRIVATE_BUNDLE SHA256
```

Launch detaches a fresh host supervisor so the laptop's connection is not its
lifetime owner. The in-container reviewed supervisor owns the actual worker
process group, releases stdin only after durable ownership, allows 2700 seconds
of cooperative work with an outer 2760 seconds limit and 10 seconds of separately
bounded cleanup, and never performs restore, restart or another dream. The
rehearsed budgets are 128 completions, 384 LLM HTTP attempts, 512 embedding HTTP
attempts and 896 total attempts, with 16 texts/128000 characters/512000 UTF8 bytes per
embedding batch. These final 45 minute limits supersede the earlier provisional
1800/1860 second limits in the gate planning note.
SIGTERM becomes a catchable interruption in the fresh worker supervisor so the
reviewed ownership cleanup path runs before the previous handler is restored.

Status returns only static status codes, numeric counts and artifact hashes.
Worker completion proves a single durable owned invocation and clean shutdown;
it does not prove episode repair or session convergence. Independent vector and
target-publication postflight remains mandatory. Durable pending digest,
profile and fact slices may remain after a successful bounded targeted dream.
Final readiness and any production vector-health claim remain separate gates.
