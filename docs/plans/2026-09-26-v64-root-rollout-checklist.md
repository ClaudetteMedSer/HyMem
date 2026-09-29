# Root rollout preparation and action checklist

Current execution note: the full suite and corrected private paid postflight
have passed (see the readiness JSON). This document retains the original
preparation checklist below, whose pending language is historical. r1 failed
before stopping production because the helper omitted sqlite-vec loading;
r2 passed migration rehearsal but refused apply before its intent because a
stopped WAL remained. The unchanged old service was restored and verified.
r3 uses the separately reviewed v3 checkpoint adapter, a fresh PID-bound
census and config `6742a033f3da4769b572d80864520294511d45b88718fc8dddad4c616c14f3a1`.
Do not reuse r1/r2 action intents or their old census for a new stop.

No action was performed by this checklist. The two r1 stage names below are
proposed fresh names: root must prove absence before use. The tools stage and
rollout stage are different; transport creates the former, and the rollout
stage action alone creates the latter. Do not precreate the rollout stage.
The supplied JSON template has null seals and therefore cannot pass admission.

## Fixed identities and pending receipts

Candidate: exact remote481 tree at
/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/cold-replay-dream-v1/episode-shadow-replay-v1/candidate.
Its canonical flat manifest digest is
5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576.
The local manifest JSON raw digest is
25ac2fb6366d6fd854d8431ba15e421c99181f762785cec0e60af4546029a300;
canonical and raw digests have different purposes. Local application/test
overlay trees are not deployment sources.

Baseline R7 raw manifest digest:
1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8;
override doctor with
2ac786c6590aa7fbece726e6380981906d514fefe9f664a3ca4f7b1d1de7c15b
and phase1 with
31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136.
Current known deployed479 canonical identity is
a59fda0427e4485e964d1b1127abb1732c6f67c0ed539e967e3ef332f4587203.
Known container ID:
359b7d270d6acd4ee53550446213750cb93d917228ee8431826db76ab07d7a41;
image sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5.

Transport file digest:
aad25e6e36150fd9d09af9c931abbcffa2c8e16a2fec1cd7db056aad887192f7.
Its five-file payload canonical seal:
7e18a40caf29b67aa6778c766e5d42a2b04b987ea8d0d3de46960be6116bf732.
Payload includes rollout, census, postdeploy, strict vector checker, and
the old postdeploy template pinned
6f4351ef2df7f07b03d518842ae5ad47df2a18ae8d541202a8ae55759ad29da9.
Recheck all local hashes immediately before transport; a changed payload
requires a new root-reviewed seal, not overriding the mismatch.

Full-suite collection was7764 with zero collection errors; terminal counts,
skips/xfails, cleanup and final result seals remain pending. Private paid
result SHA7abc29a7c4e4f0676963177f942d1994d5e12cb12b813ea3518ebc06d0e6f8d5
and closed-store SHAb67056e9f9b13067ecfd9c50ee0d19989600f8fa6e0b8cc9553c5c6396aa0ce4
are evidence awaiting corrected independent v2 postflight. V1 failed audit is
not a passing gate; bounded-work exhaustion must not be presented as full
session convergence.

## Normalize and seal gates

Root creates new metadata-only objects, retains their raw evidence identities,
and sets root_reviewed=true only after actual terminal results are inspected.
Each JSON's raw bytes receive an external SHA256; do not use the candidate's
canonical digest convention for receipt seals.

Exact fullsuite admission fields:

~~~json
{
  "status": "passed",
  "role": "fullsuite",
  "candidate_manifest_sha256": "5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576",
  "root_reviewed": true,
  "collected": 7764,
  "expected_collected": 7764,
  "passed": null,
  "skipped": null,
  "expected_skipped": null,
  "xfailed": null,
  "expected_xfailed": null,
  "xpassed": 0,
  "failed": 0,
  "errors": 0,
  "exit_code": 0,
  "cleanup_verified": true
}
~~~

The nulls are placeholders, not an executable receipt. Passed must be positive;
passed+skipped+xfailed must equal collected. Skips/xfails must match separately
reviewed expected counts. No unexpected failures/errors/xpasses are accepted.
Record raw suite, test-overlay and controller receipts as additional provenance.

Exact paid_postflight admission fields:

~~~json
{
  "status": "passed",
  "role": "paid_postflight",
  "candidate_manifest_sha256": "5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576",
  "root_reviewed": true,
  "checks": {
    "claims": true,
    "ledger": true,
    "canonical": true,
    "same_generation": true,
    "integrity": true,
    "foreign_keys": true,
    "episode_vectors": true
  },
  "aggregation_passed": true,
  "paid_calls": 59
}
~~~

59 is the observed fresh private completion count; retain actual role attempt
counts, tokens, v2 audit and closed-store/raw result hashes as provenance.
Aggregation must be proved by durable publication and no build/fusion errors;
a legitimate aggregation_blocking strategy label is not a failure.
The gate represents this bounded diagnostic and independent audit.
It does not imply all session work converged or clean summary health.

## Explicit root actions after prerequisites

After proving the proposed tools path absent, run the following locally as
three distinct root-reviewed actions. Read uses a fresh local output directory.

~~~sh
/opt/anaconda3/bin/python -I -B /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/hymem_v64_transport.py install --tools-stage /opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-r1 --manifest-sha256 7e18a40caf29b67aa6778c766e5d42a2b04b987ea8d0d3de46960be6116bf732
/opt/anaconda3/bin/python -I -B /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/hymem_v64_transport.py census --tools-stage /opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-r1 --manifest-sha256 7e18a40caf29b67aa6778c766e5d42a2b04b987ea8d0d3de46960be6116bf732
/opt/anaconda3/bin/python -I -B /Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics/hymem_v64_transport.py read --tools-stage /opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-r1 --manifest-sha256 7e18a40caf29b67aa6778c766e5d42a2b04b987ea8d0d3de46960be6116bf732 --output-directory /private/tmp/hymem-v64-census-20260926-r1
~~~

Fresh census includes Docker Config digest, schema63, all479 source hashes,
runtime recursive inventory digest, actual wrapper/hook/config/.env identities,
full raw Honcho environment hash and separate Honcho/MCP HYMEM profile hashes.
Do not normalize the role-specific aggregation flag or add EXTRA_BODY.
Wrapper host path is home/.hermes/bin/hymem-server-wrapper; hook is
home/.agent37/hooks/post-restart.sh. No config/environment values leave Afrodite.
Root reviews a new census-reviewed.json and runtime seal, sets their review
flags, and seals the resulting bytes. Keep raw captured receipts unchanged.

The existing five-file transport does not upload arbitrary metadata. Transfer
the exact flat candidate-manifest JSON, normalized two gate receipts,
census-reviewed.json and resolved rollout-config.json separately using a
reviewed exclusive metadata transfer; preserve0400 and verify raw hashes.
That transfer action/path/seal is still a prerequisite, not an existing
capability of install. Place these files in the tools stage, never the rollout
stage. Fill the template's three null receipt seals, then SHA256 the complete
config bytes. Keep that same config seal for every subsequent rollout action.

On Afrodite as uid1000, run exactly one action at a time with the reviewed
config seal substituted for CONFIG_RAW_SHA256. Do not run the next action
until its receipt passes inspection.

~~~sh
python3 -I -B /opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-r1/hymem_v64_rollout.py stage --config /opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-r1/rollout-config.json --config-sha256 CONFIG_RAW_SHA256
~~~

Repeat the same command with action rehearse, then stop, apply, migrate and
start, each separately. Stage/rehearse/stop/apply reject collisions at both
new candidate paths; pyproject changes require a separate installation review.
Stop uses120-second grace, fails on137/OOM, takes a fresh stopped backup and
rehearses it before apply. Migration checks durable old-column row fingerprints,
NULL historical replay proofs and stable reopen. Never restore a DB or retry
an intent automatically. After start the stopped database hash naturally
changes; the old pre-stop Honcho PID must not be required for post-start roles.

## Pre-dream and final verification remain distinct

The sealed hymem_v64_postdeploy.py is a final strict verifier. Running it
immediately after migration cannot be used as the pre-dream admission gate:
the inherited eight surplus vectors and17 unverifiable episodes remain.
Do not weaken or modify that sealed final helper.

Least-change pre-dream preparation is a new separately sealed read-only
adapter around the existing generated final container script. Keep source481,
health, preservation/distribution, process_env role profiles, resolved config,
schema64, integrity/FK, canonical, ledger and same-generation checks unchanged.
Run only source();health();preservation();env=process_env();config(env);schema().
Omit doctor and MCP tool calls from this admission pass. Change only the
schema function's strict vector assertion to metadata baseline recording;
do not report episode_vectors=true in store_checks. Require exact fingerprints,
counts and unverifiable count to match the current sealed pre-start census,
with no new missing/different vectors. A changed production baseline requires
inspection and fresh evidence rather than pretending it is unchanged.

This new adapter, its offline controls, receipt schema, helper pin and explicit
launch command are unresolved work for root review. No existing helper exposes
a pre-dream mode. Its passing receipt should say phase=pre_dream,
vector_health_passed=false, inherited_vector_baseline_preserved=true.

Only after that admission prepare the official targeted production dream from
the separate production-dream plan: same frozen private session/generation,
128/384/512/896 proposed call limits,2700-second cooperative deadline,
2760-second supervisor, lease/ownership/cleanup and no automatic retries.
The exact worker/supervisor and current-generation/session seals remain
unresolved. The v2 private postflight must pass first.

After successful production dream and independent postflight, run the unchanged
final helper on Afrodite with the same config and config seal:

~~~sh
python3 -I -B /opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-r1/hymem_v64_postdeploy.py --config /opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-r1/rollout-config.json --config-sha256 CONFIG_RAW_SHA256
~~~

This verifies strict exact vectors and performs the reviewed doctor/MCP probes.
Only its passing final receipt permits claiming production vector health.
If bounded targeted dreaming leaves unrelated work or unverifiable episodes,
inspect and prepare a separate narrow copy rehearsal; no automatic scope
expansion or surplus-only pruning is implied.
