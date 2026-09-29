# Record-review evaluation — isolated Afrodite rehearsal

## Scope

Continue from the [offline runner verification](2026-09-17-record-review-runner-verification.md).
Prepare a separately reviewed, one-shot launch adapter and rehearse it on Linux
without provider calls, credentials, production stores, deployments, restarts or
a full LME run. Existing paid approvals are spent. This work creates no new paid
authority and leaves all frozen authorization flags false.

Local package: `/private/tmp/hymem-record-review-launch.ZgDmVC`.
Remote package:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-record-review-stage-a-20260917-ZgDmVC`.
Neither replaces an earlier package. The 30 cases, source snapshot, expected
labels, paired schedule and frozen protocol remain unchanged.

## Implementation and independent review

A fresh implementation agent adapted the accepted offline runner and the prior
one-shot entry. Root reviewed the implementation diff, authorization path and
tests, then ran independent boundary and receipt-auditor controls.

The entry binds fresh explicit consent to the exact plan, helper bytes, root
review and actual target-rehearsal receipts. Authority must have a finite,
unexpired validity window of at most 24 hours. Exclusive durable dispatch
reservation precedes credential access, with a second authorization/expiry
check after reservation. Per-task and worker checks preserve the 60 completion /
180 HTTP-attempt caps, 120-second deadlines, two-second cleanup and prohibition
on campaign resume or semantic rerolls. Thinking is explicitly disabled in both
SDK modes. The capture/supervision helpers remain byte-identical.

Local gate: 538 tests passed (276 implementation/authority, 244 independent
auditor and 18 independent boundary controls). Missing authority is rejected by
the actual entry subprocess without producing a dispatch or touching a key.
Unit authority fixtures use invented values in disposable test directories;
there is no real consent or authority file under the staging runner.

## Initial isolation failure and correction

The first Linux container stopped before tests or collections because the image
itself contains `/home/node/.hermes/.env`, despite no credential bind mount.
The new absence guard rejected this. A separate network-disabled metadata-only
probe confirmed a regular file, mode 0600, UID 1000, size 24,322 bytes. Its
contents were not read; this does not establish whether it contains live keys.

A separate implementation agent created a supplemental v2 host launcher. Root
reviewed it and independently reran its 48 passing isolation tests. It masks the
image's entire Hermes home with a new empty read-only directory, then overlays
only the diagnostic package. Frozen code, requests and executable inputs are
read-only; only new diagnostic receipts and scratch are writable. The existing
Python environment is mounted read-only. Network is `none`, root filesystem is
read-only, UID is 1000, all capabilities are dropped, no-new-privileges is set,
and CPU/memory/PID limits are enforced and inspected before start.

The supplemental script requires its independently checked SHA-256 and uses new
exclusive v2 directories and a distinct container. Original packet bytes and
the failed attempt's receipts remain intact. No benchmark case or collection
was rerolled: the first attempt stopped before either collection began. No
production process or image was changed.

Initial container:
`49d456b43eea3dfca697e5d276c067756db0eea8a975a3f5287de1ea8d08ebab`.
Corrected rehearsal container:
`26e834a9403731a737023a7edd2714477f3b26e2829d91f02ca14435d76e88a8`.

## Frozen transfer

The allowlisted archive contains 561 files, including 502 frozen Python/SQL
sources and the unchanged preparation artifacts. It includes no production
memory, credential files, prior paid outputs, paid authority or collection
receipts. The receiver verifies the complete archive hash, member inventory,
types, sizes, traversal restrictions and individual hashes before exclusive
extraction. Original source/preparation bytes were rechecked locally afterward.

Pins:

- Archive: `ed52da88726ffdf6e79372b79a56b05eb74eeb297b936992b50038fd9e152f41`.
- Runner: `476450127d175455500de051599a3252506f6060b2f0c81bf7c2a27abf69d43c`.
- Entry: `9ee762071322638f808474ee90ab8465c744affddd175f34d117c65485f92f11`.
- Helper manifest: `26f1fab17af62529b1e3d316a5e821e042d56d1147de660094421494be375187`.
- Isolation v2: `339f286cc9585911b92b2734912c4cb1847152aa53c0dc39729375161ab5d73c`.
- Frozen protocol: `f8f6887029febf11a6a7dd4f9af7eb9a28cc64fbbf9ae7f68c212d62d49e4105`.

## Completion gate

The corrected Linux container passed the same 538 exact test identities, then
completed 60 dry invocations and 60 invocations through the maintained SDK to a
dummy localhost server. It exited zero without OOM. Its network remained
disabled. Only the latter collection made HTTP requests: 60, all localhost.
The first attempt was not resumed; neither completed collection was rerun.

An allowlisted, hash-verified archive of 1,303 receipt/input files was retrieved
for independent local review. All 28 actual-receipt attack controls passed.
The initial local snapshot comparison flagged Docker's ordering of its 35
mount records. Only list order changed: every complete mount record matched
when keyed by destination. A separate agent corrected this representation-only
comparison. Root reviewed and reran 102 controls: mount order is ignored, but
duplicate destinations, changed fields, wrong types and other configuration
drift still fail. Together with the 28 receipt attacks, that gate passed 130
tests. It did not alter a receipt, runtime input or benchmark result.

Independent request/usage/deadline/cleanup audits of all 120 invocations passed.
There were no Linux cleanup warnings. The stopped container also has PID zero.
Recomputed parser/scorer replay is byte-identical to the target replay (SHA
`fcd65bc05791a6ec1a4cb32a67d3651f4bbe7e44862be9e43547ca02f2ad14c5`).
The initial final comparison used an in-memory Python tuple against a decoded
JSON array, producing a second representation-only false alarm despite those
identical bytes. Its bounded correction compares the exact serialized receipt
contract, with negative controls for real value/type/order changes. The first
local replay directory remains preserved; no collection is being rerun.

Receipt archive:
`49e1da5e43385377508e1ee5cd98192fc0536955f659b4ec878080de9fde376b`.
Target execution plan:
`8c920ea4c3c72b84a492ad40121638cced33f4b9bc0c851eb57cd42f9b396098`.
Target test receipt:
`34ac8d7ec0c752f9800f012e4fc85168133f1f1188deceb97315b1334dfca6e0`.

## Final acceptance

Root reviewed the serialized-replay correction and independently reran the
combined 145 receipt/snapshot controls. The complete retrieval verifier then
passed, including exact target-plan relocation, original packet/helper hashes,
container isolation before and after execution, all individual request and
accounting receipts, 120 unique worker reservations, cleanup, and an independent
parser/scorer replay. No collection was repeated to obtain this result.

The final local gates contain **731 unique test identities**: 538 launch/auditor,
48 isolation and 145 actual-receipt/snapshot controls. The 538 Linux identities
repeat the corresponding local set and are not added to that unique count.
All final gates passed with zero failures/errors/skips. This is a targeted
diagnostic-readiness gate, not a new full-repository suite.

Each synthetic collection retains 48 fresh and 12 replay task instances. Scoring
keeps the two cohorts separate, with 52 and 16 labelled target instances
respectively. Dry responses are deliberately malformed; SDK responses exercise
malformed, uncertain and structurally permissive parsing. Their invented token
counts and verdicts provide no estimate of real cost or semantic accuracy.

Final evidence under the local package:

- `root-target-verification-v2/target-verification.json`, SHA
  `c8eddcf113f19a7005fed237a5b75413c918d6a6827177decfe64c29c799d05c`.
- Final verifier, SHA
  `611ec984654d0e7baaee9e7e5f70700fd23dbe90c8a264745de044067421b257`.
- `final-receipt-tests-v2.xml`, SHA
  `b25109124f6597dee00fe0e23d4ccdb7eda81e5b4996fb2a529b3fd6838c72ed`.
- `closeout.json` binds the final unique test inventory and scope limitations.

All owned rehearsal containers have exited. No provider call, real credential
read, production change, restart or full LME run occurred. The original 502
source files and 38 preparation artifacts remain unchanged. The failed first
isolation attempt and local verifier attempts are preserved, not overwritten.

## Next approval boundary

The target-host offline rehearsal is complete. The next proposed experiment is
the frozen candidate-only diagnostic: 30 benchmark-only controls (6 replay and
24 fresh synthetic), two repetitions, at most 60 paid completions / 180 HTTP
attempts to `https://api.deepseek.com` using `deepseek-v4-flash`, with 120-second
invocation deadlines. This excludes production memory, deployment, restarts,
full LME, rerolls and resumed campaigns. It requires fresh explicit approval;
no live authority or consent file has been created. A subsequent live launch
must retain the reviewed Hermes-home mask.

Live semantic accuracy and LME readiness remain unverified. Passing this
rehearsal neither establishes them nor authorizes a paid campaign.
