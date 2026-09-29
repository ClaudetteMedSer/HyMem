# Sequential repair of the current-route diagnostic failures

The user requested fixing the three recorded failures. This pass continues the
isolated deployment candidate, not the unrelated experimental main checkout.
Workspace: `/private/tmp/hymem-contract-repair-20260919.nM2ESH`;
`baseline/` is copied from the prior immutable combined-verification tree;
`candidate/` receives sequential edits. No live database modification, deployment,
service restart, benchmark resume or full LME run is included.

The paid 20-case campaign is completed and spent. Its source and replies remain
private on Afrodite, unchanged. New offline regressions use invented text and the
observed failure shapes. A later paid verification must use a fresh declared
budget and directory, never resume or rewrite those receipts.

## Repair and acceptance order

1. **Facts capacity recovery.** Keep the configured eight-item limit and never
   truncate or merge facts merely to fit it. The failure returned nine otherwise
   valid facts. Its actual request was 4,389 characters, so reducing a configured
   12,000-character maximum to 6,000 can waste an attempt on identical text.
   Permit one immediate, fresh-unit-only retry on a genuinely smaller source
   prefix measured from the selected input. Rebuild the cursor, occurrence list,
   input hash and slice identity for that prefix; its tail remains pending.
   Bound the invocation to two completions. Preserve exact historical replay
   boundaries and fail-closed behavior. A separate agent implements; root reviews
   and tests before the next implementation starts.
2. **Summary-length recovery.** Preserve the database's 500-character hard limit,
   original source/prior inputs and atomic episode/procedure publication. The
   observed 644-character summary was repaired to 635, still invalid. Review a
   bounded, explicit revision protocol instead of truncation, increasing the
   storage cap or silently treating a failed repair as success. No unbounded
   retries or new semantic-verifier architecture. Design review may run in
   parallel, but implementation follows acceptance of step 1.
3. **Episode title alias.** Read-only on-server classification confirmed that the
   sole unknown key is exactly `episode_title`, with a nonempty string value,
   while `title` is absent. Consider a narrowly versioned lossless normalization
   of this one unambiguous alias. Preserve exact title value and every other
   field; retain full item/provenance validation, reject conflicting spellings,
   unknown extra fields and genuinely missing titles. Implement with a new agent
   only after step 2 passes root verification.
4. **Combined verification.** Run adjacent regression gates and target-runtime
   checks, preserve source identity/version commitments, package incremental
   patches against the prior candidate, and verify their reconstruction. Offline
   saved-output checks must distinguish unchanged primary requests from newly
   constructed recovery requests; never feed an old reply as if it came from a
   different request. Report any untested live recovery explicitly.

## Non-goals and honest acceptance

Accepted-empty chunk results are not recall evidence and will not be changed
into invented triples. This repair does not loosen benchmark integrity checks
or claim full-LME readiness merely from synthetic tests. A bounded algorithm can
still honestly reject a model that continues violating the output contract.

## Step 1 accepted offline

Only `hymem/dreaming/facts.py` changed in application source. Root's independent
218-test facts/authority/FTS/semantic gate passed with zero failures, errors or
skips. Three root-authored regressions failed against the unchanged baseline and
passed against the candidate: same-invocation smaller-prefix recovery plus exact
remaining-tail publication, and propagation of second-call ordinary exceptions
and cancellation without persistence. All nine invented facts were retained.

Snapshot: `facts-accepted/`. Receipt: `receipts/facts-root-gate.xml`.
Source SHA-256: `85411929079bbc135968a7e0044ffc0ed4e8c3c998cbd60dc6286373130a232a`.
Historical published-unit extraction remains one call with exact old boundaries;
the configured fact cap is unchanged. Fresh extraction now has a maximum of two
completions, so later diagnostic budgets must account for that explicit bound.

## Step 2 accepted offline

The repair request includes a JSON envelope containing the exact original
generation input and the complete overlong draft explicitly marked untrusted.
It gives measured length and excess feedback and requests a targeted revision
against the original evidence. This replaces the existing repair call; it does
not add another call or loosen the length limit. The primary request is unchanged.
No offline test can guarantee the provider follows the revised instruction; a
635-character repair must still fail honestly.

Root's complete 272-test digest, summary, lossless-publication, durable-status and
semantic-generation gate passed with zero failures, errors or skips. Independent
controls prove exact original-input/draft preservation and continued atomic
rejection of the observed 644-to-635-character failure shape. An obsolete test
assertion was changed to the new JSON-envelope contract; its cursor/retry checks
remain intact. The primary request and two-call maximum remain unchanged.

Snapshot: `summary-accepted/`. Receipt: `receipts/summary-root-gate.xml`.
Digest source SHA-256:
`1d38c155e7edaf118b686291143714ab9ab159db7afceaafb72ea916f51f77c8`.
Schema remains 61; only the digest semantic identity changes from step 1.

## Step 3 accepted offline

Root reviewed the admission-only normalization and independently passed 257
tests covering alias controls, summary repair, episode/procedure behavior and
semantic identities. Thirteen independent controls include canonical parity,
non-mutation, ambiguous/invalid fields, provenance and conflicting identities;
the positive control fails against the explicitly import-pinned pre-fix snapshot.
An earlier baseline command resolved candidate imports, so its receipt is not
used as baseline proof. Strict staging still rejects the alias spelling.

Only `digest.py` changed in application source for this step; prompts, schema,
call counts and publication limits are unchanged. Snapshot: `episodes-accepted/`.
Receipt: `receipts/episode-root-gate.xml`. Source SHA-256:
`3f561ab20717b1b2b5302a46c17797c0998d043e4cd24718ed475ff6b3321ac7`.

Step 4 is accepted offline, as recorded below. A separate bounded live verification of the 17 retained
fact/digest cases (34 completion / 34 HTTP-attempt maximum) was requested from the
user; no new paid campaign has started.

## Combined verification completed

The same frozen 595-case selection ran locally and in an isolated
Python 3.11.2 container on Afrodite, network mode `none`, no credential or
production-store mounts. Source/test inputs are read-only. An initial preflight
correctly rejected macOS AppleDouble archive entries before executing tests;
clean archives passed inventory verification. No application code changed.

The independent retained-response replay has completed with zero HTTP calls.
The actual episode failure now accepts all three episodes and its 389-character
summary against the byte-identical original primary request. Fact recovery
constructs a genuinely smaller 2,041-character request after the original 4,389;
summary recovery includes the exact 644-character rejected draft. Both stop
before needing a fresh model response, so neither is represented as a live
recovery success. All 17 fact/digest primary requests, three chunk source
inventories, 225 source files and the independently pinned ten retained stores
remain unchanged as applicable. No database persistence occurred.

The seven-patch chain (prior 00–03 plus new 04–06) reconstructs all 225 final
source/config files and 204 test files, with exact hashes. The unrelated dirty
main application checkout has not been modified. Durable incremental patches,
candidate manifest, patch verification and offline replay receipt are under
`docs/patches/2026-09-19-lme-*`.

The local combined gate has now passed all 595 unique tests with zero failures,
errors or skips. Post-run hashes match all 225 source/config and 204 Python test
files. Receipt: `receipts/combined-root-gate.xml`. The identical Python 3.11.2
target-runtime selection also passed all 595 unique tests with zero failures,
errors or skips (`receipts/combined-target-gate.xml`). Root independently compared
both full test-ID sets against the frozen expected collection; all match.
Independent helper controls also passed: 29 for the offline gate and 22 for the
saved-response verifier. Those are separate from the 595 application tests.

Target container `hymem-contract-offline-portable-xCcK6D` exited 0, PID 0,
no OOM. Its source, test and dependency mounts were read-only; only the new
diagnostic output mount and temporary scratch space were writable. Network mode
was `none`, UID 1000:1000, all capabilities dropped, no-new-privileges enabled.
The gate reverified all 225 source/config and 204 Python test hashes after
execution. Source changes from the prior candidate are exactly `facts.py` and
`digest.py`. Independent cross-fix review found no introduced interaction defect.

Final machine-readable receipt:
`docs/patches/2026-09-19-lme-contract-repair-verification.json`.
The local combined gate took 620.459 seconds; the target gate took 2203.317 seconds.

## Release status at completion of the offline gate

All three implementation fixes and their offline regression gates are complete,
and the patch chain is reproducible. No production deployment, service restart,
production-memory modification or new paid completion occurred. The unrelated
main application working tree remains untouched.

The saved episode failure is demonstrably repaired. The fact and summary changes
construct improved recovery requests, but those requests have not yet received
fresh model responses. The separately requested 34-completion/34-HTTP bounded
verification remains pending user approval; the prior paid campaign is spent.
No full LME completion, retrieval quality, recall improvement or comprehensive
production-readiness claim follows from this offline gate.

## Subsequent approved live verification

The user subsequently approved the fresh 17-case/34-call verification. It
completed using 18 paid completions and 18 HTTP attempts, with 16 accepted and
one rejected case. Independent accounting and exact offline saved-response
replay passed. All source and retained-store hashes remained unchanged.

The summary repair is **not sufficient**: the same retained digest returned
674 characters, then 594 on its one repair, exceeding the 500-character hard
limit. No fact recovery call was needed in this live pass, so fresh acceptance
of the previous fact case is not causal evidence for that recovery change.
No production deployment or full LME run occurred; LME readiness remains false.
The campaign is spent and will not be resumed or rerolled.

See `docs/plans/2026-09-19-lme-contract-live-verification.md` and its source-free
machine-readable receipt under `docs/patches/` for the complete result. This
supersedes the pending-approval state above without rewriting the offline history.
