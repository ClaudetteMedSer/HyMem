# Approved bounded live verification of the three contract repairs

The user approved this fresh pass after the offline repair report. This is not
a resumed campaign, deployment, service restart or full LME benchmark.

## Frozen scope

- Exactly 17 previously retained benchmark-only cases: ten facts and seven
  digests, in the prior round-robin order with chunk cases omitted.
- The frozen candidate is the 225-source-file set in
  `docs/patches/2026-09-19-lme-contract-repair-manifest.json`; it passed the same
  595 targeted tests locally and in the Hermes Python 3.11.2 runtime.
- Endpoint `https://api.deepseek.com`, model `deepseek-flash`, temperature 0,
  JSON output, thinking disabled and existing 3,072-token fact/digest bounds.
  Primary requests must match their retained input fingerprints exactly.
- Maximum **34 completions and 34 HTTP attempts**: two per case, no HTTP retries,
  repetitions, rerolls or resumed campaign. This ceiling is not a target.
- Owning deadline: 120 seconds per completion, 30 minutes for the campaign,
  followed only by bounded cleanup and read-only evidence auditing.
- Independent application-output rejections are recorded and collection
  continues. Transport, accounting, source-integrity or cleanup failure halts.

## Isolation and admission

New Afrodite root:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-contract-live-20260919-MJTviH`.
The spent earlier campaigns and their receipts remain untouched.

Use a fresh UID-1000 container with pinned source, helpers, dependency runtime,
and retained benchmark stores read-only. Credentials are mounted read-only only
for the live pass, read in-process, and passed to owned workers through private
stdin—not command arguments, environment dumps or output artifacts. Transport
allows only the exact HTTPS chat-completion URL, with redirects, environment
proxies and retries disabled. Raw requests and replies stay on Afrodite.

Before spending any call, independently review the adapted adapter/runner/auditor,
pass local controls, and pass credential-free, network-disabled positive and
negative rehearsals on the real inputs. Both rehearsals must exercise two calls
for every fact and digest case (34 scripted completions, zero HTTP). Audit every
reservation, request/response hash, worker cleanup receipt, all 225 source hashes
and the independently pinned ten-store ledger, including the case-less store.

Only metadata such as hashes, statuses, counts and usage may leave Afrodite.
Accepted fact-prefix recovery must report its actual result source/cursor
fingerprints and pending tail; it is not proof that the original input was fully
indexed. No result may claim semantic truth, recall improvement or full LME
readiness merely from parser acceptance. Producer fingerprint availability and
stop/truncation status remain explicit.

## Status

Preflight completed: 19 adapter, 66 runner and 79 final-auditor tests passed,
plus 57 saved-replay controls. The two existing real-input rehearsals each
completed all 34 scripted calls and passed independent accounting, source and
ten-store hash checks. The auditor needed two preflight-only corrections:
source-preserving repair envelopes (not identical user strings), and UTF-8
JSON hashing of accepted fact provenance. Direct frozen-application controls
now cover both. No application changes were made during preflight.

The single approved live campaign has started in container
`hymem-contract-live-MJTviH`, manifest SHA-256
`906309c48b13c15c59ad7fefcf5058ae049a195205f8d21e5a84acc2d7ff27a9`.
Its output is exclusively `live/campaign/` under the new remote root. This
campaign becomes spent after any paid request; it will not be rerolled or resumed.
Preflight receipt: `docs/patches/2026-09-19-lme-contract-live-preflight.json`.

## Completed result — not an LME-readiness pass

The fresh campaign ran from 07:40:04 to 07:41:29 UTC on 2026-09-19 and
completed all 17 cases using **18 completions / 18 HTTP attempts** of the
34/34 ceiling. The container exited 0, PID 0, no OOM. Every response finished
with `stop`; all named `deepseek-flash` and supplied the same available producer
fingerprint. A label/fingerprint is not a hidden-weight version guarantee.

- Facts: 10 accepted, including two accepted-empty outcomes. Every fact case
  used one call; the new smaller-prefix recovery was therefore NOT exercised
  by the live provider in this pass. The previously failing fact case returned
  eight items on its unchanged primary request; this cannot be causally
  attributed to the recovery code. One different fact case accepted a primary
  source window with a pending tail, not complete session coverage.
- Digests: six accepted, one rejected. The prior episode-title failure case
  accepted three episodes, one procedure and a 429-character summary.
- The **same retained summary case** (`digest-23b1c169…`, sequence 4) still
  failed. Its primary summary was 674 trimmed Unicode code points. The one
  permitted targeted repair returned a valid summary-only JSON object but
  594 code points, exceeding the unchanged 500-character hard maximum. Both
  replies finished normally; this was not output truncation, transport failure
  or an accounting fault. The new prompt did not solve reliable length
  compliance for this case. It remains a known unresolved blocker, not a newly
  discovered schema failure.

Independent accounting reconciled every call, reservation, request, response,
token count and owned-worker cleanup receipt. All 225 source-file hashes and
all ten independent retained-store hashes remained unchanged. A separate
network-disabled replay regenerated every request byte-for-byte before feeding
its recorded response back, and reproduced all 17 complete application outcomes
exactly. It made zero additional HTTP attempts. Final auditor controls: 79
passed; saved-replay controls: 57 passed under root verification.

Reported usage: 35,505 prompt tokens + 7,926 completion tokens = 43,431 total.
No production service or memory was modified, and no deployment, restart or
full LME run occurred. This campaign is spent; unused budget is not permission
to resume, reroll or repeatedly test variants.

The next implementation target is robust summary-length recovery, with this
674-to-594 response pair retained as an offline regression. Do not present the
current summary repair as complete, silently truncate it, increase the storage
limit, or relax the benchmark integrity gate to obtain a pass. A fresh live
validation would be a separately scoped campaign after that design is tested.

Full source-free receipt:
`docs/patches/2026-09-19-lme-contract-live-verification.json`.
