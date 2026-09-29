# Fresh targeted-repair diagnostic (v15)

## Scope and status

The user requested “Run it.” after acceptance of the local targeted-summary
repair. This turn uses a **fresh** four-retained-case / eight-control diagnostic,
not the spent v14 authority. Bound: `deepseek-v4-flash` at
`https://api.deepseek.com`, at most 32 completions / 96 HTTP attempts,
120 seconds per invocation plus two seconds for cleanup, stopping on the first
failure. No automatic rerolls, campaign resume, production memory submission,
deployment, service restart, or full LME benchmark is included.

**Final status: halted on task 1/12 after three completions / three HTTP
attempts.** The semantic verifier returned incomplete JSON. Eleven tasks are
unattempted, not passed. The one-shot run is closed; LME is not cleared.

## Fresh frozen inputs

The source freeze reauthenticates the three final local gates from
`2026-09-16-targeted-summary-repair.md`: 2,248 distinct test IDs, 60 selectors
across 55 files, 453 Python/SQL files plus two unchanged auxiliary inputs.
The current digest/probe hashes match that accepted local source.

- Source manifest: `902b687ae31fd36d789fbdc8605862affaccd2e8fe45eb2e845f16564b7babbf`.
- Source archive: `a630badf63df1016d8772a57957c4cc5c0634e3b8b07d000db9bfe534b2d06e3`.
- Local private staging: `/private/tmp/hymem-readiness-v15.2oX1IS`.
- Afrodite private staging: `/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-targeted-readiness-20260916-6O540h`.

Only existing benchmark fixtures are used. Raw evidence, requests, replies,
credentials and the benchmark database stay out of repository artifacts.
The retained benchmark SQLite file is opened immutable/read-only, with its
hash and absence of sidecars checked before and after execution.

## Runner adaptation and acceptance sequence

A separate agent adapted the private helper copies; root reviews the diff and
runs independent controls. Historical gates and authorities remain untouched.

- Semantic verification has four arrays; candidate-only format verification is
  a separate mandatory call after final semantic acceptance.
- Control candidates are unchanged and routed to the task owning their
  criterion. Each control makes one call, with no repair or retry.
- One targeted repair requires validated, exact-source-linked diagnostics and
  retains the original generation input. A rejected repair cannot trigger
  another repair or gain acceptance through format checking.
- Scripted rehearsal paths are preregistered as 6/5/3/3 plus eight controls:
  25 synthetic calls. This does not force extra live calls or preapprove live
  content.
- Completion/HTTP reservations, request replay, single-use dispatch,
  process-owned deadlines and worker reaping remain fail-closed.

The agent's first broad helper gate exposed two stale call-count expectations;
those were corrected and their focused rerun passed. Failed receipts are
retained as failures. Final acceptance requires root's complete helper gate,
the exact current-source Linux gate, the matching Linux helper gate, and an
independently audited no-network rehearsal before fresh live authority exists.

Disposable Linux test/rehearsal containers have networking disabled, read-only
source/runtime mounts, no credentials or production database, and explicit
resource limits. Scripted semantic verdicts prove control flow only, not model
quality. A successful bounded live diagnostic would still need an end-to-end
LME smoke before canonical-baseline readiness could be claimed.

## Independent local helper acceptance

Root's complete gate passed **304 tests, zero failures/errors/skips**, with
unchanged inputs and no non-loopback networking. It includes seven additional
root-owned controls for mandatory final format, exclusion of source authority
from that task, and termination after a rejected repair. Dummy loopback HTTP
tests exercise the real SDK and process supervisor without provider credentials.

- Local JUnit: `2cfe0a82c2e2da1c5b32de41f76f86dae0a774d212709f4a80dc5456c72c1f40`.
- Local report: `a14ab2f3510bdabbb9f051ec6596eb99c03491184f9f88602f01554fb598e98e`.
- Helper manifest: `14cd949ab73b2eb69d79b65da7163ff29c26ac3b9b397fd9cbf58cecea7d8a73`.
- Helper test manifest: `a4e7ffd99d8fd08fea6eba6ec8f5f10fc3b246887819af216563eb3c12098724`.
- Helper archive: `05c85712303cd0a3cfc462710ef37789875b36f0480a9ce6c46cd53672e2a405`.

The exact helpers/tests are now staged for a matching isolated Linux gate.
No live authority exists at this stage.

## Linux helper acceptance

The same **304 tests passed on Afrodite**, zero failures/errors/skips or external
network attempts, exact test inventory, unchanged helper inputs. Root's separate
audit accepted the container's zero exit, network isolation, read-only mounts,
current source hashes and unchanged installed source/runtime.

- Linux helper JUnit: `fe74405da273e5466a7f80ddda74d53eb309b1a9335e3143446313453815aa14`.
- Root helper verdict: `fbd1a4c11e22937e699aea4f1bab4538a858a764e64deb9b2813ea0cae151f17`.

The source gate is still pending; these helper results are not live semantic
evidence or authority to skip that gate.

## Linux source acceptance

All **2,248 selected source tests passed on Afrodite**, with zero
failures/errors/skips/network attempts. Root independently reconciled the exact
test inventory, zero container exit, unchanged candidate source, read-only
mounts and unchanged installed source/runtime. This remains a selected
regression suite, not the complete repository suite.

Root source verdict:
`1effeb6906ce70fddfd992a30a23868fe1b3545216102810a75e8830f57f644f`.
The credential-free, network-disabled rehearsal is the next required gate.

## Rehearsal accepted and one-shot dispatch

The independent rehearsal audit passed: twelve tasks, exactly 25 synthetic
calls in the preregistered schedules, zero provider attempts, no database
writes, all workers reaped, unchanged read-only inputs and eighteen verified
mounts. This is mechanical evidence only; synthetic approvals do not measure
live model quality.

- Source Linux JUnit: `e1f33fda99b0039c78398219920c45de09b55c2b9bb66ad7222b3ad2aed54659`.
- Root rehearsal audit: `194e3157ad46200b8503b52f934eab4ddd241423237529e6cb6213df871522ec`.
- Fresh v5 authority: `487954f6943dfa8aca08de10b5c8949d63b4a78c548eb2eafa7e542414218b62`.
- Current-turn consent: `ffe682be47300eaa7b701d286bc3db40ad13bdfe61a5daa3bbce75ca92d0b26c`.

The accepted dispatcher successfully started the one-shot campaign after its
credential-free preflight. Results are pending; no additional diagnostic,
automatic reroll, deployment or baseline is authorized by unused capacity.

## Final live result: verifier parse failure

The first retained pipeline case stopped after **9.491 seconds**, with these
three calls: primary generation, summary compaction, semantic verification.
All three HTTP responses returned normally; no transport retry occurred.
The runtime rejected the last reply as `fidelity_parse_failure`. No targeted
repair, semantic reverification, candidate-only format verification or later
task was attempted. No derived output was accepted or published.

A metadata-only inspection of the authenticated saved replies found:

| Call | Response characters | Completion tokens | Requested maximum | Parsing |
| --- | ---: | ---: | ---: | --- |
| Primary | 1,898 | 535 | 3,072 | Valid strict JSON |
| Summary compaction | 433 | 94 | 3,072 | Valid strict JSON |
| Semantic verification | 794 | 254 | 3,072 | Incomplete JSON; delimiter expected at end of reply |

The JSON decoder stopped at character offset 794, line 1, column 795.
The response had no whole-response fence. This was a syntax failure, not a
parsed unsupported verdict.
The reported token usage is well below the requested output ceiling. **The
capture did not preserve the provider's finish reason**, so its reason for
ending cannot be established from this receipt. This is not evidence that the
new targeted repair succeeded or failed: that path was never reached.

Total recorded usage: 7,981 prompt tokens + 883 completion tokens = 8,864 tokens;
three completions, three HTTP attempts and three successful responses. The
campaign stopped after its first failed task; eleven tasks remain unattempted.

Root's final audit verified receipt hashes, the worker's already-recorded
mechanical replay result, exact usage, first-failure termination, all owned
workers reaped, unchanged benchmark SQLite bytes/no sidecars, and unchanged
installed source/runtime. This audit **did not execute a replay or call a
provider**. An initial safety review held the command pending source inspection;
root inspected its exact remote source and guard, then the same metadata-only
audit was accepted and completed. No restriction was bypassed.

- Final summary: `8daa1ea473ff07be0071f9e675cd47a94f46bc73e65f60db403ea33bb155d66a`.
- Root live audit: `da9f40b73564463b8efb983c04c347382f48d51fcb36637f35b984d63182a38a`.
- Structural inspection script: `b3d717d2e759f29ac3651ed310a91f6b8d0438f9a480083be4117750b90bd312`.
- Verifier response hash: `b37590c84d7b69c3fa8496a98fc50c2004a13c81503500a0d674e1ae23b4c0b5`.

All 455 frozen local inputs still match the tested candidate. This turn changed
private diagnostic helpers/tests and documentation, not application runtime.
No production database was opened, no deployment/restart occurred, no paid
reroll was made, and no full LME benchmark ran. The immediate blocker is malformed
verifier output; safe structured-output recovery and finish-reason capture need
investigation before a new bounded diagnostic. Canonical-baseline readiness
remains unconfirmed.
