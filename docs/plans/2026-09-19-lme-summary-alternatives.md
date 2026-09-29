# Bounded whole-summary alternative selection

The user approved addressing the remaining summary-length recovery failure.
The prior 17-case campaign is spent and its receipts stay immutable. Its only
rejection was the same summary case: 674 code points, then 594 after repair,
against the unchanged 500-codepoint limit. No deployment or full LME is included.

## Implementation plan

1. Work from the frozen, verified 225-source-file candidate in a new isolated
   workspace: `/private/tmp/hymem-summary-alternatives-20260919.aP3n27/candidate`.
   Preserve the unrelated dirty main checkout and all prior snapshots.
2. A separate implementation agent changes only summary recovery and related
   tests. Keep primary requests, source windows, original-input/draft envelope,
   two-completion ceiling, schema, items and atomic publication unchanged.
3. The one repair call requests three independent complete summary alternatives
   in order of decreasing detail, targeting 350, 220 and 120 characters. Every
   alternative must stand on its own and preserve source-supported meaning;
   they are not fragments. The hard storage maximum remains 500 code points.
4. Validate a strict three-string alternative envelope, then select the first
   meaningful whole candidate actually within the cap. Do not truncate or join
   strings. An overlong alternative does not invalidate a different fitting
   alternative; malformed structure or non-string members invalidate the whole
   envelope. A valid historical single-summary response may retain its exact
   existing behavior, without allowing mixed or extra fields.
5. Root independently verifies selection, Unicode boundaries, malformed and
   all-invalid bundles, source/item preservation, semantic identity changes,
   cancellation, and real dream publication/held-failure behavior. Run the
   combined regression selection before promoting the candidate artifacts.
6. Verify saved failure behavior without inventing a new provider response.
   Any fresh provider test must use a fresh declared campaign and its own cap;
   do not resume or reuse the previous campaign's unused allowance.

## Honest limits

This is a bounded alternative-selection algorithm, not guaranteed model
compliance. If every candidate is invalid, the digest still fails honestly and
advances no cursor. A parse/length check does not prove semantic fidelity or
retention of every topic; shortening must not be marketed as such. A synthetic
fitting alternative proves wiring, not successful recovery on the retained
real case. Do not increase the storage cap, silently clip output, change the
summary coverage policy, or weaken the benchmark gate to obtain a pass.

## Verification protocol and current status

Implementation is frozen in incremental patch
`docs/patches/2026-09-19-lme-07-summary-alternatives.patch`. It applies after the
previous 00–06 chain, not directly to the unrelated dirty checkout. The new
digest module SHA-256 is
`29623e62264897b9a087fc655c15c9e952e09cc995d11a1d15036557d3f5bfa7`.

Root's isolated combined gate passed all 672 tests, without failures, errors
or skips. The exact collected node set matched its manifest, and all 225 source
and 206 test file hashes were unchanged before and after. An independent
52-test audit-harness gate also passed. Target-runtime tests were still in
progress at protocol freeze; their completed results appear below.

The approved fresh live diagnostic is deliberately recovery-only: two saved
primary drafts (644 and 674 characters) from the same retained benchmark case
are supplied locally, and each receives exactly one newly generated repair.
This is two historical drafts, not two independent source cases. No fresh
primary generation, campaign resume, reroll, full benchmark, deployment or
production memory is included. Maximum: two paid completions and two HTTP
attempts, 120 seconds per call and 900 seconds for the campaign. Endpoint:
`https://api.deepseek.com`; model label `deepseek-flash`; temperature zero;
JSON-object mode; thinking disabled; 3072 output tokens. The label does not pin
hidden provider weights. All old campaign budgets remain spent.

Frozen live manifest SHA-256:
`7f035601c375ca8d54e52fe64236e7ffff39c4a402afd7d1273d510795737ae1`.
Live output directory (fresh; no calls made when this protocol was recorded):
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-summary-alternatives-live-20260919-6Ez6TF/live/campaign`.

Credential-free, network-disabled positive and negative rehearsals run first.
An independent auditor checks exact regenerated repair requests, complete
candidate selection, accounting, worker cleanup, all 225 source pins and all
ten retained store pins. It also replays the entire application path using only
recorded responses. Raw benchmark text and replies remain on Afrodite. A pass
establishes repair-path behavior on these two drafts, not general semantic
fidelity, full-LME completion or production readiness.

## Final result: candidate did not resolve the retained failures

The selected offline gates passed: 672 local tests and a 471-test subset on the
actual server runtime, with zero failures, errors or skips. Both credential-free
rehearsals passed independent accounting checks and exact application replay.
Patches 00–07 reproduce all 225 source and 206 test file pins. These are selected
regression gates, not a claim that the entire repository suite was run.

The live diagnostic ran once on 2026-09-19 at 08:18:55–08:19:09 UTC, using exactly
two new completions and two HTTP attempts. Both responses finished normally;
neither was truncated or a transport error. **Both repairs were rejected.**

| Saved draft | Generated alternative lengths | Result |
| --- | --- | --- |
| 644 characters | 638, 544, 529 | All exceed 500; rejected |
| 674 characters | 678, 585, 503 | All exceed 500; rejected |

The separate network-disabled audit replayed the exact full application path
with these saved responses and reproduced both rejections. All ten retained
store and 225 source-file hashes matched. Usage: 1,984 prompt tokens and 737
completion tokens (2,721 total). The runner and auditor exited zero with PID zero
and no OOM; a clean campaign exit means complete accounting, not successful
repair. No retries, rerolls, additional paid calls or deployment followed.

This establishes that asking for increasingly short alternatives did not fix
these observed failures. It does not establish a model-wide failure rate or a
provider-side root cause. The policy still demands preserving earlier topics,
specific values, qualifiers and new outcomes within a hard fixed limit; that
is a plausible pressure against shortening, not a causal result of this test.
Guaranteeing arbitrary retained detail in a fixed-size summary is not possible.
Further prompt-only variations should not be described as a deterministic fix.

**Do not promote patch 07 as a successful LME repair.** It is retained as an
auditable unsuccessful candidate, not applied to production or the unrelated
dirty main application files. The smallest next step is an offline semantic
feasibility check against the retained benchmark source: enumerate the required
information and construct a reviewed, faithful reference of at most 500 code
points. This experiment did not prove such a reference impossible. If feasible,
use it to evaluate a scoped recovery design; if not, review an explicit bounded
summary content-priority/overflow policy without deleting source records. A
change to topic-retention policy was outside this experiment. Do not use another
paid reroll as a substitute for resolving that design question.

Durable metadata receipt:
`docs/patches/2026-09-19-lme-summary-alternatives-verification.json`.
The candidate manifest remains immutable and records its pre-verification
pending state; this report and verification receipt give the final result.

Status: **live recovery unsuccessful; LME readiness not verified. Nothing
deployed. The two-call campaign is spent and must not be resumed.**

## Subsequent offline feasibility review

The user approved the proposed reference exercise. A separately reviewed
500-character topic/main-proposition reference was constructed and accepted
unchanged against both historical primaries using scripted responses. The
501-character negative controls still failed correctly; no API calls or
application/production changes were made. See
`docs/plans/2026-09-19-lme-summary-feasibility.md` for semantic limits and six
offline checks. This narrows the diagnosis: inability of the failed provider
responses to fit is not proof that this case's main information cannot fit.
It does not rescue the unsuccessful paid campaign or verify model recovery.
