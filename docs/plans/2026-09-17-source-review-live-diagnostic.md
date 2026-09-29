# Source-review Stage A — live diagnostic

## Scope and authorization

The user replied “Continue.” to the explicit proposal to prepare the live launch
on Afrodite and, after its safety gates, send the 24 frozen invented controls
twice to `https://api.deepseek.com` with `deepseek-v4-flash`. The hard bounds are
48 paid completions / 144 HTTP attempts. Production memory, deployment,
restarts, full LME, resumed campaigns and automatic rerolls are excluded.
Earlier campaign authority was not reused.

The frozen protocol is `2026-09-17-source-review-evaluation.md`, SHA-256
`cab3abea17848397539d360d13ea68437f1ab12410ac13067c388bd8c51d705d`.
Requests, gold, schedule and all 488 frozen Python/SQL files remain unchanged.
Stage A is a candidate-only mechanism diagnostic, not a baseline comparison or
representative LME benchmark. Even a pass cannot establish LME readiness or
authorize runtime adoption.

## Prelaunch verification

A separate implementation agent adapted the one-shot runner and authenticated
entry; root reviewed the code and tests. A fresh read-only agent independently
reviewed launch isolation, authority, accounting, cleanup and scoring. Root
strengthened the independent auditor's checks of review receipts and entry
version, nonce, status and diagnostic-only flags before freezing the packet.

Local boundary tests: **471 passed**. Afrodite's separate network-disabled Linux
container repeated **471 tests**, then executed **96 supervised invocations**:
48 dry calls and 48 loopback HTTP calls with dummy credentials only. Both actual
receipt collections passed the independent auditor; parser replay exercised
malformed, uncertain and permissive outputs. No provider calls or credential
reads occurred during that gate. Root retrieved and verified the real JUnit,
receipt hashes, plan, helper manifest and Docker isolation record before
creating fresh authority.

The live container is separate from Hermes. It uses the pinned Hermes1 image
and read-only Python environment, a read-only root filesystem, UID 1000,
dropped capabilities, no-new-privileges and explicit resource bounds. Only the
new diagnostic directory, its scratch area and the exact read-only credential
file are mounted; no production store or broad Hermes home is mounted. The
credential is parsed privately, never exported or printed. Frozen code and
requests have read-only mounts. Each invocation has a 120-second absolute
deadline and two-second cleanup allowance. Semantically rejected/malformed
outputs remain data; safety, infrastructure or accounting failure halts the run.

## Receipts and pins

- Local package: `/private/tmp/hymem-source-review-live.MZiWWp`.
- Afrodite package:
  `/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-source-review-stage-a-20260917-MZiWWp`.
- Transfer packet: 540 files, SHA-256
  `4729c332d3b8fc09d27795fb3b038893f489bed6b96e1fac890dd2dbf8559c28`.
- Helper manifest:
  `5c0638566c1be26226a18f5a9685e07d2e030bde2dc7d10371eb42d99d12747a`.
- Target execution plan:
  `bebaf3fe37dc434ed0fb431f0e68fd076e026565f1a2255c6076982604ee5d14`.
- Target JUnit:
  `e0c10482e0ce241cd07bc332ded82c7d3decf07b08eb30b6860ea930f87e2281`.
- Target rehearsal:
  `6f898b1b49195fc4f2ebceb458dd326c600f7085a1b2438a2b58447d4ee7d2ed`.
- Fresh live authority:
  `9385f830ee65958754e5d72c5a439d018e392d845a0c8368b53e986fcb6fd491`.
- Gate container:
  `acd3682b412da6a2ae619f8f602355ee3fede853fb67ccdbe973a33175419c85`.
- Live container:
  `1b54fae5acc2560e2701e534f40b2147330440ee5882df9d8d1eac62854e1180`.

## Outcome

**Collection and integrity passed; Stage A quality failed. No deployment or
full LME run was performed.** Collection ran from 17:06:28 to 17:10:05 UTC on
2026-09-17 (217.33 seconds). All 48 completions returned on their first HTTP
attempt: 48 actual attempts, no rerolls, no unreconciled reservations and no
unknown token usage. All finish reasons were `stop`. All workers were reaped
with confirmed process-group cleanup and no cleanup warnings. The one-shot
authority is spent; unused transport capacity is not permission to repeat it.

A separate credential-free, network-disabled container ran the exact independent
auditor against live receipts before scoring. Root then retrieved the receipts,
independently re-audited all 48 worker collections and replayed all 48 unmodified
raw replies through the frozen scorer. Every target/status and reported group
matched. A fresh agent independently reviewed the three failed pairs, including
their faithful/defective counterparts and both repetitions. Root also inspected
the source, candidate, transmitted request and responses for each failure.

| View / domain | Repetition 1 | Repetition 2 |
| --- | ---: | ---: |
| Primary grounding: exact targets | 7/10 | 7/10 |
| Primary retention: exact targets | 14/14 | 14/14 |
| Additional retention: exact targets | 8/8 | 8/8 |
| Auxiliary grounding: exact targets | 6/6 | 6/6 |
| Faithful whole-scope acceptance | 11/12 | 11/12 |

Do not pool these denominators or treat paired controls/repetitions as independent
representative samples. All 48 responses met the structural format contract.
The integrity, format and token-ceiling gates passed; exact-target quality and
faithful acceptance failed. All labelled retention targets passed, **not** all
possible retention judgments: the boundary-speaker pair's ambiguous retention
facets were explicitly excluded before the run and remain excluded.

Usage was 99,962 prompt + 5,004 completion = **104,966 total tokens**. Mean total
usage was 2,186.79 tokens per call; maximum was 2,403, below the frozen 5,000 mean /
8,000 maximum engineering ceilings. Mean provider-call wall time was 1.28 seconds,
maximum 1.60 seconds; supervised end-to-end duration includes setup, validation,
receipt I/O and cleanup. These are observed diagnostic measurements, not dollar
estimates or a comparative cost/speed claim against the baseline.

## Complete failure map

Each row below failed identically in direction in **both repetitions**. All other
labelled targets matched, and all other faithful scopes were accepted.

| Case | Exact target | Expected → observed | Consequence |
| --- | --- | --- | --- |
| `boundary-speaker-defective` | Body relations `c3` | unsupported → supported | False accept; no other veto |
| `opaque-identity-defective` | Body relations `c3` | unsupported → supported | False accept; no other veto |
| `exclusivity-scope-faithful` | Title relations `c1` | supported → unsupported | False reject; title assertion `c0` also vetoed |

1. **Speaker attribution is present but not correctly applied.** The current
   canonical message is spoken by the assistant and the preceding boundary
   message by the user. The candidate reverses the actors. The model supports
   the candidate using canonical text `s0` and boundary context `s2`, without
   citing current-speaker metadata `s1`. This is not the earlier dropped-role
   projection bug: both messages' actual roles are visible on the wire. The
   observed decision fails cross-message attribution; fragmented representation
   is a plausible contributor, not a proven internal model cause. The response
   also calls material-fact retention `retained`; that unlabelled/excluded facet
   is not retroactively counted as a new scored failure or a success.
2. **Opaque identity metadata is over-authorized.** Canonical text states only a
   pale-glaze preference. The candidate adds a display name found only in an
   opaque `source_peer_id`. Both responses cite that contextual metadata while
   accepting the name claim. No prior summary is present in this episode request,
   ruling out prior-summary leakage for this case. A structurally valid canonical
   citation plus metadata citation does not establish the asserted identity.
3. **Bounded exclusivity is confused with global exclusivity.** The source defines
   approved services as exactly P and Q, with availability on P and absence on Q.
   Exclusivity *among approved services* therefore follows; exclusivity *across
   all services* does not. The faithful and defective titles receive identical
   unsupported assertion/relations judgments. Overgeneralizing the warning
   against unsupported exclusivity is plausible, but the outputs do not reveal
   an internal causal explanation.

The frozen gold and selectors remain defensible. No request omission, label
binding mismatch, clipping, parse failure, transport fault or scorer error explains
these three model decisions. The failure map is complete for this frozen design;
it is not evidence that no other semantic failures exist in real LME content.

## Next decision

Do **not** deploy this candidate or advance to the retained-case/full-LME run on
these results. Proposed next work is offline: colocate canonical text with its
exact message/speaker metadata while keeping authority types distinct; preserve
opaque IDs as identifiers rather than display-name evidence; and split broad
relations judgments into inspectable actor, identity and bounded-scope obligations.
Test role swaps, genuine canonical names, opaque IDs and closed/open-world scope
symmetrically, including renamed and structurally varied fresh controls. Required
citations alone cannot prove entailment. These are design hypotheses, not fixes
already shown to work.

Keep this failed protocol, gold and paid replies unchanged as development
evidence. Any new paid evaluation needs a separately frozen protocol and fresh
approval. No automatic rerun or further paid campaign was started.

Additional receipts:

- Scoring container:
  `032e92ca2cc4fe35f860785292496b0a9eee94cf23afe6897530df59b775439a`.
- Live collection summary:
  `130b2e746119ac22d67e0cc5b5393b999d028384b1f9bf8ff5f1d19fc3da3d92`.
- Live integrity audit:
  `33de6ba9b5a2b2ab337e7d5522e8be2f28e306c38512a17b6f49ee0b471e6a98`.
- Live score:
  `809280c79669cc75558def294cabf7159012609760d94d59b4f2471de6377755`.
- Root receipt audit / all-response replay:
  `6e44255e4428ad792fa7436ae4b7fafca3b5ac435c6d95fb2e9f7d6839f2c81b`.

Raw replies remain in the dedicated diagnostic package and its local receipt
copy, not repository artifacts. All 488 frozen Python/SQL sources still match
their pre-run hashes. Existing dirty-worktree changes were preserved.
