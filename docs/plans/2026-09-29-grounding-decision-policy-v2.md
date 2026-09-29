# Grounding decision-policy repair — diagnostic failed, stopped

Current outcome: the v2 implementation and isolated diagnostic passed their
offline mechanical gates, but the fresh live diagnostic did not pass semantic
validation. All owned processes are gone; both monitors are paused. No LME
restart, production activation or further model campaign is authorized by this
receipt. The chronological implementation and launch record follows; see the
terminal outcome at the end for the final status.

## Evidence and scope

The first source-grounding diagnostic failed (20/24 controls; retained canary
correction incomplete). Root replayed all 27 returned judgments against exact
requests and independently verified cleanup. No benchmark is running. See
`2026-09-29-semantic-judge-diagnostic.md` for immutable receipts and outcomes.

Root and a separate read-only Sol independently identified a specification
contradiction: the prompt permits predicate replacement but also prohibits
predicate substitution and directs unsupported originals toward rejection,
without defining correction priority. This explains an ambiguity; it does not
establish that fixing the wording will repair every observed semantic failure.
Exact-quote and context-scope guards worked and must not be relaxed. The false
acceptance of acknowledgment as actual use is a separate semantic error.

## Sequential plan

1. Fresh Sol implements an inactive `grounding_v2.py`. Preserve the complete
   v1 implementation except the explicit version and judge instructions.
   Define verdict priority: first assess the exact original full claim; if it
   is unsupported, consider allowed predicate-only alternatives preserving all
   other fields; require exactly one clearly entailed alternative for recovery;
   otherwise reject or mark uncertain. Never replace an already supported
   original. Acknowledgment, proposal, intent or preference alone does not
   establish actual operation. Evidence is a verbatim contiguous excerpt from
   one named region, not reconstructed/concatenated prose. Context outside its
   scope is unavailable to the decision, not merely unavailable for citation.
   No fixture names, labels, special cases, retries, relaxed guards or caps.
2. Root independently reviews the complete difference and tests parser,
   request/version isolation, decision-policy consistency, all old mechanical
   controls and rejection of old-version responses. Only then can another Sol
   implement deterministic frozen-candidate integration and identity binding.
3. Independently verify that candidate derivation preserves ordinary request
   bytes, publication atomicity, budgets, canary gold and all fixed labels.
   Preserve the original 508/510-file candidates, sources and receipts.
4. Only after a source-bound diagnostic/host/reader integration is separately
   implemented and root-accepted may a justified fresh test be recorded with
   a new immutable receipt. Keep the same 24 controls plus retained canary,
   29-call/500,000 observed-token/1,800-second bounds and model/auth. It must
   distinguish false acceptance, malformed outputs and missed corrections;
   root privately replays all new results before any acceptance claim.

No new live diagnostic is launched by this plan. Both monitors remain paused.
No full LME, production changes, cap increase, automatic reroll or new score.
Offline prompt tests establish mechanics and specification, not model accuracy.

## Pure policy accepted offline

Separate Sol implemented `hymem/extraction/grounding_v2.py`. Root independently
reviewed the full byte diff and required removal of one residual blanket
rejection sentence, replacing it with the explicit ordered correction rule.
Final SHA256:
`377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec`.

Root ran **218 passing checks**: 125 v2 checks (four Sol, 121 root, including
all 93 old adversarial controls rebound in memory) plus all 93 original-v1
checks unchanged. AST comparison proves only the version and system prompt
assignments differ. All 24 fixture requests preserve every claim/context field
and label; only version and resulting batch hashes differ. Cross-version
responses are rejected even when supplied the receiving version's batch hash.
The original module and fixture hashes remain unchanged. No model call,
activation, remote mutation or performance/accuracy claim.

A separate read-only Sol integration review identified the required follow-on:
derive a fresh 510-file runtime with v2 bytes at the existing grounding module
path, preserving all other candidate files. The current canary/core/host/reader
are intentionally v1-pinned and cannot be pointed at v2 or globally monkeypatched.
New source-bound versions must be independently verified before another test.

## Candidate accepted offline

A new Sol implemented `luna_semantic_candidate_v2.py`; root found and reproduced
an isolated-CLI import failure before use. Sol repaired it by loading the old
builder's exact hash-checked bytes via a sibling path before any execution.
Root independently reviewed the final implementation and ran **47 passing
checks** (eight Sol plus 39 root). These exercise the actual candidate's
publication, full-result recheck, multi-batch collision/conflict rejection,
budgets, provider failure, legacy provenance and cache mutation controls. A
tampered helper fails before execution; the real isolated CLI also succeeds.

Inactive candidate: `/private/tmp/hymem-semantic-policy-v2.aBQTUn6q/candidate`.
Inventory: `/private/tmp/hymem-semantic-policy-v2.aBQTUn6q/map.json`.
Exactly one of 510 files differs from the accepted v1-grounding candidate:
`hymem/extraction/grounding.py`; all other 509 files are byte-identical.

- Builder SHA256: `ed0c5c006d316fb0c762a2df6977313403e45fafdb914ea3c4f2f675a0165e3f`
- Map SHA256: `efea7eaeba0c4164a4cb082962d36b6fecc55f7c944e48ed172426505def812c`
- Inventory-file SHA256: `64818ffcea5c52aef3d10142bfe9cfe7445a88280e46f3546bd01a5533fd3963`
- Extraction identity: `hymem-extraction-contract-sha256-v1:dcfb634927ad7e2a555ff3054710239195effe0436b50a9d62fb7f6aee6c3180`

No remote writes or model calls. Next is an offline, exact-source diagnostic
integration; the old v1 canary/host/reader must not be pointed at this candidate.

## Next diagnostic integration

Use a new deterministic bundle derivation tool: verify all frozen helper input
hashes before interpreting them, then emit copies in a fresh private local
directory. Existing module-relative paths inside that new bundle are retained
so source/import checks remain effective. Exact-count source transformations
may change only schema/run identity, new grounding/candidate/canary hashes,
new inventory/cache identity, and the host's explicit invocation of the accepted
v2 builder with its additional accepted-candidate/map arguments. Stage timing,
transport, fixtures, scoring, budget and cleanup implementations stay unchanged.
Carry forward the independently verified user-bus startup adapter. Do not use
global monkeypatches or mutate the frozen old bundle to make new pins pass.

Root must verify the generated source deltas and run synthetic whole-canary,
whole-entry, accounting/fault, isolation and metadata-reader checks before
staging or any new live diagnostic. This is implementation work only: no launch
is triggered by generation or test execution.

## Diagnostic bundle accepted offline

Third implementation Sol supplied deterministic source derivation. Root's full
review caught an unchanged early-entry core hash and an output-boundary gap;
both were repaired before acceptance. Root then independently ran **89 passing
controls** (four Sol, 85 root): actual core inventory/import preflight, real
canary stack, two whole-entry simulations (29 synthetic judgments plus eight
ordinary replays), failure accounting, private replay, one-shot receipt,
metadata integrity, resource-policy and bus-isolation controls. No model calls.

Generator `luna_semantic_policy_bundle_v2.py` SHA256:
`7b66fd585f98e847f775115bb10db4aa91a3258b0ec6aa2fbf7f1dcb83143504`.
Accepted local bundle `/private/tmp/hymem-policy-v2-bundle-probe-fixed-0929`;
derivation receipt SHA256:
`da48fe6d703a6512d1253b555ff7bbac402ccd4c4619356ba2bb511dbd513a37`.
All 18 generated file hashes are in that receipt. Controls, labels, original
ordinary canary, transport, stage accounting and limits are unchanged.

The next justified test measures whether this explicit ordered decision policy
reduces the observed false support/malformed citations and repairs missed
predicate corrections on exactly the same fixed schedule. It is not an LME
rerun or a claim of repair success. First stage into a fresh private Afrodite
directory and verify an actual zero-inference containment smoke under the new
source pins. Only after independent smoke cleanup and a fresh inference receipt
review may the single unchanged 29-turn bounded diagnostic start. Capture new
outputs privately and root-replay them offline; no automatic reroll afterward.

Root reran all selected old/new contract, v2 candidate and diagnostic-bundle
tests together: **354 passed**. This remains a scoped offline suite, not model
accuracy or LME readiness.

First staging `/home/atta/.hymem-policy-v2-stage-tVGSNikJ` was rejected before
preparation completed: macOS archive metadata added 21 `._*` sidecars. No model
or benchmark launched. Preserve it; do not reuse it. Root verified an archive
with `COPYFILE_DISABLE=1 tar --no-xattrs` excludes those sidecars, then copied
the identical reviewed source bytes to fresh private staging
`/home/atta/.hymem-policy-v2-stage-Pvzszd2l`. Adapter and derivation-receipt hashes
match the accepted local bundle. Zero-inference preparation is underway.

Actual new-source smoke passed on Afrodite: effective policy verified, zero
admitted turns/model calls/tokens, and independent service/cgroup cleanup.
Root also ran the actual read-only entry preflight: 510 files and all eight
retained ordinary responses verified. Smoke root
`/home/atta/.hymem-luna-semantic-policy-v2-2sul68su`, base receipt
`7c1a75ec3159597a56ab088e92e1681a547d69fb71cfaef0a14b4bd161b84d28`,
smoke-only sidecar
`df6d68f6efa0556480fce452a39af95874a48600d1c0abcf0bc0efd024a2f2d2`.
Do not reuse or change the smoke's mode. Reader correctly leaves semantic
acceptance and diagnostic completed-and-clean false for a smoke-only run.

## Fresh policy diagnostic receipt

Prepared inference-only root
`/home/atta/.hymem-luna-semantic-policy-v2-hsgdr8yy`, unit
`hymem-luna-semantic-policy-v2-hsgdr8yy.service`.
Base receipt SHA256:
`8af35fa203784fccc1a184da9852d3ade9e25e479f4516e4ef7d9c54b1ad6f93`.
Inference-only sidecar SHA256:
`536587bde9eb62d2bbba754ff9dd9f4a3f5f7bda55344a662e28bfa6c1825b30`.
Adapter SHA256:
`087030bbdb2a2e2bcc6c79541bd716c2f633fcd8e400980db341662b8dde32aa`.

Root's new read-only rehearsal helper differs from the accepted prior helper
only in synthetic verdict/report schema labels (plus whitespace). SHA256
`81627d009be4cc0d175ee1bec726c1a30012aabcd0d1a35cf1d7ff92ef850c12`.
It denies network, process creation and filesystem writes, replays all eight
retained ordinary completions privately, and uses synthetic verdicts only.

Exact metadata-only observe command, after verifying the local adapter hash:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite \
  'python3 -I -B - observe --root /home/atta/.hymem-luna-semantic-policy-v2-hsgdr8yy --receipt-sha256 8af35fa203784fccc1a184da9852d3ade9e25e479f4516e4ef7d9c54b1ad6f93 --adapter-sha256 087030bbdb2a2e2bcc6c79541bd716c2f633fcd8e400980db341662b8dde32aa --adapter-receipt-sha256 536587bde9eb62d2bbba754ff9dd9f4a3f5f7bda55344a662e28bfa6c1825b30' \
  < /private/tmp/hymem-policy-v2-bundle-probe-fixed-0929/adapter-v2.py
```

Polling cannot launch, restart, reroll or make model calls. Require terminal
reconciliation and independent cleanup, then root private offline replay using
the generated `verdict-replay.py` SHA256
`7da9778ad93c60f290c530f4090bc714d4b36d25df9d141fec4aed77c2e97140`,
with entry SHA256
`8336d68d09a29f2aab8717b0fbb7fb7ef050a2728c58bb963809f52e8cff865e`.
No LME benchmark or further model campaign is launched by this receipt.

Root private rehearsal passed on the fresh inference-only root: all eight
ordinary request/response replays match, 29 synthetic judgments, zero new model
calls/files written. The 203 synthetic token sentinels test accounting only.
The single diagnostic launch returned zero and is permanently one-shot. The
monitor has been rebound to this exact root/receipt/adapter with read-only
polling; old runs remain stopped. A successful launch is not a semantic pass.

## Terminal outcome — not accepted

The single policy-v2 diagnostic finished after **173.132 seconds**, using
**27 new Luna turns / 146,308 known tokens**, with complete usage accounting.
There were 25 recorded unit outcomes: the 24 fixed controls and the retained
canary hybrid. **19/24 controls passed**, four produced malformed verdicts,
and one missed the required predicate correction. No false-support claim was
observed in this draw. Three claim-level false rejections belong to malformed
units; they are not three additional failed controls.

The hybrid consumed all eight retained ordinary responses and two new
grounding judgments. The table judgment supported its original claim; the
prose judgment returned unsupported, so correction and full recheck did not
complete. Thus `core_completed`, `all_semantic_checks_passed`,
`semantic_fix_accepted` and `completed_and_clean` are all false. There was no
transport first-failure or budget stop. Terminal validation alone is not a
passing result.

Root independently replayed all 27 captured judgments privately against their
exact requests, reproducing all 24 control outcomes and the hybrid result.
Private result SHA256:
`7b0ad987a235288d4af84db14aa50cc726de41e853de916ca1b8feb95c66c458`.
The read-only replay made no model calls or writes and exported no raw text.
The finite failure-metadata helper SHA256 is
`690e34cba42ae2be00608beee46844c137cf0ffdb4893842bbca33903e7cb866`.

Residual failures, using the fixed control indexes and finite codes:

| Control | Observation |
| --- | --- |
| 9, conversation reference | `response:binding` |
| 10, two conversation records | `response:shape` |
| 13, prefer-to-use correction | `unsupported`; missed correction |
| 16, unrelated citation | `response:binding` |
| 19, out-of-prefix context | `evidence:context_scope`; guard correctly rejected invalid evidence |

The metadata reader independently confirmed source/receipt/inventory integrity,
all 25 owned process groups absent, the service stopped in terminal failed
state, and `failed_unit_cleanup_verified=true`. Success-specific
`unit_cleanup_verified` remains false; this is verified failed-run cleanup, not
clean successful completion. Both `monitor-luna-lme-pilot` and
`finish-lme-validation` are paused. All private failed evidence is preserved;
production is unchanged and no benchmark is running under this receipt.

## Diagnosis and next decision

Root and a separate read-only Sol agree that the explicit v2 correction order
removes the previously proved prompt ambiguity. The remaining unsupported
judgments and malformed responses are model decision/output-contract failures
under that repaired wording; this run does not demonstrate a new transport,
parser or accounting defect. The context-scope guard behaved correctly and
must remain strict. Comparing two single draws (20/24 previously, 19/24 now)
does not establish a statistical improvement or regression.

Another prompt reroll is not justified. A next architectural design could
separate trusted caller-side request/response association from semantic model
judgment. Copying a 64-character batch hash is not itself cryptographic proof
of association; an explicit versioned transport/parser contract could instead
bind the exact canonical batch to the returned response while retaining strict
verdict order/count, attribution, quote, scope and cross-batch isolation checks.
This must be designed and adversarially tested as a new contract, not implemented
by removing the v2 binding gate. The current subscription adapter records
`response_format_effective=None`; availability of a schema-enforced output path
has not been established here and must not be assumed.

Mechanical association/schema work would not by itself repair missed semantic
corrections or prove resistance to false support. Those require a separately
specified validation design and evidence. No additional inference, contract
relaxation, model switch, cap increase, LME launch or production change follows
automatically from this failure. **LME readiness remains unverified.**
