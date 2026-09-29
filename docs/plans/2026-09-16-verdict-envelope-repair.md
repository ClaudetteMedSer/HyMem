# Bounded verifier-envelope repair

## Diagnosis and scope

The closed v15 diagnostic failed on the first semantic-verifier response:
794 characters, 254/3,072 output tokens, JSON syntax failure at EOF. A read-only
inspection of the authenticated saved reply's **redacted grammar** established
that every individual verdict and diagnostic object was complete. Only the
final summary-verdict array and root-object closing delimiters (`]}`) were
missing. The reply explicitly rejected summary content and supplied two
`omitted_outcome` diagnostics; closing the envelope must not turn that rejection
into acceptance.

Provider finish reason was not recorded, so the underlying serving-side reason
for the missing trailer is unknown. No new provider calls are authorized or
needed for the local fix. Raw benchmark evidence remains private on Afrodite.
Historical receipts and the spent v15 authority stay unchanged.

## Sequential implementation and verification plan

1. A dedicated subagent implements a verdict-only, bounded closing-envelope
   parser. Root independently tests rejection boundaries and reviews the diff.
   Only a missing root `}` or direct final-array/root `]}` may be supplied, after
   explicit completion of every verdict/diagnostic object. No inferred strings,
   scalars, fields, indices, verdicts, commas, item closures, or prose stripping.
2. Verify that exact schema coverage, unsupported/uncertain vetoes, source-linked
   diagnostics, the single existing summary repair, full semantic reverification,
   and mandatory final format validation all remain intact. Any affected
   diagnostic consumer is corrected separately and verified before proceeding.
3. Run frozen, credential-cleared, network-denied regression gates, including
   the prior 2,248-test selection and new independent controls. Verify the actual
   saved reply through offline validators with no provider calls or publication.
   Record source identity and test receipts; do not claim live LME readiness.

Primary generation, summary-repair parsing and the shared extraction parser
remain strict. This fix adds no completion, HTTP retry, model reroll or deadline
reset: normal three / worst-case six calls remains the contract.

Status: runtime fix and diagnostic-consumer fix independently accepted offline.
No deployment or new live diagnostic has been performed.

## Step 1 accepted: runtime fix and authentic saved-reply check

The first agent's focused regression gate passed 581 tests. Root reviewed its
diff and independently accepted a frozen, credential-cleared, network-denied
296-test gate: zero failures/errors/skips, unchanged inputs and zero provider
calls. The only runtime change is `hymem/dreaming/digest.py`; global extraction
parsing, prompts, wire schemas, call counts and existing factual validators are
unchanged. Digest semantic identity changes automatically through loaded-code
hashing; controls confirm Phase-1, fact and profile identities are unaffected.

Root's additional five recorder controls passed. Raw replies and hashes are
preserved even when the outer trailer is recovered; rejected final semantics
or formatting still count as failed session digests rather than partial success.

An isolated **actual saved-reply** check then ran old and fixed validators on
Afrodite, with networking disabled, read-only mounts, no credentials, no database
opens and no generation/publication. The original reply hash remained
`b37590c84d7b69c3fa8496a98fc50c2004a13c81503500a0d674e1ae23b4c0b5`:

- Old validator: `fidelity_parse_failure`.
- Fixed validator: `summary_content_unsupported`, with both `omitted_outcome`
  diagnostics passing exact-source validation.
- Both agree with explicitly appending `]}` for comparison. The rejection is
  preserved; no model approval was manufactured.

This establishes that the specific parse blocker is repaired, **not** that
the subsequent model-generated summary repair will pass a new live run.

Accepted digest SHA-256:
`5fd88358644dfacf3d714655bfb0d5243091d8bb9b5c74d71b2b5b385b603ade`.
Root step-1 JUnit:
`524445ee7b6a877e95ad3ee6a1bf654b7076c4a173044804d563ba61ba533ea4`.
Root recorder JUnit:
`81b2c6c90c34c69859f0e55a3b44d15c5b6568120025398d22cecc73082cfb53`.
Private local receipts: `/private/tmp/hymem-verifier-json.6hAJk0`.
Private Afrodite receipts:
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-verdict-repair-20260916-YxSLqa`.

## Step 2: affected private diagnostic consumer

After accepting the runtime fix, root found the private direct-control runner
independently decoded replies with the old strict parser. That would disagree
with the revised production validator. A new agent aligned this consumer
and its parent-side validation to the exact captured raw reply (including character
caps), while preserving the original provider bytes and bounded single-call
controls. It also added safe per-attempt finish-reason metadata for future
captures; absent historical metadata remains unknown, never inferred.

Changes are confined to **new local helper copies**. Historical v15 helpers,
receipts and closed authority remain unchanged. The copies require fresh source
binding/review before any future live diagnostic and do not authorize another
run.

## Final application regression gate accepted

Root reran the entire prior 2,248-test selection plus 225 additional parser,
closure and recorder controls: **2,473 distinct tests passed**, zero failures,
errors or skips. Three shards share the same frozen 456-file Python/SQL input
manifest. The independent audit checked exact test identities (no duplicates or
missing prior tests), XML/manifest hashes and unchanged source after completion.
Environment credentials were cleared and all network access denied; there were
zero network attempts or provider calls. This is a selected regression gate,
not the entire repository suite.

Compared with the closed v15 candidate, `hymem/dreaming/digest.py` is the only
changed application/benchmark runtime file. `git diff --check` is clean. The
root audit SHA-256 is
`eb604f3be64bd907843c0c776a771474f7dc597cd0b9d31386d570f28c39524d`.
The authentic saved-reply validation receipt SHA-256 is
`c0af349dcc0df0f4c5d9e40fc28b041c8cd82811ca7cbc78c9e60039ba9f4aa6`.

The diagnostic agent separately passed 69 new synthetic/dummy-loopback tests.
Root reviewed the complete two-helper diff and added independent raw-envelope,
exact-response binding, output-cap and finish-metadata privacy controls. The final
combined inherited/new helper gate passed **393 distinct tests**, with zero
failures/errors/skips, unchanged inputs and zero non-loopback network violations
or provider calls. Its historical v15 candidate and bundle are binding-test
fixtures only, not current source clearance or live authority. The two test
gate totals are reported separately; they must not be added as a deduplicated
repository-wide total.

Helper JUnit SHA-256:
`908a9d797c159fdadc66625cfe8417285f608b3694ecfe8e1f9b755424e9e855`.
Helper frozen-input manifest SHA-256:
`e72e4238421e404ddd050118342fc27153f51dbb78318008c1e9100bf137e1de`.
Accepted helper hashes:

- `live_validation.py`:
  `1ab17f274cba1d5d7d282c6da5e2f4d7a4cad09bac7d70db2c343df044792538`.
- `supervised_validation.py`:
  `ed9fba02df36a0944511164302288a339be6e10289c9bf587c0b2f880a8b2fa6`.

## Handoff and remaining acceptance boundary

The specific saved-reply parse blocker is fixed. Unsupported/uncertain verdicts
still reject content, and a targeted repair must independently pass complete
semantic and format verification. Primary generation and summary repair remain
strict JSON consumers. No extra calls, retries, prompt changes or benchmark
tolerance were introduced.

The provider's reason for omitting the outer trailer remains unknown because
the historical capture lacks finish metadata. Future captures distinguish safe
known finish reasons, absent/null fields, unknown values and unavailable data;
they never infer a finish reason from token counts or JSON shape.

The verified changes and offline receipts are preserved as
`offline-fix-bundle.tar.gz` with `offline-fix-manifest.json` in the private Afrodite
receipt directory above. The bundle's helper draft is explicitly non-live and
contains no new authority or credential. Production and old stages are untouched.

**LME readiness is not yet established.** A fresh source/helper binding and
target verification, a freshly authorized bounded live diagnostic, then a
successful end-to-end smoke are still required before a canonical baseline.
