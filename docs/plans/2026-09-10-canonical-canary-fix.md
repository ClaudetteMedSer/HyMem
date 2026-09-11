# Canonical extraction canary repair

## Plan, registered before candidate model calls

The reported failure is an actual missing prose claim, not a malformed response,
source-offset error, missing context, or entity-type normalization failure. The
candidate is a general production-prompt clarification: resolve references using
the exact permitted context before extracting owned claims, and distinguish this
from inventing a relationship or publishing a context-only assertion.

Keep the canary fixture, evidence matcher, source bytes, validators, limits,
model (`deepseek-v4-flash`), thinking (`auto`, effective disabled) and temperature
unchanged. Do not swap padding, raise retry limits, loosen success criteria,
introduce fixture-specific rules, or select the first lucky passing run.

1. Independent read-only audit; separate agent implements the candidate and
   prompt-contract tests. Root reviews the patch and runs independent tests.
2. Private Hermes1 code copies, no production store access: one baseline and one
   candidate full canonical canary as a development screen. A failed candidate
   stays failed and is recorded; any subsequent revision needs a new explicit
   hypothesis, not undocumented sampling.
3. If the screen succeeds, freeze candidate bytes and run five predeclared paired
   baseline/candidate canaries in randomized within-pair order. Require all five
   candidate gates to pass; report all baseline outcomes without claiming a
   general reliability rate from this small sample.
4. Check separate synthetic positives and negatives for reference resolution,
   negation, wholly context-owned claims, unrelated adjacency and applicability
   limits. Never use private production conversations as prompt-tuning data.
5. Test extraction, retries/budgets, ownership, contract/cache identity and strict
   benchmark integration. If supported, version the extraction contract and
   document benchmark/cache consequences before a scoped deployment.

Model work is limited to 160 logical completion calls (at most 480 provider
attempts under the unchanged retry policy), across all development and
confirmation runs. Each canonical attempt keeps its existing 24-call ceiling.
No scored LME/BEAM/LoCoMo run or production dream is part of this repair.

## Initial evidence

- Independent audit: four initial leaves; right prose content 2,277 characters;
  exact 320-character boundary context; indispensable referring sentence inside
  the applicability interval. The extractor and canary see the exact context.
- The initial prompt says "never cite surrounding context" before its later
  permission to resolve a boundary-spanning assertion. This is an ambiguity,
  not yet a proven explanation of provider behavior.
- Deployed extraction prompt, chunk implementation and contract match local
  baseline hashes. The deployed canary additionally supports an explicitly
  non-comparable operator override; this repair neither uses nor changes it.
- Existing unrelated working-tree changes and remote client extensions are
  preserved. Private raw requests/responses, if retained, stay on Afrodite.

## Development screen

Baseline: failed, 8 calls, table only, prose exact-context requests 2/emissions 0.
Candidate 1 (prompt source SHA256
`9ed879038f6cf7be58d58e70984a5e764b03de5db94aa6cb9bb666e5fe407311`):
failed, 8 calls, two valid triples but only one exact expected claim. This is
not a pass. No confirmation batch is authorized by this outcome.

Next diagnostic, declared before calling: one exact candidate prose-leaf primary
request, with the production request fields and strict parser, to inspect the
unmatched synthetic claim. Keep this development observation separate from a
future confirmation batch. No fixture or expectation changes.

The single leaf diagnostic returned a clean empty (1 call). It cannot explain
the earlier unmatched item. One full development trace is therefore declared
before calling, solely to identify which leaf produced it; it is not a
confirmation run and cannot license adoption even if it happens to pass.

That development trace hit an instrument bug after extraction: the recorder
stores `(request, response)` pairs, not bare responses. The diagnostic output
was not trusted; its entire 24-call reservation remains charged conservatively.
A separate agent is fixing and testing the recorder before a repeat. This was
not a production error or a passing extraction result.

Candidate 2 hypothesis, before model calls: deterministic JSON sorting presents
the authoritative continuation before its preceding interpretation context.
Render validated fragment metadata/context before authoritative content, with
all reparsed keys, values, source offsets and context bytes unchanged. Screen
this presentation change with the ORIGINAL baseline prompt so the lever is
isolated from candidate 1. No source/fixture/padding changes or duplicate views.
The same independent confirmation gate applies if this screen succeeds.

Candidate 2 screen passed: 8 completions/8 provider attempts, both supported
claims, exact normal execution path and no markers. Original prompt SHA256
`92f2f7ad9fc7cfc4d905d2fbf15b861ab8a042b26d3b41f56f857dcb099d04da`;
derived contract identity suffix
`2efe72acfec0eabfdd862a7794cde2624eba4eabf18db67508a75d1c229e94b7`.
Candidate 1 prompt changes are being withdrawn, not combined with this fix.
The failed candidate 1 trace will not be repeated; its conservative 24-call
charge remains. Budget ledger is 49 charged/reserved calls before confirmation.

Confirmation now runs the planned five pairs (seed 20260910, shuffled order
within each pair), with identical prompt/client/gate hashes pinned per arm.
All five candidate gates must pass. Ten synthetic held-out semantic cases per
arm remain separate from this canonical criterion.

## Confirmed outcome and scope

All five original-presentation confirmation runs failed with the missing prose
claim; all five context-first runs passed the full strict canary. Every run used
eight successful completion/provider calls. No favorable-run selection occurred.

The ten predeclared synthetic primary-call cases scored 7/10 baseline and 9/10
candidate. No previously passing case failed with the candidate. One baseline
failure was only underscore-vs-space spelling in the diagnostic matcher (its
negation/polarity was correct); another cited context beyond its applicability
range, which the candidate correctly rejected. Both versions still returned
empty for a separate deployment-reference case. That remains a failed diagnostic,
not renamed as a pass; these were primary-only probes, not full extraction runs
with empty verification. No claim is made that all semantic omissions are fixed.

Root's final local regression gate passed **574 tests**, including the 15 new
presentation cases. The diagnostic-recorder repair separately passed **32 offline
tests**; its failed development attempt stays charged and excluded from evidence.
The installed Hermes1 checkout separately passed **376 tests**, zero failures,
in 169.72 seconds. The benchmark skill's new reference was checked with Hermes's
actual parser: YAML metadata and all original body bytes were preserved.

The only retained production change is the 16-line context-first serializer in
`hymem/extraction/chunk.py`. The public prompt version remains v20; exact prompt
bytes match baseline. The derived extraction contract changes automatically via
its executable component, preventing old cache/attempt rows from being reused as
current. New benchmark results must record this different contract. Activating it
in a persistent memory engine entails bounded re-extraction; no production dream
or service restart was requested or performed for this benchmark repair.

Hermes1's benchmark checkout received the exact tested file via a one-file
hash-pinned installer, with an original-code backup and no-op/unknown-edit refusal
rehearsed first. Existing client customization and operator-override code were
preserved byte-for-byte. New benchmark processes load the fix; already-running
memory services keep their previously loaded code. Hermes2/3 were not changed.

Installed-code fingerprint verification reproduces the tested candidate's exact
eight synthetic requests without provider calls or store construction:

- chunk.py SHA256: `eb96764a56a55540f87ca5ac89cefdf2952ade504d0cfae04898b75bb8f958ab`
- request-set SHA256: `c6c8139630496ba56fd7df4f1613669663fd942e6625695aaf69a9d04fcfdb0b`
- fixture SHA256: `3fedadfbcdf35a013f8f61c5ea3f6ccc3e150aca0d025d8597327704551e94de`
- active fresh-process contract: `hymem-extraction-contract-sha256-v1:2efe72acfec0eabfdd862a7794cde2624eba4eabf18db67508a75d1c229e94b7`

Model accounting: 125 measured logical calls plus the failed diagnostic's
conservative 24-call charge = **149 charged/reserved of the 160-call cap**. Known
calls used one provider attempt each; no price was available in client telemetry.
No further paid call or scored benchmark was launched after the controls.

Private receipts: `/tmp/hymem-canary-fix.8YGPKrDM/`; persistent original-code backup
and activation receipt: `/home/node/.hermes/scripts/hymem-canary-fix-20260910/`.
Raw synthetic traces stay on Afrodite; no production corpus was read or sent to a
provider. Existing September 10 recovery backups remain untouched. Local starting
HEAD was 775ce35; there is no new commit or push.

A passing canary is an entry gate, not a new benchmark performance score.
