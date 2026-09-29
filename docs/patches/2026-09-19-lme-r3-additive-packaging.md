# Additive R3 packaging

Patches 00–11, their manifests, and their existing verification receipts are unchanged.
These artifacts describe isolated candidates; packaging did not modify application
source or tests in the main checkout, deploy anything, execute tests, or call an API.

Apply the application/test increments in numerical order after patch11:

- Patch12, `2026-09-19-lme-12-registry-cli-bootstrap.patch`, changes only the direct-CLI
  import bootstrap in `benchmarks/lme_registry.py` and adds one regression file with
  seven collected cases. Its manifest records the intermediate identity map before
  the Honcho test-wait adjustment.
- Patch13, `2026-09-19-lme-13-honcho-test-waits.patch`, changes two bounded waits in
  `tests/test_honcho_server.py` from 5 to 30 seconds. It changes neither application
  code nor scheduler cooldown behavior or assertions. Its manifest records the final
  R3 identity map: 230 source files, 221 test files, and 6,972 collected test cases.

The eight test-only auxiliary assets are separately pinned. The already recorded
`2026-09-19-lme-independent-summary-auxiliary-readme.patch` accounts for the README
correction in the 459-file combined R3 verification tree. It is not included in the
application patch chain or authorized as a deployment input by this packaging.

## Evidence boundaries

The full R3 suite has **not** been run. Collected node IDs are not passing results.
The retained17 paid diagnostic runs against the original 230-source snapshot,
not the CLI-bootstrap revision. Extraction implementation bytes are unchanged
between those snapshots; the new registry CLI bootstrap itself is not validated
by that retained diagnostic. This package claims no retained17 success, deployment,
full-LME completion, or benchmark readiness.

The packaging helper generated exact unified hunks from the pinned R2 and R3 trees.
A separate verifier is provided at the task root as
`helpers/verify_patch12_13.py`; parent review and execution of that verifier are
separate from artifact generation. It reconstructs both increments without fuzzy
matching, checks every source/test identity, and separately reconstructs the README
auxiliary correction. Its receipt is created exclusively, never overwriting prior
evidence.

## Artifact identities

- Patch12 SHA256: `710114d3d9e2fa7f7aa592f2c997ab60239c2439133a8d3e4cf836083aab7c3c`.
- Patch12 manifest SHA256: `7c7b25074250a44aebc7256c14c07a9ecd97386693f5af692df7a8a4ac73dbdc`.
- Patch13 SHA256: `4782ba7b03994848fe8cdfabb31e027ae2b611c1abc51aec31feb1e88bd0cada`.
- Patch13/R3 manifest SHA256: `573fd0f5d763adbbd9f26df7fa249a546e2980304d303771ea76fcec4526aa66`.
- Source-R3 manifest SHA256: `392bcea026823d2b0b38fd2bce89e1c16a522a17eb041bd11c002d904b45e76e`.
- Combined verification-R3 manifest SHA256: `df200c11e9a286e7160d222720da705718b700b5f40b9bbaf5c4ca7029cd21cc`.
