# Luna no-dream LongMemEval-S source and production promotion

The immutable source tag `lme-luna-nodream-82p6-source-20261003` points to
commit `6301a3a0e0c93c5f12630de8bd0fd662d6756a0e`. That source produced a
complete **Luna-only no-dream diagnostic** on all 500 LongMemEval-S questions:
413 judged correct, or 82.6%. It is not an official LongMemEval score or a
canonical R9 artifact. The runner disabled dreaming, embeddings, aggregation
nodes, and episode granularity; it used a local Luna reader and legacy custom
Luna judge. The result therefore does not measure production dreaming.

Evidence pins from the completed campaign:

| Artifact | SHA-256 |
| --- | --- |
| Campaign manifest | `b52b14fdcbc86a95f64c7657a8e78aa1be37e448add1e57cb31b3da7fc8b9438` |
| Campaign result | `300ab8ce58f76354b0d70b7b03c2ec7989f48038ba212715181c82abbe2f9367` |
| Source inventory (`source-map.json` in the tag) | `022b1e60f68afd1eac10fc48b76feb02d7c2ffbf4734d84bbd526229a61a2f17` |
| No-dream runner | `787925aed73b8e4d4a82979400d3650770d249d5e19bd2bc44b31b77198a4ad0` |
| Offline aggregate verifier | `8e8d384c0dbaddbd5f19e033ac09ab9198c715a54e729299b1dd38e9c198aef1` |

The production branch includes the exact-tag no-dream runner, launcher,
campaign driver, diagnostic verifier, and aggregate verifier under
`tools/diagnostics/`. They contain no credential values or benchmark answer
dataset. Their default state paths refer to the original Afrodite environment;
prepare a new private run root and supply its own runtime and dataset before
using them elsewhere. The scored dataset and private result capsules are not
part of this Git tree.

This production branch descends from public `Beam-optimisation` commit
`ce192dc2e9cf5bddca1b5f473d838468748ee994`, which itself descends from
`main` at `9be17e4b1605e223b33db851d018094015b61d17`. It retains the
aggregation canonicality fix introduced at `4857453606e8b1eef73d9d96f6e49cddc3c42a1d`
and the Hermes 0.21.5 session-key retirement artifacts. The source tag is the
exact scored version; the production branch is derived from it and keeps the
live Hermes1 dreaming and SQLite concurrency behavior.

`hymem/config.py`, every `hymem/query/*.py` file, and the benchmark adapter
and protocol match the scored source byte for byte. The following production
core files intentionally differ from the source tag. The aggregation change
keeps both capped fields canonical; the other files retain Hermes1's live
dreaming, extraction, summary, and SQLite operation behavior:

- `hymem/api.py`
- `hymem/core/db.py` (with production-only `hymem/core/serialized_sqlite.py`)
- `hymem/doctor.py`
- `hymem/dreaming/aggregate.py`
- `hymem/dreaming/digest.py`
- `hymem/dreaming/phase1.py`
- `hymem/dreaming/runner.py`
- `hymem/dreaming/semantic_generation.py`
- `hymem/dreaming/summary_recovery.py`
- `hymem/extraction/chunk.py`
- `hymem/extraction/contract.py`
- `hymem/extraction/producer.py`
- `hymem/extraction/prompts/__init__.py`
- `hymem/extraction/retry.py`

Publishing or deploying this production branch must not be described as a
reproduction of 82.6%. A new evaluation would be required for a score claim
about its full dreaming configuration.

## Verification environment

The isolated integration host runs Python 3.13. Its SIWC test interpreter has
`requests` but lacks `ijson`; the host-side Hermes virtual-environment
interpreter also lacks `ijson`. Therefore the no-call CLI startup suite stops
at the `ijson` import before its client boundary. This is a missing local test
dependency, not a passing startup check. The Hermes service containers use
their own `/home/node/hymem-env/bin/python` runtime and must be checked there.
Run
`tests/test_lme_current_model_startup.py` in a compatible interpreter with
`ijson` installed before rollout. The cap canonicality, schema migration,
SQLite connection-lifecycle, and Hermes session-retirement regression suites
run locally without this dependency.
