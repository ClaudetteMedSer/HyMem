# Orphan quarantine: clone-only rehearsal

## Scope

The user approved preparing and verifying a quarantine repair **on a clone**.
Production deletion, database replacement, migrations, service restarts and
provider calls are outside this authorization. No missing session is fabricated.
The [backup-selection audit](2026-09-08-orphan-backup-audit.md) found no
authoritative restoration source among the inspected backups.

The reviewed target is `chunks` rowid 1058 in hermes-1. Selection requires a
fingerprint of the complete native chunk row, including column names, native
types and rowid; rowid alone is insufficient.

## Expanded dependency audit

Root created a fresh 168,337,408-byte SQLite backup from a read-only production
connection. The exact chunk row matches the earlier diagnostic snapshot.
Root separately checked 74 physical business tables, 45 scalar/reference
checks, seven structured JSON columns, keyword postings and the vector index.

The broader audit found **four `entity_mentions` links** that the earlier
limited provenance audit had not included. This corrects the earlier two-row
estimate; it is not evidence that the links were newly created. Approved clone
deltas are therefore:

| Logical data | Rows removed from working clone, preserved in archive |
| --- | ---: |
| `chunks` | 1 |
| `chunk_embeddings` | 1 |
| `entity_mentions` | 4 |
| `vec_chunks` | 1 |

No additional matching scalar or structured provenance dependencies were found.
The helper additionally scans physical rows and effective aggregation source
views, refusing unexpected dependencies instead of allowing broader cascades.

## Implementation and local acceptance

A separate implementation agent supplied
`tools/deployment/rehearse_orphan_quarantine.py` and
`tests/test_orphan_quarantine_rehearsal.py`. Root reviewed the frozen candidate.
The helper creates a new private directory and fresh exclusive files. It has
**no production apply mode** and cannot use an existing database as its write
target. Only its new working clone receives the normal HyMem connection/UDF
wiring, with foreign keys and stock triggers enabled; initialization and
migrations are not called.

Before deletion, it commits, reads back and fsyncs a native SQLite quarantine
archive. Rowids and SQLite values/types, including vector BLOBs, are checked.
It also proves a separate full-baseline backup restore. Restoring that baseline
deliberately reproduces the original known FK fault. **Archive-only reinsertion
of an orphan into the guarded runtime schema is not tested or claimed.**

Root's accepted local gate used the project's vector-enabled interpreter:

```sh
/opt/anaconda3/bin/python -m pytest -o addopts='' -q \
  tests/test_chunks.py tests/test_vectors.py \
  tests/test_embedding_recovery_health.py tests/test_terminal_chunk_loss.py \
  tests/test_orphan_quarantine_rehearsal.py \
  --junitxml=/private/tmp/hymem-orphan-neighbors-root-vector-enabled.xml
```

**99 passed in 29.54 seconds**, including all 16 rehearsal cases. The cases
cover native archive values, recovery, dependency/fingerprint refusal, reused
destination refusal, archive/precommit/cancellation rollback, unexpected
trigger changes, missing FTS deletion, vector-extension refusal and real
keyword/vector retrieval with a maintained offline embedding stub.

An initial run accidentally used `/usr/local/bin/python3`, whose SQLite build
could not load sqlite-vec: 93 passed, one skipped, five fixture setup errors.
The accepted run above exercises those cases with the extension available;
no application changes were made to bypass those failures. This is targeted
verification, not a full repository-suite run.

Frozen helper SHA-256:
`37855f4699df0776a617a36e48973aed319995bc049220a6f74979e89a7159da`.

## Afrodite rehearsal

The verified helper was streamed into hermes-1's installed runtime without
installing or modifying application code. It received only the private fresh
snapshot, not the production database. Runtime SQLite is 3.53.4, schema 59;
sqlite-vec is available. No provider/bootstrap environment was constructed.

The helper returned `verified_clone_only`, with the exact row counts above,
native archive roundtrip and full-baseline restore verified, and zero remaining
foreign-key violations. It checks schema and every unrelated logical row,
vector, keyword posting and FTS document membership before committing and again
after commit. FTS internal integrity checks preserve the deliberately selective
extraction index; no FTS rebuild or incorrect all-content comparison is used.

The read-only source snapshot is byte-identical before and after, SHA-256:
`c2dc743f660dff780a18133643c83b6972e04de97b738a990906035fe21df94c`.

Root's separately written verifier then passed in 15.7 seconds. It does not
reuse the helper's snapshots or serialization: it independently compared 89
logical table/index projections, unchanged schema, exact native archived rows,
all five databases' SQLite integrity, full-baseline restoration, and the
expected FK change from one fault to zero. An actual keyword `MATCH` retained
all unrelated matches and lost only the target; the stored-producer/dimension
candidate SQL used by semantic retrieval retained every unrelated candidate.
No production embedding service or end-to-end LLM response was exercised.

A final read-only production check confirmed the exact target chunk remained,
with its durable vector and four entity links. Production was not repaired or
modified by this rehearsal. Root's independent scripts and local test XML are
retained under `/private/tmp/hymem-recovery-root.bKUByT/` and the XML path above;
they contain code/synthetic data only, not copies of production data.

## Retained private artifacts

Inside hermes-1 on Afrodite:

- `/tmp/hymem-quarantine-preflight-ynsudhxc/baseline.sqlite`: fresh read-only
  production snapshot.
- `/tmp/hymem-quarantine-preflight-ynsudhxc/rehearsal/baseline.sqlite`: preserved
  rehearsal baseline.
- `/tmp/hymem-quarantine-preflight-ynsudhxc/rehearsal/working.sqlite`: repaired
  clone only.
- `/tmp/hymem-quarantine-preflight-ynsudhxc/rehearsal/quarantine.sqlite`: exact
  native archive of the six business rows and one logical vector-index row.
- `/tmp/hymem-quarantine-preflight-ynsudhxc/rehearsal/baseline-restored.sqlite`:
  verified full-baseline restoration.

Directories are `0700`, database files `0600`. These are temporary rehearsal
artifacts, not a durable production disaster-recovery backup. Private database
contents stayed on Afrodite; only code, counts, booleans and hashes crossed the
connection. Original backups and the earlier diagnostic clone remain intact.

## Production boundary

No production quarantine, restoration or re-embedding was performed. Applying
this disposition to production requires separate explicit approval, a new
current-state dependency check, a durable private recovery backup and a
reviewed transactional application procedure. The clone-only helper is not a
production repair command. Keyword BM25 scores may legitimately change when a
document leaves the corpus; the invariant is unchanged unrelated data/postings
and correct candidate membership, not identical ranking scores.
