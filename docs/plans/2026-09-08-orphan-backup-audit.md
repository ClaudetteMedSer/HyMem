# Orphaned chunk: backup selection audit

## Decision

**None of the located and inspected backups provides an authoritative restoration
source for the missing session. Do not roll production back to one of them to
repair this orphan.** This establishes absence in the inspected records, not
that the session never existed or that no off-site backup could contain it.

The target is the previously reported `chunks` rowid 1058 in hermes-1. Its
stored creation timestamp is `2026-08-09T22:16:24` (no timezone in that field).
The corruption is already present in the August 25 database backup, well before
the September upgrades. No message text, private session key or credential was
printed during comparison.

## Evidence

| Backup group | Result for the target |
| --- | --- |
| Original August 9 ZIP, HyMem database **including committed WAL transactions** | Neither the chunk nor its missing parent exists; zero chunk FK faults. |
| Four explicit HyMem backups: August 25, August 30, September 5, September 7 | Exact same chunk text, parent key and source span; missing parent; no raw source messages/coverage/evidence; one chunk FK fault. |
| Eight daily HyMem backups, September 1–8 | Same orphan and missing source relationships in all eight. |
| Historical snapshot copied into daily backups | All eight copies are byte-identical to each other, but differ from the standalone historical file. The distinct copy was checked separately: same orphan, no parent/source. |
| Six original Hermes conversation databases | Zero exact matching sessions, messages or child-session references in every database. |

Fourteen HyMem snapshot files were compared after accounting for the repeated
historical-copy variant. Thirteen contain the same orphan; the initial snapshot
does not contain the chunk or parent. Two distinct small `hymem.db` companion
variants are not HyMem stores. No database was initialized or migrated.

### HyMem candidates inspected

On Afrodite, the four explicit files under
`/opt/stacks/hermes/instance1/home/.hermes/` are:

- `hymem.pre-invalid-at-repair.sqlite` (schema 31)
- `hymem.sqlite.bak-v33-20260830` (schema 33)
- `hymem.sqlite.pre-v46-20260905.bak` (schema 35)
- `hymem.sqlite.bak-20260907-pre-v59` (schema 46)

Daily files are
`/opt/stacks/hermes/backups/hermes-state-202609DD-020001/db/hymem.sqlite`,
for `DD=01` through `08`, covering schemas 35, 46 and 59. Each daily directory
also contains the hash-deduplicated historical snapshot copy.

The initial archive is
`/opt/stacks/hermes/hermes-backup-20260809-132118.zip`. Only explicitly selected
SQLite members were copied to private same-host storage. Its HyMem database was
read together with `hymem.sqlite-wal` and `hymem.sqlite-shm`, then backed up into
a stable comparison snapshot. The archive's SHA-256 before/after extraction was
identical: `8b5f33523b81ff53b0c1b7ccb9b704e1ac15abf1180804574115eb4b934f047e`.

The August `hermes-files-*.tar.zst` archives were scanned for database member
names, not configuration contents. August 17–30 contained no HyMem/Hermes
database members. August 31 contained the August 30 HyMem backup, proven
byte-identical to the already inspected standalone file. The Agent37 payload
had no matching database members. The clean-instance archive had only a small
original Hermes conversation database, checked below.

### Original Hermes source databases

The implementation agent checked five of these and root independently repeated
all five checks. Root additionally inspected the clean-instance source:

- Original ZIP `state.db`.
- Original ZIP `state-snapshots/20260517-184857-pre-update/state.db`.
- Original ZIP `state-snapshots/20260518-114736-pre-update/state.db`.
- `/opt/stacks/hermes/backups/state.db.pre-sqlite3.53.4` (August 12 metadata).
- `/opt/stacks/hermes/backups/hermes-state-20260908-020001/db/state.db`.
- `/opt/stacks/hermes/instance1-clean.tar.gz`, member `home/.hermes/state.db`.

All returned zero exact matching sessions, messages and child-session
references. The current HyMem/Honcho code preserves incoming session IDs;
no undocumented historical client-side remapping was guessed. This exact-key
limitation matters: these checks cannot rule out source content under an
unverified different identity.

## Safety and next step

Production, archived originals, services and credential files were unchanged.
Static standalone files were opened read-only/immutable after checking for
journals; journal-bearing archive data was interpreted on private copies.
Copies remained on Afrodite in `0700` directories with `0600` files. All
archive-comparison copies were removed after verification and can be recreated
from the unchanged originals. No private database was transferred off-host.

The earlier diagnostic clone remains in hermes-1 at
`/tmp/hymem-orphan-audit-cyztl782/snapshot.sqlite` for a possible repair rehearsal.
The orphan and its vector remain in production, unchanged.

Without another authoritative source, the next option is an explicit data
disposition: preserve a recoverable export, verify all dependencies on a clone,
then quarantine/remove the unproven orphan and its vector from the active store.
That is **not** a source restoration and was not performed as part of this
backup-selection request. A whole-database rollback is not a suitable substitute.

## Subsequently approved clone rehearsal

The user subsequently approved a clone-only quarantine rehearsal, now verified
in [the rehearsal report](2026-09-08-orphan-quarantine-rehearsal.md). Its broader
dependency audit additionally found four `entity_mentions` links; the original
limited provenance checks above did not cover those links. All six business
rows and the logical vector-index row were preserved before deletion on the
working clone. Independent integrity, unchanged-data, search-index and recovery
checks passed. **Production remains unchanged; no production quarantine has
been authorized or performed in this rehearsal.**
