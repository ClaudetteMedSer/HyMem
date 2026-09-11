# Sep10 ownership-residue recovery (clone only)

The normal ownership guard correctly prohibits binding a legacy session after
it has source history. The incident bypassed that guard. Rolling the binding
back restored the old proofs, but necessarily invalidated the two attributed
uploads inserted during the temporary bind. There is no honest binding that
validates both histories. Rebinding or rewriting source attribution is not a
repair.

`incident_residue_recovery.py` handles only an exact, pinned two-message,
uncited, unconsumed tail on schema61. It is an offline recovery artifact, not a
migration or a production repair command. Its input is always read-only. It
produces a new private directory containing:

- `original.sqlite`: complete consistent original backup, including original
  corruption; this is the authoritative full-restore artifact.
- `rows.json`: private SQLite-type-preserving rows (including exact original
  message bytes, attribution, proofs, chunks, vectors and stale memberships).
  BLOBs are base64, integers and floating-point values preserve native values.
- `repaired.sqlite`: quarantined runtime clone, only after all checks succeed.
- `ready.json`: final success receipt with original/repaired logical hashes,
  schema/target pins and the row receipt hash. Without this file, do not adopt
  any clone, even if a partial artifact appears usable.

The unique upload remains exact in both original backup and row archive. It is
not fabricated into a new conversation, silently dropped, or treated as valid
canonical evidence. Quarantine does make these two uploads unavailable to
runtime retrieval; future recovery/import needs separate provenance review.

The artifact also requires the unchanged sibling
`rehearse_orphan_quarantine.py` for bounded native encoding, reviewed vector
storage classification, and inspection of actual FTS postings. It does not
invoke that helper's historical row1058 repair.

## Review and use

Run in the matching HyMem environment on a consistent private store copy.
The read-only `discover_pins(path, (first_id, second_id))` API returns only hashes
and the explicitly supplied IDs. Discovery is not approval: independently
check the two message content hashes, session hash and incident membership
census. The full store and schema pins prevent silently using stale evidence.
The CLI exposes the same discovery as `--discover FIRST_ID SECOND_ID`.

Call `recover(source_path, new_private_bundle_path, pins)` after review, or use
the CLI with `--pins PRIVATE_PINS_JSON --bundle NEW_PRIVATE_DIRECTORY`.
There is deliberately no `--apply` or in-place mode. Do not print `rows.json` or
private original database contents into transcripts. Public output is fixed
status codes, counts, and hashes only. Retain all artifacts on failure.

Only inside the private clone, in one SQLite transaction, the tool temporarily
suspends the ordered-stream delete guard, removes the two exact raw/proof/chunk
sets and their vectors, removes the exact pinned stale memberships after all
remaining sources are proved native, and rewinds only the producer coverage
frontier to the previous retained proof. All consumer frontiers remain byte-for-
byte unchanged. The original guard SQL is restored inside that transaction.
It refuses any other retained proof error, source citation, overlapping derived
range, JSON citation, consumer watermark reaching the incident, new tail,
missing raw source, unexpected membership, or existing failure ledger.

Every logical table, actual FTS posting and empty-document membership is compared
against the exact expected delta; vector allocation pages may legitimately
change, but every unrelated logical vector must remain exact. A full canonical
proof audit, foreign-key/integrity checks and ordinary application reopen must
pass without additional changes before the ready receipt is written. Failure
before commit rolls back rows and guards. Failure after clone commit leaves
the source/backup intact and never advertises a ready clone. A repeated call on
the same ready bundle verifies all hashes and does not perform another deletion.
That verified receipt also recognizes an input already equal to the exact
repaired snapshot; absent targets without the retained ready bundle are never
guessed to be successful recovery.

Production adoption remains a separate operation: pause all writers, take a
fresh consistent backup, rerun on that exact frozen state, verify receipts,
then separately review any database replacement and service restart. Never
swap a rehearsal clone over a live database that has received newer messages.
The durable gateway-key replacement must be enabled first so capture cannot
re-enter the retired legacy session. Do not delete the incident backups or
quarantine archive as routine temporary cleanup.

Limits: reviewed schema61, exactly two target messages, one NULL-owned session,
one or two pinned stale memberships, source size512MiB, native cell16MiB,
two million rows per logical table/posting stream, bounded cooperative deadline.
Unknown or larger states fail closed and need separate review.
