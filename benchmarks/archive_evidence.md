# Strict archive admission

Current BEAM, LongMemEval, LoCoMo, and MSC strict archives carry a
`hymem-benchmark-checkpoint-attestation-v1` receipt. The checkpoint writer emits
a minimal finalized-ledger projection: ordered IDs, committed row outcomes and
attempt counts, final counts and failure IDs, and bindings to the archived
manifest, execution segments, and reconciled rows. Readers recompute and reconcile
these fields. The receipt contains neither checkpoint paths nor duplicate source
text or questions.

This is consistency evidence, **not a signature or authentication**. Its hash
does not establish who ran the benchmark or prove the existence of an external
checkpoint file. Old strict archives carrying only an opaque checkpoint digest,
or no checkpoint receipt, no longer satisfy current strict admission. They must
not be silently upgraded by adding guessed execution evidence. Existing explicit
legacy/exploratory readers remain separate from strict admission.

Completion also requires the current, typed health receipt for the exact source
scope and manifested indexing policy. Simulation and no-dream receipts remain
explicitly skipped/non-comparable. Completed scored rows need configured live
reader/judge/pipeline roles, mode-consistent canary evidence, sufficient measured
reader/judge calls, and successful source indexing (or an explicitly exploratory
no-dream path). Rehashing mode flags alone cannot turn an emitted simulation into
scored evidence.

Failed or interrupted attempts remain in the denominator and execution history.
Missing, unattempted placeholders need no invented indexing run. Unavailable
historical usage remains unavailable; known successful calls must still cover
the completed rows. Failed indexing diagnostics cannot certify a completed row.

MSC and LoCoMo retain the store-build pointer attached by `prepare_indexing`.
Its current version, fixed filename, lifecycle and nullable/non-null digests are
checked. A freshly published pointer's indexing hash must match the canonical
indexing attestation derived from the archive. A validated reused-store pointer
names the original build, not the later validation wave; its original identity
and material digests remain opaque because the archive contains no store file.
Skipped indexing must carry an unpublished pointer with null digests. Its
optional observed status is diagnostic, not a healthy-completion certificate.
Bounded convergence, publication and store-attestation failure envelopes remain
readable failures with their source scope and available usage preserved.

LongMemEval normalized indexing summaries are now v5: their final health
certificate retains the three Phase-1 producer-authority fields rather than
discarding them. A v4 normalized summary is no longer definitive evidence.
Unavailable producer authority is a failure-only outcome; it cannot be rehashed
into a clean current-producer completion receipt.
