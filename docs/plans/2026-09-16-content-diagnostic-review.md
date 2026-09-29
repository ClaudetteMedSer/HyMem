# Read-only review of the closed v14 diagnostic

## Scope

The user explicitly approved inspection of the two rejected summaries and their
verifier verdicts against the already-approved benchmark source, prior derived
summary and boundary context. Only those fields were retrieved from authenticated
first-case receipts. No unrelated records, production memory or credentials were
retrieved. No provider calls, rerolls, deployments, restarts or runtime edits
were made. Raw benchmark text is not copied into this document or new tests.

## Findings

1. **A supported temporal relation is lost twice.** The visible user message
   explicitly states one completed stop followed by another. Both the compacted
   365-character summary and the recomposed 485-character summary turn that into
   an unordered conjunction. The current fidelity contract explicitly requires
   material event order. This is a concrete candidate defect, not proof of the
   verifier's exact reason: the verdict schema provides no explanation.
2. **The repaired summary receives a false format rejection under the written
   contract.** Human review finds one complete sentence with semicolon-joined
   clauses, no Markdown and no enclosing output quotation marks. The verifier
   explicitly permits this construction but labels it unsupported. Content and
   format classification therefore cannot be assumed accurate merely because
   the verdict JSON is well formed. The unrepaired summary is also a single
   sentence, though its final continuity phrase has less clear attachment.
3. **No source or request wiring defect was found.** Both verification requests
   contain identical source catalogs, prior continuity and item candidates; the
   evidence matches the previously approved v13 source window. The repair gets
   the exact original generation user request, temperature, token limit and
   response format. It explicitly instructs preservation of related event order.
   Raw and effective summary values match, and the repaired candidate differs
   from the first. This is not stale-candidate validation or a lost-source bug.
4. **The repair is not reliably correcting the rejected relation.** It adds
   incidental advice while repeating the same chronology omission. The current
   implementation gives it the original sources and general fidelity rules,
   but no specific diagnosis of the rejected relation. Its execution is bounded
   and correct; that does not establish semantic recovery effectiveness.

The 500-character limit alone does not make this case impossible: a human-written
321-character illustration retained the explicit order, missed location, riding
wish, supplied recommendations/directions and prior topics. That illustration
was neither sent to a model nor used as a golden benchmark answer; it is not
evidence that automated generation or verification will accept it.

## Implications

The factual veto should stay intact. Format-only adjudication correctly does not
override a simultaneous content rejection. The defect is not fixed by bypassing
that guard, raising retry counts or treating all prior unit tests as live model
accuracy evidence.

A proposed next development step is evidence-linked violation diagnostics and
targeted summary repair. A diagnostic should identify a violated relation and
its exact source references, distinguish content from grammar, and remain an
untrusted hint rather than new factual authority. Repair must use the original
sources, preserve item/citation authority, remain bounded by the same deadline
and call budget, and undergo complete fresh verification. Whether this design
improves reliability needs a preregistered live comparison after offline safety
tests; this inspection does not establish that outcome or authorize another run.

No runtime fix is claimed from this read-only review. The v14 campaign remains
closed after five actual calls, and canonical LME readiness remains unconfirmed.
Its execution and audit hashes are in `2026-09-15-content-live-diagnostic.md`.
