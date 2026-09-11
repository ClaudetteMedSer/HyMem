# Context-first fragment presentation

Phase-1 renders a context-bearing fragment's metadata and exact preceding
interpretation context before its authoritative `content`. Previously global
JSON key sorting put `content` first, so a continuation appeared before the
antecedent or table header needed to interpret it.

Only top-level presentation order changes. Decoding the request yields the same
keys and values; source IDs, text, Unicode, offsets, context applicability bounds
and nested context encoding are unchanged. Full source records and fragments
without context retain their previous bytes. No padding is replaced, no reading
view duplicates source evidence, and no source/coverage record is rewritten.

The prompt wording, temperature, model, validators, output limits, retry policy
and canonical canary fixture/expectations are unchanged. A prompt-only candidate
was tested, failed, and withdrawn. The retained fix is in `_fragment_record`.

## Verification and interpretation

In five predeclared, interleaved pairs on Hermes1's configured DeepSeek client,
the original presentation passed 0/5 canonical canaries; context-first passed
5/5. Every attempt used eight successful API calls. A separate development
screen also passed. Both supported claims, their exact citations/types, required
context paths and the list/fence negative controls remained mandatory.

Ten additional synthetic primary-call checks scored 9/10 versus 7/10 under the
predeclared matcher. The baseline's negation failure was only underscore/name
formatting, not wrong polarity; its other additional failure used a referent
beyond the allowed context range. The candidate rejected that out-of-range
claim. Both versions still returned empty for the same separate deployment-
reference example. Those diagnostics are not full extraction-pipeline runs;
the normal empty-verification pass was not run for them. No new failing case
was observed, but this small sample does not establish perfect model recall or
eliminate the need for the existing in-run extraction health gate.

Root verification: 574 focused extraction/identity/budget/benchmark tests passed,
including 15 new presentation tests (13 failed against the old implementation).
The installed Hermes1 checkout separately passed 376 relevant tests, and its
eight-request fingerprint exactly matches the live-tested private candidate.
This is not a full final-tree suite or a new LME/BEAM/LoCoMo performance score.

## Contract and rollout

The human prompt version remains `v20` because prompt wording is unchanged.
The mechanically derived extraction-contract/cache identity DOES change: its
transitive executable component includes this serializer. New benchmark runs
must record that identity; old extraction caches are not comparable/reusable
under the changed contract. On a persistent production engine, activation makes
eligible old chunks require bounded re-extraction. Historical source losses
remain losses; presentation cannot reconstruct missing evidence.

The scoped Hermes1 rollout updates the benchmark checkout without restarting
memory services or starting a production dream. New benchmark processes load
the fix; already-running processes retain their loaded code until a separately
planned restart. Preserve the existing local client extensions and canary
override support; this fix neither uses nor changes an override.

Development history, accounting and deployment receipts are in
`docs/plans/2026-09-10-canonical-canary-fix.md`.
