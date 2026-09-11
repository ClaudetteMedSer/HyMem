# Extraction texture diagnostics (offline only)

`tools/diagnostics/extraction_texture_probe.py` is an importable, synthetic-data
instrument. Running it without arguments performs a free synthetic control and
prints only fixed aggregate fields. It has no network/model client, paid runner,
file reader or raw-response printer. It never changes the canonical benchmark
fixture, prompt, temperature, model, retry policy, or extraction gate.

The September 10 ad-hoc Phase 3 C5 generator indexed a hex alphabet with
`ord(hex_character) % 16`. That maps both numeric and alphabetic hex characters
to **digits only**. Its “fresh same-class hex” inference is invalid. The corrected
`fresh_hex` consumes SHA-256's hex characters directly. It samples a hex
alphabet, **not** identical digit/letter counts or tokenization. Character/byte
length preservation does not establish token-length preservation.

Use `mutate_request` only on privately captured **synthetic** request dictionaries.
It preserves every request parameter and source field except one explicitly
selected padding span. The selected span must be lowercase hex, disjoint from a
unique protected claim. The exact JSON-line format must round-trip byte-for-byte.
The returned request stays in the caller's private memory; do not put it in logs,
canonical artifacts, a real source store, or proof/coverage records. Mutated
records are diagnostic inputs, not canonical provenance. Never replace a
canonical canary with the easiest-emitting arm.

`classify_observation` consumes counts and claim indexes **after the maintained
strict extraction validator**, not raw model text. Set each arm's expected claim
index explicitly: table control 0, prose control 1. `emitted_other` distinguishes
emission from a correct expected claim. Aggregate output has fixed keys and
counts only; invalid/incomplete output cannot be credited as a clean empty or
successful extraction.

`build_schedule` produces seeded, randomized, interleaved repetitions and explicit
minimum pre-call delays. It is a plan, not execution. Any future separately
authorized caller must enforce delay after the preceding call completes, record
actual monotonic timings, bound calls, and close its client on every exit path
without suppressing cleanup failures. The original experiment had fixed arm order
and an unused sleep setting; those receipts do not establish spaced-retry efficacy.

The observed C1 expected-claim contrast can remain a descriptive result. Neither
that contrast nor these controls prove a tokenizer mechanism, a serving-side
“basin”, or that spaced retries reliably recover production chunks. Existing raw
receipts should remain private and immutable, with the C5 correction attached to
their interpretation. A fresh canonical canary must still pass unchanged before
resuming a comparable scored baseline.
