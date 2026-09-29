# Offline feasibility review of the retained summary case

## Result

The prior and new material can be represented within 500 Unicode code points
at the **topic/main-proposition** level. A manually authored and independently
reviewed reference below is exactly 500 characters. The unchanged application
accepted it for both saved primary responses, in both supported repair formats.

This corrects the earlier suspicion that this particular case might necessarily
require a larger summary budget. It does not prove that all future cases fit,
that every stylistic directive is satisfied, or that DeepSeek can reliably
produce this degree of compression. Patch 07 remains an unsuccessful live
repair candidate and is not approved for promotion by this exercise.

## Scope and source

The user approved this offline reference review. Only the pinned 1,588-character
generation input was inspected: prior automatic summary, the 48-character
already-digested boundary context, and message 298's new range 300:1025/1025.
The boundary text completes an interrupted sentence; it was not promoted into
independent new evidence. The two rejected primary drafts concern the same
source case, not independent source examples. No benchmark answer key was used.

- Generation input SHA-256: `e80f76cdd1804811836174715e29fa290d8fbe8e0c5c9cf73e6c923fe910b227`.
- Primary request-file SHA-256: `79e2e1a1ce75b430dd7b77b5053beb7aa53513310579d509126b435376ff12eb`.
- Existing candidate live-manifest SHA-256: `7f035601c375ca8d54e52fe64236e7ffff39c4a402afd7d1273d510795737ae1`.

This is a source-relative summary of a historical discussion, **not historical
fact verification**. In particular, the source's awkward 1903 annexation claim
is preserved as supplied; this review does not endorse or silently correct it.

## Handcrafted reference

> Eulsa (1905) made Korea Japan's protectorate; Russia's failed Korea-alliance/East-Asian warm-water aims (Port Arthur/Dalny), Korean-water warship use, complex China non-alliance; Russo-Japanese tensions rose: Russia's 1903 nonrecognition of Japan's Korea annexation broke diplomacy, Japan saw Russian naval growth as a direct strategic/economic threat notably given Russia's Port Arthur lease from China, and mutual force/fleet buildups led to 1904 war with major land/sea battles in Korea/East Asia.

Python `len(text)` and `len(text.strip())` both equal 500; the text is ASCII.
Reference SHA-256:
`297d869deb8592ae1056e954c6016db16f6f53489d252b7598d37a322b7dfe25`.

## Semantic review

An independent author also constructed a 500-character topic-level summary.
A separate reviewer assessed the root reference against the source. Early
drafts were rejected as gold references when they blurred failed alliance
attempts into a failed existing alliance, dropped nonrecognition/legitimacy,
made the Port Arthur lease an independent direct threat, or left the Russian
lessee ambiguous. The final reference addresses those relations explicitly.

| Required topic or main proposition | Representation in the final reference |
| --- | --- |
| Eulsa, 1905, Korea becoming Japan's protectorate | Explicit date, parties and causal result |
| Failed Russian alliance and East-Asian warm-water ambitions | Failed aims, Korea, Port Arthur and Dalny retained |
| Purpose of Russian warships in Korean waters | Compressed to Korean-water warship use |
| Complicated Russia–China relationship without alliance | Complex China non-alliance, under the Russian topic clause |
| Rising Russo-Japanese tensions and war causes | Explicit tensions followed by diplomatic, naval and buildup causes |
| Russia's 1903 refusal to recognize Japan's Korea annexation, then diplomatic breakdown | Nonrecognition, parties, date and causal outcome retained |
| Japan's perception of a direct strategic/economic threat from Russian naval expansion | Explicit perceiver, source of threat and both interests |
| Special relevance of Russia's lease of Port Arthur from China | `notably given` plus explicit lessee, place and lessor |
| Both countries' military and naval buildups leading to war | Mutual force/fleet buildup and causal link retained |
| War in 1904, major land/naval battles in Korea and East Asia | Date, significance, domains and locations retained |

This is deliberately dense, telegraphic prose. Eulsa is shortened from the
full treaty label, and warship purpose from the prior topic is expressed as
use rather than invented detail about that purpose. The active opening and
semicolon-separated topic clauses are not proof of compliance with a maximally
literal reading of every passive-voice/one-sentence stylistic instruction.
The semantic judgment is manually reviewed, not automated entailment or
lossless preservation of every possible nuance.

## Offline application checks

The test ran on Afrodite's Python 3.11 runtime with networking disabled, no
credentials, a read-only root filesystem, and immutable source and benchmark
store mounts. The two genuine historical primary replies were supplied locally;
the repair response was explicitly **scripted from the reference**, not generated.

Each primary was exercised with three responses:

1. Exact single-summary reference: accepted unchanged at 500 characters.
2. Three-string alternatives with a 501-character first string and the
   reference second: selected the second string unchanged; accepted.
3. A 501-character single-summary negative control: rejected with
   `summary_output_cap`.

All six outcomes matched expectations. Each accepted result retained the
primary's one episode and zero procedures, with the same covered-source hash.
No result was persisted and no durable cursor moved: this was a read-only
invocation exercise, not an end-to-end dream publication test. All 225 source
file and ten retained store hashes matched before and after. Container exit 0,
PID 0, no OOM. **Zero provider completions and zero HTTP attempts.**

Machine-readable evidence:
`docs/patches/2026-09-19-lme-summary-feasibility-verification.json`.
Verifier and reference inputs remain under
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-summary-feasibility-20260919-lkCVLI`.
The previous paid campaigns and their receipts were not modified or resumed.

## What the next fix should target

The observed failure remains an automatic-generation failure under a valid
length constraint, not evidence of an off-by-one limit or broken selection.
The primary prompt asks for topic continuity; the repair additionally demands
retention of qualifiers, values, entities and outcome status. That difference
deserves explicit alignment in the next design, but its causal contribution
to the failed provider replies has not been established.

Use this reviewed reference as an offline regression oracle when designing a
generic bounded compression procedure. Do not hard-code it into runtime,
inject it into a benchmark store, or count it as a successful model response.
Retain the 500-character guard and semantic negative controls; evaluate generic
recovery on held-out cases as well as this case. Do not use these results to
justify clipping, omission of prior topics, a larger cap, or another blind reroll.

No application code, policy, production process or memory was changed in this
review. **LME readiness and automatic recovery remain unverified.**
