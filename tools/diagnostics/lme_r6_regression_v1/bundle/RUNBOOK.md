# R6 failed-question regression v1

Run the unchanged frozen stock CLI once against the original full S dataset,
with sample one and seed zero: source index 329, `gpt4_483dd43c`, 52 sessions,
532 messages. This is a targeted regression following the retained sample-eight
failure, not a representative sample or full-500 score. Historical artifacts,
helpers, checkpoints and stores are never modified or resumed.

Use the independently pinned frozen R6 manifest and exactly its 231 source
files, DeepSeek `deepseek-flash`, `https://api.deepseek.com`, thinking disabled,
the current v19 extraction canary and v11 source split policy. All other stock
flags match sample8-v1. Stock controls bound indexing to 100 cycles/3600 seconds.
Outer supervision is 5400 seconds with 10-second cleanup. There is no global
paid-call cap; stock per-operation bounds remain intact. One canary suite may
use several completions/retries under its existing contract. No wrapper retries,
question rerolls, SDK patches, skipped gates or resume are allowed.

Preflight is network-disabled and credential-free; it exercises actual SDK
producer identity and the real stock CLI up to its provider boundary. The live
container has only the scoped key mount, fresh results, and read-only source,
data and package. It cannot access production memory or the Docker socket.
The parent must complete and review all R6 gates before providing a pinned
admission record to the one-shot live start. Source packaging is not readiness.

Postvalidation is a separate credential-free, network-disabled container with
all binds read-only. It uses the corrected R6 strict artifact validator and binds
the genuine pointer, archive, physical checkpoint, every attempt history,
process cleanup, roles and separately metered canary. Both successful and failed
question rows are validated. Wrong answers remain performance outcomes; failed
indexing remains failed even if its archive is structurally valid. Summary
degradation remains an explicit warning separate from item-indexing health.
The R6 summary-recovery prompt is used by explicit recovery, not stock digest;
this run cannot claim that recovery was exercised or all summaries are healthy.

Only closed metadata leaves the remote host; benchmark text, raw output and
kept stores remain private there. Failure is retained without another paid run.
