# HyMem R8 candidate — current source snapshot

This branch preserves the exact candidate tested on September 26, 2026. It is
a development snapshot, **not an accepted release or a claim of LME readiness**.

The 503-file application, benchmark and test inventory has SHA-256
`e38a26fec4e5b756d3bca7a36418f78648b7542b3b53805012a12c3d0e8365c7`.
The companion source manifest records every file hash. Other unchanged repository
files remain inherited from base commit `5fb5ce491b254015684be2f3115a9d6b70b0c5a3`.
The primary dirty working tree, unrelated experiments, private diagnostic
captures, databases, credential files and deployment-specific launchers are not
part of this publication.

## Verification status

- Final offline full suite: **7,864 passed, 4 failed, 4 skipped**, 7,872 collected,
  zero errors. Runtime was approximately 97 minutes, with no provider calls.
- The four failures are one bounded-summary recovery integration case and
  three terminal-empty extraction policy cases. They remain recorded failures;
  their cause has not yet been accepted or corrected in this snapshot.
- Targeted application gate: 238 passed.
- Fresh paid retained-case checks: both failing extraction chunks passed;
  normal ingestion and explicit summary recovery each passed all 14 sessions.
  Four normal and four repair semantic control walks received parent review.
  The exact previously failed repair input passed in one call.
- The failed full-suite gate prevented the fresh eight-question LME run.
  No new LME accuracy score is available for this candidate.
- No production deployment or restart was performed by this publication.

The source includes the current migration/claim-replay fixes, bounded source-only
extraction contract repair, selective summary policy and v7 explicit summary
recovery. Safety limits and failed-run evidence remain intact. Further changes
must receive their own verification; this snapshot must not be relabeled green.
