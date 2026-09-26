# HyMem R8 candidate — current source snapshot

This branch preserves the corrected candidate under test on September 26, 2026. It is
a development snapshot, **not an accepted release or a claim of LME readiness**.

The 505-file application, benchmark and test inventory has SHA-256
`52c8a4938e4420a936f9c507710414ea846c9232ddf6a461c66366958e8ec7cb`.
The companion source manifest records every file hash. Other unchanged repository
files remain inherited from base commit `5fb5ce491b254015684be2f3115a9d6b70b0c5a3`.
The primary dirty working tree, unrelated experiments, private diagnostic
captures, databases, credential files and deployment-specific launchers are not
part of this publication.

## Verification status

- **Full offline regression suite in progress:** 7,890 tests collected. No final
  full-suite pass is claimed for this snapshot.
- The previous snapshot (`5f936f21aaf59294507954ecd7ae28adecedda02`, inventory
  `e38a26fec4e5b756d3bca7a36418f78648b7542b3b53805012a12c3d0e8365c7`)
  finished with 7,864 passed, four failed, four skipped and zero errors.
  Its failed gate prevented paid LME dispatch and remains failed evidence.
- All four failures were independently reproduced as stale test fixtures for
  the existing bounded-repair contract. Separate Sol agents corrected the
  summary and terminal-empty fixtures without weakening production validators;
  parent verification passed 84 and 321 affected tests respectively.
- A separate fix prevents raw provider exceptions and arbitrary labels from
  leaking through retry/watcher logs. Parent verification passed 77 affected
  tests plus two independently authored coupled watcher/retry controls. Retry
  behavior, budgets, deadlines and final exception propagation remain intact.
- Fresh paid checks on the exact corrected source: both retained extraction
  chunks passed (five calls); normal ingestion and explicit summary recovery
  each passed all 14 sessions, along with four normal and four repair control
  walks and the exact previously failed repair input (72 calls). Total:
  77 completions / 77 HTTP attempts / 219,999 tokens. Provider dollar cost was
  not reported. Parent reviewed all invented semantic controls and independently
  replayed exact extraction requests/results offline with zero additional calls.
- The fixed eight-question development regression remains gated on the new
  full-suite result and genuine startup preflight. No new LME accuracy score
  is available; this is not a canonical full-500 benchmark.
- No production deployment or restart was performed by this publication.

The source includes the current migration/claim-replay fixes, bounded source-only
extraction contract repair, selective summary policy and v7 explicit summary
recovery. Prompt v20, split contract v11 and recovery v7 are unchanged in this
follow-up. Runtime-bound producer identities are distinct from the source
inventory hash and must not be transplanted between Python runtimes. Safety
limits and failed-run evidence remain intact. Further changes must receive
their own verification; this snapshot must not be relabeled green.
