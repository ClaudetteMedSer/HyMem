# Pending stock Q1 successor — not sealed or launchable

This is a local preparation only. There is deliberately no `manifest.json`,
source pin, archive, installer or launch receipt in this directory. Wait for
the reviewed final R5 source/test gates and parent review before sealing a new
package. Do not reuse historical manifests or run directories.

The immutable recovered predecessor is `../previous-v3`. Its seven files match
the original manifest SHA-256
`e6be83fee57c238c88d989bc7d8f49fc2016e62e7642636dc4cfd240f5f1868d`.
Only those seven approved remote files were read during recovery; no results,
controls, logs, credential files or production stores were inspected.

Changes from the predecessor:

- Before any live worker, preflight now checks the actual standalone producer,
  real SDK client identity and its cleanup, then the real CLI checkpoint and
  lease cleanup. It stops before reader construction with a synthetic key and
  denies all outbound socket/DNS activity. Its private home is created
  exclusively. No provider call is made by this diagnostic.
- Live worker environment explicitly rejects `HYMEM_LLM_EXTRA_BODY` after
  resetting inherited settings. The normal canonical request body is unchanged.
- The exact source inventory derives from the pinned source map, not a stale
  hardcoded file count. Every additional file or symlink is still rejected.
- Runtime and postvalidation helpers compile verified source bytes rather than
  using a cache-writing loader. Host-side plan consumers must first verify the
  runner bytes against the independently pinned manifest, compile them, then
  call `load_verified_module(path, expected_sha256, module_name)` for the plan
  module. Never use `SourceFileLoader`/`exec_module` for the sealed package.

The live recipe, selector, dataset, 5400-second external supervision deadline,
10-second cleanup deadline and genuine scored-artifact postvalidation remain
unchanged. No global paid-call cap is invented. A correct process exit alone
does not certify indexing completion or scoring. Wrong answers remain valid
performance outcomes; they are not infrastructure failures. This single
deliberately targeted question cannot establish full-500 readiness.

`q1_stock_host.py` only returns Docker create plans; it cannot execute Docker.
The optional live plan names an existing read-only credential mount, but no
credential file has been read and no plan has been executed during preparation.
The remote source location and manifest pins must come from the final accepted
source, not the historical R3 values.
