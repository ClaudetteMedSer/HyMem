# Session-key retirement — Hermes 0.21.5 artifact set

Rebase of the original 0.19.1 set (pinned to Hermes 0.19.1, commit cc4cab2f;
HyMem `tools/deployment/`) onto
Hermes Agent **0.21.5** (tag v2026.9.24, commit f97608f178d1ffeca59860195ab7da295f7c8e5f),
made 2026-10-06 following the procedure in
`HyMem/tools/deployment/hermes-session-key-retirement.md` ("any other source
needs a fresh review/rebase and fresh pins").

- `install_hermes_session_retirement.py` is byte-identical to the 0.19.1 set's.
  Only the manifest and patch differ. The installer's default artifact dir is
  its own directory, so each set is self-contained.
- The behaviour is unchanged: same validators, same error strings, same
  fail-closed config read (0.21.5's BOM-tolerant `utf-8-sig` decode kept), same
  single substitution point in `HonchoSessionManager.get_or_create`. 0.21.5
  still derives the physical session ID from the key only there; every other
  path consumes the stored `HonchoSession.honcho_session_id`.
- Where deployed (hermes-1 on Afrodite): this directory is
  `~/.hermes/scripts/hymem-session-retirement-20260910/tools/deployment-hermes-0.21.5/`
  beside the 0.19.1 set in `tools/deployment/`; `.agent37/hooks/post-restart.sh`
  picks the set by the runtime's exact git HEAD and refuses any other revision.

Verification (throwaway containers of the 0.21.5 image, never a live instance):
- pristine 0.21.5, `scripts/run_tests.sh tests/honcho_plugin/ --file-retries 0`:
  21 files, 454 passed, 0 failed;
- patched + focused tests + test-contract update, same runner: 22 files,
  500 passed, 0 failed (the 46 focused cases among them);
- negative control, focused tests on unpatched 0.21.5: 45 failed, 1 passed;
- installer: `--check` ready -> installed -> already-installed, results
  byte-identical to the reviewed port, owner/mode preserved, no staging left;
  this set on the 0.19.1 image and the 0.19.1 set on 0.21.5 both refuse
  (`mixed_or_unknown_source_state`); the 0.19.1 set on 0.19.1 still `ready`.

Focused-test rebase (no assertion changed): 0.21.5 no longer invents a
`user-<key>` peer without a transport identity, so the fixture passes
`runtime_user_peer_name`; `_get_or_create_honcho_session` returns a 3-tuple.

Live result, hermes-1, 2026-10-06: the hook logged `installed` with this set,
and a read-only routing check (counts and hashes only) confirmed both mappings
load, the manager snapshots them, and each retired ID resolves to its target.
