# Persistent retirement of an incompatible Hermes capture key

This narrow Hermes integration patch adds `sessionKeyReplacements` to the
existing `honcho.json` configuration. It does **not** change HyMem's database,
ownership guards, immutable coverage proofs, or legacy source sessions.

## Why this is necessary

The gateway's logical conversation key is stable across restarts. A nonempty
legacy HyMem session with NULL ownership cannot safely be rebound: its immutable
coverage proofs may include that original ownership. The gateway otherwise keeps
resolving the incompatible session and capture stays blocked. Hermes's existing
in-memory `new_session()` operation is not durable retirement.

An operator can now map one exact, already-sanitized physical Honcho session ID
to a fresh physical ID. The manager retains the logical key and existing user/AI
peer resolution, and stores the replacement in `HonchoSession.honcho_session_id`,
which existing write/context/tool caches consume.

## Configuration and safeguards

Illustrative IDs only; this is not a production configuration:

```json
{
  "hosts": {
    "hermes": {
      "sessionKeyReplacements": {
        "legacy-storage-id": "fresh-storage-id-operator-generated"
      }
    }
  }
}
```

Before enabling, the operator must verify the exact source is the intended
incompatible legacy session and that the target is absent in the same workspace.
Do not redirect onto an existing owned session or reuse a target for another
logical conversation. This patch deliberately does not infer authorization from
a 409, probe/modify source ownership, retry onto arbitrary keys, or migrate old
messages. The map alone cannot prove a target is fresh; deployment preflight is
required. Source/target IDs are not sanitized, trimmed, or coerced by the parser.

Targets must use only ASCII letters, digits, `_`, or `-`, be 1–100 characters,
and be unique. Sources must be nonempty IDs in the same safe character set.
Self-maps, chains, cycles, malformed maps and duplicate JSON keys are rejected.
The active host's first non-null map replaces the root map in its entirety;
`null` inherits the root and `{}` explicitly removes the root mapping for that
host. Other host-only mappings do not affect the active host.

Keep the config and mappings durable across restarts and image upgrades.
Deleting the file or map, resetting it to `{}`, changing the active config/host,
or restoring a pre-retirement config can reactivate an old key. These are **not**
safe operational resets. Preserve a receipt and backup of the approved mapping
and verify it again after every deployment. A running manager snapshots its
map; restart/recreate the manager after an approved change rather than changing
routing underneath cached work. The preexisting `new_session()` behavior is
unchanged; use this persistent mapping for incident retirement.

Existing unreadable, malformed or duplicate-key Honcho configs now fail closed
instead of silently falling back to environment variables. Error messages do not
contain config keys/values or the original exception. An absent config retains
the existing environment fallback for installations not using this feature.

## Applying and retaining the patch

The adjacent JSON manifest pins the patch, focused tests, and complete before /
after SHA-256 for exactly three Hermes plugin files. The patch is a minimal diff,
not a vendored tree. These pins refer to the inspected September 10 source.

1. Verify the manifest's patch/test digests and each installed source digest.
   Require all three files to match the pinned `before` state, or all three to
   match the pinned `after` state for an already-installed no-op. Reject an
   unknown or mixed state; do not overwrite local customizations.
2. Apply with `git apply --check` then `git apply` only on a private source clone.
   Verify all three resulting digests equal `after` and all unrelated files are
   byte-identical. Never use `--reject`, `--3way`, or fuzzy application to an
   unreviewed image update.
3. Run the focused test file using the actual Hermes interpreter/checkout, plus
   the installed config/session/schema regression tests. Follow the installed
   Hermes `AGENTS.md`: use its hermetic `scripts/run_tests.sh` runner, which sets
   a temporary `HERMES_HOME` and uses per-file subprocess isolation; do not call
   pytest directly. Pass `--file-retries 0` to expose first-run failures; the
   installed runner strips the `HERMES_TEST_FILE_RETRIES` environment variable.
   Copy the focused test into the private clone's `tests/honcho_plugin/` package
   and run it there with the existing regressions. This artifact lives outside
   HyMem's default test collection because it imports actual Hermes internals.
   The separate `hermes-session-key-retirement-test-contract.patch` updates the
   preexisting test that asserted corrupt-config environment fallback; verify
   its manifest pins and apply it only to the private test clone. Do not skip
   that test or retain its now-unsafe expectation.
4. Rehearse the intended map and a subsequent config reload/new manager on a
   private store/service, proving the original session/proofs stay unchanged.
   Record source/target identifiers privately and export only sanitized receipts.
5. Install only reviewed deltas while the affected instance is stopped, using
   the deployment's ownership/mode-preserving atomic/rollback workflow. Preserve
   every unrelated plugin/Hermes/HyMem customization. Enable only the approved
   exact host map, restart the instance through its supported launcher, then
   verify capture, source immutability, health, and config-reload persistence.

Retain this patch and manifest in the deployment/image build inputs. On future
image updates, pinned `before` files can receive this patch, pinned `after` files
need no action, and any other source needs a fresh review/rebase and fresh pins.
Do not blindly copy the old whole plugin directory into a new image. Config
alone is ineffective if the integration patch is lost during an image rebuild.

### Offline pre-start installer

`install_hermes_session_retirement.py` is the maintained POSIX-only companion for
the startup/image-update workflow. Persist it beside the runtime manifest and
runtime patch in the instance's deployment scripts directory. The optional
test-contract files and focused tests are not required on a production boot and
are never installed by this helper. Its CLI takes only an explicit absolute
`--runtime-root` (the Hermes checkout) and optional `--check`.

For example, after independently validating the persisted artifacts, a pre-start
hook can invoke the helper using its absolute path and
`--runtime-root /usr/local/lib/hermes/hermes-agent`. Root's deployment workflow
must install the hook and config separately; generating this helper does neither.

The installer reads bounded regular files, rejects symbolic links anywhere in
the artifact/runtime paths and rejects hard-linked inputs. It accepts exactly
the three pinned runtime paths, checks the manifest and patch digest, and applies
only exact unified-diff offsets and context in memory: no Git commands, fuzzy
patching, network, config reads or database access. A cooperative directory lock
prevents two installers from interleaving; this is **not** a substitute for
stopping the service and all other code updaters first.

- All sources in the pinned `before` state: validate all resulting hashes, stage
  private same-filesystem originals/replacements, preserve file owner and mode,
  recheck sources, replace the three files, verify them and remove the staging.
- All sources in the pinned `after` state: return `already-installed` without
  replacing or staging files. `--check` returns `ready` for valid `before` sources
  and otherwise follows the same checks without writing anything.
- Any mixed, customized or unknown state: return nonzero and overwrite nothing.
  New image versions need a fresh reviewed patch/manifest rather than forced
  installation. Manifest tampering is outside its trust boundary: persist these
  reviewed deployment artifacts with appropriately restricted ownership/modes.

Caught replacement failures trigger verified rollback of already-replaced
files, preserving original bytes, owner and mode (not original inode/timestamps).
If an external writer changed a rollback target, it is not overwritten. Failed
rollback retains its private `.hymem-session-retirement-*` staging directory for
manual recovery and exits nonzero. Ordinary successful installs/rollbacks clean
only their own explicitly named staging files.

This is **not an atomic three-file transaction**. Process kill, host failure or
other uncatchable interruption between renames can leave mixed sources and
private recovery staging. The next invocation refuses that mixed state; it does
not guess which recovery operation is authorized or erase crash receipts. Keep
the separate deployment backup and investigate before starting Hermes.

The helper emits a concise status record and exits nonzero on failure. The
existing Hermes restart-hook parent may continue gateway startup despite a hook
failure; this helper cannot change that parent behavior. A failed hook must be
treated as a deployment failure, not as successful retirement installation.
HyMem's unchanged legacy ownership guard remains the safety backstop; capture
may remain blocked. Verify the actual installed source pins and effective map
after restart rather than relying only on the hook's attempted invocation.

Installer tests in `tests/test_install_hermes_session_retirement.py` use only
synthetic files and the standard library, not Hermes imports or live state.

## Verification scope

The 46 focused synthetic cases cover persistent resolution across three new
config/manager instances, exact sanitized-ID routing, unaffected logical and
peer identities, cache/write consumers, unmapped and other-host isolation,
None-aware host overrides, invalid maps, duplicate JSON keys, unreadable config,
safe error text and the declared host-scoped UI field. Socket guards forbid all
network attempts. A local source-module run used import stubs for unavailable
Hermes support modules; it is not a substitute for installed-Hermes verification.

Root subsequently verified the actual installed Hermes environment on a private
remote source clone using its maintained hermetic runner with `--file-retries 0`:
293 tests across 15 files passed with zero failures. A separate end-to-end
rehearsal used the actual Honcho SDK 2.2.0 over loopback against a private clone
of the live store. Two fresh config/manager loads captured two synthetic messages
with two valid coverage proofs under the replacement session, while the original
legacy rows remained unchanged. These are private-clone receipts, not a claim
that production deployment or every future image/config change is verified.

The offline installer separately passed 55 synthetic tests. A local rehearsal
against copies of the three pinned real source files produced the exact expected
hashes, and a second invocation returned `already-installed` without replacement.

No remote process, live store, session binding, or real message was changed by
creating/testing this patch. It does not remediate previously captured messages
whose ownership/proofs were already made incompatible; those require a separate
honest provenance assessment. It does not change benchmark prompts or retries.
