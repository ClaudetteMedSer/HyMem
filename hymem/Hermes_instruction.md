# Standing Digest in Hermes Auto-Injected Context — Operator Runbook

Runbook for a Hermes instance that wants HyMem's cross-session RAPTOR digest —
the whole-store "what do you know about this user?" narrative — to appear in
the agent's per-turn auto-injected context block.

**Update 2026-07-06 — the fix is now entirely HyMem-side.** The peer-context
route returns an SDK-parseable response carrying the representation under
*both* field names, working around the honcho-ai SDK bugs described in §Root
cause. A **stock, unpatched** Hermes harness receives the digest from a HyMem
server at or after this fix. The harness patch + post-restart hook below are
only needed while running an **older** HyMem server.

**Checklist (current HyMem):**

1. Upgrade HyMem to a build at or after 2026-07-06 (dual-field peer-context response; the session-context wiring from commit `0b5eb55` is older still) (§Prerequisites).
2. `HYMEM_AGGREGATION_NODES_ENABLED=true` in the HyMem server environment, then let one dream run (§Prerequisites).
3. Verify — server first with `curl`, then the harness **in a fresh session**; an existing session's cached base context serves a stale block (§Verification).

**Checklist (HyMem older than 2026-07-06):** as above, plus apply the two-part
patch to the harness's `plugins/memory/honcho/session.py` (§Harness patch) and
install the post-restart auto-patch hook (§Post-restart survival).

---

## Problem

HyMem builds a cross-session RAPTOR digest — a whole-store "what do you know
about this user?" narrative. It was available on the Honcho peer-card/context
routes but never reached the Hermes agent's per-turn auto-injected context
block. The block only showed session summary, knowledge graph facts, and
conversation history excerpts.

## Root cause

Two gaps, one on each side of the Honcho ↔ Hermes boundary:

1. **Honcho session context endpoint** (`hymem/honcho/app.py`) — `get_context()`
   returned plain `USER.md` as `peer_representation`. The digest existed but
   wasn't wired in.

2. **Hermes harness** (`plugins/memory/honcho/session.py`) —
   `get_prefetch_context()` called the session context to get the summary but
   **dropped** `peer_representation` (it only read `ctx.summary`). The
   representation was then re-fetched separately from the **peer** context
   endpoint — a path that never delivers the digest because of an upstream
   honcho-ai SDK bug (below).

**The peer-path smoking gun (upstream SDK bug):** the peer context endpoint
returns `peer_representation`, but the SDK's `PeerContextResponse` model
declares the field as `representation` with no alias:

```
API returns:  {"peer_representation": "digest..."}
SDK model:    PeerContextResponse(representation: str | None = None)
                        ↑                         ↑
                   field name              JSON field name
                   MISMATCH
```

Pydantic doesn't map `peer_representation` → `representation`, so the value is
silently dropped. The harness's `_fetch_peer_context()` fallback chain —
`getattr(ctx, "representation") or getattr(ctx, "peer_representation")` —
returns `None` on both names (the first is never populated from the mismatched
JSON key; the second isn't an attribute of the model). Result: the peer path
yields an empty representation every time, and never surfaced the digest even
before the session-context fix. `SessionContext.peer_representation` *is*
mapped correctly, which is why Fix 1 below works.

It's worse than a silent drop: `PeerContextResponse` also *requires*
`peer_id` and `target_id`, which the endpoint didn't return — so
`peer.context()` didn't even parse; it raised a `ValidationError` inside the
SDK, swallowed by the harness into an empty result. Both problems are fixed
server-side as of 2026-07-06: the route now returns `peer_id`/`target_id` and
sends the representation under both names, pinned by a real-SDK contract test
(`test_honcho_contract.py::test_peer_context_representation_reaches_sdk`).

---

## Prerequisites

### HyMem version

Two HyMem-side fixes matter:

- **Commit `0b5eb55`** — the session-scoped `get_context()` returns
  `peer_representation` via the same `_peer_representation()` helper the peer
  routes use — standing digest (when built) above `USER.md`. Zero
  API-contract change; when no digest exists the helper degrades to plain
  `USER.md`, i.e. the old behavior.
- **2026-07-06** — the peer-context route returns an SDK-parseable response:
  the required `peer_id`/`target_id` fields plus the representation under
  both field names. With this, the stock harness's existing peer-path fetch
  delivers the digest and **no harness patch is needed**.

### Enable aggregation

The digest only exists if the RAPTOR aggregation layer is on. In the HyMem
server's environment:

```bash
export HYMEM_AGGREGATION_NODES_ENABLED=true
# HYMEM_AGGREGATION_DIGEST_ENABLED defaults on — only set it to turn the digest OFF
```

Without the master switch, `HyMem.digest()` is `None` and
`peer_representation` is byte-for-byte plain `USER.md` — the patch below is
then harmless but does nothing visible.

### Let one dream run

The digest is rebuilt at dream time, never per query. After enabling the flag,
at least one dreaming cycle must complete before a digest exists. Two
operational notes:

- **The first dream after enabling (or after upgrading across the 2026-07
  window-alignment fix) rebuilds the aggregation tree — a one-time LLM cost.**
  Subsequent dreams on a quiescent store reuse cached fusions (keyed by
  member-set hash), so re-dreaming a stable store costs zero digest LLM calls.
- `GET /dream-status` shows the extraction backlog; the `hymem_digest` MCP
  tool (or `HyMem.digest()`) returns the digest with a coverage +
  generated-at footer, or an explanatory message if none is built yet.

---

## Harness patch (only for HyMem servers older than 2026-07-06)

With a current HyMem server this section is unnecessary — skip to
§Verification. On an older server, the two fixes below make the harness read
the representation from the session-context response, which the SDK maps
correctly.

**File:** `plugins/memory/honcho/session.py` (in the installed Hermes harness,
e.g. `/usr/local/lib/hermes/hermes-agent/`)

Two bugs in `get_prefetch_context()`:

**Fix 1 — Read `peer_representation` from the session context call.** The code
already calls `honcho_session.context(summary=True)` to get the summary. That
same call returns `peer_representation` with the digest — it was simply never
read. Add it:

```python
# In get_prefetch_context(), inside the try block that fetches the summary:
ctx = honcho_session.context(summary=True)
if ctx.summary and getattr(ctx.summary, "content", None):
    result["summary"] = ctx.summary.content
# ADD THESE LINES:
if ctx.peer_representation:
    result["representation"] = ctx.peer_representation
```

The Honcho SDK always populates `ctx.peer_representation` from the API
response regardless of whether `peer_target`/`peer_perspective` are passed —
no need to change the call signature. (`SessionContext` maps the field
correctly, unlike `PeerContextResponse` — see the smoking gun in §Root cause.)

**Fix 2 — Prevent overwrite.** The code then fetches user context from the
peer endpoint, which unconditionally overwrites `result["representation"]`:

```python
# Before:
result["representation"] = user_ctx["representation"]

# After:
result.setdefault("representation", user_ctx["representation"])
```

`setdefault` keeps the session-context digest if present; falls back to the
peer-context representation otherwise.

---

## Post-restart survival (only with the harness patch)

The Hermes harness lives in an installed package path
(`/usr/local/lib/hermes/hermes-agent/`). Container restarts restore it from
the image, wiping the patch. Add an auto-patch to the post-restart hook:

**File:** `~/.agent37/hooks/post-restart.sh`

```bash
# ── HyMem digest in Honcho auto-injection ────────────────────────────
# Two-part patch for session.py. Remove when upstreamed.
SESSION_PY="/usr/local/lib/hermes/hermes-agent/plugins/memory/honcho/session.py"
if grep -F 'ctx.peer_representation' "$SESSION_PY" 2>/dev/null | grep -q 'result.setdefault'; then
    echo "[post-restart] session.py digest patch already applied"
else
    python3 << 'PYEOF'
import re
path = "/usr/local/lib/hermes/hermes-agent/plugins/memory/honcho/session.py"
src = open(path).read()

# Fix 1: add peer_representation from session context
old1 = '''                ctx = honcho_session.context(summary=True)
                if ctx.summary and getattr(ctx.summary, "content", None):
                    result["summary"] = ctx.summary.content'''
new1 = '''                ctx = honcho_session.context(summary=True)
                if ctx.summary and getattr(ctx.summary, "content", None):
                    result["summary"] = ctx.summary.content
                if ctx.peer_representation:
                    result["representation"] = ctx.peer_representation'''
if old1 in src:
    src = src.replace(old1, new1)
    print("[post-restart] session.py fix 1 applied (peer_representation)")

# Fix 2: setdefault so session context value isn't overwritten
old2 = '            result["representation"] = user_ctx["representation"]\n            result["card"]'
new2 = '            result.setdefault("representation", user_ctx["representation"])\n            result["card"]'
if old2 in src:
    src = src.replace(old2, new2)
    print("[post-restart] session.py fix 2 applied (setdefault)")

open(path, 'w').write(src)
PYEOF
fi
```

After adding the hook, restart the gateway. The post-restart script runs
before the gateway starts, so the patch is applied before any Honcho calls.

---

## Verification

Verify in two layers, server first, so a failure is attributable.

### 1. Server side — is the digest in `peer_representation`?

With the Honcho server running and a digest built:

```bash
curl -s http://127.0.0.1:8765/v3/workspaces/<wid>/sessions/<sid>/context \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["peer_representation"][:600])'
```

Expected: the digest block first (it carries a coverage + generated-at
footer), then `USER.md`. The peer card route (`GET .../peers/<pid>/card`)
should show the same.

On a current HyMem server, also confirm the peer-context route carries the
SDK-visible field name — this is exactly what the unpatched harness reads:

```bash
curl -s http://127.0.0.1:8765/v3/workspaces/<wid>/peers/<pid>/context \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["representation"][:200])'
```

If you get plain `USER.md` instead, the harness patch is irrelevant — the
digest doesn't exist yet. Check, in order: `HYMEM_AGGREGATION_NODES_ENABLED`
is set in the *server's* environment (not just your shell), a dream has
completed since enabling it (`GET /dream-status`), and `hymem_digest` /
`HyMem.digest()` returns content.

### 2. Harness side — does the digest reach the auto-injected block?

**Start a fresh session and inspect its auto-injected context block for the
digest footer.**

Do **not** judge from an existing session: the harness caches the assembled
base context (`_base_context_cache`), so a session opened before the patch
keeps serving its old block and makes a working patch look inert. This exact
artifact once produced a false "the harness never reads it" verdict — the
wiring was fine; the cache was stale.

### 3. After a restart

Restart the container once and confirm the hook reports either
`fix 1/2 applied` or `patch already applied`, then repeat check 2 in a fresh
session.

---

## Removal

The harness patch and hook are a stopgap for older HyMem servers only. Once
the HyMem server is upgraded past 2026-07-06 (the dual-field peer-context
response), delete the hook block from `post-restart.sh` — the stock harness
path works unmodified. The HyMem side needs no rollback: it is the intended
contract, pinned by
`test_honcho_contract.py::test_peer_context_representation_reaches_sdk`.
