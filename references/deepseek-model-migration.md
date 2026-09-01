# DeepSeek model deprecation & migration (2026-07-24)

**Dead names (400s since 2026-07-24 15:59 UTC):** `deepseek-chat`, `deepseek-reasoner`. — CORRECTED 2026-08-30: live probe shows `deepseek-chat` works AGAIN (HTTP 200, `content` populated; server maps it to v4-flash non-thinking). The current hook pin `HYMEM_LLM_MODEL=deepseek-chat` is the WORKING choice; bare `deepseek-v4-flash` without thinking-disabled extra_body returns content="" + reasoning_content (the v4-pro-style trap). Before any model/env change, probe live: `curl https://api.deepseek.com/v1/chat/completions` with max_tokens=5 and read which field carries the text.

**Drop-in replacement:** `deepseek-v4-flash` + `thinking: {"type": "disabled"}` — byte-path-equivalent of old `deepseek-chat` (non-thinking mode).

**NEVER `deepseek-v4-pro` for answer/judge/distill reads.** It is a reasoning model: output lands in `reasoning_content` instead of `content`, so every client reading `choices[0].message.content` gets empty strings — answer/judge/distill calls silently fail. Reasoning tokens also burn the `max_tokens` budget: small calls (judge `max_tokens=10`, distill `max_tokens=256`) exhaust everything on reasoning before producing content.

## `HYMEM_LLM_THINKING` (gated client, landed upstream d6ebaa5 2026-08-05)

`hymem/contrib/openai_client.py` — resolved ONCE in `__init__`, not per call.
Vocabulary: `auto` / `disabled` / `enabled` / `off` (invalid value raises `ValueError` at construction).
`auto` = substring match on host OR model: `deepseek` in host or model → sends `{"thinking":{"type":"disabled"}}`; non-DeepSeek endpoints get NO extra_body (sending it to OpenAI/vLLM is a 400).
**After any merge, verify presence:** `grep -n "HYMEM_LLM_THINKING" hymem/contrib/openai_client.py` — the whole block was missing from HEAD until 2026-08-05; merges have silently dropped it.

## Benchmark adapters: CLI flags, not code patches

Upstream adapters (July 2026) use `--judge-extra-body` and `--answer-extra-body` CLI flags passing `{"thinking":{"type":"disabled"}}` instead of hardcoding in `LLMClient._call()`. `LLMClient` accepts `extra_body` in its constructor and merges it into every request; the flags thread through `args.judge_extra_body_obj` / `args.answer_extra_body_obj`. Always pass `--judge-extra-body '{"thinking":{"type":"disabled"}}'` when running benchmarks or `--rejudge` with `deepseek-v4-flash`.

## Fix locations (all upstream-complete as of 2026-08-07)

**VERIFIED 2026-08-31 on Beam-optimisation HEAD 50951e0 (local check, not upstream):**
`benchmarks/beam_adapter.py:44-45` is STILL `deepseek-chat` (not migrated), and beam_adapter has
NO `--answer-extra-body`/`--judge-extra-body`, NO `--rejudge`, and `LLMClient._call()` reads
`message.get("content","")` with no reasoning fallback. Explicit `deepseek-v4-flash` pin on this
adapter REQUIRES adding extra_body plumbing + a client-path canary first; bare pin hits the
content="" trap (see lme_runs.db id=53: 0.6% no-extra-body vs id=54: 69.8% same-day with
extra_body). The `deepseek-chat` alias path is the currently-working choice.
(UPDATE 2026-08-31 later, HEAD 90ced81: `--rejudge` now EXISTS on beam_adapter
(judge-only rejudge, gold-reparse guarded, canary + silent-0 abort + ABS/CR gate); NO
extra_body plumbing still, so the v4-flash-pin trap rule above is unchanged.)

**UPDATE 2026-09-01 (Phase 2 plumbing landed): the trap rule above is now ENFORCED, not
just documented.** `beam_adapter` has `--answer-extra-body` / `--judge-extra-body`;
`LLMClient` takes `extra_body` and merges it last; `_call` raises on FALSY content, not
merely null, because empty is the shape the trap actually takes. `check_model_pin()` runs
on both clients in `main()` and in `_rejudge_run`: a DeepSeek `v4-flash` model without
`thinking:disabled` exits 2, and a `thinking` key aimed at OpenAI/Gemini (a 400) exits 2.
A real-prompt canary now runs on BOTH clients at each path's own `max_tokens` ceiling
before the run spends anything — previously only the rejudge judge was canaried, and the
expensive answer path had no guard at all. Artifacts record `answer_extra_body` /
`judge_extra_body`, so a reader no longer has to infer from the code whether thinking was
disabled. **Defaults are unchanged**: `extra_body` is empty unless a flag sets it, so an
unflagged run sends the same four body keys it always sent and prior artifacts stay
comparable. `ANSWER_MODEL`/`JUDGE_MODEL` are still `deepseek-chat` — the flip is the
pre-registered decision, not a plumbing detail.

| File | What |
|---|---|
| `hymem/contrib/openai_client.py:51` | Default fallback → `deepseek-v4-flash` |
| `hymem/bootstrap.py:23` | `DEFAULT_LLM_MODEL` → `deepseek-v4-flash` |
| `benchmarks/longmemeval_adapter.py:183-184`, `:340` | `ANSWER_MODEL`/`JUDGE_MODEL`. **CORRECTED 2026-09-01: LME has NO three-way fallback.** `_call` (`:384-392`) raises only on `content is None`; the trap shape is `content == ""` with `finish_reason=length`, which that check lets through untouched. LME's protection is the `extra_body` flag plus its canary, not the client. The row previously listed a recommendation as landed code. |
| `benchmarks/beam_adapter.py:44-45` | Constants → `deepseek-v4-flash` |
| `~/.agent37/hooks/post-restart.sh` | `HYMEM_LLM_MODEL` → `deepseek-v4-flash` |
| `~/.hermes/bin/hymem-server-wrapper` | `HYMEM_LLM_MODEL` — the SINGLE source for MCP-server env; was the LAST live `deepseek-chat` reference (2026-08-07). After patching, kill the `hymem-server` CHILDREN only — watchdogs respawn them with the new env; verify `/proc/<pid>/environ`. Killing the watchdogs kills the bridge. |

`HYMEM_LLM_MODEL` env var beats code defaults — fix the hook too; restart honcho afterward.

## Non-DeepSeek reasoning models

`gpt-oss-120b` via OpenRouter may land output in `reasoning` instead of `content`. `LLMClient._call()` needs the three-way fallback: `content or reasoning or reasoning_content`. See `references/beam-optimisation-2026-07.md` §LLMClient reasoning-content fallback.
