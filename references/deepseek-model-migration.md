# DeepSeek model deprecation & migration (2026-07-24)

**Deprecated aliases:** `deepseek-chat` and `deepseek-reasoner` are not valid
current configuration choices. They returned 400s beginning 2026-07-24; an
August 30 probe briefly found `deepseek-chat` resolving again, but that only
demonstrated why a mutable alias is unsuitable for reproducible runs. Historical
artifacts below retain the name as provenance, never as a recommendation.

**Current pin:** `deepseek-v4-flash` + `thinking: {"type": "disabled"}`.
HyMem's library client applies that body automatically in its default `auto`
mode. Current repository defaults for LME, BEAM, MSC, and LoCoMo use the same model; their raw
DeepSeek reader/judge paths also default the body safely. A custom endpoint is
never sent the DeepSeek-only body implicitly.

## Cost-aware lower-tier screening

Exploratory lower-tier screening may select another explicit, versioned/pinned
model ID through the normal adapter flags, but it does not get an alias escape hatch:
every active reader, judge, and memory-pipeline selection still passes through
`require_active_model`. A cheap-model result is a promotion signal, never the
definitive benchmark result. Bind the choice in an explicit, immutable model-selection
manifest; that freezes the requested identity and effective
request, not the provider's underlying weights.

Use a paired, staged protocol:

1. Freeze the exact low-tier model id, prompts, effective request bodies, seed,
   judge, and a deterministic stratified subset spanning short/dense,
   single-/multi-session, temporal/update, and unanswerable cases.
2. Run baseline and candidate on the same subset with the same cheap model in
   the role being screened. Keep the judge fixed and immutable; do not let a
   simultaneous reader or judge change masquerade as an architecture delta.
   Use at least three paired repeats when provider sampling is live, or a
   paired bootstrap over fixed per-item outcomes, and require the uncertainty
   interval—not one point estimate—to clear the promotion threshold.
3. Treat deterministic storage, retrieval, ranking, persistence, and contract
   failures as relatively portable signals. Also inspect parse/shape failures,
   clean-empty rates, truncation, provider attempts, pending/quarantine/coverage
   health, extracted triples/episodes/facts, retrieval recall, evidence in the
   rendered context, and cost—not score alone.
4. For an extraction or other write-side change, run an early interaction check:
   baseline/candidate × screened/target memory-pipeline model. A weaker model can
   change what is stored, reverse a split/recovery benefit, or amplify strict-JSON
   and omission behavior. Read-side-only changes can often reuse one fixed,
   target-built store and vary only the paired reader/retrieval arm.
5. Promote only a pre-declared winner, then rerun the full definitive protocol
   with exact `deepseek-v4-flash`, the fixed judge, and all normal strict artifact
   checks. Do not merge cheap-model and target-model scores or report the screen
   as benchmark parity.

A regression on the paired cheap screen is strong evidence to stop. A win is
only evidence to spend the more expensive target-model run; model × architecture
rank reversals remain plausible for extraction reasoning, long-context recall,
strict structured output, answer synthesis, and judge behavior.

**Active enforcement:** live constructors reject `deepseek-chat` and
`deepseek-reasoner` (case/outer-whitespace normalized), including provider
forms such as `deepseek:deepseek-chat` and `deepseek/deepseek-chat`. The gate
runs before credential resolution, SDK construction, HTTP calls, and scored
benchmark dataset/store work. Exact versioned identities such as
`deepseek-chat-v4` are not rejected by substring guessing. Historical
artifacts and registries remain readable; the restriction applies only when a
model identity is selected for active execution.

**NEVER `deepseek-v4-pro` for answer/judge/distill reads.** It is a reasoning model: output lands in `reasoning_content` instead of `content`, so every client reading `choices[0].message.content` gets empty strings — answer/judge/distill calls silently fail. Reasoning tokens also burn the `max_tokens` budget: small calls (judge `max_tokens=10`, distill `max_tokens=256`) exhaust everything on reasoning before producing content.

## `HYMEM_LLM_THINKING` (gated client, landed upstream d6ebaa5 2026-08-05)

`hymem/contrib/openai_client.py` — resolved ONCE in `__init__`, not per call.
Vocabulary: `auto` / `disabled` / `enabled` / `off` (invalid value raises `ValueError` at construction).
`auto` = substring match on host OR model: `deepseek` in host or model → sends `{"thinking":{"type":"disabled"}}`; when neither identifies DeepSeek, it sends no extra_body (sending it to OpenAI/vLLM is a 400).
**After any merge, verify presence:** `rg -n "HYMEM_LLM_THINKING" hymem/contrib/openai_client.py` — the whole block was missing from HEAD until 2026-08-05; merges have silently dropped it.

## Benchmark adapters

The adapters retain `--judge-extra-body` and `--answer-extra-body` for explicit
provider overrides. When those options are omitted and the request targets the
DeepSeek endpoint with a v4-flash model, the effective body defaults to
`{"thinking":{"type":"disabled"}}` before any spend, including rejudge paths.
An explicit incompatible body fails closed. Other endpoints receive no implicit
DeepSeek extension; pass one explicitly only when that gateway documents it.

## Historical timeline and current state

**Historical finding, 2026-08-31 on Beam-optimisation HEAD 50951e0:**
`benchmarks/beam_adapter.py` still defaulted to `deepseek-chat` (not migrated), and beam_adapter had
NO `--answer-extra-body`/`--judge-extra-body`, NO `--rejudge`, and `LLMClient._call()` read
`message.get("content","")` with no reasoning fallback. An explicit `deepseek-v4-flash` pin on
that adapter required extra_body plumbing + a client-path canary first; the bare pin hit the
content="" trap (see lme_runs.db id=53: 0.6% no-extra-body vs id=54: 69.8% same-day with
extra_body). At that historical commit only, the alias was the working comparator path.
(UPDATE 2026-08-31 later, HEAD 90ced81: `--rejudge` then existed on beam_adapter
(judge-only rejudge, gold-reparse guarded, canary + silent-0 abort + ABS/CR gate); NO
extra_body plumbing still, so the v4-flash-pin trap rule above is unchanged.)

**UPDATE 2026-09-01 (Phase 2 plumbing landed): the trap rule above is now ENFORCED, not
just documented.** `beam_adapter` has `--answer-extra-body` / `--judge-extra-body`;
`LLMClient` takes `extra_body` and merges it last; `_call` raises on FALSY content, not
merely null, because empty is the shape the trap actually takes. `check_model_pin()` runs
on both clients in `main()` and in `_rejudge_run`: a DeepSeek `v4-flash` model without
`thinking:disabled` exits 2, and a `thinking` key aimed at OpenAI/Gemini (a 400) exits 2.
A real-prompt canary runs on BOTH clients at each path's own `max_tokens` ceiling
before the run spends anything — previously only the rejudge judge was canaried, and the
expensive answer path had no guard at all. Artifacts record `answer_extra_body` /
`judge_extra_body`, so a reader no longer has to infer from the code whether thinking was
disabled.

**Current repository state, 2026-09-06:** defaults are pinned to
`deepseek-v4-flash`. Omitted raw-client bodies are resolved to thinking-disabled
only on DeepSeek v4-flash; BEAM's memory pipeline uses `auto`, as do LME, MSC,
and LoCoMo. Historical aliases remain only in dated plans, result filenames,
registry fixtures, and pure compatibility transformations; they cannot be used
to construct a live client. This is not an attestation of any already-running
remote process; verify its effective environment after deployment.

| File | What |
|---|---|
| `hymem/contrib/openai_client.py` | Default fallback → `deepseek-v4-flash` |
| `hymem/bootstrap.py` | `DEFAULT_LLM_MODEL` → `deepseek-v4-flash` |
| `benchmarks/longmemeval_adapter.py` | Reader, legacy judge, and memory-pipeline defaults → `deepseek-v4-flash`; raw DeepSeek v4 bodies default thinking-disabled and reject an incompatible explicit body. |
| `benchmarks/beam_adapter.py` | Reader, legacy judge, and memory-pipeline defaults → `deepseek-v4-flash`; pipeline thinking → `auto`. |
| `benchmarks/msc_adapter.py`, `benchmarks/locomo_adapter.py` | Reader, judge, and memory-pipeline defaults → `deepseek-v4-flash`; raw clients inherit LME's safe request-body resolution. |
| `~/.agent37/hooks/post-restart.sh` | `HYMEM_LLM_MODEL` → `deepseek-v4-flash` |
| `~/.hermes/bin/hymem-server-wrapper` | `HYMEM_LLM_MODEL` — the SINGLE source for MCP-server env; was the LAST live `deepseek-chat` reference (2026-08-07). After patching, kill the `hymem-server` CHILDREN only — watchdogs respawn them with the new env; verify `/proc/<pid>/environ`. Killing the watchdogs kills the bridge. |

`HYMEM_LLM_MODEL` overrides code defaults. Deployments must remove any deprecated
alias from wrappers/hooks, set `HYMEM_LLM_MODEL=deepseek-v4-flash` and
`HYMEM_LLM_THINKING=auto` (or `disabled`), restart Honcho, and verify the
effective process environment. A current build will now refuse startup/client
construction instead of silently spending against the retired alias;
repository changes cannot update a remote process environment.

## Non-DeepSeek reasoning models

`gpt-oss-120b` via OpenRouter may land output in `reasoning` instead of
`content`. `LLMClient._call()` needs the three-way fallback: `content or
reasoning or reasoning_content`. The historical migration context is retained
in [the BEAM model-pin pre-registration](../docs/plans/2026-09-01-beam-model-pin-pre-reg.md).
