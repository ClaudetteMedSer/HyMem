# Diagnostic LME source bundle (offline)

`luna_lme_diagnostic_bundle_v1.py` copies verified source bytes into a new,
private root. It never opens a Codex session, calls a model, changes the
accepted startup bundle, or creates launch authorization. Its output contains
`candidate/` (514 mapped files), `code/` (the accepted staged transport and
the new diagnostic entry/helpers), and `source-map.json`.

Run once with a fresh, nonexistent output path:

```sh
/opt/anaconda3/bin/python3.13 tools/diagnostics/luna_lme_diagnostic_bundle_v1.py \
  --repo /Users/attavanwestreenen/AGprojects/HyMem \
  --accepted-code /private/tmp/hymem-staged-startup-root-vBg9BWn4/bundle/code \
  --candidate /private/tmp/hymem-staged-v1-root-fZvNIRgs/candidate \
  --map /private/tmp/hymem-staged-v1-root-fZvNIRgs/map.json \
  --output /private/tmp/CHOOSE-A-FRESH-DIAGNOSTIC-ROOT
```

The report includes every copied code hash, the accepted candidate mapping
hash, and the inventory file hash. It explicitly reports that the dataset,
Codex binary, launch receipt, and model calls are absent. Do not treat source
assembly as full preflight or permission to run. The local desktop does not
have the frozen `longmemeval_s_cleaned.json`; no placeholder or re-export is
acceptable. On the host with the real frozen dataset and binary, invoke the
runner's `--preflight-only` using their absolute paths and independently
verified SHA-256 values. That checks the full candidate inventory, selected
first-source questions, module origins, helper/transport identities, dataset,
and binary without inference.

The maintained profiled launcher records the hosted dataset at
`/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json`
and the pinned Codex binary at
`/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex`.
After copying the new source-only root to a fresh private host directory,
verify the binary digest there and substitute it below. This is read-only
preflight, not a service launch:

```sh
/usr/bin/python3 -I -B /FRESH-ROOT/code/tools/diagnostics/luna_lme_diagnostic_v1.py \
  --root /FRESH-ROOT \
  --inventory /FRESH-ROOT/source-map.json \
  --inventory-sha256 228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf \
  --dataset /opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json \
  --binary /home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex \
  --binary-sha256 VERIFIED-HOST-BINARY-SHA256 \
  --questions 4 --preflight-only
```

The `--run` path is intentionally separate. It requires a fresh source-bound
`launch-receipt.json`, matching one-shot `launch-attempt.json`, and an already
active systemd unit whose MainPID, cgroup, runtime and resource controls pass
the runner's live verifier. The source bundle does not create those artifacts
or launch that unit. A reviewed launcher, private-output reader, and full
hosted preflight are still required before live use. Do not reuse or mutate
the completed eight-unit staged-probe receipt, bundle, or service.

Offline verification on the assembled source root succeeded under isolated
Python 3.13: the 514-file inventory, staged transport/candidate binding, and
real frozen extraction canary, adapter, protocol, strictness, chunk, and
summary-state imports. The full `load_verified` path was not exercised on the
desktop because the real dataset and Codex binary were unavailable there.
