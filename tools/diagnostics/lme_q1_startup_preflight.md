# Provider-free Q1 startup check

This diagnostic proves startup compatibility, not benchmark completion or score.
It constructs the real OpenAI-compatible SDK client using a fixed synthetic key,
compares its exact producer declaration with the real standalone declaration,
then invokes the stock CLI through its physical checkpoint boundary. The first
reader constructor is replaced by a deliberate stop; the real CLI must release
its actual checkpoint lease. No canary, extraction, reader or judge call occurs.
The checkpoint's recorded producer must match the independently constructed
client. Completed questions and provider counters must remain zero.

Use a dedicated process: a permanent audit hook blocks DNS and outgoing socket
operations while allowing local socket allocation needed by SDK initialization.
This includes `getaddrinfo`, `gethostbyname`/`gethostbyname_ex`, `gethostbyaddr`
and `getnameinfo`, not only socket connections. A fresh mode-0700 `output/home`
is created exclusively and used as `HOME`; existing directories, files or
symlinks at that path are rejected unchanged. The private home is retained with
the diagnostic artifacts on success or failure, never deleted automatically.
Environment overrides and credentials are cleared before imports. Only the
synthetic key is installed; `HYMEM_LLM_EXTRA_BODY` is explicitly absent. The
original environment is restored for cleanup, but the audit hook stays active.
No credential files, production stores, Docker or remote operations are used.

Example (all paths must be absolute; output must not already exist):

```sh
/opt/anaconda3/bin/python -I -B tools/diagnostics/lme_q1_startup_preflight.py \
  --source /absolute/reviewed/source \
  --data-dir /absolute/benchmark/data \
  --output /absolute/fresh/diagnostic-output
```

The targeted recipe selects source position 210 from 500 questions with seed 53
and expects ID `09ba9854`. A synthetic 500-row corpus with that ID is sufficient
for helper testing. This helper does not certify dataset/source hashes or
freeze an execution package; the surrounding reviewed launcher must do that.
The sole artifact is a private diagnostic checkpoint plus bounded metadata.
The result never claims a scored benchmark ran.

Helper controls are outside the application's ordinary tests:

```sh
LME_Q1_PREFLIGHT_CURRENT_SOURCE=/absolute/reviewed/current/source \
LME_Q1_PREFLIGHT_LEGACY_SOURCE=/absolute/reconstructed/r3/source \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/opt/anaconda3/bin/python -B -m pytest -p no:cacheprovider -o addopts= \
  tools/diagnostics/tests/test_lme_q1_startup_preflight.py
```

Without the optional legacy path that single regression is explicitly skipped.
The current-source default is the checkout, not a claim that its version was
reviewed, frozen or approved for live execution. Passing this diagnostic does
not authorize paid calls or replace subsequent end-to-end validation.
