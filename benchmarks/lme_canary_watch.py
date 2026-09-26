#!/usr/bin/env python3
"""Canary-window watcher for the LME run (operator-approved 2026-09-10: "Do 1").

Probes the official extraction canary on a fixed cadence. Exits the moment a
probe PASSES (code 0, "WATCH_FIRED") or after the bounded window expires
(code 4, "WATCH_EXPIRED"). Silent on failures -- every verdict is logged;
no raw provider content and no credentials are ever written.

On fire, the operator (agent) launches the approved chain:
    bash benchmarks/lme_canary_retry_run.sh 10 1   # smoke, then (after review)
    bash benchmarks/lme_canary_retry_run.sh 0 8    # full 500

Run:  cd ~/HyMem && . .hermes env -> /home/node/hymem-env/bin/python3
      benchmarks/lme_canary_watch.py /tmp/lme-canary-watch-<stamp>.log
"""
import datetime
import os
import sys
import time

sys.path.insert(0, "/home/node/HyMem")
from hymem.contrib.model_policy import RECOMMENDED_DEEPSEEK_MODEL
os.chdir("/home/node/HyMem")

from benchmarks.extraction_canary import (
    _FAILURE_REASONS,
    ExtractionCanaryError,
    run_configured_extraction_canary,
)

INTERVAL_S = 600
MAX_PROBES = 24          # ~4 h bounded window
MAX_CONSECUTIVE_ERRORS = 3

log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lme-canary-watch.log"
key = os.environ["DEEPSEEK_API_KEY"]


def log(message: str) -> None:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with open(log_path, "a") as fh:
        fh.write(line + "\n")


def failure_diagnostic(report: object) -> str:
    """Render only closed reason labels and bounded, genuine integer counters."""
    report = report if type(report) is dict else {}
    reason = report.get("failure_reason")
    reason = reason if type(reason) is str and reason in _FAILURE_REASONS else "unknown"
    path = report.get("execution_path")
    path = path if type(path) is dict else {}
    counters = []
    for label, field in (
        ("prose_em", "prose_claim_exact_context_emissions"),
        ("table_em", "table_claim_exact_context_emissions"),
    ):
        value = path.get(field)
        safe = str(value) if type(value) is int and 0 <= value <= 2**63 - 1 else "unknown"
        counters.append(f"{label}={safe}")
    return reason + " " + " ".join(counters)


log(f"WATCH_START interval={INTERVAL_S}s max_probes={MAX_PROBES}")

consecutive_errors = 0
for n in range(1, MAX_PROBES + 1):
    t0 = time.time()
    verdict = "?"
    reason = ""
    try:
        run_configured_extraction_canary(
            api_key=key,
            base_url="https://api.deepseek.com",
            model=RECOMMENDED_DEEPSEEK_MODEL,
            thinking="auto",
        )
        verdict = "PASS"
    except ExtractionCanaryError as exc:
        verdict = "FAIL"
        reason = failure_diagnostic(getattr(exc, "report", None))
    except Exception as exc:  # noqa: BLE001 - watcher must survive probe errors
        verdict = "ERROR"
        reason = "probe_error"

    duration = time.time() - t0
    log(
        f"PROBE {n}/{MAX_PROBES} {verdict}"
        + (f" reason={reason}" if reason else "")
        + f" ({duration:.1f}s)"
    )

    if verdict == "PASS":
        log("WATCH_FIRED")
        sys.exit(0)
    if verdict == "ERROR":
        consecutive_errors += 1
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            log("WATCH_ERROR_STREAK")
            sys.exit(5)
    else:
        consecutive_errors = 0

    if n < MAX_PROBES:
        time.sleep(INTERVAL_S)

log("WATCH_EXPIRED")
sys.exit(4)
