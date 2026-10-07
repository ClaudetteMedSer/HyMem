"""Read-only, finite postmortem metadata for the stopped p9dzuk8y pilot.

No logs, stores, private rows, model text, credentials or provider calls. Only
validated terminal bookkeeping and finite classifications from bounded private
error slots are projected; private error messages are never exported.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

READER_SHA = "5e56efbadbab74af2a2b4019c19f36523d417890cce70ba1006b900553f72c00"
ROOT = "/home/atta/.hymem-lme-diagnostic-preflight-p9dzuk8y"
RECEIPT = "5df45dee407347006c136642971e333ee8282bf8d83aaee545651570acf18380"
SCHEMA = "luna-timeout-metadata-root-v1"
SLOTS = ("canary", "q-0000", "q-0001", "q-0002", "q-0003")
ERRORS = frozenset({"contextWindowExceeded", "sessionBudgetExceeded", "usageLimitExceeded",
    "rateLimitExceeded", "flexUnavailable", "serverOverloaded", "cyberPolicy",
    "misalignmentPolicyViolation", "internalServerError", "unauthorized", "badRequest",
    "threadRollbackFailed", "sandboxError", "other", "httpConnectionFailed",
    "responseStreamConnectionFailed", "responseStreamDisconnected",
    "responseTooManyFailedAttempts", "activeTurnNotSteerable", "unspecified"})


def number(value, cap=100_000_000):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= cap:
        raise ValueError("invalid_number")
    return value


def projection(terminal, errors):
    budget = terminal["budget"]
    fault = budget["first_failure"]
    if fault["code"] != "timeout" or fault["phase"] != "run" or fault["rpc"] != "turn/events":
        raise ValueError("wrong_failure")
    out = {"schema": SCHEMA, "first_failure_process_age_seconds":
           number(fault["process_age_seconds"], 1_000_000), "timings": {}, "questions": {},
           "private_error_slots": []}
    for key in ("preflight_seconds", "model_seconds", "cleanup_seconds"):
        out["timings"][key] = number(budget["timings"][key])
    questions = budget["questions"]
    if type(questions) is not dict or set(questions) - set(SLOTS):
        raise ValueError("invalid_question_slots")
    for slot in SLOTS:
        if slot not in questions:
            continue
        row = questions[slot]
        if any(type(row[k]) is not bool for k in ("usage_complete", "stopped")):
            raise ValueError("invalid_flags")
        out["questions"][slot] = {key: number(row[key]) for key in ("turns", "known_tokens", "in_flight")}
        out["questions"][slot].update({key: row[key] for key in ("usage_complete", "stopped")})
    if len(errors) > 80:
        raise ValueError("too_many_slots")
    for slot, index, record in errors:
        if slot not in SLOTS or type(index) is not int or not 0 <= index < 16:
            raise ValueError("invalid_slot")
        if (type(record) is not dict or record.get("schema") != "warm_private_failure_v1"
                or type(record.get("error_count")) is not int or not 1 <= record["error_count"] <= 4096
                or record.get("failure_code") not in {"timeout", "campaign_stopped", "unexpected_notification:error"}):
            raise ValueError("invalid_error_record")
        item = {"slot": slot, "index": index, "failure_code": record["failure_code"],
                "error_count": record["error_count"]}
        for key in ("first", "last"):
            raw = record[key]
            if (type(raw) is not dict or type(raw.get("error_class")) is not str
                    or raw["error_class"] not in ERRORS or type(raw.get("will_retry")) is not bool):
                raise ValueError("invalid_error_class")
            status = raw.get("http_status_code")
            if status is not None and (type(status) is not int or not 0 <= status <= 65535):
                raise ValueError("invalid_http_status")
            item[key] = {"error_class": raw["error_class"], "http_status_code": status,
                         "will_retry": raw["will_retry"]}
        out["private_error_slots"].append(item)
    return out


def collect(source):
    namespace = {"__name__": "reviewed_timeout_reader", "__file__": "<reviewed-reader>"}
    exec(compile(source, "<reviewed-reader>", "exec"), namespace)
    root = Path(ROOT)
    status = namespace["inspect"](root, RECEIPT)
    if (status["status"] != "terminal_incomplete_or_unclean"
            or status["runtime_cleanup_verified"] is not True
            or status["budget_stop_code"] != "timeout"):
        raise ValueError("terminal_not_verified")
    terminal = namespace["_read"](root / "run/diagnostic-result.json", root, 128_000)
    errors = []
    for slot in SLOTS:
        for index in range(16):
            path = root / "run" / slot / f"warm-private-failure-{index:02d}.json"
            if path.exists() or path.is_symlink():
                errors.append((slot, index, namespace["_read"](path, root, 20_480)))
    return projection(terminal, errors)


def main():
    reader = Path(__file__).with_name("luna_lme_diagnostic_progress_v8.py").read_bytes()
    if hashlib.sha256(reader).hexdigest() != READER_SHA:
        raise ValueError("reader_pin_mismatch")
    # Reuse only these exact reviewed definitions remotely. The collector is
    # explicitly called, so the script's local dispatcher is not run remotely.
    own_source = Path(__file__).read_text()
    # __name__ intentionally differs from __main__; no recursive SSH dispatch.
    payload = ("import json\nns={'__name__':'remote_metadata_only'}\nexec(compile("
        + repr(own_source) + ",'<metadata-only>','exec'),ns)\ntry:\n"
        + "    print(json.dumps(ns['collect'](" + repr(reader.decode()) + "),sort_keys=True))\n"
        + "except Exception:\n    print(json.dumps({'status':'metadata_unavailable'}))\n")
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "afrodite", "python3 -I -B -"],
        input=payload, text=True, capture_output=True, timeout=30)
    try:
        out = json.loads(result.stdout)
        if result.returncode != 0 or type(out) is not dict or out.get("schema") != SCHEMA:
            raise ValueError("unverified_response")
        # Re-project from the typed public shape to prevent arbitrary response text.
        assert set(out) == {"schema", "first_failure_process_age_seconds", "timings", "questions", "private_error_slots"}
        rebuilt = {"budget": {"first_failure": {"code": "timeout", "phase": "run", "rpc": "turn/events",
            "process_age_seconds": out["first_failure_process_age_seconds"]},
            "timings": out["timings"], "questions": out["questions"]}}
        records = [(x["slot"], x["index"], {**x, "schema": "warm_private_failure_v1"}) for x in out["private_error_slots"]]
        print(json.dumps(projection(rebuilt, records), sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status": "metadata_unavailable"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
