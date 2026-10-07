"""Read only bounded, validated metadata for the one-shot timeout services."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import types
from typing import Any


SCHEMA = "luna-timeout-progress-v1"
HOST_SHA256 = "f38c77ee73b05e1004dd9863b873665c55f568bd4e50782701aeaa48ce11d93c"
HOST_UID = 1000
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular(path: Path, cap: int) -> bool:
    try:
        meta = path.lstat()
        return (stat.S_ISREG(meta.st_mode) and meta.st_uid == HOST_UID and
                stat.S_IMODE(meta.st_mode) == 0o600 and meta.st_size <= cap)
    except OSError:
        return False


def _read(path: Path, cap: int) -> Any:
    _need(_regular(path, cap), "file_missing_or_invalid")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            _need(key not in out, "duplicate_json_key")
            out[key] = value
        return out
    def invalid(_: str) -> None:
        raise ValueError("nonfinite_json")
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=invalid)


def _host(root: Path) -> Any:
    _need(sys.platform == "linux" and os.getuid() == HOST_UID and
          os.geteuid() == HOST_UID, "host_user_invalid")
    path = root / "timeout-host-v1.py"
    _need(_regular(path, 100_000), "host_origin_invalid")
    source = path.read_bytes()
    _need(hashlib.sha256(source).hexdigest() == HOST_SHA256, "host_drift")
    module = types.ModuleType("pinned_luna_timeout_host_progress")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    module._root(root)
    return module


def _receipt(host: Any, root: Path, digest: str, mode: str) -> dict[str, Any]:
    _need(type(digest) is str and HEX.fullmatch(digest) is not None,
          "receipt_argument_invalid")
    return host.verify_receipt(root, digest, mode)


def _attempt(root: Path, digest: str, mode: str) -> bool:
    path = root / ("containment-attempt.json" if mode == "containment" else
                   "launch-attempt.json")
    if not path.exists() and not path.is_symlink():
        return False
    value = _read(path, 512)
    canonical = json.dumps({"receipt_sha256": digest, "one_shot": True},
                           sort_keys=True, separators=(",", ":")).encode("ascii")
    _need(value == {"receipt_sha256": digest, "one_shot": True} and
          path.read_bytes() == canonical, "attempt_invalid")
    return True


def _execution(root: Path, digest: str, mode: str) -> bool:
    path = root / ("containment-execution-marker.json" if mode == "containment" else
                   "probe-execution-marker.json")
    if not path.exists() and not path.is_symlink():
        return False
    value = _read(path, 512)
    canonical = json.dumps({"receipt_sha256": digest, "execution_started": True},
                           sort_keys=True, separators=(",", ":")).encode("ascii")
    _need(value == {"receipt_sha256": digest, "execution_started": True} and
          path.read_bytes() == canonical, "execution_marker_invalid")
    return True


def _runtime_clean(runtime: dict[str, Any]) -> bool:
    return (runtime == {"policy_verified": True, "unit_stopped": True,
            "group_matched": True, "recursive_cleanup_verified": True,
            "runtime_exit": "success"})


def inspect(root: Path, digest: str, mode: str) -> dict[str, Any]:
    _need(mode in {"containment", "probe"}, "mode_invalid")
    host = _host(root)
    receipt = _receipt(host, root, digest, mode)
    # Always reverify the complete accepted closure.  Probe result validation
    # uses these exact loaded bytes and does not execute another read/import.
    probe, (observer, _) = host.verify_sources(root)
    attempted = _attempt(root, digest, mode)
    execution_started = _execution(root, digest, mode)
    _need(not execution_started or attempted, "execution_without_attempt")
    runtime = host.terminal_runtime(receipt) if attempted else None
    disk_free = shutil.disk_usage(root).free
    result_path = root / ("containment-result.json" if mode == "containment" else
                          "probe-result.json")
    base = {"schema": SCHEMA, "mode": mode, "root": str(root),
            "unit": receipt["unit"], "receipt_sha256": digest,
            "host_sha256": HOST_SHA256, "probe_sha256": host.PROBE_SHA256,
            "observer_sha256": host.OBSERVER_SHA256,
            "preparation_sha256": host.PREPARATION_SHA256,
            "attempted": attempted, "execution_started": execution_started,
            "runtime": runtime,
            "disk_free_bytes": disk_free,
            "disk_floor_met": disk_free >= 20 * 1024**3,
            "historical_timeout_cause_proved": False,
            "lme_readiness_proved": False}
    if not result_path.exists() and not result_path.is_symlink():
        absent = ("prepared" if not attempted else "result_missing" if
                  runtime is not None and runtime["unit_stopped"] is True else
                  "pending_or_ambiguous")
        return {**base, "status": absent,
                "model_calls": None, "completed_and_clean": False,
                "result_verified": False, "probe": None, "containment": None}
    _need(attempted and execution_started, "result_without_execution")
    value = _read(result_path, 1_000_000)
    _need(host.validate_host_result(value, probe=probe, observer=observer) and
          value["mode"] == mode, "result_invalid")
    if mode == "containment":
        verified = (value["verified"] is True and value["model_calls"] == 0 and
                    runtime is not None and _runtime_clean(runtime))
        return {**base, "status": "verified_clean" if verified else "unverified",
                "model_calls": 0, "completed_and_clean": False,
                "result_verified": True, "probe": None,
                "containment": {"verified": value["verified"],
                    "resource_counters": value["resource_counters"],
                    "verified_and_clean": verified}}
    candidate = value["probe_result"]
    if candidate is None:
        return {**base, "status": "incomplete_or_failed", "model_calls": None,
                "completed_and_clean": False, "result_verified": True,
                "containment": None,
                "probe": {"status": value["status"],
                          "host_failure_code": value["failure_code"],
                          "samples": value["samples"], "counts": None,
                          "known_tokens": None, "usage_complete": None,
                          "first_failure": None, "observations": None}}
    counts = {key: candidate[key] for key in
              ("attempted", "returned", "failed", "not_attempted", "turns")}
    clean = runtime is not None and _runtime_clean(runtime)
    zero_resources = all(item["gate"] == {"containment": True,
        "denials": 0, "oom": 0} for item in value["samples"])
    completed = (value["status"] == "observed_success" and
                 candidate["returned"] == 16 and candidate["failed"] == 0 and
                 candidate["usage_complete"] is True and zero_resources and clean)
    observations = [{"record_id": row["record_id"], "worker": row["worker"],
                     "status": row["status"], "observation": row["observation"]}
                    for row in candidate["records"]]
    return {**base, "status": "completed_and_clean" if completed else
                    "incomplete_or_failed", "model_calls": None,
            "completed_and_clean": completed, "result_verified": True,
            "containment": None,
            "probe": {"status": value["status"],
                "host_failure_code": value["failure_code"],
                "samples": value["samples"], "counts": counts,
                "known_tokens": candidate["known_tokens"],
                "usage_complete": candidate["usage_complete"],
                "budget_stop_code": candidate["budget_stop_code"],
                "runner_fault": candidate["runner_fault"],
                "first_failure": candidate["first_failure"],
                "observations": observations}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--mode", choices=("containment", "probe"), required=True)
    args = parser.parse_args(argv)
    try:
        result = inspect(Path(args.root), args.receipt_sha256, args.mode)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"),
                         allow_nan=False))
        return 0 if result["status"] in {"verified_clean", "completed_and_clean"} else 1
    except BaseException:
        print(json.dumps({"schema": SCHEMA, "status": "unverified"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
