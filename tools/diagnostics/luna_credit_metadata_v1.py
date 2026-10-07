"""One bounded, zero-inference credit and quota metadata read for a stopped run.

Run --local-ssh from the repository. The remote payload embeds only hash-pinned
stdlib sources and emits a finite projection. It does not change the spending gate.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SCHEMA = "luna-credit-metadata-v1"
ROOT = Path("/home/atta/.hymem-lme-diagnostic-preflight-xvn648yr")
RECEIPT_SHA256 = "06a5d2461d679bb5a4c945995a3ce6e20d84e599e5e2d48cbdfebdd7a2d7dc0e"
ACCESS_SHA256 = "a0bc7ec37e243947418e92725dbd42e66adaea7941e3d25317c6cc4676c4c139"
HELPER_SHA256 = "3fd553772d7865a10d2505ef2a7b4ea2d9f9a8a28ab24b64c45feb9a31cfa320"
BASE_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
REACHED = frozenset({"rate_limit_reached", "workspace_owner_credits_depleted",
    "workspace_member_credits_depleted", "workspace_owner_usage_limit_reached",
    "workspace_member_usage_limit_reached"})
BALANCES = frozenset({"missing", "null", "zero", "positive", "negative", "invalid"})
GATE = frozenset({"unknown_quota", "invalid_quota", "quota_exhausted", "quota_floor",
    "credit_balance_present", "subscription_plan_unverified", "quota_above_floor",
    "quota_unverified"})
EMBEDDED_HELPER = None  # Replaced only by the hash-checked local launcher.


def _module(source: bytes, digest: str, name: str, filename: str = "<pinned>") -> dict[str, Any]:
    if type(source) is not bytes or hashlib.sha256(source).hexdigest() != digest:
        raise ValueError("source_unverified")
    namespace: dict[str, Any] = {"__name__": name, "__file__": filename}
    exec(compile(source, "<pinned>", "exec"), namespace)
    return namespace


def _bound_access(value: Any) -> None:
    result = value.get("result") if type(value) is dict else None
    fault = result.get("first_failure") if type(result) is dict else None
    if not (type(value) is dict and value.get("schema") == "luna-subscription-access-check-v1"
            and value.get("status") == "access_failed"
            and value.get("runtime") == "failed_exit"
            and value.get("recursive_cleanup_verified") is True
            and type(result) is dict and result.get("turns") == 0
            and result.get("usage_complete") is True
            and type(fault) is dict and fault.get("code") == "credit_balance_present"):
        raise ValueError("terminal_unverified")


def _balance(credits: dict[str, Any]) -> str:
    if "balance" not in credits:
        return "missing"
    value = credits["balance"]
    if value is None:
        return "null"
    if type(value) is not str or len(value) > 128 or not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?", value):
        return "invalid"
    try:
        number = Decimal(value)
        return "positive" if number > 0 else "negative" if number < 0 else "zero"
    except InvalidOperation:
        return "invalid"


def _flag(value: Any) -> bool | None:
    if value is None or type(value) is bool:
        return value
    raise ValueError("invalid_metadata")


def _window(value: Any, kind: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("invalid_window")
    field = "remainingPercent" if kind == "individual" else "usedPercent"
    percent = value.get(field)
    if type(percent) not in (int, float) or not math.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("invalid_window")
    reset = value.get("resetsAt")
    duration = None if kind == "individual" else value.get("windowDurationMins")
    for number in (reset, duration):
        if number is not None and (type(number) is not int or number < 0):
            raise ValueError("invalid_window")
    return {"kind": kind, "remaining_percent": percent if kind == "individual" else 100 - percent,
            "window_minutes": duration, "resets_at": reset}


def _project(response: Any, quota_metadata: Any, gate_error: type[Exception]) -> tuple[str, list[dict[str, Any]]]:
    if type(response) is not dict:
        raise ValueError("invalid_metadata")
    by_id = response.get("rateLimitsByLimitId")
    if by_id is not None and type(by_id) is not dict:
        raise ValueError("invalid_metadata")
    snapshots = list(by_id.values()) if by_id else [response.get("rateLimits")]
    if len(snapshots) > 16 or any(type(s) is not dict for s in snapshots):
        raise ValueError("invalid_metadata")
    output = []
    total_windows = 0
    for index, snapshot in enumerate(snapshots):
        credits = snapshot.get("credits")
        if credits is not None and type(credits) is not dict:
            raise ValueError("invalid_metadata")
        if credits is None:
            has_credits = unlimited = None
            balance = "missing"
        else:
            if "hasCredits" not in credits or "unlimited" not in credits:
                raise ValueError("invalid_metadata")
            has_credits = _flag(credits.get("hasCredits"))
            unlimited = _flag(credits.get("unlimited"))
            if has_credits is None or unlimited is None:
                raise ValueError("invalid_metadata")
            balance = _balance(credits)
        reached = snapshot.get("rateLimitReachedType")
        if reached is not None and (type(reached) is not str or reached not in REACHED):
            raise ValueError("invalid_metadata")
        windows = []
        for kind in ("primary", "secondary", "individual"):
            item = _window(snapshot.get("individualLimit" if kind == "individual" else kind), kind)
            if item is not None:
                windows.append(item)
        total_windows += len(windows)
        if total_windows > 16:
            raise ValueError("invalid_metadata")
        output.append({"index": index, "has_credits": has_credits, "unlimited": unlimited,
            "balance": balance, "spend_control_reached": _flag(snapshot.get("spendControlReached")),
            "rate_limit_reached_type": reached, "windows": windows})
    try:
        quota_metadata(response)
        gate = "quota_above_floor"
    except gate_error as exc:
        code = str(exc)
        gate = code if code in GATE else "quota_unverified"
    return gate, output


def _fallback(status: str = "source_unverified") -> dict[str, Any]:
    return {"schema": SCHEMA, "status": status, "owned_process_cleanup_verified": None}


def _inspect(helper: dict[str, Any], access: dict[str, Any]) -> dict[str, Any]:
    helper["ROOT"] = ROOT
    inspect = access["inspect"]
    _bound_access(inspect(ROOT, RECEIPT_SHA256))
    helper["_pinned_file"](helper["BINARY"], BINARY_SHA256, executable=True)
    base = helper["_pinned_file"](ROOT / "code/benchmarks/codex_subscription.py", BASE_SHA256)
    session_class, quota_metadata, gate_error = helper["_load_transport"](base)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "rpc_unverified",
        "owned_process_cleanup_verified": False, "auth": "unknown", "plan": "unknown",
        "gate_status": "quota_unverified", "snapshots": []}
    session = None
    try:
        session = helper["AllowlistedSession"](session_class, helper["BINARY"])
        session.rpc("initialize")
        session.rpc("initialized")
        report["auth"], report["plan"] = helper["_account_projection"](session.rpc("account/read"))
        if report["auth"] != "chatgpt" or report["plan"] == "unknown":
            report["status"] = "account_unverified"
        else:
            report["gate_status"], report["snapshots"] = _project(
                session.rpc("account/rateLimits/read"), quota_metadata, gate_error)
            report["status"] = "metadata_read"
    except helper["PartialStartupFailure"] as exc:
        report["owned_process_cleanup_verified"] = exc.cleanup_verified
    except Exception:
        pass
    finally:
        if session is not None:
            try:
                report["owned_process_cleanup_verified"] = bool(session.close())
            except Exception:
                report["owned_process_cleanup_verified"] = False
    return report


def _safe(value: Any) -> dict[str, Any]:
    fallback = _fallback("ssh_unverified")
    if type(value) is not dict or value.get("schema") != SCHEMA:
        return fallback
    if value == _fallback():
        return value
    if set(value) != {"schema", "status", "owned_process_cleanup_verified", "auth",
                      "plan", "gate_status", "snapshots"}:
        return fallback
    if (type(value["status"]) is not str
            or value["status"] not in {"rpc_unverified", "account_unverified", "metadata_read"}
            or type(value["owned_process_cleanup_verified"]) is not bool
            or type(value["auth"]) is not str
            or value["auth"] not in {"chatgpt", "other", "unknown"}
            or type(value["plan"]) is not str
            or value["plan"] not in {"plus", "pro", "prolite", "free", "go", "team", "unknown"}
            or type(value["gate_status"]) is not str
            or value["gate_status"] not in GATE or type(value["snapshots"]) is not list
            or len(value["snapshots"]) > 16):
        return fallback
    count = 0
    for index, item in enumerate(value["snapshots"]):
        if type(item) is not dict or set(item) != {"index", "has_credits", "unlimited",
                "balance", "spend_control_reached", "rate_limit_reached_type", "windows"}:
            return fallback
        if (type(item["index"]) is not int or item["index"] != index
                or any(flag is not None and type(flag) is not bool for flag in
                    (item["has_credits"], item["unlimited"], item["spend_control_reached"]))
                or type(item["balance"]) is not str or item["balance"] not in BALANCES
                or (item["rate_limit_reached_type"] is not None and
                    (type(item["rate_limit_reached_type"]) is not str or
                     item["rate_limit_reached_type"] not in REACHED))
                or type(item["windows"]) is not list or len(item["windows"]) > 3):
            return fallback
        for window in item["windows"]:
            count += 1
            if type(window) is not dict or set(window) != {"kind", "remaining_percent",
                    "window_minutes", "resets_at"} or type(window["kind"]) is not str or window["kind"] not in {
                    "primary", "secondary", "individual"}:
                return fallback
            percent = window["remaining_percent"]
            if type(percent) not in (int, float) or not math.isfinite(percent) or not 0 <= percent <= 100:
                return fallback
            for key in ("window_minutes", "resets_at"):
                number = window[key]
                if number is not None and (type(number) is not int or number < 0):
                    return fallback
    return value if count <= 16 else fallback


def _local_ssh() -> int:
    report = _fallback("ssh_unverified")
    try:
        directory = Path(__file__).resolve().parent
        helper = (directory / "luna_subscription_access_metadata_v1.py").read_bytes()
        if hashlib.sha256(helper).hexdigest() != HELPER_SHA256:
            raise ValueError("source_unverified")
        source = Path(__file__).read_text()
        for marker, payload in (("EMBEDDED_HELPER", helper),):
            lines = source.splitlines(keepends=True)
            matches = [i for i, line in enumerate(lines) if line.startswith(marker + " = None  #")]
            if len(matches) != 1:
                raise ValueError("source_unverified")
            lines[matches[0]] = marker + " = bytes.fromhex(" + repr(payload.hex()) + ")\n"
            source = "".join(lines)
        completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1", "afrodite", "/usr/bin/python3", "-I", "-B", "-",
            "--root", str(ROOT), "--receipt-sha256", RECEIPT_SHA256],
            input=source.encode(), capture_output=True, timeout=45, check=False)
        if len(completed.stdout) <= 16384:
            report = _safe(json.loads(completed.stdout))
        if completed.returncode and report.get("status") == "metadata_read":
            report = _fallback("ssh_unverified")
    except Exception:
        pass
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report.get("status") == "metadata_read" and report.get(
        "owned_process_cleanup_verified") is True else 1


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--local-ssh"]:
        return _local_ssh()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    report = _fallback()
    try:
        if args.root != str(ROOT) or args.receipt_sha256 != RECEIPT_SHA256:
            raise ValueError("identity_unverified")
        helper = _module(EMBEDDED_HELPER, HELPER_SHA256, "_pinned_metadata_helper")
        access_path = ROOT / "access-check-v1.py"
        access_source = helper["_pinned_file"](access_path, ACCESS_SHA256)
        access = _module(access_source, ACCESS_SHA256, "_pinned_access_source", str(access_path))
        report = _safe(_inspect(helper, access))
    except Exception:
        pass
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report.get("status") == "metadata_read" and report.get(
        "owned_process_cleanup_verified") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
