"""Source-free stdin observer for the v3 diagnostic grounding pilot."""
from __future__ import annotations

import hashlib
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import stat
import sys


GROUND_READER_SHA256 = "bce6e26197d9831a90ba1015a7b129254c5368fa2adedff29050381ac31a1710"
WARM_V3_SHA256 = "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d"
WARM_V2_SHA256 = "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593"
ROOT_PARENT = Path("/home/atta")
OBSERVER_UID = 1000
ROOT_NAME = re.compile(r"\.hymem-luna-lme-observed-([A-Za-z0-9_-]{8,})\Z")
UNIT = re.compile(r"hymem-luna-lme-observed-[A-Za-z0-9_-]{8,}\.service\Z")
SCHEMA = "luna-observed-lme-v1"
RUNNER_SHA256 = "2062a8816e8e22214a1e7febc4898322210c381abdf9bb0f34e802d61d111550"
LAUNCHER_SHA256 = "1f503eae5121306a9ca70a6d1286e7b1cb898186ac49979403ff835367e3df3e"


def _root_from_argv():
    args = sys.argv[1:]
    if args.count("--root") != 1 or args.index("--root") + 1 >= len(args):
        raise RuntimeError("observer_root_argument_invalid")
    root = Path(args[args.index("--root") + 1])
    if (not root.is_absolute() or root.parent != ROOT_PARENT or
            ROOT_NAME.fullmatch(root.name) is None):
        raise RuntimeError("observer_root_argument_invalid")
    state = root.lstat()
    if not stat.S_ISDIR(state.st_mode) or state.st_uid != OBSERVER_UID or state.st_mode & 0o077:
        raise RuntimeError("observer_root_path_invalid")
    return root


def _ground_reader_path():
    source = Path(__file__)
    if source.name == "luna_observed_lme_progress.py" and source.is_file():
        return source.resolve().with_name("luna_grounding_lme_progress.py")
    return _root_from_argv() / "luna_grounding_lme_progress.py"


def _pinned(path, digest, identity):
    if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("observer_dependency_invalid")
    spec = importlib.util.spec_from_file_location(identity, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("observer_dependency_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[identity] = module
    spec.loader.exec_module(module)
    return module


ground = _pinned(_ground_reader_path(), GROUND_READER_SHA256,
                 "pinned_observed_grounding_reader")
base = ground.base
accepted_first_failure = base.first_failure_summary
accepted_terminal = ground.verify_terminal
ground.SCHEMA = SCHEMA
base.PROFILED = RUNNER_SHA256
base.UNIT = UNIT

PINS = {
    **ground.PINS,
    "luna_grounding_lme_progress.py": GROUND_READER_SHA256,
    "luna_grounding_lme_launch.py": ground.LAUNCHER_SHA256,
    "luna_observed_lme.py": RUNNER_SHA256,
    "luna_observed_lme_launch.py": LAUNCHER_SHA256,
    "codex_subscription_warm_v3.py": WARM_V3_SHA256,
}

_EVENTS = frozenset({
    "thread_started", "thread_status_changed", "thread_closed", "turn_started",
    "turn_completed", "turn_failed", "item_started", "item_completed",
    "item_agentMessage_delta", "item_reasoning_textDelta", "thread_tokenUsage_updated",
    "account_updated", "account_rateLimits_updated", "remoteControl_status_changed",
    "warning", "error", "model_rerouted", "model_verification",
    "model_safetyBuffering_updated", "turn_plan_updated", "turn_diff_updated",
    "item_reasoning_summaryPartAdded", "item_reasoning_summaryTextDelta",
    "unknown", "invalid_method",
})
_ERROR_CLASSES = frozenset({
    "contextWindowExceeded", "sessionBudgetExceeded", "usageLimitExceeded",
    "rateLimitExceeded", "flexUnavailable", "serverOverloaded", "cyberPolicy",
    "misalignmentPolicyViolation", "internalServerError", "unauthorized",
    "badRequest", "threadRollbackFailed", "sandboxError", "other",
    "httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected", "responseTooManyFailedAttempts",
    "activeTurnNotSteerable", "invalid", "unspecified",
})
_RPC_CATEGORIES = frozenset({"parse_error", "invalid_request", "method_not_found",
    "invalid_params", "internal_error", "other_numeric", "invalid"})
_CORE = frozenset({"code", "phase", "rpc", "process_index", "request_index",
    "retired_count", "queue_count", "known_tokens", "process_age_seconds",
    "turn_admitted", "known_usage", "usage_complete"})
_EXTRA = frozenset({"event_family", "failure_family", "app_server_error", "rpc_error"})


def first_failure_summary(value):
    if (type(value) is not dict or not _CORE <= set(value) or
            set(value) - (_CORE | _EXTRA)):
        return None
    core = accepted_first_failure(value)
    if core is None:
        return None
    result = dict(core)
    event = value.get("event_family")
    if "event_family" in value:
        if type(event) is not str or event not in _EVENTS:
            return None
        result["event_family"] = event
    if "failure_family" in value:
        if value["failure_family"] != "unexpected_notification" or event is None:
            return None
        result["failure_family"] = "unexpected_notification"
    if "app_server_error" in value:
        detail = value["app_server_error"]
        if type(detail) is not dict or type(detail.get("identity")) is not str or detail.get("identity") not in {
                "invalid", "unbound", "mismatch", "matched"}:
            return None
        identity = detail["identity"]
        if identity != "matched":
            if set(detail) != {"identity"}:
                return None
        else:
            if not {"identity", "will_retry", "error_class"} <= set(detail) or set(detail) - {
                    "identity", "will_retry", "error_class", "http_status_code"}:
                return None
            if not (type(detail["will_retry"]) is bool or detail["will_retry"] == "invalid"):
                return None
            if type(detail["error_class"]) is not str or detail["error_class"] not in _ERROR_CLASSES:
                return None
            if "http_status_code" in detail:
                status = detail["http_status_code"]
                if not ((type(status) is int and 0 <= status <= 65535) or
                        status is None or status == "invalid"):
                    return None
                if detail["error_class"] not in {"httpConnectionFailed",
                        "responseStreamConnectionFailed", "responseStreamDisconnected",
                        "responseTooManyFailedAttempts"}:
                    return None
        if event != "error":
            return None
        result["app_server_error"] = dict(detail)
    if "rpc_error" in value:
        detail = value["rpc_error"]
        if (type(detail) is not dict or set(detail) - {"category", "code"} or
                type(detail.get("category")) is not str or detail["category"] not in _RPC_CATEGORIES):
            return None
        if detail["category"] == "invalid":
            if set(detail) != {"category"}:
                return None
        elif set(detail) != {"category", "code"} or type(detail["code"]) is not int or not -2147483648 <= detail["code"] <= 2147483647:
            return None
        if value["rpc"] == "turn/events":
            return None
        result["rpc_error"] = dict(detail)
    return result


def root_identity_valid(root, unit, cgroup):
    match = ROOT_NAME.fullmatch(root.name)
    return bool(root.is_absolute() and root.parent == ROOT_PARENT and match is not None
        and not root.is_symlink() and root.is_dir()
        and unit == "hymem-luna-lme-observed-" + match.group(1) + ".service"
        and UNIT.fullmatch(unit)
        and cgroup == "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit)


def receipt_valid(receipt, root, unit, cgroup):
    if type(receipt) is not dict:
        return False
    source = receipt.get("source_sha256")
    expected = set(PINS) | {ground.DERIVED_STAMP}
    if type(source) is not dict or set(source) != expected:
        return False
    if any(source.get(name) != digest for name, digest in PINS.items()):
        return False
    derived = source.get(ground.DERIVED_STAMP)
    if type(derived) is not str or base.HEX.fullmatch(derived) is None:
        return False
    return bool(receipt.get("schema") == "luna-observed-launch-v1"
        and receipt.get("root") == str(root) and receipt.get("unit") == unit
        and receipt.get("expected_cgroup") == cgroup and receipt.get("output") == "run"
        and receipt.get("candidate") == str(root / "candidate")
        and receipt.get("original_candidate") == str(ground.ORIGINAL_CANDIDATE)
        and receipt.get("inventory_stamp") == str(root / ground.DERIVED_STAMP)
        and receipt.get("inventory_sha256") == derived
        and receipt.get("candidate_source_map_sha256") == ground.GROUNDED_MAP_SHA256
        and receipt.get("runner_sha256") == RUNNER_SHA256
        and receipt.get("launcher_sha256") == LAUNCHER_SHA256
        and receipt.get("inherited_warm_transport_sha256") == WARM_V2_SHA256
        and receipt.get("effective_warm_transport_sha256") == WARM_V3_SHA256
        and receipt.get("dataset_sha256") == base.DATASET
        and receipt.get("binary") == str(base.BINARY)
        and base.HEX.fullmatch(str(receipt.get("binary_sha256", "")))
        and receipt.get("runtime_max_seconds") == 14530
        and receipt.get("timeout_stop_seconds") == 10
        and receipt.get("memory_max_bytes") == 4294967296
        and receipt.get("cpu_quota_percent") == 200 and receipt.get("tasks_max") == 256
        and receipt.get("oom_policy") == "kill" and receipt.get("kill_mode") == "control-group"
        and receipt.get("restart") == "no" and receipt.get("remain_after_exit") is True
        and receipt.get("model") == "gpt-6-luna" and receipt.get("subscription_only") is True
        and receipt.get("reported_quota_floor_percent") == 25
        and receipt.get("limits") == base.LIMITS
        and type(receipt.get("dataset")) is str
        and Path(receipt["dataset"]).is_absolute())


def verify_terminal(root, receipt, safe, result):
    invalid = {"available": safe is not None, "validated": False,
        "questions": [{"index": i, "validated": False, "correct": None} for i in range(4)],
        "stage_accounting_reconciled": False, "warm_metrics_valid": False}
    if type(safe) is not dict or type(result) is not dict:
        return invalid
    for item in (safe, result):
        if (item.get("schema") != SCHEMA or
                item.get("effective_warm_transport_sha256") != WARM_V3_SHA256 or
                item.get("inherited_warm_transport_sha256") != WARM_V2_SHA256):
            return invalid
    if (safe.get("first_failure") is not None or
            (type(result.get("budget")) is dict and result["budget"].get("first_failure") is not None)):
        return invalid
    return accepted_terminal(root, receipt, safe, result)


base.first_failure_summary = first_failure_summary
base.root_identity_valid = root_identity_valid
base.receipt_valid = receipt_valid
base.verify_live_pins = ground.verify_live_pins
base.verify_terminal = verify_terminal


def main(argv=None):
    captured = io.StringIO()
    with redirect_stdout(captured):
        code = base.main(argv)
    lines = captured.getvalue().splitlines()
    if len(lines) != 1:
        raise RuntimeError("observer_report_invalid")
    report = json.loads(lines[0])
    if type(report) is not dict:
        raise RuntimeError("observer_report_invalid")
    report["schema"] = "luna-observed-lme-progress-v1"
    report["effective_warm_transport_sha256"] = WARM_V3_SHA256
    report["inherited_warm_transport_sha256"] = WARM_V2_SHA256
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
