"""One-shot, two-call compatibility gate for the pinned native Hermes OAuth route.

Prepare is source-only. Only --run-root performs inference, inside the bounded
user service dispatched by --launch-root. Output contains finite metadata only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "hermes-native-oauth-probe-v2"
HOST_ROOT = Path("/home/atta")
HOST_UID = 1000
ROOT_NAME = re.compile(r"\.hymem-lme-diagnostic-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
TOOL = "hermes_native_oauth_probe_v2.py"
LAUNCHER = "luna_lme_diagnostic_launch_v9.py"
LAUNCHER_SHA = "baeb35caf1ddca21502e6d92a33f9c8c7a0f5f7618e7fb312eb90c1fb23fb445"
RUNNER_SHA = "1b83e3d3ecfa29cdaab5f5ad5aa0b8c4da844723c280de29b952449ef6902048"
NATIVE_V1_SHA = "fa88c3c2aa9cf042cb0110e5d9f3eea4ffd951270daf57d27dbe972bd2105e94"
NATIVE_SHA = "6845e6a395b40d22cad204bb1e6f4e62001aaa0f9a17609e15051fde620df7f4"
BRIDGE_SHA = "74d426d59db464cac2694a1633ae010febb065972f8f593a253e6ad298e28dd3"
INVENTORY_SHA = "1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
BINARY_SHA = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
BILLING_POLICY = "included_allowance_or_existing_finite_positive_credits_per_window_v2"
LIMITS = (2, 160_000, 300)
UNIT_PREFIX = "hymem-luna-native-oauth-probe-"
RUNTIME = Path("/run/user/1000")
CGROUP_ROOT = Path("/sys/fs/cgroup")
ORDINARY_SYSTEM = "Give a short plain text reply to the invented request."
ORDINARY_USER = "Invented probe: say the blue paper kite is ready."
SOURCE_TEXT = "In this invented example, Ada uses a cedar notebook."
RESULT_FIELDS = frozenset({"schema", "status", "turns", "known_tokens", "usage_complete",
    "reserved", "in_flight", "ordinary_completed", "staged_completed", "schema_acknowledged",
    "staged_response_valid", "broker_cleanup_verified", "containment_verified",
    "resource_denials", "resource_oom", "native_summary", "first_failure"})
STATUSES = frozenset({"transport_verified", "failed", "cap_reached", "accounting_unverified",
    "cleanup_unverified", "resource_unverified"})
FAULT_CODES = frozenset({"timeout", "invalid_auth_path", "auth_file_untrusted", "invalid_credentials",
    "account_mismatch", "account_unverified", "model_unverified", "auth_expired", "quota_unverified",
    "broker_closed", "admission_failure", "cleanup_failure", "broker_busy", "client_busy",
    "invalid_request_or_closed", "concurrent_completion_rejected", "transport_failure",
    "bridge_failure", "invalid_request", "request_limit", "invalid_timeout", "invalid_event",
    "invalid_usage", "missing_usage", "incomplete_response", "invalid_output", "unsupported_output",
    "output_limit", "event_limit", "event_after_completion", "response_failure",
    "missing_completion", "wire_limit", "truncated_stream", "http_failure", "invalid_content_type",
    "auth_failure", "access_failure", "quota_failure", "model_mismatch", "campaign_stopped",
    "question_stopped", "question_concurrent_invocation", "wall_limit", "campaign_wall_limit",
    "campaign_budget_exhausted", "question_budget_exhausted", "concurrency_limit",
    "budget_stopped_before_turn", "budget_exhausted_before_turn", "admission_rejected",
    "ledger_protocol_violation", "html_response", "access_challenge", "unsupported_media_type",
    "invalid_json", "budget_failure", "probe_failure", "first_turn_unverified",
    "stage_response_invalid", "stage_schema_unverified", "containment_or_resource_unverified"})
FAULT_PHASES = frozenset({"preflight", "admission", "http", "structured_output", "settlement", "cleanup"})


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def _write_once(path: Path, value: dict[str, Any]) -> None:
    data = _canonical(value)
    _require(len(data) <= 8192, "metadata_too_large")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    _fsync_parent(path)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _fsync_parent(path: Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _touch_once(path: Path) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    _fsync_parent(path)


def _root(root: Path) -> Path:
    _require(root.is_absolute() and root.parent == HOST_ROOT and ROOT_NAME.fullmatch(root.name) is not None
             and root.is_dir() and not root.is_symlink(), "root_invalid")
    info = root.stat()
    _require(info.st_uid == HOST_UID and stat.S_IMODE(info.st_mode) == 0o700, "root_permission_invalid")
    return root


def _tool(root: Path) -> Path:
    path = root / TOOL
    _require(_regular(path) and path.resolve() == Path(__file__).resolve()
             and path.stat().st_uid == HOST_UID, "tool_origin_invalid")
    return path


def _launcher(root: Path) -> Any:
    path = root / LAUNCHER
    _require(_regular(path) and _sha(path) == LAUNCHER_SHA, "launcher_source_drift")
    spec = importlib.util.spec_from_file_location("pinned_native_probe_launcher", path)
    _require(spec is not None and spec.loader is not None, "launcher_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _require(Path(module.__file__).resolve() == path.resolve(), "launcher_origin_invalid")
    return module


def verify_sources(root: Path) -> tuple[Any, Any, dict[str, Any]]:
    root = _root(root)
    _tool(root)
    launcher = _launcher(root)
    _require(launcher.HOST_ROOT == HOST_ROOT and launcher.RUNNER_SHA256 == RUNNER_SHA
             and launcher.INVENTORY_SHA256 == INVENTORY_SHA and launcher.BINARY_SHA256 == BINARY_SHA,
             "launcher_identity_invalid")
    runner, loaded = launcher.verify_sources(root)
    _require(Path(runner.__file__).resolve() == root / "code/tools/diagnostics/luna_lme_diagnostic_v10.py"
             and loaded["warm"] is loaded["staged"].warm
             and loaded["warm"].BILLING_POLICY == BILLING_POLICY
             and loaded["warm"].base.MODEL == "gpt-6-luna"
             and loaded["warm"].base.OVERRIDES["model_reasoning_effort"] == "low"
             and loaded["warm"].base.OVERRIDES["forced_login_method"] == "chatgpt",
             "staged_or_model_identity_invalid")
    code = root / "code"
    for relative, digest in (("benchmarks/hermes_codex_responses_v1.py", NATIVE_V1_SHA),
                             ("benchmarks/hermes_codex_responses_v2.py", NATIVE_SHA),
                             ("benchmarks/hermes_lme_oauth_v2.py", BRIDGE_SHA)):
        path = code / relative
        _require(_regular(path) and _sha(path) == digest, "native_source_drift")
    from benchmarks import hermes_lme_oauth_v2 as bridge
    from benchmarks import hermes_codex_responses_v2 as native
    from benchmarks import hermes_codex_responses_v1 as native_v1
    _require(Path(bridge.__file__).resolve() == code / "benchmarks/hermes_lme_oauth_v2.py"
             and Path(native.__file__).resolve() == code / "benchmarks/hermes_codex_responses_v2.py"
             and Path(native_v1.__file__).resolve() == code / "benchmarks/hermes_codex_responses_v1.py"
             and native.v1 is native_v1 and bridge.native is native
             and bridge.staged_v6 is loaded["staged"]
             and bridge.warm is loaded["warm"] and bridge.NativeLMEClient.__module__ == bridge.__name__,
             "native_import_identity_invalid")
    return launcher, bridge, loaded


def build_fixture(loaded: dict[str, Any]) -> tuple[Any, Any, Any]:
    from hymem.extraction.triples import Triple
    staged = loaded["staged"]
    _require(Path(sys.modules[Triple.__module__].__file__).resolve() ==
             loaded["candidate"] / "hymem/extraction/triples.py", "triple_origin_invalid")
    ordinary = loaded["request_type"](ORDINARY_SYSTEM, ORDINARY_USER,
        response_format="text", max_tokens=1024, temperature=0.0)
    triple = Triple("Ada", "uses", "cedar notebook", 1, source_message_id=1)
    source = staged.classification.GroundingSource(1, SOURCE_TEXT)
    request, batch = staged.staged.build_original_request((triple,), (source,))
    staged.staged.validate_original_request(request, batch)
    staged.staged.build_original_output_schema(batch)
    return ordinary, request, batch


def fixture_sha256() -> str:
    value = {"ordinary_system": ORDINARY_SYSTEM, "ordinary_user": ORDINARY_USER,
             "source": SOURCE_TEXT, "triple": ["Ada", "uses", "cedar notebook", 1, 1],
             "stage": "original", "recheck": False}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def unit_for(root: Path) -> str:
    _root(root)
    return UNIT_PREFIX + root.name.removeprefix(".hymem-lme-diagnostic-") + ".service"


def receipt_for(root: Path, launcher: Any) -> dict[str, Any]:
    runner = launcher._load_runner(root)
    unit = unit_for(root)
    return {"schema": SCHEMA, "root": str(root), "unit": unit,
        "model": "gpt-6-luna", "reasoning_effort": "low", "auth": "chatgpt",
        "billing_policy": BILLING_POLICY, "api_fallback": False, "topups": False,
        "transport": "legacy_codex_direct_responses_v2",
        "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "source_sha256": {**runner.PINS, "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256,
            "tools/diagnostics/luna_lme_diagnostic_v10.py": RUNNER_SHA,
            "benchmarks/hermes_codex_responses_v1.py": NATIVE_V1_SHA,
            "benchmarks/hermes_codex_responses_v2.py": NATIVE_SHA,
            "benchmarks/hermes_lme_oauth_v2.py": BRIDGE_SHA,
            LAUNCHER: LAUNCHER_SHA, TOOL: _sha(_tool(root))},
        "candidate_map_sha256": runner.ACCEPTED_MAP_SHA256,
        "inventory_sha256": INVENTORY_SHA, "dataset_sha256": runner.DATASET_SHA256,
        "binary_sha256": BINARY_SHA, "fixture_sha256": fixture_sha256(),
        "limits": list(LIMITS), "max_in_flight": 1, "invocation_seconds": 120,
        "runtime_seconds": 330, "stop_seconds": 10, "tasks_max": 256,
        "memory_max": 4_294_967_296, "cpu_percent": 200,
        "output": str(root / "safe-native-probe-result.json")}


def _receipt(root: Path, digest: str, launcher: Any, *, attempted: bool) -> dict[str, Any]:
    _require(type(digest) is str and HEX.fullmatch(digest) is not None, "receipt_sha_invalid")
    path = root / "native-probe-receipt.json"
    _require(_regular(path) and _sha(path) == digest, "receipt_pin_invalid")
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(raw == _canonical(receipt_for(root, launcher)), "receipt_invalid")
    marker = root / "native-probe-attempt.json"
    if attempted:
        _require(_regular(marker) and marker.read_bytes() ==
                 _canonical({"receipt_sha256": digest, "one_shot": True}), "attempt_invalid")
    else:
        _require(not marker.exists() and not (root / "native-probe-execution.json").exists()
                 and not (root / "safe-native-probe-result.json").exists(), "already_attempted")
    return value


def prepare(root: Path) -> dict[str, Any]:
    root = _root(root)
    _require(not any((root / name).exists() for name in (
        "native-probe-receipt.json", "native-probe-attempt.json", "native-probe-execution.json",
        "safe-native-probe-result.json", "native-probe-command.json", "native-probe-empty",
        "native-probe-tmp", "safe-native-probe-terminal.json", "private-native-probe-stderr.log")),
             "already_prepared")
    launcher, _, loaded = verify_sources(root)
    build_fixture(loaded)
    launcher.host_admission()
    for name in ("native-probe-empty", "native-probe-tmp"):
        (root / name).mkdir(mode=0o700)
    _write_once(root / "native-probe-receipt.json", receipt_for(root, launcher))
    return {"schema": SCHEMA, "prepared": True, "model_calls": 0,
            "root": str(root), "unit": unit_for(root),
            "receipt_sha256": _sha(root / "native-probe-receipt.json")}


def command(root: Path, receipt: dict[str, Any], digest: str) -> list[str]:
    return ["/usr/bin/systemd-run", "--user", "--quiet", "--unit", receipt["unit"],
        "--property=Type=exec", "--property=Restart=no", "--property=KillMode=control-group",
        "--property=RemainAfterExit=yes", "--property=RuntimeMaxSec=330s",
        "--property=TimeoutStopSec=10s", "--property=MemoryMax=4294967296",
        "--property=CPUQuota=200%", "--property=TasksMax=256", "--property=OOMPolicy=kill",
        "--property=UMask=0077", "--property=WorkingDirectory=" + str(root / "native-probe-empty"),
        "--property=StandardOutput=file:" + str(root / "safe-native-probe-terminal.json"),
        "--property=StandardError=file:" + str(root / "private-native-probe-stderr.log"),
        "/usr/bin/env", "-i", "HOME=/home/atta", "PATH=/usr/local/bin:/usr/bin:/bin",
        "XDG_RUNTIME_DIR=" + str(RUNTIME),
        "DBUS_SESSION_BUS_ADDRESS=unix:path=" + str(RUNTIME / "bus"),
        "TMPDIR=" + str(root / "native-probe-tmp"), "/usr/bin/python3", "-I", "-B",
        str(root / TOOL), "--run-root", str(root), "--receipt-sha256", digest]


def launch(root: Path, digest: str) -> dict[str, Any]:
    root = _root(root)
    launcher, _, _ = verify_sources(root)
    launcher.host_admission()
    receipt = _receipt(root, digest, launcher, attempted=False)
    for name in ("native-probe-empty", "native-probe-tmp"):
        path = root / name
        _require(path.is_dir() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o700
                 and not any(path.iterdir()), "workdir_invalid")
    _write_once(root / "native-probe-attempt.json", {"receipt_sha256": digest, "one_shot": True})
    for name in ("safe-native-probe-terminal.json", "private-native-probe-stderr.log"):
        _touch_once(root / name)
    started = subprocess.run(command(root, receipt, digest), capture_output=True, timeout=20,
                             check=False, env=launcher._bus_env())
    _write_once(root / "native-probe-command.json", {"returncode": started.returncode})
    return {"schema": SCHEMA, "root": str(root), "unit": receipt["unit"],
            "receipt_sha256": digest, "launch_command_returncode": started.returncode,
            "never_retry": True}


def _unit_values(receipt: dict[str, Any]) -> dict[str, str]:
    fields = ("ActiveState", "SubState", "MainPID", "ControlGroup", "NRestarts", "Result",
        "ExecMainStatus", "MemoryMax", "TasksMax", "CPUQuotaPerSecUSec", "KillMode",
        "Restart", "RemainAfterExit", "OOMPolicy", "RuntimeMaxUSec", "TimeoutStopUSec")
    env = {"HOME": str(HOST_ROOT), "PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": str(RUNTIME),
           "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(RUNTIME / "bus")}
    completed = subprocess.run(["/usr/bin/systemctl", "--user", "show", receipt["unit"],
        "--property=" + ",".join(fields), "--no-pager"], capture_output=True, text=True,
        timeout=10, check=True, env=env)
    _require(len(completed.stdout) <= 8192, "unit_state_invalid")
    pairs = [line.split("=", 1) for line in completed.stdout.splitlines()]
    _require(len(pairs) == len(fields) and all(len(pair) == 2 for pair in pairs)
             and len(dict(pairs)) == len(fields), "unit_state_invalid")
    values = dict(pairs)
    _require(set(values) == set(fields) and values["NRestarts"] == "0"
        and values["MemoryMax"] == "4294967296" and values["TasksMax"] == "256"
        and values["CPUQuotaPerSecUSec"] in {"2s", "2.000s"}
        and values["KillMode"] == "control-group" and values["Restart"] == "no"
        and values["RemainAfterExit"] == "yes" and values["OOMPolicy"] == "kill"
        and values["RuntimeMaxUSec"] in {"5min 30s", "5min 30.000s", "330s", "330.000s"}
        and values["TimeoutStopUSec"] in {"10s", "10.000s"}, "service_policy_invalid")
    return values


def _group(receipt: dict[str, Any]) -> Path:
    path = CGROUP_ROOT / receipt["expected_cgroup"].lstrip("/")
    _require(not CGROUP_ROOT.is_symlink() and path.resolve().is_relative_to(CGROUP_ROOT.resolve()),
             "cgroup_path_invalid")
    node = path
    while node != CGROUP_ROOT:
        _require(not node.is_symlink(), "cgroup_path_invalid")
        node = node.parent
    return path


def _group_policy(group: Path) -> bool:
    try:
        cpu = (group / "cpu.max").read_text().split()
        return ((group / "memory.max").read_text().strip() == "4294967296"
            and (group / "pids.max").read_text().strip() == "256"
            and len(cpu) == 2 and all(part.isdecimal() for part in cpu)
            and int(cpu[1]) > 0 and int(cpu[0]) == 2 * int(cpu[1]))
    except (OSError, ValueError):
        return False


def _live_containment(receipt: dict[str, Any]) -> bool:
    try:
        values, group = _unit_values(receipt), _group(receipt)
        return (values["ActiveState"] == "active" and values["SubState"] == "running"
            and values["MainPID"] == str(os.getpid())
            and values["ControlGroup"] == receipt["expected_cgroup"]
            and f"0::{receipt['expected_cgroup']}" in Path("/proc/self/cgroup").read_text().splitlines()
            and _group_policy(group)
            and values["MainPID"] in (group / "cgroup.procs").read_text().splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _resources(receipt: dict[str, Any]) -> tuple[int | None, int | None]:
    try:
        group = _group(receipt)
        pids = dict(line.split() for line in (group / "pids.events").read_text().splitlines())
        memory = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
        denials, oom = pids.get("max"), memory.get("oom_kill")
        if denials is not None and oom is not None and denials.isdecimal() and oom.isdecimal():
            return int(denials), int(oom)
    except (OSError, ValueError):
        pass
    return None, None


def _terminal_runtime(receipt: dict[str, Any], launcher: Any) -> tuple[str, bool]:
    try:
        values = _unit_values(receipt)
        group = _group(receipt)
        if (values["ActiveState"], values["SubState"]) == ("active", "running"):
            if values["MainPID"].isdecimal() and int(values["MainPID"]) > 0 \
                    and values["ControlGroup"] == receipt["expected_cgroup"] \
                    and _group_policy(group) and values["MainPID"] in (group / "cgroup.procs").read_text().splitlines():
                return "running", False
            return "unverified", False
        if ((values["ActiveState"], values["SubState"]) not in
                {("active", "exited"), ("inactive", "dead"), ("failed", "failed")}
                or values["MainPID"] != "0"
                or values["ControlGroup"] not in {"", receipt["expected_cgroup"]}
                or not launcher.recursive_empty(group)):
            return "unverified", False
        clean = values["Result"] == "success" and values["ExecMainStatus"] == "0"
        return ("clean_exit" if clean else "failed_exit"), True
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unverified", False


def _fault(code: Any, phase: Any, *, turn_admitted: Any = None,
           known_usage: Any = None) -> dict[str, Any]:
    value = {"code": code if type(code) is str and code in FAULT_CODES else "probe_failure",
             "phase": phase if type(phase) is str and phase in FAULT_PHASES else "preflight"}
    if type(turn_admitted) is bool:
        value["turn_admitted"] = turn_admitted
    if type(known_usage) is bool:
        value["known_usage"] = known_usage
    return value


def _valid_fault(value: Any) -> bool:
    if type(value) is not dict or not {"code", "phase"} <= set(value) \
            or not set(value) <= {"code", "phase", "turn_admitted", "known_usage"}:
        return False
    return _fault(value["code"], value["phase"],
                  turn_admitted=value.get("turn_admitted"),
                  known_usage=value.get("known_usage")) == value


def _summary(value: Any) -> dict[str, Any] | None:
    if type(value) is not dict or set(value) != {"schema", "calls", "successes", "failures",
            "turns", "known_tokens", "usage_complete", "timing_seconds", "timing_saturated",
            "first_failure", "last_failure_code"} or value["schema"] != "native_oauth_summary_v1":
        return None
    if (any(type(value[key]) is not int or not 0 <= value[key] <= 2 for key in
            ("calls", "successes", "failures", "turns"))
            or type(value["known_tokens"]) is not int or value["known_tokens"] < 0
            or type(value["usage_complete"]) is not bool
            or type(value["timing_saturated"]) is not bool
            or type(value["timing_seconds"]) is not dict
            or set(value["timing_seconds"]) != {"total", "admission", "http"}
            or any(type(value["timing_seconds"][key]) not in (int, float)
                   or not math.isfinite(value["timing_seconds"][key])
                   or not 0 <= value["timing_seconds"][key] <= 1_000_000
                   for key in ("total", "admission", "http"))
            or value["calls"] != value["successes"] + value["failures"]
            or (value["first_failure"] is not None and not _valid_fault(value["first_failure"]))):
        return None
    last = value["last_failure_code"]
    if last is not None and (type(last) is not str or last not in FAULT_CODES):
        return None
    if value["successes"] == 2 and (value["timing_saturated"] or value["first_failure"] is not None
                                    or last is not None):
        return None
    return value


def run_probe(loaded: dict[str, Any], bridge: Any, root: Path,
              receipt: dict[str, Any]) -> dict[str, Any]:
    ordinary_request, stage_request, batch = build_fixture(loaded)
    warm, staged = loaded["warm"], loaded["staged"]
    budget = warm.SharedBudget(warm.BudgetLimits(*LIMITS), max_in_flight=1)
    client = broker = None
    ordinary_done = staged_done = acknowledged = valid = False
    cleanup_ok = True
    containment = _live_containment(receipt)
    denials, oom = _resources(receipt)
    status, first = "failed", None
    summary = None
    try:
        _require(containment and denials == 0 and oom == 0, "containment_or_resource_unverified")
        broker = bridge.AdmissionBroker(str(loaded["binary"]), str(HOST_ROOT / ".codex/auth.json"),
                                        max_requests=2, max_age_seconds=300)
        client = bridge.NativeLMEClient(broker, budget, "probe",
            warm.BudgetLimits(*LIMITS))
        answer = client.complete(ordinary_request)
        ordinary_done = type(answer) is str and bool(answer)
        del answer
        state = budget.snapshot()
        _require(ordinary_done and state["turns"] == 1 and state["reserved"] == 0
                 and state["in_flight"] == 0 and state["usage_complete"] is True
                 and not state["stopped"] and 0 < state["known_tokens"] < LIMITS[1],
                 "first_turn_unverified")
        raw = client.complete_stage(stage_request, batch, "original", False)
        staged_done = type(raw) is str and bool(raw)
        controls = client.requested_controls[-1]
        acknowledged = controls.get("output_schema_sent") is True \
            and controls.get("output_schema_acknowledged") is True
        _require(acknowledged, "stage_schema_unverified")
        try:
            review = staged.staged.parse_original_response(raw, batch)
            valid = review.batch_sha256 == batch.batch_sha256
        except staged.staged.GroundingContractError:
            valid = False
        del raw
        _require(valid, "stage_response_invalid")
        status = "transport_verified"
    except BaseException as exc:
        if client is not None and client.first_failure is not None:
            first = _fault(**client.first_failure)
        else:
            first = _fault(exc.args[0] if isinstance(exc, ValueError) and len(exc.args) == 1
                           else "probe_failure", "structured_output" if staged_done else
                           "settlement" if ordinary_done else "preflight")
    finally:
        if client is not None:
            summary = _summary(client.diagnostic_summary())
            try:
                client.close()
            except BaseException:
                cleanup_ok = False
        if broker is not None:
            try:
                broker.close()
            except BaseException:
                cleanup_ok = False
        denials, oom = _resources(receipt)
    state = budget.snapshot()
    if first is None and state.get("first_failure") is not None:
        fault = state["first_failure"]
        first = _fault(fault.get("code"), fault.get("phase"),
                       turn_admitted=fault.get("turn_admitted"), known_usage=fault.get("known_usage"))
    if first is None and state.get("stop_code") is not None:
        first = _fault(state["stop_code"], "settlement")
    if not cleanup_ok:
        status = "cleanup_unverified"
        first = first or _fault("cleanup_failure", "cleanup")
    elif not containment or denials is None or oom is None or denials or oom:
        status = "resource_unverified"
        first = first or _fault("containment_or_resource_unverified", "preflight")
    elif (state["reserved"] != 0 or state["in_flight"] != 0
          or state["usage_complete"] is not True or state["turns"] > 2
          or summary is None or summary["turns"] != state["turns"]
          or summary["known_tokens"] != state["known_tokens"]):
        status = "accounting_unverified"
        first = first or _fault("probe_failure", "settlement")
    elif status == "transport_verified" and (state["turns"] != 2 or state["known_tokens"] <= 0
          or not ordinary_done or not staged_done or not acknowledged or not valid
          or state["stopped"] or summary["calls"] != 2 or summary["successes"] != 2
          or summary["failures"] != 0 or first is not None):
        status = "accounting_unverified"
        first = first or _fault("probe_failure", "settlement")
    elif state["known_tokens"] >= LIMITS[1]:
        status = "cap_reached"
    return {"schema": SCHEMA, "status": status, "turns": state["turns"],
        "known_tokens": state["known_tokens"], "usage_complete": state["usage_complete"],
        "reserved": state["reserved"], "in_flight": state["in_flight"],
        "ordinary_completed": ordinary_done, "staged_completed": staged_done,
        "schema_acknowledged": acknowledged, "staged_response_valid": valid,
        "broker_cleanup_verified": cleanup_ok, "containment_verified": containment,
        "resource_denials": denials, "resource_oom": oom, "native_summary": summary,
        "first_failure": first}


def validate_result(value: Any) -> bool:
    if type(value) is not dict or set(value) != RESULT_FIELDS or value["schema"] != SCHEMA \
            or type(value["status"]) is not str or value["status"] not in STATUSES:
        return False
    if (any(type(value[key]) is not int or value[key] < 0 for key in
            ("turns", "known_tokens", "reserved", "in_flight"))
            or value["turns"] > 2 or any(type(value[key]) is not bool for key in
            ("usage_complete", "ordinary_completed", "staged_completed", "schema_acknowledged",
             "staged_response_valid", "broker_cleanup_verified", "containment_verified"))
            or any(value[key] is not None and (type(value[key]) is not int or value[key] < 0)
                   for key in ("resource_denials", "resource_oom"))
            or value["native_summary"] is not None and _summary(value["native_summary"]) is None):
        return False
    fault = value["first_failure"]
    if fault is not None and not _valid_fault(fault):
        return False
    if (value["ordinary_completed"] and value["turns"] < 1
            or value["staged_completed"] and (not value["ordinary_completed"] or value["turns"] != 2)
            or value["schema_acknowledged"] and not value["staged_completed"]
            or value["staged_response_valid"] and not value["staged_completed"]
            or value["turns"] == 0 and value["known_tokens"] != 0):
        return False
    if value["status"] == "transport_verified":
        summary = value["native_summary"]
        return (value["turns"] == 2 and 0 < value["known_tokens"] < LIMITS[1]
            and value["reserved"] == value["in_flight"] == 0 and value["usage_complete"]
            and value["ordinary_completed"] and value["staged_completed"]
            and value["schema_acknowledged"] and value["staged_response_valid"]
            and value["broker_cleanup_verified"] and value["containment_verified"]
            and value["resource_denials"] == value["resource_oom"] == 0 and fault is None
            and summary is not None and summary["calls"] == summary["successes"] == 2
            and summary["failures"] == 0 and summary["known_tokens"] == value["known_tokens"]
            and summary["turns"] == 2 and summary["usage_complete"]
            and not summary["timing_saturated"] and summary["first_failure"] is None
            and summary["last_failure_code"] is None)
    return True


def inspect(root: Path, digest: str) -> dict[str, Any]:
    root = _root(root)
    launcher, _, _ = verify_sources(root)
    receipt = _receipt(root, digest, launcher, attempted=True)
    runtime, recursive_cleanup = _terminal_runtime(receipt, launcher)
    result = None
    path = root / "safe-native-probe-result.json"
    marker = root / "native-probe-execution.json"
    executed = _regular(marker) and marker.read_bytes() == \
        _canonical({"receipt_sha256": digest, "one_shot": True})
    if executed and _regular(path) and path.stat().st_size <= 8192:
        try:
            candidate = json.loads(path.read_text(encoding="ascii"))
            if validate_result(candidate):
                result = candidate
        except (OSError, ValueError, UnicodeError):
            pass
    status = ("compatibility_verified" if result is not None
        and result["status"] == "transport_verified" and runtime == "clean_exit" and recursive_cleanup
        else "compatibility_failed" if result is not None and recursive_cleanup
        and runtime in {"clean_exit", "failed_exit"} else "running" if runtime == "running"
        else "unverified")
    return {"schema": SCHEMA, "status": status, "runtime": runtime,
            "recursive_cleanup_verified": recursive_cleanup,
            "result": result if recursive_cleanup else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ("prepare-root", "launch-root", "run-root", "inspect-root"):
        group.add_argument("--" + action)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    root = Path(args.prepare_root or args.launch_root or args.run_root or args.inspect_root)
    try:
        if args.prepare_root:
            _require(args.receipt_sha256 is None, "unexpected_receipt")
            output = prepare(root)
        elif args.launch_root:
            _require(args.receipt_sha256 is not None, "receipt_required")
            output = launch(root, args.receipt_sha256)
        elif args.inspect_root:
            _require(args.receipt_sha256 is not None, "receipt_required")
            output = inspect(root, args.receipt_sha256)
        else:
            _require(args.receipt_sha256 is not None, "receipt_required")
            root = _root(root)
            launcher, bridge, loaded = verify_sources(root)
            receipt = _receipt(root, args.receipt_sha256, launcher, attempted=True)
            _write_once(root / "native-probe-execution.json",
                        {"receipt_sha256": args.receipt_sha256, "one_shot": True})
            output = run_probe(loaded, bridge, root, receipt)
            _require(validate_result(output), "result_invalid")
            _write_once(root / "safe-native-probe-result.json", output)
        print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0 if (args.prepare_root or args.launch_root or args.inspect_root
                     or output["status"] == "transport_verified") else 1
    except BaseException:
        print(json.dumps({"schema": SCHEMA, "status": "unverified"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
