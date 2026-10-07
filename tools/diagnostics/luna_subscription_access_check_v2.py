"""One-shot, invented-text subscription access check on the accepted Luna sources.

Install this file as <private-root>/access-check-v2.py after the accepted source-only
preflight. Prepare makes no model request. Only --run-root is invoked by the
one-shot, bounded user service. Public output is finite metadata only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "luna-subscription-access-check-v2"
BILLING_POLICY = "included_allowance_or_existing_finite_positive_credits_v1"
HOST_ROOT = Path("/home/atta")
HOST_UID = 1000
TOOL_NAME = "access-check-v2.py"
LAUNCHER_SHA256 = "5449880912f28aeeee89a4907599b82cab5b56a87df2af5414d70374cb3d15a7"
RUNNER_SHA256 = "bc055ebe4621ec729723357b7db09f66049b34aaa5d9d67cb09dae16178f680d"
INVENTORY_SHA256 = "228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf"
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
ROOT_NAME = re.compile(r"\.hymem-lme-diagnostic-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
UNIT_PREFIX = "hymem-luna-access-check-"
RUNTIME = Path("/run/user/1000")
CGROUP_ROOT = Path("/sys/fs/cgroup")
LIMITS = (2, 32_000, 300)
ORDINARY_SYSTEM = "Answer the next invented request with a short plain-text reply."
ORDINARY_USER = "Invented access probe: say that the blue paper kite is ready."
SOURCE_TEXT = "In this invented example, Ada uses a cedar notebook."
RESULT_FIELDS = frozenset({"schema", "status", "turns", "known_tokens",
    "usage_complete", "reserved", "in_flight", "ordinary_completed",
    "staged_completed", "schema_acknowledged", "staged_response_valid",
    "client_cleanup_verified", "resource_denials", "resource_oom",
    "containment_verified", "first_failure"})
STATUSES = frozenset({"transport_verified", "failed", "cap_reached",
    "accounting_unverified", "cleanup_unverified", "resource_unverified"})


def _require(ok: bool, code: str) -> None:
    if not ok:
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
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as output:
        json.dump(value, output, sort_keys=True, separators=(",", ":"), allow_nan=False)
        output.flush()
        os.fsync(output.fileno())


def _root(root: Path) -> Path:
    _require(root.is_absolute() and root.parent == HOST_ROOT
        and ROOT_NAME.fullmatch(root.name) is not None
        and root.is_dir() and not root.is_symlink(), "root_invalid")
    info = root.stat()
    _require(info.st_uid == HOST_UID and stat.S_IMODE(info.st_mode) == 0o700,
             "root_permission_invalid")
    return root


def _tool(root: Path) -> Path:
    path = root / TOOL_NAME
    _require(_regular(path) and path.resolve() == Path(__file__).resolve()
             and path.stat().st_uid == HOST_UID, "tool_origin_invalid")
    return path


def _launcher(root: Path) -> Any:
    # The frozen source-only map deliberately excludes host launch helpers.
    # Install this accepted helper beside the access tool and pin it separately.
    path = root / "luna_lme_diagnostic_launch_v6.py"
    _require(_regular(path) and _sha(path) == LAUNCHER_SHA256,
             "launcher_source_drift")
    spec = importlib.util.spec_from_file_location("pinned_luna_access_launcher", path)
    _require(spec is not None and spec.loader is not None, "launcher_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _require(Path(module.__file__).resolve() == path.resolve(), "launcher_origin_invalid")
    return module


def verify_sources(root: Path) -> tuple[Any, dict[str, Any]]:
    """Use the accepted launcher and runner's complete source/runtime gate."""
    root = _root(root)
    _tool(root)
    launcher = _launcher(root)
    _require(launcher.HOST_ROOT == HOST_ROOT and launcher.BINARY_SHA256 == BINARY_SHA256
             and launcher.RUNNER_SHA256 == RUNNER_SHA256
             and launcher.INVENTORY_SHA256 == INVENTORY_SHA256,
             "launcher_identity_invalid")
    runner = launcher.verify_sources(root)
    loaded = runner.load_verified(root, root / "source-map.json",
        INVENTORY_SHA256, launcher.DATASET, launcher.BINARY, BINARY_SHA256,
        runner.DIAGNOSTIC_HELPER_SHA256)
    _require(loaded["warm"] is loaded["staged"].warm
             and loaded["warm"].BILLING_POLICY == BILLING_POLICY
             and runner.BILLING_POLICY == BILLING_POLICY
             and loaded["warm"].base.MODEL == "gpt-6-luna"
             and loaded["warm"].base.OVERRIDES["model_reasoning_effort"] == "low"
             and loaded["warm"].base.OVERRIDES["forced_login_method"] == "chatgpt",
             "transport_binding_invalid")
    return launcher, loaded


def fixture_sha256() -> str:
    fixture = {"ordinary_system": ORDINARY_SYSTEM, "ordinary_user": ORDINARY_USER,
               "source": SOURCE_TEXT, "triple": ["Ada", "uses", "cedar notebook", 1, 1],
               "stage": "original", "recheck": False}
    encoded = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def unit_for(root: Path) -> str:
    _root(root)
    return UNIT_PREFIX + root.name.removeprefix(".hymem-lme-diagnostic-") + ".service"


def receipt_for(root: Path, launcher: Any) -> dict[str, Any]:
    unit = unit_for(root)
    runner = launcher._load_runner(root)
    return {"schema": SCHEMA, "root": str(root), "unit": unit,
        "billing_policy": BILLING_POLICY,
        "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "source_sha256": {**runner.PINS,
            "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256,
            "tools/diagnostics/luna_lme_diagnostic_v6.py": RUNNER_SHA256,
            "luna_lme_diagnostic_launch_v6.py": LAUNCHER_SHA256,
            TOOL_NAME: _sha(_tool(root))},
        "candidate_map_sha256": runner.ACCEPTED_MAP_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "dataset_sha256": runner.DATASET_SHA256,
        "binary_sha256": BINARY_SHA256,
        "fixture_sha256": fixture_sha256(), "limits": list(LIMITS),
        "invocation_seconds": 120, "runtime_seconds": 330,
        "tasks_max": 256, "memory_max": 4_294_967_296,
        "cpu_percent": 200, "stop_seconds": 10,
        "output": str(root / "safe-access-result.json")}


def _receipt(root: Path, digest: str, launcher: Any, *, attempted: bool) -> dict[str, Any]:
    _require(HEX.fullmatch(digest) is not None, "receipt_sha_invalid")
    path = root / "access-receipt.json"
    _require(_regular(path) and _sha(path) == digest, "receipt_pin_invalid")
    value = json.loads(path.read_text(encoding="ascii"))
    _require(value == receipt_for(root, launcher), "receipt_invalid")
    marker = root / "access-attempt.json"
    if attempted:
        _require(_regular(marker) and json.loads(marker.read_text(encoding="ascii")) ==
            {"receipt_sha256": digest, "one_shot": True}, "attempt_invalid")
    else:
        _require(not marker.exists() and not (root / "safe-access-result.json").exists(),
                 "already_attempted")
    return value


def prepare(root: Path) -> dict[str, Any]:
    root = _root(root)
    _require(not any((root / name).exists() for name in
        ("access-receipt.json", "access-attempt.json", "safe-access-result.json",
         "access-command-result.json", "private-access", "access-empty", "access-tmp")),
        "already_prepared")
    launcher, loaded = verify_sources(root)
    build_fixture(loaded)
    launcher.host_admission()
    for name in ("private-access", "access-empty", "access-tmp"):
        (root / name).mkdir(mode=0o700)
    receipt = receipt_for(root, launcher)
    _write_once(root / "access-receipt.json", receipt)
    return {"schema": SCHEMA, "prepared": True, "model_calls": 0,
            "receipt_sha256": _sha(root / "access-receipt.json"),
            "root": str(root), "unit": receipt["unit"]}


def command(root: Path, receipt: dict[str, Any], digest: str) -> list[str]:
    return ["/usr/bin/systemd-run", "--user", "--quiet", "--unit", receipt["unit"],
        "--property=Type=exec", "--property=Restart=no",
        "--property=KillMode=control-group", "--property=RemainAfterExit=yes",
        "--property=RuntimeMaxSec=330s", "--property=TimeoutStopSec=10s",
        "--property=MemoryMax=4294967296", "--property=CPUQuota=200%",
        "--property=TasksMax=256", "--property=OOMPolicy=kill",
        "--property=UMask=0077", "--property=WorkingDirectory=" + str(root / "access-empty"),
        "--property=StandardOutput=file:" + str(root / "safe-access-terminal.json"),
        "--property=StandardError=file:" + str(root / "private-access-stderr.log"),
        "/usr/bin/env", "-i", "HOME=/home/atta", "PATH=/usr/local/bin:/usr/bin:/bin",
        "XDG_RUNTIME_DIR=" + str(RUNTIME),
        "DBUS_SESSION_BUS_ADDRESS=unix:path=" + str(RUNTIME / "bus"),
        "TMPDIR=" + str(root / "access-tmp"), "/usr/bin/python3", "-I", "-B",
        str(root / TOOL_NAME), "--run-root", str(root), "--receipt-sha256", digest]


def launch(root: Path, digest: str) -> dict[str, Any]:
    root = _root(root)
    launcher, _ = verify_sources(root)
    launcher.host_admission()
    receipt = _receipt(root, digest, launcher, attempted=False)
    for name in ("private-access", "access-empty", "access-tmp"):
        path = root / name
        _require(path.is_dir() and not path.is_symlink()
                 and stat.S_IMODE(path.stat().st_mode) == 0o700
                 and not any(path.iterdir()), "workdir_invalid")
    _write_once(root / "access-attempt.json",
                {"receipt_sha256": digest, "one_shot": True})
    started = subprocess.run(command(root, receipt, digest), capture_output=True,
                             timeout=20, check=False, env=launcher._bus_env())
    _write_once(root / "access-command-result.json", {"returncode": started.returncode})
    return {"schema": SCHEMA, "root": str(root), "unit": receipt["unit"],
            "receipt_sha256": digest, "launch_command_returncode": started.returncode,
            "never_retry": True}


def _unit_values(receipt: dict[str, Any]) -> dict[str, str]:
    fields = ("ActiveState", "SubState", "MainPID", "ControlGroup", "NRestarts",
        "Result", "ExecMainStatus", "MemoryMax", "TasksMax", "CPUQuotaPerSecUSec",
        "KillMode", "Restart", "RemainAfterExit", "OOMPolicy", "RuntimeMaxUSec",
        "TimeoutStopUSec")
    env = {**os.environ, "XDG_RUNTIME_DIR": str(RUNTIME),
           "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(RUNTIME / "bus")}
    completed = subprocess.run(["/usr/bin/systemctl", "--user", "show", receipt["unit"],
        "--property=" + ",".join(fields), "--no-pager"], capture_output=True,
        text=True, timeout=10, check=True, env=env)
    values = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
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
    _require(path.resolve().is_relative_to(CGROUP_ROOT), "cgroup_path_invalid")
    return path


def _group_policy(group: Path) -> bool:
    try:
        cpu = (group / "cpu.max").read_text().split()
        return ((group / "memory.max").read_text().strip() == "4294967296"
            and (group / "pids.max").read_text().strip() == "256"
            and len(cpu) == 2 and all(value.isdecimal() for value in cpu)
            and int(cpu[1]) > 0 and int(cpu[0]) == 2 * int(cpu[1]))
    except (OSError, ValueError):
        return False


def _live_containment(receipt: dict[str, Any]) -> bool:
    try:
        values = _unit_values(receipt)
        group = _group(receipt)
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


def _recursive_empty(group: Path) -> bool:
    try:
        if not group.exists():
            return True
        if not _group_policy(group) or any((group / name).read_text().strip()
            for name in ("cgroup.procs", "cgroup.threads")):
            return False
        events = dict(line.split(None, 1) for line in (group / "cgroup.events").read_text().splitlines()
                      if len(line.split(None, 1)) == 2)
        return events.get("populated") == "0"
    except (OSError, ValueError):
        return False


def _terminal_runtime(receipt: dict[str, Any]) -> tuple[str, bool]:
    try:
        values = _unit_values(receipt)
        if (values["ActiveState"] == "active" and values["SubState"] == "running"):
            group = _group(receipt)
            if (values["MainPID"].isdecimal() and int(values["MainPID"]) > 0
                    and values["ControlGroup"] == receipt["expected_cgroup"]
                    and _group_policy(group)
                    and values["MainPID"] in (group / "cgroup.procs").read_text().splitlines()):
                return "running", False
            return "unverified", False
        if ((values["ActiveState"], values["SubState"]) not in
                {("active", "exited"), ("inactive", "dead"), ("failed", "failed")}
                or values["MainPID"] != "0"
                or values["ControlGroup"] not in {"", receipt["expected_cgroup"]}
                or not _recursive_empty(_group(receipt))):
            return "unverified", False
        clean = values["Result"] == "success" and values["ExecMainStatus"] == "0"
        return ("clean_exit" if clean else "failed_exit"), True
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unverified", False


def build_fixture(loaded: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Validate both invented requests before any possible model invocation."""
    staged = loaded["staged"]
    from hymem.extraction.triples import Triple
    _require(Path(sys.modules[Triple.__module__].__file__).resolve() ==
             loaded["candidate"] / "hymem/extraction/triples.py", "triple_origin_invalid")
    ordinary = loaded["request_type"](ORDINARY_SYSTEM, ORDINARY_USER,
        response_format="text", max_tokens=1024, temperature=0.0)
    triple = Triple("Ada", "uses", "cedar notebook", 1, source_message_id=1)
    source = staged.v2.GroundingSource(1, SOURCE_TEXT)
    staged_request, batch = staged.staged.build_original_request((triple,), (source,))
    staged.staged.validate_original_request(staged_request, batch)
    staged.staged.build_original_output_schema(batch)
    return ordinary, staged_request, batch


def run_access(loaded: dict[str, Any], root: Path,
               receipt: dict[str, Any]) -> dict[str, Any]:
    """Exactly one ordinary then one staged turn; no response text is retained."""
    warm, staged = loaded["warm"], loaded["staged"]
    ordinary_request, staged_request, batch = build_fixture(loaded)
    budget = warm.SharedBudget(warm.BudgetLimits(*LIMITS), max_in_flight=1)
    limits = warm.BudgetLimits(1, 32_000, 300)
    sink = warm.PrivateFailureSink(root / "private-access")
    fd = sink._open_directory()
    os.close(fd)
    ordinary = second = None
    ordinary_done = staged_done = acknowledged = False
    staged_response_valid: bool | None = None
    cleanup_ok = True
    containment = _live_containment(receipt)
    denials, oom = _resources(receipt)
    status = "failed"
    try:
        _require(containment and denials == 0 and oom == 0,
                 "containment_or_resource_unverified")
        ordinary = warm.WarmSubscriptionClient(str(loaded["binary"]), budget,
            "ordinary", limits, max_requests=1, private_failure_sink=sink)
        second = staged.StagedSubscriptionClient(str(loaded["binary"]), budget,
            "staged", limits, max_requests=1, private_failure_sink=sink)
        answer = ordinary.complete(ordinary_request)
        ordinary_done = type(answer) is str and bool(answer)
        del answer
        state = budget.snapshot()
        _require(ordinary_done and state["turns"] == 1 and state["reserved"] == 0
                 and state["in_flight"] == 0 and state["usage_complete"] is True
                 and not state["stopped"] and 0 < state["known_tokens"] <= 32_000,
                 "first_turn_unverified")
        if state["known_tokens"] == 32_000:
            status = "cap_reached"
        else:
            raw = second.complete_stage(staged_request, batch, "original", False)
            staged_done = type(raw) is str and bool(raw)
            controls = second.requested_controls[-1]
            acknowledged = (controls.get("output_schema_sent") is True
                            and controls.get("output_schema_acknowledged") is True)
            try:
                staged.staged.parse_original_response(raw, batch)
                staged_response_valid = True
            except staged.staged.GroundingContractError:
                staged_response_valid = False
            del raw
            status = "transport_verified" if staged_done and acknowledged else "failed"
    except BaseException:
        status = "failed"
    finally:
        for client in (second, ordinary):
            if client is not None:
                try:
                    client.close()
                    cleanup_ok = cleanup_ok and client.session is None and client.directory is None
                except BaseException:
                    cleanup_ok = False
        denials, oom = _resources(receipt)
    state = budget.snapshot()
    first_failure = warm.serialize_failure(state.get("first_failure"))
    if first_failure is None and state.get("stop_code") is not None:
        first_failure = warm.serialize_failure({"code": state["stop_code"],
            "phase": "run" if state["turns"] else "preflight"})
    if not cleanup_ok:
        status = "cleanup_unverified"
    elif containment is False or denials is None or oom is None or denials or oom:
        status = "resource_unverified"
    elif (state["reserved"] != 0 or state["in_flight"] != 0
          or state["usage_complete"] is not True or state["turns"] > 2
          or state["known_tokens"] > 32_000):
        status = "accounting_unverified"
    elif status == "transport_verified" and (state["turns"] != 2
          or not ordinary_done or not staged_done or not acknowledged
          or state["known_tokens"] <= 0 or state["stopped"]):
        status = "accounting_unverified"
    return {"schema": SCHEMA, "status": status, "turns": state["turns"],
        "known_tokens": state["known_tokens"], "usage_complete": state["usage_complete"],
        "reserved": state["reserved"], "in_flight": state["in_flight"],
        "ordinary_completed": ordinary_done, "staged_completed": staged_done,
        "schema_acknowledged": acknowledged,
        "staged_response_valid": staged_response_valid,
        "client_cleanup_verified": cleanup_ok, "resource_denials": denials,
        "resource_oom": oom, "containment_verified": containment,
        "first_failure": first_failure}


def validate_result(value: Any, serializer: Any = None) -> bool:
    if type(value) is not dict or set(value) != RESULT_FIELDS or value.get("schema") != SCHEMA:
        return False
    if (type(value["status"]) is not str or value["status"] not in STATUSES
            or any(type(value[key]) is not int
            or value[key] < 0 for key in ("turns", "known_tokens", "reserved", "in_flight"))
            or value["turns"] > 2 or any(type(value[key]) is not bool for key in
            ("usage_complete", "ordinary_completed", "staged_completed",
             "schema_acknowledged", "client_cleanup_verified", "containment_verified"))
            or (value["staged_response_valid"] is not None
                and type(value["staged_response_valid"]) is not bool)
            or any(value[key] is not None and (type(value[key]) is not int or value[key] < 0)
                   for key in ("resource_denials", "resource_oom"))):
        return False
    fault = value["first_failure"]
    if fault is not None and (serializer is None or type(fault) is not dict
                              or serializer(fault) != fault):
        return False
    if (value["ordinary_completed"] and value["turns"] < 1
            or value["staged_completed"] and (not value["ordinary_completed"]
                or value["turns"] != 2)
            or value["schema_acknowledged"] and not value["staged_completed"]
            or value["staged_response_valid"] is not None and not value["staged_completed"]
            or value["turns"] == 0 and value["known_tokens"] != 0):
        return False
    if value["status"] == "transport_verified":
        return (value["turns"] == 2 and 0 < value["known_tokens"] <= 32_000
            and value["reserved"] == value["in_flight"] == 0
            and value["usage_complete"] and value["ordinary_completed"]
            and value["staged_completed"] and value["schema_acknowledged"]
            and value["client_cleanup_verified"] and value["containment_verified"]
            and value["resource_denials"] == value["resource_oom"] == 0
            and fault is None)
    return True


def inspect(root: Path, digest: str) -> dict[str, Any]:
    root = _root(root)
    launcher, loaded = verify_sources(root)
    receipt = _receipt(root, digest, launcher, attempted=True)
    runtime, recursive_cleanup = _terminal_runtime(receipt)
    path = root / "safe-access-result.json"
    result = None
    if _regular(path) and path.stat().st_size <= 4096:
        try:
            candidate = json.loads(path.read_text(encoding="ascii"))
            if validate_result(candidate, loaded["warm"].serialize_failure):
                result = candidate
        except (OSError, ValueError, UnicodeError):
            pass
    status = ("access_verified" if result is not None
        and result["status"] == "transport_verified" and runtime == "clean_exit"
        and recursive_cleanup else "access_failed" if result is not None
        and recursive_cleanup and runtime in {"clean_exit", "failed_exit"}
        else "running" if runtime == "running" else "unverified")
    return {"schema": SCHEMA, "status": status, "runtime": runtime,
        "recursive_cleanup_verified": recursive_cleanup,
        "result": result if recursive_cleanup else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-root", "launch-root", "run-root", "inspect-root"):
        actions.add_argument("--" + name)
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
            launcher, loaded = verify_sources(root)
            receipt = _receipt(root, args.receipt_sha256, launcher, attempted=True)
            _require(not (root / "safe-access-result.json").exists(), "run_already_attempted")
            output = run_access(loaded, root, receipt)
            _require(validate_result(output, loaded["warm"].serialize_failure), "result_invalid")
            _write_once(root / "safe-access-result.json", output)
        print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0 if (args.prepare_root or args.inspect_root or args.launch_root
                     or output["status"] == "transport_verified") else 1
    except BaseException:
        # Neither provider text nor local exception details leave the private root.
        print(json.dumps({"schema": SCHEMA, "status": "unverified"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
