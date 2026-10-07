"""Fresh, one-shot SIWC availability check. Inspect never imports model-capable code.

Install this file and the pinned launcher v2 beside candidate/, code/, and
source-map.json in a new private source-only root. Preparation makes no call.
"""
from __future__ import annotations

import argparse
import ast
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


SCHEMA = "siwc-lme-recovery-check-v1"
ROOT_PARENT = Path("/home/atta")
ROOT_RE = re.compile(r"\.hymem-siwc-lme-diagnostic-(?:recovery|preflight)-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
UID = 1000
SELF = "siwc_lme_recovery_check_v1.py"
LAUNCHER = "siwc_lme_diagnostic_launch_v2.py"
LAUNCHER_SHA = "fee76e6ebe40ca754d92eb8a90958db014ae39fbe59ebbad8cf5eaee86f25aa6"
RUNNER = "tools/diagnostics/siwc_lme_diagnostic_v3.py"
RUNNER_SHA = "81f055ed3c64c7df03f4e038ae24d452691d37f04df1f07d7dcad375ebc4dae0"
BRIDGE = "benchmarks/chatgpt_plan_lme_v1.py"
BRIDGE_SHA = "857aeebc2695ac2bc014643f023acd1751c5f7086c0196cd4d7a3d8489569ca5"
INVENTORY_SHA = "b87ce83ca123fb2a51f1716ca3d3d71fd3b73799a30469b34a8f033babaa90b6"
MAP_SHA = "94d7b2204ed749d1b3de29c24579aeccb2730c6aa9370e2b8ebab87a7835267e"
GRANT_SHA = "5f91fe05fb7d3b0552b7247d6a81f3fd29aae894dbcf45828b1556a0b11633dc"
RUNTIME = "/home/atta/.hymem-siwc-runtime-v1/bin/python"
RUNTIME_SHA = "17b78e0a93175e86f9ac03141924fd7a7f0c0c52e66b34bfa0de20ffef989df1"
SITE_SHA = "2bed78ec3df853e3efe5052b30d514a2765a2097e183f4b3c32a3d53ef54806d"
SITE_FILES = 309
OWNER = "/home/atta/.hymem-chatgpt-plan-lme"
CGROUP = Path("/sys/fs/cgroup")
FIELDS = ("ActiveState", "SubState", "MainPID", "ControlGroup", "NRestarts",
          "Result", "ExecMainStatus", "MemoryMax", "TasksMax", "CPUQuotaPerSecUSec",
          "KillMode", "Restart", "RemainAfterExit", "OOMPolicy", "RuntimeMaxUSec",
          "TimeoutStopUSec")
EXPECTED = "READY"
SYSTEM = "This is a fictional connectivity check. Reply with exactly READY and no other text."
USER = "Invented scenario: a blue paper kite is on an empty desk. Return the required word."


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, cap: int | None = None) -> bool:
    try:
        info = path.lstat()
        return (stat.S_ISREG(info.st_mode) and info.st_uid == UID
                and (cap is None or info.st_size <= cap))
    except OSError:
        return False


def root_checked(root: Path) -> Path:
    require(root.is_absolute() and root.parent == ROOT_PARENT
            and ROOT_RE.fullmatch(root.name) is not None, "root_invalid")
    info = root.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID
            and stat.S_IMODE(info.st_mode) == 0o700, "root_invalid")
    return root


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate_field")
        value[key] = item
    return value


def read_private(path: Path, cap: int) -> dict:
    require(regular(path, cap) and stat.S_IMODE(path.stat().st_mode) == 0o600,
            "private_file_invalid")
    value = json.loads(path.read_bytes(), object_pairs_hook=unique,
                       parse_constant=lambda _: require(False, "nonfinite"))
    require(type(value) is dict, "metadata_invalid")
    return value


def exact_private(path: Path, expected: dict, cap: int) -> None:
    read_private(path, cap)
    require(path.read_bytes() == canonical(expected), "private_marker_invalid")


def write_once(path: Path, value: dict, cap: int = 8192) -> None:
    raw = canonical(value)
    require(len(raw) <= cap, "metadata_too_large")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def identity_files(root: Path) -> dict[str, str]:
    paths = {SELF: root / SELF, LAUNCHER: root / LAUNCHER,
             RUNNER: root / "code" / RUNNER, BRIDGE: root / "code" / BRIDGE,
             "source-map.json": root / "source-map.json"}
    for name, path in paths.items():
        require(regular(path) and path.stat().st_size <= 12_000_000,
                "source_missing_or_oversized")
        if name != SELF:
            expect = {LAUNCHER: LAUNCHER_SHA, RUNNER: RUNNER_SHA,
                      BRIDGE: BRIDGE_SHA, "source-map.json": INVENTORY_SHA}[name]
            require(sha(path) == expect, "source_pin_invalid")
    require(Path(__file__).resolve() == (root / SELF).resolve(), "self_origin_invalid")
    return {name: sha(path) for name, path in paths.items()}


def source_tree_readonly(root: Path) -> None:
    """Recheck all accepted bytes on inspection without importing candidate code."""
    runner = root / "code" / RUNNER
    wanted = {"PINS", "SIWC_PINS", "CANDIDATE_PINS", "DIAGNOSTIC_HELPER_SHA256",
              "ACCEPTED_FILES", "ACCEPTED_MAP_SHA256", "RUNTIME_SHA256",
              "RUNTIME_SITE_SHA256", "RUNTIME_SITE_FILES"}
    values = {}
    for node in ast.parse(runner.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in wanted:
                values[key] = ast.literal_eval(node.value)
    require(set(values) == wanted and values["ACCEPTED_FILES"] == 514
            and values["ACCEPTED_MAP_SHA256"] == MAP_SHA
            and values["RUNTIME_SHA256"] == RUNTIME_SHA
            and values["RUNTIME_SITE_SHA256"] == SITE_SHA
            and values["RUNTIME_SITE_FILES"] == SITE_FILES,
            "runner_constants_invalid")
    inventory = read_private(root / "source-map.json", 1_000_000)
    entries = inventory.get("source_sha256")
    require(type(entries) is dict and len(entries) == 514 and all(
            type(path) is str and type(digest) is str and HEX.fullmatch(digest)
            for path, digest in entries.items()), "inventory_invalid")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    require(hashlib.sha256(encoded).hexdigest() == MAP_SHA, "candidate_map_invalid")
    candidate = root / "candidate"
    require(candidate.is_dir() and not candidate.is_symlink(), "candidate_invalid")
    actual = {}
    for path in candidate.rglob("*"):
        relative = path.relative_to(candidate)
        require(not path.is_symlink(), "candidate_symlink")
        if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts) \
                or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_dir():
            continue
        require(regular(path, 2_000_000), "candidate_file_invalid")
        actual[relative.as_posix()] = sha(path)
    require(actual == entries, "candidate_drift")
    sources = {**values["PINS"], **values["SIWC_PINS"],
               "benchmarks/lme_diagnostic.py": values["DIAGNOSTIC_HELPER_SHA256"],
               RUNNER: RUNNER_SHA}
    require(len(sources) == 25 and sources[BRIDGE] == BRIDGE_SHA,
            "code_map_invalid")
    code = root / "code"
    require(code.is_dir() and not code.is_symlink(), "code_tree_invalid")
    actual_code = set()
    for path in code.rglob("*"):
        relative = path.relative_to(code)
        require(not path.is_symlink(), "code_symlink")
        if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts) \
                or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_dir():
            continue
        require(regular(path, 2_000_000), "code_file_invalid")
        actual_code.add(relative.as_posix())
    require(actual_code == set(sources), "code_tree_drift")
    for relative, digest in sources.items():
        path = root / "code" / relative
        require(regular(path, 2_000_000) and sha(path) == digest,
                "code_drift")
    runtime = Path(RUNTIME)
    require(regular(runtime, 12_000_000) and sha(runtime) == RUNTIME_SHA,
            "runtime_drift")
    site = runtime.parent.parent / "lib/python3.13/site-packages"
    require(site.is_dir() and not site.is_symlink(), "runtime_site_invalid")
    site_files = {}
    for path in site.rglob("*"):
        require(not path.is_symlink(), "runtime_site_symlink")
        if path.suffix == ".pyc":
            continue
        require(path.is_dir() or regular(path, 11_507_864), "runtime_site_file_invalid")
        if path.is_file():
            site_files[str(path.relative_to(site))] = sha(path)
    encoded_site = json.dumps(site_files, sort_keys=True,
                              separators=(",", ":")).encode()
    require(len(site_files) == SITE_FILES
            and hashlib.sha256(encoded_site).hexdigest() == SITE_SHA,
            "runtime_site_drift")


def load_pinned(root: Path, filename: str, name: str):
    path = root / filename
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path.resolve(), "import_origin_invalid")
    return module


def source_only(root: Path):
    identity_files(root)
    runner = load_pinned(root, "code/" + RUNNER, "pinned_siwc_recovery_runner")
    loaded = runner.import_source_only(root, root / "source-map.json", INVENTORY_SHA)
    require(loaded["source_only"] is True and loaded["root"] == root
            and runner.ACCEPTED_MAP_SHA256 == MAP_SHA
            and runner.GRANT_IDENTITY_SHA256 == GRANT_SHA
            and loaded["siwc"].MAX_INVOCATION == 120.0
            and loaded["siwc"].POLICY == runner.BILLING_POLICY,
            "source_contract_invalid")
    runner.verify_owner_identity(loaded["siwc"])
    return runner, loaded


def literal_set(path: Path, name: str) -> frozenset[str]:
    """Extract a closed frozenset literal from already pinned source, no import."""
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            value = node.value
            while isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
                value = value.left
            require(isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "frozenset" and len(value.args) == 1,
                    "failure_code_source_invalid")
            items = ast.literal_eval(value.args[0])
            require(type(items) is set and all(type(item) is str for item in items),
                    "failure_code_source_invalid")
            return frozenset(items)
    raise ValueError("failure_code_source_invalid")


def failure_codes_readonly(root: Path) -> frozenset[str]:
    code = root / "code"
    bridge = literal_set(code / BRIDGE, "_CODES")
    owner = literal_set(code / "tools/diagnostics/lme_chatgpt_plan_owner_v1.py", "ERRORS")
    transport = code / "benchmarks/chatgpt_plan_responses_v1.py"
    local = literal_set(transport, "_LOCAL_CODES")
    provider = literal_set(transport, "PROVIDER_CODES")
    combined = bridge | owner | local | provider
    require(len(combined) <= 128 and "subscription_sharing_user_unavailable" in combined
            and "bridge_failure" in combined and "transport_failure" in combined,
            "failure_code_source_invalid")
    return combined


def receipt_for(root: Path, sources: dict[str, str]) -> dict:
    unit = root.name.removeprefix(".") + ".service"
    return {"schema": SCHEMA, "root": str(root), "unit": unit,
            "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
            "sources": sources, "candidate_map_sha256": MAP_SHA,
            "owner_state": OWNER, "grant_identity_sha256": GRANT_SHA,
            "runtime_path": RUNTIME, "runtime_sha256": RUNTIME_SHA,
            "runtime_site_sha256": SITE_SHA, "runtime_site_files": SITE_FILES,
            "model": "gpt-5.6-luna", "reasoning": "low",
            "auth": "siwc_oauth", "endpoint": "https://api.openai.com/v1/responses",
            "store": False, "stream": True,
            "billing_policy": "siwc_server_enforced_plan_or_existing_credits_v1",
            "limits": {"turns": 1, "known_tokens": 160000,
                       "campaign_seconds": 180, "invocation_seconds": 120,
                       "service_seconds": 300, "stop_seconds": 10,
                       "tasks": 256, "memory_bytes": 4294967296, "cpu_percent": 200},
            "expected_output_sha256": hashlib.sha256(EXPECTED.encode()).hexdigest(),
            "one_shot": True}


def receipt_checked(root: Path, digest: str) -> dict:
    require(type(digest) is str and HEX.fullmatch(digest) is not None,
            "receipt_digest_invalid")
    path = root / "recovery-receipt.json"
    require(regular(path, 8192) and sha(path) == digest, "receipt_pin_invalid")
    value = read_private(path, 8192)
    expected = receipt_for(root, identity_files(root))
    require(path.read_bytes() == canonical(expected), "receipt_identity_invalid")
    source_tree_readonly(root)
    return value


def prepare(root: Path) -> dict:
    root_checked(root)
    require({item.name for item in root.iterdir()} ==
            {"candidate", "code", "source-map.json", SELF, LAUNCHER},
            "root_not_fresh")
    launcher = load_pinned_after_identity(root)
    launcher.host_admission()
    source_only(root)
    receipt = receipt_for(root, identity_files(root))
    (root / "empty").mkdir(mode=0o700)
    (root / "tmp").mkdir(mode=0o700)
    write_once(root / "recovery-receipt.json", receipt)
    return {"schema": SCHEMA, "prepared": True, "model_calls": 0,
            "root": str(root), "unit": receipt["unit"],
            "receipt_sha256": sha(root / "recovery-receipt.json")}


def load_pinned_after_identity(root: Path):
    identity_files(root)
    return load_pinned(root, LAUNCHER, "pinned_siwc_recovery_launcher")


def bus_env() -> dict[str, str]:
    runtime = Path("/run/user/1000")
    info, bus = runtime.lstat(), (runtime / "bus").lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID
            and not info.st_mode & 0o077 and stat.S_ISSOCK(bus.st_mode)
            and bus.st_uid == UID, "user_bus_invalid")
    return {"HOME": str(ROOT_PARENT), "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": str(runtime),
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(runtime / "bus")}


def service_fields(unit: str) -> dict[str, str]:
    done = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
        "--property=" + ",".join(FIELDS), "--no-pager"], capture_output=True,
        text=True, timeout=10, check=True, env=bus_env())
    require(len(done.stdout) <= 8192, "service_metadata_oversized")
    pairs = [line.split("=", 1) for line in done.stdout.splitlines()]
    require(len(pairs) == len(FIELDS) and all(len(pair) == 2 for pair in pairs),
            "service_metadata_invalid")
    values = unique(pairs)
    require(set(values) == set(FIELDS), "service_metadata_invalid")
    return values


def group_for(receipt: dict) -> Path:
    group = CGROUP / receipt["expected_cgroup"].lstrip("/")
    require(not CGROUP.is_symlink() and group.resolve().is_relative_to(CGROUP.resolve(strict=True)),
            "cgroup_path_invalid")
    node = group
    while node != CGROUP:
        require(not node.is_symlink(), "cgroup_path_invalid")
        node = node.parent
    return group


def group_policy(group: Path) -> None:
    require((group / "memory.max").read_text().strip() == "4294967296"
            and (group / "pids.max").read_text().strip() == "256", "cgroup_policy_invalid")
    cpu = (group / "cpu.max").read_text().split()
    require(len(cpu) == 2 and all(x.isdecimal() for x in cpu)
            and int(cpu[1]) > 0 and int(cpu[0]) == 2 * int(cpu[1]), "cgroup_policy_invalid")


def recursive_empty(group: Path) -> bool:
    if not group.exists():
        return True
    try:
        pending, count = [group], 0
        while pending:
            node = pending.pop()
            count += 1
            if count > 256 or node.is_symlink() or not node.is_dir():
                return False
            if any((node / name).read_text().strip() for name in ("cgroup.procs", "cgroup.threads")):
                return False
            pairs = [line.split() for line in (node / "cgroup.events").read_text().splitlines()]
            if len(pairs) > 32 or any(len(pair) != 2 for pair in pairs):
                return False
            events = unique(pairs)
            if events.get("populated") != "0":
                return False
            for child in node.iterdir():
                if child.is_symlink():
                    return False
                if child.is_dir():
                    pending.append(child)
        return True
    except (OSError, ValueError):
        return False


def service_state(receipt: dict) -> tuple[str, bool, bool]:
    values = service_fields(receipt["unit"])
    require(values["NRestarts"] == "0" and values["MemoryMax"] == "4294967296"
            and values["TasksMax"] == "256"
            and values["CPUQuotaPerSecUSec"] in {"2s", "2.000s"}
            and values["KillMode"] == "control-group" and values["Restart"] == "no"
            and values["RemainAfterExit"] == "yes" and values["OOMPolicy"] == "kill"
            and values["RuntimeMaxUSec"] in {"5min", "5min 0s", "300s", "300.000s"}
            and values["TimeoutStopUSec"] in {"10s", "10.000s"}, "service_policy_invalid")
    group = group_for(receipt)
    if values["ActiveState"] == "active" and values["SubState"] == "running":
        require(values["ControlGroup"] == receipt["expected_cgroup"]
                and values["MainPID"].isdecimal() and int(values["MainPID"]) > 0,
                "running_identity_invalid")
        group_policy(group)
        require(values["MainPID"] in (group / "cgroup.procs").read_text().splitlines(),
                "running_pid_invalid")
        return "running", False, False
    require((values["ActiveState"], values["SubState"]) in
            {("active", "exited"), ("inactive", "dead"), ("failed", "failed")}
            and values["MainPID"] == "0"
            and values["ControlGroup"] in {"", receipt["expected_cgroup"]},
            "terminal_service_invalid")
    if group.exists():
        group_policy(group)
    require(recursive_empty(group), "descendants_remain")
    clean_exit = values["Result"] == "success" and values["ExecMainStatus"] == "0"
    require(clean_exit or (values["Result"] in
            {"exit-code", "signal", "core-dump", "oom-kill", "timeout"}
            and values["ExecMainStatus"].isdecimal()), "terminal_result_invalid")
    return "terminal", True, clean_exit


def execute_containment(receipt: dict) -> Path:
    state, _, _ = service_state(receipt)
    require(state == "running", "service_not_running")
    values = service_fields(receipt["unit"])
    require(values["MainPID"] == str(os.getpid())
            and f"0::{receipt['expected_cgroup']}" in Path("/proc/self/cgroup").read_text().splitlines(),
            "process_containment_invalid")
    return group_for(receipt)


def resource(group: Path) -> dict[str, int]:
    group_policy(group)
    values = {}
    for key, filename in (("peak", "pids.peak"), ("denials", "pids.events")):
        if key == "peak":
            raw = (group / filename).read_text().strip()
        else:
            pairs = [line.split() for line in (group / filename).read_text().splitlines()]
            require(len(pairs) <= 32 and all(len(pair) == 2 for pair in pairs),
                    "resource_invalid")
            raw = unique(pairs).get("max", "")
        require(raw.isdecimal(), "resource_invalid")
        values[key] = int(raw)
    require(values["peak"] <= 256, "resource_peak_invalid")
    pairs = [line.split() for line in (group / "memory.events").read_text().splitlines()]
    require(len(pairs) <= 32 and all(len(pair) == 2 for pair in pairs),
            "memory_events_invalid")
    events = unique(pairs)
    for key in ("oom", "oom_kill", "oom_group_kill"):
        raw = events.get(key, "")
        require(raw.isdecimal(), "memory_events_invalid")
        values[key] = int(raw)
    require(values.pop("oom_group_kill") == 0, "memory_group_oom_fault")
    return values


def launch(root: Path, digest: str) -> dict:
    root_checked(root)
    receipt = receipt_checked(root, digest)
    launcher = load_pinned_after_identity(root)
    launcher.host_admission()
    source_only(root)
    require(not any((root / name).exists() or (root / name).is_symlink() for name in
            ("recovery-attempt.json", "recovery-execution.json", "recovery-result.json",
             "recovery-terminal.json", "private-recovery-stderr.log", "recovery-dispatch.json")),
            "already_attempted")
    for name in ("empty", "tmp"):
        path = root / name
        require(path.is_dir() and not path.is_symlink() and not any(path.iterdir()),
                "workdir_not_empty")
    for name in ("recovery-terminal.json", "private-recovery-stderr.log"):
        fd = os.open(root / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)
    write_once(root / "recovery-attempt.json",
               {"receipt_sha256": digest, "one_shot": True}, 512)
    cmd = ["/usr/bin/systemd-run", "--user", "--quiet", "--unit", receipt["unit"],
           "--property=Type=exec", "--property=Restart=no",
           "--property=KillMode=control-group", "--property=RemainAfterExit=yes",
           "--property=RuntimeMaxSec=300s", "--property=TimeoutStopSec=10s",
           "--property=MemoryMax=4294967296", "--property=CPUQuota=200%",
           "--property=TasksMax=256", "--property=OOMPolicy=kill",
           "--property=UMask=0077", "--property=WorkingDirectory=" + str(root / "empty"),
           "--property=StandardOutput=file:" + str(root / "recovery-terminal.json"),
           "--property=StandardError=file:" + str(root / "private-recovery-stderr.log"),
           "/usr/bin/env", "-i", "HOME=/home/atta", "PATH=/usr/local/bin:/usr/bin:/bin",
           "XDG_RUNTIME_DIR=/run/user/1000",
           "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
           "TMPDIR=" + str(root / "tmp"), RUNTIME, "-I", "-B", str(root / SELF),
           "--execute-root", str(root), "--receipt-sha256", digest]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=20, check=False,
                              env=bus_env())
        code = done.returncode
    except (OSError, subprocess.SubprocessError):
        code = -1
    write_once(root / "recovery-dispatch.json", {"returncode": code}, 512)
    return {"schema": SCHEMA, "launch_command_returncode": code,
            "root": str(root), "unit": receipt["unit"],
            "receipt_sha256": digest, "never_retry": True}


def execute(root: Path, digest: str) -> int:
    root_checked(root)
    receipt = receipt_checked(root, digest)
    exact_private(root / "recovery-attempt.json",
                  {"receipt_sha256": digest, "one_shot": True}, 512)
    group = execute_containment(receipt)
    write_once(root / "recovery-execution.json",
               {"receipt_sha256": digest, "execution_started": True}, 512)
    _, loaded = source_only(root)
    siwc, warm = loaded["siwc"], loaded["warm"]
    allowed = failure_codes_readonly(root)
    require(frozenset(siwc._CODES) == allowed, "failure_code_binding_invalid")
    before = resource(group)
    require(before["peak"] > 0 and before["denials"] == 0
            and all(before[key] == 0 for key in ("oom", "oom_kill")),
            "precall_resource_fault")
    budget = siwc.SharedBudget(warm.BudgetLimits(1, 160000, 180), max_in_flight=1)
    broker = siwc.owner.CredentialBroker(Path(OWNER), Path(RUNTIME))
    client = None
    response_ok = False
    try:
        require(broker.identity_digest == GRANT_SHA, "grant_identity_invalid")
        client = siwc.SIWCLMEClient(broker, budget, "recovery",
                                    warm.BudgetLimits(1, 160000, 180))
        request = loaded["request_type"](system=SYSTEM, user=USER,
                                          response_format="text", max_tokens=64,
                                          temperature=0.0)
        try:
            response = client.complete(request)
            response_ok = type(response) is str and response == EXPECTED
        except BaseException:
            pass
    finally:
        if client is not None:
            client.close()
        broker.close()
    snapshot = budget.snapshot()
    summary = client.diagnostic_summary() if client is not None else None
    evidence = resource(group)
    result = {"schema": SCHEMA, "receipt_sha256": digest,
              "response_match": response_ok, "calls": summary["calls"] if summary else 0,
              "successes": summary["successes"] if summary else 0,
              "failures": summary["failures"] if summary else 0,
              "http_attempts": summary["internal_http_attempts"] if summary else 0,
              "admitted_turns": snapshot["turns"],
              "known_tokens": snapshot["known_tokens"],
              "usage_complete": snapshot["usage_complete"],
              "in_flight": snapshot["in_flight"], "reserved": snapshot["reserved"],
              "first_failure": project_failure(summary["first_failure"], allowed) if summary else None,
              "stop_code": snapshot["stop_code"], "resource": evidence}
    result["check_passed"] = bool(response_ok and result["calls"] == 1
        and result["successes"] == 1 and result["failures"] == 0
        and result["http_attempts"] == 1 and result["admitted_turns"] == 1
        and 0 < result["known_tokens"] <= 160000 and result["usage_complete"]
        and result["in_flight"] == result["reserved"] == 0
        and result["first_failure"] is None and result["stop_code"] is None
        and 0 < evidence["peak"] <= 256 and evidence["denials"] == 0
        and all(evidence[key] == 0 for key in ("oom", "oom_kill")))
    write_once(root / "recovery-result.json", result, 4096)
    print(canonical({"schema": SCHEMA, "execution_finished": True}).decode())
    return 0 if result["check_passed"] else 1


def safe_failure(value: Any, allowed: frozenset[str]) -> str | None:
    if value is None:
        return None
    require(type(value) is dict and type(value.get("code")) is str
            and value["code"] in allowed
            and set(value) == {"code", "phase", "turn_admitted", "unknown_usage",
                               "http_status"}
            and value["phase"] in {"admission", "http"}
            and type(value["turn_admitted"]) is bool
            and type(value["unknown_usage"]) is bool
            and value["turn_admitted"] is (value["phase"] == "http")
            and value["unknown_usage"] is value["turn_admitted"]
            and (value["http_status"] is None or type(value["http_status"]) is int
                 and 400 <= value["http_status"] <= 599),
            "failure_metadata_invalid")
    return value["code"]


def project_failure(value: Any, allowed: frozenset[str]) -> dict | None:
    if value is None:
        return None
    projected = {"code": value.get("code"), "phase": value.get("phase"),
                 "turn_admitted": value.get("turn_admitted"),
                 "unknown_usage": value.get("unknown_usage"),
                 "http_status": value.get("http_status")}
    safe_failure(projected, allowed)
    return projected


def inspect(root: Path, digest: str) -> dict:
    root_checked(root)
    receipt = receipt_checked(root, digest)
    allowed = failure_codes_readonly(root)
    output = {"schema": SCHEMA, "root": str(root), "unit": receipt["unit"],
              "receipt_sha256": digest, "status": "prepared_not_launched",
              "recovery_verified": False, "runtime_cleanup_verified": False,
              "admitted_turns": None, "known_tokens": None,
              "usage_complete": None, "first_failure_code": None,
              "failed_turn_usage_unknown": None}
    attempt = root / "recovery-attempt.json"
    if not attempt.exists() and not attempt.is_symlink():
        require(not any((root / name).exists() for name in
                ("recovery-execution.json", "recovery-result.json", "recovery-dispatch.json")),
                "unlaunched_state_invalid")
        return output
    exact_private(attempt, {"receipt_sha256": digest, "one_shot": True}, 512)
    marker = root / "recovery-execution.json"
    started = marker.exists() or marker.is_symlink()
    if started:
        exact_private(marker,
                      {"receipt_sha256": digest, "execution_started": True}, 512)
    state, cleaned, clean_exit = service_state(receipt)
    output["runtime_cleanup_verified"] = cleaned
    output["status"] = "running" if state == "running" else "terminal_without_result"
    result_path = root / "recovery-result.json"
    if not result_path.exists() and not result_path.is_symlink():
        return output
    require(started, "result_without_execution")
    result = read_private(result_path, 4096)
    require(set(result) == {"schema", "receipt_sha256", "response_match", "calls",
            "successes", "failures", "http_attempts", "admitted_turns",
            "known_tokens", "usage_complete", "in_flight", "reserved",
            "first_failure", "stop_code", "resource", "check_passed"}
            and result["schema"] == SCHEMA and result["receipt_sha256"] == digest,
            "result_schema_invalid")
    for key in ("calls", "successes", "failures", "http_attempts", "admitted_turns",
                "known_tokens", "in_flight", "reserved"):
        require(type(result[key]) is int and 0 <= result[key] <=
                (1_000_000_000 if key == "known_tokens" else 1), "result_count_invalid")
    require(all(type(result[key]) is bool for key in
                ("response_match", "usage_complete", "check_passed"))
            and result["successes"] + result["failures"] == result["calls"]
            and result["http_attempts"] <= result["admitted_turns"]
            and result["admitted_turns"] <= result["calls"]
            and result["in_flight"] == result["reserved"] == 0,
            "result_accounting_invalid")
    code = safe_failure(result["first_failure"], allowed)
    require(result["stop_code"] is None or
            (type(result["stop_code"]) is str and result["stop_code"] in allowed),
            "stop_code_invalid")
    require(result["calls"] == 1 and (
            (result["failures"] == 0 and result["successes"] == 1
             and code is None and result["stop_code"] is None
             and result["admitted_turns"] == result["http_attempts"] == 1
             and result["usage_complete"] and result["known_tokens"] > 0)
            or (result["failures"] == 1 and result["successes"] == 0
                and code is not None and result["stop_code"] == code
                and not result["response_match"]
                and result["admitted_turns"] == result["http_attempts"]
                == int(result["first_failure"]["turn_admitted"])
                and result["known_tokens"] == 0
                and result["usage_complete"] is
                    (not result["first_failure"]["turn_admitted"]))),
            "result_failure_consistency_invalid")
    resource_value = result["resource"]
    require(type(resource_value) is dict and set(resource_value) ==
            {"peak", "denials", "oom", "oom_kill"}
            and type(resource_value["peak"]) is int and 0 <= resource_value["peak"] <= 256
            and all(type(resource_value[key]) is int and 0 <= resource_value[key] <= 1_000_000_000
                    for key in ("denials", "oom", "oom_kill")),
            "resource_invalid")
    passed = bool(result["response_match"] and result["calls"] == 1
        and result["successes"] == 1 and result["failures"] == 0
        and result["http_attempts"] == 1 and result["admitted_turns"] == 1
        and 0 < result["known_tokens"] <= 160000 and result["usage_complete"]
        and code is None and result["stop_code"] is None
        and 0 < resource_value["peak"] <= 256 and resource_value["denials"] == 0
        and all(resource_value[key] == 0 for key in ("oom", "oom_kill")))
    require(result["check_passed"] is passed, "pass_claim_invalid")
    output.update(status="recovered_and_clean" if passed and cleaned and clean_exit
                  else "terminal_check_failed" if cleaned else "running_result_unverified",
                  recovery_verified=passed and cleaned and clean_exit,
                  admitted_turns=result["admitted_turns"],
                  known_tokens=result["known_tokens"],
                  usage_complete=result["usage_complete"], first_failure_code=code)
    output["failed_turn_usage_unknown"] = (result["first_failure"] is not None
        and result["first_failure"]["turn_admitted"]
        and result["first_failure"]["unknown_usage"])
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-root", "launch-root", "execute-root", "inspect-root"):
        action.add_argument("--" + name)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    stage = "prepare" if args.prepare_root else "launch" if args.launch_root else \
            "execute" if args.execute_root else "inspect"
    root = Path(args.prepare_root or args.launch_root or args.execute_root or args.inspect_root)
    try:
        if stage == "prepare":
            require(args.receipt_sha256 is None, "unexpected_receipt")
            value = prepare(root)
            code = 0
        else:
            require(args.receipt_sha256 is not None, "receipt_required")
            if stage == "launch":
                value = launch(root, args.receipt_sha256)
                code = 0 if value["launch_command_returncode"] == 0 else 1
            elif stage == "execute":
                return execute(root, args.receipt_sha256)
            else:
                value = inspect(root, args.receipt_sha256)
                code = 0 if value["status"] != "unverified" else 1
        print(json.dumps(value, sort_keys=True))
        return code
    except BaseException:
        print(json.dumps({"schema": SCHEMA, "status": "unverified", "stage": stage,
            "never_retry_launch": stage in {"launch", "execute"}}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
