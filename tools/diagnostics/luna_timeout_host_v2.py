"""Host-side boundary for the accepted, bounded Luna timeout diagnostic.

The launcher creates the private root, immutable receipt and launch-attempt
marker. This file never dispatches a service. Its two entry points are invoked
inside separately named systemd user services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any


SCHEMA = "luna-timeout-host-v2"
HOST_HOME = Path("/home/atta")
HOST_UID = 1000
ROOT_PATTERN = re.compile(r"\.hymem-luna-timeout-v2-[a-z0-9_]{8}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
PREPARATION_SHA256 = "e1291c0e4b3e533e6165a1f12d9c5a2b4947be8341cab7f7a3a2ff5716173d21"
PROBE_RELATIVE = "tools/diagnostics/luna_timeout_probe_v2.py"
PROBE_SHA256 = "99fb78d584e7fe8ccd1fa7f7eefb6969a29f4604b23228ecc0054b95cc6b1a6c"
OBSERVER_SHA256 = "476e00bcae40c0a061a1b75b10797d4563fb9822012917382e27d9b5a6c93287"
BINARY = "/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex"
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
CGROUP_ROOT = Path("/sys/fs/cgroup")
RUNTIME_DIR = Path("/run/user/1000")
UNIT_PREFIX = "hymem-luna-timeout-v2-"
POLICY = {"tasks_max": 256, "memory_max": 4_294_967_296,
          "cpu_percent": 200, "runtime_seconds": 730, "stop_seconds": 10,
          "restart": "no", "kill_mode": "control-group", "oom_policy": "kill",
          "umask": "0077"}
FIELDS = ("ActiveState", "SubState", "MainPID", "ControlGroup", "NRestarts",
          "Type", "MemoryMax", "TasksMax", "CPUQuotaPerSecUSec", "KillMode",
          "Restart", "RemainAfterExit", "OOMPolicy", "RuntimeMaxUSec", "TimeoutStopUSec", "UMask",
          "Result", "ExecMainStatus")


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _root(root: Path) -> Path:
    _need(root.is_absolute() and root.parent == HOST_HOME and
          ROOT_PATTERN.fullmatch(root.name) is not None and
          root.is_dir() and not root.is_symlink(), "root_invalid")
    info = root.stat()
    _need(info.st_uid == HOST_UID and stat.S_IMODE(info.st_mode) == 0o700,
          "root_permissions_invalid")
    return root


def unit_for(root: Path, mode: str) -> str:
    _root(root)
    _need(mode in {"probe", "containment"}, "mode_invalid")
    suffix = root.name.removeprefix(".hymem-luna-timeout-v2-")
    return UNIT_PREFIX + mode + "-" + suffix + ".service"


def receipt_for(root: Path, host_sha256: str, mode: str) -> dict[str, Any]:
    _need(type(host_sha256) is str and HEX.fullmatch(host_sha256) is not None,
          "host_hash_invalid")
    unit = unit_for(root, mode)
    return {"schema": SCHEMA, "root": str(root), "uid": HOST_UID,
            "mode": mode, "unit": unit,
            "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
            "host_sha256": host_sha256, "probe_sha256": PROBE_SHA256,
            "observer_sha256": OBSERVER_SHA256,
            "preparation_sha256": PREPARATION_SHA256,
            "binary_path": BINARY, "binary_sha256": BINARY_SHA256,
            "policy": dict(POLICY), "turns": 16, "workers": 4,
            "known_tokens": 160_000, "campaign_seconds": 600,
            "invocation_seconds": 120, "one_shot": True}


def verify_receipt(root: Path, digest: str, mode: str) -> dict[str, Any]:
    _root(root)
    _need(type(digest) is str and HEX.fullmatch(digest) is not None,
          "receipt_hash_invalid")
    host_file = root / "timeout-host-v2.py"
    _need(Path(__file__).absolute() == host_file.absolute() and
          _regular(host_file) and host_file.stat().st_uid == HOST_UID and
          stat.S_IMODE(host_file.stat().st_mode) == 0o600,
          "host_origin_invalid")
    path = root / ("launch-receipt.json" if mode == "probe" else
                   "containment-receipt.json")
    _need(_regular(path) and path.stat().st_uid == HOST_UID and
          stat.S_IMODE(path.stat().st_mode) == 0o600 and
          path.stat().st_size <= 8192 and _sha(path) == digest,
          "receipt_drift")
    expected = receipt_for(root, _sha(host_file), mode)
    _need(path.read_bytes() == _json_bytes(expected), "receipt_invalid")
    return expected


def verify_sources(root: Path) -> tuple[Any, Any]:
    """Verify accepted bundle and execute exactly the bytes that were hashed."""
    _root(root)
    bundle = root / "bundle"
    _need(bundle.is_dir() and not bundle.is_symlink() and
          bundle.stat().st_uid == HOST_UID and
          stat.S_IMODE(bundle.stat().st_mode) == 0o700,
          "bundle_invalid")
    prep = bundle / "local-preparation-receipt.json"
    _need(_regular(prep) and _sha(prep) == PREPARATION_SHA256,
          "preparation_drift")
    path = bundle / "code" / PROBE_RELATIVE
    _need(_regular(path), "probe_missing")
    source = path.read_bytes()
    _need(hashlib.sha256(source).hexdigest() == PROBE_SHA256,
          "probe_drift")
    _need(not any(name == "hymem" or name.startswith("hymem.") or
                  name == "benchmarks" or name.startswith("benchmarks.")
                  for name in sys.modules), "ambient_module_present")
    module = types.ModuleType("pinned_luna_timeout_probe")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
        _need(module.SOURCE_PINS["benchmarks/codex_subscription_timeout_v2.py"] ==
              OBSERVER_SHA256 and module.BINARY_PATH == BINARY and
              module.BINARY_SHA256 == BINARY_SHA256 and
              module.LIMITS == {"turns": 16, "known_tokens": 160_000,
                                "seconds": 600, "workers": 4,
                                "calls_per_worker": 4,
                                "invocation_seconds": 120},
              "probe_policy_drift")
        return module, module.load_prepared(bundle, PREPARATION_SHA256)
    except BaseException:
        sys.modules.pop(module.__name__, None)
        raise


def _unit_values(unit: str) -> dict[str, str]:
    env = {"HOME": str(HOST_HOME), "PATH": "/usr/bin:/bin",
           "XDG_RUNTIME_DIR": str(RUNTIME_DIR),
           "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(RUNTIME_DIR / "bus")}
    completed = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
        "--property=" + ",".join(FIELDS), "--no-pager"], capture_output=True,
        text=True, timeout=10, check=True, env=env)
    _need(len(completed.stdout) <= 8192, "unit_report_invalid")
    pairs = [line.split("=", 1) for line in completed.stdout.splitlines()]
    _need(len(pairs) == len(FIELDS) and all(len(pair) == 2 for pair in pairs),
          "unit_report_invalid")
    values = dict(pairs)
    _need(set(values) == set(FIELDS), "unit_report_invalid")
    return values


def _seconds(value: str) -> int | None:
    # systemd show uses either a short duration or a mixed-unit human duration.
    if value.endswith("s") and value[:-1].isdecimal():
        return int(value[:-1])
    total = 0
    for part in value.split():
        match = re.fullmatch(r"([0-9]+)(?:\.0+)?(min|s)", part)
        if match is None:
            return None
        total += int(match[1]) * (60 if match[2] == "min" else 1)
    return total if value else None


def _policy_ok(values: dict[str, str]) -> bool:
    try:
        quota = values["CPUQuotaPerSecUSec"]
        cpu_ok = quota in {"2s", "2.000s", "2000000"}
        return (values["NRestarts"] == "0" and values["Type"] == "exec"
            and values["MemoryMax"] == str(POLICY["memory_max"])
            and values["TasksMax"] == str(POLICY["tasks_max"])
            and cpu_ok and values["KillMode"] == "control-group"
            and values["Restart"] == "no" and values["RemainAfterExit"] == "yes"
            and values["OOMPolicy"] == "kill"
            and _seconds(values["RuntimeMaxUSec"]) == 730
            and _seconds(values["TimeoutStopUSec"]) == 10
            and values["UMask"] in {"0077", "0o077"})
    except (KeyError, TypeError):
        return False


def _group(receipt: dict[str, Any]) -> Path:
    expected = receipt["expected_cgroup"]
    _need(type(expected) is str and expected.startswith("/user.slice/") and
          ".." not in expected.split("/"), "group_invalid")
    path = CGROUP_ROOT / expected.lstrip("/")
    _need(path.resolve().is_relative_to(CGROUP_ROOT), "group_invalid")
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


def _counter(path: Path, name: str) -> int:
    lines = path.read_text().splitlines()
    _need(len(lines) <= 32, "counter_invalid")
    pairs = [line.split() for line in lines]
    _need(all(len(pair) == 2 for pair in pairs), "counter_invalid")
    values = dict(pairs)
    _need(len(values) == len(pairs), "counter_invalid")
    value = values[name]
    _need(value.isdecimal() and len(value) <= 19 and int(value) <= 2**63 - 1,
          "counter_invalid")
    return int(value)


def _scalar(path: Path) -> int:
    value = path.read_text().strip()
    _need(value.isdecimal() and len(value) <= 19 and int(value) <= 2**63 - 1,
          "counter_invalid")
    return int(value)


def resources(receipt: dict[str, Any]) -> dict[str, int]:
    group = _group(receipt)
    return {"pids_denials": _counter(group / "pids.events", "max"),
            "memory_oom": _counter(group / "memory.events", "oom"),
            "memory_oom_kill": _counter(group / "memory.events", "oom_kill"),
            "pids_current": _scalar(group / "pids.current"),
            "memory_current": _scalar(group / "memory.current"),
            "memory_peak": _scalar(group / "memory.peak")}


def live_attestation(receipt: dict[str, Any], phase: str,
                     index: int | None) -> tuple[dict[str, Any], dict[str, int] | None]:
    _need(phase in {"initial", "before_admission", "terminal"} and
          (index is None if phase != "before_admission" else
           type(index) is int and 0 <= index < 16), "phase_invalid")
    try:
        values = _unit_values(receipt["unit"])
        group = _group(receipt)
        pid = os.getpid()
        contained = (_policy_ok(values) and values["ActiveState"] == "active"
            and values["SubState"] == "running" and values["MainPID"] == str(pid)
            and values["ControlGroup"] == receipt["expected_cgroup"]
            and group.is_dir() and _group_policy(group)
            and (group / "cgroup.procs").read_text().splitlines().count(str(pid)) == 1
            and Path("/proc/self/cgroup").read_text().splitlines() ==
                ["0::" + receipt["expected_cgroup"]])
        counters = resources(receipt) if contained else None
        return {"containment": bool(contained),
                "denials": counters["pids_denials"] if counters else None,
                "oom": max(counters["memory_oom"], counters["memory_oom_kill"])
                    if counters else None}, counters
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {"containment": False, "denials": None, "oom": None}, None


def recursive_empty(group: Path) -> bool:
    """A vanished group is clean; every extant descendant must be empty."""
    try:
        if group.is_symlink():
            return False
        if not group.exists():
            return True
        _need(group.is_dir(), "group_invalid")
        pending = [group]
        visited = 0
        while pending:
            node = pending.pop()
            visited += 1
            _need(visited <= 256, "group_tree_too_large")
            _need((node / "cgroup.procs").read_text().strip() == "" and
                  (node / "cgroup.threads").read_text().strip() == "" and
                  _counter(node / "cgroup.events", "populated") == 0,
                  "group_populated")
            for child in node.iterdir():
                _need(not child.is_symlink(), "group_symlink")
                if child.is_dir():
                    pending.append(child)
        return True
    except (OSError, ValueError, KeyError):
        return False


def terminal_runtime(receipt: dict[str, Any]) -> dict[str, Any]:
    """Independent post-exit policy/cleanup check, separate from probe result."""
    try:
        values = _unit_values(receipt["unit"])
        policy = _policy_ok(values)
        group_match = values["ControlGroup"] in {"", receipt["expected_cgroup"]}
        stopped = ((values["ActiveState"], values["SubState"]) in
            {("inactive", "dead"), ("failed", "failed"), ("active", "exited")}
            and values["MainPID"] == "0")
        group = _group(receipt)
        empty = (recursive_empty(group) and
                 (not group.exists() or _group_policy(group))) if group_match else False
        result = ("success" if values["Result"] == "success" and
                  values["ExecMainStatus"] == "0" else "failure")
        return {"policy_verified": policy, "unit_stopped": stopped,
                "group_matched": group_match,
                "recursive_cleanup_verified": bool(policy and stopped and group_match and empty),
                "runtime_exit": result if stopped else "unknown"}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {"policy_verified": False, "unit_stopped": False,
                "group_matched": False, "recursive_cleanup_verified": False,
                "runtime_exit": "unknown"}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")
    _need(len(data) <= 1_000_000, "result_too_large")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _marker(root: Path, digest: str, mode: str) -> None:
    attempt = root / ("launch-attempt.json" if mode == "probe" else
                      "containment-attempt.json")
    _need(_regular(attempt) and attempt.stat().st_uid == HOST_UID and
          stat.S_IMODE(attempt.stat().st_mode) == 0o600 and
          attempt.stat().st_size <= 512, "attempt_missing")
    _need(attempt.read_bytes() == _json_bytes(
          {"receipt_sha256": digest, "one_shot": True}), "attempt_invalid")
    lock = root / ("probe-execution-marker.json" if mode == "probe" else
                   "containment-execution-marker.json")
    _write_once(lock, {"receipt_sha256": digest, "execution_started": True})


def _base(root: Path, digest: str, mode: str) -> dict[str, Any]:
    _need(sys.platform == "linux" and os.getuid() == HOST_UID and
          os.geteuid() == HOST_UID, "host_user_invalid")
    receipt = verify_receipt(root, digest, mode)
    _marker(root, digest, mode)
    return receipt


def run_containment_only(root: Path, digest: str) -> dict[str, Any]:
    """No source import, client construction, App Server or model request."""
    receipt = _base(root, digest, "containment")
    attested, counters = live_attestation(receipt, "initial", None)
    verified = attested == {"containment": True, "denials": 0, "oom": 0}
    result = {"schema": SCHEMA, "mode": "containment", "verified": verified,
              "model_calls": 0, "resource_counters": counters,
              "independent_recursive_cleanup_verified": None}
    _need(validate_host_result(result), "host_result_invalid")
    _write_once(root / "containment-result.json", result)
    return result


def run_once(root: Path, digest: str) -> dict[str, Any]:
    receipt = _base(root, digest, "probe")
    samples: list[dict[str, Any]] = []

    def attest(phase: str, index: int | None) -> dict[str, Any]:
        gate, counters = live_attestation(receipt, phase, index)
        samples.append({"phase": phase, "index": index, "gate": gate,
                        "resources": counters})
        return gate

    probe_result = None
    status = "incomplete_or_failed"
    failure_code = "initial_containment_invalid"
    try:
        # Initial containment and pinned binary precede any source execution.
        first = attest("initial", None)
        _need(first == {"containment": True, "denials": 0, "oom": 0},
              "initial_containment_invalid")
        failure_code = "binary_drift"
        _need(_regular(Path(BINARY)) and _sha(Path(BINARY)) == BINARY_SHA256,
              "binary_drift")
        failure_code = "source_invalid"
        probe, (observer, request_type) = verify_sources(root)
        failure_code = "probe_exception"
        candidate = probe.run_probe(observer, request_type, BINARY, attest=attest)
        failure_code = "probe_result_invalid"
        _need(probe.validate_result(candidate, observer), "probe_result_invalid")
        probe_result = candidate
        status = probe_result["status"]
        failure_code = None
    except BaseException:
        status = "incomplete_or_failed"
    if not any(item["phase"] == "terminal" for item in samples):
        attest("terminal", None)
    result = {"schema": SCHEMA, "mode": "probe", "status": status,
              "failure_code": failure_code, "probe_result": probe_result,
              "samples": samples,
              "independent_recursive_cleanup_verified": None}
    _need(validate_host_result(result, probe=locals().get("probe"),
                               observer=locals().get("observer")),
          "host_result_invalid")
    _write_once(root / "probe-result.json", result)
    return result


def validate_host_result(value: Any, *, probe: Any = None,
                         observer: Any = None) -> bool:
    """Finite host projection; a probe result needs the pinned validator."""
    try:
        if (type(value) is not dict or value.get("schema") != SCHEMA or
                value.get("independent_recursive_cleanup_verified") is not None):
            return False
        if value.get("mode") == "containment":
            if (set(value) != {"schema", "mode", "verified", "model_calls",
                              "resource_counters",
                              "independent_recursive_cleanup_verified"} or
                    type(value["verified"]) is not bool or
                    type(value["model_calls"]) is not int or value["model_calls"] != 0):
                return False
            counters = value["resource_counters"]
            return (_valid_counters(counters) or counters is None) and (
                not value["verified"] or counters is not None and
                all(counters[key] == 0 for key in
                    ("pids_denials", "memory_oom", "memory_oom_kill")))
        if (value.get("mode") != "probe" or set(value) != {
                "schema", "mode", "status", "failure_code", "probe_result",
                "samples", "independent_recursive_cleanup_verified"}):
            return False
        if (value["status"] not in {"observed_success", "incomplete_or_failed"} or
                value["failure_code"] not in {None, "initial_containment_invalid",
                    "binary_drift", "source_invalid", "probe_exception",
                    "probe_result_invalid"}):
            return False
        samples = value["samples"]
        if type(samples) is not list or len(samples) > 19:
            return False
        for item in samples:
            if (type(item) is not dict or
                    set(item) != {"phase", "index", "gate", "resources"}):
                return False
            phase, index, gate, counters = (item[key] for key in
                ("phase", "index", "gate", "resources"))
            if (phase not in {"initial", "before_admission", "terminal"} or
                    (index is not None if phase != "before_admission" else
                     type(index) is not int or not 0 <= index < 16)):
                return False
            if (type(gate) is not dict or set(gate) != {"containment", "denials", "oom"} or
                    type(gate["containment"]) is not bool or not (
                        _valid_counters(counters) or counters is None)):
                return False
            if counters is None:
                if gate != {"containment": False, "denials": None, "oom": None}:
                    return False
            elif (gate["containment"] is not True or
                  type(gate["denials"]) is not int or type(gate["oom"]) is not int or
                  gate["denials"] != counters["pids_denials"] or
                  gate["oom"] != max(counters["memory_oom"], counters["memory_oom_kill"])):
                return False
        if value["status"] == "observed_success":
            if (len(samples) != 19 or
                    [item["phase"] for item in samples[:2]] != ["initial", "initial"] or
                    samples[-1]["phase"] != "terminal" or
                    sorted(item["index"] for item in samples[2:-1]) != list(range(16)) or
                    any(item["phase"] != "before_admission" for item in samples[2:-1]) or
                    any(item["gate"] != {"containment": True, "denials": 0, "oom": 0}
                        for item in samples)):
                return False
        result = value["probe_result"]
        if result is None:
            return value["status"] == "incomplete_or_failed" and value["failure_code"] is not None
        return (probe is not None and observer is not None and
                probe.validate_result(result, observer) and
                value["status"] == result["status"] and value["failure_code"] is None)
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def _valid_counters(value: Any) -> bool:
    return (type(value) is dict and set(value) == {"pids_denials", "memory_oom",
        "memory_oom_kill", "pids_current", "memory_current", "memory_peak"} and
        all(type(number) is int and 0 <= number <= 2**63 - 1
            for number in value.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-once", action="store_true")
    mode.add_argument("--containment-only", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        root = Path(args.root)
        result = (run_once(root, args.receipt_sha256) if args.run_once else
                  run_containment_only(root, args.receipt_sha256))
        return 0 if (result.get("status") == "observed_success" or
                     result.get("verified") is True) else 1
    except BaseException:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
