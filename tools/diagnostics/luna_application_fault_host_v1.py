"""One-shot host service for the accepted single-question Luna attribution probe."""
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

SCHEMA = "luna-application-fault-host-v1"
HOST_HOME = Path("/home/atta")
HOST_UID = 1000
ROOT_PATTERN = re.compile(r"\.hymem-luna-application-fault-v1-[a-z0-9_]{8}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
RUNNER_REL = "tools/diagnostics/luna_lme_diagnostic_v9.py"
RUNNER_SHA = "b3e1135893a715dec4e138c25f6bf3c70f3912dbd5df81014ee7c8f2767fd278"
CAPTURE_REL = "tools/diagnostics/luna_application_fault_capture_v2.py"
CAPTURE_SHA = "e865366dc7ae1fc5cb72367c7f3d59c3c3d3227486728e7035b6e61242a44977"
PROBE_REL = "tools/diagnostics/luna_application_fault_probe_v2.py"
PROBE_SHA = "017c8d925368517b5be17d8373112c5b362810d37cb62dbcb716fdbc24e90c7c"
MAP_SHA = "1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
MAP_PIN = "f22cd2be376f019efa1d39cb6c2f1e43ffef2d7ea3bb7a07ac64479241cd4b11"
DATASET = "/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json"
DATASET_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
BINARY = "/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex"
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"
CGROUP_ROOT = Path("/sys/fs/cgroup")
RUNTIME_DIR = Path("/run/user/1000")
UNIT_PREFIX = "hymem-luna-application-fault-v1-"
POLICY = {"tasks_max":256,"memory_max":4294967296,"cpu_percent":200,"runtime_seconds":730,"stop_seconds":10,"restart":"no","kill_mode":"control-group","oom_policy":"kill","umask":"0077"}
LIMITS = {"workers":1,"turns":12,"known_tokens":160000,"campaign_seconds":600,"question_seconds":600,"invocation_seconds":120,"warm_requests":16,"warm_seconds":300,"event_bound":4096,"index_seconds":540,"no_canary":True,"no_search":True,"no_answer":True,"no_judge":True,"no_scoring":True}
FIELDS = ("ActiveState","SubState","MainPID","ControlGroup","NRestarts","Type","MemoryMax","TasksMax","CPUQuotaPerSecUSec","KillMode","Restart","RemainAfterExit","OOMPolicy","RuntimeMaxUSec","TimeoutStopUSec","UMask","Result","ExecMainStatus")

def _need(ok: bool, code: str) -> None:
    if not ok: raise ValueError(code)

def _regular(path: Path) -> bool:
    try: return stat.S_ISREG(path.lstat().st_mode)
    except OSError: return False

def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def _json_bytes(value: dict) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")

def _root(root: Path) -> Path:
    _need(root.is_absolute() and root.parent==HOST_HOME and ROOT_PATTERN.fullmatch(root.name) is not None and root.is_dir() and not root.is_symlink(),"root_invalid")
    info=root.stat()
    _need(info.st_uid==HOST_UID and stat.S_IMODE(info.st_mode)==0o700,"root_permissions_invalid")
    return root

def unit_for(root: Path, mode: str) -> str:
    _root(root); _need(mode in {"containment","probe"},"mode_invalid")
    return UNIT_PREFIX+mode+"-"+root.name.removeprefix(".hymem-luna-application-fault-v1-")+".service"

def _literal(path: Path, names: set[str]) -> dict:
    found={}
    for node in ast.parse(path.read_bytes()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            found[node.targets[0].id]=ast.literal_eval(node.value)
    _need(set(found)==names,"source_constants_invalid")
    return found

def source_manifest(root: Path) -> dict[str,str]:
    _root(root)
    bundle=root/"bundle"
    _need(bundle.is_dir() and not bundle.is_symlink(),"bundle_invalid")
    runner=bundle/"code"/RUNNER_REL
    inventory=bundle/"source-map.json"
    _need(_regular(runner) and _sha(runner)==RUNNER_SHA and _regular(inventory) and _sha(inventory)==MAP_SHA,"source_pin_invalid")
    constants=_literal(runner,{"ACCEPTED_FILES","ACCEPTED_MAP_SHA256","PINS","DIAGNOSTIC_HELPER_SHA256"})
    _need(constants["ACCEPTED_FILES"]==514 and constants["ACCEPTED_MAP_SHA256"]==MAP_PIN,"source_constants_invalid")
    stamp=json.loads(inventory.read_bytes())
    entries=stamp.get("source_sha256") if type(stamp) is dict else None
    _need(type(entries) is dict and len(entries)==514 and hashlib.sha256(_json_bytes(entries)).hexdigest()==MAP_PIN,"source_map_invalid")
    manifest={"source-map.json":MAP_SHA,"code/"+RUNNER_REL:RUNNER_SHA,"code/"+CAPTURE_REL:CAPTURE_SHA,"code/"+PROBE_REL:PROBE_SHA,"code/benchmarks/lme_diagnostic.py":constants["DIAGNOSTIC_HELPER_SHA256"]}
    for relative,digest in entries.items(): manifest["candidate/"+relative]=digest
    for relative,digest in constants["PINS"].items(): manifest["code/"+relative]=digest
    _need(len(manifest)==514+len(constants["PINS"])+5,"source_manifest_invalid")
    actual={p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if not p.is_dir() or p.is_symlink()}
    _need(actual==set(manifest),"source_file_set_invalid")
    for relative,digest in manifest.items():
        path=bundle/relative
        _need(type(digest) is str and HEX.fullmatch(digest) is not None and _regular(path) and not path.is_symlink() and _sha(path)==digest,"source_drift")
    return manifest

def verify_sources(root: Path) -> str:
    manifest=source_manifest(root)
    return hashlib.sha256(_json_bytes(manifest)).hexdigest()

def _module(name: str, path: Path, sha: str):
    _need(_regular(path) and _sha(path)==sha,"helper_drift")
    spec=importlib.util.spec_from_file_location(name,path)
    _need(spec is not None and spec.loader is not None,"helper_invalid")
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def load_probe(root: Path):
    verify_sources(root)
    code=root/"bundle"/"code"/"tools"/"diagnostics"
    runner=_module("pinned_application_fault_runner",code/"luna_lme_diagnostic_v9.py",RUNNER_SHA)
    capture=_module("pinned_application_fault_capture",code/"luna_application_fault_capture_v2.py",CAPTURE_SHA)
    probe=_module("pinned_application_fault_probe",code/"luna_application_fault_probe_v2.py",PROBE_SHA)
    loaded=runner.load_verified(root/"bundle",root/"bundle"/"source-map.json",MAP_SHA,Path(DATASET),Path(BINARY),BINARY_SHA256,runner.DIAGNOSTIC_HELPER_SHA256)
    probe.verify_sources(loaded,runner,capture,BINARY_SHA256)
    return runner,capture,probe,loaded

def selected_digest(root: Path) -> str:
    _,_,probe,loaded=load_probe(root)
    return probe.selected_row_sha256(probe.select_question(loaded))

def receipt_for(root: Path, host_sha256: str, mode: str, row_sha256: str) -> dict:
    _need(type(host_sha256) is str and HEX.fullmatch(host_sha256) is not None and type(row_sha256) is str and HEX.fullmatch(row_sha256) is not None,"receipt_argument_invalid")
    unit=unit_for(root,mode)
    return {"schema":SCHEMA,"root":str(root),"uid":HOST_UID,"mode":mode,"unit":unit,"expected_cgroup":"/user.slice/user-1000.slice/user@1000.service/app.slice/"+unit,"host_sha256":host_sha256,"source_manifest_sha256":verify_sources(root),"runner_sha256":RUNNER_SHA,"capture_sha256":CAPTURE_SHA,"probe_sha256":PROBE_SHA,"candidate_map_sha256":MAP_PIN,"candidate_inventory_sha256":MAP_SHA,"dataset_path":DATASET,"dataset_sha256":DATASET_SHA,"binary_path":BINARY,"binary_sha256":BINARY_SHA256,"source_question_index":1,"selected_row_sha256":row_sha256,"model":"gpt-6-luna","reasoning":"low","auth":"chatgpt","billing_policy":"included_allowance_or_existing_finite_positive_credits_per_window_v2","notification_policy":"agent_message_delta_optout_v1","automatic_topup_user_attested_off":True,"reload_allowed":False,"policy":dict(POLICY),"limits":dict(LIMITS),"one_shot":True}

def verify_receipt(root: Path,digest: str,mode: str) -> dict:
    _root(root); _need(type(digest) is str and HEX.fullmatch(digest) is not None,"receipt_hash_invalid")
    host_file=root/"application-fault-host-v1.py"
    _need(Path(__file__).absolute()==host_file.absolute() and _regular(host_file) and host_file.stat().st_uid==HOST_UID and stat.S_IMODE(host_file.stat().st_mode)==0o600,"host_origin_invalid")
    path=root/("containment-receipt.json" if mode=="containment" else "launch-receipt.json")
    _need(_regular(path) and path.stat().st_uid==HOST_UID and stat.S_IMODE(path.stat().st_mode)==0o600 and path.stat().st_size<=8192 and _sha(path)==digest,"receipt_drift")
    value=json.loads(path.read_bytes())
    _need(type(value) is dict and path.read_bytes()==_json_bytes(receipt_for(root,_sha(host_file),mode,value.get("selected_row_sha256"))),"receipt_invalid")
    return value

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
           type(index) is int and 0 <= index < 12), "phase_invalid")
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



def resource_check(receipt: dict) -> dict[str,int]:
    gate,counters=live_attestation(receipt,"before_admission",0)
    _need(gate=={"containment":True,"denials":0,"oom":0} and counters is not None,"resource_observer_unverified")
    group=_group(receipt)
    peak=_scalar(group/"pids.peak")
    _need(0<=counters["pids_current"]<=peak<=256,"resource_observer_unverified")
    return {"current":counters["pids_current"],"peak":peak,"limit":256,"denials":counters["pids_denials"]}

def _valid_counters(value: Any) -> bool:
    return (type(value) is dict and set(value)=={"pids_denials","memory_oom","memory_oom_kill","pids_current","memory_current","memory_peak"}
            and all(type(v) is int and 0<=v<=2**63-1 for v in value.values())
            and value["pids_current"]<=256 and value["memory_current"]<=value["memory_peak"])

def _write_once(path: Path,value: dict) -> None:
    data=_json_bytes(value); _need(len(data)<=16384,"result_too_large")
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)

def _base(root:Path,digest:str,mode:str)->dict:
    _need(sys.platform=="linux" and os.getuid()==HOST_UID and os.geteuid()==HOST_UID,"host_user_invalid")
    receipt=verify_receipt(root,digest,mode)
    attempt=root/("containment-attempt.json" if mode=="containment" else "launch-attempt.json")
    _need(_regular(attempt) and attempt.read_bytes()==_json_bytes({"receipt_sha256":digest,"one_shot":True}),"attempt_invalid")
    marker=root/("containment-execution-marker.json" if mode=="containment" else "probe-execution-marker.json")
    _write_once(marker,{"receipt_sha256":digest,"execution_started":True})
    return receipt

def run_containment_only(root:Path,digest:str)->dict:
    receipt=_base(root,digest,"containment")
    gate,counters=live_attestation(receipt,"initial",None)
    verified=(gate=={"containment":True,"denials":0,"oom":0}
              and _regular(Path(DATASET)) and _sha(Path(DATASET))==DATASET_SHA
              and _regular(Path(BINARY)) and _sha(Path(BINARY))==BINARY_SHA256
              and selected_digest(root)==receipt["selected_row_sha256"])
    result={"schema":SCHEMA,"mode":"containment","verified":verified,"model_calls":0,"resources":counters}
    _write_once(root/"containment-result.json",result)
    return result

def run_once(root:Path,digest:str)->dict:
    receipt=_base(root,digest,"probe")
    failure="initial_containment_invalid"; status="unverified"
    try:
        gate,_=live_attestation(receipt,"initial",None)
        _need(gate=={"containment":True,"denials":0,"oom":0},failure)
        failure="binary_or_dataset_drift"
        _need(_regular(Path(BINARY)) and _sha(Path(BINARY))==BINARY_SHA256 and _regular(Path(DATASET)) and _sha(Path(DATASET))==DATASET_SHA,failure)
        failure="source_invalid"
        runner,capture,probe,loaded=load_probe(root)
        _need(probe.selected_row_sha256(probe.select_question(loaded))==receipt["selected_row_sha256"],"question_drift")
        failure="probe_exception"
        result=probe.run_probe(loaded,runner,capture,output=root/"private-probe",containment_verified=True,binary_sha256=BINARY_SHA256,resource_check=lambda:resource_check(receipt))
        failure="probe_result_invalid"
        projected=probe.validate_result(result,capture)
        _need(projected["selected_row_sha256"]==receipt["selected_row_sha256"],"question_drift")
        status=projected["status"]; failure=None
    except BaseException:
        pass
    try:
        terminal,_=live_attestation(receipt,"terminal",None)
        if terminal!={"containment":True,"denials":0,"oom":0} and failure is None: failure="terminal_containment_invalid"; status="unverified"
    except BaseException:
        if failure is None: failure="terminal_containment_invalid"; status="unverified"
    value={"schema":SCHEMA,"mode":"probe","status":status,"failure_code":failure,"probe_result_present":(root/"private-probe"/"probe-result.json").exists()}
    _write_once(root/"host-result.json",value)
    return value

def main(argv=None)->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True); parser.add_argument("--receipt-sha256",required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--containment-only",action="store_true"); group.add_argument("--run-once",action="store_true")
    args=parser.parse_args(argv)
    try:
        result=run_containment_only(Path(args.root),args.receipt_sha256) if args.containment_only else run_once(Path(args.root),args.receipt_sha256)
        print(json.dumps({"schema":SCHEMA,"mode":result["mode"],"completed":True},sort_keys=True))
        return 0 if result.get("verified") is True or result.get("failure_code") is None else 1
    except BaseException:
        print(json.dumps({"schema":SCHEMA,"status":"unverified"},sort_keys=True))
        return 1
if __name__=="__main__": raise SystemExit(main())
