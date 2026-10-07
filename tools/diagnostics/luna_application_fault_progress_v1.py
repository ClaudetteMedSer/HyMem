"""Metadata-only reader for the one-shot application-fault probe."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

SCHEMA="luna-application-fault-progress-v1"
HOST_SHA256="5a136d0a6d64018836982a9411338c2ae8adca3a5e3fa7b1d7e37e457ddc2dbe"
HEX=re.compile(r"[0-9a-f]{64}\Z")

def need(ok,code):
    if not ok: raise ValueError(code)
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def regular(path,cap):
    try:
        meta=path.lstat()
        return stat.S_ISREG(meta.st_mode) and meta.st_uid==1000 and stat.S_IMODE(meta.st_mode)==0o600 and meta.st_size<=cap
    except OSError: return False
def read(path,cap):
    need(regular(path,cap),"metadata_invalid")
    def unique(pairs):
        result={}
        for key,value in pairs:
            need(key not in result,"duplicate_json_key")
            result[key]=value
        return result
    def invalid(_): raise ValueError("nonfinite_json")
    value=json.loads(path.read_bytes(),object_pairs_hook=unique,parse_constant=invalid)
    need(type(value) is dict,"metadata_invalid")
    return value
def module(name,path,digest):
    need(sha(path)==digest,"source_drift")
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,"source_invalid")
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result
def inspect(root,digest,mode):
    need(sys.platform=="linux" and os.getuid()==1000 and os.geteuid()==1000,"host_user_invalid")
    need(mode in {"containment","probe"} and type(digest) is str and HEX.fullmatch(digest) is not None,"argument_invalid")
    host_path=root/"application-fault-host-v1.py"
    need(regular(host_path,100000) and sha(host_path)==HOST_SHA256,"host_drift")
    host=module("pinned_application_fault_host_reader",host_path,HOST_SHA256)
    host._root(root)
    receipt=host.verify_receipt(root,digest,mode)
    need(host._regular(Path(host.DATASET)) and sha(Path(host.DATASET))==host.DATASET_SHA
         and host._regular(Path(host.BINARY)) and sha(Path(host.BINARY))==host.BINARY_SHA256,
         "runtime_source_drift")
    attempt=root/("containment-attempt.json" if mode=="containment" else "launch-attempt.json")
    execution=root/("containment-execution-marker.json" if mode=="containment" else "probe-execution-marker.json")
    attempted=attempt.exists() or attempt.is_symlink()
    started=execution.exists() or execution.is_symlink()
    if attempted: need(read(attempt,512)=={"receipt_sha256":digest,"one_shot":True} and attempt.read_bytes()==host._json_bytes({"receipt_sha256":digest,"one_shot":True}),"attempt_invalid")
    if started: need(attempted and read(execution,512)=={"receipt_sha256":digest,"execution_started":True} and execution.read_bytes()==host._json_bytes({"receipt_sha256":digest,"execution_started":True}),"execution_invalid")
    runtime=host.terminal_runtime(receipt) if attempted else None
    clean=runtime is not None and runtime["policy_verified"] is True and runtime["unit_stopped"] is True and runtime["group_matched"] is True and runtime["recursive_cleanup_verified"] is True
    base={"schema":SCHEMA,"mode":mode,"root":str(root),"receipt_sha256":digest,"unit":receipt["unit"],"attempted":attempted,"execution_started":started,"source_verified":True,"runtime":runtime,"recursive_cleanup_verified":bool(clean),"lme_completion_proved":False}
    checkpoint=first_fault(root,host,attempted,started) if mode=="probe" else None
    result_path=root/("containment-result.json" if mode=="containment" else "host-result.json")
    if not result_path.exists() and not result_path.is_symlink():
        state="prepared" if not attempted else "result_missing" if runtime and runtime["unit_stopped"] else "pending_or_ambiguous"
        return {**base,"status":state,"result_verified":False,"first_fault":checkpoint,"probe":None}
    need(attempted and started,"result_without_execution")
    try: result=read(result_path,16384)
    except (OSError,ValueError,TypeError):
        if mode=="probe": return {**base,"status":"incomplete_or_failed","result_verified":False,"first_fault":checkpoint,"probe":None}
        raise
    if mode=="containment":
        counters=result.get("resources")
        need(set(result)=={"schema","mode","verified","model_calls","resources"} and result["schema"]==host.SCHEMA and result["mode"]=="containment" and type(result["verified"]) is bool and type(result["model_calls"]) is int and result["model_calls"]==0 and host._valid_counters(counters),"result_invalid")
        need(not result["verified"] or counters["pids_denials"]==counters["memory_oom"]==counters["memory_oom_kill"]==0,"result_invalid")
        success=result["verified"] and clean and runtime["runtime_exit"]=="success"
        return {**base,"status":"verified_clean" if success else "unverified","result_verified":True,"first_fault":None,"probe":None}
    if not (set(result)=={"schema","mode","status","failure_code","probe_result_present"} and result["schema"]==host.SCHEMA and result["mode"]=="probe" and result["status"] in {"application_fault","transport_stop","provider_denial","resource_stop","inconclusive","unverified"} and result["failure_code"] in {None,"initial_containment_invalid","binary_or_dataset_drift","source_invalid","probe_exception","probe_result_invalid","terminal_containment_invalid"} and type(result["probe_result_present"]) is bool):
        return {**base,"status":"incomplete_or_failed","result_verified":False,"first_fault":checkpoint,"probe":None}
    probe_path=root/"private-probe"/"probe-result.json"
    present=probe_path.exists() or probe_path.is_symlink()
    if present!=result["probe_result_present"]:
        return {**base,"status":"incomplete_or_failed","result_verified":False,"first_fault":checkpoint,"probe":None}
    if not present:
        return {**base,"status":"incomplete_or_failed","result_verified":True,"first_fault":checkpoint,"probe":{"host_failure_code":result["failure_code"]}}
    capture=module("pinned_application_fault_capture_reader",root/"bundle"/"code"/host.CAPTURE_REL,host.CAPTURE_SHA)
    probe=module("pinned_application_fault_probe_reader",root/"bundle"/"code"/host.PROBE_REL,host.PROBE_SHA)
    try: value=probe.validate_result(read(probe_path,16384),capture)
    except (OSError,ValueError,TypeError,KeyError):
        return {**base,"status":"incomplete_or_failed","result_verified":False,"first_fault":checkpoint,"probe":None}
    need(value["selected_row_sha256"]==receipt["selected_row_sha256"],"probe_result_invalid")
    if checkpoint is not None:
        need(value["first_snapshot"]["schema"]==checkpoint["schema"] and value["first_snapshot"]["first"]==checkpoint["first"],"checkpoint_mismatch")
    projection={key:value[key] for key in ("status","stop","phase","turns","known_tokens","usage_complete","in_flight","reserved","stages","resource","checkpoint_durable","adapter_cleanup_ok","client_cleanup_ok","accounting_reconciled")}
    consistent=value["status"]==result["status"] and result["failure_code"] is None
    return {**base,"status":value["status"] if consistent else "incomplete_or_failed","result_verified":consistent,"first_fault":checkpoint,"probe":projection if consistent else {"host_failure_code":result["failure_code"]}}
def first_fault(root,host,attempted,started):
    path=root/"private-probe"/"first-fault.json"
    if not path.exists() and not path.is_symlink(): return None
    need(attempted and started,"checkpoint_without_execution")
    capture=module("pinned_application_fault_capture_checkpoint",root/"bundle"/"code"/host.CAPTURE_REL,host.CAPTURE_SHA)
    snapshot=capture.validate_snapshot(read(path,16384))
    need(snapshot is not None and snapshot["first"] is not None,"checkpoint_invalid")
    return snapshot
def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True); parser.add_argument("--receipt-sha256",required=True); parser.add_argument("--mode",choices=("containment","probe"),required=True)
    args=parser.parse_args(argv)
    try:
        value=inspect(Path(args.root),args.receipt_sha256,args.mode)
        print(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False))
        return 0 if value["status"] in {"verified_clean","application_fault","transport_stop","provider_denial","resource_stop","inconclusive"} and value["recursive_cleanup_verified"] else 1
    except BaseException:
        print(json.dumps({"schema":SCHEMA,"status":"unverified"},sort_keys=True))
        return 1
if __name__=="__main__": raise SystemExit(main())
