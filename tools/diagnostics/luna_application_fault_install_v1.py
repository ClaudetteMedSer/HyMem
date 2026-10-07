"""Build an SSH-stdin installer script for the source-only application-fault bundle.

This module never connects to a host. It validates the accepted source closure
and emits a self-contained script for root to review and run.
"""
from __future__ import annotations
import argparse
import ast
import base64
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import stat
import tarfile

RUNNER="tools/diagnostics/luna_lme_diagnostic_v9.py"
RUNNER_SHA="b3e1135893a715dec4e138c25f6bf3c70f3912dbd5df81014ee7c8f2767fd278"
MAP_SHA="1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
MAP_PIN="f22cd2be376f019efa1d39cb6c2f1e43ffef2d7ea3bb7a07ac64479241cd4b11"
CAPTURE="tools/diagnostics/luna_application_fault_capture_v2.py"
CAPTURE_SHA="e865366dc7ae1fc5cb72367c7f3d59c3c3d3227486728e7035b6e61242a44977"
PROBE="tools/diagnostics/luna_application_fault_probe_v2.py"
PROBE_SHA="017c8d925368517b5be17d8373112c5b362810d37cb62dbcb716fdbc24e90c7c"
HELPERS={"application-fault-host-v1.py":"luna_application_fault_host_v1.py",
         "application-fault-launch-v1.py":"luna_application_fault_launch_v1.py",
         "application-fault-progress-v1.py":"luna_application_fault_progress_v1.py"}
MAX_ARCHIVE=32*1024*1024
HEX=re.compile(r"[0-9a-f]{64}\Z")
def need(ok,code):
    if not ok: raise ValueError(code)
def sha(data): return hashlib.sha256(data).hexdigest()
def regular(path):
    try: return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError: return False
def unique(pairs):
    result={}
    for key,value in pairs:
        need(key not in result,"duplicate_json_key"); result[key]=value
    return result
def source_manifest(bundle):
    need(bundle.is_absolute() and bundle.is_dir() and not bundle.is_symlink(),"bundle_invalid")
    runner=bundle/"code"/RUNNER
    inventory=bundle/"source-map.json"
    need(regular(runner) and sha(runner.read_bytes())==RUNNER_SHA and regular(inventory) and sha(inventory.read_bytes())==MAP_SHA,"source_pin_invalid")
    constants={}
    for node in ast.parse(runner.read_bytes()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {"ACCEPTED_FILES","ACCEPTED_MAP_SHA256","PINS","DIAGNOSTIC_HELPER_SHA256"}:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    need(set(constants)=={"ACCEPTED_FILES","ACCEPTED_MAP_SHA256","PINS","DIAGNOSTIC_HELPER_SHA256"} and constants["ACCEPTED_FILES"]==514 and constants["ACCEPTED_MAP_SHA256"]==MAP_PIN,"runner_constants_invalid")
    stamp=json.loads(inventory.read_bytes(),object_pairs_hook=unique)
    entries=stamp.get("source_sha256") if type(stamp) is dict else None
    need(type(entries) is dict and len(entries)==514 and sha(json.dumps(entries,sort_keys=True,separators=(",",":")).encode())==MAP_PIN,"map_invalid")
    manifest={"bundle/source-map.json":MAP_SHA,"bundle/code/"+RUNNER:RUNNER_SHA,"bundle/code/"+CAPTURE:CAPTURE_SHA,"bundle/code/"+PROBE:PROBE_SHA,"bundle/code/benchmarks/lme_diagnostic.py":constants["DIAGNOSTIC_HELPER_SHA256"]}
    for relative,digest in entries.items(): manifest["bundle/candidate/"+relative]=digest
    for relative,digest in constants["PINS"].items(): manifest["bundle/code/"+relative]=digest
    need(len(manifest)==514+len(constants["PINS"])+5,"manifest_invalid")
    actual={p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if not p.is_dir() or p.is_symlink()}
    need(actual=={key.removeprefix("bundle/") for key in manifest},"source_file_set_invalid")
    for relative,digest in manifest.items():
        path=bundle/relative.removeprefix("bundle/")
        need(type(digest) is str and HEX.fullmatch(digest) is not None and regular(path) and sha(path.read_bytes())==digest,"source_drift")
    return manifest
def archive_bytes(bundle,helpers):
    manifest=source_manifest(bundle)
    for target,source in HELPERS.items():
        path=helpers/source
        need(regular(path),"helper_invalid")
        manifest[target]=sha(path.read_bytes())
    need(len(manifest)==len(source_manifest(bundle))+len(HELPERS),"manifest_count_invalid")
    for target in ("application-fault-launch-v1.py","application-fault-progress-v1.py"):
        constants={}
        for node in ast.parse((helpers/HELPERS[target]).read_bytes()).body:
            if (isinstance(node,ast.Assign) and len(node.targets)==1
                    and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id=="HOST_SHA256"):
                constants["HOST_SHA256"]=ast.literal_eval(node.value)
        need(constants=={"HOST_SHA256":manifest["application-fault-host-v1.py"]},"helper_host_pin_invalid")
    output=BytesIO()
    with tarfile.open(fileobj=output,mode="w") as archive:
        encoded=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
        info=tarfile.TarInfo("manifest.json"); info.size=len(encoded); info.mode=0o600
        archive.addfile(info,BytesIO(encoded))
        for relative in sorted(manifest):
            path=(bundle/relative.removeprefix("bundle/")) if relative.startswith("bundle/") else helpers/HELPERS[relative]
            info=tarfile.TarInfo(relative); info.size=path.stat().st_size; info.mode=0o600
            with path.open("rb") as stream: archive.addfile(info,stream)
    raw=output.getvalue()
    need(len(raw)<=MAX_ARCHIVE,"archive_too_large")
    return raw,manifest
REMOTE=r'''from __future__ import annotations
import ast,base64,hashlib,io,json,os,re,stat,sys,tarfile,tempfile
from pathlib import Path
RAW=base64.b64decode(__ARCHIVE_B64__)
ARCHIVE_SHA=__ARCHIVE_SHA__
RUNNER_SHA=__RUNNER_SHA__
MAP_SHA=__MAP_SHA__
MAP_PIN=__MAP_PIN__
HELPER_PINS=__HELPER_PINS__
HOST_HOME=Path("/home/atta")
def need(ok,code):
    if not ok: raise ValueError(code)
def sha(data): return hashlib.sha256(data).hexdigest()
def unique(pairs):
    result={}
    for key,value in pairs:
        need(key not in result,"duplicate_json_key"); result[key]=value
    return result
need(sys.platform=="linux" and os.getuid()==1000 and os.geteuid()==1000 and HOST_HOME.is_dir() and not HOST_HOME.is_symlink(),"host_invalid")
need(len(RAW)<=32*1024*1024 and sha(RAW)==ARCHIVE_SHA,"archive_drift")
with tarfile.open(fileobj=io.BytesIO(RAW),mode="r:") as archive:
    members=archive.getmembers()
    need(members and members[0].name=="manifest.json" and members[0].isfile(),"manifest_missing")
    raw_manifest=archive.extractfile(members[0]).read()
    manifest=json.loads(raw_manifest,object_pairs_hook=unique)
    need(type(manifest) is dict and raw_manifest==json.dumps(manifest,sort_keys=True,separators=(",",":")).encode(),"manifest_invalid")
    names=[member.name for member in members[1:]]
    need(len(names)==len(manifest) and set(names)==set(manifest) and len(names)==len(set(names)),"file_set_invalid")
    need(all(member.isfile() and not member.issym() and not member.islnk() and member.size<=2*1024*1024 for member in members),"archive_member_invalid")
    content={}
    for member in members[1:]:
        name=member.name
        parts=Path(name).parts
        need(not Path(name).is_absolute() and ".." not in parts and parts and parts[0] in {"bundle","application-fault-host-v1.py","application-fault-launch-v1.py","application-fault-progress-v1.py"},"path_invalid")
        data=archive.extractfile(member).read()
        need(type(manifest[name]) is str and re.fullmatch("[0-9a-f]{64}",manifest[name]) and sha(data)==manifest[name],"member_drift")
        content[name]=data
    need(manifest.get("bundle/code/tools/diagnostics/luna_lme_diagnostic_v9.py")==RUNNER_SHA and manifest.get("bundle/source-map.json")==MAP_SHA and {k:manifest.get(k) for k in HELPER_PINS}==HELPER_PINS,"pin_invalid")
    constants={}
    for node in ast.parse(content["bundle/code/tools/diagnostics/luna_lme_diagnostic_v9.py"]).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {"ACCEPTED_FILES","ACCEPTED_MAP_SHA256","PINS","DIAGNOSTIC_HELPER_SHA256"}:
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    need(set(constants)=={"ACCEPTED_FILES","ACCEPTED_MAP_SHA256","PINS","DIAGNOSTIC_HELPER_SHA256"} and constants["ACCEPTED_FILES"]==514 and constants["ACCEPTED_MAP_SHA256"]==MAP_PIN,"runner_invalid")
    stamp=json.loads(content["bundle/source-map.json"],object_pairs_hook=unique)
    entries=stamp.get("source_sha256") if type(stamp) is dict else None
    need(type(entries) is dict and len(entries)==514 and sha(json.dumps(entries,sort_keys=True,separators=(",",":")).encode())==MAP_PIN,"map_invalid")
    expected={"bundle/source-map.json":MAP_SHA,"bundle/code/tools/diagnostics/luna_lme_diagnostic_v9.py":RUNNER_SHA,"bundle/code/tools/diagnostics/luna_application_fault_capture_v2.py":"e865366dc7ae1fc5cb72367c7f3d59c3c3d3227486728e7035b6e61242a44977","bundle/code/tools/diagnostics/luna_application_fault_probe_v2.py":"017c8d925368517b5be17d8373112c5b362810d37cb62dbcb716fdbc24e90c7c","bundle/code/benchmarks/lme_diagnostic.py":constants["DIAGNOSTIC_HELPER_SHA256"],**HELPER_PINS}
    for relative,digest in entries.items(): expected["bundle/candidate/"+relative]=digest
    for relative,digest in constants["PINS"].items(): expected["bundle/code/"+relative]=digest
    need(manifest==expected,"manifest_source_invalid")
    root=Path(tempfile.mkdtemp(prefix=".hymem-luna-application-fault-v1-",dir=HOST_HOME))
    os.chmod(root,0o700)
    for name,data in content.items():
        target=root/name
        target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
    directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    print(json.dumps({"schema":"luna-application-fault-install-v1","root":str(root),"source_manifest_sha256":sha(json.dumps({k.removeprefix("bundle/"):v for k,v in expected.items() if k.startswith("bundle/")},sort_keys=True,separators=(",",":")).encode()),"model_calls":0},sort_keys=True))
'''
def render_remote_script(bundle,helpers):
    raw,manifest=archive_bytes(bundle,helpers)
    pins={key:manifest[key] for key in HELPERS}
    replacements={"__ARCHIVE_B64__":repr(base64.b64encode(raw).decode()),"__ARCHIVE_SHA__":repr(sha(raw)),"__RUNNER_SHA__":repr(RUNNER_SHA),"__MAP_SHA__":repr(MAP_SHA),"__MAP_PIN__":repr(MAP_PIN),"__HELPER_PINS__":repr(pins)}
    script=REMOTE
    for key,value in replacements.items(): script=script.replace(key,value)
    return script.encode()
def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",required=True);parser.add_argument("--helpers",required=True);parser.add_argument("--script-output",required=True)
    args=parser.parse_args(argv)
    try:
        script=render_remote_script(Path(args.bundle),Path(args.helpers))
        output=Path(args.script_output)
        need(output.is_absolute() and not output.exists() and output.parent.is_dir(),"output_invalid")
        output.write_bytes(script)
        output.chmod(0o600)
        print(json.dumps({"schema":"luna-application-fault-install-v1","script":str(output),"script_sha256":sha(script),"ssh_stdin_command":"ssh -C -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 afrodite /usr/bin/python3 -I -B - < "+str(output),"host_actions":0},sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps({"schema":"luna-application-fault-install-v1","status":"unverified"},sort_keys=True))
        return 1
if __name__=="__main__":raise SystemExit(main())
