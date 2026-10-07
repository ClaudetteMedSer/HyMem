"""One-shot source install and zero-inference prepare for the native OAuth probe."""
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = "/home/atta/.hymem-lme-diagnostic-preflight-wszo8v4u"
SCHEMA = "hermes-native-oauth-probe-install-v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
FILES = {
    "code/benchmarks/hermes_codex_responses_v1.py": (
        "benchmarks/hermes_codex_responses_v1.py",
        "fa88c3c2aa9cf042cb0110e5d9f3eea4ffd951270daf57d27dbe972bd2105e94"),
    "code/benchmarks/hermes_lme_oauth_v1.py": (
        "benchmarks/hermes_lme_oauth_v1.py",
        "4c7162ec021838487a38be4ab068b1b5176bfcee531f70e0e529c4fbf501c04f"),
    "hermes_native_oauth_probe_v1.py": (
        "tools/diagnostics/hermes_native_oauth_probe_v1.py",
        "64925490875f59486fb3a3b2d98e97b2eabf3935d7ad8e08769d9367b9d5ed6a"),
    "luna_lme_diagnostic_launch_v9.py": (
        "tools/diagnostics/luna_lme_diagnostic_launch_v9.py",
        "baeb35caf1ddca21502e6d92a33f9c8c7a0f5f7618e7fb312eb90c1fb23fb445"),
}
REMOTE = r'''
import base64,hashlib,json,os,stat,subprocess
from pathlib import Path
root=Path("/home/atta/.hymem-lme-diagnostic-preflight-wszo8v4u")
expected={
 "code/benchmarks/hermes_codex_responses_v1.py":"fa88c3c2aa9cf042cb0110e5d9f3eea4ffd951270daf57d27dbe972bd2105e94",
 "code/benchmarks/hermes_lme_oauth_v1.py":"4c7162ec021838487a38be4ab068b1b5176bfcee531f70e0e529c4fbf501c04f",
 "hermes_native_oauth_probe_v1.py":"64925490875f59486fb3a3b2d98e97b2eabf3935d7ad8e08769d9367b9d5ed6a",
 "luna_lme_diagnostic_launch_v9.py":"baeb35caf1ddca21502e6d92a33f9c8c7a0f5f7618e7fb312eb90c1fb23fb445"}
assert type(PAYLOAD) is dict and set(PAYLOAD)=={"root","files"}
assert PAYLOAD["root"]==str(root) and type(PAYLOAD["files"]) is dict
assert set(PAYLOAD["files"])==set(expected)
assert os.getuid()==1000 and root.parent==Path("/home/atta")
def private_dir(path):
 info=path.lstat()
 assert stat.S_ISDIR(info.st_mode) and info.st_uid==1000 and stat.S_IMODE(info.st_mode)==0o700
private_dir(root)
assert {entry.name for entry in root.iterdir()}=={"candidate","code","source-map.json"}
private_dir(root/"candidate")
private_dir(root/"code")
private_dir(root/"code/benchmarks")
source_map=(root/"source-map.json").lstat()
assert stat.S_ISREG(source_map.st_mode) and source_map.st_uid==1000 and stat.S_IMODE(source_map.st_mode)==0o600
decoded={}
for name,digest in expected.items():
 target=root/name
 assert not target.exists() and not target.is_symlink()
 record=PAYLOAD["files"][name]
 assert type(record) is dict and set(record)=={"sha256","data"} and record["sha256"]==digest
 data=base64.b64decode(record["data"],validate=True)
 assert hashlib.sha256(data).hexdigest()==digest
 decoded[name]=data
for name,data in decoded.items():
 target=root/name
 fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,"wb") as output:
  os.fchmod(output.fileno(),0o600)
  output.write(data)
  output.flush()
  os.fsync(output.fileno())
 assert hashlib.sha256(target.read_bytes()).hexdigest()==expected[name]
completed=subprocess.run(["/usr/bin/python3","-I","-B",str(root/"hermes_native_oauth_probe_v1.py"),
 "--prepare-root",str(root)],capture_output=True,text=True,timeout=90)
try:
 result=json.loads(completed.stdout)
except (ValueError,TypeError):
 result=None
unit="hymem-luna-native-oauth-probe-preflight-wszo8v4u.service"
if (completed.returncode==0 and type(result) is dict
 and set(result)=={"schema","prepared","model_calls","root","unit","receipt_sha256"}
 and result["schema"]=="hermes-native-oauth-probe-v1"
 and result["prepared"] is True and type(result["model_calls"]) is int and result["model_calls"]==0
 and result["root"]==str(root) and result["unit"]==unit
 and type(result["receipt_sha256"]) is str and len(result["receipt_sha256"])==64
 and all(c in "0123456789abcdef" for c in result["receipt_sha256"])):
 print(json.dumps({"schema":"hermes-native-oauth-probe-install-v1","prepared":True,
  "model_calls":0,"root":str(root),"unit":unit,"receipt_sha256":result["receipt_sha256"]},sort_keys=True))
else:
 print(json.dumps({"status":"prepare_unverified"}))
 raise SystemExit(1)
'''


def _unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_field")
        value[key] = item
    return value


def _sources():
    repo = Path(__file__).resolve().parents[2]
    files = {}
    for name, (relative, digest) in FILES.items():
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("source_invalid")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("source_drift")
        files[name] = {"sha256": digest, "data": base64.b64encode(data).decode("ascii")}
    return files


def _projection(value, returncode):
    if (returncode != 0 or type(value) is not dict
            or set(value) != {"schema", "prepared", "model_calls", "root", "unit", "receipt_sha256"}):
        return None
    if (value["schema"] != SCHEMA or value["prepared"] is not True
            or type(value["model_calls"]) is not int or value["model_calls"] != 0
            or value["root"] != ROOT
            or value["unit"] != "hymem-luna-native-oauth-probe-preflight-wszo8v4u.service"
            or type(value["receipt_sha256"]) is not str
            or HEX.fullmatch(value["receipt_sha256"]) is None):
        return None
    return value


def _unverified():
    print(json.dumps({"status": "installation_unverified", "never_repeat_automatically": True}))
    return 1


def main():
    try:
        files = _sources()
        remote = "PAYLOAD=" + repr({"root": ROOT, "files": files}) + "\n" + REMOTE
        completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1", "afrodite", "python3 -I -B -"],
            input=remote, text=True, capture_output=True, timeout=120)
        if len(completed.stdout) > 8192:
            return _unverified()
        result = json.loads(completed.stdout, object_pairs_hook=_unique_fields,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")))
        safe = _projection(result, completed.returncode)
        if safe is None:
            return _unverified()
        print(json.dumps(safe, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return _unverified()


if __name__ == "__main__":
    raise SystemExit(main())
