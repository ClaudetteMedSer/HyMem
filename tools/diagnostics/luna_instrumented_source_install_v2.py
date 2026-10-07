"""Install only the repaired four-question launcher; zero-inference prepare."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess


FILES = {
    "luna_lme_diagnostic_launch_v9.py": ("tools/diagnostics/luna_lme_diagnostic_launch_v9.py", "baeb35caf1ddca21502e6d92a33f9c8c7a0f5f7618e7fb312eb90c1fb23fb445"),
}
SCHEMA = "luna-instrumented-source-install-v2"
ROOT = re.compile(r"/home/atta/\.hymem-lme-diagnostic-preflight-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
REMOTE = r'''
import base64,hashlib,json,os,re,stat,subprocess
from pathlib import Path
root=Path(PAYLOAD["root"])
assert re.fullmatch(r"/home/atta/\.hymem-lme-diagnostic-preflight-[a-z0-9_-]{8,}",str(root))
assert os.getuid()==1000 and root.is_dir() and not root.is_symlink()
assert root.stat().st_uid==1000 and stat.S_IMODE(root.stat().st_mode)==0o700
assert set(PAYLOAD["files"])=={"luna_lme_diagnostic_launch_v9.py"}
assert {entry.name for entry in root.iterdir()}=={"candidate","code","source-map.json"}
assert all(stat.S_ISDIR((root/name).lstat().st_mode) for name in ("candidate","code"))
assert stat.S_ISREG((root/"source-map.json").lstat().st_mode)
for name,record in PAYLOAD["files"].items():
    target=root/name
    assert not target.exists() and not target.is_symlink()
    data=base64.b64decode(record["data"],validate=True)
    assert hashlib.sha256(data).hexdigest()==record["sha256"]
for name,record in PAYLOAD["files"].items():
    fd=os.open(root/name,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as output:
        os.fchmod(output.fileno(),0o600)
        output.write(base64.b64decode(record["data"],validate=True))
        output.flush()
        os.fsync(output.fileno())
tool="luna_lme_diagnostic_launch_v9.py"
completed=subprocess.run(["/usr/bin/python3","-I","-B",str(root/tool),"--prepare-root",str(root)],
    capture_output=True,text=True,timeout=90)
try: result=json.loads(completed.stdout)
except (ValueError,TypeError): result={}
expected="hymem-luna-lme-diagnostic-"
expected+=root.name.removeprefix(".hymem-lme-diagnostic-")+".service"
if (completed.returncode==0 and type(result) is dict and result.get("prepared") is True
    and type(result.get("model_calls")) is int and result["model_calls"]==0
    and result.get("root")==str(root) and result.get("unit")==expected
    and type(result.get("receipt_sha256")) is str
    and re.fullmatch("[0-9a-f]{64}",result["receipt_sha256"])):
    print(json.dumps({"schema":"luna-instrumented-source-install-v2",
        "prepared":True,"model_calls":0,"root":str(root),"unit":expected,
        "kind":"pilot","receipt_sha256":result["receipt_sha256"]},sort_keys=True))
else:
    print(json.dumps({"status":"prepare_unverified"}))
    raise SystemExit(1)
'''


def _success_projection(value, *, root: str, kind: str, returncode: int):
    if (returncode != 0 or type(value) is not dict
            or set(value) != {"schema", "prepared", "model_calls", "root", "unit",
                              "kind", "receipt_sha256"}):
        return None
    unit_prefix = "hymem-luna-lme-diagnostic-"
    expected_unit = unit_prefix + Path(root).name.removeprefix(".hymem-lme-diagnostic-") + ".service"
    if (value["schema"] != SCHEMA or value["prepared"] is not True
            or type(value["model_calls"]) is not int or value["model_calls"] != 0
            or type(value["root"]) is not str or value["root"] != root
            or type(value["kind"]) is not str or value["kind"] != kind
            or type(value["unit"]) is not str or value["unit"] != expected_unit
            or type(value["receipt_sha256"]) is not str
            or HEX.fullmatch(value["receipt_sha256"]) is None):
        return None
    return {"schema": SCHEMA, "prepared": True, "model_calls": 0,
            "root": root, "unit": expected_unit, "kind": kind,
            "receipt_sha256": value["receipt_sha256"]}


def _unverified():
    print(json.dumps({"status": "installation_unverified", "never_repeat_automatically": True}))
    return 1


def _unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_remote_field")
        value[key] = item
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    if ROOT.fullmatch(args.root) is None:
        raise ValueError("root_invalid")
    repo = Path(__file__).resolve().parents[2]
    files = {}
    for name, (relative, digest) in FILES.items():
        path = repo / relative
        if path.is_symlink() or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("source_not_accepted")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("source_drift")
        files[name] = {"sha256": digest, "data": base64.b64encode(data).decode("ascii")}
    payload = {"root": args.root, "files": files}
    remote = "PAYLOAD=" + repr(payload) + "\n" + REMOTE
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1", "afrodite", "python3 -I -B -"],
            input=remote, text=True, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return _unverified()
    try:
        if len(result.stdout) > 8192:
            return _unverified()
        parsed = json.loads(result.stdout, object_pairs_hook=_unique_fields,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite_remote_receipt")))
    except (ValueError, TypeError):
        return _unverified()
    safe = _success_projection(parsed, root=args.root, kind="pilot",
                               returncode=result.returncode)
    if safe is None:
        return _unverified()
    print(json.dumps(safe, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
