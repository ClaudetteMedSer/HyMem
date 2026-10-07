"""Install two pinned sources and prepare one invented-text probe; no inference."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = "/home/atta/.hymem-lme-diagnostic-preflight-xvn648yr"
FILES = {
    "access-check-v1.py": ("tools/diagnostics/luna_subscription_access_check_v1.py",
        "a0bc7ec37e243947418e92725dbd42e66adaea7941e3d25317c6cc4676c4c139"),
    "luna_lme_diagnostic_launch_v5.py": ("tools/diagnostics/luna_lme_diagnostic_launch_v5.py",
        "a1921b4c199a5b3124ff8403c5cf31061e74028a17c91b837528478f35ab0f64"),
}
REMOTE = r'''
import base64,hashlib,json,os,stat,subprocess
from pathlib import Path
root=Path(PAYLOAD["root"])
assert str(root)=="/home/atta/.hymem-lme-diagnostic-preflight-xvn648yr"
assert os.getuid()==1000 and root.is_dir() and not root.is_symlink()
assert root.stat().st_uid==1000 and stat.S_IMODE(root.stat().st_mode)==0o700
assert set(PAYLOAD["files"])=={"access-check-v1.py","luna_lme_diagnostic_launch_v5.py"}
for name,record in PAYLOAD["files"].items():
    target=root/name
    assert not target.exists() and not target.is_symlink()
    data=base64.b64decode(record["data"],validate=True)
    assert hashlib.sha256(data).hexdigest()==record["sha256"]
assert not any((root/name).exists() for name in ("access-receipt.json","access-attempt.json","launch-receipt.json","launch-attempt.json"))
for name,record in PAYLOAD["files"].items():
    fd=os.open(root/name,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as output:
        os.fchmod(output.fileno(),0o600)
        output.write(base64.b64decode(record["data"],validate=True))
        output.flush()
        os.fsync(output.fileno())
completed=subprocess.run(["/usr/bin/python3","-I","-B",str(root/"access-check-v1.py"),"--prepare-root",str(root)],capture_output=True,text=True,timeout=90)
try:
    result=json.loads(completed.stdout)
except (ValueError,TypeError):
    result={}
if (completed.returncode==0 and set(result)=={"schema","prepared","model_calls","receipt_sha256","root","unit"}
    and result["schema"]=="luna-subscription-access-check-v1" and result["prepared"] is True
    and result["model_calls"]==0 and result["root"]==str(root)
    and result["unit"]=="hymem-luna-access-check-preflight-xvn648yr.service"
    and isinstance(result["receipt_sha256"],str) and len(result["receipt_sha256"])==64
    and all(c in "0123456789abcdef" for c in result["receipt_sha256"])):
    print(json.dumps(result,sort_keys=True))
else:
    print(json.dumps({"schema":"luna-access-source-install-root-v1","status":"prepare_unverified","model_calls":0}))
    raise SystemExit(1)
'''


def main():
    repo = Path(__file__).resolve().parents[2]
    files = {}
    for name, (relative, digest) in FILES.items():
        path = repo / relative
        if path.is_symlink():
            raise ValueError("source_symlink")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("source_drift")
        files[name] = {"sha256": digest, "data": base64.b64encode(data).decode("ascii")}
    payload = {"root": ROOT, "files": files}
    source = "PAYLOAD=" + repr(payload) + "\n" + REMOTE
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "afrodite", "python3 -I -B -"],
        input=source, text=True, capture_output=True, timeout=120)
    if result.stdout:
        # The remote script emits only finite preparation metadata, never stderr.
        parsed = json.loads(result.stdout)
        print(json.dumps(parsed, sort_keys=True))
    else:
        print(json.dumps({"status": "installation_unverified", "never_repeat_automatically": True}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
