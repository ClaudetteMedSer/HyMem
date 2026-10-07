"""Install the pinned SIWC launcher in one fresh host root and prepare once."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess


SOURCE = "tools/diagnostics/siwc_lme_diagnostic_launch_v9.py"
SOURCE_SHA256 = "8e6ccf33b5e3ff922e65f08b4bbae090eb8e26babcab915df1f7e4aff87f8828"
SCHEMA = "siwc-lme-diagnostic-source-install-v9"
ROOT = re.compile(r"/home/atta/\.hymem-siwc-lme-diagnostic-preflight-[a-z0-9_-]{8,}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


REMOTE = r'''
import base64,hashlib,json,os,re,stat,subprocess
from pathlib import Path
root=Path(PAYLOAD['root'])
need=lambda ok,code: (_ for _ in ()).throw(RuntimeError(code)) if not ok else None
need(re.fullmatch(r'/home/atta/\.hymem-siwc-lme-diagnostic-preflight-[a-z0-9_-]{8,}',str(root)) is not None,'root_invalid')
need(os.getuid()==1000 and root.is_dir() and not root.is_symlink(),'root_invalid')
need(root.stat().st_uid==1000 and stat.S_IMODE(root.stat().st_mode)==0o700,'root_invalid')
need({entry.name for entry in root.iterdir()}=={'candidate','code','source-map.json'},'root_not_fresh')
need(all(stat.S_ISDIR((root/name).lstat().st_mode) and not (root/name).is_symlink()
         for name in ('candidate','code')),'root_invalid')
need(stat.S_ISREG((root/'source-map.json').lstat().st_mode),'root_invalid')
data=base64.b64decode(PAYLOAD['source'],validate=True)
need(hashlib.sha256(data).hexdigest()==PAYLOAD['sha256'],'launcher_pin_invalid')
target=root/'siwc_lme_diagnostic_launch_v9.py'
need(not target.exists() and not target.is_symlink(),'launcher_existing')
fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
done=subprocess.run(['/home/atta/.hymem-siwc-runtime-v1/bin/python','-I','-B',
    str(target),'--prepare-root',str(root)],capture_output=True,timeout=180,
    env={'HOME':'/home/atta','PATH':'/usr/bin:/bin'})
try:result=json.loads(done.stdout)
except (ValueError,TypeError):result={}
unit=root.name.removeprefix('.')+'.service'
if (done.returncode==0 and type(result) is dict and result.get('prepared') is True
    and result.get('launched') is False and type(result.get('model_calls')) is int
    and result['model_calls']==0 and result.get('root')==str(root)
    and result.get('unit')==unit and type(result.get('receipt_sha256')) is str
    and re.fullmatch('[0-9a-f]{64}',result['receipt_sha256'])):
    print(json.dumps({'schema':'siwc-lme-diagnostic-source-install-v9',
        'prepared':True,'model_calls':0,'root':str(root),'unit':unit,
        'receipt_sha256':result['receipt_sha256']},sort_keys=True))
else:
    print(json.dumps({'status':'prepare_unverified','root':str(root)}))
    raise SystemExit(1)
'''


def _unique_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_field")
        output[key] = value
    return output


def project(value, *, root: str, returncode: int) -> dict | None:
    if returncode or type(value) is not dict or set(value) != {
            "schema", "prepared", "model_calls", "root", "unit", "receipt_sha256"}:
        return None
    unit = Path(root).name.removeprefix(".") + ".service"
    if (value["schema"] != SCHEMA or value["prepared"] is not True
            or type(value["model_calls"]) is not int or value["model_calls"] != 0
            or type(value["root"]) is not str or value["root"] != root
            or type(value["unit"]) is not str or value["unit"] != unit
            or type(value["receipt_sha256"]) is not str
            or HEX.fullmatch(value["receipt_sha256"]) is None):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--host", default="afrodite")
    args = parser.parse_args(argv)
    if ROOT.fullmatch(args.root) is None or re.fullmatch(r"[A-Za-z0-9_.-]+", args.host) is None or args.host.startswith("-"):
        print(json.dumps({"status": "installation_unverified"}))
        return 1
    path = Path(__file__).resolve().parents[2] / SOURCE
    if path.is_symlink() or not path.is_file():
        print(json.dumps({"status": "installation_unverified"}))
        return 1
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        print(json.dumps({"status": "installation_unverified"}))
        return 1
    payload = {"root": args.root, "source": base64.b64encode(raw).decode("ascii"),
        "sha256": SOURCE_SHA256}
    script = "PAYLOAD=" + repr(payload) + "\n" + REMOTE
    try:
        done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1", args.host, "python3 -I -B -"],
            input=script, text=True, capture_output=True, timeout=240)
        if len(done.stdout) > 8192:
            raise ValueError("remote_metadata_oversized")
        value = json.loads(done.stdout, object_pairs_hook=_unique_fields,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        print(json.dumps({"status": "installation_unverified", "never_repeat_automatically": True}))
        return 1
    safe = project(value, root=args.root, returncode=done.returncode)
    if safe is None:
        print(json.dumps({"status": "installation_unverified", "root": args.root,
            "never_repeat_automatically": True}))
        return 1
    print(json.dumps(safe, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
