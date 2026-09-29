"""Explicit code-only install; no stage, launch or status operations exist."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = "/opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-production-inputs-20260926-v1"
FILES = {
    "claim_conflict_v64_production_dream.py": "cb4378ce7b416f0db00ebdf2a6db6bdb99de2750446e84bd7f147c2f8bd128b6",
    "claim_conflict_v64_production_dream_host.py": "a1076c5a43114570d171b7dd26283926432f1a70c5c63eac501af70fafa4e952",
    "claim_conflict_episode_shadow_dream_instrumented.py": "e095f534da3457c5f24f7688a8f2eb6bc6b4d6f8981b630abec26c2de49739d8",
    "supervised_invocation.py": "9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc",
}
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite")
REMOTE = r'''
import hashlib,json,os,pathlib,stat,sys
root=pathlib.Path(C['root'])
def need(ok):
 if not ok: raise RuntimeError('install_admission_failed')
need(os.geteuid()==1000 and not os.path.lexists(root))
need(root.parent.is_dir() and all(not p.is_symlink() for p in (root.parent,*root.parent.parents)))
payload={}
for name,item in C['files'].items():
 raw=sys.stdin.buffer.read(item['size'])
 need(len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha256'])
 payload[name]=raw
need(sys.stdin.buffer.read(1)==b'')
root.mkdir(mode=0o700)
for name,raw in payload.items():
 fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as stream:
  stream.write(raw);stream.flush();os.fsync(stream.fileno())
fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(fd)
finally:os.close(fd)
print(json.dumps({'status':'installed_not_staged_not_launched','files':len(payload)}))
'''


def local_payload(directory=None):
    directory = Path(directory) if directory else Path(__file__).parent
    result = {}
    for name, digest in FILES.items():
        path = directory / name
        if directory == Path(__file__).parent and name == "supervised_invocation.py":
            path = directory.parents[1] / "benchmarks" / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("code_input_invalid")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError("code_pin_drift")
        result[name] = raw
    return result


def install(directory=None):
    pieces = local_payload(directory)
    config = {"root": ROOT, "files": {name: {"size": len(raw), "sha256": FILES[name]}
              for name, raw in pieces.items()}}
    script = "C=" + repr(config) + "\n" + REMOTE
    done = subprocess.run([*SSH, "python3 -I -B -c " + shlex.quote(script)],
        input=b"".join(pieces.values()), capture_output=True, timeout=180)
    if done.returncode != 0 or not 0 < len(done.stdout) <= 1024:
        return {"status": "failed_inspect_before_retry"}
    result = json.loads(done.stdout)
    if result != {"status": "installed_not_staged_not_launched", "files": 4}:
        raise RuntimeError("install_receipt_invalid")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install",))
    parser.parse_args()
    try:
        result = install()
    except BaseException:
        result = {"status": "failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(1 if result["status"] == "failed_inspect_before_retry" else 0)
