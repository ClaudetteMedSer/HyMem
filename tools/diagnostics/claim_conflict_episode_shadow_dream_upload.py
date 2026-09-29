"""Explicit local install/launch/status entry point; importing never contacts SSH."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = "/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/cold-replay-dream-v1/episode-shadow-dream-v1"
HOST = "claim_conflict_episode_shadow_dream_host.py"
FILES = {
    HOST: "38e3bebe14d117549ac8d58533b1286ffb8669b600369e9caccda2e8cb326821",
    "claim_conflict_episode_shadow_dream.py": "6edaa699b4d37c0337d41e607ebf4ee9f1c3ef028aaf3e7b9d1c1b30c5fbd380",
    "claim_conflict_episode_shadow_dream_instrumented.py": "e095f534da3457c5f24f7688a8f2eb6bc6b4d6f8981b630abec26c2de49739d8",
    "claim_conflict_episode_shadow_dream_supervisor.py": "f486300d9292f86f8777b400358de7936e99ca07d65dfff85e5730b23ba82a9d",
}
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite")
REMOTE = """
import hashlib,json,os,pathlib,stat,sys
root=pathlib.Path(C['root']);parent=root.parent
def need(ok):
 if not ok: raise RuntimeError('admission_failed')
def pin(path,digest):
 need(path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest)
need(os.geteuid()==1000 and not os.path.lexists(root))
need(parent.is_dir() and stat.S_IMODE(parent.stat().st_mode)==0o700)
need(all(not p.is_symlink() for p in (parent,*parent.parents)))
pin(parent/'claim_conflict_v64_dream_host.py','38df85589fbc561e287bc75329b3ab785af001dd0f6bc3024cbb09f629820f0c')
pin(parent/'episode-shadow-replay-v1/result.json','9a05dfe4b6383b263e3d510a77e3550aab7f140036d538d0cdebf650a75ac823')
pin(parent.parent/'capture-next-v1/reference.sqlite','7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0')
payload={}
for name,item in C['files'].items():
 raw=sys.stdin.buffer.read(item['size'])
 need(len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha'])
 payload[name]=raw
need(sys.stdin.buffer.read(1)==b'')
root.mkdir(mode=0o700)
for name,raw in payload.items():
 fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb') as stream:
  stream.write(raw);stream.flush();os.fsync(stream.fileno())
print('{"status":"uploaded"}')
"""
STATUSES = {"uploaded", "installed_not_launched", "detached_supervisor_started",
            "running_or_requires_inspection", "completed", "failed",
            "operation_failed_inspect_before_retry", "unknown_inspect_before_retry"}
EXEC_REMOTE = """
import hashlib,os,pathlib,stat,sys
root=pathlib.Path(C['root'])
def need(ok):
 if not ok: raise RuntimeError('remote_pin_admission_failed')
need(os.geteuid()==1000 and root.is_dir() and stat.S_IMODE(root.stat().st_mode)==0o700)
need(all(not p.is_symlink() for p in (root,*root.parents)))
for name,digest in C['files'].items():
 path=root/name
 info=path.lstat()
 need(stat.S_ISREG(info.st_mode) and not path.is_symlink() and stat.S_IMODE(info.st_mode)==0o400)
 need(hashlib.sha256(path.read_bytes()).hexdigest()==digest)
os.execv(sys.executable,[sys.executable,'-I','-B',str(root/C['host']),C['action']])
"""


def local_payload(directory=None):
    directory = Path(directory) if directory is not None else Path(__file__).parent
    pieces = {}
    for name, pin in FILES.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("local_diagnostic_invalid")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise RuntimeError("local_diagnostic_pin_drift")
        pieces[name] = raw
    return pieces


def projection(raw):
    if not isinstance(raw, dict) or raw.get("status") not in STATUSES:
        raise ValueError("remote_status_invalid")
    safe = {"status": raw["status"]}
    for name in ("pid", "paid_live_runs_started"):
        if name in raw:
            if type(raw[name]) is not int or not 0 <= raw[name] <= 100000000:
                raise ValueError("remote_count_invalid")
            safe[name] = raw[name]
    for name in ("candidate_sha256", "host_sha256", "worker_sha256",
                 "reference_sha256", "proof_result_sha256"):
        if name in raw:
            if not isinstance(raw[name], str) or not re.fullmatch("[0-9a-f]{64}", raw[name]):
                raise ValueError("remote_hash_invalid")
            safe[name] = raw[name]
    return safe


def ssh_json(command, data=b""):
    try:
        done = subprocess.run([*SSH, command], input=data,
                              capture_output=True, timeout=180)
        if done.returncode != 0 or not 0 < len(done.stdout) <= 16384:
            raise ValueError("remote_operation_failed")
        return projection(json.loads(done.stdout))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {"status": "unknown_inspect_before_retry"}


def run(action):
    # Verify all four local files before any action, including read-only status.
    pieces = local_payload()
    if action == "install":
        config = {"root": ROOT, "files": {
            name: {"size": len(raw), "sha": FILES[name]}
            for name, raw in pieces.items()}}
        script = "C=" + repr(config) + "\n" + REMOTE
        result = ssh_json("python3 -I -B -c " + shlex.quote(script),
                          b"".join(pieces.values()))
        if result["status"] != "uploaded":
            return result
        remote_action = "remote-install"
    elif action in ("launch", "status"):
        remote_action = "remote-" + action
    else:
        raise ValueError("invalid_action")
    config = {"root": ROOT, "files": FILES, "host": HOST, "action": remote_action}
    script = "C=" + repr(config) + "\n" + EXEC_REMOTE
    return ssh_json("python3 -I -B -c " + shlex.quote(script))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "launch", "status"))
    action = parser.parse_args().action
    try:
        result = run(action)
    except BaseException:
        result = {"status": "operation_failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))
    return 1 if result["status"] in {
        "failed", "operation_failed_inspect_before_retry",
        "unknown_inspect_before_retry"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
