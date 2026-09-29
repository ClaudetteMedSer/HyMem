"""Stage fresh diagnostics before completion; seal/run require explicit ready pins.

No action is performed on import. `install` stages code only. `seal --ready
--seal-json FILE` verifies the finished worker before creating the exclusive
seal. `run --ready --seal-sha256 HEX` launches one network-none diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = "/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/cold-replay-dream-v1/episode-shadow-dream-v1"
STAGE = ROOT + "/episode-shadow-postflight-v3"
HERE = Path(__file__).parent
PINS = {
    "claim_conflict_episode_shadow_semantic_public.json": "5f359e9b3302971f582966b6a3b016cf0d3e6c375d20ff26fa840a412c5cdbc0",
    "claim_conflict_episode_shadow_embedding_public.json": "1e1e69e036999106d4ea5bd7cf65aaca5c3cc416741a08ad4d3a0374059ae71a",
    "claim_conflict_episode_shadow_postflight_v3_host.py": "a8f94bb18e193d83a54e354f25d88cc46cd798fd6de47750be521ac2a1bcf5c0",
    "claim_conflict_episode_shadow_postflight_v3.py": "715b498b3a9c4d4b5bcf164eee78e6e60d901aa1527318b6932c43c3bcde6bc1",
    "claim_conflict_store_audit.py": "e2efe365c5aedbbe88d86d521dd37b821759afc5fb21c21a6662e6a8ce567f42",
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
SEAL_FIELDS = {"host_sha256", "install_sha256", "result_sha256", "live_container_id", "source_sha256"}

REMOTE_LAUNCH = r'''
import hashlib,os,pathlib,stat,sys
path=pathlib.Path(sys.argv[1]);pin=sys.argv[2]
assert path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode)==0o400
assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
os.execv(sys.executable,[sys.executable,'-I','-B',str(path)]+sys.argv[3:])
'''

REMOTE_INSTALL = r'''
import hashlib,json,os,pathlib,stat,sys
config=json.load(sys.stdin)
root=pathlib.Path(config['root']);stage=root/'episode-shadow-postflight-v3'
assert os.geteuid()==1000 and root.is_dir() and not root.is_symlink()
assert stat.S_IMODE(root.stat().st_mode)==0o700 and not stage.exists() and not stage.is_symlink()
assert set(config['files'])=={'claim_conflict_episode_shadow_postflight_v3_host.py','claim_conflict_episode_shadow_postflight_v3.py','claim_conflict_store_audit.py','claim_conflict_episode_shadow_embedding_public.json','claim_conflict_episode_shadow_semantic_public.json'}
stage.mkdir(mode=0o700)
for name,item in config['files'].items():
    raw=item['text'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256']
    fd=os.open(stage/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps({'status':'installed_not_launched','files':len(config['files'])}))
'''


def files():
    result = {}
    for name, pin in PINS.items():
        path = HERE / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("local_postflight_file_invalid")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError("local_postflight_pin_drift")
        result[name] = {"sha256": pin, "text": raw.decode()}
    return result


def dispatch(args):
    ssh = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "ConnectionAttempts=1", "afrodite"]
    if args.action == "install":
        command = ["python3", "-I", "-B", "-c", REMOTE_INSTALL]
        payload = json.dumps({"root": ROOT, "files": files()})
    else:
        if args.ready is not True:
            raise ValueError("explicit_ready_required")
        command = ["python3", "-I", "-B", STAGE + "/claim_conflict_episode_shadow_postflight_v3_host.py",
                   "--seal", STAGE + "/seal.json", "--ready"]
        if args.action == "seal":
            if args.seal_json is None or args.seal_json.is_symlink():
                raise ValueError("explicit_seal_required")
            value = json.loads(args.seal_json.read_bytes())
            if not isinstance(value, dict) or set(value) != SEAL_FIELDS or not all(
                isinstance(v, str) and HEX.fullmatch(v) for v in value.values()
            ):
                raise ValueError("seal_fields_invalid")
            payload = json.dumps(value, sort_keys=True)
            seal_sha = hashlib.sha256(payload.encode()).hexdigest()
            command += ["--prepare-seal", "--seal-sha256", seal_sha]
        else:
            if not isinstance(args.seal_sha256, str) or not HEX.fullmatch(args.seal_sha256):
                raise ValueError("explicit_seal_pin_required")
            payload = None
            command += ["--seal-sha256", args.seal_sha256]
        name = "claim_conflict_episode_shadow_postflight_v3_host.py"
        command = ["python3", "-I", "-B", "-c", REMOTE_LAUNCH,
                   STAGE + "/" + name, PINS[name]] + command[4:]
    return subprocess.run(ssh + [shlex.join(command)], input=payload,
                          text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "seal", "run"))
    parser.add_argument("--ready", action="store_true")
    parser.add_argument("--seal-json", type=Path)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    result = dispatch(args)
    # Host output is safe counts/status/hash only; raw stderr remains private.
    if result.returncode:
        print(json.dumps({"status": "error"}))
    else:
        value = json.loads(result.stdout)
        print(json.dumps(value, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print('{"status":"error"}')
        raise SystemExit(1)
