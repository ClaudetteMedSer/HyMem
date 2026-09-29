"""Fresh one-shot alias replay stage with the corrected cache-key worker.

This is a pinned adapter for the reviewed v1 controller. It changes only the
private stage paths and worker bytes; v1's failed receipt remains immutable.
Importing this module never contacts the host.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "alias-replay-v2"
SELF = ROOT / "claim_conflict_alias_replay_v2_host.py"
WORKER = ROOT / "claim_conflict_instrumented_replay.py"
OVERRIDE = ROOT / "canonicalize.py"
V1_HOST = BASE / "alias-replay-v1/claim_conflict_alias_replay_host.py"
V1_HOST_SHA = "c998a7179e89a55e15be4f76f996a416e9771456c38718c8f622a49d4ba120b2"
WORKER_SHA = "7b94a69f2a152f1a818ab7b5833b8d619734b3f7023de2dca89b4190f9013085"


def v1_controller():
    if (not V1_HOST.is_file() or V1_HOST.is_symlink()
            or hashlib.sha256(V1_HOST.read_bytes()).hexdigest() != V1_HOST_SHA):
        raise RuntimeError("v1_host_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location("alias_replay_v2_pinned_v1", V1_HOST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SELF = SELF
    module.WORKER = WORKER
    module.OVERRIDE = OVERRIDE
    module.CAPTURE = ROOT / "capture"
    module.BASELINE_CANDIDATE = ROOT / "baseline"
    module.FIXED_CANDIDATE = ROOT / "fixed"
    module.WORK = ROOT / "work"
    return module


def pin_worker(controller):
    if not WORKER.is_file() or WORKER.is_symlink() or controller.sha(WORKER) != WORKER_SHA:
        raise RuntimeError("replay_worker_pin_drift")


def installed(shared, helper):
    controller = v1_controller()
    pin_worker(controller)
    return controller.installed(shared, helper)


def configure(helper, mode):
    controller = v1_controller()
    pin_worker(controller)
    return controller.configure(helper, mode)


def inspect(helper, cid, mode, mounts):
    controller = v1_controller()
    pin_worker(controller)
    return controller.inspect(helper, cid, mode, mounts)


def verdict(mode, metadata, receipt):
    controller = v1_controller()
    pin_worker(controller)
    return controller.verdict(mode, metadata, receipt)


def install():
    """Upload code-only exact bytes; the remote install remains separate."""
    local_host = Path(__file__)
    local_worker = local_host.with_name("claim_conflict_instrumented_replay.py")
    local_v1 = local_host.with_name("claim_conflict_alias_replay_host.py")
    v1_sha = hashlib.sha256(local_v1.read_bytes()).hexdigest()
    worker_sha = hashlib.sha256(local_worker.read_bytes()).hexdigest()
    if v1_sha != V1_HOST_SHA or worker_sha != WORKER_SHA:
        raise RuntimeError("local_replay_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location("alias_replay_v2_local_v1", local_v1)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    if (not controller.LOCAL_CANONICALIZE.is_file()
            or controller.LOCAL_CANONICALIZE.is_symlink()
            or controller.sha(controller.LOCAL_CANONICALIZE) != controller.CANONICALIZE_SHA):
        raise RuntimeError("local_override_pin_drift")
    bodies = {SELF.name: local_host.read_bytes(),
              WORKER.name: local_worker.read_bytes(),
              OVERRIDE.name: controller.LOCAL_CANONICALIZE.read_bytes()}
    config = {"root": str(ROOT), "files": {
        name: {"size": len(raw), "sha": hashlib.sha256(raw).hexdigest()}
        for name, raw in bodies.items()}}
    code = """import hashlib,json,os,pathlib,sys
root=pathlib.Path(C['root'])
assert os.geteuid()==1000 and not root.exists()
root.mkdir(mode=0o700)
for name,item in C['files'].items():
 raw=sys.stdin.buffer.read(item['size'])
 assert len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha']
 fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sys.stdin.buffer.read(1)==b''
print(json.dumps({'status':'uploaded'}))
"""
    script = "import json\nC=json.loads(" + repr(json.dumps(config)) + ")\n" + code
    uploaded = controller.ssh_json("python3 -I -B -c " + shlex.quote(script),
                                   b"".join(bodies.values()))
    if uploaded.get("status") != "uploaded":
        return uploaded
    return controller.ssh_json("python3 -I -B " + shlex.quote(str(SELF))
                               + " remote-install")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "launch", "status",
                                           "remote-install", "remote-launch",
                                           "remote-status", "supervise"))
    action = parser.parse_args().action
    try:
        if action == "install":
            result = install()
        elif action in ("launch", "status"):
            local_worker = Path(__file__).with_name("claim_conflict_instrumented_replay.py")
            if hashlib.sha256(local_worker.read_bytes()).hexdigest() != WORKER_SHA:
                raise RuntimeError("local_replay_worker_pin_drift")
            local_v1 = Path(__file__).with_name("claim_conflict_alias_replay_host.py")
            spec = importlib.util.spec_from_file_location("alias_replay_v2_local_v1", local_v1)
            controller = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(controller)
            result = controller.ssh_json("python3 -I -B " + shlex.quote(str(SELF))
                                         + " remote-" + action)
        else:
            controller = v1_controller()
            pin_worker(controller)
            result = controller.remote(action)
    except BaseException:
        result = {"status": "operation_failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
