"""Fresh network-none R7 v63→v64 proof replay stage, never resuming v1.

This pinned adapter reuses reviewed v1 one-shot container controls while
changing the stage, container identity, v64 override pins and schema worker.
Import is local-only; install and launch remain separate explicit actions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "proof-replay-v2"
SELF = ROOT / "claim_conflict_proof_replay_v2_host.py"
WORKER = ROOT / "claim_conflict_proof_replay_v2.py"
OLD_REPLAY = ROOT / "claim_conflict_instrumented_replay.py"
OVERRIDES = ROOT / "overrides"
CANDIDATE = ROOT / "candidate"
CAPTURE = ROOT / "capture"
WORK = ROOT / "work"
V1_HOST = BASE / "proof-replay-v1/claim_conflict_proof_replay_host.py"
V1_HOST_SHA = "716c5da9f91e4cbd98cdf2ecc75338c42c4be963a25a7b31bd96ca8d04dcd5e2"
V1_AUDIT_WORKER = BASE / "proof-replay-v1/claim_conflict_proof_replay.py"
V1_AUDIT_WORKER_SHA = "014e045de6a5591eb87aabf0760247a69cd4d5a75f5bb7462e2d404caff7a9de"
OLD_REPLAY_SHA = "7b94a69f2a152f1a818ab7b5833b8d619734b3f7023de2dca89b4190f9013085"
WORKER_SHA = "8de6f54cc98abaaf5ac7f328ce2b4e4c5547128e580f3f0ecd82f8d6d353ec82"
LOCAL_FROZEN = Path("/private/tmp/hymem-r7-proof-v64.kEJ30D")
OVERRIDE_SHAS = {
    "hymem/core/db.py": "519258174b16317dcd463ba12747935ab9315e8aa15a16c6b8743436789eb84a",
    "hymem/core/schema.sql": "048b7d9e7058403197b13629749e3404c90c710aec94d9fadcd9eaed7ecde2b0",
    "hymem/core/migrations/064_local_claim_replay_proof.sql": "b4fc6c8def2d64a351aa43413f4a115c14b202ca4c0f8ac0d4c01b7c12eabd7a",
    "hymem/dreaming/phase1.py": "1037bf6d62c3981add3702f96d2ce5bd79d8ebc831f15b30f95bb3724cfd7217",
    "hymem/dreaming/evidence.py": "c921d035c4af5b6d0c5add8cb9eb89069b3dbef4447078470ae77dc0dbb36771",
    "hymem/dreaming/canonicalize.py": "73c1d656aaa471402732690712c9de86f82bea6e4a2bf6fef5677a1d1a4dcc18",
    "hymem/portability.py": "f4bc3226ce34b01a19610e1d43e4ad4ebd564a78993e8259c0cb78f630390ecb",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_overrides():
    if (set(OVERRIDE_SHAS) != {
            "hymem/core/db.py", "hymem/core/schema.sql",
            "hymem/core/migrations/064_local_claim_replay_proof.sql",
            "hymem/dreaming/phase1.py", "hymem/dreaming/evidence.py",
            "hymem/dreaming/canonicalize.py", "hymem/portability.py"}
            or not all(isinstance(value, str) and HEX64.fullmatch(value)
                       for value in OVERRIDE_SHAS.values())):
        raise RuntimeError("v64_override_pins_unreviewed")
    return OVERRIDE_SHAS


def controller(path=V1_HOST):
    if not path.is_file() or path.is_symlink() or sha(path) != V1_HOST_SHA:
        raise RuntimeError("v1_controller_dependency_pin_drift")
    spec = importlib.util.spec_from_file_location("proof_v2_pinned_v1_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SELF = SELF
    module.WORKER = WORKER
    module.OLD_REPLAY = OLD_REPLAY
    module.OVERRIDES = OVERRIDES
    module.CANDIDATE = CANDIDATE
    module.CAPTURE = CAPTURE
    module.WORK = WORK
    module.LOCAL_FROZEN = LOCAL_FROZEN
    module.OVERRIDE_SHAS = OVERRIDE_SHAS
    module.WORKER_SHA = WORKER_SHA
    module.reviewed_overrides = reviewed_overrides
    original_pins = module.pins
    original_configure = module.configure

    def pins(shared, alias, helper):
        receipt = original_pins(shared, alias, helper)
        helper.regular(V1_AUDIT_WORKER, mode=0o400)
        if sha(V1_AUDIT_WORKER) != V1_AUDIT_WORKER_SHA:
            raise RuntimeError("v1_audit_dependency_pin_drift")
        return {**receipt, "v1_audit_worker_sha256": V1_AUDIT_WORKER_SHA}

    def configure(helper):
        command, mounts = original_configure(helper)
        command[command.index("--name") + 1] = "hymem-proof-replay-v2"
        extra = (str(V1_AUDIT_WORKER), "/diag/claim_conflict_proof_replay_v1.py", False)
        mounts.append(extra)
        position = command.index("--workdir")
        command[position:position] = [
            "--mount", "type=bind,src=" + extra[0] + ",dst=" + extra[1] + ",readonly"]
        return command, mounts

    module.pins = pins
    module.configure = configure
    return module


def install():
    reviewed_overrides()
    local = Path(__file__)
    local_v1_host = local.with_name("claim_conflict_proof_replay_host.py")
    local_worker = local.with_name("claim_conflict_proof_replay_v2.py")
    local_replay = local.with_name("claim_conflict_instrumented_replay.py")
    if (sha(local_v1_host) != V1_HOST_SHA or sha(local_worker) != WORKER_SHA
            or sha(local_replay) != OLD_REPLAY_SHA):
        raise RuntimeError("local_diagnostic_pin_drift")
    module = controller(local_v1_host)
    bodies = {SELF.name: local.read_bytes(),
              WORKER.name: local_worker.read_bytes(),
              OLD_REPLAY.name: local_replay.read_bytes()}
    for relative, expected in reviewed_overrides().items():
        path = LOCAL_FROZEN / relative
        if not path.is_file() or path.is_symlink() or sha(path) != expected:
            raise RuntimeError("local_v64_override_pin_drift")
        bodies["overrides/" + relative] = path.read_bytes()
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
 path=root/name
 path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sys.stdin.buffer.read(1)==b''
print(json.dumps({'status':'uploaded'}))
"""
    script = "import json\nC=json.loads(" + repr(json.dumps(config)) + ")\n" + code
    uploaded = module.ssh_json("python3 -I -B -c " + shlex.quote(script),
                               b"".join(bodies.values()))
    if uploaded.get("status") != "uploaded":
        return uploaded
    return module.ssh_json("python3 -I -B " + shlex.quote(str(SELF))
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
            module = controller(Path(__file__).with_name(
                "claim_conflict_proof_replay_host.py"))
            result = module.ssh_json("python3 -I -B " + shlex.quote(str(SELF))
                                     + " remote-" + action)
        else:
            module = controller()
            result = module.remote(action)
    except BaseException:
        result = {"status": "operation_failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
