"""Build private unapproved bundle templates from root-provided evidence only.

No subprocess/network calls. Root spec is 0600 JSON: candidate_files (exact481
map), gates (label -> {path,sha256}), runtime_env_sha256, container_id,image_id.
Required gate labels are full_suite/private_paid_postflight/deployment_start/
production_preflight/role_profiles. Normalized receipts must have verified=true
and the exact candidate_sha256. The builder never sets root_reviewed=true.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

INPUTS = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-production-inputs-20260926-v1")
HOST_STAGE = INPUTS.parent / "hymem-v64-production-dream-20260926-v1"
STAGE = Path("/home/node/.hermes/benchmarks") / HOST_STAGE.name
CANDIDATE = "5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576"
SESSION = "8066d4f38539a067ff89c6cc7b130f38511e83af181321fe9efb58e8b38b698f"
GENERATION = "hymem-phase1-generation-v1:6075085e12e32e1b790e49b99b8c3bb50718b18be0f28762d1582c58ee8e35eb"
CODE = {
    "claim_conflict_v64_production_dream.py": "cb4378ce7b416f0db00ebdf2a6db6bdb99de2750446e84bd7f147c2f8bd128b6",
    "claim_conflict_v64_production_dream_host.py": "a1076c5a43114570d171b7dd26283926432f1a70c5c63eac501af70fafa4e952",
    "claim_conflict_episode_shadow_dream_instrumented.py": "e095f534da3457c5f24f7688a8f2eb6bc6b4d6f8981b630abec26c2de49739d8",
    "supervised_invocation.py": "9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc",
}


def need(ok):
    if not ok:
        raise RuntimeError("bundle_evidence_invalid")


def raw_private(path):
    info = Path(path).lstat()
    need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600)
    return Path(path).read_bytes()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build(spec_path):
    spec = json.loads(raw_private(spec_path))
    files = spec["candidate_files"]
    need(len(files) == 481 and digest(json.dumps(files, sort_keys=True,
         separators=(",", ":")).encode()) == CANDIDATE)
    envhash = spec["runtime_env_sha256"]
    need(type(envhash) is str and len(envhash) == 64 and all(c in "0123456789abcdef" for c in envhash))
    need(INPUTS.is_dir() and not INPUTS.is_symlink() and stat.S_IMODE(INPUTS.stat().st_mode) == 0o700)
    for name, pin in CODE.items():
        need(digest(raw_private(INPUTS / name)) == pin)
    payloads = {}
    gates = {}
    for label in ("full_suite", "private_paid_postflight", "deployment_start", "production_preflight", "role_profiles"):
        item = spec["gates"][label]
        raw = raw_private(item["path"])
        need(digest(raw) == item["sha256"])
        receipt = json.loads(raw)
        need(receipt.get("verified") is True and receipt.get("candidate_sha256") == CANDIDATE)
        name = label + "-normalized.json"
        payloads[name] = raw
        gates[label] = {"path": str(STAGE / name), "sha256": digest(raw)}
    launch = {"version": "schema64-production-targeted-dream-v1", "root_reviewed": False,
        "candidate_sha256": CANDIDATE, "candidate_files": files, "session_sha256": SESSION,
        "generation": GENERATION, "source": "/home/node/HyMem", "stage": str(STAGE),
        "runtime_env_sha256": envhash, "gates": gates,
        "bounds": {"completions": 128, "llm_http": 384, "embedding_http": 512,
                   "total_http": 896, "deadline_seconds": 2700,
                   "embedding_texts": 16, "embedding_chars": 128000, "embedding_utf8_bytes": 512000}}
    for label, name in (("worker", "claim_conflict_v64_production_dream.py"),
                        ("meter", "claim_conflict_episode_shadow_dream_instrumented.py"),
                        ("supervisor", "supervised_invocation.py")):
        launch[label] = {"path": str(STAGE / name), "sha256": CODE[name]}
    payloads["launch-manifest.json"] = encode(launch)
    inventory = {**CODE, **{name: digest(raw) for name, raw in payloads.items()}}
    host = {"version": "schema64-production-host-v1", "root_reviewed": False,
        "host_stage": str(HOST_STAGE), "container_stage": str(STAGE), "container": "hermes-1",
        "container_id": spec["container_id"], "image_id": spec["image_id"],
        "source_host": "/opt/stacks/hermes/instance1/home/HyMem",
        "mounts": {"/home/node": {"source": "/opt/stacks/hermes/instance1/home", "rw": True, "type": "bind"}},
        "interpreter": "/home/node/hymem-env/bin/python3", "files": inventory,
        "manifest": "launch-manifest.json", "manifest_sha256": inventory["launch-manifest.json"],
        "controller_sha256": CODE["claim_conflict_v64_production_dream_host.py"]}
    # Validate all evidence before the first write. Templates remain unapproved;
    # root must review/edit both approval fields and reseal every dependent SHA.
    for name, raw in payloads.items():
        write(INPUTS / name, raw)
    write(INPUTS / "reviewed-bundle-template.json", encode(host))
    return {"status": "templates_unapproved", "files": len(inventory),
            "manifest_sha256": inventory["launch-manifest.json"], "bundle_sha256": digest(encode(host))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    try:
        result = build(args.spec)
    except BaseException:
        result = {"status": "failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(1 if result["status"] == "failed_inspect_before_retry" else 0)
