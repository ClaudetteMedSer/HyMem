"""Root-only, no-inference verification in a disposable network-none container."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess


REMOTE = r'''
import base64, hashlib, json, subprocess, sys, uuid
image = "sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5"
binary = "/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex"
name = "hymem-luna-offline-retry-" + uuid.uuid4().hex[:12]
source = base64.b64decode(SOURCE)
if hashlib.sha256(source).hexdigest() != EXPECTED:
    raise SystemExit("source_integrity_failure")
result = {"schema":"luna-retry-mock-host-root-v1", "source_sha256":EXPECTED,
          "network_policy_verified":False, "cleanup_verified":False,
          "status":"not_started"}
def docker(*args, **kwargs):
    return subprocess.run(["docker", *args], stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          timeout=10, **kwargs)
try:
    created = docker("create", "-i", "--name", name, "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "256", "--memory", "1g", "--cpus", "1",
        "--user", "1000:1000", "--tmpfs", "/tmp:rw,nosuid,nodev,size=268435456",
        "--mount", "type=bind,src="+binary+",dst="+binary+",readonly",
        "--entrypoint", "/usr/bin/env", image, "-i",
        "PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp", "TMPDIR=/tmp",
        "python3", "-I", "-B", "-", "--run")
    if created.returncode:
        raise ValueError("container_create")
    inspected = docker("inspect", name)
    item = json.loads(inspected.stdout)[0]
    host = item["HostConfig"]
    mounts = item["Mounts"]
    valid = (item["Image"] == image and host["NetworkMode"] == "none"
        and host["ReadonlyRootfs"] is True and host["PidsLimit"] == 256
        and host["Memory"] == 1073741824 and host["NanoCpus"] == 1000000000
        and host["RestartPolicy"]["Name"] == "no"
        and host["CapDrop"] == ["ALL"]
        and any(value.startswith("no-new-privileges") for value in host["SecurityOpt"])
        and item["Config"]["User"] == "1000:1000"
        and len(mounts) == 1 and mounts[0]["Source"] == binary
        and mounts[0]["Destination"] == binary and mounts[0]["RW"] is False)
    if not valid:
        raise ValueError("container_policy")
    result["network_policy_verified"] = True
    child = subprocess.Popen(["docker", "start", "--attach", "--interactive", name],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        output, _ = child.communicate(source, timeout=145)
        if child.returncode != 0 or len(output) > 64_000:
            raise ValueError("mock_execution")
        observation = json.loads(output)
        if type(observation) is not dict or observation.get("schema") != SOURCE_SCHEMA:
            raise ValueError("mock_shape")
        result["observation"] = observation
        result["status"] = "observed"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
except (ValueError, OSError, subprocess.SubprocessError, KeyError, IndexError):
    result["status"] = "verification_failed"
finally:
    try:
        docker("rm", "--force", name)
        remaining = docker("ps", "--all", "--quiet", "--filter", "name=^/"+name+"$")
        result["cleanup_verified"] = remaining.returncode == 0 and not remaining.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        result["cleanup_verified"] = False
print(json.dumps(result, sort_keys=True))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--mock-version", choices=("1", "2", "3"), default="1")
    args = parser.parse_args()
    source = Path(__file__).with_name(f"luna_retry_runtime_mock_v{args.mock_version}.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != args.source_sha256:
        raise SystemExit("local_source_integrity_failure")
    payload = ("SOURCE = " + repr(base64.b64encode(source).decode())
               + "\nEXPECTED = " + repr(args.source_sha256)
               + "\nSOURCE_SCHEMA = " + repr("luna-retry-runtime-mock-v" + args.mock_version)
               + "\n" + REMOTE).encode()
    call = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "afrodite", "/usr/bin/python3 -I -B -"],
        input=payload, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=190)
    if call.returncode != 0 or len(call.stdout) > 64_000:
        raise SystemExit("host_verification_unavailable")
    print(json.dumps(json.loads(call.stdout), sort_keys=True))


if __name__ == "__main__":
    main()
