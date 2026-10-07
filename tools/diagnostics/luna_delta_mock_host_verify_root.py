"""Root-verified, metadata-only execution of the credential-free delta mock."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess


WRAPPER_SHA256 = "975307a31609990eff66b945bd1421dcbb57986c4374e934532daf6fda93d15b"
SCHEMA = "luna-delta-mock-host-root-v1"
SOURCE_SCHEMA = "luna-delta-runtime-mock-v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
COUNTS = ("http_requests", "turn_starts", "turn_completed", "delta_notifications",
          "item_started", "item_completed", "final_items", "usage_updates",
          "error_notifications")
FLAGS = ("same_turn_identity", "final_digest_matches", "usage_positive",
         "cleanup_verified", "mock_boundary_valid")
CASES = ("baseline", "opt_out", "error_control")


def project(value, digest):
    if (type(value) is not dict or value.get("schema") != SCHEMA
            or value.get("source_sha256") != digest
            or type(value.get("network_policy_verified")) is not bool
            or type(value.get("cleanup_verified")) is not bool
            or value.get("status") not in {"not_started", "observed", "verification_failed"}):
        raise ValueError("host_metadata_invalid")
    out = {key: value[key] for key in ("schema", "source_sha256", "status",
           "network_policy_verified", "cleanup_verified")}
    observed = value.get("observation")
    if observed is None:
        return out
    if (type(observed) is not dict or observed.get("schema") != SOURCE_SCHEMA
            or type(observed.get("verified")) is not bool):
        raise ValueError("mock_metadata_invalid")
    rows = observed.get("results")
    if type(rows) is not list or not 0 <= len(rows) <= 3:
        raise ValueError("mock_rows_invalid")
    safe_rows = []
    for index, row in enumerate(rows):
        if (type(row) is not dict or row.get("case") != CASES[index]
                or row.get("status") not in {"completed", "error_observed", "unverified"}
                or any(type(row.get(key)) is not int or not 0 <= row[key] <= 6000 for key in COUNTS)
                or any(type(row.get(key)) is not bool for key in FLAGS)):
            raise ValueError("mock_row_invalid")
        safe = {key: row[key] for key in ("case", "status", *COUNTS, *FLAGS)}
        safe["failure_present"] = row.get("failure_code") is not None
        usage_digest = row.get("usage_digest")
        if usage_digest is not None and (type(usage_digest) is not str or not HEX.fullmatch(usage_digest)):
            raise ValueError("usage_digest_invalid")
        safe["usage_digest"] = usage_digest
        safe_rows.append(safe)
    out["observation"] = {"schema": SOURCE_SCHEMA, "verified": observed["verified"],
                          "results": safe_rows}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    if HEX.fullmatch(args.source_sha256) is None:
        raise ValueError("source_pin_invalid")
    here = Path(__file__).resolve().parent
    source = (here / "luna_delta_runtime_mock_v1.py").read_bytes()
    prior = (here / "luna_retry_mock_host_verify_root.py").read_bytes()
    if (hashlib.sha256(source).hexdigest() != args.source_sha256
            or hashlib.sha256(prior).hexdigest() != WRAPPER_SHA256):
        raise ValueError("source_integrity_failure")
    constants = [node.value for node in ast.parse(prior).body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "REMOTE"]
    if len(constants) != 1:
        raise ValueError("wrapper_shape_invalid")
    remote = ast.literal_eval(constants[0])
    old = "luna-retry-mock-host-root-v1"
    if type(remote) is not str or remote.count(old) != 1:
        raise ValueError("wrapper_shape_invalid")
    remote = remote.replace(old, SCHEMA)
    payload = ("SOURCE=" + repr(base64.b64encode(source).decode("ascii"))
        + "\nEXPECTED=" + repr(args.source_sha256)
        + "\nSOURCE_SCHEMA=" + repr(SOURCE_SCHEMA) + "\n" + remote)
    try:
        call = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1", "afrodite", "/usr/bin/python3 -I -B -"],
            input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=190)
        if call.returncode != 0 or len(call.stdout) > 64000:
            raise ValueError("host_execution_unverified")
        safe = project(json.loads(call.stdout), args.source_sha256)
        print(json.dumps(safe, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        print(json.dumps({"schema": SCHEMA, "status": "host_verification_unavailable",
                          "never_repeat_automatically": True}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
