"""Derive an inactive staged probe bundle from the accepted A/B bundle.

Source collection and host derivation are separately callable for offline review.
The complete preparation requires real, future runner/reader/replay sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

FAMILY = "luna-staged-probe"
REPO = Path(__file__).resolve().parents[2]
ACCEPTED = Path("/private/tmp/hymem-claim-task-accepted-KXzUiPyJ/bundle")
ACCEPTED_RECEIPT_SHA = "a3924f4b61679eba6f24cc5e2e86c9d8b64fd2f0897b4d2fd484b72656440888"
CANDIDATE = Path("/private/tmp/hymem-staged-v1-root-fZvNIRgs/candidate")
CANDIDATE_MAP = CANDIDATE.with_name("map.json")
CANDIDATE_MAP_SHA = "228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf"
MAPPING_SHA = "9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae"
EXTRACTION_IDENTITY = "hymem-extraction-contract-sha256-v1:94a1adfc028694e9b1868ec8712baff38d4d7a99a9d2e3547c66819711890f95"
LOCAL = {
    "tools/diagnostics/luna_staged_candidate_v1.py": "b1b12a9a4e420c48f16a94866c2bae6cd9576b89e41c87c05373594b28303d0e",
    "tools/diagnostics/luna_staged_core_v1.py": "a389f6e8a522d878a140234ac9c040f2e3daab77bfc7cbaeadf0f4d5b12676bd",
    "benchmarks/codex_subscription_staged_v1.py": "3b4df724cafa77cf0f4ae794672a29bdf2bf76e38079337a3bd82ecae1685824",
    "hymem/extraction/grounding_classification_v4.py": "37ab836c45cb306d5e67d17066aed107578a812d37ffc4f06fd47ecf5fd667d2",
    "hymem/extraction/grounding_staged_v1.py": "4862e4aedba5be756ea877d65a91b515a142b2fb46e2efb6f91c800e5096b3c9",
    "hymem/extraction/grounding_staged_gate_v1.py": "e843a3112a2ed0e7900f97b19a944d0a74fa458682c88a9504379c08c07deaa8",
}
PENDING = (
    "tools/diagnostics/luna_staged_run_v1.py",
    "tools/diagnostics/luna_staged_progress_v1.py",
    "tools/diagnostics/luna_staged_replay_v1.py",
)
HOST = "tools/diagnostics/luna_staged_host_v1.py"
OLD_HOST = "tools/diagnostics/luna_semantic_probe_host.py"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pinned(path: Path, expected: str) -> bytes:
    if not path.is_absolute() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("source_path_invalid")
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("source_pin_invalid:" + path.name)
    return raw


def one(raw: bytes, before: str, after: str) -> bytes:
    needle = before.encode()
    if raw.count(needle) != 1:
        raise ValueError("binding_count_invalid:" + before[:64])
    return raw.replace(needle, after.encode())


def validate_candidate(candidate: Path = CANDIDATE, stamp: Path = CANDIDATE_MAP) -> dict[str, str]:
    if (not candidate.is_absolute() or candidate != candidate.resolve() or
            not candidate.is_dir() or candidate.is_symlink() or
            not stamp.is_absolute() or stamp != stamp.resolve() or
            stamp.parent != candidate.parent):
        raise ValueError("candidate_boundary_invalid")
    raw = pinned(stamp, CANDIDATE_MAP_SHA)
    value = json.loads(raw)
    mapping = value.get("source_sha256")
    if (type(mapping) is not dict or len(mapping) != 514 or
            any(type(k) is not str or type(v) is not str or len(v) != 64 or
                Path(k).is_absolute() or ".." in Path(k).parts or
                any(c not in "0123456789abcdef" for c in v)
                for k, v in mapping.items()) or
            sha(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()) != MAPPING_SHA):
        raise ValueError("candidate_map_invalid")
    found = set()
    for path in candidate.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("candidate_path_invalid")
        if path.is_file():
            name = path.relative_to(candidate).as_posix()
            if name not in mapping or sha(path.read_bytes()) != mapping[name]:
                raise ValueError("candidate_inventory_drift")
            found.add(name)
    if found != set(mapping):
        raise ValueError("candidate_inventory_incomplete")
    return mapping


def collect_sources(repo: Path) -> dict[str, bytes]:
    if not repo.is_absolute() or repo != repo.resolve() or not repo.is_dir():
        raise ValueError("repo_boundary_invalid")
    receipt = json.loads(pinned(ACCEPTED / "derivation-receipt.json", ACCEPTED_RECEIPT_SHA))
    if (receipt.get("schema") != "luna-claim-task-probe-bundle-v1" or
            receipt.get("candidate_files") != 513 or receipt.get("model_calls") != 0 or
            receipt.get("launched") is not False or
            type(receipt.get("output_sha256")) is not dict):
        raise ValueError("accepted_receipt_invalid")
    output = {name: pinned(ACCEPTED / name, digest)
              for name, digest in receipt["output_sha256"].items()}
    code = {name.removeprefix("code/"): raw for name, raw in output.items()
            if name.startswith("code/")}
    for name, digest in LOCAL.items():
        code[name] = pinned(repo / name, digest)
    # In a staged root, the transport's project root is root/candidate.
    code["benchmarks/codex_subscription_staged_v1.py"] = one(
        code["benchmarks/codex_subscription_staged_v1.py"],
        "_ROOT = Path(__file__).resolve().parents[1]",
        "_ROOT = Path(__file__).resolve().parents[2] / \"candidate\"")
    return code


def derive_host(code_pins: dict[str, str]) -> bytes:
    """Return host bytes after rechecking carried source pins."""
    carried = collect_sources(REPO)
    if (type(code_pins) is not dict or set(code_pins) !=
            set(carried) | set(PENDING) or
            any(type(v) is not str or len(v) != 64 or
                any(c not in "0123456789abcdef" for c in v)
                for v in code_pins.values()) or
            any(code_pins[name] != sha(raw) for name, raw in carried.items())):
        raise ValueError("host_pins_invalid")
    base = pinned(ACCEPTED / "code" / OLD_HOST, code_pins[OLD_HOST])
    host = base
    host = host.replace(b"luna-claim-task-probe", b"luna-staged-probe")
    accepted_start = host.index(b"ACCEPTED = {\n")
    accepted_end = host.index(b"\nNEW =", accepted_start)
    bindings = "ACCEPTED = {\n" + "".join(
        f"    {name!r}: {code_pins[name]!r},\n" for name in sorted(code_pins)
        ) + "}"
    host = host[:accepted_start] + bindings.encode() + host[accepted_end:]
    host = one(host,
        "NEW = ('tools/diagnostics/luna_semantic_probe_host.py',\n"
        "       'tools/diagnostics/luna_semantic_probe_run.py',\n"
        "       'tools/diagnostics/luna_semantic_probe_progress.py')",
        "NEW = ('tools/diagnostics/luna_staged_host_v1.py',)")
    host = one(host, "INVENTORY_SHA = 'f63f656ee1aaa204c13a44f56cf1864596117f13f36189cc3b7ba05cd7646bab'",
               f"INVENTORY_SHA = '{CANDIDATE_MAP_SHA}'")
    host = one(host, "mapping) == 513", "mapping) == 514")
    host = one(host,
        "    pinned(root / 'candidate-source-map.json', INVENTORY_SHA)\n",
        "    need(regular(ACCEPTED_MAP), 'accepted_map_missing')\n"
        "    accepted_stamp = json.loads(ACCEPTED_MAP.read_bytes())\n"
        "    accepted_map = accepted_stamp.get('source_sha256', accepted_stamp)\n"
        "    need(type(accepted_map) is dict and len(accepted_map) == 510 and\n"
        "         builder.mapping_sha(accepted_map) ==\n"
        "         'e9ca47f85046d4ad980a8abef0302e1ec36a9a91cc9616bb043784bc152640fb' and\n"
        "         builder.inventory(ACCEPTED_CANDIDATE) == accepted_map,\n"
        "         'accepted_candidate_drift')\n"
        "    pinned(root / 'candidate-source-map.json', INVENTORY_SHA)\n")
    host = one(host, "'control_turns': 1, 'control_known_tokens': 100000, 'control_seconds': 240,\n"
                    "          'hybrid_turns': 0, 'hybrid_known_tokens': 0, 'hybrid_seconds': 0,\n"
                    "          'invocation_seconds': 120, 'workers': 1, 'units': 29,\n"
                    "          'ordinary_replays': 8, 'new_judgments_max': 29",
               "'unit_turns': 3, 'unit_known_tokens': 100000, 'unit_seconds': 240,\n"
               "          'invocation_seconds': 120, 'workers': 1, 'units': 8,\n"
               "          'ordinary_replays': 8, 'new_judgments_max': 29")
    host = one(host, "'schedule': {'control_pairs': 12, 'paired_turns': 24,\n"
                    "                         'nominated_prefers': 1, 'canary_turns': 4},",
               "'schedule': {'controls': [9, 12, 13, 17, 19, 21],\n"
               "                         'canaries': ['table', 'prose'], 'max_stages_per_unit': 3},")
    host = one(host, "'extraction_identity': 'hymem-extraction-contract-sha256-v1:21372a7db114466d4ad248c344ea4493606c858c9fd010c2f77456a9ad31b6c3'",
               f"'extraction_identity': '{EXTRACTION_IDENTITY}'")
    host = one(host, "'candidate': str(root / 'candidate'),", "'candidate': str(root / 'candidate'),")
    host = one(host, "source = root / 'code/tools/diagnostics/luna_classification_candidate_v3.py'",
               "source = root / 'code/tools/diagnostics/luna_staged_candidate_v1.py'")
    host = one(host, "proof['files'] == 513", "proof['files'] == 514")
    host = one(host, "str(root / 'code/tools/diagnostics/luna_claim_task_run_v1.py'),",
               "str(root / 'code/tools/diagnostics/luna_staged_run_v1.py'),")
    if b"'schema': 'luna-staged-probe-launch-v1'" not in host:
        raise ValueError("host_family_binding_invalid")
    return host


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(repo: Path, target: Path) -> dict:
    if (not repo.is_absolute() or repo != repo.resolve() or not repo.is_dir() or
            not target.is_absolute() or target.exists() or target.is_symlink() or
            target.parent != target.parent.resolve() or not target.parent.is_dir() or
            target.is_relative_to(repo) or target.is_relative_to(ACCEPTED) or
            target.is_relative_to(CANDIDATE.parent)):
        raise ValueError("output_boundary_invalid")
    validate_candidate()
    code = collect_sources(repo)
    for name in PENDING:
        path = repo / name
        if not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("pending_source_missing:" + name)
        code[name] = path.read_bytes()
        if not code[name]:
            raise ValueError("pending_source_empty:" + name)
    pins = {name: sha(raw) for name, raw in code.items()}
    code[HOST] = derive_host(pins)
    output = {"code/" + name: raw for name, raw in code.items()}
    receipt = {"schema": FAMILY + "-bundle-v1", "accepted_derivation_receipt_sha256": ACCEPTED_RECEIPT_SHA,
               "candidate_files": 514, "candidate_inventory_sha256": CANDIDATE_MAP_SHA,
               "candidate_mapping_sha256": MAPPING_SHA, "extraction_identity": EXTRACTION_IDENTITY,
               "input_sha256": {**{name: digest for name, digest in LOCAL.items()},
                                 **{name: pins[name] for name in PENDING}},
               "output_sha256": {name: sha(raw) for name, raw in sorted(output.items())},
               "model_calls": 0, "launched": False}
    target.mkdir(mode=0o700)
    for name, raw in output.items():
        _write(target / name, raw)
    _write(target / "derivation-receipt.json",
           (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return {"target": str(target), "receipt_sha256": sha((target / "derivation-receipt.json").read_bytes()),
            "code_files": len(code), "model_calls": 0, "launched": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.repo, args.target), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
