"""Derive an inactive, source-bound semantic policy v2 diagnostic bundle.

This tool reads accepted sources as bytes and emits new source copies. It never
imports generated modules, contacts a provider, or launches the diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


FAMILY = "luna-semantic-policy-v2"
OLD_FAMILY = "luna-semantic-probe"
INVENTORY = Path("/private/tmp/hymem-semantic-policy-v2.aBQTUn6q/map.json")
CANDIDATE = INVENTORY.with_name("candidate")
ACCEPTED_CANDIDATE = Path("/home/atta/.hymem-luna-semantic-probe-30jkk7eg/candidate")
ACCEPTED_MAP = ACCEPTED_CANDIDATE.with_name("candidate-source-map.json")
INVENTORY_SHA = "64818ffcea5c52aef3d10142bfe9cfe7445a88280e46f3546bd01a5533fd3963"
EXTRACTION_IDENTITY = "hymem-extraction-contract-sha256-v1:dcfb634927ad7e2a555ff3054710239195effe0436b50a9d62fb7f6aee6c3180"
OLD_IDENTITY = "hymem-extraction-contract-sha256-v1:f349e2fa14d1778bc556869d346ca44183c025e779a919b3f15e82f7c3d78d46"
OLD_INVENTORY_SHA = "11ca4cdbba18e4b7e4b56d444062e39b789820b2f0055a32898bbd0b3a1e4664"
OLD_CANARY_SHA = "3d132415573cb46c2b15a69a56776ea69b3410b350a8fdb8ba6a1f8cc85658af"
OLD_CORE_SHA = "3ccb7ec12f93f8502fe8ed1448a071ac977dd334d17821f661913d634c27b334"
OLD_HOST_SHA = "7e190561958b126593a3e652ec2de35fb4cef27d4e5863284a75636abe531b07"
OLD_RUN_SHA = "60fc7fca900ef8320a410af2f5f01415f973f01907cf7c0456c0b4936611030a"
OLD_READER_SHA = "68f02223a561dd31a1a0417d8972ddafab7cbd26fc15500a9a2efc1b0cb9d355"
V2_GROUNDING_SHA = "377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec"
V2_BUILDER_SHA = "ed0c5c006d316fb0c762a2df6977313403e45fafdb914ea3c4f2f675a0165e3f"
ADAPTER_SHA = "e6ac48313364367756afbbaa7720e9d65047b12590237d5974f8324b188bf504"
REPLAY_SHA = "38806d0c42c60c1af89a581d96be5657ad5a4c2a3517fd53c28f354f9b39e6ea"

# The original 14-code bundle, frozen before any derivation.
INPUT_SHA = {
    "tools/diagnostics/luna_semantic_probe.py": OLD_CORE_SHA,
    "tools/diagnostics/luna_semantic_candidate.py": "a10ee5c5a1ba4f6a2694a88570a399c5fe081db02f0cb0ecfa5b92e5db06d7b2",
    "tools/diagnostics/luna_semantic_cases.py": "8876e94e6c5d3284d4d361f0238e0aef507288e2702b5df77ee706fd4c712d8a",
    "benchmarks/luna_semantic_canary.py": OLD_CANARY_SHA,
    "benchmarks/luna_semantic_stage_accounting.py": "4c654599a979e51aeb9f0985b091cd8ef38a639424f197c412da45e1f3270d30",
    "benchmarks/codex_subscription_warm_v3.py": "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d",
    "benchmarks/codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    "benchmarks/codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
    "benchmarks/codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
    "hymem/extraction/grounding.py": "dd1a49b56abf569b4476a0b735e67e72a86723bf88e3a4739998aceeabe19c18",
    "hymem/extraction/grounding_gate.py": "bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8",
    "tools/diagnostics/luna_semantic_probe_host.py": OLD_HOST_SHA,
    "tools/diagnostics/luna_semantic_probe_run.py": OLD_RUN_SHA,
    "tools/diagnostics/luna_semantic_probe_progress.py": OLD_READER_SHA,
    "hymem/extraction/grounding_v2.py": V2_GROUNDING_SHA,
    "tools/diagnostics/luna_semantic_candidate_v2.py": V2_BUILDER_SHA,
    "tools/diagnostics/luna_semantic_probe_adapter_v2.py": ADAPTER_SHA,
    "tools/diagnostics/luna_semantic_verdict_replay_root.py": REPLAY_SHA,
}
CODE = tuple(k for k in INPUT_SHA if not k.endswith(("probe_adapter_v2.py", "verdict_replay_root.py")))
EXACT_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(path: Path, expected: str) -> bytes:
    if (not path.is_absolute() or path != path.resolve() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode)):
        raise ValueError("source_path_invalid")
    data = path.read_bytes()
    if _sha(data) != expected:
        raise ValueError("source_pin_invalid:" + path.name)
    return data


def _replace(raw: bytes, old: str, new: str, count: int) -> bytes:
    left, right = old.encode(), new.encode()
    if raw.count(left) != count:
        raise ValueError(f"replacement_count_invalid:{old}:{raw.count(left)}:{count}")
    return raw.replace(left, right)


def _family(raw: bytes, count: int) -> bytes:
    return _replace(raw, OLD_FAMILY, FAMILY, count)


def _regular_tree(root: Path) -> None:
    if not root.is_absolute() or root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise ValueError("candidate_path_invalid")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("candidate_path_invalid")


def _validate_candidate() -> None:
    raw = _source(INVENTORY, INVENTORY_SHA)
    stamp = json.loads(raw)
    if type(stamp) is not dict:
        raise ValueError("candidate_map_invalid")
    mapping = stamp.get("source_sha256")
    if type(mapping) is not dict or len(mapping) != 510:
        raise ValueError("candidate_map_invalid")
    _regular_tree(CANDIDATE)
    actual = {}
    for path in CANDIDATE.rglob("*"):
        if path.is_file():
            actual[path.relative_to(CANDIDATE).as_posix()] = _sha(path.read_bytes())
    if (actual != mapping or mapping.get("hymem/extraction/grounding.py") != V2_GROUNDING_SHA
            or any(type(key) is not str or type(value) is not str or
                   EXACT_HEX.fullmatch(value) is None or Path(key).is_absolute() or
                   ".." in Path(key).parts for key, value in mapping.items())):
        raise ValueError("candidate_inventory_invalid")


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(repo: Path, target: Path) -> dict:
    """Create target/code, target/adapter-v2.py and target/verdict-replay.py.

    Both paths must be absolute and canonical; target must not exist. All input
    pins and the complete inactive candidate inventory pass before target creation.
    """
    if (not repo.is_absolute() or repo != repo.resolve() or not repo.is_dir()
            or repo.is_symlink() or not target.is_absolute() or
            target.parent != target.parent.resolve() or target.exists() or
            target.is_symlink() or not target.parent.is_dir() or
            target.is_relative_to(repo) or repo.is_relative_to(target) or
            target.is_relative_to(INVENTORY.parent) or
            INVENTORY.parent.is_relative_to(target)):
        raise ValueError("output_boundary_invalid")
    sources = {relative: _source(repo / relative, digest)
               for relative, digest in INPUT_SHA.items()}
    _validate_candidate()

    canary = sources["benchmarks/luna_semantic_canary.py"]
    canary = _replace(canary, "luna-semantic-canary-v1", "luna-semantic-canary-v2", 1)
    canary = _replace(canary, '"source-grounding-v1"', '"source-grounding-v2"', 1)
    canary = _replace(canary, OLD_IDENTITY, EXTRACTION_IDENTITY, 1)
    canary_sha = _sha(canary)

    core = sources["tools/diagnostics/luna_semantic_probe.py"]
    core = _family(core, 1)
    core = _replace(core, OLD_CANARY_SHA, canary_sha, 1)
    core = _replace(core, OLD_INVENTORY_SHA, INVENTORY_SHA, 1)
    core_sha = _sha(core)

    host = sources["tools/diagnostics/luna_semantic_probe_host.py"]
    host = _family(host, 4)
    host = _replace(host, OLD_CORE_SHA, core_sha, 1)
    host = _replace(host, OLD_CANARY_SHA, canary_sha, 1)
    host = _replace(host, OLD_INVENTORY_SHA, INVENTORY_SHA, 1)
    host = _replace(host, OLD_IDENTITY, EXTRACTION_IDENTITY, 1)
    anchor = "    'tools/diagnostics/luna_semantic_candidate.py': 'a10ee5c5a1ba4f6a2694a88570a399c5fe081db02f0cb0ecfa5b92e5db06d7b2',"
    added = (anchor + "\n" +
             "    'tools/diagnostics/luna_semantic_candidate_v2.py': '" + V2_BUILDER_SHA + "',\n" +
             "    'hymem/extraction/grounding_v2.py': '" + V2_GROUNDING_SHA + "',")
    host = _replace(host, anchor, added, 1)
    host = _replace(host, "source = root / 'code/tools/diagnostics/luna_semantic_candidate.py'",
                    "source = root / 'code/tools/diagnostics/luna_semantic_candidate_v2.py'", 1)
    host = _replace(host,
        "proof = builder.prepare(OLD_CANDIDATE, OLD_MAP, root / 'candidate',\n"
        "                            root / 'candidate-source-map.json', root / 'code')",
        "proof = builder.prepare(OLD_CANDIDATE, OLD_MAP, ACCEPTED_CANDIDATE, ACCEPTED_MAP,\n"
        "                            root / 'candidate', root / 'candidate-source-map.json', root / 'code')", 1)
    host = _replace(host, "OLD_MAP = OBSERVED / 'headless-grounding-source-map.json'",
                    "OLD_MAP = OBSERVED / 'headless-grounding-source-map.json'\n"
                    "ACCEPTED_CANDIDATE = Path('/home/atta/.hymem-luna-semantic-probe-30jkk7eg/candidate')\n"
                    "ACCEPTED_MAP = ACCEPTED_CANDIDATE.with_name('candidate-source-map.json')", 1)
    host_sha = _sha(host)

    run = sources["tools/diagnostics/luna_semantic_probe_run.py"]
    run = _family(run, 5)
    run = _replace(run, OLD_CORE_SHA, core_sha, 1)
    run_sha = _sha(run)
    reader = sources["tools/diagnostics/luna_semantic_probe_progress.py"]
    reader = _family(reader, 6)
    reader_sha = _sha(reader)
    adapter = sources["tools/diagnostics/luna_semantic_probe_adapter_v2.py"]
    adapter = _family(adapter, 12)
    for old, new in ((OLD_HOST_SHA, host_sha), (OLD_RUN_SHA, run_sha), (OLD_READER_SHA, reader_sha)):
        adapter = _replace(adapter, old, new, 1)
    replay = sources["tools/diagnostics/luna_semantic_verdict_replay_root.py"]
    replay = _replace(replay, "luna-semantic-probe-v1", FAMILY + "-v1", 1)
    replay = _replace(replay, "luna-semantic-verdict-root-replay-v1",
                      FAMILY + "-verdict-root-replay-v1", 1)

    generated = dict((relative, sources[relative]) for relative in CODE)
    generated.update({
        "benchmarks/luna_semantic_canary.py": canary,
        "tools/diagnostics/luna_semantic_probe.py": core,
        "tools/diagnostics/luna_semantic_probe_host.py": host,
        "tools/diagnostics/luna_semantic_probe_run.py": run,
        "tools/diagnostics/luna_semantic_probe_progress.py": reader,
    })
    if len(generated) != 16:
        raise ValueError("generated_file_count_invalid")
    output_sha = {"code/" + relative: _sha(raw) for relative, raw in generated.items()}
    output_sha.update({"adapter-v2.py": _sha(adapter), "verdict-replay.py": _sha(replay)})
    receipt = {"schema": "luna-semantic-policy-bundle-v2", "candidate_inventory_sha256": INVENTORY_SHA,
               "candidate_files": 510, "extraction_identity": EXTRACTION_IDENTITY,
               "input_sha256": dict(sorted(INPUT_SHA.items())),
               "output_sha256": dict(sorted(output_sha.items())),
               "unchanged_code": sorted(set(CODE) - {
                   "benchmarks/luna_semantic_canary.py", "tools/diagnostics/luna_semantic_probe.py",
                   "tools/diagnostics/luna_semantic_probe_host.py", "tools/diagnostics/luna_semantic_probe_run.py",
                   "tools/diagnostics/luna_semantic_probe_progress.py"}),
               "model_calls": 0, "launched": False}
    target.mkdir(mode=0o700)
    for relative, raw in generated.items():
        _write(target / "code" / relative, raw)
    _write(target / "adapter-v2.py", adapter)
    _write(target / "verdict-replay.py", replay)
    _write(target / "derivation-receipt.json",
           (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return {"target": str(target), "code_files": len(generated),
            "receipt_sha256": _sha((target / "derivation-receipt.json").read_bytes()),
            "output_sha256": output_sha, "model_calls": 0, "launched": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.repo, args.target), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
