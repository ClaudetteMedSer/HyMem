"""Derive an inactive v2 semantic candidate from the pinned 508-file runtime.

The accepted v1 builder supplies the exact chunk/contract transformation. This
builder changes only the grounding source selected for the same runtime path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

V1_BUILDER_SHA256 = "a10ee5c5a1ba4f6a2694a88570a399c5fe081db02f0cb0ecfa5b92e5db06d7b2"
V2_GROUNDING = "hymem/extraction/grounding_v2.py"
V2_GROUNDING_SHA256 = "377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec"
ACCEPTED_MAP_SHA256 = "e9ca47f85046d4ad980a8abef0302e1ec36a9a91cc9616bb043784bc152640fb"
ACCEPTED_FILES = 510


def _load_v1():
    path = Path(__file__).with_name("luna_semantic_candidate.py")
    if not path.is_absolute() or path != path.resolve() or not path.is_file() or path.is_symlink():
        raise ValueError("v1_builder_source_invalid")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != V1_BUILDER_SHA256:
        raise ValueError("v1_builder_drift")
    spec = importlib.util.spec_from_file_location("_luna_semantic_candidate_v1_pinned", path)
    if spec is None or spec.loader is None:
        raise ValueError("v1_builder_load_invalid")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


v1 = _load_v1()


def _validate_path(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("paths_must_be_absolute")
    if path != path.resolve():
        raise ValueError("path_alias_or_symlink")


def _map(stamp: Path, count: int, digest: str) -> dict[str, str]:
    if not stamp.is_file() or stamp.is_symlink():
        raise ValueError("source_stamp_invalid")
    data = json.loads(stamp.read_text())
    mapping = data.get("source_sha256", data)
    if type(mapping) is not dict or len(mapping) != count or v1.mapping_sha(mapping) != digest:
        raise ValueError("source_inventory_identity")
    return mapping


def prepare(source: Path, source_stamp: Path, accepted: Path, accepted_stamp: Path,
            target: Path, target_stamp: Path, repo: Path) -> dict[str, str | int]:
    """Write a fresh candidate and stamp after validating all pinned inputs."""
    for path in (source, source_stamp, accepted, accepted_stamp, target, target_stamp, repo):
        _validate_path(path)
    if target.exists() or target.is_symlink() or target_stamp.exists() or target_stamp.is_symlink():
        raise ValueError("output_not_fresh")
    inputs = (source, accepted, repo)
    if any(not root.is_dir() or root.is_symlink() for root in inputs):
        raise ValueError("source_directory_invalid")
    if (target == target_stamp or target == source or target == accepted
            or any(target.is_relative_to(root) or target_stamp.is_relative_to(root) for root in inputs)
            or any(root.is_relative_to(target) for root in inputs)
            or target_stamp.is_relative_to(target)):
        raise ValueError("output_inside_source")
    if v1.sha(Path(v1.__file__).read_bytes()) != V1_BUILDER_SHA256:
        raise ValueError("v1_builder_drift")
    original = _map(source_stamp, v1.ORIGINAL_FILES, v1.ORIGINAL_MAP_SHA256)
    accepted_map = _map(accepted_stamp, ACCEPTED_FILES, ACCEPTED_MAP_SHA256)
    if v1.inventory(source) != original:
        raise ValueError("original_inventory_drift")
    if v1.inventory(accepted) != accepted_map:
        raise ValueError("accepted_inventory_drift")
    for relative, expected_hash in ((v1.GATE, v1.GATE_SHA256),
                                    (V2_GROUNDING, V2_GROUNDING_SHA256)):
        path = repo / relative
        if not path.is_file() or path.is_symlink() or v1.sha(path.read_bytes()) != expected_hash:
            raise ValueError("helper_source_drift")
    chunk = v1.derive_chunk((source / v1.CHUNK).read_bytes())
    contract = v1.derive_contract((source / v1.CONTRACT).read_bytes())
    grounding = (repo / V2_GROUNDING).read_bytes()
    gate = (repo / v1.GATE).read_bytes()
    modified = {v1.CHUNK: chunk, v1.CONTRACT: contract,
                v1.GROUNDING: grounding, v1.GATE: gate}
    derived = dict(original)
    derived.update({relative: v1.sha(data) for relative, data in modified.items()})
    expected = dict(accepted_map)
    expected[v1.GROUNDING] = V2_GROUNDING_SHA256
    if derived != expected:
        raise ValueError("accepted_delta_drift")
    shutil.copytree(source, target, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"))
    for relative, data in modified.items():
        (target / relative).write_bytes(data)
    if v1.inventory(target) != derived:
        raise ValueError("derived_inventory_drift")
    target_stamp.write_text(json.dumps({"source_sha256": derived}, sort_keys=True, indent=2) + "\n")
    return {"files": len(derived), "source_map_sha256": v1.ORIGINAL_MAP_SHA256,
            "accepted_map_sha256": ACCEPTED_MAP_SHA256,
            "derived_map_sha256": v1.mapping_sha(derived),
            "derived_stamp_sha256": v1.sha(target_stamp.read_bytes()),
            "chunk_sha256": derived[v1.CHUNK], "contract_sha256": derived[v1.CONTRACT],
            "grounding_sha256": derived[v1.GROUNDING], "gate_sha256": derived[v1.GATE]}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("source", "source_stamp", "accepted", "accepted_stamp", "target", "target_stamp", "repo"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.source_stamp, args.accepted,
                             args.accepted_stamp, args.target, args.target_stamp,
                             args.repo), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
