"""Build an inactive classification-v2 candidate from pinned offline sources.

The frozen 508-file source and accepted 510-file inventory are read-only inputs.
Only chunk.py and contract.py change; five pinned helpers are copied byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

PRIOR_BUILDER_SHA256 = "cb0f904bf67d12c1099150adbcee68dbdbe78c63069337d77c14c296bcb3b8ca"


def _load_prior():
    path = Path(__file__).with_name("luna_classification_candidate.py")
    if not path.is_absolute() or path != path.resolve() or not path.is_file() or path.is_symlink():
        raise ValueError("prior_builder_source_invalid")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PRIOR_BUILDER_SHA256:
        raise ValueError("prior_builder_drift")
    spec = importlib.util.spec_from_file_location("_luna_classification_candidate_pinned", path)
    if spec is None or spec.loader is None:
        raise ValueError("prior_builder_load_invalid")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


prior = _load_prior()
CLASSIFICATION = "hymem/extraction/grounding_classification_v2.py"
CLASSIFICATION_GATE = "hymem/extraction/grounding_classification_gate_v2.py"
HELPERS = {
    prior.old.GROUNDING: prior.old.GROUNDING_SHA256,
    prior.old.GATE: prior.old.GATE_SHA256,
    prior.V2_GROUNDING: prior.previous.V2_GROUNDING_SHA256,
    CLASSIFICATION: "2520e4825df9e4d6f403a301451bb9701e052d9a331b5e31a9e106cbb69ad3c9",
    CLASSIFICATION_GATE: "2b4ba5d0590bf3a27d57f3fa52e1d84011917b368088628df62b6faa3664d307",
}


def _replace_imports(data: bytes) -> bytes:
    text = data.decode("utf-8")
    text = prior.old.exact_replace(
        text, "hymem.extraction.grounding_classification_gate_v1",
        "hymem.extraction.grounding_classification_gate_v2")
    text = prior.old.exact_replace(
        text, "hymem.extraction.grounding_classification_v1",
        "hymem.extraction.grounding_classification_v2")
    return text.encode("utf-8")


def derive_chunk(original: bytes) -> bytes:
    return _replace_imports(prior.derive_chunk(original))


def derive_contract(original: bytes) -> bytes:
    text = prior.derive_contract(original).decode("utf-8")
    text = prior.old.exact_replace(text, "import grounding_classification_v1 as",
                                   "import grounding_classification_v2 as")
    text = prior.old.exact_replace(text, "import grounding_classification_gate_v1 as",
                                   "import grounding_classification_gate_v2 as")
    return text.encode("utf-8")


def prepare(source: Path, source_stamp: Path, accepted: Path, accepted_stamp: Path,
            target: Path, target_stamp: Path, repo: Path) -> dict[str, str | int]:
    """Validate every input before writing a fresh 513-file candidate and map."""
    for path in (source, source_stamp, accepted, accepted_stamp, target, target_stamp, repo):
        prior.previous._validate_path(path)
    if target.exists() or target.is_symlink() or target_stamp.exists() or target_stamp.is_symlink():
        raise ValueError("output_not_fresh")
    roots = (source, accepted, repo)
    if any(not root.is_dir() or root.is_symlink() for root in roots):
        raise ValueError("source_directory_invalid")
    if (target == target_stamp or any(target.is_relative_to(root) or target_stamp.is_relative_to(root)
            or root.is_relative_to(target) for root in roots) or target_stamp.is_relative_to(target)):
        raise ValueError("output_inside_source")
    if prior.old.sha(Path(prior.__file__).read_bytes()) != PRIOR_BUILDER_SHA256:
        raise ValueError("prior_builder_drift")
    if prior.old.sha(Path(prior.old.__file__).read_bytes()) != prior.OLD_BUILDER_SHA256:
        raise ValueError("old_builder_drift")
    original = prior.previous._map(source_stamp, prior.old.ORIGINAL_FILES,
                                   prior.old.ORIGINAL_MAP_SHA256)
    accepted_map = prior.previous._map(accepted_stamp, prior.previous.ACCEPTED_FILES,
                                       prior.previous.ACCEPTED_MAP_SHA256)
    if prior.old.inventory(source) != original or prior.old.inventory(accepted) != accepted_map:
        raise ValueError("source_inventory_drift")
    source_helpers = {}
    for relative, digest in HELPERS.items():
        path = repo / relative
        if not path.is_file() or path.is_symlink() or prior.old.sha(path.read_bytes()) != digest:
            raise ValueError("helper_source_drift:" + relative)
        source_helpers[relative] = path.read_bytes()
    modified = {
        prior.old.CHUNK: derive_chunk((source / prior.old.CHUNK).read_bytes()),
        prior.old.CONTRACT: derive_contract((source / prior.old.CONTRACT).read_bytes()),
        **source_helpers,
    }
    derived = dict(original)
    derived.update({relative: prior.old.sha(data) for relative, data in modified.items()})
    if len(derived) != prior.old.ORIGINAL_FILES + 5:
        raise ValueError("derived_count_drift")
    if {relative for relative in original if derived[relative] != original[relative]} != {
            prior.old.CHUNK, prior.old.CONTRACT}:
        raise ValueError("ordinary_source_drift")
    shutil.copytree(source, target, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"))
    for relative, data in modified.items():
        (target / relative).write_bytes(data)
    if prior.old.inventory(target) != derived:
        raise ValueError("derived_inventory_drift")
    target_stamp.write_text(json.dumps({"source_sha256": derived}, sort_keys=True, indent=2) + "\n")
    return {"files": len(derived), "source_map_sha256": prior.old.ORIGINAL_MAP_SHA256,
            "accepted_map_sha256": prior.previous.ACCEPTED_MAP_SHA256,
            "derived_map_sha256": prior.old.mapping_sha(derived),
            "derived_stamp_sha256": prior.old.sha(target_stamp.read_bytes()),
            **{relative.rsplit("/", 1)[-1].removesuffix(".py") + "_sha256": derived[relative]
               for relative in modified}}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("source", "source_stamp", "accepted", "accepted_stamp", "target", "target_stamp", "repo"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.source_stamp, args.accepted,
                             args.accepted_stamp, args.target, args.target_stamp, args.repo), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
