"""Derive a fresh 508-file grounding candidate from the pinned R9 source.

Only three reviewed insertions are allowed. The original source is never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ORIGINAL_PROMPT_SHA256 = "3ed757777177d2f7a0afb0669de76be6b6ea1681a6ceed5c3a10775e4676b22c"
ORIGINAL_MAP_SHA256 = "35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51"
PROMPT_RELATIVE = "hymem/extraction/prompts/__init__.py"
INSERTIONS = (
    (b'_CHUNK_EXTRACTION_SYSTEM_TEMPLATE = """',
     b'PREDICATE_GROUNDING_VERSION = "hymem-predicate-grounding-v1"\n\n_CHUNK_EXTRACTION_SYSTEM_TEMPLATE = """'),
    (b'- polarity is -1 only when the speaker negates or retracts the relationship',
     b'- Predicate grounding ({predicate_grounding_version}): support each predicate\n'
     b'  independently from the cited source record. A preference does not establish\n'
     b'  use, ownership, or deployment; use does not establish preference. Intent,\n'
     b'  recommendations, and hypotheses do not establish actual adoption. Preserve\n'
     b'  both predicates when each is supported, including implicit language that\n'
     b'  clearly entails the relationship. Preserve exact positive and negative\n'
     b'  claims from their respective sources.\n'
     b'- polarity is -1 only when the speaker negates or retracts the relationship'),
    (b'        predicates=", ".join(ALLOWED_PREDICATES),\n    )',
     b'        predicates=", ".join(ALLOWED_PREDICATES),\n'
     b'        predicate_grounding_version=PREDICATE_GROUNDING_VERSION,\n    )'),
)
PILOT_SHA256 = "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0"


def pilot_module():
    path = Path(__file__).resolve().with_name("luna_subscription_pilot.py")
    if sha(path.read_bytes()) != PILOT_SHA256:
        raise ValueError("pilot_source_drift")
    spec = importlib.util.spec_from_file_location("pinned_grounding_pilot", path)
    if spec is None or spec.loader is None:
        raise ValueError("pilot_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive_prompt(original: bytes) -> bytes:
    if sha(original) != ORIGINAL_PROMPT_SHA256:
        raise ValueError("original_prompt_drift")
    derived = original
    boundaries = ((b'# behavioral signals" are preserved for prompt routing in tests.',
                   b'CHUNK_EXTRACTION_SYSTEM = build_chunk_extraction_system()'),
                  (b'_CHUNK_EXTRACTION_SYSTEM_TEMPLATE = """',
                   b'CHUNK_EXTRACTION_SYSTEM = build_chunk_extraction_system()'),
                  (b'def build_chunk_extraction_system() -> str:',
                   b'CHUNK_EXTRACTION_SYSTEM = build_chunk_extraction_system()'))
    for (before, after), (start, end) in zip(INSERTIONS, boundaries):
        if derived.count(start) != 1 or derived.count(end) != 1:
            raise ValueError("insertion_region_drift")
        left = derived.index(start)
        right = derived.index(end)
        region = derived[left:right]
        if region.count(before) != 1:
            raise ValueError("insertion_anchor_drift")
        derived = derived[:left] + region.replace(before, after, 1) + derived[right:]
    return derived


def source_map(stamp: dict) -> dict[str, str]:
    mapping = stamp.get("source_sha256", stamp)
    if type(mapping) is not dict or len(mapping) != 508:
        raise ValueError("original_inventory_invalid")
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    if sha(encoded) != ORIGINAL_MAP_SHA256 or mapping.get(PROMPT_RELATIVE) != ORIGINAL_PROMPT_SHA256:
        raise ValueError("original_inventory_drift")
    return mapping


def derived_map(mapping: dict[str, str], grounded: bytes) -> tuple[dict[str, str], str]:
    updated = dict(mapping)
    updated[PROMPT_RELATIVE] = sha(grounded)
    encoded = json.dumps(updated, sort_keys=True, separators=(",", ":")).encode()
    return updated, sha(encoded)


def verify_derived(original_candidate: Path, candidate: Path, inventory: Path,
                   original_inventory: Path) -> dict:
    """Verify both complete inventories and byte-exact prompt derivation."""
    pilot = pilot_module()
    original = source_map(json.loads(original_inventory.read_text()))
    source = original_candidate
    if not source.is_dir():
        raise ValueError("original_candidate_missing")
    pilot.verify_inventory(source, original_inventory, sha(original_inventory.read_bytes()))
    grounded = derive_prompt((source / PROMPT_RELATIVE).read_bytes())
    mapping, map_sha = derived_map(original, grounded)
    if (candidate / PROMPT_RELATIVE).read_bytes() != grounded:
        raise ValueError("grounded_prompt_drift")
    pilot.verify_inventory(candidate, inventory, sha(inventory.read_bytes()),
                           expected_map_sha256=map_sha)
    if json.loads(inventory.read_text()).get("source_sha256") != mapping:
        raise ValueError("derived_inventory_drift")
    return {"source_files": 508, "original_map_sha256": ORIGINAL_MAP_SHA256,
            "grounded_map_sha256": map_sha, "grounded_prompt_sha256": sha(grounded),
            "grounded_inventory_sha256": sha(inventory.read_bytes())}


def prepare(original_candidate: Path, original_inventory: Path,
            derived_candidate: Path, derived_inventory: Path) -> dict:
    """Copy a verified candidate to a fresh path, then write one changed file."""
    pilot = pilot_module()
    if not all(p.is_absolute() for p in (original_candidate, original_inventory,
                                           derived_candidate, derived_inventory)):
        raise ValueError("paths_must_be_absolute")
    if (derived_candidate.exists() or derived_candidate.is_symlink()
            or derived_inventory.exists() or derived_inventory.is_symlink()):
        raise ValueError("output_not_fresh")
    if (derived_candidate == original_candidate
            or derived_candidate.is_relative_to(original_candidate)
            or derived_inventory.is_relative_to(original_candidate)
            or derived_inventory.is_relative_to(derived_candidate)):
        raise ValueError("output_inside_source")
    pilot.verify_inventory(original_candidate, original_inventory,
                           sha(original_inventory.read_bytes()))
    original = source_map(json.loads(original_inventory.read_text()))
    grounded = derive_prompt((original_candidate / PROMPT_RELATIVE).read_bytes())
    mapping, map_sha = derived_map(original, grounded)
    shutil.copytree(original_candidate, derived_candidate, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"))
    (derived_candidate / PROMPT_RELATIVE).write_bytes(grounded)
    stamp = {"source_sha256": mapping}
    derived_inventory.write_text(json.dumps(stamp, sort_keys=True, indent=2) + "\n")
    pilot.verify_inventory(derived_candidate, derived_inventory,
                           sha(derived_inventory.read_bytes()),
                           expected_map_sha256=map_sha)
    return {"source_files": 508, "original_map_sha256": ORIGINAL_MAP_SHA256,
            "grounded_map_sha256": map_sha, "grounded_prompt_sha256": sha(grounded),
            "grounded_inventory_sha256": sha(derived_inventory.read_bytes())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "verify"))
    parser.add_argument("--original-candidate", type=Path, required=True)
    parser.add_argument("--original-inventory", type=Path, required=True)
    parser.add_argument("--derived-candidate", type=Path, required=True)
    parser.add_argument("--derived-inventory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        value = prepare(args.original_candidate, args.original_inventory,
                        args.derived_candidate, args.derived_inventory)
    else:
        value = verify_derived(args.original_candidate, args.derived_candidate,
                               args.derived_inventory, args.original_inventory)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
