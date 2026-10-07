"""Build an inactive 514-file staged grounding candidate from pinned inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

PRIOR_BUILDER_SHA256 = "9595892e727661aee37aa5d4e0aa014935a7974e92f725c2216cc8f40e67f5e6"


def _load_prior():
    path = Path(__file__).with_name("luna_classification_candidate_v3.py")
    if not path.is_absolute() or path != path.resolve() or not path.is_file() or path.is_symlink():
        raise ValueError("prior_builder_source_invalid")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PRIOR_BUILDER_SHA256:
        raise ValueError("prior_builder_drift")
    spec = importlib.util.spec_from_file_location("_luna_classification_candidate_v3_pinned", path)
    if spec is None or spec.loader is None:
        raise ValueError("prior_builder_load_invalid")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


prior = _load_prior()
old = prior.prior.old
CLASSIFICATION = "hymem/extraction/grounding_classification_v4.py"
STAGED = "hymem/extraction/grounding_staged_v1.py"
STAGED_GATE = "hymem/extraction/grounding_staged_gate_v1.py"
HELPERS = {
    old.GROUNDING: old.GROUNDING_SHA256,
    old.GATE: old.GATE_SHA256,
    prior.prior.V2_GROUNDING: prior.prior.previous.V2_GROUNDING_SHA256,
    CLASSIFICATION: "37ab836c45cb306d5e67d17066aed107578a812d37ffc4f06fd47ecf5fd667d2",
    STAGED: "4862e4aedba5be756ea877d65a91b515a142b2fb46e2efb6f91c800e5096b3c9",
    STAGED_GATE: "e843a3112a2ed0e7900f97b19a944d0a74fa458682c88a9504379c08c07deaa8",
}


def _replace(value: str, before: str, after: str) -> str:
    return old.exact_replace(value, before, after)


def derive_chunk(original: bytes) -> bytes:
    text = prior.derive_chunk(original).decode("utf-8")
    text = _replace(text, "grounding_classification_gate_v3", "grounding_staged_gate_v1")
    text = _replace(text, "grounding_classification_v3 import ClassificationBatch",
                    "grounding_staged_gate_v1 import StageBatch")
    text = _replace(text,
        "def grounding_call(request: LLMRequest, batch: ClassificationBatch, recheck: bool) -> str:",
        "def grounding_call(request: LLMRequest, batch: StageBatch, stage: str, recheck: bool) -> str:")
    text = _replace(text,
        "                if recheck:\n"
        "                    return client.complete_grounding(request, batch)\n"
        "                return client.complete_grounding(request, batch)\n",
        "                return client.complete_stage(request, batch, stage, recheck)\n")
    return text.encode("utf-8")


def derive_contract(original: bytes) -> bytes:
    text = prior.derive_contract(original).decode("utf-8")
    text = _replace(text, "grounding_classification_v3 as grounding_module",
                    "grounding_classification_v4 as grounding_module")
    text = _replace(text, "grounding_classification_gate_v3 as grounding_gate_module",
                    "grounding_staged_gate_v1 as grounding_gate_module\n"
                    "from hymem.extraction import grounding_staged_v1 as staged_module")
    text = _replace(text,
        "_GROUNDING_IMPORTED_MODULES = (grounding_module, grounding_gate_module,\n"
        "    grounding_v2_module, grounding_v1_module, grounding_v1_gate_module)",
        "_GROUNDING_IMPORTED_MODULES = (grounding_module, grounding_gate_module, staged_module,\n"
        "    grounding_v2_module, grounding_v1_module, grounding_v1_gate_module)")
    text = _replace(text,
        "    grounding_module.build_output_schema, grounding_module.validate_request,\n"
        ")\n",
        "    grounding_module.build_output_schema, grounding_module.validate_request,\n"
        "    staged_module.build_original_request, staged_module.validate_original_request,\n"
        "    staged_module.parse_original_response, staged_module.build_original_output_schema,\n"
        "    staged_module.build_alternatives_request, staged_module.validate_alternatives_request,\n"
        "    staged_module.build_alternatives_output_schema, staged_module.parse_staged_responses,\n"
        ")\n")
    text = _replace(text,
        "    if _GROUNDING_IMPORTED_MODULES != (grounding_module, grounding_gate_module,\n"
        "            grounding_v2_module, grounding_v1_module, grounding_v1_gate_module):",
        "    if _GROUNDING_IMPORTED_MODULES != (grounding_module, grounding_gate_module, staged_module,\n"
        "            grounding_v2_module, grounding_v1_module, grounding_v1_gate_module):")
    text = _replace(text,
        "            grounding_module.build_output_schema, grounding_module.validate_request):",
        "            grounding_module.build_output_schema, grounding_module.validate_request,\n"
        "            staged_module.build_original_request, staged_module.validate_original_request,\n"
        "            staged_module.parse_original_response, staged_module.build_original_output_schema,\n"
        "            staged_module.build_alternatives_request, staged_module.validate_alternatives_request,\n"
        "            staged_module.build_alternatives_output_schema, staged_module.parse_staged_responses):")
    text = _replace(text,
        "            or grounding_gate_module._source_gate is not grounding_v1_gate_module\n"
        "            or chunk_module.ground_triples is not grounding_gate_module.ground_triples):",
        "            or grounding_gate_module._source_gate is not grounding_v1_gate_module\n"
        "            or staged_module.v4 is not grounding_module\n"
        "            or staged_module._build_v2 is not grounding_v2_module.build_grounding_request\n"
        "            or staged_module._parse_v2 is not grounding_v2_module.parse_grounding_response\n"
        "            or grounding_gate_module._staged is not staged_module\n"
        "            or grounding_gate_module.ClassificationBatch is not grounding_module.ClassificationBatch\n"
        "            or chunk_module.ground_triples is not grounding_gate_module.ground_triples):")
    text = _replace(text,
        "    if (grounding_gate_module.build_grounding_request is not grounding_module.build_grounding_request\n"
        "            or grounding_gate_module.parse_grounding_response is not grounding_module.parse_grounding_response):",
        "    if (grounding_gate_module._staged.build_original_request is not staged_module.build_original_request\n"
        "            or grounding_gate_module._staged.parse_staged_responses is not staged_module.parse_staged_responses):")
    text = _replace(text,
        '            "grounding_system": _text_digest(grounding_module._SYSTEM),\n',
        '            "grounding_system": _text_digest(grounding_module._SYSTEM),\n'
        '            "grounding_staged_original_system": _text_digest(staged_module._ORIGINAL_SYSTEM),\n'
        '            "grounding_staged_alternatives_system": _text_digest(staged_module._ALTERNATIVES_SYSTEM),\n')
    text = _replace(text,
        '            "grounding_gate": grounding_gate_module.GROUNDING_GATE_VERSION,\n',
        '            "grounding_gate": grounding_gate_module.GROUNDING_GATE_VERSION,\n'
        '            "grounding_staged_original_schema": staged_module.ORIGINAL_SCHEMA,\n'
        '            "grounding_staged_alternatives_schema": staged_module.ALTERNATIVES_SCHEMA,\n'
        '            "grounding_v4_contract": grounding_module.GROUNDING_CONTRACT_VERSION,\n')
    text = _replace(text,
        '            "grounding_gate": _module_source_digest(\n'
        '                grounding_gate_module, "ground_triples", "grounding_gate_support_integrity",\n'
        '            ),\n',
        '            "grounding_gate": _module_source_digest(\n'
        '                grounding_gate_module, "ground_triples", "grounding_gate_support_integrity",\n'
        '            ),\n'
        '            "grounding_staged": _module_source_digest(\n'
        '                staged_module, "build_original_request", "validate_original_request",\n'
        '                "parse_original_response", "build_original_output_schema",\n'
        '                "build_alternatives_request", "validate_alternatives_request",\n'
        '                "build_alternatives_output_schema", "parse_staged_responses",\n'
        '            ),\n')
    return text.encode("utf-8")


def prepare(source: Path, source_stamp: Path, accepted: Path, accepted_stamp: Path,
            target: Path, target_stamp: Path, repo: Path) -> dict[str, str | int]:
    """Validate all inputs before writing a fresh candidate and source map."""
    for path in (source, source_stamp, accepted, accepted_stamp, target, target_stamp, repo):
        prior.prior.previous._validate_path(path)
    if target.exists() or target.is_symlink() or target_stamp.exists() or target_stamp.is_symlink():
        raise ValueError("output_not_fresh")
    roots = (source, accepted, repo)
    if any(not root.is_dir() or root.is_symlink() for root in roots):
        raise ValueError("source_directory_invalid")
    if (target == target_stamp or any(target.is_relative_to(root) or target_stamp.is_relative_to(root)
            or root.is_relative_to(target) for root in roots) or target_stamp.is_relative_to(target)):
        raise ValueError("output_inside_source")
    if old.sha(Path(prior.__file__).read_bytes()) != PRIOR_BUILDER_SHA256:
        raise ValueError("prior_builder_drift")
    if old.sha(Path(prior.prior.__file__).read_bytes()) != prior.PRIOR_BUILDER_SHA256:
        raise ValueError("older_builder_drift")
    if old.sha(Path(old.__file__).read_bytes()) != prior.prior.OLD_BUILDER_SHA256:
        raise ValueError("old_builder_drift")
    original = prior.prior.previous._map(source_stamp, old.ORIGINAL_FILES, old.ORIGINAL_MAP_SHA256)
    accepted_map = prior.prior.previous._map(accepted_stamp, prior.prior.previous.ACCEPTED_FILES,
                                            prior.prior.previous.ACCEPTED_MAP_SHA256)
    if old.inventory(source) != original or old.inventory(accepted) != accepted_map:
        raise ValueError("source_inventory_drift")
    source_helpers = {}
    for relative, digest in HELPERS.items():
        path = repo / relative
        if not path.is_file() or path.is_symlink() or old.sha(path.read_bytes()) != digest:
            raise ValueError("helper_source_drift:" + relative)
        source_helpers[relative] = path.read_bytes()
    modified = {
        old.CHUNK: derive_chunk((source / old.CHUNK).read_bytes()),
        old.CONTRACT: derive_contract((source / old.CONTRACT).read_bytes()),
        **source_helpers,
    }
    derived = dict(original)
    derived.update({relative: old.sha(data) for relative, data in modified.items()})
    if len(derived) != old.ORIGINAL_FILES + 6:
        raise ValueError("derived_count_drift")
    if {relative for relative in original if derived[relative] != original[relative]} != {old.CHUNK, old.CONTRACT}:
        raise ValueError("ordinary_source_drift")
    shutil.copytree(source, target, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"))
    for relative, data in modified.items():
        (target / relative).write_bytes(data)
    if old.inventory(target) != derived:
        raise ValueError("derived_inventory_drift")
    target_stamp.write_text(json.dumps({"source_sha256": derived}, sort_keys=True, indent=2) + "\n")
    return {"files": len(derived), "source_map_sha256": old.ORIGINAL_MAP_SHA256,
            "accepted_map_sha256": prior.prior.previous.ACCEPTED_MAP_SHA256,
            "derived_map_sha256": old.mapping_sha(derived),
            "derived_stamp_sha256": old.sha(target_stamp.read_bytes()),
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
