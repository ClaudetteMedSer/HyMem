"""Build an inactive, source-bound classification candidate offline.

The frozen 508-file runtime and accepted v2 inventory are inputs, never edited.
Only the old builder's pinned chunk/contract transformation is reused; the
classification delta is applied with one-occurrence byte-exact anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

V2_BUILDER_SHA256 = "ed0c5c006d316fb0c762a2df6977313403e45fafdb914ea3c4f2f675a0165e3f"


def _load_previous():
    path = Path(__file__).with_name("luna_semantic_candidate_v2.py")
    if not path.is_absolute() or path != path.resolve() or not path.is_file() or path.is_symlink():
        raise ValueError("v2_builder_source_invalid")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != V2_BUILDER_SHA256:
        raise ValueError("v2_builder_drift")
    spec = importlib.util.spec_from_file_location("_luna_semantic_candidate_v2_pinned", path)
    if spec is None or spec.loader is None:
        raise ValueError("v2_builder_load_invalid")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


previous = _load_previous()
old = previous.v1
OLD_BUILDER_SHA256 = previous.V1_BUILDER_SHA256
CLASSIFICATION = "hymem/extraction/grounding_classification_v1.py"
CLASSIFICATION_GATE = "hymem/extraction/grounding_classification_gate_v1.py"
V2_GROUNDING = "hymem/extraction/grounding_v2.py"
HELPERS = {
    old.GROUNDING: old.GROUNDING_SHA256,
    old.GATE: old.GATE_SHA256,
    V2_GROUNDING: previous.V2_GROUNDING_SHA256,
    CLASSIFICATION: "7c62c6e58305a5b7be256825a52a8cc4b9119888011bf18fe267b18dccab2c06",
    CLASSIFICATION_GATE: "0c676c9e10f9dfb9fa005be4ec7b75c799d4e69e37d61ae0953a52e86ed59081",
}


def derive_chunk(original: bytes) -> bytes:
    text = old.derive_chunk(original).decode("utf-8")
    text = old.exact_replace(text,
        "from hymem.extraction.grounding_gate import (\n",
        "from hymem.extraction.grounding_classification_gate_v1 import (\n")
    text = old.exact_replace(text,
        "from hymem.extraction.jsonio import (\n",
        "from hymem.extraction.grounding_classification_v1 import ClassificationBatch\n"
        "from hymem.extraction.jsonio import (\n")
    text = old.exact_replace(text,
        "    def grounding_call(request: LLMRequest, recheck: bool) -> str:\n",
        "    def grounding_call(request: LLMRequest, batch: ClassificationBatch, recheck: bool) -> str:\n")
    text = old.exact_replace(text,
        "                return client.complete(request)\n        except Exception as exc:\n"
        "            log.warning(\"chunk_grounding.call_failure error_type=%s\", type(exc).__name__)\n"
        "            raise GroundingGateError(\"provider:call_failed\") from exc\n",
        "                if recheck:\n"
        "                    return client.complete_grounding(request, batch)\n"
        "                return client.complete_grounding(request, batch)\n"
        "        except Exception as exc:\n"
        "            log.warning(\"chunk_grounding.call_failure error_type=%s\", type(exc).__name__)\n"
        "            raise GroundingGateError(\"provider:call_failed\") from exc\n")
    return text.encode("utf-8")


def derive_contract(original: bytes) -> bytes:
    text = old.derive_contract(original).decode("utf-8")
    text = old.exact_replace(text,
        "from hymem.extraction import grounding as grounding_module\n"
        "from hymem.extraction import grounding_gate as grounding_gate_module\n",
        "from hymem.extraction import grounding_classification_v1 as grounding_module\n"
        "from hymem.extraction import grounding_classification_gate_v1 as grounding_gate_module\n"
        "from hymem.extraction import grounding_v2 as grounding_v2_module\n"
        "from hymem.extraction import grounding as grounding_v1_module\n"
        "from hymem.extraction import grounding_gate as grounding_v1_gate_module\n")
    text = old.exact_replace(text,
        "_GROUNDING_IMPORTED_MODULES = (grounding_module, grounding_gate_module)\n"
        "_GROUNDING_IMPORTED_HELPERS = (\n"
        "    grounding_module.loads_exact_or_fenced, grounding_module.LLMRequest,\n"
        "    grounding_module.ALLOWED_PREDICATES, grounding_module.Triple,\n"
        ")\n",
        "_GROUNDING_IMPORTED_MODULES = (grounding_module, grounding_gate_module,\n"
        "    grounding_v2_module, grounding_v1_module, grounding_v1_gate_module)\n"
        "_GROUNDING_IMPORTED_HELPERS = (\n"
        "    grounding_module.loads_exact_or_fenced, grounding_module.LLMRequest,\n"
        "    grounding_module.ALLOWED_PREDICATES, grounding_module.Triple,\n"
        "    grounding_module._build_v2, grounding_module._parse_v2,\n"
        "    grounding_module.build_output_schema, grounding_module.validate_request,\n"
        ")\n")
    text = old.exact_replace(text,
        "    if _GROUNDING_IMPORTED_MODULES != (grounding_module, grounding_gate_module):\n"
        "        raise RuntimeError(\"grounding module binding changed\")\n"
        "    if _GROUNDING_IMPORTED_HELPERS != (\n"
        "            grounding_module.loads_exact_or_fenced, grounding_module.LLMRequest,\n"
        "            grounding_module.ALLOWED_PREDICATES, grounding_module.Triple):\n"
        "        raise RuntimeError(\"grounding helper binding changed\")\n",
        "    if _GROUNDING_IMPORTED_MODULES != (grounding_module, grounding_gate_module,\n"
        "            grounding_v2_module, grounding_v1_module, grounding_v1_gate_module):\n"
        "        raise RuntimeError(\"grounding module binding changed\")\n"
        "    if _GROUNDING_IMPORTED_HELPERS != (\n"
        "            grounding_module.loads_exact_or_fenced, grounding_module.LLMRequest,\n"
        "            grounding_module.ALLOWED_PREDICATES, grounding_module.Triple,\n"
        "            grounding_module._build_v2, grounding_module._parse_v2,\n"
        "            grounding_module.build_output_schema, grounding_module.validate_request):\n"
        "        raise RuntimeError(\"grounding helper binding changed\")\n"
        "    if (grounding_module._build_v2 is not grounding_v2_module.build_grounding_request\n"
        "            or grounding_module._parse_v2 is not grounding_v2_module.parse_grounding_response\n"
        "            or grounding_gate_module._source_gate is not grounding_v1_gate_module\n"
        "            or chunk_module.ground_triples is not grounding_gate_module.ground_triples):\n"
        "        raise RuntimeError(\"grounding imported binding changed\")\n")
    text = old.exact_replace(text,
        '            "grounding_system": _text_digest(grounding_module._SYSTEM),\n',
        '            "grounding_system": _text_digest(grounding_module._SYSTEM),\n'
        '            "grounding_v2_system": _text_digest(grounding_v2_module._SYSTEM),\n'
        '            "grounding_v1_system": _text_digest(grounding_v1_module._SYSTEM),\n')
    text = old.exact_replace(text,
        '            "grounding_batch_limit": grounding_module.MAX_TRIPLES,\n',
        '            "grounding_batch_limit": grounding_v2_module.MAX_TRIPLES,\n'
        '            "grounding_predicate_order": list(grounding_module.PREDICATE_ORDER),\n'
        '            "grounding_v2_contract": grounding_v2_module.GROUNDING_CONTRACT_VERSION,\n'
        '            "grounding_v1_contract": grounding_v1_module.GROUNDING_CONTRACT_VERSION,\n'
        '            "grounding_v1_gate": grounding_v1_gate_module.GROUNDING_GATE_VERSION,\n')
    text = old.exact_replace(text,
        '            "grounding_gate": _module_source_digest(\n'
        '                grounding_gate_module, "ground_triples", "grounding_gate_support_integrity",\n'
        '            ),\n',
        '            "grounding_gate": _module_source_digest(\n'
        '                grounding_gate_module, "ground_triples", "grounding_gate_support_integrity",\n'
        '            ),\n'
        '            "grounding_v2": _module_source_digest(\n'
        '                grounding_v2_module, "build_grounding_request", "parse_grounding_response",\n'
        '            ),\n'
        '            "grounding_v1": _module_source_digest(\n'
        '                grounding_v1_module, "build_grounding_request", "parse_grounding_response",\n'
        '            ),\n'
        '            "grounding_v1_gate": _module_source_digest(\n'
        '                grounding_v1_gate_module, "ground_triples", "grounding_gate_support_integrity",\n'
        '            ),\n')
    text = old.exact_replace(text,
        '                grounding_module, "build_grounding_request", "parse_grounding_response",\n',
        '                grounding_module, "build_grounding_request", "parse_grounding_response",\n'
        '                "build_output_schema", "validate_request",\n')
    return text.encode("utf-8")


def prepare(source: Path, source_stamp: Path, accepted: Path, accepted_stamp: Path,
            target: Path, target_stamp: Path, repo: Path) -> dict[str, str | int]:
    """Validate all pinned inputs, write a fresh 513-file candidate and map."""
    for path in (source, source_stamp, accepted, accepted_stamp, target, target_stamp, repo):
        previous._validate_path(path)
    if target.exists() or target.is_symlink() or target_stamp.exists() or target_stamp.is_symlink():
        raise ValueError("output_not_fresh")
    roots = (source, accepted, repo)
    if any(not root.is_dir() or root.is_symlink() for root in roots):
        raise ValueError("source_directory_invalid")
    if (target == target_stamp or any(target.is_relative_to(root) or target_stamp.is_relative_to(root)
            or root.is_relative_to(target) for root in roots) or target_stamp.is_relative_to(target)):
        raise ValueError("output_inside_source")
    if old.sha(Path(old.__file__).read_bytes()) != OLD_BUILDER_SHA256:
        raise ValueError("old_builder_drift")
    original = previous._map(source_stamp, old.ORIGINAL_FILES, old.ORIGINAL_MAP_SHA256)
    accepted_map = previous._map(accepted_stamp, previous.ACCEPTED_FILES, previous.ACCEPTED_MAP_SHA256)
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
    if len(derived) != old.ORIGINAL_FILES + 5:
        raise ValueError("derived_count_drift")
    for relative in set(original) - {old.CHUNK, old.CONTRACT, old.GROUNDING}:
        if derived[relative] != original[relative]:
            raise ValueError("ordinary_source_drift")
    shutil.copytree(source, target, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"))
    for relative, data in modified.items():
        (target / relative).write_bytes(data)
    if old.inventory(target) != derived:
        raise ValueError("derived_inventory_drift")
    target_stamp.write_text(json.dumps({"source_sha256": derived}, sort_keys=True, indent=2) + "\n")
    return {"files": len(derived), "source_map_sha256": old.ORIGINAL_MAP_SHA256,
            "accepted_map_sha256": previous.ACCEPTED_MAP_SHA256,
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
