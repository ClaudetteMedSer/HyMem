"""Derive a source-bound semantic candidate from the immutable 508-file runtime.

This is an offline derivation only. It never activates a candidate or calls a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ORIGINAL_MAP_SHA256 = "217036b8089911c352cdf5994ef2b37915b5c68d2622a23c0642d264487dbe62"
ORIGINAL_FILES = 508
GROUNDING_SHA256 = "dd1a49b56abf569b4476a0b735e67e72a86723bf88e3a4739998aceeabe19c18"
GATE_SHA256 = "bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8"
CHUNK_SHA256 = "11fe46a7d1f6fcd65a33a109104a42ebcd7af888b99d2e30460bcd90633301c9"
CONTRACT_SHA256 = "f508413885f746c28b40a67d4f3a9557e3afbe373110e597d3c0c0baf279ea60"
CHUNK = "hymem/extraction/chunk.py"
CONTRACT = "hymem/extraction/contract.py"
GROUNDING = "hymem/extraction/grounding.py"
GATE = "hymem/extraction/grounding_gate.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact_replace(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("patch_anchor_drift")
    return source.replace(old, new, 1)


def derive_chunk(original: bytes) -> bytes:
    if sha(original) != CHUNK_SHA256:
        raise ValueError("chunk_source_drift")
    text = original.decode("utf-8")
    text = exact_replace(text,
        "from hymem.extraction.jsonio import (",
        "from hymem.extraction.grounding_gate import (\n"
        "    GroundingGateError, ground_triples, grounding_gate_support_integrity,\n"
        ")\nfrom hymem.extraction.jsonio import (")
    text = exact_replace(text,
        "    Triple, normalize_combined_triple_item, triples_from_list,\n)",
        "    Triple, normalize_combined_triple_item, triples_from_list,\n"
        "    GroundingGateError, ground_triples, grounding_gate_support_integrity,\n)")
    text = exact_replace(text,
        "        Triple, normalize_combined_triple_item, triples_from_list,\n    )",
        "        Triple, normalize_combined_triple_item, triples_from_list,\n"
        "        GroundingGateError, ground_triples, grounding_gate_support_integrity,\n    )")
    text = exact_replace(text,
        '    "contract_failure",\n',
        '    "contract_failure",\n    "grounding_failure",\n')
    text = exact_replace(text,
        "    provider_attempts: int = 0\n    # Number of leaves",
        "    provider_attempts: int = 0\n"
        "    grounding_calls: int = 0\n    grounding_recheck_calls: int = 0\n"
        "    grounding_provider_attempts: int = 0\n    # Number of leaves")
    text = exact_replace(text,
        "class _CallBudget:\n    completion_calls: int = 0\n    provider_attempts: int = 0\n",
        "class _CallBudget:\n    completion_calls: int = 0\n"
        "    provider_attempts: int = 0\n    grounding_calls: int = 0\n"
        "    grounding_recheck_calls: int = 0\n    grounding_provider_attempts: int = 0\n")
    text = exact_replace(text,
        "        result.provider_attempts = budget.provider_attempts\n",
        "        result.provider_attempts = budget.provider_attempts\n"
        "        result.grounding_calls = budget.grounding_calls\n"
        "        result.grounding_recheck_calls = budget.grounding_recheck_calls\n"
        "        result.grounding_provider_attempts = budget.grounding_provider_attempts\n"
        "        if result.failed:\n"
        "            result.triples = []\n"
        "            result.markers = []\n"
        "            result.entity_type_hints = {}\n"
        "            result.entity_property_hints = {}\n")
    text = exact_replace(text,
        "    def verify_nonempty(\n",
        "    def grounding_call(request: LLMRequest, recheck: bool) -> str:\n"
        "        if budget.completion_calls >= completion_call_limit:\n"
        "            raise GroundingGateError(\"calls:max_exceeded\")\n"
        "        budget.completion_calls += 1\n"
        "        budget.grounding_calls += 1\n"
        "        if recheck:\n"
        "            budget.grounding_recheck_calls += 1\n"
        "        measurement = None\n"
        "        try:\n"
        "            with measure_provider_attempts(client) as measurement:\n"
        "                return client.complete(request)\n"
        "        except Exception as exc:\n"
        "            log.warning(\"chunk_grounding.call_failure error_type=%s\", type(exc).__name__)\n"
        "            raise GroundingGateError(\"provider:call_failed\") from exc\n"
        "        finally:\n"
        "            attempts = measurement.attempts if measurement is not None else 1\n"
        "            budget.provider_attempts += attempts\n"
        "            budget.grounding_provider_attempts += attempts\n\n"
        "    def verify_nonempty(\n")
    text = exact_replace(text,
        "        merged = _merge_verified_result(accepted, verified)\n"
        "        verifier_split_recoverable = (",
        "        merged = _merge_verified_result(accepted, verified)\n"
        "        if not merged.failed and merged.triples:\n"
        "            try:\n"
        "                merged.triples = ground_triples(\n"
        "                    merged.triples, current.source_records,\n"
        "                    current.context_records, current.text, grounding_call,\n"
        "                )\n"
        "            except GroundingGateError as exc:\n"
        "                reason = (\"resource_limit\" if exc.code == \"calls:max_exceeded\"\n"
        "                          else \"call_failure\" if exc.code == \"provider:call_failed\"\n"
        "                          else \"grounding_failure\")\n"
        "                return _failure(reason,\n"
        "                                \"grounding:\" + exc.code.replace(\":\", \"_\")), False\n"
        "        verifier_split_recoverable = (")
    return text.encode("utf-8")


def derive_contract(original: bytes) -> bytes:
    if sha(original) != CONTRACT_SHA256:
        raise ValueError("contract_source_drift")
    text = original.decode("utf-8")
    text = exact_replace(text,
        "from hymem.extraction import chunk as chunk_module\n",
        "from hymem.extraction import chunk as chunk_module\n"
        "from hymem.extraction import grounding as grounding_module\n"
        "from hymem.extraction import grounding_gate as grounding_gate_module\n")
    text = exact_replace(text,
        "_LOADED_MODULE_SLICE_SHA256 = None\n",
        "_LOADED_MODULE_SLICE_SHA256 = None\n"
        "_GROUNDING_IMPORTED_MODULES = (grounding_module, grounding_gate_module)\n"
        "_GROUNDING_IMPORTED_HELPERS = (\n"
        "    grounding_module.loads_exact_or_fenced, grounding_module.LLMRequest,\n"
        "    grounding_module.ALLOWED_PREDICATES, grounding_module.Triple,\n"
        ")\n")
    text = exact_replace(text,
        "    # These are the exact deterministic system prompts used by benchmark\n",
        "    if _GROUNDING_IMPORTED_MODULES != (grounding_module, grounding_gate_module):\n"
        "        raise RuntimeError(\"grounding module binding changed\")\n"
        "    if _GROUNDING_IMPORTED_HELPERS != (\n"
        "            grounding_module.loads_exact_or_fenced, grounding_module.LLMRequest,\n"
        "            grounding_module.ALLOWED_PREDICATES, grounding_module.Triple):\n"
        "        raise RuntimeError(\"grounding helper binding changed\")\n"
        "    gate_integrity = getattr(grounding_gate_module, \"grounding_gate_support_integrity\", None)\n"
        "    if (gate_integrity is not getattr(grounding_gate_module,\n"
        "            \"_GROUNDING_GATE_INTEGRITY_FUNCTION\", None)\n"
        "            or not callable(gate_integrity) or not gate_integrity()):\n"
        "        raise RuntimeError(\"grounding gate helper integrity changed\")\n"
        "    if (grounding_gate_module.build_grounding_request is not grounding_module.build_grounding_request\n"
        "            or grounding_gate_module.parse_grounding_response is not grounding_module.parse_grounding_response):\n"
        "        raise RuntimeError(\"grounding consumed helper binding changed\")\n"
        "    # These are the exact deterministic system prompts used by benchmark\n")
    text = exact_replace(text,
        '            "contract_repair_user_suffix": _text_digest(\n'
        '                chunk_module.CHUNK_CONTRACT_REPAIR_USER_SUFFIX\n'
        '            ),\n',
        '            "contract_repair_user_suffix": _text_digest(\n'
        '                chunk_module.CHUNK_CONTRACT_REPAIR_USER_SUFFIX\n'
        '            ),\n'
        '            "grounding_system": _text_digest(grounding_module._SYSTEM),\n')
    text = exact_replace(text,
        '            "clean_empty": chunk_module.CLEAN_EMPTY_RECOVERY_POLICY_VERSION,\n',
        '            "clean_empty": chunk_module.CLEAN_EMPTY_RECOVERY_POLICY_VERSION,\n'
        '            "grounding_gate": grounding_gate_module.GROUNDING_GATE_VERSION,\n'
        '            "grounding_contract": grounding_module.GROUNDING_CONTRACT_VERSION,\n'
        '            "grounding_batch_limit": grounding_module.MAX_TRIPLES,\n')
    text = exact_replace(text,
        '            "jsonio": _module_source_digest(\n',
        '            "grounding": _module_source_digest(\n'
        '                grounding_module, "build_grounding_request", "parse_grounding_response",\n'
        '            ),\n'
        '            "grounding_gate": _module_source_digest(\n'
        '                grounding_gate_module, "ground_triples", "grounding_gate_support_integrity",\n'
        '            ),\n'
        '            "jsonio": _module_source_digest(\n')
    return text.encode("utf-8")


def inventory(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("inventory_symlink")
        if path.is_file():
            files[relative.as_posix()] = sha(path.read_bytes())
        elif not path.is_dir():
            raise ValueError("inventory_special_file")
    return files


def mapping_sha(mapping: dict[str, str]) -> str:
    return sha(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode())


def prepare(source: Path, source_stamp: Path, target: Path, target_stamp: Path,
            repo: Path) -> dict[str, str | int]:
    for path in (source, source_stamp, target, target_stamp, repo):
        if not path.is_absolute():
            raise ValueError("paths_must_be_absolute")
        if path != path.resolve():
            raise ValueError("path_alias_or_symlink")
    if target.exists() or target.is_symlink() or target_stamp.exists() or target_stamp.is_symlink():
        raise ValueError("output_not_fresh")
    if (target == source or target.is_relative_to(source)
            or target_stamp.is_relative_to(source)
            or target_stamp.is_relative_to(target)
            or target.is_relative_to(repo) or target_stamp.is_relative_to(repo)):
        raise ValueError("output_inside_source")
    if not source.is_dir() or source.is_symlink() or not repo.is_dir() or repo.is_symlink():
        raise ValueError("source_directory_invalid")
    if not source_stamp.is_file() or source_stamp.is_symlink():
        raise ValueError("source_stamp_invalid")
    stamp = json.loads(source_stamp.read_text())
    expected = stamp.get("source_sha256", stamp)
    if len(expected) != ORIGINAL_FILES or mapping_sha(expected) != ORIGINAL_MAP_SHA256:
        raise ValueError("original_inventory_identity")
    if inventory(source) != expected:
        raise ValueError("original_inventory_drift")
    for relative in (GROUNDING, GATE):
        if (repo / relative).is_symlink() or not (repo / relative).is_file():
            raise ValueError("helper_source_invalid")
    grounding = (repo / GROUNDING).read_bytes()
    if sha(grounding) != GROUNDING_SHA256:
        raise ValueError("accepted_grounding_drift")
    gate = (repo / GATE).read_bytes()
    if sha(gate) != GATE_SHA256:
        raise ValueError("gate_source_drift")
    modified = {
        CHUNK: derive_chunk((source / CHUNK).read_bytes()),
        CONTRACT: derive_contract((source / CONTRACT).read_bytes()),
        GROUNDING: grounding,
        GATE: gate,
    }
    shutil.copytree(source, target, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"))
    for relative, data in modified.items():
        (target / relative).write_bytes(data)
    derived = dict(expected)
    derived.update({relative: sha(data) for relative, data in modified.items()})
    if len(derived) != ORIGINAL_FILES + 2 or inventory(target) != derived:
        raise ValueError("derived_inventory_drift")
    target_stamp.write_text(json.dumps({"source_sha256": derived}, sort_keys=True, indent=2) + "\n")
    return {"files": len(derived), "source_map_sha256": ORIGINAL_MAP_SHA256,
            "derived_map_sha256": mapping_sha(derived),
            "derived_stamp_sha256": sha(target_stamp.read_bytes()),
            "chunk_sha256": derived[CHUNK], "contract_sha256": derived[CONTRACT],
            "grounding_sha256": derived[GROUNDING], "gate_sha256": derived[GATE]}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("source", "source_stamp", "target", "target_stamp", "repo"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.source_stamp, args.target,
                             args.target_stamp, args.repo), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
