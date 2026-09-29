"""Source-bound four-question Luna runner for the derived grounding candidate.

The accepted profiled runner, canary, transport and stage collector are loaded
unchanged. Only the candidate inventory identity and producer identity differ.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


PROFILED_SHA256 = "53628ac7e9c6107bb68d1cd4ebdf42d5a129b96c4d96360d4e7a6525be7bc739"
BUILDER_SHA256 = "51e03ff2bec29f7517db027b103163698c18b9d34603ab950a36f5d09255e96a"
GROUNDED_MAP_SHA256 = "217036b8089911c352cdf5994ef2b37915b5c68d2622a23c0642d264487dbe62"
SCHEMA = "luna-grounding-lme-v1"
ORIGINAL_CANDIDATE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate")
ORIGINAL_INVENTORY_NAME = "headless-source-map.json"


def _pinned_module(name: str, expected: str, module_name: str):
    path = Path(__file__).resolve().with_name(name)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise RuntimeError("grounding_dependency_source_drift")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("grounding_dependency_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def bind_profile(profiled, builder, candidate: Path, inventory: Path,
                 original_inventory: Path, inventory_sha256: str) -> dict:
    """Prove the exact delta and bind accepted loader to its new map."""
    warm = profiled.warm_runner
    original_verify = warm.old.verify_inventory
    proof = builder.verify_derived(ORIGINAL_CANDIDATE, candidate, inventory,
                                   original_inventory)
    if (proof["source_files"] != 508 or
            proof["grounded_map_sha256"] != GROUNDED_MAP_SHA256 or
            proof["grounded_inventory_sha256"] != inventory_sha256):
        raise RuntimeError("grounding_inventory_pin_mismatch")

    def verify_grounded(path, stamp, stamp_sha256, *, expected_map_sha256=GROUNDED_MAP_SHA256,
                        expected_file_count=508):
        if (Path(path) != candidate or Path(stamp) != inventory
                or stamp_sha256 != inventory_sha256
                or expected_map_sha256 != GROUNDED_MAP_SHA256 or expected_file_count != 508):
            raise RuntimeError("grounding_inventory_argument_drift")
        builder.verify_derived(ORIGINAL_CANDIDATE, Path(path), Path(stamp),
                               original_inventory)
        return original_verify(path, stamp, stamp_sha256,
                               expected_map_sha256=GROUNDED_MAP_SHA256,
                               expected_file_count=508)

    warm.old.verify_inventory = verify_grounded
    warm.old.SOURCE_MAP_SHA256 = GROUNDED_MAP_SHA256
    warm.RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    warm.SCHEMA = SCHEMA
    return proof


def main(argv=None) -> int:
    here = Path(__file__).resolve()
    builder = _pinned_module("luna_grounding_candidate.py", BUILDER_SHA256,
                             "pinned_luna_grounding_candidate")
    profiled = _pinned_module("luna_subscription_lme_profiled_v2.py", PROFILED_SHA256,
                              "pinned_luna_grounding_profiled_base")
    args = list(sys.argv[1:] if argv is None else argv)

    def option(name: str) -> Path:
        flag = "--" + name
        if args.count(flag) != 1:
            raise RuntimeError("grounding_argument_invalid")
        index = args.index(flag)
        if index + 1 >= len(args):
            raise RuntimeError("grounding_argument_invalid")
        return Path(args[index + 1])

    candidate = option("candidate")
    inventory = option("inventory-stamp")
    original_inventory = here.with_name(ORIGINAL_INVENTORY_NAME)
    bind_profile(profiled, builder, candidate, inventory, original_inventory,
                 str(option("inventory-sha256")))
    return profiled.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
