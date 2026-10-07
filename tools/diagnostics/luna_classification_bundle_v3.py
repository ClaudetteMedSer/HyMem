"""Derive an inactive claim-first classification-v3 diagnostic from pinned sources.

The v1 derivation is loaded into a separate namespace and never modified. All
v3 changes are explicit, checked byte transformations over its output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat

PRIOR_SHA = "0903977301fca4588ae69c8363238b934d975a78dcd0128a2b49cfc4daf4e801"
FAMILY = "luna-classification-v3-probe"
INVENTORY = Path("/private/tmp/hymem-classification-v3-root-YFcQl4dR/map.json")
CANDIDATE = INVENTORY.with_name("candidate")
INVENTORY_SHA = "f63f656ee1aaa204c13a44f56cf1864596117f13f36189cc3b7ba05cd7646bab"
EXTRACTION_IDENTITY = "hymem-extraction-contract-sha256-v1:21372a7db114466d4ad248c344ea4493606c858c9fd010c2f77456a9ad31b6c3"
BUILDER = "tools/diagnostics/luna_classification_candidate_v3.py"
CONTRACT = "hymem/extraction/grounding_classification_v3.py"
GATE = "hymem/extraction/grounding_classification_gate_v3.py"
ADAPTER = "benchmarks/codex_subscription_classification_v3.py"
EXTRA_SHA = {
    BUILDER: "9595892e727661aee37aa5d4e0aa014935a7974e92f725c2216cc8f40e67f5e6",
    CONTRACT: "435c0edf52197a5ffa9e715db24156e26109bf7445ad7c9baba2f632ba7f7a76",
    GATE: "2f9c56e3b2bbf2a012cb6fb81cf960eeccddd0c8e05fce9bdf07c2c5fdc1c5d9",
    ADAPTER: "d3c955922219d99b359aa6619b821e7da1662e36ecea54d09b1267dce38d2c32",
}


def _prior():
    path = Path(__file__).with_name("luna_classification_bundle.py")
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != PRIOR_SHA:
        raise ValueError("prior_bundle_drift")
    spec = importlib.util.spec_from_file_location("_pinned_classification_bundle_v1", path)
    if spec is None:
        raise ValueError("prior_bundle_load_invalid")
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


old = _prior()
INPUT_SHA = {**old.INPUT_SHA, **EXTRA_SHA}
CODE = tuple(k for k in INPUT_SHA if k not in {
    "tools/diagnostics/luna_semantic_probe_adapter_v2.py",
    "tools/diagnostics/luna_semantic_verdict_replay_root.py",
    "hymem/extraction/grounding_classification_v1.py",
    "hymem/extraction/grounding_classification_gate_v1.py",
    "benchmarks/codex_subscription_classification_v1.py"})


def swap(raw: bytes, before: str, after: str, *, minimum: int = 1) -> bytes:
    count = raw.count(before.encode())
    if count < minimum:
        raise ValueError(f"v3_binding_missing:{before}:{count}")
    return raw.replace(before.encode(), after.encode())


def bind(raw: bytes, *, family: bool = False, contract: bool = False,
         gate: bool = False, adapter: bool = False, identity: bool = False) -> bytes:
    if family:
        raw = swap(raw, old.FAMILY, FAMILY)
    if contract:
        raw = swap(raw, "grounding_classification_v1", "grounding_classification_v3")
    if gate:
        raw = swap(raw, "grounding_classification_gate_v1", "grounding_classification_gate_v3")
    if adapter:
        raw = swap(raw, "codex_subscription_classification_v1", "codex_subscription_classification_v3")
    if identity:
        raw = swap(raw, old.EXTRACTION_IDENTITY, EXTRACTION_IDENTITY)
    return raw


def validate_candidate() -> dict[str, str]:
    stamp = json.loads(old.source(INVENTORY, INVENTORY_SHA))
    mapping = stamp.get("source_sha256")
    if type(mapping) is not dict or len(mapping) != 513:
        raise ValueError("candidate_map_invalid")
    if (not CANDIDATE.is_absolute() or CANDIDATE != CANDIDATE.resolve()
            or CANDIDATE.is_symlink() or not CANDIDATE.is_dir()):
        raise ValueError("candidate_path_invalid")
    actual = {}
    for path in CANDIDATE.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("candidate_path_invalid")
        if path.is_file():
            actual[path.relative_to(CANDIDATE).as_posix()] = old.sha(path.read_bytes())
    if (actual != mapping or mapping.get(CONTRACT) != EXTRA_SHA[CONTRACT]
            or mapping.get(GATE) != EXTRA_SHA[GATE]
            or any(type(k) is not str or type(v) is not str
                   or re.fullmatch(r"[0-9a-f]{64}", v) is None
                   or Path(k).is_absolute() or ".." in Path(k).parts
                   for k, v in mapping.items())):
        raise ValueError("candidate_inventory_invalid")
    return mapping


def prepare(repo: Path, target: Path) -> dict:
    if (not repo.is_absolute() or repo != repo.resolve() or not repo.is_dir()
            or repo.is_symlink() or not target.is_absolute()
            or target.parent != target.parent.resolve() or target.exists()
            or target.is_symlink() or not target.parent.is_dir()
            or target.is_relative_to(repo) or repo.is_relative_to(target)
            or target.is_relative_to(INVENTORY.parent)
            or INVENTORY.parent.is_relative_to(target)):
        raise ValueError("output_boundary_invalid")
    sources = {name: old.source(repo / name, digest) for name, digest in INPUT_SHA.items()}
    inventory = validate_candidate()
    generated = {name: sources[name] for name in CODE}

    canary = bind(old.derive_canary(sources["benchmarks/luna_semantic_canary.py"]),
                  contract=True, gate=True, identity=True)
    canary = swap(canary, "luna-classification-canary-v1", "luna-classification-canary-v3")
    stage = old.derive_stage(sources["benchmarks/luna_semantic_stage_accounting.py"], inventory)
    stage = swap(stage, "grounding_classification_gate_v1.py", "grounding_classification_gate_v3.py")
    stage = swap(stage, old.GATE_SHA, EXTRA_SHA[GATE])
    generated["benchmarks/luna_semantic_canary.py"] = canary
    generated["benchmarks/luna_semantic_stage_accounting.py"] = stage

    core = old.derive_core(sources["tools/diagnostics/luna_semantic_probe.py"],
                           old.sha(canary), old.sha(stage))
    core = bind(core, family=True, contract=True, gate=True, adapter=True)
    core = swap(core, old.INVENTORY_SHA, INVENTORY_SHA)
    generated["tools/diagnostics/luna_semantic_probe.py"] = core

    old_adapter = old.derive_transport(sources["benchmarks/codex_subscription_classification_v1.py"])
    v3_adapter = swap(sources[ADAPTER],
        'Path(__file__).resolve().parents[1] / "hymem/extraction/grounding_classification_v3.py"',
        'Path(__file__).resolve().parents[2] / "candidate/hymem/extraction/grounding_classification_v3.py"')
    generated[ADAPTER] = v3_adapter

    host = old.derive_host(sources["tools/diagnostics/luna_semantic_probe_host.py"],
                           sources, old.sha(core), old.sha(canary), old.sha(stage), old.sha(old_adapter))
    host = bind(host, family=True, identity=True)
    host = swap(host, old.INVENTORY_SHA, INVENTORY_SHA)
    host = swap(host,
                "source = root / 'code/tools/diagnostics/luna_classification_candidate.py'",
                "source = root / 'code/tools/diagnostics/luna_classification_candidate_v3.py'")
    for name, digest in (("benchmarks/codex_subscription_classification_v1.py", old.sha(old_adapter)),
                         ("hymem/extraction/grounding_classification_v1.py", old.CLASSIFICATION_SHA),
                         ("hymem/extraction/grounding_classification_gate_v1.py", old.GATE_SHA)):
        host = swap(host, f"    '{name}': '{digest}',\n", "")
    for name in (BUILDER, CONTRACT, GATE, ADAPTER):
        anchor = "    'tools/diagnostics/luna_semantic_candidate_v2.py':"
        host = swap(host, anchor,
                    f"    '{name}': '{old.sha(v3_adapter) if name == ADAPTER else EXTRA_SHA[name]}',\n" + anchor)
    generated["tools/diagnostics/luna_semantic_probe_host.py"] = host

    run = old.derive_run(sources["tools/diagnostics/luna_semantic_probe_run.py"], old.sha(core), old.sha(host))
    run = bind(run, family=True, contract=True, adapter=True)
    generated["tools/diagnostics/luna_semantic_probe_run.py"] = run
    reader = bind(old.derive_reader(sources["tools/diagnostics/luna_semantic_probe_progress.py"]),
                  family=True, adapter=True)
    generated["tools/diagnostics/luna_semantic_probe_progress.py"] = reader
    startup = old.derive_startup(sources["tools/diagnostics/luna_semantic_probe_adapter_v2.py"],
                                 old.sha(host), old.sha(run), old.sha(reader))
    startup = bind(startup, family=True)
    replay = bind(old.derive_replay(sources["tools/diagnostics/luna_semantic_verdict_replay_root.py"]),
                  family=True)

    output_sha = {"code/" + name: old.sha(raw) for name, raw in generated.items()}
    output_sha.update({"adapter-v2.py": old.sha(startup), "verdict-replay.py": old.sha(replay)})
    receipt = {"schema": FAMILY + "-bundle-v1", "candidate_inventory_sha256": INVENTORY_SHA,
               "candidate_files": 513, "extraction_identity": EXTRACTION_IDENTITY,
               "input_sha256": dict(sorted(INPUT_SHA.items())),
               "output_sha256": dict(sorted(output_sha.items())),
               "unchanged_code": sorted(name for name in generated if generated[name] == sources[name]),
               "model_calls": 0, "launched": False}
    target.mkdir(mode=0o700)
    for name, raw in generated.items():
        old.write(target / "code" / name, raw)
    old.write(target / "adapter-v2.py", startup)
    old.write(target / "verdict-replay.py", replay)
    old.write(target / "derivation-receipt.json", (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return {"target": str(target), "code_files": len(generated),
            "receipt_sha256": old.sha((target / "derivation-receipt.json").read_bytes()),
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
