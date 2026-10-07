"""Derive an inactive classification-v2 diagnostic from pinned local sources.

The v1 derivation is loaded into a separate namespace and never modified. All
v2 changes are explicit, checked byte transformations over its output.
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
FAMILY = "luna-classification-v2-probe"
INVENTORY = Path("/private/tmp/hymem-classification-v2-root-B63fzR/map.json")
CANDIDATE = INVENTORY.with_name("candidate")
INVENTORY_SHA = "bbcf2294661dd23755b38f111342725a9871e00d66c4b603c845650c328378d5"
EXTRACTION_IDENTITY = "hymem-extraction-contract-sha256-v1:4e1a59f60a1ecd2ebf40d254ffd8d6ee948ac0af8673af2658dd716797252ec2"
BUILDER = "tools/diagnostics/luna_classification_candidate_v2.py"
CONTRACT = "hymem/extraction/grounding_classification_v2.py"
GATE = "hymem/extraction/grounding_classification_gate_v2.py"
ADAPTER = "benchmarks/codex_subscription_classification_v2.py"
EXTRA_SHA = {
    BUILDER: "ad7e656b99df1248b0f526820348b6d914f3012a269207efd38d1f5ac0126519",
    CONTRACT: "2520e4825df9e4d6f403a301451bb9701e052d9a331b5e31a9e106cbb69ad3c9",
    GATE: "2b4ba5d0590bf3a27d57f3fa52e1d84011917b368088628df62b6faa3664d307",
    ADAPTER: "9d760cde8c3c547035f984049cff976df8339248cf10e8c1505243c7efd73763",
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
        raise ValueError(f"v2_binding_missing:{before}:{count}")
    return raw.replace(before.encode(), after.encode())


def bind(raw: bytes, *, family: bool = False, contract: bool = False,
         gate: bool = False, adapter: bool = False, identity: bool = False) -> bytes:
    if family:
        raw = swap(raw, old.FAMILY, FAMILY)
    if contract:
        raw = swap(raw, "grounding_classification_v1", "grounding_classification_v2")
    if gate:
        raw = swap(raw, "grounding_classification_gate_v1", "grounding_classification_gate_v2")
    if adapter:
        raw = swap(raw, "codex_subscription_classification_v1", "codex_subscription_classification_v2")
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
    canary = swap(canary, "luna-classification-canary-v1", "luna-classification-canary-v2")
    stage = old.derive_stage(sources["benchmarks/luna_semantic_stage_accounting.py"], inventory)
    stage = swap(stage, "grounding_classification_gate_v1.py", "grounding_classification_gate_v2.py")
    stage = swap(stage, old.GATE_SHA, EXTRA_SHA[GATE])
    generated["benchmarks/luna_semantic_canary.py"] = canary
    generated["benchmarks/luna_semantic_stage_accounting.py"] = stage

    core = old.derive_core(sources["tools/diagnostics/luna_semantic_probe.py"],
                           old.sha(canary), old.sha(stage))
    core = bind(core, family=True, contract=True, gate=True, adapter=True)
    core = swap(core, old.INVENTORY_SHA, INVENTORY_SHA)
    generated["tools/diagnostics/luna_semantic_probe.py"] = core

    old_adapter = old.derive_transport(sources["benchmarks/codex_subscription_classification_v1.py"])
    v2_adapter = swap(sources[ADAPTER],
        'Path(__file__).resolve().parents[1] / "hymem/extraction/grounding_classification_v2.py"',
        'Path(__file__).resolve().parents[2] / "candidate/hymem/extraction/grounding_classification_v2.py"')
    generated[ADAPTER] = v2_adapter

    host = old.derive_host(sources["tools/diagnostics/luna_semantic_probe_host.py"],
                           sources, old.sha(core), old.sha(canary), old.sha(stage), old.sha(old_adapter))
    host = bind(host, family=True, identity=True)
    host = swap(host, old.INVENTORY_SHA, INVENTORY_SHA)
    host = swap(host,
                "source = root / 'code/tools/diagnostics/luna_classification_candidate.py'",
                "source = root / 'code/tools/diagnostics/luna_classification_candidate_v2.py'")
    for name, digest in (("benchmarks/codex_subscription_classification_v1.py", old.sha(old_adapter)),
                         ("hymem/extraction/grounding_classification_v1.py", old.CLASSIFICATION_SHA),
                         ("hymem/extraction/grounding_classification_gate_v1.py", old.GATE_SHA)):
        host = swap(host, f"    '{name}': '{digest}',\n", "")
    for name in (BUILDER, CONTRACT, GATE, ADAPTER):
        anchor = "    'tools/diagnostics/luna_semantic_candidate_v2.py':"
        host = swap(host, anchor,
                    f"    '{name}': '{old.sha(v2_adapter) if name == ADAPTER else EXTRA_SHA[name]}',\n" + anchor)
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
