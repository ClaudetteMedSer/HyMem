"""Assemble the immutable SIWC four-question source bundle; no inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat


RUNNER_RELATIVE = "tools/diagnostics/siwc_lme_diagnostic_v4.py"
RUNNER_SHA256 = "c26e85f3fad194e61f4fb8bafc601a1b64d1dae4cf1eb2e7404d141a1177ee97"
FROZEN_INVENTORY_SHA256 = "1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
FROZEN_MAP_SHA256 = "f22cd2be376f019efa1d39cb6c2f1e43ffef2d7ea3bb7a07ac64479241cd4b11"
INVENTORY_SHA256 = "b87ce83ca123fb2a51f1716ca3d3d71fd3b73799a30469b34a8f033babaa90b6"
PROMPT_RELATIVE = "hymem/extraction/prompts/__init__.py"
PROMPT_SOURCE_SHA256 = "17ee5017c54a1ba255e0220aa8e246766127cb71ced65fb2e8c812034f6e184c"
PROMPT_RESULT_SHA256 = "ec50ad403d9b678d3cfc62a55e4f5391b00781140d42a28802a864b3c1905048"
TRANSFORMER_RELATIVE = "tools/diagnostics/siwc_extraction_prompt_repair_v1.py"
TRANSFORMER_SHA256 = "b777dcb3b6d0897bdeaa715d061158fd2c779a2b3e1fd2b870f1c6ba24ce47ff"
HELPER_RELATIVE = "benchmarks/lme_diagnostic.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def runner_from(repo: Path):
    path = repo / RUNNER_RELATIVE
    if not regular(path) or sha(path) != RUNNER_SHA256:
        raise ValueError("runner_drift")
    spec = importlib.util.spec_from_file_location("pinned_siwc_bundle_runner", path)
    if spec is None or spec.loader is None:
        raise ValueError("runner_load_invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_map(candidate: Path, map_path: Path, runner) -> dict[str, str]:
    if candidate.is_symlink() or not candidate.is_dir() or not regular(map_path) or sha(map_path) != FROZEN_INVENTORY_SHA256:
        raise ValueError("candidate_input_invalid")
    stamp = json.loads(map_path.read_text(encoding="utf-8"))
    entries = stamp.get("source_sha256") if type(stamp) is dict else None
    if type(entries) is not dict or len(entries) != runner.ACCEPTED_FILES:
        raise ValueError("candidate_map_invalid")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != FROZEN_MAP_SHA256:
        raise ValueError("candidate_map_pin_invalid")
    if entries.get(PROMPT_RELATIVE) != PROMPT_SOURCE_SHA256:
        raise ValueError("prompt_source_pin_invalid")
    for relative, expected in entries.items():
        if (type(relative) is not str or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or type(expected) is not str
                or not regular(candidate / relative) or sha(candidate / relative) != expected):
            raise ValueError("candidate_file_drift")
    return entries


def revised_inventory(entries: dict[str, str], runner) -> bytes:
    """Derive the sole permitted revised map using the frozen JSON formatting."""
    if (type(entries) is not dict or len(entries) != runner.ACCEPTED_FILES
            or entries.get(PROMPT_RELATIVE) != PROMPT_SOURCE_SHA256
            or hashlib.sha256(json.dumps(entries, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest() != FROZEN_MAP_SHA256):
        raise ValueError("candidate_map_pin_invalid")
    revised = {**entries, PROMPT_RELATIVE: PROMPT_RESULT_SHA256}
    digest = hashlib.sha256(json.dumps(revised, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if digest != runner.ACCEPTED_MAP_SHA256:
        raise ValueError("revised_map_pin_invalid")
    raw = (json.dumps({"source_sha256": revised}, sort_keys=True, indent=2) + "\n").encode()
    if hashlib.sha256(raw).hexdigest() != INVENTORY_SHA256:
        raise ValueError("revised_inventory_pin_invalid")
    return raw


def transformer_from(repo: Path):
    path = repo / TRANSFORMER_RELATIVE
    if not regular(path) or sha(path) != TRANSFORMER_SHA256:
        raise ValueError("transformer_drift")
    spec = importlib.util.spec_from_file_location("pinned_siwc_prompt_transformer", path)
    if spec is None or spec.loader is None:
        raise ValueError("transformer_load_invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SOURCE_SHA256 != PROMPT_SOURCE_SHA256 or module.RESULT_SHA256 != PROMPT_RESULT_SHA256:
        raise ValueError("transformer_pin_invalid")
    return module


def assemble(*, repo: Path, accepted_code: Path, candidate: Path,
             map_path: Path, output: Path) -> dict:
    if any(not path.is_absolute() for path in (repo, accepted_code, candidate, map_path, output)):
        raise ValueError("absolute_paths_required")
    if (repo.is_symlink() or not repo.is_dir() or accepted_code.is_symlink()
            or not accepted_code.is_dir() or output.exists() or output.is_symlink()
            or not output.parent.is_dir()):
        raise ValueError("bundle_input_invalid")
    runner = runner_from(repo)
    entries = candidate_map(candidate, map_path, runner)
    revised_map = revised_inventory(entries, runner)
    transformer = transformer_from(repo)
    revised_prompt = transformer.transform_prompt_source((candidate / PROMPT_RELATIVE).read_bytes())
    if hashlib.sha256(revised_prompt).hexdigest() != PROMPT_RESULT_SHA256:
        raise ValueError("prompt_result_pin_invalid")
    code_sources: dict[str, Path] = {}
    for relative, expected in {**runner.PINS, **runner.SIWC_PINS}.items():
        # Historical benchmark pin files are retained in the accepted code root.
        # The public SIWC bridge and auth closure are new, repository-pinned files.
        origin = repo if relative in runner.SIWC_PINS or relative in {
            "tools/diagnostics/luna_subscription_lme_warm_v2.py",
            "tools/diagnostics/luna_subscription_pilot.py",
            "benchmarks/codex_subscription_warm_v4.py",
            "benchmarks/codex_subscription_warm_v5.py",
            "benchmarks/codex_subscription_warm_v6.py",
            "benchmarks/codex_subscription_warm_v7.py",
            "benchmarks/codex_subscription_warm_v8.py",
            "benchmarks/codex_subscription_warm_v9.py",
            "benchmarks/codex_subscription_timeout_v3.py",
            "benchmarks/codex_subscription_staged_v6.py"} else accepted_code
        path = origin / relative
        if not regular(path) or sha(path) != expected:
            raise ValueError("code_source_drift")
        code_sources[relative] = path
    helper = repo / HELPER_RELATIVE
    if not regular(helper) or sha(helper) != runner.DIAGNOSTIC_HELPER_SHA256:
        raise ValueError("diagnostic_helper_drift")
    code_sources[HELPER_RELATIVE] = helper
    code_sources[RUNNER_RELATIVE] = repo / RUNNER_RELATIVE
    output.mkdir(mode=0o700)
    for relative, expected in entries.items():
        target = output / "candidate" / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if relative == PROMPT_RELATIVE:
            target.write_bytes(revised_prompt)
        else:
            shutil.copy2(candidate / relative, target)
        if sha(target) != (PROMPT_RESULT_SHA256 if relative == PROMPT_RELATIVE else expected):
            raise ValueError("candidate_copy_drift")
    for relative, origin in code_sources.items():
        target = output / "code" / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        if sha(target) != sha(origin):
            raise ValueError("code_copy_drift")
    (output / "source-map.json").write_bytes(revised_map)
    if sha(output / "source-map.json") != INVENTORY_SHA256:
        raise ValueError("map_copy_drift")
    return {"schema": "siwc-lme-diagnostic-source-bundle-v3",
        "root": str(output), "candidate_files": len(entries),
        "code_files": len(code_sources), "candidate_map_sha256": runner.ACCEPTED_MAP_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "code_sha256": {relative: sha(path) for relative, path in sorted(code_sources.items())},
        "dataset_present": False, "credential_present": False,
        "launch_receipt_present": False, "model_calls": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "accepted-code", "candidate", "map", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(assemble(repo=Path(args.repo), accepted_code=Path(args.accepted_code),
        candidate=Path(args.candidate), map_path=Path(args.map), output=Path(args.output)),
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
