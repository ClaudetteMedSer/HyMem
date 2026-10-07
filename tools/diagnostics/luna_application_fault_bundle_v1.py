"""Assemble a fresh, source-only diagnostic LME bundle without inference.

This does not create a launch receipt, a one-shot marker, a service, or a
dataset copy. The accepting operator supplies those separately after a real
dataset/runtime preflight. Existing staged bundles are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat


RUNNER_RELATIVE = "tools/diagnostics/luna_lme_diagnostic_v9.py"
RUNNER_SHA256 = "b3e1135893a715dec4e138c25f6bf3c70f3912dbd5df81014ee7c8f2767fd278"
CAPTURE_RELATIVE = "tools/diagnostics/luna_application_fault_capture_v2.py"
CAPTURE_SHA256 = "e865366dc7ae1fc5cb72367c7f3d59c3c3d3227486728e7035b6e61242a44977"
PROBE_RELATIVE = "tools/diagnostics/luna_application_fault_probe_v2.py"
PROBE_SHA256 = "017c8d925368517b5be17d8373112c5b362810d37cb62dbcb716fdbc24e90c7c"
INVENTORY_SHA256 = "1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
HELPER_RELATIVE = "benchmarks/lme_diagnostic.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> bool:
    try:
        return path.is_absolute() and stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _runner(repo: Path):
    path = repo / RUNNER_RELATIVE
    if not _regular(path) or _sha(path) != RUNNER_SHA256:
        raise ValueError("runner_missing")
    spec = importlib.util.spec_from_file_location("verified_lme_diagnostic_for_bundle", path)
    if spec is None or spec.loader is None:
        raise ValueError("runner_module_invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_map(candidate: Path, map_path: Path, runner) -> dict[str, str]:
    if (candidate.is_symlink() or not candidate.is_dir()
            or not _regular(map_path) or _sha(map_path) != INVENTORY_SHA256):
        raise ValueError("candidate_input_invalid")
    stamp = json.loads(map_path.read_text(encoding="utf-8"))
    entries = stamp.get("source_sha256", stamp) if type(stamp) is dict else None
    if type(entries) is not dict or len(entries) != runner.ACCEPTED_FILES:
        raise ValueError("candidate_map_shape_invalid")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != runner.ACCEPTED_MAP_SHA256:
        raise ValueError("candidate_map_pin_invalid")
    for relative, expected in entries.items():
        path = candidate / relative
        if (type(relative) is not str or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or type(expected) is not str
                or not _regular(path) or _sha(path) != expected):
            raise ValueError("candidate_file_drift")
    return entries


def assemble(*, repo: Path, accepted_code: Path, candidate: Path,
             map_path: Path, output: Path) -> dict:
    """Copy only verified files into a new private root; never mutate inputs."""
    for path in (repo, accepted_code, candidate, map_path, output):
        if not path.is_absolute():
            raise ValueError("absolute_paths_required")
    if (not repo.is_dir() or repo.is_symlink() or not accepted_code.is_dir()
            or accepted_code.is_symlink() or output.exists() or output.is_symlink()
            or not output.parent.is_dir()):
        raise ValueError("bundle_input_invalid")
    runner = _runner(repo)
    entries = _candidate_map(candidate, map_path, runner)
    code_sources: dict[str, Path] = {}
    for relative, expected in runner.PINS.items():
        origin = (repo if relative in {
            "tools/diagnostics/luna_subscription_lme_warm_v2.py",
            "tools/diagnostics/luna_subscription_pilot.py",
            "benchmarks/codex_subscription_warm_v4.py",
            "benchmarks/codex_subscription_warm_v5.py",
            "benchmarks/codex_subscription_warm_v6.py",
            "benchmarks/codex_subscription_warm_v7.py",
            "benchmarks/codex_subscription_warm_v8.py",
            "benchmarks/codex_subscription_warm_v9.py",
            "benchmarks/codex_subscription_timeout_v3.py",
            "benchmarks/codex_subscription_staged_v6.py"} else accepted_code)
        path = origin / relative
        if not _regular(path) or _sha(path) != expected:
            raise ValueError("accepted_code_drift")
        code_sources[relative] = path
    helper = repo / HELPER_RELATIVE
    if not _regular(helper) or _sha(helper) != runner.DIAGNOSTIC_HELPER_SHA256:
        raise ValueError("diagnostic_helper_drift")
    code_sources[HELPER_RELATIVE] = helper
    code_sources[RUNNER_RELATIVE] = repo / RUNNER_RELATIVE
    for relative, expected in ((CAPTURE_RELATIVE, CAPTURE_SHA256),
                               (PROBE_RELATIVE, PROBE_SHA256)):
        path = repo / relative
        if not _regular(path) or _sha(path) != expected:
            raise ValueError("probe_source_drift")
        code_sources[relative] = path
    output.mkdir(mode=0o700)
    for relative, expected in entries.items():
        target = output / "candidate" / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(candidate / relative, target)
        if _sha(target) != expected:
            raise ValueError("candidate_copy_drift")
    for relative, origin in code_sources.items():
        target = output / "code" / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        if _sha(target) != _sha(origin):
            raise ValueError("code_copy_drift")
    shutil.copy2(map_path, output / "source-map.json")
    if _sha(output / "source-map.json") != _sha(map_path):
        raise ValueError("map_copy_drift")
    return {"schema": "luna-application-fault-source-bundle-v1",
        "root": str(output), "candidate_files": len(entries),
        "candidate_map_sha256": runner.ACCEPTED_MAP_SHA256,
        "inventory_sha256": _sha(map_path),
        "code_sha256": {relative: _sha(origin)
            for relative, origin in sorted(code_sources.items())},
        "dataset_present": False, "binary_present": False,
        "launch_receipt_present": False, "model_calls": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "accepted-code", "candidate", "map", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = assemble(repo=Path(args.repo), accepted_code=Path(args.accepted_code),
        candidate=Path(args.candidate), map_path=Path(args.map),
        output=Path(args.output))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
