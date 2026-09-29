"""Offline controls for the source-bound semantic policy bundle derivation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "tools/diagnostics/luna_semantic_policy_bundle_v2.py"
spec = importlib.util.spec_from_file_location("luna_semantic_policy_bundle_v2", SOURCE)
assert spec is not None and spec.loader is not None
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_derivation_is_pinned_and_preserves_unmodified_helpers(tmp_path):
    target = tmp_path / "fresh"
    result = bundle.prepare(REPO, target)
    receipt_path = target / "derivation-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert result["code_files"] == 16
    assert result["receipt_sha256"] == digest(receipt_path)
    assert receipt["candidate_files"] == 510
    assert receipt["candidate_inventory_sha256"] == bundle.INVENTORY_SHA
    assert receipt["extraction_identity"] == bundle.EXTRACTION_IDENTITY
    assert result["model_calls"] == 0 and result["launched"] is False
    assert len(receipt["input_sha256"]) == 18
    assert len(receipt["output_sha256"]) == 18
    for relative, expected in receipt["input_sha256"].items():
        assert digest(REPO / relative) == expected
    for relative, expected in receipt["output_sha256"].items():
        assert digest(target / relative) == expected
    for relative in receipt["unchanged_code"]:
        assert (target / "code" / relative).read_bytes() == (REPO / relative).read_bytes()
    host = (target / "code/tools/diagnostics/luna_semantic_probe_host.py").read_text()
    assert "builder.prepare(OLD_CANDIDATE, OLD_MAP, ACCEPTED_CANDIDATE, ACCEPTED_MAP," in host
    assert "builder_path = code / 'tools/diagnostics/luna_semantic_candidate.py'" in host
    assert "source = root / 'code/tools/diagnostics/luna_semantic_candidate_v2.py'" in host
    assert "'turns': 29, 'known_tokens': 500000, 'seconds': 1800" in host
    assert "'model': 'gpt-6-luna'" in host
    run = (target / "code/tools/diagnostics/luna_semantic_probe_run.py").read_text()
    core_sha = digest(target / "code/tools/diagnostics/luna_semantic_probe.py")
    assert "('tools/diagnostics/luna_semantic_probe.py',\n         '" + core_sha + "')" in run
    assert bundle.OLD_CORE_SHA not in run
    assert b"source-grounding-v2" in (target / "code/benchmarks/luna_semantic_canary.py").read_bytes()
    assert b"source-grounding-v1" not in (target / "code/benchmarks/luna_semantic_canary.py").read_bytes()
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        bundle.prepare(REPO, target)
    assert result["receipt_sha256"] == digest(receipt_path)


def test_source_drift_fails_before_output_creation(tmp_path):
    staged = tmp_path / "source"
    staged.mkdir()
    for relative in bundle.INPUT_SHA:
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    corrupted = staged / "benchmarks/luna_semantic_canary.py"
    corrupted.write_bytes(corrupted.read_bytes() + b"\n# altered\n")
    output = tmp_path / "fresh"
    with pytest.raises(ValueError, match="source_pin_invalid"):
        bundle.prepare(staged, output)
    assert not output.exists()


def test_isolated_cli_and_output_immutability(tmp_path):
    target = tmp_path / "bundle"
    process = subprocess.run(
        [sys.executable, "-I", "-B", str(SOURCE), "--repo", str(REPO), "--target", str(target)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    result = json.loads(process.stdout)
    assert result["receipt_sha256"] == digest(target / "derivation-receipt.json")
    assert result["code_files"] == 16
    assert not any(target.rglob("__pycache__"))
    second = subprocess.run(
        [sys.executable, "-I", "-B", str(SOURCE), "--repo", str(REPO), "--target", str(target)],
        capture_output=True, text=True, timeout=30,
    )
    assert second.returncode != 0
    assert result["receipt_sha256"] == digest(target / "derivation-receipt.json")


def test_output_boundaries_and_symlink_guard(tmp_path):
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        bundle.prepare(REPO, REPO / "would_be_inside_repo")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        bundle.prepare(REPO, alias / "bundle")
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        bundle.prepare(REPO, bundle.CANDIDATE / "would_mutate_frozen_candidate")
