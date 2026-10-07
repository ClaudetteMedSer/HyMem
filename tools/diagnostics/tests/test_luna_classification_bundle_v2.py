"""Focused source and output-boundary checks for the inactive v2 bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.diagnostics import luna_classification_bundle_v2 as bundle


REPO = Path(__file__).resolve().parents[3]


def test_derivation_binds_v2_and_retains_pinned_builder_chain(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    proof = bundle.prepare(REPO, target)
    receipt = json.loads((target / "derivation-receipt.json").read_bytes())
    code = target / "code"
    assert proof["code_files"] == 21
    assert len(receipt["output_sha256"]) == 23
    assert receipt["candidate_inventory_sha256"] == bundle.INVENTORY_SHA
    assert receipt["extraction_identity"] == bundle.EXTRACTION_IDENTITY
    assert receipt["model_calls"] == proof["model_calls"] == 0
    assert not receipt["launched"] and not proof["launched"]
    for relative, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((target / relative).read_bytes()).hexdigest() == expected
    host = (code / "tools/diagnostics/luna_semantic_probe_host.py").read_text()
    entry = (code / "tools/diagnostics/luna_semantic_probe_run.py").read_text()
    core = (code / "tools/diagnostics/luna_semantic_probe.py").read_text()
    assert "'tools/diagnostics/luna_classification_candidate.py': '" + bundle.old.BUILDER_SHA + "'" in host
    assert "source = root / 'code/tools/diagnostics/luna_classification_candidate_v2.py'" in host
    assert "grounding_classification_v2 as grounding" in entry
    assert "codex_subscription_classification_v2 as adapter" in entry
    assert "grounding_classification_gate_v2 import _v2_source" in core
    assert "from benchmarks.codex_subscription_classification_v2 import ClassificationSubscriptionClient" in core
    assert "hymem/extraction/grounding_classification_v2.py" in core
    assert "grounding_classification_v1.py" not in receipt["output_sha256"]
    assert "code/hymem/extraction/grounding_classification_v1.py" not in receipt["output_sha256"]


def test_existing_output_rejected_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "already-there"
    target.mkdir()
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        bundle.prepare(REPO, target)
    assert list(target.iterdir()) == []


def test_candidate_map_drift_rejected_before_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory_dir = tmp_path / "inventory"
    inventory_dir.mkdir()
    bad = inventory_dir / "map.json"
    bad.write_bytes(bundle.INVENTORY.read_bytes() + b" ")
    monkeypatch.setattr(bundle, "INVENTORY", bad)
    target = tmp_path / "bundle"
    with pytest.raises(ValueError, match="source_pin_invalid"):
        bundle.prepare(REPO, target)
    assert not target.exists()


def test_source_pin_drift_rejected_before_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(bundle.INPUT_SHA, bundle.BUILDER, "0" * 64)
    target = tmp_path / "bundle"
    with pytest.raises(ValueError, match="source_pin_invalid"):
        bundle.prepare(REPO, target)
    assert not target.exists()
