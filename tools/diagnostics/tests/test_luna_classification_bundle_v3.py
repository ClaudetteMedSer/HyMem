"""Offline checks for the source-bound, inactive classification-v3 bundle."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.diagnostics import luna_classification_bundle_v3 as bundle


REPO = Path(__file__).resolve().parents[3]


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_derivation_binds_v3_and_preserves_frozen_controls(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    proof = bundle.prepare(REPO, target)
    receipt = json.loads((target / "derivation-receipt.json").read_bytes())
    code = target / "code"
    assert proof["code_files"] == 21
    assert len(receipt["output_sha256"]) == 23
    assert receipt["schema"] == "luna-classification-v3-probe-bundle-v1"
    assert receipt["candidate_inventory_sha256"] == bundle.INVENTORY_SHA
    assert receipt["candidate_files"] == 513
    assert receipt["extraction_identity"] == bundle.EXTRACTION_IDENTITY
    assert receipt["model_calls"] == proof["model_calls"] == 0
    assert not receipt["launched"] and not proof["launched"]
    assert proof["receipt_sha256"] == _digest((target / "derivation-receipt.json").read_bytes())
    assert receipt["input_sha256"] == dict(sorted(bundle.INPUT_SHA.items()))
    assert receipt["output_sha256"] == dict(sorted(proof["output_sha256"].items()))
    for relative, expected in receipt["output_sha256"].items():
        source = target / relative
        assert source.is_file() and not source.is_symlink()
        assert _digest(source.read_bytes()) == expected
        if source.suffix == ".py":
            ast.parse(source.read_text(), filename=str(source))

    host = (code / "tools/diagnostics/luna_semantic_probe_host.py").read_text()
    core = (code / "tools/diagnostics/luna_semantic_probe.py").read_text()
    entry = (code / "tools/diagnostics/luna_semantic_probe_run.py").read_text()
    canary = (code / "benchmarks/luna_semantic_canary.py").read_text()
    stage = (code / "benchmarks/luna_semantic_stage_accounting.py").read_text()
    startup = (target / "adapter-v2.py").read_text()
    replay = (target / "verdict-replay.py").read_text()
    assert f"'tools/diagnostics/luna_classification_candidate.py': '{bundle.old.BUILDER_SHA}'" in host
    assert f"'tools/diagnostics/luna_classification_candidate_v3.py': '{bundle.EXTRA_SHA[bundle.BUILDER]}'" in host
    assert "source = root / 'code/tools/diagnostics/luna_classification_candidate_v3.py'" in host
    assert "builder.prepare(OLD_CANDIDATE, OLD_MAP, ACCEPTED_CANDIDATE, ACCEPTED_MAP," in host
    assert f"'hymem/extraction/grounding_classification_v3.py': '{bundle.EXTRA_SHA[bundle.CONTRACT]}'" in host
    assert f"'hymem/extraction/grounding_classification_gate_v3.py': '{bundle.EXTRA_SHA[bundle.GATE]}'" in host
    assert f"'benchmarks/codex_subscription_classification_v3.py': '{receipt['output_sha256']['code/' + bundle.ADAPTER]}'" in host
    assert "grounding_classification_v3 as grounding" in entry
    assert "codex_subscription_classification_v3 as adapter" in entry
    assert "return adapter.ClassificationSubscriptionClient(" in entry
    assert "grounding_classification_gate_v3 import _v2_source" in core
    assert "from benchmarks.codex_subscription_classification_v3 import ClassificationSubscriptionClient" in core
    assert "hymem/extraction/grounding_classification_v3.py" in core
    assert "luna-classification-canary-v3" in canary
    assert bundle.EXTRACTION_IDENTITY in canary and bundle.EXTRACTION_IDENTITY in host
    assert bundle.EXTRA_SHA[bundle.GATE] in stage
    assert bundle.FAMILY in core and bundle.FAMILY in entry and bundle.FAMILY in startup
    assert bundle.FAMILY in replay
    assert "MAX_NEW_TURNS = 29" in core
    assert "MAX_KNOWN_TOKENS = 500_000" in core
    assert "MAX_SECONDS = 1800" in core
    assert "LIMITS = {'turns': 29, 'known_tokens': 500000, 'seconds': 1800" in host
    assert "'quota_floor_percent': 25" in host
    assert "'memory_max_bytes': 4294967296" in host
    assert "batch.canonical_json==trusted" in replay
    assert "batch.batch_sha256==expected_sha" in replay
    assert "grounding.validate_request(request,batch)" in replay
    assert "code/hymem/extraction/grounding_classification_v1.py" not in receipt["output_sha256"]
    assert "code/hymem/extraction/grounding_classification_v2.py" not in receipt["output_sha256"]
    assert "code/hymem/extraction/grounding_classification_gate_v1.py" not in receipt["output_sha256"]
    assert "code/hymem/extraction/grounding_classification_gate_v2.py" not in receipt["output_sha256"]


def test_staged_adapter_rewrites_exact_candidate_path_and_keeps_ast_pin(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    bundle.prepare(REPO, target)
    raw = (target / "code" / bundle.ADAPTER).read_bytes()
    original = (REPO / bundle.ADAPTER).read_bytes()
    before = b'Path(__file__).resolve().parents[1] / "hymem/extraction/grounding_classification_v3.py"'
    after = b'Path(__file__).resolve().parents[2] / "candidate/hymem/extraction/grounding_classification_v3.py"'
    assert original.count(before) == 1 and raw.count(after) == 1
    assert raw == original.replace(before, after)
    assert bundle.EXTRA_SHA[bundle.CONTRACT].encode() in raw
    assert b"50d0ba72b0ea5290fba0541e42bc03f1880a3aa70c8014310a7b1b22be6adede" in raw


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
