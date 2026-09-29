"""Offline integrity checks for the classification bundle derivation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.diagnostics import luna_classification_bundle as bundle


REPO = Path(__file__).resolve().parents[3]


def test_fresh_bundle_is_source_bound_and_inactive(tmp_path):
    target = tmp_path / "fresh"
    proof = bundle.prepare(REPO, target)
    receipt = json.loads((target / "derivation-receipt.json").read_bytes())
    assert proof["launched"] is False and proof["model_calls"] == 0
    assert receipt["candidate_files"] == 513
    assert receipt["candidate_inventory_sha256"] == bundle.INVENTORY_SHA
    assert receipt["extraction_identity"] == bundle.EXTRACTION_IDENTITY
    assert len(receipt["output_sha256"]) == proof["code_files"] + 2
    for relative, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((target / relative).read_bytes()).hexdigest() == expected
    assert (target / "code/tools/diagnostics/luna_semantic_cases.py").read_bytes() == (
        REPO / "tools/diagnostics/luna_semantic_cases.py").read_bytes()
    generated = (target / "code/tools/diagnostics/luna_semantic_probe.py").read_text()
    assert '"batch": batch.canonical_json' in generated
    assert "client.complete_grounding(request, batch)" in generated
    assert "client.complete_grounding(request2, batch2)" in generated


def test_existing_output_is_never_overwritten(tmp_path):
    target = tmp_path / "present"
    target.mkdir()
    marker = target / "sentinel"
    marker.write_text("owned")
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        bundle.prepare(REPO, target)
    assert marker.read_text() == "owned"


def test_pinned_source_drift_fails_before_output_creation(tmp_path, monkeypatch):
    original = bundle.INPUT_SHA
    bad = dict(original)
    bad["tools/diagnostics/luna_semantic_probe.py"] = "0" * 64
    monkeypatch.setattr(bundle, "INPUT_SHA", bad)
    target = tmp_path / "absent"
    with pytest.raises(ValueError, match="source_pin_invalid"):
        bundle.prepare(REPO, target)
    assert not target.exists()


def test_default_factory_resolves_pinned_classification_client(tmp_path):
    target = tmp_path / "fresh"
    bundle.prepare(REPO, target)
    shutil.copytree(bundle.CANDIDATE, target / "candidate")
    script = r'''
import sys, tempfile
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from tools.diagnostics import luna_semantic_probe_run as entry
from tools.diagnostics import luna_semantic_probe as core
from tools.diagnostics import luna_semantic_probe_host as host
from tools.diagnostics import luna_semantic_cases as cases
from hymem.extraction import grounding_classification_v1 as grounding, chunk
from benchmarks import extraction_canary as gold
from benchmarks import luna_semantic_canary as semantic
from benchmarks import luna_semantic_stage_accounting as stage
from benchmarks import codex_subscription_classification_v1 as adapter
warm = adapter.warm
loaded = (host, core, cases, grounding, gold, chunk, semantic, stage, warm, warm.concurrent)
entry.preflight = lambda *a: ({}, loaded, {}, ())
class Reached(Exception): pass
def client(binary, budget, key, limits, **kwargs):
    assert key == 'control-00' and limits.turns == 1
    assert kwargs['session_factory'].__mro__[1] is warm.WarmSession
    raise Reached
adapter.ClassificationSubscriptionClient = client
def campaign(**kwargs):
    kwargs['client_factory']('control-00', (1, 100000, 240), object())
core.run_campaign = campaign
root = Path(tempfile.mkdtemp(prefix='classification-factory-offline-'))
try:
    entry.execute(root, '1' * 64, containment=lambda *a: None)
except Reached:
    pass
else:
    raise AssertionError('default classification client was not reached')
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                             str(target / "candidate"), str(target / "code")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
