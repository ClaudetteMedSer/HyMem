"""Offline checks for the inactive v2 candidate derivation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.diagnostics import luna_semantic_candidate as v1
from tools.diagnostics import luna_semantic_candidate_v2 as builder

SOURCE = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/candidate")
SOURCE_STAMP = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/headless-grounding-source-map.json")
ACCEPTED = Path("/private/tmp/hymem-semantic-step2-v2-candidate-20260929")
ACCEPTED_STAMP = Path("/private/tmp/hymem-semantic-step2-v2-map-20260929.json")
REPO = Path(__file__).resolve().parents[3]


def derive(tmp_path):
    target, stamp = tmp_path / "candidate", tmp_path / "map.json"
    result = builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                             target, stamp, REPO)
    return target, stamp, result


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    return derive(tmp_path_factory.mktemp("semantic-v2"))


def test_exact_inventory_delta_and_source_identity(candidate):
    target, stamp, result = candidate
    accepted_map = json.loads(ACCEPTED_STAMP.read_text())["source_sha256"]
    derived_map = json.loads(stamp.read_text())["source_sha256"]
    assert len(derived_map) == result["files"] == 510
    assert {key for key in accepted_map if accepted_map[key] != derived_map[key]} == {v1.GROUNDING}
    assert derived_map[v1.GROUNDING] == builder.V2_GROUNDING_SHA256
    assert v1.inventory(target) == derived_map
    assert v1.mapping_sha(derived_map) == result["derived_map_sha256"]
    assert v1.sha(stamp.read_bytes()) == result["derived_stamp_sha256"]
    assert (target / v1.GROUNDING).read_bytes() == (REPO / builder.V2_GROUNDING).read_bytes()


def test_source_pins_fail_closed_without_output(tmp_path, monkeypatch):
    target = tmp_path / "candidate"
    stamp = tmp_path / "map.json"
    monkeypatch.setattr(builder, "V2_GROUNDING_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="helper_source_drift"):
        builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                        target, stamp, REPO)
    assert not target.exists() and not stamp.exists()


def test_builder_pin_fails_closed_without_output(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "V1_BUILDER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="v1_builder_drift"):
        builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                        tmp_path / "candidate", tmp_path / "map.json", REPO)
    assert list(tmp_path.iterdir()) == []


def test_accepted_inventory_and_delta_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ACCEPTED_MAP_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source_inventory_identity"):
        derive(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_freshness_and_source_boundaries(candidate, tmp_path):
    target, stamp, _ = candidate
    with pytest.raises(ValueError, match="output_not_fresh"):
        builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                        target, stamp, REPO)
    with pytest.raises(ValueError, match="output_inside_source"):
        builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                        tmp_path / "new", ACCEPTED / "new-map.json", REPO)
    alias = tmp_path / "repo-link"
    alias.symlink_to(REPO, target_is_directory=True)
    with pytest.raises(ValueError, match="path_alias_or_symlink"):
        builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                        tmp_path / "new", tmp_path / "new-map.json", alias)


def _run(candidate: Path, script: str):
    proc = subprocess.run([sys.executable, "-B", "-c", script], cwd=candidate,
                          env=dict(os.environ, PYTHONPATH=str(candidate)),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_cache_identity_and_v1_response_fail_closed(candidate):
    target, _, _ = candidate
    value = _run(target, r'''
import json
from hymem.extraction import contract, grounding
from hymem.extraction.grounding import GroundingSource, Triple

claim = Triple(subject="Ada", predicate="prefers", object="tea", polarity=1, source_message_id=1)
source = GroundingSource(source_message_id=1, content="I prefer tea.")
request, batch = grounding.build_grounding_request([claim], [source])
identity = contract.extraction_contract_identity()
original = grounding._SYSTEM
grounding._SYSTEM += " changed"
different = contract.extraction_contract_identity()
grounding._SYSTEM = original
assert contract.extraction_contract_identity() == identity
old_response = {"schema":"source-grounding-v1", "batch_sha256":batch.batch_sha256,
                "complete":True, "verdicts":[{"index":0,"status":"supported",
                "predicate":"prefers","evidence":[{"source_message_id":1,
                "region":"owned","quote":"I prefer tea."}]}]}
try:
    grounding.parse_grounding_response(json.dumps(old_response), batch)
except grounding.GroundingContractError as exc:
    reason = exc.code
else:
    raise AssertionError("v1 response accepted by v2")
print(json.dumps({"version":grounding.GROUNDING_CONTRACT_VERSION,
                  "request_v2": "source-grounding-v2" in request.system,
                  "identity": identity, "different":different, "reason":reason}))
''')
    assert value["version"] == "source-grounding-v2"
    assert value["request_v2"]
    assert value["different"] != value["identity"]
    assert value["reason"] == "response:schema"


def test_isolated_direct_cli():
    proc = subprocess.run([sys.executable, "-I", "-B", str(Path(builder.__file__)), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "--accepted-stamp" in proc.stdout


def test_changed_sibling_cannot_execute_before_pin_check(tmp_path):
    isolated = tmp_path / "luna_semantic_candidate_v2.py"
    sibling = tmp_path / "luna_semantic_candidate.py"
    marker = tmp_path / "executed"
    shutil.copy2(Path(builder.__file__), isolated)
    sibling.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
    proc = subprocess.run([sys.executable, "-I", "-B", str(isolated), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "v1_builder_drift" in proc.stderr
    assert not marker.exists()
