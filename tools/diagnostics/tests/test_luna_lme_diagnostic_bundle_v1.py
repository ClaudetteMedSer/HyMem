"""Offline source-only assembly from the accepted immutable inputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "luna_lme_diagnostic_bundle_v1.py"
SPEC = importlib.util.spec_from_file_location("luna_lme_diagnostic_bundle_tested", SOURCE)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)

REPO = SOURCE.parents[2]
ACCEPTED_CODE = Path("/private/tmp/hymem-staged-startup-root-vBg9BWn4/bundle/code")
CANDIDATE = Path("/private/tmp/hymem-staged-v1-root-fZvNIRgs/candidate")
MAP = Path("/private/tmp/hymem-staged-v1-root-fZvNIRgs/map.json")


def _available():
    if not (ACCEPTED_CODE.is_dir() and CANDIDATE.is_dir() and MAP.is_file()):
        pytest.skip("accepted immutable source inputs unavailable")


def test_accepted_source_bundle_is_fresh_and_complete(tmp_path):
    _available()
    target = tmp_path / "fresh"
    report = bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
        candidate=CANDIDATE, map_path=MAP, output=target)
    assert report["candidate_files"] == 514
    assert report["model_calls"] == 0
    assert not report["dataset_present"]
    assert not report["binary_present"]
    assert not report["launch_receipt_present"]
    assert (target / "source-map.json").read_bytes() == MAP.read_bytes()
    assert bundle._sha(target / "code/benchmarks/codex_subscription_staged_v1.py") == (
        "c483975d53ca3523708cb589a68a8b0523664312ea259057e48ee01e7dad6ca8")
    assert bundle._sha(target / "candidate/hymem/extraction/chunk.py") == (
        "c644513152e2ddefbe0d0d18dce7d597d18b4ba78ce275dd04ad4b2133ee7b92")
    with pytest.raises(ValueError, match="bundle_input_invalid"):
        bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
            candidate=CANDIDATE, map_path=MAP, output=target)


def test_map_drift_rejected_before_output(tmp_path):
    _available()
    bad_map = tmp_path / "map.json"
    bad_map.write_text("{}", encoding="utf-8")
    target = tmp_path / "never-created"
    with pytest.raises(ValueError, match="candidate_map_shape_invalid"):
        bundle.assemble(repo=REPO, accepted_code=ACCEPTED_CODE,
            candidate=CANDIDATE, map_path=bad_map, output=target)
    assert not target.exists()
