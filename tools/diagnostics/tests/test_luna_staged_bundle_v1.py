"""Offline checks for the inactive staged bundle and generated host."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from tools.diagnostics import luna_staged_bundle_v1 as bundle

REPO = Path(__file__).resolve().parents[3]


def _pins():
    code = bundle.collect_sources(REPO)
    return code, {**{name: bundle.sha(raw) for name, raw in code.items()},
                  **{name: "a" * 64 for name in bundle.PENDING}}


def test_candidate_inventory_and_source_pins():
    mapping = bundle.validate_candidate()
    assert len(mapping) == 514
    assert bundle.sha((bundle.CANDIDATE_MAP).read_bytes()) == bundle.CANDIDATE_MAP_SHA
    code, pins = _pins()
    assert bundle.sha(code["tools/diagnostics/luna_staged_core_v1.py"]) == bundle.LOCAL[
        "tools/diagnostics/luna_staged_core_v1.py"]
    assert bundle.sha(code["benchmarks/codex_subscription_staged_v1.py"]) != bundle.LOCAL[
        "benchmarks/codex_subscription_staged_v1.py"]
    assert bundle.HOST not in code and set(bundle.PENDING).isdisjoint(code)
    assert len(bundle.derive_host(pins)) > 10000


def test_pending_dependencies_fail_closed_before_output(tmp_path):
    # Keep the missing-dependency counterexample after the real modules land.
    repo = tmp_path / "incomplete-repo"
    repo.mkdir()
    for name in bundle.LOCAL:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / name).read_bytes())
    target = tmp_path / "bundle"
    with pytest.raises(ValueError, match="pending_source_missing"):
        bundle.prepare(repo, target)
    assert not target.exists()


def test_host_schedule_identity_and_one_shot_guards():
    _, pins = _pins()
    source = bundle.derive_host(pins).decode()
    assert "luna-staged-probe-launch-v1" in source
    assert "'units': 8" in source
    assert "'unit_turns': 3, 'unit_known_tokens': 100000, 'unit_seconds': 240" in source
    assert "'turns': 29, 'known_tokens': 500000, 'seconds': 1800" in source
    assert "'controls': [9, 12, 13, 17, 19, 21]" in source
    assert "'canaries': ['table', 'prose']" in source
    assert "'quota_floor_percent': 25" in source
    assert "'runtime_max_seconds': 1930, 'timeout_stop_seconds': 10" in source
    assert "'tasks_max': 256, 'memory_max_bytes': 4294967296" in source
    assert bundle.EXTRACTION_IDENTITY in source
    assert "write_once(root / 'launch-attempt.json'" in source
    assert "need(not (root / 'run').exists(), 'output_already_exists')" in source
    assert "luna_staged_candidate_v1.py" in source
    assert "proof['files'] == 514" in source
    for name in bundle.PENDING:
        assert name in source


def test_bad_pins_and_output_boundaries(tmp_path):
    _, pins = _pins()
    pins["tools/diagnostics/luna_staged_core_v1.py"] = "0" * 64
    with pytest.raises(ValueError, match="host_pins_invalid"):
        bundle.derive_host(pins)
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        bundle.prepare(REPO, REPO / "tools/diagnostics/new-bundle")
    target = tmp_path / "symlink"
    target.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="output_boundary_invalid"):
        bundle.prepare(REPO, target)


def test_candidate_faults_rejected(tmp_path):
    stamp = tmp_path / "map.json"
    stamp.write_bytes(bundle.CANDIDATE_MAP.read_bytes())
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "stray").write_text("x")
    with pytest.raises(ValueError, match="candidate_inventory_drift"):
        bundle.validate_candidate(candidate, stamp)
    stamp.unlink()
    stamp.symlink_to(bundle.CANDIDATE_MAP)
    with pytest.raises(ValueError, match="candidate_boundary_invalid"):
        bundle.validate_candidate(candidate, stamp)
