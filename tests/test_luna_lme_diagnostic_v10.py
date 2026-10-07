"""Offline controls for the repaired four-worker diagnostic entry point."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/diagnostics/luna_lme_diagnostic_v10.py"
CANDIDATE = Path("/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle")
FROZEN = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle")


def runner_module():
    spec = importlib.util.spec_from_file_location("tested_lme_runner_v10", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_receipt_rejects_coercion_extras_and_duplicate_keys(tmp_path, monkeypatch):
    runner = runner_module()
    expected = {"schema": runner.RECEIPT_SCHEMA, "workers": 4,
                "selected_count": 4, "one_shot": True,
                "selected_row_sha256": ["a" * 64] * 4}
    monkeypatch.setattr(runner, "receipt_for", lambda _root, _loaded: expected)
    receipt = tmp_path / "launch-receipt.json"
    attempt = tmp_path / "launch-attempt.json"

    def check(raw: bytes, accepted: bool):
        receipt.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        attempt.write_bytes(runner._canonical(
            {"receipt_sha256": digest, "one_shot": True}))
        if accepted:
            assert runner.verify_launch_receipt(tmp_path, digest, {}) == expected
        else:
            with pytest.raises(ValueError):
                runner.verify_launch_receipt(tmp_path, digest, {})

    check(runner._canonical(expected), True)
    for altered in ({**expected, "workers": True},
                    {**expected, "workers": 4.0},
                    {**expected, "selected_count": 3},
                    {**expected, "extra": "private"},
                    {**expected, "selected_row_sha256": ["b" * 64] * 4}):
        check(runner._canonical(altered), False)
    duplicate = runner._canonical(expected).replace(b'"workers":4',
        b'"workers":4,"workers":4')
    check(duplicate, False)


def test_execution_marker_precedes_campaign_and_prevents_reentry(tmp_path, monkeypatch, capsys):
    runner = runner_module()
    root = tmp_path / ".hymem-lme-diagnostic-testxxxx"
    root.mkdir()
    inventory = root / "source-map.json"
    inventory.write_bytes(b"invented offline fixture")
    events = []
    class Warm:
        @staticmethod
        def BudgetLimits(*items):
            return tuple(items)
    loaded = {"questions": [object() for _ in range(4)], "warm": Warm()}
    monkeypatch.setattr(runner, "load_verified", lambda *args: loaded)
    monkeypatch.setattr(runner, "verify_launch_receipt", lambda *args: (
        events.append("receipt") or {"expected_cgroup": "/invented"}))
    monkeypatch.setattr(runner, "verify_live_containment", lambda *args: (
        events.append("containment") or True))
    def no_provider_campaign(_loaded, **options):
        marker = root / runner.EXECUTION_MARKER
        assert json.loads(marker.read_text()) == {
            "receipt_sha256": "a" * 64, "execution_started": True}
        assert options["workers"] == 4
        assert options["indexing_seconds"] == 10_800
        assert options["campaign_limits"] == runner.MAX_LIMITS["campaign"]
        assert options["question_limits"] == runner.MAX_LIMITS["question"]
        assert options["canary_limits"] == runner.MAX_LIMITS["canary"]
        events.append("campaign")
        return {"run_id": "invented", "selected_denominator": 4,
                "scored_count": 4, "campaign_stop": None}
    monkeypatch.setattr(runner, "run_campaign", no_provider_campaign)
    arguments = ["--root", str(root), "--inventory", str(inventory),
        "--inventory-sha256", runner.ACCEPTED_INVENTORY_SHA256,
        "--dataset", str(root / "dataset"), "--binary", str(root / "binary"),
        "--binary-sha256", runner.BINARY_SHA256,
        "--questions", "4", "--workers", "4", "--run",
        "--receipt-sha256", "a" * 64, "--output-dir", str(root / "run")]
    assert runner.main(arguments) == 0
    assert events == ["receipt", "containment", "campaign"]
    assert runner.main(arguments) == 1
    assert events == ["receipt", "containment", "campaign", "receipt", "containment"]
    assert "unverified" in capsys.readouterr().out


def test_invalid_binary_or_worker_cannot_reach_loader(monkeypatch, tmp_path):
    runner = runner_module()
    called = []
    monkeypatch.setattr(runner, "load_verified", lambda *args: called.append(True))
    arguments = ["--root", str(tmp_path), "--inventory", str(tmp_path / "source-map.json"),
        "--inventory-sha256", runner.ACCEPTED_INVENTORY_SHA256,
        "--dataset", str(tmp_path / "dataset"), "--binary", str(tmp_path / "binary"),
        "--binary-sha256", "0" * 64, "--run", "--workers", "4"]
    assert runner.main(arguments) == 1
    assert called == []
    with pytest.raises(SystemExit):
        runner.main([*arguments[:-1], "3"])
    assert called == []


@pytest.mark.skipif(not CANDIDATE.is_dir() or not FROZEN.is_dir(),
                    reason="accepted source bundles unavailable")
def test_actual_repaired_source_only_graph_is_not_runnable(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(CANDIDATE / "candidate", bundle / "candidate")
    shutil.copytree(FROZEN / "code", bundle / "code")
    shutil.copy2(CANDIDATE / "source-map.json", bundle / "source-map.json")
    shutil.copy2(RUNNER, bundle / "code/tools/diagnostics/luna_lme_diagnostic_v10.py")
    child = r'''
import importlib.util,sys
from pathlib import Path
root=Path(sys.argv[1])
path=root/'code/tools/diagnostics/luna_lme_diagnostic_v10.py'
spec=importlib.util.spec_from_file_location('bound_v10',path)
runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner
spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and loaded['questions']==[]
try: runner.run_campaign(loaded,output=root/'run',campaign_limits=None,
    canary_limits=None,question_limits=None,indexing_seconds=10800,workers=4,
    helper_sha256=runner.DIAGNOSTIC_HELPER_SHA256,containment=lambda _:True)
except ValueError as exc: assert str(exc)=='runnable_source_binding_required'
else: raise AssertionError('source-only graph ran')
print('source-only rejected')
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", child, str(bundle)],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "source-only rejected"


def test_frozen_runner_hashes_unchanged():
    assert hashlib.sha256((ROOT / "tools/diagnostics/luna_lme_diagnostic_v9.py").read_bytes()).hexdigest() == "b3e1135893a715dec4e138c25f6bf3c70f3912dbd5df81014ee7c8f2767fd278"
    assert hashlib.sha256((ROOT / "tools/diagnostics/luna_lme_diagnostic_v8.py").read_bytes()).hexdigest() == "7f96f2ac53039805d8324055edcc0902d7210195e075300eca1b0fb961764f82"
