"""Source-only v3/v5 integration controls with the accepted candidate bytes."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile
import pytest

from tools.diagnostics import luna_lme_diagnostic_bundle_v5 as bundle
from tools.diagnostics import luna_lme_diagnostic_host_preflight_v5 as host
from tools.diagnostics import luna_lme_diagnostic_launch_v5 as launch
from tools.diagnostics import luna_lme_diagnostic_progress_v6 as progress
from tools.diagnostics import luna_lme_diagnostic_v5 as runner


REPO = Path(__file__).resolve().parents[1]
ACCEPTED = Path("/private/tmp/hymem-lme-diagnostic-offline-assembly-v5")


def test_actual_archive_and_new_receipt(monkeypatch, tmp_path):
    if not (ACCEPTED / "source-map.json").is_file():
        pytest.skip("accepted offline source assembly is unavailable on this host")
    root = tmp_path / ".hymem-lme-diagnostic-sourcev5"
    summary = bundle.assemble(repo=REPO, accepted_code=ACCEPTED / "code",
        candidate=ACCEPTED / "candidate", map_path=ACCEPTED / "source-map.json",
        output=root)
    assert summary["candidate_files"] == 514
    assert len(summary["code_sha256"]) == 12
    assert summary["code_sha256"]["benchmarks/codex_subscription_warm_v5.py"] == runner.PINS[
        "benchmarks/codex_subscription_warm_v5.py"]
    assert summary["code_sha256"]["benchmarks/codex_subscription_warm_v6.py"] == runner.PINS[
        "benchmarks/codex_subscription_warm_v6.py"]
    assert summary["code_sha256"]["benchmarks/codex_subscription_staged_v3.py"] == runner.PINS[
        "benchmarks/codex_subscription_staged_v3.py"]
    manifest = host.source_manifest(root)
    assert len(manifest) == 527
    assert sum(name.startswith("code/") for name in manifest) == 12
    with tarfile.open(fileobj=BytesIO(host.archive_bytes(root)), mode="r:") as archive:
        names = archive.getnames()
    assert names[0] == "manifest.json"
    assert len(names) == 528
    assert set(names[1:]) == set(manifest)

    binary = tmp_path / "invented-binary"
    binary.write_bytes(b"invented binary for source-only receipt test")
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    monkeypatch.setattr(launch, "_root", lambda path: path)
    monkeypatch.setattr(launch, "BINARY_SHA256", binary_sha)
    receipt = launch.receipt_for(root, runner)
    assert receipt["schema"] == "luna-lme-diagnostic-launch-v5"
    assert receipt["source_sha256"] == progress.PINS
    receipt_path = root / "launch-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    (root / "launch-attempt.json").write_text(json.dumps(
        {"receipt_sha256": receipt_sha, "one_shot": True}))
    loaded = {"root": root, "code": root / "code", "binary": binary,
        "questions": [{"question_id": f"invented-{index}"} for index in range(4)]}
    assert runner.verify_launch_receipt(root, receipt_sha, loaded) == receipt
