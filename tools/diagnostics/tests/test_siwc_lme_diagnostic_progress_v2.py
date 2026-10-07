"""Offline receipt regression using the accepted bundle and runner's receipt builder."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import siwc_lme_diagnostic_bundle_v1 as bundle
from tools.diagnostics import siwc_lme_diagnostic_launch_v1 as launch
from tools.diagnostics import siwc_lme_diagnostic_progress_v1 as v1
from tools.diagnostics import siwc_lme_diagnostic_progress_v2 as v2


REPO = Path(__file__).resolve().parents[3]
ACCEPTED = Path("/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle")
CODE = Path("/private/tmp/hymem-repaired-four-root-dtBv2p/bundle/code")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")


def test_runner_receipt_v1_regression_v2_acceptance_and_tamper(tmp_path, monkeypatch):
    root = tmp_path / ".hymem-siwc-lme-diagnostic-preflight-offline1"
    report = bundle.assemble(repo=REPO, accepted_code=CODE,
        candidate=ACCEPTED / "candidate", map_path=ACCEPTED / "source-map.json",
        output=root)
    assert report["candidate_files"] == 514 and report["model_calls"] == 0
    root.chmod(0o700)

    dataset = tmp_path / "invented-dataset.json"
    rows = [{"question_id": f"invented-{index}", "question": f"fixture-{index}"}
        for index in range(4)]
    dataset.write_text(json.dumps(rows), encoding="utf-8")
    dataset_sha = digest(dataset)
    runtime = tmp_path / "runtime" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"offline runtime fixture\n")
    site = runtime.parent.parent / "lib/python3.13/site-packages"
    site.mkdir(parents=True)
    files = {}
    for index in range(309):
        path = site / f"fixture-{index:03d}.txt"
        path.write_bytes(f"offline package {index}\n".encode())
        files[path.name] = digest(path)
    site_sha = hashlib.sha256(json.dumps(files, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()

    runner_path = root / "code" / bundle.RUNNER_RELATIVE
    spec = importlib.util.spec_from_file_location("offline_siwc_receipt_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "DATASET_SHA256", dataset_sha)
    monkeypatch.setattr(runner, "RUNTIME_PATH", runtime)
    monkeypatch.setattr(runner, "RUNTIME_SHA256", digest(runtime))
    monkeypatch.setattr(runner, "RUNTIME_SITE_SHA256", site_sha)
    loaded = {"source_only": False, "root": root, "dataset": dataset,
        "questions": rows, "protocol": object(),
        "prior": SimpleNamespace(SelectedQuestions=lambda _dataset, _count, _protocol: rows),
        "siwc": SimpleNamespace(MAX_INVOCATION=120.0)}
    receipt = runner.receipt_for(root, loaded)
    launch.write_once(root / "launch-receipt.json", receipt)
    receipt_sha = digest(root / "launch-receipt.json")
    assert (root / "launch-receipt.json").read_bytes() == canonical(receipt)

    for reader in (v1, v2):
        monkeypatch.setattr(reader, "ROOT_PARENT", tmp_path)
        monkeypatch.setattr(reader, "ROOT_UID", os.getuid())
        monkeypatch.setattr(reader, "DATASET", dataset)
        monkeypatch.setattr(reader, "DATASET_SHA256", dataset_sha)
        monkeypatch.setattr(reader, "RUNTIME_PATH", runtime)
        monkeypatch.setattr(reader, "RUNTIME_SHA256", digest(runtime))
        monkeypatch.setattr(reader, "RUNTIME_SITE_SHA256", site_sha)

    with pytest.raises(ValueError, match="^receipt_canonical_invalid$"):
        v1.receipt(root, receipt_sha)
    assert v2.receipt(root, receipt_sha) == receipt
    prepared = v2.inspect(root, receipt_sha)
    assert prepared["schema"] == "siwc-lme-diagnostic-progress-v2"
    assert prepared["status"] == "prepared_not_launched"

    receipt_path = root / "launch-receipt.json"
    receipt_path.write_bytes(json.dumps(receipt, sort_keys=True, indent=2).encode("ascii"))
    with pytest.raises(ValueError, match="^receipt_canonical_invalid$"):
        v2.receipt(root, digest(receipt_path))
    changed = dict(receipt, model="wrong-model")
    receipt_path.write_bytes(canonical(changed))
    with pytest.raises(ValueError, match="^receipt_identity_invalid$"):
        v2.receipt(root, digest(receipt_path))
