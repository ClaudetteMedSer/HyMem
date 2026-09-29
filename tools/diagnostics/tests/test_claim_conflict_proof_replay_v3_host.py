"""Offline controls for the fresh v64 proof-v3 supervisor adapter."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from tools.diagnostics import claim_conflict_proof_replay_v3_host as host


def diagnostics():
    return Path(__file__).resolve().parents[1]


def test_worker_and_inherited_dependencies_are_reviewed():
    base = diagnostics()
    assert host.sha(base / "claim_conflict_proof_replay_v3.py") == host.WORKER_SHA
    assert host.sha(base / "claim_conflict_proof_replay_v2_host.py") == host.PROOF_V2_HOST_SHA
    assert host.sha(base / "claim_conflict_proof_drift_host.py") == host.AUDIT_HOST_SHA


def test_fresh_stage_network_none_and_extra_workers_read_only():
    base = diagnostics()
    old = host.controller(base / "claim_conflict_proof_replay_v2_host.py",
                          v1_path=base / "claim_conflict_proof_replay_host.py",
                          audit_path=base / "claim_conflict_proof_drift_host.py")
    helper = SimpleNamespace(RUNTIME=Path("/readonly/runtime"),
                             IMAGE="sha256:" + "a" * 64)
    command, mounts = old.configure(helper)
    assert command[command.index("--name") + 1] == "hymem-proof-replay-v3"
    assert command[command.index("--network") + 1] == "none"
    assert (str(host.PROOF_V2_WORKER),
            "/diag/claim_conflict_proof_replay_v2.py", False) in mounts
    assert (str(host.DRIFT_WORKER),
            "/diag/claim_conflict_proof_drift.py", False) in mounts
    assert (str(host.CANDIDATE), "/candidate", False) in mounts
    assert (str(host.CAPTURE), "/capture", False) in mounts
    assert (str(host.WORK), "/work", True) in mounts
    assert not any("runtime-env.json" in word or "deepseek" in word.lower()
                   for word in command)


def test_audit_gate_rejects_nonterminal_or_changed_rows(monkeypatch):
    receipt = {"snapshot_sha256": "a" * 64,
               "candidate_sha256": "b" * 64}
    metadata = {"status": "completed", "snapshot_sha256": "a" * 64,
                "phase1_sha256": host.PHASE1_SHA, "source_unchanged": True,
                "metadata_sha256": "c" * 64,
                "exact": {"schema_before": 63, "schema_after": 64,
                          "proof_nonnull": 0, "digest_before": "a" * 64,
                          "digest_after": "b" * 64},
                "instrumented": {"digest_before": "a" * 64,
                                 "digest_after": "b" * 64,
                                 "row_comparison": {"changed": 1,
                                                    "ordered_equal": False,
                                                    "unordered_equal": False}}}
    audit = SimpleNamespace(
        dependencies=lambda: (object(), object(), object()),
        installed=lambda *_a: receipt,
        configure=lambda *_a: ([], []),
        inspect=lambda *_a: {"status": "exited", "pid": 0,
                             "exit_code": 0, "oom_killed": False},
        project=lambda raw: raw,
    )
    result = {"status": "completed", "networked_runs_started": 0,
              "stages": {"audit": {"container_id": "d" * 64, "metadata": metadata}}}
    helper = SimpleNamespace(read_json=lambda _path: result)
    with pytest.raises(RuntimeError, match="audit_source_or_rows_not_clean"):
        host.audit_gate(object(), audit, helper)


def test_failed_audit_blocks_install_before_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "CAPTURE", tmp_path / "capture")
    monkeypatch.setattr(host, "CANDIDATE", tmp_path / "candidate")
    monkeypatch.setattr(host, "WORK", tmp_path / "work")
    monkeypatch.setattr(host, "OVERRIDES", tmp_path / "overrides")
    monkeypatch.setattr(host, "audit_gate", lambda *_args: (_ for _ in ()).throw(
        RuntimeError("audit_not_completed")))
    with pytest.raises(RuntimeError, match="audit_not_completed"):
        host.remote_install(object(), object(), object(), object(), object(), object())
    assert list(tmp_path.iterdir()) == []


def test_audit_gate_requires_pinned_private_metadata_and_source_identity(monkeypatch):
    manifest = {"hymem/dreaming/phase1.py": host.PHASE1_SHA}
    candidate_sha = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    receipt = {"snapshot_sha256": "a" * 64,
               "candidate_sha256": candidate_sha}
    arm = {"schema_before": 63, "schema_after": 64, "proof_nonnull": 0,
           "digest_before": "b" * 64, "digest_after": "c" * 64}
    metadata = {"status": "completed", "snapshot_sha256": "a" * 64,
                "phase1_sha256": host.PHASE1_SHA, "source_unchanged": True,
                "metadata_sha256": host.AUDIT_METADATA_SHA, "exact": arm,
                "instrumented": {**arm, "row_comparison": {
                    "changed": 0, "ordered_equal": True, "unordered_equal": True}}}
    audit = SimpleNamespace(
        dependencies=lambda: (object(), object(), object()),
        installed=lambda *_a: receipt,
        configure=lambda *_a: ([], []),
        inspect=lambda *_a: {"status": "exited", "pid": 0,
                             "exit_code": 0, "oom_killed": False},
        project=lambda raw: raw,
    )
    result = {"status": "completed", "networked_runs_started": 0,
              "stages": {"audit": {"container_id": "d" * 64, "metadata": metadata}}}
    helper = SimpleNamespace(read_json=lambda _path: result,
                             regular=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(host, "source_manifest", lambda *_args: manifest)
    monkeypatch.setattr(host, "sha", lambda _path: host.AUDIT_METADATA_SHA)
    assert host.audit_gate(object(), audit, helper) == host.AUDIT_METADATA_SHA
    metadata["metadata_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="audit_private_metadata_pin_drift"):
        host.audit_gate(object(), audit, helper)
