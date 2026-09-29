"""Network-free controls for the sealed proof-v2 drift-audit controller."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_proof_drift_host as host


def fake_helper():
    return SimpleNamespace(RUNTIME=Path("/readonly/runtime"),
                           IMAGE="sha256:" + "a" * 64)


def test_container_has_no_network_or_credentials():
    command, mounts = host.configure(fake_helper(), "b" * 64)
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--snapshot-sha256") + 1] == "b" * 64
    assert command[command.index("--phase1-sha256") + 1] == host.PHASE1_SHA
    assert (str(host.SNAPSHOT), "/capture/source.sqlite", False) in mounts
    assert (str(host.CANDIDATE), "/candidate", False) in mounts
    assert (str(host.AUDIT_V1), "/diag/claim_conflict_proof_replay_v1.py", False) in mounts
    assert (str(host.OLD_REPLAY), "/diag/claim_conflict_instrumented_replay.py", False) in mounts
    assert (str(host.WORK), "/work", True) in mounts
    assert not any("runtime-env.json" in part or "deepseek" in part.lower()
                   for part in command)


def test_unreviewed_worker_blocks_upload_before_network(monkeypatch):
    monkeypatch.setattr(host.subprocess, "run", lambda *_a, **_k: pytest.fail("network"))
    monkeypatch.setattr(host, "WORKER_SHA", None)
    with pytest.raises(RuntimeError, match="audit_worker_unreviewed"):
        host.install()


@pytest.mark.parametrize("status,exit_code,metadata", [
    ("completed", 0, {"status": "error", "reason_code": "proof_replay_failed",
                      "error_type": "RuntimeError", "failure_captured": True}),
    ("failed", 0, {"status": "error", "reason_code": "proof_replay_failed",
                   "error_type": "RuntimeError", "failure_captured": True}),
    ("failed", 1, {"status": "completed"}),
])
def test_source_gate_rejects_nonmatching_terminal_failure(monkeypatch, status, exit_code, metadata):
    snapshot_sha = "b" * 64
    receipt = {"source_files": 481, "phase1_sha256": host.PHASE1_SHA,
               "snapshot_sha256": snapshot_sha}
    proof = SimpleNamespace(
        dependencies=lambda: (object(), object(), object()),
        installed=lambda *_a: receipt,
        configure=lambda _h: ([], []),
        inspect=lambda *_a: {"status": "exited", "pid": 0, "exit_code": exit_code,
                             "oom_killed": False},
        project=lambda raw: raw,
    )
    result = {"status": status, "networked_runs_started": 0,
              "stages": {"replay": {"container_id": "c" * 64, "metadata": metadata}}}
    helper = SimpleNamespace(read_json=lambda _path: result)
    monkeypatch.setattr(host, "sha", lambda _path: snapshot_sha)
    with pytest.raises(RuntimeError):
        host.source_gate(proof, helper)


def test_source_gate_accepts_only_exact_failed_proof_v2(monkeypatch):
    snapshot_sha = "b" * 64
    receipt = {"source_files": 481, "phase1_sha256": host.PHASE1_SHA,
               "snapshot_sha256": snapshot_sha}
    metadata = {"status": "error", "reason_code": "proof_replay_failed",
                "error_type": "RuntimeError", "failure_captured": True}
    proof = SimpleNamespace(
        dependencies=lambda: (object(), object(), object()),
        installed=lambda *_a: receipt,
        configure=lambda _h: ([], []),
        inspect=lambda *_a: {"status": "exited", "pid": 0, "exit_code": 1,
                             "oom_killed": False},
        project=lambda raw: raw,
    )
    result = {"status": "failed", "networked_runs_started": 0,
              "stages": {"replay": {"container_id": "c" * 64, "metadata": metadata}}}
    helper = SimpleNamespace(read_json=lambda _path: result)
    monkeypatch.setattr(host, "sha", lambda _path: snapshot_sha)
    assert host.source_gate(proof, helper) == (receipt, snapshot_sha)


def valid_arm(instrumented=False):
    arm = {"schema_before": 63, "schema_after": 64, "proof_nonnull": 0,
           "digest_before": "a" * 64, "digest_after": "b" * 64,
           "digest_after_repeat": "b" * 64,
           "inventory_before_sha256": "c" * 64,
           "inventory_after_sha256": "d" * 64,
           "table_changes": [{"schema": "main", "table_id": "vec_chunks_rowids",
                              "before": ["table", 2, 0, 0],
                              "after": ["shadow", 2, 0, 0]}]}
    if instrumented:
        arm["row_comparison"] = {"tables": 20, "changed": 0,
                                 "ordered_equal": True, "unordered_equal": True,
                                 "changed_table_ids": []}
    return arm


def test_project_accepts_static_metadata_only():
    raw = {"status": "completed", "snapshot_sha256": "a" * 64,
           "phase1_sha256": host.PHASE1_SHA, "metadata_sha256": "b" * 64,
           "source_unchanged": True, "exact": valid_arm(),
           "instrumented": valid_arm(instrumented=True)}
    projected = host.project(raw)
    assert projected["exact"]["table_changes"][0]["table_id"] == "vec_chunks_rowids"
    assert projected["instrumented"]["row_comparison"]["changed"] == 0
    assert "private" not in projected


@pytest.mark.parametrize("bad_id", ["my_private_table", "sha256:x", "x" * 300])
def test_project_rejects_private_table_identifier(bad_id):
    arm = valid_arm()
    arm["table_changes"][0]["table_id"] = bad_id
    with pytest.raises(RuntimeError, match="audit_table_id_invalid"):
        host.project_arm(arm, instrumented=False)
