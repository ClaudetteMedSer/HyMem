"""Offline checks for the fresh retained-response cold replay controller."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from tools.diagnostics import claim_conflict_cold_replay_host as host


LOCAL_DIAGNOSTICS = Path(
    "/Users/attavanwestreenen/AGprojects/HyMem/tools/diagnostics"
)


def _controller():
    return host.controller(
        LOCAL_DIAGNOSTICS / host.V3_HOST.name,
        v2_path=LOCAL_DIAGNOSTICS / host.V2_HOST.name,
        v1_path=LOCAL_DIAGNOSTICS / host.V1_HOST.name,
        audit_path=LOCAL_DIAGNOSTICS / host.AUDIT_HOST.name,
    )


def test_pinned_controller_and_worker_are_unchanged():
    assert host.sha(LOCAL_DIAGNOSTICS / host.V3_HOST.name) == host.V3_HOST_SHA
    assert host.sha(LOCAL_DIAGNOSTICS / host.V3_WORKER.name) == host.V3_WORKER_SHA
    assert host.sha(LOCAL_DIAGNOSTICS / host.COLD_SUITE_HOST.name) == host.COLD_SUITE_HOST_SHA


def test_exact_offline_command_uses_new_phase1_and_private_work_only():
    old = _controller()
    helper = SimpleNamespace(RUNTIME=Path("/readonly/runtime"), IMAGE="sha256:" + "a" * 64)
    command, mounts = old.configure(helper)
    assert command[command.index("--name") + 1] == "hymem-cold-replay-proof-v1"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--phase1-sha256") + 1] == host.NEW_PHASE1_SHA
    assert command[command.index("--pull") + 1] == "never"
    assert "--read-only" in command
    assert (str(host.CANDIDATE), "/candidate", False) in mounts
    assert (str(host.CAPTURE), "/capture", False) in mounts
    assert (str(host.WORK), "/work", True) in mounts
    assert (str(host.WORKER), "/diag/claim_conflict_proof_replay.py", False) in mounts
    assert sum(writable for _, _, writable in mounts) == 1
    assert not any("runtime-env.json" in word or "deepseek" in word.lower()
                   for word in command)


def test_candidate_derivation_rejects_altered_base_and_is_one_file(monkeypatch):
    parent = {f"synthetic/{i}.py": f"{i:064x}" for i in range(480)}
    parent["hymem/dreaming/phase1.py"] = host.PARENT_PHASE1_SHA
    monkeypatch.setattr(host, "PARENT_CANDIDATE_SHA", host.digest(parent))
    monkeypatch.setattr(host, "NEW_CANDIDATE_SHA", host.digest({
        **parent, "hymem/dreaming/phase1.py": host.NEW_PHASE1_SHA,
    }))
    changed = host.derive_candidate(parent)
    assert len(changed) == 481
    assert changed["hymem/dreaming/phase1.py"] == host.NEW_PHASE1_SHA
    assert {key for key in changed if changed[key] != parent[key]} == {
        "hymem/dreaming/phase1.py"
    }
    bad = dict(parent, **{"synthetic/0.py": "f" * 64})
    with pytest.raises(RuntimeError, match="baseline_candidate_manifest_drift"):
        host.derive_candidate(bad)


def test_baseline_gate_requires_original_receipt_and_terminal_verdict(tmp_path, monkeypatch):
    root = tmp_path / "proof-replay-v3"
    root.mkdir()
    result_path = root / "result.json"
    result = {"status": "completed", "networked_runs_started": 0,
              "stages": {"replay": {"container_id": "a" * 64,
                                    "metadata": {"status": "completed"}}}}
    result_path.write_text(json.dumps(result))
    monkeypatch.setattr(host, "V3_ROOT", root)
    monkeypatch.setattr(host, "BASELINE_RESULT_SHA", host.sha(result_path))
    receipt = {"candidate_sha256": host.PARENT_CANDIDATE_SHA,
               "phase1_sha256": host.PARENT_PHASE1_SHA}
    checked = []
    baseline = SimpleNamespace(
        installed=lambda *_: receipt,
        inspect=lambda *_: {"status": "exited", "pid": 0,
                            "exit_code": 0, "oom_killed": False},
        configure=lambda *_: ([], []),
        project=lambda raw: raw,
        verdict=lambda metadata, observed: checked.append((metadata, observed)),
    )
    helper = SimpleNamespace(read_json=lambda path: json.loads(path.read_text()))
    assert host.baseline_gate(None, baseline, None, None, helper) == receipt
    assert len(checked) == 1
    receipt["phase1_sha256"] = host.NEW_PHASE1_SHA
    with pytest.raises(RuntimeError, match="baseline_receipt_candidate_drift"):
        host.baseline_gate(None, baseline, None, None, helper)
    receipt["phase1_sha256"] = host.PARENT_PHASE1_SHA
    result["stages"]["replay"]["metadata"] = {"status": "error"}
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="baseline_result_pin_drift"):
        host.baseline_gate(None, baseline, None, None, helper)


def test_cold_candidate_gate_rejects_altered_inventory_or_receipt(tmp_path, monkeypatch):
    root = tmp_path / "cold-suite"
    root.mkdir()
    candidate = root / "candidate"
    candidate.mkdir()
    expected = {"hymem/dreaming/phase1.py": host.NEW_PHASE1_SHA}
    manifest = root / "candidate-manifest.json"
    manifest.write_text(json.dumps(expected))
    suite = root / "claim_conflict_cold_pytest.py"
    suite.write_text("reviewed synthetic suite")
    receipt = {
        "candidate_sha256": host.digest(expected),
        "parent_candidate_sha256": host.PARENT_CANDIDATE_SHA,
        "candidate_phase1_sha256": host.NEW_PHASE1_SHA,
        "proof_result_sha256": host.BASELINE_RESULT_SHA,
        "overlay_sha256": host.NEW_OVERLAY_SHA,
        "baseline_proof_only": True,
        "new_candidate_replay_verified": False,
        "host_sha256": host.sha(suite),
    }
    install = root / "install.json"
    install.write_text(json.dumps(receipt))
    monkeypatch.setattr(host, "COLD_MANIFEST", manifest)
    monkeypatch.setattr(host, "COLD_INSTALL", install)
    monkeypatch.setattr(host, "COLD_SUITE_HOST", suite)
    monkeypatch.setattr(host, "COLD_SUITE_HOST_SHA", host.sha(suite))
    monkeypatch.setattr(host, "COLD_CANDIDATE", candidate)
    actual = dict(expected)
    v3 = SimpleNamespace(inventory=lambda _: actual)
    helper = SimpleNamespace(read_json=lambda path: json.loads(path.read_text()))
    assert host.cold_candidate_gate(v3, helper, expected) == receipt
    actual = {"hymem/dreaming/phase1.py": "0" * 64}
    with pytest.raises(RuntimeError, match="cold_candidate_inventory_drift"):
        host.cold_candidate_gate(v3, helper, expected)
    actual = dict(expected)
    receipt["new_candidate_replay_verified"] = True
    install.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="cold_suite_receipt_drift"):
        host.cold_candidate_gate(v3, helper, expected)


def test_capture_copy_gate_rejects_changed_bytes_and_binding(tmp_path, monkeypatch):
    source = tmp_path / "retained"
    copied = tmp_path / "copy"
    source.mkdir()
    copied.mkdir()
    reference = "a" * 64
    snapshot = b"synthetic private snapshot"
    (source / "prepersist-001.sqlite").write_bytes(snapshot)
    (copied / "prepersist-001.sqlite").write_bytes(snapshot)
    capture = {"source_sha256": reference,
               "database_sha256": host.sha(source / "prepersist-001.sqlite")}
    for folder in (source, copied):
        (folder / "prepersist-001.json").write_text(json.dumps(capture))
    monkeypatch.setattr(host, "CAPTURE", copied)
    v3 = SimpleNamespace(CAPTURE=source)
    helper = SimpleNamespace(read_json=lambda path: json.loads(path.read_text()))
    assert host.capture_copy_gate(v3, helper, reference) == (
        host.sha(copied / "prepersist-001.json"),
        host.sha(copied / "prepersist-001.sqlite"),
    )
    (copied / "prepersist-001.sqlite").write_bytes(b"changed snapshot")
    with pytest.raises(RuntimeError, match="replay_capture_copy_drift"):
        host.capture_copy_gate(v3, helper, reference)
    (copied / "prepersist-001.sqlite").write_bytes(snapshot)
    with pytest.raises(RuntimeError, match="replay_capture_binding_drift"):
        host.capture_copy_gate(v3, helper, "b" * 64)


def test_inherited_verdict_rejects_wrong_proof_or_churn():
    inherited = _controller()
    receipt = {"capture_sha256": "a" * 64,
               "snapshot_sha256": "b" * 64,
               "phase1_sha256": host.NEW_PHASE1_SHA,
               "reference_sha256": "c" * 64}
    audit = {"integrity_ok": True, "foreign_key_findings": 0,
             "canonical_drift_findings": 0, "ledger_count_mismatches": 0,
             "same_generation_disagreeing_groups": 0}
    arm = {"historical_proofs_after_upgrade": 0,
           "semantic_digest_before_upgrade": "d" * 64,
           "semantic_digest_after_upgrade": "d" * 64,
           "second_initialize_unchanged": True,
           "published_before": 0, "published_after_first": 1,
           "published_after_repeat": 1,
           "proof_reopen_unchanged": True,
           "proof_repeat_unchanged": True,
           "exact_repeat_unchanged": True}
    for key in ("integrity_before", "integrity_after_upgrade",
                "integrity_after_first", "integrity_after_reopen",
                "integrity_after_repeat"):
        arm[key] = dict(audit)
    metadata = {"status": "completed", **receipt,
                "dedup_on": dict(arm), "dedup_off": dict(arm)}
    assert inherited.verdict(metadata, receipt) is None
    metadata["phase1_sha256"] = host.PARENT_PHASE1_SHA
    with pytest.raises(RuntimeError, match="worker_source_identity_drift"):
        inherited.verdict(metadata, receipt)
    metadata["phase1_sha256"] = host.NEW_PHASE1_SHA
    metadata["dedup_off"]["exact_repeat_unchanged"] = False
    with pytest.raises(RuntimeError, match="proof_arm_not_idempotent"):
        inherited.verdict(metadata, receipt)
    metadata["status"] = "error"
    with pytest.raises(RuntimeError, match="worker_source_identity_drift"):
        inherited.verdict(metadata, receipt)


def test_review_gate_blocks_remote_install_without_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "REVIEWED_FINAL_CANDIDATE", False)
    monkeypatch.setattr(host, "ROOT", tmp_path / "uncreated")
    with pytest.raises(RuntimeError, match="final_cold_replay_review_pending"):
        host.remote_install(None, None, None, None)
    assert list(tmp_path.iterdir()) == []


def test_main_failed_metadata_is_bounded_and_never_contacts_remote(monkeypatch, capsys):
    monkeypatch.setattr(host, "REVIEWED_FINAL_CANDIDATE", False)
    monkeypatch.setattr("sys.argv", ["cold-replay", "install"])
    monkeypatch.setattr(host, "install", lambda: pytest.fail("install called"))
    host.main()
    assert json.loads(capsys.readouterr().out) == {
        "status": "operation_failed_inspect_before_retry"
    }
