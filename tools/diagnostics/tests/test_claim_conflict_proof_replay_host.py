"""No-network controls for the private proof replay controller."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_proof_replay_host as host
from tools.diagnostics import claim_conflict_proof_replay_v2_host as accepted


def clean_audit():
    return {"integrity_ok": True, "foreign_key_findings": 0,
            "canonical_drift_findings": 0, "ledger_count_mismatches": 0,
            "same_generation_disagreeing_groups": 0}


def good_arm(dedup):
    return {"status": "completed", "dedup_enabled": dedup,
            "historical_outcomes": 3, "historical_proofs_after_upgrade": 0,
            "semantic_digest_before_upgrade": "a" * 64,
            "semantic_digest_after_upgrade": "a" * 64,
            "second_initialize_unchanged": True,
            "published_before": 0, "published_after_first": 1,
            "published_after_repeat": 1, "proof_sha256": "b" * 64,
            "proof_reopen_unchanged": True, "proof_repeat_unchanged": True,
            "exact_repeat_unchanged": True,
            "integrity_before": clean_audit(),
            "integrity_after_upgrade": clean_audit(),
            "integrity_after_first": clean_audit(),
            "integrity_after_reopen": clean_audit(),
            "integrity_after_repeat": clean_audit(),
            "database_sha256": "c" * 64}


def good_metadata():
    return {"status": "completed", "capture_sha256": "d" * 64,
            "snapshot_sha256": "e" * 64, "phase1_sha256": "f" * 64,
            "reference_sha256": host.REFERENCE_SHA,
            "dedup_on": good_arm(True), "dedup_off": good_arm(False)}


def test_reviewed_inventory_preserves_v1_history_and_v64_successor():
    assert len(host.OVERRIDE_SHAS) == 7
    rejected_migration = "hymem/core/migrations/062_local_claim_replay_proof.sql"
    accepted_migration = "hymem/core/migrations/064_local_claim_replay_proof.sql"
    assert rejected_migration in host.OVERRIDE_SHAS
    assert rejected_migration not in accepted.OVERRIDE_SHAS
    assert accepted_migration in accepted.OVERRIDE_SHAS
    assert accepted_migration not in host.OVERRIDE_SHAS
    host.reviewed_overrides()
    accepted.reviewed_overrides()
    diagnostics = Path(__file__).resolve().parents[1]
    assert host.sha(diagnostics / "claim_conflict_proof_replay_host.py") == accepted.V1_HOST_SHA
    assert host.WORKER_SHA == accepted.V1_AUDIT_WORKER_SHA
    assert host.sha(diagnostics / "claim_conflict_proof_replay.py") == host.WORKER_SHA
    assert host.OLD_REPLAY_SHA == accepted.OLD_REPLAY_SHA
    assert host.sha(diagnostics / "claim_conflict_instrumented_replay.py") == accepted.OLD_REPLAY_SHA
    assert accepted.sha(accepted.LOCAL_FROZEN / accepted_migration) == (
        accepted.OVERRIDE_SHAS[accepted_migration]
    )


def test_container_is_single_network_none_and_has_no_credentials():
    helper = SimpleNamespace(IMAGE="sha256:" + "f" * 64,
                             RUNTIME=Path("/runtime"))
    command, mounts = host.configure(helper)
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--read-only") + 1] == "--cap-drop"
    assert "--phase1-sha256" in command
    assert command[-1] == host.OVERRIDE_SHAS["hymem/dreaming/phase1.py"]
    assert any(dst == "/capture" and not writable for _, dst, writable in mounts)
    assert any(dst == "/reference/source.sqlite" and not writable
               for _, dst, writable in mounts)
    assert sum(writable for _, _, writable in mounts) == 1
    assert not any("runtime-env.json" in str(value) for value in command)
    assert not any("hermes-net" in str(value) for value in command)


def test_strict_projection_and_verdict():
    raw = good_metadata()
    metadata = host.project(raw)
    receipt = {key: metadata[key] for key in (
        "capture_sha256", "snapshot_sha256", "phase1_sha256",
        "reference_sha256")}
    host.verdict(metadata, receipt)
    for key, value in (("historical_proofs_after_upgrade", 1),
                       ("published_after_first", 0),
                       ("published_after_repeat", 0),
                       ("proof_reopen_unchanged", False),
                       ("exact_repeat_unchanged", False)):
        changed = deepcopy(metadata)
        changed["dedup_off"][key] = value
        with pytest.raises(RuntimeError, match="proof_arm_not_idempotent"):
            host.verdict(changed, receipt)
    changed = deepcopy(raw)
    changed["dedup_on"]["integrity_after_reopen"]["foreign_key_findings"] = 1
    with pytest.raises(RuntimeError, match="proof_arm_not_idempotent"):
        host.verdict(host.project(changed), receipt)
    changed = deepcopy(raw)
    changed["dedup_on"]["proof_sha256"] = "private proof text"
    with pytest.raises(RuntimeError, match="arm_hash_invalid"):
        host.project(changed)


def test_error_projection_is_finite():
    assert host.project({"status": "error", "reason_code": "proof_replay_failed",
                         "error_type": "RuntimeError", "failure_captured": True}) == {
        "status": "error", "reason_code": "proof_replay_failed",
        "error_type": "RuntimeError", "failure_captured": True}
    with pytest.raises(RuntimeError, match="worker_error_projection_invalid"):
        host.project({"status": "error", "reason_code": "private detail",
                      "error_type": "RuntimeError", "failure_captured": True})
