"""Independent, network-free admission checks for the private diagnostic."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_cold_pytest_v2 as suite_adapter
from tools.diagnostics import claim_conflict_proof_pytest as suite_contract
from tools.diagnostics import claim_conflict_v64_dream_host as host
from tools.diagnostics import claim_conflict_proof_replay_host as proof_contract


@pytest.mark.parametrize("damage", [
    None, "candidate_sha256", "phase1_sha256", "host_sha256",
    "worker_sha256", "baseline_candidate_sha256", "baseline_result_sha256",
    "source_files", "baseline_proof_only", "new_candidate_replay_verified",
    "failed", "network", "running", "oom", "claim_churn",
])
def test_cold_proof_gate_requires_actual_final_candidate_replay(
    tmp_path, monkeypatch, damage,
):
    receipt = {
        "source_files": 481, "candidate_sha256": host.CANDIDATE_SHA,
        "phase1_sha256": host.PHASE1_SHA, "host_sha256": host.PROOF_HOST_SHA,
        "worker_sha256": host.PROOF_WORKER_SHA,
        "baseline_candidate_sha256": host.BASELINE_CANDIDATE_SHA,
        "baseline_result_sha256": host.BASELINE_PROOF_RESULT_SHA,
        "baseline_proof_only": True, "new_candidate_replay_verified": False,
        "capture_sha256": "a" * 64, "snapshot_sha256": "b" * 64,
        "reference_sha256": "c" * 64,
    }
    audit = dict(integrity_ok=True, foreign_key_findings=0,
                 canonical_drift_findings=0, ledger_count_mismatches=0,
                 same_generation_disagreeing_groups=0)
    arm = dict(historical_proofs_after_upgrade=0,
               semantic_digest_before_upgrade="d" * 64,
               semantic_digest_after_upgrade="d" * 64,
               second_initialize_unchanged=True, published_before=0,
               published_after_first=1, published_after_repeat=1,
               proof_reopen_unchanged=True, proof_repeat_unchanged=True,
               exact_repeat_unchanged=True)
    for suffix in ("before", "after_upgrade", "after_first", "after_reopen", "after_repeat"):
        arm["integrity_" + suffix] = audit.copy()
    metadata = {"status": "completed", **receipt,
                "dedup_on": deepcopy(arm), "dedup_off": deepcopy(arm)}
    result = dict(status="completed", networked_runs_started=0,
                  stages={"replay": {"container_id": "e" * 64, "metadata": metadata}})
    state = dict(status="exited", pid=0, exit_code=0, oom_killed=False)
    if damage in receipt:
        old = receipt[damage]
        receipt[damage] = (not old if type(old) is bool else
                           0 if type(old) is int else "0" * 64)
    elif damage == "failed":
        result["status"] = "failed"
    elif damage == "network":
        result["networked_runs_started"] = 1
    elif damage == "running":
        state["status"], state["pid"] = "running", 1
    elif damage == "oom":
        state["oom_killed"] = True
    elif damage == "claim_churn":
        metadata["dedup_on"]["exact_repeat_unchanged"] = False
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    monkeypatch.setattr(host, "PROOF_ROOT", tmp_path)
    monkeypatch.setattr(host, "PROOF_RESULT_SHA", host.sha(path))
    helper = SimpleNamespace(read_json=lambda p: json.loads(p.read_text()))
    proof = SimpleNamespace(
        dependencies=lambda: (None, None, helper), installed=lambda *_: receipt,
        inspect=lambda *_: state, configure=lambda *_: ([], []),
        project=lambda raw: raw, verdict=proof_contract.verdict,
    )
    def forbidden(*_args, **_kwargs):
        raise AssertionError("proof admission must not start a process")
    monkeypatch.setattr(host.subprocess, "run", forbidden)
    monkeypatch.setattr(host.subprocess, "Popen", forbidden)
    if damage is None:
        assert host.proof_gate(proof) == host.sha(path)
    else:
        with pytest.raises(RuntimeError):
            host.proof_gate(proof)


@pytest.fixture
def suite_gate_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(suite_contract, "MIN_COLLECTED", suite_adapter.MIN_COLLECTED)
    monkeypatch.setattr(suite_contract, "CANDIDATE_SHA", host.CANDIDATE_SHA)
    expected = {
        "host_sha256": host.SUITE_HOST_SHA,
        "candidate_sha256": host.CANDIDATE_SHA,
        "overlay_sha256": host.SUITE_OVERLAY_SHA,
        "proof_result_sha256": host.BASELINE_PROOF_RESULT_SHA,
        "parent_candidate_sha256": host.BASELINE_CANDIDATE_SHA,
        "candidate_phase1_sha256": host.PHASE1_SHA,
        "baseline_proof_only": True,
        "new_candidate_replay_verified": False,
    }
    counts = dict(collected=7752, passed=7748, failed=0, skipped=4,
                  errors=0, exit_code=0)
    result = {
        "status": "passed", "networked_runs_started": 0,
        "container": {"container_id": "a" * 64},
        "metadata": {
            "status": "passed", "full_runs_started": 1,
            "candidate_sha256": host.CANDIDATE_SHA,
            "test_inventory_sha256": "b" * 64,
            "application_unchanged": True, "pinned_source_unchanged": True,
            "collect": {**counts, "passed": 0, "skipped": 0},
            "full": counts,
        },
    }
    state = dict(status="exited", pid=0, exit_code=0, oom_killed=False)
    helper = SimpleNamespace(read_json=lambda path: json.loads(path.read_text()))
    suite = SimpleNamespace(
        dependencies=lambda: (None, None, helper),
        pins=lambda *_: expected,
        inspect=lambda *_: state,
        project_worker=suite_contract.project_worker,
    )
    imports = []
    adapter = SimpleNamespace(configure_base=lambda: suite)

    def import_pinned(path, sha, _name):
        assert path == host.SUITE_HOST
        assert sha == host.SUITE_HOST_SHA
        imports.append(path)
        return adapter

    def no_process(*_args, **_kwargs):
        raise AssertionError("admission checks must not start a process")

    monkeypatch.setattr(host, "SUITE_ROOT", tmp_path)
    monkeypatch.setattr(host, "import_pinned", import_pinned)
    monkeypatch.setattr(host.subprocess, "run", no_process)
    monkeypatch.setattr(host.subprocess, "Popen", no_process)

    def seal():
        (tmp_path / "install.json").write_text(json.dumps(expected))
        path = tmp_path / "result.json"
        path.write_text(json.dumps(result))
        digest = host.sha(path)
        monkeypatch.setattr(host, "SUITE_RESULT_SHA", digest)
        return digest

    return SimpleNamespace(expected=expected, result=result, state=state,
                           seal=seal, root=tmp_path, imports=imports)


def test_full_suite_gate_accepts_only_complete_pinned_evidence(suite_gate_fixture):
    fixture = suite_gate_fixture
    digest = fixture.seal()
    before = deepcopy(fixture.result)
    assert host.suite_gate() == digest
    assert fixture.result == before
    assert len(fixture.imports) == 1


@pytest.mark.parametrize("mutation", [
    "suite_failed", "network_used", "bad_candidate", "wrong_overlay",
    "wrong_replay", "wrong_parent", "wrong_phase1", "false_baseline", "false_cold_replay",
    "worker_failed", "source_changed", "application_changed",
    "duplicate_run", "test_failure", "test_error", "missing_test",
    "collection_drift", "too_few_tests", "still_running", "live_pid",
    "nonzero_exit", "oom", "missing_container", "result_tampered",
])
def test_full_suite_gate_rejects_unproven_or_damaged_evidence(
    suite_gate_fixture, mutation,
):
    fixture = suite_gate_fixture
    result, metadata = fixture.result, fixture.result["metadata"]
    if mutation == "suite_failed":
        result["status"] = "failed"
    elif mutation == "network_used":
        result["networked_runs_started"] = 1
    elif mutation == "bad_candidate":
        metadata["candidate_sha256"] = "0" * 64
    elif mutation == "wrong_overlay":
        fixture.expected["overlay_sha256"] = "0" * 64
    elif mutation == "wrong_replay":
        fixture.expected["proof_result_sha256"] = "0" * 64
    elif mutation == "wrong_parent":
        fixture.expected["parent_candidate_sha256"] = "0" * 64
    elif mutation == "wrong_phase1":
        fixture.expected["candidate_phase1_sha256"] = "0" * 64
    elif mutation == "false_baseline":
        fixture.expected["baseline_proof_only"] = False
    elif mutation == "false_cold_replay":
        fixture.expected["new_candidate_replay_verified"] = True
    elif mutation == "worker_failed":
        metadata["status"] = "tests_failed"
    elif mutation == "source_changed":
        metadata["pinned_source_unchanged"] = False
    elif mutation == "application_changed":
        metadata["application_unchanged"] = False
    elif mutation == "duplicate_run":
        metadata["full_runs_started"] = 2
    elif mutation == "test_failure":
        metadata["full"]["failed"] = 1
    elif mutation == "test_error":
        metadata["full"]["errors"] = 1
    elif mutation == "missing_test":
        metadata["full"]["passed"] -= 1
    elif mutation == "collection_drift":
        metadata["full"]["collected"] -= 1
    elif mutation == "too_few_tests":
        metadata["collect"]["collected"] = 1
        metadata["full"].update(collected=1, passed=1, skipped=0)
    elif mutation == "still_running":
        fixture.state["status"] = "running"
    elif mutation == "live_pid":
        fixture.state["pid"] = 123
    elif mutation == "nonzero_exit":
        fixture.state["exit_code"] = 1
    elif mutation == "oom":
        fixture.state["oom_killed"] = True
    elif mutation == "missing_container":
        result["container"] = {}
    fixture.seal()
    if mutation == "result_tampered":
        with (fixture.root / "result.json").open("a") as stream:
            stream.write(" ")
    with pytest.raises(RuntimeError):
        host.suite_gate()


def test_unsealed_result_refuses_before_reading_dependencies(
    suite_gate_fixture, monkeypatch,
):
    monkeypatch.setattr(host, "SUITE_RESULT_SHA", None)
    with pytest.raises(RuntimeError, match="full_suite_result_unreviewed"):
        host.suite_gate()
    assert suite_gate_fixture.imports == []
