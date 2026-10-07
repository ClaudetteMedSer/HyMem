"""Offline controls for the new SIWC host source and metadata boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import siwc_lme_diagnostic_bundle_v1 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v1 as preflight
from tools.diagnostics import siwc_lme_diagnostic_launch_v1 as launch
from tools.diagnostics import siwc_lme_diagnostic_progress_v1 as progress
from tools.diagnostics import siwc_lme_diagnostic_source_install_v1 as install
from benchmarks import chatgpt_plan_lme_v1 as bridge


REPO = Path(__file__).resolve().parents[3]
ACCEPTED = Path("/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle")
CODE = Path("/private/tmp/hymem-repaired-four-root-dtBv2p/bundle/code")


def test_real_source_bundle_and_manifest(tmp_path):
    if not (ACCEPTED / "candidate").is_dir():
        pytest.skip("accepted local bundle unavailable")
    target = tmp_path / "fresh"
    report = bundle.assemble(repo=REPO, accepted_code=CODE,
        candidate=ACCEPTED / "candidate", map_path=ACCEPTED / "source-map.json",
        output=target)
    assert report["candidate_files"] == 514
    assert report["code_files"] == 25
    assert report["model_calls"] == 0
    manifest = preflight.source_manifest(target)
    assert len(manifest) == 540
    assert manifest["code/" + bundle.RUNNER_RELATIVE] == bundle.RUNNER_SHA256
    assert len(preflight.archive_bytes(target)) < 64 * 1024 * 1024
    extra = target / "code" / "unapproved.py"
    extra.write_text("pass")
    with pytest.raises(ValueError, match="source_file_set_invalid"):
        preflight.source_manifest(target)


def test_unit_and_systemd_policy_from_frozen_root():
    root = Path("/home/atta/.hymem-siwc-lme-diagnostic-preflight-abcdefgh")
    receipt = {"unit": "hymem-siwc-lme-diagnostic-preflight-abcdefgh.service",
        "runtime_path": "/home/atta/.hymem-siwc-runtime-v1/bin/python"}
    command = launch.command(root, receipt, "a" * 64)
    assert command[command.index("-I") - 1] == receipt["runtime_path"]
    assert all(flag in command for flag in ("--property=Restart=no",
        "--property=KillMode=control-group", "--property=RuntimeMaxSec=14530s",
        "--property=TimeoutStopSec=10s", "--property=TasksMax=256",
        "--property=MemoryMax=4294967296", "--property=CPUQuota=200%"))
    assert command[-1] == "--run"
    assert "--questions" in command and command[command.index("--questions") + 1] == "4"
    assert "--workers" in command and command[command.index("--workers") + 1] == "4"
    assert progress.ROOT_NAME.fullmatch(root.name)
    assert hashlib.sha256((REPO / install.SOURCE).read_bytes()).hexdigest() == install.SOURCE_SHA256
    projected = {"schema": install.SCHEMA, "root": str(root),
        "unit": receipt["unit"], "prepared": True, "model_calls": 0,
        "receipt_sha256": "a" * 64}
    assert install.project(projected, root=str(root), returncode=0) == projected
    projected["model_calls"] = False
    assert install.project(projected, root=str(root), returncode=0) is None


def test_recursive_cleanup_rejects_live_descendant(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "CGROUP_ROOT", tmp_path)
    group = tmp_path / "unit"
    child = group / "nested"
    child.mkdir(parents=True)
    for node in (group, child):
        (node / "cgroup.procs").write_text("")
        (node / "cgroup.threads").write_text("")
        (node / "cgroup.events").write_text("populated 0\n")
    assert launch.recursive_empty(group)
    (child / "cgroup.procs").write_text("12345\n")
    assert not launch.recursive_empty(group)


def observed(calls=1, successes=1, failures=0, turns=1, tokens=9):
    return {"schema": "siwc_lme_summary_v1", "calls": calls,
        "successes": successes, "failures": failures,
        "internal_http_attempts": successes,
        "provider_internal_retries_known": False,
        "admitted_turns": turns, "known_tokens": tokens,
        "usage_complete": True,
        "timing_seconds": {"total": 1.0, "admission": 0.25, "http": 0.75},
        "timing_saturated": False, "first_failure": None,
        "last_failure_code": None}


def test_ten_views_share_five_ledgers_without_double_counting():
    observations = {}
    rows = []
    budget = {"turns": 10, "known_tokens": 90,
        "questions": {"canary": {"turns": 2, "known_tokens": 18}}}
    for index in range(5):
        prefix = "canary" if index == 0 else f"question.{index-1}"
        qid = "canary" if index == 0 else f"q-{index-1:04d}"
        if index:
            budget["questions"][qid] = {"turns": 2, "known_tokens": 18}
        ordinary, structured = observed(turns=2, tokens=18), observed(turns=2, tokens=18)
        observations[prefix + ".ordinary"] = {"status": "observed", "summary": ordinary}
        observations[prefix + ".structured"] = {"status": "observed", "summary": structured}
        rows.append({"question_id": qid, "ordinary": ordinary, "structured": structured,
            "ledger": {"admitted_turns": 2, "known_tokens": 18, "usage_complete": True}})
    projection = {"schema": "siwc_lme_pilot_projection_v1", "canary": rows[0],
        "questions": rows[1:], "aggregate": {"calls": 10, "successes": 10,
            "failures": 0, "internal_http_attempts": 10,
            "admitted_turns": 10, "known_tokens": 90}}
    assert progress.pilot(projection, observations, budget) == projection
    bad = json.loads(json.dumps(projection))
    bad["aggregate"]["known_tokens"] = 180
    with pytest.raises(ValueError, match="siwc_pilot_aggregate_invalid"):
        progress.pilot(bad, observations, budget)
    observations["question.3.structured"] = {"status": "unknown"}
    with pytest.raises((KeyError, ValueError)):
        progress.pilot(projection, observations, budget)


def test_first_fault_and_timing_accept_finite_metadata_only():
    row = observed()
    row["calls"] = row["failures"] = 1
    row["successes"] = row["internal_http_attempts"] = 0
    row["admitted_turns"] = row["known_tokens"] = 0
    row["first_failure"] = {"code": "refresh_denied", "phase": "admission",
        "turn_admitted": False, "unknown_usage": False}
    row["last_failure_code"] = "refresh_denied"
    assert progress.summary(row) == row
    row["first_failure"]["provider_text"] = "private"
    with pytest.raises(ValueError, match="first_failure_invalid"):
        progress.summary(row)
    del row["first_failure"]["provider_text"]
    row["timing_seconds"]["http"] = float("nan")
    with pytest.raises(ValueError, match="siwc_summary_timing_invalid"):
        progress.summary(row)
    # AtomicCheckpoint sanitizes free-form runner question failures to this
    # finite artifact code; SIWC's transport first fault remains separately typed.
    assert "unspecified_failure" in progress.STOP_CODES


def test_reader_first_fault_matches_source_bound_bridge_validation():
    fault = {"code": "http_failure", "phase": "http",
        "turn_admitted": True, "unknown_usage": True,
        "http_status": 503, "body_shape": "error_object",
        "media_type_class": "json",
        "wire_observation": {"header_defect": "none", "parsed_header_count": 3,
            "transfer_encoding": "chunked", "content_encoding": "missing",
            "body_prefix": "json_prefix", "body_bytes": 40,
            "body_truncated": False, "sse_validation": "not_checked"},
        "stream_observation": {"event_type": "response.failed",
            "terminal_status": "failed", "terminal_model_matches": True,
            "terminal_output_kind": "missing", "terminal_channel": "missing",
            "terminal_content_kind": "missing", "terminal_output_state": "missing",
            "finalized_item_count": 0, "output_reconstructed": False}}
    assert progress.first_failure(fault) == bridge.project_first_failure(fault)
    for path, bad in (("wire_observation", "provider-private"),
                      ("stream_observation", "provider-private")):
        changed = json.loads(json.dumps(fault))
        changed[path]["provider_text"] = bad
        with pytest.raises(ValueError):
            progress.first_failure(changed)
        with pytest.raises(ValueError):
            bridge.project_first_failure(changed)


def test_completed_stage_accounting_matches_question_ledger():
    ids = [f"source-{index}" for index in range(4)]
    checkpoint = {"expected_ids": ids, "entries": {
        qid: {"status": "completed"} for qid in ids}}
    budget = {"questions": {f"q-{index:04d}": {"turns": 1, "known_tokens": 3}
        for index in range(4)}}
    stages = {qid: {"judge": {"attempts": 1, "returned": 1,
        "turns": 1, "known_tokens": 3}} for qid in ids}
    progress.stage_accounting(stages, checkpoint, budget, True)
    stages[ids[0]]["judge"]["known_tokens"] = 4
    with pytest.raises(ValueError, match="terminal_stage_ledger_invalid"):
        progress.stage_accounting(stages, checkpoint, budget, True)


def test_terminal_four_rows_ten_views_and_partial_owner_failure():
    ids = [f"source-{index}" for index in range(4)]
    entries = {qid: {"status": "completed", "row": {"correct": index % 2 == 0}}
        for index, qid in enumerate(ids)}
    check = {"run_id": "sha256:" + "a"*64, "expected_ids": ids, "entries": entries}
    counts = {"expected": 4, "attempted": 4, "unique_attempted": 4,
        "total_attempts": 4, "completed": 4, "failed": 0, "missing": 0}
    budget = {"turns": 10, "known_tokens": 90, "reserved": 0, "in_flight": 0,
        "usage_complete": True, "stopped": False, "stop_code": None,
        "timings": {"preflight_seconds": 1.0, "model_seconds": 1.0,
                    "cleanup_seconds": 0.0},
        "known_tokens_scope": "completed_turns_only_failed_turn_usage_unknown",
        "token_cap_kind": "stop_before_next_observed_usage",
        "first_failure": None, "resource_fault": None,
        "resource_observation": {"current": 0, "peak": 5, "limit": 256, "denials": 0},
        "questions": {key: {"turns": 2, "known_tokens": 18, "in_flight": 0,
            "usage_complete": True, "stopped": False}
            for key in ("canary", *(f"q-{i:04d}" for i in range(4)))}}
    observations = {}
    rows = []
    for index in range(5):
        prefix = "canary" if index == 0 else f"question.{index-1}"
        qid = "canary" if index == 0 else f"q-{index-1:04d}"
        a, b = observed(turns=2, tokens=18), observed(turns=2, tokens=18)
        observations[prefix + ".ordinary"] = {"status": "observed", "summary": a}
        observations[prefix + ".structured"] = {"status": "observed", "summary": b}
        rows.append({"question_id": qid, "ordinary": a, "structured": b,
            "ledger": {"admitted_turns": 2, "known_tokens": 18,
                       "usage_complete": True}})
    projection = {"schema": "siwc_lme_pilot_projection_v1", "canary": rows[0],
        "questions": rows[1:], "aggregate": {"calls": 10, "successes": 10,
            "failures": 0, "internal_http_attempts": 10,
            "admitted_turns": 10, "known_tokens": 90}}
    stages = {qid: {"judge": {"attempts": 2, "returned": 2,
        "turns": 2, "known_tokens": 18}} for qid in ids}
    result = {"schema": progress.RUN_SCHEMA, "run_id": check["run_id"],
        "canonical_r9_artifact": False, "official_model_score": False,
        "selected_denominator": 4, "scored_count": 4, "correct_count": 2,
        "incorrect_count": 2, "quality_accuracy_full_selected": 0.5,
        "failed_or_unscored_count": 0, "strict_unhealthy_count": 0,
        "canary": {"structural_valid": True, "model_gold_match": False,
                   "completion_calls": 2}, "campaign_stop": None,
        "owner_failure": None, "stage_accounting": stages,
        "stage_accounting_clean": True, "checkpoint_counts": counts,
        "budget": budget, "siwc_observations": observations,
        "siwc_pilot_projection": projection, "diagnostic_complete": True}
    assert progress.terminal(result, check, {}, counts, 4, 0) == result
    result["selected_denominator"] = True
    with pytest.raises(ValueError, match="terminal_identity_invalid"):
        progress.terminal(result, check, {}, counts, 4, 0)
    result["selected_denominator"] = 4
    result["siwc_pilot_projection"] = None
    result["diagnostic_complete"] = False
    result["campaign_stop"] = "refresh_denied"
    result["owner_failure"] = {"phase": "owner_open", "code": "refresh_denied"}
    observations["question.3.structured"] = {"status": "unknown"}
    assert progress.terminal(result, check, {}, counts, 4, 0) == result
    result["diagnostic_complete"] = True
    with pytest.raises(ValueError, match="terminal_complete_mismatch"):
        progress.terminal(result, check, {}, counts, 4, 0)


def test_cleanup_without_result_is_explicitly_not_clean(tmp_path, monkeypatch):
    receipt_sha = "a" * 64
    (tmp_path / "launch-attempt.json").write_text(json.dumps({
        "receipt_sha256": receipt_sha, "one_shot": True}))
    monkeypatch.setattr(progress, "checked_root", lambda root: root)
    monkeypatch.setattr(progress, "receipt", lambda root, digest: {
        "selected_count": 4, "expected_cgroup": "/user.slice/fake"})
    monkeypatch.setattr(progress, "runtime", lambda _receipt: ("failed_exit_cleaned", True))
    result = progress.inspect(tmp_path, receipt_sha)
    assert result["status"] == "terminal_failure_without_result"
    assert result["runtime_cleanup_verified"] is True
    assert result["completed_diagnostic_and_clean"] is False
