from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import benchmarks.strictness as strictness

from benchmarks.strictness import (
    AtomicCheckpoint,
    BenchmarkIntegrityError,
    build_manifest,
    content_hash,
    converge_indexing,
    code_hash,
    deterministic_smoke,
    deterministic_split,
    export_checkpoint_without_recompute,
    freeze_calibration,
    load_calibration,
    publish_checkpoint_artifact,
    reconcile_results,
    read_artifact_or_pointer,
    select_protocol_ids,
    sanitize_for_artifact,
    strict_accuracy,
    usage_snapshot,
    write_immutable_artifact,
)


def _manifest(ids=("q1", "q2", "q3"), *, config=None):
    effective_config = {"top_k": 3, "label_free_answer_path": True}
    if config:
        effective_config.update(config)
    return build_manifest(
        benchmark="unit",
        code_sha256=content_hash("code"),
        data_sha256=content_hash("data"),
        config=effective_config,
        models={"reader": "stub", "judge": "stub"},
        seed=7,
        expected_ids=ids,
        protocol_split="full",
    )


def test_strict_reconciliation_keeps_missing_and_parse_failures_wrong():
    reconciled = reconcile_results(
        ["q1", "q2", "q3", "q4"],
        [
            {"question_id": "q1", "correct": True},
            {"question_id": "q2", "correct": False},
            {"question_id": "q3", "correct": None, "judge_error": True},
        ],
    )

    assert [row["question_id"] for row in reconciled.rows] == [
        "q1", "q2", "q3", "q4"
    ]
    assert [row["correct"] for row in reconciled.rows] == [True, False, False, False]
    assert strict_accuracy(reconciled.rows) == 0.25
    assert (reconciled.expected, reconciled.attempted, reconciled.completed) == (4, 3, 2)
    assert (reconciled.failed, reconciled.missing) == (2, 1)
    assert reconciled.failure_ids == ("q3", "q4")
    assert reconciled.rows[-1]["benchmark_failure"] == "missing_prediction"


@pytest.mark.parametrize(
    "expected,rows,match",
    [
        (["q1", "q1"], [], "duplicate expected"),
        (["q1"], [{"question_id": "other", "correct": True}], "unknown result"),
        (
            ["q1"],
            [
                {"question_id": "q1", "correct": True},
                {"question_id": "q1", "correct": False},
            ],
            "duplicate result",
        ),
        (["q1"], [{"question_id": "q1", "correct": "yes"}], "malformed verdict"),
        (["q1"], [{"correct": True}], "question_id"),
    ],
)
def test_reconciliation_rejects_ambiguous_or_malformed_evidence(expected, rows, match):
    with pytest.raises(BenchmarkIntegrityError, match=match):
        reconcile_results(expected, rows)


def test_manifest_is_reproducible_and_every_identity_axis_is_hashed():
    first = _manifest()
    second = _manifest()
    assert first == second
    assert first["run_id"].startswith("sha256:")
    assert first["code_hash"].startswith("sha256:")
    assert first["config_hash"].startswith("sha256:")
    assert first["model_hash"].startswith("sha256:")
    assert first["data_hash"].startswith("sha256:")
    assert first["expected_ids_hash"].startswith("sha256:")
    assert first["expected_count"] == 3
    assert first["development_only"] is True
    assert _manifest(config={"top_k": 4})["run_id"] != first["run_id"]

    with pytest.raises(BenchmarkIntegrityError, match="explicitly declare"):
        build_manifest(
            benchmark="unit", code_sha256=content_hash("code"),
            data_sha256=content_hash("data"), config={"top_k": 3},
            models={}, seed=0, expected_ids=["q1"], protocol_split="full",
        )


@pytest.mark.parametrize(
    "extra_config",
    [
        {"exploratory_non_comparable": True},
        {"exploratory_label_steering": True},
        {"scored_run": False},
        {"label_free_answer_path": False},
    ],
)
def test_holdout_manifest_is_development_only_for_exploratory_or_unscored_runs(
    tmp_path: Path, extra_config: dict,
):
    ids = ["q1", "q2", "q3", "q4"]
    config = {"label_free_answer_path": True, **extra_config}
    models = {"reader": "stub"}
    receipt_path = tmp_path / "receipt.json"
    receipt = freeze_calibration(
        receipt_path, benchmark="unit", dataset_hash=content_hash("data"),
        ids=ids, config=config, models=models, seed=4, dev_fraction=0.5,
    )
    manifest = build_manifest(
        benchmark="unit", code_sha256=content_hash("code"),
        data_sha256=content_hash("data"), config=config, models=models, seed=4,
        expected_ids=receipt["holdout_ids"], protocol_split="holdout",
        calibration=receipt,
    )
    assert manifest["development_only"] is True
    assert manifest["official_comparable"] is False


@pytest.mark.parametrize(
    "field,value", [
        ("exploratory_non_comparable", "false"),
        ("exploratory_label_steering", 0),
        ("scored_run", None),
    ],
)
def test_manifest_rejects_non_boolean_protocol_disclosures(field, value):
    with pytest.raises(BenchmarkIntegrityError, match=field):
        _manifest(config={field: value})


def test_code_hash_includes_prompt_and_config_assets(tmp_path: Path):
    package = tmp_path / "surface"
    package.mkdir()
    prompt = package / "instructions.md"
    prompt.write_text("first prompt", encoding="utf-8")
    config = package / "defaults.yaml"
    config.write_text("top_k: 3", encoding="utf-8")
    before = code_hash([package], root=tmp_path)
    prompt.write_text("changed prompt", encoding="utf-8")
    after = code_hash([package], root=tmp_path)
    assert after != before


def test_indexing_convergence_repeats_until_budget_is_clear():
    exhausted = iter((True, True, False))
    calls = 0

    def dream():
        nonlocal calls
        calls += 1
        return {"budget_exhausted": next(exhausted), "chunks_processed": 50}

    result = converge_indexing(
        dream, status=lambda: {"pending_chunks": 0},
        max_cycles=4, timeout_s=10,
    )
    assert calls == 3
    assert result["cycles"] == 3
    assert result["complete"] is True and result["healthy"] is True


def test_indexing_never_converges_or_quarantines_fails_loudly():
    with pytest.raises(strictness.IndexingConvergenceError) as never:
        converge_indexing(
            lambda: {"budget_exhausted": True},
            status=lambda: {"pending_chunks": 2},
            max_cycles=2, timeout_s=10,
        )
    assert never.value.summary["cycles"] == 2
    assert never.value.summary["failure_reason"] == "max_cycles_exhausted"

    with pytest.raises(strictness.IndexingConvergenceError) as quarantine:
        converge_indexing(
            lambda: {"budget_exhausted": False},
            status=lambda: {
                "pending_chunks": 0, "quarantined_chunks": 1,
            },
            max_cycles=2, timeout_s=10, require_healthy=True,
        )
    assert quarantine.value.summary["complete"] is True
    assert quarantine.value.summary["healthy"] is False

    with pytest.raises(strictness.IndexingConvergenceError) as vector_backlog:
        converge_indexing(
            lambda: {"budget_exhausted": False},
            status=lambda: {
                "pending_chunks": 0, "pending_message_embeddings": 1,
            },
            max_cycles=2, timeout_s=10,
        )
    assert vector_backlog.value.summary["failure_reason"] == "max_cycles_exhausted"


def test_manifest_and_checkpoint_never_serialize_recursive_secrets(tmp_path: Path):
    secret = "super-secret-bearer-value"
    config = {
        "label_free_answer_path": True,
        "base_url": (
            f"https://user:{secret}@example.test/v1?api_key={secret}&mode=strict"
        ),
        "extra_body": {"authorization": f"Bearer {secret}", "temperature": 0},
    }
    manifest = build_manifest(
        benchmark="unit", code_sha256=content_hash("code"),
        data_sha256=content_hash("data"), config=config,
        models={"reader": "stub", "api-key": secret}, seed=0,
        expected_ids=["q1"], protocol_split="full",
    )
    serialized = json.dumps(manifest)
    assert secret not in serialized
    assert "example.test/v1" in serialized
    assert manifest["config"]["extra_body"]["temperature"] == 0
    ledger = AtomicCheckpoint(
        tmp_path / "run.json", manifest=manifest, expected_ids=["q1"]
    )
    assert secret not in ledger.path.read_text()
    assert sanitize_for_artifact({"password": secret})["password"]["redacted"] is True
    ledger.record(
        "q1", row=None,
        failure=f"request failed: https://user:{secret}@example.test/?token={secret}",
    )
    assert secret not in ledger.path.read_text()

    artifact = tmp_path / "final.json"
    evidence_text = f"The literal answer is password: {secret}"
    write_immutable_artifact(artifact, {
        "manifest": manifest,
        "config": config,  # callers cannot accidentally republish raw config
        "question": evidence_text,
        "per_question": [{"error": f"Bearer: {secret}"}],
    })
    saved = json.loads(artifact.read_text())
    assert secret not in json.dumps(saved["config"])
    assert secret not in saved["per_question"][0]["error"]
    assert saved["question"] == evidence_text


def test_artifact_sanitizer_uses_opaque_markers_not_guessable_secret_hashes():
    secret = "1234"
    raw_url = (
        f"https://alice:{secret}@example.test/v1?deployment=blue&"
        f"X-Amz-Signature={secret}#token={secret}"
    )
    sanitized = sanitize_for_artifact({
        "password": secret,
        "cookie": f"session={secret}",
        "set-cookie": f"session={secret}; Secure",
        "endpoint_url": raw_url,
        "service": raw_url,
        "endpoint_urls": [raw_url],
        "failure": (
            f"request {raw_url} failed; Authorization: Bearer {secret}; "
            f"cookie={secret}"
        ),
        "failure_chain": [
            f"Authorization=Bearer-{secret}", f"Set-Cookie: session={secret}",
        ],
        "detail": f"Authorization: Bearer {secret}",
        "notes": [f"Cookie: session={secret}"],
        "auth_headers": [
            "Authorization: Basic dXNlcjpTRUNSRVQ=",
            "Proxy-Authorization: Bearer PROXY_SECRET",
            "Cookie: sid=SECRET; refresh=SECRET2",
            "Set-Cookie: sid=SECRET; HttpOnly; refresh=SECRET2",
        ],
        "oauth_url": "https://example.test/callback?client_assertion=SECRET",
        "question": f"The literal string password: {secret} is benchmark data.",
    })
    wire = json.dumps(sanitized, sort_keys=True)
    assert secret not in json.dumps({
        "password": sanitized["password"],
        "cookie": sanitized["cookie"],
        "set-cookie": sanitized["set-cookie"],
        "endpoint_url": sanitized["endpoint_url"],
        "service": sanitized["service"],
        "endpoint_urls": sanitized["endpoint_urls"],
        "failure": sanitized["failure"],
        "failure_chain": sanitized["failure_chain"],
        "detail": sanitized["detail"],
        "notes": sanitized["notes"],
        "auth_headers": sanitized["auth_headers"],
        "oauth_url": sanitized["oauth_url"],
    })
    assert content_hash(secret) not in wire
    assert content_hash(raw_url) not in wire
    assert all(fragment not in wire for fragment in (
        "value_hash", "userinfo_hash", "query_value_hashes", "credentials_hash",
    ))
    assert "deployment=blue" in wire
    for leaked in (
        "dXNlcjpTRUNSRVQ=", "PROXY_SECRET", "sid=SECRET",
        "refresh=SECRET2", "client_assertion=SECRET",
    ):
        assert leaked not in wire
    assert sanitized["question"].endswith(
        f"password: {secret} is benchmark data."
    )


def test_checkpoint_resume_retries_failure_without_double_counting(tmp_path: Path):
    manifest = _manifest()
    path = tmp_path / "run.checkpoint.json"
    first = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1", "q2", "q3"])
    first.record("q1", row={"question_id": "q1", "correct": True})
    first.record("q2", row=None, failure="timeout")
    first.close()

    resumed = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1", "q2", "q3"], resume=True,
        retry_failures=True,
    )
    assert resumed.completed_ids == ("q1",)
    assert resumed.pending_ids == ("q2", "q3")
    resumed.record("q2", row={"question_id": "q2", "correct": False})
    resumed.record("q3", row=None, failure="malformed judge output")
    with pytest.raises(BenchmarkIntegrityError, match="double-count"):
        resumed.record("q1", row={"question_id": "q1", "correct": True})

    snapshot = resumed.finalize()
    result = resumed.reconcile()
    assert snapshot["counts"] == {
        "expected": 3, "attempted": 3, "unique_attempted": 3,
        "total_attempts": 4,
        "completed": 2, "failed": 1, "missing": 0
    }
    assert snapshot["entries"]["q2"]["attempts"] == 2
    assert [event["status"] for event in snapshot["entries"]["q2"]["attempt_history"]] == [
        "failed", "completed"
    ]
    assert len(result.rows) == 3
    assert strict_accuracy(result.rows) == pytest.approx(1 / 3)


def test_checkpoint_process_lease_rejects_concurrent_owner_and_recovers(
    tmp_path: Path,
):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "leased.json"
    owner = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1"])
    script = """
import json, sys
from benchmarks.strictness import AtomicCheckpoint, BenchmarkIntegrityError
path, manifest_json, expect_busy = sys.argv[1:]
try:
    ledger = AtomicCheckpoint(
        path, manifest=json.loads(manifest_json), expected_ids=['q1'], resume=True
    )
except BenchmarkIntegrityError as exc:
    if expect_busy == 'yes' and 'owned by another live process' in str(exc):
        raise SystemExit(0)
    raise
if expect_busy == 'yes':
    raise SystemExit(3)
ledger.close()
"""
    busy = subprocess.run(
        [sys.executable, "-c", script, str(path), json.dumps(manifest), "yes"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert busy.returncode == 0, busy.stderr
    owner.close()
    recovered = subprocess.run(
        [sys.executable, "-c", script, str(path), json.dumps(manifest), "no"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert recovered.returncode == 0, recovered.stderr


def test_checkpoint_process_lease_rejects_same_process_duplicate(tmp_path: Path):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "same-process.json"
    owner = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1"])
    try:
        with pytest.raises(BenchmarkIntegrityError, match="owner in this process"):
            AtomicCheckpoint(
                path, manifest=manifest, expected_ids=["q1"], resume=True
            )
    finally:
        owner.close()


def test_checkpoint_process_lease_is_released_by_process_crash(tmp_path: Path):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "crash-released.json"
    creator = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1"])
    creator.close()
    crash_script = """
import json, os, sys
from benchmarks.strictness import AtomicCheckpoint
ledger = AtomicCheckpoint(
    sys.argv[1], manifest=json.loads(sys.argv[2]),
    expected_ids=['q1'], resume=True,
)
print('lease-acquired', flush=True)
os._exit(23)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script, str(path), json.dumps(manifest)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert crashed.returncode == 23
    assert crashed.stdout.strip() == "lease-acquired"
    # The persistent .lock file is only metadata. POSIX releases the advisory
    # lock when the crashed process exits, so recovery must be immediate.
    recovered = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1"], resume=True
    )
    recovered.close()


def test_checkpoint_rejects_every_mutation_after_lease_close(tmp_path: Path):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "closed.json"
    ledger = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1"])
    ledger.close()

    with pytest.raises(BenchmarkIntegrityError, match="lease is closed"):
        ledger.record("q1", row={"question_id": "q1", "correct": True})
    with pytest.raises(BenchmarkIntegrityError, match="lease is closed"):
        ledger.update_execution_segment("late", {"status": "running"})
    with pytest.raises(BenchmarkIntegrityError, match="lease is closed"):
        ledger.finalize()
    with pytest.raises(BenchmarkIntegrityError, match="lease is closed"):
        publish_checkpoint_artifact(ledger, tmp_path / "forbidden.json")
    assert not (tmp_path / "forbidden.json").exists()


def test_resume_does_not_retry_failures_without_explicit_opt_in(tmp_path: Path):
    manifest = _manifest(ids=("q1", "q2"))
    path = tmp_path / "run.json"
    ledger = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1", "q2"])
    ledger.record("q1", row=None, failure="timeout")
    ledger.close()
    ordinary = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1", "q2"], resume=True
    )
    assert ordinary.pending_ids == ("q2",)
    ordinary.close()
    retrying = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1", "q2"], resume=True,
        retry_failures=True,
    )
    assert retrying.pending_ids == ("q1", "q2")


def test_checkpoint_enforces_retry_policy_and_finalized_state(tmp_path: Path):
    manifest = _manifest(ids=("q1", "q2"))
    path = tmp_path / "run.json"
    ledger = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1", "q2"])
    ledger.record("q1", row=None, failure="transport failed")
    with pytest.raises(BenchmarkIntegrityError, match="requires retry_failures"):
        ledger.record("q1", row={"question_id": "q1", "correct": True})

    ledger.finalize()
    with pytest.raises(BenchmarkIntegrityError, match="finalized"):
        ledger.record("q2", row={"question_id": "q2", "correct": True})
    with pytest.raises(BenchmarkIntegrityError, match="finalized"):
        ledger.update_execution_segment("late", {"status": "running"})
    ledger.close()

    terminal = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1", "q2"], resume=True,
    )
    assert terminal.pending_ids == ()
    terminal.close()

    retrying = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1", "q2"], resume=True,
        retry_failures=True,
    )
    assert retrying.pending_ids == ("q1", "q2")
    retrying.record("q1", row={"question_id": "q1", "correct": True})
    assert json.loads(path.read_text())["entries"]["q1"]["attempts"] == 2


def test_checkpoint_rejects_invalid_status_and_segment_id_override(tmp_path: Path):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "run.json"
    ledger = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1"])
    with pytest.raises(BenchmarkIntegrityError, match="cannot override"):
        ledger.update_execution_segment(
            "actual", {"segment_id": "forged", "status": "running"}
        )
    raw = json.loads(path.read_text())
    raw["status"] = "surprising"
    ledger.close()
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="status is invalid"):
        AtomicCheckpoint(
            path, manifest=manifest, expected_ids=["q1"], resume=True
        )


def test_non_scorable_checkpoint_completes_successful_diagnostics(tmp_path: Path):
    manifest = _manifest(ids=("q1", "q2"), config={"scored_run": False})
    ledger = AtomicCheckpoint(
        tmp_path / "diag.json", manifest=manifest,
        expected_ids=["q1", "q2"], scored=False,
    )
    ledger.record("q1", row={"question_id": "q1", "correct": None,
                             "retrieval_only": True})
    assert ledger.completed_ids == ("q1",)
    assert ledger.pending_ids == ("q2",)
    rows = ledger.reconcile().rows
    assert rows[0]["correct"] is None and "benchmark_failure" not in rows[0]
    assert rows[1]["diagnostic_missing"] is True


@pytest.mark.parametrize(
    "row,match",
    [
        ({"question_id": "wrong", "correct": True}, "does not match"),
        ({"question_id": "q1", "correct": "yes"}, "malformed verdict"),
    ],
)
def test_checkpoint_rejects_mismatched_ids_and_malformed_rows(
    tmp_path: Path, row: dict, match: str
):
    ledger = AtomicCheckpoint(
        tmp_path / "run.json", manifest=_manifest(ids=("q1",)),
        expected_ids=["q1"],
    )
    with pytest.raises(BenchmarkIntegrityError, match=match):
        ledger.record("q1", row=row)


def test_checkpoint_refuses_wrong_identity_tampering_and_implicit_overwrite(tmp_path: Path):
    path = tmp_path / "run.json"
    owner = AtomicCheckpoint(
        path, manifest=_manifest(), expected_ids=["q1", "q2", "q3"]
    )
    owner.close()
    with pytest.raises(BenchmarkIntegrityError, match="already exists"):
        AtomicCheckpoint(path, manifest=_manifest(), expected_ids=["q1", "q2", "q3"])
    with pytest.raises(BenchmarkIntegrityError, match="identity mismatch"):
        AtomicCheckpoint(
            path,
            manifest=_manifest(config={"top_k": 999}),
            expected_ids=["q1", "q2", "q3"],
            resume=True,
        )

    raw = json.loads(path.read_text())
    raw["manifest"]["config"]["top_k"] = 999
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="manifest was modified"):
        AtomicCheckpoint(path, manifest=_manifest(), expected_ids=["q1", "q2", "q3"], resume=True)


def test_crash_resume_preserves_cumulative_usage_without_segment_double_count(tmp_path: Path):
    manifest = _manifest(ids=("q1", "q2"))
    path = tmp_path / "usage.json"
    first = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1", "q2"])
    first.update_execution_segment("process-a", {
        "status": "running", "reader_usage": {"calls": 0, "total_tokens": 0},
    })
    first.record(
        "q1", row={"question_id": "q1", "correct": True},
        execution_segment={
            "segment_id": "process-a", "status": "running",
            "reader_usage": {"calls": 1, "total_tokens": 11},
        },
    )
    first.close()

    resumed = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1", "q2"], resume=True
    )
    resumed.update_execution_segment("process-b", {
        "status": "running", "reader_usage": {"calls": 0, "total_tokens": 0},
    })
    resumed.record(
        "q2", row={"question_id": "q2", "correct": False},
        execution_segment={
            "segment_id": "process-b", "status": "running",
            "reader_usage": {"calls": 2, "total_tokens": 23},
        },
    )
    resumed.update_execution_segment("process-b", {
        "status": "complete", "reader_usage": {"calls": 2, "total_tokens": 23},
    })
    snapshot = resumed.finalize()
    assert [s["segment_id"] for s in snapshot["execution_segments"]] == [
        "process-a", "process-b"
    ]
    assert sum(s["reader_usage"]["calls"] for s in snapshot["execution_segments"]) == 3
    assert snapshot["execution_segments"][0]["status"] == "running"


def test_frozen_calibration_is_order_stable_disjoint_and_config_bound(tmp_path: Path):
    ids = [f"q{i}" for i in range(20)]
    dev, holdout = deterministic_split(ids, seed=11, dev_fraction=0.4)
    dev_again, holdout_again = deterministic_split(reversed(ids), seed=11, dev_fraction=0.4)
    assert set(dev) == set(dev_again)
    assert set(holdout) == set(holdout_again)
    assert set(dev).isdisjoint(holdout)

    path = tmp_path / "calibration.json"
    receipt = freeze_calibration(
        path,
        benchmark="unit",
        dataset_hash=content_hash("data"),
        ids=ids,
        config={"top_k": 3},
        models={"reader": "stub"},
        seed=11,
        dev_fraction=0.4,
    )
    loaded = load_calibration(
        path,
        benchmark="unit",
        dataset_hash=content_hash("data"),
        config={"top_k": 3},
        models={"reader": "stub"},
        ids=ids,
    )
    assert loaded == receipt
    assert select_protocol_ids(ids, split="dev", receipt=loaded) == tuple(
        item for item in ids if item in set(dev)
    )
    assert set(select_protocol_ids(ids, split="holdout", receipt=loaded)) == set(holdout)
    with pytest.raises(BenchmarkIntegrityError, match="config_hash mismatch"):
        load_calibration(
            path,
            benchmark="unit",
            dataset_hash=content_hash("data"),
            config={"top_k": 4},
            models={"reader": "stub"},
            ids=ids,
        )
    with pytest.raises(BenchmarkIntegrityError, match="overwrite"):
        freeze_calibration(
            path,
            benchmark="unit",
            dataset_hash=content_hash("data"),
            ids=ids,
            config={"top_k": 3},
            models={"reader": "stub"},
            seed=11,
        )


def test_calibration_tamper_and_unreceipted_holdout_fail_closed(tmp_path: Path):
    path = tmp_path / "calibration.json"
    ids = ["a", "b", "c", "d"]
    freeze_calibration(
        path,
        benchmark="unit",
        dataset_hash=content_hash("data"),
        ids=ids,
        config={},
        models={},
        seed=0,
    )
    raw = json.loads(path.read_text())
    raw["holdout_ids"].append(raw["dev_ids"][0])
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="hash mismatch"):
        load_calibration(
            path,
            benchmark="unit",
            dataset_hash=content_hash("data"),
            config={},
            models={},
            ids=ids,
        )
    with pytest.raises(BenchmarkIntegrityError, match="requires a frozen"):
        select_protocol_ids(ids, split="holdout", receipt=None)
    with pytest.raises(BenchmarkIntegrityError, match="requires a frozen"):
        build_manifest(
            benchmark="unit", code_sha256=content_hash("code"),
            data_sha256=content_hash("data"), config={}, models={}, seed=0,
            expected_ids=["a", "b"], protocol_split="holdout",
        )


def test_calibration_rejects_self_consistent_partition_and_payload_tampering(tmp_path: Path):
    ids = ["a", "b", "c", "d"]
    path = tmp_path / "calibration.json"
    freeze_calibration(
        path, benchmark="unit", dataset_hash=content_hash("data"), ids=ids,
        config={"top_k": 3}, models={"reader": "stub"}, seed=2,
    )
    raw = json.loads(path.read_text())
    raw["holdout_ids"].pop()
    raw["receipt_hash"] = content_hash(
        {key: value for key, value in raw.items() if key != "receipt_hash"}
    )
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="partition current ids"):
        load_calibration(
            path, benchmark="unit", dataset_hash=content_hash("data"),
            config={"top_k": 3}, models={"reader": "stub"}, ids=ids,
        )

    path.unlink()
    freeze_calibration(
        path, benchmark="unit", dataset_hash=content_hash("data"), ids=ids,
        config={"top_k": 3}, models={"reader": "stub"}, seed=2,
    )
    raw = json.loads(path.read_text())
    raw["config"]["top_k"] = 999
    raw["receipt_hash"] = content_hash(
        {key: value for key, value in raw.items() if key != "receipt_hash"}
    )
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="stored config hash"):
        load_calibration(
            path, benchmark="unit", dataset_hash=content_hash("data"),
            config={"top_k": 3}, models={"reader": "stub"}, ids=ids,
        )


def test_immutable_publish_failure_never_leaves_partial_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    final = tmp_path / "artifact.json"

    def fail_link(_source, _target):
        raise OSError("simulated publish crash")

    monkeypatch.setattr(strictness.os, "link", fail_link)
    with pytest.raises(OSError, match="publish crash"):
        write_immutable_artifact(final, {"large": "payload" * 100})
    assert not final.exists()


def test_usage_snapshot_rejects_negative_nan_and_infinite_claims():
    class BadUsage:
        call_count = -1
        prompt_tokens = float("nan")
        completion_tokens = float("inf")
        total_tokens = -2
        total_latency_s = -0.1
        cost_usd = float("-inf")

    snapshot = usage_snapshot(BadUsage())
    assert snapshot == {
        "calls": None,
        "calls_available": False,
        "request_attempts": None,
        "request_attempts_available": False,
        "successful_responses": None,
        "successful_responses_available": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "latency_s": None,
        "cost_usd": None,
        "token_usage_available": False,
        "latency_available": False,
        "cost_available": False,
    }


def test_usage_snapshot_preserves_explicit_structural_zero():
    class NoCalls:
        call_count = 0
        request_attempts = 0
        successful_responses = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_latency_s = 0.0
        token_usage_available = True

    snapshot = usage_snapshot(NoCalls())
    assert snapshot["calls"] == 0
    assert snapshot["calls_available"] is True
    assert snapshot["request_attempts"] == 0
    assert snapshot["request_attempts_available"] is True
    assert snapshot["total_tokens"] == 0
    assert snapshot["token_usage_available"] is True


def test_immutable_artifact_and_deterministic_cli_smoke(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    write_immutable_artifact(artifact, {"ok": True})
    with pytest.raises(BenchmarkIntegrityError, match="overwrite"):
        write_immutable_artifact(artifact, {"ok": False})

    first = deterministic_smoke(tmp_path / "one")
    second = deterministic_smoke(tmp_path / "two")
    assert first == second == {
        "run_id": first["run_id"],
        "counts": {"expected": 3, "attempted": 3, "unique_attempted": 3,
                   "total_attempts": 3,
                   "completed": 2, "failed": 1, "missing": 0},
        "accuracy": pytest.approx(1 / 3),
        "failure_ids": ["smoke-3"],
    }

    proc = subprocess.run(
        [sys.executable, "benchmarks/strictness.py", "--smoke"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["counts"]["expected"] == 3
    assert payload["failure_ids"] == ["smoke-3"]


def test_latest_pointer_dereference_is_explicit_and_identity_checked(tmp_path: Path):
    manifest = _manifest(ids=("q1",))
    archive = tmp_path / "longmemeval-v2-hymem-20260904T120000Z-seed0.json"
    write_immutable_artifact(archive, {"manifest": manifest, "scores": {}})
    pointer = tmp_path / "longmemeval-v2-hymem.json"
    strictness.write_latest_pointer(
        pointer, archive=archive, run_id=manifest["run_id"]
    )
    assert read_artifact_or_pointer(pointer)["manifest"] == manifest

    raw = json.loads(pointer.read_text())
    raw["run_id"] = content_hash("wrong")
    pointer.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="identity mismatch"):
        read_artifact_or_pointer(pointer)

    # Matching a pointer to a forged stored run_id is insufficient: the target
    # manifest itself must still hash to that identity.
    raw_archive = json.loads(archive.read_text())
    raw_archive["manifest"]["config"]["top_k"] = 999
    archive.write_text(json.dumps(raw_archive), encoding="utf-8")
    raw["run_id"] = manifest["run_id"]
    pointer.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="manifest identity is invalid"):
        read_artifact_or_pointer(pointer)


def test_rows_publish_before_report_failure_and_recover_without_model_calls(
    tmp_path: Path,
):
    manifest = _manifest(ids=("q1", "q2"))
    checkpoint = tmp_path / "run.checkpoint.json"
    ledger = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=["q1", "q2"]
    )
    ledger.record("q1", row={"question_id": "q1", "correct": True})
    ledger.record("q2", row=None, failure="worker transport failure")
    archive = tmp_path / "archive.json"
    publish_checkpoint_artifact(
        ledger, archive, payload={"benchmark": "unit", "scores": {}}
    )

    def broken_report():
        raise RuntimeError("presentation bug")

    with pytest.raises(RuntimeError, match="presentation bug"):
        broken_report()
    assert archive.exists()
    assert json.loads(archive.read_text())["execution"]["counts"]["expected"] == 2
    ledger.close()

    recovered = tmp_path / "recovered.json"
    output = export_checkpoint_without_recompute(checkpoint, recovered)
    assert output["per_question"][0]["correct"] is True
    assert output["per_question"][1]["correct"] is False
    assert output["recovery_disclosure"].endswith("derived adapter diagnostics omitted.")
