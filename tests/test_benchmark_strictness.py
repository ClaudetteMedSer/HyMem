from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import benchmarks.strictness as strictness

from benchmarks.strictness import (
    AtomicCheckpoint,
    BenchmarkIdentityData,
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


def _current_indexing_status(**overrides):
    value = {
        "dream_status_schema": strictness.DREAM_STATUS_SCHEMA_VERSION,
        **{key: 0 for key in strictness.DURABLE_PENDING_FIELDS},
        **{key: 0 for key in strictness.DURABLE_MALFORMED_FIELDS},
        **{key: 0 for key in strictness._DURABLE_QUARANTINE_FIELDS},
        "terminal_loss_chunks": 0,
        "coverage_integrity_failures": 0,
        "phase1_backlog_status": "current_producer",
        "pending_chunks_authoritative": True,
        "phase1_generation_key": "hymem-phase1-generation-v1:" + "1" * 64,
        "in_progress": False,
    }
    value.update(overrides)
    return value


def _current_indexing_report(**overrides):
    value = {
        **{
            key: 0
            for key in strictness._CURRENT_DREAM_REPORT_FAILURE_FIELDS
        },
        **{
            key: False
            for key in strictness._CURRENT_DREAM_REPORT_BOOLEAN_FIELDS
        },
    }
    value.update(overrides)
    return value


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


def test_code_hash_excludes_incidental_prose_but_hashes_explicit_data(tmp_path: Path):
    package = tmp_path / "surface"
    package.mkdir()
    implementation = package / "implementation.py"
    prompt = package / "instructions.md"
    implementation.write_text("VALUE = 1", encoding="utf-8")
    prompt.write_text("first prompt", encoding="utf-8")
    config = package / "defaults.yaml"
    config.write_text("top_k: 3", encoding="utf-8")
    before = code_hash([package], root=tmp_path)
    prompt.write_text("changed prompt", encoding="utf-8")
    config.write_text("top_k: 4", encoding="utf-8")
    assert code_hash([package], root=tmp_path) == before

    # Parsed non-code assets remain material when explicitly inventoried.
    explicit_inputs = [
        package,
        BenchmarkIdentityData(prompt, "prompt"),
        BenchmarkIdentityData(config, "config"),
    ]
    explicit_before = code_hash(explicit_inputs, root=tmp_path)
    prompt.write_text("changed again", encoding="utf-8")
    assert code_hash(explicit_inputs, root=tmp_path) != explicit_before
    with pytest.raises(BenchmarkIntegrityError, match="explicit data classification"):
        code_hash([prompt], root=tmp_path)


def test_code_identity_schema_salt_forces_a_new_identity(monkeypatch, tmp_path: Path):
    implementation = tmp_path / "implementation.py"
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    before = code_hash([implementation], root=tmp_path)
    monkeypatch.setattr(strictness, "CODE_IDENTITY_VERSION", "test-code-v-next")
    assert code_hash([implementation], root=tmp_path) != before


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


def test_multihop_miner_does_not_stop_on_nonexhausted_report_with_pending_work():
    from benchmarks.multihop_miner import _ingest_and_dream

    class FakeHyMem:
        def __init__(self):
            self.cycles = 0

        def log_messages(self, _session_id, _entries):
            raise AssertionError("empty fixture must not ingest messages")

        def dream(self):
            self.cycles += 1
            return {"budget_exhausted": False, "skipped_locked": False}

        def dream_status(self):
            return {
                "pending_chunks": int(self.cycles < 2),
                "quarantined_chunks": 0,
                "terminal_loss_chunks": 0,
            }

    hy = FakeHyMem()
    cycles = _ingest_and_dream(
        hy,
        {"haystack_sessions": []},
        lambda value: value,
        max_cycles=3,
        timeout_s=10,
    )

    assert cycles == 2
    assert hy.cycles == 2


def test_indexing_never_converges_or_quarantines_fails_loudly():
    with pytest.raises(strictness.IndexingConvergenceError) as missing_pending:
        converge_indexing(
            lambda: {"budget_exhausted": False},
            status=lambda: {"quarantined_chunks": 0},
            max_cycles=2, timeout_s=10,
        )
    assert missing_pending.value.summary["failure_reason"] == (
        "malformed_status_shape"
    )

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

    with pytest.raises(strictness.IndexingConvergenceError) as terminal_loss:
        converge_indexing(
            lambda: {"budget_exhausted": False},
            status=lambda: {
                "pending_chunks": 0,
                "quarantined_chunks": 0,
                "terminal_loss_chunks": 1,
            },
            max_cycles=2,
            timeout_s=10,
            require_healthy=True,
        )
    assert terminal_loss.value.summary["complete"] is True
    assert terminal_loss.value.summary["healthy"] is False
    assert terminal_loss.value.summary["failure_reason"] == (
        "terminal_extraction_source_loss"
    )

    coverage_cycles = 0

    def corrupt_coverage_dream():
        nonlocal coverage_cycles
        coverage_cycles += 1
        return {"budget_exhausted": False}

    with pytest.raises(strictness.IndexingConvergenceError) as coverage_failure:
        converge_indexing(
            corrupt_coverage_dream,
            status=lambda: {
                "pending_chunks": 3,
                "coverage_integrity_failures": 1,
            },
            max_cycles=5,
            timeout_s=10,
            require_healthy=True,
        )
    assert coverage_cycles == 5
    assert coverage_failure.value.summary["complete"] is False
    assert coverage_failure.value.summary["healthy"] is False
    assert coverage_failure.value.summary["failure_reason"] == (
        "coverage_integrity_failure"
    )

    diagnostic = converge_indexing(
        lambda: {"budget_exhausted": False},
        status=lambda: {
            "pending_chunks": 0,
            "coverage_integrity_failures": 1,
        },
        max_cycles=2,
        timeout_s=10,
        require_healthy=False,
    )
    assert diagnostic["complete"] is True
    assert diagnostic["healthy"] is False

    with pytest.raises(strictness.IndexingConvergenceError) as vector_backlog:
        converge_indexing(
            lambda: {"budget_exhausted": False},
            status=lambda: {
                "pending_chunks": 0, "pending_message_embeddings": 1,
            },
            max_cycles=2, timeout_s=10,
        )
    assert vector_backlog.value.summary["failure_reason"] == (
        "malformed_status_shape"
    )


@pytest.mark.parametrize(
    "field_name", strictness._CURRENT_DREAM_REPORT_FAILURE_FIELDS,
)
def test_current_cycle_failure_retries_until_a_clean_cycle(field_name):
    cycle = 0

    def dream():
        nonlocal cycle
        cycle += 1
        return _current_indexing_report(**{field_name: int(cycle == 1)})

    summary = converge_indexing(
        dream,
        status=lambda: _current_indexing_status(),
        max_cycles=2,
        timeout_s=10,
    )
    assert summary["cycles"] == 2
    assert summary["complete"] is True


@pytest.mark.parametrize(
    "field_name", strictness._CURRENT_DREAM_REPORT_FAILURE_FIELDS,
)
def test_permanent_current_cycle_failure_exhausts_the_bound(field_name):
    with pytest.raises(strictness.IndexingConvergenceError) as caught:
        converge_indexing(
            lambda: _current_indexing_report(**{field_name: 1}),
            status=lambda: _current_indexing_status(),
            max_cycles=2,
            timeout_s=10,
        )
    assert caught.value.summary["cycles"] == 2
    assert caught.value.summary["failure_reason"] == "max_cycles_exhausted"


def test_current_status_schema_pairing_and_report_fields_fail_closed():
    report = _current_indexing_report()
    malformed_statuses = [
        {
            **_current_indexing_status(),
            "dream_status_schema": None,
            "benchmark_indexing_status_schema": (
                strictness.BENCHMARK_INDEXING_STATUS_VERSION
            ),
        },
        {
            **_current_indexing_status(),
            "pending_message_embeddings": 1,
        },
    ]
    for status in malformed_statuses:
        with pytest.raises(strictness.IndexingConvergenceError) as caught:
            converge_indexing(
                lambda: report, status=lambda status=status: status,
                max_cycles=1, timeout_s=10,
            )
        assert caught.value.summary["failure_reason"] == "malformed_status_shape"

    incomplete_report = dict(report)
    incomplete_report.pop("fact_failures")
    with pytest.raises(strictness.IndexingConvergenceError) as caught:
        converge_indexing(
            lambda: incomplete_report,
            status=lambda: _current_indexing_status(),
            max_cycles=1,
            timeout_s=10,
        )
    assert caught.value.summary["failure_reason"] == (
        "malformed_cycle_failure_report"
    )


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"pending_chunks_authoritative": "yes"}, "malformed_status_shape"),
        ({"phase1_backlog_status": "unknown"}, "malformed_status_shape"),
        ({"phase1_generation_key": ""}, "malformed_status_shape"),
        (
            {
                "pending_chunks_authoritative": False,
                "phase1_backlog_status": "producer_unavailable",
                "phase1_generation_key": None,
            },
            "phase1_producer_unavailable",
        ),
    ],
)
def test_current_status_requires_exact_phase1_producer_authority(
    overrides, expected_reason,
):
    with pytest.raises(strictness.IndexingConvergenceError) as caught:
        converge_indexing(
            lambda: _current_indexing_report(),
            status=lambda: _current_indexing_status(**overrides),
            max_cycles=1,
            timeout_s=10,
        )
    assert caught.value.summary["failure_reason"] == expected_reason


def test_embedding_status_requires_coherent_snapshot_extension():
    class LegacyMemory:
        config = object()

        @staticmethod
        def dream_status():
            return _current_indexing_status()

    class Embeddings:
        model = "unit"
        dim = 3

    with pytest.raises(
        BenchmarkIntegrityError, match="indexing status is unavailable"
    ):
        strictness.durable_indexing_status(LegacyMemory(), Embeddings())


def test_current_in_progress_and_malformed_authority_block_completion():
    statuses = iter((
        _current_indexing_status(in_progress=True),
        _current_indexing_status(),
    ))
    summary = converge_indexing(
        lambda: _current_indexing_report(),
        status=lambda: next(statuses),
        max_cycles=2,
        timeout_s=10,
    )
    assert summary["cycles"] == 2

    with pytest.raises(strictness.IndexingConvergenceError) as caught:
        converge_indexing(
            lambda: _current_indexing_report(),
            status=lambda: _current_indexing_status(malformed_facts=1),
            max_cycles=2,
            timeout_s=10,
        )
    assert caught.value.summary["complete"] is True
    assert caught.value.summary["failure_reason"] == "malformed_durable_state"


def test_indexing_cycle_exception_summary_never_persists_exception_text():
    secret = "Bearer sk-private-cycle-token"
    private_path = "/home/node/private/index.sqlite"

    def fail_cycle():
        raise RuntimeError(
            f"provider failed at {private_path}?api_key={secret}: " + "x" * 20_000
        )

    with pytest.raises(strictness.IndexingConvergenceError) as caught:
        converge_indexing(
            fail_cycle,
            status=lambda: {"pending_chunks": 1},
            max_cycles=2,
            timeout_s=10,
        )

    summary = caught.value.summary
    assert summary["failure_reason"] == "cycle_exception:RuntimeError"
    encoded = json.dumps(sanitize_for_artifact(summary), sort_keys=True)
    assert secret not in encoded
    assert private_path not in encoded
    assert "x" * 1_000 not in encoded


def test_manifest_and_checkpoint_never_serialize_recursive_secrets(tmp_path: Path):
    secret = "super-secret-bearer-value"
    config = {
        "label_free_answer_path": True,
        "summary": "/home/node/private/config.yaml",
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
    assert "/home/node/private" not in serialized
    assert "example.test/v1" in serialized
    assert manifest["config"]["extra_body"]["temperature"] == 0
    assert manifest["config"]["summary"] == {"path_redacted": True}
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
    assert "deployment=blue" not in wire
    opaque = sanitize_for_artifact({
        "endpoint_url": "https://example.test/v1?opaque=arbitrary-secret",
    })
    assert "arbitrary-secret" not in json.dumps(opaque)
    malformed = sanitize_for_artifact({
        "endpoint_url": "https://operator:malformed-secret@[broken/v1",
    })
    assert "malformed-secret" not in json.dumps(malformed)
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


def test_checkpoint_failed_atomic_record_rolls_back_row_but_keeps_orphan_attempt(
    monkeypatch, tmp_path: Path,
):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "transactional-record.json"
    ledger = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1"])
    ledger.update_execution_segment("first", {
        "segment_id": "first", "status": "running", "attempted_attempts": 0,
    })

    original_atomic_json = strictness._atomic_json
    fault = OSError("one-shot atomic replace fault")
    failed = False

    def fail_once(target, value):
        nonlocal failed
        if not failed:
            failed = True
            raise fault
        return original_atomic_json(target, value)

    monkeypatch.setattr(strictness, "_atomic_json", fail_once)
    with pytest.raises(OSError) as caught:
        ledger.record(
            "q1", row={"question_id": "q1", "correct": True},
            execution_segment={
                "segment_id": "first", "status": "running",
                "attempted_attempts": 1,
            },
        )
    assert caught.value is fault
    assert ledger.pending_ids == ("q1",)
    assert ledger.reconcile().attempted == 0
    ledger.update_execution_segment("first", {
        "segment_id": "first", "status": "complete", "attempted_attempts": 1,
    })
    ledger.close()
    assert json.loads(path.read_text())["entries"] == {}

    resumed = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1"], resume=True,
    )
    resumed.record(
        "q1", row={"question_id": "q1", "correct": True},
        execution_segment={
            "segment_id": "second", "status": "complete",
            "attempted_attempts": 1,
        },
    )
    snapshot = resumed.finalize()
    assert snapshot["counts"]["total_attempts"] == 2
    assert [segment["attempted_attempts"] for segment in
            snapshot["execution_segments"]] == [1, 1]


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
    retrying.close()


@pytest.mark.parametrize(
    "resume", ["false", "true", 0, 1, None, [], {}, ()],
    ids=[
        "false-string", "true-string", "zero", "one", "none", "list",
        "dict", "tuple",
    ],
)
def test_checkpoint_rejects_non_boolean_resume_before_path_or_lease_side_effects(
    tmp_path: Path, monkeypatch, resume: object,
):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "not-created" / "checkpoint.json"

    def forbidden_lease(_path):
        raise AssertionError("invalid resume policy acquired a checkpoint lease")

    monkeypatch.setattr(strictness, "_CheckpointLease", forbidden_lease)
    with pytest.raises(
        BenchmarkIntegrityError,
        match="checkpoint resume policy must be a boolean",
    ):
        AtomicCheckpoint(
            path,
            manifest=manifest,
            expected_ids=("q1",),
            resume=resume,  # type: ignore[arg-type]
        )

    assert not path.parent.exists()
    assert not path.exists()
    assert not path.with_name(path.name + ".lock").exists()


@pytest.mark.parametrize(
    "resume", ["false", "true", 0, 1, None, [], {}, ()],
    ids=[
        "false-string", "true-string", "zero", "one", "none", "list",
        "dict", "tuple",
    ],
)
def test_checkpoint_rejects_non_boolean_resume_without_rewriting_finalized_state(
    tmp_path: Path, monkeypatch, resume: object,
):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "terminal.json"
    ledger = AtomicCheckpoint(path, manifest=manifest, expected_ids=("q1",))
    ledger.record("q1", row=None, failure="timeout")
    finalized = ledger.finalize()
    ledger.close()
    original_bytes = path.read_bytes()
    lock_path = path.with_name(path.name + ".lock")
    original_lock_bytes = lock_path.read_bytes()

    def forbidden_lease(_path):
        raise AssertionError("invalid resume policy acquired a checkpoint lease")

    monkeypatch.setattr(strictness, "_CheckpointLease", forbidden_lease)
    with pytest.raises(
        BenchmarkIntegrityError,
        match="checkpoint resume policy must be a boolean",
    ):
        AtomicCheckpoint(
            path,
            manifest=manifest,
            expected_ids=("q1",),
            resume=resume,  # type: ignore[arg-type]
            retry_failures=True,
        )

    assert path.read_bytes() == original_bytes
    assert lock_path.read_bytes() == original_lock_bytes
    terminal = json.loads(original_bytes)
    assert terminal["status"] == "complete"
    assert terminal["counts"] == finalized["counts"]
    assert terminal["failure_ids"] == finalized["failure_ids"]


def test_checkpoint_accepts_only_boolean_resume_create_and_resume_controls(
    tmp_path: Path,
):
    manifest = _manifest(ids=("q1",))
    path = tmp_path / "controls.json"

    creator = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=("q1",), resume=False,
    )
    creator.close()
    original_bytes = path.read_bytes()

    with pytest.raises(BenchmarkIntegrityError, match="already exists"):
        AtomicCheckpoint(
            path, manifest=manifest, expected_ids=("q1",), resume=False,
        )
    assert path.read_bytes() == original_bytes

    resumed = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=("q1",), resume=True,
    )
    assert resumed.pending_ids == ("q1",)
    resumed.close()


@pytest.mark.parametrize("retry_failures", ["false", "true", 0, 1, None])
def test_checkpoint_rejects_non_boolean_retry_policy_without_touching_terminal_state(
    tmp_path: Path, monkeypatch, retry_failures: object,
):
    manifest = _manifest(ids=("q1", "q2"))
    path = tmp_path / "terminal.json"
    ledger = AtomicCheckpoint(path, manifest=manifest, expected_ids=["q1", "q2"])
    ledger.record("q1", row=None, failure="timeout")
    finalized = ledger.finalize()
    ledger.close()
    original_bytes = path.read_bytes()

    def forbidden_lease(_path):
        raise AssertionError("invalid retry policy acquired a checkpoint lease")

    monkeypatch.setattr(strictness, "_CheckpointLease", forbidden_lease)
    with pytest.raises(
        BenchmarkIntegrityError,
        match="checkpoint retry_failures policy must be a boolean",
    ):
        AtomicCheckpoint(
            path,
            manifest=manifest,
            expected_ids=["q1", "q2"],
            resume=True,
            retry_failures=retry_failures,
        )

    assert path.read_bytes() == original_bytes
    terminal = json.loads(original_bytes)
    assert terminal["status"] == "complete"
    assert terminal["counts"] == finalized["counts"]
    assert terminal["failure_ids"] == finalized["failure_ids"]


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
    assert terminal.retry_failures is False
    assert terminal.pending_ids == ()
    terminal.close()

    retrying = AtomicCheckpoint(
        path, manifest=manifest, expected_ids=["q1", "q2"], resume=True,
        retry_failures=True,
    )
    assert retrying.retry_failures is True
    assert retrying.pending_ids == ("q1", "q2")
    reopened = json.loads(path.read_text())
    assert reopened["status"] == "running"
    assert "counts" not in reopened
    assert "failure_ids" not in reopened
    retrying.record("q1", row={"question_id": "q1", "correct": True})
    assert json.loads(path.read_text())["entries"]["q1"]["attempts"] == 2
    retrying.close()


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


def test_checkpoint_recovery_does_not_publish_when_lease_close_fails(
    tmp_path: Path, monkeypatch,
):
    manifest = _manifest(ids=("q1",))
    checkpoint = tmp_path / "source.checkpoint.json"
    source = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=["q1"]
    )
    source.record("q1", row={"question_id": "q1", "correct": True})
    source.finalize()
    source.close()

    close_calls = []

    class CloseFailLedger:
        def __init__(self, *args, **kwargs):
            self.inner = AtomicCheckpoint(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def close(self):
            close_calls.append(True)
            self.inner.close()
            raise RuntimeError("secret lease close detail")

    monkeypatch.setattr(strictness, "AtomicCheckpoint", CloseFailLedger)
    recovered = tmp_path / "must-not-publish.json"
    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as caught:
        export_checkpoint_without_recompute(checkpoint, recovered)

    assert close_calls == [True]
    assert not recovered.exists()
    assert "secret lease close detail" not in str(caught.value)


def test_safe_identity_strings_and_closed_failure_reasons_survive_sanitizing():
    long_safe_config = "stable-config-text-" * 300
    identity = {
        "model": "deepseek-v4-flash",
        "protocol": "strict-v1",
        "scale": "100K",
        "question_id": "beam:100K:conversation:ordinal:0:q1",
        "digest": "sha256:" + "a" * 64,
        "embedding_model": (
            "openai-compatible:https://embed.example/v1::embed-v1"
        ),
        "tokenizer_failure_policy": "not-applicable",
    }
    assert sanitize_for_artifact(identity) == identity
    manifest = _manifest(config={"custom_prompt": long_safe_config})
    assert manifest["config"]["custom_prompt"] == long_safe_config
    for reason in (
        "materialization_failure",
        "source_stream_invalid",
        "missing_store_build_receipt",
        "store_build_identity_mismatch",
        "clean_empty",
    ):
        assert sanitize_for_artifact({"failure_reason": reason}) == {
            "failure_reason": reason
        }
    for diagnostic in (
        "memory_pipeline_usage:Unavailable",
        "embedding_usage:Unavailable",
        "probe_failure:RuntimeError",
    ):
        assert sanitize_for_artifact({"probe_error": diagnostic}) == {
            "probe_error": diagnostic,
        }


def test_checkpoint_and_artifact_bound_all_operational_diagnostics(
    tmp_path: Path,
):
    secret = "PRIVATE_BEARER_SENTINEL_742"
    absolute = f"/home/node/private/{secret}/key.json"
    file_uri = f"file:///home/node/private/{secret}/state.sqlite"
    embedded_file_uri = f"database failed at {file_uri}"
    windows_forward = f"C:/Users/private/{secret}/state.sqlite"
    slash_unc = f"//server/share/private/{secret}/state.sqlite"
    smb_uri = f"smb://server/share/private/{secret}/state.sqlite"
    credential_url = (
        f"https://operator:{secret}@provider.example/v1?token={secret}"
    )
    huge = (f"Bearer {secret} at {absolute} via {credential_url} " * 200)
    checkpoint = tmp_path / f"{secret}.checkpoint.json"
    ledger = AtomicCheckpoint(
        checkpoint, manifest=_manifest(ids=("q1", "q2")),
        expected_ids=("q1", "q2"),
    )
    ledger.update_execution_segment("segment-safe", {
        "status": "running",
        "model": "deepseek-v4-flash",
        "protocol": "strict-v1",
        "scale": "100K",
        "debug_path": absolute,
        "database_uri": file_uri,
        "database_detail": embedded_file_uri,
        "single_component_path": "/private",
        "windows_path": r"C:\private\state.sqlite",
        "windows_forward_path": windows_forward,
        "unc_path": r"\\server\private\state.sqlite",
        "slash_unc_path": f"failed at {slash_unc}",
        "smb_uri": smb_uri,
        "embedded_smb_uri": f"failed at {smb_uri}",
        "local_identity": "local://feature-hash",
        "provider_url": credential_url,
        "provider_exception": huge,
        "probe_error": f"probe_failure:RuntimeError:{huge}",
        # Segment telemetry is operational, so even a normally evidence-like
        # key cannot smuggle a host path around checkpoint sanitizing.
        "summary": huge,
        "instrumentation_errors": [f"reader_usage:RuntimeError:{huge}"],
        "extraction_canary": {
            "status": "failed",
            "failure_reason": "parse_failure",
            "failure_details": ["leaf[0]:parse_failure"],
        },
    })
    ledger.record("q1", row={
        "question_id": "q1",
        "correct": False,
        "question": "verbatim benchmark evidence remains exact",
        "benchmark_failure": f"execution_failure:RuntimeError:{huge}",
    })
    ledger.record("q2", row=None, failure=huge)
    archive = tmp_path / "artifact.json"
    artifact = publish_checkpoint_artifact(ledger, archive)
    direct_archive = tmp_path / "direct-artifact.json"
    write_immutable_artifact(direct_archive, {
        "execution": {
            "segments": [{
                "segment_id": "direct-segment",
                "summary": huge,
                "debug_path": absolute,
            }],
        },
    })

    checkpoint_wire = checkpoint.read_text(encoding="utf-8")
    archive_wire = archive.read_text(encoding="utf-8")
    direct_archive_wire = direct_archive.read_text(encoding="utf-8")
    for wire in (checkpoint_wire, archive_wire, direct_archive_wire):
        assert secret not in wire
        assert "/home/node/private" not in wire
        assert "file:///" not in wire
        assert windows_forward not in wire
        assert slash_unc not in wire
        assert smb_uri not in wire
        assert "Bearer" not in wire
        assert len(wire) < len(huge)
    assert json.loads(direct_archive_wire)["execution"]["segments"][0][
        "summary"
    ] == "<redacted-oversized-text>"
    stored = json.loads(checkpoint_wire)
    stored_segment = stored["execution_segments"][0]
    assert stored_segment["database_uri"] == {"path_redacted": True}
    assert "file:" not in stored_segment["database_detail"]
    assert stored_segment["single_component_path"] == {"path_redacted": True}
    assert stored_segment["windows_path"] == "<redacted-path>"
    assert stored_segment["windows_forward_path"] == "<redacted-path>"
    assert stored_segment["unc_path"] == "<redacted-path>"
    assert stored_segment["slash_unc_path"] == "failed at <redacted-path>"
    assert stored_segment["smb_uri"] == {"path_redacted": True}
    assert stored_segment["embedded_smb_uri"] == (
        "failed at <redacted-path-uri>"
    )
    assert stored_segment["local_identity"] == "local://feature-hash"
    assert stored_segment["model"] == "deepseek-v4-flash"
    assert stored_segment["protocol"] == "strict-v1"
    assert stored_segment["scale"] == "100K"
    assert stored_segment["summary"] == (
        "<redacted-oversized-text>"
    )
    assert stored_segment["instrumentation_errors"] == [
        "reader_usage:RuntimeError"
    ]
    assert stored_segment["probe_error"] == (
        "probe_failure:RuntimeError"
    )
    assert stored_segment["extraction_canary"] == {
        "status": "failed",
        "failure_reason": "parse_failure",
        "failure_details": ["leaf[0]:parse_failure"],
    }
    assert stored["entries"]["q1"]["failure"] == (
        "execution_failure:RuntimeError"
    )
    assert stored["entries"]["q1"]["attempt_history"][0]["failure"] == (
        "execution_failure:RuntimeError"
    )
    assert stored["entries"]["q2"]["failure"] == "unspecified_failure"
    assert artifact["per_question"][0]["question"] == (
        "verbatim benchmark evidence remains exact"
    )
    checkpoint_identity = artifact["execution"]["checkpoint"]
    assert set(checkpoint_identity) == {"schema", "state", "state_sha256"}
    assert checkpoint_identity["state_sha256"].startswith("sha256:")
    assert checkpoint.name not in archive_wire
    ledger.close()


def test_resume_rewrites_legacy_failure_history_and_segments_without_leak(
    tmp_path: Path,
):
    secret = "LEGACY_PRIVATE_SENTINEL_913"
    absolute = f"/home/node/private/{secret}/checkpoint.db"
    checkpoint = tmp_path / "legacy.json"
    manifest = _manifest(ids=("q1",))
    ledger = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=("q1",)
    )
    ledger.record("q1", row=None, failure="timeout")
    ledger.close()

    legacy = json.loads(checkpoint.read_text(encoding="utf-8"))
    raw = f"execution_failure: RuntimeError: Bearer {secret} at {absolute}"
    entry = legacy["entries"]["q1"]
    entry["failure"] = raw
    entry["row"]["benchmark_failure"] = raw
    entry["debug_payload"] = raw
    last_event = entry["attempt_history"][0]
    last_event["attempt"] = 2
    last_event["failure"] = raw
    last_event["row"]["benchmark_failure"] = raw
    last_event["debug_payload"] = raw
    entry["attempts"] = 2
    entry["attempt_history"] = [
        f"legacy event: {raw} at smb://server/share/{secret}/state.sqlite",
        last_event,
    ]
    legacy["execution_segments"] = [{
        "segment_id": "legacy-segment",
        "status": "running",
        "provider_exception": raw,
        "private_path": absolute,
        "windows_forward_path": f"C:/Users/private/{secret}/state.sqlite",
        "slash_unc_path": f"failed at //server/share/{secret}/state.sqlite",
        "embedded_smb_uri": f"failed at smb://server/share/{secret}/state.sqlite",
        "local_identity": "local://feature-hash",
    }]
    checkpoint.write_text(json.dumps(legacy), encoding="utf-8")

    recovered = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=("q1",), resume=True
    )
    recovered.close()
    wire = checkpoint.read_text(encoding="utf-8")
    assert secret not in wire
    assert absolute not in wire
    state = json.loads(wire)
    entry = state["entries"]["q1"]
    assert entry["failure"] == "execution_failure:RuntimeError"
    assert entry["row"]["benchmark_failure"] == (
        "execution_failure:RuntimeError"
    )
    assert "debug_payload" not in entry
    assert entry["attempt_history"][0] == {
        "attempt": 1,
        "status": "failed",
        "failure": "unspecified_failure",
        "row": {
            "question_id": "q1",
            "correct": False,
            "benchmark_failure": "unspecified_failure",
        },
    }
    assert entry["attempt_history"][1]["failure"] == (
        "execution_failure:RuntimeError"
    )
    assert set(entry["attempt_history"][1]) == {
        "attempt", "status", "failure", "row",
    }
    segment = state["execution_segments"][0]
    assert segment["provider_exception"] == (
        "execution_failure:RuntimeError"
    )
    assert segment["windows_forward_path"] == "<redacted-path>"
    assert segment["slash_unc_path"] == "failed at <redacted-path>"
    assert segment["embedded_smb_uri"] == "failed at <redacted-path-uri>"
    assert segment["local_identity"] == "local://feature-hash"


def test_resume_projects_running_checkpoint_root_and_is_idempotent(
    tmp_path: Path,
):
    secret = "RUNNING_ROOT_SECRET_641"
    private_path = f"/home/operator/{secret}/checkpoint.sqlite"
    manifest = _manifest(ids=("q1", "q2"))
    checkpoint = tmp_path / "running-root.json"
    ledger = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=("q1", "q2")
    )
    ledger.record("q1", row={"question_id": "q1", "correct": True})
    ledger.close()

    legacy = json.loads(checkpoint.read_text(encoding="utf-8"))
    legacy.update({
        "checkpoint_path": private_path,
        "provider_credentials": {
            "authorization": f"Bearer {secret}",
            "nested": [{"database_path": private_path}],
        },
        # Terminal fields are not part of the running schema, even if a
        # legacy writer happened to leave them behind.
        "counts": {"private_debug": secret},
        "failure_ids": [secret],
    })
    checkpoint.write_text(json.dumps(legacy), encoding="utf-8")

    resumed = AtomicCheckpoint(
        checkpoint,
        manifest=manifest,
        expected_ids=("q1", "q2"),
        resume=True,
    )
    assert resumed.pending_ids == ("q2",)
    resumed.close()
    first_wire = checkpoint.read_bytes()
    state = json.loads(first_wire)
    assert set(state) == {
        "schema", "run_id", "manifest", "expected_ids", "scored",
        "verdict_key", "entries", "execution_segments", "status",
    }
    assert state["status"] == "running"
    assert secret not in first_wire.decode("utf-8")
    assert private_path not in first_wire.decode("utf-8")
    assert AtomicCheckpoint._sanitize_runtime_state(state) == state

    resumed_again = AtomicCheckpoint(
        checkpoint,
        manifest=manifest,
        expected_ids=("q1", "q2"),
        resume=True,
    )
    resumed_again.close()
    assert checkpoint.read_bytes() == first_wire


def test_resume_projects_complete_root_and_publishes_valid_archive(
    tmp_path: Path,
):
    secret = "COMPLETE_ROOT_SECRET_852"
    manifest = _manifest(ids=("q1", "q2"))
    checkpoint = tmp_path / "complete-root.json"
    ledger = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=("q1", "q2")
    )
    ledger.record("q1", row={"question_id": "q1", "correct": True})
    ledger.record("q2", row=None, failure="timeout")
    canonical = ledger.finalize()
    ledger.close()

    legacy = json.loads(checkpoint.read_text(encoding="utf-8"))
    legacy["legacy_runtime"] = {
        "database_path": f"/private/{secret}/state.sqlite",
        "credentials": {"api_key": secret},
    }
    checkpoint.write_text(json.dumps(legacy), encoding="utf-8")

    resumed = AtomicCheckpoint(
        checkpoint,
        manifest=manifest,
        expected_ids=("q1", "q2"),
        resume=True,
    )
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert set(state) == {
        "schema", "run_id", "manifest", "expected_ids", "scored",
        "verdict_key", "entries", "execution_segments", "status",
        "counts", "failure_ids",
    }
    assert state["counts"] == canonical["counts"]
    assert state["failure_ids"] == ["q2"]
    assert AtomicCheckpoint._sanitize_runtime_state(state) == state

    archive = tmp_path / "published.json"
    artifact = publish_checkpoint_artifact(
        resumed, archive, payload={"benchmark": "unit"}
    )
    resumed.close()
    archive_wire = archive.read_text(encoding="utf-8")
    assert secret not in checkpoint.read_text(encoding="utf-8")
    assert secret not in archive_wire
    assert artifact["execution"]["counts"] == canonical["counts"]
    assert len(artifact["per_question"]) == 2


def test_retry_reopen_removes_stale_finalization_before_mutation(
    tmp_path: Path,
):
    manifest = _manifest(ids=("q1", "q2"))
    checkpoint = tmp_path / "reopened.json"
    ledger = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=("q1", "q2")
    )
    ledger.record("q1", row=None, failure="timeout")
    finalized = ledger.finalize()
    assert finalized["counts"]["missing"] == 1
    ledger.close()

    resumed = AtomicCheckpoint(
        checkpoint,
        manifest=manifest,
        expected_ids=("q1", "q2"),
        resume=True,
        retry_failures=True,
    )
    reopened = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert reopened["status"] == "running"
    assert "counts" not in reopened
    assert "failure_ids" not in reopened
    assert set(reopened) == {
        "schema", "run_id", "manifest", "expected_ids", "scored",
        "verdict_key", "entries", "execution_segments", "status",
    }
    assert resumed.pending_ids == ("q1", "q2")

    resumed.record("q1", row={"question_id": "q1", "correct": True})
    resumed.record("q2", row={"question_id": "q2", "correct": False})
    refinalized = resumed.finalize()
    resumed.close()
    assert refinalized["counts"] == {
        "expected": 2,
        "attempted": 2,
        "unique_attempted": 2,
        "total_attempts": 3,
        "completed": 2,
        "failed": 0,
        "missing": 0,
    }
    assert refinalized["failure_ids"] == []


@pytest.mark.parametrize("field", ["counts", "failure_ids"])
def test_complete_checkpoint_rejects_forged_finalization_metadata(
    tmp_path: Path, field: str,
):
    manifest = _manifest(ids=("q1", "q2"))
    checkpoint = tmp_path / f"forged-{field}.json"
    ledger = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=("q1", "q2")
    )
    ledger.record("q1", row={"question_id": "q1", "correct": True})
    ledger.record("q2", row=None, failure="timeout")
    ledger.finalize()
    ledger.close()
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    if field == "counts":
        raw["counts"]["expected"] = True
    else:
        raw["failure_ids"] = []
    checkpoint.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BenchmarkIntegrityError, match=f"finalized {field.replace('_', ' ')}"):
        AtomicCheckpoint(
            checkpoint,
            manifest=manifest,
            expected_ids=("q1", "q2"),
            resume=True,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_scored", "scored/diagnostic mode"),
        ("missing_verdict", "verdict-key"),
        ("type_equivalent_manifest", "manifest was modified"),
    ],
)
def test_resume_rejects_corrupt_known_root_identity_fields(
    tmp_path: Path, mutation: str, match: str,
):
    manifest = _manifest(ids=("q1",))
    checkpoint = tmp_path / f"identity-{mutation}.json"
    ledger = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=("q1",)
    )
    ledger.close()
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    if mutation == "missing_scored":
        raw.pop("scored")
    elif mutation == "missing_verdict":
        raw.pop("verdict_key")
    else:
        # Python considers True == 1. Canonical JSON identity must not.
        raw["manifest"]["label_free_answer_path"] = 1
    checkpoint.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BenchmarkIntegrityError, match=match):
        AtomicCheckpoint(
            checkpoint,
            manifest=manifest,
            expected_ids=("q1",),
            resume=True,
        )


def test_checkpoint_binds_denominator_and_scored_mode_to_manifest(
    tmp_path: Path,
):
    manifest = _manifest(ids=("q1", "q2"))
    with pytest.raises(BenchmarkIntegrityError, match="expected ids.*manifest"):
        AtomicCheckpoint(
            tmp_path / "wrong-denominator.json",
            manifest=manifest,
            expected_ids=("q1",),
        )
    with pytest.raises(BenchmarkIntegrityError, match="scored mode.*manifest"):
        AtomicCheckpoint(
            tmp_path / "wrong-mode.json",
            manifest=manifest,
            expected_ids=("q1", "q2"),
            scored=False,
        )
    with pytest.raises(BenchmarkIntegrityError, match="must be a boolean"):
        AtomicCheckpoint(
            tmp_path / "malformed-mode.json",
            manifest=manifest,
            expected_ids=("q1", "q2"),
            scored="true",  # type: ignore[arg-type]
        )


def test_zero_recompute_export_rejects_root_identity_cross_binding(
    tmp_path: Path,
):
    manifest = _manifest(ids=("q1", "q2"))
    checkpoint = tmp_path / "cross-bound.json"
    ledger = AtomicCheckpoint(
        checkpoint, manifest=manifest, expected_ids=("q1", "q2")
    )
    ledger.record("q1", row={"question_id": "q1", "correct": True})
    ledger.record("q2", row={"question_id": "q2", "correct": False})
    ledger.finalize()
    ledger.close()

    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    raw["expected_ids"] = ["q1"]
    raw["entries"] = {"q1": raw["entries"]["q1"]}
    raw["counts"] = {
        "expected": 1,
        "attempted": 1,
        "unique_attempted": 1,
        "total_attempts": 1,
        "completed": 1,
        "failed": 0,
        "missing": 0,
    }
    raw["failure_ids"] = []
    checkpoint.write_text(json.dumps(raw), encoding="utf-8")
    archive = tmp_path / "must-not-publish.json"

    with pytest.raises(BenchmarkIntegrityError, match="expected ids.*manifest"):
        export_checkpoint_without_recompute(checkpoint, archive)
    assert not archive.exists()
