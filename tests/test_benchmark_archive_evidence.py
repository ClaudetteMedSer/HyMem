"""Admission regressions: public rehashing must not hide contradictory evidence."""
from copy import deepcopy
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks import beam_registry, locomo_registry, msc_registry, lme_protocol
from benchmarks.archive_evidence import (
    validate_checkpoint_attestation, validate_convergence_summary,
    validate_scoped_indexing,
)
from benchmarks.strictness import (
    AtomicCheckpoint, BenchmarkIntegrityError, content_hash,
    prepare_checkpoint_artifact,
)
from tests.archive_evidence_fixtures import (
    bind_checkpoint, healthy_convergence, skipped_indexing, scoped_indexing,
)


@pytest.fixture(params=["BEAM", "LongMemEval", "LoCoMo", "MSC"])
def archive_case(request, monkeypatch, tmp_path):
    benchmark = request.param
    if benchmark == "BEAM":
        from tests.test_beam_registry_strict import _strict_artifact
        artifact = _strict_artifact()
        validate = lambda data: beam_registry._beam_row(data, Path("beam-strict.json"))
    elif benchmark == "LongMemEval":
        from tests.test_lme_protocol_hardening import make_artifact
        artifact = make_artifact()
        validate = lme_protocol.validate_strict_artifact
    elif benchmark == "LoCoMo":
        from tests.test_locomo_checkpoint_resume import _strict_archive
        path, artifact = _strict_archive(monkeypatch, tmp_path)
        validate = lambda data: locomo_registry._locomo_row(data, path)
    else:
        from tests.test_msc_checkpoint_resume import msc, _example, _success, _argv, _archive
        monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [_example("q1")])
        monkeypatch.setattr(msc, "run_recall", _success)
        monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--checkpoint", tmp_path / "msc.checkpoint.json"))
        msc.main()
        _path, artifact = _archive(tmp_path)
        validate = msc_registry.validate_msc_artifact
    validate(artifact)
    return artifact, validate


def test_finalized_writer_to_registry_roundtrip(archive_case, tmp_path):
    artifact, validate = archive_case
    rows = artifact["per_question"]
    with AtomicCheckpoint(
        tmp_path / "roundtrip.json", manifest=artifact["manifest"],
        expected_ids=[row["question_id"] for row in rows],
        scored=artifact["manifest"]["scored_run"],
        verdict_key="result_valid" if artifact["benchmark"] == "BEAM" else "correct",
    ) as ledger:
        for row in rows:
            ledger.record(row["question_id"], row=row)
        for segment in artifact["execution"]["segments"]:
            ledger.update_execution_segment(segment["segment_id"], {
                key: value for key, value in segment.items() if key != "segment_id"
            })
        payload = {key: value for key, value in artifact.items()
                   if key not in {"manifest", "config", "models", "execution", "per_question"}}
        emitted = prepare_checkpoint_artifact(ledger, payload=payload)
    if "result_digest" in emitted:
        emitted["result_digest"] = content_hash(emitted["per_question"])
    # A serialization boundary exercises the exact portable JSON receipt.
    validate(json.loads(json.dumps(emitted)))


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(state_sha256="sha256:" + "f" * 64),
    lambda r: r["state"].update(manifest_sha256="sha256:" + "f" * 64),
    lambda r: r["state"].update(rows_sha256="sha256:" + "f" * 64),
    lambda r: r["state"].update(segments_sha256="sha256:" + "f" * 64),
    lambda r: r["state"].update(scored=1),
    lambda r: r["state"].update(counts=[]),
    lambda r: r["state"].update(failure_ids=[]),
    lambda r: r["state"]["entries"][0].update(question_id=[]),
    lambda r: r["state"]["entries"][0].update(attempts=True),
    lambda r: r["state"]["entries"][0].update(row_sha256="sha256:" + "f" * 64),
])
def test_checkpoint_projection_cannot_be_forged_by_rehashing(archive_case, mutate):
    artifact, validate = archive_case
    receipt = artifact["execution"]["checkpoint"]
    original = deepcopy(receipt)
    mutate(receipt)
    if receipt == original:  # Empty failure_ids is already true for some baselines.
        receipt["state"]["failure_ids"] = ["nonexistent"]
    if receipt["state_sha256"] == original["state_sha256"]:
        receipt["state_sha256"] = content_hash(receipt["state"])
    with pytest.raises(BenchmarkIntegrityError):
        validate(artifact)


def test_legacy_opaque_checkpoint_is_not_definitive_evidence(archive_case):
    artifact, validate = archive_case
    artifact["execution"]["checkpoint"] = {"schema": "hymem-benchmark-checkpoint-v1", "state_sha256": "sha256:" + "1" * 64}
    with pytest.raises(BenchmarkIntegrityError, match="missing or obsolete"):
        validate(artifact)


def test_unattempted_denominator_needs_no_invented_indexing_or_calls(archive_case):
    artifact, validate = archive_case
    from tests.test_msc_checkpoint_resume import _zero_llm
    for row in artifact["per_question"]:
        row.update(correct=False, strict_failure=True, benchmark_failure="missing_prediction")
        if artifact["benchmark"] == "BEAM":
            row.update(result_valid=False, score=0.0, llm_judge_score=0.0,
                       scores=[], judge_criterion_results=[], judge_parse="not_called")
    count = len(artifact["per_question"])
    artifact["execution"]["counts"] = {
        "expected": count, "attempted": 0, "unique_attempted": 0,
        "total_attempts": 0, "completed": 0, "failed": count, "missing": count,
    }
    for segment in artifact["execution"]["segments"]:
        segment.update(attempted_attempts=0, indexing_runs=[], latest_indexing=None)
        if artifact["benchmark"] in {"BEAM", "LongMemEval"}:
            segment["status"] = "running"
        if "indexing_failures" in segment:
            segment["indexing_failures"] = []
        for role in ("reader_usage", "judge_usage", "memory_pipeline_usage"):
            segment[role] = _zero_llm()
    benchmark = artifact["benchmark"]
    if benchmark == "BEAM":
        artifact.pop("summary", None)
        artifact.pop("summary_counts", None)
    elif benchmark == "LongMemEval":
        artifact["scores"] = {key: {**value, "accuracy": 0.0} for key, value in artifact["scores"].items()}
        artifact["conditional_judged_only"] = {"accuracy": None, "count": 0}
        artifact["abstention_diagnostics"] = lme_protocol._abstention_from_rows(artifact["per_question"])
    elif benchmark == "LoCoMo":
        artifact["scores"] = locomo_registry._strict_scores(artifact["per_question"])
    if "result_digest" in artifact:
        artifact["result_digest"] = content_hash(artifact["per_question"])
    bind_checkpoint(artifact)
    validate(artifact)


def test_unavailable_historical_usage_preserves_lower_bound_and_unavailable_total(archive_case):
    artifact, validate = archive_case
    from benchmarks.strictness import usage_snapshot
    segment = deepcopy(artifact["execution"]["segments"][0])
    segment.update(segment_id="interrupted-zero-work", status="running", attempted_attempts=0,
                   indexing_runs=[], latest_indexing=None)
    if "indexing_failures" in segment:
        segment["indexing_failures"] = []
    for role in ("reader_usage", "judge_usage", "memory_pipeline_usage"):
        segment[role] = usage_snapshot(None)
    artifact["execution"]["segments"].append(segment)
    bind_checkpoint(artifact)
    validate(artifact)


@pytest.mark.parametrize("scope", ["msc:q1", "locomo:c1"])
@pytest.mark.parametrize("mutate", [
    lambda r: r["settings"].update(require_healthy=1),
    lambda r: r["settings"].update(max_cycles_per_convergence=True),
    lambda r: r["settings"].update(timeout_s_per_convergence=True),
    lambda r: r.update(observed_status=[]),
    lambda r: r.update(observed_status="healthy"),
    lambda r: r.update(complete=True),
    lambda r: r.update(scope_id="other:source"),
    lambda r: r.update(skip_reason="no_dream"),
])
def test_skipped_indexing_exact_mode_scope_and_types(scope, mutate):
    receipt = skipped_indexing(scope)
    config = {"sim": True, "no_dream": True, "indexing_max_cycles": 100, "indexing_timeout_s": 3600.0}
    validate_scoped_indexing(receipt, scope_id=scope, config=config)
    mutate(receipt)
    with pytest.raises(BenchmarkIntegrityError):
        validate_scoped_indexing(receipt, scope_id=scope, config=config)


@pytest.mark.parametrize("scope", ["msc:q1", "locomo:c1"])
@pytest.mark.parametrize("mutate", [
    lambda r: r.update(healthy=False),
    lambda r: r["final_status"].update(pending_chunks=1200),
    lambda r: r["final_status"].update(pending_chunks=False),
    lambda r: r["final_status"].update(dream_status_schema="hymem-dream-status-v6"),
    lambda r: r["runs"][0].update(report_count=0),
    lambda r: r["runs"][0]["final_cycle"].update(chunk_extraction_failures=1),
])
def test_live_scoped_receipt_cannot_claim_unhealthy_completion(scope, mutate):
    args = SimpleNamespace(sim=False, no_dream=False, indexing_max_cycles=100, indexing_timeout_s=3600.0)
    receipt = scoped_indexing(scope, args)
    validate_scoped_indexing(receipt, scope_id=scope, config=vars(args))
    mutate(receipt)
    with pytest.raises(BenchmarkIntegrityError):
        validate_scoped_indexing(receipt, scope_id=scope, config=vars(args))


def test_emitted_simulation_cannot_be_promoted_even_with_all_public_hashes_rebound(monkeypatch, tmp_path):
    from tests.test_locomo_checkpoint_resume import _strict_archive
    path, artifact = _strict_archive(monkeypatch, tmp_path)
    for hide_simulation in (False, True):
        forged = deepcopy(artifact)
        config = forged["config"]
        config["scored_run"] = True
        if hide_simulation:
            config.update(sim=False, no_dream=False)
        manifest = forged["manifest"]
        manifest.update(scored_run=True, config=deepcopy(config), config_hash=content_hash(config))
        manifest["run_id"] = content_hash({key: value for key, value in manifest.items() if key != "run_id"})
        for row in forged["per_question"]:
            row["strict_failure"] = False
        forged["strict_accuracy"] = 1.0
        forged["result_digest"] = content_hash(forged["per_question"])
        bind_checkpoint(forged)
        with pytest.raises(ValueError):
            locomo_registry._locomo_row(forged, path)


def test_raw_convergence_requires_current_healthy_completion():
    config = {"indexing_max_cycles": 100, "indexing_timeout_s": 3600.0}
    receipt = healthy_convergence(config)
    assert validate_convergence_summary(receipt, config=config)
    for key, value in (("pending_chunks", 1200), ("pending_chunks", False), ("in_progress", True)):
        forged = deepcopy(receipt)
        forged["final_status"][key] = value
        with pytest.raises(BenchmarkIntegrityError):
            validate_convergence_summary(forged, config=config)


@pytest.mark.parametrize("failure", ["exception", "phase1", "malformed_report"])
def test_actual_failed_convergence_writer_remains_failed_readable(failure):
    from benchmarks.strictness import converge_indexing, IndexingConvergenceError, sanitize_for_artifact
    from tests.test_lme_protocol_hardening import _current_indexing_status, _current_indexing_report
    config = {"indexing_max_cycles": 100, "indexing_timeout_s": 3600.0}
    status = _current_indexing_status()
    if failure == "phase1":
        status.update(phase1_backlog_status="producer_unavailable", pending_chunks_authoritative=False,
                      phase1_generation_key=None)
    def dream():
        if failure == "exception":
            raise RuntimeError("private operational detail must never appear")
        return {} if failure == "malformed_report" else _current_indexing_report()
    with pytest.raises(IndexingConvergenceError) as caught:
        converge_indexing(dream, status=lambda: status, max_cycles=100, timeout_s=3600.0)
    receipt = dict(caught.value.summary)
    assert sanitize_for_artifact(receipt) == receipt
    receipt = json.loads(json.dumps(sanitize_for_artifact(receipt)))
    if failure == "phase1":
        assert receipt["failure_reason"] == "phase1_producer_unavailable"
    assert "private operational" not in json.dumps(receipt)
    assert validate_convergence_summary(receipt, config=config, allow_failure=True) is False
    for scope in ("msc:q1", "locomo:c1"):
        assert validate_scoped_indexing(receipt, scope_id=scope, config=config, failed=True) is False
    with pytest.raises(BenchmarkIntegrityError):
        validate_convergence_summary(receipt, config=config)
    canonical = lme_protocol.canonicalize_lme_indexing_summary(receipt)
    assert canonical["outcome"] == "failure"
    assert lme_protocol._validate_versioned_indexing(canonical, allow_failure=True, require_healthy=True) is False


@pytest.mark.parametrize("benchmark", ["BEAM", "LongMemEval"])
@pytest.mark.parametrize("mutation", ["missing", "unavailable", "mismatch", "garbage_key", "old_normalized_schema"])
def test_rehashed_success_requires_retained_current_phase1_authority(benchmark, mutation):
    if benchmark == "BEAM":
        from tests.test_beam_registry_strict import _strict_artifact
        artifact = _strict_artifact()
        summaries = artifact["execution"]["segments"][0]["indexing_runs"]
        validate = lambda value: beam_registry._beam_row(value, Path("beam-strict.json"))
    else:
        from tests.test_lme_protocol_hardening import make_artifact, _canonical_indexing_summary, _passed_canary, _refresh_manifest
        artifact = make_artifact()
        artifact["config"]["no_dream"] = False
        summary = _canonical_indexing_summary(failed=False)
        artifact["per_question"][0]["indexing"] = summary
        segment = artifact["execution"]["segments"][0]
        segment["indexing_runs"] = [{"question_id": "qid", "summary": summary}]
        segment["latest_indexing"] = segment["indexing_runs"][0]
        segment["extraction_canary"] = _passed_canary(artifact["models"]["memory_pipeline"])
        artifact["result_digest"] = content_hash(artifact["per_question"])
        _refresh_manifest(artifact)
        summaries = [summary]
        validate = lme_protocol.validate_strict_artifact
    validate(artifact)
    for summary in summaries:
        final = summary["final_status"]
        if mutation == "missing":
            final.pop("phase1_generation_key")
        elif mutation == "unavailable":
            final.update(phase1_backlog_status="producer_unavailable", pending_chunks_authoritative=False,
                         phase1_generation_key=None)
        elif mutation == "mismatch":
            final["pending_chunks_authoritative"] = False
        elif mutation == "garbage_key":
            final["phase1_generation_key"] = "not-a-generation"
        elif benchmark == "LongMemEval":
            summary["schema"] = "hymem-lme-indexing-summary-v4"
        else:
            final["dream_status_schema"] = "hymem-dream-status-v6"
    if "result_digest" in artifact:
        artifact["result_digest"] = content_hash(artifact["per_question"])
    bind_checkpoint(artifact)
    with pytest.raises(BenchmarkIntegrityError):
        validate(artifact)


@pytest.mark.parametrize("field", ["phase1_backlog_status", "pending_chunks_authoritative", "phase1_generation_key"])
@pytest.mark.parametrize("value", [[], {}, None, 1])
def test_malformed_phase1_authority_is_an_integrity_error(field, value):
    from tests.test_lme_protocol_hardening import _current_indexing_status
    raw = _current_indexing_status()
    canonical = lme_protocol._canonical_final_indexing_status(raw)
    raw[field] = value
    canonical[field] = value
    with pytest.raises(BenchmarkIntegrityError):
        lme_protocol._canonical_final_indexing_status(raw)
    with pytest.raises(BenchmarkIntegrityError):
        lme_protocol._validate_canonical_final_status(canonical)


@pytest.mark.parametrize("benchmark", ["MSC", "LoCoMo"])
def test_actual_live_writer_requires_reader_and_judge_calls(monkeypatch, tmp_path, benchmark):
    from tests.test_msc_checkpoint_resume import _UsageClient, _zero_llm
    if benchmark == "MSC":
        from tests.test_msc_checkpoint_resume import msc as adapter, _example, _scored_success, _archive
        monkeypatch.setattr(adapter, "load_msc_data", lambda *_a, **_k: [_example("q1")])
        monkeypatch.setattr(adapter, "run_recall", _scored_success)
        validate = msc_registry.validate_msc_artifact
    else:
        from tests.test_locomo_checkpoint_resume import locomo as adapter, _conversation, _row, _runtime
        conv = _conversation("c1", "q1")
        monkeypatch.setattr(adapter, "load_locomo_data", lambda *_a, **_k: [conv])
        def evaluate(conversation, args, answer, judge, **kwargs):
            for client in (answer, judge):
                client.call_count += 1
                client.request_attempts += 1
                client.successful_responses += 1
            row = _row(conversation, conversation["qa"][0])
            runtime = _runtime(conversation)
            runtime["indexing"] = scoped_indexing("locomo:c1", args)
            kwargs["on_checkpoint"](row, runtime)
            return [row]
        monkeypatch.setattr(adapter, "evaluate_conversation", evaluate)
        monkeypatch.setattr(adapter, "_print_report", lambda *_a, **_k: None)
        validate = lambda data: locomo_registry._locomo_row(data, Path("locomo-strict.json"))
    monkeypatch.setattr(adapter, "_build_llm", lambda *_a, **_k: _UsageClient())
    monkeypatch.setattr(sys, "argv", [f"{benchmark.lower()}_adapter.py", "--no-dream",
        "--checkpoint", str(tmp_path / "live.json"), "--results-dir", str(tmp_path / "results")])
    adapter.main()
    artifact = json.loads(next((tmp_path / "results").glob("*-strict-*.json")).read_text())
    validate(artifact)
    for role in ("reader_usage", "judge_usage"):
        forged = deepcopy(artifact)
        forged["execution"]["segments"][0][role] = _zero_llm()
        bind_checkpoint(forged)
        with pytest.raises(ValueError, match="calls.*below"):
            validate(forged)


def test_rehashed_beam_extraction_attempts_cannot_exceed_pipeline_meter():
    from tests.test_beam_registry_strict import _strict_artifact
    artifact = _strict_artifact()
    segment = artifact["execution"]["segments"][0]
    segment["indexing_runs"][0]["reports"][0]["chunk_extraction_provider_attempts"] = 100
    bind_checkpoint(artifact)
    with pytest.raises(ValueError, match="attempts exceed"):
        beam_registry._beam_row(artifact, Path("beam-strict.json"))


@pytest.mark.parametrize("scope", ["msc:q1", "locomo:c1"])
def test_scoped_extraction_attempts_cannot_exceed_pipeline_meter(scope):
    args = SimpleNamespace(sim=False, no_dream=False, indexing_max_cycles=100, indexing_timeout_s=3600.0)
    receipt = scoped_indexing(scope, args)
    receipt["dream_report_totals"]["chunk_extraction_provider_attempts"] = 1
    receipt["runs"][0]["dream_report_totals"]["chunk_extraction_provider_attempts"] = 1
    with pytest.raises(BenchmarkIntegrityError, match="attempts exceed"):
        validate_scoped_indexing(receipt, scope_id=scope, config=vars(args))
