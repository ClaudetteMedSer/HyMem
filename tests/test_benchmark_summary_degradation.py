"""Versioned nonblocking summary quality never waives indexing integrity."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from benchmarks import archive_evidence, lme_protocol as protocol, strictness
from benchmarks import msc_adapter as msc
from hymem.dreaming.runner import DreamReport
from tests import test_lme_protocol_hardening as fixtures
from tests.test_msc_locomo_convergence import _adapter, _status as msc_status


def status(*, degraded=1, missing=1, **overrides):
    value = fixtures._current_indexing_status(
        summary_degraded_sessions=degraded,
        summary_missing_sessions=missing,
        malformed_summaries=0,
        summary_healthy=degraded == 0,
    )
    value.update(overrides)
    return value


def raw_summary(*, final=None, report=None):
    return strictness.converge_indexing(
        lambda: asdict(DreamReport()) if report is None else report,
        status=lambda: status() if final is None else final,
        max_cycles=1, timeout_s=10,
    )


def summary(*, final=None):
    return protocol.canonicalize_lme_indexing_summary(raw_summary(final=final))


def artifact():
    value = fixtures.make_artifact()
    value["config"].update(
        no_dream=False, indexing_completion_policy=strictness.INDEXING_COMPLETION_POLICY,
    )
    item_summary = summary()
    value["per_question"][0]["indexing"] = deepcopy(item_summary)
    segment = value["execution"]["segments"][0]
    segment["indexing_runs"] = [{"question_id": "qid", "summary": item_summary}]
    segment["latest_indexing"] = deepcopy(segment["indexing_runs"][-1])
    segment["extraction_canary"] = fixtures._passed_canary(value["models"]["memory_pipeline"])
    fixtures._refresh_manifest(value)
    fixtures._refresh_result_digest(value)
    return value


def test_summary_only_degradation_completes_in_one_cycle():
    value = raw_summary()
    assert value["complete"] is value["healthy"] is True
    assert value["summary_healthy"] is False
    assert value["outcome"] == "success_with_summary_degradation"
    assert value["cycles"] == 1
    assert value["failure_reason"] is None
    assert value["final_status"]["summary_missing_sessions"] == 1


@pytest.mark.parametrize("missing", [0, 1])
def test_lme_roundtrip_preserves_missing_and_stale_summary_counts(missing):
    value = summary(final=status(missing=missing))
    assert protocol._validate_indexing(value) is True
    assert value["schema"] == "hymem-lme-indexing-summary-v6"
    assert value["final_status"]["summary_health"] == {
        "summary_degraded_sessions": 1, "summary_missing_sessions": missing,
        "malformed_summaries": 0, "summary_healthy": False,
    }


def test_current_summary_still_reports_clean_success():
    value = summary(final=status(degraded=0, missing=0))
    assert value["outcome"] == "success"
    assert value["summary_healthy"] is True
    assert protocol._validate_indexing(value) is True


@pytest.mark.parametrize("field,bad", [
    ("summary_degraded_sessions", True), ("summary_degraded_sessions", -1),
    ("summary_degraded_sessions", 1.0), ("summary_missing_sessions", 2),
    ("summary_missing_sessions", "1"), ("summary_healthy", True),
    ("summary_healthy", 0), ("malformed_summaries", False),
    ("summary_degraded_sessions", 2_147_483_648),
    ("summary_new_backlog", 1),
])
def test_malformed_or_unclassified_summary_health_fails_closed(field, bad):
    with pytest.raises(strictness.IndexingConvergenceError):
        raw_summary(final=status(**{field: bad}))


@pytest.mark.parametrize("field", strictness.SUMMARY_STATUS_FIELDS)
def test_missing_summary_health_is_never_assumed_clean(field):
    final = status()
    del final[field]
    with pytest.raises(strictness.IndexingConvergenceError):
        raw_summary(final=final)


@pytest.mark.parametrize("field", [
    "quarantined_chunks", "quarantined_digests", "quarantined_profiles",
    "quarantined_facts", "malformed_digests", "malformed_summaries",
    "terminal_loss_chunks", "coverage_integrity_failures", "pending_digests",
    "pending_message_embeddings", "pending_aggregation",
])
def test_summary_degradation_never_waives_required_indexing_gates(field):
    with pytest.raises(strictness.IndexingConvergenceError):
        raw_summary(final=status(**{field: 1}))


@pytest.mark.parametrize("field", strictness._CURRENT_DREAM_REPORT_FAILURE_FIELDS)
def test_summary_degradation_never_waives_required_report_failures(field):
    report = asdict(DreamReport())
    report[field] = 1
    with pytest.raises(strictness.IndexingConvergenceError):
        raw_summary(report=report)


@pytest.mark.parametrize("kind", ["outcome", "summary_bool", "missing_counter", "extra_counter", "old_schema", "cleanup"])
def test_versioned_lme_rejects_forged_summary_evidence(kind):
    value = summary()
    if kind == "outcome":
        value["outcome"] = "success"
    elif kind == "summary_bool":
        value["summary_healthy"] = True
    elif kind == "missing_counter":
        del value["final_status"]["summary_health"]["summary_missing_sessions"]
    elif kind == "extra_counter":
        value["final_status"]["summary_health"]["ignored"] = 0
    elif kind == "old_schema":
        value["schema"] = "hymem-lme-indexing-summary-v5"
    else:
        value["cleanup_errors"] = [{"stage": "dream_fork_close", "exception_type": "RuntimeError"}]
    with pytest.raises(strictness.BenchmarkIntegrityError):
        protocol._validate_indexing(value)


def test_lme_complete_artifact_keeps_scores_denominator_and_paid_usage():
    value = artifact()
    validated = protocol.validate_strict_artifact(value)
    assert validated["counts"] == value["execution"]["counts"]
    assert validated["scores"]["OVERALL"] == {"accuracy": 100.0, "count": 1}
    assert validated["answer_calls"] == validated["judge_calls"] == 1
    assert validated["summary_degraded_questions"] == 1
    assert validated["summary_degraded_sessions"] == 1
    assert validated["rows"][0]["benchmark_failure"] is None


def test_registry_discloses_degradation_without_changing_score(tmp_path, monkeypatch):
    from benchmarks import lme_registry
    monkeypatch.setattr(lme_registry, "DB", tmp_path / "registry.sqlite")
    path = tmp_path / "longmemeval-v2-hymem-20260919T120000Z-seed7-strict-deadbeef.json"
    path.write_text(json.dumps(artifact()))
    conn = lme_registry.connect()
    try:
        assert lme_registry.ingest_file(conn, path) == "inserted"
        count, accuracy, extras = conn.execute("SELECT count, overall, extras FROM runs").fetchone()
        assert count == 1
        assert accuracy == 100.0
        disclosed = json.loads(extras)["strict_validation"]
        assert disclosed["summary_degraded_questions"] == 1
        assert disclosed["summary_degraded_sessions"] == 1
    finally:
        conn.close()


@pytest.mark.parametrize("mutation", ["policy", "missing_policy", "segment", "reader_usage"])
def test_rehashed_artifact_cannot_hide_policy_or_diverge_from_execution(mutation):
    value = artifact()
    if mutation == "policy":
        value["config"]["indexing_completion_policy"] = "all-summaries-required-v0"
    elif mutation == "missing_policy":
        del value["config"]["indexing_completion_policy"]
    elif mutation == "segment":
        for item in (value["execution"]["segments"][0]["indexing_runs"][0],
                     value["execution"]["segments"][0]["latest_indexing"]):
            item["summary"] = summary(final=status(degraded=0, missing=0))
    else:
        value["execution"]["segments"][0]["reader_usage"] = fixtures._zero_usage()
    fixtures._refresh_manifest(value)
    fixtures._refresh_result_digest(value)
    with pytest.raises(strictness.BenchmarkIntegrityError):
        protocol.validate_strict_artifact(value)


def test_beam_raw_admission_retains_explicit_summary_outcome():
    value = raw_summary()
    config = {"indexing_max_cycles": 1, "indexing_timeout_s": 10,
              "indexing_completion_policy": strictness.INDEXING_COMPLETION_POLICY}
    assert archive_evidence.validate_convergence_summary(value, config=config)
    value["outcome"] = "success"
    with pytest.raises(strictness.BenchmarkIntegrityError):
        archive_evidence.validate_convergence_summary(value, config=config)


def test_duplicate_malformed_summary_evidence_must_reconcile():
    value = summary()
    value["final_status"]["summary_health"]["malformed_summaries"] = 1
    with pytest.raises(strictness.BenchmarkIntegrityError, match="disagrees"):
        protocol._validate_indexing(value)


def test_failed_raw_summary_does_not_echo_invalid_summary_health_scalar():
    with pytest.raises(strictness.IndexingConvergenceError) as caught:
        raw_summary(final=status(summary_healthy="PRIVATE_INVALID_VALUE"))
    assert caught.value.summary["summary_healthy"] is None
    assert caught.value.summary["outcome"] == "failure"


def test_msc_locomo_provenance_and_store_attestation_keep_summary_degradation():
    final = msc_status(**{field: status()[field] for field in strictness.SUMMARY_STATUS_FIELDS})
    adapter, handle = _adapter([{}], [final])
    adapter.dream(max_cycles=1, timeout_s=10)
    receipt = adapter.indexing_provenance(scope_id="locomo:conv")
    attestation = msc._canonical_indexing_attestation(receipt, item={"id": "conv"})
    assert handle.closed
    assert receipt["outcome"] == "success_with_summary_degradation"
    assert receipt["summary_healthy"] is False
    assert attestation["final_status"]["summary_missing_sessions"] == 1
    assert attestation["pipeline_usage"]["calls"] == 7
    receipt["outcome"] = "success"
    with pytest.raises(strictness.BenchmarkIntegrityError):
        msc._canonical_indexing_attestation(receipt, item={"id": "conv"})
