"""Honest early indexing failures remain bounded, inspectable and unscorable.

No provider is constructed: fake dream reports exercise real convergence and
adapter canonicalization; the artifact controls reuse existing exact fixtures.
"""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from benchmarks import lme_protocol as protocol
from benchmarks import longmemeval_adapter as lme
from benchmarks.strictness import (
    BenchmarkIntegrityError,
    IndexingConvergenceError,
    converge_indexing,
)
from tests.test_lme_protocol_hardening import (
    _canonical_indexing_summary,
    _current_indexing_report,
    _current_indexing_status,
    _indexing_failure_artifact,
    _refresh_result_digest,
    _usage,
)


_CODES = (
    "quarantined_extraction",
    "terminal_extraction_source_loss",
    "malformed_durable_state",
)
_WIRES = ("current", "legacy")
_BLOCKERS = (
    *(f"pending:{name}" for name in sorted(protocol._INDEXING_PENDING_FIELDS)),
    "in_progress",
    *(f"flag:{name}" for name in sorted(protocol._INDEXING_REPORT_BOOLEAN_FIELDS)),
    *(f"failure:{name}" for name in sorted(protocol._INDEXING_CYCLE_FAILURE_FIELDS)),
)


def _raw_failure(code, blocker=None):
    final = _current_indexing_status()
    report = _current_indexing_report()
    if code == "quarantined_extraction":
        final["quarantined_facts"] = 1
    elif code == "terminal_extraction_source_loss":
        final["terminal_loss_chunks"] = 1
        final["terminal_loss_reasons"] = {"source_manifest_unrecoverable": 1}
    elif code == "malformed_durable_state":
        final["malformed_facts"] = 1
    else:
        raise AssertionError(code)
    if blocker == "in_progress":
        final["in_progress"] = True
    elif blocker is not None:
        kind, name = blocker.split(":", 1)
        if kind == "pending":
            final[name] = 1
        else:
            report[name] = True if kind == "flag" else 1
    return {
        "cycles": 1, "max_cycles": 3, "timeout_s": 10.0, "elapsed_s": 1.0,
        "complete": blocker is None, "healthy": False,
        "failure_reason": code, "reports": [report], "final_status": final,
        "quarantined": {
            name: count for name, count in final.items()
            if "quarantined" in name and count > 0
        },
    }


def _summary(code, wire, blocker=None):
    raw = _raw_failure(code, blocker)
    if wire == "current":
        return protocol.canonicalize_lme_indexing_summary(raw)
    assert wire == "legacy"
    return raw


def _set_count(summary, wire, family, field, value):
    if wire == "current":
        summary["final_status"][family][field] = value
    else:
        summary["final_status"][field] = value
        if family == "quarantined":
            summary["quarantined"] = {
                name: count for name, count in summary["final_status"].items()
                if "quarantined" in name and isinstance(count, int) and count > 0
            }


def _clear_evidence(summary, wire, code):
    if code == "quarantined_extraction":
        _set_count(summary, wire, "quarantined", "quarantined_facts", 0)
    elif code == "malformed_durable_state":
        _set_count(summary, wire, "malformed", "malformed_facts", 0)
    elif wire == "current":
        summary["final_status"]["terminal_loss"] = {"chunks": 0, "reasons": {}}
    else:
        summary["final_status"].update({
            "terminal_loss_chunks": 0, "terminal_loss_reasons": {},
        })


@pytest.mark.parametrize("wire", _WIRES)
@pytest.mark.parametrize("code", _CODES)
@pytest.mark.parametrize("blocker", _BLOCKERS)
def test_mixed_failure_accepts_each_real_blocker_without_claiming_drainage(
    wire, code, blocker,
):
    summary = _summary(code, wire, blocker)
    original = deepcopy(summary)
    assert protocol._validate_indexing(summary, allow_incomplete=True) is False
    assert summary == original
    assert summary["complete"] is False and summary["healthy"] is False
    with pytest.raises(BenchmarkIntegrityError, match="did not complete healthy"):
        protocol._validate_indexing(summary)
    summary["complete"] = True
    with pytest.raises(BenchmarkIntegrityError, match="mechanical completion"):
        protocol._validate_indexing(summary, allow_incomplete=True)


@pytest.mark.parametrize("wire", _WIRES)
@pytest.mark.parametrize("code", _CODES)
def test_drained_durable_failure_requires_true_completion_but_remains_unhealthy(
    wire, code,
):
    summary = _summary(code, wire)
    assert protocol._validate_indexing(summary, allow_incomplete=True) is False
    summary["complete"] = False
    with pytest.raises(BenchmarkIntegrityError, match="mechanical completion"):
        protocol._validate_indexing(summary, allow_incomplete=True)


@pytest.mark.parametrize("wire", _WIRES)
@pytest.mark.parametrize("code", _CODES)
@pytest.mark.parametrize("tamper", ("zero", "missing", "wrong_family", "no_cycle"))
def test_mixed_failure_cannot_invent_its_durable_reason(wire, code, tamper):
    summary = _summary(code, wire, "pending:pending_digests")
    if tamper == "missing":
        summary["final_status"] = None
    elif tamper == "no_cycle":
        summary.update({"cycles": 0, "reports": []})
    else:
        _clear_evidence(summary, wire, code)
        if tamper == "wrong_family":
            if code == "quarantined_extraction":
                _set_count(summary, wire, "malformed", "malformed_facts", 1)
            else:
                _set_count(summary, wire, "quarantined", "quarantined_facts", 1)
    with pytest.raises(BenchmarkIntegrityError):
        protocol._validate_indexing(summary, allow_incomplete=True)


@pytest.mark.parametrize("wire", _WIRES)
@pytest.mark.parametrize("code", _CODES)
@pytest.mark.parametrize("field", ("complete", "healthy"))
@pytest.mark.parametrize("bad", (0, 1, None, "false"))
def test_mixed_failure_flags_are_booleans(wire, code, field, bad):
    summary = _summary(code, wire, "pending:pending_digests")
    summary[field] = bad
    with pytest.raises(BenchmarkIntegrityError, match="completion state"):
        protocol._validate_indexing(summary, allow_incomplete=True)


@pytest.mark.parametrize("wire", _WIRES)
@pytest.mark.parametrize("field", sorted(protocol._INDEXING_CYCLE_FAILURE_FIELDS))
@pytest.mark.parametrize("bad", (-1, True, "1", None))
def test_mixed_failure_cannot_hide_malformed_supplied_report_counters(wire, field, bad):
    summary = _summary("quarantined_extraction", wire, "pending:pending_digests")
    summary["reports"][0][field] = bad
    with pytest.raises(BenchmarkIntegrityError):
        protocol._validate_indexing(summary, allow_incomplete=True)


@pytest.mark.parametrize("wire", _WIRES)
@pytest.mark.parametrize("field", sorted(protocol._INDEXING_REPORT_BOOLEAN_FIELDS))
@pytest.mark.parametrize("bad", (0, 1, "false", None))
def test_mixed_failure_cannot_hide_malformed_supplied_report_flags(wire, field, bad):
    summary = _summary("quarantined_extraction", wire, "pending:pending_digests")
    summary["reports"][0][field] = bad
    with pytest.raises(BenchmarkIntegrityError):
        protocol._validate_indexing(summary, allow_incomplete=True)


@pytest.mark.parametrize("wire", _WIRES)
@pytest.mark.parametrize("family,field", (
    ("pending", "pending_digests"),
    ("quarantined", "quarantined_facts"),
    ("malformed", "malformed_facts"),
))
@pytest.mark.parametrize("bad", (-1, True, "1", None))
def test_mixed_failure_rejects_bad_durable_counts(wire, family, field, bad):
    summary = _summary("quarantined_extraction", wire, "pending:pending_digests")
    _set_count(summary, wire, family, field, bad)
    with pytest.raises(BenchmarkIntegrityError):
        protocol._validate_indexing(summary, allow_incomplete=True)


@pytest.mark.parametrize("wire", _WIRES)
def test_mixed_failure_rejects_unknown_status_counter_and_broken_loss_proof(wire):
    summary = _summary("quarantined_extraction", wire, "pending:pending_digests")
    _set_count(summary, wire, "pending", "pending_future_magic", 0)
    with pytest.raises(BenchmarkIntegrityError):
        protocol._validate_indexing(summary, allow_incomplete=True)
    summary = _summary("terminal_extraction_source_loss", wire, "pending:pending_digests")
    if wire == "current":
        summary["final_status"]["terminal_loss"]["reasons"] = {}
    else:
        summary["final_status"]["terminal_loss_reasons"] = {}
    with pytest.raises(BenchmarkIntegrityError, match="terminal-loss"):
        protocol._validate_indexing(summary, allow_incomplete=True)


def test_legacy_failure_retains_supported_missing_report_field_defaults():
    summary = _summary("quarantined_extraction", "legacy", "pending:pending_digests")
    summary["reports"] = [{"budget_exhausted": False, "skipped_locked": False}]
    assert protocol._validate_indexing(summary, allow_incomplete=True) is False
    summary["final_status"]["pending_digests"] = 0
    summary["complete"] = True
    assert protocol._validate_indexing(summary, allow_incomplete=True) is False


@pytest.mark.parametrize("field", sorted(protocol._INDEXING_CYCLE_FAILURE_FIELDS))
@pytest.mark.parametrize("value", (1, True, -1, "1", None))
def test_legacy_healthy_claim_cannot_ignore_any_supplied_cycle_failure(field, value):
    summary = _raw_failure("quarantined_extraction")
    summary.update({"healthy": True, "failure_reason": None, "quarantined": {}})
    summary["final_status"]["quarantined_facts"] = 0
    summary["reports"][0][field] = value
    with pytest.raises(BenchmarkIntegrityError):
        protocol._validate_indexing(summary)


@pytest.mark.parametrize("value", (True, 1, "false", None))
def test_legacy_healthy_claim_cannot_ignore_provider_attempt_budget_flag(value):
    summary = _raw_failure("quarantined_extraction")
    summary.update({"healthy": True, "failure_reason": None, "quarantined": {}})
    summary["final_status"]["quarantined_facts"] = 0
    summary["reports"][0]["extraction_provider_attempt_budget_exhausted"] = value
    with pytest.raises(BenchmarkIntegrityError):
        protocol._validate_indexing(summary)


def test_legacy_healthy_claim_keeps_existing_missing_report_defaults():
    summary = _raw_failure("quarantined_extraction")
    summary.update({"healthy": True, "failure_reason": None, "quarantined": {}})
    summary["final_status"]["quarantined_facts"] = 0
    summary["reports"] = [{"budget_exhausted": False, "skipped_locked": False}]
    assert protocol._validate_indexing(summary) is True


@pytest.mark.parametrize("code", _CODES)
@pytest.mark.parametrize("blocker", ("pending:pending_digests", "failure:aggregation_fusion_failures"))
def test_real_convergence_mixed_failure_roundtrips_without_a_reclassified_exception(
    code, blocker,
):
    raw = _raw_failure(code, blocker)
    with pytest.raises(IndexingConvergenceError) as failed:
        converge_indexing(
            lambda: raw["reports"][0], status=lambda: raw["final_status"],
            max_cycles=3, timeout_s=10, _clock=lambda: 0.0,
        )
    summary = protocol.canonicalize_lme_indexing_summary(failed.value.summary)
    assert summary["failure"] == {"code": code, "exception_type": None}
    assert summary["complete"] is False and summary["healthy"] is False
    assert protocol._validate_indexing(json.loads(json.dumps(summary)), allow_incomplete=True) is False


def test_adapter_keeps_mixed_failure_summary_and_runs_both_cleanup_actions(monkeypatch):
    raw = _raw_failure("quarantined_extraction", "pending:pending_digests")
    raw["reports"][0]["digest_failures"] = 1
    events = []

    class Fork:
        def dream(self, *, deadline):
            events.append("dream")
            return raw["reports"][0]

        def close(self):
            events.append("close")

    fork = Fork()

    class Parent:
        def fork(self):
            return fork

        def invalidate_query_caches(self):
            events.append("invalidate")

    monkeypatch.setattr(lme, "durable_indexing_status", lambda *_: raw["final_status"])
    adapter = object.__new__(lme.HyMemAdapter)
    adapter.hy = Parent()
    adapter.embedding_client = None
    adapter.last_indexing_summary = None
    with pytest.raises(IndexingConvergenceError) as failed:
        adapter.dream_and_wait(timeout=10.0, max_cycles=3)
    assert events == ["dream", "close", "invalidate"]
    assert failed.value.summary is adapter.last_indexing_summary
    summary = failed.value.summary
    assert summary["failure"]["code"] == "quarantined_extraction"
    assert summary["final_status"]["pending"]["pending_digests"] == 1
    assert summary["final_status"]["quarantined"]["quarantined_facts"] == 1
    assert summary["reports"][0]["digest_failures"] == 1
    assert summary["cleanup_errors"] == []


def _mixed_failure_artifact():
    artifact = _indexing_failure_artifact()
    summary = _summary("quarantined_extraction", "current", "pending:pending_digests")
    row = artifact["per_question"][0]
    row["indexing"] = summary
    row["benchmark_failure"] = "indexing_failure:quarantined_extraction"
    segment = artifact["execution"]["segments"][0]
    recorded = {"question_id": "qid", "summary": deepcopy(summary)}
    segment["indexing_runs"] = [recorded]
    segment["latest_indexing"] = deepcopy(recorded)
    _refresh_result_digest(artifact)
    return artifact


def test_mixed_failure_artifact_stays_a_failed_unjudged_question():
    artifact = _mixed_failure_artifact()
    validated = protocol.validate_strict_artifact(artifact)
    assert validated["counts"]["completed"] == 0
    assert validated["counts"]["failed"] == 1
    assert artifact["conditional_judged_only"] == {"accuracy": None, "count": 0}


@pytest.mark.parametrize("role", ("reader_usage", "judge_usage"))
def test_mixed_failure_artifact_cannot_hide_scoring_calls(role):
    artifact = _mixed_failure_artifact()
    artifact["execution"]["segments"][0][role] = _usage(calls=1)
    _refresh_result_digest(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="fail-before-score"):
        protocol.validate_strict_artifact(artifact)


@pytest.mark.parametrize("field,value", (("answer", "leaked"), ("context_sha", "invented")))
def test_mixed_failure_artifact_cannot_carry_scoring_or_retrieval_evidence(field, value):
    artifact = _mixed_failure_artifact()
    artifact["per_question"][0][field] = value
    _refresh_result_digest(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="scoring/retrieval evidence"):
        protocol.validate_strict_artifact(artifact)


@pytest.mark.parametrize("tamper", ("healthy", "success", "completed"))
def test_mixed_failure_artifact_cannot_be_promoted_to_a_success(tamper):
    artifact = _mixed_failure_artifact()
    row = artifact["per_question"][0]
    if tamper == "healthy":
        row["indexing"]["healthy"] = True
    elif tamper == "success":
        row.update({"benchmark_failure": None, "correct": True})
    else:
        artifact["execution"]["counts"].update({"completed": 1, "failed": 0})
    _refresh_result_digest(artifact)
    with pytest.raises(BenchmarkIntegrityError):
        protocol.validate_strict_artifact(artifact)


def test_coverage_exhaustion_is_not_reinterpreted_as_an_immediate_failure():
    # A bounded coverage retry loop can report incomplete even after the final
    # mechanically drained walk. Preserve its separate terminal semantics.
    summary = _canonical_indexing_summary(failed=True)
    assert summary["complete"] is False
    assert protocol._validate_indexing(summary, allow_incomplete=True) is False
