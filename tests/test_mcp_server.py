from __future__ import annotations

import json
from dataclasses import fields, replace

import pytest

pytest.importorskip("mcp")

import hymem.server as srv
from hymem import HyMem
from hymem.dreaming.lossless import COVERAGE_INTEGRITY_CONFIG_VERSION
from hymem.dreaming.runner import (
    DREAM_REPORT_BOOLEAN_GATE_FIELDS,
    DREAM_REPORT_COUNT_FIELDS,
    DREAM_REPORT_ERROR_FIELDS,
    DREAM_REPORT_FIELD_NAMES,
    DREAM_REPORT_NULLABLE_COUNT_FIELDS,
    DREAM_REPORT_TEXT_FIELDS,
    DreamReport,
)
from hymem.dreaming.status import DREAM_STATUS_SCHEMA_VERSION
from hymem.extraction.llm import StubLLMClient
from tests.conftest import make_routed_llm

_TEST_AGGREGATION_CONFIG_VERSION = "aggregation-build-config-v1:" + ("0" * 64)
_TEST_OLD_AGGREGATION_CONFIG_VERSION = (
    "aggregation-build-config-v1:" + ("1" * 64)
)


def _clean_dream_status(**overrides) -> dict:
    status = {
        "dream_status_schema": DREAM_STATUS_SCHEMA_VERSION,
        "pending_source_materialization": 0,
        "pending_chunks": 0,
        "pending_digests": 0,
        "pending_profiles": 0,
        "pending_facts": 0,
        "quarantined_chunks": 0,
        "quarantined_digests": 0,
        "quarantined_profiles": 0,
        "quarantined_facts": 0,
        "quarantined_facts_malformed": 0,
        "terminal_loss_chunks": 0,
        "terminal_loss_reasons": {},
        "coverage_integrity_failures": 0,
        "coverage_integrity_failure_reasons": {},
        "coverage_integrity_failure_details": [],
        "coverage_integrity_failure_details_truncated": False,
        "coverage_integrity_config_version": (
            COVERAGE_INTEGRITY_CONFIG_VERSION
        ),
        "malformed_source_materialization": 0,
        "malformed_digests": 0,
        "malformed_profiles": 0,
        "malformed_facts": 0,
        "pending_aggregation": 0,
        "phase1_backlog_status": "current_producer",
        "pending_chunks_authoritative": True,
        "phase1_generation_key": (
            "hymem-phase1-generation-v1:" + "0" * 64
        ),
        "in_progress": False,
        # Sanctioned bounded diagnostics from aggregation_health_status().
        "aggregation_enabled": False,
        "aggregation_config_version": None,
        "aggregation_active_build_attempts": 0,
        "aggregation_active_caught_exceptions": 0,
        "aggregation_active_fusion_failures": 0,
        "aggregation_total_caught_exceptions": 0,
        "aggregation_total_fusion_failures": 0,
        "aggregation_superseded_pending_configs": 0,
        "aggregation_last_success_config_version": None,
        "aggregation_last_success_at": None,
        "aggregation_last_failure_config_version": None,
        "aggregation_last_failure_kind": None,
        "aggregation_last_failure_at": None,
        "aggregation_stale_pending_config_version": None,
        "aggregation_generation_key": None,
        "aggregation_publication_generation_key": None,
        "aggregation_last_success_generation_key": None,
        "aggregation_material_epoch_key": None,
        "aggregation_last_success_material_epoch_key": None,
        "aggregation_last_failure_generation_key": None,
        "aggregation_stale_pending_generation_key": None,
        "aggregation_pending_material_epoch_key": None,
        "aggregation_last_failure_material_epoch_key": None,
        "aggregation_publication_generation": None,
        "aggregation_material_binding": None,
        "extraction_provider_attempt_budget": 0,
    }
    status.update(overrides)
    if overrides.get("pending_aggregation"):
        status["aggregation_enabled"] = True
        status["aggregation_config_version"] = (
            _TEST_AGGREGATION_CONFIG_VERSION
        )
    if (
        "terminal_loss_chunks" in overrides
        and "terminal_loss_reasons" not in overrides
    ):
        count = overrides["terminal_loss_chunks"]
        status["terminal_loss_reasons"] = (
            {"test_terminal_loss": count} if count else {}
        )
    if (
        "coverage_integrity_failures" in overrides
        and "coverage_integrity_failure_reasons" not in overrides
    ):
        count = overrides["coverage_integrity_failures"]
        status["coverage_integrity_failure_reasons"] = (
            {"test_coverage_failure": count} if count else {}
        )
        status["coverage_integrity_failure_details"] = (
            [
                {
                    "session_id": f"test-session-{index}",
                    "config_version": COVERAGE_INTEGRITY_CONFIG_VERSION,
                    "failure_reason": "test_coverage_failure",
                    "occurrences": 1,
                    "first_detected_at": "2026-09-06T10:00:00Z",
                    "last_detected_at": "2026-09-06T10:00:00Z",
                }
                for index in range(min(count, 100))
            ]
            if isinstance(count, int) and not isinstance(count, bool)
            else []
        )
        status["coverage_integrity_failure_details_truncated"] = bool(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > 100
        )
    return status


class _DreamToolDouble:
    def __init__(self, report: DreamReport, status=None, *, status_error=None):
        self.report = report
        self.status = _clean_dream_status() if status is None else status
        self.status_error = status_error
        self.status_calls = 0
        self.dream_calls: list[object] = []
        self.logged: list[tuple[str, str, str]] = []

    def dream(self, *, session_ids=None):
        self.dream_calls.append(session_ids)
        return self.report

    def dream_status(self):
        self.status_calls += 1
        if self.status_error is not None:
            raise self.status_error
        return self.status

    def open_session(self, _session_id):
        return None

    def log_message(self, session_id, role, content):
        self.logged.append((session_id, role, content))

    def close_session(self, _session_id):
        return None


class _FakeMCP:
    def __init__(self, events, *, run_failure=None):
        self.events = events
        self.run_failure = run_failure

    def tool(self):
        def register(func):
            self.events.append(f"tool:{func.__name__}")
            return func

        return register

    def run(self):
        self.events.append("run")
        if self.run_failure is not None:
            raise self.run_failure


def test_dream_report_completion_schema_is_an_exact_partition():
    classified = (
        *DREAM_REPORT_COUNT_FIELDS,
        *DREAM_REPORT_ERROR_FIELDS,
        *DREAM_REPORT_NULLABLE_COUNT_FIELDS,
        *DREAM_REPORT_TEXT_FIELDS,
        *DREAM_REPORT_BOOLEAN_GATE_FIELDS,
    )

    assert tuple(classified) == DREAM_REPORT_FIELD_NAMES
    assert len(classified) == len(set(classified))
    assert set(classified) == {field.name for field in fields(DreamReport)}


def test_mcp_entrypoint_eagerly_bootstraps_and_shuts_down(monkeypatch):
    events: list[str] = []
    fake = _FakeMCP(events)
    monkeypatch.setattr(srv, "_get_mcp", lambda: fake)
    monkeypatch.setattr(srv, "_get_hy", lambda: events.append("bootstrap"))
    monkeypatch.setattr(srv, "_shutdown_hy", lambda: events.append("shutdown"))

    srv.main()

    assert events[0] == "bootstrap"
    assert events[-2:] == ["run", "shutdown"]
    assert len([event for event in events if event.startswith("tool:")]) == 12


def test_mcp_entrypoint_preserves_primary_control_flow_over_shutdown(monkeypatch):
    events: list[str] = []
    primary = KeyboardInterrupt("mcp interrupted")
    cleanup = SystemExit("cleanup exit")
    monkeypatch.setattr(
        srv, "_get_mcp", lambda: _FakeMCP(events, run_failure=primary)
    )
    monkeypatch.setattr(srv, "_get_hy", lambda: object())
    monkeypatch.setattr(
        srv,
        "_shutdown_hy",
        lambda: (_ for _ in ()).throw(cleanup),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        srv.main()

    assert caught.value is primary
    notes = " ".join(getattr(primary, "__notes__", ()))
    assert "SystemExit" in notes
    assert "cleanup exit" not in notes


def test_hymem_log_writes_to_session(hy):
    srv.set_hy(hy)
    result = srv._do_log("test-session", "user", "hello world")
    assert result == "logged"
    rows = hy.conn.execute(
        "SELECT role, content FROM messages WHERE session_id='test-session'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "hello world"


def test_hymem_capture_logs_full_conversation(hy):
    triples = [{"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1}]
    hy.set_llm(make_routed_llm(triples, []))
    srv.set_hy(hy)

    messages = json.dumps([
        {"role": "user", "content": "We use uv for python tooling now."},
        {"role": "assistant", "content": "Got it, switching to uv from pip."},
        {"role": "system", "content": "noise"},
        {"role": "weird", "content": "ignored"},
        {"role": "user", "content": ""},
    ])
    result = srv._do_capture("cap-session", messages, dream=True)

    assert "logged 3 turns" in result
    assert "cap-session" in result
    rows = hy.conn.execute(
        "SELECT role FROM messages WHERE session_id='cap-session' ORDER BY id"
    ).fetchall()
    assert [r["role"] for r in rows] == ["user", "assistant", "system"]


def test_hymem_capture_invalid_json_returns_error(hy):
    srv.set_hy(hy)
    result = srv._do_capture("bad-session", "not json", dream=False)
    assert result.startswith("error:")

    result2 = srv._do_capture("bad-session", '{"not": "an array"}', dream=False)
    assert result2.startswith("error:")


def test_hymem_capture_skips_dream_when_false(hy):
    srv.set_hy(hy)
    messages = json.dumps([{"role": "user", "content": "hello"}])
    result = srv._do_capture("nodream", messages, dream=False)
    assert "logged 1 turns" in result
    assert "dreaming" not in result


def test_hymem_dream_returns_summary(hy):
    sid = "s-dream"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "I'll set up Docker for the local dev environment.")
    hy.log_message(
        sid, "user",
        "No, we use uv and system Python. Don't suggest Docker again.",
    )
    hy.close_session(sid)
    triples = [{"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1}]
    hy.set_llm(make_routed_llm(triples, []))

    srv.set_hy(hy)
    result = srv._do_dream()
    assert "dreaming incomplete" in result
    assert "report.digest_failures=1" in result
    assert "report.profile_failures=1" in result
    assert "dreaming complete" not in result
    assert "sessions" in result and "chunks" in result


@pytest.mark.parametrize(
    "field_name",
    (
        "pending_source_materialization",
        "pending_chunks",
        "pending_digests",
        "pending_profiles",
        "pending_facts",
        "quarantined_chunks",
        "quarantined_digests",
        "quarantined_profiles",
        "quarantined_facts",
        "quarantined_facts_malformed",
        "terminal_loss_chunks",
        "coverage_integrity_failures",
        "malformed_source_materialization",
        "malformed_digests",
        "malformed_profiles",
        "malformed_facts",
        "pending_aggregation",
    ),
)
def test_dream_tool_reports_every_store_health_blocker_as_incomplete(
    monkeypatch, field_name: str,
):
    fake = _DreamToolDouble(
        DreamReport(), _clean_dream_status(**{field_name: 1})
    )
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete" in result
    assert "finished cleanly" not in result
    assert f"status.{field_name}=1" in result
    assert fake.status_calls == 1


@pytest.mark.parametrize(
    "field_name",
    (
        "pending_new_consumer",
        "new_consumer_malformed",
        "new_consumer_quarantined",
        "terminal_loss_new_consumer",
        "coverage_integrity_new_consumer",
        "new_consumer_budget_exhausted",
        "new_consumer_failure_count",
        "new_consumer_error_count",
        "new_consumer_exception_count",
        "new_consumer_loss_count",
        "new_consumer_unhealthy",
        "new_consumer_stuck",
    ),
)
def test_dream_tool_rejects_unknown_status_health_fields(
    monkeypatch, field_name: str,
):
    fake = _DreamToolDouble(
        DreamReport(), _clean_dream_status(**{field_name: 0})
    )
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "finished cleanly" not in result
    assert f"status.health_schema:unknown={field_name}" in result


def test_dream_tool_bounds_unknown_status_schema_diagnostics(monkeypatch):
    status = _clean_dream_status()
    status.update({f"pending_future_{index:02d}": 0 for index in range(20)})
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "pending_future_00" in result
    assert "pending_future_05" in result
    assert "pending_future_06" not in result
    assert "...(+14)" in result
    assert len(result) < 600


def test_dream_tool_bounds_and_deduplicates_blocker_output(monkeypatch):
    status = _clean_dream_status()
    for field_name in tuple(status)[:20]:
        if field_name != "dream_status_schema":
            del status[field_name]
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "... (+" in result
    assert len(result) < 1_200
    blockers = result.split("; blockers: ", 1)[1].split("; ", 1)[0]
    listed = blockers.split(", ")
    assert len(listed) == len(set(listed))


def test_dream_tool_accepts_sanctioned_status_health_details(monkeypatch):
    status = _clean_dream_status(
        aggregation_total_caught_exceptions=7,
        aggregation_total_fusion_failures=3,
        aggregation_superseded_pending_configs=2,
        aggregation_last_failure_config_version=(
            _TEST_OLD_AGGREGATION_CONFIG_VERSION
        ),
        aggregation_last_failure_kind="exception",
        aggregation_last_failure_at="2026-09-06T10:00:00Z",
        aggregation_stale_pending_config_version=(
            _TEST_OLD_AGGREGATION_CONFIG_VERSION
        ),
        extraction_provider_attempt_budget=250,
    )
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming cycle finished cleanly" in result
    assert "incomplete/unverified" not in result


@pytest.mark.parametrize(
    "field_name",
    (
        "aggregation_active_build_attempts",
        "aggregation_active_caught_exceptions",
        "aggregation_active_fusion_failures",
    ),
)
def test_dream_tool_rejects_active_aggregation_diagnostics_without_pending(
    monkeypatch, field_name: str,
):
    overrides = {field_name: 1}
    if field_name == "aggregation_active_caught_exceptions":
        overrides["aggregation_total_caught_exceptions"] = 1
    if field_name == "aggregation_active_fusion_failures":
        overrides["aggregation_total_fusion_failures"] = 1
    fake = _DreamToolDouble(DreamReport(), _clean_dream_status(**overrides))
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "status.aggregation_active_diagnostics:inconsistent" in result
    assert "finished cleanly" not in result


@pytest.mark.parametrize(
    "field_name",
    (
        "aggregation_active_build_attempts",
        "aggregation_enabled",
        "aggregation_last_failure_kind",
        "extraction_provider_attempt_budget",
    ),
)
def test_dream_tool_requires_current_aggregation_diagnostics(
    monkeypatch, field_name: str,
):
    status = _clean_dream_status()
    del status[field_name]
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert f"status.{field_name}:missing_or_invalid" in result
    assert "finished cleanly" not in result


@pytest.mark.parametrize(
    ("failure_count", "detail_count", "truncated"),
    (
        (0, 1, False),
        (1, 0, False),
        (101, 100, False),
    ),
    ids=(
        "zero-count-with-detail",
        "positive-count-without-detail",
        "truncation-flag-mismatch",
    ),
)
def test_dream_tool_rejects_inconsistent_coverage_details(
    monkeypatch, failure_count: int, detail_count: int, truncated: bool,
):
    status = _clean_dream_status(
        coverage_integrity_failures=failure_count
    )
    template = _clean_dream_status(coverage_integrity_failures=1)[
        "coverage_integrity_failure_details"
    ][0]
    status["coverage_integrity_failure_details"] = [
        {**template, "session_id": f"inconsistent-{index}"}
        for index in range(detail_count)
    ]
    status["coverage_integrity_failure_details_truncated"] = truncated
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "status.coverage_integrity_details:inconsistent" in result
    assert "finished cleanly" not in result


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("future_failure_count", 0),
        ("future_budget_exhausted", False),
        ("future_boolean_gate", False),
        ("future_diagnostic_count", 0),
    ),
)
def test_dream_tool_rejects_every_dynamic_report_field(
    monkeypatch, field_name: str, value: object,
):
    report = DreamReport()
    setattr(report, field_name, value)
    fake = _DreamToolDouble(report)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "finished cleanly" not in result
    assert f"report.schema:extra={field_name}" in result


def test_dream_tool_rejects_missing_current_report_field(monkeypatch):
    report = DreamReport()
    del report.__dict__["triples_extracted"]
    fake = _DreamToolDouble(report)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "report.schema:missing=triples_extracted" in result
    assert "finished cleanly" not in result


def test_dream_tool_bounds_status_counts_in_completion_output(monkeypatch):
    status = _clean_dream_status(pending_chunks=1 << 63)
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "status.pending_chunks:missing_or_invalid" in result
    assert str(1 << 63) not in result
    assert "finished cleanly" not in result


def test_dream_tool_bounds_report_counts_before_rendering(monkeypatch):
    report = DreamReport(sessions_processed=10 ** 5000)
    fake = _DreamToolDouble(report)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "report.sessions_processed:missing_or_invalid" in result
    assert "; unknown sessions," in result
    assert "finished cleanly" not in result


def test_dream_tool_rejects_partial_aggregation_failure_metadata(monkeypatch):
    status = _clean_dream_status(
        aggregation_last_failure_config_version=(
            _TEST_OLD_AGGREGATION_CONFIG_VERSION
        ),
    )
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "status.aggregation_failure_metadata:inconsistent" in result
    assert "finished cleanly" not in result


def test_dream_tool_rejects_clean_enabled_aggregation_without_current_success(
    monkeypatch,
):
    status = _clean_dream_status(
        aggregation_enabled=True,
        aggregation_config_version=_TEST_AGGREGATION_CONFIG_VERSION,
    )
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "status.aggregation_success_metadata:stale" in result
    assert "finished cleanly" not in result


def test_dream_tool_rejects_malformed_aggregation_config_identity(monkeypatch):
    status = _clean_dream_status(
        aggregation_enabled=True,
        aggregation_config_version="not-a-current-config-identity",
        pending_aggregation=1,
    )
    # The helper installs its valid default for a pending aggregation; restore
    # the intentionally malformed identity after it has reconciled enablement.
    status["aggregation_config_version"] = "not-a-current-config-identity"
    fake = _DreamToolDouble(DreamReport(), status)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete/unverified" in result
    assert "status.aggregation_config_version:missing_or_invalid" in result
    assert "finished cleanly" not in result


@pytest.mark.parametrize(
    "field_name",
    (
        "chunk_extraction_failures",
        "coverage_integrity_failures",
        "digest_failures",
        "digest_quarantined",
        "profile_failures",
        "fact_failures",
        "aggregation_fusion_failures",
        "aggregation_build_exceptions",
    ),
)
def test_dream_tool_reports_every_run_error_as_incomplete(
    monkeypatch, field_name: str,
):
    report = DreamReport()
    setattr(report, field_name, 1)
    fake = _DreamToolDouble(report)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete" in result
    assert "finished cleanly" not in result
    assert f"report.{field_name}=1" in result
    assert fake.status_calls == 1


@pytest.mark.parametrize(
    "field_name",
    ("budget_exhausted", "extraction_provider_attempt_budget_exhausted"),
)
def test_dream_tool_reports_run_budget_exhaustion_as_incomplete(
    monkeypatch, field_name: str,
):
    report = DreamReport()
    setattr(report, field_name, True)
    fake = _DreamToolDouble(report)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming incomplete" in result
    assert f"report.{field_name}=true" in result
    assert "finished cleanly" not in result


@pytest.mark.parametrize(
    "malformed_status",
    (
        None,
        {"pending_chunks": 0},
        _clean_dream_status(dream_status_schema="hymem-dream-status-v1"),
        _clean_dream_status(pending_chunks=True),
        _clean_dream_status(in_progress=0),
    ),
    ids=(
        "not-a-dict", "missing-fields", "wrong-schema", "bool-count",
        "non-bool-lock",
    ),
)
def test_dream_tool_malformed_status_is_unverified(
    monkeypatch, malformed_status,
):
    fake = _DreamToolDouble(DreamReport(), malformed_status)
    # ``None`` normally requests the double's clean default; replace it after
    # construction so this case really exercises a non-dict response.
    fake.status = malformed_status
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "incomplete/unverified" in result
    assert "finished cleanly" not in result
    assert fake.status_calls == 1


def test_dream_tool_status_failure_is_unverified_without_exception_text(
    monkeypatch,
):
    fake = _DreamToolDouble(
        DreamReport(), status_error=RuntimeError("secret status detail")
    )
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "incomplete/unverified" in result
    assert "dream_status:unavailable" in result
    assert "secret status detail" not in result
    assert fake.status_calls == 1


def test_dream_tool_malformed_report_is_unverified(monkeypatch):
    report = DreamReport()
    report.digest_failures = True
    fake = _DreamToolDouble(report)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "incomplete/unverified" in result
    assert "report.digest_failures:missing_or_invalid" in result
    assert "finished cleanly" not in result


def test_dream_tool_lock_skip_and_observed_lock_are_distinct(monkeypatch):
    report = DreamReport(skipped_locked=True)
    fake = _DreamToolDouble(report, _clean_dream_status(in_progress=True))
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming skipped" in result
    assert "store-wide indexing is in progress" in result
    assert "finished cleanly" not in result
    assert fake.status_calls == 1


def test_dream_tool_post_run_lock_is_in_progress_not_clean(monkeypatch):
    fake = _DreamToolDouble(
        DreamReport(), _clean_dream_status(in_progress=True)
    )
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming run finished" in result
    assert "store-wide indexing is in progress" in result
    assert "finished cleanly" not in result


def test_capture_clean_target_never_claims_store_wide_completion(monkeypatch):
    fake = _DreamToolDouble(DreamReport())
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_capture(
        "target-session",
        json.dumps([{"role": "user", "content": "remember this"}]),
        dream=True,
    )

    assert "targeted dreaming cycle finished cleanly" in result
    assert "store-wide durable blockers were clear" in result
    assert "later arrivals may reopen work" in result
    assert "untargeted latent work was not audited" not in result
    assert "dreaming complete" not in result
    assert fake.dream_calls == [["target-session"]]
    assert fake.status_calls == 1


def test_dream_tool_clean_snapshot_describes_new_completions_without_ratio(
    monkeypatch,
):
    report = DreamReport(
        sessions_processed=2,
        chunks_seen=7,
        chunks_processed=0,
    )
    fake = _DreamToolDouble(report)
    monkeypatch.setattr(srv, "_get_hy", lambda: fake)

    result = srv._do_dream()

    assert "dreaming cycle finished cleanly" in result
    assert "store-wide durable blockers were clear" in result
    assert "later arrivals may reopen work" in result
    assert "0 chunks newly completed this run (7 seen)" in result
    assert result.count("0 triples") == 1
    assert "0/7" not in result
    assert fake.status_calls == 1


def _authoritative_empty_pipeline_llm() -> StubLLMClient:
    return StubLLMClient(
        fixtures={
            "You analyze one conversation session": json.dumps({
                "episodes": [], "summary": "", "procedures": [],
            }),
            "structured technical relationships": json.dumps({
                "triples": [], "markers": [], "complete": True,
            }),
        },
        default=None,
    )


def test_authoritative_empty_counts_as_new_completion_then_cache_is_noop(cfg):
    local = HyMem(
        replace(
            cfg,
            aggregation_nodes_enabled=False,
            profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=_authoritative_empty_pipeline_llm(),
    )
    try:
        local.log_message(
            "empty-completion",
            "user",
            "Neutral archival context " + ("x" * 120),
        )
        local.close_session("empty-completion")
        srv.set_hy(local)

        first_text = srv._do_dream()
        first_report = local.dream_status()["last_run"]

        assert first_report["chunks_processed"] == 1
        assert first_report["triples_extracted"] == 0
        assert first_report["markers_extracted"] == 0
        assert "1 chunks newly completed this run" in first_text
        assert "finished cleanly" in first_text

        second_text = srv._do_dream()
        second_report = local.dream_status()["last_run"]

        assert second_report["chunks_processed"] == 0
        assert "0 chunks newly completed this run" in second_text
        assert "finished cleanly" in second_text
    finally:
        local.close()


def test_invalid_extraction_json_stays_pending_and_mcp_reports_incomplete(cfg):
    llm = StubLLMClient(
        fixtures={
            "You analyze one conversation session": json.dumps({
                "episodes": [], "summary": "", "procedures": [],
            }),
            "structured technical relationships": "not-json",
        },
        default=None,
    )
    local = HyMem(
        replace(
            cfg,
            aggregation_nodes_enabled=False,
            profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=llm,
    )
    try:
        srv.set_hy(local)
        result = srv._do_capture(
            "invalid-extraction",
            json.dumps([{
                "role": "user",
                "content": "Neutral archival context " + ("x" * 120),
            }]),
            dream=True,
        )
        status = local.dream_status()

        assert status["pending_chunks"] == 1
        assert status["last_run"]["chunks_processed"] == 0
        assert "logged 1 turns" in result
        assert "dreaming incomplete" in result
        assert "status.pending_chunks=1" in result
        assert "report.chunk_extraction_failures=1" in result
        assert "finished cleanly" not in result
    finally:
        local.close()


def test_exact_chunk_budget_reports_remainder_then_clean_drain(cfg):
    local = HyMem(
        replace(
            cfg,
            dream_budget=1,
            aggregation_nodes_enabled=False,
            profile_extraction_enabled=False,
            facts_extraction_enabled=False,
        ),
        llm=_authoritative_empty_pipeline_llm(),
    )
    try:
        for index in range(2):
            session_id = f"budget-session-{index}"
            local.log_message(
                session_id,
                "user",
                f"Neutral archival context {index} " + ("x" * 120),
            )
            local.close_session(session_id)
        srv.set_hy(local)

        first_text = srv._do_dream()
        first_status = local.dream_status()

        assert first_status["pending_chunks"] == 1
        assert first_status["last_run"]["chunks_processed"] == 1
        assert "report.budget_exhausted=true" in first_text
        assert "status.pending_chunks=1" in first_text
        assert "dreaming incomplete" in first_text

        second_text = srv._do_dream()
        second_status = local.dream_status()

        assert second_status["pending_chunks"] == 0
        assert second_status["last_run"]["chunks_processed"] == 1
        assert second_status["last_run"]["skipped_locked"] == 0
        assert "finished cleanly" in second_text
        assert "budget_exhausted" not in second_text
    finally:
        local.close()


def test_hymem_augment_returns_context_string(hy):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('local_dev', 'uses', 'uv', 3, 0, CURRENT_TIMESTAMP)"
    )
    srv.set_hy(hy)
    result = srv._do_augment("tell me about uv tooling")
    assert "uv" in result
    assert "Structured knowledge" in result


def test_hymem_augment_returns_empty_when_no_context(hy):
    srv.set_hy(hy)
    result = srv._do_augment("totally unknown query about nothing")
    assert result == ""


def test_hymem_ask_returns_synthesized_answer(hy, stub_llm):
    sid = "ask-tool"
    hy.open_session(sid)
    hy.log_message(sid, "user", "My favorite database is duckdb.")
    # Key the fixture on the retrieved fact: it only matches once the rendered
    # memory context (carrying the logged turn) reaches the synthesis prompt.
    stub_llm.fixtures["duckdb"] = "Your favorite database is duckdb."

    srv.set_hy(hy)
    result = srv._do_ask("what database do I like?")
    assert result == "Your favorite database is duckdb."


def test_hymem_profile_returns_user_and_memory_md(hy):
    hy.config.user_md_path.write_text("# Behavioral Profile\nPrefers terse code.", encoding="utf-8")
    hy.config.memory_md_path.write_text("# Project Insights\nUses uv for tooling.", encoding="utf-8")
    srv.set_hy(hy)
    result = srv._do_profile()
    assert "USER PROFILE" in result
    assert "PROJECT INSIGHTS" in result
    assert "Prefers terse code." in result
    assert "Uses uv for tooling." in result


def test_hymem_profile_uses_authoritative_empty_projection_when_files_missing(hy):
    srv.set_hy(hy)
    hy.config.user_md_path.unlink(missing_ok=True)
    hy.config.memory_md_path.unlink(missing_ok=True)
    assert not hy.config.user_md_path.exists()
    assert not hy.config.memory_md_path.exists()
    result = srv._do_profile()
    assert "=== USER PROFILE ===" in result
    assert "_No behavioral signals collected yet._" in result
    assert "PROJECT INSIGHTS" not in result
    assert not hy.config.user_md_path.exists()


def test_hymem_profile_handles_no_authoritative_or_file_storage(monkeypatch, hy):
    import hymem.dreaming.phase2 as phase2

    srv.set_hy(hy)
    hy.config.user_md_path.unlink(missing_ok=True)
    hy.config.memory_md_path.unlink(missing_ok=True)
    monkeypatch.setattr(phase2, "authoritative_user_markdown", lambda *_args: "")

    assert srv._do_profile() == "No profile or insights available yet."
    assert not hy.config.user_md_path.exists()
    assert not hy.config.memory_md_path.exists()


def test_hymem_profile_handles_only_user_md(hy):
    hy.config.user_md_path.write_text("# Behavioral Profile\nPrefers terse code.", encoding="utf-8")
    srv.set_hy(hy)
    result = srv._do_profile()
    assert "USER PROFILE" in result
    assert "PROJECT INSIGHTS" not in result


def _seed_root_digest(hy, *, title="User digest", summary="Works on HyMem."):
    """Build a proof-valid root before exercising the public MCP surface."""
    from hymem.dreaming.aggregate import build_aggregation_nodes
    from tests.test_aggregation_provenance import _seed_native_episode

    _seed_native_episode(
        hy.conn, "root-mcp-a", title="Project", summary=summary,
        entity="root-mcp",
    )
    _seed_native_episode(
        hy.conn, "root-mcp-b", title="Project", summary=summary,
        entity="root-mcp",
    )
    fuse = StubLLMClient(
        fixtures={
            "fuse several related episodes": json.dumps({
                "title": "Project", "summary": summary,
            }),
            "standing digest of everything known": json.dumps({
                "title": title, "summary": summary,
            }),
        },
        default="[]",
    )
    # Public digest reads are producer-generation scoped.  Build with the
    # same currently configured producer that the HyMem facade will verify.
    hy.set_llm(fuse)
    build_aggregation_nodes(hy.conn, hy.config, fuse)


def test_hymem_digest_explains_when_absent(hy):
    srv.set_hy(hy)
    result = srv._do_digest()
    assert "No digest available yet" in result


def test_hymem_digest_returns_context_block(hy):
    srv.set_hy(hy)
    _seed_root_digest(hy)
    result = srv._do_digest()
    assert result.startswith("## User digest")
    assert "Works on HyMem." in result
    # The staleness footer: coverage ratio + generated_at are always present.
    assert "Memory digest covering 2 of" in result
    assert "generated" in result


def test_hymem_alias_registers_mapping(hy):
    srv.set_hy(hy)
    result = srv._do_alias("MedFlow", "med_flow")
    assert "alias registered" in result
    row = hy.conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias='med_flow'"
    ).fetchone()
    assert row["canonical"] == "med_flow"


def test_hymem_retract_returns_no_match_for_missing_edge(hy):
    srv.set_hy(hy)
    result = srv._do_retract("ghost", "uses", "nothing")
    assert result == "no matching active edge found"


def test_hymem_retract_succeeds_for_existing_edge(hy):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('med_flow', 'depends_on', 'redis', 3, 0, CURRENT_TIMESTAMP)"
    )
    srv.set_hy(hy)
    result = srv._do_retract("med_flow", "depends_on", "redis")
    assert result == "retracted"


def test_hymem_add_and_list_rules(hy):
    srv.set_hy(hy)
    assert srv._do_add_rule("never suggest docker").startswith("rule #")
    srv._do_add_rule("prefer pytest", "contextual", "pytest, tests")
    listed = srv._do_list_rules()
    assert "never suggest docker" in listed
    assert "[always_on]" in listed
    assert "contextual(" in listed and "pytest" in listed


def test_hymem_add_rule_rejects_empty(hy):
    srv.set_hy(hy)
    assert srv._do_add_rule("   ").startswith("error")
