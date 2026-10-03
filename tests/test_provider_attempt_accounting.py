from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

from hymem import HyMem, HyMemConfig
from hymem.extraction.chunk import extract_chunk
from hymem.extraction.llm import LLMRequest, measure_provider_attempts


class _Tracker:
    def __init__(self, attempts: object) -> None:
        self.attempts = attempts


class _RaisingTracker:
    @property
    def attempts(self) -> int:
        raise RuntimeError("broken tracker property")


class _InterruptingTracker:
    @property
    def attempts(self) -> int:
        raise KeyboardInterrupt("tracker interrupted")


class _Scope:
    def __init__(
        self,
        tracker: object,
        *,
        exit_error: BaseException | None = None,
        suppress: bool = False,
    ) -> None:
        self.tracker = tracker
        self.exit_error = exit_error
        self.suppress = suppress

    def __enter__(self) -> object:
        return self.tracker

    def __exit__(self, *_exc_info: object) -> bool:
        if self.exit_error is not None:
            raise self.exit_error
        return self.suppress


class _ScopedClient:
    def __init__(
        self,
        attempts: object,
        *,
        exit_error: BaseException | None = None,
        suppress: bool = False,
    ) -> None:
        self.attempts = attempts
        self.exit_error = exit_error
        self.suppress = suppress

    def track_provider_attempts(self) -> _Scope:
        if self.attempts == "raising-property":
            tracker: object = _RaisingTracker()
        elif self.attempts == "interrupting-property":
            tracker = _InterruptingTracker()
        else:
            tracker = _Tracker(self.attempts)
        return _Scope(
            tracker,
            exit_error=self.exit_error,
            suppress=self.suppress,
        )

    def complete(self, _request: LLMRequest) -> str:
        return "ok"


@pytest.mark.parametrize(
    "attempts",
    [0, -1, True, False, 1.5, "3", None, "raising-property"],
)
def test_measure_provider_attempts_floors_invalid_scoped_values(attempts: object):
    client = _ScopedClient(attempts)

    with measure_provider_attempts(client) as measurement:
        assert client.complete(LLMRequest(system="s", user="u")) == "ok"

    assert measurement.attempts == 1
    assert measurement.exact is False


def test_measure_provider_attempts_keeps_valid_scoped_retry_total_exact():
    client = _ScopedClient(3)

    with measure_provider_attempts(client) as measurement:
        client.complete(LLMRequest(system="s", user="u"))

    assert measurement.attempts == 3
    assert measurement.exact is True


class _InvalidScopeWithCumulativeClient(_ScopedClient):
    def __init__(self) -> None:
        super().__init__(0)
        self.request_attempts = 0

    def complete(self, _request: LLMRequest) -> str:
        self.request_attempts += 3
        return "ok"


def test_invalid_scope_can_fall_back_to_positive_cumulative_delta():
    client = _InvalidScopeWithCumulativeClient()

    with measure_provider_attempts(client) as measurement:
        client.complete(LLMRequest(system="s", user="u"))

    assert measurement.attempts == 3
    assert measurement.exact is False


def test_scope_exit_failure_does_not_turn_success_into_provider_failure():
    client = _ScopedClient(0, exit_error=RuntimeError("telemetry exit failed"))

    with measure_provider_attempts(client) as measurement:
        value = client.complete(LLMRequest(system="s", user="u"))

    assert value == "ok"
    assert measurement.attempts == 1
    assert measurement.exact is False


def test_scope_exit_failure_retains_positive_attempt_floor_as_inexact():
    client = _ScopedClient(3, exit_error=RuntimeError("telemetry exit failed"))

    with measure_provider_attempts(client) as measurement:
        client.complete(LLMRequest(system="s", user="u"))

    assert measurement.attempts == 3
    assert measurement.exact is False


def test_scope_cannot_suppress_or_replace_original_provider_exception():
    provider_error = LookupError("provider failed")
    telemetry_error = KeyboardInterrupt("telemetry exit failed")
    client = _ScopedClient(0, exit_error=telemetry_error, suppress=True)
    measurement = None

    with pytest.raises(LookupError) as caught:
        with measure_provider_attempts(client) as measurement:
            raise provider_error

    assert caught.value is provider_error
    assert measurement is not None
    assert measurement.attempts == 1
    assert measurement.exact is False


def test_tracker_getter_cannot_replace_original_provider_exception():
    provider_error = LookupError("provider failed")
    client = _ScopedClient("interrupting-property")
    measurement = None

    with pytest.raises(LookupError) as caught:
        with measure_provider_attempts(client) as measurement:
            raise provider_error

    assert caught.value is provider_error
    assert measurement is not None
    assert measurement.attempts == 1
    assert measurement.exact is False


class _ZeroScopedExtractionClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[LLMRequest] = []
        self.fail = fail

    @contextmanager
    def track_provider_attempts(self):
        yield _Tracker(0)

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("provider details must not escape")
        if "Return the JSON object now" in request.user:
            return json.dumps({
                "episodes": [],
                "summary": "tail complete",
                "procedures": [],
            })
        return json.dumps({
            "triples": [],
            "markers": [],
            "complete": True,
        })


def test_extract_chunk_success_never_reports_fewer_attempts_than_calls():
    client = _ZeroScopedExtractionClient()

    result = extract_chunk(client, "No durable relationship is stated.")

    assert result.failed is False
    assert result.completion_calls == len(client.calls) == 2
    assert result.provider_attempts == result.completion_calls


def test_extract_chunk_failure_charges_zero_tracker_without_raw_error():
    client = _ZeroScopedExtractionClient(fail=True)

    result = extract_chunk(client, "The provider call fails.")

    assert result.failed is True
    assert result.failure_reason == "call_failure"
    assert "provider:call_failed" in result.failure_details
    assert result.completion_calls == len(client.calls) == 1
    assert result.provider_attempts == result.completion_calls
    assert "provider details" not in repr(result)


def _seed_closed_sessions(hy: HyMem, *session_ids: str) -> None:
    for session_id in session_ids:
        hy.open_session(session_id)
        hy.log_message(session_id, "user", f"Substantive input for {session_id}.")
        hy.close_session(session_id)


def test_dream_budget_and_report_charge_zero_scoped_tracker(
    cfg: HyMemConfig,
):
    client = _ZeroScopedExtractionClient()
    config = replace(
        cfg,
        dream_budget=50,
        dream_baseline_budget=0,
        dream_extraction_provider_attempt_budget=1,
        salience_min_chars=1,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
    )
    hy = HyMem(config, llm=client)
    try:
        _seed_closed_sessions(hy, "a-first", "z-later")

        report = hy.dream()

        assert report.chunk_extraction_completion_calls == 2
        assert report.chunk_extraction_provider_attempts == 2
        assert (
            report.chunk_extraction_provider_attempts
            >= report.chunk_extraction_completion_calls
        )
        assert report.extraction_provider_attempt_budget_exhausted is True
        assert hy.dream_status()["pending_chunks"] == 1
        persisted = hy.conn.execute(
            "SELECT chunk_extraction_completion_calls,"
            "chunk_extraction_provider_attempts "
            "FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert tuple(persisted) == (2, 2)
    finally:
        hy.close()
