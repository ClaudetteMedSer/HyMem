"""Failure origin is independent of the last attempted task or reason prefix.

Scripted responses test attribution and holding behavior, not model accuracy.
"""
from contextlib import closing
from dataclasses import replace
import json

import pytest

from hymem import HyMem
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import digest
from tests.test_digest_decision_diagnosis_root import (
    FIXED, INITIAL, RootClient, assert_held, extract,
)
from tests.test_digest_summary_contract import _config
from tests.test_digest_summary_verification import _SummaryClient, _extract, _seed


class StageClient(RootClient):
    def __init__(self, *, override_stage=None, raw=None, **kwargs):
        super().__init__(**kwargs)
        self.override_stage, self.raw = override_stage, raw

    def complete(self, request):
        result = super().complete(request)
        return self.raw if self.roles[-1] == self.override_stage else result


def assert_failure(result, client, *, stage, reason, calls):
    assert_held(result)
    assert result.failure_stage == stage
    assert result.failure_reason == reason
    assert len(client.requests) == calls
    assert result.episode_input_items == result.episode_rejected_items == 1
    assert result.procedure_input_items == result.procedure_rejected_items == 1


@pytest.mark.parametrize("stage,raw,reason,calls", [
    ("fidelity_verification", "not JSON", "fidelity_parse_failure", 2),
    ("fidelity_verification", "{}", "fidelity_shape_failure", 2),
    ("summary_diagnosis", "not JSON", "summary_diagnosis_parse_failure", 3),
    ("summary_diagnosis", '{"issues":[]}', "summary_diagnosis_unactionable", 3),
    ("summary_content_recovery", "not JSON", "summary_content_recovery_parse_failure", 4),
    ("summary_content_recovery", "{}", "summary_content_recovery_shape_failure", 4),
    ("summary_content_recovery", '{"summary":""}', "summary_content_recovery_summary_validation_failure", 4),
    ("summary_content_recovery", json.dumps({"summary": "x" * 501}), "summary_content_recovery_summary_output_cap", 4),
    ("fidelity_reverification", "not JSON", "fidelity_parse_failure", 5),
    ("fidelity_reverification", "{}", "fidelity_shape_failure", 5),
    ("format_adjudication", "not JSON", "format_adjudication_parse_failure", 6),
    ("format_adjudication", "{}", "format_adjudication_shape_failure", 6),
])
def test_normal_response_rejection_keeps_its_actual_stage(cfg, caplog, stage, raw, reason, calls):
    client = StageClient(override_stage=stage, raw=raw)
    assert_failure(extract(cfg, client), client, stage=stage, reason=reason, calls=calls)
    warning = [record.getMessage() for record in caplog.records
               if record.getMessage().startswith("digest.fidelity_failure ")]
    assert warning == [f"digest.fidelity_failure session_id=root-separated stage={stage} reason={reason}"]


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
@pytest.mark.parametrize("stage,calls", [("fidelity_reverification", 5), ("format_adjudication", 6)])
def test_late_semantic_or_format_veto_reports_its_origin(cfg, verdict, stage, calls):
    client = RootClient(**{"second" if stage == "fidelity_reverification" else "format_verdict": verdict})
    reason = ("summary_content_" if stage == "fidelity_reverification" else "summary_format_") + verdict
    assert_failure(extract(cfg, client), client, stage=stage, reason=reason, calls=calls)


@pytest.mark.parametrize("stage,constant,reason,calls", [
    ("summary_diagnosis", "_DIGEST_SUMMARY_DIAGNOSIS_MAX_INPUT_CHARS", "summary_diagnosis_input_cap", 2),
    ("summary_content_recovery", "_DIGEST_SUMMARY_CONTENT_RECOVERY_MAX_INPUT_CHARS", "summary_content_recovery_input_cap", 3),
    ("format_adjudication", "_DIGEST_FORMAT_ADJUDICATION_MAX_INPUT_CHARS", "format_adjudication_input_cap", 5),
])
def test_late_input_caps_belong_to_the_undispatched_task(cfg, monkeypatch, stage, constant, reason, calls):
    monkeypatch.setattr(digest, constant, 1)
    client = RootClient()
    assert_failure(extract(cfg, client), client, stage=stage, reason=reason, calls=calls)
    assert stage not in client.roles


@pytest.mark.parametrize("invocation,stage,calls", [(1, "fidelity_verification", 1), (2, "fidelity_reverification", 4)])
def test_initial_and_reverification_input_caps_are_distinguished(cfg, monkeypatch, invocation, stage, calls):
    original = digest._encode_digest_fidelity_payload
    seen = 0

    def capped(*args, **kwargs):
        nonlocal seen
        seen += 1
        return None if seen == invocation else original(*args, **kwargs)

    monkeypatch.setattr(digest, "_encode_digest_fidelity_payload", capped)
    client = RootClient()
    assert_failure(extract(cfg, client), client, stage=stage, reason="fidelity_input_cap", calls=calls)
    assert stage not in client.roles


def test_assembled_repair_validation_failure_is_recovery_not_initial_verification(cfg, monkeypatch):
    original = digest._validate_digest_response
    seen = 0

    def reject_assembled(*args, **kwargs):
        nonlocal seen
        seen += 1
        result = original(*args, **kwargs)
        return replace(result, parse_failed=True, failure_reason="summary_validation_failure") if seen == 2 else result

    monkeypatch.setattr(digest, "_validate_digest_response", reject_assembled)
    client = RootClient()
    assert_failure(extract(cfg, client), client, stage="summary_content_recovery",
                   reason="summary_content_recovery_summary_validation_failure", calls=4)


@pytest.mark.parametrize("repair", [INITIAL, " \n" + INITIAL + "  "])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_unchanged_repair_preserves_the_original_veto_origin(cfg, repair, verdict):
    client = RootClient(repair=repair, first=verdict)
    assert_failure(extract(cfg, client), client, stage="fidelity_verification",
                   reason="summary_content_" + verdict, calls=4)
    assert client.roles[-1] == "summary_content_recovery"


def test_noop_prior_cap_remains_initial_verification_without_extra_calls(cfg):
    client = _SummaryClient("", with_items=False)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Acknowledged.")
        result = _extract(hy, client, prior="p" * 501)
        assert_held(result)
        assert result.failure_stage == "fidelity_verification"
        assert result.failure_reason == "summary_noop_prior_output_cap"
        assert len(client.calls) == 1


@pytest.mark.parametrize("stage,helper,invocation,calls", [
    ("summary_content_recovery", "_digest_summary_content_recovery_payload", 1, 3),
    ("summary_content_recovery", "_validate_digest_response", 2, 4),
    ("fidelity_reverification", "_encode_digest_fidelity_payload", 2, 4),
    ("format_adjudication", "_digest_format_adjudication_payload", 1, 5),
    ("format_adjudication", "_encode_digest_format_adjudication_payload", 1, 5),
])
@pytest.mark.parametrize("kind", ["runtime", "deadline", "keyboard", "control", "nested"])
def test_encode_and_assembled_validation_exceptions_keep_substage(cfg, monkeypatch, stage, helper, invocation, calls, kind):
    original = getattr(digest, helper)
    error = {
        "runtime": RuntimeError("synthetic local processing failure"),
        "deadline": DeadlineExceeded("synthetic deadline"),
        "keyboard": KeyboardInterrupt(),
        "control": BaseException("synthetic control flow"),
        "nested": digest.DigestCompletionError("specific_inner_stage"),
    }[kind]
    seen = 0

    def fail_selected(*args, **kwargs):
        nonlocal seen
        seen += 1
        if seen == invocation:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(digest, helper, fail_selected)
    client = RootClient()
    expected = digest.DigestCompletionError if kind == "runtime" else type(error)
    with pytest.raises(expected) as caught:
        extract(cfg, client)
    if kind == "runtime":
        assert caught.value.failure_stage == stage
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error
        if kind == "nested":
            assert caught.value.failure_stage == "specific_inner_stage"
    assert len(client.requests) == calls


def test_success_remains_six_calls_and_has_no_failure_metadata(cfg):
    client = RootClient()
    result = extract(cfg, client)
    assert not result.parse_failed and result.summary == FIXED
    assert result.failure_reason is result.failure_stage is None
    assert client.roles == ["primary", "fidelity_verification", "summary_diagnosis",
                            "summary_content_recovery", "fidelity_reverification", "format_adjudication"]
