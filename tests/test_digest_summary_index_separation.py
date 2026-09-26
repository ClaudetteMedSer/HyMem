"""Summary rejection may not erase independently proven digest item output."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json

import pytest

from hymem import HyMem, HyMemConfig
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import digest as mod
from hymem.dreaming.lossless import coverage_chunk_id, materialize_message_coverage
from hymem.extraction.llm import StubLLMClient
from tests.test_digest_bounded_summary_repair import SequenceLLM, extract, payload, quiet, source


GOOD = "Built the image and deployed the service."


def separated(source, llm, **kwargs):
    return extract(source, llm, separate_summary=True, **kwargs)


def assert_item_authority(source, result):
    reference = extract(source, SequenceLLM(payload(source, GOOD)))
    assert not result.parse_failed
    assert result.failure_reason is result.failure_stage is None
    assert result.episodes == reference.episodes
    assert result.procedures == reference.procedures
    for key in (
        "covered_message_id", "start_message_id", "next_message_offset",
        "partial_message_id", "end_message_id", "caught_up", "source_sha256",
        "episode_input_items", "episode_rejected_items", "procedure_input_items",
        "procedure_rejected_items",
    ):
        assert getattr(result, key) == getattr(reference, key)
    assert result.summary is None


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("reply,reason", [
    ({"summary": "x" * 501}, "summary_output_cap"),
    ({"clauses": []}, "shape_failure"),
    ({"clauses": [["x" * 300], ["y" * 200]]}, "summary_output_cap"),
    ({"summary": "tiny"}, "summary_validation_failure"),
    ({"summary": None}, "summary_shape_failure"),
    ({"summary": GOOD, "episodes": []}, "shape_failure"),
    ("No summary is possible.", "parse_failure"),
    ('{"summary":"unfinished', "output_truncated"),
    ('{"summary":"first","summary":"second"}', "parse_failure"),
])
def test_returned_repair_rejection_preserves_complete_primary_items(source, granular, reply, reason):
    original = payload(source, "x" * 644)
    saved = copy.deepcopy(original)
    llm = SequenceLLM(original, reply)
    changes = source[0].conn.total_changes
    result = separated(source, llm, granular=granular, max_episodes=8)
    assert_item_authority(source, result)
    assert result.summary_failure_reason == reason
    assert len(llm.calls) == 2 and not llm.responses
    assert original == saved
    assert source[0].conn.total_changes == changes
    assert json.loads(llm.calls[1].user) == {"original_generation_input": llm.calls[0].user}


@pytest.mark.parametrize("summary,reason", [
    (None, "summary_shape_failure"),
    (12, "summary_shape_failure"),
    ({"text": GOOD}, "summary_shape_failure"),
    ([GOOD], "summary_shape_failure"),
    ("tiny", "summary_validation_failure"),
    ('"            "', "summary_validation_failure"),
    ("'tiny'", "summary_validation_failure"),
])
def test_independent_item_validation_allows_only_summary_field_rejection(source, summary, reason):
    llm = SequenceLLM(payload(source, summary))
    result = separated(source, llm)
    assert_item_authority(source, result)
    assert result.summary_failure_reason == reason
    assert len(llm.calls) == 1


@pytest.mark.parametrize("bad_summary", [None, "tiny", "x" * 501])
@pytest.mark.parametrize("fault,reason", [
    ("episode_citation", "episode_validation_failure"),
    ("procedure_citation", "procedure_validation_failure"),
    ("procedure_description", "procedure_validation_failure"),
    ("procedure_extra_key", "procedure_validation_failure"),
    ("episode_extra_key", "episode_validation_failure"),
    ("episode_cap", "episode_output_cap"),
    ("episodes_not_array", "shape_failure"),
    ("procedures_not_array", "shape_failure"),
    ("extra_top_key", "shape_failure"),
    ("missing_summary", "shape_failure"),
])
def test_bad_primary_items_or_envelope_never_gain_authority(source, bad_summary, fault, reason):
    original = payload(source, bad_summary)
    if fault == "episode_citation":
        original["episodes"][0]["chunk_ids"] = ["unknown"]
    elif fault == "procedure_citation":
        original["procedures"][0]["chunk_ids"] = ["unknown"]
    elif fault == "procedure_description":
        original["procedures"][0]["description"] = "x" * 501
    elif fault == "procedure_extra_key":
        original["procedures"][0]["extra"] = True
    elif fault == "episode_extra_key":
        original["episodes"][0]["extra"] = True
    elif fault == "episodes_not_array":
        original["episodes"] = {}
    elif fault == "procedures_not_array":
        original["procedures"] = None
    elif fault == "extra_top_key":
        original["extra"] = True
    elif fault == "missing_summary":
        del original["summary"]
    llm = SequenceLLM(original)
    result = separated(source, llm, granular=True, max_episodes=0 if fault == "episode_cap" else 8)
    assert result.parse_failed and result.failure_reason == reason
    assert result.summary_failure_reason is None
    assert result.episodes.items == result.procedures.items == []
    assert result.summary is result.source_sha256 is result.covered_message_id is None
    assert not result.caught_up and len(llm.calls) == 1


@pytest.mark.parametrize("bad", [
    "Not JSON", 'prefix {"episodes":[],"summary":"","procedures":[]}',
    '{"episodes":[],"summary":"unfinished',
    '{"episodes":[],"summary":"","summary":"","procedures":[]}', [], None,
])
def test_primary_parse_or_top_shape_failure_remains_fatal(source, bad):
    result = separated(source, SequenceLLM(bad))
    assert result.parse_failed
    assert result.summary_failure_reason is None
    assert result.source_sha256 is result.covered_message_id is None
    assert not result.episodes.items and not result.procedures.items


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("summary", [GOOD, "x" * 501, None, "tiny", ""])
def test_prior_gap_cannot_be_cleared_by_later_summary_or_waste_repair(source, granular, summary):
    llm = SequenceLLM(payload(source, summary))
    result = separated(source, llm, granular=granular, prior_summary_is_stale=True)
    assert_item_authority(source, result)
    assert result.summary_failure_reason == "prior_summary_gap"
    assert len(llm.calls) == 1
    for text in (llm.calls[0].system, llm.calls[0].user):
        assert "prior automatic summary is stale" in text
        assert "intervening indexed material is not represented" in text
        assert "Do not infer the missing material" in text
        assert "cannot establish a fresh rolling summary" in text
    assert "Configured the earlier staging project." in llm.calls[0].user
    assert "Earlier boundary: build the image, then" in llm.calls[0].user
    assert "deploy the service and verify its health." in llm.calls[0].user


def test_prior_gap_does_not_mask_invalid_primary_items(source):
    original = payload(source)
    original["procedures"][0]["chunk_ids"] = ["unknown"]
    result = separated(source, SequenceLLM(original), prior_summary_is_stale=True)
    assert result.parse_failed and result.failure_reason == "procedure_validation_failure"
    assert result.summary_failure_reason is None and result.source_sha256 is None


def test_strict_mode_rejects_stale_context_without_paid_call(source):
    llm = SequenceLLM(payload(source))
    with pytest.raises(ValueError, match="requires separated publication"):
        extract(source, llm, prior_summary_is_stale=True)
    assert llm.calls == []


@pytest.mark.parametrize("flag", ["separate_summary", "prior_summary_is_stale"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", "", [], {}])
def test_mode_flags_require_exact_bools_before_calls(source, flag, value):
    llm = SequenceLLM(payload(source))
    with pytest.raises(TypeError, match="must be bool"):
        extract(source, llm, **{flag: value})
    assert llm.calls == []


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("failure", [
    RuntimeError("LLM transport identity changed"),
    RuntimeError("LLM execution identity is unavailable"),
    RuntimeError("provider unavailable"),
    ValueError("accounting failure"),
    DeadlineExceeded("deadline"), KeyboardInterrupt(), SystemExit(2),
])
def test_completion_identity_transport_and_control_flow_errors_remain_fatal(source, repair, failure):
    responses = [payload(source, "x" * 501), failure] if repair else [failure]
    llm = SequenceLLM(*responses)
    before = source[0].conn.total_changes
    expected = mod.DigestCompletionError if isinstance(failure, Exception) else type(failure)
    with pytest.raises(expected) as raised:
        separated(source, llm)
    if isinstance(failure, Exception):
        assert raised.value.__cause__ is failure
        assert raised.value.failure_stage == ("summary_compaction" if repair else "primary")
    else:
        assert raised.value is failure
    assert len(llm.calls) == (2 if repair else 1)
    assert source[0].conn.total_changes == before


@pytest.mark.parametrize("repair", [False, True])
def test_valid_summaries_keep_legacy_output_and_wire(source, repair):
    responses = [payload(source, "x" * 501), {"summary": GOOD}] if repair else [payload(source, GOOD)]
    strict_llm = SequenceLLM(*copy.deepcopy(responses))
    separate_llm = SequenceLLM(*copy.deepcopy(responses))
    strict = extract(source, strict_llm)
    result = separated(source, separate_llm)
    assert asdict(result) == asdict(strict)
    assert result.summary_failure_reason is None and result.summary == GOOD
    assert [asdict(r) for r in strict_llm.calls] == [asdict(r) for r in separate_llm.calls]


def test_strict_default_still_rejects_overlong_failed_repair(source):
    llm = SequenceLLM(payload(source, "x" * 644), {"summary": "x" * 635})
    result = extract(source, llm)
    assert result.parse_failed and result.failure_reason == "summary_output_cap"
    assert result.summary_failure_reason is None
    assert result.source_sha256 is result.covered_message_id is None
    assert not result.episodes.items and not result.procedures.items


def test_degraded_log_discloses_no_source_summary_or_provider_text(source, caplog):
    llm = SequenceLLM(payload(source, "PRIVATE_DRAFT_" * 60), {"summary": "PRIVATE_BAD" * 60})
    result = separated(source, llm)
    assert result.summary_failure_reason == "summary_output_cap"
    assert "summary_coverage_advanced=0" in caplog.text
    for secret in ("PRIVATE_DRAFT_", "PRIVATE_BAD", source[1], "Earlier boundary", "deploy the service"):
        assert secret not in caplog.text


@pytest.mark.parametrize("prune_raw", [False, True])
def test_partial_source_proof_and_offsets_survive_degradation_and_retention(tmp_path, prune_raw):
    hy = HyMem(quiet(HyMemConfig(root=tmp_path)), llm=StubLLMClient(default="[]"))
    try:
        sid = "partial-source"
        mid = hy.log_message(sid, "user", "Build image and deploy service. " * 100)
        hy.close_session(sid)
        materialize_message_coverage(hy.conn, sid)
        if prune_raw:
            hy.conn.execute("DELETE FROM messages WHERE id=?", (mid,))
        original = {
            "episodes": [{"title": "Deploy service", "summary": "Build and deploy.",
                          "outcome": "informational", "key_entities": ["service"],
                          "chunk_ids": [coverage_chunk_id(sid, mid)]}],
            "summary": GOOD, "procedures": [],
        }
        kwargs = {"max_tokens": 3072, "max_chars": 400}
        reference = mod.extract_session_digest(hy.conn, sid, SequenceLLM(original), **kwargs)
        degraded = mod.extract_session_digest(
            hy.conn, sid, SequenceLLM({**original, "summary": None}),
            separate_summary=True, **kwargs,
        )
        assert reference.partial_message_id == mid and not reference.caught_up
        assert 0 < reference.next_message_offset < 3100
        assert degraded.source_sha256 == reference.source_sha256
        assert degraded.partial_message_id == reference.partial_message_id
        assert degraded.next_message_offset == reference.next_message_offset
        assert degraded.episodes == reference.episodes
        assert degraded.summary_failure_reason == "summary_shape_failure"
        resume = dict(kwargs, partial_message_id=mid,
                      since_message_offset=degraded.next_message_offset)
        strict_next = mod.extract_session_digest(hy.conn, sid, SequenceLLM(original), **resume)
        gap_next = mod.extract_session_digest(
            hy.conn, sid, SequenceLLM(original), separate_summary=True,
            prior_summary_is_stale=True, **resume,
        )
        assert gap_next.source_sha256 == strict_next.source_sha256
        assert gap_next.next_message_offset == strict_next.next_message_offset
        assert gap_next.next_message_offset > degraded.next_message_offset
        assert gap_next.summary_failure_reason == "prior_summary_gap"
        assert gap_next.summary is None and not gap_next.parse_failed
    finally:
        hy.close()


@pytest.mark.parametrize("target", ["_build_digest_summary_repair_request", "_validate_digest_summary_repair"])
def test_recovery_code_fault_is_fatal_not_summary_degradation(source, monkeypatch, target):
    fault = RuntimeError("internal validation failure")

    def broken(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(mod, target, broken)
    llm = SequenceLLM(payload(source, "x" * 501), {"summary": GOOD})
    with pytest.raises(mod.DigestCompletionError) as raised:
        separated(source, llm)
    assert raised.value.__cause__ is fault
    assert raised.value.failure_stage == "summary_compaction"
    assert len(llm.calls) == (1 if target == "_build_digest_summary_repair_request" else 2)
