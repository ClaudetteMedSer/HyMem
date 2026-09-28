"""Independent failure-origin controls; scripted replies do not prove semantics."""
from contextlib import closing
from dataclasses import replace
import json

import pytest

from hymem import HyMem
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import digest
from hymem.dreaming.summary_policy import LEGACY_COMPLETE_V1, BOUNDED_HIGHLIGHTS_V1
from tests.test_digest_bounded_runtime_root import PolicyScript, extract
from tests.test_digest_summary_content_recovery import REJECTED
from tests.test_digest_summary_contract import _config
from tests.test_digest_summary_verification import _seed


@pytest.mark.parametrize("policy", [LEGACY_COMPLETE_V1, BOUNDED_HIGHLIGHTS_V1])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("failed_stage,reason", [
    ("summary_diagnosis", "summary_diagnosis_parse_failure"),
    ("summary_content_recovery", "summary_content_recovery_parse_failure"),
    ("fidelity_reverification", "fidelity_parse_failure"),
    ("format_adjudication", "format_adjudication_parse_failure"),
])
def test_exact_failing_reply_origin_not_outer_verification(
    cfg, monkeypatch, caplog, policy, compact, failed_stage, reason,
):
    original = digest._complete_digest
    dispatched = []

    def dispatch(llm, request, *, stage):
        dispatched.append(stage)
        return "not JSON" if stage == failed_stage else original(llm, request, stage=stage)

    monkeypatch.setattr(digest, "_complete_digest", dispatch)
    client = PolicyScript(policy=policy, compact=compact)
    with closing(HyMem(replace(_config(cfg, False), digest_summary_policy=policy), llm=client)) as hy:
        _seed(hy, "The canyon came before the harbor; route advice was supplied.")
        source_before = tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM message_retention_coverage"))
        result = extract(hy, client)
        assert result.parse_failed and result.failure_reason == reason
        assert result.failure_stage == failed_stage == dispatched[-1]
        assert any(f"stage={failed_stage} reason={reason}" in record.getMessage()
                   for record in caplog.records)
        assert len(dispatched) <= 7 and dispatched.count(failed_stage) == 1
        assert result.episodes.items == result.procedures.items == []
        assert result.summary is result.covered_message_id is result.source_sha256 is None
        assert result.episode_input_items == result.episode_rejected_items == 1
        assert result.procedure_input_items == result.procedure_rejected_items == 1
        assert not digest.digest_failure_requires_input_shrink(reason, result.failure_stage)
        assert tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM message_retention_coverage")) == source_before
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert tuple(hy.conn.execute("SELECT auto_summary,digest_cursor_message_id FROM sessions WHERE id='summary-verification'").fetchone()) == (None, None)


@pytest.mark.parametrize("policy", [LEGACY_COMPLETE_V1, BOUNDED_HIGHLIGHTS_V1])
@pytest.mark.parametrize("first", ["unsupported", "uncertain"])
def test_unchanged_repair_keeps_original_veto_origin(cfg, policy, first):
    client = PolicyScript(policy=policy, repaired={"summary": REJECTED},
                          first=(("summary_content", 0, first),))
    with closing(HyMem(replace(_config(cfg, False), digest_summary_policy=policy), llm=client)) as hy:
        _seed(hy, "The canyon came before the harbor; route advice was supplied.")
        result = extract(hy, client)
    assert result.failure_stage == "fidelity_verification"
    assert result.failure_reason == "summary_content_" + first
    assert result.parse_failed and len(client.calls) == 4
    assert client.calls[-1].system.startswith("You repair one rolling conversation summary")
    assert result.summary is result.covered_message_id is result.source_sha256 is None


@pytest.mark.parametrize("stage,helper,skip", [
    ("summary_content_recovery", "_digest_summary_content_recovery_payload", 0),
    ("fidelity_reverification", "_encode_digest_fidelity_payload", 1),
    ("format_adjudication", "_encode_digest_format_adjudication_payload", 0),
])
@pytest.mark.parametrize("error_kind", ["ordinary", "nested", "deadline", "interrupt"])
def test_preparation_exception_origin_and_control_flow_are_preserved(
    cfg, monkeypatch, stage, helper, skip, error_kind,
):
    error = {
        "ordinary": ValueError("synthetic preparation error"),
        "nested": digest.DigestCompletionError("already_attributed"),
        "deadline": DeadlineExceeded("synthetic deadline"),
        "interrupt": KeyboardInterrupt(),
    }[error_kind]
    original = getattr(digest, helper)
    count = 0

    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count <= skip:
            return original(*args, **kwargs)
        raise error

    monkeypatch.setattr(digest, helper, fail)
    client = PolicyScript(policy=LEGACY_COMPLETE_V1)
    expected = digest.DigestCompletionError if error_kind == "ordinary" else type(error)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "The canyon came before the harbor; route advice was supplied.")
        with pytest.raises(expected) as caught:
            extract(hy, client)
        if error_kind == "ordinary":
            assert caught.value.failure_stage == stage and caught.value.__cause__ is error
        else:
            assert caught.value is error
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert tuple(hy.conn.execute("SELECT auto_summary,digest_cursor_message_id FROM sessions WHERE id='summary-verification'").fetchone()) == (None, None)


@pytest.mark.parametrize("policy", [LEGACY_COMPLETE_V1, BOUNDED_HIGHLIGHTS_V1])
@pytest.mark.parametrize("repair", [False, True])
def test_format_semantic_veto_is_never_called_fidelity(cfg, monkeypatch, policy, repair):
    original = digest._complete_digest

    def dispatch(llm, request, *, stage):
        raw = original(llm, request, stage=stage)
        if stage == "format_adjudication":
            payload = json.loads(raw)
            payload["episode_format"][0]["verdict"] = "uncertain"
            return json.dumps(payload)
        return raw

    monkeypatch.setattr(digest, "_complete_digest", dispatch)
    client = PolicyScript(policy=policy, first=(("summary_content", 0, "unsupported"),) if repair else ())
    with closing(HyMem(replace(_config(cfg, False), digest_summary_policy=policy), llm=client)) as hy:
        _seed(hy, "The canyon came before the harbor; route advice was supplied.")
        result = extract(hy, client)
    assert result.failure_stage == "format_adjudication"
    assert result.failure_reason == "episode_format_uncertain"
    assert result.parse_failed and len(client.calls) == (6 if repair else 3)
    assert result.summary is result.covered_message_id is result.source_sha256 is None
