"""Contract repair has source authority and a bounded publication ladder."""
import json

import pytest

from hymem.extraction.chunk import extract_chunk
from hymem.extraction.llm import LLMOutputTruncatedError


EMPTY = {"triples": [], "markers": [], "complete": True}
CLAIM = {"subject": "user", "predicate": "uses", "object": "sqlite", "polarity": 1}
GOOD = {**EMPTY, "triples": [CLAIM]}
BAD = {**EMPTY, "triples": [{**CLAIM, "predicate": "private_invalid_predicate"}]}


class Sequence:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
        self.request_attempts = 0

    def complete(self, request):
        self.calls.append(request)
        self.request_attempts += 2
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response) if isinstance(response, (dict, list)) else response


def test_invalid_predicate_repair_is_source_only_and_still_certified():
    client = Sequence(BAD, GOOD, EMPTY)
    result = extract_chunk(client, "The user uses sqlite.")
    assert not result.failed and len(result.triples) == 1
    assert result.completion_calls == 3 and result.provider_attempts == 6
    repair = client.calls[1]
    assert repair.system == client.calls[0].system
    assert "triples[0].predicate:not_allowed" in repair.user
    assert "private_invalid_predicate" not in repair.user
    assert "OMISSION VERIFICATION PASS" in client.calls[2].system


def test_missing_keys_repair_revalidates_complete_response():
    client = Sequence({"triples": [CLAIM]}, GOOD, EMPTY)
    result = extract_chunk(client, "The user uses sqlite.")
    assert not result.failed
    assert "top.complete:missing" in client.calls[1].user
    assert "top.markers:missing" in client.calls[1].user
    assert result.completion_calls == 3


def test_omission_repair_keeps_role_and_accepted_context_without_verifier_loop():
    client = Sequence(GOOD, BAD, GOOD)
    result = extract_chunk(client, "The user uses sqlite.")
    assert not result.failed and len(result.triples) == 1
    assert len(client.calls) == 3
    original, repair = client.calls[1:]
    assert repair.system == original.system
    assert "OMISSION VERIFICATION PASS" in repair.system
    assert original.user in repair.user
    assert "ALREADY ACCEPTED RESULT" in repair.user
    assert "private_invalid_predicate" not in repair.user


def test_malformed_repair_cannot_publish_primary_partial():
    client = Sequence(GOOD, BAD, "malformed")
    result = extract_chunk(client, "x")
    assert result.failed and result.triples == []
    assert "left" not in result.failure_reason
    assert any(d.endswith("repair:failed") for d in result.failure_details)
    assert any(d.endswith("stage:omission") for d in result.failure_details)
    assert len(client.calls) == 3


def test_primary_repaired_empty_cannot_erase_validation_obligation():
    client = Sequence(BAD, EMPTY)
    result = extract_chunk(client, "x")
    assert result.failed and result.triples == []
    assert "repair:empty_after_invalid" in result.failure_details
    assert len(client.calls) == 2


def test_omission_repaired_empty_cannot_drop_invalid_additional_claim():
    bad_extra = {**BAD, "triples": [{**BAD["triples"][0], "object": "redis"}]}
    client = Sequence(GOOD, bad_extra, EMPTY)
    result = extract_chunk(client, "Uses sqlite and redis.")
    assert result.failed and result.triples == []
    assert any(d.endswith("repair:empty_after_invalid") for d in result.failure_details)
    assert any(d.endswith("stage:omission") for d in result.failure_details)
    assert len(client.calls) == 3


def test_empty_check_cannot_repair_invalid_items_away():
    client = Sequence(EMPTY, BAD, EMPTY)
    result = extract_chunk(client, "x")
    assert result.failed and result.triples == []
    assert "repair:empty_after_invalid" in result.failure_details
    assert "stage:empty" in result.failure_details
    assert len(client.calls) == 3


def test_repaired_omission_still_validates_source_citation(monkeypatch):
    from hymem.extraction import chunk as api
    # Prevent unrelated source subdivision/retry from obscuring the violation.
    monkeypatch.setattr(api, "_split_unit", lambda unit: None)
    record = (7, json.dumps({"content": "Uses sqlite.", "source_message_id": 7,
        "source_created_at": "2026-09-26T00:00:00Z", "source_peer_id": None,
        "source_record_version": "hymem-claim-source-v2", "source_role": "user",
        "source_session_id": "s", "source_workspace_id": None}))
    good = {**GOOD, "triples": [{**CLAIM, "source_message_id": 7}]}
    invalid = {**BAD, "triples": [{**BAD["triples"][0], "source_message_id": 7}]}
    forged = {**GOOD, "triples": [{**CLAIM, "source_message_id": 99}]}
    client = Sequence(good, invalid, forged)
    result = extract_chunk(client, "unused", source_records=(record,))
    assert result.failed and result.triples == []
    assert any("source_message_id:not_in_input" in d for d in result.failure_details)
    assert len(client.calls) == 3


@pytest.mark.parametrize("limit, responses, calls", [
    (1, (BAD,), 1), (2, (BAD, GOOD), 2),
])
def test_repair_and_omission_share_existing_budget(limit, responses, calls):
    client = Sequence(*responses)
    result = extract_chunk(client, "x", completion_call_limit=limit)
    assert result.failed and result.triples == []
    assert result.failure_reason == "resource_limit" or any(
        d.endswith(":resource_limit") for d in result.failure_details)
    assert result.completion_calls == calls and result.provider_attempts == calls * 2


@pytest.mark.parametrize("response", [RuntimeError("provider"),
    LLMOutputTruncatedError(), {**EMPTY, "complete": False},
    {**EMPTY, "triples": [CLAIM] * 24}, "malformed"])
def test_noncontract_failures_never_use_repair(response):
    client = Sequence(response, response)
    result = extract_chunk(client, "x", completion_call_limit=2)
    assert result.failed
    assert all("CONTRACT REPAIR PASS" not in c.user for c in client.calls)


def test_repair_is_not_recursive_on_another_invalid_response():
    client = Sequence({"triples": []}, {"triples": []})
    result = extract_chunk(client, "x")
    assert result.failed and result.completion_calls == 2
    assert "repair:failed" in result.failure_details
    assert len(client.calls) == 2


def test_repair_respects_same_absolute_deadline(monkeypatch):
    from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline
    from hymem.extraction import chunk as api
    now = [0.0]
    deadline = MonotonicDeadline(1.0, clock=lambda: now[0])
    normalize = api.normalize_combined_triple_item

    def expire_during_validation(*args, **kwargs):
        result = normalize(*args, **kwargs)
        now[0] = 2.0
        return result

    monkeypatch.setattr(api, "normalize_combined_triple_item", expire_during_validation)
    client = Sequence(BAD)
    with pytest.raises(DeadlineExceeded):
        extract_chunk(DeadlineBoundLLMClient(client, deadline), "x")
    assert len(client.calls) == 1
    assert client.request_attempts == 2
