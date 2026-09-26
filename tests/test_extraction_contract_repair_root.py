"""Independent parent controls for the reported schema-rejection loop."""
import json

import pytest

from hymem.extraction.chunk import extract_chunk


class FeedbackRequired:
    def __init__(self, missing=False):
        self.calls = []
        self.request_attempts = 0
        self.missing = missing

    def complete(self, request):
        self.calls.append(request)
        self.request_attempts += 1
        empty = {"triples": [], "markers": [], "complete": True}
        claim = {"subject": "user", "predicate": "uses", "object": "sqlite", "polarity": 1}
        if "OMISSION VERIFICATION PASS" in request.system:
            return json.dumps(empty)
        if "CONTRACT REPAIR PASS" in request.user:
            assert ("top.complete:missing" if self.missing else "predicate:not_allowed") in request.user
            assert "SECRET_REJECTED_VALUE" not in request.user
            return json.dumps({**empty, "triples": [claim]})
        if self.missing:
            return json.dumps({"triples": [claim]})
        return json.dumps({**empty, "triples": [{**claim, "predicate": "SECRET_REJECTED_VALUE"}]})


@pytest.mark.parametrize("missing", [False, True])
def test_feedback_breaks_repeated_rejection_without_changing_allowed_items(missing):
    client = FeedbackRequired(missing)
    result = extract_chunk(client, "Uses sqlite.", completion_call_limit=4)
    assert not result.failed
    assert [(t.subject, t.predicate, t.object) for t in result.triples] == [("user", "uses", "sqlite")]
    assert result.completion_calls == result.provider_attempts == len(client.calls) == 3
    assert client.calls[0].user in client.calls[1].user
    assert client.calls[0].system == client.calls[1].system


def test_feedback_repair_cannot_publish_without_remaining_verification_budget():
    client = FeedbackRequired()
    result = extract_chunk(client, "Uses sqlite.", completion_call_limit=2)
    assert result.failed and not result.triples and not result.markers
    assert result.failure_reason == "branch_incomplete"
    assert "right:resource_limit" in result.failure_details
    assert "right.calls:max_exceeded" in result.failure_details
    assert result.completion_calls == result.provider_attempts == len(client.calls) == 2
