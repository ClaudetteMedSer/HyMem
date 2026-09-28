"""Explicit synthetic approvals for tests of unrelated digest mechanics.

These are not semantic judgments: use scripted verdicts in fidelity tests.
Production StubLLMClient and real clients never auto-approve verification.
"""
import json

from hymem.dreaming import digest
from hymem.extraction.llm import StubLLMClient


def resolve_fidelity_sources(payload, source_ids):
    """Test-only catalog lookup; assert uniqueness instead of hiding collisions."""
    catalog = payload["source_catalog"]
    records = {record["chunk_id"]: record for record in catalog}
    assert len(records) == len(catalog)
    assert len(source_ids) == len(set(source_ids))
    return [records[source_id] for source_id in source_ids]


def synthetic_fidelity_result(episode_count=0, procedure_count=0):
    """Explicitly synthetic success for unrelated mechanics, not entailment."""
    return {
        key: [{"index": index, "verdict": "supported"} for index in range(count)]
        for key, count in (
            ("episode_titles", episode_count), ("episode_content", episode_count),
            ("procedures", procedure_count), ("summary_content", 1),
        )
    }


def synthetic_summary_issues(payload):
    """Explicit repair-control fixture, not a semantic judgment about its source.

    Opt in only in tests intentionally exercising the repair state machine;
    unrelated scripted rejections must keep missing diagnostics and remain held.
    """
    source = resolve_fidelity_sources(payload, payload["summary_item"]["new_source_ids"])[-1]
    return [{
        "code": "omitted_outcome", "candidate_quote": "",
        "sources": [{"kind": "new_source", "source_id": source["chunk_id"],
                     "quote": source["visible_content"][:512]}],
    }]


def synthetic_fidelity_approval(request):
    if request.system != digest._DIGEST_FIDELITY_SYSTEM:
        return None
    payload = json.loads(request.user)
    return json.dumps(synthetic_fidelity_result(len(payload["items"]), len(payload["procedure_items"])))


def synthetic_format_result(episode_count=0):
    """Explicit synthetic format success, not a grammar judgment."""
    return {
        "summary_format": [{"index": 0, "verdict": "supported"}],
        "episode_format": [{"index": index, "verdict": "supported"}
                           for index in range(episode_count)],
    }


def synthetic_format_approval(request):
    if request.system != digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
        return None
    payload = json.loads(request.user)
    return json.dumps(synthetic_format_result(len(payload["items"])))


class VerificationStubLLM(StubLLMClient):
    """Opt-in test fixture; primary scripted responses stay unchanged."""

    def complete(self, request):
        approval = synthetic_fidelity_approval(request)
        if approval is None:
            approval = synthetic_format_approval(request)
        if approval is not None:
            self.calls.append(request)
            return approval
        return super().complete(request)
