"""Construction checks for independently labelled invented grounding controls."""
import dataclasses
import json

from hymem.extraction.grounding import (
    MAX_EVIDENCE, MAX_QUOTE_CHARS, MAX_TOTAL_SOURCE_CHARS, MAX_TRIPLES,
    build_grounding_request, parse_grounding_response,
)
from tools.diagnostics.luna_semantic_cases import cases, safe_metadata, suite_sha256


def _example_reply(case, batch):
    verdicts = []
    for index, expected in enumerate(case.expected):
        status = sorted(expected.statuses)[0]
        predicate = (expected.predicate if status == "replace_predicate" else
                     case.triples[index].predicate if status == "supported" else None)
        verdicts.append({"index": index, "status": status, "predicate": predicate,
                         "evidence": [{"source_message_id": case.triples[index].source_message_id,
                                       "region": region, "quote": quote}
                                      for region, quote in expected.evidence]})
    return json.dumps({"schema": "source-grounding-v1", "batch_sha256": batch.batch_sha256,
                       "complete": True, "verdicts": verdicts})


def test_fixed_suite_identity_and_safe_projection():
    first = cases()
    assert len(first) == 24
    assert first == cases() and first is not cases()
    assert len({case.case_id for case in first}) == len(first)
    assert suite_sha256() == "511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925"
    meta = safe_metadata()
    assert meta["case_count"] == len(first)
    assert meta["sha256"] == suite_sha256()
    projection = json.dumps(meta)
    for case in first:
        for source in case.sources:
            assert source.content not in projection
        for expected in case.expected:
            assert expected.rationale not in projection


def test_all_cases_are_real_wire_requests_and_example_verdicts():
    hashes = set()
    for case in cases():
        assert 1 <= len(case.triples) <= MAX_TRIPLES
        assert len(case.expected) == len(case.triples)
        assert sum(len(s.content) + sum(len(c.content) for c in s.contexts)
                   for s in case.sources) <= MAX_TOTAL_SOURCE_CHARS
        request, batch = build_grounding_request(case.triples, case.sources)
        assert batch.batch_sha256 not in hashes
        hashes.add(batch.batch_sha256)
        assert request.response_format == "json"
        assert request.temperature == 0
        wire = json.loads(request.user)
        assert wire["batch_sha256"] == batch.batch_sha256
        assert len(wire["batch"]["candidates"]) == len(case.triples)
        assert len(wire["batch"]["sources"]) == len(case.sources)
        for source in wire["batch"]["sources"]:
            assert source["source_message_id"] is not None
            assert source["source_role"] in {"user", "assistant"}
            assert source["source_peer_id"]
            assert source["source_created_at"]
        for expected in case.expected:
            assert expected.rationale.strip()
            assert 0 <= len(expected.evidence) <= MAX_EVIDENCE
            assert all(0 < len(quote) <= MAX_QUOTE_CHARS for _, quote in expected.evidence)
        review = parse_grounding_response(_example_reply(case, batch), batch)
        assert tuple(v.status for v in review.verdicts) == tuple(
            sorted(e.statuses)[0] for e in case.expected)
        assert review.all_supported is (case.category == "supported")


def test_context_identity_and_parent_bounds_are_on_real_wire():
    for case in cases():
        _, batch = build_grounding_request(case.triples, case.sources)
        wire = json.loads(batch.canonical_json)
        for source in wire["sources"]:
            by_region = {c["region"]: c for c in source["contexts"]}
            assert len(by_region) == len(source["contexts"])
            for context in source["contexts"]:
                assert 0 < context["owned_prefix_chars"] <= len(source["content"])
                assert set(context) == {
                    "region", "content", "owned_prefix_chars", "source_role", "source_peer_id",
                    "source_created_at", "source_message_id", "applies_to_region",
                    "applies_to_prefix_chars",
                }
                parent_region = context["applies_to_region"]
                if parent_region is None:
                    assert context["applies_to_prefix_chars"] is None
                    continue
                parent = by_region[parent_region]
                assert 0 < context["applies_to_prefix_chars"] <= len(parent["content"])
                for field in ("source_role", "source_peer_id", "source_created_at", "source_message_id"):
                    assert context[field] == parent[field]


def test_predicate_only_corrections_preserve_claim_identity():
    corrections = [case for case in cases() if case.category == "correction"]
    assert len(corrections) == 2
    for case in corrections:
        for original, expected in zip(case.triples, case.expected, strict=True):
            assert expected.statuses == frozenset({"replace_predicate"})
            assert expected.predicate and expected.predicate != original.predicate
            corrected = dataclasses.replace(original, predicate=expected.predicate)
            for field in ("subject", "object", "polarity", "source_message_id",
                          "value_text", "value_numeric", "value_unit", "temporal_scope"):
                assert getattr(corrected, field) == getattr(original, field)
            assert build_grounding_request((corrected,), case.sources)


def test_negative_labels_are_rejections_of_claims_not_empty_publications():
    rejects = [case for case in cases() if case.category == "reject"]
    assert len(rejects) == 10
    for case in rejects:
        assert all(x.statuses == frozenset({"unsupported", "uncertain"}) and not x.evidence
                   and x.predicate is None for x in case.expected)
        assert case.triples  # A gate run should fail this unit, not publish an empty answer.
