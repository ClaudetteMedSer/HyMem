"""Offline metadata-preservation checks, not model semantic-quality evidence."""
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from benchmarks import digest_evidence_assessment as assessment
from benchmarks import digest_source_review as review
from tests.test_digest_evidence_assessment import payload, template, Scripted
from tests.test_digest_source_review import response, rehash_plan, rehash_scope


def cross_message_payload():
    value = payload()
    record = value["source_catalog"][0]
    record.update(message_id=3, role="assistant", source_peer_id="answer-peer",
                  source_workspace_id="answer-space", start=0,
                  end=len(record["visible_content"]))
    context = record["interpretation_only_context"]
    context.update(message_id=1, role="user", source_peer_id="question-peer",
                   source_workspace_id="question-space")
    return value


def prepare(value=None, **kwargs):
    return review.prepare_source_review(payload() if value is None else value,
                                        template(), max_calls=3, **kwargs)


@pytest.mark.parametrize("cross_message", [False, True])
def test_exact_association_speaker_metadata_and_owner(cross_message):
    value = cross_message_payload() if cross_message else payload()
    prepared = prepare(value)
    record = value["source_catalog"][0]
    context = record["interpretation_only_context"]
    for scope in (prepared.requests[0], prepared.requests[2]):
        boundary = next(s for s in scope.context_sources if s.kind == "boundary_context")
        (attribution,) = scope.boundary_context_attributions
        assert attribution == review.BoundaryContextAttribution(
            boundary.source_id, record["chunk_id"], context["message_id"], context["role"],
            context["source_peer_id"], context["source_workspace_id"])
        assert attribution.chunk_id == boundary.chunk_id == record["chunk_id"]
        assert attribution.message_id == boundary.message_id
        canonical = next(s for s in scope.canonical_sources if s.chunk_id == attribution.chunk_id)
        assert (canonical.message_id != attribution.message_id) is cross_message
        assert json.loads(scope.request.user)["boundary_context_attributions"] == [asdict(attribution)]
        assert "NOT the current canonical speaker" in scope.request.system
        assert "not additional evidence IDs" in scope.request.system
        assert "interpretation-only" in scope.request.system
        with pytest.raises(FrozenInstanceError):
            attribution.role = "forged"
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            scope.boundary_context_attributions = ()


@pytest.mark.parametrize("peer", [None, "", "Context peer Ω 👩🏽‍💻 e\u0301"])
@pytest.mark.parametrize("workspace", [None, "", "Context workspace 雪"])
@pytest.mark.parametrize("cross_message", [False, True])
def test_nullable_empty_unicode_metadata_preserved_exactly(peer, workspace, cross_message):
    value = cross_message_payload() if cross_message else payload()
    record = value["source_catalog"][0]
    context = record["interpretation_only_context"]
    context.update(source_peer_id=peer, source_workspace_id=workspace, role="speaker Ω")
    if not cross_message:
        record.update(source_peer_id=peer, source_workspace_id=workspace, role=context["role"])
    scope = prepare(value).requests[0]
    (attribution,) = scope.boundary_context_attributions
    assert attribution.source_peer_id == peer
    assert attribution.source_workspace_id == workspace
    assert attribution.role == "speaker Ω"
    wire = json.loads(scope.request.user)["boundary_context_attributions"][0]
    assert wire["source_peer_id"] == peer
    assert wire["source_workspace_id"] == workspace
    value["source_catalog"][0]["interpretation_only_context"]["role"] = "changed after freeze"
    assert scope.boundary_context_attributions[0].role == "speaker Ω"


def test_metadata_scoped_to_existing_nonempty_context_only_and_no_new_checks():
    value = cross_message_payload()
    original = assessment.prepare_evidence_assessment(value, template(), max_calls=3)
    prepared = prepare(value)
    assert len(prepared.requests) == len(original.requests) == 3
    for scope, old in zip(prepared.requests, original.requests, strict=True):
        assert scope.base_scope == old
        assert [a.source_id for a in scope.boundary_context_attributions] == [
            s.source_id for s in scope.context_sources if s.kind == "boundary_context"]
        assert len(scope.canonical_sources + scope.context_sources + scope.prior_summary_sources) == len(
            old.evidence_sources)
        assert [(c.check_id, c.kind, c.field_ids) for c in scope.checks if c.kind != "retention"] == [
            (c.check_id, c.kind, c.field_ids) for c in old.checks if c.kind != "retention"]
        assert replace(scope.request, system=template().system, user=template().user) == template()
    assert prepared.requests[1].boundary_context_attributions == ()
    assert "question-peer" not in prepared.requests[1].request.user
    assert json.loads(prepared.requests[1].request.user)["boundary_context_attributions"] == []
    value["source_catalog"][0]["interpretation_only_context"].update(content="", start=0, end=0)
    for scope in prepare(value).requests:
        assert scope.boundary_context_attributions == ()
        assert not any(s.kind == "boundary_context" for s in scope.context_sources)


def test_multiple_boundary_mappings_follow_owned_source_order_not_message_identity():
    value = cross_message_payload()
    second = value["source_catalog"][1]
    # Both windows originate in the same earlier message but belong to different
    # canonical records. They must remain distinct context evidence references.
    second["interpretation_only_context"] = dict(
        value["source_catalog"][0]["interpretation_only_context"])
    scope = prepare(value).requests[-1]
    mappings = scope.boundary_context_attributions
    assert len(mappings) == 2
    assert [m.chunk_id for m in mappings] == ["chunk-a", "chunk-b"]
    assert mappings[0].message_id == mappings[1].message_id == 1
    assert mappings[0].source_id != mappings[1].source_id
    assert [m.source_id for m in mappings] == [
        s.source_id for s in scope.context_sources if s.kind == "boundary_context"]


@pytest.mark.parametrize("change", [
    lambda wire: wire.pop("boundary_context_attributions"),
    lambda wire: wire.update(boundary_context_attributions=[]),
    lambda wire: wire["boundary_context_attributions"].append(
        dict(wire["boundary_context_attributions"][0])),
    lambda wire: wire["boundary_context_attributions"][0].update(source_id="s999"),
    lambda wire: wire["boundary_context_attributions"][0].update(
        source_id=wire["canonical_sources"][0]["source_id"]),
    lambda wire: wire["boundary_context_attributions"][0].update(chunk_id="chunk-b"),
    lambda wire: wire["boundary_context_attributions"][0].update(message_id=3),
    lambda wire: wire["boundary_context_attributions"][0].update(message_id=True),
    lambda wire: wire["boundary_context_attributions"][0].update(role="assistant"),
    lambda wire: wire["boundary_context_attributions"][0].update(source_peer_id=None),
    lambda wire: wire["boundary_context_attributions"][0].update(source_workspace_id=""),
    lambda wire: wire["boundary_context_attributions"][0].update(allowed_use="support"),
])
def test_rehashed_wire_metadata_tampering_rejected_standalone_and_before_any_call(change):
    prepared = prepare(cross_message_payload())
    scope = prepared.requests[-1]
    wire = json.loads(scope.request.user)
    change(wire)
    forged = rehash_scope(replace(scope, request=replace(scope.request, user=review._canonical(wire))))
    with pytest.raises(ValueError, match="binding mismatch"):
        review.parse_source_review(json.dumps(response(forged)), forged)
    forged_plan = rehash_plan(replace(prepared, requests=prepared.requests[:-1] + (forged,)))
    client = Scripted([])
    with pytest.raises(ValueError, match="binding mismatch"):
        review.execute_source_review(forged_plan, client)
    assert client.calls == []


def test_context_attribution_never_creates_primary_authority_or_new_reference_ids():
    scope = prepare(cross_message_payload()).requests[0]
    (attribution,) = scope.boundary_context_attributions
    canonical = scope.canonical_sources[0].source_id
    for primary, context in (([attribution.source_id], []), ([], [attribution.source_id]),
                             ([canonical], [attribution.source_peer_id])):
        reply = response(scope)
        reply["c0"] = ["supported", primary, context]
        outcome = review.parse_source_review(json.dumps(reply), scope)
        assert outcome.status == "malformed_review" and outcome.judgments == ()
    reply = response(scope)
    reply["c0"] = ["supported", [canonical], [attribution.source_id]]
    outcome = review.parse_source_review(json.dumps(reply), scope)
    assert outcome.review_structure_valid
    assert not outcome.semantic_verified and not outcome.publication_authorized
    assert next(j for j in outcome.judgments if j.check_id == "c0").context_evidence[0].allowed_use == "interpretation"


def test_new_metadata_contributes_to_complete_input_bound_without_truncation():
    value = cross_message_payload()
    context = value["source_catalog"][0]["interpretation_only_context"]
    context["source_peer_id"] = "Ω-peer-" * 1000
    prepared = prepare(value)
    bound = max(max(len(s.request.system) + len(s.request.user),
                    len(s.base_scope.request.system) + len(s.base_scope.request.user))
                for s in prepared.requests)
    assert prepare(value, max_input_chars=bound).requests == prepared.requests
    with pytest.raises(ValueError):
        prepare(value, max_input_chars=bound - 1)
    for scope in (prepared.requests[0], prepared.requests[-1]):
        assert scope.boundary_context_attributions[0].source_peer_id == context["source_peer_id"]
        assert json.loads(scope.request.user)["boundary_context_attributions"][0]["source_peer_id"] == context["source_peer_id"]


def test_same_message_conflicting_attribution_still_rejected_by_original_validator():
    value = payload()
    value["source_catalog"][0]["interpretation_only_context"]["role"] = "assistant"
    with pytest.raises(ValueError, match="same-message context"):
        prepare(value)
