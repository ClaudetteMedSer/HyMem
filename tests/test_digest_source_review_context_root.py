"""Independent checks of previous-speaker attribution, never model accuracy."""
from dataclasses import asdict, replace
import json
import socket

import pytest

from benchmarks import digest_source_review as review
from tests.test_digest_source_review_root import Client, grounding, packet, parse, prepare, response


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("offline attribution controls prohibit networking")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def cross_speaker(peer="previous-peer-Ω", workspace="previous-workspace"):
    value = packet()
    record = value["source_catalog"][0]
    record["start"], record["end"] = 0, len(record["visible_content"])
    ctx = record["interpretation_only_context"]
    ctx.update(message_id=50, role="assistant", source_peer_id=peer,
               source_workspace_id=workspace)
    return value


@pytest.mark.parametrize("peer,workspace", [(None, None), ("", ""), ("Ω-é", "W-否")])
def test_context_speaker_metadata_is_exact_and_not_parent_speaker(peer, workspace):
    value = cross_speaker(peer, workspace)
    scope = prepare(value).requests[0]
    context = next(s for s in scope.context_sources if s.kind == "boundary_context" and s.chunk_id == "a")
    attribution = next(x for x in scope.boundary_context_attributions if x.source_id == context.source_id)
    assert asdict(attribution) == {
        "source_id": context.source_id, "chunk_id": "a", "message_id": 50,
        "role": "assistant", "source_peer_id": peer, "source_workspace_id": workspace}
    assert value["source_catalog"][0]["role"] == "user"
    assert context.message_id == attribution.message_id
    wire = json.loads(scope.request.user)
    assert wire["boundary_context_attributions"] == [asdict(x) for x in scope.boundary_context_attributions]
    assert context.source_id not in {x.source_id for x in scope.canonical_sources}


def test_same_message_attribution_and_scope_isolation():
    plan = prepare()
    for scope in plan.requests:
        contexts = {s.source_id: s for s in scope.context_sources if s.kind == "boundary_context"}
        assert {x.source_id for x in scope.boundary_context_attributions} == set(contexts)
        for attribution in scope.boundary_context_attributions:
            context = contexts[attribution.source_id]
            assert attribution.message_id == context.message_id
            assert attribution.chunk_id == context.chunk_id
            assert attribution.source_peer_id == "peer-not-a-name"
        if scope.kind == "procedure":
            assert {x.chunk_id for x in scope.boundary_context_attributions} == {"b"}


def test_empty_boundary_text_does_not_create_returnable_metadata_evidence():
    value = packet()
    record = value["source_catalog"][0]
    record["start"], record["end"] = 0, len(record["visible_content"])
    ctx = record["interpretation_only_context"]
    ctx.update(start=0, end=0, content="")
    scope = prepare(value).requests[0]
    assert "a" not in {x.chunk_id for x in scope.boundary_context_attributions}


@pytest.mark.parametrize("field,changed", [
    ("role", "user"), ("message_id", 51), ("source_peer_id", "parent-peer"),
    ("source_workspace_id", "parent-space"), ("source_id", "s0"), ("chunk_id", "b"),
])
def test_rehashed_attribution_tamper_fails_standalone_and_before_invocation(field, changed):
    plan = prepare(cross_speaker())
    scope = plan.requests[0]
    wire = json.loads(scope.request.user)
    wire["boundary_context_attributions"][0][field] = changed
    bad = replace(scope, request=replace(scope.request, user=review._canonical(wire)))
    bad = replace(bad, binding_sha256=review._sha(review._scope_body(bad)))
    with pytest.raises(ValueError):
        review.parse_source_review("{}", bad)
    forged = replace(plan, requests=(bad, *plan.requests[1:]))
    forged = replace(forged, plan_sha256=review._sha(review._plan_body(forged)))
    client = Client([])
    with pytest.raises(ValueError):
        review.execute_source_review(forged, client)
    assert client.calls == []


def test_adding_original_speaker_does_not_promote_boundary_context():
    scope = prepare(cross_speaker()).requests[0]
    metadata = scope.boundary_context_attributions[0]
    body = response(scope)
    key = grounding(scope).check_id
    body[key] = ["supported", [metadata.source_id], []]
    assert parse(scope, body).status == "malformed_review"
    canonical = next(s for s in scope.canonical_sources if s.chunk_id == metadata.chunk_id)
    body[key] = ["supported", [canonical.source_id], [metadata.source_id]]
    result = parse(scope, body)
    assert result.review_structure_valid
    assert not result.semantic_verified and not result.publication_authorized
