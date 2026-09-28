"""Root-owned adversarial contracts; scripted replies are not model accuracy."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import socket

import pytest

from benchmarks import digest_record_review as review
from hymem.deadline import DeadlineExceeded
from tests.test_digest_source_review_root import Client, TEMPLATE, packet, response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("record-review root checks are offline")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def prepare(value=None, **kwargs):
    return review.prepare_record_review(packet() if value is None else value, TEMPLATE,
        max_calls=kwargs.pop("max_calls", 3), **kwargs)


def parse(scope, value):
    return review.parse_record_review(json.dumps(value, ensure_ascii=False), scope)


def actor(scope):
    return next(c for c in scope.checks if c.kind == "actor_attribution")


def test_original_scope_sampling_and_evidence_preserved():
    prepared = prepare()
    assert len(prepared.requests) == 3
    for scope, base in zip(prepared.requests, prepared.source_plan.requests, strict=True):
        assert scope.source_scope == base
        assert (scope.kind, scope.index, scope.fields) == (base.kind, base.index, base.fields)
        assert replace(scope.request, system=TEMPLATE.system, user=TEMPLATE.user) == TEMPLATE
        assert scope.canonical_sources == base.canonical_sources
        assert scope.context_sources == base.context_sources
        assert scope.prior_summary_sources == base.prior_summary_sources
        assert [c for c in scope.checks if c.kind == "retention"] == [c for c in base.checks if c.kind == "retention"]
        assert len([c for c in scope.checks if c.kind == "actor_attribution"]) == len([c for c in base.checks if c.kind == "relations"])
        wire = json.loads(scope.request.user)
        assert "source_records" in wire
        assert "source_scope" not in wire and "source_plan" not in wire
        assert "REJECTED_RAW_ONLY" not in scope.request.user
        if scope.kind != "summary":
            assert "PRIOR_CONTINUITY_ONLY" not in scope.request.user
            assert "UNCITED_REPLACEMENT_ONLY" not in scope.request.user


def cross_message():
    value = packet()
    record = value["source_catalog"][0]
    record.update(role="assistant", start=0, end=len(record["visible_content"]))
    record["interpretation_only_context"].update(message_id=50, role="user")
    return value


def test_boundary_metadata_stays_with_its_actual_text_not_owner_speaker():
    scope = prepare(cross_message()).requests[0]
    record = next(r for r in json.loads(scope.request.user)["source_records"] if r["chunk_id"] == "a")
    assert record["current_message"]["message_id"] == 51
    assert record["current_message"]["role"] == "assistant"
    boundary = record["boundary_context"]
    assert boundary["message"]["message_id"] == 50 and boundary["message"]["role"] == "user"
    assert boundary["source"]["message_id"] == 50 and boundary["source"]["allowed_use"] == "interpretation"
    assert record["canonical_source"]["message_id"] == 51 and record["canonical_source"]["allowed_use"] == "support"
    assert record["canonical_source"]["text"] == cross_message()["source_catalog"][0]["visible_content"]
    assert boundary["content"] == cross_message()["source_catalog"][0]["interpretation_only_context"]["content"]


def test_role_exchange_changes_binding_without_changing_text_or_candidate():
    left = cross_message()
    right = deepcopy(left)
    right["source_catalog"][0]["role"] = "user"
    right["source_catalog"][0]["interpretation_only_context"]["role"] = "assistant"
    a, b = prepare(left), prepare(right)
    assert a.plan_sha256 != b.plan_sha256
    assert a.requests[0].binding_sha256 != b.requests[0].binding_sha256
    wa, wb = (json.loads(p.requests[0].request.user) for p in (a,b))
    assert wa["candidate"] == wb["candidate"]
    assert wa["source_records"][0]["canonical_source"]["text"] == wb["source_records"][0]["canonical_source"]["text"]


@pytest.mark.parametrize("fault", ["speaker", "context_speaker", "context_message", "drop_context"])
def test_rehashed_record_projection_tamper_rejected(fault):
    plan = prepare(cross_message())
    scope = plan.requests[0]
    wire = json.loads(scope.request.user)
    record = wire["source_records"][0]
    if fault == "speaker":
        record["current_message"]["role"] = "user"
    elif fault == "context_speaker":
        record["boundary_context"]["message"]["role"] = "assistant"
    elif fault == "context_message":
        record["boundary_context"]["message"]["message_id"] = 51
    else:
        record["boundary_context"] = None
    bad = replace(scope, request=replace(scope.request, user=json.dumps(wire)))
    bad = replace(bad, binding_sha256=review._sha(review._scope_body(bad)))
    forged = replace(plan, requests=(bad, *plan.requests[1:]))
    forged = replace(forged, plan_sha256=review._sha(review._plan_body(forged)))
    with pytest.raises(ValueError):
        review.parse_record_review("{}", bad)
    client = Client([])
    with pytest.raises(ValueError):
        review.execute_record_review(forged, client)
    assert not client.calls


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_separate_actor_veto_cannot_be_masked_by_other_accepts(verdict):
    scope = prepare().requests[0]
    raw = response(scope)
    raw[actor(scope).check_id] = [verdict, [], []]
    outcome = parse(scope, raw)
    assert outcome.review_structure_valid and not outcome.model_no_defect
    assert not outcome.model_grounding_supported
    assert not outcome.semantic_verified and not outcome.publication_authorized


@pytest.mark.parametrize("fault", ["missing", "unknown", "null", "wrong_shape"])
def test_actor_contract_failure_invalidates_whole_scope(fault):
    scope = prepare().requests[0]
    raw = response(scope)
    key = actor(scope).check_id
    if fault == "missing":
        del raw[key]
    elif fault == "unknown":
        raw[key+"_extra"] = raw.pop(key)
    else:
        raw[key] = None if fault == "null" else ["supported", []]
    outcome = parse(scope, raw)
    assert outcome.status == "malformed_review" and outcome.judgments == ()


@pytest.mark.parametrize("kind", ["attribution", "boundary_context"])
def test_actor_cannot_promote_metadata_or_boundary_to_primary(kind):
    scope = prepare().requests[0]
    raw = response(scope)
    context = next(s for s in scope.context_sources if s.kind == kind)
    raw[actor(scope).check_id] = ["supported", [context.source_id], []]
    assert parse(scope, raw).status == "malformed_review"


def test_actor_context_requires_its_own_canonical_record_not_another_or_prior():
    scope = prepare().requests[-1]
    context = next(s for s in scope.context_sources if s.kind == "boundary_context")
    own = next(s for s in scope.canonical_sources if s.chunk_id == context.chunk_id)
    other = next(s for s in scope.canonical_sources if s.chunk_id != context.chunk_id)
    for source in (other, scope.prior_summary_sources[0]):
        raw = response(scope)
        raw[actor(scope).check_id] = ["supported", [source.source_id], [context.source_id]]
        assert parse(scope, raw).status == "malformed_review"
    raw[actor(scope).check_id] = ["supported", [own.source_id], [context.source_id]]
    assert parse(scope, raw).review_structure_valid


def test_authorized_but_irrelevant_citation_is_not_semantic_proof():
    # Deliberately choose the pump source to sponsor an observatory assertion.
    # Structural validity cannot identify arbitrary natural-language entailment.
    scope = prepare().requests[0]
    raw = response(scope)
    pump = next(s for s in scope.canonical_sources if s.chunk_id == "b")
    for check in scope.checks:
        if check.kind != "retention":
            raw[check.check_id] = ["supported", [pump.source_id], []]
    outcome = parse(scope, raw)
    assert outcome.review_structure_valid and outcome.model_no_defect
    assert not outcome.semantic_verified and not outcome.publication_authorized


def test_old_relation_reply_cannot_be_silently_reinterpreted_as_new_facets():
    scope = prepare().requests[0]
    old_reply = response(scope.source_scope)
    outcome = parse(scope, old_reply)
    assert outcome.status == "malformed_review" and not outcome.judgments


def test_old_scorer_cannot_apply_aggregate_relations_gold_to_new_scope():
    from benchmarks.digest_source_review_evaluation import score_scope
    scope = prepare().requests[0]
    old_labels = [{"id":"old-relations","view":"primary","expected":"supported",
        "rationale":"Old aggregate labels cannot become facet labels by relabelling.",
        "selectors":[{"kind":"relations","field_path":"/candidate_body"}]}]
    with pytest.raises(ValueError):
        score_scope(json.dumps(response(scope)),scope,old_labels)


@pytest.mark.parametrize("field", ["system", "user", "temperature", "max_tokens"])
def test_tampered_last_scope_blocks_all_calls(field):
    prepared = prepare()
    scope = prepared.requests[-1]
    changed = {"system":"replacement", "user":"{}", "temperature":0.5, "max_tokens":42}[field]
    bad = replace(scope, request=replace(scope.request, **{field:changed}))
    forged = replace(prepared, requests=(*prepared.requests[:-1], bad))
    client = Client([])
    with pytest.raises(ValueError):
        review.execute_record_review(forged, client)
    assert not client.calls
    with pytest.raises(ValueError):
        review.parse_record_review("{}", bad)


@pytest.mark.parametrize("field", ["max_calls", "max_input_chars", "max_output_chars", "max_checks", "max_evidence_per_check"])
@pytest.mark.parametrize("value", [True, 0, -1, "3", 1.5])
def test_boundaries_reject_nonpositive_noninteger_caps(field, value):
    with pytest.raises(ValueError):
        prepare(**{field:value})


def test_full_expanded_check_and_output_caps_are_enforced():
    plan = prepare()
    largest = max(len(s.checks) for s in plan.requests)
    with pytest.raises(ValueError):
        prepare(max_checks=largest-1)
    with pytest.raises(ValueError):
        prepare(max_output_chars=1)
    with pytest.raises(ValueError):
        review.parse_record_review("{}", plan.requests[0], max_checks=len(plan.requests[0].checks)-1)


def test_scripted_rejection_continues_without_retry_or_semantic_repair():
    plan = prepare()
    client = Client(["bad", *(json.dumps(response(s)) for s in plan.requests[1:])])
    outcome = review.execute_record_review(plan, client)
    assert len(client.calls) == outcome.attempted_calls == 3 and outcome.complete
    assert not outcome.review_structure_valid and not outcome.model_no_defect
    assert outcome.outcomes[0].status == "malformed_review"
    assert not outcome.semantic_verified and not outcome.publication_authorized


def test_transport_failure_halts_without_exposing_error_text():
    client = Client([RuntimeError("PRIVATE_MARKER")])
    result = review.execute_record_review(prepare(), client)
    assert len(client.calls) == result.attempted_calls == 1 and not result.complete
    assert "PRIVATE_MARKER" not in repr(result)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), DeadlineExceeded("stop")])
def test_deadline_and_process_interrupt_propagate(error):
    client = Client([error])
    with pytest.raises(type(error)):
        review.execute_record_review(prepare(), client)
    assert len(client.calls) == 1
