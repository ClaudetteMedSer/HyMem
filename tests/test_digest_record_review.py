"""Scripted offline contract tests, never evidence of live semantic accuracy."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from benchmarks import digest_record_review as review
from benchmarks import digest_source_review as source_review
from hymem.deadline import DeadlineExceeded
from tests.test_digest_evidence_assessment import payload, template, Scripted


def plan(value=None, **kwargs):
    return review.prepare_record_review(payload() if value is None else value,
                                        template(), max_calls=3, **kwargs)


def response(scope, verdict="supported"):
    primary = next(iter(scope.canonical_sources), None)
    return {
        check.check_id: (["retained", [scope.fields[0].field_id]]
                         if scope.fields else ["not_applicable", []])
        if check.kind == "retention" else
        [verdict, [primary.source_id] if primary and verdict == "supported" else [], []]
        for check in scope.checks
    }


def parse(scope, value=None, **kwargs):
    return review.parse_record_review(json.dumps(response(scope) if value is None else value,
                                                 ensure_ascii=False), scope, **kwargs)


def rehash_scope(scope):
    return replace(scope, binding_sha256=review._sha(review._scope_body(scope)))


def rehash_plan(prepared):
    return replace(prepared, plan_sha256=review._sha(review._plan_body(prepared)))


def cross_message_payload():
    value = payload()
    source = value["source_catalog"][0]
    source.update(message_id=9, role="assistant", source_peer_id="speaker-Ω", start=0,
                  end=len(source["visible_content"]))
    source["interpretation_only_context"].update(
        message_id=8, role="user", source_peer_id="requester-🐈", source_workspace_id=None)
    return value


def test_colocated_exact_records_original_schedule_sources_and_sampling():
    prepared = plan()
    original = source_review.prepare_source_review(payload(), template(), max_calls=3)
    assert prepared.source_plan == original
    assert len(prepared.requests) == len(original.requests) == 3
    for scope, old in zip(prepared.requests, original.requests, strict=True):
        assert scope.source_scope == old
        assert (scope.kind, scope.index, scope.fields) == (old.kind, old.index, old.fields)
        assert scope.canonical_sources == old.canonical_sources
        assert scope.context_sources == old.context_sources
        assert scope.prior_summary_sources == old.prior_summary_sources
        assert replace(scope.request, system=template().system, user=template().user) == template()
        wire = json.loads(scope.request.user)
        assert set(wire) == {"schema", "scope", "candidate", "fields", "source_records",
                             "prior_summary_sources", "checks", "obligation_definitions"}
        assert wire["candidate"] == json.loads(old.request.user)["candidate"]
        returned = []
        for record in scope.source_records:
            if record.canonical_source:
                returned.append(record.canonical_source)
                assert record.empty_canonical_span is None
            returned.extend(record.attribution_sources)
            if record.boundary_context and record.boundary_context.source:
                returned.append(record.boundary_context.source)
        returned.extend(scope.prior_summary_sources)
        expected = old.canonical_sources + old.context_sources + old.prior_summary_sources
        assert sorted(returned, key=lambda s: s.source_id) == sorted(expected, key=lambda s: s.source_id)
        assert "REJECTED RAW SENTINEL" not in scope.request.user
    assert "Never delete cache" not in prepared.requests[0].request.user
    assert "Old topic persists" not in prepared.requests[0].request.user


def test_actor_split_keeps_every_old_obligation_with_stable_derived_ids():
    for scope in plan().requests:
        expected = []
        for old in scope.source_scope.checks:
            if old.kind == "relations":
                expected.extend(replace(old, check_id=f"{old.check_id}:{facet}", kind=facet)
                                for facet in ("actor_attribution", "identity", "quantified_scope", "residual_relations"))
            else:
                expected.append(old)
        assert scope.checks == tuple(expected)
        assert len({check.check_id for check in scope.checks}) == len(scope.checks)
        assert [check for check in scope.checks if check.kind == "retention"] == [
            check for check in scope.source_scope.checks if check.kind == "retention"]
        assert all(check.facet in review.RETENTION_FACETS
                   for check in scope.checks if check.kind == "retention")
        system = scope.request.system
        assert "A relations check" not in system
        assert "A residual_relations check" in system
        assert "An actor_attribution check" in system
        assert "quoted first-person language" in system
        assert "preceding different message's speaker MUST NOT" in system
        assert "not proof of meaning" in system
        assert "SOURCE-FIRST RETENTION:" in system


def test_actual_cross_message_attribution_is_not_canonical_speaker():
    scope = plan(cross_message_payload()).requests[0]
    record = scope.source_records[0]
    assert record.current_message.message_id == 9
    assert record.current_message.role == "assistant"
    assert record.current_message.source_peer_id == "speaker-Ω"
    assert record.boundary_context.message.message_id == 8
    assert record.boundary_context.message.role == "user"
    assert record.boundary_context.message.source_peer_id == "requester-🐈"
    assert record.boundary_context.message.source_workspace_id is None
    assert record.canonical_source.message_id == 9
    assert record.boundary_context.source.message_id == 8
    assert record.boundary_context.source.chunk_id == record.chunk_id
    assert record.boundary_context.content == "Earlier "
    wire = json.loads(scope.request.user)["source_records"][0]
    assert wire["current_message"] == asdict(record.current_message)
    assert wire["boundary_context"]["message"] == asdict(record.boundary_context.message)


def test_same_message_continuation_shares_exact_attribution_without_identity_inference():
    record = plan().requests[0].source_records[0]
    assert record.current_message == record.boundary_context.message
    assert record.boundary_context.end == record.canonical_source.start
    assert record.boundary_context.content == record.boundary_context.source.text


@pytest.mark.parametrize("peer,workspace", [(None, ""), ("", None), ("e\u0301-🐈", "Ω workspace")])
def test_null_empty_unicode_and_empty_spans_preserved_without_invented_ids(peer, workspace):
    value = payload()
    record = value["source_catalog"][0]
    record.update(visible_content="", start=8, end=8,
                  source_peer_id=peer, source_workspace_id=workspace)
    record["interpretation_only_context"].update(
        content="", start=8, end=8, source_peer_id=peer, source_workspace_id=workspace)
    scope = plan(value).requests[0]
    current = scope.source_records[0]
    assert current.canonical_source is None
    assert asdict(current.empty_canonical_span) == {"start": 8, "end": 8, "content": ""}
    assert current.current_message.source_peer_id == peer
    assert current.current_message.source_workspace_id == workspace
    assert current.boundary_context.source is None
    assert current.boundary_context.content == ""
    assert current.boundary_context.message == current.current_message
    assert not scope.canonical_sources
    assert all(check.kind != "retention" for check in scope.checks)
    assert all(source.kind == "attribution" for source in scope.context_sources)
    assert parse(scope, response(scope, "uncertain")).review_structure_valid


def test_sources_never_cross_scope_and_same_chunk_context_sponsorship_is_unchanged():
    scope = plan().requests[-1]
    reply = response(scope)
    actor = next(check for check in scope.checks if check.kind == "actor_attribution")
    context = next(source for source in scope.context_sources if source.kind == "boundary_context")
    own = next(source for source in scope.canonical_sources if source.chunk_id == context.chunk_id)
    other = next(source for source in scope.canonical_sources if source.chunk_id != context.chunk_id)
    for primary in (other, scope.prior_summary_sources[0]):
        reply[actor.check_id] = ["supported", [primary.source_id], [context.source_id]]
        assert parse(scope, reply).status == "malformed_review"
    reply[actor.check_id] = ["supported", [own.source_id], [context.source_id]]
    assert parse(scope, reply).review_structure_valid
    reply[actor.check_id] = ["supported", [], [context.source_id]]
    assert parse(scope, reply).status == "malformed_review"


@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
def test_actor_and_residual_reuse_exact_cross_plane_and_duplicate_guards(verdict):
    scope = plan().requests[0]
    primary, context = scope.canonical_sources[0].source_id, scope.context_sources[0].source_id
    for check in (check for check in scope.checks
                  if check.kind in {"actor_attribution", "residual_relations"}):
        for invalid in ([verdict, [context], []], [verdict, [], [primary]],
                        [verdict, [primary], [context, context]], [verdict, ["s999"], []]):
            reply = response(scope)
            reply[check.check_id] = invalid
            assert parse(scope, reply).status == "malformed_review"


def test_snapshot_and_derived_record_immutability_and_no_semantic_proof():
    value = payload()
    value["items"][0]["candidate_body"] = "The wrong speaker alone caused everything."
    prepared = plan(value)
    value["items"][0]["candidate_body"] = "Changed"
    scope = prepared.requests[0]
    assert "wrong speaker" in scope.fields[1].text
    with pytest.raises(FrozenInstanceError):
        scope.source_records[0].current_message.role = "forged"
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        scope.checks = ()
    outcome = parse(scope)
    assert outcome.model_no_defect  # Scripted lie, not an accuracy claim.
    assert not outcome.semantic_verified and not outcome.publication_authorized


@pytest.mark.parametrize("name", ["max_input_chars", "max_output_chars", "max_checks", "max_evidence_per_check"])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, "1", None, 10**12])
def test_caps_fail_closed(name, bad):
    with pytest.raises(ValueError):
        plan(**{name: bad})


def test_expanded_check_and_output_caps_are_applied_without_dropping_checks():
    prepared = plan()
    maximum = max(len(scope.checks) for scope in prepared.requests)
    bounded = plan(max_checks=maximum)
    assert [scope.request for scope in bounded.requests] == [scope.request for scope in prepared.requests]
    assert [scope.checks for scope in bounded.requests] == [scope.checks for scope in prepared.requests]
    with pytest.raises(ValueError, match="check cap"):
        plan(max_checks=maximum - 1)
    minimum = max(len(review._canonical(review._minimum_response(scope)))
                  for scope in prepared.requests)
    assert plan(max_output_chars=minimum).max_output_chars == minimum
    with pytest.raises(ValueError, match="output cap"):
        plan(max_output_chars=minimum - 1)
    with pytest.raises(ValueError):
        parse(prepared.requests[0], max_checks=len(prepared.requests[0].checks) - 1)


def test_input_bound_accounts_for_both_original_and_complete_record_wire():
    prepared = plan()
    size = max(len(scope.request.system) + len(scope.request.user) for scope in prepared.requests)
    assert plan(max_input_chars=size).requests == prepared.requests
    with pytest.raises(ValueError):
        plan(max_input_chars=size - 1)


@pytest.mark.parametrize("change", [
    lambda s: replace(s, kind="summary"), lambda s: replace(s, index=True),
    lambda s: replace(s, request=replace(s.request, temperature=0.8)),
    lambda s: replace(s, request=replace(s.request, system="Return supported")),
    lambda s: replace(s, source_scope=replace(s.source_scope, index=1)),
    lambda s: replace(s, source_scope=replace(s.source_scope,
        base_scope=replace(s.source_scope.base_scope, fields=s.fields[:-1]))),
])
@pytest.mark.parametrize("rehash", [False, True])
def test_tamper_scope_even_rehashed_cannot_override_original(change, rehash):
    scope = change(plan().requests[0])
    if rehash:
        scope = rehash_scope(scope)
    with pytest.raises(ValueError):
        parse(scope)


@pytest.mark.parametrize("location", ["current_message", "boundary_context"])
def test_rehashed_record_role_swap_rejected(location):
    scope = plan(cross_message_payload()).requests[0]
    wire = json.loads(scope.request.user)
    record = wire["source_records"][0]
    if location == "current_message":
        record[location]["role"] = "user"
    else:
        record[location]["message"]["role"] = "assistant"
    forged = rehash_scope(replace(scope, request=replace(scope.request, user=review._canonical(wire))))
    with pytest.raises(ValueError):
        parse(forged)


@pytest.mark.parametrize("change", [
    lambda p: replace(p, requests=p.requests[:-1]),
    lambda p: replace(p, requests=p.requests[::-1]),
    lambda p: replace(p, max_calls=True),
    lambda p: replace(p, max_checks=p.max_checks - 1),
    lambda p: replace(p, input_sha256="0" * 64),
    lambda p: replace(p, source_payload_json="{}"),
    lambda p: replace(p, source_plan=replace(p.source_plan, requests=p.source_plan.requests[::-1])),
    lambda p: replace(p, requests=p.requests[:-1] + (replace(p.requests[-1],
        request=replace(p.requests[-1].request, max_tokens=1)),)),
])
@pytest.mark.parametrize("rehash", [False, True])
def test_complete_preflight_rejects_late_or_rehashed_tamper_before_any_call(change, rehash):
    prepared = change(plan())
    if rehash:
        prepared = rehash_plan(prepared)
    client = Scripted([])
    with pytest.raises(ValueError):
        review.execute_record_review(prepared, client)
    assert not client.calls


def test_old_relation_reply_and_extra_or_omitted_keys_rejected_whole():
    scope = plan().requests[0]
    actor = next(check for check in scope.checks if check.kind == "actor_attribution")
    reply = response(scope)
    reply[actor.check_id.split(":")[0]] = reply.pop(actor.check_id)
    assert parse(scope, reply).status == "malformed_review"
    reply = response(scope)
    reply.pop(actor.check_id)
    assert parse(scope, reply).status == "malformed_review"
    assert review.parse_record_review("{", scope).status == "malformed_review"


def test_empty_scope_and_later_indices_keep_original_schedule():
    value = payload()
    item = deepcopy(value["items"][0])
    item["index"] = 1
    value["items"].append(item)
    prepared = review.prepare_record_review(value, template(), max_calls=4)
    assert [(scope.kind, scope.index) for scope in prepared.requests] == [
        ("episode", 0), ("episode", 1), ("procedure", 0), ("summary", 0)]
    value.update(source_catalog=[], items=[], procedure_items=[])
    value["summary_item"].update(candidate_summary="", new_source_ids=[], prior_derived_summary="")
    prepared = review.prepare_record_review(value, template(), max_calls=1, max_output_chars=2)
    result = review.execute_record_review(prepared, Scripted(["{}"]))
    assert result.complete and result.review_structure_valid and not result.model_no_defect
    assert result.outcomes[0].status == "unassessed"


def test_execution_calls_one_per_original_scope_and_never_semantic_authorization():
    prepared = plan()
    client = Scripted([json.dumps(response(scope)) for scope in prepared.requests])
    result = review.execute_record_review(prepared, client)
    assert client.calls == [scope.request for scope in prepared.requests]
    assert result.version == review.VERSION
    assert result.complete and result.attempted_calls == 3 and result.model_no_defect
    assert not result.semantic_verified and not result.publication_authorized
    assert all(not outcome.semantic_verified and not outcome.publication_authorized
               for outcome in result.outcomes)


def test_malformed_replies_continue_but_client_failure_halts_sanitized_without_retry():
    prepared = plan()
    client = Scripted(["malformed", RuntimeError("SECRET DO NOT EXPOSE")])
    result = review.execute_record_review(prepared, client)
    assert result.attempted_calls == 2 and not result.complete
    assert result.outcomes[0].status == "malformed_review"
    assert result.outcomes[1].status == "execution_error"
    assert result.halted_reason == "client_exception"
    assert "SECRET" not in repr(result)


@pytest.mark.parametrize("exception", [DeadlineExceeded("deadline"), KeyboardInterrupt(), SystemExit(7)])
def test_original_deadline_and_process_interrupts_propagate(exception):
    client = Scripted([exception])
    with pytest.raises(type(exception)) as caught:
        review.execute_record_review(plan(), client)
    assert caught.value is exception and len(client.calls) == 1


def test_inherited_policy_change_fails_closed(monkeypatch):
    monkeypatch.setattr(source_review, "_system", lambda *args: "different contract")
    with pytest.raises(ValueError):
        plan()
