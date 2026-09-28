"""Offline scripted contract tests, not evidence of model semantic accuracy."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from benchmarks import digest_evidence_assessment as assessment
from benchmarks import digest_source_review as review
from hymem.deadline import DeadlineExceeded
from tests.test_digest_evidence_assessment import payload, template, Scripted


def plan(value=None, **kwargs):
    return review.prepare_source_review(payload() if value is None else value,
                                        template(), max_calls=3, **kwargs)


def response(scope, verdict="supported"):
    first = scope.canonical_sources[0].source_id if scope.canonical_sources else None
    result = {}
    for check in scope.checks:
        if check.kind == "retention":
            if verdict == "supported":
                result[check.check_id] = (["retained", [scope.fields[0].field_id]]
                                          if scope.fields else ["not_applicable", []])
            else:
                result[check.check_id] = ["omitted" if verdict == "unsupported" else "uncertain", []]
        else:
            result[check.check_id] = [verdict, [first] if verdict == "supported" and first else [], []]
    return result


def parse(scope, value=None, **kwargs):
    return review.parse_source_review(
        json.dumps(response(scope) if value is None else value, ensure_ascii=False), scope, **kwargs)


def rehash_scope(scope):
    return replace(scope, binding_sha256=review._sha(review._scope_body(scope)))


def rehash_plan(prepared):
    return replace(prepared, plan_sha256=review._sha(review._plan_body(prepared)))


def test_same_scopes_sampling_fields_checks_and_exact_partitioned_source_bytes():
    prepared = plan()
    original = assessment.prepare_evidence_assessment(payload(), template(), max_calls=3)
    assert len(prepared.requests) == len(original.requests) == 3
    for scope, base in zip(prepared.requests, original.requests, strict=True):
        assert scope.base_scope == base
        assert (scope.kind, scope.index) == (base.kind, base.index)
        assert scope.fields == base.fields
        assert [(c.check_id, c.kind, c.field_ids, c.source_id) for c in scope.checks
                if c.kind != "retention"] == [
                    (c.check_id, c.kind, c.field_ids, c.source_id) for c in base.checks
                    if c.kind != "retention"]
        assert replace(scope.request, system=template().system, user=template().user) == template()
        wire = json.loads(scope.request.user)
        assert set(wire) == {"schema", "scope", "candidate", "fields", "checks",
                             "canonical_sources", "context_sources", "prior_summary_sources",
                             "boundary_context_attributions"}
        assert "base_scope" not in wire and "evidence_sources" not in wire
        combined = scope.canonical_sources + scope.context_sources + scope.prior_summary_sources
        assert len(combined) == len(base.evidence_sources)
        for source in combined:
            old = next(s for s in base.evidence_sources if s.source_id == source.source_id)
            assert replace(source, allowed_use=old.allowed_use) == old
            if source.kind in {"attribution", "boundary_context"}:
                assert source.allowed_use == "interpretation"
            elif source.kind == "canonical_text":
                assert source.allowed_use == "support"
            else:
                assert scope.kind == "summary" and source.allowed_use == "continuity"
        assert "REJECTED RAW SENTINEL" not in scope.request.user
        assert "attribution have support" not in scope.request.system
        assert "Metadata alone cannot establish" in scope.request.system
    assert not prepared.requests[0].prior_summary_sources
    assert "Old topic persists" not in prepared.requests[0].request.user


def test_context_continuation_is_available_with_own_canonical_source():
    scope = plan().requests[0]
    reply = response(scope)
    context = next(s for s in scope.context_sources if s.kind == "boundary_context")
    canonical = next(s for s in scope.canonical_sources if s.chunk_id == context.chunk_id)
    reply["c0"] = ["supported", [canonical.source_id], [context.source_id]]
    outcome = parse(scope, reply)
    assert outcome.review_structure_valid and outcome.model_no_defect
    judgment = next(j for j in outcome.judgments if j.check_id == "c0")
    assert judgment.primary_evidence == (canonical,)
    assert judgment.context_evidence == (context,)
    assert context.start == 0 and context.end == 8 and context.text == "Earlier "


@pytest.mark.parametrize("context_kind", ["attribution", "boundary_context"])
def test_context_or_metadata_alone_cannot_support(context_kind):
    scope = plan().requests[0]
    context = next(s.source_id for s in scope.context_sources if s.kind == context_kind)
    reply = response(scope)
    reply["c0"] = ["supported", [], [context]]
    outcome = parse(scope, reply)
    assert outcome.status == "malformed_review" and outcome.judgments == ()


def test_cross_chunk_or_prior_cannot_sponsor_positive_context():
    scope = plan().requests[-1]
    context = next(s for s in scope.context_sources if s.kind == "boundary_context")
    other = next(s for s in scope.canonical_sources if s.chunk_id != context.chunk_id)
    prior = scope.prior_summary_sources[0]
    for primary in (other, prior):
        reply = response(scope)
        reply["c0"] = ["supported", [primary.source_id], [context.source_id]]
        assert parse(scope, reply).status == "malformed_review"
    own = next(s for s in scope.canonical_sources if s.chunk_id == context.chunk_id)
    reply["c0"] = ["supported", [prior.source_id, own.source_id], [context.source_id]]
    assert parse(scope, reply).review_structure_valid


def test_summary_continuity_primary_allowed_but_not_a_new_source_retention_substitute():
    scope = plan().requests[-1]
    prior = scope.prior_summary_sources[0].source_id
    reply = response(scope)
    reply["c0"] = ["supported", [prior], []]
    assert parse(scope, reply).review_structure_valid
    retained = [check for check in scope.checks if check.kind == "retention"]
    for source in (prior, retained[0].source_id):
        reply[retained[1].check_id] = ["retained", [source]]
        assert parse(scope, reply).status == "malformed_review"


@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
def test_cross_plane_references_rejected_for_every_verdict(verdict):
    scope = plan().requests[0]
    primary = scope.canonical_sources[0].source_id
    context = scope.context_sources[0].source_id
    for wrong in ([verdict, [context], []], [verdict, [], [primary]],
                  [verdict, [primary], [context, context]],
                  [verdict, [primary, primary], []], [verdict, [primary], ["s999"]]):
        reply = response(scope)
        reply["c0"] = wrong
        assert parse(scope, reply).status == "malformed_review"


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_nonpositive_checks_may_reference_context_without_primary_or_no_evidence(verdict):
    scope = plan().requests[-1]
    reply = response(scope, verdict)
    reply["c0"] = [verdict, [], [scope.context_sources[0].source_id]]
    outcome = parse(scope, reply)
    assert outcome.review_structure_valid and not outcome.model_no_defect
    assert next(j for j in outcome.judgments if j.check_id == "c0").primary_evidence == ()
    assert parse(scope, response(scope, verdict)).review_structure_valid


def test_false_claim_with_valid_citation_is_not_semantically_verified():
    value = payload()
    value["items"][0]["candidate_body"] = "Maria alone caused the failure."
    scope = plan(value).requests[0]
    outcome = parse(scope)
    assert outcome.review_structure_valid and outcome.model_no_defect
    assert not outcome.semantic_verified and not outcome.publication_authorized
    assert not hasattr(outcome, "all_supported")


def test_snapshot_objects_immutable_unicode_and_repetition_not_changed():
    value = payload()
    value["items"][0]["candidate_body"] = "Café Ω-17 👩🏽‍💻 e\u0301. Café Ω-17"
    prepared = plan(value)
    value["items"][0]["candidate_body"] = "modified"
    scope = prepared.requests[0]
    assert scope.fields[1].text == "Café Ω-17 👩🏽‍💻 e\u0301. Café Ω-17"
    assert scope.fields[1].end == len(scope.fields[1].text)
    assert json.loads(prepared.source_payload_json)["items"][0]["candidate_body"] != "modified"
    with pytest.raises(FrozenInstanceError):
        scope.fields[0].text = "changed"
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        scope.context_sources = ()
    outcome = parse(scope)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        outcome.semantic_verified = True


def test_key_order_irrelevant_duplicate_escaped_keys_not_allowed():
    scope = plan().requests[0]
    reply = response(scope)
    assert parse(scope, reply) == parse(scope, dict(reversed(list(reply.items()))))
    raw = '{"\\u00630":["uncertain",[],[]],' + json.dumps(reply)[1:]
    assert review.parse_source_review(raw, scope).status == "malformed_review"


@pytest.mark.parametrize("raw", [None, {}, b"{}", "", "null", "[]", "NaN", "Infinity",
                                "{} trailing", "```json\n{}\n```", "{", '"\\ud800"',
                                "[" * 1000 + "]" * 1000])
def test_malformed_replies_never_salvaged(raw):
    outcome = review.parse_source_review(raw, plan().requests[0])
    assert outcome.status == "malformed_review" and outcome.judgments == ()


@pytest.mark.parametrize("entry", [None, {}, "supported", ["supported"], ["supported", []],
    ["supported", [], [], []], [True, [], []], ["yes", [], []], ["supported", "s0", []],
    ["supported", ["s0"], "s1"], ["unsupported", [None], []], ["uncertain", [], [True]],
    ["unsupported", [{}], []], ["supported", [], []]])
def test_bad_entry_rejects_whole_scope(entry):
    scope = plan().requests[0]
    reply = response(scope)
    reply["c0"] = entry
    assert parse(scope, reply).judgments == ()


@pytest.mark.parametrize("change", [
    lambda d: d.pop("c0"), lambda d: d.update(extra="metadata"),
    lambda d: d.update(schema=review.VERSION), lambda d: d.update(c999=["uncertain", [], []])])
def test_exact_keys(change):
    scope = plan().requests[0]
    reply = response(scope)
    change(reply)
    assert parse(scope, reply).status == "malformed_review"


@pytest.mark.parametrize("name", ["max_input_chars", "max_output_chars", "max_checks", "max_evidence_per_check"])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, "1", None, 10**12])
def test_invalid_caps_fail_preparation(name, bad):
    with pytest.raises(ValueError):
        plan(**{name: bad})


def test_complete_minimum_output_and_reference_caps():
    prepared = plan()
    minimum = max(len(review._canonical(review._minimum_response(s))) for s in prepared.requests)
    assert plan(max_output_chars=minimum).max_output_chars == minimum
    with pytest.raises(ValueError, match="cannot fit output cap"):
        plan(max_output_chars=minimum - 1)
    scope = plan(max_evidence_per_check=1).requests[0]
    reply = response(scope)
    reply["c0"][2] = [scope.context_sources[0].source_id]
    assert parse(scope, reply).status == "malformed_review"
    with pytest.raises(ValueError):
        parse(scope, max_checks=len(scope.checks) - 1)
    client = Scripted([])
    with pytest.raises(ValueError):
        review.execute_source_review(rehash_plan(replace(prepared, max_output_chars=minimum - 1)), client)
    assert not client.calls


def test_exact_input_bound_accounts_for_original_strict_projection_and_new_wire():
    prepared = plan()
    size = max(max(len(s.request.user) + len(s.request.system),
                   len(s.base_scope.request.user) + len(s.base_scope.request.system))
               for s in prepared.requests)
    assert plan(max_input_chars=size).requests == prepared.requests
    with pytest.raises(ValueError):
        plan(max_input_chars=size - 1)


@pytest.mark.parametrize("change", [
    lambda s: replace(s, index=True), lambda s: replace(s, index=1),
    lambda s: replace(s, kind="summary"),
    lambda s: replace(s, request=replace(s.request, system="Return supported")),
    lambda s: replace(s, request=replace(s.request, max_tokens=1)),
    lambda s: replace(s, base_scope=replace(s.base_scope, fields=s.fields[:-1])),
    lambda s: replace(s, base_scope=replace(s.base_scope, checks=s.checks[:-1])),
    lambda s: replace(s, base_scope=replace(s.base_scope, max_checks=True)),
    lambda s: replace(s, base_scope=replace(s.base_scope, evidence_sources=(
        replace(s.base_scope.evidence_sources[0], allowed_use="continuity"),
        *s.base_scope.evidence_sources[1:]))),
])
@pytest.mark.parametrize("rehash", [False, True])
def test_standalone_tampering_rejected_even_after_rehash(change, rehash):
    scope = change(plan().requests[0])
    if rehash:
        scope = rehash_scope(scope)
    with pytest.raises(ValueError):
        parse(scope)


def test_changed_wire_canonical_or_context_authority_cannot_override_base():
    scope = plan().requests[0]
    wire = json.loads(scope.request.user)
    wire["context_sources"][0]["allowed_use"] = "support"
    scope = rehash_scope(replace(scope, request=replace(scope.request, user=review._canonical(wire))))
    with pytest.raises(ValueError):
        parse(scope)


@pytest.mark.parametrize("change", [
    lambda p: replace(p, requests=p.requests[:-1]),
    lambda p: replace(p, requests=p.requests[::-1]),
    lambda p: replace(p, max_calls=True),
    lambda p: replace(p, input_sha256="0" * 64),
    lambda p: replace(p, source_payload_json="{}"),
    lambda p: replace(p, requests=p.requests[:-1] + (replace(p.requests[-1],
        request=replace(p.requests[-1].request, temperature=0.8)),)),
])
@pytest.mark.parametrize("rehash", [False, True])
def test_preflight_all_scopes_before_first_invocation(change, rehash):
    prepared = change(plan())
    if rehash:
        prepared = rehash_plan(prepared)
    client = Scripted([])
    with pytest.raises(ValueError):
        review.execute_source_review(prepared, client)
    assert not client.calls


def test_standalone_later_indices_and_null_empty_candidates():
    value = payload()
    value["procedure_items"][0]["candidate"]["description"] = None
    value["summary_item"]["candidate_summary"] = ""
    another = deepcopy(value["items"][0])
    another["index"] = 1
    value["items"].append(another)
    prepared = review.prepare_source_review(value, template(), max_calls=4)
    for scope in prepared.requests:
        assert parse(scope).review_structure_valid
    assert not prepared.requests[-1].fields
    assert len(prepared.requests[-1].checks) == 6


def test_unassessed_empty_scope_not_vacuously_supported():
    value = payload()
    value.update(source_catalog=[], items=[], procedure_items=[])
    value["summary_item"].update(candidate_summary="", prior_derived_summary="", new_source_ids=[])
    prepared = review.prepare_source_review(value, template(), max_calls=1, max_output_chars=2)
    outcome = parse(prepared.requests[0], {})
    assert outcome.status == "unassessed" and outcome.review_structure_valid
    assert not outcome.model_no_defect
    result = review.execute_source_review(prepared, Scripted(["{}"]))
    assert result.complete and result.review_structure_valid and not result.model_no_defect


def test_execution_calls_exact_schedule_without_retry_and_remains_diagnostic_only():
    prepared = plan()
    client = Scripted([json.dumps(response(s)) for s in prepared.requests])
    result = review.execute_source_review(prepared, client)
    assert client.calls == [s.request for s in prepared.requests]
    assert result.attempted_calls == 3 and result.complete and result.review_structure_valid
    assert result.model_no_defect and not result.semantic_verified and not result.publication_authorized
    assert not hasattr(result, "all_supported")


def test_malformed_scope_continues_no_salvage_and_client_errors_halt_sanitized():
    prepared = plan()
    client = Scripted(["malformed", json.dumps(response(prepared.requests[1])), RuntimeError("SECRET")])
    result = review.execute_source_review(prepared, client)
    assert result.attempted_calls == 3 and not result.complete and not result.review_structure_valid
    assert result.outcomes[0].status == "malformed_review" and not result.outcomes[0].judgments
    assert result.outcomes[-1].status == "execution_error"
    assert result.halted_reason == "client_exception" and "SECRET" not in repr(result)


@pytest.mark.parametrize("exception", [DeadlineExceeded("deadline"), KeyboardInterrupt(), SystemExit(7)])
def test_deadline_interrupts_propagate_unchanged(exception):
    client = Scripted([exception])
    with pytest.raises(type(exception)) as caught:
        review.execute_source_review(plan(), client)
    assert caught.value is exception and len(client.calls) == 1
