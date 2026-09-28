"""Scripted, offline judgments test contracts, never live semantic accuracy."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from benchmarks import digest_evidence_assessment as assessment
from benchmarks import digest_evidence_isolation as isolation
from benchmarks import digest_evidence_ledger as ledger
from hymem.deadline import DeadlineExceeded
from hymem.extraction.llm import LLMRequest


def payload():
    first = "Cedar is available on Aurora. Cedar is not exclusive."
    second = "Inspect cache, then verify cache. Never delete cache."
    return {
        "schema": isolation.INPUT_VERSION,
        "source_catalog": [
            {"chunk_id": "chunk-a", "message_id": 1, "role": "user",
             "source_peer_id": "peer-a", "source_workspace_id": "space-a",
             "start": 8, "end": 8 + len(first), "visible_content": first,
             "interpretation_only_context": {"message_id": 1, "role": "user",
                 "source_peer_id": "peer-a", "source_workspace_id": "space-a",
                 "start": 0, "end": 8, "content": "Earlier "}},
            {"chunk_id": "chunk-b", "message_id": 2, "role": "assistant",
             "source_peer_id": None, "source_workspace_id": "",
             "start": 0, "end": len(second), "visible_content": second,
             "interpretation_only_context": None}],
        "items": [{"index": 0, "candidate_title": "Cedar availability",
                   "candidate_body": "Cedar is exclusive to Aurora.",
                   "candidate_outcome": "informational", "candidate_key_entities": ["Cedar", "Cedar"],
                   "cited_source_ids": ["chunk-a"]}],
        "procedure_items": [{"index": 0, "candidate": {
            "name": "Cache maintenance", "description": "Inspect safely.",
            "steps": [{"order": 1, "action": "Inspect cache", "tool": "scanner"},
                      {"order": 2, "action": "Verify cache", "tool": None}],
            "triggers": ["cache repair", "cache repair"], "entities_involved": ["cache"]},
            "cited_source_ids": ["chunk-b"]}],
        "summary_item": {"index": 0, "candidate_raw_summary": "REJECTED RAW SENTINEL",
                         "candidate_summary": "Cedar is available; inspect cache.",
                         "candidate_is_noop": False, "new_source_ids": ["chunk-a", "chunk-b"],
                         "prior_derived_summary": "Old topic persists."}}


def template():
    return LLMRequest("old system", "old user", "json", 3072, 0.4)


def plan(value=None, **kwargs):
    return assessment.prepare_evidence_assessment(payload() if value is None else value,
                                                   template(), max_calls=3, **kwargs)


def response(scope, verdict="supported"):
    source = next((part.source_id for part in scope.evidence_sources
                   if part.allowed_use == "support"), None)
    return {check.check_id: [verdict,
            [check.source_id if check.kind == "retention" else source]
            if verdict == "supported" and (source or check.source_id) else []]
            for check in scope.checks}


def parse(scope, value=None, **kwargs):
    return assessment.parse_evidence_assessment(
        json.dumps(response(scope) if value is None else value, ensure_ascii=False), scope, **kwargs)


class Scripted:
    def __init__(self, values):
        self.values, self.calls = iter(values), []

    def complete(self, request):
        self.calls.append(request)
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def rehash_scope(scope):
    return replace(scope, binding_sha256=assessment._sha(assessment._scope_body(scope)))


def rehash_plan(value):
    return replace(value, plan_sha256=assessment._sha(assessment._plan_body(value)))


def test_projection_same_as_existing_ledger_with_fixed_metadata_only():
    prepared = plan()
    legacy = ledger.prepare_evidence_ledger(payload(), template(), max_calls=3)
    assert [(scope.kind, scope.index) for scope in prepared.requests] == [
        ("episode", 0), ("procedure", 0), ("summary", 0)]
    for scope, old in zip(prepared.requests, legacy.requests, strict=True):
        assert replace(scope.request, system=template().system, user=template().user) == template()
        assert [(field.path, field.text) for field in scope.fields] == [
            (field.path, field.text) for field in old.fields]
        assert scope.evidence_sources == old.evidence_sources
        assert all(field.start == 0 and field.end == len(field.text) for field in scope.fields)
        assert [field.field_id for field in scope.fields] == [f"f{i}" for i in range(len(scope.fields))]
        assert "REJECTED RAW SENTINEL" not in scope.request.user
        assert "REJECTED RAW SENTINEL" not in scope.source_projection_json
    assert "Old topic persists" not in prepared.requests[0].request.user
    assert "Never delete cache" not in prepared.requests[0].request.user
    assert "Old topic persists" in prepared.requests[-1].request.user


def test_exact_obligation_schedule_includes_outcome_and_source_retention():
    for scope in plan().requests:
        assert [check.check_id for check in scope.checks] == [f"c{i}" for i in range(len(scope.checks))]
        for field in scope.fields:
            checks = [check for check in scope.checks
                      if check.field_ids == (field.field_id,) and check.kind != "retention"]
            if field.path == "/candidate_outcome":
                assert [check.kind for check in checks] == ["outcome"]
            else:
                assert [check.kind for check in checks] == ["assertion", "relations"]
            assert all(check.source_id is None for check in checks)
        retained = [check for check in scope.checks if check.kind == "retention"]
        assert [check.source_id for check in retained] == [
            source.source_id for source in scope.evidence_sources if source.kind == "canonical_text"]
        assert all(check.field_ids == tuple(field.field_id for field in scope.fields) for check in retained)


def test_null_empty_whitespace_unicode_repeated_names_preserved():
    value = payload()
    candidate = value["procedure_items"][0]["candidate"]
    candidate["description"] = ""
    candidate["steps"][1]["tool"] = " \n\t"
    value["items"][0]["candidate_body"] = "Café 👩🏽‍💻 e\u0301.\n否。  Cedar Cedar"
    value["summary_item"]["candidate_summary"] = " \n\t"
    prepared = plan(value)
    assert json.loads(prepared.requests[1].request.user)["candidate"]["candidate"]["description"] == ""
    assert all(field.path != "/candidate/description" for field in prepared.requests[1].fields)
    assert next(field for field in prepared.requests[1].fields
                if field.path == "/candidate/steps/1/tool").text == " \n\t"
    for scope in prepared.requests:
        result = parse(scope)
        assert result.assessment_structure_valid
        assert not result.semantic_verified and not result.publication_authorized
        assert all(field.text == field.text[field.start:field.end] for field in scope.fields)
    repeated = [field for field in prepared.requests[0].fields if field.text == "Cedar"]
    assert len(repeated) == 2 and repeated[0].field_id != repeated[1].field_id


def test_snapshot_and_resolved_objects_immutable_and_exact():
    value = payload()
    prepared = plan(value)
    value["items"][0]["candidate_body"] = "Changed"
    assert prepared == plan()
    scope = prepared.requests[0]
    result = parse(scope)
    first = result.judgments[0]
    assert first.fields[0] is scope.fields[0]
    assert first.evidence[0] is scope.evidence_sources[0]
    assert first.evidence[0].start == 8
    assert first.evidence[0].end == 8 + len(first.evidence[0].text)
    with pytest.raises(FrozenInstanceError):
        scope.fields[0].text = "Changed"
    with pytest.raises(FrozenInstanceError):
        first.evidence[0].text = "Changed"
    with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
        result.semantic_verified = True
    with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
        result.publication_authorized = True


def test_json_key_order_irrelevant_but_judgment_order_is_deterministic():
    scope = plan().requests[0]
    value = response(scope)
    reversed_value = dict(reversed(list(value.items())))
    assert parse(scope, value) == parse(scope, reversed_value)


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_negative_judgments_need_no_evidence_but_remain_nonaffirmative(verdict):
    scope = plan().requests[0]
    result = parse(scope, response(scope, verdict))
    assert result.status == "valid_assessment" and not result.model_all_supported
    assert all(judgment.evidence == () and judgment.verdict == verdict for judgment in result.judgments)


def test_scripted_false_exclusivity_identity_and_causality_are_not_semantic_proof():
    value = payload()
    value["items"][0]["candidate_body"] = "Maria alone caused the failure. Cedar is exclusive."
    scope = plan(value).requests[0]
    result = parse(scope)
    # Authorized exact sources cannot establish these invented assertions.
    assert result.model_all_supported and result.assessment_structure_valid
    assert not result.semantic_verified and not result.publication_authorized
    assert not hasattr(result, "all_supported")


@pytest.mark.parametrize("outcome", ["resolved", "blocked", "deferred", "informational"])
def test_outcome_labels_use_semantic_classification_not_literal_match(outcome):
    value = payload()
    value["items"][0]["candidate_outcome"] = outcome
    scope = plan(value).requests[0]
    check = next(check for check in scope.checks if check.kind == "outcome")
    reply = response(scope)
    reply[check.check_id] = ["unsupported", []]
    result = parse(scope, reply)
    assert result.assessment_structure_valid and not result.model_all_supported
    assert next(j for j in result.judgments if j.kind == "outcome").fields[0].text == outcome
    assert "event, not literal word occurrence" in scope.request.system
    assert "Never infer completion" in scope.request.system


def test_material_source_omission_has_independent_retention_veto():
    scope = plan().requests[1]
    reply = response(scope)
    check = next(check for check in scope.checks if check.kind == "retention")
    # Candidate covers its own assertions but omits source's explicit prohibition.
    reply[check.check_id] = ["unsupported", [check.source_id]]
    result = parse(scope, reply)
    assert result.assessment_structure_valid and not result.model_all_supported
    assert all(j.verdict == "supported" for j in result.judgments if j.kind != "retention")
    assert "Never delete cache" in next(j for j in result.judgments if j.kind == "retention").evidence[0].text


def test_context_alone_cannot_support_any_field_but_may_interpret():
    scope = plan().requests[0]
    context = next(source.source_id for source in scope.evidence_sources if source.kind == "boundary_context")
    reply = response(scope)
    reply["c0"] = ["supported", [context]]
    assert not parse(scope, reply).assessment_structure_valid
    reply["c0"][1].append("s0")
    assert parse(scope, reply).assessment_structure_valid


def test_prior_supports_summary_continuity_but_cannot_replace_retention():
    scope = plan().requests[-1]
    prior = next(source.source_id for source in scope.evidence_sources if source.kind == "prior_summary")
    reply = response(scope)
    reply["c0"] = ["supported", [prior]]
    assert parse(scope, reply).assessment_structure_valid
    check = next(check for check in scope.checks if check.kind == "retention")
    reply[check.check_id] = ["supported", [prior]]
    assert not parse(scope, reply).assessment_structure_valid


def test_each_retention_requires_own_source_not_another_canonical_unit():
    scope = plan().requests[-1]
    reply = response(scope)
    checks = [check for check in scope.checks if check.kind == "retention"]
    assert len(checks) == 2
    reply[checks[1].check_id] = ["supported", [checks[0].source_id]]
    assert not parse(scope, reply).assessment_structure_valid


def test_empty_candidate_still_has_new_source_outcome_checks():
    value = payload()
    value["summary_item"]["candidate_summary"] = ""
    scope = plan(value).requests[-1]
    assert not scope.fields
    assert len(scope.checks) == 2 and all(check.kind == "retention" for check in scope.checks)
    assert not parse(scope, {}).assessment_structure_valid
    result = parse(scope, response(scope, "unsupported"))
    assert result.assessment_structure_valid and not result.model_all_supported


def test_empty_check_set_is_unassessed_never_vacuously_supported():
    value = payload()
    value.update(source_catalog=[], items=[], procedure_items=[])
    value["summary_item"].update(candidate_summary="", prior_derived_summary="", new_source_ids=[])
    prepared = assessment.prepare_evidence_assessment(value, template(), max_calls=1)
    scope = prepared.requests[0]
    assert scope.checks == scope.fields == scope.evidence_sources == ()
    outcome = parse(scope, {})
    assert outcome.status == "unassessed" and outcome.assessment_structure_valid
    assert not outcome.model_all_supported
    result = assessment.execute_evidence_assessment(prepared, Scripted(["{}"]))
    assert result.complete and result.assessment_structure_valid and not result.model_all_supported


@pytest.mark.parametrize("raw", [None, {}, b"{}", "", "null", "[]", "true", "NaN", "Infinity",
                                "{} trailing", "```json\n{}\n```", "{", '"text"', '"\\ud800"',
                                "[" * 1000 + "]" * 1000])
def test_malformed_or_nonjson_replies_are_rejected_without_salvage(raw):
    result = assessment.parse_evidence_assessment(raw, plan().requests[0])
    assert result.status == "malformed_assessment" and result.judgments == ()
    assert not result.model_all_supported and not result.semantic_verified


@pytest.mark.parametrize("entry", [None, {}, "supported", ["supported"], ["supported", [], None],
                                  [True, []], [1, []], ["yes", []], ["Supported", []],
                                  ["supported", "s0"], ["supported", {}], ["supported", []],
                                  ["unsupported", [None]], ["uncertain", [True]],
                                  ["uncertain", [0]], ["uncertain", [{}]],
                                  ["supported", ["s999"]], ["unsupported", ["s999"]],
                                  ["uncertain", ["s999"]], ["supported", ["s0", "s0"]]])
def test_bad_check_shapes_authority_types_or_duplicates_invalidate_scope(entry):
    scope = plan().requests[0]
    reply = response(scope)
    reply["c0"] = entry
    result = parse(scope, reply)
    assert result.status == "malformed_assessment" and result.judgments == ()


@pytest.mark.parametrize("mutator", [
    lambda reply: reply.pop("c0"), lambda reply: reply.update(extra="metadata"),
    lambda reply: reply.update(schema=assessment.VERSION),
    lambda reply: reply.update(scope={"kind": "episode", "index": 0}),
    lambda reply: reply.update(c999=["supported", ["s0"]])])
def test_exact_flat_keys_no_envelopes(mutator):
    scope = plan().requests[0]
    reply = response(scope)
    mutator(reply)
    assert not parse(scope, reply).assessment_structure_valid


def test_duplicate_json_keys_even_escaped_are_rejected():
    scope = plan().requests[0]
    raw = json.dumps(response(scope))
    raw = '{"\\u00630":["supported",["s0"]],' + raw[1:]
    assert assessment.parse_evidence_assessment(raw, scope).status == "malformed_assessment"


@pytest.mark.parametrize("name,maximum", [("max_checks", assessment.MAX_CHECKS),
    ("max_evidence_per_check", assessment.MAX_EVIDENCE_PER_CHECK),
    ("max_input_chars", assessment.MAX_INPUT_CHARS),
    ("max_output_chars", assessment.MAX_OUTPUT_CHARS)])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, "1", None, 10**12])
def test_invalid_limits_rejected_before_any_execution(name, maximum, bad):
    with pytest.raises(ValueError):
        plan(**{name: bad})


@pytest.mark.parametrize("name", ["max_checks", "max_output_chars", "max_evidence_per_check"])
@pytest.mark.parametrize("bad", [True, 0, 1.0, None])
def test_parser_rejects_invalid_caller_limits(name, bad):
    with pytest.raises(ValueError):
        parse(plan().requests[0], **{name: bad})


def test_exact_input_output_and_check_caps():
    prepared = plan()
    input_size = max(len(scope.request.system) + len(scope.request.user) for scope in prepared.requests)
    assert plan(max_input_chars=input_size).requests == prepared.requests
    with pytest.raises(ValueError):
        plan(max_input_chars=input_size - 1)
    check_size = max(len(scope.checks) for scope in prepared.requests)
    assert len(plan(max_checks=check_size).requests) == 3
    with pytest.raises(ValueError):
        plan(max_checks=check_size - 1)
    scope = prepared.requests[0]
    raw = json.dumps(response(scope))
    assert assessment.parse_evidence_assessment(raw, scope, max_output_chars=len(raw)).assessment_structure_valid
    assert not assessment.parse_evidence_assessment(raw, scope, max_output_chars=len(raw) - 1).assessment_structure_valid
    with pytest.raises(ValueError):
        parse(scope, max_checks=len(scope.checks) - 1)


def test_evidence_limit_is_bound_by_scope_even_if_parser_default_is_larger():
    scope = plan(max_evidence_per_check=1).requests[0]
    reply = response(scope)
    assert parse(scope, reply).assessment_structure_valid
    reply["c0"][1].append("s1")
    assert not parse(scope, reply).assessment_structure_valid


def test_complete_minimum_output_cap_exact_boundary_and_preflight():
    prepared = plan()
    minimum = max(len(assessment._canonical(response(scope, "uncertain")))
                  for scope in prepared.requests)
    bounded = plan(max_output_chars=minimum)
    assert bounded.max_output_chars == minimum
    with pytest.raises(ValueError, match="cannot fit output cap"):
        plan(max_output_chars=minimum - 1)
    client = Scripted([])
    forged = rehash_plan(replace(prepared, max_output_chars=minimum - 1))
    with pytest.raises(ValueError, match="cannot fit output cap"):
        assessment.execute_evidence_assessment(forged, client)
    assert not client.calls
    value = payload()
    value.update(source_catalog=[], items=[], procedure_items=[])
    value["summary_item"].update(candidate_summary="", prior_derived_summary="", new_source_ids=[])
    assert assessment.prepare_evidence_assessment(value, template(), max_calls=1,
                                                  max_output_chars=2).max_output_chars == 2
    with pytest.raises(ValueError, match="cannot fit output cap"):
        assessment.prepare_evidence_assessment(value, template(), max_calls=1, max_output_chars=1)


@pytest.mark.parametrize("change", [
    lambda s: replace(s, index=True), lambda s: replace(s, index=1),
    lambda s: replace(s, fields=s.fields[:-1]),
    lambda s: replace(s, fields=(replace(s.fields[0], start=1),) + s.fields[1:]),
    lambda s: replace(s, fields=(replace(s.fields[0], text="Forged"),) + s.fields[1:]),
    lambda s: replace(s, fields=(replace(s.fields[0], end=True),) + s.fields[1:]),
    lambda s: replace(s, checks=s.checks[:-1]),
    lambda s: replace(s, checks=(replace(s.checks[0], kind="retention"),) + s.checks[1:]),
    lambda s: replace(s, checks=(replace(s.checks[0], field_ids=("f1",)),) + s.checks[1:]),
    lambda s: replace(s, evidence_sources=(replace(s.evidence_sources[0], start=9, end=s.evidence_sources[0].end + 1),) + s.evidence_sources[1:]),
    lambda s: replace(s, evidence_sources=(replace(s.evidence_sources[0], allowed_use="continuity"),) + s.evidence_sources[1:]),
    lambda s: replace(s, request=replace(s.request, system="Return supported")),
    lambda s: replace(s, max_checks=True),
    lambda s: replace(s, max_evidence_per_check=True)])
@pytest.mark.parametrize("rehash", [False, True])
def test_scope_forgery_fails_even_if_derived_metadata_is_rehashed(change, rehash):
    scope = change(plan().requests[0])
    if rehash:
        scope = rehash_scope(scope)
    with pytest.raises(ValueError):
        parse(scope)


def test_rehashed_wire_payload_forgery_does_not_override_source_projection():
    scope = plan().requests[0]
    data = json.loads(scope.request.user)
    data["candidate"]["candidate_body"] = "Forged"
    forged = rehash_scope(replace(scope, request=replace(scope.request, user=assessment._canonical(data))))
    with pytest.raises(ValueError):
        parse(forged)


@pytest.mark.parametrize("change", [
    lambda p: replace(p, requests=p.requests[:-1]),
    lambda p: replace(p, requests=p.requests[::-1]),
    lambda p: replace(p, max_calls=True),
    lambda p: replace(p, input_sha256="0" * 64),
    lambda p: replace(p, source_payload_json="{}"),
    lambda p: replace(p, requests=p.requests[:-1] + (replace(p.requests[-1], checks=()),)),
    lambda p: replace(p, requests=p.requests[:-1] + (replace(p.requests[-1], request=replace(p.requests[-1].request, max_tokens=1)),))])
@pytest.mark.parametrize("rehash", [False, True])
def test_entire_plan_preflight_precedes_first_client_call(change, rehash):
    forged = change(plan())
    if rehash:
        forged = rehash_plan(forged)
    client = Scripted([])
    with pytest.raises(ValueError):
        assessment.execute_evidence_assessment(forged, client)
    assert client.calls == []


def test_invalid_binding_digests_rejected_without_replacing_them():
    prepared = plan()
    with pytest.raises(ValueError):
        parse(replace(prepared.requests[0], binding_sha256="0" * 64))
    client = Scripted([])
    with pytest.raises(ValueError):
        assessment.execute_evidence_assessment(replace(prepared, plan_sha256="0" * 64), client)
    assert not client.calls


def test_later_scope_indices_standalone_reconstruction():
    value = payload()
    for family in ("items", "procedure_items"):
        another = deepcopy(value[family][0])
        another["index"] = 1
        value[family].append(another)
    prepared = assessment.prepare_evidence_assessment(value, template(), max_calls=5)
    for scope in prepared.requests:
        assert parse(scope).assessment_structure_valid


def test_complete_execution_is_diagnostic_only_no_retry_or_hidden_calls():
    prepared = plan()
    client = Scripted([json.dumps(response(scope)) for scope in prepared.requests])
    result = assessment.execute_evidence_assessment(prepared, client)
    assert client.calls == [scope.request for scope in prepared.requests]
    assert result.attempted_calls == 3 and result.complete and result.assessment_structure_valid
    assert result.model_all_supported and result.halted_reason is None
    assert not result.semantic_verified and not result.publication_authorized
    assert all(not outcome.semantic_verified and not outcome.publication_authorized for outcome in result.outcomes)


def test_malformed_scope_continues_but_never_partial_salvage():
    prepared = plan()
    client = Scripted(["malformed", *[json.dumps(response(scope)) for scope in prepared.requests[1:]]])
    result = assessment.execute_evidence_assessment(prepared, client)
    assert len(client.calls) == 3 and result.complete
    assert not result.assessment_structure_valid and not result.model_all_supported
    assert result.outcomes[0].judgments == ()


def test_client_error_halts_sanitized_without_retry_or_later_calls():
    prepared = plan()
    client = Scripted([json.dumps(response(prepared.requests[0])), RuntimeError("SECRET RAW BODY")])
    result = assessment.execute_evidence_assessment(prepared, client)
    assert len(client.calls) == result.attempted_calls == 2
    assert not result.complete and not result.model_all_supported
    assert result.halted_reason == "client_exception" and result.outcomes[-1].status == "execution_error"
    assert "SECRET RAW BODY" not in repr(result)


@pytest.mark.parametrize("exception", [DeadlineExceeded("deadline"), KeyboardInterrupt(), SystemExit(7)])
def test_deadlines_and_process_interrupts_propagate_unchanged(exception):
    prepared = plan()
    client = Scripted([exception])
    with pytest.raises(type(exception)) as caught:
        assessment.execute_evidence_assessment(prepared, client)
    assert caught.value is exception and len(client.calls) == 1


@pytest.mark.parametrize("bad", [None, [], {}, {"schema": isolation.INPUT_VERSION}])
def test_invalid_source_payloads_rejected(bad):
    with pytest.raises(ValueError):
        assessment.prepare_evidence_assessment(bad, template(), max_calls=3)


def test_unchanged_strict_source_contract_rejects_bad_spans_cross_scope_and_noop():
    for alteration in ("span", "cross_scope", "noop", "procedure_order"):
        value = payload()
        if alteration == "span":
            value["source_catalog"][0]["end"] += 1
        elif alteration == "cross_scope":
            value["items"][0]["cited_source_ids"] = ["missing"]
        elif alteration == "noop":
            value["summary_item"]["candidate_is_noop"] = True
        else:
            value["procedure_items"][0]["candidate"]["steps"][1]["order"] = 1
        with pytest.raises(ValueError):
            plan(value)
