"""Independent structural controls, not observations of semantic model quality."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import socket

import pytest

from benchmarks import digest_source_review as review
from hymem.deadline import DeadlineExceeded
from hymem.extraction.llm import LLMRequest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("offline source-review controls prohibit networking")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def packet():
    def source(cid, mid, text, prefix, role="user"):
        return {"chunk_id": cid, "message_id": mid, "role": role,
                "source_peer_id": "peer-not-a-name", "source_workspace_id": None,
                "start": len(prefix), "end": len(prefix) + len(text),
                "visible_content": text, "interpretation_only_context": {
                    "message_id": mid, "role": role,
                    "source_peer_id": "peer-not-a-name", "source_workspace_id": None,
                    "start": 0, "end": len(prefix), "content": prefix}}
    a = source("a", 51, "nt to visit the observatory. Café Ω-17 stays open.",
               "I visited the harbor, but I wa")
    b = source("b", 52, "First isolate the pump, then inspect it. Never open the vent.",
               "For the pump: ", "assistant")
    c = source("c", 53, "UNCITED_REPLACEMENT_ONLY", "Outside scope: ")
    return {"schema": "digest-fidelity-decisions-v9", "source_catalog": [a, b, c],
        "items": [{"index": 0, "candidate_title": "Observatory interest",
                   "candidate_body": "The user wants to visit the observatory. Café Ω-17 stays open.",
                   "candidate_outcome": "informational", "candidate_key_entities": [],
                   "cited_source_ids": ["a", "b"]}],
        "procedure_items": [{"index": 0, "candidate": {
            "name": "Pump inspection", "description": "Never open the vent.",
            "steps": [{"order": 1, "action": "Isolate the pump", "tool": None},
                      {"order": 2, "action": "Inspect the pump", "tool": " \t"}],
            "triggers": [], "entities_involved": []}, "cited_source_ids": ["b"]}],
        "summary_item": {"index": 0, "candidate_raw_summary": "REJECTED_RAW_ONLY",
            "candidate_summary": "The user wants to visit the observatory. Isolate then inspect the pump; never open the vent.",
            "candidate_is_noop": False, "new_source_ids": ["a", "b", "c"],
            "prior_derived_summary": "PRIOR_CONTINUITY_ONLY A historical music topic."}}


TEMPLATE = LLMRequest("original system", "original user", "json", 3072, 0.25)


def prepare(value=None, **kwargs):
    return review.prepare_source_review(packet() if value is None else value,
        TEMPLATE, max_calls=kwargs.pop("max_calls", 3), **kwargs)


def response(scope, verdict="supported"):
    primary = scope.canonical_sources[0].source_id if scope.canonical_sources else None
    result = {}
    for check in scope.checks:
        if check.kind == "retention":
            result[check.check_id] = (["retained", [scope.fields[0].field_id]]
                if verdict == "supported" and scope.fields else ["uncertain", []])
        else:
            result[check.check_id] = [verdict, [primary]
                if verdict == "supported" and primary else [], []]
    return result


def parse(scope, body):
    return review.parse_source_review(json.dumps(body, ensure_ascii=False), scope)


def grounding(scope):
    return next(check for check in scope.checks if check.kind == "assertion")


class Client:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def complete(self, request):
        self.calls.append(request)
        result = next(self.outputs)
        if isinstance(result, BaseException):
            raise result
        return result


def test_authority_planes_exact_and_scope_isolation_unchanged():
    for scope in prepare().requests:
        wire = json.loads(scope.request.user)
        assert "base_scope" not in wire
        assert "evidence_sources" not in wire
        assert wire["canonical_sources"] == [asdict(x) for x in scope.canonical_sources]
        assert wire["context_sources"] == [asdict(x) for x in scope.context_sources]
        assert wire["prior_summary_sources"] == [asdict(x) for x in scope.prior_summary_sources]
        assert all(x.kind == "canonical_text" and x.allowed_use == "support"
                   for x in scope.canonical_sources)
        assert all(x.kind in {"boundary_context", "attribution"}
                   and x.allowed_use == "interpretation" for x in scope.context_sources)
        assert all(x.kind == "prior_summary" and x.allowed_use == "continuity"
                   for x in scope.prior_summary_sources)
        original = {x.source_id: x for x in scope.base_scope.evidence_sources}
        for x in (*scope.canonical_sources, *scope.context_sources, *scope.prior_summary_sources):
            assert replace(x, allowed_use=original[x.source_id].allowed_use) == original[x.source_id]
        assert "REJECTED_RAW_ONLY" not in scope.request.user
        if scope.kind != "summary":
            assert not scope.prior_summary_sources
            assert "PRIOR_CONTINUITY_ONLY" not in scope.request.user
            assert "UNCITED_REPLACEMENT_ONLY" not in scope.request.user
        assert (scope.request.response_format, scope.request.max_tokens, scope.request.temperature) == ("json", 3072, .25)
        assert "original user" not in scope.request.user


@pytest.mark.parametrize("source_kind", ["boundary_context", "attribution"])
def test_context_and_metadata_cannot_be_promoted_to_primary(source_kind):
    scope = prepare().requests[0]
    context = next(x for x in scope.context_sources if x.kind == source_kind)
    for primary, secondary in (([context.source_id], []), ([], [context.source_id]),
                               ([scope.canonical_sources[0].source_id, context.source_id], [])):
        value = response(scope)
        value[grounding(scope).check_id] = ["supported", primary, secondary]
        result = parse(scope, value)
        assert result.status == "malformed_review" and result.judgments == ()


def test_positive_context_must_be_anchored_to_its_own_canonical_record():
    scope = prepare().requests[0]
    a, b = scope.canonical_sources
    context = next(x for x in scope.context_sources if x.kind == "boundary_context" and x.chunk_id == "b")
    value = response(scope)
    value[grounding(scope).check_id] = ["supported", [a.source_id], [context.source_id]]
    assert parse(scope, value).status == "malformed_review"
    value[grounding(scope).check_id][1] = [b.source_id]
    assert parse(scope, value).review_structure_valid


def test_valid_continuation_not_removed_by_authority_separation():
    scope = prepare().requests[0]
    a = scope.canonical_sources[0]
    ctx = next(x for x in scope.context_sources if x.kind == "boundary_context" and x.chunk_id == "a")
    assert ctx.text.endswith("I wa") and a.text.startswith("nt to visit")
    value = response(scope)
    value[grounding(scope).check_id] = ["supported", [a.source_id], [ctx.source_id]]
    result = parse(scope, value)
    assert result.review_structure_valid and result.model_no_defect
    assert not result.semantic_verified and not result.publication_authorized


def test_irrelevant_canonical_id_is_still_not_entailment_proof():
    value = packet()
    value["items"][0]["candidate_body"] = "The user visited the harbor and exclusively caused the pump fault."
    scope = prepare(value).requests[0]
    # Deliberately wrong scripted judgment on real evidence. Structure cannot
    # prove entailment or turn an independent context-only trip into a fact.
    outcome = parse(scope, response(scope))
    assert outcome.review_structure_valid and outcome.model_no_defect
    assert not outcome.semantic_verified and not outcome.publication_authorized


def test_summary_continuity_cannot_sponsor_boundary_context():
    scope = prepare().requests[-1]
    prior = scope.prior_summary_sources[0]
    ctx = next(x for x in scope.context_sources if x.kind == "boundary_context")
    value = response(scope)
    value[grounding(scope).check_id] = ["supported", [prior.source_id], []]
    assert parse(scope, value).review_structure_valid
    value[grounding(scope).check_id][2] = [ctx.source_id]
    assert parse(scope, value).status == "malformed_review"


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_negative_verdicts_remain_honest_and_typed(verdict):
    scope = prepare().requests[0]
    value = response(scope, verdict)
    result = parse(scope, value)
    assert result.review_structure_valid and not result.model_no_defect
    value[grounding(scope).check_id][1] = [scope.context_sources[0].source_id]
    assert parse(scope, value).status == "malformed_review"


@pytest.mark.parametrize("entry", [
    ["supported", [], []], ["supported", ["s999"], []],
    ["supported", ["s0", "s0"], []], ["supported", [True], []],
    ["supported", ["s0"], ["s0"]], ["SUPPORTED", ["s0"], []],
    ["supported", ["s0"]], ["supported", ["s0"], [], "extra"],
    {"verdict": "supported"}, [True, [], []], ["supported", "s0", []],
])
def test_bad_single_check_invalidates_whole_scope(entry):
    scope = prepare().requests[0]
    value = response(scope)
    value[grounding(scope).check_id] = entry
    result = parse(scope, value)
    assert result.status == "malformed_review" and not result.judgments


@pytest.mark.parametrize("raw", ["null", "[]", "NaN", "{}", "```json\n{}\n```",
                                  '{"c0":[],"c0":[]}'])
def test_invalid_json_not_repaired(raw):
    result = review.parse_source_review(raw, prepare().requests[0])
    assert result.status == "malformed_review" and result.judgments == ()


def test_missing_and_extra_checks_rejected():
    scope = prepare().requests[0]
    for change in (lambda v: v.pop(grounding(scope).check_id),
                   lambda v: v.update(extra=["uncertain", [], []])):
        value = response(scope)
        change(value)
        assert parse(scope, value).status == "malformed_review"


def test_original_payload_mutation_cannot_change_plan():
    value = packet()
    plan = prepare(value)
    value["source_catalog"][0]["visible_content"] = "altered"
    assert plan == prepare()
    field = next(f for f in plan.requests[1].fields if f.path.endswith("/tool"))
    assert field.text == " \t"


def test_tampered_last_request_rejected_before_first_invocation():
    plan = prepare()
    last = plan.requests[-1]
    changed = replace(last, request=replace(last.request, user=last.request.user + " "))
    plan = replace(plan, requests=(*plan.requests[:-1], changed))
    client = Client([])
    with pytest.raises(ValueError):
        review.execute_source_review(plan, client)
    assert client.calls == []
    with pytest.raises(ValueError):
        review.parse_source_review("{}", changed)


def test_call_and_complete_response_caps_are_checked_before_execution():
    with pytest.raises(ValueError):
        prepare(max_calls=2)
    with pytest.raises(ValueError):
        prepare(max_output_chars=2)


def test_execution_continues_after_output_rejection_without_retry():
    plan = prepare()
    outputs = ["broken", *(json.dumps(response(s)) for s in plan.requests[1:])]
    client = Client(outputs)
    result = review.execute_source_review(plan, client)
    assert result.complete and result.attempted_calls == 3 and len(client.calls) == 3
    assert result.outcomes[0].status == "malformed_review"
    assert not result.review_structure_valid and not result.model_no_defect
    assert not result.semantic_verified and not result.publication_authorized


def test_client_error_halts_and_is_sanitized():
    client = Client([RuntimeError("SENSITIVE_MARKER")])
    result = review.execute_source_review(prepare(), client)
    assert result.attempted_calls == 1 and len(client.calls) == 1 and not result.complete
    assert "SENSITIVE_MARKER" not in repr(result)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), DeadlineExceeded("stop")])
def test_deadlines_and_interrupts_propagate(error):
    client = Client([error])
    with pytest.raises(type(error)):
        review.execute_source_review(prepare(), client)
    assert len(client.calls) == 1
