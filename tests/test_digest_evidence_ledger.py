"""Invented offline certificates demonstrate structure, never model accuracy."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from benchmarks import digest_evidence_ledger as ledger
from hymem.deadline import DeadlineExceeded
from hymem.extraction.llm import LLMRequest


def payload():
    first = "Cedar is available on Aurora. Cache repair is a suggestion."
    second = "Inspect cache then verify cache. The tool is scanner."
    procedure = {"index": 0, "candidate": {
        "name": "Inspect cache", "description": "Inspect safely.",
        "steps": [{"order": 1, "action": "Inspect cache", "tool": "scanner"},
                  {"order": 2, "action": "Verify cache", "tool": None}],
        "triggers": ["cache repair", "cache repair"], "entities_involved": ["cache"]},
        "cited_source_ids": ["c2"]}
    repeated = deepcopy(procedure)
    repeated["index"] = 1
    repeated["cited_source_ids"] = ["c1"]
    return {
        "schema": ledger.isolation.INPUT_VERSION,
        "source_catalog": [
            {"chunk_id": "c1", "message_id": 1, "role": "user", "source_peer_id": "peer-1",
             "source_workspace_id": "workspace-1", "start": 8, "end": 8 + len(first),
             "visible_content": first,
             "interpretation_only_context": {"message_id": 1, "role": "user",
                "source_peer_id": "peer-1", "source_workspace_id": "workspace-1",
                "start": 0, "end": 8, "content": "Earlier "}},
            {"chunk_id": "c2", "message_id": 2, "role": "assistant", "source_peer_id": None,
             "source_workspace_id": "", "start": 0, "end": len(second),
             "visible_content": second, "interpretation_only_context": None}],
        "items": [{"index": 0, "candidate_title": "Cedar claim",
                   "candidate_body": "Cedar is exclusive to Aurora.", "candidate_outcome": "informational",
                   "candidate_key_entities": ["Cedar", "Aurora"], "cited_source_ids": ["c1"]}],
        "procedure_items": [procedure, repeated],
        "summary_item": {"index": 0, "candidate_raw_summary": "REJECTED RAW MUST NOT ENTER",
                         "candidate_summary": "Cedar is available; inspect cache.", "candidate_is_noop": False,
                         "new_source_ids": ["c1", "c2"], "prior_derived_summary": "Old topic persists."}}


def template():
    return LLMRequest("discarded system", "discarded user", "json", 3072, 0.4)


def plan(value=None, **kwargs):
    return ledger.prepare_evidence_ledger(value or payload(), template(), max_calls=4, **kwargs)


def response(scope, verdict="supported"):
    source = next((part for part in scope.evidence_sources if part.kind == "canonical_text"), None)
    evidence = ([{"source_id": source.source_id, "quote": source.text, "use": source.allowed_use}]
                if source is not None and verdict == "supported" else [])
    return {"schema": ledger.VERSION, "scope": {"kind": scope.kind, "index": scope.index},
            "claims": [{"field": field.path, "text": field.text, "verdict": verdict,
                        "evidence": deepcopy(evidence)} for field in scope.fields if field.text]}


def parse(scope, value=None, **kwargs):
    return ledger.parse_evidence_ledger(json.dumps(response(scope) if value is None else value), scope, **kwargs)


class Scripted:
    def __init__(self, values):
        self.values, self.calls = iter(values), []

    def complete(self, request):
        self.calls.append(request)
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def test_full_order_scope_projection_and_sampling():
    prepared = plan()
    assert [(scope.kind, scope.index) for scope in prepared.requests] == [
        ("episode", 0), ("procedure", 0), ("procedure", 1), ("summary", 0)]
    assert len(prepared.requests) == prepared.max_calls
    for scope in prepared.requests:
        assert replace(scope.request, system=template().system, user=template().user) == template()
        packet = json.loads(scope.request.user)
        assert set(packet) == {"schema", "scope", "fields", "evidence_sources"}
        assert packet["fields"] == [asdict(field) for field in scope.fields]
        assert packet["evidence_sources"] == [asdict(source) for source in scope.evidence_sources]
        assert "REJECTED RAW" not in scope.request.user
        assert all(source.text for source in scope.evidence_sources)
    assert "Old topic persists" not in prepared.requests[0].request.user
    assert "Inspect cache then" not in prepared.requests[0].request.user
    assert "Old topic persists" in prepared.requests[-1].request.user
    assert "Inspect cache then" in prepared.requests[-1].request.user


def test_field_paths_procedure_multiplicity_and_null_omission():
    prepared = plan()
    assert [field.path for field in prepared.requests[0].fields] == [
        "/candidate_title", "/candidate_body", "/candidate_outcome",
        "/candidate_key_entities/0", "/candidate_key_entities/1"]
    assert [field.path for field in prepared.requests[1].fields] == [
        "/candidate/name", "/candidate/description", "/candidate/steps/0/action",
        "/candidate/steps/0/tool", "/candidate/steps/1/action", "/candidate/triggers/0",
        "/candidate/triggers/1", "/candidate/entities_involved/0"]
    assert prepared.requests[1].fields == prepared.requests[2].fields
    assert prepared.requests[1].evidence_sources != prepared.requests[2].evidence_sources


def test_source_identity_coordinates_and_order():
    scope = plan().requests[0]
    assert [(s.kind, s.field, s.start, s.allowed_use) for s in scope.evidence_sources] == [
        ("canonical_text", "visible_content", 8, "support"),
        ("attribution", "role", 0, "support"),
        ("attribution", "source_peer_id", 0, "support"),
        ("attribution", "source_workspace_id", 0, "support"),
        ("boundary_context", "content", 0, "interpretation")]
    assert [s.source_id for s in scope.evidence_sources] == [f"s{x}" for x in range(5)]
    assert all(s.chunk_id == "c1" and s.message_id == 1 for s in scope.evidence_sources)


def test_snapshot_and_records_immutable():
    value = payload()
    before = plan(value)
    value["items"][0]["candidate_body"] = "Changed"
    assert before == plan()
    with pytest.raises(FrozenInstanceError):
        before.requests[0].fields[0].text = "Changed"
    with pytest.raises(FrozenInstanceError):
        before.requests[0].evidence_sources[0].text = "Changed"


def test_irrelevant_real_quote_is_structurally_valid_not_semantically_verified():
    scope = plan().requests[0]
    value = response(scope)
    # Availability does NOT establish exclusivity. Exact citation checks cannot
    # detect that the model's supported verdict is semantically wrong.
    outcome = parse(scope, value)
    assert outcome.status == "valid_ledger" and outcome.model_all_supported
    assert outcome.claims[1].text == "Cedar is exclusive to Aurora."
    assert not outcome.semantic_verified and not outcome.publication_authorized
    assert not hasattr(outcome, "all_supported")


def test_offsets_derived_from_exact_unicode_partition():
    value = payload()
    value["items"][0]["candidate_body"] = "Café 👩🏽‍💻 e\u0301.\n否。"
    scope = plan(value).requests[0]
    reply = response(scope)
    original = reply["claims"].pop(1)
    pieces = ["Café ", "👩🏽‍💻", " e\u0301.\n", "否。"]
    reply["claims"][1:1] = [dict(original, text=piece) for piece in pieces]
    result = parse(scope, reply)
    claims = [claim for claim in result.claims if claim.field == "/candidate_body"]
    assert result.ledger_structure_valid
    position = 0
    for claim, piece in zip(claims, pieces):
        assert (claim.start, claim.end, claim.text) == (position, position + len(piece), piece)
        position += len(piece)


def test_evidence_offsets_preserve_original_coordinates_and_field_coordinates():
    scope = plan().requests[0]
    value = response(scope)
    value["claims"][0]["evidence"] = [
        {"source_id": "s0", "quote": "available", "use": "support"},
        {"source_id": "s2", "quote": "peer-1", "use": "support"},
        {"source_id": "s4", "quote": "Earlier", "use": "interpretation"}]
    result = parse(scope, value)
    assert result.ledger_structure_valid
    refs = result.claims[0].evidence
    assert (refs[0].start, refs[0].end) == (17, 26)
    assert (refs[1].start, refs[1].end) == (0, 6)
    assert (refs[2].start, refs[2].end) == (0, 7)


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_non_supported_claims_may_have_no_evidence(verdict):
    scope = plan().requests[0]
    result = parse(scope, response(scope, verdict))
    assert result.ledger_structure_valid and not result.model_all_supported
    assert all(claim.verdict == verdict and claim.evidence == () for claim in result.claims)


def test_empty_summary_is_valid_but_not_affirmative():
    value = payload()
    value["summary_item"]["candidate_summary"] = ""
    scope = plan(value).requests[-1]
    result = parse(scope)
    assert result.ledger_structure_valid and result.claims == ()
    assert not result.model_all_supported
    assert not result.semantic_verified and not result.publication_authorized


def test_whitespace_fields_and_empty_fields_preserved():
    value = payload()
    value["procedure_items"][0]["candidate"]["description"] = ""
    value["procedure_items"][0]["candidate"]["steps"][1]["tool"] = " \n"
    value["summary_item"]["candidate_summary"] = " \t\n"
    prepared = plan(value)
    scope = prepared.requests[1]
    assert all(field.path != "/candidate/description" for field in scope.fields)
    assert Candidate(scope, "/candidate/steps/1/tool") == " \n"
    assert parse(scope).ledger_structure_valid
    summary = prepared.requests[-1]
    assert parse(summary).claims[0].text == " \t\n"
    empty = response(summary)
    empty["claims"] = []
    assert not parse(summary, empty).ledger_structure_valid


def Candidate(scope, path):
    return next(field.text for field in scope.fields if field.path == path)


def test_prior_continuity_summary_only():
    scope = plan().requests[-1]
    value = response(scope)
    source = scope.evidence_sources[-1]
    assert source.kind == "prior_summary" and source.chunk_id is None and source.message_id is None
    value["claims"][0]["evidence"] = [{"source_id": source.source_id,
                                    "quote": "Old topic", "use": "continuity"}]
    result = parse(scope, value)
    assert result.ledger_structure_valid and result.model_all_supported
    assert not result.semantic_verified
    value["claims"][0]["evidence"][0]["use"] = "support"
    assert not parse(scope, value).ledger_structure_valid


@pytest.mark.parametrize("damage", [
    "schema", "extra_top", "missing_top", "scope_kind", "scope_index", "bool_index",
    "extra_scope", "extra_claim", "empty_claim", "missing_field", "missing_text", "reorder",
    "omit_negation", "trim", "unknown_path", "overlap", "duplicate_field", "field_order",
    "empty_text", "invalid_verdict", "bool_verdict", "extra_reference", "unknown_reference",
    "missing_quote", "case_quote", "whitespace_quote", "wrong_use", "duplicate_reference",
    "context_only", "no_evidence", "numeric_quote", "bool_source", "claims_dict", "refs_dict",
])
def test_malformed_certificates_never_return_partial_subset(damage):
    scope = plan().requests[0]
    value = response(scope)
    claim = value["claims"][1]
    ref = claim["evidence"][0]
    if damage == "schema": value["schema"] = "old"
    elif damage == "extra_top": value["reason"] = "fine"
    elif damage == "missing_top": value.pop("scope")
    elif damage == "scope_kind": value["scope"]["kind"] = "summary"
    elif damage == "scope_index": value["scope"]["index"] = 1
    elif damage == "bool_index": value["scope"]["index"] = False
    elif damage == "extra_scope": value["scope"]["extra"] = 1
    elif damage == "extra_claim": claim["start"] = 0
    elif damage == "empty_claim": value["claims"] = []
    elif damage == "missing_field": claim.pop("field")
    elif damage == "missing_text": value["claims"].pop()
    elif damage == "reorder": claim["text"] = "Aurora to exclusive is Cedar."
    elif damage == "omit_negation": claim["text"] = "Cedar is available."
    elif damage == "trim": claim["text"] = claim["text"][:-1]
    elif damage == "unknown_path": claim["field"] = "/candidate_unknown"
    elif damage == "overlap": value["claims"].insert(2, deepcopy(claim))
    elif damage == "duplicate_field": value["claims"].append(deepcopy(claim))
    elif damage == "field_order": value["claims"][:2] = value["claims"][:2][::-1]
    elif damage == "empty_text": claim["text"] = ""
    elif damage == "invalid_verdict": claim["verdict"] = "true"
    elif damage == "bool_verdict": claim["verdict"] = True
    elif damage == "extra_reference": ref["start"] = 8
    elif damage == "unknown_reference": ref["source_id"] = "other-item"
    elif damage == "missing_quote": ref["quote"] = "Elsewhere only"
    elif damage == "case_quote": ref["quote"] = "CEDAR"
    elif damage == "whitespace_quote": ref["quote"] = " "
    elif damage == "wrong_use": ref["use"] = "continuity"
    elif damage == "duplicate_reference": claim["evidence"].append(deepcopy(ref))
    elif damage == "context_only": claim["evidence"] = [{"source_id": "s4", "quote": "Earlier", "use": "interpretation"}]
    elif damage == "no_evidence": claim["evidence"] = []
    elif damage == "numeric_quote": ref["quote"] = 7
    elif damage == "bool_source": ref["source_id"] = True
    elif damage == "claims_dict": value["claims"] = {}
    elif damage == "refs_dict": claim["evidence"] = {}
    result = parse(scope, value)
    assert result.status == "malformed_ledger" and result.claims == ()
    assert not result.model_all_supported and not result.publication_authorized


@pytest.mark.parametrize("raw", [
    "{} }", "```json\n{}\n```", "[]", "null", "NaN", "Infinity", "{\"claims\":[],\"claims\":[]}",
    '{"x": "\\ud800"}', "\ud800", "[" * 2000 + "]" * 2000, b"{}", None,
])
def test_strict_json_no_trailer_repair(raw):
    result = ledger.parse_evidence_ledger(raw, plan().requests[0])
    assert result.status == "malformed_ledger"


def test_unique_quote_detection_includes_overlapping_occurrences():
    value = payload()
    source = value["source_catalog"][0]
    source["visible_content"] = "ababa"
    source["end"] = source["start"] + 5
    scope = plan(value).requests[0]
    reply = response(scope)
    reply["claims"][0]["evidence"][0]["quote"] = "aba"
    assert not parse(scope, reply).ledger_structure_valid


def test_identical_source_text_retains_provenance_and_is_not_ambiguous_across_units():
    value = payload()
    value["source_catalog"][0]["source_peer_id"] = "user"
    value["source_catalog"][0]["interpretation_only_context"]["source_peer_id"] = "user"
    scope = plan(value).requests[0]
    reply = response(scope)
    reply["claims"][0]["evidence"] = [{"source_id": "s1", "quote": "user", "use": "support"},
                                    {"source_id": "s2", "quote": "user", "use": "support"}]
    result = parse(scope, reply)
    assert result.ledger_structure_valid
    assert result.claims[0].evidence[0].field == "role"
    assert result.claims[0].evidence[1].field == "source_peer_id"


@pytest.mark.parametrize("kwargs", [
    {"max_calls": True}, {"max_calls": 3}, {"max_calls": 65}, {"max_input_chars": False},
    {"max_input_chars": 100}, {"max_output_chars": 0}, {"max_output_chars": 1.0},
    {"max_claims": True}, {"max_claims": 0}, {"max_claims": 7}, {"max_claims": ledger.MAX_CLAIMS + 1},
    {"max_evidence_per_claim": False}, {"max_evidence_per_claim": 33},
])
def test_whole_plan_cap_validation(kwargs):
    values = {"max_calls": 4, **kwargs}
    with pytest.raises(ValueError):
        ledger.prepare_evidence_ledger(payload(), template(), **values)


def test_output_claim_and_evidence_caps():
    scope = plan().requests[0]
    value = response(scope)
    assert not parse(scope, value, max_claims=4).ledger_structure_valid
    assert not parse(scope, value, max_output_chars=10).ledger_structure_valid
    value["claims"][0]["evidence"].append({"source_id": "s1", "quote": "user", "use": "support"})
    assert not parse(scope, value, max_evidence_per_claim=1).ledger_structure_valid


@pytest.mark.parametrize("damage", ["nan", "surrogate", "cycle", "set", "bigint", "subclass", "source_span", "source_bool"])
def test_untrusted_input_json_and_source_rejected(damage):
    value = payload()
    if damage == "nan": value["extra"] = float("nan")
    elif damage == "surrogate": value["items"][0]["candidate_body"] = "\ud800"
    elif damage == "cycle": value["extra"] = value
    elif damage == "set": value["extra"] = {"bad"}
    elif damage == "bigint": value["extra"] = 10**200
    elif damage == "subclass": value["extra"] = type("Evil", (str,), {})("text")
    elif damage == "source_span": value["source_catalog"][0]["end"] += 1
    elif damage == "source_bool": value["source_catalog"][0]["message_id"] = True
    with pytest.raises(ValueError):
        plan(value)


@pytest.mark.parametrize("damage", ["request", "field", "source", "binding", "plan_hash", "snapshot", "scope_index", "caps", "drop_scope", "swap_scope"])
def test_forged_plan_rejected_before_calls(damage):
    prepared = plan()
    first = prepared.requests[0]
    if damage == "request": first = replace(first, request=replace(first.request, user="{}"))
    elif damage == "field": first = replace(first, fields=(replace(first.fields[0], text="Different"),) + first.fields[1:])
    elif damage == "source": first = replace(first, evidence_sources=(replace(first.evidence_sources[0], start=True),) + first.evidence_sources[1:])
    elif damage == "binding": first = replace(first, binding_sha256="0" * 64)
    elif damage == "plan_hash": prepared = replace(prepared, plan_sha256="0" * 64)
    elif damage == "snapshot": prepared = replace(prepared, source_payload_json=prepared.source_payload_json + "}")
    elif damage == "scope_index": first = replace(first, index=False)
    elif damage == "caps": prepared = replace(prepared, max_claims=True)
    elif damage == "drop_scope": prepared = replace(prepared, requests=prepared.requests[:-1])
    elif damage == "swap_scope": prepared = replace(prepared, requests=(prepared.requests[1], prepared.requests[0]) + prepared.requests[2:])
    if first != prepared.requests[0] or damage == "scope_index":
        prepared = replace(prepared, requests=(first,) + prepared.requests[1:])
    llm = Scripted([])
    with pytest.raises(ValueError):
        ledger.execute_evidence_ledger(prepared, llm)
    assert llm.calls == []


def test_standalone_forged_scope_raises_not_model_malformed():
    scope = plan().requests[0]
    with pytest.raises(ValueError):
        parse(replace(scope, binding_sha256="bad"))


def test_execution_continues_malformed_and_semantic_veto_without_retry():
    prepared = plan()
    replies = [json.dumps(response(scope)) for scope in prepared.requests]
    replies[0] = "malformed"
    replies[1] = json.dumps(response(prepared.requests[1], "unsupported"))
    client = Scripted(replies)
    result = ledger.execute_evidence_ledger(prepared, client)
    assert result.complete and result.attempted_calls == 4 and len(client.calls) == 4
    assert [outcome.status for outcome in result.outcomes] == ["malformed_ledger"] + ["valid_ledger"] * 3
    assert not result.ledger_structure_valid and not result.model_all_supported
    assert not result.semantic_verified and not result.publication_authorized


def test_client_error_halts_and_is_sanitized():
    prepared = plan()
    client = Scripted([RuntimeError("secret provider text")])
    result = ledger.execute_evidence_ledger(prepared, client)
    assert result.attempted_calls == 1 and not result.complete
    assert result.outcomes[0].status == "execution_error"
    assert result.halted_reason == "client_exception"
    assert "secret" not in repr(result)
    assert not result.model_all_supported


@pytest.mark.parametrize("error", [DeadlineExceeded("deadline"), KeyboardInterrupt(), SystemExit(3)])
def test_deadlines_and_process_interrupts_propagate(error):
    client = Scripted([error])
    with pytest.raises(type(error)):
        ledger.execute_evidence_ledger(plan(), client)
    assert len(client.calls) == 1


def test_success_reports_model_only_never_publication():
    prepared = plan()
    client = Scripted(json.dumps(response(scope)) for scope in prepared.requests)
    result = ledger.execute_evidence_ledger(prepared, client)
    assert result.complete and result.ledger_structure_valid and result.model_all_supported
    assert not result.semantic_verified and not result.publication_authorized
    assert not hasattr(result, "all_supported")


@pytest.mark.parametrize("value", [[0], [True], [[]], [[], []], {"a": []}])
def test_bounded_json_accepts_exact_boundary(value):
    encoded = ledger._canonical(value)
    assert ledger._bounded_json(value, len(encoded)) == encoded


def test_empty_summary_does_not_erase_nonempty_model_claim_verdicts():
    value = payload()
    value["summary_item"]["candidate_summary"] = ""
    prepared = plan(value)
    assert prepared.requests[-1].fields == ()
    result = ledger.execute_evidence_ledger(prepared, Scripted(
        json.dumps(response(scope)) for scope in prepared.requests))
    assert not result.outcomes[-1].model_all_supported
    assert result.model_all_supported
    assert not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize("damage", ["empty_field", "summary_index", "oversize_field", "oversize_source"])
def test_standalone_scope_rejects_impossible_shapes_even_with_recomputed_hash(damage):
    scope = plan().requests[-1 if damage == "summary_index" else 0]
    if damage == "empty_field":
        scope = replace(scope, fields=(replace(scope.fields[0], text=""),) + scope.fields[1:])
    elif damage == "summary_index":
        scope = replace(scope, index=1)
    elif damage == "oversize_field":
        scope = replace(scope, fields=(replace(scope.fields[0], text="x" * (ledger.MAX_INPUT_CHARS + 1)),))
    elif damage == "oversize_source":
        scope = replace(scope, evidence_sources=(replace(scope.evidence_sources[0], text="x" * (ledger.MAX_INPUT_CHARS + 1)),))
    # Rehashing is not validation: the prepared-scope schema must still hold.
    scope = replace(scope, binding_sha256=ledger._sha(ledger._scope_body(scope)))
    with pytest.raises(ValueError):
        parse(scope)
