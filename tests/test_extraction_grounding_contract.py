"""Offline wire and evidence checks for source-grounded triple review."""
import dataclasses
import json

import pytest

from hymem.extraction.grounding import (
    GroundingContext, GroundingContractError, GroundingSource,
    MAX_CONTEXTS_PER_SOURCE, MAX_EVIDENCE, MAX_QUOTE_CHARS, MAX_SOURCE_CHARS, MAX_TRIPLES,
    build_grounding_request, parse_grounding_response,
)
from hymem.extraction.triples import Triple


def _case(*, context=False, legacy=False):
    sid = None if legacy else 7
    triple = Triple("mira", "prefers", "vim", 1, source_message_id=sid)
    contexts = (GroundingContext("prelude", "Mira is the speaker.", 18),) if context else ()
    source = GroundingSource(sid, "I like vim over emacs.", contexts, "user", "mira", "2026-01-01T00:00:00Z")
    request, batch = build_grounding_request((triple,), (source,))
    return request, batch


def _reply(batch, *, status="supported", predicate="prefers", evidence=None):
    if evidence is None:
        evidence = [{"source_message_id": batch.sources[0].source_message_id,
                     "region": "owned", "quote": "I like vim over emacs."}]
    return {"schema": "source-grounding-v1", "batch_sha256": batch.batch_sha256, "complete": True,
            "verdicts": [{"index": 0, "status": status,
                          "predicate": predicate, "evidence": evidence}]}


def _parse(value, batch, **kwargs):
    return parse_grounding_response(json.dumps(value), batch, **kwargs)


def test_implicit_claim_wire_and_exact_binding():
    request, batch = _case(context=True)
    assert request.response_format == "json" and request.temperature == 0
    assert request.max_tokens == 4096
    wire = json.loads(request.user)
    assert wire["batch_sha256"] == batch.batch_sha256
    assert wire["batch"]["candidates"][0]["value_numeric"] is None
    assert wire["batch"]["sources"][0]["source_role"] == "user"
    assert "implicit" in request.system and "lexical overlap alone" in request.system
    result = _parse(_reply(batch), batch)
    assert result.all_supported and result.verdicts[0].evidence[0].region == "owned"
    changed = dataclasses.replace(batch.sources[0], source_peer_id="other")
    assert build_grounding_request(batch.triples, (changed,))[1].batch_sha256 != batch.batch_sha256
    changed = dataclasses.replace(batch.triples[0], temporal_scope="tomorrow")
    assert build_grounding_request((changed,), batch.sources)[1].batch_sha256 != batch.batch_sha256


def test_context_requires_owned_quote_inside_its_applicability_prefix():
    _, batch = _case(context=True)
    value = _reply(batch, evidence=[
        {"source_message_id": 7, "region": "prelude", "quote": "Mira is the speaker."},
        {"source_message_id": 7, "region": "owned", "quote": "I like vim"},
    ])
    assert _parse(value, batch).all_supported
    value["verdicts"][0]["evidence"][1]["quote"] = "over emacs."
    with pytest.raises(GroundingContractError, match="evidence:context_scope"):
        _parse(value, batch)
    value["verdicts"][0]["evidence"] = value["verdicts"][0]["evidence"][:1]
    with pytest.raises(GroundingContractError, match="evidence:owned_required"):
        _parse(value, batch)


def test_boundary_context_is_distinct_from_owned_and_all_owned_quotes_must_apply():
    source = GroundingSource(7, "First sentence. Later unrelated claim.",
                             (GroundingContext("boundary", "Prior clause", 15),))
    triple = Triple("first", "uses", "clause", 1, source_message_id=7)
    _, batch = build_grounding_request((triple,), (source,))
    value = _reply(batch, predicate="uses", evidence=[
        {"source_message_id": 7, "region": "boundary", "quote": "Prior clause"},
        {"source_message_id": 7, "region": "owned", "quote": "First sentence."},
    ])
    assert _parse(value, batch).all_supported
    value["verdicts"][0]["evidence"].append(
        {"source_message_id": 7, "region": "owned", "quote": "Later unrelated claim."})
    with pytest.raises(GroundingContractError, match="evidence:context_scope"):
        _parse(value, batch)


def test_two_conversation_records_preserve_separate_body_header_prelude_and_metadata():
    regions = ("boundary", "header", "prelude", "conversation_0",
               "conversation_0_header", "conversation_0_prelude",
               "conversation_1", "conversation_1_header", "conversation_1_prelude")
    contexts = tuple(GroundingContext(region, f"{region} text", 15,
                                      source_role="assistant" if region.startswith("conversation_0") else "user",
                                      source_message_id=20 if region.startswith("conversation_0") else 21,
                                      applies_to_region=region.rsplit("_", 1)[0]
                                      if region.endswith(("_header", "_prelude")) else None,
                                      applies_to_prefix_chars=len(f"{region.rsplit('_', 1)[0]} text")
                                      if region.endswith(("_header", "_prelude")) else None)
                     for region in regions)
    source = GroundingSource(7, "First sentence. Next sentence.", contexts)
    triple = Triple("speaker", "uses", "x", 1, source_message_id=7)
    request, batch = build_grounding_request((triple,), (source,))
    wire_contexts = json.loads(request.user)["batch"]["sources"][0]["contexts"]
    assert MAX_CONTEXTS_PER_SOURCE == 9
    assert [item["region"] for item in wire_contexts] == list(regions)
    assert wire_contexts[4]["source_message_id"] == 20
    assert wire_contexts[4]["applies_to_region"] == "conversation_0"
    value = _reply(batch, predicate="uses", evidence=[
        {"source_message_id": 7, "region": "conversation_0_header", "quote": "conversation_0_header text"},
        {"source_message_id": 7, "region": "conversation_0", "quote": "conversation_0 text"},
        {"source_message_id": 7, "region": "owned", "quote": "First sentence."},
    ])
    assert _parse(value, batch).all_supported
    with pytest.raises(GroundingContractError, match="context:id"):
        build_grounding_request((triple,), (dataclasses.replace(source, contexts=(
            dataclasses.replace(contexts[0], source_message_id=True),)),))


def test_nested_header_applies_only_to_parent_body_prefix():
    table_prefix = len("| metric | value |\n| pulse | 60 |")
    body = GroundingContext("conversation_0", "| metric | value |\n| pulse | 60 |\nLater unrelated prose.", 15,
                            source_role="user", source_message_id=20)
    header = GroundingContext("conversation_0_header", "| metric | value |", 15,
                              source_role="user", source_message_id=20,
                              applies_to_region="conversation_0", applies_to_prefix_chars=table_prefix)
    source = GroundingSource(7, "I refer to that value.", (body, header))
    triple = Triple("speaker", "has_attribute", "60_bpm", 1, source_message_id=7)
    request, batch = build_grounding_request((triple,), (source,))
    assert json.loads(request.user)["batch"]["sources"][0]["contexts"][1]["applies_to_prefix_chars"] == table_prefix
    good = _reply(batch, predicate="has_attribute", evidence=[
        {"source_message_id": 7, "region": "owned", "quote": "I refer to that"},
        {"source_message_id": 7, "region": "conversation_0", "quote": "| pulse | 60 |"},
        {"source_message_id": 7, "region": "conversation_0_header", "quote": "| metric | value |"},
    ])
    assert _parse(good, batch).all_supported
    missing = json.loads(json.dumps(good))
    missing["verdicts"][0]["evidence"].pop(1)
    with pytest.raises(GroundingContractError, match="evidence:parent_required"):
        _parse(missing, batch)
    late = json.loads(json.dumps(good))
    late["verdicts"][0]["evidence"][1]["quote"] = "Later unrelated prose."
    with pytest.raises(GroundingContractError, match="evidence:parent_scope"):
        _parse(late, batch)
    with pytest.raises(GroundingContractError, match="context:parent_missing"):
        build_grounding_request((triple,), (dataclasses.replace(source, contexts=(header,)),))
    with pytest.raises(GroundingContractError, match="context:parent_prefix"):
        build_grounding_request((triple,), (dataclasses.replace(source, contexts=(body, dataclasses.replace(header, applies_to_prefix_chars=True))),))
    with pytest.raises(GroundingContractError, match="context:parent_metadata"):
        build_grounding_request((triple,), (dataclasses.replace(source, contexts=(body, dataclasses.replace(header, source_message_id=21))),))


def test_two_nested_table_records_need_seven_distinct_witnesses():
    owned = "I compare both tables today."
    contexts = []
    evidence = [{"source_message_id": 7, "region": "owned", "quote": "I compare"}]
    for index in (0, 1):
        body_region = f"conversation_{index}"
        body_content = f"row_{index} = value_{index}"
        identity = 20 + index
        contexts.append(GroundingContext(body_region, body_content, len(owned),
                                         source_role="user", source_message_id=identity))
        evidence.append({"source_message_id": 7, "region": body_region, "quote": body_content})
        for suffix in ("header", "prelude"):
            region = f"{body_region}_{suffix}"
            content = f"{suffix}_{index}"
            contexts.append(GroundingContext(region, content, len(owned),
                                             source_role="user", source_message_id=identity,
                                             applies_to_region=body_region,
                                             applies_to_prefix_chars=len(body_content)))
            evidence.append({"source_message_id": 7, "region": region, "quote": content})
    triple = Triple("speaker", "uses", "both_tables", 1, source_message_id=7)
    _, batch = build_grounding_request((triple,), (GroundingSource(7, owned, tuple(contexts)),))
    assert len(evidence) == 7 and MAX_EVIDENCE == 8
    value = _reply(batch, predicate="uses", evidence=evidence)
    assert _parse(value, batch).all_supported
    value["verdicts"][0]["evidence"].extend([
        {"source_message_id": 7, "region": "owned", "quote": "both"},
        {"source_message_id": 7, "region": "owned", "quote": "tables"},
    ])
    with pytest.raises(GroundingContractError, match="verdict:evidence_count"):
        _parse(value, batch)


@pytest.mark.parametrize("mutation,code", [
    (lambda x: x.update(extra=1), "response:shape"),
    (lambda x: x.update(schema="other"), "response:schema"),
    (lambda x: x.update(complete=1), "response:incomplete"),
    (lambda x: x.update(batch_sha256="0"*64), "response:binding"),
    (lambda x: x.update(verdicts=[]), "response:count"),
    (lambda x: x["verdicts"][0].update(index=True), "verdict:index"),
    (lambda x: x["verdicts"][0].update(index=1), "verdict:index"),
    (lambda x: x["verdicts"][0].update(status="accepted"), "verdict:status"),
    (lambda x: x["verdicts"][0].update(predicate="uses"), "verdict:predicate"),
    (lambda x: x["verdicts"][0].update(extra=1), "verdict:shape"),
    (lambda x: x["verdicts"][0]["evidence"][0].update(region="other"), "evidence:region"),
    (lambda x: x["verdicts"][0]["evidence"][0].update(quote="not present"), "evidence:quote_missing"),
    (lambda x: x["verdicts"][0]["evidence"][0].update(source_message_id=8), "evidence:source"),
    (lambda x: x["verdicts"][0]["evidence"][0].update(extra=1), "evidence:shape"),
])
def test_bad_response_shapes(mutation, code):
    _, batch = _case()
    value = _reply(batch)
    mutation(value)
    with pytest.raises(GroundingContractError) as raised:
        _parse(value, batch)
    assert raised.value.code == code


def test_duplicate_json_keys_malformed_and_fences():
    _, batch = _case()
    raw = json.dumps(_reply(batch))
    assert parse_grounding_response("```json\n" + raw + "\n```", batch).all_supported
    for bad in (raw.replace('"complete": true', '"complete": true, "complete": true'),
                raw + " trailing", "not json", "{", "{\"complete\": NaN}"):
        with pytest.raises(GroundingContractError):
            parse_grounding_response(bad, batch)


def test_negative_verdicts_are_retained_not_accepted():
    _, batch = _case()
    for status in ("unsupported", "uncertain"):
        result = _parse(_reply(batch, status=status, predicate=None, evidence=[]), batch)
        assert result.verdicts[0].status == status and not result.all_supported
        with pytest.raises(GroundingContractError, match="verdict:negative_shape"):
            _parse(_reply(batch, status=status, predicate=None), batch)


def test_narrow_predicate_replacement_and_recheck_mode():
    _, batch = _case()
    value = _reply(batch, status="replace_predicate", predicate="uses")
    result = _parse(value, batch)
    assert result.verdicts[0].predicate == "uses" and not result.all_supported
    with pytest.raises(GroundingContractError, match="verdict:correction"):
        _parse(value, batch, allow_corrections=False)
    value["verdicts"][0]["predicate"] = "prefers"
    with pytest.raises(GroundingContractError, match="verdict:correction"):
        _parse(value, batch)


def test_legacy_null_has_no_fabricated_id():
    _, batch = _case(legacy=True)
    assert _parse(_reply(batch), batch).all_supported
    with pytest.raises(GroundingContractError, match="source:legacy_scope"):
        build_grounding_request(batch.triples, (*batch.sources, GroundingSource(8, "Other text")))
    with pytest.raises(GroundingContractError, match="triple:source"):
        build_grounding_request((dataclasses.replace(batch.triples[0], source_message_id=8),), batch.sources)


def test_bounds_types_duplicates_and_forged_batch():
    _, batch = _case()
    with pytest.raises(GroundingContractError, match="triples:bounds"):
        build_grounding_request(batch.triples * (MAX_TRIPLES + 1), batch.sources)
    with pytest.raises(GroundingContractError, match="source:content"):
        build_grounding_request(batch.triples, (dataclasses.replace(batch.sources[0], content="x"*(MAX_SOURCE_CHARS+1)),))
    with pytest.raises(GroundingContractError, match="source:duplicate_id"):
        build_grounding_request(batch.triples, batch.sources * 2)
    with pytest.raises(GroundingContractError, match="triple:polarity"):
        build_grounding_request((dataclasses.replace(batch.triples[0], polarity=True),), batch.sources)
    with pytest.raises(GroundingContractError, match="triple:numeric"):
        build_grounding_request((dataclasses.replace(batch.triples[0], value_numeric=float("nan")),), batch.sources)
    with pytest.raises(GroundingContractError, match="triple:numeric"):
        build_grounding_request((dataclasses.replace(batch.triples[0], value_numeric=10**1000),), batch.sources)
    with pytest.raises(GroundingContractError, match="context:prefix"):
        build_grounding_request(batch.triples, (dataclasses.replace(batch.sources[0], contexts=(GroundingContext("prelude", "x", True),)),))
    with pytest.raises(GroundingContractError, match="context:region"):
        build_grounding_request(batch.triples, (dataclasses.replace(batch.sources[0], contexts=(GroundingContext([], "x", 1),)),))
    with pytest.raises(GroundingContractError, match="triple:source"):
        build_grounding_request((dataclasses.replace(batch.triples[0], source_message_id=[]),), batch.sources)
    with pytest.raises(GroundingContractError, match="batch:unicode"):
        build_grounding_request((dataclasses.replace(batch.triples[0], subject="bad\ud800"),), batch.sources)
    with pytest.raises(GroundingContractError, match="batch:binding"):
        parse_grounding_response(json.dumps(_reply(batch)), dataclasses.replace(batch, batch_sha256="0"*64))


def test_evidence_quote_limit_duplicate_and_secret_free_diagnostics():
    secret = "TOP_SECRET_SYNTHETIC_" + "z"*200
    triple = Triple("mira", "uses", "vim", 1, source_message_id=7)
    _, batch = build_grounding_request((triple,), (GroundingSource(7, secret),))
    value = _reply(batch, predicate="uses", evidence=[{"source_message_id": 7, "region": "owned", "quote": secret}])
    with pytest.raises(GroundingContractError) as raised:
        _parse(value, batch)
    assert raised.value.code == "evidence:quote"
    assert secret not in str(raised.value)
    value["verdicts"][0]["evidence"] = [{"source_message_id": 7, "region": "owned", "quote": "TOP_SECRET_SYNTHETIC_"}]*2
    with pytest.raises(GroundingContractError, match="evidence:duplicate"):
        _parse(value, batch)
    assert MAX_QUOTE_CHARS == 192


def test_response_bounds_and_correction_flag():
    _, batch = _case()
    with pytest.raises(GroundingContractError, match="response:bounds"):
        parse_grounding_response("x"*65_537, batch)
    with pytest.raises(GroundingContractError, match="response:correction_flag"):
        _parse(_reply(batch), batch, allow_corrections=1)
