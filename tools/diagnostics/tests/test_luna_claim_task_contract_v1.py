"""Offline controls for the inactive original-claim ablation contract."""
import copy
import json
from dataclasses import replace

import pytest

from hymem.extraction import grounding_classification_v3 as v3
from hymem.extraction import grounding_v2 as v2
from hymem.extraction.triples import Triple
from tools.diagnostics import luna_claim_task_contract_v1 as task


def sample(*, content="Mira uses CairnDB.", contexts=(), **qualifiers):
    triple = Triple("Mira", "uses", "CairnDB", 1, source_message_id=7, **qualifiers)
    source = v2.GroundingSource(7, content, contexts=contexts, source_role="user")
    return (triple,), (source,)


def supported(entries=None, *, qualifiers=()):
    entries = entries if entries is not None else [
        {"source_message_id": 7, "region": "owned", "quote": "Mira uses CairnDB"}]
    return {"state": "supported", "support": {"evidence": entries,
            "checks": {name: {"state": "supported", "evidence_indices": list(range(len(entries)))}
                       for name in ("attribution_and_roles", "relation_and_polarity", *qualifiers)}}}


def response(batch, state="supported", *, arm="B", assessment=None):
    original = assessment if assessment is not None else (
        supported() if state == "supported" else {"state": state, "support": None})
    item = {"index": 0, "original": original}
    if arm == "A":
        item["alternatives"] = ({p: {"state": "not_established", "support": None}
                                for p in v3.PREDICATE_ORDER if p != "uses"}
                                if original["state"] == "not_established" else None)
    return {"schema": task.B_SCHEMA if arm == "B" else v3.GROUNDING_CONTRACT_VERSION,
            "batch_sha256": batch.batch_sha256, "complete": True, "classifications": [item]}


def parse(payload, batch, arm="B"):
    return task.parse_arm_response(arm, json.dumps(payload, ensure_ascii=False), batch)


def test_exact_a_and_same_b_user_batch_hash_and_parameters():
    triples, sources = sample(value_numeric=42, value_unit="ms", temporal_scope="today")
    a, batch_a = task.build_arm_request("A", triples, sources)
    b, batch_b = task.build_arm_request("B", triples, sources)
    frozen, frozen_batch = v3.build_grounding_request(triples, sources)
    assert (a, batch_a) == (frozen, frozen_batch)
    assert batch_b == batch_a
    assert b.user == a.user and b.system != a.system
    assert (b.response_format, b.max_tokens, b.temperature) == ("json", 4096, 0.0)
    assert "Predicate meanings:" in b.system
    assert "every non-null qualifier" in b.system
    assert "assess every other allowed predicate" not in b.system
    assert "Each assessment is exactly {state,support}" in b.system
    task.validate_arm_request("A", a, batch_a)
    task.validate_arm_request("B", b, batch_b)
    for arm, request in (("A", b), ("B", a)):
        with pytest.raises(v2.GroundingContractError, match="request:arm_binding"):
            task.validate_arm_request(arm, request, batch_a)
    for changed in (replace(b, user=b.user + " "), replace(b, max_tokens=4096.0),
                    replace(b, temperature=False), replace(b, system=b.system + " ")):
        with pytest.raises(v2.GroundingContractError, match="request:arm_binding"):
            task.validate_arm_request("B", changed, batch_b)
    with pytest.raises(v2.GroundingContractError, match="batch:binding"):
        task.validate_arm_request("B", b, replace(batch_b, canonical_json=batch_b.canonical_json + " "))


def test_distinct_b_schema_finite_bounds_and_supported_forms():
    triples, sources = sample(value_text="fast", value_numeric=42, value_unit="ms", temporal_scope="today")
    _, batch = task.build_arm_request("B", triples, sources)
    assert task.build_arm_output_schema("A", batch) == v3.build_output_schema(batch)
    schema = task.build_arm_output_schema("B", batch)
    assert schema["properties"]["schema"]["enum"] == [task.B_SCHEMA]
    assert set(schema["$defs"]) == {"assessment_0"}
    item = schema["properties"]["classifications"]["items"]["anyOf"][0]
    assert set(item["properties"]) == {"index", "original"}
    assert set(item["required"]) == {"index", "original"}
    assert schema["properties"]["classifications"]["maxItems"] == 1
    support = schema["$defs"]["assessment_0"]["properties"]["support"]["anyOf"][1]
    assert set(support["properties"]["checks"]["required"]) == {
        "attribution_and_roles", "relation_and_polarity", "value_text", "value_numeric",
        "value_unit", "temporal_scope"}
    assert support["properties"]["evidence"]["maxItems"] == 8
    assert support["properties"]["evidence"]["items"]["properties"]["quote"]["maxLength"] == 192
    encoded = json.dumps(schema)
    for forbidden in ('"allOf"', '"if"', '"then"', '"else"', '"prefixItems"', '"uniqueItems"'):
        assert forbidden not in encoded
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    valid = response(batch, assessment=supported(qualifiers=("value_text", "value_numeric", "value_unit", "temporal_scope")))
    jsonschema.validate(valid, schema)
    invalid = copy.deepcopy(valid)
    invalid["classifications"][0]["alternatives"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_positive_ledger_and_finite_negative_states_without_correction():
    triples, sources = sample()
    _, batch = task.build_arm_request("B", triples, sources)
    positive = parse(response(batch), batch)
    assert positive.arm == "B" and positive.batch_sha256 == batch.batch_sha256
    assert positive.alternative_states is None and positive.final_verdicts is None
    assert positive.original[0].state == "supported"
    assert positive.original[0].evidence[0].quote == "Mira uses CairnDB"
    assert positive.original[0].checks == (("attribution_and_roles", (0,)),
                                            ("relation_and_polarity", (0,)))
    for state in ("not_established", "ambiguous"):
        result = parse(response(batch, state), batch)
        assert result.original[0].state == state
        assert result.original[0].evidence == result.original[0].checks == ()
        assert result.alternative_states is result.final_verdicts is None
    a = parse(response(batch, arm="A"), batch, "A")
    assert a.original == positive.original
    assert a.alternative_states == (None,)
    assert a.final_verdicts[0].status == "supported"


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema=v3.GROUNDING_CONTRACT_VERSION),
    lambda p: p.update(batch_sha256="0" * 64),
    lambda p: p.update(complete=1),
    lambda p: p["classifications"][0].update(index=True),
    lambda p: p["classifications"][0].update(alternatives=None),
    lambda p: p["classifications"][0]["original"].update(state="replace_predicate"),
    lambda p: p["classifications"][0]["original"].update(support=None),
    lambda p: p["classifications"][0]["original"]["support"]["checks"].pop("relation_and_polarity"),
    lambda p: p["classifications"][0]["original"]["support"]["evidence"].append(
        {"source_message_id": 7, "region": "owned", "quote": "Mira uses CairnDB"}),
    lambda p: p["classifications"][0]["original"]["support"]["checks"]["relation_and_polarity"].update(evidence_indices=[True]),
    lambda p: p["classifications"][0]["original"]["support"]["evidence"][0].update(quote="made up"),
])
def test_malformed_b_fail_closed(mutation):
    triples, sources = sample()
    _, batch = task.build_arm_request("B", triples, sources)
    payload = response(batch)
    mutation(payload)
    with pytest.raises(v2.GroundingContractError):
        parse(payload, batch)


def test_cross_arm_response_and_bounds_fail_closed():
    triples, sources = sample()
    _, batch = task.build_arm_request("B", triples, sources)
    for arm, wrong in (("A", "B"), ("B", "A")):
        with pytest.raises(v2.GroundingContractError):
            parse(response(batch, arm=wrong), batch, arm)
    with pytest.raises(v2.GroundingContractError, match="response:bounds"):
        task.parse_arm_response("B", "x" * (v2.MAX_RESPONSE_CHARS + 1), batch)
    with pytest.raises(v2.GroundingContractError, match="arm:invalid"):
        task.build_arm_request("C", triples, sources)


def test_nested_context_parent_and_owned_prefix_guards():
    body = v2.GroundingContext("conversation_0", "Mira prefers CairnDB.", 10,
                               source_message_id=6)
    header = v2.GroundingContext("conversation_0_header", "Speaker: Mira", 10,
                                 source_message_id=6, applies_to_region="conversation_0",
                                 applies_to_prefix_chars=10)
    triples, sources = sample(content="Mira uses CairnDB. And later text.", contexts=(body, header))
    _, batch = task.build_arm_request("B", triples, sources)
    entries = [
        {"source_message_id": 7, "region": "owned", "quote": "Mira uses CairnDB"},
        {"source_message_id": 7, "region": "conversation_0_header", "quote": "Speaker: Mira"},
    ]
    with pytest.raises(v2.GroundingContractError, match="evidence:context_scope"):
        parse(response(batch, assessment=supported(entries)), batch)
    entries[0]["quote"] = "Mira uses"
    with pytest.raises(v2.GroundingContractError, match="evidence:parent_required"):
        parse(response(batch, assessment=supported(entries)), batch)
    entries.append({"source_message_id": 7, "region": "conversation_0", "quote": "Mira prefers"})
    with pytest.raises(v2.GroundingContractError, match="evidence:parent_scope"):
        parse(response(batch, assessment=supported(entries)), batch)
    entries[2]["quote"] = "Mira"
    valid = parse(response(batch, assessment=supported(entries)), batch)
    assert tuple(entry.region for entry in valid.original[0].evidence) == (
        "owned", "conversation_0_header", "conversation_0")


def test_batch_cap_and_quote_cap():
    triples, sources = sample()
    with pytest.raises(v2.GroundingContractError, match="triples:bounds"):
        task.build_arm_request("B", triples * 9, sources)
    _, batch = task.build_arm_request("B", triples, sources)
    oversized = response(batch, assessment=supported([
        {"source_message_id": 7, "region": "owned", "quote": "x" * 193}]))
    with pytest.raises(v2.GroundingContractError, match="evidence:quote"):
        parse(oversized, batch)
