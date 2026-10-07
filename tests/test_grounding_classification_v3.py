"""Offline controls for the inactive claim-first grounding contract."""
import copy
import hashlib
import json
from dataclasses import replace

import pytest

from hymem.extraction import grounding_classification_v3 as c


def sample(*, predicate="uses", content="Mira uses CairnDB.", contexts=(), sid=7, **qualifiers):
    source = c.GroundingSource(sid, content, contexts=contexts, source_role="user")
    triple = c.Triple("Mira", predicate, "CairnDB", 1, source_message_id=sid, **qualifiers)
    return c.build_grounding_request((triple,), (source,))


def evidence(quote="Mira uses CairnDB", *, region="owned", sid=7):
    return {"source_message_id": sid, "region": region, "quote": quote}


def assessment(state="not_established", *, entries=None, qualifiers=()):
    if state != "supported":
        return {"state": state, "support": None}
    names = ("attribution_and_roles", "relation_and_polarity", *qualifiers)
    return {"state": state, "support": {"evidence": entries if entries is not None else [evidence()],
            "checks": {name: {"state": "supported", "evidence_indices": [0]} for name in names}}}


def reply(batch, *, original=None, alternatives=None):
    original = original if original is not None else assessment()
    if original["state"] == "not_established" and alternatives is None:
        alternatives = {p: assessment() for p in c.PREDICATE_ORDER if p != batch.triples[0].predicate}
    return {"schema": c.GROUNDING_CONTRACT_VERSION, "batch_sha256": batch.batch_sha256,
            "complete": True, "classifications": [{"index": 0, "original": original, "alternatives": alternatives}]}


def parse(payload, batch, **kwargs):
    return c.parse_grounding_response(json.dumps(payload, ensure_ascii=False), batch, **kwargs)


def rejects(payload, batch, code=None):
    with pytest.raises(c.GroundingContractError, match=f"^{code}$" if code else None):
        parse(payload, batch)


def test_request_identity_and_binding():
    request, batch = sample(value_numeric=42, value_unit="ms")
    canonical = json.loads(batch.canonical_json)
    wire = json.loads(request.user)
    assert c.GROUNDING_CONTRACT_VERSION == "source-grounding-classification-v3"
    assert len(c.PREDICATE_ORDER) == 22
    assert wire["batch"] == canonical
    assert canonical["candidates"][0]["predicate"] == "uses"
    assert canonical["candidates"][0]["value_numeric"] == 42
    assert batch.batch_sha256 == hashlib.sha256(batch.canonical_json.encode()).hexdigest()
    assert (request.max_tokens, request.response_format, request.temperature) == (4096, "json", 0.0)
    assert "third party" in request.system and "ordered actor/object" in request.system
    c.validate_request(request, batch)
    for changed in (replace(request, max_tokens=4096.0), replace(request, user=request.user + " "),
                    replace(request, temperature=False)):
        with pytest.raises(c.GroundingContractError, match="^request:binding$"):
            c.validate_request(changed, batch)
    with pytest.raises(c.GroundingContractError, match="^batch:binding$"):
        c.build_output_schema(replace(batch, canonical_json=batch.canonical_json + " "))
    _, other = sample(predicate="prefers")
    assert other.batch_sha256 != batch.batch_sha256


def test_schema_is_fresh_finite_and_validates_when_jsonschema_available():
    _, batch = sample(value_text="fast", value_numeric=42, value_unit="ms", temporal_scope="today")
    schema = c.build_output_schema(batch)
    defs = schema["$defs"]
    support = defs["assessment_0"]["properties"]["support"]["anyOf"][1]
    assert set(support["properties"]["checks"]["required"]) == {
        "attribution_and_roles", "relation_and_polarity", "value_text", "value_numeric", "value_unit", "temporal_scope"}
    assert len(defs["alternatives_0"]["required"]) == 21
    assert "uses" not in defs["alternatives_0"]["properties"]
    assert schema["properties"]["classifications"]["items"]["anyOf"][0]["properties"]["index"]["enum"] == [0]
    allowed_keywords = {"$schema", "$defs", "$ref", "type", "additionalProperties", "required",
                        "properties", "enum", "anyOf", "minItems", "maxItems", "items",
                        "minLength", "maxLength", "minimum", "maximum"}
    # Property and definition names are data, so walk only schema objects below.
    def schema_keywords(node):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            assert key in allowed_keywords
            if key in ("properties", "$defs"):
                for child in value.values():
                    schema_keywords(child)
            elif key in ("items",):
                schema_keywords(value)
            elif key == "anyOf":
                for child in value:
                    schema_keywords(child)
    schema_keywords(schema)
    support["properties"]["evidence"]["maxItems"] = 0
    assert c.build_output_schema(batch)["$defs"]["assessment_0"]["properties"]["support"]["anyOf"][1]["properties"]["evidence"]["maxItems"] == 8
    jsonschema = pytest.importorskip("jsonschema")
    schema = c.build_output_schema(batch)
    jsonschema.Draft202012Validator.check_schema(schema)
    valid = reply(batch, original=assessment("supported", qualifiers=("value_text", "value_numeric", "value_unit", "temporal_scope")))
    jsonschema.validate(valid, schema)
    invalid = copy.deepcopy(valid)
    invalid["classifications"][0]["alternatives"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)
    invalid = reply(batch)
    invalid["classifications"][0]["alternatives"].pop("prefers")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_selector_branches_and_conditional_coverage():
    _, batch = sample()
    assert parse(reply(batch, original=assessment("supported")), batch).verdicts[0].status == "supported"
    assert parse(reply(batch, original=assessment("ambiguous")), batch).verdicts[0].status == "uncertain"
    assert parse(reply(batch), batch).verdicts[0].status == "unsupported"
    payload = reply(batch)
    payload["classifications"][0]["alternatives"]["owns"] = assessment("supported")
    result = parse(payload, batch).verdicts[0]
    assert (result.status, result.predicate) == ("replace_predicate", "owns")
    with pytest.raises(c.GroundingContractError, match="^verdict:correction$"):
        parse(payload, batch, allow_corrections=False)
    payload["classifications"][0]["alternatives"]["prefers"] = assessment("supported")
    assert parse(payload, batch).verdicts[0].status == "uncertain"
    payload["classifications"][0]["alternatives"]["prefers"] = assessment("ambiguous")
    assert parse(payload, batch).verdicts[0].status == "uncertain"
    payload["classifications"][0]["alternatives"]["owns"] = assessment()
    assert parse(payload, batch).verdicts[0].status == "uncertain"
    payload = reply(batch, original=assessment("supported"), alternatives={})
    rejects(payload, batch, "classification:alternatives")
    payload = reply(batch)
    del payload["classifications"][0]["alternatives"]["owns"]
    rejects(payload, batch, "classification:alternatives")


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(schema="source-grounding-classification-v2"),
    lambda p: p.update(batch_sha256="0" * 64),
    lambda p: p.update(complete=1),
    lambda p: p.update(extra=1),
    lambda p: p["classifications"][0].update(index=True),
    lambda p: p["classifications"][0].update(extra=1),
    lambda p: p["classifications"][0]["original"].update(extra=1),
    lambda p: p["classifications"][0]["original"].update(state="unknown"),
    lambda p: p["classifications"][0]["original"].update(support={}),
    lambda p: p["classifications"][0].update(alternatives=None),
])
def test_old_or_malformed_response_rejected(mutate):
    _, batch = sample()
    payload = reply(batch)
    mutate(payload)
    rejects(payload, batch)


@pytest.mark.parametrize("mutate", [
    lambda s: s["checks"].pop("attribution_and_roles"),
    lambda s: s["checks"].update(value_numeric={"state": "supported", "evidence_indices": [0]}),
    lambda s: s["checks"]["relation_and_polarity"].update(state="ambiguous"),
    lambda s: s["checks"]["relation_and_polarity"].update(evidence_indices=[]),
    lambda s: s["checks"]["relation_and_polarity"].update(evidence_indices=[True]),
    lambda s: s["checks"]["relation_and_polarity"].update(evidence_indices=[1]),
    lambda s: s["checks"]["relation_and_polarity"].update(evidence_indices=[0, 0]),
    lambda s: s["evidence"].append(evidence()),
    lambda s: s["evidence"][0].update(source_message_id=True),
    lambda s: s["evidence"][0].update(quote="Mira CairnDB"),
])
def test_support_ledger_rejections(mutate):
    _, batch = sample()
    payload = reply(batch, original=assessment("supported"))
    mutate(payload["classifications"][0]["original"]["support"])
    rejects(payload, batch)


def test_qualifiers_and_all_evidence_referenced():
    _, batch = sample(value_numeric=42, value_unit="ms")
    payload = reply(batch, original=assessment("supported", qualifiers=("value_numeric", "value_unit")))
    assert parse(payload, batch).all_supported
    support = payload["classifications"][0]["original"]["support"]
    del support["checks"]["value_numeric"]
    rejects(payload, batch, "support:checks")
    support["checks"]["value_numeric"] = {"state": "supported", "evidence_indices": [0]}
    support["evidence"].append(evidence("uses CairnDB"))
    rejects(payload, batch, "support:unreferenced_evidence")
    support["checks"]["value_numeric"]["evidence_indices"].append(1)
    assert parse(payload, batch).all_supported


def test_global_distinct_evidence_limit():
    content = "Mira uses CairnDB abcdefghijklmnopqrstuvwxyz."
    _, batch = sample(content=content)
    payload = reply(batch)
    alternatives = payload["classifications"][0]["alternatives"]
    alternatives["owns"] = assessment("supported", entries=[evidence(ch) for ch in "abcdefgh"])
    for check in alternatives["owns"]["support"]["checks"].values():
        check["evidence_indices"] = list(range(8))
    assert parse(payload, batch).verdicts[0].status == "replace_predicate"
    alternatives["prefers"] = assessment("supported", entries=[evidence("Mira uses CairnDB")])
    rejects(payload, batch, "evidence:global_bounds")


def test_scope_parent_and_null_source():
    parent = c.GroundingContext("conversation_0", "Mira uses CairnDB. Later unrelated text.", 18,
                                source_role="user", source_message_id=7)
    header = c.GroundingContext("conversation_0_header", "Mira is speaking.", 18,
                                source_role="user", source_message_id=7,
                                applies_to_region="conversation_0", applies_to_prefix_chars=18)
    _, batch = sample(content="Mira uses CairnDB. Later unrelated text.", contexts=(parent, header))
    entries = [evidence(), evidence("Mira uses CairnDB", region="conversation_0"),
               evidence("Mira is speaking.", region="conversation_0_header")]
    payload = reply(batch, original=assessment("supported", entries=entries))
    for check in payload["classifications"][0]["original"]["support"]["checks"].values():
        check["evidence_indices"] = [0, 1, 2]
    assert parse(payload, batch).all_supported
    bad = copy.deepcopy(payload)
    bad["classifications"][0]["original"]["support"]["evidence"][0]["quote"] = "Later unrelated text."
    rejects(bad, batch, "evidence:context_scope")
    bad = copy.deepcopy(payload)
    bad["classifications"][0]["original"]["support"]["evidence"].pop(1)
    for check in bad["classifications"][0]["original"]["support"]["checks"].values():
        check["evidence_indices"] = [0, 1]
    rejects(bad, batch, "evidence:parent_required")
    _, legacy = sample(sid=None)
    payload = reply(legacy, original=assessment("supported", entries=[evidence(sid=None)]))
    assert parse(payload, legacy).all_supported


def test_bounds_and_multiclaim_order():
    with pytest.raises(c.GroundingContractError, match="^triple:text$"):
        c.build_grounding_request((c.Triple("x" * 2049, "uses", "y", 1, source_message_id=7),),
                                  (c.GroundingSource(7, "x"),))
    _, batch = sample()
    with pytest.raises(c.GroundingContractError, match="^response:bounds$"):
        c.parse_grounding_response(json.dumps(reply(batch)) + " " * c.MAX_RESPONSE_CHARS, batch)
    payload = reply(batch, original=assessment("supported"))
    payload["classifications"][0]["original"]["support"]["evidence"][0]["quote"] = "x" * 193
    rejects(payload, batch, "evidence:quote")
    source = c.GroundingSource(7, "Mira uses CairnDB.")
    triples = (c.Triple("Mira", "uses", "CairnDB", 1, source_message_id=7),
               c.Triple("Mira", "prefers", "CairnDB", 1, source_message_id=7))
    _, batch = c.build_grounding_request(triples, (source,))
    payload = {"schema": c.GROUNDING_CONTRACT_VERSION, "batch_sha256": batch.batch_sha256,
               "complete": True, "classifications": [
                   {"index": 0, "original": assessment("supported"), "alternatives": None},
                   {"index": 1, "original": assessment("ambiguous"), "alternatives": None}]}
    assert [v.status for v in parse(payload, batch).verdicts] == ["supported", "uncertain"]
    payload["classifications"][1]["index"] = 0
    rejects(payload, batch, "classification:index")
