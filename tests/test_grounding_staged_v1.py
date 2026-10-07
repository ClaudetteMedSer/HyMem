"""Offline controls for the inactive staged classification contract."""
import copy
import json
from dataclasses import replace

import pytest

from hymem.extraction import grounding_staged_v1 as s
from hymem.extraction import grounding_classification_v4 as v4


def fixture(*, predicates=("uses",), content="Mira uses CairnDB.", contexts=(), **qualifiers):
    source = v4.GroundingSource(7, content, contexts=contexts, source_role="user")
    triples = tuple(v4.Triple("Mira", predicate, "CairnDB", 1, source_message_id=7, **qualifiers)
                    for predicate in predicates)
    return s.build_original_request(triples, (source,))


def assessment(state="not_established", *, quote="Mira uses CairnDB", region="owned", qualifiers=(), entries=None):
    if state != "supported":
        return {"state": state, "support": None}
    evidence = entries if entries is not None else [{"source_message_id": 7, "region": region, "quote": quote}]
    checks = {key: {"state": "supported", "evidence_indices": list(range(len(evidence)))}
              for key in ("attribution_and_roles", "relation_and_polarity", *qualifiers)}
    return {"state": state, "support": {"evidence": evidence, "checks": checks}}


def originals(batch, *states):
    return {"schema": s.ORIGINAL_SCHEMA, "batch_sha256": batch.batch_sha256, "complete": True,
            "originals": [{"index": i, "original": state} for i, state in enumerate(states)]}


def alternatives(batch, prior_hash, indices, *, positive=None, ambiguous=None):
    rows = []
    for i in indices:
        values = {name: assessment() for name in v4.PREDICATE_ORDER if name != batch.triples[i].predicate}
        if positive is not None:
            values[positive] = assessment("supported")
        if ambiguous is not None:
            values[ambiguous] = assessment("ambiguous")
        rows.append({"index": i, "alternatives": values})
    return {"schema": s.ALTERNATIVES_SCHEMA, "batch_sha256": batch.batch_sha256,
            "original_response_sha256": prior_hash, "complete": True, "alternatives": rows}


def enc(obj):
    return json.dumps(obj, ensure_ascii=False)


def test_original_only_request_and_schema():
    request, batch = fixture(predicates=("uses", "prefers"), value_numeric=4)
    assert json.loads(request.user) == json.loads(v4.build_grounding_request(batch.triples, batch.sources)[0].user)
    assert (request.max_tokens, request.temperature, request.response_format) == (4096, 0.0, "json")
    s.validate_original_request(request, batch)
    for changed in (replace(request, system=request.system + " "), replace(request, max_tokens=4096.0),
                    replace(request, user=request.user + " ")):
        with pytest.raises(v4.GroundingContractError, match="^request:binding$"):
            s.validate_original_request(changed, batch)
    schema = s.build_original_output_schema(batch)
    assert "alternatives" not in schema["properties"]["originals"]["items"]["anyOf"][0]["properties"]
    assert schema["$defs"]["assessment_0"]["properties"]["support"]["anyOf"][1]["properties"]["evidence"]["items"]["properties"]["region"]["enum"] == ["owned"]
    assert schema["$defs"]["assessment_0"]["properties"]["support"]["anyOf"][1]["properties"]["checks"]["required"] == sorted(("attribution_and_roles", "relation_and_polarity", "value_numeric"))
    schema["$defs"]["assessment_0"]["properties"]["state"]["enum"] = []
    assert s.build_original_output_schema(batch)["$defs"]["assessment_0"]["properties"]["state"]["enum"]


def test_all_supported_and_ambiguous_require_no_alternatives():
    _, batch = fixture(predicates=("uses", "prefers"))
    raw = enc(originals(batch, assessment("supported"), assessment("ambiguous")))
    result = s.parse_staged_responses(batch, raw, None)
    assert [v.status for v in result.verdicts] == ["supported", "uncertain"]
    with pytest.raises(v4.GroundingContractError, match="^alternatives:none_required$"):
        s.build_alternatives_request(batch, raw)
    with pytest.raises(v4.GroundingContractError, match="^alternatives:unexpected$"):
        s.parse_staged_responses(batch, raw, "{}")


def test_mixed_noncontiguous_negatives_and_exact_binding():
    request, batch = fixture(predicates=("uses", "prefers", "uses", "prefers"))
    original = originals(batch, assessment("supported"), assessment(), assessment("ambiguous"), assessment())
    raw = enc(original)
    alt_request, alt_batch = s.build_alternatives_request(batch, raw)
    assert alt_batch.negative_indices == (1, 3)
    user = json.loads(alt_request.user)
    assert user["batch"] == json.loads(request.user)["batch"]
    assert user["negative_indices"] == [1, 3]
    assert user["original_response_sha256"] == alt_batch.original_response_sha256
    s.validate_alternatives_request(alt_request, alt_batch)
    for changed in (replace(alt_request, user=alt_request.user + " "),
                    replace(alt_request, temperature=False), replace(alt_request, system="x")):
        with pytest.raises(v4.GroundingContractError, match="^request:binding$"):
            s.validate_alternatives_request(changed, alt_batch)
    schema = s.build_alternatives_output_schema(alt_batch)
    assert set(schema["$defs"]) == {"assessment_1", "alternatives_1", "assessment_3", "alternatives_3"}
    assert schema["properties"]["original_response_sha256"]["enum"] == [alt_batch.original_response_sha256]
    assert schema["properties"]["alternatives"]["items"]["anyOf"][0]["properties"]["alternatives"] == {"$ref": "#/$defs/alternatives_1"}
    rows = alternatives(batch, alt_batch.original_response_sha256, (1, 3), positive="owns")
    result = s.parse_staged_responses(batch, raw, enc(rows))
    assert [v.status for v in result.verdicts] == ["supported", "replace_predicate", "uncertain", "replace_predicate"]
    schema["$defs"]["assessment_1"]["properties"]["state"]["enum"] = []
    assert s.build_alternatives_output_schema(alt_batch)["$defs"]["assessment_1"]["properties"]["state"]["enum"]


def test_schemas_validate_exact_stage_payloads():
    jsonschema = pytest.importorskip("jsonschema")
    _, batch = fixture()
    original = originals(batch, assessment())
    original_schema = s.build_original_output_schema(batch)
    jsonschema.Draft202012Validator.check_schema(original_schema)
    jsonschema.validate(original, original_schema)
    _, alt_batch = s.build_alternatives_request(batch, enc(original))
    schema = s.build_alternatives_output_schema(alt_batch)
    jsonschema.Draft202012Validator.check_schema(schema)
    row = alternatives(batch, alt_batch.original_response_sha256, (0,))
    jsonschema.validate(row, schema)
    row["alternatives"][0]["alternatives"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(row, schema)

    allowed = {"$schema", "$defs", "$ref", "type", "additionalProperties", "required",
               "properties", "enum", "anyOf", "minItems", "maxItems", "items",
               "minLength", "maxLength", "minimum", "maximum"}
    def walk(node):
        for key, value in node.items():
            assert key in allowed
            if key in ("properties", "$defs"):
                for child in value.values():
                    walk(child)
            elif key == "items":
                walk(value)
            elif key == "anyOf":
                for child in value:
                    walk(child)
    walk(schema)
    walk(original_schema)


def test_negative_branches_and_ambiguity():
    _, batch = fixture()
    raw = enc(originals(batch, assessment()))
    _, alt_batch = s.build_alternatives_request(batch, raw)
    assert s.parse_staged_responses(batch, raw, enc(alternatives(batch, alt_batch.original_response_sha256, (0,)))).verdicts[0].status == "unsupported"
    rows = alternatives(batch, alt_batch.original_response_sha256, (0,), positive="owns")
    assert s.parse_staged_responses(batch, raw, enc(rows)).verdicts[0].predicate == "owns"
    with pytest.raises(v4.GroundingContractError, match="^verdict:correction$"):
        s.parse_staged_responses(batch, raw, enc(rows), allow_corrections=False)
    rows["alternatives"][0]["alternatives"]["prefers"] = assessment("supported")
    assert s.parse_staged_responses(batch, raw, enc(rows)).verdicts[0].status == "uncertain"
    rows["alternatives"][0]["alternatives"]["prefers"] = assessment("ambiguous")
    assert s.parse_staged_responses(batch, raw, enc(rows)).verdicts[0].status == "uncertain"


def test_prior_response_and_batch_forgery_rejected():
    _, batch = fixture()
    original = originals(batch, assessment())
    raw = enc(original)
    _, alt_batch = s.build_alternatives_request(batch, raw)
    rows = alternatives(batch, alt_batch.original_response_sha256, (0,))
    changed = originals(batch, assessment("ambiguous"))
    with pytest.raises(v4.GroundingContractError, match="^alternatives:unexpected$"):
        s.parse_staged_responses(batch, enc(changed), enc(rows))
    changed = originals(batch, assessment())
    changed["originals"][0]["original"]["support"] = {}
    with pytest.raises(v4.GroundingContractError):
        s.parse_staged_responses(batch, enc(changed), enc(rows))
    rows["original_response_sha256"] = "0" * 64
    with pytest.raises(v4.GroundingContractError, match="^alternatives:prior_binding$"):
        s.parse_staged_responses(batch, raw, enc(rows))
    with pytest.raises(v4.GroundingContractError, match="^alternatives_batch:binding$"):
        s.validate_alternatives_request(s.build_alternatives_request(batch, raw)[0],
                                        replace(alt_batch, original_response_sha256="0" * 64))
    with pytest.raises(v4.GroundingContractError, match="^alternatives_batch:binding$"):
        s.build_alternatives_output_schema(replace(alt_batch, negative_indices=(False,)))
    _, other = fixture(content="Mira uses OtherDB.")
    with pytest.raises(v4.GroundingContractError):
        s.parse_staged_responses(other, raw, None)


def test_changed_valid_prior_and_raw_bounds_rejected():
    _, batch = fixture(predicates=("uses", "prefers"))
    raw = enc(originals(batch, assessment(), assessment()))
    _, alt_batch = s.build_alternatives_request(batch, raw)
    rows = alternatives(batch, alt_batch.original_response_sha256, (0, 1))
    changed = enc(originals(batch, assessment("ambiguous"), assessment()))
    with pytest.raises(v4.GroundingContractError, match="^alternatives:prior_binding$"):
        s.parse_staged_responses(batch, changed, enc(rows))
    with pytest.raises(v4.GroundingContractError, match="^response:bounds$"):
        s.parse_staged_responses(batch, raw + " " * s.MAX_RESPONSE_CHARS, None)
    with pytest.raises(v4.GroundingContractError, match="^response:bounds$"):
        s.parse_staged_responses(batch, raw, enc(rows) + " " * s.MAX_RESPONSE_CHARS)


@pytest.mark.parametrize("mutate", [
    lambda x: x["alternatives"].append(copy.deepcopy(x["alternatives"][0])),
    lambda x: x["alternatives"][0].update(index=True),
    lambda x: x["alternatives"][0]["alternatives"].pop("owns"),
    lambda x: x["alternatives"][0]["alternatives"].update(uses=assessment()),
    lambda x: x["alternatives"][0].update(original=assessment()),
    lambda x: x.update(complete=1),
])
def test_alternative_shape_rejections(mutate):
    _, batch = fixture()
    raw = enc(originals(batch, assessment()))
    _, alt_batch = s.build_alternatives_request(batch, raw)
    rows = alternatives(batch, alt_batch.original_response_sha256, (0,))
    mutate(rows)
    with pytest.raises(v4.GroundingContractError):
        s.parse_staged_responses(batch, raw, enc(rows))


def test_scope_parent_and_qualifier_guards_on_original_and_alternative():
    parent = v4.GroundingContext("conversation_0", "Mira uses CairnDB. Later unrelated text.", 18,
                                 source_role="user", source_message_id=7)
    header = v4.GroundingContext("conversation_0_header", "Mira is speaking.", 18,
                                 source_role="user", source_message_id=7,
                                 applies_to_region="conversation_0", applies_to_prefix_chars=18)
    _, batch = fixture(content="Mira uses CairnDB. Later unrelated text.", contexts=(parent, header), value_text="fast")
    entries = [{"source_message_id": 7, "region": "owned", "quote": "Mira uses CairnDB"},
               {"source_message_id": 7, "region": "conversation_0_header", "quote": "Mira is speaking."}]
    supported = assessment("supported", entries=entries, qualifiers=("value_text",))
    with pytest.raises(v4.GroundingContractError, match="^evidence:parent_required$"):
        s.parse_staged_responses(batch, enc(originals(batch, supported)), None)
    entries.insert(1, {"source_message_id": 7, "region": "conversation_0", "quote": "Mira uses CairnDB"})
    for check in supported["support"]["checks"].values():
        check["evidence_indices"] = [0, 1, 2]
    assert s.parse_staged_responses(batch, enc(originals(batch, supported)), None).all_supported
    supported["support"]["checks"].pop("value_text")
    with pytest.raises(v4.GroundingContractError, match="^support:checks$"):
        s.parse_staged_responses(batch, enc(originals(batch, supported)), None)
    raw = enc(originals(batch, assessment()))
    _, alt_batch = s.build_alternatives_request(batch, raw)
    rows = alternatives(batch, alt_batch.original_response_sha256, (0,))
    rows["alternatives"][0]["alternatives"]["owns"] = assessment("supported", entries=entries, qualifiers=("value_text",))
    assert s.parse_staged_responses(batch, raw, enc(rows)).verdicts[0].status == "replace_predicate"
    rows["alternatives"][0]["alternatives"]["owns"]["support"]["evidence"][0]["quote"] = "Later unrelated text."
    with pytest.raises(v4.GroundingContractError, match="^evidence:context_scope$"):
        s.parse_staged_responses(batch, raw, enc(rows))
