"""Offline controls for the inactive evidence-group classification contract."""
import copy
import hashlib
import json
from dataclasses import replace

import pytest

from hymem.extraction import grounding_classification_v2 as c


def sample(*, predicate="uses", content="Mira uses CairnDB.", contexts=(), sid=7):
    source = c.GroundingSource(sid, content, contexts=contexts, source_role="user")
    triple = c.Triple("Mira", predicate, "CairnDB", 1, source_message_id=sid)
    return c.build_grounding_request((triple,), (source,))


def evidence(quote="Mira uses CairnDB", *, region="owned", sid=7):
    return {"source_message_id": sid, "region": region, "quote": quote}


def reply(batch, *, states=None, groups=None):
    return {"schema": c.GROUNDING_CONTRACT_VERSION, "batch_sha256": batch.batch_sha256,
            "complete": True, "classifications": [{"index": 0,
                "states": states if states is not None else ["not_established"] * 22,
                "support_groups": groups if groups is not None else []}]}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def positive(batch, *predicates, group_by_predicate=False):
    states = ["not_established"] * 22
    for predicate in predicates:
        states[c.PREDICATE_ORDER.index(predicate)] = "supported"
    groups = ([{"predicates": [predicate], "evidence": [evidence()]} for predicate in predicates]
              if group_by_predicate else [{"predicates": list(predicates), "evidence": [evidence()]}])
    return reply(batch, states=states, groups=groups)


def parse(payload, batch, **kwargs):
    return c.parse_grounding_response(encoded(payload), batch, **kwargs)


def rejects(payload, batch, code=None):
    with pytest.raises(c.GroundingContractError, match=f"^{code}$" if code else None):
        parse(payload, batch)


def test_request_identity_hidden_original_and_schema_isolation():
    request, batch = sample()
    canonical = json.loads(batch.canonical_json)
    wire = json.loads(request.user)
    assert c.GROUNDING_CONTRACT_VERSION == "source-grounding-classification-v2"
    assert c.PREDICATE_ORDER == tuple(sorted(c.ALLOWED_PREDICATES))
    assert len(c.PREDICATE_ORDER) == 22
    assert canonical["candidates"][0]["predicate"] == "uses"
    assert "predicate" not in wire["batch"]["candidates"][0]
    assert wire["batch"]["sources"] == canonical["sources"]
    assert wire["batch"]["predicates"] == list(c.PREDICATE_ORDER)
    assert batch.batch_sha256 == hashlib.sha256(batch.canonical_json.encode()).hexdigest()
    assert (request.max_tokens, request.response_format, request.temperature) == (4096, "json", 0.0)
    assert "not_established" in request.system and "ambiguous" in request.system
    assert "missing facts" in request.system and "factually false" in request.system
    c.validate_request(request, batch)
    schema = c.build_output_schema(batch)
    item = schema["properties"]["classifications"]["items"]["anyOf"][0]
    group = item["properties"]["support_groups"]["items"]
    assert set(group["required"]) == {"predicates", "evidence"}
    assert c.GroundingContext.__name__ == "GroundingContext"
    def keys(value):
        if isinstance(value, dict):
            yield from value
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)
    assert "uniqueItems" not in set(keys(schema))
    assert item["properties"]["states"]["items"]["enum"] == sorted(["supported", "not_established", "ambiguous"])
    schema["properties"]["batch_sha256"]["enum"][0] = "bad"
    group["properties"]["evidence"]["items"]["properties"]["quote"]["maxLength"] = 0
    fresh = c.build_output_schema(batch)
    assert fresh["properties"]["batch_sha256"]["enum"] == [batch.batch_sha256]
    assert fresh["properties"]["classifications"]["items"]["anyOf"][0]["properties"]["support_groups"]["items"]["properties"]["evidence"]["items"]["properties"]["quote"]["maxLength"] == 192
    for changed in (replace(request, max_tokens=4096.0), replace(request, user=request.user + " "),
                    replace(request, temperature=False)):
        with pytest.raises(c.GroundingContractError, match="^request:binding$"):
            c.validate_request(changed, batch)
    with pytest.raises(c.GroundingContractError, match="^batch:binding$"):
        c.build_output_schema(replace(batch, canonical_json=batch.canonical_json + " "))
    other_request, other_batch = sample(predicate="prefers")
    assert other_batch.batch_sha256 != batch.batch_sha256
    with pytest.raises(c.GroundingContractError, match="^request:binding$"):
        c.validate_request(other_request, batch)


def test_selection_and_readable_state_distinctions():
    _, batch = sample()
    result = parse(positive(batch, "uses", "owns"), batch)
    assert result.all_supported and result.verdicts[0].predicate == "uses"
    _, wrong = sample(predicate="prefers")
    result = parse(positive(wrong, "uses"), wrong)
    assert (result.verdicts[0].status, result.verdicts[0].predicate) == ("replace_predicate", "uses")
    with pytest.raises(c.GroundingContractError, match="^verdict:correction$"):
        parse(positive(wrong, "uses"), wrong, allow_corrections=False)
    assert parse(reply(wrong), wrong).verdicts[0].status == "unsupported"
    for predicate in ("prefers", "owns"):
        payload = positive(wrong, "uses")
        payload["classifications"][0]["states"][c.PREDICATE_ORDER.index(predicate)] = "ambiguous"
        assert parse(payload, wrong).verdicts[0].status == "uncertain"
    payload = positive(wrong, "uses", "owns")
    assert parse(payload, wrong).verdicts[0].status == "uncertain"
    payload = reply(wrong)
    payload["classifications"][0]["states"][0] = "ambiguous"
    assert parse(payload, wrong).verdicts[0].status == "uncertain"
    payload["classifications"][0]["states"][0] = "u"
    rejects(payload, wrong, "classification:states")


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(schema="source-grounding-classification-v1"),
    lambda p: p.update(batch_sha256="0" * 64),
    lambda p: p.update(complete=1),
    lambda p: p.update(extra=1),
    lambda p: p["classifications"][0].update(index=True),
    lambda p: p["classifications"][0].update(states=["not_established"] * 21),
    lambda p: p["classifications"][0].update(extra=1),
    lambda p: p["classifications"][0].update(evidence_pool=[]),
])
def test_exact_response_shape(mutate):
    _, batch = sample()
    payload = reply(batch)
    mutate(payload)
    rejects(payload, batch)


@pytest.mark.parametrize("mutate", [
    lambda p: p["classifications"][0]["support_groups"].clear(),
    lambda p: p["classifications"][0]["support_groups"][0].update(predicates=[]),
    lambda p: p["classifications"][0]["support_groups"][0].update(predicates=["uses", "uses"]),
    lambda p: p["classifications"][0]["support_groups"][0].update(predicates=["uses", "prefers"]),
    lambda p: p["classifications"][0]["support_groups"][0].update(predicates=["unknown"]),
    lambda p: p["classifications"][0]["support_groups"][0].update(evidence=[]),
    lambda p: p["classifications"][0]["support_groups"][0].update(extra=1),
    lambda p: p["classifications"][0]["support_groups"].append(copy.deepcopy(p["classifications"][0]["support_groups"][0])),
    lambda p: p["classifications"][0]["support_groups"][0]["evidence"].append(evidence()),
    lambda p: p["classifications"][0]["support_groups"][0]["evidence"][0].update(source_message_id=True),
    lambda p: p["classifications"][0]["support_groups"][0]["evidence"][0].update(quote="Mira CairnDB"),
])
def test_group_partition_and_evidence_rejection(mutate):
    _, batch = sample()
    payload = positive(batch, "uses")
    mutate(payload)
    rejects(payload, batch)


def test_cross_group_overlap_and_distinct_global_bound():
    content = "Mira uses CairnDB abcdefghijklmnopqrstuvwxyz."
    _, batch = sample(content=content)
    payload = positive(batch, "uses", "owns", group_by_predicate=True)
    parse(payload, batch)  # Identical evidence is allowed across groups.
    groups = payload["classifications"][0]["support_groups"]
    groups[1]["evidence"][0] = evidence("uses CairnDB")
    parse(payload, batch)  # Overlapping, distinct excerpts are also allowed.
    groups[0]["evidence"] = [evidence("Mira uses CairnDB")] + [evidence(ch) for ch in "abcdefgh"]
    rejects(payload, batch, "group:evidence_bounds")
    groups[0]["evidence"] = [evidence("Mira uses CairnDB")] + [evidence(ch) for ch in "abcdefg"]
    rejects(payload, batch, "evidence:global_bounds")


def test_scope_parent_and_null_source():
    parent = c.GroundingContext("conversation_0", "Mira uses CairnDB. Later unrelated text.", 18,
                                source_role="user", source_message_id=7)
    header = c.GroundingContext("conversation_0_header", "Mira is speaking.", 18,
                                source_role="user", source_message_id=7,
                                applies_to_region="conversation_0", applies_to_prefix_chars=18)
    _, batch = sample(content="Mira uses CairnDB. Later unrelated text.", contexts=(parent, header))
    payload = positive(batch, "uses")
    ev = payload["classifications"][0]["support_groups"][0]["evidence"]
    ev.extend([evidence("Mira uses CairnDB", region="conversation_0"),
               evidence("Mira is speaking.", region="conversation_0_header")])
    parse(payload, batch)
    bad = copy.deepcopy(payload)
    bad["classifications"][0]["support_groups"][0]["evidence"][0]["quote"] = "Later unrelated text."
    rejects(bad, batch, "evidence:context_scope")
    bad = copy.deepcopy(payload)
    del bad["classifications"][0]["support_groups"][0]["evidence"][1]
    rejects(bad, batch, "evidence:parent_required")
    bad = copy.deepcopy(payload)
    bad["classifications"][0]["support_groups"][0]["evidence"][1]["quote"] = "Later unrelated text."
    rejects(bad, batch, "evidence:parent_scope")
    _, legacy = sample(sid=None)
    payload = positive(legacy, "uses")
    payload["classifications"][0]["support_groups"][0]["evidence"][0]["source_message_id"] = None
    assert parse(payload, legacy).all_supported
    schema = c.build_output_schema(legacy)
    assert schema["properties"]["classifications"]["items"]["anyOf"][0]["properties"]["support_groups"]["items"]["properties"]["evidence"]["items"]["properties"]["source_message_id"]["enum"] == [None]


def test_bounds_and_output_size():
    with pytest.raises(c.GroundingContractError, match="^triple:text$"):
        c.build_grounding_request((c.Triple("x" * 2049, "uses", "y", 1, source_message_id=7),),
                                  (c.GroundingSource(7, "x"),))
    _, batch = sample()
    with pytest.raises(c.GroundingContractError, match="^response:bounds$"):
        c.parse_grounding_response(encoded(reply(batch)) + " " * c.MAX_RESPONSE_CHARS, batch)
    payload = positive(batch, "uses")
    payload["classifications"][0]["support_groups"][0]["evidence"][0]["quote"] = "x" * 193
    rejects(payload, batch, "evidence:quote")
    representative = len(encoded(positive(batch, "uses")))
    adversarial = positive(batch, *c.PREDICATE_ORDER, group_by_predicate=True)
    for group in adversarial["classifications"][0]["support_groups"]:
        group["evidence"] = [evidence(str(i) + "x" * 191) for i in range(8)]
    assert representative < len(encoded(adversarial)) < c.MAX_RESPONSE_CHARS
    assert len(encoded(adversarial)) > 4096  # Character count only; no token-fit assertion.
