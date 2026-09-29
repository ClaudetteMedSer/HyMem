"""Offline mechanical controls for the inactive classification contract."""
import copy
import hashlib
import json
from dataclasses import replace

import pytest

from hymem.extraction import grounding_classification_v1 as c


def sample(*, predicate="uses", content="Mira uses CairnDB.", contexts=()):
    source = c.GroundingSource(7, content, contexts=contexts, source_role="user",
                               source_peer_id="mira", source_created_at="2026-09-29")
    triple = c.Triple("Mira", predicate, "CairnDB", 1, source_message_id=7)
    return c.build_grounding_request((triple,), (source,))


def reply(batch, *, states=None, pool=None, citations=None):
    states = states if states is not None else ["n"] * 22
    pool = pool if pool is not None else []
    citations = citations if citations is not None else [[] for _ in range(22)]
    return {"schema": c.GROUNDING_CONTRACT_VERSION, "batch_sha256": batch.batch_sha256,
            "complete": True, "classifications": [
                {"index": 0, "states": states, "evidence_pool": pool, "citations": citations}]}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def entailed(batch, predicate="uses", *, evidence=None):
    states = ["n"] * 22
    pos = c.PREDICATE_ORDER.index(predicate)
    states[pos] = "e"
    pool = evidence or [{"source_message_id": 7, "region": "owned", "quote": "Mira uses CairnDB"}]
    citations = [[] for _ in range(22)]
    citations[pos] = list(range(len(pool)))
    return reply(batch, states=states, pool=pool, citations=citations)


def test_predicate_order_and_original_is_hidden_only_from_wire():
    request, batch = sample()
    assert len(c.PREDICATE_ORDER) == 22
    assert c.PREDICATE_ORDER == tuple(sorted(c.ALLOWED_PREDICATES))
    canonical = json.loads(batch.canonical_json)
    wire = json.loads(request.user)
    assert canonical["candidates"][0]["predicate"] == "uses"
    assert "predicate" not in wire["batch"]["candidates"][0]
    assert wire["batch"]["predicates"] == list(c.PREDICATE_ORDER)
    assert wire["batch"]["sources"] == canonical["sources"]
    assert set(wire["batch"]["candidates"][0]) == set(canonical["candidates"][0]) - {"predicate"}
    assert batch.batch_sha256 == hashlib.sha256(batch.canonical_json.encode()).hexdigest()
    assert request.max_tokens == 4096 and request.response_format == "json" and request.temperature == 0.0
    c.validate_request(request, batch)


def test_supported_and_deterministic_correction():
    _, batch = sample()
    result = c.parse_grounding_response(encoded(entailed(batch)), batch)
    assert result.all_supported and result.batch_sha256 == batch.batch_sha256
    assert result.verdicts[0].predicate == "uses"
    _, wrong = sample(predicate="prefers")
    corrected = c.parse_grounding_response(encoded(entailed(wrong)), wrong)
    assert corrected.verdicts[0].status == "replace_predicate"
    assert corrected.verdicts[0].predicate == "uses"
    with pytest.raises(c.GroundingContractError, match="^verdict:correction$"):
        c.parse_grounding_response(encoded(entailed(wrong)), wrong, allow_corrections=False)


def test_uncertainty_and_ambiguity_prevent_correction():
    _, batch = sample(predicate="prefers")
    original = c.PREDICATE_ORDER.index("prefers")
    positive = c.PREDICATE_ORDER.index("uses")
    payload = entailed(batch)
    payload["classifications"][0]["states"][original] = "u"
    assert c.parse_grounding_response(encoded(payload), batch).verdicts[0].status == "uncertain"
    payload["classifications"][0]["states"][original] = "n"
    other = c.PREDICATE_ORDER.index("owns")
    payload["classifications"][0]["states"][other] = "u"
    assert c.parse_grounding_response(encoded(payload), batch).verdicts[0].status == "uncertain"
    payload["classifications"][0]["states"][other] = "e"
    payload["classifications"][0]["citations"][other] = [0]
    assert c.parse_grounding_response(encoded(payload), batch).verdicts[0].status == "uncertain"
    payload["classifications"][0]["states"][positive] = "n"
    payload["classifications"][0]["citations"][positive] = []
    assert c.parse_grounding_response(encoded(payload), batch).verdicts[0].predicate == "owns"


def test_supported_original_wins_even_with_other_positive():
    _, batch = sample()
    payload = entailed(batch)
    other = c.PREDICATE_ORDER.index("owns")
    payload["classifications"][0]["states"][other] = "e"
    payload["classifications"][0]["citations"][other] = [0]
    assert c.parse_grounding_response(encoded(payload), batch).verdicts[0].status == "supported"


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(schema="source-grounding-v2"),
    lambda x: x.update(batch_sha256="0" * 64),
    lambda x: x.update(complete=1),
    lambda x: x["classifications"][0].update(index=True),
    lambda x: x["classifications"][0].update(states=["n"] * 21),
    lambda x: x["classifications"][0]["states"].__setitem__(0, "maybe"),
    lambda x: x["classifications"][0].update(extra=1),
])
def test_response_shape_version_and_completeness(mutation):
    _, batch = sample()
    payload = reply(batch)
    mutation(payload)
    with pytest.raises(c.GroundingContractError):
        c.parse_grounding_response(encoded(payload), batch)


@pytest.mark.parametrize("mutation", [
    lambda x: x["classifications"][0]["citations"][c.PREDICATE_ORDER.index("uses")].__setitem__(0, True),
    lambda x: x["classifications"][0]["citations"][c.PREDICATE_ORDER.index("uses")].append(0),
    lambda x: x["classifications"][0]["citations"][c.PREDICATE_ORDER.index("uses")].__setitem__(0, 8),
    lambda x: x["classifications"][0]["evidence_pool"].append(x["classifications"][0]["evidence_pool"][0].copy()),
    lambda x: x["classifications"][0]["evidence_pool"][0].update(source_message_id=True),
    lambda x: x["classifications"][0]["evidence_pool"][0].update(quote="Mira CairnDB"),
])
def test_references_and_evidence_reject_invalid(mutation):
    _, batch = sample()
    payload = entailed(batch)
    mutation(payload)
    with pytest.raises(c.GroundingContractError):
        c.parse_grounding_response(encoded(payload), batch)


def test_pool_reuse_and_unused_rejection():
    _, batch = sample()
    payload = entailed(batch)
    other = c.PREDICATE_ORDER.index("owns")
    payload["classifications"][0]["states"][other] = "e"
    payload["classifications"][0]["citations"][other] = [0]
    c.parse_grounding_response(encoded(payload), batch)
    payload["classifications"][0]["states"][other] = "n"
    payload["classifications"][0]["citations"][other] = []
    payload["classifications"][0]["evidence_pool"].append(
        {"source_message_id": 7, "region": "owned", "quote": "CairnDB."})
    with pytest.raises(c.GroundingContractError, match="^evidence:unused$"):
        c.parse_grounding_response(encoded(payload), batch)


def test_negative_citations_and_missing_positive_citations_rejected():
    _, batch = sample()
    payload = entailed(batch)
    item = payload["classifications"][0]
    item["citations"][c.PREDICATE_ORDER.index("owns")] = [0]
    with pytest.raises(c.GroundingContractError, match="^citation:negative$"):
        c.parse_grounding_response(encoded(payload), batch)
    item["citations"][c.PREDICATE_ORDER.index("owns")] = []
    item["citations"][c.PREDICATE_ORDER.index("uses")] = []
    with pytest.raises(c.GroundingContractError, match="^citation:positive_required$"):
        c.parse_grounding_response(encoded(payload), batch)


def test_multi_claim_order_is_exact_and_no_entailment_is_unsupported():
    source = c.GroundingSource(7, "Mira uses CairnDB.", source_role="user")
    claims = (c.Triple("Mira", "uses", "CairnDB", 1, source_message_id=7),
              c.Triple("Mira", "prefers", "CairnDB", 1, source_message_id=7))
    _, batch = c.build_grounding_request(claims, (source,))
    payload = {"schema": c.GROUNDING_CONTRACT_VERSION, "batch_sha256": batch.batch_sha256,
               "complete": True, "classifications": [
                   {"index": i, "states": ["n"] * 22, "evidence_pool": [],
                    "citations": [[] for _ in range(22)]} for i in range(2)]}
    assert [v.status for v in c.parse_grounding_response(encoded(payload), batch).verdicts] == ["unsupported", "unsupported"]
    payload["classifications"][0]["index"] = 1
    with pytest.raises(c.GroundingContractError, match="^classification:index$"):
        c.parse_grounding_response(encoded(payload), batch)
    schema = c.build_output_schema(batch)
    assert schema["properties"]["classifications"]["minItems"] == 2
    assert [item["properties"]["index"]["enum"] for item in schema["properties"]["classifications"]["items"]["anyOf"]] == [[0], [1]]


def test_v2_context_scope_and_nested_parent_scope():
    parent = c.GroundingContext("conversation_0", "Mira uses CairnDB. Later unrelated text.", 18,
                                source_role="user", source_peer_id="mira", source_created_at="2026-09-29",
                                source_message_id=7)
    header = c.GroundingContext("conversation_0_header", "Mira is speaking.", 18,
                                source_role="user", source_peer_id="mira", source_created_at="2026-09-29",
                                source_message_id=7, applies_to_region="conversation_0", applies_to_prefix_chars=18)
    _, batch = sample(content="Mira uses CairnDB. Later unrelated text.", contexts=(parent, header))
    evidence = [
        {"source_message_id": 7, "region": "owned", "quote": "Mira uses CairnDB"},
        {"source_message_id": 7, "region": "conversation_0", "quote": "Mira uses CairnDB"},
        {"source_message_id": 7, "region": "conversation_0_header", "quote": "Mira is speaking."},
    ]
    c.parse_grounding_response(encoded(entailed(batch, evidence=evidence)), batch)
    outside = copy.deepcopy(evidence)
    outside[0]["quote"] = "Later unrelated text."
    with pytest.raises(c.GroundingContractError, match="^evidence:context_scope$"):
        c.parse_grounding_response(encoded(entailed(batch, evidence=outside)), batch)
    absent_parent = copy.deepcopy(evidence)
    del absent_parent[1]
    with pytest.raises(c.GroundingContractError, match="^evidence:parent_required$"):
        c.parse_grounding_response(encoded(entailed(batch, evidence=absent_parent)), batch)
    outside_parent = copy.deepcopy(evidence)
    outside_parent[1]["quote"] = "Later unrelated text."
    with pytest.raises(c.GroundingContractError, match="^evidence:parent_scope$"):
        c.parse_grounding_response(encoded(entailed(batch, evidence=outside_parent)), batch)


def test_schema_is_fresh_exact_and_request_is_bound():
    request, batch = sample()
    schema = c.build_output_schema(batch)
    assert schema["properties"]["batch_sha256"]["enum"] == [batch.batch_sha256]
    assert schema["properties"]["complete"]["enum"] == [True]
    assert len(schema["properties"]["classifications"]["items"]["anyOf"]) == 1
    schema["properties"]["batch_sha256"]["enum"][0] = "mutated"
    assert c.build_output_schema(batch)["properties"]["batch_sha256"]["enum"] == [batch.batch_sha256]
    with pytest.raises(c.GroundingContractError, match="^request:binding$"):
        c.validate_request(replace(request, max_tokens=8192), batch)
    with pytest.raises(c.GroundingContractError, match="^request:binding$"):
        c.validate_request(replace(request, max_tokens=4096.0), batch)
    with pytest.raises(c.GroundingContractError, match="^request:binding$"):
        c.validate_request(replace(request, temperature=False), batch)
    with pytest.raises(c.GroundingContractError, match="^request:binding$"):
        c.validate_request(replace(request, user=request.user + " "), batch)
    with pytest.raises(c.GroundingContractError, match="^batch:binding$"):
        c.build_output_schema(replace(batch, canonical_json=batch.canonical_json + " "))
    other_request, other_batch = sample(predicate="prefers")
    assert other_batch.batch_sha256 != batch.batch_sha256
    with pytest.raises(c.GroundingContractError, match="^request:binding$"):
        c.validate_request(other_request, batch)


def test_v2_limits_and_representation_size_measured_without_token_claim():
    with pytest.raises(c.GroundingContractError, match="^triple:text$"):
        c.build_grounding_request(
            (c.Triple("x" * 2049, "uses", "y", 1, source_message_id=7),),
            (c.GroundingSource(7, "x"),))
    _, batch = sample()
    oversized = encoded(reply(batch)) + " " * c.MAX_RESPONSE_CHARS
    with pytest.raises(c.GroundingContractError, match="^response:bounds$"):
        c.parse_grounding_response(oversized, batch)
    representative = len(encoded(entailed(batch)))
    worst_wire = reply(batch)
    evidence = [{"source_message_id": 7, "region": "owned", "quote": "q" * 192} for _ in range(8)]
    # Distinct pool entries at the bound; size is reported as serialization,
    # not treated as a claim about tokenizer fit or semantic validity.
    for i, item in enumerate(evidence):
        item["quote"] = str(i) + item["quote"][1:]
    worst_wire["classifications"][0]["evidence_pool"] = evidence
    worst_wire["classifications"][0]["states"] = ["e"] * 22
    worst_wire["classifications"][0]["citations"] = [list(range(8)) for _ in range(22)]
    worst_case = len(encoded(worst_wire))
    assert representative < worst_case < c.MAX_RESPONSE_CHARS
