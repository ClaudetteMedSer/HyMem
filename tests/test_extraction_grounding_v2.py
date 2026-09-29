"""Offline checks for the inactive source-grounding v2 decision policy."""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from hymem.extraction import grounding as v1
from hymem.extraction import grounding_v2 as v2
from hymem.extraction.triples import Triple


_ROOT = Path(__file__).resolve().parents[1]
_V1 = _ROOT / "hymem/extraction/grounding.py"
_V2 = _ROOT / "hymem/extraction/grounding_v2.py"


def _batch(module, *, context=False):
    owned = "Mira favors CairnDB over PineDB."
    contexts = (
        module.GroundingContext("prelude", "Mira is the speaker.",
                                len("Mira favors CairnDB")),
    ) if context else ()
    triple = Triple("Mira", "prefers", "CairnDB", 1, source_message_id=7)
    source = module.GroundingSource(7, owned, contexts, source_role="user")
    return module.build_grounding_request((triple,), (source,))


def _reply(module, batch, *, status="supported", predicate="prefers", evidence=None):
    if evidence is None:
        evidence = [{"source_message_id": 7, "region": "owned",
                     "quote": "Mira favors CairnDB over PineDB."}]
    return json.dumps({
        "schema": module.GROUNDING_CONTRACT_VERSION,
        "batch_sha256": batch.batch_sha256,
        "complete": True,
        "verdicts": [{"index": 0, "status": status, "predicate": predicate,
                      "evidence": evidence}],
    })


def _normalized_ast(path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "GROUNDING_CONTRACT_VERSION", "_SYSTEM",
            }:
                node.value = ast.Constant(None)
    return ast.dump(tree, include_attributes=False)


def test_v1_is_immutable_and_v2_has_only_two_assignment_differences():
    assert hashlib.sha256(_V1.read_bytes()).hexdigest() == (
        "dd1a49b56abf569b4476a0b735e67e72a86723bf88e3a4739998aceeabe19c18"
    )
    assert _normalized_ast(_V1) == _normalized_ast(_V2)
    assert v1.GROUNDING_CONTRACT_VERSION == "source-grounding-v1"
    assert v2.GROUNDING_CONTRACT_VERSION == "source-grounding-v2"
    assert v1._SYSTEM != v2._SYSTEM


def test_policy_has_unambiguous_priority_and_preserves_full_claim():
    prompt = v2._SYSTEM
    for phrase in (
        "First assess the exact original full claim",
        "never replace an already supported original",
        "exactly one different predicate",
        "zero or multiple alternatives",
        "subject, object, polarity, every qualifier, and cited owned source unchanged",
        "clearly entailed implicit relationship",
        "Acknowledgment, proposal, intent, or preference alone",
        "one verbatim contiguous excerpt from one named region",
        "Text outside an applicable context's scope is unavailable for both reasoning and citation",
    ):
        assert phrase in prompt
    assert "Predicate meanings: uses=" in prompt
    assert "has_attribute=has personal measurement or attribute" in prompt
    assert not any(label in prompt for label in
                   ("Canary", "Avery", "PostgreSQL", "Fly.io", "longmemeval"))


def test_request_schema_binding_and_old_schema_rejection():
    old_request, old_batch = _batch(v1)
    request, batch = _batch(v2)
    old_wire = json.loads(old_request.user)
    wire = json.loads(request.user)
    assert old_wire["batch"]["version"] == "source-grounding-v1"
    assert wire["batch"]["version"] == "source-grounding-v2"
    assert {key: value for key, value in old_wire["batch"].items() if key != "version"} == {
        key: value for key, value in wire["batch"].items() if key != "version"
    }
    assert batch.batch_sha256 != old_batch.batch_sha256
    assert wire["batch_sha256"] == batch.batch_sha256
    assert request.response_format == old_request.response_format == "json"
    assert request.max_tokens == old_request.max_tokens == 4096
    assert request.temperature == old_request.temperature == 0.0
    assert v2.parse_grounding_response(_reply(v2, batch), batch).all_supported
    with pytest.raises(v2.GroundingContractError, match="^response:schema$"):
        v2.parse_grounding_response(_reply(v1, batch), batch)
    with pytest.raises(v2.GroundingContractError, match="^response:binding$"):
        v2.parse_grounding_response(_reply(v2, old_batch), batch)


def test_strict_evidence_scope_and_correction_semantics_unchanged():
    _, batch = _batch(v2, context=True)
    evidence = [
        {"source_message_id": 7, "region": "owned", "quote": "Mira favors CairnDB"},
        {"source_message_id": 7, "region": "prelude", "quote": "Mira is the speaker."},
    ]
    assert v2.parse_grounding_response(_reply(v2, batch, evidence=evidence), batch).all_supported
    outside = [dict(evidence[0], quote="PineDB."), evidence[1]]
    with pytest.raises(v2.GroundingContractError, match="^evidence:context_scope$"):
        v2.parse_grounding_response(_reply(v2, batch, evidence=outside), batch)
    noncontiguous = [dict(evidence[0], quote="Mira CairnDB"), evidence[1]]
    with pytest.raises(v2.GroundingContractError, match="^evidence:quote_missing$"):
        v2.parse_grounding_response(_reply(v2, batch, evidence=noncontiguous), batch)

    wrong = replace(batch.triples[0], predicate="uses")
    _, wrong_batch = v2.build_grounding_request((wrong,), batch.sources)
    correction = _reply(v2, wrong_batch, status="replace_predicate",
                        predicate="prefers", evidence=evidence)
    review = v2.parse_grounding_response(correction, wrong_batch)
    assert not review.all_supported
    assert review.verdicts[0].predicate == "prefers"
    assert wrong_batch.triples == (wrong,)
    with pytest.raises(v2.GroundingContractError, match="^verdict:correction$"):
        v2.parse_grounding_response(correction, wrong_batch, allow_corrections=False)
    negative = _reply(v2, wrong_batch, status="uncertain", predicate=None, evidence=[])
    assert not v2.parse_grounding_response(negative, wrong_batch).all_supported
