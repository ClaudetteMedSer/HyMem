"""Independent byte/input binding controls for the local-only replay receipt."""

from dataclasses import replace
import re

import pytest

from hymem.config import HyMemConfig
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk
from hymem.dreaming.lossless import CoveredMessage
from hymem.extraction.triples import Triple


@pytest.fixture
def proof_input(tmp_path):
    chunk = Chunk("proof-chunk", "proof-session", 41, 41, "test", "synthetic text",
                  source_message_ids=(41,))
    source = CoveredMessage(
        41, "proof-session", "user", "Synthetic source bytes.\n", "coverage-chunk",
        source_created_at="2026-01-01T12:00:00Z", source_peer_id="peer-a",
        source_workspace_id="workspace-a",
    )
    triple = Triple("alpha", "uses", "beta", 1, source_message_id=41,
                    value_text="ten", value_numeric=10.0, value_unit="seconds",
                    temporal_scope="today")
    extraction = phase1.ChunkExtraction(
        triples=[triple], markers=[], claim_sources={41: source}, source_validated=True,
    )
    options = dict(prompt_version="derived-test-key-v1",
                   phase1_generation_key="test-generation",
                   result_hash="sha256:" + "a" * 64,
                   cfg=HyMemConfig(root=tmp_path))
    return chunk, extraction, options


def _proof(case):
    chunk, extraction, options = case
    return phase1._local_replay_proof(chunk, extraction, **options)


def test_proof_is_deterministic_and_contains_no_source_text(proof_input):
    first = _proof(proof_input)
    assert _proof(proof_input) == first
    assert re.fullmatch(r"(?:[A-Za-z0-9_-]+:)*[0-9a-f]{64}", first)
    assert "Synthetic" not in first and "alpha" not in first


@pytest.mark.parametrize("field,value", [
    ("subject", "alpha_variant"), ("object", "beta_variant"),
    ("predicate", "prefers"), ("polarity", -1),
    ("value_text", "eleven"), ("value_numeric", 11.0),
    ("value_unit", "minutes"), ("temporal_scope", "yesterday"),
])
def test_every_claim_field_changes_receipt(proof_input, field, value):
    chunk, extraction, options = proof_input
    changed = replace(extraction, triples=[replace(extraction.triples[0], **{field: value})])
    assert _proof((chunk, changed, options)) != _proof(proof_input)


@pytest.mark.parametrize("field,value", [
    ("session_id", "other-session"), ("role", "assistant"),
    ("content", "Synthetic source bytes.\n "), ("chunk_id", "other-coverage"),
    ("source_created_at", "2026-01-02T12:00:00Z"),
    ("source_peer_id", "peer-b"), ("source_workspace_id", "workspace-b"),
])
def test_full_source_binding_changes_receipt(proof_input, field, value):
    chunk, extraction, options = proof_input
    changed_source = replace(extraction.claim_sources[41], **{field: value})
    changed = replace(extraction, claim_sources={41: changed_source})
    assert _proof((chunk, changed, options)) != _proof(proof_input)


@pytest.mark.parametrize("field,value", [
    ("prompt_version", "other-derived-key-v1"),
    ("phase1_generation_key", "other-generation"),
    ("result_hash", "sha256:" + "b" * 64),
])
def test_publication_binding_changes_receipt(proof_input, field, value):
    chunk, extraction, options = proof_input
    assert _proof((chunk, extraction, {**options, field: value})) != _proof(proof_input)


def test_order_cardinality_and_weights_are_not_discarded(proof_input):
    chunk, extraction, options = proof_input
    first = extraction.triples[0]
    second = replace(first, object="gamma")
    ordered = replace(extraction, triples=[first, second])
    reversed_ = replace(extraction, triples=[second, first])
    duplicated = replace(extraction, triples=[first, second, second])
    values = {_proof((chunk, candidate, options)) for candidate in
              (extraction, ordered, reversed_, duplicated)}
    assert len(values) == 4
    cfg = replace(options["cfg"], evidence_role_weights={"user": 3})
    assert _proof((chunk, extraction, {**options, "cfg": cfg})) != _proof(proof_input)


def test_cited_message_identity_changes_receipt(proof_input):
    chunk, extraction, options = proof_input
    changed = replace(
        extraction,
        triples=[replace(extraction.triples[0], source_message_id=42)],
        claim_sources={42: replace(extraction.claim_sources[41], message_id=42)},
    )
    assert _proof((chunk, changed, options)) != _proof(proof_input)


@pytest.mark.parametrize("field,value", [
    ("id", "other-chunk"), ("session_id", "other-session"),
    ("text", "synthetic text "), ("source_message_ids", (41, 42)),
])
def test_chunk_binding_changes_receipt(proof_input, field, value):
    chunk, extraction, options = proof_input
    assert _proof((replace(chunk, **{field: value}), extraction, options)) != _proof(proof_input)
