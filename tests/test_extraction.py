from __future__ import annotations

import json

from hymem.extraction.chunk import extract_chunk
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.markers import _parse as parse_markers
from hymem.extraction.markers import extract_markers
from hymem.extraction.prompts import TRIPLE_SYSTEM, build_triple_system
from hymem.extraction.triples import (
    _parse as parse_triples,
    extract_entity_properties,
    extract_entity_types,
    extract_triples,
)


def test_triples_locked_vocabulary_filters_invalid_predicates():
    llm = StubLLMClient(default=json.dumps([
        {"subject": "service", "predicate": "uses", "object": "postgres", "polarity": 1},
        {"subject": "service", "predicate": "consumes", "object": "kafka", "polarity": 1},  # not allowed
        {"subject": "service", "predicate": "uses", "object": "redis", "polarity": 99},      # bad polarity
    ]))
    triples, _, _ = extract_triples(llm, "irrelevant text")
    assert len(triples) == 1
    assert triples[0].object == "postgres"


def test_triples_handles_negation_via_polarity():
    llm = StubLLMClient(default=json.dumps([
        {"subject": "local_dev", "predicate": "uses", "object": "docker", "polarity": -1},
        {"subject": "local_dev", "predicate": "avoids", "object": "docker", "polarity": 1},
    ]))
    triples, _, _ = extract_triples(llm, "we don't use docker anymore, we avoid it")
    assert len(triples) == 2
    polarities = {(t.predicate, t.polarity) for t in triples}
    assert ("uses", -1) in polarities
    assert ("avoids", 1) in polarities


def test_triples_handles_garbage_output():
    llm = StubLLMClient(default="not json")
    assert extract_triples(llm, "x") == ([], {}, {})

    llm = StubLLMClient(default=json.dumps({"not": "an array"}))
    assert extract_triples(llm, "x") == ([], {}, {})


def test_triple_prompt_includes_identity_artifact_linking_nudge():
    """The prompt must explicitly authorise people/teams/projects as entities
    and demonstrate the linking-edge pattern, so identity-to-artifact
    relationships ('Atta works on MedFlow') become real graph edges instead of
    fuzzy text matches across sibling canonicals. Guard against future
    refactors silently stripping the nudge."""
    prompt = TRIPLE_SYSTEM
    # Entities expanded beyond pure tech.
    assert "people" in prompt and "teams" in prompt and "projects" in prompt
    # Explicit linking-example pattern (subject person/team -> part_of/contains -> artifact).
    assert "part_of" in prompt and "contains" in prompt
    # Worked example anchors the LLM on the intended structure.
    assert "(atta, part_of, medflow)" in prompt
    # Entity-type vocabulary covers the new categories.
    assert "person" in prompt and "team" in prompt and "codebase" in prompt


def test_triple_prompt_preserves_nudge_with_negative_examples():
    """Negative-example injection (feedback-driven extraction) must not
    overwrite or hide the identity-artifact rule."""
    prompt = build_triple_system(
        negative_examples="- (foo, uses, bar) [retracted]\n"
    )
    assert "(atta, part_of, medflow)" in prompt
    assert "DO NOT extract" in prompt


def test_markers_filters_unknown_kinds():
    llm = StubLLMClient(default=json.dumps([
        {"kind": "preference", "statement": "user prefers uv"},
        {"kind": "frustration", "statement": "user seemed annoyed"},  # unknown kind
        {"kind": "rejection", "statement": ""},                          # empty stmt
    ]))
    markers = extract_markers(llm, "x")
    assert len(markers) == 1
    assert markers[0].kind == "preference"


# --- combined per-chunk extractor (triples + markers in one call) -----------


def test_extract_chunk_single_call_returns_both():
    """The merged extractor issues exactly ONE LLM call and returns both the
    triples and the markers from a single object response."""
    payload = {
        "triples": [
            {"subject": "service", "predicate": "uses", "object": "postgres",
             "polarity": 1, "subject_type": "service",
             "object_properties": {"language": "sql"}},
        ],
        "markers": [
            {"kind": "preference", "statement": "user prefers uv"},
        ],
    }
    llm = StubLLMClient(default=json.dumps(payload))
    result = extract_chunk(llm, "irrelevant text")

    assert len(llm.calls) == 1
    assert len(result.triples) == 1
    assert result.triples[0].object == "postgres"
    assert len(result.markers) == 1
    assert result.markers[0].kind == "preference"
    assert result.entity_type_hints["service"] == "service"
    assert result.entity_property_hints["postgres"]["language"] == "sql"


def test_extract_chunk_tolerates_malformed_output():
    """Bare array, missing keys, and invalid JSON each yield empty results
    without raising — same tolerance as the digest path."""
    # Bare array (a stub's default) — not an object.
    llm = StubLLMClient(default="[]")
    r = extract_chunk(llm, "x")
    assert r.triples == [] and r.markers == []

    # Object missing both keys.
    llm = StubLLMClient(default=json.dumps({"other": 1}))
    r = extract_chunk(llm, "x")
    assert r.triples == [] and r.markers == []
    assert r.entity_type_hints == {} and r.entity_property_hints == {}

    # Invalid JSON.
    llm = StubLLMClient(default="not json")
    r = extract_chunk(llm, "x")
    assert r.triples == [] and r.markers == []


def test_extract_chunk_matches_separate_parsers():
    """Triples/markers parsed from the combined object match what the old
    separate parsers produce for the equivalent sub-arrays."""
    triples_payload = [
        {"subject": "uv", "predicate": "uses", "object": "python", "polarity": 1,
         "subject_type": "package_manager",
         "subject_properties": {"language": "python"}},
        {"subject": "local_dev", "predicate": "avoids", "object": "docker",
         "polarity": 1},
    ]
    markers_payload = [
        {"kind": "preference", "statement": "user prefers uv"},
        {"kind": "rejection", "statement": "user refuses docker"},
    ]
    combined = json.dumps({"triples": triples_payload, "markers": markers_payload})
    triples_raw = json.dumps(triples_payload)
    markers_raw = json.dumps(markers_payload)

    result = extract_chunk(StubLLMClient(default=combined), "x")

    assert result.triples == parse_triples(triples_raw)
    assert result.markers == parse_markers(markers_raw)
    assert result.entity_type_hints == extract_entity_types(triples_raw)
    assert result.entity_property_hints == extract_entity_properties(triples_raw)
