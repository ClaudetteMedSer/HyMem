from __future__ import annotations

import json

from hymem.extraction.llm import StubLLMClient
from hymem.extraction.markers import extract_markers
from hymem.extraction.prompts import TRIPLE_SYSTEM, build_triple_system
from hymem.extraction.triples import extract_triples


def test_triples_locked_vocabulary_filters_invalid_predicates():
    llm = StubLLMClient(default=json.dumps([
        {"subject": "service", "predicate": "uses", "object": "postgres", "polarity": 1},
        {"subject": "service", "predicate": "consumes", "object": "kafka", "polarity": 1},  # not allowed
        {"subject": "service", "predicate": "uses", "object": "redis", "polarity": 99},      # bad polarity
    ]))
    triples, _ = extract_triples(llm, "irrelevant text")
    assert len(triples) == 1
    assert triples[0].object == "postgres"


def test_triples_handles_negation_via_polarity():
    llm = StubLLMClient(default=json.dumps([
        {"subject": "local_dev", "predicate": "uses", "object": "docker", "polarity": -1},
        {"subject": "local_dev", "predicate": "avoids", "object": "docker", "polarity": 1},
    ]))
    triples, _ = extract_triples(llm, "we don't use docker anymore, we avoid it")
    assert len(triples) == 2
    polarities = {(t.predicate, t.polarity) for t in triples}
    assert ("uses", -1) in polarities
    assert ("avoids", 1) in polarities


def test_triples_handles_garbage_output():
    llm = StubLLMClient(default="not json")
    assert extract_triples(llm, "x") == ([], {})

    llm = StubLLMClient(default=json.dumps({"not": "an array"}))
    assert extract_triples(llm, "x") == ([], {})


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
