from __future__ import annotations

import json

from hymem.extraction.llm import StubLLMClient
from hymem.extraction.markers import extract_markers
from hymem.extraction.triples import extract_triples


def test_triples_locked_vocabulary_filters_invalid_predicates():
    llm = StubLLMClient(default=json.dumps([
        {"subject": "service", "predicate": "uses", "object": "postgres", "polarity": 1},
        {"subject": "service", "predicate": "consumes", "object": "kafka", "polarity": 1},  # not allowed
        {"subject": "service", "predicate": "uses", "object": "redis", "polarity": 99},      # bad polarity
    ]))
    triples = extract_triples(llm, "irrelevant text")
    assert len(triples) == 1
    assert triples[0].object == "postgres"


def test_triples_handles_negation_via_polarity():
    llm = StubLLMClient(default=json.dumps([
        {"subject": "local_dev", "predicate": "uses", "object": "docker", "polarity": -1},
        {"subject": "local_dev", "predicate": "avoids", "object": "docker", "polarity": 1},
    ]))
    triples = extract_triples(llm, "we don't use docker anymore, we avoid it")
    assert len(triples) == 2
    polarities = {(t.predicate, t.polarity) for t in triples}
    assert ("uses", -1) in polarities
    assert ("avoids", 1) in polarities


def test_triples_handles_garbage_output():
    llm = StubLLMClient(default="not json")
    assert extract_triples(llm, "x") == []

    llm = StubLLMClient(default=json.dumps({"not": "an array"}))
    assert extract_triples(llm, "x") == []


def test_markers_filters_unknown_kinds():
    llm = StubLLMClient(default=json.dumps([
        {"kind": "preference", "statement": "user prefers uv"},
        {"kind": "frustration", "statement": "user seemed annoyed"},  # unknown kind
        {"kind": "rejection", "statement": ""},                          # empty stmt
    ]))
    markers = extract_markers(llm, "x")
    assert len(markers) == 1
    assert markers[0].kind == "preference"
