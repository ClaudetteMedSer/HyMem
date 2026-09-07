"""Unicode identity must survive every normalization and authority boundary."""

from __future__ import annotations

import dataclasses
import json
import types
import unicodedata

import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import canonicalize, facts
from hymem.extraction import contract
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.producer import phase1_generation_binding
from hymem.query.entities import match_known_entities
from hymem.query.augment import _fts_safe_text, _query_mentions_canonical
from tests.conftest import make_routed_llm


@pytest.mark.parametrize(("surface", "expected"), [
    ("東京ß", "東京ss"), ("Москваß", "москваss"),
    ("東京2", "東京2"), ("大阪2", "大阪2"),
    ("नील", "नील"), ("नाल", "नाल"), ("कुमार", "कुमार"),
    ("Й", "й"), ("И\u0306", "й"), ("Καφέ", "καφέ"),
    ("عَلِيّ", "عَلِيّ"), ("علي", "علي"),
    ("Cafe\u0301東京", "cafe東京"), ("Straße", "strasse"),
    ("Ångström", "angstrom"), ("ÀPIParser", "api_parser"),
    ("The 東京2 (city)", "東京2"), ("Ｔｏｋｙｏ２", "tokyo2"),
    ("東京\u200d駅", "東京_駅"), ("東京\x00駅", "東京_駅"),
    ("नील—नाल", "नील_नाल"), ("\u0301東京", "東京"),
    ("😀 — !!!", ""), ("\u0301\u0308", ""),
])
def test_unicode_keys_are_stable_and_script_preserving(surface, expected):
    assert canonicalize.normalize(surface) == expected
    assert canonicalize.normalize(expected) == expected
    assert canonicalize.normalize(unicodedata.normalize("NFD", surface)) == expected


@pytest.mark.parametrize("prefix", ["a", "東", "न"])
def test_all_scripts_fail_closed_without_truncation(prefix):
    assert canonicalize.normalize(prefix * 512) == prefix * 512
    assert canonicalize.normalize(prefix * 512 + "1") == ""
    assert canonicalize.normalize(prefix * 512 + "2") == ""


def test_fact_validation_is_a_fixed_point_without_entity_collisions():
    values = ["東京ß", "Москваß", "東京2", "大阪2", "नील", "नाल", "Й"]
    raw = [{"text": "Named attendees joined the meeting.", "entities": values}]
    first = facts.validate_fact_items(json.dumps(raw), max_items=8)
    assert first is not None
    assert first[0]["entities"] == [canonicalize.normalize(v) for v in values]
    assert len(first[0]["entities"]) == len(values)
    assert facts.validate_fact_items(first, max_items=8) == first


@pytest.mark.parametrize("surface", ["नील", "नाल", "عَلِيّ", "И\u0306", "東京2"])
def test_query_entity_tokens_retain_combining_marks(hy, surface):
    key = canonicalize.normalize(surface)
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
        "pos_evidence) VALUES (?, 'uses', 'tool', 1)", (key,),
    )
    assert match_known_entities(hy.conn, surface) == [key]
    assert match_known_entities(hy.conn, f"What about {surface}?") == [key]


def test_canonicalization_policy_versions_phase1_and_fact_generations(cfg, monkeypatch):
    client = StubLLMClient(default="[]")
    extraction = contract.extraction_cache_key(cfg.prompt_version)
    phase1 = phase1_generation_binding(cfg.prompt_version, client)
    fact_version = facts.facts_config_version(cfg, client=client)
    assert contract._contract_components(cfg.prompt_version)["canonicalization_policy"] == (
        canonicalize.CANONICALIZATION_POLICY_VERSION
    )
    monkeypatch.setattr(canonicalize, "CANONICALIZATION_POLICY_VERSION", "test-old-policy")
    assert contract.extraction_cache_key(cfg.prompt_version) != extraction
    assert phase1_generation_binding(cfg.prompt_version, client) != phase1
    assert facts.facts_config_version(cfg, client=client) != fact_version
    monkeypatch.undo()
    monkeypatch.setattr(canonicalize, "normalize", lambda value: value)
    assert contract.extraction_cache_key(cfg.prompt_version) != extraction
    assert phase1_generation_binding(cfg.prompt_version, client) != phase1
    assert facts.facts_config_version(cfg, client=client) != fact_version


@pytest.mark.parametrize("consumer", ["phase1", "phase1_auxiliary"])
def test_phase1_generation_tracks_actual_canonicalization_module_alias(cfg, monkeypatch, consumer):
    from hymem.dreaming import phase1, phase1_auxiliary

    client = StubLLMClient(default="[]")
    baseline = phase1_generation_binding(cfg.prompt_version, client)
    replacement = types.ModuleType(canonicalize.__name__)
    replacement.__dict__.update(vars(canonicalize))
    replacement.normalize = lambda value: value
    monkeypatch.setattr(
        {"phase1": phase1, "phase1_auxiliary": phase1_auxiliary}[consumer],
        "canonicalize", replacement,
    )
    assert phase1_generation_binding(cfg.prompt_version, client) != baseline


def test_unicode_facts_extract_persist_retrieve_and_port_without_identity_drift(tmp_path):
    values = ["東京ß", "Москваß", "東京2", "大阪2", "नील", "नाल", "Й"]
    expected = [canonicalize.normalize(value) for value in values]
    payload = [{
        "text": ", ".join(values) + " completed Project Atlas.", "entities": values,
    }]
    llm = StubLLMClient(default=json.dumps(payload))
    config = HyMemConfig(root=tmp_path / "source", redact_secrets=False)
    source = HyMem(config, llm=llm)
    wire = tmp_path / "unicode.jsonl"
    try:
        source.log_message("unicode", "user", "Project Atlas attendees: " + ", ".join(values))
        extraction = facts.extract_facts(source.conn, "unicode", llm, source.config)
        assert extraction is not None and not extraction.parse_failed
        assert extraction.items[0]["entities"] == expected
        with core_db.transaction(source.conn):
            assert facts.persist_facts(source.conn, "unicode", extraction) == 1
        row = source.conn.execute("SELECT entities FROM narrative_facts").fetchone()
        assert json.loads(row["entities"]) == expected
        assert facts.fact_session_authority_is_valid(source.conn, "unicode")
        hits = source.augment("Project Atlas").facts
        assert hits and hits[0].entities == expected
        # Full-text search is candidate retrieval, not an exact entity check:
        # SQLite unicode61 itself can omit Indic/Arabic marks. It must still
        # accept the Unicode query instead of deleting it before MATCH.
        for query in ("東京ß", "Москваß", "東京2", "नील", "नाल"):
            assert source.augment(query).facts[0].entities == expected
        source.export(wire)
    finally:
        source.close()
    target = HyMem(dataclasses.replace(config, root=tmp_path / "target"), llm=llm)
    try:
        target.import_(wire)
        assert facts.fact_session_authority_is_valid(target.conn, "unicode")
        row = target.conn.execute("SELECT entities FROM narrative_facts").fetchone()
        assert json.loads(row["entities"]) == expected
        assert target.augment("Project Atlas").facts[0].entities == expected
        assert target.augment("नील").facts[0].entities == expected
        before = "\n".join(target.conn.iterdump())
        assert sum(target.import_(wire).values()) == 0
        assert "\n".join(target.conn.iterdump()) == before
    finally:
        target.close()


def test_phase1_published_unicode_entities_are_distinct_and_exactly_queryable(cfg):
    values = ["東京2", "大阪2", "नील", "नाल", "東京ß", "Москваß"]
    llm = make_routed_llm([
        {"subject": value, "predicate": "uses", "object": "Tool", "polarity": 1,
         "subject_type": "person"}
        for value in values
    ], [])
    hy = HyMem(dataclasses.replace(cfg, aggregation_nodes_enabled=False), llm=llm)
    try:
        hy.log_message("unicode-graph", "user", ", ".join(values) + " all use Tool.")
        hy.close_session("unicode-graph")
        report = hy.dream()
        assert report.triples_extracted == len(values)
        expected = {canonicalize.normalize(value) for value in values}
        actual = {row[0] for row in hy.conn.execute(
            "SELECT subject_canonical FROM knowledge_graph WHERE predicate='uses'"
        )}
        assert actual == expected
        assert {row[0] for row in hy.conn.execute(
            "SELECT entity_canonical FROM current_entity_types WHERE type='person'"
        )} == expected
        for value in values:
            assert match_known_entities(hy.conn, value) == [canonicalize.normalize(value)]
    finally:
        hy.close()


def test_unicode_literal_mentions_do_not_conflate_vowel_marks():
    assert _query_mentions_canonical("नील", "नील")
    assert not _query_mentions_canonical("नील", "नाल")
    assert not _query_mentions_canonical("عَلِيّ", "علي")
    assert _query_mentions_canonical("И\u0306", "й")
    assert _query_mentions_canonical("東京ß", "東京ss")


def test_unicode_fts_sanitization_removes_query_syntax():
    cleaned = _fts_safe_text('नील" OR text:東京* NEAR(大阪, 4) -"Straße"\x00')
    assert "नील" in cleaned and "東京" in cleaned and "Straße" in cleaned
    assert not set('"*:(),\x00') & set(cleaned)
