"""Phase-1 extracts entity_properties from LLM output and persists them,
and augment() uses them (and entity_types) to expand entities matched by
free-text category queries.
"""
from __future__ import annotations

import json

from hymem.extraction.triples import (
    extract_entity_properties,
    extract_entity_types,
    extract_triples,
)
from hymem.extraction.llm import StubLLMClient
from tests.conftest import make_routed_llm


def test_extract_entity_properties_from_subject_object_blocks():
    raw = json.dumps([
        {
            "subject": "uv",
            "predicate": "uses",
            "object": "python",
            "polarity": 1,
            "subject_properties": {
                "language": "python",
                "category": "build_tool",
            },
            "object_properties": {"runtime": "cpython"},
        }
    ])
    props = extract_entity_properties(raw)
    assert props["uv"] == {"language": "python", "category": "build_tool"}
    assert props["python"] == {"runtime": "cpython"}


def test_extract_entity_properties_top_level_mapping():
    raw = json.dumps([
        {"subject": "a", "predicate": "uses", "object": "b", "polarity": 1,
         "properties": {"a": {"role": "client"}, "b": {"role": "server"}}}
    ])
    props = extract_entity_properties(raw)
    assert props["a"] == {"role": "client"}
    assert props["b"] == {"role": "server"}


def test_extract_entity_properties_drops_garbage():
    # Non-string keys, non-scalar values, oversize fields are skipped.
    raw = json.dumps([
        {
            "subject": "x",
            "predicate": "uses",
            "object": "y",
            "polarity": 1,
            "subject_properties": {
                "ok": "fine",
                "": "empty_key",
                "huge": "z" * 200,
                "nested": {"not": "scalar"},
            },
        }
    ])
    props = extract_entity_properties(raw)
    assert props["x"] == {"ok": "fine"}


def test_extract_triples_returns_three_tuple():
    raw = json.dumps([
        {"subject": "uv", "predicate": "uses", "object": "python", "polarity": 1,
         "subject_type": "package_manager",
         "subject_properties": {"language": "python"}}
    ])
    llm = StubLLMClient(default=raw)
    triples, types, props = extract_triples(llm, "x")
    assert len(triples) == 1
    assert types["uv"] == "package_manager"
    assert props["uv"]["language"] == "python"


def test_entity_properties_persisted_and_replaced_on_re_extraction(hy):
    sid = "s_props"
    hy.open_session(sid)
    hy.log_message(sid, "user",
        "We adopted uv for Python builds, replacing pip across the board.")
    hy.close_session(sid)

    triples = [
        {
            "subject": "build_pipeline",
            "predicate": "uses",
            "object": "uv",
            "polarity": 1,
            "object_type": "package_manager",
            "object_properties": {
                "language": "python",
                "category": "build_tool",
            },
        }
    ]
    hy.set_llm(make_routed_llm(triples, []))
    hy.dream()

    rows = {
        r["key"]: r["value"]
        for r in hy.conn.execute(
            "SELECT key, value FROM entity_properties WHERE entity_canonical='uv'"
        ).fetchall()
    }
    assert rows == {"language": "python", "category": "build_tool"}

    # Re-extracting with a changed value (e.g. corrected category) updates
    # in place — ON CONFLICT(entity_canonical, key) DO UPDATE.
    revised = [
        {
            "subject": "build_pipeline",
            "predicate": "uses",
            "object": "uv",
            "polarity": 1,
            "object_type": "package_manager",
            "object_properties": {"category": "dependency_manager"},
        }
    ]
    # Force re-extraction by bumping prompt_version.
    from dataclasses import replace as dc_replace
    new_cfg = dc_replace(hy.config, prompt_version="v14")
    hy.config = new_cfg
    hy.set_llm(make_routed_llm(revised, []))
    hy.dream()

    rows = {
        r["key"]: r["value"]
        for r in hy.conn.execute(
            "SELECT key, value FROM entity_properties WHERE entity_canonical='uv'"
        ).fetchall()
    }
    assert rows["category"] == "dependency_manager"
    # The previously-extracted language key remains — last write wins per (entity, key).
    assert rows["language"] == "python"


def test_augment_expands_entities_from_free_text_type_query(hy):
    """A query that names a *category* ('build tools') and no specific entity
    should still surface every canonical tagged with the matching type."""
    sid = "s_types"
    hy.open_session(sid)
    hy.log_message(sid, "user", "Adopted uv across the build pipeline.")
    hy.close_session(sid)

    triples = [
        {"subject": "build_pipeline", "predicate": "uses",
         "object": "uv", "polarity": 1,
         "object_type": "package_manager"},
        {"subject": "build_pipeline", "predicate": "uses",
         "object": "pip", "polarity": -1,
         "object_type": "package_manager"},
        {"subject": "build_pipeline", "predicate": "uses",
         "object": "poetry", "polarity": -1,
         "object_type": "package_manager"},
    ]
    hy.set_llm(make_routed_llm(triples, []))
    hy.dream()

    ctx = hy.augment("what build tools do we use here?")
    # Even though the query never names any of these entities, the type
    # keyword "build tools" must surface all package_manager canonicals.
    matched = set(ctx.matched_entities)
    assert {"uv", "pip", "poetry"}.issubset(matched)
    # And graph facts should reflect at least one of those expanded entities.
    objects = {f.object for f in ctx.graph_facts}
    assert objects & {"uv", "pip", "poetry"}


def test_augment_expands_entities_from_property_query(hy):
    """A query naming a property phrase ('build tools') surfaces canonicals
    tagged via entity_properties even when entity_types is empty."""
    sid = "s_props_q"
    hy.open_session(sid)
    hy.log_message(sid, "user", "We rely on make for legacy builds.")
    hy.close_session(sid)

    triples = [
        {
            "subject": "legacy",
            "predicate": "uses",
            "object": "make",
            "polarity": 1,
            "object_properties": {"category": "build_tool"},
        }
    ]
    hy.set_llm(make_routed_llm(triples, []))
    hy.dream()

    ctx = hy.augment("which build tools are still around?")
    assert "make" in ctx.matched_entities
