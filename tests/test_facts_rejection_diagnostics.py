"""Bounded, attributable fact failures without logging source or model text."""

from contextlib import closing
from dataclasses import replace
import hashlib
import json
import logging

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import facts
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import StubLLMClient


_ITEM = {"text": "Nora shipped Cedar.", "date": None, "entities": ["Nora"]}


@pytest.mark.parametrize("raw,reason", [
    ("null", "invalid_envelope"),
    (" NULL ", "invalid_json"),
    ("Null", "invalid_json"),
    ("```json\nnull\n```", "invalid_envelope"),
    ("```JSON\nnull\n```", "invalid_envelope"),
    ("```jsonnull```", "invalid_envelope"),
    ("```null```", "invalid_envelope"),
    ("```json\nNULL\n```", "invalid_json"),
    ("```json\nNull\n```", "invalid_json"),
    ("```jsonNULL```", "invalid_json"),
    ("```json\nnull", "invalid_json"),
    ("```python\nnull\n```", "invalid_json"),
    ("Before ```json\nnull\n```", "invalid_json"),
    ("```json\nnull\n``` after", "invalid_json"),
    ("", "invalid_json"),
    ('{"facts":[', "invalid_json"),
    ('{"facts":[],"facts":[]}', "invalid_json"),
    ('{"facts":NaN}', "invalid_json"),
    ('{"facts":[],"type":"json_object"}', "invalid_envelope"),
    ('{"facts":{}}', "invalid_envelope"),
    ('{"items":[]}', "invalid_envelope"),
    ('"plain text"', "invalid_envelope"),
    ('false', "invalid_envelope"),
    ('4', "invalid_envelope"),
    ('[null]', "invalid_items"),
    ('[{"text":""}]', "invalid_items"),
    ('[{"text":"Nora shipped Cedar.","date":"2024-02-30"}]', "invalid_items"),
])
def test_closed_reasons_distinguish_parse_envelope_and_items(raw, reason):
    assert facts._validate_fact_response(raw, max_items=8) == (None, reason)
    assert facts.validate_fact_items(raw, max_items=8) is None


def test_capacity_diagnostics_do_not_validate_unbounded_rejected_lists(monkeypatch):
    original = facts.validate_fact_items
    calls = []

    def recorder(raw, *, max_items):
        calls.append(max_items)
        return original(raw, max_items=max_items)

    monkeypatch.setattr(facts, "validate_fact_items", recorder)
    payload = {"facts": [_ITEM] * (facts.FACT_MAX_ACTIVE_ITEMS_PER_OUTCOME + 1)}
    assert facts._validate_fact_response(payload, max_items=8) == (None, "invalid_items")
    assert calls == [8]


@pytest.mark.parametrize("path", ["fresh", "replay"])
def test_persisted_warning_identifies_session_and_slice_without_raw_values(
    cfg, monkeypatch, tmp_path, path,
):
    config = replace(cfg, aggregation_nodes_enabled=False, profile_extraction_enabled=False)
    session_id = "private-session-key\nforged-log-entry"
    source = "Private source. Nora shipped Cedar."
    valid = StubLLMClient(default=json.dumps({"facts": [_ITEM]}))
    with closing(HyMem(config, llm=valid)) as hy:
        hy.log_message(session_id, "user", source)
        with db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, session_id)
        prior = None
        if path == "replay":
            with monkeypatch.context() as old:
                old.setattr(facts, "FACTS_PROMPT_VERSION", "facts.v2")
                prior = facts.extract_facts(hy.conn, session_id, valid, config)
                with db.transaction(hy.conn):
                    facts.persist_facts(hy.conn, session_id, prior)
        state_before = tuple(hy.conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,),
        ).fetchone())
        raw = '{"facts": [], "private-response-value": "secret response body"}'
        bad = StubLLMClient(default=raw)
        log_path = tmp_path / "benchmark.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        facts.log.addHandler(handler)
        try:
            result = (
                facts.extract_facts(hy.conn, session_id, bad, config)
                if prior is None else facts.reextract_fact_outcome(
                    hy.conn, prior.slice_key, bad, config,
                )
            )
        finally:
            facts.log.removeHandler(handler)
            handler.close()
        assert result.parse_failed and result.failure_reason == "invalid_envelope"
        assert result.items == []
        assert tuple(hy.conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,),
        ).fetchone()) == state_before
        encoded = json.dumps(
            {"version": "fact-diagnostic-session-v1", "session_id": session_id},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        session_key = "sha256:" + hashlib.sha256(encoded).hexdigest()
        assert facts.fact_diagnostic_session_key(session_id) == session_key
        assert facts.fact_diagnostic_session_key(session_id + "other") != session_key
        # A closed handler's retained file is the same normal-log path used by
        # a benchmark supervisor; attribution survives the extraction call.
        lines = log_path.read_text().splitlines()
        assert lines == [
            "facts.output_rejected reason=invalid_envelope "
            f"mode={path} session_key={session_key} slice_key={result.slice_key}"
        ]
        assert all(secret not in lines[0] for secret in (
            session_id, "private-session-key", "forged-log-entry", source, raw,
            "private-response-value", "secret response body",
        ))


@pytest.mark.parametrize("name", [
    "build_facts_request", "_validate_fact_response", "_log_fact_rejection",
    "fact_diagnostic_session_key",
])
def test_loaded_fact_helper_changes_only_fact_generation(monkeypatch, name):
    client = StubLLMClient(default="[]")
    before = {tier: semantic_generation_suffix(tier, client)
              for tier in ("facts", "digest", "profile")}
    phase1_before = client.phase1_producer_declaration()
    original = getattr(facts, name)

    def changed(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(facts, name, changed)
    after = {tier: semantic_generation_suffix(tier, client)
             for tier in before}
    assert after["facts"] != before["facts"]
    assert after["digest"] == before["digest"]
    assert after["profile"] == before["profile"]
    assert client.phase1_producer_declaration() == phase1_before


@pytest.mark.parametrize("name", ["FACTS_SYSTEM", "FACTS_USER_TEMPLATE", "FACTS_CAPACITY_TEMPLATE"])
def test_loaded_fact_prompt_changes_only_fact_generation(monkeypatch, name):
    client = StubLLMClient(default="[]")
    before = {tier: semantic_generation_suffix(tier, client)
              for tier in ("facts", "digest", "profile")}
    monkeypatch.setattr(facts, name, getattr(facts, name) + " Changed.")
    after = {tier: semantic_generation_suffix(tier, client)
             for tier in before}
    assert after["facts"] != before["facts"]
    assert after["digest"] == before["digest"]
    assert after["profile"] == before["profile"]
