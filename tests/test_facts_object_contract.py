"""The real facts producers must agree with their JSON-object transport.

These tests use synthetic source turns and an SDK recorder, never a network or
paid model. They check the envelope contract, not model extraction faithfulness.
"""

from contextlib import closing
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from benchmarks.fact_probe import FACTS_PROMPT_V2, FACTS_USER_TEMPLATE_V2
from hymem import HyMem
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.core import db
from hymem.dreaming import facts
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import StubLLMClient


_TEXT = "Nora chose Cedar on 2024-05-06."
_ITEM = {"text": _TEXT, "date": "2024-05-06", "entities": ["Nora", "Cedar"]}
_CANONICAL_ITEM = dict(_ITEM, entities=["nora", "cedar"])
_OBJECT_INSTRUCTION = (
    'Output a strict JSON object with exactly one key, "facts", whose value is '
    'an array: {"facts": [...]}. Do not add other top-level keys, prose, markdown, '
    'or code fences. Each item in "facts" has exactly:'
)


def _config(cfg):
    return replace(cfg, aggregation_nodes_enabled=False,
                   profile_extraction_enabled=False)


def _seed(hy, text=_TEXT):
    message_id = hy.log_message(
        "facts-object", "user", text, created_at="2025-07-08T09:10:11Z",
    )
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "facts-object")
    return message_id


def _state(hy):
    """Publication, cursor and exact source bytes stay untouched by extraction."""
    tables = (
        "sessions", "messages", "chunks", "chunk_message_sources",
        "fact_extraction_outcomes", "fact_extraction_revisions",
        "fact_extraction_source_occurrences", "narrative_facts",
        "narrative_fact_lifecycle",
    )
    return tuple(tuple(tuple(row) for row in hy.conn.execute(
        f"SELECT * FROM {table} ORDER BY rowid"
    )) for table in tables)


def _seed_old_outcome(hy, monkeypatch):
    with monkeypatch.context() as old:
        old.setattr(facts, "FACTS_PROMPT_VERSION", "facts.v2")
        old.setattr(facts, "FACTS_SYSTEM", FACTS_PROMPT_V2)
        old.setattr(facts, "FACTS_USER_TEMPLATE", FACTS_USER_TEMPLATE_V2)
        extraction = facts.extract_facts(
            hy.conn, "facts-object", StubLLMClient(default=json.dumps([_ITEM])),
            hy.config,
        )
        assert extraction is not None and not extraction.parse_failed
        with db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "facts-object", extraction)
        return extraction


def _extract_path(hy, llm, path, prior):
    if path == "fresh":
        return facts.extract_facts(hy.conn, "facts-object", llm, hy.config)
    return facts.reextract_fact_outcome(hy.conn, prior.slice_key, llm, hy.config)


@pytest.fixture
def wire_client(monkeypatch):
    """Exercise OpenAICompatibleClient.complete without constructing a network SDK."""
    import openai

    for name in (
        "HYMEM_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
        "HYMEM_LLM_BASE_URL", "HYMEM_LLM_MODEL", "HYMEM_LLM_THINKING",
        "HYMEM_LLM_DEPLOYMENT_REVISION", "HYMEM_LLM_DEPLOYMENT_TENANT",
    ):
        monkeypatch.delenv(name, raising=False)

    clients = []

    def build(raw):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=raw),
                )])

        def sdk(**kwargs):
            http_client = kwargs.get("http_client")
            return SimpleNamespace(
                _client=http_client,
                chat=SimpleNamespace(completions=Completions()),
                close=lambda: http_client.close() if http_client else None,
            )

        monkeypatch.setattr(openai, "OpenAI", sdk)
        client = OpenAICompatibleClient(
            api_key="synthetic-test-key", base_url="https://api.deepseek.com",
            model="deepseek-flash", thinking="disabled",
        )
        clients.append(client)
        return client, calls

    yield build
    for client in clients:
        client.close()


@pytest.mark.parametrize("path", ["fresh", "historical"])
@pytest.mark.parametrize("items", [[_ITEM], []], ids=["nonempty", "empty"])
def test_real_fact_paths_match_json_object_transport(
    cfg, monkeypatch, wire_client, path, items,
):
    with closing(HyMem(_config(cfg), llm=StubLLMClient(default="[]"))) as hy:
        message_id = _seed(hy)
        prior = _seed_old_outcome(hy, monkeypatch) if path == "historical" else None
        llm, calls = wire_client(json.dumps({"facts": items}))
        before = _state(hy)
        result = _extract_path(hy, llm, path, prior)
        assert result is not None and not result.parse_failed
        assert result.items == ([_CANONICAL_ITEM] if items else [])
        assert _state(hy) == before
        assert result.covered_message_id == message_id
        assert [source.message_id for source in result.source_occurrences] == [message_id]
        assert len(calls) == 1
        call = calls[0]
        assert call["response_format"] == {"type": "json_object"}
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == hy.config.dream_digest_max_tokens
        assert call["messages"] == [
            {"role": "system", "content": facts.build_facts_request(
                "user: " + _TEXT, hy.config,
            ).system},
            {"role": "user", "content": facts.FACTS_USER_TEMPLATE.format(
                text="user: " + _TEXT,
            )},
        ]
        assert _OBJECT_INSTRUCTION in call["messages"][0]["content"]
        assert '{"facts": []}' in call["messages"][0]["content"]
        assert call["messages"][1]["content"].endswith(
            "Return the JSON object of narrative facts now."
        )
        assert "2025-07-08" not in str(call["messages"])
        if prior:
            assert result.slice_key == prior.slice_key
            assert result.input_hash == prior.input_hash
            assert result.source_occurrences == prior.source_occurrences
            assert result.publication_version != prior.publication_version


def test_envelope_change_preserves_every_v2_factual_instruction():
    """Byte-equivalence after undoing envelope edits is not a new G-F1 verdict."""
    assert _OBJECT_INSTRUCTION in facts.FACTS_SYSTEM
    restored = facts.FACTS_SYSTEM.replace(
        _OBJECT_INSTRUCTION, "Output a strict JSON array. Each item has exactly:",
    ).replace('{"facts": []}', "[]")
    assert restored == FACTS_PROMPT_V2
    assert facts.FACTS_USER_TEMPLATE.replace(
        "JSON object of narrative facts", "JSON array of narrative facts",
    ) == FACTS_USER_TEMPLATE_V2


@pytest.mark.parametrize("path", ["fresh", "historical"])
@pytest.mark.parametrize("raw", [
    json.dumps({"facts": [_ITEM], "type": "json_object"}),
    json.dumps({"facts": [_ITEM, dict(_ITEM, date="2024-02-30")]}),
    '{"facts": [{"text": "Nora chose Cedar',
], ids=["extra-wrapper-key", "mixed-invalid-item", "incomplete"])
def test_invalid_object_never_drops_content_or_advances_authority(
    cfg, monkeypatch, path, raw,
):
    with closing(HyMem(_config(cfg), llm=StubLLMClient(default="[]"))) as hy:
        _seed(hy)
        prior = _seed_old_outcome(hy, monkeypatch) if path == "historical" else None
        before = _state(hy)
        result = _extract_path(hy, StubLLMClient(default=raw), path, prior)
        assert result is not None and result.parse_failed and result.items == []
        assert _state(hy) == before
        if prior:
            assert result.slice_key == prior.slice_key
            assert result.input_hash == prior.input_hash
            assert result.source_occurrences == prior.source_occurrences


@pytest.mark.parametrize("payload", [
    [_ITEM], {"facts": [_ITEM]}, [], {"facts": []},
])
@pytest.mark.parametrize("fenced", [False, True])
def test_exact_objects_and_legacy_arrays_remain_compatible(payload, fenced):
    raw = json.dumps(payload)
    if fenced:
        raw = "```json\n" + raw + "\n```"
    assert facts.validate_fact_items(raw, max_items=8) == (
        [_CANONICAL_ITEM] if payload in ([_ITEM], {"facts": [_ITEM]}) else []
    )


@pytest.mark.parametrize("payload", [
    {"facts": [_ITEM], "type": "json_object"},
    {"facts": [], "notes": []},
    {"items": [_ITEM]}, {"facts": {}},
    {"facts": [_ITEM, dict(_ITEM, text=" ")]},
    {"facts": [_ITEM, dict(_ITEM, text="x" * 601)]},
    {"facts": [_ITEM, dict(_ITEM, date="2024-02-30")]},
    {"facts": [_ITEM, dict(_ITEM, extra="metadata")]},
    {"facts": [_ITEM, dict(_ITEM, entities="Nora")]},
    {"facts": [_ITEM] * 9},
])
def test_object_validator_stays_atomic_and_strict(payload):
    assert facts.validate_fact_items(json.dumps(payload), max_items=8) is None


def test_current_prompt_replays_exact_old_partial_units_and_resets_only_old_retry_generation(
    cfg, monkeypatch,
):
    assert facts.FACTS_PROMPT_VERSION == "facts.v4"
    config = _config(cfg)
    llm = StubLLMClient(default=json.dumps({"facts": [_ITEM]}))
    with closing(HyMem(config, llm=llm)) as hy:
        message_id = _seed(hy, _TEXT + " " + "More source words. " * 20)
        with monkeypatch.context() as old:
            old.setattr(facts, "FACTS_PROMPT_VERSION", "facts.v2")
            old.setattr(facts, "FACTS_SYSTEM", FACTS_PROMPT_V2)
            old.setattr(facts, "FACTS_USER_TEMPLATE", FACTS_USER_TEMPLATE_V2)
            old_semantic = semantic_generation_suffix("facts", llm)
            old_version = facts.facts_config_version(config, client=llm)
            old_retry = facts.facts_retry_policy_version(config, client=llm)
            first = facts.extract_facts(
                hy.conn, "facts-object", llm, config, max_chars=60,
            )
            assert first is not None and first.partial_message_id == message_id
            assert first.next_message_offset > 0
            with db.transaction(hy.conn):
                facts.persist_facts(hy.conn, "facts-object", first)
                for _ in range(config.facts_extraction_max_attempts):
                    facts.record_fact_failure(
                        hy.conn, "facts-object",
                        max_attempts=config.facts_extraction_max_attempts,
                        retry_config_version=old_retry,
                    )

        new_version = facts.facts_config_version(config, client=llm)
        new_retry = facts.facts_retry_policy_version(config, client=llm)
        assert new_version != old_version and new_retry != old_retry
        assert semantic_generation_suffix("facts", llm) != old_semantic
        assert facts.next_fact_outcome_for_replay(
            hy.conn, "facts-object", new_version,
        ) == first.slice_key
        assert facts.fact_quarantine_status(hy.conn, config, client=llm) == {
            "quarantined_facts": 0, "quarantined_facts_malformed": 0,
        }
        before = _state(hy)
        replay = facts.reextract_fact_outcome(hy.conn, first.slice_key, llm, config)
        assert not replay.parse_failed and _state(hy) == before
        for name in (
            "slice_key", "input_hash", "source_occurrences", "partial_message_id",
            "next_message_offset", "covered_message_id", "cursor_before_offset",
        ):
            assert getattr(replay, name) == getattr(first, name)
        assert replay.publication_version == new_version
        # Framing changes, source bytes do not; no full-turn historical re-cut.
        assert llm.calls[-1].user.split('"""')[1] == llm.calls[0].user.split('"""')[1]
        with db.transaction(hy.conn):
            assert not facts.record_fact_failure(
                hy.conn, "facts-object", max_attempts=config.facts_extraction_max_attempts,
                retry_config_version=new_retry,
            )
        row = hy.conn.execute(
            "SELECT facts_retry_count,facts_retry_config_version,facts_quarantined "
            "FROM sessions WHERE id='facts-object'",
        ).fetchone()
        assert tuple(row) == (1, new_retry, 0)
